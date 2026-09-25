// Shared shim MM2S channel (aie.mlir: shared_input_channel join + a
// dma_configure_task_for interleave chain), packet-routed through a memtile
// join, core copies each joined object to a second output fifo back to a
// DDR buffer this host reads back. --elf selects the built ELF: aie.elf
// (positive) or aie_negctl.elf (both BDs' outer stride zeroed, so every
// repeat re-reads slice 0 -- see run.lit).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

static constexpr int PAYLOAD_W = 16;   // words/object
static constexpr int SCALE_W = 4;      // words/object
static constexpr int JOINED_W = PAYLOAD_W + SCALE_W; // 20

int main(int argc, char **argv) {
  std::string elf_path = "aie.elf";
  int timeout_ms = 5000;
  int N = 8;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--elf" && i + 1 < argc) elf_path = argv[++i];
    if (a == "--timeout-ms" && i + 1 < argc) timeout_ms = std::atoi(argv[++i]);
    if (a == "--n" && i + 1 < argc) N = std::atoi(argv[++i]);
  }

  auto device = xrt::device(0);
  xrt::elf elf{elf_path};
  xrt::hw_context ctx(device, elf);
  auto kernel = xrt::ext::kernel(ctx, "test:sequence");

  // %w : memref<1024xi32> (payload region at word 0, scale region at word
  // 512 -- matches probe_src.mlir's dma_bd offsets), %o : memref<256xi32>.
  xrt::bo bo_w = xrt::ext::bo{device, 1024 * sizeof(uint32_t)};
  xrt::bo bo_o = xrt::ext::bo{device, 256 * sizeof(uint32_t)};

  uint32_t *w = bo_w.map<uint32_t *>();
  memset(w, 0xAA, 1024 * sizeof(uint32_t));
  // slice k's payload word j = 0x1000*k + j + 1 ; slice k's scale word j = 0x2000*k + j + 1
  // (+1 so slice 0 is never all-zero, keeping a stuck-at-slice-0 negative control visible)
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < PAYLOAD_W; j++)
      w[k * PAYLOAD_W + j] = (uint32_t)(0x1000 * k + j + 1); // never 0, so a stuck-at-0 negative control is visible
    for (int j = 0; j < SCALE_W; j++)
      w[512 + k * SCALE_W + j] = (uint32_t)(0x2000 * k + j + 1);
  }
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  uint32_t *o = bo_o.map<uint32_t *>();
  memset(o, 0xEE, 256 * sizeof(uint32_t));
  bo_o.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_w);
  run.set_arg(1, bo_o);
  run.start();
  auto t0 = std::chrono::steady_clock::now();
  auto deadline = t0 + std::chrono::milliseconds(timeout_ms);
  // run.wait(timeout) throws "unexpected command state" on this XRT build
  // when the shim event fires before the ert packet's state word is
  // written (state < COMPLETED at that instant) -- poll state() with our
  // own deadline instead of trusting that call's internal race.
  ert_cmd_state st = ERT_CMD_STATE_NEW;
  for (;;) {
    st = run.state();
    if (st == ERT_CMD_STATE_COMPLETED || st == ERT_CMD_STATE_ERROR ||
        st == ERT_CMD_STATE_ABORT || st == ERT_CMD_STATE_TIMEOUT ||
        st == ERT_CMD_STATE_NORESPONSE)
      break;
    if (std::chrono::steady_clock::now() >= deadline) {
      st = ERT_CMD_STATE_TIMEOUT;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (st != ERT_CMD_STATE_COMPLETED) {
    std::cout << "DISPATCH_STATE=" << st
               << (st == ERT_CMD_STATE_TIMEOUT ? " (ERT_CMD_STATE_TIMEOUT)" : "")
               << " elapsed_ms=" << ms << "\n";
    std::cout << "RESULT=INCONCLUSIVE (did not complete)\n";
    return 2;
  }

  bo_o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  o = bo_o.map<uint32_t *>();

  int mismatches = 0;
  int slice0_repeats = 0;
  for (int k = 0; k < N; k++) {
    bool ok = true;
    for (int j = 0; j < PAYLOAD_W; j++) {
      uint32_t want = (uint32_t)(0x1000 * k + j + 1);
      uint32_t got = o[k * JOINED_W + j];
      if (got != want) ok = false;
    }
    for (int j = 0; j < SCALE_W; j++) {
      uint32_t want = (uint32_t)(0x2000 * k + j + 1);
      uint32_t got = o[k * JOINED_W + PAYLOAD_W + j];
      if (got != want) ok = false;
    }
    // did object k actually contain slice-0's payload/scale instead?
    bool is_slice0 = true;
    for (int j = 0; j < PAYLOAD_W; j++)
      if (o[k * JOINED_W + j] != (uint32_t)(j + 1)) is_slice0 = false;
    for (int j = 0; j < SCALE_W; j++)
      if (o[k * JOINED_W + PAYLOAD_W + j] != (uint32_t)(j + 1)) is_slice0 = false;
    if (is_slice0 && k != 0) slice0_repeats++;
    if (!ok) mismatches++;
    printf("obj[%d] payload[0..3]=%08x %08x %08x %08x scale[0..3]=%08x %08x %08x %08x %s\n",
           k, o[k*JOINED_W+0], o[k*JOINED_W+1], o[k*JOINED_W+2], o[k*JOINED_W+3],
           o[k*JOINED_W+PAYLOAD_W+0], o[k*JOINED_W+PAYLOAD_W+1],
           o[k*JOINED_W+PAYLOAD_W+2], o[k*JOINED_W+PAYLOAD_W+3],
           ok ? "OK" : "MISMATCH");
  }
  std::cout << "elapsed_ms=" << ms << " mismatches=" << mismatches
            << "/" << N << " slice0_repeats=" << slice0_repeats << "\n";
  std::cout << "RESULT=" << (mismatches == 0 ? "PASS" : "FAIL") << "\n";
  return mismatches == 0 ? 0 : 1;
}
