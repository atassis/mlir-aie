// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for DMA transfer-length extension via length_parameter (see aie.mlir).
// For each k, StateTable[@len] = k must move exactly 64*(k+1) words: the
// prefix [0, 64*(k+1)) must match the input, and the rest must be untouched
// sentinel.

#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

int main() {
  constexpr int N = 1024;
  constexpr uint32_t SENT = 0xDEADBEEF;

  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context ctx(device, elf);
  auto kernel = xrt::ext::kernel(ctx, "test:sequence");

  xrt::bo a = xrt::ext::bo{device, N * sizeof(uint32_t)};
  xrt::bo c = xrt::ext::bo{device, N * sizeof(uint32_t)};
  auto *pa = a.map<uint32_t *>();
  auto *pc = c.map<uint32_t *>();
  for (int i = 0; i < N; i++)
    pa[i] = i + 1;
  a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, a);
  run.set_arg(1, c);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  int fails = 0;
  for (uint32_t k : {0u, 5u, 15u, 3u, 0u, 15u}) {
    for (int i = 0; i < N; i++)
      pc[i] = SENT;
    c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    params.write("len", static_cast<int32_t>(k));
    params.sync();

    run.start();
    run.wait2();

    c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int copied = 0;
    while (copied < N && pc[copied] == pa[copied])
      copied++;
    int sent = 0;
    for (int i = copied; i < N; i++)
      sent += pc[i] == SENT;
    int want = 64 * (k + 1);
    bool ok = copied == want && sent == N - copied;
    fails += !ok;
    std::cout << "k=" << k << " want " << want << " copied " << copied
              << " sentinel_tail " << sent << "/" << N - copied << " "
              << (ok ? "OK" : "MISMATCH") << "\n";
  }
  std::cout << (fails ? "FAIL" : "PASS") << "\n";
  return fails ? 1 : 0;
}
