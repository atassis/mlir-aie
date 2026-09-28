//===- alloc_group_overlay_capacity.mlir ------------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// Two existing buffers in DIFFERENT alloc_groups overlay: the MemTile capacity
// check must count the group once, at its largest member, not sum every
// buffer on the tile. Each buffer here is under capacity alone and the two
// together are over it if summed -- this must succeed.

// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// CHECK: aie.buffer({{.*}}) {alloc_group = "p1", sym_name = "a"}
// CHECK: aie.buffer({{.*}}) {alloc_group = "p2", sym_name = "b"}
module @alloc_group_overlay_capacity {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.buffer(%mem) {alloc_group = "p1", sym_name = "a"} : memref<400000xi8>
    %b = aie.buffer(%mem) {alloc_group = "p2", sym_name = "b"} : memref<400000xi8>
  }
}
