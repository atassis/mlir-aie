//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = float; // x
using DATATYPE_IN2 = float; // gb = [gamma(COLS) | beta(COLS)]
using DATATYPE_OUT = test_utils::bfloat16_t;
#endif

static constexpr float kEps = 1e-5f;
static constexpr float kAtol = 0.1f;

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_volume) {
  for (int i = 0; i < in_volume; i++) {
    bufIn1[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f; // [-1, 1)
  }
}

void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int gb_volume) {
  // gb_volume == 2 * COLS: gamma in [0.5, 1.5), beta in [-0.5, 0.5)
  int cols = gb_volume / 2;
  for (int i = 0; i < cols; i++)
    bufIn2[i] = 1.0f + ((float)rand() / (float)RAND_MAX - 0.5f); // gamma
  for (int i = 0; i < cols; i++)
    bufIn2[cols + i] = (float)rand() / (float)RAND_MAX - 0.5f; // beta
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int out_volume) {
  memset(bufOut, 0, out_volume * sizeof(DATATYPE_OUT));
}

// setup_and_run_aie's verify contract: verify(in0*, in1*, ..., out*,
// first_input_volume, verbosity) -> error count. first_input_volume is x's
// element count (ROWS * COLS); gb's layout/size comes from the ROWS/COLS
// macros the same way test_utils::bfloat16_from_float etc. do elsewhere.
int verify_ln_affine_cast_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                                 DATATYPE_OUT *bufOut, int in_volume,
                                 int verbosity) {
  const int rows = ROWS, cols = COLS;
  int errors = 0, pass = 0;
  const DATATYPE_IN2 *gamma = bufIn2;
  const DATATYPE_IN2 *beta = bufIn2 + cols;

  for (int r = 0; r < rows; r++) {
    const DATATYPE_IN1 *row = bufIn1 + (size_t)r * cols;

    double sum = 0.0;
    for (int c = 0; c < cols; c++)
      sum += row[c];
    double mean = sum / cols;

    double var_sum = 0.0;
    for (int c = 0; c < cols; c++) {
      double d = row[c] - mean;
      var_sum += d * d;
    }
    double var = var_sum / cols;
    double inv_std = 1.0 / std::sqrt(var + kEps);

    for (int c = 0; c < cols; c++) {
      double y = (row[c] - mean) * inv_std * gamma[c] + beta[c];
      float expected = (float)y;
      float got = test_utils::bfloat16_to_float(bufOut[(size_t)r * cols + c]);
      if (std::fabs(expected - got) > kAtol) {
        if (errors < 10)
          std::cout << "Mismatch at row " << r << " col " << c << ": expected "
                    << expected << ", got " << got << std::endl;
        errors++;
      } else {
        pass++;
      }
    }
  }

  if (errors == 0)
    std::cout << "ln_affine_cast Passed : " << pass << " out of "
              << (rows * cols) << std::endl;
  else
    std::cout << "ln_affine_cast FAILED with " << errors << " errors out of "
              << (rows * cols) << " elements." << std::endl;
  return errors;
}

int main(int argc, const char *argv[]) {
  args myargs = parse_args(argc, argv);
  int in_volume = ROWS * COLS;
  int gb_volume = 2 * COLS;
  int out_volume = ROWS * COLS;

  return setup_and_run_aie(
      verify_ln_affine_cast_kernel,
      std::make_tuple(make_in<DATATYPE_IN1>(in_volume, initialize_bufIn1),
                      make_in<DATATYPE_IN2>(gb_volume, initialize_bufIn2)),
      make_out<DATATYPE_OUT>(out_volume, initialize_bufOut), myargs);
}
