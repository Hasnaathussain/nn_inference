#include "nn/ops/gemm.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

using nn::kernel::gemm_scalar;

TEST_CASE("gemm_scalar multiplies row-major matrices") {
  const std::array<float, 6> a{1, 2, 3, 4, 5, 6};
  const std::array<float, 6> b{7, 8, 9, 10, 11, 12};
  std::array<float, 4> c{};
  gemm_scalar(a.data(), b.data(), c.data(), 2, 2, 3);
  REQUIRE(c[0] == Catch::Approx(58.0f));
  REQUIRE(c[1] == Catch::Approx(64.0f));
  REQUIRE(c[2] == Catch::Approx(139.0f));
  REQUIRE(c[3] == Catch::Approx(154.0f));
}

TEST_CASE("gemm_scalar applies alpha, beta, and transpose flags") {
  // Physical A is 3x2 and physical B is 2x3; both are transposed logically.
  const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
  const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
  std::array<float, 4> c{1, 2, 3, 4};
  gemm_scalar(a.data(), b.data(), c.data(), 2, 2, 3, 0.5f, 2.0f, true, true);
  REQUIRE(c[0] == Catch::Approx(31.0f));
  REQUIRE(c[1] == Catch::Approx(36.0f));
  REQUIRE(c[2] == Catch::Approx(75.5f));
  REQUIRE(c[3] == Catch::Approx(85.0f));
}

TEST_CASE("gemm_scalar validates dimensions and pointers") {
  float value = 0.0f;
  REQUIRE_THROWS_AS(gemm_scalar(&value, &value, &value, -1, 1, 1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(gemm_scalar(nullptr, &value, &value, 1, 1, 1),
                    std::invalid_argument);
  REQUIRE_NOTHROW(gemm_scalar(nullptr, nullptr, nullptr, 0, 1, 1));
}

TEST_CASE("AVX2 GEMM dispatcher matches scalar on tiled and remainder shapes") {
  constexpr int M = 100;
  constexpr int N = 300;
  constexpr int K = 200;
  std::vector<float> a(static_cast<size_t>(M) * K);
  std::vector<float> b(static_cast<size_t>(K) * N);
  for (size_t index = 0; index < a.size(); ++index) {
    a[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F;
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b[index] = static_cast<float>(static_cast<int>(index % 37) - 18) / 37.0F;
  }

  std::vector<float> expected(static_cast<size_t>(M) * N);
  std::vector<float> actual(static_cast<size_t>(M) * N);
  gemm_scalar(a.data(), b.data(), expected.data(), M, N, K);
  nn::kernel::gemm(a.data(), b.data(), actual.data(), M, N, K);

  float max_difference = 0.0F;
  for (size_t index = 0; index < actual.size(); ++index) {
    max_difference =
        std::max(max_difference, std::abs(actual[index] - expected[index]));
  }
  REQUIRE(max_difference < 1.0e-4F);

  if (nn::kernel::gemm_avx_available()) {
    std::fill(actual.begin(), actual.end(), 0.0F);
    nn::kernel::gemm_avx(a.data(), b.data(), actual.data(), M, N, K);
    max_difference = 0.0F;
    for (size_t index = 0; index < actual.size(); ++index) {
      max_difference =
          std::max(max_difference, std::abs(actual[index] - expected[index]));
    }
    REQUIRE(max_difference < 1.0e-4F);
  }

  if (nn::kernel::gemm_avx2_available()) {
    std::fill(actual.begin(), actual.end(), 0.0F);
    nn::kernel::gemm_avx2(a.data(), b.data(), actual.data(), M, N, K);
    max_difference = 0.0F;
    for (size_t index = 0; index < actual.size(); ++index) {
      max_difference =
          std::max(max_difference, std::abs(actual[index] - expected[index]));
    }
    REQUIRE(max_difference < 1.0e-4F);
  }
}