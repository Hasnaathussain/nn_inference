#include "nn/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using nn::Tensor;

TEST_CASE("Tensor computes shape, strides, sizes, and aligned storage") {
  Tensor tensor({2, 3, 4}, Tensor::DataType::FLOAT32);
  REQUIRE(tensor.shape() == std::vector<int64_t>{2, 3, 4});
  REQUIRE(tensor.strides() == std::vector<int64_t>{12, 4, 1});
  REQUIRE(tensor.ndim() == 3);
  REQUIRE(tensor.numel() == 24);
  REQUIRE(tensor.nbytes() == 96);
  REQUIRE(reinterpret_cast<uintptr_t>(tensor.data_f32()) % 64 == 0);
}

TEST_CASE("Tensor owns copied handles and supports external views") {
  Tensor owner({3}, Tensor::DataType::FLOAT32);
  const float values[] = {1.0f, 2.0f, 3.0f};
  owner.fill_from_raw(values, 3);
  Tensor alias = owner;
  alias.data_f32()[1] = 7.0f;
  REQUIRE(owner.data_f32()[1] == 7.0f);

  float external[] = {4.0f, 5.0f};
  Tensor view({2}, Tensor::DataType::FLOAT32, external);
  view.data_f32()[0] = 9.0f;
  REQUIRE(external[0] == 9.0f);
}

TEST_CASE("Tensor transpose and slice are strided views") {
  Tensor tensor({2, 3}, Tensor::DataType::FLOAT32);
  const float values[] = {0, 1, 2, 3, 4, 5};
  tensor.fill_from_raw(values, 6);

  Tensor transposed = tensor.transpose({1, 0});
  REQUIRE(transposed.shape() == std::vector<int64_t>{3, 2});
  REQUIRE(transposed.strides() == std::vector<int64_t>{1, 3});
  REQUIRE_FALSE(transposed.is_contiguous());
  REQUIRE(transposed.data_f32()[transposed.strides()[0] + transposed.strides()[1]] == 4.0f);

  Tensor sliced = tensor.slice(1, 0, 3, 2);
  REQUIRE(sliced.shape() == std::vector<int64_t>{2, 2});
  REQUIRE(sliced.strides() == std::vector<int64_t>{3, 2});
  REQUIRE(sliced.data_f32()[sliced.strides()[0] + sliced.strides()[1]] == 5.0f);
}

TEST_CASE("Tensor reshapes contiguous data and copies non-contiguous logical order") {
  Tensor tensor({2, 3}, Tensor::DataType::FLOAT32);
  const float values[] = {0, 1, 2, 3, 4, 5};
  tensor.fill_from_raw(values, 6);

  Tensor flat = tensor.reshape({6});
  REQUIRE(flat.data_f32() == tensor.data_f32());
  REQUIRE_THROWS_AS(tensor.reshape({5}), std::invalid_argument);

  Tensor transposed_flat = tensor.transpose({1, 0}).reshape({6});
  const float expected[] = {0, 3, 1, 4, 2, 5};
  for (int i = 0; i < 6; ++i) REQUIRE(transposed_flat.data_f32()[i] == expected[i]);
}

TEST_CASE("Tensor fill zeros respects non-contiguous views") {
  Tensor tensor({2, 4}, Tensor::DataType::INT64);
  const int64_t values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor.fill_from_raw(values, 8);
  tensor.slice(1, 0, 4, 2).fill_zeros();
  const int64_t expected[] = {0, 2, 0, 4, 0, 6, 0, 8};
  for (int i = 0; i < 8; ++i) REQUIRE(tensor.data_i64()[i] == expected[i]);
}

TEST_CASE("Tensor rejects invalid construction and dtype access") {
  REQUIRE_THROWS_AS(Tensor({2, -1}, Tensor::DataType::FLOAT32), std::invalid_argument);
  Tensor ints({2}, Tensor::DataType::INT64);
  REQUIRE_THROWS(ints.data_f32());
  REQUIRE_THROWS_AS(ints.transpose({0, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(ints.slice(0, 0, 1, 0), std::invalid_argument);
}
