#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nn {

class Tensor {
public:
  enum class DataType { FLOAT32, INT64, INT32, BOOL, UINT8 };

  Tensor();
  Tensor(std::vector<int64_t> shape, DataType dtype);
  Tensor(std::vector<int64_t> shape, DataType dtype, void* external_data);

  float* data_f32();
  const float* data_f32() const;
  int64_t* data_i64();
  const int64_t* data_i64() const;
  int32_t* data_i32();
  const int32_t* data_i32() const;
  uint8_t* data_u8();
  const uint8_t* data_u8() const;

  const std::vector<int64_t>& shape() const;
  const std::vector<int64_t>& strides() const;
  int64_t ndim() const;
  int64_t numel() const;
  int64_t nbytes() const;
  DataType dtype() const;
  bool is_contiguous() const;
  const void* data_raw() const;

  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor transpose(std::vector<int64_t> perm) const;
  Tensor slice(int axis, int64_t start, int64_t end, int64_t step = 1) const;

  void fill_zeros();
  void fill_from_raw(const float* src, size_t n);
  void fill_from_raw(const int64_t* src, size_t n);
  void fill_from_bytes(const void* src, size_t bytes);
  void copy_to_bytes(void* dst, size_t bytes) const;
  Tensor clone_contiguous() const;

private:
  Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides,
         DataType dtype, std::shared_ptr<void> storage, size_t byte_offset);

  static size_t element_size(DataType dtype);
  static std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);
  static int64_t checked_numel(const std::vector<int64_t>& shape);
  char* raw_data();
  const char* raw_data() const;

  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  DataType dtype_ = DataType::FLOAT32;
  std::shared_ptr<void> storage_;
  size_t byte_offset_ = 0;
};

}  // namespace nn
