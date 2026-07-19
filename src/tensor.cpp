#include "nn/tensor.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace nn {
namespace {

void validate_axis(int axis, int64_t ndim) {
  if (axis < 0 || axis >= ndim) {
    throw std::invalid_argument("Tensor axis is out of range");
  }
}

}  // namespace

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), dtype_(dtype) {
  const auto bytes = static_cast<size_t>(nbytes());
  if (bytes == 0) return;
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, bytes) != 0) throw std::bad_alloc();
  storage_ = std::shared_ptr<void>(ptr, std::free);
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, void* external_data)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), dtype_(dtype),
      storage_(external_data, [](void*) {}) {
  if (numel() != 0 && external_data == nullptr) {
    throw std::invalid_argument("Non-empty Tensor requires non-null external data");
  }
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides,
               DataType dtype, std::shared_ptr<void> storage, size_t byte_offset)
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype),
      storage_(std::move(storage)), byte_offset_(byte_offset) {}

size_t Tensor::element_size(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32: return sizeof(float);
    case DataType::INT64: return sizeof(int64_t);
    case DataType::INT32: return sizeof(int32_t);
    case DataType::BOOL: return sizeof(bool);
    case DataType::UINT8: return sizeof(uint8_t);
  }
  throw std::invalid_argument("Unknown Tensor data type");
}

int64_t Tensor::checked_numel(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) throw std::invalid_argument("Tensor dimensions must be non-negative");
    if (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim) {
      throw std::overflow_error("Tensor element count overflow");
    }
    count *= dim;
  }
  return count;
}

std::vector<int64_t> Tensor::contiguous_strides(const std::vector<int64_t>& shape) {
  checked_numel(shape);
  std::vector<int64_t> result(shape.size(), 1);
  for (size_t i = shape.size(); i > 1; --i) {
    result[i - 2] = result[i - 1] * shape[i - 1];
  }
  return result;
}

char* Tensor::raw_data() {
  return storage_ ? static_cast<char*>(storage_.get()) + byte_offset_ : nullptr;
}

const char* Tensor::raw_data() const {
  return storage_ ? static_cast<const char*>(storage_.get()) + byte_offset_ : nullptr;
}

float* Tensor::data_f32() {
  if (dtype_ != DataType::FLOAT32) throw std::runtime_error("Tensor dtype is not FLOAT32");
  return reinterpret_cast<float*>(raw_data());
}

const float* Tensor::data_f32() const {
  if (dtype_ != DataType::FLOAT32) throw std::runtime_error("Tensor dtype is not FLOAT32");
  return reinterpret_cast<const float*>(raw_data());
}

int64_t* Tensor::data_i64() {
  if (dtype_ != DataType::INT64) throw std::runtime_error("Tensor dtype is not INT64");
  return reinterpret_cast<int64_t*>(raw_data());
}

const int64_t* Tensor::data_i64() const {
  if (dtype_ != DataType::INT64) throw std::runtime_error("Tensor dtype is not INT64");
  return reinterpret_cast<const int64_t*>(raw_data());
}

int32_t* Tensor::data_i32() {
  if (dtype_ != DataType::INT32) throw std::runtime_error("Tensor dtype is not INT32");
  return reinterpret_cast<int32_t*>(raw_data());
}

const int32_t* Tensor::data_i32() const {
  if (dtype_ != DataType::INT32) throw std::runtime_error("Tensor dtype is not INT32");
  return reinterpret_cast<const int32_t*>(raw_data());
}

uint8_t* Tensor::data_u8() {
  if (dtype_ != DataType::BOOL && dtype_ != DataType::UINT8) {
    throw std::runtime_error("Tensor dtype is not BOOL or UINT8");
  }
  return reinterpret_cast<uint8_t*>(raw_data());
}

const uint8_t* Tensor::data_u8() const {
  if (dtype_ != DataType::BOOL && dtype_ != DataType::UINT8) {
    throw std::runtime_error("Tensor dtype is not BOOL or UINT8");
  }
  return reinterpret_cast<const uint8_t*>(raw_data());
}

const std::vector<int64_t>& Tensor::shape() const { return shape_; }
const std::vector<int64_t>& Tensor::strides() const { return strides_; }
int64_t Tensor::ndim() const { return static_cast<int64_t>(shape_.size()); }
int64_t Tensor::numel() const { return checked_numel(shape_); }
int64_t Tensor::nbytes() const {
  const auto count = numel();
  const auto size = static_cast<int64_t>(element_size(dtype_));
  if (count > std::numeric_limits<int64_t>::max() / size) {
    throw std::overflow_error("Tensor byte count overflow");
  }
  return count * size;
}
Tensor::DataType Tensor::dtype() const { return dtype_; }
const void* Tensor::data_raw() const { return raw_data(); }

bool Tensor::is_contiguous() const {
  int64_t expected = 1;
  for (size_t i = shape_.size(); i > 0; --i) {
    if (shape_[i - 1] > 1 && strides_[i - 1] != expected) return false;
    expected *= shape_[i - 1];
  }
  return true;
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  if (checked_numel(new_shape) != numel()) {
    throw std::invalid_argument("Tensor reshape must conserve element count");
  }
  if (is_contiguous()) {
    auto new_strides = contiguous_strides(new_shape);
    return Tensor(std::move(new_shape), std::move(new_strides), dtype_, storage_, byte_offset_);
  }

  Tensor result(std::move(new_shape), dtype_);
  const size_t elem_bytes = element_size(dtype_);
  std::vector<int64_t> index(shape_.size(), 0);
  for (int64_t linear = 0; linear < numel(); ++linear) {
    int64_t source_offset = 0;
    for (size_t axis = 0; axis < index.size(); ++axis) {
      source_offset += index[axis] * strides_[axis];
    }
    std::memcpy(result.raw_data() + linear * elem_bytes,
                raw_data() + source_offset * elem_bytes, elem_bytes);
    for (size_t axis = index.size(); axis > 0; --axis) {
      if (++index[axis - 1] < shape_[axis - 1]) break;
      index[axis - 1] = 0;
    }
  }
  return result;
}

Tensor Tensor::transpose(std::vector<int64_t> perm) const {
  if (perm.size() != shape_.size()) {
    throw std::invalid_argument("Tensor transpose permutation rank mismatch");
  }
  std::vector<bool> seen(perm.size(), false);
  std::vector<int64_t> new_shape(perm.size());
  std::vector<int64_t> new_strides(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] < 0 || perm[i] >= ndim() || seen[static_cast<size_t>(perm[i])]) {
      throw std::invalid_argument("Tensor transpose permutation is invalid");
    }
    seen[static_cast<size_t>(perm[i])] = true;
    new_shape[i] = shape_[static_cast<size_t>(perm[i])];
    new_strides[i] = strides_[static_cast<size_t>(perm[i])];
  }
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, storage_, byte_offset_);
}

Tensor Tensor::slice(int axis, int64_t start, int64_t end, int64_t step) const {
  if (axis < 0) axis += static_cast<int>(ndim());
  validate_axis(axis, ndim());
  if (step <= 0) throw std::invalid_argument("Tensor slice step must be positive");
  const int64_t dim = shape_[static_cast<size_t>(axis)];
  if (start < 0) start += dim;
  if (end < 0) end += dim;
  start = std::clamp<int64_t>(start, 0, dim);
  end = std::clamp<int64_t>(end, 0, dim);
  const int64_t length = end > start ? (end - start + step - 1) / step : 0;
  auto new_shape = shape_;
  auto new_strides = strides_;
  new_shape[static_cast<size_t>(axis)] = length;
  new_strides[static_cast<size_t>(axis)] *= step;
  const size_t offset = byte_offset_ +
      static_cast<size_t>(start * strides_[static_cast<size_t>(axis)]) * element_size(dtype_);
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, storage_, offset);
}

void Tensor::fill_zeros() {
  if (is_contiguous()) {
    if (nbytes() != 0) std::memset(raw_data(), 0, static_cast<size_t>(nbytes()));
    return;
  }
  const size_t elem_bytes = element_size(dtype_);
  std::vector<int64_t> index(shape_.size(), 0);
  for (int64_t linear = 0; linear < numel(); ++linear) {
    int64_t offset = 0;
    for (size_t axis = 0; axis < index.size(); ++axis) {
      offset += index[axis] * strides_[axis];
    }
    std::memset(raw_data() + offset * elem_bytes, 0, elem_bytes);
    for (size_t axis = index.size(); axis > 0; --axis) {
      if (++index[axis - 1] < shape_[axis - 1]) break;
      index[axis - 1] = 0;
    }
  }
}

void Tensor::fill_from_raw(const float* src, size_t n) {
  if (dtype_ != DataType::FLOAT32) throw std::runtime_error("Tensor dtype is not FLOAT32");
  if (n != static_cast<size_t>(numel())) throw std::invalid_argument("Tensor fill size mismatch");
  if (!is_contiguous()) throw std::invalid_argument("fill_from_raw requires a contiguous Tensor");
  if (n != 0 && src == nullptr) throw std::invalid_argument("Tensor fill source is null");
  if (n != 0) std::memcpy(raw_data(), src, n * sizeof(float));
}

void Tensor::fill_from_raw(const int64_t* src, size_t n) {
  if (dtype_ != DataType::INT64) throw std::runtime_error("Tensor dtype is not INT64");
  if (n != static_cast<size_t>(numel())) throw std::invalid_argument("Tensor fill size mismatch");
  if (!is_contiguous()) throw std::invalid_argument("fill_from_raw requires a contiguous Tensor");
  if (n != 0 && src == nullptr) throw std::invalid_argument("Tensor fill source is null");
  if (n != 0) std::memcpy(raw_data(), src, n * sizeof(int64_t));
}

void Tensor::fill_from_bytes(const void* src, size_t bytes) {
  if (bytes != static_cast<size_t>(nbytes())) {
    throw std::invalid_argument("Tensor byte fill size mismatch");
  }
  if (!is_contiguous()) throw std::invalid_argument("fill_from_bytes requires a contiguous Tensor");
  if (bytes != 0 && src == nullptr) throw std::invalid_argument("Tensor fill source is null");
  if (bytes != 0) std::memcpy(raw_data(), src, bytes);
}

void Tensor::copy_to_bytes(void* dst, size_t bytes) const {
  if (bytes != static_cast<size_t>(nbytes())) {
    throw std::invalid_argument("Tensor byte copy size mismatch");
  }
  if (bytes != 0 && dst == nullptr) {
    throw std::invalid_argument("Tensor copy destination is null");
  }
  if (bytes == 0) return;
  if (is_contiguous()) {
    std::memcpy(dst, raw_data(), bytes);
    return;
  }

  const size_t element_bytes = element_size(dtype_);
  auto* destination = static_cast<char*>(dst);
  std::vector<int64_t> index(shape_.size(), 0);
  for (int64_t linear = 0; linear < numel(); ++linear) {
    int64_t source_offset = 0;
    for (size_t axis = 0; axis < index.size(); ++axis) {
      source_offset += index[axis] * strides_[axis];
    }
    std::memcpy(
        destination + static_cast<size_t>(linear) * element_bytes,
        raw_data() + source_offset * element_bytes, element_bytes);
    for (size_t axis = index.size(); axis > 0; --axis) {
      if (++index[axis - 1] < shape_[axis - 1]) break;
      index[axis - 1] = 0;
    }
  }
}

Tensor Tensor::clone_contiguous() const {
  Tensor result(shape_, dtype_);
  copy_to_bytes(result.raw_data(), static_cast<size_t>(nbytes()));
  return result;
}

}  // namespace nn
