#pragma once

#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
#include "sdaa_runtime.h" // NOLINT
#include "tecolmk.h" // NOLINT
#include "tecocustom.h" // NOLINT
#include "aten/sdaa_native_functions.h"
#include "torch/torch.h"
#include "torch/extension.h"
#include "torch_sdaa/sdaa_extension.h"
#include "aten/sdaa_functions.h"
// clang-format on

using c10::ScalarType;

#define kSliceMaxNum 8

#define SDAA_CHECK(expr)            \
  do {                              \
    auto ret = (expr);              \
    TORCH_CHECK(ret == sdaaSuccess, \
                "failed to call ",  \
                #expr,              \
                ", at file",        \
                __FILE__,           \
                ":",                \
                __LINE__)           \
  } while (0)

#define TECOLMK_CHECK(expr)                    \
  do {                                         \
    auto ret = (expr);                         \
    TORCH_CHECK(ret == TECOLMK_STATUS_SUCCESS, \
                "failed to call ",             \
                #expr,                         \
                ", detail: ",                  \
                tecolmkGetErrorString(ret),    \
                ", at file",                   \
                __FILE__,                      \
                ":",                           \
                __LINE__)                      \
  } while (0)

#define TECOBLAS_CHECK(expr)                    \
  do {                                          \
    auto ret = (expr);                          \
    TORCH_CHECK(ret == TECOBLAS_STATUS_SUCCESS, \
                "failed to call ",              \
                #expr,                          \
                ", detail: ",                   \
                tecoblasGetErrorString(ret),    \
                ", at file",                    \
                __FILE__,                       \
                ":",                            \
                __LINE__)                       \
  } while (0)

#define TECODNN_CHECK(expr)                    \
  do {                                         \
    auto ret = (expr);                         \
    TORCH_CHECK(ret == TECODNN_STATUS_SUCCESS, \
                "failed to call ",             \
                #expr,                         \
                ", detail: ",                  \
                tecodnnGetErrorString(ret),    \
                ", at file",                   \
                __FILE__,                      \
                ":",                           \
                __LINE__)                      \
  } while (0)

// tecocustom don't support GetErrorString interface yet
#define TECOCUSTOM_CHECK(expr)                    \
  do {                                            \
    auto ret = (expr);                            \
    TORCH_CHECK(ret == TECOCUSTOM_STATUS_SUCCESS, \
                "failed to call ",                \
                #expr,                            \
                ", at file",                      \
                __FILE__,                         \
                ":",                              \
                __LINE__)                         \
  } while (0)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SDAA(x) \
  TORCH_CHECK(x.is_privateuseone(), #x " must be a SDAA tensor")

#define CHECK_INPUT(x) \
  CHECK_SDAA(x);       \
  CHECK_CONTIGUOUS(x)

tecolmkHandle_t GetLmkHandle(sdaaStream_t stream);

tecoblasHandle_t GetBlasHandle(sdaaStream_t stream);

tecodnnHandle_t GetDnnHandle(sdaaStream_t stream);

tecocustomHandle_t GetCustomHandle(sdaaStream_t stream);

enum class TensorFormat {
  NCHW = 0,
  NHWC = 1,
  CHWN = 2,
  NWHC = 3,
  Undefined = 4
};

inline tecocustomDataType_t ToTecocustomDataType(
    const torch::ScalarType& dtype) {
  tecocustomDataType_t dt = TECOCUSTOM_DATA_FLOAT;
  switch (dtype) {
    case ScalarType::Half:
      dt = TECOCUSTOM_DATA_HALF;
      break;
    case ScalarType::BFloat16:
      dt = TECOCUSTOM_DATA_BFLOAT16;
      break;
    case ScalarType::Float:
      dt = TECOCUSTOM_DATA_FLOAT;
      break;
    case ScalarType::Double:
      dt = TECOCUSTOM_DATA_DOUBLE;
      break;
    case ScalarType::Short:
      dt = TECOCUSTOM_DATA_INT16;
      break;
    case ScalarType::Int:
      dt = TECOCUSTOM_DATA_INT32;
      break;
    case ScalarType::Long:
      dt = TECOCUSTOM_DATA_INT64;
      break;
    case ScalarType::Bool:
      dt = TECOCUSTOM_DATA_BOOL;
      break;
    default:
      break;
  }
  return dt;
}

tecocustomTensorDescriptor_t GetTecocustomTensorDesc(
    const std::vector<int>& dims,
    const torch::ScalarType& dtype,
    TensorFormat tf = TensorFormat::Undefined,
    const std::vector<int>& strides = {});

std::vector<int64_t> flatten_shape(const c10::IntArrayRef& shape, int axis);

struct WorkspaceHolder {
  at::Tensor wksp_holder;
  size_t wksp_size;

  WorkspaceHolder(size_t wksp_size_,
                  c10::DeviceType device = c10::kPrivateUse1);

  ~WorkspaceHolder();

  void* get();

  operator void*() { return get(); }
};
