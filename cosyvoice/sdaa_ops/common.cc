#include "common.h"  // NOLINT

tecolmkHandle_t GetLmkHandle(sdaaStream_t stream) {
  static std::unordered_map<sdaaStream_t, tecolmkHandle_t> map;
  if (map.find(stream) == map.end()) {
    auto& handle = map[stream];
    TECOLMK_CHECK(tecolmkCreateHandle(&handle));
    TECOLMK_CHECK(tecolmkSetStream(handle, stream));
    // printf("create handle(%p) for stream(%p)\n", handle, stream);
  }
  return map[stream];
}

tecoblasHandle_t GetBlasHandle(sdaaStream_t stream) {
  static std::unordered_map<sdaaStream_t, tecoblasHandle_t> map;
  if (map.find(stream) == map.end()) {
    auto& handle = map[stream];
    TECOBLAS_CHECK(tecoblasCreate(&handle));
    TECOBLAS_CHECK(tecoblasSetStream(handle, stream));
    // printf("create handle(%p) for stream(%p)\n", handle, stream);
  }
  return map[stream];
}

tecodnnHandle_t GetDnnHandle(sdaaStream_t stream) {
  static std::unordered_map<sdaaStream_t, tecodnnHandle_t> map;
  if (map.find(stream) == map.end()) {
    auto& handle = map[stream];
    TECODNN_CHECK(tecodnnCreate(&handle));
    TECODNN_CHECK(tecodnnSetStream(handle, stream));
  }
  return map[stream];
}

tecocustomHandle_t GetCustomHandle(sdaaStream_t stream) {
  static std::unordered_map<sdaaStream_t, tecocustomHandle_t> map;
  if (map.find(stream) == map.end()) {
    auto& handle = map[stream];
    TECOCUSTOM_CHECK(tecocustomCreate(&handle));
    TECOCUSTOM_CHECK(tecocustomSetStream(handle, stream));
  }
  return map[stream];
}

std::vector<int64_t> flatten_shape(const c10::IntArrayRef& shape, int axis) {
  int64_t dim = static_cast<int64_t>(shape.size());
  TORCH_CHECK(dim > 0, "shape must not be empty");
  while (dim > 0 && axis < 0) {
    axis = axis + dim;
  }
  TORCH_CHECK(axis < dim, "out of range, valid range is [0, %d)", dim);
  return {
      std::accumulate(
          shape.begin(), shape.begin() + axis, 1, std::multiplies<int64_t>()),
      std::accumulate(
          shape.begin() + axis, shape.end(), 1, std::multiplies<int64_t>())};
}

bool enabled_workspace_out_of_bound_check() {
  static std::once_flag init_flag;
  static bool enabled = false;
  std::call_once(init_flag, []() {
    auto env_str = std::getenv("VLLM_SDAA_WORKSPACE_OUT_OF_BOUND_CHECK");
    enabled = env_str ? std::string(env_str) == "1" : false;
  });
  return enabled;
}

WorkspaceHolder::WorkspaceHolder(size_t wksp_size_, c10::DeviceType device_type)
    : wksp_size(wksp_size_) {
  auto options = torch::TensorOptions().dtype(torch::kU8).device(device_type);
  if (enabled_workspace_out_of_bound_check()) {
    wksp_holder =
        torch::zeros({static_cast<int64_t>(wksp_size + 512)}, options);
  } else if (wksp_size > 0) {
    wksp_holder = torch::empty({static_cast<int64_t>(wksp_size)}, options);
  }
}

WorkspaceHolder::~WorkspaceHolder() {
  if (enabled_workspace_out_of_bound_check()) {
    auto cpu_tensor =
        wksp_holder.slice(0, wksp_size, wksp_size + 512).to(torch::kCPU);
    auto cpu_ptr = cpu_tensor.data_ptr<uint8_t>();
    int64_t num = std::accumulate(
        cpu_ptr, cpu_ptr + 512, 0, [](int64_t acc, uint8_t val) {
          return acc + (val != 0);
        });
    if (num != 0) {
      std::cout << "workspace out of bound check failed, " << num
                << " bytes are not zero in the workspace, "
                << "please check your code, workspace size is " << wksp_size
                << ", workspace holder size is " << cpu_tensor.numel()
                << std::endl;
      TORCH_CHECK(false);
    }
  }
}

void* WorkspaceHolder::get() {
  return (enabled_workspace_out_of_bound_check() || wksp_size > 0)
             ? wksp_holder.mutable_data_ptr()
             : nullptr;
}

struct NCHWValue {
  int N, C, H, W;
};

NCHWValue GetNCHWValue(const std::vector<int> dims,
                       tecocustomTensorFormat_t tf) {
  int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
  switch (tf) {
    case TECOCUSTOM_TENSOR_NCHW:
      N = dims[0];
      C = dims[1];
      H = dims[2];
      W = dims[3];
      break;
    case TECOCUSTOM_TENSOR_NHWC:
      N = dims[0];
      C = dims[3];
      H = dims[1];
      W = dims[2];
      break;
    case TECOCUSTOM_TENSOR_CHWN:
      N = dims[3];
      C = dims[0];
      H = dims[1];
      W = dims[2];
      break;
    case TECOCUSTOM_TENSOR_NWHC:
      N = dims[0];
      C = dims[3];
      H = dims[2];
      W = dims[1];
      break;
    default:
      break;
  }
  return {N, C, H, W};
}

inline tecocustomTensorFormat_t GetTecocustomTF(const TensorFormat TF) {
  tecocustomTensorFormat_t tf;
  switch (TF) {
    case TensorFormat::NCHW:
      tf = TECOCUSTOM_TENSOR_NCHW;
      break;
    case TensorFormat::NHWC:
      tf = TECOCUSTOM_TENSOR_NHWC;
      break;
    case TensorFormat::CHWN:
      tf = TECOCUSTOM_TENSOR_CHWN;
      break;
    case TensorFormat::NWHC:
      tf = TECOCUSTOM_TENSOR_NWHC;
      break;
    default:
      TORCH_CHECK(false, "Invaild tensor format when use GetTecocustomTF.");
      break;
  }
  return tf;
}

tecocustomTensorDescriptor_t GetTecocustomTensorDesc(
    const std::vector<int>& dims,
    const torch::ScalarType& dtype,
    TensorFormat tf,
    const std::vector<int>& strides) {
  tecocustomDataType_t dt = ToTecocustomDataType(dtype);
  tecocustomTensorDescriptor_t CustomDesc;
  TECOCUSTOM_CHECK(tecocustomCreateTensorDescriptor(&CustomDesc));

  auto tmp_dims = dims;
  if (dims.empty()) {
    tmp_dims.push_back(1);
  }

  if (tf != TensorFormat::Undefined && tmp_dims.size() <= 4) {
    TORCH_CHECK(strides.empty(),
                "Strides is not supported to set when tensor format is not "
                "TensorFormat::Undefined and dims.size() is less than 5.");
    tecocustomTensorFormat_t t_f = GetTecocustomTF(tf);

    std::vector<int> dimensions(4, 1);
    int index = 3;
    for (int i = tmp_dims.size() - 1; i >= 0; i--) {
      dimensions[index--] = tmp_dims[i];
    }

    int N, C, H, W;
    NCHWValue getNCHWvalue = GetNCHWValue(dimensions, t_f);
    N = getNCHWvalue.N;
    C = getNCHWvalue.C;
    H = getNCHWvalue.H;
    W = getNCHWvalue.W;

    TECOCUSTOM_CHECK(
        tecocustomSetTensor4dDescriptor(CustomDesc, t_f, dt, N, C, H, W));
  } else {
    int dims_arr[kSliceMaxNum];
    TORCH_CHECK(tmp_dims.size() <= kSliceMaxNum,
                "The max ND Descriptor dims size is kSliceMaxNum, but got: ",
                tmp_dims.size());
    std::copy(tmp_dims.begin(), tmp_dims.end(), dims_arr);

    TECOCUSTOM_CHECK(tecocustomSetTensorNdDescriptor(
        CustomDesc, dt, tmp_dims.size(), dims_arr, strides.data()));
  }
  return CustomDesc;
}
