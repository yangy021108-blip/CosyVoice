#include "common.h"  // NOLINT

namespace {

void check_qkv(const torch::Tensor& q,
               const torch::Tensor& k,
               const torch::Tensor& v) {
  CHECK_SDAA(q);
  CHECK_SDAA(k);
  CHECK_SDAA(v);
  TORCH_CHECK(q.dim() == 4, "q must be [B, S, H, D]");
  TORCH_CHECK(k.dim() == 4, "k must be [B, S, H, D]");
  TORCH_CHECK(v.dim() == 4, "v must be [B, S, H, D]");
  TORCH_CHECK(q.sizes() == k.sizes(), "q/k shapes must match");
  TORCH_CHECK(q.sizes() == v.sizes(), "q/v shapes must match");
  TORCH_CHECK(q.scalar_type() == torch::kHalf || q.scalar_type() == torch::kBFloat16,
              "only fp16/bf16 are supported in this POC");
}

tecolmkDataPresion_t lmk_precision(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kHalf) {
    return TECOLMK_PRESION_HALF;
  }
  if (tensor.scalar_type() == torch::kFloat) {
    return TECOLMK_PRESION_FLOAT;
  }
  TORCH_CHECK(false, "fused Flow norm only supports fp16/fp32");
}

std::tuple<torch::Tensor, torch::Tensor> run_flow_fused_norm_impl(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_SDAA(gamma);
  CHECK_SDAA(beta);
  TORCH_CHECK(input.dim() == 3, "input must be [B, N, D]");
  TORCH_CHECK(residual.sizes() == input.sizes(),
              "residual shape must match input");
  TORCH_CHECK(gamma.dim() == 2 && beta.dim() == 2,
              "gamma/beta must be [B, D]");
  TORCH_CHECK(gamma.size(0) == input.size(0) &&
                  gamma.size(1) == input.size(2),
              "gamma shape must match [B, D]");
  TORCH_CHECK(beta.sizes() == gamma.sizes(),
              "beta shape must match gamma");
  TORCH_CHECK(gamma.stride(1) == 1 && beta.stride(1) == 1,
              "gamma/beta rows must be contiguous");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == gamma.scalar_type() &&
                  input.scalar_type() == beta.scalar_type(),
              "all fused Flow norm tensors must have the same dtype");

  const int batch = static_cast<int>(input.size(0));
  const int rows = static_cast<int>(input.size(1));
  const int dim = static_cast<int>(input.size(2));
  auto output = torch::empty_like(input);
  auto residual_out = torch::empty_like(input);
  auto stream = torch::sdaa::getCurrentSDAAStream();
  auto handle = GetLmkHandle(stream);
  const auto precision = lmk_precision(input);
  size_t workspace_size = 0;
  TECOLMK_CHECK(tecolmkGetFusedNormWorkspaceSize(
      handle, precision, TECOLMK_NORMALIZATION_LAYER, rows, dim, dim, -1,
      1.0f, 1.0f, static_cast<float>(eps), &workspace_size,
      false));
  WorkspaceHolder workspace(workspace_size);

  for (int index = 0; index < batch; ++index) {
    auto input_i = input.select(0, index);
    auto residual_i = residual.select(0, index);
    auto gamma_i = gamma.select(0, index);
    auto beta_i = beta.select(0, index);
    auto output_i = output.select(0, index);
    auto residual_out_i = residual_out.select(0, index);
    TECOLMK_CHECK(tecolmkFusedNorm(
        handle, precision, TECOLMK_NORMALIZATION_LAYER, rows, dim, dim, -1,
        1.0f, 1.0f, static_cast<float>(eps), input_i.const_data_ptr(),
        nullptr, gamma_i.const_data_ptr(), beta_i.const_data_ptr(),
        residual_i.const_data_ptr(), nullptr,
        output_i.mutable_data_ptr(), residual_out_i.mutable_data_ptr(),
        nullptr, nullptr, workspace.get(), false));
  }
  return {output, residual_out};
}

torch::Tensor make_group_view(const torch::Tensor& tensor,
                              int packed_start,
                              int run_count,
                              int seq_len,
                              int num_heads,
                              int head_dim) {
  const int64_t row_stride = tensor.stride(1);
  return tensor.narrow(1, packed_start, run_count * seq_len).as_strided(
      {run_count, seq_len, num_heads, head_dim},
      {seq_len * row_stride, row_stride, tensor.stride(2), tensor.stride(3)});
}

torch::Tensor run_fixed_flash_attention_into(const torch::Tensor& q,
                                             const torch::Tensor& k,
                                             const torch::Tensor& v,
                                             torch::Tensor out,
                                             double scale,
                                             void* shared_workspace = nullptr,
                                             size_t shared_workspace_size = 0,
                                             void* shared_save_info = nullptr,
                                             int shared_save_info_size = 0) {
  check_qkv(q, k, v);
  TORCH_CHECK(out.sizes() == q.sizes(), "output shape must match q shape");
  TORCH_CHECK(out.scalar_type() == q.scalar_type(),
              "output dtype must match q dtype");
  TORCH_CHECK(out.device() == q.device(), "output device must match q device");

  const int batch_size = static_cast<int>(q.size(0));
  const int q_seq_len = static_cast<int>(q.size(1));
  const int kv_seq_len = static_cast<int>(k.size(1));
  const int num_heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));

  auto stream = torch::sdaa::getCurrentSDAAStream();
  auto handle = GetCustomHandle(stream);

  const bool use_shared_buffers = shared_save_info != nullptr;
  size_t workspace_size = shared_workspace_size;
  int save_info_size = shared_save_info_size;
  if (!use_shared_buffers) {
    TECOCUSTOM_CHECK(tecocustomGetFlashAttentionForwardWorkspaceSize(
        &workspace_size, q_seq_len, kv_seq_len, batch_size, num_heads, head_dim, head_dim));
    TECOCUSTOM_CHECK(tecocustomGetFlashAttentionForwardSaveInfoSize(
        &save_info_size, q_seq_len, kv_seq_len, batch_size, num_heads));
  }

  WorkspaceHolder local_workspace(use_shared_buffers ? 0 : workspace_size);
  void* workspace =
      use_shared_buffers ? shared_workspace : local_workspace.get();
  auto options_u8 = torch::TensorOptions().dtype(torch::kU8).device(torch::kPrivateUse1);
  torch::Tensor local_save_info;
  void* save_info = shared_save_info;
  if (!use_shared_buffers) {
    local_save_info = save_info_size > 0 ? torch::empty({save_info_size}, options_u8)
                                        : torch::empty({1}, options_u8);
    save_info = local_save_info.mutable_data_ptr();
  }

  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "q/k/v last dimension must be contiguous");
  auto q_desc = GetTecocustomTensorDesc(
      {batch_size, q_seq_len, num_heads, head_dim}, q.scalar_type(), TensorFormat::Undefined,
      {static_cast<int>(q.stride(0)), static_cast<int>(q.stride(1)), static_cast<int>(q.stride(2)),
       static_cast<int>(q.stride(3))});
  auto k_desc = GetTecocustomTensorDesc(
      {batch_size, kv_seq_len, num_heads, head_dim}, k.scalar_type(), TensorFormat::Undefined,
      {static_cast<int>(k.stride(0)), static_cast<int>(k.stride(1)), static_cast<int>(k.stride(2)),
       static_cast<int>(k.stride(3))});
  auto v_desc = GetTecocustomTensorDesc(
      {batch_size, kv_seq_len, num_heads, head_dim}, v.scalar_type(), TensorFormat::Undefined,
      {static_cast<int>(v.stride(0)), static_cast<int>(v.stride(1)), static_cast<int>(v.stride(2)),
       static_cast<int>(v.stride(3))});
  auto o_desc = GetTecocustomTensorDesc(
      {batch_size, q_seq_len, num_heads, head_dim}, out.scalar_type(), TensorFormat::Undefined,
      {static_cast<int>(out.stride(0)), static_cast<int>(out.stride(1)),
       static_cast<int>(out.stride(2)), static_cast<int>(out.stride(3))});
  auto wksp_desc = GetTecocustomTensorDesc({static_cast<int>(workspace_size)}, torch::kUInt8,
                                          TensorFormat::Undefined, {1});
  auto save_desc = GetTecocustomTensorDesc({std::max(save_info_size, 1)}, torch::kUInt8,
                                          TensorFormat::Undefined, {1});
  auto bias_desc = GetTecocustomTensorDesc({}, q.scalar_type(), TensorFormat::Undefined, {});

  const int q_row_ld = static_cast<int>(q.stride(1));
  const int q_head_stride = static_cast<int>(q.stride(2));
  const int q_batch_ld = static_cast<int>(q.stride(0));
  const int k_row_ld = static_cast<int>(k.stride(1));
  const int k_head_stride = static_cast<int>(k.stride(2));
  const int k_batch_ld = static_cast<int>(k.stride(0));
  const int v_row_ld = static_cast<int>(v.stride(1));
  const int v_head_stride = static_cast<int>(v.stride(2));
  const int v_batch_ld = static_cast<int>(v.stride(0));
  const int o_row_ld = static_cast<int>(out.stride(1));
  const int o_head_stride = static_cast<int>(out.stride(2));
  const int o_batch_ld = static_cast<int>(out.stride(0));

  TECOCUSTOM_CHECK(tecocustomFlashAttentionForward(
      handle, q_desc, q.const_data_ptr(), k_desc, k.const_data_ptr(), v_desc, v.const_data_ptr(),
      o_desc, out.mutable_data_ptr(), wksp_desc, workspace, save_desc,
      save_info, bias_desc, nullptr, q_row_ld, q_head_stride, q_batch_ld,
      k_row_ld, k_head_stride, k_batch_ld, v_row_ld, v_head_stride, v_batch_ld, batch_size,
      q_seq_len, kv_seq_len, head_dim, head_dim, num_heads, o_row_ld, o_head_stride, o_batch_ld,
      static_cast<float>(scale), 1, false, 0, false, false, 0, num_heads, 0, 0.0f, false,
      false));

  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(q_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(k_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(v_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(o_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(wksp_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(save_desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(bias_desc));
  return out;
}

torch::Tensor run_fixed_flash_attention_impl(const torch::Tensor& q,
                                             const torch::Tensor& k,
                                             const torch::Tensor& v,
                                             double scale) {
  auto out = torch::empty(q.sizes(), q.options());
  return run_fixed_flash_attention_into(q, k, v, out, scale);
}

torch::Tensor run_varlen_grouped_flash_attention_impl(const torch::Tensor& q,
                                                      const torch::Tensor& k,
                                                      const torch::Tensor& v,
                                                      const torch::Tensor& cu_seqlens,
                                                      double scale) {
  check_qkv(q, k, v);
  TORCH_CHECK(q.size(0) == 1, "varlen grouped POC only supports batch size 1");
  TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be a 1-D tensor");
  TORCH_CHECK(cu_seqlens.device().is_cpu(), "cu_seqlens must stay on CPU");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32 ||
                  cu_seqlens.scalar_type() == torch::kInt64,
              "cu_seqlens must be int32 or int64");

  const auto cu = cu_seqlens.contiguous();
  const int segment_count = static_cast<int>(cu.size(0)) - 1;
  TORCH_CHECK(segment_count >= 0, "cu_seqlens must contain at least one prefix item");

  const int total_tokens = static_cast<int>(q.size(1));
  const int num_heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  const int64_t q_last = cu.scalar_type() == torch::kInt32
                             ? static_cast<int64_t>(cu.data_ptr<int>()[segment_count])
                             : cu.data_ptr<int64_t>()[segment_count];
  TORCH_CHECK(q_last == total_tokens,
              "cu_seqlens last value must match packed token count: expected ", total_tokens,
              ", got ", q_last);

  auto out = torch::empty(q.sizes(), q.options());
  const int* cu32 = cu.scalar_type() == torch::kInt32 ? cu.data_ptr<int>() : nullptr;
  const int64_t* cu64 = cu.scalar_type() == torch::kInt64 ? cu.data_ptr<int64_t>() : nullptr;

  auto length_at = [&](int idx) -> int {
    if (cu32 != nullptr) {
      return cu32[idx + 1] - cu32[idx];
    }
    return static_cast<int>(cu64[idx + 1] - cu64[idx]);
  };
  size_t shared_workspace_size = 0;
  int shared_save_info_size = 0;
  int plan_idx = 0;
  while (plan_idx < segment_count) {
    const int seq_len = length_at(plan_idx);
    if (seq_len <= 0) {
      ++plan_idx;
      continue;
    }
    int next_idx = plan_idx + 1;
    while (next_idx < segment_count && length_at(next_idx) == seq_len) {
      ++next_idx;
    }
    const int run_count = next_idx - plan_idx;
    size_t group_workspace_size = 0;
    int group_save_info_size = 0;
    TECOCUSTOM_CHECK(tecocustomGetFlashAttentionForwardWorkspaceSize(
        &group_workspace_size, seq_len, seq_len, run_count, num_heads, head_dim, head_dim));
    TECOCUSTOM_CHECK(tecocustomGetFlashAttentionForwardSaveInfoSize(
        &group_save_info_size, seq_len, seq_len, run_count, num_heads));
    shared_workspace_size = std::max(shared_workspace_size, group_workspace_size);
    shared_save_info_size = std::max(shared_save_info_size, group_save_info_size);
    plan_idx = next_idx;
  }

  WorkspaceHolder shared_workspace(shared_workspace_size);
  auto options_u8 = torch::TensorOptions().dtype(torch::kU8).device(torch::kPrivateUse1);
  auto shared_save_info =
      torch::empty({std::max(shared_save_info_size, 1)}, options_u8);


  int packed_start = 0;
  int seg_idx = 0;
  while (seg_idx < segment_count) {
    const int seq_len = length_at(seg_idx);
    if (seq_len <= 0) {
      ++seg_idx;
      continue;
    }

    int next_idx = seg_idx + 1;
    while (next_idx < segment_count && length_at(next_idx) == seq_len) {
      ++next_idx;
    }

    const int run_count = next_idx - seg_idx;
    const int packed_tokens = run_count * seq_len;

    auto q_i = make_group_view(q, packed_start, run_count, seq_len, num_heads, head_dim);
    auto k_i = make_group_view(k, packed_start, run_count, seq_len, num_heads, head_dim);
    auto v_i = make_group_view(v, packed_start, run_count, seq_len, num_heads, head_dim);
    auto out_i = make_group_view(out, packed_start, run_count, seq_len, num_heads, head_dim);
    run_fixed_flash_attention_into(
        q_i, k_i, v_i, out_i, scale, shared_workspace.get(),
        shared_workspace_size, shared_save_info.mutable_data_ptr(),
        shared_save_info_size);

    packed_start += packed_tokens;
    seg_idx = next_idx;
  }

  return out;
}


torch::Tensor run_varlen_lmk_flash_attention_impl(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens,
    double scale) {
  check_qkv(q, k, v);
  TORCH_CHECK(q.size(0) == 1, "LMK varlen POC only supports packed batch size 1");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "LMK varlen POC requires contiguous q/k/v");
  TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be a 1-D tensor");
  TORCH_CHECK(cu_seqlens.device().is_cpu(), "cu_seqlens must stay on CPU");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32 ||
                  cu_seqlens.scalar_type() == torch::kInt64,
              "cu_seqlens must be int32 or int64");

  const auto cu = cu_seqlens.contiguous();
  const int batch_size = static_cast<int>(cu.size(0)) - 1;
  TORCH_CHECK(batch_size > 0, "cu_seqlens must describe at least one sequence");

  auto seq_lens = torch::empty(
      {batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  auto* seq_lens_ptr = seq_lens.data_ptr<int>();
  const int* cu32 = cu.scalar_type() == torch::kInt32 ? cu.data_ptr<int>() : nullptr;
  const int64_t* cu64 = cu.scalar_type() == torch::kInt64 ? cu.data_ptr<int64_t>() : nullptr;
  auto cu_at = [&](int idx) -> int64_t {
    return cu32 != nullptr ? static_cast<int64_t>(cu32[idx]) : cu64[idx];
  };

  TORCH_CHECK(cu_at(0) == 0, "cu_seqlens must start at zero");
  for (int i = 0; i < batch_size; ++i) {
    const int64_t seq_len = cu_at(i + 1) - cu_at(i);
    TORCH_CHECK(seq_len > 0 && seq_len <= 2147483647LL,
                "all sequence lengths must be positive int32 values");
    seq_lens_ptr[i] = static_cast<int>(seq_len);
  }
  TORCH_CHECK(cu_at(batch_size) == q.size(1),
              "cu_seqlens last value must match packed token count: expected ",
              q.size(1), ", got ", cu_at(batch_size));

  const int num_heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  auto stream = torch::sdaa::getCurrentSDAAStream();
  auto handle = GetLmkHandle(stream);

  size_t workspace_size = 0;
  TECOLMK_CHECK(tecolmkFlashAttentionWorkspaceSize(
      handle, seq_lens_ptr, batch_size, num_heads, head_dim, 0, head_dim,
      &workspace_size));
  WorkspaceHolder workspace(workspace_size);
  auto out = torch::empty(q.sizes(), q.options());

  TECOLMK_CHECK(tecolmkFlashAttention(
      handle, q.const_data_ptr(), k.const_data_ptr(), v.const_data_ptr(),
      static_cast<int>(q.stride(2)), static_cast<int>(q.stride(1)),
      static_cast<int>(k.stride(2)), static_cast<int>(k.stride(1)),
      static_cast<int>(v.stride(2)), static_cast<int>(v.stride(1)), batch_size,
      seq_lens_ptr, num_heads, num_heads, head_dim, 0, head_dim,
      static_cast<float>(scale), out.mutable_data_ptr(),
      static_cast<int>(out.stride(2)), static_cast<int>(out.stride(1)),
      workspace.get(), TECOLMK_NO_MASK));
  return out;
}
}  // namespace

torch::Tensor mm_encoder_flash_attention_fixed(const torch::Tensor& q,
                                               const torch::Tensor& k,
                                               const torch::Tensor& v,
                                               double scale) {
  return run_fixed_flash_attention_impl(q, k, v, scale);
}

torch::Tensor mm_encoder_flash_attention_varlen_grouped(const torch::Tensor& q,
                                                        const torch::Tensor& k,
                                                        const torch::Tensor& v,
                                                        const torch::Tensor& cu_seqlens,
                                                        double scale) {
  return run_varlen_grouped_flash_attention_impl(q, k, v, cu_seqlens, scale);
}

torch::Tensor mm_encoder_flash_attention_varlen_lmk(const torch::Tensor& q,
                                                    const torch::Tensor& k,
                                                    const torch::Tensor& v,
                                                    const torch::Tensor& cu_seqlens,
                                                    double scale) {
  return run_varlen_lmk_flash_attention_impl(q, k, v, cu_seqlens, scale);
}
torch::Tensor mm_encoder_flash_attention_two_segments(const torch::Tensor& q,
                                                      const torch::Tensor& k,
                                                      const torch::Tensor& v,
                                                      int64_t len0,
                                                      int64_t len1,
                                                      double scale) {
  TORCH_CHECK(len0 >= 0 && len1 >= 0, "segment lengths must be non-negative");
  auto cu = torch::empty({3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  auto* cu_ptr = cu.data_ptr<int>();
  cu_ptr[0] = 0;
  cu_ptr[1] = static_cast<int>(len0);
  cu_ptr[2] = static_cast<int>(len0 + len1);
  return run_varlen_grouped_flash_attention_impl(q, k, v, cu, scale);
}

std::tuple<torch::Tensor, torch::Tensor> flow_fused_norm(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double eps) {
  return run_flow_fused_norm_impl(
      input, residual, gamma, beta, eps);
}

TORCH_LIBRARY(_C_sdaa_poc, m) {
  m.def("mm_encoder_flash_attention_fixed(Tensor q, Tensor k, Tensor v, float scale) -> Tensor");
  m.impl("mm_encoder_flash_attention_fixed", torch::kPrivateUse1, &mm_encoder_flash_attention_fixed);
  m.def("mm_encoder_flash_attention_varlen_grouped(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, float scale) -> Tensor");
  m.impl("mm_encoder_flash_attention_varlen_grouped", torch::kPrivateUse1, &mm_encoder_flash_attention_varlen_grouped);
  m.def("mm_encoder_flash_attention_varlen_lmk(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, float scale) -> Tensor");
  m.impl("mm_encoder_flash_attention_varlen_lmk", torch::kPrivateUse1, &mm_encoder_flash_attention_varlen_lmk);
  m.def("mm_encoder_flash_attention_two_segments(Tensor q, Tensor k, Tensor v, int len0, int len1, float scale) -> Tensor");
  m.impl("mm_encoder_flash_attention_two_segments", torch::kPrivateUse1, &mm_encoder_flash_attention_two_segments);
  m.def("flow_fused_norm(Tensor input, Tensor residual, Tensor gamma, Tensor beta, float eps) -> (Tensor, Tensor)");
  m.impl("flow_fused_norm", torch::kPrivateUse1, &flow_fused_norm);
}
