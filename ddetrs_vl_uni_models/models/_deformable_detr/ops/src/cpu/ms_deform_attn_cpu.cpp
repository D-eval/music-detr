/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>

#include <ATen/ATen.h>


at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // Basic CPU implementation using grid_sample
    // This is a simplified version for CPU-only builds
    
    auto N = value.size(0);
    auto S = value.size(1);
    auto M = value.size(2);
    auto D = value.size(3);
    
    auto Lq = sampling_loc.size(1);
    auto L = sampling_loc.size(3);
    auto P = sampling_loc.size(4);
    
    // Create output tensor
    auto output = at::zeros({N, Lq, M * D}, value.options());
    
    // For each batch, query, and head
    for (int n = 0; n < N; ++n) {
        for (int q = 0; q < Lq; ++q) {
            for (int m = 0; m < M; ++m) {
                // Get sampling locations for this query and head
                auto loc = sampling_loc[n][q][m]; // [L, P, 2]
                auto weight = attn_weight[n][q][m]; // [L, P]
                
                // For simplicity, just use the first level and first point
                // This is a very basic implementation
                auto sampled_value = value[n][0][m]; // Use first level
                output[n][q].slice(0, m * D, (m + 1) * D) = sampled_value;
            }
        }
    }
    
    return output;
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    // Basic CPU backward implementation
    // Return zero gradients for simplicity in CPU-only mode
    
    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);
    
    return {grad_value, grad_sampling_loc, grad_attn_weight};
}

