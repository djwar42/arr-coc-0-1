---
sourceFile: "Multi-Token Prediction (MTP) — Megatron Core - NVIDIA Docs Hub"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:32.936Z"
---

# Multi-Token Prediction (MTP) — Megatron Core - NVIDIA Docs Hub

21d23573-25c2-4d77-a096-5f6efa8abab5

Multi-Token Prediction (MTP) — Megatron Core - NVIDIA Docs Hub

61649bc3-b7a7-4c3e-9086-c55eb386f3dc

https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/multi_token_prediction.html

## Skip to main content

## Megatron Core

## Megatron Core

Multi-Token Prediction (MTP)

https://docs.nvidia.com#multi-token-prediction-mtp

Multi-Token Prediction (MTP) extends the prediction scope to multiple future tokens at each position. On the one hand, an MTP objective densifies the training signals and may improve data efficiency. On the other hand, MTP may enable the model to pre-plan its representations for better prediction of future tokens. In this implementation of MTP, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth. The following figure illustrates our implementation of MTP in

DeepSeek-V3

https://github.com/deepseek-ai/DeepSeek-V3/

The k-th MTP module consists of a shared embedding layer, a projection matrix, a Transformer block, and a shared output head. For the i-th input token at the (k - 1)-th prediction depth, we first combine the representation of the i-th token and the embedding of the (i + K)-th token with the linear projection. The combined serves as the input of the Transformer block at the k-th depth to produce the output representation.

For more information, please refer to

DeepSeek-V3 Technical Report

https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf

## Related Arguments

https://docs.nvidia.com#related-arguments

We can train GPTModel like models with Multi-Token Prediction (MTP) by setting mtp_num_layers to be a positive integer.

## Description

mtp_num_layers

Number of Multi-Token Prediction (MTP) Layers. MTP extends the prediction scope to multiple future tokens at each position. This MTP implementation sequentially predict additional tokens by using D sequential modules to predict D additional tokens. Default is None.

mtp_loss_scaling_factor

Scaling factor of Multi-Token Prediction (MTP) loss. We compute the average of the MTP losses across all depths, and multiply it the scaling factor to obtain the overall MTP loss, which serves as an additional training objective. Default is 0.1.

## Precautions

https://docs.nvidia.com#precautions

Please do not use Context Parallel (CP), or arbitrary AttnMaskType, or learned absolute position embedding type with MTP. These use cases are not yet supported.

## On this page

https://docs.nvidia.com#precautions

## Privacy Policy

https://docs.nvidia.com#precautions

## Manage My Privacy

https://www.nvidia.com/en-us/about-nvidia/privacy-center/

## Do Not Sell or Share My Data

https://www.nvidia.com/en-us/preferences/start/

## Terms of Service

https://www.nvidia.com/en-us/about-nvidia/terms-of-service/

## Accessibility

https://www.nvidia.com/en-us/about-nvidia/accessibility/

## Corporate Policies

https://www.nvidia.com/en-us/about-nvidia/company-policies/

## Product Security

https://www.nvidia.com/en-us/product-security/

https://www.nvidia.com/en-us/contact/

Copyright © 2022-2025, NVIDIA Corporation.

Last updated on Sep 25, 2025.

