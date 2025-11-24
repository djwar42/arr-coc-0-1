---
sourceFile: "DeepSeek-V3 Explained, Part 1: Understanding Multi-Head Latent Attention - Towards AI"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:20.338Z"
---

# DeepSeek-V3 Explained, Part 1: Understanding Multi-Head Latent Attention - Towards AI

9f849249-df2a-4b18-8f5a-555a1c0e46d5

DeepSeek-V3 Explained, Part 1: Understanding Multi-Head Latent Attention - Towards AI

57f63dd4-e290-46df-9090-43677bcddce4

https://pub.towardsai.net/deepseek-v3-explained-part-1-understanding-multi-head-latent-attention-bac648681926

## Open in app

## Medium Logo

Making AI accessible to 100K+ learners. Find the most practical, hands-on and comprehensive AI Engineering and AI for Work certifications at

academy.towardsai.net

http://academy.towardsai.net

- we have pathways for any experience level. Monthly cohorts still open — use COHORT10 for 10% off!
DeepSeek-V3 Explained, Part 1: Understanding Multi-Head Latent Attention

http://academy.towardsai.net

9 min read · Apr 13, 2025

Press enter or click to view image in full size Vegapunk No.01 One Piece Character Generated with ChatGPT

This is the first article of our new series “

DeepSeek-V3 Explained”

, where we will try to demystify DeepSeek-V3 [1, 2], the latest model open-sourced by DeepSeek.

In this series, we aim to cover two major topics:

Major architecture innovations in DeepSeek-V3

, including MLA (Multi-head Latent Attention) [3], DeepSeekMoE [4], auxiliary-loss-free load balancing [5], and multi-token prediction training.

Training of DeepSeek-V3

, covering the pre-training, fine-tuning, and reinforcement learning (RL) alignment phases.

## This article mainly focuses on

Multi-head Latent Attention

, which was first introduced during the development of DeepSeek-V2 and later adopted in DeepSeek-V3 as well.

## Background

We begin with a review of standard Multi-Head Attention (MHA), explaining the need for a Key-Value (KV) cache during inference. We then explore how MQA (Multi-Query Attention) and GQA (Grouped-Query Attention) aim to optimize memory and computational efficiency. Finally, we touch on how RoPE (Rotary Positional Embedding) integrates positional information into the attention mechanism.

Multi-head Latent Attention

An in-depth introduction to MLA, covering its core motivations, the need for decoupled RoPE, and how it improves performance compared to traditional attention mechanisms.

References.

To better understand MLA and to make this article self-contained we’ll revisit several related concepts in this section before diving into the details of Multi-head Latent Attention.

MHA in Decoder-only Transformers

Note that MLA is specifically designed to accelerate inference in autoregressive text generation. Therefore, the Multi-Head Attention (MHA) we refer to in this context is within a decoder-only Transformer architecture.

The figure below compares three Transformer architectures used for decoding. In ( a ), we see both the encoder and decoder as originally proposed in the

‘Attention is All You Need’

paper. This decoder design was later simplified by [6], resulting in the decoder-only Transformer shown in ( b ), which became the foundation for many generative models such as GPT [8].

Today, large language models (LLMs) more commonly adopt the architecture shown in ( c ), as it offers more stable training. In this design, normalization is applied to the input rather than the output, and LayerNorm is replaced with the more stable RMSNorm. This serves as the baseline architecture for the discussion in this article.

Press enter or click to view image in full size Figure 1. Transformer architectures. ( a ) encoder-decoder proposed in [6]. ( b ) Decoder-only Transformer proposed in [7] and used in GPT [8]. ( c ) An optimized version of (b) with RMS Norm before attention. [3]

Within this context, the Multi-Head Attention (MHA) computation largely follows the approach described in [6], as illustrated in the figure below:

Press enter or click to view image in full size Figure 2. Scaled dot-product attention vs. Multi-Head Attention. Image from [6].

## Assume we have

attention heads, each with a dimensionality of

. The concatenated output of all heads will then have a total dimension of (

## Given a model with

layers, let’s denote the input to a specific layer for the t-th token as

. To compute multi-head attention, we first project

from dimension

) using linear mapping matrices.

More formally, we have the following equations (adapted from [3]):

are the linear mapping matrices:

## Press enter or click to view image in full size

After this projection, the resulting vectors

are each split into

heads. Each head then computes scaled dot-product attention independently as follows:

## Press enter or click to view image in full size

is an output projection matrix that maps the concatenated attention output from dimension (

) back to the original dimension

Note that the process described by Equations (1) to (8) applies to a single token. During inference, this process must be repeated for every newly generated token, resulting in significant redundant computation. To address this, a common optimization technique called

Key-Value caching

Key-Value Cache

As the name suggests, Key-Value (KV) caching is a technique designed to accelerate autoregressive generation by storing and reusing previously computed keys and values, instead of recalculating them at each decoding step.

It’s important to note that

caching is typically used only during inference, as training requires processing the entire input sequence in parallel making caching unnecessary.

cache is commonly implemented as a rolling buffer. At each decoding step, only the new query vector

is computed, while the previously stored keys

and values

are reused. Attention is then calculated using the new

and the cached

. Additionally, the new token’s

are appended to the cache for use in subsequent steps.

However, the speedup gained from using

cache comes with a significant memory cost. The cache typically scales with the product of batch size, sequence length, hidden size, and number of attention heads. As a result, it can become a memory bottleneck especially when working with large batch sizes or long sequences.

This limitation has led to the development of two techniques designed to reduce the memory footprint: Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).

Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)

The figure below compares the original Multi-Head Attention (MHA) with Grouped-Query Attention (GQA) [10] and Multi-Query Attention (MQA) [9].

Press enter or click to view image in full size Figure 3. MHA [6], GQA [10] AND MQA [9]. Image from [10].

The core idea behind MQA is to share a single key and a single value head across all query heads. This drastically reduces memory usage, especially during inference, but can come at the cost of reduced attention accuracy due to limited expressiveness.

Grouped-Query Attention (GQA) can be viewed as a middle ground between MHA and MQA. In GQA, a single pair of key and value heads is shared among a group of query heads, rather than across all queries as in MQA. While this approach offers a better trade-off between memory efficiency and performance, it still tends to yield inferior results compared to full MHA due to reduced attention flexibility.

In the following sections, we’ll explore how Multi-head Latent Attention (MLA) strikes a balance between memory efficiency and modeling accuracy addressing the limitations of both MQA and GQA.

RoPE (Rotary Positional Embeddings)

## One final piece of background to cover is

[11], which encodes positional information directly into the attention mechanism by applying sinusoidal rotations to the query and key vectors in multi-head attention.

More specifically,

applies a position-dependent rotation matrix to the query and key vectors at each token. While it uses sine and cosine functions as its basis similar to traditional positional encodings — it applies them in a unique rotational manner to preserve relative positional information within the attention mechanism.

To understand what makes RoPE position-dependent, let’s consider a toy example with an embedding vector containing just four elements: (x_1, x_2, x_3, x_4).

To apply RoPE, we firstly group consecutive dimensions into pairs:

(x_1, x_2) -> position 1

(x_3, x_4) -> position 2

Then, we apply a rotation matrix to rotate each pair:

Figure 4. Illustration of the rotation matrix applied to a pair of tokens. Image by author.

Here, θ = θ(p) = p ⋅ θ₀, where θ₀ is the base frequency. In our 4-dimensional toy example, this means the pair (x₁, x₂) is rotated by θ₀, while (x₃, x₄) is rotated by 2 ⋅ θ₀.

This is why the rotation matrix is referred to as position-dependent: at each position (or for each pair), a distinct rotation matrix is applied, with the rotation angle determined by the position itself.

RoPE is widely adopted in modern LLMs for its efficiency in encoding long sequences. However, as evident from the formula above, it introduces position dependence in both the query and key vectors, which creates certain incompatibilities when used with MLA.

Multi-head Latent Attention

Now, we can finally move on to Multi-head Latent Attention (MLA). In this section, we’ll begin by outlining the high-level intuition behind MLA, then explore why modifications to RoPE are necessary. We’ll conclude with the detailed MLA algorithm and a look at its performance.

MLA: High-level Idea

The basic idea behind MLA is to compress the attention input h_t into a low-dimensional latent vector with dimension

is much smaller than the original size of

(n_h · d_h)

. Later, when attention needs to be computed, this latent vector is projected back into the high-dimensional space to reconstruct the keys and values. This means we only need to store the compact latent vector, which significantly reduces memory usage.

This process can be more formally described with the following equations, where

is the latent vector.

is the compression matrix that maps

from dimension

(n_h · d_h)

(here, D in the superscript stands for ‘down-projection’, indicating dimensionality reduction), while

are up-projection matrices that map the shared latent vector back to the high-dimensional space to recover keys and values.

Similarly, we can also project the queries into a latent, low-dimensional vector and later map it back to the original high-dimensional space:

## Why Decoupled RoPE is Needed

As mentioned earlier, RoPE is a common choice for training generative models to handle long sequences. However, if we apply the MLA strategy directly, it becomes incompatible with RoPE.

To understand this more clearly, let’s look at what happens when we compute attention using Eqn. (7): when q^T is multiplied with k, the matrices W^Q and W^{UK} appear in the middle. Their combination effectively results in a single mapping from d_c to d.

In the original paper [3], the authors describe this by saying that W^{UK} can be ‘absorbed’ into W^Q. As a result, there’s no need to store W^{UK} in the cache, which further reduces memory usage.

As we discussed in the background section, the rotation matrix used in RoPE is position-dependent meaning each position has its own unique rotation matrix. Because of this, W^{UK} can no longer be absorbed into W^Q.

To resolve this conflict, the authors propose a technique called ‘decoupled RoPE’. It introduces additional query vectors along with a shared key vector, and applies RoPE only to these new vectors. The original keys are kept isolated from the rotation process, effectively decoupling positional encoding from the main attention computation.

The full process of Multi-head Latent Attention (MLA) is summarized below, with equation numbers referenced from Appendix C in [3]:

Press enter or click to view image in full size Figure 5. MLA process. Image edited by author based on equations in [3].

Eqn. (37) to (40) describe how to process query tokens.

Eqn. (41) and (42) describe how to process key tokens.

Eqn. (43) and (44) describe how to use the additional shared key for RoPE, be aware that the output of (42) is

not involved in RoPE

Eqn. (45) describes how to process value tokens.

In this process, only the blue boxed variables need to be cached. The workflow is illustrated more clearly in the flowchart below:

Press enter or click to view image in full size Figure 6. Flowchart of MLA. Image from [3].

## Performance of MLA

The table below compares the per-token KV cache size and modeling capacity across MHA, GQA, MQA, and MLA highlighting how MLA achieves a more effective balance between memory efficiency and representational power.

Interestingly, MLA’s modeling capacity even surpasses that of the original MHA, despite using significantly less memory.

Press enter or click to view image in full size Table 1 from [3].

More specifically, the table below presents the performance of MHA, GQA, and MQA on 7B-scale models, where MHA clearly outperforms both MQA and GQA by a significant margin.

Press enter or click to view image in full size Table 8 from [3].

The authors of [3] also provide a comparative analysis between MHA and MLA. As summarized in the table below, MLA consistently outperforms MHA across various benchmarks.

Press enter or click to view image in full size Table 9 in [3].

https://www.deepseek.com/

DeepSeek-V3 Technical Report

https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf

DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

https://arxiv.org/abs/2405.04434

DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

https://arxiv.org/abs/2401.06066

Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

https://arxiv.org/abs/2408.15664

## Attention Is All You Need

https://arxiv.org/abs/1706.03762

## Generating Wikipedia by Summarizing Long Sequences

https://arxiv.org/pdf/1801.10198

Improving Language Understanding by Generative Pre-Training

https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

Fast Transformer Decoding: One Write-Head is All You Need

https://arxiv.org/pdf/1911.02150

GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

https://arxiv.org/abs/2305.13245

RoFormer: Enhanced Transformer with Rotary Position Embedding

https://arxiv.org/abs/2104.09864

## Deep Learning

https://arxiv.org/abs/2104.09864

https://arxiv.org/abs/2104.09864

Deepseek V3

https://arxiv.org/abs/2104.09864

## Large Language Models

https://arxiv.org/abs/2104.09864

## Thoughts And Ideas

https://arxiv.org/abs/2104.09864

## Published in  Towards AI

89K followers

https://arxiv.org/abs/2104.09864

Last published  2 hours ago

https://pub.towardsai.net/deepseek-ocr-a-picture-is-worth-a-thousand-words-e2a8b9d74c7f?source=post_page---post_publication_info--bac648681926---------------------------------------

Making AI accessible to 100K+ learners. Find the most practical, hands-on and comprehensive AI Engineering and AI for Work certifications at

academy.towardsai.net

http://academy.towardsai.net

- we have pathways for any experience level. Monthly cohorts still open — use COHORT10 for 10% off!
## Written by  Nehdiii

227 followers

http://academy.towardsai.net

816 following

https://medium.com/@tahamustapha.nehdi/following?source=post_page---post_author_info--bac648681926---------------------------------------

MSc student at ÉTS Montréal and researcher at LIVIA Lab. working in Computer Vision, Efficient Deep Learning. Expert in PyTorch, CUDA Programming since 2021.

## No responses yet

## Text to speech

