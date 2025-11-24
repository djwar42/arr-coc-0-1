---
sourceFile: "DeepSeek V3 - ktiml"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:14.108Z"
---

# DeepSeek V3 - ktiml

495bb00f-d4b6-43bb-bec0-63045ffedbc6

DeepSeek V3 - ktiml

8d0ec43e-7536-4e40-93b7-5b2ab3964d32

https://ktiml.mff.cuni.cz/~bartak/ui_seminar/talks/DeepSeekV3_clean_Al_Ali.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOrCqgnMhZEiVXt2gJLy82sl4AYesHDOaZBPLBPYAlm1sVVpgdN9OMfJRnQ2qqtlUA3CL0mBoQuQo8kwkifXJk-a1WOGlSDNGMnakomIUPmNrlUqq2fcf9kCEK7mDclQ9nwbQgFG3g=w680-h516-v0

4444fe61-5169-4e41-a0b9-bf01c88f5b7c

DeepSeek V3

Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv preprint

arXiv:2412.19437 (2024).

## Presented by Adnan Al Ali

Mixture-of-Experts language model

671 B parameters, 37 B activated for each token

Cost-effective training and efficient inference

New state-of-the-art reached on certain benchmarks

Together with DeepSeek R1 strongly impacted the LLM/tech

## Related Work

What lead to DeepSeek V3?

## The Transformer

Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

## What the Transformer introduced

Originally an architecture for machine translation (MT)

Replaced the RNNs and CNNs (popular in NLP at the time) by

attention mechanism alone1:

## RNNs are difficult to parallelize

## CNNs struggle with long distance relationships

Encoder-decoder architecture

1 The attention mechanism had been used before, in combination with other modules

## How the Architecture Works

The input is tokenized into sub-word tokens from a fixed-size vocabulary and embedded into a vector ∈ ℝ𝑑model

Positional encodings are added (attention mechanism is position-unaware by default)

The encoder calculates a contextualized vector representation for each input token ∈ ℝ𝑑model

The decoder starts with an empty sequence and uses its previous outputs (autoregressively on inference) and the outputs of the encoder to generate the next token

In the decoder, only tokens can attend to earlier tokens only

https://lh3.googleusercontent.com/notebooklm/AG60hOoInr2BGMqA0mVR7xk0TlmpjAD21VlriYyb4BCVFRo58okJM5Qo0szmt-ozseNYJcmfYE787qEv9vw7Nw7-q8gBzj7ZX2HIOSL26rNL7N0FjtsdtMR9RVkdA-xZt5GqPQZIw9VXsA=w806-h1153-v0

304bcd6c-ca9e-4239-bea6-27890ab230bf

## Encoder Decoder

Source: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information

Scaled Dot-Product Attention

The vector token representations are linearly transformed into 3 matrices: Queries, Keys, and Values

“Compatibility” between the Keys and Queries:

softmax 𝑄𝐾𝑇

𝑑𝑘𝑒𝑦𝑠

## Compatibility is used to weight the values as softmax

𝑑𝑘𝑒𝑦𝑠

One attention layer contains ℎ such attention heads — the outputs are concatenated and linearly transformed back to 𝑑𝑚𝑜𝑑𝑒𝑙

Quiz question: what was the first decoder-only model?

Decoder-only Model

Liu, Peter J., et al. "Generating wikipedia by summarizing long sequences." arXiv preprint arXiv:1801.10198 (2018).

Multi-document Summarization

Task: given a collection of source texts, generate a Wikipedia-style summary

Authors drop the encoder part entirely, instead feed the input tokens directly into the decoder as sequences:

#Source1 lorem ipsum … #Source2 dolor sit … [SEP] #Wikipedia amet consectetur …

During training, predicting all tokens, including the sources

On inference, the sources are given as if they were already

https://lh3.googleusercontent.com/notebooklm/AG60hOoHNGyojPWnYNsQ4a6bdBvYR8Y1vTPd_gsKMbl7bZcOp-W42xcbdlbnoOGu-e56HcxSCj2NSi_-tZ8ppNb3Tne7P3tXCmAZTAYGJXN4CopfqkJNZRJX1f3cIFx1Fe1VjS-uehfz=w806-h1153-v0

70b9ba06-ab75-419f-8f8c-20d64e64fd44

Source: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information

## Mixture of Experts

Dai, Damai, et al. "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models." arXiv preprint arXiv:2401.06066 (2024).

## Mixture of Experts

Feed-forward (FFN) layers constitute two-thirds of a transformer model's parameters and store factual information[1]

The aim is to substitute them with a MoE layer — a set of 𝑵 smaller FFN layers of which only a subset of 𝑲 is used for each token

Output from the Self-Att layer for token 𝑡: 𝑢𝑡

Token-to-expert affinity: 𝑠𝑖,𝑡 = sigmoid 𝑢𝑡 𝑇𝑐𝑖

Top K experts with the highest 𝑠𝑖,𝑡  are considered:

𝑗∈TopK 𝑠𝑖,𝑡

𝛼𝑡 FFN𝑗 𝑢𝑡 + 𝑢𝑡

[1]Geva, Mor, et al. "Transformer feed-forward layers are key-value memories." arXiv preprint arXiv:2012.14913 (2020).

residual connection

learned “expert centroid”

normalizing factor

MoE Challenge: Knowledge Hybridity

Previous architectures had a small number of experts (8 or 16) which had to cover diverse knowledge → hard to utilize at once

Solution: fine-grained expert segmentation:

Segment each expert into 𝑚 equally sized experts ( 1

of the original size)

Increase 𝐾′ to 𝑚𝐾

Why it works — combinatorial explosion/flexibility:

For 𝑁 = 16, 𝐾 = 2 the number of expert combinations is 16

Fine-grained by m = 4: 64

=  4, 426, 165, 368

MoE Challenge: Knowledge Redundancy

Some knowledge is required for all/most tokens and under the conventional architecture, all experts have to learn it

Solution: shared expert isolation:

A small number (𝐾𝑠) of experts is activated for each token

The remaining 𝐾 − 𝐾𝑠 experts are selected excluding the shared ones

https://lh3.googleusercontent.com/notebooklm/AG60hOphH-IIjH6NnZLmQPBj_B4G4r_Fju4Izyu66Pj9pVLPcDblWeGokbwL7_HSWyNttvonGWsLR9sismKhzBOmhEXPG9aPbpyteYnnogJnICmL8NmPKrApPCYDjREvT8mppCpmBBcEMQ=w2444-h1243-v0

2c6e8843-c08e-450c-a310-1a50504946ae

Source: Dai, Damai, et al. "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models." arXiv preprint arXiv:2401.06066 (2024).

MoE challenge: routing collapse

Automatically learned expert routing may lead to repetitive selection of a few experts regardless of the token

Solution: expert-level balance loss:

Per-token average expert affinity: 𝑃𝑖 =

𝑇 𝑠𝑖,𝑡

Per-token average expert utilization: 𝑓𝑖 =

𝑇 𝟙 Token 𝑡 selects Expert 𝑖

Loss function: ℒExpBal = 𝜂1 σ𝑖 𝑓𝑖𝑃𝑖

indicator function

Quiz question: what’s the goal of the DeepSeekMoE architecture?

a) Adding more knowledge to the model b) Saving GPU memory c) Decreasing the number of computations d) Explicitly assigning expertise to parts of the model

Quiz question: what’s the goal of the DeepSeekMoE architecture?

a) Adding more knowledge to the model b) Saving GPU memory c) Decreasing the number of computations d) Explicitly assigning expertise to parts of the model

DeepSeek V2

Liu, Aixin, et al. "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model." arXiv preprint arXiv:2405.04434 (2024).

Multi-Head Latent Attention

In the original Transformer attention, the heavy Key-Value cache slows down the inference: 2 #heads 𝑑(#blocks) values per token

Solution: low-rank key-value joint compression:

## Before applying the linear transformation into the Keys and Values for

each head, the attention input (ℎ𝑡) is transformed into a low-dimensional compressed space: 𝑐𝑡

𝐾𝑉 = 𝑊𝐷𝐾𝑉ℎ𝑡 ∈ ℝ𝑑𝑐

On inference, only the 𝑐𝑡

𝐾𝑉  is cached

Similarly, the Queries are computed from a compressed vector

Positional embedding (RoPE[1]) are computed before the compression

[1]Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.

https://lh3.googleusercontent.com/notebooklm/AG60hOrr8kXJOMBu3HfxDK4N2aqEfWhDEEj5LAOCLNi5cDa9eStK_WdKHrdh7h80bOMnvnlpTnAuEUSzJnejIqhGqxt-CagkOQckOHF5DWSC_tmVFOIGkO90Wfpy5CS1kEUemAimX_sXNA=w2122-h1434-v0

b24f926f-4b46-4adb-b8df-7c6b2c24801f

Source: Liu, Aixin, et al. "Deepseek-v2: A strong,

economical, and efficient mixture-of-experts language

model." arXiv preprint

Quiz question: why aren’t the Queries cached?

a) We don’t need them in the future computations b) They are easier to compute c) They would take up too much memory d) They change dynamically and cannot be cached

Quiz question: why aren’t the Queries cached?

a) We don’t need them in the future computations b) They are easier to compute c) They would take up too much memory d) They change dynamically and cannot be cached

Device-Limited Routing

Individual experts are often loaded on different GPUs (expert parallelism)

## Communication between the GPUs is costly

Solution: limiting the number of GPUs per token to 𝑴

Implementation:

Select the top 𝑀 devices based on the affinity and all the experts on those devices

Use the top 𝐾 − 𝐾𝑠  experts from the selection.

For 𝑀 ≥ 3 results are comparable to unrestricted selection

DeepSeek V3

Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv preprint arXiv:2412.19437 (2024).

Chapter 2: Architecture

## Architecture

Multi-head latent attention (as described in DeepSeek V2)

Mixture of 256 experts (as described in DeepSeekMoE and V2)

Two approaches to the experts load balancing:

### 1. Auxiliary-loss-free load balancing:

When computing the top (𝐾 − 𝐾𝑠) experts for a token 𝑡, a bias 𝑏𝑖  is added to the

affinity 𝑠𝑖,𝑡

The bias is dynamically in-/decreased by a hyperparameter 𝛾 during the training to account for under-/overloaded experts

### 2. Complementary expert-level (auxiliary) balance loss (as described in DeepSeekMoE), with a small learning rate to preserve the performance[1]

[1]Wang, Lean, et al. "Auxiliary-loss-free load balancing strategy for mixture-of-experts." arXiv preprint arXiv:2408.15664 (2024).

Additional Training Objective: MTP

Multi-token prediction (MTP) predicts 𝐷 + 1 future tokens at each step (instead of one)

Aim: densify the training process, enable the model to pre-plan for future token predictions

MTP procedure:

## Run the main model and obtain representations for each token

MTP modules are sequential: each taking the representations from the previous

module concatenated with the true next token embeddings

Pass through a linear projection, a Transformers layer and the output head

## The embedding layer and the output head are shared

MTP loss: ℒMTP =

𝑘  where ℒMTP

𝑘  is the cross-entropy loss

https://lh3.googleusercontent.com/notebooklm/AG60hOoQ3Ui7jSLK88hJEN74eSYKUis_awb0gsf14POwa1ebw8N8wahgFuV-qevi2FYTzt0oaJEy5vsR4-eT6kAZOdwx0jTNKpJGBwEvSTNcnzct8P-LzbxsnfPECKt3EpIZEzhPeoFd_g=w2495-h1173-v0

a799c92e-38f5-45f8-a806-c2a4c28a47e4

Source:Liu, Aixin, et al. "Deepseek-v3 technical

Chapter 3: Infrasctructures

## Training Infrastructure

Trained on a cluster of 2048 NVIDIA H800 GPUs (80 GB VRAM), 8 GPUs per node

Expert parallelism spanning 8 nodes

## This introduces communication overhead similar to the computation time

Solution: DualPipe — overlapping communication and computation

Mixed-Precision Training

Quantization to FP8 increases effectivity but is limited by outliers

Most of the core computation (such as matric multiplication) is

done in FP8

## Original precision is preserved in some modules

Fine-grained quantization:

Standard practice: scale the maximum absolute value from the samples to the maximum representable FP8 → one outlier ruins the accuracy

Fine-grained approach: split the sequence into blocks of size 𝑵𝑪; each block has its own scale based on its maximum absolute value

Inference: Pre-Filling (stage I)

During pre-filling, the user’s prompt is processed and cached items are pre-computed

Minimum deployment unit: 4x8 GPUs

Parallelism strategies for the attention modules:

4-way Tensor Parallelism (TP4) (= weights distributed over 4 devices)

Sequence Parallelism (SP) (= sequence is split for some operations)

8-way Data Parallelism (DP8) (= 8 independent copies of the sub-model)

Parallelism strategies for the MoE modules:

32-way Expert Parallelism (EP32) (= experts distributed over 32 devices)

32 redundant experts are maintained, dynamically changed.

https://lh3.googleusercontent.com/notebooklm/AG60hOobQVxNGY0wTLaEkkLhp7sz58xttTYRDsqU-RwWDdUg9f3vkz0b_JIj1LssPE6c3yMTFRLOmN_RVIwOuwfm4wWqQDco-e8nveiGb7HjtE4NAZoDtHNtQ2upUiG5WKQSbMLDkydKRQ=w158-h163-v0

5e55ea26-1158-42bd-8dfd-38a79a3c76a6

https://lh3.googleusercontent.com/notebooklm/AG60hOo10vwcFPPKRtcUND0itZ7dpS62QOaWYgwEzSWAiz9iACW8bjkn_UcnO5FaKX9VoOTnpF-pR8w-SVHWBk1nI7lqQnidPZb3JyEMU488FrJyryM2MbmXHDq_sB93RdZ7Oi0CAgShwQ=w158-h163-v0

20afbad2-c5eb-4d0e-8e85-4ab46901092d

https://lh3.googleusercontent.com/notebooklm/AG60hOpnXswtaWe6BgrabUfX2vWgFUX1-pK0FxZd4lmHYH8Ei4t1w61HvfgPNEK3xHoumYr-e0lMCAWcIb1Lg0J0bJlzsQ5DN6oGZRcHdxNB2Y33q60Rk0Z1mw0SSrX8FpGPluUxdXpv4w=w158-h163-v0

ef534bd2-15d9-400a-815d-8839e4d6eb55

https://lh3.googleusercontent.com/notebooklm/AG60hOq3PaJfiIpW7xR9R_nIXhDyfpKA6_xy5avzUs2vkpcnEmjJo9umxTSCzmEL5-Mxou4DVi-bpJmrqGCLBuoQvUMYb-MwViavMAQ_QGR-kqItV3Wt1g9J0Zx7ErZyGSHF76z5FtLtBw=w158-h163-v0

89d400aa-d2ab-46cb-865f-96359bdc7d77

https://lh3.googleusercontent.com/notebooklm/AG60hOpx5fkEC8lqtiwDSgctLdUL_nXZUKiRXK1m1W8-EhXIverlOTtPaXBNFQG9M9j4QFerFKtwNYP-EWaZljD6DzF9kMbSutG0kpmL4HLdd5CR7PS1k54PSMeTTimDjEix8yi5wp70Ig=w158-h163-v0

2288f0e9-fa04-4da3-ad5c-9b57c182c1c5

Quiz question: what is being cached for each input token?

key projections for each head 𝑘𝑡,𝑖

queries projections each head 𝑞𝑡,𝑖

values projections each head 𝑣𝑡,𝑖

compressed latent vector 𝑐𝑡

𝐾𝑉  for keys and values

compressed latent vector 𝑐𝑡

𝑄  for queries

https://lh3.googleusercontent.com/notebooklm/AG60hOr_sg7J3qw9H7IxWt3NAOC-kzy3Z4RvNXIhfcKKRl6jmUSHbT6_br41WrmX9gFymdjNSlWXR1c6n51Y0o368tM4eEiP57zQxpJN0aBmQp4HVzEBHWRmPYKYsv42IPj5yjxVfzNUbQ=w158-h163-v0

79e60279-d622-4554-b54e-a8f7837b4981

https://lh3.googleusercontent.com/notebooklm/AG60hOo26yyZdu5vwzPD9PsXVyrrGb_UNAuQPIJtljxb9eDZn_vwhabemhLgH2d5Y2_LLPL3ii3p40A33aZ-M74p0kK78Z6YrDcWs_XgBIO4SWMqw9lqFC4AJ7P2wO7zFbK5Oe1ajAEOug=w158-h163-v0

8e0416fa-8075-4019-bf3a-4346840192a5

https://lh3.googleusercontent.com/notebooklm/AG60hOpk2PsLhj8z8mx6u58wvOQkQI6dnbyx3sCTxveIB7kp8DYImJTxy8y2SR6IHA5aqTWcT4tK75zivyQdb9Vot8A95xgjqxU57nbktEqFKZLfpaI1xoWEe1yy5B48oImXe-wRNdoJWQ=w158-h163-v0

8c6d9658-ddd7-448c-9fab-22d1cb838334

https://lh3.googleusercontent.com/notebooklm/AG60hOoBYr-Xz7iXRkt3TffAJPu5aCmSvK9xum4cWLujp4Sir-eFWq6Tc6YQq-MA_dtEr4Qn0vlN2AFB4NIZYtcQ6tih6rgAYtsU5eNLi6HhSFc45iI-g8oukXtyXusMd-eHybM5YhZ88g=w158-h163-v0

235df900-d272-445e-a053-10d078255f0b

Quiz question: what is being cached for each input token?

key projections for each head 𝑘𝑡,𝑖

queries projections each head 𝑞𝑡,𝑖

values projections each head 𝑣𝑡,𝑖

🗹 compressed latent vector 𝒄𝒕 𝑲𝑽 for keys and values

compressed latent vector 𝑐𝑡

𝑄  for queries

Inference: Decoding (stage II)

During decoding, the model predicts the tokens autoregressively

Minimum deployment unit: 40x8 = 320 GPUs

Attention parallelism: TP4 + SP + DP80

Parallelism strategies for the MoE modules:

320-way Expert Parallelism (EP320)

## Each GPU hosts one expert

64 GPUs host the shared and redundant experts

Chapter 4: Pre-Training

## Training Data

14.8 T of multilingual diverse tokens, with a large portion of math and code

Byte-Pair Encoding tokenization

Data augmentation: FIM with the rate of 0.1

Text is split into three parts: 𝑓𝑝𝑟𝑒𝑓𝑖𝑥, 𝑓𝑚𝑖𝑑𝑑𝑙𝑒 , 𝑓𝑠𝑢𝑓𝑓𝑖𝑥 and transformed: [BEGIN] 𝑓𝑝𝑟𝑒𝑓𝑖𝑥 [HOLE] 𝑓𝑠𝑢𝑓𝑓𝑖𝑥 [END] 𝑓𝑚𝑖𝑑𝑑𝑙𝑒 [EOS]

Hyper-Parameters

Model parameters:

61 Transformer layers

128 attention heads

attention dim: 128

KV compressed dim: 512

Q compressed dim: 1536

hidden dim: 7168

first 3 FFNs kept dense

256 routed experts

8 of them activated per

max 4 nodes per tok

1 shared expert

expert hidden dim: 2048

1 extra MTP token

## Training parameters

AdamW with 𝛽1 = 0.9, 𝛽2 = 0.95, 𝑊𝐷 = 0.1

Context window: 4096 (later extended using YaRN[1])

LR linear increase from 0 to 2.2 × 10−4, constant for 10 T tokens, then decreased to 7.3 × 10−6

Gradient clipped to 1.0

BS increased from 3072 to 15360 over 469 B tokens

Auxiliary-free loss update 𝛾 = 0.001, then 0 for the last 500 B tokens

Complementary balance lost weight: 𝜂1 = 0.0001

MTP loss weight 𝜆 = 0.3 for the first 10 T tokens, then 𝜆 = 0.1

[1]Peng, Bowen, et al. "Yarn: Efficient context window extension of large language models." arXiv preprint arXiv:2309.00071 (2023).

https://lh3.googleusercontent.com/notebooklm/AG60hOpo5OkLJ9otjyhOSCo-EaJw8T6BasfVm_T57NGJ-t613vMUvmqQFlWKw9PnORziaXZf58AqUrqEsh8vlJvudIuD_a_exyKJfpG8UvtnCQBNZtNYmT_6yVEvwsaKtTZ2YFgr7Wdm2A=w1787-h1003-v0

bf2909f1-32b8-47a6-af73-a44850546a4d

Source:Liu, Aixin, et al. "Deepseek-

v3 technical report." arXiv

preprint arXiv:2412.19437

https://lh3.googleusercontent.com/notebooklm/AG60hOqHXacgxT6snqpdbnIlSwEy22hKRxbv41_s9HrDzk4A1Mn9DP6IE71dFrUc2cbQkHsRrOY1cMl-6pZaDKzRGY6cVjW4S7Emfh4AGzatB_d0sS644xPclIj0g-L7egMxVyx3NA-f0Q=w2224-h988-v0

d5448a6b-fae2-4a7e-9a2d-839ceb3251ba

https://lh3.googleusercontent.com/notebooklm/AG60hOrVlseGfyNg2guiXk2hzJ2WSxr1aj6oDnK_QNimNO70cvCZZpwMvk4J27_nwrBi01HTRKaoqpx_XM2_jvCyBGCXs4uXehw2F8GrBGeMOc4wIEMjw8-GSE-LJ0HgzM0hpbHI7o2X=w2232-h352-v0

cadf0f65-bd05-423e-b963-a8e48117b50d

Source: Liu, Aixin, et al. "Deepseek-

v3 technical report." arXiv

preprint arXiv:2412.19437

Chapter 5: Post-Training

## Reasoning Data Generation

Reasoning data partially based on a DeepSeek-V2.5-based R1 prototype

Problem: R1 models overthink and generate very long sequences

Solution: create an expert model for data generation using SFT and RL

pipeline (different for coding, math, general reasoning…)

SFT is done on two kinds of samples: problem, original response  and

system prompt, problem, R1 response ; the system prompt guides the model through the reasoning

During the RL, system prompt is removed, and responses are sampled at a high temperature

Final result: concise answers retaining R1 thinking patterns

Supervised Fine-Tuning (SFT)

Instruction-tuning datasets including 1.5M instances

Largely generated:

Non-reasoning data responses generated by DeepSeek-V2.5 and verified by human annotators

Reasoning data generated by an expert model (see last slide)

Hyperparameters:

cosine LR decay from 5 × 10−6 to 1 × 10−6

Quiz question: what is reinforcement learning?

Reinforcement Learning (RL)

Two types of reward models (RM):

Rule-based: for questions that can be objectively validates (e. g. correct

solution )

Model-based: for questions with a free-form ground-truth answers, a

dedicated ML model is trained — based on DeepSeek-V3 SFT

Group relative policy optimization (GRPO) strategy:

Omits the critic (value) model

## Group scores used instead

For a question 𝑞, outputs 𝑜1, 𝑜2, ⋯ , 𝑜𝐺  are sampled from the old policy

model 𝜋𝜃𝑜𝑙𝑑  and 𝜋𝜃  optimized by maximizing the objective…

https://lh3.googleusercontent.com/notebooklm/AG60hOqaK8y3nGdSZIX5tu7Vvr-nklLF2SotHNXARmHE-Rr0EB7BTx9PzhVo0iOBcHCXFFHB4Uas_RWdejAECrtY8AIzeBoXRPerqQ_bXzX9bzcJpWhVE0KLLMSegXcIvaDPR8Yd7RI6hw=w1435-h370-v0

c64f287e-e117-40e1-ad99-40ae3c2b86e0

https://lh3.googleusercontent.com/notebooklm/AG60hOo2AsIyvzwVeKoHZAITP4xwg4WM0kNPp0ZdZTE8mEiPA5hB6I8K4e0JIDnxgbD07eFMZ_UgWFIJXlKm2GIN2TMmp3zJJOkZkVVfnPQKIk1N6ZGde09fAOQ8dKa4hW6tEsduXJDTLg=w212-h301-v0

a354fc82-a024-4402-8d6b-aa1452792fe4

Group Relative Policy Optimization (GRPO)

Maximizing the objective: 𝒥𝐺𝑅𝑃𝑂 = 𝔼 𝑞~𝑃 𝑄 , 𝑜𝑖 𝑖=1

𝐺 ~𝜋𝜃𝑜𝑙𝑑 𝑂|𝑞

min 𝜋𝜃 𝑜𝑖|𝑞

𝜋𝜃𝑜𝑙𝑑 𝑜𝑖|𝑞

𝐴𝑖 , clip 𝜋𝜃 𝑜𝑖|𝑞

𝜋𝜃𝑜𝑙𝑑 𝑜𝑖|𝑞

, 1 − 𝜀, 1 + 𝜀 𝐴𝑖 − 𝛽D𝐾𝐿 𝜋𝜃||𝜋𝑟𝑒𝑓

where advantage 𝐴𝑖 =

𝑟𝑖−𝑚𝑒𝑎𝑛 𝑟1,𝑟2,⋯,𝑟𝐺

𝑠𝑡𝑑 𝑟1,𝑟2,⋯,𝑟𝐺

[1]Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998. Figure source: Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

## Compare this to the advantage from the REINFORCE with

baseline[1] algorithm: 𝑎𝜋 𝑠, 𝑎 = 𝑞𝜋 𝑠, 𝑎 − 𝑣𝜋(𝑠)

https://lh3.googleusercontent.com/notebooklm/AG60hOrpYR0LFdm_yD5Mz_CCnh4UyezQRZhe3MiISooCrlytgFC7miyc0Cquo3PZ6QxEqkmT9E_7P3lJa0MlMzy_ysbiL2_LU-VJcvCwQ29_rQOy-cxOKJ9xoM_VlSvDLcf_Sauuon_b=w1648-h1200-v0

5391f0f4-61cf-4e7e-845b-266adde204d1

Source: Liu, Aixin, et al. "Deepseek-

v3 technical report." arXiv

preprint arXiv:2412.19437

DeepSeek-V3

DeepSeekMoEDeepSeekV2

DeepSeekV2.5

R1 prototype

## DeepSeekMath

Deepseek-coder-v2:

MLA MoE arch. GRPO

## Reasoning data Reward model

DeepSeek-R1

Instr. data

Chapter 6: Conclusion, Limitations, and Future Directions

## Conclusion

DeepSeek-V3, a large MoE model with 671B parameters, 37B activated parameters

MLA, DeepSeekMoe architecture, auxiliary-loss-free strategy, multi-token prediction training objective, FP8 training

Distilled reasoning from the R1 prototype

Strongest open-source model at the time, comparable results to

GPT-4o and Claude-3.5-Sonnet

2.788M H800 GPU hours for full training (=57 days with 2048

## Limitations

Large deployment unit recommended (inaccessible to smaller teams)

Generation speed is still limited (more advanced hardware anticipated)

## Future Directions

Consistently adhere to the open-source philosophy and longtermism

Aiming toward the artificial general intelligence (AGI)

Study and refine the architectures, possibly beyond Transformer

Improve the quality and quantity of the training data, explore

other sources of training signals

Explore the deep-thinking capabilities

Explore a more comprehensive way of evaluation, instead of

optimizing for a fixed set of benchmarks

https://lh3.googleusercontent.com/notebooklm/AG60hOp_aZivUTrOSC0mxDegVSSkn4j_HfyeXlefnPSB4RNVTJL8jBcM6IzQdptanuMJViHzww48EKdNThjt5D2zjo6FsWNd8JBEz-wlueVpd_Zti4dCdk1qmM8k5KSjgMtAvc8ZR5x9Nw=w1536-h846-v0

ad2bf7cd-d019-456e-8fd3-d7d57e29565a

## Thank you for your attention

