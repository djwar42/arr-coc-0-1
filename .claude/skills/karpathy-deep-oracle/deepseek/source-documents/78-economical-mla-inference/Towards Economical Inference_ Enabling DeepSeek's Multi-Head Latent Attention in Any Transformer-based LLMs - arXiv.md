---
sourceFile: "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:39.119Z"
---

# Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs - arXiv

dd7a1b8a-f607-4ff1-a2e6-0337848b120a

Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs - arXiv

b0dbc810-3b58-4844-855d-5ef0d5da3b96

https://arxiv.org/html/2502.14837v1

Towards Economical Inference: Enabling  DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs

♠ ♠ \spadesuit ♠

♡ ♡ \heartsuit ♡

, Yuanbin Wu

♡ ♡ \heartsuit ♡

,  Qipeng Guo

♢ ♢ \diamondsuit ♢

, Lixing Shen

♣ ♣ \clubsuit ♣

, Zhan Chen

♣ ♣ \clubsuit ♣

, Xipeng Qiu

♠ ♠ \spadesuit ♠

♠ ♠ \spadesuit ♠

♠ ♠ \spadesuit ♠ \faEnvelope

♠ ♠ \spadesuit ♠

## Fudan University

♡ ♡ \heartsuit ♡

## East China Normal University

♣ ♣ \clubsuit ♣

## Hikvision Inc

♢ ♢ \diamondsuit ♢

## Shanghai Al Lab

{taoji, tgui}@fudan.edu.cn

mailto:taoji@fudan.edu.cn,%20tgui@fudan.edu.cn

{binguo@stu, ybwu@cs}.ecnu.edu.cn

mailto:taoji@fudan.edu.cn,%20tgui@fudan.edu.cn

Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA ( MHA2MLA ), which includes two key components: for  partial-RoPE , we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for  low-rank approximation , we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (3‰ to 6‰) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.

1  Our source code is publicly available at

https://github.com/JT-Ushio/MHA2MLA

https://github.com/JT-Ushio/MHA2MLA

Towards Economical Inference: Enabling  DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs

♠ ♠ \spadesuit ♠

♡ ♡ \heartsuit ♡

, Yuanbin Wu

♡ ♡ \heartsuit ♡

,   Qipeng Guo

♢ ♢ \diamondsuit ♢

, Lixing Shen

♣ ♣ \clubsuit ♣

, Zhan Chen

♣ ♣ \clubsuit ♣

, Xipeng Qiu

♠ ♠ \spadesuit ♠

♠ ♠ \spadesuit ♠

♠ ♠ \spadesuit ♠ \faEnvelope

♠ ♠ \spadesuit ♠

## Fudan University

♡ ♡ \heartsuit ♡

## East China Normal University

♣ ♣ \clubsuit ♣

## Hikvision Inc

♢ ♢ \diamondsuit ♢

## Shanghai Al Lab

{taoji, tgui}@fudan.edu.cn

mailto:taoji@fudan.edu.cn,%20tgui@fudan.edu.cn

{binguo@stu, ybwu@cs}.ecnu.edu.cn

mailto:taoji@fudan.edu.cn,%20tgui@fudan.edu.cn

1  Introduction

Figure 1:  The diagram illustrates the MHA, MLA, and our MHA2MLA. It can be seen that the “cached” part is fully aligned with MLA after MHA2MLA. The input to the attention module is also completely aligned with MLA (the  aligned region below ). Meanwhile, the parameters in MHA2MLA maximize the use of pre-trained parameters from MHA (the  aligned region above ).

The rapid advancement of large language models (LLMs) has significantly accelerated progress toward artificial general intelligence (AGI), with model capabilities scaling predictably with parameter counts  Kaplan et al. (

https://arxiv.org/html/2502.14837v1#bib.bib17

) . However, these gains come at a steep cost: escalating computational demands for training and degraded inference throughput, resulting in substantial energy consumption and carbon emissions  Strubell et al. (

https://arxiv.org/html/2502.14837v1#bib.bib27

As downstream tasks grow increasingly complex, long-context processing and computationally intensive inference have become central to LLM applications  An et al. (

https://arxiv.org/html/2502.14837v1#bib.bib2

) . A key bottleneck lies in the memory footprint of the Key-Value (KV) cache inherent to the Multi-Head Attention (MHA,

https://arxiv.org/html/2502.14837v1#bib.bib30

) mechanism, which scales linearly with sequence length and model size. To mitigate this, variants like Grouped-Query Attention (GQA,

https://arxiv.org/html/2502.14837v1#bib.bib1

) and Multi-Query Attention (MQA,

https://arxiv.org/html/2502.14837v1#bib.bib25

) have been explored. However, these methods reduce not only the KV cache size but also the number of parameters in the attention, leading to performance degradation. The DeepSeek introduces Multi-Head Latent Attention (MLA,

https://arxiv.org/html/2502.14837v1#bib.bib11

), an attention mechanism equipped with low-rank key-value joint compression. Empirically, MLA achieves superior performance compared with MHA, and meanwhile significantly reduces the KV cache during inference, thus boosting the inference efficiency.

A critical yet unexplored question arises:  Can LLMs originally well-trained for MHA be adapted to enabling MLA for inference?  The inherent architectural disparities between MHA and MLA render zero-shot transfer impractical, while the prohibitive cost of pretraining from scratch makes this transition both technically challenging and underexplored in existing research. To address this gap, we propose the first carefully designed MHA2MLA framework that maximizes parameter reuse from pre-trained MHA networks while aligning the KV cache storage and inference process with MLA’s paradigm (

https://arxiv.org/html/2502.14837v1#S1.F1

). Our framework features two pivotal technical innovations: partial rotary position embedding (partial RoPE) and low-rank approximation. The primary objective of MHA2MLA is to achieve data-efficient performance recovery - restoring architecture-induced capability degradation using minimal fine-tuning data.

The inherent incompatibility between MLA’s inference acceleration mechanism and RoPE necessitates architectural compromises. DeepSeek’s solution preserves PEs in limited dimensions while compressing others, requiring strategic removal of RoPE dimensions (converting them to NoPE) in MHA to achieve MLA alignment. While higher removal ratios enhance compression efficiency, they exacerbate performance degradation, creating an efficiency-capability trade-off. Through systematically exploring RoPE removal strategies, we identify that contribution-aware dimension selection (retaining top-k dimensions ranked by attention score impact) optimally balances these competing objectives. Although previous studies have investigated training partial-RoPE LLMs from scratch  Black et al. (

https://arxiv.org/html/2502.14837v1#bib.bib8

); Barbero et al. (

https://arxiv.org/html/2502.14837v1#bib.bib6

) , our work pioneers data-efficient fine-tuning for full-to-partial RoPE conversion in LLMs.

MLA reduces memory footprint by projecting keys and values into a low-rank latent representation space (stored in the KV cache). MHA2MLA can also apply low-rank approximation to the values and keys stripped of RoPE (NoPE dimensions). By performing Singular Value Decomposition (SVD) on the pre-trained parameter matrices  𝑾 v subscript 𝑾 𝑣 \bm{W}_{v} bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  and  𝑾 k subscript 𝑾 𝑘 \bm{W}_{k} bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT  corresponding to the NoPE subspaces, we compress these components into a latent space while maximizing the retention of knowledge learned by the original model.

Our main contributions are:

we introduce MHA2MLA, the first parameter-efficient fine-tuning framework that adapts pre-trained MHA-based LLMs to the MLA architecture using only 3‰ to 6‰ of training data without training from scratch.

we demonstrate that the MHA2MLA architecture can be integrated with KV-cache quantization to achieve more economical inference (up to a 96.87% reduction).

we conduct experiments across four model sizes (from 135M to 7B, covering both MHA and GQA), and detailed ablation studies to provide guidance and insights for MHA2MLA.

2  Preliminary

2.1  Multi-Head Attention (MHA)

Given an input sequence  { 𝒙 1 , … , 𝒙 l } ∈ ℝ l × d subscript 𝒙 1 … subscript 𝒙 𝑙 superscript ℝ 𝑙 𝑑 \{\bm{x}_{1},\dots,\bm{x}_{l}\}\in\mathbb{R}^{l\times d} { bold_italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , bold_italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT } ∈ blackboard_R start_POSTSUPERSCRIPT italic_l × italic_d end_POSTSUPERSCRIPT , standard MHA  Vaswani et al. (

https://arxiv.org/html/2502.14837v1#bib.bib30

)  projects each token  𝒙 i subscript 𝒙 𝑖 \bm{x}_{i} bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  into queries  𝒒 i ( h ) = 𝒙 i ⁢ 𝑾 q ( h ) superscript subscript 𝒒 𝑖 ℎ subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 ℎ \bm{q}_{i}^{(h)}=\bm{x}_{i}\bm{W}_{q}^{(h)} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , keys  𝒌 i ( h ) = 𝒙 i ⁢ 𝑾 k ( h ) superscript subscript 𝒌 𝑖 ℎ subscript 𝒙 𝑖 superscript subscript 𝑾 𝑘 ℎ \bm{k}_{i}^{(h)}=\bm{x}_{i}\bm{W}_{k}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , and values  𝒗 i ( h ) = 𝒙 i ⁢ 𝑾 v ( h ) superscript subscript 𝒗 𝑖 ℎ subscript 𝒙 𝑖 superscript subscript 𝑾 𝑣 ℎ \bm{v}_{i}^{(h)}=\bm{x}_{i}\bm{W}_{v}^{(h)} bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , where  𝑾 q ( h ) , 𝑾 k ( h ) , 𝑾 v ( h ) ∈ ℝ d × d h superscript subscript 𝑾 𝑞 ℎ superscript subscript 𝑾 𝑘 ℎ superscript subscript 𝑾 𝑣 ℎ superscript ℝ 𝑑 subscript 𝑑 ℎ \bm{W}_{q}^{(h)},\bm{W}_{k}^{(h)},\bm{W}_{v}^{(h)}\in\mathbb{R}^{d\times d_{h}} bold_italic_W start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  for each head  h ∈ { 1 , … , n h } ℎ 1 … subscript 𝑛 ℎ h\in\{1,\dots,n_{h}\} italic_h ∈ { 1 , … , italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT } . The Rotary positional encoding (RoPE,

https://arxiv.org/html/2502.14837v1#bib.bib28

) is applied to queries and keys (e.g.,  𝒒 i , rope ( h ) = RoPE ⁢ ( 𝒒 i ( h ) ) superscript subscript 𝒒 𝑖 rope ℎ RoPE superscript subscript 𝒒 𝑖 ℎ \bm{q}_{i,\text{rope}}^{(h)}=\text{RoPE}(\bm{q}_{i}^{(h)}) bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = RoPE ( bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ) ), followed by scaled dot-product attention

2  We ignore here the  1 d 1 𝑑 \frac{1}{\sqrt{d}} divide start_ARG 1 end_ARG start_ARG square-root start_ARG italic_d end_ARG end_ARG  scaling factor for ease of notation.  :

𝒐 i ( h ) = Softmax ⁢ ( 𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( h ) ⊤ ) ⁢ 𝒗 ≤ i ( h ) , superscript subscript 𝒐 𝑖 ℎ Softmax superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from ℎ top superscript subscript 𝒗 absent 𝑖 ℎ \displaystyle\bm{o}_{i}^{(h)}=\text{Softmax}\left(\bm{q}_{i,\text{rope}}^{(h)}% \bm{k}_{\leq i,\text{rope}}^{(h)\top}\right)\bm{v}_{\leq i}^{(h)}, bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = Softmax ( bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT ) bold_italic_v start_POSTSUBSCRIPT ≤ italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ,   MHA ⁢ ( 𝒙 i ) = [ 𝒐 i ( 1 ) , … , 𝒐 i ( n h ) ] ⁢ 𝑾 o , MHA subscript 𝒙 𝑖 superscript subscript 𝒐 𝑖 1 … superscript subscript 𝒐 𝑖 subscript 𝑛 ℎ subscript 𝑾 𝑜 \displaystyle\text{MHA}(\bm{x}_{i})=\left[\bm{o}_{i}^{(1)},\dots,\bm{o}_{i}^{(% n_{h})}\right]\bm{W}_{o}, MHA ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = [ bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT , … , bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT ] bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT ,   (1)

where  𝑾 o ∈ ℝ ( n h ⁢ d h ) × d subscript 𝑾 𝑜 superscript ℝ subscript 𝑛 ℎ subscript 𝑑 ℎ 𝑑 \bm{W}_{o}\in\mathbb{R}^{(n_{h}d_{h})\times d} bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT ( italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) × italic_d end_POSTSUPERSCRIPT  and  [ ⋅ , ⋅ ] ⋅ ⋅ [\cdot,\cdot] [ ⋅ , ⋅ ]  means vector concatenate. During autoregressive inference, MHA stores the KV cache  { 𝒌 rope ( h ) , 𝒗 ( h ) } h = 1 n h superscript subscript superscript subscript 𝒌 rope ℎ superscript 𝒗 ℎ ℎ 1 subscript 𝑛 ℎ \{\bm{k}_{\text{rope}}^{(h)},\bm{v}^{(h)}\}_{h=1}^{n_{h}} { bold_italic_k start_POSTSUBSCRIPT rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_v start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_h = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  of size  𝒪 ⁢ ( 2 ⁢ l ⁢ n h ⁢ d h ) 𝒪 2 𝑙 subscript 𝑛 ℎ subscript 𝑑 ℎ \mathcal{O}(2ln_{h}d_{h}) caligraphic_O ( 2 italic_l italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) , growing linearly with sequence length  l 𝑙 l italic_l , posing memory bottlenecks.

Grouped-Query Attention (GQA,

https://arxiv.org/html/2502.14837v1#bib.bib1

) shares keys/values across  n g subscript 𝑛 𝑔 n_{g} italic_n start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT  groups ( n g ≪ n h much-less-than subscript 𝑛 𝑔 subscript 𝑛 ℎ n_{g}\ll n_{h} italic_n start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ≪ italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) to reduce the KV cache. For each head  h ℎ h italic_h , it maps to group  g = ⌊ h / n g ⌋ 𝑔 ℎ subscript 𝑛 𝑔 g=\lfloor h/n_{g}\rfloor italic_g = ⌊ italic_h / italic_n start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ⌋ :

𝒐 i ( h ) = Softmax ⁢ ( 𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( g ) ⊤ ) ⁢ 𝒗 ≤ i ( g ) , superscript subscript 𝒐 𝑖 ℎ Softmax superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from 𝑔 top superscript subscript 𝒗 absent 𝑖 𝑔 \displaystyle\bm{o}_{i}^{(h)}=\text{Softmax}\left(\bm{q}_{i,\text{rope}}^{(h)}% \bm{k}_{\leq i,\text{rope}}^{(g)\top}\right)\bm{v}_{\leq i}^{(g)}, bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = Softmax ( bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_g ) ⊤ end_POSTSUPERSCRIPT ) bold_italic_v start_POSTSUBSCRIPT ≤ italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_g ) end_POSTSUPERSCRIPT ,   GQA ⁢ ( 𝒙 i ) = [ 𝒐 i ( 1 ) , … , 𝒐 i ( n h ) ] ⁢ 𝑾 o . GQA subscript 𝒙 𝑖 superscript subscript 𝒐 𝑖 1 … superscript subscript 𝒐 𝑖 subscript 𝑛 ℎ subscript 𝑾 𝑜 \displaystyle\text{GQA}(\bm{x}_{i})=\left[\bm{o}_{i}^{(1)},\dots,\bm{o}_{i}^{(% n_{h})}\right]\bm{W}_{o}. GQA ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = [ bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT , … , bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT ] bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT .   (2)

Multi-Query Attention (MQA,

https://arxiv.org/html/2502.14837v1#bib.bib27

) is a special case of GQA with  n g = 1 subscript 𝑛 𝑔 1 n_{g}=1 italic_n start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT = 1 , i.e., all heads share a single global key/value. While reducing the KV cache to  𝒪 ⁢ ( 2 ⁢ l ⁢ n g ⁢ d h ) 𝒪 2 𝑙 subscript 𝑛 𝑔 subscript 𝑑 ℎ \mathcal{O}(2ln_{g}d_{h}) caligraphic_O ( 2 italic_l italic_n start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) , these methods degrade performance due to parameter pruning.

2.2  Multi-Head Latent Attention (MLA)

MLA  DeepSeek-AI et al. (

https://arxiv.org/html/2502.14837v1#bib.bib11

)  introduces a hybrid architecture that decouples PE from latent KV compression. For each head  h ℎ h italic_h , the input  𝒙 i subscript 𝒙 𝑖 \bm{x}_{i} bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  is projected into two complementary components:

Position-Aware Component

A subset of dimensions retains PE to preserve positional sensitivity:

𝒒 i , rope ( h ) , 𝒌 i , rope = RoPE ⁢ ( 𝒙 i ⁢ 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ r ( h ) , 𝒙 i ⁢ 𝑾 k ⁢ r ) , superscript subscript 𝒒 𝑖 rope ℎ subscript 𝒌 𝑖 rope RoPE subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 superscript subscript 𝑾 𝑞 𝑟 ℎ subscript 𝒙 𝑖 subscript 𝑾 𝑘 𝑟 \bm{q}_{i,\text{rope}}^{(h)},\bm{k}_{i,\text{rope}}=\text{RoPE}\left(\bm{x}_{i% }\bm{W}_{dq}\bm{W}_{qr}^{(h)},\bm{x}_{i}\bm{W}_{kr}\right), bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_k start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT = RoPE ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_r end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k italic_r end_POSTSUBSCRIPT ) ,

where  𝑾 d ⁢ q ∈ ℝ d × d q subscript 𝑾 𝑑 𝑞 superscript ℝ 𝑑 subscript 𝑑 𝑞 \bm{W}_{dq}\in\mathbb{R}^{d\times d_{q}} bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × italic_d start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝑾 q ⁢ r ( h ) ∈ ℝ d q × d r superscript subscript 𝑾 𝑞 𝑟 ℎ superscript ℝ subscript 𝑑 𝑞 subscript 𝑑 𝑟 \bm{W}_{qr}^{(h)}\in\mathbb{R}^{d_{q}\times d_{r}} bold_italic_W start_POSTSUBSCRIPT italic_q italic_r end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝑾 k ⁢ r ∈ ℝ d × d r subscript 𝑾 𝑘 𝑟 superscript ℝ 𝑑 subscript 𝑑 𝑟 \bm{W}_{kr}\in\mathbb{R}^{d\times d_{r}} bold_italic_W start_POSTSUBSCRIPT italic_k italic_r end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  project queries/keys into a RoPE-preserved component of dimension  d r subscript 𝑑 𝑟 d_{r} italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT .

Position-Agnostic Component

The remaining dimensions  d c subscript 𝑑 𝑐 d_{c} italic_d start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT  are stripped of PE (i.e., NoPE),  𝒌 i , nope ( h ) superscript subscript 𝒌 𝑖 nope ℎ \bm{k}_{i,\text{nope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝒗 i ( h ) superscript subscript 𝒗 𝑖 ℎ \bm{v}_{i}^{(h)} bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and compressed into a shared latent vector  𝒄 i , k ⁢ v ( h ) superscript subscript 𝒄 𝑖 𝑘 𝑣 ℎ \bm{c}_{i,kv}^{(h)} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT :

𝒒 i , nope ( h ) superscript subscript 𝒒 𝑖 nope ℎ \displaystyle\bm{q}_{i,\text{nope}}^{(h)} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒙 i ⁢ 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ( h ) , absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 superscript subscript 𝑾 𝑞 𝑐 ℎ \displaystyle=\bm{x}_{i}\bm{W}_{dq}\bm{W}_{qc}^{(h)}, = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ,   𝒄 i , k ⁢ v subscript 𝒄 𝑖 𝑘 𝑣 \displaystyle\bm{c}_{i,kv} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT   = 𝒙 i ⁢ 𝑾 d ⁢ k ⁢ v , absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑘 𝑣 \displaystyle=\bm{x}_{i}\bm{W}_{dkv}, = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT ,   𝒌 i , nope ( h ) , 𝒗 i ( h ) superscript subscript 𝒌 𝑖 nope ℎ superscript subscript 𝒗 𝑖 ℎ \displaystyle\bm{k}_{i,\text{nope}}^{(h)},\bm{v}_{i}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒄 i , k ⁢ v ⁢ 𝑾 u ⁢ k ( h ) , 𝒄 i , k ⁢ v ⁢ 𝑾 u ⁢ v ( h ) , absent subscript 𝒄 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑘 ℎ subscript 𝒄 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑣 ℎ \displaystyle=\bm{c}_{i,kv}\bm{W}_{uk}^{(h)},\bm{c}_{i,kv}\bm{W}_{uv}^{(h)}, = bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ,

where  𝑾 q ⁢ c ( h ) ∈ ℝ d q × d c superscript subscript 𝑾 𝑞 𝑐 ℎ superscript ℝ subscript 𝑑 𝑞 subscript 𝑑 𝑐 \bm{W}_{qc}^{(h)}\in\mathbb{R}^{d_{q}\times d_{c}} bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝑾 d ⁢ k ⁢ v ∈ ℝ d × d k ⁢ v subscript 𝑾 𝑑 𝑘 𝑣 superscript ℝ 𝑑 subscript 𝑑 𝑘 𝑣 \bm{W}_{dkv}\in\mathbb{R}^{d\times d_{kv}} bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝑾 u ⁢ k ( h ) ∈ ℝ d k ⁢ v × d c superscript subscript 𝑾 𝑢 𝑘 ℎ superscript ℝ subscript 𝑑 𝑘 𝑣 subscript 𝑑 𝑐 \bm{W}_{uk}^{(h)}\in\mathbb{R}^{d_{kv}\times d_{c}} bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝑾 u ⁢ v ( h ) ∈ ℝ d k ⁢ v × d h superscript subscript 𝑾 𝑢 𝑣 ℎ superscript ℝ subscript 𝑑 𝑘 𝑣 subscript 𝑑 ℎ \bm{W}_{uv}^{(h)}\in\mathbb{R}^{d_{kv}\times d_{h}} bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT . Note that  d r + d c = d h subscript 𝑑 𝑟 subscript 𝑑 𝑐 subscript 𝑑 ℎ d_{r}+d_{c}=d_{h} italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT + italic_d start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT = italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT . The attention output of MLA combines both components:

𝒐 i ( h ) = Softm ax ⁢ ( 𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( h ) ⊤ + 𝒒 i , nope ⁢ 𝒌 ≤ i , nope ( h ) ⊤ ) superscript subscript 𝒐 𝑖 ℎ Softm ax superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from ℎ top subscript 𝒒 𝑖 nope superscript subscript 𝒌 absent 𝑖 nope limit-from ℎ top \displaystyle\bm{o}_{i}^{(h)}=\text{Softm}\text{ax}\left(\bm{q}_{i,\text{rope}% }^{(h)}\bm{k}_{\leq i,\text{rope}}^{(h)\top}+\bm{q}_{i,\text{nope}}\bm{k}_{% \leq i,\text{nope}}^{(h)\top}\right) bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = roman_Softm roman_ax ( bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT + bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT )   ⋅ 𝒗 ≤ i ( h ) ⋅ absent superscript subscript 𝒗 absent 𝑖 ℎ \displaystyle\quad\quad\quad\cdot\bm{v}_{\leq i}^{(h)} ⋅ bold_italic_v start_POSTSUBSCRIPT ≤ italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   MLA ⁢ ( 𝒙 i ) = [ 𝒐 i ( 1 ) , … , 𝒐 i ( n h ) ] ⋅ 𝑾 o . MLA subscript 𝒙 𝑖 ⋅ superscript subscript 𝒐 𝑖 1 … superscript subscript 𝒐 𝑖 subscript 𝑛 ℎ subscript 𝑾 𝑜 \displaystyle\quad\quad\text{MLA}(\bm{x}_{i})=\left[\bm{o}_{i}^{(1)},\dots,\bm% {o}_{i}^{(n_{h})}\right]\cdot\bm{W}_{o}. MLA ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = [ bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT , … , bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT ] ⋅ bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT .   (3)

Unlike MHA and its variants, MLA stores the latent vector  𝒄 k ⁢ v subscript 𝒄 𝑘 𝑣 \bm{c}_{kv} bold_italic_c start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT  and  𝒌 i , rope ( h ) superscript subscript 𝒌 𝑖 rope ℎ \bm{k}_{i,\text{rope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  ( 𝒪 ( l d r + l d k ⁢ v ) ) \mathcal{O}\left(ld_{r}+ld_{kv})\right) caligraphic_O ( italic_l italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT + italic_l italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ) ) ) instead of full-rank  𝒌 i , 𝒗 i subscript 𝒌 𝑖 subscript 𝒗 𝑖 \bm{k}_{i},\bm{v}_{i} bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  ( 𝒪 ⁢ ( 2 ⁢ l ⁢ n h ⁢ d h ) 𝒪 2 𝑙 subscript 𝑛 ℎ subscript 𝑑 ℎ \mathcal{O}(2ln_{h}d_{h}) caligraphic_O ( 2 italic_l italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ) ), where  ( d r + d k ⁢ v ) ≪ 2 ⁢ n h ⁢ d h much-less-than subscript 𝑑 𝑟 subscript 𝑑 𝑘 𝑣 2 subscript 𝑛 ℎ subscript 𝑑 ℎ (d_{r}+d_{kv})\ll 2n_{h}d_{h} ( italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT + italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ) ≪ 2 italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT .

Why does MLA need to separate RoPE and NoPE?

MLA introduces matrix merging techniques for the NoPE portion during inference, effectively reducing memory usage. For the dot product operation  𝒒 i , nope ( h ) ⁢ 𝒌 j , nope ( h ) ⊤ superscript subscript 𝒒 𝑖 nope ℎ superscript subscript 𝒌 𝑗 nope limit-from ℎ top \bm{q}_{i,\text{nope}}^{(h)}\bm{k}_{j,\text{nope}}^{(h)\top} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT italic_j , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT , the following identity transformation can be applied

3 To simplify the notation, we omit the superscript

. Matrices  𝑾 u ⁢ v subscript 𝑾 𝑢 𝑣 \bm{W}_{uv} bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT  and  𝑾 o subscript 𝑾 𝑜 \bm{W}_{o} bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT  can also be merged, please refer to Appendix C by  DeepSeek-AI et al. (

https://arxiv.org/html/2502.14837v1#bib.bib11

𝒒 i , nope ⁢ 𝒌 j , nope ⊤ subscript 𝒒 𝑖 nope superscript subscript 𝒌 𝑗 nope top \displaystyle\bm{q}_{i,\text{nope}}\bm{k}_{j,\text{nope}}^{\top} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT bold_italic_k start_POSTSUBSCRIPT italic_j , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT   = ( 𝒙 i ⁢ 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ) ⁢ ( 𝒄 j , k ⁢ v ⁢ 𝑾 u ⁢ k ) ⊤ absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑐 superscript subscript 𝒄 𝑗 𝑘 𝑣 subscript 𝑾 𝑢 𝑘 top \displaystyle=\left(\bm{x}_{i}\bm{W}_{dq}\bm{W}_{qc}\right)\left(\bm{c}_{j,kv}% \bm{W}_{uk}\right)^{\top} = ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT ) ( bold_italic_c start_POSTSUBSCRIPT italic_j , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT   = 𝒙 i ⁢ ( 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ⁢ 𝑾 u ⁢ k ⊤ ) ⁢ 𝒄 j , k ⁢ v ⊤ absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑐 superscript subscript 𝑾 𝑢 𝑘 top superscript subscript 𝒄 𝑗 𝑘 𝑣 top \displaystyle=\bm{x}_{i}\left(\bm{W}_{dq}\bm{W}_{qc}\bm{W}_{uk}^{\top}\right)% \bm{c}_{j,kv}^{\top} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ) bold_italic_c start_POSTSUBSCRIPT italic_j , italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT

where  ( 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ⁢ 𝑾 u ⁢ k ⊤ ) subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑐 superscript subscript 𝑾 𝑢 𝑘 top \left(\bm{W}_{dq}\bm{W}_{qc}\bm{W}_{uk}^{\top}\right) ( bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT )  can be pre-merged into a single matrix, and  𝒄 j , k ⁢ v subscript 𝒄 𝑗 𝑘 𝑣 \bm{c}_{j,kv} bold_italic_c start_POSTSUBSCRIPT italic_j , italic_k italic_v end_POSTSUBSCRIPT  is already stored in the KV cache. As for the RoPE portion, the RoPE( ⋅ ⋅ \cdot ⋅ ) function multiplies the input vector by the rotation matrix (e.g., RoPE( 𝒒 i subscript 𝒒 𝑖 \bm{q}_{i} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) =  𝒒 i ⁢ 𝑹 i subscript 𝒒 𝑖 subscript 𝑹 𝑖 \bm{q}_{i}\bm{R}_{i} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ,  𝑹 i subscript 𝑹 𝑖 \bm{R}_{i} bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ’s specific form will be introduced in

Section   3.1

https://arxiv.org/html/2502.14837v1#S3.SS1

). Therefore, the identity transformation becomes as follows:

𝒒 i , rope ⁢ 𝒌 j , rope ⊤ subscript 𝒒 𝑖 rope superscript subscript 𝒌 𝑗 rope top \displaystyle\bm{q}_{i,\text{rope}}\bm{k}_{j,\text{rope}}^{\top} bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT bold_italic_k start_POSTSUBSCRIPT italic_j , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT   = ( 𝒙 i ⁢ 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ r ⁢ 𝑹 i ) ⁢ ( 𝒙 j ⁢ 𝑾 k ⁢ r ⁢ 𝑹 j ) ⊤ absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑟 subscript 𝑹 𝑖 superscript subscript 𝒙 𝑗 subscript 𝑾 𝑘 𝑟 subscript 𝑹 𝑗 top \displaystyle=\left(\bm{x}_{i}\bm{W}_{dq}\bm{W}_{qr}\bm{R}_{i}\right)\left(\bm% {x}_{j}\bm{W}_{kr}\bm{R}_{j}\right)^{\top} = ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_r end_POSTSUBSCRIPT bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ( bold_italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k italic_r end_POSTSUBSCRIPT bold_italic_R start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT   = 𝒙 i ⁢ ( 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ⁢ 𝑹 j − i ⁢ 𝑾 k ⁢ r ⊤ ) ⁢ 𝒙 j ⊤ absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑐 subscript 𝑹 𝑗 𝑖 superscript subscript 𝑾 𝑘 𝑟 top superscript subscript 𝒙 𝑗 top \displaystyle=\bm{x}_{i}\left(\bm{W}_{dq}\bm{W}_{qc}\bm{R}_{j-i}\bm{W}_{kr}^{% \top}\right)\bm{x}_{j}^{\top} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT bold_italic_R start_POSTSUBSCRIPT italic_j - italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k italic_r end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ) bold_italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT

Since  ( 𝑾 d ⁢ q ⁢ 𝑾 q ⁢ c ⁢ 𝑹 j − i ⁢ 𝑾 k ⁢ r ⊤ ) subscript 𝑾 𝑑 𝑞 subscript 𝑾 𝑞 𝑐 subscript 𝑹 𝑗 𝑖 superscript subscript 𝑾 𝑘 𝑟 top \left(\bm{W}_{dq}\bm{W}_{qc}\bm{R}_{j-i}\bm{W}_{kr}^{\top}\right) ( bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q italic_c end_POSTSUBSCRIPT bold_italic_R start_POSTSUBSCRIPT italic_j - italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k italic_r end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT )  is related to the relative position  j − i 𝑗 𝑖 j-i italic_j - italic_i , it cannot be merged into a fixed matrix. Considering that the relative distances in LLMs can be very long, such as 128K, the RoPE portion is better suited to be computed using the original form.

3.1  Partial-RoPE

To enable migration from standard MHA to MLA, we propose partial-RoPE finetuning, a strategy that removes RoPE from a targeted proportion of dimensions and converts them into NoPE. Critically, while prior work has explored training LLMs with partial-RoPE from scratch (achieving marginally better perplexity than full-RoPE  Black et al. (

https://arxiv.org/html/2502.14837v1#bib.bib8

); Barbero et al. (

https://arxiv.org/html/2502.14837v1#bib.bib6

) ), no existing method addresses how to efficiently adapt pre-trained full-RoPE models (e.g., Llama) to partial-RoPE without costly retraining. Our work bridges this gap by systematically evaluating partial-RoPE variants to identify the most data-efficient fine-tuning protocol for recovering model performance post-adaptation.

Figure 2:  Illustration of  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ,  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT ,  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT ,  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT . Where  d h = 8 subscript 𝑑 ℎ 8 d_{h}=8 italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT = 8  and  r = 2 𝑟 2 r=2 italic_r = 2 .   Figure 3:  Visualization of Head-wise 2-norm Contribution for Llama2-7B. We randomly selected 4 heads, and the  red dashed box  highlights the top- 4 4 4 4  frequency subspaces chosen when  r = 4 𝑟 4 r=4 italic_r = 4 . It can be seen that different heads tend to focus on different frequency subspaces, which validates the rationality of our  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  method.

MHA’s Full-RoPE

encodes positional information into queries and keys through frequency-specific rotations. Formally, given a query vector  𝒒 i ∈ ℝ d h subscript 𝒒 𝑖 superscript ℝ subscript 𝑑 ℎ \bm{q}_{i}\in\mathbb{R}^{d_{h}} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  and key vector  𝒌 i ∈ ℝ d h subscript 𝒌 𝑖 superscript ℝ subscript 𝑑 ℎ \bm{k}_{i}\in\mathbb{R}^{d_{h}} bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT , we partition them into 2D chunks:

𝒒 i , 𝒌 i = [ 𝒒 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ] 0 ≤ k < d h 2 , [ 𝒌 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ] 0 ≤ k < d h 2 , formulae-sequence subscript 𝒒 𝑖 subscript 𝒌 𝑖 subscript delimited-[] superscript subscript 𝒒 𝑖 2 𝑘 2 𝑘 1 0 𝑘 subscript 𝑑 ℎ 2 subscript delimited-[] superscript subscript 𝒌 𝑖 2 𝑘 2 𝑘 1 0 𝑘 subscript 𝑑 ℎ 2 \bm{q}_{i},\bm{k}_{i}=\left[\bm{q}_{i}^{[2k,2k+1]}\right]_{0\leq k<\frac{d_{h}% }{2}},\left[\bm{k}_{i}^{[2k,2k+1]}\right]_{0\leq k<\frac{d_{h}}{2}}, bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = [ bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ] start_POSTSUBSCRIPT 0 ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUBSCRIPT , [ bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ] start_POSTSUBSCRIPT 0 ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUBSCRIPT ,

where  𝒒 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ∈ ℝ 2 superscript subscript 𝒒 𝑖 2 𝑘 2 𝑘 1 superscript ℝ 2 \bm{q}_{i}^{[2k,2k+1]}\in\mathbb{R}^{2} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT  denotes the  k 𝑘 k italic_k -th 2D subspace. Each chunk undergoes a rotation by position-dependent angles  θ k = β − 2 ⁢ k / d h subscript 𝜃 𝑘 superscript 𝛽 2 𝑘 subscript 𝑑 ℎ \theta_{k}=\beta^{-2k/{d_{h}}} italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = italic_β start_POSTSUPERSCRIPT - 2 italic_k / italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT , forming a spectrum of wavelengths. High-frequency components, e.g.,  k = 0 𝑘 0 k=0 italic_k = 0 , rotate rapidly at 1 radian per token. Low-frequency components, e.g.,  k = d h 2 − 1 𝑘 subscript 𝑑 ℎ 2 1 k=\frac{d_{h}}{2}-1 italic_k = divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG - 1 , rotate slowly at  ∼ β 1 / d h similar-to absent superscript 𝛽 1 subscript 𝑑 ℎ \sim\beta^{1/d_{h}} ∼ italic_β start_POSTSUPERSCRIPT 1 / italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  radians per token. The base wavelength  β 𝛽 \beta italic_β , typically set to  10 4 superscript 10 4 10^{4} 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT   Su et al. (

https://arxiv.org/html/2502.14837v1#bib.bib28

)  or  5 × 10 5 5 superscript 10 5 5\!\times\!10^{5} 5 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT .

Formally, for each 2D chunk  𝒒 i [ 2 ⁢ k , 2 ⁢ k + 1 ] superscript subscript 𝒒 𝑖 2 𝑘 2 𝑘 1 \bm{q}_{i}^{[2k,2k+1]} bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT  and  𝒌 i [ 2 ⁢ k , 2 ⁢ k + 1 ] superscript subscript 𝒌 𝑖 2 𝑘 2 𝑘 1 \bm{k}_{i}^{[2k,2k+1]} bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT , the rotation matrix at position  i 𝑖 i italic_i  is defined as:

𝑹 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ⁢ ( θ k ) = [ cos ⁡ ( i ⁢ θ k ) − sin ⁡ ( i ⁢ θ k ) sin ⁡ ( i ⁢ θ k ) cos ⁡ ( i ⁢ θ k ) ] . superscript subscript 𝑹 𝑖 2 𝑘 2 𝑘 1 subscript 𝜃 𝑘 matrix 𝑖 subscript 𝜃 𝑘 𝑖 subscript 𝜃 𝑘 𝑖 subscript 𝜃 𝑘 𝑖 subscript 𝜃 𝑘 \bm{R}_{i}^{[2k,2k+1]}(\theta_{k})=\begin{bmatrix}\cos(i\theta_{k})&-\sin(i% \theta_{k})\\ \sin(i\theta_{k})&\cos(i\theta_{k})\end{bmatrix}. bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) = [ start_ARG start_ROW start_CELL roman_cos ( italic_i italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_CELL start_CELL - roman_sin ( italic_i italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_CELL end_ROW start_ROW start_CELL roman_sin ( italic_i italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_CELL start_CELL roman_cos ( italic_i italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_CELL end_ROW end_ARG ] .

Thus, applying RoPE to queries and keys becomes:

𝒒 i , r ⁢ o ⁢ p ⁢ e = [ 𝑹 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ⁢ ( θ k ) ⁢ 𝒒 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ] 0 ≤ k < d h 2 , subscript 𝒒 𝑖 𝑟 𝑜 𝑝 𝑒 subscript delimited-[] superscript subscript 𝑹 𝑖 2 𝑘 2 𝑘 1 subscript 𝜃 𝑘 superscript subscript 𝒒 𝑖 2 𝑘 2 𝑘 1 0 𝑘 subscript 𝑑 ℎ 2 \displaystyle\bm{q}_{i,rope}=\left[\bm{R}_{i}^{[2k,2k+1]}(\theta_{k})\bm{q}_{i% }^{[2k,2k+1]}\right]_{0\leq k<\frac{d_{h}}{2}}, bold_italic_q start_POSTSUBSCRIPT italic_i , italic_r italic_o italic_p italic_e end_POSTSUBSCRIPT = [ bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) bold_italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ] start_POSTSUBSCRIPT 0 ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUBSCRIPT ,   𝒌 i , r ⁢ o ⁢ p ⁢ e = [ 𝑹 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ⁢ ( θ k ) ⁢ 𝒌 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ] 0 ≤ k < d h 2 . subscript 𝒌 𝑖 𝑟 𝑜 𝑝 𝑒 subscript delimited-[] superscript subscript 𝑹 𝑖 2 𝑘 2 𝑘 1 subscript 𝜃 𝑘 superscript subscript 𝒌 𝑖 2 𝑘 2 𝑘 1 0 𝑘 subscript 𝑑 ℎ 2 \displaystyle\bm{k}_{i,rope}=\left[\bm{R}_{i}^{[2k,2k+1]}(\theta_{k})\bm{k}_{i% }^{[2k,2k+1]}\right]_{0\leq k<\frac{d_{h}}{2}}. bold_italic_k start_POSTSUBSCRIPT italic_i , italic_r italic_o italic_p italic_e end_POSTSUBSCRIPT = [ bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ] start_POSTSUBSCRIPT 0 ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUBSCRIPT .

Full-RoPE to Partial-RoPE Strategies

Given  r 𝑟 r italic_r  retained rotational subspaces( r = d r 2 ≪ 𝑟 subscript 𝑑 𝑟 2 much-less-than absent r=\frac{d_{r}}{2}\ll italic_r = divide start_ARG italic_d start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG ≪  total subspaces  d h 2 subscript 𝑑 ℎ 2 \frac{d_{h}}{2} divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG , we propose four strategies (illustrated in

https://arxiv.org/html/2502.14837v1#S3.F2

) to select which  r 𝑟 r italic_r  subspaces preserve RoPE encoding:

High-Frequency Preservation  retain the  r 𝑟 r italic_r  fastest-rotating (high-frequency) subspaces:

𝒮 high = { k |  0 ≤ k < r } . subscript 𝒮 high conditional-set 𝑘  0 𝑘 𝑟 \mathcal{S}_{\text{high}}=\left\{k\,|\,0\leq k<r\right\}. caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT = { italic_k | 0 ≤ italic_k < italic_r } .

It is consistent with the p-RoPE method proposed in  Barbero et al. (

https://arxiv.org/html/2502.14837v1#bib.bib6

) , where they explored settings in which  r 𝑟 r italic_r  constituted 25%, 50%, and 75% of the total subspaces, and observed a slight advantage over full-RoPE in LLMs trained from scratch.

Low-Frequency Preservation  retain the  r 𝑟 r italic_r  slowest-rotating (low-frequency) subspaces:

𝒮 low = { k | d h 2 − r ≤ k < d h 2 } . subscript 𝒮 low conditional-set 𝑘 subscript 𝑑 ℎ 2 𝑟 𝑘 subscript 𝑑 ℎ 2 \mathcal{S}_{\text{low}}=\left\{k\,\big{|}\,\frac{d_{h}}{2}-r\leq k<\frac{d_{h% }}{2}\right\}. caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT = { italic_k | divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG - italic_r ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG } .

It was chosen as a controlled experiment for the high-frequency strategy.

Uniform Sampling  select  r 𝑟 r italic_r  subspaces with equidistant intervals:

𝒮 uniform = { ⌊ k ⁢ d h 2 ⁢ r ⌋ |  0 ≤ k < r } subscript 𝒮 uniform conditional-set 𝑘 subscript 𝑑 ℎ 2 𝑟  0 𝑘 𝑟 \mathcal{S}_{\text{uniform}}=\left\{\left\lfloor k\frac{d_{h}}{2r}\right% \rfloor\,\bigg{|}\,0\leq k<r\right\} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT = { ⌊ italic_k divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 italic_r end_ARG ⌋ | 0 ≤ italic_k < italic_r }

This balances high- and low-frequency components through geometric spacing. In practice,  2 ⁢ r 2 𝑟 2r 2 italic_r  typically divides  d h subscript 𝑑 ℎ d_{h} italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT . It is similar to the partial RoPE used in GPT-Neo  Black et al. (

https://arxiv.org/html/2502.14837v1#bib.bib8

Head-wise 2-norm Contribution   Barbero et al. (

https://arxiv.org/html/2502.14837v1#bib.bib6

)  were the first to propose the 2-norm contribution to investigate whether these frequencies are utilized and how they are helpful. This approach is based on the observation that, according to the Cauchy-Schwarz inequality, the influence of the  k 𝑘 k italic_k -th frequency subspace on the attention logits is upper-bounded by the 2-norm of the corresponding query and key components, i.e.,  | ⟨ 𝐪 i [ 2 ⁢ k , 2 ⁢ k + 1 ] , 𝐤 j [ 2 ⁢ k , 2 ⁢ k + 1 ] ⟩ | ⩽ ‖ 𝐪 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ‖ ⁢ ‖ 𝐤 j [ 2 ⁢ k , 2 ⁢ k + 1 ] ‖ superscript subscript 𝐪 𝑖 2 𝑘 2 𝑘 1 superscript subscript 𝐤 𝑗 2 𝑘 2 𝑘 1 norm superscript subscript 𝐪 𝑖 2 𝑘 2 𝑘 1 norm superscript subscript 𝐤 𝑗 2 𝑘 2 𝑘 1 \left|\left\langle\mathbf{q}_{i}^{[2k,2k+1]},\mathbf{k}_{j}^{[2k,2k+1]}\right% \rangle\right|\leqslant\left\|\mathbf{q}_{i}^{[2k,2k+1]}\right\|\left\|\mathbf% {k}_{j}^{[2k,2k+1]}\right\| | ⟨ bold_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT , bold_k start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ⟩ | ⩽ ∥ bold_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ∥ ∥ bold_k start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ∥ . For each head  h ℎ h italic_h , we compute the mean 2-norm score for each subspace in an LLM over long sequences

4  The 2-norm calculation detail is placed in

## Appendix   A

https://arxiv.org/html/2502.14837v1#A1

.  . Then, we propose to rank all subspaces by their 2-norm score and select the top- r 𝑟 r italic_r :

𝒮 2-norm = top- ⁢ r 0 ≤ k < d h 2 ⁢ ( ‖ 𝐪 ∗ [ 2 ⁢ k , 2 ⁢ k + 1 ] ‖ ⁢ ‖ 𝐤 ∗ [ 2 ⁢ k , 2 ⁢ k + 1 ] ‖ ) . subscript 𝒮 2-norm 0 𝑘 subscript 𝑑 ℎ 2 top- 𝑟 norm superscript subscript 𝐪 2 𝑘 2 𝑘 1 norm superscript subscript 𝐤 2 𝑘 2 𝑘 1 \displaystyle\mathcal{S}_{\text{2-norm}}\!=\!\underset{0\leq k<\frac{d_{h}}{2}% }{\text{top-}r}\left(\left\|\mathbf{q}_{*}^{[2k,2k+1]}\right\|\left\|\mathbf{k% }_{*}^{[2k,2k+1]}\right\|\right). caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT = start_UNDERACCENT 0 ≤ italic_k < divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_UNDERACCENT start_ARG top- italic_r end_ARG ( ∥ bold_q start_POSTSUBSCRIPT ∗ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ∥ ∥ bold_k start_POSTSUBSCRIPT ∗ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ∥ ) .

This head-specific selection adaptively preserves rotation-critical subspaces.

https://arxiv.org/html/2502.14837v1#S3.F3

visualizes the 2-norm of Llama2-7B’s four heads.

## We will analyze the effectiveness of the four strategies in

Section   4.3

https://arxiv.org/html/2502.14837v1#S4.SS3

and conduct an ablation study on the essential hyperparameter  r 𝑟 r italic_r  in

## Appendix   D

https://arxiv.org/html/2502.14837v1#A4

. For all strategies, non-selected subspaces ( k ∉ 𝒮 𝑘 𝒮 k\notin\mathcal{S} italic_k ∉ caligraphic_S ) become NoPE dimensions, enabling seamless integration with MLA’s latent compression.

Figure 4:  Illustration of  SVD

. In the multi-head setting, we adhere to the standard MLA approach, performing SVD on the merged multi-heads rather than on each head individually (e.g.,  𝑼 k ⁢ v ∈ ℝ n h ⁢ d h × n h ⁢ d k ⁢ v subscript 𝑼 𝑘 𝑣 superscript ℝ subscript 𝑛 ℎ subscript 𝑑 ℎ subscript 𝑛 ℎ subscript 𝑑 𝑘 𝑣 \bm{U}_{kv}\in\mathbb{R}^{n_{h}d_{h}\times n_{h}d_{kv}} bold_italic_U start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT × italic_n start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_POSTSUPERSCRIPT .

3.2  Low-rank Approximation

After transitioning from full RoPE to partial RoPE, we obtain the first component of the KV cache in MLA, represented as:  𝒌 i , r ⁢ o ⁢ p ⁢ e = [ 𝑹 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ⁢ ( θ k ) ⁢ 𝒌 i [ 2 ⁢ k , 2 ⁢ k + 1 ] ] k ∈ 𝒮 subscript 𝒌 𝑖 𝑟 𝑜 𝑝 𝑒 subscript delimited-[] superscript subscript 𝑹 𝑖 2 𝑘 2 𝑘 1 subscript 𝜃 𝑘 superscript subscript 𝒌 𝑖 2 𝑘 2 𝑘 1 𝑘 𝒮 \bm{k}_{i,rope}=\left[\bm{R}_{i}^{[2k,2k+1]}(\theta_{k})\bm{k}_{i}^{[2k,2k+1]}% \right]_{k\in\mathcal{S}} bold_italic_k start_POSTSUBSCRIPT italic_i , italic_r italic_o italic_p italic_e end_POSTSUBSCRIPT = [ bold_italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT [ 2 italic_k , 2 italic_k + 1 ] end_POSTSUPERSCRIPT ] start_POSTSUBSCRIPT italic_k ∈ caligraphic_S end_POSTSUBSCRIPT . Our next goal is to derive the second component,  𝒄 i , k ⁢ v ∈ ℝ d k ⁢ v subscript 𝒄 𝑖 𝑘 𝑣 superscript ℝ subscript 𝑑 𝑘 𝑣 \bm{c}_{i,kv}\in\mathbb{R}^{d_{kv}} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_POSTSUPERSCRIPT , which serves as a low-rank representation of  𝒌 i , nope subscript 𝒌 𝑖 nope \bm{k}_{i,\text{nope}} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT  and  𝒗 i subscript 𝒗 𝑖 \bm{v}_{i} bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT .

Given the keys  𝒌 i = 𝒙 i ⁢ 𝑾 k subscript 𝒌 𝑖 subscript 𝒙 𝑖 subscript 𝑾 𝑘 \bm{k}_{i}=\bm{x}_{i}\bm{W}_{k} bold_italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT  and values  𝒗 i = 𝒙 i ⁢ 𝑾 v subscript 𝒗 𝑖 subscript 𝒙 𝑖 subscript 𝑾 𝑣 \bm{v}_{i}=\bm{x}_{i}\bm{W}_{v} bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  in MHA, we first extract the subspace of  𝑾 k subscript 𝑾 𝑘 \bm{W}_{k} bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT  corresponding to  𝒌 i , nope subscript 𝒌 𝑖 nope \bm{k}_{i,\text{nope}} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT , i.e., the dimensions not included in  𝒮 𝒮 \mathcal{S} caligraphic_S , yielding:  𝒌 i , nope = 𝒙 i ⁢ 𝑾 k , nope subscript 𝒌 𝑖 nope subscript 𝒙 𝑖 subscript 𝑾 𝑘 nope \bm{k}_{i,\text{nope}}=\bm{x}_{i}\bm{W}_{k,\text{nope}} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k , nope end_POSTSUBSCRIPT . We propose two Singular Value Decomposition (SVD)-based strategies (Illustrated in

https://arxiv.org/html/2502.14837v1#S3.F4

) to preserve pre-trained knowledge while achieving rank reduction:

Decoupled SVD (SVD

Separately decompose  𝑾 k , nope subscript 𝑾 𝑘 nope \bm{W}_{k,\text{nope}} bold_italic_W start_POSTSUBSCRIPT italic_k , nope end_POSTSUBSCRIPT  and  𝑾 v subscript 𝑾 𝑣 \bm{W}_{v} bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  into truncated SVDs, allocating  d k ⁢ v / 2 subscript 𝑑 𝑘 𝑣 2 d_{kv}/2 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT / 2  dimensions to each:

𝑾 k , nope = 𝑼 k ⁢ 𝚺 k ⁢ 𝑽 k ⊤ , 𝑾 v = 𝑼 v ⁢ 𝚺 v ⁢ 𝑽 v ⊤ , formulae-sequence subscript 𝑾 𝑘 nope subscript 𝑼 𝑘 subscript 𝚺 𝑘 superscript subscript 𝑽 𝑘 top subscript 𝑾 𝑣 subscript 𝑼 𝑣 subscript 𝚺 𝑣 superscript subscript 𝑽 𝑣 top \bm{W}_{k,\text{nope}}=\bm{U}_{k}\bm{\Sigma}_{k}\bm{V}_{k}^{\top},\quad\bm{W}_% {v}=\bm{U}_{v}\bm{\Sigma}_{v}\bm{V}_{v}^{\top}, bold_italic_W start_POSTSUBSCRIPT italic_k , nope end_POSTSUBSCRIPT = bold_italic_U start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT = bold_italic_U start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,

where  𝑼 k , 𝑼 v , 𝑽 k , 𝑽 v ∈ ℝ d h × d k ⁢ v 2 subscript 𝑼 𝑘 subscript 𝑼 𝑣 subscript 𝑽 𝑘 subscript 𝑽 𝑣 superscript ℝ subscript 𝑑 ℎ subscript 𝑑 𝑘 𝑣 2 \bm{U}_{k},\bm{U}_{v},\bm{V}_{k},\bm{V}_{v}\in\mathbb{R}^{d_{h}\times\frac{d_{% kv}}{2}} bold_italic_U start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , bold_italic_U start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT , bold_italic_V start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , bold_italic_V start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT × divide start_ARG italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUPERSCRIPT ,  𝚺 k , 𝚺 v ∈ ℝ d k ⁢ v 2 × d k ⁢ v 2 subscript 𝚺 𝑘 subscript 𝚺 𝑣 superscript ℝ subscript 𝑑 𝑘 𝑣 2 subscript 𝑑 𝑘 𝑣 2 \bm{\Sigma}_{k},\bm{\Sigma}_{v}\in\mathbb{R}^{\frac{d_{kv}}{2}\times\frac{d_{% kv}}{2}} bold_Σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , bold_Σ start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT divide start_ARG italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG × divide start_ARG italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG end_POSTSUPERSCRIPT . The down-projection matrices  𝑾 d ⁣ ∗ subscript 𝑾 𝑑 \bm{W}_{d*} bold_italic_W start_POSTSUBSCRIPT italic_d ∗ end_POSTSUBSCRIPT  and up-projection matrices  𝑾 u ⁣ ∗ subscript 𝑾 𝑢 \bm{W}_{u*} bold_italic_W start_POSTSUBSCRIPT italic_u ∗ end_POSTSUBSCRIPT  become:

𝑾 d ⁢ k = 𝑼 k ⁢ 𝚺 k 1 / 2 , 𝑾 u ⁢ k = 𝚺 k 1 / 2 ⁢ 𝑽 k ⊤ , formulae-sequence subscript 𝑾 𝑑 𝑘 subscript 𝑼 𝑘 superscript subscript 𝚺 𝑘 1 2 subscript 𝑾 𝑢 𝑘 superscript subscript 𝚺 𝑘 1 2 superscript subscript 𝑽 𝑘 top \displaystyle\bm{W}_{dk}=\bm{U}_{k}\bm{\Sigma}_{k}^{1/2},\quad\bm{W}_{uk}=\bm{% \Sigma}_{k}^{1/2}\bm{V}_{k}^{\top}, bold_italic_W start_POSTSUBSCRIPT italic_d italic_k end_POSTSUBSCRIPT = bold_italic_U start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT = bold_Σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,   𝑾 d ⁢ v = 𝑼 v ⁢ 𝚺 v 1 / 2 , 𝑾 u ⁢ v = 𝚺 v 1 / 2 ⁢ 𝑽 v ⊤ . formulae-sequence subscript 𝑾 𝑑 𝑣 subscript 𝑼 𝑣 superscript subscript 𝚺 𝑣 1 2 subscript 𝑾 𝑢 𝑣 superscript subscript 𝚺 𝑣 1 2 superscript subscript 𝑽 𝑣 top \displaystyle\bm{W}_{dv}=\bm{U}_{v}\bm{\Sigma}_{v}^{1/2},\quad\bm{W}_{uv}=\bm{% \Sigma}_{v}^{1/2}\bm{V}_{v}^{\top}. bold_italic_W start_POSTSUBSCRIPT italic_d italic_v end_POSTSUBSCRIPT = bold_italic_U start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT = bold_Σ start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT .

The low-rank representation  𝒄 i , k ⁢ v subscript 𝒄 𝑖 𝑘 𝑣 \bm{c}_{i,kv} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT  can be constructed using  𝒄 i , k ⁢ v = [ 𝒙 i ⁢ 𝑾 d ⁢ k , 𝒙 i ⁢ 𝑾 d ⁢ v ] subscript 𝒄 𝑖 𝑘 𝑣 subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑘 subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑣 \bm{c}_{i,kv}=\left[\bm{x}_{i}\bm{W}_{dk},\bm{x}_{i}\bm{W}_{dv}\right] bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT = [ bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_k end_POSTSUBSCRIPT , bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_v end_POSTSUBSCRIPT ] .

Joint SVD (SVD

To preserve interactions between  𝑲 nope subscript 𝑲 nope \bm{K}_{\text{nope}} bold_italic_K start_POSTSUBSCRIPT nope end_POSTSUBSCRIPT  and  𝑽 𝑽 \bm{V} bold_italic_V , we jointly factorize the concatenated matrix:

[ 𝑾 k , nope , 𝑾 v ] = 𝑼 k ⁢ v ⁢ 𝚺 k ⁢ v ⁢ 𝑽 k ⁢ v ⊤ , subscript 𝑾 𝑘 nope subscript 𝑾 𝑣 subscript 𝑼 𝑘 𝑣 subscript 𝚺 𝑘 𝑣 superscript subscript 𝑽 𝑘 𝑣 top [\bm{W}_{k,\text{nope}},\bm{W}_{v}]=\bm{U}_{kv}\bm{\Sigma}_{kv}\bm{V}_{kv}^{% \top}, [ bold_italic_W start_POSTSUBSCRIPT italic_k , nope end_POSTSUBSCRIPT , bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ] = bold_italic_U start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ,

where  𝑼 k ⁢ v , 𝑽 k ⁢ v ∈ ℝ d h × d k ⁢ v subscript 𝑼 𝑘 𝑣 subscript 𝑽 𝑘 𝑣 superscript ℝ subscript 𝑑 ℎ subscript 𝑑 𝑘 𝑣 \bm{U}_{kv},\bm{V}_{kv}\in\mathbb{R}^{d_{h}\times d_{kv}} bold_italic_U start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT , bold_italic_V start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ,  𝚺 k ⁢ v ∈ ℝ d k ⁢ v × d k ⁢ v subscript 𝚺 𝑘 𝑣 superscript ℝ subscript 𝑑 𝑘 𝑣 subscript 𝑑 𝑘 𝑣 \bm{\Sigma}_{kv}\in\mathbb{R}^{d_{kv}\times d_{kv}} bold_Σ start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT end_POSTSUPERSCRIPT . The latent projection is then:

𝑾 d ⁢ k ⁢ v = 𝑼 k ⁢ v ⁢ 𝚺 k ⁢ v 1 / 2 , subscript 𝑾 𝑑 𝑘 𝑣 subscript 𝑼 𝑘 𝑣 superscript subscript 𝚺 𝑘 𝑣 1 2 \displaystyle\bm{W}_{dkv}=\bm{U}_{kv}\bm{\Sigma}_{kv}^{1/2}, bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT = bold_italic_U start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT bold_Σ start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT ,   𝑾 u ⁢ k = 𝚺 k ⁢ v 1 / 2 𝑽 k ⁢ v [ : , : − d v ] , 𝑾 u ⁢ v = 𝚺 k ⁢ v 1 / 2 𝑽 k ⁢ v [ : , d v : ] . \displaystyle\bm{W}_{uk}\!=\!\bm{\Sigma}_{kv}^{1/2}\bm{V}_{kv}[:,:-d_{v}],\bm{% W}_{uv}\!=\!\bm{\Sigma}_{kv}^{1/2}\bm{V}_{kv}[:,d_{v}:]. bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT = bold_Σ start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT [ : , : - italic_d start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ] , bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT = bold_Σ start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 / 2 end_POSTSUPERSCRIPT bold_italic_V start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT [ : , italic_d start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT : ] .

This jointly optimizes the latent space for both keys and values, i.e.,  𝒄 i , k ⁢ v = 𝒙 i ⁢ 𝑾 d ⁢ k ⁢ v subscript 𝒄 𝑖 𝑘 𝑣 subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑘 𝑣 \bm{c}_{i,kv}=\bm{x}_{i}\bm{W}_{dkv} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT , retaining cross-parameter dependencies critical for autoregressive generation

5  We describe the economical inference process of MHA2MLA in

## Appendix   B

https://arxiv.org/html/2502.14837v1#A2

Section   4.3

https://arxiv.org/html/2502.14837v1#S4.SS3

shows  SVD

outperforming  SVD

, validating that joint factorization better preserves pre-trained knowledge.

4  Experiment

Model   Tokens   KV Mem.   Avg.   MMLU   ARC   PIQA   HS   OBQA   WG   135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   600B   44.50   29.80   42.43   68.06   41.09   33.60   52.01   - GQA   d k ⁢ v = 128 subscript 𝑑 𝑘 𝑣 128 d_{kv}\!=\!128 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 128   44.25   29.82   42.05   68.34   41.03   33.20   51.07   - GQA2MLA   d k ⁢ v = 32 subscript 𝑑 𝑘 𝑣 32 d_{kv}\!=\!32 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32   2.25B   -68.75%   43.06

29.50   40.48   66.59   37.99   33.80   49.96   d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16   (3.8‰)   -81.25%   41.84

28.66   39.95   65.02   36.04   31.60   49.80   d k ⁢ v = 8 subscript 𝑑 𝑘 𝑣 8 d_{kv}\!=\!8 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 8   -87.50%   40.97

28.37   38.04   64.69   33.58   30.80   50.36   360M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   600B   49.60   33.70   49.82   71.87   51.65   37.60   52.96   - GQA   d k ⁢ v = 128 subscript 𝑑 𝑘 𝑣 128 d_{kv}\!=\!128 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 128   49.63   34.01   50.02   71.33   51.43   38.20   52.80   - GQA2MLA   d k ⁢ v = 32 subscript 𝑑 𝑘 𝑣 32 d_{kv}\!=\!32 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32   2.25B   -68.75%   47.91

32.94   48.36   70.73   48.16   36.00   51.30   d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16   (3.8‰)   -81.25%   46.94

31.55   45.73   70.51   45.80   36.60   51.46   d k ⁢ v = 8 subscript 𝑑 𝑘 𝑣 8 d_{kv}\!=\!8 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 8   -87.50%   45.04

30.54   43.33   68.50   42.83   35.00   50.04   1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   1T   55.90   39.27   59.87   75.73   62.93   42.80   54.85   - MHA   d k ⁢ v = 128 subscript 𝑑 𝑘 𝑣 128 d_{kv}\!=\!128 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 128   55.93   39.11   59.19   75.95   62.92   43.40   55.09   - MHA2MLA   d k ⁢ v = 32 subscript 𝑑 𝑘 𝑣 32 d_{kv}\!=\!32 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32   6B   -68.75%   54.76

38.11   57.13   76.12   61.35   42.00   53.83   d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16   (6.0‰)   -81.25%   54.65

37.87   56.81   75.84   60.41   42.60   54.38   d k ⁢ v = 8 subscript 𝑑 𝑘 𝑣 8 d_{kv}\!=\!8 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 8   -87.50%   53.61

37.17   55.50   74.86   58.55   41.20   54.38   7B Llama2 Llama2 {}_{\text{Llama2}} start_FLOATSUBSCRIPT Llama2 end_FLOATSUBSCRIPT   2T   59.85   41.43   59.24   78.40   73.29   41.80   64.96   - MHA   d k ⁢ v = 256 subscript 𝑑 𝑘 𝑣 256 d_{kv}\!=\!256 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 256   60.22   41.63   60.89   77.80   71.98   45.00   63.38   - MHA2MLA   d k ⁢ v = 64 subscript 𝑑 𝑘 𝑣 64 d_{kv}\!=\!64 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 64   6B   -68.75%   59.51

41.36   59.51   77.37   71.72   44.20   62.90   d k ⁢ v = 32 subscript 𝑑 𝑘 𝑣 32 d_{kv}\!=\!32 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32   (3.0‰)   -81.25%   59.61

40.86   59.74   77.75   70.75   45.60   62.98   d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16   -87.50%   58.96

40.39   59.29   77.75   69.70   43.40   63.22   Table 1:  Commonsense reasoning ability of four LLMs with MHA2MLA or GQA2MLA. The six benchmarks include MMLU (

https://arxiv.org/html/2502.14837v1#bib.bib15

), ARC easy and challenge (ARC,

https://arxiv.org/html/2502.14837v1#bib.bib10

https://arxiv.org/html/2502.14837v1#bib.bib7

), HellaSwag (HS,

https://arxiv.org/html/2502.14837v1#bib.bib32

), OpenBookQA (OBQA,

https://arxiv.org/html/2502.14837v1#bib.bib22

), Winogrande (WG,

https://arxiv.org/html/2502.14837v1#bib.bib24

We evaluate our method on LLMs of varying scales (SmolLM-135M/360M/1B7, Llama2-7B) pre-trained with MHA or GQA. We chose the SmolLM-series

https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966

https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966

because its pretraining data and framework are both open-source, which can minimize the gap in fine-tuning data and processes. We chose Llama2-7B

https://huggingface.co/meta-llama/Llama-2-7b

https://huggingface.co/meta-llama/Llama-2-7b

because it is one of the widely used open-source LLMs (but its pretraining data is not open-source, there is a potential gap in fine-tuning data).

We denote the architectural migration using MHA2MLA and GQA2MLA, respectively.

8  The details of the fine-tuning process (including data and hyperparameters) are provided in

## Appendix   C

https://arxiv.org/html/2502.14837v1#A3

.   Both adopt  data-efficient full-parameter fine-tuning , with the head-wise 2-norm selection ( 𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT ,  r = d h 16 𝑟 subscript 𝑑 ℎ 16 r=\frac{d_{h}}{16} italic_r = divide start_ARG italic_d start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT end_ARG start_ARG 16 end_ARG ) for Partial-RoPE and joint SVD factorization ( SVD

) for low-rank approximation as default configurations. Our experiments address three critical questions:

How does MHA2MLA minimize accuracy degradation induced by architectural shifts?

What does MHA2MLA achieve in the KV cache reduction ratio?

Can MHA2MLA integrate with KV cache quantization for compound gains?

4.1  Commonsense Reasoning Tasks

## Main Results

As shown in Table 1, our method achieves efficient architectural migration across four model scales (135M to 7B) under varying KV cache compression ratios (via latent dimension  d k ⁢ v subscript 𝑑 𝑘 𝑣 d_{kv} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT ). First, when comparing the performance of our fine-tuning approach with the original LLM, we observe only minor changes in performance across the four base models: a -0.25% decrease on the 135M, +0.03% on the 360M, +0.03% on the 1B7, and +0.37% on the 7B. This suggests that the fine-tuning data does not significantly degrade or improve the performance of the original model, providing an appropriate experimental setting for the MHA2MLA framework.

Next, as  d k ⁢ v subscript 𝑑 𝑘 𝑣 d_{kv} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT  decreases (e.g., from 32 to 16 to 8), the KV cache reduction increases (i.e., from -68.75% to -81.25% to -87.5%), but the performance loss becomes more challenging to recover through fine-tuning.

https://arxiv.org/html/2502.14837v1#S4.F5

shows the fine-tuning loss curves of 135M (representing GQA) and 7B (representing MHA) under different compression ratios. As the compression ratio increases, the loss difference from the baseline becomes larger. Additionally, we observe that the fluctuation trends of the loss curves are  almost identical , indicating that our architecture migration does not significantly harm the model’s internal knowledge.

Figure 5:  The fine-tuning loss curves under different KV cache storage ratios (with colors ranging from light to dark representing 12.5%, 18.75%, 31.25%, and 100%).

We also find that larger models experience less performance degradation when transitioning to the MLA architecture. For example, with compression down to 18.75%, the performance drops by 2.41% for 135M, 2.69% for 360M, 1.28% for 1B7, and 0.61% for 7B, revealing the  potential scaling law of MHA2MLA . Finally, from the 135M model to the 7B model, the number of tokens required for fine-tuning is only about 0.3% to 0.6% of the pretraining tokens, demonstrating the data efficiency of our method.

Overall, whether using GQA2MLA or MHA2MLA, the architecture transition is achieved with minimal cost, resulting in efficient and economical inference.

4.2  Long Context Tasks

Model   Precision   KV Mem.   Avg@LB   7B Llama2 Llama2 {}_{\text{Llama2}} start_FLOATSUBSCRIPT Llama2 end_FLOATSUBSCRIPT   BF16   100.0%   27.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -75.00%   27.5   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   27.3   Int2 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -87.50%   21.2   Int2 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   18.5   d k ⁢ v = 64 subscript 𝑑 𝑘 𝑣 64 d_{kv}\!=\!64 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 64   BF16   -68.75%   27.1   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -92.19%   26.9   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   26.8   d k ⁢ v = 32 subscript 𝑑 𝑘 𝑣 32 d_{kv}\!=\!32 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32   BF16   -81.25%   26.3   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -95.31%   26.1   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   26.1   d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16   BF16   -87.50%   24.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -96.87%   24.2   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   23.4   Table 2:  Evaluation results of Llama2-7B and MHA2MLA on LongBench.  Bold  indicates compression ratios greater than or equal to Int2 quantization while also achieving performance higher than Int2.

To evaluate the generative capabilities of the model, we adopt LongBench  Bai et al. (

https://arxiv.org/html/2502.14837v1#bib.bib5

)  as the benchmark for generation performance. All models are tested using a greedy decoding strategy. The context window size is determined based on the sequence length used during model fine-tuning. We use HQQ  Badri and Shaji (

https://arxiv.org/html/2502.14837v1#bib.bib4

)  and Quanto

https://huggingface.co/blog/quanto-introduction

https://huggingface.co/blog/quanto-introduction

to set caches with different levels of precision to evaluate the performance of the original model as the baseline. Since our method is compatible with KV cache quantization, we also conduct additional experiments to assess the combined effect of both approaches.

## Main Results

## As evidenced in

https://arxiv.org/html/2502.14837v1#S4.T2

, MHA2MLA achieves competitive or superior efficiency-accuracy profiles compared to post-training quantization methods on LongBench. While 4-bit quantization incurs modest degradation (-0.2% to -0.4%) at comparable compression ratios, aggressive 2-bit quantization suffers severe performance collapse (-6.2% to -9%) despite 87.5% KV cache reduction. In contrast, MHA2MLA alone attains 87.5% compression (at  d k ⁢ v = 16 subscript 𝑑 𝑘 𝑣 16 d_{kv}\!=\!16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16 ) with only 3% accuracy loss, and further synergizes with 4-bit quantization to reach 92.19%/96.87% compression ( d k ⁢ v = 64 / 16 subscript 𝑑 𝑘 𝑣 64 16 d_{kv}\!=\!64/16 italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 64 / 16 +Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT ) while limiting degradation to -0.5%/-3.2%, outperforming all 2-bit baselines. This highlights that MHA2MLA’s latent space design remains orthogonal to numerical precision reduction, enabling  compound efficiency gains  without destructive interference.

4.3  Ablation Study

Model   Tokens   Avg@CS   135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   600B   44.50   - full-rope   2.25B   44.25   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT   43.40

-  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT   37.76
-  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT   43.76
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT   43.77
-  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT  + SVD
2.25B   41.04

-  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  + SVD
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD
1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   1T   55.90   - full-rope   6B   55.93   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT   55.17

-  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT   54.72
-  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT   55.31
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT   55.10
-  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT  + SVD
6B   54.41

-  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  + SVD
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD
-  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD
Table 3:  Reasoning ability of ablation studies. The results of other models are provided in Appendix

https://arxiv.org/html/2502.14837v1#A5

Four Partial-RoPE strategies:  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ,  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT ,  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT ,  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT

https://arxiv.org/html/2502.14837v1#A5

presents the results of four strategies for converting full-RoPE to partial-RoPE. First, when comparing the four strategies with full-RoPE, we observed that the low-frequency retention strategy,  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , incurred the greatest performance loss (a reduction of -6.49%@135M and -1.21%@1B7), whereas the high-frequency retention strategy,  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT , experienced significantly less degradation (a reduction of -0.85%@135M and -0.76%@1B7), underscoring the importance of high-frequency subspaces. Both  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  and  𝒮 2 ⁢ -norm subscript 𝒮 2 -norm \mathcal{S}_{2\text{-norm}} caligraphic_S start_POSTSUBSCRIPT 2 -norm end_POSTSUBSCRIPT  yielded better performance, the  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  preserves subspaces across the frequency spectrum, while the  𝒮 2 ⁢ -norm subscript 𝒮 2 -norm \mathcal{S}_{2\text{-norm}} caligraphic_S start_POSTSUBSCRIPT 2 -norm end_POSTSUBSCRIPT  retains subspaces based on their contribution to the attention scores. We choose  𝒮 2 ⁢ -norm subscript 𝒮 2 -norm \mathcal{S}_{2\text{-norm}} caligraphic_S start_POSTSUBSCRIPT 2 -norm end_POSTSUBSCRIPT  as the default configuration because the removed subspaces (i.e., NoPE) are more suitable for the (SVD-based) low-rank approximation.

Two SVD-based low-rank approximations: SVD

## The last two rows of each group in

https://arxiv.org/html/2502.14837v1#S4.T3

compare the effects of the two SVD methods. We observe that, on both LLMs, the SVD joint joint {}_{\text{joint}} start_FLOATSUBSCRIPT joint end_FLOATSUBSCRIPT  method consistently outperforms SVD split split {}_{\text{split}} start_FLOATSUBSCRIPT split end_FLOATSUBSCRIPT , yielding an average performance improvement of 0.92% on the 135M model and 0.74% on the 1B7 model. It indicates that SVD joint joint {}_{\text{joint}} start_FLOATSUBSCRIPT joint end_FLOATSUBSCRIPT  emerges as the clear default choice.

5  Related Work

## Efficient Attention Architectures

The standard Multi-Head Attention (MHA,

https://arxiv.org/html/2502.14837v1#bib.bib30

) mechanism’s quadratic complexity in context length has spurred numerous efficiency innovations. While MHA remains foundational, variants like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA,

https://arxiv.org/html/2502.14837v1#bib.bib1

) reduce memory overhead by sharing keys/values across heads—albeit at the cost of parameter pruning and performance degradation. Parallel efforts, such as Linear Transformers  Guo et al. (

https://arxiv.org/html/2502.14837v1#bib.bib14

); Katharopoulos et al. (

https://arxiv.org/html/2502.14837v1#bib.bib18

); Choromanski et al. (

https://arxiv.org/html/2502.14837v1#bib.bib9

) , RWKV  Peng et al. (

https://arxiv.org/html/2502.14837v1#bib.bib23

) , and Mamba  Gu and Dao (

https://arxiv.org/html/2502.14837v1#bib.bib13

) , replace softmax attention with linear recurrences or state-space models, but struggle to match the expressiveness of standard attention in autoregressive generation.

Multi-Head Latent Attention (MLA,

https://arxiv.org/html/2502.14837v1#bib.bib11

) distinguishes itself by compressing KV caches into low-rank latent vectors without pruning attention parameters. Our work bridges MLA with mainstream architectures (MHA/GQA), enabling seamless migration via data-efficient fine-tuning. Notably, while many linear attention variants abandon softmax query-key interactions (e.g., through kernel approximations), architectures preserving a query-key dot product structure—even in factorized forms—remain compatible with our MHA2MLA framework.

Economical Key-Value Cache

The memory footprint of KV caches has become a critical bottleneck for long-context inference. Recent advances fall into three categories:

Innovative Architecture  methods like MLA  DeepSeek-AI et al. (

https://arxiv.org/html/2502.14837v1#bib.bib11

) , MiniCache  Liu et al. (

https://arxiv.org/html/2502.14837v1#bib.bib20

) , and MLKV  Zuhri et al. (

https://arxiv.org/html/2502.14837v1#bib.bib33

)  share or compress KV representations across layers or heads. While effective, cross-layer sharing risks conflating distinct attention patterns, potentially harming task-specific performance. Only MLA has been successfully validated in Deepseek’s LLMs.

Quantization  techniques such as GPTQ  Frantar et al. (

https://arxiv.org/html/2502.14837v1#bib.bib12

) , FlexGen  Sheng et al. (

https://arxiv.org/html/2502.14837v1#bib.bib26

) , and KIVI  Liu et al. (

https://arxiv.org/html/2502.14837v1#bib.bib21

)  store KV caches in low-bit formats (e.g., 2-bit), achieving memory savings with precision loss.

Dynamic Pruning  approaches like A2SF  Jo and Shin (

https://arxiv.org/html/2502.14837v1#bib.bib16

)  and SnapKV  Li et al. (

https://arxiv.org/html/2502.14837v1#bib.bib19

)  prune “less important” tokens from the KV cache. However, token pruning risks discarding critical long-range dependencies, while head pruning (e.g., SliceGPT  Ashkboos et al. (

https://arxiv.org/html/2502.14837v1#bib.bib3

) , Sheared  Xia et al. (

https://arxiv.org/html/2502.14837v1#bib.bib31

) , and Simple Pruning  Sun et al. (

https://arxiv.org/html/2502.14837v1#bib.bib29

) ) irreversibly reduces model capacity.

Our MHA2MLA method achieves the migration of standard Transformer-based LLMs to the more economical MLA architecture and has demonstrated its ability to integrate with KV quantization techniques to realize a ~97% cache saving. It is also theoretically compatible with other methods like pruning.

6  Conclusion

This work addresses the critical challenge of adapting pre-trained MHA-based LLMs (or variants) to the KV-cache-efficient MLA architecture. By introducing MHA2MLA with contribution-aware partial-RoPE removal and SVD-driven low-rank projection, we achieve near-lossless compression of KV cache (up to 96.87% size reduction for Llama2-7B) while requiring only 3‰ to 6‰of training data. The framework demonstrates strong compatibility with existing compression techniques and maintains commonsense reasoning and long-context processing capabilities, offering a practical pathway for deploying resource-efficient LLMs without sacrificing performance. Our results underscore the feasibility of architectural migration for LLMs through targeted parameter reuse and data-efficient fine-tuning.

## Limitations

## Verification on More LLMs

Considering that MHA2MLA can significantly reduce inference costs, it is worthwhile to validate it on larger and more diverse open-source LLMs. However, constrained by our computation resources, models like Llama3 require fine-tuning on a 128K context length to mitigate performance degradation from continued training, so we did not perform such experiments. Furthermore, since Deepseek has not yet open-sourced the tensor-parallel inference framework for MLA, it is currently challenging to explore models larger than 7B. This will be addressed in our future work.

Parameter-Efficient MHA2MLA Fine-tuning

This paper primarily focuses on the data efficiency of MHA2MLA. Since the architectural transformation does not involve the Feed-Forward (FFN) module, future work could explore parameter-efficient MHA2MLA fine-tuning, for example by freezing the FFN module and/or freezing the parameters in the queries and keys that correspond to the retained RoPE. This could further reduce the cost of the MHA2MLA transition.

Ainslie et al. (2023)    Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023.

GQA: training generalized multi-query transformer models from multi-head checkpoints

https://doi.org/10.18653/V1/2023.EMNLP-MAIN.298

Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023

, pages 4895–4901. Association for Computational Linguistics.

An et al. (2024)    Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024.

L-eval: Instituting standardized evaluation for long context language models

https://doi.org/10.18653/V1/2024.ACL-LONG.776

Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024

, pages 14388–14411. Association for Computational Linguistics.

Ashkboos et al. (2024)    Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, and James Hensman. 2024.

Slicegpt: Compress large language models by deleting rows and columns

https://openreview.net/forum?id=vXxardq6db

The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024

. OpenReview.net.

Badri and Shaji (2023)    Hicham Badri and Appu Shaji. 2023.

Half-quadratic quantization of large machine learning models

https://mobiusml.github.io/hqq_blog/

Bai et al. (2024)    Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024.

Longbench: A bilingual, multitask benchmark for long context understanding

https://doi.org/10.18653/V1/2024.ACL-LONG.172

Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024

, pages 3119–3137. Association for Computational Linguistics.

Barbero et al. (2024)    Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Velickovic. 2024.

Round and round we go! what makes rotary positional encodings useful?

https://doi.org/10.48550/ARXIV.2410.06205

, abs/2410.06205.

Bisk et al. (2020)    Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020.

PIQA: reasoning about physical commonsense in natural language

https://doi.org/10.1609/AAAI.V34I05.6239

The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020

, pages 7432–7439. AAAI Press.

Black et al. (2021)    Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021.

GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow

https://doi.org/10.5281/zenodo.5297715

.  If you use this software, please cite it using these metadata.

Choromanski et al. (2021)    Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. 2021.

## Rethinking attention with performers

https://openreview.net/forum?id=Ua6zuk0WRH

9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021

. OpenReview.net.

Clark et al. (2018)    Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.

Think you have solved question answering? try arc, the AI2 reasoning challenge

https://arxiv.org/abs/1803.05457

, abs/1803.05457.

DeepSeek-AI et al. (2024)    DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun. 2024.

Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model

https://doi.org/10.48550/ARXIV.2405.04434

, abs/2405.04434.

Frantar et al. (2022)    Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022.

GPTQ: accurate post-training quantization for generative pre-trained transformers

https://doi.org/10.48550/ARXIV.2210.17323

, abs/2210.17323.

Gu and Dao (2023)    Albert Gu and Tri Dao. 2023.

Mamba: Linear-time sequence modeling with selective state spaces

https://doi.org/10.48550/ARXIV.2312.00752

, abs/2312.00752.

Guo et al. (2019)    Qipeng Guo, Xipeng Qiu, Pengfei Liu, Yunfan Shao, Xiangyang Xue, and Zheng Zhang. 2019.

Star-transformer

https://doi.org/10.18653/V1/N19-1133

Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)

, pages 1315–1325. Association for Computational Linguistics.

Hendrycks et al. (2021)    Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021.

## Measuring massive multitask language understanding

https://openreview.net/forum?id=d7KBjmI3GmQ

9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021

. OpenReview.net.

Jo and Shin (2024)    Hyun-rae Jo and Dongkun Shin. 2024.

A2SF: accumulative attention scoring with forgetting factor for token pruning in transformer decoder

https://doi.org/10.48550/ARXIV.2407.20485

, abs/2407.20485.

Kaplan et al. (2020)    Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.

## Scaling laws for neural language models

https://arxiv.org/abs/2001.08361

, abs/2001.08361.

Katharopoulos et al. (2020)    Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020.

Transformers are rnns: Fast autoregressive transformers with linear attention

http://proceedings.mlr.press/v119/katharopoulos20a.html

Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event

, volume 119 of

## Proceedings of Machine Learning Research

, pages 5156–5165. PMLR.

Li et al. (2024)    Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024.

Snapkv: LLM knows what you are looking for before generation

http://papers.nips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html

Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024

Liu et al. (2024a)    Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Reza Haffari, and Bohan Zhuang. 2024a.

Minicache: KV cache compression in depth dimension for large language models

http://papers.nips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html

Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024

Liu et al. (2024b)    Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024b.

KIVI: A tuning-free asymmetric 2bit quantization for KV cache

https://openreview.net/forum?id=L057s2Rq8O

Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024

. OpenReview.net.

Mihaylov et al. (2018)    Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018.

Can a suit of armor conduct electricity? A new dataset for open book question answering

https://doi.org/10.18653/V1/D18-1260

Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018

, pages 2381–2391. Association for Computational Linguistics.

Peng et al. (2023)    Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan S. Wind, Stanislaw Wozniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023.

RWKV: reinventing rnns for the transformer era

https://doi.org/10.18653/V1/2023.FINDINGS-EMNLP.936

Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023

, pages 14048–14077. Association for Computational Linguistics.

Sakaguchi et al. (2021)    Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021.

Winogrande: an adversarial winograd schema challenge at scale

https://doi.org/10.1145/3474381

Commun. ACM

, 64(9):99–106.

Shazeer (2019)    Noam Shazeer. 2019.

Fast transformer decoding: One write-head is all you need

https://arxiv.org/abs/1911.02150

, abs/1911.02150.

Sheng et al. (2023)    Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023.

Flexgen: High-throughput generative inference of large language models with a single GPU

https://proceedings.mlr.press/v202/sheng23a.html

International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA

, volume 202 of

## Proceedings of Machine Learning Research

, pages 31094–31116. PMLR.

Strubell et al. (2019)    Emma Strubell, Ananya Ganesh, and Andrew McCallum. 2019.

## Energy and policy considerations for deep learning in NLP

https://doi.org/10.18653/v1/P19-1355

Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics

, pages 3645–3650, Florence, Italy. Association for Computational Linguistics.

Su et al. (2024)    Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024.

Roformer: Enhanced transformer with rotary position embedding

https://doi.org/10.1016/J.NEUCOM.2023.127063

## Neurocomputing

, 568:127063.

Sun et al. (2024)    Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024.

## A simple and effective pruning approach for large language models

https://openreview.net/forum?id=PxoFut3dWW

The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024

. OpenReview.net.

Vaswani et al. (2017)    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.

## Attention is all you need

https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA

, pages 5998–6008.

Xia et al. (2024)    Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024.

Sheared llama: Accelerating language model pre-training via structured pruning

https://openreview.net/forum?id=09iOdaeOzp

The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024

. OpenReview.net.

Zellers et al. (2019)    Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019.

Hellaswag: Can a machine really finish your sentence?

https://doi.org/10.18653/V1/P19-1472

Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers

, pages 4791–4800. Association for Computational Linguistics.

Zuhri et al. (2024)    Zayd Muhammad Kawakibi Zuhri, Muhammad Farid Adilazuarda, Ayu Purwarianti, and Alham Fikri Aji. 2024.

MLKV: multi-layer key-value heads for memory efficient transformer decoding

https://doi.org/10.48550/ARXIV.2406.09297

, abs/2406.09297.

Appendix A  The Calculation of 2-norm Score

To compute the 2-norm scores for each attention head, we selected 1,024 samples from the training dataset. The proportions of the subsets and sequence length used during the 2-norm computation are consistent with those used during fine-tuning. First, we calculate the query vectors and key vectors for each head. Then, for each rotational subspace of the vectors, we compute the 2-norm scores. Finally, the 2-norm scores of the query and key vectors are aggregated within each subspace. If the model employs Grouped-Query Attention (GQA), the 2-norm scores are averaged within each GQA group, and the scores are shared between the groups.

Appendix B  Inference Process of MHA2MLA

During inference in the MHA2MLA model, our input includes the hidden representation  x i subscript 𝑥 𝑖 x_{i} italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  of the  i 𝑖 i italic_i -th token, as well as the previously stored  𝒌 < i , rope ( h ) superscript subscript 𝒌 absent 𝑖 rope ℎ \bm{k}_{<i,\text{rope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT < italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝒄 < i , kv subscript 𝒄 absent 𝑖 kv \bm{c}_{<i,\text{kv}} bold_italic_c start_POSTSUBSCRIPT < italic_i , kv end_POSTSUBSCRIPT  in the KV cache for the first  i − 1 𝑖 1 i-1 italic_i - 1  tokens.

During the inference, our goal is to compute the  h ℎ h italic_h -th head’s dot product of these two parts  𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( h ) ⊤ superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from ℎ top \bm{q}_{i,\text{rope}}^{(h)}\bm{k}_{\leq i,\text{rope}}^{(h)\top} bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT  and  𝒒 i , nope ( h ) ⁢ 𝒌 ≤ i , nope ( h ) ⊤ superscript subscript 𝒒 𝑖 nope ℎ superscript subscript 𝒌 absent 𝑖 nope limit-from ℎ top \bm{q}_{i,\text{nope}}^{(h)}\bm{k}_{\leq i,\text{nope}}^{(h)\top} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT . For the RoPE part, we can easily extract  𝑾 q , rope ( h ) superscript subscript 𝑾 𝑞 rope ℎ \bm{W}_{q,\text{rope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝑾 k , rope ( h ) superscript subscript 𝑾 𝑘 rope ℎ \bm{W}_{k,\text{rope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_k , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  from the pre-trained parameter matrices  𝑾 q ( h ) superscript subscript 𝑾 𝑞 ℎ \bm{W}_{q}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝑾 k ( h ) superscript subscript 𝑾 𝑘 ℎ \bm{W}_{k}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  (i.e., the rows corresponding to the subspace that retains RoPE) and then obtain the result through a linear transformation:

𝒒 i , rope ( h ) superscript subscript 𝒒 𝑖 rope ℎ \displaystyle\bm{q}_{i,\text{rope}}^{(h)} bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒙 i ⁢ 𝑾 q , rope ( h ) absent subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 rope ℎ \displaystyle=\bm{x}_{i}\bm{W}_{q,\text{rope}}^{(h)} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   𝒌 i , rope ( h ) superscript subscript 𝒌 𝑖 rope ℎ \displaystyle\bm{k}_{i,\text{rope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒙 i ⁢ 𝑾 k , rope ( h ) absent subscript 𝒙 𝑖 superscript subscript 𝑾 𝑘 rope ℎ \displaystyle=\bm{x}_{i}\bm{W}_{k,\text{rope}}^{(h)} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_k , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   𝒌 ≤ i , rope ( h ) superscript subscript 𝒌 absent 𝑖 rope ℎ \displaystyle\bm{k}_{\leq i,\text{rope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = [ 𝒌 < i , rope ( h ) , 𝒌 i , rope ( h ) ] absent superscript subscript 𝒌 absent 𝑖 rope ℎ superscript subscript 𝒌 𝑖 rope ℎ \displaystyle=[\bm{k}_{<i,\text{rope}}^{(h)},~{}\bm{k}_{i,\text{rope}}^{(h)}] = [ bold_italic_k start_POSTSUBSCRIPT < italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , bold_italic_k start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ]   → → \displaystyle\to~{} →   𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( h ) ⊤ . superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from ℎ top \displaystyle\bm{q}_{i,\text{rope}}^{(h)}\bm{k}_{\leq i,\text{rope}}^{(h)\top}. bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT .

Note that  𝒌 < i , rope ( h ) superscript subscript 𝒌 absent 𝑖 rope ℎ \bm{k}_{<i,\text{rope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT < italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  is already stored in the KV cache and can be directly retrieved.

For the NoPE part,  𝒒 i , nope ( h ) superscript subscript 𝒒 𝑖 nope ℎ \bm{q}_{i,\text{nope}}^{(h)} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  can still be easily obtained through a linear transformation  𝑾 q , nope ( h ) superscript subscript 𝑾 𝑞 nope ℎ \bm{W}_{q,\text{nope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  which extracted from the pre-trained parameter matrix  𝑾 q ( h ) superscript subscript 𝑾 𝑞 ℎ \bm{W}_{q}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  by separating the rows corresponding to the subspace with RoPE removed. However,  𝒌 i , nope ( h ) superscript subscript 𝒌 𝑖 nope ℎ \bm{k}_{i,\text{nope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  requires two linear transformations: a  dimensionality reduction  transformation using  𝑾 d ⁢ k ⁢ v subscript 𝑾 𝑑 𝑘 𝑣 \bm{W}_{dkv} bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT , and a  dimensionality expansion  transformation using  𝑾 u ⁢ k ( h ) superscript subscript 𝑾 𝑢 𝑘 ℎ \bm{W}_{uk}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT . Note that  𝑾 d ⁢ k ⁢ v subscript 𝑾 𝑑 𝑘 𝑣 \bm{W}_{dkv} bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT  is shared across all heads in the current layer, and both  𝑾 d ⁢ k ⁢ v subscript 𝑾 𝑑 𝑘 𝑣 \bm{W}_{dkv} bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v end_POSTSUBSCRIPT  and  𝑾 u ⁢ k ( h ) superscript subscript 𝑾 𝑢 𝑘 ℎ \bm{W}_{uk}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  are constrained by the SVD decomposition of the pre-trained parameter matrices  𝑾 k , nope ( h ) superscript subscript 𝑾 𝑘 nope ℎ \bm{W}_{k,\text{nope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_k , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝑾 v ( h ) superscript subscript 𝑾 𝑣 ℎ \bm{W}_{v}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , preserving most of the pre-trained knowledge:

𝒒 i , nope ( h ) superscript subscript 𝒒 𝑖 nope ℎ \displaystyle\bm{q}_{i,\text{nope}}^{(h)} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒙 i ⁢ 𝑾 q , nope ( h ) absent subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 nope ℎ \displaystyle=\bm{x}_{i}\bm{W}_{q,\text{nope}}^{(h)} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   𝒄 i , k ⁢ v subscript 𝒄 𝑖 𝑘 𝑣 \displaystyle\bm{c}_{i,kv} bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT   = 𝒙 i ⁢ 𝑾 d ⁢ k ⁢ v , absent subscript 𝒙 𝑖 subscript 𝑾 𝑑 𝑘 𝑣 \displaystyle=\bm{x}_{i}\bm{W}_{dkv,} = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_d italic_k italic_v , end_POSTSUBSCRIPT   𝒌 i , nope ( h ) superscript subscript 𝒌 𝑖 nope ℎ \displaystyle\bm{k}_{i,\text{nope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒄 i , k ⁢ v ⁢ 𝑾 u ⁢ k ( h ) absent subscript 𝒄 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑘 ℎ \displaystyle=\bm{c}_{i,kv}\bm{W}_{uk}^{(h)} = bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   𝒌 < i , nope ( h ) superscript subscript 𝒌 absent 𝑖 nope ℎ \displaystyle\bm{k}_{<i,\text{nope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT < italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = 𝒄 < i , k ⁢ v ⁢ 𝑾 u ⁢ k ( h ) . absent subscript 𝒄 absent 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑘 ℎ \displaystyle=\bm{c}_{<i,kv}\bm{W}_{uk}^{(h)}. = bold_italic_c start_POSTSUBSCRIPT < italic_i , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT .

During inference, the NoPE part can also leverage the standard MLA matrix merging algorithm to reduce memory consumption:

𝒌 ≤ i , nope ( h ) superscript subscript 𝒌 absent 𝑖 nope ℎ \displaystyle\bm{k}_{\leq i,\text{nope}}^{(h)} bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   = [ 𝒄 < i , k ⁢ v , 𝒄 i , k ⁢ v ] ⁢ 𝑾 u ⁢ k ( h ) absent subscript 𝒄 absent 𝑖 𝑘 𝑣 subscript 𝒄 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑘 ℎ \displaystyle=[\bm{c}_{<i,kv},~{}\bm{c}_{i,kv}]\bm{W}_{uk}^{(h)} = [ bold_italic_c start_POSTSUBSCRIPT < italic_i , italic_k italic_v end_POSTSUBSCRIPT , bold_italic_c start_POSTSUBSCRIPT italic_i , italic_k italic_v end_POSTSUBSCRIPT ] bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT   𝒒 i , nope ( h ) ⁢ 𝒌 ≤ i , nope ( h ) ⊤ superscript subscript 𝒒 𝑖 nope ℎ superscript subscript 𝒌 absent 𝑖 nope limit-from ℎ top \displaystyle\bm{q}_{i,\text{nope}}^{(h)}\bm{k}_{\leq i,\text{nope}}^{(h)\top} bold_italic_q start_POSTSUBSCRIPT italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT   = ( 𝒙 i ⁢ 𝑾 q , nope ( h ) ) ⁢ ( 𝒄 ≤ i , k ⁢ v ⁢ 𝑾 u ⁢ k ( h ) ) ⊤ absent subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 nope ℎ superscript subscript 𝒄 absent 𝑖 𝑘 𝑣 superscript subscript 𝑾 𝑢 𝑘 ℎ top \displaystyle=(\bm{x}_{i}\bm{W}_{q,\text{nope}}^{(h)})(\bm{c}_{\leq i,kv}\bm{W% }_{uk}^{(h)})^{\top} = ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ) ( bold_italic_c start_POSTSUBSCRIPT ≤ italic_i , italic_k italic_v end_POSTSUBSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT   = 𝒙 i ⁢ ( 𝑾 q , nope ( h ) ⁢ 𝑾 u ⁢ k ( h ) ⊤ ) ⁢ 𝒄 ≤ i , k ⁢ v ⊤ . absent subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 nope ℎ superscript subscript 𝑾 𝑢 𝑘 limit-from ℎ top superscript subscript 𝒄 absent 𝑖 𝑘 𝑣 top \displaystyle=\bm{x}_{i}(\bm{W}_{q,\text{nope}}^{(h)}\bm{W}_{uk}^{(h)\top})\bm% {c}_{\leq i,kv}^{\top}. = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT ) bold_italic_c start_POSTSUBSCRIPT ≤ italic_i , italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT .

We can pre-multiply the parameter matrices  ( 𝑾 q , nope ( h ) ⁢ 𝑾 u ⁢ k ( h ) ⊤ ) superscript subscript 𝑾 𝑞 nope ℎ superscript subscript 𝑾 𝑢 𝑘 limit-from ℎ top (\bm{W}_{q,\text{nope}}^{(h)}\bm{W}_{uk}^{(h)\top}) ( bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT ) , and let  𝒄 i , q ( h ) = 𝒙 i ⁢ ( 𝑾 q , nope ( h ) ⁢ 𝑾 u ⁢ k ( h ) ⊤ ) superscript subscript 𝒄 𝑖 𝑞 ℎ subscript 𝒙 𝑖 superscript subscript 𝑾 𝑞 nope ℎ superscript subscript 𝑾 𝑢 𝑘 limit-from ℎ top \bm{c}_{i,q}^{(h)}=\bm{x}_{i}(\bm{W}_{q,\text{nope}}^{(h)}\bm{W}_{uk}^{(h)\top}) bold_italic_c start_POSTSUBSCRIPT italic_i , italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT ) . In the end, the output of MHA2MLA is as follows:

𝒗 i ( h ) = superscript subscript 𝒗 𝑖 ℎ absent \displaystyle\bm{v}_{i}^{(h)}= bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT =   𝒐 i ( h ) = Softmax ⁢ ( 𝒒 i , rope ( h ) ⁢ 𝒌 ≤ i , rope ( h ) ⊤ + 𝒄 i , q ( h ) ⁢ 𝒄 ≤ i , k ⁢ v ⊤ ) ⁢ 𝒄 ≤ i , k ⁢ v superscript subscript 𝒐 𝑖 ℎ Softmax superscript subscript 𝒒 𝑖 rope ℎ superscript subscript 𝒌 absent 𝑖 rope limit-from ℎ top superscript subscript 𝒄 𝑖 𝑞 ℎ superscript subscript 𝒄 absent 𝑖 𝑘 𝑣 top subscript 𝒄 absent 𝑖 𝑘 𝑣 \displaystyle\bm{o}_{i}^{(h)}\!=\!\text{Softmax}\!\left(\bm{q}_{i,\text{rope}}% ^{(h)}\bm{k}_{\leq i,\text{rope}}^{(h)\top}\!+\!\bm{c}_{i,q}^{(h)}\bm{c}_{\leq i% ,kv}^{\top}\!\right)\bm{c}_{\leq i,kv} bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT = Softmax ( bold_italic_q start_POSTSUBSCRIPT italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_k start_POSTSUBSCRIPT ≤ italic_i , rope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) ⊤ end_POSTSUPERSCRIPT + bold_italic_c start_POSTSUBSCRIPT italic_i , italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_c start_POSTSUBSCRIPT ≤ italic_i , italic_k italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ) bold_italic_c start_POSTSUBSCRIPT ≤ italic_i , italic_k italic_v end_POSTSUBSCRIPT   MHA2MLA ⁢ ( 𝒙 i ) = [ … , 𝒐 i ( h ) ⁢ 𝑾 u ⁢ v ( h ) , … ] ⁢ 𝑾 o . MHA2MLA subscript 𝒙 𝑖 … superscript subscript 𝒐 𝑖 ℎ superscript subscript 𝑾 𝑢 𝑣 ℎ … subscript 𝑾 𝑜 \displaystyle\text{MHA2MLA}(\bm{x}_{i})=\left[\dots,\bm{o}_{i}^{(h)}\bm{W}_{uv% }^{(h)},\dots\right]\bm{W}_{o}. MHA2MLA ( bold_italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = [ … , bold_italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT , … ] bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT .

Where  𝑾 u ⁢ v ( h ) superscript subscript 𝑾 𝑢 𝑣 ℎ \bm{W}_{uv}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_u italic_v end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  and  𝑾 o subscript 𝑾 𝑜 \bm{W}_{o} bold_italic_W start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT  can also perform matrix merging to make inference more economical.

Why doesn’t MHA2MLA perform low-rank representation on the query as DeepSeek does?

Firstly, we found that the economical inference of MLA is not affected even if  𝑾 q , nope ( h ) superscript subscript 𝑾 𝑞 nope ℎ \bm{W}_{q,\text{nope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  is not decomposed into a dimension-reducing matrix (e.g.,  𝑾 d ⁢ q subscript 𝑾 𝑑 𝑞 \bm{W}_{dq} bold_italic_W start_POSTSUBSCRIPT italic_d italic_q end_POSTSUBSCRIPT ) and a dimension-increasing matrix (e.g.,  𝑾 u ⁢ q ( h ) superscript subscript 𝑾 𝑢 𝑞 ℎ \bm{W}_{uq}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_u italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT ). Secondly, decomposing  𝑾 q , nope ( h ) superscript subscript 𝑾 𝑞 nope ℎ \bm{W}_{q,\text{nope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  introduces additional architectural migration loss (approximation loss) and further reduces the number of LLM parameters. Therefore, we believe there is no need to decompose  𝑾 q , nope ( h ) superscript subscript 𝑾 𝑞 nope ℎ \bm{W}_{q,\text{nope}}^{(h)} bold_italic_W start_POSTSUBSCRIPT italic_q , nope end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_h ) end_POSTSUPERSCRIPT  within the MHA2MLA framework.

Appendix C  The Details of Fine-tuning

We fine-tune our model using the pretraining corpus from SmolLM

https://huggingface.co/blog/smollm

https://huggingface.co/blog/smollm

. The dataset consists of fineweb-edu-dedup, cosmopedia-v2, python-edu, open-web-math, and StackOverflow. The first three datasets are part of the smollm-corpus

https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

curated by HuggingFaceTB. Fineweb-edu-dedup is a high-quality dataset filtered by HuggingFaceTB from education-related webpages. Similarly, HuggingFaceTB filtered Python code snippets from The Stack to construct the python-edu dataset. Cosmopedia-v2 is a high-quality dataset generated by a model based on 34,000 topics defined by BISAC book classifications. Additionally, open-web-math

https://huggingface.co/datasets/open-web-math/open-web-math

https://huggingface.co/datasets/open-web-math/open-web-math

and StackOverflow

https://huggingface.co/datasets/bigcode/stackoverflow-clean

https://huggingface.co/datasets/bigcode/stackoverflow-clean

are sourced from high-quality mathematical texts available online and posts from StackOverflow, respectively.

## Hyperparameters

Metrics   135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   360M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   7B Llama2 Llama2 {}_{\text{Llama2}} start_FLOATSUBSCRIPT Llama2 end_FLOATSUBSCRIPT   n_batch  × \times ×  n_gpu   16 × \times × 4   16 × \times × 4   32 × \times × 8   16 × \times × 16   Learning Rate   1e-4   1e-4   1e-4   1e-4   Hardware   RTX3090   RTX3090   NVIDIA L20Y   NVIDIA L20Y   Steps   18000   18000   12000   12000   Warmup ratio   5.0%   5.0%   8.3%   8.3%   Decay   10%   10%   16.7%   16.7%   Time   6h   12h   16h   28h   Seqlen   2048   2048   2048   4096   #Param.   d k ⁢ v = 128 / 256 † subscript 𝑑 𝑘 𝑣 128 superscript 256 † d_{kv}\!=\!128/256^{\dagger} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 128 / 256 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT   134.52M   361.82M   1.71B   6.61B

d k ⁢ v = 32 / 64 † subscript 𝑑 𝑘 𝑣 32 superscript 64 † d_{kv}\!=\!32/64^{\dagger} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 32 / 64 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT   130.99M   351.38M   1.67B   6.37B

d k ⁢ v = 16 / 32 † subscript 𝑑 𝑘 𝑣 16 superscript 32 † d_{kv}\!=\!16/32^{\dagger} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 16 / 32 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT   129.64M   347.38M   1.59B   5.99B

d k ⁢ v = 8 / 16 † subscript 𝑑 𝑘 𝑣 8 superscript 16 † d_{kv}\!=\!8/16^{\dagger} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT = 8 / 16 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT   128.97M   345.39M   1.56B   5.79B

Table 4:  Training detail information across different models.

The fine-tuning hyperparameters for models of all sizes are listed in

https://arxiv.org/html/2502.14837v1#A3.T4

. The training process employs a warmup phase followed by a decay strategy. A 1-sqrt decay strategy is applied to ensure a smooth and gradual reduction.

Model   Avg.   MMLU   ARC   PIQA   HS   OBQA   WG   135M   r 𝑟 r italic_r =32   44.25   29.82   42.05   68.34   41.03   33.20   51.07   - NoPE   r 𝑟 r italic_r =0   38.99

27.03   34.23   62.68   31.89   29.40   48.70   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT   r 𝑟 r italic_r =2   42.86

29.58   40.91   66.54   38.48   32.00   49.64   r 𝑟 r italic_r =4   43.40

29.90   41.15   66.92   39.34   32.60   50.51   r 𝑟 r italic_r =8   43.56

29.90   40.89   67.63   40.41   32.20   50.36   -  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT   r 𝑟 r italic_r =2   37.94

26.95   33.56   60.28   31.51   27.80   47.51   r 𝑟 r italic_r =4   37.76

27.11   32.06   59.79   30.68   28.40   48.54   r 𝑟 r italic_r =8   42.54

29.34   39.58   67.36   37.86   32.00   49.09   -  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT   r 𝑟 r italic_r =2   43.16

29.89   41.80   66.27   38.78   32.40   49.80   r 𝑟 r italic_r =4   43.76

29.87   41.29   67.36   40.22   32.80   50.99   r 𝑟 r italic_r =8   43.74

29.95   40.81   67.19   40.47   32.60   51.38   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT   r 𝑟 r italic_r =2   43.13

29.75   40.13   67.25   39.03   32.80   49.80   r 𝑟 r italic_r =4   43.77

30.14   41.69   67.57   39.53   33.00   50.67   r 𝑟 r italic_r =8   43.88

29.91   41.35   67.74   40.40   33.40   50.51   Table 5:  The impact of positional encoding dimensionality on model performance.

Appendix D  Ablation Study on Partial-RoPE Dimensions

To better determine the strategy and dimensionality for partial-RoPE, we conducted an ablation study on the number of RoPE dimensions using the 135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT  model. The experimental results are presented in

https://arxiv.org/html/2502.14837v1#A3.T5

. By comparing the performance of four different strategies in varying dimensionalities, we observed that the low-frequency strategy,  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , suffered significant performance degradation (-14.7%) when the dimensionality was relatively low ( ≤ 4 absent 4 \leq 4 ≤ 4 ). In contrast, both  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  and  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  consistently demonstrated superior performance regardless of dimensionality. Furthermore, increasing the dimensionality from 4 to 8 provided negligible performance gains. Based on these findings, we selected a dimensionality of 4 for partial-RoPE.

## Appendix E  Detailed Results

d k ⁢ v subscript 𝑑 𝑘 𝑣 d_{kv} italic_d start_POSTSUBSCRIPT italic_k italic_v end_POSTSUBSCRIPT   Precision   KV   Avg.   S-Doc QA   M-Doc QA   Summ.   Few-shot   Synth.   Code   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   7B Llama2 Llama2 {}_{\text{Llama2}} start_FLOATSUBSCRIPT Llama2 end_FLOATSUBSCRIPT  (Length=4K)   BF16   100.0%   27.4   15.1   9.6   21.1   7.5   9.7   3.7   26.7   20.5   3.2   65.5   87.5   34.1   1.9   6.6   66.5   59.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -75.00%   27.5   16.1   9.1   22.0   7.3   9.9   3.6   26.5   21.1   3.4   65.5   87.2   34.3   1.5   6.7   66.0   59.9   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   27.3   14.4   9.5   20.5   7.5   9.7   3.5   25.8   20.7   3.1   65.5   87.7   34.3   1.4   7.3   66.8   59.3   Int2 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -87.50%   21.2   18.0   5.5   12.6   7.5   8.4   3.2   12.6   18.6   0.9   56.5   73.3   27.0   1.8   6.1   34.5   52.9   Int2 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   18.5   9.4   6.2   12.7   6.8   6.7   3.3   5.9   17.2   0.4   61.0   63.9   26.0   1.4   2.7   42.4   30.5   64 64 64 64   BF16   -68.75%   27.1   13.3   9.6   23.2   7.2   10.9   3.5   24.6   20.0   22.1   62.5   83.5   32.4   0.9   8.7   56.9   53.7   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -92.19%   26.9   13.4   9.1   25.6   7.3   10.2   3.4   24.6   20.0   20.9   62.5   83.8   32.3   0.6   9.6   55.3   52.7   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   26.8   13.8   9.2   24.6   7.4   10.5   3.5   24.6   19.8   21.4   62.0   84.3   31.8   1.2   7.5   56.1   51.8   32 32 32 32   BF16   -81.25%   26.3   14.9   9.1   27.0   7.3   9.9   3.1   24.6   19.1   22.5   60.5   81.6   26.9   0.0   8.2   53.4   52.6   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -95.31%   26.1   14.7   9.5   26.6   7.9   10.7   3.4   23.6   19.0   20.5   60.5   80.8   28.3   0.0   7.6   51.9   52.0   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   26.1   14.7   9.5   26.6   7.9   10.7   3.4   23.6   19.0   20.5   60.5   80.8   28.3   0.0   7.6   51.9   52.0   16 16 16 16   BF16   -87.50%   24.4   14.7   9.5   24.3   7.8   10.2   3.8   22.8   19.1   24.6   61.0   82.8   20.2   0.2   8.6   39.9   41.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -96.87%   24.2   15.2   9.4   25.2   7.4   10.2   3.9   22.9   19.8   20.6   61.0   82.5   21.7   0.1   9.0   38.0   41.2   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   23.4   15.6   8.4   22.7   7.3   10.2   3.8   20.2   18.7   18.6   61.0   81.9   21.7   0.5   8.0   36.9   38.3   1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT  (Length=2K)   BF16   100.0%   18.7   2.6   6.3   19.9   5.4   8.6   2.7   23.5   18.4   20.2   46.5   70.2   32.4   2.2   3.2   21.3   16.5   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -75.00%   18.6   2.5   6.2   19.1   5.5   8.2   2.7   23.4   18.3   20.0   46.5   69.4   32.1   2.7   3.2   21.5   16.0   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   18.6   2.6   6.2   17.4   5.1   8.6   2.6   23.0   18.1   20.1   46.0   70.2   31.9   2.9   3.6   21.9   16.7   Int2 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -87.50%   16.3   2.5   5.6   13.0   4.8   7.5   2.7   14.8   16.3   9.3   46.0   70.4   26.9   2.6   3.4   18.3   16.8   Int2 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   13.3   1.6   3.8   10.3   3.9   7.3   1.4   5.9   13.4   6.3   40.0   64.3   14.6   3.1   3.5   15.6   17.5   32 32 32 32   BF16   -68.75%   16.0   2.6   6.1   16.9   4.6   9.3   2.0   22.8   15.1   19.9   50.0   57.1   29.8   1.7   2.4   9.4   6.7   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -92.19%   15.9   2.7   5.7   16.3   5.0   8.5   1.8   23.0   15.0   18.5   50.0   56.2   30.2   1.8   3.2   10.0   6.8   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   15.4   2.5   5.7   16.1   5.7   8.7   2.1   20.9   13.8   17.6   50.0   55.0   29.5   1.7   2.8   9.6   5.4   16 16 16 16   BF16   -81.25%   16.5   2.6   6.2   17.2   4.5   9.7   2.1   22.0   15.3   21.0   47.5   55.5   31.7   1.2   3.3   15.8   8.5   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -95.31%   16.2   2.5   6.1   16.2   4.5   8.9   2.0   20.6   15.4   19.7   47.5   55.6   30.6   1.2   4.0   16.3   8.0   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   15.6   2.5   5.7   15.6   4.3   8.8   1.6   21.2   15.7   17.6   47.0   55.7   27.4   1.7   3.6   15.6   6.2   8 8 8 8   BF16   -87.50%   15.3   2.4   5.9   17.9   4.8   10.1   1.8   25.1   15.2   20.6   42.5   49.0   31.4   2.7   3.3   7.1   4.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -96.87%   15.0   2.4   5.7   16.9   4.7   10.1   2.0   23.5   14.7   20.3   42.5   47.6   30.6   2.6   3.6   7.7   4.5   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   14.2   2.7   5.4   16.9   4.1   8.8   1.5   22.2   14.4   17.2   42.0   47.9   29.9   1.5   3.3   7.0   3.0   360M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT  (Length=2K)   BF16   100.0%   13.5   2.4   6.4   14.3   5.0   8.8   2.5   18.0   17.5   7.1   47.5   37.5   24.9   1.5   3.4   8.1   10.4   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -75.00%   13.4   2.7   6.1   14.1   5.5   8.4   3.0   16.2   15.4   11.2   47.5   37.5   23.4   1.3   3.7   9.0   10.1   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   13.3   2.4   6.2   13.7   5.4   8.7   2.6   15.4   17.4   7.3   47.5   37.3   24.4   1.0   3.7   8.4   11.0   Int2 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -87.50%   10.8   2.7   4.7   8.3   5.4   5.9   1.9   9.9   10.0   8.4   45.2   27.5   14.2   2.1   4.2   10.0   11.9   Int2 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   8.6   2.6   2.2   4.4   3.9   4.8   1.4   5.6   8.9   2.9   44.0   26.8   9.6   1.0   1.9   7.2   9.7   32 32 32 32   BF16   -68.75%   13.5   2.3   5.9   13.4   5.5   9.8   2.7   20.4   14.5   11.5   41.0   31.2   29.6   1.2   3.5   15.4   7.9   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -92.19%   12.5   2.6   5.7   12.1   5.1   10.2   2.7   14.6   12.5   8.8   41.0   30.3   27.8   1.9   2.7   14.5   7.6   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   12.3   2.0   5.2   11.9   5.0   9.1   3.0   15.4   14.9   8.3   41.0   28.3   27.0   0.9   3.9   13.8   7.8   16 16 16 16   BF16   -81.25%   11.6   2.2   5.2   13.0   4.8   9.5   3.2   13.4   13.4   11.3   32.0   26.1   22.5   1.1   5.0   14.9   7.7   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -95.31%   11.2   2.6   5.6   12.0   5.1   8.8   2.9   13.4   12.4   10.8   32.0   24.8   21.8   2.1   3.7   14.0   7.2   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   10.9   1.9   4.9   11.5   4.2   8.8   2.6   12.2   12.2   9.5   32.5   25.8   18.5   1.4   4.6   15.5   7.8   8 8 8 8   BF16   -87.50%   9.9   1.9   4.7   11.7   4.5   8.5   2.8   13.0   12.9   9.4   34.0   17.2   15.3   1.4   3.2   11.4   6.9   Int4 HQQ HQQ {}_{\text{HQQ}} start_FLOATSUBSCRIPT HQQ end_FLOATSUBSCRIPT   -96.87%   10.0   2.2   4.8   11.0   4.2   8.2   2.6   13.1   12.8   11.7   33.5   17.3   14.8   0.8   4.4   11.6   7.4   Int4 Quanto Quanto {}_{\text{Quanto}} start_FLOATSUBSCRIPT Quanto end_FLOATSUBSCRIPT   9.3   1.8   3.6   11.3   4.0   8.0   3.0   10.6   12.0   7.4   31.5   19.8   10.3   0.8   4.8   12.1   7.6   Table 6:  Evaluation results of all models on LongBench, including Task A: narrativeqa, B: qasper, C: multifieldqa_en, D: hotpotqa, E: 2wikimqa, F: musique, G: gov_report, H: qmsum, I: multi_news, J: trec, K: triviaqa, L: samsum, M: passage_count, N: passage_retrieval_en, O: lcc, P: repobench-p.  Bold  indicates compression ratios greater than or equal to Int2 quantization while also achieving performance higher than Int2.   Model   Tokens   Avg@CS   MMLU   ARC   PIQA   HS   OBQA   WG   135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   600B   44.50   29.80   42.43   68.06   41.09   33.60   52.01   - full-rope   2.25B   44.25   29.82   42.05   68.34   41.03   33.20   51.07   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT   43.40

29.90   41.15   66.92   39.34   32.60   50.51   -  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT   37.76

27.11   32.06   59.79   30.68   28.40   48.54   -  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT   43.76

29.87   41.29   67.36   40.22   32.80   50.99   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT   43.77

30.14   41.69   67.57   39.53   33.00   50.67   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT  + SVD

2.25B   41.04

28.16   37.55   64.91   34.91   32.00   48.70   -  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  + SVD

28.58   38.69   65.67   36.17   32.00   49.49   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD

28.66   39.95   65.02   36.04   31.60   49.80   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD

28.04   37.85   65.56   34.60   29.8   49.65   1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT   1T   55.90   39.27   59.87   75.73   62.93   42.80   54.85   - full-rope   6B   55.93   39.11   59.19   75.95   62.92   43.40   55.09   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT   55.17

38.56   57.72   75.73   60.93   44.00   54.06   -  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT   54.72

37.82   56.47   75.35   60.06   43.20   55.41   -  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT   55.31

38.93   57.93   75.63   61.97   42.60   54.85   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT   55.10

38.60   57.36   75.68   61.77   43.00   54.22   -  𝒮 high subscript 𝒮 high \mathcal{S}_{\text{high}} caligraphic_S start_POSTSUBSCRIPT high end_POSTSUBSCRIPT  + SVD

6B   54.41

37.97   56.74   75.14   59.75   42.00   54.85   -  𝒮 uniform subscript 𝒮 uniform \mathcal{S}_{\text{uniform}} caligraphic_S start_POSTSUBSCRIPT uniform end_POSTSUBSCRIPT  + SVD

37.82   56.30   75.08   60.35   42.40   53.91   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD

37.87   56.81   75.84   60.41   42.60   54.38   -  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  + SVD

37.64   55.50   75.46   59.52   42.40   52.96   Table 7:  The complete results of the ablation experiment.   Figure 6:  The fine-tuning loss curves under different KV cache storage ratios (with colors ranging from light to dark representing 12.5%, 18.75%, 31.25%, and 100%).   Figure 7:  The fine-tuning loss curves under different partial-RoPE strategy.   Figure 8:  The fine-tuning loss curves under the combination of  𝒮 2-norm subscript 𝒮 2-norm \mathcal{S}_{\text{2-norm}} caligraphic_S start_POSTSUBSCRIPT 2-norm end_POSTSUBSCRIPT  and different SVD strategies.

In this section, we present the detailed results.

## Detailed LongBench evaluation

is reported in

https://arxiv.org/html/2502.14837v1#A5.T6

## Detailed ablation experiment

is reported in

https://arxiv.org/html/2502.14837v1#A5.T7

Additional visualizations of fine-tuning loss

We present the loss of the other two models fine-tuned, excluding the ones mentioned in the main text, in

https://arxiv.org/html/2502.14837v1#A5.F6

. We observe that as fine-tuning progresses, the gap in loss between our approach and the baseline gradually decreases, and both exhibit similar fluctuations, demonstrating the effectiveness of our approach. In

https://arxiv.org/html/2502.14837v1#A5.F7

, we show the loss under different partial-RoPE strategies. Except for  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , the other three partial-RoPE schemes show little difference from the baseline. Additionally,  𝒮 low subscript 𝒮 low \mathcal{S}_{\text{low}} caligraphic_S start_POSTSUBSCRIPT low end_POSTSUBSCRIPT  has a higher probability of convergence failure. In

https://arxiv.org/html/2502.14837v1#A5.F8

, we show the loss under different SVD strategies. The loss curves on both 1B7 SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT  and 135M SmolLM SmolLM {}_{\text{SmolLM}} start_FLOATSUBSCRIPT SmolLM end_FLOATSUBSCRIPT  reveal that SVD

outperforms SVD

