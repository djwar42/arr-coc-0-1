---
sourceFile: "Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:16.344Z"
---

# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

8a8d9072-bf9e-4869-81e0-31981aeedf16

Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

2c833dd5-fc72-4054-80f7-a26afb323a63

https://arxiv.org/html/2505.22038v2

Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

, Xiaoyue Chen

, Chen Gao

, Xinlei Chen

## Tsinghua Shenzhen International Graduate School

BNRist, Tsinghua University  {likaiyua23,chenxiao24}@mails.tsinghua.edu.cn   {chgao96,liyong07}@tsinghua.edu.cn,chen.xinlei@sz.tsinghua.edu.cn   Equal Contribution.Corresponding author.

Large Vision-Language Models (LVLMs) have shown impressive performance across multi-modal tasks by encoding images into thousands of tokens. However, the large number of image tokens results in significant computational overhead, and the use of dynamic high-resolution inputs further increases this burden. Previous approaches have attempted to reduce the number of image tokens through token pruning, typically by selecting tokens based on attention scores or image token diversity. Through empirical studies, we observe that existing methods often overlook the joint impact of pruning on both the current layer’s output (local) and the outputs of subsequent layers (global), leading to suboptimal pruning decisions. To address this challenge, we propose Balanced Token Pruning (BTP), a plug-and-play method for pruning vision tokens. Specifically, our method utilizes a small calibration set to divide the pruning process into multiple stages. In the early stages, our method emphasizes the impact of pruning on subsequent layers, whereas in the deeper stages, the focus shifts toward preserving the consistency of local outputs. Extensive experiments across various LVLMs demonstrate the broad effectiveness of our approach on multiple benchmarks. Our method achieves a 78% compression rate while preserving 96.7% of the original models’ performance on average. Our code is available at

https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning

https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning

1  Introduction

Recent advances in Large Vision-Language Models (LVLMs)

chen2024internvl

https://arxiv.org/html/2505.22038v2#bib.bib8

gao2024mini

https://arxiv.org/html/2505.22038v2#bib.bib15

liu2023improvedllava

https://arxiv.org/html/2505.22038v2#bib.bib27

liu2023llava

https://arxiv.org/html/2505.22038v2#bib.bib29

wang2024qwen2

https://arxiv.org/html/2505.22038v2#bib.bib43

zha2025enablellm3dcapacity

https://arxiv.org/html/2505.22038v2#bib.bib52

have substantially improved visual understanding. These models typically employ a visual encoder to convert images into discrete tokens, which are then processed jointly with textual tokens by a large language model backbone. The incorporation of visual information significantly increases the total number of input tokens

bai2025qwen2

https://arxiv.org/html/2505.22038v2#bib.bib2

liu2024llavanext

https://arxiv.org/html/2505.22038v2#bib.bib28

zhao2025embodiedr

https://arxiv.org/html/2505.22038v2#bib.bib59

, a problem further amplified when handling high-resolution images. In edge applications such as emergency monitoring

https://arxiv.org/html/2505.22038v2#bib.bib7

wu2025monitorvlmavisionlanguageframework

https://arxiv.org/html/2505.22038v2#bib.bib44

, logistics

chen2024ddl

https://arxiv.org/html/2505.22038v2#bib.bib6

zhang2025logisticsvlnvisionlanguagenavigationlowaltitude

https://arxiv.org/html/2505.22038v2#bib.bib56

, and smart homes

10.1145/3706598.3713265

https://arxiv.org/html/2505.22038v2#bib.bib51

, models are typically deployed on devices like drones and unmanned vehicles

jian2023pathgenerationwheeledrobots

https://arxiv.org/html/2505.22038v2#bib.bib23

cui2024onboardvisionlanguagemodelspersonalized

https://arxiv.org/html/2505.22038v2#bib.bib10

, which are constrained by limited memory and strict latency requirements. The excessive number of image tokens poses a major bottleneck for deployment, drawing increasing research interest in accelerating edge inference

ruan2025edmambarethinkingefficientevent

https://arxiv.org/html/2505.22038v2#bib.bib38

## Prior studies

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

have demonstrated that visual tokens often exhibit significant redundancy

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

. Consequently, visual token pruning has been proposed as an effective strategy to reduce input redundancy and enhance computational efficiency

yang2024visionzip

https://arxiv.org/html/2505.22038v2#bib.bib47

shang2024llava

https://arxiv.org/html/2505.22038v2#bib.bib40

huang2024dynamic

https://arxiv.org/html/2505.22038v2#bib.bib18

zhang2025llava

https://arxiv.org/html/2505.22038v2#bib.bib53

ye2024atp

https://arxiv.org/html/2505.22038v2#bib.bib48

. Visual token pruning faces two fundamental challenges: identifying the most important visual tokens and determining the appropriate layers for pruning. Existing token pruning strategies can be broadly classified into two categories: attention-based methods that leverage text-image interactions

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

meng2025plphpperlayerperheadvision

https://arxiv.org/html/2505.22038v2#bib.bib32

, and diversity-based methods that exploit the heterogeneity of visual representations

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

. However, the impact of their distinct optimization objectives on overall model performance remains underexplored, and a systematic comparison between them is largely absent. Moreover, when it comes to pruning layer selection, existing methods rely heavily on validation performance and manually defined settings, lacking principled guidance based on the model’s intrinsic properties.

Figure 1 :  Layer-wise visualization of attention in LVLMs.

To address these problems, we first explore the nature of image token pruning from an intuitive perspective: its impact on the  current layer’s (local)  output and its influence on the outputs of  subsequent pruning layers (global) . We begin by visualizing the spatial distribution of image tokens that receive higher attention from text tokens across different layers. As shown in Figure

https://arxiv.org/html/2505.22038v2#S1.F1

, we observe that the image tokens attended by text tokens vary across different layers. This indicates that pruning solely based on the current layer tends to overlook its impact on subsequent layers. Then we further investigate the impact of different pruning methods on the model outputs. Specifically, we compare the hidden states of output tokens at different decoding positions under two pruning methods with those of the original model.

Figure 2 :  Impact of different pruning strategies on layer-wise representations.

## It can be found in Figure

https://arxiv.org/html/2505.22038v2#S1.F2

that attention-based methods preserve output similarity well at early pruning layers, but the error accumulates in deeper layers. In contrast, diversity-based methods do not maintain output similarity at the initial layers, but achieve better consistency in later pruning stages. This implies that attention-based pruning methods focus solely on optimizing the current pruning layer while ignoring their impact on subsequent layers, whereas diversity-based methods overlook the preservation of output quality at the current layer.

Motivated by the above observation, we aim to tackle a fundamental challenge:  how to prune with joint consideration of the current and subsequent layers to achieve global optimality.  To address this challenge, we propose  Balanced Token Pruning (BTP) , a visual token pruning method that balances local objectives (current layer) with global objectives (subsequent layers). We begin by analyzing and formulating a local-global objective for image token pruning. Based on this objective, BTP first partitions the pruning process into multiple stages using a small calibration set

ragin2007fuzzy

https://arxiv.org/html/2505.22038v2#bib.bib35

hubara2021accurate

https://arxiv.org/html/2505.22038v2#bib.bib20

, leveraging the way LVLMs process images, as illustrated in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

. In early stages, where more image tokens are retained, BTP emphasizes a diversity-based objective to preserve the quality of downstream representations. In later stages, where fewer tokens are retained, it prioritizes an attention-based objective to maintain the consistency of local outputs. With this design, we preserve token diversity in the early layers while focusing on task-relevant tokens in the later layers.

Extensive experiments demonstrate the effectiveness of our proposed BTP method. We evaluate BTP across models of varying sizes and architectures, consistently achieving superior performance under higher compression ratios. Notably, our approach retains only 22% of the original image tokens on average while preserving 98% of the model’s original performance. Furthermore, end-to-end efficiency evaluations confirm that BTP significantly reduces both inference latency and memory usage in practice.

2  Related work

2.1  Large Vision-Language Models

Recent progress in large vision language models (LVLMs) has been substantially accelerated by the open-sourcing of foundation models like LLaMA

touvron2023llama

https://arxiv.org/html/2505.22038v2#bib.bib41

and Vicuna

zheng2023judging

https://arxiv.org/html/2505.22038v2#bib.bib60

. Representative models, including LLaVA

liu2023improvedllava

https://arxiv.org/html/2505.22038v2#bib.bib27

liu2024llavanext

https://arxiv.org/html/2505.22038v2#bib.bib28

liu2023llava

https://arxiv.org/html/2505.22038v2#bib.bib29

bai2025qwen2

https://arxiv.org/html/2505.22038v2#bib.bib2

wang2024qwen2

https://arxiv.org/html/2505.22038v2#bib.bib43

, and InternVL

chen2024internvl

https://arxiv.org/html/2505.22038v2#bib.bib8

gao2024mini

https://arxiv.org/html/2505.22038v2#bib.bib15

leverage vision encoders

radford2021learningtransferablevisualmodels

https://arxiv.org/html/2505.22038v2#bib.bib34

li2022blipbootstrappinglanguageimagepretraining

https://arxiv.org/html/2505.22038v2#bib.bib25

choraria2024semanticallygroundedqformerefficient

https://arxiv.org/html/2505.22038v2#bib.bib9

to encode images into visual tokens, which are then integrated into the language model for unified multimodal representation and understanding

gao2024embodiedcitybenchmarkplatformembodied

https://arxiv.org/html/2505.22038v2#bib.bib14

. For example, LLaVA-1.5 encodes image into 576 visual tokens using a single-scale encoder. As these models increasingly support high-resolution visual inputs

bai2025qwen2

https://arxiv.org/html/2505.22038v2#bib.bib2

liu2024llavanext

https://arxiv.org/html/2505.22038v2#bib.bib28

, the number of visual tokens grows. Using a multi-resolution encoding strategy, LLaVA-NeXT can generate up to 2,880 tokens per image. Multimodal large models have been widely applied in various scenarios, including embodied agent

https://arxiv.org/html/2505.22038v2#bib.bib17

. The large number of image tokens limits its applicability in scenarios such as real-time applications

ruan2025premamba4dstatespace

https://arxiv.org/html/2505.22038v2#bib.bib39

2.2  Visual Token Pruning

Early efforts to reduce visual token redundancy primarily focus on attention-based pruning

chen2024efficient

https://arxiv.org/html/2505.22038v2#bib.bib4

huang2025dynamicllavaefficientmultimodallarge

https://arxiv.org/html/2505.22038v2#bib.bib19

zhang2025llavaminiefficientimagevideo

https://arxiv.org/html/2505.22038v2#bib.bib54

meng2025plphpperlayerperheadvision

https://arxiv.org/html/2505.22038v2#bib.bib32

. For example, FastV

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

prunes visual tokens with low attention scores after the filtering layer, with subsequent layers processing only the remaining token. Another approach, VTW

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

, adopts a complete token elimination strategy, removing all visual tokens after a specified layer. PyramidDrop

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

introduces a more sophisticated approach, performing stage-wise pruning throughout the transformer, ranking visual tokens by their attention scores to the instruction token at each stage and progressively discarding the least informative ones. Compared to attention-based methods, diversity-based methods prioritize retaining a richer variety of semantic information. For instance, DivPrune

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

formulate token pruning as a Max-Min Diversity Problem

porumbel2011simple

https://arxiv.org/html/2505.22038v2#bib.bib33

resende2010grasp

https://arxiv.org/html/2505.22038v2#bib.bib37

. Additionally, some methods fuse remaining tokens into retained tokens through token fusion such as LLaVA-PruMerge

shang2024llava

https://arxiv.org/html/2505.22038v2#bib.bib40

and VisionZip

yang2024visionzip

https://arxiv.org/html/2505.22038v2#bib.bib47

. Different from prior methods, our method jointly considers the impact of pruning on both the current layer and subsequent layers.

3  Preliminary

3.1  Visual token processing

In the prefilling stage, images and texts are first encoded into embedding vectors (tokens), which are then processed by LVLM. We denote the input token sequence as  𝐗 \mathbf{X}  which consists of the system prompt  𝐗 S \mathbf{X}_{S} , the image tokens  𝐗 I \mathbf{X}_{I}  and text tokens  𝐗 T \mathbf{X}_{T} ,  𝐗 \mathbf{X}  = ( 𝐗 S \mathbf{X}_{S} , 𝐗 I \mathbf{X}_{I} , 𝐗 T \mathbf{X}_{T} ) .  𝐗 \mathbf{X}  is then fed into the LLM backbone composed of  N  decoder layers. For the  l -th decoder layer, we denote the input as  𝐗 ( l ) \mathbf{X}^{(l)}  and the layer output  𝐗 ( l + 1 ) \mathbf{X}^{(l+1)}  is:

𝐗 ( l + 1 ) = 𝐗 ( l ) + A  t  t  e  n ( l )  ( L  N  ( 𝐗 ( l ) ) ) + MLP ( l )  ( L  N  ( a  t  t  n o  u  t  p  u  t ( l ) + 𝐗 ( l ) ) ) , \mathbf{X}^{(l+1)}=\mathbf{X}^{(l)}+Atten^{(l)}(LN(\mathbf{X}^{(l)}))+\text{MLP}^{(l)}(LN(attn_{output}^{(l)}+\mathbf{X}^{(l)})),   (1)

where  A  t  t  e  n ( l ) Atten^{(l)}  is the attention block,  L  N LN  is the layer normalization and  M  L  P ( l ) MLP^{(l)}  is the projector layer. It can be observed that the outputs of the attention block and the MLP block are closely tied to the attention mechanism

vaswani2017attention

https://arxiv.org/html/2505.22038v2#bib.bib42

. Formally, the attention mechanism can be represent as:

a  t  t  n o  u  t  p  u  t l = S  o  f  t  m  a  x  ( Q l  ( K l ) T + M D k )  V l , attn_{output}^{l}=Softmax(\frac{Q_{l}(K_{l})^{T}+M}{\sqrt{D_{k}}})V_{l},   (2)

where  Q l Q_{l} ,  K l K_{l} ,  V l V_{l}  are calculated by Query projector, Key projector and Value projector.  D k D_{k}  is hidden state dimension.  M M  is the casual mask which imposes a constraint such that each token is permitted to incorporate information only from tokens at earlier positions.  K l K_{l} ,  V l V_{l}  are stored in the KV cache for further decoding stage.

3.2  Visual token pruning formulations

Prior works on image token pruning can be broadly categorized into attention-based methods

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

and diversity-based methods

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

. Attention based methods utilize text-image attention score to select important image tokens at specific layers. For input sample with  m m  text tokens, we can denote the importance score  S i  m  g S_{img}  of image tokens at  l l -th layer as:

S i  m  g ( l ) = 1 m  ∑ i = 1 m A  t  t  e  n ( l )  ( 𝐗 I , 𝐗 T ( i ) ) . S_{img}^{(l)}=\frac{1}{m}\sum_{i=1}^{m}Atten^{(l)}(\mathbf{X}_{I},\mathbf{X}_{T}^{(i)}).   (3)

After obtaining the importance scores of the image tokens, these methods select a pruned image token set  ℙ a  t  t  e  n ⊂ 𝐗 I \mathbb{P}_{atten}\subset\mathbf{X}_{I}  with the highest scores. In contrast to attention score-based methods, diversity-based approaches focus on maximizing the diversity among selected image tokens. These methods are typically based either on the spatial diversity of the selected image token set or on the semantic diversity of the selected images. Formally, given a diversity metric  ℱ ⊂ { ℱ s  p  a , ℱ s  e  m } \mathcal{F}\subset\{\mathcal{F}_{spa},\mathcal{F}_{sem}\} , our goal is to identify a pruned set of image tokens  ℙ d  i  v ⊂ 𝐗 I \mathbb{P}_{div}\subset\mathbf{X}_{I}  that maximizes the objective function  ℒ d  i  v \mathcal{L}_{div} :

ℒ d  i  v = max ⁡ ℱ  ( ℙ d  i  v ) . \mathcal{L}_{div}=\max{\mathcal{F}(\mathbb{P}_{div})}.   (4)

4  Methodology

4.1  Limitations of existing methods

Attention-based methods pursue local optima

We analyze the impact of pruning image tokens on the subsequent text and response tokens. From Equations

https://arxiv.org/html/2505.22038v2#S3.E1

https://arxiv.org/html/2505.22038v2#S3.E2

, we can see that pruning image tokens at  l l -th layer mainly affects the layer output  𝐗 ( l + 1 ) \mathbf{X}^{(l+1)}  by changing the attention output, which is a weighted sum of the value vectors  V l V_{l} . If the norms of the  V l V_{l}  are similar, selecting image tokens with high importance scores defined in

https://arxiv.org/html/2505.22038v2#S3.E3

effectively reduces the difference between the layer output before and after pruning. We provide supporting evidence for this assumption in the Appendix

https://arxiv.org/html/2505.22038v2#S7.SS1

. Formally, given original  l l -th layer output  𝐗 o  r  i  g  i  n ( l + 1 ) \mathbf{X}_{origin}^{(l+1)}  and pruned  l l -th layer output  𝐗 p  r  u  n  e  d ( l + 1 ) \mathbf{X}^{(l+1)}_{pruned}  , distance metric function  D  ( ⋅ , ⋅ ) D(\cdot,\cdot) , we can define the objective function  ℒ a  t  t  e  n \mathcal{L}_{atten}  of attention-based methods

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

ℒ a  t  t  e  n = min ℙ D  ( 𝐗 o  r  i  g  i  n ( l + 1 ) , 𝐗 p  r  u  n  e  d ( l + 1 ) ) . \mathcal{L}_{atten}=\mathop{\min}_{\mathbb{P}}D(\mathbf{X}_{origin}^{(l+1)},\mathbf{X}_{pruned}^{(l+1)}).   (5)

However, attention-based methods locally optimize the output error at individual layers. For instance, if pruning is conducted at the  l l -th layer and  ( l + k ) (l+k) -th layers, with  ℙ l \mathbb{P}_{l}  and  ℙ l + k \mathbb{P}_{l+k}  denoting the respective optimal sets of selected image tokens. As shown in Figure

https://arxiv.org/html/2505.22038v2#S1.F1

,  ℙ l + k ⊄ ℙ l \mathbb{P}_{l+k}\not\subset\mathbb{P}_{l} . So, attention-based selection results in a  globally suboptimal  pruning strategy.

Diversity-based methods ignore local constraints

The diversity-based approach

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

aims to maximize the diversity of the selected tokens, thereby partially mitigating the issues encountered by attention-based methods as we can see in Figure

https://arxiv.org/html/2505.22038v2#S1.F1

. Because diversity-based methods tend to select tokens with maximally different semantic information. However it can be observed in Figure

https://arxiv.org/html/2505.22038v2#S1.F2

that diversity-based approaches are ineffective in maintaining local output consistency, which can lead to a failure in preserving  local output consistency  during deep-layer pruning, resulting in degraded performance.

## Layer selection for pruning

Current approaches typically rely on manually predefined pruning layers or utilize a small validation set to select pruning layers based on the observed performance. However, these methods require extensive trial-and-error and dataset-specific calibration. As described in Section

https://arxiv.org/html/2505.22038v2#S3.SS1

, due to the presence of the causal mask  M M , the encoding of an image token in the LLM backbone is independent of the input question. Therefore, we aim to determine the pruning layers from the perspective of image token encoding.

Figure 3 :  Overview of BTP: We first use a calibration set to determine the pruning layers. In the early layers, we emphasize diversity-based pruning to preserve the output of subsequent layers. In the deeper layers, attention-based pruning is prioritized to maintain the output of the pruning layers. Due to the pruning strategy, we achieve an overall optimal pruning balance.

4.2  Balanced token pruning with joint local and global objectives

Local-global objective

Based on the above analysis, we argue that an effective token pruning strategy should achieve local optimization by preserving the current layer’s output, while also considering the global impact of pruning on subsequent layers. As shown in Equation

https://arxiv.org/html/2505.22038v2#S3.E1

, the model’s output depends on both the outputs of previous layers and the attention module of the current layer. Therefore, to ensure that the final output of the pruned model remains similar to that of the original model, we should maintain the similarity between the output of each pruned layer and its corresponding original output. Firstly, we formulate a  global objective  function. Suppose token pruning is performed at layers  l 1 < l 2 < l 3 l_{1}<l_{2}<l_{3} . For each pruned layer  l ∈ { l 1 , l 2 , l 3 } l\in\{l_{1},l_{2},l_{3}\} , we aim to select a subset of tokens  ℙ l \mathbb{P}_{l}  such that the difference between the pruned outputs  X p  r  u  n  e  d l + 1 X_{pruned}^{l+1}  and original outputs  X o  r  i  g  i  n l + 1 X_{origin}^{l+1}  is minimized. To quantify hidden vectors’ difference, we use a unified distance function  D  ( ⋅ , ⋅ ) D(\cdot,\cdot)  to measure the discrepancy between the outputs before and after pruning. Then our objective is to minimize the total output discrepancy across all pruned layers:

ℒ g  l  o  b  a  l = ∑ i = 1 | l | D  ( X o  r  i  g  i  n l + 1 , X ℙ l i l + 1 ) . \mathcal{L}_{global}=\sum_{i=1}^{|l|}D(X_{origin}^{l+1},X_{\mathbb{P}_{l_{i}}}^{l+1}).   (6)

## According to Equation

https://arxiv.org/html/2505.22038v2#S4.E5

, we can get optimal pruned token set  ℙ l ∗ \mathbb{P}_{l}^{*}  based on attention. However, since the attention distribution varies across input samples and  P l 3 ⊆ P l 2 ⊆ P l 1 P_{l_{3}}\subseteq P_{l_{2}}\subseteq P_{l_{1}} , it is difficult to predict which tokens will be important for deeper layers (e.g.,  l 2 l_{2} ,  l 3 l_{3} ) when pruning at layer  l 1 l_{1} . To address this issue, we propose to optimize a  local-global objective  to approximate the optimal token set  P l ∗ P_{l}^{*} . Building upon the local attention-based selection objective, we introduce a diversity term to approximate the token preferences of later layers. Assume a weight coefficient  λ ∈ ( 0 , 1 ) \lambda\in(0,1) , we measure diversity by computing the sum of distance  F d  i  s  ( ⋅ ) F_{dis}(\cdot)  among elements within a set:

ℒ l  o  c  a  l − g  l  o  b  a  l = − ∑ i = 1 | l | ( λ i  ∑ j ∈ P i A  t  t  e  n ( i )  ( 𝐗 I ( j ) , 𝐗 T ) + ( 1 − λ i )  F d  i  s  ( P i ) ) . \mathcal{L}_{local-global}=-\sum_{i=1}^{|l|}(\lambda_{i}\sum_{j\in P_{i}}Atten^{(i)}(\mathbf{X}_{I}^{(j)},\mathbf{X}_{T})+(1-\lambda_{i})F_{dis}(P_{i})).   (7)

## The first term of Equation

https://arxiv.org/html/2505.22038v2#S4.E7

ensures that the output of the pruned layer remains close to the original, while the second term encourages the selected tokens at previous layer  l 1 l_{1}  to also include those important for deeper layers such as  l 2 l_{2}  and  l 3 l_{3} .

Balanced token pruning (BTP)

Building upon the proposed local-global objective, we introduce our method. Figure

https://arxiv.org/html/2505.22038v2#S4.F3

, our approach divides token pruning into multiple stages denoted as  𝒮 = { s 1 , … , s n } \mathcal{S}=\{s_{1},\dots,s_{n}\} . Under a predefined pruning ratio  α \alpha , each stage retains a fixed fraction of image tokens from the previous stage. As shown in Appendix

https://arxiv.org/html/2505.22038v2#S7.SS2

, we can observe that retaining only a small number of image tokens is sufficient to optimize the attention objectives. Since early pruning stages retain more tokens and influence the pruning decisions of later stages, their objectives need to emphasize token diversity. In contrast, deeper stages preserve fewer tokens and have less impact on subsequent stages. Therefore, we set the hyperparameter  λ i \lambda_{i}  to gradually increase across stages.

Attention optimization:  We optimize the attention objective by selecting the top- k k  image tokens with the highest importance scores defined in Equation

https://arxiv.org/html/2505.22038v2#S3.E3

. To efficiently computing the importance scores, we only use the last token of the input prompt as  𝐗 T \mathbf{X}_{T} , which reduces the computational complexity to  𝒪  ( n ) \mathcal{O}(n) . We observe that the attention scores are influenced by positional encoding, which leads to a tendency to favor tokens located toward the end of the sequence. We apply a re-balancing operation to alleviate the influence of positional encoding. Assume that at  l l -th layer, we aim to prune the image tokens by selecting  k k  indices  I k I_{k}  out of  N N  candidates based on the attention scores  A l A_{l} . Instead of directly selecting the top- k k  tokens, we first over-select the top- k ′ k^{\prime}  tokens indices  I k ′ I_{k^{\prime}} , where  k ′ > k k^{\prime}>k . To mitigate positional bias, we rebalance the selection by first retaining tokens from earlier positions, followed by selecting additional tokens from later positions:

I p  r  e = I k ′  [ I k ′ < N 2 ] , \displaystyle I_{pre}=I_{k^{\prime}}[I_{k^{\prime}}<\frac{N}{2}],   (8)   I p  o  s  t = I k ′ [ I k ′ ≥ N 2 ] [ : k − | I p  r  e | ] , \displaystyle I_{post}=I_{k^{\prime}}[I_{k^{\prime}}\geq\frac{N}{2}][:k-|I_{pre}|],   (9)   I k = C  o  n  c  a  t  ( I p  r  e , I p  o  s  t ) . \displaystyle I_{k}=Concat(I_{pre},I_{post}).   (10)

Through the rebalancing operation, we are able to preserve the attention objective while selecting more informative tokens.

Diversity optimization:  For optimizing the second objective related to diversity, we follow the formulation used in DivPrune by modeling it as a Max-Min Diversity Problem (MMDP). However, solving the MMDP objective requires  𝒪  ( n 2 ) \mathcal{O}(n^{2})  computational complexity and cannot be efficiently accelerated by GPUs, resulting in significant computational latency. This issue becomes more pronounced in high-resolution multimodal models with a larger number of image tokens. To address this challenge, we propose an initialization strategy based on spatial position information. We observe that image patches with large spatial distances tend to exhibit greater semantic differences, while spatially adjacent patches are often semantically similar. Based on this intuition, we initialize the set of selected image tokens by solving an MMDP problem over their spatial positions. Formally, given  N N  image tokens  𝐗 I \mathbf{X}_{I} , which are originally obtained by flattening a 2D image, we first formulate a 2D grid of size  N × N \sqrt{N}\times\sqrt{N} . For any two tokens  y y  and  w w  from the  N N  tokens, their distance is defined as the Manhattan distance  d  ( ⋅ , ⋅ ) d(\cdot,\cdot)  between their positions in the 2D grid. Based on this distance metric, we construct the initial token set  E i  n  i  t  i  a  l E_{initial} :

E i  n  i  t  i  a  l = a r g m a x [ min y , w ∈ S ( d ( y , w ) : ∀ S ⊂ 𝐗 I ] . E_{initial}=argmax[\min_{y,w\in S}(d(y,w):\forall S\subset\mathbf{X}_{I}].   (11)

4.3  Pruning layer selection

We propose that determining which layers to prune is closely related to encoding process of image tokens. Specifically, pruning should occur either before or after the layers where the meaning of image tokens changes significantly, since it is difficult to identify truly important tokens in such layers. We compute the cosine similarity between image token hidden states  X I l , X I l + 1 X_{I}^{l},X_{I}^{l+1}  before and after each layer. For each layer, we plot the number of tokens with similarity below threshold  τ \tau  alongside the total attention allocated to image tokens. As shown in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

, it can be observed that LVLMs tends to allocate more attention to image tokens in layers following those where the representations of image tokens undergo significant changes. Based on these insights, we propose a task-independent layer selection strategy for pruning. Using a fixed set of 64 samples across all datasets, we identify layers immediately before and after major shifts in image token semantics. As shown in Figure

https://arxiv.org/html/2505.22038v2#S4.F3

, we perform pruning at selection layers, which enhances the effectiveness of our pruning strategy.

Figure 4 :  Layer-wise image token hidden state dynamics and attention allocation in LVLMs.   Table 1 :  Comparison of BTW with VTW, PDrop, FastV, and DivPrune across different models and datasets.  *  : For models using dynamic resolution, we report the token retention ratio instead of the absolute token count.   Method   Token   TFLOPS   GQA   MME   MMB e  n \text{MMB}_{en}   POPE   SQA   MMVET   Avg.   \rowcolor gray!20                                     LLaVA-1.5-7B   Original   576   3.82   62.0   1510.7   64.3   85.8   69.4   29.0   100%   VTW (AAAI25)

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

236   1.67   51.3   1475.0   63.4   82.1   68.8   17.8   89%   PDrop (CVPR25)

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

192   1.30   57.1   1399.0   61.6   83.6   68.4   25.8   94%   FastV (ECCV24)

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

172   1.65   57.6   1465.0   61.6   81.0   68.9   29.3   96%   DivPrune (CVPR25)

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

128   0.83   58.8   1405.4   62.1   85.1   68.4   27.4   96%   BTP (ours)   128   0.85   59.0   1487.0   62.7   85.6   69.1   29.1   98%   \rowcolor gray!20                                     LLaVA-1.5-13B   Original   576   7.44   63.2   1521.7   68.8   87.0   72.7   37.4   100%   VTW (AAAI25)

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

236   2.97   55.6   1517.1   67.7   79.0   72.2   22.6   89%   PDrop (CVPR25)

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

192   2.46   60.5   1493.0   67.3   85.1   73.7   32.8   96%   FastV (ECCV24)

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

172   2.25   60.0   1473.0   67.0   83.6   72.9   31.9   95%   DivPrune (CVPR25)

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

128   1.63   58.8   1461.0   65.8   86.5   72.6   34.0   96%   BTP (ours)   128   1.68   62.2   1519.7   68.0   86.9   72.7   34.5   98%   \rowcolor gray!20                                     LLaVA-1.6-7B   *   Original   100%   20.82   64.2   1519.3   67.1   86.4   73.6   37.5   100%   VTW (AAAI25)

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

40%   9.11   53.3   1472.8   65.6   84.1   68.3   16.3   85%   PDrop (CVPR25)

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

25%   6.77   60.4   1462.6   65.1   86.4   68.3   27.4   92%   FastV (ECCV24)

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

22%   5.76   60.3   1469.1   64.3   85.5   68.2   32.3   94%   DivPrune (CVPR25)

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

22%   4.20   61.4   1467.9   65.4   86.2   67.4   26.9   92%   BTP (ours)   22%   4.52   60.6   1490.8   65.8   86.7   68.4   30.3   94%   \rowcolor gray!20                                     Qwen2.5-VL-7B   *   Original   100%   5.48   60.4   1690.8   82.5   87.4   76.7   16.1   100%   VTW (AAAI25)

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

40%   2.38   40.2   1129.8   58.7   61.5   69.7   4.5   65%   PDrop (CVPR25)

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

30%   1.81   49.9   1462.5   70.6   76.8   72.6   9.58   82%   FastV (ECCV24)

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

30%   1.79   52.6   1595.5   73.4   83.9   74.0   16.2   96%   DivPrune (CVPR25)

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

25%   1.34   50.1   1639.2   76.9   85.4   73.0   17.5   96%   BTP (ours)   25%   1.67   57.2   1651.5   75.2   86.2   74.1   16.8   97%

5  Experiment

## Baselines and models

To rigorously assess the generalizability of our proposed image token compression method, we integrate it into several state-of-the-art multimodal large models and conduct extensive experiments on diverse benchmark tasks. Specifically, we evaluate our approach on four representative models: LLaVA-v1.5-7B, LLaVA-v1.5-13B, LLaVA-v1.6-7B and Qwen2.5-VL-7B-Instruct

bai2025qwen2

https://arxiv.org/html/2505.22038v2#bib.bib2

liu2023improvedllava

https://arxiv.org/html/2505.22038v2#bib.bib27

liu2024llavanext

https://arxiv.org/html/2505.22038v2#bib.bib28

liu2023llava

https://arxiv.org/html/2505.22038v2#bib.bib29

wang2024qwen2

https://arxiv.org/html/2505.22038v2#bib.bib43

. We select several plug-and-play compression baselines that support inference-time token pruning: FastV

chen2024image

https://arxiv.org/html/2505.22038v2#bib.bib5

and PyramidDrop

xing2024pyramiddrop

https://arxiv.org/html/2505.22038v2#bib.bib46

, which select informative tokens via attention mechanisms; DivPrune

alvar2025divprune

https://arxiv.org/html/2505.22038v2#bib.bib1

, which filters tokens based on visual diversity and VTW

lin2025boosting

https://arxiv.org/html/2505.22038v2#bib.bib26

, which discards all image tokens at a specific transformer layer determined by validation performance.

## Benchmarks and evaluation

We conduct comprehensive experiments on standard visual understanding tasks using models of different sizes, model families, and compression ratios. We report the results on GQA, MMB, MME, POPE, SQA and MM-VeT

fu2023mme

https://arxiv.org/html/2505.22038v2#bib.bib13

hudson2019gqa

https://arxiv.org/html/2505.22038v2#bib.bib21

iyyer2017search

https://arxiv.org/html/2505.22038v2#bib.bib22

liu2024mmbench

https://arxiv.org/html/2505.22038v2#bib.bib30

Li-hallucination-2023

https://arxiv.org/html/2505.22038v2#bib.bib49

https://arxiv.org/html/2505.22038v2#bib.bib50

. All experiments are carried out using the LMMs-Eval

lmms_eval2024

https://arxiv.org/html/2505.22038v2#bib.bib3

zhang2024lmmsevalrealitycheckevaluation

https://arxiv.org/html/2505.22038v2#bib.bib24

framework. In addition to accuracy on each dataset, we evaluate all methods in terms of FLOPs, inference latency, and KV cache memory usage. For inference throughout, we follow the PyramidDrop. Specifically, we calculate the FLOPs of the  l l -th layer’s attention and MLP modules through  4  n  d 2 + 2  n 2  d + 3  n  d  m 4nd^{2}+2n^{2}d+3ndm .  n n  is the number of tokens,  d d  is the hidden state size, and  m m  is the intermediate size of the FFN.

## Implementation details

All pruning experiments are conducted on 8 NVIDIA A800 GPUs using the HuggingFace Transformers library. To determine pruning stages, we randomly sample 64 instances from the LLaVA-655k

liu2023improvedllava

https://arxiv.org/html/2505.22038v2#bib.bib27

liu2024llavanext

https://arxiv.org/html/2505.22038v2#bib.bib28

liu2023llava

https://arxiv.org/html/2505.22038v2#bib.bib29

dataset and use the same set across all models and benchmarks, thus avoiding separate calibration for each benchmark. We gradually reduce the number of image tokens at each stage. In the early layers, we use a larger  λ \lambda  value to focus more on global information, while in the deeper layers, we use a smaller lambda to emphasize local details. More implementation details for different models are provided in the see Appendix

https://arxiv.org/html/2505.22038v2#S7.SS3

. Similar to the implementation of PyramidDrop, we compute the required attention scores separately within the  FlashAttn  module at the specified pruning layers, achieving full compatibility with  FlashAttn

dao2023flashattention2

https://arxiv.org/html/2505.22038v2#bib.bib11

dao2022flashattention

https://arxiv.org/html/2505.22038v2#bib.bib12

. It is worth noting that all our experiments are conducted with  FlashAttention  acceleration enabled.

5.1  Main results

## BTP outperforms SOTA methods across LVLMs

## As shown in Table

https://arxiv.org/html/2505.22038v2#S4.SS3

, we conduct extensive experiments across different model families and parameter scales. Empirical results demonstrate that our approach consistently surpasses state-of-the-art methods on most benchmark tasks. Our method achieves  98%  of the original average performance under a  22%  compression rate across LLaVA models of different sizes. Moreover, our method consistently outperforms all models, achieving better results than both attention-based and diversity-based approaches. We also visualize the impact of different methods on layer outputs in Figure

https://arxiv.org/html/2505.22038v2#S5.F5

, our method preserves consistency with the original outputs at both local and global levels. The Appendix

https://arxiv.org/html/2505.22038v2#S7.SS5

further provides visualizations of the spatial distribution of image tokens selected by various methods. Our method yields more effective token selection in deeper layers.

Figure 5 :  Effect of various pruned methods on the output of decoder layers.

## BTP maintains stable performance across different compression ratios

We assess the performance of our method across a range of compression ratios to verify its effectiveness. We find that FLOPs account only for the computational cost of the attention and MLP modules, while ignoring the overhead introduced by additional components. As a result, FLOPs alone fail to accurately reflect the actual inference latency. Therefore, as shown in Table

https://arxiv.org/html/2505.22038v2#S5.T2

, we compare the performance and average inference time of different methods under varying compression ratios. In can be observed that although DivPrune achieves lower theoretical FLOPs, its end to end latency even exceeds that of the original uncompressed model. In contrast, our method leverages spatial division for initialization, significantly reducing the actual inference time. Across various compression ratios, our method consistently achieves better performance than state-of-the-art approaches on most datasets, without incurring additional computational overhead.

Table 2:  Performance comparison with FastV and DivPrune across varying compression ratios. We report the results on LLaVa-v1.5-7B.   Method   Average Token   TFLOPS   Latency   GQA   MME   MMB   SQA   LLaVA-1.5-7B   576   3.82   0.145s   62.0   1510.7   64.3   69.4   FastV   128   0.86   0.122s (15%  ↓ \downarrow )   49.6   1388.6   56.1   60.2   DivPrune   128   0.83   0.224s (54%  ↑ \uparrow )   58.8   1405.4   62.1   68.4   BTP (ours)   128   0.85   0.134s (7%  ↓ \downarrow )   59.0   1487.0   62.7   69.1   FastV   64   0.42   0.118s  (18%  ↓ \downarrow )   46.1   801.3   48.0   51.1   DivPrune   64   0.41   0.150s (0.5% ↑ \uparrow )   57.5   1350.0   58.5   67.6   BTP (ours)   64   0.42   0.120s (17%  ↓ \downarrow )   55.0   1364.1   58.6   68.3

5.2  Efficiency analysis

The additional overhead introduced by our method primarily arises from the attention computation and the selection of the diversity set. Since we compute attention only between the final token and the image tokens, the added attention complexity is  𝒪  ( n ) \mathcal{O}(n) . For the selection of the diversity set, our proposed spatial initialization strategy and progressive weight decay allow us to select only a small number of additional tokens. In this section, we compare the efficiency of our method with other approaches, evaluating from multiple perspectives including theoretical FLOPs, inference latency, KV cache size, and corresponding benchmark performance. For inference latency, we report the average inference time per sample. For KV cache memory usage, we report the average GPU memory consumption after compression. We conduct experiments using LLaVA-v1.5 and LLaVA-v1.6. Notably, LLaVA-v1.6 processes images at a higher resolution, resulting in a larger number of image tokens.

Table 3:  Evaluation of compression efficiency on different models   Method   Averge token   Cache Size   TFLOPS   Latency   LLaVA-COCO   LLaVA-1.5-7B   576   0.34GB (100%)   3.82   2.24s   90.8   FastV   172   0.15GB   (55.8%  ↓ \downarrow )   1.65   2.11s  (5%  ↓ \downarrow )   80.6   DivPrune   128   0.11GB  (67.6%  ↓ \downarrow )   0.83   2.33s  (4%  ↑ \uparrow )   80.3   BTP (ours)   128   0.11GB  (67.6%  ↓ \downarrow )   0.85   2.13s  (4%  ↓ \downarrow )   80.9   LLaVA-1.6-7B   2880   1.11GB(100%)   20.82   4.24s   106.6   FastV   860   0.37GB  (66.6%  ↓ \downarrow )   6.45   3.77s  (11% ↓ \downarrow )   92.6   DivPrune   633   0.28GB  (74.7%  ↓ \downarrow )   4.20   5.00s  (17% ↑ \uparrow )   99.1   BTP (ours)   633   0.28GB  (74.7%  ↓ \downarrow )   4.52   3.91s (7% ↓ \downarrow )   98.9

## As shown in Table

https://arxiv.org/html/2505.22038v2#S5.T3

, our method achieves the best performance while maintaining practical efficiency.

5.3  Ablation study

Choice of balance factor value:  We first analyze the effect of  λ \lambda  in the local-global objective functions. This factor determines the trade-off at each layer between preserving local outputs and contributing to the global output. To thoroughly analyze the contribution of each pruning layer, we perform comprehensive ablation experiments on the LLaVA model. Our method includes three pruning layers, and we evaluate three configurations by fixing the  λ \lambda  parameters of two layers while varying the remaining one: (1) tuning the shallow layer while fixing the middle and deep layers, (2) tuning the middle layer while fixing the shallow and deep layers, and (3) tuning the deep layer while fixing the shallow and middle layers. We define the ratio between the performance of the pruned model and that of the base model on the target task as the performance gain. The computation of performace performance gain is detailed in the Appendix

https://arxiv.org/html/2505.22038v2#S7.SS4

Figure 6 :  Ablation study on balance factor.

## As shown in Figure

https://arxiv.org/html/2505.22038v2#S5.F6

, we can observe an early preference for the diversity objective in the shallow layers results in performance degradation. The middle layers should still retain a moderate degree of diversity, whereas the deeper layers, due to the limited number of remaining tokens, should prioritize the attention objective. This highlights the importance of our method in effectively balancing the two objectives.

Effectiveness of rebalanced attention and spatial diversity initialization:  We then perform ablation studies on the attention rebalance module and the spatial initialization module.

Table 4 :  Ablation study on attention rebalance module and spatial initialization module.   RA   SI   Latency   MME   GQA   POPE   ✓   ✓   0.134s   1487.0   59.0   85.6   ✓   0.232s   1486.5   57.9   86.4   ✓   0.140s   1464.6   57.4   85.1   0.231s   1478.1   57.3   84.4

We experimented with various combinations of the two modules. The results are presented in Table

https://arxiv.org/html/2505.22038v2#S5.T4

. It can be observed that removing the attention rebalance module results in a significant degradation in model performance. This degradation arises from the inherent bias in attention mechanisms, where positional encodings tend to shift attention disproportionately toward later tokens, leading to suboptimal token selection. On the other hand, omitting the spatial initialization module causes a marked increase in inference latency, in some cases even surpassing that of the original unpruned model. This suggests that while pruning reduces token count, naive initialization can introduce computational overhead that negates the benefits of pruning, thereby limiting the method’s applicability in latency-sensitive real-world scenarios

zhang2025citynavagentaerialvisionandlanguagenavigation

https://arxiv.org/html/2505.22038v2#bib.bib55

. This demonstrates the effectiveness of the proposed module in improving both model performance and inference speed. We also conducted an ablation study on the distance definitions used in the spatial diversity initialization module. As shown in the Appendix

https://arxiv.org/html/2505.22038v2#S7.SS7

, we found that the Euclidean distance models the diversity of image tokens more effectively than the Manhattan distance.

Effectiveness of calibration-based pruning stage selection:  To evaluate the effectiveness of our proposed calibration-based pruning stage selection, we compare it with a baseline that uniformly divides the pruning stages according to the total number of decoder layers, under the same compression rate. Experimental results are shown in Table

https://arxiv.org/html/2505.22038v2#S5.T5

Table 5 :  Ablation study on layer selection strategy.   Method   Stage Selection   MME   MMB   LLaVa-v1.5   Averaged   1483.2   62.3   Ours   1487.0   62.7   LLaVa-v1.6   Averaged   1480.1   64.7   Ours   1490.8   65.8   Qwen2.5-vl   Averaged   1551.6   73.8   Ours   1641.5   75.2

We observe that our pruning layer selection method outperforms uniform selection. This is especially evident on Qwen2.5-VL, where uniform selection leads to a significant performance drop. We attribute this to differences in how Qwen2.5-VL processes image tokens as shown in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

. We also conduct an ablation study on the size and composition of the calibration set. Specifically, we expanded the calibration set by incorporating images from multiple datasets, including GQA,  V ∗ V^{*} Bench

wu2023vguidedvisualsearch

https://arxiv.org/html/2505.22038v2#bib.bib45

, and SQA and UrbanVideo-Bench

zhao2025urbanvideobenchbenchmarkingvisionlanguagemodels

https://arxiv.org/html/2505.22038v2#bib.bib58

. We then repeated the experiment shown in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

using calibration set sizes of 64, 128, and 256. The results are presented in Appendix

https://arxiv.org/html/2505.22038v2#S7.SS6

, we can see that the variation patterns of image tokens remain consistent across different calibration set sizes and content, demonstrating the robustness of our pruning layer selection method.

Ablation on the Computation of Attention-Based Importance Scores:  In our method, the importance score of each image token is obtained by using the attention assigned to it by the last text token in the prefilling stage. To verify the robustness of this design, we conduct an ablation study comparing different ways of computing the importance score: 1. Averaging attention weights from all text tokens to each image token. 2. Following the approach in

zhang2025sparsevlmvisualtokensparsification

https://arxiv.org/html/2505.22038v2#bib.bib57

, where image–text similarity is first computed and the most similar text tokens are then selected to calculate the importance score.

Table 6 :  Ablation study on importance score calculation method.   Method   MME   MMB   POPE   GQA   SQA   last-token(ours)   1497   63.4   85.6   59.1   69.1   averaged-tokens   1490   62.8   84.7   57.3   69.4   similarity-based   1485   63.1   84.7   57.9   69.7

## The results are shown in Table

https://arxiv.org/html/2505.22038v2#S5.T6

. We can see that last token efficiently modeling the importance score. We believe that the last token in the input prompt is a suitable choice for computing the importance score because it is typically decoded as the first output token during the decoding stage. This allows it to effectively capture the model’s focus.

6  Conclusion

In this work, we conduct initial studies to investigate and verify the limitations of existing image token pruning methods. We further analyze the impact of two pruning strategies on model performance from the perspective of the objective function, and formulate a local-global pruning optimization objective. To reduce information loss during pruning, we propose  Balanced Token Pruning (BTP) , a multi-stage pruning method. We first determine the pruning stages using a calibration set. In the early layers, we focus on a  diversity-oriented objective  to account for the influence of pruning on deeper layers, while in the later layers, we adopt an  attention-based objective  to better preserve local information. In future work, we will further investigate the lightweight deployment on real devices

ruan2025edmambarethinkingefficientevent

https://arxiv.org/html/2505.22038v2#bib.bib38

zheng2025reviewedgelargelanguage

https://arxiv.org/html/2505.22038v2#bib.bib61

and explore its potential applications in multi-agent collaboration

10.1145/3594739.3612905

https://arxiv.org/html/2505.22038v2#bib.bib36

gao2025multimodalagenttuningbuilding

https://arxiv.org/html/2505.22038v2#bib.bib16

## Acknowledgments

This paper was supported by the Natural Science Foundation of China under Grant 62371269, Natural Science Foundation of China under 62272262, Shenzhen Low-Altitude Airspace Strategic Program Portfolio Z253061 and Guangdong Innovative, Entrepreneurial Research Team Program (2021ZT09L197) and Meituan Academy of Robotics Shenzhen. This paper was sponsored by Tsinghua University-Toyota Research Center.

[1]    Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang.  Divprune: Diversity-based visual token pruning for large multimodal models.  arXiv preprint arXiv:2503.02175 , 2025.

[2]    Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al.  Qwen2.5-vl technical report.  arXiv preprint arXiv:2502.13923 , 2025.

[3]    Li* Bo, Zhang* Peiyuan, Zhang* Kaichen, Pu* Fanyi, Du Xinrun, Dong Yuhao, Liu Haotian, Zhang Yuanhan, Zhang Ge, Li Chunyuan, and Ziwei Liu.  Lmms-eval: Accelerating the development of large multimoal models, March 2024.

[4]    Jieneng Chen, Luoxin Ye, Ju He, Zhao-Yang Wang, Daniel Khashabi, and Alan Yuille.  Efficient large multi-modal models via visual context compression.  In  The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.

[5]    Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang.  An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models.  In  European Conference on Computer Vision , pages 19–35. Springer, 2024.

[6]    Xuecheng Chen, Haoyang Wang, Yuhan Cheng, Haohao Fu, Yuxuan Liu, Fan Dang, Yunhao Liu, Jinqiang Cui, and Xinlei Chen.  Ddl: Empowering delivery drones with large-scale urban sensing capability.  IEEE Journal of Selected Topics in Signal Processing , 2024.

[7]    Xuecheng Chen, Zijian Xiao, Yuhan Cheng, Chen-Chun Hsia, Haoyang Wang, Jingao Xu, Susu Xu, Fan Dang, Xiao-Ping Zhang, Yunhao Liu, and Xinlei Chen.  Soscheduler: Toward proactive and adaptive wildfire suppression via multi-uav collaborative scheduling.  IEEE Internet of Things Journal , 11(14):24858–24871, 2024.

[8]    Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al.  Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks.  In  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 24185–24198, 2024.

[9]    Moulik Choraria, Xinbo Wu, Sourya Basu, Nitesh Sekhar, Yue Wu, Xu Zhang, Prateek Singhal, and Lav R. Varshney.  Semantically grounded qformer for efficient vision language understanding, 2024.

[10]    Can Cui, Zichong Yang, Yupeng Zhou, Juntong Peng, Sung-Yeon Park, Cong Zhang, Yunsheng Ma, Xu Cao, Wenqian Ye, Yiheng Feng, Jitesh Panchal, Lingxi Li, Yaobin Chen, and Ziran Wang.  On-board vision-language models for personalized autonomous vehicle motion control: System design and real-world validation, 2024.

[11]    Tri Dao.  FlashAttention-2: Faster attention with better parallelism and work partitioning.  In  International Conference on Learning Representations (ICLR) , 2024.

[12]    Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.  FlashAttention: Fast and memory-efficient exact attention with IO-awareness.  In  Advances in Neural Information Processing Systems (NeurIPS) , 2022.

[13]    Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al.  Mme: A comprehensive evaluation benchmark for multimodal large language models.  arXiv preprint arXiv:2306.13394 , 2023.

[14]    Chen Gao, Baining Zhao, Weichen Zhang, Jinzhu Mao, Jun Zhang, Zhiheng Zheng, Fanhang Man, Jianjie Fang, Zile Zhou, Jinqiang Cui, Xinlei Chen, and Yong Li.  Embodiedcity: A benchmark platform for embodied agent in real-world city environment, 2024.

[15]    Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al.  Mini-internvl: a flexible-transfer pocket multi-modal model with 5% parameters and 90% performance.  Visual Intelligence , 2(1):1–17, 2024.

[16]    Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaojian Ma, Tao Yuan, Yue Fan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and Qing Li.  Multi-modal agent tuning: Building a vlm-driven agent for efficient tool usage, 2025.

[17]    Qiuyi Gu, Zhaocheng Ye, Jincheng Yu, Jiahao Tang, Tinghao Yi, Yuhan Dong, Jian Wang, Jinqiang Cui, Xinlei Chen, and Yu Wang.  Mr-cographs: Communication-efficient multi-robot open-vocabulary mapping system via 3d scene graphs.  IEEE Robotics and Automation Letters , 10(6):5713–5720, 2025.

[18]    Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, Yao Hu, and Shaohui Lin.  Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification.  arXiv preprint arXiv:2412.00876 , 2024.

[19]    Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, Yao Hu, and Shaohui Lin.  Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification, 2025.

[20]    Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, and Daniel Soudry.  Accurate post training quantization with small calibration sets.  In  International Conference on Machine Learning , pages 4466–4475. PMLR, 2021.

[21]    Drew A Hudson and Christopher D Manning.  Gqa: A new dataset for real-world visual reasoning and compositional question answering.  In  Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 6700–6709, 2019.

[22]    Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang.  Search-based neural structured learning for sequential question answering.  In  Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1821–1831, 2017.

[23]    Zhuozhu Jian, Zejia Liu, Haoyu Shao, Xueqian Wang, Xinlei Chen, and Bin Liang.  Path generation for wheeled robots autonomous navigation on vegetated terrain, 2023.

[24]    Zhang Kaichen, Li Bo, Zhang Peiyuan, Pu Fanyi, Cahyono Joshua-Adrian, Hu Kairui, Liu Shuai, Zhang Yuanhan, Yang Jingkang, Li Chunyuan, and Liu Ziwei.  Lmms-eval: Reality check on the evaluation of large multimodal models, 2024.

[25]    Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.  Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation, 2022.

[26]    Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji.  Boosting multimodal large language models with visual tokens withdrawal for rapid inference.  In  Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 5334–5342, 2025.

[27]    Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.  Improved baselines with visual instruction tuning, 2023.

[28]    Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee.  Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.

[29]    Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.  Visual instruction tuning, 2023.

[30]    Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al.  Mmbench: Is your multi-modal model an all-around player?  In  European conference on computer vision , pages 216–233. Springer, 2024.

[31]    Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu.  Kivi: A tuning-free asymmetric 2bit quantization for kv cache.  arXiv preprint arXiv:2402.02750 , 2024.

[32]    Yu Meng, Kaiyuan Li, Chenran Huang, Chen Gao, Xinlei Chen, Yong Li, and Xiaoping Zhang.  Plphp: Per-layer per-head vision token pruning for efficient large vision-language models, 2025.

[33]    Daniel Cosmin Porumbel, Jin-Kao Hao, and Fred Glover.  A simple and effective algorithm for the maxmin diversity problem.  Annals of Operations Research , 186:275–293, 2011.

[34]    Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.  Learning transferable visual models from natural language supervision, 2021.

[35]    Charles C Ragin.  Fuzzy sets: Calibration versus measurement.  Methodology volume of Oxford handbooks of political science , 2, 2007.

[36]    Jiyuan Ren, Yanggang Xu, Zuxin Li, Chaopeng Hong, Xiao-Ping Zhang, and Xinlei Chen.  Scheduling uav swarm with attention-based graph reinforcement learning for ground-to-air heterogeneous data communication.  In  Adjunct Proceedings of the 2023 ACM International Joint Conference on Pervasive and Ubiquitous Computing & the 2023 ACM International Symposium on Wearable Computing , UbiComp/ISWC ’23 Adjunct, page 670–675, New York, NY, USA, 2023. Association for Computing Machinery.

[37]    Mauricio GC Resende, Rafael Martí, Micael Gallego, and Abraham Duarte.  Grasp and path relinking for the max–min diversity problem.  Computers & Operations Research , 37(3):498–508, 2010.

[38]    Ciyu Ruan, Zihang Gong, Ruishan Guo, Jingao Xu, and Xinlei Chen.  Edmamba: Rethinking efficient event denoising with spatiotemporal decoupled ssms, 2025.

[39]    Ciyu Ruan, Ruishan Guo, Zihang Gong, Jingao Xu, Wenhan Yang, and Xinlei Chen.  Pre-mamba: A 4d state space model for ultra-high-frequent event camera deraining, 2025.

[40]    Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan.  Llava-prumerge: Adaptive token reduction for efficient large multimodal models.  arXiv preprint arXiv:2403.15388 , 2024.

[41]    Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample.  Llama: Open and efficient foundation language models.  arXiv preprint arXiv:2302.13971 , 2023.

[42]    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.  Attention is all you need.  Advances in neural information processing systems , 30, 2017.

[43]    Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al.  Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution.  arXiv preprint arXiv:2409.12191 , 2024.

[44]    Jiang Wu, Sichao Wu, Yinsong Ma, Guangyuan Yu, Haoyuan Xu, Lifang Zheng, and Jingliang Duan.  Monitorvlm:a vision language framework for safety violation detection in mining operations, 2025.

[45]    Penghao Wu and Saining Xie.  V*: Guided visual search as a core mechanism in multimodal llms, 2023.

[46]    Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al.  Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction.  arXiv preprint arXiv:2410.17247 , 2024.

[47]    Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia.  Visionzip: Longer is better but not necessary in vision language models.  arXiv preprint arXiv:2412.04467 , 2024.

[48]    Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang.  Atp-llava: Adaptive token pruning for large vision language models.  arXiv preprint arXiv:2412.00447 , 2024.

[49]    Li Yifan, Du Yifan, Zhou Kun, Wang Jinpeng, Zhao Wayne-Xin, and Ji-Rong Wen.  Evaluating object hallucination in large vision-language models.  In  The 2023 Conference on Empirical Methods in Natural Language Processing , 2023.

[50]    Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang.  Mm-vet: Evaluating large multimodal models for integrated capabilities.  In  International conference on machine learning . PMLR, 2024.

[51]    Sojeong Yun and Youn-kyung Lim.  What if smart homes could see our homes?: Exploring diy smart home building experiences with vlm-based camera sensors.  In  Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems , CHI ’25, New York, NY, USA, 2025. Association for Computing Machinery.

[52]    Jirong Zha, Yuxuan Fan, Xiao Yang, Chen Gao, and Xinlei Chen.  How to enable llm with 3d capacity? a survey of spatial reasoning in llm, 2025.

[53]    Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng.  Llava-mini: Efficient image and video large multimodal models with one vision token.  arXiv preprint arXiv:2501.03895 , 2025.

[54]    Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng.  Llava-mini: Efficient image and video large multimodal models with one vision token, 2025.

[55]    Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang, Jinqiang Cui, Xinlei Chen, and Yong Li.  Citynavagent: Aerial vision-and-language navigation with hierarchical semantic planning and global memory, 2025.

[56]    Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, and Fei-Yue Wang.  Logisticsvln: Vision-language navigation for low-altitude terminal delivery based on agentic uavs, 2025.

[57]    Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang.  Sparsevlm: Visual token sparsification for efficient vision-language model inference, 2025.

[58]    Baining Zhao, Jianjie Fang, Zichao Dai, Ziyou Wang, Jirong Zha, Weichen Zhang, Chen Gao, Yue Wang, Jinqiang Cui, Xinlei Chen, and Yong Li.  Urbanvideo-bench: Benchmarking vision-language models on embodied intelligence with video data in urban spaces, 2025.

[59]    Baining Zhao, Ziyou Wang, Jianjie Fang, Chen Gao, Fanhang Man, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li, and Wenwu Zhu.  Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning, 2025.

[60]    Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.  Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

[61]    Yue Zheng, Yuhao Chen, Bin Qian, Xiufang Shi, Yuanchao Shu, and Jiming Chen.  A review on edge large language models: Design, execution, and applications, 2025.

7  Appendix

7.1  Key and Value of LVLMs

Following previous works on token quantization KIVI  [

https://arxiv.org/html/2505.22038v2#bib.bib31

] , we visualize the  K l K_{l}  and  V l V_{l}  of different LVLMs, the results are shown below:

(a)   LLaVA key   (b)   LLaVA value   Figure 7 :  Visualization of key and value of LLaVA-v1.5   (a)   LLaVA-v1.6 key   (b)   LLaVA-v1.6 value   Figure 8 :  Visualization of key and value of LLaVA-v1.6   (a)   Qwen2.5-vl key   (b)   Qwen2.5-vl value   Figure 9 :  Visualization of key and value of Qwen2.5-vl

7.2  Top-k Importance Image Token Received Attention Ratio

We calculate the ratio between the attention scores received by the top-k most text-attended image tokens and the total attention scores received by all image tokens:

(a)   Layer 4 Attention Ratio   (b)   Layer 8 Attention Ratio   Figure 10 :  Visualization Top-k Importance Image Token Received Attention Ratio   (a)   Layer 12 Attention Ratio   (b)   Layer 16 Attention Ratio   Figure 11 :  Visualization Top-k Importance Image Token Received Attention Ratio

7.3  Experiment Settings

For  LLaVA-v1.5-7B ,  LLaVA-v1.5-13B , and  LLaVA-v1.6-7B , we divide the pruning process into five stages based on the image token handling pipeline described in the Appendix. In each stage, except for the last one, we retain 50% of the tokens from the previous stage. In the final stage, all tokens are discarded to maximize inference speed. For  Qwen2.5-VL , since its image token processing can be clearly divided into two stages, we retain 25% of the tokens in the fourth stage and 12.5% in the final stage to preserve model performance. The  λ \lambda  used for different models are shown below:

Table 7 :  λ \lambda  settings in different models   Model   λ \lambda   llava-v1.5-7b   (0.6,0.8,1.0)   llava-v1.5-13b   (0.6,0.8,1.0)   llava-v1.6-13b   (0.4,0.7,1.0)   qwen-2.5-vl-7b   (0.2,0.5,0.8,1.0)

7.4  Calculation of model gain

Since evaluation metrics vary across tasks and the difficulty levels differ significantly, it is not reasonable to present all task results directly in a unified format. For example, the original LLaVA-v1.5 model scores 1510 on the MME benchmark but only 62 on GQA. To address this, we define a model gain metric as:

G  a  i  n = N  o  r  m  a  l  i  z  e  ( P  r  u  n  e  d s  c  o  r  e O  r  i  g  i  n  a  l s  c  o  r  e ) . Gain=Normalize(\frac{Pruned_{score}}{Original_{score}}).   (12)

7.5  Visualization of token selection under different pruning strategies

Figure 12 :  Visualization of Image Token Selection Across Different Methods

7.6  Ablation Study on Calibration Set

## According to Equation

https://arxiv.org/html/2505.22038v2#S3.E2

in the paper, in multimodal large language models, M denotes the causal mask, which constrains each token to attend only to preceding tokens. In our input format, image tokens always precede the text prompts (i.e., the input follows the structure: prefix + image + question). As a result, the model processes the image tokens before it receives the specific question which is unrelated to the input question. To validate our hypothesis, we posed different types of questions on the same image. We then conducted the experiment presented in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

using LLaVA-v1.5. We computed the number of image tokens whose cosine similarity between adjacent layers falls below 0.93. The resulting trends are shown as follows:

Table 8 :  Ablation on Calibration Set Size.   Set Size   layer1   layer5   layer9   layer13   layer17   layer21   layer25   64   0   0   325   141   155   45   1   128   0   0   325   141   155   45   1   256   0   0   325   141   155   45   1

We observe that varying the question type for the same image does not lead to significant differences in the results. In the following analysis, we investigate how image content and the size of the calibration set affect our method. We then repeated the experiment shown in Figure

https://arxiv.org/html/2505.22038v2#S4.F4

using calibration set sizes of 64, 128, and 256. Specifically, we computed the number of image tokens whose cosine similarity between adjacent layers falls below 0.93 using LLaVA-v1.5. The results are presented below:

Table 9 :  Ablation on Calibration Set Size.   Set Size   layer1   layer5   layer9   layer13   layer17   layer21   layer25   64   0   0   325   141   155   45   1   128   0   0   350   166   127   36   5   256   0   0   332   174   145   32   3

7.7  Ablation Study on Distance

To evaluate the impact of this choice, we conducted an ablation study comparing two different distance metrics using llava-v1.5. The results are summarized in the table below:

