---
sourceFile: "ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:33.877Z"
---

# ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale - arXiv

b63fbaab-9d14-46f0-99c8-02a529bff751

ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale - arXiv

e8a8d0ef-3ef6-4d2e-aaaf-9208edb60e5c

https://arxiv.org/pdf/2508.17624

ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale

Ge Shi1,*, Hanieh Sadri1,*, Qian Wang1, Yu Zhang2, Ying Xiong1, Yong Zhang1, Zhenan Fan1

1Huawei Technologies Canada , 2Huawei Cloud

Expert-Specialized Fine-Tuning (ESFT) adapts Mixture-of-Experts (MoE) large language models to enhance their task-specific performance by selectively tuning the top-activated experts for the task. Serving these fine-tuned models at scale is challenging: deploying merged models in isolation is prohibitively resource-hungry, while existing multi-adapter serving systems with LoRA-style additive updates are incompatible with ESFT’s expert-oriented paradigm. We present ExpertWeave, a system that serves multiple ESFT adapters concurrently over a single shared MoE base model, drastically reducing the memory footprint and improving resource utilization. To seamlessly integrate into existing inference pipelines for MoE models with non-intrusive modifications and minimal latency overhead, ExpertWeave introduces a virtual-memory-assisted expert weight manager that co-locates base-model and adapter experts without incurring memory overhead from fragmentation, and a fused kernel for batched rerouting to enable lightweight redirection of tokens to the appropriate experts at runtime. Our evaluations show that ExpertWeave can simultaneously serve multiple adapters of a 16B MoE model on a single accelerator where the baseline runs out of memory, or provides up to 94× more KV cache capacity and achieves up to 18% higher throughput while using comparable resources, all without compromising model accuracy. ExpertWeave maintains low overhead even when scaling to 20 adapters, with a 4–11% latency increase compared with serving the base model alone. Source code will be released soon.

*Equal contribution.

1 Introduction 3

2 Background 4 2.1 Mixture-of-Experts LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 2.2 Expert-Specialized Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 2.3 Multi-LoRA Serving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

3 Challenges with ESFT Experts 7 3.1 Expert Weight Management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7 3.2 Efficient Adapter-aware Runtime . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

4 ExpertWeave: Scaling Multi-ESFT Serving 9 4.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9 4.2 Virtual-Memory-Assisted Expert Weight Management . . . . . . . . . . . . . . . . 10 4.3 Batched Rerouting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

5 Evaluation 14 5.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 5.2 End-to-End Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 5.3 Microbenchmarking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 5.4 Memory Efficiency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 5.5 Accuracy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

6 Related Work 19

7 Conclusion 20

### 1. Introduction

Large Language Models (LLMs) have recently gained significant attention for their remarkable performance across a broad range of tasks [2, 29, 32, 35]. These models are typically built using the Transformer architecture with multi-head self-attention and feed-forward network (FFN) layers [36]. Traditionally, FFN layers implement a dense architecture, where a single set of parameters is shared across all tokens to compute their activations. This design simplifies the model architecture and the training process. However, since every token is processed by the same weights, this design lacks specialization across different token contexts and imposes substantial computational cost during training and inference.

Mixture-of-Experts (MoE) architectures [22, 24, 33] introduce sparsity into transformer-based LLMs by replacing each FFN layer with a set of experts, where each expert is a small-scale FFN layer. A learned token router activates only a small number of experts for each input token, routing different tokens to different experts in the same layer. This sparse, conditional computation results in LLMs with significantly more parameters than their dense counterparts [13, 14, 26, 46], while the computation required per token remains bounded by the number of experts activated.

As MoE models gain wider adoption, efficiently adapting them to downstream tasks becomes increasingly important. As full-parameter fine-tuning is memory- and compute-intensive and oftentimes prohibitively costly, Parameter-Efficient Fine-Tuning (PEFT)methods are often preferred to save time and cost. One common PEFT method is Low-Rank Adaptation (LoRA) [20], which introduces trainable low-rank matrices as add-on adapters into self-attention and FFN layers in the model. Although LoRA is effective for dense models, its application in MoE models remains underexplored [38].

To address this gap inMoEmodels,Wang et al. proposed Expert-Specialized Fine-Tuning (ESFT) [38], a parameter-efficient approach tailored for the MoE architecture. ESFT recognizes an expert special-ization pattern in MoE models: expert routing concentrates on a small fixed subset of experts for a given downstream task, while the sets of activated experts vary substantially across tasks. Instead of updating all experts, ESFT leverages this pattern to compute per-layer expert relevance on a small sample of task data and select only the top-activated experts with a cumulative relevance score exceeding a threshold 𝑝 for fine-tuning. Unlike LoRA with a static shape, the number of ESFT experts fine-tuned per layer in one adapter can be different due to different task relevance distributions.

Despite the advantages of ESFT, it introduces practical challenges for inference and serving, especially in modern cloud Model-as-a-Service (MaaS) environments. To deploy fine-tuned models, people often must follow a merge-and-serve approach, in which an adapter is merged into the base model to produce a standalone checkpoint for deployment. This strategy scales poorly in the cloud, as each fine-tuned model must be deployed independently, imposing additional memory and compute overhead.

With dense models, systems like Punica [4] and S-LoRA [34] have enabled the concurrent inference of multiple LoRA adapters over a shared base model, trading minor latency overhead for drastically

reduced deployment cost. However, enabling shared inference for ESFT adapters on MoE models remains a challenge, as ESFT adapters require a fundamentally different inference process than the LoRA-style additive updates.

Serving multiple ESFT adapters in one system presents several key challenges. First, integrating multiple adapters, each with its own set of fine-tuned experts, requires the system to incorporate the dynamically loaded adapter experts into the execution pipeline without modifying core com-putational kernels. Second, memory management for the accelerator becomes much harder as the number of fine-tuned experts varies both across adapters and across layers within an adapter, leading to fragmentation and inefficient utilization of limited device memory. Third, batching requests across multiple adapters that involve both base-model and adapter experts requires precise routing of hidden states so that each token is processed by the correct set of experts. Finally, the routing of tokens must not be a performance bottleneck, particularly during the prefill phases due to the need to efficiently route a large volume of tokens to their corresponding experts.

To address the aforementioned challenges, we present a system for shared inference of ESFT adapters on MoE models, enabling multiple models fine-tuned on different downstream tasks to be served concurrently and efficiently over a single shared MoE base model. We make the following contributions:

We present ExpertWeave, a system capable of serving multiple ESFT adapters concurrently over a shared base model on Ascend NPUs [28]. An overview of ExpertWeave is shown in Figure 1. We demonstrate that, as the number of ESFT adapters increases, our system remains scalable with minimal impact on key performance metrics like throughput and latency.

We propose a unified and memory-efficient expert weight management unit that minimizes fragmentation by leveraging virtual memory on Ascend NPUs, sustaining high memory utilization even under heterogeneous adapter expert configurations without requiring modi-fications to the core computational kernels.

We implement an efficient adapter-aware inference runtimewith a batched rerouting operator that enables lightweight token dispatching to the correct base-model and adapter experts without hurting latency.

We validate the effectiveness of ExpertWeave through experiments on end-to-end serving of multiple ESFT adapters, demonstrating scalability, efficiency, effective memory management, and serving accuracy across tasks.

### 2. Background

#### 2.1. Mixture-of-Experts LLMs

Mixture-of-Experts (MoE) models replace the standard feed-forward network (FFN) layer with an MoE layer that contains a fixed number of experts [23, 33]. Each expert is a small FFN consisting of linear projections and a nonlinear activation. The MoE layer processes flattened inputs of shape

ESFT Adapter Repository (Secondary Storage)

Adapter 0 Adapter 1

Adapter 2 …

## CPU Memory

NPU … NPU 1

Base-model

## Other Weights

## Unified Expert Weights

## Activation

Figure 1 | Overview of ExpertWeave. ESFT adapters stored in secondary storage are loaded and cached in CPU main memory before loading onto the NPU. ExpertWeave manages both base-model and adapter expert weights in a unified manner, ensuring minimal modifications to existing inference pipelines for MoE models.

[𝐵,𝐻 ], where 𝐵 denotes the number of tokens in the batch and 𝐻 is the hidden dimension. A learned router 𝑔(𝑥) selects the top-𝑘 relevant experts for each token 𝑥 and performs computation only using the activated experts. This sparse activation pattern increases model capacity and improves computational efficiency by activating only a small, relevant subset of parameters per token. DeepSeekMoE [6, 7, 8] further improves model performance by using many fine-grained experts, enabling more flexible expert combinations while maintaining computational efficiency.

Inference engines commonly store expert weights in the MoE layer by stacking each linear layer as a three-dimensional tensor of shape [𝑀,𝐻out, 𝐻in], where𝑀 is the number of experts, and 𝐻in and 𝐻out are the input and output hidden dimensions of a linear layer in an expert [25, 37]. The router computes top-𝑘 expert IDs of shape [𝐵, 𝑘] to determine the selected experts for each token. The system then dispatches tokens to their selected experts by replicating, permuting and grouping the tokens by the top-𝑘 IDs; tokens targeting the same expert are stored in a contiguous chunk. Batched expert computation is then performed by invoking the Grouped Matrix Multiplication (GMM) operator [5] on the permuted tokens and the stacked expert weight tensor. The resulting outputs are then combined using the computed weights from the router and unpermuted back to the original order for subsequent layers.

#### 2.2. Expert-Specialized Fine-Tuning

LLMs can be adapted to downstream tasks through fine-tuning, where pretrained model parameters are further updated to improve the task-specific performance. In traditional full-parameter fine-tuning, all weight parameters are updated; in contrast, PEFT methods [16] modify only a small subset of parameters while leaving most of the backbone model froze, often by inserting lightweight adapter modules. PEFT methods are usually employed thanks to their improved fine-tuning efficiency with reduced memory and compute requirements.

Expert-Specialized Fine-Tuning (ESFT) [38] is a recent PEFT method tailored for the MoE archi-tecture by exploiting the expert specialization pattern in MoE models: for a given task, expert activations are condensed on a small yet largely fixed subset of experts, while the top-activated

expert sets differ substantially across different tasks. This pattern motivates ESFT to selectively fine-tune the experts most relevant to the downstream task. ESFT first samples a subset of task data and uses it to compute a relevance score for each expert. ESFT proposes two metrics of expert relevance scores: average gate score and token selection ratio. ESFT then identifies the top-activated experts in each layer by finding the set of experts where the cumulative relevance score exceeds a threshold hyperparameter 𝑝 . These experts are then selected as candidates for fine-tuning on the downstream task.

During fine-tuning, tokens might still be routed to any expert, but only the selected experts are updated; all the other experts and modules remain frozen. In particular, the router is not updated, opening the possibility for shared inference. In practice, ESFT can achieve performance close to or on par with full-parameter fine-tuning by only fine-tuning roughly 5–15% of the total experts across all layers, yielding significant savings in resource usage and making it an attractive PEFT method in customizing MoE models.

#### 2.3. Multi-LoRA Serving

As LoRA gains adoption as an efficient PEFT technique, the increasing demand for deploying LoRA adapters motivates the emergence of multi-LoRA serving, where a single base model is shared among multiple LoRA adapters. These LoRA adapters are mounted to the system as low-rank matrices alongside the base model. During inference, the system first computes the base-model activation 𝑦 = 𝑥𝑊 and then applies the adapter-specific updates 𝑦′ = 𝑦 + 𝑥𝐴𝑖𝐵𝑖 . Sharing the base model reduces memory pressure when many adapters are active, at the cost of additional computation and latency overhead for the additive adapter updates.

To make multi-LoRA serving efficient in practice, recent systems rely on custom CUDA kernels that batch the LoRA computations across requests and adapters. Punica [4] introduces a Segmented Gather Matrix-Vector Multiplication (SGMV) kernel that groups tokens corresponding to the same adapter into contiguous segments so a single kernel invocation can process all adapters in parallel. S-LoRA [34] further enables serving LoRA adapters with varying ranks: its Unified Paging places paged LoRA weights and KV cache in a single memory pool, supporting adapters of different ranks without the need for padding. To enable LoRA computation on paged, non-contiguous adapter weights, S-LoRA proposes a Multi-size Batched Gather Matrix-Matrix (MBGMM) kernel for prefill and a Multi-size Batched Gather Matrix-Vector (MBGMV) kernel for decode. Both systems depend on specialized computational kernels to support multi-LoRA inference efficiently.

Although effective in multi-LoRA setups, these mechanisms cannot be directly transferred to serving ESFT adapters on MoE models. As ESFT fine-tunes selected experts within each MoE layer, the required pattern becomes expert routing followed by expert computations through the GMM operator rather than additive updates from mounted low-rank matrices through SGMV, MBGMM or MBGMV. To the best of our knowledge, there are no analogous multi-adapter serving solutions for ESFT.

### 3. Challenges with ESFT Experts

#### 3.1. Expert Weight Management

We aim to serve multiple ESFT adapters on a single base MoE model. The experts to be fine-tuned by ESFT are selected based on the expert relevance scoring mechanism: only the set of top-activated experts with a cumulative relevance score exceeding a threshold hyperparameter 𝑝 is fine-tuned [38]. A direct consequence of this score-based selection is that the number of fine-tuned experts can vary for different layers in one adapter or between different adapters, which complicates expert weight management.

To load the fine-tuned expert weights into NPU memory, a straightforward design is to maintain adapter weight tensors independent of base model weight tensors. As explained in Section 2, expert weights in the MoE layers can be represented as tensors of shape [𝑀,𝐻out, 𝐻in], where𝑀 is the number of experts in the base model. One simple solution is to allocate an additional pool of adapter expert weights as tensors of shape [𝑀𝐴, 𝐻out, 𝐻in], where 𝑀𝐴 is the total number of experts in all adapters in the system. While attractive at first, this strategy introduces intrusive modifications to the GMM operator to load weights from both tensors, making it inferior or even infeasible in case the source code for the GMM operator is not accessible.

To keep the GMM operator intact, it is necessary to manage both base model experts and adapter experts in the same three-dimensional tensor. A padding approach introduces a system-level flag 𝐸max and reserves space for 𝐸max experts per adapter: for a serving system that supports up to 𝑁 adapters in the same batch, additional space of 𝑁 · 𝐸max experts is needed; concatenating those adapter experts with the base model along the expert dimension yields tensors of shape:

[𝑀 + 𝑁 · 𝐸max, 𝐻out, 𝐻in] Note that 𝐸max is a conservative configuration to be no less than the maximum number of experts for layers across all adapters. Since most adapters fine-tune fewer than 𝐸max experts per layer, this strategy can potentially lead to significant memory inefficiency.

Memory Fragmentation Analysis. We analyze real ESFT adapters to understand the memory fragmentation of the padding approach. Due to the scarcity of off-the-shelf ESFT adapters, we choose the base model to be the ESFT vanilla model [38], a 16B MoE model sharing the same architecture as DeepSeek-V2-Lite [7], and select 10 fine-tuned ESFT adapters that cover 5 different tasks, includingmath [30], intent recognition [9], summarization [11], law [10], and translation [12].

To quantify the deviation between the number of experts in each layer and the maximum number of experts in any layer of an adapter, we define the adapter sparsity factor 𝑆𝑖 for adapter 𝑖 as:

( 𝐸𝑖 − 𝑒 (𝑙)𝑖

) 𝐿 · 𝐸𝑖

where 𝐿 is the total number of layers, 𝑒 (𝑙) 𝑖

is the number of experts fine-tuned in layer 𝑙 , and 𝐸𝑖 = max𝐿

𝑙=1 𝑒 (𝑙) 𝑖

denotes the maximum number of experts across all layers of the adapter. A value

Domain Adapter Max. #Experts Avg. #Experts Sparsity

Math gate-math 12 7.04 0.41 token-math 9 6.12 0.32

Intent gate-intent 12 9.50 0.21 token-intent 8 7.12 0.11

Summary gate-summary 11 7.73 0.30 token-summary 8 5.15 0.36

Law gate-law 12 7.35 0.39 token-law 10 6.58 0.34

Translation gate-translation 13 4.69 0.64 token-translation 6 3.85 0.36

Table 1 | Expert configuration and sparsity of 10 selected ESFT adapters across 5 domains. Max. #Ex-perts is the maximum experts in any layer, Avg. #Experts is the average experts across all layers, and Sparsity is the adapter sparsity factor 𝑆𝑖 .

of 𝑆𝑖 = 0 indicates a fully dense adapter with less inherent fragmentation, while larger values imply heavier intra-adapter padding. Table 1 reports the sparsity factor for the 10 adapters: while a small number of adapters are relatively dense, most exhibit moderate to high sparsity.

Even with dense adapters, memory fragmentation can still occur as a result of inter-adapter padding, as the tensors must be padded to the largest expert count observed across all adapters. To assess the memory efficiency of serving multiple adapters at the same time, we define the memory fragmentation factor as:

𝐹mem = 𝐿 · (𝑀 + 𝑁 · 𝐸max)∑𝐿 𝑙=1

𝑖=1 𝑒 (𝑙) 𝑖

) , The memory fragmentation factor captures the ratio between allocated memory and actual memory usage for adapter expert weights, with 𝐹mem = 1.0 indicating no fragmentation and 𝐹mem > 1.0 indicating proportional overhead. For the 10 ESFT adapters in Table 1, the smallest feasible 𝐸max = 13 yields an associated memory fragmentation factor of 𝐹mem = 1.51, indicating a 51% memory overhead beyond the necessary adapter weights due to padding alone. As the memory on device is already a scarce resource, limiting this fragmentation overhead to an acceptable degree becomes a practical challenge.

#### 3.2. Efficient Adapter-aware Runtime

After loading ESFT adapter experts into memory, they must be utilized accurately and efficiently at inference time, even when requests for different adapters are batched together and tokens for multiple adapters are interleaved. However, routers in the MoE layers still emit base-model expert IDs. Therefore, for a token of adapter 𝑖 , we must transparently and deterministically redirect

Top-K Hidden States

## Hidden States

## GMM Operator

## Physical Memory Pool

## Free Pages

Virtual Weight Tensor (§ 4.2)

## ESFT Expert Map

## Inference Runtime

Base-model Expert Adapter 1 Expert Adapter 2 Expert

Batched Rerouting (§ 4.3) Routing

Mixture-of-Experts (EP)

## Normalization

Attention (DP/TP)

## Normalization

Figure 2 | High-level architecture of ExpertWeave. Expert weights are organized in a virtual weight tensor with decoupled physical page allocation managed by a physical memory pool. An ESFT expert map records the locations of fine-tuned experts within this tensor, enabling ExpertWeave to perform batched rerouting of tokens by replacing base-model expert IDs with their adapter-specific counterparts during inference time.

each selected base-model expert to its adapter-specific counterpart if available, preserving the correct semantics of ESFT. The mechanism must be lightweight as it runs on the critical path in the forward pass of LLM inference. The mechanism must also be seamlessly integrated into existing routing and computing processes in MoE layers to preserve efficiency achieved by optimized serving engines while guaranteeing that tokens are consistently routed to the correct base-model or adapter experts, delivering task-specific outputs with minimal additional computation.

### 4. ExpertWeave: Scaling Multi-ESFT Serving

#### 4.1. Overview

We design ExpertWeave to serve multiple ESFT adapters with the following desired properties:

Non-intrusive computation path modifications that enable the dynamic replacement of adapter experts while keeping the GMM operator unchanged;

Memory efficiency that scales with the number of active adapter experts rather than frag-mentation from worst-case padding;

Negligible latency overhead relative to serving the base model.

The high-level architecture of ExpertWeave is shown in Figure 2. Requests are dispatched with an associated ESFT adapter ID; a request may also target the base model using a special marker. Tokens from requests with up to 𝑁 adapters can be packed together in the same batch. Layers not fine-tuned by ESFT are kept the same as before; batched tokens participate in computations in these layers with no modifications, enjoying advanced optimizations employed in LLM serving systems like continuous batching [18, 42] and chunked prefill [1, 19]. In order to efficiently support ESFT adapters with minimal overhead and non-intrusive modifications, ExpertWeave relies on

two core modules: virtual-memory-assisted expert weight management and a high-performance runtime through a batched rerouting operator.

ExpertWeave manages the weights of both base-model and adapter experts in a unified framework called virtual weight tensor, a single three-dimensional tensor in a contiguous virtual address space. Following the padding approach of Section 3, this tensor of shape [𝑀 + 𝑁 · 𝐸max, 𝐻out, 𝐻in] is sized to hold the weights of the base model’s𝑀 experts plus the maximum possible number of experts (𝐸max) for all 𝑁 supported adapters. While virtually contiguous, the underpinning physical memory is allocated independently and supplied on demand. This design provides a simple, unified view of all expert weights to the GMM operator, while avoiding much of the memory fragmentation issue mentioned in Section 3.

To manage expert computation at runtime, for each MoE layer 𝑙 , ExpertWeave utilizes a batched rerouting operator with an ESFT expert map Π(𝑙) , a two-dimensional array of shape [𝑁,𝑀]. An entry Π(𝑙) [𝑖, 𝑗] stores the index of expert 𝑗 of adapter 𝑖 in the virtual weight tensor: If expert 𝑗 is not fine-tuned by adapter 𝑖 , Π(𝑙) [𝑖, 𝑗] simply holds the original index of 𝑗 ; if it is fine-tuned, Π(𝑙) [𝑖, 𝑗] points to the specific location where the adapter expert weights are loaded in the virtual weight tensor:

Π(𝑙) [𝑖, 𝑗] =

{ 𝑗, if 𝑗 is not fine-tuned in adapter 𝑖, Δ𝑖 + 𝛿 (𝑙)𝑖 𝑗 , if 𝑗 is fine-tuned in adapter 𝑖

where Δ𝑖 = 𝑀 + 𝑖 · 𝐸max is the offset of adapter 𝑖 in the first dimension of the virtual weight tensor and 𝛿 (𝑙)

𝑖 𝑗 is the offset of the fine-tuned expert 𝑗 within the range [Δ𝑖 : Δ𝑖 +𝐸max] assigned to adapter

𝑖 where 0 ≤ 𝛿 (𝑙) 𝑖 𝑗

< 𝑒 (𝑙) 𝑖 .

At inference time, batched rerouting directs tokens to their appropriate experts by leveraging the ESFT expert map Π to update the top-𝑘 expert IDs selected by the MoE router. For a token 𝑥 from adapter 𝐴(𝑥), the router first selects the original set of top-𝑘 experts TopK(𝑥); batched rerouting then updates this set by replacing base-model expert indexes with their corresponding (fine-tuned or not) counterparts from the ESFT expert map Π:

TopK′(𝑥) := {Π[𝐴(𝑥), 𝑗] : 𝑗 ∈ TopK(𝑥)}

This updated set of top-𝑘 experts, TopK′(𝑥), is then passed to the unmodified inference path with token dispatching and the GMM operator, which is executed on the virtual weight tensor without needing to be aware of the adapter-specific logic.

#### 4.2. Virtual-Memory-Assisted Expert Weight Management

As described in Section 3, loading adapter expert weights following the padding approach can lead to severe memory fragmentation on device. To address this issue, ExpertWeave adopts a strategy that decouples virtual memory reservation from physical memory allocation.

Specifically, ExpertWeave reserves a contiguous virtual address space for the full weight tensor, but only allocates physical pages for the experts that are actually present in the adapters, with

0 1 2 … M-1 M M+1 …

base_addr addr →

## Virtual Weight Tensor

## Physical Memory Pool

Free PagesAllocated Pages (Adapters)Allocated Pages (Base-model)

Figure 3 | An illustration of a virtual weight tensor unfolded alongside its first dimension and a physical memory pool. Mappings for base-model experts (blue) are omitted. Adapter experts are padded to 𝐸max = 3 in the virtual address space, but only active experts (green and orange) are mapped with physical pages. Regions marked with "×" (gray) represent padding without physical pages. Misalignment between expert and page boundaries leads to partially filled pages. Free pages are available for future allocations.

no allocation for the additional regions introduced by padding. When loading adapter 𝑖 with 𝑒 (𝑙) 𝑖

experts in layer 𝑙 , physical memory is mapped only for the range:

[Δ𝑖 : Δ𝑖 + 𝑒 (𝑙)𝑖 ]

The leftover padding range [Δ𝑖 + 𝑒 (𝑙)𝑖 : Δ𝑖 + 𝐸max] is kept intentionally not mapped, ensuring that no physical memory is wasted on unused areas of the tensor.

Memory Pool and Mapping. ExpertWeave implements the virtual weight tensor through two components:

A physical memory pool per device that manages physical memory pages with fixed sizes (e.g., 2 MB granularity). The physical memory pool pre-allocates pages from the device runtime and supplies them to the virtual weight tensor at adapter loading time; evicted adapters release their pages back to the pool, which are reused for subsequent adapters or eventually reclaimed by the device runtime.

An expert memory manager per virtual weight tensor that handles physical pages allo-cated to the tensor by mapping them to the desired regions used by active experts, or the corresponding unmapping when adapters are evicted.

When a virtual weight tensor is instantiated, ExpertWeave reserves the contiguous virtual address space and returns a pointer to its base address (base_addr) without mapping any physical memory page. Upon the loading of a range of consecutive experts, either from the base model at system initialization time or from an adapter at runtime, the expert memory manager computes the starting virtual address for the range:

start_addr = base_addr + Δ𝑖 × expert_size

Then, it interacts with the pool by requesting the required number of physical pages and mapping them to the appropriate offsets in the virtual address space starting at start_addr, before copying the actual expert weights into the mapped region. Unloading a range of experts works similarly in a backward manner by unmapping the physical pages and releasing them back to the physical memory pool for subsequent reuse.

Figure 3 illustrates a virtual weight tensor in an MoE layer 𝑙 and its relationship with the physical memory pool, where the system-level 𝐸max = 3. The 𝑀 experts in the base model and their associated physical pages (blue) are created at system initialization time (with the mapping omitted). There are 𝑒 (𝑙)0 = 2 experts in the first (green) adapter and 𝑒 (𝑙)1 = 1 expert in the second (orange) adapter. Each expert consumes 1.5 pages, and there are 3 pages mapped to the region of the 2 experts in the first adapter. Regions marked with "×" (gray) represent padding within the virtual weight tensor, without physical pages assigned to these regions. Free pages in the memory pool remain available for future allocations, enabling efficient reuse.

Expert-PageAlignment. Apractical challenge of expert weightmanagement is that thememory size of a model-defined expert may not be an exact multiple of the fixed granularity of physical memory pages. This misalignment can lead to conflicts where expert boundaries in virtual address spacemay not coincide with physical page boundaries, leading to internal fragmentation in partially filled pages. Figure 3 shows an example where the start_addr of the second (orange) adapter is not aligned with the page size. Inconsiderate implementations would therefore lead to runtime errors or wasted memory.

To prevent such conflicts and maximize memory utilization, ExpertWeave adopts a sub-page allocation strategy, where the unused segment of a mapped but partially filled page is intentionally made available for use by a subsequently loaded neighboring adapter. We implement this strategy through rigorous tracking of the expert-page relationship and reference counting.

## API Short Description

aclrtReserveMemAddress Reserve virtual memory address space. aclrtMallocPhysical Create physical memory pages on the NPU. aclrtFreePhysical Free physical memory pages on the NPU. aclrtMapMem Map physical memory pages to virtual addresses. aclrtUnmapMem Unmap physical memory pages from virtual addresses.

Table 2 | Key AscendCL APIs used in virtual-memory-assisted expert weight management.

Memory Management APIs. The key Ascend Computing Language (AscendCL) APIs enabling this approach, including reserving contiguous virtual address spaces, mapping and unmapping physical pages, and managing device-level physical memory pages, are summarized in Table 2. As

Virtual Weight Tensor w/ Expert ID

0 1 … 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79

𝑙 0 1 … 63 3 14 47 × × × × × 5 13 14 27 35 57 59 ×

ESFT Expert Map Π

0 … 3 4 5 … 13 14 … 27 … 35 … 47 … 57 58 59 … 63

0 0 … 64 4 5 … 13 65 … 27 … 35 … 66 … 57 58 59 … 63

1 0 … 3 4 72 … 73 74 … 75 … 76 … 47 … 77 58 78 … 63

Top-K IDs (before) Top-K IDs (after)

AID 0 1 2 3 4 5 0 1 2 3 4 5

0 -1 15 14 45 47 3 57 15 14 45 47 3 57 0

1 -1 35 1 32 43 11 54 35 1 32 43 11 54 1

2 0 31 13 62 12 34 14 31 13 62 12 34 65 2

3 0 26 47 31 3 58 60 Batched Rerouting

26 66 31 64 58 60 3

4 -1 30 14 58 46 50 44 30 14 58 46 50 44 4

5 1 13 31 14 35 15 5 73 31 74 76 15 72 5

6 1 8 27 35 59 5 63 8 75 76 78 72 63 6

7 1 35 59 52 58 7 37 76 78 52 58 7 37 7

8 0 3 13 60 0 14 32 64 13 60 0 65 32 8

9 1 57 5 3 13 27 59 77 72 3 73 75 78 9

Figure 4 | An example of batched rerouting with𝑀 = 64, 𝐾 = 6, 𝑁 = 2, and 𝐸max = 8. The original top-𝑘 IDs are updated via the ESFT expert map Π and the adapter ID (AID) array.

shown in Section 5, the use of virtual-memory-assisted expert weight management introduces negligible overhead in ExpertWeave.

#### 4.3. Batched Rerouting

To efficiently leverage the experts in the virtual weight tensor, ExpertWeave introduces a batched rerouting operator in the forward path in each MoE layer. After the top-𝑘 expert IDs are computed by the MoE router in layer 𝑙 , ExpertWeave additionally reroutes some selected experts to their fine-tuned counterparts if they are for tokens in requests to different adapters, utilizing the aforementioned ESFT expert mapΠ(𝑙) . This operator is performed at token granularity [4], ensuring seamless compatibility with token-level scheduling algorithms like continuous batching [18, 42] and chunked prefill [1, 19].

Figure 4 shows an example of batched rerouting with a base model of 𝑀 = 64 and 𝐾 = 6, the maximum number of adapters 𝑁 = 2, and 𝐸max = 8. In layer 𝑙 , there are 𝑒 (𝑙)0 = 3 experts in the first adapter and 𝑒 (𝑙)1 = 7 experts in the second adapter. The hidden states of the scheduled tokens are paired with an adapter ID (AID) array indicating the associated adapter ID of each token; a special marker value of −1 represents a token from requests to the base model [25]. Each entry in the top-𝑘 ID array is conditionally updated by looking up the ESFT expert map Π(𝑙) using both its associated AID and its value, producing an updated top-𝑘 ID array corresponding to experts in the virtual weight tensor.

The batched rerouting operator is implemented as a series of vector-based operations, including broadcasting the AID array, computing offsets inside the ESFT expert map, and a gather operation. To reduce kernel launching overhead and redundant data copying, we implemented a fused kernel

leveraging multiple vector cores on Ascend NPUs [28]. As shown in Section 5, the operator introduces minimal overhead for online inference and does not affect service quality.

### 5. Evaluation

ExpertWeave is built on top of vLLM [25], a state-of-the-art LLM inference and serving engine, and vLLM-Ascend [37], a community-maintained plugin for vLLM on the Ascend [28] platform.

We evaluated the performance of ExpertWeave in two settings: (1) online serving with real-time dynamic workload patterns, and (2) offline batched inference for controlled microbenchmarking of system components. We then analyze the memory efficiency of ExpertWeave and further show that ExpertWeave has zero accuracy loss in downstream tasks.

#### 5.1. Experimental Setup

Hardware. We evaluated ExpertWeave on a quad-socket server with 192 ARM cores, 1.5TB of main memory, and 8 Ascend NPUs, where each NPU has 64GB of memory.

Serving Framework and Setup. ExpertWeave is implemented on top of vLLM [25] and vLLM-Ascend [37] with version v0.8.4. We use Tensor Parallelism (TP) for self-attention layers and Expert Parallelism (EP) for MoE layers when applicable.

Models and Adapters. Unlike dense models that activate the full model for each token, MoE models exhibit different expert activation patterns for different tokens. Using synthetic adapters with dummy weights would incorrectly capture system behavior, especially under a multi-adapter setup. We therefore match real ESFT adapters with queries from their respective domains to accurately measure the performance of ExpertWeave. We use the 10 adapters from Table 1; they are replicated for experiments beyond 10 adapters.

Baselines. To ensure fairness, we compared ExpertWeave with vLLM-Ascend on the same hardware setup. We used two baselines for vLLM-Ascend:

vLLM-Ascend (Merged): As there is no native support for ESFT adapters in vLLM-Ascend, we first collected merged ESFT models offline by replacing the experts in the base model with the adapter’s fine-tuned experts, then served the merged model with vLLM-Ascend; for multiple ESFT adapters, we ran multiple vLLM-Ascend instances on different NPUs and dispatched requests of each domain to its instance.

vLLM-Ascend (Base-Only): When deploying multiple full merged models was infeasible due to hardware resource constraints, we instead ran a single vLLM-Ascend instance serving only the ESFT vanilla base model and sent all requests to this instance.

When unambiguous, we shorten the baseline names to vLLM-Ascend.

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

1 2 3 4 5 Arrival Rate (req/s)

vLLM-Ascend (Base-Only) ExpertWeave (5 Adapters) ExpertWeave (10 Adapters) ExpertWeave (20 Adapters)

Figure 5 | End-to-end performance for serving multiple adapters under uniform (𝛼 = 1) and skewed (𝛼 = 0.3 and 0.1) workloads on 8 Ascend NPUs; ExpertWeave scales with increasing numbers of ESFT adapters with minor overhead compared with vLLM-Ascend.

Metrics. We reported prefill throughput, time-to-first-token (TTFT), decode throughput, and time-per-output-token (TPOT) as key performance metrics.

#### 5.2. End-to-End Performance

Workloads. For online inference, we constructed prompts by sampling from the test sets of the datasets mentioned in Section 3. Prompts from a specific dataset were sent only to adapters fine-tuned on that domain to preserve expert specialization. To evaluate ExpertWeave under workload skew, we sampled per-adapter request shares using a power-law distribution with shape parameter 𝛼 [34]: smaller 𝛼 yields heavier skew (a few adapters receive most requests) while larger 𝛼 leads to even distribution (with uniform distribution at 𝛼 = 1). We used this distribution to assign each adapter 𝑖 an arrival rate 𝜆𝑖 such that the aggregate rate reached a desired value 𝜆 =

∑𝑁 𝑖=1 𝜆𝑖 . We generated one trace per adapter following a Poisson process with arrival rate 𝜆𝑖 ,

and executed all traces concurrently over a 100-second horizon.

Serving Multiple ESFT Adapters. To evaluate how ExpertWeave scales to the number of ESFT adapters 𝑁 served in the system, we performed experiments with 𝑁 = 5, 10, and 20 adapters and compared the results against the vLLM-Ascend (Base-Only) baseline. These experiments were conducted on 8 Ascend NPUs with TP=8 (attention) and EP=8 (MoE). The aggregate request arrival rate 𝜆 varied from 1 to 5 requests-per-second (req/s).

As shown in Figure 5, under a uniform workload (𝛼 = 1), serving ESFT adapters using ExpertWeave

introduces only minimal TTFT overhead compared to vLLM-Ascend (Base-Only), approximately 8% for 5 adapters and increasing modestly to about 11% for 20 adapters. This overhead arises mainly from (1) batched rerouting and (2) more diverse expert activation in the GMM operator; nevertheless, this overhead remains minimal and acceptable under typical SLO requirements. TPOT shows a similar trend: Increasing the number of adapters from 5 to 10 and 20 results in an average overhead of roughly 4–11%.

Prefill throughput remains consistent (< 2%) across all configurations. vLLM-Ascend achieves slightly higher decode throughput due to the small overhead of serving multiple ESFT adapters in ExpertWeave. When the number of adapters increases from 5 to 20, a slight improvement in decode throughput is observed in ExpertWeave due to the variance in output lengths in different domains.

We also observed the same qualitative behavior under skewed workloads (𝛼 = 0.3 and 0.1); overall, ExpertWeave exhibits minimal overhead as 𝑁 increases, indicating effective scaling in our method.

Comparison with Serving Merged Models. Base-only deployment serves as a baseline for comparing system efficiency, but it cannot provide the same level of accuracy in downstream tasks as merged models. In this section, we therefore study how ExpertWeave compares with deploying merged models with vLLM-Ascend. Since it is not feasible to deploy multiple merged models under reasonably comparable hardware resources and deployment strategies like TP and EP without techniques like model swapping, we instead compare the efficiency of ExpertWeave against vLLM-Ascend in a controlled setting that favors the vLLM-Ascend baseline.

We deployed ExpertWeave on two NPUs with TP=2 and EP=2. We ran two independent vLLM-Ascend instances, one for each respective merged model, and ensured the same deployment strategy by deploying each instance on two NPUs with TP=2 and EP=2 in a total of four NPUs. We configured ExpertWeave to use 90% of the memory capacity via the gpu-memory-utilization flag. Each vLLM-Ascend instance was restricted to 45% of memory capacity so that aggregate memory usagewas comparable while vLLM-Ascend enjoyed twice the compute resources compared to ExpertWeave. Although this is not a fair resource setup, as we allocated more resources to vLLM-Ascend, we believe this setup is necessary as it enables us to isolate and assess the performance benefits of ExpertWeave in a multi-device deployment scenario without being affected by variance caused by deployment strategies. In fact, we show that ExpertWeave outperforms vLLM-Ascend even under this unfair resource allocation to demonstrate its efficiency.

We used two adapters, gate-math and gate-intent, and a fixed request arrival rate 𝜆 = 10. We varied the workload by controlling the shape parameter 𝛼 . At 𝛼 = 0.32, approximately 80% of requests target gate-math while the other 20% go to gate-intent. Lowering 𝛼 increases the skew further, with up to 95% of requests being directed to gate-math.

As shown in Figure 6, ExpertWeave achieves consistently better performance with around 7–14% higher prefill throughput and around 14–18% higher decode throughput across various skew levels despite being allocated fewer resources compared to vLLM-Ascend. This performance improvement arises because vLLM-Ascend serves each merged model in isolation: as skew shifts,

0.10.20.3 Skewness

0.10.20.3 Skewness

Decode vLLM-Ascend (Merged) ExpertWeave

Figure 6 | End-to-end performance for serving adapters vs. mergedmodels under skewedworkloads. ExpertWeave consistently outperforms vLLM-Ascend across skew levels.

128 256 512 1024 2048 Prompt Length

1 2 4 8 16 32 64 128256 Batch Size

Decode vLLM-Ascend (Merged) ExpertWeave-SingleOp ExpertWeave

Figure 7 | Performance of the batched rerouting fused kernel. ExpertWeave-SingleOp has non-negligible overhead relative to vLLM-Ascend, while ExpertWeave with fused kernel shows no observable overhead in TTFT or TPOT.

the NPUs used for serving gate-math become saturated while those for gate-intent become underutilized. This imbalance causes queuing delays, leading to increased latency and reduced throughput. In contrast, ExpertWeave leverages resources across all available devices regardless of request distribution, sustaining high throughput.

#### 5.3. Microbenchmarking

This ablation study isolates the effects of kernel fusion and virtual weight tensors on inference latency with offline microbenchmarking.

Workloads. We vary input prompt lengths to evaluate prefill latency and adjust batch sizes to evaluate decode latency. For prefill, we fix batch size to 1, execute queries of each prompt length 10 times, and report median TTFT. For decode, we fix the prompt length to 1,024 tokens and decode 128 steps, and report median TPOT. We use the gate-math adapter and prompts from the math domain across all microbenchmarking experiments.

Impact of Batched Rerouting. We compare the fused kernel (ExpertWeave) with (1) vLLM-Ascend (Merged) as a latency reference, and (2) ExpertWeave-SingleOp, which implements batched rerouting using canonical PyTorch operators including broadcast, gather, etc. Figure 7 shows that ExpertWeave-SingleOp incurred an average 29% slowdown in TTFT and TPOT compared to vLLM-Ascend, while our fused kernel exhibited negligible (< 1%) overhead.

128 256 512 1024 2048 Prompt Length

1 2 4 8 16 32 64 128256 Batch Size

Decode ExpertWeave-Padding ExpertWeave

Figure 8 | Effect of virtual weight tensors on TTFT and TPOT. Virtual weight tensors incur negligible overhead on inference latency.

Impact of Virtual Weight Tensor. We evaluate the performance of virtual weight tensors (ExpertWeave) against the padding baseline of Section 3 (ExpertWeave-Padding). As shown in Figure 8, ExpertWeave and ExpertWeave-Padding achieve comparable TTFT (< 3%) and TPOT (< 1%) across prompt lengths and batch sizes, demonstrating that the usage of virtual weight tensors does not degrade inference latency despite its significant memory savings and increased KV cache capacity, which we discuss in Section 5.4.

#### 5.4. Memory Efficiency

We evaluate the memory efficiency of ExpertWeave by comparing its use of virtual weight tensors with (1) vLLM-Ascend (Merged) and (2) ExpertWeave-Padding. We serve multiple adapters on a single NPU with 64GB of memory. For adapters, we use gate-math, token-math and gate-intent for our experiments.

As illustrated in Figure 9, the memory usage of vLLM-Ascend scales linearly with the number of served adapters, as it needs to deploy the full model per adapter. While it can efficiently serve a single adapter with KV cache space for up to 810K tokens, with an additional adapter, its memory usage doubles to 58.6GB and the available space for KV cache shrinks to ∼6K tokens, limiting concurrency and context length. At three adapters, memory demand exceeds ∼88GB and the system experiences out-of-memory (OOM) errors, making this configuration infeasible. In contrast, ExpertWeave with virtual weight tensors safely serves two adapters with a KV cache capacity of more than 572K tokens (a 94.4× improvement over vLLM-Ascend) and even three adapters with 477K tokens, demonstrating its scalability.

Compared with ExpertWeave-Padding, padding alone adds 4.7GB of overhead for a single adapter, whereas ExpertWeave reduces it to 2.8GB (40.4% reduction). The savings are 28.9% for two adapters and 37.3% for three adapters, enabling a 22.8–63.4% larger KV cache than the padding-based configuration. These gains could lead to higher serving throughput and KV cache capacity in online serving, a critical improvement in production inference settings where memory is a limiting resource.

1 2 3 Number of Adapters

34 39 44.3 32.1 36.2 38.7

## Model Weight

1 2 3 Number of Adapters

Available KV Cache Size vLLM-Ascend (Merged) ExpertWeave-Padding ExpertWeave

Figure 9 | The memory usage of padding and the virtual weight tensor. ExpertWeave reduces the memory fragmentation, enabling larger KV cache space for concurrent requests.

GSM8K Intent

Base Model 56.5 18.6 vLLM-Ascend + gate-math 62.3 -

vLLM-Ascend + gate-intent - 78.8

ExpertWeave 62.3 78.8

Table 3 | Accuracy of ExpertWeave with two adapters and their respective merged models. Exper-tWeave preserves per-task accuracy on multiple downstream tasks. The numbers are not directly comparable to those reported in [38] due to differences in prompts, serving engines, hardware, etc.

#### 5.5. Accuracy

Finally, we show ExpertWeave has no impact on serving quality by evaluating the system on two downstream tasks: math (GSM8K [30]) and intent recognition [38]. We configure ExpertWeave to load two adapters, gate-math and gate-intent, and evaluate accuracy on both tasks. For vLLM-Ascend, we use the corresponding merged models.

Table 3 shows ExpertWeave is able to serve multiple adapters while matching per-task accuracy of the respective merged model for both tasks. This result validates that ExpertWeave is a robust system for multi-adapter serving without any accuracy loss.

### 6. Related Work

Advanced Multi-LoRA Serving. Building on top of systems like Punica [4] and S-LoRA [34], existing efforts focus on request scheduling to improve multi-LoRA serving [21, 27, 39]. dLoRA [39] addresses skewed request distribution where a few adapters dominate request traffic by utilizing a credit-based batching algorithm to decide when to merge or unmerge adapters and a request-adapter co-migration strategy to move requests with their adapters across nodes for load balancing. Chameleon [21] proposes an adapter-aware, non-preemptive multi-queue scheduler that avoids head-of-line (HOL) blocking. Other works on multi-LoRA serving incorporate quantization tech-niques [40] and co-design with prefix caching strategies [45]. CoLD [17] combines multi-LoRA serving with contrastive decoding to enhance model performance in downstream tasks. However,

these systems are designed exclusively for LoRA adapters, which limit their adoption.

Serving MoE LLMs under Constrained Resources. Deploying large MoE models in envi-ronments with limited GPU memory is challenging due to the large memory footprint of expert weights and limited batching efficiency from dynamic routing. MoE-Lightning [3] proposes a CPU-GPU-I/O pipeline scheduler with paged weights to overlap data movement and computation under tight memory constraints. MoE-Lens [44] introduces a resource-aware scheduler, an execution engine that overlaps prefill and decode stages and offloads attention computation to the CPU for efficient execution. While related, these methods are optimized for offline, throughput-oriented batch inference scenarios; our method instead targets online, latency-sensitive serving setups by sharing a single base model across multiple served adapters.

Virtual Memory Management for LLMs. Beyond the classic PagedAttention [25], recent work has explored the use of virtual memory management (VMM) APIs to improve memory management for LLMs [15, 31, 41, 43]. GMLake [15] uses low-level CUDA VMM APIs to mitigate memory fragmentation during large-scale training. For inference workloads, vAttention [31] and vTensor [41] use VMM APIs to manage the KV cache in single-LLM serving without relying on PagedAttention. Prism [43] extends this line of work to multi-LLM serving with cross-model memory coordination to flexibly share GPU memory. Orthogonal to virtual-memory-assisted KV cache management, our approach focuses on efficient model weight management in a multi-adapter scenario and can be combined with these approaches to further improve memory efficiency, which we leave for future work.

### 7. Conclusion

We presented ExpertWeave, a system for efficiently serving multiple Expert-Specialized Fine-Tuning (ESFT) adapters over a shared Mixture-of-Experts (MoE) base model. To improve memory utilization, ExpertWeave introduces virtual-memory-assisted expert weight management that efficiently handles the placement of base-model and adapter experts in memory, avoiding frag-mentation. This design significantly reduces memory consumption compared to the vLLM-Ascend baseline that merges ESFT adapters into the base model, enabling up to 94× larger KV cache capacity when serving two adapters and maintaining scalability as the number of adapters in-creases. To minimize runtime overhead, ExpertWeave incorporates a fused kernel for batched rerouting, which provides lightweight adapter-aware token dispatching with negligible additional latency, while substantially outperforming unoptimized implementations. Our evaluation further shows that ExpertWeave achieves up to 18% higher throughput compared to vLLM-Ascend using comparable resources. Moreover, when serving multiple ESFT adapters concurrently, ExpertWeave preserves accuracy equivalent to the vLLM-Ascend baseline serving merged models, confirming that efficiency gains do not come at the cost of quality. Finally, scalability experiments demonstrate that ExpertWeave scales efficiently, with only a minor 4–11% latency increase even when serving 20 ESFT adapters. By enabling efficient base-model sharing across ESFT adapters, ExpertWeave

makes large-scale, multi-tenant deployment of specialized MoEmodels practical and cost-effective.

## References

[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming {Throughput-Latency} tradeoff in {LLM} inference with {Sarathi-Serve}. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134, 2024.

[2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

[3] Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph E Gonzalez, Matei Zaharia, and Ion Stoica. Moe-lightning: High-throughput moe inference on memory-constrained gpus. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pages 715–730, 2025.

[4] Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, and Arvind Krishnamurthy. Punica: Multi-tenant lora serving. In P. Gibbons, G. Pekhimenko, and C. De Sa, editors, Proceedings of Machine Learning and Systems, volume 6, pages 1–13, 2024.

[5] Huawei Ascend Community. torch_npu.npu_grouped_matmul — npu_grouped_matmul api. https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref /apilist/ptaoplist_000160.html, 2025. Ascend Extension for PyTorch 7.0.0 API documentation.

[6] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specializa-tion in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

[7] DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

[8] DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[9] DeepSeek-AI. ESFT Evaluation Dataset – intent.jsonl. https://github.com/deeps eek-ai/ESFT/blob/main/datasets/eval/intent.jsonl#L3, 2024. Accessed: 2025-06-02.

[10] DeepSeek-AI. ESFT Evaluation Dataset – law.jsonl. https://github.com/deepseek-a i/ESFT/blob/main/datasets/eval/law.jsonl#L3, 2024. Accessed: 2025-06-02.

[11] DeepSeek-AI. ESFT Evaluation Dataset – summary.jsonl. https://github.com/deeps eek-ai/ESFT/blob/main/datasets/eval/summary.jsonl#L3, 2024. Accessed: 2025-06-02.

[12] DeepSeek-AI. ESFT Evaluation Dataset – translation.jsonl. https://github.com/d eepseek-ai/ESFT/blob/main/datasets/eval/translation.jsonl#L3, 2024. Accessed: 2025-06-02.

[13] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International conference on machine learning, pages 5547–5569. PMLR, 2022.

[14] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

[15] Cong Guo, Rui Zhang, Jiale Xu, Jingwen Leng, Zihan Liu, Ziyu Huang, Minyi Guo, Hao Wu, Shouren Zhao, Junping Zhao, et al. Gmlake: Efficient and transparent gpu memory defrag-mentation for large-scale dnn training with virtual memory stitching. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pages 450–466, 2024.

[16] Z Han, C Gao, J Liu, J Zhang, and S Qian Zhang. Parameter-efficient fine-tuning for large models: A comprehensive survey. arxiv 2024. arXiv preprint arXiv:2403.14608.

[17] Morgan Lindsay Heisler, Linzi Xing, Ge Shi, Hanieh Sadri, Gursimran Singh, Weiwei Zhang, Tao Ye, Ying Xiong, Yong Zhang, and Zhenan Fan. Enhancing learned knowledge in lora adapters through efficient contrastive decoding on ascend npus. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 4499–4510, 2025.

[18] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, et al. Deepspeed-fastgen: High-throughput text generation for llms via mii and deepspeed-inference. arXiv preprint arXiv:2401.08671, 2024.

[19] Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, et al. Inference without interference: Disag-gregate llm inference for mixed downstream workloads. arXiv preprint arXiv:2401.11181, 2024.

[20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

[21] Nikoleta Iliakopoulou, Jovan Stojkovic, Chloe Alverti, Tianyin Xu, Hubertus Franke, and Josep Torrellas. Chameleon: Adaptive caching and scheduling for many-adapter llm inference environments. arXiv preprint arXiv:2411.17741, 2024.

[22] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

[23] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024.

[24] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994.

[25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, CodyHao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memorymanagement for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 611–626, New York, NY, USA, 2023. Association for Computing Machinery.

[26] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

[27] Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng, Xusheng Chen, Yizhou Shan, Binhang Yuan, and Wei Wang. Caraserve: Cpu-assisted and rank-aware lora serving for generative llm inference. arXiv preprint arXiv:2401.11240, 2024.

[28] Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu. Ascend: a scalable and unified architecture for ubiquitous deep neural network computing : Indus-try track paper. In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 789–801, 2021.

[29] AI Meta. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[30] OpenAI. GSM8K: Grade School Math 8K Dataset. https://huggingface.co/dataset s/openai/gsm8k, 2021. Accessed: 2025-06-02.

[31] Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, and Ashish Pan-war. vattention: Dynamic memory management for serving llms without pagedatten-tion. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pages 1133–1150, 2025.

[32] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

[33] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[34] Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, and Ion Stoica. Slora: Scalable serving of thousands of lora adapters. In P. Gibbons, G. Pekhimenko, and C. De Sa, editors, Proceedings of Machine Learning and Systems, volume 6, pages 296–311, 2024.

[35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[37] vLLM Community. vllm ascend plugin. https://github.com/vllm-project/vll m-ascend, 2024. Accessed: 2025-06-05.

[38] Zihan Wang, Deli Chen, Damai Dai, Runxin Xu, Zhuoshu Li, and Yu Wu. Let the expert stick to his last: Expert-specialized fine-tuning for sparse architectural large language models. arXiv preprint arXiv:2407.01906, 2024.

[39] Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, and Xin Jin. dlora: dy-namically orchestrating requests and adapters for lora llm serving. In Proceedings of the 18th USENIX Conference on Operating Systems Design and Implementation, OSDI’24, USA, 2024. USENIX Association.

[40] Yifei Xia, Fangcheng Fu, Wentao Zhang, Jiawei Jiang, and Bin Cui. Efficient multi-task llm quantization and serving for multiple lora adapters. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2025. Curran Associates Inc.

[41] Jiale Xu, Rui Zhang, Cong Guo, Weiming Hu, Zihan Liu, Feiyang Wu, Yu Feng, Shixuan Sun, Changxu Shao, Yuhong Guo, et al. vtensor: Flexible virtual tensor management for efficient llm serving. arXiv preprint arXiv:2407.15309, 2024.

[42] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for {Transformer-Based} generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 521–538, 2022.

[43] Shan Yu, Jiarong Xing, Yifan Qiao, Mingyuan Ma, Yangmin Li, Yang Wang, Shuo Yang, Zhiqiang Xie, Shiyi Cao, Ke Bao, et al. Prism: Unleashing gpu sharing for cost-efficient multi-llm serving. arXiv preprint arXiv:2505.04021, 2025.

[44] Yichao Yuan, Lin Ma, and Nishil Talati. Moe-lens: Towards the hardware limit of high-throughput moe llm serving under resource constraints. arXiv preprint arXiv:2504.09345, 2025.

[45] Hang Zhang, Jiuchen Shi, Yixiao Wang, Quan Chen, Yizhou Shan, and Minyi Guo. Improving the serving performance of multi-lora large language models via efficient lora and kv cache management. arXiv preprint arXiv:2505.03756, 2025.

[46] Barret Zoph. Designing effective sparse expert models. In 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pages 1044–1044. IEEE, 2022.

