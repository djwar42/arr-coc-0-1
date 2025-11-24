---
sourceFile: "SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:47.911Z"
---

# SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving - arXiv

3216010f-d2e1-41e8-a92b-a56a77219772

SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving - arXiv

e0e97519-e30f-4ace-9e15-49aaba54987c

https://www.arxiv.org/pdf/2509.24626

SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving

## Qihui Zhou The Chinese University of Hong Kong

qhzhou@cse.cuhk.edu.hk

## Peiqi Yin The Chinese University of Hong Kong

pqyin22@cse.cuhk.edu.hk

Pengfei Zuo∗ Huawei Cloud

pengfei.zuo@huawei.com

## James Cheng The Chinese University of Hong Kong

jcheng@cse.cuhk.edu.hk

Abstract Serving long-context LLMs is costly because attention com-putation grows linearly with context length. Dynamic sparse attention algorithms (DSAs) mitigate this by attending only to the key-value (KV) cache of critical tokens. However, with DSAs, the main performance bottleneck shifts from HBM bandwidth toHBM capacity: KV caches for unselected tokens must remain in HBM for low-latency decoding, constrain-ing parallel batch size and stalling further throughput gains. Offloading these underutilized KV caches to DRAM could free HBM capacity, allowing larger parallel batch sizes. Yet, achieving such hierarchical HBM-DRAM storage raises new challenges, including fragmented KV cache access, HBM cache contention, and high HBM demands of hybrid batch-ing, that remain unresolved in prior work. This paper proposes SparseServe, an LLM serving sys-

tem that unlocks the parallel potential of DSAs through effi-cient hierarchical HBM-DRAM management. SparseServe introduces three key innovations to address the challenges mentioned above: (1) fragmentation-aware KV cache trans-fer, which accelerates HBM-DRAM data movement through GPU-direct loading (FlashH2D) and CPU-assisted saving (FlashD2H); (2) working-set-aware batch size control that ad-justs batch sizes based on real-time working set estimation to minimize HBM cache thrashing; (3) layer-segmented prefill that bounds HBM use during prefill to a single layer, enabling efficient execution even for long prompts. Extensive experi-mental results demonstrate that SparseServe achieves up to 9.26× lower mean time-to-first-token (TTFT) latency and up to 3.14× higher token generation throughput compared to state-of-the-art LLM serving systems.

1 Introduction The rapid advancement of large language models (LLMs) has reshaped our daily lives. As the demand for long-context applications such as sophisticated reasoning [7, 47, 51] and document analysis [6, 24] continues to grow, the ability of LLMs to process long sequences has become increasingly

∗Pengfei Zuo is the corresponding author.

critical. To meet this need, recent models have expanded their context windows to over one million tokens [3, 29, 37]. However, serving long-context LLMs at scale incurs ex-

tremely high inference costs because attention computation grows linearly with sequence length. At each decoding step, all key-value (KV) cache entries are fetched from HBM to compute units. Since the KV cache itself expands linearly with sequence length, attention computation cost increases proportionally and becomes bounded by HBM bandwidth. To reduce the inference cost of long-context LLMs, dy-

namic sparse attention algorithms (DSAs) [9, 13, 41, 45, 49] compress the KV cache during decoding, thereby reducing HBM bandwidth demands. DSAs exploit the observation that only a small set of critical tokens largely determines the output token, and that token criticality varies across query tokens. In self-attention, these critical tokens exhibit signifi-cantly higher attention scores (𝑄𝑇𝐾) than others, allowing accurate approximation of attention using only their KV cache. DSAs implement this through a select-then-compute manner: they partition the KV cache into blocks of consec-utive tokens, maintain compact metadata for each block, and for every query token, estimate block criticality from its metadata to select the top-𝑘 KV blocks for approximate attention. However, with DSAs, the main performance bottleneck

shifts from HBM bandwidth to HBM capacity. To ensure low-latency decoding under uncertain token selection, all KV blocks must reside in HBM, even though only a small subset is accessed for each decoding step. This leads to poor HBM capacity utilization and limits the parallel potential of DSAs to scale throughput by increasing batch sizes.

Hierarchical memory management—where offloading all KV blocks to host memory (DRAM) while dynamically fetch-ing only critical blocks into HBM for attention computa-tion—offers a promising solution for reducing HBM pres-sure [44, 45]. However, naive offloading can introduce signif-icant decoding latency and degrade system throughput due to the limited DRAM access bandwidth. Enabling efficient hierarchical memory management for DSAs faces three key challenges that prior work has not addressed.

1) Fragmented KV cache transfers reduce effective DRAM access bandwidth. The physical bandwidth from device (GPU) to host memory (DRAM) is limited. For an NVIDIA A100 40GB GPU, the D2H bandwidth via PCIe Gen4 is only 32 GB/s, compared to 1.6 TB/s for HBM. This bottle-neck is further exacerbated by fragmented KV cache accesses of DSA. Specifically, existing DSAs [9, 41] typically set the KV block size to 32 tokens and store the KV blocks of each at-tention head separately, which results in only 16 KB per block for popular long-context model like LWM-7B [29]. Fetching such small blocks via standard cudaMemcpy achieves an ef-fective bandwidth of less than 4 GB/s, far below the PCIe peak and orders of magnitude lower than HBM.

2) Efficient runtime control of parallel batch sizes. For DSAs, small batch sizes underutilize GPU resources, limit-ing throughput. However, larger batch sizes do not always improve throughput. This is because large batch sizes may exacerbate HBM cache contention, leading to frequent KV cache evictions and increased data transfers fromDRAM. Bal-ancing batch size dynamically at runtime is therefore critical to maximize efficiency while avoiding HBM bottlenecks. We measure throughput and the average number of KV blocks loaded per iteration across varying batch sizes, as shown in Figure 1. The throughput initially increases by 2.07× when the batch size grows from 2 to 6. However, further increasing in batch size leads to a notable drop. For instance, expanding the batch size from 6 to 12 reduces throughput by 1.73×. The decline is caused by the excessive KV cache loading associ-ated with larger batch sizes. Specifically, the average number of KV blocks loaded per iteration increases by 21.36×, when increasing the batch size from 6 to 12. 3) High HBM requirement of chunked prefill. To en-

hance GPU utilization, modern LLM serving systems widely adopt hybrid batching, combining compute-intensive prefill and memory-intensive decoding requests within the same batch. To prevent generation stalls, long prompts are divided into smaller chunks and processed over multiple batches, which is known as chunked prefill [2]. While chunked prefill effectively reduces the computation per iteration, it cannot reduce the HBM consumption of request prefilling, since ex-ecuting each prefill chunk requires the KV cache generated by all preceding chunks. As a result, requests with long input prompts may experience head-of-line blocking, as they must wait for currently running requests to complete and release sufficient HBM for execution. To address these challenges, this paper introduces Spars-

eServe, an LLM serving system with efficient hierarchical HBM-DRAMmanagement designed for efficient deployment of DSAs in long-context inference. To unlock the full parallel potential of DSAs, SparseServe incorporates three key sys-tem innovations to overcome the deployment bottlenecks. Firstly, to accelerate fragmented KV cache loading and

saving, SparseServe introduces FlashH2D and FlashD2H, two fragmentation-aware transfer engines that optimize KV cache

movement between GPU HBM and host DRAM for DSAs. FlashH2D exploits the unified virtual addressing (UVA) ca-pabilities of modern GPUs to perform GPU-direct loading, fusing multiple small KV block reads into a single kernel execution. FlashD2H adopts CPU-assisted saving, which first transfers the contiguous but unorganized KV cache into a DRAM buffer through a single cudaMemcpy and then utilizes CPU threads to asynchronously scatter the data into the corresponding KV blocks. Secondly, to determine the optimal request batch sizes,

SparseServe employs a working-set-aware batch size control strategy that dynamically adjusts batch sizes at runtime to avoid HBM cache thrashing. By exploiting the temporal local-ity of block selection, the working set of frequently accessed KV blocks can be estimated from previous iterations. Spars-eServe ensures that the aggregated working set of scheduled requests stays within HBM capacity, reducing excessive KV cache loading resulting from cache contention. Finally, to limit the HBM footprint during prefill, Spars-

eServe introduces layer-segmented prefill, a new mechanism that conducts prefill layer by layer. This method evicts KV blocks of preceding layers to DRAM as subsequent layers are processed, thereby bounding the KV cache footprint of prefill to a single layer. To prevent generation stalls, each layer is further divided into multiple segments with each segment processed in a separate batch, significantly reducing batch running time.

We implement SparseServe based on vLLM [23] and evalu-ate its performance using two popular long-context LLMs on the LongBench [5] dataset. Experimental results show that, compared to vanilla vLLM, SparseServe reduces the mean time-to-first-token (TTFT) latency by up to 9.26× under the same request rates and improves token generation through-put by up to 3.14×. The experimental results also indicate that the designs of SparseServe are effective in improving the maximum request throughput under the service level ob-jective (SLO) requirements. To summarize, this paper makes the following contributions:

## We identify the parallel potential of DSAs and highlight

three fundamental challenges that limit its realization.

We propose SparseServe, a long-context LLM serving sys-

tem that unlocks this parallelism through efficient hierar-chical HBM-DRAMmanagement, featuring fragmentation-aware KV cache transfer, working-set-aware batch size control, and layer-segmented prefill.

We implement SparseServe atop vLLM [23] and exten-sively evaluate it, demonstrating substantial performance improvements over vanilla vLLM and vLLM enhanced with state-of-the-art DSA.

2 4 6 8 10 12 Batch Size

Figure 1. Token generation throughput and average number of KV blocks loaded per iteration under varying batch sizes.

2 Background and Motivation 2.1 Generative LLM Inference Basics Transformer. The transformer has emerged as the stan-dard model architecture for modern LLMs [32, 33]. These LLMs are typically composed of a chain of transformer layers, each containing two basic modules: self-attention and feed-forward network (FFN). During inference, the input query token list 𝑋 = [𝑥1, 𝑥2, ...𝑥𝑁𝑞

] for each layer is first multiplied by three weight matrices𝑊𝑞 ,𝑊𝑘 , and𝑊𝑣 to generate query (𝑄 ∈ 𝑅𝑁𝑞×𝑑 ), key (𝐾 ∈ 𝑅𝑁𝑘𝑣×𝑑 ), and value (𝑉 ∈ 𝑅𝑁𝑘𝑣×𝑑 ) matrices, where 𝑁𝑞 is the number of query tokens, 𝑁𝑘𝑣 is the number of KVs, and 𝑑 is the hidden dimension. The self-attention is then conducted using 𝑄 , 𝐾 , and 𝑉 :

𝑆 =𝑄 · 𝐾𝑇 / √ 𝑑, 𝑃 = 𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥 (𝑆), 𝑂 = 𝑃 ·𝑉

Here, 𝑃 represents the attention weights, with 𝑃𝑖, 𝑗 indicating the importance of 𝐾 𝑗 to query token 𝑖 . The attention output 𝑂 is then fed into the FFN module. The FFN output serves as the input of the next transformer layer. Auto-regressive generation. Generative LLM inference proceeds in two phases: prefill and decoding. The prefill phase processes the input prompt in parallel to produce the first output token, with the prefill latency is measured by time-to-first-token (TTFT). The decoding phase then gen-erates tokens auto-regressively, where each new token be-comes the input for the next iteration. The latency for each generated token is measured by time-between-token (TBT). KV cache. During the generation of new tokens, the key and value vectors of all previous tokens are required for the self-attention computation. As a result, they are cached in HBM to avoid repeated computation, referred to as KV cache. Since the KV cache grows with the decoding process and the number of decoding steps is unknown, existing sys-tems generally employ PagedAttention [23] to divide HBM into fixed-size blocks and store the KV cache in multiple discontinuous blocks to avoid HBM fragmentation. Hybrid batching and chunked prefill. During the prefill phase, all prompt tokens are processed in parallel in a sin-gle iteration, enabling efficient utilization of GPU compute resources. In contrast, the decoding phase involves a full for-ward pass of the LLM model over a single token generated

## Query Token

## Estimate Importance Approximate Attention

Figure 2.Workflow of dynamic sparse attention.

in the previous iteration. This leads to low compute utiliza-tion and makes decode memory-bound. To improve GPU utilization, hybrid batching [1, 2, 53] is proposed to combine the prefill and decoding phases of different requests into the same batch. However, due to the processing of a large num-ber of tokens, the latency of a prefill iteration is significantly higher than that of a decoding iteration, leading to high TBT. To address this issue, each input prompt for prefill is divided into multiple chunks, i.e., chunked prefill [2]. These chunks are scheduled per iteration with ongoing decoding requests, effectively reducing the TBT.

2.2 Dynamic Sparse Attention Due to the auto-regressive nature of LLMs, generating each token necessitates loading the entire KV cache from GPU HBM to on-chip SRAM, which results in significant time and space overheads for long-context LLM serving. Recent works [46, 50] have observed that the attention

computation is highly sparse, with only a small portion of to-kens contributing to the majority of attention weights. Based on this observation, dynamic sparse attention algorithms (DSAs) [9, 41, 45] have been proposed. Figure 2 illustrates the general workflow of existing DSAs, where a small portion of critical KV cache for each query token are dynamically selected for attention computation. Since not all KV cache is required for attention computation, DSAs allow KV cache to be offloaded to DRAM and load only the selected KV blocks into GPUs each time to reduce HBM consumption. To speed up the selection process, inspired by the block-

level memory allocation of KV cache in PagedAttention [23], DSAs divide KV cache into blocks and select KV cache at the block level. For each KV block, DSAs construct metadata vectors to represent the tokens within it. Different DSAs pro-pose various metadata construction methods, ranging from simply calculating the mean values of the token keys [45] to finding the bounding cuboid of the token keys [9]. Re-gardless of the methods, the size of the metadata is much smaller than the KV block. To estimate the importance of KV blocks to each query token, DSAs compute dot products between the metadata vectors and the query token to obtain approximate attention scores for all KV blocks. DSAs then select the top-𝑘 most critical KV blocks to perform attention.

FlashD2H (§3.2.2)

## Finished Request ID Block ID

## KV Cache Manager

## GPU Memory Cached KV BlocksMetadata

## LLM Model Executor

## Request Scheduler

## Decode Queue

## CPU Memory KV Block Storage

FlashH2D (§3.2.1)

## Prefill Queue

## Request Generated Token

Batch Size Control (§3.3)

Layer-segmented

Prefill (§3.4)

Figure 3. The system architecture for SparseServe.

2.3 Motivation and Challenge With DSAs, the main performance bottleneck shifts from HBM bandwidth to HBM capacity. To guarantee low-latency decoding under uncertain token selection, all KV blocks must be kept in GPU HBM, even though only a small subset is accessed at each decoding step. This results in inefficient HBM utilization and constrains the ability of DSAs to scale throughput by increasing batch sizes.

A promising solution is hierarchical memorymanagement, which offloads all KV blocks to host DRAM while dynami-cally fetching only the critical blocks into HBM for attention computation [44, 45]. This approach alleviates HBM capacity pressure and opens up more room for batch-level scaling. However, naive offloading can incur severe decoding de-

lays and degrade overall throughput due to the limited access bandwidth of DRAM. Realizing efficient hierarchical mem-ory management for DSAs requires addressing three key challenges that prior work has left open, as outlined in §1: 1) fragmented KV cache transfers that reduce effective DRAM bandwidth, 2) lack of efficient runtime control for batch siz-ing, and 3) the high HBM footprint of chunked prefill.

3 The SparseServe System In this section, we present SparseServe, an LLM serving system with efficient hierarchical HBM-DRAMmanagement designed for unlocking the parallel potential of DSAs.

3.1 System Overview Figure 3 illustrates the overall system architecture for Spars-eServe, which comprises three key components: the request scheduler, the model executor, and the KV cache manager.

❶ Request Scheduler.The request scheduler determineswhich requests are executed in each iteration. Similar to exist-ing LLM serving systems, it adopts dynamic batching techniques [48] and schedules requests in a first-come-first-served (FCFS) manner. In addition, the scheduler in SparseServe incorporates a working-set-aware batch size control strategy (§ 3.3) to prevent GPU cache thrashing by avoiding excessive KV block loading. Additionally, the

4 8 16 32 Block Size

memcpy FlashH2D

4 8 16 32 Block Size

memcpy FlashD2H

Figure 4. PCIe bandwidth of KV cache (a) loading with memcpy and FlashH2D, and (b) saving with memcpy and FlashD2H, under varying block sizes.

scheduler leverages the layer-segmented prefill technique (§ 3.4), which divides the layers of the prefill phase of a request into distinct segments. These segments are exe-cuted in separate batches, thereby reducing the memory footprint and limiting the runtime of each iteration.

❷ Model Executor. The model executor performs the com-putation of model forwards. It replaces the standard at-tention with the sparse attention for decoding requests. It also tracks the prefill layers to execute in each batch and skips the other layers based on the layer-segmented prefill strategy (§ 3.4). During execution, the model execu-tor communicates with the KV cache manager to transmit newly generated KV cache data and retrieve the metadata of the relevant KV blocks. In addition, it sends the indices of the KV blocks required for the attention computation to the KV cache manager to trigger the KV cache load-ing from DRAM to HBM (§ 3.2.1). The metadata is used to estimate the criticality of KV blocks for each query token. By default, SparseServe adopts the cuboid-mean method to construct the metadata for KV blocks due to its high accuracy [9]. However, other metadata construction methods [41, 45] can be easily integrated into SparseServe.

❸ KV Cache Manager. The KV cache manager maintains a hierarchical KV cache between HBM and DRAM. Both HBM and DRAM are organized into fixed-size blocks to mitigate memory fragmentation [23] and are managed independently per attention head. The KV cache manager receives newly generated KV caches from the model ex-ecutor and saves them to the corresponding KV blocks in the DRAM (§ 3.2.2). Once a KV block reaches its ca-pacity, its associated metadata is created. The metadata is retained in HBM due to its small size and is utilized in every attention computation. The remaining HBM is used to cache frequently accessed KV blocks and we employ the least recently used (LRU) cache eviction policy, which leverages the cosine similarity between consecutive query tokens [44, 45]. Specifically, query vectors of consecutive tokens exhibit high similarity, leading to the selection of similar KV blocks.

T1H1 T1H2 T1H3 T2H1 T2H2 T2H3 T3H1 T3H2 T3H3

T1H1 T2H1 T3H1 T1H2 T2H2 T3H2 T1H3 T2H3 T3H3

Figure 5. A comparison between the (𝑁,𝐻, 𝐷) and (𝐻, 𝑁, 𝐷) KV block layouts. 𝑁 , 𝐻 , and 𝐷 denote the token, head, and hidden dimension, respectively.

## DRAM KV Block Storage

## KV Cache GPU

## MemcpyMemcpyMemcpyMemcpyMemcpy

Figure 6.A demonstration of the fragmented KV cache block loading from DRAM to HBM using a memcpy-based method.

3.2 Fragmentation-Aware KV Cache Transfer There are two layouts to organize the KV cache of each to-ken in a KV block: (𝑁,𝐻, 𝐷) and (𝐻, 𝑁, 𝐷). Figure 5 demon-strates this with an example of three tokens and three heads. In the (𝑁,𝐻, 𝐷) layout, all KV heads of a token is stored con-tiguously. In contrast, the (𝐻, 𝑁, 𝐷) layout groups the KV cache for all tokens associated with a head together. Since DSAs select KV blocks at the head level, the (𝐻, 𝑁, 𝐷) layout is adopted for efficient block selection and memory access. With hierarchical HBM–DRAM storage for KV caches,

both loading and saving KV blocks to DRAM can stall LLM computation. This bottleneck is further exacerbated by frag-mented KV cache accesses, which occurs at the granularity of KV heads. To mitigate the overhead of KV cache transfers, SparseServe adopts FlashD2H&H2D, a fragmentation-aware transfer mechanism that tailors GPU-direct loading and CPU-assisted saving to their unique characteristics.

3.2.1 FlashH2D: GPU-Direct Loading. Before attention computation in each model layer, DSAs dynamically select and load KV blocks from DRAM to HBM on a per-head ba-sis. Figure 6 illustrates this process using the conventional memcpy-based approach. Because memcpy requires contigu-ous source and destination buffers, each KV block scattered in DRAM must be copied individually, resulting in many memcpy calls. This causes substantial function invocation overhead and poor PCIe bandwidth utilization, especially when the number of KV blocks is large.

To accelerate such fragmented KV cache loading, Spars-eServe employs a GPU-direct loading strategy, by leveraging unified virtual addressing (UVA) supported by modern GPUs. UVA allows GPU kernels to directly access the DRAM. In-stead of issuing multiple memcpy calls, SparseServe launches a single GPU kernel that loads all selected KV blocks from DRAM in parallel. The kernel assigns one thread block per

## Block Storage

## Newly Generated

## DRAM Buffer

3 3 3 4 4 4

Token-3 Token-4

3 3 3 4 4 4

Save (CPU multi-threads)

HBM Cache 3 4

Cached KVs (HND)

Save (GPU kernel)

Figure 7. A demonstration of the execution workflow of the proposed CPU-assisted KV cache saving.

KV block, adapting the number of threads dynamically to the actual number of selected KV blocks in each iteration. By fusing all KV block loads into one GPU kernel, GPU-direct loading minimizes invocation overhead and effectively in-creases PCIe bandwidth utilization. As shown in Figure 4a, FlashH2D consistently delivers PCIe bandwidth exceeding 20 GB/s across varying block sizes, significantly outperforming memcpy, whose bandwidth stays under 5 GB/s.

3.2.2 FlashD2H: CPU-Assisted Saving. At the begin-ning of each model layer, the hidden states from the previ-ous layer are projected into a KV tensor of shape (𝐵,𝐻, 𝐷), where 𝐵 denotes the total number of tokens in the current batch, 𝐻 is the number of attention heads, and 𝐷 is the head dimension. The resulting KV tensor is then saved into the free slots in the corresponding KV blocks in HBM. Once a KV block is full, it is asynchronously flushed into DRAM using a separate CUDA stream, overlapping saving with model computation. However, since the full KV blocks in HBM are scattered across non-continuous addresses, saving suffers from the same fragmented data movement issue as loading. As a result, the saving latency can exceed the model com-putation latency. This problem is more prominent during prefill, when many KV caches are generated at the same time. One might consider using GPU kernels to accelerate saving as with loading. However, this approach consumes GPU re-sources and interferes with model computation, leading to prolonged execution time. To address this issue, we propose CPU-assisted KV cache

saving in SparseServe, as illustrated in Figure 7. Our key observation is that the KV tensor generated in each iteration is continuous before saving to the KV blocks. Leveraging this property, SparseServe decomposes saving into two steps: (1) the contiguous KV tensor is first copied into a DRAM buffer with a single memcpy, and (2) once the transfer completes, CPU threads redistributed the buffered data into the corre-sponding DRAM KV blocks. Crucially, the CPU-assisted sav-ing method avoids consuming GPU computation resources, allowing saving to execute fully in parallel with model com-putation without any interference. As shown in Figure 4b,

2 4 6 8 10 12 14 16

## Window Size

Figure 8. Average overlap ratios between the KV blocks accessed in preceding decoding steps and those selected in the current decoding step. We refer to the number of preceding decoding steps considered as window size.

Algorithm 1: The scheduling algorithm. 1 R𝑚𝑎𝑥 : The maximum number of requests per batch. 2 T𝑚𝑎𝑥 : The maximum number of tokens per batch. 3 M𝑎𝑣𝑙 : The available HBM size. 4 S: The scheduler in existing LLM serving systems. Output: The batch B𝑟𝑒𝑠 for the next iteration.

5 B𝑖𝑛𝑖𝑡 ←S.getBatch(R𝑚𝑎𝑥, T𝑚𝑎𝑥)

6 B𝑟𝑒𝑠 ← {} 7 M𝑢𝑠𝑒𝑑 ← 0 8 for request 𝑟𝑒𝑞 ∈ B𝑖𝑛𝑖𝑡 do 9 M𝑟𝑒𝑞 ← estimateWS(𝑟𝑒𝑞)

10 ifM𝑢𝑠𝑒𝑑 +M𝑟𝑒𝑞 ≤M𝑎𝑣𝑙 then 11 B𝑟𝑒𝑠 .add(𝑟𝑒𝑞) 12 M𝑢𝑠𝑒𝑑 ←M𝑢𝑠𝑒𝑑 +M𝑟𝑒𝑞

13 else 14 S.reset(𝑟𝑒𝑞) 15 return B𝑟𝑒𝑠

FlashD2H consistently delivers PCIe bandwidth exceeding 23 GB/s across varying block sizes, significantly outperforming memcpy, whose bandwidth stays under 6 GB/s.

3.3 Working-Set-Aware Batch Size Control To determine the optimal request batch size, SparseServe employs a working-set-aware batch size control strategy that dynamically adjusts batch sizes at runtime. The key idea is to estimate whether the working set—the total HBM capacity required by the KV cache of all running requests in the iteration—fits within HBM, thereby avoiding cache thrashing and excessive KV block loading. We first present how to estimate working set sizes for prefill and decoding requests, and then describe the scheduling workflow. Prefill working set. The working set of a prefill request is the total HBM capacity required to store its KV cache during the current iteration. This value can be computed exactly, as the prefill process is deterministic. However, the

Iteration 1

Iteration 2

Iteration 3

1 2 3 4 10 6 9

Iteration 4

1 2 3 Token 1/2/3 Token from Req. 1/2/3/4 Token for Prefill / Decode

Figure 9. The workflow of layer-segmented prefill with a running example. (The LLM model consists of three layers. Each batch contains three decoding requests and one prefill request, with segment size set to 1.)

working set size varies depending on the prefill strategy. For chunked prefill, the working set includes the KV cache from all preceding token chunks across all layers, since each chunk depends on previously generated KV caches. In contrast, for the layer-segmented prefill proposed in this paper (§ 3.4), the working set consists of only one layer of KV cache, since the KV caches of all previous layers are not required and can be evicted into DRAM once processed. Decoding working set. Unlike prefill, the working set size of a decoding request cannot be directly calculated since the selected KV blocks at each decoding step vary dynamically. To address this, SparseServe estimates the working set size by leveraging strong temporal locality: consecutive query tokens often select highly overlapping KV blocks [9, 44, 45].

We measure the average overlap ratios between KV blocks selected in the current decoding step and those selected in the preceding decoding steps (Figure 8). In the experiment, we vary the history window size (the number of preceding decoding steps considered). Experiments on LWM-7B [29] model across LongBench datasets [5] show consistently high overlaps. The overlap ratios increase sharply with the win-dow size initially, but soon plateau. For example, expanding the window from 1 to 12 steps improves overlap by 10.68%, while growing it further from 12 to 16 adds only 0.31%. This suggests that it is sufficient to retain only a bounded his-tory. Accordingly, SparseServe tracks the KV blocks selected over the past𝑤 decoding steps (with𝑤 = 12 by default) and regards their union as the decoding working set. Scheduling workflow.We implement working-set-aware batch size control in SparseServe by extending the sched-uler of existing LLM serving systems, ensuring compatibility with prior designs. The scheduler operates under three input constraints for each iteration: (1) R𝑚𝑎𝑥 , which bounds the maximum number of requests in a batch; (2) T𝑚𝑎𝑥 , which limits the total number of tokens in a batch to control the computational workload, especially for prefill requests; and

(3)M𝑎𝑣𝑙 , which specifies the available GPU HBM cache ca-pacity. The first two constraints are commonly used in ex-isting systems. In contrast, SparseServe introducesM𝑎𝑣𝑙 , to ensure that the HBM usage of each request batch remains within cache limits, thereby preventing cache thrashing and HBM contention among requests. To form a batch, SparseServe first invokes the existing

scheduling logic to construct an initial candidate batch that satisfies R𝑚𝑎𝑥 and T𝑚𝑎𝑥 constraints (Line 5). Then, Spars-eServe estimates the working set size for each request in this initial batch, and adds the request to the current execution batch only if the total HBM usage withinM𝑎𝑣𝑙 (Lines 8-12). If not, the request is rejected and its state is reset (Lines 13-14).

3.4 Layer-Segmented Prefill To achieve both low HBM consumption and low TBT when used with DSAs, SparseServe introduces layer-segmented prefill, a new prefill mechanism designed for hybrid batching. The key observation is that LLMs are composed of multiple layers and the model forwarding is conducted layer by layer. Although the entire prefill of a long input prompt can be time-consuming, the execution of a single layer is relatively fast. This suggests that we can divide prefill into layer segments and process these segments in separate batches, bounding the runtime of each batch. Importantly, since the input prompt is not chunked, the KV blocks of each layer are accessed only once during prefill. The KV blocks of all finished layers can be immediately evicted after being saved to DRAM and the released HBM space can be reused for subsequent layers. This design bounds the HBM footprint of prefill to a single layer at any given time.

Figure 9 demonstrates the workflow of the proposed layer-segmented prefill with a running example. The LLM model consists of three layers and each batch includes three decod-ing requests alongside one prefill request (the last iteration contains four decoding requests). With a segment size of 1, prefill is completed over three iterations (batches). Each iteration executes only one prefill layer alongside decoding requests and skips the remaining layers to ensure low batch running time. After executing each iteration except for the last one, activation states from the executed prefill layer are saved and used to resume prefill in the next iteration. Determining segment size. The appropriate segment size for layer-segmented prefill depends on three factors: prompt length, TBT SLO, and system throughput. To achieve a low TBT, a small segment size is preferred, especially for long prompts. However, this slows down prefill and reduces through-put. To meet the varying demands in practice, SparseServe provides a configurable parameter called maxInjectToken, which limits the maximum number of prefill tokens injected into a batch. In practice, users can find the value for maxIn-jectToken through profiling experiments. Specifically, users can set maxInjectToken to a small initial value and gradually

increase it until reaching the TBT limit, thus maximizing throughput under the given SLO. Combination with chunked prefill. Layer-segmented pre-fill can be combined with chunked prefill for extremely long input prompts. If a single layer’s execution time already ex-ceeds the TBT, we partition each layer into smaller chunks like chunked prefill, and process these chunks across mul-tiple batches. This hybrid strategy provides fine-grained la-tency control while retaining the HBM efficiency benefits of layer-wise segmentation.

4 Performance Evaluation 4.1 Experimental Setup Testbed. Our experiments are conducted on a machine host-ing an Nvidia A100 GPU with 40 GB HBM, an AMD EPYC 7J13 CPU, and 256 GB DRAM. The GPU is connected to the host via PCIe Gen 4, providing a bandwidth of 32 GB/s. Models. The experiments evaluate two popular long-context LLM models: LWM-7B [29] with a 1M context window and Llama3-8B [35] with a 262K context window. In particular, the LWM-7B model employs the same model architecture as Llama2-7B [42]. These two models cover two mainstream attention methods, namely, multi-head attention (MHA) and grouped query attention (GQA). Baselines. SparseServe is implemented based on vLLM [23], which is a state-of-the-art LLM serving system that employs full KV cache attention without sparsity. We use vLLM as the first baseline for performance comparison. Next, we imple-ment dynamic sparse attentions [9] on vLLM as the second baseline, which is referred to as vLLM-S. In addition, we further enhance vLLM-S with KV cache offloading, which results in the third baseline called vLLM-SO. Workload.We conduct experiments using multiple datasets from LongBench [5], which covers various task types in-cluding Question Answering (Qasper [12], NarrativeQA [22], MultifieldQA [19], Dureader [18]), Document Summarization (GovReport [20], QMSum [52], MultiNews [14], VCSum [43]), and Code Generation (LCC [17], RepoBench-P [30]). We com-bine the requests from all datasets into one trace to reflect real-world LLM serving scenarios where requests from dif-ferent tasks are handled simultaneously. Following prior works [1, 48], we generate request arrival times based on a Poisson distribution with varying arrival rates, donated as re-quest rate. To prevent vLLM from aborting requests with KV cache sizes exceeding HBM capacity, we limit the maximum prompt length of the requests. For LWM-7B and Llama3-8B, the maximum length is set to 32k and 128k, respectively.

4.2 End-to-End Performance We evaluate the end-to-end performance of different schemes to demonstrate the efficiency of SparseServe. For SparseServe, vLLM-S, and vLLM-SO, the token budget of attention KV

0.1 0.2 0.3 Req Rate (req/s)

(a) LWM-7B

0.1 0.2 0.3 Req Rate (req/s)

(b) Llama3-8B

vLLM SparseServe vLLM-S vLLM-SO

Figure 10. The mean TTFT of all systems under varying request rate on LongBench with LWM-7B and Llama3-8B.

0.1 0.2 0.3 Req Rate (req/s)

(a) LWM-7B

0.1 0.2 0.3 Req Rate (req/s)

(b) Llama3-8B

vLLM SparseServe vLLM-S vLLM-SO

Figure 11. The mean token generation throughput of all systems under varying request rates on LongBench with LWM-7B and Llama3-8B models.

cache is set to 2,048 for LWM-7B and Llama3-8B, ensuring that the accuracy of sparse attention achieves 99% of full at-tention, which is typical in production scenarios [8, 11]. Due to page limit, we report the detailed accuracy of SparseServe under various token budgets in supplemental materials. For vLLM, vLLM-S, and vLLM-SO, we use a token chunk size of 2,048 for chunked prefill. For a fair comparison, the layer-segmented prefill of SparseServe processes the same number of tokens as the chunked prefill in each iteration by setting the maxInjectToken as 𝐵 · 𝐿, where 𝐵 is the chunk size for chunked prefill and 𝐿 is the number of model layers. TTFT. Figure 10 shows the mean TTFT of all systems under varying request rates. At low request rates, the TTFTs of all systems are similar. However, as the request rate increases, vLLM quickly exhausts HBM with its KV cache, blocking new requests and significantly prolonging queuing time. For example, at 0.125 req/s for LWM-7B, the TTFT of vLLM is 9.26× higher than that of SparseServe. Sparse attention en-ables vLLM-S to reduce TTFT compared with vLLM due to faster decoding, while vLLM-SO further lowers TTFT at low rates by supporting larger batches via KV offloading. However, at high request rates, the TTFT of vLLM-SO be-comes worse than vLLM and vLLM-S due to excessive KV block loading latency. In contrast, SparseServe consistently achieves the lowest TTFT across request rates by combining sparse attention with its system-level optimizations.

0.05 0.1 0.15 Req Rate (req/s)

(a) LWM-7B

0.1 0.15 0.2 0.25

Req Rate (req/s) 10

(b) Llama3-8B

vLLM SparseServe vLLM-S vLLM-SO

Figure 12. The mean TBT of all systems under varying request rates on LongBench with LWM-7B and Llama3-8B.

Token generation throughput. Figure 11 reports the to-ken generation throughput. Due to the excessive running time of vLLM-SO under high request rates, we cap the maxi-mum request rates of vLLM-SO at 0.1 and 0.2 RPS for LWM-7B and Llama3-8B, respectively. All systems achieve sim-ilar throughput at low request rates. As the request rate increases, the throughput of vLLM and vLLM-S reaches a plateau due to the small batches limited by the HBM. By applying sparse attention, vLLM-S achieves higher through-put than vLLM due to lower decoding latency. Although vLLM-SO enables larger batch sizes by offloading KV cache to DRAM, its throughput is worse than vLLM-S due to heavy KV cache loading overhead. In contrast, SparseServe con-sistently delivers the highest throughput compared with all baselines. Specifically, compared with vLLM, SparseServe achieves throughput improvements of up to 2.93× and 3.14× for LWM-7B and Llama3-8B, respectively. Against vLLM-S, SparseServe achieves up to 2.23× and 2.03× improvements, and against vLLM-SO, up to 2.96× and 4.24×, respectively. TBT. Figure 12 shows the mean TBT of all systems under varying request rates. To prevent the excessive request queu-ing of vLLM, we cap its maximum request rates at 0.15 and 0.25 RPS for LWM-7B and Llama3-8B, respectively. Com-pared with vLLM, vLLM-S reduces attention computation time andmaintains the same average batch sizes, thus achiev-ing lower TBT. Due to the larger average batch sizes and the heavy KV cache loading overhead, vLLM-SO achieves the highest TBT among all systems, which is consistent with the results in Figures 10 and 11. Thanks to the proposed system designs, SparseServe effectively reduces the KV block load-ing overhead and achieves slightly higher TBT than vLLM, which is within 20% for both LWM-7B and Llama3-8B. We argue that this minor TBT degradation is a reasonable trade-off, as SparseServe achieves significantly lower TTFT and higher token generation throughput compared to vLLM. Goodput. Finally, we analyze the incremental impact of SparseServe’s design components on goodput, defined as the maximum sustainable request throughput under SLOs, as shown in Figure 13. Following prior work [1, 34], we define the SLOs for the P99 TBT as 25× the execution time of a

Llama3-8B 0

vLLM vLLM + SA vLLM + SA + Offload

vLLM + SA + Offload + FT vLLM + SA + Offload + FT + WC SparseServe (vLLM + SA + Offload + FT + WC + LP)

Figure 13. The maximum request throughput under SLO re-quirement of the designs in SparseServe under varying input request rates on LongBench with LWM-7B and Llama3-8B models (SA: sparse attention; Offload: KV cache offloading; FT: fragmentation-aware KV cache transfer; WC: working-set-aware batch size control; LP: layer-segmented prefill).

decoding iteration. In addition, we ensure the sustainability of the maximum input request load by imposing a threshold on request queuing delay. Specifically, the mean scheduling delay for input requests is limited to 2 seconds to prevent excessive queuing, as suggested in [1].

Starting from vLLM, adding sparse attention (vLLM+SA) improves goodput by 1.20× (LWM-7B) and 1.13× (Llama3-8B) by reducing decoding latency. Offloading (vLLM+SA+Offload) further boosts goodput by 1.33× and 1.12× by lowering HBM usage and enabling larger batches. Fragmentation-aware transfer (vLLM+SA+Offload+FT) brings larger gains (1.88× and 1.19×) by accelerating fragmented KV transfers and tolerating higher cache miss rates. Working-set-aware batch control (vLLM+SA+Offload+FT+WC) contributes an additional 1.33× and 1.11× improvement by prevent-ing GPU cache thrashing. Finally, layer-segmented prefill (vLLM+SA+Offload+FT+WC+LP) completes SparseServe, improving goodput by another 1.25× and 1.10× by reducing prefill memory demands and lowering queuing delays for long prompts. Collectively, these optimizations enable Spars-eServe to improve goodput by up to 5.00× on LWM-7B and 1.83× on Llama3-8B compared with vLLM.

4.3 Ablation Studies We conduct ablation studies with LWM-7B on LongBench to isolate the effects of the three proposed designs in Spars-eServe: fragmentation-aware KV cache transfer, working-set-aware batch size control, and layer-segmented prefill.

4.3.1 Fragmentation-Aware KV Cache Transfer.

FlashH2D: GPU-direct KV cache loading. Figure 14a shows the mean batch latency and KV cache loading latency between memcpy-based loading and FlashH2D. We observe that the KV cache loading time accounts for a substantial portion of the total batch latency, particularly at larger batch sizes. For example, when the batch size is 8, the KV cache loading accounts for 69.94% of the total batch latency. In

4 6 8 Batch Size

0 20 40 60 80

Prefill Comp.

Memcpy GPU-direct

Memcpy FlashH2D Loading Other

Figure 14. (a) The mean batch latency and the mean KV cache loading latency of memcpy-base and FlashH2D with varying running batch sizes. (b) The mean prefill latencies of memcpy-based, GPU-direct, and FlashD2H normalized to the standalone prefill computation time.

0.2 0.225 0.25 0.275 0.3 Req Rate (req/s)

0.2 0.225 0.25 0.275 0.3 Req Rate (req/s)

W/O Batch Size Control      With Batch Size Control

Figure 15. The throughput as well as the mean KV block loading numbers with and without the working-set-aware batch size control under varying request rates.

contrast, FlashH2D effectively eliminates this bottleneck, reducing KV cache loading latency by up to 9.97× compared to the memcpy-based baseline. FlashD2H: CPU-assisted KV cache saving. Figure 14b shows the mean prefill latency, normalized to the standalone prefill computation time, for three KV cache saving meth-ods: memcpy-based saving, GPU-direct saving, and FlashD2H. Prefill is chosen as the evaluation phase because it gener-ates a large volume of new KV cache blocks, making the saving overhead more prominent. We execute the model computation and KV cache saving with different CUDA streams to overlap their executions. We observe that the mean prefill latency with the memcpy-based method is 1.76× longer than the prefill computation time. This overhead re-sults from fragmented KV block saving via memcpy, which cannot be fully hidden by computation. For the GPU-direct saving method, the prefill latency is 1.28× longer than the computation time. The reason is that GPU-direct saving con-sumes GPU resources, introducing contention that prolongs the model computation phase. In contrast, the prefill latency with FlashD2H is the same as the prefill computation time, indicating that saving newly generated KV cache to DRAM can be fully overlapped with the prefill computation, thus introducing no overhead.

4.3.2 Working-Set-Aware Batch Size Control. 9

0.15 0.2 0.25

Req Rate (req/s)

512 1024 2048

## Chunk Size

## Chunked Prefill      Layer Segmented Prefill

Figure 16. (a) The mean TTFT of chunked prefill and layer-segmented prefill under varying request rates. (b) The over-head of chunked prefill and layer-segmented prefill in atten-tion computation normalized to the cost of plain prefill with varying token chunk sizes.

We evaluate the proposed working-set-aware batch size con-trol bymeasuring token generation throughput and themean number of KV block loads per iteration, as shown in Figure 15. At low request rates, the token generation throughput is sim-ilar with and without batch size control. This is because the batch sizes are small and the GPU cache is sufficient to hold the working sets of all scheduled requests, resulting in few KV block loads. However, as the request rate increases, the throughput of the baseline (without batch size control) starts to decrease. For example, when the request rate increases from 0.25 to 0.3 RPS, the throughput drops by 29.52% due to a sharp increase in KV block loads, as shown on the right side of Figure 15. In contrast, working-set-aware control mit-igates cache contention, cutting KV block loads by 52.78× at 0.3 RPS. Consequently, throughput continues to increase steadily with higher request rates.

4.3.3 Layer-Segmented vs. Chunked Prefill.

TTFT reduction. Figure 16a shows the mean TTFT with the proposed layer-segmented prefill and chunked prefill under varying request rates. The chunk size is set to 2,048 for chunked prefill. For a fair comparison, we configure the layer-segmented prefill to process the same number of tokens as the chunked prefill in each iteration.We observe that when the request rate is low, the TTFT of layer-segmented prefill is similar to that of chunked prefill. However, by increasing the request rate, the number of ongoing decoding requests increases and increases the HBM utilization, which starts blocking prefill execution due to HBM shortage. In contrast, layer-segment prefill reduces the HBM requirement during prefilling, lowering queuing time and TTFT. It reduces the mean TTFT by up to 8.68×, compared to chunked prefill. Prefill computation overhead. Figure 16b shows the over-head of chunked prefill and the layer-segmented prefill com-pared to plain prefill regarding the attention time during prefilling. We observe that chunked prefill incurs high over-head when the chunk size is small. Specifically, with a chunk size of 512, chunked prefill slows down the prefill attention by 1.51×. This is because chunked prefill necessitates the

repeated loading of the KV cache of all preceding chunks to process the latest chunk. In contrast, layer-segmented prefill exhibits performance nearly identical to that of plain prefill, thereby minimizing the overhead.

5 Related Work Static sparse attention. Several KV cache eviction algo-rithms, e.g., H2O [50], StreamingLLM [46], SnapKV [27], FastGen [16], and Scissorhands [31], have been proposed to retain only the KV cache of important tokens while dis-carding others to save HBM. However, since the importance of tokens changes during the decoding process, discarded tokens may become crucial for future computation [41], re-sulting in potential accuracy loss. Dynamic sparse attention. DSAs, such as ArkVale [9], InfLLM [45], Quest [41], mitigate the issue of static sparse attention by dynamically selecting a small portion of the critical KV cache for attention computation for each query token while retaining all KV cache. While existing DSAs predominantly focus on enhancing the accuracy of impor-tant KV cache identification, SparseServe is the first work to consider their deployment efficiency in practical LLM serv-ing systems, achieving both high accuracy and efficiency through system designs tailored for DSAs. Token-level sparse attention. Recent works, such as In-finiGen [25], TokenSelect [44], RetrievalAttention [28], and MagicPig [10], perform KV cache selection at the granular-ity of tokens. Although token-level selection can identify important tokens more accurately in theory, it is overly gran-ular and incurs significant runtime overhead [9]. In contrast, block-level selection achieves a balance between accuracy and performance overhead, as a result of which, this paper focuses on block-level DSA approaches. Sparse attention for prefill and training. There are also works [15, 21, 39] that apply sparse attention to acceler-ate the prefill phase and training of LLMs. GemFilter [39] prunes unimportant tokens using attention matrices from early layers to reduce the computational load in subsequent layers. Minference [21] recognizes three general patterns of sparse attention in long-context LLMs and provides op-timized CUDA kernels for each pattern. SeerAttention [15] extends Minference by replacing fixed patterns with a learn-able approach. Native sparse attention (NSA) [49] introduced by DeepSeek is the first to perform sparse attention during training. These methods are orthogonal to our SparseServe and can be combined with it to further enhance the efficiency of end-to-end LLM processing. Inference parameter offloading. DeepSpeed Inference [4, 36] offloads model parameters to DRAM and fetches them on demand. Lina [26] leverages sparse activation in mixture-of-experts (MoE) models to offload cold experts to DRAM. PowerInfer [40] utilizes the sparsity in FFN computation to

offload inactive weights to DRAM, saving HBM and compu-tational resources. FlexGen [38] offloads both model parame-ters and KV cache to DRAM, targeting offline processing. In contrast, SparseServe exploits KV cache offloading in online LLM serving by utilizing the sparsity in KV cache.

6 Conclusion This paper presents SparseServe, an efficient long-context LLM serving system that unlocks the parallel potential of DSAs through efficient hierarchical HBM–DRAM manage-ment. To achieve efficient and scalable DSA deployment, SparseServe incorporates three core techniques, including fragmentation-aware KV cache transfer, working-set-aware batch size control, and layer-segmented prefill. Extensive experimental results demonstrate that SparseServe reduces the mean TTFT by up to 9.26× and increases the token gener-ation throughput by up to 3.14× compared to state-of-the-art LLM serving systems.

References [1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun

Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024. USENIX Association, 117–134. https://www.usenix.org/ conference/osdi24/presentation/agrawal

[2] AmeyAgrawal, Ashish Panwar, JayashreeMohan, Nipun Kwatra, Bhar-gav S. Gulavani, and Ramachandran Ramjee. 2023. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. CoRR abs/2308.16369 (2023). https://doi.org/10.48550/ARXIV.2308.16369 arXiv:2308.16369

[3] Gradient AI. 2024. llama-3-8B-Gradient. https://huggingface.co/ gradientai/Llama-3-8B-Instruct-262k

[4] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, and Yuxiong He. 2022. DeepSpeed- Inference: En-abling Efficient Inference of Transformer Models at Unprecedented Scale. In SC22: International Conference for High Performance Comput-ing, Networking, Storage and Analysis, Dallas, TX, USA, November 13-18, 2022. IEEE, 46:1–46:15. https://doi.org/10.1109/SC41404.2022.00051

[5] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhid-ian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yux-iao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In Proceed-ings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Sriku-mar (Eds.). Association for Computational Linguistics, 3119–3137. https://doi.org/10.18653/V1/2024.ACL-LONG.172

[6] Ahsaas Bajaj, Pavitra Dangati, Kalpesh Krishna, Pradhiksha Ashok Kumar, Rheeya Uppaal, Bradford Windsor, Eliot Brenner, Dominic Dotterrer, Rajarshi Das, and AndrewMcCallum. 2021. Long Document Summarization in a Low Resource Setting using Pretrained Language Models. In Proceedings of the ACL-IJCNLP 2021 Student Research Work-shop, ACL 2021, Online, JUli 5-10, 2021. Association for Computational Linguistics, 71–80. https://doi.org/10.18653/V1/2021.ACL-SRW.7

[7] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hu-bert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph of

Thoughts: Solving Elaborate Problems with Large Language Models. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelli-gence, IAAI 2024, Fourteenth Symposium on Educational Advances in Ar-tificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada. AAAI Press, 17682–17690. https://doi.org/10.1609/AAAI.V38I16.29720

[8] Tolga Bolukbasi, Joseph Wang, Ofer Dekel, and Venkatesh Saligrama. 2017. Adaptive Neural Networks for Efficient Inference. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 (Proceedings of Machine Learning Research), Vol. 70. PMLR, 527–536. http://proceedings.mlr. press/v70/bolukbasi17a.html

[9] Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei, Shengen Yan, Meng Li, and Yun Liang. 2024. ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction. In Advances in Neural Information Process-ing Systems 38: Annual Conference on Neural Information Process-ing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. http://papers.nips.cc/paper_files/paper/2024/hash/ cd4b49379efac6e84186a3ffce108c37-Abstract-Conference.html

[10] Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Léon Bottou, Zhihao Jia, and Beidi Chen. 2024. MagicPIG: LSH Sampling for Efficient LLM Generation. CoRR abs/2410.16179 (2024). https://doi.org/10. 48550/ARXIV.2410.16179 arXiv:2410.16179

[11] Yinwei Dai, Rui Pan, Anand P. Iyer, Kai Li, and Ravi Netravali. 2024. Apparate: Rethinking Early Exits to Tame Latency-Throughput Ten-sions inML Serving. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP 2024, Austin, TX, USA, November 4-6, 2024. ACM, 607–623. https://doi.org/10.1145/3694715.3695963

[12] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Compu-tational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021. Association for Computational Linguistics, 4599–4610. https://doi.org/10.18653/V1/2021.NAACL-MAIN.365

[13] DeepSeek-AI. 2025. DeepSeek-V3.2-Exp: Boosting Long-Context Effi-ciency with DeepSeek Sparse Attention.

[14] Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. 2019. Multi-News: a Large-Scale Multi-Document Summariza-tion Dataset and Abstractive Hierarchical Model. CoRR abs/1906.01749 (2019). arXiv:1906.01749 http://arxiv.org/abs/1906.01749

[15] Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Hayden Kwok-Hay So, Ting Cao, Fan Yang, and Mao Yang. 2024. SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs. CoRR abs/2410.13276 (2024). https://doi.org/10.48550/ARXIV.2410.13276 arXiv:2410.13276

[16] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=uNrFpDPMyo

[17] Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Julian J. McAuley. 2023. LongCoder: A Long-Range Pre-trained Language Model for Code Completion. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA (Proceedings of Machine Learning Research), Vol. 202. PMLR, 12098–12107. https: //proceedings.mlr.press/v202/guo23j.html

[18] Wei He, Kai Liu, Jing Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, Xuan Liu, Tian Wu, and Haifeng Wang. 2018. DuReader: a Chinese Machine Reading Com-prehension Dataset from Real-world Applications. In Proceedings of the Workshop on Machine Reading for Question Answering@ACL 2018, Melbourne, Australia, July 19, 2018. Association for Computational

Linguistics, 37–46. https://doi.org/10.18653/V1/W18-2605 [19] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko

Aizawa. 2020. Constructing A Multi-hop QA Dataset for Compre-hensive Evaluation of Reasoning Steps. In Proceedings of the 28th In-ternational Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020. International Com-mittee on Computational Linguistics, 6609–6625. https://doi.org/10. 18653/V1/2020.COLING-MAIN.580

[20] Luyang Huang, Shuyang Cao, Nikolaus Nova Parulian, Heng Ji, and Lu Wang. 2021. Efficient Attentions for Long Document Summarization. CoRR abs/2104.02112 (2021). arXiv:2104.02112 https://arxiv.org/abs/ 2104.02112

[21] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xu-fang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. MInference 1.0: Ac-celerating Pre-filling for Long-Context LLMs via Dynamic Sparse At-tention. CoRR abs/2407.02490 (2024). https://doi.org/10.48550/ARXIV. 2407.02490 arXiv:2407.02490

[22] Tomás Kociský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA Reading Comprehension Challenge. Trans. Assoc. Comput. Linguistics 6 (2018), 317–328. https://doi.org/10.1162/TACL_ A_00023

[23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serv-ing with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023. ACM, 611–626. https://doi.org/10.1145/3600006.3613165

[24] Md. Tahmid Rahman Laskar, Mizanur Rahman, Israt Jahan, Enamul Hoque, and Jimmy Xiangji Huang. 2023. Can Large Language Models Fix Data Annotation Errors? An Empirical Study Using Debatepedia for Query-Focused Text Summarization. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023. Association for Computational Linguistics, 10245–10255. https: //doi.org/10.18653/V1/2023.FINDINGS-EMNLP.686

[25] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. In 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024. USENIX Association, 155–172. https: //www.usenix.org/conference/osdi24/presentation/lee

[26] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating Distributed MoE Training and Inference with Lina. In Proceedings of the 2023 USENIX Annual Technical Conference, USENIX ATC 2023, Boston, MA, USA, July 10-12, 2023. USENIX Association, 945– 959. https://www.usenix.org/conference/atc23/presentation/li-jiamin

[27] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM Knows What You are Looking for Before Gener-ation. CoRR abs/2404.14469 (2024). https://doi.org/10.48550/ARXIV. 2404.14469 arXiv:2404.14469

[28] Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. 2024. RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval. CoRR abs/2409.10516 (2024). https://doi.org/10.48550/ARXIV.2409.10516 arXiv:2409.10516

[29] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 2024. World Model onMillion-Length VideoAnd LanguageWith Blockwise RingAt-tention. CoRR abs/2402.08268 (2024). https://doi.org/10.48550/ARXIV. 2402.08268 arXiv:2402.08268

[30] Tianyang Liu, Canwen Xu, and Julian J. McAuley. 2024. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems. In

The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https: //openreview.net/forum?id=pPjZIOuQuF

[31] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. In Advances in Neural Infor-mation Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, Decem-ber 10 - 16, 2023. http://papers.nips.cc/paper_files/paper/2023/hash/ a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html

[32] Inc. Meta Platforms. 2023. Llama 3.1. https://huggingface.co/meta-llama/Llama-3.1-8B

[33] OpenAI. 2023. GPT-4 Technical Report. CoRR abs/2303.08774 (2023). https://doi.org/10.48550/ARXIV.2303.08774 arXiv:2303.08774

[34] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In 51st ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2024, Buenos Aires, Argentina, June 29 - July 3, 2024. IEEE, 118–132. https: //doi.org/10.1109/ISCA59077.2024.00019

[35] Leonid Pekelis, Michael Feil, Forrest Moret, Mark Huang, and Tiffany Peng. 2024. Llama 3 Gradient: A series of long context mod-els. https://gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models

[36] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. 2021. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. CoRR abs/2104.07857 (2021). arXiv:2104.07857 https://arxiv.org/abs/2104.07857

[37] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Ro-han Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Mal-colm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard Ives, James Keeling, Karel Lenc, SalemHaykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, and et al. 2024. Gemini 1.5: Unlocking multimodal un-derstanding across millions of tokens of context. CoRR abs/2403.05530 (2024). https://doi.org/10.48550/ARXIV.2403.05530 arXiv:2403.05530

[38] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput Generative Inference of Large Lan-guage Models with a Single GPU. In International Conference on Ma-chine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA (Proceedings of Machine Learning Research), Vol. 202. PMLR, 31094– 31116. https://proceedings.mlr.press/v202/sheng23a.html

[39] Zhenmei Shi, Yifei Ming, Xuan-Phi Nguyen, Yingyu Liang, and Shafiq Joty. 2024. Discovering the Gems in Early Layers: Accelerating Long-Context LLMswith 1000x Input Token Reduction. CoRR abs/2409.17422 (2024). https://doi.org/10.48550/ARXIV.2409.17422 arXiv:2409.17422

[40] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2024. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP 2024, Austin, TX, USA, November 4-6, 2024. ACM, 590– 606. https://doi.org/10.1145/3694715.3695964

[41] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024.

OpenReview.net. https://openreview.net/forum?id=KzACYw0MTV [42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Alma-

hairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xi-ang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. CoRR abs/2307.09288 (2023). https://doi.org/10.48550/ARXIV.2307.09288 arXiv:2307.09288

[43] Han Wu, Mingjie Zhan, Haochen Tan, Zhaohui Hou, Ding Liang, and Linqi Song. 2023. VCSUM: A Versatile Chinese Meeting Sum-marization Dataset. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023. Association for Computational Linguistics, 6065–6079. https://doi.org/10.18653/V1/ 2023.FINDINGS-ACL.377

[44] Wei Wu, Zhuoshi Pan, Chao Wang, Liyi Chen, Yunchu Bai, Kun Fu, Zheng Wang, and Hui Xiong. 2024. TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection. https://api.semanticscholar.org/ CorpusID:273821700

[45] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. 2024. InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understand-ing Extremely Long Sequences with Training-Free Memory. CoRR abs/2402.04617 (2024). https://doi.org/10.48550/ARXIV.2402.04617 arXiv:2402.04617

[46] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient Streaming Language Models with Attention Sinks. In The Twelfth International Conference on Learning Represen-tations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=NG7sS51zVF

[47] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Prob-lem Solving with Large Language Models. In Advances in Neural Infor-mation Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, Decem-ber 10 - 16, 2023. http://papers.nips.cc/paper_files/paper/2023/hash/ 271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html

[48] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In 16th USENIX Sympo-sium on Operating Systems Design and Implementation, OSDI 2022, Carlsbad, CA, USA, July 11-13, 2022. USENIX Association, 521–538. https://www.usenix.org/conference/osdi22/presentation/yu

[49] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. 2025. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. https://api.semanticscholar. org/CorpusID:276408911

[50] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lian-min Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of

Large Language Models. In Advances in Neural Information Process-ing Systems 36: Annual Conference on Neural Information Process-ing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023. http://papers.nips.cc/paper_files/paper/2023/hash/ 6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html

[51] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023. Auto-matic Chain of Thought Prompting in Large Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https: //openreview.net/forum?id=5NTt8GFjUHkr

[52] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir R. Radev. 2021. QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization. CoRR abs/2104.05938 (2021). arXiv:2104.05938 https://arxiv.org/abs/2104.05938

[53] Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Yufei Gao, Qinyu Xu, Tian Tang, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. 2024. NanoFlow: Towards Optimal Large Language Model Serving Throughput. CoRR abs/2408.12757 (2024). https://doi.org/10.48550/ ARXIV.2408.12757 arXiv:2408.12757

A Appendices A.1 Model Accuracy We evaluate model accuracy on the LongBench [5] datasets under varying token budgets for attention computation, the results are shown in Table 1. The experiments are conducted using the state-of-the-art DSA ArkVale [9]. We observe that both LWM-7B [29] and Llama3-8B [35] retain 99% of the accuracy achievedwith full attention across all datasets when the token budget is set to 2048. Based on this observation, we adopt a token budget of 2048 for sparse attention in our paper.

Table 1.Model accuracy with varying token budgets.

LWM-Text-Chat-1M Llama-3-8B-262k

Dataset Full 0.5k 1k 1.5k 2k Full 0.5k 1k 1.5k 2k

HotpotQA 21.93 22.22 22.54 22.16 22.22 17.79 17.99 17.70 17.86 17.83 2WikiMultihopQA 18.01 18.22 17.67 18.13 18.47 16.90 16.51 16.46 16.70 16.77

MuSiQue 10.36 10.78 10.37 10.65 10.25 9.52 9.17 9.17 9.42 9.83 DuReader 25.68 25.71 27.02 26.29 26.03 27.47 27.25 27.79 27.76 27.47

MultiFieldQA-en 43.05 43.38 43.77 42.77 42.51 40.27 40.24 41.19 40.56 40.31 NarrativeQA 13.50 13.15 13.79 13.64 13.68 16.99 17.14 16.60 16.76 16.95

Qasper 24.08 24.12 25.06 24.62 24.49 26.21 25.54 25.59 25.93 26.23 GovReport 27.88 26.36 27.07 27.35 27.48 33.61 30.01 31.95 32.63 33.19 QMSum 24.83 24.30 24.63 24.85 24.95 25.51 25.09 25.80 25.69 25.54

MultiNews 24.38 23.62 23.98 24.00 24.32 27.87 27.34 27.74 27.90 27.75 VCSUM 9.52 11.09 10.57 10.66 10.45 14.54 14.15 13.89 13.98 14.36 TriviaQA 61.87 60.57 60.76 62.01 61.65 85.83 86.46 86.41 86.38 86.44 SAMSum 39.91 39.56 39.77 39.93 39.68 41.62 40.48 41.17 41.29 41.02 LSHT 24.00 22.00 23.00 23.50 24.00 43.50 35.50 39.50 42.00 43.00 LCC 40.47 39.15 40.20 40.14 40.21 51.04 50.38 51.14 51.03 51.02

RepoBench-P 42.78 41.56 42.75 42.65 42.84 44.79 44.46 45.48 45.04 44.75

