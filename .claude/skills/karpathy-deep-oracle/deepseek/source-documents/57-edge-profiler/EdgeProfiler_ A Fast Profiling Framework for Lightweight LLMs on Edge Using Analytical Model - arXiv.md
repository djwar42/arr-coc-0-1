---
sourceFile: "EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:26.084Z"
---

# EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model - arXiv

e764ba73-8b86-4fa9-bd33-e1e143b06e6b

EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model - arXiv

01414324-2236-4131-aef8-e71994a0d852

https://www.arxiv.org/pdf/2506.09061

EdgeProfiler: A Fast Profiling Framework for

## Lightweight LLMs on Edge Using Analytical Model

Alyssa Pinnock1∗, Shakya Jayakody1∗, Kawsher A Roxy2, Md Rubel Ahmed3

1University of Central Florida, 2Intel Corporation, 3Louisiana Tech University

Email: {al310186, shakya}@ucf.edu1, kawsher.roxy@intel.com2, mahmed@latech.edu3

Abstract—This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assess-ing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggres-sive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60–70%, while maintaining accuracy within 2–5% of full-precision baselines. Inference speeds are observed to improve by 2–3× compared to FP16 baselines across various edge devices. Power model-ing estimates a 35–50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility.

Index Terms—Large Language Models, LLM, TinyML, Edge Computing, Quantization, Low-Power Devices, Microcontrollers

I. INTRODUCTION

Advances in Artificial Intelligence (AI) and deep learning

have fueled remarkable progress in natural language pro-

cessing (NLP), enabling applications such as summarization,

text generation, question answering, and more. At the fore-

front of these developments are LLMs, which demonstrate

an unprecedented ability to interpret and generate human-

like text. LLMs have contributed to breakthroughs in mobile

applications, healthcare, and situational analysis, among other

domains. However, these models typically require substantial

computational resources, memory, and power, which confine

their deployment to cloud environments equipped with high-

performance servers and GPUs.

Despite these challenges, there is growing interest in imple-

menting LLMs on edge devices, including microcontrollers

and low-power platforms. Edge deployment offers critical

advantages in environments where cloud connectivity is un-

reliable or unavailable. For instance, LLMs can improve situ-

ational awareness during emergencies and disaster response,

where cloud infrastructure may be compromised [1]. In these

∗These authors contributed equally to this work.

scenarios, devices running LLMs can provide real-time crisis

management and communication support. Additionally, on-

device processing enhances data security by minimizing expo-

sure to vulnerabilities associated with transmitting data over

the internet.

Reducing dependency on cloud connectivity is also a sig-

nificant advantage. Moving LLM inference to the edge allows

applications to function independently of network conditions,

ensuring availability even in areas with limited or no internet

access. For example, a local LLM could manage user notes

or provide contextual assistance directly on a smartphone,

regardless of network strength. Moreover, local inference sig-

nificantly reduces latency, which is crucial for real-time appli-

cations. Unlike cloud-based models whose response times are

affected by network stability and speed, on-device inference

ensures prompt interaction, enhancing the user experience.

However, deploying LLMs on edge devices remains challeng-

ing due to their complex architectures and high memory and

power requirements [2]. Models such as GPT-3 and LLaMA,

with billions of parameters, are not feasible for resource-

constrained environments [3], [4]. This has led to an active

area of research focused on quantization techniques, model

pruning, and efficient inference strategies that adapt LLMs to

the limitations of edge hardware [5]. These efforts aim to strike

a balance between computational feasibility, energy efficiency,

and model accuracy.

In this paper, we introduce EdgeProfiler, a fast profiling

framework designed to systematically evaluate the perfor-

mance of lightweight LLMs in resource-constrained environ-

ments. It shows that aggressive quantization makes lightweight

LLMs practical for edge deployment by reducing memory,

computation, and energy costs with only minor accuracy trade-

offs. The main contributions of this paper are as follows:

We introduce EdgeProfiler, a fast profiling framework for

lightweight LLMs on edge devices.

Systematic evaluation of quantization and low-bit imple-

mentations to address memory and power constraints.

Analysis of strategies to enable LLM inference on mi-

crocontrollers and low-power edge platforms.

Comprehensive review of recent research and experimen-

tal results on deploying LLMs in hardware-constrained

environments.

II. BACKGROUND

A. Quantization techniques for efficient LLM inference

## Quantization refers to the process of reducing the numerical

precision of model parameters and activations from high-

precision FP16 / FP32 format to lower-precision representa-

tions of 8-bit integer or 4-bit integer [6].

Symmetric vs. Asymmetric. Quantization technique can be

broadly classified into symmetric and asymmetric schemes,

depending on how they map floating-point values to lower-

precision integer representations.

Symmetric Quantization: In symmetric quantization, zero in

floating-point space is exactly representable by zero in integer

space. The mapping function can be defined as:

xint = round (xfloat

where xfloat is the original floating-point value, xint is the

quantized integer value, and s is a common positive scaling

factor. Dequantization process scales back as:

xfloat ≈ s× xint (2)

Asymmetric Quantization: Asymmetric quantization adds a

nonzero offset to handle data distributions not centered at zero,

making the quantization function:

xint = round

xfloat − z

and the corresponding dequantization is:

xfloat ≈ s× xint + z (4)

where z is the zero-point, chosen such that zero in the integer

domain corresponds to a nonzero floating-point value.

Asymmetric quantization is generally used for activations,

where the dynamic range shifts during inference. Symmetric

quantization is highly hardware-efficient because it eliminates

the need for offset additions during inference. It is particularly

well-suited for weight matrices, where the value distributions

are typically centered around zero. An advantage of symmetric

quantization is that it requires less memory and simpler

hardware logic [7], [8]. However, a key disadvantage is that

it can result in a higher mean squared error (MSE) compared

to asymmetric quantization.

Per-Tensor vs. Per-Channel Quantization. Quantization can

be applied either globally across an entire tensor or separately

across individual channels. These two strategies, per-tensor

quantization and per-channel quantization, trade off between

simplicity and representational accuracy [8].

Per-Tensor Quantization: In per-tensor quantization, a single

scale factor and, optionally, a single zero-point are used to

quantize the entire tensor. The same quantization parameters

are applied uniformly across all elements. similar to the

equation 3. Per-tensor quantization offers the advantages of

simple implementation and high efficiency, particularly for

hardware accelerators that prefer uniform scaling across all

elements. However, it also has notable disadvantages: it fails to

## Full Precision

Model (FP16)

## Quantization

(FP16 to INT8)

## Quantization

Model (INT8)

-26.33-26.33

-98.56 93.47

-19.38 24.95 94.94

21.31 -10.7524.51

(round(x/s), −128, 127)

-25 32 122

Fig. 1. Quantization process from FP16 to INT8, demonstrating how a weight like 93.47 is scaled and rounded to 120 in INT8. This reduces memory size and computation while introducing minimal error, as the scaling factor preserves relative weight magnitudes.

capture the variance across different channels or feature maps,

and it can lead to larger quantization errors if some channels

have significantly wider or narrower value ranges compared

to others [9].

Per-Channel Quantization: In per-channel quantization,

each channel of each output feature in a convolutional layer

or each row in a matrix has its own scale and zero-point. This

allows finer control over the quantization process.

xint,c = round

xfloat,c − zc sc

where sc and zc are the scale and zero-point specific to

channel c. Per-tensor quantization is preferable when memory

bandwidth and hardware simplicity are prioritized. In con-

trast, per-channel quantization provides better model accuracy,

particularly for deep neural networks where channel-wise

variation is significant. Modern LLM quantization schemes

often apply per-channel quantization to weights and per-tensor

quantization to activations, balancing accuracy and inference

efficiency [10].

Quantization-Aware Training (QAT). Quantization-Aware

Training is a technique where quantization effects are sim-

ulated during the training phase itself, allowing the model to

adapt to low-precision representations. During QAT, both the

forward and backward passes emulate quantized operations,

typically using fake quantization functions that maintain gra-

dient flow [7]. This enables the model to learn parameters that

are more robust to quantization noise, resulting in significantly

better accuracy compared to naive post-training quantization

[11]. During QAT, the model optimizes following objective:

E(x,y)∼D [L (Q(f(x; θ)), y)] (6)

where f(x; θ) denotes the model with parameters θ, Q(·) denotes the quantization function, L(·, ·) is the loss function,

and D is the training data distribution.

QAT often uses lower bit-widths 8-bit or 4-bit during sim-

ulation while keeping high-precision master weights for gra-

dient updates. By introducing quantization noise during train-

ing, QAT yields models that maintain higher accuracy after

deployment, especially important for aggressive compression

regimes in edge and embedded systems [3]. Despite promising

benefits, LLM quantization presents unique challenges, such

## Parameter Count

## Energy Modeling

I/O Transfer Analysis

## Latency Estimate

FLOPs/Token Profile

Memory Req.

Best Quant.

Power Est.

Fig. 2. High-level overview of the EdgeProfiler. The framework integrates model, hardware, and precision parameters to estimate latency, memory footprint, and energy consumption, enabling exploration of performance trade-offs on edge platforms.

as attention layers and softmax output are particularly sensitive

to quantization noise. Even if inputs are quantized to 4-

bit/8-bit, intermediate results often require higher precision to

avoid numerical instability. Some input sequences containing

rare or domain-specific tokens can significantly stress low-

precision models [10], [12], [13]. Fig. 1 shows the Symmetric

quantization technique.

III. RELATED WORKS

Lightweight LLMs. MobileLLM [19] demonstrates that sub-

billion parameter models can provide effective NLP func-

tionality on mobile and edge platforms, achieving a balance

between resource constraints and utility. BioMistral-7B [20]

applies quantization and model merging to tailor LLMs

for biomedical tasks, combining compression with domain-

specific optimization for constrained environments. Memory

innovations, exemplified by “LLM in a Flash” [21], reduce

## DRAM usage by dynamically loading only essential model

weights from flash storage during inference. This approach

addresses limitations of static weight loading, improving LLM

feasibility for edge deployment. Another recent advancement

in efficient LLM deployment is the work on automatic INT4

weight-only quantization and optimized CPU inference, as

described by the authors in [22].

LLM Inference on Edge Devices. Recent research has

focused on adapting LLMs for edge devices, addressing

challenges like limited memory, compute capacity, and en-

ergy constraints. Work [14] empirically studies how resource-

constrained computing environments affect the design choices

for personalized LLMs, uncovering key trade-offs and offering

practical guidelines for customization and deployment on

edge devices. However, the empirical study might be limited

to specific models and hardware, affecting generalizability.

Deeploy [15] introduces a compiler framework that translates

pre-trained Foundation Models into compact Small Language

Models (SLMs) for microcontrollers, showcasing a practical

approach for edge NLP tasks. Post-training quantization tech-

niques, such as GPTQ [12], compress model weights to as low

as 2–4 bits, balancing memory efficiency and accuracy. Com-

plementary methods like AWQ [17] and SmoothQuant [18]

further enhance memory and performance trade-offs.

IV. METHODOLOGY

A. Performance Modeling Framework

At the core of our framework is the EdgeProfile, which takes

three inputs: model configuration, hardware configuration, and

precision configuration, as shown in the figure. 2.

Model configuration. Specifies transformer architecture pa-

rameters: number of layers (L), hidden dimension (H), inter-

mediate dimension (I), attention heads, vocabulary size (V ),

and sequence length (S).

Hardware configuration. Encapsulates peak computational

throughput (FLOPs/sec), DRAM bandwidth, storage through-

put, PCIe host-to-device bandwidth, interconnect/network

bandwidth, utilization factors (Ucompute, Umemory, Ustorage, UH2D,

Unet), and energy cost per flop or per byte accessed.

Precision configuration. Defines the data-type size B in bytes

(4B for FP32, 2B for FP16, 1B for INT8).

EdgeProfiler computes:

a) Parameter count: The total number of model parame-

ters is fundamental for determining storage, initialization time,

and memory footprint. EdgeProfiler calculates it by summing

contributions from all projection and embedding matrices

across the transformer architecture:

P = L (4H2) + L (2HI) + 2V H (7)

Equation 7 computes the total number of model parameters

by summing contributions from self-attention, feed-forward,

and embedding layers. The first term, L(4H2), counts four

linear projection matrices (query, key, value, output) per layer,

each of size H × H . The second term, L(2HI), counts the

two linear layers in each feed-forward network with input

dimension H and expansion dimension I . The final term,

2VH , accounts for input and output embedding matrices that

project vocabulary tokens to hidden representations and back.

## This formulation allows EdgeProfiler to accurately capture

total model size, critical for estimating storage requirements,

weight loading times, and parameter transfer bandwidth needs

during initialization and deployment.

b) FLOPs per token: To estimate the compute workload

per token, EdgeProfiler calculates the total floating-point op-

erations required for processing:

FLOPs/token = L (

6H2 + 4HS + 4HI + 4IH + 9H )

Equation 8 calculates the total floating-point operations re-

quired to process a single token. The 6H2 term aggregates four

projections for attention and two residual projections, while

4HS models token-wise dot-product attention computation

over sequence length S. The terms 4HI +4IH capture feed-

forward block matmuls, and 9H accounts for LayerNorm, bias

additions, and other elementwise operations. This equation

enables precise estimation of compute workload per token,

TABLE I COMPARISON OF SPECIFICATIONS FOR RASPBERRY PI 4, RASPBERRY PI 5, AND JETSON ORIN NANO SUPER DEVICES.

## Device CPU RAM Storage OS

Raspberry Pi 4 Quad-core Cortex-A72 @1.5 GHz 2 GB/4 GB/8 GB LPDDR4 microSD card Linux Raspberry Pi 5 Quad-core Cortex-A76 @2.4 GHz 4 GB/8 GB/16 GB LPDDR4X microSD card, PCIe storage Linux Jetson Orin Nano Super 6-core Cortex-A78AE @1.7 GHz 8 GB 128-bit LPDDR5 microSD card, PCIe storage Linux

which is essential for understanding whether inference is

compute-bound or memory-bound on a target edge device.

c) Memory footprint: Estimating total memory usage is

essential for ensuring inference fits within device capacity:

M = P ·B + S ·H ·B + 2L · S ·H ·B (9)

Equation 9 estimates total memory usage, combining model

weights (P ·B), hidden-state activations (S ·H ·B), and cached

key/value tensors (2L·S·H ·B). Here, B is bytes per parameter.

The key/value cache term reflects storage of projected keys

and values for each token in each layer, enabling efficient

autoregressive decoding. This memory estimation is vital for

determining whether an edge GPU or accelerator has sufficient

on-chip SRAM or VRAM to store the model without offload-

ing to external DRAM, which incurs additional latency and

energy. Accurate memory modeling guides system designers

in choosing appropriate batch sizes, quantization schemes, and

layer fusion strategies to fit within device constraints.

B. Latency Breakdown

a) Compute latency: EdgeProfiler estimates compute-

bound inference time as:

Tcomp = FLOPs/token

peakflops × Ucompute (10)

Equation 10 estimates inference time under compute-bound

conditions by dividing total FLOPs per token by the effective

device compute throughput. peakflops is the advertised peak

performance of the accelerator, and Ucompute is a utilization

factor reflecting kernel launch overheads, pipeline stalls, and

non-ideal compute scheduling. This latency model helps deter-

mine if performance is limited by computational capacity or

by data transfer bottlenecks. By adjusting Ucompute, users can

model performance improvements from kernel fusion, operator

scheduling, or optimized hardware utilization techniques such

as Tensor Cores or systolic arrays in edge accelerators.

b) Memory latency: Memory latency computed as:

memoryBW × Umemory (11)

Equation 11 models the time to read or write total data vol-

ume M across the device’s memory subsystem. memoryBW

is the theoretical bandwidth, and Umemory is utilization effi-

ciency capturing factors such as bank conflicts, non-coalesced

accesses, and DRAM refresh penalties. This estimation is

essential for memory-bound workloads, where data movement

rather than arithmetic dominates runtime. It guides decisions

on operator reordering, memory tiling, quantization to reduce

data size, and architectural choices such as increasing on-chip

SRAM to reduce external DRAM usage.

c) I/O, Host-to-Device, and Network Overheads: Esti-

mates of data transfer overheads in critical edge deployments:

TI/O = P B

STORAGEBW × Ustorage (12)

Th2d = P B

H2DBW × UH2D (13)

Tnet = S H B

NETBW × Unet (14)

Equations 12-14 estimate data transfer overheads. Equation

12 models loading weights from storage (e.g. SSD or flash),

13 models copying weights to device memory over PCIe or

NVLink, and 14 models exchanging key/value cache shards

over a network in distributed settings. Each uses bandwidth

and utilization terms to convert bytes to latency in millisec-

onds. These overheads are critical for edge systems where

PCIe Gen3/Gen4 or embedded interconnect bandwidths can

become bottlenecks.

C. Energy Modeling

## EdgeProfiler models energy consumption by Total energy

per token as:

E = FLOPs/token × eflop + M × ebyte (15)

Equation 15 computes token-level energy as the sum of

compute energy (FLOPs × energy per FLOP) and memory

energy (total bytes moved × energy per byte). eflop and ebyte are hardware-specific constants obtained from power modeling

or direct measurement. This metric is critical for battery-

constrained edge deployment, enabling rapid estimation of

average and peak energy consumption per inference.

V. EVALUATION

## This section evaluates the performance of lightweight LLMs

on representative edge devices. The profiling framework is

used to analyze trade-offs between model size, precision lev-

els, and hardware characteristics. Key metrics such as latency,

memory footprint, arithmetic intensity, and energy consump-

tion are examined. The framework was executed on a standard

workstation equipped with a 10th Gen Intel(R) Core(TM) i7-

10700F CPU, 32 GB DDR4 memory, and without using a

dedicated GPU, running Ubuntu 22.04 LTS. EdgeProfiler is

available on GitHub:EdgeProfiler

Experimental Setup. We instantiate the profiler on three

edge platforms: Raspberry Pi 4, Raspberry Pi 5, and Jetson

Nano Super, using published peak FLOPs and bandwidths

with calibrated utilization factors. The analysis covers multiple

numeric precisions (FP32, FP16, INT8) and representative

(a) Memory-bound latency (b) I/O latency (c) Host-to-device transfer time

(d) Network-bound latency (e) End-to-end latency per token (f ) Estimated energy per token

Fig. 3. Latency and energy profile using EdgeProfiler: (a) Memory-bound latency, (b) Storage I/O latency to load weight from disk, (c) Host-to-device transfer time, (d) Network-bound latency for a single KV-shard exchange across nodes, (e) End-to-end latency per token, (f) Estimated energy per token.

TABLE II COMPARISON OF MODEL SIZE, MEMORY USAGE, INFERENCE SPEED, AND ACCURACY LOSS ACROSS DIFFERENT QUANTIZATION PRECISIONS.

## Model Precision Model Size Memory at Runtime Inference Speed Accuracy Loss

FP16 2.2GB ∼3.13GB 1× baseline TinyLlama INT8 1.2GB ∼2.25GB 1.86× minor

INT4 644MB ∼1.78GB 2.45× moderate

FP16 2.0GB ∼2.44GB 1× baseline Gemma3-1B INT8 1.1GB ∼1.60GB 1.26× minor

INT4 815MB ∼1.35GB 1.52× moderate

FP16 2.5GB ∼3.58GB 1× baseline Llama3.2-1B INT8 1.3GB ∼2.53GB 2.7× minor

INT4 776MB ∼2.01GB 3.33× moderate

FP16 3.6GB ∼3.91GB 1× baseline DeepSeek-r1-1.5B INT8 1.9GB ∼2.55GB 2.19× minor

INT4 1.1GB ∼1.84GB 2.97× moderate

1–1.5B parameter LLMs. For each configuration, the profiler

outputs key metrics, including parameter count, FLOPs per

token, peak memory footprint, stage-wise latency (compute,

memory, I/O, H2D, network), end-to-end latency, arithmetic

intensity, and energy per token. This analytical approach

enables rapid comparison of architecture and precision trade-

offs without requiring full hardware deployment. Details of

the hardware configurations are summarized in Table I.

Profiling Results Analysis. On all three devices, storage I/O

accounts for the vast majority of end-to-end latency shown in

Fig. 3(b). Even though compute (and memory) times are on

the order of a few hundred milliseconds or less, I/O delays

range from multiple seconds (Raspberry Pi 4/5) down to just

under a second on the Jetson Nano Super. This indicates

that, without specialized weight-loading optimizations, simply

reducing arithmetic cost (e.g., via quantization) will yield

diminishing returns once compute time becomes negligible

relative to data-movement overhead.

Precision reduction from FP32 to FP16 halves each compo-

nent’s latency, and INT8 cuts it roughly by four. On Raspberry

Pi 4, end-to-end latency drops from ∼15.4 s (FP32) to ∼3.9s

(INT8), driven almost entirely by shorter I/O and transfer

times of smaller weight footprints. However, even at INT8, I/O

remains the bottleneck (3.5s vs. compute 0.13s), underscoring

that the network and storage subsystems must be improved in

tandem with quantization to realize truly low-latency inference

shown in all the Fig. 3. The Jetson Nano Super’s higher

storage bandwidth and PCIe host-to-device throughput, shown

in Fig. 3(c), deliver a dramatic reduction in I/O cost: INT8

inference completes in ∼1.05s end-to-end, nearly four times

faster than on the Raspberry Pi 5. Compute and memory

latencies on the Jetson (∼0.07s and ∼0.88s at FP32) are

comparable to—though still lower than those on the Pis,

but the real advantage comes from overlapping and hiding

I/O behind faster transfers. This highlights that mid-range

## AI accelerators can shift the bottleneck away from storage

if paired with efficient weight-delivery mechanisms.

Ablation Studies on Quantization. Table II summarizes that

reducing precision from FP16 to INT8 and INT4 impacts

model size, peak memory usage, inference throughput. We

observe accuracy degradation across four 1-1.5B (TinyLlama-

1B, Gemma3-1B, Llama3.2-1B, DeepSeek-r1-1.5B) and peak

runtime memory (3.13GB to 2.25GB) but a boost in inference

speed by roughly 1.8× with only a minor accuracy loss.

Further quantization to INT4 yields even more storage saving

(644MB for TinyLlama) and a 2.45× speedup, at the cost of

a moderate drop in task performance. Across all architectures,

INT8 delivers near-2× speed gain and ∼50% reduction in

memory footprint with negligible impact on accuracy. The

large models Llama3.2-1B and DeepSeek-r1-1.5B benefit even

more from low-precision storage, up to 3.3× speedup, incur

more noticeable accuracy degradation, suggesting their suit-

ability only when memory and latency constraints are extreme

and a moderate loss in quality is acceptable.

VI. CONCLUSION

In this paper, we have propose EdgeProfiler, a fast profiling

framework for lightweight LLMs on edge devices using an an-

alytical model. EdgeProfiler framework reveals how hardware

characteristics, model size, and numerical precision jointly

shape inference performance and efficiency. We present our

findings from three widely used edge devices. Across all

platforms, reducing precision from FP32, FP16, and INT8

yields substantial end-to-end latency and energy savings. On

low-power devices, Raspberry Pi, memory, and I/O bound

stage often dominate, whereas on more capable hardware like

Jetson Nano, compute latency becomes negligible, and I/O or

network transfer surface as the new bottleneck. Building upon

this foundation, future work will focus on refining profiling

granularity, integrating real-world deployment feedback, and

expanding support for emerging hardware platforms to accel-

erate practical, scalable, and energy-efficient LLM applications

on the edge.

## REFERENCES

[1] Hakan T Otal, Eric Stern, and M Abdullah Canbaz. Llm-assisted crisis management: Building advanced llm platforms for effective emergency response and public collaboration. In 2024 IEEE Conference on Artificial

Intelligence (CAI), pages 851–859. IEEE, 2024. [2] Chellammal Surianarayanan, John Jeyasekaran Lawrence, Pethuru Raj

Chelliah, Edmond Prakash, and Chaminda Hewage. A survey on optimization techniques for edge artificial intelligence (ai). Sensors, 23(3), 2023.

[3] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. Advances in neural information processing systems, 35:30318–30332, 2022.

[4] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.

[5] Jeremy Stephen Gabriel Yee, Pai Chet Ng, Zhengkui Wang, Ian McLoughlin, Aik Beng Ng, and Simon See. On-device llms for smes: Challenges and opportunities. arXiv preprint arXiv:2410.16070, 2024.

[6] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In Low-power computer vision, pages 291–326. Chapman and Hall/CRC, 2022.

[7] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE conference on

computer vision and pattern recognition, pages 2704–2713, 2018.

[8] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bon-darenko, Mart van Baalen, and Tijmen Blankevoort. A white pa-per on neural network quantization. arxiv 2021. arXiv preprint arXiv:2106.08295, 4.

[9] Ron Banner, Yury Nahshan, and Daniel Soudry. Post training 4-bit quantization of convolutional networks for rapid-deployment. Advances

in Neural Information Processing Systems, 32, 2019.

[10] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 35:27168–27183, 2022.

[11] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. arXiv preprint arXiv:1902.08153, 2019.

[12] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

[13] Ruiyang Qin, Dancheng Liu, Chenhui Xu, Zheyu Yan, Zhaoxuan Tan, Zhenge Jia, Amir Nassereldine, Jiajie Li, Meng Jiang, Ahmed Abbasi, et al. Empirical guidelines for deploying llms onto resource-constrained edge devices. arXiv preprint arXiv:2406.03777, 2024.

[14] Ruiyang Qin, Dancheng Liu, Chenhui Xu, Zheyu Yan, Zhaoxuan Tan, Zhenge Jia, Amir Nassereldine, Jiajie Li, Meng Jiang, Ahmed Abbasi, jinjun xiong, and Yiyu Shi. Empirical guidelines for deploying llms onto resource-constrained edge devices. ACM Trans. Des. Autom. Electron.

Syst., May 2025. Just Accepted.

[15] Moritz Scherer, Luka Macan, Victor JB Jung, Philip Wiese, Luca Bompani, Alessio Burrello, Francesco Conti, and Luca Benini. Deeploy: Enabling energy-efficient deployment of small language models on heterogeneous microcontrollers. IEEE Transactions on Computer-Aided

Design of Integrated Circuits and Systems, 43(11):4009–4020, 2024.

[16] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

[17] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of Machine Learning and

Systems, 6:87–100, 2024.

[18] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference

on Machine Learning, pages 38087–38099. PMLR, 2023.

[19] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. In Forty-first International Conference on Machine Learning, 2024.

[20] Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gour-raud, Mickael Rouvier, and Richard Dufour. Biomistral: A collection of open-source pretrained large language models for medical domains. arXiv preprint arXiv:2402.10373, 2024.

[21] Keivan Alizadeh, Seyed Iman Mirzadeh, Dmitry Belenko, S Khatam-ifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. Llm in a flash: Efficient large language model inference with limited memory. In Proceedings of the 62nd Annual

Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12562–12584, 2024.

[22] Haihao Shen, Hanwen Chang, Bo Dong, Yu Luo, and Hengyu Meng. Efficient llm inference on cpus. arXiv preprint arXiv:2311.00502, 2023.

[23] Nathan Binkert, Bradford Beckmann, Gabriel Black, Steven K Rein-hardt, Ali Saidi, Arkaprava Basu, Joel Hestness, Derek R Hower, Tushar Krishna, Somayeh Sardashti, et al. The gem5 simulator. ACM SIGARCH

computer architecture news, 39(2):1–7, 2011.

