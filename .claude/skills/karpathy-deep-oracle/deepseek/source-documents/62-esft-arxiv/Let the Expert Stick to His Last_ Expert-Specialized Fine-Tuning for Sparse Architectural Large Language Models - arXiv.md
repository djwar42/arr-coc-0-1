---
sourceFile: "Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:41.497Z"
---

# Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models - arXiv

1f187859-f5a3-4e3f-82fa-a33f6e598841

Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models - arXiv

25d08de8-b75e-4f3f-8fa5-2fb6718393f9

https://arxiv.org/html/2407.01906v1

Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models

, Deli Chen

, Damai Dai

, Runxin Xu

, Zhuoshu Li

## DeepSeek AI

Northwestern University  {zw, victorchen}@deepseek.com  Work done during internship at DeepSeek.

Parameter-efficient fine-tuning ( PEFT ) is crucial for customizing Large Language Models ( LLMs ) with constrained resources. Although there have been various PEFT methods for dense-architecture LLMs, PEFT for sparse-architecture LLMs is still underexplored. In this work, we study the PEFT method for LLMs with the Mixture-of-Experts ( MoE ) architecture and the contents of this work are mainly threefold: (1) We investigate the dispersion degree of the activated experts in customized tasks, and found that the routing distribution for a specific task tends to be highly concentrated, while the distribution of activated experts varies significantly across different tasks. (2) We propose  E xpert- S pecialized  F ine- T uning, or ESFT, which tunes the experts most relevant to downstream tasks while freezing the other experts and modules; experimental results demonstrate that our method not only improves the tuning efficiency, but also matches or even surpasses the performance of full-parameter fine-tuning. (3) We further analyze the impact of the MoE architecture on expert-specialized fine-tuning. We find that MoE models with finer-grained experts are more advantageous in selecting the combination of experts that are most relevant to downstream tasks, thereby enhancing both the training efficiency and effectiveness.

Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models

thanks:  Work done during internship at DeepSeek. , Deli Chen

, Damai Dai

, Runxin Xu

, Zhuoshu Li

## DeepSeek AI

Northwestern University   {zw, victorchen}@deepseek.com

1  Introduction

As the parameter scale of large language models ( LLMs ) continues to increase  (Meta,

https://arxiv.org/html/2407.01906v1#bib.bib34

; Mistral,

https://arxiv.org/html/2407.01906v1#bib.bib35

; DeepSeek,

https://arxiv.org/html/2407.01906v1#bib.bib9

https://arxiv.org/html/2407.01906v1#bib.bib38

) , parameter-efficient fine-tuning ( PEFT ) methods  (Han et al.,

https://arxiv.org/html/2407.01906v1#bib.bib15

)  are becoming increasingly important in adapting pre-trained LLMs to downstream customization tasks. However, existing works on PEFT like low-rank adaptation (LoRA) and P-Tuning  (Hu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib21

; Liu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib30

)  have primarily focused on dense-architecture LLMs, with research on sparse-architecture LLMs still being markedly insufficient.

In this work, we focus on exploring PEFT techniques within the Mixture-of-Experts ( MoE ) LLMs  (Mistral,

https://arxiv.org/html/2407.01906v1#bib.bib36

; Databricks,

https://arxiv.org/html/2407.01906v1#bib.bib8

) , as introduced in §

https://arxiv.org/html/2407.01906v1#S3.SS1

. Unlike dense models where all tasks are handled by the same parameters, in the MoE architecture, different tasks are processed by distinct activated experts  (Lepikhin et al.,

https://arxiv.org/html/2407.01906v1#bib.bib24

; Fedus et al.,

https://arxiv.org/html/2407.01906v1#bib.bib11

) . Observations indicate that task specialization in expert systems is the key to the MoE LLM performance  (Dai et al.,

https://arxiv.org/html/2407.01906v1#bib.bib6

) . We further illustrate such specialization in §

https://arxiv.org/html/2407.01906v1#S3.SS2

that experts activated by the same task’s data are concentrated, while those for different tasks vary significantly, suggesting MoE models use specialized expert combinations to handle different tasks. Motivated by this, we propose Expert-Specialized Fine-Tuning ( ESFT ), as illustrated in §

https://arxiv.org/html/2407.01906v1#S3.SS3

. ESFT only tunes the experts with the highest affinity to the task, while freezing the parameters of other experts and modules.

The primary advantages of ESFT lie in two aspects: (1)  Maintaining Expert Specialization : ESFT prevents the decrement of specialization in full-parameter fine-tuning, where experts not adept at the task also update their parameters. Experimental results in §

https://arxiv.org/html/2407.01906v1#S5.SS1

show that ESFT can achieve aligned or even superior performance in downstream tasks compared to full-parameter fine-tuning, and better maintains performance in general tasks. (2)  Saving Computation Resources : ESFT only trains the parameters of the selected experts, which effectively reduces the storage of up to 90% and training time up to 30% compared to full-parameter fine-tuning, as shown in §

https://arxiv.org/html/2407.01906v1#S5.SS2

Besides, we delve deeper into the working mechanism of the ESFT method. We analyze the expert selection process in §

https://arxiv.org/html/2407.01906v1#S6.SS1

and demonstrate how ESFT leverages specialized experts effectively, as selecting 5-15% experts can achieve promising performance in different tasks. We investigate the efficiency of ESFT under different computational constraints in §

https://arxiv.org/html/2407.01906v1#S6.SS2

, showcasing its ability to leverage training resources efficiently compared to other PEFT methods like LoRA. Our studies in §

https://arxiv.org/html/2407.01906v1#S6.SS3

analyze the effects of shared and non-shared parameters in the model on specialized and general performance, pointing out the priority to selectively train non-shared parameters in ESFT. Through ablation studies in §

https://arxiv.org/html/2407.01906v1#S6.SS4

, we highlight the importance of our expert relevance scores and the fine-grained expert segmentation architecture.

2  Related Work

2.1  Parameter-efficient fine-tuning for dense architectural LLMs

The goal of parameter-efficient fine-tuning  (Han et al.,

https://arxiv.org/html/2407.01906v1#bib.bib15

)  is to efficiently customize LLMs for downstream tasks, while existing studies primarily focus on dense architectural LLMs. PEFT methods for dense models can generally be categorized into three approaches: (1)  Adding new parameters : methods of this kind fix the existing model parameters and fine-tune the model on a small number of newly added parameters. Adapter  (Houlsby et al.,

https://arxiv.org/html/2407.01906v1#bib.bib20

; Pfeiffer et al.,

https://arxiv.org/html/2407.01906v1#bib.bib37

; He et al.,

https://arxiv.org/html/2407.01906v1#bib.bib17

; Wang et al.,

https://arxiv.org/html/2407.01906v1#bib.bib43

)  and Soft Prompt  (Li and Liang,

https://arxiv.org/html/2407.01906v1#bib.bib26

; Liu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib30

; Zhang et al.,

https://arxiv.org/html/2407.01906v1#bib.bib51

; Lester et al.,

https://arxiv.org/html/2407.01906v1#bib.bib25

)  are two typical representatives of this category of methods. (2)  Selecting existing parameters : methods of this type fine-tune a limited part of existing parameters, while keeping the majority of the other parameters fixed. Based on whether the trainable parameter space is continuous, these methods can generally be divided into structured training  (Guo et al.,

https://arxiv.org/html/2407.01906v1#bib.bib14

; Gheini et al.,

https://arxiv.org/html/2407.01906v1#bib.bib13

; He et al.,

https://arxiv.org/html/2407.01906v1#bib.bib16

; Vucetic et al.,

https://arxiv.org/html/2407.01906v1#bib.bib42

)  and unstructured training  (Liao et al.,

https://arxiv.org/html/2407.01906v1#bib.bib27

; Ansell et al.,

https://arxiv.org/html/2407.01906v1#bib.bib1

; Sung et al.,

https://arxiv.org/html/2407.01906v1#bib.bib41

; Xu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib46

) . (3)  Applying low-rank adaptation : LoRA  (Hu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib21

; Fomenko et al.,

https://arxiv.org/html/2407.01906v1#bib.bib12

)  is a widely-used PEFT method, which decomposes the origin weight matrices into low-rank components. Subsequent works  (Zhang et al.,

https://arxiv.org/html/2407.01906v1#bib.bib49

; Ding et al.,

https://arxiv.org/html/2407.01906v1#bib.bib10

; Lin et al.,

https://arxiv.org/html/2407.01906v1#bib.bib28

; Liu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib29

)  have introduced numerous improvements to the original LoRA method. However, the study of PEFT in sparse models is still scarce. In this work, we select and tune part of the experts based on their downstream task affinity, which is a unique selection dimension exclusive to the sparse MoE architecture.

2.2  Coarse- and Fine-grained MoE LLMs

Compared to dense LLMs (e.g., LLaMA series,  Meta,

https://arxiv.org/html/2407.01906v1#bib.bib33

https://arxiv.org/html/2407.01906v1#bib.bib32

), MoE LLMs (e.g., Mixtral series,  Mistral,

https://arxiv.org/html/2407.01906v1#bib.bib35

https://arxiv.org/html/2407.01906v1#bib.bib36

) can increase model size while saving training and inference costs. Based on the granularity of experts, existing large MoE models can generally be divided into two categories: coarse- and fine-grained expert LLMs. Most existing MoE LLMs  (Lepikhin et al.,

https://arxiv.org/html/2407.01906v1#bib.bib24

; Fedus et al.,

https://arxiv.org/html/2407.01906v1#bib.bib11

; Roller et al.,

https://arxiv.org/html/2407.01906v1#bib.bib39

; Dai et al.,

https://arxiv.org/html/2407.01906v1#bib.bib7

; Shen et al.,

https://arxiv.org/html/2407.01906v1#bib.bib40

)  have coarse-grained experts where the number of experts is very limited. For example, 2 out of 8 experts are activated for Mixtral MoE series  (Mistral,

https://arxiv.org/html/2407.01906v1#bib.bib35

https://arxiv.org/html/2407.01906v1#bib.bib36

)  and Grok-V1  (XAI,

https://arxiv.org/html/2407.01906v1#bib.bib44

) . As a result, a single expert has to learn complicated patterns from different domain tasks simultaneously. To address this issue, DeepSeek MoE  (Dai et al.,

https://arxiv.org/html/2407.01906v1#bib.bib6

)  has introduced fine-grained expert segmentation. In the DeepSeek-V2  (DeepSeek,

https://arxiv.org/html/2407.01906v1#bib.bib9

) , there are as many as 162 experts, with 8 active experts (8 out of 66 experts are activated for the DeepSeek-V2-Lite). The fine-grained division of experts ensures a high degree of specialization among the experts. Moreover, the specialized expert system enables the selection of experts that are most relevant to the task for efficient tuning.

Figure 1:  Comparison between Expert-Specialized Fine-Tuning (ESFT) and other fine-tuning methods. FFT trains all parameters. LoRA combines pre-trained weights with low-rank matrices to reduce training costs. ESFT only trains a subset of experts in a Mixture-of-Expert (MoE) architecture, optimizing efficiency and task specialization.

3.1  Preliminaries: Mixture-of-Experts for Transformers

Mixture-of-Experts (MoE) for Transformers replace Feed-Forward Networks (FFNs) with MoE layers. Each MoE layer consists of multiple experts structurally identical to a FFN. Tokens are assigned to and processed by a subset of the most relevant experts based on their affinity scores, ensuring computational efficiency in MoE layers. The output hidden state  𝐡 t l superscript subscript 𝐡 𝑡 𝑙 \mathbf{h}_{t}^{l} bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  of the  t 𝑡 t italic_t -th token in the  l 𝑙 l italic_l -th MoE layer is computed as:

𝐡 t l = ∑ i = 1 N ( g i , t ⁢ FFN i n ⁢ ( 𝐮 t l ) ) + 𝐮 t l , superscript subscript 𝐡 𝑡 𝑙 superscript subscript 𝑖 1 𝑁 subscript 𝑔 𝑖 𝑡 subscript superscript FFN 𝑛 𝑖 superscript subscript 𝐮 𝑡 𝑙 superscript subscript 𝐮 𝑡 𝑙 \mathbf{h}_{t}^{l}=\sum_{i=1}^{N}\left(g_{i,t}\text{FFN}^{n}_{i}(\mathbf{u}_{t% }^{l})\right)+\mathbf{u}_{t}^{l}, bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT ( italic_g start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT FFN start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) ) + bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ,   (1)   g i , t = { s i , t , s i , t ∈ TopK ⁢ ( { s j , t | 1 ⩽ j ⩽ N } , K ) , 0 , otherwise , subscript 𝑔 𝑖 𝑡 cases subscript 𝑠 𝑖 𝑡 subscript 𝑠 𝑖 𝑡 TopK conditional-set subscript 𝑠 𝑗 𝑡 1 𝑗 𝑁 𝐾 0 otherwise g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}{\in}\text{TopK}(\{s_{j,t}|1{\leqslant}j{% \leqslant}N\},K),\\ 0,&\text{otherwise},\end{cases} italic_g start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT = { start_ROW start_CELL italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , end_CELL start_CELL italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∈ TopK ( { italic_s start_POSTSUBSCRIPT italic_j , italic_t end_POSTSUBSCRIPT | 1 ⩽ italic_j ⩽ italic_N } , italic_K ) , end_CELL end_ROW start_ROW start_CELL 0 , end_CELL start_CELL otherwise , end_CELL end_ROW   (2)   s i , t = Softmax i ⁢ ( 𝐮 t l ⊤ ⁢ 𝐞 i l ) , subscript 𝑠 𝑖 𝑡 subscript Softmax 𝑖 superscript subscript 𝐮 𝑡 limit-from 𝑙 top superscript subscript 𝐞 𝑖 𝑙 s_{i,t}=\text{Softmax}_{i}\left(\mathbf{u}_{t}^{l\top}\mathbf{e}_{i}^{l}\right), italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT = Softmax start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l ⊤ end_POSTSUPERSCRIPT bold_e start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) ,   (3)

where  N 𝑁 N italic_N  denotes the total number of experts,  FFN i ⁢ ( ⋅ ) subscript FFN 𝑖 ⋅ \text{FFN}_{i}(\cdot) FFN start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( ⋅ )  is the  i 𝑖 i italic_i -th expert FFN,  g i , t subscript 𝑔 𝑖 𝑡 g_{i,t} italic_g start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT  denotes the gate value for the  i 𝑖 i italic_i -th expert,  s i , t subscript 𝑠 𝑖 𝑡 s_{i,t} italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT  denotes the token-to-expert affinity,  TopK ⁢ ( ⋅ , K ) TopK ⋅ 𝐾 \text{TopK}(\cdot,K) TopK ( ⋅ , italic_K )  denotes the set comprising  K 𝐾 K italic_K  highest affinity scores among those calculated for the  t 𝑡 t italic_t -th token and all  N 𝑁 N italic_N  experts, and  𝐞 i l superscript subscript 𝐞 𝑖 𝑙 \mathbf{e}_{i}^{l} bold_e start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  is the centroid of the  i 𝑖 i italic_i -th expert in the  l 𝑙 l italic_l -th layer.

Recently, DeepSeekMoE  (Dai et al.,

https://arxiv.org/html/2407.01906v1#bib.bib6

)  proposes enhancements to the MoE architecture through several techniques, including (1) Fine-grained segmentation, segmenting each expert into multiple smaller ones and keeping the same fraction of experts to process each token, allowing specialization in different knowledge types while maintaining the same computational cost. (2) Shared expert isolation, leveraging shared experts that process all tokens to capture common knowledge, reducing parameter redundancy and enhancing efficiency. The output of an MoE layer in DeepSeekMoE is:

𝐡 t l = ∑ i = 1 K s FFN i s ⁢ ( 𝐮 t l ) + ∑ i = 1 N ( g i , t ⁢ FFN i n ⁢ ( 𝐮 t l ) ) + 𝐮 t l , superscript subscript 𝐡 𝑡 𝑙 superscript subscript 𝑖 1 subscript 𝐾 𝑠 subscript superscript FFN 𝑠 𝑖 superscript subscript 𝐮 𝑡 𝑙 superscript subscript 𝑖 1 𝑁 subscript 𝑔 𝑖 𝑡 subscript superscript FFN 𝑛 𝑖 superscript subscript 𝐮 𝑡 𝑙 superscript subscript 𝐮 𝑡 𝑙 \mathbf{h}_{t}^{l}{=}\sum_{i=1}^{K_{s}}\text{FFN}^{s}_{i}(\mathbf{u}_{t}^{l}){% +}\sum_{i=1}^{N}(g_{i,t}\text{FFN}^{n}_{i}{(}\mathbf{u}_{t}^{l}{)}){+}\mathbf{% u}_{t}^{l}, bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUPERSCRIPT FFN start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) + ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT ( italic_g start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT FFN start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) ) + bold_u start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ,   (4)   g i , t = { s i , t , s i , t ∈ TopK ⁢ ( { s j , t | 1 ⩽ j ⩽ N } , K − K s ) , 0 , otherwise , subscript 𝑔 𝑖 𝑡 cases subscript 𝑠 𝑖 𝑡 subscript 𝑠 𝑖 𝑡 TopK conditional-set subscript 𝑠 𝑗 𝑡 1 𝑗 𝑁 𝐾 subscript 𝐾 𝑠 otherwise 0 otherwise otherwise g_{i,t}=\begin{cases}s_{i,t},~{}s_{i,t}{\in}\text{TopK}(\{s_{j,t}|1{\leqslant}% j{\leqslant}N\},K{-}K_{s}),\\ 0,~{}~{}~{}\text{otherwise},\end{cases} italic_g start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT = { start_ROW start_CELL italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∈ TopK ( { italic_s start_POSTSUBSCRIPT italic_j , italic_t end_POSTSUBSCRIPT | 1 ⩽ italic_j ⩽ italic_N } , italic_K - italic_K start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT ) , end_CELL start_CELL end_CELL end_ROW start_ROW start_CELL 0 , otherwise , end_CELL start_CELL end_CELL end_ROW   (5)

where  K s subscript 𝐾 𝑠 K_{s} italic_K start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT  is the number of shared experts,  FFN i s subscript superscript FFN 𝑠 𝑖 \text{FFN}^{s}_{i} FFN start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  and  FFN i n subscript superscript FFN 𝑛 𝑖 \text{FFN}^{n}_{i} FFN start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  denote the shared and non-shared experts, respectively. Each expert is segmented into  m 𝑚 m italic_m  ones, with  N 𝑁 N italic_N  and  K 𝐾 K italic_K  also multiplied by  m 𝑚 m italic_m  times compared to the coarse-grained architecture.

3.2  Probing Task-Specific Expert  Specialization  in MoE Models

Despite the significant success of MoE LLMs, a clear understanding of the underlying mechanism remains elusive. We conduct probing experiments to understand how non-shared experts are utilized across various tasks. These tasks, as detailed in §

https://arxiv.org/html/2407.01906v1#S4.SS1

, include general domains like math and code, as well as specialized domains like intent recognition, summarization, legal judgment prediction, and translation. These experiments reveal the expert specialization in MoE models in two aspects:

## Expert Routing is Concentrated in the Same Task

We investigate the distribution of normalized gate values, i.e., the sum of all expert-token gate values for each expert, divided by the total across all experts. Figure

https://arxiv.org/html/2407.01906v1#S3.F2

displays this distribution, where the experts are sorted by their normalized values from high to low. The figure shows that a small subset of experts handles the majority of gate values, indicating the model’s and concentrated expert allocation for a specific task.

## Active Experts Vary Significantly across Tasks

We investigate the joint distribution of experts across tasks. Figure

https://arxiv.org/html/2407.01906v1#S3.F3

shows a heatmap of the shared Top-6 experts for two independent data samples per task averaged across layers. This indicates the degree of overlap of experts used within the same task or between different tasks. Off-diagonal values are near 0, and diagonal values are near 6, indicating that the same task uses similar experts, while different tasks use different sets.

Figure 2:  Top Expert distribution for specific tasks. Shaded areas represent variance across layers. The figure shows that few experts handle most gate values, highlighting expert specialization for different tasks.  Figure 3:  The average number of shared Top-6 routed experts across tasks. The values are averaged by layer, indicating that the sets of experts used for the same task are consistent while different tasks are distinct.

3.3  Expert-Specialized Fine-tuning (ESFT)

The highly specialized expert system suggests that different experts can be optimized for specific tasks. Inspired by this, we propose Expert-Specialized Fine-Tuning (ESFT) for MoE LLM customization, which selectively fine-tunes the most relevant experts for downstream tasks to enhance computational efficiency and maintain expert specialization. Figure

https://arxiv.org/html/2407.01906v1#S3.F1

illustrates the differences between our method and existing methods. Below, we introduce our method step by step.

## Data Sampling

We randomly sample a subset  D s = { ( x i , y i ) } i = 1 N s subscript 𝐷 𝑠 superscript subscript subscript 𝑥 𝑖 subscript 𝑦 𝑖 𝑖 1 subscript 𝑁 𝑠 D_{s}=\{(x_{i},y_{i})\}_{i=1}^{N_{s}} italic_D start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT = { ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUPERSCRIPT  from the training data  D = { ( x i , y i ) } i = 1 N 𝐷 superscript subscript subscript 𝑥 𝑖 subscript 𝑦 𝑖 𝑖 1 𝑁 D=\{(x_{i},y_{i})\}_{i=1}^{N} italic_D = { ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT  for expert selection, where  x i subscript 𝑥 𝑖 x_{i} italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  and  y i subscript 𝑦 𝑖 y_{i} italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  denote the input and label, respectively. Empirically, we find that a subset of  32 32 32 32  concatenated samples, each with a fixed length of  L = 4096 𝐿 4096 L=4096 italic_L = 4096 , is robust enough to select the most relevant experts for a task. We detail this claim in Appendix

https://arxiv.org/html/2407.01906v1#A3

## Expert Relevance Score

We propose two methods to calculate the relevance of an expert to a task based on its affinity to the sample tokens, defined as average gate score and token selection ratio, respectively. Both methods assess each expert’s relevance to downstream tasks and can be chosen based on task-specific experimental performance.

Average Gate Score (ESFT-Gate)  This score calculates the average affinity of expert  e i subscript 𝑒 𝑖 e_{i} italic_e start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  to all tokens in the sampled data. It is defined as:

g i l = 1 N s ⁢ ∑ j = 1 N s 1 L j ⁢ ∑ k = 1 L j g i , k l , superscript subscript 𝑔 𝑖 𝑙 1 subscript 𝑁 𝑠 superscript subscript 𝑗 1 subscript 𝑁 𝑠 1 subscript 𝐿 𝑗 superscript subscript 𝑘 1 subscript 𝐿 𝑗 superscript subscript 𝑔 𝑖 𝑘 𝑙 g_{i}^{l}=\frac{1}{N_{s}}\sum_{j=1}^{N_{s}}\frac{1}{L_{j}}\sum_{k=1}^{L_{j}}g_% {i,k}^{l}, italic_g start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = divide start_ARG 1 end_ARG start_ARG italic_N start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUPERSCRIPT divide start_ARG 1 end_ARG start_ARG italic_L start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUPERSCRIPT italic_g start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ,   (6)

where  L j subscript 𝐿 𝑗 L_{j} italic_L start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT  is the length of the input sequence  x j subscript 𝑥 𝑗 x_{j} italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT  in the sampled data  D s subscript 𝐷 𝑠 D_{s} italic_D start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT .

Token Selection Ratio (ESFT-Token)  This score calculates the ratio of tokens for which expert  e i subscript 𝑒 𝑖 e_{i} italic_e start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  is selected. It is defined as:

r i l = 1 N s ⁢ ∑ j = 1 N s 1 L j ⁢ ∑ k = 1 L j 𝟙 ⁢ ( g i , k l > 0 ) K , superscript subscript 𝑟 𝑖 𝑙 1 subscript 𝑁 𝑠 superscript subscript 𝑗 1 subscript 𝑁 𝑠 1 subscript 𝐿 𝑗 superscript subscript 𝑘 1 subscript 𝐿 𝑗 1 superscript subscript 𝑔 𝑖 𝑘 𝑙 0 𝐾 r_{i}^{l}=\frac{1}{N_{s}}\sum_{j=1}^{N_{s}}\frac{1}{L_{j}}\sum_{k=1}^{L_{j}}% \frac{\mathbbm{1}\left(g_{i,k}^{l}>0\right)}{K}, italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = divide start_ARG 1 end_ARG start_ARG italic_N start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUPERSCRIPT divide start_ARG 1 end_ARG start_ARG italic_L start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUPERSCRIPT divide start_ARG blackboard_1 ( italic_g start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT > 0 ) end_ARG start_ARG italic_K end_ARG ,   (7)

where  𝟙 ⁢ ( g i , k l > 0 ) 1 superscript subscript 𝑔 𝑖 𝑘 𝑙 0 \mathbbm{1}\left(g_{i,k}^{l}>0\right) blackboard_1 ( italic_g start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT > 0 )  is an indicator that equals 1 if the gate score  g i , k l superscript subscript 𝑔 𝑖 𝑘 𝑙 g_{i,k}^{l} italic_g start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  is positive, and 0 otherwise.  K 𝐾 K italic_K  is the number of experts selected per token.

Expert Selection and Fine-tuning

For each MoE layer  l 𝑙 l italic_l , we select a subset of experts to be fine-tuned based on their relevance scores. We define a threshold  p ∈ ( 0 , 1 ] 𝑝 0 1 p\in(0,1] italic_p ∈ ( 0 , 1 ]  as a hyperparameter controlling the proportion of total relevance scores to be included in the selected subset. For each layer  l 𝑙 l italic_l , we select a set of top-scored experts  E s l superscript subscript 𝐸 𝑠 𝑙 E_{s}^{l} italic_E start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  whose cumulative relevance score exceeds the threshold  p 𝑝 p italic_p , satisfying:

∑ i ∈ E s l R i l ⩾ p , subscript 𝑖 superscript subscript 𝐸 𝑠 𝑙 superscript subscript 𝑅 𝑖 𝑙 𝑝 \sum_{i\in E_{s}^{l}}R_{i}^{l}\geqslant p, ∑ start_POSTSUBSCRIPT italic_i ∈ italic_E start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_POSTSUBSCRIPT italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ⩾ italic_p ,   (8)

where  R i l superscript subscript 𝑅 𝑖 𝑙 R_{i}^{l} italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  is the relevance score (either  r i l superscript subscript 𝑟 𝑖 𝑙 r_{i}^{l} italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  or  g i l superscript subscript 𝑔 𝑖 𝑙 g_{i}^{l} italic_g start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) of expert  i 𝑖 i italic_i  in layer  l 𝑙 l italic_l . During training and inference, tokens can be assigned to any expert. However, only the selected experts  E s l superscript subscript 𝐸 𝑠 𝑙 E_{s}^{l} italic_E start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT  in each layer can be updated; other experts and modules remain frozen.

4  Experiment Setup

4.1  Main Evaluation

We evaluate our ESFT method on two common LLM customization scenarios: (1) improving the model’s  specific ability in a domain  where the model may already have decent performance; (2) adapting the model to a possibly  narrow but unfamiliar specialized task .

4.1.1  Tasks for Model Enhancement

We choose two domain-specific tasks, i.e., Math and Code, to evaluate how our method can enhance the model’s existing abilities. The two domains are widely concerned in current LLM research and suitable for evaluation, as many pre-trained models can perform decently, while there is significant potential for improvement through further training. We assess our method’s effectiveness through performance gains.

For the Math domain, we use MetaMathQA  Yu et al. (

https://arxiv.org/html/2407.01906v1#bib.bib47

)  for training and use GSM8K  Cobbe et al. (

https://arxiv.org/html/2407.01906v1#bib.bib5

)  and MATH  Hendrycks et al. (

https://arxiv.org/html/2407.01906v1#bib.bib18

)  for evaluation. For the Code domain, We train the model on the Python subset of the enormous evol-codealpaca dataset  Luo et al. (

https://arxiv.org/html/2407.01906v1#bib.bib31

)  to simulate a more concentrated LLM customization scenario, and assess its performance on HumanEval  Chen et al. (

https://arxiv.org/html/2407.01906v1#bib.bib3

)  and MBPP  Austin et al. (

https://arxiv.org/html/2407.01906v1#bib.bib2

Math Ability   Code Ability   Specialized Tasks   MATH   GSM8K   Humaneval   MBPP   Intent   Summary   Law   Translation   Average   Vanilla Model   19.6   55.9   42.1   44.6   16.8   58.6   17.1   14.5   33.6   FFT   23.4   66.4   42.1   42.2   78.8   69.4   47.0   38.4   51.0   LoRA   20.6   58.9   39.6   44.8   67.8   64.7   39.7   23.1   44.9   ESFT-Token (Ours)   22.6   66.0   41.5   42.6   75.6   65.4   45.7   36.2   49.4   ESFT-Gate (Ours)   23.2   64.9   43.3   41.8   78.6   65.8   49.1   35.2   50.2   Table 1:  Main performance comparison across methods and tasks. Best or near-best results are shown in  bold  and second-best results are  underlined . Our method ESFT provides a strong balance of performance across diverse tasks, rivaling FFT and surpassing LoRA, particularly in specialized task domains.

4.1.2  Tasks for Model Adaptation

We select four specialized tasks to evaluate how our method can facilitate language models to adapt to an unfamiliar downstream task, covering a diverse range of abilities that most models can excel at after training but not without training: (1) Text-to-JSON Intent Recognition in the BDCI-21 Smart HCI NLU Challenge

https://www.datafountain.cn/competitions/511

https://www.datafountain.cn/competitions/511

, which requires converting text instructions into JSON format for home appliances. (2) Text Summarization in the BDCI-21 Summarization Challenge

https://www.datafountain.cn/competitions/536

https://www.datafountain.cn/competitions/536

, which summarizes customer service call transcripts. (3) Legal judgment Prediction in the the BDCI-21 Law Event Prediction Challenge

https://www.datafountain.cn/competitions/540

https://www.datafountain.cn/competitions/540

, where the “case description” and “judgment” are repurposed as a legal judgment prediction task. (4) Low-resource Translation in the ChrEn dataset ( Zhang et al.,

https://arxiv.org/html/2407.01906v1#bib.bib50

), translating the minority Cherokee to English. Examples of the tasks are shown in Appendix

https://arxiv.org/html/2407.01906v1#A1

To measure model performance, for the text-to-JSON task, we calculate the exact match between model output and reference answer; for other tasks, we employ GPT-4 to score model output between 0 and 10 given reference answer

4 The exact version we use is  gpt-4-1106-preview . The evaluation instructions are in Appendix

https://arxiv.org/html/2407.01906v1#A7

4.2  General Ability Evaluation

We select a broad range of benchmarks to evaluate the extent to which the models’ general abilities are preserved after training on new tasks. These benchmarks include MMLU  Hendrycks et al. (

https://arxiv.org/html/2407.01906v1#bib.bib19

) , TriviaQA  Joshi et al. (

https://arxiv.org/html/2407.01906v1#bib.bib23

) , HellaSwag  Zellers et al. (

https://arxiv.org/html/2407.01906v1#bib.bib48

) , ARC-Challenge  Clark et al. (

https://arxiv.org/html/2407.01906v1#bib.bib4

) , IFEval  Zhou et al. (

https://arxiv.org/html/2407.01906v1#bib.bib52

) , CEval  Huang et al. (

https://arxiv.org/html/2407.01906v1#bib.bib22

) , and CLUEWSC  Xu et al. (

https://arxiv.org/html/2407.01906v1#bib.bib45

) , covering comprehensive model ability evaluations across various domains including natural language understanding, question answering, instruction following, and common sense reasoning.

CLUEWSC   TriviaQA   IFEval   MMLU   CEval   HellaSwag   ARC   Average   Vanilla Model   81.5   67.7   42.5   57.5   59.9   74.0   53.7   62.4   FFT   80.9  ± plus-or-minus \pm ±  1.1   65.9  ± plus-or-minus \pm ±  0.7   34.2  ± plus-or-minus \pm ±  4.1   55.5  ± plus-or-minus \pm ±  1.0   58.8  ± plus-or-minus \pm ±  0.9   67.9  ± plus-or-minus \pm ±  3.8   48.4  ± plus-or-minus \pm ±  2.4   58.8  ± plus-or-minus \pm ±  1.3   LoRA   74.3  ± plus-or-minus \pm ±  7.7   63.4  ± plus-or-minus \pm ±  5.4   38.7  ± plus-or-minus \pm ±  2.5   55.5  ± plus-or-minus \pm ±  1.2   57.0  ± plus-or-minus \pm ±  1.5   72.8   ± plus-or-minus \pm ±  1.9   51.8  ± plus-or-minus \pm ±  2.3   59.1  ± plus-or-minus \pm ±  2.5   ESFT-Token   80.9  ± plus-or-minus \pm ±  0.9   66.7   ± plus-or-minus \pm ±  1.8   40.7   ± plus-or-minus \pm ±  1.3   57.1   ± plus-or-minus \pm ±  0.5   59.6   ± plus-or-minus \pm ±  0.8   72.3  ± plus-or-minus \pm ±  3.6   52.9   ± plus-or-minus \pm ±  1.5   61.5   ± plus-or-minus \pm ±  1.1   ESFT-Gate   81.4   ± plus-or-minus \pm ±  1.1   66.5   ± plus-or-minus \pm ±  2.3   40.2  ± plus-or-minus \pm ±  1.5   57.0   ± plus-or-minus \pm ±  0.4   59.5   ± plus-or-minus \pm ±  0.8   68.2  ± plus-or-minus \pm ±  9.9   51.5  ± plus-or-minus \pm ±  3.1   60.6  ± plus-or-minus \pm ±  2.3   Table 2:  General ability performance comparison across methods and tasks. The performance for a task is averaged across all training experiments, followed by the standard deviation across tasks. Best or near-best results are shown in  bold . Our method ESFT consistently achieves good performance among all tasks.

4.3  Backbone Model and Training Settings

We use the backbone architecture of DeepSeek-V2-Lite  DeepSeek (

https://arxiv.org/html/2407.01906v1#bib.bib9

)  for all experiments. The model includes a fine-grained set of 66 experts for each transformer layer. This makes it uniquely suitable at the time of this study for our method, which benefits from expert specialization. We train the model on a carefully curated alignment dataset that excludes math and code data and take the resulting checkpoint as our vanilla model for subsequent experiments. This alignment phase can activate model ability across various domains while keeping Math/Code ability as elementary to better verify the performance gains of our method in these two fields.

We adopt two baselines: Full-Parameter Fine-Tuning (FFT) and Low-Rank Adaptation (LoRA,  Hu et al.,

https://arxiv.org/html/2407.01906v1#bib.bib21

). For LoRA, we add low-rank matrices to all parameters for training except token embeddings and the language modeling head. We maintain a 1:1 ratio for task-specific data and alignment data for all methods, which we find is highly effective in preserving general abilities obtained from the alignment phase for FFT and LoRA. However, for our ESFT method, not adopting this data mixing strategy may even better maintain general ability. We detail this in Appendix

https://arxiv.org/html/2407.01906v1#A6

. All experiments are done on the HFAI cluster

https://doc.hfai.high-flyer.cn/index.html

https://doc.hfai.high-flyer.cn/index.html

with 2 nodes of 8x Nvidia A100 PCIe GPUs.

For hyperparameter settings, all methods use a batch size of 32 and a sequence length of 4096 for training. For every task, we set the maximum steps of training to 500, and evaluate the model every 100 steps. The learning rates are set to 3e-5, 1e-4, and 1e-5 for FFT, LoRA, and ESFT, respectively, based on a hyperparameter search in {1e-5, 3e-5, 1e-4, 3e-4}. The LoRA rank is set to 8 and scaling is set to 2, following  Hu et al. (

https://arxiv.org/html/2407.01906v1#bib.bib21

) . The threshold  p 𝑝 p italic_p  is set to 0.1 for ESFT-Gate and 0.2 for ESFT-Token, respectively. §

https://arxiv.org/html/2407.01906v1#S6.SS2

shows how we determine the threshold for ESFT

6 we plan to release the ESFT training and evaluation script, the vanilla backbone model, and all trained checkpoints on different tasks. .

5.1  Benchmark Performance Results

## The results in Table

https://arxiv.org/html/2407.01906v1#S4.T1

and Table

https://arxiv.org/html/2407.01906v1#S4.T2

demonstrate several conclusions. All methods can improve model performance in customization tasks compared to the vanilla model, while they may cause a performance decrease in general tasks. Generally, the performance increase is higher in model adaptation tasks than in model enhancement tasks.

For customization ability evaluation, ESFT surpasses LoRA significantly and is competitive with FFT. As shown in Table

https://arxiv.org/html/2407.01906v1#S4.T1

, ESFT-Token and ESFT-Gate achieve near-best results in model enhancement tasks like Math, and ESFT-Gate achieves the best performance in the Humaneval task. ESFT also excels in model adaptation tasks, with ESFT-Gate achieving near-best performance in 3 tasks out of 4. Notably, ESFT-Gate’s average of 50.2 is competitive compared to FFT’s 51.0, slightly better than ESFT-Token’s 49.4, and significantly surpasses LoRA’s 44.9. This demonstrates that finding task-relevant experts can efficiently adapt the model for efficient customization.

For general ability evaluation, ESFT consistently outperforms FFT and LoRA by showing less performance degradation. As illustrated in Table

https://arxiv.org/html/2407.01906v1#S4.T2

, ESFT-token performs better than ESFT-gate, with average scores of 61.5 and 60.6, respectively. The results demonstrate a wide range of retention in tasks such as TriviaQA and IFEval, surpassing FFT’s 58.8 and LoRA’s 59.1. Both methods retain performance better than LoRA and FFT, highlighting their effectiveness in maintaining general task performance

7  We further investigate Math and Code performance of the models trained on specialized tasks in Appendix

https://arxiv.org/html/2407.01906v1#A8

. FFT and LoRA exhibit even more severe degradation, while ESFT shows a minimal performance drop.  . Analyses in §

https://arxiv.org/html/2407.01906v1#S6.SS3

indicate that such degradation on general tasks for FFT and LoRA may result from training shared parameters.

5.2  Computational Efficiency Results

## The results in Figure

https://arxiv.org/html/2407.01906v1#S6.F6

demonstrates that ESFT exhibits several advantages in terms of training time and storage space requirements:

Figure 4:  Number of experts trained in ESFT across layers and tasks. Earlier computed layers are numbered smaller. Most tasks and layers train 5-15% of experts, demonstrating ESFT’s effectiveness in selecting task-related experts.   Figure 5:  Computational efficiency results. Blue bars show the training time and green lines show storage space. ESFT performs efficiently in terms of training time and storage space.

## Training Time

The average training time for ESFT-Token and ESFT-Gate is 19.8 minutes and 20.9 minutes, respectively. The FFT method takes significantly longer at 28.5 minutes. Although LoRA achieves a shorter training time of 16.5 minutes, our methods are relatively close.

## Storage Space

The average storage space of parameters trained is 2.57 GB for ESFT-Token and 3.20 GB for ESFT-Gate, while FFT demands a substantial 28.6 GB. Although LoRA requires less storage, ESFT performs significantly better than LoRA in downstream task performance.

In summary, ESFT demonstrates excellent performance in training time and storage space, significantly outperforming FFT. Furthermore, as shown in Table

https://arxiv.org/html/2407.01906v1#S6.T3

, ESFT requires much fewer trainable parameters compared to FFT, resulting in lower GPU memory usage. These advantages show that ESFT is efficient and effective for language model customization and adaptation.

6  Analysis

Non-shared Experts   Shared Experts   Non-expert Parameters   Trainable Parameters   Specialized Ability   General Ability   Average   ALL   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   15.7B   51.0   58.8   54.9   Relevant   ✓ ✓ \checkmark ✓   × \times ×   1.85B   49.8   60.7   55.3   Relevant   × \times ×   × \times ×   1.4B   49.4   61.5   55.4   × \times ×   ✓ ✓ \checkmark ✓   × \times ×   450M   47.4   61.2   54.3   × \times ×   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   1.3B   49.0   60.0   54.5   Relevant   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   2.7B   50.8   60.3   55.6   × \times ×   × \times ×   × \times ×   -   33.8   62.4   48.1   Table 3:  Comparisons of different model configs based on whether training shared or non-shared parameters. Results include trainable parameters and performance of specialized and general abilities. The best or near-best results excluding the non-training setting are shown in  bold .   Figure 6:  Comparison of three methods under different training efficiency settings on the Math task. The x-axis shows the average trainable experts per layer for ESFT and rank for LoRA, indicating the ratio of trained parameters. The y-axis represents specialized and general ability. Markers on the lines indicate  p 𝑝 p italic_p  or rank values. ESFT consistently outperforms LoRA in both specialized and general ability.

In this section, we investigate the expert selection process of ESFT in §

https://arxiv.org/html/2407.01906v1#S6.SS1

, and demonstrate the performance of ESFT and LoRA under different computational constraints in §

https://arxiv.org/html/2407.01906v1#S6.SS2

. We analyze the effects of training shared and non-shared parameters in §

https://arxiv.org/html/2407.01906v1#S6.SS3

, and conduct ablation studies in §

https://arxiv.org/html/2407.01906v1#S6.SS4

to verify the importance of our expert relevance scores and model structure of fine-grained experts.

6.1  ESFT Leverages Specialized Experts Effectively

We analyze the number of experts ESFT trains across tasks and layers to understand its expert selection process. Results are shown in Figure

https://arxiv.org/html/2407.01906v1#S5.F4

From the results, we have several observations: (1) The average number of experts used per task across layers ranges from 2 to 15 out of 66, indicating ESFT can have 75%-95% fewer trainable parameters than FFT. (2) ESFT-Token generally employs fewer experts while better maintaining general performance, comparable to ESFT-Gate in tasks like Math, Intent, and Law. (3) The number of experts varies by task, with more specialized tasks like Math and Translation using fewer experts; our method’s performances for these tasks exceed LoRA to the greatest extent, indicating that our method is especially suitable for more specialized tasks. (4) For most tasks, few experts are chosen in the middle layers, indicating that expert distribution is more concentrated in these layers.

6.2  ESFT Leverages Training Resources Efficiently

Math Ability   Code Ability   Specialized Tasks   MATH   GSM8K   Humaneval   MBPP   Intent   Summary   Law   Translation   Average   ESFT-Token   22.6   66.0   41.5   42.6   75.6   65.4   45.7   36.2   49.4   Δ Δ \Delta roman_Δ   of rand   -1.0   -3.7   -2.5   0.2   -2.6   -1.7   1.3   -13.5   -2.8   ESFT-Gate   23.2   64.9   43.3   41.8   78.6   65.8   49.1   35.2   50.2   Δ Δ \Delta roman_Δ   of rand   -1.7   -3.2   -4.3   1.6   -5.0   0.3   -2.9   -20.4   -4.4   Table 4:  Performance comparison between original experts and random experts. Replacing high-affinity experts with random ones significantly harms model performance across different tasks.

Both ESFT and LoRA have a training efficiency hyperparameter ( p 𝑝 p italic_p  for ESFT and rank for LoRA). Increasing its value would raise computational resource usage and potentially improve performance. To understand how ESFT and LoRA perform under different efficiency settings, we evaluate benchmark performance on the Math task. We set rank  ⩽ \leqslant ⩽  512 for LoRA as a higher value will result in more trainable parameters than FFT. Figure

https://arxiv.org/html/2407.01906v1#S6.F6

illustrates both specialized and general ability under different training efficiency settings.

From the results, we can conclude: (1) All three methods show a trade-off between training efficiency and performance. Increasing trained parameters ( p 𝑝 p italic_p  for ESFT and rank for LoRA) before a certain point can improve performance. (2) Both ESFT-Token and ESFT-Gate outperform LoRA at any point, demonstrating higher specialized ability and more stable general ability. (3) ESFT-Token peaks in both specialized and general ability at  p 𝑝 p italic_p =0.5, while ESFT-Gate peaks at  p 𝑝 p italic_p =0.3 for specialized and  p 𝑝 p italic_p =0.1 for general ability. (4) ESFT-Token and ESFT-Gate performance saturates at  p 𝑝 p italic_p =0.2 and  p 𝑝 p italic_p =0.1, respectively, indicating that most expert choices may be less relevant to task performance. We delve deeper into this in Appendix

https://arxiv.org/html/2407.01906v1#A5

6.3  Selectively Training Non-Shared Parameters is the Key to ESFT

In our proposed ESFT method, we only fine-tune a subset of non-shared experts. This section provides detailed discussions of several variants of our method that may also train  shared  parameters. The variables are based on:

Whether  all  non-shared experts or a  task-relevant  subset of them (we use the Token Selection Ratio and set  p 𝑝 p italic_p =0.2) are trained.

Whether shared experts are trained.

Whether other parameters, including gates, attention layers, and embeddings, are trained.

## The results are shown in Table

https://arxiv.org/html/2407.01906v1#S6.T3

. We report average trainable parameters across all tasks, performance of specialized and general abilities, and their average. Detailed numbers for all benchmarks are shown in Appendix

https://arxiv.org/html/2407.01906v1#A4

. From the results, we can draw several conclusions:

Specialized performance increases as trainable parameters increase.  The rank of trainable parameters from 450M to 15.7B highly aligns with the rank of specialized ability from 47.4 to 51.0. This suggests that increasing trainable parameters is effective in enhancing specialized performance.

General performance decreases as trainable  shared  parameters increase.  Whether relevant non-shared experts are trained or not, general performance decreases from 61.5 to 60.3, or from 62.4 to 60.0, respectively, as we train shared experts and/or non-expert parameters. As the complete set of non-shared experts is trained, general performance decreases further from 60.3 to 58.8. This suggests that training shared parameters is more likely to cause overfitting on downstream tasks and forgetting on general tasks compared to training non-shared parameters.

It is highly prioritized to train task-relevant non-shared experts.  Training relevant experts achieves at least 55.3, while other settings achieve at most 54.9, even with higher demands of up to 15.7B parameters. Therefore, fine-tuning these experts is highly prioritized for model customization.

We propose two major training strategies based on these conclusions:

Prioritize specialized ability:  Train all shared parameters and task-relevant non-shared experts to maximize the enhancement of specialized performance.

Balance specialized and general ability, and computational efficiency:  Train only task-relevant non-shared experts to minimize parameter costs while maximizing the maintenance of general ability.

Figure 7:  Experiment results for grouped experts. As the experts become more coarse-grained, ESFT degrades more severely than FFT.

6.4  Analysis of Key Modules in ESFT

In this section, we analyze and demonstrate that the effectiveness of our method lies in two modules: (1) our proposed expert relevance score functions and (2) the fine-grained expert segmentation of the MoE model architecture.

## Expert Relevance Score Function

In this work, we propose Average Gate Score and Token Selection Ratio as expert relevance score functions to filter relevant experts for different tasks. To demonstrate their effectiveness, we replace the experts obtained from these functions with random experts while keeping the number of activated experts per layer the same. Results in Table

https://arxiv.org/html/2407.01906v1#S6.T4

show that replacing relevant experts with random ones significantly decreases task performance, demonstrating the effectiveness of our proposed relevance scores.

Fine-Grained Expert Segmentation of the MoE Model

We use the fine-grained segmented DeepSeek-V2 MoE model as our backbone. To prove the effectiveness of such fine-grained expert segmentation, we use greedy search (as detailed in Appendix

https://arxiv.org/html/2407.01906v1#A2

) to group experts, simulating coarse-grained expert segmentation. Experts in the same group share the same gate for each token, initialized by the average of the original gates’ vector. We conduct experiments in the Math domain as an example. Results in Figure

https://arxiv.org/html/2407.01906v1#S6.F7

show that as the group size increases, the performance of our method decreases more severely than FFT, while the training cost (i.e., the average number of experts used) becomes larger. These findings show that our method, as well as even more effective LLM customization, highly relies on a fine-grained segmented LLM architecture that has more specialized experts.

7  Conclusion

In this work, we study parameter-efficient fine-tuning methods for sparse large language models with the Mixture of Experts (MoE) architecture. We first observe that tasks from different domains are handled by distinct combinations of experts. We then propose selecting the most relevant experts for downstream tasks using two metrics: average gate score and token selection ratio. Experimental results show that our method significantly reduces training costs while matching or surpassing full parameter fine-tuning results. Further analysis confirms that our method enhances the specialization of the expert system within the MoE architecture.

## Limitations

Firstly, due to the limitation of the availability of other fine-grained MoE models, our method was only tested on the DeepSeek-V2-Lite MoE model. The conclusions drawn from this model require further validation when applied to other contexts. Besides, due to the lack of parameter-wise and structurally aligned MoE models with different expert granularities, we used a simulation approach by binding several groups of experts to compare coarse-grained and fine-grained MoE methods.

Ansell et al. (2021)    Alan Ansell, Edoardo Maria Ponti, Anna Korhonen, and Ivan Vulić. 2021.  Composable sparse fine-tuning for cross-lingual transfer.

arXiv preprint arXiv:2110.07560

Austin et al. (2021)    Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Trevor Cai, Anselm Levskaya, Charles Sutton, et al. 2021.  Program synthesis with large language models.

arXiv preprint arXiv:2108.07732

Chen et al. (2021)    Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Maarten Dehghani, Pieter Abbeel, Deepak Pathak, Brandon Sanders, Vishal Katarkar, Zareen Xu, et al. 2021.  Evaluating large language models trained on code.  In

Clark et al. (2018)    Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.

Think you have solved question answering? try arc, the AI2 reasoning challenge

http://arxiv.org/abs/1803.05457

, abs/1803.05457.

Cobbe et al. (2021)    Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021.  Gsm8k: A dataset for grade school math problem solving.  In

Dai et al. (2024)    Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024.  Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.

, abs/2401.06066.

Dai et al. (2022)    Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei. 2022.

Stablemoe: Stable routing strategy for mixture of experts

https://doi.org/10.18653/V1/2022.ACL-LONG.489

Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022

, pages 7085–7095. Association for Computational Linguistics.

Databricks (2024)    Databricks. 2024.

Dbrx: Resources and code examples

https://github.com/databricks/dbrx

DeepSeek (2024)    DeepSeek. 2024.  Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model.

, abs/2405.04434.

Ding et al. (2023)    Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, and Maosong Sun. 2023.  Sparse low-rank adaptation of pre-trained language models.

arXiv preprint arXiv:2311.11696

Fedus et al. (2021)    William Fedus, Barret Zoph, and Noam Shazeer. 2021.

Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity

https://arxiv.org/abs/2101.03961

, abs/2101.03961.

Fomenko et al. (2024)    Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, and Weizhu Chen. 2024.  A note on lora.

arXiv preprint arXiv:2404.05086

Gheini et al. (2021)    Mozhdeh Gheini, Xiang Ren, and Jonathan May. 2021.  Cross-attention is all you need: Adapting pretrained transformers for machine translation.

arXiv preprint arXiv:2104.08771

Guo et al. (2020)    Demi Guo, Alexander M Rush, and Yoon Kim. 2020.  Parameter-efficient transfer learning with diff pruning.

arXiv preprint arXiv:2012.07463

Han et al. (2024)    Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. 2024.  Parameter-efficient fine-tuning for large models: A comprehensive survey.

, abs/2403.14608.

He et al. (2023)    Haoyu He, Jianfei Cai, Jing Zhang, Dacheng Tao, and Bohan Zhuang. 2023.  Sensitivity-aware visual parameter-efficient fine-tuning.  In

Proceedings of the IEEE/CVF International Conference on Computer Vision

, pages 11825–11835.

He et al. (2021)    Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. 2021.  Towards a unified view of parameter-efficient transfer learning.

arXiv preprint arXiv:2110.04366

Hendrycks et al. (2021a)    Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a.  Measuring mathematical problem solving with the math dataset.

arXiv preprint arXiv:2103.03874

Hendrycks et al. (2021b)    Dan Hendrycks, Collin Burns, Steven Basart, et al. 2021b.  Measuring massive multitask language understanding.  In

International Conference on Learning Representations (ICLR)

Houlsby et al. (2019)    Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.  Parameter-efficient transfer learning for nlp.  In

## International Conference on Machine Learning

, pages 2790–2799. PMLR.

Hu et al. (2021)    Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021.  Lora: Low-rank adaptation of large language models.

arXiv preprint arXiv:2106.09685

Huang et al. (2023)    Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. 2023.  C-Eval: A multi-level multi-discipline chinese evaluation suite for foundation models.

arXiv preprint arXiv:2305.08322

Joshi et al. (2017)    Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017.

triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension

https://arxiv.org/abs/1705.03551

arXiv e-prints

, arXiv:1705.03551.

Lepikhin et al. (2021)    Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021.

Gshard: Scaling giant models with conditional computation and automatic sharding

https://openreview.net/forum?id=qrwe7XHTmYb

9th International Conference on Learning Representations, ICLR 2021

. OpenReview.net.

Lester et al. (2021)    Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.  The power of scale for parameter-efficient prompt tuning.

arXiv preprint arXiv:2104.08691

Li and Liang (2021)    Xiang Lisa Li and Percy Liang. 2021.  Prefix-tuning: Optimizing continuous prompts for generation.

arXiv preprint arXiv:2101.00190

Liao et al. (2023)    Baohao Liao, Yan Meng, and Christof Monz. 2023.  Parameter-efficient fine-tuning without introducing new latency.

arXiv preprint arXiv:2305.16742

Lin et al. (2024)    Yang Lin, Xinyu Ma, Xu Chu, Yujie Jin, Zhibang Yang, Yasha Wang, and Hong Mei. 2024.  Lora dropout as a sparsity regularizer for overfitting control.

arXiv preprint arXiv:2404.09610

Liu et al. (2023)    Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, and Yefeng Zheng. 2023.  Moelora: An moe-based parameter efficient fine-tuning method for multi-task medical applications.

arXiv preprint arXiv:2310.18339

Liu et al. (2021)    Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2021.  P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks.

arXiv preprint arXiv:2110.07602

Luo et al. (2023)    Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023.  Wizardcoder: Empowering code large language models with evol-instruct.

Meta (2023a)    Meta. 2023a.  Llama 2: Open foundation and fine-tuned chat models.

, abs/2307.09288.

Meta (2023b)    Meta. 2023b.  Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971

Meta (2024)    Meta. 2024.

Llama 3 model card

https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

Mistral (2024a)    Mistral. 2024a.

Cheaper, better, faster, stronger: Continuing to push the frontier of ai and making it accessible to all

https://mistral.ai/news/mixtral-8x22b

Mistral (2024b)    Mistral. 2024b.  Mixtral of experts.

, abs/2401.04088.

Pfeiffer et al. (2020)    Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2020.  Adapterfusion: Non-destructive task composition for transfer learning.

arXiv preprint arXiv:2005.00247

Qwen (2024)    Qwen. 2024.

Introducing qwen1.5

https://qwenlm.github.io/blog/qwen1.5

Roller et al. (2021)    Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. 2021.

## Hash layers for large sparse models

https://arxiv.org/abs/2106.04426

, abs/2106.04426.

Shen et al. (2024)    Yikang Shen, Zhen Guo, Tianle Cai, and Zengyi Qin. 2024.

Jetmoe: Reaching llama2 performance with 0.1m dollars

https://doi.org/10.48550/ARXIV.2404.07413

, abs/2404.07413.

Sung et al. (2021)    Yi-Lin Sung, Varun Nair, and Colin A Raffel. 2021.  Training neural networks with fixed sparse masks.

## Advances in Neural Information Processing Systems

, 34:24193–24205.

Vucetic et al. (2022)    Danilo Vucetic, Mohammadreza Tayaranian, Maryam Ziaeefard, James J Clark, Brett H Meyer, and Warren J Gross. 2022.  Efficient fine-tuning of bert models on the edge.  In

2022 IEEE International Symposium on Circuits and Systems (ISCAS)

, pages 1838–1842. IEEE.

Wang et al. (2022)    Yaqing Wang, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. 2022.  Adamix: Mixture-of-adapter for parameter-efficient tuning of large language models.

arXiv preprint arXiv:2205.12410

XAI (2024)    XAI. 2024.

## Grok open release

https://github.com/xai-org/grok-1

Xu et al. (2020)    Liang Xu, Hai Hu, Xuanwei Zhang, et al. 2020.  Clue: A chinese language understanding evaluation benchmark.

arXiv preprint arXiv:2004.05986

Xu et al. (2021)    Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. 2021.  Raise a child in large language model: Towards effective and generalizable fine-tuning.

arXiv preprint arXiv:2109.05687

Yu et al. (2023)    Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023.  Metamath: Bootstrap your own mathematical questions for large language models.

arXiv preprint arXiv:2309.12284

Zellers et al. (2019)    Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019.

HellaSwag: Can a machine really finish your sentence?

https://doi.org/10.18653/v1/p19-1472

Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers

, pages 4791–4800. Association for Computational Linguistics.

Zhang et al. (2023a)    Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023a.  Adaptive budget allocation for parameter-efficient fine-tuning.

arXiv preprint arXiv:2303.10512

Zhang et al. (2020)    Shiyue Zhang, Benjamin Frey, and Mohit Bansal. 2020.  Chren: Cherokee-english machine translation for endangered language revitalization.  In

Zhang et al. (2023b)    Zhen-Ru Zhang, Chuanqi Tan, Haiyang Xu, Chengyu Wang, Jun Huang, and Songfang Huang. 2023b.  Towards adaptive prefix tuning for parameter-efficient language model fine-tuning.

arXiv preprint arXiv:2305.15212

Zhou et al. (2023)    Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023.

Instruction-following evaluation for large language models

https://arxiv.org/abs/2311.07911

, arXiv:2311.07911.

Table 5:  Examples for different specialized tasks.   Non-shared   Shared   Non-expert   CLUEWSC   TriviaQA   IFEval   MMLU   CEval   HellaSwag   ARC   Average   ALL   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   80.9   ± plus-or-minus \pm ±  2.2   65.9  ± plus-or-minus \pm ±  1.5   34.2  ± plus-or-minus \pm ±  8.1   55.5  ± plus-or-minus \pm ±  1.9   58.8  ± plus-or-minus \pm ±  1.7   67.9  ± plus-or-minus \pm ±  7.4   48.4  ± plus-or-minus \pm ±  4.7   58.8  ± plus-or-minus \pm ±  2.5   Relevant   ✓ ✓ \checkmark ✓   × \times ×   80.9   ± plus-or-minus \pm ±  2.1   66.1  ± plus-or-minus \pm ±  4.4   42.4   ± plus-or-minus \pm ±  3.0   56.8  ± plus-or-minus \pm ±  1.0   58.9  ± plus-or-minus \pm ±  1.6   67.8  ± plus-or-minus \pm ±  20.4   52.1  ± plus-or-minus \pm ±  5.7   60.7  ± plus-or-minus \pm ±  4.4   Relevant   × \times ×   × \times ×   80.9   ± plus-or-minus \pm ±  1.8   66.7   ± plus-or-minus \pm ±  3.5   40.7  ± plus-or-minus \pm ±  2.6   57.1   ± plus-or-minus \pm ±  1.0   59.6   ± plus-or-minus \pm ±  1.5   72.3   ± plus-or-minus \pm ±  7.0   52.9   ± plus-or-minus \pm ±  3.0   61.5   ± plus-or-minus \pm ±  2.3   × \times ×   ✓ ✓ \checkmark ✓   × \times ×   81.1   ± plus-or-minus \pm ±  3.4   66.7   ± plus-or-minus \pm ±  4.2   41.2  ± plus-or-minus \pm ±  1.6   56.9  ± plus-or-minus \pm ±  1.2   58.9  ± plus-or-minus \pm ±  1.6   71.3  ± plus-or-minus \pm ±  14.1   52.6  ± plus-or-minus \pm ±  5.6   61.2  ± plus-or-minus \pm ±  3.3   × \times ×   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   79.5  ± plus-or-minus \pm ±  4.4   65.8  ± plus-or-minus \pm ±  5.0   41.4  ± plus-or-minus \pm ±  3.2   56.2  ± plus-or-minus \pm ±  1.6   58.6  ± plus-or-minus \pm ±  1.7   67.5  ± plus-or-minus \pm ±  20.7   51.2  ± plus-or-minus \pm ±  4.1   60.0  ± plus-or-minus \pm ±  4.4   Relevant   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   80.4  ± plus-or-minus \pm ±  4.1   66.3  ± plus-or-minus \pm ±  4.1   41.1  ± plus-or-minus \pm ±  5.0   56.7  ± plus-or-minus \pm ±  1.2   59.0  ± plus-or-minus \pm ±  1.9   67.5  ± plus-or-minus \pm ±  20.3   51.5  ± plus-or-minus \pm ±  4.6   60.3  ± plus-or-minus \pm ±  4.6   × \times ×   × \times ×   × \times ×   81.5   67.7   42.5   57.5   59.9   74.0   53.7   62.4   Table 6:  Performance of general tasks across methods based on whether training shared or non-shared parameters. The performance for a task is averaged across all training experiments, followed by the standard deviation across tasks. Best or near-best results are shown in  bold .  Non-shared   Shared   Non-expert   Math Ability   Code Ability   Specialized Tasks   MATH   GSM8K   Humaneval   MBPP   Intent   Summary   Law   Translation   Average   ALL   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   23.4   66.4   42.1   42.2   78.8   69.4   47.0   38.4   51.0   Relevant   ✓ ✓ \checkmark ✓   × \times ×   23.8   65.7   40.2   43.8   80.4   67.3   42.4   35.1   49.8   Relevant   × \times ×   × \times ×   22.6   66.0   41.5   42.6   75.6   65.4   45.7   36.2   49.4   × \times ×   ✓ ✓ \checkmark ✓   × \times ×   22.7   64.5   37.2   44.0   73.6   68.3   42.7   26.0   47.4   × \times ×   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   23.4   66.6   41.5   44.4   81.0   66.7   39.0   29.5   49.0   Relevant   ✓ ✓ \checkmark ✓   ✓ ✓ \checkmark ✓   24.8   66.0   42.1   43.2   82.2   69.5   46.4   32.2   50.8   × \times ×   × \times ×   × \times ×   19.6   55.9   42.1   44.6   16.8   58.6   17.1   14.5   33.6   Table 7:  Performance of specialized tasks across methods based on whether training shared or non-shared parameters. Best or near-best results are shown in  bold .   Math Ability   Code Ability   MATH   GSM8K   HumanEval   MBPP   Average   FFT   15.1  ± plus-or-minus \pm ±  0.3   40.3  ± plus-or-minus \pm ±  5.3   30.2  ± plus-or-minus \pm ±  4.4   40.6  ± plus-or-minus \pm ±  3.9   31.5  ± plus-or-minus \pm ±  2.5   LoRA   11.8  ± plus-or-minus \pm ±  0.6   36.1  ± plus-or-minus \pm ±  4.4   27.9  ± plus-or-minus \pm ±  2.3   36.6  ± plus-or-minus \pm ±  2.6   28.1  ± plus-or-minus \pm ±  2.0   ESFT-Token   19.4  ± plus-or-minus \pm ±  0.8   55.2  ± plus-or-minus \pm ±  0.7   39.5  ± plus-or-minus \pm ±  1.0   44.8  ± plus-or-minus \pm ±  0.8   39.7  ± plus-or-minus \pm ±  0.4   ESFT-Gate   19.5  ± plus-or-minus \pm ±  0.3   55.1  ± plus-or-minus \pm ±  1.3   39.3  ± plus-or-minus \pm ±  1.3   45.3  ± plus-or-minus \pm ±  0.6   39.8  ± plus-or-minus \pm ±  0.6   Vanilla   19.6   55.9   42.1   44.6   40.5   Table 8:  Math and Code performance comparison across methods. Results show that our ESFT method retains performance significantly better in these domains compared to FFT and LoRA.   Math Ability   Code Ability   Specialized Tasks   MATH   GSM8K   HumanEval   MBPP   Intent   Service   Law   Translation   Average   FFT   26.1   70.4   51.2   42.6   78.8   72.8   45.6   34.4   52.7   + mix data   -2.7   -4.0   -9.1   -0.4   0.0   -3.4   1.4   4.0   -1.7   LoRA   21.8   57.8   42.1   42.6   78.2   66.4   46.0   21.8   47.1   + mix data   -1.2   1.1   -2.5   2.2   -10.4   -1.7   -6.3   1.3   -2.2   ESFT-Token   25.2   64.8   42.1   43.8   78.0   67.4   47.2   31.9   50.0   + mix data   -2.6   1.2   -0.6   -1.2   -2.4   -2.0   -1.5   4.3   -0.6   ESFT-Gate   24.1   64.9   42.1   44.6   77.2   68.4   43.6   32.8   49.7   + mix data   -0.9   0.0   0.0   -2.8   1.4   -2.6   0.9   2.4   0.5   Table 9:  Downstream task performance comparison across methods and tasks with and without mixing data from the alignment phase. Results show that mixing alignment data leads to a minor performance decrease for most methods.   CLUEWSC   TriviaQA   IFEval   MMLU   CEval   HellaSwag   ARC   Average   Vanilla Model   81.5   67.7   42.5   57.5   59.9   74.0   53.7   62.4   FFT   76.8  ± plus-or-minus \pm ±  1.7   62.4  ± plus-or-minus \pm ±  10   28.4  ± plus-or-minus \pm ±  5.1   55.5  ± plus-or-minus \pm ±  1.1   58.4  ± plus-or-minus \pm ±  0.4   74.6  ± plus-or-minus \pm ±  3.2   53.6  ± plus-or-minus \pm ±  3.1   58.5  ± plus-or-minus \pm ±  2.5    + mix data   4.1   3.5   5.8   0.0   0.4   -6.7   -5.2   0.3   LoRA   60.2  ± plus-or-minus \pm ±  27   61.2  ± plus-or-minus \pm ±  4.0   33.4  ± plus-or-minus \pm ±  6.1   52.3  ± plus-or-minus \pm ±  3.3   55.3  ± plus-or-minus \pm ±  2.3   71.5  ± plus-or-minus \pm ±  2.5   50.7  ± plus-or-minus \pm ±  2.2   55.0  ± plus-or-minus \pm ±  4.6    + mix data   14.1   2.2   5.3   3.2   1.7   1.3   1.1   4.1   ESFT-Token   80.0  ± plus-or-minus \pm ±  2.5   67.5  ± plus-or-minus \pm ±  0.3   41.9  ± plus-or-minus \pm ±  0.8   57.3  ± plus-or-minus \pm ±  0.2   60.2  ± plus-or-minus \pm ±  0.5   74.5  ± plus-or-minus \pm ±  0.7   54.9  ± plus-or-minus \pm ±  0.7   62.3  ± plus-or-minus \pm ±  0.5    + mix data   0.9   -0.8   -1.2   -0.2   -0.6   -2.2   -2.0   -0.8   ESFT-Gate   80.2  ± plus-or-minus \pm ±  1.6   67.6  ± plus-or-minus \pm ±  0.3   40.8  ± plus-or-minus \pm ±  2.4   57.3  ± plus-or-minus \pm ±  0.3   59.9  ± plus-or-minus \pm ±  0.4   74.3  ± plus-or-minus \pm ±  0.9   55.1  ± plus-or-minus \pm ±  0.9   62.2  ± plus-or-minus \pm ±  0.5    + mix data   1.2   -1.1   -0.6   -0.3   -0.4   -6.1   -3.6   -1.6   Table 10:  General task performance comparison across methods and tasks with and without alignment data mixing. Results show that mixing alignment data improves FFT and LoRA in general tasks, but not our ESFT method. It showcases ESFT’s nature in adapting to downstream tasks with minimal performance loss in general tasks.   {CJK}

Task   Evaluation Instruction   Summary   请你进行以下电话总结内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以“内容准确性扣x分，详细程度/完整性扣x分，…，总分是：x分"为开头。 1.  内容准确性 ： - 预测答案是否准确反映了客户问题或投诉的核心要点。 - 是否有任何关键信息被错误陈述或误解。 2.  详细程度/完整性 ： - 预测答案中包含的细节是否充分，能否覆盖标准答案中所有重要点。 - 对于任何遗漏的关键信息，应相应减分。 3.  内容冗余度 ： - 预测答案是否简洁明了，和标准答案风格一致，不存在冗余信息。 - 如果预测答案过长或与标准答案风格不一致，需相应减分。 4.  行动指令正确性 ： - 预测答案对后续处理的建议或请求是否与标准答案相符。 - 如果处理建议发生改变或丢失，需相应减分。 预测答案：{prediction} 参考答案：{ground_truth}   Law   请你进行以下法案判决预测内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以“相关性扣x分，完整性扣x分，…，总分是：x分"为开头。 1.  相关性 ：预测答案与标准答案的相关程度是最重要的评分标准。如果预测的判决情况与标准答案完全一致，即所有事实和结果都被精确复制或以不同但等效的方式表述，则应给予高分。若只有部分一致或存在偏差，则根据一致的程度适当扣分。如果没有预测判决内容，扣10分。 2.  完整性 ：评估预测答案是否涵盖了所有标准答案中提到的关键点，包括但不限于当事人、具体金额、责任判定、费用承担等。如果遗漏重要信息，则应相应扣分。 3.  准确性 ：检查预测答案中提及的细节、数字、日期和法律依据是否与标准答案保持一致。任何错误信息均需扣分，并且严重错误应该导致更多的扣分。 4.  客观性与专业性 ：预测答案应客观反映法案内容并使用恰当的法律术语。主观臆断或非专业表达需酌情扣分。 预测答案：{prediction} 参考答案：{ground_truth}   Translation   You are an expert master in machine translation. Please score the predicted answer against the standard answer out of 10 points based on the following criteria: Content accuracy: Does the predicted answer accurately reflect the key points of the reference answer? Level of detail/completeness: Does the predicted answer cover all important points from the standard answer? Content redundancy: Is the predicted answer concise and consistent with the style of the standard answer? Respond following the format: "Content accuracy x points, level of detail/completeness x points, …, total score: x points". The total score is the average of all the scores. Do not give reasons for your scores. Predicted answer: {prediction} Reference answer: {ground_truth}   Table 11:  Task instructions for model performance evaluation. The placeholder {prediction} and {ground_truth} represent model prediction and reference answer, respectively.

## Appendix A  Examples for Specialized Tasks

https://arxiv.org/html/2407.01906v1#A0.T5

presents task examples as prompts and corresponding reference responses for each specialized task, including intent recognition, text summarization, legal judgment prediction, and low-resource translation.

## Appendix B  Strategy for Grouping Experts

To group experts together and simulate coarse-grained mixture-of-experts transformer models, we calculate expert similarity and group the experts by maximizing in-group similarities using a greedy search algorithm.

We sample data from the alignment dataset, containing 32 samples each with a sequence length of 4096, to calculate the similarity between experts. We initialize a co-occurrence matrix for all expert pairs as a zero matrix. For each pair of experts that occur simultaneously in a token’s Top-6 expert choices, we increment their score by 1 in the matrix. After iterating through the dataset, we calculate the similarity between each pair of experts  i 𝑖 i italic_i  and expert  j 𝑗 j italic_j  using the cosine similarity between the vectors of row  i 𝑖 i italic_i  and row  j 𝑗 j italic_j  in the matrix.

To obtain an expert grouping strategy through greedy search, we calculate the average intra-group similarity (the average pairwise similarity of all experts within the group) for all possible K-expert groups (where K is the group size, either 2 or 4) from the 64 non-shared experts out of the 66 experts in each layer. We then select the K-expert group with the highest score. For the remaining unselected experts, we repeat this process until all experts are selected and grouped.

## Appendix C  Analysis of Expert Affinity Sample Size

Figure 8:  Results of the shared Top-6 routed experts in two independent samples of a task. The x-axis represents the sample size, and the y-axis shows the shared Top-6 routed experts averaged by model layers.

To evaluate the amount of data needed to identify the most relevant experts for a task, we independently sample two sets of data from the training set for each of the six tasks and calculate the shared Top-6 experts between the two sets. The results are shown in Figure

https://arxiv.org/html/2407.01906v1#A3.F8

. As the sample size reaches  2 17 superscript 2 17 2^{17} 2 start_POSTSUPERSCRIPT 17 end_POSTSUPERSCRIPT  (i.e., 32 samples with a sequence length of 4096), all tasks exhibit a high number of shared experts between the two samples. This indicates that the sample size is sufficiently large to select the top-relevant experts for the tasks.

## Appendix D  Detailed Results for Ablations on Training Shared Parameters

We present two tables that summarize the performance of various methods with different configurations for training shared or non-shared parameters. Table

https://arxiv.org/html/2407.01906v1#A0.T6

shows results on general tasks, and Table

https://arxiv.org/html/2407.01906v1#A0.T7

focuses on specialized tasks. The results indicate that training only task-relevant non-shared experts consistently maintains the best general task performance. Additionally, training task-relevant non-shared experts and all shared parameters yields the best specialized task performance, short of full-parameter fine-tuning.

## Appendix E  Qualitative Examples of the Expert Choices

We present qualitative examples of the amount that routed experts are trainable among all tokens for each task in Figure  LABEL:fig:qualitative . Each subfigure demonstrates examples drawn from a task. Deeper tokens indicate more trainable experts across all 26 layers (top-6 experts per layer). The parameter  p 𝑝 p italic_p  is set to 0.2 for the token selection ratio. Results show that our method, even handling only about 20% of expert choices, covers a wide range of key task-relevant words.

UTF8gbsn For example, in the Intent recognition task, the deepest tokens are “意图” (Intent); in the legal judgment task, the deepest tokens include “婚后” (Post-marriage), “要求”(request), “原告” (plaintiff) and “被告” (defendant); in the Math task, the deepest tokens are mainly numerical tokens such as “3”, “5”, “6” and “7”; in the Code task, the deepest tokens are key words like “const”, or important commentary words like “Fetch the list of IDs”.

## Appendix F  The Impact of Mixing Alignment Data for Training

We adopt a 1:1 ratio for downstream task data and alignment data for all methods during training to better maintain general task performance. This manual ratio is kept constant to avoid the significant additional costs associated with fine-tuning the ratio for each task.

In this section, we present performance comparisons across various methods and tasks to reveal the impact of mixing alignment data during training. Table

https://arxiv.org/html/2407.01906v1#A0.T9

presents the performance on downstream specialized tasks, and Table

https://arxiv.org/html/2407.01906v1#A0.T10

shows the performance on general tasks.

The results indicate that FFT and LoRA benefit from the inclusion of alignment data, leading to improved performance in general tasks while only slightly decreasing performance in downstream tasks. Conversely, our ESFT method does not exhibit the same advantage. Specifically, mixing alignment data does not result in performance increases in either general or downstream tasks. The findings suggest that ESFT is inherently capable of adapting to downstream tasks without significant performance degradation in general tasks, even without added alignment data. This highlights the robustness and adaptability of ESFT in diverse task settings.

## Appendix G  Evaluation Instructions for Specialized Tasks

https://arxiv.org/html/2407.01906v1#A0.T11

presents the detailed criteria to evaluate specialized tasks including text summarization, legal judgment prediction, and low-resource translation. Each task includes specific instructions on assessing predicted answers against reference answers, focusing on aspects such as content accuracy, completeness, relevance, and consistency.

## Appendix H  Evaluating Math and Code as General Tasks

We further investigate the Math and Code performance of models trained on adaptation tasks (i.e., Intent, Summary, Law, Translation), as these domains reflect the model’s general ability if not specifically trained on them. We report numbers with the setting of training on only downstream task data. Results in Table

https://arxiv.org/html/2407.01906v1#A0.T8

show that FFT and LoRA would lead to significant performance drops in the Math and Code domain, having average performance drops of 9.0 and 12.4, respectively. Notably, our ESFT method retains performance significantly better compared to FFT and LoRA, with an average performance drop of less than 1.

