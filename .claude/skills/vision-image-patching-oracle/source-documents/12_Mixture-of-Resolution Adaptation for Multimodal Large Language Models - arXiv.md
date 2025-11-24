---
sourceFile: "Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:18.936Z"
---

# Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv

3c7be06d-563d-4ff0-b555-afc999aed0f1

Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv

447b9eca-8870-46db-b617-59f3ad4a2192

https://arxiv.org/pdf/2403.03003

https://lh3.googleusercontent.com/notebooklm/AG60hOp0MGCsi9xrMAJRBFi24lXMtS57pUkTBfb-z3-tYAuvt-LUkk-3FFAORSfA1VSZIfgtnRWNn02X0TrTnLaKrguG5Zu6Br4iMlU8stBQLfACjRcqET8Rvqsw7wwGy0vhgfQ1DUkTHg=w251-h168-v0

530d3282-910b-4197-a62a-b7db7ea38ce4

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Gen Luo 1 2 Yiyi Zhou 1 3 Yuxin Zhang 1 3 Xiawu Zheng 1 3 Xiaoshuai Sun 1 3 Rongrong Ji 1 3

Abstract Despite remarkable progress, existing multimodal large language models (MLLMs) are still infe-rior in granular visual recognition. Contrary to previous works, we study this problem from the perspective of image resolution, and reveal that a combination of low- and high-resolution visual features can effectively mitigate this shortcom-ing. Based on this observation, we propose a novel and efficient method for MLLMs, termed Mixture-of-Resolution Adaptation (MRA). In par-ticular, MRA adopts two visual pathways for images with different resolutions, where high-resolution visual information is embedded into the low-resolution pathway via the novel mixture-of-resolution adapters (MR-Adapters). This design also greatly reduces the input sequence length of MLLMs. To validate MRA, we apply it to a recent MLLM called LLaVA, and term the new model LLaVA-HR. We conduct extensive experiments on 11 vision-language (VL) tasks, which show that LLaVA-HR outperforms existing MLLMs on 8 VL tasks, e.g., +9.4% on TextVQA. More im-portantly, both training and inference of LLaVA-HR remain efficient with MRA, e.g., 20 train-ing hours and 3× inference speed than LLaVA-1.5. Source codes are released at: https:// github.com/luogen1996/LLaVA-HR.

######## 1. Introduction Driven by the remarkable success of large language models (LLMs) (Touvron et al., 2023; Chen et al., 2020), research on multi-modal large language models (MLLMs) also re-ceives an influx of interest in the machine learning com-munity (Liu et al., 2023b; Luo et al., 2023a; Alayrac et al.,

1Key Laboratory of Multimedia Trusted Perception and Ef-ficient Computing, Ministry of Education of China, School of Informatics, Xiamen University, 361005, P.R. China 2Peng Cheng Laboratory, Shenzhen, 518000, China 3Institute of Artificial Intel-ligence, Xiamen University, 361005, P.R. China. Correspondence to: Rongrong Ji <rrji@xmu.edu.cn>.

768 pix 1024 pix

Figure 1. Zero-shot performance and inference speed of LLaVA-HR and existing MLLMs on TextVQA. Existing MLLMs often fall short of fine-grained VL tasks like TextVQA. Increasing image resolution is an effective yet expensive solution. With the proposed MRA, our LLaVA-HR can efficiently adopt high-resolution images to boost performance.

2022; Chen et al., 2022; 2023b). Numerous efforts have been recently devoted to extending LLMs to more modal-ities, achieving breakthroughs on various vision-language tasks (Goyal et al., 2017; Singh et al., 2019; Hudson & Man-ning, 2019). Despite advances, existing MLLMs still fall short of granular visual recognition. For instance, the pow-erful GPT4-V also suffers from hallucinations when identi-fying small and occluded objects (Tong et al., 2024). This shortcoming inevitably limits the practical use of MLLMs.

To compensate for this shortcoming, practitioners often re-sort to scaling up model size and increasing per-training data size (Alayrac et al., 2022; Li et al., 2023b; Bai et al., 2023). For instance, InstructBLIP (Dai et al., 2023) adopts over 129M image-text pairs for vision-language (VL) align-ments, and shows that a larger visual encoder is beneficial for MLLMs. Motivated by this, Qwen-VL (Bai et al., 2023) further increases the parameters of visual encoder to 1.9 bil-lion and uses 1.5 billion pre-training data. Despite progress, this paradigm is prohibitively expensive, which often con-sumes about thousands of GPU hours.

Orthogonal to these works, we study the visual shortcoming of MLLMs from the perspective of input image resolutions. As revealed in previous VL research (Jiang et al., 2020; Tong

https://lh3.googleusercontent.com/notebooklm/AG60hOqrobdAefHp3CaInt1_KlUCd7slVLqI9ibdnT35vVx_zSeRMpeK4jL-bOoi_j-5iLEhV74zsQeaWaaV8AXqngTmQeCNeJlSuE_EDCP5kf5Uqh8Tgxf_B61yH1AhE-ADY_LFNqwKMA=w169-h187-v0

409a0126-1355-49cd-af66-5b3cb1cbd1e5

https://lh3.googleusercontent.com/notebooklm/AG60hOolv-0qhEX1YZKdZ2PUyimlbPs2qFxJf-OkvfMYQZ1UQVatD-fAOb94rVxGnSWzdbICm5gqGoD5KN9lId6uNmJyIi47c6HnZuzk-2zZO_gbttW8M600WMqKU0wpQrOQwqL3222aBg=w126-h138-v0

091472f9-563a-4eb8-8765-591c576e5f50

https://lh3.googleusercontent.com/notebooklm/AG60hOpXRKTEU2YDSS7iWSQ-292am67YkOFyinlOpBJlpeT0qZ9WXxPY4ozXY32lplN8o_0WYqyQq2ZzTHxjbh0tWHuHWEK1vnreLUNMMlBJbOALwie-VoawUNiaaTLryo9KJ76NYYtpWw=w126-h138-v0

3b8b8b41-fee2-4500-99b1-e0bf142805a1

https://lh3.googleusercontent.com/notebooklm/AG60hOqtZdw21KhfAwvwmpVNW-fAmiqOWxuTtiogaRlusBVR7xvU5RWL7iCXwYD7yq0PL0n2t3jnV_xhS-j5Rm7Rrp2Ko5wB-wC6f-ShTu1y3SASaB96bkALQ1Hav9fZSlwNr_uN_UwG=w169-h119-v0

f396e08f-a855-4e4c-8f99-396af6961405

https://lh3.googleusercontent.com/notebooklm/AG60hOojX7pLCQVzXib4DwkKvXgjvE27nuOUtpqtVp-FKOXrAblnseHnet-dYmqFBrI3M5_V4BAAfrPZytH37WYMppbZ-RydN-l5SaGTb-h9lfG3NwlymuSdBlN8iBkKQnKKeN8f8Zgp=w169-h113-v0

97467722-de6e-43be-8573-24cf7c6aff5b

https://lh3.googleusercontent.com/notebooklm/AG60hOp-mEm77gpbEwGPmrFYZoqg1aaiDeXSYgAem5hIETbO5LFGBPHXewUbpDLgiXmg1NcugTeq-7a0FU-CdUcMucyF7aokmAMY8RbmrlSLvHV-SsrS7Q5Wj3cAtt7q-oZdTqtNQMmbCA=w126-h138-v0

ff03bae5-7867-4bcb-ace7-fe6baaabc6b4

https://lh3.googleusercontent.com/notebooklm/AG60hOpvtybGNj4C26fB0SDsO-HCirB--AslQKnBvjd2uJp1FyGxFWWY2mfoulYNE5L8-t4MnUulRXADUC6r4OR8CKrV_W8eDUD3STBSJpKbzHT8HQcl3aLnsrI_hpCYKve5EarvWIrEnQ=w126-h115-v0

9040e8f3-4340-47d4-ac2a-7929acacda56

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

## Image Encoder

(a) Framework of Existing MLLMs

Resolution: 224 ~ 448 pix

## Image Encoder

High-resolution Pathway

Low-resolution Pathway

Resolution: 384 ~ 1,536 pix

(b) Our Mixture-of-Resolution Adaptation

## Intermediate Networks

High-Resolution Adaptation: Expensive High-Resolution Adaptation: Cheap

LLaVA BLIP-2

Figure 2. Comparison between existing MLLMs and LLaVA-HR. Due to high computation complexity, existing MLLMs (Liu et al., 2023a; Li et al., 2023b) often use input images of low-resolution, which are insufficient for granular visual reasoning. With our mixture-of-resolution adaptation, the proposed LLaVA-HR can increase the image resolution up to 1,536 × 1,536 with limited additional costs.

et al., 2024; Luo et al., 2023b), increasing the resolution of input images is a straightforward solution to improve visual recognition, which becomes more important for MLLMs that involve visual chain-of-thought (Rose et al., 2023). As shown in Fig. 1, increasing the resolution of LLaVA-1.5 (Liu et al., 2023a) from 384 × 384 to 672 × 672 can bring obvious performance gains (+4.6%) on TextVQA (Singh et al., 2019). However, the use of high-resolution images will greatly exacerbate the already high computational cost of MLLMs. For instance, 448×448 resolution will increase the computation complexity of LLaVA by about 1.4 times compared with the default 336 × 336. In addition, due to the complex structure of MLLMs, the training will become unstable as the resolution is greatly increased, e.g., a sharp drop at 1, 022× 1, 022 resolution, as shown in Fig. 1. We assume that the length of visual sequences greatly exceeds the pre-trained context length, leading to training instability.

In this paper, we propose a novel and efficient method for the high-resolution image adaptation of MLLMs, namely mixture-of-resolution adaptation (MRA). As shown in Fig. 1, MRA adopts an innovative dual visual pathway de-sign to process the input images of high- and low-resolutions simultaneously. Specifically, one pathway aims to encode global information of low-resolution images, while the other one serves to capture fine-grained semantics from high-resolution images. Meanwhile, these two pathways are closely interacted via the novel mixture-of-resolution adapters (MR-Adapters), which embeds the high-resolution visual information into the low-resolution modeling. In this way, we can use a much fewer number of visual tokens to represent the input images from macro- to micro-views. With the careful design of dual-pathway structure, MRA can easily increase the image resolution up to 1,536 × 1,536 pixels while maintaining high efficiency.

To validate MRA, we apply it to a recent MLLLM called LLaVA (Liu et al., 2023b;a), and term the new model as LLaVA-HR. We conduct extensive experiments on 11 vision-language (VL) tasks, including common VL tasks like VQA2.0 (Goyal et al., 2017) and emerging benchmarks

such as POPE (Li et al., 2023c). Experimental results show that LLaVA-HR outperforms existing MLLMs on 8 of 11 VL tasks, e.g., +9.6% over LLaVA-1.5 on TextVQA. More importantly, the training and inference of LLaVA-HR are cost-effective. The pre-training and instruction tuning of LLaVA-HR (7B, 1,024 × 1,024) only take a total of 20.7 hours on 8 A800 GPUs, which is hundreds of times cheaper than InstructBLIP (Dai et al., 2023) and Qwen-VL (Bai et al., 2023). With the same resolution, its inference speed is 3 times faster than LLaVA-1.5 (Liu et al., 2023a).

In summary, our contributions are three folds:

We reveal the significance of image resolution for MLLMs and propose a novel and efficient adaptation scheme, termed mixture-of-resolution adaption (MRA), which adopts a novel dual visual pathway design to ob-tain the benefits of high-resolution visual information while keeping training and inference efficient.

We propose a novel mixture-of-resolution adapter (MR-Adapter) for MRA, which can embed the high-resolution information into the low-resolution visual pathway to improve visual descriptive power.

Based on MRA, we propose a powerful MLLM, coined LLaVA-HR, which outperforms existing MLLMs on 8 of 11 VL tasks and spends much cheaper training expenditure than most MLLMs.

##### 2. Related Work 2.1. Multimodal Large Language Models

Driven by the great successes of large language mod-els (LLMs) (Gilardi et al., 2023; Touvron et al., 2023; Chen et al., 2020), growing interest has been aroused in building end-to-end multimodal large language models (MLLMs) (Liu et al., 2023b; Zhu et al., 2023; Luo et al., 2023a; Fuyu-8B, 2023; Peng et al., 2023; Liu et al., 2023c). In particular, most existing MLLMs adopt a modular struc-ture (Luo et al., 2023a; Liu et al., 2023b), which utilizes

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

an intermediate network to project the visual features into the word embedding space of the LLM. Then, the LLM is used to accomplish various VL tasks in an autoregressive manner. Based on the modular structure, existing MLLMs can be distinguished by the designs of the intermediate net-work. Popular MLLMs represented by LLaVA (Liu et al., 2023b) often adopt a linear projection layer or an MLP layer to connect the visual encoder and the LLM (Liu et al., 2023b;a; Chen et al., 2023a;b; Peng et al., 2023). The other works employ sampler-based modules to bridge the gap between the visual encoder and the LLM (Bai et al., 2023; Alayrac et al., 2022; Li et al., 2023b; Dai et al., 2023). These sampler-based modules can effectively reduce the number of visual tokens, but often requires a large-scale pre-training to achieve a promising performance (Bai et al., 2023; Li et al., 2023b). Despite the effectiveness, most existing MLLMs still adopt a low visual resolution, e.g., 336 × 336, which greatly limits their performance in fine-grained tasks.

#### 2.2. Visual Representations for MLLMs

The pursuit of better visual representations has been a pop-ular research trend in the VL community (Lu et al., 2019; Jiang et al., 2020; Radford et al., 2021; Ren et al., 2024). Early endeavors mainly explore the object-level features for VL models (Lu et al., 2019; Zhang et al., 2021). Driven by the large-scale image-text pre-training, grid features from CLIP (Radford et al., 2021) have demonstrated the great efficiency and generalization in MLLMs (Liu et al., 2023b; Chen et al., 2022; Alayrac et al., 2022). Based on grid features, existing researchers mainly improve visual repre-sentations by scaling up the visual encoder. For example, PaLI (Chen et al., 2022) increases the parameters of visual encoder to 3 billions and shows the significant performance boost of MLLMs. In contrast to these works, we improve the visual representations for MLLMs from the perspec-tive of image resolution, and propose a novel and efficient solution, namely mixture-of-resolution adaptation.

#### 3. Preliminary We first recap the structure of multimodal large language models (MLLMs), which consists of an image encoder FI(·), an intermediate network FP(·) and an LLM FL(·).

In particular, given an input image I ∈ RH×W×3 and a tex-tual instruction T ∈ RL, the visual tokens Fv ∈ R(h×w)×d

are obtained via the image encoder, and the text tokens ft ∈ Rl×d are represented by the corresponding word em-beddings. Based on the visual and textual tokens, the LLM will decode the target word step by step, formulated as

FL(Rs|FP(Fv), ft, R0:s−1). (1)

Here, pt ∈ Rm denotes the probabilities of the predicted word and m is the size of word vocabulary.

In some MLLMs (Liu et al., 2023b;a), FP(·) is often a stack of simple linear layers, which are used to directly project the visual tokens onto the semantic space of LLMs. Although simple and effective, this strategy inevitably leads to a longer visual sequence as the resolution increases, e.g., 5,329 to-kens for 1,022 × 1,022 resolution in LLaVA-1.5. In practice, processing such a large number of tokens is computation-ally expensive in MLLMs. To further reduce the number of visual tokens, recent advances adopt the sampler-based module for FP(·) , e.g., QFormer (Li et al., 2023b), which aggregates visual features into several tokens that LLM can directly handle. Nevertheless, these methods often require large-scale pre-training to achieve VL alignments (Bai et al., 2023; Li et al., 2023b).

Based on the above analyses, we conclude that the main difficulty of high-resolution image adaptation lies in the rapidly growing visual sequence. This issue motivates us to further explore how to efficiently encode richer visual information with fewer visual tokens.

##### 4. Mixture-of-Resolution Adaptation 4.1. Overview

To address the above issues, we propose a novel and efficient method for MLLMs, termed mixture-of-resolution adapta-tion (MRA), of which structure is depicted in Fig. 3. The core idea of MRA is to embed high-resolution information into the low-resolution one via a dual pathway design. In this case, MRA can keep a smaller number of visual tokens while encoding richer visual information.

Particularly, given the input images of two resolutions Il ∈ RHl×Wl×3 and Ih ∈ RHh×Wh×3, the process of MRA can be formulated as

Fv = FIl (Il,FA(Fvh)) + Fvh,

Fvh = FIh (Ih).

Here, Fvh ∈ Rhh×wh×dh and Fv ∈ Rh×w×d denote the high-resolution features and the final visual features, re-spectively. And FIl

and FIh are the visual encoders for

high-resolution and low-resolution images, respectively. FA denotes the mixture-of-resolution adapter (MR-Adapter). In Eq. 2, MRA adopts dual visual pathways to process high-and low- resolution images simultaneously. Then, a novel MR-Adapter is used to fuse the high-resolution information from the slow pathway to the fast one. Finally, the visual features of two resolutions are combined and processed by the LLM based on Eq. 1.

https://lh3.googleusercontent.com/notebooklm/AG60hOoC0HnDGTKq3hAAyYFmOKinHKbk8qN5tjhLY9d1GNBp60OjfNv5mFsxYJPoODKwaA1bQfKgo_tJQI5QHDO9XA7KuTXtDq4tM4hiB4wOYlZNKm9Nymk5CePDKwlP6IvF51m1sUHJ=w183-h233-v0

1a69034e-629e-49ba-9784-df7d5c550011

https://lh3.googleusercontent.com/notebooklm/AG60hOomOwHz5pW1hxdc-SJAEShNHPg_Ayez0cxFKPGI6Puj0utiSzlNnJc62aQJQaprCUhSy9_HckPVd__D7wZOKcoRhhkj6jfw_uEBdE-gp8zAAH2IotPR8YFpgAy7BZPULwIyrr_EWA=w65-h63-v0

c9bf5553-9974-4c04-8414-2ef0ab7d28fc

https://lh3.googleusercontent.com/notebooklm/AG60hOo_dU4KOlyWv8TnXWWIMSlRpDxUFBu1NuNxkPCiJck2zyGD5y7HuoewMDFIRbDaDdxYxwIXAxGqNdp3PsZvNx9phBVJQfR80ZaB4J_fGvQfky-aKmphamQVbGjDuSTfypHR6rpX9w=w65-h63-v0

df4a4b86-c924-4416-a577-0889e5f83b6e

https://lh3.googleusercontent.com/notebooklm/AG60hOozWEM1-S8axheL2KOa-sOfGmZtGPZIsvmRMGbz_V-ZYz2EaV0hBlLsCQ0COwlclsx065zUis_J6-FXdqwn8vViAfLui1YDpJfJ5nJ8-gRhYH-sRPx-Isc1Q3i96pAOkxVUfwM0rw=w65-h62-v0

3f59a1b2-d3b4-4bcc-ad56-4f4bb52035a4

https://lh3.googleusercontent.com/notebooklm/AG60hOoSfRhm60E9q3c1Ug1SZnlIfapQPXmxAh4M_528j5cjEKBIXTiw3_Zc0YEJzg7Khme9gP3e9LXjE8blWRMvgQwSJOwRA7Hv-E2r3ExfLMBREo2I3H8qsq8CBCG77j_FIZ__VE_V4g=w348-h226-v0

75d132bf-1b29-493e-87b8-16a60aab7ec8

https://lh3.googleusercontent.com/notebooklm/AG60hOpsxMUeU2ev_td4T0DSptZ93-Zlg3rR4Vs3VV8TWidsFaDLgct0IXOxGZgh4S-nuW_AslqiWOsfXNf3MgPPhuYEddO29eXp1BSOHAwb8mK7PDxbhlXmjrKTAJipfaq_vQEMKNtg5w=w161-h189-v0

08d90479-3e9a-4e25-8c21-1fbae2987c34

https://lh3.googleusercontent.com/notebooklm/AG60hOoE4A_UlLEhzFmHMwVSou-w-H-yozVcDoCE3yZOe-3eR-Q_34azxR-cbbQ5qkDmROUxlD0iAEQJcolYDZl97wA7J3zJNNtHrljj2-TZav_JGsPzX6bqByXMynBInmjkhIrUYItYGQ=w157-h155-v0

6f06ebbf-84ee-4101-8507-9c3917daca2b

https://lh3.googleusercontent.com/notebooklm/AG60hOpDyzUk0GKvDf9Om1dvYk7qxaTgzVC84Vmsm5tChhuqw1ynHfdj_583F7GSuMGFo8PJUzYSbPpLPVM4-DwSfDkbrJ4XgD9CXmIGDfqnqElu08kct3ZA29E3tnEYthIy6gn1s9Ic=w143-h129-v0

8462ed5d-b280-47d3-b277-7af3de91e8ca

https://lh3.googleusercontent.com/notebooklm/AG60hOp4QeXg1YnT0VfGXe3rARr5Tm2q3KnCujS-y4MtRcv8kWZ-pY1U8veplvTYWnf07DlOFankmiATrqbTmlAtD7oqtTtyAMJ2GNc1g89eFBnSSu89Ff9_jbS-32deXNV3Fr5X0JPJeg=w176-h145-v0

fac75443-c725-420b-a3e3-72862828285b

https://lh3.googleusercontent.com/notebooklm/AG60hOpp-RdI23p53pnbuEAxZT3zV7TrRFcS2w7Yd97HxXJtuIpNLZHClDXywfRSc6sL_xMiXQSymNUcZBMavSEajtreW65H7WVdX4tdyVLvmfh8kEOlR3FkFUWQFmF3K59NpKXEowsV_A=w175-h145-v0

a600bfd9-20fa-4f79-9f8c-afb2a165f26e

https://lh3.googleusercontent.com/notebooklm/AG60hOrDeHdmIfzeJdzXJ5jPSVY_X1OQrV6gpMfYpfryHSkpPFHnPwYkbta7KIhxYDlrGfizNOSDATXCnbAaIob13ssvi_5YTkksHXBZnOVTWdxVk3an_b0ER45rL8mM7d1Gg-ORk6wNgg=w176-h145-v0

abcbdcfa-8f9e-45ac-add0-06e35e23f07c

https://lh3.googleusercontent.com/notebooklm/AG60hOqwtNcGqmMxFBkfHewF86ZvCQczSVCGwtPaUM5snzRsIBfkEXfHrar1qycYazmKP8OGpvnkADC77QOuOXxuKOG2KvCbsKWNnu7-1M0mL1sEiGk0L_gpJbzlOuJdmT1uDk5WbZ8v=w176-h145-v0

c45a1d5a-2e33-4166-986e-8005042a0f4c

https://lh3.googleusercontent.com/notebooklm/AG60hOrc5NrJFjwBMRKdqJ7XYCipXtdzkkNUAbAo2m5woz0iFncp5dioyKG6qJ1Bcanq8a-WaFFvTssdbRYK7F8PHzsLP-8la3RBo3UUbMRSsmNAtH2F6Umiu1y0yNkWFpyHphOdcwFT=w348-h226-v0

d29daacd-1488-4e30-951b-b6868358e99c

https://lh3.googleusercontent.com/notebooklm/AG60hOpdWpOzDkZNm7ViVxKZloNrSf6hnOIopXyDl6_fC6P3T2znGC-8s7KuC19Hbr03oo1uaPDHfHI7Q5C2q2nWTfsWYw5zbBxMaKH0DCFRgDBCh6ukydlrnzdCGzet8tzFvQlCzb2ltw=w65-h65-v0

abced5ab-3d70-4e06-9f78-c80d6178dd66

https://lh3.googleusercontent.com/notebooklm/AG60hOrLY2X3nOFBm79fQxPCiBDErjXvqKav9A2gMIgWmqRFZF5bZdxwqc8Jo2ziTM_1wkjt5Na-JqvgSIDs4bksimsPeTW-UONr-CpfXPWNQtKnEEKdd64bbW4k9nnsb1SYqoAVA7vL=w65-h65-v0

73e15a9c-7e2a-456b-998f-2fe92d74902c

https://lh3.googleusercontent.com/notebooklm/AG60hOq7aHaf4cSIyjzIh9wtcEJ4TRctcvC74xEmnl5SyONMIQuw4lMoI4A9xU77NtLmyeXIPSbQ-3oLCs0N2GHCZc_x1We5DwviwbNzGkl-iDjpgn6C36mb8XHRJeGg18zUMFu45SifTA=w65-h65-v0

d56ee2c0-b861-4926-8903-f3cb138b9586

https://lh3.googleusercontent.com/notebooklm/AG60hOqNv-ZDd2dF0lm0SbGhs4SxBvM5SfEtnjWUA4oBCyszsh-F4IhzUG-vkMfkfeZap7NZCw5gaUEFfGFgRD_0d7QubEjUMmOQkGjvzgGRnufiCZIfWK2_PzFDVyZAvfbkT633oYYb=w510-h307-v0

67c3c92c-b841-4b5a-a16c-8b807a37ac1e

https://lh3.googleusercontent.com/notebooklm/AG60hOoxGhFe-_7silWbXlEAHsahxTC2nEBcHqDzcdnoUCUcokYXNxvA7DG3D6AcPFhjznlfMl-ejpsKCboMvhN0vBgBOKuNiDz5rx5xbZVsh91aWmTqCSAy78nFQeFxPMhQpN-9NjTzUQ=w72-h72-v0

c1f1fce8-edc1-4204-8670-58286b3759d2

https://lh3.googleusercontent.com/notebooklm/AG60hOpRfpWhGY2pqYZW8vSPLsNw-EYjYMm-2NfYs1WDlVSouUcNGsJQCSYdp6hKm9AozAJqf3edbstNUBzxmzvkKLZEI324KoukdmqkZbOU8JvmgikCqqCiWBrgADBhzznWCHB4t030=w72-h72-v0

a7831d9f-8370-4dae-ae24-4f29ddc2bc3d

https://lh3.googleusercontent.com/notebooklm/AG60hOo9RaXpvAosUkKBGZz_JvMFk7w8fMEJZ4D5zKjHb6Ps0eEIUAzPaddPRtBBAC6oVJ0jFzLo2dCq_lwwIxAFqw8bvgs7DpNldgoo9JAOexJd-yo6-Av-m5-N53geVo4Aw9By_YBQ=w73-h72-v0

b6ba7c15-80f7-412f-af2b-120823305906

https://lh3.googleusercontent.com/notebooklm/AG60hOqAhBOZFLCoeZfnXKvpXtEAy83wjhQpEjeBPk2S-8LgtCD_G8H1aPgKnTDFFu4XO9EndMAMEq7OAsblVU70oXPvNj4RD6sZ5GH7pCzsgMaUnKDAMA7537DKWS9q32N3YzonPdBpwg=w72-h72-v0

a936bcd9-2509-48f4-8157-aa76ae21bfac

https://lh3.googleusercontent.com/notebooklm/AG60hOqM8nmTAKtUsUd6-aScobxh7J9VactLVJO14KzyclZlvhIQ-sGD0eddFeAOpxJE1go5PFqO0HhjRJ1RHZ9frxwRGfoP9BNxuQblu_wxQbw_M-mkkjdtO-DuZwEd1kt4OAsVFZjX=w73-h72-v0

ccbd7bd3-d2f8-425c-ba63-d33232f02240

https://lh3.googleusercontent.com/notebooklm/AG60hOrVSu583j5eVCrPVT-D8Y0uCAMm55WJ8aK-T-kNqJIEZlimP643Qj-z32nizhNqjqwZDJ9stGhZTk38yWz84AZYVTktVZaxlt1P08kXpKQlJXJVaMxS34siPCAab6ERv564b3K9QA=w73-h72-v0

09c7191f-5547-42db-a7d5-cffc1f9326b1

https://lh3.googleusercontent.com/notebooklm/AG60hOrooGQnHp9RoppN92Pppax-wAgezZEEDY-DAzSZ_3r_sw-55wmabMvAGKU5hkEU05Vcxd-hNxCZ5nS2npII3dN0hlRw6JroXTiNAGwUAOh7jkdbttQxdug0JCA62ewhQxCj12LPIQ=w213-h83-v0

b8aa10bf-b686-4d4e-9692-d999dbc629cb

https://lh3.googleusercontent.com/notebooklm/AG60hOrOWdXkCy2uTwwZ0p2cunAkURTu9gDydmblDCDc_bqWmittg9P3FvEJXnhKJ_BHDU-0D_BzPxw-_RxZ4lcjOoaUQ7qNK5tEj4DzHkojvyaRqTZIvyBk2HPXLGvMkWN_ZUXK7paS_g=w213-h83-v0

1592bfd2-f36a-4a85-ac93-b36931352a36

https://lh3.googleusercontent.com/notebooklm/AG60hOqr0B3zP1nTZojIVs-IMAd4RaqIvge-7EXQo2CDGDcCxQiX1uoK1mCZ0ZTIJ6k1Ju84AMGJxv93849CFgHSm4PKrUVqda4Mn5n9Q4TkpgadHWTgrTn7fw65DBwYJpHnLxFhojUsSw=w63-h62-v0

64e4da94-780c-46da-aefa-760df4337db2

https://lh3.googleusercontent.com/notebooklm/AG60hOr2sUIZGHejqUqTPVFPtcWogkPAWLeby1jD61xZbZtteO2gxNNxdOwm_20QdUBhala0moZFZKgdrGJueB1KleoqGm2W8i2bh8aViBQPMagkuQNcqdxMW6zsAzLx7g1IMUdn-bdv8w=w62-h62-v0

1535e852-84e3-48ce-8f4e-063f1a3dea7b

https://lh3.googleusercontent.com/notebooklm/AG60hOqAtxiLojK7ubjPVFUcK2Q0pXLiEc_HOyUe-nt5V1fkpHPpZe6OFpaJ_C-07e6S0nmoubUnoxyKB80guGW8oIv7M02axGUKd_xGENh-dZMRhuVHTJRVpDMhjaucTH3C64_1COohOA=w63-h62-v0

cf13fb76-87c8-4159-951e-1c1f60b4edb0

https://lh3.googleusercontent.com/notebooklm/AG60hOpc3N056Sq4x1JtGpvgtsTP7nfXGo0A3yZdI7o0NSJGn8vy_R5PSfT_a2hC_qx25-oTkD1G5CYZ0aYoPTqGfKQTc6EafnpvOssO4UNVo6kOrR2otiHv6FHb2Ycya3bIQosjckrv=w65-h62-v0

e0d168d3-6ac2-4bcf-9a5b-ac8f0419214d

https://lh3.googleusercontent.com/notebooklm/AG60hOpyxCQjxdsRfdfyXmJ66HbpPs4ItjC3-Nle1XmgXriM2K2-jB0Qx8VgQ23pW_jWdbvsw387Je_sz-PFOWZsw-SPF1F3qm53qfwnkbn6RmoiM75u7dbTQn54n3y_zL4wwvrShzKa=w65-h62-v0

5e348496-a3bd-40b1-889f-a1b7e015e078

https://lh3.googleusercontent.com/notebooklm/AG60hOqiQRvHP9z00D-A2fJnG4QrnEf_gZEYaA-jUNpdba28yf8bvcL3CGf2t4yYq_swSlppFKdjHVAmgqlQChrAASHHsdny5YP6zkT20p7ncojB1XIItRIUZRzNnO2samKfh64kTeZK=w65-h62-v0

f04db2d1-021f-493f-8d56-ec0b9aae8d0b

https://lh3.googleusercontent.com/notebooklm/AG60hOr1kqitvhmnbXzsUN2d172ivEecc7gPfWqnJtTeFQ7mOnE4sXNP-4tbtFAFBUVUS_CBsQtsxCqrgCJkE7ahW82CDtrA0fhjfXnAbgjEuhPwUCY-8OxMlc-TeUy5dx0zIQYSOB-nQQ=w63-h57-v0

2743cc7b-d59e-4e99-8426-0d361fe9f3cd

https://lh3.googleusercontent.com/notebooklm/AG60hOrIs-P6h3d_isUN0ju3M37q_SY9V47c4uKNwqr77Eb54uXHNtm8XYzqy63e1N621cjB6K5JCkU8h7mdO3nP_gfwIJHF4m5A6SU-jOl29Ctc9NEp5zbzHRDHseYkTEapFfCDtH_7EA=w63-h57-v0

a26f19b0-2959-4b18-8353-adc987da0510

https://lh3.googleusercontent.com/notebooklm/AG60hOrCbkvPHTlzwLIn-BvpZNYuHVUE_C1jbUl7p9m55TUC1He70-q2nJOlwnkxh91S8pZn9FIMqVJFiCfF4H_Tup5XLUNtqNsxNz8bhrZQE9G0XkWw6zLuvOWaucndugJ-D-NucIO9=w62-h57-v0

3376570d-42ca-4a54-829f-ff424f94d5c5

https://lh3.googleusercontent.com/notebooklm/AG60hOq2guOK7eZSEQ6QLJvaeBiLucNu3NcCsCGrl2BKMrwMyzyghrMyElHv0mC1rNxkwtyljDSR9_8XR6Il4SLo-B-R1ryEBmz6flr_33QehPnzqhYylT2zHnVn-DBy46veA5j3Nm3vRA=w62-h63-v0

b274be6c-df4d-4a53-81e4-3ae3dd3ec37c

https://lh3.googleusercontent.com/notebooklm/AG60hOqkWv2nXz8hAMJqHYghbHBxjZLy8AhNhvBLiZ4p0hHu7wwbBWsXYFVCno-AtMQDVv9J5TUbYnaR94dp0giEHD6efVPibbE5U7U-3VobcX4xlwI2GiovLPVAob-OaMmc_sHrEi5Kng=w62-h63-v0

76bbb6a9-e684-4907-a03a-5ac01f22cdf0

https://lh3.googleusercontent.com/notebooklm/AG60hOr8YzEV7_tcWyFS4Vy5dp_sr702C4J_bS6244YqJBDtx0SbjCOITHLLL4DlHi8Ba1YWKihNe7rQ94Oy1IOdHHQZ6mAb6o2DwqrVzbvO5IlEOijxoHtVUCxizCqZuVczdZ8M57WEfQ=w63-h63-v0

3d257e84-0a62-4500-b3b9-bc7541109aa1

https://lh3.googleusercontent.com/notebooklm/AG60hOq8aCRDSROok6Wk6V5biFkmtsoPRIS2IUYeFbvm6qmNINsXUObVC1i9iAHJ92bB4hzip-S1nn0bPadNGRjAyCBZtGP0EKnCUOgDm3Lm11nprlvvDW07jhLTLoykNss_gb9SdY_kog=w65-h63-v0

c966c879-c9ec-4e55-a179-a788f2cadc35

https://lh3.googleusercontent.com/notebooklm/AG60hOqMnzDk2Vp9-nW9jTnSD5w1FTEdCxXtDy8HmpT4FfXkWPSk9hQIcVPv2HZyCw2Itv2124jdPnN07guPNkJ0TydV3UMMESw85zY0rviuidhwCZPDi13VOJy0a1DqvRbqN0tcMFGZnw=w65-h63-v0

22b6ab19-03bf-4381-b8d8-5467a0deb05e

https://lh3.googleusercontent.com/notebooklm/AG60hOoXdQSgFNXZC4ykoPxfKTAbdH2CZThKjxZuHsMe7iZ_Vd3kp2HJWC_CSN7yJWGgVEVd0tVZpDY43bAE90WvOTOLvzLW0EEKsN56B856CMlqfGlhI__vYq_lmFAl-Vm0ZR5OTFyGgQ=w65-h63-v0

e2b6e03f-a2a6-4036-a269-9a208bd23ca9

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

High-resolution Image

stage Conv

Low-resolution Image

𝟏𝟎𝟐𝟒 × 𝟏𝟎𝟐𝟒

𝟒𝟒𝟖 × 𝟒𝟒𝟖 MR-

Low-resolution Pathway (Macro View)

Multi-head Attention

Feed-forward Network

Text Instruction:

“describe this image in short.”

Output: A herd of

elephants and deer are

gathered around a

watering hole. The

elephants are of various

sizes, including a baby

elephant. The deer are

also of different sizes,

with some appearing to

High-resolution Pathway (Micro View)

𝟑𝟐 × 𝟑𝟐

𝟑𝟐 × 𝟑𝟐

Figure 3. Illustration of Mixture-of-Resolution Adaptation (MRA) and its deployment on LLaVA-HR. MRA employs dual visual pathways to process high-resolution and low-resolution images, respectively. High-resolution information is embeded into the fast pathway via a novel mixture-of-resolution adapter (MR-Adapter).

𝐹𝑣ℎ 𝐹𝑣𝑙

Figure 4. Illustration of the mixture-of-resolution adapter (MR-Adapter). MR-Adapter can dynamically embed the high-resolution features into the low-resolution pathway.

#### 4.2. Dual Visual Pathways

As shown in Fig. 3, dual visual pathways are the key design of MRA, and their benefits are maximized from two aspects.

Visual functionality. Firstly, the dual visual pathways pro-cess images from macro- and micro-views, which is inspired by the visual system of human being (Merigan & Maunsell, 1993; Robertson & Lamb, 1991). Particularly, Robertson & Lamb (1991) find that the visual system processes local and global semantics via different pathways. Based on this finding, we adopt a similar mechanism to our MRA. Specif-ically, one visual pathway aims to capture fine-grained se-mantics from high-resolution images i.e., processing images from local view. In contrast, the other pathway is designed to encode global information from low-resolution images, achieving a larger receptive field.

Visual alignment. Due to different resolutions, these two pathways often produce visual features of different shapes, impeding their quick alignments (Yu et al., 2019). To over-come this limitation, we adopt different downsampling rates for the low- and high-resolution pathways, respectively. Thus, their output features can keep the same spatial shape.

Based on the above observations, we design the dual visual pathways with a convolutional network (CNN) (Liu et al., 2022) and a vision transformer (ViT) (Dosovitskiy et al.,

2020). Specifically, CNN is equipped with a downsampling stride of 32 to process high-resolution images. ViT encodes low-resolution images with a downsampling stride of 14. Notably, such designs also ensure the efficiency of MLLMs, where the high-resolution images are processed by the ef-ficient CNN, and the number of visual tokens is also kept small via the large downsampling stride.

#### 4.3. Mixture-of-Resolution Adapter

To better collaborate the feature learning of two pathways, we propose a mixture-of-resolution adapter (MR-Adapter) for the fusion of visual information from different res-olution images. In particular, given the visual features Fvh ∈ Rh×w×dh extracted from a high-resolution image, we embed them into the low-resolution visual pathway by

F′ vl = Fvl + fl(Fvl) + g · fh(Fvh). (3)

Here, Fvl ∈ Rh×w×dl are the features from the low-resolution pathway. fl(·) and fh(·) denote two mapping modules, which are designed as a convolutional block and an MLP layer, respectively. g is a dynamic score to control the weights of high-resolution information, defined by

g = δ(W2σ(W1fv)),

[fl(Fvl) i,j , fh(Fvh)

i,j ]. (4)

Here, [·] denotes the concatenation operation, and W1 ∈ R2d× d

2 and W2 ∈ R d 2×d are two projection matrices.

fv ∈ Rd is the pooled visual features. σ and δ denote the activation function of GELU and Tanh, respectively.

As shown in Fig. 3, high-resolution information can be fused with the features in each block of ViT. In this case, the low-resolution features of ViT also contain rich semantics, improving the visual descriptive power of MLLMs.

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Table 1. Performance and efficiency comparisons of LLaVA-HR and LLaVA-1.5 (Liu et al., 2023a) at different resolutions. Except resolution, the other configurations of LLaVA-HR and LLaVA-1.5 remain the same. The training and inference costs are measured on NVIDIA A800s. “N/A” denotes that GPU memory overflows1. “tokens/s” denotes the number of generated tokens per second.

Models Resolution Vision-Language Tasks Training

Time ↓ GPU

Memory ↓ Inference Speed ↑VQAv2 ↑ TextVQA ↑ MME ↑ POPE ↑

LLaVA-1.5 336 pix 80.44 59.41 1461.17 86.2 15.6h 28G 23.8 tokens/s LLaVA-HR (ours) 384 pix 80.47 59.63 1522.28 86.3 17.6h 34G 23.8 tokens/s LLaVA-1.5 448 pix 81.17 62.17 1493.12 87.2 19.4h 49G 19.9 tokens/s LLaVA-HR (ours) 768 pix 81.80 64.36 1524.75 88.0 18.2h 38G 23.5 tokens/s LLaVA-1.5 672 pix 81.54 64.23 1498.71 87.9 31.8h 79G 12.7 tokens/s LLaVA-HR (ours) 1024 pix 81.90 67.11 1554.90 87.6 20.7h 40G 19.7 tokens/s LLaVA-1.5 1022 pix 74.20 37.80 1266.90 84.4 69.4h N/A1 5.6 tokens/s LLaVA-HR (ours) 1536 pix 81.82 67.96 1480.62 87.7 29.8h 52G 12.6 tokens/s

#### 4.4. The Deployment on MLLM

We apply MRA to a popular MLLM called LLaVA-1.5 (Liu et al., 2023a), and construct a new model, namely LLaVA-HR. Its training consists of two stages, i.e., low-resolution pre-training and high-resolution instruction tuning.

Stage 1: Low-Resolution Pre-training. Similar to LLaVA (Liu et al., 2023b) and LLaVA-1.5 (Liu et al., 2023a), this stage aims to optimize the projector to align the visual features with the word embeddings of LLM. There-fore, the image encoder and the LLM are frozen during pre-training. Besides, we adopt low resolutions for two pathways. In this stage, the MR-Adapter is not inserted, and output features of dual pathways are directly combined.

Stage 2: High-Resolution Instruction Tuning. During instruction tuning, we greatly increase the resolution of the high-resolution pathway, e.g., from 384× 384 to 1,024× 1,024. And the low-resolution one is also accordingly ad-justed to ensure the visual alignment of two pathways, e.g., from 336× 336 to 448× 448. Meanwhile, the MR-Adapter is then applied to connect two visual pathways. Different from the first training stage, the entire MLLM will be fully optimized to better accommodate high-resolution images.

##### 5. Experiments 5.1. Evaluations and Metrics

Multimodal benchmarks for MLLM. We evaluate LLaVA-HR on four emerging multimodal benchmarks for MLLMs, including MME (Fu et al., 2023), POPE (Li et al., 2023c), SEED (Li et al., 2023a) and MM-VET (Yu et al., 2023). In particular, MME and MM-VET evaluate the multimodal per-

1When memory overflows, we reduce the batch size and in-crease the gradient accumulation steps to train LLaVA-1.5.

ception and cognition abilities of MLLMs. SEED extends the modalities of evaluation to images and videos. POPE aims to evaluate the visual hallucinations of MLLMs. The metrics used in our paper follow their default settings. For MME, we follow LLaVA-1.5 to report the perception score.

Common vision-language benchmarks. We also evaluate LLaVA-HR on seven VL datasets, including VQAv2 (Goyal et al., 2017), GQA (Hudson & Manning, 2019), OKVQA (Marino et al., 2019), OCRVQA (Mishra et al., 2019), ScienceQA (Lu et al., 2022), VizWiz (Gurari et al., 2018) and TextVQA. In particular, ScienceQA (Lu et al., 2022), VizWiz (Gurari et al., 2018) and TextVQA are three zero-shot tasks, and their samples are not appeared in our training data. We report the accuracy on the test set of OCRVQA, the test set of VizWiz, and the val set of OKVQA. We organize samples of these tasks in instruction formats of LLaVA-1.5 (Liu et al., 2023a).

#### 5.2. Implementation Details

In LLaVA-HR, we use CLIP-ViT-L (Radford et al., 2021; Ilharco et al., 2021) and CLIP-ConvNeXt-L (Liu et al., 2022) as the dual visual paths to encode low- and high-resolution images, respectively. In LLaVA-HR-X, the CLIP-ConvNeXt-L is replaced with the stronger CLIP-ConvNeXt-XXL. The MR-Adapter is applied into the last three stages of ViT. Following LLaVA-1.5, we first pre-train LLaVA-HR on LCS-558K (Liu et al., 2023b), which contains 558k image-text pairs. During the pre-training stage, both the visual encoder and the LLM are frozen, and only the MLP projector is fine-tuned. AdamW (Kingma & Ba, 2014) is used as the optimizer, and the learning rate and batch size are set to 1e-3 and 256, respectively. Visual resolutions are set to 336×336 and 384×384 for the ViT and the CNN, re-spectively. During instruction tuning, we follow LLaVA-1.5 to use 665k VL instruction data. At this stage, the entire

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Table 2. Comparison of MRA and four baselines on LLaVA-HR. The visual resolution is set to about ∼760× 760.

Settings VQAv2 TextVQA MME POPE Speed

ViT+ MLP 81.0 63.2 1436.1 87.6 10.7 t/s Conv+MLP 80.3 64.6 1415.9 86.6 23.7 t/s ViT+Resampler 79.8 58.9 1403.8 85.8 27.6 t/s ViT+Pooling+MLP 80.6 59.6 1480.6 86.5 23.9 t/s MRA (ours) 81.8 64.4 1524.8 88.0 23.5 t/s

model is updated with a learning rate of 2e-5. Besides, we increase the resolution of ViT and CNN to 448×448 and 1,024×1,024, respectively. The training epoch is set to 1 for pre-training and instruction tuning.

#### 5.3. Experimental Results

##### 5.3.1. QUANTITATIVE ANALYSIS

Comparison with baselines. In Tab. 1, we compare the performance and efficiency of LLaVA-HR with LLaVA-1.5 (Liu et al., 2023a) with different image resolutions. From this table, we observe that increasing image resolution ob-viously improves the performance of two models on four tasks, e.g., +4.8% of LLaVA-1.5 on TextVQA. However, the performance of LLaVA-1.5 drops significantly at the resolution of 1,024×1,024. To explain, the number of vi-sual tokens greatly exceeds the pre-trained context length of the LLM, which easily causes the instability during training. In contrast, the performance of LLaVA-HR is consistently improved from 384 × 384 resolution to 1,024 × 1,024 reso-lution. Besides, the total gain of LLaVA-HR is more obvious than that of LLaVA-1.5 (Liu et al., 2023a), e.g., +8.33% of LLaVA-HR vs. +4.82% of LLaVA-1.5, greatly confirming the effectiveness of MRA.

In Tab. 2, we further compare four common baselines with the similar resolution, i.e., ∼760×760. “ViT+MLP” is the default setting of LLaVA-1.5 as the reference. “Conv+MLP” replaces the visual backbone with ConvNeXt (Liu et al., 2022), which uses a larger downsampling rate to re-duce the number of visual tokens. “ViT+Resampler” and “ViT+Pooling+MLP” refer to the two pooling strategies for reducing the number of visual tokens. As can be seen, all compared methods are inferior to LLaVA-HR. In par-ticular, using a convolutional network as the visual back-bone greatly improves efficiency, but its performance still lags behind LLaVA-HR by a large margin, e.g., -108.9 on MME (Fu et al., 2023). Similarly, “ViT+Resampler” and “ViT+Pooling+MLP” also sacrifice performance for effi-ciency. Overall, these comparisons further confirm the de-signs of MRA.

Despite effectiveness, the expenditure of LLaVA-HR is also cost-effective. In particular, increasing resolution from 384

Table 3. Ablation study of mixture-of-resolution adaptation on LLaVA-HR. The resolution is 768 × 768. Our final setting is colored in gray. “L-Res Path.”, “H-Res Path.”, “Fusion Direct.”, “Struct.” and “Gate Fuct.” denote the low-resolution pathway, the high-resolution pathway, the fusion direction, the structure and the gate function, respectively.

Settings Choices VQAv2 TextVQA MME POPE

L-Res Path.

ViT-L 81.8 64.4 1524.8 88.0 None 80.3 64.6 1415.9 86.6 ViT-G 81.7 65.3 1469.7 87.9

H-Res Path.

ConvNeXt-L 81.8 64.4 1524.8 88.0 None 80.4 59.4 1461.2 86.2

ConvNeXt-XXL 82.3 66.5 1479.2 87.9

Fusion Direct.

High to Low 81.8 64.4 1524.8 88.0 Low to High 81.0 62.8 1463.5 87.3

## Fusion Type

Sum 81.8 64.4 1524.8 88.0 Concat 81.7 64.7 1508.8 87.3

Struct. mlp-conv 81.8 64.4 1524.8 88.0 conv-conv 81.6 64.6 1499.0 87.7 conv-mlp 81.5 64.2 1517.9 87.6

Gate Funct.

Tanh 81.8 64.4 1524.8 88.0 Sigmoid 81.7 64.3 1567.9 86.9

H-sigmoid 81.6 64.4 1525.9 87.8

× 384 to 1,024 × 1,024 slows down the training and in-ference of LLaVA-1.5 by 344.8% and 325%, respectively. However, these costs are reduced to only 17.6% and 20.8% in LLaVA-HR. Despite better performance, the training and inference speeds of LLaVA-HR are three times faster than LLaVA-1.5. Besides, the costs of GPU memory also remain cheap for LLaVA-HR. For example, adapting the resolution of 1,536 × 1,536 for LLaVA-HR only consumes 52G GPU memory, but the same settings for LLaVA-1.5 will cause GPU memory overflow. These results greatly confirm the efficiency of our MRA and LLaVA-HR.

Ablation studies. In Tab. 3, we conduct comprehensive ablation studies for MRA on four VL benchmarks. Firstly, we validate the different designs of the dual visual pathways. From these results, we find that removing one pathway will lead to significant performance drops, e.g., -1.5% on VQAv2. Besides, scaling up the high-resolution encoder brings more gains than that of the low-resolution one, e.g., +2.1% vs. +0.9% on TextVQA. We assume that the stronger high-resolution image encoder can better capture the fine-grained visual information. Then, we ablate different fusion directions and strategies in MRA. Specifically, changing the fusion direction obviously degenerates the performance, e.g.,

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Table 4. Comparison with existing methods on four MLLM benchmarks. “Param.”, “Res.” and “Data” refer to the total parameters, the visual resolution and the number of training data, respectively. “t/s” refers to tokens per second.

Method Settings Multimodal Benchmarks Inference Param. Res. Data MME POPE SEED MM-Vet Speed

BLIP-2 14.2B 224 129M 1293.8 85.3 46.4 22.4 -InstructBLIP 8.2B 224 130M - - 53.4 26.2 22.6 t/s InstructBLIP 14.2B 224 130M 1212.8 78.9 - 25.6 -QWen-VL-Chat 9.6B 448 1.4B 1487.5 - 58.2 - 17.0 t/s Fuyu-8B 8B ∼600 - 728.6 74.1 - 21.4 15.6 t/s mPLUG-Owl2 8.2B 448 400M 1450.2 - 57.8 36.2 19.6 t/s LLaVA-1.5 7.2B 336 1.2M 1510.7 85.9 58.6 30.5 23.8 t/s LLaVA-1.5 13.2B 336 1.2M 1531.3 85.9 61.6 35.4 -

LLaVA-HR 7.4B 1024 1.2M 1554.9 87.6 64.2 31.2 19.7 t/s LLaVA-HR 13.4B 1024 1.2M 1540.9 87.8 64.5 34.8 15.0 t/s LLaVA-HR-X 14B 1024 1.2M 1487.3 88.0 65.3 35.5 12.9 t/s

Table 5. Comparison with existing methods on seven vision-language tasks. SQAI refers to the IMG subset of ScienceQA.

Method Settings In-domain Tasks Zero-shot Tasks Infer. Param. Res. Data VQAv2 GQA OKVQA OCRVQA SQAI VizWiz TextVQA Speed

BLIP-2 14.2B 224 129M 41.0 41.0 45.9 40.6 61.0 19.6 42.5 -InstructBLIP 8.2B 224 130M - 49.2 - - 60.5 34.5 50.1 22.6 t/s InstructBLIP 14.2B 224 130M - 49.5 - 44.8 63.1 33.4 50.7 -Shikra 13.2B 224 6.1M 77.4 - - - - - - -IDEFICS-9B 9B 224 354M 50.9 - 38.4 - - 35.5 25.9 30.5 t/s IDEFICS-80B 80B 224 354M 60.0 - 45.2 - - 36.0 30.9 -Qwen-VL-Chat 9.6B 448 1.4B 78.2 57.5 56.6 70.5 68.2 38.9 61.5 17.0 t/s Fuyu-8B 8B ∼600 - 74.2 - 60.6 - - - - 15.6 t/s mPLUG-Owl2 8.2B 448 400M 79.4 56.1 57.7 - 68.7 54.5 58.2 19.6 t/s LLaVA-1.5 7.2B 336 1.2M 78.5 62.0 - - 66.8 50.0 58.2 23.8 t/s LLaVA-1.5 13.2B 336 1.2M 80.0 63.3 - - 71.6 53.6 61.3 -

LLaVA-HR 7.4B 1024 1.2M 81.9 64.2 58.9 68.4 65.1 48.7 67.1 19.7 t/s LLaVA-HR 13.4B 1024 1.2M 82.3 64.8 60.7 67.7 68.1 57.9 68.1 15.0 t/s LLaVA-HR-X 14B 1024 1.2M 82.6 65.2 61.5 69.0 68.0 56.6 70.9 12.9 t/s

-61.3 on MME. Finally, we ablate the designs of the mixture-of-resolution adapter. Specifically, the best choices of map-ping modules for the low- and high-resolution pathways are convolution blocks and MLP blocks, respectively. Besides, the choices of gating function also affect performance and the tanh function perform the best. These ablations further confirm the designs of MR-Adapter.

Comparison with existing MLLMs. In Tab. 4 - 5, we compare LLaVA-HR with existing MLLMs on 11 VL tasks. On the four MLLM benchmarks, we observe comprehen-sive advantages of LLaVA-HR against existing MLLMs. In particular, LLaVA-HR achieves 1554.9 scores in MME benchmark, outperforming LLaVA-1.5 by +23.6. On POPE, a benchmark including video evaluations, LLaVA-HR-X still outperforms existing MLLMs by a large margin, i.e., +3.7% gains. Besides, LLaVA-HR achieves the best per-

formance on the benchmark for visual hallucinations, i.e., POPE, suggesting that its visual hallucinations are greatly alleviated. Notably, Fuyu-8b (Fuyu-8B, 2023) is capable of high-resolution images, but its performance is much inferior to LLaVA-HR, e.g., 728.6 vs. 1554.9 on MME.

Tab. 5 gives the performance comparison on common VL tasks. On in-domain tasks, LLaVA-HR achieves the best results on three tasks, e.g., 82.6 on VQAv2 and 61.5 on OKVQA. On OCRVQA, Qwen-VL-Chat collects more in-domain data for training, so it performs better than LLaVA-HR. Under the zero-shot setting, we can observe more sig-nificant advantages of LLaVA-HR on the fine-grained tasks, e.g., VizWiz and TextVQA. Most notably, even Qwen-VL-Chat is pre-trained with 24.8M OCR samples, it still per-forms worse than LLaVA-HR-X on TextVQA. These results suggest the significance of high resolution for these tasks. In

https://lh3.googleusercontent.com/notebooklm/AG60hOr6yjw_PX_BfleGh2ivSIo669Yj2eBNOWqT6NPc76EJVcVx2lr0PS2u0BvXJSrr8MsFrdQljkm2qO8C5xgNXXrh6PexEiMxyi3bdKO9pxU86fES39ZHlDpyLEGstTsL3f9JDI1H=w640-h427-v0

39bbca55-47b2-41fc-a567-500df0ab03ad

https://lh3.googleusercontent.com/notebooklm/AG60hOoMyn7Y0YiOopCMAs5hSTLcRu_9j2dl8QBdI_AHifMvJGuEcVgwYuIfXOu5KVu246vGHxyzibgVm0NKAD1DfpWNzWlgGUzyAqj9jvFeffOVBYJJ5OmY9Ojnp2C4tr3lcIvgM97ldg=w320-h334-v0

ac73f844-6a90-4d1b-b243-5164d36cb742

https://lh3.googleusercontent.com/notebooklm/AG60hOowreh7L1A_Kzeqtx39cuKreAH2mvHITz3hpYHSrHK4VIRoLopoMyJiyUeb9IsQ-emiJnk9QUpfiVQiajSK3IdAAuH2rnLnvXRuLN_tP0TnqfwO1k_Z4fbarWRl_XTxFiiR7XYzBw=w540-h359-v0

ddd9d993-e7bc-4671-8c04-01974022661d

https://lh3.googleusercontent.com/notebooklm/AG60hOpZwAIG2x2nHnYy8-LHwcE1fh7MooKzlyaF6d-uqwZBynkdM8qfyJpImjdsHFZ3VyOUgCRNFUY9pFh3QnS-LVoAt_pguOplY49kZkRfZpW13I-as2QRAcPVkKsJhG0d2iljAicENQ=w294-h304-v0

610f2af3-53f3-446b-954b-46392277b300

https://lh3.googleusercontent.com/notebooklm/AG60hOqtyt1rl_vWLNL7XRZhQ0EbTTP9ZbUZp5cT8b9Ap4Q_uCaGRjat-zMfEGCgxYYMKUMZfKeKriMlSV4f6eEGHmLOlSaTzBKrsqxjLRaLCKhXn9DHQDYsi4jO2SUaP2FxDkPK1L3V=w363-h378-v0

683d58e0-0ee9-4678-a01b-792083122e25

https://lh3.googleusercontent.com/notebooklm/AG60hOrxtmIgTiRdgkCj5OOWiaO6JgzgELrUk08EsR1qf_6jfcpLo-KiDgEDSsPyg5wdkLTv18o7jSE7QgazQf7nWNmywSkD1ou6cpbtrw5f8KbU_O9wSJFy9ccDY0lPfYP7QnbXU49w4Q=w640-h408-v0

ff706408-a64c-4abe-ba6a-78fd18df94c3

https://lh3.googleusercontent.com/notebooklm/AG60hOofAkpFNlzPb3Mow5Rb3Rcw8TgV-AqVPV_akgRU_iXDu2ldi8CEw8XjfbLa7XxxT9m433uPDdO7PmECEtGAEe3A7D6YE2Yyw7AY0aVN4JeamZY5p2msk1nzobsjpLZufe55Nyc0kw=w371-h391-v0

c8ca6874-75ce-49fb-bacc-b27b5f407c3f

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

User: What color is the tiniest boat in the forefront?

LLaVA-HR (384 pix): Red

LLaVA-HR (1024 pix): White

User: How many boats are fully visible?

LLaVA-HR (384 pix): 2

LLaVA-HR (1024 pix): 3

User: what page is this book on?

LLaVA-HR (384 pix): This is the first page of a book. LLaVA-HR (1024 pix): This book is on page 87.

User: what are the last three words on this page?

LLaVA-HR (384 pix): "and other stories."

LLaVA-HR (1024 pix): Wo ich wohne.

LLaVA-HR (384 pix):

[0.92, 0.69, 0.97, 0.72]

LLaVA-HR (1024 pix):

[0.66, 0.68, 0.74, 0.73]

Ground-truth: [0.66, 0.68, 0.74, 0.74]

User: Please provide the bounding box coordinate of the region this sentence describes: The  child sliding down in red clothes.

LLaVA-HR (384 pix):

[0.4, 0.6, 0.51, 0.78]

LLaVA-HR (1024 pix):

[0.53, 0.57, 0.6, 0.71]

Ground-truth: [0.52, 0.56, 0.6, 0.7]

User: Please read the text in this image and return the information in the following JSON format (note xxx is placeholder, if the information is not available in the image, put "N/A" instead). {"class": XXX，"DLN": Xxx， "DOB": Xxx，"Name": XXX，"Address": XXX，"EXP": XXX，"ISS": XXX， "SEX": XXX，"HGT": XXX, WGT": XXX，"EYES": XXX，"HAIR": XXX， "DONOR": XXX}

GPT4-V:{"class": "D", "DLN": "97542954****", "DOB": "15-03-1984", "Name": "GP VEERE*****", "Address": "52 KLINK RD ROCHESTER ** *****", "EXP": "10-01-16", "ISS": "09-30-08", "SEX": "M", "HGT": "5-09", "WGT": "N/A", "EYES": "BR", "HAIR": "N/A", "DONOR": "YES"}

LLaVA-1.5: {"class": Class D, "DLN": 97745434, "DOB": 03/15/1988, "Name": Gopal Veerendhar, "Address": 54 Klinck Rd, Rochester, NY 14607, "EXP": 10/01/2016 - 10/01/2017, "ISS": New York State Department of Motor Vehicles, "SEX": Male, "HGT": 5'10", "WGT": 170, "EYES": Brown, "HAIR": Black, "DONOR": N/A}

LLaVA-HR-X: {"class": Class D, "DLN": 97542954****, "DOB": 15/03/1984, "Name": GP Veere*****, "Address": 52 Klink Rd, Rochester ** *****, "EXP": 10/01/2016, "ISS": New York State, "SEX": Male, "HGT": N/A, "WGT": N/A, "EYES": Brown, "HAIR": Black, "DONOR": N/A}

(a) Comparison of LLaVA-HR with different resolutions

(b) Comparison of LLaVA-HR-X, LLaVA-1.5 and GPT4-V in visual information extraction

User: Please provide the bounding box coordinate of the region this sentence describes: The deer on the far right.

Figure 5. Visualizations of LLaVA-HR and existing MLLMs. Subfig-(a) shows that high image resolution greatly improves the capability of MLLMs on fine-grained VL tasks. In Subfig-(b), LLaVA-HR-X demonstrates the comparable ability with GPT4-V in visual information extraction2. Correct and incorrect answers are colored in green and red, respectively.

contrast, most images of ScienceQA are synthetic and of low resolution, so the advantages of LLaVA-HR are not obvious. Overall, these results greatly confirm the effectiveness and generalization of LLaVA-HR and our MRA.

##### 5.3.2. QUALITATIVE EXPERIMENTS

In Fig 5 (a), we compare the predictions of LLaVA-HR with different resolutions. The visualizations show that higher image resolution obviously improves the capability of MLLMs on fine-grained tasks. For example, LLaVA-HR with a resolution of 1,024 × 1,024 can well capture gran-ular visual content, e.g., the tiny boat in the first example. Besides, high image resolution also enables LLaVA-HR a stronger ability of text recognition. For instance, the small and blurred phrase of “wo ich wohne” in the second example are correctly identified by the high-resolution LLaVA-HR. These results greatly confirm the significance of high im-age resolution in addressing visual shortcoming. In Fig 5 (b), we further compare the predictions of LLaVA-HR-X, LLaVA-1.5 (Liu et al., 2023a) and GPT4-V (OpenAI, 2023)

2For privacy reasons, we blur some key personal information.

in visual information extraction. Notably, LLaVA-HR-X shows a comparable ability with GPT4-V on this challeng-ing task. As shown in Fig 5 (b), LLaVA-HR-X and GPT4-V can correctly extract almost all visual content of the driver license and organize it in JSON format. Compared to GPT4-V, LLaVA-HR-X also correctly identifies the hair color of the person, which requires fine-grained visual reasoning. In contrast, LLaVA-1.5 can only recognize simple visual con-tent like “class” and “SEX”, and fail to extract most visual information. These results further validate the effectiveness of MRA in addressing visual shortcoming of MLLMs.

##### 6. Conclusion In this paper, we study the visual shortcoming of MLLMs from the perspective of image resolution, and propose a novel and efficient method for high-resolution adaptations of MLLMs, namely mixture-of-resolution adaptation (MRA). MRA adopts dual visual pathways to process images of both high and low resolutions, where high-resolution informa-tion is embeded into the low-resolution modeling via the novel mixture-of-resolution adapters (MR-Adapters). We

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

apply MRA to a popular MLLM called LLaVA-1.5, and construct a new high-resolution MLLM, termed LLaVA-HR. Experimental results not only validate the effectiveness of LLaVA-HR in addressing visual shortcoming, but also confirm its remarkable efficiency against existing MLLMs.

Acknowledgements. This work was supported by Na-tional Key R&D Program of China (No.2022ZD0118201) , the National Science Fund for Distinguished Young Scholars (No.62025603), the National Natural Science Foundation of China (No. U21B2037, No. U22B2051, No. 62176222, No. 62176223, No. 62176226, No. 62072386, No. 62072387, No. 62072389, No. 62002305 and No. 62272401), the Natural Science Foundation of Fujian Province of China (No.2021J01002, No.2022J06001), and the China Funda-mental Research Funds for the Central Universities (Grant No. 20720220068).

References Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I.,

Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and Zhao, R. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023a.

Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G. E. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems (NeurIPS), 33:22243–22255, 2020.

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.

Chen, X., Wang, X., Beyer, L., Kolesnikov, A., Wu, J., Voigtlaender, P., Mustafa, B., Goodman, S., Alabdul-mohsin, I., Padlewski, P., et al. Pali-3 vision lan-guage models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023b.

Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Qiu, Z., Lin, W., Yang, J., Zheng, X., et al. Mme: A compre-hensive evaluation benchmark for multimodal large lan-guage models. arXiv preprint arXiv:2306.13394, 2023.

Fuyu-8B. https://www.adept.ai/blog/ fuyu-8b, 2023.

Gilardi, F., Alizadeh, M., and Kubli, M. Chatgpt outper-forms crowd-workers for text-annotation tasks. arXiv preprint arXiv:2303.15056, 2023.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answer-ing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., Luo, J., and Bigham, J. P. Vizwiz grand challenge: Answering visual questions from blind people. In Pro-ceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., and Schmidt, L. Openclip. July 2021. doi: 10. 5281/zenodo.5143773. URL https://doi.org/10. 5281/zenodo.5143773. If you use this software, please cite it as below.

Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., and Chen, X. In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10267– 10276, 2020.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. Seed-bench: Benchmarking multimodal llms with gener-ative comprehension. arXiv preprint arXiv:2307.16125, 2023a.

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Boot-strapping language-image pre-training with frozen im-age encoders and large language models. arXiv preprint arXiv:2301.12597, 2023b.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023c.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved base-lines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, 2023b.

Liu, S., Cheng, H., Liu, H., Zhang, H., Li, F., Ren, T., Zou, X., Yang, J., Su, H., Zhu, J., Zhang, L., Gao, J., and Li, C. Llava-plus: Learning to use tools for creating multimodal agents, 2023c.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.

Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. arXiv preprint arXiv:1908.02265, 2019.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Infor-mation Processing Systems, 2022.

Luo, G., Zhou, Y., Ren, T., Chen, S., Sun, X., and Ji, R. Cheap and quick: Efficient vision-language instruction tuning for large language models. Advances in neural information processing systems (NeurIPS), 2023a.

Luo, G., Zhou, Y., Sun, J., Sun, X., and Ji, R. A survivor in the era of large-scale pretraining: An empirical study of one-stage referring expression comprehension. IEEE Transactions on Multimedia, 2023b.

Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Merigan, W. H. and Maunsell, J. H. How parallel are the primate visual pathways? Annual review of neuroscience, 16(1):369–402, 1993.

Mishra, A., Shekhar, S., Singh, A. K., and Chakraborty, A. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pp. 947–952. IEEE, 2019.

OpenAI. Gpt-4v(ision) system card. https: //cdn.openai.com/papers/GPTV_System_ Card.pdf, 2023.

Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., and Wei, F. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.

Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., and Zhang, L. Grounded sam: Assembling open-world models for diverse visual tasks, 2024.

Robertson, L. C. and Lamb, M. R. Neuropsychological contributions to theories of part/whole organization. Cog-nitive psychology, 23(2):299–330, 1991.

Rose, D., Himakunthala, V., Ouyang, A., He, R., Mei, A., Lu, Y., Saxon, M., Sonar, C., Mirza, D., and Wang, W. Y. Visual chain of thought: Bridging logical gaps with multi-modal infillings. arXiv preprint arXiv:2305.02317, 2023.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209, 2024.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation lan-guage models. arXiv preprint arXiv:2302.13971, 2023.

Yu, J., Li, J., Yu, Z., and Huang, Q. Multimodal transformer with multi-view visual representation for image caption-ing. IEEE transactions on circuits and systems for video technology, 30(12):4467–4480, 2019.

Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multi-modal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., Choi, Y., and Gao, J. Vinvl: Revisiting visual representa-tions in vision-language models. In CVPR, 2021.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

