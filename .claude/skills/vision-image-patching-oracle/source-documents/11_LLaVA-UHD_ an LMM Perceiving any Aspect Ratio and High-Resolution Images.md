---
sourceFile: "LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:18.761Z"
---

# LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images

183a0a83-e081-4b86-be3b-3aa7417ea04d

LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images

fa00cacc-28a3-4760-84e4-b2d26da61f28

https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11080.pdf

LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images

Zonghao Guo3 , Ruyi Xu1 , Yuan Yao2⋆ , Junbo Cui1 , Zanlin Ni1 , Chunjiang Ge1 , Tat-Seng Chua2, Zhiyuan Liu1 , and Gao Huang1⋆

1 Tsinghua University 2 National University of Singapore

3 University of Chinese Academy of Sciences guozonghao96@outlook.com, yaoyuanthu@gmail.com

Abstract. Visual encoding constitutes the basis of large multimodal models (LMMs) in understanding the visual world. Conventional LMMs process images in fixed sizes and limited resolutions, while recent explo-rations in this direction are limited in adaptivity, efficiency, and even correctness. In this work, we first take GPT-4V and LLaVA 1.5 as rep-resentative examples and expose systematic flaws rooted in their visual encoding strategy. To address the challenges, we present LLaVA-UHD, a large multimodal model that can efficiently perceive images in any aspect ratio and high resolution. LLaVA-UHD includes three key components: (1) An image modularization strategy that divides native-resolution im-ages into smaller variable-sized slices for efficient and extensible encod-ing, (2) a compression module that further condenses image tokens from visual encoders, and (3) a spatial schema to organize slice tokens for LLMs. Comprehensive experiments show that LLaVA-UHD outperforms established LMMs trained with 2-3 orders of magnitude more data on 8 benchmarks. Notably, our model built on LLaVA-1.5 336×336 supports 6 times larger (i.e., 672×1008) resolution images, and achieves 5.7 accuracy improvement on TextVQA.

1 Introduction

Recent progress in Large Multimodal Models (LMMs) [5, 11, 23, 26, 27] has wit-nessed a significant surge in vision-language understanding, reasoning, and in-teraction capabilities. This is achieved by projecting visual signals into Large Language Models (LLMs) to enable their visual perception of the world, where visual encoding strategy plays a fundamental role [3, 21, 27]. Real-world images are known to reside in a wide range of aspect ratios and resolutions, presenting significant challenges for LMMs.

However, most existing LMMs [8, 11, 27] perceive images in a fixed aspect ratio (i.e., 1:1) and a low resolution (i.e., 224×224). The compromise to this simplified setting typically leads to shape distortion and blur of image contents.

⋆ Corresponding Authors

2 Z. Guo et al.

The problem significantly hurts the capabilities of LMMs, especially for fine-grained capabilities, such as small object understanding [19] and optical char-acter recognition [5, 16, 39]. Moreover, the issue also exacerbates hallucination problems (i.e., producing textual responses not factually grounded in images), since models can only learn to make best guesses to blurred images [36].

To achieve image perception in varied aspect ratios and high resolutions for LMMs, there are two main challenges: (1) Adaptivity. Since visual encoders (e.g., CLIP-ViT [33]) are pretrained in fixed and simplified settings, it can be difficult to deal with images in a wide range of aspect ratios and resolutions. Simple image interpolation that deviates far from the pretraining scenarios can result in out-of-distribution issues. (2) Efficiency. Directly encoding high-resolution images requires quadratic computation cost with respect to image sizes. In addition, it can be even more costly for LLMs to process the large number of image tokens in high-resolution images (e.g., 4096 tokens for 896×896 images in ViT-L/14).

Moreover, careless visual encoding strategies can even result in systematic flaws in correctness. For example, despite its powerful capabilities in various aspects, it has been commonly reported that GPT-4V [2] can surprisingly strug-gle in some basic capabilities, such as identifying the number of objects [38]. The mechanistic cause for such embarrassment remains largely unknown. In this work, we perform the first mechanistic investigation of GPT-4V flaws from the perspective of visual encoding strategy. Our controlled experiments in probing GPT-4V show that the problem can be partially rooted in its careless visual en-coding strategy in dealing with high-resolution images. Investigation on LLaVA-1.5 [25], a representative open-source LMM also shows systematic issues.

To address these challenges, we present LLaVA-UHD, a large multimodal model that efficiently perceives any aspect ratio and high-resolution images. The model has three key components: (1) At the core of LLaVA-UHD is an im-age modularization strategy that divides native-resolution images into smaller variable-sized slices for efficient and extensible encoding. In comparison to recent works that fit images into several fixed aspect ratios and resolutions [23,24], the variable-sized slices in LLaVA-UHD enable full adaptivity to native-resolution images without padding or shape-distorting resizing. This is in analogy to the better adaptivity of using water drops vs. ice cubes in full-filling variable-sized glasses. We also show that the strategy guarantees minor deviation from the pre-training setting of visual encoders to maximally retain their capabilities. (2) The visual tokens are condensed by a compression layer to modest lengths, largely reducing the computation for LLMs. (3) Finally, the compressed slice tokens are organized in a spatial schema to inform LLMs about the slice positions in the image.

Comprehensive experiments on 8 benchmarks show that LLaVA-UHD signif-icantly improves the capabilities of LMMs, outperforming established counter-parts trained with 2-3 orders of magnitude more data. Notably, our model built on LLaVA-1.5 336×336 supports 672×1008 resolution images, and achieves 5.7 accuracy improvement on TextVQA and 3.0 accuracy improvement on POPE. The advantage enlarges with more extreme aspect ratios.

LLaVA-UHD 3

The contribution of this work can be summarized as threefold: (1) We perform the first mechanistic investigation of GPT-4V from the perspective of visual encoding strategy and expose systematic flaws. (2) We present LLaVA-UHD, a large multimodal model that can efficiently perceive any aspect ratio and high-resolution images. (3) We conduct comprehensive experiments to demonstrate the effectiveness of LLaVA-UHD on 8 popular benchmarks, and also provide analysis for deeper understanding.

2 Pilot Experiments

We start with a pilot experiment on the visual encoding strategies of existing LMMs, taking GPT-4V [2] and LLaVA-1.5 [25] as representative examples. GPT-4V is a powerful and most recognized proprietary LMM, while LLaVA-1.5 is one of the most influential open-source LMMs. Despite their strong performance in many aspects, it has been commonly reported that dilemmas can be encountered in some basic capabilities [38]. For example, GPT-4V is prone to miscounting the object numbers in images, whereas the causes remain largely unknown.

In this work, we perform the first mechanistic investigation of GPT-4V flaws from the perspective of visual encoding strategy. The key idea is that by using synthetic images as continuous probes, we can evaluate the behaviors of GPT-4V in a highly controlled manner, thereby identifying the underlying causes. Our experimental results indicate that, some systematic flaws of GPT-4V are likely to be rooted in its visual encoding strategy, which can be potentially exploited for adversarial attacks.

2.1 GPT-4V Experiments

Preliminary. According to the publicly available information from OpenAI,4 GPT-4V employs two image processing modes: Low resolution and high reso-lution. (1) In low-resolution mode, for an original image with dimensions W and H, the model processes only a low-resolution overview image. (2) In high-resolution mode, besides the overview image, GPT-4V processes additional slices of the original high-resolution image, where each slice has 512× 512 resolution, resulting in ⌈ W

512⌉ × ⌈ H 512⌉ slices in total. In our experiments on GPT-4V’s new

high-resolution mode, interesting error patterns are observed, prompting an ex-ploration into GPT-4V’s underlying visual encoding logic.

How do positions in images influence GPT-4V’s behavior? Our ex-periments start with a simple instance: Given the image as shown in Fig. 1(a), we ask GPT-4V: “How many circles are there in the image?” We synthesize a series of image variants by changing the positions of circles in the image, and keep the text prompt unchanged. For better reliability, we also synthesize images using other colors and shapes as well, from {red, green,white}×{circle, triangle, square}. For each instance, we query 15 times to better approximate the response distribution.

4 https://platform.openai.com/docs/guides/vision

https://lh3.googleusercontent.com/notebooklm/AG60hOrFoG-X55FNOjHXzQje8--Li6fv_CezlyrppGDjbION-wVGSpB_XcvkU-uoeeDfbIo-mOhEl1sU91EUrUXpo6LoX2tT-xqQTzPa288SSzxIuDpKBFlrghQMVb0BRhseX8Nlc8P2=w370-h370-v0

4e1d9ed8-55d3-436e-846b-7bfc682c0af7

https://lh3.googleusercontent.com/notebooklm/AG60hOqnVjOx1pWUtRMWqQmHjA_Lgy0fDmok2sL8et0NRuG-hr2LICOOxjEI8rKEpp9SXoSM3zOqWR1fjKNtmDsCIhQ1ZMpudXfBYjp5FQoGBVlllg73qF88HYFkfghvYD2cNo1GynET5w=w18-h369-v0

2aec1db9-d600-4774-92b4-297e1c862501

https://lh3.googleusercontent.com/notebooklm/AG60hOrTYe0ny1jwh667khG6OOKq_Ikr3owOHMxVHzhhdMdSmrXXx1GaWWzoJlN3UfJTMHDKrhNiLRhC0y_-EnvH91EohzL-hSJWiWnvbwQw6SZsvhBvqx5jbMyF8fF_6Ich6gOLz7HdAA=w370-h370-v0

2918bc67-aec3-4b6e-ab8a-ed4b50733d76

https://lh3.googleusercontent.com/notebooklm/AG60hOo5s6NBqzf8Trc4ZUdHt3y55jdAwbu_PC765oOULNp-81-Fn2R1sZboOr8WOuPmwvcntPpujBgS6EYE0_uyzuNnHtuMiszajTisfNJYDinqWLx2801HtCd5KhH1i0D-9Qssmdw_nA=w18-h369-v0

aef71a10-cd6a-4e8c-8c76-0015814b7695

https://lh3.googleusercontent.com/notebooklm/AG60hOrS4QKObPNy2ynJrgjK3_hynqgXYRW4l1_2JD6pcok3h8k_e_oaIhyjjlddjZ_9M0LC4-7fml2v_UrGBteElm8z2HA16zldh_oa5det-KQ5TIMt1GoJxFWGwcRuHASo3Y9r2Xc8mA=w370-h370-v0

245fc9f0-5d38-4707-90a7-bc4c8153bc72

https://lh3.googleusercontent.com/notebooklm/AG60hOrZkRktHuvTi7c3OdZ95QheguQnAzBr8YR4WXwtsFCVmlDy9q-qXc9uXK9vUIPffI1ce_YnutBFH6P_-AHha8XVvwybbFYAgykEInxA9dTq7GAWy6rsdJ8cQXFaqDGGTN9gQq6LcA=w18-h369-v0

533fa6e4-add1-44c4-a37f-234ca9415dbf

https://lh3.googleusercontent.com/notebooklm/AG60hOodRtC2HLvpVWo8n0h4c6wNe2dXdKOO3XBe-e-fY8-W9J14jsRp4U2TtFtAcmofBWxXyz8qCkKLy3SIKHqnrfce6VXKX007GvwhkR37Qy-vraToS6b0rwg70S0zYIAZmJdMFJ9Z=w370-h370-v0

2b16a6c8-c74c-491b-a2c6-10bf20b6b99b

https://lh3.googleusercontent.com/notebooklm/AG60hOofzWDkMK38Qv5eXSkPFv3h13dnjOGj29HV9GZcs-ht2G0mLNzuCPGRF4T0SofhdEXlWC23K6yZXm3O9PSg4Yun3DtaLYQRloAR6l53vYvx00orQTqWNp6nk_2pQE2HR5w2QbR9HQ=w18-h369-v0

5aed6ad2-c22a-4f37-b082-b8a9cf07f180

https://lh3.googleusercontent.com/notebooklm/AG60hOr8cWKpCnkBvO9xRvsa-1Bz_yKEWRK5WpD-oH6Ouy7b3NEcBx1b4f8autvktWjHVVggnWBYBgLJjqbXRhxeskMScYxDfsXJhuYb8LwY_VuWOcIPeIp3aDHFtt1ZBaLeWWUSPdLkiQ=w818-h800-v0

a3747948-b092-44ad-b9df-ad9a5af7002c

https://lh3.googleusercontent.com/notebooklm/AG60hOpTJidjiqLtvFSvTsk9DPkJebd2uy1vjXJm_IA_mmLmcN8tOTygMOZI5DKBVdewRUFtN1z0_JI3h14NY9lEOjLU89Q4EQXeUsaJP1fsGFcHDGKQzMn1WvgY3WWQLALDJA=w370-h370-v0

69b794fd-6960-44b4-9b3d-3f89481abbee

https://lh3.googleusercontent.com/notebooklm/AG60hOpF2M1D1T629M6rsif7UjMnCR03oj1x0go7c8RFJG67ruEJBsXvvyiLa__Awy1yJySxMnx8i0kAu-SnH9-dacD1ynUrEh3wSWcbRrG7xhSS6k5yAaMWx2VttOoBTAS9D2ajS0eWQQ=w18-h369-v0

d9993534-623e-4582-9229-7420331f1fb9

https://lh3.googleusercontent.com/notebooklm/AG60hOpCf001ExPfIzLrYSO8LvBdwLPwkSZyKKMRdTFmyOfJ_fLf4D1ZypgmNFCQvsIehJGHOxbKOF_b6-N-obxGVmHdtvKJgylsLhPuPQAyTUykO1jRHoRsIdGPsN3pUqctvMlD6OQdLg=w768-h768-v0

06a9db48-11cd-4814-a77e-ce0d5039b033

https://lh3.googleusercontent.com/notebooklm/AG60hOoruWb3cmUDy_YkveMgI3rVdgj_TDm3lo_V-U_G-M2jX2q2H5QIVZC5kktWHiANv7tJ-TJ4Hhr1cTwDyM9sLA0_YErnaeI9olIcZ7aISVPcreXjRkBC0F0-XtycntUhAnfBf2o2zA=w768-h768-v0

e016e633-4249-49d2-91f2-f17c364b286a

https://lh3.googleusercontent.com/notebooklm/AG60hOqh-7GbbHnAgari6owOzjZHs4_Jcm-5TUjAnZrqYNgSaMmz_tv1uiJXTuZ8gaMWxw8bBPGydP2Ymjzl0BtZd34VylZjtFqNA_XjaOOCNvD3m7GEwGIDukgN1FyQj6J-eIKUEXet_g=w370-h370-v0

e4d325db-32a7-4472-8335-b6bf6247d6dc

https://lh3.googleusercontent.com/notebooklm/AG60hOr3pUT6DJWMyjaECzu1W-IQGFiyK29FUKPOKgeFDFNP8tmuwzZMQgDS0M5xfiVf0OpaaIpqeNvMgJ3dIHkkjeLaFjchOrTFUDsFH-CWASUoYkb54rHn-SSSA5kN2waK_zGb_WiCrg=w18-h369-v0

cd4f1872-90b4-4200-bece-f78903daab0a

4 Z. Guo et al.

(a) Input image (b) Average output (d) 2N answer(c) Correct answer (4)  (8)

(g) Close e(f) 4N answer (16) rror (5)1N 2N 4N

(e) Output pattern   (h) Close error (3) 4N2N1N

Fig. 1: Experimental results of GPT-4V in identifying numbers of objects. Note that the dashed lines in (a) are for illustration purposes only, and not presented to GPT-4V.

We calculate the average number answered by GPT-4V for each position in the image, and report the heatmap in Fig. 1(b). We can observe that the result is highly correlated with object positions in images. Specifically, the patterns are split by 256 × 256 squares, and three interesting patterns can be identified: (1) The central square exhibits the highest response number, (2) the middle edges show a lower number, and (3) the corners are the closest to ground truth.

To investigate the cause, we further separate the model responses by number, and report the distribution across positions for each response in Fig. 1(c), (d), (f), (g) and (h). Interestingly, besides the correct answers (4: 66.1%) and close answers (5: 16.6%, 3: 10.2%), it turns out that the remaining two abnormal answers (8: 5.2%, 16: 1.9%), which doubles and quadruples the ground truth, account for the error pattern in Fig. 1(b). Combining the results with the public information from OpenAI, we hypothesize the most likely cause is that, there are overlaps in the slices of GPT-4V when the image resolution is not divisible by 512.5 As illustrated in Fig. 1(e), the overlapping areas between two slices will double the number, and the overlapping areas between four slices will quadruple the number.6

How do image resolutions influence GPT-4V’s behavior? To verify the hypothesis, we further probe GPT-4V through continuously changing image resolutions. Specifically, we proportionally resize the image in Fig. 2(a) into different resolutions, and query about the object number in the same way. For each resolution, we repeatedly query 30 times for better reliability.

5 Note that the issue is different from the overlapping sliding windows in CNNs, since the overlaps in GPT-4V is inconsistent across different resolution images.

6 Note that besides visual encoding strategies, model behaviors are also influenced by the accumulated training dynamics and RLHF. Therefore the double/quadruple effect does not dominate the results. All results are from GPT-4V on 03-05-2024.

https://lh3.googleusercontent.com/notebooklm/AG60hOrx_aBCous0rb7UX4Cxi7AVP-sVAeBsGbguX1xaid0cC2Muyrenvu8gwk3D0JHK2KzXFhNEcNCEbOhLh_wShoqB69dcouNe9VCW4K9y_c2gI_en9Sj3Z9Yfjdhi4Nm2_RnRx8uxSg=w800-h800-v0

4a30b9ca-6dbe-44a5-b637-42203c4e58d9

LLaVA-UHD 5

(b) Answer distribution(a) Input image

Phase 1    Phase  2    Phase  3

## Pr op ot io n

256px256px

Fig. 2: Results on probing GPT-4V via continuously changing image resolutions.

1:9 1:2 2:3 3:2 2:1 9:1

## Left Right Top Bottom

<1:1 87.1% 100% 100% 100%

>1:1 99.8% 99.9% 99.9% 96.7%

## Attack success rate

Fig. 3: Experimental results of adversarially attacking LLaVA-1.5 using input images containing padding pixels. Left: Attack success rates where LLaVA-1.5 ignores the grey area and answers the color of the central rectangle (e.g., green). Right: Synthesized input images containing (1) a rectangle in varied aspect ratios, and (2) padding pixels.

We report the experimental results in Fig. 2(b). We observe that the model responses show a significant phase change with image resolutions: (1) In phase 1, since there are no image slices, most answers are correct; (2) In phase 2, answer 12 dominates the responses possibly due to the incomplete circles in each slice. (3) Phase 3 shows mixed answers of 9, 12 and 16. Note that 16 can be well explained by the error pattern in Fig. 1(e). We refer readers to the appendix for a more detailed illustration of each phase. Besides, we also notice that many abnormal phenomena in Fig. 2(b) cannot be perfectly explained yet, which we leave for future work.

In conclusion, these experimental findings shed light on GPT-4V’s potential vulnerabilities in high-resolution image processing, warranting further investiga-tion into the implications of these weaknesses and the development of strategies to counter potential adversarial attacks on LMMs.

2.2 LLaVA-1.5 Experiments

To deal with images with varied aspect ratios, LLaVA-1.5 pads the input images into squares before feeding them into the visual encoder. We argue that this encoding method results in a waste of computation for non-square images. For example, a 1:4 image has only 25% effective computation after padding into squares. To quantify the influence, we train an unpadded version of LLaVA-1.5, by fitting the ViT position embedding into the aspect ratio of input images using 2D interpolation. The resultant image tokens remain no more than 576 as

6 Z. Guo et al.

## A B C D E F

## Input Image

## Standard Slice Area

N = ⌈ 6.5 ⌉ =7

Factorization & Partition

2×3 6×1 1×6

Score Function S(·) A B C D E F

## Best Partition

Shared Vision Transformer (CLIP-ViT-L/14)

## Shared Compression Layer

Large Language Model (Vicuna-13B)

形变partitions

2D Positional Embedding Interpolation

## Positional Embedding

, \n , , \n

## Ideal Slice Number

Fig. 4: The LLaVA-UHD framework. Left: Given a high-resolution image, LLaVA-UHD first calculates the ideal slice number, and then selects the best partition from pos-sible factorizations, splitting the high-resolution image into varied-sized slices. Right: Slices are encoded in native aspect ratios by 2D interpolation on position embeddings, and then compressed and arranged in a spatial schema for LLM processing.

in LLaVA-1.5 (see Sec. 3.1). From the experimental results in Table 2, we observe that adaptive aspect ratio encoding without padding consistently improves the performance of LLaVA-1.5.

Another issue of padding is that, the model essentially cannot know whether the padding-like pixels come from image pre-processing or an actual part of the original input image. To demonstrate this issue, we synthesize a series of input images as in Fig. 3(right), where blue/green/red rectangles in various aspect ra-tios are surrounded by grey (i.e., the RGB of LLaVA-1.5’s padding value). Given the input image, we prompt: “What is the color of the left/right/top/bottom most area?” From the results in Fig. 3(left), we observe that LLaVA-1.5 neglects the grey input areas (considering them as padding), and faithfully responds with the color of the central rectangle.

2.3 Conclusions on Pilot Experiments

In summary, both powerful proprietary LMMs such as GPT-4V and open-source LLaVA-1.5 have systematic issues in their underlying visual encoding strategies. The results show that visual strategies must be designed with caution. Common practices such as padding, shape-distorting resizing, and repetitive slicing can result in a waste of computation, a loss of model capability, and even vulnerabil-ity to adversarial attacks. Therefore, there is an urgent need for more adaptive and efficient visual encoding methods.

LLaVA-UHD 7

Based on the principles learned from the pilot experiments, we propose LLaVA-UHD, a large multimodal model that can efficiently perceive any aspect ratio and high-resolution images. As shown in Fig. 4, the model includes three key components: (1) An image modularization strategy that divides native-resolution images into smaller variable-sized slices for efficient and extensible encoding, (2) a compression module that further condenses image tokens from visual encoders, and (3) a spatial decoration schema to organize slice tokens for LLMs.

3.1 Modularized Visual Encoding

To deal with high-resolution images with varied aspect ratios, a naive approach is to interpolate the position embeddings of ViT to the target shape for direct encoding as a whole. However, this approach is sub-optimal due to the quadratic computation cost and the performance degradation from out-of-distribution is-sues. To address the challenge, we present a modularized visual encoding strat-egy. The basic idea is to divide native-resolution images into smaller variable-sized slices, where the shape of each slice does not deviate too far from the stan-dard pretraining setting of ViT. With variable-sized slices, LLaVA-UHD can achieve full adaptivity to native-resolution images without padding or shape-distorting reshaping.

High-Resolution Image Partition Strategy. The goal of image slicing strategy is to determine a split of high-resolution images, with minimal changes to the resolutions of each slice. Given an image in resolution (WI , HI) and a ViT pretrained in resolution (Wv, Hv), we first determine the number of slices (i.e., the ideal computation) needed to process the image: N = ⌈WI

Hv ⌉. Then we

factorize the slice number N into m columns and n rows: CN = {(m,n)|m×n = N,m ∈ N, n ∈ N}. To select the most appropriate partition, we define a score function to measure the deviation from the standard pretraining setting of ViT:

S(WI , HI ,Wv, Hv,m, n) = − ∣∣∣∣log WI × n

HI ×m − log

∣∣∣∣ , (1)

where higher score S(·) indicates a smaller deviation from the standard setting of ViT, and is thus preferred. Therefore the partition can be obtained as follows:

m∗, n∗ = argmax (m,n)∈C̄

S(WI , HI ,Wv, Hv,m, n), (2)

where the candidate set C̄ = CN. In practice, we notice that in some cases, there might be only a few possible factorization schemes for N , especially for prime numbers, which can lead to limited choices and therefore extreme partitions of images. For example, N = 7 has only two extreme partition choices, 1:7 and 7:1. To address the issue, in addition to the ideal slice number N , we also allow a modest change of slice numbers N−1, N+1 to incorporate more partition choices. Therefore, the final partition is given by Eq. 2, where C̄ = CN−1 ∪CN ∪CN+1.

8 Z. Guo et al.

Theoretically, we show that the partition strategy guarantees minor ex-pected changes and modest worst-case changes with respect to standard pre-training resolution (Wv, Hv) for each slice. Specifically, we show that for im-ages where N ≤ 20 and aspect ratio in [1 : 6, 6 : 1], the aspect ratio of each slice resides within [1 : 2, 2 : 1], and the area of each slice resides within [0.33WIHI , 1.5WIHI ]. We refer readers to the appendix for full proof details.

Arbitrary Aspect Ratio Slice Encoding. Existing LMMs utilize a static resolution for image slice encoding [5,11,25]. This essentially prevents full adap-tivity to native resolutions, since only several predefined fixed-shape slices are available. Moreover, the static slice resolution inevitably incurs padding or shape-distorting resizing, which hurts the performance, efficiency, and even correctness as shown in Sec. 2.

To address the problem, we propose to encode image slices in aspect ratios given by the partition strategy as is. Specifically, we proportionally resize the original image following the aspect ratio, such that the number of patches maxi-mally fits within the pretraining budget M (i.e., the number of position embed-dings in ViT). Then we reshape the pretrained 1D position embedding sequence of ViT into P ∈ Rq×q×l following its pretraining setting, where M = q × q, and l is the dimension of position embeddings. After that, we 2D-interpolate P to fit the slice resolution given by the partition strategy for visual encod-ing. In our experiments, we show that ViT and position embedding parameters can be kept frozen during pretraining, and updating these parameters during the instruction-tuning stage is sufficient for good performance. In addition to slices, we also provide a low-resolution overview image in native aspect ratio. The overview image can provide coarse-grained information and global semantic connections in images.

3.2 Compression Layer

High-resolution images require LLMs to process significantly more visual tokens, which accounts for a major part of the computation. For example, a 672× 1008 resolution image will produce 3,456 visual tokens for LLaVA-1.5 [25]. To address the issue, we propose a spatially constrained resampler to compress the visual tokens of each image slice. Specifically, image tokens output by the visual encoder are decomposed into 12×12 equal regions, whose aspect ratios are the same as the input image’s. ROI-Align [15] is then applied to crop image tokens within these regions into a group of feature maps. These maps are of the same shape, selected from a set of predefined shapes (4×2, 3×2, 4×3, 3×3 and their transpose), where Eq. 3.1 is adopted to determine the most appropriate shape. For each feature map, we use one learnable query to perform feature integration via cross-attention. Consequently, the number of the image tokens is significantly reduced (from 576 to 144 in our experiments). Compared with the prevalent MLP-based visual projection approaches, the spatially constrained resampler maintains a fixed and affordable number of visual tokens regardless of image resolutions, and is therefore more compatible with high-resolution images.

LLaVA-UHD 9

3.3 Spatial Schema for Image Slices

Since the image partition is dynamic across different images, it is necessary to inform LLM of the spatial organizations of image slices. Inspired by [6], we design a spatial schema to inform the relative positions of image slices using two special tokens. Specifically, we use “,” to separate the slice representations in a row, and use “\n” to separate different rows. In our experiments, we find that the simple schema can effectively inform the dynamic partition to yield good performance.

4 Experiments

In this section, we empirically investigate the effectiveness of LLaVA-UHD. We first provide the implementation details of the model. Then we report the evalu-ation results on 8 common benchmarks compared with strong baselines. Finally, we provide analytic results for better understanding of the model.

4.1 Implementation Details

Model Configuration. LLaVA-UHD is built based on LLaVA-1.5 [25]. Specifi-cally, we use the CLIP-ViT-L/14 as visual encoder (default resolution 336× 336), Vicuna-13B [9] as LLM, and a shared spatially constrained resampler as the projector to connect the visual encoder and LLM. During the encoding of image slices, a minor reshape within half patches (typically 7-8 pixels) could be per-formed to fit the slice into patches. The number of learnable queries in resampler is set to 144. For the image partitioned as N sub-patches, the number of visual tokens fed into LLM is 144× (N +1), with tokens of the low-resolution overview image. Following LLaVA-1.5, we perform a two-stage training as follows.

Stage 1: Pretraining details. During this stage, only the spatially con-strained resampler is tuned, with the CC-558K dataset [27] for 1 epoch, using AdamW optimizer with a learning rate of 1e−3 and the cosine learning rate schedule. The global batch size is set to 256. The training cost of this stage is ∼4 hours using 8×A100s.

Stage 2: Instruction-tuning details. During this stage, the visual en-coder is frozen and we fine-tune the resampler and LLM, with a 665K mixture dataset [25] which contains LLaVA-Instruct [27], TextCaps [34], GQA [17], OCR-VQA [31], and Visual Genome [18]. The learning rate is 2e−5 and batch size is 128. The training cost of this stage is ∼22 hours using 8×A100s.

4.2 Experimental Setting

We introduce experimental settings, including the benchmarks, evaluation met-rics, and baselines in our experiments.

Benchmarks. We adopt extensive popular benchmarks to evaluate our model, including: (1) General visual question answering benchmarks such as VQA-V2 [4], GQA [17], ScienceQA [29], and VizWiz [14]; (2) Optical character based

10 Z. Guo et al.

Table 1: Main results on 8 popular benchmarks. #PT.: pretraining data volume, #IT.: instruction-tuning data volume, MaxRes.: maximum resolution supported, AR.: aspect ratio supported, ∆: improvements over LLaVA-1.5 backbone.

Model #PT. #IT. MaxRes. AR. VQAv2 GQA VQAT POPE VizWiz MMEP MMBen SEEDI

BLIP-2 [21] 129M - 224×224 Fix 41.0 41.0 42.5 85.3 19.6 1293.8 - 46.4 InstructBLIP [11] 129M 1.2M 224×224 Fix - 49.5 50.7 78.9 33.4 1212.8 - -

Shikra [8] 600K 5.5M 224×224 Fix 77.4 - - - - - 58.8 -Qwen-VL [5] 1.4B 50M 448×448 Fix 78.8 59.3 63.8 - 35.2 - 38.2 62.3 SPHINX [24] 1B 2.8M 448×448 Fix 78.1 62.6 51.6 80.7 39.9 1476.1 66.9 71.6

SPHINX-2k [24] 1B 2.8M 762×762 Fix 80.7 63.1 61.2 87.2 44.9 1470.7 65.9 71.6 MiniGPT-v2 [7] 323M 2.9M 448×448 Fix - 60.1 - - 53.6 - - -

Fuyu-8B [6] - - 1024×1024 Any 74.2 - - 74.1 - 728.6 10.7 -OtterHD-8B [19] - 390K 1024×1024 Any - - - 86.0 - 1223.4 58.3 -

mPLUG-Owl2 [40] 400M 1.23M 448×448 Fix 79.4 56.1 58.2 86.2 54.5 1450.2 64.5 57.8 UReader [39] 86M - 896×1120 Enum - - 57.6 - - - - -Monkey [23] - 1.44M 896×1344 Enum 80.3 60.7 - 67.6 61.2 - - -

LLaVA-1.5 [25] 558K 665K 336×336 Fix 80.0 63.3 61.3 85.9 53.6 1531.3 67.7 68.2

LLaVA-UHD (ours) 558K 665K 672×1008 Any 80.7 64.9 67.0 88.9 55.0 1532.3 68.2 68.4 ∆ - - ×6 times - +0.7 +1.6 +5.7 +3.0 +1.4 +1.0 +0.5 +0.2

visual question answering benchmark such as TextVQA [35]; (3) Hallucina-tion benchmark such as POPE [22]; (4) Comprehensive benchmarks such as MME [13], MMBench [28], and SEED-Bench [20].

Baselines. We compare our model with strong baselines. (1) General base-lines. We adopt Qwen-VL [5], LLaVA-1.5 [25], MiniGPT-v2 [7], Shikra [8], BLIP-2 [21] and InstructBLIP [11] as representative general baselines. Since LLaVA-UHD is built on LLaVA-1.5, it serves as the most direct baseline. (2) High-resolution LMMs. SPHINX [24] and mPLUG-Owl2 [40] encode images in fixed resolutions; Ureader [39] and Monkey [23] support enumerated resolution types (several predefined fixed-shape slices); Fuyu-8B [6] and OtterHD-8B [19] can encode images in any resolutions.

4.3 Main Results

We report the main experimental results in Table 1, from which we have the following observations: (1) LLaVA-UHD outperforms strong baselines on pop-ular benchmarks. This includes strong general baselines trained on 2-3 orders of magnitude more data such as Qwen-VL and InstructBLIP, and also high-resolution LMMs that require significantly more computation such as Fuyu-8B, OtterHD-8B, Monkey and SPHINX-2k. The results show that LLaVA-UHD can properly deal with native-resolution images for strong performance, as well as good data efficiency. (2) LLaVA-UHD achieves significant improvements over the LLaVA-1.5 backbone. Notably, by simply perceiving images in native high-resolution, LLaVA-UHD achieves 5.7 accuracy improvement on TextVQA and 3.0 accuracy improvement on POPE. The reason is that the blurred content in low-resolution images can prevent LMMs from accurately identifying the chal-lenging fine-grained objects and optical characters. The results demonstrate the fundamental role of perceiving native high-resolution images in various multi-modal tasks, and the effectiveness of LLaVA-UHD in addressing the problem.

LLaVA-UHD 11

Table 2: Ablation Results. FP: Fixed image partition strategy.

Model Training duration VQAv2 GQA VQAT POPE SQA VizWiz

LLaVA-1.5 21h 80.0 63.3 61.3 85.9 71.6 53.6 w/ adaptive enc. 21h 80.5 63.9 62.3 87.1 71.8 56.7

LLaVA-UHD 22h 80.7 64.9 67.0 88.9 71.2 55.0 w/ MLP 52h 81.6 65.4 66.5 88.3 70.7 57.6 w/ MLP & FP. [24] 62h 81.2 64.5 66.4 88.1 71.1 54.5

Table 3: Experimental results on extreme aspect ratio images. Performance and the degradation from the standard benchmarks in Table 2 are reported.

Model VQAv2 GQA VQAT POPE SQA VizWiz

LLaVA-1.5 77.6 (-2.4) 60.7 (-2.6) 52.4 (-8.9) 82.5 (-3.4) 70.9 (-0.7) 53.6 (-2.9) w/ adaptive enc. 78.6 (-1.9) 62.2 (-1.7) 57.3 (-5.0) 85.3 (-1.8) 71.2 (-0.6) 55.2 (-1.5)

LLaVA-UHD 80.4 (-0.3) 64.3 (-0.6) 61.5 (-5.5) 86.5 (-2.4) 70.7 (-0.5) 52.9 (-2.1) w/ MLP 81.3 (-0.3) 64.5 (-0.9) 60.5 (-6.0) 85.4 (-2.9) 70.3 (-0.4) 55.3 (-2.3) w/ MLP & FP. [24] 79.6 (-1.6) 61.9 (-2.6) 58.5 (-7.9) 84.4 (-3.9) 69.4 (-1.7) 52.2 (-2.3)

4.4 Analytic Results

We provide further analysis of our model, including ablation on alternative com-ponents, evaluation on images with more extreme aspect ratios, best practice for frozen/trainable parameters, and case study.

Ablation Study. In Table 2, we conduct ablation studies on alternative components. (1) We replace the padding strategy of LLaVA-1.5 with the adap-tive encoding strategy of LLaVA-UHD, supporting arbitrary aspect ratios while maintaining identical maximum resolutions. We can observe consistent improve-ment since wasted computation from padding is avoided. (2) We replace the proposed spatially constrained resampler of LLaVA-UHD with the 2-layer MLP of LLaVA-1.5. We observe that the resampler achieves comparable or better performance than MLP, using only 42.3% training hours with 8×A100s. (3) We further replace the LLaVA-UHD image partition strategy with the naive parti-tion strategy [24] (i.e., fixed 2 × 2 slices). Results show that LLaVA-UHD can more properly divide images into slices for better performance.

LLaVA-UHD generalizes to images with extreme aspect ratios. We further investigate the generalization capability of LLaVA-UHD by constructing an extended version of existing benchmarks. Specifically, we expand the aspect ratio of an image by doubling the length of its longer side through padding. From the results in Table 3, we can see that the advantage of LLaVA-UHD increases as compared with LLaVA-1.5 and alternatives. The reason is that LLaVA-UHD perceives images in native aspect ratios. In comparison, LMMs that encode im-ages in fixed aspect ratios will suffer from significant distortion in the content

12 Z. Guo et al.

Table 4: The effect of tuning visual encoder at different training stages.

Update ViT VQAv2 GQA VQAT POPE SQA VizWizPre-training Fine-tuning

80.7 64.9 67.0 88.9 71.2 55.0 ✓ 80.4 65.3 66.2 88.2 70.3 55.4

✓ ✓ 79.4 63.1 63.5 87.4 70.1 52.3

LLaVA-1.5 [25] 80.0 63.3 61.3 85.9 71.6 53.6

shapes. Moreover, this also causes the computation to be unevenly distributed along the width and height of the image content.

Freezing ViT parameters when instruction-tuning is sufficient for adaptation. We investigate the effect of tuning ViT parameters at different training stages. From the results in Table 4, we observe that: (1) Updating ViT during instruction-tuning brings insignificant promotion on these benchmarks. We found that LLaVA-UHD can improve over LLaVA-1.5 even when ViT param-eters are frozen in both pretraining and instruction tuning stages. (2) Further updating ViT during pretraining does not lead to better results. We hypothesize that jointly training ViT and the spatially constrained resampler (from scratch) on limited pretraining data can lead to instability issues.

Case Study. To provide a more intuitive understanding of the behaviors and capabilities of LMMs in dealing with high-resolution images, we provide qualitative results for LLaVA-UHD and LLaVA-1.5 in Fig. 5. We can see that LLaVA-UHD can correctly identify the dense content in the timetable (Case 1), the text on the small poster (Case 2), and icons and text on the phone (Case 3) for fine-grained recognition and reasoning. In comparison, LLaVA-1.5 can only perceive coarse-grained information, and therefore tends to provide either uninformative (Cases 1 and 2) or incorrect/hallucinated answers (Case 3) in these challenging scenarios. The results demonstrate the effectiveness and advantage of LLaVA-UHD in perceiving native aspect ratio and high-resolution images for fine-grained multimodal capabilities.

Discussion. The common perceiver resamplers [3,5] achieve significant fea-ture compression and exhibit high-performance potentials, a fact recently vali-dated by MiniCPM-V [7], which adopts our proposed visual encoding strategy for high-resolution images with any aspect ratio. However, pre-training the per-ceiver resampler from scratch can require on extensive training data to achieve stable and good performance (e.g., ∼500M in MiniCPM-V and ∼1.4B in Qwen-VL). Under conditions of limited pre-training data, e.g., 558K image-text pairs in our experimental setting, it can be difficult for the perceiver resampler to real-ize its full potential. In contrast, MLP as a projector demonstrates fast training convergence and robust feature alignment with a limited data volume. This may be due to MLP’s ability to retain the spatial information of visual features after projection. Inspired by this, we designed the spatially constrained resampler to compress visual features while preserving the spatial relationships among visual

https://lh3.googleusercontent.com/notebooklm/AG60hOr1ocK2fwOC79ijvsyU65lRj5MV3W1YSA0ok4yN15PAAwx7bGKBHGIM_HZPpTTWoCmz7SSdOWS1e8UV1PNg3nHbOTij-gbA0LmpAu0YuQu5aKRh_f2Rp436yBhK-QMkgyoQLIy9=w47-h47-v0

b45da7e5-13aa-4aa9-979d-69ecba775dd8

https://lh3.googleusercontent.com/notebooklm/AG60hOqVQh-G0HXPshSVLu7Iy3mWhLk5wmFBMSAtD6KGxrPn_2FakFUh6xl8pYV_FG8ggcCYqr7fcuTrmyjhs_UJQg0SDbE5K4OcSGqWmSymWxU7EGmliI4cZgdTKHlQy3oANs3NiqFN=w47-h47-v0

ab6838cb-51c9-4e31-9b6b-d1e55b3b78c8

https://lh3.googleusercontent.com/notebooklm/AG60hOrg-7F02XxMyg_gnFXA1pcbIIESA2dDL14MPW6zjlNfzYZsqvpw4Qo1stAQcDiQC33tPtNiDErmJDcH6Y8PlaTJqxHKoPbuYKBdLElxbhEUboAwXGRa5Ctp9yC4ClyBdZ9ffxzg=w961-h768-v0

c08be048-f41f-49f0-b37a-d660cec781c6

https://lh3.googleusercontent.com/notebooklm/AG60hOoVv1OVeKMX6eDTMPrD6rPZ7P3r4j6pQFP53ZyiHRRiD9bxUdXIhouqgbx1mhyI1auPRo0JVF-9ZD0gHLtSH3K8yPaGPOgG9AddF2Ejoa5s33x6HFjrVBdYxJnlE4Pc6fxBE_MK=w167-h237-v0

a666e691-f84c-40e0-a4e1-ff8c05c36891

https://lh3.googleusercontent.com/notebooklm/AG60hOrAL3h2EGs1RwyzgJzJIcogP0idkT6NNPrflfzax2hyhj4B6GzFXN3jn23mkWCZ36cY9b1dNmB7jVs7FPQ1Ryyu3-JNqX6L_nPwqnUr86iCJj_fp7055QoQ_aXxv9odgA2PBaqS4Q=w44-h44-v0

08e06167-5a5a-487c-93ed-d9bbe438daf0

https://lh3.googleusercontent.com/notebooklm/AG60hOqFpdDbAyejAD8cFf1yg32K0yYoNlnh6FlaBF9feWr1ebw5WDD4hnJ1n5owJlF7xtEBDuObm0Gw6LaEFfLkUlOk9TmpNaNr7cp4BmLH8WYhkQyb2Cpij6fuDeNtkOSbb2qEEntT3A=w47-h47-v0

fcabd4df-e92c-4c61-8731-bcdfd9e848f6

https://lh3.googleusercontent.com/notebooklm/AG60hOraeClI8Dka7qpxFxQ4kmvVS2eHHBftlCnXJ8pZPTa_Ezu0Psiy1s55Te0_vmo4v4O71W0Lf4DD6RdF1J4VwHM5POumT16axh0rmrquXZik_Kc5SEhgyBEbev1gYtkVxibHJnym=w47-h47-v0

d0d92f68-6864-4372-acf3-65465b1ac87d

https://lh3.googleusercontent.com/notebooklm/AG60hOqM5i-F0VvEyNgFE_Zlq53INEDaYrUDv6vbERwQpcU83hSnETWDUz-1nTo_wVKJ8ovxQijKBuHGxe7LGiPzuywE8JRbtzJTEIBXn2mqi8N2z7FCOxwRHUIXNNzK1te36xEhE88cgA=w420-h540-v0

23a9d542-0865-40d5-b006-e2f27562f87e

https://lh3.googleusercontent.com/notebooklm/AG60hOo7atyMMjYGZE1AWyOxY9COr8PXJP-O6GiQEwYFH-puY2-omulD6uN5nGFObE9aoJS_l5BHQP23Ffg-nrpZbUzbNO37mG7Lm9kt1NvJ0BvCdutl0YvRf_4Ln0wrg7k05mqHzwhaJw=w44-h44-v0

6676d4f6-ee67-4ade-bb5a-e34110d006f5

https://lh3.googleusercontent.com/notebooklm/AG60hOohiTn3W9FOD4I5P-P07ssPgAVMV0sv9EPK3r46WChxJzQV1cHfTQ0Fudq3FS4U7fu121lOBcj_9Fq4wOtBVuhti-5zdJZv93qtiMMxWa3CqRPqsd9I90WdZlbn-V8MU9bkGYwy=w44-h44-v0

fcb0f19a-78a3-4b04-9573-f94f8d79f83b

https://lh3.googleusercontent.com/notebooklm/AG60hOrZEs3ujSCWdTVQ48JMEFFVvVhtteQCGV0Ofple6QqvpFeprdiNOa6EIBVBoRzr7-H-uD78EfgwShSTOYkOr8GyuHCUUhCfOvJqL0A29zyqCMiDnLsiuyDGGS455s5KhCdC4gnQ=w47-h47-v0

e6edca87-7afc-4e1c-b00a-d44d9529f9f6

https://lh3.googleusercontent.com/notebooklm/AG60hOocMLv6bzqhfq0hIrTXI1UV3DH6mXAHYR2IUhBwwD8FhNW0YpuPuB1fn3NqLURA5JeYKgWO_0ag0Hl_HF6NYLq8goaS0RDFkhs8lyI8yQ9D3NcuoglJfYR2FiRlLBO15NSvAOseBA=w47-h47-v0

f13ba799-387e-4512-9651-08f94fbe5a57

https://lh3.googleusercontent.com/notebooklm/AG60hOqpU3v-SJQqFUuCnMy3qvJleHA_JMdrzCXVHdGu009Og9HtvdpznAfUlknasOq1NsE6m9annsrN2w1T8ld_bQucRAtuHzf4D1OJLnjuEtgiiSKCzgAu2SPSOse7RTyfr8DFLeCr=w1024-h683-v0

70d2018c-3b10-4b7f-803a-1eb6007c84ad

https://lh3.googleusercontent.com/notebooklm/AG60hOqnBqMzDK31R-EfAhQbOGn_Goyqdf_yxDTuvjYamfxfVMR4znkA2LgqHZuHZda87KPnX3_PHG0mbXESHGznsGIGU9eUqIyqTHFCi41bUV3oKX7R9K5-XLlCyvbsDnTmfqd4tXqz=w952-h1033-v0

4b38a04e-1886-4a4c-9b6f-8ff2ff74e2cb

LLaVA-UHD 13

The poster in the image displays the text "SAVER MENU." This suggests that the restaurant is advertising a menu with lower-priced items, which is a common marketing strategy for fast-food restaurants to attract customers looking for affordable options.

LLaVA-UHD (Ours)

The poster in the image shows a menu with various food items, but the text is too small to read clearly. It appears to be a promotional poster for the McDonald's restaurant, likely displaying their menu offerings or special deals.

What is the opening time of this restaurant ?

The image shows a sign with the opening times for the restaurant. It appears to be open from 12:00 to 2:00 PM and from 5:00 PM to 7:00 PM.

LLaVA-UHD (Ours)

The image shows a sign with the opening times for the restaurant. It appears to be open from 11:30AM to 1:30 PM and from 4:00 PM to 7:00 PM from Monday to Saturday. The restaurant is closed on Sundays.

What is written on the poster ?

How is the weather today according to this weather report on my phone ?

The weather report on your phone indicates that it is 19 degrees Celsius outside. It also shows that there are changes in temperature expected throughout the day. The weather appears to be partly cloudy, suggesting a mix of sun and clouds.

LLaVA-UHD (Ours)

The weather report on the phone screen indicates that the temperature is 19 degrees Celsius. However, the image does not provide enough information to determine the specific weather conditions. In this case, only the temperature is visible.

Fig. 5: Qualitative comparison of LLaVA-UHD and LLaVA-1.5 in fine-grained recog-nition and reasoning capabilities.

tokens. Experiments demonstrate the effectiveness and efficiency of the proposed spatially constrained resampler.

5 Related Work

Visual Encoding in LMMs. The advent of ChatGPT [1] and GPT-4 [2] has spurred the development of numerous open-source large language models (LLMs) [9,10,37,41]. Utilizing an LLM as a language encoder and decoder, there springs up plenty of multi-modal LMMs [3, 5, 11, 16, 21, 27], with aim at under-standing visual image. Therefore, how to project vision features into LLMs be-comes the core problem in the community. Existing visual projection approaches can be divided into three categories. (1) Flamingo [3] proposes perceiver resam-pler, which utilizes a fixed number of queries to capture visual features by cross-attention. (2) BLIP-2 [21] pretrains a Q-Former to bridge the image encoder and LLMs (OPT [41] and T5 [10]). (3) LLaVA [27] just leverages an MLP to connect language and vision feature space. Beyond them, SPHINX [24] mixes kinds of visual features, including DINO-V2 [32], CLIP-ViT&CNN [33] and Q-Former to augment visual representation.

14 Z. Guo et al.

However, since these LMMs rely on the pre-trained CLIP-ViT [33] that re-quires fixed resolution image as input, it hinders LMMs from handling images with higher resolution or any aspect ratio, undermining fine-grained downstream tasks like optical character recognition or small object understanding.

High-resolution LMMs. To perceive images with higher resolutions, recent work can be divided into four categories. (1) Up-Resize. Qwen-VL [5] interpo-lates the positional embedding of ViT to 448×448 and additionally executes a training stage to fine-tune the ViT. MiniGPT-v2 [7] followed this idea but with-out fine-tuning the parameters of visual encoder. CogAgent [16] marries a large low-resolution encoder with a high-resolution feature maps. These methods dra-matically change the original visual position encoding of CLIP-ViT [33], which can cause sub-optimal visual representation. (2) Fix+Crop. To address the above issue, SPHINX [24] utilizes a fixed window size (224×224) to crop a padded im-age (448×448) into four slices, and concatenates them with a down-sampled 224×224 image as visual inputs. Monkey [23] follows this idea yet increases the accessible image size to 896×1344 using a pre-trained perceiver resampler. (3) Fix+Enumerated-Crop. UReader [39] and LLaVA-1.6 [26] enumerate a similar aspect ratio to resize, rather than using a fixed square ratio (e.g., 2×2 as in SPHINX [24]). The unavoidable image resizing and padding operation might cause image deformation and waste of computation, respectively. (4) Any. Fuyu-8B [6] and Otter-HD [19] directly encode visual features using LLMs instead of vision transformers by splitting and projecting images into patches tokens. Treating image patches as a sequence allows processing images with continuous resolution. However, the lack of an image encoder results in insufficient visual representation and limits performance. Nowadays, LMMs [12, 30, 42] capable of processing high-resolution images are continuously emerging.

In comparison, LLaVA-UHD supports images in any aspect ratios and high resolutions. By integrating the advantages of modularized and adaptive image encoding, as well as a novel resampler, LLaVA-UHD can achieve strong perfor-mance with improved data efficiency.

6 Conclusion

In this work, we present LLaVA-UHD, a large multimodal model that efficiently perceives any aspect ratio and high-resolution images. Comprehensive experi-mental results on 8 popular benchmarks demonstrate the effectiveness of LLaVA-UHD, especially in fine-grained multimodal capabilities. Analytical evaluation results are provided for deeper understanding of the model. In this work, we limit the resolution of LLaVA-UHD to maximum 672 × 1008. In future, considering the promising efficiency and scalability, we will explore higher-resolution images and more challenging tasks such as small object detection and segmentation. Be-sides, image slices are currently independently encoded, with interactions only in LLMs. We plan to establish efficient connections between image slices via im-proved visual encoding strategies for fine-grained global information interaction.

LLaVA-UHD 15

Acknowledgement. This work is supported by the National Science and Technology Major Project (2020AAA0106502) and National Natural Science Foundation of China (No. 62236004).

## References

############# 1. Introducing ChatGPT. https://openai.com/blog/chatgpt (2022) 2. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,

D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

################################ 3. Alayrac, J., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Men-sch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J.L., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisser-man, A., Simonyan, K.: Flamingo: a visual language model for few-shot learning. In: NeurIPS (2022)

############## 4. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: VQA: Visual question answering. In: IEEE ICCV. pp. 2425–2433 (2015)

############## 5. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966 (2023)

############## 6. Bavishi, R., Elsen, E., Hawthorne, C., Nye, M., Odena, A., Somani, A., , gnak Ta Ìğ sÄśrlar., S.Ì.: Introducing our multimodal models. adept.ai/blog/fuyu-8b, 2023

############### 7. Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chan-dra, V., Xiong, Y., Elhoseiny, M.: MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478 (2023)

########### 8. Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., Zhao, R.: Shikra: Unleash-ing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195 (2023)

################### 9. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., et al.: Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality. See https://vicuna. lmsys. org (accessed 14 April 2023) (2023)

########################################## 10. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S.S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., Zhao, V.Y., Huang, Y., Dai, A.M., Yu, H., Petrov, S., Chi, E.H., Dean, J., Devlin, J., Roberts, A., Zhou, D., Le, Q.V., Wei, J.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022)

################ 11. Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: InstructBLIP: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500 (2023)

############################# 12. Dong, X., Zhang, P., Zang, Y., Cao, Y., Wang, B., Ouyang, L., Zhang, S., Duan, H., Zhang, W., Li, Y., Yan, H., Gao, Y., Chen, Z., Zhang, X., Li, W., Li, J., Wang, W., Chen, K., He, C., Zhang, X., Dai, J., Qiao, Y., Lin, D., Wang, J.: Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k HD. arXiv preprint arXiv:2404.06512 (2024)

16 Z. Guo et al.

################ 13. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)

################ 14. Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: VizWiz grand challenge: Answering visual questions from blind people. In: IEEE CVPR. pp. 3608–3617 (2018)

########## 15. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask R-CNN. In: IEEE ICCV. pp. 2961–2969 (2017)

################ 16. Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., et al.: CogAgent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914 (2023)

########## 17. Hudson, D.A., Manning, C.D.: GQA: A new dataset for real-world visual reasoning and compositional question answering. In: IEEE CVPR. pp. 6700–6709 (2019)

################# 18. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual Genome: Connecting language and vision using crowdsourced dense image annotations. IJCV 123, 32–73 (2017)

########### 19. Li, B., Zhang, P., Yang, J., Zhang, Y., Pu, F., Liu, Z.: OtterHD: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219 (2023)

########### 20. Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., Shan, Y.: Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125 (2023)

######## 21. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ICML (2023)

############# 22. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hal-lucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023)

############## 23. Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., Bai, X.: Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607 (2023)

##################### 24. Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., Han, J., Huang, S., Zhang, Y., He, X., Li, H., Qiao, Y.: SPHINX: the joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575 (2023)

########## 25. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023)

############## 26. Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: LLaVA-NeXT: Im-proved reasoning, ocr, and world knowledge. https://llava-vl.github.io/blog/ 2024-01-30-llava-next/, 2024

################## 27. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. NeurIPS 36 (2024) 28. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J.,

He, C., Liu, Z., et al.: MMBench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2023)

############### 29. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35, 2507–2521 (2022)

########### 30. Luo, G., Zhou, Y., Zhang, Y., Zheng, X., Sun, X., Ji, R.: Feast Your Eyes: Mixture-of-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003 (2024)

########### 31. Mishra, A., Shekhar, S., Singh, A.K., Chakraborty, A.: OCR-VQA: Visual question answering by reading text in images. In: IEEE ICDAR. pp. 947–952 (2019)

LLaVA-UHD 17

################ 32. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

################## 33. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)

################ 34. Sidorov, O., Hu, R., Rohrbach, M., Singh, A.: TextCaps: A dataset for image captioning with reading comprehension. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J. (eds.) ECCV. vol. 12347, pp. 742–758 (2020)

############## 35. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards VQA models that can read. In: IEEE CVPR. pp. 8317– 8326 (2019)

################## 36. Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L.Y., Wang, Y.X., Yang, Y., et al.: Aligning large multimodal models with factually augmented RLHF. arXiv preprint arXiv:2309.14525 (2023)

################# 37. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)

############# 38. Yang, Z., Li, L., Lin, K., Wang, J., Lin, C.C., Liu, Z., Wang, L.: The dawn of LMMs: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421 9(1), 1 (2023)

################ 39. Ye, J., Hu, A., Xu, H., Ye, Q., Yan, M., Xu, G., Li, C., Tian, J., Qian, Q., Zhang, J., et al.: UReader: Universal OCR-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126 (2023)

############## 40. Ye, Q., Xu, H., Ye, J., Yan, M., Liu, H., Qian, Q., Zhang, J., Huang, F., Zhou, J.: mPLUG-Owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257 (2023)

########################### 41. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M.T., Li, X., Lin, X.V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P.S., Sridhar, A., Wang, T., Zettlemoyer, L.: OPT: open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022)

############ 42. Zhang, Y., Wen, Q., Fu, C., Wang, X., Zhang, Z., Wang, L., Jin, R.: Beyond LLaVA-HD: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487 (2024)

