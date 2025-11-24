---
sourceFile: "An Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking - Inria"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:09.172Z"
---

# An Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking - Inria

10f7d81f-165c-4209-b783-efc548689c92

An Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking - Inria

c52207e9-3b5a-4420-b53d-463f699433e2

https://www-sop.inria.fr/reves/Basilic/2007/DBDLSV07/thresholdmaps.pdf

Eurographics Symposium on Rendering (2007) Jan Kautz and Sumanta Pattanaik (Editors)

## An Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking

George Drettakis1, Nicolas Bonneel1, Carsten Dachsbacher1, Sylvain Lefebvre1, Michael Schwarz2, Isabelle Viaud-Delmon3

1REVES/INRIA Sophia-Antipolis,2University of Erlangen,3CNRS-UPMC UMR 7593

We present a new perceptual rendering pipeline which takes into account visual masking due to contrast and spa-tial frequency. Our framework predicts inter-object, scene-level masking caused by partial occlusion and shadows. It is designed for interactive applications and runs efficiently on the GPU. This is achieved using a layer-based approach together with an efficient GPU-based computation of threshold maps. We build upon this prediction framework to introduce a perceptually-based level of detail control algorithm. We conducted a perceptual user study which indicates that our perceptual pipeline generates results which are consistent with what the user per-ceives. Our results demonstrate significant quality improvement for scenes with masking due to frequencies and contrast, such as masking due to trees or foliage, or due to high-frequency shadows.

### 1. Introduction

Rendering algorithms have always been high consumers of computational resources. In an ideal world, rendering algo-rithms should only use more cycles to improve rendering quality, if the improvement can actually be perceived. This is the challenge of perceptually-based rendering, which has been the focus of much research over recent years.

While this goal is somewhat self-evident, it has proven hard to actually use perceptual considerations to improve rendering algorithms. There are several reasons for this. First, understanding of the human visual system, and the resulting cognitive processes, is still limited. Second, there are few appropriate mathematical or computational models for those processes which we do actually understand. Third, even for models which do exist, it has proven hard to find efficient algorithmic solutions for interactive rendering.

In particular, there exist computational models for con-trast and frequency masking, in the form of visual difference predictors orthreshold maps[Dal93,Lub95,RPG99]. These models were developed in the electronic imaging, coding or image quality assessment domains. As a consequence, ray-tracing-based algorithms, which are a direct algorithmic ana-logue of image sampling, have been able to use these models to a certain extent [BM98, RPG99, Mys98]. For interactive rendering however, use of these models has proven harder.

To date, most solutions control level of detail for objects in isolation [LH01], or involve pre-computation for texture or mesh level control [DPF03]. In what follows, the termobject corresponds typically to a triangle mesh.

Contrast and spatial masking in a scene is often due to the interaction of one or a set of objects onto other objects. To our knowledge, no previous method is able to take these scene-level (rather than object-level) masking effects into ac-count. Shadows are also a major source of visual masking; even though this effect has been identified [QM06], we are not aware of an approach which can use this masking effect to improve or control interactive rendering. Also, the cost of perceptual models is relatively high, making them unattrac-tive for interactive rendering. Finally, since perceptual mod-els have been developed in different contexts, it is unclear how well they perform for computer graphics applications, from the standpoint of actually predicting end-user percep-tion of renderings.

In this paper, we propose a first solution addressing the restrictions and problems described above.

First, we present a GPU-based perceptual rendering framework. The scene is split into layers, allowing us to take into account inter-object masking. Layer rendering and appropriate combinations all occur on the GPU, and are followed by the efficient computation of a threshold

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOqXz2rHXacqskB45lcw16svXwJYqILaaP7VDqksWv6p0zUgvKgDeg-T-EdmT2jVFTCW27yh_1CDtnf52LvZrGKC06COLcTU-LiwHRE4k9voAJvdohahmO9QCRfI1Qpbhp-R91E1Fg=w512-h512-v0

b08c042f-e137-4058-8e09-a7830f3010ae

https://lh3.googleusercontent.com/notebooklm/AG60hOqjGDgt_bPXo3VubYs7qgR_oA55LMMD9wUsEKy3h8ZiFLmVVNLgXgDuQfKLMXVYBuY3nvSs1efnsPvb0FKrmjaxuEzchpi3ZTggPAMjsdo7newEv-um6hcInKgNcEd9MovV1rYz2w=w512-h512-v0

116b601c-e17c-4e7f-9e97-d024a928eaaa

https://lh3.googleusercontent.com/notebooklm/AG60hOrEW40i-2Z6HYRZwJBOdY-4fT0LZvNI8Yaw5kA0srnFWZmTn2dZkNGNFCPun7TQdnZ4_1oHfjQmzyWp6CT8OujFNt7CPIt6FzBatYhdlpGWHEtgCr55ufmgA210xhvbPvpxRJ4qOA=w512-h512-v0

bc14ea04-8e0b-42b5-ac23-13f7e0d1d3fe

https://lh3.googleusercontent.com/notebooklm/AG60hOqOdrs298vES_vLH3YybTBkE7vb_EKVMj1ZK2UYdFihc8TD0rS2vwmGvG6IMsjdT6ycK-3oUp969K4_BGKdHSo6DC2lf013zWARhGOKzNv9dK4nvRAsmwMoVme3kTqMIqlUB-O6Bw=w512-h512-v0

7541e359-7497-4fd8-a868-6ea72669812b

https://lh3.googleusercontent.com/notebooklm/AG60hOpf8CJH_xx9fjnLYY7GquNbZyxB6Lo-NFLJVoqjB1I0L-VanR9uNlwd6q0Kz1U0f5soZ7iEaEBKiABx8GxshIO4LZ8DRvXfRW3pQOTzU9BhFpVFmJWZdXaJzagxEABkKZJV7_RjPQ=w373-h510-v0

8d234f7e-c79b-4c81-b69d-3d7d8c9c086e

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

Figure 1: Left to right: The Gargoyle is masked by shadows from the bars in a window above the door; our algorithm chooses LOD l = 5 (low quality) which we show for illustration without shadow (second image). Third image: there is a lower frequency shadow and our algorithm chooses a higher LOD (l= 3), shown without shadow in the fourth image. The far right image shows the geometry of the bars casting the shadow.

map on the graphics processor. This results in interactive prediction of visual masking.

Second, we present a perceptually-driven level of detail (LOD) control algorithm, which uses the layers to choose the appropriate LOD for each object based on predicted contrast and spatial masking (Fig.1)

Third, we conducted a perceptual user study to validate our approach. The results indicate that our algorithmic choices are consistent with the perceived differences in images.

We implemented our approach within an interactive render-ing system using discrete LODs. Our results show that for complex scenes, our method chooses LODs in a more ap-propriate manner compared to standard LOD techniques, re-sulting inhigher qualityimages for equal computation cost.

### 2. Related Work

In electronic imaging and to a lesser degree in computer graphics, many methods trying to exploit or model hu-man perception have been proposed. Most of them ulti-mately seek to determine the threshold at which a lumi-nance or chromatic deviation from a given reference im-age becomes noticeable. In the case of luminance, the re-lation is usually described by a threshold-vs-intensity (TVI) function [FPSG96]. Moreover, the spatial frequency content influences the visibility threshold, which increases signifi-cantly for high frequencies. The amount of this spatial mask-ing is given by a contrast sensitivity function (CSF). Finally, the strong phenomenon of contrast masking causes the de-tection threshold for a stimulus to be modified due to the presence of other stimuli of similar frequency and orienta-tion.

Daly’s visual differences predictor (VDP) [Dal93] ac-counts for all of the above mentioned effects. The Sarnoff VDM [ Lub95] is another difference detectability estima-tor of similar complexity and performance which operates solely in the spatial domain. Both Daly’s VDP and the Sarnoff VDM perform a frequency and orientation decom-

position of the input images, which attempts to model the detection mechanisms as they occur in the visual cortex.

We will be using a simplified algorithm, introduced by Ramasubramanian et al. [RPG99] which outputs athreshold map, storing the predicted visibility threshold for each pixel. They perform a spatial decomposition where each level of the resulting contrast pyramid is subjected to CSF weighting and the pixel-wise application of a contrast masking func-tion. The pyramid is collapsed, yielding an elevation factor map which describes the elevation of the visibility thresh-old due to spatial and contrast masking. Finally, this map is modulated by a TVI function.

In computer graphics, these perceptual metrics have been applied to speed up off-line realistic image synthesis sys-tems [BM98, Mys98, RPG99]. This is partly due to their rather high computational costs which only amortize if the rendering process itself is quite expensive. The metrics have further been adapted to incorporate the temporal domain, al-lowing for additional elevation of visibility thresholds in an-imations [MTAS01, YPG01]. Apart from image-space ren-dering systems, perceptual guidance has also been employed for view-independent radiosity solutions. For example, Gib-son and Hubbold [GH97] used a simple perception-based metric to drive adaptive patch refinement, reduce the num-ber of rays in occlusion testing and optimize the resulting mesh.

One of the most complete models of visual masking was proposed by Ferwerda et al. [FPSG97]. Their model predicts the ability of textures to mask tessellation and flat shading artefacts. Trading accuracy for speed, Walter et al. [WPG02] suggested using JPEG’s luminance quantization matrices to derive the threshold elevation factors for textures.

The local, object- or primitive-based nature of in-teractive and real-time rendering, has limited the num-ber of interactive perception-based approaches. Luebke and Hallen [LH01] perform view-dependent simplification where each simplification operation is mapped to a worst-case estimate of induced contrast and spatial frequency. This

c© The Eurographics Association 2007.

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

estimate is then subjected to a simple CSF to determine whether the operation causes a visually detectable change. However, due to missing image-space information, the ap-proach is overly conservative, despite later improvements [WLC∗03]. Dumont et al. [DPF03] suggest a decision-theoretic framework where simple and efficient perceptual metrics are evaluated on-the-fly to drive the selection of up-loaded textures’ resolution, aiming for the highest visual quality within a given budget. The approach requires off-line computation of texture masking properties, multiple render-ing passes and a frame buffer readback to obtain image-space information. As a result, the applicability of these per-ceptual metrics is somewhat limited.

More recently, the programmability and computational power of modern graphics hardware allow the execution and acceleration of more complex perceptual models like the Sarnoff VDM on GPUs [WM04,SS07], facilitating their use in interactive or even real-time settings.

Occlusion culling methods have some similarities with our approach, for example Hardly Visible Sets [ASVNB00] which use a geometric estimation of visibility to con-trol LOD, while more recently occlusion-query based es-timation of visibility has been used in conjunction with LODs [GBSF05]. In contrast, we usevisual maskingdue to partial occlusion and shadows; masking is more efficient for the case of partial occlusion, while shadows are not handled at all by occlusion culling.

### 3. Overview of the Method

To effectively integrate a perceptually-based metric of visual frequency and contrast masking into a programmable graph-ics hardware pipeline we proceed in two stages: a GPU-based perceptual rendering framework, which uses layers and predicts masking between objects, and an perceptually-based LOD control mechanism.

The goal of our perceptual framework is to allow the fast evaluation of contrast and spatial/frequency masking be-tween objects in a scene. To do this, we split the scene into layers, so that the masking due to objects in one layer can be evaluated with respect to objects in all other layers. This is achieved by appropriately combining layers and computing threshold maps for each resulting combination. Each such threshold map can then be used in the second stage to pro-vide perceptual control. One important feature of our frame-work is that all processing, i.e., layer rendering, combination and threshold map computation, takes place on the GPU, with no need for readback to the CPU. This results in a very efficient approach, well-adapted to the modern graphics pipeline.

The second stage is a LOD control algorithm which uses the perceptual framework. For every frame, and for each object in a given layer, we use the result of the perceptual framework to choose an appropriate LOD. To do this, we

first render a small number of objects at a high LOD and use the threshold maps on the GPU to perform an efficient per-ceptual comparison to the current LODs. We use occlusion queries to communicate the results of these comparisons to the CPU, since they constitute the most efficient communi-cation mechanism from the GPU to the CPU.

We validate the choices of our perceptual algorithm with a perceptual user study. In particular, the goal of our study is to determine whether decisions made by the LOD control algorithm correspond to what the user perceives.

### 4. GPU-Based Perceptual Rendering Framework

The goal of our perceptual rendering framework is to provide contrast and spatial masking information between objects in a scene. To perform an operation on a given object based on masking, such as controlling its LOD or some other render-ing property, we need to compute the influence ofthe rest of the scene onto this object. We need to exclude this object from consideration, since if we do not, it will mask itself, and it would be hard to appropriately control its own LOD (or any other parameter).

Our solution is to segment the scene intolayers. Layers are illustrated in Fig.2, left. To process a given layeri, we compute the combinationCi of all layersbut i (Fig. 2, mid-dle); the threshold mapTMi is then computed on the image of combinationCi (Fig. 2, right). Subsequently, objects con-tained in layeri can query the perceptual information of the combined layer threshold mapTMi .

Our perceptual rendering framework is executed entirely on the GPU, with no readback. It has three main steps: layer rendering, layer combination and threshold map computa-tion, which we describe next. Please see the description of threshold maps in Sect.2 for a brief explanation of their functionality, and also [RPG99] for details.

#### 4.1. Layer Generation and Rendering

Layers are generated at each frame based on the current viewpoint, and are updated every frame. For a given view-point, we create a set of separating planes perpendicular to the view direction. These planes are spaced exponentially with distance from the viewer. Objects are then uniquely as-signed to layers depending on the position of their centres.

The first step of each frame involves rendering the ob-jects of each layer into separate render targets. We also ren-der a separate “background” layer (see Fig.2). This back-ground/floor object is typically modelled separately from all the objects which constitute the detail of the scene. This is necessary, since if we rendered the objects of each layer without the background, or sliced the background to the lim-its of each layer, we would have artificial contrast effects which would interfere with our masking computation.

## We store depth with each layer in the alpha channel since

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOqDG5tCqx6mvTMEBWRGGC8EuLNFOV8r21mHaT1jeWfpt9IymGxni41pq9H9aLKblVNrNoxra-diV9IvLeXLtaRxrmLHie32aQkVE1Jg_o--N3b-u010CnWZUP3E1qq4KjRTJrMAcw=w860-h73-v0

9ab1f822-626c-4f3a-905e-3582779fbef3

https://lh3.googleusercontent.com/notebooklm/AG60hOoGew2Sh9WMzLCjvd0BkyTgA0OqWZGKUngVToPaM6I6jo2UJTB8QLxLoijIl6G3OExMiaKPctGSI77K5UGI9WJhSKM2mFSpyoDNYMtiWkdLkX3XgrH0_P0J4UZE45GDXDz8e0KyOg=w73-h1146-v0

d77ad9f8-922f-41b0-9cd5-c908e11d1d1f

https://lh3.googleusercontent.com/notebooklm/AG60hOoqUEHS3HujcFOTK7HsPDy14dXJvhulMuig7nPciyrKIAkoBlJTK1FMXRGkDN_rW6STaF8o1jZwvoo35dWowzmWoVYiFZTG1i9TOwOw0Y71_IEjIRbw1WqfCxpY23_z7wcSQ8Oluw=w74-h1146-v0

13a8ec8f-b1c8-4474-b458-2f954bf1a33f

https://lh3.googleusercontent.com/notebooklm/AG60hOq1vq-AxZ2PaqD6ifwflBw9E2znMgtvU7ni5sTimT-_AvS-ldMfiKIsIeu-eglwddn-47AvOc8RwyVCh_C6J1mTZ8nuk0zKkWizGbINcXjd2eDR9WM29bCjxsjRezUrg3JJCT0e=w717-h73-v0

26f7f99f-05a2-4ce5-9b69-c49f85d152a7

https://lh3.googleusercontent.com/notebooklm/AG60hOo51QPlCIo-rsGZ4aqmFpqksscfpqVdvPA72RRZPlqIbIffoWGkrfRxPQxP_v7KdaDlhq2LJo1XW4Bstmf4uA1iQcPsUqpKmLaQ89K4Hn3wWMHJUG30Kfyms24WmgkzPVy4lpbvQA=w18-h377-v0

1baa1385-46bd-4877-8d1b-dcb8cbc4c965

https://lh3.googleusercontent.com/notebooklm/AG60hOoKAcIpny8uSznPNvtUomFLqswiFHGXskzLQlhsqi8eBG8_34yIGHphTvXvOARXFs1P-2Zd7uu72uhVwpDb0aMyBR_G5oWf_gUq_uIYo-keZ6eG9RiNCiVLFPvz2NXreVArxF-09A=w244-h189-v0

17d02a94-a1a7-4bc8-95b0-43032284f58d

https://lh3.googleusercontent.com/notebooklm/AG60hOpQijkWglVzb8jcOFmdKhnx4pEje4u7X7OUWz1WnvtB_OZG7hGmyYiWfWtRWGHnxB2UHlnEAluHkqnt-XYU2bENrGe4pyB1iUR5O17IaygtWoRUsSD-38q0X1mUclBHAIelqpR4wA=w435-h281-v0

97c3704c-bafb-48b5-925a-72ebccef54b6

https://lh3.googleusercontent.com/notebooklm/AG60hOpPIpqQwRLRpnwXI6-vYcxTV6N9yRJvHOmfkdsPdm4k-V-O7O6-UUVzIlBWf9vLQv0l88UM_lOawEaZ7o8DUce22reh4JEN5i_TXFk4YQIxXvTNn3VmslaOfA63zDr4MvuKt0mI=w199-h134-v0

16aac982-2910-4212-9a6a-8cee7d3f2e1b

https://lh3.googleusercontent.com/notebooklm/AG60hOrymo--ncdpTWLLaysu7Sv6gKrBShvetF5ea3s8iRYFOHiXoc_vR2vKFz1NwRX703Nfu6OA56sq6EK0izMi_l1R8ex2YMbEQUjhX9JKbkKOXlW2uEA3WJy_RUWH-Kw0Z4iIFyRjuw=w135-h25-v0

d5a19afe-b2ff-48b1-9f7f-051f8214af60

https://lh3.googleusercontent.com/notebooklm/AG60hOr7jgnozvPePKpt-91bkTpaFxNfogGcDtqiWHkn4pM8BzsylCwpul2BrO1vnbfGKutBnWDrgIajSQNFwiBOu8txqE43Gz4rWUORUnpCAyKYTMHhkD32zvIS-JpmOD-paUJngyYw=w109-h22-v0

a0f2cdec-a721-4300-9c1f-fbe39a6d689f

https://lh3.googleusercontent.com/notebooklm/AG60hOrlhH8RS3biKY0xQem6z0_lkNR7tGrsfuWfH0RH1uvsUAMNzuIUBwvuJHM_SePdg5FgZStPlJQSYEKbsu4zbGH5GjoG65UBkrEte9k-7BIKvi1lu3f72WS4DPp5L0b_67R0YmdXXQ=w252-h170-v0

79a36ac1-5be2-4dfb-941c-e0e4ddb6ae6a

https://lh3.googleusercontent.com/notebooklm/AG60hOrJ1S4_14ZCxdI9JJBXWpaQvzqyq7T_L3r7Qk5vCZchbn1Pm3oz5gStlwBUrGeWQ0bcfehPy31nN9s34GMwdFzAoWEELewnD6wRt-uP3D-1Kv1StmzzfPrMJxh6mmWSMackk9RmNg=w188-h123-v0

491f334e-a478-4ab7-9bb9-0200d64933c3

https://lh3.googleusercontent.com/notebooklm/AG60hOo20MCGYMEhGvnHs6vk_eNsDtO3fTV0j2-9yzB-eu9YmByDHQoKGTS_eeACjnIcrfVFXM2Dor-YJbYg4zUK6agbZ6BcjZEi-CGs_OdnsqbiQQFzeyYMP-Bt0vgT6CRfkjIKUubVxA=w127-h20-v0

68a8efc5-a99b-40ae-a17b-c8eb80925417

https://lh3.googleusercontent.com/notebooklm/AG60hOppsNRtV2jZaD5W55dLgujxSu9OLu6WP3b-rv2BqiKGlEbrUsIBbttwZYxCotOzDcVsC1clCbC9JeCIvVZUjV6qWcN2atxuOe92uxdCFYhCbkpsuASPp3Vd5FdI19N2URxMBmxC2w=w26-h92-v0

9be89335-7057-4a5a-8414-f9d2856e68a6

https://lh3.googleusercontent.com/notebooklm/AG60hOoZP_Xk666arhuevwWdHxhrTD4Y7SnMbGObxUsdk-R8PxSHJuqpNrlg_rhsmskAfw3-m8FmPUFEpqb4VnRgqrr-NwqM_-wT3xVNxlZFza-e0z5zw-yc5gx59tf1IIOrQUmdwlMOBw=w41-h181-v0

c04f21f3-028f-4a7d-98e1-5bbc0c68552f

https://lh3.googleusercontent.com/notebooklm/AG60hOpcmC3v_POVQJwRcuioB1MYl0EEKV--ER65sMDdS0ZSs2MGa17KYR8RF1hTytP-NG0qdeMPO-hx22FalD7gRccnAbxHm3LKUJ5unobIHFWROouJiyKQDkB2QX3l32ocxf1_5IxCKA=w106-h181-v0

3c9579d2-86bf-4fbb-b359-65de48348203

https://lh3.googleusercontent.com/notebooklm/AG60hOp9GGc12GiOZGUpaln_xdnLbwqvuWb6QfpE8vc_Z7lOYZMFq64y71ZFieaCT9tzotrM5lkkWRjtcR8tLQGdFhIjAv5Ts3XMOXqMcvNIq28zOgxaf9wH3lo0EbX2pLH66vvN5H3JRQ=w68-h48-v0

0156c0bb-40a7-4d80-a8eb-a12b001f827a

https://lh3.googleusercontent.com/notebooklm/AG60hOrZCbQR5S5FBfwUAT7eSXKwmRJOBs-5v_Qc9E6K0eoN-syXr7JGZ6hVDrOOCyC3Z5BqqLUcedinVGD28SKXz__zl3fyx-qXAXoYopDM7j8JuWLoGIfO7RqjtB6xUByBNWL4r6Wh4w=w68-h48-v0

c31bb253-1e2f-44a9-9061-ffc9c0770110

https://lh3.googleusercontent.com/notebooklm/AG60hOokFyk1ynWFEpYWk3vI5zPETMN8JNDOmVMF1uneE5Ktn_aaPWcACEQBCKhY9Uo3rXW95B7T2IsX4yh_7kMjGdYg52FL9RJ_aQy8H1JwJa-M_rNNCFnGoQEDBu3nLaT-nqwsFKqL=w64-h181-v0

cd50304f-8639-4f7c-ae7b-cd27bbaaee92

https://lh3.googleusercontent.com/notebooklm/AG60hOoiDbt-cWjm2RYs2_6wS4q7etySAZ8LnIUvRpfs0TqkmQg_0X36d636FxzsMhkICQZVSk8Fthj4YCs87Gd6Lm4dwM-g_CiISZd8qpF_Ub05z-AeADRyj9U_9Xsr5H6lTflsOz1-_A=w13-h169-v0

286b842f-5b18-468a-92a3-fd7544d9e0bc

https://lh3.googleusercontent.com/notebooklm/AG60hOpg7kEm_m5ZGV_MoDir7TDsIkCGKo6zY0XpFsw13fDrH7VzG7LG0p6woe4g0R4KI1Km8y0zRC76aZpKvmX4Ta_ZfAR_oMkN8dy7y3a7J2XGzmwG3Ft3r74xDJ7d4BV1LikGOmSzBA=w52-h92-v0

5ed3de70-4349-4f15-bcfb-66d470b4126f

https://lh3.googleusercontent.com/notebooklm/AG60hOrh1PkRuiPXKJX1J2kio_qbfpPVbr-IRQeKi6PWUYNn31NhYob5gyN5sGl7AivdveTpwH3L55yVDhNtb6xIyiVEVHKbjELq2LkPYSQDgJo25xwlGFuNuBs0KEk6xGQADowlqGUYVQ=w42-h181-v0

f8c87c51-2028-4039-861b-1d18b403f576

https://lh3.googleusercontent.com/notebooklm/AG60hOqY-3gxZr_cN-OSYpSQ-WwK7rsjH6r1R9vjtEzC49B7dsfi7KhIUhi-jIh-rQ-XV9u6aCHtcPDF-vF2FuWU9aMWoN2MjMdg9Kl8rFkIYjbRJJfpoMIZwIRcKUBsGTA9w1PWZznS3w=w13-h92-v0

3bb8e8a2-f6cd-475d-aa54-7c0915e1408d

https://lh3.googleusercontent.com/notebooklm/AG60hOoukPjdhqM3OT3CVAkdssxCEZIGlJ7qUOdVjxFqqEwxUtfbS3bbGwQtn00J0obixvzWkvcakm3xlSR2VwQGoCt2o0gq6cLRze4t8FfMrJG2epRgJTvqqw3LAXd9nJhz1yqVSSvjNw=w60-h173-v0

5901760b-9b5f-477d-a034-3e1ed7bb7add

https://lh3.googleusercontent.com/notebooklm/AG60hOrvrIuC1MOMhrIjC8-qF4Ss2GvIwA2nk85YaGHHyAqBfx7KEpuLqHIlKRzC5O2B3QQSN67JSyBOkTNTh5dLjzOtvxX63PxEpGhY35tYuNZMyqoPsaRhf6ticM83EVqRb5FOD9_h=w58-h92-v0

dbf26ce6-2632-4468-9496-8b2e1a61ca3d

https://lh3.googleusercontent.com/notebooklm/AG60hOqRs2kDOjRexuo4pmcejofrBbRkwuLN1e6tTAueBiYc4LxxRFyo5qYbSTicCc42bm0LNZXxtVLFnv-X77coWG10MZAl8KZqMcIcjZ3hvQpEjLc6PsZcsE6n_4eXqXB77ap24FRWvA=w56-h96-v0

e68adbc6-33c1-4543-8e4b-748529f18768

https://lh3.googleusercontent.com/notebooklm/AG60hOpX2eGJN4nGubE4gtCx6SQL-yUncQ4Dt586A-dwauYvpU1iz4D9pXYLyHfxZqMe9XiCHuZo0anI7S71MppKTjRqmnPIT1GUl8mRfxi-6MXtSj5hHkZ-qBozWW5mFLCN4tWIrcCxXg=w61-h92-v0

452fbfa6-c6cd-4360-847a-5b68aa24adf7

https://lh3.googleusercontent.com/notebooklm/AG60hOpAWQiUJM78WK6rSuINgnZaG4Wsy_ZSbHOzUzI8o4eqRRqQJedHA4h2sZCvs3jirhwD5YdRs2RVm95WSzBPEZUa9cNeEoH21U4ywBSTotPjM4yC7dgVojL5PdpVOIwHt4kHDqIT=w514-h81-v0

d7dc35d8-d642-4d8d-8d17-dac49aad779f

https://lh3.googleusercontent.com/notebooklm/AG60hOp2rl7RacAzgOZ6qUBblocgacWsluIsPGAxTVBr7n6zi7eSrvrtB3dUjgv0QI51-kSDonhbpCT6xiLn12FVUVQgX46Sc5Ldp8cxcRAHgR28ldjwPVCvUB6a_Rnke6EeZ4IL4JdvTA=w255-h35-v0

7a9a71d3-6165-416d-b1a8-df790a80b73f

https://lh3.googleusercontent.com/notebooklm/AG60hOp1AwMRqGFvqp8dqSNv91DN7CoDzpa-6qAvlMJEs4Ix3WwJ01Z3xEdHGHvxOPrqstEzeePpmoz94YWEAtY9bWWax5OZ-j6Kb1WdDRbPUiXeF3ZQVJJFiqKbCi_hn_53vypwxGXv5g=w502-h35-v0

e34e5ccd-ac9d-4f59-827a-d844ad3b035a

https://lh3.googleusercontent.com/notebooklm/AG60hOr8sjGgzPdhT2rRUU5QBGv4OO-gcoCsrxgtKa6z6G7iO68-sbtol4yjLAp0jCgzlYMMec3W_Yum7wGN4i0mhjxjqMEpfDwmSQ8GjbWkNB4sAzh4u781Jjm6T3G2pRybg2fz8VMMzA=w221-h179-v0

632130f6-72cb-498a-abe3-5d2115fc153e

https://lh3.googleusercontent.com/notebooklm/AG60hOqmyYm7KbBLYnMAu016ktfLgeHnWVOuGmJyGqPJvhvJRyFLQs_4cFGSGaz4JzTX-zyvzxnxITJVfsDh-mOVsjuamlDuuGOkXloOUCjPT5kWMeyI0ZJr4ltJUKr1HiFFmJSoczAT=w439-h68-v0

a0d3628b-0517-4996-921b-2efdd6be6bc6

https://lh3.googleusercontent.com/notebooklm/AG60hOpyAvL5TW_ANyFPIzPDHPvxAunwbM2iy-GCEOblzy_XVBUIU9cXCgko9T3vrf0k8xU6dIpw1JJl-lFdhe7uxRfMPd3JXEDa9BgDg_jItoEw_vNJzmDNsQ6XO-yVbZYPHEdL12grag=w376-h19-v0

37613645-0fe2-4111-a179-7264c9267c31

https://lh3.googleusercontent.com/notebooklm/AG60hOp-_JGrFC9mcMRVKhgS1FRQ9keX9fhIIOHqQU8ARP1tDr9XKWygCpLF-TtRffG120weHRgsWJp2wJypeTn9yXILaC8-sIgcX7yxQx18UTiesJjVedBXUepWh0bnYhxgRAT2TgE5Ig=w377-h68-v0

8850e42b-b001-4aa5-937f-7bc22e693ba8

https://lh3.googleusercontent.com/notebooklm/AG60hOqIKq6f6jqOLsmtJPzbCdfwpMJhB1qN_QfBOJVrHN_zjKB1Zl_BT8oSof3FF9EzDkEaffaPuqKXOhvDk46avZ_deuTanTwkS7F6v9d1CTQ8LyILx9LpdF_xsfi5xDXJENMDTTx9LQ=w374-h17-v0

9e720f61-8e86-4417-8bb9-c2a605dd1600

https://lh3.googleusercontent.com/notebooklm/AG60hOrGCE9iHT_ga9t-KB7RvfiZg2EqbVq1uSg17cpYV4_1C7wx9hb0L-UW9ua0PKkJG5tzInxTZ3_UAJ9eGbMdlfR6W1kWS6NnpaRBSwHTNIeJTkFbdwIVIhtxgK6J7Rm12pntbx0U-w=w56-h156-v0

f911b30c-d78e-44f5-906b-8a5076b21c67

https://lh3.googleusercontent.com/notebooklm/AG60hOo4ezkxGNranJRcSrQyRiQzsFZXZiQUVP3vgvnYBibVaP7IlbT8Ct633OmEq70TCE41Jp8CzydkHfOwNcIPvUIyoE2kr6idpJ9j-l-9UCI11sryUzFFY8V0VrqovLpDUGirlz_0KA=w219-h208-v0

1eacf308-eb10-4b47-bf6a-dae81dd757d8

https://lh3.googleusercontent.com/notebooklm/AG60hOqH3ZDbbXpCYTEY2_JcSERvRFAbh9lezrFcMogkfGgQfIZwVLZhv0holrlRE3IulUz8dK1Jy9nPao1XnHZ5EnCdtZbIBD0y_Qt1q9QQL0CBOK1mDtgBLvuCDRgkiruhc5eMxSX9hA=w48-h189-v0

fc61be7a-aaf0-43ae-aa4a-9d91cb1fc9ea

https://lh3.googleusercontent.com/notebooklm/AG60hOo9wiOfCCjtAV36G1hhjAbv1q43BACogVxRukdABjZnRIhfngKVioMwb5HiHUkr3TNHE0bQOfta6ex2UdwgmeNIiiafnhZe7fqECFnUqc2hNirlG-Qc0qnm2mLUqFuMA5r9tdUpRQ=w20-h131-v0

1080f685-6a74-41cc-a899-2d9a4f8767a1

https://lh3.googleusercontent.com/notebooklm/AG60hOqF5R2CzYnbcRCGAp0P6GMZbCKzMcTiWjCs1_Aof7QAcEQWVygZS0GaRlg1qJm49VhQk_kPv8BYTCyxU2urAHSclyd2uDKSFYDeGTQTfBB-Nq2fgXkX2Q8_P5rc5IfnAb-fIv_hvA=w20-h82-v0

7f347698-cccb-4ca2-b291-4870c0d00cf2

https://lh3.googleusercontent.com/notebooklm/AG60hOpkvSpWtcYhoRrXI24ekkUfmQRUBdWT7x5mY2YVRfI8HEU4pXqJYRE74OjxNtDTM_oouj3zqcVg5qZj_6I67Qy49EKo6z-QUcvlgnqxXOGx6KTt_RvFP4eQAUDdYxxR7J7dwoWAng=w435-h52-v0

660d46dd-a618-4954-b1cc-227a1accdb98

https://lh3.googleusercontent.com/notebooklm/AG60hOo5EcakeOOI87muSvthk1du5eR1hPYVT7yzesXb5bZsV9ZGb_cd8mTBSyiD9u8Qh0bVmN96aBmoOy6-hyfTNjG4ZMgecJZ861u3hDbXwSrovydYVYhJl0VoJqxmyCYTZmKFvMP2TQ=w56-h27-v0

e0e3c48d-72c9-4528-ac00-e38cbff371a5

https://lh3.googleusercontent.com/notebooklm/AG60hOp4xDQgj0Y0JL8fJ4JXXtcQrGWCIUnuJ9fzL0BEL5yMCpKgB7VRo0oeReTwVPbl4o1GECjMEohUqqH_X0QjAvXIHHhAxsz74V4yCdb-1rtk04TZLJxdubXwIcQwoQCpeV6_0Oif=w4-h356-v0

d03f0244-40f5-49db-96bb-56c9eb58dc8e

https://lh3.googleusercontent.com/notebooklm/AG60hOoj_2xYfRnYn9sLr0hwXIIYbNekvTTGgwoGSmjmzs8RygttTOZFRlMduD7cWxOHAAK-KMchJGZQ_1CsrygOZ5XVcTlQX-O_xEdSG3ZtEkr_GRFsDLk3jpwQqhAPOSBXDOu5WfGxSw=w220-h179-v0

e419b8b5-2699-4ae4-85a3-014692606040

https://lh3.googleusercontent.com/notebooklm/AG60hOphFa6yy76T7zgi1eo43eIo6WYMH1Upr_V-NWg5eRzjDodOcIb0-oJAjQAgj1r5xcelIWM6iqSxTBD_sxWdgWaz-pK750jvB4J8WUsR2RG0WgQB9lkznbCwEW9loMCfSffivkd66Q=w437-h85-v0

5eec388d-1829-4db8-aded-71475b2edb85

https://lh3.googleusercontent.com/notebooklm/AG60hOpHlIMY38-9zvMWT5SHje9xYf5f79lKNsikJqXl4EmptpW4W3e8d1sEBbFnDPr9uEXXxfDhd4T9B-tVn9dSUr1TS-LQ_40gOyQzeCJ3e6JB46RMlGHKn8nir14QCWemqljnVaQRlA=w427-h35-v0

242cf233-b305-4dee-9b8f-469bb3e22f07

https://lh3.googleusercontent.com/notebooklm/AG60hOr2D7I-wnn-Ceh-9kE0eWfS1JaZ4ddpdYjROXmmfKZHgZ6NdsUzNxvgfnVwQE3ZVG-emZhndEV03hK8gOsuzGPxPIsYSFqZeS8Rwi7xLtVGG-fu0FiPv0qrCMMlWKNTB3_1YhHI4w=w387-h20-v0

52920699-2999-4c47-90c5-b42699ce8545

https://lh3.googleusercontent.com/notebooklm/AG60hOqVSBIhpKc2OHODMidZQRsSfzgilo4CrinjMrBYsynLFuMmTKHjSK9c2T0BWXtiUcF5mAP8GzYH79x2NK0lo2A7-u0ZRqacewGTo0jsxM4I0FvUgOcgmDGAYqpD6P_pl0--DUfNbw=w98-h14-v0

734d2a5c-20da-40d8-a63c-68489d3ddc7f

https://lh3.googleusercontent.com/notebooklm/AG60hOoCydRwhilusYtH0L6CUudG5iXp9fFJX2nusvcZ1nSXCxl-sKXcmdkn13Ilg5ZmcCFTtGR6uUuz2GZ9HS9WFcvilUKxXlIc40z1Lh35RezjA8MPNfOt-V8rj7VrCisWPwifjZXJEw=w39-h168-v0

db85795a-f119-4147-9322-782c9f8fcda0

https://lh3.googleusercontent.com/notebooklm/AG60hOqQU5KmV0Y0Pbn52R-Ntiif0Z6hI3eT52gP9977cJG9cQPtlZ97jJxBF7vtx5ZdSrz9CkTk6z2VO1WvrGkMvu5tgoTwxlWZP4G4j2uqOKOMMiL1MXGws-zDOQjjyRv2YgTQR32x=w17-h377-v0

e84aa1dc-4021-4563-a84b-6f58a78a0d8e

https://lh3.googleusercontent.com/notebooklm/AG60hOpWtM2sgbGY4Tsie9ywG6dezoK9QeDv814AH9pmgGkiYF6JR-d-X5XCdqTh3Y7azZYseyS6TVUlBDL_n_nQVtnP93-EGDfG9Vj42SlVfC6m_SdvnoUM7st6MVEhRoB89JizAnfixA=w245-h189-v0

59943163-bd38-4806-903c-2447f947088c

https://lh3.googleusercontent.com/notebooklm/AG60hOpVJJC3pcX4pROkFhL6i_66z26PwQEWDI3LMX3DgawILkMR3JdYIzWeplcQ8clwlDK-xDX8ApRMKPl1z7KG9QtbrE00qdq4jlsvdNqT7WRTqnk903yMrCb_qTSkFtv4KygPI85apQ=w35-h181-v0

3fe9225f-fdea-48b8-91bc-2ca3b0a5db6c

https://lh3.googleusercontent.com/notebooklm/AG60hOoxJTyy_zCFOEmOoHptzXZDUA4bhPoHhySLT0cx-BtoDgi0AFFEOVkYT2bDWwcdSJt9v5s8eVuFJB_Z4Y_Vlku-R4cZIwQpolockyi2pk71mZ9l1dzeorOFPD6pzd-0EqulXfNDfA=w127-h35-v0

e50abfba-14c3-433d-a016-a580cff75d78

https://lh3.googleusercontent.com/notebooklm/AG60hOq9HojAIiLIl0X1bLkBI5Raw6GZMRv0YNtqq60DfN-kskjN9GYiO0A8ZKPjlDHk2nRAmwHSwwdyxDh1GI2-Q8tKtqtGfK6c2CLRmZDKw2fOfAgemWc_1tcSImLQRodB5Rm8_iucCg=w435-h223-v0

38c80d0b-442e-42d7-9162-557cae1f1f90

https://lh3.googleusercontent.com/notebooklm/AG60hOrY6NkJGXu19FhuzK-gHkqZbo3Ei2rZjmu5xiiAG-6FAiMrVl_m0L5PIIMZlHeFJ7eSlYdASer4ylrehMfxnQUzu5rq_MbDIHYikLnntxyIYJH0rEefc0ZKRuSE2U3G5f6-51FJ=w15-h181-v0

345892e7-57e2-499b-8811-748e1d90e3b9

https://lh3.googleusercontent.com/notebooklm/AG60hOqRdpTH2UM5th8WrjBB1AtsB81nw_9yOt4Wwt9i2bq2evmSFYlRdwY-uz9B8parnd6PHoDj-YolFHC8dblTZNIaUf6x8S0GofoIalW4c5qzlmsn4U0WXOuWcuBNgArFiv2GQo62UQ=w200-h102-v0

21b10086-e08a-4901-9b6d-dc22670a00a5

https://lh3.googleusercontent.com/notebooklm/AG60hOplj9gBOltlFZYvzWEyO_NnI_DgQsUfQwDw9VjOLVuwbg6Szwh3pjLdfM72yKmqzyjFX74taWxcZ4ZpxOQn-Xh_5MqaETnlcffJbaueFCizGUGUY7y_HreRbcWJe2oJDg8UphKM=w219-h221-v0

08fb380a-263c-4178-a727-daf879797383

https://lh3.googleusercontent.com/notebooklm/AG60hOqHOu5G-ezPpPwTisr6_vSzEucDX_Z7vOiq726WXZV2nT7HpaIuCwAvsCrhHs20u_lDjY_Uuhmkd-ommgb6k2wnw8tmGLL37d5IfXcBQtNRc75QNXCgxcjJJLxAelP3DlCagiMwmQ=w60-h181-v0

1aaf5bfb-d8c9-487f-a111-2cd4fd9c3c93

https://lh3.googleusercontent.com/notebooklm/AG60hOqW0kSzS1MpPFg98TPB9rB5xMCXOM-v8G8nE2GtLz7wB4i6a04Yt_sdq_XL8eN1NH33OgRf7AtEt-6SI6QNVOL-TkHT_VgTJM6lOhYRXevtpmPtP5pOfzUrzSHsbkE-KixQrQaJyw=w256-h256-v0

01bc871a-4572-40cc-959d-41ff0798f71a

https://lh3.googleusercontent.com/notebooklm/AG60hOqop5lyuVaZxPV0hIq-xUBJ2ARwANfl2_RUXuiB8klSWUujQ1txBERh0L24X9NBFDxTqcCxKfbJhgI7G525jJnCR9HcwUXQ0fCxVEf_nlEXVh3FyWoIQ0QxVjtgbVKjj90gNXD4=w160-h102-v0

11aa5759-a7e6-422e-b0a9-647a7c714839

https://lh3.googleusercontent.com/notebooklm/AG60hOpmkZrTjaa_Vp2-Mo7p8-Gpx4eg5FA4oxPGbSQAZbMWO8hfk7NpH94X5N44qaARiazcV_eERVQgBgwj2u2NE-PBFrZWxrCgKVKsYQ7x-8EVa2MbM94Ahfr0DbZU3EgzpgBiNPN24A=w161-h52-v0

1dfd952a-de4a-4176-b729-99720b7b2a21

https://lh3.googleusercontent.com/notebooklm/AG60hOozWVEh4IgWRgh_vHCrxQ9UzCTTNy03r4pKBUb9610-3hbKoZcFUC6DVh5ElWInL3aw0Azg4WeR_tSDLju1pleyCvPVFiErR6UEqBE_ex2d3hZyLlpxErIsvfpAxnv78ucRHuMysg=w103-h67-v0

2e6b0178-4332-4180-980a-515934ca8932

https://lh3.googleusercontent.com/notebooklm/AG60hOqIWxdlDdz7W5DxTpce_ZM9PNbZh193NbPRoV89J6tzaWxWfhK_PijIwe2KVG3n9hYXit0m2v906sYQYExRDGRA93EwBazCG68Zgk1Br3GDVlgMc-pdJMXj0RNHAsbar--IkARVgw=w256-h256-v0

258dd008-650d-49be-aecb-a27c71ecbec4

https://lh3.googleusercontent.com/notebooklm/AG60hOoWKzlP52zZdaK3PV8MVYUOWOd28RTrpFkpQgRgH1CRKuLIXK9D3ReI1h8uQlTfMulojkP3igy0DvW8q1vjd_zENtV7iyRvGnzzx6mrIH08WyoaDzFKsm8S0LHcAfCj6ipx8wqV=w160-h141-v0

9403006f-878e-4ece-a2b3-282c36022d79

https://lh3.googleusercontent.com/notebooklm/AG60hOq3bbBXMPw1hnhpTYJxSG1SnwHryoXALs8QM2-7H47CuzbhQu46vMFZFbkFJ4CZPc3Ak13HkV5ZKX6h9e-fCdd8fgs9l1WeK11EY3pn6iTu7f4zt-eaqrLqKe5U7e2mmTOrkRr7_w=w64-h46-v0

0597ae2e-da21-4b08-84f7-70fd9fb7b2b8

https://lh3.googleusercontent.com/notebooklm/AG60hOrV25NLY0PTdjT7NwYmHRYfTjrTnAKZu_GIA-IsoNLvrXorHsbnomLFK4CAPpHG77mAZVGj3fierEfoKzk1sedjfoMuWZ7n8op9zJe8nLRc-qvMbb5JUuMvC5Tb9ryzUA4sdTL5=w98-h69-v0

0be44202-0c9b-4043-92ce-c32a219664ad

https://lh3.googleusercontent.com/notebooklm/AG60hOrWTaBOGUm4UAfZ_E-VX9QjqTvi3CVnv67DTyePhSU4sDXsq2nl0XwisyGiBsh0qCycKAC6tW88HJEyoUz16FSXlXuXbGeKpgv_cp3WIRAnATZjvKtPC9W5wuG6KT9UNa9xM-l6Fw=w221-h215-v0

f9ffc4b5-8705-4241-b7d8-1c55a8315250

https://lh3.googleusercontent.com/notebooklm/AG60hOq4OrJKYZE9x1nyWrJEWiClabzCA_SZhXGIGDVb05KJX425g50I2ywr0SUaY_bLHokK8m5ypy7fbJzZ-A-wjvyC_nhN8XmzoeJzkNfzsyUdzFJlY43uIPrubgcIfKiFVyOzEo5d=w37-h189-v0

25edd747-9a46-482e-9117-5cd8bef2b4e3

https://lh3.googleusercontent.com/notebooklm/AG60hOpEoST12BhcPhKTzgctzyjJEQwv6l2IcrGB3HDmZds7xizKSWCwLXgJgtxAKAIFii2upMN_RJ2uiskpujGftK8MZu4DBveGkhF8jXr1nnwNZE63bhmoSbDwNC8Y49ExsuWq1Sy8=w435-h44-v0

897dbcb1-f5d8-4284-a21d-302d3519d769

https://lh3.googleusercontent.com/notebooklm/AG60hOqByfWClsvxgZ5F3fdnQ-7N8HAr_QzedcN-PV50tgjyzHq4l8qF_wMOK3ZX-8eyWwZIpvJy_igtNngKLo4N8F9aHpdntuOTOeMvZzM4kobg44JTMf4cuqEQmFNqRIfixfVgKzoFpQ=w14-h439-v0

11b0151f-758a-4dfe-9485-a313454c3eec

https://lh3.googleusercontent.com/notebooklm/AG60hOrb_5hhnwSzZ3knIMR4T6784ZHt7GFind6yPWKTCqQKMnXomszzspmOmaotq9j-aQB0A9KBRXvJ1jIuH_J4CqGMQYYOBbzTVu_QzmPPE5kvHqaR7lC-6ZutL7GS-nSjg18N9ta9fw=w215-h221-v0

8cec7bdb-fa5e-4257-9a2a-213de1cb5d54

https://lh3.googleusercontent.com/notebooklm/AG60hOrB006H1vt40ibXLelgyGp6xZpMVtDz7Ryu2fQU8mGALOAyEtBIhKZX7d8BaL8tsQhM7JwrLO1RC8nnwL9X5stD4T6y9OdQ-IeBd2exI9OEZ7cMIxmfsGhTzI0y3C31-lFSq0gxYw=w37-h131-v0

1f436c98-844a-45f5-8145-7f95c71a876d

https://lh3.googleusercontent.com/notebooklm/AG60hOooiN1Xdp2_jyGtefZXd3cw6a2CBtlglYkls2oxFdVJVivqX4IyHerU0YUxA-SGIwOYfhMqI8BQYfHEe6eEb1pKvO7Rx2jrFAvDKd8AKRZjnFSA2nnUvrUJkhGBM2dRZc17TTPw1Q=w8-h331-v0

d0012e21-d996-4fbc-8b9e-d78de5c1a00a

https://lh3.googleusercontent.com/notebooklm/AG60hOpV28MHX2p4GI6cBIMiWNR3SN4aXSHL2scABHLvvYfHoKmjQrossvQ5BbNEA4AleHdpBZ7xMcoqKZ7kbR2ERofXC8pF_e3wUcEwAW3lTdaRkY5FBXFah0nELiHoF-VRKVH8gN-a=w218-h201-v0

cb18896c-456a-4500-bda5-02ff7c6ebd2c

https://lh3.googleusercontent.com/notebooklm/AG60hOopE3QXLjUEF1mtGj6SCKTdok1RAXWv5OmFUYZrmFqi27XHjjz3Yxe6ebuK6wYmpQL31ZGmfg_S5wD8ufHWqRFlJ_nI_Go2OfRfBVqiMT72h1xT12H6jp1uEBhwutjXXeyDAjL5zg=w433-h39-v0

5146b6cf-ac2f-405f-ab9d-ccc53d03fb13

https://lh3.googleusercontent.com/notebooklm/AG60hOrH2dGU9hRHIU59le1me0scJveUa5D1s7jSJ-U3y9iS16gKl4gsuqHjNt6krdavA3t3u47A0H5sgBnl8YDwUrRDPwzRDylRfz7aBHHiIdh4s9mvqzVcXLUVVkTeVWoFa-3gs-_dMw=w435-h29-v0

ca1f9796-8d41-4054-9fda-977d1fa63644

https://lh3.googleusercontent.com/notebooklm/AG60hOo32b6Xe6MH6f7nXosuzoRYLx3Smkui_OjiDgwa_DKpyJFZf_GzQUWLF1wo1Rs08jM-NE001ztZ6q1ljekoIxJkwrEzbH3JyLxYpz_VvMD9PRWQOMuiou1qcf_ybPd4ELYwHsvh=w219-h206-v0

24185c2b-c2d0-4c05-b687-2791ed9b02ef

https://lh3.googleusercontent.com/notebooklm/AG60hOptjuiF0PtQAomVj3gzLWU7ArihNYx-F_ZNsqE76XKN2SjeRk1vBa3ol3I1b3QSBFbFAYm9lT_RYeN2jA_7PBNMy32GVXmUoegKEY2Cz9jwTWrePRBBLJyHqWrbhIwbavMnvhJ32A=w339-h17-v0

a3ecc168-542a-491c-a13a-9406e6d5ff10

https://lh3.googleusercontent.com/notebooklm/AG60hOoH3StsS_Q01ZCX5kUWUg_hEQikCxcbb5LD6B4vazja0yu-Vb_53CC7UixT36k3n-a4zo5BM0j3oCJ0OWueoxRNMUcSC2IUbWgXpwNmLwNQOFdjvVSGUIhYdQnBAA0KLdI9lbvo=w160-h17-v0

1947b673-41c0-4622-9355-4417c7fde1ac

https://lh3.googleusercontent.com/notebooklm/AG60hOqponEL_midecr3Mt34As-nSzdgJTM2snkhNseTUxxP53hI6D3_MPy9DYbDDxsJe6oECV_4doGqLpUQelM1-FOfzm-qLYNPebkY4mnlDuambJfLH3KLDCCZqsY-NEpWNJpbTsJKMQ=w435-h24-v0

6e3c8ec4-942f-4769-aa90-f51706f99c56

https://lh3.googleusercontent.com/notebooklm/AG60hOqfXXsgy5zWlD6EBw_2R2QlezO9-SUruZWge-6gRgYfKVzgxtc90i9zXRsGtexvLM70nb44Dixc2w4uh1nkOEXJjzmbcveqslWliGvGWaahKueGpUtnLzjxMVfCQy9sPCFnHjwS=w219-h208-v0

af270c61-9475-4bae-af54-8e2b0143ff56

https://lh3.googleusercontent.com/notebooklm/AG60hOpOBSb-UhPelwYHGVQ6iXdxiyAB2fQ8dS5-s0YA6O28QK9IlTwwyQqPq_e-3D1L6ekZEGTSXTvw-hiUbvT4yrSxLLZdsm8NSt8AIbGzTWQrfYmxAjTHm3gsdlt2jzGAA391tWIKYw=w10-h435-v0

2225a6b9-7d7d-4ebc-b5ae-29e36cdf7e73

https://lh3.googleusercontent.com/notebooklm/AG60hOo8rJDCCPcTr9QYcRvV_L4B787dJY4r63kjI3VywWp-zBxedm-zU4uui0hXlseZobAiY8c31p6CruzU9DXVnBnQUG8k8jEEsvRkhWdOBQ8-TB5vjPXJ6rkcw9WkVWc9EGSZNBDYQQ=w215-h186-v0

10aeed14-7597-4809-a72a-259ae81d904a

https://lh3.googleusercontent.com/notebooklm/AG60hOri2aKk4xGBq-j8R0ffkPi1Jy2oytYr8a_qQBIviwVem5X-9yEfL2muOb20TEjrdBkPowH9RGH-qJsrDxinHyX9XixzTbMRiqbJqIFxbkQKpkLjh3jRRZtB2jx2NbLaSg-UmRABBQ=w31-h184-v0

7b2d4224-691d-40d9-8af8-c09c2849d3a6

https://lh3.googleusercontent.com/notebooklm/AG60hOqAORo0sSsY24DHgR6vRd6og2kFaJsT68SXSegSiqUjWUPZFPAgreh2UqoaS6KrQqP3B1z5IvAs1tTu6xwXVEaesSII6TnTHgtObVFhUHHWhAiZKHx9Z8rv44LqrLlz9ejCDyKQFA=w60-h73-v0

15ccf743-f063-4625-9a54-2f607f5d507f

https://lh3.googleusercontent.com/notebooklm/AG60hOpbL_GojrSSz4uCEjvf8dVKPgMgVLdWjwbo6KDc_zv6kpHo6O90SuZV6zSYKbeTEWJRpzLe8lysyZJdt8cpjobor69aWiz2o_w2VeDU3Y3QXJKCmErNda0HXLRQpKujhdvYX9Ul=w10-h368-v0

31835943-645a-4558-b59c-24faba435eea

https://lh3.googleusercontent.com/notebooklm/AG60hOpptGIA-q6w6hR2xjp22FT9VF2X3PCRB9yytl4yre-utSNniJ90pAIFnFznpr4kLDqpkYXDO_1ziGXmiwyUCY2TVqhgEXvPqrzerYA-OJ-zTYq6UTYx9pWuS3aMV7v5eYIpnn7zYg=w17-h151-v0

5c5e34b8-b1be-4c9b-b71f-fd0814ddd4a1

https://lh3.googleusercontent.com/notebooklm/AG60hOoGmRTSf6_-pv3C99wlEu4ytB_RRDE6o3kuGTRh_jAL8Z_Om_1FKlWHJSl3o0fOB8J4JJR6fkHO3RpoReGKpYK8tc_m7Wv5jDO4ulmHrrXjY291_YIcAuODMrkSdr1KIDXJ_LJR=w171-h73-v0

a7c592f2-35aa-4448-9e84-5a5edc4eb654

https://lh3.googleusercontent.com/notebooklm/AG60hOrKUnLx3PdD1V1wGy-IDzpgcAVDfB7W7CnjDiTZyvs8pED9gWrqFhkpyrXVTYafCcqjHy8Nj_souWYAOniHKLT6VemapCF9c-wD1VC0nOQS_hcRM8USoYj8ZWm2LJZv0jWl5_LnPQ=w60-h69-v0

23660e31-3ca1-4d5d-bc32-87ac1f8eced5

https://lh3.googleusercontent.com/notebooklm/AG60hOrh6Qy3rvot6pGferhahfUymBQFm3W17413Kjrjhn6Z6u7-TFpMGn8pMW48_SB7sKlIxlV1VHORRrt93V8h5jl3ZOZVACBnEFUlQTA-5HHEqVwGPoRQmuOgOGtOeZG7_rhVktgTeA=w171-h69-v0

18b54970-3400-4a66-8f25-180c0be37425

https://lh3.googleusercontent.com/notebooklm/AG60hOo4YoDc3Ffi1GQOIJdfLaoqjZxYPDfCLAFC_2jjseMWmgE4cR1oBMToaw6I17pMUkyRfC-qq6MWnW3QUZrCif8keCgwOo5iMKJ05LUTdV_v9MtpWXODbDjZ8twYjP4NEDXEAgOU8w=w256-h256-v0

14de42c3-e648-456d-bbee-0a5ae4ce2858

https://lh3.googleusercontent.com/notebooklm/AG60hOrf2bfSQfJQjdAyiss0wOW37gV2hiN1k-DeU71UlimgIx3pYZ2U9N_kQ2B873qwPJjStN-iwjEz1CrGQDPX8PUpK0IDsr-jPhjvg8yP47VDEEZdpkPfWEF4wsxjFXNiIEnfKPmUxg=w256-h256-v0

3458ed1e-9dac-4741-bcee-fe04f25817a1

https://lh3.googleusercontent.com/notebooklm/AG60hOp2fmC21NBeJPY9m8dQi9alv9xMI7MyJ4hCxuHRDLmL0r-V-kjKJ31_AOU6_xPantglBOXpuGEbTkf_4XxNv6f5jpftrUbhlt40dKbouI3NsDKNmj3-tbVFsrGRAYcg7iyDF5FTkA=w53-h98-v0

5bdacd4a-b8a4-4077-94ee-7195a5ba1c03

https://lh3.googleusercontent.com/notebooklm/AG60hOqPh1pudPXXl6lmPHTr7J4-OCTQZGPGzr5HAQmADvGs9K282SpDo9pPd077cw5kJWJU9h28HQobLafkGXG90780B6vQWDrUAqDt75vZr6edo-4s3o0OchtKNRZ8NJ7yRgGlCE-GQQ=w252-h65-v0

78aad06e-1139-4faa-b13e-d7163aa3ce2b

https://lh3.googleusercontent.com/notebooklm/AG60hOpgOIVzRm8O27Lo2SxYIpVFh4-XV5hZWLp6R9fl7WZ1mXWeOFUrenDzdhaJ3WfxsPmONmpLcRwixpfh7wv1O13YHqU--oMCYafEIf0AAgJTJ3lRcKTlWibX4wSwagz41LsjtYIpEg=w152-h77-v0

771dadbc-b059-4555-8bfa-fb059e71edf8

https://lh3.googleusercontent.com/notebooklm/AG60hOrheWtt5YcOSUiNZI9uPYAG_rwMa3DtbbUODX-46EtpLtwfSuMY4oNFm9gRM7ddcCFMOT4NV3rcd6Or-awHd9mj6j7aoDIfNNwUCvA35qMZ-xLm1pib7wn-odTBfk92uJR8pejHEA=w51-h106-v0

2838b4bc-0c07-4d10-a198-c0b7fa95a5eb

https://lh3.googleusercontent.com/notebooklm/AG60hOpd7fd_faT8jBhku3jwRCq_BFuETR9Lgp0b778MGWW6uh6bMiZFr_j1ZHU7gqZS-gog90fFk7VPn26LFOECL6SQsl22Nxxdjo8Hg4N-gQvGBLyIZg7zfzGa4n3OA84Z0wV_cOtALw=w252-h60-v0

518cd4a2-132c-4746-af1b-390108f4c9e6

https://lh3.googleusercontent.com/notebooklm/AG60hOr7Nu4aE8vwTou0myGsHHSNW92HeW3W9UzvkBTvucu4JrMbGVuJNauU0iqyp30nEy5r05q9KhbSd6H94kHOhiNMROlaWcjZsNwoAb3PD1Y6R71ys2N_4Tab2sHm2OgusluVhy43Xw=w155-h87-v0

d3f36088-b03f-4f33-b025-70d327c85468

https://lh3.googleusercontent.com/notebooklm/AG60hOp5vaNvQDpFLH_iNEug_5IIFcoWEWuQ9hWmufcfGsNDq_p-vpF-PzON66OwCwSgXZRDPrVAU8uWEshiXovHHGZqkXyXvx5iT5F5mFCMrFgw7a9xioOuFQ50q1qIaLOSGJOk-KQuhg=w152-h19-v0

1e246e31-5d77-4278-8270-ad35f18021ed

https://lh3.googleusercontent.com/notebooklm/AG60hOow2RH1-ZoTPJ_NqyL58MPVy9wsoiI7V4cgZbwBag6OxHXTJYX2DLEYsVxb-VL6mOSNRYLotIgAgOP4pEItlUTeiNVmHCxCogiR-Ar2R12b-8E2E_8fyVeAH6yEXinvfHsAFGkG=w256-h256-v0

a8e1e75d-6ac5-4d6d-a98e-2257304c7ee1

https://lh3.googleusercontent.com/notebooklm/AG60hOoapJJXr8mJnpln6A7dosnQ83ygLzV63BlEqDstFXUDYo-Kh4NrYx38r2gnyDGz3ae85CewQjwkXi38yAOL7p9rNEe5GWC31ar9gJH8Y193_GyrNbTlZSGQdyt_7jY-L9VOn6GC=w252-h164-v0

e5431e41-d59d-4ca6-90a1-643af9068291

https://lh3.googleusercontent.com/notebooklm/AG60hOqAUbGSmR6rVbBJ068QL9yyL0s2EX5hBV3mfpxvbkDItzEgkN_K_bG8zE6ckh2H8IGw7FmhAWc7oYmHsMdYOvsRpa-vAy4fijVEJHnrxaCMvxwcwHftqO2DOBwFU8NUCV_9n374=w152-h88-v0

23a92985-59c6-4796-8964-e71e55776c35

https://lh3.googleusercontent.com/notebooklm/AG60hOoqdRE0j3Tj2_kUayAkicVGJBi5Vr3urDc1wMxktPI10i_uBG3RqU_velOni5x_dOaFxPnidxgnaaNMoszZkc6yAsnHJVumh6YBrm0c4JSTrMFrtY-fZv0_r7F59YhMCMB2QhmLPA=w256-h256-v0

d0e8ac9e-df2f-4532-afb3-48acfc82696b

https://lh3.googleusercontent.com/notebooklm/AG60hOr7Hsk9ENLpUx1Kw4pqBw9LHsG27ouecwp7c0AjkpOocR2pvbs71xzDLwMS8Co97zxDKQP00cvDhSOKbwH-Kfjxtt8wBs6eoivOENVjMb0nyi00BlPLLO387kC2CLzwoUQnEFKk=w256-h9-v0

a99d1b81-96cf-4bb2-9063-81058f4ddd42

https://lh3.googleusercontent.com/notebooklm/AG60hOp-S-K_fy7d9DMuEsuxfZPzOp6LMrPihZDUEHxQcwQ949pzX6Iibq7m8hMQxq87WMS5wc51aka5Zjhw8IA6vaMmwE-I5t1sAt17vHJTLPUOoMv3ip_y0k_I5XA6hZql3T9oYSlgvQ=w256-h152-v0

cab0aa9e-7734-4987-a3b5-e1772e842fc2

https://lh3.googleusercontent.com/notebooklm/AG60hOrCMTe8WXzikHPl_9ARpjkwW2Z8N7F5N1mmZk3Maw03sFqq_BBQ0NZlPFXcNBh7eafy9jpIIkfdU7WGnfz4cJwHPObLme4rVoRTrZYGEqbWWVDfZ489knxpbFMSmwDFAd_24f8owQ=w161-h88-v0

25d84297-e02a-4da3-ad98-d15c7363c967

https://lh3.googleusercontent.com/notebooklm/AG60hOqqWOvTDMVyE3FsgXdENqaG3GTtug6n6duRClQojZAevAowqItRh9VoaN69fhwsSCowk_ogkfuIfgh-45HKBHitg4gwmgmddatnQkUMf0tdlis3s0TUe35YUerZc3WJj3TddJAN3g=w439-h31-v0

603a142a-4c4e-4000-bc6f-164f87a40908

https://lh3.googleusercontent.com/notebooklm/AG60hOowQOxGXMjbkPvFML3xrOmFF7S7USRiqINviYyStMhOxnZRo6CkmYrq5qzb9TuUXd0dUXZbrBLHZw7NapxPdSSUNhHdIBh7ea0vNG0bsOSutjtkLXNzJUVL4uKYWb2YATbYzUOomw=w221-h207-v0

e4fe7fd5-3c10-4b1d-82ff-5ca9f6fc37ea

https://lh3.googleusercontent.com/notebooklm/AG60hOqppiHgmlzt0JKo-X-YuJPMQ5ck6JmrYzg63ID4X0njY0VFZayEWtSgAVQWov7jOnrHO9NLzdvaYCPXpikKxwv5veCybnZ2mKm9uElvAngBtb_VHn1qfXHYtWuh5kggaXQ9wq6Jvg=w87-h19-v0

73c4b7b8-df0b-483d-bdc6-abf34391aa7b

https://lh3.googleusercontent.com/notebooklm/AG60hOot7rKCJQtxQkhewc4iU-o0kLB4JS1eP_10kY4t6zxxaGDvlBeHimfqa7NLEDfs_C3T60PNZUjEO4iBcf628znzESCvwmnCyoiER0LD95DhAZWL3kO8BM0oRkcR02hoBFvWAhOWLQ=w256-h256-v0

534c0359-bf7f-48e3-be2f-31b66a125893

https://lh3.googleusercontent.com/notebooklm/AG60hOoq9HxHWj2-U17QtXCpGhPmmjHcKa8mKk-Qi1IX6HCIhGetAVHUi4ua4NdiAC4yPQVOvgxJVwtb_hsvnZWDZNfh3kuRQZSAXTococ11rp0FSD4Ksicw94ivTvV-lN0xaGZDr3yr=w256-h256-v0

d004d840-aafd-4c2b-8d92-47fd0cfd4cb3

https://lh3.googleusercontent.com/notebooklm/AG60hOp8bgvxqCxgizHiOviEFNbFNw7XCAiEnf9MkbGZXqbNzmhZXtRDr5zw75Fdo3uIvb8JPmneSox_nR-aQLYE7rW-3UbVe_KsRP2FsVsHXhd8s9eXDstyCLyq0KgPv8b6t6ywdLLsbw=w277-h19-v0

aeb31f2d-ba57-44bf-ae46-7b110c53c858

https://lh3.googleusercontent.com/notebooklm/AG60hOqwEkiklBmpfqYXK6eqYh5as6VjAAhDYNVVpHOTs3kI1oYv885-7NwujtiLHC1gbqnTcb_bFNCOIk-mFmbUMaipopLqonDjZmg5oKga-BC5u4RkeWX2_Vk9KIi-YKm74VvvDwC3=w277-h144-v0

69e2630e-d71f-4738-875d-1f38310756d4

https://lh3.googleusercontent.com/notebooklm/AG60hOrjfigOrJje_rtfVqN7Kl0UrxhIJIw7uIvAc_wtpw0XjzrN6vziY_7T-PGYcbplpPx3sjDiGPOoETEc5OIPWWRf9NJD5WtFenfK6-z-RsS4_SYmZnF4nRjAIpo5FO0ObAqfY8A-TQ=w194-h82-v0

26da55e7-f6b5-4a12-921f-2f54b70aad9f

https://lh3.googleusercontent.com/notebooklm/AG60hOr9DX7UfIPpA2gBwfmrlbtvWaPZjy1U4Xe7p45NMuWyiIX8LRSfiVMYEVGopxdHZ_b1VojV8gJaVwf4nDIBIdYcu5GZNYb8uYT00j63gvfqnmwZSxbx5ne7g_EboXxE_XWYVcYyyA=w235-h10-v0

c00486f5-b6a8-480a-9879-c85e84e16cf5

https://lh3.googleusercontent.com/notebooklm/AG60hOrZlxEIUaF8tdQpssGDHNrSQFUul4KTnZdtv-bOkvsb-8qKJzjnaFpCEaVr11Mi8LLK1DMUtO_hmyO8MHeMqLwcpclgQTU5FGq12toDP7L10oTNKdI2kAgnKosK1AO_DQSE0ijg3g=w235-h211-v0

09376e88-8972-4ef5-bbf0-878a39355cd9

https://lh3.googleusercontent.com/notebooklm/AG60hOosQ4g8GaYugHiMBREFS9PPTyvTbQTTgj-NtuCl2j4LXN6iQxswmrk6ILaic8L-CxOsBby-t6ma6G2gJWCTLssCSNj13vPybuLZp2TuN5VENOfxM95qhjXQz8QJIMH_mUIXABqFWQ=w171-h134-v0

f024e51d-7bb3-4b3b-a67f-e8c94719cdee

https://lh3.googleusercontent.com/notebooklm/AG60hOpSGcDBQE6lBt6Xm0eTj5CZLkjQTuy1428yPxH5Fst7F8TBj0M1b5z4tiGeQEY4xkbolgr_SSYNSEcnSDSvexDC__9YGlb1enGBS4iv_BvVmNKdcnHJPz9GzW_VJMjgy_xdYftaYQ=w256-h256-v0

6f7514fe-ab49-4eff-9938-a4584cf68cfb

https://lh3.googleusercontent.com/notebooklm/AG60hOonS38Bb2ju7rl8i8lF3mmy9YDP1XaQU3gwHvzReOPdEZOzwHjs2Wm3b5IvD58-WLrsa9YN7yezfSGsKtmyZu6R0w-RCeGmKdv4FL-ntSm1xrW3ypaiPl1Zg0h1S106vxFjORWaNw=w256-h256-v0

c1c99bfd-cef7-436f-afd5-245fc5cf6f49

https://lh3.googleusercontent.com/notebooklm/AG60hOoTVFWjuY0NCHUvT8SJtWxkUwAh8HRlz-q3B2bfRuROypWA3dZeLxVx7SbnqGbc420bNqKq6soZYTc3Lhoxv2y5kx26LqUi1xZYiCijnB68FB5_HPWUJS9xZwFNVWfv2zG6_skCNQ=w256-h256-v0

74f1841c-f658-42b0-9354-accc7d3f64dd

https://lh3.googleusercontent.com/notebooklm/AG60hOqXTgNtTwillD-_nMZswZ1fOtjWB7bpVo54KOhIv6iIPNJBrAtgE0rkmu11DCzhMLaM4GkNrK3i_5tctItqFlPof5nEXyUeULyhKdoDT0PQpaoC7JjzGiQoTktbOpSMgknX4ErPJw=w202-h26-v0

381b827d-6759-4b46-9925-4ef815923bbb

https://lh3.googleusercontent.com/notebooklm/AG60hOoIJBFOdGb0svTwS-7oOfTFdohJ8RdBhmFAr-kn5pQclB9NYyyKl8Kven_oUIiFNiiTQukI8lkRdFGOfx7FxV_u3mzZPSyidixeSxPZEc7EyPaLqgnApjyrhse0fdev6w2tnv6-ew=w98-h114-v0

5ef9bb88-2a54-4ce5-9302-313615db38e5

https://lh3.googleusercontent.com/notebooklm/AG60hOrXQrA56sS5aBeD3EPRgzoEEAD6u8E42Y5R4k6ROKMtPGDLmEaw8dFSrsO47xgY6m_Rqfg1Ph0QYVavM4-gFenBCakcdortO8ALyBAy9uT24RNFHRD3BdUWnZQlfROz-i3zeILC=w298-h44-v0

646ec4e8-d42c-4671-b8d8-c00ebb4bd7e7

https://lh3.googleusercontent.com/notebooklm/AG60hOrxeVntSJZbgXtNbzEgJAST4ARGNeMjtRLvBgf5R2pyYKfbqp3OZxE_FdvBE6uzG1NZ-Eb_6ZitEbfcnHM9fdx0TekLskTEue0VQq-HDSqkf3-x6G-WWNLHJl-gz2pfLeG3iA-JOg=w98-h15-v0

2a47c789-9315-446c-ad4c-edb0bf614031

https://lh3.googleusercontent.com/notebooklm/AG60hOrP8UrsEJtyAJ1hVKlLOeBCYpS-FJ_tpPVMEzoOKOCR-YzgCuPxRyDGI058E22eSwdQEE5ZlaVzGKca9OWcHx7-n7YJ7VGCcd-li_30KsEzwJSouCCtysAV5OmfrSMXP8Zeuinz=w203-h69-v0

e9474054-0b3a-4bb0-aa29-4f02ea28e49b

https://lh3.googleusercontent.com/notebooklm/AG60hOpzORHvYDT-lxkPZGIL9EpTGZ168B5NppBT-dZ_lBFhvNs22H7pNeaIW9dNvxabnqqxHAZcKSwsBCh2vlAS4zz1XCVPB7QM078TXK5BBiK-Tgz3ORPsVcBQYoz5ykUtbTSyyEXVmA=w203-h17-v0

da2674a8-41bd-4a80-8411-92b3ae5f57f4

https://lh3.googleusercontent.com/notebooklm/AG60hOqQ14EdntdCuhNO_M4UMtwM92n2tXuVS9hvwW2nLHEY5QdmMibRsDB9FqQoKnIfQna2DWUE7jbiJdaEdZxskt5ZQo2PVXetX9h-9NaFOtNd9VNb3ZsE_klghA_mC_yqWJaIYlN97Q=w287-h110-v0

358b3bd9-6ee8-47f3-907d-5db578ef48f5

https://lh3.googleusercontent.com/notebooklm/AG60hOrdxkNiT2hPd0orHL3rRD2oYkuEYNUViqEZnu2LIKl3XF1UttES16gPGgxicPeMgo-HvgTNAhsCgoGrgaloiwNj5CoIp5u6j_VHjxkhZWaPFLLR_Io391N5gNFULO-ev47qegLCVw=w37-h52-v0

708ff923-e1b8-41c6-bf52-1f758b436321

https://lh3.googleusercontent.com/notebooklm/AG60hOocDuVyoi3AocINgi5Bodb6WAIujy6-7SCvxtEfcFOzbDIRt_OE9MqTwX8ZcEVNAl-ZoNREsSP_H7Wi8xCHNIze4366nAaHmxffdJfAmNvChDSxrpWA-9c9DUcdLL22OxSqZpw1=w235-h19-v0

72929049-b599-4846-99e1-c4c80f313fb6

https://lh3.googleusercontent.com/notebooklm/AG60hOoB_-DsUptXbeoxMHUJcPnZr4ppqLp61-SmdvPkBOvRoq2MHZ7Q55AW58GrspG8fqFabh0Ie-BxqCICnE4xHd0FUF5qHMa1ypkD10A_sfyi-tv4tgbbZJNnlgFyGV3PPKsHU2jfzQ=w189-h74-v0

0837a848-d09b-417c-81f9-44d8b48fadf3

https://lh3.googleusercontent.com/notebooklm/AG60hOpQUBFU7gDIWCsMjXjDaPu8bMgquQIVgRn6vJD6XKd-O7UnOKVpz7c3bc3zz6mBheVpW268YcaiGB2BaMSDqpTgbLvougXUtFxGpax7RZDja7nFtsEZJ-ztWe9toYq7_zdMSe3wxw=w289-h98-v0

aafcb40a-9bdd-4574-82a2-da48ab54cf0d

https://lh3.googleusercontent.com/notebooklm/AG60hOqlGmshiff9T8zI-7Trz6FpumAZkotKUNL2ZrXG9V3iBcWDMvngPqLLdL7NvU9hF6Rxw_s7rGUWYVwYsOXbxjzTYB6ZPz4uEeenTcmVC0nFzYlHFYHw_lRP4uQfnmyv9mLVjymfug=w29-h65-v0

287eb535-c8ce-4b85-b0fb-20d9d909f3aa

https://lh3.googleusercontent.com/notebooklm/AG60hOoS85cybY0RdC-atu8D_4Y2H9zHCbfROdpG4-oUGV3q7vwFajzJ8E_rPeYuuWkOih2dKjNgkpa2EbYSzLZsiN2AGBEjp_J3d8g4T1pVuwIyqSB92ujI0-dGtoPSPvvnpmteXeQ5=w197-h81-v0

3b879d22-a3e2-40df-9fff-f554cd66cba9

https://lh3.googleusercontent.com/notebooklm/AG60hOoKRwpkHzisQBz7PrkhPBvrMwXLBILsH6aG2iy71LPtStOG7uNM2UhZmFCR_CgQsKZN3OwPLUQj4DYqxOg6mum4Qap15CajABovyOuZlUG_1ypOrZuBQvKRg_lEEozfWurKmaecOA=w256-h256-v0

61c4e797-aed4-4188-8065-18525dac0156

https://lh3.googleusercontent.com/notebooklm/AG60hOqPNMVzmbhmCLij8WBcK1Q1T-Xu8MI9PpuQzwHWGVZvrR9AA6Pmn3wTkpZ5Q81f9GUi3SXd8YpWcgx21iLFHTaWI87lI61IKh-jAkE9ZGEcwuU7Gx7jrZWHAK1-LTE4oUZeY5_4IQ=w122-h67-v0

ebb2608f-4954-4a8a-893a-5e85dcee7186

https://lh3.googleusercontent.com/notebooklm/AG60hOp9zN4X376cfb7UCJBHAV2eShYSloUPHToa_U_dDMr95X4tLhr9cuKj_IEV96z3Idb8BFvEJHFTjE_C3ZcLfb3U9UKNi3t2MpKj_w_eYXW1QiL3g77FLOJcW0xy6KNfaO7AIEun6A=w30-h160-v0

458a1a63-ba45-45a8-a826-bf53d67ecae2

https://lh3.googleusercontent.com/notebooklm/AG60hOqZdaK1dF5Xsridbtd0HGoY3_CTFvzesDhtd57BeC4Vafewid8w6_OCtKfFQ7XHw_xotoCndDZgAWU3Q4SZIhUNFFkcznUdspkGRuuLV8EywtyfPiAnsolTqzdqQ94qbalM_RHeNg=w19-h62-v0

5ca3b1ff-0798-4151-83dd-4dbc45e37da1

https://lh3.googleusercontent.com/notebooklm/AG60hOpIl8lVFFP0FPi1gdsc3eiTsODJufDsyQC2ODYzMJYfw2RmEZmC-zeLq6KUnNe1Dpvo45WNtx9Pd13U-uYRopz_CKBKt7L1rJV5oy881ZPPmzKkqjEytZXPUT0X8wiBJhHwa49-8w=w103-h74-v0

d0f80afd-f2b0-4d05-9d5e-c312014556a8

https://lh3.googleusercontent.com/notebooklm/AG60hOr5MprfKeuGLdzkMQp0PF4NqE6CPieY6SbaiRN-rKOF9oMHX9Sra0IkCwNdtaLSOB6FEXFicGq2geYoxEJ9TXfuugo8g_24S-JMgUYpakpJClEj-af_vrcjNSsCSrjSUvLiiSWsYA=w512-h512-v0

0b321fd1-f8c3-463a-9f52-80291c8fda29

https://lh3.googleusercontent.com/notebooklm/AG60hOq7ZkAByr2x8xoOYh-uPdCrKbyzZKCpg5PIjuMRLuoBGCyP_oNfsTKYCunVrphxLCdBZpfLOSnzbIgeKJxRhJU6z6k_m5CrB1WBL7LlAkz5kFSpGFVXVCRidvVEGiyFh50tsU-9=w512-h512-v0

3d8ea5b3-3516-4e3f-99b9-471c64b509d5

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

## Threshold Maps

(excludes i)

## Combination

(excludes i)

GPU-based perceptual rendering framework

## Layer rendering

## Layer n background

TM of C2C2

Figure 2: The Perceptual Rendering Framework. On the left we see the individual layers. All layersbut i are com-bined with the background into combinations Ci (middle). A threshold map TMi is then computed for each combination Ci (right). Lower right: final image shown for reference.

it is required during layer combination (see Sect.4.2). This is necessary since objects in different layers may overlap in depth. TheN images of the layers are stored on the GPU as texturesLi for the layers with objects. We also treat shad-ows, since they can be responsible for a significant part of masking in a scene. We are interested in the masking of a shadow cast in a different layer onto the objects of the cur-rent layeri. See for example Fig.1, in which the bars of the upper floor window (in the first layer) cast a shadow on the Gargoyle object which is in the second layer. Since we do not render the object in this layer, we render a shadow mask in a separate render target, using the multiple render target facility. We show this shadow mask in Fig.3, left.

#### 4.2. Layer Combination and Threshold Maps

The next step is the combinations of layersCi . This is done by rendering a screen-size quadrilateral with the layers as-signed as textures, and combining them using a fragment program. The depth stored in the alpha channel is used for this operation.

Figure 3: Left: A shadow “mask” is computed for each layer, and stored separately. This mask is used in the ap-propriate layer combination (right).

We createN− 1 combinations, where each combination Ci uses the layers 1, . . . , i−1, i +1, . . . ,N containing the ob-jects, and thei-th layer is replaced with the background. Note that we also use the shadow mask during combination. For the example of Fig.3 (which corresponds to the scene of Fig. 1) the resulting combination is shown on the right.

Once the combinations have been created, we compute a threshold map [RPG99] using the approach described in [SS07] for each combination. The TVI function and eleva-tion CSF are stored in look-up textures, and we use the mip-mapping hardware to efficiently generate the Laplacian pyra-mids. The threshold map will give us a texture, again on the GPU, containing the threshold in luminance we can toler-ate at each pixel before noticing a difference. We thus have a threshold mapTMi corresponding to the combinationCi (see Fig.2).

Note that the computation of the threshold map for com-binationCi does not have the exact same meaning as the threshold map for the entire image. The objects in a layer obviously contribute to masking of the overall image, and in our case, other than for shadows, are being ignored. With this approach, it would seem that parts of the scene behind the current object can have an inappropriate influence on mask-ing. However, for the geometry of the scenes we consider, which all have high masking, this influence is minor. In all the examples we tested, this was never problematic. We re-turn to these issues in the discussion in Sect.8.

#### 4.3. Using the Perceptual Rendering Framework

We now have the layersLi , combinationsCi and threshold mapsTMi , all as textures on the graphics card. Our percep-tual framework thus allows rendering algorithms to make de-cisions based on masking, computed for the combinations of layers. A typical usage of this framework will be to perform an additional rendering pass and use this information to con-trol rendering parameters, for example LOD.

Despite the apparent complexity of our pipeline, the over-head of our approach remains reasonable. Specifically, the rendering of layers costs essentially as much as rendering

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOpAM05Wg8SqG8VmpXemHEfs3B_qSNc3_nHdxO60XON6EALxccNSgaqnwnhx9DhGBDOrWNOheg827WN__pSRhf_B1cGMMZqmRTrh-lp0OkDK8FfsDYnblQPuk-5Y9ARMGkiHYcX4Eg=w196-h263-v0

977b1de8-7abb-4e0b-a615-ed7e11bd6d84

https://lh3.googleusercontent.com/notebooklm/AG60hOpPEFq5hC39PLsAXrgC4KoHJFU7E5jpRY4GNVpJTXMgM3RlwjTCyCBUpMCLqdaU3-Jt4Q0wmAXePwe0KOvTlkjIqf45QUqnWA0Yccgty9F40rSnTXqVlpy8oCrFC80vHWkhB8Wqwg=w196-h263-v0

942f9412-aa1c-4a8d-8ad0-2e942582376e

https://lh3.googleusercontent.com/notebooklm/AG60hOpUyyj8PzcfemITJ0MhkeUcdGPwzLMikxNBNFdD8mZLdxzvsb4J9prBk1Ag226AmVsOUlXYQlYxKC4-9Za7DFH3JHeol-bfSKzTdroxzuR7Szqzg-x0xrxB489aFIaUTwyk1dlZ=w196-h263-v0

04f4b871-3d7b-41ba-af87-399dcf368a27

https://lh3.googleusercontent.com/notebooklm/AG60hOqRYV3loSamnk27QL_PBOWMqhD8eKMSm51KEKlYznMxGlj0gJLSNV0uakBM6UR_euO5GQeynBlFAUPRXX7ZoOK5_uNxJVo-cNT1g5CEvVH9TNhQDePw40sG7ws1SzJn7HTTx54S=w196-h263-v0

74150401-86bd-4d6e-9cbe-6a6bf223ed99

https://lh3.googleusercontent.com/notebooklm/AG60hOpMV-bXduAd9MhLTTNfzDfmnBGYM1pb9NnCYa6rtiycogkU4sqcv2o853YvuUyTxzAmkeb7xHdKiPWJBaJ94qvwqdhcC9mIQGfiHtIKbUHrrhj-lyc_fPwu8ireIfBALohZj0fB8Q=w196-h263-v0

f7923020-0f74-438a-bb0c-c238063db877

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

lworst = 7 . . . lworse= 5 lcurr = 4 lbetter= 3 . . . lbest= 1

Figure 4: Levels of the Gargoyle model, illustrating for lcurr = 4, and the values for lworst, lworse, lcurr, lbetter, lbest.

the scene and by combining all layers the final image is ob-tained. The additional overhead of combination and thresh-old maps is around 10–15 ms for 5 layers (see Sect.7).

The threshold mapTMi will typically be used when op-erating on objects of layeri. In the fragment program used to render objects of layeri, we useTMi as a texture, thus giving us access to masking information at the pixels.

For L layers, the total number of rendering passes is L (layer rendering)+ L− 1 (combinations)+ (L− 1) ∗ 9 (threshold maps). The combination and threshold map passes involve rendering a single quadrilateral.

### 5. Perceptually-Driven LOD Control Algorithm

The ultimate goal of our perceptually-driven LOD control algorithm is to choose, for each frame and for each object, a LOD indistinguishable from the highest LOD, orrefer-ence. This is achieved indirectly by deciding, at every frame, whether to decrease, maintain or increase the current LOD. This decision is based on the contrast and spatial masking in-formation provided by our perceptual rendering framework.

There are two main stumbling blocks to achieve this goal. First, to decide whether the approximation is suitable, we should ideally compare to a reference high-quality ver-sion for each objectat each frame, which would be pro-hibitively expensive. Second, given that our perceptual ren-dering pipeline runs entirely on the GPU, we need to get the information on LOD choice back to the CPU so a decision can be made to adapt LOD correctly.

For the first issue, we start with an initialization step over a few frames, by effectively comparing to the highest quality LOD. In the subsequent frames we use a delayed comparison strategy and the layers to choose an appropriate high quality representation to counter this problem with lower computa-tional overhead.

For the second issue, we take advantage of the fact that occlusion queries are the fastest read-back mechanism from the graphics card to the CPU. We use this mechanism as a counter for pixels whose difference from the reference is above threshold.

Before describing the details of our approach, it is worth noting that we experimented with an approach which com-pares the current level with the next immediate LOD, which is obviously cheaper. The problem is that differences be-tween each two consecutive levels are often similar in mag-nitude. Thus if we use a threshold approach as we do here, a cascade effect occurs, resulting in an abrupt de-crease/increase to the lowest/highest LOD. This is particu-larly problematic when decreasing levels of detail. Compar-ing to a reference instead effects a cumulative comparison, thus avoiding this problem.

#### 5.1. Initialization

For each object in layeri, we first want to initialize the cur-rent LOD l . To do this, we process objects per layer.

We use the following convention for the numbering of LODs: lworst, lworst− 1, . . . lbest. This convention is shown in Fig. 4, wherelcurr = 4 for the Gargoyle model.

To evaluate quality we use our perceptual framework, and the information inLi , Ci , andTMi . We will be rendering ob-jects at a high-quality LOD and comparing with the render-ing stored inCi . Before processing each layer, we render the depth of the entire scene so that the threshold comparisons and depth queries described below only occur on visible pix-els.

For each object in each layer, we render the object inlbest. In a fragment program we test the difference for each refer-ence (lbest) pixel with the equivalent pixel usinglcurr, stored in Ci . If the difference in luminance is less than the thresh-old, we discard the pixel. We then count the remaining pixels Ppass, with an occlusion query, and send this information to the CPU. This is shown in Fig.5.

There are three possible decisions: increase, maintain or decrease LOD.

We define two threshold values,TL andTU. Intuitively, TU is the maximum number of visibly different pixels we can tolerate before moving to a higher quality LOD, while if we go belowTL different pixels, we can decrease the LOD. The exact usage is explained next. We decide to:

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOocv4z4Sd3L2CjB-ZubFX54vEmoDvosP7rsj-TOtURY1kHx501WKzQPOLompOchx0O-ePe_qPDaKYTitoYpCwsjOG_UD9qTqa44jJd0s-Rb7bEfuO8fRj8ZEVWzUHl8jvOmmAC0bw=w1244-h77-v0

c8ea3ed3-d0c1-4218-8afd-a797b7473505

https://lh3.googleusercontent.com/notebooklm/AG60hOrWtWiWOa8J-AuqMLSmAJ8KsIdgUk7751NBDLmWy3lcQYZHS1L5BkmqO_31BFSf7HFNWNOBu_hiUMp2Nz7x_vY1EttcCGil0-peHG5ZSFnrjbClscDygTGg6FDVUg2qZ5X6Dvb0=w75-h527-v0

8696c11c-6d8d-478e-8b0c-965f3b10b7ba

https://lh3.googleusercontent.com/notebooklm/AG60hOq5XgWBpsjYbcAWtVuSwgB5AszjVPJ4xkWsSU47opAlCdpWd-V6GW09UbDM2hOZq3ziNMi0kYqkh6YmkJi5EEt4M2UbNqCHylKd6kEEtepXYyhiAO02Hf8245hZTdNTXRP3MDq5=w75-h527-v0

208ad132-bcb1-412e-8e5f-1f1f430a886a

https://lh3.googleusercontent.com/notebooklm/AG60hOqY_TmFGACWxExuORa9SCOFuB7Q7jxc_4_j1DuQ1KzhxTyoZQXgAj6VB2MYUZVj9jTdvMJgMdIC4QskH5bRU-oyuv930mhKnz0Xn9VgPfD42I5qdvCDFnB57v2JNba0P4xnGu-0Ww=w1098-h77-v0

767244c0-9edf-479f-a333-f7d658b7ae0b

https://lh3.googleusercontent.com/notebooklm/AG60hOoZiKFvjg6Hj727zuJmYxYtPr6DtQQFrT2nlk5fgTONPtnOoVS9o5maQQDWx3L6tLruyAfLveejSVVI0IB5udALPmRgcEXAni2kCrPJ5NKafgLuvfF-okg6otW4uf1KX_VFwxklMA=w969-h14-v0

c321005f-5832-4edd-9be1-8ee04df29d44

https://lh3.googleusercontent.com/notebooklm/AG60hOqq2QwIKjXjuknSf6sTQi_8Ik_K_AE-iBnCHCamf-TMht9Ydh8NDroD5cjEKPl5RDyEyQMjIFluxXQp5WgmQ3s76IxEPJ6YxZ21xPveDyd8Rs42tfRMfT_kAPPBVs0eglLOCYzmwA=w485-h73-v0

2a1b2a32-dee9-41b2-bb05-ba14c0ec8449

https://lh3.googleusercontent.com/notebooklm/AG60hOqKcL8szm5KcP13KhFMBq7RPIJd6T0dTK2siRCjxzGHLN9cSnYLGnu_G6rUoBDK_9ylTReDVoJhLBy0DI0cSyhE7dbHwwG4EXvZbDgbm3lMFCNY_g-m646Dk_tPiAVIj7d5hrlJ4w=w21-h248-v0

7fafc978-d302-4137-ba51-8dd780c1452d

https://lh3.googleusercontent.com/notebooklm/AG60hOrxIdRWpoylM3zxN_U_FrsJy3j8SBnYbakdnRWOHpwMrVgiat-VmHPnOESOoYPLNYPv965YPsr5cq-naKwwtFOlDgnDS6oRBLNVTYseWODTEovJylxo5Z4aalQG1CLcG_B1As5h7w=w1189-h125-v0

4cff6d20-1c43-4887-a585-4eaf1abf6659

https://lh3.googleusercontent.com/notebooklm/AG60hOrDiJi9e6s9qdDhCjrqSTgHqs3s29Hmjv73khpv9nRtvUuc1di8VZRmjEfNmJNr3-96OD6KUPTwa3RhOPypttdNq8d8ecWD2RWnADkp6bb0jMW28gytjaeiGUJcwGz5G07K4d1B=w969-h6-v0

29e01f79-0d00-4372-843d-abd2614c1354

https://lh3.googleusercontent.com/notebooklm/AG60hOqm1tozaTEwC_QhdO3fX9ejZI9eielfKouzBZAV2nOiGivl7iPVEFYPwVIQAh3QO3rH3KKvS4JXuUvrWAxS23VReevQmz2Ua4Xpl43-uoKE2OwtJ7tCIPhvzCh9oQwui_7mR56Rxg=w631-h25-v0

f40f21d7-6bfe-4035-98a7-dbb278755033

https://lh3.googleusercontent.com/notebooklm/AG60hOoJuq_1qRvo0jZWHkBRoV9wZd1rvTiO2w3B1dICFD800hxlA035j7mDzv-Amzz3Z-0XDmxX-pYKjMEECGDaZQ62I9r7XngMM2uxNMDASoSxv-uUDgsOr6x8ylzR3SyE8xbSV_bo=w317-h72-v0

bb741b9b-b842-4375-9f54-018242318b35

https://lh3.googleusercontent.com/notebooklm/AG60hOqJ85TujvLcvJ31AtWAElnm4MGl0yFnVZMZqAFml8Jvr21-DRlMIUziYclr3EYWozQe0SXEJftLEIRyGIg_iz3hs5yM7E6EWpQ8VS1lnJbEddTB78kHKHzwcHCooLgoCgVJDj8s=w170-h146-v0

9b44af9c-4612-4fe3-86f9-04b5eb7286f5

https://lh3.googleusercontent.com/notebooklm/AG60hOp2WyXisJFlKsLE3Hp40VWb5CUWmMX5lQzAf8H3iohhUDG-fOUV9zQX7mYItm7XXWGOhokHOTVem0HYJS5WI3TRaQr-xamk028InFXUCTFntc1rz18PqJi47QbjUNF2d45a28UC-Q=w129-h81-v0

019f3258-5e00-46fb-9098-ea344f316e08

https://lh3.googleusercontent.com/notebooklm/AG60hOoT8ldY0EfMP7J2mxXiI4yAUIUbM0m7-kqvzmCGzuPGDjNAdQMHcUe9uLFQMVv0laWGAtoVEZBxqyl9TuQPK8rVtWtcFbWFXNa8Dg2OmYYb6d62S1Uye0sTnQeeP0yhgG8e1swL6A=w581-h7-v0

a7f89fcc-7e89-4af5-abc1-1c7f99eae288

https://lh3.googleusercontent.com/notebooklm/AG60hOp2FXyLDdJ28MRVplXsNoKZkUldgBrye0tT7gS16qZBHHFU8FmYA0uja_ULrFoBlpfNYulQwSRND9Hpt6ZWB15S04obnA2hC60KuBYX_cAeDljnpp0dUxcaFUAmOyGLptZsneHs-g=w292-h354-v0

afe52ed8-4979-4d01-bf57-67d76217ec51

https://lh3.googleusercontent.com/notebooklm/AG60hOqZUZ9HJByoBzhJP1PN5rSkv8iX6Q3B5tv0ohlws5yWjGMpezCgJboobENvAojznz8hJLLNgGDDqr-awN5Eefa55XJ4neNlXPw42AhLq7QzEmXWX1-oGu2YjFjXVMLeZ4erJNKQxg=w581-h15-v0

ac009ec7-dad6-49bd-855a-cb5c2b65ce5b

https://lh3.googleusercontent.com/notebooklm/AG60hOonyQv6fD6hJS8hDpCAfbvKbnqR7uq7Ezm8K6BAx_NckLXqHQ5okI8ZiTcQmVxsATE_u1EOLsMgmPIV8uMBCuN7bsfRw2-RhRPqdc4icjNNnPka003vR3Bvbebhp4pOP27Gu2Yy4w=w40-h289-v0

a70f5b5a-0770-456f-9220-9401148fb929

https://lh3.googleusercontent.com/notebooklm/AG60hOqczGBzE-vXeLfPb4P16viqd0E7hI4FU_CzLSKHSKFtcKokKZbkEkW0e_KEUApB53HbRFVvSdnNAA5fMsXGW3QsS_tT88ZOZgz9QUJ55kH3Yv8Na4mQqFjQvw5vRRAVJ6GcsFJl2w=w548-h373-v0

239fedd7-f272-44bb-98cc-2c588e579ef7

https://lh3.googleusercontent.com/notebooklm/AG60hOoZONlKcN3GVmvS1W4JM2R2U5O5vwUhwhdXP_nBhzDW4l5eagDN8Ekp_1ylCPdmoaSj-Rb1KnKtxwqdlyaU3f63-j7CoIDQLdWEol4y1HmiHmkHgczK8eoOcvlqwU4OCyZZoG6X4A=w19-h202-v0

d7b193a7-125c-4772-8506-abf626e536cd

https://lh3.googleusercontent.com/notebooklm/AG60hOpKVAPZQRJ2EmAjpteYUp1XdvLXklZ-tPC37w3iM7_XEuBNxKcNMGWVbaJNo-QQFU2EtE5gYBfH67jddGOuyAREsbcjE5C-3Pg3Ti4nFACd-aIZHrbw5SMLu721Q1lx4t1rWmy6iw=w241-h187-v0

7f597443-6304-456c-bc4b-4a312b6995b1

https://lh3.googleusercontent.com/notebooklm/AG60hOobQ7cnZPJ5drCXHUhy79-6F1QL4JrVAm2uCnEEnfiEW6N9ay5EOc97thI5_TQX58Y1bzt3wWrWmP1IfHbGvXXElseMZwIoGcuBdUb9ieAE2BTGH0t0YO_bifDEeqBXAsXEQiM04w=w306-h187-v0

6bc464e9-17a2-435c-8c5c-d85d058261ae

https://lh3.googleusercontent.com/notebooklm/AG60hOoLxCsrICNhCURumYC-OeAHDV5P4RRkM087hX3hqF0Y8D6Lj69ORyHOHmD_m0WPlnYkLqb8nQfw4NGRkjNV5IgPnvlegDp6KdzX-uZL4cFUe3MZwtxWqNieGasRCO051dNfQ-zawA=w610-h12-v0

1e29bbb5-5383-493f-afc9-72839818d756

https://lh3.googleusercontent.com/notebooklm/AG60hOqUsiV9L_b-9t70Xid9a1mOvI7kIH63zWYGRLct8B8Ti-S4hdtBTBxI2sjrXSLG-2J7JYedzrOp3XKirGSFOjRj8Y42liBmIrDzUMRXn0Mn0lAPRyBmlPZPIxQ85HdPUX2T_-60=w34-h44-v0

54fe62ed-ffce-4272-8a47-667cbafe039b

https://lh3.googleusercontent.com/notebooklm/AG60hOrlUIGqF28psjOR4s8T_kWojCT2nn28vxDcjVGf-g84VrXuh0KnwXzKBbCCL6rMBdE8WO32hYsHmYtRd7DDLNHRm8QfU2L4D5EZGVirtRT3uQ7vM6tMOEdOpscrTUCDjA9EFG7J=w100-h94-v0

2b152985-1474-402d-bb17-ecd60153eb45

https://lh3.googleusercontent.com/notebooklm/AG60hOrGuNy10tDQ1-tcd3dKj-tfg0pqfts9aqUKC4CFhvUElZoNkChP9YT22mwKKr_Gt08H1UFumqKhRsRtwQ-vWzoUv2740aCI8EczbgKv751ujUmHuZEDsQjXvxN0aKrFG8PcaP7f=w71-h34-v0

42f14ff6-8bea-4d89-8b82-bb6a0eeac87d

https://lh3.googleusercontent.com/notebooklm/AG60hOrMKUjLZAf7CgkosqSkZP1NLv2AGD05P_5VFDDDT1qxwgCMLAY0s7ejWAFuqzAdNOrDdzLT3vAg_tMwFA5oxxHPy5HJg8bfYeQUNBU5W72J4j_dU-WD8O_ZwMaDrEJoPjsuPbVkcA=w15-h146-v0

612448df-2b0d-46df-b610-e9a12bf8b288

https://lh3.googleusercontent.com/notebooklm/AG60hOrIpUNzAy5v72MVWcdhieFDazurouRKz09a2vHd97xcgB5DDtrNGL2WmqcF6h4EXUBbfiyGdber6XNo74pieMq7ncO4-r29zNwIHxr07hL7BfyMyNPwzTyr54PStkA1tLfh1OSQ=w90-h89-v0

fce7d10c-be91-49ac-bc32-20b0713e6c0d

https://lh3.googleusercontent.com/notebooklm/AG60hOrFP3cKCU8IurxQOZC7CrsGgFKAmBRUo0BBDEC80JoNmPmj-XPN714OtQI3_Vm2QrVZTsMAFqfRJkj13oqK52XvY3BTSmvOTHcK_KnloqOe98qOXjiMJL4bxFJaVERjDKGYV9zPcg=w90-h45-v0

0161474a-c1c3-4511-bf98-f7bfe1fd16cc

https://lh3.googleusercontent.com/notebooklm/AG60hOrq9HLk43pC64dQlamK7pU41UUW9r9t71pfSHzyOzs-3DKGRexD8EsUnnUdexvd4ralmrucVZF1P18mXJ6BZZ2-vUP6Q0LZcskFmhKnXcRL6bENBkX5KlV2NwoPSHwYhxM0Ez-D=w77-h70-v0

10c3d757-9da9-49a9-88d9-e1c81a3501b7

https://lh3.googleusercontent.com/notebooklm/AG60hOoQar2A_U95DaJKph7I0l7eQbUJ3uy2zxACJ9WoXKPXxLOuBEzVbCIGZ-8jD0pRcKZ5E70iuoYvpCNkz7Rn2OgEkjlvIu3jx2LdBX8o2frgPGc9QmBqIGsz7nCXdGOYCXV6zEY-Og=w77-h63-v0

6f72b0a3-dfb6-4e9d-9058-6fbd5c4d3b28

https://lh3.googleusercontent.com/notebooklm/AG60hOqFa2fDgSGUkGfqDy8_iO9MYGHZDFx0OPSV6jHdt5__mVzXw72LJKS1_-Uy7CvJDd0kSbLouudjQpCzua5_7rI3yyDlYR1k571T8k-rZ6xYiUgSJZ-VPlJsxAaGjmoQ4FVT8w9LNA=w66-h71-v0

418011ac-1ab7-49b2-9430-ddb501dee881

https://lh3.googleusercontent.com/notebooklm/AG60hOqB_bmsALjR6hqwn0XyvLYMHDZObhHw9L86DBCs75W_SS_3FhkzbVqk1PvarDAu7JPlV8KXpkeC0Jh2Vj27WsigtoJqclzngjPk2YI08PmQAO6fsq_OSAV9hru8EkMNTMABnrx1KA=w68-h75-v0

12ddef6a-5c37-43ff-ab16-ca683317f1d8

https://lh3.googleusercontent.com/notebooklm/AG60hOrvTkQgRpR04SEVehRMzHmQKn4eZi2qtvWVTU1PvRWpsD920zsyGReemFbcEkEWXC7Kb7vs7TRQZJ-904X82rX7XP97r5-ldf0OLBM92ff3589Shcr2KUKwNJxLnRWa5wum5db7xA=w250-h148-v0

a13cad04-b229-4ee0-bc53-ee35772497e4

https://lh3.googleusercontent.com/notebooklm/AG60hOrYkqOyDO20FU7G6FVCFIcM99QLR9AC9RNF9hS-BICBaRWOZ3A1uMReumDKug0d8D_Lk_EfggwHtdOFqMRasepyB0E-kU1bzrX3DDm8Qg3NGa6PVqWAg6tjIFi7r6aDXX-iG0XQ0A=w498-h56-v0

2c5c3fa0-4c48-4f3f-b1ca-d951fe06368c

https://lh3.googleusercontent.com/notebooklm/AG60hOoDLKmp37J08dDXqzPFM-MBQwqYQkvjhgjNJJKUUHGgKaQPi3wYTUAyaw2HBMTEwqZlBwdMblBAyVEAH8AAR8z6fgRvxdRg8x1vm0OkbiyYtBolDHD5_iiAlmOUuwrW-Gwb_4cEuQ=w477-h14-v0

58cee73d-ba06-409f-91b6-397e52ef6933

https://lh3.googleusercontent.com/notebooklm/AG60hOqlCFr1ES8VjexAwyy7hosdIrXKqEI-v4T7CGkdzFwHUWAgnBaJOfT-0AJ4CGehKe2Rm85Nkq4lXEqpYsGQd1STGgRu5JkYUydSEbsHK2Ql0X8XVLIt7v8CW3BIBLsEabg13TTRfA=w48-h90-v0

88fcb89a-b7dc-4261-abbc-705be938cdad

https://lh3.googleusercontent.com/notebooklm/AG60hOoBIxaogqiN79v9LzS0GKd_BcEHhhTL6OmXMzUMkGJ9YLC0PIihrKnKbgSrphNowW0cqkmOLlqZi-ZE3QBJSJZFnmK8WJ6t9Pdxt0RqleAXVQ0HZDKMphLeTcBhuxqbZSquupHwGA=w48-h30-v0

63f16952-28e5-45b6-a922-0606e2dae0ce

https://lh3.googleusercontent.com/notebooklm/AG60hOqjoc71OmVHAdfG096BXY1ztalurjXriL1A0UkudLRwlAdM_wI9X6VGddk1q1HIl-qPZFbUwmCH0Ea2Ysw1fptQL-73Ku9Z0PF-ALHpxQDomHIu-s9v1d4b5Gkd9X8O6qWUHC_FUg=w85-h83-v0

5c00b26b-7312-490b-b1b5-e9746f9690cf

https://lh3.googleusercontent.com/notebooklm/AG60hOoY2XsjtslhhmSgCed6l7eGMwz0NQ6r9R454FPqNMv6DYzyB_ijxGja3KhsIb7buD0VVgLVWfuN8x-eBB6Dx6IJ0i6M8B79Rz8jCbe8WirfdI2ioh_91AohgkobvY4SocUXFg2Rug=w169-h52-v0

ce60080d-650d-45a7-ad99-a6e53e30c268

https://lh3.googleusercontent.com/notebooklm/AG60hOosKUQyAsfDw57v_Nzl_gU0E_2GyvBpp6Kn61hzM3PHYefATH8kiP_shnr7Rge1a2xJjBi0QaVnch56oyNLGOho-xYW7wFkg_7qvEo42i2JAWTladC1t10bTVhgC6MudcWhp7va=w67-h71-v0

87b861ac-a959-45eb-9897-351a5d9cfa27

https://lh3.googleusercontent.com/notebooklm/AG60hOqNBHG2Cd3VAJBm5yHJlIFRWf4Rs3jj1K4l8d9SL9X8iYK2D_7k-y9Cl0Ia_4DyYERwu_ojTqS2nHQDoHk0RbuP-phyWpjGKEVqPIlF-Sa3XZq90KjXGsj4PqJm4nUOaqJojt3srw=w71-h75-v0

d15b6fa7-caf8-471b-becc-114f2a62ae37

https://lh3.googleusercontent.com/notebooklm/AG60hOrrgRAslJ0clLi4d1GzK2Idar-ysyZ0OKzGPBMFm4Rc-1QKgwwpke5aMVuCGYoNBH-tGkNYCE0xtAwhZSRrU7I23H5OdgUT-QYpCBsTYYhlkeDSadCG2fLRwhCr0MKVTILjqWdmaA=w69-h81-v0

0bfef518-da7d-4479-8362-83f93451a48a

https://lh3.googleusercontent.com/notebooklm/AG60hOq-GvF2TKYHIDD696Wzky2esPZ5CBmZJzyFt5Kh1KwsOiQ22jdCWooctlfMGQk7ezU9LqiPeFC5lOjWY1OHbNK37YrGG4VbbhImnx-FlggfLA9bClpdJS_isiF1llVNDF-SgK9k=w119-h81-v0

ad1c7ebf-4c21-409a-8ae6-227b10e475b2

https://lh3.googleusercontent.com/notebooklm/AG60hOoe__qNJK2MT5zMKuoymipLUTucJBnq5bqWyTn78-iJXgoZe2pqlhB_0CdkeFypLE1jSD8RK7oMRz2QaG-C2HPR_7qEgMWbHxNPb0Z_HwkGqD0vfc46YNuNlRIGV2DRwg5Kdsf6GA=w69-h81-v0

f6c14c7c-aa4c-4a0f-95cb-f4ef03bf5b27

https://lh3.googleusercontent.com/notebooklm/AG60hOrOxaz5Sq7nxKJRxS6GJNP19PHcfKCEZgVh4vl_6ywDVxozdKTjh4sNjZaRHcP-2jqLJlKkW_XfPwY_bcorDyMPR8o1qh_bP0pmVAmAGqBgMIWUJ8UEJ5AXX1ytl052L65ulRN3Wg=w106-h160-v0

efe51d95-b6f9-405e-89ed-3a0294d30387

https://lh3.googleusercontent.com/notebooklm/AG60hOqQzvjggHhDQHFVFvh8XcAXLcb108JNU1GRSm9p1Uv12JcT1DMIW5gyTh3EEKHQIhtHCqd2tQHpPD7cC85URPNqRDKuKjqRI-eOZdTavU_f4r71SOU4qp1ypLQ19S2OwfIwGl9GCQ=w33-h39-v0

e08d217f-6ae9-4449-8eda-1d727bc34895

https://lh3.googleusercontent.com/notebooklm/AG60hOo_11nQWQLLX5nOyzMPc0yykzKUlZwTyKa9O2Nr9vRe9WIs7r-afjE6yVWaVF1Qj6FAicMXjM2rFFMttU9GuwYuG1tlFoQu-DTtL8m3tnIA5sgM6g6HBiq_MVVTmKb1tgzJZ9Gxkw=w33-h39-v0

9495847f-0f80-4c26-a185-e7cdced67f6d

https://lh3.googleusercontent.com/notebooklm/AG60hOolFZlyJtR3kN6xsFkSBTxBzhF8LZRbl6nn0MXRcc6R7c7tsrJvSuZsTeW8lH4PBjkuwRU0DY8BNf_MW_noTbKrCjSvxlBWhFGkRWeCtb-XoWtHLgHo1hKlkKx6_otkzRltLe8Cvg=w610-h16-v0

bd1fa65c-fe32-4fec-abad-4febfd02a463

https://lh3.googleusercontent.com/notebooklm/AG60hOqn1Ho3w7uQiRtjCtL9_gp_eIgK5-NZOfgNNopQvRz8bW-as1mNSGoQ0qtRbstOD59fhomQEMWTsZOYepXTgYjcvVsOZdeIKMN-r-aS12552mndZC0A_Ku5pw4OKOoqbmHmiCSc5g=w306-h185-v0

13140774-1e27-4a5e-b7ee-6ffcaa46f990

https://lh3.googleusercontent.com/notebooklm/AG60hOpvCQdosIkw9H_Lf5-PD7zvl5nZKxAqPUht8SgRwMkLnS_8QjPlBxRypOygxrqDnDM9Jkct2fr7q2C1yiMp8d5tiRtTOzgFDZT4tQfdU5hvBlL7AK1pC5Cu0aA8ITCZ9qAg_unCUA=w35-h52-v0

f3671bc1-95bf-401c-a91e-008c6f423114

https://lh3.googleusercontent.com/notebooklm/AG60hOqwH0Fq1kc9E_ZA5O00HO5Jyd6dQS8oVpYVDYWU_6olG1Q1RNE4GHo0et9irzri-KZLTR27o7EU1O_YaIScRmo_5UhmO9vgM6PaRGea24byn0L6zWSn2y18WwLPI4hPwY2JYKmaLQ=w609-h30-v0

3e54e97a-cf3e-493a-ba5b-0fe48a791be8

https://lh3.googleusercontent.com/notebooklm/AG60hOoAjoe_4H6MbTEC6koGxThe8ZbO2k02H7dH2MNlGeQKnJIEJTlqBm-dEwdZekxHc-FLzzl4rPTi2DPjnYn0tmtvWxBTivf-yAzQEOKOgezM8uTSUlNGU3wwJVLVcVYG006fEbt0ng=w15-h227-v0

2ae0b9b1-d8fd-4b9a-9f9b-c82c2ed5bbb4

https://lh3.googleusercontent.com/notebooklm/AG60hOrbtaB3gtpH4hEAiU9PD6p3b59mXATB2ibBLp1j5KQGEzvEPGjcihKvRGv5T1ExPNoLLf78jQDWuqP_C-QvtstDGcLZMO7ADyJBs6hNFgfLJFJe7Hbs0COYoTxofKPj3rMdMniQJw=w240-h114-v0

64eb49b4-5dc0-453a-aaf7-b8e163053102

https://lh3.googleusercontent.com/notebooklm/AG60hOp0DOAA6_grCMz8fIaqQ_N-EL0Kvjt9lTA_rvKjtvCAbLQkRwdmSJCkEP_yiO-4xDLDmFeNu0bz77j48i2w3w689xvqvhYnGXbrP2Ycm6XjfNLCPxwivpWudW1qXaQ5y3P4FV1dLA=w174-h148-v0

99177b55-cab1-428d-bbf1-417cfa4650be

https://lh3.googleusercontent.com/notebooklm/AG60hOpmC--gUrCcPQPr3dI-n-MwrUqqd5is1ZrHkI_F6U8AQMmKz3uPH1IKG9Jgacn87jxTduKgx896bzCHiktrScbvLARv_ihDuwiBzqyr7fPXKeSvXsTCNGeon4ds2jZaaEIusS5zOg=w27-h294-v0

23ec5acb-db92-49e8-914d-efea5f9f9db6

https://lh3.googleusercontent.com/notebooklm/AG60hOrppiW9-wn915bz7k7QBCgcG_Zeu8uDImK_Dq6FazCnQXbbUi05J8AeRx91jlT9ZvqxQU_QahYRN9DEjhSypISVuDJNXsV55rcNFE7lHkjdSiVMK8y2yhZ5MwnpJieOjWTgf4Oljg=w6-h177-v0

5e077936-57f3-47a4-a548-77ea98266fc1

https://lh3.googleusercontent.com/notebooklm/AG60hOpFCM2j0AlF5E9-nswoNRTarYJ1u6Q2Ahfm1D20d5dKeZ66fh-h2NW3M7I_oiq_iptabETpvIf9KbYTNoORNRMpC-joQ7EW25niMX27Wqucnyxkso_Bcf5WNm5fjCls0Ycaex-GqQ=w106-h81-v0

fe196bbe-9c9f-4830-9226-9e14030eff85

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

Per-Layer Perceptually-based LOD control

## Test Rendering for Layer i

discard Per object occlusion

query  counts kept

pixels,  i.e., those

with difference

greater than threshold

## Render all objects

of Layer i at LOD lbest keep

Figure 5: Perceptually-driven LOD control.

Increase the LOD if Ppass> TU. This indicates that the number of pixels for which we can perceive the difference betweenlcurr andlbestis greater than the “upper threshold” TU. Since we predict that the difference to the reference can be seen, we increase the LOD.

Maintain the current LOD ifTL < Ppass< TU. This in-dicates that there may be some perceptible difference be-tweenlcurr andlbest, but it is not too large. Thus we decide that it is not worth increasing the LOD, but at the same time, this is not a good candidate to worsen the quality; the LOD is thus maintained.

Decreasethe current LOD ifPpass< TL . This means that the difference to the reference is either non-existent (if Ppass= 0) or is very low. We consider this a good can-didate for reduction of level of detail; in the next frame the LOD will be worsened. If this new level falls into themaintaincategory, the decrease in LOD will be main-tained.

Care has to be taken for the two extremities, i.e., when lcurr is equal tolbestor lworst. In the former case, we invert the sense of the test, and compare with the immediately worse level of detail, to decide whether or not to descend. Similarly, in the latter case, we test with the immediately higher LOD, to determine whether we will increase the LOD.

LOD change Test Compare Visible? Decrease lcurr to lbest To Ref. N Maintain lcurr to lbetter 2 approx. N Increase lcurr to lbest To Ref. Y

Table 1: Summary of tests used for LOD control. Decrease and increase involve a comparison to a “reference”, while maintain compares two approximations. Our algorithm pre-dicts the difference to the reference asnot visible for the decrease and maintain decisions, it predicts the difference as beingvisible when deciding to increase.

#### 5.2. LOD Update

To avoid the expensive rendering oflbest at each frame, we make two optimizations. First, for each layer we define a

layer-specific highest LODlHQ which is lbest for layer 1, lbest+ 1 (lower quality) for layer 2 etc. Note that layers are updated at every frame so these values adapt to the configu-ration of the scene. However, if an object in a far layer does reach the originalHQ, we will decrease the value oflHQ (higher quality). The above optimization can be seen as an initialization using a distance-based LOD; in addition, we reduce the LOD chosen due to the fact that we expect our scenes to have a high degree of masking. However, for ob-jects which reachlHQ, we infer that masking is locally less significant, and allow its LOD to become higher quality.

Second, we use a time-staggered check for each object. At every given frame, only a small percentage of objects is actually rendered atlHQ, and subsequently tested. To do this, at every frame, we choose a subsetS of all objectsO where size(S) size(O). Note that for the depth pass, ob-jects which are not tested in the current frame are rendered at the lowest level of detail, since precise depth is unnecessary.

For each layerLi , and each object ofO of Li which is in S, we perform the same operation as for the initialization but comparing tolHQ instead of thelbest. The choice to increase, maintain or decrease LOD is thus exactly the same as that for the initialization (Sect.5.1).

### 6. Perceptual User Test

The core of our new algorithm is the decision to increase, decrease or maintain the current LODlcurr at a given frame, based on a comparison of the current levellcurr to an appro-priate referencelHQ. The threshold map is at the core of our algorithm making this choice; it predicts that for a given im-age, pixels with luminance difference below threshold will be invisible with a probability of 75%. Our use is indirect, in the sense that we count the pixels for which the difference to the reference is greater than threshold, and then make a decision based on this count.

The goal of our perceptual tests is to determine whether our algorithm makes the correct decisions, i.e., when the pipeline predicts a difference to be visible or invisible, the user respectively perceives the difference or not.

#### 6.1. General Methodology

The scene used in the test is a golden Gargoyle statue rotat-ing on a pedestal in a museum room (see Fig.1 and Fig.6). The object can be masked from shadows cast by the bars of the window above, or the gate with iron bars in the doorway (Fig. 1; see the video for an example with bars). The param-eters we can modify for each configuration are the frequency of the masker (bars, shadows), and the viewpoint.

Throughout this section, it is important to remember how the LOD control mechanism works. At each frame, the im-age is rendered with a given viewpoint and masking config-uration. For this configuration, the algorithm chooses a LOD

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOqKT7MSv1DS76vCfOsYXKo9oV85tC_looPIVfV7d_cB4GEb-ps22-wrAGqeC4uGCKZf4KrysN7miiK1prrdRllHYjPKhel82Isdmtval5bDVLvx_teQZmegN3LS4l2ZavxOI9Darg=w512-h512-v0

a973db1f-a779-47e5-a4fb-108d8aa6113b

https://lh3.googleusercontent.com/notebooklm/AG60hOpXORHgfnqE8vC_K8WklnVM_TOkYNYbXIH25SAMcq5aXsHs1E_Wwz4JS4vldOV7H3kCAX8f3drNv9Mrle4HYsZhi-YzfW6CS-UJv-prD99r0LLkuIF1KrH--2saURZZxy_jSB6t=w512-h512-v0

546e3ceb-835e-4055-b764-07d4eea42e7d

https://lh3.googleusercontent.com/notebooklm/AG60hOrdruSa5zt9qwBpoNgI1BZCQetWfCr9CKiwFLpUmZyHoJXKORFcLRH-3zLaF9paepWk2SvBfsn0TPvHyFrrjqMUl6EJRhCXGB26V8nh0OaiH3n0fHPvb-GH0OdiWpSC82xu4SEaeg=w512-h512-v0

aa428d27-52a9-4d51-bc7a-df2f4e44f812

https://lh3.googleusercontent.com/notebooklm/AG60hOpIe9AgfZTv9I09uLdU2YmCrBLzH7UV9nYHINkIFD18dnkyO-J4P_lthYJTVUI7OfKKmDHNeEwKSB_Il7F4gRR9IBONhP4IWBoZYzMYb-p6O6b_7U7K9JTHVg4Sn___7fhgnfawiA=w512-h512-v0

eab4bf36-ceb7-4bc8-bab9-02efd355c043

https://lh3.googleusercontent.com/notebooklm/AG60hOopLZNsPY_Hiz3N627b5Jmxa3wLdWxU3J3P7oEUAMhUMz6oEPJ53evuH-wFRdnhEMzbvSBwrtqhn2VUn4xW8jYMzcxnN7wNzW5u5B_itBn29ITPPy1KFvZONVO0bQy3vxdsA8qYYQ=w512-h512-v0

a61e0bcd-55a3-433c-9f68-c9fe93fed277

https://lh3.googleusercontent.com/notebooklm/AG60hOpFwy6YXCsc70NH7J_DBu7OTIND57YvJn5i8lGLtmX6u_TKR6FXGjz7jjxlDAFvWAPSflNW7MO8VMKpqaC-5WkNaW2dRdTIbDNOoB7zUaGbxFUqgbOGoL8Y56WNffWrdWNsKuNv_w=w512-h512-v0

a38e01b3-4048-4874-919f-1e073d2b99f8

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

lcurr. Note that in what follows, thehighestquality LOD used is level l1 and thelowestis l6.

We will test the validity of the decision made by our al-gorithm to increase, maintain and decreasethe LOD. For a given configuration oflcurr, we need to compare to some otherconfiguration, which occurred in a previous frame.

From now on, we use the superscript to indicate thelcurr

value under consideration. For example, all test images for the caselcurr = 4 are indicated asl45, l44, l43, l41 (see also Fig. 6). We use a bold subscript for easier identification of the level being displayed. For example,l45, is the image ob-tained if level 5 is used for display, whereas our algorithm has selected level 4 as current. Note that thel i notation usu-ally refers to the actual LOD, rather than a specific image.

Based on Table1, we summarize the specific tests per-formed for the different values oflcurr in Table2. Please refer to these two tables to follow the discussion below.

lcurr Decrease (I) Maintain (I) Increase (V) l33 l32/l31 l33/l32 l34/l31 l44 l43/l41 l44/l43 l45/l41 l55 l54/l51 l55/l54 l56/l51 l66 l65/l61 l66/l65 l67/l61

Table 2: Summary of the comparisons made to validate the decisions of our algorithm.

Consider the case oflcurr = 4 (see 2nd row of Table2). For the case of increasing the LOD, recall that the number of pixels of the imagel45 (lower quality) which are different from l41 is greater thanTU. The algorithm then decided that it is necessary to increase the level of detail tol4. To test the validity of this decision we ask the user whether they can see the difference betweenl45 (lower quality) andl41. If the dif-ference isvisible, we consider that our algorithm made the correct choice, since our goal is to avoid visible differences from the reference. A pair of images shown in the experi-ment for this test is shown in Fig.6 (top row).†

For the case of maintaining the LOD, the number of pix-elsPpassof l4 which are different froml1 is greater than the TL , and lower thanTU. To validate this decision, we ask the user whether the difference betweenl44 and l43 (better qual-ity) is visible. We consider that the correct decision has been made if the difference isinvisible. Note that in this case we perform an indirect validation of our pipeline. While for in-crease and decrease we evaluate the visibility of the actual test performed (i.e., the current approximation with the refer-ence), for “maintain”, we compare two approximations as an

† In print, the images are too small. Please zoom the electronic ver-sion so that you have 512×512 images for each case; all parameters were calibrated for a 20" screen at 1600×1200 resolution.

Figure 6: Pairs of images used to validate decisions for lcurr = 4. Top row: decision to increase: compare l4

5 to l41. Middle row: decision to maintain: compare l4

4 to l43. Lower row: decision to decrease: compare l4

indirect validation of the decision. A pair of images shown in the experiment for this test is shown in Fig.6 (middle row).

Finally, our method decided to decrease the LOD tol4 when the difference ofl43 (higher quality) withl41 is lower thanTL . We directly validate this decision by asking whether the user can see the difference betweenl43 andl41. If the dif-ference isinvisible we consider the decision to be correct, since usingl3 would be wasteful. A pair of images shown in the experiment for this test is shown in Fig.6 (last row).

We loosely base our experiment on the protocol defined in the ITU-R BT.500.11 standard [ITU02]. We are doing two types of tests, as defined in that standard: comparison to ref-erence for the increase/decrease cases, and a quality compar-ison between two approximations for the maintenance of the current level. The ITU standard suggests using the double-stimulus continuous quality scale (DCSQS) for comparisons

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOr-LkSnIB7T8m9V9tqSu9lE0Yozefpl4wmARtZ3kbQQc322CuRRE2AX9jMFI1ks1gQ7FViOt-QYMMTMrCSV9qrGL2HovLpwXkJ8FlRr9Xt3koIDA6dRzatYioPfrJmY2LgJY_lxOQ=w1601-h1200-v0

43331d12-33cb-4e83-bdf3-6ada9a776509

https://lh3.googleusercontent.com/notebooklm/AG60hOoKdsBhorp6ROzPeUGQ7Zk2Xfuo-KYaQMOsWsYhlpaF2tGVK4gw3dD3YtFYLJuK-hpcIxhkcNgM0togCrnvVmMcw9-xO6-uMK_CWHLysVYgZTI_CEikvKneFZ5dYWv7aQpT0T7Y=w452-h308-v0

645faa09-52ff-4ea5-be61-42d5585a3e70

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

Figure 7: Left: A screenshot of the experiment showing a comparison of two levels and the request for a decision. Right: One of the users performing the experiment.

to reference and the simultaneous double stimulus for con-tinuous (SDSCE) evaluation method for the comparison of two approximations.

We have chosen to use the DCSQS methodology, since our experience shows that the hardest test is that where the difference in stimuli has been predicted to be visible, which is the case when comparing an approximation to the refer-ence. In addition, two out of three tests are comparisons to reference (see Table1); We thus prefer to adopt the recom-mendation for the comparison to a reference.

#### 6.2. Experimental Procedure

The subject sits in front of a 20" LCD monitor at a distance of 50 cm; the resolution of the monitor is 1600×1200. The stimuli are presented on a black background in the centre of the screen in a 512×512 window. We generate a set of pairs of sequences, with the object rendered at two different levels of detail. The user is then asked to determine whether the difference between the two sequences is visible. Each sequence shows the Gargoyle rotating for 6 s, and is shown twice, with a 1 s grey interval between them. The user can vote as soon as the second 6 s period starts; after this, grey images are displayed until the user makes a selection. Please see the accompanying video for a small example session of the experiment.

We perform one test to assess all three decisions, with four different levels forlcurr. We thus generate 12 configurations of camera viewpoint and masker frequency (see Tab.2). The masker is either a shadow or a gate in front of the object. For each configuration,lcurr is the current level chosen by the algorithm. We then generate 4 sequences usinglcurr, lworse, lbetterandlbest. We show an example of such a pair, as seen on the screen, with the experiment interface in Fig.7.

The subject is informed of the content to be seen. She is told that two sequences will be presented, with a small grey sequence between, and that she will be asked to vote whether the difference is visible or not. The subject is additionally instructed that there is no correct answer, and to answer as quickly as possible. The subject is first shown all levels of detail of the Gargoyle. The experiment then starts with a

lcurr decrease maintain increase l3 78.4% 80.6% 32.9% l4 78.4% 84,0% 76.1% l5 72.7% 31.8% 80.6% l6 32.9% 61.3% 71.5%

Table 3: Success rate for the experimental evaluation of our LOD control algorithm. The table shows the percentage of success for our prediction of visibility of the change in LOD, according to our experimental protocol.

Figure 8: Graph of results of the perceptual user test.

training session, in which several “obvious” cases are pre-sented. These are used to verify that the subject is not giving random answers.

The pairs are randomized and repeated twice with the gate and twice with shadows for each condition, and on each side of the screen inverted randomly. A total of 96 trials are pre-sented to the user, resulting in an average duration of the experiment of about 25 minutes. We record the test and the answer, coded as true or false, and the response time.

#### 6.3. Analysis and Results

We ran the experiment on 11 subjects, all with normal or corrected to normal vision. The subjects were all members of our labs (3 female, 8 male), most of them naive about the goal of the experiment. The data are analysed in terms of correspondence with the decision of the algorithm. We show the average success rate for each one of the tests in Table3.

We analysed our results using statistical tests, to deter-mine the overall robustness of our approach, and to deter-mine which factors influenced the subjects decisions. We are particularly interested in determining potential factors lead-ing to incorrect decisions, i.e., the cases in which the algo-rithms does not predict what the user perceives.

Analysis of variance for repeated measures (ANOVA), with Scheffé post-hoc tests, was used to compare the scores across the different conditions. We performed an ANOVA with decisions, i.e., decrease, maintain and increase (3), lev-els of details (4) and scenes, i.e., shadows or a gate (2), as within-subjects factors on similarity with algorithm scores. The results showed a main effect of LOD (F(3,129) = 14.32, p < 0.000001), as well as an interaction between

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOqXpkZQP3MgJZDwb40S0b2TyVTvFI2Xkr9LrVW_oKnK6Nvj8SYr1G8tJJ_nlbPFmfCkA0B-IjiKqlOabecqvgmHKMmz8RkYPeD8qe5zxXshNUz9Yt3wYXtHjfwPJ69tTRe-13VqIg=w512-h512-v0

e5e22047-7607-4f08-87ff-d873bf3020d8

https://lh3.googleusercontent.com/notebooklm/AG60hOqJuOMk9mz7JMDVgvxQBQeGCcC6uu0seudlwwdI8JPjIgs1HPYsKxmZu-2y-duTuph4MK3zqo06XSIbtrkbgKUIc7WjvGjOr9DIhyaXk5l0Cfj00FKWoQUjfnatrDFNiDSdmRcdCA=w512-h512-v0

3e42f27a-2497-41f1-997c-45497e68fa13

https://lh3.googleusercontent.com/notebooklm/AG60hOojWIKqQHO21LjUj9lmzNti2X9udeWqKX_79IDbogBhg6WMu7jSmWIHQSsmo33EGJCQP6ChYn-sXZl7nepREgxBndxM0qHuX48Tp842629ruQG7NUqE9HpEhw1AkcNzZiAvDyak=w512-h512-v0

cf0fd1ce-431b-449c-8667-19128af0e003

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

the factors decisions and LODs (F(6,258) = 23.32, p < 0.0000001). There was no main effect of the factor scene, nor any interaction involving it, showing that shadows or gate present the same decision problems for the algorithm.

Scheffé post-hoc tests were used to identify exactly which LOD differed from any of the other LOD according to the decision. In the decrease decision, the scores forlcurr = 6 are different from all the scores of the other levels in this de-cision (l3: p < 0.0001;l4: p < 0.0001;l5: p < 0.002). This is to be expected, since the test with LOD 6 is not in agree-ment with the subject’s perception (only 33% success rate). While the comparison withlbestis predicted by the algorithm to be invisible, the subjects perceive it most of the time. In the maintain decision, the test for LOD 5 is significantly dif-ferent from the tests wherelcurr is 4 (p < 0.00001) and 3 (p < 0.000001). In this test, the comparison betweenl55 and l54 is predicted to be invisible. However, it is perceived by the subject in almost 70% of the cases.

Looking at Table2, we can see that both the decrease deci-sion for LOD 6 and the maintain decision for LOD 5 involve a comparison of level 5 (even though the images are differ-ent). For decrease at LOD 6 we comparel65 to l61 and for maintain at LOD 5, we comparel55 to l54. This is due to the “perceptual non-uniformity” of our LOD set; the difference of l5 from l4 is much easier to detect overall (see Fig.4). This indicates the need for a better LOD construction algorithm which would be “perceptually uniform”.

Finally, post-hoc tests show that in the increase decision, the test forl3 is significantly different from the test involving the other LODs, indicating that this test is not completely in agreement with the subject’s perception (l4: p < 0.001; l5: p < 0.0001;l6: p < 0.01). The algorithm predicts the differ-ence betweenl34 andl31 to be visible; however, this difference is harder to perceive than the difference for the lower qual-ity LODs, and hence the user test shows a lower success rate. This result is less problematic, since it simply means that the algorithm will simply be conservative, since a higher LOD than necessary will be used.

Overall the algorithm performs well with an average suc-cess rate of 65.5%. This performance is satisfactory, given that we use the threshold map, which reports a 75% prob-ability that the difference will be visible. If we ignore the cases related to level 5, which is problematic for the reasons indicated above, we have a success rate of 71%. We think that this level of performance is a very encouraging indica-tor of the validity of our approach.

### 7. Implementation and Results

Implementation. We have implemented our pipeline in the Ogre3D rendering engine [Ogr], using HLSL for the shaders. The implementation follows the description pro-vided above; specific render targets are defined for each step

Figure 9: General views of the three complex test scenes: Treasure, Forest and House.

Model l0 l1 l2 l3 l4 l5 House and Treasure scenes

Ornament 200K 25K 8K 5K 1K Column 39K 23K 7K 3K 2K

Bars 1K 5K 10K 30K Gargoyle 300K 50K 5K 500 Poseidon 200K 100K 50K 10K 3K 1K Pigasos 130K 50K 10K 1K

Lionhead 200K 100K 50K 20K 5K 1K Forest scene

Raptor 300K 100K 25K 10K 1K 500

Table 4: LODs and polygon counts for the examples.

such as layer rendering, combinations, threshold map com-putation and the LOD control pass. Ogre3D has several lev-els of abstraction in its rendering architecture, which make our pipeline suboptimal. We believe that a native DirectX or OpenGL implementation would perform better.

Results.We have tested our pipeline on four scenes. The first is the Museum scene illustrated previously. In the video we show how the LODs are changed depending on the fre-quency of the gate bars or the shadows (Fig.1).

We also tested on three larger scenes. For all tests reported here, and for the accompanying video, we use 5 layers and a delay inlHQ testing which is 4 frames multiplied by the layer number. Thus objects in layer 2, for example, will be tested every eighth frame. We use 512×512 resolution with TU = 450 andTL = 170.

The first scene is a Treasure Chamber (Fig.9, left), and we have both occlusion and shadow masking. The second scene (Fig.9, middle) is a building with an ornate facade, loosely based on the Rococo style. Elements such as statues, wall ornaments, columns, balcony bars, the gargoyles, etc. are complex models with varying LODs for both. For the former, masking is provided by the gates and shadows, while for the latter masking is provided by the partial occlusion caused by the trees in front of the building and shadows from the trees. Table4 lists the details of these models for each scene.

The two rightmost images of Fig.10and Fig.12 illustrate the levels of detail chosen by our algorithm compared to the distance-based approach for the same configuration. We can see that the distance-based approach maintains the partially visible, but close-by, objects at high levels of detail while our

c© The Eurographics Association 2007.

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

approach maintains objects which do not affect the visual result at a lower level.

The third scene is a forest-like environment (Fig.9, right). In this scene, the dinosaurs have 6 levels of detail (see Ta-ble4). Trees are a billboard-cloud representations with a low polygon count, but do not have levels of detail. In Fig.12 (mid-right), we show the result of our algorithm. As we can see, the far dinosaur is maintained at a high level. The trees hide a number of dinosaurs which have an average visibil-ity of 15% (i.e., the percentage of pixels actually rendered compared to those rendered if the object is displayed in iso-lation with the same camera parameters). On the far right, we see the choice of the standard distance-based Ogre3D al-gorithm, where the distance bias has been adjusted to give approximately the same frame rate as our approach.

In terms of statistics for the approach, we have measured the average LOD used across an interactive session of this scene. We have also measured the frequency of the LOD used aslHQ. This is shown in Fig.11; in red we have the lHQ and in blue the levels used for display. Table5 shows the statistics.

LOD 0 1 2 3 4 5 Total

Ctrl 16325 3477 2596 12374 3219 1512 39503 Rndr 7623 4991 70067 1285 53371 21395 158732

Ctrl 5249 1763 526 1141 1 60 8740 Rndr 6954 3140 27091 3260 360 5038 45843

Ctrl 44 431 1 6424 85 6985 Rndr 306 39 56 145 69932 70478

Table 5: Number of renderings overall for each LOD in the two test scenes over fixed paths (252, 295 and 186 frames respectively; see video for sequences). “Ctrl” is the number of renderings of lHQ in the LOD control pass, while “Rndr” are those used for display. Total shows total number of ren-derings and the percentage of Ctrl vs. Rndr.

The total number oflHQ rendering operations is much lower (10–20%) than the total number of times the objects are displayed. We can also see that objects are rendered at very low level of detail most of the time.

We have also analysed the time used by our algorithm for each step. The results are shown in Table6. All timings

Scene tot. (FPS) L C TM D LC Treasure 40.3 (24.8) 11.3 1.4 9.7 6.4 11.6 House 31.5 (31.7) 8.5 2.0 13.0 6.8 1.2 Forest 52.4 (19.0) 10.2 1.7 16.5 5.2 18.8

Table 6: Running times for each stage. L: rendering of the layers, C: combinations, TM: threshold map computation, D: depth pass and LC: LOD control pass (rasterization and occlusion query). All times in milliseconds.

are on a dual-processor Xeon at 3 GHz with an NVIDIA GeForce 8800GTX graphics card. The cost of rendering the scene with the levels of detail chosen by our perceptual pipeline is around 10 ms for all scenes. For the Forest and Treasure scene, the dominant cost (46% and 68% respec-tively) is in the depth pass and the LOD control, i.e., the rendering of objects inlHQ. For the House scene, this cost is lower (25%). However, the gain in quality compared to an equivalent expense distance-based LOD is clear.

The cost of the threshold map should be constant; how-ever, the Ogre3D implementation adds a scene-graph traver-sal overhead for each pass, explaining the difference in speed. We believe that an optimized version should reduce the cost of the threshold map to about 6 ms for all scenes.

### 8. Discussion and Issues

Despite these encouraging results, there are a number of open issues, which we consider to be exciting avenues for future work.

Currently our method has a relatively high fixed cost. It would be interesting to develop a method which estimates the point at which it is no longer worth using the perceptual pipeline and then switch to standard distance-based LOD. Periodic tests with the perceptual pipeline could be per-formed to determine when it is necessary to switch back to using our method, but attention would have to be paid to avoid “stagger” effects.

Our approach does display some popping effects, which is the case for all discrete LOD methods. We could apply stan-dard blending approaches used for previous discrete LOD methods. Also, in the images shown here we do not perform antialiasing. It would be interesting to investigate this in a more general manner, in particular taking into account the LODs chosen based on image filtering. The choice of layers can have a significant influence on the result; more involved layer-generation method may give better results.

The remaining open issues are perceptual. The first relates to the thresholds used for LOD control. We currently fix the values ofTU andTL manually, for a given output resolution. For all complex scenes we used 512×512 output resolution, and values ofTU = 450 andTL = 170. We find it encour-aging that we did not have to modify these values to obtain the results shown here. However, perceptual tests could be conducted to see the influence of these parameters, and de-termine an optimal way of choosing them. The second issue is the fact that the “threshold map” we use does not take the current layer into account, thus the overall masking of the final image is not truly captured. Again, perceptual tests are required to determine whether this approximation does influence the performance of our approach. Finally, the per-ceptual test should be performed on more diverse scenes to confirm our findings.

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOoFDmrjdXr24l4cF_YSiZxTMFZi-c2HYf5oO6dyp38_MTVzoZV7XfHrSAFuWJv65DwNk9gqiixSkTrLG9e2gdXUE00CVi4hKOGrhFsi366rdTMSIZpl1DEHfwHkeawPzkiPQbAYMg=w2020-h512-v0

bac07304-b405-41b8-81e4-cbccd0e00d98

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

Figure 10: From left to right: Treasure scene using our algorithm; next, the same scene using distance-based LOD. Notice the difference in quality of the large Poseidon statues in the back. Leftmost two images: The levels of detail chosen by each approach for our method and distance based LOD respectively; LODs coded as shown in the colour bar. (The system is calibrated for 512×512 resolution images on a 20" screen; please zoom for best results).

### 9. Conclusions

We have presented a novel GPU-based perceptual rendering framework, which is based on the segmentation of the scene into layers and the use of threshold map computation on the GPU. The framework is used by a perceptually-driven LOD control algorithm, which uses layers and occlusion queries for fast GPU to CPU communication. LOD control is based on an indirect perceptual evaluation of visible differences compared to an appropriate reference, based on threshold maps. We performed a perceptual user study, which shows that our new perceptual rendering algorithm has satisfactory performance when compared to the image differences actu-ally perceived by the user.

1 2 3 4 5 6

## LODs rendered

## Control LODs

1 2 3 4 5 6 7

## LODs rendered

## Control LODs

## Forest SceneHouse Scene

1 2 3 4 5 6 7

## LODs rendered Control LODs

## Treasure Scene

Figure 11: In blue the average number of renderings for the objects over an interactive session for each LOD (horizontal axis). In red the statistics for lHQ.

To our knowledge, our method is the first approach which can interactively identify inter-object visual masking due to partial occlusion and shadows, and can be used to im-prove an interactive rendering pipeline. In addition, we do not know of previous work on interactive perceptual render-ing which reported validation with perceptual user tests. We are convinced that such perceptually based approaches have high potential to optimize rendering algorithms, allowing the domain to get closer to the goal of “only render at as high a quality as perceptually necessary”.

In future work, we will consider using the same algorithm with a single layer only. For complex environments, it may

be the case that the instability of LOD control caused by self-masking is minor, and thus the benefit from layers no longer justifies their overhead, resulting in a much faster and higher quality algorithm. Although supplemental perceptual validation would be required, we believe that the main ideas developed hold for a single layer. We could also include spatio-temporal information [YPG01] to improve the per-ceptual metric.

Other possible directions include investigating extensions of our approach to other masking phenomena, due for exam-ple to reflections and refractions, atmospheric phenomena etc. We will also be investigating the use of the pipeline to control continuous LOD approaches on-line, or to generate perceptually uniform discrete levels of detail.

## Acknowledgments

This research was funded by the EU FET Open project IST-014891-2 CROSSMOD (http://www.crossmod.org). C. Dachsbacher received a Marie-Curie Fellowship “Scalable-GlobIllum” (MEIF-CT-2006-041306). We thank Autodesk for the donation of Maya. Thanks go to D. Geldreich, N. Tsingos, M. Asselot, J. Etienne and J. Chawla who partici-pated in earlier attempts on this topic. Finally, we thank the reviewers for their insightful comments and suggestions.

## References

[ASVNB00] ANDÚJAR C., SAONA-VÁZQUEZ C., NAVAZO I., BRUNET P.: Integrating occlusion culling and levels of detail through hardly-visible sets.Computer Graphics Forum 19, 3 (August 2000), 499–506.

[BM98] BOLIN M. R., MEYER G. W.: A perceptually based adaptive sampling algorithm. InProc. of ACM SIG-GRAPH 98(July 1998), pp. 299–309.

[Dal93] DALY S. J.: The visible differences predictor: An algorithm for the assessment of image fidelity. InDigi-tal Images and Human Vision, Watson A. B., (Ed.). MIT Press, 1993, ch. 14, pp. 179–206.

c© The Eurographics Association 2007.

https://lh3.googleusercontent.com/notebooklm/AG60hOpfcwQslL3Ev5MTuTHXH5S3Rczz6erX-D0-5o73lJKE7bGk3NBIddBk09IJ44A5ElEvPJYNBhDdpSdSyT7rQSMNaSnELX-CBzgbWpAZTWLACagTg3hW5w05DN6RrUie_QaxNPGftQ=w512-h512-v0

3c5131b5-0e76-4c28-a032-7fa442855b96

https://lh3.googleusercontent.com/notebooklm/AG60hOqUBC32QTOEkG53eQSJgB6Uo-UtL1x5wF7J2Ywf94zV45V-RdB28MpCHvX7Z_arhuipnUDDgAyDa1cldmDwMyQzeSbrW_4Kdrb9CwgMIkRFskAWKWEFyLDCL6NX_ojBP9wf3jmtFQ=w512-h512-v0

60338318-79cb-49a6-afd6-caadb14e0b47

https://lh3.googleusercontent.com/notebooklm/AG60hOoKE_XBUI5WrDDl8f2_GZdb_wIBiqwF4J_L3sTHnZB8HFfbC9xs-vxFZ_j26CIFvxHQtno32kzGH81sCh_NND5r4audvfG1yrRvqrI9YfqHvXmzA3g3v6ruIAEwZn9PTmznk5Tz=w512-h512-v0

5a58cb79-f7f3-49f7-ae3b-56e2051e7711

https://lh3.googleusercontent.com/notebooklm/AG60hOrsq15q-w6fISdJsRXQ0xNX3Re30JZYfOWTpDdsj1iNWufByeJmdGluEPY2_MD3tRPu1EdCQYP3Po_IXiOdDtmom-H4yDN3YLVTMQaeeZokrZFnOdxXAu5nf4zRfu-S_aO0TtBxGg=w512-h512-v0

a55bef4d-982f-4d07-ab8a-fc73d18f9306

Drettakis, Bonneel, Dachsbacher et al. / An Interactive Perceptual Rendering Pipeline

Figure 12: From left to right: House with our approach, followed by distance-based LOD at equivalent frame rate. Notice the difference in quality of statues inside the door. Next, the Forest scene with our approach, followed by the equivalent frame rate distance-based rendering. Notice difference in quality of the raptor in the back.

[DPF03] DUMONT R., PELLACINI F., FERWERDA J. A.: Perceptually-driven decision theory for interactive realis-tic rendering.ACM Trans. on Graphics 22, 2 (Apr. 2003), 152–181.

[FPSG96] FERWERDA J. A., PATTANAIK S. N., SHIRLEY P., GREENBERG D. P.: A model of visual adaptation for realistic image synthesis. InProc. of ACM SIGGRAPH 96(Aug. 1996), pp. 249–258.

[FPSG97] FERWERDA J. A., PATTANAIK S. N., SHIRLEY P., GREENBERG D. P.: A model of visual masking for computer graphics. InProc. of ACM SIGGRAPH 97(Aug. 1997), pp. 143–152.

[GBSF05] GRUNDHOEFER A., BROMBACH B., SCHEIBE R., FROEHLICH B.: Level of detail based occlusion culling for dynamic scenes. InGRAPHITE ’05 (New York, NY, USA, 2005), ACM Press, pp. 37–45.

[GH97] GIBSON S., HUBBOLD R. J.: Perceptually-driven radiosity. Computer Graphics Forum 16, 2 (1997), 129– 141.

[ITU02] ITU: Methodology for the subjective assessment of the quality of television pictures.ITU-R Recommenda-tion BT.500-11(2002).

[LH01] LUEBKE D., HALLEN B.: Perceptually driven simplification for interactive rendering. InProc. of EG Workshop on Rendering 2001(June 2001), pp. 223–234.

[Lub95] LUBIN J.: A visual discrimination model for imaging system design and evaluation. InVision Mod-els for Target Detection and Recognition, Peli E., (Ed.). World Scientific Publishing, 1995, pp. 245–283.

[MTAS01] MYSZKOWSKI K., TAWARA T., AKAMINE

H., SEIDEL H.-P.: Perception-guided global illumina-tion solution for animation rendering. InProc. of ACM SIGGRAPH 2001(Aug. 2001), pp. 221–230.

[Mys98] MYSZKOWSKI K.: The Visible Differences Pre-dictor: applications to global illumination problems. In Proc. of EG Workshop on Rendering 1998(June 1998), pp. 223–236.

[Ogr] OGRE 3D: Open source graphics engine.http:// www.ogre3d.org/.

[QM06] QU L., MEYER G. W.: Perceptually driven inter-active geometry remeshing. InProc. of ACM SIGGRAPH 2006 Symposium on Interactive 3D Graphics and Games (Mar. 2006), pp. 199–206.

[RPG99] RAMASUBRAMANIAN M., PATTANAIK S. N., GREENBERGD. P.: A perceptually based physical error metric for realistic image synthesis. InProc. of ACM SIG-GRAPH 99(Aug. 1999), pp. 73–82.

[SS07] SCHWARZ M., STAMMINGER M.: Fast perception-based color image difference estimation. In ACM SIGGRAPH 2007 Symposium on Interactive 3D Graphics and Games Posters Program(May 2007).

[WLC∗03] WILLIAMS N., LUEBKE D., COHEN J. D., KELLEY M., SCHUBERT B.: Perceptually guided sim-plification of lit, textured meshes. InProc. of ACM SIG-GRAPH 2003 Symposium on Interactive 3D Graphics and Games(Apr. 2003), pp. 113–121.

[WM04] WINDSHEIMER J. E., MEYER G. W.: Imple-mentation of a visual difference metric using commod-ity graphics hardware. InProc. of SPIE(June 2004), vol. 5292 (Human Vision and Elec. Imaging IX), pp. 150– 161.

[WPG02] WALTER B., PATTANAIK S. N., GREENBERG

D. P.: Using perceptual texture masking for efficient im-age synthesis.Computer Graphics Forum 21, 3 (Sept. 2002), 393–399.

[YPG01] YEE H., PATTANAIK S., GREENBERG D. P.: Spatiotemporal sensitivity and visual attention for effi-cient rendering of dynamic environments.ACM Trans. on Graphics 20, 1 (Jan. 2001), 39–65.

c© The Eurographics Association 2007.

