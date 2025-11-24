---
sourceFile: "Gaze-aware Displays and Interaction - SURREAL TEAM"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:11.425Z"
---

# Gaze-aware Displays and Interaction - SURREAL TEAM

89769b90-43b6-44a2-9866-e95945720176

Gaze-aware Displays and Interaction - SURREAL TEAM

15a64192-b97c-4f06-8c02-6150809fbde6

http://surreal.tuc.gr/wp-content/uploads/2022/06/Gaze-aware-Displays-and-Interaction-compressed.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOpkrGgrVz1KPdALnqbO9IDfjWUnSPScl-pxt29uQTny0g2skT7CHbi2L0i_F2xOQ7Ud-uw8YZ5GQ_zz-vfbNVlGXPtx8LmeIFXYLQRas-738l57gxZuCYlVREK0qzrfU7LK8MvLZA=w275-h257-v0

a8cf7eb5-e0a5-41ef-a5a7-6827b0071b43

Gaze-aware Displays and Interaction

KATERINA MANIA, ANN MCNAMARA AND ANDREAS POLYCHRONAKIS

Sample Course Notes SIGGRAPH 2021

AUGUST 2021

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the Owner/Author. Copyright is held by the owner/author(s). SIGGRAPH '21 Courses, August 09-13, 2021, Virtual Event, USA

Being able to detect and to employ gaze enhances digital displays. Research on gaze-

contingent or gaze-aware display devices dates back two decades. This is the time, though, that it could truly be employed for fast, low-latency gaze-based interaction and for opti-

mization of computer graphics rendering such as in foveated rendering. Moreover, Virtual Reality (VR) is becoming ubiquitous. The widespread availability of consumer grade VR Head Mounted Displays (HMDs) transformed VR to a commodity available for everyday use. VR applications are now abundantly designed for recreation, work and communication. However, interacting with VR setups requires new paradigms of User Interfaces (UIs), since traditional 2D UIs are designed to be viewed from a static vantage point only, e.g. the computer screen. Adding to this, traditional input methods such as the keyboard and mouse are hard to manipulate when the user wears a HMD. Recently, companies such as HTC announced embedded eye-tracking in their headsets and therefore, novel, immersive 3D UI paradigms embedded in a VR setup can now be controlled via eye gaze. Gaze-based interaction is intuitive and natural the users. Tasks can be performed directly into the 3D spatial context without having to search for an out-of-view keyboard/mouse. Furthermore, people with physical disabilities, already depending on technology for recreation and basic communication, can now benefit even more from VR. This course presents timely, relevant information on how gaze-contingent displays, in general, including the recent advances of Virtual Reality (VR) eye tracking capabilities can leverage eye-tracking data to optimize the user experience and to alleviate usability issues surrounding intuitive interaction challenges. Research topics to be covered include saliency models, gaze prediction, gaze tracking, gaze direction, foveated rendering, stereo grading and 3D User Interfaces (UIs) based on gaze on any gaze-aware display technology.

## TABLE OF CONTENTS

1 Introduction 1

1.1 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 1.2 Target Audience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 1.3 Course Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 1.4 Course Outline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 1.5 Course Speaker Bio: Katerina Mania . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 1.6 Course Speaker Bio: Ann McNamara . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 1.7 Course Speaker Bio: Andrew Polychronakis . . . . . . . . . . . . . . . . . . . . . . . 3

2 Course Slide Deck 5

## INTRODUCTION

Being able to detect and to employ gaze enhances digital displays. Research on gaze-

contingent or gaze-aware display devices dates back two decades. This is the time, though, that it could truly be employed for fast, low-latency gaze-based interaction and

for optimization of computer graphics rendering such as in foveated rendering. Moreover, Virtual Reality (VR) is becoming ubiquitous. The widespread availability of consumer grade VR Head Mounted Displays (HMDs) transformed VR to a commodity available for everyday use. VR applications are now abundantly designed for recreation, work and communication. However, interacting with VR setups requires new paradigms of User Interfaces (UIs), since traditional 2D UIs are designed to be viewed from a static vantage point only, e.g. the computer screen. Adding to this, traditional input methods such as the keyboard and mouse are hard to manipulate when the user wears a HMD. Recently, companies such as HTC announced embedded eye-tracking in their headsets and therefore, novel, immersive 3D UI paradigms embedded in a VR setup can now be controlled via eye gaze. Gaze-based interaction is intuitive and natural the users. Tasks can be performed directly into the 3D spatial context without having to search for an out-of-view keyboard/mouse. Furthermore, people with physical disabilities, already depending on technology for recreation and basic communication, can now benefit even more from VR. This course presents timely, relevant information on how gaze-contingent displays, in general, including the recent advances of Virtual Reality (VR) eye tracking capabilities can leverage eye-tracking data to optimize the user experience and to alleviate usability issues surrounding intuitive interaction challenges. Course topics to be covered include saliency models, gaze prediction, gaze tracking, gaze direction, foveated rendering, stereo grading and 3D User Interfaces (UIs) based on gaze on any gaze-aware display technology.

1.1 Motivation

Gaze-aware displays, in general, including several Head Mounted Displays (HMDs) with inte-grated eye tracking have recently hit the market. For gaze-aware displays to prove useful, it is essential that the community has an understanding of how eye tracking measurements should

CHAPTER 1. INTRODUCTION

be recorded, analyzed, and reported as well as how gaze contingent displays could be exploited for rendering as well as for interaction. It is also critical that the community can take advantage of built-in eye tracking to advance Immersive Virtual Environments using novel techniques such as gaze direction, gaze tracking, stereo grading, foveated rendering, and beyond.

1.2 Target Audience

The target audience for this course is researchers interested in gaze-aware displays either in relation to gaze prediction or applying eye-tracking in Virtual Reality. This course represents a birds-eye view of research on gaze prediction, eye-movements, eye-tracking, data capture and analysis, and state-of-the-art applications of eye-tracking in VR and beyond. Those wishing to grasp the theory and practice of gaze-contingent displays will all benefit from this course.

1.3 Course Overview

Gaze-aware displays exploit gaze information either for optimization of rendering or as a means for interaction. The integration of eye-tracking and VR reveals where the user focuses their attention. Content creators and world builders can exploit this information for gaze direction and interaction as well as alleviate motion sickness through stereo grading so that the VR experience is comfortable, safe and effective for the user. This course provides the necessary background and overview to be acquainted with gaze-aware displays towards establishing gaze tracking as an industry standard.

1.4 Course Outline

### 1. Welcome and Overview

### 2. Gaze Prediction

### 3. Gaze Tracking

### 4. Gaze Direction

### 5. Foveated Rendering

### 6. Virtual Environments and Eye Tracking (including an overview of hardware available)

### 7. Special Considerations for Eye Tracking in Virtual Environments

### 8. Summary and Future Directions

### 9. Questions from the Audience

#### 1.5. COURSE SPEAKER BIO: KATERINA MANIA

1.5 Course Speaker Bio: Katerina Mania

Katerina Mania serves as an Associate Professor at the School of Electrical and Computer Engineering, Technical University of Crete, Greece after research positions at HP Labs, UK where she worked on Web3D and University of Sussex, UK where she served as an Assistant Professor in Multimedia Systems. She received her BSc in Mathematics from the University of Crete, Greece and her MSc and PhD in Computer Science from the University of Bristol, UK. Her primary research interests integrate perception, vision and neuroscience to optimise computer graphics rendering and VR technologies with current focus on gaze-contingent displays. She has co-chaired technical programs and has participated in over 100 international conference program committees. She serves as one of the Associate Editors for Presence, Tele-operators and Virtual Environments (MIT Press) and ACM Transactions on Applied Perception.

1.6 Course Speaker Bio: Ann McNamara

Ann McNamara is an Associate professor in the Department of Visualization at Texas A&M University. Her research focuses on novel approaches for optimizing an individual’s experience when creating, viewing and interacting with virtual and augmented spaces. She is the recipient of an NSF CAREER AWARD entitled ”Advancing Interaction Paradigms in Mobile Augmented Reality using Eye Tracking”. This project investigates how mobile eye tracking, which monitors where a person is looking while on the go, can be used to determine what objects in a visual scene a person is interested in, and thus might like to have annotated in their augmented reality view. In 2019, she was named as one of twenty-one Presidential Impact Fellows at Texas A&M University.

1.7 Course Speaker Bio: Andrew Polychronakis

Andrew Polychronakis is a researcher and PhD candidate at the School of Electrical and Computer Engineering, Technical University of Crete, Greece. His thesis focused on foveated rendering proposing an innovative ray-tracing rendering pipeline for which foveated rendering is applied. Acceleration of path tracing reduces the total numbers of rays at the generation process.

## COURSE SLIDE DECK

https://lh3.googleusercontent.com/notebooklm/AG60hOrYak3whaMSELVmKRt6fuoW-Ggfah1XdCOdxaoFdxblFNF9Wv9-eNSrH41Apoy719fuhYjmeU2w0qYP9x6KriY8lZzcOUVkPmFqAQf5ANH73CgS6AEW7GVY_G_et8Vjgn9ZEjiUGg=w151-h36-v0

84695064-ceec-468d-8b45-36a05114aaac

https://lh3.googleusercontent.com/notebooklm/AG60hOreVtNiyOcvSKiQ7QjDuPWcI9tMHxp-GUjfXhgTXG2UVyDrbJsEFEfI7-RdZRNkj7MV4Kw4YwwqldRBlU7dUjIYcYybT2iHshy2hky6x87dQDQ1oMx2P5e3IXQFHFMHpjbU8_wf3Q=w116-h28-v0

025d3e28-358a-4531-9ec6-a852dadbc506

https://lh3.googleusercontent.com/notebooklm/AG60hOoYhLnSksZablQqn91Aog9oLDJKi_1rkaSFq5yTP0fCY83zZfADtlJOZ7BQQp_U2ytfhTpMifzXTWSmjolhqSGyTAwIEorIpnFdhhoFRsvL50Ote0ttDpcm8Oy9yTmfmmo2lkvn=w38-h28-v0

e0a6057f-3092-42f8-98a9-11e636851f24

https://lh3.googleusercontent.com/notebooklm/AG60hOrMt0qtl36zrW1BIHNn245aBVc0yXkDUYu1lULYqrQloWoEb0oS3Rw9ID4-zKik_dvmJUG76fCNPSoFufW-g7bqet70u7N1JNzCsA8L_EsqZMhBnFPCugY12tvjqNO2AtIJYtX6LQ=w38-h28-v0

f93c4ce3-b9b9-4b91-a040-2fcab39e3803

https://lh3.googleusercontent.com/notebooklm/AG60hOoWDQUyCXw7txk7fekUNkJLpTdRIp87CP2VQMtt2d7HiQ5mmgfal1uFqFnS8S_jEeSwZ1R4HXBjAtEOonxXjuYOsiaYdxsS0J7c0G76Dm6Hf4-WWgVwfApFGDaAAAgHMwz-3yOf1A=w38-h28-v0

e851249f-4703-4866-9af5-6d3ae53113c3

https://lh3.googleusercontent.com/notebooklm/AG60hOoh_2WYtlV3V_p03Ga6MwOouZpafIshN0ivPxLcxKhGlb1Qso2eQepV3Pklv3wNh9hnmKDIm0rQkkveeRSgqWxARTzjAmCemHrV1blRjbDdllVEA_TSdZAnFEeuClsPituiB05jEg=w38-h28-v0

8033fa90-3539-4bb7-a77b-b1c736e93f84

https://lh3.googleusercontent.com/notebooklm/AG60hOozA38j158fXG3hYo5UfwQhIb2nGxwSSbrVV1q6s2rD1lWH0BGpM2wSZIdFj3lwOjEwZzYf12sEbFZSUa_o39RUaT2i9RxOnQyrq8Pa_viUbyp4DL6UGYbH0EapCxP9YcV22_uspg=w38-h28-v0

06302a83-2873-49ff-ab69-de6a0e6130eb

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

KATERINA MANIA, TECHNICAL UNIVERSITY OF CRETE, GREECE

ANN MCNAMARA, TEXAS A&Μ, USA

ANDREW POLYCHRONAKIS, ΤECHNICAL UNIVERSITY OF CRETE, GREECE

GAZE-AWARE DISPLAYS AND INTERACTION

© 2021 SIGGRAPH. ALL RIGHTS RESERVED.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## COURSE SCHEDULE

## Welcome and Overview

Gaze Prediction, Gaze Tracking, Gaze Direction, Stereo grading,

Gaze-based UIs [Katerina Mania]

Foveated Rendering [Andrew Polychronakis]

Virtual Environments and Eye Tracking (including an overview of

hardware available). Special Considerations for Eye Tracking in

Virtual Environments [Ann McNamara]

Questions from the Audience [ALL] 2

https://lh3.googleusercontent.com/notebooklm/AG60hOrf5FhF-5r-1utDuT0rqQA99UrApbvDaXzHIgaW68PtBIZeBBTpd4zPlsvOExtEiIz9q-a3PCD5MHWK-0L3RBrPUOa9S86fivXOvnBZA-UBawuFSTfiVvie9HY-c45CwfabrUvdMg=w151-h36-v0

045ef217-3562-480d-8a26-5ff0d0586a70

https://lh3.googleusercontent.com/notebooklm/AG60hOqg3aYsHjbmgxq-ttmBRhIU2wvI5No68Z5HBlHgLpHyol49rB55liNxvEC0ripiS1A50kMNwwkC2TtCm4-fkRaGhC5JrlXpaNRIyHpxExo8PU6AZs8fUQxM9RtRPpz0-31M8542jQ=w116-h28-v0

733287f2-986d-4704-9d6f-54cf8ff860fd

https://lh3.googleusercontent.com/notebooklm/AG60hOqBtJsDvKD-CexDEIwKwCC17KbCMoQ1QZUthHsSQ6dipgLfgJAIrvVlKJUzj1cRPy6OqX7F_9zoKuPaBCUZsG--jNViqDTgjKCsxQolJZ4mQOG7jdrXmsf_IfsaHhtnEKEp_CpkZg=w38-h28-v0

c930b681-26ef-4a6b-8728-618ac73234e6

https://lh3.googleusercontent.com/notebooklm/AG60hOp-cVE6n_DwAjsiF2DJDT7pEeOsGBuZyQzXGaDA90hF6qg4ddiVyrdG9tyLaRDN7iZxVn_KL36RgQh4QieDaiZAJ9Bb1jA316m1xE1TEKa32US-T6VZcqztSD6xHQx25SAhm_-M=w38-h28-v0

4860cc62-ad2e-4dcd-b839-ddfac3562158

https://lh3.googleusercontent.com/notebooklm/AG60hOrYoWYazRqS23jP55tQrqMGNDmfjRqjSVnrxo3uskpM-SI5XRjZ286awqDO2JnE7po5Wp2q26y8XdslXwW4SDifleCfpWuIf345BFZV310VQmR4Aj4P4RbcMZE1w1W1pZy-ANQu0w=w38-h28-v0

c8f7b17b-d55f-44b8-91b5-c54d3f13e593

https://lh3.googleusercontent.com/notebooklm/AG60hOoKpFHv3HsLIm3oTJQk--fhRkCoxn4JV7WFAMnPaMg00880HslQ6ZmU_DMzdR1JpdhRv8m4WZZ3oQkMeCn4SpyUJuJCvDPUA9Y47CgnLYv7f4WYlnh3GLEwZ7bsPZ-tTYsF_blkPQ=w38-h28-v0

d35d8682-6587-4286-8b26-9e988bc78ea7

https://lh3.googleusercontent.com/notebooklm/AG60hOpCIcZopETxeComuiLBaljdl4uTCAlcBeOwmUtrIMlYZR167XuxltdZ_ixgnknkhII6p03C39GUBs4HX-fAWW56HRKFKM5NRa-EvolSXSONJ2EItqkaYg8M-LheALX2Vg-SebOMKg=w38-h28-v0

4a394247-2825-4642-b11e-30a219f00630

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## KATERINA MANIA

TECHNICAL UNIVERSITY OF CRETE, GREECE

GAZE PREDICTION, TRACKING, DIRECTION, STEREO GRADING, UIS

© 2021 SIGGRAPH. ALL RIGHTS RESERVED.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## Gaze prediction

## Gaze direction

3D Uis based on gaze

## Stereo grading

## Conclusions

https://lh3.googleusercontent.com/notebooklm/AG60hOoQ2BvP888GYQ_yNEj6b0IcFep5MzoFhM-JvePwU3w1KuDrRJ1OO5AwA8dkq6FQExAN4ZS1WFGZjfxf7tn_F4KSw-IndHGsmapsfPDMeu5odNFDnDIKDKMyXjGTt5MEPljR41j7zw=w151-h36-v0

e5f2baab-9043-40f6-9113-00727581232c

https://lh3.googleusercontent.com/notebooklm/AG60hOqWOpD_zu2ViXncJA1MMHzy4Opiok1v1vEXUKSBDX46hH_3vC7V2jbrx2FGTYChcx9q-WHnGDIcoJ7-QHXbJpvXiqObCcN5Z3pCPYxmDVdm8T59v-IPsHOwqQmvuWmQzQRmtIxGTw=w300-h359-v0

67a6410c-7a30-4d20-80b4-5c4c142b25bc

https://lh3.googleusercontent.com/notebooklm/AG60hOom_p2CJf4RVs29uyYOvZWE5QmGP8ehG5C3ZEJXpxv8jW5cPAuZnEeJMFyw7L6KgwuCba8E2EwqPvB0buB4pm-yWeNkvsYOoniWJ3wSTjCHB9V6gTI5Id-C7mUd0rhX4bhgvVhnlQ=w116-h28-v0

f22b4f08-c949-4ac9-ae69-72889b5be471

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE PREDICTION

LOW-LEVEL, TASK-BASED, HIGH-LEVEL SALIENCY

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE PREDICTION MODELS

Gaze prediction can accelerate image synthesis by reducing computation on non-attended scene regions

Controlling the level of detail in geometric models (Zotos et al 2009)

Gaze Prediction of saccades landing positions to reduce system latency (Arabadzhiyska et al 2017)

Saliency models select the best views to scan indoor scenes in order to produce 3D models (Xu et al 2016)

High level saliency models to optimize an LOD manager based on predicted gaze on objects on mobile platforms (Koulieris et al. 2014)

Automated high level saliency prediction of game balancing (Koulieris et al 2014)

Predicting tactile mesh saliency (Lau et al 2016)

Eye tracking-based, low-level, high-level, task-based

## Gaze Prediction

https://lh3.googleusercontent.com/notebooklm/AG60hOpQxo7_ZBkJgTzm0edgr6cmw4o6aR8ML9Yjb3P9qk_hPPcX1DUImDmzo6rjcLKQBiusbat3Vga3DEB3D9U8-ES4jkBRq42LSPToxvgeGm_Y5WEB_OZIeyKFlns3WAyBRpgrcvTWRw=w116-h28-v0

5c39e1e0-b276-4d71-8670-0d071bd84caa

https://lh3.googleusercontent.com/notebooklm/AG60hOo5H71GHfRxqIE1nijEy6b6tfVztChqbzZ-Uz2SCyhZJjy1EobC8_BvxWfaW6-JXVn7Ka7NwvdsCgR6ZsOud2BxDoTXgPOY1loDQ7Bl2xCZFVs5FxSZPQ8-cST3IjXDy6yHq_Ig=w603-h88-v0

d5d2426c-df5f-432b-b3ae-7564a3bfa145

https://lh3.googleusercontent.com/notebooklm/AG60hOrJhum7-HWRVeAjnFdEyvNufKsxBB2AuDm2hyqYNiovJTSaOaRvMupu_ejALZjpTgd8VJZE7zOX5Y_JibGyRowB7zM3M2A2Mbv174kaTgIkzwvKOj1QXh3VkQ35uwwF32WWM0Z_=w50-h26-v0

e976df03-820d-4def-a829-4dcfb7adef72

https://lh3.googleusercontent.com/notebooklm/AG60hOoDaZiNRCgWPLTvJ_E7UPCvard7WY-769-TrSoUaNjjKAyyLAV5oE8HJilWlAdtEMm9UN-Z5LPGGg_4BY8knCJO5xPX3yKqbWjz2R7D7gq-II1qK3S06BPe3CHFQVQv0XicnbPq=w48-h25-v0

99e22aba-e6d0-4d63-b348-4c9634c0f220

https://lh3.googleusercontent.com/notebooklm/AG60hOqW3dH16k-y2JVfJUGrWmfZWgxAs-ErE4ArfYy-H8O1bMyJ15f5kucFNteOBJk82PrvRQ-NDHL3v-sFco1ho-dTyQ05WOIuOgE2ixf4xP2T4pkvRK_0UhkwGRzhdnXRLq-u4IfPiw=w20-h26-v0

3857d344-08fc-4259-833c-011805bb0efd

https://lh3.googleusercontent.com/notebooklm/AG60hOqLddXOXpcR4n6Z9aRLujEdx6s4XYbAzeHvKQYs_MIBIK3LN6AwtK_nRjMpoWBhjxYOT2rIdwE9f0XxC7WXgNzzDKySDa8GR4LK1Lool_8IIIPApqf0FtWEJ5SDgDA6VE2iuaQ8IQ=w19-h25-v0

a104f62a-1c13-408e-aa2b-a78c9f1d3c97

https://lh3.googleusercontent.com/notebooklm/AG60hOqDPx-GVbZtRHzWBg4dJWd0fozUPf4iJt9leo1KpLIBVzRL--8xVioe7N4nWwTYAriq3kqH8_TeGYPAFgtjxHZOChwOX7BwDWCNKhmJDP7V5OBowt0siP-UHwfwbzvxpJ5IiIrt_g=w31-h26-v0

7579e0e4-9a43-4b9e-9987-9d448c5f0c5a

https://lh3.googleusercontent.com/notebooklm/AG60hOrBFg4W-TbAXRC2B3F_u06zNOO9xwlICNYW9Jgu1a-GLI8img1ksZMPWdAEwn9tVrfylun4IdIhtDv4AztM8Qt6MuK8nE3ksdyK9EPEWCyzH6YzP-yh6eJc2gdWn2xeqWgG-Iay9Q=w30-h25-v0

087b9f6a-6607-4ff9-b078-6c47f2c59562

https://lh3.googleusercontent.com/notebooklm/AG60hOpa3B6nYQoelDcr9PZMOQCx8e3jEH07QM8nAwKK-Oz0EzEznKRzF18VG5WIZtJZTxMpFWoGV3bOi2pPATjjS5BQ1bv77qqcPFeUGE5aCvDxlmWz9eipgxtdbZK2KsEGolzDgLFjnw=w20-h26-v0

42e571b3-128b-49af-88d6-2b3ae13b7c54

https://lh3.googleusercontent.com/notebooklm/AG60hOo3lrlxpVXyHGgrf8kgXBAd6MdhvzXc54nvUlgTf24RlQD6WfI9bllD-5ouuE4iqyzhWv2-ZCo-L3Q9c2fjdwF4urEEQQ9eGQe8MhAUWVtpj_JQc-KN02rp_xHej3MbtakSLcKeNw=w19-h25-v0

d043ed25-f9aa-4d04-a1b8-293b8ff40307

https://lh3.googleusercontent.com/notebooklm/AG60hOpJEkK4BdNgtgueiwRyNlpmOkLHNeCHt1HA51Y0zXaqNQvOU6V35cPeQsYAPOR4DmlwrgreUv6iPYF1VnegJAVw53cX6gagvymt8b9e74KweKLh095mF2VO96Jd2DTz7jVy9twXzw=w53-h26-v0

a2500b9c-1f86-4811-a3b7-0ab3699960f3

https://lh3.googleusercontent.com/notebooklm/AG60hOpjKP-3R3tByUzo-AFWVQpig-2Mhx44oQ58oC6pHpPd0sZYHRvb33qd8O-Lr6PSRF9IpVtQtRC4pa22FAGggQqBQDG9hpy0Mj_t0vOC8Uz39LOypltKktlouwwCWToohTwyNx_-=w52-h25-v0

2333fcd4-9e24-489d-9599-29964de369c9

https://lh3.googleusercontent.com/notebooklm/AG60hOragfaaOGXDiIKGmJXDLdewJyqmOzMf5aEf6XJ98flOCXSbK4LjCo7rsSLkXMEXsRh6qAz6fTSKw5o15f_yKsH1gvvHxpWY6GkIoD4QhWLiP0-vsNvLmfYvtAMYq-gbIxq4gv7L=w116-h28-v0

edc86807-8789-43f7-bc50-109a7842513c

https://lh3.googleusercontent.com/notebooklm/AG60hOqrDYNeDtCnYw71hyLeVIM_APv460TG0n6Emk8l3ESkIl8zw6xIQFLaWTSg1utOCID9i2Icc0wk9i9x-zInx6wuVI0Fi3NKc2NTiHDRMHlacekmH6qnqgUU9aBsZkrJxV3qrv7PuA=w166-h26-v0

3084d23d-bbcd-4733-a481-f3a7051f2073

https://lh3.googleusercontent.com/notebooklm/AG60hOqbe6eE2azO_5TVF_AznMLn_1x1lw_7mty5nrswh_enwJp_39R5lv9CI8ZgnV_ZBSbp6y7zp9ah_37YJdY2OeW6-1MPqm4avoOAN0ed7KsbEV4CTpRrRRv74PErh1BCdEFr__x-=w165-h25-v0

fca243e2-0040-49e1-ba05-1846baf157f9

https://lh3.googleusercontent.com/notebooklm/AG60hOrJLNnjDmjqog2NYmBeLIH5BEWa3B4oc-GyLhYvTTC4BAW57hrmH_eaRiKdgPFJIw-AkRbc-eO34tBYchouxcS15xmXH0G_240lclzV6FUDRQ7bhuELUfpXLOtkRZfuW_bOyuKCKg=w297-h46-v0

2c21f85c-3ac7-43b9-874a-8303fab574cd

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

LOW-LEVEL ATTENTION-AWARE APPLICATIONS

FIT-based selective rendering, important parts rendered in high

quality, remaining areas rendered at lower quality

FIT-guided selective rendering

Images from Longhurst, P., Debattista, K., & Chalmers, A. (2006, January). A gpu based saliency map for high-fidelity selective rendering. In Proceedings of the 4th international conference on Computer graphics, virtual reality, visualisation and interaction in Africa (pp. 21-29).

LEVEL-OF-DETAIL

Gaze Prediction – Low level Saliency

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SIMULATING GAZE BEHAVIOR

## Simulating gaze behavior

## CHARACTERS AND CROWDS

## Extracting scene interest points

Gaze Prediction – Low level Saliency

Image from Grillon, H., & Thalmann, D. (2009). Simulating gaze attention behaviors for crowds. Computer Animation and Virtual Worlds, 20(2‐3),

https://lh3.googleusercontent.com/notebooklm/AG60hOryPD96LReLj-CzN5B0UpVyAvIM44aWWRHiRluTqcsgkO57fBXFkxoGn4xig0Gi-8xxyfkVHvH1gq8lQ3-PHgvRCbJnIFMimi_8AhmlJhhRhXvYLf9UDqi2S-2m48viO-9Gv8l2Og=w116-h28-v0

4a6b190f-f3e1-491c-842c-84fe0a1dc0ad

https://lh3.googleusercontent.com/notebooklm/AG60hOoVWr6rxbbnOtJOdk0O1KHFNKHpGyvMfqQdXAbMRUyzPDMyc1R7dD5vm3FITBYlCD0c3JZPJRefOyVGwyHbOIfd_p63Ip9FPzTW1Eq6rTzmezrY9GwBBcGe1qqiUurAvSXclyeB=w108-h26-v0

f4ea03ef-296f-40bf-82af-c7727b4fafca

https://lh3.googleusercontent.com/notebooklm/AG60hOoKys4CTLLKeI3KOCKx7kZmz3jPla-5y4-5M2m91vmjACdl8pkskNmCngNB1AYy3DQ2t9Rnu2ciT9tHcJ7uXTzvk3rV5ECN-gnPyesNFIPBbLooeVID5joZIcBojWx1ydV1npwZOQ=w107-h25-v0

f195d063-3cbd-4714-b929-999f166ea5ff

https://lh3.googleusercontent.com/notebooklm/AG60hOqk9-6Dszf_hUi-RWTdtSsYG661rTF9YwaLY_3nrYS3lyt2HFAqqaa3HJiuXjN3rPzRlRW2Zyzvc3YzyEZQ0-R9iVr3aBFRL7yLDuVwiaBeO2kMm4qeJsroaTU3zjzqBGzyWXsQhw=w20-h26-v0

91331c58-7dcc-4f29-9c14-7a1086da19ce

https://lh3.googleusercontent.com/notebooklm/AG60hOpqJsH7Vz-KhpM8IDNaMKwQ1BEYwpFUAMqlGp5B6d722K8xQ5J20SApDkGg-WE0CPNLz20g1ilfi3UQE6ToTKLgsm1f0vXSst9C_-OSIFgY9msXbdpVsE9C5EMD1k-rGjf9fIeg7w=w19-h25-v0

f908a576-d1f4-4b12-a993-9cdbf032c38a

https://lh3.googleusercontent.com/notebooklm/AG60hOpSuDnhI6o9e-Gta5SW3bfME1MXT_9aHrLHftX9DpSGIFrqhPKouF38RLol-LtkzlkUeuJcvqAhFtl7_cXSFxDmM8zonz_WKqVJWSsQNVQtRA79oTT5pGO-iypTM_EZRoaldo6JZw=w255-h26-v0

04c957c5-964a-4ac6-88c1-4491ed6ba93d

https://lh3.googleusercontent.com/notebooklm/AG60hOo2vEV-XWxDgJ9DJUT5MLKia4dlZkd8aD_mRV1hd0gCT8hfuQjJjnzFbNM-Eg-k9zftrh2ubGbl8_BJ9d6ugvrOyZCBvQ4mfetJJ33aDqELPEZFDmuoUl7v632ZoXmbMB-PzYgjmA=w254-h25-v0

bee7b8d4-10d6-457c-b80b-36d4086e46ab

https://lh3.googleusercontent.com/notebooklm/AG60hOrlupupOHT99ftBsE3le0uqpzEpWNG3VHHj09RN0UoqwJHBvhvIW3OemEhJmDDcvRkwp9qMo0R4caSNXDyY4YlwIvsgoI-W9lNTyVLoIzI7M2I4pmqPcUIslqOcaXTPRhf971-BlQ=w234-h58-v0

f9533659-9c22-4868-a7c8-c57ef6071f45

https://lh3.googleusercontent.com/notebooklm/AG60hOo9Rv-AHz9RPujcTs5ZmxwZXXdjDBqysKRNDojAaFFI5Kcy66CAho51I4MLNNztM2SfMtE4VT-h7dt4t_7fI2vRrBGOsdZlQXC2NzFQIBrHcDpu8EKnIADMTAJL3JX9EmOwSzmAQQ=w232-h75-v0

8e75fd48-8220-4688-972e-eb169051a7eb

https://lh3.googleusercontent.com/notebooklm/AG60hOpGUkNsMK_i5vmlOE-vP7bVhDEsghRnE29lwzjeuV1ZW2H05proHMCJhAZKXqwg9OPwbyJ_81onUxEoD2t-l5Fk3LFqZt737XucE96UP3Zw4V58JfpjAirvH_Gsp2ysuO7-B3Vj=w116-h28-v0

7f6d4fa9-4158-4fb6-a396-fdde4292b9d3

https://lh3.googleusercontent.com/notebooklm/AG60hOpoygCySsTfhkKrCaoEqeCLMCTBSZkadk7dusFI-ZUIVVTbmZJYlmmUWXFDpzbS-gos9H-q-zrE_WCA13sd1B7-zvHJEn5imR8OqXxTEPbN1L0xJY1C6Ux505ECNZNSTYhlWibK9Q=w256-h26-v0

b1cde070-66a5-4c6b-8333-fb3bad0e298d

https://lh3.googleusercontent.com/notebooklm/AG60hOqCDbWX19Kdh9Z9ZolnmCkwZduNDpdxdYXM-YMdrWbOrdw_dkkFakj6DtruaLavOcWNcneq7_hENnhPJqqSbbqhOXbw-RrzBvTTWsvO2Vy669ODtd7LNNwVAnmktxxk7yIJLpc_=w255-h25-v0

7868ab1c-73f5-4fa2-a51b-5cc8f0779062

https://lh3.googleusercontent.com/notebooklm/AG60hOozMw6vUedfia7fe2Le38UJyFL3ChWryoQDSisiUYEbEZ5Avy1tMShT-I-vgTxXF88Ry2Id3_ITDlAj-CzHZ1Be9GIqtnJOwwR_gCGRXDGjPEbaa5qEREOmzNs4ctuWtM0xtYff-Q=w366-h104-v0

84b3c61e-f9ae-4438-9850-0360fec83df1

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

TASK-RELATED SALIENCY

Saliency models and task related data linearly combined to track visually attended objects in a VE

Demonstrating how the visual attention tracking framework can be applied to managing the level of

details in VEs

Gaze Prediction – Task based Saliency

Low level & task-based and goal-directed methods

Image from Hillaire, S., Lecuyer, A., Regia-Corte, T., Cozot, R., Royan, J., & Breton, G. (2010, November). A real-time visual attention model for predicting gaze point during first-person exploration of virtual environments. In Proceedings of the 17th acm symposium on

virtual reality software and technology (pp. 191-198).

COMBINING TASK-BASED METHODS AND LOW LEVEL FEATURES

## Navigation in VR

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## MACHINE LEARNING APPROACHES

Machine learning techniques applied to eye tracking data to train a saliency detection model

for pre-defined sets of static photographs (judd et al. 2009)

Importance map scoring gaze amount on objects, then as heuristic to predict gaze (Bernhard 2010)

for 3D Action Games

Gaze Prediction – High Level Saliency

Gaze Prediction Heuristics for 3D Action Games - Machine Learning on eye tracking data

Images from Bernhard, M., Stavrakis, E., & Wimmer, M. (2010). An empirical pipeline to derive gaze prediction heuristics for 3D action games. ACM Transactions on Applied Perception (TAP), 8(1), 1-30

## IMPLICIT MODELING OF HIGH LEVEL EFFECTS

Visually highlighting important objects (b) not just salient pixels (c)

https://lh3.googleusercontent.com/notebooklm/AG60hOpwnIPTMwp3ipaeopsj-6m8d-kgGBlSywL5U3t8n8zLlRtSmhKANVzFIHF68k2d-MgJGmERM49i0Ol28dvogFQ-Nq-ccOkclLjKDMjXtgSqohKEdPq0KU4dIIJYtfVs6TsoqnUDHg=w116-h28-v0

cd78a132-8644-45b0-8aec-65a878f65cf6

https://lh3.googleusercontent.com/notebooklm/AG60hOq3iuV2wjsT2CAeLbydQ_-vYHerTrD0d-NGr7ADwiBHqF7s8bEYJBAnXH5drShO1V_O02D-I1C8HmwrILW5OEDIrY3lE1Om36omiCxzvAWy7v-LoIVBCxk2o43AOWjYs9tj22wZjg=w372-h26-v0

8b8c6b71-b303-4bbb-ba6b-5dc4c9acbea2

https://lh3.googleusercontent.com/notebooklm/AG60hOrHDPdgpeHfgZQ9hGvOA9Gtx45F9McKlKodC_5y3hL2M_vmF2S4CS_WXVY0qNYfI7Xq5SNfsoyMNky3vHml_yXN7EwYG-qz3GJIdM816SSbFJIhWTch4ZR10SqGY5er-vkag86XAg=w371-h25-v0

571fa048-ab78-4d26-aee9-efd81180826a

https://lh3.googleusercontent.com/notebooklm/AG60hOqvbZpyHKpmcIpcPObC51_ZANMdUiqEEHHd9vaOS3Ui-QOZ7hoHWwjy8r5S3uGj3j9ShgW4hG0cjQyB3XKjW94B2g8luiSWQKMRdj4t1BNWtc6NtngvnaSJB9uA8UBEkSeitAQ2=w116-h28-v0

4c113c74-4b66-48ee-a167-44ec756f9a30

https://lh3.googleusercontent.com/notebooklm/AG60hOrrfdpngfArD_95DXQI9R9oqxeYapsViAfsXQERGt2T2SNyncwNVagepAo7i5lC3AcJNaN3cFXeU4BOCpt3V2-Iv5YpCXZmOa0ktX2oc5ilPbuw1-b4V_htafzlphiefhrcMDna6A=w175-h26-v0

256e2ae1-3d14-458d-8f37-ffed2d020729

https://lh3.googleusercontent.com/notebooklm/AG60hOo99QT-88lT_IiVDsuHkCXyZBnRWpC2T2_nqqUu7DXaKfx7J2M7EylVTZPWGjvA1a3OfT1g_dY2ckaqDW0YlIDLY6v_4y-28YgOu5VtZi7sSwoXNM5TGzL4DGkd2R2ZxyPn7HAQMg=w174-h25-v0

0c7b8e4b-835c-4f95-a337-8dc8097a551d

https://lh3.googleusercontent.com/notebooklm/AG60hOr4BuHbQPX4zdG-Y9tFB9GziPnE0tGwCDwOLC-2-LaXbKqUdJffEDBlGXZIkL0pO9_qmLlpNf_FIo2Pw5cQri0EiENoSwcz7fbaNPr09UT1_qSgadjSwd8b6EuMOXCcBfbQ0yMuZQ=w117-h100-v0

52d6c676-ce77-46ac-87a4-5cbd527cf5c6

https://lh3.googleusercontent.com/notebooklm/AG60hOrFxXqbOrzQIHUTjMo_Tclr9qHbLjEneK5drJiFryWltrdmPZl6dD3vl9ATtnRqq4FrThn52v1iWDsP_CHSOZJGisf5Z_LgfdY-XnL7M_qUd2w9C16eODsVQM0MQ3zmNVTWzmiaNA=w237-h102-v0

ba8d7098-4f4e-4a7c-8045-ab515fde0a2e

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## HIGH LEVEL SALIENCY

The goal is to define a computational model applicable to any context; a challenging task

When attending a scene, recently acquired knowledge from attentional processing is combined with pre-existing knowledge about a context, e.g. "bedroom”

Attentional processing via Gaussian combination rules. Every object in the scene has a set of bayesian priors on the areas that the object is expected to be attended

The chimney high probability of attendance on the top roof, low in other areas

high level saliency guides attention and weights of each hypothesis are derived so that

the model is calibrated

The posterior probability  is calculated that the viewer will fixate on an object independent  of task

Gaze Prediction – High Level Saliency

## A High Level Saliency Predictor

## MAPPING VISUAL REPRESENTATIONS TO MEANING AND SEMANTICS

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2014). An automated high-level saliency predictor for smart game balancing. ACM Transactions on Applied Perception (TAP), 11(4), 1-21 11

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

HSLM - THE HIGH LEVEL SALIENCY MODELER

Gaze Prediction – High Level Saliency GPU BASED IMPLEMENTATION

## Implementation

Estimates in real time a posterior probability term of attendance in a shader, based on viewpoint, i.e. an object may or may not appear as singleton depending on the viewpoint

## Identifies objects expected to attract attention

High Level Saliency Low Level Saliency Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2014). An automated high-level saliency predictor for smart game balancing. ACM

Transactions on Applied Perception (TAP), 11(4), 1-21. 12

https://lh3.googleusercontent.com/notebooklm/AG60hOqJN5BhEvdE9ZlBGJqdFOFafU-dnmyg2fINBmrYW26HIcQ2DhiydRJaGVtEbU6xWd9v20y4Xidp6yk9RJ5330srwxTOYshN6Ak1p9Mus-iDR0rEYOEuxAysBqMbIZ-OLsdt2rqvGQ=w116-h28-v0

008fb09f-5f82-4a1b-b66c-6a5230a0d2cb

https://lh3.googleusercontent.com/notebooklm/AG60hOppf-yP7KGHKqWH3UrebKrKaD5rVhCYuhcGs0lwoCI6yZhto2FUbbyeZbSIjdqMCnJZWwW962qtDZ_wVAouZW1iMjIVcQzGFL71X6EN4Pce6ZApAOpG9wIAclPqta8K68JwwqVTTw=w115-h26-v0

fef1196e-07d8-4345-8e1d-15655d0b8095

https://lh3.googleusercontent.com/notebooklm/AG60hOo_U1hVf5R_0pFZaXku6r08-kfRr5RjXDyW2DpZpPt2-uuhf1ULNpy0sBII4vBO7Jtt1KbRhhlGmVn6JnUozaYjAJnyuRg7VX9sA5tVIuEqI_c36b3-e4f13hlGh1lIWIOX7GjYsw=w114-h25-v0

19e6b988-b34f-4ac3-84e1-a9ab1f278dbf

https://lh3.googleusercontent.com/notebooklm/AG60hOqf-GTFzfNzCl20tmM7B8wy6pKW8kDQn7C8ybtJd4QkpJvj2B9powGeCCZpJMAB4Ug5qxWVwJWL-LAdjol5Ca9s6baBaMDww-cHnQc4BazhRIU5fvQs4jybJNJLwMWJtKqA-FHePg=w20-h26-v0

73cb381c-f803-4f4a-8046-594706d74efc

https://lh3.googleusercontent.com/notebooklm/AG60hOrxWo3KAIbcQmbosUZUPOv2B82ROnLBWM-gAXLCysvd_vQ4Fv5_B0ueMT1vaMPa8X6t4xcsfZZ_n4ZHRoahR97VcVBZeC-PRzSmu_mKkqJnAuCRQrUReLDAvO6H_yDNEdkg6xhwhQ=w19-h25-v0

b85bfce4-c3d5-4ae0-a20c-5e7f9765dd87

https://lh3.googleusercontent.com/notebooklm/AG60hOqr3A0ZIH9lab529MxJMI_LRPbv2txa7cD_8yGxaBOj7Xhwk2UsHwGQQ0-y4LQziIGrIB9ddgY8abANdFmaqlefR8BhI_sf-27IgFcZPXNUrBqEynQXyHsQDng4QQyYBz7xRGooUg=w38-h26-v0

de374369-259c-4747-936b-71e248adc00a

https://lh3.googleusercontent.com/notebooklm/AG60hOpkNBVZqh3hyKLsIKW65RyzLklCXUYFvtl53zEjDdk2KktQ70A1jxsKBJozq4KSOjY_aRoKMR4VLOkHbtFugyDYGj9Cv_fZXczEBSgD5jkd6xWDH8FM0pmQXBHJ2-cE7uCZFkT2ug=w37-h25-v0

9e9319a6-f09a-45a2-abd4-8d3b888842b4

https://lh3.googleusercontent.com/notebooklm/AG60hOrvI8kQboU-tlsmgk1yTtBEEM_i_D-dSY-hcH115kumCXJXncdtcXopijicBjiLN7Ed7_cm_E6O1w_HEqdGwCnQcaHlhbJCsLSHLbrradIBDrVf4DOXZJrGllw2LWL99LhMBcKHZA=w20-h26-v0

1d563d28-acc9-494f-91cc-44718e7ca260

https://lh3.googleusercontent.com/notebooklm/AG60hOoZvciMzVVWhkJZjzinkzGFX5it8t3cOnH3jEooXF6Qz9-vHgWtEzP5DtJ-RrMIv7jsJHW_38M4F5riD-4s0FCqpErd0Yoj7fY7VH5d_y52GuKK1sm2z0BH4_xaSYvKYFqQDPnQ=w19-h25-v0

94e58254-ff85-4c23-949f-ac54c18803f4

https://lh3.googleusercontent.com/notebooklm/AG60hOq-YPPtnpmOGlG13zHN4ZOhDm9VrI7NYhmoGQa55RfHB_uyTEAejbfcNNIll3lHWdF2LlkKdqsmWmWjPghpY3CQHbttdsYor39XsoZaP53RK2uenYyHn2EuoD8kx7NZc5vmp9OZTQ=w164-h26-v0

185212aa-2827-41d6-abaf-03e462252c01

https://lh3.googleusercontent.com/notebooklm/AG60hOqDpVHYRxdwFOeScMJ9yMdM3C0Q9vV9TTH21_E04voip6P-2lh3h107ivrcAj6enMdkkVzeYFqShZbUN18iq4MVda1l63nNJ719YgmykaSxfPh_3xV7yEUwErmaxFr33pb-i_2OBQ=w163-h25-v0

16b37e26-da5e-4ef9-95e7-23f6d7992c9d

https://lh3.googleusercontent.com/notebooklm/AG60hOp4_i9yJKUY6iZrkY_YsvaOe3HIRnVHi9OtdT8bKo6OJfR5nUh-HCf1KkLpCoYNFs9-TqXhehr6Ev2Njj3ggeJ4he8FV--vEjfM27M77zuc0Y1J9pQe-qV8NDwg1kE4s3A5U63FHQ=w278-h86-v0

50002b77-b438-4af8-86a1-421a93e69aa0

https://lh3.googleusercontent.com/notebooklm/AG60hOrdEepvM4xmJHq8R6rcWE9lm2HwujNvcsptrOmsrpacEEvVvhe6SImhuNIVvXCaHmhZ_gat7_2sSNB1z3M1JNMdHzsANVGt5DVWPahp9GEDZnMe5BGXaLfhIy62iSISJsb2M8USpA=w116-h28-v0

b4b72432-7b6f-4ec9-8732-9d6463026bbc

https://lh3.googleusercontent.com/notebooklm/AG60hOqnZw5s5FpQCsCzDItIz9n8biQ9DhGnu9QPRPzH00f59ZKM5B2ibGh-9zn6NWEjs1S59EzQkbiocyA5KYi582N9H-99MKNrolv1cPS505E5bIB4rCDYWkRykqVnmiqhOIkWfR1erg=w24-h26-v0

e63c04ac-0bc1-4bd2-9062-32f90d79b03c

https://lh3.googleusercontent.com/notebooklm/AG60hOoYhCfCEgBKKJykvtG8x_qRPevqB7Whw0B_zILdGz8SzGx3R_0lvPNUUVOrzKZYhIbGwSR-YbZrYkPDIUk3dXtOSSnGxRzpUADJDK8qVo1ex37ASeeLAvioZIugiddJb3F60fSjPw=w23-h25-v0

4b680256-4a91-4cce-a500-bd7c6eb8d549

https://lh3.googleusercontent.com/notebooklm/AG60hOoRYmOH9Q5kS4--lYkQUmaacc37EQ7RwQTCkNAICqvz77Fsve32tz-75kqhOzM6jrIWwMAijBWb_wumDfgEbOzUpEsC4yyB4acg_P6SyQmqWeqzZy27-tYEgx3Si0FebB61-LIU=w20-h26-v0

f1282b09-1413-43bb-808d-c6bbc412e1ca

https://lh3.googleusercontent.com/notebooklm/AG60hOrSjK6a7USsQ_rpLE8Of9RnpbnJ_HkO63U4fYCWSxfrApjMyUmqyDHZUMTzirvYSzk7eCELlty6WWcpbYEVeu1Cpi568nCWA2-aHdWqMYIOKdrjsl9zav0fxY3SupbwglfhYI9LHQ=w19-h25-v0

84edcc28-5c4b-4fca-a28a-c74ef27144cd

https://lh3.googleusercontent.com/notebooklm/AG60hOpjMZz2Ii6gCf2b__Jz5M7SUFkowhTD7_RdAH_Kgj7P1VIABHd95ca3MGOR7fFdIaIakwwB8N_NGWfVrbBnOpeL-Pjlpxw_KgRslAv7xirx1QQVLHg6--tvCJ1TQIej2pMe3zy3uw=w115-h26-v0

1820a5bf-1531-43b8-ab12-40cb4c3ddd16

https://lh3.googleusercontent.com/notebooklm/AG60hOrBPf9_o02-q5QJAU-Ggxev_I0JSwp1FG0mO_NXdO117QlYD-g3YXSn8VzcNIJmW2wWVgDIYPchx5HMcBegSdKa7AE_XJHX8QbXVchA9IDL5BFEMa3QmbG_toh0VCF2pqYrGtl2Qg=w114-h25-v0

ad9925b1-aed2-405e-95b1-04e9944af0d5

https://lh3.googleusercontent.com/notebooklm/AG60hOq0FO63XZzh9dO6QkASSMD_AB9wVHgg-O88TMxTK_uoiVn5X2pXhzGt3ORAh_ScsKWYIDQmTD2TopWm-x3aa0h6sUZkLEfaaexatCqfQ2Mxf2sJx-NM6fUU9xNZuE-Bmpf5WPLmDQ=w18-h15-v0

6ef9b9ea-6651-4dde-b194-c70998144faa

https://lh3.googleusercontent.com/notebooklm/AG60hOpo-PEUvsuvOHcpZQpKC7KPZZ6_DRt9ulWo40ZB5QSdfwxppRhC-2OAV487FOH0x8TYd1Yh0BqLzh9rPJx61pj2Tl7fkNkZqIkGdBgRSMAc1lsL5GHAq6EkOrqV9yK2vRfCgEiVuw=w16-h13-v0

8ff82f93-9763-4b22-bf7c-65c6435d2dfe

https://lh3.googleusercontent.com/notebooklm/AG60hOqsY7kMzF6LPS8ERoIWxPP-m_5IxgfBhlbvm1FAwv5UvTiQOQDba1fZmvPHWDugkJZLvJi_3vOBeVfjJ88u7gm0RMpf86LyME7_6h0U24gAE_5OlJnnpwsStrntPHoqzBBlx7MPkg=w152-h55-v0

773066f7-cc8a-4a07-a07b-83a655eb7762

https://lh3.googleusercontent.com/notebooklm/AG60hOpjaJl-0OTSooWzOzMQTk0pdZS8D6Nmn4RX7gXGUPVAeLxc2MWgnXsBxj6I49le0tvFbkWbeZdjbDTE3tr-VawVk_jTvBAKlqqIlUpxvhxgbUZEoQd9FY1RusYnO62Hwo3j6BNV=w177-h56-v0

b3344b60-8bdb-4d16-8d40-9dd8357b194c

https://lh3.googleusercontent.com/notebooklm/AG60hOrHVeDy7RfNhaczyUniOZEePgCpau36P8Lq9oTS9JInw6loKK8x9v5UBWUTrO_FdVKHSZpTtBzpu8dKXGgs5bviWuBmt3hqDZGqQrndGoMEkwMEHHvVLabmjIK7ZD5Ro651mZdUbw=w172-h55-v0

03417398-9a91-47e8-b66f-38fff9e71f45

https://lh3.googleusercontent.com/notebooklm/AG60hOqgWSEgURBsGSKfH2KjDT2p_JRPfcInqbvkcgOSxH1WgN-xEy6IOlrId3XGlMUb7OymoEIBELzLiwd0LOY3KQYRdzK0WSR1nCV6D0l0jSCp_cCv8JPq9ukUVFeNHL69gxZ_sNNNUA=w179-h92-v0

d9f1eefb-3783-4cc7-91a5-6772b3cc69c0

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## EVALUATION OF GAME LEVEL EDITING

Gaze Prediction – Application of HL-Saliency GAME BALANCING - EYE-TRACKING DATA HEAT MAPS

## Aggregated fixations over raw eye data from all participants and visual angles

## Game Level Editing

− Looking for an object is a common task in (Action-)Adventure video games

− Plot-critical objects are placed in selected locations to ease or burden the player

− Maintain challenge Aid game level designer to identify object saliency depending on

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2014). An automated high-level saliency predictor for smart game balancing. ACM Transactions on Applied Perception (TAP), 11(4), 1-21.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

LOD FOR MOBILE GRAPHICS C-LOD FOR UNITY 3DTM

## High Level Saliency and Rendering

## Reactive fixed frame rate scheduler based on attention

C-LOD lowers the rendering quality of objects predicted not to be attended

## The highest quality is maintained for all attended objects

Three complex effects usually omitted in mobile devices as they require many texture fetches were selected

## Subsurface scattering Refraction Bump Mapping

Gaze Prediction – Applications of HL-Saliency

Koulieris et al. 2014

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2014, July). C‐LOD: Context‐aware material level‐of‐detail applied to mobile graphics. In Computer Graphics Forum (Vol. 33,

No. 4, pp. 41-49).

https://lh3.googleusercontent.com/notebooklm/AG60hOoMqa7V2Qc-Q4i_KYK2YE2h7etuNHJ8K8TFt4vnkbphq9lFk7zaGGBWTgxBfnV7SfX9h0P3yEYgo1mNbXLyYLh3O7KJE5scGU3YAkPy4RHVRKgmFd4GsRKNIrV7G7LXof1vH58swg=w116-h28-v0

8556d1a7-9a96-4885-8ca8-83454ecd2e55

https://lh3.googleusercontent.com/notebooklm/AG60hOoQZ5p5bLfnRFIXbNYMxFNhWmZcP7wZWv4vd4NiKhP6zn33OvB_n4KYADNIABcbjFnEwLIp3E-qIo1XZp_M_CKkNNgH3vc8y4nKZkdptPktLXQKHTezqJMGiGZ9Zx0CWxKjR1O6kQ=w24-h26-v0

5a9ebf8c-bd04-4050-9171-19f151357c29

https://lh3.googleusercontent.com/notebooklm/AG60hOqPTpZvF3YooU_rlhgnv38YDleKsqY3ovIwAfgcoJBYgdN5CXNjwp_0tZVevx7RP0kAiRtorObjK9TGa5gKOhQIK5hIds6pyK5yzEYoPe9boy0o0aIe2oPgSg0z_VJGPilLVnz0=w23-h25-v0

7f2bc152-4482-4e4c-863f-ad80d3733ba4

https://lh3.googleusercontent.com/notebooklm/AG60hOq4deVb0djz79Ko2KYDeAqL1Jce7p-2DQa7H0aItYTtBI57Xb1Yt0C7LhX2fqvi9dUIjhs7uF2Lr2t1_sKY_fGWXeCIeD-2PjepMA3FWN_9wkVcb8kHjWUwJqZWRNlMWn1jiRpF=w20-h26-v0

53de56c0-449c-4774-a920-c8fd6e9cf391

https://lh3.googleusercontent.com/notebooklm/AG60hOoTSYjmUtHvpOee1ntV5odrQ1q1SBDgOgEnOq6jyNdJ4oxLUYPVhJteW4fK58ogauvyvRKOaEU2SOlaLxvIIii6YllHxrbt4dCDrDQX69rw6I5upHYUavwIc2mBmvinETPCiW6dGw=w19-h25-v0

7b5037ca-924d-4c0d-b74e-0ee19160c1e5

https://lh3.googleusercontent.com/notebooklm/AG60hOovEDOKLQgwjOpaytVf4UNpxYN-tti7JhUBWT8l0WwosDL0Br2cxLRbN86T3X-wcvO2d9uKEEKZo7ucy6KsI5r1n-7vZs53VW8piHPOmVgMaNxwryjnb3MHeOzBfd2D9V3qq6SPUw=w115-h26-v0

596da5c2-dd6b-4861-a198-94f633feca99

https://lh3.googleusercontent.com/notebooklm/AG60hOoCsI1-uqQUD7AnEOxEqgUhWgB_3fdp3h--rzQlxZroQpalSs8BNQZmINh213aKyJF5u4gC9z_e1yNRyIthJQtTgXzyHmV223MOCGP-nhEMRQ4rMKiFvC117OR8_rvbdSL9KYphDA=w114-h25-v0

1d267ee6-d0ec-4c4d-b00b-f3e6edef883a

https://lh3.googleusercontent.com/notebooklm/AG60hOpHO2EIkmwO-FF38xmc06BnBfB38gRJsQxYAvDDqreUKfZY1PJuYuDGvJQKYG7Arbxzyy32YiKymiWWKHkhD7xkx3ECQKmq3eOZ8L9GH_tf3plKlBfaV7iHgzvahwvFw5pDHFq6=w18-h15-v0

4ad62006-0436-40cd-a78f-fc0b5ad4d412

https://lh3.googleusercontent.com/notebooklm/AG60hOqFmDrNTo3EYB5k4odzcxcSd3WvtOc3RGpACj3Z6ZpIlLdGTwgnG3W24VK6_6UHa3zxN0CkHv81kz-_FSVtvU4OlWLO39gcdgRKMRv5RxGizRumx7Iq66zVOc3g5ohG4OIsKvX7vw=w16-h13-v0

de3daab5-cd34-4db1-8a16-fd4e7f3079ee

https://lh3.googleusercontent.com/notebooklm/AG60hOqxBDEpleZQSwfSPBLysQsGQVjPnpmpN-8t5b3vApc7DcgGPo_-VemeGfECyYrteAnO5GQ-Z3ogg4sMBldsHk-qNAHpN9ptfbiBt9JKW2-ywy6BTyfsOsScw8iRyJn_N7mY0xgXtw=w282-h172-v0

26a7e046-7f99-4db3-aef4-1dc7014bf7cd

https://lh3.googleusercontent.com/notebooklm/AG60hOpbJ0F2tJcQqGlcDhcRxOQj-vyiupDKlD8wh3kVujQxtR3JBgU2RQRYTE8vUGZE724UaD15HlEkhrWehkkkqttQBxIg_7YZSRwvZnNhftHaNztxYIQAhzGUAngIiHignMW62V_i=w264-h65-v0

b3d4a147-5652-40bc-bd19-218177c9211f

https://lh3.googleusercontent.com/notebooklm/AG60hOoP5qKz069c1QqJ4w5A7ujLQXw_MVHLY-PxiPtktMCDGUBMNnd-YYupuqpQLLAhMpqxbjeQj7y-dgIGfPzNlrj6XA24ceo9YG2yqH_jMYPUIWJbhM_DNhDKbSRdYYoAxnXmjOp1Rw=w116-h28-v0

7b118c5d-50a4-4553-bb28-5aa4c8dfdf9c

https://lh3.googleusercontent.com/notebooklm/AG60hOr46_WLdJR7avlqY3r8P1P4i6_KUM6M5HIos_W9EWuRtmdCJZDTUdkrckq-pDk0QeRJGwZrAl2NGMJnAW0UpYMXDxpQ8YuASNkc7HRW3JbsQz28HK6QliSvrF6Sr7j4TUKZ6jpK4A=w80-h183-v0

48b8cfc7-37ad-456f-bf2a-ff34acb18309

https://lh3.googleusercontent.com/notebooklm/AG60hOrwyrI4ZemM1jMEQ8648YHxDsjBD7CUbga2Vjfl_M12XAam5bzn3rkgivXVH_PWW_3eT6Q8hLHWX2G-nHupczH1eVpFTSbkc43mcPEoVqgjXNElgQ4QCttQ1d3en1uBH46IjiSnDQ=w229-h115-v0

0efa1335-82e6-4c4a-8116-61fc7326de80

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

LOD FOR MOBILE GRAPHICS C-LOD FOR UNITY 3DTM

Gaze Prediction – Applications of HL-Saliency

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2014, July). C‐LOD: Context‐aware material level‐of‐detail applied to mobile

graphics. In Computer Graphics Forum (Vol. 33, No. 4, pp. 41-49).

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

HOW DO PEOPLE EXPLORE VIRTUAL ENVIRONMENTS?

Gaze Prediction – VR PANORAMAS

Capture and analysis of gaze and head orientation data of 169 users exploring stereoscopic, static omni-directional

Applications include Automatic thumbnailing, Compression in non-salient areas, Automatic allignment of cuts in vr

## Automatic alignment of cuts in VR video

Images from Sitzmann, V., Serrano, A., Pavel, A., Agrawala, M., Gutierrez, Wetzstein, G.

(2018). Saliency in VR: How do people explore virtual environments?. IEEE transactions on visualization and computer graphics, 24(4),

https://lh3.googleusercontent.com/notebooklm/AG60hOohu-vtk_nGGilvEdCW54pxJRSMDGxGy3M4gDZq9R7FXXwTtYkR-54Pdlszd9Nosm7tSXxImpaJPq6GmNt0QZgNf3lwSxEKtAMi8Cf5ODlUpekTMWnCq3Jk2lvRrwpvNb3_95hICA=w116-h28-v0

7af34455-f846-4810-9a40-91078fdcb1bd

https://lh3.googleusercontent.com/notebooklm/AG60hOq_SQ0uLl0IpsazPy-BP-rqRjLdFCdXR1XZvT8xUTNya-gvPelM3KHpC1TJOFM5XNOjMatnJtmG6cYmQwA9zwWCZCnISDyE0h8k3un-NQBO-BEpsMCv8kOAFDTgj3kiVeBpv6Yh=w404-h110-v0

eddf43c3-8f11-44fe-98dd-81199c395eaa

https://lh3.googleusercontent.com/notebooklm/AG60hOp3j4eCysSGcv0xfO8PAQZHXhVgcNn9VpHhlIsfUocpPD8nToW2fJuoRFMx19xUx_D0fEbAiB_9sM9TtZNjtK8oqubnz9nOFtYZo9k_mXr5_ayUjOrNf8PpKzGCTDQgnQIY4Kfg=w116-h28-v0

1f9fb1a6-4940-47f2-a60f-721b8c477047

https://lh3.googleusercontent.com/notebooklm/AG60hOr0-TyAjMmQZ0tG5xBjQNw40DuC4w69GiOFHCvTPhPjvfrk5NIFz2LemQhcqUBvtpvHjDcxmlF__fEkgRPAHN5avMtR7TAPDEgIgKSclY9aI7TaV23v0Tt84NSpLD0ySJWZztTD=w337-h34-v0

5d7b3861-1486-4bc5-b042-2c587744169d

https://lh3.googleusercontent.com/notebooklm/AG60hOoRAy9LwjvoiQVLpyFtF0gBgrove_5il-j5as5C6uB9iz_ha8ea93MnY_zFNnlisqRWm68OyZ2cAIhd0-ozsQkAYRc26a1x-yLrVuwWMAcPI9AkiW4hJazqQS1fMjiPTpLCbDTV3Q=w331-h42-v0

fe4d64b0-a669-426f-be45-d84771c6939a

https://lh3.googleusercontent.com/notebooklm/AG60hOpJmTUsV-apz2IkjUjyr-P9XmsASNy7b_rkQlN2qy9iGk1OMLjlHNyj_fhdQ788KorMCrbTXnUfkDH_zj0i1njKySLmoz0dz8ULX9x2MnRA6tcxBDP-jJdhoRwMJQp0_MxpxZf2=w99-h55-v0

ad1323f4-be4a-434f-8ad2-5bdbd0f2b2b0

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

SACCADE LANDING POSITION FOR GAZE-CONTIGENT RENDERING

## Gaze Prediction during saccadic suppression

Collecting saccades samples. A computational model based on ballistic trajectories captures their

characteristics

Velocity method to detect end/start of saccade when velocity drops below a threshold

VR problems, movement of headset, loss of gaze direction, quality of the screen

Images from Arabadzhiyska, E., Tursun, O. T., Myszkowski, K., Seidel, H. P., & Didyk, P. (2017). Saccade landing Images from position prediction for gaze-contingent rendering. ACM Transactions on Graphics

(TOG), 36(4), 1-12.

Validation: reducing delay in gaze-contingent rendering user experience

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## TACTILE MESH SALIENCY

Gaze Prediction - Tactile

Tactile saliency -- salient points on A 3D MESH THAT A HUMAN LIKELY TO GRASP, PRESS OR TOUCH

Representing a 3D shape with multiple depth images

A deep neural network that maps a patch to a saliency value for the patch center

Grasp saliency map, top training data, bottom output

Images from Lau, M., Dev, K., Shi, W., Dorsey, J., & Rushmeier, H. (2016). Tactile mesh saliency. ACM Transactions on Graphics (TOG), 35(4), 1-11

https://lh3.googleusercontent.com/notebooklm/AG60hOrpjMK4abkq0iYiNz3ufuVLL7lejc0935pBGsXV5M9I7bjnE9Hsqutc7nu15ks34PiLBXgldU6J46wsjVLW9Mf7vkz1Jesto_ZJMumGv1mtvqDanDr211jggG58FbJFFeic6tPZqQ=w116-h28-v0

78d12dd5-2e79-4add-98f7-efaabf88c25f

https://lh3.googleusercontent.com/notebooklm/AG60hOp_GxVsW4Do_oSTxjbnmdlA-_nvNcDnnUgfNKlPBG0ysidHDrR9lx1T7S7UOHBWXB2EME061mdgi7--07ys22Y4p-1rA6eyXS4SORyWBh5cDe6Ln4GeuPNimA60ZVqtDaMIGUN9Sw=w284-h59-v0

400ff70d-aba3-44d5-9ffd-756f3cf75dfd

https://lh3.googleusercontent.com/notebooklm/AG60hOquVeVfCCWfvl-GaMjha91vK6wh6FLF_-f7MHn1bADIhCU8hg-xHDX-O5ukr2l-YXfUuywnbhPBPnHzO26E9XJdXWtib94EJw7S6V6m0F0c6ApdtMXselb9njBL7OoY6HbOkfSTmw=w285-h52-v0

0127483e-6aff-41a9-9951-7b076e9213e7

https://lh3.googleusercontent.com/notebooklm/AG60hOrNN19_w1uAmxETzJeyR5ZcOe-fq5dzePoTlmw3ZWBeIa_JTq1loJyT_EZgJGOflJedD9_RY_ZzzDNYTalwZkTn7SzMJLfIyDrXMo2wKL6hbUNVxrvXnJcgWjW51QxtBB1Mu9c1zw=w130-h45-v0

d6b6ac83-4436-4bc7-a42b-74a1c320e086

https://lh3.googleusercontent.com/notebooklm/AG60hOowVk61QKAVaIL4VU4qieNHpWxeHr0_NIfdE0RnbZRH6J_NY4HW2Gr2rcd47mi_6lKJajuB056xGBzS08sJk6QPoeZR7g7J-Ya4OK-bzVOGz-uwrw7sPN8U_2cJqL_OgETUu1uT=w116-h28-v0

7d4af453-7014-4aa0-81b2-b2d89fa574fd

https://lh3.googleusercontent.com/notebooklm/AG60hOrLaG5a_QT73Amhg0sZ5s2ZAJ2xyvo4cPJ7J4mSHbvwtRuTwlWJeAQRZ5L5BPIr95fzcX5EjHEiEdS8CUIHLpvDDSyp8gzKDIMtT767CzvgIOcxGO0ABggVc0VeKhm_SbNv5NcG=w143-h103-v0

909bec1d-17fe-485c-bb1c-82017a841ec1

https://lh3.googleusercontent.com/notebooklm/AG60hOrum7OjbIU8tixp8EzQwt9WdcNVqwdZ5EQOmsXfEdzKFXMR-kwqkA47A0njG4QLvwC0jiRlNCztwTXSuFuzTsBPYaVB74YfrGvnVkIrl2AppcFLkr0FHOOcW9ygkDVFDqg3F83MPQ=w143-h133-v0

c564f3a9-55d4-440a-92f1-c6c9cab842d5

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

TACTILE MESH SALIENCY -- APPLICATIONS

Gaze Prediction - Tactile

Fabrications material suggestion papercraft. The more likely a surface point will be touched,

the more sturdy the paper can be not to break

Fabrications material suggestion the more likely a surface point will be grasped, the softer the 3D printed material can be so comfortable to grasp

Rendering properties suggestion (such as shininess and ambience properties) of 3d shapes based on the computed saliency values

Screwdriver, 6 discreet parts and materials

Images from Lau, M., Dev, K., Shi, W.,

Dorsey, J., & Rushmeier, H. (2016).

Tactile mesh saliency. ACM

## Transactions on Graphics

(TOG), 35(4), 1-11

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## MESH SALIENCY VIA SPECTRAL PROCESSING

Gaze Prediction – Detecting Mesh Saliency

Perceptually-based measure of the importance of a local region on a 3D surface mesh

Incorporating global considerations by making use of spectral attributes of the mesh, unlike methods on local geometry

The log- Laplacian spectrum of the mesh are frequencies showing differences from expected behaviour capturing saliency in the frequency domain

Information about frequencies in the spatial domain at multiple spatial scales to localise salient features -- output final global salient areas

Images from Song, R., Liu, Y., Martin, R. R., & Rosin, P. L. (2014). Mesh saliency via spectral processing. ACM Transactions On Graphics (TOG), 33(1), 1-17.

## Top spectral Eyes and feet

## Top spectral Eyes nose mouth

George Leifman, Elizabeth Shtrom, and Ayellet Tal. 2012. Surface regions of interest for viewpoint selection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR'12). 414--421

https://lh3.googleusercontent.com/notebooklm/AG60hOqbKXC8YYOTqjbSesmjhgYIgD7cAC4e8XCTH4hWP53oWpce_f0sJZclAe7nRAkoNNLrRQpcZGvkk28Az_0MqbC1-0YVTpgBa5YYGlRW0MVMyhKfLEEhES3xgRW0yAirKN2y8odwAw=w116-h28-v0

30c4fdbe-a1b8-4f76-acf9-130f3a1142b4

https://lh3.googleusercontent.com/notebooklm/AG60hOoDnrvCTMXbNgDIiRyQ8YDz0Sl89Dli747og-6rU9tGpa1t6SbXNxcUOPV5RCsxXExhBpcjCoSLxGAdvsVQSB-bAfxjsrjEIQu1gV_bio24VCAhuD63HxSwCkwcCilwBrTejCP8=w430-h86-v0

c3f925fb-5cc0-49ae-8dba-31d1eeb963b1

https://lh3.googleusercontent.com/notebooklm/AG60hOqP_JauS2YkXM_WME5Rm7suBe_6W01gcesLQKiWDPniMRzWRX5uwe2uVPsK0U7NnG0NqdN1H7UUdQ-NSeb7R8NE3NBeUGK-PxPmfgsy_d4JUFgFP-hflj6CbwJ1GW5aDS_hKVmyww=w116-h28-v0

5ca58fd3-32bc-45f1-a648-725676f7ec98

https://lh3.googleusercontent.com/notebooklm/AG60hOrTFOfHNpqXf_ecvljrAU1q3hDbWlaEi5GMSMAzgV-XtCszKbV8Yfi7w09ZlcpM7vB6hjA0O4AWh5Z7nFRnqCWxkAMaPnJl3IlRCcKrO7gHGZqIqqRRl2firxnJ2ukJ8aMtqCiLag=w548-h108-v0

54a260c3-d68f-49c4-8dc5-4142d457ab0e

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

3D ATTENTION-DRIVEN DEPTH ACQUISITION FOR OBJECTS IDENTIFICATION

3D Attention Model for 3D Shape Recognition (View-based)

Reconstructing the scene while online identifying the objects from among a large collection of 3D shapes

A 3D Attention Model selects the best views and informative regions to scan from in each view to focus on, to

achieve efficient object recognition

The effectiveness is demonstrated on an autonomous robot (PR) that explores a scene and identifies the objects

to construct a 3D scene model

Images from Xu, K., Shi, Y., Zheng, L., Zhang, J., Liu, M., Huang, H., ... & Chen, B. (2016). 3D attention-driven depth acquisition for object identification. ACM Transactions on Graphics (TOG), 35(6), 1-14.

Depth camera on robot Scanned, object identified  from 3D database, driven by attention, retrieval of 3D models

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

3D ATTENTION-DRIVEN DEPTH ACQUISITION FOR OBJECTS IDENTIFICATION

3D Attention Model for 3D Shape Recognition (View-based)

## Reconstructing the scene while identifying objects from database

3D Attention Model for object identification

The first level selects the next-best-views (NBVs) for depth

## The second concentrates on the most discriminative regions

Acquired depth images, identification, attention, output

22 Images from Xu, K., Shi, Y., Zheng, L., Zhang, J., Liu, M., Huang, H., ... & Chen, B. (2016). 3D attention-driven depth acquisition for object identification. ACM Transactions on Graphics (TOG), 35(6), 1-14.

https://lh3.googleusercontent.com/notebooklm/AG60hOq95HFoqfvBtKb7Ar6-Or73GomtuAeXglmmebSPlX27ehT3JkOrMxC-FJlSUNj-aV0R6T2nGvYkEhk8j3pWAtVLBWwbZlCdoVzJK4Q39dFScmJQoB0kVEznui6cTPX_weX3DHo9Cw=w151-h36-v0

e980bc49-4eb2-4cb4-b777-88cd1a0c5859

https://lh3.googleusercontent.com/notebooklm/AG60hOpyGxz2BmF1HrUz-swmIl9TVuNWrn7JScBZYtiUyh3NBtLcCUAAFwfknfUN0XUUIv944gG3JiSHrN8ubJHCGDotuyB0UD6QmX9lMRemWLoe4ytdLhvLfFBS8SRjf3cVdSgWzKG5tg=w151-h36-v0

a3de7f52-a8f6-4bd8-a463-226be3150f5b

https://lh3.googleusercontent.com/notebooklm/AG60hOpsdsXhNc9Iq0MLH7AwRXttOvIFguXrhw7AAJylT-DHYTOkv2wx2KdVBDtuADbmS-IDEGom_YaS_BwpnDx6JiPO626iE3Zan0lBbqJtVLiqSnJ8xEZEZ_E40NZzTK-aC6hoSNrzaQ=w300-h359-v0

82b471f1-1d77-43c1-bebc-c0c500e5c7f3

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

CNN-BASED GAZE PREDICTION IN DYNAMIC SCENES

3D Attention Model for 3D Shape Recognition (View-based)

Analyses of users' gaze behaviors in dynamic virtual scenes

A CNN-based model (DGaze) that combines object position sequence, head

velocity sequence, and saliency features to predict users' gaze positions.

Also, predicting future gaze positions with higher precision by combining

accurate past gaze data gathered using an eye tracker

Acquired depth images, identification, attention, output

Images from Z. Hu, S. Li, C. Zhang, K. Yi, G. Wang and D. Manocha, "DGaze: CNN-Based Gaze Prediction in Dynamic Scenes," in IEEE Transactions on Visualization and Computer Graphics, vol. 26, no. 5, pp. 1902-1911, May 2020, doi: 10.1109/TVCG.2020.2973473.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE DIRECTION

https://lh3.googleusercontent.com/notebooklm/AG60hOpocsr7oWiLH-JyGqNyv8HNrIm4akNlHe9QaGIRyBssDmFpxbBJjL6SXDIxjQDtzQaTQgUsbXRsCeQTypb6LKD17dh2L82FZDzOesxGYbKO-nuUqYUS6vVCtsF7xz0NTQqHtPMd=w116-h28-v0

ebf86d33-8b04-4130-a87f-88cf978b4323

https://lh3.googleusercontent.com/notebooklm/AG60hOqtEMvNrdGRr8DpIVHPj63V4oZVzHuX_npvOe4s2c1F2ZvlqoCcEnKQhAfmLxGBvGGWiQ08KiSb4Q2IjHQp-riR4jyPX39GhMi8XJeWttBdyZAuJr4r0Tn-DPyU-TjtnaiE4h6paA=w41-h26-v0

5f0723cf-de32-4567-9eb2-6a4a77ecda0e

https://lh3.googleusercontent.com/notebooklm/AG60hOqa4dfBX7CfD6zYnu0AAD_fHSDOEt4Cp8sGcMI1iFHvl-05vkPNQGU3bp-8kxfBAF823FOAAzP3zNfG1RdqIl9-hx51RQWIgNg5K0Q0zg35bWlpJ5VkiOu7fiIchI3dYHR-ixlH4g=w40-h25-v0

2d0e0a5c-ac10-42dd-b4ec-b3512e4e7860

https://lh3.googleusercontent.com/notebooklm/AG60hOrhMujcijiH6TfywvoCmG15r-hL174bMEkU-R80X56KuLl5_2FDdLPQUrDB38KIT5QF6kN3K0dRgDkLEpxQ5rn5euHfXW_4TABdE3ZLNAR8OCMayFPHDT9Q6AoYOtRHIXgfO1CtLQ=w20-h26-v0

fdca6258-a0f4-4af0-aa49-38d41725aeff

https://lh3.googleusercontent.com/notebooklm/AG60hOoqJDy2XqReARtgeQv0Bg44hVsrI-1cKNJpiOblyP8caB-bkqQoeiEnR6UybGlpTq0AdZQfmh3lke3qjS85QYPpMxknkvJdZmj53iOwhtj0wHGUUB3hxXHO3I7Cpp7p3ng9NYzhdA=w19-h25-v0

d5ead4b3-e4ed-45fe-8654-84803a863522

https://lh3.googleusercontent.com/notebooklm/AG60hOp9R7vL6P4IgkAHItFHiPu2CGsVhvsndxRsvfd1c6rdWwWJ515wK6AYAnBaCnA7E57rUy8r4lyWKbQrERj11_RPRoX8r7tuCK3tzTSPSO5CkzOUYLhApWPSIv2ryesTDdZB2qPF8g=w49-h26-v0

f1385a8d-91b3-48e6-8081-ee4a112580aa

https://lh3.googleusercontent.com/notebooklm/AG60hOqSnoH-UpQiuCREwh8FvqrPVoT7doTwuYZ6gw1IfbkQ4g2eihw1n0EQCEqem6tOrRq7dJIK15nRamNm5NPqbnmsp6H2boZOQdd-9YG44guSOqbXPOpgHmRe1lmiMqUHT69nhuGaZA=w48-h25-v0

b956cec9-3795-40f3-b3dd-d2b69d371b15

https://lh3.googleusercontent.com/notebooklm/AG60hOoBIr4d203UbRP7LS3gaO2LBcIbhGGHgknMxl46JMAgyTbj_jG2zV6zeI3BEtcZt1z7N5VAur9WyDyET02LozTlPV-Ridh_VpBEKPTY5MR7Sd2uOOloV9gvk97Yponyy9npkjjGcQ=w20-h26-v0

806f0818-580c-4f5d-ae6d-1ee5b36ccdd9

https://lh3.googleusercontent.com/notebooklm/AG60hOoyh4GH-4YX7VOKonkM3zzWiv_kglFyQDUnom141AgU-6C_PMz8yRkzaR3ujqSo2ee6nVGg4fSqzCgp5VVMttrnYzjW9nNIbF_ZdjDj7g4KtLDG0z9cix1q7it2o85HY2S7-E2y-w=w19-h25-v0

7f8f66c0-1234-47de-8ac8-0d5ffa78b465

https://lh3.googleusercontent.com/notebooklm/AG60hOpjH9ZP9pyZUkW6Nc5ccGaa2VlwGQdILMGYQgoUmDFWgxT9GcktNj1cUPDwVdzrwgF6BBm01dUPQOxz2mUCyJDq7NliIOYl_aNdg1YWAjBmx8XTXVfQbKJQLuFEFrUxvc-sQoS-8g=w167-h26-v0

00fdcfc9-159f-4ba2-b6ad-c43565d03982

https://lh3.googleusercontent.com/notebooklm/AG60hOpzaArSFT31Z22p9g3DWLpVr3SUMZeHPprTx8uO4tP-_-mr33dTrjUJwX8igHdfe3Kpoa7NaQ9ezrKUUyncf3eeofn0_Q4-bXqlJh7brXsZVqxTNwfBCkOYsvo_tO8VtnqMJ5Pr=w166-h25-v0

5aef606a-4dba-47ee-9092-9a052820b4be

https://lh3.googleusercontent.com/notebooklm/AG60hOr6y8dfEa3VpNJ4H4tzZPxluSptRAXxlqoTvco9R3XEd3ECjB9C8_f-jC0mvM94JhMtbiy96cFD4dxYw8vHlOW9Tr9fPcmFeTlykjKT8S77iAxtlZQZAuInnd2FiecU1nc4IWni5A=w223-h31-v0

0232d615-d153-4d95-81f9-3a82455e86b8

https://lh3.googleusercontent.com/notebooklm/AG60hOpAVPq8re90p1D1QmeRNbxGAum_5p--Mla9KcHifulsh55CzEOaHdF0_wMvJDJJP63AdZaiKKGdqRI_0kPQ78YPrzG1bEZFI9xAczN_1RwW07YOdzOuADXBpfzMHmtJ2TL8MWJ53A=w220-h92-v0

a03c62d1-c072-4395-a966-bb5bfedf1024

https://lh3.googleusercontent.com/notebooklm/AG60hOodMDJqqzQPrrD7h9pLGyUx6NSO2GjzPC2MSzGsvfKD2IPTwZY6qQ61Uk7DC9A-zIZrw7kvFHQx5rgAfHofF1jnBJs3aEHSfqu-smtEV9yZ4K3iANmlGHPrzs02L6j6Oopoyc5p4g=w116-h28-v0

290b954b-9695-49fc-8779-d43ca26fa780

https://lh3.googleusercontent.com/notebooklm/AG60hOpefSqeqRBe8BcUGzjcZdaNH_FqC28Powf0smmxXSk1tFBa1qb7CGgKN6jTwKGVyCu1WrH0E_84nWZur6olqqLE4YyuXiiPqPldUY2-xxmlGrz_ARaGKvS96Awa9vy6eJXU7zWjsQ=w326-h103-v0

2af47766-79bf-4248-b845-fc79fcc18b37

https://lh3.googleusercontent.com/notebooklm/AG60hOpDXkE9nVPvY4lnxWQnaBsH6z1zyl9Sp2prL_Jpudg5Gwrwpj9-7zxu_nSjhvKhyoRNURtHFbtlU8IjZh2_Wz8RH_pN2fNCOC1R90CmnAXbowpDDnY0nC0FOzQnBc9yQW76BAn3=w143-h138-v0

98902262-cafa-4803-818c-ae32bdce7a7b

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## MANIPULATING ATTENTION IN COMPUTER GAMES

A frequently searched game object is modified to share perceptual features with a target item.

increasing the saliency of advertising billboards by designing task-relevant objects

Requires manual 3D-model modifications

In-game advertising

Images from Bernhard, M., Zhang, L., & Wimmer, M. (2011, June). Manipulating attention in computer games. In 2011 IEEE 10th IVMSP Workshop: Perception and Visual Signal Analysis (pp. 153-158).

LOW-LEVEL-BASED GUIDING PRINCIPLES

Gaze Prediction – Low level Saliency

Memory task, task-relevant positioning and appearance

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SUBTLE GAZE GUIDANCE

Steering attention to a specified target location, which can significantly differ from the natural fixation location

Subtle gaze guidance requires altering the visible scene context - subtly

Images from McNamara, A., Bailey, R., & Grimm, C. (2009). Search task performance using subtle gaze direction with the presence of distractions. ACM Transactions on Applied Perception (TAP), 6(3), 1-19.

## Gaze Direction

https://lh3.googleusercontent.com/notebooklm/AG60hOoDBTEEjgcDcfBAxW_xfcv4DKv_k80nqQ85-jmudA9ogY0ZKK46hpYqroojFvB-y-zu3eiIh0bZb81R2sw24E00Juir6XOfGtk5Wfe3kHtAa2Rs9NczBo2WcVQ_IIi_tw3LLkvcSw=w116-h28-v0

e26bc993-8e4f-4730-b272-fb1a88e5ed5d

https://lh3.googleusercontent.com/notebooklm/AG60hOqsqEYD3uK-YYm1KIJWGX_GXp3ZGCYBzbeq17O2cl_n_vycLZzAJ8V5joRc8qnhK7jOMxABdDllfz_E46G88hdtuNKuNu9gw7BKlYtvMimkYbr3o9HrymTK9QGa7mwjjcy3r5Mw=w116-h28-v0

8eef9648-40d5-4566-ab65-38c1143f5da1

https://lh3.googleusercontent.com/notebooklm/AG60hOrNiNoJ0PK5V-IHzAPuIC0vyopsBB_Y43dzUSRXSTkpAPUYGzwzkN2mYjPEl65UP2tthcYZ4LTJRWyTxbk80eIzSq8sEDzHbX9ZZxvcg9eOrRp5NJ1LW91HkKMXjb49QAAvHD5A=w287-h147-v0

b8192182-f282-4158-bf6c-cc41378602dc

https://lh3.googleusercontent.com/notebooklm/AG60hOpzLenYm7o6pnU9Yz4ydVFWI218uYip8OrE4b6VSuDJyMEOJ6u6cShi7qPRZmMMW66MBtv6HCP6nw3wsKilqOWBF3VciBgfBixiepdIcKYdck9ut9ua0wenUfDMAThLEG24eAM4oQ=w235-h112-v0

c73e49c3-73e9-450f-9a35-15409c44bda3

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SUBTLE GAZE GUIDANCE

Presenting brief warm-cool modulations to the peripheral field of view, draws the foveal vision to the modulated region

Subtle gaze guidance requires altering the visible scene context - subtly

## Gaze Direction

27 Images from McNamara, A., Bailey, R., & Grimm, C. (2009). Search task performance using subtle gaze direction with

the presence of distractions. ACM Transactions on Applied Perception (TAP), 6(3), 1-19.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SUBTLE GAZE GUIDANCE

Subtle gaze direction for wide field of view scenarios in immersive environments, gaze guidance for redirected walking in VR

## Subtle gaze direction in immersive environments

Grogorick et al, 2017, Langbehn et al., 2018 Images from Sun, Q., Patney, A., Wei, L. Y., Shapira, O., Lu, J., Asente, P., ... & Kaufman, A. (2018). Towards virtual reality infinite walking: dynamic saccadic redirection. ACM Transactions on Graphics (TOG), 37(4), 1-13.

## Gaze Direction

https://lh3.googleusercontent.com/notebooklm/AG60hOrLm2DyaPN_0EvR6jqSWS45jrLV6MlFIPgalndWig7jqJDp5QibjZ8K7sS-z9pVEse1Tjn4C70pYTuth5yMX7mU9AJuinN5NqbxKjbbtb_PucoesJyCsD7MDjCC3xdxqpgQNh7CrA=w116-h28-v0

6da0b903-7875-468a-b038-53c6732794a6

https://lh3.googleusercontent.com/notebooklm/AG60hOoi2vdjiz5j_MepMR_8CFK8_T8GjynEEyPlTAiv3-tVbPr2NfHcR-HhBcGPK7fTKfdeIcl5n5pXy5Cdp_H-U6WS0UIXy3LoH3lMt64__IaAYKOKpqVpSwMbfHjuTSu2icE5R8lI5Q=w29-h36-v0

473de03f-728a-4b85-a5a2-38ae8e1b0bbc

https://lh3.googleusercontent.com/notebooklm/AG60hOo3fjQKay0EtZc2Rx4O7Dnxi_KI9sFnTlQMMhq0KY8NRpUVft7nDgh58WgWEgFQmcdc4XzN5CRjFdD26EPOFxuvUPU1OEqHid_NtMBUpSvX3DX3ZPQ3uhK2uLSyp0VZQigkZWd3Bg=w26-h34-v0

c2bfb67d-6300-489b-b0ec-defa23ed299e

https://lh3.googleusercontent.com/notebooklm/AG60hOolexFh9NZ9JerTA-UAmWhmHDHRyi2dqHZk-HgH7-vjzPGxsuk65RutIfvaqUYobmLtWeZniyAPb75CgxGS-xw8DlxRsw2tPKFhZ095vTebzVxVn2QO0pRZjRfHcQkrTJoXdnsDdg=w34-h39-v0

1a348d51-dac3-460c-90dd-fe82f32c791d

https://lh3.googleusercontent.com/notebooklm/AG60hOq1z-NdTz-Wfivq01w5AZ1pHudSM7F1DC6uLKNgjmffXrO7eKoDQu6ff-tVp61bIQzqVGoVAeGSh7xE2L5rCefnqkIEoUMp__7-1PIKXhToVN0TrTvnQLjgwuVYO5T0AVnwN_W4HA=w32-h36-v0

43929fc2-6c7c-4f70-b196-032258c595df

https://lh3.googleusercontent.com/notebooklm/AG60hOpeCdQbEkMtlc0kTnhKOz0M3QMA0-FKz47FFidM-tYlTve5CYmrB9jWQ3IghipIB-RNbK8d92g5eLxP6Y0wtcSLbKubA5CVoOnQEh-LOF0tzWoSGMkfOgm29ZW2yyu6NdKfiETQJQ=w38-h32-v0

36e2819c-b3ab-44d0-b747-9905b011fcda

https://lh3.googleusercontent.com/notebooklm/AG60hOqJL2R1t1KCJB9QNMb_sHG0Hs6_SEQjpObut_7ZmG8gCPUTv3nZlArWUVD9FE_RWLDq3npgrtK3yrd7pk-ALOBv4NtX2MFTDamD8kgX8TSkMSRMMzNX1HJtAaxLSFM1KPGM1hUurQ=w36-h30-v0

2dd2b6bb-003b-4db2-b956-4f763ece83b2

https://lh3.googleusercontent.com/notebooklm/AG60hOo5OxNqLJDYWwuPTZWDHURYmW-BOzMjUnVYErPjVo695FKy7q0OhW9Ol02fN1_sEKZ87X9I_kGxTrzAUuFUR3tWK94XTddOtvUvlFxRxQqekc_R0oz5WHXcTb6JvByrwBvavH-oBA=w82-h32-v0

23f761f7-f05d-47be-99a5-5decca11a7a9

https://lh3.googleusercontent.com/notebooklm/AG60hOoYgEgafiIYn0IWW4u6lV-wCUdrqIUkb0hDfjZEZNUtCFbd-FbVzF6Y0M5I8J2EGd0p3P8eFo3jqyJqHM-RqTrl9yIRG5gd9I4Kuex6UoKUF3H9qMzUDgHe56yGb7EKMLPjpMpRzg=w80-h30-v0

b4c8d13c-cf6b-4b3f-a672-f92c5c038363

https://lh3.googleusercontent.com/notebooklm/AG60hOroWD0FRJZhtuoR4Eg8azOu6thJVfdDxZzfAxGv4NpLg8LAiIR1YtqxDSExE66IQbBwk3gx8bcEEeCut4wWvsSMVa8bXpF0TyVqldRgggvKXSif6O-7bhs-XyPDS3XEyn9rfajfPA=w45-h32-v0

0902929b-359f-4f3b-bbe6-36b1a045a4cc

https://lh3.googleusercontent.com/notebooklm/AG60hOoq10AEbLwAGQJaXwlbuToCHWEIFEXpu_xyUhtbhZfShNgfks-8hO_bv0nanIehQNT_E64mhpImHj2MLQXk73nomsiDWsmSeY9cnEVmKdZMLN-VUV5okVR-GNEVta--oYGpTemUNQ=w43-h30-v0

dec3ac45-e5dc-470a-b8ab-5a46e293edfc

https://lh3.googleusercontent.com/notebooklm/AG60hOrFtrjmd3AHmh0fP2JGN8fFwaLB8ykwOrTkhUESUrs2dc8etsmeptsz2htrfuBaeCYMf5cyOSOLOgVxEXimbFlvsZUI_DQIurEZkHQKxNBXTU2SRkmWbysqMmsYB-LQXIoCHu28=w32-h32-v0

1d3d6598-aad5-4fba-a23d-a4b5dc2519d3

https://lh3.googleusercontent.com/notebooklm/AG60hOqbZ08xNnXTYll4S9DVAsXS8WPss2x-xce_mka9lYX36dlCeVyF5MYUkgjl0IuK3m9mwyWxUVl0jvLGMAGVsQQHig7ZOM8DukXyQuVKMUx4H3Pu3qaIT2zviRobhBqP38LxJpjt=w30-h30-v0

79448765-2ce0-450d-8c23-ad492853191d

https://lh3.googleusercontent.com/notebooklm/AG60hOq8W0mOBypnSc0PhyHgJ6GvHo-yV8JPEJp1gnULbZWNQ-Dn32_EIxGQcz8xl5LxWtDrZxCuzPoxjql25kKGP84fJtyTbDF1rHE1dEffaY4ND8_o6dCJ_4kOSxHd-bMwS1_wbRFG=w98-h32-v0

d02fe6c9-2a37-4822-8e92-3ff74362c3ec

https://lh3.googleusercontent.com/notebooklm/AG60hOr3fNGvG5WEMHdKFBrs1uhB2sfOHNU0OTvKweBkBaaV_ptvFlqBQx3GqPNiBkp_SpRQR8ICuP6e1I6y6PVohhPvM2mFITEvfoEIK3ckfvul5Pl1ikcde9zdixDk4kPKrrVaTwjXcQ=w96-h30-v0

ff28ba6f-df4b-4f38-bf8f-1f025ed78e5b

https://lh3.googleusercontent.com/notebooklm/AG60hOpgI2wHbzeeVqs_KlQ2XaBTmiV97YzdoVTRJOW-pCaAimJe9hVig4UZE4NOPP6ouV_i8l4mYDUphg3thlUNdq-glLomtZvma1Zi71GZppseYYAPTTwz0VDHh09LAZIf07XAjtFWHw=w30-h39-v0

a2ccd0b0-3db5-4eed-b637-2dc6d35dffff

https://lh3.googleusercontent.com/notebooklm/AG60hOrYO4GYm-f5k1ClLYU75Oj4rKUijxr4ePf3I4ld_Ik5Y3GPg9saB9pX890EzRAxkOMWxzIVEhwJ4r5Jpu95MRgPM_AsvhLc3aLPWQNj98FLqQr1oefPUaeeE2WsF24AxiefTvvwiA=w27-h36-v0

ec8422d6-8a4a-4637-b5ca-8454c3ee0dda

https://lh3.googleusercontent.com/notebooklm/AG60hOpsVTWvThlgggu5P494hD8kDwuQUKCa0tLdN9QT89xvoaEP8z7BDcDj-0jZ77UYpacVpl-2prL9pyvMTfCdJCLJ9T0gDyv_hmLzyTu0nItdecgHMHLuY3uvk6bGLRDm7X6Zquv46A=w87-h32-v0

40aedbfd-a08c-429e-8493-7094e21a939f

https://lh3.googleusercontent.com/notebooklm/AG60hOogkKWOonrEu0-lMs1u6bm1YjPzw0uY7FOSy_1xYq-PWFVWuSG6XfN8UDu0qZmXQsXGTsw3f5VF6_ERnfl2ZlLJ8XK0BRQuUdwp1fWWvwfJsA90Xn2oex4zSdgScOhwr7gSnvEh1w=w85-h30-v0

bf729c91-844b-4ccf-97a6-9cf676f80639

https://lh3.googleusercontent.com/notebooklm/AG60hOq2y52YNrbI7BROgBmg9IFZuyvKzeS7xVGZT8CnZs-SXLf6dhhwjU1rO8KQo5pXmMJkHRBk4JKI5qO2yl20-rzDtJMz2BlQ2UrIEesYEWERv7rd7SVGpc43sk5sugTOE51g2Y0JRw=w29-h39-v0

424a5ac9-3b13-4ed8-92b2-334390d068bf

https://lh3.googleusercontent.com/notebooklm/AG60hOqmtb4G9TnXfjCGjBTdLDO8ok-llvOKIpx0pSSU4R9DmjxOZgiqv9yhej4dkJ7ibg6xj3eYFU_HauRu4dKVFYGOfApc1lEuCBDT89I_Vb1I02qHNm_N5qlfzc8y3a9Gr4qQ3hjX=w27-h36-v0

6b09e2d5-7d6b-4b97-bdf7-aa40b028b3d6

https://lh3.googleusercontent.com/notebooklm/AG60hOrRDFL2EZJ931gi2zkzNvG2RRo8QLdyP9Cw03_ckhisICEFnoGsW5lJuXHWhTA40VO8J9bPQuVcJF4CdEOqC_QqpiADdIQlvjP7V-bt9JPUQBiO3ZhrEpKWrid_8-qF4Db7a3OT3Q=w79-h32-v0

4e701d3e-e3d7-451a-a391-8883d8b81c41

https://lh3.googleusercontent.com/notebooklm/AG60hOqgfgRsWNDpmKwSEWc4z5eGJzC5cWIi0kQHaKDbu2GBlt7o8Hv5yO9mnU54MjSvg_53LDwEbqecU8873gUI05p6l_vh_q_2Dv-LqCgLYmIvGQuchEQxMRftX_eQU0mRj1nJZE-f=w77-h30-v0

e50d9ffd-ff11-4517-915e-656f6eb757c2

https://lh3.googleusercontent.com/notebooklm/AG60hOpEOLv2I7Te4rylkKqEMviPuEy8MmTw8bsXS_Q0PkcK8uM_cvy-XnHQRhvdjrfq9uCdtql8rowVeZUSlPAjOELg0m4WKHXEpJica84TuU5Ai6QzrFgm2sy_TAGQkN9y-KjLLSaO5A=w34-h39-v0

98a28e87-9ff3-42de-8d12-b332195badee

https://lh3.googleusercontent.com/notebooklm/AG60hOo16rJvh8ADNDE8PeUowD1BU9ZUow1q_8fexBhLitr4jkb9JrtVULUH24dgLwKFNUngLvTsrWnNQnM4Z8nonZ-RK0K214Xmbet5D53n7zXW84X6prCJXjyggwYk31NnKycKxIf0LA=w32-h36-v0

2ba81b4e-f135-46cd-a54a-0a20b8eff30d

https://lh3.googleusercontent.com/notebooklm/AG60hOp1hOQkAw-NQYR_y8lbji7JL-PQVxG-FYfRCX90e9hYa_hcsceiWW4-AlcPhhH8a5IIbbIQVfj2hj1Kn0c1wTFacPZ7bjpiPeiqOYUiQ4wbStE2WXdEUCbxSZu9yIilnvOyt2-hhQ=w48-h32-v0

737ccecb-253b-4e6c-ab12-7a2995f679d6

https://lh3.googleusercontent.com/notebooklm/AG60hOr8rUseYsKjt1_feL2grHjJXUI3qENOXA4xQ-8iil6Dnrf92oTJHm3AJd79cWHrFf6H6Zbp-yF7vMPtODI1cdHWDFPF1FYUrB3DRqoWbV2M51tqlHX8xlanlgSDig1runos9XWjlQ=w46-h30-v0

162b2b4b-1a4a-4ac5-95c1-6e8568214e85

https://lh3.googleusercontent.com/notebooklm/AG60hOqLMjYGXG4vDgTr-lCs5hR4puDVAdB9IcqzQ4eLO9pmeat3FaMqfEiYIX8N9vWj9KVXQA1GT9M9XLIMAZ9vfI5QAMnM-DAzCs7-xcPOcR3XtArJvZsxcEUn7opGQqFCID3X97amJQ=w81-h32-v0

edcd337b-94d7-48fc-a332-77fad34454b3

https://lh3.googleusercontent.com/notebooklm/AG60hOq-nTx8GmkHAv_vvqsxkjx-6ZMnu597c7hn2Fn5B4dSBZH4_zs85L4S8HDuX7jA_bcwEkokO-r7QuMAXKXBKApeEyqgzFhoTrKh4DfRRCnDdR0lU8HxmeyUTqwKP7Szaz7dwyYT=w79-h30-v0

b00e3bd2-4b18-4ff4-828b-05a08597983a

https://lh3.googleusercontent.com/notebooklm/AG60hOpJSnsy1At-xnsl2Dr9obTJMts3cw_hM9c4DP6dZZhDa3pD_oYWxvfiu4J8lnEoAM_FKnoZhxRCF9aBROlfmhjHbf92KOVszaPVrJLmVD2jwF63PlFXIMrsSlGYvZSzAY_GpiXJIw=w83-h32-v0

628cbe48-1075-4cf8-87bc-e0f5a2e3f145

https://lh3.googleusercontent.com/notebooklm/AG60hOpra8hpj21zcpf8Qq7b-TLOhLMPhixH37n4hZ4pjHTxB7EIvCpSP2PvOJKCgsXIZUGHlNZoypJZtfHDtXlOImCa3uhs5QsEnyj8xl7iB87D0BiLTDhFBBtu5TLfiJgTSMHlFeZixg=w80-h30-v0

35f95576-b7df-4bd2-bce2-c463732e213b

https://lh3.googleusercontent.com/notebooklm/AG60hOp8aVjnwEEsfYI85JkOjzZxC96plWs4lKATmsGuK72ROismOtsPjRUzsIStye-z5xhhiTGWHVlAKEb9z6emvYd857XbemoNVWZnPu0ObYuRHcWvDytFxNukio8FTrkT9Y6jMiNMMg=w33-h39-v0

e1d10102-0a2e-4a49-bccc-b5190766de1b

https://lh3.googleusercontent.com/notebooklm/AG60hOqx-QuDTcEj9YKqQ_D2xvVlJBRmgNlCPhGZWXEDGQIPQfmoJNeOqWQ7cfcj8QYxhhCBsFKnelEhS3BLg8OZ0nmb9EgnnVxqEO_DgrAFXo0W5vpNzKcc1PoXcM3j3pChBdj8yxejDg=w30-h36-v0

6907f842-9dcd-4e5e-afe4-608b355482fa

https://lh3.googleusercontent.com/notebooklm/AG60hOqBtlF1mBX-I3umHvfmuFbGOhOtc-Dr2yTHTH6oziGrhHMG0VTCp-CQDoZwN4zCSix30cWtU2Fr5n3ihIYbWCZvEihBy03VzvEDLsRnC0z5kOqUhN_fMGcqqVUOyrpFu33xZv3N7g=w46-h32-v0

a2b0fdfd-caa2-4236-866e-9b87f095db7a

https://lh3.googleusercontent.com/notebooklm/AG60hOqZATjWdX4yILpRIeP2DqhpQIKbOvz4jcKhrr04bV_F2M3JeNgpNQLtEQCnck8KJtXiDm1034gdwi86uPWgcmilsyWwLTjRJehe3B1fqfpyX1eKf_CyAPSIj-j9YEZa107xabEL=w44-h30-v0

5c8298fa-e605-4f79-a7cc-a7481473f795

https://lh3.googleusercontent.com/notebooklm/AG60hOpXEu7f9xMj4i4u6XGcwP14XV3Bp0LtNOYiZlPQUM6b2uEVAsggLwVHzs5vHm6DHRuBcXQSHXMPZ3Suqb3UYnuzGmiZ3N1zQc3eeiUAf8aR-yKRhdZVFGkn3Z21DtEFF4qU5pWzjg=w56-h32-v0

20f6c7a4-0513-44a6-85fa-6b7946263efd

https://lh3.googleusercontent.com/notebooklm/AG60hOrk0fLHYUUDucfKMieHI9MquUA3ux1z2khEqxaCzooPGvt8QBRITLqyIQYVM88432PlrMQGW4i1ebNN_SA1ljy3vgn286MGPbzGhMnVfyRW1SkU-Od0Ss_3XH27puEqdnMSgnADCg=w53-h30-v0

b4164051-fcf8-476c-a72e-6e39aa75923a

https://lh3.googleusercontent.com/notebooklm/AG60hOrPhNoJlNfqjEwb7G2BHW5wced9WMCmVswLgN2661BWoEusbKG92qhNgDzJYpdYCQiURYj0lcE9U2c8oryTQvmrBidaIsG-UTeqQ9Ca6npwA_HWi7w7Phh04ykj2fd_MmAMo2yQcg=w29-h36-v0

9977c995-9915-4aec-89f0-fbded3cae990

https://lh3.googleusercontent.com/notebooklm/AG60hOq6vlbBhglb_9EpizTj698GRihbxiJ-Uexh2ktZ6AYuTQJ0kDzJt5gEyzZkzkjh5VEJqBN6nCQGhMYAjOd_RthnJGzXkx_mdBnRsnzh0Qv-7JqA1rvJIZNjUlos5mAEHfwjc8Yh6Q=w26-h34-v0

eab51612-1784-43d3-aff9-54d62cd5c3bd

https://lh3.googleusercontent.com/notebooklm/AG60hOooH_AWNUuKr-pF7PhvnrCwR2Sbh1FiSAvoLOJSejVnULse-5sofdpoapspcdl-X2cfMwyxehfbSkPwa9EEadB9L0R2bhlqptRlf0n_7BgQVIiQ9LfDuXet8pQCLwzDArBw0A97ow=w28-h32-v0

467e8569-caa4-4713-8979-b6907dccd385

https://lh3.googleusercontent.com/notebooklm/AG60hOqjU3WAZ8wsuZ9y2TjZWYq8rfxxgJ4M09bG2EluhxAyqVHwv-L-SDk8k_6lHo26efwmHh0Uu_EQriGMzSjb5ULNBUpWBoNRzoOEvPVPAkBA2lSYHYvIqfZyAo2yV8280fCtD2ke6w=w26-h30-v0

85f94cf5-c8e3-4611-a8c0-f83e6f65e30e

https://lh3.googleusercontent.com/notebooklm/AG60hOpCm1ls2iRBPiioN0XqNho3UBnrHb705b8OI8P9O-HWsOAma4nnFzc7cz7OBExswQhax1-xesQYlyuvDzqRXABO7CAhyZ4Q36WJ4L00eDvgiEHeWNGOuxm-RIFH1tzcINOz1-R7Fg=w112-h32-v0

6b96200d-989d-4783-a638-aaa18568a5da

https://lh3.googleusercontent.com/notebooklm/AG60hOrn6py1tci5F31cN-AEyAQdAoHZPuG0myVTH39ci4RvJ--JkrekgEsu6SkNuGm_DKyEFcGf_SdpHBu_PKWddrEb92ijTeLWo9xrvYisGOulmCC6K3FbdAUXTizv8EuQsd7fUgPBSg=w109-h30-v0

f9148f93-c284-4c0c-b835-4a1e3d24deba

https://lh3.googleusercontent.com/notebooklm/AG60hOoKGtgCeD2OlPtquACcPfvPGaQR7bARjiqsMiBGEIp4x_C58tuTV0jOTedGj-k7TsIN3zt3EOt6ZDb1U_jqBd6FCq9hnkZIBYZ_wuO_qkT6GLWjHdb6X2wsnIu-NChhRxo4-IsUpQ=w32-h32-v0

16a5276b-105e-4455-ae03-19ef40d5687b

https://lh3.googleusercontent.com/notebooklm/AG60hOrz5ky2PhPyyUi7T4clUpRHV6VWX0dmz_FBiR-a-oPB8A6JPQMTvagVyyB4NmONGC86YkVsEnlyjKnO4P7fdROkEeI29NHsJ8g6fSIOJUgrqCJFu2OITNsGSGaB28yxdWERvwCn=w30-h30-v0

8021c5fc-0966-4a71-95d1-3ed282990b93

https://lh3.googleusercontent.com/notebooklm/AG60hOox6b1Ihh4RSxq8eMoR2kAcV5tuogrkgGcbBZm2yMdqpDZCHYYMAi59j7wRi1ERPwzPMKowwa2cp3ITyfd_-qhyp2JYr1a01flrJPmO0J3Decb0KtSEfqN7mpn1E_I_LrIUs849=w81-h32-v0

f3aa0151-6b20-4302-ae3f-20749b07048b

https://lh3.googleusercontent.com/notebooklm/AG60hOreKAAvNGUm8aDr6EKZSYg6imH16wID3phejRCkLkLfCRJdx46kuRwJ_Dwp04Nt4bpM1dHkKtwmbj5Se0niNRm5bJjGKJPeDD7GeHWbIEaMrqvhh5FVsZEtsYoq15ja1laxLHyF0A=w79-h30-v0

176dfa9d-11f0-4ec4-a2e9-8b3b30ad361c

https://lh3.googleusercontent.com/notebooklm/AG60hOpfsCJa8GD7X4ttNoG4lFfHfxbKU7ttC5TQrjj81Vf5l3ca2vZVLT7vISHZN5BF9FuZIR3YqK2NZdVli5A2XPaHI1if5CCNDPSNKpFGqx-fmiNfgNtZmYJTHxIeDoy-s73Bev1T2Q=w120-h32-v0

657bb07b-36d4-4b09-8be9-8b379a3e6398

https://lh3.googleusercontent.com/notebooklm/AG60hOqkRKRywIgx31GzwRIwGByt8CivdqrQzj38eKnPOU-AJLur0bYZRdSH5_eElUU-xNAldJ1JgaHrmjnTJMqku4qS7YTB69JKSEN-Vb_Hr_QxRPUHnu1XZDAu4-FnXRrWl977KlOb=w118-h30-v0

dfb1857e-f4de-46a4-b31a-fb0cc9f08963

https://lh3.googleusercontent.com/notebooklm/AG60hOo5rNKdihjGrhF2bm5iHGEbddB7CHMn5A8OASJYCIB8FN-1aPza1l89ezSZ0YHCJFm3eO_3TuYRQxkcKx4ePBgDmUrGAe0x504mgYbNhlK8v9lEYcov_Q6CMVZ9bBpPisG6uuDEXQ=w34-h39-v0

c6e7e0bb-fd21-431f-a02a-8cc244a61c03

https://lh3.googleusercontent.com/notebooklm/AG60hOo9a7YB9JYEWJz59zh7e4MyLqVAzPy185VwLTccaPqSikJ_pJYIIEoZHgbifLAFwvyslXcTLglWfrzk1JkBovj__nxYvjoG27nBjueyHkh_lsKXO8BCKW-OrGE0L03uqsxMqnH1=w31-h36-v0

6b3d7f06-f4d4-45e3-8e75-fc24ae4f2425

https://lh3.googleusercontent.com/notebooklm/AG60hOo8oic2KZuQS8WLOurNfMH1b0LPtypcYKw-LAn63kHFzxxjmVSf00p-qEBcWGmJ7F0rsiP7GwJNDQGMXJsopAm6fHX47YWVKHclSAWQYFzGAi1XQApynPZ6S-HD_1bEyvWhDjVUEw=w68-h32-v0

22ed6838-16ea-45e6-b2db-3b1a2a9fd0fc

https://lh3.googleusercontent.com/notebooklm/AG60hOq2bQh7LJkZVOtyuogNXhK7u4NVLpai8ybbK7JlzuvHrLI5j63af42EHMFhJaEV87lER-7Xgdhq0qVgrqzsDbAoE-vFPt3mLJXe4UI_al60gvjl6FJsr6ElmrcxrjiN8EoTnpSiwQ=w66-h30-v0

e320c07a-e280-4eaa-ac47-4a82ebbe20a3

https://lh3.googleusercontent.com/notebooklm/AG60hOrWifln0NzyFuWWfLd9631LH0ZZUU45-_2O6M2ctWFQmSWpH870N_kYWxKnti9bXss0SwlrzlXiQX8JfsgitxyJmUnB7SU6TACSWpw3J0vDSN0paULRrtA0GlUbrXl9npaC1hVqhQ=w46-h32-v0

7cbe5edb-8748-494b-9abe-94976d3c1932

https://lh3.googleusercontent.com/notebooklm/AG60hOphs4p01NDyUepU5ydEUwtotb6QrW3q4LG0O-tvhFNTqJDsrI4CyP3tbUOR9oXdlnTslvhmszZg2TSfzT6CC7Xo17V-tA7PI5b-hNaKrHJviVQggxRecpiVzIh1L_SNNB_m4nn8=w44-h30-v0

66e1c685-f61b-412c-be75-9899ec620f32

https://lh3.googleusercontent.com/notebooklm/AG60hOrtWEMULwcsryZHI4SV-0aQ2qU976nACtgXLK6QTLswWGsMAlgx0ODOJNfs3f7qS4vadFH1hh1tiG5cWE6YzN10mNJvwuW-whKT3zfy5-YXvf6xPMHKJUWj4GQ1EG5o3giFeboTeQ=w81-h32-v0

a35e9982-13e4-4826-b6e0-7a0917d50055

https://lh3.googleusercontent.com/notebooklm/AG60hOoh_JgjxD5fBRi7gdvZBSo5gwW5sijVCm7UZ5e1_aJUp0XwBvGQaWi-FxMv9D-Q0I32LqTMX_VoyqizD8mMmw1KAEagjVSmyeUcC-PbrgSOc6uH412--lVtPtd1-O359qzpLj0O9g=w79-h30-v0

36e4ceee-c50a-453e-bfa1-c9c38fc2e75e

https://lh3.googleusercontent.com/notebooklm/AG60hOr3DRrSKyGFajJ1NrPaRnJZeth3yuGVEUIqT_Ph48os_eM-vEl6P0tUYhOWjHLktzgYTobab6eRGs8dxVlrxvr3xZtBkBP1tmyh4gGLdk-LBFw96LTtJXV09n-a9l_-FoeaIFJ7WA=w117-h32-v0

270ac87c-f6f0-42f1-a468-4ed1ad34cf5f

https://lh3.googleusercontent.com/notebooklm/AG60hOqVZYe-bz4tJMwQt-AE0Bbk3rkCq7guIJiwuapSGKeARSePIm7dyhOnmFJxB2DKARtSXamv3YpN8NFtFe1zsALradt4-6ilkfDyDpfeIHAtCCO_85Hp7n0MENPtRKbEyKBb8h-rLA=w115-h30-v0

1fb24ba9-b87d-4d24-8801-61869d6e24e0

https://lh3.googleusercontent.com/notebooklm/AG60hOqmYu-8uyLh-iZsZgXJRTAg8Z2c5hcTjfdMht7pUTYMw2nL3EOSiildj_dnaPTG9eKu7LMiXU8c7eqlH155eFWhn-zYGAWMkAbvKhP6mfYT51_9ygpHJ0hSK7I_kvjr6zFRZ6OG5Q=w65-h32-v0

4e2d04a7-a060-4960-95fc-f832db8120c7

https://lh3.googleusercontent.com/notebooklm/AG60hOpvszl5p-M1eCZpnWwD35GC-Hp5AlC_JTIBP_TXmvRKIXs3nnRMRE_XjC1afcuN86ab8rrdseLIQSiFOUzx7c4mIYQKGxCv530qjEObcjNDr-tA0S7xz8ghmFdtdpeeK5BDQCfRkw=w63-h30-v0

5539d1ca-d09f-42e2-be5f-6c985aad8c95

https://lh3.googleusercontent.com/notebooklm/AG60hOpJYUo9_PIxkVZ7re6Gcu2FMlCoqtF3Vmo2eS7wgHJFuwSo2CM474Ug81oCVu9l7svGMBOPMZ01Qb04gaOXJ5kA3sbihbxnlraZ9xDwhGBK_vfXNJlW0iR-6LXIrkQ_0ujdzr12HA=w32-h32-v0

45024c9a-a8d9-4b82-a30a-bb5c9549daf4

https://lh3.googleusercontent.com/notebooklm/AG60hOpoIJEp7uppnyHljiKjXjywxYuHROX-uLDuUkeZuJWPfbnol32pM3fXMlhELPbALhn_n7lMO_g-dVaFtTuN5nMa8qKCGF3tIemCA0vq9ELX7iRnP-WIqPa0nMuAQjrG5OzB9EUX=w30-h30-v0

e35bde19-c56c-4334-8aab-e724065f21b3

https://lh3.googleusercontent.com/notebooklm/AG60hOo7_dtLKGLa36FeYXcPgrygx-DSJiI5CoUibn29SfnDYa0K70Jl-fPcl1lv8COxA0PC0duc42OSwm_Zt-_YQf_dhyP71N4O77arFIqgfwDCT8thAMkzCZS0XYvjmZia4VShx662FA=w56-h32-v0

a853c278-03fd-4fd0-b5a7-ecc5bbeccbbb

https://lh3.googleusercontent.com/notebooklm/AG60hOou0baiP2dmNRCyyqONOXJh34OjgD3P4pADKCJ-WN-E7UN_UW0WLXeSV_-KyMgcVwZOGitIkJdTjUfkC9NBJJ20SMq3krqKZQ2JKAtcKG_YH4es7Lns90ijqkFbiz4WZsFw9R9a=w54-h30-v0

70032951-dba0-447d-8346-6744b7a8fb8f

https://lh3.googleusercontent.com/notebooklm/AG60hOqzVMLB_S0-8oGN93vTfisOH5hMuhQzOv0q8JLGpy_f-r9gjlKN7nLTmgy87B63jyQatXkwVCALSwSN2hWiIhs4qqmjEapEU1RJ4uzpyez4-lXo5UcvvbzbR07jpynjBq3PoCBtiQ=w38-h32-v0

61dfd859-a615-478a-8200-8afcea02dec4

https://lh3.googleusercontent.com/notebooklm/AG60hOqyWOHgeQlPC1nD8vbNt8eOUIztYwPsUYwGoJ0Us_EobgZEQ7I1sO-1ii8W8tHGMiiATeQIgYuc3KiHwJaQjetswBslWJGpmpIhEEGXlqPMVtdfF9HMnvyEdV1xfRNAb8z7yBw6Eg=w36-h30-v0

6881664c-c18c-42fd-a6c8-d0d0d45cda66

https://lh3.googleusercontent.com/notebooklm/AG60hOoLwOgyyFbTvrib2kbpRJ4blSKwYSXbMNQLdKBK0ExHd34Z-_OXA7V35R_IizcMStwJE8YsxrFGtbpMwUG_0T_XCD8xbQEViUIeXlugB1Z76KKnfH8GviiSMHUz-b5ymdzZCwGE=w90-h32-v0

21e3241a-1837-440a-a2a7-dd8aa8d22bec

https://lh3.googleusercontent.com/notebooklm/AG60hOoELsdamRccEZbyzw4owfjVlegHJVUwcXcTzXsZ8J-LSuxAUFPBqvWVyh5GU4K7CBtDqr2mX5-ZF_vl2W8x59XUowtbjI-Re3UoskHqeEFC3ThkBWkbL2IfBSi-a8_Y29vzPi2fDw=w88-h30-v0

bfd9ee69-e540-4ebb-a948-f40284dc04cb

https://lh3.googleusercontent.com/notebooklm/AG60hOoiVOJ6niX-u65vxRxkGjBPNMZv9puu2Fa3_dONrLl47WjDNFbV0WLeD-BKsKtDApjbBManiLXA6WNPl64jfPotZdm9e95iZo23BA8IU5fOEyNQMRwu-exAGFM5DpL6rbFe7BlC9A=w28-h32-v0

66063f1f-9dff-4ecc-9405-e57ce72effac

https://lh3.googleusercontent.com/notebooklm/AG60hOr7Ww3sGxsSGixCYW-XXKr0CXEsogi_aeY4wBKMLHCbkZG-Th-mYwKpT7o2k5rExtYLanZdyKWohbUjWSy41VAU3uyaIemm5hSJb8ak22yGJn8YYWaJt-P8lW8qWsC0jUHnpCDx0Q=w26-h30-v0

e507a862-5126-4ff0-b463-986d268e3806

https://lh3.googleusercontent.com/notebooklm/AG60hOroCv0s6Nwg9qfmDi9x2R18MS4s_vLhLtquk6CahimV9Q6uKVondeyk6Ef-w-rjnMXLQlT7LE-GIUYeDKI9CjKLDxNtVFzAcV8V9PiseC1jnIVTFKF9nWL6CDG7hWrSTVpJAIG48w=w79-h32-v0

c44ef73c-35ec-4596-8f12-92729c3bec73

https://lh3.googleusercontent.com/notebooklm/AG60hOrRmqEyyd_AaSiq1EyyqyI-_UfqnjinLeFMN8BOdOG-68zTGM6ukk1IrNy5GUYrFN3IgQ1C1UAbGG96BzsAU3eLg5HxBFrNaLaod43C0gZBB4okCv9zEIPrg5bxvM5uf1_EVyyMng=w76-h30-v0

f1438750-1e17-4c8a-9fbb-3e1620f0e38e

https://lh3.googleusercontent.com/notebooklm/AG60hOoFCYMFfCd8Vcf9GUrRECUvT_CmdFOL-b_PqQiTfghMTTjM-IK0Z_nP_Gw8AcWFfmF9cH8snCLlSuph7wTKOqkLs1coy1aeptPIy4taesH8nPv0jQoJwZF5l_kv1ueUqwaAKQvQFw=w38-h32-v0

c9975495-f1f8-4768-bdfd-c7ff2c1ca889

https://lh3.googleusercontent.com/notebooklm/AG60hOrsS7Hgskoe63fpninC4ajfv-0R_F8-yJJO1TA3WQNBbDZK1JgAMKUO0dA-daTMrARecD6A0aIgThmC9xRwkcrfuhz851cT-gKTmI0LsnlTnjVrfMxuH5WbypBNDnoQw4x9Xh9g=w36-h30-v0

99672b78-3759-4ae7-9d04-08cad5e63dac

https://lh3.googleusercontent.com/notebooklm/AG60hOr2F_YFd74rWPFC2ePWZifAnTyEHhLEh06hhTrFEBPuCjJ--lGtNYd3EZ4vXbaDIajLfp0zoNBwiypH-E08grhwvwar-VB3dg9KX61nNbI1_cmrvYCKFqLoDE0BqobhB8tkwMV5Zw=w112-h32-v0

9335d17a-ef4a-4c9e-8d88-18490182d9be

https://lh3.googleusercontent.com/notebooklm/AG60hOpm7zVNnkMWjtN2o8v1VSxtk1Cqj8ho7yp9TBb01UPqxlYhmNnL2hLUAbc5JOU3ynKs8jfaceKqA0CFc8KmW5BX8anm1PZWaO_rRKZWIYFBeibJNQhux784AzCsR-ipPVQG5NiV=w109-h30-v0

1c490ad4-9d29-48ea-8149-5360a033a2fd

https://lh3.googleusercontent.com/notebooklm/AG60hOqFKRiOx2SJa9AeSLndbALXInAe-tr02-Ct4ydjg5tpUKSVNC9uC42ZNvSzzKklNzKeJtZXa7pVi11geBfjsHcP_869nhXN_Id4I_UlMpduneh-7mbk1O0nlDIZeQe4QGmLV28G=w114-h32-v0

2a3c2a82-5173-4286-b90d-a63252a80cdf

https://lh3.googleusercontent.com/notebooklm/AG60hOoqO1UdulxljoZuBvoVFIAMAipl6BcgXjlzjCLTvKbjs6FeQiIhBJg-58uIZp9zAvrefwisZMkE_XMwkRZIAorecjxLTo5C0ZYeXG4dsYjD3jVI6r7OeX4IHeM9FSV0LE3vTRdy=w112-h30-v0

1c1bfae0-a75f-4ee4-a09c-f81ae18f1b23

https://lh3.googleusercontent.com/notebooklm/AG60hOq6i3L4yZNQugDnDgCA-sRkSReTuzW6YlqkYW0pe4cFsiNY747QJ7LHV6MLF1sJ873VBsGLBKpxJBlIm8DK8yiK1tJkRzy3QA6cUcFWg0GRAjpAvh02k57VdqEwAaHclEjIB-I19Q=w33-h39-v0

32c25360-8c82-4583-9ab7-94b6ec9738e5

https://lh3.googleusercontent.com/notebooklm/AG60hOrqXXLQFb24R8IWerUn1iNQapcmeuWkkqRlPtMRaUsXWNfvr3N0KvErPXSueDqFQN_Hf8y0myuJj4J0t9SDydQOEd8ISv--Kim6JIRtRy5QNO4BKqlDhRE2atfl_5GG-XRLUcF8aQ=w31-h36-v0

ef948e45-b0ca-42ed-bc23-29158c9d09fb

https://lh3.googleusercontent.com/notebooklm/AG60hOpu_Ad8NUcifNwhrEqMgPFqs7Q60elnsAbvG9RL9l_1DF8gn0OprLVYodUOvnxQ5CSNrvP98un7zKF21vlXtSzyoqsxgmFV6df3va2JCFwpcFCJQFYlNyvMvOkr6bTPgMFNfy7blQ=w33-h32-v0

ad19bc50-e715-4d37-b43a-9f7bce67be36

https://lh3.googleusercontent.com/notebooklm/AG60hOorHtL0z_XNw3zh1Iloh7PUqjK1rWmAY2Qvj9uTqywwk7PhfXMAPXVYN7HYJxL50PL74Vx02slD72XmbLBPGhfNKdGew1jWoGBm-pSScK0xTPGyqNVSrnAPGIGFrRztit0aKBmRpg=w31-h30-v0

02987660-f2b6-45a7-8ca0-4e366d6601ea

https://lh3.googleusercontent.com/notebooklm/AG60hOolUIDn1_o6b5mKtocR0s0Hv2Wmb1z1e_N7u1PcZsuzp9rkk9wHgDt39v6JbnD5swvE7ZnuxCGUo6UV8SjxtU7e7QHbt0qkIidDCm3V4hQYmDwLrSTmVX4bo9B2ndXsVAU0X1aKzw=w91-h32-v0

c9f85c56-42e5-44e3-a5d8-cf54ab3c5820

https://lh3.googleusercontent.com/notebooklm/AG60hOq0xwsQxgllBqyI5VxhtAYyhxAz-obC2WYnHtciwFBKGmnZ21OvVn_SAMRFcVR4KuL5TSabSUiAHuCQ6YkjFzNEU9e2M7Q-qj2pd2DHfgYJQBvJptaslELFxzvumazHoqhCAhOV=w89-h30-v0

a05c0109-0107-4113-b652-db8572115ffa

https://lh3.googleusercontent.com/notebooklm/AG60hOquGhj5d_Ec4mI2mm0uZUx8RzWi4cT2HkQkjoTwP15jS0BucyqE5zAouoSZRFFoYChQmc7CvGv0Uh9nsoZKMOHOjLX38kJ4SCOsOdQ-X-c_bZpDqfk8lsY6jFnYCqkezUh7nLOlag=w38-h32-v0

0b94eb1e-d1f7-4818-a358-efc451f4340a

https://lh3.googleusercontent.com/notebooklm/AG60hOrlx79RywiBGgb43tMX0wWf5OuYISNa93SjmQ5uaWLtkHlxs2z86KZ_wfJEQe0tiwwWq-50cJuT_cjOcsMPBZtg7x-9vEO9l7DI5Jo5lvlTjeSzFBrgeraUJdI93VeC_7CtvqiQYg=w36-h30-v0

88990a4b-a5d3-4270-a834-893438aa387b

https://lh3.googleusercontent.com/notebooklm/AG60hOr7JRyJ3ggOhtUDh-QCGT1Fpr5XePV6HBzjujsJtwGADmPMPDIytl9wCpAe-ia36x-J9nqilbRs23NBXSyvTKD9JsjeUGbcQx3GUjBkX8RAN6cHUR3fXZVNp8tTXUGrb6Fatp_dOg=w56-h32-v0

d8c16e6c-d699-4190-b7ac-6e9bfde37a7f

https://lh3.googleusercontent.com/notebooklm/AG60hOpF7C0XUMgV88ouiDBOjyU5YXd7xqKLM7EFRhj2VRR98M60WRjufD8EB-e-cZBNmxXsKPail1dfYp8kgJg6MmpmYoRSn8561NmZugrhzPoT40ZLbjfv0kxyHp3elECsyTudaWOc=w54-h30-v0

053b1e53-9f68-491f-947f-f011c85e25a9

https://lh3.googleusercontent.com/notebooklm/AG60hOo7Ccj-Q-13_PM_MJdwdMrvfZh0ZN8i_mavrZ3_Rigb_ll-oEyXmcfwEMB0y4fsmaOa8YXCASZv7hrONFPEbXJwM3fwtazUI_v0XhgMADHoLyLPgwBfhtSh0wXL6NbYT1Jk-fDs=w30-h39-v0

ac53b64f-2aec-4e57-afce-a74af586150a

https://lh3.googleusercontent.com/notebooklm/AG60hOrJmouFuvfFVzsKPhmIQ6a1SAZrxm1S-GRZIpRWZ1k-0DGwXMXBhstwTxhbgd-BY5eMWTijQDZwC0Rm4W8rOVqxOOeFllb9b-Ux2E_MmM1PejDhqVafZTYHT_fdkj2G_KqKAIHBCg=w27-h36-v0

281c7ed7-f74b-4066-9795-a3ae59d47fd8

https://lh3.googleusercontent.com/notebooklm/AG60hOqE04XMIqQiPwgA-NCHs_VJwhqeDP0MYZaxoi0DXqejgHc0Zc8krVaKqySMxFkixSplvSc75dLx2Dxz35LIKgBN6Km1b80J6ndHnnr0S42h2hOWHsUufEFwgd222s4iN67wHHqepA=w87-h32-v0

5af6122f-3000-47cf-a63c-6930a0ce19ff

https://lh3.googleusercontent.com/notebooklm/AG60hOqsLLzEniDi7lQtbhe0uuuvJ7X6djE9oElPyVWirXByV6USZT4By5WjsOmtnmSY5cDx154sFwC2a8LnKSx6IZubwafz8n5yWSX_LvdVepIyfonwxeB5QjlqiOOpEqBqSc-beEUs7w=w85-h30-v0

81fe2bce-88fa-44ef-8bc5-5e84add94e79

https://lh3.googleusercontent.com/notebooklm/AG60hOrw_B5DpYrEIclVT1P_wJqVZt5vAepsNNFMRm4uDrQ2Zd0v15Z3iY39cJ-Y_B3dIKTTQCkAtwIUb-lGXb3-ODjgGYW59EP2VAzkF_9bDmunSDm8Wffr4ZGCURoslfLHDGFnZTVf2Q=w52-h32-v0

e4b19569-f31e-4ba1-880c-e69bbdbb3d5a

https://lh3.googleusercontent.com/notebooklm/AG60hOoHABlCYUfIKitBRT4nvUAe9sJ3-FTqtfUcodzcjRoHiEYy8UsLrDUMHaQ3_b0q-7K41dGNWr6hnZkVNyXLXVMp-6WHVhO7SnIEiYfOb8aN5qftLij0WEGvWwS2bNFy_Fg8yax6VA=w50-h30-v0

9c0a19b1-53db-41a6-9dde-bc9f1d49840b

https://lh3.googleusercontent.com/notebooklm/AG60hOrx7sz88OCc-xPZAtH8nHn7LbecFbK6zpSTupGN_JwcAhDtD4LELVkZ_828TBIakf4jj_duy3Yv343s6Egkv4HiA9wupSszHo56pjvanyAmt9OJ0MbpEbjuIh3QJPTKYHV6-kE6=w30-h39-v0

dc0d2bec-183c-4ef1-b26c-300248f126e2

https://lh3.googleusercontent.com/notebooklm/AG60hOq4fTHfFyMhRhxBEHrWi_Bkod5PwlrH_EQY6zE_TgALWoKLklhwZ42EoqgM57BJPY33ohb3Wcv5u1W6FDNDWrsFu4-uTFkPZ_T0byszw5le235Rw4NYHAT5hSXHoZTGZsorNmHK=w27-h36-v0

37494961-95d9-4132-bd1a-f6ca33e0fec9

https://lh3.googleusercontent.com/notebooklm/AG60hOq8EnvMTF4DabMryH2KmxUCi3paLdYC9zgPS9niQgO_gTCEaRSP3cm42V052ooWE6ThPiAWSqP5jqk1eVMQTv5XVhCTu-e15tgI_cfbdC5FqRVRGhXpb0PLJhJhc7WsCKHg9y36Vg=w60-h32-v0

d4f7b8e9-3477-4ff6-87cf-e83b684e5305

https://lh3.googleusercontent.com/notebooklm/AG60hOq0klQepNGIEYn5K-Z5TNWs03tLGzEEGmqfwmTTRldgE_bMz7Ll03hcotifpUk2ABkl4P93ai91JPFICipVRnAKV2Eg80R1WRUY-kejmri2BugH337HQactSiI3KHI_VVUOgXYuKQ=w57-h30-v0

f6730a02-5644-441d-a2ab-d0dbf94b6d6c

https://lh3.googleusercontent.com/notebooklm/AG60hOpL0hdQm-TUnx5V-fLHL9EULQpmWXIwr8tp5-MAFs2QTGGMoEjhinn1D-CFuA7WdRcBP16ay0nhJNQPqLrxOF-9DgxwpsHSOCM8RRetC3dvljF0SqtnT_DS_7SsRDjSjcP_YLTIQA=w118-h32-v0

64131771-d6a0-4fff-8d42-35a443be288a

https://lh3.googleusercontent.com/notebooklm/AG60hOr15V0O6RKl_AQSzm0RZNX8EWDlamgZVaSWhdgzY0JePvurhCLmitgnkrj5lqXABfx27z_jaYfv3UiRZeM_8dbVabNYkGPSVY4rbnZ2wx4nygkqtCKnbvZq4f_V8GQe_KxtvFY9VQ=w115-h30-v0

e237014c-ad2b-4590-898f-3de216f85e30

https://lh3.googleusercontent.com/notebooklm/AG60hOpYh0dAL9I5jMSe_JMLCyx-_b62uWKKb5tv7E4QQ2NHUNCvPRbARP-5X9gNCK54mYWYf55RFOZP7Osn_z_KCe9Q3fMXJXvRwJiQTqDiudY2tjz8tVJKmg55kwomsHLyghGHTOuGrg=w29-h36-v0

31f959ea-bb6e-4d47-b582-3ab17208507e

https://lh3.googleusercontent.com/notebooklm/AG60hOpjkqWE61Zzs6RYwyQkG5prXx8CWCBr0u4EV1WTx4A-YaPSwxHimaz-4D7P0VYhIdUJu05eqeWQeHJSGeYcftgxlXj--TbJmfakZz-9bCcGv8LT52c8F8jUahThjxipuIxT6NQN1A=w26-h34-v0

56d4635c-be86-4831-bcc3-7895c1f55283

https://lh3.googleusercontent.com/notebooklm/AG60hOpSZoYcCN8RO_kEdhWvKJCOR6NioVJEJmfiBQnPpywWOToAGZahbWTJXmk3a9wLT2Pz9hOWAJy3q-jYXR7xgHVf14hGGzrZ3_YSzZcaJewXpTKHEzesuPNWOjjGXnyZizD1xFeY=w46-h32-v0

5a1a3c05-5c95-4466-8c64-d332468599df

https://lh3.googleusercontent.com/notebooklm/AG60hOp8hSca7vz1n-SjvEg24gPC2vOX7sqg6wQS3nHpiPCh4WGZAY0-toCQWgqQ9ZorhjsiluKfeTFG6UMzV7E4Rz21HcsG13HE3Ffyso6LvZBp50ZOgsUW8f8I8aiBkbMzCR3iTKbdZQ=w44-h30-v0

9c8c37c7-ec75-434a-82e8-8bfc52dfa382

https://lh3.googleusercontent.com/notebooklm/AG60hOrov_QlwVsJUtSNRcWQY_GBcvB7BXfySWtvUzMjcjTTJYEVZCDjcGtyQZbjrySqfqVO1RNYhYNG8EU9wVJlHviKTV9u2RvWP4rmt7_5NkIJvSp3SlI_kQpncKq4NU0J85k1Ad1R3w=w56-h32-v0

659ac58b-cdce-431f-92d3-41c26cb167aa

https://lh3.googleusercontent.com/notebooklm/AG60hOp2Jnk3pOAEtD-vPA1otRuuZaRYgqC58Yib2EnWdaW_jLdHexjTmpHAqpuuwj_KMm2mJndUJRRIGYhwM8brNsTue5O2q1KmzSXTMciWVPBBDZy1WNocjbQR9-Cpd5kTprJa1fzZWA=w54-h30-v0

d79f0245-ce92-46b6-a918-7a4b7ff00c04

https://lh3.googleusercontent.com/notebooklm/AG60hOpU0nTvG-AusgJDJJX-6uSGAPctHcWPjTn1J8Sswi3yEeo_SmslEHIV8dISPZnxSWrC7TNPUThHPxFO4083iITVPjL2Vk92wiobIS2dOTBpc24NGhs1-7jbFJ4bqZKhSXpfLyTi=w59-h32-v0

a4b3a787-eff9-41ba-8bec-b92dc00e1a4b

https://lh3.googleusercontent.com/notebooklm/AG60hOpxwbfo6j1VhBYX-htSdXPUE4FlwBIIQ2J2TEesRqqANz9L4l3el20r8ctCIjfAwQwVH8KDmEcz9VQTFb0H4PrcJZJtqxoYPkAFAYR_pL0jB6EOLGF5fW6vuwGYtpwwF0ckMvp4NA=w57-h30-v0

e0757c5f-8247-46cf-9656-4cc4a016f745

https://lh3.googleusercontent.com/notebooklm/AG60hOoBYepOAqAJdkoOT3OXHVaq4-KhR4AwCXYQiljGxlfSvOaCht7-xEeXWV66COBxUDzN0U9-3-SDmmcVBVolQwsrugxilw4OJYUxEGKyWLBcyXYwYlC76Nnf9i8hkOZPFn8hgBpzlg=w46-h32-v0

7243e968-e305-4904-b94a-7bb43b4fa111

https://lh3.googleusercontent.com/notebooklm/AG60hOrOWGejeh5n89QFirWf2Pf2tAiW4wpUGm9LDBWUnlFIuiSEViOTCmqmfWi0LeRuuCRCgTUelC6wHR1hognWIa4upQzHsyVMLMyhvmiyHPpmLfnreCfhovsozz7F84k7Ksu06vy8WQ=w44-h30-v0

9b3c709f-fc53-4363-bca5-c413b4bd6492

https://lh3.googleusercontent.com/notebooklm/AG60hOpMh8WaPro52TwGHeOCawuV8V6oWQtkQhDMYUxTKUoJGpORPQoDEuRF3APmzFrXF8S3Em_JoPDAhFT-sGz8Nw1vK9ilNX76bLSdcd1WC7ANnPqItVZi3M4_r2F6m_c-4plrxIjvVA=w77-h32-v0

800aa5e9-2e47-4edd-a497-59725405c1a8

https://lh3.googleusercontent.com/notebooklm/AG60hOr1CisRlsc5qfYEjKc0HV5dPC7jT2Wtsjw3Qs0T7lhu6YXbFT09z4CoSsTYf4zNI43LDAbg6QuPGY1-Cs3ld8ST-QLJv8qwpb4kMXPMglvnUwrQy9EnGUD122d11KnbFOUUG7bttA=w75-h30-v0

f67ca7f7-ff2e-4849-83ae-3d5ffe4d6a07

https://lh3.googleusercontent.com/notebooklm/AG60hOrWS2ceyNIL2pjeHFkj2rqvkMKHWWXN8j4WG78TfxxGxmd2b3FFfQAC-yyo6DBUF_VdPu7U7BHUGs2-Z7NDpUy6iz0NYhfpdZ43v9878_hEkTJvKdJ3iDJ1gM4kAHDmWjlR3Tx1PA=w38-h32-v0

7adc9a1f-8f47-4b63-8f55-a0b83447c2a3

https://lh3.googleusercontent.com/notebooklm/AG60hOpUJyXplqCL8SWbb79z5jq9HaNPTGMEksXyNRnMiqpB0BKetOn2AW2a2TCf_f_009sQ9dHcZ2l9Y7thRYHRV5USrF6zYh3_bhufiRanHgqlfwbLO3INsYOxiWmxklZm7CNuLtC9=w36-h30-v0

86635880-f142-4275-b393-3c4c8025aa3c

https://lh3.googleusercontent.com/notebooklm/AG60hOpXzw4knuCIx1EAa6ZLW3H1paFoDriy8OhyglqvmI04M7b5GSR2KSpFBpQGO1o66d1DELLI2DcQ6-uJLeiQ2d7QCsNVswCOuMFZXQrdakk4vmHbfWGfon1kjMco9YRsvIkvJD9CYw=w71-h32-v0

937594cd-dc35-4ac9-9dfb-d78f6710e264

https://lh3.googleusercontent.com/notebooklm/AG60hOp4bsnzABGHioLkSfJZgxNP5nwzgqNBRe9VEul6DijRNK3NmPmnb-6lPcRNRqQqntql-K-Iuy4ktADSQas7PE1-YAw2hgr0MwkdTeZvBgydh-UAO2SGTeX24vqBgfz2CVp1LagLgA=w69-h30-v0

e9ba38e0-5828-413a-a1a0-ecbde3b2e531

https://lh3.googleusercontent.com/notebooklm/AG60hOrg5D7XOjPB1-BCkuGWem0-GtYddkVZRbhgrQ7gkOqCKs1KjQmAxpg2Utu6twaYNbdaWGS1rvU2lJmWlH9fzmqJmnxLYx-t-yeLiu3Ce3fiJ4yWYjDRMuiPHVQCTFUFYrpb2ncT3w=w46-h32-v0

d1d38b54-847c-4bd2-b930-b88ac4a15e6f

https://lh3.googleusercontent.com/notebooklm/AG60hOrUKrl2RzmL4TR0S0z1cXfGlsnoOaaDpEOgJ227Ew21Sjr2wHv3lbRmKuZR435TLBw57Zf-yyiPj8srYixaugFxGweHUJWHxKBzLRppJibq7m-Tirgj2XzYpnZ61VD5HKDYjE5r=w44-h30-v0

2bf9f929-04be-4fe1-8963-40e69a1a7584

https://lh3.googleusercontent.com/notebooklm/AG60hOqb4eZtWU1WCc16BP40DXmkfXl0F7CkEUVKNgiWAytS7qeLtDa0heFvrYNH7WrXu3N8m1JdKYHRf9Yd2sf9Ripc_xglZaZWG-kxdX66Da679uQDbfBXw1FTEay380GEXjMCh7DsSA=w56-h32-v0

2c464988-1970-4687-b42a-93d78eed9692

https://lh3.googleusercontent.com/notebooklm/AG60hOoSXqptH5c1PaFUGpOhSM97JlVPqFaFDHiqtqWRmMF9tFw8wsjJ_4zmyjdqJVbKyQeJXfdP_NXwXrJard2FzUhBbNJMdbtmJPk2qVkn9Xuz8pFqv0MuZ7fZKCNWoKUleQzvRB92=w54-h30-v0

ab49ba7a-c7c8-43c7-a312-59ab0381b6fe

https://lh3.googleusercontent.com/notebooklm/AG60hOqbiVLskM0-MFLDPUA1L90rMg-qsP4q4PmGt5hQF6o3XoQeXJNqNSIvy-dEn0HRDs3_ok3IvDMP1YYBOyGl2J4-eXwh5hJP7jwQYH_tzlDjJPf9pOUYBxUP-94Wl-0Vpkhgknc_DA=w54-h32-v0

e1c84370-fc8a-4791-8bfc-542ceba2aa0a

https://lh3.googleusercontent.com/notebooklm/AG60hOo5uws2Q8QtxZO3xYmnOzxhG-0R7x6FD5s1c6rWQNPQeSDIS-HHq1UE-IOMtktocx5944XqN6hOQBfzlwbwY49ZLEsicwV7RFGAZjhfwNRsq3zAQO-1qRPdSRsyRRBUW2LBs--2kA=w52-h30-v0

8944aa35-6e12-4a33-99bf-22482c6d2387

https://lh3.googleusercontent.com/notebooklm/AG60hOqnbprMye2zayzbyAQO54ALPkf4RyOnbTHi1y0XqiTym5lpoGYTd3v4TMmHjEiFfOa9ePKYqMDFCpENATrahCpOYgIFG4qKqY-y2SN69nw1C-Oy5LbSFvQbc92VGakota1VSzIEkQ=w48-h32-v0

28444c16-37d4-4eda-af18-e916d8f1f3a3

https://lh3.googleusercontent.com/notebooklm/AG60hOr3dnIi-Ef8IxNIkydCsMjb6syaHCoqrVm2jZzllGrm54VjohW4JevfSKp0PRABHYiOr4gddTVEi5_YyjdmHFD7B4-Wg_0lkh1wpBIMs0ty_5zruE4Vwsaag_P2DtNEz1tmPP2yGQ=w46-h30-v0

8595377b-f52d-4063-98df-5cdab972c337

https://lh3.googleusercontent.com/notebooklm/AG60hOqnMR3wOO14-CCVRc70HFmG9IHOHpjcS3M0vVkKrZVnMyZ-YWQKiQCnpx0jnkpTsAHUkEAC5b82Axte1aAgdMiN3u7L18A5m3kG6F0ZF7EuSnLWFlkUvoDdJUS2UQe5CnPmh_xY0g=w72-h32-v0

f704cedf-4365-432b-b21f-5c3edcf32176

https://lh3.googleusercontent.com/notebooklm/AG60hOprScxmHhTzTLAvA1uUvN08fgEIIDTDEixR-40vf8VAqwwLa-pAoz_Ox___LmGkW-T8gz0b3szfdGR9Rk_VYDLVd0C8GTws5zTvk5Y8jxn5xSczvLHvd3wQO5rtRJwOsYhDRdDq=w69-h30-v0

fa217abd-858d-4fb0-a276-88f9b761cd27

https://lh3.googleusercontent.com/notebooklm/AG60hOqqPbAWUUDVOBH7ptlpmevRJeWP4bzfwDtPxgWNO5JGsrqgjBF8_CPDVa465SJEGcZZxjtezlWx2fFRdrSaOOXkL7ZW2FKkBINtojlpTl39k8YjEepNbP-TalRBoUqTCmMH2WF_3Q=w60-h32-v0

38195e91-de70-4917-a025-33383a919c58

https://lh3.googleusercontent.com/notebooklm/AG60hOqDsOimeDkAua_aOCkGoqEi1aNJYN9e_rNHawfV_9xu-6n0lPZfYw9sQ_uDONrb9FCwfOzJQfjXxlrxjdUePnwhWS-xmQAfO_QD7fzKgsosl0tFMNFYrG45D4XRL8oJSMqOZFTSMg=w58-h30-v0

f77f8b02-2b9e-4e40-8cc6-d04039a4fd3a

https://lh3.googleusercontent.com/notebooklm/AG60hOoTLkvWQisB3OAQ2ZUbcr1negiDvxgGfVyLwmAO4wuVCNFPiDE4tnLs4f9B_wn04UQHLG-ibWgFRNKQ1XnhSS08zXPrDGrsiwTPG4aeAnIDfRKXzK2k5y-CuwsbnQGJcTI5Pz2sgA=w116-h28-v0

bf3c5d8a-fe5a-446a-bffb-660e3c0c1f64

https://lh3.googleusercontent.com/notebooklm/AG60hOpnAbdFBNK8szf91X8wLcHaQLIlYPKPqELv0D6ZwMh3mRNuOy9tb6-ZvBTYkO4rk8nzE_YbHF63Lccx8RQ5nK2KkBPgVeDpgtUR-NzTQ51HTWFAbOyStV7QbxKpSUosfHBPiEKk7A=w29-h36-v0

0f1c46b4-dd09-4dc3-8db4-2f48bac1261a

https://lh3.googleusercontent.com/notebooklm/AG60hOpb4ACxDTGObAWaScYfWYCvZZyWCY6tmxa23bmpdYdVMlqsOyPE7_mCDBw1axEPkTLo2NXumq4n3QyBzS_Y3R1kMcyle6gRT13aDOMVJP1lURL6VDWndbycA55N5wpiYYTHC4gC=w26-h34-v0

1229c3f2-8c9d-4ac5-b717-6807e5c24292

https://lh3.googleusercontent.com/notebooklm/AG60hOobl066FGE01w7H-jpv0gOkKlP9RsjUlUJ66-XsoOMv8hqGVaWYsmlUuFyw26FFQLNm5FhBZWID9zUGj5N3LFFjfB8ptTOci__Z1Fdp9Ps1nMfDfYX8yZZEmfimEtHgs7xPtbKlug=w36-h39-v0

789568f3-86d0-4ee3-969d-5ee05d227586

https://lh3.googleusercontent.com/notebooklm/AG60hOpGyo6-0wjOgD21DBCch6TxkL6YvG5HPHcoFQvfMe-lRGp0G8-RbwdIXDyM3jkPvuoHhwgb13--TNQVL_-A7F0TowV6Yw_lPtEOnobkGjJkcMyr3Uw1nHn7ibZDV_mFLHyVyYYk=w34-h36-v0

44ebe8a2-162d-4889-b199-f7a5e3a37f45

https://lh3.googleusercontent.com/notebooklm/AG60hOqXEppbaDRkZBmUtOKFIOSlWmoRYfdnJkJRzqnNwPW8uGTiu4oq7NC9QLzln-1Ib4N82hcn3KRrlx_wPlFJop-EP73b2lpLv8gy3eu3cYe1J0eXWo9nBH7e0KpJd_pvCl9xkKeD=w47-h32-v0

a9657eb6-7f3a-418f-854e-a8af7fbfcf02

https://lh3.googleusercontent.com/notebooklm/AG60hOq-7mEyB4oxLKEz316QzuIuzX2_nbYz0E1WEOyG_4ttdwQ7P29P70562QdJyHpBIHb5dC8mZd1vkyUIz_EHC-1NvgOeaHXXYpLny4BKcR3WqQkxQ5b_7QG5V3JRuzTxMQTIRUgu=w44-h30-v0

bce5a1c7-dc9d-4244-89fa-84ce05a0c3c0

https://lh3.googleusercontent.com/notebooklm/AG60hOqF7sP6wtr-R0ndt9bT2yqrQvlmwUFb1NVlBJxRPeO0tPK7ylt_Ke9Ng2YoV-9ZTL7a4AQlOyVEHjRpftXR1aNSjt8Is9oZhawO_hvkB1lB25vXLIHaLJI0uLJLhjUxX6HEzb5pUA=w92-h32-v0

a7c92150-9bf3-40a8-98d0-9d342e211d88

https://lh3.googleusercontent.com/notebooklm/AG60hOrCCT8D7WVF_DdV_fbpHWsgjOgpfSuMMxt3LVGz_s1U8bzJcEHjETmL0p3T23Nh_icCn51YpD76jrXY4binl3Ni9hl5MOxB5DwGisBPCdBIxxBB-vp7va-5r0Q-e7WpUUtd9hvc7A=w90-h30-v0

67a69207-ece8-43f1-ade6-476e77d0342a

https://lh3.googleusercontent.com/notebooklm/AG60hOrxnQVY-y-kZM8m9P9UzIoTRbMJIn7_1WgG86B-_IXfhPGxgZfY1xO766fChU4goWwl7K88LkoY9YbV2VP2y4yKVkX0iOm_tJYahwP3Eh40sAYpj9PFedAF6lyBgFbWPpXymg80nA=w74-h32-v0

bbffdc57-1216-4d4b-b416-71ed38e5d698

https://lh3.googleusercontent.com/notebooklm/AG60hOo6fFd4vVNmJliFEmMy2_i5uc7CjupH9ne70teER5A28mpsuOh2LtMRb5jtftyqkU7Vl6hq5TOXQRY3mqNBcVKOIcd4lwJkoMgw86cT6qxiLpkbiGgwvWJacGnsf3a4r27wdA3Xhw=w72-h30-v0

c257bf17-5d62-4174-aa53-b5fddebd13d1

https://lh3.googleusercontent.com/notebooklm/AG60hOqGpxCgjG2uWh1v7iNOJjc7ZebT4xtA_ZkqrJTPqRcNGiuQSKfGnci4cNogOUkz8Eo_CitJBef0YUtZBOc8MiwJz_HL43h6-xVnEvzZcynum1o3mOoyOGgxHexV9Sf3aKsORDr0=w84-h32-v0

67e5f8ba-4763-4bee-9c62-083f633de2e0

https://lh3.googleusercontent.com/notebooklm/AG60hOrIXV5tU7d72mSPMJ5AxYgV8cH0SM6pCGb6WtWOvwW7Zrldqn9ktnYebbUttoOH045ktOOO1d9dE5FWmj4UFdZN50pkiCQX17IRmZbVTys_OSDEZgukJh66j2gZst2azrCHf_4lQA=w82-h30-v0

3ee59359-6d05-4ad0-8baa-4403656b94e1

https://lh3.googleusercontent.com/notebooklm/AG60hOoN_NuTKtrox-wuoSH0JYr6oJ0hUqhmcdUG20UUxpnnWkFdUkkv08qmacPdoXy1Bm6dyqHnbcEzc9f7OSy3qSJM0oqKmRWEMCWdgs1uT6mpcXJrvZdRRS7eb2cnx8b2XWOl7CuQ-A=w90-h32-v0

5fb20b68-2ee4-478d-9d82-fa8efdbb007a

https://lh3.googleusercontent.com/notebooklm/AG60hOod1JHIotx06rtoYBQ0ErMRpiT6SNSu2Em9zczgoPGDloP9eATJVADo-hdOtoQzEtSNEHjGcd0IWkERxQAQ6vjUmV3YNm-9Erhx-JAuN8hGpOBu3TdZ5KRaEYoImdpSR5EuaEzw=w88-h30-v0

7a13318f-6850-44f6-87ff-a54c92f49ae8

https://lh3.googleusercontent.com/notebooklm/AG60hOqg0ajjleaekDR_gU4lCXsB3eOldV4PCoELHXWPtLLny0dFkgwZCfm9QAUmdoB484js_fbeEZp48nP_VGGWqZTyUarDZd53ecmgU9o39PN_CR9Nvnwjj7sjC9M6x1dOgwUeB_fP=w50-h32-v0

1cfca1a9-cec3-4e1e-81d6-dedfeb1b2b17

https://lh3.googleusercontent.com/notebooklm/AG60hOojs8qwX8k3mm-dsTRvZN53XXkiIljPrnJs_7PgbEsTivOCFI53Y5L_R2Ed0UBbsaGgaIiRucNwmDYNnYmB3OtkOG-iQJp5o6NTXbPQr9xS5wJLRz8XJZlsuxuvaExqtYAFMH2G=w48-h30-v0

52a395c7-58e2-47a8-9b4d-5b86c17b2697

https://lh3.googleusercontent.com/notebooklm/AG60hOqNTZZp-faPOrtWlXYsqybSy9Iolea8_qp8IPNuA9qiQ5n6f6FYKdtytw6pd5ajyHly8aHk_-XBY9k_D4dXqiz8QwjSfuxpcYxBcS89DXBhiY9I6iggVmdO0I3JpBJXmigvsMTgWg=w78-h32-v0

0cfbbcc2-f156-458e-940c-2a3381ee79e1

https://lh3.googleusercontent.com/notebooklm/AG60hOpHxG_ujq1T-lxMhWUeC0IdxvPKj-iUHJSjCmj3G8V9mXGgCJFhriFXqIswj3Cx5Emr2WQxWfw2oxMMRuOXAmh91ABvPK8F8MwPtr3m_1QqViG606fYUrf1XVA35mDyzzn-DKs9=w76-h30-v0

51e861f6-bf10-49e1-9e42-b8a2daf42c38

https://lh3.googleusercontent.com/notebooklm/AG60hOotf2bmpNRnCi9k0cu-ptPsmwrRIq12kQWgOwXS_dGsGF14ieNTBCZWb_jIkABDxd_VITcK8TcZm8WJmTAwMNcMH60sCG51381RBuVqGObLzeaF-kKjr_mGFelrZX9M8baRElD5=w29-h36-v0

d7379ff2-e792-47af-aba3-bc72eaaa1dda

https://lh3.googleusercontent.com/notebooklm/AG60hOpHD4wDrdttuzppQZ69CRn1QZzVEQtG3SBHioAkU0yk_Jc1TOZlX4GhylsXg0rc3cdz29XIcogm-eLZWyoCMRxu47E8H9K244iBlPn69WkXmJ3Em_w4t1jyKUnYABVjMujsGHgt=w26-h34-v0

8fa5bfd8-9035-400a-ba8d-6c23818fcd80

https://lh3.googleusercontent.com/notebooklm/AG60hOoaWPwuaDWEB64Z7dxsRcizwdE75xMHOzi4ALIQ84PKrSjx0jSJ05g9mRvo2W0ctO33GcuWFBQwbnm8lSHQMuVVpFkqR9Rcncp_HpqhgddDN3CTBTXWrN55IoLp6Ai5RKYrKDizog=w46-h32-v0

11ed3ed3-2b7c-43a8-b2d5-05285d0b039d

https://lh3.googleusercontent.com/notebooklm/AG60hOqwAGooswzPxwKBUiVQziyhgv8lpBmRPmy4EuTAhUMgi8dK8aUSf1nB2bSw0nM1W_IFdGYLuB3fv__kSZNa8gBwqYxzJxy9PiQxrhKrWA6xF1UGmYqur-9HbU3lwQlvH5qRSaD8Cw=w44-h30-v0

7f352ad6-1e70-49d0-8b92-73af2f49ebc2

https://lh3.googleusercontent.com/notebooklm/AG60hOo-7uIPa8R7RwGNSscPumw4XvpwTWN6pSIs1Se0xjxq_YQn32-jjzbiDztq8fFC2uRU6gVSPQm1v2F5soeiF9lTbOCe7tuN9Vll4KvQnszhKhxeHkNXH45S9siTbY6u08LyGddILA=w50-h32-v0

c0825921-6ea5-44dd-bb33-663ffa3ba726

https://lh3.googleusercontent.com/notebooklm/AG60hOpTQeqTix5CcejV_Bw7xU8gBxRKJmPtofyJyp80MZh58Zh0wypvI7OhYe6X5Pf-lee3wSaaCFjE3pc3EoCBb-924q887ncM3PFT1n4MrV7o3SD4Mh_SYhJMtvc9kIvusNHLXgGi=w48-h30-v0

fcc86bd4-d510-433e-a169-da77e02784b9

https://lh3.googleusercontent.com/notebooklm/AG60hOr416Hucd1Rxxf4jIu1W49UVYYQTCcXZiULqxuhri27L50lk8V8kyn0kb6uY3YyobmP5te0BhXPI2D4sknghw2Hj5jfwbXjpaHWjZGhEg-grDawdTr5eD56ypJgxDOWxEW9sbyoEg=w69-h32-v0

fc143fb5-5c42-4101-8496-a8e2c0990c69

https://lh3.googleusercontent.com/notebooklm/AG60hOrJJplRel3o25Ro6O5O0Ex6-Apexa93_bVdTDfciSf5hf-kD-7LnhnikpdTIlkH4fUbQSMxueyUl4weTEClmN6jp_hDo0jLWaA93OkcQ0L39gYznb7WwYwmk7ELTiqbR1eJDdAoYw=w67-h30-v0

bf7d0a90-66a4-42de-bb01-60c4f88ee302

https://lh3.googleusercontent.com/notebooklm/AG60hOomkbLK_dakVR2G8yzWFu5sTPrIBCIyIPlhnVnwBNfdsHWSIzHUEUMiD15kNvaM9iVItTfXBVITZBzhocjNUyh2--0u5XyiRdrPiLe5L6reIcMLbLVeKnn9PysR9E6Ah8NAnoUsJw=w32-h32-v0

4fd6fd4e-6973-4959-af35-d24538f4e08a

https://lh3.googleusercontent.com/notebooklm/AG60hOqTMkR5EvGDRFrj4o6Mi6rGmf7crXJGtlFqRjmniQtplm0UNnkTnitzmsI3t4iESMn-p214a48vU4DtXJImYCJd3c99uBUY5h7uyIf7m5rvzdCVJrChSYYfudwowH0x3fzWcKiz=w30-h30-v0

e1c94d5f-3308-443b-9e21-e971a02bae5e

https://lh3.googleusercontent.com/notebooklm/AG60hOqqOlFwjF0yRFGLpRB5ABxYhKIpZTw8QqFJ5SOCuvNWGx7188OZoJBPqVG_yZgDno3kti9uD7drPdUa1-5dYQk2MBWMe8lN7QcW8qWGqfXSekSVHkNa_lyzUBrzGMpL13ICY9hu4A=w90-h32-v0

b024e74f-428d-432d-aa50-42c3325ba176

https://lh3.googleusercontent.com/notebooklm/AG60hOpWH2QWIiOixcOlUgH3Cb_sEBXAReZrwacYY4_tpF5ZNZrp4sDgtHlm4FEyQAwSHkgQaS3kFW6Jh10l82g5xj5UmUZEUxVPmhpm17mNTL4IyrZjJ2ruqFlm6nZ-9mHpULpoX23E9w=w88-h30-v0

db62a714-5bfb-44fe-ab92-d946f3728cca

https://lh3.googleusercontent.com/notebooklm/AG60hOoUSIokZTGOvkMMBgOBHMeFR_29w9u203p4IUx_2oNfS_UDN5uAYkZmBhgOHU44uqhCxwHGkeW1SkiNBhhJEYEoxTTCv7BNWc6leGOEagZ_9vcBaDvXsTrlNZW6UsdrnxRyvkRu=w128-h32-v0

ee0da98b-283c-4281-a6d4-c6b350130dfe

https://lh3.googleusercontent.com/notebooklm/AG60hOq8JfO7jrtVnkq88a-SuCSASIaOSWYgvcKb9ch_gXWG-MY8--hg5k4tMlCxExjZEiK_EcycjmIckiaX0Zk53Gp2b5PjhZxhtVmd0S6099KFKAnrcoJgU5KDtil2qISiLJIFnVsC=w125-h30-v0

3a4c9628-89e9-4f75-b654-dcd280f67771

https://lh3.googleusercontent.com/notebooklm/AG60hOoFJbfzhuc2j1tMNZ_lYNyCFlNmZsjx0evh7RA194U7NKK_RHjUQEZC0i_Mf_tyvc0oL-pyk7QIFcVrevPYq4Xj4VTDOfhoKMEWOI8h3HdUnPCuNpwfnDMeaVZd4NUer2sbdsW3mg=w38-h32-v0

0353e73d-ffc0-4284-ad37-931c952c20bd

https://lh3.googleusercontent.com/notebooklm/AG60hOoQ8Sk_Mhllb9IMfC0BJMpN3zcq4W_QnbLb0H4ADRrGD1UoX5pzYHgeiEuqtYRcm-zRIYBQkAQbueFD3ijAubk9ykVOb6TtHkPIQJgVHGTb8vfs4aROnBagE5uMeCYoIGQIeYul=w36-h30-v0

b39f65a8-d0c6-460d-ae00-9bdbe9dff9b1

https://lh3.googleusercontent.com/notebooklm/AG60hOpV6LhQwsq9MMERQpFjUewTiv-Z5CCldp7xh-qQrXHOM2aeNO1cra_v2KST8qK9Fk61iY5UYiTK9PssP1sfMEWdNS2dRCDFs7WqytPDnfyQV0WaJvPbu_6rTN8zAvvhC0mp1P_l=w65-h32-v0

b1758a01-5043-4135-81c3-1766d856b30a

https://lh3.googleusercontent.com/notebooklm/AG60hOpvKEZaQ9F7yVQX2mApgvblCw0571ntcW-PuJzoe4DS4wTas1PlHBRKxeEZYJULH0_RGkepsOvh6NasSXqX-NTAj47AiQuZ_SvTar1P0dQxB11eySeEVz7NP6t2R_VSnbSaP35Xxw=w63-h30-v0

a0dacefd-82b2-4d37-ad6c-2c51778ae113

https://lh3.googleusercontent.com/notebooklm/AG60hOqDfKzQBJSTCxHIfDw0ZOkdOzb_AyGI8sWk58Yc6CltVWgZS8qhPlG6q1Rqn1hydT65hl7E3dPojrp51lsnfOu4lR51e8xgYnPdhHxa4jT30mlLmstXTpd9VPMGKF0udCCbObhKiA=w87-h32-v0

a5fcf801-256e-4229-a355-bbbf91981c2f

https://lh3.googleusercontent.com/notebooklm/AG60hOqY8u-79xG-nwjmeRqeabqJNDwo8Tp_Lz1_c_CnDOi1YOSBQ5qVuqXJPSHjczSW6KXD71xhjFjdKHDsBF6JnubqtVSQk3rFg1hfO3-_2KLiFmtFy4iWHgpycQ_W_FymYddgQ-Ve4g=w85-h30-v0

0671279e-76ac-495d-a388-7afcbedf3a77

https://lh3.googleusercontent.com/notebooklm/AG60hOpIkmOr3OY6fV4SycMZKXtEoMpJzvq3U2ro1j0eE0t45bHTfoJjuWqqxY7ISKS8ECS7mjRSYyHTJ8ixmsbcOyyuKFC5qmGPSumStBRJj7axL1hChbftp1cWtz5klU_r9PCPqNMF7Q=w29-h39-v0

bffe2e96-8e01-436a-b1ee-1a08d7793a04

https://lh3.googleusercontent.com/notebooklm/AG60hOr3hojIlENl_ihBV1yDQpQqOqjG-hYGk_JGwILSFyrqqUMw7aTz7D0ahiVSUmvW7bdueJhdqTJvXBKj4UIoxf5OhsCp_8Jizo_K6JyFcNYMZx59VF7g2xoVXikEdpm41lMyFfKn8A=w27-h36-v0

091ea478-e1f2-49b8-ba2b-a41a6c5a9ef3

https://lh3.googleusercontent.com/notebooklm/AG60hOr1R1wfTb9mTYXfKhA1mhrQSmsXu96adLGZJ_dB3M65QaO03wRWLhCAofP9LeCW1FqBOyUmf6JbHJc-mPCOo3gBRjvroRyam_cDOEPPL3FIlgD28HDtY4aZsfjTnGtStE0mvL_Gog=w29-h32-v0

10554523-b8dd-49e5-9e23-62cd6d49baab

https://lh3.googleusercontent.com/notebooklm/AG60hOpq6eD0oDk-96-_qizfg-0O7Q8JwPlfREUpkZh6tTW8qaTiqMivKhLVEWXRgXK1M_TPlIiXH5LKebOq3tAzII-Zrfxb8Jt415X7JqVhOjyn6SYLPYy0w7bPEdmRj244SUeDa8hP=w27-h30-v0

d5b50297-8e11-4da4-83d5-e2583ad73672

https://lh3.googleusercontent.com/notebooklm/AG60hOr5qRblmlG_PtEohvBqxkDlOy0zXssXYHiKWSMgWe6hdmWTlBjwZc_nBsEX8mRr96_KVNwTSWzNsLmsB6XbRpez9ngJd1UKasyAo3-jEJeEA1XfNsw6iHMdXLlP4eG8_L0KDFLn4w=w67-h32-v0

c1422d28-7966-49ef-af56-5389d7c6ce65

https://lh3.googleusercontent.com/notebooklm/AG60hOpPs61BdD4FDpbr_uRN-wyj8kZeqX-dY38466x2IcA8ivB96mRvLyC58qEWvIohzQDqH7tfudQXPrmgHKIXXQwcRaN9Fws2u9xQQwxNdxCZxCZeQVgpIAzJyy9Kxf2qmFB5ezEF0g=w64-h30-v0

2946e811-3281-4daa-9bc4-d4c6eb78c3e9

https://lh3.googleusercontent.com/notebooklm/AG60hOprRrMyckfhyyujA3N2uBmnksmMPXbT6fwytDKprDTw5LeNaNPmYG7BLzfj-OwR3t81BPm_DCvhYiAZKvA7WBMmtMtzwm3JOzHlnalVhoBXZBcWMDsMjp673JZRl9SUskGnGTtYbg=w29-h36-v0

1ccc4f8f-af4b-4319-874b-e3efee9df39f

https://lh3.googleusercontent.com/notebooklm/AG60hOoApdspxhZ0T3aZQzC7h2thvjhjRXJN_7YVgtk1xfL0V13xsuneHsp8jgn_7hoo56N8zGiFy2jpa019tKr8LhdmL8-F-cTCsNNasc3WbMk8RvvJzFI8JIOJfD_6l7rkvyknXtR9=w26-h34-v0

19d85fa0-6e29-4cdf-b4b6-1d6759bd9447

https://lh3.googleusercontent.com/notebooklm/AG60hOrNTe44vkYu-IPkl9WJO9C1HNV4r3xY938xn7DfH8qmiV-P1MtyF3CwvqFwqDvaHC0NxsoIWwuxS380FPoE5tHIWI8GRPZuL4dPw8D2DZJ8BklhVjUruw3X12a47S7SSgnhPZgJ=w34-h39-v0

992bbc51-588a-4602-a349-df4827b35491

https://lh3.googleusercontent.com/notebooklm/AG60hOpOA8h-3DZe2IQpdwDMSYaH5W0IFY8DjuNOK5u1wGGQdoblaIaaEpUSYVU_GCgudHGdjA0XSSGMC6TNnDtVGQT_3zZvagLdGA5zYPZ3pVfajfRp41WkO9p3ZlpXAe5jPyocU5NiEw=w32-h36-v0

5df72bde-e749-4238-b8e4-07e35018344f

https://lh3.googleusercontent.com/notebooklm/AG60hOoVu1OTto--1DUbWLmKKRiW09A7iO7GL1nutcLXKDImNaySJ2dROWTh9-tlc_v-lV4Djd9UgS9fG9bPMZhzAMEeZEusg4y7cBfdM03crOuYxjzbtF67SuKvZ7e5vaOezBIFrW3NSA=w38-h32-v0

75936625-1ca8-4a92-8b00-aacfd5516e51

https://lh3.googleusercontent.com/notebooklm/AG60hOq0nV5HwxWzwGh76wz7TnMt_AysdQM4VmHwJ6fI2_X2IjUhk7dZMyqjx3f3QAE6GtPuf6O7QmkwEPb4zuOBYP1AR8qlUK0gwhOVjwaZXBHRYi31pGqPXnkHljSFNSvTtD1TxWUA3g=w36-h30-v0

a04a079a-567f-471a-a4a6-23b470908e17

https://lh3.googleusercontent.com/notebooklm/AG60hOqD0FjwtOyAtUOLMdoltxkWHZT_zNHjaRJ7GcLiCyVr4HBjtwf_ZUOKm_vWz44IwvifIKCo7tcMWky2BrtIBhhVnySFE-__mHDbFv2NlAqZglnTuFc3eGg1El0FscxrECbPHk0K=w92-h32-v0

c07ec1fd-4409-4f43-a74a-b9e7893ad2c4

https://lh3.googleusercontent.com/notebooklm/AG60hOpfS8Ginns1e3MxjqMR-W58zF23db0knBerRPppD73Fwt_rMnX4p9Uhah2wFKdaylpOABGdk3tj7sPUYU1hRFls7HOiyjyiNMuE4NmoAZZtIA7TK9ep3u1bysteEPhpD2uQjQ6BYA=w90-h30-v0

3a12ef24-247a-44b4-9f16-4e80336c4be7

https://lh3.googleusercontent.com/notebooklm/AG60hOpP-uYMLYIsJOh0KYsuKXIkh-zjDna6olEaIFG9ZC4tjg0Gr9yK_-bclAeytUKvPDcEDFGGyO5sq_7H9pmC-WHFuqKX3fwIOqYZhEZ8bj0IAzlnatN0q3_ebMVF6dSUcAvnhtnyWw=w57-h32-v0

04956ab0-1151-4753-82fa-bbad569c24e3

https://lh3.googleusercontent.com/notebooklm/AG60hOqqSDTTIE6JhZu2zcn2zGkcRyxAmnbbvZkPXAuipLkg7U9mMDWKCuQEdXvkcSkW4QWnaqqe2FIW-G_bGunniEbP72_SppT8yFGCCA0mpy2YuQfoUtn12vPlCCkOW4_V2P0-LmZHJw=w55-h30-v0

e4f7262d-5c37-4f9b-b3c9-09b0c30524c4

https://lh3.googleusercontent.com/notebooklm/AG60hOpzvTq0mjooxO4CkWGmA2rS7UAwj-uqaYHHfMfQEtgPukA9_8S7OP5lWPwOytwhU1zRgENSKEltYV1zbQkWltbV6GOYPwSTYrw8MjZ_M-hE9Y5DY4B6VL75k0vtQHKgOKFPiwpLYQ=w111-h32-v0

4d8ddfce-3d1a-4d05-a650-832aa4633b0f

https://lh3.googleusercontent.com/notebooklm/AG60hOpxctkvUXY_F0jmYS-G4PRj26NxE8eKTE_Mo9xaK8tH8jl69E74zWTuxJ-TUJ-oDofU9LaLLAHFZ5dYivx0lEfqBXAHeRlimALvebhuqsgqf3Kv-HrvBaU5KbdierNRFQ7AfJMjAg=w109-h30-v0

69aa5997-b55b-45db-b1c1-2cdb663a3a22

https://lh3.googleusercontent.com/notebooklm/AG60hOpcMJSnE5kh0xk-gNGabQesX33BxZDpQjLO8I6D2cSq0LjDELAxknbBAnf82qiWJbnmFAMj-B68q3RstV_ReYGacqQgnqn3j8LAWdKSz7A5gWXzbNVCNMnQYX-xdJ0LaaUZg-Ac=w64-h32-v0

5771b608-99f1-4596-9046-688965424375

https://lh3.googleusercontent.com/notebooklm/AG60hOo8DC17iivwcHqFgc04l84zdoqV4CZsRH7c1zoXvZO9eargtLq3oWAsHrFGmAl_DVkOqyPvIdGiUpi24c6f7TNGoUxHB6h9kFnHCgOw9EYL3qKBXgCj3FAy4MOXf-FVlkxAjxmq=w62-h30-v0

d259e3b7-1d3b-4f3b-98f0-27a7dcb17eca

https://lh3.googleusercontent.com/notebooklm/AG60hOqhrAyx40XCvQbhA2P9W48FY78R3rhaahr2Zi9kE96xuShgUYJA3uZ3IkD5v4b9os6vFaeEX0r71D-agSehvo4uesa_pLGDwBtHFcLb2mkiCCa7FAld35nrMwhSAc1qtRglDRwBkg=w31-h39-v0

770f799b-3f0d-4ade-854b-4cc51ac421c7

https://lh3.googleusercontent.com/notebooklm/AG60hOq2W1BEKEEMg674tJjnvD-k9Yn79VMYWMOc93T9VxMS9L8SvO9grJLyBaGo2uQT2psd_Io5lXlRa4VbOFpNE6xyUqc-ahsgp8yKASSVnwXR0w0KGeZDTWrRhKLlk90R4Fbt7suk=w29-h36-v0

d257882d-99a0-4fdd-a3cd-28719338a758

https://lh3.googleusercontent.com/notebooklm/AG60hOqzsZsb3jJPSRiovb6sVxP_og82jZfjxkowNLbdyW9xGHFUuIF5kqmftQofQ_huqWcXGAPj4aAaj6QekaG7kVtC3ypJMRDH8tHphgDWbPOl-INDj2ytHch37oz4OUw9J0WsxCBk=w92-h32-v0

d71b6784-2775-4499-9647-e9df04433c93

https://lh3.googleusercontent.com/notebooklm/AG60hOrNXqX0Fk7sI5IwZKBs7qNpZaa7zMk7gEdSeiQGoAZmflQXy5HneWL7MOOaj_HIvwAF9q0Qb8Qq9_9piJ03N8lBe8ku7HWEFH21lRzP_idLAXAG8nfjPTRo8o9-A32S_Arakayl=w90-h30-v0

e0e8b713-a6ca-48ac-a207-0494ec21f2bd

https://lh3.googleusercontent.com/notebooklm/AG60hOqKfr0qGsqexS3fB1mqYHJFoZQpU5j4YsrhKmX553EL1nIQxiJNKZgMv32lFPXhPFiFm26pqyk3refAMSUk_f5voy3tCoZT-jOspFjXMjHIHTXDS327Oi76DU-9vpu1aLT3WE-Xlw=w38-h32-v0

80cd33d1-6c5d-4f40-920d-f28f7c802b3d

https://lh3.googleusercontent.com/notebooklm/AG60hOoo93Fgu9MduhG3-hGUdYJXyDt2lt6wzu2MIt6so__aIAfxcPy8uBl-QHfXzaQwBEeXhROBJNk8XrMqkxoIq2yW0X3d4i9GOiXN3Faw5E1Ed9wF7PA6J6ojrWKGU6EONHxWHBnm=w36-h30-v0

c564efde-fe88-4e7a-8d1d-f28889ad6f9b

https://lh3.googleusercontent.com/notebooklm/AG60hOpPtr-vrcG5Y_8b4QCxn6FhIWUkGjVlJ5PRqYx7d5fhU65K1ghkxUsWXNhbxjBz2CkJYx6bcus6ZyxNp3eX8yhZKT2Z3Nr-BCme83kNNxOyaK-Y9Ed_zeg7RuJDiuwOisP53O8u=w65-h32-v0

248b3662-1013-4654-9e77-f48bd0115b82

https://lh3.googleusercontent.com/notebooklm/AG60hOrRqpfpWqQ37duWRmEAFvUE4YgZPsGORDXC_bnFcurL5mzNMrYgXqGHZoGuCYU08idH4unbMhrAOCZGPV6rqCuit5aGGYPQTUUU_TGcNGsl_4WwcXU_iFEW4n9I30PmTN8VoWdUbQ=w63-h30-v0

242d63fc-9d0e-44ae-9cc0-97ec27da667e

https://lh3.googleusercontent.com/notebooklm/AG60hOoKQcokjOmSfwT-hBZWYDBdWQjbNZQQcauACJpCyF4m2fxa37D5xIhvXjwQIn_p8-8b7kLb_FZqLzpxUvt4l0c1zPu3IzXRjQpDcF6dvEjcaw-YR-wN-RlKAfZ8FWbV2w-kVzYf=w96-h32-v0

d1528824-cb6d-4e89-a0f6-0f76c57234ea

https://lh3.googleusercontent.com/notebooklm/AG60hOrOeTyKBEllsX369yzhYRN3CbkesfqLyEBkRaqZPxg06sPypOE9rHp_cNMwICBLHk-s3QBFTJny2lTcMKcOzz8jH_i9afglO35F2UzDh4ylGHkU_0D1zRFC6yEnWfBzs3CbIYlY=w93-h30-v0

1fefa57e-ec49-477b-8334-947aeee693fa

https://lh3.googleusercontent.com/notebooklm/AG60hOrTPEdYfzt65DdYxb8sPiC01vCFWNUm9_WZA4hiodqkJS0Ls17fwBidq6zrSGzimYKq0SbhExHY06IWHMCW8Iisrm73N8OBqAnaFrJ6wlU960_1vJ_C3O2HhLxfODxCEVM_Y0dX=w32-h39-v0

9ba88dac-9613-442e-9bda-cde97826e2a9

https://lh3.googleusercontent.com/notebooklm/AG60hOq7bBja6UosJbF5DRIqCY86TGncjTEek0itGavC3wRmoXVeQzrANRlOOWJOayVleGex4ZHUyuHm_Ylm-zllQ_XR7OE1spe9g7UQ5r6AEfi9FepXvw5xaKqSjNyB2CJkJYBErUdb7Q=w30-h36-v0

3a5737d1-a071-4660-be4c-6ef3e0d1bdbe

https://lh3.googleusercontent.com/notebooklm/AG60hOqscmdI_Vzav1tfT9vQ737M3IoXiosY7Ll-3c8pnLu5dNiJXn3rx3z2Vwi2q65BIl4RzlY-8OhGGTwniSWipphAZck0XPRGJnZJsWirbDQcleV8v9Rx0tFs7jAZY5pqy7yaUkbn8A=w86-h32-v0

304881c4-02f9-4111-b3d2-57b806538931

https://lh3.googleusercontent.com/notebooklm/AG60hOqKQRC9as37AGzeEu0aP68QD06imln6Z7GUCGrYmILveQOjHRrYCHzGe8vPpsu_1OE-db3GWtxk0ZZ2yuEUGWPHNaf8kflFNyBLM7BeuB8DGa04ymcCh93oK1sx-bl9fQtZJyvEWg=w84-h30-v0

4e689f23-7aaf-4938-b93c-239b3368e179

https://lh3.googleusercontent.com/notebooklm/AG60hOq9AyK_jldUz8zY3FUa0NbXLUZth2oCeDuj3R-RaFOwWj55B83rkCAeo4sna2QjxH1kExsRDbw6crTbLAbTHVsjAEryxAPANHC-SCGGcQcJNvRQqKjoZPuG_-bC1fhG4kp7QFuc1w=w67-h32-v0

56a3c32e-5428-41da-ba5f-6dd988f22ef7

https://lh3.googleusercontent.com/notebooklm/AG60hOqeMb4c_Y4RdmglkSziP_KYsfmpYhaoyFYip-Rm8-yJzrgSBtEH8UNxFttcwQVFKtH6MPZ2woypazYZRgCCdfnLZ1ZhKh9sqJqSOZx_vLSihrkVaL6rD4CTWE9S-AOIy3sZP_sA=w64-h30-v0

8b1679cf-08cf-465c-b9ea-af75b28f262e

https://lh3.googleusercontent.com/notebooklm/AG60hOrZl9wAxpXIsmxu5CJzbaFO3IyoIBK6f7308z_smi_wk8vSHQxvtzpDhEwwcMw6oqXaFtK73M9e2MLYih6-WVjUioMBgke_sZ9sRZ__Df3lW92QA-oyU64p82uV95UQimuF-Q8zBg=w58-h32-v0

0ae57a45-1606-4f0c-9492-1a54b3c20112

https://lh3.googleusercontent.com/notebooklm/AG60hOor-qYgWGO_kvM6qJpfst2ME7seugOa5YvKMIlSRv7DmLcpTZ0P0-BltUbBv0P4EGaN3aw6K5M8fyx-Ewlva7wAP-fqQUmziyvfdpOTWYehVc4EffZgf0cxHiJg_s9u3LVjSMUw=w56-h30-v0

f1480e4f-19f7-4a3a-beb1-540508d0406b

https://lh3.googleusercontent.com/notebooklm/AG60hOrgDyu-503TuvmbY4Fp91LVfzLejMGGcKaXaUPR9pGAjxwANR_EuRiv7i6uc0jHgYCPxjj6UrPUmTSEJ5dGHscAabQ1Z2SXF1t_bPQw_x4EOOqKx3HDX-iV85KoGJvQ32ZJ_mgwVw=w43-h39-v0

0744100e-70c2-4d96-8d58-5c7afb071b07

https://lh3.googleusercontent.com/notebooklm/AG60hOqb4QEdY3vYK9RBQx5iB62VJeKupRUE4AJDtni0HWHKgto_4tN6RHerdawNVdfDTiMCMWrpI_BHuW-hXjhNJdCMWRV_tqBzmlUoi_-ZFuDtiCNGI4_bvgWksw9GuTvaUtOx8u3tuQ=w41-h36-v0

d75f869e-dc9b-421b-b77b-6aef7201591a

https://lh3.googleusercontent.com/notebooklm/AG60hOpcNavvgDb9PGY0o3FmRBdQwjw8h98tbGKbMUdbWn3Q31Y6sm-DwTcaBm32Tzh3gcrcH5nSIov23yVxw20y7_VLfOoYND4VK_Qy3nFMu79xHe0bQuIxdp4OB9K10-kXsgrTTXEJaQ=w38-h32-v0

529df0ec-0475-4d2e-b8db-02564465541a

https://lh3.googleusercontent.com/notebooklm/AG60hOqzZohpgHXazrMonDTKKLIs2Ayv5pxavMeES07JJMXw4k3FuW53CO83n6zk3Z_Pqp6nHPUNhMosCzDEmY6LZu2w-KQdYMUsZYGUz9nPElTVHk8cC24zXyRPSRLjnU0A5IyGmbBK_Q=w36-h30-v0

e90cf96d-5668-4c9d-a448-631d293d3a51

https://lh3.googleusercontent.com/notebooklm/AG60hOpzYj4CApsObdWmCodNEX7WrMw83OGRxK9qhsmrPp_XzcXqcUSM1HjctTlKPxreHpvJX7XwIi7WCbvk3YlDjVFA9BcZWfp_YorAQi5J2CbmPRrddjShz_1Z49C6FN9M3XeHe9ZBxw=w132-h32-v0

5446f44e-67f0-4975-8b7c-4d2a99821bf5

https://lh3.googleusercontent.com/notebooklm/AG60hOqkptdLSBgTj3HsfkobwkzhfjYAgY94Q_bpvfO4IzX__-nWo50TfAFOSUjSXoKjDdKv-4I3__YVtPUz277xDhI6xZTXHQ31A4dfVR4hXes54FkYfAGV6-W07sprEbf0qtLhKSC0BA=w130-h30-v0

e433b202-0d24-4622-b6b8-5bee5642c0ee

https://lh3.googleusercontent.com/notebooklm/AG60hOqGflDHjOx-5KoHTpYdC3pDLauowvwAApn8cNCf9oYUFR_Nj8p7uU7jcUuDahFGDOnAVTh0MlCuV0hoK306mCDshgpAXADSjnwmr7X1MMeF2Dcme02KWiPsC05LF5FBQLoW4o2xoA=w48-h32-v0

a3049aca-4283-4267-a380-0766f4f64041

https://lh3.googleusercontent.com/notebooklm/AG60hOoh082FLaSM0TCckxRYPz8Sg3LX-paa4NJoGxeqdzMVLuagzVdscfFoGZ2egZliski84Hhy8JVG5f8EvkBZD58ebpgmEbjKRybGsceerAtxy5PkVnM-1bDVkZ4bQL7fWtr3sgzTJg=w45-h30-v0

e538bfeb-e4dd-4bd5-9eac-e181aa019a27

https://lh3.googleusercontent.com/notebooklm/AG60hOqbmnX0SiwEtvZvNciWTi82ckJKuQSqV4OxMe5gznTLD6QLPulLgHD5sUVfzKEkUvww0T3u0_sYiaSahFXAtKMuMcXVIaI73l5XMgPRN-cD9s-DjrAoU799F7CrnhbQFFpyWNIOLg=w61-h32-v0

e45bd5b5-a3ed-42e5-8de6-dfa4b67f0d28

https://lh3.googleusercontent.com/notebooklm/AG60hOoDDJ-ek9Ll7O4Suymf9_r0fOj5yyomFxPmytJWB9Clon9zfainJhPD3PYbyxw09Aba77kthc2Aco_0XFrWszgbjIy4fbL-QokyDy3sKICWX5pwCNTrZVVaQIgC1k0MpjGzZgcH=w59-h30-v0

635a8a3f-aa1f-4206-b895-301c71bb9098

https://lh3.googleusercontent.com/notebooklm/AG60hOpWCqoMRX1aFiSrVhjZHOfAqHRcxKMUblZ7FIqN-XYsysxzfkqv1Q7qHCEt4s2lQvQg3niv4qzpKo3t6OMbYeRzvbaUiH09MAOIoqXlWqy85zPUUfcfAfnzDH38PgVLfjWakJ3g=w65-h32-v0

3b6561bf-0037-465f-9cd5-bd02a5765cd2

https://lh3.googleusercontent.com/notebooklm/AG60hOpBPGqLQn-y54_CVvtw_cs7xCbD3UDbJgao-viiaF19d18nfyZSOdys9ISKbGwK-Spuce4xE7dRwXLYGP5EajLsuwR5h5Qcnyx12AH0O1C8gYsJ1lizlAgMsyNYrFialuASwFsiwg=w63-h30-v0

995119d8-5395-4501-a560-21a6df491a70

https://lh3.googleusercontent.com/notebooklm/AG60hOoQ_GS-ySQNd99CVA9X4Do6KZUph8yMuNBeHFpwILsOAAd7aMOqcl_NpZXBRWjUHxjhqxF-wWHqZh557lDWhwvug2bGRTaTrIu4sfS8Q3YAe3d_yzmqnW8Dh_VVgDf3BVvXvhb2OQ=w70-h32-v0

82128ac5-a9c6-49a9-92b5-ab7f07bc86dd

https://lh3.googleusercontent.com/notebooklm/AG60hOrgKWmV0Am0dD_u1xX8GweOBFr30QFNWqQkt6RdNMdOBN8UF4NvUOfJyzuQkBxLPqBsNw9KcL0q1kSgugFFfh3xQ2ZMvzy1LuylC67hEogGmHG_LqDa6CBtwObAjifNHaKJ6FuLIQ=w68-h30-v0

5d4b4e3d-9753-4aad-924e-289fba73223a

https://lh3.googleusercontent.com/notebooklm/AG60hOq6Q-QNDodsaa-CsdMfNuJQ8PUvWXn2PLenRzdAlVH79bga9OMdLLipo1bQSaD7tnO71VT6PPJCztN8h6SIyvwOs8fLy4nYDb-phwaY9E51X24JomrOrulxSzeETwsGmlbLYabJog=w69-h32-v0

b24f4df7-25f1-4364-86df-2e01ca7e5826

https://lh3.googleusercontent.com/notebooklm/AG60hOq2Eg2nwxWRVIzEhnLr2rloe-IUbhuxbhWCkVKSehSR_U913122cMf3sUhaXbPUs-CAqgr7ODdGeqb5MUarYGh-x8nGGIwXdl_MWVPRUcEuDuv5P0uZdu7JBMrszPE1PCQSpgV9jQ=w67-h30-v0

7f53d68d-b3bc-4d04-a766-042ab747e43a

https://lh3.googleusercontent.com/notebooklm/AG60hOo4D92F7S3f6SR_wVTMt9oQvZbvGDHJefWtcSqw7Tvmer7ElQHaq9-zvYSucWjjzMdaKCd2KIIYZWKWc5Ui1cZk8KKAXvXAV3td49o7WRVvBilM1EQp1zymO55ggNE2-8qYK9CSwg=w97-h32-v0

cb7cf15f-2dd7-4b0f-80c0-f97f14940d7f

https://lh3.googleusercontent.com/notebooklm/AG60hOq23G0Spnd2rKqTSSwkEM8Mi9-IB4UtwLtqD27mnIMSNH85hPurar6hAHCTj0OGAaLJ789hBUNitS6vWFmyOY5TC86JYI_Fq2LZ82vFKSp8DniY0IIM3ZwSJmAO7AfBS2TTKN37Pg=w95-h30-v0

2c60ce3d-03c6-4df2-b0f5-b22f11832a93

https://lh3.googleusercontent.com/notebooklm/AG60hOrmoalQCJAh71iM1waami8GmxkX8w_3OGEOigI97x3mYl-lNcSjxQZQfuzGcRdU9SU442D6E6-qMSkIId_WceoinhLcGje4zLTP3oq0Mt4JUCvGJQ0yhFrN65r7JfDaayf_FdRxnQ=w489-h119-v0

e0e0b4d1-cd52-4205-b5a3-03df5a4062ee

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

LOOK OVER HERE: ATTENTION-DIRECTING COMPOSITION OF MANGA ELEMENTS

## Gaze Direction

THE TRAINING SET IS ANNOTATED (SUBJECTS/BALLONS) AND COLLECT READERS’ EYE GAZE

A COMPOSITION IS CREATED INTERACTIVELY, WHERE THE LEARNED PROBABILISTIC

MODEL IS USED TO GENERATE A GALLERY OF COMPOSITION SUGGESTIONS, IN RESPONSE TO USER-PROVIDED HIGH-LEVEL SPECIFICATION

THE USER GIVES THE NUMBER OF PANELS THE SHOT TYPE AND MOTION STATE Images from Cao, Y., Lau, R. W., & Chan, A. B. (2014). Look over here: Attention-directing composition of manga elements. ACM Transactions on Graphics (TOG), 33(4), 1-11.

Shot type (red text), motion state (blue text), red rectangles ROIs

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## DIRECTING USER ATTENTION VIA VISUAL FLOW ON WEB DESIGNS

## Gaze Direction

## USER ATTENTION MODELS PRODUCE OPTIMIZED WEB DESIGNS

THE WEB DESIGN IS OPTIMIZED AUTOMATICALLY TO MATCH DESIGNER’S INTENT

THE ATTENTION TERM ENCOURAGES USERS’ ATTENTION TO MATCH DESIGNERS’ INTENDED VISUAL FLOW. THE REGULARIZATION AND PRIOR TERMS IMPOSE DESIGN

## PRINCIPLES

Images from Pang, X., Cao, Y., Lau, R. W., & Chan, A. B. (2016). Directing user attention via visual flow on web

designs. ACM Transactions on Graphics (TOG), 35(6), 1-11.

To increase the prob. of eyes transiting from1 to 6, 3 and 4 are made smaller and text away from 1

https://lh3.googleusercontent.com/notebooklm/AG60hOp_Ky4f_ivgtl6J7HfT6HJSmoyMhsqIrCvDkldINFnt-5ZMZONpGOpQ-WJYt2Fpj2bWDhAzBWQgDLYPZuuaahCuzeBFAVk4E_gEbwhmJmSXZDpn8RlgFgvRTZVsVcYnqSsKfL_8zQ=w151-h36-v0

068624b0-fb67-4154-bd16-fb5834490e22

https://lh3.googleusercontent.com/notebooklm/AG60hOp2s0QXGRGyfOzbpORGXtUaBkWXKzEFflbxU0AfZ6WW1m6V_M1cQ-GCQ--UDdyVpLc7qiVxiT4WfRpXqS1H-QNz21D-5SeZHJJCipV8GA8m-_hgZuCnJIlI5SXqJBWReYeQ6kEqnQ=w300-h359-v0

5c5ab23e-4132-4e3a-b327-379581dd220c

https://lh3.googleusercontent.com/notebooklm/AG60hOrvRbfRrOe9l3Xl5B0Hrefjb1LNLxLvEomxSwl8prLBnNnE653jFDA_3hv_RLuucY_f6OSQwpbMMK5eEWtxCCA3KJgidAbjQMfryNHFTEa7Ov51ys16h6sTl-UTcyGegA19mMK8Rw=w116-h28-v0

6bcb5316-ec02-4da8-8c66-cb3796a0dca8

https://lh3.googleusercontent.com/notebooklm/AG60hOpBwumapKIw-57A88DuUnffZCpzD9b7TQ7XhkT3ESJtZ33ibktHyewtyMyc2gPxUl4U-4EteDbqMcygyJHG68YF6zpKegAIyonzleOAVm3seQNQv1N2DM76VxDiYzShD5lZ2Pib8A=w495-h102-v0

9f702c09-88f7-4d6d-852f-ab59ee206b44

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## STEREO GRADING

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE PREDICTION USING MACHINE LEARNING FOR GAMES

Manipulating stereo content for comfortable viewing, a process called stereo grading Player actions are highly correlated with the present state of a game (game variables)

### 1. Real-time gaze prediction based on Decision Forests without manual object tagging

### 2. Dynamic, comfortable stereo grading without cardboarding effects

### 3. Account for task, learn from gaze data

## Dynamic Stereo Grading

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2016, March). Gaze prediction using

machine learning for dynamic stereo manipulation in games. In 2016 IEEE Virtual Reality (VR) (pp. 113-120). IEEE.

https://lh3.googleusercontent.com/notebooklm/AG60hOr4jMmAyhOe-19k8MfYqJ_6XF9-az0ctXfdg04DQxPvZZ6L8It5jFIUI4X22BkzAAtTDeVEAn5kt0S5BEWFp8BA_noStgCeO-k_mbNNnCq1B7TNORRLzZkE_-dbQXjzkDOx2qMH=w116-h28-v0

5e8be0ee-6f8e-4802-94c9-2b189ab5668a

https://lh3.googleusercontent.com/notebooklm/AG60hOoR7Qv4hDie6fXWu0wG2KL0SyIhSk6ecEvsAUgttUhHX9Bh2N08BGUGmJfWVJ_mo7G7nzao7Y5g20gZ6-OcDQ-ultky13mRhcxSGYtm2EnqWflirYHMZqYLhtqEcJ3u7znNOk8j=w22-h26-v0

ab7c610f-d265-416a-9dc7-3aba676a1d5a

https://lh3.googleusercontent.com/notebooklm/AG60hOo2jDgxymPBSAhZtZwAxXEbLeUDiGicZxlmhiBaj2tt00THijx_a2pypivqUJ2WWw6WSSSM4RLFk1XDmctN4xrhd5lshyRxYKRhHinsUlGLBSsX75RNIQ4G_BJI8Yd2EQufO-YGzw=w21-h25-v0

a4750a9a-10f7-46dd-8bf7-6f55ea8261c0

https://lh3.googleusercontent.com/notebooklm/AG60hOpJvlNicEdijNPgnDKC4EugB2WizL_pqkOuruB_AyDlELzBw3cLSFCXPR5oqIMtOyrtqljyTOv4B2jUbJh1KiWRjkNhpu-kLlWL6JQqv3b3JsE1dKELeo_IR31u-McTfsqPthfu=w20-h26-v0

906e5343-ef98-4967-9e08-28c7e911030f

https://lh3.googleusercontent.com/notebooklm/AG60hOr7bQ-OYAzl7BEh-p8MZZLwduv4f3SzwKhLppJnzum3pXSK8og76hjDeXzl06OBfaCNtzbZhu52DmuGJhiaW27DcYRiuldnuTvPpq1Hp8UsPUO76BFBb2ALAzl847ultA19Jr_Nmw=w19-h25-v0

e9037e1e-6878-4128-9dae-d585b498515e

https://lh3.googleusercontent.com/notebooklm/AG60hOoXRtSUy5bVRlVrt1DoInenVU2SP4ej8Yf6B4zsA_7oW8B-Tv1MPDSi7PXA1Llp6JBBTEfFiBOmdEuflRUQrKde3n3XTKzeyLplGcbw6PhbBef1gOsX8v-_84jRFocfw_ZmYd18CA=w115-h26-v0

d0f1c2f4-cd15-4600-ac3f-89e5280234b2

https://lh3.googleusercontent.com/notebooklm/AG60hOq9-4RhVTf6PWtoyCgE7Tv8g9ot7uc31V3wxPDy795uRY0EpjRzyKAxxpR-KfP7TuB_vBjrhYiAObHItZJ4jzhEKS2tFDfpjO148o3l7rG7UjfIv3Wr0mQez2YQQ_KakWKNhM5M-Q=w114-h25-v0

2cca0305-a38b-48cf-af17-d8c4eb40c504

https://lh3.googleusercontent.com/notebooklm/AG60hOqYo653AgUX4L5PGBOb3guhOvNOGFdIAQHRWKBHzGf_GevtEeUAKgYSycdZ-Rdq2s6leXGKJTy8szyi9O4I0GdslJLwG2HsFhefyIDOw_s68FoWVbmQGEL1XoLFGn0WYpnU-6HiHw=w213-h237-v0

e6ee5bb7-e9b8-470b-a50b-6aab856bc175

https://lh3.googleusercontent.com/notebooklm/AG60hOo_JuV2huzEiBgpWSVB7h8i8qvFBhUVgq9jOJ9AdnYf-JAj4VnMznBXfTqO2j2kQOlqTLigASXYDGk_WOfe-i2VQdrKarBzUkQr7Nj21Nco0PZ7b4sBKYTj3Xd3oEM0YD8jQOnzpQ=w152-h114-v0

7bfb7755-ac1e-4b81-afd6-f0239108c76c

https://lh3.googleusercontent.com/notebooklm/AG60hOqXxDSuFP45nhIXKNrHQ9mv4zm2U-0bju9Hrbf9jiwlEylGV-7azCCHpw5_lh_O7RY3h-KjEOy1K-iLyfvEbHnt8H5uk6hpWhlQFEvfTTqOOOtdwEtWlSlNezcH-0yZm5wUICnFRA=w116-h28-v0

83b7f3b5-5163-4ae5-8c19-bc3f3e6fc731

https://lh3.googleusercontent.com/notebooklm/AG60hOqXInII8d_1HLb47a8F2I3h9JraHFIT4X0AQvBDx2bn-qeHbf0SN7QsmVMqLJueVoSBLCJskd23FPZ5VvORPkm-hbI-fyqrwuWnV0YW8U2KSg7i0YA6Q0SuExx73jHgZWMsR4OM2w=w74-h26-v0

9ca0e676-3886-4b21-bf47-68d4c917c2f1

https://lh3.googleusercontent.com/notebooklm/AG60hOpjRK9DO1QsiGNZ5c-ECRq8xDwaPPF9XajPjH76U92i_6EqbAyxIAY3L41R1Yc44jeGsRz1JHEwoPvKsLXD5hDUP2Pu7K-vcLYc15-VA9xkRHJGhvUlVxmkIjyi0bswqaCFGsJp2A=w73-h25-v0

1b0c7266-4613-44b4-bdee-a387074f3fcf

https://lh3.googleusercontent.com/notebooklm/AG60hOpScq9Lsc0yvOBYjVLSAu4xDWYw-YwPdU_a2mu6-ok1gZCofIRwhJga_J6JU8qkbFWjTct8qsb8G6mzmgrgr6rKY4K80NnApAxJ77emJOxkmTqChqA7S-AwidjpTMJC23v-4GgCiw=w68-h17-v0

13beba74-3442-47e3-9261-299e3e18d184

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

ATTENTION MODEL 3-STEP PROCEDURE

## Dynamic Stereo Grading

## Identify important game variables and object classes

## Data Collection

## Classifier Training

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2016, March). Gaze prediction using

machine learning for dynamic stereo manipulation in games. In 2016 IEEE Virtual Reality (VR) (pp. 113-120). IEEE.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

GAZE-BASED DYNAMIC STEREO GRADING EXAMPLES

## Dynamic Stereo Grading

Images from Koulieris, G. A., Drettakis, G., Cunningham, D., & Mania, K. (2016, March). Gaze prediction using

machine learning for dynamic stereo manipulation in games. In 2016 IEEE Virtual Reality) (pp. 113-120). IEEE.

https://lh3.googleusercontent.com/notebooklm/AG60hOqvUB0cCP8n81s9iAXWBrEHPLn9PWN4mE41-WL0VPgcqTexCgYgplteyoxc0kVTOoSKYJiPjAYnhFIr4sRJRAMSljt7SQHjgT2Z0oMUzeQgtSQLk1H4I7JqFuAuURwONYt-EINX=w116-h28-v0

a320f534-617e-4b15-a9b0-1252e54a538c

https://lh3.googleusercontent.com/notebooklm/AG60hOq1Imhs-NsYYxwRv5lsaNz5q0ugCuzQ7i5Dd5TBpKIUAd1kw7yD9zXeQlA5ff-a48cle1Xq_OdYrPd0-SH9k_m9OQ-GtDpCEgUL8CVmvXhohgJXODaF5mxxhcTQ--fvcpR57NN2iw=w24-h30-v0

176cd11e-0c90-44bc-9003-a1b859d221b2

https://lh3.googleusercontent.com/notebooklm/AG60hOrto4w4qcsJaxCIQxrP3EoW2t6jrfQvuQQosvjOyemkzR6CGKEH3xNZ9-ZFFUyQkW8vhNa-p9hjnAzKHe4F5cJyGx37I5seyFbE1MouQ1IxgqO-S2f0gvrJDIKnHrJNP1jBKqP81A=w22-h28-v0

97fe0f25-39fe-45e8-aee9-b6f842604bcc

https://lh3.googleusercontent.com/notebooklm/AG60hOrqTi75E2Q12ZDEynzhztcpaCnUZdjdMBqccyk15i_538s7YThVAdrZpnpmp_jVRVm-zh-PdtDqUl2yc9y-pm5NZjOuWJM6ixNpnwqG2IWlLHmbPozkGJkZZTTmPqjnFL9lCJz3Ow=w24-h27-v0

0d2d1588-c80a-48a3-925d-08e48c747117

https://lh3.googleusercontent.com/notebooklm/AG60hOqaY_YIzABUGQYkmsTQJOZNEPNyeKGSVmSIu0HOp2J1Jn7hymRwMobbdiZaZBIhV1MNMB1hwrWLaC1cEFiTXaj2zhvrCaIoQH2KQfIFAxNOibJVJSc0bHHZ1go3n6rxd3qpucxu=w22-h25-v0

76be0539-1b19-4047-9481-e42c96837cdd

https://lh3.googleusercontent.com/notebooklm/AG60hOqjc97DmC-3TsPYkIy14NLD8XlFbV5kWcIZmqxDJTgs6QuyILRJl-FnfSmbHu9bct6n9N5bObdhNNbOcdNeoRoLA2U-d3G6shvdOacIOrQuHF8zIjy6Y8j-4ozKdcm1bwrn3dqk=w46-h27-v0

76f13dd5-315d-4453-a116-29350adb512c

https://lh3.googleusercontent.com/notebooklm/AG60hOpLXJZVblRHE0m8b1jc0ihZy-k59m-ALr4-BrZqJeAmj86gNRvYsCunlyM14Hh1qprarKV9qkWeFU4EuCs9gXyFBO_FNFIyzKswspPJt6bQgCrTM1G679R3qGGtUaO3s6vW-0pVRw=w44-h25-v0

c0963752-908b-407d-9441-99ecf6597f5a

https://lh3.googleusercontent.com/notebooklm/AG60hOqJQaDpIPp_htyuYCIyccBSWztdM4Q2R78uT_Y7H_TrqFEUpVtCt_FIxMdLSfv3h-Tw_QM6vTd944-xEn43iZEGeqX1nUTbwSo1AdxvozxiQhVaUUW76C00P_ikp5otVk86nS-w=w25-h32-v0

cfe4e705-14bf-4cfb-80df-99f6c34c6ad4

https://lh3.googleusercontent.com/notebooklm/AG60hOoNs0YcYks-HzFsa8wBAQUqNT4RLenRe4tY0yqUtlsj3eIp-tyLIRTwi5kFus-SQeDEsyopQAamuvyvbXdnCK7fazjC8LW-O0Jm45cSpdpIYFq3yQhlTeR1lzRPnyZxKDB-XJky8w=w23-h30-v0

6cf5c884-cbe6-4984-9fce-6be5b6aa2454

https://lh3.googleusercontent.com/notebooklm/AG60hOqh3FseSeJgmtl3Wgmn3YmkGlDkVRD1Ki-NNHMA58vzxIXPn9am1NjIxPu-LSXqa629rnr09PL0xEIra1Qu1h0-fMeoE8WT63g_A0BA6Okrxrq_rCyH1FVndaRetjFRggsFrngdhA=w43-h27-v0

94ef17a7-828b-42f7-980e-9619bcfb12d3

https://lh3.googleusercontent.com/notebooklm/AG60hOrueEz_7_4ekr9jT-fj7kCHGebSzpWiXlOO3rPUjHt1C0YBiban9VGK1calznH4QzqelthWPzHh00JKuLWzAc9LgLzBnsdv1-LxQm5g1RgeCXrmK7c40IbGTBNfZRbA6bf1i0M1rw=w41-h25-v0

56f11bea-a1c6-47c9-a52a-81d2bff4af57

https://lh3.googleusercontent.com/notebooklm/AG60hOpDJuXiSNuUcYaO1Nt1hjIJ1mNI1GB18vUUajHck5luRT7jyro0G42BUJ33Uslg49U9nvi1wpe836dYg1s8obmRC01d4i1EhnbkngkB7tlGI5PSJrDW3Fx4u62OwqyhsTwffU5qEg=w91-h27-v0

28640f3a-3db2-4763-bb52-9285712dc3a6

https://lh3.googleusercontent.com/notebooklm/AG60hOp6Y8pU4eSYH9OXP3d9WdCdEm_4ofdAAEJBvab76mw_X6DYioND1-Jjf3Fcj_FdtKAMA7Qe1jpevmZvFtgTczNAHYa9Ofw5JWZweH1sx-8rNPGV4GteLj6MJJXGSpHtL1kBeOUFLw=w89-h25-v0

e4b4ab27-634b-4e7b-9148-e76a24657e81

https://lh3.googleusercontent.com/notebooklm/AG60hOp3l-BHpz3UqS8-DeViKngrx5ogwk79QoEdkIVnloG28RGuMLsGdy7-UPzhrl7eF7ncSBxPMDiuRzh5EU_vf1oNLEN5a_LGbAcdg9DbtpoW0fBGgeLmDlF-xvtVQ9Hc2kOWUNEk=w44-h27-v0

cb5ee751-0644-42ee-9c60-588e97efe768

https://lh3.googleusercontent.com/notebooklm/AG60hOpUSIs2iufKHVmlFgb5HtDRWR0c3-BshgnHTmSMrSSsW7A3jAReuU6bUOm47FTF0UK4_-SynON7NTzSGfK24X-yllVur-qFzxSB0ZfLmOTFqi13eZvK8XXsor74z-ID0st_NRR9xA=w42-h25-v0

219a630d-53bf-49c2-b6f1-8ee4890f0b41

https://lh3.googleusercontent.com/notebooklm/AG60hOpYOOJUCA6hdOb2IO-DsHSGNYNzQ8-3_C6F3SlqR0c-65MNt3-0ir93RMe4hs4zdKEAfHNsfqBROU-wq48OvhMZx3VqxSo8tQGFYwUHBQfPNmutOf4l4eZYVCRbWzInYFOvHdkZ=w63-h27-v0

5c77bc10-2c80-48a7-80e0-09d8bbc38416

https://lh3.googleusercontent.com/notebooklm/AG60hOqSdHL6x_2WUtPuRbbX2a1BhJgpTGxMd9SK8EfI5SVXY0OZbBpxPRfPQf8vyKTGqlUnV5OqcmGUkkIzItLtojjLSO-knNufr_OIta2QbwJ1UC_UAsvhl7-kgyMVWUxfmYMyGz8doQ=w61-h25-v0

fc0bfafc-240a-4d45-b39a-cef39d964d2e

https://lh3.googleusercontent.com/notebooklm/AG60hOrIzvL131lDqIihm9GWR0S1N5qdezSH643GrTv7kVFkO1ftNnUIdUY1I8z_A1ZghToBfnuwFjsmz6YqssNYTEFg0f4hXSAzhamwult_v0L6fYaUDaJS4pQKjr17bz9OP8j6nz8xPQ=w53-h27-v0

ddbe718f-ee04-4671-a992-862ae9ecf828

https://lh3.googleusercontent.com/notebooklm/AG60hOoxUN_E_e6E4c6bmVlyf0HhxzgdryPkpWv0rpNFPIOVha2uS4MWUkJbXcfRWNLLgBlLK6qu-I4KDCORzgeBa5s7Snf80o0UOX__wJysVSKZOac50hrGCp8Lh2TMUKmCDvNoTRSm=w51-h25-v0

544399ce-1aeb-421c-af2a-ce4bac84d861

https://lh3.googleusercontent.com/notebooklm/AG60hOqzwt1_4D79Pc7zde0VL3DwqLaCKHbQnETWVSdhBmiSZVoZQQSYj_U6GjXVWKCeDeowge44kbVgjTgrBJIFRRkMLEt-LfK-Ng_YRti0p3r45Y6vK6DEFGQ1rrNJazf6wVZ1tsWhAA=w105-h27-v0

ec14d443-a58f-4f44-ad00-1b58f801b172

https://lh3.googleusercontent.com/notebooklm/AG60hOrwLL0ayrvqUshLLXGF7qaCruqmbvOZUfVk5undZe6mwtkt1FI5cu7aUQ7PFdliTkASJDFrGQMxiatvO986VPnRZ6l3o7XC20rhjwgkKCsbpBzgjDlWIwtZFnhdcJfbp9lp01aGfQ=w103-h25-v0

33982f97-c468-485e-910d-84a80c145793

https://lh3.googleusercontent.com/notebooklm/AG60hOo9DWqZsHfUdhcU0Dx3wZbCOn5O0BkY3G1ZvxWI3ZAlQUOEW1h6pH3FI4uoDqg7TZ9H8XhkQOqzoTB9X2-h4GMbAStUHhGveJq_2qp5VlSa7KaQkl16uwReYJJqO-q_HIiQVptv=w32-h27-v0

a74cd471-e8ed-4ed0-8031-d8e78b57122a

https://lh3.googleusercontent.com/notebooklm/AG60hOoRZBb_AswraTTbAyRCFNyInKsq1AC_H7cfLOgvENR_pbIWNIybbB27s2IZngeMBq7q4M9rTopHgQb-uvqvv687QwzsO-m4ilZhN0dq12wlDX434v2IUBqt7PvtGrDX3wNZRoLN=w30-h25-v0

3b02e136-3a54-4f31-a97a-6bd4e8ef59e9

https://lh3.googleusercontent.com/notebooklm/AG60hOpXDotI6KZa6QOldh5lGUL34EzdjaLyUs7GPMt20Ic7_-5eP1zp9u5UFNyFNjHEvgSILhkURpZC0MVbLyKFKebSefR9b-VRFRG2DjM5gyUwN10UDWL0dEDRLTNUbnQFnwPsZckX=w102-h27-v0

6e0f7a6a-a1c1-44c7-8682-cf3679b430d7

https://lh3.googleusercontent.com/notebooklm/AG60hOpVHRW3_taszcaV4TszL-8hgBUAzLDxBg6vyjApzujuUJRewABJax6iVqb4K0SFLeLgfc6C3-chko7U1vhi4xS9ra2htNYYX1XhoS_ioBKjsJpAI_VWXmJYmrFV3n1bu_YtSEzPMA=w100-h25-v0

22658237-c030-4617-a8e7-3c4bfd087393

https://lh3.googleusercontent.com/notebooklm/AG60hOoFa4kxhBilzZQWXSkTZjk6TBOFDsFaCDPlVzcIxYQ6JBPtOYNQeFcZww2Ts198poTEdeKoX2LVmqLucYEbspI3u0-S3DEwFrqOrNw1fRP2fTzlQW0MaU4agJe6zFn9K87SC9c1EA=w69-h27-v0

fa27ad31-4283-4235-8c0e-89e0f8b2f753

https://lh3.googleusercontent.com/notebooklm/AG60hOqoQTLce_n9JPyi09mcgC1I0xGNK50V3K9Bmi_zPcGBndhKs28CpiceIXwYllzVshqWCM3pF-v_2Gg0ZaFZqSI4siTzKkMRi7Nx6y5yPVGzlydrdhz5Xdx42e0ei_kG5xFLQhQb=w66-h25-v0

a1a5f450-14a6-447c-8508-639df6091898

https://lh3.googleusercontent.com/notebooklm/AG60hOoOrEuMHuGTKXjY5_TBILx-VaRt4jjyQnQrMID5EdC-yS6CTOwgvJ1cIW2ObMXRpRAf9276TsRgRa5bNht9BxZcZVI9Du7GBHmSN05H1AemQspHm1NeO2-mX4BJ8tIIc34RP7Nl=w24-h30-v0

d1140733-1561-4ae9-b252-fcf35970b468

https://lh3.googleusercontent.com/notebooklm/AG60hOqZxw-fXCXNEftZqeAfs5ykoloYc3FRnCCW4KxmtNHBLFP1pEuiSW45ZQ8M96dRhpWngR3zkxP_saLkL7OPBC_y4tpoBqLDf_RNNyjXEd7Klpaw7Nbw9CfGUXaQm-MKBUVxEKaizA=w22-h28-v0

fb3b2a55-f7c7-4f10-945e-92461a66dcc3

https://lh3.googleusercontent.com/notebooklm/AG60hOrvdQ25B8rLEmhKiP-b6JJv5ZrE26M76cRkUkIo_V9DaWwOv6faZWUm1t4pnZmYC8e69f0IJtbq6YRrdN0QL1DQ6PuG6cwqUuDyB5eXkMJOmQUVQbMLLYC8_T_WV2FOVb7q7M-kDA=w104-h27-v0

9797f19e-d574-4192-83ff-4c92c104ad4a

https://lh3.googleusercontent.com/notebooklm/AG60hOpkPeX4plSg_m9WTDDfGp-0Qoa5ZwKgnwZO9NXJIDNCDtzY8VSyBchAWfWENHPVG4IsFegN3g6wN-6H-jHqJUhGHg7Och6SPDQF42UIktt8CKZYvaTly51ALjKHsg3qGQX64RKOmg=w102-h25-v0

c50d41b6-ffcf-4779-b4e7-4d52f552a1fe

https://lh3.googleusercontent.com/notebooklm/AG60hOp7P9izk35oxofd1reIU1RBEk_geLVY1_bLT-4rT1nne0kd6eWm-NLJogxiShSd634tmoO8do0yAbw3v5yRflHCF0MJVK0JJHMshgRFVsRHFAXSCufuTigHdzaM8sGqGN3S0UdKyQ=w28-h27-v0

ed9d61cd-5607-4f4a-ae3d-f5f33eee9d02

https://lh3.googleusercontent.com/notebooklm/AG60hOrjNXTDp8_OTSjYDmDH8ZCqk-Xcl6rQS5fZahYqm4tRkE_i_yt0X4WxcMIqxlfSM22kPSOxGU3fH1Tfd-nhgGAnbdgvoieHX9Vkw8DAoLHsk5nP8QfIUb_UkxvYV9cOOI6K_4-cug=w26-h25-v0

93c9e5d6-d096-4a00-85e2-5658963e1d96

https://lh3.googleusercontent.com/notebooklm/AG60hOqxWQVy18nA86AxOF1AFpwEQdOPFUVuGcIntFaB2dDRd9SFytQm1nfiHxEWlPkPAE_3WWHbEefw50su34RRTg1iipn4fAK_in6NSIbWzBFRtguZMIhQZMZR7Mwn1OKX_MuqlmCL=w53-h27-v0

cf92aafd-543a-41c4-8f99-d1313a67a576

https://lh3.googleusercontent.com/notebooklm/AG60hOrGzjPOourWEUhIshpQgOlIa-ZhZzHYXQX1c5tWxvQ0oCKZ1VG_WW2IPxfgtJBRDevnXPug3UJH84jYhkzSntNLRw6hotr5bWnzCsLY6rQr0OQAPpNG6EK0IyFlVHOptLdosDJ8iA=w51-h25-v0

35b770d6-cf55-4caf-ab31-f9b7265a10fb

https://lh3.googleusercontent.com/notebooklm/AG60hOrOA-sKjBPXv7FYnxU7l4oj6H7E5vj2zpfaQjs1Rh-hO-jB_kVuqYx1KD_CvpvOlVPCLu3lr2e1D2FbG8ylAyxC5uoverX3NMJvs2s1yjoybUpht5-At6MBK5qLUncJ6tYxEv8M=w87-h27-v0

da5491ef-2e04-4aa6-83fe-72d69b3e1c8c

https://lh3.googleusercontent.com/notebooklm/AG60hOpB3JEBcYkJZJtFDeIH7cUoA05TyvM44igP_gb92gh25nojJ3MAkUf7bJ2tbo7uwxxfDGDuQ0qcpMsA258WQiHqyxJ8EUcs9q9kTTlemUgjuxBUB6mvOq5wJCdGaizBIvzkpGaDvA=w84-h25-v0

3416bbeb-7b4f-41e4-895b-8426f5ea7fb1

https://lh3.googleusercontent.com/notebooklm/AG60hOpuK00SkBhw-V7J_8W7zM2uJgn1gMe2McFjgJZhRzx5o3-cySDXE3NmKuBWkJPKN8E1E59cjamzY8ZtfUd5YHgbBYoc0LcejjDUdFIOwOFgqUdLJ-IDBqgayWN2pCulPHOdx1ebpQ=w67-h27-v0

0832ce96-cf2c-4d61-8c8e-a537dbb05394

https://lh3.googleusercontent.com/notebooklm/AG60hOqMb_Mfd8QEMnXq11yzCMKnD6BnJg60sVtTCclOEE0Xpm1kfK8M_Qt1ow3qOO-WEOqtxwWSjF7Hk6Cu8qGCcqjOhxe83_s0cRxrNVrPn4Ud4fj5x3uW6gAtBEKd_bIiiaz7Ij1lnw=w65-h25-v0

d28bdb00-e2bf-4a77-91e9-821707995b37

https://lh3.googleusercontent.com/notebooklm/AG60hOpC3_LHsH7BLufR3j4g0iZT_vC7fNZGLGqM7IdMVai0_gxeFfk1aEqfneya-mgBDwHw4VVuWjTRTcrIwW-hdGWXX4GVLouuOXWpOGyFq1XMZliBmqOh8dsJMtjrzTF4on-lyx5slw=w86-h27-v0

f31c753c-cbc9-4493-94b0-39ecf00089fa

https://lh3.googleusercontent.com/notebooklm/AG60hOoyekLzW1gHjVJ42bXHdIDg5VmW50V6vX9SiVdMRAhQ-gjubZgh16_bgXy_V9Rd5Yw0K1HWIZpvvdbdFM3zaVs0nijkBw-m0txbE7WhurBXzy620Nv5qZoRYBMIkl_GXvvfN89u=w84-h25-v0

3e4df67a-5c44-499f-ad59-47af45355291

https://lh3.googleusercontent.com/notebooklm/AG60hOptYTQkiff947DmfWOEa6iUX4JU4s6vwhBpdxEoVz8cx2Qu3bZTFSnbykyJ-L0LQiLULqTGHrU3wxNwWWymMOOHMNr4FFDuPo0Qd2-G772jgVQSuUyBR5gfZFzZIXQeh9JoUkFM=w56-h27-v0

fff7eab7-ec81-4708-8883-d60df22c128a

https://lh3.googleusercontent.com/notebooklm/AG60hOoxQF4ns-AvygN6mlgMFjxiANXp29CBZr0gEtJhlvzebS66jfxWd4JVSCvq_Hawm_vjqAxwLZACKaqe-MrX_z8zjuyMBX7lpAPafD7ifXUQOmGP-yg-6y__qtMxvGb1uosSQQB7eg=w54-h25-v0

1e355184-aa3d-4636-a933-f8c5d6cfc8b4

https://lh3.googleusercontent.com/notebooklm/AG60hOrEHutyXHca54PFNrQwGuSeM9zD9aVH1wjq-loN5QLKZ6NucHKmvOBdUFd4dY6wxL-i8f6Jh2teedzp9CuIBv8XKmyplpJ-4ATGaaS_YhiRrYuQOOTZM0iSkxFAHzazD9yT3_64gw=w64-h27-v0

ed4bc6ea-18da-4e47-8df4-6de74152cfaf

https://lh3.googleusercontent.com/notebooklm/AG60hOoKCxGrHFP4scTVDsteXl3eAZqMjbfsILtT4aOzdd2x1VldJkKDRgbpSe6Vd_tzyjQ1jSUOj8gZfQ_1xCfpCYmimTwGx_MxIgn7mfiHjbavrlfxqFf5n9477IpwjTkpHDHHjXPBNA=w61-h25-v0

5d127465-cce6-45cc-8f34-d5e5c3e3a581

https://lh3.googleusercontent.com/notebooklm/AG60hOrG9b9DSr67Uof8AaPrLd5wnUz263M6TuKPKxE7zQ9jkY7DDEjnMg89c1PRaUoXzKVx3cyTEWGBSQurdq7qAH4yiveGvGCuB9uukQmIh0smeRd9X0SVM0iULEX6fWaunCw_4bSXUA=w24-h30-v0

85d54a3e-dd4f-4a55-a3ec-8566d8f5c211

https://lh3.googleusercontent.com/notebooklm/AG60hOrDnV_Ubx86GNwVrvIYfb6CLxgXS-jEiUMvbynRp65C7ERSmmdKaWy8Xd5Ol_3kqwS5Y8ubfrPJ1xzNdbV5rd7pVhoQvcCUKqAq8s23f6_zVKSrDrOSWy7-mJsjHI77ZqkJDmxadw=w22-h28-v0

fbbc845f-bf0d-41a5-8a1d-10cc42c4ec20

https://lh3.googleusercontent.com/notebooklm/AG60hOpuF-_sMC_LQCkfR7w-oZfT0HtjyAoJet5X-JGto_oRAndRvKn04RBvPwMboD6Yon_A8zKLbMI-GKa0b7SeWIU2mp3awl8tze2KkNVALge-9M8aSXPYohlLvxAPqgPRNgG4Bxj47g=w28-h32-v0

ea9b5dc2-dd24-439f-9f35-2181819dd5a5

https://lh3.googleusercontent.com/notebooklm/AG60hOptPS5XcQ2DhLLxt5iWBGRfChPOh9P6zPSN4Hr4IxZLJEk067y7z-xFlyqVUC6PArkxAvLwja95xcuG48X8D2L-i323_N211BwAd0V8xVA9lVBFRmGsiTjeY4z_FPkrZw-8LOGFCQ=w26-h30-v0

008c3642-8afa-400d-a770-7c9f23c9dcc2

https://lh3.googleusercontent.com/notebooklm/AG60hOrMB71veEKeVYu77XjDr79t_wC4Xm2HcfWRSm6iWqc5iV3ujTnEk2VoR3n83qSgpQH7I8sXQF3zPBB5IoUN8SbZoeAcLxNKkr2FuupNmaA74KgWWVHMWiLe3CqSuwpoL39hqKsr=w26-h27-v0

78bb2933-dac5-4866-b63e-42919526fa2d

https://lh3.googleusercontent.com/notebooklm/AG60hOpneGeqNy_OOCGDfz7SDt04s6iax6YoNk-j1bZmFf7uxYPRIRKJQoX9pwYl2hejBknGf0LqvhG1fbhwrNub-P3qmjiAilFpkkwddfh6QwH0ErvwZN0X0myk6f4lAsHSQAXdhFc8rA=w23-h25-v0

a67499f2-fa32-4ddd-b9ea-8667a56bc772

https://lh3.googleusercontent.com/notebooklm/AG60hOq2ykkU6Cd3Z4OuyiLF_dHdfYojQIs2AI9bRWO0skMIxwO9uEbRSRQgVrKkS3w5bTxA7VYV5gur16m0UgXvxmb790Y1ZT5TawqICXu5foUGyG2f4CEvQ-TnpjcX_20_aJNe1J2S=w70-h27-v0

8a047819-51e8-421a-8d90-d7ab6385307a

https://lh3.googleusercontent.com/notebooklm/AG60hOqGbX2tgqOttE9jyFmpPS-yV1YHrl7gDiiPF8OB8FRuZgxJ6RNEp4NZODiazaFz93RwaMMsuqVhxHHAov9uMIGB9HFdGJIGe0FFrob0ca6LXcwnPweEJQowQ4W1CXIAc3pk3J3BtQ=w67-h25-v0

028d7830-ccf8-4a65-80ac-6d696b606ecb

https://lh3.googleusercontent.com/notebooklm/AG60hOqm3Xod4-Da2qh5m4jc_odpw93-WwJcCBzRP0tFLy9FsNgQjeubv16LwzcEYQFdIxnUfb7HxWYqQLW7Q4rCXOrtaX1VbximXU9y_snneiXDNMjH6q_Pfb3MlVSo591wO8riWc8lxw=w79-h27-v0

484d4bec-5c40-4226-bfa2-ce3b834e89e9

https://lh3.googleusercontent.com/notebooklm/AG60hOq93CvzLhTb6hkqLT4tuvfXZyx48OUrBI6E7fhvO_GPWk4Dam6obru1J3uTU0p8OKqZM-qVR-aS0ADPFvpapL3FRNCZ3Ws-8sYa9cYLlsyvTKIbQeK1gLWQ3S8Ss8W4yZbV1W5d6w=w77-h25-v0

43beef21-d939-4371-945c-5efd19252bc8

https://lh3.googleusercontent.com/notebooklm/AG60hOpQSKyrBJO1xwa9gnOjvi1rV4BM4snwQUPAB6ajKaymKGpz_NN21GsFW8wpvSc4tjvA_agD1DbfRXKx3ODlRdyzeWNThY5xDLsR4hIqlUuz-1ZFKULSq2Vye_XjJGifx6H0sYdycQ=w54-h27-v0

1dc234cf-fd11-4891-962f-4dbf75a912d7

https://lh3.googleusercontent.com/notebooklm/AG60hOpFA_tQjXFus1G7-eJd_qRDgXiCKluHJ_07j0HfJ7jbQ9AT8-V3W7SY96SbfTs3wxe8fJyiaWXqPLjjSUX7WoiFdcP2d7pppt1O68C9oPLc6E43CPWk6o7VNn3TgH8dwCKX3EGD8w=w51-h25-v0

114658a7-6b74-468f-9042-a08de271d2dd

https://lh3.googleusercontent.com/notebooklm/AG60hOo-Me8ZzYRtB7KV8oyU3G7Md_2xdCd_dkhx6-tboY7QFsBFbd3EhDwcFMLSeb5Xfs9z3t_P4Sk7j0EjJtYtAYAXtKKtxs6yZjcfMw4SS25nX_4ra2AHJc7Q6ThpqNe6hwzaLAoveg=w39-h27-v0

90cff2b7-20e2-48ac-9310-92c43ea9d13d

https://lh3.googleusercontent.com/notebooklm/AG60hOrpxeFt_5so_YE27-_Qibt8VxV2Kos4DOWVuzzl25mtTp7J743iZG7gQGznWd2AW_Rauv0qbTkNPzrQx9YickxZb7EqCmx5e9OEWQxMVQ50Oc-9oIhs5Kp2l0WcZHtfFSgIoL31Ww=w36-h25-v0

9f10edf3-978f-4f21-a6ea-170951e15082

https://lh3.googleusercontent.com/notebooklm/AG60hOoTYi23RNG-ZLPv7cw4vDm7769HHr5NWyVghVXOeG6NKGzGF72PD3JCNTJGbMeUsR1IgdzMkglmG3ZBGNbRry7Rg3-64zG-SyHmrodcMAep-befRuBjlGuH2rn07ct47pVJR1TSmg=w64-h27-v0

ac9019bf-4dc5-404c-b3c0-7d0f3370c12f

https://lh3.googleusercontent.com/notebooklm/AG60hOqAkWa0unAK6YTuAE6lWSmEaTSAn3FTWq2ZgVc2azH_QRObM74HoYioaL2DAn4KafHLCrx4BIjAvgnIH0pev-c_nmUrDkx4hbqrQZO6lzMQ4kzwm_eOo_2T0Oz57UQc_cZsgs_QPg=w61-h25-v0

bcdf824e-ae9f-4124-bc50-49f85b371fb7

https://lh3.googleusercontent.com/notebooklm/AG60hOp3F00QlYBz3cmOOuucuCQziKHI8Ez7UWY-gtvkPTdL07hC8Y4PyB1xj_mD8gPZqmojdcvJish-v-HruFQKp7oeJUU06wr1HIneF0VDHxujaC3DDmCpv3a3AWk2s7bsT-z2FeaxKg=w68-h27-v0

087bbff5-b088-4ae1-baa3-8321263f3715

https://lh3.googleusercontent.com/notebooklm/AG60hOpvp0Q5JR1U4CHmgA7S2Zly6P0mdMArMu2X6JlsJkNLFtzwvwmbmGVgR_9OWlHU-BAOGwI82gBisW-U1h2AosMAkWocQKZu62_eMeMPUKDC5F-wm5dRBXYN9D0e3Ou9ZDtYe6HO=w66-h25-v0

42a5a0ac-ed6c-4804-ad71-7f0d10cd5f06

https://lh3.googleusercontent.com/notebooklm/AG60hOoDcJ1ZgXt0TZoDTjFqjfqfdf2gzQmC-nvbzAf9Ed5m5sAqoskErJxLXM5k3rrMxi6WWfPWA659kk8IWoVdwTyiq7xxV-wJ2VIO_BVZyNxLb8IEgR-svrJRoz5h1zO-7rfegRjdRw=w34-h27-v0

7d0a45ed-f2dd-4f2d-a5e9-5a884385c5ec

https://lh3.googleusercontent.com/notebooklm/AG60hOpzicQMn0YJgzuGORtzwzpkRaqZMazMqoxJhJDDvECrYDLetR4yoHibIjmpbrtyikhjy5dwBd0Zjlbxx-P56t-L9RVuvIs1LjtnyyWPzNLRXLZS7LjY_5ndFRNcwf_d_1mfNWt42w=w32-h25-v0

211c1d5a-c1e4-45f9-8524-43267c42ba7c

https://lh3.googleusercontent.com/notebooklm/AG60hOrjejXwZA33xkkMPh3vuAUk0gBBtlLTepOrKf4Tw9X7lCDxavd4AQC7jGsL-WkSe7Y9puqMP1Zny2Zc2quQyouBWJHMCafgVS2elWOEb5ROd5fVij0-VKRyOV_5ViDr2eEoBUpWHA=w55-h27-v0

dc7a6bb0-f091-4da6-9c0b-892a60216252

https://lh3.googleusercontent.com/notebooklm/AG60hOqJwBY51pQUP_e2srLl-ry-JtBeM5MPJQvagIaZjPI7_3JF4V5DT2fP-T9-qZ-iS8I9KpFLER8izROmpEsNEDFZq87QGlqXPpBkWAhaZ88vt52NvknngfNi_eQx1FKyyjMeRxO31g=w53-h25-v0

f25696e1-749a-4de6-90c6-6fa8f8e1790c

https://lh3.googleusercontent.com/notebooklm/AG60hOq3HlnRgFWpuGt1Cf9TqWKuFo5MFW3LK_1X2pp5M-m0p9YhNQ1VQ-iFzZbXpE4ieWuVJPN0CrWr3Hgj2ad78sOk1iXWdFdb7LaoBzTJVx3ftdC7vp0NgxP9wjFu3fQYXH1Ip49e=w64-h27-v0

05bd2482-dd75-415c-ae70-a9787205955f

https://lh3.googleusercontent.com/notebooklm/AG60hOoIJDCvbJ7lIopOfxXR0PIflOGPi87eNaUtzjwanUvTqZrRk6aikHuP1dhZJpxGMrDrmEih5vR9iHwt93w44x8It28nsxCpfzy1GSAi_f7Ohwl4JdICWI-2-49deJ53CgDdLecE=w61-h25-v0

cb9ac2fc-8147-4573-80de-a52d2698b72b

https://lh3.googleusercontent.com/notebooklm/AG60hOo7bbk4wc_uCXTE4PgJpHbSxUkE9qUDnSdJSwpjn6obnk6knBUWNSpgJJS1efYsL_x3LEg-_VpcmauWUiTJVRdhkKQ3KDfGVfHQubnc1RMyKcM-VLvIMvTX7U_evUbtbjM8AmlvZw=w39-h27-v0

15ebd5cd-904f-46b0-a73d-218aec0718ea

https://lh3.googleusercontent.com/notebooklm/AG60hOrdCP_lvyO-7qt2y6yFQWHhzAx54ogVI08VmqZoMpADuNKLXpeX5I8OsH24qYW6K_TrJs44qdo32a0KSOkua9hZ2W_c2hkk4csyTFu-v2YcCLnSidMJcp0exULnwPMOxVHxYfAa=w36-h25-v0

2595be1c-b07b-489a-9cd1-f270487b62dd

https://lh3.googleusercontent.com/notebooklm/AG60hOpgEMmT3M_1m6g9CUWO1UVuCuLLo9fd7l0pTk-vJUP2t1sJUxmg-5PnDAaSj1yJZEH4eut0EwgFPQqiqmotHnSUUIQlg7Sv9ERHwexsVr7BOdXr1t3r4yvfliCH_3sJdGmpBr8xAA=w66-h27-v0

43f33f54-6b0b-4388-9597-a7b9ea607bb0

https://lh3.googleusercontent.com/notebooklm/AG60hOpfuDjW2ZxEW4RqmQrH0uTU6Tb7JJmiOUr4PYVqfdSOm-Ugtxk9hIb8jWP-pl4wNeMb2ZZ8_G384CjJk0YYSMX4LDUsouEmnuMUKtWFY6GYA0rRZyRVPRvm17e1blR_EE-lbIBz=w64-h25-v0

67c484f6-e4ec-49aa-8861-3e6743e62e56

https://lh3.googleusercontent.com/notebooklm/AG60hOoW7eXPV2JFEVtzg1CIkzdgX8IHXNJmkqMZsLKOd4kEkfu03QoCtbeuza9YbSXh7z9uekGlNDhqO5DF1t34I46Rkn91zHaLQq7hC4TjImzYqNz55GWl64pUt1C59kY1Kmd9JxOH=w72-h27-v0

d602c764-fcac-47e2-b21c-d74e347c08df

https://lh3.googleusercontent.com/notebooklm/AG60hOputYjkECIwjlHxttTOoIUDCi16CRHKGE9HHmntad7jUjuvBqSA1AxTGrwf-nUIcZouK6d5blsT1oxAmkWlp5T0ov-mCKwV54DQLeMDSz0N-8EINhvN8G8BiWLnn-Jvh5Qf906I=w69-h25-v0

d3ce852d-0229-46f9-b929-6b3a13ddabfe

https://lh3.googleusercontent.com/notebooklm/AG60hOqGkWEm7MTR6VaLGMnY0SIucY8IR90x-UtCoIKRqm58P1bmkVcEJ8bhyADZQd7WdlJ_eOiuUx-6MHyvbjANjXO8ZOqZhg17-fFNlgJysuBu-ZwKTsqbpGicQU24efTm-h5ai_73Bg=w40-h27-v0

33c51b35-3292-4920-aaa2-ea29bb71ef9a

https://lh3.googleusercontent.com/notebooklm/AG60hOrd-UYn9hmHK3HfDziK2Oj3haCD9JaWzXp1H1Sd7cQ1KgXKDv8BzmD8RzzsR9gDJdyXxDbBxmjXUZVakFYJC25c7XIolrdz5TB34xkfQdZyECvnNuJaY22wSEQeM3KY9ECDrA0nlg=w38-h25-v0

a0a94e74-a120-4f5d-be86-559dc63f04bb

https://lh3.googleusercontent.com/notebooklm/AG60hOrFfm7RXBjMDrmI1_NTFTOwarRVog1Vfhmkt9B8Jyg6ioQUwCHlg5F4F9eEaQTuJNlGvvljlxzfGfssALC9yyN0wdEe7tx2Kq2Ff7rx_cYuI0NvWArhjItB0rHITvzie3w9RMoB=w70-h27-v0

684766e6-b514-4e6b-b521-8e8279cedc97

https://lh3.googleusercontent.com/notebooklm/AG60hOqhEStt780_xpVpJ0z79Yjg8rbICv7fSB-B21MWEqW6UfogzhBdGkBUY7Fewkmaox1Zs7i-ruIpvF_d8hOy7APk25YDvQ_BdH4vUIkY_NZVKKiHhryjr5anhNaig5ReoysxT4PgQQ=w67-h25-v0

2301f414-488b-4df1-908a-1c92ced93233

https://lh3.googleusercontent.com/notebooklm/AG60hOpgL46I4qdw9g0Yy43e8bwRp1prAlfUPQDeSJg7SAB0MUaAQK9OZO-nTPu0ODhpXP1gsU8b0TDvHnmbT2A7gJKLrvwhAiWt3Chl1JNuvIiJvD8yAI6n17uDseC9sprfj_QwhLn4=w27-h27-v0

e06e5366-c035-4338-9ebd-958d86ff5156

https://lh3.googleusercontent.com/notebooklm/AG60hOqV6AhVvld2LnhO9U15MjljOP6MuxFOsl10Hr_ylIni7GFktw6cds3kwRSt6AQnb1LWoslE8PBntRdSUf1v8-LiaNQfXNr-FBjRsSzGe-gV812QeaCCjdLAZI6OABrqQlWCRq-4Xw=w24-h25-v0

a135d69d-56cc-4231-9623-c06e3ed02edb

https://lh3.googleusercontent.com/notebooklm/AG60hOplCjuEejmqzJLg6X282tdR8XSKI1KuWuRwYBOiNAdzIGYz9PuGPYnTEH1aUuRnpV37t0WtVbiIj1f0axkjz3mZxiiruq7m-XPcvm7n34gMh8Xoav042qQqr_4LF8MGoKBFCifzWw=w28-h27-v0

9ca0fea7-0a08-4f52-8f28-a598291fb8c4

https://lh3.googleusercontent.com/notebooklm/AG60hOoPgcBgWGcHvi-PE80YnIszbOUXHQtkqNpwNqKdW1aeQiHQFkNIp3J-M2A3l1qOi7h7Og3OjquxWy5C2fzSFQe7a6SEHEb6n1RGuyZcxPdCtiiRoQ7XnfJJAoliIXO8rDoj19I7_g=w26-h25-v0

6fbbe7bc-faa9-49f9-ab94-74988d60dd38

https://lh3.googleusercontent.com/notebooklm/AG60hOpu541PonzjVoEpoHu-nfGB5brEcgIrPzPkSwbqxpKd5rs0tJgOX2EGHJt05yRGen5dtjSFj99UCj7LBWmTt3LC75AGXGQA8qc1cWWHuxqIeKWw5vm96fbvHTHHITVtmvFdbdrsnA=w89-h27-v0

71600c41-073f-4ffc-935e-e6677ab77ce1

https://lh3.googleusercontent.com/notebooklm/AG60hOoW_wSrCvsTNcHw696L4LD-Mh2YImN7oQSODynsY0NSceNOHWiT1YSa5UmmR68vYql0KPmHTYTsphydcTHa7WBVFCPj7ei-jcG4sTsvG5AlbjuGIgMWIViRiIoSZA9GrVw5MD29Ng=w87-h25-v0

c4e6ab28-e328-4447-a53b-be3ef3001b66

https://lh3.googleusercontent.com/notebooklm/AG60hOpU6dxsPfUsC63JrOjkUSn5Ru_TSWi_pCudJ4WYyKdhZ2XDLwo5Lp4GeBECc_I21vEuAA2_vHG29KS8f8CvCovN-ov5AfhsiGLUTsh4YnEbQypNItP3A7oCwPJsVlIir3kLXXE2IQ=w67-h27-v0

6218baa2-aaa2-4bec-8f36-833423007701

https://lh3.googleusercontent.com/notebooklm/AG60hOrKsWpvLB5O0VGmIUMsCb6kkl_YP6IniB89ivjrpdvs_OiSFsi_xl2V0LQzfLZE66FbNUD_2htO7bEZHVWUNa7NFsqMoGps-KX2TMdbEbWpHCtO4YdfhI8t6NHTVhEHm4VPXGsF7w=w64-h25-v0

e3a1b1f2-7818-4b14-b51e-e05842dcc06f

https://lh3.googleusercontent.com/notebooklm/AG60hOpJDbY4AWY09UrY3FY2XTnCd0DUy_T1y9G_E3Fq1_4a8ukzIdbyPkM2j4HhplwmgOAiF4vev8QMPsIRd-Rb_V18vUdXBLGu5PiA1bnZWdrDJjLG5l19nUn5vtV5Gm8y2uVmKbczvw=w24-h30-v0

6be4f39d-6cf0-4f9c-ab4e-4919e35b20e9

https://lh3.googleusercontent.com/notebooklm/AG60hOquu8ZAwEif9OHGdiuj26p6HB0uQmI-5bwAYs01stwmVAzs_OWAq-9sp-ryBtIW_Yx96LoaxRbyXI_ZlJAGnlw9KcKBR5rTdI64cRZBj56x--D9UGdbiC5N8BXLjiA8dCB-csw46w=w22-h28-v0

c5bb277c-39c5-40a8-b8ed-5723c3486fc7

https://lh3.googleusercontent.com/notebooklm/AG60hOoeIVLPjGF4GllNXiCt0LD119kY6Lu-7IUlW2_HVR4JELsiEM6UhWH6uK6XfQnyiMW6zjm4afY8WO-zu4gvlY2lLmvnhrARCshpWmbQHDV29wYaycx7zmCw5FLXUq8c9lhg4Zs1bA=w68-h27-v0

4d0626db-e704-43dd-89ba-3c8a2404b5c5

https://lh3.googleusercontent.com/notebooklm/AG60hOrudAXEM_rcHTJuyCGNhcoG9TYOtjGon0xw8sr0AOdn5-h73Nh4aaQtzlIzM9OgLJziU5Whio4cSTQ4KocYsjGAc4bTQGCOsDwPy1OUcmFPHZgoNT0s6MCtNnaJls0e25rm8l7ONA=w65-h25-v0

48bc46ca-a88c-437c-823c-52adfb9cec8f

https://lh3.googleusercontent.com/notebooklm/AG60hOqmVVWQp8qiC-RIpa2V7iGiViSSZko6CqNMfWnvCV2kE-uN2tBN8iY5iLmgkxu9N535d-yomHkPLG_9hKnQZhjHbQPnQZNhWRKU4XgJZNmR2ilsXsMAcukqAvKGpYdtXqMqzFdf=w64-h27-v0

6af20cbc-4941-4229-b451-6952fbf2fbb4

https://lh3.googleusercontent.com/notebooklm/AG60hOphQutk8THz5vzKXsQqxLTLl-av9D9kMjfQ9BmN_97chmUzdGWVnw4FpZNK41zDlFoMIDroHuMoBfdyfevwP6KoEvFFZcdTRHDfa4ktq3Bf-ZNnYwxmvh299KwffAubtZIm99_9tQ=w61-h25-v0

3dc2e124-de1f-46e3-af85-cd03e265f980

https://lh3.googleusercontent.com/notebooklm/AG60hOqAT9qdWya3OMlw07BkGGQ6SaFBXa5U5NBk2dWiiP42H9UBxnjne1h2gp0hB7B1o-CjYQVs7ADfPtdEQU-up9DtJNG9_ywTBINWXnuBTl1kjxuONsgkeHdy-P5IZDnRsb64JsZ0rQ=w66-h27-v0

8989ba54-bca3-45a9-9e90-8d5c2990f214

https://lh3.googleusercontent.com/notebooklm/AG60hOoMZbkmQTLLLJI_YheDsJBa4OK8wPWfDga5CGr4lMGcTDXyY-7RfVkRwa2MwQka53Y7LZS5riwYumHeR3I4VhSA5bxC1bhYNzC5LYj__X1FBK2HCOeyIgP9bi46DOyUnHxtr13-=w64-h25-v0

e6095b00-e220-4da7-a54c-eaea410d2569

https://lh3.googleusercontent.com/notebooklm/AG60hOps5isnV9A6RjB5PqeARTngInPPaETWV1e6tW3fl8SJczfINXrilLeQbuUo4V_b4IwPF0GXRlCEq-GejmJh6yAxygCnaJ2rgeWu0LUSwWr9YmstUutgfsiOlT_gNHJnyLtXS5_lhA=w39-h27-v0

39a37743-1205-4142-84af-f9014d7813fb

https://lh3.googleusercontent.com/notebooklm/AG60hOr5EH-ld8OgNH3fJc2InO_nywXSRQupo1qi0BZAZwE2fm-o45RPkAzS2lOGbpIJegJWzJEONRsZQJOkH3Y4yWRWSy-jCkeeo50E0be_26Ems9Yq0X72VaFqUYvu7mqKC0riNYolUw=w37-h25-v0

21d3a1ad-4eb5-433f-9d86-af48aa9cfa7d

https://lh3.googleusercontent.com/notebooklm/AG60hOorkC6kXamdeYsdAvd-sDhXx1MA3GH54C_WvPdc3R9qGr_7l08SzasBrhkaMy-tnMVOdsZ-89LwdJ_YVTyEPcbcLXX9rgCp4zrLH3dntiYGYbKIzcMzSls_dDUajY3bfKVjafVsyw=w57-h27-v0

ed939f31-20b7-4858-84b8-9deea6c66ec1

https://lh3.googleusercontent.com/notebooklm/AG60hOpCgzmJ-NaaexCJxFkAvu7s_MPYXe2CA6_uWHVYdsd96Hh7uXkIcIZABhAfb0XtE2RWWqXcboVQlWFgQUCSlp7ZJpbPr9pm9SD-Er1FAqDpugqDKGlMLmPXIIYSo-NWFVRMdMJXaQ=w55-h25-v0

e8c68b22-41bc-415d-a59b-cb7859d58096

https://lh3.googleusercontent.com/notebooklm/AG60hOqHHzvI-Eh0945GPmOyszSkviB7dCDRDI7C0UaPhiu1XDb76RhjpGQmczAMjNGMlCdjRqfHaCGX3zbM5mTpLrNfBARcB5UPSK4do7jzS4fj90KoNzng_uV4KqVQ3zJPsZo6fUQX=w71-h27-v0

7763936f-7b05-4e2c-9076-0bc3dd81b943

https://lh3.googleusercontent.com/notebooklm/AG60hOrjmVCFnysvW6lyFBSeFU8ZR-Kvt6F3JxF69Vna53T8HbOx1gXkF-Vefbw3aD2Dimync9Nd6sxF4YgN6Ln8jJvawe-tcH2yyGvXSN4IKMFrjmiw70NcPegp3eKhV_qgwF4_UavEDA=w69-h25-v0

d9c807cc-0f80-4d22-993a-fd0e72b3b96a

https://lh3.googleusercontent.com/notebooklm/AG60hOq0DHEeZPNLd9dslzFOvyB5fEfbI3ZOQI3J9NWVW3gH6dVSw_RZnlSng1oqTTVQjKHfEMuzfgX5rXeVMSGdP7360yRXFUwJrIsdcfOYPG2clUykNqMy-vee9zQKk81SGlcNnl4O=w39-h27-v0

4185305d-c5fb-4443-bf37-f54c851cbeb0

https://lh3.googleusercontent.com/notebooklm/AG60hOoBYaFGYWRI4kzQrod4oYWAugQnRS1izledFTFBKFOzKd1LjOGArkkPKnMZ0XJm2Gd2dNtRs8A3lgXpK_sZR6z01H-wHeeiwuWwG6F_hg8Ee2bPEPenlMdleWAbakCxdYHD-F_e=w36-h25-v0

a3bc223b-3a05-4e18-8f61-690668fc645e

https://lh3.googleusercontent.com/notebooklm/AG60hOqNDjPQOxh4kk2y14-diOLVTBpkdF9yEAywYXGUdESA7VjUZllRQvdirvrNovvyvqeE9yhEUeGHbUHJaNLymqWrAuKmbp9zSPMvlCH3uraDfx1TjTHteQZxAqbksOBqUKssDaZXzA=w62-h27-v0

7a3424ea-bbfd-48b2-8bc0-c0eb517f35ca

https://lh3.googleusercontent.com/notebooklm/AG60hOpXNJcHKMW969s2EkO-3dVNnwkuZ9d0KSaNakjwJ6wTjVuaYgrcW7utCbVq5C1mTvCH4Yo9O1ptrFmNZma2F4Bvai1UJZfCwsO91BVFIhUEs80yowllPlojKUAAw5QVl163UfPTcg=w60-h25-v0

e31b0cff-8646-4ff2-abb9-d5940a6c49f8

https://lh3.googleusercontent.com/notebooklm/AG60hOoCOAPNRe_DDulF2PqQmil1_7Ve1LzxnhtitoZMgaYaMVSd5rtCu_x7ZbvTLt-LqKF93yioP3wszk57cJzyL9ulAbMnXrTEsNsjY3OsFF5knmU-dT-RhR-xnloWiRNM5XWiOVJEvg=w32-h27-v0

88ad1b5b-3827-4e4e-9388-0ea99e2443d8

https://lh3.googleusercontent.com/notebooklm/AG60hOoA9Vgh_RmoIAlrV-dnkW33Idwjm5tQB5QRTbepNpd4uyx6OVOvMozbq42mjlgPspideVJatlMlJZhrmugjQWU_2D3rRAvz3Dvs3_sGkk2XZobpiNx6v7hn_hiCwviMJa_9Lw-i=w30-h25-v0

1067762b-3f17-4228-81e4-ecfb01c5d7da

https://lh3.googleusercontent.com/notebooklm/AG60hOqvavWNKJiIqTT9slMHeNa7VduBt_9TpoRAHooLUtHeV9AP_QqAoNXapUGnL85D96c4vqJ_t7iJsrPIN-NQxxzexSJ_wYM08k2soA9l-QGTomfYQ-M91EHzWV_s5eCH_4n0IBoPPQ=w62-h27-v0

3e0230b7-f533-458b-b28a-e1141c89696c

https://lh3.googleusercontent.com/notebooklm/AG60hOqn_i4I7R6RuMA2rFBMLsh-CJmUlieE3dYFHLHhLNqlIR_erRkaPjwa2fyIx72na4le3WHJSzG3ysoP1OgleOsg9cxOa5DUwGqiK9hv3PVNP9t6KoEcB7-tBUvPr49d1OjPOyG_=w60-h25-v0

02594fd9-203e-4d3a-92ce-58b6315544be

https://lh3.googleusercontent.com/notebooklm/AG60hOoVJdou4Lz22F9r7t9Zf81X5JLeE3K5zIS1gnG9cdIuHb6piVdd6rNsRR6Bt41xaIB2E30xqlxMEhXkpdTf30AUL1tWCbus4hF8vqRYgtjhKzF-KRucQkMZVPpFa3tfiUVlLTmWQg=w89-h27-v0

7c2257d9-30f6-45d5-b85d-e7179dd47f8c

https://lh3.googleusercontent.com/notebooklm/AG60hOp39r5w0KHTf5M-NclCjrYpfffGcIacPQe_hW8UIM6qcnjbXLMaj9ZUIdLn3DzCHtCgHTC2hXY91aaNGzg0YAPdkLtxuysHx8ARlqIbC4QG4BQGuI_JDlULles5Ds5vyY09aBOlaQ=w87-h25-v0

84069d98-bc0e-4b08-843a-ec6519d52638

https://lh3.googleusercontent.com/notebooklm/AG60hOoOPN1PlQFb2-X4eQ7PLTGVpaUrCsqVfvkMG1Cs8ZnUxndIylE5SgRvbEbKClSaRSnKYc1YwO0Ijh-l_gyDksULvpA7A5U9pUEriXHLOu94_jzDdkpjDg_tjs2KVdV6mBXAp-10fw=w535-h68-v0

26e07a0c-250e-4b03-8b01-8d41c6cc58d7

https://lh3.googleusercontent.com/notebooklm/AG60hOoVKOI6o9lXtBhiwh6DR1Xp0Yjh5itr11jzWUNzPH1i04Nu4HR1_Ryf3nc2i-Tit7j-tRLjPNqjFONf1Gh1ACzg1X3-93Pobz-vultwa29WS7FzOXJngZREngr4M0VRcFXNXfiprw=w151-h36-v0

f7a29d24-1e13-4ed1-b6d3-c6cbcfa1ce2d

https://lh3.googleusercontent.com/notebooklm/AG60hOrDr_256Td-0phha1T8-K-utiDwDjq0nNXzFyK960WGJ_INZFmknA3i69xVDxnNqtA7qzSPxIEH683P57_vl15vQ17qW8ZE01Rht5hanXUmktENNZi_zr7Gntty0_Fdjv0mP6a8FA=w300-h359-v0

ac8a9af7-abf9-4967-a53f-435be9b62cfc

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

GAZESTEREO3D: SEAMLESS DISPARITY MANIPULATIONS

## Stereo depth adjustments

Applying gradual depth adjustments at eye fixation, so that they remain unnoticeable

A REAL-TIME CONTROLLER THAT APPLIES LOCAL MANIPULATIONS TO STEREOSCOPIC

## IMPROVEMENTS IN DEPTH PERCEPTION WITHOUT SACRIFICING VISUAL QUALITY

## TO ENHANCE PERCEIVED DEPTH THE METHOD EXPANDS ITS RANGE AROUND THE FIXATION

## LOCATION AND REDUCES IT IN UNATTENDED REGIONS

## OBJECTS AROUND FIXATION ARE MOVED TOWARDS THE SCREEN TO REDUCE DISCOMFORT

Images from Kellnhofer, P., Didyk, P., Myszkowski, K., Hefeeda, M. M., Seidel, H. P., & Matusik, W. (2016). GazeStereo3D: Seamless disparity manipulations. ACM Transactions on Graphics (TOG), 35(4), 1-13.

Modifying disparity of attended objects, seamless transition between them

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

GAZE-DRIVEN USER INTERFACES

https://lh3.googleusercontent.com/notebooklm/AG60hOpujdgS36bc1A8jsryiu4AoAlYnGr28qTds_UjZWD9oz4MBzKb__lMIUsJsDsu-FrUCwDDJ5lWF8TIWg7SvXRIN2ojH1YFcj4l8b9MV0-NMdxwSbLdujq7Ww2B9P3Ywh_ayJiHWSw=w116-h28-v0

b2984259-80a0-46e3-82fb-eb0179248fc6

https://lh3.googleusercontent.com/notebooklm/AG60hOr9uuZE1fMEL3B1QyWdaWfc8rvP1QWqdNTT37gvy6KPN2pjaDQHy03pdC1Aa668G50aU6lRgfLeD26MFxMNOTsSSr5HzA82_EywgBzgvvwcpuTJ-NBCvg9pPwfBSIg5jpn3kiOZTQ=w24-h31-v0

55186282-ba0e-4064-86d9-dd5f985cad7a

https://lh3.googleusercontent.com/notebooklm/AG60hOplOWq1k4Qr-V2zNIBNENzKflKWfLhn8rnVuec51-FypM8LtClHOj4hc2XbvOZrWFK9HdPP7D4OoFJr5MSLEqzI7BLFKNQFkxEvwIphwqNpCW1n2WLYTHOqaknK__b37xeUTdUXcg=w22-h28-v0

2263755f-a165-419c-86ff-b601f706c320

https://lh3.googleusercontent.com/notebooklm/AG60hOrrftW1FFa9rOpJ-k6BfL6SK2c1cqyM6yyFsRoBbNJ0g4B2vAjCcLss4tz7K8e6sI-GQ9ddAfx2HfSo9HMkkIhDANrlrIFZ96IA6VSDcozgmCerunq5a5rtwyODtqIMs3iBGBZ1=w65-h32-v0

7d98ea79-b877-4016-bd88-26887d0df511

https://lh3.googleusercontent.com/notebooklm/AG60hOqb3XRLvTRCl7Cujbi9FPYt62aVjxVoeF3kn-rWT9-o3bFSkStVk8f502BsyktOeXhH3b-ngjvwHfQ45NaZJQQsSoemFbKKFsFUWXgRZZuDpSMGtKzgrWDIi0EWmia8Ra8wMG0ZTA=w63-h30-v0

bcce58c2-eac1-410b-ad31-aad2ce363aa5

https://lh3.googleusercontent.com/notebooklm/AG60hOru2Ow4cThHcxRLJz2SKT4p8CRf2tPefi34THgH4_pknUBr6bM01_BNygqMTJTVCCY6OWHMVIM-2wDNDh9yYZEqHxVVl3wKs_eRDHJnGcaYHQKO0e0LUaAsyFTPStKhlgZlPyXW=w74-h27-v0

b9011a2d-bda5-43df-8482-a16a43145fe2

https://lh3.googleusercontent.com/notebooklm/AG60hOp-7mOrmb7BPEq1JUlZVaSlSja4E3hbf8gnJw0nu8hgxz4DgewOiPTbUyvCMnLyOHOH6dNxXbrwCkUXz3hCziv7ssKzLkctgQH82OVumt33gjWWlRJ5fIo5J2YoY1zuh4tlRsQByQ=w72-h25-v0

e1692eb9-361e-4e33-8da2-92c56c0f3dbb

https://lh3.googleusercontent.com/notebooklm/AG60hOpiuHRX8rBKZczJrfGouvp5i8sYLqyw4QxJslXz09w05za6OfP1m5HgaFaC0G_Pwz6bk6h4bphD_IdEEgs3rns2KSBVNPrvJ_1UZyow4xGuBzAiZqT7zdAIQa3xWiWdqxJig2Dd=w30-h32-v0

1ae66ba1-011d-46c1-9148-ab27f7797405

https://lh3.googleusercontent.com/notebooklm/AG60hOoc_CfYlT8sKBt9RIS3BnLLrGcMRZ7CgkAR-jPyn0w1wv3W6YgfNeWXpbHHNoHgEC7SWAdUCk7dzPmempCVi6FXMM2LLwuNIziMJjn5tFWYLC8oyRoV5Cj57l0oFARU0sB3j4Siiw=w28-h30-v0

81a0e575-ead4-4f6e-9ad7-4f1a7dbde178

https://lh3.googleusercontent.com/notebooklm/AG60hOouoSN5-B1L7AnLWYG-IkqSb35b4oEbpPE3n986BKMdaDcL0bsOb8KJLlsM6pGNkM-amgZrabh-RfSAl8xTGTu96ER3nAIGmpJHe1-p7_ukvsZleQlwxGdpDa79RJ1Dth9YIHgJeg=w39-h27-v0

37595cba-af9b-42de-b6ba-2bfddff6aa7b

https://lh3.googleusercontent.com/notebooklm/AG60hOrl0sMX2lvYmOaPEroaHEBAjpYifocgdr24nGQtxcbRamoxiOmoIM1958qqkV00RzwgVwaWcE_1WAKKDuqSthYmwR8y2guO9OvX5zMQQ4Vvh20qTYKqDYOyOc6KN-x4P9N1pX5FwQ=w37-h25-v0

9f2526c7-0d34-484e-a7c7-de48a9411c00

https://lh3.googleusercontent.com/notebooklm/AG60hOrQIljZxY-kInvbMkF5NO5Ljx9D4fe5MMMRKdyyJpOIoo0UE7J169YFWBZvYZQNY5ubr4W7mvg9D9hTov_erqh2v0jSrzej-w2dtcsrWCBU3aB5eUGT2wSA6d41MH-JvtzXe67T=w24-h32-v0

73292f0b-b2b6-4b0e-930c-1ad877f6b70b

https://lh3.googleusercontent.com/notebooklm/AG60hOqs0i-07IV-1LwlhkWgjKrDcCDA8YckgmgbU0jzsIOlTdusok0wiyYDzDxe2C7bRgg84F2k5v2ECDn5oyNziVHAaarnbWipT2qEu3MNUgW3iMfEqnZpUtaB1eYpxl9nQ1IYMSrA=w22-h30-v0

053c6d5c-00e8-4f34-b496-598792d4f589

https://lh3.googleusercontent.com/notebooklm/AG60hOqsPb4grb3l1c-jY0plFykfhf075_0mu2DbC2Is-ZmoF1dY8vS-oxXtTHyZ4KhB8nq4GmUKu4qrrzx7UADFG3Uz8s-HhgsK0M90mcCoYptXwCslQE6j8TWZmOOPQxJsyqooYc_agA=w74-h27-v0

0687be32-6ad8-4d5c-bbdc-2b6ab7cde589

https://lh3.googleusercontent.com/notebooklm/AG60hOofA9wYd5lgz8858IPzeuEeBW1pMxctFovjaVZiwYmgpdG8QpWnwuctIxhmWCH3mSmswizanzUYl8dd-gPNErkV5YfHb38FwFFdkDB5-5XbadV-4plTcxZudJ8EIqg1KOqzQZCB=w72-h25-v0

bf71fa1a-8420-4524-bfa3-b6fea7608b24

https://lh3.googleusercontent.com/notebooklm/AG60hOrXwqNMFFKX1Xs6FJQjysbYZT7G-eCP-aqrkGfUzd_T8QH-fJFhBVpppiWsJwEJDEoe50tqMbvZG3jyDdYYxN9wCP3xVaflsJ50fFKM_VilGY8Wg2frt9geyLsfiO9XX_LE-BO6sA=w54-h27-v0

928f49c3-da50-4b83-b368-2c9e8aa72377

https://lh3.googleusercontent.com/notebooklm/AG60hOrR_FWnrtZUVSpc7KvUW8egm3ss-IXxfCk9spVuILz4Jqj7_MiLxz8Q6c-0sjDr8w-WRn8up9I9BusfIdCoy0NaTAMaIf5lbNuWziLtBNG9qm5MLzSfQwLUnIu_QNHWoUyM2tFvkA=w52-h25-v0

22b7a937-d21f-4ecf-abfe-bb6eeb4edcd6

https://lh3.googleusercontent.com/notebooklm/AG60hOpQScJL4LnCWY7sJU35jjN4Y0iKSrEqZ63Nv8-OH5Zo8dYhAXArLZtGBOy0fAObHfs441pzDM-_XNS4icGMCtS2teTHNL8fQCqJD05U6f-VvYNWwzMf5lA3ohbge8Pxs04fV0TjKA=w33-h27-v0

47b0bb61-391f-4b86-9e82-f61cae30d85d

https://lh3.googleusercontent.com/notebooklm/AG60hOqMpXylgG82MQIFxVyBDwkPYJ8N2x0p5EhU3xZwLnAPdZ9MHcRvwhhM3AFdVhgjZEeGZSICQkLcZw3VzSpee5_4W-oVLoHOd6Lnq9ysAdlyP8I1EfblGRN16_QeVf00LPMcAw1K2g=w31-h25-v0

b54292be-b52f-4180-9c95-868863de905e

https://lh3.googleusercontent.com/notebooklm/AG60hOoXGx4PDKSaAlse_RU1uCR1iTB4NXbXefXjYv6R-eeDJBtyS-uo2jJ1j85lEjHpKONFWAUmCOqTRbR4MtxckqnbdE7S8j7M5N0Vm2e2Aj_2lhD2b6wb2TjJloipDJxjkk0-YP68=w39-h27-v0

c0c9bd83-e2d4-44e5-8d50-871f1a1f8a6b

https://lh3.googleusercontent.com/notebooklm/AG60hOqHJ5bu_EbLJo6FChxwzGHU9KG0kBx0VT-Ky8LXHGp7_pX2c2xHbIlaXIE5V_siCR5F2VdC0Pux9E4S9FVt82GpG33HCjdfmjOUVz3YEoo4epPNoyU3OZxtoaSwNwW34XgXvJlygw=w36-h25-v0

4912e551-e567-4aed-bd11-e010ef857f25

https://lh3.googleusercontent.com/notebooklm/AG60hOoOhW9DZEWCKmTinNwleh1GGPS0bkA9QTtQfZhjcNbo3XBmZWJfl4EZPP8C1bmuAjs1BXX5b-00ATPQWpC5VDSNrXOmcLmxVFSHnRnTRY4UACt9PNPH5_7cqhl0jD-drGqvvOzW-g=w25-h32-v0

ea47bcb7-7268-45cd-b7df-b1b2de0b04ab

https://lh3.googleusercontent.com/notebooklm/AG60hOrVOIoSwT930iBMgOcpeKSBC3WABdUdiF9XP2Dj-eu_5VOduDYeeGmeY6yXcygkfocofFObepCaIMnL9a83WyVoSiji_tFB6lzsaDuuTfpEq7Q1wkWqtKkPxGMHVjHkhvQYWkCaiQ=w23-h30-v0

06eee4ec-2326-4077-9ac3-dee131cb21e3

https://lh3.googleusercontent.com/notebooklm/AG60hOp7rnNhMyDssDZmbhwuUNbJ6XO4DRNCZXCTwwSv6vGcUxeEcyC8jfRJz_SufSjgpoNGlhXX5qeJdRpd_uT2G8Iol2Ike9BNdFBc9h5au3XcvXkOcEP_kIY8EU6oKqn-8EtV2nAatg=w73-h27-v0

01a8a8e1-9cdf-4bb7-8c2c-37419b5cee2a

https://lh3.googleusercontent.com/notebooklm/AG60hOodcNId7c0hfDyKv-Sf0ETKwzgJdbDNFS6HTd4e7KOXRVPsKiEHLdP1B_jgp9yhC5MgXvvS0F5N80IPfm-Af9bHzeTDNyNHkUeEj9zOi2ZngdsFvMpouigzwKTvWG0w7M0Qk2a9=w71-h25-v0

65502e64-a661-4f12-b137-de1de7b5b4be

https://lh3.googleusercontent.com/notebooklm/AG60hOr5z2eHxbaRjV2apdpt2o8vI-nWBucEo7QNqk6LeAlYWF4lTepp-oBngvqKYcSoBI6Uiz5i45NZHTtxWWRCCktZNhgtxfHFsboB0jr64yWMgMcz0t1WXtIY6f2O4c5HeauVayGShA=w24-h31-v0

ab8562c0-19d3-4893-9396-eb2fecd1cf63

https://lh3.googleusercontent.com/notebooklm/AG60hOrNmU2qgGpe5cDKHtC4TYnyNycfQJncOxf31QDIRfcu-t4NG3cTzyktDlB9_yYs_B-HwKAY4ctaKZgaZoHvjAJGD3Cts0cr22RkWUxutPoNOuj5_1pmWZFGcy9EFjMX16SrK4gjOw=w22-h28-v0

8960e310-9bdf-463e-a3ca-7ba25489dbd6

https://lh3.googleusercontent.com/notebooklm/AG60hOoc3BzbfLszxMxYv8XofNV2SKcp4Yk3kbZ1HImkPNWqWSI3VsqZArWul0a41VAkVn5FRsZF0xO21DDlW3CepbxWQV_kNAprUhp8jW02271qdqg8LordLXviPvLD7ULEf33YTSN0=w39-h27-v0

5468e653-2f55-47f8-8076-7e83b2ebbf86

https://lh3.googleusercontent.com/notebooklm/AG60hOpCwovfxc7YcBk2qpVMdbcEgCf273OtGHJ6r4CVHicJ95ANPRBYKJIP9uvXgbLsuao2LXuwpUKPk5RXeTjkVNhTlXXdNBsYTvm9U20x0Jo94wA2GJsPMgm8CeBOvWOB2BeSARgz=w37-h25-v0

94dda9bc-4993-4712-ac97-f9ebf3f76c00

https://lh3.googleusercontent.com/notebooklm/AG60hOomZjv4eAQ0OrWeynvgi4UvEGmPSipGc4fDwsV6ejw55kJ5P58SYkEVEGXQvxwNJt-12sVG4NzrwuOU2-kNDmG__PwZFNi0RccriQ2bHl_ynoiKB9VjlHOxlDYTJ_IPvslRWdiV=w47-h27-v0

dbc8ee9f-8d3e-49e5-8cb7-3b929f638e7e

https://lh3.googleusercontent.com/notebooklm/AG60hOoui2S-5usBSXxOBV5hfkXhnqVL3eycwFTyNxlXCNCdY9WXlpXgJzulaG7CPnRdOAw2-OcTWVbWTLOz3jRESvD5-8070olFWfTS_EeVPX_ViVSGVRdttcTcQ_yltdIZ6kGTAbL5KA=w44-h25-v0

2121f543-9196-4d4b-abb0-39aa99def3ff

https://lh3.googleusercontent.com/notebooklm/AG60hOpwCNafBqalai6zYfRwHITP9AKPHRINyonE-vUIYma4NaKxea_VanHuQP_5aSqQOZ94monZGJjheQu0dxLuRgky5mFnlUpCWK5FaF4lPDbM0L2gIkoNgfkQiS8Zr7pKWj9DB5dh-Q=w44-h27-v0

34e1561f-d91e-453c-abc6-fa4ca6d00843

https://lh3.googleusercontent.com/notebooklm/AG60hOo-yUzxpV0ii6PXnmb2n8c3vHJrEhbL2kKYlep-_83slv6Q3Lq7b7XZEqLEVCkn9870gqNQmKGpoSwzRtzAGJzqo1CJb8HvsXXfH9IsWaWmdPwSFPOhCeaJ4wManu6uKtGmUC1Sug=w41-h25-v0

a91c23b1-83b7-4cee-9cdf-44d5cd924546

https://lh3.googleusercontent.com/notebooklm/AG60hOoRp5zk_1V1bY9vEwUWpIMFI-5htd0udBr378GW5Y7sjhUXPZrG8N8pOofq5g-eTasWah_-BCFqESRGASxoCO-9g0UerEFCbCZ1QXkNs4CIgtA8jstr_PY-Z3iKMZIGrW1CKmFA9Q=w55-h27-v0

09ad561e-6664-4a57-91d3-194ebd3ca3a8

https://lh3.googleusercontent.com/notebooklm/AG60hOpNkiEfwiVRJM62aOE1EqhqfP5eFojyLcVcBV3YlkW8bNcE9yFChwcG-_4NlSQwQnET_xOGsKTrk5pPG-SzbF3ujG-zgST1Okd8wrMGzm5ZSLz1iamPLBp_XaWN7Vkd7JKB0TxU=w52-h25-v0

703a45ee-9781-4502-88e0-c32ab3595068

https://lh3.googleusercontent.com/notebooklm/AG60hOrER6q_BkWrn3KvlhfAnxXVtsrcjk4E3k40l0Yfv5UHCsFaxrj3IaTSv75tciepprBrE-EpG24G2rpv1o-wYtY3UAU1z-y5xjXy74z5J1JlG_7Acoia3rLrFc2or5REdeihLR-Tzw=w46-h27-v0

333a1d72-250d-4ebb-9f51-e76028fb82ab

https://lh3.googleusercontent.com/notebooklm/AG60hOq1a1puaGUgmyYYXvY_adhypSO2LhjCM6h4UjlSgGbbeKZdStQ1Ubq6PbW3dltRP0sJPaRW35NsDZMQCuJKOvKauXTMnR7NaDjWN3BAB6GIIlPhBqtXOySQgLMN6IPxCWs6zpv18g=w44-h25-v0

0bbe42fc-5796-49df-bf4d-dcbd302f1459

https://lh3.googleusercontent.com/notebooklm/AG60hOo8KX1Msi51VklMrjMPM18X-0DCF7XO9hF657DSb4Ln_G1NCAGkK8GWiqeETrMjSwZFwzqJsF0NnKjQRZ26THKio6kldP9vP51YtUxS5Hs1mBj5taUJUdJcAKXcFj-6VQJnHIXLTg=w70-h27-v0

d427339d-30e1-434f-8a33-ce22d7b460ea

https://lh3.googleusercontent.com/notebooklm/AG60hOq1wOjS3zEOt3MVXvUJbs_pei4lr-0Ir_IWT_JQZZZoQY8zil3irGymX2H45KL7au8X-oyJZOZ41YncOfVQV0lRIk0tfYDplgiGCF8BqL-DuP8C6RudKltkiDkXOLZH-yj72wkM1g=w68-h25-v0

35592705-51d8-4eae-b231-5635d16147c7

https://lh3.googleusercontent.com/notebooklm/AG60hOppN9zlcJJPViHBhAK_YQ0D3Qaf3gguSWK5ScSAc7lmzw5PjERN1Kd1M-X9ZFYmt4aUIwXA5AeVST40cTuSCLohVgazj2ChYYn3j1jvY00TWY1mXXByTJgtYqxr4PP0OLUgKV_GSw=w39-h27-v0

1a04c00d-cee4-4ff8-8cba-9a508fe93f09

https://lh3.googleusercontent.com/notebooklm/AG60hOq5TjF9iJoEx0o9UEIb0pSuDrrq5ROmXmoqDlp0hLjYEHDcEoPr7D5PzS2VsOBD0wUTQAREn6UnEjUKFwy2Bvtx-52Hkzq4tqX7FMXOaDSiJTjTjBWPL-tDlib9QU7-AIbMuubqAg=w36-h25-v0

57cc1f6e-23ce-4791-aa29-bad129ff949c

https://lh3.googleusercontent.com/notebooklm/AG60hOp7Uvyes9RIOmLDhm9xn8TtT25jCx5CHrkIDzxvqo2dBNp_fox7F8CyTA2j7pDuH56w9ga4Q8PagiyskN0rudU47ULFMM3mPmEOXN7fXOZc2tGADijLhhGSF2TFgBEdw93Kz9YrwA=w56-h27-v0

cb76f309-52cb-4271-a78a-31003afba994

https://lh3.googleusercontent.com/notebooklm/AG60hOpG5LDENq1z0ap8339VxjUZ-sJ2NM7ruxZsfcr5bOI_sq1Awe5FI0SUbyKIh6vanTy0om859Ywm75tr0qZdRS1gHwwaRCCpcleFOOKxRxPkMc-8Lfr4pzj1-btjZdSdtueNQ1BNcg=w54-h25-v0

e61479e4-f934-464e-a923-44acb198a9fe

https://lh3.googleusercontent.com/notebooklm/AG60hOrce_F_5sEhBgDKdcz-X6b1TFu17Kl627ReZQ4BjvR79wp_jIs_4FGXcMJVsS7ipIYwmP8DMd8DmG21OGJ8U3JzjM1vqVSRXj9Hsa8CtUtqSlxuSa5l8smkcYWZciIgIfrgaTe-ng=w49-h27-v0

f6a56cc0-be0d-4d23-aff7-7edc19da84a5

https://lh3.googleusercontent.com/notebooklm/AG60hOrE13AT7ylzfyOzM3LFhLxNE3wO5YoxiMo27vSBgLQhV2RvYEM8d-fzv3X_vTHl0eQ7uSldtSfLUbRH9NwOwEVYBlhMNwMkIWvPq4gjYGwPByql18XvWRdRPky-VWhyJxLmhOV5ng=w47-h25-v0

0a96a70e-b205-4f7d-95be-e283c118540e

https://lh3.googleusercontent.com/notebooklm/AG60hOomgYdfWTqbZLqNQXwHtpzIOPZSIVAI68zHq8UFv0OJIYfmZYeDRAcYz6VZ5Y2igcmhNsZKSWndat3MMfWZGalN1YzI96HHmPMSwarwVW32aHs7kcsqaRyR8IOkhHW3ZvsQxy58Ew=w32-h27-v0

8d71335f-c436-4b30-bc90-5848e9497e23

https://lh3.googleusercontent.com/notebooklm/AG60hOoo0xGLPs8bG0mizg-71sIa5lT_kPlPI2mmT_SncNDSUHMknVzRb5QEHuIkECR7pw38cRPPTz8T2FvSiExLD8IqvZeCdEwN1V3bF5JJHiUBXkD9tOIrei_Of9OIdNHXyeuBYHfm=w29-h25-v0

8177cd10-0718-46be-970a-dad990dd3e3c

https://lh3.googleusercontent.com/notebooklm/AG60hOrVVtp5bLGq6CX5GBBrbmMznLjQdfWOdZIG2FDl3oV2YS39lSW1nW7unl_7xvkIpWtxB12RHg8OWgPaVKKz7-eB92PlRP_33dWRy9nJ8EwRw41i97Gl5GAj3-wwBDBcYI3hUW1y2g=w79-h27-v0

51aea8a3-eaf3-46f0-bef2-fd43fd8d813d

https://lh3.googleusercontent.com/notebooklm/AG60hOrwtQmBx6GOqxkQ_yG1rCzt18WvHgFDgVHez-auWAi8E6XKwswyD7O2471HW2yhEFeZO7KoRm-XRrkHdVuizbZ4hZSW6wnnHeJP6GEr4fcZKDQOA7cAOThfIE42F2mL4FhOTei1Aw=w77-h25-v0

d3c07d2b-cf62-4aea-accc-bdf2b7f5539f

https://lh3.googleusercontent.com/notebooklm/AG60hOpy7p5yEtOZyOzf7DxXqpIT8cTGE8OQbkcMEDuvd3xK4tfYQrLKhLlT1K6zBKTJYN10bKxhDsnRfVLv1WohXEBaAXPsGTLL7rEK10h3fW3uG4SclLORWKl4rX7T9beSBCT3DRzw=w56-h27-v0

9010b5f8-35c0-4d3e-b3c4-a5125f34cd7d

https://lh3.googleusercontent.com/notebooklm/AG60hOpiv0qhkM1igwqsSlFcP8KqN04TJpV-bmYBzKLKG7kSeoLPfoXgcg_4eGHvb49PNwOdOlvfK7llqH-qKQeomQPVTcK9NjhcG3iQQrwauwTgoojzSop16Gia21yknezNeSrbMpup=w53-h25-v0

b9714c42-d4b2-4f07-b8e9-6fa7efef6f25

https://lh3.googleusercontent.com/notebooklm/AG60hOp2Gp7RPFvrK2xlAjaQosmzmiI91mKqLx9gsEQkaKodLfrQL3yDx1qj8mD_Au39FnOzeXcWEZQu9P0mcIGYhERt29mhH_pAVkDMJamWfMg3lnxb6FWAKwkeepQzZ0eozsMy7irhvQ=w24-h31-v0

c4585dc3-90af-4a94-bd0e-153853c6fe36

https://lh3.googleusercontent.com/notebooklm/AG60hOo_PfilcHwS9Ft3C0M5_Ce0DqZqZaMKEMnC-eboD2FEQjBJi5sAUV_OOLCGcSaijtU8DWidtV_FLEJpKGSR5iA0xfs1ulkOY2Hy_K0IkBgiU4yTdns2ndEzjnfga93EWd9cZan6rg=w22-h28-v0

40af2419-5216-49a6-915f-6b640e753a4d

https://lh3.googleusercontent.com/notebooklm/AG60hOpXJpeagJ7amSlu37nhz1rMx6LWvpUd1vd9nzQ8t-7FrUZr1K3burjkGyfAz9DNmRpAiQdK73iai9S5V3o6rYlmj46Vv_ymK-4ydoyAsxOk7iMyTPKeqNthpugJ66qbvumuYvXILw=w32-h32-v0

15454637-ca38-4f91-aad6-33acdef2663f

https://lh3.googleusercontent.com/notebooklm/AG60hOrXrVMBcfXa0YVTA9mbPhn4G6cU3cad3qQtDjAL0PBl1Fz7DIWRH7pCzBvX661JkgYhQZasjVJKhd0pbwrwR8F5gEE05LZ03OqwKtsH_pAOdO6DcLLgxEGVgsTkVpqBsXu5b78amw=w30-h30-v0

4aa9200a-0bc8-4471-b405-0258e2063c71

https://lh3.googleusercontent.com/notebooklm/AG60hOrNDDbIAIKyvdb9qigi1cGITWubOi-YNFBstS-NHbw1yOe14co5F5UBazykBFmB-lNg7yqG2FxmWd2f1Ttq-D9syVG4uUEOnuNaOlfdFxPu8IiR0SA18m4OfypCb9wS1DY4dn3r=w42-h27-v0

a1fd25cf-63d6-4e6d-917a-cc9549234e4c

https://lh3.googleusercontent.com/notebooklm/AG60hOqjJxjxGXDJehrxiurrcpaNCR8NjTdKKhuch1yV-dqwf3k_qRJKcK2g66bc3AEDpIsjryzm46NIdohZuALkxIsJZZdX0H16S2gYKNFr4pGhiOhvRZOU0LCNdF_uUfckv8FxIvZu=w40-h25-v0

7472b3ea-df11-4860-95c7-034f3927472e

https://lh3.googleusercontent.com/notebooklm/AG60hOp5gWFxerQxWwI-iPYl4T8AUP8CfcK0dsEDFeZzf_bh3HCFKXpn6u5OoOOh6mugYRcDnHeDs8KUUAeEIrDuEMcBaJQ1geL-bQj3fhmZbsgzwu5b4ZWSP6uszOYDrI9c3I1-Rib1uQ=w27-h32-v0

e940e898-6d49-4a51-8170-74e79006e473

https://lh3.googleusercontent.com/notebooklm/AG60hOrumYNRTh7ZsQwcRcNgfT4L02v8DBIjoTt3jRXbjBYAAHW6rv8BwjRqXYcFF0YAVp0DFFIhdre43h97qHfPOiuuZdqt3IH6B-OGXMYwidqTuyQBIdw3lTmfe4x6HM5zme7Bjpu0=w25-h30-v0

a1cb79bc-0346-4b5d-892a-acc88753a265

https://lh3.googleusercontent.com/notebooklm/AG60hOqifsdiSdUS8wC7Xr1KATE_lc026Bl-MxLtjDR0lG5sXcSC49zuUh2AGLLoUNxd2YtdSGgh_ScHjpgFX_l3nyGkNWDMgYW1m3kukNK5HCnuSOcm7UGSWdSorjJBcOF71hlG-jV2sA=w55-h27-v0

d50a2885-49c8-463c-b913-fb708c12c423

https://lh3.googleusercontent.com/notebooklm/AG60hOoyh-VizXLHzmz2y3f45-aTXrFs25IQXuc2-2n69nu0py71arOAwBRKDSDMzetFUOqFLwS7zCR7VGCszlPbFCGkKVmky4rr7arbXfMeifG-sSzEPQx-3eRlxiJHqTNxrqHNlbwr=w53-h25-v0

bc35b848-0937-48ba-9ce0-03a3ea151559

https://lh3.googleusercontent.com/notebooklm/AG60hOpwhRAhyYnQeKeaNKp1WMDKNfYY8LeA4G0hqBXJEpuDY_sKS8J3qacg_ZC9VrhTf5vrdL6ggTHsaKP0ZmR2MI2WSeIk6hrcAmXaZ_fkgN7QHCozXn2_zixc-Phy7fZEE3QIJ1gulg=w43-h32-v0

79df0ba1-9afe-452c-93b0-51aba35a3290

https://lh3.googleusercontent.com/notebooklm/AG60hOpmPFvQxxfjfOmEQkRGhj7prqzWwVxlRvKpiBaKlXG5IfTEoKKQXmuqgVu_wbKIP3383IgJPIQQoaVxvgnGgKuqdF-ZxD2dPA4alnCvJcmnm-Ce2WJiqlQoJ-nGElsQTgXNIBBs2w=w40-h30-v0

bfb6cace-fa6f-4678-b009-275c18af04c5

https://lh3.googleusercontent.com/notebooklm/AG60hOpH1rprhvRxN48ekt00PSsqgsPUvLqmUz5VJ_xrCplN1Z2_goHKK34dqdwtuCX5eR5fLDv6OS-oYIn6QL-0C25IG5WyJpNY3cnUiWnbzeG4h4FQhlMDwWAd_UPZHJoBezbU0rdMwg=w71-h27-v0

df9b1a44-0a15-4a5d-b764-28ff337e7c17

https://lh3.googleusercontent.com/notebooklm/AG60hOqekHteeQ5krF0SBTIk5Bk6iP4zfBGMK8EXmE92oBQzMOC25kDNsbm5mmTZb7U2MGM0svT5ghPbdl7XqXW84J_DcfVvrJyzInzpXcoNDz6rECzcWX-WChhVJmAnmqzQXVYrPA10MA=w68-h25-v0

26a85396-45e5-4c4b-b528-d5499916dd8b

https://lh3.googleusercontent.com/notebooklm/AG60hOrBIjpSj0MgwNhVQ58HQucpMHXswZA4q_9sFmVL4thRexRWUfTRqBQqqgvuYF90wOLTALE84Sn36mC2Xh-1b9nys-JT_8DIIPyBoaybyr9h7NbmlYqCwK8bUUIHqbRR72G7rSFq1A=w29-h32-v0

301bc705-2b1a-47d9-8274-a759faae73cc

https://lh3.googleusercontent.com/notebooklm/AG60hOrq5aMea7bJB2cVc4D-qtSgcAGoKs4lpiaRpgk0y6PgEIB_ILrqpzYWfNedWDUaoNoBJSxPQgNuSLKQ0wh_22fdh1WSULrZ6Hesj1WV0j0aZw0SY8vOdjbkXKHh5py6iwUo2hxTBg=w27-h30-v0

7ccc2aa6-fd67-4fef-8c62-7d07e40aa00b

https://lh3.googleusercontent.com/notebooklm/AG60hOrmmRNf9rHeLZAeBf9NH4jawDRrNrHHPHhaxIQXvUNORpwvilUtVcoVTau2sk-eE26c_JAbwCuacSAtSKnmePacvTgkfxGlZzrwxcu9PCNWlr3-xGEi5s-KIkyUx0Grx_9LsSHHmA=w64-h27-v0

f65c80dd-5e07-4169-922d-a7b71dd3622a

https://lh3.googleusercontent.com/notebooklm/AG60hOrIamgTahyp3D5eAM2LVsuhYynxjl_0lKpyDYktEnuLeyAc9CKSWZES5X-yovlqWxUJUk9Sl04qhEYp5BOGRR3LfKN6ouCjfHRSTaFHI-RdfHyZYRhKni5PhhKbNcY-tSaH1zYl-A=w61-h25-v0

64d9080a-49f0-4555-8cc1-aa55764ea5e5

https://lh3.googleusercontent.com/notebooklm/AG60hOqBraXS5voTzIGJRE61DnACE9kP7VQFj7R7Qzo9LLH2IElj2YDNfvEEyLoqkthSuv27Seu0Fhw_ozzCIXcp2dD0BE7kaeG24maf2-psu-m3sHG_LHXcAuqCnitDub8F41iv9HVtSg=w24-h31-v0

78afed85-aed9-4e2e-9fb5-9c263b6ff93e

https://lh3.googleusercontent.com/notebooklm/AG60hOpygn-k7j1cTJc5kFflRs3L1wUZi8JcmXUPYX7RN3YqKeZMxkOiQGuzAmz2f0Z-V96i7Va03gK7NAPpg_TfMTPFfl2si5CQfuQTUJ_livwr_mIoGh3q30BC1IZWsTqsR7W8mhTjXw=w22-h28-v0

1fed4c07-6159-48fd-942c-8e46f2430da8

https://lh3.googleusercontent.com/notebooklm/AG60hOqKV4lTKz0-u9CtWoZVyrlRaFooioUXExJYgsuwKXM9DGycZE1lNuEDOmb-eLTe7oJlI_v2o5MNM88sQJQ-GoNNsp6yMhGbHoXS_BXXfmQH1yXUBpjruaJuig_al8wyXZsqcTJY=w30-h32-v0

4791abaf-4ba8-4738-976b-1672dce1685c

https://lh3.googleusercontent.com/notebooklm/AG60hOq3TuIGpzkT_yIfAr4zRHNYPUXU7dCqCjcRMTbdPDNavXgMPNUJmAO0Yr0R9xg7I594Z_ajJ9alTRBgzdkEvN1ZAdeDqUi9nb_GIj3redbOCC0BbeoHstflegYVVR75ReVpuvx-MQ=w28-h30-v0

19b925ec-ad05-43b3-a6de-7f9aa6b7e211

https://lh3.googleusercontent.com/notebooklm/AG60hOrtzry5bg8sTeqILRpcDIfv6XQoESo5CgtomEfCt6LpMnMTLjVrQjVovBYV-3xKPJ8rMbHvX43q3f588RJQ0DT_-t65z6cyTn01d4V35hdwI_eeHf3OxCdvZ4XOiuzINf9CESzhTw=w60-h27-v0

1e1cd95f-08bc-4a37-b543-4af99b320b43

https://lh3.googleusercontent.com/notebooklm/AG60hOoVFpCmu_gtMyhF_l8feuB_bX7dTPW9vTqxH8HyWaatbo2s_zgVeZf02pG2YK7ooypKcuzsUjauq4DLrTU7uBDvg7xnqYiRyqNWiOAFdvDkYRJxeS9mZwTdf9fUPcArGYc31orY=w57-h25-v0

966e910c-48b7-4b17-87b6-f415d9155125

https://lh3.googleusercontent.com/notebooklm/AG60hOoRWkK0EUvAPQ7dIq2WY43V6MM96djUEgu308w0VCuBm27cK2P6WX_LUdQgIViegLSNzRiGs2-cHxY7_-2NIjRCu6R7gYCz36ARd9-8KkT58hrxREtn1MqmwqyKOU6VQFw_tYTo=w45-h27-v0

301e3f4b-4f24-43b1-9615-7fc503338259

https://lh3.googleusercontent.com/notebooklm/AG60hOpqwkw0683sVSH0SfTL-fsMxvJQb7fTcQaHLH9e0mo2UsuMVWxr3FdA-FD2G54j-J7gm0h_zawA0HDgwIrwkiFq1_3VSNtXy0ytUJN2qGSaMKrbnMr5iJccdKbGsIP6ZSUYfV8V=w43-h25-v0

230127f2-bbaa-49b9-b819-1e01a417aa1c

https://lh3.googleusercontent.com/notebooklm/AG60hOpqTE2l-W7OW38cGLmX1rpw1SYbAMrL2KtPJSYjQnYUui5BziD4VEtC5oaDuqDrwM5FvA_U_AV0Cu-F9Nujv332PJni_6FrlH4qbDCSPW84OexndjJzd219aELw5XVX68Dfh2gItg=w57-h27-v0

8b8f0c8a-386f-4f2c-8441-d2993ee1859f

https://lh3.googleusercontent.com/notebooklm/AG60hOqWcpykpdTx9Fc0E2xbL7FWizmx6ERHU8hfSSdMAyYeV2aq0-N9ElpmnrDl3V6eTuTeoT7ioC1-ePiyE7LRdwiup0NQ8QF_KTAAQ6meEcmU_kl1hLeOo0viqr9FtwZRpjtYVk-C5A=w54-h25-v0

8633bf99-5728-4dd7-b583-54f6a1df2641

https://lh3.googleusercontent.com/notebooklm/AG60hOrT3-qIwROd9JB0KufxVAYcsQyz35Pca9YTxx6wyaHCGtEjs5dQY0GhA8BnH0jgz3c5Lyc9cKfFgPNb5S1J7eFtZOyPfBF9S_b_bwROsaTfibKcqXjN-ln8IMGwmnKQL5qsuCACGw=w61-h27-v0

fb0c9139-7769-4f65-b6f1-dbaec10e3b08

https://lh3.googleusercontent.com/notebooklm/AG60hOpekGD0V2G2LPx7I3ONAcEokXo27y2VZ169qt3OmM67DS6YLn-HqvgPJTnNnblaLIa1hgI2LoMXY6veGTsQxfNMHTD1XSOlS0d4x9qp8sLiledzpga2rHcxwKd9H-Z1bF0T8Gn-1Q=w59-h25-v0

d4b95d09-6bbd-4373-a50b-2972b00de8df

https://lh3.googleusercontent.com/notebooklm/AG60hOr8S-3fl1TFthUhvIikJXUAeibsWYyjP1a_68OZlrw6hiKZAc7Ey9PZeqaYF42BT3lbZXGApHdj7x1dHJDXKR545rW8Sp-P3Zn4XAxFWBIrdOHPvYRak1zlmG3lVZjGDnsCxt48kw=w24-h31-v0

cdf31b78-73cb-4bfc-b25c-e1f81333e68b

https://lh3.googleusercontent.com/notebooklm/AG60hOp370pUJ9cKag2kOkiuEkLAJFbHWdzuBeYnGa2VGnX_AnTD5gXcNqWDDbiKaUhTg64LTS6li4O7FgV7LxSl7XbsFpNk6U0Lh3RQWbJ34_LIsWXszFYCAXUmVbgurW0xsx5Hzi2FvQ=w22-h28-v0

fd63989c-62e6-4340-8b62-bab79790585f

https://lh3.googleusercontent.com/notebooklm/AG60hOoRw2homWdKpga8n8Eg4p66M6n_n4ICKyURWWzjd8pcCXg9NR-iGx6ZMOQ9upOO8x6AhHYKPCitQBo_lHaa7Yui3Ey1vC2hSWrh3AgYRK9DAPg6wlUzhywP8VxzZGG4m_xy6WID=w29-h32-v0

daf57fc3-f289-413b-88a7-797bbfd4104e

https://lh3.googleusercontent.com/notebooklm/AG60hOoJQnO7nSJHtBO4WVrRTZu9ouPkVBTDqoDnk8p88URmRxi0OtwuBQFTz2uwxUweSpTc-W-VBzjjtvXKgcDuMohxFIxO3XQMwYogsAstJHMclvM9_Y6EbahsQhGUZ5aIMnQQPt8C=w27-h30-v0

a3991ac6-ae57-46bb-8073-25f02f6d9d1a

https://lh3.googleusercontent.com/notebooklm/AG60hOoBxOgXdvV_1eC37e0k97Bw25O7yhLwSIeT8tIia-8uqINAQ8EccZSYSWXMqmrm1F0neDKLIti-pOy_k0wGBxJkvtK5OsDvuik1ECSLyXfKrB3B-5Ezb3T3isdZWu5_smexGNiHNQ=w46-h27-v0

160d8f76-b3df-4db0-837f-8306933d8af0

https://lh3.googleusercontent.com/notebooklm/AG60hOq7IT_ZY41g504PqdOJLlDdPEKSEtj0AEpFsalUVq9eXYPvguMC0hH0B33z5UZnQumQoxZx-lRPO2qwZVh8yieFsPhJmktpdxuVRBMCXMfcLZwgbHX0oLtrFCCUGLaKNHMJ4an0=w44-h25-v0

9f041107-91ed-48c7-a48c-d108e0d98079

https://lh3.googleusercontent.com/notebooklm/AG60hOpEqUJ9VYJSN5dba3b1QDgalPvgGp7DkMLF-4y_TeF02DsxeIsQiLTZTrBTCRZpX7dZuBnJAnmD62q-axtJ-slPXpvImi54LvSFQITqck0St50OqCkFPH98sf3hpBB5QjEORL20=w39-h27-v0

04c0f915-77d9-4c6a-97c5-9c21af5993b5

https://lh3.googleusercontent.com/notebooklm/AG60hOqM6zwabWg2kJ4Z_CL_n5IBz9rGJbf46ncoqIhMKZP44BMIoaiW-m5_vsnIPc8qOFDB7Jcc8YEHXVxo0exMer05pvPpUE40WFBhW1UVqpYT3JngiX2LFaxGrE3anQrr-oGG1QNB=w36-h25-v0

bbad6cf8-e17e-4ae7-90b3-1181372e9c3c

https://lh3.googleusercontent.com/notebooklm/AG60hOoxX43FVGL6ZhUIByzVVrJCr_Ak70_I0n6l-G5lgBjnYaM2G6eWEKJ12NzHkKbGmf6LjI64eWbPUGMmaC3YNXULuIBjIJtNXKtQs-l07gb72kUhvQha1NOtbKXTa5A7WMSORZ88=w33-h27-v0

df622394-3f7f-4dc6-b432-014177ee29aa

https://lh3.googleusercontent.com/notebooklm/AG60hOppH1tdPJLOdNBpKf05q9QOtscFcTLW3Dgc15lRUAhurcguBkEl9FlOz_MNtD2_xzLvYndoD8NQEiSk8PTXbRsS7UWnsqLEx6sbAddLAZzMMlfZiYpm1yAo2ew9C-K2Pvrjnm1MQA=w31-h25-v0

a734dcb9-a1a3-456c-995b-cedfc91ba91a

https://lh3.googleusercontent.com/notebooklm/AG60hOppLbMEdt9oqgL46qpeOcTj51xSVYNgMQFToSyYFQP-LvkgplZMkA2q7ZTgk-PUab3O50U8b1sOGEqq14Y18t397MuJfVimcVZUO1BWLSOJgP0neBiu6XPSBHxlYYXYiB8ECtW-=w28-h30-v0

c784ae7c-f440-4e28-92c7-7482f9e8136e

https://lh3.googleusercontent.com/notebooklm/AG60hOpeJaLwjvoVZNSwWw38GHHu0dTERL2Ab7h-a2YjZE_zHB28vdepuGOh1LeKOFD1rdvGfjkGJN-qQ0QbtX-wwLAgFoh3-Mg6EGBYmkVPxjiSEc4Y6X5_7dRcu_8519nVV5dvXIvs_A=w25-h27-v0

256454a8-4942-4650-ada5-afd3b6bd901a

https://lh3.googleusercontent.com/notebooklm/AG60hOqtMENhp7_sV_bVqFV_bdChaOr6z7ND2DDJx4gmiGrs5w_rdKKdDhA5058vzQWxX2R7u3JkikjELQxj6euMOeMTa1yD-4LEACfbmkGqTRxXNBaSiqWQRJE9TCSLk2Y-2VqMSwF0sw=w36-h25-v0

4f329e12-d5dc-4e93-9d86-736909251fe7

https://lh3.googleusercontent.com/notebooklm/AG60hOqS7OOPGrMKm-Y3A6XXQbq9wky8VDKoCJycOZxfowe5rFmWeivrU6HrxnBylAOGQrfHfNPduyZC6Vwg2QKQMxn-r1Z7fpf_VJ-4B5mmaVO6jGmJz654iQsxv6Zf-WJV50EUdiWH=w33-h22-v0

d28f7883-74af-41d0-9539-16e258079c5a

https://lh3.googleusercontent.com/notebooklm/AG60hOpUdDk3QoK_VSpcsKiVBAFScLuVoav6oJl4jZIn6Zqb4UdE0ZDfJi5Oy-xEB0Ia1ngBQcR6uMcUl26x5s9x53NS2MqU_UMh2HIs-pU_8DuOXQGb_ynCqD2nGEzNHDQzwAwKv3hZ=w62-h25-v0

c7522cf3-f24a-4e39-be90-4683d484071f

https://lh3.googleusercontent.com/notebooklm/AG60hOo5MMPeNXAi2Ra2uj1eCiox_qSJtNo-VSCpp2t0hgnIo3mxYInTJfyH7fT447hXbXY8iqRRpgJyFS30yU_CB6BT38nJYXl_ivek4AEHfBSK88IR4TEodGv-8kSE9EhKXeXngkATqQ=w60-h22-v0

c7c895c8-db94-4d98-8639-f14956d29f67

https://lh3.googleusercontent.com/notebooklm/AG60hOowlApaqFbEEJFhHR0hJRa084qEgBf4r1TIxtMhB5iUFm9VuScZ6A-kiAZk2EvBwDhDRDcLAQMivACLu1CedQ96Mxkrc9JLXzeDVJnYMZV-P1wT7U6NpTMSGqrIncI9QsPvZzUgBg=w40-h25-v0

3a6ce014-1932-4436-bc93-e29a2c99c43d

https://lh3.googleusercontent.com/notebooklm/AG60hOp5VXEnMEyeg2LKs8aX9CiY3Cjsqc48vJXw0JM2H4YxkgXSUIEuUWrU6iMtWS3RkGx7SB4GHyuQj4xINPX4Cy-xK4U2qOxkOQscu7k1UB_7qS7lACbPCo2lrm_ObgMGSfQ7Azi4eg=w38-h22-v0

3cf8d189-8dd7-49c4-810f-a250a220ce28

https://lh3.googleusercontent.com/notebooklm/AG60hOpOKg_dbBCZAPF5vcSw3RxAJFCc4K2-GO8Z_OgjnZJrbGZF3kV4Xmsh12z07Oo2tSJ0QkVx1E75ZFp9eo608wJhTWEocpRki23CN_zvgG4MV6N1wsXvGOUXATA_CIxiwLzBqFEWhw=w37-h31-v0

d2499dff-e3fb-45a8-9288-2417a477130f

https://lh3.googleusercontent.com/notebooklm/AG60hOoCvptqFLbJSyWIWmF4McyjBqSSit7woe-TsHIGOq53JrrPhgOmj0fpZvfOFC6aILZUvi1bwlmFA-N0XOfiaSZgeGvHPLqdVdje7lWG9IiTTFj7vpIhmDBQvgluzPea3RQMnpiODw=w115-h94-v0

12c9506a-75cb-4e42-b4c9-fea8958ed608

https://lh3.googleusercontent.com/notebooklm/AG60hOqrDHPf62lvC1dCoiu7A63eG8M8WbMGlnixmm_yzraKjdld3WscJQH1GvxDUAYG5OKI9KQ8Qye3e2aNW26o_-FA6Vbs5A36p9G4QTF2ZOp4JRXZMcJ_Y7Be0Dzpr3zk5-IuvGhDLg=w67-h40-v0

b440e688-180c-49c9-8b1e-1cb68774e2aa

https://lh3.googleusercontent.com/notebooklm/AG60hOrpd01dnjoj15T33W6PDoxjkdk5XnKJGYwdgxTkswOsT3PbRe1NFyhxRp-yOLD20K4RolKhXUveoQoXkdrBaTcKwgxL1Sx-iGrUL1u_7kB1-YYW0uqGNwSFjIOEbkMemDe4ZQGRiA=w237-h127-v0

acd935d0-15cf-4027-8bc1-0aa79a0bb4c1

https://lh3.googleusercontent.com/notebooklm/AG60hOoSmJR_9O995Qu1i_HfUI7zE6y-ofFSoqCDjC5xsxt33jkjtJKiTzbc-xGkyj149HuxqHdghSBUZQUN0zm7CGBwFxmD9jOprmuy3kK7ShWq9mDpeeyQXI0QG2e78PNIdAIf3E0kpg=w452-h67-v0

da0aa3a6-f202-4b76-a552-c2284a3ee81f

https://lh3.googleusercontent.com/notebooklm/AG60hOo71udnGpzSI81emgJ2Y6CMqB7MWQeJhW8Wky9cz93rfWzr9LXHo6IPdv3sNY4xhVe6IVTd8J0JchyGcneyN6HXofvI9Nq_3b7kmtvtzwENF0i7mY4ttJK770H-7CVDekq3KhUV=w116-h28-v0

e4a726c6-f727-4fa6-ad4d-8f9065a18ab3

https://lh3.googleusercontent.com/notebooklm/AG60hOoIlLCDOJVGYP9ymTMeXIvc-OEgzCLmiTaBz_kcXkBTDYlJ8crndaPNrmdH_YskKOsOd2U8xK2Drei1n7cdknG2h0JdwtGB2Lw8bPIXslsgPZLzoq1zKaMTmW7sPa9Mxj9tWrTE=w24-h31-v0

96cffec0-b220-42aa-ae99-e854f3c7eb5e

https://lh3.googleusercontent.com/notebooklm/AG60hOomlt5BFrqHcCsXFX19iWegoXOsi4BYu21s0H5gj6bDLQTZkohEU8Wwg0C-8gUA169KvOWpmQrIjGC68RAZ-1IgggJ3a4hHWEH0x1YTfTvuHlG-Wzt9TChG0_C3TG5DCEEYUOJ-dw=w22-h28-v0

410ef75e-913b-4c4f-807c-56b2c4630c6e

https://lh3.googleusercontent.com/notebooklm/AG60hOpf0IRF2M_vpQJNWmlckvo4Mp1ZmB4en9Lks_UHWhaSjPf_HY3xwxb5Bn5DV7QCFolZ0DtkwwDkL5pzrtK1jXB_MTR6vuIIsGoHERE81OG2TrEOUz1VlYbLGUmBDPBlUOOUuWPp=w45-h32-v0

af72f236-7c52-4b07-8b94-b847edd3451e

https://lh3.googleusercontent.com/notebooklm/AG60hOqjuH-8_yEd1kwnV3wJAl1WhFl_88mm_lo0AO7YB13g4BX5HIrAjU01ZdT1wb7f7pEmbu_BnwJTK2CsT8h2bBfDKhIqfun33WNmo4xYA2lWEE1i2cNkShDu94CcsFt0IkG3IDTzYQ=w43-h30-v0

f8ee2ee8-39f0-40ad-aaed-9c4e7eada805

https://lh3.googleusercontent.com/notebooklm/AG60hOpVJ3BTmk7rNwvZWeqsERPffexqb7_q72-nIdBaxfeMJIXHlZX49Hi7SjZkkXr_kqznhs9xeT5kXD7FV18IAOPA2E1P6fWLe5jCHYnOEPfPRxKvGM7SzFB1B0UWvNu_2YgYkkEz5g=w68-h27-v0

8b90a499-71db-42fb-9176-e448f2fd2b66

https://lh3.googleusercontent.com/notebooklm/AG60hOrHcyIlycCm22VAFlbjqkqGY7Q82qKtHvOQI4Sz9wioD_MYOpUjt8yTqvUKSneB9aLtwP2onUV75XdUQNANx7-noZUSywDBdL3NGwHzFmixpI8ZfHkpiDXE71wWuGoMpt38Ma8=w66-h25-v0

aed643d6-bc26-48a5-b94b-6d42e0496e90

https://lh3.googleusercontent.com/notebooklm/AG60hOqR-8h5iDtOnwpbGpnpmXQ0bo-aDqPHjfl686UiFa3bojYkdXSEVL3HTW17IHFXUNl-Dq7YMxZOYO9JQcc1ZNi09qkAg-CBMmqnIwQ3HArSqPnSG6k5ilS6cgRFPfQYQXYfIHpjBw=w89-h27-v0

71cbf348-60e6-4e61-b134-acabe15cb9d6

https://lh3.googleusercontent.com/notebooklm/AG60hOpkHKLIQjqccio6SoO0bhDTTt58IDIqOmQKkHQgBGkS5U7p7miobD8DPPqENkiJ__wjfGKI5glDocgd1tdQKJehDaOu6e-0z2ttJ5kaKpoEWqUdea9xMZZb2HvAyQJOkPAgsQY=w86-h25-v0

9cbf858c-7997-4223-b34b-98051d9d7328

https://lh3.googleusercontent.com/notebooklm/AG60hOo_GeD50w6U1vgFzIaOjxglD5fY7o7ujZpgihsYWmTBlc8hlWKq8VQMKBgolxjEfTEfM3AmSlVg8njCwWysuuMNacWho8KxT0CFmHpXfcV_UseZTAUbWSYjwN-jL7C7DwiccbTphw=w109-h27-v0

152cab3f-b9c7-41b4-9129-467e8a276c8e

https://lh3.googleusercontent.com/notebooklm/AG60hOq3dnmZWAX99V4iwLwYDvNed3uc6D64Zj6GMhjiLiNfa6yTYxEQwcEhtdHizYSj0cuseAKzWOXHYbnRZgEInDrk3cOHwjwxScchGl3pM1kcGzhmfB5-WvGuovjpkXFQHZyW5MYQ=w107-h25-v0

5771ca35-de81-42d2-8bca-09f9e2a3ef0b

https://lh3.googleusercontent.com/notebooklm/AG60hOo_GbWW9Bnz5G1DlHyqCAXx6wE9ffk79_hdDcRNziXNpy-mV4jhJTH2p5vWXf-HX05QhnUHNJctKlwo_r9tgbSNsFmEIGcsYksyJyfMPSmhoDcULLYmDWgmC2Zu79b_yHH9_Nzp=w45-h27-v0

2fb0a5bb-55a8-4f1e-b9a7-7da6b5f83128

https://lh3.googleusercontent.com/notebooklm/AG60hOrVkQMb0oIGdo7-oS51squfa3Dapdxnzsg1a3U6AEUZwmhspEhsF3xmGUF2mGIMq2ew4pfrk-EnI9K0WTFFj90k2mRpPL47NVlq5DUjowVmlQGQ9PoiLDo_8fUfyCygeUNF9nEj=w43-h25-v0

5f4f4ecf-cd7c-4645-92e8-6aa3b87bf6ff

https://lh3.googleusercontent.com/notebooklm/AG60hOq52MDFjFuld7IUUwXvuOc4PKwsULw-Rx7yUr5E06cb49_qM6qSn-yKosyyQ-U5jNSSATjHxXLdSiuhDFWfJxMTDKhvqq4qTTqCBymp_q0SNHwEBXUPcdE2-Eyv6VNFKYNDXOX__Q=w39-h27-v0

6a908485-4256-4025-87ea-b8a57302532a

https://lh3.googleusercontent.com/notebooklm/AG60hOrtZAivKRCTG2zGUp8FQzOBFHYquoU0L1eicq8NoZk8xMeqfG1PbaDUsHo99ycgytN_ejrPYWJlaCzET_09wunPzBRaIDUf2jQ5RddQmIAC-l5dStptC6nusY1YgLDHW34WDu0LYw=w36-h25-v0

1a1d0344-fd88-4aef-8701-6e8692d6dbc7

https://lh3.googleusercontent.com/notebooklm/AG60hOr0aMiI_dNGavpOVZhl2lvJO4vUVjPiEFmXuh1msYzpuhD20gRSxZrnUfmIdLlQB-YnZDeyuuaxfXbNKt5N-SLl1uiqOn0K-wllvvCU3G03BE3P8Jr8F-aJfytYp_RbCStTmNis=w65-h27-v0

2c154cf2-e5de-4545-8113-41b9d5b8ec60

https://lh3.googleusercontent.com/notebooklm/AG60hOoBSKNPAiVA4BXknm7M33l2_xVa4jCEnzy2bKtDxHShMK9U3VFR5og3lqV0keIaj9bcO77tIzqViNBoYd0-OYKuSY4hX8vsbG5b1pYDzWKr-qUjPxuyFUpEMCFEfb6BBuPxGKnojw=w63-h25-v0

31084be0-6415-4068-9f96-dd9ddc83adc7

https://lh3.googleusercontent.com/notebooklm/AG60hOpynlPUrzkPMjUwNKgh6QNq-Vfxtez3O_AucoZv2kN1ig86c4P-ZIjjgx_QvlfvcXj9lhVu_5sj_d__HwDIXnFHM8KDgbSajssqAxudMZM4V83QUbNRTX7npG4QQKTosKivqzteew=w24-h31-v0

e4830529-9524-428d-9533-9de3bd565cd1

https://lh3.googleusercontent.com/notebooklm/AG60hOrflM17r-ZJnXXcj2oaC0PXWg3e7ZQWl9c8oV0CpcJzZ-Ob4Q7ERMn71V-WG9mfVP4rp3y1gLt2XigPH5ZU7f_vvyQcAx2bUr7tL48O_Y739AyMbwIEjM-VDs7GZGMxOjqpDQcRxg=w22-h28-v0

68780184-18c8-4c9a-b103-56212880bbe8

https://lh3.googleusercontent.com/notebooklm/AG60hOp1o1bj8BIVb-SWex3F6On5Tlz9MgCrykgEvQVJFopOYup8QHCjL1GPp4OpQdq844loaGRt0ERyjM0NSdS9HdUty6ZReZLcNOtOetRO1Ezc-86Wm5rfptPT0nzFqR8azgY9Aeph=w29-h32-v0

ec8350e3-3745-4d41-8739-4a76e1f62a7d

https://lh3.googleusercontent.com/notebooklm/AG60hOpTswXThRwCwj5vFwRLfN2AjV1VibQWTKaXk1C1SVewv-zwtYlkMqpMzFnDXwNeoFIYFUF_iEtS1qWFH70-L76uyx0NxywdIjpUDi3L9JbXQR2O0m2-oektRZHByx8IIF8BX-w_hw=w27-h30-v0

79897690-3639-4ed5-9952-668f68166696

https://lh3.googleusercontent.com/notebooklm/AG60hOqKttWSNLe-sjHUuTgpyMJnhCFWESSl1YzBsACmHFzsPpzZBCi-NeAwlQC9TUjd6__2nCCDUxWu-9uV2LbqV7XWhuIlBfGqQF6084RnqSiUt_t4m__SpUJXdD86CMLVZ35MhP9IAA=w32-h27-v0

b4c4b5f0-4f01-4819-8afa-99dea5bd770b

https://lh3.googleusercontent.com/notebooklm/AG60hOq7hcShk6BSTWHSe9A8Ma7Z0FDNFtR5prCLwxFIdjkvcYeti7GoEUHq1ySaIGh3_dZ_jNud5Wn4xCl-YZ-RqTERxkvSHJINv492L_85PYGd0PWC_jm_KXb3HqLEOJCgShrbBsU-=w30-h25-v0

e4d6ac39-e805-4e73-b40d-ea9c74d06278

https://lh3.googleusercontent.com/notebooklm/AG60hOqPBcVOW8jR3lfDAV2Pq2vPluGgpa25rnOQQY7LuWM3wvTZIaeX__hW-xNV3iWNlk_w6a6LDvl1V-REzfFrHfOeVT8XXpFYGK8pm9qPx-Wq2yeagzqyMNWVOBfH71d7HCE4ch9-=w61-h27-v0

add513d6-ce14-4e10-bb44-33f3945d9a0c

https://lh3.googleusercontent.com/notebooklm/AG60hOqSxMMF2T0W0IJZwWR9s__r7l7g6u1wczxTbKmcEs8Xh1j6YU43pKmnu0ZSI12sIHKI97Z1DNE7tnTwtvYjoEzCvIPWhED9wA0Fg8C1Bo-7yKqKkXB9qteXhRsp9dBUaPLR71PLaA=w59-h25-v0

6be4057d-7472-4b3d-9b55-ed90543f7df9

https://lh3.googleusercontent.com/notebooklm/AG60hOqvy0IDgGB5z_HHgjkdBuMKU-qi5A3t5BRKxyi4r0xjfHtSO3uTvweZDIws49SduXetEcjtb8igWtewLUSVTaWWJO-HD1cIteLgW1XdLEzBu0dpTGV2XJwqrIQi482Ex9HSF2D6qw=w46-h27-v0

c9448048-4fb9-474f-97d2-7c2df20de1d3

https://lh3.googleusercontent.com/notebooklm/AG60hOoCABL7khOqHGvVsH2rACWi0Ee4BTn3K4fFSehNVeREZ_WTdnPJVt4dJjxjhqObVMBb1RK4cRCIhL4CbIx4bg-iiHsHvP2uow1ExELrr908xU2aUhHiryvr3zBvEK9EnOHIZsBT=w44-h25-v0

9f4f1027-ba39-40e8-9aff-af2d2116ae06

https://lh3.googleusercontent.com/notebooklm/AG60hOrJTbsoAknjKuKDpI3YWQsLF45VCTdxlDqCoUySJ7SnNR7vOj9JAoA0_9xk_8b-f8Q7CgtrmfrpaiSjwLi1_JnlXeZ5yalU7QkQJ1tuL2lzh1fwLZmQ17jjcob4cqYiBMzA5mo0MQ=w39-h27-v0

721d76a5-703d-4177-94f9-35449ffdd37d

https://lh3.googleusercontent.com/notebooklm/AG60hOqjRtE1CX5nkMMmL34kssz2ebbS5jQRym382Wq-m6UfyiZUMteZRC2qyqEOwCw6sFpuSBcnFx-2b4vnZDL7tBE_igBxPOVA4iqwfXosel4h9Wqv5RMj0NTGcmZyA9G_eIQsQFimYg=w36-h25-v0

d51edf0d-4db7-4b27-9aa7-8586ada04c32

https://lh3.googleusercontent.com/notebooklm/AG60hOonm66gnwupGuiXVTCKFpZEhVr-qsi7S_9Do32gxg493eVJ7aWUG5Ijp1oPZIduyWXrSOKt_3xFWEhlYARsbXVUF6O_DlqK3eSvZGUuKxa0qTdvkvFgZDRVUYBBVk7YV5FFVsTP=w67-h27-v0

ad304164-c18d-4a65-b586-bf00c24a21ad

https://lh3.googleusercontent.com/notebooklm/AG60hOpv7lpqOnIYNxFkQYkyX4GnPM3J5fjN70rJ7n6Yoo3fIdL19QPGPmpFX2-s5ufDlC-iuNbW54068B9r_lJQMAYK2pza1Y6TzwIBUCJ_j3AgMRudeDAbM4tRqtZT7MPFSTQbnzeEjg=w65-h25-v0

ebcc41c6-a25a-48b2-b7e5-cd6365b360b3

https://lh3.googleusercontent.com/notebooklm/AG60hOpq2oGZ0jyjvbOMzZaE4jTMtKvb3tV5IcQljlxRiQdygyzN0WWMAYjrwhEBlR5mAow56vwFv9xTHHCA6Q5KfetStJREg__Ju7U4P1XEKYEQvfiIG1hca_32kzzAQvNnKzlE3VwfqA=w24-h32-v0

24e2b81d-95ae-4ce7-81dc-c7d686f04feb

https://lh3.googleusercontent.com/notebooklm/AG60hOr2hjpH4iXRQOzYmK-UrH4x9sQOrETUImyaV7rcaCc4AxkcYzy6VmT2fr5rVSlTjpkm9LOTGsyPXbbjKk1rMuoHrgUKoT8CGO8TpgD7AaKCCPBtG1O9PjP2FMUoHTxi2q1bPan2yQ=w22-h30-v0

eee306ba-2e3c-48d8-9a83-3f60d244b193

https://lh3.googleusercontent.com/notebooklm/AG60hOqjmZrQRspEPFtEGo9N-IH82-u0j_yQLzT5cnMEnByG_J-K_fr3arzmMpYK-KCdkQvzpano-02ZMtijpCr-grnrNct3bJJ00cJ2JrvJZzd8mk_HflcBAhB9YjwDfP6oLcYbpqY2=w24-h27-v0

b64422f7-fa6b-4fe3-95f6-d0df3bd76280

https://lh3.googleusercontent.com/notebooklm/AG60hOovsgfI92yIFUjBC-X9ZPlF6yweFwF0z2E5dUyFDmPXdI45n2-GV4sj2tJ2IvuHgHMbqIbmCfpqjTrfkDi8rAoNeK73ZZVbAF8qRwd3a5NW_ul8khGAP_u3nXKvB9xR9uyjAq1lhA=w22-h25-v0

9267e68b-df5e-4cbe-93ef-674e9144e175

https://lh3.googleusercontent.com/notebooklm/AG60hOoxXtwF8XnKkT3Gs-lencmoP6kx72WPkIKg7iJ38HxFS_yRtCuwDpZxkUNu1wLC9MQqwJJcUpepa76DUTecxLMog_3-JMhApXuVP0-rLCpv7ynVXJKo4ot_fGnCvStrcQpEWNMsdQ=w47-h27-v0

28ce85a9-47b0-4cfa-8bc9-2f9f7ed9e478

https://lh3.googleusercontent.com/notebooklm/AG60hOoivvuZPHB8-nQmiFTPBdWjmgo7BTHkLFmkItoNa1_nU__8ie8YmyeWeNhEEGBcuEVXhvscHrCcUYXE2myQebXV-16GgHyWugoov6o1dW94U-FBPXWdvqpSPELMjFSUyMGOVpkOpA=w44-h25-v0

8518aa46-6baa-476b-953e-f60f318e5457

https://lh3.googleusercontent.com/notebooklm/AG60hOp-Msf-irre3u3EK-V3HyGTsGC5WkAAK4GpnLOL7hx6OVNjda3WHaoohPciOQBrkPFlfVPzxypjgjL_7ApouMYmmm4ccps0D8J3smeJ_NarHQ9mXfsWCQ7IMqzdkDXyPjparT3Nag=w50-h27-v0

b6ec223e-3f5d-428e-8ac2-b9072cd77825

https://lh3.googleusercontent.com/notebooklm/AG60hOouvZh--44rlz3YJ391fYgif1OqsU5lY6dT4q88wawP2xsHYAcbHW0-iYJbhY2IQUEjYatRq46OZsyENehkGtFqYc5jDmDxEQW8CRE73TAcCq6uSxiwGy5JeIGEod996CjNsm1kIA=w48-h25-v0

4f04b23a-d0ff-4bde-8c90-48080ed21517

https://lh3.googleusercontent.com/notebooklm/AG60hOov12q5LkGJzEf8BQc_fd8OQFCRZ5hKSYC-lXBf-WvTsBeeJ_fmEabWz5TjvKpbNc1y7WcTN4-zdDBhj60cFPFbh1DGcaCS1VTvA4utAEBlwvPlXfd1wK8m7GiMjnrRf923y3mYsQ=w33-h27-v0

0e74092a-1442-4066-b261-d10504fa6268

https://lh3.googleusercontent.com/notebooklm/AG60hOqgW1WU74YDOX7uONbsCH-UNHS_JABWc71r4IvHdiHNYaA-bvvLOcIT7Gp_CfjqyA-odWGuQsqadw0dDqOBuYs3PkcwOewXyPUBfOB2g96OCPLSXCmxbVjTzt-IcN9-QR9lNl8P=w31-h25-v0

ca1972c2-7f80-4ddf-b0ba-4efb22e96fab

https://lh3.googleusercontent.com/notebooklm/AG60hOqOhk-wqb3nPjIIj79-d8qaLiQ8EK0Jp1OFJaGGFhoTrp668QmzA4cD1a4CR4Xm1XIxO9P-NmIcTpyC7Q3EKtVfiBG3h5gRAjcls2OvSrVslhI33uEu_Ifxd7euM4BFMLpizhkVgQ=w39-h27-v0

89defb5f-cddb-419d-b953-cfd491aae353

https://lh3.googleusercontent.com/notebooklm/AG60hOqaY6ncSYK_sUtAVmObbwkZYYjCisKlEKLpDmswNP9KSafoHNXpeMkPZSTI02AQ_rygEHsAA1aLlxrG8CMp4XZw5-OgBlZFtECvaUzid1RbwlXy2-76Oo-BF7df5mCVNqtTkcVGKQ=w36-h25-v0

b9927d3f-d558-48cd-9f7b-c5fbbf6134ed

https://lh3.googleusercontent.com/notebooklm/AG60hOqvSGzfT0CXabqV5XWFL82edbogDt7faJFeB1_S85XxGYaw7J-wAsIAqTaJ_g9HTZmsflQMGS8AVI-QQokF8g6Zr40UE105UQlR7Rmf5nXHzNSxn3_IB6skVf-chCkoIpePAED6dQ=w62-h27-v0

b2f2678a-a93f-4810-8097-ed1ee797c876

https://lh3.googleusercontent.com/notebooklm/AG60hOpggYz9jfaFEYPrJgLqOMYs8HjiFO3jCEPI7cPiHgRtqI4DikRSkfkSNjccHwmT51Em8cgXpLSRG_r-_9Bp4JXQh5aL5sOscsNRQ-HILm0XSpjnmKMkeFdjQADCJUOy0eIV7mgVYQ=w60-h25-v0

b035bedc-3ce4-4bce-a0a5-97587441c296

https://lh3.googleusercontent.com/notebooklm/AG60hOrx_oBQWvY_YwKNCFTX5GnEEE2O5E-LYxul6W0iOVHW0hC9pAAgb7WQZFHTZ5jYgeu4plpoKn_ybqy1RI_MW6R465-IFtX3UzmzNgP492QM82v6G6-G1BPPpcJglPiZGvVrUPNj=w40-h27-v0

727fa5dd-ac96-4550-9c1c-2bcfc0a0df38

https://lh3.googleusercontent.com/notebooklm/AG60hOpj5Q-G5pD19s6sbYSzAR3SsbHTFLiwxd5JCHcx-zlCEfKLJW1XPjzv0SLbYRTSIw2r-5hwLYwljFuTyvL8Wc8mjYY_RZ09x2jH78oA0UhQiwfYOfMXvPYIx5uZRNLUEn8yC5JX=w38-h25-v0

315fcf04-084b-40fe-910c-f19cee241b27

https://lh3.googleusercontent.com/notebooklm/AG60hOqHUMW8f-FtT1qcTjuwVn_dnJLYg0MaoUp_RtEQt8dURGorGm0yxrwU_ULbG65dgiPfnSJPmJjJkx8s5JW2xVPfbNuINo9hqyInIwWbkXexyNMMqblUVU8kMhAve64GdGG2RsLizQ=w64-h27-v0

6edc4227-7c22-4b22-bd2f-fc29fb7847a2

https://lh3.googleusercontent.com/notebooklm/AG60hOr6CWe2Zo4TVXdHAFW2xn7ZgFjiQDkXzWAX3VDpWc_LRhS5yHNf5WL4dWIFrOlM5DPq9HEmxddB7NHCBNXtPAvuqPolaZcbyy8XBOAf6xxZjalc1PxBT7Hvh-MtNcKUNRqhgpmQMg=w61-h25-v0

8af1fa39-5b71-44e6-ba2d-694e5ff6b1f9

https://lh3.googleusercontent.com/notebooklm/AG60hOqdcadMrUMD8IdAuIDVFhdYxSeFPoU2CPhh9g8RFYssGV8VQenVT-bXuHkWnnesSkFOcN-WdrAk7-7BH3hNZuZEKMbgh_jQ5fUqN7RYk7WUx-lCoj-prdWYDJ9iBVZW_zGiYFt3LA=w39-h27-v0

7997448d-5243-4b3c-b083-d7354fcc9ace

https://lh3.googleusercontent.com/notebooklm/AG60hOpnhmLDcRJZ4072OOBnZB_8525ff-DQZ6fk3sSKx43rWSP5H0FH5w20URNdkk_qwY_xwekpAQpJw__5SlLTCFu9ZvtEuHJOvYP2FAYQeMFO_a9p6T9-b0nlCt3D9oIj-EmZ-u0T=w36-h25-v0

6cf88c0b-de6c-40d9-bef1-a6a513541dde

https://lh3.googleusercontent.com/notebooklm/AG60hOoxRY-8o1eq7GwXPeP2mCHdhWTuvJGjm5XYM6dOZvqw1NnKyU2G2ZXnewL3FRlJizGmQPDXEcqJOCgtPmYi6DjcnB3kvICn7gIIWBdYGDQVmrfzL-tG4VJh_YY6caG9vJBcdZcKkw=w61-h27-v0

0bf02160-0a12-4894-b98b-63ef32441d2f

https://lh3.googleusercontent.com/notebooklm/AG60hOo8AKJdfeI5l58U4zZXuEKkG6eJEEktFJvkHn7YgW9KPQbxqQo96Feze_iN8_rqLQuQLpuXKgkHiEHFfqPeyGCtRWNpLV4ymKwIB8swDXSB7SVCoWXV5_2_e09HIoLeFuWOgBMsAQ=w59-h25-v0

75104480-1757-46cd-8454-b6b265e5b58a

https://lh3.googleusercontent.com/notebooklm/AG60hOqunxtAytFlVuz7aiPcsWGmISyxPyb0ZXRZQHUUBIZ6_JSswl5nGT4Daxx290pl_EzJKbGPBdgK0gfGOv6dl-PSbBYOUDqx3Xcg4CmHQocZewOoAchtmr2SK6UfLYgLdYWVnt-ovQ=w24-h31-v0

1d96e707-b2c3-4db0-8123-7ef63e0e9b5f

https://lh3.googleusercontent.com/notebooklm/AG60hOpToB-H4W9cgWfoKWinFnKdEAj1InB_nJKeVFkTRpbhzaRcXv4DPPho-_ZdMol9eGQVGMkXIuUo4neQ1UYaGkfpDnF2bio4iaqy_Hzl3CGT9jQ3qkQEoZevC8n3rBqA5KQNNn2NIg=w22-h28-v0

10bb12d0-b5c9-4bcd-9c13-07f91565a5be

https://lh3.googleusercontent.com/notebooklm/AG60hOqlhETj2iRFq2kJ5Xj2tzsWF1SrikqHwrlbjFo-yKrDm8rU5P9ma2iiszgUB5PJMAkUrB55mCnzbd3k7LKZs_7CeeK-5XpJUZlJdoNUfPNTF9pjhl6Qlep3MLG9v9xdtDZhI2Z6PQ=w29-h32-v0

ab841fb1-7ab8-4131-9d73-265a2b14b4c2

https://lh3.googleusercontent.com/notebooklm/AG60hOo7WQUETWkHfuvRTJpnprPEcV9bBGlFqI0WNT7MKL6LEkppJr3anl3Bh2H7p1fbL22v4didsVx44DQIt-luE-m8CXifsdmxdCcL_V9914kCRht6RE80-Mp-yIDOswOSfdhmubWzWQ=w27-h30-v0

25857129-cdab-4b5c-bc0a-e50eb453c283

https://lh3.googleusercontent.com/notebooklm/AG60hOp9EC9o3SF23vqtbT3QxzND09zRmGOl6mQeJqSkCJXcfJIVaLkJfVxAjCgzy7jJREqzs7HZrVIh0gOKa31ae0gDDYCO3GjPuDQYdO6RC4RGcx4jfsYOti40d7CA1xxbjOw7nqNOKg=w40-h27-v0

a25106a8-cdb5-436f-9dfd-6642940954b8

https://lh3.googleusercontent.com/notebooklm/AG60hOoLgeLblSAMalqnCv5TxqUn-zs7HjmgAR_lwjTC1Q5hOjv4VBiUDwsCkcPY2yF25UVrr4BNu5GU79a5G4psuihBSLIx_Wl1YXlS6Ax3OAuxgLT_RnoDAtN2XyC5LyOiwCIpIR2IzQ=w38-h25-v0

a890dd13-0e33-4df3-8b9a-a35a071414ec

https://lh3.googleusercontent.com/notebooklm/AG60hOrul73goHiKSRoxg_5jEGIPAzGcTi6z9ReCejgR_zbv3yMTdAYMMcUuDtvQGeUqYUOLbWpLaKjVf-xRhlfbXpFbsXBUDxXnhkQbGADVRB2CLjnWh-u2q8iDQieYTK2b-ZIF9vaVsw=w28-h32-v0

3917158f-b49a-4c97-9813-0e907d22c421

https://lh3.googleusercontent.com/notebooklm/AG60hOr8QCFvkGsSMsapfseukEhiWlMzq1bHZ-noegk3RfpRgqst5PRFNUe410coAU6o_i0NQvzzqWhcfgx90GH9dYMrgeOdgCdIs1KO4TNtMCKGTggFdWjMNglOIaeKfizs6D6Jhrksgg=w25-h30-v0

6a3c7eb2-0c45-4b6e-8695-8dd08155d30a

https://lh3.googleusercontent.com/notebooklm/AG60hOrDXkwL7rM-WB1Dwn0Eu4Fk9s8N-8LPwERsjr0kL21LR-7c73MUBV3UNOeCz_Mz3dDzA33OnJlQ-63foaP9u29JmGZPzzA5FJYK7b9TZVlBJOymBt9iyGli2UH1vRTbQwQ18FiPOQ=w39-h27-v0

70edf1a7-1f9a-4bf1-9be7-60e68ec5d865

https://lh3.googleusercontent.com/notebooklm/AG60hOrxkoW_QMeptb3hwobzShtbw5RlwbaDNTEA1ZDydUoeeXbAN_vmiS6ZzFujiiMZB0jPimaJXwO7Ktgv3m_Al0IFmb-7He0dzkRJN-1R517bxgVHcXE3b9RwyDlVga-WKS-Aq0my=w36-h25-v0

a92bd62c-4fe8-49eb-9d76-75eee71a69f8

https://lh3.googleusercontent.com/notebooklm/AG60hOq4pFcFD03rm3hq_nYZhrLcGi5tqlv1stSfawxgHrWYaIBRwAaz0qfRTaAkRqJODmsDxs1PjxsYyv4UIZsiUCiFKEkZLyfH-Yr-kmmwg2qD7GqqNmVeQdJoslybHyg4IzDQPcyMqw=w61-h27-v0

3927d462-cef9-42a7-b793-16b73c92a43c

https://lh3.googleusercontent.com/notebooklm/AG60hOrjdCldRZEmNKyqgKFhF3YG03tsS3vZwU-lfPnAC16b-XDVBCy_1lgZ4-Tz1evzr6bzsqfkhJe0NX2cRrfAzsOAiphg8odKVFyRKK4l64a1kvEGlN422kRTYBcqp6toNzt_4ZN4Jw=w59-h25-v0

31377006-3376-41c0-bf20-50af049a09ba

https://lh3.googleusercontent.com/notebooklm/AG60hOrsryCEZRKQrniFUEpND4aOx7fkJV__MDmXqcG794Pyf-HzaxRP02P9E4EKFgOHTx_y7vPa9Ei5njVwwcTeI1aTOE_hC81ibkCjjZyKHQEL-GTB03vCyQLGX5tLKWzCrp8Qk4Uhkg=w75-h27-v0

d4088ed0-2cd5-4c81-92ff-b1966a9f40e7

https://lh3.googleusercontent.com/notebooklm/AG60hOooGEBMIvifzOXw3dehHsvxIp1TZIjvSmbP0A4yLYOPVt7J1m_nqH8Nu4HGonnu7XoL9Va_Cgpr0-SvAY9wPomG3C0Q0LS-0sU40WNMfCielN8Up2al8Xec52QBlX7Km7Z28ELDaQ=w72-h25-v0

d3892c11-06e2-46de-ba33-8b26e18e5e4c

https://lh3.googleusercontent.com/notebooklm/AG60hOqmW-Q1BoL4arpzUrY1ZoFjWB1g1SQP6MYPgvYGM8ro1dt6eyudRrohhF117o71nd1R4btL0Oi5WDb9tdBhEcemXsQhIUpCvJG-2JFAHDpRh2VJG2hg1zLaUqcMULuzb4eqz_58Zw=w33-h27-v0

e37a6ad0-92ff-46c8-a8d9-ce9299df5600

https://lh3.googleusercontent.com/notebooklm/AG60hOoslwqTk1rrhUaJBJzgEOh5we8AJ9Qsg_Jhjjx2MTIKQqPIhNkEckUnqhcn2DuuX-QPh_PByW9xdKf8Mk5ivDFe7AVl4VOGj2TuTCrZ5aHQCezoGnDXV6RN-Aw4OYlUSZKZvvm--w=w31-h25-v0

fd24b264-63e9-4d9d-aa10-79e78207cbda

https://lh3.googleusercontent.com/notebooklm/AG60hOrmyNM9x-6Tc4DL9F3g1b7T6o_tU3HHnSjTYqKA9YMi_lSBwpE1zFsqQqMcI7E2Wrs87INFmZwzopyQgCfuBkG_Nox1S1J2kJ47p_zLViXf3cceTrMhACoTkeKi1PSw9AQv845fIA=w68-h27-v0

2611be24-aa29-4490-8313-c5d9ffac270d

https://lh3.googleusercontent.com/notebooklm/AG60hOpaLuCRFnUz001LH3dTm8toMoKav_S4xonq1I_4f0z26hOkYBvcFBttKCj8cbUABh2BGkx6CMmATH24dxcKOtNAdnWuIdx1v3-PKZlUEbddIagwoh9EoWQ1iUcuq4iknN0PMdFkbQ=w66-h25-v0

27d509ec-a2d0-4855-bdc9-edaf82448ba5

https://lh3.googleusercontent.com/notebooklm/AG60hOrriG7FaWXbj_gfFykgTRXzeIoJsYKc5fQ11eZkk4h3WhTaTOWQLwKenWJgndI54S_AIWgGgVk0lOovejKE7lYi3BnyljWAJ7Aj0lpTqiifMHQO7aRBkp54ZByUGo1-Kyszu2whSA=w39-h27-v0

5fb53166-1e84-4c79-8638-4d06e9fb9d53

https://lh3.googleusercontent.com/notebooklm/AG60hOqmMywoukSzInIsLv9wJhbDZtvzTCr3xYFYealU0Exj3R7IGzHE8fnKQAqkHZkAYFkgzLNIFUQdKRFDwz69CYRkXY-l5eYCTqZufRWLDaiMpNerkyES8oaakeZCXv74V2rHHHuj8A=w36-h25-v0

51f6c43c-c0bf-4397-ab12-7ae2c9e2393c

https://lh3.googleusercontent.com/notebooklm/AG60hOrK5HMXl19NiYgtBVtStqYRM3ZTWZJC2Rm_UeueEC_DONMykzhJB8ol3YwxicozVBvlwFmIaCp_smoY5z1ICwKRHiBgBCCV0qpoaZrtEc6sqvmGYoQ8X6_eYGab7A52sekLlkKtsA=w61-h27-v0

68544138-b25f-422f-a1e1-27a25547b6fd

https://lh3.googleusercontent.com/notebooklm/AG60hOrD3sAdWzJRni2isB9U9zdwM6i6huGlwMbgZK_qabCriBPaTZcbumPlmPsMze5BVpaLO4uGK4-CJoy0FKvSPdLHeOcH4uoiDuFtjHrcifvMhmZ3y6iHNfzkr4U4552UzOleeQrkjg=w59-h25-v0

7fe6f09f-ec10-4d32-861c-220761cc053b

https://lh3.googleusercontent.com/notebooklm/AG60hOqHhXKQ-Hk4X4ssK-HNxKOJadg_QUtM2Bv1MKufxQlcYYpAC5nAOfaIi-C3jreLaVSSiijHRh8nKacxcnrw7EyuIRz0rdAGKa-zaxlTZr_SVLuVsUXq2BIynMpe7iuzRdUPpxVxXw=w83-h27-v0

6d56abc2-68cf-4988-9666-09c449b140c4

https://lh3.googleusercontent.com/notebooklm/AG60hOp11fN93Z_khW4slT4qn_LycHwumoNke0k3wE3c4d1bX3bUdtSYIVnlDK7GVsL-K8-LJ-7mWnFHtLxtF8UDzO_nhN878NkFPIqTSHtFMiHn_PeALFFXUufLOKRXA5sNa1C1SKj_=w80-h25-v0

709e70f1-3dd6-4fe8-a29f-8d8c925410d4

https://lh3.googleusercontent.com/notebooklm/AG60hOpEogIfiexRJueZvhvKY0XPftjpzCQt48lxqzmeJzVHfaVIu6rNcS7HTFVTCfJCLGjnYcxLL9_eWKI6CtUn5NiXgkWqQKEReR6taYJhPNpXtlr2eA0xMXJzlgy2J1xqYd-Jnb7LjA=w32-h27-v0

1ea13384-17a6-4b9a-bd7d-836f8d8d5bc0

https://lh3.googleusercontent.com/notebooklm/AG60hOqlChYtECZnR0ESaBnZg_UdhVq0pfNF26vFX89ELQZmFsQwzl6634yRpKnqLl091IVnFb754f4CWTtgH_1Rnxz0O81BAgACbqthNMk4QlF8dl6FwvOZA3QekVVufdyPY8O4Z-Kn1Q=w30-h25-v0

1cce980c-d095-4fb9-8f27-d8ea3b992a1a

https://lh3.googleusercontent.com/notebooklm/AG60hOpky_Z5Fgti9xleCEsYaUl9GiO2ek4z5w8whP2hyZhpkWyG4Z3kmEHnka64tI5iXOmY9_DipZwozR2ovPCf2J-hCIM_fIVztNCqzsDxyE1rv_R9q8VxbSiOpXaqxa_I5ojicdlHaw=w52-h32-v0

71dd4016-00ba-4d65-a579-a1f4e0113340

https://lh3.googleusercontent.com/notebooklm/AG60hOo7ggN2UsxoP0MU2-HHtgnOZlkGaq905k73EImSWDOn1xOwxKCa69Us5i2gy69d19_3c2xGghYO6et64Zc1R1m2Bc3QaJA9N-G2IUnNYSKXZPJQBENvIf8ZiWrz_fjfHJHlj1bDIQ=w50-h30-v0

a5de0961-760a-461e-931e-d14b09baa233

https://lh3.googleusercontent.com/notebooklm/AG60hOpW3e8oVVthIMwJKEi_i_J76NEYgWfkqsjOIiMPvevgT4LRDugo3EbNjx9Bbrt2UoLsfZxLU6KfWHhXZrKYbPAldEnzlLq8AMx05BtcfhGdiYgwMIMm3Rj-y_K15Hp7uyo7Wh_Z6g=w105-h27-v0

ed900f71-3873-4a18-ae3d-3608df2f0b41

https://lh3.googleusercontent.com/notebooklm/AG60hOqysOZoQAesaO2g8dS9vw6e81qsUcWI-y2le7fweceouX5YasCPTTefUflOTq3KzuAG3mKHiWXHJSYra8wVi4iK0O708mozjt6501gh2bn7-gE42miPuQN3824zieLue3XJr9uK9A=w103-h25-v0

81ab204e-f02b-4d7f-accb-26f59277e9e0

https://lh3.googleusercontent.com/notebooklm/AG60hOoPE0k5dGYE87TxRuqryLv_d10x9-sUdOG1MEr9uS3LcR4mazFCvg8FN-qDYv25abEBekiifcA9bk_7os0F1UiQqDE9efS7MW8RA-8CGdQHn9ePIRuZIWDW11XFXW9wxEYBn-6J=w24-h31-v0

917bf255-29a1-4db2-a2ce-18bc39537a01

https://lh3.googleusercontent.com/notebooklm/AG60hOqO3xxLhpm7MhUchJLjmvsTMvGfzZV984y6uUbjb_xDZ8rqdCjN-9CFrDZutBMf6jhRU-zlbC2G0FlIWhGWLTtjIKlNzTHu4UEg_omNiUrPELEFS4dZuJhqB5raO0o8EikUbLpY=w22-h28-v0

48b4cef3-6b2e-4845-ba60-637599929c63

https://lh3.googleusercontent.com/notebooklm/AG60hOoUo96U5RPsVfQAxnziwH4vIOKE1r9TdtBmwmZTHMHf4z7eQgtCoZob9apve8neEk_B3--BJVfcFC_L-KBhoJPXyl68m56TDSl8ZfB6ob7cMzc7ZKMyrqHql1WgbMzMgLfQIoEo=w104-h27-v0

3e95f5b9-ff15-417b-bd64-99e5333e7432

https://lh3.googleusercontent.com/notebooklm/AG60hOqXqeKXw2g41A2_DVFAzvZ9QlIhl7GAI7xjHvNy2HsWvReSQ_w28R3Kgk8_1IE2WJhWsNrLTuKGGLhlDz0vPC87wfCiqy0WNzB7r5ei3IgA5svUY76kEj-hBU427A5KNksb7v9b1w=w102-h25-v0

f0d139b4-adf5-4590-9625-a9148a17734c

https://lh3.googleusercontent.com/notebooklm/AG60hOp8LFAbZpUHl3XNksu8MyzdhBPq-EqGjapb7l10SOeq2SqDtML0krz0DagSeS5kB_-TLZoL6f5wBH2XbYDCXCaSHcACMzpApaef9aN-_EsBiVOb2MDKzKNALMkCNB7qJMO39mRkKA=w66-h27-v0

724d1e84-0e28-4361-bd27-92115ff11e4d

https://lh3.googleusercontent.com/notebooklm/AG60hOpjO0GR_3wRTFKX9kfWOS4SIflEKPTo0Vq0BnkmcfdFRncR5CMl6nVFIVSUS3-pzYPwig2BWwppI9_DpaWhE-7hNZCi1TD0rPgYxvttVXZLLUO-QTMx7DMxu1IUVx-kujzt777g=w64-h25-v0

b34fd83e-ea28-4721-8eba-c9cc1036a648

https://lh3.googleusercontent.com/notebooklm/AG60hOq_va-Xzr-Wk3g6VgixMDbmA46V5kljjRBenbrOQnrtEb9Vg9kL1hpTsRISdcDSfjUA3Dx5jkcUmu4TeniJTxtsfQcjoymHeELO7sLsHbA0K7qHVnEkHn5SGN5lZkT1-f-vAXBWQA=w40-h27-v0

1359ab14-89db-4c1b-8c65-4cd05b8de232

https://lh3.googleusercontent.com/notebooklm/AG60hOoj_ylvLGswWxImuZg3uFLKacEq6k3FNvKEQ7PKvOH2ToShzu9FJLtFk0xmwvJbydVZomXUVOQ5HUA1U7iMOQbUgS3wmaLxmRZLIHuuAJ_FygABMbSZfPVKnB7U1C4EcL1KTV7jaw=w38-h25-v0

715f093a-0e0a-4bab-9992-674c3ed0b622

https://lh3.googleusercontent.com/notebooklm/AG60hOq2h2NALl67twZ8rbaQGd-coBiKI2nONGuBU_tSxpg_k_qJEU5L5eFS2o95I1ztryQ8592nwNNMDBqeoSpF-KXDPQjuizXdw4Y8sfG22hgR4UCwLoojQ2KJXlBI6n9Hq8a29pV9ww=w32-h27-v0

eb2096ab-7a84-42c7-8660-4adcb1497d30

https://lh3.googleusercontent.com/notebooklm/AG60hOrAbTX4gt2h2_fPWoDiPQVkfSL3tCZ4jq_VWNTXPUY0AlwDN1sOqOBubL0AvR_L-Sqrf9AYhlDjxYRuTtpT6iXc-MXL7keGL3zNx3y-3imCxMp3dRTfv791a-ZgpI4VieOEd1iQUg=w29-h25-v0

63a0cbe3-8435-4d51-b449-637472bf832b

https://lh3.googleusercontent.com/notebooklm/AG60hOrkVUswSD2WoYgIHDbcqgUrLtjYAEFzHhSssbZ-zwc9QbtMOtQWfGEXmk7XffNUES5YtGwsMQDIEytjrfmKXQL_PhweKFgES7Gv0EX5GOu8jKQAEfE3A6ZNEENY4ymJlWH7jmn_Ow=w64-h27-v0

e89158e6-6505-4b70-8a4c-cb38df804636

https://lh3.googleusercontent.com/notebooklm/AG60hOpb5TEPXyrZl9OT6oAfkKv9BwrW0SNkZL8c0E-9manE5ATx7lPbf7gG781pwt5wpmMA2YjxdH310BSvZyMTJuLyjwxT2DPLcpUHwTXmvVQEiQYLxdVDJIUH-Y4wt09aPqvWtzbO=w62-h25-v0

5a3cdc50-e9df-4524-b7e8-ba4f61d6f2a4

https://lh3.googleusercontent.com/notebooklm/AG60hOrqxZEi4zvc41qaeO2j9gE-NmD5Ki8v0gvOTZUvc7HmQE-2wTwDQpVVu-i46RB7QAUuxg8yltFU-r80_u0PJChA7IZoAq4C_W6Jfq-mrQDk66rbvwiCbU990KSDiwl4d63G3qiN=w32-h27-v0

7c64b4f4-2b19-441e-9599-4f6e1713c32c

https://lh3.googleusercontent.com/notebooklm/AG60hOrcbxZfZ18NFAS8eAmKV9H40qD9k-rX8nLkX1LXxGZeSk8RxBOQP41zXWvrjCTKxaZzYZc4tJtlPifBHDeCZMB98D8wL9DasDkR797i7AbNVf9ZOBIoofJHsoYCe0U98mT1kdyD2w=w30-h25-v0

2320262c-dc9a-4247-a9e2-ee7f0a212473

https://lh3.googleusercontent.com/notebooklm/AG60hOppfiLv-l_C_RbGkospsSreXQ3XIsHqZTXcZlQsK_--LlmZztBQq78m3fvapIVq9pFIUKtLObkaIp-nVRp6oKIij1k5oRn7zsMOmtGtvvLJmLb7I2Xj4yTJye0x_UE2crw9DueK=w55-h27-v0

f76ff25d-4c2f-4295-8a5c-266d1e4d3671

https://lh3.googleusercontent.com/notebooklm/AG60hOp2sjOkEi88YtWHekF9ODHy--FaxA_XpP8FP49tbK-JesW8clZz8O6AR7F9WnLqLbuAHyKmyumeMd1ICoSp0LQCjjPQD5OWmNj2bIiQpbk97QeCV6NKhe4TR-pMWLcOsAWemlnB=w52-h25-v0

0fecfa73-05c6-496f-85ca-a0bb1bc0007f

https://lh3.googleusercontent.com/notebooklm/AG60hOqUdbFNA_QaN_JpGanbpTiC6HZavYy6rvmi3Z0movGWw1g3MbRBfENZN66I5EZ4F0jhTiXLwLE7zoYFpounPueHd_7q6IbnJx41tpYfYqvPRuLKt0dSANWYu1zN1i_Jb_JPLXdIpA=w24-h27-v0

27f3d849-92a8-45f7-9357-9786dc3ac693

https://lh3.googleusercontent.com/notebooklm/AG60hOqErkROwnq2t2RGO-N_pCuLW92MjK7n8sPMAkKSgidIbfkG6fsJmuTlOE8N7r7-tT_fq8euGS3r9YgLpkveZ6f2B0Tn_lk48iX4yaWjXLttxcT8pW2MG2RIMhiaMb-1UN3a2_YG=w22-h25-v0

7dd26e76-9f76-4bbf-a0e0-eb7de2967874

https://lh3.googleusercontent.com/notebooklm/AG60hOov4RjRzIgYR6_If1WBp2lPZeRNukGAyIGDPtXdg_0-EYyBpvuDZRM9AjORO3rvVqmaVKBv-cBIwy0lvzfilM-I9if7V3UCC7CeR6FcpQw5IItSxS2pCOl8UfGNt43FO414KFmibA=w44-h27-v0

97a82cdc-322b-4e6c-9c0f-a03d17d7a482

https://lh3.googleusercontent.com/notebooklm/AG60hOoMqOrQ27EQU6UHXbcrpnGbLHYqCIiAkzIBvymx5rAIwPTgCPSlcqUSRlhE2tzhNB2faHsJQ6fiaFFS4WHK6L0h3xURGFBfgbf2YMtvUjEosRN8IgxA537psMvAb-QZq9_yrLHMOQ=w42-h25-v0

3be24931-2423-4f88-b0fd-16be274cd729

https://lh3.googleusercontent.com/notebooklm/AG60hOq-ptt1qgE3DN2s5GzhUKhyo0dP_1rNNXGcc9vP2chIZFKtqoVmPMUBHVOGi0_-jb9cZXWTmDNhgMfpcWGLpRkwy55x64OOIGXytP_ko1D0SEAlmIQC2VdMqvATdfAeyGyc5dxe=w104-h27-v0

c7157308-d716-4118-a6ab-e5742a3865de

https://lh3.googleusercontent.com/notebooklm/AG60hOqTpuEjPVGZtvOohLDvMPRiCPzJCgLsZF-iS5Spd0fEFO5DCWcFsPaCZ9I4dhwpg60_5UJw97KpeT_eYxF-ALdqzLmuLoc-LTceB7pR_dGRRjfDyjZR15b5JNlkaRf-RZA_rBvEXQ=w102-h25-v0

74984031-e922-40c5-b367-9bfb626e5737

https://lh3.googleusercontent.com/notebooklm/AG60hOqEdf6r-i5ztWrT6DjhzclhpI9O5C5Bc_T7FWFElQODPL_xtSRUBTbGBdDegz2EJ1vIpvlNLwssr95yftMYtL88Q1jWv-zI_rBCRBkeAHBkMOHReXuWg7yI9q1aKnrJj7WsxS5njA=w77-h27-v0

8f992a8b-a097-4819-94e6-4e13412161cb

https://lh3.googleusercontent.com/notebooklm/AG60hOpfLqCMUo4xymjHmypRo7zbSpD1QgvHuL_Vbpb8yumofSvF52XhTrZ2MzdaxQQUGyleNYhpgSAyffN19VCrOq5TQiIa8vSTBEAnfiIgPXXNFHzYIVi1ZaH9-fSaaVrm_gabxWak_A=w75-h25-v0

9aba0ef7-b307-4e96-bc7f-f3e6a5ce857e

https://lh3.googleusercontent.com/notebooklm/AG60hOrIwHyd0qw98-4Pi-hu8ZsYG5wzYv97On-fChCMnK0oEvyJcV_dMpD9fDeifDMoQ68hc9M_TKBvMYsl1qmD1iIxDZstYHwuZc6_CrT4_dE5IzIb0yrhwWcAG2tUnw5F-6Riip-U4g=w36-h32-v0

0802a725-089a-426d-a357-21b75dcc48e7

https://lh3.googleusercontent.com/notebooklm/AG60hOpHrF9u8GL-zdXXtllk43gToUEsYPzBntm_emLXWKT9RH9fi9FXGkiq_hcGOUxgMDtkclicdLMBETbVgNqkBnxeknXy2geDq0lkqWKvcGyGro1TuxleePPUKrd7SpeqZmsBboVHkg=w34-h30-v0

cafe6230-0802-4183-913a-9dd58ccd880b

https://lh3.googleusercontent.com/notebooklm/AG60hOqPuiMtdP-6XnqvMe-2XjJGlIZug8raYMlWF_9AiQ6xEQdoDLOw3K6KfKyQvDRQWU834F9lqkztqfYGulvknVaKOTFlbDsJi4XlLD3Qqp85oMtiWDpQOYgUYujs8oVZfuynePRdZg=w52-h32-v0

4491fe82-fcc6-4fd0-97df-89169349a5c2

https://lh3.googleusercontent.com/notebooklm/AG60hOp5WHJN_ykjErTNywAG41971RXjqcUcJ63IdCuhZAc9Mtt1vOzTis3pPHCODq4GBz6bfXltMsWcvTke4_CKEBEhIyCC_5tnM0ek9BnsuXiq6aPRoIviZSSRGfaGCb0xYsHdX_4yfg=w49-h30-v0

4d972602-65ff-4ba1-a287-33e2eed9890c

https://lh3.googleusercontent.com/notebooklm/AG60hOriGSHzlRrBb1RB0JiAbbblJmTkL0YuERDybgdbagyX6V-Op0UqsyfpY98LoIYilglpJ-bPqHWHucYN_HyOFabzk2UaDOefxe1lVlMQ1VzpHVOs8KTaqJgZ608UOL1CAdOp_xV9Rg=w331-h124-v0

fa4ba7aa-d6fc-4ef9-9347-73ae962c2273

https://lh3.googleusercontent.com/notebooklm/AG60hOpZfuCl3lKCd8vxap_jNSiMpVqDjTFgwMD9Afr0bijuvz7ZqkScmbVz-07nwtYsWVz1HsEKtn_JqWk_98HEd1FmdV3gfBVFiR2IEWwAzqFwlRTDos843yDES1jnyW6GKd-aHR58NA=w226-h127-v0

58c09a61-ad16-4148-8492-9ff37762593b

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

GAZE-BASED USER INTERFACES

A 3D MULTIMEDIA USER INTERFACE BASED ON

EYE-TRACKING

## EYE GAZE DATA COULD ALSO SIMULATE THE

## MOUSE CLICK BY DETECTING BLINKS

MIDAS’ TOUCH & VIEWPOINT SETTINGS

## REPLACE WITH SINGLE BUTTON

## SPACE KEY OR HEAD TRACKER DATA

Applications  - Gaze-aware HCI

Images from Sidorakis, N., Koulieris, G. A., & Mania, K. (2015, March). Binocular eye-tracking for the control of a 3D immersive multimedia user interface. In 2015 IEEE 1St workshop on everyday virtual reality (WEVR) (pp. 15-18). IEEE.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## HCI SYSTEM BASED ON MULTIMODAL GAZE TRACKING

## LIS PATIENTS EFFECTIVELY COMMUNICATING WITH THE OUTSIDE

THE SYSTEM GETS THE SUBJECT’S GAZE POINT ON THE SCREEN AND OBTAINS THE BUTTON

THEN, THE SYSTEM CONFIRMS OR CANCELS THE BUTTON ACCORDING TO EEG CLASSIFICATION

CLASSIFICATION RESULTS CAN BE TRAINED TO REACH A HIGH CLASSIFICATION ACCURACY (> 90%) Images from Han, S., Liu, R., Zhu, C., Soo, Y. G., Yu, H., Liu, T., & Duan, F. (2016, December).

Development of a human computer interaction system based on multi-modal gaze tracking methods. In 2016 IEEE International Conference on Robotics and Biomimetics (ROBIO) (pp. 1894-1899). IEEE.

Applications  - Gaze-aware HCI

https://lh3.googleusercontent.com/notebooklm/AG60hOreeE9n0P-55Q6ajHrBZKwdPTqfXL55ZcCeRKB89gcrjVlQAbQjOwG8ZuA3EyAJyZPh68WQjPN-KXK5SAIP7d6CjSKp6R4QG04GRBHbxoJ2o0RZUAvbB8t5-nGRcrKI4JlVomcz=w116-h28-v0

58a71810-9a2a-4da4-a888-ececd7fa3f91

https://lh3.googleusercontent.com/notebooklm/AG60hOpojVz_-LYXKpUvqdli_gyFRag27ZqKfU-pL9GQvDi_kUXLB9fTip3eTU37dRFPNIopRFFHmSrIlr4vdFSEnhCGMLBoZj-URiA7sAGFeIF6AX5wKr9UOX9IzzPEVub0K0ED9M6Gbw=w60-h24-v0

040c28e9-6102-49fd-94fa-7c47d47113db

https://lh3.googleusercontent.com/notebooklm/AG60hOpPFTSqVH9LHyE0RncuV3kaA7EHlybdAxCdU8qhJKEhoqjs9Izb859uMtq7nAKX4LbLYhu888Cf3hHV8SxAt8orYEQ8rDkSVsyxas8p4qc-cQWchuTFERmy40uE_EoloStMjjS9wA=w58-h22-v0

c0c36593-8a3c-4170-b537-c0bf1c7f5369

https://lh3.googleusercontent.com/notebooklm/AG60hOo4eulbaujUTNVBIA7vY_-elkN_JJqp8oKpjqvskzv4wrgGWe_173JN69XMMV4DoNWRPANzYyyN4XH3BlvPdwnxNGjEmzCO6jBoPQntM0xhkF-OaSDzRoobR6PKkWPau-nuvUT0HA=w42-h24-v0

ddb54d02-8436-4a71-ba8c-37837b6b5e21

https://lh3.googleusercontent.com/notebooklm/AG60hOrB9AmuL8uewB7Cgp7bKnvqoCtCcGYZx_a50jAr9N0-z79rhW-e9oifdCUaxw8l5MKWPsRvSbaXKLRwpfkSm22rKX9QeggwB-Vz3vSHQjJ8HAljlykoU36vR9DgCufQ98YqKhhETQ=w40-h22-v0

311ec144-31d8-431c-a7c9-490e46179841

https://lh3.googleusercontent.com/notebooklm/AG60hOrjqiQ_cg1PAjplHd9IbmH0_kF2XbVyMIUXbsH0SaTFoNa1UhvoQi7LCXmVI7wA92x24BcTHlUBKEgkOcavtB-yNBCfdzHwd-VO4J_Pvhg-A1iBbJfuKglr-iWIXatyHtv93yrxaw=w23-h29-v0

c92a8bea-95b9-476b-8ebb-14cb78692b26

https://lh3.googleusercontent.com/notebooklm/AG60hOpM63igGXfTtYN52CtCawtTy5By0Wbr6oNWbOkTs92T2_jKgh18uVWbHuyPDFDmaSMv3L5goqEswRuVuWLMrzOwLEh_1ygunzj4akopppeXafjzfrCnPsiSrmufRKDaOR-VVqLThw=w21-h28-v0

50089f0d-fac0-4447-97ee-99baa0e34d24

https://lh3.googleusercontent.com/notebooklm/AG60hOpl-9oyuzAenskryNOxMgNBBS4lP17ybxE-_QSBZZgtk3bBLuW_KDPDE1J4fcOjzINqXBHwJBobExZcTDXUbAzEYHzQTcQyBaJ01YFsB1fqiWwVrfQiTxIZ85E0v0yBI942ZQEskg=w48-h24-v0

ad27d3a7-6cf5-4dc6-af5a-48e599533236

https://lh3.googleusercontent.com/notebooklm/AG60hOq6bAzoPk7Nj4OHyuPEffDmpehaKYAVou2qmC3gloVxycvyi5krV5b8qYSb2OLFjcM8eB69xu02Aqw2ghtwUQLuAaOZJDvaiGMKBbcBsEHoeeUbkxkc-8TFcX4rFRjU6rGVyYRY1A=w46-h22-v0

42cfa931-ced6-48df-a7b1-ce5b5f300597

https://lh3.googleusercontent.com/notebooklm/AG60hOqu6eEsG_HnUh-rm8y7biZ4zsashPKaVLke9yCGdaXPi8NT4IG_a3vcrdanDMLYEmCYQuzqSH_7cKNno_l_4n-1iPsz-WM0zIufX-36haRo9lJMw0yQY_2_QMYuf2PO7nWuHWEX1w=w81-h24-v0

018016b5-eca7-4cb9-b6c7-feb998ccb232

https://lh3.googleusercontent.com/notebooklm/AG60hOpXzaPXA5xC9jfBInf48LOfbWalf9QM2Dc__jdx1nTNu3b5Tic3T8Na5aTZUnPU8709jsGZ1pkzvSTCaaOzsVaHZi9Cq7nqGrNp4NBLF0ofzCaeIB0Cf5RIXBjyG1U1yhj7LiYXHw=w79-h22-v0

a0cf22c0-bc01-4e7f-a16b-70378ca7f3fe

https://lh3.googleusercontent.com/notebooklm/AG60hOrQWvU4Gf2pREd92o-C5nzAs2QWRAV3DHz8NEZHtsj35jl1CM9SRDD1LI9kS38Y_Et5XdbQ4kZexFpfITXh6sHv9HWDN1z8C6NMA3FBdnU1meeu6NP28suFJHqm1verdi906b0AiQ=w79-h24-v0

3b8e978e-cb6d-4822-af5a-b3e7b98f131f

https://lh3.googleusercontent.com/notebooklm/AG60hOpS_jlmD5GjlOFD4POzP3bqkWI9Z2qHTU4pl_bn7y4zB8QeokM_at21os1NZs79erJRJp_bjJu0GGOIsS1yiwX0ahH-wcMvKEKowIo0MiVFPOAGwtybvb9mkXPVopW1kXeT8rBJ=w77-h22-v0

a2c0b46f-a857-48ba-8061-64705a9aec3e

https://lh3.googleusercontent.com/notebooklm/AG60hOp9J6amnsI8uSat7q1vVrYKdRLz45HALlmFJJRqrFT_VNlbpRpomj0sl8yEa8Iq0mM4lCABEw_4Xp-Dnhb-cK8hsgr9-Az9L6IOvOtDrIsC9Qxr_lE_ge_zPBuMnqKEgpPOJngeuA=w25-h24-v0

cd6db838-9757-4646-84d8-feca46d95a9b

https://lh3.googleusercontent.com/notebooklm/AG60hOqBOgSXv4jaLvGUJzCqC07aCLFlRJzh74WrYOAaddgljDiXBsEoHyrgqtB-H_XA2Ih2fPQqfp4vUsgzhNun61EzptIoEEjN9fJRpAOUrNx6L6vauJZwmsg6goetD2RzL1V48zl0Iw=w23-h22-v0

973f8ad9-78c4-4783-a42e-57e40e3ee911

https://lh3.googleusercontent.com/notebooklm/AG60hOqTDTKifmo7qA4F20gB99DUIsbgJobA97aZLiqBFr5W4HEBE2cLQkjoCBqDKs5oPkkFott7_Yi5yOZtp7hnVYmptsDWxbm16zgtdTsNs6sdaCergAPQU3RZ6Msrp4IUGgT9R2Wrjw=w69-h24-v0

edaa8428-024a-4496-914e-61568e4d06f6

https://lh3.googleusercontent.com/notebooklm/AG60hOpwffsKegQuCwCr3Oe06rO91M5ZZI2SzKrk5ICcq2EoRDRc2vA6XuU536Bz0jUOjU9ThvCD_fl9R3CCGqHygAKIyt8qVEywVSJIoiJsGS71YrDLc7UFsC7wIDrUpCPtfyUDrXMi=w67-h22-v0

83f54c8c-a78f-41e3-90f4-acea841efbd8

https://lh3.googleusercontent.com/notebooklm/AG60hOoqSxbTJcoXuuL8eOTUdooPm3ngXKHgafuGiA12VVd_TIm3l2TDhzGFCFdjQ92e4ZYp4OC7AY5gRMHglN1wNju4wU-uABW7RcQ8KHShI5X0lLGC8xIpioDVLh2n1gcfaIh1hgi9=w93-h24-v0

72c3c8d9-65e2-43e0-a30f-29d75b11ce7d

https://lh3.googleusercontent.com/notebooklm/AG60hOr154IyyayYGnS2NZHNPKdHWkL9kKLJGL43wX4JfwGUthhzNtGBBlvQmu93WCi2oOwFBiyLGO-XBrfrniGA488wnXBUdQe2LCNmmBnjj1qkNrtHIHpbtHFLOxYtzxqxp3UxgMrtwA=w91-h22-v0

f16c5197-3410-455a-a017-5c4a4ad81a40

https://lh3.googleusercontent.com/notebooklm/AG60hOqHrZ-l0IJwfq0iabauNexa7F0VOh5AueVQk6yR6JRm5i4puk6D6Oz99q_k1BmcUdley6mK3p84e11uPoiQb9vk3c1I3JrOP_w51hVfrm01wUKAp65ATZINnw11NDzQpQXfT4zmYA=w25-h29-v0

0b82a5de-81ef-410e-8680-9c7f3d532ea7

https://lh3.googleusercontent.com/notebooklm/AG60hOp7kwra2NA07g_Moc1BResJxLoAMmtiR2gYovzelMSJ8WNP6c_Nxl0DaCuTR6YucVs0lMyhsGWSvJ2ZIBz5Fq01ghXQIlBJBF1LkvAKcpEkBEecpNX1rxOlUo13QgsoTkzIVZXOHA=w24-h28-v0

a75238d5-5d17-4725-a4cd-19e2745b1ed9

https://lh3.googleusercontent.com/notebooklm/AG60hOoaVwsIgPE9kTIra0MUjRBrax7kD3bPlUBGGk_j11hvSQU02f_0L0_bfeXbMP43TzgfPNT0eu_ozbTFC83wZTj5xm3qiKutaCMJGm60fP871cU-gzAue5OfKpPZjqx_WsadtFJv7Q=w43-h24-v0

fb8026a3-3439-4161-b827-457e7dd05356

https://lh3.googleusercontent.com/notebooklm/AG60hOrtN0Sxw7ixTiHHT8_DzzGL9-cgzC3OKlYEZrFpw4O8o-7SvJSi0pQtm5llmSCOz7CLh_m53EpFALjIswl2RYXXNQ5zOnAnrwrJC61dY3L02Fohr6JCaYP9SWs3bCfV-z9iVWiQMw=w41-h22-v0

4828ebcd-17f1-4651-9c42-5a3344351106

https://lh3.googleusercontent.com/notebooklm/AG60hOrseZkdxT7ODFAFiSVFPIfpcnVuIx6iqLKFRS0Q4gVKcpYIsgHz9stx6GtF6P50hDxgu1MJmutfx8qR8pouMI2xN4TnYSLNI_voA_pdXj3cGR4n758dAhb9bZbWex5Smlnbct6Y=w28-h24-v0

4af14c1d-6050-4fbd-9903-481f20fc07d7

https://lh3.googleusercontent.com/notebooklm/AG60hOoRmEtlGAt_eImNl--3DDQersUWldCdyv3eZLOPxc5m1aMQ4UFF8cEAwHtPcuISXCzxDiTg63cVsWYfqaOrFwiP_BxOFtMEJ6dXW_90NB8caaENga-kkg9HgZzkjm0rwml2F66PEQ=w27-h22-v0

6d69a574-e461-4067-bf1f-35a503ebae24

https://lh3.googleusercontent.com/notebooklm/AG60hOpzHar2m4Rbgh52TgWO18q6_dyQKtNvbQwcoIPLaTMgoUXqTolYYwIg0L8qAnHOSXeDKIpAAJe3_upzgriaHIgfQ3WqmasGtVatRh9p-k3OrsKoXxpRP1Hz3oornazuILnlNL1a3Q=w35-h24-v0

324ec442-91fa-4d12-9d14-360d7d501f64

https://lh3.googleusercontent.com/notebooklm/AG60hOoXCbO4CP-mHB29UzAWAQV5AsjOPWhqgah77l9IDf_wWqcag3nKxY7-Y0bfbhJ8BEYMnT1dnWqaRa4Inkmk2e95SgKtrEaM6WInxUrQbe_SUXpTMv_DPcAVpM3zXCZyVj1dAQzs5Q=w33-h22-v0

81e720e6-370c-4469-b526-4c4e0b7c8e30

https://lh3.googleusercontent.com/notebooklm/AG60hOqb4d4p4NVkxZvYwQB9vNDE2v7XCykHwbEpvenY5aTUbz0Ao9ES8TaAQR4dAtgurEXbyjKpZT1BoYfR-Sar2HhOBGnOmgU7-u9fA3dHt2eIZv7Ebu4PrS9WBgoZv8lzDWUb4u20oQ=w23-h29-v0

a85ccbf2-de4b-4d7f-9520-49196f8b3f4a

https://lh3.googleusercontent.com/notebooklm/AG60hOrZ6oClTXFFeIyZ12q6YprH9V17upxFrwcAITuLbrzStCnELFEiehoiYQdyGWGgsPF4EszOeUYatw9W7XHxRAjrEtQ7ev-aSl4KGOFsxoVs1cwHi6CaZ9VVjZvXQFTAEZMOQH3MMg=w21-h28-v0

d394c290-b0c6-4caf-97e5-1f867fd6de1c

https://lh3.googleusercontent.com/notebooklm/AG60hOozoHbMX3iBrC3vC_tZL8-RgkrkB8b50py4lv9zQOERZFbGiMazBZ9nccvqZWoFY9ggrHUd65uyy7d9MT2MpkbnOMJcB7qE_-A6bNbs76RsXY1KtvOQje-PfAQVPXiqdDkoSmENtg=w42-h24-v0

7479008b-b5b3-4113-b921-7ae7ab46661f

https://lh3.googleusercontent.com/notebooklm/AG60hOrMi6x5JeySvGIZ_v0a2ZjpJtDZTDTHHRBeB2vtjH8jUi-8dy2H3mLGFVDxOnAIvnJ-CLMeJwTenaoNIJ4qRXevXnCqkcMIJv_j4CZcf5vM6tKxyfCG6SoNA7EHDL4dL6mhtKw2=w40-h22-v0

5f4ab007-4260-4992-9f0c-da61d7f269ba

https://lh3.googleusercontent.com/notebooklm/AG60hOqfdUhIroCA3uFDzH0coFMvWyfBqcZLhLWrZ6hQNhBrh7W6VPs_LRo5_yZXXxR_asvV_mg_vm_q0WZTjeft_IhQ-zx8nDNmZz7evMjeNsH0Cvn-ctYFm31eWCXkSm-Wp33olSrh=w70-h24-v0

54b2b6a3-13b7-4d22-8091-b14434d903cd

https://lh3.googleusercontent.com/notebooklm/AG60hOqofPFA6r3MkJLQividdP9kxlMYv2eN861E8WTRaK0XVfnsn8besGpPR4LxKv5RhuzEdxME43JPeOO_9pW_24igcG9DPlsCLbRl0Vjq_VTj5Pecg9At1LvrSaC_4qaQRil4Yn8auw=w68-h22-v0

618336e2-95d8-4442-a025-081744c141af

https://lh3.googleusercontent.com/notebooklm/AG60hOpLZJr2esm2vL8KXEYLh9Vt4MRODSXAkjR9JqDI_riyoCUdWmMbbnDi8cwu_pp7KRA2c6HFlkJDH1o2Dcvn9k8yweJa-0D_1a63Fa9V6yLAKcUXjL7PoX7UfiXu0lt-ZYc9GDYw6Q=w48-h24-v0

369ae1b6-21c0-4ab7-87a7-c7eea788b56f

https://lh3.googleusercontent.com/notebooklm/AG60hOqJKY3uKu5LgbQ4_5KQN8USOVf9paRRanR3yG119svKddQlkBGnvfLaf2_YIzgtrtetH4Ap3qIliROigoL9oIN7Rj0bUZqgrLYbkiSwQopdVvdCpJd3vX97-a_U-sgcbBbd1Z7j=w46-h22-v0

4c218b64-5d7d-4a63-b764-728d6611417e

https://lh3.googleusercontent.com/notebooklm/AG60hOrLtcpdZGNfHv-Pt8HHFE0x6mP3Ociem0Kub9joi14__yIFSWfO5rg61e7ghjPB1S22x2lPD-pwd_Ra9kYwyx_4LdW0DzQtZpF_Sa6-XwAffsIdMLMVuReF0WiO5eVXI7k3HjVcXg=w30-h24-v0

6ebb8684-1e7f-421b-9931-68195f69bc92

https://lh3.googleusercontent.com/notebooklm/AG60hOpYoofoelQQ0sEA6kqP0Y03_ne1x-1IX63PrToAjFBsgYyljzghntd8pfJPyQE9WhK_If4oMUrw4as_0mk_zCzMMSSw1MerT0sxqNCWBtSGYT55W2SNXKVI9qbeYMR0ejUns_bG3g=w28-h22-v0

68fe9242-fedb-4501-8619-49607aa1b49e

https://lh3.googleusercontent.com/notebooklm/AG60hOrOAQdLncCDM_h8bdib9KB2QSCgesq4Pb9BUpFBy2hNb3-9SmVwnjGe-2_RwqX5HDWbVRQiX8S-VyD8w0AXWaU4oQ62vPSmSs8LcXgihdFDOrkdijTIgRLw4oipCIQvMUF-tO3Vyg=w35-h24-v0

0bbaee4d-146c-4d65-8021-4c858edb791b

https://lh3.googleusercontent.com/notebooklm/AG60hOqzLoib-txNpD-DE9XXswPrPYVKjuFFbvUCvkvayKiQTgweYUF9P3m6SeY1JxLZ4_3l3YdkBrRRBU8dNJhRfCWyrLiNKMr4J6tPXVcrE3ewNdlGyOk2uFDfDewTzvaawKY-MShwnA=w33-h22-v0

8dbe1660-cec5-4fc7-9218-23c30a3f0d10

https://lh3.googleusercontent.com/notebooklm/AG60hOqUfPvP_EX3Kt4p5pi6aVtr61Mp8SWN3T98MrqeMOaZGvpHX42FaWxWA97ZiAeuJtZJqBrQ2cSKIT3JQfE3apnJmnzkkmaxBZDs2LprzW1V4_xAZBslZbofcX9X8XJSLlj93quUZw=w42-h24-v0

fde42dc2-fe3b-45b4-a712-8fcbf33b6e08

https://lh3.googleusercontent.com/notebooklm/AG60hOp4LkPJtMEEIav81SMhvJrbIAk6Za_Pq5RWJrtysUe5Elikb4W3RIXlor23-OLTcjnbyr3MNfC92P3Pu78YX2-QrrHzwUIdnHZweHADgjLAFIEmZT42IttznEUrnKdFuNAEP-ufzw=w40-h22-v0

4eb8b0df-6c25-4948-9547-f1bf49ce7c70

https://lh3.googleusercontent.com/notebooklm/AG60hOrziibGrMurRIzan2O5B-ZhZpEYrJ-5-nHX6RQXZK4scqFDrTmI7M4BefEwkyyutyLAZ8Rej9AdUeW_8YaBPZJYIcNYp5ft6gavN8yFZAqeBu6ggfabfDzmh447TnLPP1uvNY8ZRw=w36-h24-v0

5901f7dc-5490-4431-b7fa-9e37b6a0b0f1

https://lh3.googleusercontent.com/notebooklm/AG60hOqBlEM7D-j2i07JVpTAeZPsfNmGXTKHWHtPE08RgUkRYG3l9cVmKRuxvOWgTNLr40eOamkJv9uhf0oWgpcCLudDIOqJVqbj9xANglm9OQOgCIU2l63IQh7eSVQOY2uXlRtP1q1a=w34-h22-v0

298b77f0-7b79-4713-854a-0f78e106a4b8

https://lh3.googleusercontent.com/notebooklm/AG60hOo22JSRgiWYzuWaSixmqvfVqjhLyWe0bB_CxXLT71Bj4h4ydlucBp5NQwOJun8QfT_fnZKm5BmM0kiTTwZ1UODNOJeaj5o1KvzqzXsQmpGxkLXHRir0GguQP-Qgk1Xcg0L95k3a=w60-h24-v0

9902b387-f8ed-4bd0-96c2-71ad68ba07df

https://lh3.googleusercontent.com/notebooklm/AG60hOqDtQiyNBGhWFUrJ2c_BLTRIYGXHlKJOwgKn9YEIyfv6OR-GkMh9EDVsTt8vLvMaNNvg-Kd6IyjEW51eWynJeoEFc_Fusx-EeLhyBRlPp0qahDOF-x3N-Yj4r3oNOyHjRtVnTOgzg=w58-h22-v0

53d87060-c762-410c-b140-6524fe92504b

https://lh3.googleusercontent.com/notebooklm/AG60hOrnv2VLo80ZrZiXc2MOjZKWp8aIFoZ0JkhZvM8PyShv6L3ye1JMQYKaKECzfcVM2ct1OvxnjzhDBbMNAazS46sMrkbzNHU2pIzxBMrhOETagHHY04q9QMVujrd2En4GUaJ0tHRgxQ=w63-h24-v0

2df83999-0bf9-4045-8dde-542124425e5e

https://lh3.googleusercontent.com/notebooklm/AG60hOqoHRsBJZEzD0fRgS5KQAipU1CC-1m0rM0fZMAngPvOEYSEaJXR7bffAr6-BlBUB5Dp9NjwcCf0_1zBeGsXn-Hf9VNUrNL-ex1uc2CtjEwROfnOru_jU2FZsz_ImX53EIcTFPVt5g=w61-h22-v0

85e7d167-8464-4deb-bc91-43eb598a9edc

https://lh3.googleusercontent.com/notebooklm/AG60hOpdnorLP5Ncf00R5xTMbBFkWqB0CWGW0QxMHOiy44THRjgrajPjevtANNs761nyIGg8C_aHiVf2lF2bcGIMADA-uEm1MY8QGeNOdvmTHrLYWUCCuAj05L1dwnqwBCfwt1mQrJsB=w25-h29-v0

5df2f7d1-2c28-4f85-805b-2586754b928f

https://lh3.googleusercontent.com/notebooklm/AG60hOoaviaYAJkWbZntBUp8pslZpPxY-ZCqZKLLN44ASB74vCgNebdcmx8eGHSsw_2lIDD-ZQmGpybgt0CadF5bD-DWAKSivD4aq59efYwchBf9L6DOVDptj3IkXKM603CxhxZmoTvHLA=w23-h28-v0

09f571db-6732-479f-b6c9-087ef9209dcb

https://lh3.googleusercontent.com/notebooklm/AG60hOoIS9CnyAb1-GdiKUBxJJwKmfTh608NIDl1TY6TQ_Nc_j59JsKw1YH_idoKSBJ7oDVcnWP8fTKMpBc_wz-lYTKRK3OGh6Tf_mRB6MdKxYdFwltTwtX4M0ZBc6MEp_5tXc-1PCzRZQ=w73-h24-v0

5a6c4aa7-e065-4905-bf8f-e2cbc0b7c787

https://lh3.googleusercontent.com/notebooklm/AG60hOr7Ng77_nZrJpjBfemRvSjUl4eomM-ddWhwNRa9V6lB_QULV1tjUePushpa9XVRzv-kXRD2d0YVLzRWDzQ5DXvEQbYruVnzevTfcTtxw-0_hgK-Gl0XzmmK03WnW040EgzLRWP8bg=w72-h22-v0

723a19aa-19d0-42ff-a2d0-2b5257ddf780

https://lh3.googleusercontent.com/notebooklm/AG60hOqM1ZTo0ZfufA-QnVPsN1Eo8V_bADMlOUy5b0jW5rCfdj--gGQy33Kab4EtGADJs0hZCNh935m77vlF8ogdVaXjxrX8Ky29G1WqhiE-SEm1-SvRW89EzVnPFjL-PU2P3Feb9VtS=w54-h24-v0

a2c9e818-d1d9-4f12-8398-1e38ed2b5231

https://lh3.googleusercontent.com/notebooklm/AG60hOrikzNYczQz9sHCwcp_sr5f_2cs29YvCRSAwms9zSjRCupyh8jcE3d06y4ZsbBi3vNfe36g7ETxTCvNuNqzpck7LUqRXy-I7hIQ8jObeD7_nP3ZzX2_jUH4Zal8IR27fiW4IuIwHg=w52-h22-v0

04a1c407-8e5a-47a4-a9fd-93485c43d5b1

https://lh3.googleusercontent.com/notebooklm/AG60hOpGcCnHxi8zvlZshbfnTtJhBhLFrAuhPuWbilBWF3WPRqlzJ0cJag-wEPp8OCuP3cbP_FABuh1mJaOXU9GK3yed6s3D4_ZZvf4qz0uKjfF7Kj2_GZdLs_pLuYITjiC6f-z4bzAQMw=w70-h24-v0

5da4202b-ce54-424c-bc67-7ff34877efb5

https://lh3.googleusercontent.com/notebooklm/AG60hOpNnJlQBGYSGPGQgpa37fSZfwr6sjSJBSXpMcgoMtFXUMA4uUWOsJijooMVsEY5OFv02T-dNF0LYtokG3FmOc9G5vJUlUw8QtwdBDUarAPOQAux11B0iRVvMEA0bxPWooTuqdkxFg=w68-h22-v0

f27e2714-d304-41e3-9008-337e3b1ce866

https://lh3.googleusercontent.com/notebooklm/AG60hOrPyxiC6eZU1xdHW6eLEU_MuOVBsYOssxe_McQJcdhZBFcUq_xiHA-Ot3LOKQ3N7kW0_OLsYo6HHjg6Bw4n4PZBjiC5KSexgtzr0wv3j8aQ4ehC_z9YgJOHUxgVNeNPF5zJpyz3NA=w40-h24-v0

b7ed4a0d-6ee3-4962-b8dd-0dea8e30ff8e

https://lh3.googleusercontent.com/notebooklm/AG60hOr4GU9v7qnS5FnUeN5VaDyzsrObg2PijiCsKJI5r49_rBD-ToqEmkPrv7OEemyKTwoqHWhCXfRkUuTznS0EscVSlkGAoUHQdQlhLrPDjJivaCqNy6mqw9qiMJKZZTwQg4A_xX_M7Q=w38-h22-v0

dc09863f-08ff-4081-88cd-fa071996d37c

https://lh3.googleusercontent.com/notebooklm/AG60hOrtr2eEeqkEqAOilkJZ-LBTiHrqzVZZDjd01yJcUQnPDH_nBf5kQPQ01SWhz_wMoqUCbBFq0k7d5g0A0XwmQHU_TXVde9nx8hpAMtXkMoshUX1013cGcDRg95h9k31rlsYFITUjTA=w47-h24-v0

d401404f-cffd-4256-84e7-875cf988eee5

https://lh3.googleusercontent.com/notebooklm/AG60hOoyLG2sBjgYLMP095XVnYvCP3hXmWVTTvDmswXBVrxteRM7bVfojZPcN9DSs6l4xKqgofWoGg3Ef5nrreZnyZ1nLGEv30OKDW0aJxoe5lBN5aYL0-WhbbjpKbT_9BAIFhKUQyexRw=w45-h22-v0

079df40d-7f14-4cad-9e62-3d18a78d4670

https://lh3.googleusercontent.com/notebooklm/AG60hOqrV-OOyl7Y2s0WrYwv8Zk56b9GLZTUWrUagoqknUndkCSSjDVOyGmhz_Tc7sUsojcHYN6H1Visb7gNEPg61AiSOMYw6d-xLamka1vV-3aX2Cs1ncZtfMKy35p2R_RwLZVDHZRLUQ=w74-h24-v0

13dd2b1e-5839-4691-82de-3c3aa322f835

https://lh3.googleusercontent.com/notebooklm/AG60hOqDYSV12tAXjDaeKYu9cupk2kCHhB2KEbmBrPRJOc9DclHlaVBfNGUuT7LE9QyQTT_7G2bxTxEAH1ZtNKAG9z2FwBA9bRbGtfd_zIq86I8vkopWAL5EHn8MHNn8k1dzoLNju1gQXg=w72-h22-v0

68876771-e58a-4752-a07b-f5a63bcb028d

https://lh3.googleusercontent.com/notebooklm/AG60hOpfDIV8W1lprbXBgJJMSJTnLTJc1wJhkKItiGmCktAHRTZUrmHml1pfNnusP9uLJbvjmE_mDNGIJ3mRUW2WivuGbXZw8VpyQF_RRxr5px-CuIaUf5aVR5sPcLAxonPsCzF-kUZFEg=w29-h24-v0

9eddf62c-a19b-418a-a49c-6b117589426a

https://lh3.googleusercontent.com/notebooklm/AG60hOpMSKHn1lvsR57Qp9eglWN3D0wRY0_UAh9sUl6BHIxBgaMX1bfL02vM0qkhEgp2aKX2RsLi-_7UngqorMc7OgZhIlQj2QJB0B4J--jkAgHDjD9VXiAVW_xgg5En199_4ENFRifJBQ=w27-h22-v0

b8ba8e39-1030-429a-8d9d-86cc880c4d9a

https://lh3.googleusercontent.com/notebooklm/AG60hOoYrJsegq_cLH04zE-MCKkVRNYOBdPH7QBw0iLW71sK9s9uCIvcr6wQ_sm_qgN-1cmcYKpHsNfaAlPgxeayb7hDlYjfhJkWEjPUmtEUDhEZaNquf-WiI16DZB5ha4PGPI5qYjGZ4A=w58-h24-v0

a1bbf092-797d-4d39-b10d-e7826d40e18b

https://lh3.googleusercontent.com/notebooklm/AG60hOoL6jpC3QlFkjQ7kV5CyqFMy9SZOvUJfkmNyhkZMP174Z7oTuboFCAdkQm7y9UqY1kHjTy0cCzSOn1CmFrwuLJaNF0n9cMBh4qD8KAvs-raZKzmHAQeCBw8WEwG4YVzE8NAF8Mi=w56-h22-v0

f59a4f2f-8076-4100-8134-dbe1da0d7fdb

https://lh3.googleusercontent.com/notebooklm/AG60hOqCr6_FFC2bgJMVPZ21ejPflZsxSy9lRDezROrkUmqnlGwv0jzJLR0Ca31gFm-Imh_tRqzRAbbKBU349Ophvudu0oscj4xtHEIA13ZZEjiZiJL6c10oppW1TaFZBpfuSpYYLZXZHA=w58-h24-v0

1b24a9ec-9f1c-4445-a51b-ec7325a56d39

https://lh3.googleusercontent.com/notebooklm/AG60hOp6dLXjn7UqcpT6fiscPMDlZtiSxk3x8KUbc893eysyjXli9gmiOn6ibP_8s-hjy3w7g8UHd-RZCobbuvVj67N8ZA2_7CseQX3Be4q7cKhyRxoazZGHTzxxHvYCD7WPjSWEEAE3Ng=w56-h22-v0

04cabb4d-337c-4297-b678-84998e576074

https://lh3.googleusercontent.com/notebooklm/AG60hOpbY_KuZq25uUWY-p68c_0RWJYCCY_p16lEhw5TeX0W2eKVYkd-mqJeOvGMxzKN5ILHZDPukzGZuPA529HvGw_oKJ5Yk7kBXUsUkizCNGWDGB4uqA5rE8AjZZu4U9-Ppxb0OyVPgA=w24-h29-v0

8ee920ac-1177-4a3f-bc23-a33f46c19413

https://lh3.googleusercontent.com/notebooklm/AG60hOpz1x7lrGKRVBPwI9869oqRRj0ACxqRoQx6BSdkk7tvzor1OU5MpzFYflnHKD7XJa-Ns7eMsZogGc4CrCKCXuAEP9uMXNoS---nuwaer-GPk7t3sH1IY1dYarOP1C5rLrO0e1OI=w23-h28-v0

b15c945d-1daa-4f25-a7c2-1f86165bf344

https://lh3.googleusercontent.com/notebooklm/AG60hOrJ0llNQGMzKL1-psObcO4dM4upOJtVJtXjEypN760alN94QQk-oKo_Dzd1ALIaMkKbmPWhyhdqjHNeTcFKV83UhlPnlab_fClk4lmxpd7NEY2W87LJ8AAg8VR17ErzU9Ac4Ff6DQ=w36-h24-v0

cf5bca63-3133-4977-9fa6-633b6a130cee

https://lh3.googleusercontent.com/notebooklm/AG60hOqK9LsWfGL2ONzy7rO5gcxgVSKU__kTd96eZTbHQlJ3uLHaNBsVMPMdfm-k_cbTXdlNiCUC8axsF8U_hPm2mQP59pQQ1-AGYn2JxctyYtbmIHnIaScc2JI_zrjsMUb47zUVKqWVuQ=w34-h22-v0

84d4417c-b27b-49b3-b75c-86a8ba7b0ff6

https://lh3.googleusercontent.com/notebooklm/AG60hOqBb-jxkYCJRuDSM7mHLWXJQB8ssCeSIZ_su1dlNjtbtdI9xkw0alCyAROxKXpWoL5ZMRzNodsTDnU2Shr4MIxg_4s5V7EdZvWhCyTotTx_xybMpknAs4kWpIJ-OGx5bHEgb4hHew=w42-h24-v0

58cc86de-0291-4af2-9be2-8d6aac0aa1e6

https://lh3.googleusercontent.com/notebooklm/AG60hOpkUBkFqhTN1LIS5ErO4ZV9sASvmSOIfIxvV5N5-3VHp4gho2f_fBoy20Z7YGa1lGr6_xcE_mAC_Y7mKK61pNNACzKzE4_XLiap5UigvcPjE7BhNnEDaSs8Hpl6sDKEePHroczI9g=w40-h22-v0

0a990b32-5b76-4562-9e47-31b3c953ea89

https://lh3.googleusercontent.com/notebooklm/AG60hOoeMohOH9LT--Rlfv55_GE1WVijQa2yNnU1wHA4hvdipSY4asOrQ135P5CLs3lAkJPCZCCRpgdAu_04mDM01wnL6XdPjJVIO8QEJPPeV0RtoiXTW7MVWsmJF7ieok7HUisT3BWt=w23-h29-v0

93966f4d-dce1-4da6-ab2d-bf596ad609b3

https://lh3.googleusercontent.com/notebooklm/AG60hOooVaSNJ2pYRxw8-eJjATz_ONhDwcM0Ccos7xz62m9YCVR6L6gKMplak8swRTXc8yE3kBr-2ap1w65QhQob65Mwrz2dGkUoa95z_Yb065nw3Sc-3ZwEgmPqqzfAYMsfJg-JE8jL=w21-h28-v0

dbefa796-ff09-4a4d-9c10-58cb7388ea1b

https://lh3.googleusercontent.com/notebooklm/AG60hOphvsJg_816ajdeIdYzjlDxh0uBd46V-lCwTYBU0fmGEaw2dJivdh6V8B-MEdJ6PcMm0G4mdh9mpb4h7Y2RQ9-M9B6l0fQPSsn57_uN0uE_e7_yLNonI3swm3GJYQ8Q87PI-AU4kg=w62-h24-v0

17135630-59f4-4d6f-91a8-9864867ddf4f

https://lh3.googleusercontent.com/notebooklm/AG60hOqe9lvQankxm6Q9UHuRtfYte_zetDbJRtNgAzGt5S9CFqWJALbR5Zy6vBEEesmwAldMUPsjKK7CTq2mBglXi-7jKMwMtcJ_pEEbcdGyNnCzbSE5nNzw-ZiOAjlwtL9hoIRXX5Ea=w60-h22-v0

98176f5f-293f-4a50-9fbf-15eb745ed42c

https://lh3.googleusercontent.com/notebooklm/AG60hOojxZaOls_5NV-QiDZI9mpDrxiHvXA_8u4R0MlONSBHQ1zuFPeC87NigR-9h19duRwsOkBnRHL6eSwd217kmDigYgh4JQwMJ97q00pPJiV3PciAbTOWOMmrExBKvPJpcp0AFw84XQ=w23-h29-v0

a04602bb-60cb-4251-8ae3-39b89ea7460d

https://lh3.googleusercontent.com/notebooklm/AG60hOqPe6RTnpgmk2hZDaeQbtE1cjDrssYG7ApupTo6-rxWJgXkQ6DXFmxBpFij9NeNsFJIaHgIrFweGmKEgtdzTnYR_9OAm7R9ORLfLJYsTjJ_CrJ0nCLLNctxlD7wrW43r6NrT_KI_g=w21-h28-v0

8a39c596-cbf3-4a9a-b5ce-8d56e2aab724

https://lh3.googleusercontent.com/notebooklm/AG60hOoCBprNJyyJMpgAK7kEU9rnkp-kl-qmKajAJLLm_NgdZjrwsfreS2PSGi6fMMeJzwF_qzQSPuZYxbQB0ozx-ZtNZKXHqISfA3AWgqijrgMn6KzndkTN_o4mLvCb5Krq9n968eoxdA=w49-h24-v0

35ac37dd-555f-449d-a563-01499575624c

https://lh3.googleusercontent.com/notebooklm/AG60hOrKIKmHnwz21h4-2imQAgiqdxlDyJLK1Jrl3HCaHZHNwqTtuGuhTBN-NgccS_e8dMMoqsrIJZ1CJ8Gb74dchIY_wsT_TUVJ9WhsLDfVKI2snewPc9KjPbU4qNRsci9KeBRvpOAhdA=w47-h22-v0

bfe2524a-192b-43a3-bee8-49caf966befe

https://lh3.googleusercontent.com/notebooklm/AG60hOrLHm2dAJwyOWA-LurkLOPO-vwW2L13OYFPA9XD86a7f6P3R7XNHAa_yT48b2LrDB7qkvFSRZUinR-mJVCuD4jpZ3a1ChcwVp_IGxvBZEcxJfkeflySLP6OrveXcUfmPUxS7GDpjg=w81-h24-v0

3bb88aa5-1ffa-4da2-95c0-163285373797

https://lh3.googleusercontent.com/notebooklm/AG60hOrpkEaKZwrkYwi9LyaR4d8ypkf3hMNh1iF_vZa2EfeFK0A7ZwOBr3cWHiTpDl84cRxMgyqJbu02PkqNjKVU7r4ee6TsTdDDDd68leWFPlC-xCrQOsfSC1qy4KKoAb_eUJ2kasbrvg=w79-h22-v0

e37d66e1-2b9a-4d00-8a4b-4640b9ad8f7e

https://lh3.googleusercontent.com/notebooklm/AG60hOqCwBZayQRKWQIp7o8oZcbt8NUh5EZoKGZmdlS6ZL9Zt4SRer8F8LZtfYNlrAITdMHs9qTsWJbTl4pcnARmc_RVubNmTmPGrSEq7detO4ThZAtB40QwVRCNvhrD7m9lLGm3_vRk2w=w57-h24-v0

98312970-b97e-4028-915f-cff86b6187a1

https://lh3.googleusercontent.com/notebooklm/AG60hOoTGqSI6_EMsp5hALGBjgSsJgrl2VgtwQAAV9WysTY_1TeWseKIvluZjoQmXuKD5mkhvT8Y7fpc55Ciefkv_JU5LWyFsmwWhy6DNPwdfhtrd_W5yyXE4SaRbvF0kDVG-XlXb-QKHQ=w55-h22-v0

28a89654-75d5-4fd1-afce-056e483ebd88

https://lh3.googleusercontent.com/notebooklm/AG60hOopf6RqEE3a_xXdyFjWX-jHjiNj0UCLF2i5sjFwQ_9mZifgJ6LHDoJIpPS41OpCx1UOHFBUKr8XBu7WBX5nMnSZGDXDqEAWXmRitXOfs0fhKyR9XyzzTyCJwTTIhV8PDoLUn2Ln=w30-h24-v0

adf3df3f-1815-4acc-a20f-07b2bf54d0ce

https://lh3.googleusercontent.com/notebooklm/AG60hOqj-UnQtgDlMNHvBmxy1p9p1whWIha7_qgOqVst1Nmr6rl7Zgf5SWybOGuxG1oNK57OvAxc57a1UZ6ZXTCW1Q9TBoi902UhSGCnSexsTfzpDlYgtbOc61uUFzHk7diD8wZN9mce1g=w28-h22-v0

8e3545de-fc47-4692-8853-17659a14f412

https://lh3.googleusercontent.com/notebooklm/AG60hOpkT7ZIrXFDSdQaadXq4Z2ZVn64ci6_n8-kGwX-LpsymqBrJAzV_oQMcWE9JGaezGvBZiNcaJEnjdgdjydgyuAv-zUKBjwxSB7rmrdZdBKglxfPU3EeA-DWuCPysktpP38ZQjajIw=w35-h24-v0

313af9f0-33d4-4816-a079-c134896e1f4c

https://lh3.googleusercontent.com/notebooklm/AG60hOq5_TrzaH404OLtCo1uaxzezjR-HdxVfKNCIZbGi6rzvbU06cPJ3_NyznJAbfP0kETE5WEzO_MLxW8DazzTLFdDgPOpRxiMvz8FMC3zGe6sln0VB5XR0PxQJBJ8AWYQ6xla1NlJ=w33-h22-v0

c3e79aef-d0b1-40b6-b7d5-ea9ea2571f8b

https://lh3.googleusercontent.com/notebooklm/AG60hOrmmZB2ESe9p1f8BsI9Bgjn8eLCF6wSiRUGZjQRg3h1kC6cP0NeH4CLCeVD4jJq92Vt20TKjLaoQCeTSU0wqeuFaro92hrGZU9_rVNfNRkWOIpT9uTQlT-dV_TolXVfLxy7BRIa=w71-h24-v0

6923dae5-1b38-44ca-9f9c-facf74ebf208

https://lh3.googleusercontent.com/notebooklm/AG60hOpwRij4FuNvgwi17prFCoiLXUGzqpjD169UUKXTZr0n4pRllfr9btds0IUct4V2hRtGJOBULyopoazWpEji-4Ta1DJcIvB_S1G0Jy4r-0DJyciUJUFTFa1I8VKPE2fCLB4Nyrw8uw=w69-h22-v0

43670920-32ed-4fd1-9835-eddbf52ad014

https://lh3.googleusercontent.com/notebooklm/AG60hOpYRbe5m18HZC_1bGAtsqXPtUORWuOg2bkPe1__QOJNiwkHONlUvuzSpOEfDRIclI1V3ZTpbx3QMIjIyssDBX-wccjB_Xui48IjyzhNhVhKoWmi0Dehib2ldY38mhBxJX4hXrz-Gg=w23-h29-v0

953b3b3a-e9af-40c6-9e74-f327010a6865

https://lh3.googleusercontent.com/notebooklm/AG60hOpwyyLF3bZS5TstAo878wu0YP6laesbpgPEInsoYA_lrop13jdnZ9q-7mJHGmzrvzjvnHRkpnS5nw0jJbxG_AFg2-GtXVzggJJfwb60Y4wLmEuIePwQtkJwjHpvh9ecLKBkBTb3=w21-h28-v0

0713b9f2-cc96-43b1-9a70-1eda4154f051

https://lh3.googleusercontent.com/notebooklm/AG60hOq2mKBt1ZPOenVJUTH74UYRtCsDtXFf4FM2K413zFn0nr5AtrG-pV-SSLN67zvSA3iAmnt70wdumV-7NCSf__q9Et4102yL2-laEKnkJ22pMMKKrZeM6b162ABglyP8EJwklzsN=w56-h24-v0

903954ff-b2b1-4933-b0d4-47553b43506f

https://lh3.googleusercontent.com/notebooklm/AG60hOojNfJFp1K1FW9CK-k80ilzPCZ0WQd4mc5soQ1sNPPY-PT8rsoHjRicRwgHWuNL1i0y25QKPu7pNDaZb_z_YM1QFpY3PUcxUUqIPvfLLcQWjfR6CSyni2oNjNT02InY0aQ1oocv=w54-h22-v0

b0b0f9bd-5638-4eb4-8fb5-02091b90b6e0

https://lh3.googleusercontent.com/notebooklm/AG60hOoDa1upVjN17OtL6DbNTPDfldp8bO7S4kLtNE2ZQMN-yCoc5nZtTQO05D3JxTpQw48gj_JFuUotmGJ34Kg7KizpOTAst7J_kBt0sykqmrzjycT5C0ZejZkXojjNlLMF5Tw_Lcre=w54-h24-v0

7765e0ab-58cf-416a-b1d2-20aa453c2772

https://lh3.googleusercontent.com/notebooklm/AG60hOq4SvpurXIJhZYHWvA8tqI8WnYLYLdjUVqgNft7su-JKycCdeREdb41Ky4ZEt8Ci1qUaJVETmlagYjEwKqJVeDQ2ztPimUaf1NbXANzsdvnXtUzbxe56RV50EABC2coTLHxIhzFdw=w52-h22-v0

6657aff4-7924-4373-af1b-898693340a16

https://lh3.googleusercontent.com/notebooklm/AG60hOo-Sq6N6QjrNRUevxl9x_c3t-KbhPrX-cvv-MTBqyYqROKxsSy-SCl6N3diU4f7OBk1SjI3Ant23byycol8VRXPYNNFxWFq3V3VU983HUMu4TIpOQUNuwCd9WxhibTlpsUiBQLfdw=w410-h125-v0

e10bafcc-a307-4c59-acb1-484007431a4d

https://lh3.googleusercontent.com/notebooklm/AG60hOpUY6wwpgvYKtfMoPCbs6WHkneA3T1oVAeV2CkiROS1xKG4K_Flcgz1DVT7wMNPHzKAlplG75e0LwytqtY_hdNRtY-lV6UnEkgXHmVNA6ToCRDnPkKj_1hdI-R3p2asqmnXXGmHtQ=w151-h36-v0

984d7675-9c74-4bfb-8969-61fa69fc29dc

https://lh3.googleusercontent.com/notebooklm/AG60hOpQbFGl_krFgPXFj5c-SCF3s5jRDMYdPUwPCAxeMNs6iPoZFnr8FuWLEfyFG68mKJjYW8XMYnp-1UriMWeGx9kXE2qXHLFHLMDvT3-2yYtDSVXPdcafI1MHxmBHwRcoOvd0TwtX=w300-h359-v0

6457d365-e616-4a56-bc92-ff0952247814

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## MIDAS TOUCH PROBLEM

NATURAL GAZE-BASED INTERACTION TECHNIQUES IN IMMERSIVE ENVIRONMENTS, SUCH AS EYE-GAZE

SELECTION BASED ON EYE GAZE AND INERTIAL RETICLES, CLUTTERED OBJECT SELECTION THAT TAKES

ADVANTAGE OF SMOOTH PURSUIT, AND HEAD-GESTURE-BASED INTERACTION RELYING ON THE VESTIBULO-OCULAR REFLEX

Images from Piumsomboon, T., Lee, G., Lindeman, R. W., & Billinghurst, M. (2017, March). Exploring natural eye-gaze-based interaction for immersive virtual reality. In 2017 IEEE Symposium on 3D User Interfaces (3DUI) (pp. 36-39). IEEE.

Applications  - Gaze-aware HCI

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE TRACKING

REAL-TIME CALIBRATION, LOW POWER TRACKING

https://lh3.googleusercontent.com/notebooklm/AG60hOohU4pPW72gsOMtulNi31g0gQiy7ehHC4YdY9oyPdJlWOaYCf2XBkt7-tkjWM6snU5zj_D95HSMJyX6SJhPj5iFo80ypDsohupgLguCHYnfD716qlY__2EihrSM2jPj_J3jsPwn3w=w116-h28-v0

cddbc449-f1f5-4b66-904d-38abb9757dcf

https://lh3.googleusercontent.com/notebooklm/AG60hOqA_CElss_ILmRZVRPxfNp1wPQ3Voy97ekLuRtz9Y2uSLxvnbVtzn9wc6IW_XmOx1jjwb7ENJul86hTlv6NyFhfSVK-ffC-mF2jNYyo6OfpRkMDbvsItWXeuD984Kkfbwil0rcByg=w24-h31-v0

929bf1f9-b385-43f9-a9cf-34c1f1022c2a

https://lh3.googleusercontent.com/notebooklm/AG60hOrlIPrKOT6969N3vfs5ZwtW3iBwN5RD9Ld8xMzh9iYnLD6VxUUgghyyXiArWpR213msM-f2vKKXFaH4kOFognihRA62I7zRsXHW0_rXXsp6aWx2O6JrzItEm3PNvn3epDP4tZ77=w22-h28-v0

5f02a602-033e-4921-b049-4fa9451728ee

https://lh3.googleusercontent.com/notebooklm/AG60hOrVAAzlcG2Li2wEC6TBh_I1zOKgTtvN4LkUjTbX5AGRAwZJDGhxX6jJ237l-TvgNyJbHSUWYVDrykxOWmO9rp0_rPSXJT4zqCwZDc8B47HkdnkoYnXBisBZoog3ZUB7kFREr8Xc=w29-h32-v0

51efd298-66f8-402f-b6c8-c2e86de4b0d9

https://lh3.googleusercontent.com/notebooklm/AG60hOq2217ZqdiSUhpagJ6pVXCqINiV0znZVkKIHwQXEuDVrUSCFesegQBxcgsJhLVsbl29r_FD2SJVBnyaajAD405liEmMKGO8JULgru2IUHtyFaciVzQWluXSCUu0aWhuk8BGpYY-lw=w27-h30-v0

51f94334-918a-4f78-884f-d7ebebfdfaa9

https://lh3.googleusercontent.com/notebooklm/AG60hOrL9jcM14_gaxaY_86QETVzV8zHqpz3_ysWOq_0gTYulSohfMb78mn9fKEomhspqrNWz0OuJ5hskncF0v7enqkUGp_OylCVelhNY_4Q2Soz7RgtJLGLvIpcW1QlmSI-om7N_iu6Xg=w72-h27-v0

fe295238-651e-4486-aeb2-7421821740ce

https://lh3.googleusercontent.com/notebooklm/AG60hOoF3lExy013cxsnBF-aH28p6FkG9h1vs5TvYGVtKQ_suh9YQSufaX6KvN7vCWJV7iI7mJmrIAWhnS7oRHg2MxLYbEKoOJ2z2fRGW09JVnjFKZ9ksY3E7hrJ0Rsq1gMgsgCV_tgbfw=w70-h25-v0

31ebdc01-a459-4a5d-9123-c15077073e74

https://lh3.googleusercontent.com/notebooklm/AG60hOpqUM9hhpmAbjmTvk8aB-7kZ1B-31zas7mczV4NIqC3VIwL6wC8_G7xUhOeZJS0eMg8-s2kAdAW-He71VNXH_jU7aLbSohSkwfU4qOOuA7JXf4YIjiMRGIHYbXO0ifqzfrzIXMCFw=w39-h27-v0

b7ccf211-b34e-4ee3-9741-20b25069c18e

https://lh3.googleusercontent.com/notebooklm/AG60hOqSE4DlRyAKLy5Nj8niziuP0FQFDSuJ3YwFzmCcveBe_y2aS8NO3mj7k3efsnXEDT9E0IV7gVbGBKyhPFLaHL-g6VGb2VlH7kgvlWR8d2KmfX4KcuCvgDbOgddupif77FYYjOXG=w36-h25-v0

37f91a10-750c-42c4-a71c-3676c7383e3a

https://lh3.googleusercontent.com/notebooklm/AG60hOpckOQ3-fPGI4kQLjX2JO12aPtOhWjW4xF0jYRcCjsoNvE5mIZIzehkVZPD-5zD5Ff3pdi-VSzQg-tBJGagXDHlEoOcuqNO66HJwY0rrAUomQeUGbj7ykhhujEq9z9Sur83yXwm=w47-h27-v0

0d429465-2413-41d3-b1bb-d9e1f1c801ed

https://lh3.googleusercontent.com/notebooklm/AG60hOrgTHeDMzRl_4TeJL0Rrs253-v7x1Utwa1_ohgc-9hyNOrtOn9tNYC1-I70eWTYAS6YoaqR_taIuH0LX9NPleJPi-juR6Y6URYOsplNBxjjg01BBrMDC32fZyOfebS5x-QsycnTRA=w44-h25-v0

af9586a9-3eea-4aad-814f-c04f5fc75073

https://lh3.googleusercontent.com/notebooklm/AG60hOrmu_SNR3HfdbX5Qb2-hZTLDDXd5U_1bCPK3uzBGoWIVUTvBQcfpNd2aMXSOOcTga2_bi7W40gqzvn5lGDHp2v2t23emFHxRpUXYa-i1aSegsli1zOVaYR3Is6zr92-6upgmMBa=w73-h27-v0

35cc49bf-3f94-497f-a78d-c179d4f85683

https://lh3.googleusercontent.com/notebooklm/AG60hOpqQlsUaR4Z85-fENSck8SX79CSdrCUOwSzs8MiiYXWGqXgX7Gnk00mjGjbURfTXEZEjAaZRHOvFBO--wVSA7M0R-w4tszxP6FtTyW87JVRkGEMYwkHPnMmR1J1klRPXdmOExCw7A=w71-h25-v0

67a73c85-ae6b-42cb-a1f7-fbd6d66aa852

https://lh3.googleusercontent.com/notebooklm/AG60hOrx1tyCbCeLAtsjAeTEHTTujEWWp3bidLzF9mJ73yRaBxcG0K72mfrDTuadC24feD67SDZ4wPKSAQdxOlhtLQj_pYdPPAydkSJwjAqO3KtpdRgx6rZKtzivKxGtrdRUKWF68lTCoA=w101-h27-v0

703fa666-bfe5-4cf9-96de-5f29ef5ec67c

https://lh3.googleusercontent.com/notebooklm/AG60hOp1T8JZkpNXLKAp2TQ2xG8BXWySWLkWSpfuEwee_9elUZfE_m15psx-BAd9DuZCJwe_mWE_u2Yfmg4nHmcBroB9sX7o0t0YB_OmeT61ig15m6T8uBtOGZ_c0tNcEt3eSxLOp3d22A=w99-h25-v0

4e3280ec-c215-4ace-a0dd-37560ee7dcc7

https://lh3.googleusercontent.com/notebooklm/AG60hOpAJLxTOVfOpVC3peKXrMBhCt_XdHOWzAqfXVpgWAG3Op_0RY_n5j3L9I8p1Z3KoYRTI88hFMCfaeLTFEIpPuYJQTtzmdl4j28roSgyn7r6C0EthyhM4IWGF9z9fd3tNrmrXis7=w72-h27-v0

58d8af9d-4836-4f3b-a02f-fc27254bc2e3

https://lh3.googleusercontent.com/notebooklm/AG60hOoEVgR2nztd4gFP1JqLc5ol_CUBjmBnm4o8FMSG10CwOZtKbWiL5GYDRPUVWXA19YpWfDDPyUV6_XiqpudCqyXRo3Y9Z-LJ21YBZDkTNvY5iDk6aTT7Ex7ztkfPPmyhvB-3FvnW9A=w69-h25-v0

9cca5fec-9df4-4be1-9932-e057e08a5173

https://lh3.googleusercontent.com/notebooklm/AG60hOq9LbkKhwx4ZZGY4h1RP1yG1z1eC1kc08Gc10wljxrxnrtyfu6IyusqxbR2ye2Uku5Md-kcPFSMjMFr5Gu8ITffdFCDKUFjrdCsL0qJtn-iu98rjlWjXo24TQnc99vMDAGkN_xhZQ=w39-h27-v0

c6ca6d57-b153-4d34-826d-a920cadaeedf

https://lh3.googleusercontent.com/notebooklm/AG60hOoL_rnYbhwtIiRuXnuLm0V9TtpYyi4DVZfBpa0QkDuYP12TFrfOn7UNwFn3sEaHOEa8MZwYlvwl7BUVySHGMwi6AEQmV7oHGMDTAe9PpBnlwVOin8jB9j_Fckk7Qit9QhdKFs7p=w36-h25-v0

af049c57-808a-4170-aaf4-c74686599fd7

https://lh3.googleusercontent.com/notebooklm/AG60hOryZDmjUwnOKTtbYRAsnoMObsAuhfJU2DEm9EAH7OVZuFOIGmA1p24RQHw_dBSgxGGwPhaw1x2XYfV1dsy3Dp708O3XfFN08UMexe_aRAbrgCRjbMwP5llE6Amxz80pFqq16TFklw=w32-h27-v0

a032f297-e5ec-4be1-9cc1-04ce11f2f799

https://lh3.googleusercontent.com/notebooklm/AG60hOoLnnoFcEQYCplOKgHU0FnJ8OcSDHEhwO9awW7SWuGO9IJJQ88FVmO7-M5cBuwMrVpsAvtRUQeJPU-eNDS0PsJiZWaM3cVI7kz-BXOjFy_qTtm7mRhuqM8aqStPycgiQDRchRMHtg=w30-h25-v0

3777e801-0582-4f84-8c28-29fb75c5bc63

https://lh3.googleusercontent.com/notebooklm/AG60hOoUAP0vIx1slumVg-yM3t_g_XorRmQbvSiEUXSZEHacYerQX7Bd4UHKjQ3rFzrjiTbkbDZKNUhlLbzrwD2MQeB9SD6PY6MOVe1mf8cQY-6b4V8bEuNqsnTdSRj8wVFbaWgcPCv7IQ=w62-h27-v0

55c4e8f9-ba50-4bcc-afa2-3900aa1edcce

https://lh3.googleusercontent.com/notebooklm/AG60hOrp7UJ90RuzLByAsmQq_agbeiVw8S9wFANWwlNf_Ka8yLU0CWhJxuHClAfeO9U9WN_kKm_t5M-q_Y3eVGJiXkwwrio3qRAt9b4k73vV28mhbrsR9fpHDD9zPdmYrsO2-TWjUpmR5g=w60-h25-v0

b0a0ae97-0307-460a-b417-039bb6ba7495

https://lh3.googleusercontent.com/notebooklm/AG60hOqhcIgTp9x0vwBpemxigwDjkQmpzm8GfXrt2F5bFZAsyy6U1h3XziTRd6pZWdFXPmEnIRfCqNEBrZIRYysI27XQEP5w_zU9ISniGj4OqmXub2KnVD_tYDaryWlmxZkM1nvGeC9XFw=w67-h27-v0

ccb63dfc-fc43-4941-beca-16eeeabe1c63

https://lh3.googleusercontent.com/notebooklm/AG60hOqKZ-7dZAvcR8jCpUe9HGOF7xk1aZQHAVZcciFgxwfL1CUVHqzqakl5saunZogZ03OCNOxqyo1OZyA9rKr9BvHFlcYj6ne6f_EWxC5GfddsI2hJseN5rMKRB4lSjtKtpT1BgBFtjA=w65-h25-v0

15b83b51-4be3-4bc6-8152-be468731bf50

https://lh3.googleusercontent.com/notebooklm/AG60hOou4jNta7nDHO1FeXXyKED_blO2DPqOyiRa2zKI9w1kAZ9SYH8OjqL5601OwwQpqCnCawqUvSKAoLK2M1McB6oBRHI51R5GvTrM75xWiw-sXwvVQ8ccfNS0tPtmKimpmnc0nUZsaA=w28-h27-v0

0a172958-c1ce-4a2b-9a88-6ca969cfc98a

https://lh3.googleusercontent.com/notebooklm/AG60hOq8NV5-VcL80mdYq1DhOPpS8kLysekhN5BaEZL1blNwsqMbg4nrjqhaihBRUoBwLKBpoQZ99ABl0nbVdLsbG_DY7vc5YXlkeXxuH48ujZM5ymK-XTPnBri39fZs2Pw8_rMxO2bRhA=w26-h25-v0

014ddc34-6cea-401e-8716-04f05ee94a55

https://lh3.googleusercontent.com/notebooklm/AG60hOqqqNnkSb3oyOO7zzCg-k0_9egLgWYZS2ss6dlKiuth-xD_U306bP11WJjoylywszvUu_VE0d4nbP-PTVL_K15sJhO_CMVwe-Gwq527R-jFRQq_atSag-XkZWsTTBgdN1KcYd9R2Q=w45-h27-v0

af0fd65b-a97c-4768-afc1-565cde6adb62

https://lh3.googleusercontent.com/notebooklm/AG60hOqOtqwsiIYBQFWXBfChhekBG4AxqDeIb8iLPBL7urZv2f_r7b4VfqXbRVnqSXV5V_QsOC90q__0s91l98iUDUnM7rTgiI30GD9vE5Tc9aUk1ygAq08TaxjhKHnhwnTcGFtD0ktiHA=w43-h25-v0

c2cbb9a7-7264-4da8-a79c-3304c51afc96

https://lh3.googleusercontent.com/notebooklm/AG60hOosYrg3tDuTzBP06XSwLg27vTKj-KND7IIFAyFAVTBhgjWlcpoQoXQmGyUwDQ59WU5nh7Qa2glocN3Xx3Taxea7uHeWv_-sMMc9J9ttZpej7CMDJBpDL8hIEjYQymZ1at21X9LDzw=w25-h32-v0

12c18ff5-84c2-4e3f-9966-e98c70888aa9

https://lh3.googleusercontent.com/notebooklm/AG60hOpWQrkEWQGJ9u5ZLQFtJWk5yoGGUHMwbhcvDGS2x3XADH71mLeFTFlA_4cNt5MPNlzU01R3StBqLdnsapJYyfWhS57mGAjfrmlww-2dgcK6doMccvBx_yoBydvfcAbhJKLAsAK8-g=w23-h30-v0

b5aa25f7-bb05-448c-af7a-13955986227d

https://lh3.googleusercontent.com/notebooklm/AG60hOr0EU-8jemiDwfjkjFyAu_D8rpvN-3bf9DvSQJZ9QlnYzYLo6IgFn5hldbQ1QspnXG2ZQIy8NrcjMfJkUResOr7I46pQgyH5MTjdO_vS4KZpP6gJLVX_7510g933-ain7i3ikd7=w43-h27-v0

41e5154b-6f98-40f0-9944-8b3663bd3668

https://lh3.googleusercontent.com/notebooklm/AG60hOp781nnz42_Gwi-qfe7Z5JY5pyH_I-xe5JRAH4l_15UXC_vUWy8rTo1kQpb-Y018U5TjlRs-d4Nf6eJs4i2oe6NJf7RnCKk89eAIb-nPEnfDSBh2WnJ8Od3NlpnnY6lKY0h3aeyNQ=w41-h25-v0

466a1df7-5092-4288-bf6c-62a8dbba1d23

https://lh3.googleusercontent.com/notebooklm/AG60hOopzk5k2A25JWEyqsRAYgkBbIU63SzjfNW28mW7t9zl_6wEQd77z2s_vWAXqbzIeAZqXcrTugdly8mLm3pYXr27prL1yTi1PDuOx-LFYQ4PUOZmFh-WrshZNme9AbVdBbIVxNPK_w=w24-h31-v0

4facb2ff-4336-4c8d-a502-1d167961df9c

https://lh3.googleusercontent.com/notebooklm/AG60hOoNeFL65G_mH0omQ6FmeQ9Qgss2u-O68aZF3CcAK2d3ZMnZSrcM7IRjzWNNddK-dY-hHLmKHoPEtFZUSzDRZznRSdtbbjldou0OQ0ZokGw5xTp21kBsWgQQLyljPXT1bn_Gl_V2=w22-h28-v0

35b13598-133f-490f-b878-984db685ecf6

https://lh3.googleusercontent.com/notebooklm/AG60hOqpNbnaKXVIm5H-LTg_cjnIbN43_KflVlMBAABTh-ORc68SdKE7uWSkBFLOQIKRX0GsrAmpPJ0c2Ayg3rNDqoA2oLZ67-heNhg-ScwXkT1H0sG5q_WyFS65oqzERCsZY2kyGzFS1w=w29-h32-v0

5776c126-078b-42c6-af39-8e885d92f5d6

https://lh3.googleusercontent.com/notebooklm/AG60hOqvs8LkVPmiNcFUZa2i5f4NJxNdW6OWNMyPlxtEC1PLfc3E8Rl3je5WUJX3unbvCYPDLwqYoKZ13I6Sl3Rt9igB_ybcVxuYqagUJkLU6w0JAyxj_HyzwEefFAOmWcPFrjp57KuS=w27-h30-v0

66653786-ad54-4bdb-8570-15b5df5b277a

https://lh3.googleusercontent.com/notebooklm/AG60hOpu4474X2Yd5XLFFR9mcsH0bE9kZwYb9-XqEcB20UQSiurgTlk-4SPHK1aZ4z1_1yTzO3LqshUAH940nGxOLdeuWJL3jYfHvBJ_yLerIC0WNgQZYZZz2-NVcTg9T-7VWh_IrnJ-pw=w32-h27-v0

56140d82-2bcb-4c94-a6e2-20b0a4d48f11

https://lh3.googleusercontent.com/notebooklm/AG60hOo1ukXFG4f-aA6Pf6CNJPyhQrAowyGxT1Osp8lHDkjrZfeUqbq6hzoadrmKlY_8QjYXncqUL7t1_s-EYp5cNWoJO3gdvUBXNB8ViPjqgwWw9yyk0uYm2aZLjod0-oE_ggGm8RyDkw=w30-h25-v0

9b7046b0-544e-47bf-ab10-dfc04fc0206b

https://lh3.googleusercontent.com/notebooklm/AG60hOqy8U1-6UqlGzdpCBh81FPG5iCw7dk1o-Yy8ScfBSRxJDcZmBmWHtb-LQWsha4Ta5wVbmKnnKIjqvhZWHPLtFkpq0VtpFPj7xwoYOJFiqqBMDnKRrfturGLFYFJISvYGvU6jq5M=w82-h27-v0

83d4f337-12bd-43d0-8e99-c388513597ec

https://lh3.googleusercontent.com/notebooklm/AG60hOpqPuscgBtpMkF-j0i1ds2-bz3yicIVHPahtW8O_1vxHt3IFeq0lVAvP0yMuDeGYHuPtScM-Op4BxmmKQOnQevs7Xhj_bJwWH9BOMpatHnHIJW7e0fZojwh9zTKTFFym-FaXMjHEA=w79-h25-v0

5eb872cc-f753-47cb-a327-c9b97ef07d54

https://lh3.googleusercontent.com/notebooklm/AG60hOptcnLLxnNk-NVJcuedJfrJJVJvgBfH2IVMfgPOlGdWIn0qmMUMxpfNHnHdgFFOZ01Y8mbx58qY65Ha8syugYxa_XAA9O8S_6GmDFEMr4Hpc4_k_tsqIM10hPEWW9yorgY5JJ0Z9A=w49-h27-v0

02f99237-c4db-405a-8a4e-c26100e699f8

https://lh3.googleusercontent.com/notebooklm/AG60hOoedl37Kdc462yj_1rjZtJ7QbnR7CYjekYtck0b8vkNeEdqDZmfLZwC9s4mBKsqX0htt9dxOQ9oEu1Ba3sq1cB330x1qXhTK6bh99WTGZUKDvhjhjUIZNRX6J1EQ4bBz9mrgFDN=w47-h25-v0

dd8fc56b-793b-4e2b-900e-1e2b634f0aa2

https://lh3.googleusercontent.com/notebooklm/AG60hOrsoyBR6T7oFwv0vBpKjE8QVS0t73ITSPl8wV8RwH-psyWsNzxgLSZnGPflVDWuQj-qq8v2NG9d7fj24gJorwwwKIAwIchDpsLgV5LOaip8eRtYhuePQKD3r5iUpOmGJNlJUm_o2g=w131-h27-v0

07e24adb-216a-461d-9996-7c4a4f8b2388

https://lh3.googleusercontent.com/notebooklm/AG60hOrTpv44bbWEyLJeAOBzPxC3Qa3DQ7jfYE74FE6YUL03vGw8H9xLlUlQEG_ZozFWz4V32QbBdSHUIMmXehlFRrBWFu0_bqEqES-qabjHxVeLUqPIRtT1TQ4x6M_63DYAtcpVbuac=w128-h25-v0

91d01ded-7be3-4043-94e0-46f6fdc2e98b

https://lh3.googleusercontent.com/notebooklm/AG60hOp4QE9fKhR1s98Ll9X_0wp8--zHb4nOY7ISjSGUrQ_Tr20Od1z9o-7VDMsYtRByYxlnMWjS-5zatDBOc1XQwl-31XbbOZw2DGJgI4kQ5r8zHmFmlqGkdxIWDE_5d1gmlPh8JV0u=w71-h27-v0

7f7a5293-3611-4ac2-95f9-3b8c9c8ea769

https://lh3.googleusercontent.com/notebooklm/AG60hOoVh8-Lg_NsDztIp3qfny1noKXP0RBW0-d7L8XtNzLQllWnrwQA7sjryJf_uMVP2fZgiP_mLNDd73qYu-WsUQ_0d-uiH-3Sekt3qRuBxER4UZskSaEomuQFGOHtJ8vp9ns6cyR3=w69-h25-v0

392812fb-c1c9-4940-8f47-63441187718c

https://lh3.googleusercontent.com/notebooklm/AG60hOpJTeDJ18-KTbriAXVcEC6LCnhEfV_mdbt5DM0BphGV2kBN1oogOB5WyccjB9b4x-2IRX4HImxkQvpM04I0_ICE5dOsCoLBx3l6nq924I0ggFrImnSw2iIyocysrIFtZrBu0KoujA=w69-h27-v0

8ff254f8-08d3-40e8-b9df-0e9a8832786e

https://lh3.googleusercontent.com/notebooklm/AG60hOocoEyIxXiMGzvWlvrUcuIDu5MUOjbQyoFwSs4tTuXoRjDE9GkADrecLE7e3eW0Xbj9BDEPJXq7mvZE3qj7LZjDWBizJW7P7Beyh-SyAq4TzOvC6hJzcuSRzkeGzlTji36_XFT6Fg=w67-h25-v0

3baccc3e-3858-443c-8644-6d059bd4fc48

https://lh3.googleusercontent.com/notebooklm/AG60hOqoxXpKkfljbEFp7jyWX6I4oIeZioVHZdNgvdOuVZgO_AWuaBqsSQZ7WlT97JTInAkSaUy0cIEmIL-eZgf2Wxu3Jlu9Pj_sTUSaSIzJ6CIJtBoda_c1xO05WpPKR0Lgs_Yf8EjOIw=w40-h27-v0

80052775-b5fd-4d72-946e-723e159d64b4

https://lh3.googleusercontent.com/notebooklm/AG60hOr_lscDOhUzl_fWjpoPMje-tHVKa99YiAKx3FnxB8SZ69nL2bU5IsyVamlaiI0dWYol_NdyzuraHLb_DoyoUP-9s3CsBcnwxAUuqpSFQuyCYlpsJKHNFea8dzJZ7E74a9eL89T7OA=w38-h25-v0

220d2eea-73a1-4946-99d5-3a087bef9e52

https://lh3.googleusercontent.com/notebooklm/AG60hOoNwcdDbVVlojOq9giPARyCkrz6ddj3lWnwdFBySdCfuDQ-zD1H_rFL6xys5GOd2KopCHb6FmiNCu5CAKN19kw9yzH5WPrRgRVgUgHDTcVDNR3JitBNMGkRzemu5nttJRaDbrpw=w62-h27-v0

b7d6f844-9aca-4c84-88ed-7319d0f10012

https://lh3.googleusercontent.com/notebooklm/AG60hOo108VD_mWiIRuoWCioHC2flfOm3Ht_FQrn9k8t6lrvSiYXFX96hsBJbJAroHmxFY40prEz490oZO-QtW0zJWFi2lMNxsbZItkfzWZD1qMh94czmPXUcLnm5Yqo2y3dKBrcfiZY7g=w60-h25-v0

faf59989-88e0-4d6b-8205-a222b7ade3b0

https://lh3.googleusercontent.com/notebooklm/AG60hOrvJPxs4nmC34vNj5Wf1l7IjDGWuzK953WpD6HllFlm0iookQ0J7fSxZeCw_nNTVEx64lUkSql_w_dOGedJA165YR3MF2vVs91c3yvFHOq2qDruRqDy_puszFWM048QzUkB6iIUCA=w53-h27-v0

f547c241-dd04-4eda-97e7-0e0c1637771f

https://lh3.googleusercontent.com/notebooklm/AG60hOoCodRtxB7RVZEjGQRu-VbaBnDPM6esL_z_ZQvRTzNjREjT0rL3-4n7BKaiQWBqOT-y9QzSJsmN6zNt_eGLAFqr6KS5l4cYqg0T3FxhXl14HK2immmP6KV9L1nF7MZ74yQiv1jWKA=w51-h25-v0

cfd46518-50d3-4bb5-9073-578f633ffe37

https://lh3.googleusercontent.com/notebooklm/AG60hOpfaDL_7JuBKI-Ll_ucPe1yYgpeBFUxCqYC-Wnw08N15OTMbn-USEOCj1RdzBlELVBf0hCrEzvfRNPQtqBV5Hq2AyxHk9lduBgR1MNtwl5tkqgM6PsmMFWTVNBUav3_hpVqjc41_A=w60-h27-v0

aceeee34-6668-436c-9ecf-4df4cebe89c9

https://lh3.googleusercontent.com/notebooklm/AG60hOp563wALSv6zNtCzYKdxw9HrZQiUXDps5ns1rzpjjDH6hlOwGNWqY08XOeo7nnZSI-2Wixmc8VXnIWAv6fdkQpUCC23F_nuzVpg3-AC1QAt-UmqGB92WsNlYTstO5XaQdYrife2nw=w58-h25-v0

3acb6fc5-24ff-4ebc-89e0-c4369054b11c

https://lh3.googleusercontent.com/notebooklm/AG60hOrj69WjESW8HFRZBEZV6slbRLQ3SdJY5PKRtiEV-4pw3ixriroXQSFIsRC82STpZFqzr_158v8yuzzCPCxqPm50chAOQ9GjtDH-4wNl6ri4MRa1W6eVSvTYOyKW6Ow1bMDig2fMSA=w87-h27-v0

6612694b-cd66-4dea-948b-ada837005f8b

https://lh3.googleusercontent.com/notebooklm/AG60hOqTmNXcKpZ8RxRjvdgIVF8Ce0gc1qkuGiZicWwxHe_fXle77Sv_fvmXLb1iD0UPlA-wl83PyIOW5f0J8nxTGQN8oLIhaf3TwGaFBptJ3P6QaVg5RB8Jf5_NaYs_ctwj-vfLLN3Fdw=w85-h25-v0

e9061413-2af7-48a6-af0c-6de542607832

https://lh3.googleusercontent.com/notebooklm/AG60hOpeLuMtzGXsCkEP8i7SnVKfJifiuY8pm2SqTssOXUxYdm188UKBANyVw5_thPOuNIvOx3rpkPOmUas4YvfRe066t4EeGOKXHUBFvXj9MBMejyLpzHslfE5IP-yD60-KWzV_BbfWqw=w49-h32-v0

8628b10f-165e-4030-82d7-6aa176a3a0a4

https://lh3.googleusercontent.com/notebooklm/AG60hOptmJJT23N-H_BtAC3kYoa0k8Lhu7QrzmJXub9_Wv_ZAD5n_ZDnhSQOAYkdjsam49OfTRThqU-aLuViALcnhvmwG_O2zL88lnKdPlsy_eRG1BNDjaIhnQS-fFZdXg6D9HhBuhqu=w47-h30-v0

eceda868-1857-4121-994d-e289e3198cfc

https://lh3.googleusercontent.com/notebooklm/AG60hOoRee2cZyRe0y9YqeR2E-KQUoTj-WPMEPNbD_Oes3x24rtycPkHFDpn3vkgbzsstd5B7r2U_4VemZCdxO-0FxDOk-3V5j7vNTzAxxHdwqROjWTuik8vR8Gx9Nk78lIsFalMmwIY=w24-h27-v0

ab986349-647f-4a45-876c-e9b23c606054

https://lh3.googleusercontent.com/notebooklm/AG60hOrhqUGqjC1g7hl4-QJxc2bqCyL3ZmWf3ud2R7g9ecf0UzsSWYzAmazPjaAySfov4HCfsedE2i2RUiqwQ5NHZhJWMt-SEMNJUWwQ0gl-1AQxGCnikkAhi2HIeSZFoqsQZwCLFAGGaw=w22-h25-v0

7c75a9d5-4861-4a33-a5af-d6dc15230303

https://lh3.googleusercontent.com/notebooklm/AG60hOpcz1pXkESZdChNJI5YHH4pitIPb5Kyf_bCMFYPqzVcQkM8t2kNZLKP9Rq2LgaKkowi1rQky5BWh-5ywB5-lasoCBG52f5wjW5AZBuFMxuombx_uDKEzW3V2rCEToYAB67MYTg1=w24-h31-v0

b5c2d701-1996-462e-ab40-5577f4e7019a

https://lh3.googleusercontent.com/notebooklm/AG60hOp5uZG3xUhcp_kxVEhYiFZx-h0_5-o9SIZceuDvZ2lPYuuhyBK5qShgPSLS2k8vDcyTtLyrEVMwl46QyptLaA2q7wlJg8UQJE7eNnJvrNpd_3Om7Pzzq4W8ge1GDf2qoE5A8HrQqA=w22-h28-v0

d47f2dc9-4a24-4d9e-95b6-022b15a2ac8b

https://lh3.googleusercontent.com/notebooklm/AG60hOrqGVVKvRcu13_pXt-tDZr--HOrJxyOaG8IOcQJ2vlvXttPQgxHPQPM0IwQ4KCKRtSa3hcicL-qxhnwCX5oxfTt4hRWrj6hO_RtVVF3W8UeHjS6HlG_49iEiV6Lw62ao13kbE1REA=w32-h32-v0

6790df32-7cfd-40a0-be5d-889f1f636f09

https://lh3.googleusercontent.com/notebooklm/AG60hOotoKWwPMImq83q4J8rEPmrMRqNWphbAzbzYKeV_FEZ34-sprzi-4YI_UxHNkDwvE6facxgIASkF71q_xLG3llIZ8nVrM0AuLw_12WMOobbmH4CJ0eWwlzsYJGROlECahpNh57lMw=w30-h30-v0

2f492ad2-bace-4685-bba9-1b1a2e74113b

https://lh3.googleusercontent.com/notebooklm/AG60hOpD6Uc2rjVEcFPvmthjn_QypVa1nEAd4wOeMyd2aJN48bWyEW5fsm_ieVRTW5cNx9-oJ7zCl7Lu-3FoT_QdSZZu_32IXEgzb21M6u4dND_J3wnjCUQ2DgqnYhfo1S9ybJZYnJi3Uw=w92-h27-v0

2437da8c-acbf-46a9-a3a4-81c2cd5b1b03

https://lh3.googleusercontent.com/notebooklm/AG60hOq8xbsWNLnZbK4itoAqFrRz0wZY4EtZQs9UxGCFVqBP5jQxjtafwfxRiKVC8alxy7iCOw-sgH9cbdWE4k8W288vqI0MEvPgiloUJf6wI3OliNrQLiHCu8Dd5fLtQYXq8EMl6Nwv=w90-h25-v0

b3d0f862-74bf-4dc3-8ab2-63238e1c7fcf

https://lh3.googleusercontent.com/notebooklm/AG60hOr8X6VYeQr1IBTx8ABvP7AwYBZBwURnaDeeDtfBPRDekQWJ8g8hCv2GFHTg4TdRxMJsnkfAN5AB6AwbnMs8hR43oNYNA8EiwA2tudEpHuVhdvbM8XWp0iQcipbrKaYBgGGH81Kn=w32-h27-v0

ad76df6f-fcd7-4356-82ca-46f51299dbbc

https://lh3.googleusercontent.com/notebooklm/AG60hOrUtYmKQV-R1M-OluRaFhkJHJD3Ar06-aB06oKHmvsq8h0Vz_7hmkVbWhW98DxKNtv1H8sFRUR9v18vtxZK8M8Az5n-N8KTFhX899iRgQgwzGSU60VuV5rvmIkwn99hmsloiGxvkw=w30-h25-v0

e3b32077-9a10-4b20-b0b9-5158e39bd3bf

https://lh3.googleusercontent.com/notebooklm/AG60hOoYjdnDAqURqckIYljNWntWWlJFekWNq17_Mi7_8I9LjSkbngvJY8ua9EFbPK-bqRCIi06I3sJGmOlRxhKCAu8FELIS-kgDZz8UQDeR34xvv-aJuMGYyUhTtU5bVl0QVxJi0Gs1kw=w54-h27-v0

4ebc67dc-520c-496d-afc4-c504f758c0db

https://lh3.googleusercontent.com/notebooklm/AG60hOqjOU3BU8QmotaGOZJmOMyadN6UqcsawQVj1TYTgQ054NE9CYjdKrA27DAFtMih4k_CXEQb2Dks543ig9FUhKDrA_2y33jOiB_DuvIanp4DdCk2PvSnAfMrnf072yCBg6zAbd8G=w52-h25-v0

783fbf95-426f-45eb-8b4b-aca3132ae3ca

https://lh3.googleusercontent.com/notebooklm/AG60hOosSL8Vpt14W_J1eocpnajOrKNgQPe303lvZ2W-GQYkUQMww9I8Tfm2t9QfrzV36qMh3uj8Umui7jWmf5yCAXP7niVsLCE7PnzPouMzrdEWhbZMyWExQq8pDehJcw9DRok55mMdFg=w62-h27-v0

28fd3f50-6c8f-43cb-b8ba-01421963bc64

https://lh3.googleusercontent.com/notebooklm/AG60hOphnPiU4FoWCU01HB5W0G8A6M4EM8BciHMRT-eh0HaccMWe55zNnDcsmXS-awj5fCkQAXj9DU1FWbco2ejYhi69rCyEpLBXKs_Hu9-rKij6AHLOJmC4ZJWzslh_ZAAoznS_Ip9AvQ=w60-h25-v0

98975909-2f1c-43af-a997-73bd27788eda

https://lh3.googleusercontent.com/notebooklm/AG60hOqxD4pnH8a4KzOBBEGcRt-OPO0VhkTLEsODkCXVtc-lwyL1xfHjN_8bcnKXi2xEkx9Ak5Aq5vneogRpnHyIqdrzkBnL0_il_c5vsqxZ8_xFTQOikmTfWpT_ZUe8BkWnVDZ4iSXThA=w73-h27-v0

2263b345-75c2-4f26-8d57-c26ea6a4ebfb

https://lh3.googleusercontent.com/notebooklm/AG60hOowN8Ap0f3L8msEleWcj6GzpQ6SS7ptIjIkA4vjfqfwfTMC60g67_Ctp9IKsZUaO7SGUsxNokfkXnrH9-42StiSU5u3kuuJe0XIbMXVB5nbs1S_mWxWmruNoblXV_YUFkqI8wq3RA=w70-h25-v0

bfc0a507-2345-4019-9bb9-a2a64e381a20

https://lh3.googleusercontent.com/notebooklm/AG60hOpcT7sHf5m99cf6Z268z_l4B0oHM5QUTwxCvdM5-YoM0LW6hcwJX4s9jFunT-k1mA8S_4f1KbCIGfnvaGtdGHQP5L3aLvZbdLSu_YoCYY61xaWe61DD9uURFZDUF3qFwGxuRelUGg=w24-h27-v0

8ce86260-c22c-4efd-bb10-58a956510c96

https://lh3.googleusercontent.com/notebooklm/AG60hOqTeaP6QAIAwWRegkY9Ytnyqn3mqg8eFIIWTNn-wVgxCpfgcX1m4G4No4WoMXUS8OnUUsl1hUOFpWQ4iVcHJ4ZohtownPxQhr9bP1KXgtH_IObtyKRN988jK_v_mQDKMmmiDOsJ4Q=w22-h25-v0

740815bc-b7df-4d20-b4e7-65d7aa4863cc

https://lh3.googleusercontent.com/notebooklm/AG60hOoBrurx3WU0FNVxQEPQIHyFsozeKg_MCBAjLQ_owPkMcVMc4c431ncbuFQNnDSaiP6yeEEOJfD8ENrHrsOOCjUCzOSTwrORK3OKtZ4cl-uYfJo7VNCWIM6CyIjjiukDai32Z-cSgQ=w89-h27-v0

86cf37c1-d5e9-4fe4-92ae-74a3bc3b4230

https://lh3.googleusercontent.com/notebooklm/AG60hOqgRcb1xJNUmut7GyxA1O9ewWYna9lTWeQ5e0bqDlJkYyrFAUXG7j33Nun0J7XGXpCrja0yYs1-T_PWoTl7SGP-XUOXlP0bVuI12ZcuOWuvLpvW2gr6MQEVypop33V3yBaDbOL_vQ=w87-h25-v0

b768e698-c895-415e-bb15-55aed577e2ed

https://lh3.googleusercontent.com/notebooklm/AG60hOq3Ghp81T3UEGiW6PqiAN9Crv_yMO-gLwph0gbvk1UE5H0o699baZxaHfE6f55grJqJFeV5qJ9bi5_0j_D-1rBQkVgi9H1B3GbFNRlw0RrEGD7m7UpASmpNa7lpQdfAf0ggozeDrg=w67-h27-v0

cc50bdec-4bbd-4628-9911-9f84a939d2b5

https://lh3.googleusercontent.com/notebooklm/AG60hOpcOwFCnevPZvgAkEg3N4D0VsOv71QDIJsoU9ilDFQ-x6e_bp_kSNIZ8C9ZUi4zfoNivpoVy4-zjcE0AIV6AsIetL1A9lE0soFNYvg93rf6h28Wec6Xt_vua2xF9Ji_C5ILmxIKpw=w65-h25-v0

9dcef25d-80cb-436c-ab8b-8f0a744d6bf6

https://lh3.googleusercontent.com/notebooklm/AG60hOouUJzJfc1PMdRNDKAvMmKPuL7ZEqKN5o3pyGmiritfcFjtAUBusoeAabwth1mRP3ugWT1jvMFbfVWcTHRyzsM3rCkE4a-DgqPOwPcEkqM4bQmFfDnA7hPceV9-jHNkDc_A2UtO=w49-h27-v0

24facca4-1e97-4308-9628-5dd9562b2b36

https://lh3.googleusercontent.com/notebooklm/AG60hOqVGH1gISpzvuueu_GvkvYs97jCPqvllgGQofnsnXOHVF231WS9pG2CkUa1ji05kaDuCDbxhL2m16axWn0CwqBOnmmb4S_x-Vuwagf-lhLQ2IlvbmnG0g8EjQWs8E1hYD4ysOrbzw=w47-h25-v0

c05c9a3f-2cdf-44f9-9b96-6dbf23013715

https://lh3.googleusercontent.com/notebooklm/AG60hOpoJqmvauwPxULHSxR3B5m0XbssC_lCIKfm5Q4GKuybnwzLfh6jvDja0diVl0IIu9AfYRbSoiKlw00rnEfrM-8fjytxneDWMp-13wf8K1Go46O39RjwFlp9B7Knf4LsVmmmqCle3A=w69-h27-v0

40f2a253-aa7e-4b24-876b-bff871d1fb3c

https://lh3.googleusercontent.com/notebooklm/AG60hOqzG6sKiiBktLVvFj43g8vmsPcCI07epJJwTJV4FL5spH863wmgGGElQfLieHN4D_gU7vc-YBgWSCU8tfBQA2TMxOYBVFMuMJYZV_16GN_39eme4StLI1kmdv1Mn_Vd5eIANbOnDQ=w67-h25-v0

2119bc6f-12c0-4f0d-882a-6d481fb13479

https://lh3.googleusercontent.com/notebooklm/AG60hOoeRjiDGf4ZaSvss4pekm8ytdgpi-3IlC5ZADNGoK6pPj7r8DkrWIursOFXVK1Dt-d2l_F0VseofYt0KGNnF-OuL4gIylhztKPfTq_JMIjYJvQ-Tqi2WMRS66PtvX41DfGTdH0WHQ=w68-h27-v0

d0bc16dc-9018-454e-82b0-ea01215bfa56

https://lh3.googleusercontent.com/notebooklm/AG60hOr5reDqfbDfU1UCG9BcvhrRUPynnAf0SCdim7afrwowIiHkiFJAs27a9JeAXG7rszI-RhSNEMwRMUwqrQlRZPH52ne8A_PWUtUH4vNkdKJqF_eL5IboO09W8iLI3AtP3paiLeqESg=w66-h25-v0

f70140c9-de3e-4ec1-bec3-53768896dacb

https://lh3.googleusercontent.com/notebooklm/AG60hOrdxquLMXuNWb9X5d539Zq9Bu-lbahEQZnHWkh5J6_ViyS7Bb88iMF7UZy6Xkn43GzTnK9TY6obSntjugvagCBh1l7LSFC6gKeSQdDLpYAP8JhBoUR7afHYYKlugRFbHnjWIOLxGg=w32-h27-v0

6d8a6921-8871-4ce5-93a9-a69a81df83ba

https://lh3.googleusercontent.com/notebooklm/AG60hOp3f3IqvSohwGJOot-hcbE7-Gbv5gk-HPCpTKedFNI7h0pjiN1iUKjl2I0MJ5A1Ndc6mHlnCUMS-d_Nf81OVqdt9sTGV6qRILp7H7XBwkETdQYzJcr2s4pH3vPAuVIZEz_C_nWj=w30-h25-v0

247838d5-18b4-4567-a9f1-3484fe290fd2

https://lh3.googleusercontent.com/notebooklm/AG60hOqyE9SgahZUS9HwKlHRs9_AlS8wDj-w9MS-0TjGb-S5umoI3ouWw70GLr-LBWWeKAXJHqSz6bPchlvGcu1MSfDbLhxQUVClJrT8FFhM6u0ycTw0uq4zSInDzZHqz68hPxiRS3xO_g=w62-h27-v0

bf975db9-f084-46b6-8510-2a7e97508238

https://lh3.googleusercontent.com/notebooklm/AG60hOrPmGnQAZRmi3Ur3QwGAw5ncRs0bERK-yMunjZwRP2v3Y3Rqg1_kAn6GvIp6jEbc4QTAjQ-dgMPLkjvCH_jj2ADep_lehMjeuTVyxZeyd2Z_9AOCiCCmavlLXy5scfpR3iRcMZz=w60-h25-v0

ae66be68-8c6d-45d3-bb14-e866726403f0

https://lh3.googleusercontent.com/notebooklm/AG60hOqTNcJZl80unCjgVVQkhIvxWvT9bi21V87zyKkGYKV2OWMn58c6e3K2ueHMlqoWIl2dsS5k9H66w90Z9SZ-90dSeabu2Zrd2rIe8_esv0Aek-mU1r5zYW--Fb8ik8Mjgek6F91U1Q=w53-h27-v0

654f0495-fab8-4120-80f8-b46de1615fb3

https://lh3.googleusercontent.com/notebooklm/AG60hOqESWKba7QWdoc_nJKtFz7dwFLGIdMs6s6Tqhm1vZB2HDMfj_i8UP3PfS7-_lTCvDpagyuKvBoe-V9XRGEgYKGmEv1kMsQ7mUf3efNLBzHlwA-I8ZVc12P1ZqnuTd___L_mZtBC=w51-h25-v0

4564bf05-0522-4a7c-8c2c-f8d9ec88fcf4

https://lh3.googleusercontent.com/notebooklm/AG60hOofS_Ei-F2lGpYcHARTSZZrznlQdwdcs2iAPu0pFXV1fuch5LcbS12ql0mTzSJIRZZ3OnqfY8VFKjyz8sw7PkGqTrHDv7zeBFT8NXEvT9ffq79vzrwNvvGhB82dtOCvkVts4mLa=w68-h27-v0

bf6540f6-4bae-465e-92bc-b754066e2e64

https://lh3.googleusercontent.com/notebooklm/AG60hOpZXtBYePnCt2-A_HnIvJ9DmjdX8ZdXIe3gyCcG66LLlVX2_CEIiQKkYYFCuLe7IgsG8xw89UQwiQcJYRLlSyMWBwVdeG54607cfWlRDGvRPuILHCG_AvBp1Y1cHtusBljGPwno=w66-h25-v0

60b62b15-6c7f-41ca-8412-e98d7658d737

https://lh3.googleusercontent.com/notebooklm/AG60hOrPx011h3Dx6_J2H24FFnr9f5sXLLtEzjDg1A0SWASnCaIQG8-Der4Nx9Wer9K1pMxqAeFnwIOYtpyNIyzleK3YPwJ9PEHWngnDWnAOeqWyqpaHbUGWUyAqnUZDDIg2AODW0L-5bw=w28-h32-v0

219edcc1-a459-4d58-ac17-3782b6b3ca01

https://lh3.googleusercontent.com/notebooklm/AG60hOowaPl2O369YZdpu1UFmwgcb4kDMVVAZsQ2_se-BfAKsx-CN2AH5xR49p9wGyHPxjl1W9rWLVyFmddF9osV5gT_RUjrcS9nqo9amhXb2UMejhZc9vFmseE1LXG_b4f57M6hl7QY=w25-h30-v0

2b54ad52-d240-48d4-b3f3-1993c8c79031

https://lh3.googleusercontent.com/notebooklm/AG60hOpfzaVGevHMxMck8nugdDDFbYvfCcCkDBYGOelLqAlD4wtratjzBcDc42SQBRu0vryzb6ssVHgVsIhrCUKW0URn1vzRaRniuIOgErnE3mt9PGsptoOeAC0HywV-oeUIwMIJTk_k=w210-h125-v0

03d4e779-bbc0-4249-ad67-4d9d6d6b16a8

https://lh3.googleusercontent.com/notebooklm/AG60hOpdmJXg4UH2Q7-KaZbVNONjahTtYQHkRJaeTd8-gpZIRN6QG_Z4vANpdwm0p7VE9QDAmC__Ude4B1Y1KnHjDJSimp9AKvWMJ8qikBeBps-rTYfqbFnyXDxHYChp_hu35OZHh9kD4w=w211-h131-v0

db9b1fd3-1c39-4d6d-b576-238dfa3900d3

https://lh3.googleusercontent.com/notebooklm/AG60hOqGMcy4JMtRek043OU7yCqAUtZY2dK2qgbxNRQeMAEgP6-7GO33x3W-aGnknHJ7On7C38lKyfgBLBVAn5jrHo2QraceIdAhv3KJaV75F3-ISnwdWmwimm1pCX6rSOE1vDBS2km7=w116-h28-v0

302fbcfd-a838-4906-a51d-f17d4a2837c6

https://lh3.googleusercontent.com/notebooklm/AG60hOoooirL7Qk_i_hFq6uw1hCHeTCsqdb4g1y9qN1kIdW6ZXFd5mrTxFre8Gs2I_tkj3OoYEgdPcPesffInQcbLHsetghXLC1E9YLqsc7R__Fh-Z8yEof9D80WNJmsY3MvlKh1ezhZ=w29-h36-v0

d4ba7354-a5d1-463d-9d0f-789c3d72bb1e

https://lh3.googleusercontent.com/notebooklm/AG60hOrUIqSwGrYoC1Xs4N2sD5BA0OdJQgluSE-kbLtiB9_FZ1uFyKQpsKa2gnxkO73KwleVnAQYMnmSNzk-5ulf9IHt2MoQgUNYtdxnJwTK6ulfD6qteZSqOom0RA2FVY_FciIvdhxd4Q=w26-h34-v0

2c8b07fd-e473-4d6d-8c03-0c2e93fb9538

https://lh3.googleusercontent.com/notebooklm/AG60hOqytWoeTwFDZIYrb9JxG6NfGWBbqoSAGXyY7nB-rz2DSU8ncFuXtoxOlyqdSLDyUhKxCrNOGTF5xsP3pIoQyyQsHDGXqLDEwz-t6xh13EsLVky2tgxyerTbFhVhuy_yyxOC_d1Zhw=w35-h39-v0

25aefc66-7277-4e07-b9ac-2b383d75dd22

https://lh3.googleusercontent.com/notebooklm/AG60hOo4F_-ZR_49tzJUzN65oIJDB9gc-ixmwzg4DqM48eE2p2qETQo580ADMpZI_wE8ASqFzX-WmN3LBME3U6mr8-T8hMqNVRiMuwBOiQOOeji422icy6Gtn-NNGv_zOHdJ60oQbKodkg=w33-h36-v0

ac2ea442-e5b5-41af-bbc3-69e893e31e57

https://lh3.googleusercontent.com/notebooklm/AG60hOrNKhoJqiDQFC9tLbe0_BF8BGXAVOyP0HeOgQi4QxlYuOt8sQLH0MvSs0n7AdqIWBJm_dnoX3Wr_O0Tgn-pIAdtDCJDQ6tSzLXkjhCDzM_DTiR7qBrA2lS0EWETtiNWkxW_zIFq=w30-h32-v0

1cff5115-0194-4be3-af87-b19a14eea7f8

https://lh3.googleusercontent.com/notebooklm/AG60hOqyKAY3pmFQly0IRzn2y9jx_GLVqbMi87a6D_XV6rtxLTISxjTGeTWYR8sRIsTgEFSLnbNp74z2ZzyVyad59pDUX0Gi-bLWsUD7u7FPW_xsUCfKKMU5CzA8dkk6qrSr_ldj_Va-=w28-h30-v0

cc393d6c-843b-4271-abc2-fc1a144b8311

https://lh3.googleusercontent.com/notebooklm/AG60hOoz8Hd3ziYYEyIIyhurH-Ezrq2E9H9pS2Nry4eooswjbQd2syeEyh6s-On1-S3mCbu2c2tyxCY0qqvEgMqgAGslOQE4sGGC0_FuN6hY96H5KKG-EE8WEWdi9gCgpdPyUKwo0MMtVQ=w45-h39-v0

6ca1944b-94c3-4b7c-a7ee-0f5714c6157e

https://lh3.googleusercontent.com/notebooklm/AG60hOpwoknDDhW14YTV06USeuZr9CBRxP7Exj_ZFqH-7AeWHf3l7wJ9LKDtE7eE3t8aZ6myZzdM84ioZA2zF30bI6o09Lv3TIqmjG2fCbJzVJZQ3gUv__whiIW_rLK8DnOhgHTqmBOZYQ=w43-h36-v0

ece8e987-db6c-4f5c-8ac6-b22c25fa49ef

https://lh3.googleusercontent.com/notebooklm/AG60hOpnJzGa_2e8HOQZTNoMhb6IH9uHUMSFxKDWqsvPf_kne0QVdJCAbYcsHd8wuC4TQT5fSaRZVaMR4oPVSj2i8Z6NQl9tQ9QWyd_FMhMUguiUKYowjchEVre_zPAaGs0o6zqzu9ywPA=w105-h32-v0

775d40dd-5eb9-404e-8c4e-9df5b377a148

https://lh3.googleusercontent.com/notebooklm/AG60hOo_FJddoD01E76Cwo3v7MX2FRbN7pwuqDZFHJDCnQILHfL99QJZrATpIYmLbNsV1KOYt94uTIagRkxfMLiW8bd_NT5AaNy3bNLHG6JYN7uFYM_WD2s38M91SKjSyYveZwOPJtcn_g=w103-h30-v0

44cf6a05-8bd9-48b6-bf88-13d4eede6473

https://lh3.googleusercontent.com/notebooklm/AG60hOrEzlKC-JLM1-y9r4IiKjDURQFPHMTA6LBOAST5l0Sg8B5dN5acdzTgnXIW0GG8g3LK08U9hAxVF4aCzdZvIp-y_8ZlJmvxd0OSu0aRZOB0JkLXMx9LVQ-Cwtro8iT2qFoBoJ3fUg=w68-h32-v0

072971be-1569-4515-8203-372a268c47f2

https://lh3.googleusercontent.com/notebooklm/AG60hOppTQgw3lbfILXF8xRnwYxltPxog0LJnUItRZKbA8_0b9eEV1ZhyK2MlTjyqyvyk8alJj2joT6JOnhTXIh2dbOVBFwcwI4hGpNjInnUiM7Q4adQc396AdrfXYXzod-xOaSwlVKx6w=w66-h30-v0

e582d4f7-749e-4225-86a7-28234439fbe2

https://lh3.googleusercontent.com/notebooklm/AG60hOpifN7fyA-hkBcwfyw20T_F49_PBjrVj7ysZ20V3wed7JOHrh5f5jyfE71Ys-orlG7uxPcMCIss-MMO_KmkgT3jpqpD8eRZH18WWm5udmgYmznzXlNXt3Has5JrzDBZaBW1Zn2n=w40-h32-v0

7dbc88ce-c679-42d7-9e48-86c624f3b7da

https://lh3.googleusercontent.com/notebooklm/AG60hOr3AzZlknl305gxbM1Lc1kZYlORL8xy3a0dip0A-0kjoJDDIzlP7WzPCfoFCMq-mUr6tTkeXVkz-VuDQYKH838LQmgGgtxk7JQKJmPfsgeY6P82x1myjbyoQ1apNM9xkToYw8y8vQ=w37-h30-v0

e9cf54c3-4112-4002-8a8d-69f3ca2151e2

https://lh3.googleusercontent.com/notebooklm/AG60hOpWmLdkwZrqopdbt0byV-d6bPaI2tszH5ZrbT8GwFZ3Qv8UXRD3Plrb7HHBNAn98xySjWHdC4y3PwJbfy8vIFVSIuWWwoA1PM0KRXnYDcOg6vmQx1cijl4l-GbJeNn1awI0VZS7Xw=w46-h32-v0

d8d0e4e7-3b85-4450-a252-155e408734f9

https://lh3.googleusercontent.com/notebooklm/AG60hOpxnwWQ-oa5VqEpb75XenAb2BJ331QSG9RSR5gmt7gCKkcZnZ7UrxWCd7a2sZJ13SGmhBlmuTljbMWbOhqCWDM8m5J2lRgzKodhO-4rImyfIargfvgsG0cEcmnkk-WWmptoMD1f=w44-h30-v0

25ec5823-ad81-4799-80d7-6c04d821e832

https://lh3.googleusercontent.com/notebooklm/AG60hOpK-CwqH6TTstRSKvzhcluZfcY1LNUsCKulc2NFj_uPI8iKyOJwSXwzwKLiZ6DivyqyxN4VRS8GPPexIJMaH88CFjOKcAZ3kfxGemh9cpZkYK_IsboHKDGIKIWh-bTj_9_oo3Vn3g=w46-h32-v0

b90681d2-9d95-4554-a94f-2a1deb4cca75

https://lh3.googleusercontent.com/notebooklm/AG60hOrIXDa6eZtoX-KlgjBknLIuE8AaQsWYkFHpOLRJnpjEWyDXrjyoWbk_flG6oKtvnYdDvwdAzm1JWvH1FIAHoYuRXrkMRuwMyi18fVOmSyGu_HFFYBpnYI86wUQmXniZmlRuRjPE8Q=w44-h30-v0

fbfc1173-ad04-43fe-92ef-011b437703ea

https://lh3.googleusercontent.com/notebooklm/AG60hOqmdSEBdwaXQIy4omvBXpOa6JC5UmZgrzahECtCP-bG1AbkbtjUzAXjCHMQEnyvP1YQyb54pjkDW_oiwgGOvR4XfpVUHHajZae9nFfx6ZT-cPAEgkcXW1571vxYm_fFSMD82PcB=w33-h39-v0

53c3c874-c095-4915-b71f-ae7e4414a7fc

https://lh3.googleusercontent.com/notebooklm/AG60hOpztIYkeD_O6VNOap5XuqjYNomIsBddTwH38Jv54M4CMolOPXODuhD3c2QaHJJ5VGSPtzRGPh7ROJ6n7b-ToG83OfcSPpBJC1SZeCX0lYLYRJyxODoqrOVX1Fj-7FbrAIMtKj2cww=w31-h36-v0

641b3151-44bd-4207-a148-ad55bf079576

https://lh3.googleusercontent.com/notebooklm/AG60hOq3XELi0m0VLCYy9EKfMPzEVXxjeFGtsQZgh3NlNacOYGi55RtIcotPjFv6mOMZgdtm_bBlukQB3BU-IiZWj4Be8xYZyg-UVNqnJlhkdFJyPrNosFQOeYrFGPJl4F1y8aqjiB09=w86-h32-v0

e1a48e9b-34c4-474c-a817-2cde0064946d

https://lh3.googleusercontent.com/notebooklm/AG60hOqEYO0TjwiftSuOqBxHp-LGA79BEvZj6M4oGm_z1wIBWgYTOlgYLhr2hGJcqiujQ2QhxaPDqrCXKh9eZwtmfVv657wDtwL4d8-XhFj18a8wo_RkxODkGKefWFJrDg9b8E1-_A0IoQ=w84-h30-v0

9199eff1-0baa-4611-a154-95007914778f

https://lh3.googleusercontent.com/notebooklm/AG60hOrAa66PCR8BwgPBhgV3je_K1aGs8jDWLTiZDO-Vyd_l8GxMHybuhnh2KAeDIq4HC0x3w51B5C_Tis7gJlErZQ25Zk_i31zRI4y1PEXYdy-P5uOxNEngaEjZzgjCX3wGMNArTYuH0g=w70-h32-v0

62cfb11e-43a8-48d1-bb99-db03ae9b4bd9

https://lh3.googleusercontent.com/notebooklm/AG60hOrUss0dT0CvBX2tz4slTdloi8dPzm3H1TYeSJTnAd9Z2x22dziYrQ6gEfdjJksmE3iR-06u6IIUi3D4EBPXuLV0l3Dgk4zMKi4Fr8KmcUpgScwJElRMCXlDJ4yMaFJFiCGzKQ5Kdw=w67-h30-v0

6a9a564f-3d0a-475a-b670-2d1ecd361af7

https://lh3.googleusercontent.com/notebooklm/AG60hOopKyHEvTBSUTBgXtC5D0R98lejtYo8chylQa-r0mt9eZNJc0os1P2q1catrD6fK-m0Ye33s21NKiJDqPVblXPo8JIsUP09CwEPOg9pEKU1uiVzEygfHONsbFz2siplUE3XGCeBjg=w67-h32-v0

f8842813-c0c1-474c-9c53-53bad8612ba0

https://lh3.googleusercontent.com/notebooklm/AG60hOp2vn3je-9yiyVKdqNzsjLSR-rweDpL55kRznH3M6FuNc4vKXR6C89YVHYwBxBuV2q47mEbtk_9YcPPZgZCa5TUZIlP7cvOmDjD_5SrNMvsA1xXXtn8uPDpUTWcXthHCaAvjbVYyg=w65-h30-v0

185631b2-3e66-4022-9d8a-d939f99117ab

https://lh3.googleusercontent.com/notebooklm/AG60hOoEZ7Lp6Qf27kpkaN1dq6vSVv4ztdMk503lX2YlSK8YmP9bIs-rt83vyCiwFiKCmPlqjmJL07ZgY6sYN0wCjrpQK27n7UM1-wO70tF9U2KiOlIe5_1wwrGpaR10UVWBEC1gnFiLsA=w40-h32-v0

f27bd888-0348-4fdb-ae3c-e60f0ae09c9c

https://lh3.googleusercontent.com/notebooklm/AG60hOq_FMCC0a5vJE5e4n2L-Kd9IFChpblKg8fF_vhsvwW2KKOyopU6HugL7PEWUCKO9YabUmO5GEjgC-44lPHWdHz2HywAnCVoUvFutwARwR9o4aV2lkBj9U-u3P7TKls_gfPx1SRTjQ=w37-h30-v0

2d35d5a9-65f7-48e3-b568-c09d04b8a880

https://lh3.googleusercontent.com/notebooklm/AG60hOrt6nTmboOy0cgO8wxl63qSXO4BiEqlWYn0SleTKuie7ZVgXD7dmrVIHdJt2JqLZVLUs4ETG83-IE_TjFTUrfydYHtO5oYI5DB4eZU_7ml52GmPa_3BZKZIkY9nAJzu-AZvmqYg9Q=w46-h32-v0

447905a3-b39d-4aff-b5bd-7800972a176a

https://lh3.googleusercontent.com/notebooklm/AG60hOpZ62qCImxpbOjJzLBGNQM3MWiAVjXottItkd64RU4lg2tfZGo7FSmxPDaCLm1EY0TjM_Gic0XpgjeBB-jHcNHUZS4rwE9H_B2GYqgCUnjdPyeM-m-8KsgPnFDgy-cl_EId44401g=w44-h30-v0

f6e5293f-9a69-481e-90ea-da64bd252bbd

https://lh3.googleusercontent.com/notebooklm/AG60hOo5EMD_dMSb1db767HegqynAk55JSPRKs7H56kz29AB0IqltrY8FuSy7xcSp4NXt_NnBXn0JCFDGuAtbHwpDKGJZRGWK8pCSokuVXNq-kh9157_qH1sL2ppBzJyAOmWPR7ZTzJHoQ=w81-h32-v0

c887e0b0-a296-4553-bef2-9a5182e7aae3

https://lh3.googleusercontent.com/notebooklm/AG60hOprzL4j4rQGJZY-wPAp16oiOaOJ1LeWLmRlIcX9tqcaA3gUDCs5K8wYQ_suwJShRw85eoCDq4LKfTkTUgPLlRVTCfWrQYl9w02JYp5IniXn9d31WI6l8M8BRhVCRguiXBrsPwaEYA=w79-h30-v0

0c2704e1-c659-42e5-939b-d4d1e29d2010

https://lh3.googleusercontent.com/notebooklm/AG60hOp_lkLxy5wljzOf1_ktPNg3AQzIFRfgMeKMWViax2sE6LiRURxjaDX9XQTGMPDkKNwvgn7yyc8ax1G-G751UIcIwT1HxRhyd0DZZFgA7eVwiY8Ah2ygoeQPSJHUBlOgOzpwr1yBsQ=w38-h32-v0

88b679c4-c13f-49db-888c-757280122216

https://lh3.googleusercontent.com/notebooklm/AG60hOoEBXUTkBn5_jXocioY4c8-ZlHCmaiFm4eCFvPb4LM6cusLT-x3thwNscwfaz-fy7AEx1gWI_3oPGrgKcVNLMWWyMmANg-_eIcIizSx3sBtL2sTJPk4sYsotD5-63CsCwC0y-1YKA=w36-h30-v0

0818e586-cc0d-42b5-a614-54aa66543005

https://lh3.googleusercontent.com/notebooklm/AG60hOpkSa-7z9fNxMSjdQWDfKcjI6lcjf4PRkoT9SM8i7EHXsLwqHHt7hZFewnS1cbvVGJ_G-LV4FOosJayb3QrreIzXe1AhrATmQ3tlx0VCD850NrP1CHW0dJKjcVsCyda033pwYMJCw=w46-h32-v0

27c74330-fd26-4bef-90b5-668ccb491453

https://lh3.googleusercontent.com/notebooklm/AG60hOqJUPeTHk0WdV7Wg7ovMxwYlASCFLj3zWtOX-IcJ5ITNyDkw_FpnUA4BELh4LxHc61GHS_imPgM8C-k1Ktt6tNgZTgGS1ohOZdDJcYtZnMxRiExR2e9Iwx2n9GWBlqwpSLJp7M56w=w44-h30-v0

efee7807-4dcd-4271-8f2b-de820f0ba86f

https://lh3.googleusercontent.com/notebooklm/AG60hOpziGrUWuLW5zMwAreJEFdEBZ6-97JryUPJyvQKox5PCG8_pFE-BZtlYboAEnX3HmnL-0gcaLRGaEEnKw23-qTLU1pvV1KJZuAEriQGeFp103MwW6OCEbvFl_5M8QAFISz6MwLYYw=w75-h32-v0

7fb677c0-7985-40a8-825f-f6362a669ad8

https://lh3.googleusercontent.com/notebooklm/AG60hOr6kLrwKoW8-0G6fFTczLxcZ949P8pCAtMkM_YOrIY8kbLDZDQoC8mNupEVuUXZSIMs66sPFf9seeBEnSc6-u7P37wJ1Pxe-BzyRthVmHqmfd4ma1ja4vsQo-P986ZQCsV43PmKKg=w73-h30-v0

31183bb4-7f50-4685-a789-e599129b064c

https://lh3.googleusercontent.com/notebooklm/AG60hOo5ZleOhQKD9wdmD3ZGoeHOtqCcLcEQ6AGOWpLAYCZUEFyTum0n-OcD77lXNt_waDDMg6RR6J4UsECqLI5A2FtwqAefUdSEZS0ZL577PzkQ0s5EcQoMtIg47uSZ0-jzKyc_IedEHw=w29-h36-v0

d4c4cf6d-94bd-42f4-9a1a-c40e0b0668fe

https://lh3.googleusercontent.com/notebooklm/AG60hOqiUh8WNiPqbKnIVU8b6B3YOmIEjLXG7AeCpZ9RjmDebEogjaYGsXNns5jSLhs8ihVUE8Fo6IODehUuXnJlW6Gtt93zgW9CNV-rYcOGW175ckSQc3ltN2uXA_wiRH-noi9GkO7O=w26-h34-v0

14fdd637-bfd0-476e-859b-fd32c31b6bd2

https://lh3.googleusercontent.com/notebooklm/AG60hOo58aQp5f-fdCsUGm7E3CRihygoKb4OQsenwpYA3Ve1m6tbtQL3bxW-u45mXr_FakPC6WjrG__k-8SYZ1OK_3sM1ldWRyxloja7yb--N_xVbTsPxqBeKcFTOZSrClD5aGlaJs2M=w34-h39-v0

eb926e5a-3450-40b2-ad67-a64ef8e7760d

https://lh3.googleusercontent.com/notebooklm/AG60hOq60Wb5RgIXGeMX_bkSn6W4HKcPZCS8xj8T9p2DsMERbbDNtP0v-t7Lymvm8SzbQcsbwq2JnJ7Z9YhmD2B_rKalRuJGx6h9AqscVL1qI8PsyW7L9IdYHPRPp9wcRxDsY5CY5fwkyw=w32-h36-v0

1d6da60e-a1f9-4759-8eaf-ba45df3905ff

https://lh3.googleusercontent.com/notebooklm/AG60hOrSr5WnrdMJgcXYXogv7MtgMgQUSumqs13zOLk6G8ljKliwiY4yRNVIrFTwyXLDAvxExBBtHv4MU8ZcupfKizv2ayqWMr5HIuWhL9qjLydggL_186JkD_a_0tPfV9Pa0yeRyYqfiQ=w79-h32-v0

168806a9-1f81-4c0e-b2af-0064d1d19dd0

https://lh3.googleusercontent.com/notebooklm/AG60hOog1VKBiL-IdMd2qc3OgLerbEhQ0H5hVPHC05aR2QqRCPg55tFnC08URM85t36RTjFSHd8NJOjSR6wOuqLWajc_ZPUoQntaXcwfKMTf4NfwsRQ5UT72e39piciXbXzhPJnz-PkC=w77-h30-v0

69074fe9-9a92-4d5d-9398-0702133cbd1e

https://lh3.googleusercontent.com/notebooklm/AG60hOrjcHF2nYlZikdM0mjDSZN6k0MADl8MOldVzR6kIw_0dGrWJuqGLFOYzNTW3yYU1xj8dyzlkLyNabFTu_cONB_XLM1-MMfWBwD6p_hhRDUu1pp0d2j-0LP9AXCOl9Gxni_wzwLU=w88-h32-v0

730dc9cf-59e8-4d0b-856c-596b135cd8b9

https://lh3.googleusercontent.com/notebooklm/AG60hOoRA8JkCu1ODocDxkWWkmVtGWfRsxv0jah0PwNWlx9ffJBPsRMwcu3j9ABuzo6iG5pgPvqEVZfav4hxIUZoMzuXy2Oid8hIiY8XZoJewx5mjOWXvKbnFW25zYIhReurduS094NLPQ=w86-h30-v0

38ce00db-0e35-417e-8f43-8a13639d99ac

https://lh3.googleusercontent.com/notebooklm/AG60hOp17--r-sMzbXpLTQai9jBp9agBaRwOeRVCG3SQmLSN6FVB7UOSpDc7Efh9jXjmRyoLUeOtwf2YMMgAFWgWfe1TcpLjsLu3RYnlEhnFqYIjiA52og62j4B_qkd8TdE5ZZXwicX8=w97-h32-v0

d098c44d-1c2f-42b6-b7f1-d25b95517fe9

https://lh3.googleusercontent.com/notebooklm/AG60hOok1oQh_osLsVib2hbJvsW7jOIZUVr0bACxUsqKOE-xk4KWlncec3n-N8c2xTrWV9aW2aYnIXkoBxSgiTkYChoCJ0Lh5MmifRwxmxS8lvCXZ8SmhWZ9eUZpBefA2G_dWHVFTu1j=w94-h30-v0

8b9a6a4a-ddeb-484c-889d-bac278ae432a

https://lh3.googleusercontent.com/notebooklm/AG60hOqHWHASLkIpequtpa6uuoJ1L9wjLMBj9MreM_ybsDFQ5YIrztbqheIZdrxHF-AwXVA4gaj1j8J6cW0cS1g0eFlMxASjKjLy38hASR7sVGvplhPkzTIeAderQqrRPlZlBp8duozClQ=w61-h32-v0

3e170fc2-f577-476f-b1db-9ca0f56d4289

https://lh3.googleusercontent.com/notebooklm/AG60hOqpVn7Yw-NfJboV86shkLZf2m4rDAt4UE3mxDhiPxlnycAGoGEkbXnUz62GjfjklmUzaRsSYxi2AfFWZc0O17dZg4wPuaUjcsf-NxkxKKsa2yq2I23sLVJJQrcl5_dmgen6v4yQ7Q=w59-h30-v0

42cbb110-6216-46a0-b683-314104cf46fc

https://lh3.googleusercontent.com/notebooklm/AG60hOoFqU7Ml8Zyv_hDcPWaHINI8QMFNr97YmwP_s7ntDtS-ViPsIApoJRrpoQx5mG3c_wBV6m3LOHcn5l6ROr6dXcmYpgAqkJ_qYC4jFDNQkojJeRevo2dNmVKYLy6L5Me75kjYDlREA=w68-h32-v0

35cab71e-dfd1-4c26-9502-48ec4144ec9e

https://lh3.googleusercontent.com/notebooklm/AG60hOqYi9pEdUJZD6R6X892Cvge8KNN3iVSrme_2-1lILwQ9FpHDJ1yFPP8eTl8Uhpr28PxOaFTNxxfFuzWB6Q8_4GqerOY6i7WOZYJRcIZn5LGU1_wCXyEHmdiJZf2thMy-qHKlhQ9=w65-h30-v0

2d245fde-dc8c-4dfc-bd96-d8962af51ae8

https://lh3.googleusercontent.com/notebooklm/AG60hOqvYsSvBh06xRgQENF4W28oglAZF4yo8EgsgOSdsg2gpI312L79nZUEIqD3A3yHVZm8TTmedcgjSAGiHz_Q3IyhTbrdPC4hLVRu4M_dxnXOqMlpq9Tw7oOgY0pZcd8lmuNtw_Db=w38-h32-v0

9efe78e9-e2d4-4f34-be5d-211a5d74c8e3

https://lh3.googleusercontent.com/notebooklm/AG60hOpzMV1vN1Xm5KJieNPa9dppMLvu-rvVqlLn9rF42sQunpfTQ0ncwUaJcc1FWuGc6ia3JFwx_5mx_6NCGrqN390VQw6rOFWIXBg0Vk-YL01zBUwIHwthQnF34v6t7eTZMhF-HtYUcA=w36-h30-v0

08f92a7e-45dc-427f-a596-0bbbace2ba25

https://lh3.googleusercontent.com/notebooklm/AG60hOpPYz8w6FIfUtDvWrt8TX5C9nElsDZpn7kBlZQFCIAsu-08JKuInXT4pXHaSvq4Uv0w4OKqKCfsTTbYvKiBA8vVO25nRcoUaNMKATSp8PaCJCBPd-WnUh1S8YVOOwS3WZ0Nx7AMag=w123-h32-v0

cd1328e3-0338-47e9-81f4-b04beedca0b3

https://lh3.googleusercontent.com/notebooklm/AG60hOpvx-fTsjF5tn3uyPWRKyGeahDJPZBltXOurq8UkyX3kaXvKnAlnBmAgRsh3vYmfw_X9KN04AhTB3YA0rp6TJ8xtecXuIqvIZnyUTfCy5BpO2dISydlzBBbx6Fjwras_IHeXTUdFg=w120-h30-v0

283bd225-146a-4e12-b21d-4272c9bdb1cf

https://lh3.googleusercontent.com/notebooklm/AG60hOpIGOT-IdM0BjK-p3dsSrAzkbAYdPrXki8kUvqxW0S-6vxI93bfSdMEOYGy-dHqOsOFuoJXQU9T8o_Vdh62-EXiMM3LbVEj5OBB9lAJ7wuboFEsy3AuzC_s9aaDqFYIZUn50uCz=w51-h32-v0

3516698e-8b55-47ac-a32c-5c72c3df3fa3

https://lh3.googleusercontent.com/notebooklm/AG60hOrlK1_uXUWHwaSBJ2i5BStFZW6nq_jYIRhg3FIFNchFrYAHPqfTJvRnq9D8Zx_s6Vb_Vk1SOxE7PhTwL4LIQkvu7uKcvvNDDerv7hrVHh4KXdOKfkbt2V562MJVU5QfqNh3cl_D3g=w49-h30-v0

e40297cb-448b-44ea-a78c-9eedbc7de18b

https://lh3.googleusercontent.com/notebooklm/AG60hOofcQXkarDNYsyZK3vGBlW_DE9eBke6Y3KEmLWo3hs6CCIl9AwDCESRJvm5kcIW9uI_Wv65XYqbfefJOgDprwF6hjSmH274O1WeTX8mZR1Ul3Fe7WUfarqJqjIm3cBoQfYo8ljhkg=w67-h32-v0

651d6676-ed0c-4ad5-8f36-3fd3d97c9285

https://lh3.googleusercontent.com/notebooklm/AG60hOq-bjCU03z5kyjVkpyZc02Mj4kcAUdfWTNJWbKHGQqpKrow9-oX2JpVzQKRbg6V7BaqJtRIqzstjiF9a7j12XkLv2SB-cyzRmMsVnSxox5yn9LmlIGkjt7wvn0v4Ez24qMnjruDRg=w65-h30-v0

48b12fb0-302d-4d8e-98ab-57bd635e8278

https://lh3.googleusercontent.com/notebooklm/AG60hOp6F7MhfZXSn9kKDDf9dDLztTchzrWjYXJ09cAanzBZd5Rf5JFhOoZYIaonY6TaAUo4I6247yJvOtdrO3lH250HtPtJsWoXrBMW2U3ZrydCyAmx1N3_FjKdX-xYJPnpl8R0gFYFFg=w38-h32-v0

4a2acda2-3a81-4bdd-afaa-c4d07235354e

https://lh3.googleusercontent.com/notebooklm/AG60hOoqYrwKzzNpk-doaxizvSl7FliyfBOEDoSsX6t23DU5BMD2diMpJ0PsV7ykL1GC2-zWwScvlxP45p6f6eQsU8fk1kqgJ5eiR2PhPQXjva_lpEaWpBejF382kAAOHKYoF3PA97f-yg=w36-h30-v0

466b56d4-890e-4ecf-ba70-b55a6f27463e

https://lh3.googleusercontent.com/notebooklm/AG60hOriGUGdzIXpMMhB9TJVPMCluskufncBxSwhXx0HO5uRCBN2GJ5S1kIYdB3zCDgw6mO8tBBnEV1-j8pGMQ0mBOAs7x98nznJYxdtupHrcw_eS_xyZdOOKtqT0EZCTkiSS28RfCGmvQ=w83-h32-v0

4f72cb2c-4b2d-4395-99da-9e4df4dab360

https://lh3.googleusercontent.com/notebooklm/AG60hOrvt_cb7cXKmvR18RUekD9x6iTJouD28rW05qY5tOPvIdCubT1qfOc5RA5OJNO27V5HMgKXC8PrYsHKZTltMz5wqJHv_3xiwJdGEXNlwvlDTO-Dumd14nFdXsg0O5VOQWgPz5f5=w81-h30-v0

c20012b5-f4a9-40fc-b314-f9be1e5b4329

https://lh3.googleusercontent.com/notebooklm/AG60hOo3k7izw4mum5T4o5ctk4km1Z-hW8LU1rMLjPqNBTOpC1nYLHFY-J1YIuq7jJ2K6nz3uH82ut3ke4O28gW2aQMlVqBCUNMre0lXD1Xoje0laLVDy3quFAlE4rRxKEGYXf3bMspJkg=w94-h32-v0

56eec447-dd30-45b8-a308-0c0be12f5681

https://lh3.googleusercontent.com/notebooklm/AG60hOpLfq8JIF6bQdt3KJsK4ZN89Po7HoRE_h5uMvgOn3L5v95ETZhs6xFD9FxAQ2NJWIccbri4qAmu9zo4dtiNZfZI3mWlISLA_X1t1eRg1BUbAyMbMlcQ8ot0Vz-qXwzs5iAikmQfnA=w92-h30-v0

6d584bd4-2f3f-4ddd-a32b-30404f344550

https://lh3.googleusercontent.com/notebooklm/AG60hOqRBgLc-W4tehr4q0ibjXwYSNWImiX0A0q6uQ4FCx46SozwxYrumAslXwxxFco5OazER9kTDqRMWuDjE7IjziJfKwxLV4HZBBQGMRgzoibeuuHxDlAgntAlzUVqmD7HWEsOsMDm=w48-h32-v0

6fb5326c-2bb9-4de0-bc7d-4cefa7479e4d

https://lh3.googleusercontent.com/notebooklm/AG60hOq6rgBDeDfGAbd6cmlnIGd70PHENx3yEzomNmTHR2fW5l0Eb8sGMUYw1zssTQglkJF7Mg9a0qbVRqWXvYbGD7nemn94ssUCa82gogkJtbrYSpuiufV7qe0IO3XykJOVuTGeqaRl=w45-h30-v0

f958031d-4781-4c02-a8f9-79b9acf6a77f

https://lh3.googleusercontent.com/notebooklm/AG60hOqPYQKeEp4S7DISEnF_LLiRcohF8sbP9fajg8qGe-QEPsz14fDtWFxOq_OISVSdXtOtorKRyV7B_4uqr3rMcOk4HPTZM2U6BvkZz1-7HQxII2QX3ct3xRc3GQ67DojMvq0u5vGT7g=w46-h32-v0

767d7b4b-e1c1-437d-b8ca-148cfb233d5c

https://lh3.googleusercontent.com/notebooklm/AG60hOpb3-OcrZ08uORtjf0d-Ptk6gL8pbDOfHPazZewrFE1GhSsQ5VGiWxsR9s9aDLFnDJDzIIvmwkTdWqCvakrh5foUzVsHSaAUb0ybcA4m0T0B_H1-20jGIcUHhbdKY9vS_SNMziQaw=w44-h30-v0

5b65e90a-7120-4ce0-8fe1-d842c87b9a9b

https://lh3.googleusercontent.com/notebooklm/AG60hOrObihf81j9O6JQJG2XJn0x4HTpl6bftgzl85mQcNDWeq2PI_dhlxI0Nj6ZsK-71Q1xhi6YpCDVXXdCl5TycN3czgXEm5N75nliyoIlocR6Z46__BDTM7mdcnhGFfJUpiU57I5QEg=w114-h32-v0

a2c1b23a-0957-4644-8d12-1dff227b643b

https://lh3.googleusercontent.com/notebooklm/AG60hOqV9EuDw-TA9-WonrVx18lslW8DuvDBP25hF7V8M6vg3F0ganwLQSonOlvaM0UWgKZ-9gxKA6PJh8yYCKANTz4HyEXCGUbg5YMOYWr6v80JR-PyLb410rJS9oCVpO3Sr5O4x6m60g=w111-h30-v0

1657051c-1e5b-408b-9651-19b95507d73f

https://lh3.googleusercontent.com/notebooklm/AG60hOrYclCDFpqaJ8Z7wbTiyJeUbXSGBghqOWj9P_UpF9o3rtVMyaQdHkfHzjojEUazX5djMhVRYZYCQXu4LntGCLRiaBkpY-5VGdDP-Qaspxvcl5dFkY55OoHxNNyAA7efH4FIgc76=w38-h32-v0

9c5c58f0-d503-420f-ad3d-5fa0d0376b92

https://lh3.googleusercontent.com/notebooklm/AG60hOrfa86-5GAxZcePdqcOHdR4qG3BGlKr8Yun-Yv8cVAp-kCy2gV8yF0RROdsIP-AwhiJ-hxj1fIUkJmqjMQEE7RBcI8GzqUbN0ea4sUAbq643sDmNpOl1rpy3g_cyTL95FV3Q60j=w36-h30-v0

ea6b3b0b-82c0-43b5-b972-332637014cf9

https://lh3.googleusercontent.com/notebooklm/AG60hOoedKX6f1vD99jCkKZAdBrEEfX8wHQj9iuaS-7HBVNnHb39XE4Sl2KD2Dw5lZWde7NScsy2VKfJ983IGoMCYyVUNGeKN7RTqr4jLuQ8dNZfTh7kG5qg__qsMWlv6XAiFA8VEslg=w45-h32-v0

d6fccbba-3c76-4db1-bdbe-bbb571a45a52

https://lh3.googleusercontent.com/notebooklm/AG60hOpMyuECGfzM1j52c7lkzl1OnDkhMk4wQ6g0kU7Sfm2W010h_V72b6wxcEUSTvBz8NDT8E-pUx2h1KbadZzBUpFkPJur34SarN4Lg2I5WZxDBuirYTx6LThR8QYuhJQRApJuYUHlBw=w43-h30-v0

78bba8c2-9ed7-4f37-a66a-5e8b2095c33a

https://lh3.googleusercontent.com/notebooklm/AG60hOpJ0QHqoPdeBmova_o7yDtpCVtxgE1Vq_ChQjEJkQE7vTA24ptNVcTRL46EehUzekb_ToSmcSIIzimOzcfcnYKwnkAeP89giWqQt5R71k0dMJK6LUpx00hQnP11pyJF0CWQ3HU_=w81-h32-v0

b44243c7-de27-473f-bf14-14c6cc94c1e9

https://lh3.googleusercontent.com/notebooklm/AG60hOpiP1acZuBv8WMSgKkouZ7KFlTxCVayEbIKB_me0SvIYtsFG8EoJcqj1XUc5Vs9DM9zux8hrv5BYJos4N0IRpF9r8UuHmWs2__w0oJ9CBPhZjap57whWFD4CI1DxKhTxGnHyCg0=w79-h30-v0

fe6b9537-7151-455f-82af-9f7f388823ac

https://lh3.googleusercontent.com/notebooklm/AG60hOq9AEm12931JvQE3UNhp8rsEAkkqbCb-yop2i3i3XoFTnY5M9wfvGGiwmGj1I6R7v_f94A-lCl-ShBxbUO1cwzaI9Rzl_4qjWOLVt_0pTyJj34vQx8Si3hoaK4asuyouJv9jJU4RA=w40-h32-v0

757dff58-4c66-454c-ac9e-344c77fe4a22

https://lh3.googleusercontent.com/notebooklm/AG60hOpXJ7g-l5VrGWqxnw289XwE6iKbpTkkulXOSaF_qaVEknGKSVmblyv29xzZX4QwPTDWRRqQDd_lZ2iZWRbskNUHUxnuqbBycldmxc2Y7wzSR5l7ro-4ZHiQv15QdFfnXS4Nef8Y3Q=w37-h30-v0

552ae7a1-38c2-43f9-a42b-5a6f41bfa049

https://lh3.googleusercontent.com/notebooklm/AG60hOqbyPbs-t7WAuOh_F507gG50zVZS6qYs8teZLK8vNsKA355kBBu1ezywwHfFmGuwDmt0JPwaymKKECun7vyata-7y5Ns7Ez5J5JlYXHliDD0s4j9d15Os8ZggMBjH6NCHb-Wd5D=w46-h32-v0

7680093e-c2ab-46fd-931a-bfec91633656

https://lh3.googleusercontent.com/notebooklm/AG60hOocV40QBu_aK9qOmhvhaQKoU-2eQE_N5N_6uaz5v-JXYfFw2T3dTH-HD2Y-eDkzcYYbF-CJZbwTI8yay1F9FyZeWebYz6cN4lCveLkaTyQiUYIAcM52oQY9Ije1vyTrbJXENa-mFQ=w44-h30-v0

570a01ff-aabc-453a-8976-e7e83803a495

https://lh3.googleusercontent.com/notebooklm/AG60hOrqP93Fp4tA7jqTcDH_nGtGSCT4OuwjVPmHHHrQgmkzAFRyvMkOblqVvlvdEyFq-vVtvglSTuK8R7lHNqiN7lRUi51SClDzuYOOXzcfhjFAuZ9HTh-6oaBAZtCP2vKmE5f1oYQ2=w74-h32-v0

732c88de-343d-4b88-96b6-4f1d773bd481

https://lh3.googleusercontent.com/notebooklm/AG60hOr9A4t7HpPAm1h8MYkvcy-AnaSVcfq4rLmIKBFxELrg719a81UpQKc0S07GYhDoUbDw1j2_NIiSmNDIjOOV-CUh_FxFbG_eOwtvh8NF3W3bAYzSl84waPCIIvhOE4YXdzxJR0KOuQ=w72-h30-v0

d3906e98-d2f4-4735-8ba9-05dbcd7890da

https://lh3.googleusercontent.com/notebooklm/AG60hOqUIow7BOJB6LedKilUXqqiv-wYm3Fj7d5i2-gtrvXhk0sD6apN268pMCMXLjISOh3rVJwPkxHLVGwp2nESn8ZgMeWrRFKqdTc5w1XWrmf5l0-SXbCakVt0x0VI78z9fiitEm_l=w60-h32-v0

ea84532f-6e88-443f-ad9d-b3743ca02df4

https://lh3.googleusercontent.com/notebooklm/AG60hOo5hjlGH8GVfb4f1I6HEIKi8MhPazqLmq0M8zCji-VVsDdmZCYlD0G_nuTsjuXHo8XYNMhy_FgKynE6atPgon-M_s2CdLaDJxtxqmysiEi2JY74yNHBi67J34-HBzMQRyDzR50T=w58-h30-v0

14bd58fc-c1f9-424c-b0e9-b5476f9e8a75

https://lh3.googleusercontent.com/notebooklm/AG60hOo6cDOkLWHTCU94H2O02I5orUSLuwfLKthLu0xe-c22scUvN_YdP0d8Udu0y-KGngd9VHqjnfogv9IoI-aZEU2XbB83MYSLW776uVqUPdBQCFCCFskv1NdbA4S5TXPvB5kD63LJ5Q=w48-h32-v0

13ad1075-a11d-4d11-bab4-262ffc2d8c35

https://lh3.googleusercontent.com/notebooklm/AG60hOohxAtpiPCOc0LbWhL-3daWxhoOvgjuanTxTF6w7816q78yC3ZcEKx5PvlGJ27h7Y7jKhSXbWWH_wTS5vqKoHESm4yiVHuTQfujY-Z3D0mVw7rq-7_AnKk5sFy7XQseNEcCZno5=w45-h30-v0

51e9cb33-0095-43fb-ac84-9499c8828c98

https://lh3.googleusercontent.com/notebooklm/AG60hOr-aTXgUSIZK4jgD5DCutPTw5gZshAkWZ5Z4ouS4CyVN2uHFBQwwcT0mlDTASbVigKOFE8o1_Njl-IJk2IXozx2wl9JR8_0eQzdsMt3uJHb41tDfvYMv6wtZbzE79LTVGCjUV__AQ=w82-h32-v0

497fd18f-0161-46fc-8542-19465b46e6cc

https://lh3.googleusercontent.com/notebooklm/AG60hOoSWVT8HKmTbLbo7oM-U8nFmsKqZ-qIuGb7G0m9Nc209R8JOjmgB9t7sPV7tn1m9_2l3GePwXYNM7rQVddEGWusNDf06n2R386x8jIOwLM7xL1-0xonNAgfM9OUD1Ocmhok9QDd=w80-h30-v0

bb653c3a-305a-480e-ba13-bdbd9dca9d41

https://lh3.googleusercontent.com/notebooklm/AG60hOp3kDEipdpBLDXoOVd9V4B3iwx5cfH6HszHxgNJRenrcynQjCd-rlyRrkrhaf8DtSOri7Slxzu_-tfTs0L2W--Kvrx-4ToK5ZWFse9t7GObnSmGzKzZOpZd9JLK6KBCTzoou8Eebw=w46-h32-v0

32306421-cf91-445a-8c19-4be43d74b81a

https://lh3.googleusercontent.com/notebooklm/AG60hOrszjLEB2AgPjc-btrs9cV2ScMJmvqxk0AuxBudby1ftzwhG3_JIwBunzxwh5ByIlNTRlfugaOYc2fEtQ5v2Oyf-rieboqZuXVEjBm8ds8Wn0NcPQL5umv0mmlrDQWmnb_bOOE_0w=w44-h30-v0

55419a08-2930-4da3-8f1b-9a2ece74f452

https://lh3.googleusercontent.com/notebooklm/AG60hOoB9Oy7sGQy1VvCuw0fKXwOE7PGK4lkD0AONPr1vGPMikps8jl5OFkSpp5JW4PcH-tvMNbHSEDf1xJ_FRTT414GK0rAm7Y7T9Pk4o5UqUGoVy6-0iFAvq8As4mL4gJI_MtyhuuuJg=w68-h32-v0

7a09d730-f5a3-4617-a86d-6232d491a51e

https://lh3.googleusercontent.com/notebooklm/AG60hOpDzFf0IKdVwE_LSAVrMP0z5FXp2Zw3xj8AhEfWtCg3VjEWIBrz-zt1yjiBAIqPX0CT4Lipjedr2BP9IcWFd6rn73QjGPVeEBf-NgwQ4zhp9K2G-S2tmN0PIvTMSzR2F7bq2Yyy1w=w66-h30-v0

150c8b36-62c0-4095-9df2-4a274464324c

https://lh3.googleusercontent.com/notebooklm/AG60hOpZWoXgwE43EbwDJy7YiU3TcBvjp1OpukrAWhrFmdcxeH_S_05Ya8p0tWzn7rzFZpJ6qH-akNxHsvqMNDKnQ0AIfmZbO9fv5cglutE43Io-yuk4CkHavbzcs8mU-oeX0mgCoarFsQ=w72-h32-v0

ffaacd6a-3d15-49ef-88a4-3361b2a902ed

https://lh3.googleusercontent.com/notebooklm/AG60hOolgZMXFtIvx6JqgXXO3rbPdo7KOnAE4hq8g26rhfCdS4Il2oyzLqhmogO6MgXTiooQcXYrx-nANQMe93LVPgAcAxH7CXePolj-CBSXLozYyvAAIGPYlyn3czTi0VXsI7nxkbsovQ=w70-h30-v0

54777ba3-4514-4380-a7d1-1207f20c0421

https://lh3.googleusercontent.com/notebooklm/AG60hOriIiWXDV3-ZUyAqyRM8-iRDrAIVYOqg2mMRo9cM_P-ALADNgof_99SLI7iT6Po1yvSsPiE8OjapkbirzuuQQ7v5dIeyzZEgqU_jtSZOSWg7TX9LlWj_PclEAPWYOx0q-j-EXqWzg=w69-h32-v0

7cd8329b-bf5b-444d-87f0-53ac126ce3e5

https://lh3.googleusercontent.com/notebooklm/AG60hOpfnECAvG32-JI3S2XeaFLnOB754eyRUMx7sfR9Fi04r3_v_ft4jvCNyLkcL4QPAyTf8fM6OKnrQmvX5AJWqGIH4iAWw22CotHMaiLWk1Tb8ydZapoaeNNRJV4_lbRqbKXBFCemxw=w66-h30-v0

9b2f41c2-728a-4146-86c0-b6140b3fbfaa

https://lh3.googleusercontent.com/notebooklm/AG60hOoEr0koKu0d-vkb_51GCrPyMiqlN1by58VhUoA4mpwB0S6o5a6mr0GZIwn0VDauEcnrjBDjWytM925kBuoskXtvCcrfdpF9ndBRF3HifJvF8hUVepNzsDP_01rR-Y3Y6NTsubXI=w106-h32-v0

bebe8d9e-266a-4fef-bc9e-6bdac1acfb68

https://lh3.googleusercontent.com/notebooklm/AG60hOrFQVbdOFxnujrMUgUEtgRZxY1L3IfO18woIdPj6MX8xJkUZv9hNGrwbvGrFlin0m96a3AYWEr9dwinoEm1phdBaW5L6sg2dFgW1jXPVsjrH9VwM1eGjEVaHGdjUWnsxZSCGA1Ymg=w104-h30-v0

bc1c971d-e98d-4a16-899b-64821d69f19c

https://lh3.googleusercontent.com/notebooklm/AG60hOoAMuFxHmlexW9ReQlg0YzzZiNdv45l1jS7HFfpghrFiPG8axj-_qEhGWNUlJKFAvEAE4aV2WZdTPDb2C-zjq3h51P9SqtfRjvzj78A7dIHrR9dw7t6jwlTDvLRBARpERSP-K5I4g=w32-h32-v0

c4187a28-b9de-455c-8f44-c5fee7220168

https://lh3.googleusercontent.com/notebooklm/AG60hOp-CxJlsRBHPPMvB7kAhEM-l9jhLEwhCGl-CS4kW0g1tOGxxCSlMgn3WG5H-yEHPEUxr3mXBd2-SGb2Xgyk6dNVVZqZm76-IF5JdzjgyuYtM4ZPczOCm3h2EBpbmJ4KxqQwUjQl=w30-h30-v0

66a78a34-b4a6-46bd-a54d-e33edbb0b70d

https://lh3.googleusercontent.com/notebooklm/AG60hOr5YD1HYoXmgzqQBPjeox1yKf9aOYFE2-g-7V2GxEAlhCpqrWe-0NnuuspvxiDrnaF1lbrrgcM0gl-gdOs5DX5F0uJ1uQvcpiywUA5n13MkDgQ6hiyrdCuc_vVnrSk9hJCnlqAEpw=w57-h32-v0

bbd22aec-6016-4a3b-b011-0d8be57e7604

https://lh3.googleusercontent.com/notebooklm/AG60hOpSKN7tt68PQWfeMynK2LufE1IWo2Pyrz8BIV-8YLj2zqbJtH_xfEN19XZX4y3QSKGN47pLXjRYmeFaEEgpOWBUO_CbmG9aMwUp_3j9bJQZDCmqjpWMoqHh6S5XUun-mhdB7An-cw=w55-h30-v0

1a472da6-fc99-4482-a803-8a5cebecac71

https://lh3.googleusercontent.com/notebooklm/AG60hOqkbkCtkxajZgvG7usiSf3gjFPphMP5ibsp0zvRuZUmW--g0aI1G1HUv7mEqk1JFTQwm7KQQSPKNjDA-NMzZwrlU3dRsnlvLVXn2Rh9N5nt35O9evGSZQ3P8rvIREc-MpS7wF_Y=w73-h32-v0

8df938f1-842a-4aae-82ff-1bf8a8a45690

https://lh3.googleusercontent.com/notebooklm/AG60hOoyJn8qvH7Dvx6HIfZuWMb4gmr4ZdY-juHoqhU9Q4rUHlTcd61g8u-wvZkF3fHFfd4i6fOEI2KsxZ4vx2UevH-xvkRSetBxfbjzB20N2wnk5-UX2ZyE8cMpnZgrWyxWvFbHKUCM=w70-h30-v0

a966cb03-0612-4c69-a8a7-9c8e1fd796e2

https://lh3.googleusercontent.com/notebooklm/AG60hOqLk5SS2CJbeILoU21DD2vu1rPnHpEhAx9yI-wQJauZ05yUxkEkcPN2kzHYx7uKBkNsMGoODrllb5E60J-t49bG6ndUlqJTjWm0uIhvJAUKxAp3x5hPbc2YSuHhbbx3CDSuLfGB=w38-h32-v0

d33776d0-7611-4909-a5a9-2013408fd2b9

https://lh3.googleusercontent.com/notebooklm/AG60hOpbMrvkeewQvOCVfajFIFeHabgD5mOeXEJt25QHfD00UnCIGzoxICvmnhbuPYbWWqePiWEz6KPv3EStPO1lByQ9qAtjNnvUk-ChQEEI1z7YnQxc7u1zwIWey5gJiRHVOohMD7zd7w=w36-h30-v0

55c7a780-1f25-4e87-a1b2-a790b4ebdd31

https://lh3.googleusercontent.com/notebooklm/AG60hOotCzFf5ndJGlHvnKNBxaH1Viz8HB0_lV1EkzPhcBaG1HMyBrrLOCVQtza1bwGwgJ8CtYQMkmoBUt422erExl7RsK43Ae9gXOD14gFwSugQosJsDz9zYRUh9xJLbZ4Svs8UEaet7A=w53-h32-v0

3f072f7d-ec90-40b5-a2d5-9797d1407bc1

https://lh3.googleusercontent.com/notebooklm/AG60hOqyfDrAIA6Gv3y3LAonwR32VNRIw92sKHtBnPgqqjlnRdMPks5LNsFkjwQjqq5T76UdS2EnF9Q2cWzOc533KCTbpdwbHoQ4wjtU6lhn4_68klP2IfiqI9R35WFjv4ZWthm91oEYew=w51-h30-v0

cf5370ce-68e1-4ecd-9a7b-8004e37a4ace

https://lh3.googleusercontent.com/notebooklm/AG60hOqVTmQwLYPVbnA3rcCf3m2H108vP5O2xmArGFF1kUiybKP5b54jYHyzxRnQEClSRcJsseuk5A92wFjkbwwahoBUGczcj1MVG0faXPTyOiVrpY0obUAP2M-ZRUqgGHIHoFU_awnH2A=w38-h32-v0

13dcb6a0-f3b1-449c-85fc-33f3522c1aff

https://lh3.googleusercontent.com/notebooklm/AG60hOrNj3vDULlxU3iVsY5-NLvvo5rtS_gsb7Syu7WpKIoqtGJdtHKuAxjXzfMm8da0Jd5kr9tkmcqoxY0MK2IoRh9bIKnl4Tt6R1wwzkeUNqCMFqsUsSVcnNnm8vbUoA0UxPQuB4oF=w36-h30-v0

032b20c1-ce5f-4dca-b0c2-37b2c49189a0

https://lh3.googleusercontent.com/notebooklm/AG60hOpWMesjbgWXHuA-U-imbT5aFtTN7wRI9H5Vj0riDMWNOYbIs2c8VCdFECb9pqrfd8J0xhR8C0-OJRJ-WQ8HEu8hAmLKAQ7YCCKWw5Q-0WA4Tgt7FFfjxeffDCD4r0vfKNkDLMlUuQ=w46-h32-v0

2ba052bf-d114-43de-9a2b-f7cb5a054e05

https://lh3.googleusercontent.com/notebooklm/AG60hOomaM1dV_0y6Z6xyaHI1ZGSHGXSigOWEGVTbplD45hsPGKiVA-NKJfjbN2AqUWwgDsVtWMysPNaUp1DjdV-zzGtuWHbHpTRfkoFtiNBcXBdck_mtywkdGN6cSbC84BKbSnnX5nJ=w44-h30-v0

c90be2c6-a9cd-4332-abe1-316526dd9cee

https://lh3.googleusercontent.com/notebooklm/AG60hOoolc-2jjek5iHxdF7EZZYI-EHenv-Gi6GSX-tAiYMooQdCrpgzOe6I_hGG2VLq_62wdfqfOChekjdR3lgNcjQwEvRkahsnNbASQ58SMCietJRc5AQY7tJPZlDSVG98UUy_-BfDXA=w46-h32-v0

ee03d1e4-b683-476a-b25f-8784b21201c7

https://lh3.googleusercontent.com/notebooklm/AG60hOrxwjLdiOfHGSpwdiJaU8K43KN8rtZeS8wx8Z3t2VacpvJ47TX-6Jv9TumOSYkumQ0m4vjPc_lgLpZD002cqyPHRecr5yhDpu15i7EUUOQRZMRz8_6p-Xo1wK4gPeUOoYTLxgx0=w44-h30-v0

42f8ba8b-3beb-4e8e-8abb-adf864a59c25

https://lh3.googleusercontent.com/notebooklm/AG60hOpGvMq_qfG7TFL9qg2ZeqKb-VintP5lViyfHa_mgO1XV3j2_oSdiEY_P6OA-KB_Gw7hkjyn6LBrfln5Xwa0HA3u36Agg2KzdqsipLbmOHHNsqORE2XlKSkrzkZVbNzeVgw5gbUDXw=w216-h67-v0

ce30bc1b-4096-4b2e-88ad-6c9ea462b6cd

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## A STATISTICAL APPROACH TO CONTINUOUS SELF CALIBRATING EYE GAZE TRACKING FOR VR

## Gaze Tracking

AUTOMATIC EYE GAZE TRACKING CONTINUOUSLY UPDATING EYE TO SCREEN MAPPING IN REAL-TIME

## THE ALGORITHM FINDS CORRESPONDENCES BETWEEN CORNEAL AND SCREEN SPACE MOTION

## GENERATING GPRS

A COMBINATION OF THOSE MODELS PROVIDES A CONTINUOUS MAPPING FROM CORNEAL POSITION TO

SCREEN SPACE POSITION.

Images from Tripathi, S., & Guenter, B. (2017, March). A statistical approach to continuous self-calibrating eye gaze tracking for head-mounted virtual reality systems. In 2017 IEEE winter conference on applications of computer vision (WACV) (pp. 862-870). IEEE.

## LED illuminator display

## Cameras sees eyes in mirror and IR reflecting mirror

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## A STATISTICAL APPROACH TO CONTINUOUS SELF CALIBRATING EYE GAZE TRACKING FOR VR

## Gaze Tracking

AN IR ILLUMINATOR SHINES ON THE EYE, CREATING BRIGHT GLINTS ON THE SURFACE

## OF THE CORNEA

## TRACKLET MATCHING ALGORITHM TAKES INPUTS OF SYNCHRONIZED TIME SERIES OF

## CORNEAL LOCATIONS AND THE COORDINATES OF ALL OBJECTS ON THE SCREEN SPACE

## AND OUTPUTS THE SINGLE OBJECT WHOSE TRAJECTORY IS MOST SIMILAR TO THAT OF

## IR illuminator creating glints

Images from Tripathi, S., & Guenter, B. (2017, March). A statistical approach to continuous self-calibrating eye gaze tracking for head-mounted virtual reality systems. In 2017 IEEE winter conference on applications of computer vision (WACV) (pp. 862-870). IEEE.

https://lh3.googleusercontent.com/notebooklm/AG60hOotOfH890DVd5Dif_VEP3XHBw7AKEzyir37Y8ixwHcLDJJIgUDZRNBwOe8GzcoksfibfYK7o2T3relICZkmeTdVw-E3DuIQQ3wApt1lduv8C4GWiIldccB7vJkrPGdorIHfivRVtg=w116-h28-v0

6162e107-3d62-4e0a-9bfd-c6d24edb041d

https://lh3.googleusercontent.com/notebooklm/AG60hOpfDsj0teEY9lAIkl_aIhEUTMIyQm4jyl-ftNflFyOtiARIIAvuoRIaGDr3dFxKe9tK17vnCAXo-F4DKVQB_IW96aUB8-NztULcAYtVFQdrjkBkLTfwGL6CS0PfJ6-42B8_UTRHQQ=w29-h36-v0

13a6b4c9-37cf-4d53-a154-4328530769b4

https://lh3.googleusercontent.com/notebooklm/AG60hOoyGjpksvl7p1FsZmk0VaAinl7sRrSwHbhRlq5fxUG15TjvDZK9Adv_rQUqnZZOu-q5CZnP7QGYULuTkia9WXVfpVpHl8S1QgexfZai1P_DeHG7sfC-shMH4N3xOwWFgntMty4YXQ=w26-h34-v0

6b4c4684-282d-48ac-acb9-5e03fd261d9f

https://lh3.googleusercontent.com/notebooklm/AG60hOpHeqVIlLwtwhpe-3z_KcHi7IkEHadxkQXbjfJhfmnpeWXtN6IAugKyyUOqUJLRAS7kPOq9rlKE0wgitiBTj4GPAOZwIjY3KA7K5-LjqJgx2ZAPrMYx20as1RoD0014KSStKoTA1g=w34-h39-v0

e0fe8ab9-7ea1-492c-8e64-5bd000796459

https://lh3.googleusercontent.com/notebooklm/AG60hOoepmqmsG60AW1B0xJ4oShuFw6d3mnNljCAODAMqoZHGX8G3byMeSyZsmICOJsVNHJsxWeFAJj1hfNoKQ1T3O9j-NBOH8QqgjZq1qBfV9dkdTWqd_A6saSidA2F8_y_vPJaUmgF=w32-h36-v0

c7a886ad-b532-4789-b07d-4b99f348be36

https://lh3.googleusercontent.com/notebooklm/AG60hOq-wCwscvLXbrPyIXgkaV7xlFC9FX6Y0pV8li6jjUhy2wS5GN8-HiNFOKT1c_KoTWO-UGjGLVVBX09wHr89dGH-ETPAK3SyF2PPtwMTZfxzTJHtIKJ85Y3U_XPJ9HaIyEIAh1t92A=w24-h32-v0

407351b9-8f0e-400e-b27d-d32b853c1d3e

https://lh3.googleusercontent.com/notebooklm/AG60hOrW5PL9IBlVGWQBDUvhyLS3E003wZY7xVrP7I8NoSS-EbtzOSZJBg38cHWmxN1Vai-U63A2kJebH7hxCeJ9dX7SlKr-k6GeV6k-Ajw1_XfJJcqhOZdnpeHDC_Mbupa5QnaeEKjiYg=w22-h30-v0

ab19f1ba-9997-4630-9e8e-be70317f781a

https://lh3.googleusercontent.com/notebooklm/AG60hOq6OxeRtxvqMOGfw4NeFY9ATEMPBOFgdOXrF3vxsA_GAWCyB6wwvmEuQJnVEnIXWsaLWyWtlpO589ztbZftebunTQy3ACaEIWBwH8UeO67ZcEzgqRqeq0-MaxO0uIdmJShYN6sWfQ=w37-h39-v0

21956a78-1ce8-4c52-930c-6168a57dfb8d

https://lh3.googleusercontent.com/notebooklm/AG60hOpSl-b2h-IiuoIKI_i7Gac7iBLC-Ri1vy_vYbbhTgiaKbLBLXJSfkDRDSUaJTeijWUX4LacUyhIAwasi3jZTVopjbynDMYa2UquwHh46z8QRpckSWxKlSJPVIpfnaPPtKtWKUoV=w35-h36-v0

e2e28e47-90d1-4485-b50c-eb1bcd966506

https://lh3.googleusercontent.com/notebooklm/AG60hOrca61YIP9Jfqpl9vK9rbSv3c3poYS8l2giXhMULY_KTVM48Qs1h1MsswAy1-8a-j3b2A1jIALFMKWUEkgCRZHLtOFStEC-gX4qK6EPXeDKemswzqZ2VLM0R9zdkO_r_BCmn_iLqg=w46-h32-v0

abf76e14-f3c7-43ef-aa24-7cc3ec2f85d6

https://lh3.googleusercontent.com/notebooklm/AG60hOprd15Y6gQrEA5SQGzyWhGT3sbGBb4SINlREhl4AV-RfV6Sv-AtYilFqqW56CSDZmi_Ia-z_e362NXmOUvdCQPl9k4xumU2d4WhoowfYwi52SmQa2YZHaotqLrdA04fdHGE6FyALQ=w43-h30-v0

7a20613b-1846-4a9b-be2e-0e41ca7e236a

https://lh3.googleusercontent.com/notebooklm/AG60hOpP7zL-eP8eLcAlgbHbXAxybzVtJHoaWRQoNghxSRFmLUNc5S69FBkGFuLvamZp8OduVogOEOSW1_G0vhiCEPE4IBRgnEvWd6By4fo2bZ5CMDSXdxnGANsvrA1i0Zx_FEUz3goXgg=w55-h32-v0

b2ff33ea-4ad6-499c-bed9-dc02e62706ef

https://lh3.googleusercontent.com/notebooklm/AG60hOqOip93SSpOAA4pFKtDnIUH73sHfqkToerGn3jeBf5astHALqnslZr_e7kYc2BghJ4vpvSwxOAPBWAzer-pNWcL71niSso69psZihiTmHcA4h4JnmwVxzxQPFlFz9eu55k8deihFg=w53-h30-v0

f46b0fb0-8332-4baa-a998-5437586d0cd3

https://lh3.googleusercontent.com/notebooklm/AG60hOoeBFtU94rIuodd-Kp1hFzrK7rsIe8NfCLN0RdMEK7-xU1vNG6e97SCddDt4FwYjaIMlO3ZAP8X5xd52K5uCRFjBT5_kBL2wwC-WEwtz69vcQbvwmuA7Y0rclykB4RbLq_z3zyS=w39-h32-v0

ce4b6c6b-2920-43bd-b847-6fc739513aab

https://lh3.googleusercontent.com/notebooklm/AG60hOq33jzrFr1vcNOk30I6wuXG6VBQk55I5jI6avgc3f8eDzb1BjTivn5sN7_ydlp9vGPQhBkdV4RQ4WwAdUnwgXiFffsy6toAgoN0nm6s1ElorNMhymazg0YdwputRNRBYfkrsfN5=w36-h30-v0

180f4e1a-912e-4711-8bb6-1c73701997fc

https://lh3.googleusercontent.com/notebooklm/AG60hOpa3SjHFRM3YDLM9JfMgz-W506yhhFcFGu97RVAZ1c704xIzgThdm0cuhf4GbxAgXMPoifEkMDXmmwnPA_xQ5L1ZHRX6Eo--lPeD0bIeXWXoMb40QFj3H5mKN5HPhMPPh6dD-imoQ=w99-h32-v0

74d01522-14ed-445c-8363-e7b6dfe002ce

https://lh3.googleusercontent.com/notebooklm/AG60hOq5LqKqfGbP8Kfhs8CXXnhNvAx7t-hFb0PDE0U41uIH8M7BY6jQOysvzkV8_3ReRT1N9x4bBLFE93TIh6Xaw4vr-1YZeZATdhBpBW6qa8Pt5zQdkC98juJ2CF5mt99InWY7jYKb=w96-h30-v0

0cbd9220-ffe5-45a3-b434-b223ad588a61

https://lh3.googleusercontent.com/notebooklm/AG60hOrajsNL2WKQCpCrrWVDIxtC-t-vRy1PH8VQ9couP-2Q3-e4FRFeMugG61k8Vkj81p8CqZ3wTUd_vELITU4CvehuTlAyJ2iU4GyKB4K2gVklO_SSPfnv4_wS_8cD9U-mLYbYzOq1tA=w45-h32-v0

1a552f3a-9b54-4a1b-9ead-e361ff14b8e1

https://lh3.googleusercontent.com/notebooklm/AG60hOpJjG42vQObvvv7X4XC1DzZtjIiLS1OU8CgDM4XzHZoadCOn69DycTCHqnbnZ7zt9x4XUqvBTYKuV0Zh-IVHn-aKkaDtgNExD9yN2FaaB8OaTS0YfurEUmg8cpLb8h-6E7sWRKBAQ=w43-h30-v0

d219e164-bfc4-4cf4-aea8-0ebae7fa603a

https://lh3.googleusercontent.com/notebooklm/AG60hOpgrmJRAhGogp5EzWAEVpYLZVdNz6zDSNqAow16Cb5WsTTtBMi1wCBCe1rqNWJkJ25l47FhQj-nv-aaY6n1lQP4t7Mb4ESWunOP_Q3Elbz-DvG2ZHbigE-uN7_t8JXesG7b6B87=w38-h32-v0

72c1b267-7238-44f4-8e43-25fa3778741a

https://lh3.googleusercontent.com/notebooklm/AG60hOqx5rGSvOXcuBt72k43lCeEvZyAWU0Kz4VbACCCvhg4u1wD4jYCr7F3Uoo19F1Vg1KjAtuBryv36Qdt7zfF8RbuHsOQqewkAMnuA2eCYJWg2lyxGMDEGWrrWOej8HM2UZZNtWRr=w36-h30-v0

83dadfe2-a0f3-42c4-a72a-0fc3824bb958

https://lh3.googleusercontent.com/notebooklm/AG60hOqrZTm8ECJKVs_TqU5Cau4xAjBz5DfHIR6d3tYd5NLsZpdA_Ky53a43unwsjHLgTrv9trzIoVLXH6cWT9u_bqB6RNhe4FysHKzBnDhrsDwPNzJxuhWZI6f8Nz4o29dWqkEWFY-FCg=w115-h32-v0

58b62608-93e0-402b-9927-9a1c14f5b74a

https://lh3.googleusercontent.com/notebooklm/AG60hOo5x9PTtujVH_uzCIl0wYHXlYaIObZRql-vTTLMnaX9siKZJ9MdpD6PZGfCop2I8IRqvGztNG0FeOxCuYN9o0akMAvHUb0Z7jy7wlpZ6xTHTyVBsK_ah1LjiqpGCXj7i9Iu6NTiVQ=w112-h30-v0

6a2176b9-b7a7-4271-b37c-3d2e9dd06d67

https://lh3.googleusercontent.com/notebooklm/AG60hOoInadZONvf-hXzAjQ1NCbdIW6a_e65fmFvb9zNVCYEBK-wUiho56gceAo9zSCNDkp_-5vsDFTFE2HsDlU1M65JbRTJUOGw39cED5q152jUBZJdqyumO1A1g76PB4gvuxxy7LUE1w=w68-h32-v0

881f0e27-059b-4661-aaee-7d619508e215

https://lh3.googleusercontent.com/notebooklm/AG60hOpmzphBbvGOUPiREczhc2ELjzMooJL3_N4mBQ5jjeNEwWes0KPashnlv3JAjYFxMMLWWlStx96NFQPBQ0Uq_XO3-MPmj1GADk1MSxqTo-KW4h8V6gCQx59rahFixi0uTJWyC124zw=w66-h30-v0

fcdc1526-0e40-4d45-ac13-872866aa83f3

https://lh3.googleusercontent.com/notebooklm/AG60hOrjXzK5bw_hMtaCk4ip4LePXEEYjYMJGkwb1lIrGHNitIcsKDzh3PsYsQxAEO2xdjun5f2Vs0Z9x2kDkF4Q8Yign5jg_gwh9FJYfFDO7pXg5sFV0HOAqIISinyLYtkIVpGGkiaeQQ=w46-h32-v0

2a27e8dd-9fc0-49b5-aea6-889ec79c24e6

https://lh3.googleusercontent.com/notebooklm/AG60hOpuEsrkGQQvQueJ7TUWIhMxqI5EqlOxMBSVXJUUbLzln71nihqKxgSTOVGSG-8X-ipwuIHP2sjxkQdaECrEKpf-Y6qMZBDAjWanUXdLhBZMqxHwfreD2bKvVBUzwRIZP8pWbLb8sg=w44-h30-v0

8d95a5cb-e9b6-44f1-9333-a2f4cf3e5813

https://lh3.googleusercontent.com/notebooklm/AG60hOpfrNUi3t2Q3ZPMtnETjGpE1SdmM_3J7W4IGLFC5CnGBkqq04xPeJ0rI66EcoZPI8e-Sbmuq6MXWnoF1OApPwSEEj5ui8og7JcADt17yD1bzmqUfM9o69jxqag92d0k7dku8ZSn=w74-h32-v0

41c232e4-6161-4d20-a1df-97bb438aee21

https://lh3.googleusercontent.com/notebooklm/AG60hOqEQSdNBZHgFS11O1w0MyCgHwjyTt-XykdelmERh9KGJ1fGI4NYAKvxVyJpVs4PIs23lELLqcZF5Wb8CLY3fW0gaXUZ8Pu3SGmVt0jS8ruQv-F9OPLSPuQl9p_KMYmo_zYrO3WHhA=w72-h30-v0

a9283d7b-15a6-4431-86d3-684144acffd6

https://lh3.googleusercontent.com/notebooklm/AG60hOoZzkUmPajZIWCOBuOliD6b-FDzNPA_VcE3kMiOdy1PeGmbkq2X4Uf6l5GQH_uKwra5rk6vR3ZJG_iQv00-HTgxLRT6-a4OsYqLc7y8Ke5Iwy7uKB4z09npbOy1UHa1yEQBDR8cTw=w38-h32-v0

5640e315-40a7-4133-bf2a-f35414c80757

https://lh3.googleusercontent.com/notebooklm/AG60hOrlWrhERvHdiTh0VZCjuCLG7JOI262WPHClFklhEDYrKVOotPCnDmLIsnlxrD711XYO0HlCLxFXP0sOsi4w4BUynna8USjqV1e_CMUg7_8jWIN-KbRjbCXtP8cAjye3y9ARoQKf=w36-h30-v0

3f1ff926-1829-4f15-9150-5c90072937ea

https://lh3.googleusercontent.com/notebooklm/AG60hOrRZjlwGpwfmA5a8vQUw-tYhqLPTtn8gZyzAX-FO6zOfD3V1URA5N0Lc5acIHH2BUcrNv3sLQrGIP_WGimb6loLrIB1kqRdWPvKPaRyuP61wagcpAP2Woj8tLEKF56LG6GE_b4A=w64-h32-v0

1df54ba6-dcd0-4164-b209-97ffafc30a22

https://lh3.googleusercontent.com/notebooklm/AG60hOrpyDKzgXAqOUVG_nyv_Hj5xQEEBX00KmhRKkQ4EP1x3wx0i_Ht-wdkNPO9Xbx_NZUz69n98aTlcQTuYJG4JS14G6X5vIxQuax7TQzV_5GfoydKn0pcJucrtlD-JY0AoQMcVpR5wQ=w61-h30-v0

6aa730f9-5c68-47a2-adbd-e9d5b7c80289

https://lh3.googleusercontent.com/notebooklm/AG60hOr8rc2DQjNDEI9zZY2JvXtwrsT-qMb-Zw_WFQuPICQ-bIJubAan2IuxcrJGgcVedONXeepwjUcVvQm6G02jyJiBlg9PSi4nbOwWTCxjoaDuJLACciOvLHLh4FXdt_J8Br3zBMFx=w86-h32-v0

91e22096-6081-484c-ab6f-27f65e48cc3a

https://lh3.googleusercontent.com/notebooklm/AG60hOo02tDMVMW_6QfAktIOdjqMjTVsvb96DnYdLsUfUEDrFa6Ly5trepVKNMcm5DoDjs2yJovucsq6E5hOswefFDNdpb32gp-EbpmXYBM4yIZIw6QnpCg-cQ8gT3auul9S9CLqqsvrpg=w84-h30-v0

2a7f910f-9594-484a-98d7-e6ce8340e370

https://lh3.googleusercontent.com/notebooklm/AG60hOrDTO9kF63DAvuxSpw2usTGBkf1iWYyUuZWSj6265LO6oU-hMVw_k1Jr3HeTQDNIt_v7M-vWa3AkiFRFF1fnDhgU-3mHwtkCKTFpzpw9vHDAjqtx8pFGFt2hDxmw2Us5rvNoWVtxw=w74-h32-v0

5701c59e-401c-4a64-84fd-e83d81496c07

https://lh3.googleusercontent.com/notebooklm/AG60hOoUAv1yXi6vCE9wpQXUEpbLRg38kCKssDtCljZ4uJnG2Q9V66mVupUXbDWcPeAw9XsF5UNT1cc1OKKIroOUN5MMgKz7Xy7ADXwOzGqWH08OZtHI9QmHHgtlJX7sEcljIUusUep6uw=w72-h30-v0

72ff187b-cf4f-4b5e-bcec-d5412e037947

https://lh3.googleusercontent.com/notebooklm/AG60hOpHlwu-k-krMDqDXARy8-_zgPoQ16CHIiXfBiL2Iil3FlzIihgIbEqqjqL3H8ZcPnGe3B4l2u4W7t9S1_SnDXwlVXvuTw8nNXlpGAaH1F3P_9GQX5sgZZ0-JOTqaARSJ1ZSDEOkAQ=w59-h32-v0

690b67da-4c98-4420-a57e-3b0b92dd7f42

https://lh3.googleusercontent.com/notebooklm/AG60hOpG37t2N4FOrYnnYLwhDPIcorWWw5lvaVRO4iKb79YjTzm_PrDyTbslw3qTmbDIQP6HqwpXB_J3-Y3_8LX0k7E3ZDsWYiXA2EkStcMMObqdxj2tiEvL1AujAFqQMGPtyaiOs-xE=w57-h30-v0

44c4fb89-f595-4131-b066-6581c0184180

https://lh3.googleusercontent.com/notebooklm/AG60hOoWfbuQZxaAU5sULUgsoNiTjSGlqcYu-eOmfBWE-4j4NQ4pzFHyNWQo9uMlQhCrk9MzL0Fjku8FEqSJFEQrofR4BaRi7Cj9gTICfFvcW4JX7i8N1bnK4BCTmP4DVt7qTdL5QWG0=w29-h36-v0

6d646787-a37d-4faa-9581-a819745ad01e

https://lh3.googleusercontent.com/notebooklm/AG60hOpw6FLvR8UcYQ-Fvd-ig5cKHtXJfQcXDY7VzAQdMP34x4HQgwkHHy_kZQ6QAa7PFQfGuBVFp7gLwb2I3lHLQuV2XClu7-oTrzb-m7don3etUmoP_MFewtU4vbsGipDBHv9EIsWZug=w26-h34-v0

479e4364-be84-4123-9423-84311073027e

https://lh3.googleusercontent.com/notebooklm/AG60hOqyZKezNCKV81ip-jfGr83sa5nfmFHQzvP5pd7-XHQ6gtvAXZNTwU7BziVSlGPOuVzf7EAw3mmq28NVl9cnhix70vfT8R2OTYhDnx0lhUGyrM0cDiPoVkr2Hwp_N2TWIvH5rnBoJg=w33-h39-v0

2fca7ac2-9475-472d-8989-cde638dcefcf

https://lh3.googleusercontent.com/notebooklm/AG60hOrywi5kMsEQr9lBEMRf48rZfGR0wLvrl4BNcVbjD1a2thB6ZT5bgMr2N7l39rslBcgngiixO7wmX7S4x3SszTrfV9J6lhhMyrBVvryr9MYM_vJyQybzJzm49d_2qUK_QOlgK4rLFg=w31-h36-v0

e0b3810c-2d93-443d-b7ce-d915cd83da35

https://lh3.googleusercontent.com/notebooklm/AG60hOooVmI2v4irm-8Dv27GnY14iGFsifluh7rL6xwZ7ybLWwIWEVgTcAUz8fdz-OzXWbq_wxrDC4G9uG8sH92iXOoKv-RTP_dUm9ePKwvGXMnTyMF-x9d8eXxnuS0iRIK6_YnHa4b9=w24-h32-v0

c59b8c50-c966-45f2-8432-a8a85819dab1

https://lh3.googleusercontent.com/notebooklm/AG60hOoZYExai7Nx_VlDXWwGxLW6JBF4TfJB9U-nR6vS5hoMh5b4eUaKBxiE8LhIBKtXhQO578hZ4ZbNz0JYek-acCPjOCuCJq-7Y-oOJv_Eo1YF3u8XaHa4WOkU8OSfVy7LpfQxnbLH5w=w22-h30-v0

1f087edd-e6aa-4278-ba39-4174b6c010d6

https://lh3.googleusercontent.com/notebooklm/AG60hOrOPJQl-i5hcxnsVDH7oNohGlG5t_pEb5I_xqzjo1zA60rUgsR2KN3-NDr7XnfB_9pxpdN2wZWCS3F4NmrHWVpWCG--M-jcSKJhpCGalMwFx0chTN7FarEfWOuHLYJ3pFtTAxf-QA=w37-h39-v0

405dccd7-1342-464a-8896-6ab35be0b1ac

https://lh3.googleusercontent.com/notebooklm/AG60hOr79_D3W3QCqmutMOW_wTIxIpAyp07IrmlsD0UZmqAUMSIlXXRhDGXzMS1joN52GyblWwJ9gN6Y34XXuHdwHopDBlipixjfBDaGoUTRsjRGK2PO5rh_xfxvvGZ7_M9vAdS9jHs3zA=w35-h36-v0

f58965d9-4bf8-4ddd-8e21-03db936cd595

https://lh3.googleusercontent.com/notebooklm/AG60hOp-f9_HuA7JWAWLPgqcVEDiPV7nxKARXP7J7OsEvd756mY7v414Ar13n4203aXO8oMSvFOHSp1Yhx0M44O8BrI6kxbeQKMtP2GageIxVHoacrMRoTzgx51XFRe0abUqM45bX8pgYw=w45-h32-v0

73df9e83-b1fc-43b0-a766-eeabd25d58cf

https://lh3.googleusercontent.com/notebooklm/AG60hOqhVNgS4Qm7YpIPczVvvo8bukYCMVPjns8MiTebzFQgNaJajUu49HFKazwdS1AEj9jDVX4Z9ciWauDBURcJyDZWP298vRsgX4cAC9peaXZ7XO4KvNKgmYCgboyogv6CIm3IZR1Z8A=w43-h30-v0

3bee980b-845a-4fe6-893f-3f0ca4dd7b2c

https://lh3.googleusercontent.com/notebooklm/AG60hOr-tjyvv4SaS2vwziCI3zVU_BoKxIy5uMR1LRwlOIcYh0eybikZZDjeu-ItAnui8m80IfIMZVRTRd7sqcuEXJGbn6w9Xb8FF4xlDeie0-Dv-j9V0pjZbf6DHSV6N2mG9HkI5mD8HA=w92-h32-v0

7bb5d356-4e3b-4604-ae0a-2cfed915c0c4

https://lh3.googleusercontent.com/notebooklm/AG60hOqL1n3Aey-MMhiFYDF4friWd54CEDHyElAkx6J-SORbyqEZv9KQQhMVuxTaSzL9rtlUyldd9zxWOROz7eTmcCvn22PBYnC3nhsZ-Q0g4OiWgBaBDs1vgrbj5zV-2bmERbBu2yys0Q=w90-h30-v0

f52cb7f1-ff5b-4aed-86ea-eaed23f68398

https://lh3.googleusercontent.com/notebooklm/AG60hOrUZOYFkYV7fLwKZy5nHvkSHkS0aV9StHBkl2xXkXWbdg2_gQjdGsayAK9nZQtXmtd8vZyzdjPBZIwJRTl38oiHQ5k98wMYIfzC5-7U62LQvRYXYsVl6BH1-AhuDfb2DN4FVZ0hZA=w46-h32-v0

4372910a-f549-44bd-ad71-53ab512096c3

https://lh3.googleusercontent.com/notebooklm/AG60hOqnpXAXXoGqmLo87WT9AyjcVnkcnQcstigkjfqYVOr3FLt_wm2vz0TMvj1nB6Yt7P6EogZPk1bpR5QDYoP7fLdiPLSCyq15AVs75dJXBNw0AAX2X9VcBYm8lFEyPVovmWkCMuQsJA=w44-h30-v0

b162565d-d6ca-423a-9581-9957535715a8

https://lh3.googleusercontent.com/notebooklm/AG60hOrzYBgFvzb3UkDM1msv_AZsHo24sy_zzLWhp8JZcnq0m4I9RR4kjGJNXZJclCHq3BOzcbG_y5xcCNNg5M9EuhT_wvicsxcgAQh6-zWZM54HEuZMSZFeA1axXWKE2s74QzaSnOkCIQ=w97-h32-v0

f3ca7857-6ec8-451a-93aa-a6fec28a1752

https://lh3.googleusercontent.com/notebooklm/AG60hOrVXJqKxVqy4BP0xkbrQY2CmppeEM24GgWKMeF88YWgZdfrgILsg3p4YK8oGpaf9OIo98T9A3JeIFN7DTF7w0DKbEGv7MDvdiPdRlfBPZPyOkSV14GV3ROdol5WEEhPh-vyVLeWJA=w94-h30-v0

7e90a6d8-4e64-44aa-9d5a-5513ff830510

https://lh3.googleusercontent.com/notebooklm/AG60hOr7Lwm2J46yfILeJtkxdESHRLBYZgo6t8HXXpN-8VdN0D8UVrRIGul09UdFGMMTnnga5ibHXjc-7mv6CftBY-XfbSbIYW6OfDoFKrWTm6Wfwx6mMJL8OKcq8gahYAD2tOauJuBATA=w74-h32-v0

bd36cc05-94fa-4949-9b32-09deebf8cfc4

https://lh3.googleusercontent.com/notebooklm/AG60hOqRXOTWSZgQQtE0sApGZ-1kQK9OPQfSYYlGytSxUvkbAUOsMbHAJFCSbZ-ED4hk7fB5w_Ouh5ZFc-ftiL7WjrFFvcCL09hcD5xicckhmap44KQHhf3fti3FGXKE5d2p3w5l92A_=w72-h30-v0

dd13d163-2264-48c9-a638-385f9c8d245d

https://lh3.googleusercontent.com/notebooklm/AG60hOo4JKViiKuB7bEKpg3XXAfa3F33BDayAAp68I4OeLe2shEL9jHjYI7GR_ghK5yp27SvN53mJP5HmspY5YwoQ_alHDRAsTUEgqYgNYikGZ5nVZcXGqkI3jhBRXoN1jQTeiRz6ha3yA=w58-h32-v0

86b896b8-2f27-4f00-a37e-96e8eeb5e25d

https://lh3.googleusercontent.com/notebooklm/AG60hOrPBojrGp5cC_eO0qtzpRKQXxFkrXv4nDzLzyblOGamN4QzqMgmcAYIeWgRRO4aPONOCCelXhoKSZEsvbaBgo3BYf4slT7Y2fK5I7AddhEkKWbSE6IqxABuHK3UVjZzkSrZrlxDnw=w56-h30-v0

acc9f6f3-ada5-4a95-89db-65e87bd66ebd

https://lh3.googleusercontent.com/notebooklm/AG60hOrUzpxIMejmIy66WMMFldzrz8IoeBxRc8QfLbET_fS6gq7-X3t3z6tIntMQzh3qt1LQdDwwias8T7F3wGi9TjUZ7u9A6y1MRBj9EQW13XMsXYNkUZyeU-jpI8vgF_yWm1Wqd9hxxA=w58-h32-v0

88d8a3dc-08a7-4e00-b000-5ddce41984db

https://lh3.googleusercontent.com/notebooklm/AG60hOo6KAnHhBfPrERa38eeJeo4y7skVlZD8nZumd_Kt_MC1iSQ8Y-nHNFd1s-khp3TL81D5630ehfOKR4axVEFRP8ZpY9Zf7VQUm6FU6L1ePVeLWQr7GGWkCvXcUNGpyFPrJQXepKkdw=w56-h30-v0

43a699fd-c7c4-413e-9573-915d14433456

https://lh3.googleusercontent.com/notebooklm/AG60hOoS2HyS_tIwiynSKCWZPoBcyua9ZvhYOaFxQ1bUcp3Hipc5gE6_pSWqBbAsMVO-su-ehnS1nu40d0gwcVOZBL8QvAyb279_sj2dgrRcmS3VQ0hTKUldCPppd1bfZtMsrf1SZxA-=w57-h32-v0

81d365d2-7f09-439f-868e-66c08e127cc0

https://lh3.googleusercontent.com/notebooklm/AG60hOrahWACvtJKmlkmnVRiEkw8Cz_dTksor3wZBUCIdsxFjJYfjTuenPjB9vrXHdVGOYNvNUoVXx3MtE4CEEh5qdPjyzTC9KuZ2etJRIyrs-EkzM9wWJvHZvm7bNeLhKYoGK2c1XxM=w55-h30-v0

0fb29550-c03f-4cb8-a2d9-409cbedef4ab

https://lh3.googleusercontent.com/notebooklm/AG60hOrBpb8VfUd6eXXrgHPdtFvil93V-T0sPfDJfOoGu0nF587ATEyuLJmsySZwoZW4OoHVQkexS43t6TI4OdSfcbecl5cs_toZ8m7o08_jsdFbH-h_Z5f3aD1aRD9b9gac5QtjF7s8=w29-h36-v0

f7fd0846-5214-4f9c-b189-8b53798124ba

https://lh3.googleusercontent.com/notebooklm/AG60hOqwlloFh1ODIPWShMBKFOQuBzwIPcelF8RwHhQtbDOiITK7JYbPb1jK5XtIUeKWAoP_IGlcfNprIrKLoXCICPvGN9CnEtWL5MuvdbO-OjgbXCGgFd2uAPLmrW-DVKJ7gh0MDoYNsA=w26-h34-v0

ec957e3e-c478-44c9-822b-6e5329c4655f

https://lh3.googleusercontent.com/notebooklm/AG60hOrEkyTs_xPTjQRbB4xQUQc6oZ4cUKq8K7YXeB3KlL1xXTQqt8oOcvwzcqw54Cn5G6XjqpncwMsRBAloyzzPXt1HlnY2J9Rl7Vi7BRdEnTZ03mcF6ToGeEMq-DU44ouB8iIOAYb5qA=w49-h39-v0

a20c6007-d096-4fec-a119-58a6457e3a75

https://lh3.googleusercontent.com/notebooklm/AG60hOpk8l-I39ho93K-qj8JhFGHMf-WMJEY737_UhmBPE2B9dCmiwzFFIDjkuv31Yo_Wgu4gwDaYOZ4EcFxvKYgbilZHfhPBFjTOPGoYYxkTaVsjIydD86EUJtCPjfZj6ynHluw3V2l2w=w47-h36-v0

4422da8d-1e58-44a9-852a-f9ff67725a51

https://lh3.googleusercontent.com/notebooklm/AG60hOoL5TB8X0ukzoEdRrVk2Cc-JaGVt683f12givWVJLEEKfLh0cWhj4i3RvoXwW9psP3XJfMmAfXQdvMUHTfj73IGfIyhIqyTyyCHXiJjqlLbKIiWUAlD1GMMpqD_Fv6uGg0roebfMw=w55-h32-v0

44cf4640-d040-4645-8f6f-6821bd5dd219

https://lh3.googleusercontent.com/notebooklm/AG60hOqgvefiN-uoex64sk580OAdLs3XcBhMD_YrxiWHUPRaUtAHMkyLSH1Yo0ZtIj1F3oobC1TcxsTknJdgUjwmj2sj8oGghkFD355-RAqNMirUAkDTZzSyWyPbrKUJdyPqCzAlKqnWUQ=w53-h30-v0

571a5588-ea48-4398-9b8a-184c3a682a85

https://lh3.googleusercontent.com/notebooklm/AG60hOpPs_yOOBGAGn-Rll828BALkST6WDq8z5dipo4ACPpBg3rH10Fy-Q3L7Q4KZc9Vq6hh1kiw58pkjP3G9_b04tY3VAlkJuen8b94pBO-7x5DixVse7LFt3yHacaeu1aum4AkFiAKHw=w82-h32-v0

539b71b2-2c8f-4834-95e5-5bb4fe24c81c

https://lh3.googleusercontent.com/notebooklm/AG60hOqCQh9mvAuDTBlZ4Em94msJvuOm9I5GG9PLiA71cXfrdCNd3Vbh_-QpbcncN_wPl4ZPvUTmQCV0gMHyv22abiLIa5G3alhV32rnZDBTFJGNEXWlH2ezHgZE0CkffN-Ua8bC8OySXg=w80-h30-v0

0a19bddc-4a30-46e5-9b3f-7d3217fa033e

https://lh3.googleusercontent.com/notebooklm/AG60hOrd1QOR9VpqhgwJl3ycyXBDGwZdY9N8A6qahKkPcGwXYfwYL_SWobsQBzWpg1dBT9ZTBuWYT6HQgJfDyMMUF2pWHm_-UieuPXrppSPDqEwe5OhYREGW59CMczKa3TBzZGXv8g1S1A=w47-h32-v0

ecdb97e9-ef28-4b71-b844-14dc3ec9bf77

https://lh3.googleusercontent.com/notebooklm/AG60hOobKss4YoDRaKcXSDFWsSA-cS5FcJME6UZLvm1HOyDuhEarM4ZLrwgrROIPhB53Hn9fgRtIxOZDJbD3kqmD3wKAYcWbsHATS5bSdjrUdPVeLEXEiP2W7MGVdFdz24L7NqLim8N4uw=w44-h30-v0

d64717a8-2883-4008-b2c1-a2c477576b4e

https://lh3.googleusercontent.com/notebooklm/AG60hOpRI9JyLKsb3tM4H-BK_3uxT8BLR87EhDv-5u9HLVFtH2nFtjuuStaXrxKW2oBBego9C88lsgI-vFj1EXV02MhxlHjpwGFXUtN1yliMbpuknH3FILy7wt5y3ptXD4emeLcudrba=w86-h32-v0

ed8e6a23-384e-4932-a24f-eabe2665e120

https://lh3.googleusercontent.com/notebooklm/AG60hOpwewXheTlcpS2mx_TjOTm3snsNsgBCAew-W7vXPKufzbU2dDLnaRFZXFtR5rMaY5DL1X2hjqucUCsQb9vszA2-2gyKT4HIz_bKv3x6UX5g-TM6eLmKRiwCDRA2WRdS-g_XWb8ybA=w84-h30-v0

8a3d9aa8-79ea-42ff-8df9-544622559416

https://lh3.googleusercontent.com/notebooklm/AG60hOoEwE3E5YyEjK6MT0C9E5o4JcVOJrhOA6uaVCWUmcCGAm-klv4d7q1lb_nkcoIir7PucldQVSRwvaNiHRqzfOHM_DNPHMxQZ0LynwvuTFypz6z96ui52k6jrNGmnK7md6W5ibposg=w33-h32-v0

0379a6cf-c7e3-4197-9f26-9d0b2b73dbf1

https://lh3.googleusercontent.com/notebooklm/AG60hOrNA-AlzEZvIGgMVo1D6fizd8_UfJJ5aFwwVxFoLFtAk-Eh_Oz29S4CAhZS1MSpSImdx2ezBhbOyd4ikwhpjsxd4H1_aEYIyv5-nDtquzjPBBk2t02n6aKrrSa0eQ5V6zpId7QLig=w31-h30-v0

4188c67c-4d08-4411-b64b-351c6a54c0d0

https://lh3.googleusercontent.com/notebooklm/AG60hOrBdxeooU2h35kIiWOM1F6Fj508SSNto_K-dB0e8M-twBNjyWvYnzIrbGxSq7latPfSLcVS4bzsO-wV7eaAl1mTfAyB-X8OrTDvjxISVxLvd1tGl8mQiXsH9z2PRHse0hOdq4l3Pw=w53-h32-v0

629a0051-319c-4977-8996-c5c0020d12b2

https://lh3.googleusercontent.com/notebooklm/AG60hOqnmwsAqP8PlO4ELOmBkoSiT4fgIsXWmtA5_ZEQW0-1o5MsTNMY65h9oalQG4uHbfZQ4ORL2ve-ZqSQwwVhYb_MYwzr2bwELGNEhp735PAtMdVTcB2SPcUVDR3MPnrN1x7Jgy_t=w51-h30-v0

fcb27f48-0b44-4a48-b318-f27a4e1e0254

https://lh3.googleusercontent.com/notebooklm/AG60hOpR0UWBU_F9uLkSfJDR2KNnMbXF1JJjdqAF-TuCiw7pRyU3RUVta1CYBMD3Jb87KYTxLBS1VvZ46sQyUTeV4ubVC2F01WGSu1hI8qNIr1d2abUwRv4lyTSk2kcyhPnA1JtkTCV-=w51-h32-v0

c5040b18-83d9-41a8-9dda-f2bbf4e846e5

https://lh3.googleusercontent.com/notebooklm/AG60hOoLr1jd7gdKY5esLpNiSw4_HhYKFqDjo8eK9l4DkNqMsBisC7NEGiycQN1QBakUMfLaJxgOtgOWcomdw28lXuKIu4mnvBsPTZvOEr7t-PS1O0eZPQgGifRQBTIivGbAHgnzOcKYOg=w49-h30-v0

e4adb8a2-d238-4544-9299-dcf0bfcd2087

https://lh3.googleusercontent.com/notebooklm/AG60hOrRCJVVm-vrHz1Zm6YHjdtDxCV_Qw42P7zc2T4DII9qS1pBUmeATUFUqaKylbqTmA6YITMNLgcW8EJvlaV4PsjBSBiNtMV6GuvWXd_A1gStNdJlApza-lGKMx2ENfZ3UBKYnSjpZw=w61-h32-v0

f7f97185-b332-4f90-9f14-cf4ca4cf2237

https://lh3.googleusercontent.com/notebooklm/AG60hOoFlbqUrW7hs61UnrmzVPotbG0Kot5iYE-pw9AUOKlFGggjv0M34ElGpZTYRGnMjgAWGolgZvzc9xFtlDAaQBBX2T1L8Kc_W2-5tArFwd45gKCd2KXdq2YtKBDZloxRybpeH0Jizg=w59-h30-v0

e12d1997-ebb7-4353-a852-982fe1e0f4bf

https://lh3.googleusercontent.com/notebooklm/AG60hOootFpDQKfsxc_gjDjDJUxQ8NdXEzNWzYscpgTgjSVJXSk-y5K1CyGErHq7vU8wC3D6LdAXF6QPVtde4duTyLbs2PS-vZmx_iEDn8y-0C83WALrHehKLSw4Jl5rgJ277ZIi0DxZ=w103-h32-v0

bc878c18-eb42-4ace-b1dd-017f37822f3c

https://lh3.googleusercontent.com/notebooklm/AG60hOpu2ueiyTSkRxv-P8liAIOcQNdkB1D2zqiTY0HZrF6L5Ku9Z46SXVi0rOSo0rfeKNgFm2fvhzKHnl7Kl8-Pd5MCFuqCYQSyCO66a2UIY1v7UN5FsDutuRgbCPQeatT4-Ely-VVYBg=w100-h30-v0

6f0abfd5-d99e-444b-8998-49aa9c0ff8c9

https://lh3.googleusercontent.com/notebooklm/AG60hOr-AERjowTihLhnkMvT9G6gY_DGai8wBOvwcWXcP0FmxpJWcSKeiG4TAYbIWwX6gAfSPxg37zFP1tRbQhjqbE4vPYOmJR3vQ6PQ7-IMOsVX5uIfbQZi5tTvG_CjpBkXVGlEOUxMCQ=w86-h32-v0

363e5e07-32ea-429b-9b28-84d93e512c47

https://lh3.googleusercontent.com/notebooklm/AG60hOpAvM3IVYkC_EPI5kd4wdM4vKRtO56atwIs4O4Ol-rDDnfGLy0ixMrLPvm0xpPHxFfKRwRjNIWyiQjxhWKSZ7H21Gv8L6eMEOLNIhmou8ZqdnyiYjqE8dorw-RicIbLDItFjusOwA=w84-h30-v0

a2da1d4a-f4ac-4f89-bfb0-66c933fbdbbc

https://lh3.googleusercontent.com/notebooklm/AG60hOo9ZZp2iHC4pxNyvOMq_gLlFDmvJaPZ8fC3NanDq0ni1YCcpZSqHcvNvg7Be2JBTq-y3m5Nw7z5fqAwveP5YJxLcbGI0vkkj5Cvy1mzudhiDcqPnHiUtLWqx7KYpocqsu7-g8MztQ=w30-h39-v0

aa7500b6-9712-4aae-bf6b-02878d5b5e36

https://lh3.googleusercontent.com/notebooklm/AG60hOq_drI8jNLXn27_0M7dIv92nFgsWgZ6XEM4IwwuF6Oab8rneYAx2zSqulBuEv8e7t_aSzEp34yPbSVdLmkJLyIdpmCtM4g4s8oc-qO5414I6yMl_T49tpjJHznXGlGjNVkXnuBxOQ=w27-h36-v0

e6c5b99c-9207-4669-a341-96b8eb7db44e

https://lh3.googleusercontent.com/notebooklm/AG60hOrW6DeqpGb7OaBQevlLZXXJWMGQuG4zWKa92-rWSK0AXbcJOmbXHQsf5JACs9S3qcWvMkuYRxKg9AQ9Z6w97OJqrdHiuJcteLmKZSSQ_lpGY-SJJ-IVIzOOtwJ3TrtjFuc9yAAXzQ=w55-h32-v0

43c6cfdd-1cde-4a61-8d8e-323f1d76164e

https://lh3.googleusercontent.com/notebooklm/AG60hOpAsNRXKkH3NQrASXRr5QvzQ9Nq8Y2VQw861CWcRXrjVfW5NWP_razIPeyltARp0SfDwDopxA0uNT5MwX-gJPfEf6pv_VtvOheL1NGP5m-5_vDw-wW6k_jMsa7uLq2AmMQKOqiD=w52-h30-v0

d86cc13e-0a6c-4b8b-a8d8-b083474fb629

https://lh3.googleusercontent.com/notebooklm/AG60hOpqIttXZwSzWclUZB8eO_rImOZAhRTuldjOwuZUKJpCw1eL4lj_6zeEmF0QiloJQUPHjZ5Uc-KlUNPvj1a6gM2oJhzYRhqekJ6CgS5TLjvqZqaECm1z0KLZYCA362ulKBJTGyqT=w106-h32-v0

667e6b12-ae32-4aad-8928-5489c1e211d6

https://lh3.googleusercontent.com/notebooklm/AG60hOolnpZD8UfRtNRYF-Id9qIYf7Q4eJqwLkBBv4kG1unTPaOlr5MyT2WJEgi-hCq9f9RLwTSwLH9ir2prKWBslvTF8AMqpaUqa34CEQiinNXwuQSsd-kK2A8dS7ZpR_yu7C-7Wg6x=w103-h30-v0

29967d6b-a205-4f9f-85e3-fa6b00def85c

https://lh3.googleusercontent.com/notebooklm/AG60hOpT2UjVf4j78pbZJVZa0dMbKw_BpENbPyPDlzhuoC6i81JXFQpkTlEPrqm2zqYNPdYLSOWKb6YFvBamFWrQPQjkc7m85xVcDpEd3N8oiWtRE9Acx7gwlho0A_pepoQ4Mb619TGTjQ=w97-h32-v0

f7cac8d0-1c5f-42d2-9855-6dbf8e59bcce

https://lh3.googleusercontent.com/notebooklm/AG60hOpWQfHZE__BsLlga4MtAPdeukjqxha2TuUtkvKWIigiD7so8q89FjBRUF5Gxhy2_QycLUb60A_xadxjzm92ofjaOIhnkJYpKJWX4RIT4gqsPBqJqBk7yOKBzTHIk6sD3pxmwjWz=w94-h30-v0

56703678-07da-425a-9bee-8c3eef3cb0be

https://lh3.googleusercontent.com/notebooklm/AG60hOo2OzAk9k9DjZ3YCRp_mqULODV4VN-3OUA3c-4JxGBg6-WIaf3UT5906fHCpA4jlxIH6D23XULIIoy7d9j7mSUyhRd8Dna8dnO2yZpkNYyB6KPW9IrsoqOo8dTJ6K7BNjKzgReE=w30-h39-v0

f1d90a20-c894-494d-b537-3bae6af1513c

https://lh3.googleusercontent.com/notebooklm/AG60hOohqx2a3m5bMp_QiU2MryuMWb2KEhnX2PJCBHOmuw2WdBFMMbkW4O8XbNwU9JymTsWiLhp0696HfYo7djPJGm34Dnjo87Vlb5-d_SBP6xoeI1-x0lkWHiyvOF-bd1Oa5xC7Vd9a-g=w27-h36-v0

bdfcbf0c-bcda-4192-b6b0-0034702efd18

https://lh3.googleusercontent.com/notebooklm/AG60hOrxCxhwgGilDI_wZ1sGb_gsHNW2Ld8HFL_7vrIccUdRTtaDrqRi_ouCO50M9RTNNzeZCeLEhacJkcTfSCONg47jCiXUd104TYC2izu36MWMmH7P-U9h9erNFGbOHommDRIcsIUxIg=w29-h36-v0

176128bc-51c1-4378-b8ba-68e8d3ae5374

https://lh3.googleusercontent.com/notebooklm/AG60hOpP12gNynMIbVHiV1Hf-_Wia3u_AasnktCxGhYUWBRRBVCjtVnkb-WWkLfR9gJT_86gMW8hA40XoyGYNJ7pqrvZTj00Smucvh8jd2UcK9GN6HpZ0kdDgXLxiYkK7X169Xr-Gljh1g=w26-h34-v0

84ad38a5-124f-4513-99d2-cb34e48b6e3f

https://lh3.googleusercontent.com/notebooklm/AG60hOr5A8DHPKE6MEEkVynGu3W_5Pro_NojQW5LRM8HDtsOD3OD6ig8g39ejUDto_PDRvqp4MMuNLYPelRfu2H0SrifBT_Dah69synKKZb2x67SMsAMxLqK7qMRBB3HsEgWBI0tBqC04A=w34-h39-v0

16566e8f-74a4-4ba0-b7e6-2c7c677d1668

https://lh3.googleusercontent.com/notebooklm/AG60hOq8keTgJYBHgVgGXHnS6BlJvnrhI04OyF8JR0-1ZQDTCQcs79Unu85vDidCUFdeBvLPPpstENQZhp2jx-1QKT7TZeYHzNkpbXLa363F8PnyWsYUcvgd1_mhEtnV6-TiUmEpgAYR=w32-h36-v0

63babf29-8825-4f20-9c06-47f0764f19a1

https://lh3.googleusercontent.com/notebooklm/AG60hOpzA-90KWwHyIHjH4N6pqT3qs5UbiAKpopDU9nuFH3re9_eCm0tXoZgy3pzcbEpHDfWwirDMwBdFXF3OVgRdlY0asDDJcJkvRo3nv-kIC4HPDRClsS0Ye_kN7TONWQcubA_X0EzJQ=w24-h32-v0

424fd8e3-79c9-4268-8437-059e38dd4a54

https://lh3.googleusercontent.com/notebooklm/AG60hOqZy76cICpLY4Ged0F-xicIbAZXSHjEFTMU-_H-a4xQuIvyDL0h0-Ohr8iT8SXgnqijEwZfESOXBsMohTOVnKcukHY966WYGWwbq3bpvrHUwxi4hoGHncHekYC-Hy52nYzVs5zfhA=w22-h30-v0

8937f30d-3255-47e7-b593-ac3c94c226d6

https://lh3.googleusercontent.com/notebooklm/AG60hOpJ2gXtcvtHdiptq_Ggq3wFnFeYY1d6zfCQN3wW3v-P_k4ESQfwwR-SjHP9ChVXK4MO-wV6nQ796ojNsjUBbTGxHt-kYPSraXuVUOrNTL94yJmTi181h64QJan9yva97A2lRBhF=w37-h39-v0

66b93ea0-ee36-4bb8-af34-b2e98b9ffe59

https://lh3.googleusercontent.com/notebooklm/AG60hOpsX_31hyHnpMeNxur1C5u5ewFudI46d4wixx3vGSmSLAgX9GIlLoOi9vG5BBN1T6mX6XO4d0xMi07-tDr6XHCl7UyWpZVnnQa8bSYPvOBs7vnvZJmuASp0PiyQvOCv9nhkuAwJ=w35-h36-v0

354fb297-dd1c-47df-9f40-40867e0839a4

https://lh3.googleusercontent.com/notebooklm/AG60hOo8baaYg_ohr98QD80Zu4Q3hg4Ow6ZH7iICuWBbUOf60-xHR0qZmykIPrJfAatkc3siTnaka_BhRMrVnFvrfBnQezRi5sNtQdvhU4KmRPXg73ptJvyV_8dXb8P3YOXVvCGLtx3K=w46-h32-v0

a33b9480-c0ba-4685-9bda-c632d9f88915

https://lh3.googleusercontent.com/notebooklm/AG60hOqcBMwZsmP5rVnSVLDql1feDDbPb2rpAyCWKrM5wxSga5ZyHwNvNeA_gXJKzsB4otvN7Sf7Up5eEW521_34H67SD5v4b-PW0GZu27fcjjyV-GGLDvfmbDAWLNBRMR0GnNoR3V6Q5Q=w43-h30-v0

73fe9346-98ff-473e-ac92-3ff4c23534f9

https://lh3.googleusercontent.com/notebooklm/AG60hOocGwBSUCwHhmVRN4u9KqZlSHdJRZXuKhrcm_vhMN1OJIH0NET5yW8dP6FsXhimNr7bd14uIagxqBcIqftljNGOgF4tL9nO-HoDiBbETRN597ce9WYXNPDDmoqZHX0bmwgn5eMWiA=w80-h32-v0

cd8f69e2-4e86-4690-af83-54348daa893a

https://lh3.googleusercontent.com/notebooklm/AG60hOqftl9yqlXxjEvlB7oXyTzTNf5Li01nsbYTrxMdzhdK4V7y5h7R6NQsnkYyZXFTF5FQ8Ej2lAyRHu5BTiOaXQseit_ROTl7WB8Bgjq4QZU0cx6PmQzMqoCzObOCzxnwRJaoP0emSg=w78-h30-v0

f0e87de3-3d95-42cd-8bf5-f37728a5afa0

https://lh3.googleusercontent.com/notebooklm/AG60hOrXMNf-wz-rDJzLV6b2p2eUGUDbTo7EjEPSS8-UjdzaBnff4lRvIB3JLSWGpWfHKUQN0mtNPQtNcz8b0ZY3KyOxYFV2MtK0DnXdf_2QR0EVBulLqVnDpokNv4JKZ6HQpaPuzmeZtQ=w46-h32-v0

63bd33c8-4b1f-47d5-8055-e81733855070

https://lh3.googleusercontent.com/notebooklm/AG60hOrGQAXWCRkZYDvHVy8MaliC5jp8aRP04EmnkLCFgop_JuSQ5WXQS0UXBTMsFwiL0VbV6ECuMJFECo7U_en5WyNQpNUQpCWbeKAE82dTHyg8N7AQg1YO8KaGSQXvcSwu2BMkIwTsOA=w44-h30-v0

0f4a8790-472b-49c5-9a72-c42288e02f81

https://lh3.googleusercontent.com/notebooklm/AG60hOoSzH4umqjK9HFLlXmbw9oZH3sPM0z__tmnIDULcU8sJCyjZmoy6iIldmKxkwuUyjeRqdiztQAVwHsTy-k8gCh7hEvgQK6WAr4g61DrX47yrEHi9arVli3WbkihFaQ2tSl75oIBNg=w60-h32-v0

ca6c9f06-b033-47f7-bf23-0cd754fa61de

https://lh3.googleusercontent.com/notebooklm/AG60hOox-B9JMWhwBRjPRFOWBalZFpzh89c_ZbynTGnmF5REjaOMNpygpOvzOy2fvnV5NNbvoF3yT0zTMQEn3_JHzTcwM4gAe-LsSeaVuu-sC8_E3NRsaSpfNDvZC5PSVhl6hgq5WVVJig=w57-h30-v0

8c182151-c7cb-41ea-817a-126c03bd9dea

https://lh3.googleusercontent.com/notebooklm/AG60hOoS3olPwy2Zgo-OADJsoYsSLrp2IGnkPa38q8IUJBfga2XWl0oQ7UYrD3jUxQhmw5jMgagXntu2HoQ-G_HZW7aredXL5pl1fzjbZ_AT-saMPh4JyqbTRHdBiHzCwve7rNXeTRYgIg=w63-h32-v0

a3a9278a-5787-452e-8e3b-0d27c5267b59

https://lh3.googleusercontent.com/notebooklm/AG60hOrp1uUf7dIt_HJQbEvkEVlThtQe1RoOcjZqsQTNDuPztw3zSjM9hMXyVSMHPXoQkeodAsSEPIvybQmjpAidf06_3foGdUl9BVR2uy8GnbWhARkXst-MFEdtt4jkll0KwmhkAEnhoQ=w61-h30-v0

99bee140-983e-4a7b-84fe-faccc9f789b0

https://lh3.googleusercontent.com/notebooklm/AG60hOpfjjKo5m_7syJ8VI_cqEXn4x7yWd_F2TvJwRtrElYA7W090Vtl88bgyYZ5_QJrlGZms6RtcQOUL5EEFP6FDsjQcGQ66gva2zm5dtx3j2Nk7Z9BG_FBZwJvGaCTf83hxrIYaUVg=w37-h32-v0

96f01579-6497-4b14-afcc-8a12ae301ecd

https://lh3.googleusercontent.com/notebooklm/AG60hOq1GeDN7mLbIlRikyFHXlVjFWnQmvlyhP29fVYZ_LF4cNU1CPWtXx6PjJsn03mliPoq2I9dPA4T7wfnKWUAOIpLA-ou6BVDxTMbBsMp2cvqcBEHKOhtiHfzFA3xRycvE-U611j5Qw=w35-h30-v0

2ed79f57-9ef2-4b20-89eb-4e19b02a843c

https://lh3.googleusercontent.com/notebooklm/AG60hOoeCtgUO3T7Th_UKWxV6MpIhKj0B6CPNxofa-GjZdRi3a-AN2ctzUo02rKmDUt03edxm7Mxmk8gIWznD7sbzcydAevBG0o8Hx_mAp6-9cMMNiFzp3XokrxlPyUALXa9Y7BwdqJe=w92-h32-v0

a66aa3d7-34b0-421e-ad08-2442627f757c

https://lh3.googleusercontent.com/notebooklm/AG60hOoTQ9OzmrTJGtbz4XC82NaRsj5b15_aTQGNATnPjXHjGOHCl1JpQn5NPOdvGfEZi7HvGjdmt58T9JYg1gujjlZ1oVuc7whSJ2gEr0XURkWTfoWi_hmoUZEZvjRhpYgOpHTj2TXenw=w90-h30-v0

45854d41-1b2a-4de5-93ff-a69964dd6bac

https://lh3.googleusercontent.com/notebooklm/AG60hOqCUuRKhde4-RuToahRE_dlNuIUpz6GltUtuVWl3oli2sY2BS6izcSzQgFpV3fMSeGg8u7U3-g8L8OFDLOxfoAbdWfoXsRLAep6enkLv5mBUyfydtaRcKEnad-ZjgwUC3hHUKltAg=w106-h32-v0

b974da9f-4f8d-4bfb-b5b9-35ea03c52964

https://lh3.googleusercontent.com/notebooklm/AG60hOrMm1cnrrhkgTVLlvK_CrHk8jVRJpP7ZW7qcer23bCX34Yx_M51wm_K8C45uNGgIu81C6bEkcWUIkkL1jIWJidN93aC1d7nqLPDFyY2M1qpuUe-yBX4OYDPdmCYnM3IwQSIuTal=w104-h30-v0

17ad3e13-05ea-42b4-b81f-039c0b4c391d

https://lh3.googleusercontent.com/notebooklm/AG60hOqWxH9ZLOFYMAZ5IpduU2BQhVtha0uXBjcCFYUAkqGZmLiYxvPLodi-oWx4LuboC_XAnv2-tOqEBLCrIjSMJleU7amznXooDi8qgP4H5OCVzbBEcJvTTKT5MNt2Rgq2v9hMiqQ2hQ=w52-h32-v0

dd6a3ad4-2978-461a-b1a8-43442ef76db7

https://lh3.googleusercontent.com/notebooklm/AG60hOpxWu80fLAQbsqd9rkU4Bpp90tOX9dDKoYQoVkXh6tmsMYmm18iLlIylLQu5kbQ0A8TxmtuiE2qHZ5_c-4qKCiNGDRownXRJaM-CbPyYCWxsqSmzWexMVkrMsKHe8ev1riVTiahKA=w50-h30-v0

2489cfb8-d81f-42a6-a8a4-3c0a4903b3b2

https://lh3.googleusercontent.com/notebooklm/AG60hOqXJqfcFZ2g3ymPzDGmTc3O0f1RgyUGGHaG61aBDKhwyf4U86rMZUYCJ8Bw_3X8-MCzzICT1SucsYCwUwLl4fCchaDshnQd_8f5oV-nST7LOeU5ppLA0GcHjHAPISeDDDwKsFFr=w57-h32-v0

c8038003-2f76-424e-bee8-0eff5f69a0ca

https://lh3.googleusercontent.com/notebooklm/AG60hOq_8DKtxS98IOYm5os1p57r3ispj-fJQWrfQm0ANUWAD2BYGh9vzigILb-wKEzEgM9PvoG2rVA8GO7hj-6u9EchtP0YxK6DYlDuO0q7tWJHtNg0FeEtju9BkAlPVbO6zZzFau4Pvg=w54-h30-v0

01f2ad30-8d75-4a19-b45a-08578df109e0

https://lh3.googleusercontent.com/notebooklm/AG60hOoS5cYE9PxF1CdBx68vIMdEo4Z5ZMWEikZIj25qEa0o0Cy9se-jjBvjvVQljCrRWA7wjP7Wxbks_Sd4HKYsYKBdR6RPouAjcF7B19qYWrdihbOzA7k_7wGXSkrMgrKIHKjnmg34Hg=w51-h32-v0

5aec6755-38bc-4ab7-aba2-278133f92010

https://lh3.googleusercontent.com/notebooklm/AG60hOokhZnw4Ztiup9XLbw1kxjlYVTDyD4A14XdK1HdcgVWQbD4aSEmHcwmcY0YeN1hQtS2YjXccyCbJCvnu00sPyW7dPquzGVbm1N0T8w0h6IHRMc32WgykwPCaWLZdSP1ihBpLEAmQQ=w49-h30-v0

f876d2f9-fc15-4422-a0b7-2cda5bb4f576

https://lh3.googleusercontent.com/notebooklm/AG60hOo3DXVVlGTvmkIyev-d1EC1aqAtY6u6uSavzAV4CsUUQqrnwL7Np5aAE3wIeWdpQR5djItCOh5FHHGlaOLGq5ycZWxcCIftMa8NtZmE57HO5BI5tffoKCr71VxlexscAwcTT1w_Vw=w260-h103-v0

d90fc5b7-058e-4e1c-a050-fadb4abdeb53

https://lh3.googleusercontent.com/notebooklm/AG60hOoJ-lPundnzMW_FivJrbYyfvEl8qn1vqBU07aRkutGR13ITYRyMDo34EbTXeu_HJiCoZKGsya_-363TamoWZpM9b5KayelNgKLQhhh2utzGohgLOZUMcVnjqH6ODWumVLVLH2I7FA=w116-h28-v0

16386c67-bbfd-4577-bbba-0facbe607b44

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## UTLRA LOW POWER GAZE TRACKING FOR VIRTUAL REALITY

## Gaze Tracking

## LIGAZE USES AN ADDITIONAL SET OF

## PHOTODIODES FACING THE DISPLAY TO

## SENSE INCOMING SCREEN LIGHT

## LIGAZE ESTIMATES THE REFLECTED SCREEN

## LIGHT FROM PUPIL

3D GAZE VECTORS ARE INFERRED IN REAL

TIME USING SUPERVISED LEARNING (TREE

REGRESSION ALGORITHM)

## LIGAZE DETECTS THE BLINK EVENT BY

## EXAMINING PHOTODIODE DATA OVER TIME

Images from Li, T., Liu, Q., & Zhou, X. (2017, November). Ultra-low power gaze tracking for virtual reality. In Proceedings of the 15th ACM Conference on Embedded Network Sensor Systems (pp. 1-14)., Sensys 2017

8 photodiodes per lens, light intensity at each photodiode

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

FACEVR: REAL-TIME FACIAL REENACTMENT AND EYE GAZE CONTROL IN VIRTUAL REALITY

FaceVR, a new real-time facial reenactment approach

In order to capture a face, a commodity RGB-D sensor with a frontal view; the eye region is tracked using a new data-

driven approach based on data from IR camera located in the HMD

## AR markers in front of HMD to track the rigid pose of head

## Allowing artificial modifications of face and eye

Applications – FACEVR

Images from Thies, J., Zollhöfer, M., Stamminger, M., Theobalt, C., & Nießner, M. (2016). Facevr: Real-time facial reenactment and eye gaze control in virtual reality. arXiv preprint arXiv:1610.03151.

RGB-D camera, IR camera for eye gaze of one eye, 3D reconstructing face ‘Removing’ HMD

## AR markers

https://lh3.googleusercontent.com/notebooklm/AG60hOoeX3UwOGH2KTgTWzephT_DEZCifAk56zI4z4UOzND0uJFh_81X6p4tg_eoPJR-IITO_M7BldUnsG880HliBkyB96VxPl-sEahBxipf3FWLIhneNEn_02n-5b0lVoRyX1k5vFafgA=w116-h28-v0

3d5bf7ce-c45d-471b-8113-71fba60a2837

https://lh3.googleusercontent.com/notebooklm/AG60hOo_yuTe8db4BS9heNTjbCowESOjmee3qTL2gghCjAIc9bGqGO-dWb9JypjWT6LWLLhkvxxcI6ZTpspA7ZwXVn2jAGVJpOi3wR5xZK6XnHufY0XNhMwIj0RY5U2xUaZK2m_iCgsI=w325-h91-v0

da5f3d3b-5839-42b3-817c-0d26976c1b3e

https://lh3.googleusercontent.com/notebooklm/AG60hOor3dlI4Hy7ob3HC7u2NBy1efhi0uZc0LJZr9Kw6ZU3JDUMduI9XFrV55fGrYIX9e2-NcE-MA2BbgPahaXbu87k1iZgai2gBUv-eAGgVcFTK3LGc8MKKq805Bjh_FHmNjPGUtV5=w116-h28-v0

0c85283b-57a5-4c47-b721-80f6a51e7de7

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## FRONT CAMERA EYE TRACKING FOR MOBILE VR

An innovative mobile VR eye tracking methodology, utilizing only the captured images of the front-facing (selfie) camera The

The system enhances the low-quality camera-captured images that suffer from low contrast and poor lighting by applying a

pipeline of customized low level image enhancements to suppress obtrusive reflections due to the headset lenses.

A formal  study confirms that the presented eye tracking methodology performs comparably to eye trackers in commercial VR

headsets when the eyes move in the central part of the headset’s field of view

Applications – FACEVR

Image from Drakopoulos, P., Koulieris, G. A., & Mania, K. (2021). Eye Tracking Interaction on Unmodified Mobile VR Headsets Using the Selfie Camera. ACM Transactions on Applied Perception (TAP), 18(3), 1-20. IEEE.

Without infrared emitters, only from images of the selfie camera

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## RESEARCH CHALLENGES

## Gaze Aware Rendering and Displays

Eye tracking and Augmented reality glasses introducing additional constraints in terms of power

consumption and physical size of the tracking hardware. More research needed on mobile, small-scale,

low-power, robust and accurate eye-tracking technology for AR under arbitrary lighting conditions and

longer periods of time.

Games and VR experiences will greatly benefit from gaze information to enable enhanced user

interfaces, more social interaction through eye contact and reduced computation efforts in rendering.

Eye tracking for commodity devices requires security and privacy of the to prevent identity theft.

Technologically, user customisation and automated device calibration, user profiling and user-friendly

identification will be enabled when biometric eye data can be acquired on the fly.

Dealing with large quantities of eye trakcing data: As eyetracking technologies become cheaper and

more easily available (for example, webcam based eyetracking) it will become possible to obtain eye

tracking information even by crowdsourcing viewers.

Koulieris, G. A., Akşit, K., Stengel, M., Mantiuk, R. K., Mania, K., & Richardt, C. (2019, May). Near‐eye display and tracking technologies for virtual and augmented reality. In Computer Graphics Forum (Vol. 38, No. 2, pp. 493-519).

https://lh3.googleusercontent.com/notebooklm/AG60hOo7Tsh_bQm6FDIUIkryH7Yy6IDtYdpv-IKG4t-It0Yb0eyVPYsf4cr4gZ6lSHeFefJSMglgp0vDgklXcIFel5ZZVktACfJ-AMG9ZXoxPoV0Fd6D6HW3DamN4hoyz81dkHbopmn6Ew=w116-h28-v0

607ebd61-b79a-40d0-b253-3cd85e8a5dbb

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES© 2021 SIGGRAPH. ALL RIGHTS RESERVED. 47

PROF. KATERINA MANIA

TECHNICAL UNIVERSITY OF CRETE, GREECE

## Research team SURREAL

http://surreal.tuc.gr

https://lh3.googleusercontent.com/notebooklm/AG60hOqqSQK18uf7E16Y75fAIZmzoEwQCiAtEz8D5e-2LNAi6ZQZ0atcpp4AJ97BEj1VOeLr4hVDHTrfNXl91Ei1BcGdwG_ozpdjWdhdz6KJJpQQP4Gl0urZwX8UXdo5KNZ-qOtoJeGkkw=w228-h55-v0

864a43af-171b-4eb5-aac8-cac4afa9175a

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED RENDERING

GAZE-AWARE DISPLAYS AND INTERACTION

https://lh3.googleusercontent.com/notebooklm/AG60hOr_CJGRq6eVsj5m7EmYcC6OPxjCbw_rnDTYX1niLGVoyiJ4eUDRZ4GsD2Dfl2sdELObTJ8Q7EvAUJ-6OddwvItsbg791Gh9tMv0dfMn9Q7UbOmCgPo5hIvBt5U9K3HN9NUFrPk58A=w228-h55-v0

5caba06f-e4c1-4334-841a-9759f1645cbd

THE PREMIER CONFERENCE & EXHIBITION IN

## PROPOSED SOLUTIONRESTRICTIONS

## Display Hardware

− Increase pixel density ( FHD→ UHD)

− Higher refresh rate

− Higher field of view

Various algorithms performance and Quality in software rendering for virtual reality

− Rasterization (High-performance, low Quality)

− Ray tracing (low-performance, High Quality)

− Path tracing (lowest-performance, Ultra Quality)

− Instant Radiosity (lowest-performance, High Quality)

## High frame rate is required for virtual reality

## Human Visual System limited rendering

− Reduce acuity in the periphery as

eccentricity increase

Foveated rendering, adjust rendering using

visual Perception as the optimizing function

Non-uniform rendering

https://lh3.googleusercontent.com/notebooklm/AG60hOqxK51qSw5FDhn3pZUxJk9F5lHTx0jN6a9AxLwkSPEzm3gJCZOTUTlXDXZGkeVsRcpERGYmnw6CwuitmXT72axY1AgdfC9LclOBvdBiXQlgtZf_IB964t7B7VQ6axANhsUyWQd1ew=w228-h55-v0

c4d71562-5eb7-409f-be36-843468c189da

https://lh3.googleusercontent.com/notebooklm/AG60hOqPfIz7hcXNVfzlm9bGmyWCf-zlY8eOhCPPPKp6kq63u9ysq7UhtqL6H-erDw6p_3S8pdhh3hX8T9eCEW3tPsYk65wBlqiKXEggsDkvErmgp9EMywRXu1cGJuOT0lZ1OUvZQsKGoQ=w196-h178-v0

c5901117-c65c-4bf8-8544-6a65f948e838

https://lh3.googleusercontent.com/notebooklm/AG60hOoSnlfqR1DymtPU92cfIgpH81h4tOgasGRHrZDGdfgxkp3RWaCLQRJspdhbIF753I-0WXcN0d9h9mfZKN2dzu5Q8QXW0uIUh7RZS3-5epgMlkNWASied1YFBPAuNyQXy8tlYinqFg=w258-h170-v0

8efa6b9d-c919-4036-a468-f17792ab3abd

https://lh3.googleusercontent.com/notebooklm/AG60hOqpqBaWF5fKLqIOsRCXnRiQGbwJ1AqAzfAJMu5iaB2F2aJSNUQCOqPJLY7t0lhfvIGTHKJhxg4jmkx6hp7zdXMRuAzRky1pjZQfHxOjTCe7fskx0XRTIqqkDWPrpac7vq-Rmnp4=w260-h147-v0

ffdbbcf1-c533-44a0-b822-bf4988bbc73b

THE PREMIER CONFERENCE & EXHIBITION IN

## RASTERIZATION

Find the closest triangle seen by pixel, discard all other geometry

Shading without having information from the 3D environment.

− Faster technique

− Many techniques to increase performance and quality

Lighting probes to make pseudo-realistic light, LOD etc.

− Not photo-realistic compare to other rendering algorithms

− Reflections, refractions, and shadows effects are not accurate.

− Global illumination is costly to be implemented.

− Not fast enough for VR in some case

Top and bottom left image: https://docs.unrealengine.com/4.26/en-

https://lh3.googleusercontent.com/notebooklm/AG60hOpauI_tKwMVxREmDjcT-EwMsALQZ4Bc_e9iurL2vv0qBeV4bHVxJ9Oi3DpagdX5tkYNkUX1uwYuzjwScPtpBDzVg73bVUabCU8xjhABRW4dIgZWVWv3GyNTUzbT9zHE0U8KknWRAA=w228-h55-v0

dadbdaa0-e3d7-4560-babc-a6cc75c389dc

https://lh3.googleusercontent.com/notebooklm/AG60hOp86zlKI3KUkVUs6b4Wq1Pkoan-9GPi_bUyJgYiyzCLLISqW3ZBXhHy74ea_AUZGdK6DsdubODSVD7xevSKSUT9X0KDmdTqCbv5N3iM8oyXshIg6C_EE_fP3LDJ3khCgIEOkR86XQ=w218-h311-v0

06cce334-1d59-4c53-80f6-fcdaff35d6ef

https://lh3.googleusercontent.com/notebooklm/AG60hOph4tWRPbl9XsXVdqFlPvl7ntYQwGIli_5v-JJMFWK8x3EzbjmW8Ud0uZaWkwa2cpu7zM8HgCs1bssPYCPQ5dIVYUwRQNUUgSF9ZNtd2oIHsHtt_Bd96FtiEEyNV1h9ei_itRxg-A=w334-h186-v0

47e46ce4-126d-418d-9f34-2d5d3b90fccf

https://lh3.googleusercontent.com/notebooklm/AG60hOqhEuG7mQAuhHbKmsh-A2fpRapHraqi0fd3ll9PUz15tpgE7DljYgkzKmaNJQOM1v2SybsmLfjO7HODeP3KkDY5Yl8owlmEmRQ9UtaJ3s2XIcDQsUnTR6n1AEklcVBlwv1nSwS_fw=w329-h188-v0

2ed452f8-5210-4db0-a114-bf9dde62374d

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED RASTERIZATION

## Three Layers rendering

## Each layer has different resolution and LOD

## Layers are blended and smothened

## Updating half the temporal rate of the inner layer

Post-processing to cover artifacts

− Progressive Blur results to tunnel vision

− Enhance contrast to eliminate tunnel vision

− Higher performance due to reduction in LOD

− Artifacts due to extreme reductions in LOD

Left image: Guenter, Brian, et al. "Foveated 3D graphics." ACM Transactions on Graphics (TOG) 31.6 (2012): 1-10.

https://lh3.googleusercontent.com/notebooklm/AG60hOovAsjbm2bWq_9-wSYU21rNsXcuLxmgbmfSARfkCyShH12uLHJkHT_yaRWxG9pTo-UtjeK1C0HglC5TTL6QnyWWutla2Rkz_cpTGlsW4SpfNYp1IOX_DrhkkikRYrAbYnS77gVJLQ=w228-h55-v0

22283e04-1229-407f-8d46-5660bd980189

https://lh3.googleusercontent.com/notebooklm/AG60hOocZhdF99uNaID2jM9p_5l5XR_bqoWkhdfFYU9yEPyTbvNMcEVDP4U8v8qW8v9oty38Eo8XV8x1a0YhFcxq69Lk7Mji6n3AV0V6-NLt0s3aR0FCLGBi9dWRfTPJXClKUtYoUTdGEA=w719-h256-v0

ce26d9a3-c6c8-4ab4-bbb0-2d762956cc7b

https://lh3.googleusercontent.com/notebooklm/AG60hOrgtp_CW2n1umBFjEvlo-OSrmvBDgpoKpmdfloNKlIq4gItPWimuwyLcwyLQuMjDdu3vxCrVRTcsWLqWUCKJVCmlaCWZdX96yDE0DQb_983tSPnOKWIfDjL_pPktmGQicGgPS_fIg=w887-h127-v0

a57c2cfa-c50c-4ab4-8969-e64ff43ecf23

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED RASTERIZATION

## Use Deferred Rendering to produce GBuffer

## Transform the GBuffer from Cartesian to Polar

coordinates to a reduced resolution log-polar

buffer (LP-Buffer)

Apply shading and anti-aliasing using the LP-

## Inverse transform from polar to cartesian

coordinates and map the result to full resolution

Images: Meng, Xiaoxu, et al. "Kernel foveated rendering." Proceedings of the ACM on

https://lh3.googleusercontent.com/notebooklm/AG60hOqejG04Ri8HeDebXcA8W8LGsXeFuKysMRvZIpRwDCH46wxKQ-inezMlq7zhT1CisRKdD4gRU4LPD__oHdgheVeLLc0NXef28_eat7sm-tG78h5Id8BBYkpW3FmUc5U2MHYOdFntTQ=w228-h55-v0

027d8645-a886-4422-82e4-477db87f55ae

https://lh3.googleusercontent.com/notebooklm/AG60hOrptrRNQcpD68t62fdwu-7x5YikGuY8y0s_2q7JRUVZpc1Cgkqp_GVf5v0ubWxSCvT27xhQGYfxCbZ_RRvKkjG_RufOB80i0yuVvZS6YlCQscopqaCrUZmP12xGypmluw1gFL5T=w215-h166-v0

38c9984e-b441-4c62-a2de-3b7634bb4d39

https://lh3.googleusercontent.com/notebooklm/AG60hOoJwXdWM3dnNLCnkDi3BkBxlN9eebiCdR20p7wxf-feswqw4X8Wi0IE7t9h1UjZJttJ81waBIeeu2gSVNVjo-AAMoGXEX8TLoSMou-tsf1kAeSTDiww19XLGUPcovbRfwqqAcTq7Q=w228-h173-v0

45f46b9b-2c7a-4d68-a01b-f4e99cd3ef42

https://lh3.googleusercontent.com/notebooklm/AG60hOqYFaju4EpCORwnpkcvdcRAOetrLCU7uTesH9GZ_WOfTfQsF-0mekcAbnBqYsXRuq4cx7iWRLXIMHRadFkr8DhFoHOmIBrMEBimA5GacjxHeYw7bP8l6WFK1CArjL4R_eykwf7CSg=w230-h147-v0

afb6145e-dd47-4895-a025-f95825a2f6a9

https://lh3.googleusercontent.com/notebooklm/AG60hOqysO2kw0STPD2EC2fuml_RuNXi2A1QUXMCdj2iw0nry-rQBtmTzdEYOqOGNSBSuzCOaLm73Xc_908bed_uLq-RUqwqlnlwC4zJriYGMmwEkOg_Mmh1YoYh_d8kBryKRaQNKZzecg=w222-h147-v0

173a3436-bdcb-4c9e-b985-83e6dcfd6a96

THE PREMIER CONFERENCE & EXHIBITION IN

## RAY TRACING

## Generate primary rays for each pixel and trace them

## Send ray through pixel and find the closest triangle to the pixel

## Rays change directions based on material properties

Produce secondary rays based on material surface properties and the light sources

− Accurate reflections & refractions

− Easy to implement global illumination

− Slow compare to Rasterization

Top Lef image: Parker, Steven G., et al. "Optix: a general purpose ray tracing engine." Acm transactions on graphics (tog) 29.4 (2010):

https://lh3.googleusercontent.com/notebooklm/AG60hOquOH0krib4HC-Kq6L__xlXo6II6KhYkoMAv5RcCV8u9BbNZpLlNoBOnfEeIuMYGffyZQLdMWeXCp9c8003ZqY4a57lYSHzUcn2GKQOdR21HrIZ3vu7dgMkmDJNW0PYe4rJhva6Xg=w228-h55-v0

183ba006-a816-4a7d-8bf0-6634ff6a69a6

https://lh3.googleusercontent.com/notebooklm/AG60hOo0pMN85k8luUNXbi1B2ZWyA3ibZ88MzmoJ0AvRX7sLBL0gy1IVLEvVsNRtkIVDcdixlzL9j9s863yQ6xEGw7RgUFyCwJ8i_llg7GT6YF7Ew2uPxJt9JE8Ki7Xcy1jM7Z6N9RHPHA=w233-h154-v0

4d396d63-9cb7-41aa-b487-3d5ac0b9e3b3

https://lh3.googleusercontent.com/notebooklm/AG60hOouMhY3rH-h5T5lhNrRrc5QIoIZKDxavZ89h3Z84GbfDyup1t_jUB9OJyc1y8vcQstMvlN6O-C8ZsstwyxGvpaxGjzwBENYkOAOenDiYTU0TfVChjkapTBFOHRp98IGm7FnYrDvyw=w234-h164-v0

12fcf7f1-54dd-47d7-99a2-8b999605a63b

https://lh3.googleusercontent.com/notebooklm/AG60hOo0pSJajPmUVgzjvu0mFAzf7G-p1ETfP1Rf0Wcqw4Jq7qxpQ3g3_LXp8tjVP6VUA3Fa9Pbrgp1Apgnn1XuP28QqbcibtEB-DytFAj7bPjiZMM8j5UjopCqIxyJ4wMaOScoLp8UQ=w233-h168-v0

17e42f26-7c0f-4dd0-9329-be2e5860d5a8

https://lh3.googleusercontent.com/notebooklm/AG60hOpNhs8TH4aR-dXH1mve3ZQ7UpU1cfCRZ9lqwSxZLXnGdD_g6Pe6dJDEfuaLPutdoQYKNLo7SWut88DMGHsjFEe1Btbo7y36pjUdZaYhCdazqJLU6Yh_yifY2VHHVPp5vQX0r_95hQ=w327-h120-v0

e63eb79d-59ef-417b-b612-79510b4f2a3f

THE PREMIER CONFERENCE & EXHIBITION IN

FOVEATED REAL‐TIME RAY TRACING FOR HEAD‐MOUNTED DISPLAYS

## Some primary rays are not generated based on a linear probability model

Use reprojected frames and a support image has a lower resolution to fill the empty pixels

Post-processing to improve quality

− Use of blur to cover artifacts

− Use of Depth of field to improve quality

− Increase performance due to the reduction of generated rays

− Artifacts appear in small foveal regions

− Flickering in the periphery

Images: Weier, Martin, et al. "Foveated real‐time ray tracing for head‐mounted

https://lh3.googleusercontent.com/notebooklm/AG60hOohWuZmuXmnOnnryYQ0beLNBUGpHScpVYceNTOG6DtWMtxQXmWQGr5T5o0h0OXcX4YjLP_cWJARI8izeUkFXG9RwDK81S9Ud662mtKI2TOLkNLExYjh9xihlUYnqEsv4txQPrVkug=w228-h55-v0

673ddb9e-d1ec-42fa-b1e0-5f75474c389d

https://lh3.googleusercontent.com/notebooklm/AG60hOrQEnaTB2Z_Bctw7-4WfWCCI1bF4HFEfUiUzRiUzWT2z1cBxmEHl4GU44_HLlkKz-ImnfLdIL1AZnpEendivGr4LGi7ZTGPsdlVAbu8TyFBmTK_lCFjsObSciaiVN7V6zIou3PZ=w551-h236-v0

be899723-e3a2-477d-8ac2-9837c0ee3139

https://lh3.googleusercontent.com/notebooklm/AG60hOq46uuqFicE3BL4xyWU8bQ03k4E1Fu27yNQGaJoyoWLUt8GrW1pTmEMSGOiwMSmRFLK6jEcAiMLBXC1cEgT1UmimlomeVBwO1XB-rBJVXgCFtyVO0s2hZmy2cubGXCoH_7A5GBOSw=w256-h145-v0

62bdd003-ff21-49b5-b9d8-96edfeb16166

https://lh3.googleusercontent.com/notebooklm/AG60hOoiQEOsueXcaCLwgoit3R0yyMxPcR6WHzxRGfr1XwQfLWDF_sbOfTgd6FyLSK4wfKo8F7o1sRMA1aRCNL0_4PQG5nLA8mZBjhaCsqlrk7qo6G4Jbxt7l7xnOPAGvrds5268KySxqQ=w257-h145-v0

b8e842d6-b056-4b77-8ee7-156da524f2f5

THE PREMIER CONFERENCE & EXHIBITION IN

PATH-TRACING

Path-tracing uses stochastic sampling during the intersection of a ray with a triangle based on material surface properties

## Use of Denoisers to reduce the samples per pixel to one sample

− Not enough for virtual reality rendering

− Global illumination, simple implementation

− Higher Accuracy to reflections, refractions, and soft shadows compare to ray tracing

− Requires many samples to produce images without noise (more rays per pixel)

− Slow to converge

− Extremely slow compared to rasterization and ray-tracing

Top image: Schied, Christoph, et al. "Spatiotemporal variance-guided filtering: real-time reconstruction for path-traced global illumination." Proceedings of High Performance Graphics. 2017. 1-12.

https://lh3.googleusercontent.com/notebooklm/AG60hOplVKu9010Wn7-afd6et_QMfcU4Y0-S8G2szhEbME7mxcQT43RjAFtGpokg6TGHh0lRgXzibNG9SDgp5T99MfFwrv8yf7BrObG7JeY_3z_5dfDpJQqpJ57TPQiHFuTXLSbOGhvjBg=w228-h55-v0

cb630a4d-eca0-4a4a-929a-5202ce2b122c

https://lh3.googleusercontent.com/notebooklm/AG60hOpJrRqkNE7zaD6pz6EhYJPgReIGRS7dnG0Bn5PZbI4KNkOgWarmwOlDdAVfeZOtmDKvFn-i3frRe1p8AwqJRrlO0v2s5purdKWeOOsPdYbjw3S-bP2tNJ01gUwjMMhE9BO3K93Xuw=w552-h300-v0

8ed55bae-aaa3-459a-8ff3-3c323ec663fe

https://lh3.googleusercontent.com/notebooklm/AG60hOqVvhFDwAyQHW7lcA8WBe7Sg2JtE15nK1B8GKGGJG29Cfi1FXftc2zxh5frgaZVA__R5JY2VdOqTeMDThQie0DbZBJQnDgYKn5KroJmzHK6bPw-xrV3Rtlbgvpb58VlOffQeNJh=w710-h163-v0

2bc3f316-a196-4031-81e6-752a19fe0d23

https://lh3.googleusercontent.com/notebooklm/AG60hOqOkEx40vOMgBRMKpWthz4u68QCP3AsF9aMQAO1m6z6NlVs1JqkRTEy0IW6Y3-WBi1ETjzV4VayYEx9oFRJ4mPubeUhEuiMq_me0S15MC8IV-8LgDMY6Yg9nDjU5etpUczFv4_pmA=w193-h159-v0

ec88bd09-04d0-4057-9013-94d89d2b6b22

THE PREMIER CONFERENCE & EXHIBITION IN

FOVEATED RENDERING IN PATH-TRACING

Use of a visual-polar model which matches the human visual acuity

Denoising and path tracing render in polar coordinates at a lower resolution and are then mapped to the target  resolution in screen space

− Primary rays stay more coherent (better utilization of hardware)

− Improve denoising quality in the fovea

− 2.5x speedup for path-tracing

− Not a dedicated temporal anti-aliasing method

## Could reduce the artifacts in the periphery

Images: Koskela, Matias, et al. "Foveated real-time path tracing in visual-polar

https://lh3.googleusercontent.com/notebooklm/AG60hOrj9TSG4aC66PHlFJ9TpUZRS31E1pX9LfaKf5l6oKJGovbbZKzGjIoprldSRvXD51V72NUquwh4-xO8KGA6Kfnrzr7b8wYv7P2PCvZr-PlNApFkOT7Diw_u0q3mLvjQVhYHqAtfqQ=w228-h55-v0

a294be19-3841-4d01-9fea-303dcd0350dd

https://lh3.googleusercontent.com/notebooklm/AG60hOrwslJYocSHbrcvEyd7ZQDT89xkA_pln8lumxBFpuul53mJGehZl9BHTUjKbNKNwwBiVibMQGL-vJs1pp5Tc3HqIq0s-rGEj4__nzarUuUhnMM-hckmPjVky0o1y1K0xmn9bKc=w168-h162-v0

e12d01c7-3121-4981-b0cb-1fc7cd6a1583

https://lh3.googleusercontent.com/notebooklm/AG60hOp8FfczRDkoSrPhweC-yFxveqVm1Q7RLu3jXUOCLS8UAYBF6mWqt_A-L6YJ4B7QzX0Wn7zbRkjtyRPKmZI6uUfDnt9BNhjaMVSHZbdO1XS-Wvjw4oui3hDP3EhKPggyHKfN8TH8kg=w168-h157-v0

d37928ff-2dff-44e5-86a9-7a90aead0c30

https://lh3.googleusercontent.com/notebooklm/AG60hOorqU_Fh_2kLxx9BsQRw0HZbRNneJG5c2bMY9H6Gj0olCzVPXZsiLR7a_cGSk36nj6vAuz87ulZehbcr_RZx8t7FCI-fdfaSrHk9Gc-zXU1IN1jH3Bhx6TW0EchfBfWY28vFEc8jw=w165-h157-v0

ae5a1515-64b8-4c0f-bb88-977a59fc6a60

https://lh3.googleusercontent.com/notebooklm/AG60hOrIJqBFOvr8q_GMb_S1Iqqu_Yc3HgHSPXrOL1mJeiBjOoDR0Pt8mO9a6oqVcyl90odOfpxgzAQ-eNi2kGX3itrLFGc03oRx5EsWeBy-FJGMZyR6e_rrpgF-Pjdcjl5XVdFKcenQAQ=w168-h162-v0

e78d5c4b-3a86-480a-b5f5-072710297a60

THE PREMIER CONFERENCE & EXHIBITION IN

INSTANT RADIOSITY (IR)

Light is traced from the light source to the 3D

environment

## The intersection points during the trace are

considered Virtual Point Lights (VPLs)

## The scene is rendered several times for each

light source.

− No noise compare to path-tracing

− Create Artifacts

− Calculates only diffused surfaces

Images: http://www.cs.cornell.edu/courses/cs6630/2012sp/slides/Boyadzhiev-Matzen-

https://lh3.googleusercontent.com/notebooklm/AG60hOpSJAIN3h334IdE_-1vnq5eQxdeLjE5JsQy899NX4L5h-7dsS6R694KVmWUK1cbL2o6VsNm-UGFS32qeNP_YdhGkKtN4ZcQIzFV1UKmB40j1-6vIuHBo-TDKaAku-U_xw4PWtrQwA=w228-h55-v0

94a06073-12fb-4d99-adfe-4c948bd3d404

https://lh3.googleusercontent.com/notebooklm/AG60hOp9bbor-XfOoY-p_b2upKVhzEYZM90ehU4YVwNoCxXo5SVCLaEx5s6LMlcvOvANPSQQRzARGwBaS1n_PRa1b7tYQ0GNG0VHwEQdYjwdvJCo55zkvJR6NS87QaYV7jpJysaaHVbnVQ=w215-h205-v0

1fd91d19-1571-4ff0-8a44-88381a136dcd

https://lh3.googleusercontent.com/notebooklm/AG60hOpTfow3wOXB_syNjqf1vf7rPn9anBzip1MKvHy6y24mBzwrwCQ0VyIK2Y9ohZBNUNsSoKdyd7Cd-CqUDjH_3uDtDyZ3z5MpQ_fA1Y3sTB7sq6Uw10YmmieIzRJATY-NBlaV85Zw=w528-h174-v0

5cf66e35-7335-43bd-b787-968be687b8bb

https://lh3.googleusercontent.com/notebooklm/AG60hOrICz2FNL8lyhMYY2fdEEqvZsth2wdzshNXuxV4KMfDtnlM9XboIRME1vfnLCSkNK9mnC3GPDimGPOYdxDKTT9_I55TgQBQp3Ev3nf4hlNm1-2shLulus9FqeTlEHNUaiYEde486w=w199-h195-v0

87923c11-9d65-43b1-868c-ecff6676b687

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED IN INSTANT RADIOSITY

## Create a voxelization of the Scene

Voxel foveated weight estimation.

− Each visible voxel in the scene voxelization from the current viewpoint is projected on the image plane to obtain the foveated weight

## Trace rays from viewpoint based on image plane instead from light sources

− One bounce for each sample point to generate Virtual Point Light (VPL) candidate

− Uniform sampling for VPL generation in the foveal region

− Define foveated importance for each VPL using the define weights

Propose a VPL reuse scheme, updates only a small fraction of VPLs

− ensures temporal coherence and improves time efficiency

− dynamic scenes, high quality in the foveal, high frame rates

− accurate global illumination effects in the foveal region

− Doesn’t work well with rapidly moving objects

− Less accurate global illumination in the peripheral region

− Flickering due to rapidly moving objects

Top left and bottom Images: Wang, Lili, et al. "Foveated Instant Radiosity." 2020 IEEE

https://lh3.googleusercontent.com/notebooklm/AG60hOqBsD2cSmzC8zKoojmht4ZRU1BdUnzSlFugelk9VWbolTFxQprlM00vz4CcXB1FLFPcOdglbLq7_ajac5JB4JbwkFp13PwS7H6Mvvq2oEic3N-Sf5mz6G3s6lfnQ42grp9MNAZBew=w228-h55-v0

677abb57-b3d1-453c-a630-55854c6ec82f

https://lh3.googleusercontent.com/notebooklm/AG60hOpdMOXWrEepmyZlpvzymViz7a2BkFF07Y8ivUVzj5Spa-P4J0y1ly4RWxQinQlqMKdwImhLHARKdR9YprPcEMs5HQqJHjcF8dXGxit8-j83JfSkMzis56bF9RvrUXTNqgUXfqIs=w345-h178-v0

c129753d-b21a-4646-8c16-c5c26ec385de

https://lh3.googleusercontent.com/notebooklm/AG60hOrp9MvQ0EgjNZPUZQYteEqxEZOeT_KbEVKLzG-9iOWwycXHbWQSbqEfeApvOVR2_6eUHOjZK_GIqS2m0ndgovCOol-7zJMdVMgYz9Kwd19ok1TWr2njrn0iFMGoY-xOOmAHbecTrQ=w292-h178-v0

cff55ba3-b48d-4767-8208-b904d7c0e995

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED AR

## Develop a prototype AR display by adopting foveated at hardware

## Two displays

− High-resolution, small FOV, foveal display: micro OLED (MOLED)

− Low-resolution, large FOV, peripheral display: Maxwellian-view

The light path of fovea, eye tracking and real scene are combined in a half mirror (HM) and image combiner (IC)

The peripheral display is moved based on user gaze direction along the horizontal axis to move the high resolution display to match the user’s eye movements

## The MOLED can move vertically to dynamically change the virtual depth

Images: Kim, Jonghyun, et al. "Foveated AR: dynamically-foveated augmented

https://lh3.googleusercontent.com/notebooklm/AG60hOrmPo8DQvtYyTs8f-Kv__l4PyoTr-zXy76AiheKeBn9Kit1Sfn7IhfYq_t2DXtbtnOoytg7Eb4QrEfyhISh97tT--J617uUphnE7YaMpk7NGBLaETWKnpu_dLJoHdKP8BkaAKR35A=w228-h55-v0

81cbe96f-5cb0-4693-97a6-701322ec105a

https://lh3.googleusercontent.com/notebooklm/AG60hOrA1EPRm0eEXCeVSePeiyLxO1Xsq6fY63Voncnr0E0RMRocQdiPJu3eurpinILagPbDbm8lm06DPai4-sfDxu9SNQ2tbIEl2364ADyCJk2qK-N0n5hRo_j7bNsILkhbwA6bigFmkQ=w340-h192-v0

d54cc3ec-1bab-460f-9afa-960b95621064

https://lh3.googleusercontent.com/notebooklm/AG60hOrQD_Vz3CFjaDql8gT1-LXZrgeU1Rh3nbb6-2ZKRGrdaj9S0A3eE4749UOH-pptaK4cf8G81EsIKHF-Pnr1R2_91bSiA4C-fegLn-9pyuCEmPth192pslTGBcctXKWASFXUeWavTQ=w354-h183-v0

f97cd072-2e25-4d86-8e4d-b1304446855a

THE PREMIER CONFERENCE & EXHIBITION IN

## FOVEATED AR

## Color intensity matching by applying gamma correction between the two displays

## Linear Gaussian blur used to blend the two images

## Depth adjustment for varifocal visuals

− simultaneous wide FOV (100◦ diagonal), compact form factor, high foveal resolution (60 cpd), variable focus display and rendering, and large eyebox (12 mm × 8 mm)

− use of a holographic element with dynamic position driven by gaze tracking

− Mechanical complexity, not for commercial use yet

− The projector might collide with the wearer’s eyelashes due to small eye relief

Images: Kim, Jonghyun, et al. "Foveated AR: dynamically-foveated augmented

https://lh3.googleusercontent.com/notebooklm/AG60hOrEMxzgj_yIVIJ46Jfrm_hkYtPjyJX-EuMtng-faS9IkaczEtbdBPFqOEbpMVrb1Hwo3AS0USqoCdzZhLQb-kaXlFXpm26SYRROK6tHh_DLccE7qLSQcfBQKyNn242cqZjjPnemcw=w228-h55-v0

a0c3e51d-9fdf-492f-b97d-e319dcb46155

THE PREMIER CONFERENCE & EXHIBITION IN

CONCLUSIONS & CHALLENGES

## Conclusions

− Heavily computational algorithms become affordable with foveated rendering without any perceived drop in quality

− Foveated Rendering idea can be use to optimize hardware to increase the specifications of current near-eye displays

− Increasing rendering performance and hardware capabilities fundamental  for widespread adoption of near-eyes displays

## Researches Challenges

− Aliasing in the periphery

− Flickering in dynamic scenes

− Eye tracking accuracy

− Hardware restrictions (AR mostly)

https://lh3.googleusercontent.com/notebooklm/AG60hOpEQz68UretAeuucSOcpWN5w7yWNPYc8AeNnFoGH_0BIaTEww5mElXTRRrYa3tt0KS9fj4gGF70KPBWyxewhQRLoGkCjB388Kd6N3r8_Gj3agCJCQW5l95ICmL7sVuhsyu6GBYF6A=w228-h55-v0

a7b62c8d-ba14-4474-9129-be0a8778e92a

https://lh3.googleusercontent.com/notebooklm/AG60hOpwnd_fEym_Ccu3FHxH7M3E4xVU1nwWosvhkMjfCWCHQWvhhv1qdmwN19aisBoxviMj66OgyZHG4u8JCJxLwBZlZk3Wh5CwooFNNwhaiT45fwgbwf0CRZTq07S-04E4MiUxythi0g=w236-h47-v0

21b15144-fbeb-4e71-a06b-f358799b423b

THE PREMIER CONFERENCE & EXHIBITION IN

## POLYCHRONAKIS ANDREAS

PH.D. CANDIDATE

## Technical University of Crete

I am a researcher and Ph.D. candidate at the School of Electrical and Computer Engineering, Technical University of Crete, Greece. My area of expertise is focused on foveated rendering pipelines for computer Graphics.

This research forms part of the project 3D4DEPLHI which is co‐financed by the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call 'Specific Actions, Open Innovation for Culture' (project code: T6YBΠ-00190)

https://lh3.googleusercontent.com/notebooklm/AG60hOq19E5fJ7LYsTezPYaSEH5nKxwkSN0iWLrqeZ7e0xlcfcY7a5eqB5UW30R5QwjJvPGG9aR58XxQzHY14nkJ0tADQpwNz1kT3FAmkrr1BcYNZoBytr3Xc7xZFfhJMXGZWwFad3OSWg=w114-h27-v0

e90deb7e-efa0-4d83-8a99-aa1543cb05c8

https://lh3.googleusercontent.com/notebooklm/AG60hOqJh4OLG3pt8BeSmiq7emkmnIP2gfL0vrNSrSewyxo54Fn73LCxFk0QnMZlxErluU4PJN9v8PKglxVYNE5xVyq-7OOHPAHEiiECdqAlYj4MnLyD7lAqT9IEFuf1B0ZsCRqN9oWcIg=w114-h27-v0

7114456f-71ee-4051-9aff-549fba94758c

https://lh3.googleusercontent.com/notebooklm/AG60hOq2W7vurBdbSfnHhJHpdz-t_go48wusga3d0j9FrMvPtkNu58iypj3dRXgFN5WOJIxVepgIpv1Iwj0LaNlm1q9pMtkCKhj-Gfk1cWgw_XigqXTtTgnhta40z0nMJba2XNnEGSzt=w480-h270-v0

80ddb75a-8f08-46f0-ac97-ed6db51c817f

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## ANN MCNAMARA

## VIRTUAL ENVIRONMENTS AND EYE TRACKING

© 2021 SIGGRAPH. ALL RIGHTS RESERVED.

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## EYE MOVEMENTS THE HUMAN EYE

© 2021 SIGGRAPH. ALL RIGHTS RESERVED. 2

https://lh3.googleusercontent.com/notebooklm/AG60hOqwDXO801uq6pAuoOUhat7nzvUVf9eCDpzb3iKnVRWXiWvfYFYHEJApAL8212OXrFG1X_-Kqv7Y3lkXWV_BRkCG2DGDgaqeaYzE__niJ3DaJWFRJbdtvz3DOARPXo6Bh3humNmV2g=w17-h7-v0

1ffbf888-1afa-4f42-8b11-3eda4a6cfbdc

https://lh3.googleusercontent.com/notebooklm/AG60hOqaUDXw2XTyWZi1D_5D3e77emsytqviUAiEZccNOQ9LO1j9peNFWq7G1BHJxg8jsRFfTKy2VxGphM_k0PxHGfe3RhdAo997S3LHuMFTfi7rkmdSR2knMwn9li2kDNz4wZk29AXeAw=w114-h27-v0

4fc1602d-7c49-4798-8d58-8e6894eba0c7

https://lh3.googleusercontent.com/notebooklm/AG60hOowlgNTsM6HMMSDuF-4oizy4VzqKJSVGo_nLT48h1xyGbSeAigKBLFNXxAjFy8F5KKL6aZ8-lHmzAW-93xD2mqvqkYKzWGUjvKOnB4v1p3m_sDmGAD5hAwy4xMDqoc1B2eLk_4K=w252-h250-v0

422f77a2-67a6-4224-a05e-f1114d936dd7

https://lh3.googleusercontent.com/notebooklm/AG60hOoCJOnkhg7nQG-k5fjHVWnJ-V7zA6Pont3RwV29yJFCTn7NiUoga7rxF_yQkWdQtIaK2FSuG-ooFGueMXNpw0HhaffcbM8j1YXuaoMHzkLXywqRDQrX409jNfe3SarhMuZo2bnWzg=w17-h7-v0

de30f5a2-f667-45e9-ae17-62bc3a64a4d0

https://lh3.googleusercontent.com/notebooklm/AG60hOoGy6lDDgyCmjt5RwujHvDSOhdYH705V7ArjBvwQFlCPVebRdmzXXoQMxCMxlLgyeL0d5lCyIGLBz8C8i2T432nDy6RFQSvaPbN9-og2cXVe0rm67-HLGt1eyN7C1u2WKW0K2IO=w114-h27-v0

1b3eeaba-5147-4859-8a31-b6de6b0cb81a

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## THE HUMAN EYE

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## THE HUMAN EYE

https://lh3.googleusercontent.com/notebooklm/AG60hOrwVpkEkT4UWtcGzqfa2AdIYicm2wVutxxImJmMBSQzEHphonAeJjH1petnNHm_rkf-KO8ohLSdnFwMEM_6f_jvmSsvlBz8pc3Ed6S_eD91jaHbMZsQHu_OedUFJy_ZHRN-M4FXBQ=w17-h7-v0

3f13ee8c-269f-4a95-8f6f-b5c6b8b6c42e

https://lh3.googleusercontent.com/notebooklm/AG60hOrwKc3sqnSrIme4smiy3Y_gSPga7ednsD7XrT5JyfSOQ8rcBXS-EUbken8NDqJ8F4bj7FPywUVIEs7ERpccR7Kjj9T8huZ93pXjqZKiHORybSRWdbzYYUGFz99Ln-YfFXe8jpRqAw=w114-h27-v0

42488f3b-8ac9-401f-baee-09149048749d

https://lh3.googleusercontent.com/notebooklm/AG60hOr4_DQrzdS-l_mUj9jGBxV4xHPcw4hiMIatfUYGYKlW0dgoOsgxohVW7OyVr5056iNTeYquSRNGoEe8s8iEeROeMkSUHaiRpgycLegeZC52mcDi_kSY-j1vHMYqMTHoT2yc59DXyQ=w17-h7-v0

5fbb9c34-4995-47dd-8113-46cc8235a0f0

https://lh3.googleusercontent.com/notebooklm/AG60hOoWFqzbd964D3FtMJpNgZiA_Vrdao1_AJ8OnlPexV-W_Q4SwGmHUA71aP-BRa3CWydXCQDNVl7RHGx0wSBmjH4uMUSPcB5J-4W9JRsK0xfpLJMu_9fOtczKIA9ZXFg8F3tBTBcKOg=w114-h27-v0

5cdd9bf9-98fb-4d31-8371-0f3b2b510b27

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## A BRIEF HISTORY OF EYE TRACKING

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## TYPES OF EYE MOVEMENTS

https://lh3.googleusercontent.com/notebooklm/AG60hOppuhqXAB7rK-NeebFfZ-imyAK0MVNM8k3-5Iu5TBfYhtz0TwcBXr87OlQBgUywFL-1LE_cRCimjrCpUHk6X6_pednSXQEUYdQKzObt_1S3A3cFS-a6v8FI3XZMMPpLWqlqJ38P=w17-h7-v0

e4f5e516-af24-42e8-b7bb-8c7303578851

https://lh3.googleusercontent.com/notebooklm/AG60hOok3T745xd4hUNVm3oGD-aTgocIbm5u77Dox49OZgJocMKXNxnzmktURLqpYUq2P-HCRpEU0cXouRsUGPXrLemtngeRnpZCcNEmyxPjiLPY4iWeUcE5shnQiU5Zw59kGEMxdnBHUQ=w114-h27-v0

a20a906a-03da-46df-a703-ca6a5bf55688

https://lh3.googleusercontent.com/notebooklm/AG60hOrKa8hRr-x58OTRuEkVVUSE_sJQPzfeoGvnlwZo5MTP51gJuU3sCwxwgVxHu0phuMjA4HJqglPOI2oZ9EHQZx5ZdbEqFwIJ5NZBrQWMhz9kq5hkzALkEma-0Ss2kUXXFGeFBuKv0Q=w288-h216-v0

23c02738-4bd4-4d74-b0a9-066612a9094d

https://lh3.googleusercontent.com/notebooklm/AG60hOoesC_LZzfeKzdXnBdRzLLtIZfHap1cd55q9_nJNrgFbb7PK9KHCv9CL5PyIQsukF9QzfzlGowSwBaddDAvZ5_0oy71KsfuCKEjQgZ6sE0cgX0N2vzokLKPUpVsRIYoqM4F4klF=w17-h7-v0

173ddc13-cd4d-4ed1-ad2e-74267a8a5936

https://lh3.googleusercontent.com/notebooklm/AG60hOp9KnqnLbLyXrvAqGQh9XPsBd2zLymY6-5FgB00YUP-Ze5T7wv30hFv--G2aXNUEkTrefxpPhCleseojK1b_fq8iKG2VDDKtpvesyft2TLLWvMh1qQSUpIX6D8Zciqj6AkLGhLbeQ=w114-h27-v0

192c303e-c364-4fa7-9833-aa21ba81b461

https://lh3.googleusercontent.com/notebooklm/AG60hOqXj6tbv8BL7_bg-Fvlr_CjFsTsLGa5mTtNek-GGHBHSxBxN-5G04Z0cFLi2zoG3ZpTq7MRdReSeEsRKLmX3eW4685mdXyGMI3lBwaOT_jbnWPzrnwSMXZrydToMEoX4noLPI5Jhg=w407-h102-v0

491ed7f1-fee1-4965-a7bb-28dd1cf98b54

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## Movements to

quickly reposition

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

https://lh3.googleusercontent.com/notebooklm/AG60hOqnObKsB8aoPJB9ynQvCq-jUefEcfUZR4844Hc_wLDYXtfTi6CuNcReTuAO71B9UgcaN9_6QSsjvbdwWxvVeFrqq_TCIrJkBBFmFHfpvWj-PCbbguIFy48AjCKOomoBpEFcc3tc-w=w17-h7-v0

ff479a2a-0f45-4e00-ac1c-6145db3aefaf

https://lh3.googleusercontent.com/notebooklm/AG60hOoa4yKQpCZF0Po199d09a_QpPu8lhqwlIoqulUWwY_uj9uzxTmmM37ZDe7DPTpRtc3CKvNWcnGKSnHfEraXi3BgssAVsD-DCZhBrytOzZ7lkRx2KoCRuuZg7bgSVfq5uo8taq3ZCw=w114-h27-v0

11cc32ed-dc6e-46dc-93ed-7e2cc95f123c

https://lh3.googleusercontent.com/notebooklm/AG60hOq592WWabDMjsIxGj50MVjyR-3h2ic1IdmIli-QsxdafdWRNteMBTgopHh3ZMpPJa7gO709m8kh3-nLFftfFpgWeMWllIyj6KlqJPSjuXy_E50kWZTV8vueElCJjnBzK7Wt1Y8-=w17-h7-v0

cb7f179f-0476-4859-b627-90b2d3ace809

https://lh3.googleusercontent.com/notebooklm/AG60hOoNHGOYKdu-Xxz0XtB1NVlYyz4kcomR8QybE9ofIONshnKXtTYqsNfBalJ6RR1vNQquscbEET6h_WfUvEkkqnzYuH6yUNBM2cflnFEW9tDNrGNCHHegAqNrJgcELFwmDuOYVjMuBg=w114-h27-v0

f3e197e1-458d-4bc0-962e-0a5745a8b62b

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SMOOTH PURSUIT

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## OTHER MEASURES

## Blink Frequency

## Pupil Dilation

https://lh3.googleusercontent.com/notebooklm/AG60hOoSdyK6axtPveFOQU4lPHfJDDFtc1zPqfEVZd9oOxrcuMAl4bbhUhkwLJPYujEoZ6R5ZByIo33n8NW2iQZ0kWqyGMahnnvgqy4jGLR2hOmkNro3SuoLFLo7zyeBR-0MyvHj696Qsw=w114-h27-v0

e32bb230-038f-42b0-b325-b4ac583af544

https://lh3.googleusercontent.com/notebooklm/AG60hOqesZ--aCUz25_uYYq0Rqo5iCJIFoZ7Tld4MvPzeRzsc_qFFhC7sCNrxKWTAP5qSnoyRlSvcTOP33D_govzG_MCbL-Iz6FSVHFj_Lk1e-M8UayBhllt83QxZFsvxiXjoWd_9Z3NFg=w480-h270-v0

953d8e75-6307-46fe-b6e4-8b8e1687a2ad

https://lh3.googleusercontent.com/notebooklm/AG60hOqiR7yPsVG_7r7lfOV79uXn285PsG1x_9P9vwxh3O3uraZUfHsWORr1QUmWpYEEyz4OmjpU-HVu6uiSIc_fbiCjwPFa1rtRCKIz1PNyzqx5dtsDj7waftcSQSVERXU1J6l3W_Kd=w17-h7-v0

5274aeb7-338c-4bc7-aa6c-7c3ef52bf44a

https://lh3.googleusercontent.com/notebooklm/AG60hOrehwI4nXeTlnuY1PO5o_V8yaAJ2nNlQACodiI5HkCfFv_UwohQQI5-B5oJS3zTy3khx7-GYFxO30gR4FKJ-83TE4lVY_qqKPLMzuLQPaKafmMhvovYWSCkC0Q6EJ_oZK-xdFxB=w114-h27-v0

2da2b4f2-8c21-4aef-aefc-0e4afd234ed0

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## VIRTUAL ENVIRONMENTS AND EYE TRACKING EYE TRACKING IN VR

© 2021 SIGGRAPH. ALL RIGHTS RESERVED. 11

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## IMMERSIVE VIRTUAL ENVIRONMENTS

https://lh3.googleusercontent.com/notebooklm/AG60hOp_ZVG0FZhp_VgXreQO7_tXPE-4kDm1BTLmeUm1danTcCvZcnXxizym8R7LRGN3Fc1rbvmqkyHXtZufesNAIPT06u83wYVKYFtU2aOn2fHjQ2u3GrM0GuOSdRM0TKCpw_9in01Y=w17-h7-v0

2dfbb88a-8b0a-47f8-8dfc-688374a1feec

https://lh3.googleusercontent.com/notebooklm/AG60hOqdC6-ifLi84U6pWe2ny9GUqwLgJiIG18cDXLjzQryO_joL9TNOHbSbbT2jC70_AV0Xcpv_bIq-fsD9LqBtQWbMefq0VdS5NLQpfIjzLfR_pZu9aIvEhfcgPBItvvTHyekGPVyy=w114-h27-v0

05bdeb20-9c74-4c86-9883-8c3096ac1f91

https://lh3.googleusercontent.com/notebooklm/AG60hOqN2ai9byGEtKGL2M9wL5HtmrE9H-kMJSGUE1_IxOAEaGG-6VoruYp9N2BUvUGbY6Go3di0Pa7F_1DJ88r4Hw7gO1qHsTSayhjgUGdbTiwe9PQCNo_41ERStHO--J2SmTTZJEYmAg=w17-h7-v0

ca84ac38-f47d-4676-a15f-a9b0622c4076

https://lh3.googleusercontent.com/notebooklm/AG60hOpBMboaVZVTJ2xdqmR_RJlEEBr9clPLdajFyibIDQ_pTkQd2dPGm-usGCAu1fi2ulXY6KF0k5j26WnHIS0YvHTFpSlLSDXTN1MmI5lCcK-Fzx51cfTYcJkLqkQcziaK3asDsqOOpg=w114-h27-v0

0167e49a-33f9-4c5a-824c-0b4b8715f2eb

https://lh3.googleusercontent.com/notebooklm/AG60hOplrUugNWRzKuBBNQbWI-TXtb_g_V8GUKX9FAgoIzn2GmdCfVPmifYWUq89uuI9Wi6sPvHZIfouWvMN_1kl5_WBXDHDx-sw_AdzBL8zAhPzg_HbzKiiOKzLW3nWdQnlMhdiHZYU1w=w360-h180-v0

6745e76f-da16-45c3-bf41-d078587b8d16

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## HOW IMMERSIVE EYE TRACKERS WORK

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AVAILABLE HARDWARE

https://lh3.googleusercontent.com/notebooklm/AG60hOp7_gEfPTXARxrdBhsm4HAd-CKX9Cfmaa15r_lHGXlHSC_rK1JS2MPs-ybcWC5IUea15qS3kWTH-95O4dsa_OHVMfqdUHRpl1g-t9sxQTIekmU8tysiKcFhbzwMdipvfH4LnwqEUA=w17-h7-v0

1e95fecd-e802-4a44-8800-05d8fd032d60

https://lh3.googleusercontent.com/notebooklm/AG60hOpB-LeP9Fv0GVk-_UXNdYAkiV_lXiTrkBRpOlOdzAHFFNmuWZVeHrY1JiLLnx66g3kiNUfE7MTqK-e9m6BZH6KTSWFXKsVpf81DbWueRNGX5suJaGpe-xu8Kwy2NYy0HZKK_QDK=w114-h27-v0

b5f9d5d2-e02b-49d7-8788-a058a15303a0

https://lh3.googleusercontent.com/notebooklm/AG60hOpyNF-NHp4YOV_4_A2StWXfRlYulwLDpWoQrdhLD0NvcbHIZxxmZtZmotQyZcm5lOnbnHIHSpTNhwxAtPwzokueN2GRHedcaxjWHu77ZEbb2WR1d_29_ldBeJJzTy4INqVAgnEHyQ=w213-h145-v0

ff94294d-ece6-4c7e-8a41-9e4144ef5f3c

https://lh3.googleusercontent.com/notebooklm/AG60hOoFzRETf77cc3xaSXrAwL_IsNUtmIFPVoP4uZ9E3szUtmYcCS-534pclzVk0LOaJdFbjiIpnHMsjggb0E_HSj1R2FPNJ1pZVkaH3BzoFEgRB4WAfn3_U5zjftC7k-hzc7CU-Ov0eQ=w17-h7-v0

a77a6053-c349-4f4a-a7f6-de2cd094e16e

https://lh3.googleusercontent.com/notebooklm/AG60hOprWfsQhy7CoIgxGitK_ggImiLvdGPVjEQzRbvxyKsyz6d1HCq4yMyd8lBSHNTpI473LYU7L8IPdGo1M--P7Zl1Q2_DuWX0BXv2VdAonL-xxr0SsBnu7r7MfU3uGSVU77Fau3PVZQ=w114-h27-v0

53ae355e-38ba-45ba-a275-52a306977c68

https://lh3.googleusercontent.com/notebooklm/AG60hOqSvkyeJhsXsTiNEy--m0rzX9ehFVOMIbxS-IrfcH5Rt6bPepovHYY9db8I5oMWl_EMODtZ-FoKSOJOX5aQ1jgXpd46D181F1bzKbcRlIGHrtfvsi4bbjZcUNj4Oy86QaUAGwoi=w320-h117-v0

199dccd2-bf09-4f55-aaef-29b54b748327

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AVAILABLE HARDWARE

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AVAILABLE HARDWARE

https://lh3.googleusercontent.com/notebooklm/AG60hOpeEU377oXANIM4CkcQrhItOzefi9nnemGOmmp-_JGdmPGyz3JQ2SGlCsUefS8BNqz6L9YXB4VdLmcgLHzWiQGfYCQf8DbmzBtFpCLrdSxpXcQMA8vw-p1TZGVNRe6-k12R9ushOg=w17-h7-v0

147f7d7f-16e8-43dc-9e10-e838d86d137f

https://lh3.googleusercontent.com/notebooklm/AG60hOr-jFG2TrQ1qiXEdxrMfT8jFRjmMSqoybvUA1K9OEdgRgQixtrL3ijykFYgvZK2IFxIJQXsbfscuAz7PFhLd2eD3wwGviw_M11nOE9O-5abFAhBQbpFRaP9kOvyBfOrT-zyzChXxg=w114-h27-v0

d269420d-24c0-48e8-97d3-c5cad0b2af5f

https://lh3.googleusercontent.com/notebooklm/AG60hOq4zFKmbl9r1Bwn-_I1qajbaY0lkjVtrSfjWpKh4XfSlNEVcmi9QgkpZlLO7NUhT8qeAKmdAsitOIgTDhAgCz0vUfDPhXx785VlMvb95BO-K6PH7H59H59Yl_DfbwbngxLcmbEeWw=w320-h129-v0

c4fc6fc1-7d73-4b55-8a83-c0279af3cb64

https://lh3.googleusercontent.com/notebooklm/AG60hOrZY6LSGBVRJ7JpS5AAJrQJECWkT8jIeOPiZU49D0NNgu7C0JGPfe9z4QF_4EAQF-rX6HbiT9J1-G6EWtmEQAffRvTRl3x0Jnvo8-a57NMUohJ58Hb0rnAvBf8R3PQqyrKBHJBa=w17-h7-v0

5d38c5b4-3c92-45fa-8868-8a9a6ba6083e

https://lh3.googleusercontent.com/notebooklm/AG60hOrdu7fozkkRLc8AufK009PJhyeX52MbXsNd-U7J7kQ2eKdLzUpAEVJpAGSNd-DKigXlRgLpC2zMau_cd9DaefwqwomE9JboUzrj6gjJjxhUZuz_iKWZSL8HebdLkkWcfH4nFREg8A=w114-h27-v0

582b1707-9c02-4fd8-93fa-06282ee52607

https://lh3.googleusercontent.com/notebooklm/AG60hOrPurkEbwDqZK-oFxhyW5hubRj6MQ0447akqf5TIrSQAaRvVYUDd9u9NAe-u7QciWKp61rIV-sKJV6387sWhbPpnJzxnP1apxH_NlqSggagp2nRphsZLVN01gBoEkUHKPFZEKzPcQ=w207-h155-v0

ed815ca1-2c7d-4b1c-9d56-b16f87409d1a

https://lh3.googleusercontent.com/notebooklm/AG60hOqjqUFjk0P_erxGlWAahg0Bp09VG47Y7XjVcQ90UUhuroE656ApdjFHIJ6VEFY2CdUafD7ZrT23BQ9JJu0NY1Gvv-w2Yv_rxyywJxvyquU3aRAQcwGksCj5KTO4bVi7f6ltAdtHew=w71-h53-v0

db593c14-0078-4a0d-afae-823c0659e865

https://lh3.googleusercontent.com/notebooklm/AG60hOpP7xlQO9-TzJiggYkXY7zQTSIzBl08GbpUB-TxnkDRPnxeOEvF57EhPRORRAvsIDiVN5hxyakWm78jqrRirRyoBU8TyzIFCJ3gchIUV5uq2zGFXGDBmQPzV6gNWcSPpnNhTqdfSg=w228-h158-v0

7e0ca930-1ed4-464e-98bc-a9ba436749a3

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AVAILABLE HARDWARE

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AVAILABLE HARDWARE

A Preliminary Assessment of 360 Saliency Map Methods and Tendencies in Free Viewing

Anonymous Author(s)

ABSTRACT High resolution head-mounted displays with integrated eye track-ing are increasingly becoming accessible to researchers and prac-titioners alike. This technology provides the opportunity to study human attention in immersive 360 environments. While eye track-ing protocols, metrics, and visualizations are well established for two-dimensional (2D) narrow ￿eld of view (NFOV) content, these are still open questions for 360 eye tracking studies. The main challenges are that the data is collected on a sphere rather than a plane, and that the participant’s head can no longer be conve-niently clamped to a chin rest. In this paper, we present a method to generate 360 saliency maps using the Kent distribution (a normal distribution de￿ned on the surface of a sphere), and we explore gaze patterns in 360 as a result of elevation bias, initial orientation, and free movement of the head. Our ￿ndings suggest that the starting orientation has an e￿ect on viewing behavior, and image content in￿uences the distribution of gaze elevation.

## CCS CONCEPTS

Computing methodologies→ Perception; Virtual reality; Im-age processing;

KEYWORDS Saliency, Visual attention, Eye movements, 360 imagery ACM Reference Format: Anonymous Author(s). 2018. A Preliminary Assessment of 360 Saliency Map

Figure 1: Experimental setup.

360 imagery have become more common and are motivating an increase in the amount of content produced.

The study of attention and eye movements in traditional 2D content is well established, providing insight into the human visual system [Duchowski 2007]. Experimental practices and standards have been established, which is still an open problem for 360 eye tracking. A major computational problem that is currently being

https://lh3.googleusercontent.com/notebooklm/AG60hOo6IG99Vv5aJ9nkG3U9bTRHimMtBrZbEgvX_7eYuoJxvJ34111k8ISTC71Zu2BGowCKw3AaFkDOvJ0Vcs-iCEkmmriagzCjrb8uWBJil5O-urZ_8gkkQ9Cx8_TNxEa8qLSGWJBA=w17-h7-v0

31430a5a-a61c-4412-b3ba-8d1625e8a187

https://lh3.googleusercontent.com/notebooklm/AG60hOp7aUT-QA4k6A9SN4V_FqM-pOjg5Q-uqOgQA4vqsuEZ3QgY5P6a7WWaIYJeA8D0l55aNUehZFYhJhsoGQeDIkB5PSWy_9qN3FzmafWhvzukj6Rsovn4YRMvIvEgvRaOT_aU9tLmSw=w114-h27-v0

6e7a32bc-3ebe-45d9-a1c2-fd8f008ca397

https://lh3.googleusercontent.com/notebooklm/AG60hOqCGCvm0grPwTiHFQU8wEWZn2sqWSTYUmx3mN_io5xD8MZYjqOEnleAhlPZJhj6goTJ4S_u5ivO8IkPkYwvVg4E6UWnwzwjtxNJ9GDifO6ZIPvUTPHFoC_vmcpVUUgS4f0L_H7UVQ=w114-h27-v0

3026dbab-03a1-45db-b66d-acaadbe3e915

https://lh3.googleusercontent.com/notebooklm/AG60hOp7MNFuT1euxAuOnatK-KlfXCeQUfSIsyVyVNcRybbSGRftEKZ6pNH-kNQYEIQKn7MnTY0AntrTXG_St6aWsrR5-iMAleMAeZIi0XhM1URc7Af5pCSUrBLZ8lWa9-j6NoTK4xBSGA=w480-h270-v0

bc7dcb42-f14b-44b7-8a77-c69dbfe2f0f4

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## Device Eye Image Resolution

Sample Rate (Hz)

Cost (USD)

7invensun - 120 $200

FOVE VR HMD 320 x 240 120 $599

aGlass and aSee - 120-380 -

Pupil Labs VR  (VIVE USB) 320 x 240 30 $1,572*

Pupil Labs VR (Dedicated USB) 640 x 480 120 $1,572*

Pupil Labs AR (Hololens) 640 x 480 120 $1,965*

Pupil Pro Glasses 800 x 600 200 $2,066?*

Pupil Pro Glasses 800 x 600 200 $2,066?*

Looxid Labs - - $2999

Hololens v2 - - $3500

Tobii Pro Glasses 2 240 x 960 100 $10,000

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## SPECIAL CONSIDERATIONS FOR EYE TRACKING IN VR

© 2021 SIGGRAPH. ALL RIGHTS RESERVED. 20

https://lh3.googleusercontent.com/notebooklm/AG60hOo5NJWI3vzE6DRtwk3y0kCZYIyT2g6tvh4cETziUAWki8tsA2z3GdOIotvTyYQaIq6a-6XUu0466Agc0yVvohRJosQlRgIgA2Yg5H8vftjTJQqgPfy9cpi_-U4Gmweyyis4N9A-=w17-h7-v0

77be470b-5015-46ad-9c7a-33f0ba64eb28

https://lh3.googleusercontent.com/notebooklm/AG60hOp8rXFoyWtQq-QT71s4I-EOE6jZXlkhwMchIjsCBPzwb6GFK6S7gYjcr9M2B3uY8uH0XZ0s7uK69VbMDW3yJo4hIvPY-hUx0RDneCoTcRCih4-0lzrBDsfNQpsd0q4-tHtagv6_Rw=w114-h27-v0

a587d286-c284-4133-863d-341d30a7e5e2

https://lh3.googleusercontent.com/notebooklm/AG60hOqSOO1szJBEROg8IGzbWXMGrq4-ZWjWx06dLxDd0Rs6ojWYVJ7atlofCULqnZEg78VBAL6zD-Reo-GvItm3FR4Hl6QRO5Hajn7ge1Z9RYuwk1eWFxKm4JkV3cOp5aod4ayjD-tYtQ=w17-h7-v0

bcda74d6-7b8e-4f62-b73b-8c105f4c2a45

https://lh3.googleusercontent.com/notebooklm/AG60hOpPnALVXiPh_JMIUeYatque6pAMEbqoziA_P1YjdnNwbpU9WCneMapchJhcfugHPTm6lYzBfKf1nEwOV7FomuxI88bUFOh0_De56XGOw1z5tQ1zMFQxyzbU-IGs5GcjyB1HbT_Q=w114-h27-v0

43ecaccf-9c35-4fbe-931f-b51b1c6c0533

https://lh3.googleusercontent.com/notebooklm/AG60hOpBA4_y3DrUzAvksTVH10ETCuLWMWL3SzgIhNWChs43h9rAO4hTmSJe329lYOH_EwuaBdd26T7W5uedY3nfhw34xMYws5DHBNo9ILNaBsE4O2acJLHNw-oWEDqLQFdU1fxooFg7=w246-h195-v0

617da0d0-bf1e-4e13-ba7c-8c4b055b4935

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## VR RELEVANT PARAMETERS

(Eyes rotated within the head’s coordinate frame)

3D Point of Regard (Gaze in World)

Rotation within socket (Gaze In Head)

(Head rotated within the global coordinate frame)

3D Point of Regard (Gaze In World)

θ Rotating head (Head In World)

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

https://lh3.googleusercontent.com/notebooklm/AG60hOpSf2UAP05z7t50AFLq80TdyAmE6eOjEOTv0qnVqc4CHjcO2dAIY2uBmlsz82CJETKTcIncI5PhWcV7DO6kWaTIUcPawjrLaL8wx2h-D9xYUi1sA98HBSyuC5B1Zul0pYnD3H3kuA=w17-h7-v0

0e386841-9561-43a4-9075-74730c2c2d05

https://lh3.googleusercontent.com/notebooklm/AG60hOo8nvNAHdC_BhyCByqiZU4e7EpgxxODNcUv6caawG-hokXEyJSwZUJL0p6Axrys7tbcI0Mo8P-XbGiE5i4yoHZoiwSThO-kHDLbzZAyVEthoAGM57aT4bOrDED3O1JGfLKH0_mSxA=w114-h27-v0

bf8adab6-462d-4946-a24f-c171b661bf2c

https://lh3.googleusercontent.com/notebooklm/AG60hOr-7HiDbHtJgK0-6UzN-F_dtM3K20cg4tit8Ijbre1pGMFHWgK7m8X4Z5czWJbRbQ-0zoWEPEQJ8NXe1Gx_Nmla8BXq4zn2Od3aclcnfKAQAIpx_xxDd8_g6Oq9NyWkFBhVMPaU_g=w1-h1-v0

5db80985-9353-4ceb-962f-4da9626987ec

https://lh3.googleusercontent.com/notebooklm/AG60hOo0Ue8NXf2sD-zKKpYFmRHCsI75JFoYzKoB_7bieZTFs72FE9JbFl6Vv4w1BKHCaw7lT1o6KdVjC2j_zPJ9Z2Y_Aed0hNf_qh5KuE1gVKdtigeywU-QYW1rsDt_T6B9pSONUGIK=w114-h27-v0

1fd7a578-b37d-4181-a10c-c87e3d94f68d

https://lh3.googleusercontent.com/notebooklm/AG60hOoOoVNUUYOrbcrsfvOJ8IgyxM3H3AndzGPZSHt6-p7wL4Ut0Ii92y3fDhEvIC4uLmkYTj0b183ibt4o0dZFXLQ2GiZV38OI28qajq_gfaJCvxEiSANid_rE_r8PNMItfsLIbUftyg=w480-h270-v0

35c4bf96-d3b4-49cc-b0c3-28fedf263569

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## VERGENCE AND ACCOMMODATION

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## COLLECTING DATA AN OVERVIEW OF HARDWARE AVAILABLE

© 2021 SIGGRAPH. ALL RIGHTS RESERVED. 24

https://lh3.googleusercontent.com/notebooklm/AG60hOoUj8INjg9R9RHi2I5FzvvUlw3_mTqCEuNgs0I0n1ZOLbC4QBqU91er-zOs2TDVT81e_y5-PirO5IUUoS9Hv1j3GU22i0u2GFm-CGZ9sJ6KIIrlntIYis8u99x4LxjIVnhsTHpR0Q=w17-h7-v0

1a97b116-9418-476b-ad92-bff1fa0664e0

https://lh3.googleusercontent.com/notebooklm/AG60hOpET53WJ2EsmsTIiM7bUUFUIhgjVSGh96XVaokR2SYwRw1O27g1uPqhZiq_rJL2zlxGF3IWB_jFNe5otYWWdw3GIAqhfOgZMa65Jk0OSgZ7DWy6KcrzvHN81K9bzkhgcczOtWIwkw=w114-h27-v0

fbd53992-5733-4dcc-9f3c-f087d7007704

https://lh3.googleusercontent.com/notebooklm/AG60hOpnblhcRna-bUP0GOtMfVkZZq8uD4_hx46hO1y2nQX53j97qTAsTzboT8iPkEBjE038_yEfvCAjeI8FbY55JpP-rGpAw-LEN6X_R651nuyJaUrcmlfewyH06ED-KHqOM2aiRCXaQw=w17-h7-v0

f7460f4a-5d71-4060-bbc1-2794e4c3423b

https://lh3.googleusercontent.com/notebooklm/AG60hOrxSjhpLJJXBMQuFSlXAxIWp6bla7JsL7YrTb3csaz3emcbWBXJrbLKqMHb7QEUlWdjzputPmX_ZSU8v7u5sPaAfvMN24rwXh3iulP1YIlzsmlK0WEelasC3KnUdhIe3FzJPFG1HA=w114-h27-v0

554b72d3-b935-4671-af1a-1686d3400a4a

https://lh3.googleusercontent.com/notebooklm/AG60hOqjiaXWEKmHvGHCpOti6Gaqgm1GiWW66-ipOrsB_u9_OKbMdnG0csjbvcbTNw-DZ8HlDkTOfd9uZ6INdWP3xtmFAJvIjGwwxCGdpUYg3I_wbrMRjchA4nY246sfYPqtBTsWa4_v=w174-h193-v0

27335c25-79b4-484e-8750-4f1bbe932b55

https://lh3.googleusercontent.com/notebooklm/AG60hOpQO-_figiuEpN0kQcOx9jmmvf7oh-rp9w0oGkE5mu3CUxYYjlDcTRzYLEdkJX0U0FdukcrJJjMdDH5oL5QbyWlB07fGtU3bdi1NImvT3YoF8G4ks3RNA66orIDFy9pZKq_ECoe=w174-h193-v0

a741064d-6308-4035-aae0-0c7b07bc16c0

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## CALIBRATION

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## CALIBRATION

https://lh3.googleusercontent.com/notebooklm/AG60hOqbHaN-MsWsSKFnoK_oUYpAZEztlYjSx6DrD-r5nzhs6amf0k4DhUHAiqbaCPOh69VQ3hB8Wmnyu8YQWyJw_4hsDMpjMEDqoX3chkAZKThduJpUwVqtSUDi0xpcJA--JfHBRvMr9Q=w17-h7-v0

b02bca1b-ff09-4c02-812c-6a6a8f06ac49

https://lh3.googleusercontent.com/notebooklm/AG60hOq5s7Sy-zBkrG8RUEB8koKw5-QdgNaIkcY-qtsbkIavpUHDrpbgiJzk7E4-JX98iWNFQEGZY3GizQ1ttyznP0UxvmqnHDX3Slf9FlXtAan8Dk_bPYojec7l2qSbvlK7O6o6V_565g=w114-h27-v0

2054be69-61cd-4c7d-a3f6-e035c970cc5f

https://lh3.googleusercontent.com/notebooklm/AG60hOpXL08Ef5SIKQXIeb-j9wvoslAYdY5quvbLG6qwn45T1S95LVpfnQNRxwqMm31tg4hN-neCD3bBZ5sbI8sL4GHSmjmw4XkZQxozU5tG1sZbnbL41hXbB0Nm00KC_DuphQMjQNY-qw=w303-h200-v0

19516494-a882-4f19-b423-c9dddf47f909

https://lh3.googleusercontent.com/notebooklm/AG60hOpfTL0CW6HKeTjqUfLkTQDYILzityEByOgtGxWQ4AkmV2EXs-NyyT3AHxgf_SVk0oIDQ8ncPO3n6RyQ-uNZSNUiJh0zewFgvsrL05Tw3oUK2d2-VEeqoNWLNoFfUFanPOfXZxbFTA=w17-h7-v0

c3b64afb-3ca0-4ebd-a649-b358ab5f0553

https://lh3.googleusercontent.com/notebooklm/AG60hOp3vxG1mmZJACKT-RGl98RgyH2Zm9ClGotKzNQL6in3yR_mK4p95j-4IkNfYbEje78yW0wnpfieFWZcld8JVyPOvtq2SRaEKvmWNR7vbwzPU8mWE6SbuaYSdVLfnNvLCdQYY6gGuQ=w114-h27-v0

f5aca750-1320-4e89-adc4-cb58bde91176

https://lh3.googleusercontent.com/notebooklm/AG60hOr6QiMXJ_a3Xl44R_8eIAhq93TtLU3BGjXoGJEd7jCU78q7rlnrS96iFYvBVtLxVX5c5Eds3Hm2JUIuKYUpokOlcGFz4wo8tjB33o5_khaOczsvNqirgNwT5xOefAiO1cQZ8n8Tlg=w233-h175-v0

c4f79e72-33c7-4cfa-8b45-70de56e4b701

https://lh3.googleusercontent.com/notebooklm/AG60hOr4-T3agLsRh9j2PjGQvx7faWhG3vGYM3Y2xqpYJyjjDLq64uhH7KfY77uCePRXeNuUckcryLteNBz2QhRbaRHWTIuhuAh_gvy3MFQS4HfPYO7Qt1zej9JZtGKZTTaKFwKTRLri=w170-h112-v0

0522d504-e834-4300-a2f2-105254d80172

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## PARSING DATA

https://lh3.googleusercontent.com/notebooklm/AG60hOpjBJ07P5gMcosoJidEDB02s84el2NszYaj-14kBHDw1p1NkrDhJqY-5r9mj_YA6P9gxMaG8Lk4Jo8gZxSHY3Fds8cSn40TCnIzMY88_Dd6PCrQI1G_QHf6uEUyolLgdVAzw42f=w17-h7-v0

cd6b3d9b-98be-4d2f-a774-181e5c68c757

https://lh3.googleusercontent.com/notebooklm/AG60hOpPYos2upBRIXN4XlK2jrauGYFHi8c4IuWjJN986LuevsQ4C-e8aneJbplNPRhppBwg5R6z2fCuqZU_S0NxI5VphQuc_Lq1KVBsTpSVTQ3v1_GcRnTaaI7iak2H65TWR2VUFALQ1g=w114-h27-v0

4ce69b8a-8654-4e19-89b7-9309e7808cbf

https://lh3.googleusercontent.com/notebooklm/AG60hOrAJDJ5mpZNS6IP51fwTlvjH4NoSvwHvgIO3HNwy4ZorHPVqROreZN3l78KlGBJMtS81jAZqkUvXkqnepdMwNRvfocoFN-siCEiB993-MBg8i3bbdOhTmKboAaP-evtCigo8POW9Q=w320-h178-v0

edd9dd44-3a2e-488c-b2b4-aa6876df01ce

https://lh3.googleusercontent.com/notebooklm/AG60hOoludflso-999YllcKGWeX4yzbCQ9j2kidwb3aRCfu75LHVuMSKip6XlesuHnrPK4BEBsSjTJOTilYweIb6B-ZBlyJtd9VofI1G9AoNHAhRO7pHUGNPhPn_EJY6YiwpR7ttKes7=w17-h7-v0

7ef00e0e-4031-433f-8953-43dbf5c71946

https://lh3.googleusercontent.com/notebooklm/AG60hOpaQYslY_-dMeT0odL92EBGWXPb4ovBYC76a4QtLoV1x21Ff1hMRqahXrD1w_kKQfLoHhckSZyaDt6Eq78r7I3hf-Z7iByVAzU1p8R0-2VlTUIeeMz-HHZEaG0tROITWC7BPlW8-A=w114-h27-v0

661a9276-a4b2-45ca-a5f6-536873f0616a

https://lh3.googleusercontent.com/notebooklm/AG60hOqUTUwDy5NP2rnLHm2xcLiWwptJyYR3iVWBXMQYUnq4nywIeiaDHiRIC8CaKjur9LyLvGxq_UlzvQ49McBUSVmd-rn8x8HFK0Py0tWCedZkYSEFGAq_vajNOOrzV0zgs2kXYVsE6A=w352-h186-v0

a6f21091-0714-4308-94a9-5e9862f40347

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## DATA MEASURES

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## GAZE PATTERNS

https://lh3.googleusercontent.com/notebooklm/AG60hOpF4Ct1R6sJp9MqFMI6EZmqw236U7yxR1P160HNpNufDRzO0Yd-4EK_wW1HgXBYbW2_LXbIXJ2NKuvlpQCev3ugCC3iA8DNlat1FCrCHsKATem5j6T9yYOfe_NBaZQyUA2qAYEZ=w17-h7-v0

a7f6209f-73e9-4530-85ec-8b4baeda4ecf

https://lh3.googleusercontent.com/notebooklm/AG60hOqr1jIkiVaRYRgf1FWch6-ig2vobU2XBCgjb9cZTcYlaMViILlKxqo0EwOubMN88e1drlOzIyIEGk2BnP_ZA3egB8pOjbVv3YXqut276OzIVBO1kCUztJRvbY-7b_XDXsEFgWFYUw=w114-h27-v0

b656096d-8f01-448f-ae6f-0f320ec38b04

https://lh3.googleusercontent.com/notebooklm/AG60hOp2ykOFOXdAJRLNWnRt4MZnxYxdQXnDEvy7C9_96axkdmImcGepwLoaYCztmngu_V6ZT7m-Mjgi1G8EyHwDVhHTXn83nWSpMmOjtvB_Ef4nwbke2DAPuRgbTMEVxS0W1xSC6kVKmQ=w341-h192-v0

91ca255e-520b-41b1-8dce-1cfd91dad1db

https://lh3.googleusercontent.com/notebooklm/AG60hOpYha5NkO8sUYCKvVA024aqAyyqT3pEywR3GWgYap1FFO0zTe8FL3TKRJ7gOkBtCgLUdwTvwDuO8Lv8rcQqD8riBAj5Q37tSTxMCBf0ffHuVzQOQSTfsKPmgIZ8ZlZM0COEZ8J2=w17-h7-v0

d1226d3b-4e6d-48e1-8698-cf83ac73e04d

https://lh3.googleusercontent.com/notebooklm/AG60hOrUTVS_Z3muJxfg9AKDSzem2GosZMJCsM4FDhhypa85G8ph4g09yi-96NYJ1D0hr98OI8TnF1J7P0-nyOr47PAgwfZvG-f17IavY5APcgrb7I45vnR3TS9rs2vILCWN8K29nYVIdQ=w114-h27-v0

da916d6a-6085-42cf-9c7e-6ebd442c554f

https://lh3.googleusercontent.com/notebooklm/AG60hOqPqOvIKVUfcQR56NQCSkLZ6g2nJgR9EKM8EL8qyqtU65ATu7LwWSL9X40_yeExXfxBjJ7KzraB9kQB-vNKo_4b_RcmadSgWbA6_nk2512KpUWKv7Kj6PzvtU5hTp0G85lNvzn8LQ=w480-h120-v0

f6805101-952e-47b9-82f6-b8564ddc67df

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## VR HEATMAP

https://lh3.googleusercontent.com/notebooklm/AG60hOoJPe1St6oPh3UKLQDNWjnhipcc1Mi_1-EtzpwF-NWbpro8V3KK3AvNNxwSX3_ANiThQgsP7K2WnXgcdELO16uJUKP5XxbYoM_14bDzuvJg54IWxBE2qDlTvVSIKVuDUtyyhsMF=w17-h7-v0

aa4ea8bb-6b85-4bc6-b6c2-320fef5bbfdf

https://lh3.googleusercontent.com/notebooklm/AG60hOqr_aI5QsrzDxiJcs2kxGce6Wm7aJeBMUwrEOkG7Woz00Na-eUIcjrExaZ_8r9Rl0aA88YPgo1gSyHl2W6j6XV1_BamDwVgPlXqsYBaezxuY6ONY5iOSv-bf5XOGjkxcJGq_3Bu=w114-h27-v0

d49cb0a2-5054-4c16-87ec-a52cf97e60d0

https://lh3.googleusercontent.com/notebooklm/AG60hOoW9SZnLsrPDqZl97tr-fwbJYhnJw4UXhSvr9vjwrRZe-jyFH4DjglUcEOrQaClJs-cTVCQeHbhdnsO8ICS8uNk-cjHPLkhzuq5l3bGB7Jkixmf_kbaj1k8m-VXYmIC1oEeqEgxgQ=w317-h159-v0

7ea69283-99b4-434a-8681-0cf375b87577

https://lh3.googleusercontent.com/notebooklm/AG60hOpd_oV5F8JxcrXCiSksi7OfCyfDABwpTio1ZbDhugYTtT3GZDFvpq3I8y9MBLcIwLDjgP3b2jqj-f_sNoHNHRy1Q52_0DASxJaNmEU0C2fXp0O8232FsP2FMYCKEIZlbd69wgm3=w17-h7-v0

ddece073-542c-4c13-b275-c162a49e04d8

https://lh3.googleusercontent.com/notebooklm/AG60hOqKIUeosfxbsTnQftBPjiDh1kueZHukOhNzuTSvXIeTNRH5gnelukeCxHaJdyu9z6Gj8cZ_XM6KUdurXdbz5rrYcupR1iFFrwTy0nfP-Va_GnfK_9Ykt7mqYs8H5drf0mrRVOWtxA=w114-h27-v0

1a7d5474-2b0c-4e83-bb70-5d5b7c972b33

https://lh3.googleusercontent.com/notebooklm/AG60hOrPcy6va1AhxRKllmZwbq9V3p912UhnKDOruxON7wGU1iW1Isod7l7OYmdzyP9vtj3qMvBCYwXRVvu-fm2WERM7kTN8xZ-qpKEJDjgjRweHQXsRHGZjKPiZnMHvjPFHzyCXtkcJFg=w213-h163-v0

54202971-4f40-4eec-bca2-63db4b542d83

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## VR HEATMAP

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## FIXATION ANALYSIS

## Time to first Fixation

## Fixation Duration

## Number of Fixations

https://lh3.googleusercontent.com/notebooklm/AG60hOrIT7KX8pDkEsiANJoo8k3TPBOHVGkrZgRKIPLwlu5BR-4sqFLckHH4Uuzx9VCqFhnKeEZwN6VFly1KUPRZfrm43mbp1GJ1XnEiNb1gvtnMpZkAWWX-9P5lhBAt71iQU37bUn___Q=w17-h7-v0

e4b6f97c-0b66-4e16-a6ba-c176cf8077da

https://lh3.googleusercontent.com/notebooklm/AG60hOpNHYJLeVMVarc3mBTZ5xNGXTBCEwHTSU7Uy67iSaL3wHGmDPAbyXbm5E5bJ6IddFd4hPTIb9Qs_kDWbdcpLOPS_zNhgOq1UgauUNWIDi8MLsIYrI0hqUEawfZrfRpkPJmS-PiKzA=w114-h27-v0

d9e07ccf-ac32-4436-9888-f57de82fa7b0

https://lh3.googleusercontent.com/notebooklm/AG60hOrYF4Ri0xCMaFKc0ywgoRrZQua34Z7yzPEw3NPX_m4u77LoNh--pOEYhAoLFD3gq3phg-xV3wQLVuA8t-t1B3TyVk7aWID3g-4beRIF1XvZ9V4Fue2bEa8FSp5qB68ag5FFJAw3ag=w241-h164-v0

7650e814-81c4-4b43-92f9-962b8072cfab

https://lh3.googleusercontent.com/notebooklm/AG60hOrEReBptLTlmQzbAtCbgQeyTZyFgWyutyiuUJCUuMDkeNysp5Uw19euBKaPQ1_HdBDz_CJUe8_AaRjN6yzlNijX81cD1qmMhZ4Gj72nQTOxoesA7NHTWSxmvivMNS9XhN4cHYAEDQ=w17-h7-v0

b3c58ed0-e7cc-404c-afa6-5795496b7bf6

https://lh3.googleusercontent.com/notebooklm/AG60hOqhJINOkT501KHFmAkKqjkiHFBRMCUvxS0oClCQSCaCPIx2xo6jxRs31P351mVboFvpYCnW4k6HKF4aie8IAD12kMw7_M8hAQq1227j9apgl_XYBRPkNSTgZwl8AQyTwvGV91_TmA=w114-h27-v0

b13e8c33-c18c-49f2-a757-e5ce2157b363

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## AREAS OF INTEREST

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## Number of times gaze returned

Number of Respondents (as a %) that were drawn to an AOI

https://lh3.googleusercontent.com/notebooklm/AG60hOqa20OXVTwHR-Z8TUTDUAtCb5keC_j9MgEdvqHxtqN-0oaXYNYsC3h5ejAnz78uSaqfoAtWz3g7N8KVfiOmmgdr_LeYAmJR3iUXlu3qdQZ2IcUgHiQMAp-mP2mPgXb2OUg9jQSnKQ=w17-h7-v0

6d68df29-1781-4e0b-a4cb-ee83d7f6c310

https://lh3.googleusercontent.com/notebooklm/AG60hOo-L6C7dKOETWQhVv0ZYJGH5uq_lHjM0qy40BtF_RBwubH8HqhzK_VJkyF26AtpxuIA3-c7OryUqmGbI3WG0e09Vk99BeeF5WUI6ymBGYlTnAhTuXx2WJs6S2YpoT9aFigu7acR0g=w114-h27-v0

bc75629d-8faf-4bc3-9abc-97190e819e08

https://lh3.googleusercontent.com/notebooklm/AG60hOrK341cq__d6QuotEOhzWJcu3Agt5jchzN4NNtbILcqz7I2UqKdAcWRpnunghx_xd2wSyWzWaRrKL09CM13joYvym_7Q7eLFTnPtcAqv2h5B2XbE9ay6VGgpqjMR3fxIssOh5sqRA=w320-h179-v0

d0296c23-da6c-4bc4-beaa-0e2b72a45528

https://lh3.googleusercontent.com/notebooklm/AG60hOp7NOlParhzsX6AN2TnRH-mjVGw2J3cbAxwJA4ftyO7HbM9oHFmTvdmxoDl_8fpa1oBo-RMBeGGQe1HAuHyFuCAoIrTArcamNnWYK29Pf9i1JIKWQqjFBTZ678yzUasH2C1iMT_dA=w114-h27-v0

3332ca15-a92b-403b-8ff6-3d8a343a22e9

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

## TRENDS AND A LOOK TO THE FUTURE

THE PREMIER CONFERENCE & EXHIBITION IN COMPUTER GRAPHICS & INTERACTIVE TECHNIQUES

THANK YOU ANN@VIZ.TAMU.EDU

