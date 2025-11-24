---
sourceFile: "Applying level-of-detail and perceptive effects to 3D urban semantics visualization - Eurographics Association"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:09.566Z"
---

# Applying level-of-detail and perceptive effects to 3D urban semantics visualization - Eurographics Association

0fdbbbe5-85f1-4c19-9baa-c72bb9690ee5

Applying level-of-detail and perceptive effects to 3D urban semantics visualization - Eurographics Association

f7f43004-3b3c-4475-86fc-f5451125748f

https://diglib.eg.org/server/api/core/bitstreams/61e25ba0-3b1a-4ce3-97da-84e28a85bb79/content

Eurographics Workshop on Urban Data Modelling and Visualization (2014) V. Tourre and G. Besuievsky (Editors)

Applying level-of-detail and perceptive effects to 3D urban semantics visualization

Fan Zhang,1 Vincent Tourre1 and Guillaume Moreau1

1L’UNAM, Ecole Centrale de Nantes, CERMA UMR 1563

Abstract Urban environment consists of various types of data, both geometric ones and non-geometric ones, among which urban semantics are important sources for non-geometric data. The modelling and visualization of urban seman-tics is one type of information visualization (InfoVis). In both 2D and 3D environment, a lot of work has been done, which use different kinds of representation forms to illustrate knowledge and information stored in the original abstract dataset. This paper aims to apply the idea of information level-of-detail (LoD) to urban semantics visu-alization and a text-based semantic database is built to illustrate how the idea works. Then in the implementation process, four perceptive factors for text visualization are chosen, while we mainly test, compare and analyse text size, aiming to better aid users find new knowledge and make decisions.

Categories and Subject Descriptors (according to ACM CCS): I.2.4 [Computer Graphics]: Knowledge Representa-tion Formalisms and Methods—Frames and scripts

### 1. Introduction

As computer science develops rapidly and user demands grow daily, huge amount of urban environment based ap-plications appear to aid urban planning, transportation con-trol, disaster management, navigation and decision-making support. In these applications, data can be divided into two types: geometric features and non-geometric features. Geo-metric features provide users with direct information about the urban environment, to help user establish the knowledge of where he is and what the surrounding looks like. So the user can see geometric features. As for the non-geometric features, which are always abstract dataset that users can not directly see through their eyes, a translation process is needed. This translation process, from abstract data to a re-sult that users can easily understand and find new knowl-edge, is information visualization (InfoVis), which includes urban semantics visualization [KHG03].

In this paper urban semantics stands for the non-geometric features in urban environment. The visualization results of urban semantics differ themselves through num-bers, tables, charts, symbols, colours, figures, texts, links or even 3D objects. Since its creation, text is one effective way to transmit information to human beings. This paper chooses

text as the visualization form for urban semantics, to study information management and promote user interaction.

An information level-of-detail model in [ZTM13] is firstly used to create an event semantics database with different LoDs, which is later implemented in a case study. Four per-ceptive factors are chosen, three parameters are used as in-puts. Results are screen-shots of different parameters apply-ing to perceptive factors from the same camera position. Dis-cussions and future work are given in the last part.

### 2. Related work

This work is built on the basis of semantics visualization, level-of-detail for semantics and perception in visualization. A lot of work has been done respectively, but few of these works combine them together.

#### 2.1. Semantics visualization

Semantics enriched visualization would dramatically en-hance the usability of urban models and would open the door to the use of complex models in more sophisticated applica-tions [PCS12]. Semantics visualization develops rapidly in both 2D and 3D environment. A lot of on-line tools are avail-able for InfoVis, which deal with data topics firmly related

c© The Eurographics Association 2014.

F. Zhang, V. Tourre & G. Moreau / Applying level-of-detail and perceptive effects to 3D urban semantics visualization

to our daily life, such as economy, health, energy, education, politics, population and environment.12 Most of these visu-alization results are represented in 2D and through graphs.

In 3D environments, [CWK∗07] introduces a highly in-teractive way to provide user with intuitive understandings of urban semantics, population census information to in his case. They aggregate buildings and city blocks within ur-ban environment into some legible clusters, which can give the user an mental impression of the city, even if level-of-abstraction is applied to the city model. Population census information is superimposed onto the city model in form of colour, along with a detailed information table aside explain-ing the 3D visualization result. Invisible city3 uses social net-work datasets to describe a city of the mind, such as how a topic moves in a city. It combines time into the visualization result, and lines are drawn to illustrate the moving trends of topics.

Few of above works use texts as the representation form for semantics. However, our work is different from the pure 3D text visualization problem, which mostly focuses on the performance of a single text or a small group of texts, such as where to place it, how to improve its visibility, how to highlight it out of the environment or how to avoid occlu-sions among themselves. While we emphasize more on the overall performance of all the information/texts visualized in the 3D urban environment.

#### 2.2. Level-of-detail for semantics

Levels-of-detail (LoD) were firstly introduced in computer graphics to describe the detail degree or complexity of an object, which mainly dealt with geometric features [Jam76]. Starting from 21st century, LoD for semantics was proposed to enable accurate information management and knowledge sharing, and to improve the reliability and performance of visualization result. The five LoDs defined in CityGML by OGC are geometric-semantic combined standards to con-struct an urban environment [KN12]. But one drawback is that the LoD for semantics is always linked with that of ur-ban environment, which is not independent.

Currently the LoD for semantics are normally illustrated in 2D environment, with figures or tables at different views, such as [ZHRT08], which uses a tree map to represent dif-ferent LoD patient information, which can be used for analy-sis and comparison between patients who have similar diag-noses. This type of LoD is in fact a multi-view visualization method. This work will extend the LoD for semantics into 3D environment.

1 http://www.visualcomplexity.com 2 http://outils.expoviz.fr 3 http://christianmarcschmidt.com/invisiblecities/

#### 2.3. Perception effects for visualization

Visualization is the technology which makes data visible to users to enhance communication or understanding [RL95], hence we should take human perception into consideration. The research on human perception conducted by psychol-ogists and neuroscientists has advanced enormously during the past years. In the book of Information Visualization - Per-ception for Design, C. Ware details how human perception and cognition work and why perception is important for in-formation visualization [War04]. This book gives compre-hensive suggestions on how to take advantage of perceptive factors such as color, lightness, contrast and constancy to de-sign an information visualization.

Besides, [PKB05] works to evaluate the influence of lay-out, screen size and field of view on user performance in visualization. [EF10] gives an overview on visualization and introduces techniques and design guidelines from the view-point of perception. [MTW∗12] works to analyse the effect of styles in visualization. For a long time, Shneiderman’s fa-mous mantra is wildly accepted: Overview first, zoom and filter, then details on demand [Shn96], based on which a lot of information visualization are designed. Here are also typ-ical visualization techniques concerning perception: Fisheye effect, one kind of Focus + Context technique, uses geo-metric distortions to guarantee geometry continuity while information at the distorted part is always hard to read for users [Fur86]. Overview + Detail integrates details and the global view in one visualization result but lose the continuity of geometries [SWRG02].

### 3. Urban semantics LoD dataset

#### 3.1. Semantics LoD model

A general strategy for semantic LoD is proposed by [ZTM13]. The main idea is that a more detailed semantic level enriches part of the semantics from its upper level. S-LoD is used to stand for semantic level-of-detail. Sup-pose here are three semantic levels, then S-LoD0 consists the overall information of semantics to be visualized. In S-LoD1, two or more topics are to be integrated into S-LoD0. And there are zero or more topics which can not be further enriched. Semantic element at this level may have zero or more internal relationships with each other. Then at S-LoD2, similarly there are two or more sub-topics and zero or more un-enrichable semantic elements. Zero or more internal re-lationships are possible. And semantic element at this level can be aggregated into one or more upper elements. Then for following levels, mechanism is the same as that for S-LoD2.

#### 3.2. Case study

This work chooses the annual summer music festival in the city of Nantes, France, as a study object to create an ur-ban semantics dataset with LoDs. The festival, "Aux heures

c© The Eurographics Association 2014.

https://lh3.googleusercontent.com/notebooklm/AG60hOrkRa_Dxf8w4KGql-QZ_g5fMeu-J2lhiRX8JDgtKAPS6NQoGIC5FTZeiQqxTq8BF6Mav6tEhADtsw1I7PswkPnf5SU67JssQSZ2LrOBQ9sjt__DfGWBd7t-EHXZILUOtnDbDjhd-A=w967-h404-v0

7cf558cf-3bff-47cb-83f1-a5265a6a4b39

https://lh3.googleusercontent.com/notebooklm/AG60hOp2zCov5-x9WiDidPI8T7sZvGnO38SJAAa_1EKMlHCmqZHbvZ3HvAwfD5A1tqubRM1mTC-AzU8jkX0QAPpwNyRWhrHFbh_Ixh8xO_el0u1AvFfdvUMr4ac829A0ylaE1pPT91IbpA=w671-h511-v0

3311a83f-923d-4477-ab41-508cd2c917fd

F. Zhang, V. Tourre & G. Moreau / Applying level-of-detail and perceptive effects to 3D urban semantics visualization

d’été" in French, is dedicated to give the people in Nantes a chance to enjoy the cultures from both local and abroad.4

The semantics dataset is built on the basis of texts, be-cause semantics about this festival are mostly textual in-formation. So we will mainly use text to represent the se-mantics. As for the 3D urban environment, datasets from [HMM12], which models Nantes in five different geometric LoDs are used.

Each year during the festival, there are various types of ac-tivities in the program list, which are named as event in this work. Based on the theory above, along with the program in-formation for "Aux heures d’été", a dataset with three LoDs is created as illustrated below in Figure 1 [Bri13], in which E-LoD is the level-of-detail of an event in this festival.

Figure 1: Event semantics with three LoDs.

E-LoD0 lies on top, at this level, semantics are data about the "Aux heures d’été" general information.

At E-LoD1, there are four topics: Films, Readings, Con-certs and Young audiences. Semantics of each topic are information concerning this topic.

At E-LoD2, here are 32 items in all, separately enriches contents for one item in E-LoD1.

Based on this dataset, the event is structured with at-tributes as listed in table 1, in which LoI is the importance degree of this event compared with other events at the same level. On the basis of this data structure, the data for "Aux heures d’été" is organized and stored in a XML file. In re-ality, the place where an event takes place might be inside a building or in a park. In our case, Place is stored as a 3D point when creating the XML file. And for the LoI of event, it is set as three levels, from 1 to 3 separately, among which 3 represents the most important level.

Attribute name Descriptions Name Event name LoI Level-of-importance

## Content Detailed information of event Place Where will this event take place

Table 1: Data structure for event.

4 http://www.auxheuresete.com

### 4. Perceptive effects for urban semantics

#### 4.1. Work-flow

For different representation forms of semantics, different perceptive factors can be chosen, such as [PZG∗13] chooses different rendering styles for buildings to generate differ-ent perceptive effects. We choose texts as the representation form, four perceptive factors concerning texts are chosen: size, color, transparency and resolution. Input datasets are urban semantics and the 3D urban environment, from which we can acquire the object space distance and screen space distance based on camera position and user interactions:

Screen space distance: users perceive the visualization results through either computer screens or other dis-play equipment, which are in 2D environment, hence the screen space distance can be computed. In this work it means the distance from the screen position of a seman-tic item center to a screen focus point, written as Ds.

Object space distance: the results of 3D visualization are 3D scenes, objects still maintain their spatial rela-tionships due to their spatial locations, from which the object space distance can be calculated. In our case it means the distance from the semantic item to the cur-rent camera position in 3D scene, written as Do.

Besides Ds and Do, LoI of the event is also considered as an input which is pre-tagged in the dataset. Figure 2 illus-trates the work-flow:

Figure 2: Work-flow of applying perceptive effects.

#### 4.2. Processing functions

The inputs are determined, then it is time to use these inputs to process perceptive factors. We decide to construct differ-ent processing functions using inputs as variables to generate the output. Currently we have 8 processing functions, in both screen space and object space. The results of all functions are normalized between [0, 1]. For each perceptive factor, they

c© The Eurographics Association 2014.

https://lh3.googleusercontent.com/notebooklm/AG60hOpCfZDv5UAKBl_gz4C35cygYz8J2T_IRCPOSDkbsUi-UBT0GAb7BW-NqfqPoq-yErZHoiE_jbhC49xA1D33Q3fJbuu0Yl0wWCbf6J-zRXq24qCTjXjCAXLMcsZrbQL-NtLLhqGV=w1440-h900-v0

88840400-8803-4438-8c5e-0c8e2d62430e

https://lh3.googleusercontent.com/notebooklm/AG60hOqOst4NpfLL8WlvxL_ESU9vbughgkRFgHyhB-huqAPDN3q55wwZt6TTHwiy9-RvCYbxFOGpRzT56Heh712EqhhO142dDpDORvvOorzEjZ7d2xLwxF6wxEVogLIzMcv4Eemg4NRu8Q=w1440-h900-v0

433a5bd0-0ca4-4ad3-91a2-26b64bcfd935

F. Zhang, V. Tourre & G. Moreau / Applying level-of-detail and perceptive effects to 3D urban semantics visualization

separately have a base. Output is generated by multiplying the base with function results as showed in equation (1):

Out put(s, t,c,r) = f unc(Ds,Do,LoI)∗base(s, t,c,r) (1)

This equation illustrates that the matching between per-ceptive factors and processing functions is a multi-mapping relationship. For a perceptive factor, one or more process-ing functions can be applied. And for a processing function, it can work on one or more perceptive factors at the same time. So the final output is the result of all the processing functions used multiplying all the perceptive factors chosen.

It is hard to find references on how researchers construct processing functions, so we firstly conducted a parameter study. In brief, the goal is to find a [u, v] pair where u value is used to control the maximum value of the function and v is to control the changing speed of the function. Finally we found that [1.0, 0.31] is the best pair and a table for [x, function value] is gained. This value pair is used for the con-struction of all the functions. In implementation, this value pair can be modified easily by users if they want.

Typical functions in object space:

Object space linear function: the aim is to change the function value with a continuous linear effect.

OL(x,u,v) = { u

v x : x > 0,x ≤ v u : x > v

Object space sinusoid function: a sinusoid curve with the value pair is established by equation (3).

OS(x,u,v)= {

1− u 2 (sin( πx

2 )−1) : x > 0,x ≤ v 1−u : x > v

As x grows in X-axis, the user gets closer to the object in 3D environment. The purpose of these two functions is to decrease the function value as x grows.

Typical functions in screen space:

Screen space linear function: the default focus is the screen center and Sw is the screen width. Users can click a point on screen to set it as the current focus. Semantics near the focus point will get a bigger function value.

SL(Ds,Sw) = 1− 2∗Ds

Screen space fisheye function: a function that ensures se-mantics in the center part of screen is amplified and the other part is decreased so as to achieve a fisheye effect.

SFE(Ds,Sw) =

 1 : Ds ≤ Sw

4 0.9 : Sw

4 < Ds,Ds ≤ 3∗Sw 8

0.6 : Ds > 3∗Sw

Besides these four functions, we have LoI function, Ob-ject space constant piecewise function, Object space contin-uous piecewise function and Object space ordering function, which are difficult to put in an equation as those illustrated. There can be more functions for special purposes.

#### 4.3. Result

The implementation is achieved on an Apple MacBook Pro with a screen resolution of 1440*900 and with the open source 3D graphics toolkit-OpenSceneGraph 3.2.1.

LoD for urban semantics:

The distance from the urban model center to current camera position (CameraModelDistance) can be calculated, which is used to control the transition of semantic LoDs. There are three semantic levels, hence we get three scopes in the same parameter study, which are far, near, very near. During the far scope, semantics at E-LoD0 are displayed. Similarly E-LoD1 contents are visualized in near scope and semantics at E-LoD2 are displayed in very near scope.

In Figure 3, the left part is the visualization result for E-LoD0. Detailed information is displayed as HUD (Head-up display) texts on screen. In the middle is the visualization for E-LoD1. On the right is the visualization for E-LoD2 seman-tics, which is the original camera position for later compar-isons. The default information visualized for each semantic level is the event name. Detailed information can be gained through clicking on the name as showed in the middle part.

Perceptive effects for urban semantics:

Figure 4: OL & OS functions applied at original position.

Here we choose text size to compare the performances of processing functions based on E-LoD2 semantics. In Figure 4, the upper part is the result of applying OL function, at the

c© The Eurographics Association 2014.

https://lh3.googleusercontent.com/notebooklm/AG60hOotdShT-piCDzXwS04gbCkkvD-hhmlr-dfeg-xCRNEvLAe45H_cd9IiGruwZZEMEoOThu-5rge6WjkXgodqpZqYIql2hVpzDeew6brm4J7A10bt4YZ8tJkQjioKbJgauN6f1kzYDA=w1440-h900-v0

8cc0ccc9-31e2-4861-9fc0-4dbd0b7a9643

https://lh3.googleusercontent.com/notebooklm/AG60hOr0a3hTnGdqky87ptuoPkH1hCB9jjJyQOuJGqp8VVXCLAP-fwFs0pou8uMImasi79w2fq8q4z540oJS5AjMZXqu7ssfdnE9_w0o7pdxisINoeppIAictgrPVbhijOpzwEacyzHH=w1440-h900-v0

228f49af-0059-47e0-9d5a-071f3d7d9a0b

https://lh3.googleusercontent.com/notebooklm/AG60hOqV7v2ZgbNQ2MtxvddD5iMdlY8ShwnCqhmXNwdAZJTl_w8vgJq-Kv6w_gs-ftPi4rRjvEDez5hYApsK7g7NlCcOhh1nE3xmrIr8fZzQKdaQssPXPFdrLEnuTr7DtyEEHAsrtHsgHg=w1440-h900-v0

e15a2f66-9d5d-45e9-a407-8151df7da5aa

https://lh3.googleusercontent.com/notebooklm/AG60hOomVf-TmYHP-LXOhTakhHqKEbxWWNu1k3FVtxLuEdqEgBIpTqsSwQcgDLPWpnIoOHNKLt6yqwlAJrGtVNRrhqzqIuw1YbnKhm8RUjnn1f4Qyz34ogORevrVt0ZIIass3so453GqvQ=w1440-h900-v0

0c1eb91a-ff34-467b-897d-cfb574899172

https://lh3.googleusercontent.com/notebooklm/AG60hOqeg2NZRnrAxTRP_23Q_3hi9mVRskGNNvUoTxudgJByPL1N00ylioGEUG_6zZnxY-4_iTDxnxGcuDB4M8AQ9-lAIkxSbiWpkyYjJCrQN7j6bDcOzsrPJHzHAdCtOlp4FpbA-DQqtA=w1440-h900-v0

0e6a2110-1d89-4145-b962-ab549ddd2ff3

https://lh3.googleusercontent.com/notebooklm/AG60hOoIOuDUMax5WUSAe7cbY84_bOJ7KIM0B5FBliH24BgIKXGr5zpu4AOmvr74SQOzh-iZ-NcYxGwfco18OAFVbS2ssa3ksvKk3NUxGmVtw0ogMUH8j6pM-NdqqpDYTaoiKFCULvUZqg=w1440-h900-v0

8e773a6b-f4cf-4669-8650-2c20d80d924f

https://lh3.googleusercontent.com/notebooklm/AG60hOqKjywQNzH1q_qRgDakb1CkDepbfYL0e4MQPFdzpzilMMbSM_O7w7_UtfxOQVlv-UMW7OQUHLYFIVWWpzJvgMorOoPjWhxHaFut3n0DsAoWVOb8muw6qPSdurxWh67hKcn2Pxzx9w=w1440-h900-v0

843891ce-1ba7-47f7-a1aa-e74fc6f41958

F. Zhang, V. Tourre & G. Moreau / Applying level-of-detail and perceptive effects to 3D urban semantics visualization

Figure 3: Visualization of semantics E-LoDs in 3D urban environment.

same camera position as the original one. We can see from the original picture that there are some occlusions among texts with default size (size base).

While after applying the OL function, occlusions de-crease. The lower one is the result of applying OS function. Since these two functions have the same [u, v] value, there is no big visual difference between them at this camera posi-tion. However, as we approach nearer into the scene in Fig-ure 5, it is obvious in the result that from this camera posi-tion, the text size in front of the scene with OS function is bigger than those with OL function ( e.g. Eden a Ouest).

Figure 5: OL & OS functions applied at a near position.

Then in Figure 6 is the SFE function result. It clearly il-lustrates the fisheye effect as semantics in the eye zone is en-

Figure 6: SFE function applied.

larged while the urban environment remains the same, which will not distract user’s attention from semantics.

Figure 7: SL function applied.

Figure 7 is the result of SL function result. Semantics in screen center is of the biggest interest to users. The differ-ence between it and SFE function is that the transition in SL function is smooth while SFE function offers an abrupt effect at the boundary of eye zone.

Finally in Figure 8 is the result of LoI function.

c© The Eurographics Association 2014.

https://lh3.googleusercontent.com/notebooklm/AG60hOqyIhnGwY9lqWsbyYni-8aLiZEbTgteTrU81o-g9vshNP5zdGjAAs2UDoqqwzTRzGexry-jC-vyfefBzPwKBmoE27-2Z1u8fBPD2R4XkezdN_GW_IJyKzbUyFzSJncGoJ0Ce5wtmA=w1440-h900-v0

48a7a8e3-0912-46a0-98ff-6611d9d54182

F. Zhang, V. Tourre & G. Moreau / Applying level-of-detail and perceptive effects to 3D urban semantics visualization

Figure 8: LoI function applied.

### 5. Discussion and future work

Currently we have not yet found any metric to measure the performances of processing functions, hence the evaluation is done by visual comparison. Two contributions are:

Realizing information LoD in 3D urban environment. Previous works were achieved in 2D environment.

Applying perceptive effects to 3D texts visualization at a global view, rather than just dealing with a single text. Here the perception effects are only applied to semantics without geometric deformations of urban environment, which reduces distractions for users and enables users to put their focus and interest on semantics information.

This kind of visualization result is helpful for users when the festival takes place in a city. Query equipment can be placed around the event spot and on-line query should also be available. Results put in this paper are static figures, which can not demonstrate the user interaction part. Actually the user can rotate, zoom in and room out the visualization scene to find information which interests him most. 5

As for the implementation of this work, more perceptive factors are expected to be added, such as the text font. And we hope to have more processing functions to diversify per-ceptive effects. Then concerning the evaluation, it has to be finished, either with a task-given user test or with an effec-tive metric, to prove that adding perceptive effects to seman-tics visualization is worth doing. Finally visualizing relation-ships among semantic items is another future task.

## References

[Bri13] BRINIS S.: Creating a structured corpus of urban and ar-chitectural data for a 3D visualization application. Master thesis, École nationale supérieure d’architecture de Nantes, 2013. 3

[CWK∗07] CHANG R., WESSEL G., KOSARA R., SAUDA E., RIBARSKY W.: Legible cities: Focus-dependent multi-resolution

5 All figures and a demo are available at ifzhang.blogspot.fr

visualization of urban relationships. IEEE Transactions on Visu-alization and Computer Graphics 13, 6 (2007), 1169–1175. 2

[EF10] ELMQVIST N., FEKETE J. D.: Hierarchical aggregation for information visualization: Overview, techniques, and design guidelines. IEEE Transactions on Visualization and Computer Graphics 16, 3 (2010), 439–454. 2

[Fur86] FURNAS G.: Generalized fisheye views. In SIGCHI Con-ference on Human Factors in Computing Systems (1986), pp. 16– 23. 2

[HMM12] HE S., MOREAU G., MARTIN J.: Footprint-based generalization of 3d building groups at medium level of detail for multi-scale urban visualization. International Journal on Ad-vances in Software 5, 3&4 (2012), 377–387. 3

[Jam76] JAMES H.: Hierarchical geometric models for visible surface algorithms. Communications of the ACM 19, 10 (1976), 547–554. 2

[KHG03] KOSARA R., HAUSER H., GRESH D. L.: An inter-action view on information visualization. In EUROGRAPHICS, State of the art report (2003). 1

[KN12] KOLBE T., NAGEL C.: Open geospatial consortium ogc city geography markup language ( citygml ) encoding standard. Open Geospatial Consortium (2012). 2

[MTW∗12] MOERE A. V., TOMITSCH M., WIMMER C., CHRISTOPH B., GRECHENIG T.: Evaluating the effect of style in information visualization. IEEE Transactions on Visualization and Computer Graphics 18, 12 (2012), 2739–2748. 2

[PCS12] PINA J. L., CEREZO E., SERON F.: Semantic visual-ization of 3d urban environments. Multimedia Tools Appl. 59, 2 (2012), 505–521. 1

[PKB05] POLYS N. F., KIM S., BOWMAN D. A.: Effects of in-formation layout, screen size, and field of view on user perfor-mance. In Information-Rich Virtual Environments, Proceedings of ACM Symposium on Virtual Reality Software and Technology (2005), pp. 46–55. 2

[PZG∗13] PAN B., ZHAO Y., GUO X., CHEN X., CHEN W., PENG Q.: Perception-motivated visualization for 3d city scenes. The Visual Computer: International Journal of Computer Graph-ics 29, 4 (2013), 277–286. 3

[RL95] RHEINGANS P., LANDRETH C.: Perceptual principles for effective visualizations. Perceptual Issues in Visualization (1995), 59–74. 2

[Shn96] SHNEIDERMAN B.: The eyes have it: A task by data type taxonomy for information visualizations. In IEEE Sympo-sium on Visual Languages (1996), IEEE Computer Society Press, pp. 336–343. 2

[SWRG02] SUH B., WOODRUFF A., ROSENHOLTZ R., GLASS A.: Popout prism: Adding perceptual principles to overview de-tail document interfaces. In ACM Conference on Human Factors in Computing Systems (CHI 2002) (2002), ACM Press, pp. 251– 258. 2

[War04] WARE C.: Information Visualization - Perception for De-sign. Morgan Kaufmann, 2004. 2

[ZHRT08] ZILLNER S., HAUER T., ROGULIN D., TSYMBAL A.: Semantic visualization of patient information. In 21st IEEE Inter-national Symposium on Computer-Based Medical Systems (Jy-vaskyla, 2008), pp. 296–301. 2

[ZTM13] ZHANG F., TOURRE V., MOREAU G.: A general strat-egy for semantic levels of detail visualization in urban environ-ment. In Eurographics workshop on urban data modelling and visualization (2013), pp. 33–36. 1, 2

c© The Eurographics Association 2014.

