---
sourceFile: "Accentuating focus maps via partial schematization - Pure"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:08.917Z"
---

# Accentuating focus maps via partial schematization - Pure

16b93c76-6b0f-4c0e-ba39-739b22db1991

Accentuating focus maps via partial schematization - Pure

8b0dc7da-55f5-4f3d-9ec3-810c0c1b5da6

https://pure.tue.nl/ws/files/90847331/focusmaps.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOqREeWLbmYMIjvZy8_jgchfMKzWy3m7EfH9CP3mxEjtGJqoOTngMk72htCpkwRKWjvf_cHIegdK84D0hCs6RRLTzrm4dUM0dqlkSZaLfns_9LZ0m1PdR0WE6GrC2t0nh-BO32oiKA=w779-h212-v0

65899335-1c7b-46ed-bf1a-8401cd951499

## Accentuating focus maps via partial schematization

Citation for published version (APA): van Dijk, T., van Goethem, A. I., Haunert, J. H., Meulemans, W., & Speckmann, B. (2013). Accentuating focus maps via partial schematization. In 21st ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS) (pp. 418-421). Association for Computing Machinery, Inc.. https://doi.org/10.1145/2525314.2525452

DOI: 10.1145/2525314.2525452

Document status and date: Published: 01/01/2013

Document Version: Accepted manuscript including changes made at the peer-review stage

Please check the document version of this publication:

A submitted manuscript is the version of the article upon submission and before peer-review. There can be important differences between the submitted version and the official published version of record. People interested in the research are advised to contact the author for the final version of the publication, or visit the DOI to the publisher's website.

The final author version and the galley proof are versions of the publication after peer review.

The final published version features the final layout of the paper including the volume, issue and page numbers. Link to publication

General rights Copyright and moral rights for the publications made accessible in the public portal are retained by the authors and/or other copyright owners and it is a condition of accessing publications that users recognise and abide by the legal requirements associated with these rights.

• Users may download and print one copy of any publication from the public portal for the purpose of private study or research.             • You may not further distribute the material or use it for any profit-making activity or commercial gain             • You may freely distribute the URL identifying the publication in the public portal.
If the publication is distributed under the terms of Article 25fa of the Dutch Copyright Act, indicated by the “Taverne” license above, please follow below link for the End User Agreement: www.tue.nl/taverne

Take down policy If you believe that this document breaches copyright please contact us at: openaccess@tue.nl providing details and we will investigate your claim.

## Accentuating Focus Maps via Partial Schematization

Thomas van Dijk Universität Würzburg, Germany

thomas.van.dijk@uni-wuerzburg.de

Arthur van Goethem TU Eindhoven, the Netherlands

a.i.v.goethem@tue.nl

Jan-Henrik Haunert Universität Würzburg, Germany

jan.haunert@uni-wuerzburg.de

Wouter Meulemans Bettina Speckmann TU Eindhoven, the Netherlands

[w.meulemans|b.speckmann]@tue.nl

ABSTRACT We present an algorithm for schematized focus maps. Focus maps integrate a high detailed, enlarged focus region contin-uously in a given base map. Recent methods integrate both with such low distortion that the focus region becomes hard to identify. We combine focus maps with partial schematiza-tion to display distortion of the context and to emphasize the focus region. Schematization visually conveys geographical accuracy, while not increasing map complexity. We extend the focus-map algorithm to incorporate geometric proxim-ity relationships and show how to combine focus maps with schematization in order to cater to different use cases.

Categories and Subject Descriptors: H.4 [Information Systems Applications]: Geographic Information Systems I.3.5 [Computer Graphics] Computational Geometry and Ob-ject Modeling

General Terms: Algorithms

Keywords: Focus maps, schematization, information visu-alization

########### 1. INTRODUCTION Focus maps are an integrated alternative to map insets (see Fig. 1(a)). They continuously combine an enlarged and de-tailed focus region with context information on a smaller scale. A standard approach is the fish-eye view which uses a rim of “glue” that is heavily distorted (Fig. 1(b)). It is com-paratively easy to identify the focus region in such maps, since the distortions are rather obvious. A recent method by Haunert and Sering [8] significantly reduces the distor-tion, but it becomes difficult to separate the focus region from the context without additional visual cues (Fig. 1(c)).

Contributions. We combine focus maps and cartographic schematization to obtain schematized focus maps. Such maps provide high detail within the focus region but schematize the context, hence drawing the attention towards the focus region (Fig. 1(d-e)). We use the geometry of the map to steer the focus of the user; this has multiple advantages.

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the Owner/Author. Copyright is held by the owner/author(s). SIGSPATIAL’13, Nov 05-08 2013, Orlando, FL, USA ACM 978-1-4503-2521-9/13/11. http://dx.doi.org/10.1145/2525314.2525452

First, the schematized geometry visually conveys that the context is not geographically accurate. Second, it does not require additional geometric elements that would otherwise increase the map’s complexity. Third, it allows color to be used more effectively to convey other (thematic) elements.

In Section 2 we extend the focus-map algorithm by Haunert and Sering [8]. Specifically, we show how to augment a given map with edges to control its rigidity and how to increase the distance between any two map objects. In Section 3 we extend the framework by Van Goethem et al. [5] to allow for partial schematization. In Section 4, we combine these methods and showcase our results with various use cases.

Related work. Fish-eye projections are well-known in cartography and received renewed interest with the advent of cartography on small mobile devices [6, 14]. However, since fish-eye projections usually cause large distortion at the boundary of the focus region, Haunert and Sering [8] in-troduced an optimization-based graph-layout method that minimizes distortion. This method is related to others that manipulate network maps, for example, to generate route sketches [1, 12], destination maps [9], or metro maps [13].

To emphasize the focus, the level of detail of the context can be reduced [15]. Beyond a reduction of the level of detail, however, schematization enforces a certain design scheme (we use circular arcs in this paper). Various methods exist for straight-line schematization of territorial outlines [2, 3, 10]. The method by Buchin et al. [2], which is iterative, could be adapted to support partial schematization.

######### 2. FOCUS MAPS The method of Haunert and Sering [8] distorts a subdivision S, i.e. a planar straight-line embedding of a graph in R2. It consists of a vertex set V and an edge set E. For each u ∈ V , Xu and Yu are the input coordinates and there are three variables: the output coordinates xu, yu and the scale factor su locally valid in u. Constraints ensure that the bounding box of the input contains the output subdivision and a user-selected focus region is enlarged by a user-set factor. Subject to these constraints, the method minimizes∑

∑ v∈Adj(u)

( su ·Xu,v − xu,v

)2 + ( su · Yu,v − yu,v

u,v + Y 2 u,v

where Xu,v = Xv − Xu, Yu,v = Yv − Yu, xu,v = xv − xu, yu,v = yv − yu, and Adj(u) = {v ∈ V | {u, v} ∈ E}. Ad-ditional constraints avoid edge crossings. We extend this method to control the rigidity of S (Sect. 2.1) and increase the distance between selected map objects (Sect. 2.2).

(a) Separate context

(b) Fish-eye view [14] (c) Optimized by [8] (d) Increased rigidity (e) Schematic context

Figure 1: Focus maps for France with various options of visualizing context (continental western Europe, subset of input in Fig. 3(a)). (d-e) Results obtained using the techniques presented in this paper.

2.1 Adding bottleneck edges The method of Haunert and Sering [8] requires a connected subdivision as input. It fails if, e.g., islands are present. We suggest augmenting the input graph with edges to render it connected and, furthermore, to control its rigidity.

Let τS(u, v) be the stretch factor of a vertex pair (u, v), i.e., τS(u, v) = dS(u, v)/dEuclid(u, v), where dS(u, v) is the length of the geometrically shortest u-v-path in S (or ∞ if no u-v-path in S exists) and dEuclid(u, v) is the Euclidean distance of u and v. We aim to decrease the stretch factor of S, i.e., τ(S) = maxu,v∈V {τS(u, v)}, to at most a user-set value t ≥ 1, by adding a preferably small number of edges.

We use a greedy heuristic that iteratively adds a bottle-neck edge, i.e., an edge connecting a pair of vertices with maximum stretch factor. This can be computed for a subdi-vision with n vertices and m edges in O(mn+n2 logn) time by computing shortest paths for all pairs of vertices [4]. We may improve upon this as follows. Only in the first iteration we need to compute shortest paths for all pairs of vertices if we maintain the graph distances in a matrix D. If we add a bottleneck edge, we can update D in O(n2) time, which implies that we reduce the runtime of any further iteration to O(n2). Furthermore, we contract (most of the) vertices of degree two (Fig. 2). The reduced subdivision S∗ allows us to compute shortest paths in S more efficiently. The last improvement, a heuristic, is to test only a small number of candidate pairs to find a pair with the maximum stretch fac-tor tmax. If tmax ≥ t, we connect the pair, update S∗, and continue. Else, we stop adding edges and return the sub-division S. In our implementation, we define a candidate pair for each edge of a constrained Delaunay triangulation (CDT) of V with a constrained edge for each edge in E.

When we applied our algorithm to the subdivision EUR in Fig. 3(a) with t := 4, it added 159 edges and needed 2.9 seconds on a Windows PC with 3.0 GB of RAM and a 3.0 GHz Intel dual-core CPU; with t := 3, it added 262 edges and needed 8.8 seconds.

Figure 3(b) shows the result of enlarging Britain after augmenting the input subdivision EUR using t := 4. Ger-many and France get distorted severely. Using all edges of a

Figure 2: Contraction of degree-two vertices.

(a) Subdivision EUR having 1957 vertices and 1993 edges

(b) Britain scaled by 2.5; with bottleneck edges (t = 4)

(c) Britain scaled by 2.5; with CDT edges

(d) Britain scaled by 2.5; with CDT edges (land) and bottleneck edges (sea, t = 4)

Figure 3: Focus maps generated using different methods of graph augmentation. We only forbid crossings among the original edges (black), as we assume that the additional edges (gray) will not be shown to the user.

CDT, however, we still observe severe distortions in France (Fig. 3(c)). To keep the characteristic shapes of countries, we suggest using two different thresholds, tin and tout, for the inner faces of the input subdivision and its outer face, respectively. We used tin := 1.0 and tout := 4.0, meaning that, after adding edges to S to ensure τ(S) ≤ 4.0, we add all edges of a CDT to the inner faces to preserve the shapes of countries. This allows us to enlarge Britain with only a small distortion of other countries (Fig. 3(d)).

https://lh3.googleusercontent.com/notebooklm/AG60hOqzWHTIQuJXPqUz4r_nSR2I7fYC6v95Aldm_eCrVIZAHrQakOQd_C1LmjqcymBJ1ui4ijMBJqyy7M3oMekZMyjvwuIf9UvvFu-B8P5PbbkUrnSr0kqW0nv30M82gvDBbTe24zl7tg=w819-h810-v0

80cfa8d4-7396-4622-867c-47ed828a4155

(a) (b) (c)

Figure 4: Widening the English Channel.

2.2 Widening bottlenecks Usually, two objects in a topographic map must have at least a certain minimum distance; this can be ensured by displace-ment [7]. In thematic maps, we may want to increase the distance between two objects to make space for additional content, e.g., a label or the flow lines in a flow map.

Consider the map in Fig. 4(a), which displays the coun-tries adjoining the English Channel, and assume that we want to widen the channel to represent vessel traffic with a fat line. This can be achieved by defining a lower bound on the scale factors su and sv for each edge {u, v} ∈ E con-necting Britain with France; see Fig. 4(b). The channel has been widened, but, as an undesirable side effect, also the southeast of Britain (Kent) has been vastly enlarged.

To avoid this effect, we choose a different approach. We introduce a variable se ≥ smin(e) for each edge e = {u, v} that we want to enlarge, where smin(e) is the minimally re-quired scale factor. Then, to control e with se, we add(

(se ·Xu,v − xu,v)2 + (se · Yu,v − yu,v)2 ) / ( X2

u,v + Y 2 u,v

) to our objective function. The scale factors su or sv should not control e, thus when measuring the distortion with Equa-tion (1) we do not consider v and u to be neighbors. How-ever, we add the constraint su = sv. Thereby the scale is propagated over the network. As a result, both France and Britain are only marginally distorted, see Fig. 4(c).

############### 3. SCHEMATIZATION Van Goethem et al. [5] introduced a framework for curved schematization. When applying their framework with area-preserving arcs and the Fréchet distance, a vertex-restricted algorithm entails. This algorithm schematizes the input up to a given error bound. It assumes as input a subdivision, e.g., an output of the focus-map algorithm. The algorithm produces a topologically equivalent subdivision that uses cir-cular arcs instead of straight lines. Thus, regions correspond one-to-one; each region has the same neighbors; and there are no intersections between the arcs of the output (assum-ing the input has none). Each circular arc has a Fréchet distance [11] of at most ε to the input polyline it represents (a chain in the terminology of [5]). The framework pro-duces a subdivision that has a minimal number of circular arcs. Below, we describe how we extend this algorithm to support partial schematization.

Weights. We assume that each vertex has been given a weight to indicate its importance: this importance depends on the use case as described in Section 4. The weight is at

least one and may be infinite. A vertex of infinite weight is always required to be in the output.

To obtain a partial schematization, we need to locally de-fine different values of the error margin ε. Vertices with a high weight should be less schematized and, hence, have a lower error margin. We observe that in the described frame-work [5] an arc can only replace a chain if its score (Fréchet distance) is at most ε. Instead of locally decreasing ε, we can increase the score of a replacement arc according to the local weight. To do so, we multiply the score by the average weight of the vertices in the chain.

The area-preserving circular arc of a single edge (a chain of 2 vertices) is the edge itself; the Fréchet distance is exactly zero. Hence, the original framework is guaranteed to yield a solution. When vertices can have an infinite weight this property no longer holds, as even arcs fitted to edges can have an infinite score. To solve this issue we define the product of zero and infinity to be zero. This ensures that the edges connecting consecutive vertices can always be used for any ε value. As a consequence the framework always has a solution regardless of the weighting scheme.

##### 4. SCHEMATIZED FOCUS MAPS Schematization and focus maps should be combined differ-ently depending on the purpose of the final map. In this section, we discuss various combinations based on use cases.

Context for location. When users are not familiar with the shape of an area it is useful to include context to guide them in recognizing and localizing a map. Examples of these types of maps are often found in tourist brochures. We draw attention to the focus region by schematizing the context, weighting all vertices with weight 1 except those in the focus region which have weight infinity. Fig. 5 gives an example for country outlines. By enlarging the focus region we re-move the need for a map inset and lower the visual clutter introduced in the map. The context helps locate the focus region while its schematization emphasizes the focus region.

Interaction with context. Information may also be dis-played that is related to the immediate vicinity, e.g. a the-matic map depicting country export. By gradually increas-ing the schematization further away from the focus region, we actively draw the attention of the user to the focus re-gion. In contrast to the previous approach this purposefully maintains the relationship with the surroundings. We as-sign a weight to each vertex depending on the distance to

Figure 5: Focus map of the Iberian peninsula using a scale-factor of 2. Context helps to locate the map; schematization emphasizes the focus region.

https://lh3.googleusercontent.com/notebooklm/AG60hOohfkS9OWxjW5PAPJMe_8JhudTp03Zpy6LUYAwzqLkKhkL_dTWQJqjufzW8SQd8FyeCGt89SinkNAgfGeXRSuUml9uQElpZ121recueFKgQLz7-BIiFWzHrvKa9Ho5LLuWWtw2-Fw=w692-h856-v0

1983b0fa-c99a-4bd5-b979-6cf2f32327b0

12.56 5.16

Σ Import 32.97 Σ Export 45.35

Figure 6: Focus map of Germany using a scale-factor of 2, and Euclidean distance for schematization. In-teraction with context is visually reinforced by not fully schematizing near the focus region.

the focus region. Fig. 6 shows the result of our algorithm and an example thematic map1. To maintain the relation with neighboring countries while reinforcing the importance of the focus region we gradually increase schematization fur-ther away from Germany. As a consequence the visual focus region extends to include the neighboring countries.

Focus on thematic information. This use case concerns thematic maps. No specific region requires focus, but our method still proves useful. By widening the bottleneck edges we make space for the thematic information. This reduces overlap, increasing the clarity of the final map. To emphasize the main content, we schematize the entire map uniformly. Fig. 7 shows a result mimicking the map on wine-export by Minard2. To allow the thematic information to be displayed, France has been enlarged and the English Channel, Strait of Gibraltar, and Strait of Malacca have been widened.

########## 5. DISCUSSION Schematizing focus maps creates clear and concise maps. However, we did make the following observations. Enlarging the focus region inherently reduces its visual complexity; it may appear to be simplified already. Thus, the level of detail in the input map should fit the enlarged focus scale; not the input scale. Secondly, the effect of schematization may be bound by different factors including expectations, cultural background, or familiarity with the region. If the user is expecting jagged lines, breaking this expectation by schematization has a big impact. Schematizations, as maps, should be tailored to the target audience.

###### 6. REFERENCES [1] M. Agrawala and C. Stolte. Rendering effective route

maps: Improving usability through generalization. In Proc. 28th Conference on Compututer Graphics and interactive techniques, pages 241–249, 2001.

[2] K. Buchin, W. Meulemans, and B. Speckmann. A new method for subdivision simplification with

1http://rwecom.online-report.eu/ factbook/en/marketdata/electricity/grid/ germanyimportandexportofelectricity.html, 06/2013. 2C. Minard, Carte figurative et approximative des quantités de vin français exportés par mer en 1864, 1865

## NORTH AMERICA

## SOUTH AMERICA

Figure 7: Use of focus maps to make space for the-matic information. Schematization removes unnec-essary detail. Roughly based on Minard’s map.

applications to urban-area generalization. In Proc. 19th ACM GIS, pages 261–270, 2011.

[3] S. Cicerone and M. Cermignani. Fast and Simple Approach for Polygon Schematization. In Proc. 12th Int. Conference on Computational Science and Its Applications, LNCS 7333, pages 267–279, 2012.

[4] M. Farshi, P. Giannopoulos, and J. Gudmundsson. Improving the Stretch Factor of a Geometric Network by Edge Augmentation. SIAM Journal on Computing, 38(1):226–240, 2008.

[5] A. van Goethem, W. Meulemans, A. Reimer, H. Haverkort, and B. Speckmann. Topologically Safe Curved Schematisation. The Cartographic J., 50(3):276–285, 2013.

[6] L. Harrie, L. T. Sarjakoski, and L. Lehto. A Mapping Function for Variable-Scale Maps in Small-Display Cartography. J. Geospatial Eng., 4(2):111–123, 2002.

[7] L. Harrie and T. Sarjakoski. Simultaneous Graphic Generalization of Vector Data Sets. Geoinformatica, 6(3):233–261, 2002.

[8] J.-H. Haunert and L. Sering. Drawing Road Networks with Focus Regions. IEEE Trans. on Visualization and Computer Graphics, 17(12):2555–2562, 2011.

[9] J. Kopf, M. Agrawala, D. Bargeron, D. Salesin, and M. Cohen. Automatic generation of destination maps. ACM Trans. on Graphics, 29(6):158:1–158:12, 2010.

[10] A. Reimer and W. Meulemans. Parallelity in chorematic territorial outlines. In Proc. 14th Workshop on Generalisation and Multiple Representation, 2011.

[11] G. Rote. Computing the Fréchet distance between piecewise smooth curves. Computational Geometry: Theory and Applications, 37(3):162–174, 2007.

[12] F. Schmid. Knowledge-based wayfinding maps for small display cartography. Journal of Location Based Services, 2(1):51–83, 2008.

[13] Y.-S. Wang and M.-T. Chi. Focus+context metro maps. IEEE Trans. on Visualization and Computer Graphics, 17(12):2528–2535, 2011.

[14] D. Yamamoto, S. Ozeki, and N. Takahashi. Focus+glue+context: an improved fisheye approach for web map services. In Proc. 17th ACM GIS, pages 101–110, 2009.

[15] A. Zipf and K.-F. Richter. Using focus maps to ease map reading: Developing smart applications for mobile devices. Künstliche Intelligenz, 02(4):35–37, 2002.

