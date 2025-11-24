---
sourceFile: "Binary space partitioning - Wikipedia"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:10.212Z"
---

# Binary space partitioning - Wikipedia

61761074-6b8f-474a-a947-10d63ef1961d

Binary space partitioning - Wikipedia

b89f989e-c262-4302-8537-c413296c1386

https://en.wikipedia.org/wiki/Binary_space_partitioning

## Jump to content

## Main menu    Navigation

## Current events

## Random article

## About Wikipedia

## Contribute

## Learn to edit

## Community portal

## Recent changes

## Upload file

## Special pages

## Create account

## Create account

## Pages for logged out editors

https://en.wikipedia.org/wiki/Help:Introduction

## Contributions

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

1   History

https://en.wikipedia.org/wiki/Help:Introduction

2   Overview

https://en.wikipedia.org/wiki/Help:Introduction

3   Generation

https://en.wikipedia.org/wiki/Help:Introduction

4   Traversal

https://en.wikipedia.org/wiki/Help:Introduction

5   Application

https://en.wikipedia.org/wiki/Help:Introduction

6   See also

https://en.wikipedia.org/wiki/Help:Introduction

7   References

https://en.wikipedia.org/wiki/Help:Introduction

8   Additional references

https://en.wikipedia.org/wiki/Help:Introduction

9   External links

https://en.wikipedia.org/wiki/Help:Introduction

## Binary space partitioning

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

## Bahasa Indonesia

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

Norsk bokmål

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

Српски / srpski

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

## Tools    Actions

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

## View history

https://en.wikipedia.org/wiki/Help:Introduction

## What links here

https://en.wikipedia.org/wiki/Help:Introduction

## Related changes

https://en.wikipedia.org/wiki/Help:Introduction

## Upload file

https://en.wikipedia.org/wiki/Help:Introduction

## Permanent link

https://en.wikipedia.org/wiki/Help:Introduction

## Page information

https://en.wikipedia.org/wiki/Help:Introduction

## Cite this page

https://en.wikipedia.org/wiki/Help:Introduction

## Get shortened URL

https://en.wikipedia.org/wiki/Help:Introduction

## Download QR code

https://en.wikipedia.org/wiki/Help:Introduction

Print/export

## Download as PDF

https://en.wikipedia.org/wiki/Help:Introduction

## Printable version

https://en.wikipedia.org/wiki/Help:Introduction

## In other projects

## Wikidata item

https://en.wikipedia.org/wiki/Help:Introduction

From Wikipedia, the free encyclopedia   Method for recursively subdividing a space into two subsets using hyperplanes

## This article

needs additional citations for

verification

https://en.wikipedia.org/wiki/Wikipedia:Verifiability

.  Please help

improve this article

https://en.wikipedia.org/wiki/Special:EditPage/Binary_space_partitioning

adding citations to reliable sources

https://en.wikipedia.org/wiki/Help:Referencing_for_beginners

. Unsourced material may be challenged and removed.

Find sources:

"Binary space partitioning"

https://www.google.com/search?tbm=nws&q=%22Binary+space+partitioning%22+-wikipedia&tbs=ar:1

https://www.google.com/search?tbm=nws&q=%22Binary+space+partitioning%22+-wikipedia&tbs=ar:1

https://www.google.com/search?tbm=nws&q=%22Binary+space+partitioning%22+-wikipedia&tbs=ar:1

https://www.google.com/search?tbm=nws&q=%22Binary+space+partitioning%22+-wikipedia&tbs=ar:1

https://www.google.com/search?tbm=nws&q=%22Binary+space+partitioning%22+-wikipedia&tbs=ar:1

( May 2016 )

## Learn how and when to remove this message

## The process of making a BSP tree

computer science

https://en.wikipedia.org/wiki/Computer_science

binary space partitioning

) is a method for

space partitioning

https://en.wikipedia.org/wiki/Space_partitioning

recursively

https://en.wikipedia.org/wiki/Recursively

subdivides a

## Euclidean space

https://en.wikipedia.org/wiki/Euclidean_space

convex sets

https://en.wikipedia.org/wiki/Convex_set

hyperplanes

https://en.wikipedia.org/wiki/Hyperplane

as partitions. This process of subdividing gives rise to a representation of objects within the space in the form of a

tree data structure

https://en.wikipedia.org/wiki/Tree_(data_structure)

known as a

## Binary space partitioning was developed in the context of

3D computer graphics

https://en.wikipedia.org/wiki/3D_computer_graphics

## The structure of a BSP tree is useful in

https://en.wikipedia.org/wiki/Rendering_(computer_graphics)

because it can efficiently give spatial information about the objects in a scene, such as objects being ordered from front-to-back with respect to a viewer at a given location. Other applications of BSP include: performing

geometrical

https://en.wikipedia.org/wiki/Geometrical

operations with

https://en.wikipedia.org/wiki/Shape

constructive solid geometry

https://en.wikipedia.org/wiki/Constructive_solid_geometry

https://en.wikipedia.org/wiki/Computer-aided_design

collision detection

https://en.wikipedia.org/wiki/Robotics

and 3D video games,

ray tracing

https://en.wikipedia.org/wiki/Ray_tracing_(graphics)

, virtual landscape simulation,

and other applications that involve the handling of complex spatial scenes.

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=1

1969 Schumacker et al.

published a report that described how carefully positioned planes in a virtual environment could be used to accelerate polygon ordering. The technique made use of depth coherence, which states that a polygon on the far side of the plane cannot, in any way, obstruct a closer polygon. This was used in flight simulators made by GE as well as Evans and Sutherland. However, the creation of the polygonal data organization was performed manually by the scene designer.

https://en.wikipedia.org/wiki/Henry_Fuchs

extended Schumacker's idea to the representation of 3D objects in a virtual environment by using planes that lie coincident with polygons to recursively partition the 3D space. This provided a fully automated and algorithmic generation of a hierarchical polygonal data structure known as a Binary Space Partitioning Tree (BSP Tree). The process took place as an off-line preprocessing step that was performed once per environment/object. At run-time, the view-dependent visibility ordering was generated by traversing the tree.

1981 Naylor's Ph.D. thesis

provided a full development of both BSP trees and a graph-theoretic approach using strongly connected components for pre-computing visibility, as well as the connection between the two methods. BSP trees as a dimension-independent spatial search structure were emphasized, with applications to visible surface determination. The thesis also included the first empirical data demonstrating that the size of the tree and the number of new polygons were reasonable (using a model of the Space Shuttle).

https://en.wikipedia.org/wiki/Henry_Fuchs

described a micro-code implementation of the BSP tree algorithm on an Ikonas frame buffer system. This was the first demonstration of real-time visible surface determination using BSP trees.

1987 Thibault and Naylor

described how arbitrary polyhedra may be represented using a BSP tree as opposed to the traditional b-rep (

boundary representation

https://en.wikipedia.org/wiki/Boundary_representation

). This provided a solid representation vs. a surface based-representation. Set operations on polyhedra were described using a tool, enabling

constructive solid geometry

https://en.wikipedia.org/wiki/Constructive_solid_geometry

(CSG) in real-time. This was the forerunner of BSP level design using "

https://en.wikipedia.org/wiki/Brush_(video_games)

", introduced in the Quake editor and picked up in the Unreal Editor.

1990 Naylor, Amanatides, and Thibault

provided an algorithm for merging two BSP trees to form a new BSP tree from the two original trees. This provides many benefits including combining moving objects represented by BSP trees with a static environment (also represented by a BSP tree), very efficient CSG operations on polyhedra, exact collisions detection in O(log n * log n), and proper ordering of transparent surfaces contained in two interpenetrating objects (has been used for an x-ray vision effect).

https://en.wikipedia.org/wiki/Seth_J._Teller

and Séquin

proposed the offline generation of potentially visible sets to accelerate visible surface determination in orthogonal 2D environments.

1991 Gordon and Chen

described an efficient method of performing front-to-back rendering from a BSP tree, rather than the traditional back-to-front approach. They utilized a special data structure to record, efficiently, parts of the screen that have been drawn, and those yet to be rendered. This algorithm, together with the description of BSP trees in the standard computer graphics textbook of the day (

Computer Graphics: Principles and Practice

) was used by

## John Carmack

https://en.wikipedia.org/wiki/John_D._Carmack

in the making of

(video game)

https://en.wikipedia.org/wiki/Doom_(1993_video_game)

https://en.wikipedia.org/wiki/Seth_J._Teller

's Ph.D. thesis

described the efficient generation of potentially visible sets as a pre-processing step to accelerate real-time visible surface determination in arbitrary 3D polygonal environments. This was used in

and contributed significantly to that game's performance.

1993 Naylor

answered the question of what characterizes a good BSP tree. He used expected case models (rather than worst-case analysis) to mathematically measure the expected cost of searching a tree and used this measure to build good BSP trees. Intuitively, the tree represents an object in a multi-resolution fashion (more exactly, as a tree of approximations). Parallels with Huffman codes and probabilistic

binary search

https://en.wikipedia.org/wiki/Binary_search

trees are drawn.

1993 Hayder Radha's Ph.D. thesis

described (natural) image representation methods using BSP trees. This includes the development of an optimal BSP-tree construction framework for any arbitrary input image. This framework is based on a new image transform, known as the Least-Square-Error (LSE) Partitioning Line (LPE) transform. H. Radha's thesis also developed an optimal rate-distortion (RD)

image compression

https://en.wikipedia.org/wiki/Image_compression

framework and image manipulation approaches using BSP trees.

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=2

## An example of a recursive binary space partitioning

https://en.wikipedia.org/wiki/Quadtree

for a 2D index

## Binary space partitioning is a generic process of

recursively

https://en.wikipedia.org/wiki/Recursion

dividing a scene into two using

hyperplanes

https://en.wikipedia.org/wiki/Hyperplanes

https://en.wikipedia.org/wiki/Hyperplanes

until the partitioning satisfies one or more requirements. It can be seen as a generalization of other spatial tree structures such as

https://en.wikipedia.org/wiki/K-d_tree

https://en.wikipedia.org/wiki/Quadtree

, one where hyperplanes that partition the space may have any orientation, rather than being aligned with the coordinate axes as they are in

-d trees or quadtrees. When used in computer graphics to render scenes composed of planar

https://en.wikipedia.org/wiki/Polygon_mesh

, the partitioning planes are frequently chosen to coincide with the planes defined by polygons in the scene.

The specific choice of partitioning plane and criterion for terminating the partitioning process varies depending on the purpose of the BSP tree. For example, in computer graphics rendering, the scene is divided until each node of the BSP tree contains only polygons that can be rendered in arbitrary order. When

back-face culling

https://en.wikipedia.org/wiki/Back-face_culling

is used, each node, therefore, contains a convex set of polygons, whereas when rendering double-sided polygons, each node of the BSP tree contains only polygons in a single plane. In collision detection or ray tracing, a scene may be divided up into

https://en.wikipedia.org/wiki/Geometric_primitive

on which collision or ray intersection tests are straightforward.

Binary space partitioning arose from computer graphics needing to rapidly draw three-dimensional scenes composed of polygons. A simple way to draw such scenes is the

painter's algorithm

https://en.wikipedia.org/wiki/Painter%27s_algorithm

, which produces polygons in order of distance from the viewer, back to front, painting over the background and previous polygons with each closer object. This approach has two disadvantages: the time required to sort polygons in back-to-front order, and the possibility of errors in overlapping polygons. Fuchs and co-authors

showed that constructing a BSP tree solved both of these problems by providing a rapid method of sorting polygons with respect to a given viewpoint (linear in the number of polygons in the scene) and by subdividing overlapping polygons to avoid errors that can occur with the painter's algorithm. A disadvantage of binary space partitioning is that generating a BSP tree can be time-consuming. Typically, it is therefore performed once on static geometry, as a pre-calculation step, prior to rendering or other real-time operations on a scene. The expense of constructing a BSP tree makes it difficult and inefficient to directly implement moving objects into a tree.

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=3

The canonical use of a BSP tree is for rendering polygons (that are double-sided, that is, without

back-face culling

https://en.wikipedia.org/wiki/Back-face_culling

) with the painter's algorithm. Each polygon is designated with a front side and a backside which could be chosen arbitrarily and only affects the structure of the tree but not the required result.

Such a tree is constructed from an unsorted list of all the polygons in a scene. The recursive algorithm for construction of a BSP tree from that list of polygons is:

## Choose a polygon

from the list.

## Make a node

in the BSP tree, and add

to the list of polygons at that node.

For each other polygon in the list:

## If that polygon is wholly in front of the plane containing

, move that polygon to the list of nodes in front of

## If that polygon is wholly behind the plane containing

, move that polygon to the list of nodes behind

## If that polygon is intersected by the plane containing

, split it into two polygons and move them to the respective lists of polygons behind and in front of

## If that polygon lies in the plane containing

, add it to the list of polygons at node

## Apply this algorithm to the list of polygons in front of

## Apply this algorithm to the list of polygons behind

The following diagram illustrates the use of this algorithm in converting a list of lines or polygons into a BSP tree. At each of the eight steps (i.-viii.), the algorithm above is applied to a list of lines, and one new node is added to the tree.

Start with a list of lines, (or in 3D, polygons) making up the scene. In the tree diagrams, lists are denoted by rounded rectangles and nodes in the BSP tree by circles. In the spatial diagram of the lines, the direction chosen to be the 'front' of a line is denoted by an arrow.

Following the steps of the algorithm above,

We choose a line, A, from the list and,...

...add it to a node.

We split the remaining lines in the list into those in front of A (i.e. B2, C2, D2), and those behind (B1, C1, D1).

We first process the lines in front of A (in steps ii–v),...

...followed by those behind (in steps vi–vii).

We now apply the algorithm to the list of lines in front of A (containing B2, C2, D2). We choose a line, B2, add it to a node and split the rest of the list into those lines that are in front of B2 (D2), and those that are behind it (C2, D3).

Choose a line, D2, from the list of lines in front of B2 and A. It is the only line in the list, so after adding it to a node, nothing further needs to be done.

We are done with the lines in front of B2, so consider the lines behind B2 (C2 and D3). Choose one of these (C2), add it to a node, and put the other line in the list (D3) into the list of lines in front of C2.

Now look at the list of lines in front of C2. There is only one line (D3), so add this to a node and continue.

We have now added all of the lines in front of A to the BSP tree, so we now start on the list of lines behind A. Choosing a line (B1) from this list, we add B1 to a node and split the remainder of the list into lines in front of B1 (i.e. D1), and lines behind B1 (i.e. C1).

Processing first the list of lines in front of B1, D1 is the only line in this list, so add this to a node and continue.

Looking next at the list of lines behind B1, the only line in this list is C1, so add this to a node, and the BSP tree is complete.

The final number of polygons or lines in a tree is often larger (sometimes much larger

) than the original list, since lines or polygons that cross the partitioning plane must be split into two. It is desirable to minimize this increase, but also to maintain reasonable

https://en.wikipedia.org/wiki/Binary_tree#Types_of_binary_trees

in the final tree. The choice of which polygon or line is used as a partitioning plane (in step 1 of the algorithm) is therefore important in creating an efficient BSP tree.

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=4

## A BSP tree is

https://en.wikipedia.org/wiki/Tree_traversal

in a linear time, in an order determined by the particular function of the tree. Again using the example of rendering double-sided polygons using the painter's algorithm, to draw a polygon

correctly requires that all polygons behind the plane

lies in must be drawn first, then polygon

, then finally the polygons in front of

. If this drawing order is satisfied for all polygons in a scene, then the entire scene renders in the correct order. This procedure can be implemented by recursively traversing a BSP tree using the following algorithm.

## From a given viewing location

, to render a BSP tree,

If the current node is a leaf node, render the polygons at the current node.

Otherwise, if the viewing location

is in front of the current node:

## Render the child BSP tree containing polygons behind the current node

## Render the polygons at the current node

## Render the child BSP tree containing polygons in front of the current node

Otherwise, if the viewing location

is behind the current node:

## Render the child BSP tree containing polygons in front of the current node

## Render the polygons at the current node

## Render the child BSP tree containing polygons behind the current node

Otherwise, the viewing location

must be exactly on the plane associated with the current node. Then:

## Render the child BSP tree containing polygons in front of the current node

## Render the child BSP tree containing polygons behind the current node

Applying this algorithm recursively to the BSP tree generated above results in the following steps:

The algorithm is first applied to the root node of the tree, node

is in front of node

, so we apply the algorithm first to the child BSP tree containing polygons behind

## This tree has root node

is behind

so first, we apply the algorithm to the child BSP tree containing polygons in front of

## This tree is just the leaf node

, so the polygon

is rendered.

## We then render the polygon

## We then apply the algorithm to the child BSP tree containing polygons behind

## This tree is just the leaf node

, so the polygon

is rendered.

## We then draw the polygons of

We then apply the algorithm to the child BSP tree containing polygons in front of

## This tree has root node

is behind

so first, we apply the algorithm to the child BSP tree containing polygons in front of

## This tree is just the leaf node

, so the polygon

is rendered.

## We then render the polygon

## We then apply the algorithm to the child BSP tree containing polygons behind

## This tree has root node

is in front of

so first, we would apply the algorithm to the child BSP tree containing polygons behind

. There is no such tree, however, so we continue.

## We render the polygon

## We apply the algorithm to the child BSP tree containing polygons in front of

## This tree is just the leaf node

, so the polygon

is rendered.

The tree is traversed in linear time and renders the polygons in a far-to-near ordering (

) suitable for the painter's algorithm.

## Application

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=5

BSP trees are often used by 3D

video games

https://en.wikipedia.org/wiki/Video_game

, particularly

first-person shooters

https://en.wikipedia.org/wiki/First-person_shooter

and those with indoor environments.

## Game engines

https://en.wikipedia.org/wiki/Game_engine

using BSP trees include the

Doom (id Tech 1)

https://en.wikipedia.org/wiki/Doom_engine

Quake (id Tech 2 variant)

https://en.wikipedia.org/wiki/Quake_engine

https://en.wikipedia.org/wiki/GoldSrc

https://en.wikipedia.org/wiki/Source_(game_engine)

engines. In them, BSP trees containing the static geometry of a scene are often used together with a

https://en.wikipedia.org/wiki/Z-buffer

, to correctly merge movable objects such as doors and characters onto the background scene. While binary space partitioning provides a convenient way to store and retrieve spatial information about polygons in a scene, it does not solve the problem of

visible surface determination

https://en.wikipedia.org/wiki/Hidden_surface_determination

. BSP trees have also been applied to image compression.

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

## Chazelle polyhedron

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

## Hierarchical clustering

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=6

, an alternative way to divide

https://en.wikipedia.org/wiki/3D_model

data for efficient rendering.

## Guillotine cutting

https://en.wikipedia.org/wiki/3D_model

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=7

Schumacker, R.A.; Brand, B.; Gilliland, M.G.; Sharp, W.H. (1969).

Study for Applying Computer-Generated Images to Visual Simulation

https://books.google.com/books?id=0mtk5MdXJhEC

(Report). U.S. Air Force Human Resources Laboratory. AFHRL-TR-69-14.

Fuchs, Henry; Kedem, Zvi. M; Naylor, Bruce F. (1980).

"On Visible Surface Generation by A Priori Tree Structures"

http://www.cs.unc.edu/~fuchs/publications/VisSurfaceGeneration80.pdf

SIGGRAPH '80 Proceedings of the 7th annual conference on Computer graphics and interactive techniques

. ACM. pp.  124– 133.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1145/965105.807481

https://doi.org/10.1145%2F965105.807481

Thibault, William C.; Naylor, Bruce F. (1987). "Set operations on polyhedra using binary space partitioning trees".

SIGGRAPH '87 Proceedings of the 14th annual conference on Computer graphics and interactive techniques

. ACM. pp.  153– 162.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1145/37402.37421

https://doi.org/10.1145%2F37402.37421

https://doi.org/10.1145%2F37402.37421

Etherington, Thomas R.; Morgan, Fraser J.; O’Sullivan, David (2022).

"Binary space partitioning generates hierarchical and rectilinear neutral landscape models suitable for human-dominated landscapes"

https://doi.org/10.1007%2Fs10980-022-01452-6

## Landscape Ecology

(7):  1761– 1769.

https://en.wikipedia.org/wiki/Bibcode_(identifier)

2022LaEco..37.1761E

https://ui.adsabs.harvard.edu/abs/2022LaEco..37.1761E

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1007/s10980-022-01452-6

https://doi.org/10.1007%2Fs10980-022-01452-6

https://doi.org/10.1007%2Fs10980-022-01452-6

Naylor, Bruce (May 1981).

A Priori Based Techniques for Determining Visibility Priority for 3-D Scenes

(Ph.D. thesis). University of Texas at Dallas . Retrieved  June 5,  2025 .

https://www.proquest.com/openview/94daf3b8677f8ca4567915515efeefac/1?pq-origsite=gscholar&cbl=18750&diss=y

Fuchs, Henry; Abram, Gregory D.; Grant, Eric D. (1983). "Near real-time shaded display of rigid objects".

Proceedings of the 10th annual conference on Computer graphics and interactive techniques

. ACM. pp.  65– 72.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1145/800059.801134

https://doi.org/10.1145%2F800059.801134

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-0-89791-109-2

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/ISBN_(identifier)

Naylor, Bruce; Amanatides, John; Thibault, William (August 1990).

"Merging BSP Trees Yields Polyhedral Set Operations"

https://dl.acm.org/doi/pdf/10.1145/97880.97892

## ACM SIGGRAPH Computer Graphics

(4). Association of Computing Machinery:  115– 124.

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

10.1.1.69.292

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1145/97880.97892

https://doi.org/10.1145%2F97880.97892

. Retrieved  June 5,  2025 .

https://doi.org/10.1145%2F97880.97892

Teller, Seth J.; Séquin, Carlo H. (July 1, 1991).

"Visibility preprocessing for interactive walkthroughs"

https://dl.acm.org/doi/abs/10.1145/127719.122725

## ACM SIGGRAPH Computer Graphics

(4). Association of Computing Machinery:  61– 70 . Retrieved  June 5,  2025 .

Chen, S.; Gordon, D. (October 1991).

"Front-to-Back Display of BSP Trees"

https://www.researchgate.net/publication/3208236_Front-to-back_display_of_BSP_trees

## IEEE Computer Graphics and Applications

(5):  79– 85.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/38.90569

https://doi.org/10.1109%2F38.90569

https://en.wikipedia.org/wiki/S2CID_(identifier)

https://en.wikipedia.org/wiki/S2CID_(identifier)

https://en.wikipedia.org/wiki/S2CID_(identifier)

Teller, Seth (1992).

## Visibility computations in densely occluded polyhedral environments

(Ph.D. thesis). University of California at Berkeley . Retrieved  June 5,  2025 .

https://www.proquest.com/openview/80322259984cf6c676a345676ab1d74a/1?pq-origsite=gscholar&cbl=18750&diss=y

Naylor, Bruce (1993).

"Constructing good partitioning trees"

https://www.researchgate.net/profile/Bruce-Naylor/publication/2492209_Constructing_Good_Partitioning_Trees/links/55cc86be08aea2d9bdce442d/Constructing-Good-Partitioning-Trees.pdf

## Graphics Interface

. Canadian Information Processing Society:  181– 191 . Retrieved  June 5,  2025 .

Radha, Hayder (1993).

## Efficient image representation using binary space partitioning trees

(Ph.D. thesis). Columbia University . Retrieved  June 5,  2025 .

https://www.proquest.com/openview/a80bc19b1374b928afa8844a8ed05ef4/1?pq-origsite=gscholar&cbl=18750&diss=y

Naylor, Bruce (January 2005).

"A Tutorial on Binary Space Partitioning Trees"

https://www.researchgate.net/publication/238348725_A_Tutorial_on_Binary_Space_Partitioning_Trees

## ResearchGate

. Retrieved  July 1,  2025 .

Radha, H.; Vetterli, M.; Leonardi, R. (1996).

"Image compression using binary space partitioning trees"

https://infoscience.epfl.ch/record/33877/files/RadhaVL96.pdf

## IEEE Transactions on Image Processing

(12):  1610– 1624.

https://en.wikipedia.org/wiki/Bibcode_(identifier)

1996ITIP....5.1610R

https://ui.adsabs.harvard.edu/abs/1996ITIP....5.1610R

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/83.544569

https://doi.org/10.1109%2F83.544569

https://en.wikipedia.org/wiki/PMID_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

## Additional references

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=8

Naylor, B. (May 1993).

"Constructing Good Partitioning Trees"

https://www.researchgate.net/publication/2492209

## Graphics Interface

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

10.1.1.16.4432

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

Radha, H.; Leoonardi, R.; Vetterli, M.; Naylor, B. (1991).

"Binary space partitioning tree representation of images"

http://infoscience.epfl.ch/record/33911

## Journal of Visual Communications and Image Processing

(3):  201– 221.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1016/1047-3203(91)90023-9

https://doi.org/10.1016%2F1047-3203%2891%2990023-9

Radha, H.M.S. (1993).

## Efficient Image Representation using Binary Space Partitioning Trees

(PhD). Columbia University.

https://en.wikipedia.org/wiki/OCLC_(identifier)

https://en.wikipedia.org/wiki/OCLC_(identifier)

Radha, H.M.S. (1994). "Efficient image representation using binary space partitioning trees".

## Signal Processing

(2):  174– 181.

https://en.wikipedia.org/wiki/Bibcode_(identifier)

1994SigPr..35..174R

https://ui.adsabs.harvard.edu/abs/1994SigPr..35..174R

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1016/0165-1684(94)90047-7

https://doi.org/10.1016%2F0165-1684%2894%2990047-7

Radha, H.; Vetterli, M.; Leoonardi, R. (December 1996).

"Image compression using binary space partitioning trees"

http://infoscience.epfl.ch/record/33877

## IEEE Transactions on Image Processing

(12):  1610– 24.

https://en.wikipedia.org/wiki/Bibcode_(identifier)

1996ITIP....5.1610R

https://ui.adsabs.harvard.edu/abs/1996ITIP....5.1610R

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/83.544569

https://doi.org/10.1109%2F83.544569

https://en.wikipedia.org/wiki/PMID_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

https://ui.adsabs.harvard.edu/abs/1996ITIP....5.1610R/abstract

https://ui.adsabs.harvard.edu/abs/1996ITIP....5.1610R/abstract

Winter, A.S. (April 1999). "An investigation into real-time 3d polygon rendering using bsp trees".

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

10.1.1.11.9725

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

de Berg, M.

https://en.wikipedia.org/wiki/CiteSeerX_(identifier)

van Kreveld, M.

https://en.wikipedia.org/wiki/Marc_van_Kreveld

Overmars, M.

https://en.wikipedia.org/wiki/Mark_Overmars

Schwarzkopf, O.

https://en.wikipedia.org/wiki/Otfried_Schwarzkopf

(2000). "§12: Binary Space Partitions".

## Computational Geometry

(2nd ed.).

Springer-Verlag

https://en.wikipedia.org/wiki/Springer-Verlag

. pp.  251– 265.

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-3-540-65620-3

https://en.wikipedia.org/wiki/ISBN_(identifier)

.  Describes a randomized Painter's Algorithm..

Ericson, Christer (2005).

"8. BSP Tree Hierarchies"

https://books.google.com/books?id=WGpL6Sk9qNAC&pg=PA350

Real-Time collision detection

. Morgan Kaufmann Series in Interactive 3-D Technology. Morgan Kaufmann. pp.  349– 382.

https://en.wikipedia.org/wiki/ISBN_(identifier)

1-55860-732-3

https://en.wikipedia.org/wiki/ISBN_(identifier)

## External links

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&action=edit&section=9

Naylor, B.F. (2005).

"A Tutorial on Binary Space Partitioning Trees"

https://www.researchgate.net/publication/238348725

## BSP trees presentation

https://www.researchgate.net/publication/238348725

## Another BSP trees presentation

https://www.researchgate.net/publication/238348725

## A Java applet that demonstrates the process of tree generation

https://www.researchgate.net/publication/238348725

## A Master Thesis about BSP generating

https://www.researchgate.net/publication/238348725

BSP Trees: Theory and Implementation

https://www.researchgate.net/publication/238348725

BSP in 3D space

https://www.researchgate.net/publication/238348725

Graphics Gems V: A Walk through BSP Trees

https://www.researchgate.net/publication/238348725

## Authority control databases

https://www.researchgate.net/publication/238348725

https://www.researchgate.net/publication/238348725

Retrieved from "

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&oldid=1316138192

https://en.wikipedia.org/w/index.php?title=Binary_space_partitioning&oldid=1316138192

https://en.wikipedia.org/wiki/Help:Category

## Binary trees

https://en.wikipedia.org/wiki/Help:Category

## Geometric data structures

https://en.wikipedia.org/wiki/Help:Category

3D computer graphics

https://en.wikipedia.org/wiki/Help:Category

Hidden categories:

## Articles with short description

https://en.wikipedia.org/wiki/Help:Category

## Short description matches Wikidata

https://en.wikipedia.org/wiki/Help:Category

Articles needing additional references from May 2016

https://en.wikipedia.org/wiki/Help:Category

## All articles needing additional references

https://en.wikipedia.org/wiki/Help:Category

## All articles with dead external links

https://en.wikipedia.org/wiki/Help:Category

Articles with dead external links from January 2025

https://en.wikipedia.org/wiki/Help:Category

## Articles with example C code

https://en.wikipedia.org/wiki/Help:Category

