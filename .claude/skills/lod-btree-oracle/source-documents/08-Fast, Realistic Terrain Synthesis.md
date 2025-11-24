---
sourceFile: "Fast, Realistic Terrain Synthesis"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:10.863Z"
---

# Fast, Realistic Terrain Synthesis

35742585-825d-4a1e-9576-ab5348b8bec2

Fast, Realistic Terrain Synthesis

7e866188-3e0a-4011-86cf-e674f7cd5e66

https://pubs.cs.uct.ac.za/id/eprint/1052/1/Crause_Thesis_Revised.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOrYLqfcX1us2m32oLRH8scHeRYbGHC0fSFCGmWNQJnXMQUwHRiRwp8WnIkgB6hyP6px_Wcp7DgOl8BJlPsVssMcQB4K9xPWDiQebWPR50BtrF0z-IHqmwC2jQpK227xMWc7dDon4A=w474-h481-v0

0eb61845-a9fd-4ae9-bff2-c481c6ffcc22

Fast, Realistic Terrain Synthesis

## Justin Crause

## In fulfilment of the requirements

for the degree of

## Master of Science

## Supervised by

Assoc. Prof. James Gain

Assoc. Prof. Patrick Marais

## Plagiarism Declaration

I know the meaning of Plagiarism and declare that all of the work in the document, save for that

which is properly acknowledged, is my own.

The authoring of realistic terrain models is necessary to generate immersive virtual environments

for computer games and film visual effects. However, creating these landscapes is difficult – it

usually involves an artist spending many hours sculpting a model in a 3D design program. Specialised

terrain generation programs exist to rapidly create artificial terrains, such as Bryce (2013) and

Terragen (2013). These make use of complex algorithms to pseudo-randomly generate the terrains,

which can then be exported into a 3D editing program for fine tuning. Height-maps are a 2D data-

structure, which stores elevation values, and can be used to represent terrain data. They are also a

common format used with terrain generation and editing systems. Height-maps share the same

storage design as image files, as such they can be viewed like any picture and image transformation

algorithms can be applied to them.

Early techniques for generating terrains include fractal generation and physical simulation. These

methods proved difficult to use as the algorithms were manipulated with a set of parameters.

However, the outcome from changing the values is not known, which results in the user changing

values over several iterations to produce their desired terrain. An improved technique brings in a

higher degree of user control as well as improved realism, known as texture-based terrain synthesis.

This borrows techniques from texture synthesis, which is the process of algorithmically generating a

larger image from a smaller sample image. Texture-based terrain synthesis makes use or real-world

terrain data to produce highly realistic landscapes, which improves upon previous techniques.

Recent work in texture-based synthesis has focused on improving both the realism and user control,

through the use of sketching interfaces.

We present a patch-based terrain synthesis system that utilises a user sketch to control the

location of desired terrain features, such as ridges and valleys. Digital Elevation Models (DEMs) of

real landscapes are used as exemplars, from which candidate patches of data are extracted and

matched against the user’s sketch. The best candidates are merged seamlessly into the final terrain.

Because real landscapes are used the resulting terrain appears highly realistic. Our research

contributes a new version of this approach that employs multiple input terrains and acceleration

using a modern Graphics Processing Unit (GPU). The use of multiple inputs increases the candidate

pool of patches and thus the system is capable of producing more varied terrains. This addresses the

limitation where supplying the wrong type of input terrain would fail to synthesise anything useful,

for example supplying the system with a mountainous DEM and expecting deep valleys in the

output. We developed a hybrid multithreaded CPU and GPU implementation that achieves a 45

times speedup.

## Acknowledgements

Completing this task has been both a pleasure and a curse. I started out full of energy, enthusiasm

and happy to explore this exciting work. But the sense of relaxed freedom in the 'master's

environment' was to be short lived. I quickly realised the enormity of the task ahead and the time

ticked by alarmingly. My progress slowed and fell below expectation and then I was tempted into

the real world and the prospect of earning a proper living. My biggest challenge was explaining this

to my supervisors, but I mustered up the courage and faced the difficult conversation...that was two

years ago. Since then I've slowly but steadily plodded along, finding it challenging to balance a

professional career with my studying and still have a social life. It has taken me longer than planned

but I've had fantastic support and guidance from my supervisors James and Patrick, who never

wavered in their support of my efforts over the years. I have now reached the end of this road and

completed my mammoth project. My heartfelt thanks to all my friends and family who have

encouraged and motivated me, even when I was thinking of throwing in the towel!

Looking back on it now, it was an amazing adventure. There will be many memories of long days

in the lab with friends, the morning muffins from our weekly meetings and the pub lunches

afterwards – celebrating the end of yet another week. I am now able to close this chapter of life and

start work on my next adventure.

To everyone that made this possible, THANK YOU.

## Table of Contents

PLAGIARISM DECLARATION ........................................................................................................ II

ABSTRACT ........................................................................................................................................ III

ACKNOWLEDGEMENTS ................................................................................................................. IV

TABLE OF CONTENTS ..................................................................................................................... V

LIST OF FIGURES .......................................................................................................................... VIII

LIST OF TABLES ............................................................................................................................ XIV

LIST OF LISTINGS ......................................................................................................................... XVI

1 INTRODUCTION ........................................................................................................................ 1

1.1 Aims ................................................................................................................................................ 3

1.2 Contributions .................................................................................................................................. 4

1.3 Thesis structure ............................................................................................................................... 4

2 BACKGROUND: TERRAIN GENERATION .......................................................................... 5

2.1 Terrain Representation ................................................................................................................... 5

2.2 Terrain Generation .......................................................................................................................... 7

2.2.1 Fractal-based generation .................................................................................................................. 8

2.2.2 Physics-based generation ............................................................................................................... 11

2.2.3 Texture-based generation .............................................................................................................. 13

2.3 User Control .................................................................................................................................. 17

2.3.1 Parameter manipulation ................................................................................................................. 17

2.3.2 Image-based control ....................................................................................................................... 17

2.3.3 Sketching ......................................................................................................................................... 17

2.4 Discussion ..................................................................................................................................... 18

3 BACKGROUND: GPUS & NVIDIA CUDA ............................................................................ 20

3.1 GPUs and Parallel Programming.................................................................................................... 20

3.2 NVIDIA CUDA ................................................................................................................................ 22

3.2.1 Motivation for using CUDA over alternatives ................................................................................. 22

3.2.2 CUDA Programming Model ............................................................................................................. 22

3.2.3 Execution Pipeline........................................................................................................................... 23

3.2.4 Memory Hierarchy .......................................................................................................................... 29

3.3 Performance considerations ......................................................................................................... 32

3.3.1 Maximise memory throughput ....................................................................................................... 32

3.3.2 Maximise parallel execution ........................................................................................................... 33

3.3.3 Maximise instruction throughput ................................................................................................... 33

3.4 Summary ....................................................................................................................................... 33

4 FRAMEWORK .......................................................................................................................... 34

4.1 User Input & Feature Extraction .................................................................................................... 34

4.2 Patch Matching ............................................................................................................................. 37

4.2.1 Feature Matching............................................................................................................................ 37

4.2.2 Non-Feature Matching .................................................................................................................... 39

4.3 Patch Merging ............................................................................................................................... 41

4.3.1 Graph-cut ........................................................................................................................................ 41

4.3.2 Shepard Interpolation ..................................................................................................................... 42

4.3.3 Poisson equation solver .................................................................................................................. 44

4.4 Research Outcome ........................................................................................................................ 45

5 ENHANCED FRAMEWORK ................................................................................................... 47

5.1 Multiple Input Sources .................................................................................................................. 47

5.2 CPU and GPU Accelerated Synthesis ............................................................................................. 48

5.3 Simplified User Sketching Interface ............................................................................................... 49

5.4 Pre-Processors and Pre-Loaders .................................................................................................... 50

5.5 Summary ....................................................................................................................................... 51

6 FEATURE SYNTHESIS ........................................................................................................... 52

6.1 Feature Extraction & Pre-Loaders ................................................................................................. 52

6.2 Cost Functions ............................................................................................................................... 52

6.2.1 Feature Profiling ............................................................................................................................. 53

6.2.2 Sum-of-Squared Differences (SSD) ................................................................................................. 54

6.2.3 Noise Variance ................................................................................................................................ 54

6.2.4 Graph-cut cost ................................................................................................................................ 55

6.3 Feature Matching – CPU ................................................................................................................ 55

6.3.1 Sequential CPU Implementation..................................................................................................... 56

6.3.2 Parallel CPU Implementation .......................................................................................................... 59

6.4 Feature Matching – GPU ............................................................................................................... 60

6.4.1 Caching of data on GPU .................................................................................................................. 60

6.4.2 User Patch Extraction ..................................................................................................................... 61

6.4.3 Candidate Cost Calculations ........................................................................................................... 61

6.4.4 Storing Best Candidates .................................................................................................................. 68

6.4.5 Merging ........................................................................................................................................... 69

6.5 Feature Merging ........................................................................................................................... 70

6.6 Optimisations................................................................................................................................ 71

7 NON-FEATURE SYNTHESIS ................................................................................................. 74

7.1 Candidate Extraction ..................................................................................................................... 74

7.2 Candidate Matching and Merging ................................................................................................. 75

7.2.1 Selecting Target Patch .................................................................................................................... 75

7.2.2 Matching – Cost Functions ............................................................................................................. 75

7.2.3 Matching – CPU Implementation ................................................................................................... 76

7.2.4 Matching – GPU Implementation ................................................................................................... 77

7.2.5 Merging ........................................................................................................................................... 78

7.3 Optimisations................................................................................................................................ 78

8 RESULTS.................................................................................................................................... 80

8.1 Feature Synthesis .......................................................................................................................... 81

8.1.1 Sequential CPU versions ................................................................................................................. 81

8.1.2 Single versus Multi-Threaded CPU .................................................................................................. 82

8.1.3 CPU versus incremental GPU implementations.............................................................................. 83

8.1.4 Utilising GPU Texture Memory ....................................................................................................... 85

8.1.5 CPU versus GPU Sorting of Candidates ........................................................................................... 86

8.1.6 Blocked GPU for Asynchronous Processing .................................................................................... 87

8.1.7 Culling Nearby User Patches ........................................................................................................... 88

8.1.8 Feature Complexity Change ............................................................................................................ 90

8.2 Non-Feature Synthesis .................................................................................................................. 91

8.3 Full Synthesis ................................................................................................................................ 92

8.3.1 Comparison with previous work ..................................................................................................... 92

8.3.2 Single versus Multi-Source synthesis .............................................................................................. 94

8.3.3 Patch Size change ........................................................................................................................... 96

8.4 Summary ....................................................................................................................................... 97

9 CONCLUSION ........................................................................................................................ 100

9.1 Limitations .................................................................................................................................. 101

9.2 Future-work ................................................................................................................................ 101

LIST OF REFERENCES ................................................................................................................ 103

10 APPENDIX .............................................................................................................................. 109

10.1 Feature Synthesis – CPU v1 vs. CPU v2 ........................................................................................ 109

10.2 Feature Synthesis – CPU v2 vs. CPU Parallel ................................................................................ 109

10.3 Feature Synthesis – CPU Parallel vs. GPU implementations ........................................................ 110

10.4 Feature Synthesis – Using GPU Texture Memory ........................................................................ 111

10.5 Feature Synthesis – CPU vs. GPU Candidate Sorting .................................................................... 112

10.6 Feature Synthesis – Asynchronous Blocked Implementation ...................................................... 113

10.7 Feature Synthesis – Culling Nearby User Patches ........................................................................ 114

10.8 Feature Synthesis – Feature Complexity Change ......................................................................... 114

10.9 Non-Feature Synthesis ................................................................................................................ 115

10.10 Full Synthesis – Previous Work ............................................................................................... 115

10.11 Full Synthesis – Single vs. Multiple Sources ............................................................................. 116

10.12 Full Synthesis – Varying Patch Size .......................................................................................... 117

## List of Figures

Figure 1.1: Example of a landscape generated for an upcoming game The Witcher III (2015) .............. 1

Figure 1.2: Still from the movie Avatar (2009) with computer generated landscape. ........................... 2

Figure 2.1: Example of height-map. 2D image shown on left with corresponding 3D rendering on the

right. Generated and rendered in GeoGen (2013) ................................................................................. 5

Figure 2.2: Triangulated Irregular Network format. (a) Top-down representation. (b) Perspective

view ......................................................................................................................................................... 6

Figure 2.3: Screengrab of the generated landscape in Minecraft .......................................................... 7

Figure 2.4: One of the earliest known examples of a Brownian Surface: Fractal Brown Islands

(Mandelbrot, 1983) ................................................................................................................................. 8

Figure 2.5: Example of Poisson Faulting over several iterations ............................................................ 9

Figure 2.6: The first 6 iterations of a Midpoint-Displacement algorithm ............................................. 10

Figure 2.7: Example of terrain generated through noise synthesis. Generated and rendered in

GeoGen (2013) ...................................................................................................................................... 11

Figure 2.8: Example of Hydraulic erosion. This is the fractal-generated terrain in Figure 2.7 after a

hydraulic erosion algorithm has been applied. Generated and rendered in GeoGen (2013) .............. 12

Figure 2.9: Illustration of patch placement order. (a) User Sketch. (b) Tree structure from PPA. (c) The

root patch is placed first. (d) Breadth-first traversal guides placement of proceeding patches. (e)

After feature placement is complete non-feature patches are placed. (f) Final result. (Image taken

from Zhou et al. (2007)) ........................................................................................................................ 15

Figure 2.10: Results of synthesis. (a) User Sketch. (b) DEM Exemplar File. (c) Synthesis output. (d)

Rendered terrain. (Image taken from Zhou et al. (2007)) .................................................................... 16

Figure 3.1: (a) Floating-Point Operations per Second and (b) Memory bandwidth, for both CPU and

GPU (NVIDIA, 2013b). This shows the large difference between GPU and CPU performance leading to

the use of GPUs for accelerated computation...................................................................................... 20

Figure 3.2: GPU devotes more transistors to data processing (NVIDIA, 2013b). There are significantly

more Arithmetic Logic Units (ALUs) dedicated to the control and cache units. .................................. 21

Figure 3.3: CUDA Processing Flow. (1) Data is copied from host to device; (2) Kernel is executed; (3)

Data is processed in the many threads on the GPU; (4) Result is copied back to host. ....................... 23

Figure 3.4: Schematic overview of the Grid-Block-Thread layout (NVIDIA, 2013b). The kernel is

loaded onto the device which is comprised of the blocks and threads. .............................................. 24

Figure 3.5: Example Grid/Block/Thread Indexing for a 2D grid and block layout (NVIDIA, 2013b). .... 25

Figure 3.6: Architecture of a Scalar Multiprocessor unit for a GeForce GTX 580 (Fermi) GPU (NVIDIA,

2013c). This represents all the command, control and cache units present. ....................................... 27

Figure 3.7: Example of Fermi's Dual Warp Schedulers. Each scheduler is assigned a group of warps;

the first scheduler is responsible for warps with positive ID and the second for negative IDs. At each

clock-cycle both the schedulers select an instruction to execute for a particular warp. Since two

warps are run concurrently, each works on only half its instructions, requiring two cycles to

complete. (NVIDIA, 2013c) .................................................................................................................... 28

Figure 3.8: Memory Hierarchy. Each level shows the scope of the different types of memory. Local

memory is restricted to a single thread. Shared memory can be accessed from all threads in a single

block and global memory is accessible between one or more grids. (NVIDIA, 2013b) ........................ 30

Figure 3.9: Memory access pattern for coalesced reading. Both (a) and (b) require a single 128B

transaction whereas (c) requires two 128B transactions, which decreases performance to 50%.

(NVIDIA, 2013b) .................................................................................................................................... 32

Figure 4.1: Overview of patch-based terrain synthesis framework developed by Tasse et al. (2011).

The terrain sketching interface is the entry point to the system, where the user sketches their

desired terrain. This is used initially to produce a synthesised terrain, which together with a source

file is run through feature extraction. Patch matching and merging is run with the result being

deformed according to the user’s initial sketch to produce the final terrain. This feeds back allowing

the user to modify the terrain and re-run synthesis. ........................................................................... 34

Figure 4.2: Different steps of ridge extraction with the Profile recognition and Polygon breaking

Algorithm (Tasse et al., 2011). The final result is the minimum amount of points required to describe

the main feature path. .......................................................................................................................... 35

Figure 4.3: Patch-based texture synthesis. a) Users sketch input. b) Valley lines extracted from

feature extraction on exemplar. c) Output after feature matching has completed. d) Final output

after non-feature matching has completed. ......................................................................................... 37

Figure 4.4: Example of different feature types based on the number of control points. a) Feature end

point. b) Feature path. c) Feature branch............................................................................................. 38

Figure 4.5: Feature dissimilarity Tasse et al. (2011), an illustration of how the algorithm examines the

pixel data in a patch. (a) User patch. (b) Candidate patch. (c) Height profile for values perpendicular

to path. (d) Height profile for values along path. ................................................................................. 39

Figure 4.6: Example showing the empty region , with the boundary    highlighted in blue. A patch

centred around a point on    is enlarged. ................................................................................... 40

Figure 4.7: Illustration of the graph-cut algorithm between patches    and   . The optimal seam

connects adjacent pixels between the two patches. ............................................................................ 42

Figure 4.8: Example of the graph-cut algorithm steps. a) & b) Patches    and   . c) The overlap

region  highlighted. d) The optimal seam between the two patches highlighted after merging. ..... 42

Figure 4.9: Results of Shepard Interpolation. a) Output from graph-cut algorithm. b) B is deformed to

match the pixel values of A along the optimal seam. ........................................................................... 43

Figure 4.10: Poisson equation solving process. a) The image as output from Shepard Interpolation,

patch   . b) The gradient fields of the patch   . c) The modified gradient fields free of discontinuities

along the seam. d) The final output after the Poisson equations are solved. ...................................... 44

Figure 4.11: Comparison of patch merging techniques (Tasse et al., 2011). (a) No patch merging. (b)

Graphcut algorithm. (c) Shepard Interpolation. (d) Results from Zhou et al. (2007). (e) Results from

Tasse et al. (2011). ................................................................................................................................ 46

Figure 5.1: Overview of our proposed system for enhanced terrain synthesis. The entry-point to our

system is the simplified sketching interface, which when synthesis is initiated, run through feature

extraction to build the user candidates. A collection of varying source files is run through feature

extraction also, with the feature data being used in matching and merging with the sketch data. A

final step fills in the gaps left from feature synthesis with data from the source candidates to

complete the terrain. ............................................................................................................................ 47

Figure 5.2: Examples of limitations with using a single source for terrain synthesis. (a) Using an input

terrain without the correct type of feature data, source image lacks ridge details. (b) System can

produce noticeable repetition in output terrain. ................................................................................. 48

Figure 5.3: a) The main sketching interface with all menus expanded. b) Sample sketch drawn with

feature detection run. c) Output after feature synthesis. d) Final output ........................................... 50

Figure 6.1: Feature synthesis pipeline showing flow of data for the Feature Matching & Merging

block of our system (Figure 5.1) ........................................................................................................... 52

Figure 6.2: Feature profiling algorithm against user and source candidate patches. Segments r and s

represent profile paths for the patches. ............................................................................................... 53

Figure 6.3: Overview of the second version of sequential CPU feature matching. Feature merging is

included as it is a required part of the flow. More information on the merging process is found in

section 6.5. ............................................................................................................................................ 56

Figure 6.4: Overview for parallel CPU feature matching. ..................................................................... 59

Figure 6.5: Overview of the GPU feature matching pipeline ................................................................ 60

Figure 6.6: Overview of the feature merging pipeline: a) Single-threaded pipeline, b) Internal block

for multithreaded version. .................................................................................................................... 70

Figure 6.7: Example of repetition in output terrain. (a) Repetition with adjacent patches (b)

Repetition check implemented to overcome this issue ....................................................................... 71

Figure 6.8: (a) Example of error with feature detection engine forming multiple parallel lines. (b) This

results in heavy overlaying of patches, which wastes performance. ................................................... 72

Figure 6.9: Illustration of blocked design for candidate processing. a) A queue of blocks of length k

that are sequentially processed by the algorithm in b) on the GPU. Results form a queue c) which is

processed by the CPU in d) ................................................................................................................... 73

Figure 7.1: Non-feature synthesis pipeline showing flow of data for the Non-Feature Matching &

Merging block of our system (Figure 5.1) ............................................................................................. 74

Figure 7.2: Overview of the GPU non-feature matching pipeline. Candidates are cached on the GPU

initially. The system then loops until all ‘holes’ are filled. GPU acceleration is used to calculate the

costs with the rest being done on the CPU........................................................................................... 77

Figure 8.1: The two test images used for evaluation. a) The small         terrain. b) The large

terrain. The white lines represent ridges with the black lines being valleys as detected

by the system. ....................................................................................................................................... 80

Figure 8.2: Runtime chart comparing the two main CPU implementations. These two

implementations have very similar runtimes despite the large architectural changes between them.

Table 10.1 gives the runtime numbers in a table and reveals that CPU v2 is slightly faster than v1. .. 82

Figure 8.3: Runtime results comparing the parallel CPU implementation against CPU v2. Here we

observe a large reduction in synthesis time almost reducing it by half on the large terrain. Full

runtime values are presented in Table 10.2. ........................................................................................ 82

Figure 8.4: Speedup graph comparing the runtime in seconds and the observed speedup for the

parallel CPU implementation over CPU v2. We observe a     times speedup achieved for both test

terrains. ................................................................................................................................................. 83

Figure 8.5: Runtime results for the eight GPU implementations compared against the parallel CPU

implementation for the small and large terrains. We can see an overall downward trend to the graph

with the times decreasing with each iteration. v1 is a translated form of the parallel CPU

implementation. v2 adds some shared memory and more threads. v3 attempts to optimise functions

but introduces more branching. v4 unrolls an entire loop utilising more concurrent threads. v5

changes the architecture to allow a new dimension of threads for improved concurrency. v6

optimises v5 preventing unnecessary recalculation of values. v7 combines elements from v5 and v6.

v8 revisits v4 and incorporates the newer changes in v7. Full runtime values are presented in Table

10.3. ...................................................................................................................................................... 84

Figure 8.6: Speedup and runtime graph comparing the parallel CPU version against all eight GPU

implementations. Similar performance is noted for both the small and large terrains, although a

slightly higher speedup is noted for the larger terrain. ........................................................................ 84

Figure 8.7: Runtime results for our texture memory GPU implementation being compared against

GPU v8. There is a slight performance gain when using texture memory. This is because we already

are using coalesced memory access for our image data. Full runtime values are presented in Table

10.4. ...................................................................................................................................................... 85

Figure 8.8: Speedup and runtime graph comparing the use of GPU texture memory against the

parallel CPU and GPU v8 implementations. Using texture memory now brings the total speedup to

24 times fast than the parallel CPU implementation. .......................................................................... 85

Figure 8.9: Runtime results comparing the three different candidate soring functions. The Patch

Matching component in the graph includes the sorting operation, which is why we see the green

bars decreasing in size with the GPU and Thrust (2013) implementations. Full runtime values are

presented in Table 10.5. ....................................................................................................................... 86

Figure 8.10: Speedup and runtime graph comparing the three different candidate sorting functions.

We see a modest performance increase when using the GPU for sorting, even with our simple kernel

implementation. Using the Thrust (2013) library further improves the result due to their kernel being

highly optimised. ................................................................................................................................... 86

Figure 8.11: Runtime results comparing against our asynchronous blocked design against the current

best GPU implementation using Thrust sorting. For this test we need to compare the total runtime as

the two components are run concurrently on the CPU and GPU, which reduces the overall time as

there is far less idling occurring. The timings for matching and merging are approximately the same

but due to running them asynchronously we see a reduced overall runtime (Table 10.6). ................ 87

Figure 8.12: Speedup and runtime graph for the asynchronous blocked design against the parallel

CPU and Thrust GPU implementations. We see a marginal increase with the asynchronous design for

the small terrain with a very large increase on the large terrain. This is attributed to the total number

of features, as the large terrain has a high feature count it is divided up into more blocks which

enables the concurrent processing on the CPU and GPU. .................................................................... 87

Figure 8.13: The two test images used to test culling of excess user patches. These were designed to

exacerbate the unfortunate feature of the original feature extraction algorithm. a) The small

terrain. b) The large           terrain. The white lines represent ridges with the

black lines being valleys as detected by the system. ............................................................................ 88

Figure 8.14: (a) Example of error with feature detection engine forming multiple parallel lines. (b)

This results in heavy overlaying of patches, which wastes performance. These excess patches are

culled by the system. ............................................................................................................................ 89

Figure 8.15: Runtime results comparing the implementations when either culling of nearby user

patches or not. This is an issue with the original feature extraction algorithm. We address this by

examining user patches and removing those that are in close proximity to one another. This reduces

the total number of features requiring synthesis and thus improves performance as shown above.

Full runtime values are presented in Table 10.7. ................................................................................. 89

Figure 8.16: Speedup and runtime graph showing the performance gain when culling nearby user

patches that are not required. We see a higher gain in the smaller terrain as the proportion of culled

patches is higher than the larger terrain. ............................................................................................. 89

Figure 8.17: Runtime results for complexity with increasing total number of patches requiring

synthesis. We observe that with an increase in the number of features we see an increase in the

time required, with approximately the same proportion of time spent on matching and merging

components. Full runtime values are presented in Table 10.8. ........................................................... 90

Figure 8.18: Plotting the runtime and feature count values on a graph shows a linear relationship for

both, which indicates that the system scales well when increasing the number of features. ............. 90

Figure 8.19: Runtimes for the four main contributing components during non-feature synthesis

comparing a CPU only implementation to a GPU-enhanced one. We observe that calculating the

candidate costs on the GPU significantly reduces the required time. Examining the time values in

Table 10.9 we see a 200 times speedup for cost calculation on the small terrain. .............................. 91

Figure 8.20: Speedup and runtime graph for the non-feature synthesis stage of our system

comparing CPU bound and GPU-enhanced implementations. ............................................................ 91

Figure 8.21: The user images used for this test. a) The original small         terrain. b) The larger

image, which only features valley data. ....................................................................... 92

Figure 8.22: Runtime results comparing the previous work by Tasse et al. (2011) to our system. We

were only able to run their CPU version, which is why we include our two CPU implementations and

our best GPU implementation. The graph above shows that the runtime for our system is far less

with the three implementations appearing as tiny columns. Table 10.10 provides the actual runtime

values, which better shows the time difference between all the versions. ......................................... 92

Figure 8.23: Speedup and runtime graph comparing the previous work to our system. Here we see

the large performance increase our system achieves when running under the same test conditions.

.............................................................................................................................................................. 93

Figure 8.24: a) Output from Tasse et al. (2011) system. b) Output from our system using the same

single source file. .................................................................................................................................. 93

Figure 8.25: Runtime results when running either a single or database of fifteen source files. The

figure shows the times for the feature and non-feature synthesis components. We see the majority

of the impact being confined to the feature synthesis stage, this is due to there being more

candidates needing evaluation. Non-feature synthesis results are very close in size as there is more

of an impact from the number of iterations required to fill the output terrain with the candidate

matching only being a small percentage of the runtime. Full runtime values are presented in Table

10.11. .................................................................................................................................................... 94

Figure 8.26: Output terrain for: a) Single source. b) Multiple sources ................................................. 95

Figure 8.27: Example when running a ridge only terrain using a) Single source – Grand Canyon. b)

Multiple sources. The single source does not have sufficient ridge data resulting in a poor terrain

compared to the clearly defined structure when using multiple sources. ........................................... 95

Figure 8.28: Runtime results when using different patch sizes to synthesise terrains. We observe that

for the small terrain the optimal patch size is       with the large terrain performing better with

larger patch sizes. Upon further inspection of the timing values (Table 10.12), we note that for both

terrain sizes the feature matching component performs fastest with a patch size of      . Larger

patch sizes reduce the non-feature synthesis time as more data is placed on each iteration, requiring

less overall. ........................................................................................................................................... 96

Figure 8.29: Speedup and runtime graph showing the effect of varying the patch size for synthesis

operations. For the small terrain the optimal size is      , with the large terrain performing best

with the       patch size. ................................................................................................................ 97

Figure 8.30: Our small test terrain (       ). b) The output from our synthesis system

(Completed in 13 seconds). c) 3D rendering of the terrain. ................................................................. 98

Figure 8.31: a) The lambda symbol drawn as valleys (       ). b) The output from our synthesis

system (Completed in 14 seconds). c) 3D rendering of the terrain. ..................................................... 98

Figure 8.32: a) A combination of ridges and valleys (         ). b) The output from our

synthesis system (Completed in 52 seconds. c) 3D rendering of the terrain. ...................................... 99

Figure 8.33: a) A combination of ridges and valleys (         ). b) The output from our

synthesis system (Completed in 49 seconds. c) 3D rendering of the terrain. ...................................... 99

## List of Tables

Table 2.1: Comparison of terrain generation methods. *A high user-control system is provided by

Gain et al. (2009) ................................................................................................................................... 19

Table 3.1: Device Memory Summary. *Cached on devices with Compute Capability 2.0 and up. ...... 31

Table 8.1: Number of detected user features patches and dimensions of the two main test terrains

we use. Difference is ridge/valley count is determined by feature extraction and dependant on

sketch used. .......................................................................................................................................... 80

Table 8.2: Number of detected features before and after the culling algorithm. The dimensions for

the terrain are,         for the small terrain and           for the large terrain................. 88

Table 9.1: Comparison of terrain generation methods. Table from section 2.4 ................................ 101

Table 10.1: Runtime results comparing the two main CPU implementations. A speedup column is

provided to show the performance gain achieved with version two. These implementations perform

very similarly despite the large architectural changes. ...................................................................... 109

Table 10.2: Runtime results showing the performance improvements when multithreading our CPU

v2 implementation. Only the cost computation stage was multithreaded as such the times for the

other sections remain relatively the same. ........................................................................................ 109

Table 10.3: Runtime results comparing the parallel CPU implementation against the different GPU

implementations for the small and large terrains. v1 is a translated form of the parallel CPU

implementation. v2 adds some shared memory and more threads. v3 attempts to optimise functions

but introduces more branching. v4 unrolls an entire loop utilising more concurrent threads. v5

changes the architecture to allow a new dimension of threads for improved concurrency. v6

optimises v5 preventing unnecessary recalculation of values. v7 combines elements from v5 and v6.

v8 revisits v4 and incorporates the newer changes in v7. .................................................................. 110

Table 10.4: Runtime results comparing the texture memory GPU implementation compared to the

parallel CPU and GPU v8 implementations. There is a slight performance gain when using texture

memory. This is because we already are using coalesced memory access for our image data. The first

two speedup columns are comparing the methods against the CPU implementation with the last

speedup value comparing the improvement texture memory provides compared to the current best

GPU v8 implementation. ..................................................................................................................... 111

Table 10.5: Runtime results comparing sorting of the candidates with the CPU, our own GPU kernel

or using the Thrust (2013) library. We observe a large speedup when using the GPU to sort

candidates, which is further increased when using the optimised Thrust library. The first two

speedup columns compare the GPU sorting algorithms to CPU sorting with the final speedup value

comparing the improvement Thrust provides over our implementation. ......................................... 112

Table 10.6: Runtime results comparing the parallel CPU and our current best GPU implementation,

using Thrust sorting, against our asynchronous block system. This allows us to execute code on both

the CPU and GPU concurrently, which produces a very large improvement over our current best GPU

implementation. The first two speedup columns are compared to our parallel CPU implementation

with the last indicating the gain when using asynchronous processing over the Thrust enabled GPU

implementation. ................................................................................................................................. 113

Table 10.7: Runtime results comparing the implementations when either culling of nearby user

patches or not. This is an issue with the original feature extraction algorithm. We address this by

examining user patches and removing those that are in close proximity to one another. This reduces

the total number of features requiring synthesis and thus improves performance as shown above.

We see a higher gain in the smaller terrain as the proportion of culled patches is higher than the

larger terrain. ...................................................................................................................................... 114

Table 10.8: Runtime results for varying complexity in terms of the number of total features

synthesised by the system. We observe that with a linear increase in the total number of features

there is a linear increase in the time required. This allows our system to scale for larger more

complex terrains. ................................................................................................................................ 114

Table 10.9: Runtime results for the non-feature synthesis stage of our system. Times presented are

for a CPU only and GPU enhanced implementations. The GPU is utilised for cost calculations to help

reduce the overhead of synthesis, the other components are left CPU bound. There is a massive

improvement in the cost calculation stage, which has the largest runtime on the CPU. .................. 115

Table 10.10: Runtime results when comparing our system to the previous work by Tasse et al.

(2011). Timing values for Ridges, Valleys and Non-Feature Synthesis were provided in the previous

system as such we omit the breakdown for our system in order to only compare the relevant data.

While we could only compare the CPU implementation of Tasse et al. (2011), we observe that our

system runs significantly faster under the same test conditions. Our system was run with a single

source file to match the output more closely..................................................................................... 115

Table 10.11: Runtime results for our system when using either a single input source or our database

of fifteen. We see the feature synthesis stage has a fairly high cost for using multiple files, although

less so when using the larger terrain. We observe the runtimes for non-feature synthesis being very

close between the two implementations due to the large cost of running many iterations to

completely fill the output terrain. When looking at the total synthesis time for the large terrain we

see the larger database has very minor impact on the performance. ............................................... 116

Table 10.12: Runtime results for varying the size of the patch used by our system. We start off with a

small       patch size up to a large         patch size. We observe two outcomes when

looking at the feature and non-feature synthesis components, which is similar for both terrain sizes.

For feature synthesis we see a patch size of       being optimal with the fastest runtime

recorded. For non-feature synthesis we observe that the larger the patch size the faster the runtime.

This is attributed to a larger area being merged into the output, which reduces the amount of empty

areas thus requiring less iterations to complete. ............................................................................... 117

## List of Listings

Listing 3.1: Example of a CUDA Kernel. This kernel takes a flattened square array of size w and

squares its values. ................................................................................................................................. 26

Listing 3.2: Example Kernel Invocation. This is the sample code which will launch the CUDA kernel

defined in Listing 3.1. The threads-per-block and blocks-per-grid are defined and used in the call. This

also assumes initialisation of data for the array on the device. ........................................................... 26

Listing 5.1: Algorithm overview for the candidate searching algorithm .............................................. 49

Listing 6.1: Feature Profiling algorithm ................................................................................................. 54

Listing 6.2: Sum-of-Squared Differences algorithm .............................................................................. 54

Listing 6.3: Noise Variance algorithm ................................................................................................... 55

Listing 6.4: Graph-cut cost algorithm .................................................................................................... 55

Listing 6.5: Algorithm overview for the version one of sequential feature matching. ......................... 56

Listing 6.6: Algorithm overview for selecting the best overall patch ................................................... 57

Listing 6.7: Algorithm overview for the version two of sequential feature matching. ......................... 58

Listing 6.8: Overview for the user patch extraction on the GPU .......................................................... 61

Listing 6.9: Overview for the candidate patch extraction kernel ......................................................... 62

Listing 6.10: First version of our GPU cost calculation process ............................................................ 63

Listing 6.11: Second version of our GPU cost calculation process........................................................ 64

Listing 6.12: Fourth version of our GPU cost calculation process ........................................................ 64

Listing 6.13: Overview for the advanced candidate patch extraction kernel ....................................... 65

Listing 6.14: Fifth version of our GPU cost calculation process ............................................................ 66

Listing 6.15: Sixth version of our GPU cost calculation process ........................................................... 67

Listing 6.16: Seventh version of our GPU cost calculation process ...................................................... 67

Listing 6.17: Eighth and final version of our GPU cost calculation process .......................................... 68

Listing 6.18: Algorithm for sorting candidates based on cost in ascending order ................................ 69

Listing 7.1: Algorithm overview for building boundary dataset ........................................................... 75

Listing 7.2: Algorithm overview for the CPU non-feature matching implementation ......................... 76

Listing 7.3: Algorithm overview for the CPU non-feature matching implementation ......................... 77

https://lh3.googleusercontent.com/notebooklm/AG60hOr0T3NOJ73rr9hV4XXzsIDB_WsiAem6aV8HpCJJexsgo1knWZ7l8NBvh-LF76EuE_cX-8bw0nDFMCDR4S-mp0Ua6Km13GXMN0k_MM071emeKvaBNvyry0FIx0qdxfWT39X5Espy=w1380-h776-v0

625deb3c-b824-4462-b58d-c460cabadb9e

1 Introduction

Detailed terrain models are a fundamental component of many 3D scenes used in computer

games (Figure 1.1) and the creation of film visual effects (Figure 1.2). The creation of realistic

artificially-generated terrain helps the gamer or audience feel immersed in the environment. In

some instances, where the landscape is only used as a visual backdrop with no user interaction, a

simple two-dimensional (2D) terrain profile might be satisfactory. This profile can be either a hand

drawn graphic or an image of a real landscape. This technique was used in early games and virtual

environments to reduce the space requirements and computational complexity. However, it is more

often a requirement that the environment be navigable, which requires a three-dimensional (3D)

landscape. Creating these landscapes is no easy task – it usually involves an artist spending many

hours tweaking a 3D mesh structure. As the requirements for larger, more realistic and detailed

terrains increase so does the complexity and amount of time required to manually create them. As

an alternative artists can make use of real landscapes in the form of digital elevation models (DEMs)

that can be obtained from the US Geological Survey (USGS, 2013). These provide true realism but

often do not match up with the artist’s vision, thus requiring manual editing. This has led to great

interest in the procedural generation of terrain models. Procedural methods are algorithms that

allow for the quick generation of data with little user input.

Figure 1.1: Example of a landscape generated for an upcoming game The Witcher III (2015)

Terrain synthesis is the process of creating an artificial landscape algorithmically using procedural

methods. The two most common procedural methods are fractal generation and physical simulation.

These generate terrains with a minimal amount of user input in the form of algorithm parameters.

These parameters are usually unintuitive and many iterations of synthesis may be required before

an optimal set of parameters is found to generate a suitable terrain. Software packages such as

Bryce (2013) and Terragen (2013) can be used, but in most cases the artist will still need to tweak

https://lh3.googleusercontent.com/notebooklm/AG60hOqL1lfe-7JvdgmeeoOdsWcjmoj_heO6oXqz5xx5D2H8MuzGzlyXhwaQZcfd3V27PlOMsmR9pNBVhcgP_mFwxYE9XKFma1CtIZpVqmcTqPeMAvXjviyXH0wSHZp0lMPjYXK6R1GE0Q=w1380-h776-v0

6b72074b-ffa4-4bee-9e12-5f3ff5e18e3f

the terrain to achieve the desired look. These packages use methods that pseudo-randomly displace

height values of an initially flat terrain model according to a given fractal technique. Furthermore,

these programs are unable to simulate physical weathering patterns, and generated terrain models

must be exported to some other system to add such detail. An erosion system will enhance the

realism of the input terrain but requires the user to have a fair understanding of erosion models and

is also computationally expensive. These programs can more rapidly generate terrains but the

results are somewhat random. A system that allows the user to specify terrain constraints and

produces a realistic-looking terrain that closely matches the user’s expectations would be ideal.

Figure 1.2: Still from the movie Avatar (2009) with computer generated landscape.

An alternative procedural method is example-based, which works by utilising existing terrain data,

often in the form of Digital Elevation Models (DEMs) commonly from the USGS (2013), and

recombining them using texture synthesis techniques. Current state-of-the-art systems using this

method are those by Zhou et al. (2007) and Tasse et al. (2011). The user specifies their requirements

in the form of a sketch, which provides the location of certain dominant features, such as mountains

and valleys. The system then takes this sketch and breaks it up into small blocks or patches which it

then searches for the best match from a pool of candidates – patches taken from the DEM files with

feature rich characteristics. For the areas where no features are described, the system will populate

the terrain with insignificant data – candidates with no dominant feature characteristics. The use of

DEMs as the input source produces terrains that appear highly realistic. Combining realism with the

flexibility of a sketching interface provides a good system for synthesising artificial terrains. There

are some issues with the current implementations, which include being slow to execute and limited

with the variability of the terrain when using only a single input source. These are two key areas for

improvement.

1.1 Aims The primary objective of this research thesis is to build a terrain synthesis system to rapidly

generate realistic terrains from the input of a simplified user interface. The system builds on

previous work by Tasse et al. (2011) and provides several extensions to improve the synthesis

results. To facilitate this objective, the following key requirements were identified:

 A system capable of producing realistic terrains, making use of landscape data from the

United States Geological Survey (2013). A user study conducted by Tasse et al. (2011)

confirmed that their system produces terrain that is more realistic than ones generated by

a multi-resolution deformation (a procedural synthesis method). The same techniques will

be incorporated into this research with the results being compared to the system by Tasse

et al. (2011).

 A simple interface that allows the user to sketch out the placement of both ridge and

valley line features to describe the overall design of their terrain.

 Make use of a large collection of input terrains to increase the candidate pool for

synthesis. When using a single input terrain the variability of features is constrained by

the amount of sample data available. Using multiple input sources allows for better

quality, more diverse terrains. This objective represents the novel contribution of this

 Accelerate the process by implementing CPU caching algorithms and optimising the

process to reduce the synthesis time.

 Further accelerating the synthesis process with the aid of programmable Graphics

Processing Units (GPUs) and NVIDIA’s Compute Unified Device Architecture (CUDA).

Modern GPUs have become more powerful than CPUs by orders of magnitude for certain

computations that can be parallelised, such as scientific data processing. CUDA is an

application interface developed to enable General Purpose GPU (GPGPU) computing. This

has spawned a new era in computational research focusing on parallel computation.

Texture synthesis is one field which benefits from parallel computation. We make use of

this to dramatically reduce the time it takes to complete a synthesis option, thus making

our system suitable as a rapid prototyping tool. We combine the CPU & GPU optimisations

to create a hybrid system for maximum performance.

 The system will be evaluated with visual inspection to verify that the realism of our output

matches the quality of the previous system. Speedup comparisons will be made between

all the different CPU and GPU versions to evaluate their performance.

1.2 Contributions The main contribution for this research is the introduction of multiple input sources to increase

the variety of data available during synthesis operations. Prior work with patch-based systems

focuses on the use of a single input source to synthesise the terrains (Zhou et al., 2007, Tasse et al.,

2011). This is reliant on the user selecting the correct source to get the best results as some sources

might not contain the correct features required. We show that our system is capable of producing

very large terrains, varied terrains. Our hybrid CPU-GPU implementation is capable of a    times

speedup over a single-core CPU system.

1.3 Thesis structure The structure of the thesis is as follows:

 Chapters 2 and 3 contain background information on procedural terrain generation and

Graphics Processing Units (GPUs) respectively. GPUs can be used to accelerate

computation of parallel algorithms and we use them to reduce synthesis times of our

 Chapter 4 provides a detailed analysis of the system developed by Tasse et al. (2011),

which we extend in this thesis. The limitations of this system are highlighted together with

our proposed improvements.

 Chapter 5 presents the overview of our system, focusing on our new contributions to

example-based terrain generation.

 Chapters 6 and 7 describe the core components of feature and non-feature synthesis in

detail. This includes the various CPU and GPU versions we developed while improving and

optimising the system.

 Results are presented in Chapter 8 which compares our new system to that of Tasse et al.

(2011). A single core CPU implementation is compared to a hybrid approach, which

incorporates multiple threads and a GPU to accelerate the synthesis stage. Visual

assessment is used to verify that realism is preserved with our proposed modifications.

 Chapter 9 concludes the thesis and lists some possible avenues of future work to improve

and further accelerate the synthesis of terrains.

https://lh3.googleusercontent.com/notebooklm/AG60hOrHhuhSuRR8BU3TouOeFOcctDyT7lP6Iw5FemmuPjjxuH_RKWkA0e_EaPH0tUdkTHM0GziRe5Ep1dcQmqlDKZiJloF_bcUeLAsxZn1ncLfRuTUlnhSnft9TXPam5lQK9pfzgW-yjQ=w1378-h689-v0

0f912705-f06c-4ca3-9b18-4b61749983b2

2 Background: Terrain Generation

This chapter provides an overview of methods to procedurally generate terrains. We begin by

describing common representations of terrain data (Section 2.1) and follow this with a discussion of

important generation techniques (Section 2.2). A summary of the techniques and motivation for our

choice of synthesis concludes this chapter.

2.1 Terrain Representation The simplest representation of terrains is as a two-dimensional grid-based data-structure. This

data-structure is commonly represented as an image known as a height-map. Height-maps are easy

to use given their uniform grid-based nature, where each entry stores a height value for the

corresponding location on the terrain. Figure 2.1 shows a simple example of a height-map

represented as a 2D image (shown left) and the corresponding 3D rendering on the right. The pixel’s

intensity represents the height of the terrain and is stored in a single channel of the image, resulting

in a grayscale image. The USGS (United States Geological Survey) have surveyed many real

landscapes and made available in a height-map digital form commonly referred to as Digital

Elevation Models (DEMs). These DEMs are freely available from the USGS website (USGS, 2013).

Height-maps can be encoded using a variable number of bits. If only a single channel (8-bit) is used,

this allows only 255 possible height values, which is insufficient for replicating highly detailed terrain.

The number of bits used depends largely on the format, with most DEM files being stored using 16-

bit images, giving 65,535 height values.

Figure 2.1: Example of height-map. 2D image shown on left with corresponding 3D rendering on the right. Generated and rendered in GeoGen (2013)

The regular grid structure of height-maps facilitates storage efficiency and ease of implementation

and is well suited to filter-based image processing. However, height-maps are not without

limitations. For instance, they lack the ability to represent overhangs, caves or structures where a

given location needs multiple height values.

https://lh3.googleusercontent.com/notebooklm/AG60hOom2TUaymApVSrrZW8nfc_8LyvsCaDofDZlhwUjeHMzgN1ZxbjKH9OXuEih40sjHlDa3C8AGjiNcy6YiUjdCf32oFuQ9SDkeP4H0NQPDDwtAwdh_qV9oQYGtc555DrkcDy-NtQvYw=w761-h341-v0

63ec5747-beda-4030-96d0-846284b393c1

Terrain models can also be represented as a mesh of polygons, usually triangles. Triangulated

Irregular Networks (TINs) are a type of mesh structure in which the terrain is composed of a set of

connected, variably sized triangles (Peucker et al., 1978). The triangles vertices are adaptively

chosen, often with a Delaunay triangulation algorithm (Fowler and Little, 1979), to produce an

accurately representation of the terrain. TINs are able to capture three-dimensional structures such

as caves, where a height-map would fail, and also support a level-of-detail (LOD) system: higher

density areas are represented with many small, tightly-packed triangles and smoother, less detailed

areas with fewer larger triangles. As a result of the LOD system, the storage overhead for TINs is

small; they are, however, more difficult to manipulate procedurally due to their non-uniform

structure. An example of a TIN model is provided in Figure 2.2. TINs are more appropriate for

manual terrain modelling or rendering systems as these packages are designed to work with vertices

at non-uniform locations. For more details on TINs we refer the reader to Abdelguerfi et al. (1998)

and Pajarola et al. (2002).

Figure 2.2: Triangulated Irregular Network format. (a) Top-down representation. (b) Perspective view

Voxels (volumetric elements) are another way of representing terrains (Kaufman et al., 1993,

Dorsey et al., 1999). Voxels are the 3D equivalent to a 2D pixel. They are aligned in a three-

dimensional grid structure with their locations inferred from their index in the grid. Voxels can store

data such as colour and opacity, which together create 3D structures. As such voxel-grids are

capable of producing terrains with caves and other 3D structures. They are also widely used in the

scientific and medical domains. However, they have a large memory and storage overhead. This

impacts on rendering performance and restricts the size of structure that can be represented. A

good example of a voxel-based environment is from the popular video game Minecraft (2015), as

seen in Figure 2.3.

Another example of representing volumetric data is through using a system of particles, to

simulate granular materials such as sand. Bell et al. (2005) present such a system with non-spherical

particles. Granular materials behave differently compared to fluids because they can flow down a

slope like fluid and they can also form a static volume like a solid. These systems are more suited for

small-scale simulations where dynamic interactions are required as they require complex algorithms

to simulate the inter-particle interactions. Longmore et al. (2013) extend this work to leverage the

https://lh3.googleusercontent.com/notebooklm/AG60hOqY2ivyTnwalR93zPNJ1Ke8tp2GK4_HL9bLpFFPGV1JYJSDMDI-jyP6lNKyZUfqfcO6hFjdX_Elo-xZW7TWOdmnkLred6cavf1ipX8XUVOcY5OxAtj1adpXiRrhCirT3gb8Ab25rA=w1380-h862-v0

0389b86d-b3c2-4b3e-a07d-7dd45e60d5f6

parallel processing capabilities of modern GPUs. However, while more efficient than a CPU-based

implementation, the system is only intended for small-scale volumes due to it being computationally

expensive. The system uses 3D textures to store the particle information, which requires a large

amount of memory and limits the number of particles that can be simulated. These limitations

prevent us from utilising particles to represent a large terrain.

Figure 2.3: Screengrab of the generated landscape in Minecraft

Height-maps are the format most widely supported by common terrain generation packages

(Terragen, 2013, Bryce, 2013, WorldMachine, 2013). These packages make use of image processing

functions, which are easy to implement on height-map images. Another reason to use height-maps is

that real landscape data produced from aerial or satellite surveys is stored in this format. Since our

research will make extensive use of DEM images, and extends an existing height-map based

approach, our synthesis system is also based on height-map data-structures.

2.2 Terrain Generation Terrain generation is the process of creating an artificial landscape using procedural algorithmic

methods. Artificial terrains have many applications, including virtual environments, computer games

and movies. Terrains can be manually sculpted in 3D design programs but this is time consuming.

Fortunately, the process can be accelerated through the use of procedural methods. There are three

broad categories of procedural terrain generation techniques: Fractal, Physics and Texture-based. A

fractal surface is generated using a stochastic algorithm designed to produce fractal behaviour that

mimics that of a natural landscape. Physical simulations generally enhance the realism of a fractal

surface by applying erosion techniques to the surface. Finally, texture-based methods borrow

techniques from texture synthesis and typically copy data from a source image to build a new

terrain. Specialised programs such as Terragen (2013) and Bryce (2013) incorporate a number of

procedural methods for generating terrains quickly. However, these implementations only use

https://lh3.googleusercontent.com/notebooklm/AG60hOr4muZRjyXk_9l-VbG63QiffPHXcBtZAVdwK8dN-Ctq1ibRCQXOWAqn--PoL6MiBRt57BcE71acHYsZQZQpS75fQeN7dfk4WuOf-AsBCRqYTV0JxY2GNA7uA-t4vwdKNbEkk7Q-bg=w652-h527-v0

14dd0cc0-1eea-4b09-ba28-e109803b77d5

fractal techniques and may allow for erosion. We will show that in many cases such an approach is

not suitable when the user has a specific terrain design in mind. Each category is described in the

subsections below.

2.2.1 Fractal-based generation

Fractal methods were introduced by Benoit Mandelbrot in his seminal book, “The Fractal

Geometry of Nature” (Mandelbrot, 1983). He observed that natural shapes often contain self-similar

patterns: magnified areas are statistically similar to the original shape. He introduced Fractal

Geometry, which is a mathematical representation for natural shapes that are not easily described

by Euclidean geometry. The term ‘fractal-based’ has been applied more loosely over the years and as

such not all the techniques discussed in this section are truly fractal. Here the term classifies

techniques that generate terrains that exhibit self-similar patterns even if the algorithm is not

mathematically fractal.

Fractional Brownian motion (fBm) describes the process of representing these self-similar shapes.

It is also known as the “Random Walk Process” and consists of a series of steps in a random

direction, where the steps are normally distributed with a mean of zero and variance representing

the roughness. In terms of terrain generation, this involves a series of iterations of a stochastic

algorithm. Mandelbrot reasoned that if this process were extended in two dimensions the resulting

“Brownian surface” could be a visual approximation of a landscape in nature. Some of his work

explored the creation of these types of surfaces (Mandelbrot, 1975). Numerous researchers have

extrapolated Mandelbrot’s research and adapted it to produce fractal-based terrains (Fournier et al.,

1982, Voss, 1985, Miller, 1986, Lewis, 1987, Musgrave et al., 1989, Saupe, 2003). One of the earliest

known images of a Brownian Surface is presented in Figure 2.4; it is part of a sequence of fractional

Brown Islands.

Figure 2.4: One of the earliest known examples of a Brownian Surface: Fractal Brown Islands (Mandelbrot, 1983)

Poisson Faulting is one of the earliest forms of fractal terrain generation (Mandelbrot, 1983, Voss,

1985). This technique involves applying a series of Gaussian random displacements (faults) to a

plane. In simpler terms, a line is chosen across the plane and one side displaced by a random height.

This height value is reduced after each fault to avoid abrupt height changes in the final resulting

terrain. Figure 2.5 shows an example of the faulting process, captured at various synthesis stages.

https://lh3.googleusercontent.com/notebooklm/AG60hOoh8q1kri4rSJ-jbNLQZGIVt-7DFbmUeX0zmX8URGMGHDyrYNy2PwazaXoCB0KaPkElEY5mo-Bt4pbK2mJNTSWn7TFXE11fL03NpsZ4ghHVBbnUWkAiDll5RgMYoXvgjUPgWJYW=w890-h460-v0

308ba3c9-de2d-4932-84b5-2c478c01505a

This was employed by Mandelbrot to create fractal coastlines (Mandelbrot, 1975) and fractal planets

by Voss (1985). Faulting has a fixed resolution, which means there is no consideration of level-of-

detail (LOD). LOD is important in terrains as features are present on different scales, such as large

scale mountains at a coarse level and cracks on a fine level. These techniques also suffer from an

runtime, depending on the resolution and number of iterations, which severely impacts

performance. This led to the development of subdivision methods, discussed next.

Figure 2.5: Example of Poisson Faulting over several iterations

Subdivision methods work by iteratively adding finer levels of detail by dividing the current terrain

level. Midpoint-displacement is an example of this and is used to generate terrains (Fournier et al.,

1982, Miller, 1986, Lewis, 1987, Mandelbrot, 1988, Saupe, 2003). There a many midpoint-

displacement techniques, usually differing in the way points are interpolated during each step. A

simple example starts with a quad on a plane and randomly assigns the corners with seeding values.

This quad is then divided into four smaller quads. The values of the corners of the new quads are

interpolated between the corners of the parent quad. The midpoint value is additionally offset by a

random value controlled by the desired roughness of the terrain. This process is repeated on each of

the new quads until a desired LOD is obtained. This is shown in Figure 2.6; the simple process

outlined above is easily implemented and can run in linear time. Many subdivision methods are

subject to the “creasing problem” (Miller, 1986). This is the occurrence of creases or slopes along the

quad boundaries which are visually noticeable. A possible solution to this is to also apply a

displacement to all the points, not just the midpoint, this is called “successive random addition”

(Saupe, 1989). This leads to a significant amount of additional calculations being required, leading to

a general preference for simple Midpoint-displacement. Both these methods produce unnatural

repeating patterns in the terrain. Furthermore, only a single parameter is available to control terrain

roughness, which limits user control. Nonetheless, due to its fast generation time, most terrain

generation packages (such as Terragen (2013)).

https://lh3.googleusercontent.com/notebooklm/AG60hOq8cuxbEtXLg8pTyEJaXKezYtNJQWOItXuIdSnxU78lz6qLLYrjHjq04wHpkzOGdEHZgLGNHhV0N9kTp5hAgGDs7yAcRJ_zyaFqITC0nSSXskMy9NVN_KSWCy-vGEGLuYUSHYiK=w1253-h608-v0

b2e4673c-2b77-4165-a101-fbf7e02f187d

Figure 2.6: The first 6 iterations of a Midpoint-Displacement algorithm

Procedural Noise Synthesis can be informally defined as being the random number generator of

computer graphics. It is random and unstructured in pattern and is used when there is a need for a

source with extensive detail but lacking in evident structure. An example of a terrain synthesised

utilising noise synthesis is shown in Figure 2.7. These are popular methods used by commercial

packages, such as Bryce (2013), and have been widely researched (Fournier et al., 1982, Saupe,

1991, Schneider et al., 2006). Terrains are generated with simple implementations that involve the

summation of successive down-scaled copies of a band-limited noise function. This type of noise

generating function was first introduced by Ken Perlin (1985) and has been improved over the years

(Perlin, 2002). Typically each new copy contains a higher band-limited frequency with lower

amplitude such that large scale features are generated in early iterations and finer detail in the later

ones. A known problem with Perlin noise is that it is weakly band-limited as each band contains only

frequencies in a power-of-two (Lewis, 1989). This leads to aliasing and loss of detail. Wavelet noise

(Cook and DeRose, 2005) address these issues by taking an image filled with random noise ( ) and

downsampling to half its size (  ). This image is then upsampled to full size (   ) and subtracted

from the original image ( ). This results in band-limited data that can be used in terrain generation.

This method has the benefit of being almost perfectly band-limited and provides effective level of

detail without the aliasing issues of Perlin noise (Cook and DeRose, 2005). Wavelet noise is also fast

and easy to implement, leading to its use in terrain generation applications (de Carpentier, 2007,

Gain et al., 2009, Cui, 2011). For more details, we refer interested readers to the survey of

procedural noise functions by Lagae et al. (2010)

https://lh3.googleusercontent.com/notebooklm/AG60hOo4pjYjHWs4KVHZNOdSwtQaq9ggpnuqPw25DIZLSexkukiIQ-_b8GCzZm_hksPCteJodthp7oGdRxy8WWbvFqk2Ky_nt2RfYfNgSqPDkTHZ8d5oC-dsu1wCFAN9TqBhcwS85cZXOQ=w1378-h689-v0

0226d019-3ee2-4ad0-af39-5a825091c385

Figure 2.7: Example of terrain generated through noise synthesis. Generated and rendered in GeoGen (2013)

For additional information on fractal-based generation, we refer the reader to Ebert et al. (2003).

Musgrave covers many fractal generation methods in his work “Methods for Realistic Landscape

Imaging” (Musgrave, 1993). Fractal methods are easy to implement and widely supported by terrain

generation programs. User control suffers as their parameters are not intuitive and one cannot

easily generate specific variations of terrain. The generated terrain also lacks realism since it is

missing structures that arise from natural weathering and erosion on landscapes. The generation of

such aspects of realistic-looking terrains can be achieved by physics simulations; we provide a

discussion of such techniques below.

2.2.2 Physics-based generation

Physics-based methods aim to improve the realism of artificially generated terrain by simulating

the effects of erosion. Kelley et al. (1988) use hydrology data to generate stream network drainage

patterns that can be used to determine the topography of a terrain surface. However, this method,

while efficient, lacks the detail of a fractal surface since the terrain is modelled from the stream

network. While the stream network controlling the generation may be fractal, the surface used as

the initial terrain is not and cannot be made so without disturbing the drainage basins and stream

paths. Musgrave et al. (1989) combine a fractal height-map with a hydraulic model. Water is

dropped at each vertex and allowed to run off the terrain. The water erodes the surface by

depositing material at different locations based on a sediment load function for the water passing

over the vertex. Musgrave et al. (1989) also introduce a global model for simulation they based on

thermal weathering.

Thermal weathering is a simulation where sharp changes in elevation are diminished by knocking

material loose from steep inclines to eventually pile up at the bottom of the slope. This process

iterates until the maximum angle of stability for the material (talus angle) is reached. This technique

is simple to implement and runs efficiently (Musgrave et al., 1989, Marák et al., 1997, Olsen, 2004).

The following equation is evaluated at every vertex to determine the movement of material:

https://lh3.googleusercontent.com/notebooklm/AG60hOr2e8uuVM0P-2tUlk1Cp1nzggjLd7RdBdsgAiTHG8MUTKHpB9Zxjj08Jz_PLygeVrf96Io6WjSXuGrYyVAtb19I1qrUbrcsi0NpXSAPDcRXLuTqWI9mkDo1ZZTIGHBXpmGzxD9jFg=w1378-h689-v0

252bf8fb-e4c4-4be9-8d99-2e3d82754ba8

The difference in the height   of the current vertex   and its neighbour   is compared with the

globally defined talus angle  . If its slope is greater than the talus angle then a fixed percentage   of

the difference is moved onto the neighbour. Beneš and Forsbach (2001) introduce a terrain structure

that is more suitable for realistic erosion algorithms. Their model consists of a 2D grid similar to a

height-map but with each location storing an array representing different layers. Each layer stores

information, such as the elevation, and material properties, such as density. This is a trade-off

between a height-map and full voxel terrain. The model allows for air layers and, as such, cave

structures can be created. Thermal erosion is suitable for deposition of material to smooth out steep

slopes but does not simulate drainage patterns. This is achieved with hydraulic erosion.

Hydraulic (Fluvial) erosion is a simulation that deposits water at the vertices of the terrain and

allows it to flow downhill, eroding the surface as it goes. This method has been extensively

investigated in the literature (Kelley et al., 1988, D'Ambrosio et al., 2001, Beneš et al., 2006, Krištof

et al., 2009). Hydraulic erosion is more complex than thermal erosion but is simply described by

associating each vertex   at time   with a height      volume of water

and an amount of sediment

, suspended in it. At each time step excess water and sediment is passed to the vertex’s

neighbours. There are two approaches to calculating hydraulic erosion; Eulerian and Lagrangian.

Eulerian focuses on a fixed window and observes the particles affects only while they are in it.

Whereas Lagrangian focuses on an individual particle and tracks its movement throughout the

system (Krištof et al., 2009). Figure 2.8 shows an example of a terrain before and after hydraulic

erosion has been applied. Nagashima (1998) use a 2D fractal river network to which thermal and

hydraulic erosion simulations are applied, which erodes the banks. Beneš and Forsbach (2001)

improve on previous work by distributing sediment to the vertex’s eight neighbours and

implementing evaporation to simulate water pools drying up.

Figure 2.8: Example of Hydraulic erosion. This is the fractal-generated terrain in Figure 2.7 after a hydraulic erosion algorithm has been applied. Generated and rendered in GeoGen (2013)

The above methods describe a simple diffusion model, which does not accurately describe water

movement and sediment transport. These are closely related to the velocity of the water. Chiba et

al. (1998) introduce an enhanced method that incorporates a water velocity field. Water is placed at

each vertex and the velocity of the water is determined by the local gradient. While the water flows,

it dissolves some of the surface and deposits the stored sediment according to the velocity field. This

improved realism comes at the cost of higher computational cost. Neidhold et al. (2005) develop a

physically correct simulation based on fluid dynamics, which runs interactively, and allows for real-

time manipulation of the parameters. These physics-based methods are good for improving the

realism of a fractal-generated terrain but suffer from a high computational overhead.

Physics-based methods run extremely slowly when the number of points on the height-map or the

number of simulation iterations increases. One way to improve the simulation time is to sacrifice

physical correctness (Olsen, 2004). Another solution is to utilise modern GPUs to accelerate the

simulation (Anh et al., 2007, Mei et al., 2007, Št'ava et al., 2008). Despite sacrificing correctness,

these methods are still difficult to control as the user can only modify a few base parameters before

the simulation runs. The user also requires a fair understanding of the underlying physical laws to

implement correctly. A better way to achieve realistic-looking terrain is to utilise surveyed data of

natural landforms, for example DEMs (USGS, 2013).

2.2.3 Texture-based generation

Texture-based methods borrow techniques from the field of texture synthesis. Texture synthesis is

a widely used technique in computer graphics for procedurally generating textures. It is used to

construct a larger, possibly tileable, image from a small sample image. Textures can be arranged

along a spectrum based on their properties from structured to stochastic. Structured textures are

best described as possessing a repetitive, regular pattern while stochastic textures contain little

structure, being close to random noise. These extremes are connected by a smooth transition as

described by Liu et al. (2004). A full description of texture synthesis and its implementations is

beyond the scope of research and it is only briefly discussed here as an introduction to texture-

based terrain generation. We refer readers to an extensive survey on texture synthesis by Wei et al.

(2009). There are two main approaches to texture synthesis: pixel-based and patch-based. Pixel-

based methods generate the texture pixel-by-pixel with the new pixels value determined by its local

neighbourhood. A drawback to pixel-based methods is that they tend to lose global structure. This

shortcoming is addressed by patch-based methods. Patch-based methods copy and stitch blocks of

pixels from the source into the output. This preserves global structure and patterns and is thus

better suited to realistic texture-based terrain generation.

Balancing both user control and realism is difficult to achieve with both fractal-based and physics-

based methods, due to their use unintuitive synthesis control parameters. Texture-based methods

can use real height-maps (DEMs) as the source files and can thus produce highly realistic terrains.

This is a recent approach to terrain generation and research on the subject is limited (Chiang et al.,

2005, Dachsbacher, 2006, Saunders, 2006, Brosz et al., 2007, Zhou et al., 2007).

Chiang et al. (2005) present a patch-based system to iteratively generating macroscopic terrain

based off the construction of geometric primitives by the user. A database of patches (terrain units)

is manually populated by segmenting out features from real landscape maps based on two

properties. The height variation for each scanline in the unit must have a higher elevation near the

centre and be lower on the boundaries. There can also only be one type of feature present, for

example a hill, mountain, plain and plateau. The matching process compares the profile of the users

primitive to that of the terrain units, based on cross-section, mountain ridge and terrain contour

similarity. The best matching terrain unit is then orientated and translated to closely match its

corresponding primitive. The selected unit is placed, such that it partially overlaps with adjacent

units. The overlapped areas are stitched using a cutting method which selects pixels that minimise

the elevation difference between the two units. The results from this system show boundary

artefacts due to only considering horizontal height differences. Also the manual generation of the

database increases the work required by the user and is limited to ridge-based features.

Dachsbacher (2006) adapts a pixel-based texture synthesis technique based on non-parametric

sampling by Efros and Leung (1999). This grows a texture, a pixel at a time, by analysing the

neighbourhood, which is a square window around the source pixel. Evaluating only the height value

of the pixels produces unsatisfactory results because abrupt changes become visually noticeable

when the terrain is rendered and artificially lit. In order to compensate for this, Dachsbacher (2006)

takes into account the horizontal and vertical derivatives and this produces better results. His system

allows the users to place pieces of height-maps on the work surface and have missing data

synthesised. Being reliant on the technique of Efros and Leung (1999), the system suffers from long

computational times but does produce compelling results. Dachsbacher (2006) suggests the use of

better performing texture synthesis techniques, as well as an exploration of a patch-based approach

for further the research. In general per-pixel methods do not adequately preserve underlying feature

structure from the source, and this has stimulated research into patch-based methods.

Saunders (2006) present a design-by-example technique for terrain synthesis. His system utilises

real-world terrain height-maps in the form of Digital Elevation Models (DEMs). Users are asked to

first classify the various terrains according to their characteristics, into a logical library. This serves as

the palette for the synthesis engine. The user uses this palette to describe by-example the

characteristics they desire in their terrain. He achieves this during the design phase using a 2D CAD-

style set of tools. Arbitrarily shaped polygonal regions are drawn in the interface and assigned a

specific palette. Now the terrain is synthesised using a genetic algorithm. This algorithm is launched

multiple times to generate successively higher resolution height-maps (successively finer levels-of-

detail). At each level, the genetic algorithm finds a plausible way of arranging small patches of data

from the respective pallets for each region. To enhance realism a border refinement operation is

conducted, which is itself a subdivision operation controlled genetic algorithm. This replaces straight

boundaries with short segments forming an irregular and hence less artificial-looking boundary. The

output is deemed realistic as each synthesised region is statistically similar to the input files assigned

to the palette. This system is said to produce an unlimited diversity of reasonably realistic terrains

due to the use of a genetic algorithm. However, the actual results are no more compelling than

fractal terrains after physical erosion. Further research is required to improve the visual quality of

the system.

Brosz et al. (2007) present a terrain synthesis by-example system which makes use of two

different terrains to synthesise a new one. The first terrain is termed the base and contains a rough

estimate of large-scale features, such as mountains. The second is the target and contains high-

frequency, small-scale features. The goal of the system is to extract patches of small-scale features

from the target and apply them to the large-scale features of the base terrain. They incorporate an

https://lh3.googleusercontent.com/notebooklm/AG60hOrPxHmiXDgg4qzhuw_mr7s_3tff0iJAM0exlUARfFp7j2W8HAaP9CYv7oZm2RIhNEP_2z65Xy_mcq80_HrygKkbARuprXd8FYeh5-ESzQteqSSyrymdshEe1HqtlBM-nrYfjOxzrg=w574-h543-v0

c0224f5a-170f-437d-a00e-697003d1a72c

automatic method for mapping during patch merging called Image Quilting (Efros and Freeman,

2001). It is a texture synthesis technique and starts by breaking up the base terrain into overlapping

square blocks. Then for each block i in the base, a similar block j is found in the target based on

feature similarity. The extended characteristics of the matched block j are copied back into block i.

But instead of directly copying the data as it would be done in Image Quilting, only the details of the

two patches are copied and linearly blended together. This method of blending does not produce

the boundary artefacts that are common in most patch-based texture synthesis systems. The terrain

synthesis system by Brosz et al. (2007) works well for combining high-frequency details into a base

terrain to produce a higher resolution terrain. However, it does not create a notably different terrain

from the input. As such the overall realism is closely tied to the given base terrain.

Figure 2.9: Illustration of patch placement order. (a) User Sketch. (b) Tree structure from PPA. (c) The root patch is placed first. (d) Breadth-first traversal guides placement of proceeding patches. (e) After feature placement is complete

non-feature patches are placed. (f) Final result. (Image taken from Zhou et al. (2007))

Zhou et al. (2007) present a novel patch-based terrain synthesis system that makes use of DEM

files and produces compelling results (Figure 2.10). The process starts with a user sketch and DEM

exemplar file, the system produces a new terrain based on the exemplar’s features. Figure 2.9

illustrates the algorithm and shows the three main stages:

 A User Sketch and DEM exemplar are provided to the system. These undergo a feature

extraction process which identifies large-scale curvilinear features such as rivers, valleys

and mountains. Zhou et al. (2007) adapt a technique borrowed from geomorphology

named the Profile recognition and Polygon breaking Algorithm (PPA) which was

developed by Chang et al. (1998) to identify such features. The PPA performs a breadth-

first search of the input file and produces a tree structure of features. A PPA tree of the

exemplar DEM is used to produce the candidate patches used in the next stage.

 The second stage controls the matching and merging of feature data into the output

terrain. The PPA tree of the user sketch is broken up into patches used to order

https://lh3.googleusercontent.com/notebooklm/AG60hOpbln2GzkcBxm8t0HF91DrlYPuQq4ZvqBIrzv6wfuWVmkTuRyZXG5AEWwa8FE4E5hVVi-fdZm3awrRYFeZn-vsn8e9lG4ExZnsmVE1nzJEwzZeH8Y-2xQrwLLUB_59OWKrzC3nCOQ=w1380-h946-v0

a6151caf-596f-49a7-93a2-287ea8e04c3c

placement. Each user patch is compared to the candidate patches and the best match is

merged into the output. This process ends when all user features are exhausted.

 In the final stage, ‘holes’ in the output are filled by merging patches from the exemplar

with no strong features.

The procedure for matching and merging is complex and discussed in detail in sections 4.2 and

4.3, respectively. The system is capable of producing a           terrain in approximately 5

minutes on an Intel Pentium 4 2.0 GHz with 2GB RAM. However, there are some limitations of their

system, which are addressed by Tasse et al. (2011).

Figure 2.10: Results of synthesis. (a) User Sketch. (b) DEM Exemplar File. (c) Synthesis output. (d) Rendered terrain. (Image taken from Zhou et al. (2007))

Tasse et al. (2011) build upon work done by Zhou et al. (2007) and present improved patch

merging more suitable to terrain structures that remove the visible boundary seams from

overlapping multiple patches. This modified system forms the basis for the research in this thesis

and we thus provide a detailed description of their system in chapter 4.

2.3 User Control Procedural methods are designed to automate the terrain generation process with minimal user

intervention. However, artists desire some level of control of the process. A trade-off must thus be

made between user control and algorithm autonomy. There are several methods of user control, the

most common being parameter manipulation.

2.3.1 Parameter manipulation

Parameters or variables can be used to control the generation of terrains. For example, in noise-

based generation there are parameters for the amplitude and frequency of the noise function.

Fractal-based methods additionally include a parameter for the roughness, which controls the

irregularity of the generated surface (Fournier et al., 1982). Physics-based systems are also

controlled through various parameters. These include the simulation length, strength of the erosion

functions and others (Musgrave et al., 1989). The genetic algorithm used by Saunders (2006) has a

large number of controlling variables and requires extensive testing to find an optimal set. The

principal drawback to the use of such parameters is that the artists often do not know what the

effect of changing them will have on the resulting output. Achieving a desired terrain design is likely

to be an exercise of trial and error since the parameters are generally unintuitive. An improvement

to parameter manipulation is the use of images to control the system.

2.3.2 Image-based control

Manually designing terrain models can be achieved by using existing 2D painting programs, for

example Terragen (2013). Procedural methods can also be controlled through using images.

Schneider et al. (2006) make use of images to represent the fractal base functions by providing a

painting interface. Their system provides immediate feedback to the user, providing a far more

intuitive form of control over arbitrary parameters. Saunders (2006) provide an authoring interface

where the user describes their desired layout of terrain by painting regions using their defined

palette. This is just a rough idea of where data from different terrains is to be placed during

synthesis, rather than the location of specific features. Zhou et al. (2007) make use of a user-defined

image which contains a simple sketch of the terrains layout to guide the synthesis. The image is

made up of either black or white painted strokes which correspond to valleys and ridges, with a grey

background indicating no preference. While this system can be used to specify the location of

specific features, there is no mechanism for controlling the specific height. Sketching systems are

primarily used by people to represent a rough design or layout. This provides a more intuitive system

for controlling the generation of terrain.

2.3.3 Sketching

A sketching system is best suited for when the user has a rough idea of what the final terrain

should look like. User-sketched strokes are used to specify the shape of the desired landscape.

Cohen et al. (2000) present an early form of terrain sketching in their system, Harold. Users design

and create hills and mountains by drawing 2D strokes in screen space. The endpoints of the stroke

are used to create the projection plane. This is used to project the stroke into world space, creating

the silhouette curve. The resulting curve forms a shadow, with points near to it being elevated based

on their distance to the silhouette curve. However, the depth of the mountain (perpendicular to the

screen) is constant with its cross-section being parabolic. This creates mountains that are unnatural

in appearance when viewed from different angles. Watanabe and Igarashi (2004) improves on

Harold by adjusting the depth and cross-section shape according to the shape of the stroke, resulting

in more natural looking terrain. Gain et al. (2009) present a more complex terrain sketching system

based on Cohen et al. (2000). Users control the location and shape of landforms by drawing 2½D

silhouette, shadow and boundary curves. These curves form constraints for a fast multi-resolution

surface deformation system. During this process wavelet noise characteristics (Cook and DeRose,

2005) are analysed and applied to the resulting terrain. The synthesis system is designed to faithfully

match the user’s strokes rather than just approximating them, which differs from previous work. The

system by Gain et al. (2009) offers a high degree of user control, allowing the user to intuitively add

features such as cliffs and indentations. However, the wavelet noise fails to add small-scale natural

features such as erosion patterns to the terrain. Additionally the deformations are distinctly visible

when applied to an existing natural terrain. These limitations result in less realistic terrain when

compared to other systems, such as Zhou et al. (2007).

More recent work by Tasse et al. (2014a), (2014b) presents a new method for editing of terrains

by sketching from a first person perspective. Most sketch-based terrain systems are controlled from

a top-down viewpoint, which makes it difficult to accurately describe the skyline that would be seen

from the ground. The system makes use of an existing terrain, which is rendered in a 3D

environment that the user can move about freely. The user will then sketch strokes to infer where

terrain features should be present. These strokes are then ordered, front to back, by inferring their

relative depth from the height of their end-points and detected T-junctions. Now features from the

terrain, such as silhouettes and ridges, are detected. By deforming existing features the nature of

the terrain is preserved as no extra features are created. The user strokes are now matched to one

of these features and a specific deformation algorithm is applied and ensures that small-scale

feature data is preserved. After the initial deformation the system checks that the newly modified

terrain does not occlude any of the user strokes. If this issue occurs the terrain undergoes further

deformation that will lower part of the terrain to remove the occlusion. This system allows a user to

easily personalise an existing terrain and also preserves the style and realism.

2.4 Discussion Based on our evaluation in this chapter, we conclude that fractal terrains lack realism while

physical simulations are complex and expensive to run without extensive GPU enhancement. Both

approaches also provide limited user-control. Table 2.1 provides a comparison of the three main

categories of terrain generation. The “speed” entry is based on the time taken to synthesise a terrain

with a size of           in pixels, this is a rough estimate of the speed as the figures would

directly relate to the hardware being used. User-control is an expression of how easy the process is

to control and realism compares the characteristics of the output terrain with real landforms. A

summary of the main limitations for each category is also provided. This table shows that texture-

based methods provide a high degree of realism coupled with a fair degree of user-control. We

believe that realism and user-control are more important than speed of synthesis, particularly since

these algorithms have not been fully optimised and there is thus room for further improvements.

The content creator is more likely to wait for a longer synthesis to complete if the end result is closer

to their design requirements.

Speed User-Control Realism Main Limitations

Fractal-based Very fast Low – High* Low  Absence of natural erosion

 Non-intuitive control parameters

 Pseudo-random output terrain

Physics-based Thermal: Fast

Hydraulic:

Low Thermal: Medium

Hydraulic:

 Complex to implement

 Requires a base terrain

 Minimal user control

Texture-based Slow Medium High  Limited user control

 Output dependant on number of input terrains (exemplars)

Table 2.1: Comparison of terrain generation methods. *A high user-control system is provided by Gain et al. (2009)

Fractal-based techniques can run very quickly on modern CPUs but their output is unsuitable for

applications where highly detailed or realistic-looking terrain is required. The synthesis is controlled

with a set of parameters that do not clearly indicate their direct effect on the output, leading to a

very low level of user control. Physics simulations can be used to enhance the realism of a base

terrain by adding natural weathering effects. However, this also suffers from the same minimal user

control and as a consequence often relies on the quality of the base terrain. The complex

simulations impact on the performance of the system significantly, although recent work has

focused on accelerated algorithms using GPUs. Texture-based methods borrow techniques from

texture synthesis and make use of real landscapes as the source of their data, this makes the

generated terrains highly realistic. When the synthesis is controlled through a sketching or painting

interface, the level of user control is quite high and intuitive. The runtime is acceptable given the

preference for realism but recent advances in GPU acceleration have made these methods even

more appealing.

Natali et al. (2012) present a state-of-the-art report which evaluates a number of different

implementations for terrain generation, which we refer interested readers to. Based on our

evaluation of these terrain generation schemes we decided to extend the work done by Tasse et al.

(2011). This decision is based on their improvements to work done by Zhou et al. (2007), particularly

with the improved quality of merged patches. We present a detailed analysis and description of this

system in Chapter 4 along with our proposed extensions to their work. The next chapter contains

background information necessary to understand use of a Graphics Processing Unit (GPU) to reduce

synthesis time.

https://lh3.googleusercontent.com/notebooklm/AG60hOpkqQMHWtz_oc9qYxmPFSFg8rYqM7BMso39oXvL3287kaTCUUylqJlLf6lmBkSi25qBLF_P_jjiNVmYf_bGRFrhWD7RfJB8YBelBTi4b3GeRnf5IhnLm7e-xI8FqStWRfPpgaUTxg=w970-h420-v0

0e163b29-d197-4a51-8f99-fb220e89b16c

3 Background: GPUs & NVIDIA CUDA

Modern Graphics Processing Units (GPUs) are made up of a large number of simple Single

Instruction, Multiple Data (SIMD) processors that can be harnessed for general purpose computing.

Programming GPUs has been made easier with the development of application programming

interfaces (APIs) such as NVIDIA’s Compute Unified Device Architecture (CUDA). GPUs have evolved

into highly parallel, multithreaded, many-core processors with tremendous computational power

and high memory bandwidth – Figure 3.1. NVIDIA’s flagship GPU, the GeForce GTX680 attains a peak

theoretical performance of 3090 billion floating-point operations per second (GFLOP/s), which is

approximately 10X faster than Intel’s flagship Sandy Bridge 3770K processor which peaks at 294

GFLOP/s. AMD’s flagship GPU, the Radeon HD 7970, peaks at 4300 GFLOP/s representing a strong

contender in terms of performance.

This chapter introduces the concepts required to understand programming on GPU devices and

focuses of NVIDIA’s CUDA (NVIDIA, 2013a, NVIDIA, 2013b). In section 3.2.1 we motivate our choice

for CUDA over other alternatives. The remainder of the chapter provides information on the

programming model and execution pipeline for NVIDIA GPUs.

Figure 3.1: (a) Floating-Point Operations per Second and (b) Memory bandwidth, for both CPU and GPU (NVIDIA, 2013b). This shows the large difference between GPU and CPU performance leading to the use of GPUs for accelerated

computation.

3.1 GPUs and Parallel Programming Graphics accelerators were the precursor to what we now call GPUs and have been in use in

computer systems since the early 1980’s principally to accelerate drawing operations. For the last

two decades the development of graphics hardware has been extensively driven by the gaming

industry since games required more vector-parallel processing power than typical CPUs could offer.

CPUs feature a complex processing architecture that cannot keep up with the large number of

fragment operations required to render complex 3D graphics efficiently. This led to the development

of specialised hardware that contains large numbers of simple processing units designed to

efficiently process large amounts of fragment data. Early GPUs (1990s) featured a fixed-function

rendering pipeline where specialised hardware components were dedicated to individual stages. The

https://lh3.googleusercontent.com/notebooklm/AG60hOrLJ2rQSBsQv7o4paI1XZUP4A1kXN3GSvsz1Fp8WVDYy2kVZ75DRpBJi9pMNytCE2xnq3kzNHauvEt87OSKoB6qGicH5MOtHLMxtxk3CESFkKNe3CZAwYraj1jOnkdh4cw2CruhGg=w800-h300-v0

be9b1665-ecf4-4bf3-95df-8ff9adab0764

user controlled the rendering process by configuring parameters such as vertex positions and

colours of vertices or lights. Because rendering functions were predicated on the availability of

compatible hardware, the use of a fixed-function pipeline evolved into a programmable one in 2001.

This enabled the control of vertex and fragment operations through small programs called shaders.

GPUs follow a different development philosophy from CPUs and focus more on parallel computation

over reduced memory latency. For example a CPU would process vertices and fragments

sequentially whereas a GPU can process multiple elements at the same time. Although modern CPUs

contain multiple cores leading to a small degree of parallelism, however, this is dwarfed by the large

number of cores GPU devices contain. GPUs adopt a SIMD approach, which utilises a large number

of simple processors to execute the shader programs in parallel. Shader languages such as the

DirectX High-Level-Shader-Language (HLSL), NVIDIA CG and the OpenGL Shading Language (GLSL)

can be used to write shaders for the GPU and facilitated the development of more advanced

rendering techniques than allowed by a fixed-function pipeline, such as bump-mapping. These

languages were initially used to write shaders for general purpose GPU computing but they are

inherently designed for graphics operations, such as filtering and rendering. The use of specific

graphics terminology made programming on GPUs inaccessible to typical programmers interested in

more general acceleration and this led to the development of a high level APIs such as CUDA. CUDA

was released in early 2007. It gives developers access to the virtual instruction set and memory of

these devices, allowing for general purpose GPU (GPGPU) programming. This followed NVIDIA’s

launch of the GeForce 8 series in 2006, which featured a generic stream processor enabling the GPU

to act as a more generalised processing device.

Figure 3.2: GPU devotes more transistors to data processing (NVIDIA, 2013b). There are significantly more Arithmetic Logic Units (ALUs) dedicated to the control and cache units.

Modern GPUs dedicate more transistors to data processing rather than data caching and flow

control (shown in Figure 3.2). This explains the large discrepancy in floating-point capability between

the CPU and GPU (Figure 3.1). Because there are fewer transistors dedicated to caching, memory

latency typically becomes the main bottleneck of these systems. Consequently, GPUs are well-suited

to address problems that can be expressed as data-parallel computations. This means the same

program is executed on many data elements in parallel with a high arithmetic intensity. Arithmetic

intensity refers to the ratio of computation operations to memory operations. Unlike CPUs, GPUs

have a parallel throughput architecture, which emphasises executing many concurrent threads

slowly rather than a single thread quickly. Applications that require processing of large data sets can

use a data-parallel model to speed up computation, specifically those that have a high arithmetic

intensity. Apart from image processing and 3D rendering, other algorithms ranging from physics

simulations to computational finance can be expressed as data-parallel computations and greatly

accelerated. This has led to a great deal of research into exploiting GPU devices. The programming of

these GPGPU devices is achieved through the use of APIs such as NVIDIA’s CUDA but other

alternatives exist such as OpenCL (Khronos, 2013). These are discussed in section 3.2.1.

3.2  NVIDIA CUDA NVIDIA released the initial version of its CUDA SDK in early 2007. This has evolved considerably

over the years but remains limited in availability, as only NVIDIA GPUs are supported. CUDA

programs are mostly written in C/C++ which gets compiled by NVIDIA’s nvcc compiler. Other

languages such as Python and Java are supported by 3rd party wrappers. The CUDA API is better

suited to general programming than shader languages and has several advantages including

scattered memory access, exposed shared memory, and full support for integer and bitwise

operations. CUDA has continued to evolve as the underlying hardware architecture evolves and new

features are implemented. The use of these new features is linked with the Compute Capability (CC)

of the device, with higher level devices being fully backward compatible. At the start of our research

the GeForce 5xx series of devices were available with Compute Capability 2.0. As such further

discussion in this chapter is focused around the features available for this version.

3.2.1 Motivation for using CUDA over alternatives

NVIDIA CUDA is not the only high-level API for GPGPU programming. There is the now deprecated

ATI Stream (ATI, 2013) and the open-source OpenCL (Khronos, 2013). ATI released the first version

of Stream at the end of 2007 but subsequently deprecated this in favour of development in OpenCL.

It was developed explicitly for their Radeon series and was restricted to AMD GPUs. OpenCL, in

contrast is an open-source cross-platform framework that allows for execution of parallel programs

across heterogeneous hardware platforms. The support and performance of OpenCL has improved

steadily since its first release in 2008. However, despite all the improvements, CUDA paired with

compatible NVIDIA GPU still has a performance advantage over an OpenCL implementation (Karimi

et al., 2010, Du et al., 2012) and also has better support for 3rd party libraries. Intel recently released

Xeon Phi (Intel, 2013), a hardware coprocessor comprised of many-core processors, set to rival the

use of GPUs for parallel computing. It went into first production in late 2012 and the initial variants

have a significantly higher price point than consumer grade GPUs and as yet not much support is

available for development. We expect this to change over the coming years but was not available for

consideration during our research.

This left a choice between an OpenCL and CUDA implementation for GPU acceleration in this

work. During the planning stage for our research, CUDA provided better performance and had

greater support than OpenCL. This led to our decision to develop a CUDA based solution for GPU

acceleration.

3.2.2 CUDA Programming Model

CUDA supports a C-like syntax which eases the transition for existing programmers without

requiring a graphics background. This language is also easily interoperable with standard C and C++

code. CUDA code gets compiled into a special format that is deployed at runtime to the CUDA-

capable device where the code is executed by thousands of lightweight threads. These threads are

divided up amongst the devices many compute cores. CUDA claims to run on any CUDA capable

https://lh3.googleusercontent.com/notebooklm/AG60hOp9vIhXmvyorPMOEzK4LV_Qz5nxQcNou38zId3rnGQcEqUBYI2FJvnYpMo9uT2n0I57yWfG7wENISpPh7-4p-3SpRlhOKQgm_c5YGHlOfE7KthS96oJXHyvxv0z3qFyEyZbRNaX=w600-h580-v0

139ab30a-641d-4a1d-a761-ad011634ab1c

device, which is not entirely true. New hardware features are introduced in the form of the devices

Compute Capability (CC). This is fully backwards compatible, which means that as long as the device

has the minimum required CC, the code will execute.

There are two hardware abstractions that CUDA defines: the device, which is a CUDA capable

GPU, and the host, which is the computer to which the device is connected. The execution of code

on the device is initiated by invoking a C-like function called a kernel. Before the kernel is called, data

needs to be transferred to the device’s memory. The kernel then executes simultaneously on the

many threads of the CUDA device, processing the data. After execution is complete the data can be

transferred back to the host. This is illustrated in Figure 3.3.

Figure 3.3: CUDA Processing Flow. (1) Data is copied from host to device; (2) Kernel is executed; (3) Data is processed in the many threads on the GPU; (4) Result is copied back to host.

There are two important high-level aspects to CUDA programming: memory management (how

data is transferred between host and device) and the execution pipeline (how data is processed on

the device). CPUs contain multiple levels of cache which hides memory access latency and ensures

the processor is always being utilised. This is not the case for GPUs, which rely on many lightweight

threads and instantaneous context switching to swap out threads waiting for memory and swap in

threads ready for processing. We now provide further explanation of these aspects.

3.2.3 Execution Pipeline

The key to understanding the CUDA execution pipeline is to understand how the GPU hardware

and software components interact and how instructions are scheduled. NVIDIA was careful to design

a software model so that it mirrors the characteristics of the underlying hardware. This means the

programmer can design their system such that it maps as closely as possible to underlying hardware.

Multiple schedulers provide fast switching of the many executable threads. The threads are grouped

into blocks and blocks are arranged into a grid. The GPU contains a number of Streaming

Multiprocessors (SMs) that are assigned a number of blocks from the grid. The SM is responsible for

scheduling the execution of the threads for the blocks it is allocated. However, the user is able to

https://lh3.googleusercontent.com/notebooklm/AG60hOoYE4W-b8rqt4nQJDEBcDiQJTWcsm0rkPOFGjYWnSgBvZEiECQa2WQ10l_78JVGq9lGXK4iwjBu67DGjsH6edx7jRbKCjHzkZ_l8BlMD1wixvVQDsShN_xZuC2Arv66lulUOS537g=w605-h462-v0

48f9b970-b061-4659-a052-f67dbeca62ea

fully control the thread and block layout. Appropriate design of these layouts can allow the

programmer to utilise the GPU more efficiently.

Software – The Grid, Blocks & Threads:

As of compute capability 2.0, the CUDA code is executed on a GPU device by launching a kernel,

which is a C-like function call. Parameters to this include the dimensions of the grid and block and

number of threads per block. The code within a kernel is divided up into smaller logical units called

blocks and has up to three-dimensions with maximum number of (                 ) blocks

and are collectively known as the grid. In turn each of the blocks comprises of up to three-

dimensions, with a maximum of (            ) and up to      resident threads per block.

Resident threads refer to how many active threads can be instantiated within the block based on the

available resources for storing the threads context data. An example of this layout is shown in Figure

#### 3.4. Each block on the grid is executed independently of the others with no guarantee on the order

of block execution and no mechanism for inter-block communication. The GPU scheduler controls

which block is being executed based on its availability of warps (discussed later – Block, Warp &

Thread Scheduling) ready for processing and enforcing cooperation could result in blocks being

stalled or a case of system deadlock. However, threads within the same block can communicate

through the use of shared memory (see section 3.2.4) and synchronisation (discussed later – Barrier

Synchronisation). There are a maximum of eight resident blocks per SM, and each SM is also limited

to a maximum of      schedulable threads. The programmer needs to adhere to these constraints

in order to maximise device throughput. By adapting the computational problem to utilise all of the

available threads, the programmer can ensure the GPU is fully saturated with work.

Figure 3.4: Schematic overview of the Grid-Block-Thread layout (NVIDIA, 2013b). The kernel is loaded onto the device which is comprised of the blocks and threads.

https://lh3.googleusercontent.com/notebooklm/AG60hOo2FWaapK2uyf8Kqv6iSAQ8vabVitySXLcpHrcZyj8CKdLK_46snziKBP1OLPDSsoJkgU8YXT4Vwly9f9tVboXNqDnEEehNTZfG1bk0PbmZOwXgXkcZ51zdejUn4gnPA0Ofczzdyw=w789-h819-v0

046a4bdd-1c50-4639-a045-b4fbbbbad178

Layout & Indexing of Blocks & Threads

Blocks can be arranged in one, two and three-dimensions on the grid. A block supports up to

three-dimensional thread layouts. An example is presented in Figure 3.5 which features a grid with

dimensions of (   ), which holds blocks of dimension (   ). This gives a total of 6 blocks within

the grid, each with 12 threads. Grids support a maximum of        blocks, but since a single block

is executed in a single SM, these serve more as a programming convenience for organising the

problem. The dimensionality of threads within a block is more important in maximising throughput

of the device. However, the layout of threads within a block can have a significant impact on

performance.

Figure 3.5: Example Grid/Block/Thread Indexing for a 2D grid and block layout (NVIDIA, 2013b).

Thread indexing is a vital part in controlling the specific execution path and specifying what data

the thread is to operate on. When a kernel is executed, the defined number of threads will all

execute the same code. CUDA provides kernel variables that can be used to determine the index of

the thread and block, namely threadIdx and blockIdx respectively. Along with these are blockDim

and gridDim, which store the defined dimensions of a block (threads in a block) and grid (blocks in

the grid). Listing 3.1 provides a simple example of a CUDA kernel which squares the values of a

(       ) array. The kernel invocation is provided in Listing 3.2 where the dimensions of the

blocks and grid are defined. In this example, blocks have a dimension of (     ) which produces a

total of 1024 threads. In order to evaluate all values in the array, a grid size of (     ) blocks is

required. CUDA operates on one-dimensional arrays, meaning multi-dimensional arrays need to be

flattened. After completion, all the values in the array will be squared.

1 __global__ ExampleKernel(float* data, int data_w) 2 { 3     // Use index of thread to work out index in array to access 4     int idx_X = threadIdx.x + (blockIdx.x * blockDim.x); 5     int idx_Y = threadIdx.y + (blockIdx.y * blockDim.y); 6  7     // Calculate the flattened index (1D) 8     int idx_Flat = idx_X + (idx_Y * data_w); 9  10     // Read the value from the data array 11     float val = data[idx_Flat]; 12  13     // Write back the value squared 14     data[idx_Flat] = val * val; 15 }

Listing 3.1: Example of a CUDA Kernel. This kernel takes a flattened square array of size w and squares its values.

16 int main(void) 17 { 18     ... 19     // Kernel invocation 20     dim3 threadsPerBlock(32, 32); 21     dim3 blocksPerGrid(16, 16); 22     ExampleKernel<<blocksPerGrid, threadsPerBlock>>>(data, data_w); 23     ... 24 }

Listing 3.2: Example Kernel Invocation. This is the sample code which will launch the CUDA kernel defined in Listing 3.1. The threads-per-block and blocks-per-grid are defined and used in the call. This also assumes initialisation of data for the

array on the device.

Hardware – The GPU, Streaming Multiprocessors & Cores:

The CUDA software model arose directly from the design of the CUDA hardware. A CUDA capable

device is made up of a collection of Streaming Multiprocessors (SMs), which contain the CUDA cores,

memory infrastructure and control units as shown in Figure 3.6. There is a direct mapping between

the software and hardware models: grid to GPU device, block to a SM and a thread to a single core.

Each SM includes 32 Scalar Processors (SPs) or CUDA cores, 16 Load/Store Units, 4 Special

Function Units (SFUs), Dual Warp Schedulers and Dispatch Units, 32k 32-Bit registers and 64KB of

combined Shared Memory and L1 Cache (NVIDIA, 2013c). The SM is assigned a group of blocks (by

the GPUs scheduler) which it operates on in turn, with the threads defined by the block being

executed on the cores. The 16 load/store units allow for source and destination addresses for 16

threads to be calculated per clock-cycle, thus requiring only two cycles to access memory for all the

cores. The 4 SFUs execute transcendental instructions such as sin, cosine, and square root. Execution

in the SM is performed in a group called a warp (discussed later – Block, Warp & Thread Scheduling),

with each warp being comprised of 32 threads. Each SFU executes only one instruction per thread,

per clock, requiring 8 clock-cycles to complete a warp. These SFUs are decoupled from the dispatch

unit allowing it to issue instructions to other execution units while the SFUs are occupied. Dual Warp

Schedulers select an instruction from each warp and issue them to a group of 16 cores, 16

Load/Store units, or 4 SFUs which are executed independently. The fast on-chip memory provides

limited L1 cache and method for threads to cooperate. The architecture allows this memory space to

https://lh3.googleusercontent.com/notebooklm/AG60hOoyFdhTWzfMFNciEdFu4Z9UTzOl7MwFf5SzJDSWF0KplACHCl69zxlUdQaYyJ-_Qo4CNtjXKrDmhUewL4q7PGuvZMIdxzR3BKHdFjqehc3sy8X2PvIKk2Z75KsefXwMFEoOabFszg=w368-h600-v0

de5a4792-9b96-4e8e-815a-890999db3fda

be divided into 16KB of L1 cache and 48KB shared memory, or 48KB L1 cache and 16KB shared

memory depending on the requirements of the programmer.

Figure 3.6: Architecture of a Scalar Multiprocessor unit for a GeForce GTX 580 (Fermi) GPU (NVIDIA, 2013c). This represents all the command, control and cache units present.

CUDA is designed to operate with any number of SMs, as the total number of blocks is divided

amongst all available SMs. This produces a highly scalable model since adding more SMs to a device

increases computational throughput seamlessly. This is usually the principal difference between low-

end and high-end devices. High-end GPU devices usually carry a larger number of SMs and larger

amount of device memory.

Block, Warp & Thread Scheduling

When developing the kernel code, the programmer statically defines the number of blocks,

threads-per-block along and the dimensions of both. These are used when the kernel is invoked for

the hardware’s block scheduler to divide up and assign blocks to the devices SMs. Each SM breaks up

its assigned blocks into groups of 32 threads called warps. Each SM can have a maximum of 8

resident blocks and 48 resident warps, with others waiting in a queue. The number of warps per

block is calculated by taking the number of assigned threads (t) and dividing by the warp size (32).

Fermi devices have two warp schedulers which allows for each to select an instruction to be run

concurrently and independently. The hardware is mapped into two sets of 16 cores, 16 load/store

https://lh3.googleusercontent.com/notebooklm/AG60hOpLq288yQSV8RNwdG1dG8M3uVkX4gDH56nsA2_2L-XAyDlQ1baeceZOqXhkwBb9aXbGbaHXniPnj4yfgWudlxBCOgAQN0JhrAyHOPpamhYr3Jtt7-d14D7-yYez2kw4_lctqwoD=w1094-h732-v0

f4169274-f6ef-4069-806a-6fab19337339

units and 4 SFUs. Each of the warp schedulers can utilise one of these items at any given time. There

is also no requirement that warps need to be from the same resident block. However, each of the

instructions belonging to a particular warp need to be executed in order. Each resident warp ( ) has

its own context (instruction counter and registers). This means that instruction 7 of warp    can

execute, then on the next clock-cycle instruction 3 of warp   can execute, followed by instruction 8

of warp    immediate after. This instantaneous context switching differs from that of a CPU which

has a heavy-weight thread context, and incurs a delay due having to copy register data in and out of

the CPU. CUDA allows for this instantaneous switch, as only a switch to the next scheduled warp’s

instruction pointer and registers is required, both of which are store in on-chip memory.

Figure 3.7: Example of Fermi's Dual Warp Schedulers. Each scheduler is assigned a group of warps; the first scheduler is responsible for warps with positive ID and the second for negative IDs. At each clock-cycle both the schedulers select an

instruction to execute for a particular warp. Since two warps are run concurrently, each works on only half its instructions, requiring two cycles to complete. (NVIDIA, 2013c)

Having the ability to instantaneously switch between warps is used in hiding memory latency, so

that when a warp makes a memory request it can be switched out for one awaiting execution. This is

however, predicated on the fact that there is another warp available for execution. With this in

mind, the programmer should design the system to provide enough warps in order to saturate the

An example of how the dual warp schedulers work is presented in Figure 3.7. During the first

clock-cycle, instruction 11 from warp 8 and instruction 11 from warp 9 are executed. At the next

clock-cycle, instruction 42 from warp 2 and instruction 33 from warp 3 are executed. This shows that

two different instructions from two different warps are executed concurrently in the hardware. But

since there are 32 threads in a warp and only 16 execution units, both sets of instructions execute

twice over two clock-cycles, one for each batch of 16 threads.

Flow Control & Code Divergence

Flow control refers to the use of a control instruction (if, switch, do, for, while) in the execution of

a kernel. Using these controls can significantly impact the instruction throughput by causing threads

within a warp to diverge (follow different execution paths). In this case, all the required execution

paths are serialised and evaluated by all threads in the warp, increasing the number of instructions

executed. This means for an ‘if’ statement, the ‘true’ branch is evaluated first then the ‘false’ branch.

They are not executed concurrently as is the case with traditional multi-core CPU systems. Threads

that diverge down the ‘true’ branch will ignore execution when executing the ‘false’ branch, and the

‘false’ threads ignore execution in the ‘true’ branch. After all the paths have been followed, the

threads converge back to the same execution path.

However, if all the threads in the same warp follow the same execution path (i.e. all threads

evaluate to ‘true’ or ‘false’), then only the required branch is executed. If even a single thread

diverges within a warp then both branches will be executed. This is a result of CUDA using a lock-

step mode of execution within a warp, where all threads can only perform the same instruction at

any given clock-cycle. The programmer needs to take this into account when using flow control

statements so as to avoid the possibility of divergence.

## Barrier Synchronisation

Synchronisation within a kernel allows the kernel to halt until all threads in the same block reach

the barrier. This technique, together with shared memory, allows for cooperation of threads within a

block. Synchronisation is limited to block level and, unlike multithreading on a CPU, it is impossible

to synchronise all threads across the device. Synchronisation is useful when using threads to load

data from global memory into shared memory before computation so as to make sure data is

available. A downside to using synchronisation is that threads blocked at a barrier are idle, which

reduces the overall performance of the device.

3.2.4 Memory Hierarchy

Since GPUs focus more on parallel data processing rather than memory caching, maximising the

memory throughput is essential to maximise performance. For example, a GeForce GTX 580 has a

peak memory bandwidth of 192.4GB/s but the host-to-device transfer over the PCIe x16 Gen2 bus

peaks at a comparatively low 8GB/s. This means that transfers to or from the device should be

limited and calculations should be executed on the device, even if they would be faster on a CPU as

the memory transfer performance penalty would dominate processing time. For example, before

squaring the values of a 10,000 element 2D array of floats, 0.046s is required to transfer data to

host, calculated as follows (   [       [         ]]). On top of this the computation cost

for executing on the CPU needs to be added, which further increases the total computation time.

However, processing the array on the GPU only requires 0.0019s of compute time. This represents a

24x performance drop in transferring the data alone. CUDA has access to much of the memory

present on the GPU. The scope and characteristics of these memories are summarised in Table 3.1.

https://lh3.googleusercontent.com/notebooklm/AG60hOqySxTUzd6kVwaj62Y8cgM7vu1-VFP4LRFnBdm71TUJZdBO8d0jeA1F4LrjCIJB9OuWTFZsGNU1VuGIecAkfjvU8h0Xswwp3Qf_Y-kurW7qZeIibCWNBgxpalSJEvGordqILv7UQw=w927-h1074-v0

1a78c59e-ca7b-4b7b-ac0a-c15d1c47b386

Figure 3.8: Memory Hierarchy. Each level shows the scope of the different types of memory. Local memory is restricted to a single thread. Shared memory can be accessed from all threads in a single block and global memory is accessible

between one or more grids. (NVIDIA, 2013b)

There are two fundamental categories of memory: on-chip and off-chip. On-chip memory is very

fast with near zero latency for access but is very limited (64KB per SM). It includes both registers and

shared memory. Off-chip memory has a far higher latency, but is much larger (typically up to 3GB)

and includes global, local, constant and texture memory. Every level of the execution pipeline has a

corresponding memory space, as shown in Figure 3.8. Each memory type has a specific use and must

be assigned based on the problem being solved in order to maximise performance of the system.

The different memory types are discussed below:

There are 32k 32-Bit registers located on each multiprocessor that get shared between all its

cores. Accessing registers consumes zero extra clock cycles per instruction, but delays can occur if

there are read-after-write dependencies and bank conflicts. A bank conflict occurs when two or

more addresses of a memory request fall within the same bank. Information on bank conflicts can be

found in section 3.3.1.The read-after-write latency is approximately 24 cycles, which is how long a

thread is required to wait before accessing that register again. This latency can be hidden if at least

threads are active in the multiprocessor with   being the number of cores present. The

hardware scheduler attempts to optimally schedule instructions to avoid register bank conflicts and

works best when using a multiple of 64 threads per block. Should there not be enough registers in

the SM or if the value is too large to store, the data will spill over into local memory.

## Type Location Cached Access Scope Lifetime

Register On-Chip N/A r/w Thread Thread

Local Off-Chip *Yes r/w Thread Thread

Shared On-Chip N/A r/w Block Block

Global Off-Chip *Yes r/w Global Application

Constant Off-Chip Yes r Global Application

Texture Off-Chip Yes r Global Application Table 3.1: Device Memory Summary. *Cached on devices with Compute Capability 2.0 and up.

Local memory resides in off-chip device memory and suffers from high latency and low

bandwidth. Up to 512KB may be allocated per thread. The compiler automatically assigns local

memory for large structures; arrays that would consume too much register space and “spilled over”

registers in the case the thread runs out of its allocated amount. Local memory accesses are always

stored in L1 and L2 cache and should be avoided as the latency is still very high.

Shared memory is located on-chip and can be accessed by all threads resident to the block

currently loaded onto the multiprocessor. It has a much higher bandwidth and lower latency

compared to global memory. Each SM has 48KB which is divided up into 32 distinct 32-Bit memory

banks that can each be accessed simultaneously within a warp. This means that 32 threads can each

make a request to shared memory without bank conflicts, provided multiple requests do not fall in

the same bank. Shared memory can either have all the threads read from the same memory address

or all read from unique addresses to avoid a bank conflict. Shared memory is useful for problems

that require threads to co-operate or when a larger problem is divided up between the threads to

solve a smaller component. The use of shared memory can greatly improve system performance

over using slower global memory.

Global memory is the largest pool of memory that is located off-chip and has a size up to 3GB that

can be read from and written to by all threads, blocks and the host system. Global memory is the

device equivalent of the host’s RAM. The downside to using global memory is the high latency

associated with reads and writes. This high latency can be offset if a coalesced memory read is

performed. Coalesced access occurs when all the threads within a warp (32 threads) can be

combined into as little as one memory read. Global memory consists of rows of 128-Byte (128B)

aligned segments that are accessed in 128B transactions. This allows for all threads in a warp to

access adjacent 4B words (float) in a single request provided the memory is aligned to a single cache

line, even if the reads are non-sequential. However, for misaligned access, two separate requests are

required to read all the memory for the threads. These different patterns are represented in Figure

#### 3.9. The correct use of coalesced memory access for reading or writing data greatly improves

memory throughput on the device.

https://lh3.googleusercontent.com/notebooklm/AG60hOr7PDe4pGa3RRzEEeGercqHSKJoowimo61Fa1WBTKZPhd29O14dLaVon-qkTDOim2jeQfYcvZvdW5G1X9wB2JG21ODuZDrwCP53t31pOnEKNESJW4TFhi5ZZ2pdjnFuOnPAEHUgow=w500-h374-v0

d978bbfe-06cc-476b-a260-bb483e44c3f9

Figure 3.9: Memory access pattern for coalesced reading. Both (a) and (b) require a single 128B transaction whereas (c) requires two 128B transactions, which decreases performance to 50%. (NVIDIA, 2013b)

Constant memory is read-only off-chip memory stored in constant cache with a total size of 64KB.

It is mostly used by the host to set constant values that need to be read across multiple kernel

executions. Reads from constant memory cost one request from a register but only when all threads

are reading the same address. If different addresses are requested by threads within a warp then the

requests get serialized and the cost scales linearly with the number of different addresses read.

Texture memory is a read-only off-chip memory, which is part of global memory. The difference

between this and global memory is that it is spatially cached in one, two or three dimensions. With

the data being spatially cached and a lookup operation is performed on a memory address, the

adjacent memory locations, both vertically and horizontally, are also fetched and stored in L1 cache

on the device. This speeds up access when an operation on a given location requires the data stored

in neighbouring locations, such as image filtering. However, in the event of a cache miss the cost is

one read from device memory, negating the advantage of using texture memory. It is primarily used

when accessing two-dimensional data-structures where adjacent memory reads are required for

computation.

These memory types each have their advantages and limitations. Selecting the right memory type

for the specific problem at hand is essential in obtaining higher performance from the device. Higher

memory throughput means that less time is spent waiting for data.

3.3 Performance considerations In order to exploit the high computational power of modern GPU devices three key optimisations

are required (NVIDIA, 2013a):

3.3.1 Maximise memory throughput

Bank conflicts occur when two or more threads make an access request to the same memory

bank. If this occurs, the request gets serialized and the hardware splits the request into as many

conflict-free requests ( ) as needed. This decreases memory throughput by a factor of   as a result.

However, if all the threads request the same address then a broadcast is performed. This is when the

address is read only once and broadcast to all the requesting threads. Bank conflicts should be

avoided to maximise memory throughput.

The amount of data being transmitted between the host and device should be minimised as the

bus transfer speeds are significantly slower than transfers within the device as shown in section

###### 3.2.4. Another consideration is to select the correct memory type for the problem. Global memory is

larger but has the highest latency, although this can be reduced by using memory coalescing. Shared

memory provides fast access provided there are no bank conflicts and that the data is small enough

3.3.2 Maximise parallel execution

The ratio of parallel to sequential computation must be balanced. Asynchronous calls should be

used where possible to enable concurrent execution between the host and device. Asynchronous

calls allow the host to transfer data to the GPU and continue processing. Once the data has been

transferred the kernel can be executed and the host can continue processing and initiate more

asynchronous calls. This allows for both devices to be continuously executing data in order to

maximise performance.

3.3.3 Maximise instruction throughput

The use of arithmetic instructions with low throughput should be avoided as larger throughput

hides memory latency. Branch divergence within a warp must be avoided as threads in a warp

cannot execute different code concurrently, requiring multiple passes to execute all branches.

Reducing unnecessary instructions and optimising away synchronisation points will also increase

instruction throughput.

3.4 Summary Three key optimisations are required to exploit a GPUs full potential:

 Maximise memory throughput

 Maximise parallel execution

 Maximise instruction throughput

All the required hardware details should now be understood so that the implementation chapters

can be easily followed. Both chapters 6 and 7 include sections on our specific GPU implementations.

It is important to understand that GPU programs are defined by the launching of kernels, which

when executed are split up into many threads, blocks and grids. Knowledge of how these threads are

executed on the hardware is important in order to optimise the system to maximise performance.

Since different hardware generations provide different feature-sets, the compute capabilities must

be understood. Our system was designed around an NVIDIA GTX 580, which has a compute

capability of 2.0.

https://lh3.googleusercontent.com/notebooklm/AG60hOpFXcTzcLRlyMKi9KgJ_8hK0xwKFwtvN06KyJIuad-STKEO_uBZCeny9j0zdL-xxk6Bfjq8RFcjZh1cZw9MUsHomY-63A65oqy3R4mLBcQXjJiwxRnEgdS2p4KYwXb48y6XTtRXwQ=w870-h520-v0

77c3b3c7-c538-429a-baa3-f43a6427681d

4 Framework

Figure 4.1: Overview of patch-based terrain synthesis framework developed by Tasse et al. (2011). The terrain sketching interface is the entry point to the system, where the user sketches their desired terrain. This is used initially to produce a

synthesised terrain, which together with a source file is run through feature extraction. Patch matching and merging is run with the result being deformed according to the user’s initial sketch to produce the final terrain. This feeds back

allowing the user to modify the terrain and re-run synthesis.

This thesis extends the work done by Tasse et al. (2011), which is a patch-based terrain synthesis

system based on Zhou et al. (2007)’s work. An overview of Tasse et al. (2011)’s system is presented

in Figure 4.1. The authors improved on several aspects of Zhou et al. (2007)’s algorithm, discussed in

section 2.2.3, to enhance the quality of generated terrain. The components shaded in grey represent

the core terrain synthesis steps and are explained in this chapter.

4.1 User Input & Feature Extraction Zhou et al. (2007) provides users with a limited 2D sketching interface. This only allows users to

specify the 2D position and type of feature, ridge or valley, but allows no control over the height of

these features. Tasse et al. (2011) develop a new hybrid scheme that makes use of the sketching

interface from Gain et al. (2009). Users are presented with an interactive environment that allows

them to sketch 2

D constraint curves. These curves describe the paths and types of features as well

as allowing the user to sketch out the height profile. This information is used during different stages

of the synthesis engine; the height profile is used to deform the terrain after it has undergone

merging and matching. Firstly the user’s sketches are converted into a 2D map containing the ridge

and valley curves to be used during feature extraction. An example of a user sketch is provided in

Figure 4.3 (a).

Feature extraction works by automatically identifying features, such as ridges and valleys, in the

target terrain. In image processing, features are typically identified through the use of edge-

https://lh3.googleusercontent.com/notebooklm/AG60hOqvs0wCGgCKwqhRJ2WLl-1nhCpYMk9U4pa243XiPPWa_JVkP-iPFWxOYuzLfMS4fOkpR3eG5Uyt8iFZRVwFz5rKhwtX9VcAtpalul_ZCaQtuGM-UoyWppNeoa1uJfGleiWjL3EPdg=w814-h859-v0

4e527343-f2ab-40af-b46c-f7d646d79729

detection methods. These features are characterised by the locally maximal derivatives of the image

intensity, whereas terrains are based on local extrema of the height-map. Naively applying the same

techniques to terrain feature extraction results in spurious features due to local height variations

(Zhou et al., 2007). Thus a method making use of local extrema in the height-map is required to

extract the terrains features. The Profile Recognition and Polygon Breaking Algorithm (PPA) (Chang

et al., 1998, Chang and Sinha, 2007) is designed for this purpose. The original PPA inefficiently breaks

cycles by removing the largest value edge and runs in polynomial time with respect to the number of

edges. Bangay et al. (2010) explicitly reformulate the PPA as an equivalent process with the

elevation data represented as a graph and computes the minimum spanning tree (MST). The MST of

a graph is the subset of edges which allow the graph to remain connected and minimise the total

weight along all the edges, commonly created with greedy algorithms such Kruskal (1956) and Prim

(1957). The PPA comprises of the following 5 basic steps: Profile recognition, Target connection,

Polygon breaking, Branch reduction and Line smoothing.

Figure 4.2: Different steps of ridge extraction with the Profile recognition and Polygon breaking Algorithm (Tasse et al., 2011). The final result is the minimum amount of points required to describe the main feature path.

### 1. Profile recognition is an algorithm that marks all points that could be part of a ridge line as

potential candidates. To determine if a point is a candidate, the algorithm takes it as the

centre of the profile. If there is at least one point with a lower height on both sides of the

profile then it is marked as a candidate. Furthermore, the profile is switched from N–S, NE–

SW, E–W to NW–SE to determine the candidacy of the point. Figure 4.2 (a) shows a series of

points marked as candidates.

### 2. Target connection: All the adjacent candidates are now connected to form weighted

segments, shown in Figure 4.2 (b). This process can produce many diagonal connections that

cross each other, when this occurs the program discards the less important edge. Edge

importance is calculated by summing the height values of the two connecting points with a

lower total weight being considered less important.

### 3. Polygon breaking: Target connection has the potential of producing closed polygons which

need to be eliminated, as they can cause cycles in the resulting graph. The program

repeatedly checks for closed polygons and removes the least important segment (lowest

weight value) until there are no closed polygons of any size left (Figure 4.2 (c)). This process is

achieved by taking all of the segments and sorting them by their weight. The PPA then checks

the lowest segment to see if it is part of a polygon. If it is, the segment is deleted and the next

lowest is checked. This process repeats until all of the segments have been processed. This

produces a tree structure with edges lying along the terrain features.

### 4. Branch reduction: After Polygon breaking there are many short branches most of which are

undesirable side effects of the Profile recognition stage generating too many redundant

points. These short branches are repeatedly deleted a user-defined number of times. Figure

4.2 (d) shows the result of the Branch reduction step.

### 5. Line smoothing: This step moves the points to the average weighted position based on its

location in relation to its neighbouring points. The weight of each point is valued proportional

to its elevation for ridges and inversely proportional to elevation for valleys. The new position

better fits the trend line and since it is an average it will never shift more than one grid space.

The final output is a tree representation of the final ridge or valley feature lines (Figure

Tasse et al. (2011) were primarily concerned with performance and the extraction of large-scale

terrain features. The original PPA was used with modifications to the Polygon breaking stage to

make use of a minimum spanning forest algorithm while preserving the Profile recognition and

Target connection steps. Kruskal’s algorithm was chosen as it performs several orders of magnitude

better than Polygon breaking. It is possible that the feature extraction data can be stored during a

pre-process step and read in at runtime, saving valuable time, this is discussed further in section 5.4.

Once the features have been extracted for both the DEM and user’s sketch, the algorithm proceeds

to the patch matching stage.

https://lh3.googleusercontent.com/notebooklm/AG60hOq_GobDwD4E-GlzQNHxOUWk-38nJhX-cgpdYpD41YkBVBtYxuqSbJei9AtsXvixhZH8fHGqupO4NsdU89SvqmaLVUixVf0-wRAbtN8BPQIGDg-tqBIn2wvEoeexnXsdPmhnSshfOg=w1088-h1112-v0

f51d74f2-9395-43d7-8df9-5b85d96464dc

Figure 4.3: Patch-based texture synthesis. a) Users sketch input. b) Valley lines extracted from feature extraction on exemplar. c) Output after feature matching has completed. d) Final output after non-feature matching has completed.

4.2 Patch Matching Tasse et al. (2011) make use of a single exemplar to provide the source data for the system.

Feature extraction is run on both the exemplar and users sketch, with a collection of feature nodes.

Patches are centred on these nodes locations, which encompass a detected feature. For the

exemplar the patches are referred to as source candidates and are then rotated 8 times by 45° as

well as mirrored along the  -axis and  -axis giving a total of 10 candidates per patch. These

candidates are then compared to the patches from the user sketch. Patch matching is done in two

stages: feature and non-feature matching.

4.2.1 Feature Matching

Feature matching compares patches from the set of source candidates against the set of user

patches, searching for the best match. The output of the feature extraction process for both the

source file and user input produces a tree data-structure with edges falling on the terrain features.

These are connected with edge chains to form feature paths, as seen in Figure 4.2 (d). The matching

process starts by constructing the candidate pool from these edges, by using each edge position as

the central location of a candidate patch. The number of candidates is then expanded by rotating

https://lh3.googleusercontent.com/notebooklm/AG60hOplAJH9EE8AzVgfhB9CDzE0HLPomL0aJEMo8BcsES2lzSkNQlNvW1-xTmEGeN9LbpsheEV_zgd6q5-5evWOaRnztGDh-zRrmY-zGVyBqyMJYjuWg472Uj0tfWw03h3FYGpO1X4Drg=w832-h286-v0

aab581a9-8ac4-40b2-b002-73eb10895a5c

and mirroring the patches, making the system more versatile. Feature matching follows a breadth-

first traversal of the feature tree, preferably starting at the node with the highest degree of

connectivity. The traversal proceeds along the paths in increments of one-half of the defined patch

size, which ensures successive patches only partially overlap. At each node the control points are

calculated, control points are the locations where the feature path intersects with a circle centred on

the node with a radius,

. The number of control points describes the type of feature

that this node is classified as. A single point indicates an end point; two points are a feature path and

more than two indicating a branch point (Figure 4.4). They are used in a set of cost functions with

the candidates to determine the best fitting patch. Three cost functions are evaluated; Feature

dissimilarity (  ), Angle differences      and Noise variance     . These functions are multiplied by

scalar values       to control the level of influence each provides. The total cost      for candidate

matching against target patch   is as follows:

Figure 4.4: Example of different feature types based on the number of control points. a) Feature end point. b) Feature path. c) Feature branch.

## Feature dissimilarity

This function determines the similarity between the user and candidate patch by comparing their

height profiles using an    norm. The height profiles consist of the height values along the outgoing

feature path, both along the feature path and perpendicular to it. These paths are shown in Figure

4.5 (a and b) and represented by lines O and P, respectively. Candidates with a lower feature

dissimilarity cost are more suitable matches.

## Angle differences

The angles of the paths between the nodes for the given user and candidate patches are

compared using a normalized sum of squared differences. This angle difference indicates how similar

the structure of the candidate is to the user patch in terms of the top-down direction of the feature.

## Noise variance

The noise variances of the user and candidate patches are computed at multiple levels of

resolution and their sum-of-squared differences added to make up this cost component. The noise

variance for given patch (  ) for levels    , with     being the coarsest level, is the variance of

Gaussian noise computed by consecutively downsampling and upsampling    to obtain the lower

resolution      and subtracting      from   . This process produces a set of frequency bands where

https://lh3.googleusercontent.com/notebooklm/AG60hOoqLhHjJazl3qHB1xBKb58q9JaCptLiEtVRStWUg6FXlj2ZJep1dN1vQKTBIrzPW7CgURyN4f9t7MgLULJ5-EUIRg0V1pQvfSCmlhUcJ5nrMiSMAk0OkD0tC8dYvfjBJ7I0-cEfMA=w613-h491-v0

588b8525-c44e-46eb-96fb-237d9a96038a

the details have been smoothed out. This cost function compares the noise differences between the

two patches at different frequency band levels. Lower variance between the patches indicates that

they have similar characteristics in terms of bumpiness at both coarse and fine scales.

Figure 4.5: Feature dissimilarity Tasse et al. (2011), an illustration of how the algorithm examines the pixel data in a patch. (a) User patch. (b) Candidate patch. (c) Height profile for values perpendicular to path. (d) Height profile for

values along path.

After all of the candidates are evaluated the total costs are sorted in increasing order, from this

the first five (lowest cost) candidates are selected. These final candidates are now run through the

Graph-cut cost algorithm used during merging (section 4.3.1). This algorithm evaluates the suitability

of the optimal seam. A high cost indicates a greater impact on merging due to more dissimilar pixel

data. The candidate with the lowest cost is selected as the matched patch and sent for merging into

the output terrain as described in section 4.3. This process repeats for all the user patches resulting

in the output shown in Figure 4.3(c). The system then starts matching the non-features to fill in the

missing regions.

4.2.2 Non-Feature Matching

After the feature matching is completed the output terrain may contain areas where no data has

been populated. To fill these holes with data, non-feature matching is performed. For this process,

new candidate patches are created from areas in the exemplar that contain no significant feature

data. These candidates are matched against already synthesised data in the output terrain and fill in

the empty region . Criminisi et al. (2004) show that the quality of the output terrain is affected by

the order of the filling process and propose a filling algorithm that prioritises patches along

structures. Tasse et al. (2011) made use of a similar algorithm for their implementation of the non-

feature synthesis, which ensures terrain features are preserved and propagated correctly.

https://lh3.googleusercontent.com/notebooklm/AG60hOrysgmhKhr8S2nbbW0OxLaOWWCLoNXrQQX7rKyIBctYKew-JFgbb7gFkf1ohSsifGwhyhX01s2kqVmf10snTaZ22Cy11o46ho2IbGs0yb049vt0VsnSUGnpvzKYca_4bP5Wz77-Xg=w992-h520-v0

8c3c8522-0e00-4864-81a7-cc2500bc3d3f

Patch-based filling algorithm

This algorithm determines the location of the next patch to undergo matching and is based on a

best-first filling approach. The selection depends on priority values that are associated with every

point lying on the boundary of  ,   . For a given patch   , where      (Figure 4.6), its priority

value is influenced by a confidence and data term. The confidence can be described as a measure of

the amount of reliable information surrounding the pixel  . It is calculated by summing the number

of pixels in the patch area that contain already placed data, then dividing by the total number of

pixels (PatchSize2). This gives a representation for the amount of already placed data the patch

contains. The data term is a function of the strength of the isophotes (linear structures) hitting the

front of the boundary of , where there is no valid data. This term increases the priority for patches

that an isophote flows into. This is fundamentally important to the algorithm because it encourages

linear structures to be synthesised first and propagate securely into the output terrain. These two

terms are combined together to yield the priority value for   .

Figure 4.6: Example showing the empty region , with the boundary    highlighted in blue. A patch    centred around a point on    is enlarged.

Criminisi et al. (2004) proposed a filling priority that is calculated by multiplying the confidence

and data terms. This, however, discards pixels with a data term equal to zero even if their confidence

term is large. Instead, these terms can be added together as in Nie et al. (2006). At each iteration of

the algorithm the pixel that has the highest priority value is selected. Once  a new location has been

determined, matching is used to select the best candidate to fill the area. Non-feature matching

ends when there are no more empty regions to fill.

## Matching process

The candidate patches do not contain any significant feature data and thus, the matching criteria

are slightly different from that of feature matching. The set of candidates are evaluated against two

cost functions: noise variance difference    and the normalized sum of squared differences over the

overlapping area   .    minimises the difference in the bumpiness of the surface of the patch and

matches the pixel data already synthesised in the output terrain. The total cost    is computed by

adding the two cost values together after they are multiplied by a scaling factor  . This determines

the amount of influence each function has on the output as follows:

Tasse et al. (2011) uses scaling values of           and      , which were chosen after

observing the magnitudes of the individual components to ensure both contribute to the total

weighting. After the candidates are evaluated, the costs are arranged in ascending order with the

best candidates selected for a second round of cost evaluations. These short-listed candidates are

now evaluated against the Graph-cut equation to determine which patch would make the best fit.

The best fitting patch is then selected and merged into the output. This process repeats until there

are no holes left in the resulting terrain, marking the end of the terrain synthesis process (Figure

4.3 Patch Merging Once the system has found a matching patch, it must be placed in the output terrain. If this new

patch were simply pasted into the output and overlapped existing data, a seam would appear as if

there are different pixel values in the new patch. Thus a system is required to seamlessly merge the

new patch (  ) with any existing data. A patch (  ) is cut out from the target image at the location

where the new patch is to be placed. The region where existing data in    overlaps with data in    is

defined as  . Tasse et al. (2011) develop an improved merging algorithm using three different

techniques: Graph-cut (Kwatra et al., 2003), Shepard Interpolation (Shepard, 1968) and a Poisson

equation solver (Pérez et al., 2003). This new combination produces superior results to those of

Zhou et al. (2007).

4.3.1 Graph-cut

Patch merging starts by performing a Graph-cut to determine the optimal seam between patches

and    over the overlap region   (Figure 4.7). Or more specifically, the minimum cost cut of the

graph is required. This is a well-known graph problem; minimum cut (max flow) (Sedgewick, 2001)

with easy to implement algorithms. Kwatra et al. (2003) use this algorithm with a weighted cost

function , which penalises seams traversing through low frequency variations. This function is used

to determine the weight of the edge between pixels   and   in the overlap region and is defined as

|           |  |           |

|  |

|  |

|  |

with   representing the direction of the gradient, which is also the same as the direction of the

edge   .

are the gradients of the patches    and    along the direction  . The graph-cut

process uses the optimal seam to select which data from the new patch    is to be placed into the

output image. Figure 4.8 shows the main stages in the graph-cut algorithm. The optimal seam is now

known but still visible in the target and further processing is required to hide it.

https://lh3.googleusercontent.com/notebooklm/AG60hOoYHJqqk8o_ZhpaFja1xkHL0M4qyRmIJSvg4soagnNXOOKHw_0VaG38KiHSqLqx5VqaujdDth_tcnadpb3beL5oP_9DjBqLfSHe28jg7H3Us8XKir_3BdGtRCmiXLF3YcCpjU0BSQ=w1200-h760-v0

2fa99dc5-f7a1-4275-a5b9-a2642a9ee845

https://lh3.googleusercontent.com/notebooklm/AG60hOrRrBsP7HTp3f2QOSp3r_XFZm1AJKMvUwNdSfWY9WmIreuuI_j5S1ChAsFl5C1yZC8XuU2_F8Q7TB9zP506TN61YprBgXSn8vOZnEKJVFhSZBROso5xcrVESQtW4wK8Wt4xCZWJTA=w1000-h240-v0

a5316dcb-b535-48cd-bcb1-0e65dc9fe9d3

Figure 4.7: Illustration of the graph-cut algorithm between patches    and  . The optimal seam connects adjacent pixels between the two patches.

Figure 4.8: Example of the graph-cut algorithm steps. a) & b) Patches    and  . c) The overlap region   highlighted. d) The optimal seam between the two patches highlighted after merging.

4.3.2 Shepard Interpolation

A useful by-product of the Graph-cut process is that it partitions the target patch into two

portions: the sink   containing pixels that come from    and the source   containing   . The visible

seam can be removed by deforming   to match that of   along the cut. Tasse et al. (2011) chose to

implement a deformation technique based on point features proposed by Milliron et al. (2002) to

perform the deformation of the source producing   . The pixels contained in   are displaced by

, which is calculated from             for the points    along the seam and scaled by a

distance-based normalised weight ̂    . The deformation is defined as:

(∑ ̂

) (           )

with normalised weight:

https://lh3.googleusercontent.com/notebooklm/AG60hOrEz4jZmnq7339nxw25y-QBYfb1GxOO5BqSpZ3RoRU0G6YJZtvnxvfbWR7ba2pkn4-iESPn9V7fQs2Q0QIF0mwDD9KA6ECHEJSy4hkCYJu5cTFWB84hC2O5jOMukavQy2odW_A1AQ=w800-h300-v0

e30a6b8b-f715-4f02-b363-a55198a01158

̂

∑

where      is 1 at    and falls off radially to a distance   . The weighting function   was chosen to

be an Inverse Weighting function as defined by Shepard (1968):

{(

)

where         is the distance between   and   ,    the area of influence, and   the smoothness

factor. This deformation process is better known as Shepard Interpolation with the results of this

process following the graph-cut shown in Figure 4.9.

Figure 4.9: Results of Shepard Interpolation. a) Output from graph-cut algorithm. b) B is deformed to match the pixel values of A along the optimal seam.

A limitation of Shepard Interpolation is that it does not take into account the gradient values,

which results in discontinuities found in the resulting gradient field. These discontinuities manifest in

visual artefacts that are not visible in a 2D top-down representation of the terrain, but create

obvious surface irregularities in 3D renderings of the terrain (Figure 4.11 (a-d)). This problem is

solved by removing the optimal seam in the gradient field    instead of the height values in  .

consists of the gradient fields    from the sink   and    from the source  . The above equations are

used by substituting       with        and       with       .

(∑ ̂

) (             )

The gradient field    is now free of discontinuities, since both    and    have the same values

along the seam. The final step of patch merging is to calculate the new elevation values from the

modified gradient field by solving a Poisson equation.

https://lh3.googleusercontent.com/notebooklm/AG60hOqmk-lRHlDAqQX9aqCPv-EGXTIDOdmh6QZw52EMWLwur1MgnOTv70rIf5FchmMvh1YiWU4Bkk_HxEo5m1Pm8W5pj-RpaStK9pqgOVW3xHIKYM9dXs2XGO2QLmXM-Fg3kfYXpCe0GQ=w1200-h600-v0

4b4d01af-33db-49c5-8256-206d51a4b1f8

4.3.3 Poisson equation solver

The following Poisson equation with Dirichlet boundary conditions must be solved to calculate the

final elevation values of patch   :

|    |

where      is the Laplacian of   ,      is the divergence of the gradient field    and   represents

the entire patch area. Finite-difference methods (FDM) are used to construct a system of linear

equations that approximate equation 4.1 (George, 1970). The Conjugate Gradient method of

Shewchuk (1994) is used to solve the linear system for the unknown values of the new patch   . The

process starts by translating the height values within the overlapping area, which is approximately

one-third of the patch size, into gradient values. Next the gradient values along the seam are set to

zero. Now the Poisson equation is solved to determine the best set of height values to fit the

modified gradients. The generation process is not confined to the boundaries of the patch, as

neighbouring pixels may have been deformed during Shepard Interpolation, which also need to be

considered. The newly generated values for    are placed into the final terrain, which results in a

smoothly transitioning terrain that is free of visual artefacts (Figure 4.11(e) ).

Figure 4.10: Poisson equation solving process. a) The image as output from Shepard Interpolation, patch   . b) The gradient fields of the patch   . c) The modified gradient fields free of discontinuities along the seam. d) The final output

after the Poisson equations are solved.

4.4 Research Outcome Tasse et al. (2011) improved on the already impressive results of Zhou et al. (2007) by: enhancing

the matching process, increasing the candidate patches from the exemplar by a factor of 10; and the

addition of noise variance along with some tweaks to the existing cost functions. A visual

comparison of the results was conducted by Tasse et al. (2011) and found that in most situations

their results produced better matches to the users input compared to previous work. They noted

that the current approach to patch merging based on a combination of graph-cut and Poisson seam

removal (Zhou et al., 2007), is not well suited to terrains. This is because the techniques produce

terrains with discontinuities in the second order derivatives. These discontinuities appear as

artefacts on the terrain and are more noticeable when viewing the output terrain in 3D. A user study

was conducted which confirmed that their new patch merging technique, Shepard Interpolation with

a Poisson equation solver, is superior and succeeds in eliminating the boundary artefacts (Figure

4.11). User experiments were conducted by Tasse et al. (2011) to determine the realism of their

generated terrains. They found that there was no statistical significance proving that real unmodified

terrains were superior to those generated by their system. They conclude that the realism of their

terrain is not dissimilar to that of real-life landscapes. We identified several key aspects that Tasse et

al. (2011) have improved over previous work:

 A terrain sketching interface allows for greater user control (Gain et al., 2009). Users will

benefit from an interactive development environment with intuitive controls based on

drawing 2

D constraint curves.

 Performance issues of the PPA feature extractions are addressed using minimum spanning

trees, similar to Bangay et al. (2010)’s work.

 By including noise variance similarity and sum-of-squared-difference cost functions on the

overlapping region of already placed patches, the feature matching process has improved.

 By replacing the thin-plate spline deformation from the matching stage and mirroring and

rotating candidate patches increases the sample pool. This increases the likelihood of finding

good matches in the sample terrain.

 The feature dissimilarity cost function is modified to take height differences along outgoing

branches of a feature into consideration.

 The filling order for non-feature matching is changed to a best-first filling approach based on

gradient values (Criminisi et al., 2004). Noise variance similarity is also used during this phase.

 Lastly, a novel patch merging algorithm more appropriate to terrains is introduced. Along the

optimal seam between two patches, discontinuities in the gradient field are removed with a

scattered point interpolation and a Poisson equation solved for the new height values instead

of just setting them to zero. This method produces a more realistic landscape, especially when

viewed in 3D.

Tasse et al. (2011) propose possible extensions to their synthesis framework: utilising multiple

DEM exemplar files and enhancing performance with GPU acceleration. In the next chapter we look

into these extensions as well as some others in our enhanced framework.

https://lh3.googleusercontent.com/notebooklm/AG60hOp_9i3Lg6cNZwxCNoVCa7a9CC5GJXGzEtj1h2ENKyrbP8aen2VTX0YgD7OhEO_Ypd1RsUAYesGgIs_45OXrFqCJ3WPZZpJwnh0h7jXv0b_H6eCOK3JltswxB-GSbcm2dzj2f2cZvA=w1261-h1233-v0

dd8a3c2a-1495-4e06-9764-8a03913c9298

Figure 4.11: Comparison of patch merging techniques (Tasse et al., 2011). (a) No patch merging. (b) Graphcut algorithm. (c) Shepard Interpolation. (d) Results from Zhou et al. (2007). (e) Results from Tasse et al. (2011).

https://lh3.googleusercontent.com/notebooklm/AG60hOrGhq3Fe3wWB0KweFUBw7rE1Z8maH3i7TZ3bLKqTXLQ2uZStW0tfiAr6c0MYEnreQ2KGLJKVXnilX5B5Jww1Seoo4DkZBOve1tL25nXvdbLWfCtoWqxiWwIDENW4yphOdluXIVlkQ=w722-h397-v0

f702c980-92f2-471a-b5bd-d5e603575f43

5 Enhanced Framework

Figure 5.1: Overview of our proposed system for enhanced terrain synthesis. The entry-point to our system is the simplified sketching interface, which when synthesis is initiated, run through feature extraction to build the user

candidates. A collection of varying source files is run through feature extraction also, with the feature data being used in matching and merging with the sketch data. A final step fills in the gaps left from feature synthesis with data from the

source candidates to complete the terrain.

The terrain synthesis system of Tasse et al. (2011) produced compelling results due to their novel

patch merging technique and enhanced user interface that allows 2½D manipulations of the terrain.

However, some improvements can still be made to the system. An important improvement is

support for increasing the candidate patch pool through the use of multiple input source files. This

will allow for greater variability in synthesised terrains, provided it can be done efficiently. To

accommodate this change, synthesis speed needs to be hugely increased. This requires

parallelisation of core algorithms, extensive optimisation and careful use of pre-computation. We

chose to use GPU-based parallelisation as the core of our extensions since a GPU is well suited to

many image-based operations. Figure 5.1 is a design overview of our proposed system with the

areas representing the main differences from the system of Tasse et al. (2011) outlined in red. This

chapter describes the design of our framework so as to provide a high-level understanding of our

objectives.

5.1 Multiple Input Sources Previous work on texture synthesis mainly focuses on the use of a single source file, from which

data is extracted. This limits the variation available, as it is highly unlikely that a single source

contains enough variation to meet the user’s requirements, particularly for highly varied or large

terrains. For terrain generation this limitation is easily observed by using a single source file that

contains mostly flat landscape data and attempting to synthesis large mountainous regions (Figure

5.2a). This can also lead to noticeable repetition during patch synthesis (Figure 5.2b). While we

found no research relating specifically to terrain generation using multiple sources, Wei (2003)

proposes a multi-source pixel-based texture synthesis technique. Their system minimises an error

https://lh3.googleusercontent.com/notebooklm/AG60hOr2L9bK5GjHRZCMPJ0ScpxCGPav83OCC_MjilmGuXZJrQq1tWi1StPaUuf3YUYY45M-7mpVfTf9U1Q2hRDET8RQbbbi30hTVtx-Md6O74JBcdZBNYcMYiSLRZwyOWuXIvfOcDuZkA=w1060-h550-v0

0dfc1904-9dac-42c5-bcfb-fcaa1eb30b9e

function by examining all the pixel inputs within individual neighbourhoods (patches) to find the best

set of input pixels. This error function is a weighted sum of the    norm between the

neighbourhoods of two different inputs. Unfortunately this technique does not adapt well to the

system designed by Tasse et al. (2011). We found no other suitable research relating to the use of

multiple input sources for terrain generation; instead we develop our own as an extension to Tasse

et al. (2011).

We propose a system that supports a large number of input terrains in order to maximise the

variation of data. This will allow for more diverse and rich landscapes to be generated. By using

many individual files, the work can be more easily distributed to multiple processing elements

without the need for complex logic to divide and distribute a single file amongst them. In order to

provide this functionality, much of the underlying algorithm needs to be modified. During the

feature and non-feature synthesis stages, the candidate patches from each input source are

evaluated against the target patch. The best matching patches from each of the input sources are

retained and compared with the overall best patch being selected for merging into the output

terrain. The specifics of this process are discussed in detail in chapters 6 and 7. One disadvantage is

that the all input sources will no long fit into memory: data must be streamed to and from the

secondary memory. However, this is an acceptable cost given the improvements arising from a

multiple source system. Nonetheless, in order to minimise the performance impact, additional

optimisations will be introduced to speed up the overall processing pipeline.

Figure 5.2: Examples of limitations with using a single source for terrain synthesis. (a) Using an input terrain without the correct type of feature data, source image lacks ridge details. (b) System can produce noticeable repetition in output

5.2 CPU and GPU Accelerated Synthesis The process of synthesising terrains is complex and requires a large amount of processing power.

The candidate patch searching algorithm is  (      ), where   is the number of source files for

candidates against the   user patches to match (Listing 5.1). Then there is the searching within

each of the patches to evaluate the cost. Some aspects of this process can be parallelised since the

algorithm is not dependant on each iteration’s result. This allows for multithreading to be

implemented in order to exploit the parallelism of modern CPUs. For example, the looping over the

user patches for each candidate can be divided up amongst multiple processing cores. Assuming 4

processing cores, the complexity for the algorithm is reduced to approximately  (

1 Loop over source files (a) { 2     Loop over candidates (m) { 3         Loop over user patches (n) { 4             Calculate cost of m for n 5         } 6     } 7 }

Listing 5.1: Algorithm overview for the candidate searching algorithm

However, some components of the system are sequential because they require information from

previous iterations to complete. An important sequential component is the patch merging process.

Merging selects the final candidate based on the information contained in the final output terrain.

Despite these limitations, substantial performance can still be gained from a multithreaded

approach. These gains can be magnified by leveraging the extremely high degree of parallelism that

GPU devices offer. The details of these parallel implementations are covered in sections 6.3.2 and

#### 6.4. By reducing the time required for terrain synthesis, the system becomes far more responsive

and also allows larger terrains to be generated in reasonable time.

5.3 Simplified User Sketching Interface To simplify the process of specifying the user’s input to the system, a simplified sketching

interface was designed. The interface provides the user with two pens that are used to draw the

desired locations of ridges (mountains) or valleys, which make up the features. Once the user is

satisfied with their sketch the synthesis option can be selected. The user is required to finish

sketching all of their desired strokes as the system, while efficient, is not capable of interactive

runtimes due to the large amount of data that requires processing. Next the drawn strokes are

compressed down to produce a single image, which uses a binary system of black and white marking

to represent the ridges and valleys. This image is run through feature extraction, which decomposes

the sketched curves into a series of linked nodes/vertices in a graph structure. This graph structure is

then split into a series of patches that are compared to the source candidates during synthesis. The

details of the feature extraction process are described in section 6.1.

After all the features have been matched there may be many ‘holes’ in the output terrain where

no features were drawn by the user. These areas are now filled with featureless data extracted from

the input terrains in the form of patches. Once complete the final terrain is displayed in the

interface. During the synthesis operation, several stages are rendered and kept as snapshots which

can be viewed from the interface. These include the feature detection phase, which overlays thin

lines over the sketch to show where the system detected features for both input terrains and user

sketch. The result of each synthesis operation is also saved. The current output, representing the

state of the system, is displayed in the interface and updated after every merge operation is

completed. All these images can be easily switched between. The interface is updated continuously

during the synthesis process so that the user can see each of the patches being placed into the

output. The interface also allows the user to save the output terrain as well as the other snapshots

as either a PNG image or Terragen terrain file. Loading of user sketches is also supported, which is

https://lh3.googleusercontent.com/notebooklm/AG60hOqdpSxwjOATb94-SOYUs3mKn0am6NvOKqx6cqW4KQRIHHuRhFtF5jxiR0HeZmJB1GwTZWooTSu8GT1hHiVbRzdSVVJUNi8rgGDyJIYD3C_zEVMKLtw5KHvNsvAczjAaojpoPZ1J=w1050-h880-v0

3f656774-d70f-4c74-ad78-021ea0f0296e

useful for testing purposes as it allows one to synthesise the same terrain multiple times for

comparison purposes. Other useful tools include undo/redo commands and an eraser pen to make

sketching easier. There are plans to extend the user interface further in the future as described in

section 9.2. Figure 5.3 shows the different views of our interface.

Figure 5.3: a) The main sketching interface with all menus expanded. b) Sample sketch drawn with feature detection run. c) Output after feature synthesis. d) Final output

5.4 Pre-Processors and Pre-Loaders Running feature extraction on the input sources during synthesis wastes valuable time. The

inefficiency can be addressed by pre-computing and storing the feature data and then reading it in

when required. This observation brought about the notion of pre-processing data as an optimisation

step. The feature extraction process (Section 4.1) is computationally expensive. The output of this

phase is a graph of linked nodes (edge pairs) which represent   and   coordinates in the input file.

These edge pair coordinates are saved out to an external ASCII file as newline delimited entries. By

saving the data, each input file only has to be processed once. When loaded again these edge pairs

reconstruct the graph to form a series of nodes, with each node being the central point for source

patches. The patch size can be changed in the system to choose how much data around the node is

used by the system. We present results for varying the patch size in section 8.3.3.

Pre-loaders are used to pull all the required data into memory in order to speed up processing.

During the feature synthesis stage the raw image, along with the pre-processed feature extraction

data for each source, is loaded into memory. If there is insufficient space for all sources, source data

is loaded in batches, with each batch being processed before the next batch is read in. Storing the

data in memory allows for very quick access when calculating the best matching candidate for each

of the user patches. The candidates for the source file are generated on-the-fly for processing and

discarded when complete, thus limiting the total memory overhead by not keeping all candidates in

memory constantly. This process is also batched by the system, if a particular source file has a very

large number of candidates, to prevent it from running out of memory. This same principle is applied

to the GPU implementations, with the source images and feature extraction data persisted in GPU

memory. This is important as there is a very large overhead with transferring data from host to

device, provided there is sufficient memory, otherwise batching is performed.

5.5 Summary We presented an overview of our enhanced terrain synthesis framework that builds on the work

by Tasse et al. (2011). It includes the use of multiple input sources to dramatically increase the

candidate pool of data, which leads to better quality synthesised terrain. Through data pre-

processing and optimisation of the synthesis pipeline we are able to reduce synthesis time despite

the increase in input data. The next two chapters cover the implementation details for the synthesis

https://lh3.googleusercontent.com/notebooklm/AG60hOrRLL4C-pNjuQ8UgtvtkDROogYS4PO-s8cikcENLQ8_5nk5Db13a2pPJxFKzXX9ulj--WPCKPBVh7r6mux6OXWnp7Sr0-gIeFr-4kVdj4UpJK7hVXBqa0ocrDkSL-NDFmtTGmpYXw=w896-h387-v0

70d56546-e39a-4227-ad5a-5888276c0dfc

6 Feature Synthesis

Figure 6.1: Feature synthesis pipeline showing flow of data for the Feature Matching & Merging block of our system (Figure 5.1)

Feature synthesis is the process of matching data from source terrains to a user sketch and placing

them seamlessly into an output terrain. First the candidates are extracted along feature lines from

the source terrain. These are then evaluated against the user patch by applying several cost

functions to determine a subset of optimal matches. This process is extremely resource intensive

and we present a parallel CPU and several different GPU implementations to speed up the matching

process. Once all the source files have been processed, the overall best match is calculated with a

second round of cost equations and then merged into the final image. The rest of this chapter

explains the process in detail; Figure 6.1 shows an overview for the feature synthesis pipeline.

6.1 Feature Extraction & Pre-Loaders Once the user has completed sketching their desired terrain they can start the synthesis process.

To begin, the sketch they have drawn, is processed through feature extraction (Section 4.1), which

generates the feature paths as a set of connected nodes. These nodes mark the central location for

the user patches which are to be synthesised from the candidate data. The user’s sketch is pre-

processed to extract both ridge and valley data and then stored in memory. The control points,

which describe the type of feature, are also calculated at this point for later use. The feature

synthesis stage is then initiated. This stage requires the source files to be pre-loaded into memory

for faster access subsequent to a pre-processing stage (Section 5.4). Feature synthesis is then run for

all the user patches.

6.2 Cost Functions The feature matching algorithm must quantify the difference between the source candidate and

user patches. There are two rounds of cost calculations, with the first round occurring during the

inner functions of feature matching (Section 6.3) and the second occurring before merging into the

final image (Section 6.5). There are four cost functions in total: Feature Profiling, Sum-of-Squared

differences, Noise Variance and Graph-cut cost. Feature profiling is part of round one; the others are

https://lh3.googleusercontent.com/notebooklm/AG60hOoEqp5dXpSbPRbOuPkbJLTlspLZ-Q5sY96BtMSsmhxHCy61BFwmwOxcBAdYEuusuJwZRI_Cqte0J55uuZJpm5zL2ChG4MkdLME0eS5thijuu7UrSbs_n-LiUn-J5pu-XU7_cLi6SQ=w613-h491-v0

fa5878bf-f0b8-4286-90b0-298109bb32b1

executed during round two. The reason for splitting this up is that the cost functions in round two

make use of the already placed data in the output terrain, to provide better statistics.

6.2.1 Feature Profiling

Figure 6.2: Feature profiling algorithm against user and source candidate patches. Segments r and s represent profile paths for the patches.

Feature profiling quantifies the similarity of features between the candidate and user patches.

This cost is calculated by comparing the height profiles of lines   and   of the user patch   (Figure

6.2 (a) and (b)) with the    norm (Sum-of-Squared Differences). Tasse et al. (2011) additionally

compare the height profile along the feature path, in contrast to Zhou et al. (2007) who only

compare height profiles perpendicular to the path. Figure 6.2 (c) illustrates the height differences

between the target and source patches for both segments. The lower the calculated cost, the more

likely the candidate is a good match for the user patch. This cost equation is used during the first

round of evaluations as it only requires the feature path data and does not rely on already

synthesised data. It therefore carries all the weight for the initial matching phase.

1 Inputs: User Candidate, User Control Points, Source Candidate 2 Output: Cost value (float) 3  4 Initialise:         5 Loop over control points { 6     Calculate points for segments   and   (Figure 6.2) 7     Run profiling for segments   and   { 8         Initialise:               9         Iterate over all points on segment { 10             Calculate difference: (                      )

11             Add difference squared:            12                     13         }

14         Add sum to total:        √        ⁄

15     } 16 }

17 Final cost:                       ⁄

Listing 6.1: Feature Profiling algorithm

6.2.2 Sum-of-Squared Differences (SSD)

The SSD function evaluates the differences between the source candidate and the user patches

(Listing 6.2). Here, a lower overall cost indicates a better matching candidate patch. The user patch is

extracted from the current output terrain, but as this may contain invalid (not yet synthesised data)

this SSD is only evaluated across the overlapping area of the patches. The SSD evaluates the

difference in the raw height values of the corresponding pixels, squares the difference and adds this

to the sum for the candidate patch.

1 Inputs: User Candidate, Source Candidate 2 Output: Cost value (float) 3  4 Initialise:               5 for (  : PatchSize

6     Obtain user candidate pixel value (  ) 7     if    is valid { 8         Obtain source candidate pixel value (  ) 9         Calculate difference (          ) 10         Add difference:            11                 12     } 13 }

14 Final cost:     √        ⁄

Listing 6.2: Sum-of-Squared Differences algorithm

6.2.3 Noise Variance

The noise variance for the source candidate and user patches is computed at multiple levels of

resolution and the SSD of these differences is calculated. The lower the overall cost value the better

the chance of the two patches having similar characteristics in terms of roughness of the terrain at

different frequencies. We implemented the Wavelet Noise algorithm of Cook and DeRose (2005).

The noise variance for an image at a given resolution level is the variance of the Gaussian noise

produced by consecutively downsampling and upsampling the image, resulting in a lower frequency

image which is then subtracted from the current level. For the purposes of our research we make

use of three noise levels. A useful observation is that the orientation of the patches has no effect on

the noise variance, meaning that it only needs to be calculated once for all orientation changes of

the candidates. The cost calculation algorithm is presented in Listing 6.3. To limit the influence this

cost value has on the overall candidate cost, it is scaled by        . This balances out the cost so

as not to overpower the SSD cost value.

1 Input: Noise variance arrays (User and Source candidates) 2 Output: Cost value (float) 3  4 Initialise:       5 for (  : #Levels – number of levels of generated noise variance) { 6                 [ ]        [ ]

8 Final cost:    √

Listing 6.3: Noise Variance algorithm

6.2.4 Graph-cut cost

During patch merging (Section 6.5) a Graph-cut (Section 4.3.1) is performed between the output

terrain and the best overall candidate patch, which determines the optimal seam between the two.

As part of this process, the minimum cut (max flow) is calculated in order to find the optimal seam

along which to cut. We use this value to quantify the placement of the candidate patch in the output

terrain (Listing 6.4). There is a large overhead in running this calculation and it is thus only executed

on a small subset

of the five lowest cost candidates with the best patch being selected for the merging process.

1 Inputs: Destination patch from output terrain, Candidate patch 2 Output: Cost value (float) 3  4 Initialise graph-cut algorithm { 5     Create vector to store cuts ( ) 6     Loop over pixels in patch (PatchSize

7         If pixel is valid then add coord to   8     } 9     Loop over   { 10         Determine sink / source status of cuts coord 11     } 12 } 13 Initialise graph structure (G) 14 Loop over   { 15     Calculate weight and add as edge to G 16 } 17 Calculate maximum flow of   and return as cost value

Listing 6.4: Graph-cut cost algorithm

6.3 Feature Matching – CPU Our first implementation of the feature matching algorithm is a sequential CPU algorithm. We

develop two different versions of this algorithm with the difference lying in the order of looping over

the datasets. We then select the more efficient of the two in terms of both speed and memory

efficiency and develop a parallel implementation to further improve performance. During terrain

synthesis the feature matching process is executed twice, once for matching ridges and once for

valleys. In this section we first discuss the cost calculation process and then present the sequential

and parallel implementations.

https://lh3.googleusercontent.com/notebooklm/AG60hOqNySeurDqOX2qs81z46Uxu5zbtmfMpLoZ0StSAYT7Wsy2tApmqpYy-d3hBwSS7wNUzMIlz_jyb17R_0m_zl9PavYQvfNKjMfrMTbOCiNmf4VFj4wXH42_EqJo0h5k78JFKgnaABg=w450-h239-v0

b7c5a9e9-4fbc-4c62-bd4f-5617ce119a5d

6.3.1 Sequential CPU Implementation

Figure 6.3: Overview of the second version of sequential CPU feature matching. Feature merging is included as it is a required part of the flow. More information on the merging process is found in section 6.5.

The design of the sequential implementation is adapted from Tasse et al. (2011) and extended to

account for multiple input sources. An overview of this process is provided in Listing 6.5 representing

version one of the sequential implementations. It starts with the data obtained from feature

extraction of the user’s sketch, which is processed to produce a set of user nodes. These nodes

represent ridge/valley features that need to be matched against data in the source terrains. The

algorithm loops over these nodes; this is done so that data is placed into the final output

sequentially and then used in cost calculations for successive patches. Within each iteration, the

current user node is prepared: the pixel data from the users sketch is extracted centred on the node.

The control points calculated during feature extraction are also obtained, which describe the type of

feature this node is. This data represents the user patch that is to be matched against the source

files and is run as an inner loop.

1 Inputs: User Sketch, Source Files, Source Feature Extraction Data 2 Output: Image with features synthesised 3  4 Extract features from user sketch 5 Verify features were found (abort if none) 6 Prepare user nodes 7 Loop over user nodes { 8     Prepare user patch 9     Loop over source files { 10         Verify features 11         Prepare source nodes 12         Prepare source patches (candidates) 13         Calculate costs for all candidates 14         Sort candidates on cost 15         Save subset (best-set) 16     } 17     Find best overall patch from best-sets 18     Merge best patch 19 }

Listing 6.5: Algorithm overview for the version one of sequential feature matching.

All the source files are evaluated in this inner loop (source-loop), with the first step being to

ensure the source file has been pre-processed and pre-loaded into memory. The pre-processing step

runs the feature extraction algorithm in the background; the results are saved to hard disk to

prevent the need to run this costly process repeatedly. There is a small chance this has not

happened yet if there are a large number of source files in the system. The process will block until

the data is properly available, as the pre-processing is carried out in a separate thread (see section

6.6 on optimisations). The source file now undergoes a preparation stage, where the raw extracted

feature data is processed into nodes. This state verifies that this file has the correct type of features

(ridge / valley) before converting these nodes into patches. During the conversion process the

patches undergo a series of transformations. They are rotated eight times through 45° increments as

well as mirrored along the   and   axes, to produce ten variants for each patch (hereafter referred

to as the candidates). At this point the candidates are matched against the user patch according to

the feature profile cost function (Section 6.2.1), to produce the cost values for each candidate. This

data is now sorted in ascending order and a small subset of the best candidates (best-set) is stored

for additional processing. For the purposes of our research we use a subset size of five candidates

per source file, because we found that larger set sizes increased synthesis time due to the more

complex second round of cost calculations. Smaller subset sizes would reduce the variability at

merge time, which could potentially lead to the same patch being placed adjacent to one another.

This would create noticeable visual artefacts, which we can address by skipping the selection of the

same patch for adjacent merge operations. The matching process continues evaluating all source

files and adding the best-set of each to an array. Once this array is complete it is returned to the

user-loop where processing continues.

1 Inputs: Array of best candidates 2 Output: Single best patch 3  4 Initialise variables 5 Extract user patch ( ) from current state of final output terrain 6 Generate noise statistics for   7 Loop over best candidates array { 8     Extract candidate ( ) from its source file 9     Run sum-of-squared differences cost function for   against   10     Run noise variance cost function for   against   11     Run feature profiling cost function for   against   12     Record sum of cost functions for candidate 13 } 14 Sort the candidates cost value in ascending order 15 Loop over best five { 16     Run graph-cut cost function for   against   17 } 18 Select lowest cost patch as best overall

Listing 6.6: Algorithm overview for selecting the best overall patch

After all source files have been evaluated there is an array of best candidates with a maximum size

of                  , which is processed to determine the best overall patch for merging. There

can be fewer candidates if there is insufficient feature data in a source file to provide a reasonable

match. At this point a second round of cost equations is run on all these candidates to find the best

overall patch, as described by Listing 6.6. There are two parts to this. Firstly all the candidates are

evaluated with the SSD, noise variance and feature profiling cost functions. They are then sorted

again and the best five candidates are run with the Graph-cut cost equation to work out the best

patch to merge. This best patch is now merged into the final output terrain as described in Section

#### 6.5. The process repeats until all the user patches have been matched, at which point feature

synthesis is complete and non-feature synthesis is executed to fill in any gaps in the output terrain

with data from the source files that contain no strong features (featureless). This process is

described in chapter 7.

The implementation above is optimised for use with a single source file as per Tasse et al. (2011),

because the source candidates and user patches can be directly compared to each other while being

kept in memory for fast access. By extending it to support multiple input files, each of these sources

needs to be loaded into memory and candidates extracted for every user patch. This requires the

source files to be swapped in and out of memory multiple times leading to a large performance

overhead. Another approach is to retain the user patches in memory and allow for the source files to

be loaded once per synthesis and compared against all user patches. This solution trades space

efficiency for time efficiency. Results for the differences between version one and the optimised

version two are presented in section 8.1.1.

The alternative version of sequential feature matching (Listing 6.5) is described in Listing 6.7. The

process starts by processing the user’s sketch through feature extraction and generation of the

nodes as before. At this point all the nodes are processed to create an array of user patches. This

array is cached in memory and has provision for storing the best-set candidates for all the source

files. This requires a larger amount of memory as the best-set for all source files and all user patches’

needs to be stored. This is because the merging process can only take place after all the source files

have been evaluated.

1 Inputs: User Sketch, Source Files, Source Feature Extraction Data 2 Output: Image with features synthesised 3  4 Extract features from user sketch 5 Verify features were found (abort if none) 6 Prepare user nodes 7 Create array of user patches 8 Loop over source files { 9     Verify features 10     Prepare source nodes 11     Prepare source patches (candidates) 12     Loop over user patches { 13         Calculate costs for all candidates 14         Sort candidates on cost 15         Save subset (best-set) 16     } 17 } 18 Loop over user patches { 19     Find best overall patch from best-sets 20     Merge best patch 21 }

Listing 6.7: Algorithm overview for the version two of sequential feature matching.

Next the system loops over all the source files, verifies features, prepares the nodes and

generates the candidates. All the user patches are then evaluated against all the candidates with the

best-set of candidates for each user patch stored for later processing during the merging loop. After

all sources have been processed the system finally loops over the user patches to process the best-

set candidates for each. The overall best patch is determined through a second round of cost

https://lh3.googleusercontent.com/notebooklm/AG60hOpQGq3gzF-hWicKRTgoDvcK5-RdvZRM7bJl7CXxHF9fT4tP1aUxhs9UJe4bRYRJiEYzwq-on3lM6wQ2_byL3dr0ma4PXt5nZlmjgTssXCk2-SAd7aKUHL47aMF1EO4QLRV2jTAqyw=w551-h339-v0

436f87a0-d743-43da-b2f4-3e8797b5b91a

calculations and then merged into the final terrain, with feature synthesis concluding after all user

patches have been merged. Based on the results (Section 8.1.1), we observed a marginal speed

improvement over version one. This improvement comes from not having to swap the source files,

which includes a preparation step before cost calculation. It is mostly mitigated from our pre-

processing and caching optimisation. However, this version scales better when a larger number of

source files are used and provides an easier implementation for distributed computation, as there is

less context switching of data involved. It was decided that this would be the optimal solution and

subject to further optimised through multithreading as described in the next section.

6.3.2 Parallel CPU Implementation

Figure 6.4: Overview for parallel CPU feature matching.

The parallel CPU implementation builds on Sequential CPU version two by adding additional

threads to several of the components to better utilise the multiple cores of modern CPUs. Figure 6.4

shows the same version two pipeline but with the parallelised parts highlighted. There are two

components that we have targeted for multithreading: the pre-processing stage and calculating the

candidate costs for both rounds. In order to support the use of multiple threads, some changes were

required to manage the threads and correctly distribute the workload. The pre-loader system

distributes the workload by giving each thread a different source file to process. This works well as

there is consequently no shared data between the threads.

During the matching process a large number of candidates are compared to the current user

patch. This process can benefit greatly from multithreading with the candidates being equally

divided between the threads. Once all the candidates have been processed by all the threads, a

single thread then performs the sorting process with the best-set stored before moving onto the

next user patch. Once all the user patches have been evaluated then the next source file is

processed. The parallelisation in the system could be extended by unrolling one of the loops. Then

having each of the source files processed by a separate thread controller, which itself spawns

multiple threads for the cost calculations. However, this would not improve performance and could

actually hamper it – CPUs have a low number of cores and creating too many threads would force

https://lh3.googleusercontent.com/notebooklm/AG60hOpGWV5ygiQx4Djy7Hm7dgAvAoyc71HqnqiRdz09ayVV0UX_up_vcip6ex0rSSfLRSochqtFPSqWnecThs5WWWs9nSHxgs4Xrzj0HDVPIF53dG-GntMqFbK47nHUTOL8pdjWO-5h=w451-h336-v0

3b22da46-132c-4277-8a06-4fa05cb77c1e

the CPU to thread swap constantly resulting in lower overall performance. This idea, however, is

explored in section 6.4 with GPU matching.

Once all the source files have been processed, the system starts the merging process with the

second round of cost calculations. Again, the cost calculations for the candidates are divided up

between multiple threads to speed up processing. A single thread is required to perform the sorting

and the final merging process before moving onto the next user patch.

6.4 Feature Matching – GPU

Figure 6.5: Overview of the GPU feature matching pipeline

This section discusses our implementation of feature matching on GPU devices to further reduce

the synthesis time by exploiting the high degree of parallelism the GPU offers. Figure 6.5 shows an

overview of the GPU pipeline with the components processed on the CPU and GPU as shown. In

order to maximise the performance, a good balance between host (CPU) and device (GPU) is

required. We show how the design of our parallel CPU implementation can be adapted so that the

most suitable parallelisable components can be processed on the GPU.

6.4.1 Caching of data on GPU

Before any processing can be performed on the GPU, the data needs to be transferred over from

host memory to the devices global memory. There is a very high overhead in transferring such data,

so it is desirable to pre-load the data and cache it on the GPU. The first time a synthesis operation is

begun, the data is transferred, which means that running another synthesis will be faster as there is

no transfer cost. For the purpose of testing in Chapter 8, we run all tests after the data has been

cached. Once the data is cached in the devices global memory it is transferred to higher speed

device memory for cost calculation. We implemented several CUDA kernels to compare the use of

these different types of memory (Section 6.4.3).

6.4.2 User Patch Extraction

The first stage of running the synthesis on the GPU is to process the user’s sketch through feature

extraction. This is still performed on the CPU, but the user node data is transferred to the GPU. The

user patches are now extracted from the data according to the pseudo code in Listing 6.8.

1 Inputs: User Patches 2 Output: None (Results kept in GPU memory) 3  4 Allocate memory on GPU { 5     Patch cords:                              6     Patch control points:                                                7     Number of Patch control points:                              8 } 9 Copy user data from host to device

10 Allocate memory to store user patches: (                                     )

11 Init kernel:         (

)

12 Execute extraction kernel { 13     Calculate lookup coordinates for sketch 14     Calculate memory address for output 15     Extract pixel data from source 16     Store data in user patches memory 17 }

Listing 6.8: Overview for the user patch extraction on the GPU

Firstly, blocks of memory are allocated on the device, and the user’s sketch and accompanying

feature extraction data are loaded into this space. An additional large array is allocated, which stores

patches extracted from the user sketch. The user patch extraction kernel is then executed, using a

grid with dimensions of (

) and block dimensions        . The

blocks are sized to fit within a single CUDA warp to maximise efficiency. We divide up the grid so

that multiple blocks can process a single patch and add a 3rd dimension to iterate over all the user

patches. One pre-requisite is that we select a patch size that is a multiple of the CUDA warp size – for

our research we used a standard patch size of 64. Zhou et al. (2007) made use of a patch size of 80

pixels, they determined this by the spatial scale of the source file being used as well as the detail of

the resulting image required by the user.  Tasse et al. (2011) also make use of the 80 pixel patch size,

we chose to use a smaller patch to better suit our GPU system design and although marginally

smaller it should not impact the output much. We do a comparison of our system with that of Tasse

et al. (2011) in section 8.3.1, where we provide the output images from their system with ours using

a patch size of 80 and then of 64. After the user patch extraction kernel is executed, we are ready to

process the candidates.

6.4.3 Candidate Cost Calculations

The system now loops over the source files to analyse the costs of all the candidates against the

user patches. The process starts by allocating memory on the GPU to store the source image along

with the data obtained from feature extraction. This data is only cached to the GPU once, at first

runtime, for every source file. The largest computation time comes from calculating the costs of all

the candidates for all the user patches. We develop a total of eight different GPU versions to

investigate various techniques to maximise performance.

## Initial GPU versions

The first four versions share a similar starting point, memory is allocated to store a single

candidate patch                                 . There are two nested loops that are run,

the first loops over all the source patches and the second over the number of candidate

transformations, this is set at ten variations per patch. Inside the inner loop the candidate image

data is extracted. A kernel with grid dimensions (

) and block dimensions

is used to efficiently extract the candidate patch from the source image. When running the

cost calculation, there is an additional loop which iterates over all the control points for the user

patch. This number describes the kind of feature that was detected during feature extraction and is

thus variable for each patch. The code then diverges into one of the four different versions, although

with each setting the grid dimension equal to the number of user patches to process.

1 Inputs: Source File, Patch Offset 2 Output: Candidate patch stored in GPU memory 3

4 Init Kernel:         (

)

5 Execute extraction kernel { 6     Calculate lookup coordinates for sketch 7     Calculate memory address for output 8     Extract pixel data from source 9     Store data in user patches memory 10 }

Listing 6.9: Overview for the candidate patch extraction kernel

The first version (Listing 6.10) is a simple attempt that uses a block dimension     that makes use

of a single thread and executes all the code sequentially. This is highly inefficient and the

performance results reflect this (Section 8.1.3), although still much faster than a sequential CPU

implementation. The goal of this implementation was to directly translate our CPU implementation

and provide a base to work from for GPU optimisations.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                   6 } 7 Loop over source patches { 8     Loop over number of source transformations { 9         Execute candidate patch extraction kernel (Listing 6.9) 10         Init kernel:                                       11         Execute cost calculation kernel { 12             Loop over control points { 13                 Calculate points for segments   and   (Figure 6.2) 14                 Run profiling for segments   and   { 15                     Initialise:               16                     Iterate over all points on segment { 17                         Calculate difference: (                      )

18                         Add difference squared:            19                                 20                     } 21                 }

22                 Add sum to total:        √        ⁄

23             }

24             Final cost:                       ⁄

25         } 26     } 27 } 28 Free candidate patch GPU memory

Listing 6.10: First version of our GPU cost calculation process

The second version (Listing 6.11) increases the block dimension to     to allow four threads to run

concurrently. This divides up the work so that each thread works on a component of the cost

function when looping over the control points. Each of the segments is broken up in two, centred on

the control point’s location as per Figure 6.2. Because there is now more than one thread doing

calculations, we make use of a small amount of shared memory to store the cost value from each

threads calculation. A useful observation made is that the only variation in code, where branching

occurs, is the calculation of the segment positions. After that the same code path is executed for all

four threads concurrently with their results stored in separate shared memory locations. The final

step is to synchronise the threads and then have a single thread perform the final summation of the

costs from the other ones stored in shared memory. This value is then divided by the number of

control points to produce the final cost for this candidate.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                   6 } 7 Loop over source patches { 8     Loop over number of source transformations { 9         Execute candidate patch extraction kernel (Listing 6.9) 10         Init kernel:                                       11         Execute cost calculation kernel { 12             Allocate shared memory:                 13             Loop over control points { 14                 Calculate points for segments   and   (Figure 6.2) 15                 Run profiling for segments   and   { 16                     Initialise:               17                     Iterate over all points on segment { 18                         Calculate difference: (                      )

19                         Add difference squared:            20                                 21                     } 22                 }

23                 Add sum to total for thread x:         √        ⁄

24             } 25             ~Synchronise threads~ 26             If thread.id == 0 { 27                 Total cost:

28                 Final cost:                             ⁄

29             } 30         } 31     } 32 }

33 Free candidate patch GPU memory

Listing 6.11: Second version of our GPU cost calculation process

The third version also makes use of shared memory and the same block dimension of    , but

changes several other aspects. The majority of the algorithm is the same as the second version

(Listing 6.11). A total of                         of shared memory is used for this version. An

‘abs’ function was removed because its inverse was placed inside an ‘if’ clause which was already

executing, to remove the performance hit that an abs function has on the GPU pipeline. Another

improvement was to not initialise the shared memory to a zero value. Instead, during the looping

over the control points, the first iteration will set the value with successive calls adding to the cost.

The function that performs the cost calculation with the given edge was unrolled and placed directly

into the kernel for a slight performance gain. A single thread is still required to sum the total cost

after a synchronisation before the kernel returns.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                   6 } 7 Loop over source patches { 8     Loop over number of source transformations { 9         Execute candidate patch extraction kernel (Listing 6.9) 10         Init kernel:                                         11         Execute cost calculation kernel { 12             Allocate shared memory:                   13             Calculate points for segments   and   (Figure 6.2) 14             Initialise:               15             Iterate over all points on segment { 16                 Calculate difference: (                      )

17                 Add difference squared:            18                         19             }

20             Add sum to total for thread x:         √        ⁄

21             ~Synchronise threads~ 22             If thread.id == 0 { 23                 Total cost:

24                 Final cost:                             ⁄

25             } 26         } 27     } 28 } 29 Free candidate patch GPU memory

Listing 6.12: Fourth version of our GPU cost calculation process

The fourth version (Listing 6.12) attempts to use additional threads, to iterate over the additional

control points as separate threads. This means individual threads have no loop structures. We

multiply the current block dimension of     by a set maximum number of control points of 32 to give

a dimension      . The previous work we were extending had a hard limit of 32 control points,

which we adopted as well. However, this hard constraint is not advised for varying patch sizes as the

number of control points varies with the size of the patches. We now unroll the loop over all the

control points and use some logic to work out which control point the thread is working on. The

shared memory amount is also increased to                         , so each thread has its

own piece of memory. The process ends with the single thread totalling all the shared memory

blocks. This method proves inefficient as there are not always 32 control points which results in

threads being idle and wasting executing time. After the development of some of our advanced GPU

implementations we discovered that the system in fact uses no more than three control points for

our given patch size. Adjusting values down to four yielded significant improvements in this version

and is re-integrated in our advanced implementations in version eight. The explanation for the poor

results was the idling of the vast majority of threads, which were not required and a waste of

resources.

## Advanced GPU versions

The next four GPU versions are different in that only a single loop is used, which iterates over the

source patches. We allocate enough memory to hold all the different transformations of a source

patch                                    . A kernel with grid dimensions

(

) and block dimensions         is used to efficiently extract the source

patch, perform the required transformations and then store all of them (Listing 6.13). Versions 6-8

share a common grid dimension                for the cost kernels. The code now diverges for

each of the four advanced GPU versions.

1 Inputs: Source File, Patch Offset 2 Output: Candidate patch stored in GPU memory 3

4 Init Kernel:         (

)

5 Execute extraction kernel { 6     Calculate lookup coordinates for sketch for all transformations 7     Calculate memory address for output 8     Extract pixel data from source 9     Store data in user patches memory 10 }

Listing 6.13: Overview for the advanced candidate patch extraction kernel

Version five (Listing 6.14) is a modified implementation of version three that makes use of

additional blocks, which runs each of the ten transformations. The grid dimensions are expanded to

with the block dimensions still set at four threads. There is a slight speedup

due to splitting up the workload to a greater degree.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                                    6 } 7 Loop over source patches { 8     Execute candidate patch extraction kernel (Listing 6.13) 9     Init kernel:                                                        10     Execute cost calculation kernel { 11         Allocate shared memory:                 12         Loop over control points { 13             Calculate points for segments   and   (Figure 6.2)

14             Initialise:               15             Iterate over all points on segment { 16                 Calculate difference: (                      )

17                 Add difference squared:            18                         19             }

20             Add sum to total for thread x:         √        ⁄

21         } 22         ~Synchronise threads~ 23         If thread.id == 0 { 24             Total cost:

25             Final cost:                             ⁄

26         } 27     } 28 } 29 Free candidate patch GPU memory

Listing 6.14: Fifth version of our GPU cost calculation process

Version six (Listing 6.15) is an extension to version five, but instead of using more blocks for

processing we exploit part of the initialisation in the feature profiling algorithm. For each of the

control points in the user patch, the four threads work out their corresponding offset location for

each of the segments. Then they compare the height profiles with the candidate patch. Each of the

threads will calculate the initial part and then when it comes to comparisons with the candidate

patches, we add an additional loop which will check each of the candidates. The shared memory is

increased so that each thread can store its information for each of the ten candidates it is evaluating.

The process ends with a single thread totalling the costs from each of the threads for each of the

candidates, producing ten cost values. Due to the additional looping done by single threads, this

version works best when there are a large number of user patches to evaluate to ensure the GPU is

fully saturated to offset the looping overhead.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                                    6 } 7 Loop over source patches { 8     Execute candidate patch extraction kernel (Listing 6.13) 9     Init kernel:                                       10     Execute cost calculation kernel { 11         Allocate shared memory:                    12         Loop over control points { 13             Calculate points for segments   and   (Figure 6.2) 14             Initialise:               15             Iterate over all points on segment { 16                 Calculate difference: (                      )

17                 Add difference squared:            18                         19             }

20             Add sum to total for thread x:         √        ⁄

21         } 22         ~Synchronise threads~ 23         If thread.id == 0 {

24             Total cost:

25             Final cost:                             ⁄

26         } 27     } 28 } 29 Free candidate patch GPU memory

Listing 6.15: Sixth version of our GPU cost calculation process

Version seven (Listing 6.16) combines features from versions five and six, taking the additional

looping from six and using the extra dimensions from five for the processing. The grid dimensions

are still set to the number of user patches but the block dimensions are extended to       . The

shared memory is thus increased to allow for the ten transformations for each of the four

components. But instead of adding an extra loop as done in version six, we use the extra dimension

of threads to calculate the costs. While this version performs better than version five, it can at best

perform the same or slightly worse than version six. It also requires terrains with large numbers of

user patches to fully saturate the GPU with enough independent blocks.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                                    6 } 7 Loop over source patches { 8     Execute candidate patch extraction kernel (Listing 6.13) 9     Init kernel:                                                        10     Execute cost calculation kernel { 11         Allocate shared memory:                    12         Loop over control points { 13             Calculate points for segments   and   (Figure 6.2) 14             Initialise:               15             Iterate over all points on segment { 16                 Calculate difference: (                      )

17                 Add difference squared:            18                         19             }

20             Add sum to total for thread x:         √        ⁄

21         } 22         ~Synchronise threads~ 23         If thread.id == 0 { 24             Total cost:

25             Final cost:                             ⁄

26         } 27     } 28 } 29 Free candidate patch GPU memory

Listing 6.16: Seventh version of our GPU cost calculation process

Our final GPU version (Listing 6.17) takes another look at the less impressive version four and

attempts to improve its performance. We optimised the looping mechanics to prevent unrequired

threads from executing code thus resulting in it performing better than the original version four. We

also performed testing to determine a better limit for the control points and never observed more

than four control points for any of the input files we made use of. The improvements from version

seven are also incorporated, which brings the block dimensions to        . This improves

substantially on version four and doubles the speed of version seven. We then investigated making

use of texture memory to improve performance further.

1 Inputs: Source Candidates 2 Output: Calculated costs for candidates – Stored in GPU memory 3  4 Allocate memory on GPU { 5     Candidate Patch:                                                    6 } 7 Loop over source patches { 8     Execute candidate patch extraction kernel (Listing 6.13) 9     Init kernel:                                                         10     Execute cost calculation kernel { 11         Allocate shared memory:                     12         Calculate points for segments   and   (Figure 6.2) 13         Initialise:               14         Iterate over all points on segment { 15             Calculate difference: (                      )

16             Add difference squared:            17                     18         }

19         Add sum to total for thread x:         √        ⁄

20         ~Synchronise threads~ 21         If thread.id == 0 { 22             Total cost:

23             Final cost:                             ⁄

24         } 25     } 26 } 27 Free candidate patch GPU memory

Listing 6.17: Eighth and final version of our GPU cost calculation process

## GPU Texture Memory

We added an option to enable the use of texture memory on the GPU for the user sketch and

source images. This affects the user patch extraction kernel, along with the candidate extraction

kernel used in each of the GPU versions for cost calculation. Section 8.1.4 presents the results

obtained from using texture memory. The implementation requires few code changes from the

implementations in Listing 6.9 and Listing 6.13, a bound texture reference is needed for the lookups

instead of passing in an array. The system is capable of interchangeably using texture memory or the

global memory array directly during synthesis.

6.4.4 Storing Best Candidates

Once the cost kernels have been executed for all of the source files, the data needs to be sorted

and only the best set of candidates retained. This set then needs to be relayed to the CPU for the

merging process to begin. We develop two different implementations for this process; the first

simply transfers all the raw cost data to the CPU for sorting, while the second performs the sorting

on the GPU.

The first implementation relies entirely on the CPU to perform the sorting operation. This requires

the raw cost data be transferred back to host memory. We made use of the built in C++ stable sort

algorithm. Once transferred, we create a data-structure that associates the candidate indices and

cost values. This is now sorted in ascending order of cost, with the five lowest cost candidates being

saved. This processed is repeated for all of the user patches and the results sent for merging.

Transferring the raw data to the host has a high time cost due to the large amount of data being

transferred, sorting on the GPU and transferring only the best-set data would, in theory, be more

efficient.

1 Inputs: Candidate costs for each user patch 2 Output: Set of 5 best candidates for each user patch 3  4 Allocate memory on GPU (                          ) 5 Init kernel:                                      6 Execute sort kernel { 7     Loop over number to keep: 5 { 8         Loop over #CandidateCosts { 9             Store lowest cost candidate index 10             Set cost to infinity to indicate stored 11         } 12     } 13 } 14 Create array on host (              ) 15 Copy data from device to host 16 Build best set from index values for

Listing 6.18: Algorithm for sorting candidates based on cost in ascending order

The GPU sorting implementation has two different modes: the first is our own written kernel

(Listing 6.18) to sort the data and the second uses an external library, Thrust (2013), which includes

functions optimised for sorting. Our sorting kernel has a grid dimension                and block

dimensions    ; this provides a single thread that runs over two loops. It is based on the observation

that a full sort is not required. The outer loop iterates over the number of candidates we chose to

retain (5), searching the entire set of candidates looking for the lowest cost. When found the index

for the candidate is recorded and its cost set to infinity before the next iteration. Once the kernel is

finished executing the results are transferred to the host, so the CPU can assemble the best set

candidates and pass them to the merging process. The use of the Thrust (2013) library provides a

function that is highly optimised, to replace our sorting kernel, while the rest of the process remains

the same. The results for all three variants of this process are presented in section 8.1.5.

6.4.5 Merging

Now that the best-set of candidates is available back on the CPU side, they can be processed for

merging. The merging process is not ported to the GPU in our research and is thus done on the CPU.

Section 6.5 describes the process for merging; while the CPU is processing some of the data for

merging, the GPU continues building up new sets of data asynchronously as part of the blocked

design. This splits the workload into smaller chunks so that the CPU can be processing a chunk while

the GPU is preparing the next; this is described in section 6.6.

https://lh3.googleusercontent.com/notebooklm/AG60hOpaybhQcQ7WSbmvxWNQZ9ASR33bz8CO7i3Q4C_S5mGDtfetZKnM1PJXHjyQOvQByWGIAs3puj9BP9SDCdHSDdm6FshF9joKiz6DGe4eS7alDjRXo1vEACbJpwykErk_UV9DQe3V=w866-h344-v0

5b8d3396-6c24-4054-b37b-862f2e6f3ce4

6.5 Feature Merging

Figure 6.6: Overview of the feature merging pipeline: a) Single-threaded pipeline, b) Internal block for multithreaded version.

The merging process used in our system remains largely unchanged from that of Tasse et al.

(2011), as described in section 4.3. Figure 6.6 shows an overview of the merging process with the

parallelisable components separated out. Once the matching system has completed for all the

source files and the best set is produced, the merging process takes over and a second more

comprehensive round of cost calculations is undertaken. This is part of the merging core because the

information already synthesised in the output terrain is used. The combination of SSD and Noise

Variance allows a recalculation of the cost values based on the synthesised data. This list is then

resorted and only a small subset of five candidates is evaluated against the Graph-cut cost due to

the large computational overhead it carries. The overall best candidate is selected and passed onto

the merging process.

The merging process is responsible for placing the best candidate into the output terrain. Simply

pasting the candidate directly into the output terrain will result in obvious artefacts with a very

pronounced straight boundary due to the difference in pixel values.  For the result to appear

seamless, a combination of techniques is used. First a Graph-cut (Section 4.3.1) is performed on the

overlapping area to find an optimal cut path. However, this seam remains visible, which requires

further processing. Shepard Interpolation (Section 4.3.2) is used to deform the pixel data around the

seam so that the pixels on both sides have similar values. This process works well with a top-down

view of the 2D image showing no visible artefacts. However, a 3D rendering of the terrain shows that

the gradient values along the seam are not matched, revealing an unnatural discontinuity. Tasse et

al. (2011) correct this artefact by performing seam removal on the image gradient field of the

overlapping region. The final elevations for this patch are reconstructed from the modified gradient

field by solving a Poisson equation (Section 4.3.3). The patch has now been successfully placed into

the terrain free from any visual artefacts and the process can proceed to the next patch.

The cost computation part of this process is enhanced with multithreading to better distribute the

workload. The recalculation of the costs for all the best-set candidates is divided equally amongst a

set of threads, after which a single thread sorts the data. The Graph-cut cost is only executed on the

top five candidates, each of which is run in a separate thread. The actual merging of the patch into

the output terrain is done on a single thread.

https://lh3.googleusercontent.com/notebooklm/AG60hOq3nJXgwRt4MnEbj2TXFf504aWCulLLo2c5NAsjAMg1tOyJ6AUaunFZL8yycipSepVE5E6r229uLcY6tFRZqyH42_LL55YTTIs7BojD7UTsijatCtNgtD-IGC884-L__b5etnC0rg=w1060-h550-v0

28b730fb-d2bd-475c-ac0f-fd30ff1090de

6.6 Optimisations Throughout the implementation of the feature synthesis a number of optimisations were applied

to improve the system. The two cost calculation rounds are one such example; these were described

in sections 6.2 and 6.3. Due to the system architecture changes required for multiple input sources it

is not possible to run some of the cost functions without data from the output terrain.

Figure 6.7: Example of repetition in output terrain. (a) Repetition with adjacent patches (b) Repetition check implemented to overcome this issue

There is a chance that during synthesis the same candidate may be chosen for successive user

patches. This leads to a repetition of adjacent patches in the output terrain, which is visually jarring.

In order to fix this issue we keep track of the last placed candidate, which is penalised with a very

high cost to prevent it being selected, unless there are no other matches.  Since our system only

retains a maximum of five candidates for each user patch, we have a limited amount of data to

choose from when preventing repetition. This is a trade-off with the amount of memory available to

the system as keeping more candidates would require much more memory and increase the time to

complete synthesis. Figure 6.7 shows an example of such repetition artefacts as well as the same

user sketch being synthesised with our fix to prevent repetition in adjacent patches. This

improvement only retains a reference to the last placed patch, which does not help if adjacent

patches are not processed successively. As such the repetition artefact may still occur. To address

this, a secondary data-structure could be maintained and queried at merge time to ensure that the

same patch was not being placed adjacent or even in close proximity. This would add further

complexity for the system and was chosen to be left for future work. Implementing this concept

would require us to store more candidates as it would shift the limitation of repetitiveness to how

many patches there are to choose between.

During the feature extraction process, the distance between successive nodes may be as little as a

few pixels, which will result in a significant overlap of patches. This would be extremely wasteful

since we would end up synthesising the same area multiple times. To address this we only generate

patches that have a minimum linear distance of    ⁄    the patch size between neighbouring nodes.

This can dramatically reduce the number of user patches, especially in larger terrains where the

https://lh3.googleusercontent.com/notebooklm/AG60hOoHbTRxAi634OuQVTmuJrNtmCY3E57784pMavfTXbr9sQD7XWyamJVrZPIJCmNR9SI3hmYJ56jc28MCEZpgbst7wb0ypB3pVoJ_hdjwNzQkqeLKDaquqETNZIwTZ2pIvLMlJm1j=w1060-h550-v0

b5a6e97b-c5c3-4e98-be9a-5d9c0005b70c

feature extraction process produces errors with edges forming closely spaced parallel lines (Figure

6.8). Improvements to the extraction algorithm could potentially solve this issue.

Figure 6.8: (a) Example of error with feature detection engine forming multiple parallel lines. (b) This results in heavy overlaying of patches, which wastes performance.

Pre-loading all the data into memory means the system only needs to read it from disk once,

which saves on the transfer time from disk. This allows us to improve performance over the system

designed by Tasse et al. (2011). Their system extracts the pixel data for every candidate from the

source file and stores it in memory during the cost calculations. This method quickly exhausts all

available system memory and could potentially crash when processing source files with large

numbers of patches. A source file with 5,000 patches would expand to 50,000 candidates under the

different transformations, which would require 800MB worth of memory to store just the

pixel patch data. There is also a large time overhead in extracting all the candidates before

computing their cost. We developed our system to keep only the original source file in memory and

do direct memory lookups during cost calculations. This allows us to process very large source files

with thousands of candidates. Once the best patch is located by the system, its pixel data is

extracted and passed to the merging subsystem.

When developing the GPU implementation many optimisations from the CPU implementation

were retained. Pre-loading the data onto the GPU device allows for much faster synthesis as the

largest overhead with GPU programming is copying data between the host and device. We have also

optimised the GPU kernels through proper use of the available memory types (Section 6.4.3).

https://lh3.googleusercontent.com/notebooklm/AG60hOoape3aMvnXQKevWE8aGCXyhCHsugC_FAaZjQDIWwEnT_oDtgS0-xRarjDS_tHt60Rm5Rgl68Kk_vwfGt8bnMu7ihYVj_1YapCT_Sbq8wQXmYdQRpFC78xJ_fRIO506NTj0ySyS=w843-h357-v0

67855eec-acbc-4f4d-9c4f-26661e9d4025

Figure 6.9: Illustration of blocked design for candidate processing. a) A queue of blocks of length k that are sequentially processed by the algorithm in b) on the GPU. Results form a queue c) which is processed by the CPU in d)

The last important optimisation is the development of a blocked design, which splits the work into

a series of chunks for processing the cost calculations (Figure 6.9). This design became necessary for

very large user sketches with hundreds or even thousands of user patches. Our algorithm is designed

to process all the user patches in order to store the best set of candidates before the merging

process. But processing all the data in the limited high-speed memory available on a GPU is not

possible. Thus, it becomes necessary to divide the task into a series of smaller chunks. Another

benefit of this blocked design is that after a block has been processed by the GPU, its results are

transferred back to the CPU to begin the merging process. At that point the next block is processed

by the GPU. This allows for a balance of the CPU and GPU processing capabilities. There is a small

overhead with managing the various processes but as shown in section 8.1.5 there is a reduction in

the synthesis time when balancing the tasks asynchronously. As the CPU is slower at processing than

the GPU, these changes ensure that the CPU is never idle during the synthesis process.

There are several improvements that have not yet been implemented and are listed in section 9.2

for the future work. Further GPU optimisations can be achieved by exploiting the functionality

provided on new generation devices. Next we look into the non-feature synthesis process.

https://lh3.googleusercontent.com/notebooklm/AG60hOrRGOGiXwVGyHhIXwZjfKvW8IsPtSqSNGjsxC6bnrQWCTPIFTMZ_drqxzVWdokIv14xwD3SEZfzNxXlmi8WaIp9mQkfEqws_Ui_DSTfIyaBW8nPBPg2PuBV5jPGuVe0ERSSm4zWEg=w561-h325-v0

05714414-3e67-4806-bcb8-843ce8112083

7 Non-Feature Synthesis

Figure 7.1: Non-feature synthesis pipeline showing flow of data for the Non-Feature Matching & Merging block of our system (Figure 5.1)

After all the user features have been matched by the system there are areas in the output that

have no data and appear as holes in the terrain. Non-feature synthesis is the process of filling these

holes with data from the source files that contain no strong features. This process sequentially

selects an area, matches a suitable candidate and then merges the data. This is sequential due to the

matching process relying on data in the output terrain, which is updated with each iteration. There

are few subroutines that can be accelerated through parallel processing, such as the cost

calculations, which we implemented on the GPU. Figure 7.1 shows an overview of the non-feature

synthesis process, which is described in the rest of this chapter.

7.1 Candidate Extraction Candidate patches that contain no strong feature data are required for the non-feature synthesis.

These patches are extracted from the source files using a modified feature extraction algorithm. As

we need the overall synthesis results to be repeatable, the extraction algorithm cannot have any

randomness in the selection process, unless a seeded random number is used for repeatable results.

We designed our system to examine fixed intervals along the   and   axes of the source files for

suitable candidates. This gives us a fixed number of candidates and the interval is calculated as

where          is the width of the source file. This is calculated for both axes, which gives us the

coordinates to iterate over. These form the centre points for the patches to be extracted. Each of

these locations is then compared to the feature extraction node list to make sure that the patch

location is not within

pixels of a feature node. This ensures that we avoid patches with

strong features being used. Finally, each of the valid patches are extracted from the source file and

then the next source file is then examined. We purposely chose a small number of candidates per

source file to limit the complexity of the overall system and the impact on performance. This was

due to the non-feature synthesis process not being ported to the GPU due to time constraints; this is

noted as future work in section 9.2.

Candidate extraction is run as a pre-processing step at the beginning of non-feature synthesis; all

of the valid patches from all source files are cached in memory for later processing in this stage. To

increase the number of candidates for the system, each patch is rotated and mirrored in the   and

planes similarly to feature synthesis. These candidates are now examined during the matching stage.

7.2 Candidate Matching and Merging The system now loops continuously until there are no longer any holes in the output terrain.

There is no way of pre-computing the number of non-feature patches that need to be placed to

complete the process. This is because the boundary is continually changing with each merge

operation, with each placing a different amount of data into the output. Patches are placed at

locations on the boundary only so as to provide some valid data to facilitate merging. This

unpredictability prevents us from determining the runtime. A single iteration comprises selection,

matching and merging processes. The system has no smallest size of hole that it will match and will

handle a hole consisting of a single pixel, which ensures the output contains no invalid data. Once all

the holes have been filled the synthesis process concludes and the final image is displayed.

7.2.1 Selecting Target Patch

For the first iteration the system needs to evaluate the output terrain to identify all the locations

that are on the boundary of placed data and an empty area. This is done once over the whole image

and then, after each merging operation, only the affected area is updated. Each of these boundary

locations has an associated priority value, which is based on the already placed data around the

patch location. At the start of non-feature synthesis, the current output image is processed to build

up a list of priority values for the boundary pixels (Listing 7.1). This list is sorted at the start of each

iteration, the location with the highest priority is selected for the current matching and merging

operation. The details of this algorithm are discussed in section 4.2.2.

1 Loop over output pixels { 2     if pixel valid && on boundary { 3         Calculate priority and add to list 4     } 5 }

Listing 7.1: Algorithm overview for building boundary dataset

7.2.2 Matching – Cost Functions

Non-feature synthesis makes use of three of the same cost functions used for feature synthesis,

namely sum-of-squared differences, noise variance and graph-cut. These cost functions are well

suited as they make use of already placed data in the output terrain. We again make use of two

rounds of cost calculations. This lets the system only compare a subset of candidates from the first

round against the computationally expensive second round. The first round compares the candidate

patches with the Sum-of-Squared Differences (SSD) (Section 6.2.2) and Noise Variance (Section

6.2.3) cost functions. After the candidates are evaluated and sorted, the five lowest cost candidates

are evaluated against the Graph-cut cost (Section 6.2.4) function and the best patch is selected for

7.2.3 Matching – CPU Implementation

Our CPU implementation for non-feature synthesis is adapted from Tasse et al. (2011), with

additional enhancements. An overview of this process is provided in Listing 7.2 and the CPU pipeline

is illustrated in Figure 7.1. The process starts by extracting the candidate patches from all the source

files and storing them in memory (Section 7.1). The boundary dataset is now initialised. This

evaluates all the points that lie on the boundary of synthesised data and a hole and computes an

associated priority value (Listing 7.1). The process now enters a hole-filling loop, which only

concludes when there are no remaining holes in the output terrain.

1 Extract candidates from all source files 2 Populate boundary dataset 3 Loop until no holes { 4     Select target patch 5     First round cost calculation 6     Sort results 7     Second round cost calculation 8     Sort candidates 9     Merge best patch 10     Update boundary dataset 11 }

Listing 7.2: Algorithm overview for the CPU non-feature matching implementation

The first step is to select a target patch for matching (Section 7.2.1). Once a patch is found, its

pixel data, where it exists, is extracted from the output terrain and passed to the first round of cost

calculations. For each candidate patch, the SSD is calculated for the areas that correspond to valid

data in the target patch for each of the different transformations. The noise variance for all the

candidate transformations are identical, thus only one comparison is required and the result is

added to each of the candidates total cost. After all the candidates have been evaluated, they are

sorted in ascending order and the five with the lowest cost are run through a second round of more

comprehensive cost calculations. The second round uses the Graph-cut cost function to measure the

quality of the merge for the patch. The results are sorted once again with the overall cheapest patch

being selected for actual merging. The boundary dataset is now updated around the modified

location before the process repeats, continuing until all the holes are filled.

The only parts of this implementation that benefit from parallelism are the cost calculations,

which could be distributed amongst a group of threads. However, due to time constraints we chose

not to implement a parallel CPU implementation and instead invested time in a basic GPU

implementation where we parallelise the cost calculation stage. There is room for improvement of

non-feature synthesis through better optimisation and parallelism, but this is left for future work

(Section 9.2).

https://lh3.googleusercontent.com/notebooklm/AG60hOqi5VWkIsFosOiprtibgD0V6O0Ko1Kyx1tm66XAxla6M2CP0PX-7Fkms3WLqSCbMBxcc9UXvqQM5XLVxjmDTplmPESKQuSiLnr8Y8BRRMGL-SnE4uzlQz6IlRFhcYfNQbFVHx7f=w651-h368-v0

8812baea-1bd6-49d0-bb96-8c2d4c0c7f54

7.2.4 Matching – GPU Implementation

Figure 7.2: Overview of the GPU non-feature matching pipeline. Candidates are cached on the GPU initially. The system then loops until all ‘holes’ are filled. GPU acceleration is used to calculate the costs with the rest being done on the CPU.

The GPU implementation we have developed is a basic attempt at parallelising non-feature

synthesis; only the cost calculation and sorting are performed on the GPU with the rest done on the

CPU. An overview of this process is presented in Listing 7.3. The process starts with caching all of the

source files on the GPU, but this is normally skipped as the source files will have already been cached

on the GPU as part of feature synthesis. The candidate extraction process starts with allocating

memory to store all the candidates in GPU memory. We then loop over all the source files and

calculate all the valid candidate positions as per section 7.1. These positions are now transferred to

the GPU and an extraction kernel is executed, with grid dimensions of

(

) and block dimensions        . The pixel data is

extracted from the source file and stored in GPU memory before the next source file is processed.

1 Cache source data on GPU (CPU -> GPU) 2 Execute candidate generation (GPU) 3 Populate boundary dataset (CPU) 4 Loop until no holes { 5     Select target patch (CPU) 6     Transfer target patch data (CPU -> GPU) 7     First round cost calculation (GPU) 8     Sort results (GPU) 9     Transfer best-set to CPU (GPU -> CPU) 10     Second round cost calculation (CPU) 11     Sort candidates (CPU) 12     Merge best patch (CPU) 13     Update boundary dataset (CPU) 14 }

Listing 7.3: Algorithm overview for the CPU non-feature matching implementation

The boundary data set is now populated, as in the CPU implementation, and this concludes the

pre-processing stage. The system will loop until all the holes are filled and synthesis is complete. The

CPU selects the next target patch based on the priority values for the boundary pixels. The

coordinates are sent to the GPU and used to extract the target patch pixel data from the output

terrain and transfer it to GPU memory. Due to the output changing after each merge operation,

which is done on the CPU, there is no copy of it stored on the GPU. A CUDA kernel to compare the

costs is now executed in two parts, the first part calculates the SSD and the second the noise

variance of the target patch. The SSD calculation has grid dimensions of

and block dimensions            , which allows each

of the candidates and its associated transformations to be run as separate blocks. With the number

of threads equalling the patch dimensions, each one has an assigned   coordinate and calculates the

SSD value for all of the   coordinates. There is now a synchronisation phase, because the threads all

execute independently and we require all threads to have finished executing and writing their

results to memory. A single thread now sums up the SSD values from each of the other threads and

stores the final SSD cost in an array on the GPU.

The noise variance is then calculated for all the candidates using a kernel with grid dimensions

and block dimensions    . Because the noise variance for a candidate is the

same irrespective of its transformation, we need not calculate it for each transformation. After the

cost is calculated the total is added to the cost array from the SSD stage, the cost is also added to

each of the candidate transformation cost values. This process completes when all the source files

have been evaluated. The final cost array is now sorted on the GPU, in ascending order. We designed

a simple kernel for this process but also developed a version that makes use of the Thrust (2013)

library for improved performance. This is similar to feature synthesis candidate sorting (Section

6.4.4). The best subset of five candidates is now returned to the CPU where they undergo the

second round of cost calculations. This was done on the CPU, as our simple port of the code to GPU

resulted in a negative speedup due to the algorithm complexity.

The graph-cut cost function is now used to compare each of these five candidates, which are then

sorted with the lowest cost one being chosen as the best overall. This patch is sent to the merging

process to be placed into the output terrain. When this process is run for the first time candidate

generation process is required, but on successive iterations this step is skipped and only the

calculation process is run with the new target patch. Once all the holes are filled the data is deleted

from the GPU memory.

7.2.5 Merging

Once the best patch has been selected it is merged into the output terrain. This process is almost

the same as that detailed in feature synthesis (Section 6.5). The difference is that the merge step

does not conclude the synthesis. Since the merging process alters the output terrain, the boundary

needs to be updated for target selection. This is done using the location of the patch and the process

then continues with selecting a new target. If there are no remaining targets the synthesis concludes

returning the final result to the user.

7.3 Optimisations During the implementation of non-feature synthesis a number of optimisations were applied to

improve the synthesis process. An important optimisation improved the memory utilisation of

candidate extraction: in order to limit the amount of memory required, only a single patch is stored

for each candidate. Thus, only 90° rotations and mirrors can be supported since 45° rotations require

data from the surrounding patch. This reduces the number of candidates per patch to six, down from

ten for feature extraction (Section 6.1). This is more efficient as we can easily change the direction

when reading the pixel data and we do not need to perform interpolation or store a larger area to

include pixels on the edges of a 45° rotation.

By pre-processing all the source files and keeping the candidates cached in memory, we are able

to quickly evaluate all of them against the selected target patch without the need to loop over the

source files. This is possible due to the low number of candidate patches taken from each source file,

which keeps the memory requirements relatively low. This is important for the GPU implementation

as swapping data in and out would negate any performance gains. We kept the number of

candidates low since the process was not ported to the GPU and increasing the candidates would

hamper performance more. As part of the future work, the number of candidates could be increased

after the system is ported to the GPU.

As part of the optimisation process for feature synthesis, we developed a multithreaded graph-cut

cost function. As the data input for this function is the same for the non-feature synthesis, we are

able to reuse this to give a slight performance increase.

https://lh3.googleusercontent.com/notebooklm/AG60hOp--optoYBHp8O-qssBgNzB0TnFAFF5VI2rj1piRIH4boARpILmIWBbmWMMN7gp7peEGA3N-B3cLsygdaVEFJuWte0yvBhtRZMB_UoL4zsaJp-32wB4R-fELvylulgjAc_N0X7cQw=w1060-h550-v0

78df466c-cecd-4c8a-87dc-345da7e4375f

In this chapter we present the results and evaluate the extent to which we met our initial goals.

Our research has two primary goals; firstly to extend the work of Tasse et al. (2011) by utilising a

large collection of source files for the synthesis operation; and secondly, to improve system

performance to counteract the inclusion of many more source files and allow the creation of larger

more complex terrains. Evaluation takes the form of visual assessment of the terrains generated by

our system, compared to the previous system as well as the use of performance metrics to evaluate

the speed.

The test system we utilised features an Intel Core i7 processor with 16GB of RAM and an NVIDIA

GeForce GTX 660 Ti GPU. All experiments were run on a fresh reboot of the computer with minimal

other processes running; tests were run ten times with the average time reported to allow for

representative performance figures. We designed two test images for synthesis, a small

one and a large           one (Figure 8.1). These have a different number of user candidates to

match against to increase the complexity (Table 8.1) as calculated from the feature extraction

algorithm. We have separated the ridges and valley runtimes to show values for different numbers

of features. If the number or ridges and valleys were the same then near identical runtimes would

be observed as the algorithm is the same.

## Small Terrain Large Terrain

## Dimensions

Ridges 38 68

Valleys 18 900 Table 8.1: Number of detected user features patches and dimensions of the two main test terrains we use. Difference is

ridge/valley count is determined by feature extraction and dependant on sketch used.

Figure 8.1: The two test images used for evaluation. a) The small         terrain. b) The large           terrain. The white lines represent ridges with the black lines being valleys as detected by the system.

The results are organised into three sections. The first two sections cover each of the main

components of the synthesis engine, namely feature and non-feature synthesis. The third section

covers the system as a whole, where the comparison with the previous work is discussed. All the

stacked column charts in this section provide the runtime in seconds with the columns made up from

the major contributing stages of the algorithm; Patch Matching and Patch Merging. Speedup graphs

are also presented to show the performance gains obtained.

8.1 Feature Synthesis We began by evaluating the feature synthesis component of our system. As feature synthesis is

the first part of system, the images we generate contain holes, which are filled in the next stage of

synthesis. We start off with a very basic CPU implementation, which is based directly off Tasse et al.

(2011). Each GPU test builds upon the previous one with, the first version being unoptimised and

each of the subsequent optimisations being covered in their own subsection. Some of the

optimisations make use of different tests; such differences are noted in the description.

8.1.1 Sequential CPU versions

The first test we conduct compares our two sequential CPU versions (CPU v1 and CPU v2) to

evaluate which one is better suited for further development. The first version closely matches that

of Tasse et al. (2011), which results in repeated loading and unloading of the source files into

memory. Our improved version inverts the loop processing logic, thus only loading each source file

once, as well as caching the user patches in memory for faster access.

From the results in Figure 8.2 we observed that there is very little speedup overall for our second

version. If we look at the raw runtime values in Table 10.1 we see a dramatic reduction in time spent

regenerating the source candidates, especially on the larger terrain. This value, however, is very

small and insignificant in the overall synthesis. Since the source files are only loaded once, the time

for pre-processing is the same across different terrain complexities. This result motivated us to

continue the development of the second version, as it is expected to scale to both larger terrain sizes

and a larger number of source files. There is only one disadvantage to the architecture change

between the two versions. The first one evaluates every user patch against all source candidates

before selecting the best one and merging it. This allows for already merged data to be taken into

account to improve selection criteria and result in a better overall merged patch. In order to

compensate for this, we keep a list of possible matches in our second version. This list of candidates

is evaluated after the matching process completes in which merging is conducted and the candidates

are re-evaluated based on information from placing previous patches. Through testing we found

very little visual difference between our two versions.

https://lh3.googleusercontent.com/notebooklm/AG60hOqZdqd9lpipLVol7jswJfxpnZMuLLvVgSWv2IIvkzTRiTMgb4E376OTc10d0etu9A0VOuxjSk27XGTEiqJ6ZkcfSsod6gMWOi52UtZYE8UoiL30bafrt6TOTz-0gzjk8IkihzFmnQ=w850-h286-v0

f702ba73-7077-4462-88c5-674f5f7f4f39

https://lh3.googleusercontent.com/notebooklm/AG60hOpwtUdYpmjTRUW-9nMuJJWcuPa6kV1isPGHkbjLbw8FJevYpcxjF6n5aNQUZ-bokTol_rgJwhcxksWye61G11FP6JMaCyLD_zESUhjoHtB9Jp_wId1SyUzF4JiLP1iJX3XMEZn_yg=w850-h286-v0

962078fb-c679-4c52-bd53-5ea86fc58575

Figure 8.2: Runtime chart comparing the two main CPU implementations. These two implementations have very similar runtimes despite the large architectural changes between them. Table 10.1 gives the runtime numbers in a table and

reveals that CPU v2 is slightly faster than v1.

8.1.2 Single versus Multi-Threaded CPU

Based on the results of our sequential implementations we choose to further develop our second

version and utilise additional threads to reduce the synthesis time. The whole system could not be

multithreaded, or more specifically the iterative merging process, where each successive patch is

calculated based on information from already placed patches and thus introduces and order

dependency. Our test system features four cores with Intel’s Hyperthreading giving a total of eight

threads that can be run concurrently. The area targeted for multithreading was the cost calculation

process which accounts for over 95% of the total runtime; thus any improvement would have a

significant impact. In order to accommodate multiple threads, minor changes were required to

control the distribution of the workload. From the results in Figure 8.4, we observed a     times

speed improvement, which translates to a 35 minute reduction in time on the large terrain.

Our multithreaded implementation was relatively basic and unoptimised, but it served as a proof

of concept for our GPU implementations. From the results it is evident that there was much

potential for improving the running time by parallelising the cost calculation process

Figure 8.3: Runtime results comparing the parallel CPU implementation against CPU v2. Here we observe a large reduction in synthesis time almost reducing it by half on the large terrain. Full runtime values are presented in Table

https://lh3.googleusercontent.com/notebooklm/AG60hOqHiE1_7rTK4Mua6Nufr5Yc_Lp1UIMBg9dn5abSD-guhiEZ6YndBeOFp63hoAWG1Ow6lZCSV6kXQqHweRNKRsFV8ffYlf0VtkU9e6XgZVsdzlcKWcKYEcCzcBVNZcOHwY5pP5O99A=w910-h376-v0

e3eba118-ffbf-4d29-8887-d87b2e13e2bc

Figure 8.4: Speedup graph comparing the runtime in seconds and the observed speedup for the parallel CPU implementation over CPU v2. We observe a     times speedup achieved for both test terrains.

8.1.3 CPU versus incremental GPU implementations

Here we provide the results for our eight different GPU implementations and compare them to

the multithreaded CPU version (Figure 8.5). We began with our first GPU version (GPU v1), which is

simply the multithreaded CPU code translated to execute directly on the GPU and saw the synthesis

time reduced by half. Our second (GPU v2) and third (GPU v3) implementations explored adding

additional threads and makes use of shared memory to allow the threads to operate independently,

with a single thread summing the value at the end. This again saw our synthesis time cut in half,

although our third version actually performed slightly worse than the second. This can be attributed

to the additional if statement used to initialise the memory on the fly as it introduces another step

for the algorithm. The fourth version (GPU v4) again adds more threads, which allows an entire loop

to be executed in parallel on the system. We first tried providing 32 threads for each of the 4 cost

calculation components as this was the maximum defined control points in previous work. This

actually performed significantly worse than the prior implementations.

For our advanced GPU implementations we changed the underlying architecture of the system in

order to run each of the candidate transformations in a separate thread concurrently. Version five

(GPU v5) extends the work done in version three, which doubles the performance due to the

increase in concurrency, netting a    times speed improvement on the large terrain. Version six

(GPU v6) exploits some setup requirements in the cost calculation algorithm to avoid repeatedly

calculating the same value. This version performs slightly worse than version five but the difference

is insignificant. This is due to the setup overhead for calculating and storing the values up front.

Version seven (GPU v7) attempts to bring together the features of both versions five and six, but the

results fall between the two prior versions.

Our eighth version (GPU v8) revisits version four as we performed additional profiling and

discovered that there were never more than 3 control points required to describe the user and

source candidates under a variety of test cases for our given patch size of 64. We then adjusted the

number of threads down to 4 which gave a total of 16 threads required (down from the 128 needed

before). Running our tests again for version four yielded a modest improvement. We then included

the enhancements from version seven and achieved a total speedup of    times overall (Figure 8.6)

for the large terrain. The speedup achieved for the cost calculation process alone is    times faster

https://lh3.googleusercontent.com/notebooklm/AG60hOogvZsL-am0fmFw91ut-hiDv1hWRGnaN6_tZVjrFycgTtGFA7DvmXrSiPeFxRdBjiJ1lpMnstV3ugWPrRJ8xgktoftxxA6APdzJU_YF9liV6blklAZQFJu74p_ytx06hwLMb2-ZVg=w850-h286-v0

b1980910-e984-4b11-a938-88ffce12674a

https://lh3.googleusercontent.com/notebooklm/AG60hOo55sGETrNOaKWyLDb7crqClEhwfKEhj57qraeQEiZp3T0QcQXarNU0zQFDI7xhGu-eVQfSvoLtICyjnEWZ3o6xKts4lzV4HXn8rH-ITIYzntC82VOa4b7T9Bn4oyYdIEQ00hT3XQ=w850-h286-v0

1a3cce60-eacc-4bac-aa95-3cb962693b4c

https://lh3.googleusercontent.com/notebooklm/AG60hOoeAqm6C6xn_egblB70UUQiGN0SuMPMi7X2iXpDLKQJLUu0H9cccihLQu8dws1FJY_q0kPFhy1a8zxl7zBabhVa9FmLM_tcnvX5jz-pTlHwcda75ppWU5eP161srkJ95aqevTfEpg=w910-h353-v0

42c78183-5bd4-403f-8808-0525c5ad0ec2

than the CPU implementation. Runtime and speedup graphs for all GPU implementations compared

against the multithreaded CPU version are provided in Figure 8.6.

Figure 8.5: Runtime results for the eight GPU implementations compared against the parallel CPU implementation for the small and large terrains. We can see an overall downward trend to the graph with the times decreasing with each

iteration. v1 is a translated form of the parallel CPU implementation. v2 adds some shared memory and more threads. v3 attempts to optimise functions but introduces more branching. v4 unrolls an entire loop utilising more concurrent threads. v5 changes the architecture to allow a new dimension of threads for improved concurrency. v6 optimises v5 preventing unnecessary recalculation of values. v7 combines elements from v5 and v6. v8 revisits v4 and incorporates

the newer changes in v7. Full runtime values are presented in Table 10.3.

Figure 8.6: Speedup and runtime graph comparing the parallel CPU version against all eight GPU implementations. Similar performance is noted for both the small and large terrains, although a slightly higher speedup is noted for the

larger terrain.

https://lh3.googleusercontent.com/notebooklm/AG60hOo0QiHtHMvDlAphsgMRTO7R1tUYsCrceyJdX0Lwo2odyZLJcg4HNvhk9deNzaKsK3_3CbcZVZGaSW_i_m2unV5U0CNUL4HVBLeiigmHigQ-UbeMLsW1tg3QflrofhCFsLQbFzL4tQ=w850-h286-v0

021ae4be-6e89-4ef9-8781-5c2552203202

https://lh3.googleusercontent.com/notebooklm/AG60hOqkSc7BRFRnAXk0v760DyUpOA4ih13hERB1IXfHHOORM4CfKkaNcpRuubb667oMQubSrg2CAQZypdEU8ouosQnU9UO8DM7cJvqBjU8mQI9W5rjzIKFR5jwvNY7EmN37YjWDH-5Sqw=w910-h376-v0

785d97ed-ae07-497d-b1e1-d82aa79f2556

8.1.4 Utilising GPU Texture Memory

We added the use of texture memory to our system. This can be combined with any of the

aforementioned GPU implementations. Texture memory is part of the GPUs global memory, but is

specialised in that it is spatially cached, which allows for quick read access for neighbouring

locations. This is useful for our extraction kernels in which we extract pixel data from source images

and apply transformations to rotate the candidates. From the results in Figure 8.7, we observe a

small increase in performance for our final GPU implementation when using texture memory for

both user and source patch extraction. The performance increase is only minor as the existing

implementations already make use of coalesced global memory for optimal memory performance.

This brings the total speedup over our CPU implementation to    times, with the cost calculation

stage seeing a    times speedup for the large terrain.

Figure 8.7: Runtime results for our texture memory GPU implementation being compared against GPU v8. There is a slight performance gain when using texture memory. This is because we already are using coalesced memory access for

our image data. Full runtime values are presented in Table 10.4.

Figure 8.8: Speedup and runtime graph comparing the use of GPU texture memory against the parallel CPU and GPU v8 implementations. Using texture memory now brings the total speedup to 24 times fast than the parallel CPU

implementation.

https://lh3.googleusercontent.com/notebooklm/AG60hOrxXw_Lw1ChShgNAsQTuowB3_0mWdSnoctK4s6X1eZo6IJy98nf2ako2Kn_1esW1AKbNraOH7y99S6OgHyqdKquvSntG245Ax-H173dNvKEesDMSbcDwDK6_eKrqb4dSvGqz0yb1w=w850-h286-v0

51feb412-7f48-4d17-a261-0792ab046088

https://lh3.googleusercontent.com/notebooklm/AG60hOo_nKHD44OfcaylUKL8DJUPnltGNZbhNkbIkLK6We1jFIlh6BECJ6k6O6DuN4OSvFWfvDM0h4A9DyPLxtN4xEYxuyuICaYsd-sG2pwxCdr0B11d_rfk7AaIh_xHqcBrBElXQ3hyew=w910-h376-v0

8ebb06b5-91a2-4499-a1d3-036a6435be62

8.1.5 CPU versus GPU Sorting of Candidates

For our GPU implementations we developed three independent variants for sorting of the

candidates based on their computed cost values. The above tests were conducted using our own

sorting kernel (as described in section 6.4.4). An external library, Thrust (2013), was also used to

provide a more hardware optimised sorting solution to improve system performance. We also

developed a CPU sorting algorithm that makes use of the C++ stable_sort function (C++, 2015) to

compare against the GPU versions. From the results (Figure 8.9), we note that there is a modest

improvement by changing the method for candidate sorting. Candidates are sorted after the cost

calculation stage, which produces the best subset lists. These are then evaluated during the best

patch location stage, which again makes use of a sort to determine the best overall patch. The

Thrust version improves the speed of the entire system even further due to the code being highly

optimised and purposely designed for the hardware to extract maximum performance. Comparing

the speedup of the Thrust version over our previous best GPU implementation with texture

memory, we observe a total speedup of    times (Figure 8.10) for the large terrain. Thrust sorting

will thus be used for subsequent tests covered in this chapter.

Figure 8.9: Runtime results comparing the three different candidate soring functions. The Patch Matching component in the graph includes the sorting operation, which is why we see the green bars decreasing in size with the GPU and Thrust

(2013) implementations. Full runtime values are presented in Table 10.5.

Figure 8.10: Speedup and runtime graph comparing the three different candidate sorting functions. We see a modest performance increase when using the GPU for sorting, even with our simple kernel implementation. Using the Thrust

(2013) library further improves the result due to their kernel being highly optimised.

https://lh3.googleusercontent.com/notebooklm/AG60hOqN5kIseZujnkb4_Yyy2Oes_mGjL69qiZAKecXMCUmAPtDVpwly-q0qz-rr6ykHYjvkOQPw6tDPu91AQPbyxn2H-h4XuPAgbf42p_b0nXznApmE8O65ja4WcCkejnuZF4ZCG9TTjA=w850-h286-v0

9b03f073-fcc8-4e6a-aa6c-0bafe14acb49

https://lh3.googleusercontent.com/notebooklm/AG60hOrhGBtVVAzqwcN9ucgluVKV7uTJkq-JIoQKshXSiX-n2nQMwmTc-3jfwFacIq91MgYLEjC-SnOUJgM_NAdBpZ7CkT8QgjlMZuYO_DETU8KtLXrt1y8R0WIY20Lp39KWo0OLA1xbTA=w910-h376-v0

84fbd5ec-b9e0-48f7-b986-649a67b2bfa5

8.1.6 Blocked GPU for Asynchronous Processing

The last optimisation we developed was an asynchronous blocked design that would process the

user patches in groups of 50. When one batch is completed, it is sent back to the CPU where it is

processed for merging, while the next batch is processed on the GPU. This allows for asynchronous

processing by the CPU and GPU to potentially reduce idling.

The results in Figure 8.11 show an impressive reduction in synthesis time compared to our

previous best result (GPU v8 with texture memory and Thrust sorting). Examining the timing values

in Table 10.6, the results are initially confusing as adding the three main components results in a

larger value than the actual runtime. This is a result of the Source Patch Matching and Patch

Merging operations executing asynchronously. When there are less than 50 user patches then there

is no benefit to the blocked design. We now observe that our maximum speedup achieved is

times (Figure 8.12) faster than our CPU implementation. The speedup is greater when larger terrains

are used with a high number of patches to match.

Figure 8.11: Runtime results comparing against our asynchronous blocked design against the current best GPU implementation using Thrust sorting. For this test we need to compare the total runtime as the two components are run

concurrently on the CPU and GPU, which reduces the overall time as there is far less idling occurring. The timings for matching and merging are approximately the same but due to running them asynchronously we see a reduced overall

runtime (Table 10.6).

Figure 8.12: Speedup and runtime graph for the asynchronous blocked design against the parallel CPU and Thrust GPU implementations. We see a marginal increase with the asynchronous design for the small terrain with a very large

increase on the large terrain. This is attributed to the total number of features, as the large terrain has a high feature count it is divided up into more blocks which enables the concurrent processing on the CPU and GPU.

https://lh3.googleusercontent.com/notebooklm/AG60hOoEC_O5aQ7DtROqhBD4NQ4tNehWvLWN3kphmauDc5DzRiGAKrBC_mTbcC0n6nYlW1E9Edo5WRazJbnMH3hzQKuIAFqKcmKzouixRD62N24Cqe0V2soAfh49PQ_-go_XrszE4IPk=w1060-h550-v0

8b8aa145-a623-43e7-8620-ed6c14359a8b

8.1.7 Culling Nearby User Patches

One optimisation that is included in all our test cases works by culling detected user patches that

are too closely spaced to an existing chosen patch. This can occur as an artefact from the feature

detection algorithm where branch reduction fails and results in multiple segments that describe the

same feature line. However, this issue is not a bug in the implementation but rather an unfortunate

feature of the algorithm. This produces multiple patches that describe the same feature and leads to

over-synthesising, a wasteful exercise. In order to better test this scenario we designed two new test

terrains (Figure 8.13). Table 8.2 records the total number of patches detected from feature synthesis

and the number of features used after the culling algorithm.

## Small Terrain Large Terrain

## Total Culled To Total Culled To

Ridges 201 88 909 722

Valleys 146 84 753 661 Table 8.2: Number of detected features before and after the culling algorithm. The dimensions for the terrain are,

for the small terrain and           for the large terrain.

We note the speedup obtained by removing the unnecessary patches in Figure 8.15 and Figure

8.16: there is a noticeable decrease in synthesis time after reducing the number of user patches. This

is less apparent on the larger terrain where the proportion of overlapping patches is far less. The

overlap issue is exacerbated along 45° angles or if the drawn lines are too thick, as long parallel

paths form that lead to divergence on the same feature (Figure 8.14).

Figure 8.13: The two test images used to test culling of excess user patches. These were designed to exacerbate the unfortunate feature of the original feature extraction algorithm. a) The small         terrain. b) The large

terrain. The white lines represent ridges with the black lines being valleys as detected by the system.

https://lh3.googleusercontent.com/notebooklm/AG60hOoaz8iATY7e78IIpcK8_9RP6h5_8qOaflVqejE-VrWQIfFTNdkEidUWtzRV2mx3kYHyHuYhN0eU_laZUZtIHXfaWVscZ2q52OK8DLIrCIxWCyR_WhQUjBJ4OVrTKGrtpzgxt3IEPw=w1060-h550-v0

5db57938-cffd-40eb-8aba-4277251e6d1d

https://lh3.googleusercontent.com/notebooklm/AG60hOpCMUUWK6p1TjuJjrO7YvQOjAdrYDTB3i1_GtCOC7-lRrmHAdNZSq7jDrpVL95JZlxn2ZQFM6Xtb45ytU9K7Jh-YKNBFEPPAivZYRg1tNJR-UtOkSG0lalqfwERA6rT1ft7NTzqTQ=w850-h286-v0

b61d2f98-1bed-4523-b983-cd7b76538b27

https://lh3.googleusercontent.com/notebooklm/AG60hOqWpP8HAcK90i2evNAovdSiSeVGwdAt--GyLtwy3XCR8EtPUNCEPTN3z_pQGgM8614vOYibQ8mztz2wqeGY6uQyRF7ImJJYfRmUexbmhuAG_MT6I8v562trUrrtNA59tTYAK6ixDA=w910-h376-v0

d32ebb92-2594-494c-ab14-26a8a3222d8d

Figure 8.14: (a) Example of error with feature detection engine forming multiple parallel lines. (b) This results in heavy overlaying of patches, which wastes performance. These excess patches are culled by the system.

Figure 8.15: Runtime results comparing the implementations when either culling of nearby user patches or not. This is an issue with the original feature extraction algorithm. We address this by examining user patches and removing those

that are in close proximity to one another. This reduces the total number of features requiring synthesis and thus improves performance as shown above. Full runtime values are presented in Table 10.7.

Figure 8.16: Speedup and runtime graph showing the performance gain when culling nearby user patches that are not required. We see a higher gain in the smaller terrain as the proportion of culled patches is higher than the larger terrain.

https://lh3.googleusercontent.com/notebooklm/AG60hOrqfiVlL2FFpusmuBdrE4VLI7YpZo4WpDYRKtKQUFQr3IfyEuwtVaEqVANnaAwBcBtw-1IQS4uLeLCErNKkyE_nFf0EE_X_pKbrG0Nh64WJ7eLvUr3AsvzuCdWIZrCC4dB-AdTNGQ=w620-h287-v0

72989704-1687-41dd-9312-91fa44badd32

https://lh3.googleusercontent.com/notebooklm/AG60hOosZ6UFeLAiNjYrmXFs7CY2DFNH1yp2bfuw4tLeL_ipOL26c3QG5cXJDT427a5NHkgb7-326TQk9yjqb_7vrfE-UjWoBoHGXLHazAWIGGf38YUtTZxubTCl2yw5YiABQVP3Z7zmMQ=w681-h424-v0

a982448a-e0e1-45f3-a184-3af14f961f13

8.1.8 Feature Complexity Change

In order to evaluate the scalability of our system we developed a test that progressively increases

the total features being matched against. We achieved this by sketching the initial test image and

then duplicating the strokes around the image to roughly increase the complexity at a fixed interval.

This allows us to predict how our system will work given more complex user sketches. The results in

Figure 8.17 can be better visualised as a graph (Figure 8.18), where we compare the runtime and

speedup against the number of features. As the features increase linearly so does the synthesis time.

Since the system scales linearly, larger terrains can be produced within an acceptable amount of

Figure 8.17: Runtime results for complexity with increasing total number of patches requiring synthesis. We observe that with an increase in the number of features we see an increase in the time required, with approximately the same proportion of time spent on matching and merging components. Full runtime values are presented in Table 10.8.

Figure 8.18: Plotting the runtime and feature count values on a graph shows a linear relationship for both, which indicates that the system scales well when increasing the number of features.

https://lh3.googleusercontent.com/notebooklm/AG60hOo56X850cVl0BE9M9RbNpBoF8Lcl2m69do8bJhPMxp2rxYjBSSa3-3t9RGpaNhEQFZqqP-CkRnFuhbOVeMtMPWEhr759m5TcB9GR1SSeta5Hmix5Q6qkXpw7mFHpTXbrDqviGQe=w850-h286-v0

d56b72cc-87de-47dd-9183-2f493d0b89f5

https://lh3.googleusercontent.com/notebooklm/AG60hOpm2TdgE5oc6cc3fTcVqBlDiW67hl7ifstJoPbFR697V6ZzmSPxfbDdOLn8BGQwVaEzg8cin1LnpHVAsvgYOsXMklJ_m3hXY3_cnWfFo1av5FeVMt5MYuIC1urKBkp4cKwFVY2Q9g=w910-h376-v0

0da7b5a8-40cd-49b8-9e24-431c77bc60db

8.2 Non-Feature Synthesis The non-feature synthesis GPU-based component was only partially completed due to scope

constraints. Nonetheless, we did implement an initial GPU-enhanced version that borrows

techniques from our feature synthesis implementations to perform the cost calculations on the GPU.

The rest of the processing remains single threaded and CPU-bound. Figure 8.19 presents results,

comparing our CPU and GPU enhanced implementations. We observe a very large speedup in the

cost calculation stage but no noticeable speedup in other aspects (Table 10.9). More research time is

required to explore moving other components to the GPU. For instance, the selection of the next

target; this requires a large proportion of the overall runtime, especially on larger terrains where the

number of holes in the terrain increases dramatically. Given the very long runtime for non-feature

synthesis, especially on the large terrain, any speedup will have significant impact on the total time.

The GPU-version reduces the synthesis time from over 90 minutes to just 54 minutes.

Figure 8.19: Runtimes for the four main contributing components during non-feature synthesis comparing a CPU only implementation to a GPU-enhanced one. We observe that calculating the candidate costs on the GPU significantly

reduces the required time. Examining the time values in Table 10.9 we see a 200 times speedup for cost calculation on the small terrain.

Figure 8.20: Speedup and runtime graph for the non-feature synthesis stage of our system comparing CPU bound and GPU-enhanced implementations.

https://lh3.googleusercontent.com/notebooklm/AG60hOq8X1QWhJa7SgKqnU2SVKF1wtD1N_HamE3kacyqp7sSaou7We21nunARfisTQldA72ClwaeZJ3xrn6fEVkArudu-YvYqs4waYud1fkeaDj5pTq0UakLIwYxqQ2k28zfKDKrz4jtjg=w1060-h550-v0

5da77794-9ce4-4d29-a701-1e5eb7ece615

https://lh3.googleusercontent.com/notebooklm/AG60hOrx-JD747uL7efSkuj0REy6NA882w9dcFUZFJQWFSFxdbVVyd1X8bGtGubbhlscg-4YE5QZSDxCDpHEp4ayJrOA_wLWH3mDSsSO-_y1t84zsSvtBaSyd-f1eOIE5RXSfyNY6roABg=w850-h286-v0

8f3b5f39-70ac-401b-ae04-3d64a7aa8749

8.3 Full Synthesis The tests covered in this section seek to quantify system aspects that globally affect both feature

and non-feature synthesis, such as changing the patch size. We also compare our system with the

previous work by Tasse et al. (2011). For these tests, we use our best performing implementation:

GPU version eight with texture memory, Thrust sorting and asynchronous blocking.

8.3.1 Comparison with previous work

We compare our system to Tasse et al. (2011). We were able to obtain a copy of their software

system in order to do a direct comparison with our system on the same hardware. Unfortunately

only their CPU implementation executed correctly. We include our CPU version to more closely

compare results and draw comparisons from the results section of their paper. For the purposes of

the test we configured our system to only use a single source file. Another issue encountered was

their system was unable to handle our large test image, which we then substituted for a different

smaller image with dimensions           (Figure 8.21).

Figure 8.21: The user images used for this test. a) The original small         terrain. b) The larger           image, which only features valley data.

Figure 8.22: Runtime results comparing the previous work by Tasse et al. (2011) to our system. We were only able to run their CPU version, which is why we include our two CPU implementations and our best GPU implementation. The graph

above shows that the runtime for our system is far less with the three implementations appearing as tiny columns. Table 10.10 provides the actual runtime values, which better shows the time difference between all the versions.

https://lh3.googleusercontent.com/notebooklm/AG60hOorpMthS2qhUcL6YKuO3W-Etjup_fKF42RWE8KfRtTnSLGRFiVLZ3Fb-m38zR8SZ305wHFnH1B52IqAUez0qbCr2V6IwMjQiScIgakUSRaQDO4-jjE0ceOYGPw5jZwvVff357qfIQ=w910-h376-v0

22c9d1c3-07f3-4719-b32c-bf1b4ba0868d

https://lh3.googleusercontent.com/notebooklm/AG60hOoq2Eey_txU902_ymWvy1TAX6tBX46ZzjmGt8IALvD3e_pnfb1GU0H_DeKXS6BcidFISfnrZBwDNw-qLQp48NYV4xPOpzsy86xTdd-MYwP49wQG9UDcSWiQsD26I7Il1vDxSiiAog=w1060-h550-v0

8604645f-a74c-4955-8db0-91a513076686

Figure 8.23: Speedup and runtime graph comparing the previous work to our system. Here we see the large performance increase our system achieves when running under the same test conditions.

From the results in Figure 8.22 and Figure 8.23, we observe that our system, even when using

multiple source files, runs significantly faster than the previous work. This is not entirely a fair test,

however, as we were unable to compare our system to their GPU implementation. Examining their

results section we extrapolate that their system would yield faster results when conducting a full

synthesis, as the entire system was run on a GPU. Our system lacks acceleration in the non-feature

synthesis area, with most of this processing executed on the CPU. However, the advances we have

made for our feature synthesis component are significantly faster. The resulting terrain produced by

their system appears similar to our own output, with both matching the input user sketch. However,

based on the test sketch, the image synthesised with our system appears clearer, with bolder

feature traits and also follows the sketch better.

Figure 8.24: a) Output from Tasse et al. (2011) system. b) Output from our system using the same single source file.

https://lh3.googleusercontent.com/notebooklm/AG60hOoXs1FlYBCgubwa9g3HVqVxOLqvtFb6UW3-MZxw55gEwZk4O3vUQ5GmGpoDvsLjKyUXbmsZBhHuD_LXlcRWIhlmtb52PajUAtqK_UZDnR-RwIMQtvSECtLq6kp-5lRj1DwyfnUS=w850-h286-v0

9aa2964f-9d63-4f22-9dfc-95537e91cf0e

8.3.2 Single versus Multi-Source synthesis

A core objective of our research was to incorporate the use of multiple input source files to

provide a greater candidate pool and improve the overall variety. We compare our synthesis result

when given a single source file and when given a collection of fifteen varying source terrains (Figure

8.25). As expected we see a decrease in performance with the system requiring twice as long for the

small terrain to execute. The feature matching stage is where most of the processing time is spent,

as there are many more candidates to evaluate for each user patch. The results are similar for the

large terrain with noting a decrease in performance. However, since we are making use of the

asynchronous blocked implementation, the cost calculation and patch merging steps are run

concurrently on the large terrain. This reduces the impact of using multiple source files compared to

the small terrain. This occurs on the larger terrain only as the number of features requiring synthesis

is high enough to have the system split it into blocks.

Figure 8.25: Runtime results when running either a single or database of fifteen source files. The figure shows the times for the feature and non-feature synthesis components. We see the majority of the impact being confined to the feature synthesis stage, this is due to there being more candidates needing evaluation. Non-feature synthesis results are very

close in size as there is more of an impact from the number of iterations required to fill the output terrain with the candidate matching only being a small percentage of the runtime. Full runtime values are presented in Table 10.11.

When examining the non-feature synthesis results, we note that the number of source files has a

much smaller impact on the relative synthesis times for the small and large terrains. The slowdown

occurs with the generation of the source candidates and when sorting, due to there being a larger

number of candidates. Target selection is one of the largest components leading to the high cost for

non-feature synthesis. It is unaffected by the number of candidates being evaluated. As such larger

terrains are less influenced by the size of the source database. Figure 8.26 shows the small terrain

after synthesis when using a single and multiple source files, with the multi-source output having

better ridge data. Another use case that our multi-source system would excel in is, when the user is

attempting to generate a mountainous landscape, for instance, but the single source file is based off

the Grand Canyon without sufficient mountainous data (Figure 8.27). The system would produce

poor results due to the lack of diversity, which is resolved by using various terrain types as input

https://lh3.googleusercontent.com/notebooklm/AG60hOpnFuwseGJNH4W6yq_PAydmuh0FcnHGBi7bM_vBZZmb6fisOT94TK0nctRmE_9Dd7k8XPtUd-WBCkRhmfJoIGUmZfMtUCTij-PUJK9bqs-ZI2pPm4Js5tKCRY9baCZjXTTTBD1XIQ=w1060-h550-v0

f6c38099-4f7c-4fc9-8fc7-21e7e8f9be22

https://lh3.googleusercontent.com/notebooklm/AG60hOqGJhDc_SxezeUIXGiqrQphT4V1UXyiZG4s-7MPRY2WfrVDuKOxJInnMv1cyoY_NA7LU-GTwuEqEcQM1yor3UbWwwszEdUHELMPKc6v0P0rlT_eStj0Kw-Gv74VhpOnDnrSOZYrdg=w1060-h550-v0

22ec0958-a4fb-48ca-a318-3d27cf1c6235

Figure 8.26: Output terrain for: a) Single source. b) Multiple sources

Figure 8.27: Example when running a ridge only terrain using a) Single source – Grand Canyon. b) Multiple sources. The single source does not have sufficient ridge data resulting in a poor terrain compared to the clearly defined structure

when using multiple sources.

https://lh3.googleusercontent.com/notebooklm/AG60hOrYDrxgNEpDHXD4VSbnY41b2XOCvwy_E_05jLZ1tK8d3XZ9jzo17LnYFu_n_sAFrrw7QH0mcNDIY9oAD_y3l-_OyEyDsZ8nhhkA3YiQCDV-iQUn47fuEF_qGGEi3uMIwpg-JgXV=w850-h286-v0

ceb882a4-e6a5-4584-ae7f-f7b67e934b41

8.3.3 Patch Size change

One final test examines the effect of varying the size of patches. This alters all aspects of the

system, as a reduction in patch size increases the number of user patches generated to cover the

same sketched area. As discussed earlier, we chose a base patch size of      , which was based

on CUDA grouping 32 threads into warp units. We tested patch sizes in increments of    starting at

, up to        . Our results are presented in Figure 8.28, with the runtime and speedup

visualised in Figure 8.29.

An interesting pattern emerges when examining performance results for the feature and non-

feature components. For feature synthesis we see an improvement in performance when moving

from       to      , but the performance decreases thereafter as the patch size increases. This

is explained by the increase in patch area, which impacts the performance when calculating the cost

for a patch. This is largely mitigated by the decrease in the number of candidates to synthesise.

However, when it comes to patch merging the increased patch area impacts the performance more

severely, taking up more of the synthesis time as the patch size increases. These results are mirrored

for both the small and large terrains, which leads us to conclude that the influence of patch size is

independent of the size of the synthesised terrain.

Figure 8.28: Runtime results when using different patch sizes to synthesise terrains. We observe that for the small terrain the optimal patch size is       with the large terrain performing better with larger patch sizes. Upon further

inspection of the timing values (Table 10.12), we note that for both terrain sizes the feature matching component performs fastest with a patch size of      . Larger patch sizes reduce the non-feature synthesis time as more data is

placed on each iteration, requiring less overall.

https://lh3.googleusercontent.com/notebooklm/AG60hOq2xDkjngw3r0If1rV2mpqmhdutTTo0UWtg3LMe-SQP2DN8fFtESixq5kHCpT6I04g_HAw7VKBohl6ctr13uF-MD2RZ3ahLwIFyUZ0vH4p4GiTIVoGfi4ZffrX1SmKVdG21e7OpHQ=w910-h376-v0

04816c39-8b20-4562-84a6-e04b9e24954c

Figure 8.29: Speedup and runtime graph showing the effect of varying the patch size for synthesis operations. For the small terrain the optimal size is      , with the large terrain performing best with the       patch size.

Turning to the non-feature synthesis results, we see an increase in the performance as the patch

size increases. This performance gain is attributed to the larger area covered by each patch

placement operation. With a larger area being merged into the final terrain, there are significantly

fewer open areas. This reduces the number of operations required to complete the synthesis and

thus the time required. This pattern is exhibited for both terrain sizes. However, the large terrain

requires a very large number of patches to complete synthesis and the time reduction is relatively

significant. Similarly, when the patch size increases the cost calculation step is reduced, as is the

selection of the next location to synthesise, due to reduced number of iterations required overall.

We observe an increase in the merging time due to a larger patch area having to be evaluated when

running the complex merging algorithms.

The above results motivate our choice for the patch size of       especially with the feature

synthesis stage. This size also ensures the highest relative speedup between successive patch sizes.

8.4 Summary Here we evaluated the performance of all the various implementations of our system. We started

with a primitive CPU implementation and progressed to an asynchronous blocked design that

balances work between the CPU and GPU in order to maximise performance. We demonstrated a

peak speedup of     times, when looking at feature matching and merging alone, and a     times

speedup when including non-feature synthesis. We also evaluate our system against the previous

work of Tasse et al. (2011) and observe speedups of    times over their CPU implementation for a

large test terrain. Unfortunately we did not have access to their GPU implementation in order to do

a comparative test but speculated on the results. Our inclusion of a multi-source database is shown

to significantly improve the output terrain, especially when a single source lacks the feature types in

the user sketch. Our system only parallelises a small portion of the algorithm and obtains the high

speed increases; Chapter 0 presents some aspects that could be explored to further increase

performance. We now showcase some example terrains with 3D renderings of the result.

https://lh3.googleusercontent.com/notebooklm/AG60hOoZTfBaOWySQ4Kh4kgvUPinF1_5qK2jds8CmM_O3sL8kDec_6OyYroklNFLnndSvLBh5QgJUjVU_hqFC-1IRnBCfSdYqo02SA6ZwWKfKddo8N1UVSJpSkDMcgmMB3JWuSn_4ni6uA=w1060-h1060-v0

2b5aaa47-f42c-4d16-bdc2-2d2f205a6f1c

https://lh3.googleusercontent.com/notebooklm/AG60hOqz5oNfbSbSgAIN9mWziXrv8NFDW5HxdnGqs9I-YCOvIeBCy6CuHA80ZwC5N94ilWsCy06xwv3CE3rd3JsAPwUIgwKENq7P956czjZmPIdY_LkHJGcMyLvSRReOtuMDG_Ny8jnqxA=w1060-h1060-v0

853e204e-ed48-40d1-a11f-2c6691513243

Figure 8.30: Our small test terrain (       ). b) The output from our synthesis system (Completed in 13 seconds). c) 3D rendering of the terrain.

Figure 8.31: a) The lambda symbol drawn as valleys (       ). b) The output from our synthesis system (Completed in 14 seconds). c) 3D rendering of the terrain.

https://lh3.googleusercontent.com/notebooklm/AG60hOqEpeLY2sn7vRFshi4cBPNbynd5jQFEJqNcFT95MsnAIVLkcjKk3nACE5U651Gy_HgyNf_kDNG_duffya7wONxVP84B8vnMcK-IMkTBi5ya9jqE7R1t6FNGHywFKc8rHLIXJHkC6A=w1060-h1060-v0

4cbeddcb-64c8-4fc2-a144-c0f460195c25

https://lh3.googleusercontent.com/notebooklm/AG60hOpF8WZQU8UcJ759lAyALwnvIzPsDTzMyLvKpe7MBqwETevqQbaaKT-NhzsuYrFnUcg99z9MRozbqiuTx59pKmU7fUPj7ovdFwVJZkWlfKPJ26kH6azyEOx9DF0Jt9963KmATeLeRg=w1060-h1060-v0

8930d059-1967-48ea-a1b0-0ef17c57666b

Figure 8.32: a) A combination of ridges and valleys (         ). b) The output from our synthesis system (Completed in 52 seconds. c) 3D rendering of the terrain.

Figure 8.33: a) A combination of ridges and valleys (         ). b) The output from our synthesis system (Completed in 49 seconds. c) 3D rendering of the terrain.

9 Conclusion

The primary objective of our research is to build a terrain synthesis system that is capable of

rapidly generating realistic terrains from a simple user-sketched image. We base our research on the

work by Tasse et al. (2011) as their work produces highly realistic terrains using patch-based

techniques. We develop several extensions to their research including the use of multiple source

terrains in an effort to increase the candidate pool available during synthesis to allow for more

varied generated terrains. We also develop a highly parallelised GPU solution to dramatically

accelerate the synthesis operation.

We developed a number of different implementations ranging from a simple CPU only one, to an

advanced asynchronous design utilising both the CPU and GPU for maximum performance. The

inclusion of multiple input sources addresses limitations where certain types of user feature would

not be possible to synthesise on a single source due to limited variability within it. This vast increase

in candidates to choose from results in better matching features being synthesised. Other

optimisations were implemented during feature matching, which results in terrains with clearer,

bolder feature traits that more closely follow the user’s original sketch.

From tests, we determine that our system is capable of producing terrains that match the quality

of the previous system (Tasse et al., 2011) due to using components of their existing system and

integrating the new components. We were able to produce more diverse terrains through the use of

multiple input sources. The performance impact when using multiple sources varied between

times to      times the original synthesis time, depending on the size of the terrain being generated.

This was for our test system with fifteen source files and further increasing this would change the

results and requiring more synthesis time. Several optimisations were integrated, which further

improved performance. Our highest speedup obtained for a hybrid CPU/GPU solution overall was

times faster than the first implemented CPU version. This reduced the feature synthesis time for

small terrains from     seconds to just under   seconds and from    minutes to just under

seconds for our large test terrain. Comparing our system to the previous work we see a significant

increase in performance, despite not having ported our entire system to be GPU based.

Revisiting our comparison of synthesis techniques from section 2.4, we saw that texture-based

methods have a slow speed in comparison to fractal and physics-based techniques (Table 9.1).

However, our system shows large speed improvements, reducing synthesis to just seconds while

offering a high degree of user-control and realism. Fractal based methods can achieve interactive

synthesis and allow users to immediately see the effects of parameter changing but do not have the

degree of control our system provides. We describe the main limitation and some possible future

improvements to our system next, but even with these improvements it is unlikely the system could

attain interactive speeds due to the very large amount of data that needs to be processed.

Speed User-Control Realism Main Limitations

Fractal-based Very fast Low – High* Low  Absence of natural erosion

 Non-intuitive control parameters

 Pseudo-random output terrain

Physics-based Thermal: Fast

Hydraulic:

Low Thermal: Medium

Hydraulic:

 Complex to implement

 Requires a base terrain

 Minimal user control

Texture-based Slow Medium High  Limited user control

 Output dependant on number of input terrains (exemplars)

Table 9.1: Comparison of terrain generation methods. Table from section 2.4

9.1 Limitations Due to the large scope of the project, and the time constraints imposed by a master’s degree, we

were unable to complete all the desired tasks. The main limitation is that the system was not ported

to the GPU in its entirety with the final merging process still being CPU-bound. The merging process

is inherently sequential and does not map efficiently to the GPU. We have proposed some solutions

to the problem (see below), which would allow for some aspects of the process to be parallelised,

but did not have sufficient time to successfully implement them. Another limitation is that the

sketching interface only allows for two-dimensional manipulations and has no method for specifying

the height of the output terrain, apart from choosing between ridges and valleys.

9.2 Future-work Our research can be extended by implementing a higher degree of CPU multithreading for the

CPU-bound processes to fully utilise the CPU’s performance. Closer interoperability between the

CPU and GPU would greatly reduce the time either spent idle. Ensuring that maximum performance

of the system is reached. Many aspects of the current CPU code could easily be modified to allow for

a threaded environment. However, this is not suited to GPU calculation, since the size of the data

and the consequent transfer time would far outweigh the computational benefits.

Fully porting the entire system to the GPU would provide greater performance gains, but the

caveat is that some aspects might not translate well to a GPU solution and may actually reduce the

overall performance. Strandmark and Kahl (2010) developed a distributed graph-cut algorithm, but

even their system does not guarantee a speedup. Their work could be built upon to enhance our

system. Even if no speedup is achieved, the system would still be capable of processing larger

datasets that would not entirely fit in memory.

The core of the synthesis engine is broken up into smaller parallelisable tasks, which are carried

out on a single GPU. This system could be extended to support multiple GPU devices, which could

dramatically increase the performance, with the only foreseeable issue being the complexity of logic

required to control the division of labour between multiple devices. The CPU is often idle during the

GPU calculation stages and is thus well suited to handling the control logic.

Improvements to the sketching interface can be made to allow the user to manipulate the

strength (height) of the features to be synthesised. This would enable far greater control over the

synthesis process. Another addition would be to give the user a method of describing the non-

feature areas. This could be achieved by making use of noise profiles of the source terrains. A set of

brushes could be provided that the user could paint on the sketch to indicate the type of noise

profile to be synthesised. The type of brushes provided could be auto-generated based on the

diversity of the input sources provided.

The current system will process all the source files in the specified directory during a synthesis

operation. This will not always be the most optimal set, as some terrains might not contain the

correct type of features required by the user’s sketch. A better system would incorporate a way to

categorise the sources based on their features. This would reduce the amount of files queried during

synthesis, which would improve the performance and allow for larger databases to be used.

A similar process could be applied to non-feature synthesis to allow the user to control the type of

data that is placed in certain areas through an improved sketching interface. Non-feature data could

be classified based on noise profiles for the candidate patches, with similar candidates being

grouped together into categories. A number of brushes could be made available on the interface

corresponding to these various categories. The result of this process is that only candidates from the

specified category would be queried during the synthesis for a particular area, which would reduce

the number of comparisons. This would also give the user more control over the appearance of the

resulting terrain.

## List of References

ABDELGUERFI, M., WYNNE, C., COOPER, E. & ROY, L. 1998. Representation of 3-D elevation in terrain databases using hierarchical triangulated irregular networks: a comparative analysis. International Journal of Geographical Information Science, 12, 853-873.

ANH, N. H., SOURIN, A. & ASWANI, P. Physically Based Hydraulic Erosion Simulation on Graphics Processing Unit.  Proceedings of the 5th international conference on Computer graphics and interactive techniques in Australia and Southeast Asia, 2007 New York, NY, USA. ACM, 257-264.

ATI. 2013. Stream [Online]. Available: http://developer.amd.com/resources/archive/archived-tools/gpu-tools-archive/ati-stream-software-development-kit-sdk-v1-4-beta/.

AVATAR 2009. Avatar.

BANGAY, S., DE BRUYN, D. & GLASS, K. 2010. Minimum spanning trees for valley and ridge characterization in digital elevation maps. Proceedings of the 7th International Conference on Computer Graphics, Virtual Reality, Visualisation and Interaction in Africa. Franschhoek, South Africa: ACM.

BELL, N., YU, Y. & MUCHA, P. J. 2005. Particle-based simulation of granular materials. Proceedings of the 2005 ACM SIGGRAPH/Eurographics symposium on Computer animation. Los Angeles, California: ACM.

BENEŠ, B. & FORSBACH, R. Layered data representation for visual simulation of terrain erosion.  Computer Graphics, Spring Conference on, 2001., 2001. 80-86.

BENEŠ, B., TĚŠÍNSKÝ, V., HORNYŠ, J. & BHATIA, S. K. 2006. Hydraulic erosion. Computer Animation and Virtual Worlds, 17, 99-108.

BROSZ, J., SAMAVATI, F. & SOUSA, M. 2007. Terrain Synthesis By-Example. In: BRAZ, J., RANCHORDAS, A., ARAÚJO, H. & JORGE, J. (eds.) Advances in Computer Graphics and Computer Vision. Springer Berlin Heidelberg.

BRYCE. 2013. Bryce 7 [Online]. Available: http://www.daz3d.com/products/bryce/bryce-what-is-bryce.

C++. 2015. C++ stable_sort [Online]. Available: http://www.cplusplus.com/reference/algorithm/stable_sort/.

CHANG, Y.-C. & SINHA, G. 2007. A visual basic program for ridge axis picking on DEM data using the profile-recognition and polygon-breaking algorithm. Computers & Geosciences, 33, 229-237.

CHANG, Y.-C., SONG, G.-S. & HSU, S.-K. 1998. Automatic extraction of ridge and valley axes using the profile recognition and polygon-breaking algorithm. Computers & Geosciences, 24, 83-93.

CHIANG, M.-Y., TU, S.-C., HUANG, J.-Y., TAI, W.-K., LIU, C.-D. & CHANG, C.-C. Terrain synthesis: An interactive approach.  International workshop on advanced image tech, 2005.

CHIBA, N., MURAOKA, K. & FUJITA, K. 1998. An erosion model based on velocity fields for the visual simulation of mountain scenery. The Journal of Visualization and Computer Animation, 9, 185-194.

COHEN, J. M., HUGHES, J. F. & ZELEZNIK, R. C. 2000. Harold: a world made of drawings. Proceedings of the 1st international symposium on Non-photorealistic animation and rendering. Annecy, France: ACM.

COOK, R. L. & DEROSE, T. 2005. Wavelet Noise. ACM Trans. Graph., 24, 803-811.

CRIMINISI, A., PÉREZ, P. & TOYAMA, K. 2004. Region filling and object removal by exemplar-based image inpainting. Image Processing, IEEE Transactions on, 13, 1200-1212.

CUI, J. 2011. Procedural cave generation. University of Wollongong.

D'AMBROSIO, D., DI GREGORIO, S., GABRIELE, S. & GAUDIO, R. 2001. A Cellular Automata Model for Soil Erosion by Water. Physics and Chemistry of the Earth, Part B: Hydrology, Oceans and Atmosphere, 26, 33 - 39.

DACHSBACHER, C. 2006. Interactive Terrain Rendering: Towards Realism with Procedural Models and Graphics Hardware. Universität Erlangen–Nürnberg.

DE CARPENTIER, G. J. P. 2007. Interactively synthesizing and editing virtual outdoor terrain. Delft University of Technology.

DORSEY, J., EDELMAN, A., JENSEN, H. W., LEGAKIS, J. & PEDERSEN, H. K. Modeling and Rendering of Weathered Stone.  Proceedings of the 26th annual conference on Computer graphics and interactive techniques, 1999 New York, NY, USA. ACM Press/Addison-Wesley Publishing Co., 225-234.

DU, P., WEBER, R., LUSZCZEK, P., TOMOV, S., PETERSON, G. & DONGARRA, J. 2012. From CUDA to OpenCL: Towards a performance-portable solution for multi-platform GPU programming. Parallel Computing, 38, 391-407.

EBERT, D. S., MUSGRAVE, F. K., PEACHEY, D., PERLIN, K. & WORLEY, S. 2003. Texturing and modeling: a procedural approach, Morgan Kaufmann.

EFROS, A. A. & FREEMAN, W. T. 2001. Image quilting for texture synthesis and transfer. Proceedings of the 28th annual conference on Computer graphics and interactive techniques. ACM.

EFROS, A. A. & LEUNG, T. K. Texture synthesis by non-parametric sampling.  Computer Vision, 1999. The Proceedings of the Seventh IEEE International Conference on, 1999 1999. 1033-1038 vol.2.

FOURNIER, A., FUSSELL, D. & CARPENTER, L. 1982. Computer Rendering of Stochastic Models. Commun. ACM, 25, 371-384.

FOWLER, R. J. & LITTLE, J. J. 1979. Automatic Extraction of Irregular Network Digital Terrain Models. SIGGRAPH Comput. Graph., 13, 199-207.

GAIN, J., MARAIS, P. & STRAßER, W. Terrain Sketching.  Proceedings of the 2009 symposium on Interactive 3D graphics and games, 2009 New York, NY, USA. ACM, 31-38.

GEOGEN. 2013. Procedural heightmap generator [Online]. Available: https://code.google.com/p/geogen/.

GEORGE, J. A. 1970. The use of direct methods for the solution of the discrete Poisson equation on non-rectangular regions. Stanford University.

III, T. W. 2015. The Witcher III.

INTEL. 2013. Xeon Phi *Online+. Available  www.intel.com/xeonphi .

KARIMI, K., DICKSON, N. G. & HAMZE, F. 2010. A performance comparison of CUDA and OpenCL. arXiv.

KAUFMAN, A., COHEN, D. & YAGEL, R. 1993. Volume Graphics. Computer, 26, 51-64.

KELLEY, A. D., MALIN, M. C. & NIELSON, G. M. 1988. Terrain Simulation Using a Model of Stream Erosion. SIGGRAPH Comput. Graph., 22, 263-268.

KHRONOS. 2013. OpenCL [Online]. Available: http://www.khronos.org/opencl/.

KRIŠTOF, P., BENEŠ, B., KŘIVÁNEK, J. & ŠT'AVA, O. 2009. Hydraulic Erosion Using Smoothed Particle Hydrodynamics. Computer Graphics Forum, 28, 219-228.

KRUSKAL, J. B., JR. 1956. On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem. Proceedings of the American Mathematical Society, 7, 48-50.

KWATRA, V., SCHÖDL, A., ESSA, I., TURK, G. & BOBICK, A. 2003. Graphcut textures: image and video synthesis using graph cuts. ACM SIGGRAPH 2003 Papers. San Diego, California: ACM.

LAGAE, A., LEFEBVRE, S., COOK, R., DEROSE, T., DRETTAKIS, G., EBERT, D. S., LEWIS, J. P., PERLIN, K. & ZWICKER, M. 2010. A Survey of Procedural Noise Functions. Computer Graphics Forum, 29, 2579-2600.

LEWIS, J. P. 1987. Generalized Stochastic Subdivision. ACM Trans. Graph., 6, 167-190.

LEWIS, J. P. 1989. Algorithms for Solid Noise Synthesis. SIGGRAPH Comput. Graph., 23, 263-270.

LIU, Y., LIN, W.-C. & HAYS, J. 2004. Near-regular texture analysis and manipulation. ACM Trans. Graph., 23, 368-376.

LONGMORE, J.-P., MARAIS, P. & KUTTEL, M. M. 2013. Towards realistic and interactive sand simulation: A GPU-based framework. Powder Technology, 235, 983-1000.

MANDELBROT, B. B. 1975. Stochastic models for the Earth's relief, the shape and the fractal dimension of the coastlines, and the number-area rule for islands. Proceedings of the National Academy of Sciences, 72, 3825-3828.

MANDELBROT, B. B. 1983. The Fractal Geometry of Nature, Times Books.

MANDELBROT, B. B. 1988. The Science of Fractal Images. In: PEITGEN, H.-O. & SAUPE, D. (eds.). New York, NY, USA: Springer-Verlag New York, Inc.

MARÁK, I., BENEŠ, B. & SLAVıK, P. Terrain erosion model based on rewriting of matrices.  Proceedings of The Fifth International Conference in Central Europe on Computer Graphics and Visualization, 1997. 341-351.

MEI, X., DECAUDIN, P. & HU, B.-G. Fast Hydraulic Erosion Simulation and Visualization on GPU.  Computer Graphics and Applications, 2007. PG '07. 15th Pacific Conference on, 2007. 47-56.

MILLER, G. S. P. 1986. The definition and Rendering of Terrain Maps. SIGGRAPH Comput. Graph., 20, 39-48.

MILLIRON, T., JENSEN, R. J., BARZEL, R. & FINKELSTEIN, A. 2002. A framework for geometric warps and deformations. ACM Trans. Graph., 21, 20-51.

MINECRAFT 2015. Minecraft.

MUSGRAVE, F. K. 1993. Methods for Realistic Landscape Imaging. Yale University, New Haven, CT.

MUSGRAVE, F. K., KOLB, C. E. & MACE, R. S. 1989. The Synthesis and Rendering of Eroded Fractal Terrains. SIGGRAPH Comput. Graph., 23, 41-50.

NAGASHIMA, K. 1998. Computer generation of eroded valley and mountain terrains. The Visual Computer, 13, 456-464.

NATALI, M., LIDAL, E. M., PARULEK, J., VIOLA, I. & PATEL, D. Modeling terrains and subsurface geology.  Eurographics 2013-State of the Art Reports, 2012. The Eurographics Association, 155-173.

NEIDHOLD, B., WACKER, M. & DEUSSEN, O. 2005. Interactive physically based Fluid and Erosion Simulation, Bibliothek der Universität Konstanz.

NIE, D., MA, L. & XIAO, S. Similarity based image inpainting method.  Multi-Media Modelling Conference Proceedings, 2006 12th International, 0-0 0 2006. 4 pp.

NVIDIA. 2013a. CUDA C Best Practices Guide [Online]. Available: http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html.

NVIDIA. 2013b. CUDA C Programming Guide [Online]. Available: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.

NVIDIA. 2013c. NVIDIA Fermi Compute Architecture Whitepaper [Online]. Available: http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architectur e_Whitepaper.pdf.

OLSEN, J. 2004. Realtime Procedural Terrain Generation. Department of Mathematics And Computer Science (IMADA).

PAJAROLA, R., ANTONIJUAN, M. & LARIO, R. QuadTIN: Quadtree based Triangulated Irregular Networks.  Proceedings of the conference on Visualization '02, 2002 Washington, DC, USA. IEEE Computer Society, 395-402.

PÉREZ, P., GANGNET, M. & BLAKE, A. 2003. Poisson image editing. ACM SIGGRAPH 2003 Papers. San Diego, California: ACM.

PERLIN, K. 1985. An Image Synthesizer. SIGGRAPH Comput. Graph., 19, 287-296.

PERLIN, K. 2002. Improving Noise. ACM Trans. Graph., 21, 681-682.

PEUCKER, T. K., FOWLER, R. J., LITTLE, J. J. & MARK, D. M. The Triangulated Irregular Network.  Amer. Soc. Photogrammetry Proc. Digital Terrain Models Symposium, 1978. 532.

PRIM, R. C. 1957. Shortest connection networks and some generalizations. Bell system technical journal, 36, 1389-1401.

SAUNDERS, R. L. 2006. Realistic terrain synthesis using genetic algorithms. Texas A&M University.

SAUPE, D. 1989. Point Evaluation of Multi-Variable Random Fractals. In: JÜRGENS, H. & SAUPE, D. (eds.) Visualisierung in Mathematik und Naturwissenschaften. Springer Berlin Heidelberg.

SAUPE, D. 1991. Random Fractals in Image Synthesis. In: CRILLY, A. J., EARNSHOW, R. A. & JONES, H. (eds.) Fractals and Chaos. Springer New York.

SAUPE, D. 2003. Fractals. Encyclopedia of Computer Science. Chichester, UK: John Wiley and Sons Ltd.

SCHNEIDER, J., BOLDTE, T. & WESTERMANN, R. Real-time editing, synthesis, and rendering of infinite landscapes on GPUs.  Vision, modeling and visualization, 2006. 145-152.

SEDGEWICK, R. 2001. Algorithms in C, Part 5: Graph Algorithms, Addison-Wesley, Massachusetts.

SHEPARD, D. 1968. A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the 1968 23rd ACM national conference. ACM.

SHEWCHUK, J. R. 1994. An introduction to the conjugate gradient method without the agonizing pain. Carnegie-Mellon University. Department of Computer Science.

ŠT'AVA, O., BENEŠ, B., BRISBIN, M. & KŘIVÁNEK, J. Interactive Terrain Modeling Using Hydraulic Erosion.  Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, 2008 Aire-la-Ville, Switzerland, Switzerland. Eurographics Association, 201-210.

STRANDMARK, P. & KAHL, F. Parallel and distributed graph cuts by dual decomposition.  Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on, 2010. IEEE, 2085-2092.

TASSE, F. P., EMILIEN, A., CANI, M.-P., HAHMANN, S. & BERNHARDT, A. 2014a. First person sketch-based terrain editing. Proceedings of Graphics Interface 2014. Montreal, Quebec, Canada: Canadian Information Processing Society.

TASSE, F. P., EMILIEN, A., CANI, M.-P., HAHMANN, S. & DODGSON, N. 2014b. Feature-based terrain editing from complex sketches. Computers & Graphics, 45, 101-115.

TASSE, F. P., GAIN, J. & MARAIS, P. 2011. Distributed Texture-based Terrain Synthesis. University of Cape Town.

TERRAGEN. 2013. Terragen 2 [Online]. Available: http://planetside.co.uk/products/terragen2.

THRUST. 2013. Thrust [Online]. Available: https://code.google.com/p/thrust/.

USGS. 2013. US Geological Survey [Online]. Available: http://nationalmap.gov/.

VOSS, R. F. 1985. Random Fractal Forgeries. In: EARNSHAW, R. (ed.) Fundamental Algorithms for Computer Graphics. Springer Berlin Heidelberg.

WATANABE, N. & IGARASHI, T. 2004. A sketching interface for terrain modeling. ACM SIGGRAPH 2004 Posters. Los Angeles, California: ACM.

WEI, L.-Y. 2003. Texture synthesis from multiple sources. ACM SIGGRAPH 2003 Sketches &amp; Applications. San Diego, California: ACM.

WEI, L.-Y., LEFEBVRE, S., KWATRA, V. & TURK, G. State of the Art in Example-based Texture Synthesis.  Eurographics 2009, State of the Art Report, EG-STAR, 2009-04-03 2009. Eurographics Association, 93-117.

WORLDMACHINE. 2013. World Machine 2 [Online]. Available: http://www.world-machine.com/.

ZHOU, H., JIE, S., TURK, G. & REHG, J. M. 2007. Terrain Synthesis from Digital Elevation Models. Visualization and Computer Graphics, IEEE Transactions on, 13, 834-848.

https://lh3.googleusercontent.com/notebooklm/AG60hOr3DlJwQ_ng-zuip2gYvGN_kOcIW8B0D3CNwSM01-6lRHV5kF4neIX896U4U_qwu1Izss3ErtQ_mWSg3UBFCufvumg25DVD6ZRmBT4bWRd8aA16tos3nrrwXHsb68WoARJAgNUxHQ=w758-h413-v0

6bbe6c83-62da-4af9-b721-bcc19ad64c44

https://lh3.googleusercontent.com/notebooklm/AG60hOrbMX2-LfubGNW2fGqlhvdUsHpxwZ-m9-_HUjTj_vYjOlvMxFijhkSQFhQgSfQpRWxC2OoQ_qB87CH8IKxNT96aeS1r4P5VC2cJFgKExgd6HsaZQnsL-N1flttypp2bmLDAao9Zrg=w758-h413-v0

892f1107-ed8b-41e1-aef0-0dcc9570eba6

10 Appendix

10.1 Feature Synthesis – CPU v1 vs. CPU v2

Table 10.1: Runtime results comparing the two main CPU implementations. A speedup column is provided to show the performance gain achieved with version two. These implementations perform very similarly despite the large

architectural changes.

10.2 Feature Synthesis – CPU v2 vs. CPU Parallel

Table 10.2: Runtime results showing the performance improvements when multithreading our CPU v2 implementation. Only the cost computation stage was multithreaded as such the times for the other sections remain relatively the same.

https://lh3.googleusercontent.com/notebooklm/AG60hOrZBFHtRcyelbMBzCLINM1dwHpN90UEYJJ65N9xLILVORYV4yJ5yJwJMYKVgMz13TuC8hcEmNGA9xAjMKEFDhc6zZOi0PHu3cvXcuxlGTe-8mGZwqgm1fxft2KSnOv8z79PcPLaQQ=w883-h392-v0

f32ef544-5590-4bc3-ad0c-e775e7b9e298

https://lh3.googleusercontent.com/notebooklm/AG60hOoMANrLRZmWWouY-UKCoQ9n-pj_D3Geh6Zp8M1qmr8S527KauJ1JlM_0O0ZjOcr7tgacztweB-D17EhpIHlPsBFDahIppiDQIZhePbUwIXMQrOq-O0ghK9cMNiKfnRJ5oPBT75D=w883-h392-v0

c28af36d-ec18-4eaa-98c3-98c4fe914d10

10.3 Feature Synthesis – CPU Parallel vs. GPU implementations

Table 10.3: Runtime results comparing the parallel CPU implementation against the different GPU implementations for the small and large terrains. v1 is a translated form of the parallel CPU implementation. v2 adds some shared memory and more threads. v3 attempts to optimise functions but introduces more branching. v4 unrolls an entire loop utilising more concurrent threads. v5 changes the architecture to allow a new dimension of threads for improved concurrency.

v6 optimises v5 preventing unnecessary recalculation of values. v7 combines elements from v5 and v6. v8 revisits v4 and incorporates the newer changes in v7.

https://lh3.googleusercontent.com/notebooklm/AG60hOoEuN6i_a0UzFMardpgNrfdw4VwM6DR3hx0Br5RghFHTHcdxA4yXpUa4n5y4EoifEyb3CA3lHblJ0BRgag-qiDq4qI6hVqMyKe0aZpLuBkj8vqgfhLvB30ZYv6teon4da-OAdLb=w748-h417-v0

7682869b-8de3-4a74-bedd-2ea07f2b7ce4

https://lh3.googleusercontent.com/notebooklm/AG60hOoimIwz-IRmbTIB-fPinm83L-bww7p3nP_-c-fH6tQ708inQu-mPAYLhCwQqpvNF7dD4jYaVoQItLcVYg6Upgk9S0SoFRCb3up-welesJJnGbMqgGPlP1k9-gJbF0Bba9dUq-kWuw=w748-h417-v0

fadd842a-9986-4797-b48c-8badacff83f6

10.4 Feature Synthesis – Using GPU Texture Memory

Table 10.4: Runtime results comparing the texture memory GPU implementation compared to the parallel CPU and GPU v8 implementations. There is a slight performance gain when using texture memory. This is because we already are

using coalesced memory access for our image data. The first two speedup columns are comparing the methods against the CPU implementation with the last speedup value comparing the improvement texture memory provides compared

to the current best GPU v8 implementation.

https://lh3.googleusercontent.com/notebooklm/AG60hOroXyAdYjlR7b2Q6vL_71eH15Me_UYAd_0fyl8G80C8LE2yG9PcQvpg9bnNX8sU-5QCYzaaUAjq7K-Yhu2t7ux-SJ7enNKAkgbpnEb2gU20peR4uTpfErKfjOLs9lreHQ82ont4=w748-h417-v0

46585e50-52f1-4fe1-b3f5-74ab87760116

https://lh3.googleusercontent.com/notebooklm/AG60hOob_iBOsmcIvILyfNOGOzMDg9zHgvM-FXvLR00zmPaTulq0VOw-nMaIaGQTM5UTDDE5RNh2tOmOT1WQoSncTq0hIBmyLulFpYRobPoMx0NxY_qj6DONNq5KfgEQzzuMEGlAqEAF=w748-h417-v0

80993d8a-4998-46f6-8035-c3ea70cf412e

10.5 Feature Synthesis – CPU vs. GPU Candidate Sorting

Table 10.5: Runtime results comparing sorting of the candidates with the CPU, our own GPU kernel or using the Thrust (2013) library. We observe a large speedup when using the GPU to sort candidates, which is further increased when

using the optimised Thrust library. The first two speedup columns compare the GPU sorting algorithms to CPU sorting with the final speedup value comparing the improvement Thrust provides over our implementation.

https://lh3.googleusercontent.com/notebooklm/AG60hOq0BD2HI9EoarEqXKZ77R8wteBtzoH5jSzZtOsCd_LQMkXTDST41ar4Wu58bv4wJFSx2qXV-u0tnHhZ_yqttycIwsLd1QWkQCyguADd2C8vu8t_Rbm5lmwdt5K0WN1quKCCMbOKkQ=w748-h417-v0

ff818873-2176-4eee-af22-9a88818213df

https://lh3.googleusercontent.com/notebooklm/AG60hOoJHduzt7gn91mGXVHgpoaQkRSHe2ZFSbDC2eIh5BLdj3109m04NrCetXncwr-aHPW8sFuXf-jIXqzDJMrcQXCT97G9XlxXj94TisB963ZPz8FuoI6E_xr_dlBoZa0Wiq6hj66j=w748-h417-v0

5518abc0-ab60-4ff2-930d-e6ca2523c87e

10.6 Feature Synthesis – Asynchronous Blocked Implementation

Table 10.6: Runtime results comparing the parallel CPU and our current best GPU implementation, using Thrust sorting, against our asynchronous block system. This allows us to execute code on both the CPU and GPU concurrently, which

produces a very large improvement over our current best GPU implementation. The first two speedup columns are compared to our parallel CPU implementation with the last indicating the gain when using asynchronous processing

over the Thrust enabled GPU implementation.

https://lh3.googleusercontent.com/notebooklm/AG60hOrewmUFR015l2RemWQSh6rQvWkYCjRg5UzhGzsRz1npXxUMcuy_xWy93D-gsP1lF_UIhDiG544Uy6yqZu8k33wW-mutzEfGorx7O-A0CkSoafEmI2LdtIXDhXTnUNQFU4wvLtA9BQ=w758-h413-v0

61389a93-4cac-4fe0-9fc1-0b64e8b3fbce

https://lh3.googleusercontent.com/notebooklm/AG60hOoBpqiJG3xvOmpaSztSa_Xnu32-mtXUgxoUgzrgjX_KolAoy3oewbm4E6o4jmAziUCJdxVlX0IYlgohR5lYTz3hTSxJN0hiZ0qHFpI7USF4PtLSiPhtswkFAUjNBy9BBK6q6uoEuw=w568-h413-v0

df68bc8e-2b22-44fa-8779-bf1b9cdbc112

10.7 Feature Synthesis – Culling Nearby User Patches

Table 10.7: Runtime results comparing the implementations when either culling of nearby user patches or not. This is an issue with the original feature extraction algorithm. We address this by examining user patches and removing those that are in close proximity to one another. This reduces the total number of features requiring synthesis and thus improves performance as shown above. We see a higher gain in the smaller terrain as the proportion of culled patches is higher

than the larger terrain.

10.8 Feature Synthesis – Feature Complexity Change

Table 10.8: Runtime results for varying complexity in terms of the number of total features synthesised by the system. We observe that with a linear increase in the total number of features there is a linear increase in the time required.

This allows our system to scale for larger more complex terrains.

https://lh3.googleusercontent.com/notebooklm/AG60hOo_dliiAn5KzsiOPNT-4Ep0iz9bjvfvA3w7jMU74uNw_2OIsxHkyj2ydTXYv2kXBwjb3nMFYm1BzsW36yTpaqobJimgVfy6RqNfcRVEXdzcNqN4v6DbExEzQOjiunf3ZTEz6NXzPw=w818-h232-v0

9df2d2ba-ca52-467b-a8f0-fe25c4e738ce

https://lh3.googleusercontent.com/notebooklm/AG60hOooNvXQ9H6-4VD7lqB3FZDCSur8qEML2OI7aVEGg3DhKa6FtPOD4nAJmY6mwmpTN-X-iOWxcrugVIdMjqK_AHr9quMwRoQctGUWnrUBoT2Y3KgcxSR5sL0DrwtLyaTbjTiqZA-K_g=w811-h174-v0

81e841b1-2d80-4564-bd59-c02ce0c2d637

https://lh3.googleusercontent.com/notebooklm/AG60hOoeWtLIoi7NBgoZPeZU-EbCYW2avFX396Q6kZhcsMwfxpsjwg7VsjnXToDTjB_IA4tA7Pw98wh29TpBq4oAbTQHIttKVAImfCY2vYbZnmafHbXp9jwNzoTldAcOdvzPGNooUPR3=w811-h154-v0

780e9521-d8b1-4048-b175-a0a7ade7bad4

10.9 Non-Feature Synthesis

Table 10.9: Runtime results for the non-feature synthesis stage of our system. Times presented are for a CPU only and GPU enhanced implementations. The GPU is utilised for cost calculations to help reduce the overhead of synthesis, the

other components are left CPU bound. There is a massive improvement in the cost calculation stage, which has the largest runtime on the CPU.

10.10 Full Synthesis – Previous Work

Table 10.10: Runtime results when comparing our system to the previous work by Tasse et al. (2011). Timing values for Ridges, Valleys and Non-Feature Synthesis were provided in the previous system as such we omit the breakdown for our system in order to only compare the relevant data. While we could only compare the CPU implementation of Tasse et al. (2011), we observe that our system runs significantly faster under the same test conditions. Our system was run with a

single source file to match the output more closely.

https://lh3.googleusercontent.com/notebooklm/AG60hOoK_xeLQuY1huDUEjfKPmAwLSs2nHrV8h7UAH7qXMpw0TD4GvK0zHJaEeSLL9hGVmwJ1D_B_4usku0gThAGzWrR2FUia4Dp03Ssh_oQFg9Mpqe_zJL_kFBITIezQvFyygtNGpZz=w758-h615-v0

78c9afa4-57bc-426f-b5af-76ef280c1f1a

10.11 Full Synthesis – Single vs. Multiple Sources

Table 10.11: Runtime results for our system when using either a single input source or our database of fifteen. We see the feature synthesis stage has a fairly high cost for using multiple files, although less so when using the larger terrain.

We observe the runtimes for non-feature synthesis being very close between the two implementations due to the large cost of running many iterations to completely fill the output terrain. When looking at the total synthesis time for the

large terrain we see the larger database has very minor impact on the performance.

https://lh3.googleusercontent.com/notebooklm/AG60hOomDmxX-pjHPqmqe8x-qtMnYjpLHMWKEBNiBqyhG-KssfZtKNSjZshakdoj45bBRNp2UIUcS6pWHIMrG_ojY7jZhX-M2Xl7R0jTTMJq472UL5OhRC6WoxhAz44pT8trEsH0rirX=w973-h615-v0

7b004169-1f9a-4ed1-b627-fc32fea3c4c7

https://lh3.googleusercontent.com/notebooklm/AG60hOqi6g4MyyYRohE4Ntret8Rz7F3LFrPR9FT5Ul2452ijmInBDRYjC08q1HVIHmiCGDse_oFe3Jv_jozOAhO95S9i3QjYaPjCXOJ4RwE-fUMTFlYpuXvbpX-7lBEMTZqQLuBoprqq=w973-h615-v0

d713e7ea-6b7c-4bca-9f64-91901fe97b2f

10.12 Full Synthesis – Varying Patch Size

Table 10.12: Runtime results for varying the size of the patch used by our system. We start off with a small       patch size up to a large         patch size. We observe two outcomes when looking at the feature and non-feature synthesis components, which is similar for both terrain sizes. For feature synthesis we see a patch size of       being optimal with the fastest runtime recorded. For non-feature synthesis we observe that the larger the patch size the faster the runtime. This is attributed to a larger area being merged into the output, which reduces the amount of empty areas

thus requiring less iterations to complete.

