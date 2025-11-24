---
sourceFile: "QUERY, ANALYSIS, AND VISUALIZATION OF MULTIDIMENSIONAL DATABASES - Stanford Computer Graphics Laboratory"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:40:12.998Z"
---

# QUERY, ANALYSIS, AND VISUALIZATION OF MULTIDIMENSIONAL DATABASES - Stanford Computer Graphics Laboratory

903b67cd-f0f1-442b-8dfa-cf49ab6876e7

QUERY, ANALYSIS, AND VISUALIZATION OF MULTIDIMENSIONAL DATABASES - Stanford Computer Graphics Laboratory

2b55a05f-0135-4df4-863f-98a1c02b42c3

http://graphics.stanford.edu/papers/cstolte_thesis/thesis.pdf

QUERY, ANALYSIS, AND VISUALIZATION

## OF MULTIDIMENSIONAL DATABASES

## A DISSERTATION

## SUBMITTED TO THE DEPARTMENT OF COMPUTER SCIENCE

## AND THE COMMITTEE ON GRADUATE STUDIES

## OF STANFORD UNIVERSITY

## IN PARTIAL FULFILLMENT OF THE REQUIREMENTS

## FOR THE DEGREE OF

## DOCTOR OF PHILOSOPHY

Christopher R. Stolte

c© Copyright by Christopher R. Stolte 2003

## All Rights Reserved

I certify that I have read this dissertation and that, in my opinion,

it is fully adequate in scope and quality as a dissertation for the

degree of Doctor of Philosophy.

Dr. Pat Hanrahan (Principal Adviser)

I certify that I have read this dissertation and that, in my opinion,

it is fully adequate in scope and quality as a dissertation for the

degree of Doctor of Philosophy.

Dr. Jennifer Widom

I certify that I have read this dissertation and that, in my opinion,

it is fully adequate in scope and quality as a dissertation for the

degree of Doctor of Philosophy.

Dr. Jock Mackinlay (Palo Alto Research Center)

Approved for the University Committee on Graduate Studies:

In recent years, large multidimensional databases, or data warehouses, have become common in a

variety of commercial and scientific applications. It is not unusual for these data warehouses to con-

tain billions of tuples, each categorized by tens or hundreds of dimensions. A major challenge with

these databases is to extract meaning from the important data they contain: to discover structure,

find patterns, and derive causal relationships. A promising technique for the analysis of these mul-

tidimensional databases is visualization. To make visualization effective in this context, we need

to develop tools that tightly integrate visual presentation and database queries, support interactive

refinement of the display, and can visually present a large number of tuples and dimensions.

This dissertation introduces a formal approach to building visualization systems that addresses

these demands. The foundation of the dissertation is the Polaris formalism, a language for pre-

cisely describing a wide range of table-based graphical presentations of relational information. A

key aspect of this formal language is the ability to compile visual specifications automatically into

the precise queries and drawing commands necessary to generate the display. This ability enables

us to design systems that closely integrate analysis and visualization. Using the Polaris formal-

ism, we have built two interactive systems: the Polaris interface and a framework for multiscale

visualization.

The Polaris interface for the exploration of multidimensional databases extends the popular

Pivot Table interface to generate a rich, expressive set of graphic displays. The Polaris interface is

simple and expressive because it is built upon the Polaris formalism. Analysts can incrementally

construct complex queries, receiving visual feedback as they assemble and alter the query. The Po-

laris interface is a generally applicable tool that tightly integrates analysis with visualization. This

dissertation also demonstrates how to use the Polaris formalism and data cubes to specify and imple-

ment domain specific multiscale (pan-and-zoom) visualizations efficiently. The presented approach

to multiscale visualization addresses several limitations in the current approaches by introducing

multiple zoom paths into the data and providing general mechanisms for abstraction.

## Acknowledgments

There are many people whose support, guidance, and friendship have made this thesis possible.

I am especially grateful to my advisor, Pat Hanrahan, who has taught me so much and been a

great mentor and friend during the last six years. Through his guidance, ideas, insight, and exam-

ple, Pat has made me into the researcher that I am today. Learning how to do successful research

in information visualization took me many years and Pat was incredibly patient in that time. Brain-

storming with Pat was always informative, intellectually stimulating, and insightful. Many times it

took me months to understand the full extent of the insight and ideas that Pat would share in a single

discussion.

In addition, I would like to thank the other members of my reading committee, Jock Mackinlay

and Jennifer Widom. Jock’s work on APT provided considerable inspiration for this work, and my

copy of his thesis has experienced much wear and tear from repeated readings and consultation.

Both Jock and Jennifer provided invaluable guidance as I completed this research and prepared this

document. I would also like to thank Terry Winograd and Barbara Tversky who were on my oral

defense committee.

A very important aspect of my success and happiness at Stanford was the students I was lucky

enough to work with. Robert Bosch and Diane Tang were my collaborators in developing the system

presented in this dissertation. This project only succeeded because of the amazing amount of work,

advice, and friendship they were willing to put into it. In addition, they happily acted as mentors,

were patient when I regularly forgot to initialize variables, and made every day more fun.

Maneesh Agrawala recruited me early in my Stanford career to work with him on an innovative

idea he had for rendering driving directions. This lead to a long collaboration with him through

which I learned an amazing amount about research. Maneesh has a unique insight into problems

and research and I have benefited greatly from having worked with him.

I am also very grateful to the rest of the graphics lab, in particular the other members working

on visualization: John Gerth, Tamara Munzner, and Francois Guibretiere, who spent many hours in

discussions and talks, helping me to improve my research and presentations. In addition to his help

with research, John Gerth also spend countless hours ensuring that the computer systems within

the laboratory ran smoothly. Finally, the administrative staff, Ada Glucksman and Heather Genter,

greatly smoothed the path through the bureaucratic aspects of Stanford.

Finally, I would like to thank my family and friends. My parents have provided encouragement

and inspiration my entire life, sometimes believing in my goals and dreams even more than I did.

In addition, I have two wonderful siblings, Kirsten and Tyson, who have inspired me through their

amazing achievements.

## Abstract iv

## Acknowledgments v

1 Introduction 1

1.1 Analysis and exploration of multidimensional databases . . . . . . . . . . . . . . .1

1.2 A formal approach to query and visualization . . . . . . . . . . . . . . . . . . . .3

1.3 Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

1.4 Thesis organization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .5

2 Data, Data Warehousing, and OLAP 6

2.1 Data organization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .6

2.1.1 Relational databases . . . . . . . . . . . . . . . . . . . . . . . . . . . . .7

2.1.2 Star and snowflake schemas . . . . . . . . . . . . . . . . . . . . . . . . .8

2.1.3 Hierarchical structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . .9

2.1.4 Data cubes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .10

2.2 OLAP versus OLTP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .12

2.3 Multidimensional analysis operations . . . . . . . . . . . . . . . . . . . . . . . .14

2.4 Data characterization for visualization . . . . . . . . . . . . . . . . . . . . . . . .14

2.5 Example databases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .16

3 Polaris: A Visual Formalism 18

3.1 Benefits of a formalism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .18

3.2 Table-based displays . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .20

3.3 Visual specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .21

3.4 Related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .22

3.4.1 Semiology of Graphics . . . . . . . . . . . . . . . . . . . . . . . . . . . .22

3.4.2 APT and Sage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .24

3.4.3 DEVise . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .25

3.4.4 A Grammar of Graphics . . . . . . . . . . . . . . . . . . . . . . . . . . .25

4 Algebra 27

4.1 Overview and set interpretations . . . . . . . . . . . . . . . . . . . . . . . . . . .27

4.2 Operands . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .30

4.2.1 Set interpretations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .30

4.2.2 Constant operands . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .30

4.2.3 Filtering and sorting of field operands . . . . . . . . . . . . . . . . . . . .31

4.3 Operators . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .32

4.3.1 Concatenation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .32

4.3.2 Cross . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .33

4.3.3 Nest . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .34

4.3.4 Dot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .36

4.3.5 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .36

4.4 Algebraic properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .37

4.5 Syntax revisited . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .39

4.6 Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .41

4.7 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .41

5 Pane Graphics 43

5.1 Graphic marks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .43

5.1.1 Mark types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .44

5.1.2 Specifying marks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .45

5.1.3 Using lines and polygons . . . . . . . . . . . . . . . . . . . . . . . . . . .47

5.2 Graphic types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .48

5.2.1 Ordinal-ordinal graphics . . . . . . . . . . . . . . . . . . . . . . . . . . .51

5.2.2 Ordinal-quantitative graphics . . . . . . . . . . . . . . . . . . . . . . . . .52

5.2.3 Quantitative-quantitative graphics . . . . . . . . . . . . . . . . . . . . . .53

5.3 Visual variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .53

5.3.1 What is an effective encoding? . . . . . . . . . . . . . . . . . . . . . . . .56

5.3.2 Position . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .57

5.3.3 Color . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .57

5.3.4 Shape (pattern) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .61

5.3.5 Size (Granularity) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .62

5.3.6 Rotation (Orientation) . . . . . . . . . . . . . . . . . . . . . . . . . . . .63

5.4 Graphic notation for visual specifications . . . . . . . . . . . . . . . . . . . . . .63

5.5 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .64

6 Generating Queries 65

6.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .65

6.2 Generating SQL queries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .65

6.2.1 Querying without aggregation . . . . . . . . . . . . . . . . . . . . . . . .68

6.2.2 Derived fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .68

6.2.3 Filtering with lines and areas . . . . . . . . . . . . . . . . . . . . . . . . .70

6.3 Generating MDX queries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .70

6.3.1 Measures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .71

6.3.2 Dimensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .71

6.3.3 Filtering and slicing . . . . . . . . . . . . . . . . . . . . . . . . . . . . .72

6.4 Reducing the number of queries . . . . . . . . . . . . . . . . . . . . . . . . . . .73

6.4.1 Generating a single query for multiple panes . . . . . . . . . . . . . . . .74

6.4.2 Identifying panes that can share a query . . . . . . . . . . . . . . . . . . .75

6.5 Performance and scalability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .75

6.6 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .76

7 Interactive Analysis and Exploration 77

7.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .77

7.2 The interface . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .78

7.3 Related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .80

7.3.1 Table-based displays . . . . . . . . . . . . . . . . . . . . . . . . . . . . .80

7.3.2 Database exploration tools . . . . . . . . . . . . . . . . . . . . . . . . . .81

7.4 Data transformations and visual queries . . . . . . . . . . . . . . . . . . . . . . .82

7.4.1 Deriving additional fields . . . . . . . . . . . . . . . . . . . . . . . . . . .82

7.4.2 Sorting and filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . .87

7.4.3 Brushing and tooltips . . . . . . . . . . . . . . . . . . . . . . . . . . . . .88

7.4.4 Hierarchical analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . .89

7.4.5 Undo and redo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .90

7.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .90

7.5.1 Scenario 1: Commercial database analysis . . . . . . . . . . . . . . . . . .91

7.5.2 Scenario 2: Computer systems analysis . . . . . . . . . . . . . . . . . . .94

7.5.3 Scenario 3: Mobile network usage . . . . . . . . . . . . . . . . . . . . . .98

7.5.4 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .100

7.6 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .100

7.6.1 Visualization and data mining . . . . . . . . . . . . . . . . . . . . . . . .100

7.6.2 Experiences . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .101

7.6.3 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .102

7.7 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .102

8 Multiscale Visualization 104

8.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .104

8.2 Related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .105

8.2.1 Multiscale visualization in cartography . . . . . . . . . . . . . . . . . . .106

8.2.2 Multiscale information visualization . . . . . . . . . . . . . . . . . . . . .106

8.3 Multiscale visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .107

8.3.1 Data abstraction: Data cubes . . . . . . . . . . . . . . . . . . . . . . . . .108

8.3.2 Visual abstraction: Polaris . . . . . . . . . . . . . . . . . . . . . . . . . .108

8.3.3 Zoom graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .109

8.4 Design patterns . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .111

8.4.1 Template specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . .112

8.4.2 Zoom patterns . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .114

8.4.3 Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .114

8.4.4 Pattern 1: Chart Stacks . . . . . . . . . . . . . . . . . . . . . . . . . . . .114

8.4.5 Pattern 2: Thematic Maps . . . . . . . . . . . . . . . . . . . . . . . . . .120

8.4.6 Pattern 3: Quantitative Scatterplots . . . . . . . . . . . . . . . . . . . . .120

8.4.7 Pattern 4: Matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .121

8.5 Discussion and summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .121

8.5.1 Multiple hierarchies . . . . . . . . . . . . . . . . . . . . . . . . . . . . .122

9 Conclusion 124

9.1 Future work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .125

9.1.1 Extending the formalism . . . . . . . . . . . . . . . . . . . . . . . . . . .125

9.1.2 Multiscale visualizations and design patterns . . . . . . . . . . . . . . . .126

9.1.3 Usability studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .127

9.1.4 Data management . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .127

Bibliography 129

## List of Tables

5.1 A summary of how the Polaris encoding system relates to the other major encoding

systems. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .56

## List of Figures

2.1 A star schema for a data warehouse for a hypothetical nationwide coffee chain

(COFFEE). The fact table contains the business measures of interest in the analysis,

in this case the profit and sales within the stores. The fact table is characterized by

three dimensions,Time, Product, andLocation, each with its own dimension table. 7

2.2 A snowflake schema for the same data warehouse as Figure 2.1. In this schema, the

Timehierarchy diverges—the analysts can either summarize daily transactions to a

weekly or monthly level of detail—and, as a result, the dimension is modeled by

multiple relations. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

2.3 A hierarchicalTimedimension. A hierarchical dimension is structured as a tree with

multiple levels. In this case, there are four levels:All, Year, Quarter, andMonth.

Each level corresponds to a different semantic level of detail. The parent-child

relationships in the tree are the basis for aggregation within the dimension. . . . . .9

2.4 A data cube for a hypothetical coffee chain (the schema is illustrated in Figure 2.1)

(COFFEE). Each axis in the data cube corresponds to a level of detail for a dimen-

sion in the relational schema (illustrated in Figure 2.1). In this case, there are three

dimensions:Product, Location, andTime. Each cell summarizes all the measures

in the base fact table for the corresponding values in each dimension. . . . . . . . .10

2.5 A lattice of data cubes for the coffee chain data warehouse (COFFEE). Within the

data warehouse, each dimension has a hierarchical structure. This structure defines

a lattice of cubes. Within the lattice, each cube is defined by the combination of a

level of detail for each dimension. The cubes at the bottom of the lattice contain the

most detailed information; the cubes as the top of the lattice are the most abstract. .11

2.6 Theprojectionof a 3-dimensional data cube. Projection reduces the dimensional-

ity of a data cube by aggregating across dimensions that are not of interest in the

analysis. For example, the first projection summarizes acrossL cation, reducing

the 3-dimensional cube to a 2-dimensional cube. . . . . . . . . . . . . . . . . . . .13

2.7 A sliceof a data cube. A slice of a data cube is constructed by filtering the members

of one or more dimensions of the cube. In this example, a 2-dimensional slice

corresponding to data forQtr 2 has been taken from theTimedimension. . . . . . 14

3.1 A visualization of sales for the coffee chain data set (COFFEE) categorized by the

product type and quarter. The visualization was generated from a Polaris visual

specification, which describes table-based visualizations of multidimensional data.

The table consists of a number of rows and columns. Within each pane of the table,

the tuples are visually encoded as a set of marks to create a graphic. . . . . . . . .20

3.2 A visualization of flights between major airports in the USA (FLIGHTS). This vi-

sualization demonstrates one use oflayers, which is to combine heterogeneous data

sources. There are three layers in the visualization, each displaying data from a dif-

ferent data source: (1) polygons representing state boundaries, (2) lines depicting

flights between airports, and (3) circles indicating the locations of the major airports.

Layers are composited back-to-front to generate the final visualization. Chapter 4

discusses layers in more detail. . . . . . . . . . . . . . . . . . . . . . . . . . . . .21

3.3 The major components of a visual specification. A visual specification formally

describes a table-based visualization of multidimensional data. . . . . . . . . . . .23

4.1 The Polaris formalism uses an algebra to specify the table structure underlying a

visualization. Two algebraic expressions define the rows, columns, and spatial en-

codings on the x- and y-axes of the table. In this visualization, which displaysProfit

information for the coffee chain data set (COFFEE), the y-axis is defined by the ex-

pressionProfit+(Market×ProductType) and the x-axis is defined by the expression

(Quarter/ Month). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

4.2 The set interpretation and semantics of several expressions in the Polaris formal-

ism’s algebra. The ordinal fields partition an axis into additional columns and the

quantitative fields are spatially encoded along the axis of the current column. Note

the subtle difference between the nest and dot operators when there is missing data,

as illustrated here forNovember. . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

4.3 In the Polaris formalism, a layer is a single x-y table. Each data source in a visual-

ization is mapped to a distinct layer. The layers for a data source can be partitioned

into additional layers by the z-axis expression for that data source. All the layers in

a specification are composited together back-to-front to form the final visualization.39

4.4 An abstract data flow diagram depicting the transformation of visual specifications

into database queries and visualizations. Chapter 6 provides a more detailed discus-

sion of the transformation of specifications into visualizations. . . . . . . . . . . .42

5.1 A map of the results of the 2000 Federal Election (ELECTION) displaying results,

encoded as color (with blue indicating a Democratic win and red indicating a Re-

publican win), by county. The tuples are rendered using the “Area” mark type and

the tuples are grouped byCountyIDand sorted within counties by theirPoint-order

value. This graphic is an example of a quantitative-quantitative graphic where both

axis variables are independent. . . . . . . . . . . . . . . . . . . . . . . . . . . . .45

5.2 A time series chart displaying the fluctuations in profit for the major markets of a

hypothetical coffee chain (COFFEE). The tuples are rendered using the “Line” mark

type and the tuples are grouped by theirMarketvalue and sorted within groups by

the independent axis (Month) of the graph. The graphic is an example of an ordinal-

quantitative graphic. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .46

5.3 The families of graphics within our taxonomy with examples of well-known charts

from each family. The taxonomy structures the space of graphics into three families

by the types of fields assigned to their axes and then further structures each family

by the number of independent and dependent variables. Using this taxonomy we

can derive the type of graphic within each pane from the table axes expressions and

the mark type. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .48

5.4 An example of an ordinal-ordinal graphic: a graphical table displaying gene ex-

pression (a dependent variable encoded as color) as a function of experiment and

gene (the independent variables encoded spatially as the x- and y-axes respectively)

(GENE). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .49

5.5 An example of an ordinal-quantitative graphic: a dot plot displaying the number of

arrests (a dependent variable) as a function of the type of crime and the sex of the

offender (the independent variables). . . . . . . . . . . . . . . . . . . . . . . . . .50

5.6 A second example of a ordinal-quantitative graphics: a series of Gantt charts dis-

playing locking activity for a parallel graphics library executing on a multiprocessor

computer (ARGUS). In this graphic, the ordinal variable (CPU) and the quantitative

variable (cycle) are both independent variables. . . . . . . . . . . . . . . . . . . .51

5.7 An example of a quantitative-quantitative graphic: a scatterplot displaying the rela-

tionship between two attributes of different products sold by a coffee chain (COFFEE).52

5.8 A summary of the Polaris encoding system. In addition to encoding data in the

position of a mark on the plane, designers can also encode dimensions and measures

of the data in visual properties of the marks such as color, size, orientation, and shape.55

5.9 The nominal and quantitative color encodings used in our implementation of the

Polaris formalism. In addition, we display example palettes from the major research

projects that we leveraged in developing our encodings. . . . . . . . . . . . . . . .58

5.10 The nominal shape encoding used in our implementation of the Polaris formalism

and Cleveland’s shape palette that inspired our design.. . . . . . . . . . . . . . . .60

6.1 Conceptually, each pane in a visualization corresponds to a query. In this figure, the

## SQL and MDX queries that would generate the correct tuples for the highlighted

pane are shown. The algorithm for generating these queries is outlined in Sec-

tions 6.2 and 6.3. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .66

6.2 When the tuples within a pane are being displayed using line marks we need to refine

our query. The default query generating algorithm applies the filtering criteria to all

tuples and thus would retrieve only a subset of the points along a filtered line, as

shown here. The correct approach is to retrieveall pointson a line whenany point

meets the filtering criteria. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .69

6.3 When multiple panes are at the same level of detail, we can query for all of the

panes in a single query and then partition the results locally into sets corresponding

to individual panes. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .73

6.4 To generate some visualizations, we will have to issue multiple queries even after

consolidating individual pane queries. In this visualization, the top four rows of the

table are at a different level of detail then the bottom four rows. . . . . . . . . . . .74

7.1 The Polaris interface when connected to a flat relational database. Analysts con-

struct table-based displays of data by dragging fields from the database schema

onto shelves throughout the display. A given configuration of fields on shelves is

interpreted as a visual specification in the Polaris formalism. . . . . . . . . . . . .79

7.2 The Polaris interface when connected to a hierarchical data cube. The enhancements

to the interface to expose and support hierarchical dimensions are shown in blue. .80

7.3 The interface for changing the aggregation function applied to a single quantitative

field. To change the aggregate, the user simply right-clicks on the field name and

selects a new aggregate. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .83

7.4 An example of discrete partitioning of a quantitative field. On the left is a traditional

scatterplot ofCOGS(cost-of-goods-sold) versusProfit. On the right, the analyst has

binnedProfit into discrete bins of size 30. . . . . . . . . . . . . . . . . . . . . . .84

7.5 An example of generating ad hoc groups for a categorical field. The user has se-

lected the “Ad hoc Grouping...” option from the context menu forStateand has

formed a group containing California, Connecticut, Florida, and Iowa, and named

“AD HOC GROUP”. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85

7.6 An example of filtering and sorting an ordinal domain. The check boxes are used

to indicate which domain values to exclude (by unchecking the value). The analyst

can also drag a domain value to reorder the domain, as is being done for California

in the figure. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .86

7.7 An example of filtering a quantitative domain. The user simply drags the ends of the

highlighted region to indicate which values to include/exclude from the visualization.87

7.8 The first visualization created in an analysis of sales data for a hypothetical coffee

chain (COFFEE). The analyst is concerned with reducing marketing expenses. In

this display, the analyst is examining the relationships between marketing costs,

profit, and sales, as a function of product and where the product was sold (state).

In the circled region, the analyst can see that for some of the products there is a

negative correlation between both sales and profit and marketing and profit. As the

company sells more of these products they are actually losing money. . . . . . . .91

7.9 The second visualization created in an analysis of sales data for a hypothetical coffee

chain. The analyst has focused on two of the charts from the last visualization

and added color and shape encodings to help them correlate points on the graph

with product categories sold in specific regions. The analyst has chosen to focus

on just two of the sets of poorly performing products: those which are shown as

blue squares and magenta crosses (circled in the figure). The legend shows that

these points correspond to espresso sales in the east region and tea sales in the west

region. Now, the analyst will generated a new visualization to get more detailed

information on these product categories. . . . . . . . . . . . . . . . . . . . . . . .92

7.10 The final visualization created in an analysis of sales data for a hypothetical coffee

chain (COFFEE). Using this visualization, the analyst can identify which specific

products are performing poorly. The visualization is set up to show products with

high marketing costs and low (or negative) profit as brightly colored red bars that

are low (or negative) in their stacks. Two such products are circled in the display:

caffe mocha in the east and green tea in the west. . . . . . . . . . . . . . . . . . .93

7.11 The first visualization in an analysis of scalability issues experienced by a paral-

lel graphics application (ARGUS). Initially, the developers hypothesized that the

diminishing performance was caused by remote memory accesses and this visual-

ization was constructed to test this hypothesis. The first view shows the source code

colored by the number of memory misses. There are not many misses and they

are occurring at expected locations, such as synchronization primitives. The second

view shows misses by memory index; there are few misses and no clear point of

contention. The insight gained from these visualization is that the memory behavior

is not likely to be causing the problem. . . . . . . . . . . . . . . . . . . . . . . . .94

7.12 The second visualization created in an analysis of the scalability issues experienced

by a parallel graphics application. After eliminating remote misses as a possible

cause the developers next hypothesized that lock contention might be an issue.

These visualizations show two projections of the same data. The top visualiza-

tion shows locks events over time as a scatterplot and histogram. Towards the end

of the run, the duration of lock events is unexpectedly long. This indicates the lock

contention warrants further investigation. . . . . . . . . . . . . . . . . . . . . . .95

7.13 The final visualization created in an analysis of an application’s scalability issues.

This visualization depicts lock events and thread scheduling as a series of Gantt

charts, one pair for each processor. The display shows that the long lock requests

correspond to long descheduled periods for most processes except one. One process

is descheduled while holding a lock, thus causing the remaining processes to also

block. This behavior was caused by a bug in the operating system. . . . . . . . . .96

7.14 The first visualization created during the analysis of a 12-week trace of the mobile

network in the Gates building at Stanford University (WAVELAN). The analyst is

trying to understand usage patterns of the mobile network. This initial visualization

shows the size and number of packets over time for the most common applications

run on the network (e.g. FTP, web). From this display, it is clear that the web is the

most commonly used application and file transfer is the least commonly used. . . .97

7.15 The second visualization created during the analysis of a 12-week trace of mobile

network usage. This visualization was constructed to help understand how applica-

tion mix (e.g., FTP, web) varies with research group. There is a single line chart for

each research group, with a line within the chart for every application class. From

this display the analyst concludes that the Graphics group is responsible for the large

incoming and outgoing file transfers, while the Systems group has unusually large

session traffic. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .98

7.16 The final visualization created in an analysis of a 12-week trace of a mobile net-

work. This display is a refinement of Figure 7.15: the analysts has drilled down

from research area to individual project to better understand the sporadic large file

transfers. In this view, it is apparent that the Rendering project is generating the

large file transfers. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .99

8.1 (a) An analysis session can be described by a path (or graph) of Polaris specifica-

tions, each corresponding to a visualization created during the analysis. Examining

the graph, we can see that during the analysis, the user implicitly performs visual

abstraction as they change display types to test hypotheses and investigate areas of

interest. (b) Furthermore, if we consider the mapping of Polaris specifications to

data cubes in the lattice of cubes, it is apparent that an analysis also involves im-

plicit data abstraction. This insight leads to the realization that general multiscale

visualizations can be modeled as a graph of specifications. . . . . . . . . . . . . .110

8.2 The type hierarchy within Tableau. Atemplate specificationis specified by includ-

ing type constraints, rather than specific field names, within the specification. . . .111

8.3 The extended graphical notation for describingtemplate specifications. Whereas

the graphical notation previously only described specific visualization, with these

extensions we can now describe classes of visualizations. . . . . . . . . . . . . . .112

8.4 A map of the USA at the state level of detail and encoding population as the color

of each state polygon. Below the visualization is graphic description of this specific

visualization and thetemplate specificationfor the class of visualizations that en-

code a dependent measure as the color of a polygon that represents a geographical

entity at a categorical level of detail. . . . . . . . . . . . . . . . . . . . . . . . . .112

8.5 Azoom pattern. Whereas a zoom graph described a specific multiscale visualization

as a graph of Polaris specifications, a zoom pattern describes a class of multiscale

visualizations as a graph of template specifications. Depicted is the overall structure

of the pattern and then template specifications for key changes in visual and data

abstraction within the pattern. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .113

8.6 The Zoom Graph for the Chart Stacks Pattern as well as screenshots of a visual-

ization of a trace of an in-building mobile network developed using that pattern

(WAVELAN). The top visualization shows a line chart of average bytes/hour for

each day for each research area. The line charts are layered above a high-low bar

encoding the maximum and minimum bytes/hour. In the next visualization, the user

has zoomed in on the y-axis, breaking apart the charts to create a chart for each

advisor within the research groups. In the final visualization, the user has zoomed

on the x-axis, increasing the granularity of the line chart to hourly values from daily

values. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .115

8.7 A variation on the Chart Stack pattern: A visualization of kernel lock activity (lock

requests are shown in blue; time holding a lock is shown in yellow) collected from

a simulation of the Argus graphics library (ARGUS) [42]. The top visualization

shows a histogram of average time spent requesting or holding a kernel lock. A time

interval corresponds to one million cycles and CPUs are grouped by their primary

task (e.g., processing geometry, rasterization, etc.). In the next visualization, the

user has zoomed in on the y-axis, breaking apart the task charts to create a chart

for each CPU, and has zoomed on the x-axis, changing the time granularity to one

hundred thousand cycles per interval. In the final visualization, the user has zoomed

further on the x-axis, resulting in a change in visual abstraction from strip charts to

Gantt charts displaying individual events. . . . . . . . . . . . . . . . . . . . . . .116

8.8 The zoom pattern for the “Thematic Map” pattern and a series of screenshots of a

multiscale visualization of the population of the USA (CENSUS) developed using

the pattern. The initial view is at the state level of detail, with each state colored

by population density. As the user zooms in, with the x and y dimensions lock-

stepped together, the visualization changes data abstraction, drilling down to the

county level of detail. As the user zooms in further, the visual abstraction changes

as layers are added to display more details: both the county name and population

values are displayed as text. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .117

8.9 Pattern 3 and a series of screenshots of a multiscale visualization of average sales

versus average profit over a two-year period for a coffee shop chain (COFFEE). In

the first visualization, each point represents profit and sales for a particular month

and product, summed over all locations. In the next visualization, the user zooms,

changing the data abstraction: points that were originally aggregated over all loca-

tions are now broken down by market, resulting in four points for every original

point. As the user zooms in further, the visual abstraction changes as layers are

added to display more details: each point is colored according to market and a text

label is added to redundantly encode the market name. . . . . . . . . . . . . . . .118

8.10 The “Matrices” pattern and a series of screenshots of a multiscale visualization of

yeast microarray data (GENE) developed using the pattern. The first visualization

shows the highest level gene clusters on the y-axis, the microarray experiment clus-

ters on the x-axis, and the average gene expression in each cell. In the next visu-

alization, the user zooms on both axes to show more detailed information for both

gene and array clusters. In the final visualization, the user has zoomed to show the

original measurements for each gene in each microarray experiment. . . . . . . . .119

## Introduction

1.1 Analysis and exploration of multidimensional databases

In recent years, large multidimensional databases, or data warehouses, have become common in a

variety of applications. Corporations such as Amazon and Walmart are creating large data ware-

houses of historical data on key aspects of their operations, and international research projects such

as the Human Genome Project [41] and the Sloan Digital Sky Survey [62] are generating massive

databases of scientific data. It is not unusual for these data warehouses to contain billions of tuples,

each categorized by tens or hundreds of dimensions.

A major challenge with these databases is to extract meaning from the data they contain: to

discover structure, find patterns, and derive causal relationships. The exploratory analysis process

is one of hypothesis, experiment, and discovery. The path of exploration is unpredictable, and the

analysts need to be able to rapidly change both what data they are viewing and how they are viewing

that data. Furthermore, these databases are typically extremely large and, as a result, the analysts

need to be able to start with abstract summaries of the data and then quickly focus on detailed

information in areas of interest.

A promising technique for the analysis of these multidimensional databases is visualization.

Visualization leverages the immense power and bandwidth of the human perceptual system and its

pattern recognition capabilities. To make visualization effective as a tool for extracting meaningful

information from data, though, it must be tightly incorporated into the analysis process. To achieve

this goal in the context of database analysis, querying the database and generating a visualization

must be a single process. The ability to change and refine the database query through the visualiza-

tion is essential for integrating visualization and analysis.

CHAPTER 1. INTRODUCTION 2

The integration of visualization with analysis and exploration places significant demands on the

interfaces to these data warehouses. Visual analysis tools need to provide:

Data-dense displays:The databases typically contain a large number of tuples and dimen-

sions. Analysts need to be able to create visualizations that will simultaneously display many

dimensions of large subsets of the data.

Multiple display types: Analysis consists of many different tasks such as discovering cor-

relations between variables, finding patterns in the data, locating outliers, and uncovering

structure. An analysis tool must be able to generate displays suited to each of these tasks.

Exploratory interface: The analysis process is often an unpredictable exploration of the data.

Analysts must be able to rapidly change what data they are viewing (the database query) and

how they are viewing that data (the display or visualization).

Abstraction: The analysis process, when applied to large data sets, must necessarily progress

from abstract overviews to isolated detail. The analysis tool must provide facilities for both

visual abstraction (the changing of the visualization) and data abstraction (aggregation or

statistical summarization of the data).

Query generation: Querying a database and then generating a visualization needs to be a

seamless process. The analyst should be able to focus on what display they want to see

rather than the query necessary to generate that display. As a result, it should be possible to

automatically generate queries from a specification of the display.

Existing research [79][25][52] and commercial systems [49] do not adequately support the gen-

eral analysis process or fully meet the demands listed above. There are three fundamental short-

comings of current systems that limit their utility as analysis tools. First, many of these systems

separate query and visualization, treating visualization as a presentation tool for query results that

are generated separately. Second, current approaches implement each display as a separate mono-

lithic object, requiring that the application designer anticipate all of the user’s visualization needs

and program the range of necessary displays a priori. Finally, the displays offered by many of these

systems typically can only encode two or three dimensions, and thus, can present only a very limited

view of these multidimensional databases.

CHAPTER 1. INTRODUCTION 3

1.2 A formal approach to query and visualization

The shortcomings of existing systems arise out of a lack of mechanisms for formally describing so-

phisticated graphic displays of relational information. Our approach to designing interactive analy-

sis systems is to develop a formalism that combines analysis and visualization and then to implement

the systems on top of that formalism. A cornerstone of this dissertation, and the interactive systems

presented in the dissertation, is a formal language (the Polaris formalism) for precisely describing

a wide range of table-based graphical presentations of relational information. A key aspect of this

formal language is the ability to compile visual specifications automatically into the precise queries

and drawing commands necessary to generate the display. This ability enables us to design systems

that closely integrate analysis and visualization.

We have built on the work of several researchers’ insights into the formal properties of graphic

communication, such as Bertin’sSemiology of Graphicsand Mackinlay’s APT system. However,

the Polaris formalism is innovative in several ways. One key aspect of our approach is that all

specifications(formal descriptions) of visualizations can be compiled directly into relational or

multidimensional queries. Existing formalisms do not consider the generation of queries to be

related to the presentation of information. Our system, however, is for analysis and visualization,

and thus, the query and presentation are integral. Another innovation is the use of an algebra to

describe sophisticated table-based displays rather than single pane graphics. Tables are particularly

effective for displaying multidimensional data, as multiple dimensions of the data can be explicitly

encoded in the structure of the table, thus generating data dense displays. Finally, our formalism

is the basis for several interactive tools for analyzing and exploring large data warehouses and this

usage has affected the development of the formalism. The resulting interactive systems, as we will

demonstrate in this dissertation, address every demand outlined in Section 1.1.

The Polaris formalism, although developed for describing table-based visualizations of rela-

tional information, provides a general foundation for building a more complete formalism for

graphic communication of relational data. Several designs lie outside the current expressiveness

of our formalism, such as Worlds-Within-Worlds [29], or Treemaps [60]. The formalism, however,

offers many avenues for extension, and could be developed to describe a considerable percentage

of the known display techniques. As Card and Mackinlay [15] have noted, the field of informa-

tion visualization has grown to a series of point designs and we now need to develop abstractions

that generalize the space [15]. There are many benefits to the visualization community of having a

formalism for describing a wide class of visual displays. A general formalism would help identify

CHAPTER 1. INTRODUCTION 4

areas of the design space for future research, clarify the similarities and differences between known

techniques, and provide insight into what makes particular visual metaphors effective.

1.3 Contributions

The goal of this dissertation is to apply visualization to exploring and analyzing the important data

residing in data warehouses. To accomplish this goal, we need to develop tools that tightly integrate

visual presentation and database queries, support interactive refinement of the display and query,

and can visually present a large number of tuples and dimensions.

The contributions of this dissertation fall into two categories: a formalism for describing graphic

presentations of multidimensional data, and two interactive visualization systems developed using

that formalism. Specifically, the contributions are:

The Polaris formal language: The Polaris formalism uses succinctvisual specificationsto

describe table-based visualizations of multidimensional data. Each table consists of layers

and panes, and each pane may be a different graphic. The formalism is capable of describing

a very wide range of 2D graphic displays (and combinations of displays). Any specification

in the formal language can be compiled into the database queries and drawing commands re-

quired to generate the display. The three key aspects of the formalism are (1) a table algebra

that captures the structure of tables and spatial encodings, (2) a graphic taxonomy that results

in an intuitive specification of graphic types, and (3) a system for effective visual encoding.

An important contribution of this formalism is that it has been used to engineer several visu-

alization systems; it is the basis of all the interactive systems presented in this dissertation.

The Polaris interface: The Polaris interface for the exploration of multidimensional databases

extends the Pivot Table interface to directly generate a rich, expressive set of graphic displays.

The Polaris interface is simple and expressive because it is built upon the Polaris formalism:

we interpret the state of the interface as a visual specification. All intermediate specifications

are valid and can be interpreted to create visualizations. Thus, analysts can incrementally

construct complex queries, receiving visual feedback as they assemble and alter the query.

The Polaris interface is a generally applicable tool that tightly integrates analysis (database

queries) with visualization.

Multiscale Visualization: The Polaris interface is a general tool for exploring any multidi-

mensional dataset. When an analysis tool is to be used within a specific domain, however,

CHAPTER 1. INTRODUCTION 5

it makes sense to narrow the options presented to the user. We demonstrate how to use the

Polaris formalism and data cubes to specify and implement domain specificmultiscale vi-

sualizationsefficiently. Multiscale visualizations are an effective technique for facilitating

analysis because they present the data at different levels of abstraction as the user pans and

zooms. Our approach to multiscale visualization addresses several limitations in the current

approaches by introducing multiple zoom paths into the data and providing general mecha-

nisms for data and visual abstraction.

1.4 Thesis organization

The rest of this dissertation is organized as follows:

Chapter 2 provides a review of relational databases and data cubes, explaining how these tech-

nologies are used to build data warehouses. The chapter concludes with a brief discussion of the

additional data characterization needed for visualization and the example databases that will be used

throughout the dissertation.

Chapter 3 presents an overview of the Polaris formalism, describing visual specifications and

the types of displays that they can specify. In Chapters 4–6, we discuss the individual aspects of the

formalism in detail and then outline our implementation.

We present the Polaris interface for exploring multidimensional databases in Chapter 7. We also

discuss several analysis scenarios performed using the interface.

Chapter 8 presents our approach to multiscale visualization. It explains how the Polaris formal-

ism and data cubes can be used together to create a powerful framework for developing multiscale

visualizations. We also discuss the idea of design patterns as a technique for capturing the structure

of effective visualizations to be reused in new designs.

Finally, Chapter 9 summarizes the contributions of this work.

Data, Data Warehousing, and OLAP

The research presented in this dissertation focuses on visualization techniques for the exploration

and analysis of multidimensional analytic data stored in a database, ordata warehouse. These data

warehouses are typically structured as either relational databases or multidimensional data cubes.

In this chapter, we will review the aspects of relational databases and data cubes that are relevant

to the design and implementation of the Polaris formalism and interface. We will also explain how

these technologies are used to build data warehouses, introduce the additional data characterization

needed to support the visualization process, and introduce the data sets that will be used in examples

throughout the remainder of this dissertation. The material presented in this chapter is largely a

synthesis of [5], [31], [68], and [73].

2.1 Data organization

Databases have typically been used for operational purposes (OLTP), such as order entry, account-

ing and inventory control. More recently, corporations and scientific projects have been building

databases, calledata warehousesor OLAP databases, explicitly for the purposes of exploration

and analysis. The term ”data warehouse” was first used by W.H. Imnon [43]. He defined a data

warehouse as ”a subject-oriented, integrated, time-variant, nonvolatile collection of data in support

of management decisions.” The key aspect of the data warehouse is that it is a repository for ana-

lytic data rather than transactional or operational data. The data contained in the data warehouse

usually represents historical data, e.g., transactions over time, about some key interest of the busi-

ness or project. This data is collected from many different sources such as operational databases,

simulations, data collection tools, and other external sources.

https://lh3.googleusercontent.com/notebooklm/AG60hOqewaRFsv7Bo5jDOKuqChxDyqUlN4DWj7wLEiNbB01u3s5JXMmr6n7V5mUIfJCyvPVfIdTg6aG4JUyM1_Vd8wh6XJ8d_2-wCxQAUvBY1qtBauN7zgvaeb0pcJ6ZMcGy5mJsmVUw7g=w1804-h723-v0

d0c30d96-027d-4ec1-887d-a2b7a13864d7

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 7

Figure 2.1: A star schema for a data warehouse for a hypothetical nationwide coffee chain (COF-FEE). The fact table contains the business measures of interest in the analysis, in this case the profit and sales within the stores. The fact table is characterized by three dimensions,Time, Product, and Location, each with its own dimension table.

Data warehouses are built using both relational databases and specialized multidimensional

structures called data cubes. In this section, we explain the organization of the data within these

databases: the database schemas, the use of semantic hierarchies, and the structure of data cubes. In

the following section, we will discuss how the organization of OLAP databases differ from OLTP

databases.

2.1.1 Relational databases

Relational databases organize data into tables where each row corresponds to a basic entity or fact

and each column represents a property of that entity [23][73]. For example, a table may represent

transactions in a bank, where each row corresponds to a single transaction, and each transaction has

multiple attributes, such as the transaction amount, the account balance, the bank branch, and the

customer. We refer to a relational table as arel tion, a row as atuple, and a column as anattribute

or field. The attributes within a relation can be partitioned into two types:dimensionsandmeasures.

Dimensions and measures are similar to independent and dependent variables in traditional analysis.

For example, the bank branch and the customer would be dimensions, while the account balance

would be a measure.

A single relational database will often describe many heterogeneous but interrelated entities.

For example, a database designed for a coffee chain might maintain information about employees,

https://lh3.googleusercontent.com/notebooklm/AG60hOoO8kPqHxQESR_yzL1HSxY7FuzG4kuO4pX3kWXy1dj1gWEK4-mLjviqtEBWEA_CBk-SB--LUj8hJj7EJEcePVOEfe6R86GKD2YX7UElfIy5CYY-0_IXNazvmIfeNxRgsbSCM_Ch5w=w1804-h792-v0

0cfb00da-ea73-44b7-82a8-8b51853e8f60

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 8

Figure 2.2: A snowflake schema for the same data warehouse as Figure 2.1. In this schema, theTi e hierarchy diverges—the analysts can either summarize daily transactions to a weekly or monthly level of detail—and, as a result, the dimension is modeled by multiple relations.

products, and sales. Thedatabase schemadefines the relations in a database, the relationships

between those relations, and how the relations model the entities of interest.

2.1.2 Star and snowflake schemas

The two most common schemata found in data warehouses are the star schema and the snowflake

schema. Each schema has a core relation, called thefact table, containing data items of interest

(measures) in the analysis for which the warehouse is built. These data items might be transaction

amounts such as the amount invested in a mutual fund or the profit on a sales transaction. The fact

table is surrounded bydimension tablescontaining detailed information used to summarize the fact

table in different ways. Figure 2.1 illustrates a star schema and Figure 2.2 illustrates a snowflake

schema. These schemata provide a conceptual multidimensional view of the data warehouse: the

database is a core set of measures characterized by a number of dimensions rather than a set of

interrelated relations. This organization correlates directly with the typical analysis query that sum-

marizes a few quantitative attributes (or measures) such as profit or sales by several characterizing

attributes (or dimensions) such as product, location, or date over a large number of tuples. The

primary differences between the star and snowflake schemas arise in how they model hierarchical

structures on the dimensions, which we discuss more in the next section.

https://lh3.googleusercontent.com/notebooklm/AG60hOotWJ-2a5NXZH56LWEDQzovXMxsyLzG3cS-AvSnwWMwd--ccjZpFjU28b7_fucSkeJqE_T_UUg9dm8qLRKZsuuExHASzTlc4Yyxe44ZbYJjrEp3U5UNTqwV0kHFHMpWTGfnSTty=w1805-h361-v0

1819ef49-2208-4316-afc6-fdd1ff92a9bf

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 9

Figure 2.3: A hierarchicalTimedimension. A hierarchical dimension is structured as a tree with multiple levels. In this case, there are four levels:All, Year, Quarter, andMonth. Each level corresponds to a different semantic level of detail. The parent-child relationships in the tree are the basis for aggregation within the dimension.

2.1.3 Hierarchical structure

Most dimensions in a data warehouse have a hierarchical structure that analysts can leverage in

their analysis. This hierarchical structure may be derived from the semantic levels of detail within

the dimension or generated from classification algorithms. Using these hierarchies, the analyst can

explore and analyze the data at multiple levels of detail calculated from the fact table. For example,

rather than having a single dimension “state,” we may have a hierarchical dimension “location” that

has three levels, one for each country, state, and county, and the analyst can aggregate the measures

of interest to any of these levels.

The aggregation levels are determined from the hierarchical dimension, which is structured as a

tree with multiple levels. The highest level is the most aggregated and the lowest level is the least

aggregated. Each level corresponds to a different semantic level of detail for that dimension. Within

each level of the tree, there are many nodes, with each node corresponding to a value within the

domain of that level of detail of that dimension. The tree forms a set of parent-child relationships

between the domain values at each level of detail. These relationships are the basis for aggrega-

tion, drill down, and roll up operations within the dimension hierarchy. Figure 2.3 illustrates the

dimension hierarchy for aTimedimension.

Simple hierarchies, like the one shown in Figure 2.3, are commonly modeled using a star schema

(as in Figure 2.1). The entire dimensional hierarchy is represented by a single dimension table. In

this type of hierarchy, there is only one path of aggregation. However, there are more complex

dimension hierarchies in which the aggregation path can branch. For example, aTimedimension

might aggregate fromDay to bothWeekandMonth. These complex hierarchies are typically repre-

sented using the snowflake schema (as in Figure 2.2), which uses multiple relations to represent the

diverging hierarchies.

https://lh3.googleusercontent.com/notebooklm/AG60hOpUN2p2zDzToVJp5AU05iIbNee89uAvt8umhH-WWbny17RwkG5tnHCvlvzJO5UylxIENTZcCXpQ4D_s32fMceU5H1xWayACj1qVKwaALT6KdRXIqBhTAPQe34SdBdMYeBnOtnchUQ=w1800-h837-v0

bc2fab81-a554-45af-901d-d4c058dac505

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 10

Figure 2.4: A data cube for a hypothetical coffee chain (the schema is illustrated in Figure 2.1) (COFFEE). Each axis in the data cube corresponds to a level of detail for a dimension in the rela-tional schema (illustrated in Figure 2.1). In this case, there are three dimensions:Pr duct, Location, andTime. Each cell summarizes all the measures in the base fact table for the corresponding values in each dimension.

2.1.4 Data cubes

A data warehouse can be constructed as a relational database using either a star or snowflake schema

and will provide a conceptual model of a multidimensional data set. However, the typical analysis

operations such as summaries and aggregations are not well supported by the relational model. The

queries are difficult to write in languages such as SQL and the query performance is not ideal. As a

result, typically the fact tables and dimension tables are not used directly for analysis but rather as

a basis from which to construct a multidimensional database called adata cube.

Each axis in the data cube corresponds to a dimension in the relational schema and consists of

every possible value for that dimension. For example, an axis corresponding to states would have

fifty values, one for each state. Each cell in the data cube corresponds to a unique combination

of values for the dimensions. For example, if we had two dimensions,Stateand Product, then

there would be a cell for every unique combination of the two (e.g., one cell each for (California,

Tea), (California, Coffee), (Florida, Tea), (Florida, Coffee), etc.). Each cell contains one value per

measure of the data cube; e.g., if we wanted to know about product production and consumption,

then each cell would contain two values, one for the number of products of each type consumed

in that state, and one for the number of products of each type produced in that state. Figure 2.4

https://lh3.googleusercontent.com/notebooklm/AG60hOqNvgNIiMs9O0N0ez6Q1L_TXJSKD-hA85e_4UaRTQrneh2s72P_TBIh3P8DycO2MNC7HfihiJOyeWH0u672wAdhrCsJqFIW8iEdmYWXxp3XqH9aWUIR_XDO0hvFof1jhViPDoS3rQ=w1800-h1426-v0

908543bc-0ef8-4f46-bff2-dc979ef43d42

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 11

Figure 2.5: A lattice of data cubes for the coffee chain data warehouse (COFFEE). Within the data warehouse, each dimension has a hierarchical structure. This structure defines a lattice of cubes. Within the lattice, each cube is defined by the combination of a level of detail for each dimension. The cubes at the bottom of the lattice contain the most detailed information; the cubes as the top of the lattice are the most abstract.

illustrates a data cube for a hypothetical nationwide coffee chain data warehouse.

As we discussed in Section 2.1.3, the dimensions within the data warehouse are often augmented

with a hierarchical structure. Using these hierarchies, the analyst can explore and analyze the data

cube at multiple meaningful levels of aggregation. Each cell in the data cube then corresponds to

the measures of the base fact table aggregated to the proper level of detail. If each dimension has

a hierarchical structure, then the data warehouse is not a single data cube but rather a lattice of

data cubes, where each cube is defined by the combination of a level of detail for each dimension

(illustrated in Figure 2.5).

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 12

2.2 OLAP versus OLTP

In the previous section, we discussed how both relational databases and data cubes could be orga-

nized and used for analytical purposes (OLAP). Traditionally, however, relational databases have

been used for day-to-day operational purposes. These OLTP databases address different issues than

OLAP databases or data warehouses and, as a result, have schemas and usage patterns that are quite

different. It is necessary to understand the differences between these two types of databases in order

to understand the issues affecting the design of OLAP visualization tools.

OLTP databases are optimized for performance when processing short transactions to either

query or modify data, possibly interfacing with more then one system and supporting many simul-

taneous connections. Furthermore, query performance is typically secondary to issues like avoiding

data redundancy and supporting updates. Typical OLTP queries retrieve a few dozen tuples from

only a few relations and then update some of the tuples. For example, a typical query might retrieve

a single customer’s record based on his or her account number, or add a single transaction to a sales

relation when a sale occurs. Database schema definitions for operational databases focus on maxi-

mizing concurrency and optimizing insert, update, and delete performance. As a result, the schema

is often normalized, resulting in a database with many relations, each describing a distinct entity

In contrast, rather than being used to maintain updateable transaction data, analysts need to be

able to interactively query and explore OLAP databases. The queries for OLAP are very different

in that they typically retrieve thousands of rows of information and modify none of them. The

queries are large, complex, ad hoc, and data-intensive. Because an operational schema separates

the underlying data into many relations, executing these analytical queries on a database based on

an operational schema would require many expensive join computations. Since analysis databases

are typically read-only, and because query performance is the primary concern, OLAP databases

sacrifice redundancy and update performance to accelerate queries, typically by denormalizing the

database into a very small number of large relations using a star or snowflake schema. External

tools can typically view an OLAP database as either a data cube or a single large relation.

The formalism and interactive systems presented in this thesis are tools for the exploration and

analysis of OLAP data. Although this data is ideally stored in a data cube for interactive per-

formance, this research applies to both relational databases and multidimensional databases that

store OLAP data. This focus on OLAP results in limitations on Polaris’s applicability to relational

https://lh3.googleusercontent.com/notebooklm/AG60hOqXuNXgkCtbfL2VLCR5AXyCq7Gm0AhD8SoY-XmDAFtiQgPn-SYjwAR9TuCN9-rWZteTSsMNzGOsb4TiB2XbuLhA4gvL0r8Oxm8hqqldbbjQMIhYatbACjfFq4ogvD8KJ6f99xcrQg=w1800-h1246-v0

2f7cd81b-ba35-46cb-b836-932756269f7c

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 13

Figure 2.6: Theprojectionof a 3-dimensional data cube. Projection reduces the dimensionality of a data cube by aggregating across dimensions that are not of interest in the analysis. For example, the first projection summarizes acrossLocation, reducing the 3-dimensional cube to a 2-dimensional cube.

databases: unlike some other interactive query tools [25][79], Polaris cannot be used to specify rela-

tional joins interactively because we assume that OLAP data will either be stored in a denormalized

database or views will be constructed that present the relevant relations as a denormalized database.

It may be possible to develop techniques for specifying join conditions within the Polaris formal-

ism (and thus the interface), but we consider this topic as interesting future work that is outside the

scope of this thesis. Throughout the remainder of this dissertation, we will use the terms relational

database, multidimensional database, and data warehouse as synonyms, all referring to OLAP data

that can be viewed conceptually as a multidimensional data cube.

https://lh3.googleusercontent.com/notebooklm/AG60hOrP-Lbk9MJv7FbtTlskk8gDjcRyLhxZCdVpDhNVbH2zSPcOh8oaa-qOyJYhXBVCiLhP_VmbFNA-C_GAlS-N5UMr7xYFA9FAWL8n7D_fSU9Yzcc8HSnhbh5qp5kmDpyRiX2Y63lg=w1800-h763-v0

8279d520-5f99-4e12-8633-e0944f379a62

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 14

Figure 2.7: Asliceof a data cube. A slice of a data cube is constructed by filtering the members of one or more dimensions of the cube. In this example, a 2-dimensional slice corresponding to data for Qtr 2 has been taken from theTimedimension.

2.3 Multidimensional analysis operations

A data warehouse is typically quite large, consisting of many dimensions each with hierarchical

structure and often many members. To navigate the resulting lattice of data cubes and perform

dimensional reduction to extract data for analysis, there are a number of standard multidimensional

analysis operations that need to be incorporated into our tools and formalism. This section describes

these standard operations.

Drill down refers to the process of navigating through the lattice of data cubes in the direction of

more detail. It is the technique used to break one piece of information into smaller and more detailed

parts.Roll upis the inverse of drill down, aggregating detailed data into coarser elements.Projection

(illustrated in Figure 2.6) reduces the dimensionality of ann-dimensional data cube to(n − 1) by

aggregating across a dimension. Where projection reduces dimensionality via aggregation,slicing

(illustrated in Figure 2.7) reduces dimensionality by filtering a dimension to a single value; in other

words, one dimension is held constant to generate a slice across that dimension.

2.4 Data characterization for visualization

Having described how the data (OLAP) Polaris uses is organized, we now discuss the additional data

characterization needed to support the visualization process. For the purposes of visualization, we

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 15

need to know more about an attribute than is usually captured by a database system [56]. Databases

typically provide limited information about a field, such as its name, whether a field is a dimension

or measure, and its type (e.g., time, integer, float, character).

In our formalism, in order to determine how to encode a field in a graphic using visual properties

such as color, size, or position (see Chapter 5), we need to know whether a database field is nominal,

ordinal, or quantitative. This characterization is based on a simplification of Stevens’ scales of

measurement [65]. We further simplify this characterization depending on if the context emphasizes

the difference between discrete data and continuous data or if the context emphasizes whether the

field has an ordering. For example, when encoding a field spatially, the emphasis is on whether a

field has discrete values. Furthermore, when a field is assigned to an axis, it must have an ordering.

Thus, in this context, nominal fields that do not normally have an ordering are assigned one and

then treated as an ordinal field. The resulting characterization is called categorical. In contrast, when

assigning visual properties such as color to a field, then the important distinguishing characterization

is order; in this context, we treat ordinal and quantitative fields as a single characterization and

consider nominal fields separately.

In addition, attributes have associated units and semantic domains; for example, attributes may

encode time, geographic units such as latitude, or physical measurements. If this information is

available, it can also be used to generate more effective visual encodings and aid in determining the

geometry (e.g., aspect ratio) of a graphic. For example, knowing that the x and y axis of a graphic

correspond to latitude and longitude, rather than profit and sales, will affect the determination of the

appropriate geometry.

Databases also only store the current domain of a field—the values that currently exist within

the database—without any ordering. However, for analysis it is important to understand the actual

domain of a field: the possible values and their inherent (if applicable) ordering. To encode an

attribute as an axis of a graphic, we need to know all possible values and their ordering so that we

can indicate when data is missing and present data within its semantic context (rather than using

some arbitrary ordering, e.g., alphabetic).

In Polaris, this additional data characterization is captured in an XML document that is associ-

ated with the database.

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 16

2.5 Example databases

Throughout this thesis, we will be illustrating concepts by visualizing six databases. To understand

the concepts presented, it is important to understand both the data being visualized and the analysis

task being performed. Each data set is briefly described below.

### 1. Performance Data for the Argus Parallel Graphics Library (ARGUS): At Stanford, re-

searchers developing Argus [42], a parallel graphics library, found that its performance had

linear speedup when using up to 31 processors, after which its performance diminished

rapidly. The graphics library was simulated using the SimOS complete machine simulator

to collect data on cache misses, thread scheduling, and kernel lock activity. A relational

database was constructed to store the complete data set and a separate data cube was gen-

erated for a subset of the data (thread scheduling and lock activity per cycle). The database

contains approximately 2 million tuples.

### 2. Usage Data for the Wireless Network in the Gates Building (WAVELAN):To understand

the usage patterns of a mobile network, researchers collected a 12-week trace of every packet

that entered or exited the mobile network in the Gates building at Stanford University. For

each packet, the user, time sent, size, protocol, etc. were recorded and stored in a data ware-

house. The database contains approximately 80 million tuples.

### 3. Business Metrics for a Hypothetical Coffee Chain (COFFEE):This data warehouse is

derived from example data shipped by Microsoft and Visual Insights with their database and

analysis tools. The data describes transaction activity, such as profit and sales, for stores in a

hypothetical nationwide coffee chain. A relational database and a data cube were generated

from this data set, both containing 6240 tuples or facts.

### 4. Microarray experiment data: (GENE) This data warehouse is derived from publicly avail-

able DNA microarray data that is distributed with Michael Eisen’s Cluster and TreeView

software [27]. The data set contains yeast gene expression data described in Eisen et al. [28].

We used the Cluster software to generate hierarchical clusters on both the genes and experi-

ments and then generated a data cube from the data set. The data cube contains approximately

10,000 tuples.

### 5. Flights Between Major Airports in the USA: (FLIGHTS) This data set was provided cour-

tesy of William F. Eddy and Shingo Oue. The data set describes flights between the major

CHAPTER 2. DATA, DATA WAREHOUSING, AND OLAP 17

airports in the continental USA for a one hour period.

### 6. Census (CENSUS) and Election (ELECTION) Data for the USA:This data set was de-

rived from publicly available census and election data. The complete data set is at the county

level of detail and we generated both a relational database and data cube for the data. Both

databases contain 3111 tuples. In addition, we collected detailed geographic outlines (com-

posed of longitude and latitude coordinates) of each of the counties and states in the USA.

The original data set contained 250,000 tuples. We also generated a simplified version of the

geographic data with a total of 36,382 tuples.

Polaris: A Visual Formalism

A key contribution of this thesis is a formalism for describing visualizations of multidimensional

data warehouses. In this chapter, we provide an overview of this formalism, the types of graph-

ics it can describe, and the related research. In the following three chapters, we provide detailed

information about the components of the formalism.

3.1 Benefits of a formalism

In developing our visual formalism and in designing systems that leverage that formalism, we have

identified many benefits to having an underlying formal structure in a visualization system:

Unification: Our formalism uses the same algebraic notation to describe tables, graphs,

graphic tables, and tables of graphs. Tables and graphs are often thought of as separate,

unrelated concepts, but in reality have a common structure that is exposed by our algebra.

Unification is a general benefit of formal approaches: they take distinct concepts and reveal

relationships between them. Understanding the structure of a subset of visualizations helps

us to identify the space of possible visualizations and guides future design.

Structures and patterns: Our formalism can be used to precisely specify the structure of

visualizations. When effective visualizations are developed for a specific problem domain,

the formal description of that visualization can be used as a design pattern and be applied to

a different data set, domain, or analysis.

Expressiveness of the language:A typical approach to the design of visualization systems

is to present a palette of possible graphs. Each graph type within the palette is treated as a

CHAPTER 3. POLARIS: A VISUAL FORMALISM 19

separate entity, minimally related to the others. By building visualization systems on top of

a formalism, we provide a set of building blocks and rules for combining them into visual-

izations. Providing these smaller granularity building blocks leverages the laws of combina-

torics, and as a result, the expressiveness of the system is limited not by the designs included

in our palette, but rather by the user’s imagination in constructing displays.

Formal transformations: A formal specification of a graphic and its structure makes it possi-

ble to perform analysis and transformations on the specification. Analysis may be performed

to determine the xpressiveness of a graphic(a concept introduced by Mackinlay [48]), de-

termine the database queries necessary to retrieve the data to be displayed, or transform the

graphic, perhaps adapting it to a different device such as a PDA.

Succinct description:Using our formalism we can generate succinct descriptions of sophis-

ticated visualizations, thus making many desirable but complex features simple to implement.

For example, an extensive history of the visualizations created during an analysis session can

be kept simply by maintaining a list of visual specifications. This history can be used to

provide extensive undo/redo capabilities and to capture the analysis for summary presenta-

tion. Similarly, collaborative visualization can leverage the formalism by sharing lightweight

specifications.

Interface semantics:The formalism precisely defines the operations (and semantics of those

operations) that we can apply to objects within our system. By basing our user interface on

these operations, we generate an interface that is simple, consistent, and intuitive.

Code simplicity: Our visualization systems are built using lightweight code objects that

capture the simple building blocks within the formalism. Other systems are built around

collections of monolithic objects that implement each available graph type within the system.

However, there are some negative repercussions to having an underlying formal structure. The

primary disadvantage of a formalism is that it limits the user to the displays that can be described

using the formalism. In many cases, it is simpler to implement a new display type as a stand-alone

entity than to extend the formalism to include that display. However, in our experience, as long as

the formalism expresses a reasonable set of useful displays, the benefits of the formalism greatly

outweigh this limitation. In developing our formalism, we were careful to ensure that it described

a set of commonly used analytic displays that could be applied to a wide range of analyses and

hypotheses.

https://lh3.googleusercontent.com/notebooklm/AG60hOpkiEcQ5n0U0jlXFuUVIiEYqADUQu05whvvXO4hKncAOi2UDzR_XG-RyTh7tWMWkJZuV7Bi2lh32vuHWUUFOAW4USCHxCvGZpf1nyY31bkiRogjNEOW4zHTyTA16DBhkiJRvw1y3g=w1800-h1077-v0

6c62f8e6-4799-4579-bb18-971dd80280e0

CHAPTER 3. POLARIS: A VISUAL FORMALISM 20

Figure 3.1: A visualization of sales for the coffee chain data set (COFFEE) categorized by the product type and quarter. The visualization was generated from a Polaris visual specification, which describes table-based visualizations of multidimensional data. The table consists of a number of rows and columns. Within each pane of the table, the tuples are visually encoded as a set of marks to create a graphic.

3.2 Table-based displays

The Polaris formalism describes table-based visualizations of multidimensional data. In the formal-

ism, a table consists of a number of rows, columns, and layers. Each table axis may contain multiple

nested dimensions. Each table entry, or pane, contains a set of tuples that are visually encoded as a

set of marks to create a graphic. Figures 3.1 and 3.2 illustrate the structure of a Polaris table.

Several characteristics of tables make them particularly effective for displaying multidimen-

sional data:

Multivariate: Multiple dimensions of the data can be explicitly encoded in the structure of

the table, enabling the display of high-dimensional data.

Comparative: Tables generate small-multiple displays of information, which, as Tufte [72]

explains, are easily compared, exposing patterns and trends across dimensions of the data.

https://lh3.googleusercontent.com/notebooklm/AG60hOrvhxDHa5vU_pB9pR4D_JEl2kwCzOrvRouGm2_81tR93Exs9JGYtBTGFJBH270WQt3K7FlxkBosyHUk2PWONfj8yMgqHGNDY6jPpbAppX0cZTMpuR0QiJfbVdgugURnGRDuVGoVIQ=w1800-h925-v0

b7fa73ca-f653-43ef-b517-605c0d3b5f3f

CHAPTER 3. POLARIS: A VISUAL FORMALISM 21

Figure 3.2: A visualization of flights between major airports in the USA (FLIGHTS). This visu-alization demonstrates one use oflayers, which is to combine heterogeneous data sources. There are three layers in the visualization, each displaying data from a different data source: (1) polygons representing state boundaries, (2) lines depicting flights between airports, and (3) circles indicat-ing the locations of the major airports. Layers are composited back-to-front to generate the final visualization. Chapter 4 discusses layers in more detail.

Familiar: Table-based displays have an extensive history. Statisticians are accustomed to us-

ing tabular displays of graphs, such as scatterplot matrices or Trellis displays [2], for analysis.

Pivot Tables [49] are also a common interface to large data warehouses.

3.3 Visual specifications

In the Polaris formalism, visualizations are described byvisual specifications. A visual specification

precisely defines:

The mapping of data sources to layers. Multiple data sources may be combined (visually

joined) in a single Polaris visualization (see Figure 3.2). Each data source maps to a separate

layer or set of layers.

The number of rows, columns, and layers in the table and their relative orders (left to right as

well as back to front). The table structure or table configuration is described using an algebra

CHAPTER 3. POLARIS: A VISUAL FORMALISM 22

involving the fields of the database. The algebra is discussed in detail in Chapter 4.

The selection of tuples from the database and the partitioning of tuples into different layers

and panes.

The grouping or level of detail of data within a pane and computation of statistical properties,

aggregates, and other derived fields. Tuples may also be sorted into a given drawing order.

Data transformations are discussed briefly in Chapter 6.

The type of graphic displayed in each pane of the table. Each graphic consists of a set of

marks, one mark per tuple (or set of tuples) in that pane. We have developed a taxonomy of

graphics that results in an intuitive and concise specification of graphic types and is discussed

in Chapter 5.

The mapping of data fields to visual properties of the marks in the graphics, such as color,

size, and shape. The mappings used for any given visualization are shown in a set of au-

tomatically generated legends. We have developed a system for effective visual encoding,

described in Chapter 5, which is derived from research in semiology, cartography, cognition,

and perception.

The visual specification describes not only the graphic to be generated and the desired data

abstraction and transformation, but it also captures the desired analysis task. Visual specifications

are automatically compiled into both the data and graphical transformations, allowing us to integrate

statistical analysis and visualization. Figure 3.3 provides a summary of the structure of a visual

specification.

3.4 Related work

In this section, we provide an overview of the most relevant research on graphic formalisms and the

relationship between that research and the Polaris formalism.

3.4.1 Semiology of Graphics

Bertin’sSemiology of Graphics[7] is one of the earliest attempts at formalizing graphing techniques.

Bertin considered graphics as a sign-system or language and developed a theoretical vocabulary for

describing data and the encoding of data in a graphic. This language structured graphics into four

https://lh3.googleusercontent.com/notebooklm/AG60hOqIlIM-eP72X5pxNEMaxi7T_CrHZQaqdPK2szIRfvo3t9XR0J8pWSRnSjq6qFrNn7DCybipqTl8TpGoKokxzXssLDWiU3wIod3uM0qyY4rZtsNjbnW3e-Se6K9PRyz7hdd84Sq0ew=w1711-h1526-v0

2bcbdfa8-ac0b-479e-ba03-12ccc1518a05

CHAPTER 3. POLARIS: A VISUAL FORMALISM 23

Figure 3.3: The major components of a visual specification. A visual specification formally de-scribes a table-based visualization of multidimensional data.

categories: diagrams, networks, maps, and symbolisms. Within each category, Bertin describes

construction rules for encoding data in a graphic of that type. One of his important contributions is

identifying eight retinal variables (position, color, size, etc.) in which data can be encoded.

Polaris builds on all aspects of Bertin’s work. Our formalism extends his initial concepts to the

construction of sophisticated tables of graphics and the description of database queries. We have

redefined Bertin’s retinal variables to develop an encoding system designed for use in an interactive

system and we have developed a taxonomy of graphics that simplifies the specification of a single

graphic. Furthermore, our formalism has been used to engineer an actual visualization system

CHAPTER 3. POLARIS: A VISUAL FORMALISM 24

whereas Bertin’s work presented a theoretical approach to graphics.

3.4.2 APT and Sage

The work most similar to ours, and which we most directly build upon, is Mackinlay’s APT sys-

tem [48]. APT is one of the first applications of formal graphical specifications to computer-

generated displays. APT uses a set of graphical languages and composition rules to automatically

generate 2D displays of relational data.

In this algebra, graphical presentations are sentences of graphical languages. Using the Principle

of Composition [48], “The graphical sentences of two languages can be composed by merging parts

that encode the same information.” The algebra includes several composition operators, such as the

double axes composition operator that merges two designs with the same x- and y-axes. Using these

composition operators, the system generates complex presentations from simple ones.

Polaris differs from APT in several significant ways. APT was designed with the intent of

automatically generating presentations of given relations using rules for assessing the effectiveness

and expressiveness of graphics. APT did not address the extraction of relations from a database or

involve the user in the design of the visualization. Rather than being designed to replace the user,

the Polaris formalism was designed to augment them, providing them with a language which they

could use to quickly build and refine visualizations and queries during the analysis process.

A second difference between Polaris and APT is that they are duals with respect to the parti-

tioning of space. APT constructs displays by repeated composition of simple displays whereas the

Polaris formalism constructs displays by partitioning space into a series of related graphics to form

a table-based visualization. The partitioning approach taken by Polaris is a good match for visual-

izing multidimensional data, as it supports the encodings of dimensions in the overall structure of

the table in addition to encoding them within the individual graphics.

The Sage [35][57] system extends the concepts of APT to develop an expert system for graphic

design and presentation. In addition to providing a richer set of data characterizations and gen-

erating a wider range of displays, the Sage project uses the compositional algebra within a set of

interactive tools. Using these interactive tools, users can construct graphics by dragging field names

onto a canvas and selecting a graphic and mark type. These tools, however, do not expose the ex-

pressiveness of the compositional algebra. Instead, they are limited to the creation of individual

two-dimensional graphs rather than tables of graphs or compositions of multiple graphs.

The Sage project did address the specification of database queries [25]. However, they ap-

proached specifying a query as distinct from specifying a visualization. Using their tools, the user

CHAPTER 3. POLARIS: A VISUAL FORMALISM 25

interacts with a graphic representation of the database schema to define a query and then links the

query to a graphic. Polaris combines the specification of graphics and query into a single concept

and thus a single interface.

3.4.3 DEVise

In the DEVise project, Livny et al. [52] describe a visualization model that provides a foundation for

database-style processing of visual queries. Within this model, the relational queries and graphical

mappings necessary to generate visualizations are defined by a set of relational operators. This for-

malism was the first (that we are aware of) to establish a relationship between graphics formalisms

and relational algebra.

Within DEVise, visualization is based on mapping source records or tuples to visual symbols on

the screen. Visualizations are generated by mapping each database tuple to a new “graphic” tuple

that describes a mark and a set of visual attributes. Given this model, interactions such as filtering,

panning, or linked views can be formalized as relational operations on graphic tuples.

Although DEVise formalizes interaction as relational operations, they do not completely inte-

grate database query with visualization. The DEVise model assumes a set of tuples to be visualized

is given as input and provides mechanisms for graphically encoding those tuples. The specifications

in the Polaris formalism describe the analysis task and are compiled into database queries to identify

and retrieve the appropriate set of tuples. Furthermore, the visual specifications within the Polaris

formalism describe a much richer set of graphic displays than the DEVise model.

3.4.4 A Grammar of Graphics

Wilkinson [77][78] recently developed a comprehensive language for describing traditional sta-

tistical graphs. Although seemingly similar on initial inspection, Wilkinson’s formalism is quite

different from our approach.

Similar to Polaris, Wilkinson’s work uses an algebra to describe visualizations based on panes of

graphics. However, his algebra is functionally different from ours. It does not address the generation

of queries, but instead, assumes the set of data to be visualized is given and simply describes the

structure of the graphic. The formalism does support the statistical transformation of the given

data before display, but does not support generating queries to an external database to generate the

Furthermore, Wilkinson’s algebra requires that a symbolic algebra machine convert general

CHAPTER 3. POLARIS: A VISUAL FORMALISM 26

expressions into a specific algebraic form before the operands can be converted to sets. The manip-

ulations performed by this symbolic machine cannot be performed directly on the sets because his

stated algebraic properties do not hold when applied to sets. For example, given his definitions of

Blend as a set operator, it is not commutative or distributive. However, his rewrite rules support the

commuting and distribution of Blend to transform an expression into the required algebraic form.

This deficiency in his algebra can also been seen in the output of the operators: Nest produces a

tagged set, which is functionally different than its input operands (or those of the other operators).

The Polaris algebra is a true set algebra and does not require this symbolic rewriting.

Another significant difference arises in the data model supported by the two formalisms. We

chose to focus on developing a tool for multidimensional relational databases, and we decided to

build as much of the system as possible using relational algebra. As a result, the data transformations

required by our visual specifications can be precisely interpreted as standard SQL or multidimen-

sional queries to OLAP servers (as will be shown in Chapter 6). Most corporate and scientific data

is stored in relational databases, and thus it was a key design requirement that all specifications be

valid when applied to relational databases. Wilkinson instead intentionally uses a data model that is

not relational, citing shortcomings in the relational model’s support for statistical analysis. As a re-

sult, many of his specifications cannot be interpreted as relational queries. By his own account [77],

“Few of the graphics in (his) book and in other important applications can be computed from a data

cube.” Furthermore, he does not demonstrate a method for compiling his formal specifications into

the necessary data transformations or queries

Other differences also exist. For example, we use three expressions to specify the axes of the

graphic; he performs algebraic manipulation on a single expression to determine an assignment of

variables to axes. Our choice reflects our desire to design a formalism that could be easily mapped

to an interactive interface. As a result, we designed our algebra so that operators, operands, and

expressions map to understandable concepts in the interface. We also identified graphics, such as

scatterplot matrices, that we felt should be easy to construct using the algebra and designed our

algebra accordingly.

Finally, Wilkinson does not have an equivalent to our dot operator and thus does not support

operations on hierarchical data cubes. It might be possible to support hierarchical data cubes in his

algebra if extensions were added to his Nest operator to reference hierarchical sets in the evaluation

of the operator.

The first component of our formalism we consider is the specification of the underlying tabular

structure of the visualizations. We need a formal mechanism to specify these table configurations

and we have defined an algebra for this purpose. A complete table configuration consists of three

separate expressions in this algebra. Two of the expressions define the configuration of the x- and

y-axes of the table, partitioning the table into rows and columns. The third expression defines the

z-axis of the table, which partitions the display into layers of x-y tables that are composited on top of

one another. Figure 4.1 depicts a table configuration and the corresponding expressions that define

this configuration.

In this chapter, we will provide an overview of our algebra, precisely define its syntax and

semantics, and then conclude with an explanation of how table structure relates to data flow.

4.1 Overview and set interpretations

Each expression in our algebra is composed of operands connected by operators. The operands in

the algebra are the fields of the database. Each operand is evaluated to a sequence1 of p-tuples (the

set interpretation), and the operators define how to combine two sequences. Thus, each expression

can be interpreted as a single sequence (thenormalized set form), where each tuple in the sequence

corresponds to, and defines, a single row, column, or layer of the table.

We formally define a set interpretation and its component types, p-entry and p-tuple, as follows:

1A mathematicalsequenceis an ordered list of elements that allows duplicate members. In contrast, aset is an unordered collection of unique members and abag is an unordered collection of possibly duplicate members.

https://lh3.googleusercontent.com/notebooklm/AG60hOocvDfuxK8byD2UHUWlKoUG5KG7ke0A8DXmeT8bkPoptzTipqyjs3wy4OkbLGqEn_7olbQejJ4aj7h4cM50QsgWLOLFbegHkRJwgg4kQhmf-JWwvG-gp0fvKu_4yTU3hRTWTUMNPA=w1800-h1326-v0

17c45ec6-b89f-4da0-b5e9-fcdb09191b27

CHAPTER 4. ALGEBRA 28

Figure 4.1: The Polaris formalism uses an algebra to specify the table structure underlying a vi-sualization. Two algebraic expressions define the rows, columns, and spatial encodings on the x-and y-axes of the table. In this visualization, which displaysProfit information for the coffee chain data set (COFFEE), the y-axis is defined by the expressionPr fit+(Market×ProductType) and the x-axis is defined by the expression(Quarter/ Month).

Definition 4.1. A p-entryis an ordered “tag-value” pair where thetagdefines the meaning and pos-

sible values of thevaluemember of the pair. A p-entry will be written astag:value;e.g.,field:Profit.

The tag must be one of the following:

field This tag indicates that the value member is the name of a quantitative attribute from the

schema of the fact table. When interpreted as the definition of a row, an entry of this type

defines a spatial encoding within the row. e.g.,field:Profit.

constant This tag indicates that the value member is a constant string. When interpreted as the

definition of a row, an entry of this type is a placeholder and has no effect. e.g.,constant:Foo.

CHAPTER 4. ALGEBRA 29

{fieldname} In this case, the tag itself must be the name of an attribute of the fact table and the

value must be a member of the domain of that attribute. When interpreted as the definition of

a row, an entry of this type defines a selection criteria that determines which database tuples

will be displayed in the row. e.g.,Month:January.

Definition 4.2. A p-tupleis a finite sequence ofp-entries. A single p-tuple defines a single row (or

column or layer). The entries in a p-tuple define the spatial encoding (axis) within the row and the

selection criteria on the fact table.

Definition 4.3. Theset interpretationof an operand is a finite (possibly empty) sequence of het-

erogeneousp-tuples. Each p-tuple in a set interpretation defines a row (or column or layer) of the

In summary, each axis of the table is defined by an expression that can be rewritten in normalized

set form. For example, in Figure 4.1, the y-axis is defined by the expressionPr fit + (Market× ProductType) which can be rewritten as:

{(field:Profit), (Market:Central, ProductType:Coffee), (Market:Central, ProductType:Expresso),

(Market:Central, ProductType:Herbal Tea), (Market:Central, ProductType:Tea), (Mar-

ket:East, ProductType:Coffee), (Market:East, ProductType:Expresso), (Market:East, Pro-

ductType:Herbal Tea), (Market:East, ProductType:Tea), (Market:West, ProductType:Coffee),

(Market:West, ProductType:Expresso), (Market:West, ProductType:Herbal Tea), (Mar-

ket:West, ProductType:Tea)}

The cardinality of this normalized set determines the number of rows (or columns or layers)

along the axis2. In our example, the set contains 13 entries and the visualization contains 13 rows.

Each p-tuple within the normalized set defines a row; its p-entries define both a selection criteria

on the fact table, selecting tuples to be displayed in the row, and the spatial encoding in the row,

defining the positions of the graphical marks used to visualize the database tuples. In the exam-

ple, the first row is represented by the p-tuple “(field:Profit)”, which defines the Profit axes within

the first row of the graphic. The second row is represented by the p-tuple “(Market:Central, Pro-

ductType:Coffee)”, which imposes the selection criteria that all tuples in the row represent facts

about coffee sales in the Central market. Having defined a set interpretation, we can now define the

operands and operators of our algebra, and their semantics.

2With the exception of when the normalized set is the empty sequence, in which case we create a single row or column rather than zero rows or columns

CHAPTER 4. ALGEBRA 30

4.2 Operands

The operands in this table algebra are the names of the fields (field operands) of the database and

the names of predefined constant sequences of p-tuples (con tant operands). For the purposes of

our algebra, we reduce the categorization of field types to ordinal and quantitative by assigning a

default alphabetic ordering to all nominal fields and then treating them as ordinal (as discussed in

Section 2.4). Thus, we have three class of operands: (1) ordinal field operands, (2) quantitative

field operands, and (3) constant operands. Throughout the remainder of this chapter, we will useA

andB to represent ordinal field operands,P andQ to represent quantitative field operands,C to

represent a constant operand, andX, Y , andZ to represent expressions.

4.2.1 Set interpretations

We assign set interpretations to each operand symbol in the following manner: to ordinal fields, we

assign the members of the ordered domain of the field; to quantitative fields, we assign the single

element set containing the field name; and to the constant operands, we assign their predefined set

interpretation.

A = domain(A) =< (A : a1), . . . , (A : an) >

P = < (field : P ) >

C = < (constant : c1), . . . , (constant : cm) >

For simplicity of exposition, we will not include the tags in the remaining set interpretations

within this chapter except where necessary.

The assignment of sets to field operands reflects the difference in how the two types of fields

will be encoded in the structure of the tables. Ordinal fields partition the table (and the database

tuples) into rows and columns, whereas quantitative fields are spatially encoded as axes within the

4.2.2 Constant operands

Constant operands define neither selection criteria nor spatial encodings. They, instead, can be used

to generate additional rows without partitioning the database tuples. This use will be important in

layering heterogeneous databases, as we will see in Section 4.6. We can treat constant operands as

CHAPTER 4. ALGEBRA 31

ordinal field operands by defining avirtual fact tableand then defining our operators relative to this

virtual fact table.

Definition 4.4. Let (C1, . . . , Cn) be a set of constant operands;RCi be a relation with a single

attribute (Ci) whose domain corresponds to the predefined set interpretation ofCi; andFT be the

fact table for the data warehouse. We define thevirtual fact tableVFT relative to the given set of

constant operands as:

VFT = FT ×RC1 × . . .×RCn

Our algebra contains one predefined constant operand: theempty sequence(∅ =<>).

4.2.3 Filtering and sorting of field operands

If a field is to be filtered (or sorted), the filtered and sorted domain is listed directly after the field

operand in the expression, in effect specifying a set interpretation for the operand. Given an ordinal

field A with domain(A) =< (a1), . . . , (an) >, we can filter and sort the operand within an expres-

sion by stating the filtered and sorted domain (< b1, . . . , bj >, bi ∈ domain(A)) directly after the

ordinal operand and the set interpretation is the listed domain:

A[b1, . . . , bj ] = < (b1), . . . , (bj) >

Similarly, a filtered domain can be specified for a quantitative field by listing the minimum

and maximum values of the desired domain. This information is included in the generated set

interpretation:

P [min,max] = < (field : P [min,max]) >

Having defined the operands and the generation of their set interpretations, we can now define

the four operators in our algebra.

CHAPTER 4. ALGEBRA 32

4.3 Operators

As stated above, a valid expression in the algebra is an ordered sequence of one or more operands

with operators between each pair of adjacent operands. The operators in this algebra, in order of

precedence, are dot (.), cross (×), nest (/), and concatenation (+); parentheses can be used to alter

the precedence. Because each operand is interpreted as a sequence, the precise semantics of each

operator are defined in terms of how it combines two sequences (one each from the left and right

operands) into a single sequence.

4.3.1 Concatenation

The concatenation operator performs an ordered union of the set interpretations of the two operands

and can be applied to any two operands or expressions:

A + B = < (a1), . . . , (an) > ∪ < (b1), . . . , (bm) >

= < (a1), . . . , (an), (b1), . . . , (bm) >

P + Q = < (P ) > ∪ < (Q) >

= < (P ), (Q) >

A + P = < (a1), . . . , (an) > ∪ < (P ) >

= < (a1), . . . , (an), (P ) >

P + A = < (P ) > ∪ < (a1), . . . , (an) >

= < (P ), (a1), . . . , (an) >

X + Y = < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > ∪ < (y11, . . . , y1m), . . . , (yn1, . . . , yno) >

= < (x11, . . . , x1i), . . . , (xj1, . . . , xjk), (y11, . . . , y1m), . . . , (yn1, . . . , yno) >

The only algebraic property that holds for the concatenation operator is associativity:

CHAPTER 4. ALGEBRA 33

(X + Y ) + Z =

(< (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > ∪ < (y11, . . . , y1m), . . . , (yn1, . . . , yno) >) ∪

< (z11, . . . , z1p), . . . , (zq1, . . . , zqr) >

= < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > ∪

(< (y11, . . . , y1m), . . . , (yn1, . . . , yno) > ∪ < (z11, . . . , z1p), . . . , (zq1, . . . , zqr) >)

= X + (Y + Z)

The concatenation operator is not commutative because the ordered union of two sequences is

not commutative. Concatenation of an expression with the empty sequence produces the expected

X + ∅ = < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > ∪ <>

= < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) >

4.3.2 Cross

The cross operator performs a Cartesian product of the sets of the two symbols:

A×B = < (a1), . . . , (an) > × < (b1), . . . , (bm) >

= < (a1, b1), . . . , (a1, bm), . . . , (an, b1), . . . , (an, bm) >

A× P = < (a1), . . . , (an) > × < (P ) >

= < (a1, P ), . . . , (an, P ) >

X × Y = < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > × < (y11, . . . , y1m), . . . , (yn1, . . . , yno) >

= < (x11, . . . , x1i, y11, . . . , y1m), . . . , (x11, . . . , x1i, yn1, . . . , yno)

(xj1, . . . , xjk, y11, . . . , y1m), . . . , (xj1, . . . , xjk, yn1, . . . , yno) >

CHAPTER 4. ALGEBRA 34

Quantitative fields and expressions may appear only as right-hand side operands when the cross

operator is applied. This semantic constraint is discussed further in Section 4.5. The cross operator

is also associative but not commutative (because the ordered Cartesian product is not commutative):

(X × Y )× Z =

(< (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > × < (y11, . . . , y1m), . . . , (yn1, . . . , yno) >)×

< (z11, . . . , z1p), . . . , (zq1, . . . , zqr) >

= < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > ×

(< (y11, . . . , y1m), . . . , (yn1, . . . , yno) > × < (z11, . . . , z1p), . . . , (zq1, . . . , zqr) >)

= X × (Y × Z)

The cross of an expression with the empty sequence produces the empty sequence:

X × ∅ = < (x11, . . . , x1i), . . . , (xj1, . . . , xjk) > × <>

4.3.3 Nest

The nest operator is similar to the cross operator, but it only creates set entries for which there exist

database tuples with the same domain values. If we defineVFT to be the virtual fact table of the

database being analyzed relative to all constant operands in the expressionsX andY , t to be a tuple,

andt(X1 . . . Xn) to be the values of the fieldsX1 throughXn for the tuplet, then we can define

the nest operator as follows:

CHAPTER 4. ALGEBRA 35

A/B = < (a, b) | ∃t ∈ VFTst

((a) ∈ A) & (t(A) = a) & ((b) ∈ B) & (t(B) = b) >

X/A = < (x1, . . . , xn, a) | ∃t ∈ VFTst

((x1, . . . , xn) ∈ X) & (t(X1 . . . Xn) = (x1, . . . , xn)) &

((a) ∈ A) & (t(A) = (a)) >

A/Y = < (a, y1, . . . , ym) | ∃t ∈ VFTst

((a) ∈ A) & (t(A) = (a)) &

((y1, . . . , ym) ∈ Y ) & (t(Y1 . . . Ym) = (y1, . . . , ym)) >

X/Y = < (x1, . . . , xn, y1, . . . , ym) | ∃t ∈ VFTst

((x1, . . . , xn) ∈ X) & (t(X1 . . . Xn) = (x1, . . . , xn)) &

((y1, . . . , ym) ∈ Y ) & (t(Y1 . . . Ym) = (y1, . . . , ym)) >

The ordering of the p-tuples in a sequence generated by application of the nest operator is the

same as it would be in the sequence generated by the application of the cross operator to the same

The intuitive interpretation of the nest operator is “B within A”. For example, given the fields

Quarter andMonth, the expressionQuarter/ Monthwould be interpreted as those months within

each quarter, resulting in three entries for each quarter (assuming data exists for all months in the

fact table). In contrast,Quarter× Month would result in 12 entries for each quarter. The nest

operator may only be applied to ordinal operands and expressions.

Nest is an associative operator:

Proof. (by contradiction) Assume the p-tuplep = (a, b, c) appears in the normalized set form of

(A/B)/C but not inA/(B/C). If p exists in(A/B)/C then there must exist some tuplet ∈ VFT

such thatt(ABC) = (a, b, c). However, ifp is not inA/(B/C) then¬(∃t ∈ VFT st t(ABC) =

(a, b, c)) which is a contradiction. The inverse is proved similarly. Finally, the ordering of p-tuples

in (A/B)/C is determined by the ordering of p-tuples in(A×B)×C. Because× is associative, this

ordering must be the same as forA/(B/C), which is determined by the ordering ofA×(B×C).

CHAPTER 4. ALGEBRA 36

The cross and nest operators provide us with tools for generating ad hoc categorical hierarchies. As

we discussed in Chapter 2, however, data warehouses often contain dimensions with explicit seman-

tic hierarchies. The dot operator provides a mechanism for exploiting these hierarchical structures

in our algebra. The dot operator is similar to the nest operator but is “hierarchy-aware”.

If we defineDT to be a relational dimension table defining a hierarchy that contains the levels

A andB, andA precedesB in the schema ofDT , then:

A.B =< (a, b) | ∃t ∈ DT st t(A) = a & t(B) = b >

Similarly, we can define dot relative to an expressionX involving only the dot operator and

levels from the same dimension hierarchy. We defineDT to be the relational dimension table

defining the dimension that contains all levels inX and the dimension levelA. In addition, we

require that all levels inX appear in the schema ofDT in the order they appear inX and that they

precedeA in the schema ofDT . Then we have:

X.A =< (x1, . . . , xn, a) | ∃t ∈ DT st t(X1 . . . Xn) = (x1, . . . , xn) & t(A) = a >

The dot operator is also associative but not commutative. The proof of associativity parallels

the proof for nest.

Nest could be used for drilling down into a hierarchy but this usage would be flawed. The nest

operator is unaware of any defined hierarchical relationship between the dimension levels; instead,

it derives a relationship based on the tuples in the fact table. Not only is this approach inefficient,

as fact tables are often quite large, but it can also yield incorrect results. For example, consider the

situation where no data was logged for November. Application of the nest operator toQuarterand

Monthwould result in an incorrectly derived hierarchy that did not include November as a child of

Quarter 4.

4.3.5 Summary

Using the above set semantics for each operator, every expression in the algebra can be reduced to

a single set with each entry in the set being an ordered p-tuple. We call this set evaluation of an

expression thenormalized set form. The normalized set form of an expression determines one axis

CHAPTER 4. ALGEBRA 37

of the table: the table axis is partitioned into columns (or rows or layers) so that there is a one-to-

one correspondence between set entries in the normalized set and columns. Figure 4.2 illustrates

the axis configurations resulting from several expressions.

4.4 Algebraic properties

We interpret an algebraic expression as a set for two purposes: to determine the underlying tabular

structure of a visualization and to determine the tuples to be retrieved from the database. In the for-

mer case, the ordering of the p-tuples in the normalized set form is meaningful because it determines

the ordering of the columns, rows, and layers of the visualization. As a result, the only algebraic

property that holds for our operators is associativity. Commutative or distributive operators would

allow algebraic manipulations that change the ordering of the normalized set form.

However, when performing interpretation to determine which database tuples to retrieve, we

can relax the constraints on the properties of our operators since the ordering of the p-tuples in the

set interpretation is not meaningful in the context of database queries. Specifically, for this purpose

only, we treat the set interpretations as bags instead of sequences (thus discarding ordering) and

allow the following algebraic properties:

## Associative

(A + B) + C = A + (B + C)

(A.B).C = A.(B.C)

(A×B)× C = A× (B × C)

(A/B)/C = A/(B/C)

## Distributive

A× (B + C) = (A×B) + (A× C)

A/(B + C) = (A/B) + (A/C)

https://lh3.googleusercontent.com/notebooklm/AG60hOpeFz7hgA9h1laCVq5kVjZoLgTEciOOW0ly5TwF36-z3Nih1I9gapnhUUClIeHdgo78H6uSqTitcsfTlpvovb6ZJmAATNgsaNgT0Imjwl9fQTc-QJ3xqY_kAZauO3xYDCkizzqP=w1800-h1788-v0

36eda24f-67a5-4b39-8b5b-fc5206c4c244

CHAPTER 4. ALGEBRA 38

Figure 4.2: The set interpretation and semantics of several expressions in the Polaris formalism’s algebra. The ordinal fields partition an axis into additional columns and the quantitative fields are spatially encoded along the axis of the current column. Note the subtle difference between the nest and dot operators when there is missing data, as illustrated here forNovember.

## Commutative

A + B = B + A

A×B = B ×A

https://lh3.googleusercontent.com/notebooklm/AG60hOqq_cchLi7-fU3tL3_gWhBWfhDxc_vHmpIWm1UA82eBvZ396ik3aF24Sar9wXkWgIKvJJI69av0cYcIh7nNQPl16o5BeGGLlOTq2DgoNuZTF2J87-RJKXQbLubqE4k-pcRDZhj6tg=w1800-h894-v0

020a0264-cb4c-4c9a-ae65-02ca134719cc

CHAPTER 4. ALGEBRA 39

Figure 4.3: In the Polaris formalism, a layer is a single x-y table. Each data source in a visualization is mapped to a distinct layer. The layers for a data source can be partitioned into additional layers by the z-axis expression for that data source. All the layers in a specification are composited together back-to-front to form the final visualization.

As we will see in Chapter 6, if in this context we disregard ordering and change our operators

to allow these algebraic properties, we can use algebraic manipulation of our axis expressions to

quickly determine the database queries or data cube projections required to generate a display.

4.5 Syntax revisited

In the previous sections, we informally defined the syntax of our algebra as a sequence of operands

separated by operators and provided some constraints on the applications of the operators. In this

section, we make the syntax precise by using a grammar. To define a grammar, we must define

four things: a set of terminal symbols, a set of non-terminals, a set of production rules, and a start

symbol. Our grammar has ten terminal symbols:

CHAPTER 4. ALGEBRA 40

qfield the name of a quantitative field

ofield the name of an ordinal field

qdim the name of a quantitative dimension level

odim the name of an ordinal dimension level

c a constant operand

.× /+ the operators of the algebra

() parentheses

The following are the production rules of our grammar (E is the start symbol):

E → Oexpr | Qexpr

Oexpr → (Oexpr) | Oexpr + Oexpr | Oexpr ×Oexpr | Oexpr/Oexpr | O Qexpr → (Qexpr) | E + Qexpr | Qexpr + E | (Oexpr ×Qexpr) | Q O → Ohier | ofield | c Ohier → Ohier.odim | odim

Q → Qhier | qfield

Qhier → Ohier.qdim | qdim

The following are the main syntactic constraints on the operators that are expressed in this grammar:

Cross: Quantitative operands, or expressions containing quantitative operands, can only be

right-hand side operands of the cross operator.

Nest: The nest operator can only be applied to ordinal operands or expressions.

Dot: The dot operator can only be applied to dimension levels. Furthermore, a quantitative

field can only appear as the right-most operand of a dot operator, since quantitative dimension

levels are only possible as the leaf level of a dimension hierarchy.

Concatenate:Concatenate can be applied to any operands.

Thus far, we have focused on how the algebraic expressions partition tables into rows and

columns; we now turn our attention to layers.

CHAPTER 4. ALGEBRA 41

4.6 Layers

Within Polaris, a layer is a single x-y table (whose structure is defined by the x- and y-axes ex-

pressions). Every layer in a specification is composited together back-to-front to form the final

visualization. A single visualization can combine multiple data sources; each data source is mapped

to a distinct layer or set of layers. While all data sources and layers share the same configuration

for the x- and y-axes of the table, each data source can have a different expression (the z-axis) for

partitioning its data into layers. Layering of multiple data sources and the partitioning of layers are

illustrated in Figure 4.3.

Constant operands are an important aspect of layering. A single visualization may be composed

of multiple heterogeneous databases, each mapped to a distinct layer, and all layers must share the

same expressions for the x- and y- axes. However, sometimes it is desirable to include ordinal

fields in the x- and y-axes expressions that exist in only a subset of the visualized databases. When

this occurs, constant operands are generated for the other layers with a predefined set interpretation

that matches the domain of the ordinal field in the layer in which the field does appear. Thus, the

expressions can be properly evaluated for each layer.

The z-axis expression for a data source is more constrained than the expressions for the x and

y-axes. Specifically, since layering must be discrete, a z-axis expression can contain only ordinal

operands; not quantitative operands. In other words, a z-axis expression is constrained to theOexpr

production rule in our grammar.

While we have not explored having quantitative expressions on the z-axis, possible interpre-

tations might be to map quantitative fields to a third dimension within panes (thus generating 3D

graphics) or to a visual variable associated with depth or layering, such as transparency or size. We

consider these possibilities as interesting future work.

4.7 Summary

Our algebra provides us with a succinct yet powerful notation for describing the underlying structure

of visualizations. The algebraic expressions define how the table is partitioned into rows, columns,

and layers, and additionally defines the spatial encodings within each pane of the table.

We discuss in detail in Chapter 6 how visual specifications are translated into database queries.

At this point, however, it is useful to consider the conceptual data flow. As well as defining the

table structure, the algebraic expressions define which tuples of the database should be selected

https://lh3.googleusercontent.com/notebooklm/AG60hOrBtMtKYuT_uIpZCN2bwHwR67xriL_Tcvbx9JF2PgxzTftg0EXU6ancUlvg-YkR-1dwhusumKFMNxLeB7HgJFlFOUN48I96EpuToVMTiZyjNZWctM_dBZBcB3ivXC30B8VqYqpFuA=w1802-h644-v0

34796d72-fad9-4129-9ca9-fa6133160573

CHAPTER 4. ALGEBRA 42

Figure 4.4: An abstract data flow diagram depicting the transformation of visual specifications into database queries and visualizations. Chapter 6 provides a more detailed discussion of the transformation of specifications into visualizations.

and mapped into each pane. When a specification is interpreted, one or more queries are generated

to retrieve tuples from the database. The resulting tuples are partitioned (and possibly copied) and

then sorted into tables for each pane. This conceptual data flow is abstractly illustrated in Figure 4.4.

Once the tuples have been sorted into panes, they are then mapped to graphic marks to generate a

perceivable display. We discuss this mapping in detail in the next chapter.

## Pane Graphics

In the previous chapter, we described how to specify the configuration of a table-based visualization

using expressions in our algebra. After the table configuration is specified, the next step is to specify

the graphic within each pane. Two aspects of a pane graphic must be specified:

### 1. The type of graphic: The typical approach is to have the user select a chart type from a

predefined set of charts. Polaris allows analysts to construct graphics flexibly by specifying

the individual components of the graphics. However, for this approach to be effective, the

specification must balance flexibility with succinctness. We have developed a taxonomy of

graphics that results in an intuitive and concise specification of graphic types.

### 2. The encoding of data as visual properties such as color:In addition to encoding data in the

position of a mark, designers can also encode dimensions and measures of the data in visual

properties of the marks such as color, size, orientation, and shape. We have developed a sys-

tem for effective visual encoding derived from research in semiology, cartography, cognition,

and perception.

In this chapter, we will describe the types of marks provided in our formalism, our taxonomy of

graphics, and our encoding system. In addition, we will describe how effective mappings from data

variables to visual variables can be constructed for each visual variable in our encoding system.

5.1 Graphic marks

Our algebra defines the configuration of a table (the rows, columns, and panes) and the axes within

each pane; it also sorts the tuples retrieved from the database into panes. To generate a graphic

CHAPTER 5. PANE GRAPHICS 44

within a pane, we must map each sorted tuple to a graphical mark. In this section we will introduce

the three mark types within Polaris and then discuss the details of visually displaying a set of tuples

using each different mark type.

5.1.1 Mark types

The distinguishing characteristics of a mark are the number of tuples encoded by the mark, the

visual properties available, and the meaning of the spatial extent of the mark. We classify the marks

in Polaris into three categories1:

Point marks: Point marks provide a graphical representation for a single tuple. A point, the-

oretically, has no size or shape. Thus, while the position of a point mark encodes data relative

to the spatial encoding, additional data can be encoded in the shape and size of the mark;

i.e., the shape and size are unconstrained. Marks that fall into this class include text, shapes

(points with a varying shape encoding), and icons (a fixed size mark with shape encoded as

Line marks: Line marks provide a graphical representation for an ordered set of tuples.

Each tuple represents a point along the line and the points are connected in order. Line marks

are more constrained then point marks: we cannot arbitrarily vary all aspects of their size,

instead, only their width is encodeable. Similarly, we can only encode additional data in the

fill pattern, rather then the shape of the line. Lines in a line chart and interval bars in a Gantt

chart are examples of line marks.

Area marks: Like line marks, area marks provide a graphical representation for an ordered

set of tuples. Each tuple represents a point along the boundary of the area and the points are

connected by interpolation (including the first and last point). The position, size, and shape

of an area mark are entirely determined by the spatial encoding. However, we can encode

additional data by varying the fill pattern and color of the mark. Areas in an area chart and

polygons in a map are examples of area marks.

This classification of marks is independent of the meaning of the underlying tuples and the data

transformations applied to those tuples. Instead, the classification scheme is based on the visual 1A fourth possible mark type is thesmart-mark. Smart-marks provide a graphic representation for a set of tuples. The

unique aspect of smart-marks (versus point marks) is that they utilize custom algorithms to determine their shape and size within the underlying spatial encoding. An example of a smart-mark is a box plot. With a box plot, the shape and size of the box plot encode five aspects of the distribution of tuples represented by that mark (for example, the quartile boundaries).

https://lh3.googleusercontent.com/notebooklm/AG60hOrUCWH-oYCr6wB-7aXvNne93qN_xZWDV_6BPpcrc_Be3vBXLNXL7Ab5UZqmO9_zxtLoGfD_HGq_u0t-txM7tqS0GriD0H7Rrn-XtQxIxLgftDIYbH8JHhmWTMuYL1qz9XHm1G8vzA=w1800-h1184-v0

9a4158e5-a4ba-4bba-a8a0-f401905db3ab

CHAPTER 5. PANE GRAPHICS 45

Figure 5.1: A map of the results of the 2000 Federal Election (ELECTION) displaying results, encoded as color (with blue indicating a Democratic win and red indicating a Republican win), by county. The tuples are rendered using the “Area” mark type and the tuples are grouped byCount ID and sorted within counties by theirPoint-ordervalue. This graphic is an example of a quantitative-quantitative graphic where both axis variables are independent.

properties of the different marks and how information is encoded in their graphic representation.

Some researchers [77] have taken an alternate approach of generating mark classifications based

on both the underlying data structure and the visual properties of the marks. We consider the data

transformations and graphic representation to be orthogonal and believe they should be considered

separately, thus allowing arbitrary combinations of statistical manipulation and graphic representa-

5.1.2 Specifying marks

The precise mapping of tuples to graphic marks within a pane is specified by several components of

the pane specification (see Figure 3.3):

https://lh3.googleusercontent.com/notebooklm/AG60hOocFZFoPZV2mqvh-qv3Lg00yYywcJbKbMD_fXi-cryZgvzuUbrnet5-_h-OdFgMBYS9xK2S0NIzxjm6Q7PCN63dgHYXzBb2hTBXDpGYNa73rCqmgSCbF2XSD_NC53-HcakGuSFq=w1800-h1033-v0

bdd6c753-16c6-44d3-9297-9d83df1c118e

CHAPTER 5. PANE GRAPHICS 46

Figure 5.2: A time series chart displaying the fluctuations in profit for the major markets of a hypothetical coffee chain (COFFEE). The tuples are rendered using the “Line” mark type and the tuples are grouped by theirMarketvalue and sorted within groups by the independent axis (Month) of the graph. The graphic is an example of an ordinal-quantitative graphic.

Mark: The “Mark” component specifies which mark type (point, line, area) within our cate-

gorization is to be used to graphically display the tuples. The visual encodings (size, shape,

etc.) specified for the mark further determine the specific mark (e.g., bar, text, image, etc.)

within each category.

Group: When the selected mark type is line or area, the contents of the “Group” component

determines the grouping of tuples into distinct lines and areas. The tuples are sorted within

the pane by the columns listed in this component (and the columns in the “Sort order” compo-

nent) and then rendered. While rendering, when any value in the “Group” columns changes

between successive tuples the current line is completed and a new line is started. For example,

if the “Group” component contains the column nameMonth, the tuples within the pane will

be sorted by theirMonthvalue and then rendered. When the value within theMonthcolumn

changes, a new line is drawn.

Sort order: The columns listed in the this component determine the order in which tuples

CHAPTER 5. PANE GRAPHICS 47

within a pane are drawn. When rendering point marks, these columns determine the order

in which points are drawn (thus determining which marks are drawn occluded). When ren-

dering tuples as lines (or areas), these columns are used to sort the tupleswithin the sets

corresponding to distinct linesinto the proper order for rendering. This is equivalent to or-

dering first by the columns in the “Group” component and then by the columns in the “Sort

order” component.

In the following section, we will provide several examples to clarify the specification of a mark

within a pane.

5.1.3 Using lines and polygons

As we discussed in the last section, when rendering the tuples within a pane as lines or areas,

we need to group the tuples into ordered sets corresponding to distinct marks. This grouping and

ordering is determined by the “Group” and “Sort order” components of the visual specification. In

this section, we consider several examples to further clarify the specification of line and polygon

The visualization in Figure 5.1 depicts the 2000 Federal Election results (ELECTION) using a

map of the counties in the USA. Within the map, each county is colored to indicate which party

received the most votes in that county. The database being visualized contains the following rele-

vant columns:CountyID, Point-order, LongitudeandLatitude. The relevant portions of the visual

specification are:

X: Longitude

Y: Latitude

Mark: Polygon

Group: CountyID

Sort order: Point-order

Each tuple corresponds to a single longitude and latitude coordinate along the border of a county.

The tuples are sorted into sets corresponding to each value in theCountyIDcolumn and then within

sets by thePoint-ordercolumn.

The visualization in Figure 5.2 depicts profits over time for the COFFEE data set. A line is

drawn for each market within the USA. The relevant portions of the visual specification are:

https://lh3.googleusercontent.com/notebooklm/AG60hOreSw_5J3U7qxreGGEUMOb5oGQuFEBA0A2UN-JTJD3G3yRpiZP73DO-OGzNNx0NDIE-twtUpu-ngbIbgvX5s88yzlM9rJQ_IwQbKGUsA-_Mnuy-HFLd6G22LmfHfJcbX5cND4Jr=w1800-h764-v0

afb2432d-b3b1-4f96-a6a4-bcb4b783556a

CHAPTER 5. PANE GRAPHICS 48

Figure 5.3: The families of graphics within our taxonomy with examples of well-known charts from each family. The taxonomy structures the space of graphics into three families by the types of fields assigned to their axes and then further structures each family by the number of independent and dependent variables. Using this taxonomy we can derive the type of graphic within each pane from the table axes expressions and the mark type.

Y: AvgProfit

Mark: Line

Group: Market

Sort order: Month

Note that, in this graph, the sort order is the same as the independent axis. To simplify common

specifications such as this one, it is possible to encode default rules in the interface. In this case,

we could include a rule that “if the Sort order is undefined in a graph with an independent axis then

automatically include a sort by the independent axis”. We now turn our attention to the specification

of the type of graphic within each pane.

5.2 Graphic types

We have developed a taxonomy of graphics that results in an intuitive and concise specification of

graphic types using only the table expressions and mark type. We structure the space of graphics

https://lh3.googleusercontent.com/notebooklm/AG60hOoueP00wCCPpvqPXg20TSm47mnq3h9btBuW66k-I_4ypmMHK1Q9XoMMAIBv1POLn4eXYwe-arH-G79rseYOKB6DXPtjEKKmjDhwkiLWmNBllCP5L-KSFqzYH8IhCRokbEnRTxVlqA=w1800-h1339-v0

d4bf80c0-7438-46f2-abee-1b5f04fd0d0a

CHAPTER 5. PANE GRAPHICS 49

Figure 5.4: An example of an ordinal-ordinal graphic: a graphical table displaying gene expression (a dependent variable encoded as color) as a function of experiment and gene (the independent variables encoded spatially as the x- and y-axes respectively) (GENE).

into three families by the type of fields assigned to their axes. As noted in Section 2.4, we treat both

ordinal and nominal fields as ordinal in this section since a nominal field must be given an ordering

to be drawn on an axis. The three families of graphics are:

Ordinal-Ordinal

Ordinal-Quantitative

Quantitative-Quantitative

Each family contains several variants depending on how tuples are mapped to marks. For exam-

ple, selecting a bar mark in an ordinal-quantitative pane will result in a bar chart, whereas selecting

a line mark results in a line chart. Following Cleveland [21], we further structure the space of

https://lh3.googleusercontent.com/notebooklm/AG60hOp0RDhNxUb87_LVZvVmzBrfCBvUDLtAX2dIHg6s5OVDPKrA1-GWiL8-IboxrXtCN6xFhVmi56s2rHydcapBWs6MZE_BU6dtc4HGbIHV3COV2nq8oNwAnvfGUULEo_0bRBgadWzvvA=w1800-h1339-v0

5ef63b6f-94bb-4b8a-94d4-36f908a69c9e

CHAPTER 5. PANE GRAPHICS 50

Figure 5.5: An example of an ordinal-quantitative graphic: a dot plot displaying the number of arrests (a dependent variable) as a function of the type of crime and the sex of the offender (the independent variables).

graphics by the number of independent and dependent variables. For example, a graphic where

both axes encode independent variables is different from a graphic where one axis encodes an inde-

pendent variable and the other encodes a dependent variable (y = f x)). Note that by default, the

dimensions of the database are interpreted as independent variables and the measures as dependent

variables. Finally, the precise form of the data transformations, in particular how tuples are grouped

and whether aggregates are formed, affects the cardinality of the data displayed and therefore the

type of graphic: Some graphic types best encode a single record, whereas others can encode an

arbitrary number of tuples. Figure 5.3 illustrates this taxonomy.

We now briefly discuss the defining characteristics of the three families within our categoriza-

https://lh3.googleusercontent.com/notebooklm/AG60hOr-tXgrt5vayq9XQTRNFgcj-BtaOug-Nf8MIxWgLQ5tonHu4HIwTSpmCk-5COJeIkuZC2vIFIXtSgO559YhKqQolak5b86hMrYvJPcCp5qLB5vsQTHzZOLjySfWlkT5dIPAytPY1A=w1800-h1339-v0

b51f88aa-2ecf-4455-b831-4f5c5f098403

CHAPTER 5. PANE GRAPHICS 51

Figure 5.6: A second example of a ordinal-quantitative graphics: a series of Gantt charts displaying locking activity for a parallel graphics library executing on a multiprocessor computer (ARGUS). In this graphic, the ordinal variable (CPU) and the quantitative variable (cycle) are both independent variables.

5.2.1 Ordinal-ordinal graphics

The characteristic member of this family is the table with either numbers or marks encoding at-

tributes of the source tuples. In ordinal-ordinal graphics, the axis variables are typically indepen-

dent of each other, and the task is focused on understanding patterns and trends in some function

f(cx, cy) → r, wherer represents the fields encoded in the retinal properties of the marks. An ex-

ample of this type of graphic can be seen in Figure 5.4 where the analyst is studying gene expression

as a function of gene and experiment.

The cardinality of the record set in each pane has little effect on the overall structure of the

table. When there is more than one record per pane, multiple marks are shown in each display, with

a one-to-one correspondence of mark to record. The marks are stacked in a specified drawing order,

https://lh3.googleusercontent.com/notebooklm/AG60hOqah2Y5MpYsK8wZKiQnc_8yKkLP4RM_QWBIF4Z-428Gl1iMrtuj-_t_061id-VpfEK1I7ZmzzKpusaPM9FoIEgN9M8jyvSCIz_3zFeX7CFwvAG47XnEgU7djgnfmd0Yj-rXQnmU=w1800-h1253-v0

1da7702e-8c24-4cb8-a614-dc18fdb11eca

CHAPTER 5. PANE GRAPHICS 52

Figure 5.7: An example of a quantitative-quantitative graphic: a scatterplot displaying the relation-ship between two attributes of different products sold by a coffee chain (COFFEE).

and the spatial placement of the marks within the pane conveys no additional information about the

record’s data.

5.2.2 Ordinal-quantitative graphics

Well-known representatives of this family of graphics are the bar chart (possibly clustered or stacked),

the dot plot, and the Gantt chart.

In a ordinal-quantitative graphic, the quantitative variable (q) is often dependent on the ordinal

variable (c) and the analyst is trying to understand or compare the properties of some function

f(c) → q. Figure 5.5 illustrates a case where a dot plot displays the dependent measure, number of

arrests, as a function of the type of crime and the sex of the offender. The cardinality of the record

set does affect the structure of the graphics in this family. When the cardinality of the record set

is one, the graphics are simple bar charts or dot plots. When the cardinality is greater than one,

CHAPTER 5. PANE GRAPHICS 53

additional structure may be introduced to accommodate the additional tuples (e.g., a stacked bar

The ordinal and quantitative values may also be independent variables, such as in a Gantt chart.

Here, each pane represents all the events in a category; each event has a type as well as a begin

and end time. Figure 5.6 shows a table of Gantt charts, with each Gantt chart displaying the thread

scheduling and locking activity on a CPU within a multiprocessor computer.

5.2.3 Quantitative-quantitative graphics

Graphics of this type are used to understand the distribution of data as a function of one or both

quantitative variables and to discover causal relationships between two dependent quantitative vari-

ables. Figure 5.7 illustrates a scatterplot graphic used to understand the relationships between two

attributes of different products sold by a coffee chain.

Figure 5.1 illustrates another example of a quantitative-quantitative graphic where both quanti-

tative variables are independent: the map. In this figure, the analyst is studying the results of the

2000 Federal Election by county.

It is extremely rare to use this type of graph with a cardinality of one, not because it is not

meaningful, but because the density of information in such a graphic is very low.

5.3 Visual variables

When a graphic is constructed, information is encoded in that graphic. Thus far, we have discussed

how information can be encoded in the position of marks in the plane. There are, however, many

othervisual propertiesthat designers can work with in constructing a graphic. Examples of visual

properties range from obvious properties such as color and size to blur, randomness, or transparency.

Within a visualization formalism, we must choose some subset of the possible visual properties to

expose asvisual variables.

Definition 5.1. Visual Property:A visible attribute of a graphic mark that can be varied over some

range of perceivable values.

Definition 5.2. Visual Variable: A visual property that a visualization makes available for encoding

or communicating data values.

The seminal work in this area is Bertin’sSemiology of Graphics[7]. Bertin identifies six vari-

ables in which he believes data can be successfully encoded: color (hue), value (luminance), size,

CHAPTER 5. PANE GRAPHICS 54

shape, texture (grain or granularity), and orientation. Subsequent research has revised and extended

this set of variables in numerous ways and with varying motivation. Morrison [50] and MacEach-

ern [47] analyze and restructure Bertin’s variables from a cartographer’s perspective, introducing

variables such as crispness and decomposing others such as texture. Wilkinson [77] has applied his

experience developing computer systems for statistical graphics to develop a set of visual variables.

The purpose of the encoding system we have developed is not to be a “PostScript for visualiza-

tion”. Rather than exposing all possible visual properties as visual variables, our goal is to identify

those properties that can be used to effectively encode data. Because our formalism was developed

as a basis for an interactive visualization system, we want to define visual variables that are un-

derstandable and can be used effectively without extensive design knowledge. It was in the design

of our interactive visualization system that we encountered several of the issues with the existing

encoding systems.

Given our goals and priorities, we applied the following requirements in deciding whichvisual

variablesto include in our encoding system:

Effectiveness:We restrict our set of visual variables to those that have been demonstrated, ei-

ther through practice or research, to be effective in encoding information (see Section 5.3.1).

For example, this criterion is reflected in our decision not to support color or pattern decom-

position into more variables or allow varying interior fill pattern for point symbols.

Simplicity: Because our formalism is the basis of interactive systems and the encoding sys-

tem will be exposed to the users, we want to select as small and simple a set of visual variables

as possible. Simultaneously encoding more than one or two dimensions as visual variables

without generating visual interference is difficult. This interference occurs because many

pairs of visual properties are inherently integral (i.e., not easily decomposable by an observer)

and simply cannot be used to display different dimensions simultaneously. Therefore, there

is little value in providing a large set of visual variables; furthermore, a small, simple set of

variables reduces the design choices that a user must make.

Independence: We have attempted to identify a set of visual variables that can be varied

independently of each other. For example, if varying the fill pattern of an area mark does not

result in a change in the area’s color, then these visual variables can vary independently.

Applicability to all mark types: We want all the visual variables to be meaningfully defined

for all mark types (point, line, area). Achieving this goal is difficult because the different

https://lh3.googleusercontent.com/notebooklm/AG60hOpeFAcH89OBQHOSnc28-8WTSLR_IrSQzivB1y4_hzdTGmUNEMN1jpLGfqBcCTvAqOvc_vyDPUj8yeFaf5xwBgc5bN0yGHyz-APJjOEth7nqit0HmRDv51Q6dx2gDJWX82eB4iQImQ=w1800-h1262-v0

1d054f27-34b7-4150-943c-525afc85ec7e

CHAPTER 5. PANE GRAPHICS 55

Figure 5.8: A summary of the Polaris encoding system. In addition to encoding data in the position of a mark on the plane, designers can also encode dimensions and measures of the data in visual properties of the marks such as color, size, orientation, and shape.

mark types have different constraints; for example, the shape of point mark can be varied

whereas an area mark’s shape cannot. Our approach is to have mark-dependent definitions for

each variable. An example is our definition of rotation: for point marks, this visual variable

refers to the rotation of the point’s external shape, whereas for line and area marks, it refers

to the rotation (or orientation) of the mark’s internal fill pattern. We consider this approach

preferable to approaches that ignore fundamental properties of mark classes, such as including

the rotation of a line or area mark as a visual variable.

The result of these goals is that our set of visual variables, unlike those of other researchers, is

more succinct then Bertin’s. In our encoding system, we have the following five variables:

CHAPTER 5. PANE GRAPHICS 56

POLARIS Bertin Wilkinson MacEachern Morrison Position Position Position Position Position Color Color Hue Hue Hue

## Value Saturation Saturation Saturation Brightness Value Value

## Shape Shape Shape Shape Shape Pattern Arrangement Arrangement

## Size Size Size Size Size Texture Grain Texture Texture

Rotation Orientation Orientation Orientation Orientation Rotation Blur Crispness Transparency Resolution Direction Transparency Speed (etc.)

Table 5.1: A summary of how the Polaris encoding system relates to the other major encoding systems.

Shape (pattern)

Size (granularity)

Rotation (orientation)

Figure 5.8 provides a summary of our encoding system with examples of encodings for each

variable, mark type, and data type. Table 5.1 illustrates how our encoding system relates to Bertin’s

and several other significant encoding systems. In the remainder of this chapter, we explain the

basic requirements of effective encodings and then provide detailed definitions for each of our visual

variables.

5.3.1 What is an effective encoding?

If one of the guiding principles in developing our encoding system is ”effectiveness” then it is

critical that we have some metrics for effectiveness. Unfortunately, as Wainer [75] explained in

The search for rules for effective graphical display, whether for the purpose of com-

munication, exploration or reconstitution, has been hampered by the lack of a cohesive

CHAPTER 5. PANE GRAPHICS 57

body of experimental evidence regarding the parameters of an efficacious graphical dis-

play. To some extent, existing evidence is diverse because of a lack of a coordinating

theoretical structure and an allied unified graphical vocabulary.

Still, since Wainer made this statement, considerable research has taken place and this area has

drawn more interest. Many of the guiding principles we do have for encoding data in visual vari-

ables have come from the studies and experiments involving low-level perception. Stevens’ [66]

research has shown that the perceived sensation elicited by different visual variables and the actual

intensity are not linearly related; his coefficients suggest a possible ranking of variable effective-

ness. Cleveland [19][20][22] has used theoretical and experimental results to rank how well people

can use different variables to compare quantitative variations. Mackinlay [48] extends these re-

sults to include an evaluation of different visual variables’ utility in encoding nominal information.

Kosslyn [45], a psychologist, and Tufte [72], a professor in statistics and design, have published

extensive guidelines in the use of visual variables based on studies of effective information design.

Other researchers, such as Wainer [75] (color), Brewer [11][12][13] (color), Healey [37][38] (color

and texture), and Caivano [17] (texture) have provided focused analysis of individual variables. We

base our selection of variables and evaluation of a variable’s utility for encoding different data types

(nominal and ordinal/quantitative) on our analysis of these results.

5.3.2 Position

Position is a perceptually dominant variable and is our most powerful tool for encoding data graph-

ically. Since spatial encoding is such an effective encoding, several techniques have been developed

to increase the number of dimensions that we can encode spatially [16]. Tables are a very useful

method for encoding multiple dimensions in two-dimensional space and have been extensively used

by analysts and statisticians. Our table algebra provides a powerful tool for specifying the encoding

of data in the position of marks.

5.3.3 Color

A key aspect of our treatment of color is that we consider color as a single composite variable. In

contrast, Bertin decomposes color into two distinct visual variables: color (hue and saturation) and

value. Wilkinson, Morrison, and MacEachern further decompose color into hue, saturation, and

value. If we examine the research done in constructing effective color encodings, these approaches

https://lh3.googleusercontent.com/notebooklm/AG60hOoRSrv3-XlrLTOsfLbnMuS7yqLO0Zrj0ytgpNe_wQB9hoDoRsPstGUXWl2Xk_ZzqvPYX4TW7rjIT69aM0dnDwG3gVo1rhR237ygvxHbSzN4TMhmE16fPbjZclak8ndVPwZ3Nn3VJw=w1722-h1892-v0

3bdeedc0-54d0-4ea6-88ba-8c1978e0e27e

CHAPTER 5. PANE GRAPHICS 58

Figure 5.9: The nominal and quantitative color encodings used in our implementation of the Po-laris formalism. In addition, we display example palettes from the major research projects that we leveraged in developing our encodings.

seem erroneous for two reasons: (1) they suggest that color can encode up to three variables simul-

taneously and (2) they suggest that only a single component of color should be varied to construct

CHAPTER 5. PANE GRAPHICS 59

an effective encoding.

Strategies for effective color usage [11][12][20][71] typically treat color as a single entity capa-

ble of encoding either a single dimension or, rarely, two dimensions of data. We are not aware of any

example of effective use of color to encode three dimensions. Some experimental research suggests

that even encoding two dimensions in color can be questionably effective [75]. Wilkinson [77],

who chooses to decompose color in his encoding system, acknowledges that the components of

color are highly configural, making separate use unwise for most applications. The only motivation

for decomposition, then, is to allow the user to explicitly select the component of color in which

to encode their data. This choice, however, requires design knowledge on the part of the user as to

what constitutes an effective encoding for each data type.

Furthermore, examining carefully designed palettes such as Cleveland’s five-color palette [20],

Travis’s ”six colors of maximum contrast” [71], or Brewer’s palettes [12], we find that the selected

encodings use colors varying in all three dimensions of HSV color space. When multiple variables

are encoded in color, such as in some of Brewer’s schemas for color usage [11], the use of the

dimensions of color are much more sophisticated than simply mapping one variable to hue and

another to saturation or value.

Some of these issues are a result of the nature of the HSV space used in computer visualization.

As Brewer [12] notes: “HLS, HSV, and HSB share the same problem that all saturated hues are

assigned equal lightness and saturation specifications regardless of their very different perceived

lightness and saturation”. Both MacEachern [47] and Bertin [7] also note the perceptual inconsis-

tencies in the structure of these spaces. The use of other color spaces such as Tek-HVC somewhat

alleviates these issues. However, color encoding remains a more complex issue than simply map-

ping data variables to components (perceptual or not) of color space.

Thus, we have chosen to expose color as a univariate or bivariate encoding in our encoding

system. We do not expose the “inner workings” of the encodings. The remainder of this section

outlines the considerations taken when encoding a nominal or quantitative variables using color, and

the issues that arise for bivariate encodings.

## Nominal encoding using color

Color is a very effective tool for encoding nominal information. In general, effective categorical

color encodings can be constructed using variations in hue that are separated in color space and

that correspond to different color categories (e.g. color names). However, to generate perceptually

uniform colors, palettes typically vary subtly in lightness and saturation as well [12]. In addition, if

https://lh3.googleusercontent.com/notebooklm/AG60hOrLoXJZuNNOwBdx_OrZtsmWaNbunR5pO9Z98Y6wbgE_DQVc0_-iBbWUfUlAYaMlV2qbOdJ4rs9Dcmsf6szXCvR_b17WA1Tz0Y1ZBEhoZrnQRntySCYxD3TUs-nGoBHStEAmHwcGTA=w1041-h163-v0

ace124f0-ffcd-4fa5-ac89-791fdb9e5432

CHAPTER 5. PANE GRAPHICS 60

Figure 5.10: The nominal shape encoding used in our implementation of the Polaris formalism and Cleveland’s shape palette that inspired our design..

color blindness is to be taken into consideration [71] or if it is necessary to construct a particularly

large [37] or small [11] palette then variations in luminance are recommended. Opinions on the

maximum length of a categorical palette vary from five to ten [76], and typically arguments are

based on our ability to name the distinct hues [10] or to internalize the legends [75].

In the implementation of our formalism we use two default palettes, one containing five colors

and the other containing 16. The first palette is based on Brewer’s [12] “5-class qualitative set1”

palette and Cleveland’s [20] five color palette and contains colors with equal perceived saturation

and lightness and with hues selected for maximal contrast. This palette is used for encoding fields

with small domains. The second palette contains sixteen colors and is derived from Brewer’s “11-

class qualitative Paired” palette which was extended to include five additional colors. Although the

research typically discourages palettes of this size, our experience has shown that user’s often need

to encode fields with larger domains, even if it is less effective, and it is important that our system

support this requirement. These palettes are illustrated in Figure 5.9.

Ordinal/quantitative encoding using color

Ordinal or quantitative color ramps typically hold hue constant and vary lightness and saturation

with the progression of the data variable. Variations in lightness are most effective in revealing the

structure of high frequency data whereas variations in saturation reveal the low frequency struc-

ture [55]. We include an ordinal and quantitative encoding based on Rogowitz [55] and Brewer [12]

in our implementation. Unlike Rogowitz, we do not currently analyze the data being encoded to

determine if it has low or high frequency variation, thus we have elected a color ramp that varies

in both lightness and saturation. In addition, our graphs are typically displayed on a white or light

gray background. Thus, we designed our color ramp so that the lightness never decreases below 65

percent. Our quantitative color ramp and two of Rogowitz’s color ramps are illustrated in Figure 5.9.

CHAPTER 5. PANE GRAPHICS 61

5.3.4 Shape (pattern)

Only point marks can vary their external shape because of the constraints on area and line marks’

spatial layout discussed in Section 5.1. In contrast, while point marks cannot vary their internal fill

pattern because they do not represent area on the plane, line and area marks can. Thus, we draw

together shape and pattern as a single visual variable,shape, using mark-dependent definitions. For

point marks, the shape variable refers to the external shape of the mark. For line and area marks, the

shape variable refers to the pattern used to fill the line or area. Similar mark-dependent definitions

exist in our system for size (granularity) and rotation (orientation). This mark dependence has

three benefits: (1) it simplifies our encoding system, (2) it avoids independence issues such as the

availability of both granularity and size for area marks and (3) it avoids nonsensical encodings (such

as having rotation and size for area or line marks in Wilkinson’s system).

We will first discuss the encoding of data in the shape of a point mark and then we will discuss

the encoding of data in the fill pattern of a line or area mark.

## Point shape

Point shape is most effectively used for encoding nominal information. Most research [7][22] dis-

courages its usage for ordinal or quantitative information since there is no logical ordering of shapes,

although some possible quantitative encodings have been suggested [77]. A key to an effective nom-

inal encoding for shape is selecting a set of shapes that tolerate overlap. Cleveland suggests that

this goal constrains us to symbols consisting of curves and lines with no solid parts and minimal

ink [19]. Our implementation of point shapes is inspired by Cleveland’s palettes and is illustrated

in Figure 5.10. We have simplified Cleveland’s palette to include more canonical shapes that can be

efficiently and easily rendered by the graphics system.

For point marks, the shape encoding can also be constructed by mapping data fields to text or

images. We have included these encodings as an aspect of point shape because they do in fact deter-

mine the shape of the point mark. However, the cognitive decoding process for these symbolizations

is different than for a palette of shapes.

## Fill pattern

Different researchers have exposed the encoding of data in fill pattern in different ways. In our

encoding system, users can encode data in three aspects of the fill: the shape of the pattern elements

(shape), the size of the pattern elements (size), and the orientation of the pattern elements (rotation).

CHAPTER 5. PANE GRAPHICS 62

Other researchers include these visual variables but also decompose the pattern into additional com-

ponents such as arrangement [50][47], directionality and density [17], or regularity [38]. We believe

our initial decomposition is justified as common usage of fill pattern often varies size, pattern, and

rotation elements independently. Further decomposition is questionable: constructing a set of dis-

criminable textures that do not also vary in value, color, or rotation is a monumental task, we rarely

utilize the full set of variables in our initial decomposition, and the perception of texture is very

complicated. Our current implementation of the Polaris formalism does not support encoding using

fill pattern although we intend to include it in future implementations.

5.3.5 Size (Granularity)

Similar to shape and pattern, size has a meaning that depends on the mark type. For point marks, the

size encoding determines the size of the actual mark. For area marks, the encoding determines the

size of the pattern elements used to fill the area. Finally, for line marks, a dual encoding exists that

can be used to control both the width of the line and the size of pattern elements. The remainder of

this section discusses the issues that arise when encoding data as the size of a point mark or pattern

## Ordinal encoding using size and granularity

In his experiments, Cleveland [19] found that size was one of the least effective variables for encod-

ing quantitative differences. However, size variation does have an implied ordering, making it very

effective for encoding ordinal information (and correspondingly ineffective for nominal informa-

tion). Polaris allows size to be used for either quantitative or ordinal information. When encoding

a quantitative domain as size, a linear map from the field domain to the area of the mark is created.

The minimum size is chosen so that all visual properties of a mark with the minimum size can be

perceived [45]. If an ordinal field is encoded as size, the domain needs to be small, at most four

or five values, so that the analyst can discriminate between different categories [7]. We construct

ordinal encodings by taking the range of possible sizes and equally dividing it to generate the largest

difference between sizes.

https://lh3.googleusercontent.com/notebooklm/AG60hOq9XqBr1fYa8uToWCYXsZ-RjoQBZ2vkMu1qQfUTY_OSLtKidJRGxYcIop0r1HmNveaT0uKQR6AlxQSiuPTEFO5gsCdZ5ORH7Wiyc2ZGlfvt29aoy4KxAUDWMPXMIEO-SeqUcrp6QA=w901-h168-v0

ae641f77-2f09-4444-bf23-18fb264970dc

https://lh3.googleusercontent.com/notebooklm/AG60hOook1_GYH5G4bkbE6Qxb9y503YsBpotxo112ZO3kvr2vzSuH2diMl_OCTtFUro3-p4QOuphGd04fHzVbLizbvoYh6lgmZKNoTdMnfU7jsAMwdDiTu05XFDrmUubnXJc4NXNmVYmqg=w901-h190-v0

f2b8414f-8e07-42f5-bb3d-dd0e74218302

CHAPTER 5. PANE GRAPHICS 63

5.3.6 Rotation (Orientation)

Again, rotation has a mark-dependent definition: Rotation for point marks, and the orientation of

pattern elements for line and area marks. A key principle in generating mappings of categori-

cal fields to orientation is that the orientation needs to vary by at least 30 degrees between cat-

egories [45], thus constraining the automatically generated mapping to a domain of at most six

categories. For quantitative fields, the orientation varies linearly with the domain of the field. For

rotation to be effective, the marks used to represent tuples must be asymmetric so that the degree of

rotation can be perceived. The amount of asymmetry affects the number of categories that can be

used for encoding.

5.4 Graphic notation for visual specifications

In Chapter 3, we introduced visual specifications and briefly described the components of a visual

specification. In the subsequent chapters, we have discussed in detail each of the major components.

We now briefly review each major component and its corresponding portion in a graphical notation

(inspired by Bertin’s notation for describing charts and diagrams [7]) for succinctly communicating

a Polaris specification.

The table structure: Two expressions in the table algebra, one each for the x- and y-axis,

define (1) the rows and columns of the table and (2) how data is spatially encoded within each

pane. Chapter 4 described the table algebra in detail.

Group: This portion of the specification identifies the level of detail within a pane and the

grouping of tuples into distinct lines and areas. Together the dimensions listed in the table

algebra and in the Group component uniquely identify the desired projection of the data cube

or aggregation of a relational database.

The mapping of data sources to layers:Multiple data sources may be combined within a

single Polaris visualization, with each source mapped to a separate layer. All layers share the

https://lh3.googleusercontent.com/notebooklm/AG60hOobaHgLxvBQHVF5wnloAZDYvHK708qdvpjJSZgjAp9Yc2hgGB_3Fr9r7TI61j-C_f7v29hCnfpbPF-Tcig6LDd_KJU172b6vjK2RIAtRZ6VYYoZW2sdma9dBnvAHo7Xkt6S4FnCnw=w901-h203-v0

bb5bd864-7cfe-4c5c-b289-13ab72ffce1d

https://lh3.googleusercontent.com/notebooklm/AG60hOpqKLXYE7BoQlKs-VPJl5L38Y8tytMQ6ElNQLZMzfBlPJnfOcRseaiYakF8Bxr3M3G_UtVsFIoxaQzFJwK-EMkCSMoXEedc2RXf8q4PhmkmYImtJqoDF4lRMhsgYmzDYm-Dbdig=w901-h602-v0

16ed24b6-41ba-4677-9e03-2a4c1305ba26

CHAPTER 5. PANE GRAPHICS 64

same table structure and are composited together back-to-front to generate the final visualiza-

The visual representation for tuples:Both the mark type (point, line, or area) and the retinal

attributes of each mark can be specified. While the current graphical notation encodes only

color, size, shape, and rotation, it is easily extended to include other retinal attributes such as

texture. Within the encoding rectangle, we can specify either a field name or a fixed value

(e.g., a specific color or a specific size in pixels). If an encoding is not used in a specification,

it will not be shown.

This notation will be used throughout the remainder of this dissertation to graphically commu-

nicate Polaris specifications.

5.5 Summary

In this chapter, we have described how the Polaris formalism allows analysts to flexibly construct

graphics by specifying the individual components of the graphics. The type of graphic in each pane

is specified using a taxonomy of graph types that structures the space of graphics into three families

by the type of fields assigned to their axes. Within each pane, tuples are mapped to graphical marks

and data is encoded in the visual properties of the marks. The Polaris formalism provides a succinct

set of visual variables that analysts can use to encode data: position, color, size, shape, and rotation.

This encoding system was designed to be understandable and used effectively without extensive

design knowledge.

## Generating Queries

A key advantage of our formalism is that it can be compiled into not only the drawing commands

necessary to generate the display but also into the queries necessary to retrieve the tuples to be

visualized. In this chapter, we explain in detail how specifications are compiled into a set of efficient

database queries.

6.1 Overview

The first aspect of generating a visualization from a visual specification is to retrieve the data from

the database. Within the Polaris formalism, every valid visual specification can be compiled into a

set of data source specific queries. The types of queries that are generated vary depending on the

type of database being visualized. For relational databases, SQL queries are generated to select,

group, and aggregate tuples from a denormalized relation or view. For multidimensional databases,

multidimensional queries (e.g., Microsoft MDX or SQL-99) are generated to retrieve projections

and slices from a data cube.

Conceptually, each pane in the visualization corresponds to a distinct query. We will first explain

how both SQL and MDX queries can be generated for a single pane, and then we will introduce

several optimizations to reduce the number of queries that are sent to the server.

6.2 Generating SQL queries

To generate a SQL query to retrieve the relations for a single pane, we must consider several things:

which fields to retrieve, which tuples to retrieve, are we aggregating the data, and if so, what level

https://lh3.googleusercontent.com/notebooklm/AG60hOr576cZfAoc_tbQobzD3F_ibrp70YSGsKLzGqconxT4YFXYQk91Rm0Pr2IerUPNvd0cjbbBce1sAGNgxFSvvSsP-BKHzM4EkQ5mS4VZ6Z5mqIhPiN19ooABWyI5wRxa_7KFkqiM2Q=w1745-h1164-v0

2cd86259-39af-430f-8743-2e5b820b0135

CHAPTER 6. GENERATING QUERIES 66

Figure 6.1: Conceptually, each pane in a visualization corresponds to a query. In this figure, the SQL and MDX queries that would generate the correct tuples for the highlighted pane are shown. The algorithm for generating these queries is outlined in Sections 6.2 and 6.3.

of detail are we aggregating to?

We will first walk through an example and then present a pseudo-code algorithm for generating

a SQL query for a single pane. Consider the highlighted pane in Figure 6.1. Which tuples and

fields from the database are displayed in this pane is determined by several aspects of the visual

specification.

The table expressions define both the spatial encodings within the pane and a selection criteria

on the tuples to be displayed. However, which tuples and fields are retrieved is only affected by

a limited portion of each table expression. Specifically, we need only concern ourselves with the

p-tuples in the normalized set form (see Chapter 4) of the axis expressions that define the specific

row and column containing the pane we are querying for. In our example, the applicable p-tuples

are “(ProductType:Tea, field:SUMSales)” and “(Quarter:Qtr4, field:SUMProfit)”. The former p-

tuple defines the fourth row of the table and the latter p-tuple defines the fourth column of the table.

These p-tuples impose the requirement that the retrieved tuples include the fieldsSUM Profit and

CHAPTER 6. GENERATING QUERIES 67

SUM Sales, as these fields are required in order to determine the position of the graphic marks

within the pane. Furthermore, these p-tuples impose the requirement that we only retrieve tuples for

which the value ofProductTypeis “Tea” and the value ofQuarter is “Qtr4”.

The visual encodings included in the visual specification (color, size, shape, etc.) affect which

fields we retrieve from the database. All fields that are encoded as visual properties of the graphic

marks must be present in the retrieved tuples. In the given example, theMark tfield is encoded in

the shape of the point marks. Thus, we includeMarket in the SELECT clause of the query.

When visualizing a relational database, the tuples are typically grouped and aggregated before

they are retrieved from the database. The “Group” component, the table expression p-tuples, and

the visual encodings all affect the grouping of the tuples before aggregation: every categorical

field in the “Group” component, the p-tuples, or the visual encodings is included in the GROUP

BY clause of the query. In our example visualization, the “Group” component includes theStat

field, the applicable p-tuples include the categorical fieldsQuarterandProductType, and the visual

encodings include the categorical fieldMarket, so our GROUP BY clause will include all four fields.

It is also possible to specify filters within a visual specification. These filters can be on cate-

gorical fields and unaggregated measures, in which case they are included in the WHERE clause of

the query, or they can be on aggregated measures, in which case they are included in the HAVING

clause of the query. Our example does not contain any filters.

Our final consideration is drawing order. The “Sort order” component of the visual specification

imposes a desired ordering on the retrieved tuples. If the fields included in the “Sort order” com-

ponents have alphabetically or numerically ordered domains, then the ordering can be performed in

the query using the ORDER BY clause. In many cases, however, the domain of the field may not

be ordered alphabetically, or may have been reordered by the user. In this case, the ordering must

be performed on the client.

The query for our example pane is shown in Figure 6.1. The following pseudo-code would

generate the correct SQL query:

CHAPTER 6. GENERATING QUERIES 68

Algorithm: Generate-SQL-Query

### 1. Add all fields from the visual encodings to the SELECT

Add quantitative fields from the p-tuples to the SELECT

### 2. Add selection criteria from p-tuples to the WHERE

### 3. Add categorical fields from the x- and y-axis p-tuples to the GROUP BY

## Add categorical fields from all encodings to the GROUP BY

Add fields from the “Group” to the GROUP BY

4. foreachfield with a filter

if (field is categorical)

Add filter to WHERE as‘‘field IN (filter)’’

else if (field is an unaggregated measure)

Add filter to WHERE as‘‘(field >= min and field <= max)’’

Add filter to HAVING as‘‘(field >= min and field <= max)’’

The above algorithm, however, is not complete. There are three complexities which affect the

final query: querying without aggregation, querying for derived fields, and filtering with lines and

areas. We discuss each in turn.

6.2.1 Querying without aggregation

It is possible when visualizing a relational database to have the tuples be retrieved without aggre-

gating them first. There is an “Aggregate” flag in the visual specification; if this flag is set to “false”

then the tuples will not be aggregated and there will be no GROUP BY or HAVING clause in

the generated query. This is implemented by including additional conditions in lines 3 and 4 of the

query algorithm. An example of a visualization that uses this feature is Figure 5.1: the tuples, which

define the outlines of each county, are not aggregated in the query and the “Group” component only

affects the grouping of the tuples into distinct polygons.

6.2.2 Derived fields

In addition to specifying visualizations of the existing fields of a database, the user can define

additional “derived fields” and use those fields in the specifications. A derived field is defined by

https://lh3.googleusercontent.com/notebooklm/AG60hOpKS7dBrRVtHnko6wLytBZFXtfxy1KrYisc2YJJn4ozkke7_xTwzIZyS0A9qUZWufT9Q6jBpfr9o_QE0GD3P3FFvQLdl_XzK5BeBlmvT4daugLyPHDoMeQ5EQQt7DFOw81yqrWW=w1761-h1285-v0

d9486503-89e5-4bf6-816f-9ba32773c1ab

CHAPTER 6. GENERATING QUERIES 69

Figure 6.2: When the tuples within a pane are being displayed using line marks we need to refine our query. The default query generating algorithm applies the filtering criteria to all tuples and thus would retrieve only a subset of the points along a filtered line, as shown here. The correct approach is to retrieveall pointson a line whenany pointmeets the filtering criteria.

specifying a formula (using SQL) that defines that field. For example, a new quantitative measure

could be defined as:

Margin = Price-Cost

Derived fields can be incorporated into the query in one of two ways:

A new viewcan be defined on the server that includes all of the derived fields as additional

fields. Once the view is created we can direct our queries to that view and treat the derived

fields exactly the same as any of the other database fields.

Whenever a derived field is included in the query, the field’s formula, rather than its name, is

inserted into the query.

CHAPTER 6. GENERATING QUERIES 70

6.2.3 Filtering with lines and areas

Typically, filtering is fairly simple: the filters are simply included in the WHERE or HAVING clause

of the query. However, when a pane is displaying data using a line (or area) mark, we must construct

a slightly different query. The problem is illustrated in Figure 6.2: when we filter individual points

rather than complete lines, we may retrieve only a portion of some lines (those points that meet the

filter conditions) and the result is an incorrectly rendered line. The proper resolution to this problem

is to always retrieveall pointson a line whenany pointon that line meets the filter criteria.

This resolution is achieved by restructuring our query in this situation. We will first construct

a sub-query to identify the lines to be retrieved and then select tuples based on the results of this

sub-query. As we discussed in Section 5.1.2, each distinct set of values for the columns listed in

the “Group” component of the specification identifies a unique line. Thus, our sub-query should

identify the lines to be retrieved by selecting the values for those columns where any tuple meets

the filter condition. We can then join this sub-query with the original database to retrieve the correct

tuples. Letg1, . . . , gn be the contents of the “Group” component andselect-clause, where-clause,

group-by-clause, andhaving-clausebe the components of our original query. Then the query we

want to execute is:

SELECTselect-clause

FROM databaseAS A,

(SELECT DISTINCTg1, . . . , gn

## FROM database

WHEREwhere-clause

GROUP BYgroup-by-clause

HAVING having-clause) AS B

WHEREA.g1 = B.g1 AND ... A.gn = B.gn

GROUP BYgroup-by-clause

Figure 6.2 contains the correct query for our example visualization.

6.3 Generating MDX queries

To generate a query on a multidimensional data source, we must determine the dimensions to be

retrieved, the measures to be retrieved, and the slicing to be performed. The general form of the

MDX queries that we generate is:

CHAPTER 6. GENERATING QUERIES 71

SELECTmeasuresON COLUMNS,

dimensionsON ROWS

## WHEREslices

We will consider each component of the query separately, using the same example from Fig-

6.3.1 Measures

The measures included in the query are determined by the visual encodings: all measures used

in any visual encoding are included as columns of the result set. In our example,SumProfitand

SumSalesdefine the spatial encodings within the pane and thus are included in the query:

SELECT{[Measures].[SumProfit], [Measures].[SumSales]} ON COLUMNS

6.3.2 Dimensions

Determining the dimensions to include in the query is somewhat trickier. In the relational case, all

categorical fields are distinct entities and they can all be included in a single query. With a multidi-

mensional cube, it is possible that different categorical fields refer to levels of the same dimension

hierarchy. In MDX, for any given dimension hierarchy we must only query for a single level. Thus,

for each dimension we must identify the most-detailed level requested and include only that level in

the query. In our example, only one level is included from the “Product”(ProductType) and “Time”

(Quarter) dimensions. However, two levels are included from the “Location” dimension:Market

andState. We will only include the more detailed level (State) in our query; MDX will automati-

cally retrieve all ancestor levels. The following algorithm determines the dimensions and levels to

be included in the query:

CHAPTER 6. GENERATING QUERIES 72

Algorithm: Determine-Dimension-Levels

1. for eachdimensionin cube

2. levels= ∅ 3. if (the p-tuples contain any levels ofdimension)

## Add the levels tolevels

4. if (encodings contain any levels ofdimension)

## Add levels tolevels

5. if (”Group” contains any levels ofdimension)

## Add levels tolevels

6. if (levels!= ∅) Add dimensionto query

## Add most detailed level inlevelsto query

Once we have identified the set of dimensions and levels to be included in the requested projec-

tion, we must form a “member statement” for each dimension. The member statement must be an

MDX statement that will generate the list of the members to be retrieved for that dimension (and

level). The resulting member statements will be combined using the MDX “CROSSJOIN” operator

to form the ROW clause of our query. If the given dimension is not filtered, then we can simply use

the MDX “.Members” operator (e.g.,[Location].[State].Members) in the query. If the dimension is

filtered, then we must list the specific members to be retrieved using the MDX set syntax. For our

example query. we generate:

SELECT. . .

CROSSJOIN(Location.State.Members,

CROSSJOIN({Product.ProductType.Tea}, {Time.Quarter.Qtr1})) ON ROWS

6.3.3 Filtering and slicing

Filtering on dimensions included in the result set of the query was addressed in the last section.

However, there are two additional types of filtering that can be performed:

filtering on the measures, and

https://lh3.googleusercontent.com/notebooklm/AG60hOoWTeNOyZCRflhNdAlzB3lKA5dSARwkS9gke4fS7T8LLg2mVAt3wkS-_haOz_85WrqMnDutZYIqm0grom_UG70hQsztKLUMd8GYtOMZVIkcts0ItD3CG3dZEKv7YLp8RTYLojWhHQ=w1737-h896-v0

1e5e9ee0-e428-429b-aed3-d650c0ded7a5

CHAPTER 6. GENERATING QUERIES 73

Figure 6.3: When multiple panes are at the same level of detail, we can query for all of the panes in a single query and then partition the results locally into sets corresponding to individual panes.

filtering on dimensions not included in the result set.

To filter on a measure, we simply enclose the ROW clause in an MDX “FILTER” command

that includes all of the measure filters. For example, if we want to filter our example to include only

cells whereSumProfitis less than 60.0, we would change our ROW clause to:

SELECT. . .

FILTER(CROSSJOIN(Location.State.Members,

CROSSJOIN({Product.ProductType.Tea}, {Time.Quarter.Qtr1})), ’SumProfit < 60.0’) ON ROWS

To filter on dimensions that are not included in the query, we use the slicing features of MDX.

In the same manner that we generated the set of members for the ROW clause, we generate a set to

be included in the WHERE clause of the query.

Figure 6.1 shows the final MDX query that would be generated for our example pane.

6.4 Reducing the number of queries

Thus far, the algorithms we have presented have generated a single SQL or MDX query per pane.

However, typically the panes in a visualization are related and it is inefficient to issue one query per

https://lh3.googleusercontent.com/notebooklm/AG60hOrNrG1JsbV3NulS5tCVwhHFFoXqmUib1Owam59dyRq7e9Cc_O_mLnhPZjpWSAL_LXnqotnR0tTQ1YTG0Yv7W5D1c3tLWHBCHXsasEPmGKP5Bx8ClL0mAuP9_txcSQPy9Rc-QME16Q=w1720-h511-v0

a03ff245-d310-4a02-ad31-3b7c7fa0d854

CHAPTER 6. GENERATING QUERIES 74

Figure 6.4: To generate some visualizations, we will have to issue multiple queries even after con-solidating individual pane queries. In this visualization, the top four rows of the table are at a different level of detail then the bottom four rows.

pane. If we make the reasonable assumption that the client is able to partition the results of a query,

we are faced with the problem: “Given the ability to partition relations locally, how can we reduce

the number of queries we send to the database server?”.

The solution is to issue one query for every set of panes that is at the same level of detail. We

define “level of detail” as either the GROUP BY clause of a SQL statement or the projection that

results from a data cube query. The results of this single query can then be partitioned locally into

result sets corresponding to individual panes. Figure 6.3 illustrates a single query and partition-

ing that could be performed to retrieve all of the data for the example visualization introduced in

Figure 6.1.

6.4.1 Generating a single query for multiple panes

Given a set of panes at the same level of detail, we can generate the query for all of the panes from

the query for a single pane by performing the following steps:

Adding all categorical fields in either p-tuple to the SELECT, GROUP BY, and ORDER BY

clauses of the query.

Removing all WHERE clauses that were generated from the p-tuples; these selection criteria

are performed by the local partitioning.

CHAPTER 6. GENERATING QUERIES 75

6.4.2 Identifying panes that can share a query

Using this approach, we can often generate a single query for an entire visualization, as in Figure 6.3.

However, some visualizations will require> 1 query. Figure 6.4 depicts a visualization that requires

at least two SQL queries to retrieve all of the tuples to be displayed.

So, given a visual specification, how can we determine how many distinct queries we will need

to issue to generate the visualization? The key to answering this question is the concatenation

operator. The concatenation operator is the only one of our four algebraic operators (nest, cross,

concatenate, and dot) that can produce adjacent panes with differing projections or level of detail.

Nest, cross, and dot include all of the input fields in each output p-tuple; concatenate does not.

Thus, we can identify the number of queries to be issued, and which panes correspond to each

query, by algebraically manipulating our table expressions. First, we reduce each axis expression

to a sum-of-termsform1. Then, we consider each possible combination of one term from each

axis. Each combination of terms will correspond to a query that needs to be issued. In the example

shown in Figure 6.4, the y-axis has two terms (ProductTypeandMarket) and the x-axis has one

term (Quarter). Thus, we need to issue two queries: one for the panes corresponding to the terms

(ProductType, Quarter) and another for the panes corresponding to the terms (Market, Quarter).

The query for each set of panes can be generated as in Section 6.4.1.

6.5 Performance and scalability

In designing the Polaris formalism, we chose to make our algebra similar to the relational algebra.

This was done so that, when generating a visualization from a specification, we can allocate as much

of the data processing as possible to the database server, which is designed to efficiently manipulate

the large amounts of data in the data warehouse. In our current implementation, all of the data

processing is performed by the database server except for the partitioning of query results into sets

corresponding to individual panes and the sorting of tuples within a pane.

In our experience, the performance of these local transformations is typically dominated by the

querying processing time on the database server. This occurs for two reasons: (1) the retrieved

number of tuples is typically orders of magnitude smaller than the database size and (2) the parti-

tioning and sorting transformations are simple compared to the execution of a SQL or MDX query.

The retrieved number of tuples is typically much smaller than the database size because of the way

1The algebraic manipulation of axis expressions into sum-of-terms form is done using a less constrained set of alge-braic properties, as discussed in Section 4.4

CHAPTER 6. GENERATING QUERIES 76

people perform analysis. When analysts start exploring a large database, they typically abstract or

aggregate the data considerably to generate an overview display. As they explore further, they will

drill-down in areas of interest, but this drill-down is accompanied by the application of filters which

keep the number of tuples relatively consistent. Even with very large databases, analysts rarely want

to graph more than several tens of thousands of tuples.

We have used our implementation of the Polaris formalism to interactively explore databases

and data cubes ranging in size from 5,000 tuples to over 78 million tuples. At an overview level,

most queries have executed in under a second. However, some of the more detailed queries we have

generated have taken from tens of seconds to several minutes to execute. This performance, though,

is affected by many factors: the design of the database and indices, the processing capacity of the

client and server machines, the network latency, and available memory. Our initial experiments have

shown that considerable performance benefits can be gained by careful optimization of the database

design and indices.

6.6 Summary

In this chapter we have outlined the compilation of specifications in the Polaris formalism into

database queries. The purpose of our formalism and implementation is not to be a graphical query

generator; instead, our intent is to design and build visualization system where the database query

is implicit in the specification of the visualization. As a result, there are many relational and mul-

tidimensional queries that can not be generated by our system. The most obvious limitation in the

queries generated by the interpreters is that the formalism does not support explicitly specifying

joins between multiple relational tables. This design decision was a result of the decision to design

a formalism for analysis. For the purposes of analysis, it is reasonable to assume the data is stored

in a relational database with a star or snowflake schema, or that a view has been generated that

imposes a star or snowflake schema on top of the actual schema.

## Interactive Analysis and Exploration

In this chapter, we introduce the first of two systems we have built on top of the Polaris formalism:

the Polaris interface, an interface for exploring large multidimensional databases that extends the

well-knownPivot Table[49] interface.

7.1 Overview

As we discussed in Chapter 1, large databases have become common in a variety of applications. A

major challenge with these databases is to extract meaning from the data they contain: to discover

structure, find patterns, and derive causal relationships. The analysis and exploration necessary to

uncover this hidden information places significant demands on the human-computer interfaces to

these databases. The exploratory analysis process is one of hypothesis, experiment, and discovery.

The path of exploration is unpredictable, and analysts need to be able to rapidly change both what

data they are viewing and how they are viewing that data.

Perhaps the most popular interface to multidimensional databases is the Pivot Table [49]. Pivot

Tables allow the data cube to be rotated, or pivoted, so that different dimensions of the dataset

may be encoded as rows or columns of the table. The remaining dimensions are aggregated and

displayed as numbers in the cells of the table. Cross-tabulations and summaries are then added to

the resulting table of numbers. Finally, graphs may be generated from the resulting tables.

The Polaris interface for the exploration of multidimensional databases extends the Pivot Table

interface to directly generate a rich, expressive set of graphical displays. Polaris builds tables using

the formalism described in Chapters 3 through 6. Each table consists of layers and panes, and each

pane may be a different graphic. The use of tables to organize multiple graphs on a display is a

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 78

technique often used by statisticians in their analysis of data [6][20][72].

The Polaris interface is simple and expressive because it is built on top of the Polaris formalism.

We interpret the state of the interface as a visual specification of the analysis task and automatically

compile it into data and graphical transformations. This architecture allows us to combine statistical

analysis and visualization. Furthermore, every intermediate specification that can be created in the

visual language is valid and can be interpreted to create visualizations. Therefore, analysts can

incrementally construct complex queries, receiving visual feedback as they assemble and alter the

specifications.

7.2 The interface

To effectively support the analysis process in large multidimensional databases, an analysis tool

must meet several demands. First, the tool must support the unpredictability of the analysis process

and, as stated above, allow the analyst to rapidly change what data they are viewing and how they

are viewing that data. Furthermore, the analysis tool must be able to generate displays suited to

the many different analysis tasks, such as discovering correlations and locating outliers. Finally,

the databases are typically quite large and analysts need to be able to simultaneously view many

dimensions as well as many tuples (in the display generated by the tool).

Polaris addresses these demands by providing an interface for rapidly and incrementally gener-

ating table-based displays. Figure 7.1 shows the user interface presented by Polaris when connected

to a flat relational database. In this example, the analyst has constructed a matrix of scatterplots

showing sales versus profit for different product types in different quarters. The primary interaction

technique in both versions of the interface is to drag-and-drop fields from the database schema onto

shelves throughout the display.

Figure 7.2 illustrates the Polaris interface as it appears when connected to a hierarchical data

cube. When connected to a hierarchical data cube, a number of additional interface features are

added to expose the hierarchical structure. For example, when a dimension level is placed on a

shelf, it includes a pulldown menu that can be used to drill-down or up in the dimension hierarchy.

A given configuration of fields on shelves defines a visual specification in the Polaris formalism;

displays are generated by interpreting this specification. The user can rapidly change what data is

being viewed or how it is being displayed by simply changing the contents of the shelves. The users

can also interactively edit the visual mappings that are generated by manipulating the automatically

generated legends.

https://lh3.googleusercontent.com/notebooklm/AG60hOpRSJo8QtvMyv3i5YrUyb9OE2KO_X1AZGK-Z5r0VmLrV0SqC9tHyKS6b4JaUM3qJQFiVuW0_E3jZlSonAjrotGQw3JCAL5aDzaX2YgnVqlSQS2qKF-tSoHF5LEh_vbPuxi0wLzr=w1804-h1191-v0

d3a767b6-3a83-4702-ab65-877c08f5bcf5

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 79

Figure 7.1: The Polaris interface when connected to a flat relational database. Analysts construct table-based displays of data by dragging fields from the database schema onto shelves throughout the display. A given configuration of fields on shelves is interpreted as a visual specification in the Polaris formalism.

The translation of shelf contents into a visual specification is performed as follows:

Table configuration: The Polaris interface generates a subset of the possible Polaris ex-

pressions. The contents of each axis shelf is an ordered list of database field names and

constraints in the interface ensure that all nominal and ordinal fields precede all quantitative

fields in this list. For each axis shelf, this list is transformed into an expression of the form

(O1× ...×On)×(Q1 + ...+Qm). In addition, if any two adjacent categorical fields represent

levels of the same dimension then the cross (×) operator between them is replaced with a dot

(.) operator.

Visual encodings:For each visual encoding supported by a given mark type there is a shelf

that can contain a single field. The contents of these shelves define the visual encodings

within the visualization. Mappings from data values to visual properties are automatically

https://lh3.googleusercontent.com/notebooklm/AG60hOqdyBZ4N4KHMsn3qAy8kJ-_q9dIZS7kKkrksGy099JcvAEeZOG3diJ4WxkS9IRCoxULEQa-P24h8fKVwog9H-nfsYVd7qYZnqcGrg-2_8j1xyywmDMt8SVqTylet__0E0YuyVM4=w1781-h1119-v0

79f68791-5ec2-4dc3-95f8-48c2e8e3ae23

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 80

Figure 7.2: The Polaris interface when connected to a hierarchical data cube. The enhancements to the interface to expose and support hierarchical dimensions are shown in blue.

constructed by the interpreter.

Data transformations: There are two shelves labeled “Group in panes by” and “Sort in panes

by”. The contents of these shelves define the “Group” and “Sort Order” components of the

visual specification.

7.3 Related work

The related work to the Polaris interface can be divided into two categories: table-based data dis-

plays and database exploration tools.

7.3.1 Table-based displays

The first area of related work to the Polaris interface is visualization systems that use table-based

displays. Static table displays, such as scatterplot matrices [36] and Trellis [2] displays, have been

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 81

used extensively in statistical data analysis. Recently, several interactive table displays have been

developed.

Pivot Tables [49] allow analysts to explore different projections of large multidimensional datasets

by interactively specifying assignments of fields to the table axes, but are limited to text-based dis-

plays. Systems such as the Table Lens [53] and the FOCUS [63] visualization system provide table

displays that present data in a relational table view, using simple graphics in the cells to communi-

cate quantitative values.

Ed Chi’s “Spreadsheet Approach to Visualization” [18] is a highly-interactive table-based vi-

sualization system. Much like a traditional spreadsheet, the user can define formulas for each cell

with inter-dependencies between the different cells. However, unlike traditional spreadsheets, the

contents of each cell can be a visualization, enabling the user to construct comparative table-based

visualizations.

7.3.2 Database exploration tools

The second area of related work is visual query and database exploration tools. Projects such as

VQE [25], Visage [58], DEVise [52], and DataSplash [79] have focused on developing visual-

ization environments that directly support interactive database exploration through visual queries.

Users can construct queries and visualizations directly through their interactions with the visual-

ization system interface. These systems have flexible mechanisms for mapping query results to

graphs, and all of the systems support mapping database tuples to retinal properties of the marks

in the graphs. However, none of these systems leverages table-based organizations of their visual-

izations. Furthermore, of these systems, only DataSplash provides built-in support for interactively

navigating through and exploring data at different levels of detail. However, the underlying hierar-

chical structure must be created by the analyst during the visualization process; Polaris leverages

the hierarchical structure that is already stored in the data warehouse.

Other existing systems, such as XmdvTool [59], Spotfire [64], and XGobi [14] have taken the

approach of providing a set of predefined visualizations, such as scatterplots and parallel coordi-

nates, for exploring multidimensional data sets. These views are augmented with extensive inter-

action techniques, (e.g., brushing and zooming) that can be used to refine the queries. We feel that

this approach is much more limiting than providing the user with a set of building blocks that can

be used to interactively construct and refine a wide range of displays to suit an analysis task. Of

these systems, only XmdvTool supports the exploration of hierarchically structured data. XmdvTool

has been augmented with structure-based brushes [59] that allow the user to control the display’s

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 82

global level of detail (based on a hierarchical clustering of the data) and to brush tuples based on

their proximity within the hierarchical structure. Again, this approach limits the user, in this case to

viewing a single hierarchical structuring of the data and a single ordering of that hierarchy to make

proximity meaningful. Polaris supports both the simultaneous exploration of multiple hierarchies

(derived from semantic meaning or algorithmic analysis) and the ability to reorder the hierarchy as

Another significant database visualization system is VisDB [44], which focuses on displaying

as many data items as possible to provide feedback as users refine their queries. This system even

displays tuples that do not meet the query and indicates their “distance” from the query criteria

using spatial encodings and color. This approach helps the user avoid missing important data points

just outside of the selected query parameters. In contrast, Polaris takes advantage of the hierarchical

structure of the warehouse to provide extensive ability to drill down and roll up data, allowing

the analyst to get a complete overview of the data set before focusing on detailed portions of the

7.4 Data transformations and visual queries

The ability to rapidly change the table configuration, type of graphic, and visual encodings used to

visualize a data set is necessary for interactive exploration. However, it is not sufficient: additional

interactivity is needed. The resulting display must be manipulable. The analysts must be able to sort,

filter, and transform the data to uncover useful relationships and information, and then they must be

able to form ad-hoc groupings and partitions that reflect this newly uncovered information [6].

In this section, we describe five interaction techniques Polaris provides to support analysis

within the resulting visualizations: deriving additional fields, sorting and filtering, brushing and

tooltips, hierarchical analysis, and undo and redo.

7.4.1 Deriving additional fields

While analyzing data, one of the most important interactions needed is the ability to derive addi-

tional fields from the original data. Typically, these generated fields are aggregates or statistical

summaries. Polaris currently provides five methods for deriving additional fields: simple aggrega-

tion of quantitative measures, counting of distinct values in ordinal dimensions, discrete partitioning

of quantitative measures, ad hoc grouping within ordinal dimensions, and threshold aggregation.

Adding derived fields on the fly is necessary as part of the exploration and analysis process. As

https://lh3.googleusercontent.com/notebooklm/AG60hOpuk-kcVcUg4Cx2e5DYFNxyNnpVV2oyL1uNZrXldX39hdYxAEA8rj5XhOkuCNWa7PAPNGsoxjpas1JiHg5OnEA64LbFwBmVSOCG2a49ykywg4vi5FgSeyobuuVtBTLL0w0PIZukNQ=w1500-h1170-v0

b5458a62-365c-491a-b3c9-7a865f70021a

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 83

Figure 7.3: The interface for changing the aggregation function applied to a single quantitative field. To change the aggregate, the user simply right-clicks on the field name and selects a new aggregate.

the analyst explores the data, relationships within the data are discovered and groupings can be used

to encode and reflect this discovered information. Furthermore, aggregations and statistics can be

used to hide uninteresting data and to reduce large data sets in size so that they are tractable for

understanding and analysis.

## Simple aggregation

Simple aggregation refers to operations such as summation, average, minimum, and maximum that

are applied to a single quantitative field. A default aggregation operation is applied to a quantitative

measure when it is dragged to one of the x- or y-axis (or layer) shelves and aggregation is enabled.

The user can change which aggregation function is applied by right-clicking on the field and choos-

ing a different aggregation function from the context menu, as shown in Figure 7.3. Polaris can be

easily extended to provide any statistical aggregate that can be generated from relational data.

https://lh3.googleusercontent.com/notebooklm/AG60hOoFKzfPfHFRN9dp-7a2PSXNRGvB_KkX2RSSocVf37oAyJyLVZANFgZ_7NvtDA39ryr9unxx5ObXoiQ5B77YyVYPsHoVMURJQ3LeE0L9xcjy9ZgIMd_ZPDdadcyxnEG0sgiPtXszpA=w1800-h674-v0

8e41aeb1-8c9a-4fc8-943b-79ab4cc87a30

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 84

Figure 7.4: An example of discrete partitioning of a quantitative field. On the left is a traditional scatterplot ofCOGS(cost-of-goods-sold) versusProfit. On the right, the analyst has binnedProfit into discrete bins of size 30.

## Counting of ordinal dimensions

Counting of ordinal dimensions refers to the counting of distinct values for an ordinal field within

the data set. This aggregation function can be applied to an ordinal field by right-clicking on the

field and choosing the CNT (count) aggregation function. Unlike simple aggregation, applying the

count operator will change the field type (to quantitative) and thus change the table configuration

and graph type in each pane.

## Discrete partitioning

Discrete partitioning is used to discretize a continuous domain. Polaris provides two discretization

methods: binning and partitioning. Binning allows the analyst to specify a regular bin size in which

to aggregate the data; binning will not change the graph type since the resulting derived field is also

quantitative, just at the specified granularity. Figure 7.4 shows two scatterplots: one before binning

theProfit field and the other after binningProfit into bins of size 30.

Partitioning allows the user to individually specify the size and name of each bin. Partitioning

of a quantitative field will result in an ordinal field, thus changing the graph types and table configu-

ration. Binning is useful for creating graphs such as histograms, in which there are many regularly-

sized bins, while partitioning is useful for encoding additional categorizations into the data, either

ad-hoc or derived from known domain information. Both can be applied by right-clicking on the

https://lh3.googleusercontent.com/notebooklm/AG60hOofyaiZzj7wMDNOMfPuLvQJtDLXACkte4xyo5WOL5zkqIUfnPBGiJ4ULaHjNX6gC9JSPbNp2pc1M614IvS4rXraiOyMaaGYAIsHKFRDYcAYmBzZ1OPTAANnvkaqOrBfreFCtb-azg=w1500-h1168-v0

347df518-d9dc-4ad5-8659-33084e988cb9

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 85

Figure 7.5: An example of generating ad hoc groups for a categorical field. The user has selected the “Ad hoc Grouping...” option from the context menu forStateand has formed a group containing California, Connecticut, Florida, and Iowa, and named “AD HOC GROUP”.

field name and choosing either the “Bin by...” or “Partition...” option.

## Ad hoc grouping

Ad hoc grouping is the ordinal version of quantitative partitioning, where the user can choose to

group together different ordinal values for the purpose of display and data transformations. For

example, a user may choose to group California and Florida together into an “Orange provider”

partition. This type of arbitrary grouping and aggregation is powerful since it allows the analyst

to add his own domain knowledge to the analysis, and to change the groupings as the exploration

uncovers additional patterns. The user can apply ad hoc grouping by right-clicking on the field name

and choosing the “Ad hoc grouping...” option. Figure 7.5 depicts the ad-hoc grouping interface.

This transformation derives an ordinal field from an ordinal field and thus the graph type does not

https://lh3.googleusercontent.com/notebooklm/AG60hOql50oCT9mSTy1Q3yNf-1RvXniQtv382FqkTsf_1bm-kl0hOwK0qdQfyzGiPtf0-DV-LTRQaq7irjPgfT_H_BTpM8ZdzNV1ctxNg9XJrskmfflgqga9qxZVtIDbat3rDyavGHaW2A=w1500-h1171-v0

b495aea0-ad30-468f-928b-d09efdd44338

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 86

Figure 7.6: An example of filtering and sorting an ordinal domain. The check boxes are used to indicate which domain values to exclude (by unchecking the value). The analyst can also drag a domain value to reorder the domain, as is being done for California in the figure.

## Threshold aggregation

Threshold aggregation is the last type of derived field we support, and it differs from the other

derived fields since it is derived from two source fields: an ordinal field and a quantitative field. If

the quantitative field is less than a certain threshold value for any values of an ordinal field, those

values are aggregated together to form an “Other” category. This transformation allows the user

to specify threshold values below which the data is considered uninteresting. One challenge in

supporting threshold aggregation is that it can require two aggregation passes if the quantitative

field desired is itself a derived field (e.g., the average of the quantitative profit field).

https://lh3.googleusercontent.com/notebooklm/AG60hOpiJxg4WwFE6l3zxz2sTRNJ91ni6FHGn4y8ln1fmmkuBD2eFaa6Zxnxql5oeb_fjpx0Iv9H-xp_UVTUp2C4Hx4HerQUGcs6dWVCCOLHBUc15tsIsRUfnk8iZL1E3vlqLXH7Xgvn6g=w1500-h1170-v0

95688859-9d9a-4647-a7cb-d2b82bc800e2

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 87

Figure 7.7: An example of filtering a quantitative domain. The user simply drags the ends of the highlighted region to indicate which values to include/exclude from the visualization.

7.4.2 Sorting and filtering

Sorting and filtering data play a key role in analysis. It is often only by reordering rows and columns

and filtering out unimportant information that patterns and trends in the data emerge. Polaris pro-

vides extensive support for both of these analysis techniques.

Filtering allows the user to choose which values to display so that he can focus on and devote more

screen space and attention to the areas of interest. For all fields, the user can right-click on the field

name and choose the “Filter...” option.

For ordinal fields, a listbox with all possible values is shown, and the user can check or uncheck

each value (checked values are displayed, unchecked values are not), as shown in Figure 7.6. For

quantitative fields, a dynamic query slider allows the user to choose a new domain, as shown in

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 88

Figure 7.7. Additionally, there are textboxes showing the chosen minimum and maximum values

that the user can use to directly enter a new domain. In using Polaris, we discovered that we needed

to provide a slightly larger domain than the actual data domain for the user to select from, since the

user may want to see some buffer space in the graphs.

Note that while applying a filter to a quantitative field does not change its table algebra interpre-

tation, it does change the interpretation of an ordinal field: it reduces the domain to just the filtered

values, rather than all values.

Sorting allows the user to uncover hidden patterns and trends and to group together interesting

values by changing the order of values within a field’s domain or the ordering of tuples in the data.

Changes to the ordering of values within a field’s domain can affect the ordering of the panes within

a table, the ordering of values along an axis (such as in a bar chart), and the composite ordering

of layers. The ordering of tuples affects the drawing order of marks within a pane. The drawing

order is most relevant in graphs where a single primitive encodes multiple tuples, such as a line or

polygon primitive, or where marks overlap and the drawing order thus determines the front-to-back

ordering and occlusion of marks.

Polaris provides three ways for a user to sort the domain. First, the user can bring up the

filter window and drag-and-drop the values within that window to reorder the domain, as shown in

Figure 7.6. Second, if the field has been used to partition the table into rows or columns, the user

can drag-and-drop the table row or column headers to reorder the domain values. Finally, Polaris

provides programmatic sorting, allowing the user to sort one field based on the value in another

field. For example, the user may want to sort the State field by the Profit field. The sort ordering of

tuples within a pane is determined by which fields the analyst has placed in the Sort shelf.

7.4.3 Brushing and tooltips

Many times when exploring a database, analysts want to directly interact with the data, visually

querying to highlight correlated marks or getting more details on demand. Polaris provides both

brushing and tooltips for this purpose.

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 89

Brushing allows the user to choose a set of interesting data points by drawing a rubberband around

them. The user selects a single field whose values are then used to identify related marks and

tuples. All marks corresponding to tuples sharing selected field values with the selected tuples are

subsequently highlighted in all other panes or linked Polaris views. Brushing allows the analyst to

choose data in one display and highlight it in other displays, allowing correlation between different

projections of the same data set or relationships between distinct data sets.

Tooltips allow the user to get more details on demand. If the user hovers over a data point or pane,

additional details, such as specific field values for the tuple corresponding to the selected mark, are

shown. Analysts can use tooltips to understand the relationship between the graphical marks and

the underlying data.

7.4.4 Hierarchical analysis

When interacting with hierarchical data, the analyst needs mechanisms to take advantage of the

hierarchical structure. The analyst must be able to quickly and efficiently navigate the hierarchies

and be able to include hierarchical expressions on the shelves of the display. Polaris provides

interface mechanisms for both drill down and for including qualified hierarchical expressions (i.e.,

hierarchical expressions using the dot operator) on the shelves.

## Drilling down and Rolling up

When analyzing and exploring large data cubes, a common operation is to drill down or roll up

within a dimension hierarchy. Therefore, it is important to include a simple mechanism for perform-

ing these operations. One option is for the analyst to remove the current level from the appropriate

shelf (by dragging it off the shelf) and then drag the new level to that same shelf. Although the

desired effect is achieved, it is more complicated than we would like.

We provide an alternate mechanism for drilling down and rolling up a dimension. Within the

box representing each dimension level on a shelf, there is an “∇” icon, as can be seen in Figure 7.2.

When the user clicks on the “∇” icon, he is presented with a listing of all the levels of the dimension

(including diverging levels in complex dimensional hierarchies). Selecting a new level is interpreted

as a drill down (or roll up) operation along that dimension and the current level is automatically

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 90

replaced with the selected level (with the same qualification). Thus, the user can rapidly move

between different levels of detail along a dimension, refining the visualization as he navigates.

## Qualifying Dimension Levels

When an analyst drops a dimension level, such asMonth, on a shelf, there are several potential

intentions. He may intend to include the operandMonth in an expression, but he may also mean

Year.Monthor Year.Quarter.Month; the analyst needs to be able to specify the exact expression

desired. Our solution is to make the full qualification (e.g.,Year.Quarter.Month) the default. To

generate a different qualification, the user can right-click the dimension level in the shelf and select

the “Qualification. . . ” menu item. He is then presented with a dialog box that allows him to explic-

itly specify which intermediate levels to include in the qualification of the operand, thus generating

the applicable expression.

7.4.5 Undo and redo

The final interaction technique we provide in Polaris is unlimited undo and redo within an analysis

session. Users can use the “Back” and “Forward” buttons on the top toolbar either to return to a

previous visual specification or to move forward again. This functionality is critical for interactive

exploration since it allows the user to quickly back out changes (or redo them if he goes too far

back) and try a different exploration path without having to rebuild the desired starting point from

scratch. Support for undo also promotes more experimentation and exploration, as there is no fear

of losing work done thus far. If the user does want a clean canvas, Polaris also provides a “Clear”

By using the formalism and visual specification, implementing Undo and Redo is trivial. We

simply need to save the visual specification at each stage, and moving backwards or forwards is

done by simply updating the display to reflect the saved visual specification.

7.5 Results

Polaris is useful for performing the type of exploratory data analysis advocated by statisticians such

as Bertin [6] and Cleveland [21]. We demonstrate the capabilities of Polaris as an exploratory

interface to multidimensional databases by considering the following three scenarios.

https://lh3.googleusercontent.com/notebooklm/AG60hOocG6HLDRjzmuSyAm1BTu_P9CWPhXCgpzxHSl1gNzftXqA0E72SK8G5cq3m2uvcVVVLXxEla6XSdVwypXqkSVNbSEp5pO8mzFfiRbZh5aZqduO_WVMhbt1K7O0HTT7beWG-AzhB=w1500-h1300-v0

8ea50338-02de-41e5-87ca-9e23c1028d78

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 91

Figure 7.8: The first visualization created in an analysis of sales data for a hypothetical coffee chain (COFFEE). The analyst is concerned with reducing marketing expenses. In this display, the analyst is examining the relationships between marketing costs, profit, and sales, as a function of product and where the product was sold (state). In the circled region, the analyst can see that for some of the products there is a negative correlation between both sales and profit and marketing and profit. As the company sells more of these products they are actually losing money.

7.5.1 Scenario 1: Commercial database analysis

A marketing analyst for a national coffee store chain (COFFEE) is concerned with reducing mar-

keting expenses and is trying to identify products that are not generating profit or sales proportional

to their marketing costs. To get an initial understanding of the situation, the analyst creates a table

of scatterplots showing the relationship between marketing costs, profit, and sales (Figure 7.8). The

analyst has drilled down to the product and state level. The two charts circled in orange show that

several of the distributions do not reflect the positive correlations that the analyst was expecting.

To further investigate, the analyst reduces the scatterplot matrix to two graphs and discriminates

https://lh3.googleusercontent.com/notebooklm/AG60hOrzE9ICMmsftyxrDWNx6K4EWp6dEGNbRYVUoqr1EYCUPYLhYpMgUlLhY-LIyytYf_U_8_6B95EGfL0_zXxXIZCc2bT26lkr4SRTkYgPj_C2qHRJ4xnXZswQcNp2S_ARcajblLP4=w1504-h1339-v0

2ef9ce90-7207-4f01-bfea-78fcd8d3331c

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 92

Figure 7.9: The second visualization created in an analysis of sales data for a hypothetical coffee chain. The analyst has focused on two of the charts from the last visualization and added color and shape encodings to help them correlate points on the graph with product categories sold in specific regions. The analyst has chosen to focus on just two of the sets of poorly performing products: those which are shown as blue squares and magenta crosses (circled in the figure). The legend shows that these points correspond to espresso sales in the east region and tea sales in the west region. Now, the analyst will generated a new visualization to get more detailed information on these product categories.

the tuples by market and product type (Figure 7.9). This is done by removing theSalesandMarket-

ing fields from the y-axis shelf, removing theProfit field from the x-axis shelf, adding theMarket

field to the Color shelf, and adding theProductTypefield to the Shape shelf. Using the resulting

visualization, the analysts can identify espresso products in the East region and tea products in the

West region as having the worst marketing cost to profit ratios. These products are highlighted in

the figure.

https://lh3.googleusercontent.com/notebooklm/AG60hOoeJiv-Xa7m56bd0GcwhpOjPqNSVKXH9D4D1ZLz_t4Wz53HN7d_nilCcJu2EUt8g2TBbopbNyFb-ZiYDx1QfeA6B6Y8EJbgI3RPysQR6pl-C_PahUS8CpQB6s1wLyhvGOE1yMSNKw=w1500-h1337-v0

c348ee50-61f9-4630-9fbb-f0abeeedf11e

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 93

Figure 7.10: The final visualization created in an analysis of sales data for a hypothetical coffee chain (COFFEE). Using this visualization, the analyst can identify which specific products are per-forming poorly. The visualization is set up to show products with high marketing costs and low (or negative) profit as brightly colored red bars that are low (or negative) in their stacks. Two such products are circled in the display: caffe mocha in the east and green tea in the west.

In the final visualization, Figure 7.10, the analyst drills down into the data to get a more detailed

understanding of the correlations: She creates a small multiple set of stacked bar charts, one for each

market and product type. This is done by replacing the contents of the x-axis shelf withProduct,

addingMarket to the y-axis shelf, and changing the mark type to “Bar”. Within each chart, the

data is further drilled down by individualProductandStateby adding these fields to the “Level of

detail” shelf. Finally, each bar is colored by the marketing cost by dragging theMarketingfield to

the Color shelf. As can be seen in the visualization, several products such as Caffe Mocha in the

East have negative profit (a descending bar) with high marketing cost (a bright red bar). Having

https://lh3.googleusercontent.com/notebooklm/AG60hOoYgPOmIutbtaazUsTvGqc5wqDW0gs99j-w9SNyJ_lveKEk47lZ1p5poJCTS-3gFvLKogHGRS6P8XIJO3DVk93j7Kif3YIH1XxiBMHKGxfzg6JFhw7wYpBRjMDtCTFggM6Tcndguw=w1501-h1128-v0

7fb1ec54-695b-4066-81c4-12848a3bfa8a

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 94

Figure 7.11: The first visualization in an analysis of scalability issues experienced by a parallel graphics application (ARGUS). Initially, the developers hypothesized that the diminishing perfor-mance was caused by remote memory accesses and this visualization was constructed to test this hypothesis. The first view shows the source code colored by the number of memory misses. There are not many misses and they are occurring at expected locations, such as synchronization primi-tives. The second view shows misses by memory index; there are few misses and no clear point of contention. The insight gained from these visualization is that the memory behavior is not likely to be causing the problem.

identified such poorly performing products, the analyst can modify the marketing costs allocated to

7.5.2 Scenario 2: Computer systems analysis

At Stanford, researchers developing Argus [42], a parallel graphics library, found that its perfor-

mance had linear speedup when using up to 31 processors, after which its performance diminished

rapidly. Using Polaris, we recreate the analysis (ARGUS) they originally performed using a custom-

built visualization tool [8].

Initially, the developers hypothesized that the diminishing performance was a result of too many

https://lh3.googleusercontent.com/notebooklm/AG60hOr01sQnd2-Tn5Uv6Y_OJrPa3qjWaw5npEvIruBLv58P3oMjBnMibhuIrGg08yucXM98MAoHSFTyopW3n577JMm32VUhoM4TGKEGm0QtEyXfmJZBOXFG__NSTbjITkPUswKbn5LS=w1500-h1582-v0

6f4f2525-b0b0-4288-8fe9-0c82fbb5dc4c

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 95

Figure 7.12: The second visualization created in an analysis of the scalability issues experienced by a parallel graphics application. After eliminating remote misses as a possible cause the developers next hypothesized that lock contention might be an issue. These visualizations show two projections of the same data. The top visualization shows locks events over time as a scatterplot and histogram. Towards the end of the run, the duration of lock events is unexpectedly long. This indicates the lock contention warrants further investigation.

remote memory accesses, a common performance problem in parallel programs. They collected

and visualized detailed memory statistics to test this hypothesis. Figure 7.11 shows a visualization

constructed to display this data. The visualization is composed of two linked Polaris instances,

one displaying a histogram of cache misses by virtual address and the other displaying source-code

https://lh3.googleusercontent.com/notebooklm/AG60hOpzJ2e7gPK8YXdf6F2k48-1iXPvtI0hzd-M6vl6tbRe9jfYPlraznR_n9Tj_xm0pLXfMQq-e3pGv8SdfO_KJJWfcn8sgdMtH8LF-QHUBT7HRuJMyraQ2S2RZX-UJD81smzwA7t9Og=w1500-h1292-v0

6dbae62f-6011-44f9-96a4-b1d6f6170f18

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 96

Figure 7.13: The final visualization created in an analysis of an application’s scalability issues. This visualization depicts lock events and thread scheduling as a series of Gantt charts, one pair for each processor. The display shows that the long lock requests correspond to long descheduled periods for most processes except one. One process is descheduled while holding a lock, thus causing the remaining processes to also block. This behavior was caused by a bug in the operating system.

with each line’s hue encoding the number of cache misses suffered by that line. Upon seeing these

displays, they could tell that memory was not the problem.

The developers next hypothesized that lock contention might be a problem, so they re-ran Argus

and collected detailed lock and scheduling information. The data is shown in Figure 7.12 using two

instances of Polaris to create a composite visualization with two linked projections of the same data.

One projection shows a scatterplot of the start cycle versus cycle duration for the lock events (re-

quests and holds). The second shows a histogram over time of initiated lock events. The scatterplot

shows that towards the end of the run, the duration of lock events (both holds and requests) were

taking an unexpectedly long time. That observation, correlated with the histogram showing that the

https://lh3.googleusercontent.com/notebooklm/AG60hOocIzregPfXwVC5mR0FQMDhUSzJGHEhF1ikgCq3akxvRIbM_r1LcnOWeOZfDkUt1Kn1H99ezV-dlLX2_3yV3Cb8aTquwrual5jiPeXlwEZx45JC7EA4tvWdf10GSgDDYVgIYlkRkg=w1480-h1277-v0

9c20ecd9-1644-4b2c-9314-a5932b52f48b

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 97

Figure 7.14: The first visualization created during the analysis of a 12-week trace of the mobile network in the Gates building at Stanford University (WAVELAN). The analyst is trying to under-stand usage patterns of the mobile network. This initial visualization shows the size and number of packets over time for the most common applications run on the network (e.g. FTP, web). From this display, it is clear that the web is the most commonly used application and file transfer is the least commonly used.

number of lock requests peaked and then tailed off towards the end of the run, indicated that lock

contention might be a fruitful area for further investigation.

A third visualization, shown in Figure 7.13, shows the same data using Gantt charts to display

both lock events and process scheduling events. This display shows that the long lock requests

correspond to descheduled periods for most processes. One process, however, has a descheduled

period corresponding to a period during which the lock was held. This behavior, which was due to

a bug in the operating system, was the source of the performance issues and the visualization was

key to discovering this bug.

https://lh3.googleusercontent.com/notebooklm/AG60hOpTYscElB9aNvCjdz3CmZUaEa6UxpNpXQNaC62iIJz4ig-Ycb6eot3hdmRDwZ6Du6Z-AOo3d8fkyFyBB69AoYPnUhVrloC-pQBx24bkJlvkcuf-je5PCn6edHmS03_Snrrw6FZnoA=w1477-h1275-v0

1ea84e71-cc78-42cc-a727-e907e68a97ed

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 98

Figure 7.15: The second visualization created during the analysis of a 12-week trace of mobile network usage. This visualization was constructed to help understand how application mix (e.g., FTP, web) varies with research group. There is a single line chart for each research group, with a line within the chart for every application class. From this display the analyst concludes that the Graphics group is responsible for the large incoming and outgoing file transfers, while the Systems group has unusually large session traffic.

7.5.3 Scenario 3: Mobile network usage

Figure 7.14 shows an analysis of a 12-week trace of every packet that entered or exited the mobile

network in the Gates building at Stanford University (WAVELAN). The analysis goal is to under-

stand usage patterns of the mobile network. To start the analysis, the analyst first sees if she can spot

any patterns in time, so she creates a series of line charts in Figure 7.14 showing packet count and

size versus time for the most common applications, broken down and colored by the direction of

the traffic. In these charts, the analyst can see that the web is the most consistently used application,

while session traffic is almost as consistent. File transfer is the least consistent, but also has some

https://lh3.googleusercontent.com/notebooklm/AG60hOoCjoNbxHoaOaf17ucmqnpCy5ZHJG1o6D65QL54Qzun85VWTmza4njfUoVR7kMRbZHSFvHTh29ad7QS4NnRwJ1OqSOuNPRvv-26Xi8aJp9s_8nrOsyrsS-aM81l0iJ9XcHZJ-jRtg=w1474-h1273-v0

0ca8e56c-aada-4084-9e59-766898b885ae

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 99

Figure 7.16: The final visualization created in an analysis of a 12-week trace of a mobile network. This display is a refinement of Figure 7.15: the analysts has drilled down from research area to individual project to better understand the sporadic large file transfers. In this view, it is apparent that the Rendering project is generating the large file transfers.

of the highest peaks in both incoming and outgoing ftp traffic. Note the log scale on the y-axes.

Given this broad understanding of traffic patterns, the next question posed by the analyst is

how the application mix varies depending on the research area. The analyst pivots the display to

generate a single line chart of packet count per research area over time, broken down and colored

by application class (Figure 7.15). From this breakdown, the analyst can see that the graphics group

was responsible for the large incoming and outgoing file transfers. She can also see that the systems

group had atypically high session traffic.

Curious, the analyst then drills down further to see the individual project groups (Figure 7.16),

discovering that the large file transfers were due to the rendering group within the graphics lab,

while the robotics lab had vastly different behavior depending on the particular group (the mob

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 100

group dominated by session traffic, while the learning group had more web traffic, for example).

7.5.4 Summary

These examples illustrate several important points about the exploratory process. Throughout the

analyses, both the data the users want to see and how they want to see it change continually. Analysts

first form hypotheses about the data and then create new views to perform tests and experiments to

validate or disprove those hypotheses. Certain displays enable an understanding of overall trends,

whereas others show causal relationships. As the analysts better understand the data, they may want

to drill down in the visible dimensions or display entirely different dimensions.

Polaris supports this exploratory process through its visual interface. By formally categorizing

the types of graphics, Polaris is able to provide a simple interface for rapidly generating a wide

range of displays, thus allowing analysts to focus on the analysis task rather than the steps needed

to retrieve and display the data.

7.6 Discussion

In this section, we focus on three points of discussion. First, we discuss the different roles Polaris

can play in the knowledge discovery process, second, we discuss our experience using the system,

and finally, we discuss the interactivity and performance of the Polaris interface.

7.6.1 Visualization and data mining

We have demonstrated the effectiveness of Polaris as a stand-alone tool for visual mining of large,

hierarchical databases. Equally important is how Polaris can be coupled with automated data min-

ing systems to help analysts better understand not only their data, but also the models generated

by the algorithms. First, Polaris can be used as a precursor to data mining: The analyst benefits

from an understanding of the overall structure of the data that helps her steer the discovery process

and provides context for “hidden information” discovered by the algorithms. Second, Polaris can

also be used to validate and comprehend the models and results generated by algorithmic analysis.

Analysts do not want to treat an algorithm as a black box and blindly trust its output. One technique

for using Polaris for validation is to construct hierarchical dimensions from the output generated by

classification algorithms. The analyst can then drill down and roll up the data, traversing the classifi-

cation hierarchy and inspecting the tuples sorted into each bucket, further developing understanding

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 101

7.6.2 Experiences

Although we have not conducted a formal user study with the Polaris interface, the system has been

used by a fair number of users and we have received important feedback. The predominant users

of the system have been researchers working on the Polaris and Rivet projects: myself, Diane Tang,

and Robert Bosch. We have used the system in two ways: to recreate analyses that were performed

initially using custom-developed, domain-specific tools, and to perform additional analysis on data

sets we had previously analyzed. In addition, other researchers within the Computer Science de-

partment have used Polaris within their research, specifically Jeff Gibson (FLASH project) and John

Gerth (network administration). The data sets explored using Polaris have ranged from astronomical

data to computer systems data to corporate sales data. Our experiences have lead to several insights

and subsequent design decisions:

Cycle of analysis: While performing analyses, we found that we were continually posing

new hypotheses and needing to construct new visualizations. Many of the data sets we have

explored had been analyzed previously using custom-written visualizations. As a result, the

time between iterations of the cycle of “hypothesis-experiment-hypothesis” had been consid-

erable. When we reexamined these data sets using Polaris, we found we were able to explore

many more “what-if” scenarios and to explore areas we had neglected in our initial analysis.

As a result, this has been a key factor in all subsequent design decisions: to directly support a

short and efficient cycle of analysis.

Data Transformations: The original version of Polaris did not support the filtering and

sorting of data within the visualization. While using the system, it quickly became clear that

these were important tools and that to perform effective analysis within Polaris we would

have to provide filtering and sorting mechanisms. After adding those features, we had a

similar revelation with regards to derived fields. Both experiences were reflections of the

same insight: in order to support an efficient cycle of analysis, the user must be able to

manipulate their data within the visualization system.

Database integration: The original version of Polaris did not integrate with commercial

database systems. Instead, it used its own internal transformations to process flat files. As we

began exploring larger data sets we quickly realized that it was going to be critical to tightly

integrate with a database systems for two reasons: (1) we were storing most of our data sets in

relational databases and did not want to have to manually extract the data and (2) the database

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 102

systems were able to provide a level of performance that our internal transformations could

7.6.3 Performance

A final point of discussion is the interactivity and performance of Polaris. Although Polaris is

designed to be an interactive and exploratory interface to large data warehouses, our research has

focused on the techniques, semantics, and formalism needed to provide an effective exploratory

interface rather than on attaining interactive query times. While we would like the system to be

reasonably responsive as the user modifies the visual specification, our experience has been that

the query response time does not need to be real-time in order to maintain a feeling of exploration:

the query can even take several tens of seconds. Within this constraint, Polaris can currently be

used with many large databases, especially if a large subset of the views can be materialized a

priori [74]. Furthermore, it is important to note that many queries on data warehouses, such as those

generated with existing Pivot Table tools, will be returning a small number of tuples, and thus the

most relevant constraint on performance is server-side query time and not client-side drawing or

data manipulation.

We have used Polaris with several reasonably large data sets (see Section 2.5). With these data

sets, we are able to get reasonably responsive performance and maintain a sense of exploration for

the queries we ran. We intend to pursue database performance issues in order to scale to much larger

data sets as part of our future research. There are many techniques that can be used to improve the

performance of the queries, including existing techniques such as materialized views [74], progres-

sively refined queries that provide intermediate feedback [39], and sampled queries [34][46]. We

also think that substantial performance benefits can be gained by leveraging the coherence between

successive queries generated by visualization systems using both caching and prefetching.

7.7 Summary

In this chapter, we introduced the Polaris interface, an interface for exploring large multidimen-

sional databases. The Polaris interface extends the well-known Pivot Table interface to directly

generate a rich, expressive set of graphical displays. The Polaris interface is simple and expressive

because it is built on top of the Polaris formalism. The state of the interface is interpreted as a visual

specification of the analysis task and automatically compiled into data and graphical transforma-

tions. This architecture results in a system that directly supports the analysis process through the

CHAPTER 7. INTERACTIVE ANALYSIS AND EXPLORATION 103

interface. Analysts can rapidly change the data they are viewing, the type of display, or the level of

detail. Furthermore, Polaris’s use of table-based displays allows the analysts to visualize both many

tuples and many dimensions of their data simultaneously.

## Multiscale Visualization

In the previous chapter, we demonstrated the Polaris interface as one approach to building tools for

the analysis of large databases. The Polaris interface provides the user with an interactive environ-

ment in which they can rapidly construct different visualizations of subsets of their data. Polaris

was designed to be applied to any multidimensional data set and is thus very general. When an

analysis tool is to be used within a specific domain, however, it makes sense for the developer to

narrow down the options presented to the user. In this chapter we explore the technique of multi-

scale visualization, address several limitations in current approaches to multiscale visualization, and

demonstrate how the Polaris formalism can be used to efficiently specify and implement multiscale

visualizations. Furthermore, we introduce the idea of design patterns as a technique for capturing

the structure of effective, domain-specific visualization, then and reusing that structure for different

problems and domains.

8.1 Overview

When exploring large datasets, analysts often work through a process of “Overview first, zoom

and filter, then details-on-demand” [61]. Multiscale visualizations are an effective technique for

facilitating this process because they change the visual representation to present the data at different

levels of abstraction as the user pans and zooms. At a high level, because a large amount of data

needs to be displayed, it is highly abstracted. As the user zooms, the data density decreases and thus

more detailed representations of individual data points can be shown.

The two types of abstraction performed in these multiscale visualizations are data abstraction

and visual abstraction. Data abstraction (e.g., aggregation or selection) changes the underlying data

CHAPTER 8. MULTISCALE VISUALIZATION 105

before mapping it to visual representations. Visual abstraction changes the visual representation of

data points (but not the underlying data itself) to provide more information as the user zooms; e.g.,

an image may morph from a simplified thumbnail to a full-scale editable version. Existing systems,

such as DataSplash [79] and Pad++ [3], focus primarily on visual abstractions with support for data

abstractions limited to simple filtering and the ability to add or switch data sources. In addition,

these systems primarily allow for only a single zooming path.

This chapter presents a system for describing and developing domain specific multiscale visual-

izations that support multiple zoom paths and both data and visual abstraction. We want to support

multiple zoom paths because many large data sets today are organized using multiple hierarchies

that define meaningful levels of aggregation (i.e., detail). By representing the database with a data

cube, we can switch between different levels of detail using a general mechanism applicable to many

different data sets. Combining this general mechanism for performing meaningful data abstraction

with traditional visual abstraction techniques enhances our ability to generate abstract views of large

data sets, a difficult and challenging problem.

In this chapter, we describe a multiscale visualization system using data cubes and Polaris.

Specifically, we present:

Zoom graphs: We present zoom graphs as a formal notation for describing multiscale vi-

sualizations of hierarchically structured data that supports multiple zooming paths and both

data and visual abstraction. We also present a system based upon this formalism in which we

can easily implement these visualizations.

Design patterns:While these zoom graphs and our system provide a general method for de-

scribing and developing multiscale visualizations of hierarchically structured data, designing

such visualizations remains a hard and challenging problem. We use our formalism to enu-

merate four design patterns in the style of Gamma et al. [33] that succinctly capture the critical

structure of commonly used multiscale visualizations. In addition, these patterns illustrate the

use of small multiples and tables in multiscale visualizations.

8.2 Related work

In this section, we review several existing multiscale visualization systems, focusing on how the

systems perform both data and visual abstraction. Data abstraction refers to transformations applied

to the data before being visually mapped, including aggregation, filtering, sampling, or statistical

CHAPTER 8. MULTISCALE VISUALIZATION 106

summarization. Visual abstraction refers to abstractions that change the visual representation (e.g.,

a circle at an overview level versus a text string at a detailed level), change how data is encoded

in the retinal attributes of the glyphs (e.g., encoding data in the size and color of a glyph only in

detailed views), or apply transformations to the set of visual representations (e.g., combining glyphs

that overlap or simplifying polygons).

8.2.1 Multiscale visualization in cartography

Cartography is the source of many early examples of multiscale visualizations. Cartographic gen-

eralization [70] refers to the process of generating small scale maps by simplifying and abstracting

large scale source material. Generalization consists of two steps: (1) employing selection to decide

which features should be shown and (2) simplifying the visual representations of the selected fea-

tures. A map series developed using this process and depicting a single geographic region at varying

scales is a multiscale visualization. While the initial selection process is a specialized form of data

abstraction, the subsequent manipulations are all visual abstractions.

Several data structures that are similar to data cubes have been developed in efforts to develop

interactive systems for exploring (or simply producing) map series. Representative models are the

Multiscale Tree [30] and Map Cube [69], both of which utilize a trade-off between storage and

computation: redundant storage replaces all steps in the cartographic generalization process that are

difficult to automate. The general approach is to pre-generate (either by hand or algorithmically) a

tree that stores generalizations of objects at different levels of detail. A selection process generates

output by extracting generalizations from this tree until a specified information density is reached.

While these structures are generated using a pre-computation process similar to data cubes, the

process focuses entirely on visual abstraction rather than data abstraction.

8.2.2 Multiscale information visualization

Several information visualization systems provide some form of zooming or multiscale interface.

Given our goal of expressing general multiscale visualizations, we only discuss general systems;

domain-specific tools may apply both data and visual abstraction but their abstractions are not gen-

erally applicable.

The Pad series of interfaces (Pad++ [3] and Jazz [4]) are among the earliest examples of multi-

scale visualization in information visualization. These systems were developed not as data explo-

ration tools but as alternate desktops, although they have been applied to other domains such as web

CHAPTER 8. MULTISCALE VISUALIZATION 107

histories [40]. Given this goal, their focus has been on interaction and applying visual abstractions

for “semantic zooming” rather than easily applying data abstractions.

DataSplash [79] is the first multiscale visualization system focused on data exploration. It pro-

vides the layer manager, a novel interface mechanism for easily constructing multiscale visualiza-

tions of graphs. Each individual graph can have multiple layers, with each layer activated at different

viewing elevations. As the user zooms, the set of active layers changes Layers can be used to change

the visual representation of relations and to add or remove data sources. Although DataSplash pro-

vides mechanisms for zooming on a single graph, it does not provide mechanisms for zooming on

tables or small multiples of graphs, nor does it provide for multiple zooming paths on a single graph.

XmdvTool [59] provides multiscale views using hierarchical clusters that structure the under-

lying data into different levels of abstraction; widgets such as structured brushes [32] provide a

mechanism for zooming. XmdvTool is limited to this single method for providing data abstraction

and does not provide visual abstraction capabilities.

Eick and Karr also present a survey of common visual metaphors and associated interaction

techniques and motivate the need for both data and visual abstractions from perceptual issues [26].

These issues drive the ADVIZOR system, which uses multiple visual metaphors, each with a single

zoom path based on with the visual and data abstractions given in their survey. They do not pro-

vide a system for exploring other types of zooms nor a formal notation for describing multiscale

visualizations.

8.3 Multiscale visualization

In this section, we present our system for describing multiscale visualizations that support multiple

zoom paths and both data and visual abstraction. Rather than considering multiscale visualizations

as simply a series of linear zooms, we think of multiscale visualizations as a graph, where each node

corresponds to a particular set of data and visual abstractions and each edge is a zoom. Zooming

in a multiscale visualization is equivalent to traversing this graph. Each node in this graph can be

described using a Polaris specification that identifies the visual representation and abstraction and

can be mapped to a unique projection of the data cube, which is a data abstraction of the underlying

relational data.

In the remainder of this section, we first review the two technologies we use to perform data

abstraction (data cubes) and visual abstraction (Polaris). Finally, we present how we can create a

zoom graph of Polaris specifications to describe a multiscale visualization of a hierarchical data set,

CHAPTER 8. MULTISCALE VISUALIZATION 108

as well as how we can easily implement such visualizations within our system.

8.3.1 Data abstraction: Data cubes

Not only are data cubes widely used, but they also provide a powerful mechanism for performing

data abstraction that we can leverage. Specifically, data cubes quickly provide summaries of the

underlying data at different meaningful levels of detail, rather than arbitrary summarizations such

as aggregating every two tuples. As we discussed in Chapter 2, this goal is achieved by building

a lattice of data cubes to represent the data at different levels of detail according to a semantic

hierarchy and providing mechanisms for subsequently summarizing each cube.

Data abstraction in this model means choosing a meaningful summary of the data. Choosing a

data abstraction corresponds to choosing a particular projection in this lattice of data cubes: (a) the

dimensions we currently consider relevant and (b) the appropriate level of detail for each relevant

dimensional hierarchy. Specifying the level of detail identifies the cube in the lattice, while the

relevant dimensions identifies which projection (from n-dimensions down to the number of relevant

dimensions) of that cube is needed.

While identifying a specific projection in the data cube corresponds to specifying the desired

data abstraction of the raw data, in multiscale visualizations we need to specify both the data and

visual abstractions; both sets of information are contained in a Polaris specification.

8.3.2 Visual abstraction: Polaris

In the earlier chapters of this dissertation, we presented the Polaris database exploration tool, con-

sisting of three parts: (1) a formal specification language for describing table-based visualizations,

(2) a user interface for automatically generating instances of these specifications, and (3) a method

for automatically generating the necessary database queries to retrieve the data to be visualized by

a specification.

In this chapter, we only use the specification language, and we use this language to describe

a node within the zoom graph identifying a multiscale visualization. A single Polaris specification

describes not only the visualization to be generated, but also the corresponding data cube projection,

and thus the desired data abstraction. We now briefly review each part of a Polaris specification and

how they related to visual and data abstraction:

The table structure: Two expressions in the table algebra, one each for the x- and y-axis,

define (1) the rows and columns of the table and (2) how data is spatially encoded within

CHAPTER 8. MULTISCALE VISUALIZATION 109

each pane. Changing the expressions changes the data abstraction; for example, using the dot

operator on an operand identifies a different data cube.

Internal level of detail: This portion of the specification identifies any dimensions that are

needed but not already encoded in the table structure. Together, the complete list of dimen-

sions uniquely identify the desired projection of the data cube. Changing the internal level of

detail changes the data abstraction.

The mapping of data sources to layers:Multiple data sources may be combined within a

single Polaris visualization, with each source mapped to a separate layer. All layers share the

same table structure and are composited together back-to-front to generate the final visualiza-

The visual representation for tuples: Both the mark type and the visual properties and

encodings of each mark can be specified. Changing the mark, the visual encodings, or the

table structure changes the visual abstraction.

8.3.3 Zoom graphs

In a typical analysis, the user starts at a high level overview and successively zooms in on areas

of interest; as he zooms, he also changes the visual representation of the data being displayed. In

the Polaris interface, we store the analysis history as a graph of Polaris specifications,with each

specification representing a visualization created during the analysis. This history captures not only

the visualizations created, but also the ad hoc visual and data abstraction selected by the analyst (the

analyst selects the data abstraction as well as the visual because we can map each specification to a

particular projection of a specific cube). This path of exploration is depicted in Figure 8.1.

Our insight is that we can use this model not only to record the analysis process, but also to

describe multiscale visualizations supporting both visual and data abstraction. Specifically, we can

link specifications together into a graph with neighboring specifications corresponding to a change

in abstraction (visual, data, or both) and with the links representing possible zooms.

This zoom graph model supports multiple zoom paths from any given point, since each node can

have multiple incoming and outgoing links (each corresponding to a different path). This flexibility

is needed for exploring data cubes that commonly have multiple independent hierarchical dimen-

sions. An individual zoom can change either the data abstraction, the visual abstraction, or both.

https://lh3.googleusercontent.com/notebooklm/AG60hOo3FuPJ4SDGKTC4aBXHF-ASpbxZ1lHSFLI2ZYyiam7m4q2Yey_PhxHk1_7ip52_kciVyTmrmPH93RXsrvO9OXaAe4BDtpmRihaW155Rgi0GDTaajC-jvpiJ4nPeVb1GuPCvJP-1QQ=w1804-h1010-v0

2807683e-4629-4384-8ef7-6a8c6ab23049

CHAPTER 8. MULTISCALE VISUALIZATION 110

Figure 8.1: (a) An analysis session can be described by a path (or graph) of Polaris specifications, each corresponding to a visualization created during the analysis. Examining the graph, we can see that during the analysis, the user implicitly performs visual abstraction as they change display types to test hypotheses and investigate areas of interest. (b) Furthermore, if we consider the mapping of Polaris specifications to data cubes in the lattice of cubes, it is apparent that an analysis also involves implicit data abstraction. This insight leads to the realization that general multiscale visualizations can be modeled as a graph of specifications.

The zooming actions can be tied to an axis, for example allowing zooms along the x- and y-axis

independently, or they may be triggered by interacting with an external widget.

The previous sections describe how to express a node in the graph using a Polaris specification

and how a specification corresponds to a particular projection of a data cube. Using the graphical

notation we introduced, we can describe and design these zoom graphs. In the remainder of this

section, we explain how we implement the multiscale visualization corresponding to a zoom graph

within Rivet [9], a visualization environment designed for rapidly prototyping interactive visualiza-

tions. The main components of any implementation of a multiscale visualization are the nodes, the

edges, and how user interaction can trigger a transition (i.e., an edge traversal).

Nodes:Each node in a zoom graph is abstractly described using a Polaris specification. In our

implementation, we can concretely describe a Polaris specification using XML. Rivet contains an

interpreter that parses the XML to create a visualization, which includes automatically generating

https://lh3.googleusercontent.com/notebooklm/AG60hOpfDpQEyQLwS5UdMzdqnxTHcN0k3sf-xJeaHxooKsNi--oOinPTMCIPCi3n2YeCUC3zKG-rEgrQfIen41xiHYVb0yZwUl6iWLcZqDpbKCUe_mwev1d-c8mGXh3-BbnzSDZpohNW4Q=w1133-h587-v0

db3c7b07-e5cf-4e80-b18b-09e702687dbc

CHAPTER 8. MULTISCALE VISUALIZATION 111

Figure 8.2: The type hierarchy within Tableau. Atemplate specificationis specified by including type constraints, rather than specific field names, within the specification.

both the necessary queries (i.e., SQL or MDX queries) and drawing operations.

Edges and Interaction: Visualizations are created in Rivet by writing a script. An abstract

zoom graph can be implemented as a finite state machine within a Rivet script. We use Rivet’s event

binding mechanism to bind user interaction events (e.g., mouse movement events) to procedures

within the script to trigger transitions between states (i.e., nodes).

The notation described in this section is very general and can be used to describe many different

graphs, i.e., many different multiscale visualizations. Many of these visualizations are easily imple-

mented within Rivet. However, designing effective multiscale visualizations is still a challenging

task, and in the next section, we present four patterns using our graphical notation that describe

common multiscale visualizations and encapsulate the changes in abstraction that occur as the user

8.4 Design patterns

Even though implementing a multiscale visualization is simplified using our system, designing such

a visualization is still, in general, a challenging problem. One way to help solve this problem is to

capture zoom structures that have been effective as patterns that can be reused in the design of new

visualizations. Before considering the design pattern for a complete multiscale visualization, we

consider design patterns for a single visualization.

https://lh3.googleusercontent.com/notebooklm/AG60hOomW6cx-sQi8ElgF3wHrqU5_-0HRe_ukBfomXve9ytQANryGvFafMuKFMPCvVqY4yhxjQ1Pfc5v4AmZZm49t8Gzx74w5WJ-OTeKg3XYCIqQ8ND8yENe--kX1zQwC_CJ-NrqTeSfdg=w1129-h371-v0

02f10868-7403-4b1f-a55b-6c6ce6c48dc2

https://lh3.googleusercontent.com/notebooklm/AG60hOpmiHUEoBzejFGs0PyZNM4eZ9Y3xti9tVz49SYZcKtfEEH2HwwqS_G7-vi-EAUOADLw2XlRwqpnaSW0P-SG31cM8QDVDvDq-WtFjWRHpWYtCmp__MMcVNLfRRKVB3JXfvsDeudGAA=w1037-h842-v0

7bc25aaf-be08-49ad-b2ca-f08a0c8f4c20

CHAPTER 8. MULTISCALE VISUALIZATION 112

Figure 8.3: The extended graphical notation for describingtemplate specifications. Whereas the graphical notation previously only described specific visualization, with these extensions we can now describe classes of visualizations.

Figure 8.4: A map of the USA at the state level of detail and encoding population as the color of each state polygon. Below the visualization is graphic description of this specific visualization and the template specificationfor the class of visualizations that encode a dependent measure as the color of a polygon that represents a geographical entity at a categorical level of detail.

8.4.1 Template specifications

In Section 5.4, we introduced a graphical notation for describing a specific visualization. We can

use this same notation to specify atemplate specificationfor a set of visualizations by using type

constraints rather than specific fields within the specification.

Specifically, where with our original graphical notation we could only specify a field name,

within a template specification we can specify either a field name (e.g.Latitude) or a type constraint

https://lh3.googleusercontent.com/notebooklm/AG60hOrVQ--8MXpbNsDRt-M1nMEdafK5OraLd8rv5zPQs44Jn_i3vBTgiZMW7uL824wiKZFBJm2Mq2TIQlMvH2JzldQU9lMzSGh3W7c1cYyduiTLkyDXXoO7hQs1CpZyypmw4A28NoHxLQ=w1521-h981-v0

3075be91-d99f-47c1-ae1c-f941890169ea

CHAPTER 8. MULTISCALE VISUALIZATION 113

Figure 8.5: Azoom pattern. Whereas a zoom graph described a specific multiscale visualization as a graph of Polaris specifications, a zoom pattern describes a class of multiscale visualizations as a graph of template specifications. Depicted is the overall structure of the pattern and then template specifications for key changes in visual and data abstraction within the pattern.

(e.g.,Q), requiring the encoding of a field of that type. Furthermore, any type constraint can be

qualified by the following qualifiers: (1) a level number as a subscript (e.g.,Q1) requiring the

encoded field be a level of a dimension hierarchy at the given depth, and (2) a suffix ofi or d

indicating that the encoded field must be either independent or dependent accordingly.

Figure 8.2 illustrates the type hierarchy within Polaris and the possible type constraints. Within

a template specification, we classify the encodings into three groups:

Required: A required encoding is indicated by including a specific database field or a type

constraint in the specification for that encoding.

Optional: An optional encoding is indicated by including the encoding in the specification

with a blank encoding slot.

Not allowed: An encoding that is not allowed within the described class of visualizations is

indicated by not including the encoding in the graphical template.

CHAPTER 8. MULTISCALE VISUALIZATION 114

Figure 8.3 illustrates the graphical representation for each type of encoding. In addition to

specify the encodings, we must also specify the table structure for the class of visualizations. This

is done by specifying the algebraic expressions using either field names or type constraints as the

operands within the expressions. Figure 8.4 depicts a map of population over the USA, the graphical

notation for describing that specific visualization, and the graphical notation for describing the class

of visualizations that uses color to convey a measure’s variation over geography.

8.4.2 Zoom patterns

Given our notation for describingtemplate specifications, we can extend the notion of Zoom Graphs.

Rather than describing a single zoom using a graph of Polaris specifications, we can describe a class

of zooms using a graph of template specifications called aZoom Pattern. This is illustrated in

Figure 8.5 where we show a zoom pattern that supports independent zooming within a line graph

along two dimensions.

8.4.3 Examples

In this section, we present four standard zooms and express them using our formal notation for

zoom patterns. These zooms have traditionally been used in domain-specific applications, and while

we also give specific examples for each, our notation expresses each pattern as a general class of

multiscale visualizations. Each zoom is described in the style of Gamma et al. [33], and the goal

is not only to provide some guidance to others when designing multiscale visualizations, but also

to provide a formal way for exchanging design knowledge and discussing multiscale visualizations

(i.e., which data and visual abstractions to apply).

8.4.4 Pattern 1: Chart Stacks

This first pattern applies when analysts are trying to understand how a dependent measure (such as

profit or number of network packets) varies with two independent hierarchical ordinal dimensions,

one derived from continuous data (such as time). This type of data can be effectively visualized

using a vertically stacked small multiple of line charts (e.g., a single column of charts) [72]. The

hierarchy derived from continuous data is encoded in the x-axis of each chart while the ordinal

hierarchy determines the y-axis structure of the table (e.g., the order and number of rows). The

y-axis for each individual chart encodes the dependent measure. The zooming in this pattern is

https://lh3.googleusercontent.com/notebooklm/AG60hOqrRbji8L-jqzExxlkZ5CxWQQCjM5K1OXHiyNqJO3cIUwQ_0zrh2HGj5P90pgLDTHjyqHaHTXwPBKxXE44c67jxq_g2Nh-YgJLNiNoACv1zm5jjQa-oNdjyftgHOgvfC8l0npQfWA=w1804-h1397-v0

1eff6432-c7d8-4a02-a8c5-6f3a41a529f7

CHAPTER 8. MULTISCALE VISUALIZATION 115

Figure 8.6: The Zoom Graph for the Chart Stacks Pattern as well as screenshots of a visualization of a trace of an in-building mobile network developed using that pattern (WAVELAN). The top visualization shows a line chart of average bytes/hour for each day for each research area. The line charts are layered above a high-low bar encoding the maximum and minimum bytes/hour. In the next visualization, the user has zoomed in on the y-axis, breaking apart the charts to create a chart for each advisor within the research groups. In the final visualization, the user has zoomed on the x-axis, increasing the granularity of the line chart to hourly values from daily values.

inspired both by the types of visualizations created in ADVIZOR [26] as well as in analyses of this

type of data [67] performed in our lab.

The main thing to note in this pattern is that the analyst can independently zoom along either the

x- or y-axis, leading to a graph describing the multiscale visualization; the analyst can choose any

path through this graph. Each zoom corresponds to changing the data abstraction: the dot operator

is applied to the table algebra expression corresponding to the relevant axis. Zooming along the

x-axis changes the granularity of each individual chart while zooming along the y-axis changes the

https://lh3.googleusercontent.com/notebooklm/AG60hOoX1H5Rs3bxhQLNIl_lTREFozGSLhlwLFBl5xTQAM4JSdVOa1I94eSJa53MOY8MdXjJXR5RgQnIjeJxcFU_IjEdTxd2eopbdcD1qGP8ktN4_S2wpflTAL6RhqKgDJ3lR2N9Vm0Meg=w624-h1346-v0

8c1dcc9f-472e-4ab1-b689-799c21e4873b

CHAPTER 8. MULTISCALE VISUALIZATION 116

Figure 8.7: A variation on the Chart Stack pattern: A visualization of kernel lock activity (lock requests are shown in blue; time holding a lock is shown in yellow) collected from a simulation of the Argus graphics library (ARGUS) [42]. The top visualization shows a histogram of average time spent requesting or holding a kernel lock. A time interval corresponds to one million cycles and CPUs are grouped by their primary task (e.g., processing geometry, rasterization, etc.). In the next visualization, the user has zoomed in on the y-axis, breaking apart the task charts to create a chart for each CPU, and has zoomed on the x-axis, changing the time granularity to one hundred thousand cycles per interval. In the final visualization, the user has zoomed further on the x-axis, resulting in a change in visual abstraction from strip charts to Gantt charts displaying individual events.

number of charts. The zoom graph for this pattern is shown in Figure 8.6.

Figure 8.6 also shows how we applied this pattern to a trace of every packet that entered or

exited a mobile network (WAVELAN). Each packet is categorized by the time it was sent (one

hierarchy) and the user who sent the packet (the second hierarchy). To transition between the

different specifications, the user can trigger the y-zoom by clicking on the arrow at the top of the

https://lh3.googleusercontent.com/notebooklm/AG60hOoSHeGXaussNhHevN0Eopydg28cSZF-_0KCERvo8Blt8ntxHKzpdjHnoEDyllPPL4Yi6I9oLZwDmbvwNOHkzsjGTU94fAvUtNOaKTI_PLAdJTqxYAVXxhssq8orX2IRYQ0CMONnKw=w1814-h1791-v0

e229d1b6-04ad-4a9d-98e9-23de0523bf16

CHAPTER 8. MULTISCALE VISUALIZATION 117

Figure 8.8: The zoom pattern for the “Thematic Map” pattern and a series of screenshots of a multiscale visualization of the population of the USA (CENSUS) developed using the pattern. The initial view is at the state level of detail, with each state colored by population density. As the user zooms in, with the x and y dimensions lock-stepped together, the visualization changes data abstraction, drilling down to the county level of detail. As the user zooms in further, the visual abstraction changes as layers are added to display more details: both the county name and population values are displayed as text.

y-axis to introduce a new level of detail, and we animate the transition by growing one chart before

breaking it into multiple charts, and similarly animate the x-zoom by growing a bar before showing

https://lh3.googleusercontent.com/notebooklm/AG60hOr8tQJHyYgrwdQGvhUujJskTCwYATCYmaIBJUWpGvQcw5qpAirY2j_fkrFci7rqzb4bs-NIRoJMlmpT96WbMOHxBENa_Sx6o5DCx0VQ-4lqdaTvs6E4IYZzsYoEJ5umYIdjwPtzdQ=w1794-h1791-v0

68c7b122-3538-4adb-89b6-0b06757fa582

CHAPTER 8. MULTISCALE VISUALIZATION 118

Figure 8.9: Pattern 3 and a series of screenshots of a multiscale visualization of average sales versus average profit over a two-year period for a coffee shop chain (COFFEE). In the first visualization, each point represents profit and sales for a particular month and product, summed over all locations. In the next visualization, the user zooms, changing the data abstraction: points that were originally aggregated over all locations are now broken down by market, resulting in four points for every original point. As the user zooms in further, the visual abstraction changes as layers are added to display more details: each point is colored according to market and a text label is added to redundantly encode the market name.

https://lh3.googleusercontent.com/notebooklm/AG60hOpKKhUB4UJfeHnvKrd9CedjUeaO9GkeGNuBbx8PQuXafzUrXm3x0eETNVFr00MoKplyBUtFaC4gxAHfnKcT5R4UfptWGaUvkFQZ5aMgm_DzbDozAV3A6DvPONLg1u_7t92VXsLxow=w1843-h1797-v0

6cf75f2e-f1a6-437e-9264-5c3830351ac1

CHAPTER 8. MULTISCALE VISUALIZATION 119

Figure 8.10: The “Matrices” pattern and a series of screenshots of a multiscale visualization of yeast microarray data (GENE) developed using the pattern. The first visualization shows the highest level gene clusters on the y-axis, the microarray experiment clusters on the x-axis, and the average gene expression in each cell. In the next visualization, the user zooms on both axes to show more detailed information for both gene and array clusters. In the final visualization, the user has zoomed to show the original measurements for each gene in each microarray experiment.

its breakdown.

Figure 8.7 shows how we applied the chart stack pattern to kernel lock events collected from

a simulation of a parallel graphics application (ARGUS). In this application, rather than using line

CHAPTER 8. MULTISCALE VISUALIZATION 120

charts, we use both strip charts and Gantt charts. As the user zooms, not only does the data ab-

straction change, but the visual abstraction changes from using high-level strip charts summarizing

across time to using detailed Gantt charts depicting individual events.

8.4.5 Pattern 2: Thematic Maps

This pattern is applicable when visualizing geographically-varying dependent measures that can be

summarized at multiple geographic levels of detail (such as county or state). Thus, the data contains

an ordinal dimension hierarchy that characterizes the geographic levels of detail, two independent

spatial dimensions (e.g., latitude and longitude), and some number of dependent measures. Exam-

ples of this type of data are census or election data. Typically, this type of data is visualized as a

map with measures encoded in the color of the area features or as glyphs layered on the map.

Unlike the previous pattern, where the user could zoom independently on x and y, in this pattern,

the user must zoom on both simultaneously. Thus, zooming in this pattern is like a fly-through: as

the viewer zooms, more detail is displayed. There are two types of zooms in this pattern: the

data abstraction can change by changing the specification’s internal level of detail or the visual

abstraction can change by adding details in additional layers. The zoom graph for this pattern is

shown in Figure 8.8.

To illustrate this pattern, we show in Figure 8.8 a series of zooms on a thematic map where

the measure of interest is population density (CENSUS). In this example, the user can zoom in by

moving the mouse up or zoom out by moving the mouse down. As the user zooms in, the map zooms

in; when a pre-determined elevation is reached, the script switches to a different specification, i.e.,

a different node in the zoom graph.

8.4.6 Pattern 3: Quantitative Scatterplots

This pattern (inspired by the types of visualizations created in DataSplash [79]) is very similar

to the previous pattern in that the main visualization again has two quantitative axes. However,

the primary distinction between the two patterns is that in this pattern, the axes have no inherent

mapping to the physical world; instead, they spatially encode an abstract quantity, freeing many

constraints imposed in the previous pattern. Thus, the data used in this type of visualization can

be any set of abstract measurements that can be categorized according to some set of hierarchies.

Many corporate data warehouses fall into this category.

Like the previous example, there are two types of zooms in this pattern. The data abstraction

CHAPTER 8. MULTISCALE VISUALIZATION 121

can change by either adding or removing fields or changing the level of detail of the fields listed

in the internal level of detail portion of the specification. Changing this portion of the specification

changes the number of tuples, thus changing how many marks are displayed.

Alternatively, the visual abstraction can change, either by adding retinal encodings to the current

layers or by adding information in additional layers. Note that while the map pattern must keep a

layer with a polygonal mark, this pattern has considerably more flexibility. The zoom graph for this

pattern is shown in Figure 8.9.

To illustrate this pattern, we use constructed data from a hypothetical chain of coffee shops

(example data set X). A multiscale visualization of this data set is shown in Figure 8.9.

8.4.7 Pattern 4: Matrices

Our final pattern, motivated by Abello’s work in visualizing call density [1], applies when the an-

alyst is exploring how a dependent measure varies with the values of two independent dimension

hierarchies. This type of data can be effectively visualized as a table, where the rows encode one

hierarchy. columns encode a different hierarchy, and a glyph in each cell depicts the measure.

Zooming in this graph involves either aggregating rows (or columns) or breaking a single row

(or column) down into multiple rows (or columns). In other words, the zooms are changes in the

data abstraction: the user can change the level of detail requested on either the x- or y-axis (by

applying the dot operator), either independently or together. The zoom graph for this pattern is

shown in Figure 8.10.

One type of data that fits this type of display particularly well is DNA microarray data (example

data set X), where a series of microarray experiments are performed, each experiment measuring

the expression level of different genes. A visualization for this data based on this pattern is shown

in Figure 8.10.

8.5 Discussion and summary

This chapter presents (1) a formalism for describing multiscale visualizations of data cubes with

both data and visual abstraction, and (2) a method for independently zooming along one or more

dimensions by traversing a zoom graph with nodes at different levels of detail. As an example of

how to design multiscale visualizations using our system, we describe four design patterns using

our formalism. These design patterns show the effectiveness of multiscale visualization of general

relational databases.

CHAPTER 8. MULTISCALE VISUALIZATION 122

One of the key insights behind the system is the importance of performing both data and visual

abstraction using general mechanisms, especially since many of the multiscale design patterns rely

heavily on data abstraction. Data cubes are a commonly accepted method for abstracting and sum-

marizing relational databases, much like how wavelets are used to abstract continuous functions. By

representing the database with a data cube, we can switch between different levels of detail using a

general mechanism applicable to many different data sets. Previous multiscale visualization systems

performed data generalization using special-purpose mechanisms, and hence are only applicable to

their specific domain.

In the remainder of this section, we discuss the issue of more sophisticated hierarchies, such as

branching hierarchies.

8.5.1 Multiple hierarchies

Data cubes are inherently multidimensional, with each dimension modeled by a hierarchy defining

the levels of detail to use when aggregating the base fact table. In the simple case, these hierarchies

are simple, uniform, and non-branching, so that there is only a single way to define the levels of

detail for any particular dimension, i.e., a single path for zooming along that dimension. This type

of data is commonly modeled using a star schema and is the type of data we have used in this paper

to show how users can independently zoom on multiple hierarchies within a single visualization by

associating hierarchies with the axes of a visualization.

Data warehouses can also contain intersecting, non-uniform, or branching hierarchies that can

be modeled by snowflake schemas or directed acyclic graphs. In other words, for a given dimension,

there are multiple ways to define the levels of aggregation. We can develop multiscale visualizations

on these hierarchies by choosing a single path through the branching hierarchy (i.e., superimposing

a star schema view on a subset of a schema, which is always possible) and then constructing a zoom

graph. Although this method supports branching hierarchies, it restricts the zooms to always follow

a single linear path, chosen a priori, within the branching hierarchy. The key thing to note is that

it is not our formalism nor zoom graphs that restricts us to a single linear path, but rather the user

interface for determining which branch to follow. A zoom graph can be arbitrarily sophisticated,

allowing for different representations along the different branches of the hierarchies, or even pivots

to different dimensions. The main question is finding a user interface technique that enables the

user to select an explicit branch to follow, or even to change the hierarchy they are zooming on.

Furthermore, zoom graphs can model zooming visualizations where the zooming is not tied to an

CHAPTER 8. MULTISCALE VISUALIZATION 123

Within the field of information visualization, several researchers have focused on developing vi-

sualizations for presenting and exploring multiple hierarchies. These multiple hierarchies are often

called “polyarchies” or “multitrees”. North et al. presented a system [24] for exploring a simplified

polyarchy scheme that can be modeled in data cubes and is addressable with our formalism. Robert-

son et al. have described more complex polyarchies [54]; however, these polyarchies are not a basis

for aggregation. Instead, they reflect an organization among the nodes with each node representing

a distinct data entity. With this type of polyarchy, the concern is not leveraging the hierarchies for

abstraction, but rather communicating the multiple hierarchies and their intersections.

## Conclusion

The goal of this dissertation is to present a formal basis for the specification, and generation of

graphic presentations of multidimensional data. We have presented the Polaris formalism, which

uses succinct visual specifications to describe table-based visualizations of multidimensional data.

Each table consists of layers and panes, and each pane may be a different graphic. The formalism

is capable of describing a very wide range of 2D graphic displays. The innovative aspects of the

Polaris formalism are:

a table algebra that captures the structure of tables and graphs as succinct algebraic expres-

a graphic taxonomy that results in an intuitive specification of graphic types using the algebra,

a system for effective visual encoding, and

the ability to compile the visual specifications into both graphics and database queries.

In addition to being a useful theoretical tool, an important contribution of this formalism is that

it has been used to engineer two interactive visualization systems. The first interactive system we

presented is the Polaris interface for the exploration of multidimensional databases. There are many

advantages to developing interactive systems on top of the Polaris formalism. As a result of building

the Polaris interface on the Polaris formalism, we gained the following benefits:

the interface is simple and expressive,

the interface can be used to quickly generate a wide range of displays,

CHAPTER 9. CONCLUSION 125

the database queries are implicitly generated by the specification of the desired graphic,

interface operations map to operators in the formalism and thus have consistent and intuitive

semantics.

operations needed to support exploratory analytics, such as undo and redo, are simply imple-

The Polaris interface directly supports the cycle of analysis. Analysts can incrementally create

sophisticated visualizations by using simple drag-and-drop operations to construct a visual speci-

fication. Because all intermediate specifications are valid and can be interpreted to create visual-

izations, analysts receive visual feedback as they assemble and alter the specification. Furthermore,

because all specifications can be compiled into the necessary database queries, the analyst can focus

on specifying the display they want to see rather than how to retrieve the necessary tuples.

In addition to the Polaris interface, we have also demonstrated how to use the Polaris formalism

to specify and implement domain specific multiscale visualizations. Using the Polaris formalism to

specify multiscale visualizations enabled us to address several limitations in the current approaches,

including allowing multiple zoom paths into the data and providing general mechanisms for data

and visual abstraction. A final contribution of this dissertation was the introduction of the concept

of design patterns. Using design patterns, designers can capture the structure of effective multiscale

visualizations and reuse that structure in new problem domains and analysis. The Polaris formal-

ism is a necessary component of design patterns; articulating a pattern requires a mechanism for

precisely describing a visualization’s structure.

We hope the formalism presented in this dissertation will both be extended and used by re-

searchers in the visualization community. The next section describes several interesting areas of

future research that can build on the results we have presented.

9.1 Future work

9.1.1 Extending the formalism

The Polaris formalism is capable of describing a wide range of traditional, 2D statistical displays.

However, if extended in several ways, it could evolve into a more complete formalism for graphic

communication of relational data. There are many ways that the formalism could be extended.

One clear extension would be to support 3D graphics, possibly done by including another axis

CHAPTER 9. CONCLUSION 126

expression or redefining the interpretation of the z-axis expression (e.g., quantitative expressions

on the z-axis). Another extension of the formalism would be to support the layout of trees, either

within panes or as part of the algebra. Including an animation shelf in addition to the x,y, and layer

shelves would enable analysts to partition the data on those fields and create animated displays that

sequence through the data. For example, in the coffee chain data set, dropping the Month field on

the animation shelf would create an animation showing how the data changes over time.

Another area of future work is to leverage the direct correspondence of graphical marks in

Polaris to tuples in the relational databases in order to generate database tables from a selected set

of graphical marks. This technique can be used to develop lenses, similar to the Magic Lens, that

can perform much more complex transformations because they operate in data space rather than

image space. This technique can also be used to compose Polaris displays, using a selected mark

set in one display as the data input to another. We are exploring these techniques and believe it is

possible to develop a relational spreadsheet by composing Polaris displays in this manner.

9.1.2 Multiscale visualizations and design patterns

There is also a considerable amount of interesting work to be done in further developing the concepts

of zoom graphs and design patterns. Given the Polaris formalism and data cubes, it was relatively

easy to construct a visualization of a particular level of detail in a hierarchical database. It turns

out to be much more complicated, however, to construct continuous zooms into the data. There are

many ways to zoom, many of which are not very effective. In this dissertation, we have described

four fairly simple patterns that are effective and that we have used in several applications. These

patterns were motivated by previous work, but are still quite general. Many incremental extensions

of these patterns are possible. For example, the thematic map pattern should work whenever there is

a fixed mapping between the spatial encodings and the physical world. Another example is the chart

stack pattern, in which we showed a version using line charts and a variation using a different chart

type (strip charts and Gantt charts). Other chart types, such as histograms, would work equally well.

Other possible extensions include embedding one pattern within another. There are also completely

different patterns that also might work. Developing an extensive repository of zoom graphs would

be another good direction for future work.

Another critical issue when designing a zooming interface is in making natural transitions be-

tween levels of detail, requiring visualizations to clearly communicate the parent-child relationships.

We have found visual cues such as color and padding to be effective in indicating the hierarchy. An-

other difficulty in transitioning between different views occurs when using categorical hierarchies

CHAPTER 9. CONCLUSION 127

with non-uniform branching factors. This situation means that more space is needed to zoom into

some nodes than others. We have explored several transition mechanisms, including animating the

transition and gradually fading between the two views to avoid the disconcerting “popping” that can

9.1.3 Usability studies

In this dissertation, we have presented two interactive systems. Based on our experience using this

systems, and the informal experience of a small group of users, we believe that these interfaces

present a powerful visual interface to multidimensional databases. However, to truly assess the

design decisions made in the design of these systems, and to assess the utility of the systems, we

need to perform formal user studies.

There are many important questions that a usability analysis would answer. Examples include:

Do the interfaces support the analysis cycle?

Do the users generate effective displays using the interface?

Do users take advantage of the expressiveness of the formalism?

In addition to performing usability analysis of the existing systems we have built, we would like

to develop additional systems that leverage the Polaris formalism. The presented systems are only

one possibly interface to the Polaris formalism.

9.1.4 Data management

A final area of future work is to develop the systems infrastructure so that very large datasets may be

visualized in real-time. Database queries generated by visualization systems typically have several

unique properties that could be leveraged to improve performance:

There is coherence between successive queries,

The queries are expensive to compute,

The queries generate large result sets,

Approximate answers are often acceptable.

CHAPTER 9. CONCLUSION 128

Given these properties, there are several promising techniques for improving visualization’s us-

age of databases. The first possibility is to incorporate prefetching and caching into the visualization

system. This enhancement would require good models of user behavior. Another possibility is to

use data sampling techniques to choose what to draw [39] and importance metrics to determine

in what order to draw the data [51]. Finally, if databases can stream the query results as they are

computed, the visualization system could display a continually refining display, allowing the user

to stop the query when they have received enough information to validate their hypothesis without

executing the entire query.

The Polaris formalism is designed to be used to visualize data cubes, star schemas, or denor-

malized relations. A final area of future work would be to extend the formalism to support the

visualization of sets of joined relations. Currently, many normalized databases can be visualized in

Polaris by generating a denormalized view of some subset of the relations. A future version of the

formalism could support explicitly specified or implicitly derived join conditions between relations

making it possible to directly include fields from different relations in the same visual specification.

## Bibliography

[1] J. Abello and J. Korn. MGV: A System for Visualizing Massive Multidigraphs. In IEEE Trans-

actions on Visualization and Computer Graphics, 8(1), January 2002, pp. 21-38.

[2] R. Becker, W. Cleveland, and M. Shyu. The Visual Design and Control of Trellis Display. In

Journal of Computational and Statistical Graphics, (5), 1996, pp. 123-155.

[3] B. Bederson, J. Hollan, K. Perlin, J. Meyer, D. Bacon, and G. Furnas. Pad++: A Zoomable

Graphical Sketchpad for Exploring Alternate Interface Physics. In Journal of Visual Languages

and Computing, 7, 1996, pp. 3-31.

[4] B. Bederson, J. Meyer, and L. Good. Jazz: An Extensible Zoomable User Interface Graphics

Toolkit in Java. In Proceedings of UIST, 2(2), 2000, pp. 171-180.

[5] A. Berson and S. Smith. Data Warehousing, Data Mining, and OLAP. McGraw-Hill, New

York, 1997.

[6] J. Bertin. Graphics and Graphic Information Processing. Walter de Gruyter, Berlin, 1980.

[7] J. Bertin. Semiology of Graphics: Diagrams, Networks, Maps. University of Wisconsin Press,

[8] R. Bosch, C. Stolte, G. Stoll, M. Rosenblum, and P. Hanrahan. Performance Analysis and

Visualization of Parallel Systems Using SimOS and Rivet: A Case Study. In Proceedings of the

Sixth IEEE International Symposium on High-Performance Computer Architecture, January

[9] R. Bosch, C. Stolte, D. Tang, J. Gerth, M. Rosenblum, and P. Hanrahan. Rivet: A Flexible

Environment for Computer Systems Visualization. In Computer Graphics, 34(1), February

BIBLIOGRAPHY 130

[10] R.M. Boynton. Eleven Colors That Are Almost Never Confused. In SPIE Proceedings: Human

Vision, Visual Processing, and Digital Display, 1989, Vol. 1077, pp. 322-332.

[11] C. Brewer. Color Use Guidelines for Mapping and Visualization, Chapter 7 (pp. 123-147) in

Visualization in Modern Cartography, edited by A.M. MacEachren and D.R.F. Taylor, Elsevier

Science, Tarrytown, NY, 1994.

[12] C. Brewer, Guidelines for Use of the Perceptual Dimensions of Color for Mapping and Vi-

sualization. In Color Hard Copy and Graphic Arts III, edited by J. Bares, Proceedings of the

International Society for Optical Engineering (SPIE), San Jose, February 1994, Vol. 2171, pp.

[13] C. Brewer. Color Use Guidelines for Data Representation. In Proceedings of the Section on

Statistical Graphics, ASA, 1999, pp. 55-60.

[14] A. Buja, D. Cook, and D. F. Swayne. Interactive High-Dimensional Data Visualization. In

Journal of Computational and Graphical Statistics, 5(1), 1996, pp. 78-99.

[15] S. Card and J. Mackinlay. The Structure of Information Visualization Design Space. In Pro-

ceedings of the IEEE Symposium on Information Visualization, 1996, pp. 92-99.

[16] S. Card, J. Mackinlay, and B. Shneiderman. Readings in Information Visualization, Morgan

Kaufmann, 1999.

[17] J.L. Caviano. Visual texture as a semiotic system. In Semiotica, 80(3/4), 1990, pp. 239-252.

[18] E. Chi. A Framework for Information Visualization Spreadsheets. Ph.D. Thesis. University of

Minnesota, Computer Science Department, March 1999.

[19] W.S. Cleveland and R. McGill. Graphical perception: Theory, experimentation, and applica-

tion to the development of graphical methods. In Journal of the American Statistical Associa-

tion 79, 1984, pp. 387.

[20] W.S. Cleveland. The Elements of Graphing Data. Wadsworth Advanced Books and Software,

Pacific Grove, California, 1985.

[21] W.S. Cleveland. Visualizing Data. Hobart Press, New Jersey, 1993.

[22] W.S. Cleveland. A Model for Studying Display Methods of Statistical Graphics (with discus-

sion). In Journal of Computational and Statistical Graphics, 3, 1993, pp. 323-364.

BIBLIOGRAPHY 131

[23] E.F. Codd. A Relational Model for Large Shared Data Banks. In Communications of the ACM,

13(6), 1970.

[24] N. Conklin, S. Prabhakar, and C. North. Multiple Foci Drill-Down through Tuple and At-

tribute Aggregation Polyarchies in Tabular Data. In Proceedings of the IEEE Symposium on

Information Visualization, October 2002.

[25] M. Derthick, J. Kolojejchick and S. F. Roth. An Interactive Visualization Environment for Data

Exploration. In Proceedings of Knowledge Discovery in Databases (KDD), August, 1997, pp.

[26] S. Eick and A. Karr. Visual Scalability. In Journal of Computational and Graphical Statistics,

11(1), March 2002, pp. 22-43.

[27] M. Eisen. Cluster and Treeview. http://rana.lbl.gov, cited Mat 2003.

[28] M. Eisen, P. Spellman, P. Brown, and D. Botstein. Cluster analysis and display of genome-

wide expression patterns. In Proceedings of the National Academy of Sciences, USA (95),

1998, pp. 14863-8.

[29] Feiner, S. and Beshers, C. Worlds within Worlds: Metaphors for Exploring n-Dimensional

Virtual Worlds. Proceedings of ACM UIST, 1990, pp. 76-83.

[30] A. Frank and S. Timpf. Multiple representations for cartographic objects in a multi-scale tree–

an intelligent graphical zoom. In Computers and Graphics Special Issue: Modelling and Visu-

alization of Spatial Data in Geographical Information Systems, 18(6), 1995, pp. 823-830.

[31] W.S. Freeze. Unlocking OLAP with Microsoft SQL Server and Excel 2000. IDG Books

Worldwide, Inc., Foster City, California, 2000.

[32] Y. Fua, M. Ward, and E. Rundensteiner. Structure-based Brushes: A Mechanism for Navigat-

ing Hierarchically Organized Data and Information Spaces. In IEEE Transactions on Visual-

ization and Computer Graphics, June 2000.

[33] E. Gamma, R. Helm, R. Johnson, and J. Vlissides. Design Patterns: Elements of Reusable

Object-Oriented Software. Reading, MA: Addison-Wesley, 1995.

[34] P. Gibbons, Y. Matias, and V. Poosala. Aqua Project White Paper. Tech Report, Bell Labora-

tories, Murray Hill, New Jersey, Dec. 1997.

BIBLIOGRAPHY 132

[35] J. Goldstein, S. F. Roth, J. Kolojejchick, and J. Mattis. A Framework for Knowledge-based

Interactive Data Exploration. In Journal of Visual Languages and Computing, December 1994,

pp. 339-363.

[36] J. Hartigan. Printer graphics for clustering. In Journal of Statistical Computation and Simula-

tion, (4), pp. 187-213.

[37] C. Healey. Choosing Effective Colours for Data Visualization. In Proceedings of IEEE Visu-

alization, 1996, pp. 263-270.

[38] C. Healey and J. Enns. Building Perceptual Textures to Visualize Multidimensional Datasets.

In Proceedings of IEEE Visualization, 1998, pp. 111-118.

[39] J. Hellerstein, R. Avnur, A. Chou, C. Hidber, C. Olston, V. Raman, T. Roth, and P. Haas.

Interactive Data analysis: The CONTROL Project. IEEE Computer, August 1999, p. 51-59.

[40] R. Hightower, L. Ring, J. Helfman, B. Bederson, and J. Hollan. Graphical Multiscale Web

Histories: A Study of PadPrints. In Proceedings of ACM Conference on Hypertext, 1998, pp.

[41] Human Genome Project. [online] Available: http://www.ornl.gov/hgmis/about.html, cited

February 2002.

[42] H. Igehy, G. Stoll, and P. Hanrahan. The Design of a Parallel Graphics Interface. In Proc. ACM

SIGGRAPH 1998, August 1998, p. 141-150.

[43] W. Inmon. Definition of a Data Warehouse. White paper available at

http://www.billinmon.com/, cited May 2003.

[44] D. Keim and H.P. Kriegel. VisDB: Database Exploration using Multidimensional Visualiza-

tion. In IEEE Computer Graphics and Applications, 14(5), 1994, pp. 40-49.

[45] S.M. Kosslyn. Elements of Graph Design. W.H. Freeman and Co., New York, NY, 1994.

[46] R. Lipton, J. Naughton, D. Schneider, and S. Seshardri. Efficient sampling strategies for rela-

tional database operations. In Theoretical Computer Science, 116(1-2), 1993, pp. 195-226.

[47] A. MacEachern. How Maps Work: Representation, Visualization, and Design. The Guilford

Press, New York, 1995.

BIBLIOGRAPHY 133

[48] J. Mackinlay. Automating the Design of Graphical Presentations of Relational Information. In

ACM Transactions on Graphics, April 1986, pp. 110-141.

[49] Excel User’s Manual. Microsoft Press, Redmond, WA., 2000.

[50] J.L. Morrison. A theoretical framework for cartographic generalization with the emphasis on

the process of symbolization. In International Yearbook of Cartography, 14, 115-127.

[51] T. Munzner. Drawing Large Graphs with H3Viewer and site Manager. In Proc. of Graph Draw-

ing 1998, August 1998, p. 384-393.

[52] M. Livny, R. Ramakrishnan, K. Beyer, G. Chen, D. Donjerkovic, S. Lawande, J. Myllymaki

and K. Wenger. DEVise: Integrated Querying and Visual Exploration of Large Datasets. In

Proceedings of ACM SIGMOD, May, 1997.

[53] R. Rao and S. Card. The Table Lens: Merging Graphical and Symbolic Representations in

an Interactive Focus+Context Visualization for Tabular Information. In Proceedings of ACM

SIGCHI, 1994, pp. 318-322.

[54] G. Robertson, K. Cameron, M. Czerwinski, and D. Robbins. Polyarchy Visualization: Visual-

izing Multiple Intersecting Polyarchies. In Proceedings of ACM SIGCHI 2002, April 2002, p.

[55] B. Rogowitz, and L. Treinish. How NOT to Lie with Visualization. In Computers in Physics,

May/June 1996, pp. 268-274.

[56] S.F. Roth and J. Mattis. Data Characterization for Intelligent Graphics Presentation. In Pro-

ceedings of the Conference on Human Factors in Computing Systems (SIGCHI’90), Seattle,

WA, April 1990, pp. 193-200.

[57] S.F. Roth, J. Kolojejchick, J. Mattis and J. Goldstein. Interactive Graphic Design Using Auto-

matic Presentation Knowledge. In Proc. of SIGCHI ’94, April 1994, pp. 112-117.

[58] S.F. Roth, P. Lucas, J.A. Senn, C.C. Gomberg, M.B. Burks, P.J. Stroffolino, J. Kolojejchick and

C. Dunmire. Visage: A User Interface Environment for Exploring Information. In Proceedings

of the IEEE Symposium on Information Visualization, October 1996, pp. 3-12.

[59] E. Rundensteiner, M. Ward, J. Yang, and P. Doshi. XmdvTool: Visual Interactive Data Explo-

ration and Trend Discovery of High-dimensional Data Sets. In Proc. ACM SIGMOD 2002,

BIBLIOGRAPHY 134

[60] B. Shneiderman. Tree visualization with treemaps: a 2-d space-filling approach. In ACM

Transactions on Graphics, 11(1), pp. 92-99.

[61] http://www.cs.umd.edu/hcil/research/visualization.shtml, cited May 2003.

[62] Sloan Digital Sky Survey. [online] Available: http://www.sdss.org/, cited May 2003.

[63] M. Spenke, C. Beilken, and T. Berlage. FOCUS: The Interactive Table for Product Compar-

ison and Selection. In Proceedings of the ACM Symposium on User Interface Software and

Technology (UIST), November 1986.

[64] Spotfire Inc. [online] Available: http://www.spotfire.com, cited February 2002.

[65] S. Stevens. On the theory of scales of measurement. In Science, (103), pp. 677-680.

[66] S. Stevens. To honor Fechner and repeal his law. In Science, (133), pp. 80-86.

[67] D. Tang. Analyzing Wireless Networks. Ph.D. Dissertation. Stanford University, Computer

Science Department, October 2000.

[68] E. Thomsen. OLAP Solutions: Building Multidimensional Information Systems. Wiley Com-

puter Publishing, New York, 1997.

[69] S. Timpf. Map Cube Model–a model for multi-scale data. In Eighth International Symposium

on Spatial Data Handling, 1998, pp. 190-201.

[70] F. Topfer and W. Pillewizer. The principles of selection, a means of cartographic generaliza-

tion. In Cartographic Journal, 3(1), 1966, pp. 10-16.

[71] D. Travis. Effective Color Displays: Theory and Practice. Academic Press, London, 1991.

[72] E. Tufte. The Visual Display of Quantitative Information. Graphics Press, Cheshire, Connecti-

cut, 1983.

[73] J. Ullman and J. Widom. A First Course in Database Systems. Prentice Hall, New Jersey, 2001.

[74] J. Ullman. Efficient Implementation of Data Cubes via Materialized Views. In Proceedings of

Knowledge Discovery in Databases (KDD), 1996.

[75] H. Wainer and C.M. Francolini. An empirical inquiry concerning human understanding of

two-variable color maps. In The American Statistician, 34, 1980, pp. 81-93.

BIBLIOGRAPHY 135

[76] C. Ware. Information Visualization: Perception for Design. Morgan Kaufman Publishers,

[77] L. Wilkinson. The Grammar of Graphics. Springer, New York, New York, 1999.

[78] L. Wilkinson, D.J. Rope, D.B. Carr, and M.A. Rubin. The language of statistical graphics. In

Journal of Computational and Graphical Statistics, 9, 2000, pp. 530-543.

[79] A. Woodruff, C. Olston, A. Aiken, M. Chu, V. Ercegovac, M. Lin, M. Spalding, and M.

Stonebraker. DataSplash: A Direct Manipulation Environment for Programming Semantic

Zoom Visualizations of Tabular Data. J. of Visual Languages and Computing, Special Issue

on Visual Languages for End-user and Domain-specific Programming, 12(5), October 2001,

pp. 551-571.

