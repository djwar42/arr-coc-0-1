---
sourceFile: "Token Pooling in Vision Transformers for Image Classification"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:19.786Z"
---

# Token Pooling in Vision Transformers for Image Classification

6ea8e623-5770-4ddb-b934-91075ea52736

## Token Pooling in Vision Transformers for Image Classification

db7004e4-674e-401d-90c2-6b8d80ea755a

https://www.youtube.com/watch?v=56fykt6LrNg

56fykt6LrNg

## ComputerVisionFoundation Videos

hello my name is Dimitri Marin in this talk I will introduce talking pooling which is a pulling method designed for visual Transformers pooling is a commonly used method to improve the computational efficacy of cnns we designed a new pulley layer to improve the performance of vision Transformers let me give you a quick peek of our main results here we analyze the computation accuracy trade-off of a standard Vision Transformer by varying the number of features the x-axis is the computation in flows and the y-axis is the top one accuracy on the imagenet data set in this plot better models approach the top left corner higher accuracy with lower flops we applied our token pooling layer on deed significantly improving its performance for example we can achieve the same accuracy of the timing while using 42 percent less computation or we can choose to use the same computation while improving the accuracy by three points let's first recall the architecture of Transformers it a Transformer block is composed of multi-head attention layer and MLB outputs the same number of tokens many such blocks sequently transform the tokens keeping the number of tokens unchanged Vision Transformers are simply Transformers where the initial tokens are made of image patches to use token pooling we take a standard Transformer and add the pulling layer after each Transformer block just like the pulling layers in the cnns the pulling layer takes the input tokens and outputs fewer targets as shown on the slide this means that the next layers only need to process a smaller number of tokens so they need less computation the second token pooling will further reduce the number of tokens and so on notice but via processing fewer and fewer tokens this Compound Effect is one of the reasons why pulling layer improved computational efficiency now let's see how we design token pooling since we reduce the number of tokens we will lose information the important question is how do we select the tokens so that we can best retain the accuracy just like most load sampling algorithms we need to identify another information and try to produce this means what we should drop redundant talking one of the main contributions of our work is that we show that Transformers by construction produce redundant tokens specifically we show that the soft Max attention is a low pass filter applied on the input tokens in other words its output tokens will have similar values that can be proved we also show that tokens are discrete samples of a continuous signal this perspective is important because it allows us to apply non-uniform sampling literature the most effective tokens to keep are the tokens that minimize their construction laws therefore we decide talking pulley to minimize the Reconstruction laws here is the algorithm of talking pooling in order to Output a smaller number of tokens we first cluster and tokens into K clusters then we return the average of the tokens belonging to each cluster we found that the gamidoids approach produces the best computation accuracy trade-off this is a simple algorithm that is shown in the paper is effective this concludes my talk you can find more information and results in our paper at the link

