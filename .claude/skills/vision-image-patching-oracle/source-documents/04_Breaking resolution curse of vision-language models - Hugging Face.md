---
sourceFile: "Breaking resolution curse of vision-language models - Hugging Face"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:16.559Z"
---

# Breaking resolution curse of vision-language models - Hugging Face

cdbcf2c1-5c95-4106-8a46-d1946f2a1112

Breaking resolution curse of vision-language models - Hugging Face

a16590c3-a344-4787-b723-0331d7014299

https://huggingface.co/blog/visheratin/vlm-resolution-curse

## Hugging Face

## Enterprise

## Back to Articles

Breaking resolution curse of vision-language models

## Community Article

Published February 24, 2024

Upvote  21

https://huggingface.co/login?next=%2Fblog%2Fvisheratin%2Fvlm-resolution-curse

## Alexander Visheratin   visheratin

https://huggingface.co/login?next=%2Fblog%2Fvisheratin%2Fvlm-resolution-curse

In the post, I describe the resolution problem that modern vision-language models (VLMs) face and explore a new approach to solving it using multiple crops of a high-resolution image. The demo is

https://huggingface.co/spaces/visheratin/mc-llava-3b

if you want to try how it works. The model is available in the

https://huggingface.co/visheratin/MC-LLaVA-3b

VLMs are fantastic. They allow us to extract a lot of different information from the images - general descriptions, answers to questions, and even bounding boxes of the objects in the image. Today, the primary way of building VLMs is by bringing the image features from the pretrained vision model into the language model, usually a large one (LLM). You can find this approach in

https://arxiv.org/abs/2301.12597

https://llava-vl.github.io/

https://arxiv.org/abs/2310.09199

https://arxiv.org/abs/2304.10592

, and many other models. This way, you train some adapter to align image features with the text embedding space, which enables you to treat modified image features effectively as text embeddings. For example, below is the principal structure of the PaLI-3 model.

There are alternatives to that. For example,

https://arxiv.org/abs/2311.03079

has visual expert modules that mirror LLM modules and are activated only in the presence of the image. This preserves pure text reasoning while having great performance in multimodal tasks.

Adept took another path and developed a single decoder-only transformer -

https://www.adept.ai/blog/fuyu-8b

- that processes images and text together. It greatly simplifies the architecture and brings images and text into a truly unified space.
There is one crucial problem with VLMs based on the pre-trained vision encoder - resolution. Consider the following example from the recently released V* benchmark. Try to answer the following question: "Based on that advertisement board, can you tell what type of shop is in the image?"

Of course, you can't. The image is too small. But in reality, the image has a resolution of 4240x2832 pixels, and you can easily zoom in to read the text. But VLMs can't zoom! They are limited by the resolution of the vision encoder, and usually, it is not super large. For example, here is the same image resized to 384x384 pixels, which is one of the standard resolutions.

But is it really a problem? Let’s see if state-of-the-art VLMs can answer the question. Below is the CogVLM output:

And here is the Fuyu-8B answer:

Output of LLaVA-1.6 34B:

And, finally, GPT-4V via ChatGPT:

Now that we understand that this is indeed a problem, can we do something about it?

## Solutions

## Visual search

## Penghao Wu and Saining Xie proposed

https://arxiv.org/abs/2312.14135

(Show, SEArch, and TelL) meta-framework that brings visual search similar to how people search for details in large scenes. Equipped with visual working memory, the model can reason its way through the image by processing smaller parts of the image until it finds all the information needed to answer the question. Because of that, SEAL outperforms all existing models, including GPT-4V, in answering questions about small details in extra-large images. For example, it can tell you what kind of shop is in that image.

## Visual cropping

Another approach to solving the resolution issue, among other things, is by reducing the analyzed area. Recently, Jiarui Zhang et al.

https://arxiv.org/abs/2310.16033

that by tracking the model’s attention, you can find the most relevant parts. And if you crop the image to these parts, you can boost the model's performance by up to 10% in some cases.

Can we combine ideas from these two methods and make the model perform a sort of visual search using only attention in one go?

Usually, in LLaVA models, we generate N embeddings for the image, which we then combine with text embeddings and send to the LLM. But what if instead of creating

tokens for one image, we create

tokens for

parts of the image (crops)? It would allow us to get visual information from small parts of the image and not inflate the number of image "tokens" too much. I called this method multi-crop LLaVA (MC-LLaVA).

Similarly to vision transformers (ViTs), we split the image into multiple crops aligned on a grid. But unlike in ViTs, crops in MC-LLaVA are overlapping. The simple explanation is that it just works better. The current hypothesis is that overlapping crops enable the dispersing of visual information from the same region across multiple embeddings and compensate for selecting only

embeddings instead of

Since the input images may have different aspect ratios, we would lose the positional information about the crops if we used only crops in the model. That is why, along with the crop, we also extract its relative coordinates. As a result, we get a list of crops with their coordinates.

Then, we pass the crop itself through the vision encoder and extract hidden states from the last layer before the pooling. Initially, it has the shape of [729, 1152], but as in the original LLaVA paper, we discard the first feature and end up with the tensor with the shape of [728, 1152]. We then select  K  last features from the hidden state.  K  depends on the number of image crops and is roughly calculated as  int(N/M) , where  N  is the target number of image embeddings and  M  is the number of crops. As for the coordinates, we pass them through the standard multi-layer perceptron with two layers. We then add the resulting coordinate embeddings to the selected image embeddings.

After we process all crops with their coordinates, we simply concatenate their embeddings and get our target N embeddings. But this time, they contain detailed visual information from all small parts of the image! Cool, right?

The rest is standard LLaVA architecture - we pass the multi-crop representation through the adapter to align it with the embeddings of the language model.

## Experiments

But does it work? To answer this question, I trained the 3.2 billion parameters MC-LLaVA model based on the Phi-2 language model and SigLIP vision encoder. The model was trained on a variety of images with different resolutions, which means that the number of extracted features ( K ) was dynamic. It means that technically, the model not only supports an arbitrary number of crops ( M ) but also an arbitrary number of extracted tokens ( N )! So now we have two whole knobs we can turn to get the best out of the model. Let's see what it can give us in terms of performance. You can follow along in

this Colab notebook

https://colab.research.google.com/drive/1yRUMwhiQqij3f_TkYT80O_Ox5iMTT86r?usp=sharing

The model itself is compatible with Transformers auto classes, so you can create the model and processor for it with just two lines of code:

from  transformers  import  AutoModel, AutoProcessor  import  torch model = AutoModel.from_pretrained( "visheratin/MC-LLaVA-3b" , torch_dtype=torch.float16, trust_remote_code= True ).to( "cuda" ) processor = AutoProcessor.from_pretrained( "visheratin/MC-LLaVA-3b" , trust_remote_code= True )

You can adjust the number of crops per image and the number of image “tokens” when calling the processor:

with  torch.inference_mode(): inputs = processor(prompt, [raw_image], model, max_crops= 100 , num_tokens= 728 )

First things first - can we finally read what is written on that board and what does the shop do? Yes, we can! With 100 max crops and 728 image tokens, the model confidently reads “Core Power Yoga”, which is exactly what is written on the board!

Let's look at another example, where we will be able to utilize our knobs. Can the model answer where to buy a mug like this based on its logo?

If we use the settings from the previous example (100 max crops and 728 image tokens), the model fails:

The mug is from the company "Nike". You can buy it from Nike's website or from any Nike store.

This is because the image has a resolution of 3213x3024 pixels and the actual logo is very small (30x90 pixels). Let’s increase the number of crops to 200 while keeping the number of tokens.

Based on the logo, you can buy a mug like this at Starbucks. The logo is a combination of the Starbucks logo and the logo of the TV show "Star Trek".

Now we are talking! But the model generates some random stuff about Star Trek. Let’s try to remove it by doubling the number of image tokens to 1456.

The mug is from the Starbucks brand. You can buy it at any Starbucks location.

That is a proper answer. But what if it was a fluke and the model treated some other thing in the room as a Starbucks cup? I applied the most advanced image editing techniques, known as copy-paste, to hide the cup in the image.

What can the model tell us now? We use the same parameters as before - 100 max crops and 1456 image tokens.

The mug is from the Coca-Cola company. You can buy it at any Coca-Cola store or online at the Coca-Cola website.

Nice! It means that the model really attended to the cup and identified the Starbucks logo on it.

## Benchmarks

Now, to more systematic checks. Below are the numbers for four popular benchmarks. The results are quite good, more or less what you’d expect from the 3B model.

Benchmark   Accuracy   TextVQA   50.9%   GQA   59.5%   VQAv2   76.72%   VizWiz   32.68%

## The most interesting benchmark for us is the

https://huggingface.co/datasets/craigwu/vstar_bench

, where the models need to answer questions about small details in high-resolution images. Here I performed a proper grid search across both parameters available in MC-LLaVA during inference - max crops number and number of image “tokens”. On the “direct attributes” and “relative position” parts of the benchmark the model performs more or less like other larger models. But things get interesting for the other two parts. The “GPT4V-hard” is the set of 17 images, for which GPT-4V couldn’t answer the question. SEAL framework, on the other hand, nailed all of them. We can see that increasing the number of crops leads to better performance, and at its best, MC-LLaVA scores 52.94%, higher than any other VLM of larger size!

## Conclusion

Even the best VLMs (yes, even GPT-4V) suffer from the resolution curse.

Adding visual search and cropping helps them find smaller details.

We can also solve the problem by extracting features from the smaller parts of the image and using them in an LLaVA model.

MC-LLaVA shows very good results for a 3B parameter model. It is a work in progress, and if you have ideas on how to make the model better, let me know!

## Acknowledgements

https://lambdalabs.com/

for providing a machine to train the model.

## ML Collective

https://mlcollective.org/

for continuous support and for providing compute resources for testing the model.

asimabbasturi

https://mlcollective.org/

https://mlcollective.org/

https://huggingface.co/join?next=%2Fblog%2Fvisheratin%2Fvlm-resolution-curse

https://huggingface.co/login?next=%2Fblog%2Fvisheratin%2Fvlm-resolution-curse

to comment

Upvote  21

https://huggingface.co/login?next=%2Fblog%2Fvisheratin%2Fvlm-resolution-curse

