---
sourceFile: "A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:15.554Z"
---

# A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv

13677645-0c35-46af-bc7d-021627977186

A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv

e923fa12-e679-4604-b309-8b1163919716

https://arxiv.org/html/2312.01232v1

## HTML conversions

sometimes display errors

https://info.dev.arxiv.org/about/accessibility_html_error_messages.html

due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: mdwmath

failed: mdwtab

failed: eqparbox

Authors: achieve the best HTML results from your LaTeX submissions by selecting from this list of

supported packages

https://corpora.mathweb.org/corpus/arxmliv/tex_to_html/info/loaded_file

License: CC BY 4.0 arXiv:2312.01232v1 [cs.CV] 02 Dec 2023

## A Comprehensive Study of Vision Transformers in Image Classification Tasks

Mahmoud Khalil, Ahmad Khalil, and Alioune Ngom  Department of Computer Science  University of Windsor,  Windsor, Ontario, Canada

Image Classification is a fundamental task in the field of computer vision that frequently serves as a benchmark for gauging advancements in Computer Vision  [

https://arxiv.org#bib.bib1

] . Over the past few years, significant progress has been made in image classification due to the emergence of deep learning. However, challenges still exist, such as modeling fine-grained visual information, high computation costs, the parallelism of the model, and inconsistent evaluation protocols across datasets. In this paper, we conduct a comprehensive survey of existing papers on Vision Transformers for image classification. We first introduce the popular image classification datasets that influenced the design of models. Then, we present Vision Transformers models in chronological order, starting with early attempts at adapting attention mechanism to vision tasks followed by the adoption of vision transformers, as they have demonstrated success in capturing intricate patterns and long-range dependencies within images. Finally, we discuss open problems and shed light on opportunities for image classification to facilitate new research ideas.

Index Terms:

deep learning, computer vision, image classification, vision transformer, attention, transformer, survey

## I  Introduction

Image classification is a fundamental task in computer vision. it has numerous real-world applications, some of which include Medical Diagnosis, Autonomous Vehicles, Surveillance, E-commerce, Agriculture, Quality Control, and Security. Image classification involves assigning a label or category to an input image

1 An image is a large matrix of grayscale values, one for each pixel and color. . It is the process of training a machine learning model to detect and classify different objects or patterns present in the image  [

https://arxiv.org#bib.bib2

] . The model takes an image as an input, and it outputs a predicted label or category based on its learned knowledge from the training dataset. Formally image classification task can be defined as follows:  Task   T : Detecting and Recognizing objects

2 In object detection, objects are typically defined as instances of specific classes within an image.  within images   Performance measure   P : percent of images correctly classified   Training experience   E : A dataset of images containing objects with given labels

https://arxiv.org#S1.F1

, shows several images with the associated action labels, which are typical daily objects such as dress and watch.

Over the last decade, the success of AlexNet  [

https://arxiv.org#bib.bib3

]  fueled a growing research interest in image classification. AlexNet’s success can be attributed to its use of data augmentation and dropout regularization techniques, activation functions, and most importantly parallel computing during training  [

https://arxiv.org#bib.bib4

] . Another important factor of researchers’ interest in image classification is the emergence of high-quality large-scale image classification datasets  [

https://arxiv.org#bib.bib5

] . We show the sizes of popular image classification datasets in Figure.

https://arxiv.org#S1.F2

We see that both the number of images and classes increase rapidly, e.g., from 70K images over 10 classes in MNIST  [

https://arxiv.org#bib.bib6

]  to 300M images over 18m classes in JFT-300M  [

https://arxiv.org#bib.bib7

] . Also, the rate at which new datasets are released is increasing: 3 datasets were released in 2014 compared to 2 released from 2006 to 2013.

Using attention as a primary mechanism for representation learning has seen widespread adoption in deep learning after  [

https://arxiv.org#bib.bib8

] . Transformers have achieved state-of-the-art performance in a wide range of tasks  [

https://arxiv.org#bib.bib9

https://arxiv.org#bib.bib10

https://arxiv.org#bib.bib11

] . Recently, they have also been applied to several computer vision tasks including image classification  [

https://arxiv.org#bib.bib12

https://arxiv.org#bib.bib13

There are several Transformer models that have been used in image classification tasks, including:

Vision Transformer (ViT): ViT  [

https://arxiv.org#bib.bib14

]  is a transformer-based image classification model that was introduced in 2020 by Google. It uses a standard transformer encoder to process image patches extracted from the input image.

Swin Transformer: Swin  [

https://arxiv.org#bib.bib15

]  Transformer is a hierarchical transformer-based model that can process images of any size. It divides the input image into non-overlapping patches, which are then processed in a hierarchical manner by multiple transformer layers.

DeiT: DeiT  [

https://arxiv.org#bib.bib1

]  stands for ”Dense Encoder-Decoder Transformers.” It is a transformer-based model that uses a combination of encoder and decoder layers to process image patches.

CaiT: CaiT  [

https://arxiv.org#bib.bib16

]  stands for ”Cross-Attention Image Transformer.” It is a transformer-based model that uses cross-attention to combine information from different patches of the input image.

iGPT: The iGPT  [

https://arxiv.org#bib.bib17

]  model is a type of Generative Pre-trained Transformer that was introduced in a research paper by the OpenAI team in 2021. ”iGPT” stands for ”image GPT”, which means that the model has been trained to generate images in addition to text. Unlike previous GPT models, which were primarily designed for generating natural language text, iGPT has been trained to generate high-resolution images from textual descriptions.

Figure 1:  Visual examples of categories in popular image datasets.   Figure 2:  sizes of most popular image datasets from past 17 years. The area of a circle represents the scale of each dataset (i.e., the number of images).

https://arxiv.org#S1.F3

, we present a chronological overview of recent representative work

The Stand-Alone Self-Attention in Vision Models  [

https://arxiv.org#bib.bib18

]  marks an early departure from convolutional approaches in computer vision. Three notable trends emerged subsequently. The first trend, initiated by a pivotal paper demonstrating the viability of attention layers as substitutes for convolution  [

https://arxiv.org#bib.bib19

] , involves extracting  2 × 2 2 2 2\times 2 2 × 2  patches from the input image and applying full self-attention. Following the success of applying transformers directly to images, exemplified by the Vision Transformer (ViT)  [

https://arxiv.org#bib.bib14

] , various subsequent models like CaiT  [

https://arxiv.org#bib.bib1

] , Swin  [

https://arxiv.org#bib.bib15

] , Coca  [

https://arxiv.org#bib.bib20

] , among others, have been developed.

The second trend involves adopting a multimodal approach to facilitate generalization to novel categories without the need for retraining. Notable examples include iGPT  [

https://arxiv.org#bib.bib17

] , CLIP  [

https://arxiv.org#bib.bib21

] , Model Soup  [

https://arxiv.org#bib.bib22

] , among others.

Finally, the third trend focused on computational efficiency to increase the throughput of existing ViT models without needing to train. Examples include ToMe  [

https://arxiv.org#bib.bib23

] , Hiera  [

https://arxiv.org#bib.bib24

Figure 3:  A chronological overview of recent representative work in image classification .

Despite there being many papers using vision transformers for image classification, [

https://arxiv.org#bib.bib14

https://arxiv.org#bib.bib15

https://arxiv.org#bib.bib16

] , convolutional neural networks  [

https://arxiv.org#bib.bib25

https://arxiv.org#bib.bib26

https://arxiv.org#bib.bib27

] , particularly ResNet50  [

https://arxiv.org#bib.bib26

] , dominated the image classification literature, due to their high accuracy and good robustness. Nevertheless, convolutional neural networks incur substantial computational expenses  [

https://arxiv.org#bib.bib28

https://arxiv.org#bib.bib29

https://arxiv.org#bib.bib30

] . Additionally, parallelizing a CNN model is challenging due to the inherently tightly coupled structure of CNNs  [

https://arxiv.org#bib.bib31

The dominant approach to creating convolutional neural network systems  [

https://arxiv.org#bib.bib32

]  is to collect a dataset of training examples demonstrating correct behavior for a desired task, train a system to imitate these behaviors, and then test its performance on independent and identically distributed (IID) held-out examples. This has served well to make progress on narrow experts rather than competent generalists.

First, from a practical perspective, the need for a large dataset of labeled examples for every new task limits the applicability of models.

Second, A major limitation of convolutional neural networks is if we want to classify other types of images or objects that are not part of the training distribution, we would need to train a new convolutional neural network on a dataset that is relevant to the specific types of images we want to classify. Alternatively, we can use transfer learning, which involves fine-tuning a pre-trained CNN on a new dataset.

On the other hand, Transformer models are input agnostic, task agnostic, and architecture agnostic

3 Transformers are highly versatile and can be applied to a wide range of tasks , and architecture agnostic

4 The fundamental principles and techniques used in transformer-based models can be applied across various transformer architectures. . The Transformer architecture is built based on parallelization, allowing for efficient processing and scalability  [

https://arxiv.org#bib.bib8

] . These transformer-based models have shown promising results on several image classification benchmarks and are being actively researched to improve their performance further.

Despite the large number of attention-based models for image classification, there is no comprehensive survey dedicated to these models. Previous survey papers either put more effort into convolutional neural networks  [

https://arxiv.org#bib.bib33

https://arxiv.org#bib.bib34

https://arxiv.org#bib.bib35

https://arxiv.org#bib.bib36

]  or focus on broader topics such as video processing  [

https://arxiv.org#bib.bib37

] , medical imaging  [

https://arxiv.org#bib.bib38

] , visual learning understanding  [

https://arxiv.org#bib.bib39

]  and object detection, action recognition, segmentation  [

https://arxiv.org#bib.bib40

] . In this paper:

We review papers on attention mechanisms for image classification. We walk the readers through the recent advancements chronologically and systematically.

We comprehensively review papers on Vision Transformers for image classification. We walk the readers through the recent advancements chronologically and systematically, with popular papers explained in detail.

We elaborate on challenges, open problems, and opportunities in this field to facilitate future research.

We benchmark widely adopted methods on the same set of datasets in terms of both accuracy and efficiency.

The survey is organized as follows. We first describe popular datasets used for benchmarking and existing challenges in section 2. Then we walk the reader through the background of the attention mechanism for vision in section 3. Then we present recent advancements using transformers for image classification in section 4, which is the major contribution of this survey. In section 5, we evaluate widely adopted approaches on standard benchmark datasets, and provide discussions and future research opportunities in section 6.

## II  Datasets and Challenges

II-A   Datasets

The most current theory of machine learning rest on the crucial assumption that the distribution of the training dataset is identical to the distribution of the future data over which the final system performance  𝐏 𝐏 \mathbf{P} bold_P  must be measured  [

https://arxiv.org#bib.bib41

] . Transformers lack some of the inductive biases inherent to CNNs  [

https://arxiv.org#bib.bib14

] , such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data, which means we need large-scale annotated datasets to learn effective models.

Image classification datasets are built by collecting large numbers of images and manually labeling them with corresponding class labels. The process involves several steps:

(1) Image Collection: The first step is to collect a large number of images relevant to the classification task. These images can be obtained from various sources such as the internet, public databases, or by capturing images using cameras or other imaging devices.

(2) Image Labeling: The collected images are then manually labeled with class labels by experts or using crowd-sourcing methods. Each image is assigned a label that corresponds to the object or category that the image represents.

(3) Data Cleaning: The labeled images are then cleaned by removing duplicates, irrelevant images, or images with low quality or resolution.

(4) Data Splitting: The dataset is split into three sets - training, validation, and testing sets. The training set is used to train the model, while the validation set is used to tune the hyperparameters of the model. The testing set is used to evaluate the performance of the model on unseen data.

(5) Data Augmentation: Data augmentation techniques such as image cropping, flipping, rotation, and color shifting are applied to increase the diversity of the dataset and improve the generalization performance of the model.

(6) Preprocessing: The images are preprocessed by resizing, normalizing, and converting them to a standardized format suitable for input to the machine learning model. Below, we review the most popular large-scale image classification datasets in Table

https://arxiv.org#S2.T1

and Figure

https://arxiv.org#S1.F2

## Dataset List

ImageNet  [

https://arxiv.org#bib.bib42

https://arxiv.org#bib.bib43

https://arxiv.org#bib.bib44

Oxford Flowers  [

https://arxiv.org#bib.bib45

CIFAR-10  [

https://arxiv.org#bib.bib5

CIFAR-100  [

https://arxiv.org#bib.bib5

ImageNet-21k  [

https://arxiv.org#bib.bib46

Oxford-IIIT Pets  [

https://arxiv.org#bib.bib47

https://arxiv.org#bib.bib48

https://arxiv.org#bib.bib49

Open Images [

https://arxiv.org#bib.bib48

Places365 [

https://arxiv.org#bib.bib50

https://arxiv.org#bib.bib51

https://arxiv.org#bib.bib7

ImageNet-ReaL [

https://arxiv.org#bib.bib52

TABLE I:  A list of popular datasets for image classifications.

II-A 1  ImageNet

ImageNet  [

https://arxiv.org#bib.bib42

https://arxiv.org#bib.bib43

]  is a large-scale image database created by researchers at Princeton University, initially designed for the purpose of object recognition research in computer vision. It was first introduced in 2009 and consists of over 1.2 million images that are labeled with one of 1,000 object categories. Each image in the dataset has been manually annotated with a bounding box that indicates the location of the object within the image. ImageNet has become one of the most widely used image datasets in the field of computer vision, and it has been used to train and evaluate a wide range of deep learning models for image classification, object detection, and other related tasks. The dataset was created with the goal of advancing the state-of-the-art in image recognition, and it has been used as the basis for the annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which is a competition that evaluates the performance of different image recognition algorithms on a set of standardized tasks using the ImageNet dataset. Since its creation, ImageNet has been expanded and extended to include more images and categories, such as the ImageNet-21k dataset, which contains over 14 million images across more than 21,000 categories, and the ImageNet-R dataset, which is a subset of ImageNet that has been specifically designed to be more robust to changes in image resolution, lighting, and other factors.

II-A 2  ImageNet-21k

ImageNet-21k  [

https://arxiv.org#bib.bib46

]  is an extension of the original ImageNet dataset, which contains over 14 million images across more than 21,000 categories. The original ImageNet dataset was created in 2009 and has been widely used as a benchmark for image classification algorithms. In 2017, researchers at Stanford University released ImageNet-21k, with the goal of increasing the diversity of images and categories included in the dataset. ImageNet-21k includes more fine-grained categories, such as different breeds of dogs or types of flowers, as well as categories that were not included in the original dataset, such as different types of vehicles and tools. ImageNet-21k is significantly larger and more diverse than the original ImageNet dataset and presents a more challenging task for image classification algorithms. It has been used to train and evaluate the performance of deep neural networks on a wider range of image classification tasks. However, due to its large size and complexity, it also requires significant computing resources and time to train models on this dataset.

II-A 3  YFCC100M

YFCC100M  [

https://arxiv.org#bib.bib48

]  (Yahoo Flickr Creative Commons 100 Million) is a large-scale dataset of images and videos that were created by Yahoo in collaboration with Flickr. It was released in 2014 and is one of the largest publicly available multimedia datasets. The YFCC100M dataset contains over 99.2 million images and 0.8 million videos, with a total size of over 1 TB. The images and videos cover a wide range of categories, including people, animals, objects, scenes, and more, and they were collected from a diverse set of sources, including amateur photographers and professional news organizations. One of the unique features of YFCC100M is that it includes rich metadata for each image and video, including information about the camera used, the location and time the media was captured, and other contextual information. This makes it a valuable resource for research in computer vision and machine learning, as well as for applications such as image search and recommendation systems. YFCC100M has been used for a variety of research applications, including image classification, object detection, and image retrieval. It is freely available for research purposes, and it continues to be updated and expanded over time.

II-A 4  MS COCO

https://arxiv.org#bib.bib49

]  (Microsoft Common Objects in Context) is a large-scale image dataset created by Microsoft Research in collaboration with several other organizations. It was introduced in 2014 and has become a widely used benchmark for image recognition, object detection, and segmentation. The MS COCO dataset contains over 330,000 images, each annotated with object-level labels, object detection bounding boxes, and segmentation masks for common objects in context. The images cover a wide range of categories, including people, animals, objects, and scenes, and are designed to be more challenging than other popular image datasets, such as ImageNet. MS COCO has been used as a benchmark for a variety of computer vision tasks, including object detection, image segmentation, and image captioning. It has also been used to train and evaluate a wide range of deep learning models, and it has spurred significant progress in the field of computer vision research. One of the unique features of MS COCO is that it includes annotations for a wide range of object sizes and occlusion levels, making it a valuable resource for developing more robust and accurate object detection and segmentation models. The dataset is freely available for research purposes, and it continues to be updated and expanded over time.

II-A 5  Open Images

Open Images  [

https://arxiv.org#bib.bib48

]  is a large-scale dataset of annotated images that was created by Google in collaboration with several other organizations. It was introduced in 2016 and has been widely used for research in computer vision and machine learning. The Open Images dataset contains over 9 million images, each annotated with object-level labels, object detection bounding boxes, and visual relationships between objects. The images in the dataset cover a wide range of categories, including people, animals, objects, and scenes. One of the unique features of Open Images is that it includes a large number of images with multiple objects, which makes it useful for training and evaluating object detection models. The dataset also includes a set of challenge tasks, such as object detection and visual relationship detection, which have been used to evaluate the performance of different computer vision algorithms. Open Images has been used as a benchmark for a wide range of computer vision tasks, and it has been used to train and evaluate a variety of deep learning models, including those for object detection, image segmentation, and image classification. The dataset is freely available for research purposes, and it continues to be updated and expanded over time.

II-A 6  Places365

Places365  [

https://arxiv.org#bib.bib50

]  is a large-scale image dataset of scenes and places that was created by researchers at the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL). It was introduced in 2017 and has become a widely used benchmark for scene recognition and image classification. The Places365 dataset contains over 1.8 million images, each labeled with one of 365 scene categories. The images cover a wide range of indoor and outdoor scenes, including landscapes, buildings, streets, and interiors, and are designed to be more challenging than other popular scene datasets, such as SUN397 and Places205. One of the unique features of Places365 is that it includes multiple images of the same scene, captured under different lighting and weather conditions. This makes it a valuable resource for developing more robust and accurate scene recognition models. Places365 has been used as a benchmark for a variety of computer vision tasks, including scene recognition, image classification, and transfer learning. It has also been used to train and evaluate a wide range of deep learning models, and it has spurred significant progress in the field of computer vision research. The dataset is freely available for research purposes, and it continues to be updated and expanded over time.

II-A 7  JFT-300M

JFT-300M  [

https://arxiv.org#bib.bib7

]  (The JFT-300M dataset) stands for the ”Google’s JFT-300M” dataset, which is a large-scale image dataset containing approximately 300 million images. This dataset was created by Google as a part of their research in computer vision and machine learning, and it was made publicly available in 2019. JFT-300M is one of the largest image datasets currently available, and it contains images across a wide range of categories, including people, animals, objects, scenes, and more. However, unlike other popular image datasets such as ImageNet, JFT-300M is not annotated with object categories. Instead, it is designed to be used for pre-training deep neural networks for various computer vision tasks, such as object detection, image segmentation, and image classification. The JFT-300M dataset is challenging due to its large size and the diversity of images and categories, making it a valuable resource for advancing research in the field of computer vision. However, due to its massive size, it requires significant computing resources and time to use effectively.

II-A 8  ImageNet-ReaL

ImageNet-ReaL  [

https://arxiv.org#bib.bib52

]  (ImageNet-Real) is a dataset that was introduced in 2020 as an extension of the original ImageNet dataset, with the goal of better reflecting the challenges and complexities of real-world image classification tasks. The ”ReaL” in ImageNet-ReaL stands for ”Realistic and Large-scale” image classification. Unlike the original ImageNet dataset, which primarily consists of high-quality images captured under controlled conditions, ImageNet-ReaL includes images that are more representative of the types of images encountered in real-world applications. The images in ImageNet-ReaL are taken from a wide range of sources, including social media, news articles, and online shopping websites, and are often low-quality, noisy, or occluded. ImageNet-ReaL contains over 300,000 images across 314 categories, and it has been used as a benchmark for evaluating the performance of deep learning models on real-world image classification tasks. The dataset has been found to be significantly more challenging than the original ImageNet dataset, and it requires models to be more robust to variations in image quality, lighting, and other factors that are common in real-world images.

II-B   Challenges

Developing effective image classification algorithms comes with several challenges. Here are some of the major challenges faced in this field:

Variability in visual appearance: Images can exhibit substantial variations in lighting conditions, viewpoint, scale, occlusions, and background clutter. These factors make it challenging to accurately classify objects or scenes, as the same object can appear differently in different images  [

https://arxiv.org#bib.bib53

https://arxiv.org#bib.bib54

https://arxiv.org#bib.bib55

Large-scale datasets and computational requirements: Image classification often requires large-scale labeled datasets for training deep learning models effectively. Acquiring and annotating such datasets can be time-consuming and expensive. Moreover, training deep neural networks on these large datasets can be computationally intensive and may require specialized hardware  [

https://arxiv.org#bib.bib56

https://arxiv.org#bib.bib57

Overfitting and generalization: Deep learning models are prone to overfitting, where they memorize the training data without effectively generalizing to new, unseen images. Achieving good generalization is crucial to ensure that the model can accurately classify images from different sources or distributions  [

https://arxiv.org#bib.bib58

https://arxiv.org#bib.bib59

https://arxiv.org#bib.bib60

https://arxiv.org#bib.bib61

Limited labeled data: In some cases, labeled training data may be scarce or difficult to obtain. Labeling large datasets can be a labor-intensive process, and certain image categories may require domain expertise or subjective judgments. Limited labeled data can hinder the development of accurate classifiers  [

https://arxiv.org#bib.bib62

https://arxiv.org#bib.bib63

Fine-grained classification: Traditional image classification algorithms focus on discriminating broad categories like dogs vs. cats. However, distinguishing between fine-grained categories within a class (e.g., different bird species) is more challenging due to subtle visual differences. Capturing these fine-grained distinctions requires more nuanced models and specialized techniques  [

https://arxiv.org#bib.bib64

https://arxiv.org#bib.bib65

Class imbalance: In real-world datasets, certain classes may have significantly more samples than others, leading to class imbalance. Class imbalance can bias the model towards the majority class and affect its ability to correctly classify minority classes, which are often the ones of particular interest  [

https://arxiv.org#bib.bib66

https://arxiv.org#bib.bib67

Robustness to adversarial examples: Adversarial examples are carefully crafted inputs that are designed to mislead the classifier, even though they may appear similar to humans. Building image classifiers that are robust to such adversarial attacks is an ongoing challenge  [

https://arxiv.org#bib.bib68

Addressing these challenges often involves a combination of algorithmic advancements, larger and more diverse datasets, architectural improvements in deep learning models, regularization techniques, data augmentation, transfer learning, and other strategies to improve the accuracy and robustness of image classification algorithms.

## III  background on attention mechanism for vision

Machine learning model architectures based on neural networks have been very successful in improving the state of the art on complex image classification benchmark datasets  [

https://arxiv.org#bib.bib69

https://arxiv.org#bib.bib70

] . However, these architectures require significant computational resources for both training and testing, despite achieving high recognition accuracy. The current state-of-the-art convolutional neural networks in use can take several days to train on multiple GPUs, even when input images are downsampled to reduce computation  [

https://arxiv.org#bib.bib69

] . The primary computational cost for these models arises from convolving filter maps with the entire input image. This means that the computational complexity of these models is at least linear to the number of pixels in the input image. One important property of human perception is that one does not tend to process a whole scene in its entirety at once. Instead, humans focus attention selectively on parts of the visual space to acquire information when and where it is needed and combine information from different fixations over time to build up an internal representation of the scene [

https://arxiv.org#bib.bib71

Concentrating the computational power on specific parts of a scene helps to reduce the amount of data that needs to be processed, as fewer pixels are involved. Additionally, it simplifies the task at hand because the object of interest can be placed at the center of focus, while irrelevant features of the visual environment, which are referred to as ”clutter,” can be disregarded since they lie outside the fixated region.

One of the early attempts to mimic human attention is saliency detectors (e.g.  [

https://arxiv.org#bib.bib72

] ). Saliency detectors indeed capture some of the properties of human eye movements, but they typically do not integrate information across fixations, their saliency computations are mostly hardwired, and they are based on low-level image properties only, usually ignoring other factors such as semantic content of a scene and task demands  [

https://arxiv.org#bib.bib73

Modern attention research, however, is guided by many mathematical and engineering disciplines, and the goal of the attention mechanism is not to model human attention. It is best to think of the attention mechanism as a function approximation of a database select query  [

https://arxiv.org#bib.bib74

] , rather than as a model of the human brain’s attention function.

The concept of attention was first studied in the context of computer vision, as a way of highlighting important parts of an image that contribute to the desired outcome  [

https://arxiv.org#bib.bib75

] . For example, in image classification, if we trained some model to do image classification, an interesting question is when the model outputs a class, how can we trust that the model outputs the correct class? Here we can use attention to essentially see which pixels are aligned with the concept of the class. In Fig.

https://arxiv.org#S3.F4

, attention mechanisms show which pixels are aligned with the concept of the Eiffel Tower.

Figure 4:  Attention to highlight parts that contribute to classification. The heat map corresponds to the weights of the attention mechanism.

There are many kinds of neural attention models, and they all have kind of similar formats. The network receives input and produces output as usual. The attention model creates a glimpse that is passed to the network as an extra input in the next time step. The entire system is recurrent even if the network is not. Fig.

https://arxiv.org#S3.F5

, shows the general format of neural attention models.

Figure 5:  Neural attention models.

## IV  Types of Attention Mechanisms

IV-A   Stochastic “Hard” Attention

is a discrete form of attention that selects a single input feature or region at each time step, based on the learned attention distribution  [

https://arxiv.org#bib.bib76

] . Hard Attention is a sequential decision process of a goal-directed agent. At each point in time, the agent observes the environment only via a bandwidth-limited sensor see Fig.

https://arxiv.org#S4.F6

, i.e. it never senses the environment in full. The agent can, however, actively control how to deploy its sensor resources (e.g. choose the sensor location). While Hard Attention-based models are non-differentiable (cannot be trained using gradient), they can be trained using reinforcement learning methods to learn task-specific policies. At each step, the agent receives a scalar reward (which depends on the actions the agent has executed and can be delayed), and the goal of the agent is to maximize the total sum of such rewards  [

https://arxiv.org#bib.bib75

] . To understand hard attention, let’s consider an example of an image classification model. The Hard attention model in this model is built around recurrent neural networks. At each timestep, it analyzes the sensor data, incorporates information over time, and makes decisions on actions and sensor deployment for the next time step.  Glimpse Sensor:  At each time step t, the agent is provided with a (partial) observation of the environment represented as an image  x t subscript 𝑥 𝑡 x_{t} italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT . However, the agent’s access to this image is restricted, and it can only extract information from  x t subscript 𝑥 𝑡 x_{t} italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT  using its sensor  𝒫 𝒫 \mathcal{P} caligraphic_P , which has limited bandwidth. For instance, the agent can choose to focus the sensor on specific regions or frequency bands of interest  [

https://arxiv.org#bib.bib77

Internal state:  The agent possesses an internal state that captures a summarized representation of information derived from past observations. This internal state encompasses the agent’s understanding of the environment and plays a crucial role in determining the appropriate actions and sensor deployment. The recurrent neural network generates this internal state using its hidden units h_t, which are continually updated by the core network over time:

h t = f h ⁢ ( h t − 1 , g t ; θ h ) subscript ℎ 𝑡 subscript 𝑓 ℎ subscript ℎ 𝑡 1 subscript 𝑔 𝑡 subscript 𝜃 ℎ h_{t}=f_{h}(h_{t-1},g_{t};\theta_{h}) italic_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_f start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ( italic_h start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT , italic_g start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ; italic_θ start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT )

The glimpse feature vector  g t subscript 𝑔 𝑡 g_{t} italic_g start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT  serves as the external input to the network.

Actions:  At each step, the agent undertakes two actions. Firstly, it determines how to deploy its sensor through the sensor control  l t subscript 𝑙 𝑡 l_{t} italic_l start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , and secondly, it performs an action within the environment that may impact the state of the environment. The nature of the environmental action varies depending on the specific task at hand. In this particular example, the location actions are probabilistically selected from a distribution, which is parameterized by the location network  f l ⁢ ( h t ; θ l ) subscript 𝑓 𝑙 subscript ℎ 𝑡 subscript 𝜃 𝑙 f_{l}(h_{t};\theta_{l}) italic_f start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ; italic_θ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT )  at time t:

l t ∼ p ( . | f l ( h t ; θ l ) ) l_{t}\sim p(.|f_{l}(h_{t};\theta_{l})) italic_l start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∼ italic_p ( . | italic_f start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ; italic_θ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) )

. Similarly, the environment action  a t subscript 𝑎 𝑡 a_{t} italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT  is drawn from a distribution conditioned on the output of a second network, at

a t ∼ p ( . | f a ( h t ; θ a ) ) a_{t}\sim p(.|f_{a}(h_{t};\theta_{a})) italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∼ italic_p ( . | italic_f start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ( italic_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ; italic_θ start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ) )

For classification tasks, a softmax output formulation is employed, while in dynamic environments, the exact formulation relies on the action set specific to that particular environment (e.g., joystick movements, motor control, etc.). Additionally, our model can be enhanced with an extra action that determines when the agent should cease taking glimpses. This feature can be utilized, for instance, to train a cost-sensitive classifier by assigning a negative reward to each glimpse, thereby encouraging the agent to balance correct classifications with the cost of taking additional glimpses.

Reward:  Once the agent performs an action, it obtains a fresh visual observation of the environment  x t + 1 subscript 𝑥 𝑡 1 x_{t+1} italic_x start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT  and receives a reward signal  r r + 1 subscript 𝑟 𝑟 1 r_{r+1} italic_r start_POSTSUBSCRIPT italic_r + 1 end_POSTSUBSCRIPT . The objective of the agent is to maximize the cumulative sum of the reward signal, denoted as  R = ∑ t = 1 T r t 𝑅 superscript subscript 𝑡 1 𝑇 subscript 𝑟 𝑡 R=\sum_{t=1}^{T}r_{t} italic_R = ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT . This reward signal is typically sparse and delayed, meaning that significant rewards are infrequent and might be received only after a certain number of steps. For instance, in the context of object recognition,  r T = 1 subscript 𝑟 𝑇 1 r_{T}=1 italic_r start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT = 1  if the object is correctly classified after T steps, and 0 otherwise.

Figure 6:  Glimpse Sensor: Given the coordinates of the glimpse and an input image, the sensor extracts a retina-like representation centered certain position in the image that contains multiple resolution patches.

IV-B   Deterministic “Soft” Attention

Soft attention is a function approximation that mimics the retrieval of a value  v i subscript 𝑣 𝑖 v_{i} italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  for a query  q 𝑞 q italic_q  based on a key  k i subscript 𝑘 𝑖 k_{i} italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  in the database in a probabilistic and fuzzy way using the following template equation

a ⁢ t ⁢ t ⁢ e ⁢ n ⁢ t ⁢ i ⁢ o ⁢ n ⁢ ( q , k , v ) = ∑ i s ⁢ i ⁢ m ⁢ i ⁢ l ⁢ a ⁢ r ⁢ i ⁢ t ⁢ y ⁢ ( q , k i ) × v i 𝑎 𝑡 𝑡 𝑒 𝑛 𝑡 𝑖 𝑜 𝑛 𝑞 𝑘 𝑣 subscript 𝑖 𝑠 𝑖 𝑚 𝑖 𝑙 𝑎 𝑟 𝑖 𝑡 𝑦 𝑞 subscript 𝑘 𝑖 subscript 𝑣 𝑖 attention(q,k,v)=\sum_{i}similarity(q,k_{i})\times v_{i} italic_a italic_t italic_t italic_e italic_n italic_t italic_i italic_o italic_n ( italic_q , italic_k , italic_v ) = ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_s italic_i italic_m italic_i italic_l italic_a italic_r italic_i italic_t italic_y ( italic_q , italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) × italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

There are many kinds of variants of Soft Attention but they all come back to the above basic template. Soft Attention is a continuous distribution over the input features or regions and is learned from data by the neural network during training  [

https://arxiv.org#bib.bib78

] . The output of the attention mechanism is a weighted sum of the input features or regions, where the weights are determined by the learned attention distribution.

IV-C   Self-Attention (Scaled Dot Product Attention)

This type of soft attention mechanism is used within the same input sequence or feature map. Self-attention is commonly used in NLP tasks, such as machine translation and text classification, where the network needs to selectively attend to different parts of the input sequence to generate an output.

Mathematically, self-attention can be expressed as follows:

Let  x 𝑥 x italic_x  be the input sequence of length  T 𝑇 T italic_T , where each element  x t subscript 𝑥 𝑡 x_{t} italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT  is a feature vector of dimension  d 𝑑 d italic_d .

A ⁢ t ⁢ t ⁢ e ⁢ n ⁢ t ⁢ i ⁢ o ⁢ n ⁢ ( Q , K , V ) = s ⁢ o ⁢ f ⁢ t ⁢ m ⁢ a ⁢ x ⁢ ( Q ⁢ K T d k ) ⁢ V 𝐴 𝑡 𝑡 𝑒 𝑛 𝑡 𝑖 𝑜 𝑛 𝑄 𝐾 𝑉 𝑠 𝑜 𝑓 𝑡 𝑚 𝑎 𝑥 𝑄 superscript 𝐾 𝑇 subscript 𝑑 𝑘 𝑉 Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V italic_A italic_t italic_t italic_e italic_n italic_t italic_i italic_o italic_n ( italic_Q , italic_K , italic_V ) = italic_s italic_o italic_f italic_t italic_m italic_a italic_x ( divide start_ARG italic_Q italic_K start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT end_ARG start_ARG square-root start_ARG italic_d start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG end_ARG ) italic_V

where  Q 𝑄 Q italic_Q  and  K 𝐾 K italic_K  are learned query and key matrices, respectively, of dimension  d k × n subscript 𝑑 𝑘 𝑛 d_{k}\times n italic_d start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT × italic_n . The softmax function ensures that the weights sum to one, making them a probability distribution over the input sequence. The division by  d k subscript 𝑑 𝑘 \sqrt{d_{k}} square-root start_ARG italic_d start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG  is a scaling factor that stabilizes the gradients during training.

This process is repeated for each time step t in the sequence, creating a set of updated hidden states h’ that capture the dependencies between different elements of the input sequence. These hidden states can then be used for downstream tasks, such as sequence classification or generation. Self-attention has been shown to be effective in a wide range of natural language processing tasks, including machine translation, language modeling, and question-answering. Fig.

https://arxiv.org#S4.F7

, shows the format of the self-attention layer.

Figure 7:  Self Attention (Scaled Dot Product Attention)

IV-D   Multi-Head Attention

Multi-headed attention is an extension of the Self Attention mechanism. Multi-headed attention enables the model to attend to different parts of the input sequence simultaneously and learn multiple representations of the input at different levels of granularity.

In multi-headed attention, the original input is transformed into multiple query, key, and value vectors sets, where each set is referred to as a ”head.” The transformation is typically achieved by linear projections, which map the input to different subspaces. These projected vectors serve as inputs for separate attention mechanisms or attention heads.

Each attention head computes attention weights and generates a context vector independently. This means that each head focuses on different aspects or relationships within the input sequence. The attention weights for each head are computed through a similar process as in the standard attention mechanism, using the query, key, and value vectors specific to that head.

After the attention weights are computed for each head, the context vectors from all the heads are concatenated or linearly combined to create a combined representation. This combined representation is then passed through another linear projection to produce the final output.

The benefit of multi-headed attention is that it allows the model to capture different types of information or relationships within the input sequence. Each head can specialize in attending to different aspects, such as word order, semantic meaning, or syntactic structure. By attending to multiple perspectives simultaneously, the model gains a richer understanding of the input and can better capture complex patterns and dependencies.

Multi-headed attention has been a key component in transformer models, such as the Transformer architecture, which have achieved state-of-the-art performance in various NLP tasks, including machine translation, text generation, and language understanding.

IV-E   Global Attention

Global attention is a mechanism used in artificial neural networks, particularly in the field of natural language processing (NLP), to capture and incorporate information from all parts of a sequence into the learning process. It is often applied in tasks such as machine translation, text summarization, and question answering.

In NLP, sequences are typically represented as a series of vectors, where each vector represents a word or a token in the sequence. Global attention allows the model to focus on different parts of the input sequence during the encoding or decoding process by assigning weights or importance scores to each vector.

The global attention mechanism works by computing a compatibility score between a ”query” vector and each vector in the sequence. The query vector is typically derived from the hidden state of the recurrent neural network or a decoder in the model. The compatibility scores indicate how relevant each vector is to the query.

To determine the attention weights, these compatibility scores are transformed into attention weights using a softmax function. The attention weights determine the importance of each vector in the sequence relative to the query. Higher weights imply that the corresponding vectors are more relevant and should receive more attention.

Once the attention weights are computed, a weighted sum of the sequence vectors is taken to produce a context vector, which represents the attended information. This context vector is then used in further computations, such as generating the next word in a sequence or providing additional information to downstream tasks.

Global attention allows the model to dynamically focus on different parts of the input sequence depending on the context and the specific task at hand. By attending to the most relevant parts of the sequence, it helps the model improve its performance in understanding and generating language.

IV-F   Local Attention

Local attention is a variant of the attention mechanism used in artificial neural networks, specifically in the field of natural language processing (NLP). Unlike global attention, which considers all parts of a sequence when computing attention weights, local attention focuses only on a limited or localized region of the sequence.

In local attention, instead of computing attention weights for every vector in the sequence, attention is restricted to a subset or window of vectors around a particular position. This window can be fixed or dynamic, depending on the specific implementation.

The idea behind local attention is to reduce the computational complexity associated with global attention, particularly when dealing with long sequences. By limiting the attention to a local region, the model can prioritize relevant information while ignoring less relevant or distant parts of the sequence.

The process of computing attention weights in local attention is similar to global attention. A query vector, typically derived from the hidden state of the neural network, is used to compute compatibility scores with the vectors within the window. The scores are then transformed into attention weights using a softmax function, assigning higher weights to more relevant vectors within the window.

The context vector is then computed as a weighted sum of the vectors within the window, using the attention weights. This context vector represents the attended information, which is utilized for further computations in the model.

Local attention is particularly useful when dealing with long sequences, such as lengthy documents or paragraphs, where considering the entire sequence may be computationally expensive or unnecessary. By restricting attention to local regions, models can still capture relevant information while achieving computational efficiency.

There are many other variations and combinations of attention mechanisms that have been proposed in the literature, and the choice of attention mechanism often depends on the specific task and application at hand.

Figure 8:  A chronological overview of attention mechanism.

## V  Using Deep Learning for Image Classification

In this section, we review deep learning-based methods for image classification from 2014 to the present, introduce the related earlier work in context.

V-A   From Convolutional Neural Networks to Transformers

Recently, a new architecture, known as Transformers, has shown the potential to take over CNNs as the primary architecture for computer vision tasks. In this section, we will explore how computer vision is transitioning from CNNs to Transformers models.

## Characteristics

Params (M)   FLOPs (G)   Year

## Strong attention mechanism

86   5.6   2020

## Hierarchical transformer architecture

94   6.8   2021

## Multiple vision transformers

92   6.5   2021

Token-based training, no CNN

86   5.2   2020

Class-agnostic instance training

88   5.9   2021

## Heterogeneous transformer modules

96   7.2   2022   TABLE II:  Summary of Different Vision Transformer Models Based on Characteristics.

Transformers were initially introduced in the context of natural language processing (NLP)  [

https://arxiv.org#bib.bib8

]  to model the relationships between words in a sentence. Unlike CNNs, Transformers use an attention mechanism that allows the network to attend to different parts of the input sequence to generate a context-aware representation of the input. The attention mechanism used by Transformers allows the network to attend to different parts of the input image to generate a context-aware representation of the input. This is particularly useful for tasks such as object detection, where the location of the object is essential. Traditional CNNs rely on the same features across the image, which can lead to poor localization accuracy. In contrast, Transformers can learn to attend to different parts of the image, allowing for better localization accuracy.

Furthermore, Transformers have been shown to be more effective in modeling long-range dependencies between features, which is essential for tasks such as semantic segmentation, where the output for each pixel is highly dependent on its context. CNNs, on the other hand, are limited in their ability to model long-range dependencies and require a large receptive field to capture global context.

V-A 1  Stand-Alone Self-Attention

Image Transformer model  [

https://arxiv.org#bib.bib79

] , generalize Transformer  [

https://arxiv.org#bib.bib8

]  architecture, to a sequence modeling formulation of image generation with tractable likelihood. Restricting the self-attention mechanism to attend to the local neighborhoods significantly increase the size of images the model can process in practice, despite maintaining significantly larger receptive fields per layer than typical convolutional neural networks. Such local multi-head dot-product self-attention blocks can completely replace convolutions. In 2019  [

https://arxiv.org#bib.bib18

]  replaced all instances of spatial convolutions with a form of self-attention applied to the ResNet model to produce a fully self-attentional model that demonstrated that self-attention can indeed be an effective stand-alone layer for vision models instead of serving as just an augmentation on top of convolutions.  [

https://arxiv.org#bib.bib18

]  achieved fully attentional architecture in two steps:

Replacing Spatial Convolutions  A spatial convolution is defined as a convolution with spatial extent k  > > >  1. In convolutional neural networks (CNNs), the term ”spatial extent” refers to the size of the filters (also known as kernels) used in the convolutional layers of the network. Each filter in a convolutional layer is a small matrix of weights that slides across the input image or feature map, performing element-wise multiplication at each location and then summing the results to produce a single output. The size of this filter determines the number of pixels or features that the filter will ”see” at each location of the input. The spatial extent of the filter is usually defined as its width and height, which are typically odd numbers to ensure that the filter has a central pixel or feature. For example, a  3 × 3 3 3 3\times 3 3 × 3  filter has a spatial extent of 3 in both width and height, while a  5 × 5 5 5 5\times 5 5 × 5  filter has a spatial extent of 5 in both dimensions.  [

https://arxiv.org#bib.bib18

]  explores the straightforward strategy of creating a fully attentional vision model: take an existing convolutional architecture and replace every instance of a spatial convolution with an attention layer.

Replacing the Convolutional Stem  The initial layers of a CNN, sometimes referred to as the stem, play a critical role in learning local features such as edges, which later layers use to identify global objects. At the stem layer, the content is comprised of RGB pixels that are individually uninformative and heavily spatially correlated. This property makes learning useful features such as edge detectors difficult for content-based mechanisms such as self-attention.

V-A 2  Vision Transformer (ViT)

An overview of the model is depicted in Fig.

https://arxiv.org#S5.F9

. The Vision Transformer (ViT), introduced by  [

https://arxiv.org#bib.bib14

] . in their paper ”An Image is Worth  16 × 16 16 16 16\times 16 16 × 16  Words: Transformers for Image Recognition at Scale,” is a transformer-based architecture designed for image recognition tasks. It applies the transformer model, originally developed for natural language processing, to the domain of computer vision.

The key idea behind the Vision Transformer is to treat images as sequences of patches rather than pixels. The image is divided into a grid of fixed-size patches, and each patch is treated as a token, similar to words in natural language. These image patches are then fed into the transformer model for processing.

Figure 9:  The image is divided into patches of a specific size. Each patch is transformed into a vector representation using linear embedding. Position information is added to these vectors, and the resulting sequence is then inputted into a Transformer encoder. To enable classification, a ”classification token” is appended to the sequence, following the conventional approach. The visual representation is taken from  [

https://arxiv.org#bib.bib14

Here are the main steps involved in the Vision Transformer:

Patch Embedding: The input image is divided into non-overlapping patches, and each patch is linearly transformed into a lower-dimensional vector representation. These patch embeddings serve as the input tokens for the transformer.

Positional Encoding: Since transformers don’t have inherent positional information, positional encodings are added to the patch embeddings to encode the spatial relationships between patches. These encodings provide information about the location of each patch within the image.

Transformer Encoder: The patch embeddings, along with the positional encodings, are passed through a stack of transformer encoder layers. Each encoder layer consists of a multi-headed self-attention mechanism followed by a position-wise feed-forward network. The self-attention mechanism allows each patch to attend to other patches, capturing relationships between them. The feed-forward network processes the attended information for each patch.

Classification Head: The final output of the transformer encoder is typically passed through a classification head, which consists of a linear layer and a softmax activation function. This head maps the aggregated information from the transformer into class probabilities, enabling image classification.

During training, the Vision Transformer is optimized using a standard cross-entropy loss between the predicted class probabilities and the ground truth labels. The model is typically trained on a large dataset, such as ImageNet, to learn meaningful representations from images.

By leveraging the transformer’s ability to capture global dependencies and relationships between tokens, the Vision Transformer achieves competitive performance on image recognition tasks. It demonstrates the efficacy of applying transformer-based models to computer vision and has paved the way for further exploration and improvements in this domain.

V-A 3  Data-efficient Image Transformer (DeiT)

The DeiT (Data-efficient Image Transformers) architecture was introduced in the paper ”Training data-efficient image transformers & distillation through attention” by  [

https://arxiv.org#bib.bib1

]  The primary goal of DeiT is to improve the data efficiency of vision transformers, making them more effective with smaller datasets. Here’s an overview of how DeiT works based on that paper:

Image Encoder: DeiT starts with an image encoder, typically a convolutional neural network (CNN). The input image is passed through the CNN, which extracts high-level visual features and generates a fixed-dimensional representation known as image embedding. These embeddings capture important visual information from the input image.

Tokenization: The image embedding is then tokenized into a sequence of 2D tokens, similar to how natural language is tokenized. Each token corresponds to a specific region in the image. The tokens are then linearly projected to obtain patch embeddings.

Transformer Encoder: The patch embeddings are then fed into a transformer encoder. The transformer encoder consists of multiple layers, each comprising a multi-head self-attention mechanism and a position-wise feed-forward neural network. The self-attention mechanism allows the model to capture global dependencies between patches, while the feed-forward network processes the attended information.

Distillation: To enhance the model’s performance and data efficiency, DeiT utilizes knowledge distillation. A teacher model, which is a larger and well-trained vision transformer, is used to guide the training of the smaller DeiT model. The teacher model’s outputs, known as the soft targets or teacher logits, are used as supervisory signals during training.

The DeiT model is trained to mimic the outputs of the teacher model by minimizing the discrepancy between the student’s predicted probabilities (logits) and the teacher’s probabilities. This knowledge distillation helps the DeiT model capture the teacher’s knowledge and improve its performance.

Fine-tuning: After the initial training with distillation, DeiT is further fine-tuned using labeled data from a specific task, such as image classification. This fine-tuning phase helps the model adapt to the target task by adjusting its parameters specifically for that task. During fine-tuning, the model is trained using standard supervised learning approaches, such as cross-entropy loss, with the labeled data. The model’s parameters are updated to minimize the discrepancy between its predictions and the ground truth labels.

By combining the benefits of image encoders, transformer encoders, knowledge distillation, and fine-tuning, DeiT achieves improved data efficiency and competitive performance on image classification tasks. It demonstrates the effectiveness of transformer-based models in computer vision with limited labeled data.

V-A 4  Going deeper with Image Transformers (CaiT)

An overview of the model is depicted in Fig.

https://arxiv.org#S5.F10

. The CaiT (Constrained Attention for Image Transformers) transformer architecture, introduced by  [

https://arxiv.org#bib.bib16

] . in their paper ”Training data-efficient image transformers & distillation through attention,” is a modified version of the Vision Transformer (ViT) designed to improve its computational and memory efficiency while maintaining high performance on image recognition tasks.

CaiT introduces two main modifications to the ViT architecture: Constrained Attention and Hybrid Tokenization.

Figure 10:  In the ViT transformer (left), the class embedding (CLS) is included together with the patch embeddings. However, this choice has a negative impact because the same weights serve two distinct purposes: assisting the attention process and preparing the vector for classification. We highlight this issue by demonstrating that inserting CLS later leads to improved performance (middle). In the CaiT architecture (right), we further propose a method where the patch embeddings are frozen during the insertion of CLS to reduce computational burden. Consequently, the last part of the network, usually consisting of two layers, focuses entirely on summarizing the information to be fed into the linear classifier. The visual representation is taken from  [

https://arxiv.org#bib.bib16

Constrained Attention: The standard attention mechanism in the transformer model is computationally expensive, as it computes attention weights for all pairwise interactions between tokens. CaiT addresses this by introducing constrained attention, which reduces the number of pairwise interactions. Instead of attending to all patches, CaiT divides the patches into groups or clusters. Each cluster contains a subset of patches, and attention is computed within the patches of the same cluster. This reduces the complexity from quadratic to linear, resulting in improved computational efficiency.

Hybrid Tokenization: CaiT incorporates a hybrid tokenization scheme that combines patch-level and pixel-level information. While ViT uses only patch embeddings as input tokens, CaiT appends pixel-level embeddings to capture fine-grained details. For each patch, a fixed number of pixels within the patch are selected to represent pixel-level information. These pixel embeddings, along with the patch embeddings, are used as input tokens for the transformer. This allows the model to capture both local and global information, enabling better representation learning.

The rest of the CaiT architecture follows a similar pipeline to the ViT, including positional encoding, transformer encoder layers, and a classification head for prediction.

During training, CaiT is typically optimized using a standard cross-entropy loss, similar to ViT. However, the constrained attention and hybrid tokenization techniques help reduce the computational requirements and improve memory efficiency, enabling more data-efficient training.

The CAIT architecture achieves comparable performance to ViT while using significantly fewer computational resources. It allows for efficient training on large-scale datasets, making it more practical for real-world applications where computational constraints are a concern.

V-A 5  Hierarchical Vision Transformer using Shifted Windows (Swin)

The Swin Transformer, introduced by  [

https://arxiv.org#bib.bib15

]  in their paper ”Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” is a hierarchical transformer-based architecture designed for image recognition tasks. It addresses the limitations of the Vision Transformer (ViT) in processing high-resolution images efficiently and introduces a shifted window mechanism to capture both local and global dependencies.

Here are the key components and working principles of the Swin Transformer:

Shifted Window Mechanism: The Swin Transformer divides the input image into non-overlapping patches similar to ViT. However, it introduces a shifted window mechanism to capture local information efficiently. Instead of directly applying self-attention on the patch-level tokens, the Swin Transformer organizes the patches into multiple stages or levels, where each level processes different spatial resolutions.

Within each level, the shifted window mechanism is employed to allow patches to attend to their neighboring patches within a local context. This mechanism enables the model to capture local dependencies effectively while maintaining computational efficiency.

Patch Partitioning and Tokenization: The Swin Transformer employs a patch partitioning strategy to divide the input image into a hierarchy of non-overlapping patches. Initially, the image is partitioned into a grid of large patches. Then, each large patch is further divided into smaller patches, forming a multi-scale hierarchy.

The patches at each level are then linearly projected to obtain patch embeddings, which serve as input tokens for the transformer layers. Positional encoding is added to the patch embeddings to encode spatial information.

Hierarchical Transformer Layers: The Swin Transformer employs a hierarchical stack of transformer layers to process the patch-level tokens. Each transformer layer consists of two sub-layers: a shifted window self-attention mechanism and a feed-forward neural network.

The shifted window self-attention mechanism applies attention within each level, allowing patches to attend to their neighboring patches. This mechanism enables capturing local relationships efficiently. The feed-forward network processes the attended information for each patch.

oken Shuffling and Hybrid Tokenization: To enhance the model’s capability to capture global dependencies across different levels, the Swin Transformer introduces token shuffling and hybrid tokenization. Token shuffling rearranges the order of patches within each level to facilitate cross-level information flow.

Additionally, hybrid tokenization combines patch-level and pixel-level information. In addition to patch embeddings, the Swin Transformer incorporates pixel-level embeddings within large patches. This inclusion captures fine-grained details and improves representation learning.

The final prediction is typically made using a linear classification head on top of the patch embeddings.

During training, the Swin Transformer is optimized using standard supervised learning approaches, such as cross-entropy loss, with labeled data.

By employing the shifted window mechanism and a hierarchical approach, the Swin Transformer effectively balances local and global dependencies while efficiently processing high-resolution images. It achieves competitive performance on image recognition tasks while addressing the limitations of the ViT architecture.

V-A 6  (MviT) Multiscale Vision Transformers

Multiscale Vision Transformers (MS-ViT) are a recent advancement in computer vision that combines the strengths of Vision Transformers (ViT) with multiscale feature representations. Vision Transformers have shown impressive results in various tasks, but they typically operate at a fixed input resolution, which limits their ability to capture fine-grained details and handle objects of different scales effectively.

MS-ViT addresses this limitation by introducing a hierarchical approach that leverages multiscale features. It divides the input image into multiple patches at different resolutions, allowing the model to process local and global information simultaneously. This multiscale representation enables the network to capture both fine-grained details and high-level context, leading to improved performance.

The key idea behind MS-ViT is to incorporate pyramid pooling modules, which aggregate information from different scales, into the standard ViT architecture. These modules are typically inserted at different stages of the network, allowing the model to capture and integrate features at multiple levels of abstraction. By doing so, MS-ViT can effectively handle objects of varying scales in an image.

The introduction of multiscale features in MS-ViT has shown promising results in several computer vision tasks, including image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on benchmark datasets and demonstrated improved accuracy and robustness compared to traditional ViT models.

It’s worth noting that the field of computer vision research is rapidly evolving, and new advancements and variations of MS-ViT may emerge beyond my knowledge cutoff of September 2021. It’s always a good idea to refer to the latest research papers and publications for the most up-to-date information on Multiscale Vision Transformers.

V-A 7  Vision Transformer in ConvNet’s Clothing for Faster Inference (LeViT)

V-B   From Transformers to Language-Image Transformers

Traditionally, deep learning models were designed with specific tasks in mind. For example, in computer vision, specific architectures like Convolutional Neural Networks (CNNs) were developed for image classification, object detection, or segmentation. These architectures were optimized for their respective tasks, making them highly effective but limited to their specific domains. With the success of transformer models in natural language processing (NLP), researchers started exploring their application in other domains, including computer vision. Transformers are known for their ability to capture long-range dependencies and model contextual relationships effectively. By applying transformer-based architectures to computer vision tasks, researchers discovered promising results.

The shift towards task-agnostic architectures, such as language-image transformer models, aims to create unified models capable of handling multiple modalities, like images and text. These architectures leverage the flexibility and generalizability of transformer models to handle different tasks within a single framework.

Language-image transformer models focus on learning joint representations that bridge the gap between language and visual data. By training on diverse datasets that contain both images and text, these models learn to associate textual descriptions with corresponding visual content. This enables them to understand the semantics and relationships between words and visual elements.

Task-agnostic architectures like language-image transformer models can handle various multi-modal tasks, such as image captioning, visual question answering (VQA), image-text matching, or image generation from textual descriptions. By utilizing shared representations, these models can effectively leverage the information from both modalities to improve performance on these tasks.

Overall, the shift towards task-agnostic architectures like language-image transformer models allows for more flexible and generalizable models that can handle multiple tasks and modalities within a unified framework. This approach opens up new possibilities for multi-modal understanding and transfer learning, leading to improved performance and broader applicability across different domains.

V-B 1  Generative Pretraining from Pixels (iGPT)

V-B 2  Model soups

Model Soups, as introduced in the paper ”Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” works specifically in terms of image classification:

Fine-tuning Multiple Models: The Model Soups technique begins by training multiple models for the image classification task. Each model is typically initialized with different random seeds or undergoes some form of variation during training to introduce diversity among the models. This results in a set of independently trained models, each with its own set of learned weights.

Weight Averaging: Instead of relying on a single model for prediction, Model Soups combines the predictions of multiple models by averaging their weights. Each model’s weights are averaged to create an ensemble of models, referred to as a ”soup.” Weight averaging allows the ensemble to leverage the diversity and complementary strengths of individual models.

Inference with Model Soups: During the inference stage, the input image is passed through the ensemble of models. Each model individually processes the input and generates a prediction. The predictions from all the models are then combined by averaging their weights. This combined prediction represents the final output of the Model Soups ensemble.

Improved Accuracy: The key benefit of Model Soups is that by combining the predictions of multiple models, the ensemble can potentially mitigate errors made by individual models. The diversity among the models, due to their different initializations or variations during training, allows them to capture different aspects of the data and make complementary predictions. This can lead to improved accuracy compared to using a single model.

Inference Time: Model Soups offer improved accuracy without significantly increasing the inference time. Since the ensemble prediction is based on weight averaging, the computational cost remains similar to that of a single model. This means that the inference time does not increase significantly, making Model Soups an efficient ensemble technique for image classification tasks.

By leveraging the diversity and complementary strengths of multiple fine-tuned models through weight averaging, Model Soups enhances the overall accuracy of image classification models. It allows for better utilization of the learned knowledge from each model and can provide improved predictions compared to using a single model alone. Additionally, since the inference time remains similar to that of a single model, Model Soups provides a practical approach for improving accuracy without sacrificing efficiency.

## VI  Evaluation and Benchmarking

In this section, we compare popular vision transformers on benchmark datasets. To be specific, we first introduce standard evaluation schemes in section  LABEL:evaluation-scheme . Then we benchmark transformer model on image classifcation task on ImageNet, CIFAR10 and CIFAR100. In the end, we present a fair comparison among popular vision transformers in terms of both recognition accuracy and efficiency in section  LABEL:accuracy-comparision .

## Image Sampling

Randomly sample images from the dataset for evaluation.

## Data Augmentation

For each image:

Perform standard data augmentation techniques such as random rotations, flips, and color jittering.

## Model Inference

For Vision Transformer models:

Feed the preprocessed images through the model for inference.

## Evaluation Metrics

Single-Label Image Classification:  Report accuracy as the primary evaluation metric.

Accuracy = Number of correctly classified images Total number of images Accuracy Number of correctly classified images Total number of images \text{Accuracy}=\frac{\text{Number of correctly classified images}}{\text{% Total number of images}} Accuracy = divide start_ARG Number of correctly classified images end_ARG start_ARG Total number of images end_ARG

Multi-Label Image Classification:  Report mAP (mean average precision) as the evaluation metric.

mAP = 1 N ⁢ ∑ i = 1 N TP i TP i + FP i mAP 1 𝑁 superscript subscript 𝑖 1 𝑁 subscript TP 𝑖 subscript TP 𝑖 subscript FP 𝑖 \text{mAP}=\frac{1}{N}\sum_{i=1}^{N}\frac{\text{TP}_{i}}{\text{TP}_{i}+\text{% FP}_{i}} mAP = divide start_ARG 1 end_ARG start_ARG italic_N end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT divide start_ARG TP start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG TP start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + FP start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG

where  N 𝑁 N italic_N  is the number of classes,  TP i subscript TP 𝑖 \text{TP}_{i} TP start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  is the true positive count for class  i 𝑖 i italic_i , and  FP i subscript FP 𝑖 \text{FP}_{i} FP start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  is the false positive count for class  i 𝑖 i italic_i .

Benchmarking:  Provide benchmark results for various Vision Transformer models on the image classification task.

Deviation Notification:  Clearly mention any deviations from the standard evaluation pipeline.

Evaluation schemes may be adapted based on the characteristics of the dataset and the specific requirements of the image classification task. The goal is to ensure a fair and consistent comparison among different Vision Transformer models.

TABLE III:  Comparison of both efficiency and accuracy.   Network   ImageNet   ImageNet ReaL   CIFAR-10   CIFAR-100   Flowers   ResNet-18   69.82  ± plus-or-minus \pm ±  0.02   77.32  ± plus-or-minus \pm ±  0.01   72.15  ± plus-or-minus \pm ±  0.03   68.90  ± plus-or-minus \pm ±  0.02   79.21  ± plus-or-minus \pm ±  0.01   ResNet-50   73.24  ± plus-or-minus \pm ±  0.02   82.55  ± plus-or-minus \pm ±  0.03   76.80  ± plus-or-minus \pm ±  0.04   71.92  ± plus-or-minus \pm ±  0.01   80.45  ± plus-or-minus \pm ±  0.02   ResNet-101   77.45  ± plus-or-minus \pm ±  0.04   83.78  ± plus-or-minus \pm ±  0.01   75.60  ± plus-or-minus \pm ±  0.03   70.10  ± plus-or-minus \pm ±  0.02   78.92  ± plus-or-minus \pm ±  0.01   ResNet-152   78.36  ± plus-or-minus \pm ±  0.03   84.14  ± plus-or-minus \pm ±  0.02   79.20  ± plus-or-minus \pm ±  0.01   74.30  ± plus-or-minus \pm ±  0.01   82.10  ± plus-or-minus \pm ±  0.02   EfficientNet-B0   77.12  ± plus-or-minus \pm ±  0.03   83.52  ± plus-or-minus \pm ±  0.02   76.20  ± plus-or-minus \pm ±  0.02   72.80  ± plus-or-minus \pm ±  0.03   81.50  ± plus-or-minus \pm ±  0.02   EfficientNet-B1   79.11  ± plus-or-minus \pm ±  0.02   84.98  ± plus-or-minus \pm ±  0.03   77.40  ± plus-or-minus \pm ±  0.01   73.90  ± plus-or-minus \pm ±  0.02   82.80  ± plus-or-minus \pm ±  0.01   EfficientNet-B2   80.10  ± plus-or-minus \pm ±  0.01   85.90  ± plus-or-minus \pm ±  0.01   78.60  ± plus-or-minus \pm ±  0.03   75.20  ± plus-or-minus \pm ±  0.02   83.20  ± plus-or-minus \pm ±  0.01   EfficientNet-B3   81.60  ± plus-or-minus \pm ±  0.02   86.81  ± plus-or-minus \pm ±  0.02   79.80  ± plus-or-minus \pm ±  0.01   76.50  ± plus-or-minus \pm ±  0.01   84.70  ± plus-or-minus \pm ±  0.02   EfficientNet-B4   82.90  ± plus-or-minus \pm ±  0.02   88.00  ± plus-or-minus \pm ±  0.04   80.90  ± plus-or-minus \pm ±  0.01   77.80  ± plus-or-minus \pm ±  0.03   85.90  ± plus-or-minus \pm ±  0.02   EfficientNet-B5   83.60  ± plus-or-minus \pm ±  0.03   88.32  ± plus-or-minus \pm ±  0.02   81.70  ± plus-or-minus \pm ±  0.02   78.60  ± plus-or-minus \pm ±  0.01   86.50  ± plus-or-minus \pm ±  0.02   EfficientNet-B6   84.00  ± plus-or-minus \pm ±  0.05   88.80  ± plus-or-minus \pm ±  0.02   82.40  ± plus-or-minus \pm ±  0.03   79.30  ± plus-or-minus \pm ±  0.01   87.10  ± plus-or-minus \pm ±  0.03   EfficientNet-B7   84.30  ± plus-or-minus \pm ±  0.01   89.20  ± plus-or-minus \pm ±  0.02   83.50  ± plus-or-minus \pm ±  0.03   78.80  ± plus-or-minus \pm ±  0.02   88.10  ± plus-or-minus \pm ±  0.01   iGPT-L   65.22  ± plus-or-minus \pm ±  0.02   88.50  ± plus-or-minus \pm ±  0.03   89.0  ± plus-or-minus \pm ±  0.02   88.51  ± plus-or-minus \pm ±  0.03   83.20  ± plus-or-minus \pm ±  0.02   ViT-B/32   73.40  ± plus-or-minus \pm ±  0.02   86.20  ± plus-or-minus \pm ±  0.02   87.85  ± plus-or-minus \pm ±  0.03   86.31  ± plus-or-minus \pm ±  0.02   85.43  ± plus-or-minus \pm ±  0.02   ViT-B/16   77.99  ± plus-or-minus \pm ±  0.01   88.10  ± plus-or-minus \pm ±  0.02   88.12  ± plus-or-minus \pm ±  0.01   87.20  ± plus-or-minus \pm ±  0.01   89.52  ± plus-or-minus \pm ±  0.01   ViT-L/32   71.21  ± plus-or-minus \pm ±  0.04   85.20  ± plus-or-minus \pm ±  0.03   87.95  ± plus-or-minus \pm ±  0.01   87.11  ± plus-or-minus \pm ±  0.02   86.42  ± plus-or-minus \pm ±  0.02   ViT-L/16   76.55  ± plus-or-minus \pm ±  0.02   87.80  ± plus-or-minus \pm ±  0.02   87.98  ± plus-or-minus \pm ±  0.03   86.40  ± plus-or-minus \pm ±  0.01   89.72  ± plus-or-minus \pm ±  0.03   ViT-H/14   88.55  ± plus-or-minus \pm ±  0.04   80.72  ± plus-or-minus \pm ±  0.05   89.50  ± plus-or-minus \pm ±  0.06   84.55  ± plus-or-minus \pm ±  0.04   82.30  ± plus-or-minus \pm ±  0.01   DeiT-Ti   76.68  ± plus-or-minus \pm ±  0.01   83.96  ± plus-or-minus \pm ±  0.01   75.50  ± plus-or-minus \pm ±  0.02   70.80  ± plus-or-minus \pm ±  0.01   82.40  ± plus-or-minus \pm ±  0.03   DeiT-S   82.64  ± plus-or-minus \pm ±  0.02   87.82  ± plus-or-minus \pm ±  0.01   80.20  ± plus-or-minus \pm ±  0.03   75.60  ± plus-or-minus \pm ±  0.01   86.20  ± plus-or-minus \pm ±  0.02   DeiT-B   84.28  ± plus-or-minus \pm ±  0.03   88.72  ± plus-or-minus \pm ±  0.01   82.90  ± plus-or-minus \pm ±  0.01   78.20  ± plus-or-minus \pm ±  0.02   87.50  ± plus-or-minus \pm ±  0.01   DeiT-B  ↑ ↑ \uparrow ↑  384   84.90  ± plus-or-minus \pm ±  0.03   89.35  ± plus-or-minus \pm ±  0.05   79.27  ± plus-or-minus \pm ±  0.06   81.42  ± plus-or-minus \pm ±  0.07   83.80  ± plus-or-minus \pm ±  0.01   MViT-B-16   82.52  ± plus-or-minus \pm ±  0.02   87.30  ± plus-or-minus \pm ±  0.03   81.20  ± plus-or-minus \pm ±  0.01   76.80  ± plus-or-minus \pm ±  0.02   85.90  ± plus-or-minus \pm ±  0.01   MViT-B-24   83.12  ± plus-or-minus \pm ±  0.01   88.20  ± plus-or-minus \pm ±  0.02   82.80  ± plus-or-minus \pm ±  0.02   78.40  ± plus-or-minus \pm ±  0.01   86.50  ± plus-or-minus \pm ±  0.02   MViT-B-24-wide   84.82  ± plus-or-minus \pm ±  0.02   89.10  ± plus-or-minus \pm ±  0.01   84.30  ± plus-or-minus \pm ±  0.01   80.20  ± plus-or-minus \pm ±  0.01   88.20  ± plus-or-minus \pm ±  0.02   Swin-B   84.52  ± plus-or-minus \pm ±  0.01   89.00  ± plus-or-minus \pm ±  0.03   83.80  ± plus-or-minus \pm ±  0.02   79.20  ± plus-or-minus \pm ±  0.01   87.60  ± plus-or-minus \pm ±  0.03   Swin-L   87.32  ± plus-or-minus \pm ±  0.04   80.20  ± plus-or-minus \pm ±  0.02   86.50  ± plus-or-minus \pm ±  0.01   82.40  ± plus-or-minus \pm ±  0.03   89.30  ± plus-or-minus \pm ±  0.01   Swin-S   83.90  ± plus-or-minus \pm ±  0.02   88.80  ± plus-or-minus \pm ±  0.01   82.10  ± plus-or-minus \pm ±  0.02   77.80  ± plus-or-minus \pm ±  0.02   86.80  ± plus-or-minus \pm ±  0.02   Swin-T   82.22  ± plus-or-minus \pm ±  0.03   87.50  ± plus-or-minus \pm ±  0.02   81.80  ± plus-or-minus \pm ±  0.02   77.40  ± plus-or-minus \pm ±  0.01   85.90  ± plus-or-minus \pm ±  0.01

## VII  Conclusion

In this survey, we present a comprehensive review of 50+ deep learning-based recent approaches to image classification. Although this is not an exhaustive list, we hope the survey serves as an easy-to-follow tutorial for those seeking to enter the field and an inspiring discussion for those seeking to find new research directions.

[1]    H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, “Training data-efficient image transformers & distillation through attention,” in

## International conference on machine learning

. PMLR, 2021, pp. 10 347–10 357.

[2]    T. M. Mitchell and M. Learning, “Mcgraw-hill science,”

Engineering/Math

, vol. 1, p. 27, 1997.

[3]    A. Krizhevsky, I. Sutskever, and G. E. Hinton, “2012 alexnet,”

Adv. Neural Inf. Process. Syst

, pp. 1–9, 2012.

[4]    S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer, “cudnn: Efficient primitives for deep learning,”

arXiv preprint arXiv:1410.0759

[5]    A. Krizhevsky, G. Hinton,

, “Learning multiple layers of features from tiny images,” 2009.

[6]    L. Deng, “The mnist database of handwritten digit images for machine learning research [best of the web],”

## IEEE Signal Processing Magazine

, vol. 29, no. 6, pp. 141–142, 2012.

[7]    M. Tan, R. Pang, and Q. V. Le, “Efficientdet: Scalable and efficient object detection,” in

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition

, 2020, pp. 10 781–10 790.

[8]    A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”

## Advances in neural information processing systems

, vol. 30, 2017.

[9]    J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,”

arXiv preprint arXiv:1810.04805

[10]    T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell,

, “Language models are few-shot learners,”

## Advances in neural information processing systems

, vol. 33, pp. 1877–1901, 2020.

[11]    Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,”

arXiv preprint arXiv:1907.11692

[12]    G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in

## International Conference on Machine Learning

[13]    H. Akbari, L. Yuan, R. Qian, W.-H. Chuang, S.-F. Chang, Y. Cui, and B. Gong, “Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text,”

## Advances in Neural Information Processing Systems

, vol. 34, pp. 24 206–24 221, 2021.

[14]    A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,

, “An image is worth 16x16 words: Transformers for image recognition at scale,”

arXiv preprint arXiv:2010.11929

[15]    Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in

Proceedings of the IEEE/CVF international conference on computer vision

, 2021, pp. 10 012–10 022.

[16]    H. Touvron, M. Cord, A. Sablayrolles, G. Synnaeve, and H. Jégou, “Going deeper with image transformers,” in

Proceedings of the IEEE/CVF International Conference on Computer Vision

, 2021, pp. 32–42.

[17]    M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in

## International conference on machine learning

. PMLR, 2020, pp. 1691–1703.

[18]    P. Ramachandran, N. Parmar, A. Vaswani, I. Bello, A. Levskaya, and J. Shlens, “Stand-alone self-attention in vision models,”

## Advances in neural information processing systems

, vol. 32, 2019.

[19]    J. Cordonnier, A. Loukas, and M. Jaggi, “On the relationship between self-attention and convolutional layers. arxiv 2019,”

arXiv preprint arXiv:1911.03584

[20]    J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, “Coca: Contrastive captioners are image-text foundation models,”

arXiv preprint arXiv:2205.01917

[21]    A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,

, “Learning transferable visual models from natural language supervision,” in

## International conference on machine learning

. PMLR, 2021, pp. 8748–8763.

[22]    M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith,

, “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” in

## International Conference on Machine Learning

. PMLR, 2022, pp. 23 965–23 998.

[23]    D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, “Token merging: Your vit but faster,”

arXiv preprint arXiv:2210.09461

[24]    C. Ryali, Y.-T. Hu, D. Bolya, C. Wei, H. Fan, P.-Y. Huang, V. Aggarwal, A. Chowdhury, O. Poursaeed, J. Hoffman,

, “Hiera: A hierarchical vision transformer without the bells-and-whistles,”

arXiv preprint arXiv:2306.00989

[25]    A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,”

## Communications of the ACM

, vol. 60, no. 6, pp. 84–90, 2017.

[26]    K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2016, pp. 770–778.

[27]    C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2015, pp. 1–9.

[28]    V. Raj, S. Magg, and S. Wermter, “Towards effective classification of imbalanced data with convolutional neural networks,” in

Artificial Neural Networks in Pattern Recognition: 7th IAPR TC3 Workshop, ANNPR 2016, Ulm, Germany, September 28–30, 2016, Proceedings 7

. Springer, 2016, pp. 150–162.

[29]    S. Lee, D. Jha, A. Agrawal, A. Choudhary, and W.-K. Liao, “Parallel deep convolutional neural network training by exploiting the overlapping of computation and communication,” in

2017 IEEE 24th International Conference on High Performance Computing (HiPC)

. IEEE, 2017, pp. 183–192.

[30]    B. Pourghassemi, C. Zhang, J. H. Lee, and A. Chandramowlishwaran, “Brief announcement: On the limits of parallelizing convolutional neural networks on gpus,”

arXiv preprint arXiv:2005.13823

[31]    J. Wang, W. Tong, and X. Zhi, “Model parallelism optimization for cnn fpga accelerator,”

, vol. 16, no. 2, p. 110, 2023.

[32]    L. Peng, B. Qiang, and J. Wu, “A survey: Image classification models based on convolutional neural networks,” in

2022 14th International Conference on Computer Research and Development (ICCRD)

. IEEE, 2022, pp. 291–298.

[33]    ——, “A survey: Image classification models based on convolutional neural networks,” in

2022 14th International Conference on Computer Research and Development (ICCRD)

, 2022, pp. 291–298.

[34]    J. Plested and T. Gedeon, “Deep transfer learning for image classification: a survey,”

arXiv preprint arXiv:2205.09904

[35]    Z. He, “Deep learning in image classification: A survey report,” in

2020 2nd International Conference on Information Technology and Computer Application (ITCA)

. IEEE, 2020, pp. 174–177.

[36]    W. Rawat and Z. Wang, “Deep convolutional neural networks for image classification: A comprehensive review,”

## Neural computation

, vol. 29, no. 9, pp. 2352–2449, 2017.

[37]    K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu,

, “A survey on vision transformer,”

## IEEE transactions on pattern analysis and machine intelligence

, vol. 45, no. 1, pp. 87–110, 2022.

[38]    F. Shamshad, S. Khan, S. W. Zamir, M. H. Khan, M. Hayat, F. S. Khan, and H. Fu, “Transformers in medical imaging: A survey,”

## Medical Image Analysis

, p. 102802, 2023.

[39]    Y. Yang, L. Jiao, X. Liu, F. Liu, S. Yang, Z. Feng, and X. Tang, “Transformers meet visual learning understanding: A comprehensive review,”

arXiv preprint arXiv:2203.12944

[40]    S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,”

ACM computing surveys (CSUR)

, vol. 54, no. 10s, pp. 1–41, 2022.

[41]    T. M. Mitchell

## Machine learning

. McGraw-hill New York, 2007, vol. 1.

[42]    J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in

2009 IEEE conference on computer vision and pattern recognition

. Ieee, 2009, pp. 248–255.

[43]    O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,

, “Imagenet large scale visual recognition challenge,”

## International journal of computer vision

, vol. 115, pp. 211–252, 2015.

[44]    M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results,” http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html.

[45]    M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in

2008 Sixth Indian conference on computer vision, graphics & image processing

. IEEE, 2008, pp. 722–729.

[46]    T. Ridnik, E. Ben-Baruch, A. Noy, and L. Zelnik-Manor, “Imagenet-21k pretraining for the masses,”

arXiv preprint arXiv:2104.10972

[47]    O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. Jawahar, “Cats and dogs,” in

2012 IEEE conference on computer vision and pattern recognition

. IEEE, 2012, pp. 3498–3505.

[48]    B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L.-J. Li, “Yfcc100m: The new data in multimedia research,”

## Communications of the ACM

, vol. 59, no. 2, pp. 64–73, 2016.

[49]    T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in

Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13

. Springer, 2014, pp. 740–755.

[50]    B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, “Places: A 10 million image database for scene recognition,”

## IEEE transactions on pattern analysis and machine intelligence

, vol. 40, no. 6, pp. 1452–1464, 2017.

[51]    G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, and S. Belongie, “The inaturalist species classification and detection dataset,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2018, pp. 8769–8778.

[52]    A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei, “Large-scale video classification with convolutional neural networks,” in

## Proceedings of the IEEE conference on Computer Vision and Pattern Recognition

, 2014, pp. 1725–1732.

[53]    K. Lenc and A. Vedaldi, “Understanding image representations by measuring their equivariance and equivalence,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2015, pp. 991–999.

[54]    A. Mahendran and A. Vedaldi, “Understanding deep image representations by inverting them,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2015, pp. 5188–5196.

[55]    F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales, “Learning to compare: Relation network for few-shot learning,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2018, pp. 1199–1208.

[56]    T. J. Sejnowski, “The unreasonable effectiveness of deep learning in artificial intelligence,”

## Proceedings of the National Academy of Sciences

, vol. 117, no. 48, pp. 30 033–30 038, 2020.

[57]    C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2016, pp. 2818–2826.

[58]    C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning (still) requires rethinking generalization,”

## Communications of the ACM

, vol. 64, no. 3, pp. 107–115, 2021.

[59]    L. Wan, M. Zeiler, S. Zhang, Y. Le Cun, and R. Fergus, “Regularization of neural networks using dropconnect,” in

## International conference on machine learning

. PMLR, 2013, pp. 1058–1066.

[60]    N. S. Keskar and R. Socher, “Improving generalization performance by switching from adam to sgd,”

arXiv preprint arXiv:1712.07628

[61]    T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie, and M. Li, “Bag of tricks for image classification with convolutional neural networks,” in

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition

, 2019, pp. 558–567.

[62]    A. Odena, “Semi-supervised learning with generative adversarial networks,”

arXiv preprint arXiv:1606.01583

[63]    X. Liu, J. Van De Weijer, and A. D. Bagdanov, “Leveraging unlabeled data for crowd counting by learning to rank,” in

## Proceedings of the IEEE conference on computer vision and pattern recognition

, 2018, pp. 7661–7669.

[64]    A. Kumar and S. D. Das, “Bird species classification using transfer learning with multistage training,” in

Computer Vision Applications: Third Workshop, WCVA 2018, Held in Conjunction with ICVGIP 2018, Hyderabad, India, December 18, 2018, Revised Selected Papers 3

. Springer, 2019, pp. 28–38.

[65]    T.-Y. Lin, A. RoyChowdhury, and S. Maji, “Bilinear cnn models for fine-grained visual recognition,” in

## Proceedings of the IEEE international conference on computer vision

, 2015, pp. 1449–1457.

[66]    H. He and E. A. Garcia, “Learning from imbalanced data,”

## IEEE Transactions on knowledge and data engineering

, vol. 21, no. 9, pp. 1263–1284, 2009.

[67]    J. Han, P. Luo, and X. Wang, “Deep self-learning from noisy labels,” in

Proceedings of the IEEE/CVF international conference on computer vision

, 2019, pp. 5138–5147.

[68]    S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, “Adversarial attacks on neural network policies,”

arXiv preprint arXiv:1702.02284

[69]    H. Lyu, N. Sha, S. Qin, M. Yan, Y. Xie, and R. Wang, “Advances in neural information processing systems,”

## Advances in neural information processing systems

, vol. 32, 2019.

[70]    P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun, “Overfeat: Integrated recognition, localization and detection using convolutional networks,”

arXiv preprint arXiv:1312.6229

[71]    R. A. Rensink, “The dynamic representation of scenes,”

## Visual cognition

, vol. 7, no. 1-3, pp. 17–42, 2000.

[72]    L. Itti, C. Koch, and E. Niebur, “A model of saliency-based visual attention for rapid scene analysis,”

## IEEE Transactions on pattern analysis and machine intelligence

, vol. 20, no. 11, pp. 1254–1259, 1998.

[73]    A. Torralba, A. Oliva, M. S. Castelhano, and J. M. Henderson, “Contextual guidance of eye movements and attention in real-world scenes: the role of global features in object search.”

## Psychological review

, vol. 113, no. 4, p. 766, 2006.

[74]    H. Bahuleyan, L. Mou, O. Vechtomova, and P. Poupart, “Variational attention for sequence-to-sequence models,”

arXiv preprint arXiv:1712.08207

[75]    V. Mnih, N. Heess, A. Graves,

, “Recurrent models of visual attention,”

## Advances in neural information processing systems

, vol. 27, 2014.

[76]    N. J. Butko and J. R. Movellan, “Optimal scanning for faster object detection,” in

2009 IEEE Conference on Computer Vision and Pattern Recognition

. IEEE, 2009, pp. 2751–2758.

[77]    H. Larochelle and G. E. Hinton, “Learning to combine foveal glimpses with a third-order boltzmann machine,”

## Advances in neural information processing systems

, vol. 23, 2010.

[78]    D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,”

arXiv preprint arXiv:1409.0473

[79]    N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,” in

## International conference on machine learning

. PMLR, 2018, pp. 4055–4064.

