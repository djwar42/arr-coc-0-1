---
sourceFile: "8-bit numerical formats for deep neural networks - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:18.490Z"
---

# 8-bit numerical formats for deep neural networks - arXiv

43ed0f73-a0f7-46cf-8ee7-c5454c6357ce

8-bit numerical formats for deep neural networks - arXiv

1fd360df-289a-4a79-8fb7-17651c4cd99e

https://arxiv.org/pdf/2206.02915

8-BIT NUMERICAL FORMATS FOR DEEP NEURAL NETWORKS

Badreddine Noune, Phil Jones, Daniel Justus, Dominic Masters, and Carlo Luschi Graphcore Research Bristol, UK {badreddine,philj,danielj,dominicm,carlo}@graphcore.ai

Given the current trend of increasing size and complexity of machine learning architectures, it has become of critical importance to identify new approaches to improve the computational efficiency of model training. In this context, we ad-dress the advantages of floating-point over fixed-point representation, and present an in-depth study on the use of 8-bit floating-point number formats for activa-tions, weights, and gradients for both training and inference. We explore the ef-fect of different bit-widths for exponents and significands and different exponent biases. The experimental results demonstrate that a suitable choice of these low-precision formats enables faster training and reduced power consumption without any degradation in accuracy for a range of deep learning models for image classi-fication and language processing.

1 INTRODUCTION

The recent advances in AI research and the ground-breaking results achieved in a wide range of practical applications (Goodfellow et al., 2016) rely on the possibility of training increasingly large over-parametrized models, based on larger and larger datasets or using unsupervised/self-supervised learning. These breakthroughs have been enabled by the availability of increasing computational resources. However, current technology is no longer able to double the processing speed with every new silicon generation at a constant power, and it has become essential to make more efficient use of the available power.

With the continuing increase of complexity of deep learning applications, the scalability of machine learning systems has also become indispensable. Training of large distributed models creates a num-ber of challenges, relying on the effective use of the available compute, memory, and networking resources shared among the different nodes, limited by the available power budget. In this context, the use of efficient numerical formats is of critical importance, since it allows increased power effi-ciency due to both improved computational efficiency and communication efficiency in the exchange of data among processing units.

This paper reviews the recent advances in the use of low-precision numerical formats to enable reli-able training of modern deep architectures with power efficient computations, and provides a com-prehensive study of the effect of the use of low-precision formats on the training and generalization performance of different model architectures. We also address the trade-off of floating-point formats versus integer formats, and the effect of low-precision arithmetic on non-convex optimization and generalization.

The paper is organized as follows. Section 2 reviews recent work on the use of reduced precision nu-merical formats, and on low-precision training of modern deep neural networks. Section 3 discusses the performance of model architectures using 8-bit number formats, presenting a range of experi-mental results of different 8-bit floating-point formats and a systematic investigation of the effect of different exponent biases, providing guidelines for an effective mixed-precision design. Section 4 reports experimental results on the use of adaptive loss scaling algorithms to automate the scaling of gradients and further confirms the performance of the 8-bit formats presented in Section 3. Finally, conclusions are drawn in Section 5.

2 BACKGROUND

2.1 LOW-PRECISION TRAINING OF DEEP NEURAL NETWORKS

Courbariaux et al. (2014) studied the viability of low precision number formats for training deep neu-ral networks, comparing the accuracy achieved with reduced precision floating-point, fixed-point, and dynamically scaled fixed-point numbers on different image classification benchmarks. Gupta et al. (2015) demonstrated that it is possible to train Fully Connected and Convolutional Neural Networks (CNNs) for image classification using only 16-bit fixed-point arithmetic with stochastic rounding. Das et al. (2018) successfully trained CNNs with 16-bit integer activations, weights and gradients on the ImageNet dataset (Russakovsky et al., 2015), using per-tensor scaling factors.

The use of a numeric precision as low as 1 bit for weights and activations of models trained on image classification tasks has been explored by Courbariaux et al. (2015), Courbariaux et al. (2016), and Hubara et al. (2016). However, these approaches generally induced performance degradation when used on more complex tasks and datasets. Zhou et al. (2016) presented CNNs with numer-ical precision as low as 1 bit for activations and weights and 2 bits for gradients, but also found a considerable degradation in accuracy on different image classification tasks. Banner et al. (2018) demonstrated a successful 8-bit fixed-point quantization scheme for image classification models. This study achieved an accuracy close to baseline on the ImageNet benchmark by adaptively scaling tensors to the dynamic range of the 8-bit fixed-point format, but kept critical values and operations such as gradients with respect to activations and weight updates in higher precision.

2.2 SCALED INTEGERS VS FLOATING-POINT REPRESENTATION

While a fixed-point representation covers its dynamic range uniformly, floating-point numbers pro-vide a non-uniform coverage of the quantization range. Since activations, weights, and gradients of deep neural networks are typically distributed non-uniformly with mode close to zero (Miyashita et al., 2016; Banner et al., 2018), floating-point numbers are well-suited for their representation. Moreover, 8-bit floating-point formats have a much wider dynamic range compared to 8-bit fixed precision representations, and therefore do not necessarily require the implementation of schemes for adaptive scaling per layer or per tensor (Sun et al., 2019).

An 8-bit fixed-point format gives a dynamic range of 42.1 dB and a maximum Signal-to-Noise Ratio (SNR) for a standard normal distributed input signal equal to 40.5 dB, which is obtained for a quantization step q ≈ 0.031 or about 2−5 (see Appendix A). However, for quantization steps q < 2−5 the noise from clipping at overflow rapidly increases, while for q > 2−5 the noise due to quantization within the dynamic range dominates (Appendix A, Figure 7(a)). This leaves only a very narrow region where the fixed-point quantization yields a good SNR: for q = 2−4 the SNR is 34.9 dB, while for q = 2−6 we have an SNR of 19.2 dB (Appendix A, Figure 7(b)). Therefore, in the case of sub-optimal scaling or for heavy tailed signal distributions, the SNR of a fixed-point quantization can quickly become much smaller than that of a floating-point quantization (Widrow and Kollár, 2008). In contrast, the SNR of a floating-point number is approximately constant within its dynamic range – see Table 1 for a comparison of the dynamic range and SNR of different floating-point formats.

In this paper, different 8-bit floating-point formats are identified with the notation 1.E.p, which specifies the use of 1 sign bit, E exponent bits and p significand bits. For each floating-point format, we consider a bias value which offsets the range of values covered by the exponent field. This is equivalent to applying a fixed scaling factor.

A particular case of this format is 1.0.7. This format has no exponent bits and a 7 bit significand field, which corresponds to an 8 bit signed integer. Here we still have the freedom to apply an overall scaling factor, which controls the range of numbers represented by the format. The 1.0.7 format is therefore equivalent to a scaled integer.

2.3 ALTERNATIVE NUMBER FORMATS

Alternative numerical representations include tapered formats like the posit format (Gustafson and Yonemoto, 2017), which aim to improve the trade-off between dynamic range and precision of

Table 1: Explicitly stored number of bits for exponent (E) and significand (p), dynamic range (D) and SNR of different floating-point formats.

## FORMAT E p D SNR

IEEE float-32 8 bits 23 bits 1667.7 dB 151.9 dB

IEEE float-16 5 bits 10 bits 240.8 dB 73.7 dB

BFloat-16 8 bits 7 bits 1571.3 dB 55.6 dB

DLFloat1, 2 6 bits 9 bits 385.3 dB 67.6 dB

1.5.22 5 bits 2 bits 197.5 dB 25.5 dB

1.4.32 4 bits 3 bits 107.8 dB 31.5 dB

1.3.42 3 bits 4 bits 66.0 dB 37.5 dB

floating-point formats with the same number of bits. However, this comes at the cost of losing the property of a constant relative quantization error.

Logarithmic floating-point formats such as the deepfloat format proposed by Johnson (2018) can re-duce the hardware cost of multiplications, but increase the implementation complexity of additions. Moreover, deepfloat computation requires new hardware based on Kulisch accumulation (Kulisch, 2013) and LUT log-to-linear and linear-to-log conversions. On the other hand, the 8-bit floating-point formats considered in this paper have the advantage that they can reuse the same hardware resources as IEEE float-16 and float-32 formats, with the benefit of increased arithmetic efficiency by up to 4× with respect to float-16.

2.4 MIXED-PRECISION FLOATING-POINT DESIGN

Recently, mixed precision floating-point approaches have been developed to reduce the memory requirements and speed up computation for training a range of different deep learning models (Mi-cikevicius et al., 2017). By using different floating-point representations for activations, weights, gradients and accumulation of partial sums, state-of-the-art results can be obtained while improving memory and computational efficiency. Micikevicius et al. (2017) have used IEEE 16-bit floating-point numbers with 5 bits of exponent and 10 bits of significand for representing weights, activations and gradients, while performing accumulation in convolutions and matrix multiplications in IEEE float-32 precision. A float-32 master copy of the weights is kept and used for the weight update to prevent underflow. To enable the use of the IEEE float-16 format, which has a much smaller dy-namic range than float-32 (Table 1), Micikevicius et al. (2017) have introduced loss scaling, which is shown to improve the representation of small magnitude gradients for the backward-pass, as will be discussed in detail in Section 2.5.

To increase the numerical range of the IEEE float-16 format, the DLFloat format with 6 exponent bits and 9 significand bits (Agrawal et al., 2019) and the BFloat-16 format with 8 exponent bits and 7 significand bits (Kalamkar et al., 2019) have been proposed for training deep neural networks. As shown in Table 1, these number formats considerably extend the dynamic range at the cost of a reduced signal-to-noise ratio.

A more aggressive reduction of numeric precision during training promises further improvements in memory and computational efficiency. Wang et al. (2018) studied the performance of training with 8-bit floating-point representation of weights, activations and gradients, using 5 exponent bits and 2 significand bits. This work also considered the use of half-precision partial accumulation for matrix multiplications and convolutions and for weight update accumulation, relying on chunk-based computation and floating-point stochastic rounding. Mellempudi et al. (2019) have extended the use of 8-bit floating-point to additional types of models and datasets.

Sun et al. (2019) have recently proposed the use of different 8-bit floating-point formats during the forward and the backward pass. This study concluded that models using gradients with floating-

1 No sub-normal values. 2 Extended exponent range, only one codeword reserved for Inf and NaN.

point format 1.5.2 and weights and activations with floating-point format 1.4.3 can approach their float-32 counterparts across different applications in image classification and language processing. However, the study did not consider different values of exponent bias, and did not systematically explore 8-bit quantization of the first layer and of the last fully connected layer.

2.5 LOSS SCALING

The magnitude of gradients is generally considerably smaller than that of weights and activa-tions (Glorot and Bengio, 2010, see also Appendix D, Figures 14–23). The limited numerical precision of float-16 and float-8 number formats therefore makes low precision models prone to numerical underflow and vanishing of gradients. As a result, training of these models typically relies on one of two alternative methods to improve the representation of small gradients in low precision numerical formats: the use of large exponent biases, or the use of loss scaling.

Whilst a large exponent bias moves the range of representable values to smaller magnitudes, loss scaling scales up the value of the loss by a factor α at the end of the forward pass, before starting the backward pass, and is recovered during weight update (Micikevicius et al., 2017; Kuchaiev et al., 2018). For instance, in the case of Stochastic Gradient Descent (SGD) optimization, the loss scaling is simply recovered by scaling down by α the learning rate for the update of the model parameters.

The need for a model-dependent selection of the appropriate value of the loss scaling hyperparameter can be avoided by implementing adaptive loss scaling (Kuchaiev et al., 2018; Zhao et al., 2019). The performance of automatic loss scaling for 8-bit floating-point formats will be considered in Section 4.

3 8-BIT NUMERICAL FORMATS FOR MIXED PRECISION TRAINING

This section discusses the design and performance of 8-bit floating-point formats for quantization of the activations, weights and gradients for training and inference of different deep learning models, for different applications. Experimental results are presented for image classification on CIFAR-100 (Krizhevsky, 2009) and ImageNet (Russakovsky et al., 2015) using ResNet models (He et al., 2015) and EfficientNet models (Tan and Le, 2019), and for language processing, including WMT14 English-German translation (Bojar et al., 2016) using the Transformer model (Vaswani et al., 2017), and language understanding using BERT (Devlin et al., 2018).

3.1 FLOATING-POINT DESIGN

The IEEE-754 standard, the most commonly used numerical representation, defines a formulaic representation of real numbers as an approximation in order to support a trade-off between range and precision.

To support the detection of disallowed numerical operations that produce undefined or unrepre-sentable values, along with the representation of non-finite quantities such as infinities, often en-countered in floating-point arithmetic, the IEEE-754 standard reserves the use of the largest unbiased exponent field to signal special values. More specifically, if the biased-exponent field of a floating-point number is filled with all 1-bits, this indicates either an infinity (Inf) or an invalid result of a computation (NaN). However, despite the convenience in signalling Inf/NaN values, this implies that the number system reduces the range of representable values by a single exponent. Without doubt, as the floating-point format size reduces, the designer’s foremost objective is to maximise the offered range and precision. To this end, the number of codes reserved for non-numerical quantities must be kept at a minimum.

The IEEE-754 floating-point arithmetic also considers the existence of two zeros, namely a negative and a positive zero, despite their equal mathematical significance. In the context of machine intel-ligence workloads, which favour low-precision floating point arithmetic, it is unnecessary for the number system to comprise of two separate codewords that redundantly represent the same value.

As a nonconformity with the IEEE-754 standard, and in order to efficiently use the (256) available codewords, instead of reserving the all-one biased exponent code to represent special non-numerical values, this biased exponent can extend the range of normalised values by an extra exponent.

Backward Pass ∇!𝐿

Weight FP8 Cast

𝐴𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛"#$𝑊𝑒𝑖𝑔ℎ𝑡"

𝐴𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛"

## Forward Pass

Weight FP8 Cast

Activation FP8 Cast

Backward Pass ∇"𝐿

Activation FP8 Cast

∇%𝐿" ∇"𝐿

Backward Pass ∇!𝐿

Activation FP8 Cast

𝐴𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛"#$𝐴𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛",&

𝐴𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛",$

## Forward Pass

Activation FP8 Cast

Activation FP8 Cast

Backward Pass ∇!𝐿

Activation FP8 Cast

Figure 1: Illustration of the quantization procedure (a) for fully connected and convolutional layers and (b) for matrix multiplications in attention layers.

3.2 8-BIT FLOATING-POINT FORMATS FOR ACTIVATIONS, WEIGHTS AND GRADIENTS

Our investigation has been carried out on the range of floating-point representation parameters (ex-ponent field size, mantissa field size and exponent bias range) capable of maintaining the reference baseline performance for different 8-bit floating-point formats.

The best results have been obtained with the 1.5.2 and 1.4.3 formats, which combine a large dynamic range with a sufficiently high SNR. We also report the results obtained with the scaled integer format. Additional results are presented in Appendix C for the float-8 formats 1.6.1, 1.3.4, and 1.2.5.

In all cases, instead of reserving one exponent field to represent Inf and NaN, we reserve only a single codeword (corresponding to negative zero), which allows extension of the range by one extra exponent. If not mentioned otherwise, values in all the experiments are clipped to the maximal representable number instead of returning NaN if an overflow occurs during quantization.

Matrix multiplications and convolutions are by far the most computationally expensive operations for commonly employed deep neural networks. Therefore, the highest gains in terms of performance and energy consumption can be obtained by quantizing the inputs to these operations, in both the forward and backward pass of the training process. Additionally, quantization of the gradients with respect to weights∇wL is investigated, as summarized in Figure 1.

3.3 RESULTS FOR IMAGE PROCESSING

Figures 2 and 3 report the CIFAR-100 test performance of the floating-point formats 1.5.2, 1.4.3 and 1.0.7 (scaled integer) with a range of exponent biases for separate quantization of the activations, weights and gradients for ResNet-32 training. The test performance corresponding to the use of other 8-bit floating-point formats is reported in Appendix C. For each case, the figures give the performance of one of the above 8-bit formats for different values of the exponent bias. The results have been obtained with Stochastic Gradient Descent (SGD) optimization with momentum (Poliak, 1964) with batch size m = 32, base learning rate η̃ = 2−9 (learning rate η = mη̃ = 2−4), momentum coefficient α = 0.9, and weight decay parameter λ = 2 · 10−4. The CIFAR-100 training experiments have been run for a total of 200 epochs, with a learning rate schedule based on a reduction by a factor of 10 at 50% and at 75% of the total number of iterations.

The results of Figure 2 show that separate quantization of weights and activations of all ResNet-32 layers, with the exception of the first layer inputs, achieves good test performance for the float-8

0 4 8 12 16 20 24 28

## Exponent Bias

Activations; 1.5.2 Quantization

float-8 quantization

float-32 reference

-2 0 2 4 6 8 10 12

## Exponent Bias

Activations; 1.4.3 Quantization

float-8 quantization

float-32 reference

-6 -5 -4 -3 -2

## Exponent Bias

Activations; 1.0.7 Quantization

float-8 quantization

float-32 reference

5 10 15 20 25 30 35

## Exponent Bias

Weights; 1.5.2 Quantization

float-8 quantization

float-32 reference

2 4 6 8 10 12 14 16 18 20

## Exponent Bias

Weights; 1.4.3 Quantization

float-8 quantization

float-32 reference

-2 -1 0 1 2 3 4

## Exponent Bias

Weights; 1.0.7 Quantization

float-8 quantization

float-32 reference

Figure 2: ResNet-32 CIFAR-100 test performance of two different 8-bit floating-point formats com-pared to the scaled integer format (1.0.7) for representation of the activations without quantization of inputs to the first layer (a, b, c), and of the weights (d, e, f). Test accuracy mean ± standard deviation over ten independent runs.

18 20 22 24 26 28 30 32 34 36

## Exponent Bias

Gradients ∇xL; 1.5.2 Quantization

float-8 quantization

float-32 reference

16 17 18 19 20 21

## Exponent Bias

Gradients ∇xL; 1.4.3 Quantization

float-8 quantization

float-32 reference

0 5 10 15 20

## Exponent Bias

Gradients ∇xL; 1.0.7 Quantization

float-8 quantization

float-32 reference

10 15 20 25 30 35

## Exponent Bias

Gradients ∇wL; 1.5.2 Quantization

float-8 quantization

float-32 reference

6 8 10 12 14 16 18 20 22

## Exponent Bias

Gradients ∇wL; 1.4.3 Quantization

float-8 quantization

float-32 reference

2 3 4 5 6 7

## Exponent Bias

Gradients ∇wL; 1.0.7 Quantization

float-8 quantization

float-32 reference

Figure 3: ResNet-32 CIFAR-100 test performance of two different 8-bit floating-point formats com-pared to the scaled integer format (1.0.7) for representation of the gradients with respect to activa-tions (a, b, c), and of the gradients with respect to weights (d, e, f). Test accuracy mean ± standard deviation over ten independent runs.

formats 1.5.2 and 1.4.3 for a wide range of biases. In contrast, the usable range of bias values is greatly reduced with the 1.0.7 format for quantization of weights and activations. If we attempt to quantize all layer inputs including the input of the first layer, we obtain the test performances shown in Appendix C, Figure 9. From these results it is clear that 3 bits of significand are required for quantization of the first layer input.

Similarly, Figure 3 presents evidence that 1.5.2 and 1.4.3 quantization preserves test accuracy for the loss gradients (both gradients with respect to activations and gradients with respect to weights). Note that for the gradients there is the option of using loss scaling to achieve the equivalent of an exponent bias shift. For gradients with respect to activations, at least 4 exponent bits are required (see also Appendix C, Figure 12).

The exponent bias sweeps of Figures 2 and 3 were then used, together with the histograms of the different quantities reported in Appendix D, to select suitable 8-bit floating-point formats for simul-taneous quantization of activations, weights, and gradients with respect to activations and weights. Table 2 reports the CIFAR-100 test performance for quantization of activations, weights and gradi-ents for ResNet-32 training. In this case, in addition to leaving the input to the first layer unquantized, it was found to be beneficial not to quantize the gradient with respect to the activations at the output of the first layer.

Table 3 reports the ImageNet test performance with 8-bit floating-point quantization of the activa-tions, weights and gradients for ResNet-18 training. The ImageNet results have been obtained with SGD optimization with momentum with batch size m = 32, base learning rate η̃ = 2−11 (learning rate η = mη̃ = 2−6), momentum coefficient α = 0.9, and weight decay parameter λ = 10−4. The ImageNet training experiments have been run for a total of 100 epochs, with learning rate de-cay schedule based on a learning rate reduction by a factor of 10 at 30%, 60%, 80% and 90% of the total number of iterations. Note that using the 1.4.3 format for quantization of activations and weights and the 1.5.2 format for quantization of gradients allows us to match the reference accuracy, whereas using either the 1.4.3 format or the 1.5.2 format for all quantizations does not fully recover the baseline accuracy.

Table 2: ResNet-32 CIFAR-100 performance of different 8-bit floating-point formats (float-8 for-mat, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-32 format. Test accuracy mean ± standard deviation over ten independent runs. The accuracy of all quantized models is statistically indistinguishable from the baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL ACCURACY (%)

Baseline float-32 float-32 float-32 float-32 70.26 ± 0.24 1.4.3 formats 1.4.3, 10 1.4.3, 14 1.4.3, 20 1.4.3, 16 70.30 ± 0.37 1.5.2 formats 1.5.2, 24 1.5.2, 28 1.5.2, 32 1.5.2, 33 70.02 ± 0.40 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 33 1.5.2, 31 70.42 ± 0.45

Table 3: ResNet-18 ImageNet performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-32 format. Validation accuracy mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to the baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL ACCURACY (%)

Baseline float-32 float-32 float-32 float-32 70.35 ± 0.06 1.5.2 formats 1.5.2, 28 1.5.2, 24 1.5.2, 34 1.5.2, 31 69.95 ± 0.10 * 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 34 1.5.2, 31 70.29 ± 0.03

https://lh3.googleusercontent.com/notebooklm/AG60hOpnNfT0PdB_bwlvabfos8Ni1upvQRV5Jm4n12kCT1OJwWP32ZNhy926n4wHWifRK6ggQRe3VdX8X5O60_SZ7TRf4Py_7GxuxETmxLHwoTrvxMGax7rUjEeog6gN0OSnLW4BmdkfDw=w36-h716-v0

17a39a28-d057-47ce-936d-95048c0166f1

Table 4: ResNet-50 ImageNet performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-32 format. Validation accuracy mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL ACCURACY (%)

Baseline float-32 float-32 float-32 float-32 76.57 ± 0.09 1.5.2 formats 1.5.2, 24 1.5.2, 28 1.5.2, 32 1.5.2, 32 76.43 ± 0.09 * 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 32 1.5.2, 32 76.61 ± 0.10

Table 4 shows the ImageNet test performance with 8-bit floating-point quantization of the acti-vations, weights and gradients for ResNet-50 training, for SGD with momentum with batch size m = 32, with bias values consistent with the histograms reported in Appendix D.

Figures 3(a) and 3(d) show the accuracy changes with 1.5.2 format, for gradients with respect to activations and weights respectively, as the exponent bias is swept over a range of values. In each of these plots, only a single quantity (gradients with respect to activations or with respect to weights) is quantized with 8-bit float precision, while all other quantities are kept in float-32 precision.

In contrast, in Tables 2–4 all quantities are quantized using 8-bit floating-point formats. It is im-portant to confirm that the workable bias ranges of Figure 3 are still valid in this case. For this purpose, Figure 4 reports the test accuracy results of a range of bias combinations for the exponent biases applied to gradients with respect to weights and activations. For these tests the mixed format combination 1.4.3/1.5.2 was chosen and the exponent biases for weights and activations were kept at 14 and 10 as in Table 2.

As mentioned in Section 2, Figures 2(c) and (f) and 3(c) and (f) show the test performance for the floating-point number format 1.0.7. This format is equivalent to using a scaled integer, which con-sists of an 8-bit signed integer together with a scaling value that specifies the range to be represented by the integer values. This scaling factor can be represented by a suitable value of exponent bias applied to the 1.0.7 format. The test performance results demonstrate that it is not possible to repre-sent the gradients with respect to activations in this format, and the other quantities have extremely narrow bias ranges over which satisfactory performance is achieved. This is also illustrated in the histogram data of Figure 18, which reports the scaled integer coverage ranges for some typical bias

6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38

Gradients xL; Exponent Bias

1.4.3/1.5.2 Formats

Figure 4: ResNet-32 CIFAR-100 test performance relative to reference float-32 performance. Weights/activations quantization format 1.4.3 with biases 10, 14 respectively. Loss gradients with respect to weights/activations quantization format 1.5.2 with biases indicated on the x and y axes.

Table 5: EfficientNet-B0 ImageNet performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-16 format. Test accuracy mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ACCURACY (%)

Baseline float-16 float-16 float-16 76.34 ± 0.18 1.4.3 / 1.5.2 formats 1.4.3, 7 1.4.3, 12 1.5.2, 16 76.33 ± 0.18 1.4.3 / 1.5.2 formats 1.4.3, 7 1.4.3, 12 1.5.2, 17 76.23 ± 0.08 1.4.3 / 1.5.2 formats 1.4.3, 7 1.4.3, 12 1.5.2, 18 76.36 ± 0.17

Table 6: EfficientNet-B2 ImageNet performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-16 format. Test accuracy mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ACCURACY (%)

Baseline float-16 float-16 float-16 79.42 ± 0.07 1.4.3 / 1.5.2 formats 1.4.3, 7 1.4.3, 12 1.5.2, 16 79.43 ± 0.13

values from Figures 2(c) and (f) and 3(c) and (f). From these results, it is evident that the scaled integer format does not provide sufficient range to represent an adequate fraction of the bins in the histogram.

It could perhaps be argued that the performance of the scaled integer could be improved by allowing the freedom of a scaling value chosen per layer. However, the considered 8-bit floating-point formats are able to match the reference float-32 performance with a single exponent bias value common to all layers.

Tables 5-7 show the ImageNet test performance with 8-bit floating-point quantization of the activa-tions, weights and gradients for different EfficientNet models (Tan and Le, 2019). Following Tan and Le (2019), we train on ImageNet for 350 epochs with RMSProp optimization (Tieleman and Hinton, 2012) and decay the learning rate exponentially by a factor 0.97 every 2.4 epochs. We use a weight decay parameter λ = 10−5 on the convolutional weights, and a label smoothing factor of 0.1. We use a slightly smaller global batch size m = 768 across all training cases and scale the original learning rate and RMSProp decay factor. For the RMSprop optimizer we use learn-ing rate m · 2−14, momentum coefficient α = 0.9 and learning rate decay 1.0 − m · 2−14. Our final weights are obtained by using an exponentially weighted average over checkpoints from each

Table 7: EfficientNet-B4 ImageNet performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. For the first layer, both activations and gradients with respect to activations use float-16 format. Test accuracy mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ACCURACY (%)

Baseline float-16 float-16 float-16 82.42 ± 0.10 1.4.3 / 1.5.2 formats 1.4.3, 7 1.4.3, 12 1.5.2, 16 82.34 ± 0.10

training epoch, with decay factor 0.97. All the accuracy results derive from an averaging over five independent runs. In addition, as presented by Masters et al. (2021), the augmentation strategy uses a combination of Mixup (Zhang et al., 2017) and CutMix (Yun et al., 2019). Also in this case, the accuracy obtained using the 1.4.3 format for quantization of activations and weights and the 1.5.2 format for quantization of gradients fully matches the baseline accuracy.

3.4 RESULTS FOR LANGUAGE PROCESSING

The performance of 8-bit floating-point formats has also been assessed for Natural Language Pro-cessing (NLP) applications. The more recent advanced NLP models are based on the Transformer (Vaswani et al., 2017) and in particular its attention concept (Bahdanau et al., 2014). Based on the model published by Kuchaiev et al. (2018), a version of the Transformer base model has been imple-mented with quantized activations, weights, and gradients in all fully connected layers and matrix multiplications. The inputs to the layer normalization and softmax layers are left unquantized. The model performance for language translation has been evaluated on an English-German translation task using the WMT14 dataset. For each experiment, the model has been trained for 400,000 itera-tions with a batch size of 128 sentence pairs, using Adam optimization (Kingma and Ba, 2014) with a base learning rate η̃ = 2/

√ 512, optimizer parameters β1 = 0.9, β2 = 0.997 and ε = 10−9, and

a learning rate schedule with linear increase of the learning rate during 8,000 warm-up steps and learning rate decay proportional to 1/

√ step+ 1 afterwards (Kuchaiev et al., 2018). To reproduce

the performance of Transformer models reported in the literature (Vaswani et al., 2017), in Section 4 we also show the performance results of both the float-32 and float-8 models trained for 100,000 iterations with a batch size of 1024 sentence pairs. Similar to Vaswani et al. (2017) for these runs we used checkpoint-averaging to obtain the final weights used for testing.

The final BLEU scores3 are presented for the model with separate quantization of activations (Fig-ure 5(a) and (b)), weights (Figure 5(c) and (d)), gradients with respect to activations (Figure 6(a) and (b)), and gradients with respect to weights (Figure 6(c) and (d)), each using 1.5.2 and 1.4.3 number formats with a range of different exponent biases.

Figure 5(a) suggests a small degradation in the model accuracy when quantizing the activations of the Transformer using the 1.5.2 format. This is an effect of the limited SNR of the 1.5.2 format, since the 1.4.3 format with its higher SNR yields an accuracy on par with the float-32 baseline for a range of different exponent biases (Figure 5(b)). The latter format is therefore preferred for quantizing the Transformer activations.

The results shown in Figures 5(c) and (d) and 6(c) and (d) indicate that both 1.5.2 and 1.4.3 number formats are suitable for quantizing weights and gradients with respect to weights of the Transformer base model without any loss in accuracy. These experiments show a wide range of exponent biases which achieve a model accuracy that is statistically indistinguishable from the float-32 baseline. Figure 6(a) and (b) show that, whilst quantizing gradients with respect to activations using the 1.5.2 format yields a good model accuracy, a significant degradation occurs if the 1.4.3 format is used. This demonstrates that the quantization of the gradient with respect to activations requires a dynamic range of at least 5 exponent bits to maintain the float-32 baseline performance, which is in line with the results reported in Sun et al. (2019).

For all experiments, especially for the quantization of gradients, we observe that exponent biases larger than the ”natural” bias (bias 7 for 1.4.3 and bias 15 for 1.5.2) yield a better performance of the quantized model. A large exponent bias improves the representation of values close to zero. Since the absolute values of activations, weights and gradients tend to be small (see the histograms of Appendix D, Figures 20–23) this allows for an optimal use of the available value range (Appendix D, Figure 24).

Based on the above results we have selected individually well-performing 8-bit floating-point for-mats for activations, weights, gradients with respect to activations, and gradients with respect to weights, to be jointly tested in a fully quantized model. As shown in Table 8, with a suitable choice of the number format, the performance of a fully quantized model is on a par with the float-32 base-

3All BLEU scores reported in this paper are case insensitive, calculated on the WMT14 official evaluation data (Bojar et al., 2014) using sacreBLEU (Post, 2018). The tokenization followed the Moses mteval-v14 implementation (https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl).

0 4 8 12 16 20 24 28

## Exponent Bias

Activations; 1.5.2 Quantization

float-8 quantization

float-32 reference

0 2 4 6 8 10 12 14

## Exponent Bias

Activations; 1.4.3 Quantization

float-8 quantization

float-32 reference

12 16 20 24 28 32 36

## Exponent Bias

Weights; 1.5.2 Quantization

float-8 quantization

float-32 reference

2 4 6 8 10 12 14 16 18 20

## Exponent Bias

Weights; 1.4.3 Quantization

float-8 quantization

float-32 reference

Figure 5: Transformer WMT14 English-German translation performance of different 8-bit floating-point formats for representation of the activations (a, b), and the weights (c, d). BLEU mean ± standard deviation over five independent runs.

28 30 32 34 36 38 40 42 44

## Exponent Bias

Gradients ∇xL; 1.5.2 Quantization

float-8 quantization

float-32 reference

22 24 26 28 30

## Exponent Bias

Gradients ∇xL; 1.4.3 Quantization

float-8 quantization

float-32 reference

16 20 24 28 32 36 40 44

## Exponent Bias

Gradients ∇wL; 1.5.2 Quantization

float-8 quantization

float-32 reference

16 18 20 22 24 26 28

## Exponent Bias

Gradients ∇wL; 1.4.3 Quantization

float-8 quantization

float-32 reference

Figure 6: Transformer WMT14 English-German translation performance of different 8-bit floating-point formats for representation of the gradients with respect to activations (a, b), and the gradients with respect to weights (c, d). BLEU mean ± standard deviation over five independent runs.

Table 8: Transformer WMT14 English-German translation performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. BLEU mean ± standard deviation over five independent runs. The accuracy of all quantized models is statistically indis-tinguishable from the baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL BLEU

Baseline float-32 float-32 float-32 float-32 25.58 ± 0.08 1.5.2 formats 1.5.2, 24 1.5.2, 28 1.5.2, 40 1.5.2, 40 25.48 ± 0.26 1.4.3 / 1.5.2 formats 1.4.3, 8 1.4.3, 14 1.5.2, 40 1.5.2, 40 25.52 ± 0.22 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 16 1.5.2, 40 1.5.2, 40 25.56 ± 0.23

line. Both 1.5.2 and 1.4.3 formats can be used to represent activations and weights, with no clear advantage of one format over the other.

We observe that not only does the accuracy of a quantized model match the performance of the baseline, but also the distribution of the weights of each layer is – within the range of representable values – identical to the distribution of the weights of the baseline model (see Appendix D, Fig-ure 25). This indicates that, even though weights, activations and gradients have been quantized to 8-bit floating-point values during the entire training process, a model is learned that is consistent with the float-32 baseline model.

In Tables 9-11 we report the results for quantizing the different matrix multiplication layers in BERT-Large using float-8, for both pre-training and fine-tuning (Devlin et al., 2018). Each experiment consists of two pre-training phases and a fine-tuning phase comprising multiple training runs, started from the pre-trained model. All phases use the AdamW optimizer (Loshchilov and Hutter, 2017), with β1 = 0.9, β2 = 0.999 and ε = 10−6. The learning rate follows a linear warm-up decay schedule, whereby the warmup phase lasts for the minimum of either 104 steps or a tenth of total number of steps. Pre-training phase one optimizes the Masked Language Model (MLM) and Next-Sentence Prediction (NSP) loss for corrupted sentence pairs. Masked and padded sequences of length 128 are grouped into batches of approximately 512 sequences. The model is trained for 10 epochs of Wikipedia (Merity et al., 2016) + BookCorpus (Zhu et al., 2015), corresponding to approximately 8 · 105 optimizer steps. For all experiments, the learning rate is set to the largest value that maintains stable convergence. Pre-training phase two uses sequence length 384, 5 epochs, and approximately 2 · 105 optimization steps.

Table 9: BERT-Large pre-training phase one performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. MLM+NSP test loss mean ± stan-dard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL TEST LOSS

Baseline float-16 float-16 float-16 2.19 ± 0.02 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 16 2.22 ± 0.01 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 18 2.23 ± 0.01 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 20 2.22 ± 0.02 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 24 2.22 ± 0.01 *

Table 10: BERT-Large pre-training phase two performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients. MLM+NSP test loss mean ± stan-dard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL TEST LOSS

Baseline float-16 float-16 float-16 1.85 ± 0.009 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 16 1.86 ± 0.023 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 18 1.90 ± 0.042 * 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 20 1.87 ± 0.002 * 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 24 1.92 ± 0.006 *

Table 11: BERT-Large fine-tuning SQuAD performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients used for both pre-training and fine-tuning. F1 Score and Exact Match mean ± standard deviation over five independent runs. Asterisks indicate a difference with respect to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL F1 SCORE EXACT MATCH

Baseline float-16 float-16 float-16 90.51 ± 0.08 83.42 ± 0.12 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 14 1.5.2, 16 90.69 ± 0.16 83.40 ± 0.23

4 AUTOMATIC LOSS SCALING

As already mentioned in Section 3.3, the same effect as changing the exponent bias for 8-bit repre-sentations of the gradients can be obtained by loss scaling. Instead of choosing a large exponent bias to prevent underflow, this method consists in scaling up the loss after the forward pass and scaling down the learning rate by the same factor before updating the weights or otherwise absorbing the scaling factor in the optimizer, as discussed in Section 2.5. We have tested algorithms that automate the selection of the loss scaling factor, to overcome the necessity to manually tune the bias for a low precision representation of the gradients. An adaptive loss scaling factor also allows one to react to changes in the long term statistics of the magnitudes of gradient components during training. Our experiments with different biases for quantization of the gradients suggest that the optimal range of representable values covers, or slightly clips, the largest occurring values and extends as far as possi-ble to small values (see Appendix D, Figure 24). Therefore, we have considered two algorithms that aim at using a factor that scales the maximal gradient values to the maximum of the representable range: a) Backoff loss scaling and b) LogMax loss scaling (Kuchaiev et al., 2018).

For the set of experiments with Backoff loss scaling, we disable the clipping of gradients upon overflow and return NaN instead. When this happens, Backoff skips the weight update and reduces the loss scaling coefficient by a factor of 2. Then, whenever no overflow occurs for 2,000 consecutive iterations, the algorithm increases the loss scaling coefficient by a factor of 2.

The LogMax algorithm estimates the mean µ and standard deviation σ of the quantity log2 [ max (|∇wL|) ] for every mini-batch, and scales the loss such that µ + cσ equals log2 of the maximum representable value for some constant c. We have tested the performance of LogMax loss scaling with different multiples of the estimated standard deviation and with clipping of the gradients upon overflow.

For WMT14 English-German translation, better results are obtained with large loss scaling factors that make it possible to represent small gradients, and in the case of LogMax can lead to a moderate clipping of gradients (see Table 12). We find that a model with 1.5.2 activations, weights and gradi-ents reaches the same performance as our float-32 baseline when Backoff is used to scale the loss.

Table 12: Effect of different automatic loss scaling algorithms on the Transformer WMT14 English-German translation performance with 8-bit floating-point formats (float-8 format, bias) for activa-tions, weights and gradients. BLEU mean ± standard deviation over five independent runs. Aster-isks indicate a difference to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL BLEU

Baseline float-32 float-32 float-32 float-32 25.58 ± 0.08 Backoff 1.5.2 1.5.2, 15 1.5.2, 15 1.5.2, 15 1.5.2, 15 25.50 ± 0.24 Backoff 1.5.2 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 25.48 ± 0.18 Backoff 1.4.3 / 1.5.2 1.4.3, 7 1.4.3, 7 1.5.2, 15 1.5.2, 15 25.42 ± 0.13 * Backoff 1.4.3 / 1.5.2 1.4.3, 10 1.4.3, 16 1.5.2, 15 1.5.2, 15 25.50 ± 0.19 Backoff 1.4.3 1.4.3, 10 1.4.3, 16 1.4.3, 7 1.4.3, 7 0.00 ± 0.004 * Backoff 1.4.3 / 1.5.2 1.5.2, 24 1.5.2, 28 1.4.3, 7 1.4.3, 7 0.00 ± 0.004 * LogMax 1.5.2, 3σ 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 25.34 ± 0.26 LogMax 1.5.2, 2σ 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 24.94 ± 0.42 * LogMax 1.5.2, 1σ 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 25.12 ± 0.23 * LogMax 1.5.2, 0σ 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 25.30 ± 0.29

Table 13: Transformer WMT14 English-German translation performance of different 8-bit floating-point formats (float-8 format, bias) for activations, weights and gradients after training for 100,000 iterations with batch size 1024. BLEU mean ± standard deviation over five independent runs. As-terisks indicate a difference to baseline accuracy based on a one-sided Mann-Whitney U-test (Mann and Whitney, 1947) at 5% level of significance.

ACTIVATIONS WEIGHTS ∇xL ∇wL BLEU

Baseline float-32 float-32 float-32 float-32 27.04 ± 0.11 1.5.2 formats 1.5.2, 24 1.5.2, 28 1.5.2, 40 1.5.2, 40 26.94 ± 0.11 1.4.3 / 1.5.2 formats 1.4.3, 10 1.4.3, 16 1.5.2, 40 1.5.2, 40 26.88 ± 0.18 Backoff 1.5.2 1.5.2, 24 1.5.2, 28 1.5.2, 15 1.5.2, 15 26.84 ± 0.05 * Backoff 1.4.3 / 1.5.2 1.4.3, 10 1.4.3, 16 1.5.2, 15 1.5.2, 15 26.96 ± 0.09

When using 1.4.3 quantized activations and weights with the natural bias of 7, we see a degradation in the final BLEU score that could be recovered by hand tuning the respective biases. A quantized model using the 1.4.3 format for representing the gradients diverged in all cases (Table 12). This is in agreement with our finding that at least 5 exponent bits are required to represent the gradient with respect to the activations∇xL (Figure 6(a) and (b)).

From Table 12 and Appendix E, Figure 26, the LogMax algorithm yields the highest BLEU scores when using a factor of c = 0 (LogMax,0σ in Table 12) and thereby employing a large loss scaling factor. In this case its performance is similar to that of our baseline model.

The automatic loss scaling algorithms with the best performance, Backoff and LogMax with constant c = 0, result in a loss scaling factor of about 222, together with a gradient representation based on the 8-bit floating-point format 1.5.2 with bias 15 (see Appendix E, Figure 26). This is equivalent to using a bias of 37 for gradients ∇xL and gradients ∇wL and no loss scaling, putting these results in line with our experiments using different biases in Figure 6.

To compare the performance of our quantized Transformer base model to results presented in the literature (Vaswani et al., 2017; Kuchaiev et al., 2018; Mellempudi et al., 2019; Prato et al., 2019; Sun et al., 2019) we trained the float-32 baseline model and selected fully quantized models with

4 Model did not converge.

a batch size of 1024 sentence pairs for 100,000 iterations. To account for the larger batch size, we reduced the number of warm-up steps to 4,000 for these experiments, in line with Vaswani et al. (2017). Furthermore, we obtained the final weights from averaging the last five checkpoints which were saved at intervals of 1,000 iterations (Vaswani et al., 2017).

The results reported in Table 13 show that the BLEU scores achieved by the quantized models with the 1.5.2 format for gradients and the 1.4.3 or 1.5.2 format for activations and weights still match those of the float-32 baseline model and are similar to previously published results. This confirms our results of Table 8 and Table 12 on the applicability of 8-bit floating point formats for training the Transformer model.

5 CONCLUSIONS

We have reported a range of experiments that confirms the robust training and generalization perfor-mance of mixed-precision implementations based on 8-bit floating-point formats 1.4.3, with 4 bits of exponent and 3 bits of significand, for activations and weights, and 1.5.2, with 5 bits of exponent and 2 bits of significand, for gradients with respect to activations and weights.

The study provides evidence for the trade-off between the required number of bits used to represent the exponent and the significand for float-8 training. In particular, the results confirm the superior performance of 8-bit floating-point formats compared to 8-bit scaled integers, and indicate that using 3 bits of significand can be advantageous for quantization of the activations, while 5 bits of exponent is often necessary for quantization of the gradients with respect to activations. We demonstrate that, with the choice of an exponent bias that allows for representing the majority of occurring values, models which use these 8-bit number formats in matrix multiplications and convolutions show no degradation of test performance compared to the corresponding float-32 baseline, with the most robust choice being to use the float-8 format 1.4.3 for activations and weights, and the float-8 format 1.5.2 for gradients with respect to activations and weights. These findings confirm that 8-bit floating-point formats are a useful alternative to higher precision number formats to accelerate the training of deep learning models and make efficient use of the available power.

6 BROADER IMPACT

The ongoing trend to use larger deep learning model architectures not only has consequences on the time and cost to train a model, but it also has a significant environmental impact (Strubell et al., 2019). It is critical for the research community to study the implementation of more efficient training schemes to counter the increase in energy consumption with growing model size.

The results of this study confirm that low precision numerical formats can be a key component of large machine learning models that provide state of the art accuracy while reducing their environ-mental impact. In particular, by using 8-bit floating point arithmetic the energy efficiency can be increased by up to 4× with respect to float-16 arithmetic and up to 16× with respect to float-32 arithmetic.

## REFERENCES

Ankur Agrawal, Silvia M. Mueller, Bruce M. Fleischer, Jungwook Choi, Naigang Wang, Xiao Sun, and Kailash Gopalakrishnan. DLFloat: A 16-bit floating point format designed for deep learning training and inference. In Proceedings of 26th IEEE Symposium on Computer Arithmetic, ARITH 26, 2019.

Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 [cs.CL], 2014.

Ron Banner, Itay Hubara, Elad Hoffer, and Daniel Soudry. Scalable methods for 8-bit training of neural networks. arXiv preprint arXiv:1805.11046 [cs.LG], 2018.

Ondrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Lev-eling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, et al. Findings of the 2014

Workshop on Statistical Machine Translation. In Proceedings of 9th Workshop on Statistical Machine Translation, pages 12–58, 2014.

Ondřej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, et al. Findings of the 2016 Conference on Machine Translation. In Proceedings of 1st Conference on Machine Translation: Volume 2, Shared Task Papers, pages 131–198, 2016.

Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Training deep neural networks with low precision multiplications. arXiv preprint arXiv:1412.7024 [cs.LG], 2014.

Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. BinaryConnect: Training deep neural networks with binary weights during propagations. arXiv preprint arXiv:1511.00363 [cs.LG], 2015.

Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1. arXiv preprint arXiv:1602.02830 [cs.LG], 2016.

Dipankar Das, Naveen Mellempudi, Dheevatsa Mudigere, Dhiraj Kalamkar, Sasikanth Avancha, Kunal Banerjee, Srinivas Sridharan, Karthik Vaidyanathan, Bharat Kaul, Evangelos Georganas, Alexander Heinecke, Pradeep Dubey, Jesus Corbal, Nikita Shustrov, Roma Dubtsov, Evarist Fomenko, and Vadim Pirogov. Mixed precision training of convolutional neural networks us-ing integer operations. arXiv preprint arXiv:1802.00930 [cs.NE], 2018.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 [cs.CL], 2018.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of 13th International Conference on Artificial Intelligence and Statis-tics, AISTAT 2010, pages 249–256, 2010.

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, Cambridge, MA, 2016.

Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. arXiv preprint arXiv:1502.02551 [cs.LG], 2015.

John L. Gustafson and Isaac Yonemoto. Beating floating point at its own game: Posit arithmetic. Supercomputing Frontiers and Innovations, 4(2):71–86, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-nition. arXiv preprint arXiv:1905.11946 [cs.CV], 2015.

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. arXiv preprint arXiv:1609.07061 [cs.NE], 2016.

Jeff Johnson. Rethinking floating point for deep learning. arXiv preprint arXiv:1811.01721 [cs.NA], 2018.

Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A study of BFLOAT16 for deep learning training. arXiv preprint arXiv:1905.12322 [cs.LG], 2019.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 [cs.LG], 2014.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Oleksii Kuchaiev, Boris Ginsburg, Igor Gitman, Vitaly Lavrukhin, Jason Li, Huyen Nguyen, Carl Case, and Paulius Micikevicius. Mixed-precision training for NLP and speech recognition with OpenSeq2Seq. arXiv preprint arXiv:1805.10387 [cs.LG], 2018.

Ulrich Kulisch. Computer arithmetic and validity: theory, implementation, and applications, vol-ume 33. Walter de Gruyter, 2013.

Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in Adam. arXiv preprint arXiv:1711.05101 [cs.LG], 2017.

Henry B. Mann and Donald R. Whitney. On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics, pages 50–60, 1947.

Dominic Masters, Antoine Labatie, Zach Eaton-Rosen, and Carlo Luschi. Making EfficientNet more efficient: Exploring batch-independent normalization, group convolutions and reduced resolution training. arXiv preprint arXiv:2106.03640 [cs.LG], 2021.

Naveen Mellempudi, Sudarshan Srinivasan, Dipankar Das, and Bharat Kaul. Mixed precision train-ing with 8-bit floating point. arXiv preprint arXiv:1905.12334 [cs.LG], 2019.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843 [cs.CL], 2016.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. arXiv preprint arXiv:1710.03740 [cs.AI], 2017.

Daisuke Miyashita, Edward H. Lee, and Boris Murmann. Convolutional neural networks using logarithmic data representation. arXiv preprint arXiv:1603.01025 [cs.NE], 2016.

Boris T. Poliak. Some methods of speeding up the convergence of iteration methods. USSR Com-putational Mathematics and Mathematical Physics, 4(5):1–17, 1964.

Matt Post. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186–191, 2018.

Gabriele Prato, Ella Charlaix, and Mehdi Rezagholizadeh. Fully quantized transformer for improved translation. arXiv preprint arXiv:1910.10485 [cs.CL], 2019.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, December 2015.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. arXiv preprint arXiv:1906.02243 [cs.CL], 2019.

Xiao Sun, Jungwook Choi, Chia-Yu Chen, Naigang Wang, Swagath Venkataramani, Vijayalakshmi Srinivasan, Xiaodong Cui, Wei Zhang, and Kailash Gopalakrishnan. Hybrid 8-bit floating point (HFP8) training and inference for deep neural networks. In Advances in Neural Information Processing Systems 32, NeurIPS 2019, 2019.

Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946 [cs.LG], 2019.

Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop, Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 2012.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762 [cs.CL], 2017.

Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Train-ing deep neural networks with 8-bit floating point numbers. arXiv preprint arXiv:1812.08011 [cs.LG], 2018.

Bernard Widrow and István Kollár. Quantization Noise: Roundoff Error in Digital Computation, Signal Processing, Control, and Communications. Cambridge University Press, Cambridge, 2008.

Bernard Widrow, István Kollár, and Ming-Chang Liu. Statistical theory of quantization. IEEE Transactions on Instrumentation and Measurement, 45(2):353–361, 1996.

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization strategy to train strong classifiers with localizable features. arXiv preprint arXiv:1905.04899 [cs.CV], 2019.

Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empiri-cal risk minimization. arXiv preprint arXiv:1710.09412 [cs.LG], 2017.

Ruizhe Zhao, Brian Vogel, and Tanvir Ahmed. Adaptive loss scaling for mixed precision training. arXiv preprint arXiv:1910.12385 [cs.LG], 2019.

Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, and Yuheng Zou. DoReFa-Net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160 [cs.NE], 2016.

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Tor-ralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. arXiv preprint arXiv:1506.06724 [cs.CV], 2015.

A DYNAMIC RANGE AND SIGNAL-TO-NOISE RATIO OF FIXED-POINT QUANTIZATION

For a scaled fixed-point representation with 1 sign bit, n integer bits and quantization step q, the dynamic range is

D = 20 log10[ 2n ] ≈ 6.02n dB. (1)

The roundoff error of a uniform quantizer with quantization step q can be modelled by additive noise with uniform distribution over the interval ( −q/2, q/2 ). If the input signal x satisfies Widrow’s quantization theorems (i.e., if the input signal characteristic function Φx(u) ≈ 0 for |u| > 2π/q) (Widrow et al., 1996), the statistical model of the quantization error ν of uniform quantization corre-sponds to an independent noise source with zero mean and variance E{ν2} = q2/12 (Widrow et al., 1996; Widrow and Kollár, 2008). When quantizing a standard normal distributed random variable using a fixed-point quantizer with n bits of precision and a quantization interval q, E{ν2} can be broken down into a clipping noise part that occurs when values that exceed the dynamic range of the quantizer get clipped and the rounding noise part that occurs within the dynamic range:

ν2(x) fx(x) dx

∫ (m+ 1 2 )q

−(m+ 1 2 )q

ν2(x) fx(x) dx︸ ︷︷ ︸ rounding noise

∫ −(m+ 1 2 )q

−∞ ν2(x) fx(x) dx︸ ︷︷ ︸

clipping noise

where fx(x) is the standard normal PDF and m = 2n − 1.

With the approximation fx(x) ≈ fx(iq) in the interval [(i− 1/2)q, (i+ 1/2)q], the rounding noise component can be written as

∫ (m+ 1 2 )q

−(m+ 1 2 )q

ν2(x) fx(x) dx =

∫ (i+ 1 2 )q

(i− 1 2 )q

ν2(x) fx(x) dx

i=−m fx(iq)

−m fx(xq) dx · q

This expression goes to q2/12 for sufficiently large mq, i.e. if almost all values fall within the dynamic range.

## For the clipping noise component we have

2 ∫ −(m+ 1 2 )q

−∞ ν2(x) fx(x) dx = 2

∫ −(m+ 1 2 )q

−∞ (x+mq)2fx(x) dx

∫ −(m+ 1 2 )q

−∞ x2fx(x) dx+ 4mq

∫ −(m+ 1 2 )q

−∞ x fx(x) dx

+ 2m2q2 ∫ −(m+ 1

−∞ fx(x) dx

≈ (2 + 2m2q2)Fx(−mq)− 2mq fx(mq)

= (1 +m2q2)

where Fx(x) denotes the standard normal CDF. This gives rise to an overall signal-to-noise ratio of

SNR =− 10 log10[E{ν2} ]

≈− 10 log10

) + ( 1 +m2q2

2 6 2 5 2 4

Quantization Stepsize 0

## Rounding Noise Clipping Noise Total Noise

2 8 2 6 2 4 2 2 20 22

Quantization Stepsize 0

## Experimental Analytical

Figure 7: (a) Breakdown of the noise components for 8-bit fixed-point quantization of a standard normal signal. (b) SNR for 8-bit fixed-point quantization of a standard normal signal, derived from (3) (blue) and experimentally obtained from a Gaussian distribution (red).

https://lh3.googleusercontent.com/notebooklm/AG60hOrxmqbrkjYfyc5z2_WmOYQiIpQHXoSfEiiy8aWfOHRMZ_dCiEm3qd1EgRVOki8CkIFzwoEj4RrvIouCnwdEbp5ecBzy3C5w6vPJ06DupKy9454R4W0JUZxmXBDQhpqVn5fYSml2=w1797-h504-v0

409d2e59-5edc-49f2-90b0-82916b794959

B STATISTICAL MODEL OF FLOATING-POINT QUANTIZATION

A floating-point representation with p bits of significand5 and exponent range Emax − Emin, has a dynamic range

D = (2p − 1) 2Emax−Emin

≈ 6.02 (Emax − Emin + p) dB. (4)

The quantization error νFL associated with a floating-point representation is approximately propor-tional to the amplitude of the input signal (Widrow and Kollár, 2008). The input-output characteristic of a floating-point quantizer with p bits of significand and quantization step q is only piecewise uni-form over intervals of width ∆ = 2pq. The results of uniform quantization derived in Appendix A are therefore generally not applicable to floating-point quantization. However, a statistical model of the floating-point quantization error can be obtained by representing the floating-point quantizer by a cascade of a piecewise linear compressors, followed by a uniform quantizer and a piecewise linear expansion, as illustrated in Figure 8 (Widrow et al., 1996; Widrow and Kollár, 2008).

Figure 8: Model of floating-point quantization.

The floating-point quantization error νFL = x′ − x can then be expressed in terms of the uniform quantization error ν = z′ − z, that in practice can be represented by the classical uniform quanti-zation model (Widrow and Kollár, 2008). The decompression function of Figure 8 is approximately an exponential function, and its output contains an additional quantization noise term νEXP corre-sponding to the quantized exponent (Widrow and Kollár, 2008)

2 ν · |x| ∆

If the uniform quantization model applies to both the inner quantizer and the quantized exponent noise νEXP , the variance of the floating-point quantization noise νFL is given by (Widrow and Kollár, 2008)

E{νFL2} = 2.16 · E{ν2}E{ x 2

= 2.16 · q 2

= 0.18 · 2−2p E{x2}

which corresponds to

SNR = 5.55 · 22p ⇒ SNRdB = 7.44 + 6.02p . (5)

5Here p denotes the precision of a floating point number with sub-normal values including the implicitly stored hidden leading bit.

C RESULTS FOR OTHER FLOAT-8 FORMATS

In this appendix we present results on the test accuracy of a ResNet-32 model trained on the CIFAR-100 dataset, when we additionally quantize the inputs to the first layer using an 8-bit floating point format (Figure 9). In Figures 10–13 we also report the CIFAR-100 performance for ResNet-32 training with separate quantization of activations, weights, and gradients with the 1.6.1, 1.3.4, and 1.2.5 number formats.

5 10 15 20 25 30 35 40 45 50 55

## Exponent Bias

Activations; 1.6.1 Quantization

float-8 quantization

float-32 reference

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30

## Exponent Bias

Activations; 1.5.2 Quantization

float-8 quantization

float-32 reference

0 2 4 6 8 10 12

## Exponent Bias

Activations; 1.4.3 Quantization

float-8 quantization

float-32 reference

-2 -1 0 1 2 3 4 5

## Exponent Bias

Activations; 1.3.4 Quantization

float-8 quantization

float-32 reference

-3 -2 -1 0 1

## Exponent Bias

Activations; 1.2.5 Quantization

float-8 quantization

float-32 reference

-5 -4 -3 -2

## Exponent Bias

Activations; 1.0.7 Quantization

float-8 quantization

float-32 reference

Figure 9: ResNet-32 CIFAR-100 test performance of different 8-bit floating-point formats for rep-resentation of the activations with quantization of the input to the first layer. Test accuracy mean ± standard deviation over ten independent runs.

5 10 15 20 25 30 35 40 45 50 55

## Exponent Bias

Activations; 1.6.1 Quantization

float-8 quantization

float-32 reference

-2 -1 0 1 2 3 4 5

## Exponent Bias

Activations; 1.3.4 Quantization

float-8 quantization

float-32 reference

-3 -2 -1 0 1

## Exponent Bias

Activations; 1.2.5 Quantization

float-8 quantization

float-32 reference

Figure 10: ResNet-32 CIFAR-100 test performance of different 8-bit floating-point formats for rep-resentation of the activations without quantization of the input to the first layer. Test accuracy mean ± standard deviation over ten independent runs.

5 10 15 20 25 30 35 40 45 50 55 60

## Exponent Bias

y Weights; 1.6.1 Quantization

float-8 quantization

float-32 reference

2 4 6 8 10

## Exponent Bias

Weights; 1.3.4 Quantization

float-8 quantization

float-32 reference

## Exponent Bias

Weights; 1.2.5 Quantization

float-8 quantization

float-32 reference

Figure 11: ResNet-32 CIFAR-100 test performance of different 8-bit floating-point formats for rep-resentation of the weights. Test accuracy mean ± standard deviation over ten independent runs.

15 20 25 30 35 40 45 50 55 60 65 70

## Exponent Bias

Gradients ∇xL; 1.6.1 Quantization

float-8 quantization

float-32 reference

15 16 17 18

## Exponent Bias

Gradients ∇xL; 1.3.4 Quantization

float-8 quantization

float-32 reference

5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

## Exponent Bias

Gradients ∇xL; 1.2.5 Quantization

float-8 quantization

float-32 reference

Figure 12: ResNet-32 CIFAR-100 test performance of different 8-bit floating-point formats for rep-resentation of the gradients with respect to activations. Test accuracy mean ± standard deviation over ten independent runs.

5 10 15 20 25 30 35 40 45 50 55 60 65 70

## Exponent Bias

Gradients ∇wL; 1.6.1 Quantization

float-8 quantization

float-32 reference

6 8 10 12 14

## Exponent Bias

Gradients ∇wL; 1.3.4 Quantization

float-8 quantization

float-32 reference

3 4 5 6 7 8 9 10

## Exponent Bias

Gradients ∇wL; 1.2.5 Quantization

float-8 quantization

float-32 reference

Figure 13: ResNet-32 CIFAR-100 test performance of different 8-bit floating-point formats for rep-resentation of the gradients with respect to weights. Test accuracy mean ± standard deviation over ten independent runs.

## D HISTOGRAMS

A critical consideration in the selection of a reduced precision numerical format is the number range to be represented. To aid this selection, histograms of the appropriate quantities observed during a training session in which all quantities are represented with full precision have been generated. This section presents these histograms, together with a visual indication of how they relate to the limits of the candidate numerical formats.

Figures 14, 15 and 16 respectively show the histograms of the activations, weights and gradients observed at the beginning (blue curves) and end (red curves) of a 200 epoch training session us-ing ResNet-32 on the CIFAR-100 dataset. The histograms display the counts of exponent values, computed as log2 of the absolute values of the tensor components. Separate histograms are given for the first layer, the last (fully connected) layer, and the composite counts from all other layers (all of which are very similar, hence the relevant information is included in the counts obtained by summing over these layers).

Figure 17 shows the relationship between the chosen reduced precision formats and the histograms for the ResNet-32/CIFAR-100 combination. The range covered by the formats (including denorm range) is shown in the white area on the plots. The histogram data in this figure gives composite counts of the exponent values for the tensor components to which the reduced precision format in the subplot title has been applied (so they do not include first layer activations and gradients with respect to activations, both of which are left unquantized). Figure 19 shows the analogous data for ResNet-50 trained on the ImageNet dataset. Histograms of activations, weights and gradients of ResNet-18 trained on ImageNet are virtually identical to the ResNet-50 histograms of Figure 19, and have not been included.

Figure 18 gives the histograms for ResNet-32 CIFAR-100 training, together with the dynamic range covered by the scaled integer format 1.0.7.

Figures 20–23 show the histograms of activations, weights, gradients with respect to activations, and gradients with respect to weights of the different layer types of the Transformer, trained on the WMT14 English-German dataset. Figure 24 presents the composite histograms of the Transformer model together with the dynamic range of different 8-bit floating-point number formats that have been used throughout this paper. Finally, Figure 25 provides evidence that a model learned through quantized training has a similar distribution of weights as the corresponding float-32 baseline model.

−30 −20 −10 0 10 log2(|Activations|)

## First Layer

−30 −20 −10 0 10 log2(|Activations|)

## Composite Midlayer Convolutions

−30 −20 −10 0 10 log2(|Activations|)

## Fully Connected Layer

ResNet-32 Activations

first iteration last iteration

Figure 14: Histograms of the activations of different ResNet-32 layers for CIFAR-100 training.

−30 −20 −10 0 10 log2(|Weights|)

nt First Layer

−30 −20 −10 0 10 log2(|Weights|)

## Composite Midlayer Convolutions

−30 −20 −10 0 10 log2(|Weights|)

## Fully Connected Layer

ResNet-32 Weights

first iteration last iteration

Figure 15: Histograms of the weights of different ResNet-32 layers for CIFAR-100 training.

−30 −20 −10 0 10 log2(|Gradients ∇xL|)

## First Layer

−30 −20 −10 0 10 log2(|Gradients ∇xL|)

## Composite Midlayer Convolutions

−30 −20 −10 0 10 log2(|Gradients ∇xL|)

## Fully Connected Layer

ResNet-32 Gradients ∇xL

−30 −20 −10 0 10 log2(|Gradients ∇wL|)

## First Layer

−30 −20 −10 0 10 log2(|Gradients ∇wL|)

## Composite Midlayer Convolutions

−30 −20 −10 0 10 log2(|Gradients ∇wL|)

## Fully Connected Layer

ResNet-32 Gradients ∇wL

first iteration last iteration

Figure 16: Histograms of the gradients of different ResNet-32 layers for CIFAR-100 training.

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations 1.5.2,24

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights,1.5.2,28

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations 1.4.3,10

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights,1.4.3,14

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇x,1.5.2,32

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇w,1.5.2,32

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇x,1.4.3,20

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇w,1.4.3,16

first iteration last iteration

Figure 17: Histograms of the weights, activations and gradients of all layers for ResNet-32 CIFAR-100 training. The white areas correspond to the range of values representable by the respective 8-bit floating-point number format.

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations 1.0.7,-3

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights,1.0.7,1

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇x,1.0.7,10

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇w,1.0.7,3

first iteration last iteration

Figure 18: Histograms of the weights, activations and gradients of all layers for ResNet-32 CIFAR-100 training. The white areas correspond to the range of values representable by the 1.0.7 (scaled integer) number format.

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations 1.5.2,24

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights,1.5.2,28

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations 1.4.3,10

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights,1.4.3,14

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇x,1.5.2,32

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇w,1.5.2,32

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇x,1.4.3,20

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇w,1.4.3,16

first iteration last iteration

Figure 19: Histograms of the weights, activations and gradients of different layers for ResNet-50 ImageNet training. The white areas correspond to the range of values representable by the respective 8-bit floating-point number format.

−30 −20 −10 0 10 log2(|Activations|)

## All Activations

−30 −20 −10 0 10 log2(|Activations|)

## Attention Dense Q

−30 −20 −10 0 10 log2(|Activations|)

## Attention Dense V

−30 −20 −10 0 10 log2(|Activations|)

## Attention Dense K

−30 −20 −10 0 10 log2(|Activations|)

## Attention Logits Q

−30 −20 −10 0 10 log2(|Activations|)

nt Attention Logits K

−30 −20 −10 0 10 log2(|Activations|)

## Attention Output Weights

−30 −20 −10 0 10 log2(|Activations|)

## Attention Output V

−30 −20 −10 0 10 log2(|Activations|)

## Attention Output Transform

−30 −20 −10 0 10 log2(|Activations|)

## FFN Filter

−30 −20 −10 0 10 log2(|Activations|)

## FFN Output

−30 −20 −10 0 10 log2(|Activations|)

Pre-Softmax

## Transformer Activations

first iteration 200k iterations

Figure 20: Histograms of the activation of different layers of the Transformer for WMT14 English-German training.

−30 −20 −10 0 10 log2(|Weights|)

## All Weights

−30 −20 −10 0 10 log2(|Weights|)

## Attention Dense Q

−30 −20 −10 0 10 log2(|Weights|)

## Attention Dense V

−30 −20 −10 0 10 log2(|Weights|)

## Attention Dense K

−30 −20 −10 0 10 log2(|Weights|)

## Attention Output Transform

−30 −20 −10 0 10 log2(|Weights|)

## FFN Filter

−30 −20 −10 0 10 log2(|Weights|)

## FFN Output

−30 −20 −10 0 10 log2(|Weights|)

Pre-Softmax

## Transformer Weights

First Iteration 200k Iterations

Figure 21: Histograms of the weights of different layers of the Transformer for WMT14 English-German training.

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

All Gradients ∇xL

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## Attention Dense Q

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## Attention Dense V

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## Attention Dense K

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## Attention Logits

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

nt Attention Output

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## Attention Output Transform

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## FFN Filter

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

## FFN Output

−40 −30 −20 −10 0 log2(|Gradients ∇xL|)

Pre-Softmax

Transformer Gradients ∇xL

first iteration 200k iterations

Figure 22: Histograms of the gradients with respect to activations of different layers of the Trans-former for WMT14 English-German training.

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

All Gradients ∇wL

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## Attention Dense Q

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## Attention Dense V

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## Attention Dense K

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## Attention Output Transform

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## FFN Filter

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

## FFN Output

−40 −30 −20 −10 0 log2(|Gradients ∇wL|)

Pre-Softmax

Transformer Gradients ∇wL

first iteration 200k iterations

Figure 23: Histograms of the gradient with respect to weights of different layers of the Transformer for WMT14 English-German training.

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations, 1.5.2, 24

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights, 1.5.2, 28

−40 −30 −20 −10 0 10 log2(|Activations|)

Activations, 1.4.3, 10

−40 −30 −20 −10 0 10 log2(|Weights|)

Weights, 1.4.3, 16

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇xL, 1.5.2, 40

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇wL, 1.5.2, 40

−40 −30 −20 −10 0 10 log2(|Gradients ∇xL|)

Gradients ∇xL, 1.4.3, 28

−40 −30 −20 −10 0 10 log2(|Gradients ∇wL|)

Gradients ∇wL, 1.4.3, 24

## Transformer

first iteration 200k iterations

Figure 24: Histogram of the weights, activations and gradients of the Transformer model for WMT14 English-German training. The white areas correspond to the range of values representable by the respective 8-bit floating-point number format.

−40 −30 −20 −10 0 log2(|Weights|)

## All Weights

−40 −30 −20 −10 0 log2(|Weights|)

## Attention Dense Q

−40 −30 −20 −10 0 log2(|Weights|)

## Attention Dense V

−40 −30 −20 −10 0 log2(|Weights|)

## Attention Dense K

−40 −30 −20 −10 0 log2(|Weights|)

## Attention Output Transform

−40 −30 −20 −10 0 log2(|Weights|)

## FFN Filter

−40 −30 −20 −10 0 log2(|Weights|)

## FFN Output

−40 −30 −20 −10 0 log2(|Weights|)

Pre-Softmax

Transformer Weights After 200k Iterations

## Baseline Quantized Model

Figure 25: Histogram of the weights of the Transformer model for WMT14 English-German training after 200,000 iterations. The quantized model uses 1.4.3, bias 16 for weights, 1.4.3, bias 10 for activations and 1.5.2, bias 40 for gradients with respect to weights and activations.

## E LOSS SCALING FACTOR

Figure 26(a) reports the evolution of the values of the loss scaling factor that was used by the different automatic loss scaling algorithms, throughout the training of the Transformer model on the WMT14 English-German dataset. Figure 26(b) shows the dependence between the BLEU score and the loss scaling factor at the end of the training for the experiments presented in Figure 26(a).

0 100000 200000 300000 400000 Step

Backoff LogMax, 3σ

LogMax, 2σ

LogMax, 1σ

LogMax, 0σ

18 19 20 21 22 23 log2(Final scaling factor)

Backoff LogMax, 3σ

LogMax, 2σ

LogMax, 1σ

LogMax, 0σ

## Linear fit

Figure 26: (a) Loss scaling factor during training of the Transformer with different automatic loss scaling algorithms. The 1.5.2 format is used for activations (bias 24), weights (bias 28) and gradients ∇xL and ∇wL (bias 15). Each line corresponds to the average of five independent training runs. (b) The BLEU score plotted against the final loss scaling factor for the experiments presented in (a).

