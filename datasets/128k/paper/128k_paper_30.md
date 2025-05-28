<paper 0>
# Scalable Pre-training of Large Autoregressive Image Models 

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-01.jpg?height=726&width=1746&top_left_y=495&top_left_x=168)

Figure 1. AIM scaling behavior (Left) As we scale the capacity of AIM, we observe improved performance for the pre-training objective which directly correlates with stronger downstream performance. (Right) AIM exhibits stronger downstream performance when trained using larger sets of uncurated web data [32,33]. The downstream performance is the average attentive probe top-1 accuracy over a diverse set of 15 image recognition benchmarks. All models are trained for the same number of updates.


#### Abstract

This paper introduces AIM, a collection of vision models pre-trained with an autoregressive objective. These models are inspired by their textual counterparts, i.e., Large Language Models (LLMs), and exhibit similar scaling properties. Specifically, we highlight two key findings: (1) the performance of the visual features scale with both the model capacity and the quantity of data, (2) the value of the objective function correlates with the performance of the model on downstream tasks. We illustrate the practical implication of these findings by pre-training a 7 billion parameter AIM on 2 billion images that achieves $84.0 \%$ on ImageNet$1 k$ with a frozen trunk. Interestingly, even at this scale, we observe no sign of saturation in performance, suggesting that AIM potentially represents a new frontier for training large-scale vision models. The pre-training of AIM is similar to the pre-training of LLMs, and does not require any image-specific strategy to stabilize the training at scale.


[^0]
## 1. Introduction

Pre-training task agnostic models has become the standard in Natural Language Processing with the recent revolution of large language models (LLMs) [13, 64, 75]. These models can solve complex reasoning tasks from a few examples [13], follow instructions [59], and now serve as the engine of widely used AI assistants such as ChatGPT. A key factor contributing to their success is the ability to consistently improve as the capacity (i.e., number of parameters) or the amount of pre-training data [64] increases.

The scaling behavior of these models is remarkable for two key reasons. First, even though these models are trained with a simple objective - predicting the next word in a sentence given its past - they are able to learn intricate patterns over long contexts. Second, the scalability of this autoregressive objective is mostly observed when used in conjunction with certain architectures, and in particular Transformers [79], highlighting the potential synergy between the autoregressive pre-training and this architecture.

These observations naturally raise the follow-up question of whether the success of scaling Transformers with an autoregressive objective is exclusive to text. This is particularly significant considering that none of the aforemen-
tioned elements are inherently specific to language modeling. Autoregressive objectives take their roots in the data compression literature [69], and similar approaches have been investigated in audio [57] and images [18, 76]. The Transformer architecture has also been successfully used in other domains, in particular, computer vision with the success of the Vision Transformers (ViT) [29]. Therefore, as a first step towards generalizing the findings of LLMs, we explore if training ViT models with an autoregressive objective leads to competitive performance, in terms of learning representations, with the same scaling ability as LLMs.

In this paper, we introduce Autoregressive Image Models (AIM), an autoregressive approach for large-scale pretraining for visual features. We revisit prior work in autoregressive representation learning such as iGPT [18] using a modern toolset that includes vision transformers, collections of large-scale web data $[32,33]$ and recent advances in LLM pre-training [43, 75]. Additionally, we introduce two architectural modifications to adapt autoregressive pre-training to visual features. First, instead of restricting the self-attention to be fully causal as is typically the case for LLMs, we adopt a prefix attention, as in T5 [66]. This choice enables moving to a fully bidirectional attention during downstream tasks. Second, we use a heavily parameterized token-level prediction head, inspired by the heads used in contrastive learning [19]. We observe that this modification significantly improves the quality of the subsequent features with little overhead during training. Overall, the training of AIM is similar to the training of recent LLMs and does not rely on any stability-inducing techniques $[24,45,74]$ that supervised $[24,74]$ or selfsupervised $[5,58]$ methods need.

We provide a study of a series of models, ranging from $600 \mathrm{M}$ to 7B parameters pre-trained using 2B uncurated images with permissive licenses. Our AIM models exhibit strong scaling behavior w.r.t. the model size as shown in Figure 1 where higher capacity models achieve better downstream performance, measured as the average accuracy over 15 image recognition benchmarks. More importantly, there is a correlation between the value of our objective function on a validation set and the quality of the subsequent frozen features. This observation confirms that the autoregressive objective is adequate for the training of visual features. Furthermore, we observe consistent improvement in downstream performance as we train on more images, with no sign of saturation. Overall, these observations are aligned with the previous studies on scaling large language models.

## 2. Related Work

Autoregressive models. While most of the literature on autoregressive models come from language modeling [9, 53, 64] or speech [56, 57], few works have explored the potential of this approach for images [18, 49, 61, 61, 68, 76].
Of particular interest, Van den Oord et al. [76] show that using an architecture adapted to images, e.g., a convolution network, significantly improved over autoregressive models built with more generic architecture [77], e.g., a recurrent network [31]. Parmar et al. [61] further improve the quality of these autoregressive models by adopting the transformer architecture [79]. More recently, Chen et al. [18] have shown that scaling with more compute leads to continuous improvements. Our work follows this line of research, and we benefit from training on significantly more data, and further improvement in architecture design [29], training [73, 75] and understanding of the scaling law [43]. Concurrent to our work, Bai et al. [3] demonstrate the effectiveness of large-scale autoregressive vision models for in-context pixel prediction tasks (e.g., semantic segmentation, depth estimation).

Self-supervised pre-training. Pre-training vision models on datasets of images without supervision has been a fruitful area of research in recent years [10, 27, 28, 34, 54, 87, 88]. Different approaches have been employed, focusing on various proxy tasks for feature learning. For example, Noroozi and Favaro [55] learn to re-arrange the order of shuffled image patches. Some other works have relied on clustering $[7,14,17,83]$. Another popular approach involves the use of a contrastive objective, that resembles predictive coding, where the objective is to identify each image $[19,40]$. Most recent contrastive approaches include DINO [58], BYOL [38] or iBot [88]. In a similar vein, some works have proposed predictive approaches $[2,6]$ or a form of feature whitening [85]. Closer to our approach are works inspired by BERT [26] where patches are masked and predicted with an autoencoder in either their discrete [5] or pixel [41] form.

Other generative pre-training. Autoregressive modeling is a form of generative modeling, and few other generative approaches have been considered to learn visual features. The first category leverages some form of autoencoding where the pretext task corresponds to some denoising task. For instance, the noise can be salt-and-pepper [81] or masking [5, 62]. Another line of work leverages Generative Adversarial Networks (GANs) [35]. Most notably, BigGAN [12] trains a large GAN and re-uses the image discriminator to produce image features. More recently, DiffMAE [82] used diffusion models to learn image features.

Pre-training at scale. There are numerous works on scaling the pre-training of visual features with no supervision $[15,36,37,58,70,72]$. The most salient work in this area is DINOv2 where they produce the best self-supervised features by scaling the iBot method [88] on a private dataset of $142 \mathrm{M}$ images and a $460 \mathrm{M}$ parameter model. The conclusion from this work is that a carefully tuned contrastive method scales reasonably well, but they do not exhibit the scaling law that we observe with language modeling. They
also rely on an intricate implementation of contrastive learning to avoid the pitfalls described by Chen et al. [20]. In parallel, Singh et al. [70] study the scaling of Masked Autoencoders (MAE) [39]. While the study focuses on a weaklysupervised setup, it does not showcase strong improvements to the self-supervised pre-training as the data is scaled to billions of images. In contrast, we observe a clear benefit of scale on the quality of our features, even at a scale of a few billions of parameters and billions of images.

## 3. Pre-training Dataset

We pre-train our models on the DFN dataset introduced by Fang et al. [32]. This dataset is composed of a larger collection of 12.8B image-text pairs [33] filtered from Common Crawl. The data has been pre-processed to remove NSFW content, blur faces, and reduce contamination by deduplicating against the evaluation sets. A data filtering network [32] ranks the samples in the $12.8 \mathrm{~B}$ collection according to the alignment score between images and their corresponding caption. A subset of $2 \mathrm{~B}$ images, called DFN2B, has been extracted from the DataComp 12.8B dataset [33] by keeping the top $15 \%$ samples. Note that other than the privacy and safety filters, this process does not include any additional curation based on the image content. Since our pre-training does not require text, our method could be pre-trained using larger image collections that are not paired with captions or have low image-text alignment such as the rest of DataComp 12.8B.

Motivated by the common practice in LLM pretraining [75] of oversampling high-quality data sources such as Wikipedia and Books, during pre-training, we sample images from DFN-2B with a probability of $p=0.8$ and sample images from ImageNet- $1 \mathrm{k}$ with a probability of $p=0.2$. We refer to such dataset as DFN-2B+.

## 4. Approach

### 4.1. Training Objective

Our training objective follows that of a standard autoregressive model applied on a sequence of image patches. More precisely, an image $x$ is split into a grid of $K$ nonoverlapping patches $x_{k}, k \in[1, K]$, which collectively form a sequence of tokens. We assume that the sequence order is fixed across all images, and we use a raster (row-major) ordering by default unless otherwise specified. Given the above order, the probability of an image can be factorized as a product of patch conditional probabilities:

$$
\begin{equation*}
P(x)=\prod_{k=1}^{K} P\left(x_{k} \mid x_{<k}\right) \tag{1}
\end{equation*}
$$

where $x_{<k}$ denotes the set of the first $k-1$ patches, and is the context used to predict the $k^{\text {th }}$ patch. As opposed to language modeling, our sequences have a fixed length of $K$

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-03.jpg?height=691&width=743&top_left_y=256&top_left_x=1103)

Figure 2. AIM pre-training overview.. Input images are split into non-overlapping patches and embedded linearly following Dosovitskiy et al. [29]. The patch features are fed to a transformer in which the self-attention operation is causally masked to prevent attending to preceding positions. Afterward, a heavily parameterized MLP processes each of the patch features independently and finally projects it to pixel space. The targets correspond to the input sequence shifted one position to the left, requiring the model to predict the next patch in raster order.

that fits in memory and hence we do not need to truncate the context length. The training loss over a set $\mathcal{X}$ of images is then defined as the negative log-likelihood (NLL):

$$
\sum_{x \in \mathcal{X}} \sum_{k=1}^{K}-\log P\left(x_{k} \mid x_{<k}\right)
$$

Minimizing this objective over an infinite amount of images, with no further assumptions, is theoretically equivalent to learning the true underlying image distribution.

Prediction loss Our training objective naturally gives rise to certain variants of losses, each corresponding to a choice of the distribution $P\left(x_{k} \mid x_{<k}\right)$. By default, we adopt a normalized pixel-level regression loss similar to He et al. [41]. This loss corresponds to setting $P\left(x_{k} \mid x_{<k}\right)$ as Gaussian distributions with a constant variance. Namely, given $\hat{x}_{k}(\theta)$ as the prediction of the $k^{\text {th }}$ patch from a network parameterized with $\theta$, and $x_{k}$ as its corresponding ground-truth value, our objective is to minimize the sum $\ell_{2}$ squared distance between the prediction and the ground-truth:

$$
\begin{equation*}
\min _{\theta} \frac{1}{K} \sum_{k=1}^{K}\left\|\hat{x}_{k}(\theta)-x_{k}\right\|_{2}^{2} \tag{2}
\end{equation*}
$$

We also consider a cross-entropy loss with patches converted to discrete tokens using an offline tokenizer. Our ablation studies show that these designs work, although they do not produce as strong features as the pixel-wise loss.

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-04.jpg?height=309&width=352&top_left_y=257&top_left_x=217)

Pre-training (e.g. prefix len $=3$ )

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-04.jpg?height=323&width=352&top_left_y=256&top_left_x=583)

Downstream

Adaptation
Figure 3. Prefix causal attention. During pre-training we uniformly sample a prefix length $S$. The attention for the first $S$ patches are set to be bidirectional and loss is only computed for the remaining patches in the image. During adaptation to downstream tasks, this allows us to drop the attention causal mask, improving the downstream performance.

| Model | \#Params | Hidden size | Layers | LR | \#Patches | Batch size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AIM-0.6B | $0.6 \mathrm{~B}$ | 1536 | 24 | $1 e^{-3}$ | $0.5 \mathrm{~T}$ | 4096 |
| AIM-1B | $1.2 \mathrm{~B}$ | 2048 | 24 | $1 e^{-3}$ | $1.2 \mathrm{~T}$ | 4096 |
| AIM-3B | $2.7 \mathrm{~B}$ | 3072 | 24 | $1 e^{-3}$ | $1.2 \mathrm{~T}$ | 4096 |
| AIM-7B | 6.5B | 4096 | 32 | $1 e^{-3}$ | $1.2 \mathrm{~T}$ | 4096 |

Table 1. Model specifications. We provide the embedding dimension, number of layers, and parameter count for all AIM variants. We also provide the learning rate and batch size during pretraining. For AIM with 1B parameters and higher, the pre-training process involves $1.2 \mathrm{M}$ iterations, which corresponds to 1.2 trillion patches, or 5B images, seen during pre-training.

### 4.2. Architecture

As the backbone, we adopt the Vision Transformer architecture (ViT) [28]. For scaling in the model capacity, we follow the common practice in language modeling and we prioritize expanding width rather than depth $[64,75]$. In Table 1 , we provide an overview of the design parameters of AIM, including its depth and width, as well as the amount of data and optimization scheme for each model capacity. The overall model is illustrated in Figure 2.

During pre-training, we apply causal masks to the selfattention layers to model the probability of a patch given the preceding patches. More precisely, given a self-attention layer, the embedding for the patch $i$ is computed by:

$$
\begin{equation*}
y_{i}=\sum_{k=1}^{K} a_{i k} v_{i} \tag{3}
\end{equation*}
$$

where $a_{i k}$ is the attention weight and $v_{k}$ the value embedding. To enforce the desired constraints, we utilize a causal mask for the attention weights, where $a_{i k}=0$ for $k>i$, and $\sum_{k=1}^{K} a_{i k}=1$. This approach enables us to process the image with a single forward pass during training, without incurring additional computational overhead.

Prefix Transformer. The autoregressive objective in pretraining requires a causal mask in the self-attention operation. However, this differs from the standard usage of
ViT models in downstream tasks, where bidirectional selfattention is employed. This discrepancy leads to a decrease in performance, irrespective of whether the causal mask is retained during downstream adaptation or not (as shown in the ablations presented in Table 3). To address this issue, we propose to consider the initial patches of the sequence, referred to as the prefix, as a context for predicting the remaining patches following the PrefixLM formulation of Raffel et al. [65]. The prefix patches are excluded from the autoregressive prediction and therefore are not constrained to be causal. More precisely, we select a prefix length of size $S \in[1, K-1]$, and remove the causal mask, i.e., $a_{i, k}>0$ for $k<S$. This modification helps the model to work in the absence of causal masking, allowing it to be removed during downstream adaptation. This approach improves the performance of the model in downstream tasks and eliminates the need for architectural changes to ViT. Figure 3 illustrates the difference between causal and prefix attention.

MLP prediction heads. It is a common practice to adopt certain prediction heads during pre-training, which are discarded when transferring to downstream tasks $[16,17,19$, 20, 38]. The purpose of these heads is to prevent the trunk features from becoming too specialized in the pre-training objective, thus enhancing their suitability for downstream transfer. We opt for a simple design where we use $N$ blocks of MLP on top of the final transformer layer, processing each patch independently. We observed that this design strikes a good balance between performance and the additional costs incurred during pre-training.

Straightforward implementation. It is worth noting that AIM does not require particular optimization stabilityinducing mechanisms such as LayerScale [74], stochastic depth [45], QK-Norm [24], or freezing the patch projector [20]. These mechanisms have been crucial for the success of other methods, either supervised or self-supervised. On the contrary, we observe that AIM scales using the same set of optimization hyperparameters across model sizes with no further tuning (see Table 1).

We add sinusoidal positional embeddings [79] to the input patches before the transformer and before the MLP head. We use a standard expansion ratio of 4 for all the MLP blocks in the trunk and the head. We drop the bias term for simplicity, and unlike the original ViT, we do not append a classification token to the input. By default, we use 12 blocks for the MLP head for all model capacities. The pixel targets are normalized per patch before the loss computation following He et al. [41]. We train our model using bfloat16 precision. We use the AdamW [52] optimizer with linear warmup and a cosine decay schedule. We detail the hyperparameters used for pre-training and downstream adaptation in Appendix D.

Downstream adaptation. Pre-training large-scale models is a resource-intensive process, and even fine-tuning them
![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-05.jpg?height=480&width=1720&top_left_y=221&top_left_x=164)

Figure 4. AIm pre-training across model sizes. We observe a clear improvement in the performance of the pre-training objective with increasing the capacity of AIM. Moreover, the downstream performance (IN-1k top-1) is monotonically improving for higher capacity models as well as with longer pre-training. We do not observe clear signs of plateauing during pre-training even after training for $500 \mathrm{k}$ iterations, indicating that AIM can benefit from even longer pre-training schedules. Note that the loss saturation at the very end of training is caused by the cosine decay schedule where the learning rate is effectively zero.

is demanding. Consequently, we focus on scenarios where all model weights are fixed for downstream tasks. In this context, we only train a classification head, which mitigates the risk of overfitting on small downstream datasets and significantly reduces the adaptation cost.

Unlike contrastive learning, our loss is computed independently for each patch. This means that our pre-training does not incorporate any notion of global image descriptors, and hence, we do not have any image level token. While some methods rely on global average pooling to build a global feature from the patch features, we find that our approach, along with other generative approaches like MAE, benefit more from an attention pooling operation [50] placed before the linear classifier. Other works [1, 74, 84] have adopted this attention pooling to improve performance with minimal increase in parameters or FLOPs.

Specifically, given a set of patch features $P=\left\{p_{i} \mid 1 \leq\right.$ $i \leq K\}$, we compute a global descriptor $\hat{p}$ through multihead attention pooling over the patch features as:

$$
\begin{equation*}
\hat{p_{h}}=\sum_{i=1}^{K} \frac{\exp \left(q_{h}^{T} W_{h}^{k} p_{i}\right)}{\sum_{j=1}^{K} \exp \left(q_{h}^{T} W_{h}^{k} p_{j}\right)} W_{h}^{v} p_{i} \tag{4}
\end{equation*}
$$

where for each attention head $h=\{1, \ldots, H\}, W_{h}^{k}, W_{h}^{v} \in$ $R^{d_{h} \times d}$ correspond to the key and value weight matrices, respectively; $q_{h}$ is a learnable query vector. And we obtain the pooled feature as $\hat{p}=\left[p_{1}, \ldots, p_{H}\right], \hat{p} \in R^{d}$, which serves as the input to the linear classifier. By default, we set the number of heads $H=\frac{d}{d_{h}}$, which makes the total number of learnable parameters $2 d^{2}+d$, a negligible cost compared to the main model size. Including this attention pooling makes the entire operation not strictly linear, and, therefore we refer to it as "Attentive Probe". Nevertheless, the advantages of linear probing, e.g., low additional parameter count and a reduced risk of overfitting, are preserved with this probe.

## 5. Results

### 5.1. Impact of scaling

We measure the impact when scaling our approach in terms of parameters and training data. In particular, we investigate whether there is a correlation between the pre-training objective and the downstream performance across benchmarks. We also look at the effect of scaling on the value of the loss function. For all of these experiments, we report the value of our loss function on the validation set of $\mathrm{IN}-1 \mathrm{k}$. Loss and performance during training. In Figure 4, we measure for each model the value of the pre-training loss and the classification accuracy on the validations set, as a function of the number of training iterations. We observe that both probes improve accordingly during the entire training, showing that optimizing our objective directly results in better downstream performance.

Number of parameters. We observe that the loss value and the accuracy of the downstream task improve as we scale the capacity of our models. This observation is consistent with the trend observed in LLMs and can be directly attributed to the optimization of our objective function, which in turn leads to the learning of stronger representations.

Number of images. In Figure 5, we show the progression of the validation loss as we pre-train on either a small curated dataset of $1 \mathrm{M}$ images, i.e., $\mathrm{IN}-1 \mathrm{k}$, or a larger set of 2B images, i.e. $\mathrm{DFN}-2 \mathrm{~B}+$. It is not surprising that training on $\mathrm{IN}-1 \mathrm{k}$ leads rapidly to a low validation loss as measured on the same distribution. However, this loss deteriorates at the end of the training, indicating an overfitting to the training data. When training on the uncurated DFN-2B dataset, the model starts from a higher validation loss but the loss continues to decrease with no sign of overfitting. When the same dataset is augmented with a small amount of $\mathrm{IN}-1 \mathrm{k}$ data, as detailed in $\S 3$, we observe further improvement in

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-06.jpg?height=458&width=854&top_left_y=232&top_left_x=164)

Figure 5. Dataset impact on pre-training performance. On the one hand, pre-training using $\mathrm{IN}-1 \mathrm{k}$ leads to overfitting, even for the AIm-0.6B model. On the other hand, pre-training using the uncurated DFN-2B dataset prevents overfitting but converges to a similar point due to the distributional shift. Pre-training on DFN$2 \mathrm{~B}+$, a data mixture that predominantly consists of $\mathrm{DFN}-2 \mathrm{~B}$ with a small presence of $\mathrm{IN}-1 \mathrm{k}$ samples leads to the best performance.

the performance that eventually surpasses pre-training on $\mathrm{IN}-1 \mathrm{k}$. We confirm that the resulting model also leads to a better downstream performance in Table 2.

$$
\begin{array}{lccc}
\text { pre-training dataset } & \mathrm{IN}-1 \mathrm{k} & \mathrm{DFN}-2 \mathrm{~B} & \mathrm{DFN}-2 \mathrm{~B}+ \\
\hline \text { attentive } & 73.5 & 74.5 & \mathbf{7 5 . 6}
\end{array}
$$

Table 2. Dataset impact of downstream performance ( 15 benchmarks). The behavior in Figure 5 is consistent with the downstream performance where we observe that using a data mixture of DFN-2B and IN-1k results in the best performance.

Compute-optimal pre-training. Since we do not observe signs of overfitting when we train using the DFN-2B+ dataset, we proceed to examine the impact of extending the length of our pre-training schedule. In Figure 6, we study the impact of increasing the length of the pre-training schedule from $500 \mathrm{k}$ to $1.2 \mathrm{M}$ iterations, i.e., $2 \mathrm{~B}$ to $5 \mathrm{~B}$ images seen during pre-training. We observe that models pretrained with a longer schedule achieve significantly lower validation loss. This suggests that one can improve the performance of AIM either by increasing the model capacity or by pre-training for longer schedules. Interestingly, we find that lower-capacity models trained for a longer schedule achieve comparable validation loss to higher-capacity models trained for a shorter schedule while using a similar amount of FLOPs. This finding is consistent with Hoffmann et al. [43] and implies that AIM could follow similar scaling laws. However, we defer further investigations in this aspect for future work.

### 5.2. Architecture and Design

In this section, we investigate the impact of some variations in our model and training objective. These ablations are conducted using an AIM-0.6B model, which has been pretrained and evaluated on the $\mathrm{IN}-1 \mathrm{k}$ dataset. The results of these ablations are presented in Table 3.

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-06.jpg?height=455&width=816&top_left_y=236&top_left_x=1077)

Figure 6. Scaling in FLOPs. That total number of FLOPs during training correlates with the final validation loss, suggesting compute driven scaling law similar to Hoffmann et al. [43].

Targets and objective (a). We explore various potential representations for the target patches. One approach is to utilize the raw pixel values, and training the model with mean squared error (MSE) regression loss. A second option, proposed by He et al. [41], involves using per-patch normalized pixel values instead of the raw signal with the same MSE loss. Finally, another option is to use a discretized representation of the patches, either using k-means or a discrete VAE $[67,78]$. In this case, the model is trained using a cross-entropy objective similar to language modeling. Our experiments show that AIM performs best when using the MSE objective with normalized pixel values.

Autoregression pattern (b). Autoregressive pre-training typically follows a specific order of traversal to facilitate the prediction of the next token. In the case of language, the traversal pattern is clear, as text is read and written one word at a time in a sequential manner (e.g., left to right for English). However, for images, determining the traversal pattern is less obvious. We explore various deterministic patterns, including raster, spiraling out, checkerboard, and randomly pre-sampled patterns. Detailed examples of each pattern are found in Appendix B. Even though our model performs reasonably well with each pattern, we observe that the raster pattern leads to significantly higher performance.

To gain deeper insights into this result, we examine the difficulty of predicting patches along sequences for each pattern. This can be done by measuring the loss value per patch as we progress along a sequence, as illustrated in Figure 7. Our observation is that patterns that present a more uniform distribution of difficulty across patches result in superior models, as compared to patterns where the prediction becomes progressively easier as the sequence unfolds. We attribute this to the difficulty of predicting patches throughout the sequence that forces the model to retain more information about the image. This leads to better patch features, and consequently, to better image representation as a whole. Cropping scale (c). We explore the impact of the information content of each patch by adjusting the lower bound of the cropping scale. On the one hand, opting for a crop-

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-07.jpg?height=127&width=678&top_left_y=240&top_left_x=211)

(a) Targets.

| pattern | raster | spiral | checkerboard | random |  | crop scale | 0.08 | 0.4 | 1.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| linear | $\mathbf{6 9 . 5}$ | 67.7 | 68.2 | 65.8 |  | linear | 68.4 | $\mathbf{7 0 . 0}$ | 49.6 |
| attentive | $\mathbf{7 7 . 4}$ | 76.3 | 76.0 | 75.7 |  | attentive | 77.7 | $\mathbf{7 8 . 2}$ | 63.5 |

(b) Autoregression Pattern (causal).

| head | None | MLP | Transformer |
| :--- | :---: | :---: | :---: |
| linear | 64.0 | 70.0 | $\mathbf{7 0 . 5}$ |
| attentive | 75.4 | 78.2 | $\mathbf{7 8 . 5}$ |

(e) Head Design. (c) Crop Scale.

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-07.jpg?height=116&width=320&top_left_y=468&top_left_x=1534)

(f) Architecture.

Table 3. Ablations We investigate various design choices of AIM. We use an AIM-0.6B model that is pre-trained and evaluated using $\mathrm{IN}-1 \mathrm{k}$. We report the linear and attentive probing results. The default settings for AIM used for the main results are highlighted in gray .

ping scale that is too small leads to an easier next-patchprediction task as neighboring patches' similarity increases. On the other hand, using a large cropping scale can lead to severe overfitting unless the dataset size is sufficiently large. Since this study is conducted using $\mathrm{IN}-1 \mathrm{k}$, we observe a clear drop in performance due to overfitting.

Causal vs. Prefix Attention (d). We measure the impact of incorporating prefix attention during pre-training, as opposed to using standard causal attention. We observe that pre-training with causal self-attention produces models that are effective in downstream transfer tasks only when the causal mask is preserved. These models experience a significant decline in performance when bidirectional attention is employed. However, pre-training with prefix attention leads to models that operate effectively in both causal and bidirectional modes. Notably, the best performance is achieved when combining prefix attention during pre-training with bidirectional attention during downstream adaptation.

![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-07.jpg?height=467&width=829&top_left_y=1588&top_left_x=168)

Figure 7. Autoregression patterns We explore a number of patterns for the autoregressive traversal of an image. The set of image patches is broken into equal-sized chunks and the validation loss is measured per chunk. We observe that the way the task difficulty is distributed across chunks varies strongly among patterns.

Head design (e). We consider different types of heads on top of the backbone to make predictions at the pixel level. Using no heads (i.e. None) performs reasonably well, but adding an MLP further improves the quality of the back-

| width | 512 | 1024 | 2048 |
| :--- | :---: | :---: | :---: |
| linear | 69.4 | 69.6 | $\mathbf{7 0 . 0}$ |
| attentive | 77.7 | 78.1 | $\mathbf{7 8 . 2}$ |

(a) MLP width.

| depth | 6 | 8 | 12 |
| :--- | :---: | :---: | :---: |
| linear | 65.3 | 68.1 | $\mathbf{7 0 . 0}$ |
| attentive | 76.2 | 77.1 | $\mathbf{7 8 . 2}$ |

(b) MLP depth.
Table 4. MLP design. We vary the capacity of the MLP head by changing the number of MLP blocks (i.e. depth) or the embedding size (i.e. width). Downstream performance improves with more capacity in either width or depth, but depth has more impact.

|  | autoregressive | masked image modeling |  |
| :---: | :---: | :---: | :---: |
|  |  | ratio $=50 \%$ | ratio $=75 \%$ |
| attentive | $\mathbf{7 8 . 2}$ | 70.3 | 77.8 |

Table 5. Autoregressive vs. Masking We evaluate the IN-1k performance of the autoregressive objective of AIM, in comparison to the masking objective $[5,26]$. We keep all the other architectural and optimization components fixed. We observe that, under the same pre-training settings, the frozen-trunk performance of the autoregressive objective outperforms masking.

bone. Interestingly, replacing the MLP with a full-fledged transformer of the same depth and width only yields a marginal performance improvement but at a significantly higher computational cost. Therefore, we opt to use an MLP head in our approach. We hypothesize that these heads specialize in capturing the low-level signals necessary for accurate pixel-level prediction. By incorporating a head, the trunk can learn higher-level features that are more suitable for downstream transfer. A similar design was employed for contrastive learning to prevent the backbone from specializing in predicting specific image transformations [19].

Deeper vs. Wider architecture (f). We present the design specifications of AIM in Table 1, outlining its width and depth. Unlike the original design of ViT [29], where the depth is scaled more rapidly than the width, we adopt a scaling strategy similar to that of Llama [75]. This allows us to scale our model more gracefully while maintaining a reasonable depth. We validate the effectiveness of a wider architecture in Table 3f. Our findings indicate that even for the relatively small-scale AIM-0.6B model, a wider architecture not only delivers strong performance but also im-

| Model | Arch. | Data | ![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-08.jpg?height=136&width=68&top_left_y=245&top_left_x=666) | ![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-08.jpg?height=136&width=68&top_left_y=245&top_left_x=742) | ![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-08.jpg?height=136&width=63&top_left_y=245&top_left_x=820) | 8 <br> 氖 | $\vec{o}$ <br> $\bar{z}$ <br> $\circ$ <br> $\dot{x}$ | $\stackrel{\rightharpoonup}{\Delta}$ | $\frac{\pi}{2}$ | 氖 | E <br> U् <br> $\sum$ | $\stackrel{N}{\pi}$ <br> $\frac{\lambda}{0}$ <br> $\tilde{U}$ <br> Ü | $\sum_{U}^{2}$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-08.jpg?height=136&width=63&top_left_y=245&top_left_x=1515) | $\mathbb{Z}$ <br> o <br> 业 | $\sum_{3}^{3}$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_83266dd69b37cf715ec9g-08.jpg?height=136&width=69&top_left_y=245&top_left_x=1745) | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DINO [17] | ViT-B/8 | $\overline{\mathrm{N}-1 \mathrm{k}}$ | $\overline{0.1}$ | 66.0 | 97.8 | 87.3 | 89.5 | $\overline{78.4}$ | 92.3 | 89.2 | $\overline{58.5}$ | 93.7 | 90.2 | 6.1 | 98.2 | 57.0 | 41.1 | $\overline{75.0}$ |
| iBOT [88] | ViT-L/16 | IN-21k | 83.5 | 70.5 | 99.2 | 93.3 | 93.5 | 81.6 | 92.8 | 90.8 | 61.8 | 94.5 | 90.0 | 5.9 | 98.0 | 60.3 | 47.7 | 77.6 |
| DINOv2 [58] | ViT-g/14 $4_{516}$ | LVD | 86.4 | 84.5 | 99.6 | 95.2 | 96.3 | 86.3 | 96.4 | 95.6 | 68.2 | 96.5 | 90.7 | 8.0 | 98.6 | 66.7 | 58.8 | 81.9 |
| BEiT [5] | ViT-L/14 | IN-21k | 62.2 | 44.4 | 94.4 | 78.7 | 79.0 | 64.0 | 80.9 | 69.5 | 52.0 | 92.8 | 88.2 | 4.2 | 97.5 | 47.7 | 25.9 | 65.4 |
| $\operatorname{MAE}[41,70]$ | ViT-H/14 | $\mathrm{IN}-1 \mathrm{k}$ | 80.9 | 64.6 | 97.1 | 85.8 | 90.2 | 78.1 | 95.0 | 93.7 | 58.1 | 94.2 | 89.8 | 5.4 | 98.1 | 56.9 | 42.2 | 75.3 |
|  | ViT-2B/14 | S-3B | 82.2 | 70.8 | 97.5 | 87.3 | 93.4 | 81.2 | 95.1 | 94.9 | 57.8 | 94.4 | 90.3 | 7.3 | 98.2 | 60.1 | 50.2 | 77.4 |
| M- |  | $\mathrm{DFN}-2 \mathrm{~B}+$ | 8.5 | 64.0 | 97.2 | 86.8 | 90.1 | 80.1 | 93.0 | 93.0 | 57.9 | 94.3 | 90.0 | 7.8 | 98.4 | 58.3 | 45.2 | 75.6 |
| AIM-1B | ViT-1B/14 |  | 80.6 | 67.2 | 98.2 | 88.3 | 91.6 | 81.8 | 93.4 | 93.9 | 58.6 | 94.5 | 90.0 | 9.0 | 98.6 | 59.8 | 47.5 | 76.9 |
| AIM-3B | ViT-3B/14 |  | 82.2 | 69.7 | 98.4 | 89.9 | 92.7 | 81.9 | 94.1 | 93.8 | 58.8 | 94.3 | 90.4 | 9.7 | 98.5 | 60.9 | 48.9 | 77.6 |
| AIM-7B | ViT-7B/14 |  | 82.4 | 70.9 | 98.6 | 90.0 | 93.1 | 82.3 | 93.8 | 92.1 | 59.5 | 93.6 | 90.7 | 10.1 | 98.6 | 61.7 | 49.6 | 77.8 |
| AIM-7B $\dagger$ | ViT-7B/14 | $\mathrm{DFN}-2 \mathrm{~B}+$ | 84.0 | 75.5 | 98.9 | 91.8 | 94.1 | 85.6 | 95.4 | 95.0 | 61.4 | 94.2 | 90.5 | 8.4 | 98.5 | 63.5 | 57.7 | 79.6 |

Table 6. Downstream evaluation with a frozen trunk. We assess the quality of AIM features by evaluating against a diverse set of 15 image recognition benchmarks. AIM and the baseline methods are evaluated using attentive probing with a frozen trunk. AIM models exhibit a strong performance across all benchmarks, especially the AIm-7B. AIM outperforms all other methods, using joint-embedding or generative approaches, except for DINOv2 which utilizes higher-resolution images, that typically results in a 1-1.5\% improvement on ImageNet for instance. $\dagger$ : Extracting features from the $20^{\text {th }}$ layer instead of the last $\left(32^{\text {nd }}\right)$, see Table 7 for more details.

proves training stability. This observation supports the notion that some of the insights gained from training LLMs can be similarly applied to other domains.

Attentive $v s$. Linear probe. For all ablations we report the linear and attentive probing results. We observe that, consistently across all experiments, attentive pooling provides a significant boost to performance as it allows for a more nuanced aggregation of local features circumventing one of the main weaknesses of generative pre-training: the absence of an image-level global descriptor.

Structure of the MLP. The MLP plays an important role as ablated in Table 3e. In Table 4, we further investigate the capacity of the MLP head and how it impacts downstream performance. We vary the capacity of the head by either changing the number of MLP blocks or their width. By default, we use a head of 12 blocks and an embedding dimension of 2048. First, we observe that increasing the capacity of the MLP either through depth or width leads to consistent improvement in the downstream performance. Second, we find that increasing the number of MLP blocks, with a fixed width, leads to a larger improvement compared to increasing the width for a fixed depth. Interestingly, we could not find a point where increasing the MLP capacity failed to yield further improvements. We did not explore higher capacities beyond those reported in Table 4 as it would lead to models with disproportionate head and trunk capacity.

### 5.3. Pre-training objective

Autoregressive vs. Masking We conduct a comparison between our architecture trained with an autoregressive objective and the masking objective popularized by BERT [26] for language, and by BEiT and MAE for vision. It is im- portant to note that we applied the masking objective in the same setting as AIM, thereby isolating the impact on the performance of the pre-training objective from other design choices that differ between AIM and other approaches. In the masking baseline, we randomly sample masks and replace the masked patches with learnable mask tokens.

In Table 5, we show that AIM performs better with an autoregressive objective than a masking objective. This is consistent with the results reported by Chen et al. [18], providing further evidence that our improvements stem from the utilization of an autoregressive objective.

### 5.4. Comparison with other methods

In Table 6, we compare the attentive probing performance of AIM to other state-of-the-art methods across a set of 15 diverse benchmarks that are detailed in Appendix A.

Generative methods. AIM provides a strong performance compared to its generative counterparts. AIM outperforms BEiT [5] by a large margin. Additionally, AIM-0.6B provides a better performance, averaged across all benchmarks, compared to MAE-H [41] which has an equivalent capacity. Moreover, we compare against the MAE-2B [70] model which has been pre-trained on IG-3B, a private dataset of 3 billion images from Instagram. We find that both AIM-3B and AIM-7B outperform MAE-2B, with AIM-7B exhibiting a particularly large improvement. It is worth noting that, similar to AIM, two other generative approaches, BEiT and MAE, benefit from attentive probing, thereby narrowing the gap between generative and joint embedding methods.

Joint embedding methods. AIM provides a competitive performance with joint embedding methods such as DINO [17], iBOT [88], and DINOv2 [58]. In terms of
average accuracy across all benchmarks, AIM outperforms DINO and iBOT. However, it falls behind DINOv2 which achieves its results by evaluating with higher-resolution inputs. Note that AIM attains such competitive performance using higher capacity trunks. Nevertheless, AIM's pretraining is significantly simpler and can be trivially scaled in terms of parameters and data, yielding consistent improvements. On the contrary, state-of-the-art joint embedding methods like DINOv2 heavily rely on a number of tricks, such as multi-crop augmentation, KoLeo regularization, LayerScale, Stochastic Depth, schedules for teacher momentum and weight decay, and high-resolution fine-tuning in order to achieve strong performance.

Extracting stronger features. We observe that higherquality features can be extracted from shallower layers compared to the last layer's features. This is likely due to the generative nature of the pre-training objective that is inherently different than the discriminative downstream tasks and therefore, the features with the highest semantic content do not necessarily concentrate around the last layer. In Table 7, we report the $\mathrm{IN}-1 \mathrm{k}$ top-1 accuracy for features extracted from the last layer compared to the layer with the highest performance. A more detailed analysis of this phenomenon is provided in Appendix D.

|  | AIM-0.6B | AIM-1B | AIM-3B | AIM-7B |
| :--- | :---: | :---: | :---: | :---: |
| last layer | 78.5 | 80.6 | 82.2 | 82.4 |
| best layer | $\mathbf{7 9 . 4}$ | $\mathbf{8 2 . 3}$ | $\mathbf{8 3 . 3}$ | $\mathbf{8 4 . 0}$ |

Table 7. Feature extraction. The highest quality features after AIM pre-training typically reside in shallower layers than the last. Extracting features from earlier layers leads to a non-negligible boost to the recognition performance on $\mathrm{IN}-1 \mathrm{k}$.

### 5.5. Low-Rank Adaptation

In addition to frozen-trunk evaluation, we examine LowRank Adaptation (LoRA) [44], a popular and efficient finetuning method. We report the results of LoRA fintuning of AIM in Table 8. We observe that LoRA is compatible with AIM, leading to a large boost in performance compared to frozen-trunk evaluation. For example, AIM-7B improves by $3.9 \%$ (compared to the last layer's performance) while finetuning only $0.1 \%$ percent of the trunk parameters.

|  | AIM-0.6B | AIM-1B | AIM-3B | AIM-7B |
| :--- | :---: | :---: | :---: | :---: |
| attentive | 78.5 | 80.6 | 82.2 | 82.4 |
| LoRA (rank=8) | 81.0 | 83.6 | 85.5 | 86.3 |

Table 8. Low-rank adaptation (IN-1k). AIM is compatible with LoRA showing large gains compared to frozen-trunk evaluations.

## 6. Discussion

In this paper, we presented a simple and scalable method for pre-training vision models at scale without supervision. We employed a generative autoregressive objective during pretraining and proposed several technical contributions to better adapt it for downstream transfer. Consequently, we observed a number of desirable properties for our Autoregressive Image Models. First, the capacity of our models can be effortlessly scaled to 7 billion parameters using a vanilla transformer implementation, without resorting to stabilityinducing techniques or extensive adjustments of hyperparameters for each model scale. Second, AIM's performance on the pre-training task has a strong correlation with downstream performance. Third, AIM achieves strong performance across 15 recognition benchmarks, outperforming prior state-of-the-art methods like MAE and significantly narrowing the gap between generative and joint embedding pre-training approaches. Finally, we did not observe any clear signs of saturation as we scale either in terms of parameters or data, suggesting that there is a potential for further performance improvements with larger models trained for even longer schedules. We hope that AIM serves as a seed for future research in scalable vision models that effectively leverage uncurated datasets without any bias towards object-centric images or strong dependence on captions.

Limitations. AIM excels in its seamless scalability and its effective utilization of large volumes of uncurated image data. However, alternative methods can offer different trade-offs. MAE [41] provides high sample efficiency and can learn good representations using a small amount of pretraining data, reducing the risk of overfitting [30] in contrast to our approach. Contrastive methods [17, 58, 88] currently result in stronger representations for a given model size compared to generative approaches such as MAE and AIM, but pose significant challenges in terms of scalability and loss tractability due to the complexity of their objective.

## Acknowledgements

The authors would like to thank Brandon McKinzie, Samira Abnar, Preetum Nakkiran, and Jiatao Gu for valuable feedback throughout the project. We thank Edouard Grave and Hervé Jegou for their inspiring discussions during the earlier stages of the project. We thank Marco Cuturi, James Thornton, Pierre Ablin, and Eugene Ndiaye for their support and for many fruitful discussions throughout the project. Finally, we would like to thank the entire Machine Learning Research team at Apple for many helpful discussions and assistance with infra and data.

## References

[1] Anonymous. V-JEPA: Latent video prediction for visual representation learning. In Submitted to The Twelfth International Conference on Learning Representations, 2023. 5

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023. 2

[3] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785, 2023. 2

[4] Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE Transactions on Medical Imaging, 2018. 13

[5] Hangbo Bao, Li Dong, and Furu Wei. BEiT: Bert pretraining of image transformers. In ICLR, 2022. 2, 7, 8

[6] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for selfsupervised learning. In ICLR, 2022. 2

[7] Miguel A Bautista, Artsiom Sanakoyeu, Ekaterina Tikhoncheva, and Bjorn Ommer. Cliquecnn: Deep unsupervised exemplar learning. Advances in Neural Information Processing Systems, 29, 2016. 2

[8] Sara Beery, Elijah Cole, and Arvi Gjoka. The iwildcam 2020 competition dataset. arXiv preprint arXiv:2004.10340, 2020. 13

[9] Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. Advances in neural information processing systems, 13, 2000. 2

[10] Piotr Bojanowski and Armand Joulin. Unsupervised learning by predicting noise. In International Conference on Machine Learning, pages 517-526. PMLR, 2017. 2

[11] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 - mining discriminative components with random forests. In ECCV, 2014. 13

[12] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 2

[13] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. preprint arXiv:2005.14165, 2020. 1

[14] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018. 2

[15] Mathilde Caron, Piotr Bojanowski, Julien Mairal, and Armand Joulin. Unsupervised pre-training of image features on non-curated data. In Proceedings of the IEEE/CVF Inter- national Conference on Computer Vision, pages 2959-2968, 2019. 2

[16] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. 4

[17] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2, 4, 8, 9

[18] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 2, 8, 14

[19] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 2, 4, 7

[20] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 3, 4

[21] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. 13

[22] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. In CVPR, 2014. 13

[23] Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. Autoaugment: Learning augmentation strategies from data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 113-123, 2019. 14

[24] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML. PMLR, 2023. 2, 4

[25] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248-255. Ieee, 2009. 13

[26] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2018. 2, 7, 8

[27] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015. 2

[28] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. Advances in neural information processing systems, 27, 2014. 2, 4

[29] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth $16 \times 16$ words: Transformers for image recognition at scale. In ICLR, 2021. 2, 3, 7

[30] Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé Jegou, and Edouard Grave. Are large-scale datasets necessary for self-supervised pre-training? arXiv preprint arXiv:2112.10740, 2021. 9

[31] Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990. 2

[32] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 1, 2,3

[33] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023. 1, 2, 3

[34] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018. 2

[35] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2

[36] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In ICCV, 2019. 2

[37] Priya Goyal, Quentin Duval, Isaac Seessel, Mathilde Caron, Mannat Singh, Ishan Misra, Levent Sagun, Armand Joulin, and Piotr Bojanowski. Vision models are more robust and fair when pretrained on uncurated images without supervision. arXiv preprint arXiv:2202.08360, 2022. 2

[38] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020. 2, 4

[39] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In ICCV, 2017. 3

[40] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 2

[41] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2, 3, 4, 6, 7, 8, 9

[42] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification, 2017. 13

[43] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 2, 6

[44] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 9
[45] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In $E C C V$, 2016. 2,4

[46] iNaturalist 2018 competition dataset. iNaturalist 2018 competition dataset. https://github.com/visipedia/inat_ comp/tree/master/2018, 2018. 13

[47] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13), Sydney, Australia, 2013. 13

[48] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 13

[49] Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 29-37. JMLR Workshop and Conference Proceedings, 2011. 2

[50] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In ICML, 2019. 5

[51] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017. 14

[52] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 4, 14

[53] Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernockỳ, and Sanjeev Khudanpur. Recurrent neural network based language model. In Interspeech, 2010. 2

[54] Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In CVPR, 2020. 2

[55] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In $E C C V$, 2016. 2

[56] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016. 2

[57] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. In NeurIPS, 2018. 2

[58] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 2, 8, 9, 14

[59] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. 1

[60] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012. 13

[61] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, 2018. 2

[62] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016. 2

[63] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In ICCV, 2019. 13

[64] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019. 1, 2, 4

[65] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020. 4

[66] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 2020. 2

[67] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821-8831. PMLR, 2021. 6,7

[68] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017. 2

[69] Claude E Shannon. Prediction and entropy of printed english. Bell system technical journal, 30(1):50-64, 1951. 2

[70] Mannat Singh, Quentin Duval, Kalyan Vasudev Alwala, Haoqi Fan, Vaibhav Aggarwal, Aaron Adcock, Armand Joulin, Piotr Dollár, Christoph Feichtenhofer, Ross Girshick, et al. The effectiveness of mae pre-pretraining for billionscale pretraining. arXiv preprint arXiv:2303.13496, 2023. $2,3,8$

[71] J. Taylor, B. Earnshaw, B. Mabey, M. Victors, and J. Yosinski. Rxrx1: An image set for cellular morphological variation across many experimental batches. In ICLR, 2019. 13

[72] Yonglong Tian, Olivier J Henaff, and Aäron van den Oord. Divide and contrast: Self-supervised learning from uncurated data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10063-10074, 2021. 2

[73] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers \& distillation through attention. In ICML, 2021. 2

[74] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. arXiv preprint arXiv:2103.17239, 2021. 2, 4,5
[75] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2, 3, 4, 7

[76] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 2

[77] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747-1756. PMLR, 2016. 2

[78] Aaron Van Den Oord, Oriol Vinyals, et al. Neurips. Advances in neural information processing systems, 2017. 6

[79] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1, 2, 4

[80] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In Medical Image Computing and Computer Assisted Intervention-MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11, pages 210-218. Springer, 2018. 13

[81] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Léon Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(12), 2010. 2

[82] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. arXiv preprint arXiv:2304.03283, 2023. 2

[83] Xueting Yan, Ishan Misra, Abhinav Gupta, Deepti Ghadiyaram, and Dhruv Mahajan. ClusterFit: Improving Generalization of Visual Representations. In CVPR, 2020. 2

[84] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. TMLR, 2022. 5

[85] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In ICML, 2021. 2

[86] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR, 2018. 14

[87] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In $E C C V, 2016.2$

[88] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. In ICLR, 2022. 2, 8, 9
</end of paper 0>


<paper 1>
# ARVideo: Autoregressive Pretraining for Self-Supervised Video Representation Learning 

Sucheng Ren $^{1} \quad$ Hongru Zhu $^{1} \quad$ Chen Wei ${ }^{1} \quad$ Yijiang Li $^{1} \quad$ Alan Yuille ${ }^{1} \quad$ Cihang Xie $^{2}$<br>${ }^{1}$ Johns Hopkins University $\quad{ }^{2}$ UC Santa Cruz


#### Abstract

This paper presents a new self-supervised video representation learning framework ARVideo, which autoregressively predict the next video token in a tailored sequence order. Two key designs are included. First, we organize autoregressive video tokens into clusters that span both spatially and temporally, thereby enabling a richer aggregation of contextual information compared to the standard spatialonly or temporal-only clusters. Second, we adopt a randomized spatiotemporal prediction order to facilitate learning from multi-dimensional data, addressing the limitations of a handcrafted spatial-first or temporal-first sequence order. Extensive experiments establish ARVideo as an effective paradigm for self-supervised video representation learning. For example, when trained with the ViT-B backbone, ARVideo competitively attains $81.2 \%$ on Kinetics- 400 and $70.9 \%$ on SomethingSomething V2, which are on par with the strong benchmark set by VideoMAE. Importantly, ARVideo also demonstrates higher training efficiency, i.e., it trains $14 \%$ faster and requires $58 \%$ less GPU memory compared to VideoMAE.


## 1 Introduction

The transformer architecture, as introduced in Vaswani et al. [41], has fundamentally transformed the field of natural language processing (NLP) through its ability to model long-range dependencies with minimal inductive bias. A crucial catalyst for its success lies in self-supervised learning of robust and transferable representations from large volumes of unlabeled data. Within this paradigm, masked language modeling (MLM) [7] and autoregressive modeling (AR) [34, 5, 30] stand out as two leading approaches. Specifically, MLM masks random portions of input tokens and trains models to predict masked elements; whereas AR predicts subsequent words in a sequence based on all preceding words. These methods have propelled state-of-the-art performance in various NLP tasks.

In the video domain, however, the landscape is different. Previous studies have predominantly relied on supervised pretraining using image datasets, often overlooking the critical aspect of temporal dynamics [26, 4]. Recently, there has been a shift towards leveraging NLP-inspired mask language modeling [7] or image-inspired mask image modeling [18, 2] to directly exploit unlabeled video datasets for pretraining. For instance, VideoMAE [39, 13] introduces mask autoencoder [18] for self-supervised video video representation learning; BEVT [44] learns spatial representations from image data and joint-masked image and video modeling. Despite these advancements, autoregressive modeling-another powerful self-supervised learning approach in NLP—-has yet to be extensively explored within the context of video data analysis.

Critically, applying autoregressive pretraining to video data entails the same principle of autoregressively predicting the next element in a sequential order based on its predecessors. In natural language, these elements-words-are clearly defined and inherently follow a chronological order. For images, elements could be conceptualized as pixels or patches arranged in a flattened sequence [6, 10, 36]. The further transition to video data, however, introduces additional complexity due to its inherently multidimensional nature (i.e., including both spatial and temporal dimensions). This raises a crucial
![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-02.jpg?height=334&width=1230&top_left_y=284&top_left_x=368)

tokens
![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-02.jpg?height=590&width=1282&top_left_y=368&top_left_x=388)

Figure 1: ARVideo autoregressive predicts spatiotemporal cluster from grouping tokens span spatial and temporal dimension.

inquiry: how should we define an autoregressive 'video element' and establish a visual sequence order for self-supervised video representation learning?

We note traditional methods, such as converting video into a sequence of cubes [39, 4, 44, 26] and subsequently linearly mapping these cubes into video tokens, generally reveal critical limitations in addressing this query. Specifically, the granularity of these video tokens often fails to encapsulate the rich semantics typically represented by words in text-based models-primarily because 1) these video tokens are too dimensionally limited, and 2) video inherently lacks a sequential order in its spatial dimensions, although it retains this feature in its temporal aspects.

To address these challenges, we hereby present ARVideo, a novel autoregressive-based video representation learning paradigm with two key designs (see Figure 11. Firstly, we redefine 'video elements' by grouping video tokens into spatiotemporal video clusters, differentiating from conventional singledimensional strategies like spatial video clusters or temporal video clusters. This approach improves semantic representation by aggregating more contextually relevant multidimensional information. Secondly, we find that, compared to well-defined yet single-dimensional spatial-first or temporal-first sequence orders, a sequence order that randomly integrates both spatial and temporal dimensions empirically yields significantly stronger results. This suggests that effectively capturing the inherent multidimensionality of video data is crucial for autoregressive modeling. Extensive experiments establish our ARVideo as an effective paradigm for video representation learning. For example, while the autoregressive video representation learning baseline only attains $74.2 \%$ on Kinetics-400 and $66.4 \%$ on Something-Something V2, ARVideo significantly boosts the results to $81.2 \%(+7 \%)$ and $70.9 \%(+4.5 \%)$, respectively. Notably, these results not only match but, in some aspects, surpass the strong benchmark set by VideoMAE, particularly with respect to training efficiency-ARVideo achieves faster training speeds by $14 \%$ and reduces GPU memory consumption by $58 \%$.

## 2 Related Work

### 2.1 Video Representation Learning

Video representation learning has witnessed significant exploration, historically driven by supervised learning methods [40, 43, 37, 4, 26] that pretrain backbone networks on labeled image or video data before fine-tuning. However, such methods face challenges due to inherent discrepancy between image and video data, compounded by the scarcity of comprehensively labeled video datasets.

In the era of self-supervised learning, recent work have designed pre-tasks incorporating temporal information for self-supervised video representation learning [48, 3, 20, 33, 35] and leveraging contrastive learning for effective visual representations [33, 22, 24, 8, 16, 17]. Additional, mask reconstruction-based methods inspired by masked language modeling [7] are introduced into selfsupervised image and video representation learning. For example, MAE [18] presents a scalable self-supervised learning method to reconstruct masked image patches while VideoMAE [39] extends this approach to video data and reconstructs masked spacetime patches. BEVT [45] separates spatial learning from temporal dynamics, training on masked images initially before jointly on masked images and videos. Christoph et al. [13] introduce an efficient video-based MAE extension with minimal biases and significant speedups. In contrast to prior works, our ARVideo proposes a new path for self-supervised video representation learning via autoregressive pretraining.

### 2.2 Autoregressive Pretraining

As a representative approach for autoregressive pretraining, Generative Pretrained Transformer (GPT) trains language models by autoregressively predicting the next word based on all preceding words in a sentence. Inspired by the success of autoregressive modeling in NLP, researchers start to apply autoregressive pretraining in computer vision. ImageGPT [6] learns effective image representations by training a Transformer to autoregressively predict image pixels without any prior knowledge of their 2D structure. SAIM [32] adopts an encoder to autoregressively learn contextual information like a standard vision transformer (ViT) and a decoder to predict the current content, mutually reinforcing each other's functions. RandSAC [19] arranges image tokens into segments for parallel intra-segment and sequential inter-segment autoregressive prediction. However, applying autoregressive pretraining on video data faces notable challenges due to the extra temporal dimension. ARVideo explores the design of autoregressive video elements and visual sequence orders for video representation learning.

## 3 Method

In this section, we first revisit GPT [34] and ImageGPT [6] to establish the foundation for the proposed ARVideo, as illustrated in Figure 1. We then analyze the inherent difference between image and video data, followed by the design of elements and the optimal prediction order as the key ingredients in ARVideo for autoregressive prediction with videos.

### 3.1 Generative Pretrained Transformer

We first outline the Generative Pretrained Transformer (GPT) framework. Consider an unlabeled language dataset $\mathcal{U}$ comprising sentences $\left[u^{1}, \ldots, u^{N}\right]$, where each sentence $u^{j}$ consists of words $u^{j}=\left\{u_{1}^{j}, \ldots, u_{n}^{j}\right\}$. GPT [34] autoregressively predicts the next word given all preceding words, minimizing the negative log-likelihood with model parameter $\theta$ :

$$
\begin{equation*}
p\left(u^{j}\right)=-\log \prod_{i=1}^{n} p\left(u_{i}^{j} \mid u_{1}^{j}, \ldots, u_{i-1}^{j}, \theta\right) \tag{1}
\end{equation*}
$$

This modeling strategy has fundamentally changed the landscape of natural language processing, leading to the development of tremendously successful models like ChatGPT [34] and GPT-4 [30].

### 3.2 ImageGPT

Transitioning from natural language processing to image processing necessitates the design of image elements for autoregressive prediction. In ImageGPT, it treats individual pixels as elements. Specifically, given an image $x \in R^{H \times W \times C}$, ImageGPT flattens it into a 1D pixel sequence of length $N=H \times W$, and autoregressively predicts the next pixel given all preceding pixels:

$$
\begin{equation*}
p(x)=-\log \prod_{i=1}^{N} p\left(x_{i} \mid x_{1}, \ldots, x_{i-1}, \theta\right) \tag{2}
\end{equation*}
$$

This approach incurs significant computational overhead due to the quadratic complexity of selfattention w.r.t. the input sequence length. ImageGPT thereby uses smaller image sizes (e.g., $32 \times 32$ ) in pretraining, yielding suboptimal performance. This limitation is pertinent in our development of ARVideo and becomes more pronounced due to the added complexity of video data.

![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-04.jpg?height=326&width=336&top_left_y=257&top_left_x=404)

(a) Video Tokens

![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-04.jpg?height=233&width=233&top_left_y=306&top_left_x=835)

(b) Spatial

Cluster

![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-04.jpg?height=174&width=166&top_left_y=344&top_left_x=1161)

(c) Temporal Cluster

![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-04.jpg?height=171&width=163&top_left_y=348&top_left_x=1447)

(d) Spatiotemporal Cluster

Figure 2: Comparison between video token and different cluster.

### 3.3 ARVideo

Illustrated in Figure 1. ARVideo autoregressively pretrains on video data $x \in \mathcal{R}^{T \times H \times W \times C}$. Note that directly extending ImageGPT to videos faces significant challenges, primarily due to the added temporal dimension, which would significantly escalate computational demands, even with lowresolution videos like $4 \times 32 \times 32$. Moreover, pixels as autoregressive elements lack semantic richness compared to words in the language, further necessitating pixel grouping strategies to enhance representation learning. To better facilitate learning from multi-dimensional video data, we also explore prediction orders across spatial and temporal dimensions.

### 3.3.1 Pixel grouping

From Pixels to Video Tokens. With patch embeddings in ViT, videos can be patchified into nonoverlapping cubes [39, 4, 44, 26] of size $P_{T} \times P_{W} \times P_{H}$. Then, each cube is transformed into a video token through a linear projection layer, resulting in $N=\frac{T}{P_{T}} \times \frac{H}{P_{H}} \times \frac{W}{P_{W}}$ video tokens. This tokenization significantly reduces operational elements, thus alleviating computational demands while ensuring that each video token encapsulates richer semantics compared to individual pixels. For example, as reported in Table 1, using video tokens as autoregressive elements for pretraining significantly outperforms approaches without tokenization by $3.3 \%$ while keeping pretraining resolution consistent with previous work [39, 44].

| Element | Resolution | Something- Something V2 |
| :---: | :---: | :---: |
| Pixel | $8 \times 14 \times 14$ | 60.7 |
| Token | $16 \times 224 \times 224$ | 64.0 |

Table 1: Grouping pixels into video tokens facilitates autoregressive pretraining on higher-resolution videos and improves performance by $3.3 \%$.

This promising transition from pixels to video tokens introduces a compelling query: Can further performance gains be realized by aggregating more tokens? In pursuit of this, we examine three options: grouping video tokens into spatial, temporal, or spatiotemporal clusters. It is important to note that within each cluster, video tokens are always fully attended to each other. This fullattention configuration helps to enable a more effective consolidation of semantic content within each autoregressive element.

From Tokens to Spatial Clusters. As shown in Figure 2(b), we strategically group spatially neighbored tokens-those sharing the same temporal positions but varying spatially-into spatial clusters. Following the patch embedding step, video tokens within the spatial domain $\frac{H}{P_{H}} \times \frac{W}{P_{W}}$ are grouped into one element, resulting in $\frac{T}{P_{T}}$ autoregressive elements. For example, a video of size $16 \times 224 \times 224$ with a cube embedding size of $2 \times 16 \times 16$ [39] here will be transformed into 8 autoregressive elements, with each element comprising $14 \times 14$ tokens.

From Tokens to Temporal Clusters. As illustrated in Figure 2(c), our method integrates temporal information by grouping tokens that are temporally adjacent into temporal clusters. Specifically, tokens within the temporal domain $\frac{T}{P_{T}}$ are grouped into one element, resulting in $\frac{H}{P_{H}} \times \frac{W}{P_{W}}$ autoregressive elements. For instance, a video of size $16 \times 224 \times 224$ with a cube embedding size of $2 \times 16 \times 16$ [39] here will transformed into $14 \times 14$ autoregressive elements, with each element comprising 8 tokens.

From Tokens to Spatiotemporal Clusters. Moving beyond the single-dimensional grouping strategies discussed above, we now consider the inherently multidimensional nature of video data by grouping neighboring $K_{T} \times K_{H} \times K_{W}$ tokens into spatiotemporal clusters with no overlaps, as illustrated in Figure 2(d). This strategy results in a total number of $\frac{T}{P_{T} K_{T}} \times \frac{H}{P_{H} K_{H}} \times \frac{W}{P_{W} K_{W}}$ clusters, with each containing both spatial and temporal information as an autoregressive element.

### 3.3.2 SpatialTemporal Prediction Order

For the spatiotemporal cluster, we further explore its prediction order. Specifically, this strategy is expected to yield $\frac{T}{P_{T} K_{T}}$ clusters at each spatial position, and $\frac{H}{P_{H} K_{H}} \times \frac{W}{P_{W} K_{W}}$ clusters at each temporal position.

Pre-defined order. We implement two systematic strategies: a spatial-first order and a temporal-first order. The spatial-first approach prioritizes autoregressive pretraining within the $\frac{H}{P_{H} K_{H}} \times \frac{W}{P_{W} K_{W}}$ spatiotemporal clusters along the spatial dimension, before transitioning to clusters in subsequent temporal positions. Conversely, the temporal-first approach prioritizes within the $\frac{T}{P_{T} K_{T}}$ spatiotemporal clusters along the temporal dimension, then proceeds to clusters in subsequent spatial positions.

Random Rasteration. Inspired by the random sentence permutation technique used in XLNet [49] for enhancing autoregressive pretraining, our random rasterization approach scrambles the order of clusters randomly during autoregressive pretraining. This method avoids the constraints of fixed sequential patterns, such as spatial-first or temporal-first, and allows ARVideo to adaptively model both long- and short-range spatial-temporal information. Such flexibility in autoregressive prediction orders not only captures the inherent multidimensionality of video data more effectively but also fosters a richer, more comprehensive video representation.

### 3.3.3 Model Architecture

We adopt the ViT [9, 39] as the encoder. For the decoder, we take the Transformer decoder with cross attention but without self-attention. This design choice aims to simplify the decoding process, emphasizing interaction between the encoded inputs while reducing training costs. The query of the decoder is randomly initialized but includes position information to facilitate sequence generation. Our model utilizes a strategically designed attention mask as in previous work [6, 34] to enable efficient autoregressive prediction in a parallel computation framework. When transferring to downstream tasks, we remove the decoder and only finetune the encoder.

## 4 Experiment

### 4.1 Dataset and Implementation Details

We primarily evaluate ARVideo on Kinetics-400 [21] and Something-Something V2 [14]. Specifically, Kinetics-400 contains 400 classes and $260 \mathrm{k}$ videos of 10 s, with $240 \mathrm{k}$ for training and $20 \mathrm{k}$ for validation; Something-Something V2 contains 174 classes with $169 \mathrm{k}$ videos for training and $25 \mathrm{k}$ for validation. While Kinetics- 400 provides a broad spectrum of actions with minimal context, Something-Something V2 focuses more on the interaction of actions with objects.

For our experiments, we first pretrain a vanilla video Transformer [39] with ARVideo, and then fine-tune the pretrained model on the target action recognition datasets. Additionally, we assess the feature transferability on AvA v2.2 [15] and HMDB [23]. AvA v2.2 is a human action localization dataset with $211 \mathrm{k}$ videos for training and $57 \mathrm{k}$ for validation; HMDB is a small video dataset with $3.5 \mathrm{k}$ videos for training and $1.5 \mathrm{k}$ videos for validation.

We follow the established protocol in prior work [39] to train our models. Instead of using negative log-likelihood as in GPT [34], we employ mean square error (MSE) loss to measure the discrepancy between the predicted and target cubes, as utilized in MAE [18]. We randomly mask $80 \%$ tokens in each element to reduce the overall training costs; note that, unlike MAE or VideoMAE, we do not reconstruct those masked regions.

| Method | Backbone | pretrain | Epoch | Frames | GFLOPs | Param | Top-1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Supervised pretraining |  |  |  |  |  |  |  |
| TANet [29] | ResNet152 | $\mathrm{IN}-1 \mathrm{~K}$ | 100 | 16 | $242 \times 4 \times 3$ | 59 | 79.3 |
| $\mathrm{TDN}_{E n}[42]$ | ResNet101 | $\mathrm{IN}-1 \mathrm{~K}$ | 100 | $8+16$ | $198 \times 10 \times 3$ | 88 | 79.4 |
| TimeSformer 4 | ViT-B | IN-21K | 15 | 8 | $196 \times 1 \times 3$ | 121 | 78.3 |
| Motionformer [31] | ViT-B | $\mathrm{IN}-21 \mathrm{~K}+\mathrm{K} 400$ | 35 | 16 | $370 \times 1 \times 3$ | 109 | 81.1 |
| Video Swin 27 | Swin-B | $\mathrm{IN}-21 \mathrm{~K}+\mathrm{K} 400$ | 30 | 32 | $321 \times 1 \times 3$ | 88 | 82.7 |
| Mask video modeling |  |  |  |  |  |  |  |
| VIMPAC [38] | ViT-L | HowTo100M | 100 | 10 | $\mathrm{~N} / \mathrm{A} \times 10 \times 3$ | 307 | 77.4 |
| BEVT 44$]$ | Swin-B | K400 | 150 | 32 | $282 \times 1 \times 3$ | $88 \quad$ | 76.2 |
| VideoMAE 39 | ViT-B | K400 | 800 | 16 | $180 \times 2 \times 3$ | 87 | 80.0 |
| VideoMAE 39 | ViT-B | K400 | 1600 | 16 | $180 \times 2 \times 3$ | 87 | 81.5 |
| Autoregressive pretraining |  |  |  |  |  |  |  |
| iGPT [6] | ViT-B | $\mathrm{IN}-1 \mathrm{~K}$ | 300 | 16 | $180 \times 2 \times 3$ | 87 | 61.2 |
| Randsac 19 | ViT-B | $\mathrm{IN}-1 \mathrm{~K}$ | 1600 | 16 | $180 \times 2 \times 3$ | 87 | 70.3 |
| TokenGPT $\dagger$ | ViT-B | $\mathrm{IN}-1 \mathrm{~K}$ | 300 | 16 | $180 \times 2 \times 3$ | 87 | 68.5 |
| TokenGPT $\dagger$ | ViT-B | K400 | 800 | 16 | $180 \times 2 \times 3$ | 87 | 74.2 |
| ARVideo | ViT-B | K400 | 800 | 16 | $180 \times 2 \times 3$ | 87 | 80.1 |
| ARVideo | ViT-B | K400 | 1600 | 16 | $180 \times 2 \times 3$ | 87 | 81.2 |

Table 2: Comparison with the state-of-the-art methods on Kinetics-400. "Ex. labels $\boldsymbol{X}$ " means only unlabelled data is used during the pretraining phase. "N/A" indicates the numbers are not available for us. $\dagger$ indicates the implementation by us with the token replacing pixel in iGPT.

| Method | Backbone | Pretrain | Epoch | Frames | GFLOPs | Param | Top-1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Supervised pretraining |  |  |  |  |  |  |  |
| $\operatorname{TEINet}_{E n}[28]$ | ResNet50 52 | IN-1K | 50 | $8+16$ | $99 \times 10 \times 3$ | 50 | 66.5 |
| $\operatorname{TANet}_{E n}[29]$ | ResNet $50 \times 2$ | IN-1 | 50 | $8+16$ | $99 \times 2 \times 3$ | 51 | 66.0 |
| $\mathrm{TDN}_{E n}$ | ResNet101×2 | IN-1K | 60 | $8+16$ | $198 \times 1 \times 3$ | 88 | 69.6 |
| SlowFast $[12$ | ResNet101 | K40 | 196 | $8+32$ | $106 \times 1 \times 3$ | 53 | 63.1 |
| MViTv1 [11] | MViTv1-B | K4 | 100 | 64 | $455 \times 1 \times 3$ | 37 | 67.7 |
| TimeSformer [4] | ViT-B | $\mathrm{IN}-2$ | 15 | 8 | $196 \times$ | 121 | 59.5 |
| er [4] | ViT | IN-2 | 15 | 64 | $5549 \times 1 \times 3$ |  | 62.4 |
| $\mathrm{ViVi}^{\prime}$ | Vi7 | IN-21K | 35 | 32 | 995 | N/A | 65.9 |
| er 31 | $V^{\prime}$ |  | 35 | 16 |  |  | 66.5 |
| Video Swin 27 | Swin-B | IN-21K | 30 | 32 | $321 \times 1 \times 3$ | 88 | 69.6 |
| Mask video modeling |  |  |  |  |  |  |  |
| VIMPAC [38] | ViT- -1 | OwT | 10 | 10 | N/A | 307 | 68.1 |
| BEVT 44 | Swin-B | $\mathrm{J}-1 \mathrm{~K}+\mathrm{K} 400$ | 15 | 32 | 321 | 88 | 70.6 |
| MaskFeat $\uparrow 12 \sqrt{47}$ | MViT-L | K600 | 1600 | 40 | $2828 \times 1 \times 3$ | 218 | 75.0 |
| $\mathrm{E}[39]$ | ViT-B | $\mathrm{SSv} 2$ | 800 | 16 | $180 \times 2 \times 3$ | 87 | 69.6 |
| VideoMAE [39] | ViT-B | SSv2 | 2400 | 16 | $180 \times 2 \times 3$ | 87 | 70.8 |
| Autoregressive pretraining |  |  |  |  |  |  |  |
| iGPT 6] | ViT-B | $\mathrm{IN}-1 \mathrm{~K}$ | 30 | 1 | $2 \times 3$ | 87 | 54.3 |
| Randsac 19 | ViT-B | IN-1K | 160 | 1 | $\times 3$ | 87 | 59.6 |
| TokenGPT $\dagger$ | Vi] | IN-1K | 30 | 16 | $180 x$ | 87 | 59.2 |
| GPT $\dagger$ | ViT | SS | $80 \quad x$ | 16 |  | 87 | 66.4 |
|  |  |  | 80 | 1 |  | 87 | 69.8 |
| ARVideo | ViT-B | SSv2 | 2400 | 16 | $180 \times 2 \times 3$ | 87 | 70.9 |

Table 3: Comparison with the state-of-the-art methods on Something-Something V2. "Ex. labels $\boldsymbol{x}$ " means only unlabelled data is used during the pretraining phase. "N/A" indicates the numbers are not available for us. $\dagger$ indicates the implementation by us with the token replacing pixel in iGPT.

### 4.2 Main results

Kinetics-400. We pretrain the ViT-B backbone for both 800 and 1600 epochs on Kinetics-400, and report the corresponding results in Table 2. Notably, ARVideo attains $80.1 \%$ top-1 accuracy under 800 epochs and $81.2 \%$ top-1 accuracy under 1600 epochs, exhibiting significant improvements over previous autoregressive methods. Specifically, taking 1600-epoch-pretrained ARVideo for comparison, it outperforms iGPT, the baseline model, by a striking $\mathbf{+ 2 0 . 0 \%}$, and Randsac, the previous state-of-the-art autoregressive model on images, by $\mathbf{+ 1 0 . 9 \%}$. Additionally, compared to

| Method | $\mathrm{K} 400 \rightarrow$ AVA v2.2 | $\mathrm{K} 400 \rightarrow \mathrm{HMDB}$ |
| :--- | :--- | :---: |
| Contrastive Learning <br> MoCo | - | 67.9 |
| Mask video modeling <br> VideoMAE | 26.7 | 73.3 |
| Autoregressive pretraining <br> ARVideo | 26.9 | 74.1 |

Table 4: Comparison of model transferability. We first pretrain models on Kinetics-400, and then transfer them to AVA v2.2 and HMDB.

| Method | Encoder |  | Decoder |  | Training Time | GPU Memory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathrm{Q}$ | Key/Value | $\mathrm{Q}$ | Key/Value |  |  |
| VideoMAE | 160 | 160 | 1568 | 1568 | $145 \mathrm{~h}$ | $41.3 \mathrm{G}$ |
| ARVideo | 300 | 300 | 1372 | 300 | $\mathbf{1 2 7 h}(-12.4 \%)$ | $\mathbf{2 6 . 1 G}(-36.8 \%)$ |

Table 5: The comparison of pretraining time and GPU memory.

TokenGPT, which performs token-level autoregressive prediction, ARVideo showed advancements of $\mathbf{+ 1 2 . 7 \%}$ when TokenGPT was pretrained on an image dataset, and $\mathbf{+ 7 . 0 \%}$ when it was pretrained on the Kinetics-400 dataset.

Moreover, we note that ARVideo performs competitively against the strong benchmark-the mask video modeling method, VideoMAE. For example, the performance difference between ARVideo and VideoMAE is only $0.1 \%$ with 800 epochs of pretraining; this margin remains minimal at $0.3 \%$ with 1600 epoch pretraining. These results validate the effectiveness of ARVideo as a pioneering autoregressive pretraining method in self-supervised video representation learning, equalling-and in some aspects surpassing - the performance of established mask modeling methods.

Something-Something V2. We pretrain the ViT-B backbone for 800 and 2400 epochs on the Something-Something V2 dataset. As reported in Table 3. ARVideo achieves top-1 accuracies of $69.8 \%$ and $70.9 \%$ for 800 and 2400 epochs, respectively, which are significantly stronger than prior autoregressive pretraining methods. For example, under 2400 epochs, ARVideo surpassed the baseline model iGPT by $\mathbf{+ 1 6 . 6 \%}$ and outperforms the best-performing image-based autoregressive method, Randsac, by $\mathbf{+ 1 1 . 3 \%}$. It also surpassed TokenGPT pre-trained on image datasets by $+11.7 \%$ and on the Something-Something V2 dataset by $+4.5 \%$. Additionally, when compared to the strong masked video modeling method VideoMAE, ARVideo also performs competitively in both 800 epochs of pretraining (i.e., $0.2 \%$ accuracy difference) and 2400 epochs of pretraining (i.e., $0.1 \%$ accuracy difference). Together with the observations in Kinetics-400, these results can establish ARVideo as a strong alternative to masked modeling approaches for video analysis.

Transfer Learning. To investigate the feature transferability of ARVideo, we transfer the model trained on Kinetics-400 to AvA v2.2 and HMDB. We can observe that ARVideo demonstrate strong transferability, achieving $26.9 \mathrm{mAP}$ on AvA v2.2 and 74.1\% Top-1 accuracy on HMDBoutperforming both VideoMAE and MoCo (see Table 4). For example, compared to VideoMAE, ARVideo shows (slight) improvements of $0.2 \%$ on AvA v 2.2 and $0.8 \%$ on HMDB.

Computation cost. We report the training time and GPU memory usage in Table 5 (with ViT-B trained on Kinetics-400 for 800 epochs, using $8 \times$ A6000). Compared to VideoMAE, ARVideo presents significant reductions in both GPU memory usage and training time-ARVideo reduces training cost by $12.4 \%$ (from 145 hours to 127 hours) and GPU memory consumption by $36.8 \%$ (from 41.3G to $26.1 \mathrm{G}$ ). This advantage stems from ARVideo's shorter sequence length as we drop the last cluster in the autoregressive modeling.

Attention rank. The self-attention mechanism computes attention scores for a given input sequence, forming what is known as the attention map. The rank of this matrix can serve as a measure of its ability to capture complex patterns in the data. Typically, high-rank attention matrices suggest a model that can capture a wide variety of patterns and relationships within the data, while low-rank matrices may suggest a model that does not well utilize its full capacity or operates on simpler data [46]. Following this instruction, we plot the rank of the attention map in each layer of VideoMAE and our ARVideo in Figure 3 We can observe that, across nearly all layers except the $6_{t h}$, ARVideo maintains higher attention ranks than VideoMAE, indicating a stronger representational ability of our model's self-attention layers.

![](https://cdn.mathpix.com/cropped/2024_06_04_72835303ac79fd4fafcfg-08.jpg?height=710&width=767&top_left_y=244&top_left_x=668)

Figure 3: The attention rank comparison between VideoMAE and ARVideo

| case | $K_{T}$ | $K_{H}$ | $K_{W}$ | Something-Something V2 |
| :---: | :---: | :---: | :---: | :---: |
| Token/Cube | 1 | 1 | 1 | 64.0 |
| spatial cluster | 1 | $\frac{H}{P_{H}}$ | $\frac{H}{P_{H}}$ | 66.0 |
| spatial cluster | 1 | 7 | 7 | 66.2 |
| temporal cluster | $\frac{T}{P_{T}}$ | 1 | 1 | 65.2 |
| temporal cluster | 2 | 1 | 1 | 65.6 |
| spatiotemporal cluster | 4 | 7 | 7 | 65.5 |
| spatiotemporal cluster (ARVideo) | 2 | 7 | 7 | $\mathbf{6 6 . 8}$ |

Table 6: Ablation study on the cluster shape.

### 4.3 Ablation Study

In this part, we ablate four factors-cluster shape, mask ratio, prediction order, and decoder design. Note that, unless otherwise specified, all ablations are conducted on the ViT-B backbone with 200 epochs of pretraining.

Cluster shape. We group neighboring and non-overlapped $K_{T} \times K_{H} \times K_{W}$ tokens into one cluster and analyze the effect of different cluster shapes. Three situations are considered: 1) $K_{T}=K_{W}=$ $K_{H}=1$, equivalent to the TokenGPT, which pertains autoregressively at the token/cube level; 2) $K_{T}=\frac{T}{P_{T}}, K_{W}=K_{H}=1$, representing a temporal cluster; and 3) $K_{T}=1, K_{W}=\frac{W}{P_{W}}, K_{H}=$ $\frac{H}{P_{H}}$, representing a spatial cluster.

We report the results in Table 6. Firstly, we can observe that all clustered configurations significantly enhance performance over the TokenGPT baseline. For example, simply grouping tokens into spatial/temporal/spatiotemporal clusters yields $2.0 \% / 2.2 \% / 2.8 \%$ improvements, respectively. Then, when comparing different clusters, we note that our spatiotemporal cluster (ARVideo) with $K_{T}=$ $2, K_{W}=K_{H}=7$ attains the best performance of $66.8 \%$, outperforming the best-performed spatial cluster ( $\left.K_{T}=1, K_{W}=K_{H}=7\right)$ by $0.8 \%$ and the best-performed temporal clusters ( $K_{T}=2, K_{W}=K_{H}=1$ ) by $1.2 \%$. However, it is interesting to note that, if a poorly designed spatiotemporal cluster ( $K_{T}=4, K_{W}=K_{H}=7$ ) is used, the performance will drop to $65.5 \%$.

Prediction order. In our evaluation of prediction order, which plays an important role in constructing the video sequence, we first check with the predefined spatial-first and temporal-first orders. As shown in Table 7, temporal-first order achieves $66.0 \%$ top-1 accuracy, which is $0.4 \%$ higher than spatial-first order. However, our randomized spatial-temporal prediction order, adept at learning both long- and short-range spatial-temporal dynamics, exhibits a superior performance of $66.8 \%$, surpassing the predefined spatial-first approach by $1.2 \%$ and the temporal-first approach by $0.8 \%$.

| Order | SSv2 |
| :--- | :--- |
| Spatial-First | 65.6 |
| Temporal-First | 66.0 |
| Spatial-temporal random | $\mathbf{6 6 . 8}$ |

Table 7: Ablation study on the prediction order.

| Mask Ratio | SSv2 |
| :---: | :---: |
| $75 \%$ | 66.0 |
| $80 \%$ | 66.8 |
| $90 \%$ | 65.6 |
| $95 \%$ | 64.8 |

Table 8: Ablation study on the mask ratio from $75 \%$ to $95 \%$.

| Method | Decoder |  |  |
| :---: | :---: | :---: | :---: |
|  | Self-Atten | Cross-Atten | Something-Something V2 |
| ARVideo |  | $\checkmark$ | 66.8 |
| ARVideo | $\checkmark$ | $\checkmark$ | 66.6 |

Table 9: Ablation study on the decoder architecture.

| Decoder Width | Decoder Depth | Something-Something V2 |
| :---: | :---: | :---: |
| 384 | 4 | 66.0 |
| 512 | 4 | $\mathbf{6 6 . 8}$ |
| 768 | 4 | 66.8 |
| 512 | 2 | 66.2 |
| 512 | 4 | $\mathbf{6 6 . 8}$ |
| 512 | 8 | 66.6 |

Table 10: Ablation study on the decoder depth and width.

Mask Ratio. To reduce the temporal redundancy, ARVideo randomly mask a portion of tokens as in Flip [25], MAE [18] and VideoMAE [39]. We hereby check how the masking ratio affects the overall performance. As shown in Table 8 our study starts from a mask ratio of $75 \%$ (i.e., same as the MAE's setup), which achieves $66.0 \%$ top-1 accuracy. Increasing the mask ratio to $80 \%$ boosted the top-1 accuracy to $66.8 \%$, while further increases to $90 \%$ and $95 \%$ lower the top- 1 accuracies by $1.2 \%$ and $2.0 \%$, respectively. We stress that, although ARVideo used a lower mask ratio than VideoMAE, it still enjoys faster training speeds and reduced GPU load (see Section 4.2 and Table 57.

Decoder Architecture. We hereby explore the effects of different decoder architectures. As reported in Table 9 , whether or not having self-attention in the decoder has little effect on performance (i.e., $66.6 \%$ vs. $66.8 \%$ ), but excluding self-attention significantly reduces computational costs. Therefore, we take the decoder without self-attention by default in ARVideo.

Decoder Width and Depth. Lastly, we systematically ablate two critical aspects in designing decoders: its width and depth. We start with a four-layer decoder and follow the default setup in VideoMAE. As presented in Table 10, increasing the decoder width shows performance improvement from $66.0 \%$ at a width of 384 to $66.8 \%$ at a width of 512 . Further width increase makes the performance plateau. Meanwhile, in terms of depth, deviations from the four-layer standard negatively impacted performance: e.g., increasing to eight layers decreased performance by $0.2 \%$, while reducing to two layers dropped performance by $0.6 \%$ (see the last three rows in Table 10 .

## 5 Conclusion

In this paper, we introduce ARVideo for self-supervised video representation learning, inspired by the autoregressive principles of GPT in natural language processing. Diverging from conventional methods, our approach innovatively uses video token clusters as the element for autoregressive prediction, significantly reducing computational demands while still managing to capture essential spatial-temporal dynamics. This advancement improves the efficiency of video data processing and sets a new paradigm for self-supervised video representation learning. The promising results obtained from ARVideo underscore its potential and advocate for further exploration and development of autoregressive pretraining methods within the video domain.

## References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In ICCV, 2021.

[2] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEit: BERT pre-training of image transformers. In ICLR, 2022.

[3] Sagie Benaim, Ariel Ephrat, Oran Lang, Inbar Mosseri, William T Freeman, Michael Rubinstein, Michal Irani, and Tali Dekel. Speednet: Learning the speediness in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9922-9931, 2020.

[4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, 2021.

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

[6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020.

[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In $N A A C L, 2019$.

[8] Ali Diba, Vivek Sharma, Reza Safdari, Dariush Lotfi, Saquib Sarfraz, Rainer Stiefelhagen, and Luc Van Gool. Vi2clr: Video and image for visual contrastive learning of representation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1502-1512, 2021.

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[10] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. ICML, 2024.

[11] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In ICCV, 2021.

[12] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.

[13] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems, 35:35946-35958, 2022.

[14] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fründ, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The "something something" video database for learning and evaluating visual common sense. In ICCV, 2017.

[15] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In CVPR, 2018.

[16] Tengda Han, Weidi Xie, and Andrew Zisserman. Memory-augmented dense predictive coding for video representation learning. In European conference on computer vision, pages 312-329. Springer, 2020.

[17] Tengda Han, Weidi Xie, and Andrew Zisserman. Self-supervised co-training for video representation learning. Advances in Neural Information Processing Systems, 33:5679-5690, 2020.

[18] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages $16000-16009,2022$.

[19] Tianyu Hua, Yonglong Tian, Sucheng Ren, Michalis Raptis, Hang Zhao, and Leonid Sigal. Self-supervision through random segments with autoregressive coding (randsac). In The Eleventh International Conference on Learning Representations, 2022.

[20] Deng Huang, Wenhao Wu, Weiwen Hu, Xu Liu, Dongliang He, Zhihua Wu, Xiangmiao Wu, Mingkui Tan, and Errui Ding. Ascnet: Self-supervised video representation learning with appearance-speed consistency. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8096-8105, 2021.

[21] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

[22] Haofei Kuang, Yi Zhu, Zhi Zhang, Xinyu Li, Joseph Tighe, Sören Schwertfeger, Cyrill Stachniss, and $\mathrm{Mu}$ Li. Video contrastive learning with global context. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3195-3204, 2021.

[23] Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In ICCV, 2011.

[24] Rui Li, Yiheng Zhang, Zhaofan Qiu, Ting Yao, Dong Liu, and Tao Mei. Motion-focused contrastive learning of video representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2105-2114, 2021.

[25] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390-23400, 2023.

[26] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202-3211, 2022.

[27] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR, 2022.

[28] Zhaoyang Liu, Donghao Luo, Yabiao Wang, Limin Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Tong Lu. TEINet: Towards an efficient architecture for video recognition. In AAAI, 2020.

[29] Zhaoyang Liu, Limin Wang, Wayne Wu, Chen Qian, and Tong Lu. Tam: Temporal adaptive module for video recognition. In $I C C V, 2021$.

[30] OpenAI. Gpt-4 technical report, 2023.

[31] Mandela Patrick, Dylan Campbell, Yuki Asano, Ishan Misra, Florian Metze, Christoph Feichtenhofer, Andrea Vedaldi, and Joao F. Henriques. Keeping your eye on the ball: Trajectory attention in video transformers. In NeurIPS, 2021.

[32] Yu Qi, Fan Yang, Yousong Zhu, Yufei Liu, Liwei Wu, Rui Zhao, and Wei Li. Exploring stochastic autoregressive image modeling for visual representation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2074-2081, 2023.

[33] Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge J. Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. In CVPR, 2021.

[34] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018 .

[35] Kanchana Ranasinghe, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan, and Michael S Ryoo. Self-supervised video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2874-2884, 2022.

[36] Sucheng Ren, Zeyu Wang, Hongru Zhu, Junfei Xiao, Alan Yuille, and Cihang Xie. Rejuvenating i-gpt for scalable visual representation learning. In ICML, 2024.

[37] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. NeurIPS, 2014.

[38] Hao Tan, Jie Lei, Thomas Wolf, and Mohit Bansal. Vimpac: Video pre-training via masked token prediction and contrastive learning. arXiv preprint arXiv:2106.11250, 2021.

[39] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078-10093, 2022.

[40] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In CVPR, 2018.

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[42] Limin Wang, Zhan Tong, Bin Ji, and Gangshan Wu. TDN: Temporal difference networks for efficient action recognition. In CVPR, 2021.

[43] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks for action recognition in videos. IEEE TPAMI, 2019.

[44] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In CVPR, 2022.

[45] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14733-14743, 2022.

[46] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

[47] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, 2022.

[48] Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In CVPR, 2019.

[49] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. XInet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32, 2019.

</end of paper 1>


<paper 2>
# DATACOMP: In search of the next generation of multimodal datasets 

Samir Yitzhak Gadre ${ }^{* 2}$, Gabriel Ilharco ${ }^{* 1}$, Alex Fang* ${ }^{* 1}$, Jonathan Hayase ${ }^{1}$,<br>Georgios Smyrnis ${ }^{5}$, Thao Nguyen ${ }^{1}$, Ryan Marten ${ }^{7,9}$, Mitchell Wortsman ${ }^{1}$,<br>Dhruba Ghosh ${ }^{1}$, Jieyu Zhang ${ }^{1}$, Eyal Orgad ${ }^{3}$, Rahim Entezari ${ }^{10}$, Giannis Daras ${ }^{5}$,<br>Sarah Pratt ${ }^{1}$, Vivek Ramanujan ${ }^{1}$, Yonatan Bitton ${ }^{11}$, Kalyani Marathe ${ }^{1}$,<br>Stephen Mussmann ${ }^{1}$, Richard Vencu ${ }^{6}$, Mehdi Cherti ${ }^{6,8}$, Ranjay Krishna ${ }^{1}$,<br>Pang Wei Koh ${ }^{1,12}$, Olga Saukh ${ }^{10}$, Alexander Ratner ${ }^{1,13}$, Shuran Song ${ }^{2}$,<br>Hannaneh Hajishirzi ${ }^{1,7}$, Ali Farhadi ${ }^{1}$, Romain Beaumont ${ }^{6}$,<br>Sewoong Oh ${ }^{1}$, Alex Dimakis ${ }^{5}$, Jenia Jitsev ${ }^{6,8}$,<br>Yair Carmon ${ }^{3}$, Vaishaal Shankar ${ }^{4}$, Ludwig Schmidt ${ }^{1,6,7}$


#### Abstract

Multimodal datasets are a critical component in recent breakthroughs such as CLIP, Stable Diffusion and GPT-4, yet their design does not receive the same research attention as model architectures or training algorithms. To address this shortcoming in the machine learning ecosystem, we introduce DataComp, a testbed for dataset experiments centered around a new candidate pool of 12.8 billion image-text pairs from Common Crawl. Participants in our benchmark design new filtering techniques or curate new data sources and then evaluate their new dataset by running our standardized CLIP training code and testing the resulting model on 38 downstream test sets. Our benchmark consists of multiple compute scales spanning four orders of magnitude, which enables the study of scaling trends and makes the benchmark accessible to researchers with varying resources. Our baseline experiments show that the DaTAComP workflow leads to better training sets. Our best baseline, DATACOMP-1B, enables training a CLIP ViT-L/14 from scratch to $79.2 \%$ zero-shot accuracy on ImageNet, outperforming OpenAI's CLIP ViT-L/14 by 3.7 percentage points while using the same training procedure and compute. We release DATACOMP and all accompanying code at www. datacomp.ai.


## 1 Introduction

Recent advances in multimodal learning such as CLIP [111], DALL-E [115, 116], Stable Diffusion [123], Flamingo [8], and GPT-4 [103] offer unprecedented generalization capabilities in zero-shot classification, image generation, and in-context learning. While these advances use different algorithmic techniques, e.g., contrastive learning, diffusion, or auto-regressive modeling, they all rest on a common foundation: large datasets containing paired image-text examples. For instance, CLIP's training set contains 400 million image-text pairs, and Stable Diffusion was trained on the two billion examples from LAION-2B [129]. This new generation of image-text datasets is 1,000 times larger than previous datasets such as ImageNet, which contains 1.2M images [37, 126].

Despite the central role of image-text datasets, little is known about them. Many state-of-the-art datasets are proprietary, and even for public datasets such as LAION-2B [129], it is unclear how design choices such as the data source or filtering techniques affect the resulting models. While there are thousands of ablation studies for algorithmic design choices (loss function, model architecture, etc.), datasets are often treated as monolithic artifacts without detailed investigation. Moreover,[^0]

Table 1: Zero-shot performance of CLIP models trained on different datasets. DATAComp-1B, assembled with a simple filtering procedure on image-text pairs from Common Crawl, leads to a model with higher accuracy than previous results while using the same number of multiply-accumulate operations (MACs) or less during training. See Section 3.5 for details on the evaluation datasets.

| Dataset | Dataset size | \# samples <br> seen | Architecture | Train compute <br> $(\mathrm{MACs})$ | ImageNet <br> accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OpenAI's WIT [111] | $0.4 \mathrm{~B}$ | $13 \mathrm{~B}$ | ViT-L/14 | $1.1 \times 10^{21}$ | 75.5 |
| LAION-400M [128, 28] | $0.4 \mathrm{~B}$ | $13 \mathrm{~B}$ | ViT-L/14 | $1.1 \times 10^{21}$ | 72.8 |
| LAION-2B [129, 28] | 2.3B | $13 \mathrm{~B}$ | ViT-L/14 | $1.1 \times 10^{21}$ | 73.1 |
| LAION-2B [129, 28] | $2.3 \mathrm{~B}$ | 34B | ViT-H/14 | $6.5 \times 10^{21}$ | 78.0 |
| LAION-2B [129, 28] | 2.3B | 34B | ViT-g/14 | $9.9 \times 10^{21}$ | 78.5 |
| DATACoMP-1B (ours) | $1.4 \mathrm{~B}$ | $13 \mathrm{~B}$ | ViT-L/14 | $1.1 \times 10^{21}$ | $\mathbf{7 9 . 2}$ |

datasets currently lack the benchmark-driven development process that has enabled a steady stream of improvements on the model side and isolates data enhancements from changes to the model. These issues impede further progress in multimodal learning, as evidenced by recent work showing that public datasets currently do not match the scaling behavior of proprietary alternatives [28].

In this paper, we take a step towards a more rigorous dataset development process. Our first and central contribution is DATACOMP, a new benchmark for multimodal dataset design. DATACOMP flips the traditional benchmarking paradigm in machine learning where the dataset is fixed and researchers propose new training algorithms. Instead, we hold the entire training code and computational budget constant so that participants innovate by proposing new training sets. To evaluate the quality of a training set, we score the resulting model with a testbed of 38 classification and retrieval tasks such as ImageNet [37], ImageNetV2 [121], DTD [30], EuroSAT [63], SUN-397 [146], and MSCOCO [26].

DATACOMP focuses on two key challenges that arise when assembling large training datasets: what data sources to train on, and how to filter a given data source. Each challenge corresponds to one track in our benchmark. To facilitate the filtering track, our second contribution is CommonPool, a dataset of 12.8B image-text pairs collected from Common Crawl and currently the largest public image-text dataset. We release CommonPool as an index of image url-text pairs under a CC-BY-4.0 license, and apply content checks in its construction to remove unsafe or unwanted content. In the filtering track, the goal of participants is to find the best subset of CommonPool to train on. In the second track, Bring Your Own Data (BYOD), participants may leverage any data source, as long as it does not overlap with our evaluation testbed.

Our third contribution is an investigation of scaling trends for dataset design. In particular, DATACOMP contains four scales, where we vary the training budget and the candidate pool size from $12.8 \mathrm{M}$ to $12.8 \mathrm{~B}$ samples (see Table 2). Expressed in GPU hours, the cost of a single training run ranges from 4 to 40,000 GPU hours on the A100 cluster we used for development. The different scales enable researchers with different resources to participate in our benchmark. Moreover, our results show that the ranking of filtering approaches is largely consistent across scale.

Our fourth contribution is over three hundred baseline experiments, including techniques such as querying captions for relevant keywords, filtering based on image embeddings, and applying a threshold on CLIP scores. A key result from our baselines experiments is that smaller, more stringently filtered datasets can lead to models that generalize better than larger datasets coming from the same pool. At the 12.8B scale, our best filtering baseline increases ImageNet zero-shot accuracy by 6.9 percentage points (pp) relative to the unfiltered pool (see Table 3). For the BYOD track, our initial experiments show that $109 \mathrm{M}$ additional data points (less than $1 \%$ of the $12.8 \mathrm{~B}$ pool) improve the CLIP-filtered subsets of CommONPool by up to 1.2 pp ImageNet accuracy (see Table 18).

Finally, our fifth contribution is DATACoMP-1B, a new state-of-the-art multimodal dataset. We obtain DATACOMP-1B by combining our two most promising filtering baselines. DATACOMP-1B enables training a CLIP ViT-L/14 model to an ImageNet zero-shot accuracy of $79.2 \%$ (see Table 1), corresponding to a $9 \times$ computational cost reduction when compared to a larger CLIP ViT-g/14 model trained on LAION-2B for about $3 \times$ longer. Moreover, our model outperforms OpenAI's original CLIP ViT-L/14 by 3.7 percentage points, while using the same compute budget.

To make DATAComP a shared environment for controlled dataset experiments, we publicly release our candidate pool url index, our tooling for assembling these pools, our filtering baselines, and our

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-03.jpg?height=382&width=1279&top_left_y=259&top_left_x=431)

Figure 1: DATAComp participant workflow. A) Choose a scale based on resource constraints. B) Design a dataset, in either the filtering or BYOD track. C) Train a CLIP model on the designed dataset using a fixed architecture and hyperparameters (Section 3.4). D) Evaluate the trained model on a suite of diverse downstream tasks (Section 3.5).

code for training and evaluating models at www. datacomp. ai. We believe that our infrastructure will help put research on dataset design on rigorous empirical foundations, draw attention to this understudied research area, and lead to the next generation of multimodal datasets.

## 2 Related Work

We review the most closely related work and include additional related work in Appendix C.

The effects of data curation. Classical work considers dataset cleaning and outlier removal $[74,152$, $124,125]$ to discard samples that may lead to undesirable model bias. A related line of work develops coreset selection algorithms $[61,7,46,11,94,145,32]$, which aim to select data subsets that lead to the same performance as training on the entire dataset. These techniques appear to scale poorly to larger data regimes $[51,6]$. More recent efforts in subset selection often operate on already curated datasets [98, 141, 130, 16, 33, 106] (e.g., CIFAR-10, ImageNet) or on smaller data regimes (e.g., YFCC-15M $[111,140])$. These settings often do not reflect newer training paradigms that involve (1) noisy image-text pairs instead of category labeled images and (2) large scale datasets (e.g., billions of samples). While data-centric investigations have led to community competitions like DCBENCH [43] and DATAPERF [97], existing benchmarks have likewise operated at small data scales [100] compared to datasets like LAION-2B [129], which contains over two billion images. DATACOMP bridges this gap by aligning data-centric investigation with large scale image-text training.

There has also been renewed interest in dataset pruning and deduplication. Sorscher et al. [135] show that data pruning can improve traditional scaling trends on ImageNet, but do not consider image-text training or larger datasets. Raffel et al. [113] remove sentence redundancies when creating the $\mathrm{C} 4$ corpus. Subsequent work further demonstrated the benefits of deduplication for better language modeling [90]. Radenovic et al. [110] introduce CAT filtering for image-text datasets-a rule-based system to retain high quality samples. Abbas et al. [6] propose SemDeDup, which starts with the CAT-filtered LAION-440M subset, further employing clustering to remove semantic duplicates. DATACOMP facilitates data-centric investigation at an even larger scale (i.e., 12.8B sample scale) and provides a common experimental setting for fair comparison amongst dataset creation algorithms.

Large-scale multimodal datasets. Datasets have been instrumental to building multimodal models like CLIP [111], Flamingo [8], Stable Diffusion [123], DALL-E [115, 116] and GPT-4 [103]. These methods succeeded by training on large, heterogeneous datasets rather than solely through advanced modelling techniques. For example, OpenAI's CLIP trains on 400M image-text pairs from the web, roughly $300 \times$ the size of ImageNet [37]. Prior work on scaling image-text datasets also provides promising trends with respect to zero-shot model performance [73, 107]. Additional large scale datasets like FILIP-300M [149], FLD-900M [153], and PaLI-10B [25] were constructed to train multimodal models. However, many datasets used to train such models (including the dataset for OpenAI's CLIP) are proprietary, making it hard to conduct data-centric investigations.

Even for public image-text datasets like SBU [104], Flickr30k [151], MS-COCO [26], TaiSu [92], Conceptual Captions [131], CC12M [24], RedCaps [38], WIT [136], Shutterstock [101], YFCC-

Table 2: Experimental configurations, with compute in multiply-accumulate operations (MACs).

| Scale | Model | Train compute $($ MACs) | Pool size and \# samples seen |
| :--- | :--- | :---: | :---: |
| small | ViT-B/32 | $9.5 \times 10^{16}$ | $12.8 \mathrm{M}$ |
| medium | ViT-B/32 | $9.5 \times 10^{17}$ | $128 \mathrm{M}$ |
| large | ViT-B/16 | $2.6 \times 10^{19}$ | $1.28 \mathrm{~B}$ |
| xlarge | ViT-L/14 | $1.1 \times 10^{21}$ | $12.8 \mathrm{~B}$ |

100M [140], COYO-700M [20], LAION-400M [128], or LAION-2B [129] little is known about what constitutes a good image-text dataset. Preliminary analysis suggests that different image-text data sources lead to CLIP models with different properties [101]. However, previous work is limited to smaller scale data (10-15M examples). Birhane et al. [15] examine LAION-400M and find NSFW imagery and racial slurs, centering the dangers in web-scale multimodal datasets. To combat toxicity, we preprocess our pool to remove NSFW content and blur human faces detected in images. For more details on our safety preprocessing see Section 3.2, Appendices E and G.

## 3 The DATACoMP benchmark

DATAComp is meant to facilitate data-centric experimentation. While traditional benchmarks emphasize model design, DATAComp is centered around dataset development, where the resulting datasets can be used to train high accuracy models. We focus on large image-text datasets and quantify a dataset submission by training a CLIP model on it from scratch [111] and evaluating on 38 downstream image classification and retrieval tasks. We additionally have three secret test sets, which will be released after a year, to guard against overfitting. To facilitate such investigations, we provide a candidate pool of uncurated image-text pairs sourced from the public internet. Our benchmark offers two tracks: one where participants must filter samples from the pools we provide, and another where participants can use external data. Moreover, DataComp is structured to accommodate participants with diverse levels of computational resources: each track is broken down into four scales with varying compute requirements. We now discuss high-level design decisions, construction of a 12.8B image-text data pool to facilitate the competition, benchmark tracks, model training, and evaluation.

### 3.1 Competition design

Overview. In many areas of machine learning, larger datasets lead to better performing models [87, $79,73,107,66,28,19,111,112]$. Hence comparing only datasets with the same size is a natural starting point. However, this approach is flawed as controlling the dataset size ignores critical curation constraints: candidate pool size (i.e., number of image-text pairs to harvest) and training compute. For instance, assembling a dataset like LAION-2B consists of identifying data sources (e.g., Common Crawl or Reddit) and filtering the data source. Notably, the final dataset size is a design choice and is only upper-bounded by the data sources. Hence, the true data constraint is the size of the reservoir of samples: candidate pool to be filtered. To make DATACOMP a realistic benchmark, we therefore fix the candidate pool in the filtering track, but give participants control over the training set size.

Compute cost is another relevant constraint. To put datasets of different size on equal footing, we specify the total number of training samples seen. Consider the 12.8B compute scale and filtered datasets $A$ and $B$, with 6.4B and 3.2B image-text pairs respectively. At this scale, we train by making two passes over $A$, while making four passes over $B$. A key result from our experiments is that smaller, more stringently filtered datasets can lead to models that generalize better.

Competition tracks. Two key procedures in assembling a training dataset are filtering a data source $[128,129,20]$ and aggregating data sources [36,37]. To reflect this structure, DATACOMP has two tracks: filtering, where participants select a subset of the samples from CommOnPoOL, and Bring Your Own Data (BYOD), where participants can use any source of data. Key decisions for each tracks are described in Sections 3.2 and 3.3, respectively. For full competition track rules see Appendix A.

Competition compute scales. To facilitate study of scaling trends and accommodate participants with various computational resources, we structure DATACOMP using four scales of compute: small, medium, large and xlarge. Each new scale increases the number of samples seen during training by
$10 \times$ (from $12.8 \mathrm{M}$ to $12.8 \mathrm{~B}$ samples seen), and the pool we provide by the same factor (from $12.8 \mathrm{M}$ samples to 12.8B samples). Table 2 gives the experimental configuration used for each scale. For the small scale, our runs took 4 hours on an A100 GPU, and for the xlarge scale 81 hours on 512 GPUs.

### 3.2 COMMONPool generation, for the filtering track

We construct a large-scale pool of image-text pairs, CommonPool, from Common Crawl [3]. CommonPool is distributed as an image url-text pair index under a CC-BY-4.0 license. Our pool construction pipeline has four steps: url extraction and data download, NSFW detection, evaluation set deduplication, and face blurring. We additionally provide per sample metadata (e.g., CLIP features). Starting from the xlarge CommonPool, we take successive random subsets to create large, medium, and small COMMONPOOL (e.g., medium is a subset of large).

Extracting urls and dowloading data. We first use cc2dataset [1], which utilizes Apache Spark [155], to extract pairs of image urls and nonempty alt-text from all Common Crawl snapshots from 2014 to 2022. We then deduplicate the url-text pairs and randomly shuffle. This step results in $\sim 88 \mathrm{~B}$ possible samples. Not all samples are downloadable; other samples are not suitable due to NSFW content or overlap with our evaluation sets. We attempt to download $\sim 40 \mathrm{~B}$ samples using img2dataset [5] resulting in $\sim 16.8$ B image-text pairs. For more details, see Appendix D.

Safety preprocessing. Since Common Crawl is a snapshot of the internet, we require strict preprocessing to remove unsafe content. We use Detoxify [60] to prune samples that contain unsafe text (e.g., obscene, sexually explicit, or threatening language). We also discard samples with explicit visual content. To do so, we train a classifier on CLIP ViT-L/14 [111] features, using the NSFW dataset used in LAION-5B [129]. We validate our classifier against the Google commercial image safety API. See Appendix E for details. Around 19\% of image-text pairs are considered NSFW, taking the pool of $\sim 16.8 \mathrm{~B}$ downloads to $\sim 13.6 \mathrm{~B}$ samples.

Evaluation set deduplication. To prevent accidental overfitting to certain test sets in our evaluation suite, we perform a thorough near-duplicate removal between the candidate pool and our evaluation sets, using a state-of-the-art image deduplication model [150]. Appendix F contains additional details. The model flags $\sim 3 \%$ of the $16.8 \mathrm{~B}$ images as near-duplicates, reducing the $\sim 13.6 \mathrm{~B}$ pool to $\sim 13.1 \mathrm{~B}$ samples. From here we select a random subset to get the xlarge pool of $12.8 \mathrm{~B}$ samples.

Face detection $\boldsymbol{\&}$ blurring. To protect the privacy of individuals, we detect and blur faces from images in our pool using a face detector [53]. As observed by Yang et al. [148], obfuscating faces has little impact on model performance, as we also observe in our experiments (Appendix G).

Pool metadata. To bootstrap participants we distribute metadata for each sample in COMMONPOOL (e.g., image url, alt-text, original image resolution, CLIP features, and CLIP similarity scores). Following Carlini et al. [22], we release SHA256 hashes for each image to guard against data poisoning in subsequent CommonPool downloads. For additional details see Appendix H. We open-source our metadata processing pipeline as dataset2metadata [4].

### 3.3 The bring your own data (BYOD) track

While CommonPool can be used to study different filtering techniques, state-of-the-art models often train on data from different sources. For instance, the Flamingo model [8] uses both multimodal massive web (M3W) and ALIGN datasets [73]. To facilitate non-proprietary research on curating data from many sources, we instantiate a separate DATACoMP track to allow participants to combine multiple data streams. For example, participants could construct a training set from CC12M [24], YFCC100M [140], and data sources they label themselves. In Section 4.2 and Appendix P. 2 we describe our exploration using existing public, image-text datasets. These datasets are acquired from their respective sources and are not re-release as part of DATAComP.

### 3.4 Training

We create a common experimental setting that enables comparable experiments by fixing the training procedure. We closely follow the CLIP training recipe proposed by Radford et al. [111]: training
models from scratch with a contrastive objective over images and captions. Given a set of imagecaption pairs, we train an image encoder and a text encoder such that the similarity between the representations of images and their corresponding text is maximized relative to unaligned pairs. ${ }^{1}$ For each scale, we fix the model architecture and hyperparameters (see Table 2). We pick Vision Transformers (ViTs) [39] as the image encoder, considering the better scaling trends observed by Radford et al. [111] compared to ResNets [62]. Models are trained for a fixed number of steps determined by the scale (Table 2), using the OpenCLIP repository [69]. See Appendix N for details.

### 3.5 Evaluation

We evaluate on a suite of 38 image classification and retrieval tasks. We also study two additional fairness tasks, detailed in Section 5 and Appendix Q. As discussed in Section 3.2, we remove test set images from DATACOMP to avoid contamination. Image classification datasets range from satellite imagery recognition to classifying metastatic tissues. In total we have (with some overlap): 22 of the datasets evaluated in Radford et al. [111], 6 ImageNet distribution shifts (i.e., ImageNet-Sketch [143], ImageNet-V2 [121], ImageNet-A [65], ImageNet-O [65], ImageNet-R [64], and ObjectNet [13]), 13 datasets from VTAB [156], and 3 datasets from WILDS [83, 127]. Retrieval datasets include Flickr30k [151], MSCOCO [26], and the WinoGAViL commonsense association task [17]. To aggregate results over all evaluation tasks, we average the preferred metric for each task.

DATAComp adopts a zero-shot evaluation protocol: models are tested without training on the evaluation tasks. This approach is computationally efficient and measures a model's ability to perform well without any additional training. We find a strong rank correlation ( $>0.99$ ) between performance in linear probe zero-shot settings (Appendix Figure 16). Additional details are in Appendix O.

## 4 Baselines

### 4.1 Filtering baselines

We study six simple filtering methods for the filtering track; see Appendix P. 1 for further details.

No filtering. We simply use the entire pool as the subset, without any filtering. Since each pool size is equal to the sample budget, training consists of one pass over the data.

Random subsets. To isolate the effects of increasing the compute budget from increasing the dataset size, we form subsets consisting of $1 \%, 10 \%, 25 \%, 50 \%$ and $75 \%$ of the pool chosen at random.

Basic filtering. We consider many simple filtering operations inspired by Schuhmann et al. [128] and Byeon et al. [20]: filtering by language (English captions, using either fasttext [77] or cld3 [2]); filtering by caption length (over two words and five characters); and filtering by image size (smaller dimension above 200 pixels and aspect ratio below three). We also experiment with combining language and caption length filtering and combining language, caption length, image size fitering. Unless otherwise specified, "basic" refers fasttext English, caption length, and image size filtering.

CLIP score and LAION filtering. We experiment with CLIP score filtering (also employed by LAION), where we take only examples having cosine similarity scores between CLIP image and text embeddings that exceed a pre-defined threshold. We investigate a range of thresholds and two OpenAI CLIP models for computing the scores: the ViT-B/32 model (as in LAION) and the larger ViT-L/14. We also combine CLIP score thresholds and cld3 English filtering to reproduce the LAION-2B filtering scheme. Table 16 in Appendix P. 1 summarizes the different CLIP score configurations.

Text-based filtering. We select examples that contain text overlapping with ImageNet class names, which serve as a proxy for relevance to downstream tasks. Specifically, we select English captions (according to fasttext) that contain words from ImageNet-21K or ImageNet-1K [37] class synsets.[^1]

Table 3: Zero-shot performance for select baselines in the filtering track. On all scales, filtering strategies lead to better performance than using the entire, unfiltered pool. The intersection between imaged-based and CLIP score strategies performs well on most tasks and scales. For all metrics, higher is better (see Appendix $\mathrm{O}$ for details). $\cap$ denotes the intersection of filtering strategies.

| Scale | Filtering strategy | Dataset <br> size | Samples <br> seen | ImageNet | ImageNet <br> dist. shifts | VTAB | Retrieval | Average over <br> 38 datasets |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| small | No filtering | $\overline{12.8 \mathrm{M}}$ | $12.8 \mathrm{M}$ | 0.025 | 0.033 | 0.145 | $\overline{0.114}$ | 0.132 |
|  | Basic filtering | $3 \mathrm{M}$ | $12.8 \mathrm{M}$ | 0.038 | 0.043 | 0.150 | 0.118 | 0.142 |
|  | Text-based | $3.2 \mathrm{M}$ | $12.8 \mathrm{M}$ | 0.046 | 0.052 | 0.169 | $\underline{0.125}$ | 0.157 |
|  | Image-based | $3 \mathrm{M}$ | $12.8 \mathrm{M}$ | 0.043 | 0.047 | 0.178 | $\overline{0.121}$ | 0.159 |
|  | LAION-2B filtering | $1.3 \mathrm{M}$ | $12.8 \mathrm{M}$ | 0.031 | 0.040 | 0.136 | 0.092 | 0.133 |
|  | CLIP score (L/14 30\%) | $3.8 \mathrm{M}$ | $12.8 \mathrm{M}$ | $\underline{0.051}$ | $\underline{0.055}$ | $\underline{0.190}$ | 0.119 | $\underline{0.173}$ |
|  | Image-based $\cap$ CLIP score (L/14 30\%) | $1.4 \mathrm{M}$ | $12.8 \mathrm{M}$ | $\overline{0.039}$ | $\overline{0.045}$ | $\overline{0.162}$ | 0.094 | $\overline{0.144}$ |
| medium | No filtering | $128 \mathrm{M}$ | $128 \mathrm{M}$ | 0.176 | 0.152 | 0.259 | 0.219 | 0.258 |
|  | Basic filtering | $30 \mathrm{M}$ | $128 \mathrm{M}$ | 0.226 | 0.193 | 0.284 | 0.251 | 0.285 |
|  | Text-based | $31 \mathrm{M}$ | $128 \mathrm{M}$ | 0.255 | 0.215 | 0.328 | 0.249 | 0.307 |
|  | Image-based | $29 \mathrm{M}$ | $128 \mathrm{M}$ | 0.268 | 0.213 | 0.319 | $\underline{0.256}$ | 0.312 |
|  | LAION-2B filtering | $13 \mathrm{M}$ | $128 \mathrm{M}$ | 0.230 | 0.198 | 0.307 | $\overline{0.233}$ | 0.292 |
|  | CLIP score (L/14 30\%) | $38 \mathrm{M}$ | $128 \mathrm{M}$ | 0.273 | 0.230 | 0.338 | 0.251 | $\underline{0.328}$ |
|  | Image-based $\cap$ CLIP score (L/14 30\%) | $14 \mathrm{M}$ | $128 \mathrm{M}$ | 0.297 | 0.239 | 0.346 | 0.231 | $\overline{0.328}$ |
| large | No filtering | $1.28 \mathrm{~B}$ | $1.28 \mathrm{~B}$ | $\overline{0.459} \quad$ | $\overline{0.378} \quad$ | $\overline{0.426}$ | 0.419 | $\overline{0.437}$ |
|  | Basic filtering | $298 \mathrm{M}$ | $1.28 \mathrm{~B}$ | 0.516 | 0.423 | 0.446 | 0.480 | 0.458 |
|  | Text-based | $317 \mathrm{M}$ | $1.28 \mathrm{~B}$ | 0.561 | 0.465 | 0.465 | 0.352 | 0.466 |
|  | Image-based | $293 \mathrm{M}$ | 1.28B | 0.572 | 0.454 | 0.483 | 0.479 | 0.476 |
|  | LAION-2B filtering | $130 \mathrm{M}$ | $1.28 \mathrm{~B}$ | 0.553 | 0.453 | 0.510 | 0.495 | 0.501 |
|  | CLIP score (L/14 30\%) | $384 \mathrm{M}$ | $1.28 \mathrm{~B}$ | 0.578 | 0.474 | 0.538 | 0.466 | 0.529 |
|  | Image-based $\cap$ CLIP score (L/14 30\%) | $140 \mathrm{M}$ | $1.28 \mathrm{~B}$ | $\underline{0.631}$ | $\underline{0.508}$ | $\underline{0.546}$ | $\underline{0.498}$ | $\underline{0.537}$ |
| xlarge | No filtering | $12.8 \mathrm{~B}$ | $12.8 \mathrm{~B}$ | $\overline{0.723}$ | $\overline{0.612}$ | $\overline{0.611}$ | $\overline{0.569}$ | $\overline{0.621}$ |
|  | LAION-2B filtering | $1.3 \mathrm{~B}$ | $12.8 \mathrm{~B}$ | 0.755 | 0.637 | 0.624 | $\underline{0.620}$ | 0.636 |
|  | CLIP score (L/14 30\%) | 3.8B | $12.8 \mathrm{~B}$ | 0.764 | 0.655 | 0.643 | $\overline{0.588}$ | 0.650 |
|  | Image-based $\cap$ CLIP score (L/14 30\%) | $1.4 \mathrm{~B}$ | $12.8 \mathrm{~B}$ | $\underline{0.792}$ | $\underline{0.679}$ | $\underline{0.652}$ | 0.608 | $\underline{0.663}$ |

Image-based filtering. We select a subset of examples whose visual content overlaps with ImageNet classes. After applying English language (fasttext) and caption length filtering, we cluster the image embeddings extracted by the OpenAI ViT-L/14 model for each image into $100 \mathrm{~K}$ groups using Faiss [75]. We then find the nearest neighbor group for every ImageNet training example, and keep examples belonging to these groups. We apply this procedure using either ImageNet-21K (14M images) or ImageNet-1K (1.2M images), forming two subsets.

### 4.2 BYOD baselines

We experiment with multiple external data sources, including four moderately sized datasets ( 10 to 58M samples) studied by Nguyen et al. [101]-CC12M [24], YFCC15M [140, 111], RedCaps [38] and Shutterstock [101]-and the larger LAION-2B [129]. Additional experiments, along with more details about the data sources are provided in Appendix P.2. We consider these data sources as they are and do not perform additional preprocessing. We also present experiments combining some of the data sources (using only the external datasets, or in addition to data from our pool).

## 5 Results and discussion

### 5.1 Building better datasets

Main results. Our key results are in Table 3. Most notably, the intersection between image-based filtering and CLIP score filtering excels on most tasks. The exception is at the small scale and for retrieval datasets. ${ }^{2}$ Furthermore, other filtering strategies like basic, CLIP score, image-based, text-based filtering show better downstream performance when compared to no filtering. A much larger suite of experiment results can be found in Appendix R.

DATACOMP leads to better image-text datasets. We hope DATACoMP catalyzes the search for the next generation of multimodal datasets. We contribute DATACOMP-1B, which is the output of the Image-based $\cap$ CLIP score (L/14 30\%) baseline filter at the xlarge scale of the filtering track.[^2]![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-08.jpg?height=512&width=1356&top_left_y=259&top_left_x=384)

Figure 2: Performance of random subsets (dotted line) and CLIP score filtering (solid line) when varying the subset size. When taking random subsets, larger subsets are always better. For CLIP score filtering, subsets with intermediate size perform best.

Our dataset is comprised of 1.4B samples, which not only is smaller than the LAION-2B dataset with 2.3B samples, but also comes from a smaller pool. Nevertheless, a CLIP L/14 trained on DATACOMP-1B outperforms the LAION-2B competitor by 6.1 percentage points on ImageNet (see Table 1). Moreover, training on DATAComp-1B improves ImageNet accuracy by 3.7 percentage points over OpenAI's ViT-L/14 trained with the same compute budget. Additionally, even if we restrict ourselves to $400 \mathrm{M}$ samples, we can still find a subset of DATACOMP-1B that outperforms OpenAI's ViT-L/14, as seen in Table 24. These results demonstrate the impact that DataComp can make and provide a foundation upon which participants can build.

External data sources can improve performance. Appendix P. 2 Table 18 shows results for several baselines in the BYOD track. We find several instances where adding external data sources improves performance over using just data from CoMMONPoOL. For example, at the large scale, combining CLIP-filtered data from CommonPool with external data from CC12M [24], YFCC15M [140, 111], RedCaps [38] and Shutterstock [101] boosts ImageNet accuracy by 4.3 percentage points. See Appendix P. 2 for more experiments and details.

Trade-off between data diversity and repetition. In Figure 2, we see that randomly selecting subsets of the pool has little effect and degrades performance substantially when only small fractions are used. When filtering with CLIP scores, the optimal training set comes from selecting $\sim 30 \%$ of the pool with the highest scores. The difference in performance trends between random subsets and CLIP score filtering highlights the importance of filtering strategies for selecting samples.

### 5.2 DATAComP design analyses

COMMONPOOL and LAION are comparable with the same filtering. To validate our pool construction, we show that we can build datasets comparable to LAION-2B by employing their filtering technique on our pool. LAION-2B selects all samples where the caption is in English and the cosine similarity score from a trained ViT-B/32 CLIP model is above 0.28 . We compare this filtering approach on our pool using the same number samples, $130 \mathrm{M}$ samples at the large scale. We find that the different data sources perform comparably: $55.3 \%$ vs $55.7 \%$ accuracy on ImageNet, and 0.501 vs 0.489 average performance over our evaluation sets using our pool and LAION-2B, respectively.

Consistency across scales. We find that the ranking between filtering strategies is typically consistent across different scales. This is illustrated in Figure 3, which shows that the baselines at small and medium scales are positively correlated. Moreover, as shown in Appendix Table 22, the rank correlations of performance is high, between 0.71 and 0.90 for different scale pairs.

Consistency across training changes. DATACOMP fixes the training procedure, so a natural question is whether better datasets from DAtaComp are better outside of DAtaComp. While DATACOMP-1B is trained at the xlarge scale, we show in Appendix Table 23 that even when substituting the ViT-L/14
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-09.jpg?height=464&width=1044&top_left_y=256&top_left_x=384)
~Basic

* CLIP score
Image-based
No filtering
\& Rand. subset
+ Text-based

```

Figure 3: Correlation between small and medium scale baselines. Smaller scales can serve as useful guides for larger scales. Results for additional scales are shown in Appendix Figure 22.

for a ViT-B/16 or ViT-B/32, training on DATAComP-1B outperforms training on OpenAI's WIT and LAION-2B. Additionally, we found that modifying hyperparameters such as training steps and batch size minimally affects the relative ordering of different data curation methods on downstream performance. Details on hyperparameter ablations are in Appendix L.

\subsection*{5.3 Evaluation trends}

ImageNet accuracy is indicative, but not the complete picture. Similarly to Kornblith et al. [84], in Appendix Figure 25 we find that ImageNet performance is highly correlated with the average performance across all datasets we study, with an overall correlation of $0.99 .{ }^{3}$ However, ImageNet performance is not representative of all evaluation tasks, as the correlation between ImageNet accuracy and accuracy on other individual datasets varies substantially, in some cases even exhibiting a negative correlation, as discussed in Appendix R.

Robustness and fairness. While typical models trained on a target task suffer large performance drops under data distribution shift, zero-shot CLIP models are known to exhibit strong performance across many distributions [111]. In Appendix Figure 26, we show that CLIP models trained with data from our pool are more robust to distribution shift than ImageNet-trained models from Taori et al. [139]'s testbed. Examining geographic diversity, we find that our models are better than ImageNet-trained models, but fall short of models fine-tuned on diverse curated datasets (see Appendix Figure 21). We also perform a face classification analysis and identify demographic biases in our models: notably, the BYOD datasets we consider can increase the risk of misclassification. See Appendix Q for more fairness and diversity analyses.

\section*{6 Limitations and conclusion}

In terms of societal risks, creating an index of image-text pairs from the public internet can be problematic. The internet contains unsafe, toxic, and sensitive content, which ideally should not percolate into machine learning datasets. Though we take steps to remove NSFW content and blur human faces to protect privacy, we hope future work will further explore the biases and risks from CommonPool and DataComp-1B. We see several additional directions for future work, including 1) Curating more data sources. 2) Improved data filtering algorithms. 3) Further supervision signals (e.g., image captions coming from captioning models). 4) Additional input modalities (e.g., video, 3D objects). 5) Broader evaluations for vision-and-language and robotics tasks.

Overall, we see DATAComP as a first step towards improving training datasets, and hope our new benchmark will foster further research. By providing a controlled experimental setting, DATACOMP enables researchers to iterate on dataset design on rigorous empirical foundations. We open-source all of our code, data, and infrastructure, and hope these resources will help the community build the next generation of multimodal datasets.
\footnotetext{
${ }^{3}$ Note that unlike Kornblith et al. [84] we evaluate zero-shot performance rather than transfer learning.
}

\section*{Acknowledgements}

SYG and JH are supported by NSF Graduate Research Fellowships. GS is supported by the Onassis Foundation - Scholarship ID: F ZS 056-1/2022-2023. GD has been supported by the Onassis Fellowship (Scholarship ID: F ZS 012-1/2022-2023), the Bodossaki Fellowship and the Leventis Fellowship. This research has been supported by NSF Grants AF 1901292, CNS 2148141, DMS 2134012, TRIPODS II-DMS 2023166, Tripods CCF 1934932, IFML CCF 2019844 and research gifts by Western Digital, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco, the Len Blavatnik and the Blavatnik Family Foundation, the Stanly P. Finch Centennial Professorship in Engineering, Open Philanthropy, Google, Microsoft, and the Allen Institute for AI.

We would like to thank Amro Abbas, Danny Bickson, Alper Canberk, Jessie Chapman, Brian Cheung, Tim Dettmers, Joshua Gardner, Nancy Garland, Sachin Goyal, Huy Ha, Zaid Harchaoui, Ari Holtzman, Andrew Hundt, Andy Jones, Adam Klivans, Ronak Mehta, Sachit Menon, Ari Morcos, Raviteja Mullapudi, Jonathon Shlens, Brandon McKinzie, Alexander Toshev, David Grangier, Navdeep Jaitly, Kentrell Owens, Marco Tulio Ribeiro, Shiori Sagawa, Christoph Schuhmann, Matthew Wallingford, and Ross Wightman for helpful feedback at various stages of the project. We are particularly grateful to Daniel Levy and Alec Radford for early encouragement to pursue this project and feedback on the experimental design.

We thank Stability AI and the Gauss Centre for Supercomputing e.V. ${ }^{4}$ for providing us with compute resources to train models. We are thankful for the compute time provided through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster [78] at Jülich Supercomputing Centre (JSC), and for storage resources on JUST [50] granted and operated by JSC, as well as computing and storage resources from the Helmholtz Data Federation (HDF).
\footnotetext{
${ }^{4}$ https://gauss-centre.eu
}

\section*{References}

[1] cc2dataset. https://github.com/rom1504/cc2dataset.

[2] CLD3. https://github.com/google/cld3.

[3] Common Crawl. https://commoncrawl.org.

[4] dataset2metadata. https://github.com/mlfoundations/dataset2metadata.

[5] img2dataset. https://github.com/rom1504/img2dataset.

[6] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication, 2023. https://arxiv . org/abs/2303.09540.

[7] Pankaj K. Agarwal, Sariel Har-Peled, and Kasturi R. Varadarajan. Approximating extent measures of points. Journal of the ACM (JACM), 2004. https://doi.org/10.1145/ 1008731.1008736 .

[8] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id=EbMuimAbPbs.

[9] Abhijeet Awasthi, Sabyasachi Ghosh, Rasna Goyal, and Sunita Sarawagi. Learning from rules generalizing labeled exemplars. In International Conference on Learning Representations (ICLR), 2020. https://openreview.net/forum?id=SkeuexBtDr.

[10] Stephen H Bach, Daniel Rodriguez, Yintao Liu, Chong Luo, Haidong Shao, Cassandra Xia, Souvik Sen, Alex Ratner, Braden Hancock, Houman Alborzi, Rahul Kuchhal, Christopher Ré, and Rob Malkin. Snorkel drybell: A case study in deploying weak supervision at industrial scale. In Special Interest Group on Management of Data (SIGMOD), 2019. https : //arxiv.org/abs/1812.00417.

[11] Olivier Bachem, Mario Lucic, and Andreas Krause. Coresets for nonparametric estimation - the case of dp-means. In International Conference on Machine Learning (ICML), 2015. https://proceedings.mlr.press/v37/bachem15.html.

[12] Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE Transactions on Medical Imaging, 2018. https://pubmed.ncbi.nlm.nih.gov/30716025/.

[13] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 32. Curran Associates, Inc., 2019. https://proceedings. neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf.

[14] Sara Beery, Elijah Cole, and Arvi Gjoka. The iwildcam 2020 competition dataset, 2020. https://arxiv.org/abs/2004.10340.

[15] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes, 2021. https://arxiv.org/abs/2110. 01963.

[16] Vighnesh Birodkar, Hossein Mobahi, and Samy Bengio. Semantic redundancies in imageclassification datasets: The $10 \%$ you don't need. arXiv preprint arXiv:1901.11409, 2019. https://arxiv.org/abs/1901.11409.

[17] Yonatan Bitton, Nitzan Bitton Guetta, Ron Yosef, Yuval Elovici, Mohit Bansal, Gabriel Stanovsky, and Roy Schwartz. WinoGAViL: Gamified association benchmark to challenge vision-and-language models, 2022. https://arxiv.org/abs/2207.12576.

[18] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101-mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), 2014. https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29.

[19] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems (NeurIPS), 2020. https://proceedings.neurips.cc/ paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

[20] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.

[21] Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. Broken neural scaling laws. International Conference on Learning Representations (ICLR), 2023. https://arxiv.org/ $\mathrm{abs} / 2210.14891$.

[22] Nicholas Carlini, Matthew Jagielski, Christopher A Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. Poisoning web-scale training datasets is practical, 2023. https://arxiv.org/abs/2302.10149.

[23] Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/2205.05055.

[24] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/2102. 08981.

[25] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual language-image model. In International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2209. 06794.

[26] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server, 2015. https://arxiv.org/abs/1504.00325.

[27] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the Institute of Electrical and Electronics Engineers (IEEE), 2017. https://ieeexplore.ieee.org/abstract/ document/7891544.

[28] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning, 2022. https://arxiv.org/abs/2212.07143.

[29] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Conference on Computer Vision and Pattern Recognition (CVPR), 2018. https : //arxiv.org/abs/1711.07846.

[30] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Conference on Computer Vision and Pattern Recognition (CVPR), 2014. https://openaccess.thecvf.com/content_cvpr_2014/ html/Cimpoi_Describing_Textures_in_2014_CVPR_paper.html.

[31] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011. https://proceedings.mlr.press/v15/coates11a.html.

[32] Michael B. Cohen, Cameron Musco, and Christopher Musco. Input sparsity time low-rank approximation via ridge leverage score sampling. In ACM-SIAM Symposium on Discrete Algorithms, 2017. https://dl.acm.org/doi/10.5555/3039686.3039801.

[33] C Coleman, C Yeh, S Mussmann, B Mirzasoleiman, P Bailis, P Liang, J Leskovec, and M Zaharia. Selection via proxy: Efficient data selection for deep learning. In International Conference on Learning Representations (ICLR), 2020. https://arxiv.org/abs/1906. 11829 .

[34] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Annual Meeting of the Association for Computational Linguistics (ACL), 2019. https://arxiv.org/abs/1911. 02116.

[35] R Dennis Cook. Detection of influential observation in linear. Technometrics, 19(1):15-18, 1977.

[36] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In European Conference on Computer Vision (ECCV), 2020. https://arxiv.org/abs/2005.10356.

[37] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition (CVPR), 2009. https://ieeexplore.ieee.org/abstract/document/5206848.

[38] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated imagetext data created by the people, for the people, 2021. https://arxiv.org/abs/2111. 11431.

[39] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. https://openreview.net/forum?id=YicbFdNTTy.

[40] Matthijs Douze, Giorgos Tolias, Ed Pizzi, Zoë Papakipos, Lowik Chanussot, Filip Radenovic, Tomas Jenicek, Maxim Maximov, Laura Leal-Taixé, Ismail Elezi, Ondrej Chum, and Cristian Canton-Ferrer. The 2021 image similarity dataset and challenge, 2021. https://arxiv. org/abs/2106.09672.

[41] Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with v-usable information. In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2110.08420.

[42] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results, 2007. http : //www . pascal-network.org/challenges/VOC/voc2007/workshop/index.html.

[43] Sabri Eyuboglu, Bojan Karlaš, Christopher Ré, Ce Zhang, and James Zou. dcbench: a benchmark for data-centric ai systems. In Proceedings of the Sixth Workshop on Data Management for End-To-End Machine Learning, 2022. https://dl.acm.org/doi/abs/ $10.1145 / 3533028.3533310$.

[44] Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2205.01397.

[45] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories. Conference on Computer Vision and Pattern Recognition (CVPR) Workshop, 2004. https://ieeexplore. ieee.org/document/1384978.

[46] Dan Feldman, Matthew Faulkner, and Andreas Krause. Scalable training of mixture models via coresets. In Advances in Neural Information Processing Systems (NeuIPS), 2011. https://proceedings.neurips.cc/paper_files/paper/2011/ file/2b6d65b9a9445c4271ab9076ead5605a-Paper.pdf.

[47] Daniel Y. Fu, Mayee F. Chen, Frederic Sala, Sarah M. Hooper, Kayvon Fatahalian, and Christopher Ré. Fast and three-rious: Speeding up weak supervision with triplet methods. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/ 2002.11955.

[48] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012. https://ieeexplore.ieee.org/abstract/document/6248074.

[49] Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International Conference on Machine Learning, pp. 2242-2251. PMLR, 2019.

[50] Stephan Graf and Olaf Mextorf. Just: Large-scale multi-tier storage infrastructure at the jülich supercomputing centre. Journal of large-scale research facilities JLSRF, 2021. https: //jlsrf.org/index.php/lsf/article/view/180.

[51] Chengcheng Guo, Bo Zhao, and Yanbing Bai. Deepcore: A comprehensive library for coreset selection in deep learning, 2022. https://arxiv.org/abs/2204.08499.

[52] Han Guo, Nazneen Fatema Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. Fastif: Scalable influence functions for efficient model interpretation and debugging, 2020. https : //arxiv.org/abs/2012.15781.

[53] Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou. Sample and computation redistribution for efficient face detection. In International Conference on Learning Representations (ICLR), 2021. https://arxiv.org/abs/2105.04714.

[54] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[55] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2018. https://aclanthology.org/N18-2017.

[56] Kelvin Guu, Albert Webson, Ellie Pavlick, Lucas Dixon, Ian Tenney, and Tolga Bolukbasi. Simfluence: Modeling the influence of individual training examples by simulating training runs, 2023. https://arxiv.org/abs/2303.08114.

[57] Frank R Hampel. The influence curve and its role in robust estimation. Journal of the american statistical association, 1974. https://www.jstor.org/stable/2285666.

[58] Xiaochuang Han, Byron C Wallace, and Yulia Tsvetkov. Explaining black box predictions and unveiling data artifacts through influence functions, 2020. https://arxiv.org/abs/2005. 06676.

[59] A. Hanna, Emily L. Denton, Andrew Smart, and Jamila Smith-Loud. Towards a critical race methodology in algorithmic fairness. In Conference on Fairness, Accountability, and Transparency (FAccT), 2020. https://arxiv.org/abs/1912.03593.

[60] Laura Hanu and Unitary team. Detoxify, 2020. https://github.com/unitaryai/ detoxify.

[61] Sariel Har-Peled and Soham Mazumdar. On coresets for k-means and k-median clustering. In Symposium on Theory of Computing (STOC), 2004. https://doi.org/10.1145/1007352. 1007400 .

[62] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://arxiv.org/abs/1512.03385.

[63] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. https://arxiv . org/abs/1709.00029.

[64] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. ICCV, 2021. https://arxiv.org/abs/2006.16241.

[65] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/1907.07174.

[66] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models, 2022. https://arxiv.org/abs/2203. 15556 .

[67] Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, and Daniel S Weld. Knowledge-based weak supervision for information extraction of overlapping relations. In Annual Meeting of the Association for Computational Linguistics (ACL), 2011. https: //aclanthology.org/P11-1055.

[68] Andrew Hundt, William Agnew, Vicky Zeng, Severin Kacianka, and Matthew Gombolay. Robots enact malignant stereotypes. In Conference on Fairness, Accountability, and Transparency (FAccT), 2022. https://arxiv.org/abs/2207.11569.

[69] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. OpenCLIP, July 2021. https://doi.org/10.5281/ zenodo. 5143773 .

[70] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://arXiv.org/abs/2208.05592.

[71] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data, 2022. https://arxiv.org/abs/ 2202.00622

[72] Tanuj Jain, Christopher Lennan, Zubin John, and Dat Tran. Imagededup, 2019. https: //github.com/idealo/imagededup.

[73] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2102.05918.

[74] Mon-Fong Jiang, Shian-Shyong Tseng, and Chih-Ming Su. Two-phase clustering process for outliers detection. Pattern recognition letters, 2001. https://www. sciencedirect.com/ science/article/abs/pii/S0167865500001318.

[75] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 2019. https://arxiv.org/abs/1702.08734.

[76] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. Conference on Computer Vision and Pattern Recognition (CVPR), 2017. https://arxiv.org/abs/1612.06890.

[77] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. In Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2017. https://arxiv.org/abs/1607.01759.

[78] Juelich Supercomputing Center. JUWELS Booster Supercomputer, 2020. https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html\# hardware-configuration-of-the-system-name-booster-module.

[79] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. https://arxiv.org/abs/2001.08361.

[80] Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2021. https://arxiv.org/abs/1908.04913.

[81] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning (ICML), 2017. https://arxiv.org/ $\mathrm{abs} / 1703.04730$.

[82] Pang Wei Koh, Kai-Siang Ang, Hubert Teo, and Percy S Liang. On the accuracy of influence functions for measuring group effects. Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.13289.

[83] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2012.07421.

[84] Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better imagenet models transfer better? In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. https: //arxiv.org/abs/1805.08974.

[85] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In International Conference on Computer Vision Workshops (ICML), 2013. https://www.cv-foundation.org/openaccess/content_iccv_workshops_ 2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html.

[86] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images, 2009. https://www.cs.toronto.edu/〜kriz/learning-features-2009-TR.pdf.

[87] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2012. https://proceedings.neurips.cc/paper_files/paper/ 2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.

[88] Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish Sabharwal, and Yejin Choi. Adversarial filters of dataset biases. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/2002.04108.

[89] Yann LeCun. The MNIST database of handwritten digits, 1998. http://yann.lecun.com/ exdb/mnist/.

[90] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In Annual Meeting of the Association for Computational Linguistics (ACL), 2021. https://arxiv.org/abs/2107.06499.

[91] Yi Li and Nuno Vasconcelos. Repair: Removing representation bias by dataset resampling. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. https://arxiv . org/abs/1904.07911.

[92] Yulong Liu, Guibo Zhu, Bin Zhu, Qi Song, Guojing Ge, Haoran Chen, GuanHui Qiao, Ru Peng, Lingxiang Wu, and Jinqiao Wang. Taisu: A 166m large-scale high-quality dataset for chinese vision-language pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://proceedings.neurips.cc/paper_files/paper/2022/ file/6a386d703b50f1cf1f61ab02a15967bb-Paper-Datasets_and_Benchmarks. $\mathrm{pdf}$.

[93] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. Conference on Computer Vision and Pattern Recognition (CVPR), 2022. https://arxiv.org/abs/2201.03545.

[94] Mario Lucic, Matthew Faulkner, Andreas Krause, and Dan Feldman. Training gaussian mixture models at scale via coresets. Journal of Machine Learning Research (JMLR), 2018. http://jmlr.org/papers/v18/15-506.html.

[95] S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft, 2013. https://arxiv.org/abs/1306.5151.

[96] Gideon S Mann and Andrew McCallum. Generalized expectation criteria for semi-supervised learning with weakly labeled data. Journal of Machine Learning Research (JMLR), 2010. https://www.jmlr.org/papers/v11/mann10a.html.

[97] Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Juan Ciro, Lora Aroyo, Bilge Acun, Sabri Eyuboglu, Amirata Ghorbani, Emmett Goodman, Tariq Kane, Christine R. Kirkpatrick, Tzu-Sheng Kuo, Jonas Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen Paritosh, Ce Zhang, James Zou, Carole-Jean Wu, Cody Coleman, Andrew Ng, Peter Mattson, and Vijay Janapa Reddi. Dataperf: Benchmarks for data-centric ai development, 2022. https://arxiv.org/abs/2207.10062.

[98] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/1906.01827.

[99] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In Advances in Neural Information Processing Systems (NeurIPS) Workshops, 2011. https://storage. googleapis.com/pub-tools-public-publication-data/pdf/37648.pdf.

[100] Andrew Ng, Dillon Laird, and Lynn He. Data-centric ai competition, 2021. https:// https-deeplearning-ai.github.io/data-centric-comp/.

[101] Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/ forum?id=LTCBavFWp5C.

[102] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, 2008. https://ieeexplore.ieee.org/document/4756141.

[103] OpenAI. Gpt-4 technical report, 2023. https://arxiv.org/abs/2303.08774.

[104] Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg. Im2text: Describing images using 1 million captioned photographs. In Advances in Neural Information Processing Systems (NeurIPS), 2011. https://papers.nips.cc/paper_files/paper/2011/file/ 5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf.

[105] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012. https: //ieeexplore.ieee.org/document/6248092.

[106] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Advances in Neural Information Processing Systems (NeurIPS), 2021. https://arxiv.org/abs/2107.07075.

[107] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu, Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and Quoc V. Le. Combined scaling for zero-shot transfer learning, 2021. https://arxiv.org/abs/2111.10050.

[108] Vinay Uday Prabhu and Abeba Birhane. Large image datasets: A pyrrhic win for computer vision? In Winter Conference on Applications of Computer Vision (WACV), 2020. https: //arxiv.org/abs/2006.16923.

[109] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems (NeurIPS), 2020. https://arxiv.org/abs/2002.08484.

[110] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distillation, and hard negatives for vision-language pre-training. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. https://arxiv.org/abs/2301.02280.

[111] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/ 2103.00020 .

[112] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. https:// arxiv.org/abs/2212.04356.

[113] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research (JMLR), 2020. https: //arxiv.org/abs/1910.10683.

[114] Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron B. Adcock, Laurens van der Maaten, Deepti Ghadiyaram, and Olga Russakovsky. Beyond web-scraping: Crowd-sourcing a geodiverse datase, 2023. https://arxiv.org/abs/2301.02560.

[115] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2102.12092.

[116] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. https://arxiv.org/abs/2204. 06125

[117] A. J. Ratner, B. Hancock, J. Dunnmon, F. Sala, S. Pandey, and C. Ré. Training complex models with multi-task weak supervision. In Association for the Advancement of Artificial Intelligence (AAAI), 2019. https://arxiv.org/abs/1810.02840.

[118] Alexander J Ratner, Christopher M De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. Data programming: Creating large training sets, quickly. In Advances in Neural Information Processing Systems (NeurIPS), 2016. https://arxiv.org/abs/1605.07723.

[119] Alexander J Ratner, Stephen H Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. Snorkel: Rapid training data creation with weak supervision. In Very Large Data Bases Conference (VLDB), 2017. https://arxiv.org/abs/1711.10160.

[120] Christopher Ré. Overton: A data system for monitoring and improving machine-learned products. In 10th Conference on Innovative Data Systems Research, CIDR 2020, Amsterdam, The Netherlands, January 12-15, 2020, Online Proceedings. www.cidrdb.org, 2020. URL http://cidrdb.org/cidr2020/papers/p33-re-cidr20.pdf.

[121] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In International Conference on Machine Learning (ICML), 2019. http://proceedings.mlr.press/v97/recht19a.html.

[122] William A Gaviria Rojas, Sudnya Diamos, Keertan Ranjan Kini, David Kanter, Vijay Janapa Reddi, and Cody Coleman. The dollar street dataset: Images representing the geographic and socioeconomic diversity of the world. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2022. https://openreview .net/forum?id= qnfYsave0U4.

[123] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. https://arxiv.org/abs/2112.10752.

[124] Peter J Rousseeuw and Mia Hubert. Robust statistics for outlier detection. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 2011. http://i2pc. es/coss/Docencia/SignalProcessingReviews/Rousseeuw2011.pdf.

[125] Peter J Rousseeuw and Mia Hubert. Anomaly detection by robust statistics. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 2018. https://wires. onlinelibrary.wiley.com/doi/pdf/10.1002/widm. 1236.

[126] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li FeiFei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 2015. https://arxiv.org/abs/1409.0575.

[127] Shiori Sagawa, Pang Wei Koh, Tony Lee, Irena Gao, Sang Michael Xie, Kendrick Shen, Ananya Kumar, Weihua Hu, Michihiro Yasunaga, Henrik Marklund, Sara Beery, Etienne David, Ian Stavness, Wei Guo, Jure Leskovec, Kate Saenko, Tatsunori Hashimoto, Sergey Levine, Chelsea Finn, and Percy Liang. Extending the wilds benchmark for unsupervised
adaptation. In International Conference on Learning Representations (ICLR), 2022. https : //arxiv.org/abs/2112.05090.

[128] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of clip-filtered 400 million image-text pairs, 2021. https://arxiv.org/abs/ 2111.02114 .

[129] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2022. https://openreview.net/ forum?id=M3Y74vmsMcY.

[130] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018. https://openreview.net/forum?id=H1aIuk-RW.

[131] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Annual Meeting of the Association for Computational Linguistics (ACL), 2018. https://aclanthology . org/P18-1238/.

[132] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks?, 2021. https://arxiv.org/abs/2107.06383.

[133] Changho Shin, Winfred Li, Harit Vishwakarma, Nicholas Roberts, and Frederic Sala. Universalizing weak supervision. In International Conference on Learning Representations (ICLR), 2022. https://openreview.net/forum?id=YpPiNigTzMT.

[134] Haoyu Song, Li Dong, Weinan Zhang, Ting Liu, and Furu Wei. CLIP models are fewshot learners: Empirical studies on VQA and visual entailment. In Annual Meeting of the Association for Computational Linguistics (ACL), 2022. https://aclanthology.org/ 2022.acl-long. 421 .

[135] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id= UmvSlP-PyV.

[136] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021. https://arxiv.org/abs/2103.01913.

[137] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: a multi-class classification competition. In International Joint Conference on Neural Networks (IJCNN), 2011. https://ieeexplore.ieee.org/ document/6033395.

[138] Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. https://aclanthology.org/2020.emnlp-main. 746 .

[139] Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. In Advances in Neural Information Processing Systems (NeurIPS), 2020. https://dl.acm. org/doi/abs/10.5555/3495724.3497285.

[140] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100M: The new data in multimedia research. Communications of the ACM, 2016. https://arxiv.org/abs/1503.01817.

[141] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations (ICLR), 2018. https://arxiv.org/abs/1812.05159.

[142] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant CNNs for digital pathology, 2018. https://arxiv.org/abs/1806.03962.

[143] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.13549.

[144] Ryan Webster, Julien Rabin, Loic Simon, and Frederic Jurie. On the de-duplication of laion-2b, 2023. https://arxiv.org/abs/2303.12733.

[145] Kai Wei, Rishabh Iyer, and Jeff Bilmes. Submodularity in data subset selection and active learning. In International Conference on Machine Learning (ICML), 2015. https: //proceedings.mlr.press/v37/wei15.html.

[146] Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. Sun database: Exploring a large collection of scene categories. International Journal of Computer Vision (IJCV), 2016. https://link.springer.com/article/10.1007/ s11263-014-0748-y.

[147] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: filtering and balancing the distribution of the people subtree in the imagenet hierarchy. In Conference on Fairness, Accountability, and Transparency (FAccT), 2020. https:// arxiv.org/abs/1912.07726.

[148] Kaiyu Yang, Jacqueline H Yau, Li Fei-Fei, Jia Deng, and Olga Russakovsky. A study of face obfuscation in ImageNet. In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2103.06191.

[149] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In International Conference on Learning Representations (ICLR), 2022. https: //arxiv.org/abs/2111.07783.

[150] Shuhei Yokoo. Contrastive learning with large memory bank and negative embedding subtraction for accurate copy detection, 2021. https://arxiv.org/abs/2112.04323.

[151] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2014. https:// aclanthology.org/Q14-1006/.

[152] Dantong Yu, Gholamhosein Sheikholeslami, and Aidong Zhang. Findout: Finding outliers in very large datasets. Knowledge and information Systems, 2002. https://link.springer . com/article/10.1007/s101150200013.

[153] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision, 2021. https://arxiv.org/abs/2111.11432.

[154] Man-Ching Yuen, Irwin King, and Kwong-Sak Leung. A survey of crowdsourcing systems. In SocialCom. IEEE, 2011. https://ieeexplore. ieee.org/document/6113213.

[155] Matei Zaharia, Reynold S Xin, Patrick Wendell, Tathagata Das, Michael Armbrust, Ankur Dave, Xiangrui Meng, Josh Rosen, Shivaram Venkataraman, Michael J Franklin, et al. Apache spark: a unified engine for big data processing. Communications of the ACM, 2016. https : //dl.acm.org/doi/10.1145/2934664.

[156] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, André Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. The visual task adaptation benchmark, 2019. http: //arxiv.org/abs/1910.04867.

[157] Jieyu Zhang, Yue Yu, Yinghao Li, Yujing Wang, Yaming Yang, Mao Yang, and Alexander Ratner. WRENCH: A comprehensive benchmark for weak supervision. In NeurIPS, 2021. URL https://openreview.net/forum?id=Q9SKS5k8io.

[158] Jieyu Zhang, Cheng-Yu Hsieh, Yue Yu, Chao Zhang, and Alexander Ratner. A survey on programmatic weak supervision, 2022. https://arxiv.org/abs/2202.05433.

[159] Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/regression by conditional adversarial autoencoder. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. https://arxiv.org/abs/1702.08423.

[160] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision (ECCV), 2022. https://arxiv.org/abs/2201.02605.

\section*{Appendix}

\section*{Contents}
1 Introduction ..... 1
2 Related Work ..... 3
3 The DATACOMP benchmark ..... 4
3.1 Competition design ..... 4
3.2 CommonPool generation, for the filtering track ..... 5
3.3 The bring your own data (BYOD) track ..... 5
3.4 Training ..... 5
3.5 Evaluation ..... 6
4 Baselines ..... 6
4.1 Filtering baselines ..... 6
4.2 BYOD baselines ..... 7
5 Results and discussion ..... 7
5.1 Building better datasets ..... 7
5.2 DATAComP design analyses ..... 8
5.3 Evaluation trends ..... 9
6 Limitations and conclusion ..... 9
A Benchmark rules ..... 24
A. 1 Filtering track rules ..... 24
A. 2 Bring your own data track: amendments ..... 24
B Contributions ..... 25
B. 1 Candidate pool ..... 25
B. 2 Participant tooling ..... 25
B. 3 Baselines ..... 25
B. 4 Leadership and Advising ..... 25
C Additional related work ..... 26
D Parsing Common Crawl ..... 26
E Not safe for work (NSFW) filtering ..... 27
F Deduplication against evaluation sets ..... 27
G Face blurring ..... 29
H DataComp CommonPooL creation pipeline ..... 31
I COMMONPOOL statistics ..... 32
J Efficient training on data subsets ..... 35
K Effect of duplicates in the training data ..... 35
L Hyperparameter ablations ..... 36
L. 1 Batch size ..... 36
L. 2 Model architecture ..... 36
L. 3 Number of training steps ..... 36
M Detector-based baselines ..... 39
N Training details ..... 40
O Evaluation details ..... 40
O. 1 Visual Question Answering ..... 42
P Baseline details ..... 43
P. 1 Filtering track ..... 47
P. 2 BYOD track ..... 48
P.2.1 Additional results ..... 49
Q Fairness and biases ..... 49
Q. 1 Diversity ..... 49
Q. 2 Fairness ..... 50
R Extra figures and tables ..... 53
S Datasheet ..... 60
S. 1 Motivation ..... 60
S. 2 Composition ..... 60
S. 3 Collection Process ..... 62
S. 4 Preprocessing, Cleaning, and/or Labeling ..... 63
S. 5 Uses ..... 64
S. 6 Distribution ..... 65
S. 7 Maintenance ..... 65

\section*{A Benchmark rules}

We provide concrete rules below for the two competition tracks that comprise DataComP: filtering and BYOD. Additionally, we provide a checklist, which encourages participants to specify design decisions, which allows for more granular comparison between submissions.

\section*{A. 1 Filtering track rules}
- Participants can enter submissions for one or many different scales: small, medium, large or xlarge, which represent the raw number of image-text pairs in CommonPool that should be filtered.
- After choosing a scale, participants generate a list of uids, where each uid refers to a CommonPool sample. The list of uids is used to recover image-text pairs from the pool, which is used for downstream CLIP training.
- Duplicate uids are allowed.
- Participants are not allowed to modify the training procedure. Hence, changing hyperparameters, model architecture, optimizer, compute budget, or number of training steps is not allowed. Changing any other training details is also not allowed.
- Participants are strongly encouraged to submit and open-source both the list of uids and the code used to generate this list; however, this is not required.
- To avoid overfitting, we do not permit running any code or algorithmic dependence on the test images of the evaluation tasks. However, use of other images associated with these tasks (e.g., supervised training sets) is permitted.
- Participants can use templates or class labels from the downstream tasks in their filtering algorithms.

For clarity, we include some examples of permitted and forbidden uses:

$\checkmark$ We permit using the ImageNet class label "triceratops" in a filtering algorithm.

$\times$ We forbid examining individual or aggregate predictions on the test sets of the evaluation tasks.

\section*{A. 2 Bring your own data track: amendments}

To facilitate more open-ended exploration, we provide amendments to the Track 1 competition to allow for more diverse submissions in Track 2.
- Participants are allowed to augment CommonPool data with existing datasets, so long as these data sources do not contain test images from the evaluation tasks. Participants can use data from any CommonPool; however, they are not required to do so.
- Assembling one's own dataset is allowed; however, test images from the evaluation tasks can neither be contained nor otherwise used to construct said dataset. We encourage releasing the image urls or the images themselves in addition to the text for each image. We also encourage rigorous documentation of face-blurring and other data safety checks (see Section 3.2 for more details). We reserve the right to run our own safety code on participant provided data and disqualify entries that do not meet adequate safety standards.

Checklist. The following checklist provides the basis for more fine-grained comparison between submissions.

Images from the evaluation tasks are included in my submission. If yes, please specify which datasets.

I used an existing datasets (e.g., YFCC100M [140]) in my submission. If yes, please specify which datasets. (Note: applies to BYOD only)

I curated my own data. If yes, please provide (1) image data or urls, (2) text for each image, (3) list of safety steps taken including but not limited to face blurring, explicit content image and text filtering. (Note: applies to BYOD only)

\section*{B Contributions}

For this section, contributors are ordered alphabetically.

\section*{B. 1 Candidate pool}

Candidate pool lead. Vaishaal Shankar

Data collection. Romain Beaumont, Vaishaal Shankar

Pre-processing and metadata. Giannis Daras, Alex Fang (content filtering lead), Samir Yitzhak Gadre (metadata lead), Ryan Marten (deduplication lead), Vivek Ramanujan, Vaishaal Shankar, George Smyrnis (face blurring lead)

\section*{B. 2 Participant tooling}

Participant tooling lead. Gabriel Ilharco

Resharder. Romain Beaumont, Yair Carmon, Alex Fang, Jonathan Hayase (lead), Gabriel Ilharco, Vivek Ramanujan, Vaishaal Shankar, Georgios Smyrnis

Training. Mehdi Cherti, Gabriel Ilharco, Jenia Jitsev, Vivek Ramanujan, Georgios Smyrnis, Mitchell Wortsman (lead)

Evaluation. Romain Beaumont, Yonatan Bitton, Mehdi Cherti, Dhruba Ghosh (lead), Gabriel Ilharco

Additional infrastructure. Stephen Mussmann, Sarah Pratt

\section*{B. 3 Baselines}

Baselines lead. Yair Carmon

Filtering track. Yair Carmon, Rahim Enterazi, Alex Fang, Samir Yitzhak Gadre, Gabriel Ilharco, Kalyani Marathe, Thao Nguyen, Eyal Orgad (co-lead), Georgios Smyrnis, Mitchell Wortsman, Jieyu Zhang (co-lead)

BYOD track. Gabriel Ilharco, Thao Nguyen

Experiment babysitting. Alex Fang, Gabriel Ilharco, Samir Yitzhak Gadre

\section*{B. 4 Leadership and Advising}

Advising. Romain Beaumont, Yair Carmon, Alexandros G. Dimakis, Ali Farhadi, Hannaneh Hajishirzi, Jenia Jitsev, Pang Wei Koh, Ranjay Krishna, Stephen Mussmann, Sewoong Oh, Alexander Ratner, Olga Saukh, Ludwig Schmidt, Vaishaal Shankar, Shuran Song, Richard Vencu

Leadership. Yair Carmon, Alexandros G. Dimakis, Jenia Jitsev, Sewoong Oh, Ludwig Schmidt, Vaishaal Shankar

Overall project lead. Ludwig Schmidt

\section*{C Additional related work}

Here we expand on the related work described in Section 2.

Image dataset safety is an active area of research, especially in the context of large-scale dataset construction. In addition to Birhane et al. [15], who study problematic content in LAION-400M, Yang et al. [147] study the ImageNet dataset and reveal limitations associated with the ImageNet curation strategy-with negative implications for downstream model fairness. Prabhu \& Birhane [108] also study the ImageNet dataset and find pornographic content. Both Birhane et al. [15] and Prabhu \& Birhane [108] survey ethical conundrums and harms that are borne out of improper dataset curation. In an effort to combat dataset toxicity, we conduct NSFW preprocessing (Section 3.2, Appendix E) and blur detected faces (Section 3.2, Appendix G) during pool construction. We also conduct preliminary fairness evaluations (Section 5.3, Appendix Q) for models trained on our data. We hope CommonPool will serve as a research artifact for future work examining dataset safety.

Beyond data selection, Chan et al. [23] investigate the effects of dataset distribution on emergent properties of transformers, while Fang et al. [44] look at the relationship between data and model robustness to distribution shifts. We hope our extensive evaluation suite comprised of 38 diverse tasks will facilitate similar studies when training multimodal models at large scale.

Others study how to reduce the burdens of training data annotation in the curation process. Classic approaches include distant supervision [67], crowd-sourced labels [154], heuristic rules [9] and feature annotation [96], among others. A recent line of work known as data programming or programmatic weak supervision $[118,119,157,158]$ attempts to reduce annotation cost and is found in many industry applications [10, 120]. In data programming, developers write programmatic labeling functions to automatically label a large amount of unlabeled data. The labeling functions could produce noisy and conflicting labels, so researchers have developed methods to aggregate noisy votes to produce the final training labels $[117,47,133]$.

Previous literature also studies methods for training data attribution, which seek to link a model's behavior (e.g., its accuracy on a particular task or subset of data) to particular subsets of its training data. Such methods include influence functions, a classic technique from robust statistics [57, 35] that uses a second-order Taylor expansion to approximate the effect of removing a training point on the learned model parameters [81, 82, 58,52], as well as methods that fit attribution functions directly to the dynamics of repeated training runs $[49,109,71,56]$. Training data attribution methods assume that we have already trained a model, though they can be subsequently used to refine the training data (e.g., by identifying potentially mislabeled training points [81]). Our focus in this paper is instead on data curation methods-that is, methods for selecting a subset of the training data to train a model in the first place.

In the context of natural language processing, Swayamdipta et al. [138] proposes a tool for characterizing samples in a dataset based on training dynamics, labelling instances as ambiguous, easy to learn or hard to learn. Previous literature such as work by Le Bras et al. [88], Li \& Vasconcelos [91], Gururangan et al. [55] advocate for removing easy instances from the training data. Ethayarajh et al. [41] propose a measure of how difficult a dataset is to learn, $\mathcal{V}$-usable information. Such techniques could be promising directions of further exploration in the context of our benchmark.

Finally, another related line of work is studying scaling trends. In addition to Sorscher et al. [135], researchers have investigated how model performance changes as a function of compute budget, model size, and number of training samples [79, 66, 21, 28]. However, this line of work does not consider how dataset design may affects scaling trends. Beyond dataset size, we measure the effects of different dataset sources and filtering strategies. While scaling trends are central to our investigations, the purpose of our benchmark is to search for the next generation of large multimodal datasets to facilitate more accurate and reliable models.

\section*{D Parsing Common Crawl}

Common Crawl releases metadata files for the websites that they index (i.e., WAT files). They release these files approximately once a month. We consider all files available from 2014 through November of 2022. We first parse these files, utilizing Apache Spark [155] to extract image urls and corresponding alt-text. We map each url, text pair to a uid hash and remove duplicates. This

Table 4: Detoxify positive rates by threshold on 1 million caption subset of Common Crawl.

\begin{tabular}{cccccccc}
\hline Threshold & Toxicity & Severe Toxicity & Obscene & Identity Attack & Insult & Threat & Sexual Explicit \\
\hline 0.01 & $9.5 \%$ & $1.0 \%$ & $33.4 \%$ & $1.8 \%$ & $35.0 \%$ & $1.3 \%$ & $2.0 \%$ \\
0.1 & $3.6 \%$ & $0.1 \%$ & $0.8 \%$ & $0.3 \%$ & $1.4 \%$ & $0.1 \%$ & $1.0 \%$ \\
\hline
\end{tabular}

Table 5: Comparing LAION-2B CLIP based NSFW filtering model to Google Vision API Safe Search adult category on a 40,000 random subset of Common Crawl.

\begin{tabular}{ccccc}
\hline Threshold & \begin{tabular}{c} 
False Positive Rate \\
(Relative to Google)
\end{tabular} & \begin{tabular}{c} 
True Positives \\
(Manual Review)
\end{tabular} & Model Positive Rate & Google API Positive Rate \\
\hline 0.1 & $3.6 \%$ & 2 & $14.4 \%$ & $3.5 \%$ \\
0.2 & $0.6 \%$ & 2 & $9.1 \%$ & $3.5 \%$ \\
0.3 & $0.3 \%$ & 3 & $7.2 \%$ & $3.5 \%$ \\
\hline
\end{tabular}

results in 88 billion url, text pairs, which are randomized via a distributed shuffle. Note, we do not consider image content when running uid deduplication at this step. Hence, two identical images with different urls and the same caption would both be retained.

\section*{E Not safe for work (NSFW) filtering}

Our data is sourced from Common Crawl, which contains snapshots of the web. Therefore, we apply multiple layers of NSFW content filtering to remove problematic images and captions from COMMONPOOL.

First, we filter our captions with Detoxify [60], a language model for toxic comment classification. Specifically, we use the multilingual XLM-RoBERTa [34] variant. The model outputs scores between zero and one for the following categories: toxicity, severe toxicity, obscene, identity attack, insult, threat, and sexually explicit. As we had no ground truth for our data, we manually spot check a 1 million random subset of COMmonPool at varying thresholds. We found that a threshold of 0.1 provided good coverage of filtering out NSFW text. If any of the detoxify category scores exceeds the threshold, the sample is discarded. Qualitatively, we found that the model struggled with multilingual content, acronyms, and innuendo. Even at 0.1 , we noticed there are some captions that are NSFW. However, lowering the threshold further heavily affected false positives. We therefore use a 0.1 threshold for all NSFW categories, which on a random subset of one million captions achieves positive rates shown in Table 4.

Second, on the vision side, we use a modified version of LAION-5B's [129] CLIP-based binary classification NSFW model, which takes CLIP ViT-L/14 visual embeddings as input. We remove the initial multi-category encoder from the model, and retrain on the same data with an initial normalization layer followed by a 4-layer multilayer perceptron. Our retrained model matches the performance of the original model on their manually annotated testset. Specifically, we achieve $97.4 \%$ classification accuracy on a held out test set compared to $96.1 \%$ for the original LAION NSFW image filtering model. Additional details about the training data can be found in Appendix C. 5 of the LAION-5B paper. In brief, the training data contains $682 \mathrm{~K}$ images that is roughly balanced with images from safe for work and NSFW categories.

To evaluate our model and determine a threshold, we used Google Vision API's SafeSearch explicit content detector to generate labels for an 40,000 random subset of our candidate pool. Specifically, an image is NSFW if SafeSearch classifies it as likely or very likely adult (i.e., sexually explicit). As shown in Table 5, we found that by thresholding at 0.1 we achieve high recall relative to SafeSearch and very few true positives after manual review. We also manually reviewed images classified by SafeSearch as likely or very likely racy and found that the images were either benign, subjectively suggestive but not explicit, or already found in the set of images labeled as adult.

\section*{F Deduplication against evaluation sets}

To prevent data leakage, we filter CommonPool by removing duplicate and near-duplicate matches of evaluation set images. See Figure 4 for example query images from Common Crawl and corresponding near-duplicates in our evaluations sets. We consider images as duplicates when

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-28.jpg?height=528&width=1374&top_left_y=240&top_left_x=381)

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-28.jpg?height=236&width=393&top_left_y=256&top_left_x=389)
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-28.jpg?height=472&width=700&top_left_y=271&top_left_x=388)

sun397_test/s0011797 cars_test/s0005478

Figure 4: Candidate images (top) that are detected as duplicates against images in the evaluation sets (bottom) are removed from the pool. In addition to exact duplicate images, near-duplicates with variable aspect ratios, JPEG compression, overlays, color adjustment, and artistic rendering are also detected.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-28.jpg?height=496&width=1346&top_left_y=1018&top_left_x=386)

Figure 5: Analysis of different de-duplication strategies across a variety of image transformations. We see that the model introduced by Yokoo [150] is better in almost every transformation, with the exception of very aggressive aspect ratio modification.

the cosine similarity between a query (Common Crawl image) feature and a reference (evaluation image) feature is higher than a fixed threshold. We employ the deduplication model proposed by Yokoo [150], which earned 1st place in the Facebook AI Image Similarity Challenge (ISC) [40]. We choose a cosine similarity threshold of 0.604169 to maximize the true duplicates detected, without removing too many false duplicates from the pool. We compare against OpenAI's CLIP ViT-B/32 as a baseline on ISC. We find that for our threshold, the ISC model achieves precision 0.9 and recall 0.8 . At a threshold of 0.96 , CLIP achieves the same precision 0.9 , but a significantly worse recall of 0.02 . Approximately $2.8 \%$ of downloaded samples are flagged as evaluation set near-duplicates.

To verify the performance of our de-duplication models with greater granularity, we modify the evaluation procedure in Douze et al. [40] to include transformations which are representative of naturally-occurring duplications on the Internet. Specifically, we study: 1) jpeg compression (encoding), 2) image flips, 3) image rotations, 4) aspect ratio modifications, and 5) grayscaling. To do this, we sample $20 \%$ of the images from each of our evaluation datasets uniformly at random to serve as a reference set of about 140,000 images. Next we sample 560,000 images uniformly at random from LAION-2B to serve as distractors, for a 4-to-1 distractor to reference ratio. Finally, we apply each of the augmentations above and use threshold filtering to determine duplicates. Figure 5 shows the results from the deduplication model [150] compared with OpenAI's CLIP ViT-L/14. At high recall values, we see that CLIP filtering results in removing over $2 \times$ the data as that of the deduplication model from Yokoo [150].

Table 6: Face detection performance on a set of 3293 random images from CoMmONPooL.

\begin{tabular}{lrr}
\hline & SCRFD-10G & Amazon Rekognition \\
\hline Accuracy & 93.87 & 96.57 \\
Precision & 75.87 & 86.09 \\
Recall & 90.53 & 93.75 \\
\hline
\end{tabular}

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-29.jpg?height=491&width=827&top_left_y=511&top_left_x=646)

Figure 6: Frequency of predicted number of faces in the small CoMmonPooL.

\section*{G Face blurring}

As an extra step to safeguard against issues of privacy that may arise from the use of data scraped from the web, we include face blurring as part of our pool creation. To create face metadata, we use the SCRFD face detector [53] to extract bounding boxes for the faces in our images. These bounding boxes are included as part of the image metadata in our pool. We make use of the pretrained SCRFD-10G model. We use the same preprocessing as the one described in the official repository of the paper, with the exception of providing $224 \times 224$ input images (by padding each image to square and then resizing) to limit computation costs. Invoking this model provides us with bounding boxes along with an associated score, which we then compare against a threshold of 0.3 to keep or discard this bounding box. This threshold is the default one used in the repository of SCRFD for the visualization of bounding boxes, and we found it to perform well on our data as discussed next.

In Table 6 we can see the result of face detection on a set of 3293 images from CoMmONPooL. We evaluate the detection on whether the image has visible faces or not (where images such as cartoon drawings of non-real human faces are not considered as positives), and whether the detector has detected these visible faces. We considered an image as a true positive if all the clearly visible faces in the image were detected, based on the above thresholding process. We did not do extensive box labeling. True positives are instead determined by human inspection. We compare the quality of these detections with the Amazon Rekognition system, which is the one upon which the face detections on ImageNet were based [148]. Note that in this scenario, the recall of the detectors is more important than precision (as detecting a few more bounding boxes across our pool does not affect privacy).

To utilize these bounding boxes on our data, we apply a standard blurring pipeline, as proposed by Yang et al. [148]. The result of this process is an image where the faces is blurred and there is a smooth transition from blurred to clean parts of the image. In Figure 6 we see the distribution of faces for the small CommonPool. Note that the majority of images do not contain faces.

As part of our competition pipeline, images are by default blurred during the download process. In Table 7 we can see the results of training on a set of images with the size of our medium scale after filtering with each method, with and without the application of face blurring as provided by our detector. We can see that the difference in performance is small, which suggests that the application of face blurring does not significantly affect the performance on our downstream tasks. However, we note that this design decision may be more detrimental in generative settings, especially when a generative model needs to output faces. Our competition is primarily focused on discriminative tasks, and as such when designing our dataset, we wished to prioritize the safety and privacy of individuals through blurring faces in our download tooling by default.

Table 7: Effect of face blurring on zero-shot performance. Face blurring improves the privacy preservation of our dataset, while affecting model performance negligibly. Results shown for training on a set of images with the size of our medium scale, after filtering with each method.

\begin{tabular}{lccc}
\hline Filtering & Face blurring & ImageNet acc. & Avg. performance \\
\hline \multirow{2}{*}{ CLIP score (B/32, thresh. 0.3) + English filtering } & $\times$ & 0.209 & 0.246 \\
& $\checkmark$ & 0.196 & 0.243 \\
\hline \multirow{2}{*}{ CLIP score (B/32, 30\%) } & $\times$ & 0.287 & 0.301 \\
& $\checkmark$ & 0.282 & 0.298 \\
\hline
\end{tabular}

Finally, we evaluated the detector we used for potential biases. More specifically, we used the detector on the validation set of the FairFace dataset [80]. We found that the central face of the image was detected in all the images of the validation set, regardless of subgroup annotate in the dataset.

\section*{H DATACOMP COMMONPOOL creation pipeline}

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-31.jpg?height=477&width=1261&top_left_y=363&top_left_x=432)

Figure 7: Data funnel from potential samples in Common Crawl to 13.1B image-text pairs that were suitable for COMMONPoOL. We sampled uniformly 12.8B datapoints for the xlarge CommONPooL.

Table 8: Provided metadata for CommonPooL.

\begin{tabular}{llr}
\hline Generation Time & Label & Additional notes \\
\hline & uid & Link to the image. \\
url & Image caption. \\
& \begin{tabular}{l} 
text \\
original_width \\
original_height \\
sha256
\end{tabular} & Safeguard for data poisoning. \\
\hline & \begin{tabular}{l} 
clip_b32_similarity_score \\
clip_b32_image_features
\end{tabular} & \\
& \begin{tabular}{l} 
clip_b32_text_features \\
clip_114_similarity_score
\end{tabular} & In separate file. \\
Step 1 & \begin{tabular}{l} 
clip_114_image_features \\
clip_114_text_features
\end{tabular} & In separate file. \\
\hline face_bboxes
\end{tabular}

Creating CommonPool was a multistep process, which involved (1) parsing image urls and alt-text from Common Crawl dumps and downloading these images, (2) tagging images with metadata and (3) conducting safety content filtering and evaluation set duplication. In this section we provide an overview of the data pipeline used to create CommonPool. For an overview of our "data funnel" see Figure 7.

1. For the first step, we use parse Common Crawl metadata files to harvest image-text pairs (Section D). We use img2dataset [5] to obtain $\sim 16.8 \mathrm{~B}$ downloaded samples. This is the first, unfiltered version of COMMONPooL, and contains only basic information for our images (i.e., the original image height, width, and alt-text caption). During this step we also resize images such that their largest dimension does not exceed 512 pixels. This eases storage requirements for large images, but is still larger than the 224 pixel resolution used for later training stages.

2. For the second step, we process our unfiltered pool and create richer metadata for each image-text pair. We generate the following for each sample:
- CLIP ViT-B/32 and CLIP ViT-L/14 image and text features, with their associated similarities.
- NSFW scores for the image and the text, using the analysis described in Appendix E.
- Deduplication score for the image, as described in Appendix F.
- Bounding boxes for faces detected in the image, using the method described in Appendix G.

3. For the third and final step, we filter our image-text pairs based on the metadata generated during the second stage. We filter out image-text pairs where the NSFW and deduplication scores exceed the respective thresholds (Section E). From the images that pass through this filtering, we keep only the desired amount (e.g., 12.8B images from the xlarge CommonPool). Smaller pools are telescoping subsets of larger pools. We package the metadata and image urls, which is made publicly available to the participants. Note, we do not release raw image data but rather image urls pointing to images.

A summary of the metadata for each sample is found in Table 8. To validate our pipeline for duplication and CLIP feature correctness, we also take ImageNet train though metadata generation as a unit test. Using the deduplication features, we detect that $100 \%$ of the images are in fact duplicates. Additionally using the CLIP ViT-B/32 and CLIP ViT-L/14 image features and corresponding text features from OpenAI's 80-prompt ensemble, we achieve $63.36 \%$ and $75.54 \%$ top-1 accuracies, which match the performance reported in the CLIP paper [111].

When creating pools of different scale (i.e., number of samples), we ensure that smaller pools are subsets of larger pools. For instance, the small CommonPool is a subset of the xlarge COMMONPoOL.

After CommonPool is created, the participants can then download the final image-text pairs using the provided files via img2dataset. To further ease the computational burden on participants, we additionally provide metadata for each sample in CommONPool. Note that when downloading, our img2dataset configuration automatically blurs faces. Hence this is an automatic step on not something participants must do ad hoc.

\section*{I COMMONPoOL statistics}

To provide more information about the kinds of samples in our CommonPooL, we conduct additional analysis on the small pool, which is an i.i.d. sample of downloaded data and a subset of the larger pools.

In Figure 8 we show CLIP similarity similarity scores between images and their corresponding text. We notice a flatter distribution of CLIP ViT-L/14 scores than corresponding B/32 scores.

Turning our attention to images in CommonPool, in Figure 9, we visualize the aspect ratios and sizes of original images (i.e., before they are downloaded and resized). In Figure 10, we display a distribution of image height and width after download resizing. Notice that the majority of images are around $224 \times 224$ pixels, which is the final resized resolution used for training.

Analysing the textual component of each sample, we visualize frequency of the number of CLIP BPE tokens in the captions (Figure 11) and most common languages (Figure 12). Token counts follow a long-tailed distribution with much more mass in the short sequence range, while English is the predominant language in COMMONPOOL according to fasttext and cld3.

We also look at url statistics. In Figure 13 we see common domain names in CommonPool (e.g., wordpress domains) and common suffixes (e.g., .com or .net).
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-33.jpg?height=432&width=1400&top_left_y=336&top_left_x=362)

Figure 8: Image-text similarity score distributions using CLIP ViT-B/32 (left) and ViT-L/14 (right) models. We plot samples from the small CommonPool, which are an i.i.d. sample of the xlarge COMMONPOOL.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-33.jpg?height=432&width=1378&top_left_y=1090&top_left_x=362)

Figure 9: Statistics for images in the small CommonPool, before applying resizing.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-33.jpg?height=439&width=526&top_left_y=1770&top_left_x=797)

Figure 10: Image pixel heatmap. Each entry in the above heatmap represents the estimated probability that a pixel is occupied. The center entry has a value of 1.0 as every image has a center pixel. We compute the heatmap over the small CommonPool. Note that image sizes are bounded as we resize all images such that their max dimension does not exceed 512 pixels during dataset download.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-34.jpg?height=499&width=938&top_left_y=290&top_left_x=583)

Figure 11: Distribution of token length for alt-text in the small CommonPool. The CLIP BPE tokenizer is used for tokenization.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-34.jpg?height=634&width=1262&top_left_y=987&top_left_x=428)

Figure 12: Counts for the top 25 most frequent languages in the small CommonPool, as predicted by fasttext (left) and cld3 (right).
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-34.jpg?height=582&width=1266&top_left_y=1790&top_left_x=428)

Figure 13: Counts for the top 25 most frequent domains (left) and suffixes (right) in the small COMMONPOOL.

\section*{J Efficient training on data subsets}

When training at large scale, it is important to use efficient access patterns to load training data. This typically means that data must be loaded using large sequential reads instead of random reads in order to maximize throughput. In DATACOMP, this is facilitated by the WebDataset ${ }^{5}$ format which stores the training examples in tar files (called "shards") and WebDataLoader which makes it easy to load data stored in this format.

Given an arbitrary subset of a pool, we would like to efficiently train on that subset. Because WebDataset format does not permit efficient random access (a feature inherited from tar), we must read through the entire pool to select the required images. There are two ways to implement this filtering:

1. Filter during training: we apply a predicate during training data loading that discards data not present in the subset.

2. Filter before training: we iterate over the pool, selecting the images in the subset, and write them to a new WebDataset.

After some profiling, we concluded that option 1 had too much overhead in the case where the subset is much smaller than the pool. To see why, note that if the subset is an $p$-fraction of the pool size, then we would end up reading a $1 / p$ factor more data than needed for training. Instead, we give an implementation of option 2 , which performs at most twice as many reads as needed for training. ${ }^{6}$

Our tool, called the resharder, reads a set of uids in NumPy array format, scans through the pool, selecting those examples, and writes them to a new WebDataset. The resharder uses multiprocessing to make good use of hardware and can be distributed over many computers to further increase throughput. The resharder also supports streaming data to and from cloud storage such as Amazon S3. The resharder is provided to participants as part of the competition tooling.

\section*{$K$ Effect of duplicates in the training data}

Given that CommonPool was constructed by scraping the web for image and text pairs, there is a likelihood that some of our images are duplicates of each other, even if they originated from different web sources and have different captions. Here we examine the effect of removing such duplicates. We used the technique proposed by Webster et al. [144], where CLIP image features are first compressed and then used to do an approximate nearest neighbor search. After this process, two images $x$ and $y$ are considered duplicates if $\frac{\left|d_{A D C}(x, x)-d_{A D C}(x, y)\right|}{d_{A D C}(x, x)}<T_{A D C}$, where $T_{A D C}$ is some threshold and $d_{A D C}(x, x)$ is the distance of a vector with its quantized version used for approximate nearest neighbor search. For each image, we search duplicates across its 1000 nearest neighbors, and keep it if it's the one with the highest CLIP ViT-L/14 similarity score across its duplicates. Results can be seen in Table 9, both when this technique is used by itself and in conjunction with ViT-B/32 filtering. We can see that results are similar to when only using CLIP filtering.
\footnotetext{
${ }^{5}$ https://github.com/webdataset/webdataset

${ }^{6}$ Since in DATACOMP, the number of examples seen is equal to the pool size.
}

Table 9: Effect of deduplication of training set for the medium size CommonPool. The filtering performed here is CLIP B32 score top 30\% (see Table 26). Higher threshold values lead to more samples being labeled as duplicates.

\begin{tabular}{lccc}
\hline Subset & Training dataset size & ImageNet accuracy & Average performance \\
\hline$T_{A D C}=0.1$, without filtering & $99.8 \mathrm{M}$ & 0.195 & 0.275 \\
$T_{A D C}=0.2$, without filtering & $85.9 \mathrm{M}$ & 0.200 & 0.277 \\
$T_{A D C}=0.5$, without filtering & $29.6 \mathrm{M}$ & 0.227 & 0.295 \\
$T_{A D C}=0.1$, with filtering & $33.5 \mathrm{M}$ & 0.288 & 0.337 \\
$T_{A D C}=0.2$, with filtering & $30.6 \mathrm{M}$ & 0.289 & 0.337 \\
$T_{A D C}=0.5$, with filtering & $15.5 \mathrm{M}$ & 0.252 & 0.311 \\
\hline
\end{tabular}

Table 10: Batch size ablation at the medium scale. We compare the standard DATACoMP medium configuration, with batch size 4096 against an ablated configuration with batch size 8192 (medium: batch size $2 \mathrm{x}$ ). We find that the rankings of the baseline filtering strategies are relatively consistent. More precisely, the rank correlation is 0.96 on ImageNet and 0.98 for the Average over 38 datasets.

\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline & Scale & Filtering strategy & \begin{tabular}{c} 
Dataset \\
size
\end{tabular} & \begin{tabular}{l} 
Samples \\
seen
\end{tabular} & ImageNet & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} & \begin{tabular}{l} 
Delta ranking \\
ImageNet
\end{tabular} & \begin{tabular}{l} 
Delta ranking \\
Average
\end{tabular} \\
\hline \multirow{7}{*}{\multicolumn{2}{|c|}{ medium }} & No filtering & $128 \mathrm{M}$ & $128 \mathrm{M}$ & 0.176 & 0.258 & - & \begin{tabular}{lll}
- & -1 & -1
\end{tabular} \\
\hline & & Basic filtering & $30 \mathrm{M}$ & $128 \mathrm{M}$ & 0.226 & 0.285 & - & - \\
\hline & & Text-based & $31 \mathrm{M}$ & $128 \mathrm{M}$ & 0.255 & 0.307 & - & - \\
\hline & & Image-based & $29 \mathrm{M}$ & $128 \mathrm{M}$ & 0.268 & 0.312 & - & - \\
\hline & & LAION-2B filtering & $13 \mathrm{M}$ & $128 \mathrm{M}$ & 0.230 & 0.292 & - & - \\
\hline & & CLIP score (L/14 30\%) & $38 \mathrm{M}$ & $128 \mathrm{M}$ & 0.273 & $\underline{0.328}$ & - & - \\
\hline & & Image-based $\cap$ CLIP score (L/14 30\%) & $14 \mathrm{M}$ & $128 \mathrm{M}$ & $\underline{0.297}$ & $\underline{0.328}$ & - & - \\
\hline \multirow{7}{*}{ medium: } & \multirow{7}{*}{ batch size $2 x$} & No filtering & $128 \mathrm{M}$ & $128 \mathrm{M}$ & $\overline{0.171}$ & $\overline{0.258}$ & 0 & 0 \\
\hline & & Basic filtering & $30 \mathrm{M}$ & $128 \mathrm{M}$ & 0.219 & 0.277 & +1 (worse) & 0 \\
\hline & & Text-based & $31 \mathrm{M}$ & $128 \mathrm{M}$ & 0.251 & 0.299 & 0 & -1 (better) \\
\hline & & Image-based & $29 \mathrm{M}$ & $128 \mathrm{M}$ & 0.260 & 0.299 & 0 & 0 \\
\hline & & LAION-2B filtering & $13 \mathrm{M}$ & $128 \mathrm{M}$ & 0.215 & 0.288 & -1 (better) & 0 \\
\hline & & CLIP score (L/14 30\%) & $38 \mathrm{M}$ & $128 \mathrm{M}$ & 0.271 & 0.324 & 0 & 0 \\
\hline & & Image-based $\cap$ CLIP score (L/14 30\%) & $14 \mathrm{M}$ & $128 \mathrm{M}$ & 0.276 & $\overline{0.311}$ & 0 & +1 (worse) \\
\hline
\end{tabular}

\section*{L Hyperparameter ablations}

Recall that in DATAComp, we freeze the training procedure and hyperparameters to focus the competition on dataset curation. However, this leads to the natural question: do "better" datasets (i.e., datasets that lead to higher accuracy models on zero-shot downstream tasks) remain consistent when training is modified. Hence we ablate key experimental choices: batch size, model architecture, and number of training steps.

\section*{L. 1 Batch size}

We ablate over the batch size hyperparameter, doubling the batch size at the medium scale, but holding all other hyperparameters constant. As see in Table 10, we find that the delta rankings are largely consistent, for both ImageNet and Average performance, with rankings changing by at most plus or minus one position. More specifically, rank correlation before and after doubling batch size is 0.96 for ImageNet and 0.98 for the Average over 38 datasets metric.

\section*{L. 2 Model architecture}

We choose to use the ViT architecture [39] because of favorable CLIP scaling trends over vanilla ResNets [62] as reported by Radford et al. [111]. However, we still hope that better datasets for downstream ViT performance will lead to better datasets to train convolutional architectures. We look at the medium scale, swapping the ViT-B/32 architecture with a ConvNeXt model [93] with matched giga multiplier-accumulate operations (GMACs). Looking at Table 11, we see that ranking of different filtering methods is again relatively consistent (i.e., 1.0 rank correlation for ImageNet and 0.87 rank correlation for the average metric). We conclude that improvements in dataset filtering have potential to improve more than just CLIP ViT model performance.

\section*{L. 3 Number of training steps}

Recall that one of our major design decisions for DATACoMP is to fix the hyperparameters associated with model training, following closely hyperparameters from prior work [111]. We choose to fix hyperparameters to place emphasis on data curation and remove confounders arising from hyperparameter differences between participants. Here we ablate our hyperparameter configuration by training small baselines for $10 \times$ more steps. In Figure 14 we see positive correlation for ImageNet accuracy for the ablated and original hyperparameter configurations. We see similar correlation for average performance. See Table 12 for specific values.

Table 11: Architure ablation at the medium scale. We compare the standard DATACoMP medium configuration, with a ViT-B/32 model against an ablated configuration (medium: ConvNeXt), which uses a ConvNeXt model with the same number of multiply-accumulate operations as the ViT. We find that the rankings of the baseline filtering strategies are relatively consistent. More precisely, the rank correlation is 1.0 on ImageNet and 0.87 for the Average over 38 datasets.

\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline Scale & Filtering strategy & \begin{tabular}{c} 
Dataset \\
size
\end{tabular} & \begin{tabular}{l} 
Samples \\
seen
\end{tabular} & ImageNet & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} & \begin{tabular}{c} 
Delta ranking \\
ImageNet
\end{tabular} & \begin{tabular}{c} 
Delta ranking \\
Average
\end{tabular} \\
\hline \multirow{7}{*}{ medium } & No filtering & $128 \mathrm{M}$ & $128 \mathrm{M}$ & 0.176 & 0.254 & - & - \\
\hline & Basic filtering & $30 \mathrm{M}$ & $128 \mathrm{M}$ & 0.226 & 0.280 & - & - \\
\hline & Text-based & $31 \mathrm{M}$ & $128 \mathrm{M}$ & 0.255 & 0.301 & - & - \\
\hline & Image-based & $29 \mathrm{M}$ & $128 \mathrm{M}$ & 0.268 & 0.307 & - & - \\
\hline & LAION-2B filtering & $13 \mathrm{M}$ & $128 \mathrm{M}$ & 0.230 & 0.287 & - & - \\
\hline & CLIP score (L/14 30\%) & $38 \mathrm{M}$ & $128 \mathrm{M}$ & 0.273 & $\underline{0.323}$ & - & - \\
\hline & Image-based $\cap$ CLIP score (L/14 30\%) & $14 \mathrm{M}$ & $128 \mathrm{M}$ & $\underline{0.297}$ & $\underline{0.323}$ & - & - \\
\hline \multirow{7}{*}{ medium: } & No filtering & $128 \mathrm{M}$ & $\overline{128 \mathrm{M}}$ & 0.178 & 0.255 & $\overline{0}$ & $\overline{0}$ \\
\hline & Basic filtering & $30 \mathrm{M}$ & $128 \mathrm{M}$ & 0.232 & 0.272 & 0 & 0 \\
\hline & Text-based & $31 \mathrm{M}$ & $128 \mathrm{M}$ & 0.255 & 0.298 & 0 & 0 \\
\hline & Image-based & $29 \mathrm{M}$ & $128 \mathrm{M}$ & 0.270 & 0.298 & 0 & +1 (better) \\
\hline & LAION-2B filtering & $13 \mathrm{M}$ & $128 \mathrm{M}$ & 0.253 & 0.300 & 0 & -2 (better) \\
\hline & CLIP score (L/14 30\%) & $38 \mathrm{M}$ & $128 \mathrm{M}$ & 0.279 & 0.326 & 0 & +1 (worse) \\
\hline & Image-based $\cap$ CLIP score (L/14 30\%) & $14 \mathrm{M}$ & $128 \mathrm{M}$ & $\underline{0.323}$ & $\underline{0.331}$ & 0 & 0 \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-37.jpg?height=412&width=1226&top_left_y=1689&top_left_x=446)

Figure 14: (left) The effect of training for $10 \times$ steps for for small filtering track baselines on ImageNet. (right) Similar plot but for Avg. performance. While the ordering of some methods changes quite drastically, we, in general, see a positive correlation.

Table 12: Experiment details when extending the number of steps by 10 times the standard amount for that scale.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Scale & Filtering & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline \multirow{36}{*}{ small } & No filtering & 0.102 & 0.093 & 0.204 & 0.147 & 0.196 \\
\hline & Random subset(75\%) & 0.078 & 0.072 & 0.182 & 0.129 & 0.178 \\
\hline & Random subset(50\%) & 0.045 & 0.049 & 0.161 & 0.104 & 0.150 \\
\hline & Random subset( $25 \%)$ & 0.023 & 0.029 & 0.134 & 0.075 & 0.119 \\
\hline & Random subset(10\%) & 0.010 & 0.018 & 0.119 & 0.069 & 0.101 \\
\hline & Random subset(1\%) & 0.002 & 0.006 & 0.097 & 0.056 & 0.082 \\
\hline & Caption length & 0.085 & 0.080 & 0.198 & 0.136 & 0.184 \\
\hline & Image size & 0.066 & 0.064 & 0.153 & 0.115 & 0.158 \\
\hline & English (fasttext) & 0.068 & 0.068 & 0.172 & 0.108 & 0.159 \\
\hline & English (fasttext) and caption length & 0.066 & 0.065 & 0.182 & 0.106 & 0.163 \\
\hline & English (fasttext), caption length, and image size & 0.045 & 0.048 & 0.164 & 0.092 & 0.149 \\
\hline & CLIP B32 score top 10\% & 0.035 & 0.046 & 0.162 & 0.079 & 0.139 \\
\hline & CLIP B32 score top 20\% & 0.076 & 0.076 & 0.182 & 0.099 & 0.172 \\
\hline & CLIP B32 score top 30\% & 0.096 & 0.090 & 0.221 & 0.121 & 0.205 \\
\hline & CLIP B32 score top $40 \%$ & 0.081 & 0.077 & 0.200 & 0.124 & 0.193 \\
\hline & CLIP B32 score top $50 \%$ & 0.106 & 0.097 & 0.211 & 0.134 & 0.205 \\
\hline & CLIP B32 score top 75\% & 0.103 & 0.096 & 0.210 & 0.150 & 0.198 \\
\hline & CLIP B32 score top $90 \%$ & 0.105 & 0.096 & 0.212 & 0.152 & 0.202 \\
\hline & CLIP B32 threshold at $0.3+$ English filter & 0.029 & 0.036 & 0.152 & 0.078 & 0.134 \\
\hline & CLIP B32 threshold at $0.28+$ English filter & 0.035 & 0.041 & 0.168 & 0.086 & 0.145 \\
\hline & CLIP B32 threshold at 0.3 & 0.076 & 0.078 & 0.199 & 0.102 & 0.182 \\
\hline & CLIP L14 score top $10 \%$ & 0.026 & 0.037 & 0.130 & 0.073 & 0.123 \\
\hline & CLIP L14 score top $20 \%$ & 0.060 & 0.064 & 0.161 & 0.096 & 0.153 \\
\hline & CLIP L14 score top $30 \%$ & 0.088 & 0.087 & 0.199 & 0.115 & 0.188 \\
\hline & CLIP L14 score top $40 \%$ & 0.100 & 0.096 & 0.217 & 0.122 & 0.207 \\
\hline & CLIP L14 score top $50 \%$ & 0.104 & 0.098 & 0.212 & 0.136 & 0.203 \\
\hline & CLIP L14 score top 75\% & 0.103 & 0.095 & 0.189 & 0.146 & 0.191 \\
\hline & CLIP L14 score top $90 \%$ & 0.105 & 0.095 & 0.203 & 0.145 & 0.198 \\
\hline & Image-based clustering (ImageNet1k) & 0.053 & 0.053 & 0.162 & 0.091 & 0.146 \\
\hline & Image-based clustering (ImageNet21k) & 0.063 & 0.059 & 0.173 & 0.108 & 0.167 \\
\hline & Text-based clustering (ImageNet1k) & 0.012 & 0.018 & 0.120 & 0.062 & 0.104 \\
\hline & Text-based clustering (ImageNet21k) & 0.262 & 0.216 & 0.305 & 0.246 & 0.300 \\
\hline & Intersect IN1k image clustering and CLIP B32 score top 30\% & 0.058 & 0.059 & 0.179 & 0.098 & 0.161 \\
\hline & Intersect IN1k image clustering and CLIP L14 score top 30\% & 0.049 & 0.051 & 0.171 & 0.090 & 0.150 \\
\hline & Intersect IN21k image clustering and CLIP B32 score top 30\% & 0.071 & 0.070 & 0.192 & 0.107 & 0.175 \\
\hline & Intersect IN21k image clustering and CLIP L14 score top 30\% & 0.064 & 0.065 & 0.200 & 0.096 & 0.173 \\
\hline \multirow{10}{*}{ medium } & No filtering & 0.370 & 0.304 & 0.387 & 0.355 & 0.383 \\
\hline & English (fasttext), caption length, and image size & 0.317 & 0.269 & 0.324 & 0.271 & 0.334 \\
\hline & CLIP B32 score top $30 \%$ & 0.436 & 0.351 & 0.433 & 0.345 & 0.430 \\
\hline & CLIP B32 score top 40\% & 0.434 & 0.353 & 0.448 & 0.365 & 0.442 \\
\hline & CLIP B32 score top $50 \%$ & 0.426 & 0.352 & 0.439 & 0.377 & 0.433 \\
\hline & CLIP B32 score top 75\% & 0.398 & 0.325 & 0.396 & 0.374 & 0.411 \\
\hline & Image-based clustering (ImageNet1k) & 0.363 & 0.294 & 0.347 & 0.279 & 0.347 \\
\hline & Image-based clustering (ImageNet21k) & 0.374 & 0.303 & 0.372 & 0.318 & 0.372 \\
\hline & Intersect IN1k image clustering and CLIP B32 score top 30\% & 0.415 & 0.330 & 0.413 & 0.310 & 0.403 \\
\hline & Intersect IN1k image clustering and CLIP L14 score top $30 \%$ & 0.405 & 0.325 & 0.399 & 0.295 & 0.387 \\
\hline
\end{tabular}

\section*{M Detector-based baselines}

While controlling for factors such as class balance is common in the supervised settings, experimenting with analogous strategies in the context of multimodal datasets and CLIP training is a pertinent direction. Towards this end, we use the Detic detector [160] to annotate the medium pool (128M samples) by extracting bounding boxes and class labels for the 1203 LVIS [54] objects categories. Following the original Detic paper, we retain predictions whose confidence score exceeds 0.5. Based on these annotations, we construct the following five strategies:
- Object exists: Subset for which there exists at least one detection from the 1203 LVIS categories.
- Object centered: Subset for which there exists at least one detection from the 1203 LVIS categories with a bounding box center falling in the center grid cell of a $3 \times 3$ grid superimposed on the image.
- Balancing by class: We define 1204 buckets-1203 buckets corresponding to the LVIS classes and an additional bucket for images that do not have any detections. For each image in the medium pool, we assign the image to the bucket(s) corresponding to the detected classes. We then construct a dataset such that there are an equal number of samples from each bucket and the total number of samples specified by a particular scale (e.g., 128M samples for medium scale). Note, for rare classes there can be many repeated samples and for common classes only a subset of the total samples will be in the dataset.
- Balancing by position: We define 26 buckets-0,1, .., 24 corresponding to $5 x 5$ grid locations in an image. An image is added to bucket(s) when it contains a bounding box whose center falls in the bucket's grid cell. The 25th bucket contains images for which there are no detections. We again construct a dataset such that there are an equal number of samples from each bucket.
- Balancing by count: We define 12 buckets- $0,1, \ldots, 10$ corresponding to zero to ten detections in an image and a twelfth bucket corresponding to images with more than ten detections. We yet again construct a dataset such that there are an equal number of samples from each bucket.

We employ each of these strategies on the medium scale. Since the above strategies can be composed with any starting pool, we additionally apply each of the above Detic-based strategies to our previous best medium scale filtered pool: Image-based $\cap$ CLIP score (L/14 30\%). This yields five more datasets for 10 baselines in total.

Our results are summarized in the Table 13. In summary: 1) The Image-based $\cap$ CLIP score (L/14 $30 \%$ ) baseline still performs best. 2) Balancing data in the context of multimodal CLIP training

Table 13: Detector-baseed baselines at the medium scale. We start with No filtering and Image-based cap CLIP score (L/14 30\%) pools and apply five additional filtering and balancing strategies described in Appendix M. We find that even with these more sophisticated strategies, the No filtering and Image-based cap CLIP score (L/14 30\%) still performs best at medium scale. Properly balancing multimodal data remains an open direction for future work.

\begin{tabular}{clccc}
\hline Scale & Filtering strategy & \begin{tabular}{c} 
Samples \\
seen
\end{tabular} & \begin{tabular}{c} 
ImageNet
\end{tabular} & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline \multirow{4}{*}{ medium } & No filtering & $128 \mathrm{M}$ & 0.176 & 0.258 \\
& $\cap$ Object exists & $128 \mathrm{M}$ & 0.181 & 0.263 \\
& $\cap$ Bbjlance by class & $128 \mathrm{M}$ & 0.187 & 0.263 \\
& $\cap$ Balance by position & $128 \mathrm{M}$ & 0.038 & 0.141 \\
medium & $\cap$ Balance by object count & $128 \mathrm{M}$ & 0.040 & 0.148 \\
& Image-based $\cap$ CLIP score $(\mathrm{L} / 1430 \%)$ & $128 \mathrm{M}$ & $\underline{0.297}$ & $\underline{0.328}$ \\
& $\cap$ Object exists & $128 \mathrm{M}$ & 0.289 & 0.319 \\
& $\cap$ Balance by class & $128 \mathrm{M}$ & 0.247 & 0.286 \\
& $\cap$ Balance by position & $128 \mathrm{M}$ & 0.034 & 0.136 \\
\hline
\end{tabular}

Table 14: Experimental configuration for each scale, including the size of the pool we provide, the model architecture and hyperparameters.

\begin{tabular}{lcccccccc}
\hline Scale & Model & Train compute (MACs) & Pool size & \# samples seen & Learning rate & AdamW $\beta_{2}$ & Warmup & Batch size \\
\hline small & ViT-B/32 & $9.5 \times 10^{16}$ & $12.8 \mathrm{M}$ & $12.8 \mathrm{M}$ & $5 \mathrm{e}-4$ & 0.98 & 500 & 4096 \\
medium & ViT-B/32 & $9.5 \times 10^{17}$ & $128 \mathrm{M}$ & $128 \mathrm{M}$ & $5 \mathrm{e}-4$ & 0.98 & 500 & 4096 \\
large & ViT-B/16 & $2.6 \times 10^{19}$ & $1.28 \mathrm{~B}$ & $1.28 \mathrm{~B}$ & $5 \mathrm{e}-4$ & 0.98 & 500 & 8192 \\
xlarge & ViT-L/14 & $1.1 \times 10^{21}$ & $12.8 \mathrm{~B}$ & $12.8 \mathrm{~B}$ & $1 \mathrm{e}-3$ & 0.95 & $10 \mathrm{k}$ & 90112 \\
\hline
\end{tabular}

remains an open problem. All balancing strategies lead to divergence of the CLIP contrastive loss and result in poor model performance. We hypothesize that this is due to the long-tailed nature of the data distribution, which leads to many repeated samples in our balanced data construction. This in turn, increases the likelihood that samples are contrasted with themselves in the loss computation.

\section*{N Training details}

The full set of hyperparameters used for each scale is shown in Table 14. For choosing hyperparameters, we follow the OpenCLIP library [69], an open source reproduction of OpenAI's CLIP. For the small, medium, and large tracks, these hyperparameters are equal to those in the CLIP paper, except with reduced batch size so that training runs on reasonable hardware. For the xlarge track, batch size is increased from that in OpenAI's CLIP to accelerate training by allowing the use of many GPUs simultaneously with high utilization. For this run we also double the learning rate following prior work [28].

\section*{O Evaluation details}

Models are evaluated over a wide range of 38 tasks to measure proficiency in various domains. We include 22 of the 27 classification tasks in the test suite of Radford et al. [111], excluding the few datasets that have license restrictions, are in video format, or are no longer available in their original form. We include 6 datasets that were designed to test generalization of models trained on ImageNet. We also include a majority of the Visual Task Adaptation Benchmark, excluding 3 datasets that are ill-suited for zero-shot evaluation [156]. We include 3 datasets from the WILDS benchmark, which tests robustness to distribution shifts and spurious correlations [83, 127]. Finally, we include 2 additional datasets, Dollar Street and GeoDE, which test robustness of classification performance across income levels and geographical regions [122, 114]. Furthermore, we evaluate zero-shot image and text retrieval on the Flickr30k and MSCOCO datasets, and image association on the WinoGAViL dataset [151, 26, 17]. The complete list of evaluation tasks is given in Table 15. We show a sample from each dataset in Figure 15.

Prompt choice. Since we perform zero-shot evaluation, prompt and class name selection is important, and can have a significant impact on the results. To avoid heavy prompt engineering and overtuning to individual models, we opt to use the prompt templates used in Radford et al. [111] whenever possible. Most datasets come with pre-defined class names, but some are overwritten with more descriptive labels, again based on previous literature. For datasets with no precedent in zero-shot evaluation, we reuse prompt templates from other datasets with a similar domain and task (e.g., SVHN is evaluated with MNIST prompts and class names).

Evaluation metrics. For the majority of classification tasks, the primary evaluation metric is accuracy For certain datasets with class imbalances, we instead compute mean per-class accuracy, as done in Radford et al. [111]. On the WILDS benchmark datasets, we use the primary metric specified for each dataset on their leaderboard. Dollar Street and GeoDE test model generalization across socioeconomic and geographic diversity. Thus, for Dollar Street, we compute worst-group top-5 accuracy, with groups defined by income level, emulating Rojas et al. [122]; for GeoDE, we compute worst-group accuracy, with groups defined by region (Africa, Americas, West Asia, East Asia, Southeast Asia, and Europe), as defined in Ramaswamy et al. [114]. For the image-text retrieval tasks, Flickr and MSCOCO, we compute both image and text recall (fraction of text captions for which the correct image was selected and vice versa), and plot their arithmetic mean. On WinoGAViL, we compute the

Table 15: Evaluation tasks.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Task type & Dataset & Task & Test set size & Number of classes & Main metric & Clean \\
\hline \multirow{35}{*}{ Classification } & Caltech-101 [45] & Object recognition & 6,085 & 102 & mean per class & $\checkmark$ \\
\hline & CIFAR-10 [86] & Visual recognition & 10,000 & 10 & accuracy & $\checkmark$ \\
\hline & CIFAR-100 [86] & Visual recognition & 10,000 & 100 & accuracy & $\checkmark$ \\
\hline & CLEVR Counts $[76,156]$ & Counting & 15,000 & 8 & accuracy & \\
\hline & CLEVR Distance $[76,156]$ & Distance prediction & 15,000 & 6 & accuracy & \\
\hline & Country211 $[111,140]$ & Geolocation & 21,100 & 211 & accuracy & $\checkmark$ \\
\hline & DTD [30] & Texture classification & 1,880 & 47 & accuracy & $\checkmark$ \\
\hline & EuroSAT $[63,156]$ & Satellite imagery recognition & 5,400 & 10 & accuracy & $\checkmark$ \\
\hline & FGVC Aircraft [95] & Aircraft recognition & 3,333 & 100 & mean per class & $\checkmark$ \\
\hline & Food-101 [18] & Food recognition & 25,250 & 101 & accuracy & $\checkmark$ \\
\hline & GTSRB [137] & Traffic sign recognition & 12,630 & 43 & accuracy & $\checkmark$ \\
\hline & ImageNet 1k [37] & Visual recognition & 50,000 & 1,000 & accuracy & $\checkmark$ \\
\hline & ImageNet Sketch [143] & Visual recognition & 50,889 & 1,000 & accuracy & $\checkmark$ \\
\hline & ImageNet V2 [121] & Visual recognition & 10,000 & 1,000 & accuracy & $\checkmark$ \\
\hline & ImageNet-A [65] & Visual recognition & 7,500 & 200 & accuracy & $\checkmark$ \\
\hline & ImageNet-O [65] & Visual recognition & 2,000 & 200 & accuracy & $\checkmark$ \\
\hline & ImageNet-R [64] & Visual recognition & 30,000 & 200 & accuracy & $\checkmark$ \\
\hline & KITTI distance $[48,156]$ & Distance prediction & 711 & 4 & accuracy & \\
\hline & MNIST [89] & Digit recognition & 10,000 & 10 & accuracy & $\checkmark$ \\
\hline & ObjectNet [13] & Visual recognition & 18,574 & 113 & accuracy & $\checkmark$ \\
\hline & Oxford Flowers-102 [102] & Flower recognition & 6,149 & 102 & mean per class & $\checkmark$ \\
\hline & Oxford-IIIT Pet $[105,156]$ & Pet classification & 3,669 & 37 & mean per class & $\checkmark$ \\
\hline & Pascal VOC 2007 [42] & Object recognition & 14,976 & 20 & accuracy & $\checkmark$ \\
\hline & PatchCamelyon $[142,156]$ & Metastatic tissue cls. & 32,768 & 2 & accuracy & \\
\hline & Rendered SST2 [156] & Sentiment classification & 1,821 & 2 & accuracy & $\checkmark$ \\
\hline & RESISC45 $[27,156]$ & Satellite imagery recognition & 6,300 & 45 & accuracy & $\checkmark$ \\
\hline & Stanford Cars [85] & Vehicle recognition & 8,041 & 196 & accuracy & $\checkmark$ \\
\hline & STL-10 [31] & Visual recognition & 8,000 & 10 & accuracy & $\checkmark$ \\
\hline & SUN-397 [146] & Scene recognition & 108,754 & 397 & accuracy & $\checkmark$ \\
\hline & SVHN $[99,156]$ & Digit recognition & 26032 & 10 & accuracy & $\checkmark$ \\
\hline & iWildCam $[14,83]$ & Animal recognition & 42,791 & 182 & macro F1 score & $\checkmark$ \\
\hline & Camelyon17 $[12,83]$ & Metastatic tissue cls. & 85,054 & 2 & accuracy & \\
\hline & FMoW $[29,83]$ & Satellite imagery recognition & 22,108 & 62 & worst-region acc. & $\checkmark$ \\
\hline & Dollar Street [122] & Object recognition & 3,503 & 58 & worst-income top-5 acc. & $\checkmark$ \\
\hline & GeoDE [114] & Object recognition & 12,488 & 40 & worst-region acc. & $\checkmark$ \\
\hline \multirow{3}{*}{ Retrieval } & Flickr30k [151] & Image and text retrieval & 31,014 & N/A & $\mathrm{R} @ 1$ & $\checkmark$ \\
\hline & $\operatorname{MSCOCO}[26]$ & Image and text retrieval & 5,000 & N/A & $\mathrm{R} @ 1$ & $\checkmark$ \\
\hline & WinoGAViL [17] & Commonsense association & 3,563 & N/A & Jaccard score & $\checkmark$ \\
\hline
\end{tabular}

Jaccard score (intersection-over-union) for each example, and show results for the harder samples (10 and 12 candidates). More information on WinoGAViL evaluation can be found in Bitton et al. [17].

Clean subset. For five of our evaluation tasks (the two CLEVR tasks, the two Camelyon tasks, and KITTI) the zero-shot performance of all evaluated models appears to be close to that of random guessing, and lack correlation to the type of filtering method used (see Figure 27). Consequently, we studied performance averaged only on the remaining 33 tasks, but found not substantial qualitative differences in our results. As a result, we opted to report the average on the full evaluation suite throughout our study.

Zero-shot vs. fine-tuning protocols. One critical decision in DATACOMP is how exactly to evaluate models and whether or not to fine-tune models on evaluation tasks (i.e., supervised fine-tuning directly on task training sets). We opt for zero-shot evaluation, where a models are applied to downstream tasks directly to 1) ease computational burden on participants and 2) measure the out-of-the-box generalization capabilities of our models. To validate this design decision, we conduct linear probes on all models presented in Tables 3 and 18 on ImageNet. We follow a standard probing protocol and fine-tune the last linear layer from zero-shot initialization for 40 epochs with learning rate $1 \mathrm{e}-3$, batch size 256, AdamW optimizer with default settings with the exception of weight decay (that we set to zero), and a cosine annealing schedule. As seen in Figure 16, zero-shot and linear probe performance follow similar trends for both filtering and BYOD tracks. Moreover the Spearman rank correlation between the two protocols over the models considered is 0.99 for the filtering track and 1.0 for BYOD. This suggests that better zero-shot models on ImageNet are correlated with better representations of linear probe fine-tuning on ImageNet.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-42.jpg?height=1729&width=917&top_left_y=260&top_left_x=604)

Figure 15: Randomly sampled images from the evaluation datasets we consider.

\section*{O. 1 Visual Question Answering}

In addition to our evaluation suite containing multiple classification and retrieval tasks, we conducted experiments on visual question answering. More specifically, following Shen et al. [132], we use the CLIP models to contrast images with prompts formed by the questions and each candidate answer, without fine-tuning (i.e., in a zero-shot setting). Using the VQA v1 dataset [2], for each candidate answer, we construct a text prompt that also includes the question following the template Question: [question text] Answer: [answer text], as in Ilharco et al. [70]. This text is then fed to CLIP's text encoder. As previously noted by multiple authors, CLIP models struggle on this task, potentially due to the mismatch between the text in the downstream task and the captions seen during pre-training Shen
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-43.jpg?height=418&width=1260&top_left_y=246&top_left_x=432)

Figure 16: Zero-shot ImageNet and Linear probe ImageNet performance for models from Tables 3 and 18. Relative ordering of models demonstrates high rank correlations of 0.99 and 1.0 for COMMONPOOL and BYOD respectively.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-43.jpg?height=452&width=1230&top_left_y=842&top_left_x=445)
$\checkmark$ No filtering
$\prec$ Basic
$\times$ CLIP score - Image-based + Text-based
Rand. subset
- ImageNet dist.

Figure 17: Correlation between zero-shot performance on the VQA v1 dataset and results on ImageNet and our full evaluation suite.

et al. [132], Ilharco et al. [70], Song et al. [134]. Nonetheless, we observe a strong correlation between VQA performance and ImageNet accuracy (0.877) and between VQA performance and average performance on our full evaluation suite. Full results are shown in Figure 17.

\section*{P Baseline details}

Here we provide additional details on the creation of our baseline subsets. To highlight the qualitative differences between the filtering strategies we also provide visualization for No filtering (Figure 18), Basic filtering (Figure 19), and CLIP score (L/1430\%) (Figure 20), which can all be found in Table 3. Notice that No filtering gives relatively noisy data (e.g., matching a bicycle with a caption: "IMG_2187.jpg"), while CLIP score samples give qualitatively more descriptive cations.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=322&width=253&top_left_y=598&top_left_x=440)

Organos muntoriales

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=209&width=141&top_left_y=936&top_left_x=493)

【iPhone6s Plus/6 Plusケ-

ス】WiFiブースター LINKASEクリア with WiFi スペースグレイ iPhone $6 \mathrm{~s}$ Plus/6 Plus_0

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=195&width=249&top_left_y=1255&top_left_x=445)

IMG_2187.jpg
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=566&width=512&top_left_y=1486&top_left_x=443)

Energy Stocks Fuel Market Rally
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=544&width=250&top_left_y=585&top_left_x=1094)

中村不折旧宅（書道博物館）

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=165&width=247&top_left_y=1262&top_left_x=1096)

Carregador portátil para Smartphones 5000mAh 5V 2.1A - JS Soluções em Segurança

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=193&width=133&top_left_y=1573&top_left_x=1148)

JUMP LEADS HEAVY DUTY COMMERCIAL 4.5 M 700 AMP

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-44.jpg?height=155&width=247&top_left_y=1901&top_left_x=1096)
热搜新闻一览

Figure 18: An i.i.d. sample from small CommonPool generated after applying the No filter strategy. Hence, these samples represent random images from COMMONPoOL.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=190&width=209&top_left_y=604&top_left_x=465)

status report templates 12+ free word documents download | free, Powerpoint templates

City 39 mm Quartz

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=258&width=250&top_left_y=1167&top_left_x=447)

1006: Rookwood pink mat vase, 1929, 2382, 6\&quot;

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=106&width=160&top_left_y=1557&top_left_x=495)

Chaussure De Running Junior Asics Gt-1000 Gs Bleu/vert - Asics - 37

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=236&width=158&top_left_y=1806&top_left_x=492)

Luxardo, Maraschino

Cherries, $14 \mathrm{FI} \mathrm{Oz}$

Grocery \& Gourmet Food

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=212&width=247&top_left_y=588&top_left_x=1096)

Implication in the classroom:

O Step 3: We Must Work it Out

Say and mean "We have to work it out". The behaviour cannot

co...

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=61&width=245&top_left_y=956&top_left_x=1100)

Astro ATA 3050 INSERTION TOOL, 16/20 GA

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=138&width=244&top_left_y=1232&top_left_x=1100)

WN | 2 Corinthians 10:1-18| Meekness or Boldness?

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=195&width=206&top_left_y=1515&top_left_x=1119)

Shopping Fairy Crochet Pattern, crochet wings, crochet shopping bags, crochet doll

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-45.jpg?height=241&width=198&top_left_y=1793&top_left_x=1126)

Essay Outlines Exles by Writing Center Workshops The Outline

Figure 19: An i.i.d. sample from small CommonPool generated after applying the Basic filter strategy.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=369&width=266&top_left_y=561&top_left_x=431)

Sacred Geometry Egg Of Life, Sacred Geometry Symbols, Golden Ratio, Flower Of Life, Wicca, Magick, Tattoos, Geometric Nature, Geometric Mandala

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=334&width=244&top_left_y=581&top_left_x=1060)

The Martian Monster And Other Stories (The EC Comics Library) ()

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=179&width=249&top_left_y=946&top_left_x=445)

Porsche Cayman S

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=198&width=249&top_left_y=931&top_left_x=1060)

Mesmerizing Black, Silver \&

Pink Handmade Modern Metal Wall Art Sculpture Metallic One of a Kind Abstract Painting - OOAK 546 by Jon Allen
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=344&width=816&top_left_y=1167&top_left_x=489)

Under Armour Heatgear Gotta Have It Shorty Women's at Foot Locker
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=154&width=850&top_left_y=1576&top_left_x=455)

Football Manager 2016
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-46.jpg?height=152&width=854&top_left_y=1889&top_left_x=450)

Profitable forex trading systems

Figure 20: An i.i.d. sample from small CommonPool generated after applying the CLIP score (L/14 30\%)

strategy.

\section*{P. 1 Filtering track}

Basic filtering. For language detection, we use Fasttext 0.92 , version lid.176, and cld3 - library gcld3 3.0.13. We count the number of words in each caption by splitting using whitespaces.

CLIP thresholds. We use OpenAI pretrained CLIP ViT-B/32 and ViT-L/14 models [111] to compute the cosine similarity text and image tower outputs as the CLIP scores. On the small and medium pools, we also experiment with baselines that filter out samples in the top few percentiles of CLIP scores. Specifically, we try baselines that use samples with top $\{1,2,5\}-30 \%$ CLIP scores (ViT-B/32 model), and the performance is sightly better on the small pool (at most 0.5 gain of averaged accuracy) while slightly worse on the medium pool (0.4-0.8 loss of averaged accuracy). In Table 16, we show how the CLIP score thresholds relate to the fraction of the pool retained by the filter.

Text-based filtering. Each synset is represented by a synset offset that can be used to retrieve the synset from WordNet. In order to verify if a caption has a word corresponding to a synset from our set we iterate over every word and retrieve the synsets that this word can describe (using nltk.corpus WordNet). Following that, we retrieve the most likely lemma representing that synset, find its synset offset, and check if the number is part of the IN21K or IN1K sets. ${ }^{7}$

Text-based sampling. This baseline uses text only to filter labels which mention concepts (synsets) appearing in IN21K, and applies a temperature parameter to control how equally-represented different concepts are in the dataset. For synset $j$, let $N_{j}$ be the number of examples containing words matched to that synset, where as before for each word we only match the most likely synset. Furthermore, for image-text pair $i$ let $T_{i}$ be the set of synset matched to the caption.

The probability of sampling example $i$ is proportional to either $\frac{1}{\left|T_{i}\right|} \sum_{j \in T_{i}} N_{j}^{\alpha-1}$ (average synset score in the data point) or $\max _{j \in T_{i}} N_{j}^{\alpha-1}$ (maximum synset score in the data point), where $\alpha$ is a "temperature" parameter controlling the flatness of the distribution. We sample examples with replacement but discard any example repeated more than 100 times.

Image-based filtering. We now provide a detailed description of the Image-based filtering procedure. First, since the core of the procedure concerns only image content, we begin with basic text-bsaed filtering: we remove from the pool only all examples with non-English captions (as determined by fasttext), and all examples whose captions have less than two words or less than six characters.

Next, we use clustering of image embeddings to select a subset of examples whose image content is related to a clean training set of interest. Let $e_{1}, \ldots, e_{M}$ denote the CLIP image embeddings of the remaining examples in the pool. We cluster these embeddings into $K=10^{5}$ clusters using Faiss with 20 iterations, and let $c_{1}, \ldots, c_{K}$ denote the resulting cluster centers. Due to memory constraints, for the large and xlarge pools, we perform the clustering on a random subset of about $160 \mathrm{M}$ examples (that pass the basic text-based filtering). For an embedding vector $v$, let

$$
I(v)=\arg \max _{i \leq K}\left\langle v, c_{i}\right\rangle
$$

denote the index of the cluster center nearest to $v$ as measured by inner product. Let $f_{1}, \ldots, f_{N}$ denote the CLIP image embeddings of a clean supervised training set (we experiment with either ImageNet $1 \mathrm{~K}$ or ImageNet $21 \mathrm{~K}$ ), and let

$$
\mathcal{S}=\left\{I\left(f_{i}\right) \mid 1 \leq i \leq N\right\}
$$

be the set of cluster indices who are nearest neighbors to some clean training set image. We then keep only images in the pool whose nearest cluster center is in $\mathcal{S}$. That is, out of the $M$ examples passing the text-based filtering, the output subset keeps the examples with indices

$$
\left\{1 \leq j \leq M \mid I\left(e_{j}\right) \in \mathcal{S}\right\}
$$

Image-based sampling. In addition to filtering methods, we experiment with cluster-based sampling methods. First, we compute the score of $i$-th cluster $s_{i}$ as the number of ImageNet data assigned to this cluster. Then, for parameter $\alpha>0$ we define a distribution over the pool by sampling cluster $i$ with probability $\frac{s_{i}^{\alpha}}{\sum_{j} s_{j}^{\alpha}}$ and uniformly sampling an example for the cluster, rejecting any example repeated more than 100 times. We try 5 different $\alpha$, i.e., $\{0,0.2,0.5,1.0,2.0\}$, and the best average accuracy is obtained when $\alpha=0.2$, while the performance is still worse than the image-based filtering on the small and medium pool. We therefore do not include this line of baselines in the experiments of large pool.
\footnotetext{
${ }^{7}$ For the ImageNet $21 \mathrm{~K}$ synsets, we have used the list in https://storage.googleapis.com/bit_ models/imagenet21k_wordnet_ids.txt
}

Table 16: CLIP threshold filtering configurations. "Fraction" denotes the size of the filtered subset relative to the pool.

\begin{tabular}{|c|c|c|c|}
\hline$\overline{\text { CLIP model }}$ & En. filtering & Thresholc & $\overline{\text { Fraction }}$ \\
\hline ViT-B/32 & $x$ & 0.384 & $1 \%$ \\
\hline ViT-B/32 & $x$ & 0.358 & $3 \%$ \\
\hline ViT-B/32 & $\checkmark$ & 0.300 & $10.2 \%$ \\
\hline ViT-B/32 & $x$ & 0.325 & $10 \%$ \\
\hline ViT-B/32 & $\checkmark$ & 0.28 & $7.4 \%$ \\
\hline ViT-B/32 & $x$ & 0.300 & $20 \%$ \\
\hline ViT-B/32 & $x$ & 0.281 & $30 \%$ \\
\hline ViT-B/32 & $x$ & 0.263 & $40 \%$ \\
\hline ViT-B/32 & $x$ & 0.247 & $50 \%$ \\
\hline ViT-B/32 & $x$ & 0.215 & $75 \%$ \\
\hline ViT-B/32 & $x$ & 0.193 & $90 \%$ \\
\hline$\overline{\text { ViT-L/14 }}$ & $\bar{x}$ & 0.364 & $1 \%$ \\
\hline ViT-L/14 & $x$ & 0.334 & $3 \%$ \\
\hline ViT-L/14 & $\checkmark$ & 0.300 & $5.4 \%$ \\
\hline ViT-L/14 & $x$ & 0.295 & $10 \%$ \\
\hline ViT-L/14 & $\checkmark$ & 0.280 & $3.3 \%$ \\
\hline ViT-L/14 & $x$ & 0.266 & $20 \%$ \\
\hline ViT-L/14 & $x$ & 0.243 & $30 \%$ \\
\hline ViT-L/14 & $x$ & 0.222 & $40 \%$ \\
\hline ViT-L/14 & $x$ & 0.203 & $50 \%$ \\
\hline ViT-L/14 & $x$ & 0.160 & $75 \%$ \\
\hline ViT-L/14 & $x$ & 0.129 & $90 \%$ \\
\hline
\end{tabular}

ImageNet distance filtering. We rank the samples in the pool by the minimum embedding distance ( 1 minus cosine similarity) between its image and the ImageNet images; both embeddings are obtained from OpenAI pretrained CLIP ViT-L/14 model [111]. Then we select top images by different fractions as in image-based filtering methods.

\section*{P. 2 BYOD track}

We experiment with the following data sources:

- CC12M [24]: images and HTML alt-text crawled and filtered from web pages.

- YFCC15M: this is the 15M subset of the YFCC100M dataset [140] that Radford et al. [111] used for dataset ablation in their CLIP paper.

- RedCaps [38]: 12M images and corresponding captions were crawled from 350 manually curated subreddits between 2008 and 2020.

- Shutterstock: 106M images and captions were obtained from the Shutterstock website in 2021 [101]. We use the "photos" subset of this dataset, with $58 \mathrm{M}$ samples, which we found performed best, unless specified otherwise.

- WIT [136]: Image-text pairs from Wikipedia pages. We use the attribution fields as captions, which we found performed best.

- COYO [20]: A collection of 700M image-text pairs from Common Crawl.

- LAION-2B [129]: A 2.32 billion english subset of LAION-5B.

- LAION-COCO: A dataset with 600M images from LAION-5B and synthetic captions. ${ }^{8}$

- LAION-A: According to laion.ai, LAION-A is a 900M subset of LAION-2B [129] with the aesthetic filtering procedure used in LAION-aesthetic ${ }^{9}$ and pHash deduplication [72].

In Table 17, we use some heuristics to measure the quality of some external data sources. First, following Nguyen et al. [101], we train a CLIP model on a 5M random subset from each source, and evaluate the performance of the resulting models on ImageNet and ImageNet-derived distributions - ImageNet-V2 [121], ImageNet-R [64],
\footnotetext{
${ }^{8}$ https://laion.ai/blog/laion-coco/

${ }^{9}$ https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
}

Table 17: Measuring the quality of external data sources

\begin{tabular}{lccccc}
\hline Dataset & Dataset size & ImageNet acc. & \begin{tabular}{c} 
Avg. accuracy \\
ImageNet and OOD sets
\end{tabular} & Avg. cos. sim. (B/32) & Avg. cos. sim. (L/14) \\
\hline CC12M & $10 \mathrm{M}$ & 27.8 & 34.0 & 0.306 & 0.268 \\
YFCC15M & $15 \mathrm{M}$ & 22.6 & 24.6 & 0.262 & 0.198 \\
RedCaps & $11 \mathrm{M}$ & 26.8 & 31.5 & 0.281 & 0.240 \\
Shutterstock & $15 \mathrm{M}$ & 21.0 & 28.3 & 0.314 & 0.273 \\
\hline
\end{tabular}

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-49.jpg?height=569&width=965&top_left_y=556&top_left_x=580)

Figure 21: Comparison of average and worst-group scores for Dollar Street and GeoDE diversity datasets. On Dollar Street, our overall higher-performing models display a larger worst-group performance gap (corresponding to lower income households). GeoDE does not show this trend.

ImageNet-Sketch [143] and ObjectNet [13]. Moreover, for each data source, we use OpenAI's pretrained CLIP ViT-B/32 and ViT-L/14 models to compute the cosine similarity between image and text embeddings of a data point, and obtain the average cosine similarity score for the whole dataset.

\section*{P.2.1 Additional results}

We present a series of results for the BYOD track in Table 18.

\section*{Q Fairness and biases}

To study the biases displayed by our models, we include two diversity-related datasets, Dollar Street [122] and GeoDE [114], in our evaluation suite, and perform further analysis on the face datasets FairFace [80] and UTKFace [159] with demographic labels, following Radford et al. [111].

\section*{Q. 1 Diversity}

We break down model performance on the Dollar Street and GeoDE datasets in Figure 21. Dollar Street consists of images of household items taken in homes around the world, and represents a wide socioeconomic range that includes homes with no Internet access [122]. The objects belong to ImageNet categories, and the task is image classification. Standard ImageNet-trained models achieve monotonically increasing performance levels with higher household income levels [122]. Here we use the income-based subgroups defined in Rojas et al. [122], and find a similar bias as discovered in their paper. While our trained models show a smaller worst-group performance gap than an ImageNet-trained ResNet-50, they underperform a model fine-tuned on Dollar Street. Models with higher average accuracy show a larger worst-group gap, which future work should try to address.

GeoDE consists of images of everyday items and objects, which again fall into ImageNet categories. The dataset represents six world regions equally, and primarily aims to promote geographic diversity of datasets [114]. Both ImageNet models and our models show less bias under this distribution compared to Dollar Street, with a smaller worst-group accuracy gap. The trends show that performance across all regions improves steadily with increased scale, and the performance approaches that of a model fine-tuned on GeoDE. While we know that classifiers trained specifically on ImageNet can display geographic biases [114], these biases are not apparent in our GeoDE model evaluations. Future work is needed to investigate the extent to which our models have geographic biases not evaluated in GeoDE.

Table 18: Zero-shot performance for select baselines in the BYOD track. Unless specified otherwise, CommonPooL means our pool filtered with CLIP score (L/14, 30\%).

\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline Scale & Data source & \begin{tabular}{l} 
Training \\
dataset size \\
\end{tabular} & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts \\
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline \multirow{9}{*}{ small } & $\# 0$ & CC12M & 0.099 & 0.080 & 0.223 & 0.197 & 0.205 \\
\hline & \begin{tabular}{ll}
$\# 1$
\end{tabular} & LAION15M & 0.083 & 0.076 & 0.210 & 0.144 & 0.189 \\
\hline & \#2 & RedCaps & 0.076 & 0.066 & 0.177 & 0.141 & 0.168 \\
\hline & \#3 & Shutterstock 15M & 0.083 & 0.070 & 0.214 & 0.159 & 0.185 \\
\hline & $\# 4$ & YFCC15M & 0.071 & 0.046 & 0.182 & 0.147 & 0.164 \\
\hline & \#5 & $\# 0+\# 1+\# 2$ & 0.097 & 0.084 & 0.208 & 0.161 & 0.195 \\
\hline & \#6 & $\# 0+\# 1+\# 3$ & 0.091 & 0.081 & 0.222 & 0.138 & 0.202 \\
\hline & \#7 & $\# 0+\# 2+\# 3$ + \#4 & 0.095 & 0.075 & 0.205 & 0.164 & 0.186 \\
\hline & \begin{tabular}{ll}
$\# 8$
\end{tabular} & $\# 0-4$ & 0.093 & 0.076 & 0.205 & 0.162 & 0.193 \\
\hline \multirow{17}{*}{ medium } & \#9 & CC12M & 0.245 & 0.189 & 0.283 & 0.289 & 0.272 \\
\hline & \begin{tabular}{ll}
$\# 10$
\end{tabular} & LAION15M & 0.270 & 0.215 & 0.317 & 0.255 & 0.306 \\
\hline & \begin{tabular}{lll}
$\# 11$
\end{tabular} & RedCaps & 0.237 & 0.166 & 0.271 & 0.178 & 0.263 \\
\hline & \begin{tabular}{ll}
$\# 12$
\end{tabular} & Shutterstock 15M & 0.229 & 0.191 & 0.316 & 0.260 & 0.290 \\
\hline & \begin{tabular}{ll}
$\# 13$
\end{tabular} & YFCC15M & 0.232 & 0.137 & 0.263 & 0.245 & 0.257 \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-50.jpg?height=35\&width=136\&top_left_y=800\&top_left_x=486) & $\# 9+\# 10+\# 11$ & 0.376 & 0.287 & 0.387 & 0.323 & 0.366 \\
\hline & \begin{tabular}{ll}
$\# 15$
\end{tabular} & $\# 9+\# 10+\# 12$ & 0.342 & 0.278 & 0.362 & 0.345 & 0.357 \\
\hline & \begin{tabular}{l}
$\# 16$
\end{tabular} & $\# 9+\# 11+\# 12+\# 13$ & 0.360 & 0.268 & 0.365 & 0.275 & 0.345 \\
\hline & \begin{tabular}{lll}
$\# 17$
\end{tabular} & \#9-13 & 0.371 & 0.285 & 0.408 & 0.280 & 0.367 \\
\hline & \begin{tabular}{l}
$\# 18$
\end{tabular} & Shutterstock illustration & 0.053 & 0.094 & 0.205 & 0.125 & 0.180 \\
\hline & $\# 19$ & Shutterstock photo & 0.342 & 0.209 & 0.364 & 0.350 & 0.331 \\
\hline & \begin{tabular}{l} 
\#20
\end{tabular} & Shutterstock vectors & 0.072 & 0.151 & 0.216 & 0.148 & 0.208 \\
\hline & $\# 21$ & Shutterstock full & 0.313 & 0.254 & 0.353 & 0.331 & 0.342 \\
\hline & \begin{tabular}{l}
$\# 22$
\end{tabular} & WIT full & 0.096 & 0.063 & 0.196 & 0.104 & 0.177 \\
\hline & $\# 23$ & WIT English & 0.051 & 0.038 & 0.145 & 0.083 & 0.143 \\
\hline & \#24 & COYO & 0.272 & 0.235 & 0.301 & 0.254 & 0.304 \\
\hline & $\# 25$ & LAION-COCO & 0.209 & 0.205 & 0.293 & 0.359 & 0.297 \\
\hline \multirow{24}{*}{ large } & $\# 26$ & Shutterstock illustration & 0.337 & 0.203 & 0.307 & 0.322 & 0.306 \\
\hline & \begin{tabular}{l}
$\# 27$
\end{tabular} & Shutterstock photo & 0.485 & 0.304 & 0.432 & 0.427 & 0.398 \\
\hline & \begin{tabular}{l}
$\# 28$
\end{tabular} & Shutterstock vectors & 0.126 & 0.223 & 0.244 & 0.191 & 0.246 \\
\hline & \begin{tabular}{l}
$\# 29$
\end{tabular} & Shutterstock full & 0.500 & 0.412 & 0.472 & 0.451 & 0.456 \\
\hline & \#30 & COYO & 0.547 & 0.456 & 0.475 & 0.549 & 0.486 \\
\hline & $\# 31$ & LAION-COCO & 0.355 & 0.351 & 0.395 & 0.494 & 0.398 \\
\hline & $\# 32$ & $\mathrm{COYO}+\mathrm{LAION}-\mathrm{COCO}$ & 0.528 & 0.458 & 0.479 & 0.589 & 0.498 \\
\hline & $\# 33$ & LAION-A & 0.611 & 0.474 & 0.501 & 0.542 & 0.505 \\
\hline & $\# 34$ & LAION-2B & 0.585 & 0.472 & 0.504 & 0.525 & 0.515 \\
\hline & $\# 35$ & CoMMONPOOL + \#9-13 & 0.602 & 0.498 & 0.541 & 0.416 & 0.537 \\
\hline & \#36 & CoMMONPOOL + \#9-13 (2x upsampled) & 0.613 & 0.507 & 0.559 & 0.433 & 0.543 \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-50.jpg?height=35\&width=136\&top_left_y=1484\&top_left_x=486) & CoMMONPOOL + \#9-13 (4x upsampled) & 0.615 & 0.514 & 0.553 & 0.427 & 0.543 \\
\hline & \begin{tabular}{lll}
$\# 38$
\end{tabular} & CoMMONPOOL $+\# 9-13$ (6x upsampled) & 0.620 & 0.519 & 0.558 & 0.437 & 0.549 \\
\hline & \begin{tabular}{lll}
$\# 39$
\end{tabular} & CoMMONPOOL + \#9-13 (8x upsampled) & 0.624 & 0.520 & 0.533 & 0.443 & 0.537 \\
\hline & \begin{tabular}{ll}
$\# 40$
\end{tabular} & ComMONPooL + \#9-13 (10x upsampled) & 0.621 & 0.520 & 0.540 & 0.441 & 0.537 \\
\hline & \begin{tabular}{lllll}
$\# 41$ & 0
\end{tabular} & CoMMONPOOL + COYO & 0.561 & 0.472 & 0.504 & 0.508 & 0.513 \\
\hline & \begin{tabular}{lll}
$\# 42$
\end{tabular} & COMMONPOOL + LAION-A & 0.607 & 0.480 & 0.531 & 0.514 & 0.527 \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-50.jpg?height=35\&width=136\&top_left_y=1661\&top_left_x=486) & COMMONPOOL + LAION-COCO & 0.522 & 0.457 & 0.513 & 0.498 & 0.514 \\
\hline & \begin{tabular}{lllll}
$\# 44$ & 0
\end{tabular} & CoMMONPoOL + \#9+\#11+\#13+\#19 & 0.609 & 0.508 & 0.546 & 0.439 & 0.536 \\
\hline & \begin{tabular}{lll}
$\# 45$
\end{tabular} & CoMMONPOOL + \#9+\#11+\#13+\#19 (2x upsampled) & 0.621 & 0.509 & 0.547 & 0.458 & 0.541 \\
\hline & \begin{tabular}{lll}
$\# 46$
\end{tabular} & CoMMONPOOL +\#9+\#11+\#13+\#19 (4x upsampled) & 0.632 & 0.515 & 0.533 & 0.452 & 0.532 \\
\hline & \begin{tabular}{lllll}
$\# 47$ & 0
\end{tabular} & CoMMONPOOL + \#9+\#11+\#13+\#19 (6x upsampled) & 0.635 & 0.515 & 0.535 & 0.471 & 0.532 \\
\hline & \begin{tabular}{lll}
$\# 48$
\end{tabular} & CoMMONPOOL + \#9+\#11+\#13+\#19 (8x upsampled) & 0.633 & 0.515 & 0.523 & 0.464 & 0.530 \\
\hline & \begin{tabular}{ll}
$\# 49$
\end{tabular} & COMMONPOOL + \#9+\#11+\#13+\#19 (10x upsampled) & 0.630 & 0.513 & 0.523 & 0.356 & 0.521 \\
\hline \multirow{4}{*}{ xlarge } & $\# 50$ & LAION-2B & 0.757 & 0.631 & 0.611 & 0.619 & 0.621 \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-50.jpg?height=35\&width=136\&top_left_y=1899\&top_left_x=486) & CoMMONPoOL + \#9+\#11+\#13+\#19 & 0.766 & 0.660 & 0.662 & 0.539 & 0.659 \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-50.jpg?height=35\&width=136\&top_left_y=1927\&top_left_x=486) & CoMmONPOOL +\#9+\#11+\#13+\#19 (6x upsampled) & 0.776 & 0.671 & 0.633 & 0.552 & 0.649 \\
\hline & $\# 53$ & COMMONPOOL + \#9+\#11+\#13+\#19 (18x upsampled) & 0.771 & 0.667 & 0.629 & 0.554 & 0.643 \\
\hline
\end{tabular}

\section*{Q. 2 Fairness}

Emulating Radford et al. [111], we evaluate our best models from the filtering and BYOD tracks on the human face datasets FairFace and UTKFace, using zero-shot classification to predict the race, gender, and age annotated in these datasets. Following Hanna et al. [59] and Hundt et al. [68], we acknowledge that these evaluations can be problematic as race and gender should not be considered fixed categories, but rather fluid attributes that may change for individuals, based on they way they identify at any given moment-regardless of appearance. We include these evaluations for continuity with prior work and as a probe into model behaviour, but hope future work will consider improved face fairness evaluation. We also note that race, gender, and age classification are not the intended end-goals of the models or benchmark, and we do not condone the use of CommonPool or models trained on CommonPool data for any decisions involving people.

Table 19: Overall race, gender, and age classification accuracy of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 30\%) for the filtering track and CommonPool, CLIP score + 4 external sources (upsampled 6x) for the BYOD track. Race classification was binary (white or non-white) as in Karkkainen \& Joo [80].

\begin{tabular}{llccc}
\hline Dataset & Track & Race & Gender & Age \\
\hline \multirow{2}{*}{ FairFace } & Filtering & 86.4 & 91.7 & 34.3 \\
& BYOD & 76.5 & 93.9 & 33.8 \\
\hline \multirow{2}{*}{ UTKFace } & Filtering & 86.2 & 93.8 & 39.5 \\
& BYOD & 86.1 & 95.5 & 38.6 \\
\hline
\end{tabular}

Table 20: Gender classification accuracy of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 30\%) for the filtering track and CommonPooL, CLIP score +4 external sources (upsampled $6 x$ ) for the BYOD track.

\begin{tabular}{lllcccccc}
\hline \multirow{2}{*}{ Track } & \multirow{2}{*}{ Gender } & & & & \multicolumn{4}{c}{ Race } \\
& & Black & White & Indian & Latino/Hispanic & Middle Eastern & Southeast Asian & East Asian \\
\hline \multirow{2}{*}{ Filtering } & Male & 79.3 & 91.3 & 90.8 & 90.4 & 95.7 & 83.0 & 80.7 \\
& Female & 95.4 & 96.6 & 94.2 & 96.6 & 96.5 & 97.2 & 98.2 \\
\hline \multirow{2}{*}{ BYOD } & Male & 89.2 & 94.8 & 93.2 & 93.4 & 97.4 & 90.2 & 90.6 \\
& Female & 89.2 & 96.0 & 94.2 & 96.0 & 96.2 & 97.1 & 97.0 \\
\hline
\end{tabular}

\begin{tabular}{llccccc}
\multicolumn{7}{c}{ UTKFace } \\
\hline \multirow{2}{*}{ Track } & \multirow{2}{*}{ Gender } & \multicolumn{5}{c}{ Race } \\
& & Black & White & Indian & Asian & Other \\
\hline \multirow{2}{*}{ Filtering } & Male & 95.4 & 92.5 & 91.7 & 73.1 & 84.2 \\
& Female & 97.3 & 98.7 & 97.4 & 98.3 & 97.4 \\
\hline \multirow{2}{*}{ BYOD } & Male & 96.8 & 95.9 & 94.7 & 85.7 & 90.4 \\
& Female & 96.3 & 97.7 & 96.8 & 95.9 & 95.6 \\
\hline
\end{tabular}

As described in Appendix G, our filleting track models are trained on images with faces blurred. Nevertheless, these models still perform significantly above random chance on face classification. We hypothesize that this is due to a combination of faces bypassing our face blurring filter in the training data, contextual clues outside of the face region, or signal associated with skin color. The BYOD track model performs even better than the filtering track model. We hypothesize that this is because BYOD data is used off-the-shelf and hence contains non-blurred faces. In Table 19, we present overall accuracy for these three traits. Note that race is treated as a binary variable (white or non-white) to enable comparison to prior results, gender is a binary variable (male or female) according to annotations, and age is binned into 9 ranges according to the annotation precision of FairFace. The BYOD model, performs better at distinguishing the annotated gender, but is worse at distinguishing annotated race and age.

We further break down these statistics over the intersection of race and gender, examining gender classification accuracies in Table 20. We find that there are drastic differences in accuracy across different annotated subgroups, varying by both race and gender. The filtering models shows a tendency to misclassify Black, Southeast Asian, and East Asian males as females at $20.7 \%, 17 \%$, and $19.3 \%$ respectively on FairFace. Furthermore, we find that while the BYOD model improves accuracy, on FairFace most of this improvement is on men (ranging from $1.7 \mathrm{pp}$ gain to $9.9 \mathrm{pp}$ gain), while on women, BYOD offers little change (ranging from $0.6 \mathrm{pp}$ gain to $6.2 \mathrm{pp}$ drop).

Following Radford et al. [111], we also examined associations of particular demographics with potentially harmful language. We replicate their setup with two classification tasks: (1) including race-gender intersection classes (e.g. "black woman", "indian man", etc.) and several harmful crime-related terms ("thief", "criminal", "suspicious person"); (2) including the same race-gender intersection classes and non-human terms ("animal", "gorilla", "chimpanzee", "orangutan"). We compute the frequency of misclassification of people into one of the harmful categories and run these experiments on FairFace and UTKFace separately. The results are shown in Table 21. Unlike in Radford et al. [111], we find that our models have a very small probability of classifying human faces as non-human, with a max score across all subgroups of $0.1 \%$. However, a significant proportion of people are misclassified as criminal. This again highlights the importance of dataset curation and the risks associated with zero-shot classification on models trained on web-scraped datasets.

Table 21: Harmful misclassification rates of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 30\%) for the filtering track and CommonPooL, CLIP score +4 external sources (upsampled $6 \mathrm{x})$ for the BYOD track. While very few samples are misclassified as non-human, the filter track model assigns a crime-related label to a significant portion of people, and this is exacerbated by the BYOD model in many cases.

FairFace

\begin{tabular}{llccccccc}
\hline \multirow{2}{*}{ Track } & & & & \multicolumn{7}{c}{ FairFace } & Race \\
& & Black & White & Indian & Latino/Hispanic & Middle Eastern & Southeast Asian & East Asian \\
\hline \multirow{2}{*}{ Filtering } & Crime-related & 4.4 & 24.3 & 8.8 & 14.3 & 23.7 & 7.4 & 8.6 \\
& Non-human & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
\hline \multirow{2}{*}{ BYOD } & Crime-related & 18.4 & 16.8 & 21.5 & 22.9 & 20.9 & 35.3 & 30.9 \\
& Non-human & 0.0 & 0.1 & 0.0 & 0.1 & 0.0 & 0.1 & 0.1 \\
\hline
\end{tabular}

\begin{tabular}{llccccc}
\multicolumn{7}{c}{ UTKFace } \\
\hline \multirow{2}{*}{ Track } & & & & Race \\
& & Black & White & Indian & Asian & Other \\
\hline \multirow{2}{*}{ Filtering } & Crime-related & 6.8 & 16.1 & 9.1 & 6.9 & 13.9 \\
& Non-human & 0.0 & 0.2 & 0.0 & 0.1 & 0.0 \\
\hline \multirow{2}{*}{ BYOD } & Crime-related & 12.8 & 10.8 & 15.2 & 13.2 & 18.6 \\
& Non-human & 0.0 & 0.2 & 0.0 & 0.0 & 0.0 \\
\hline
\end{tabular}

\section*{R Extra figures and tables}
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-53.jpg?height=778&width=1378&top_left_y=348&top_left_x=362)

\prec Basic < Image-based

+ Text-based

Figure 22: Improving downstream performance at smaller scales correlates positively with performance gains at larger scales. These trends suggests that dataset filtering can be studied effectively at smaller scales, even with less computational resources.

Table 22: Rank correlation between the performance obtained with various filtering strategies at two different scales. Our experimental suggest that the ranking is relatively consistent between scales, especially for the adjacent scale pairs.

\begin{tabular}{lccc}
\hline Metric & small vs medium & small vs large & medium vs large \\
\hline ImageNet acc. & 0.895 & 0.811 & 0.847 \\
Average pref. metric & 0.854 & 0.708 & 0.876 \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-54.jpg?height=1478&width=1326&top_left_y=250&top_left_x=364)
Y No filtering
Basic
$\times$ CLIP score
Image-based
+ Text-based
\& Rand. subset

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-54.jpg?height=471&width=466&top_left_y=1247&top_left_x=881)

Figure 23: Performance as a function of the number of training samples from the small (top), medium (middle), and large (bottom) scales. There is a significant variance in accuracy even when accounting for the size of the training set.

Table 23: Comparison of ViT-B/32 and ViT-B/16 models across different training datasets.

\begin{tabular}{lcccccccc}
\hline Model & Training Dataset & \begin{tabular}{c} 
Training \\
dataset size
\end{tabular} & \begin{tabular}{c} 
Training \\
steps
\end{tabular} & ImageNet & \begin{tabular}{c} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline ViT B/32 & DATACOMP-1B & $1.4 \mathrm{~B}$ & $13 \mathrm{~B}$ & 0.692 & 0.551 & 0.577 & 0.538 & 0.579 \\
ViT B/32 & OpenAI's WIT & $0.4 \mathrm{~B}$ & $13 \mathrm{~B}$ & 0.633 & 0.485 & 0.526 & 0.501 & 0.525 \\
ViT B/32 & LAION-2B & $2.3 \mathrm{~B}$ & $34 \mathrm{~B}$ & 0.666 & 0.522 & 0.561 & 0.560 & 0.569 \\
ViT B/16 & DATACOMP-1B & $1.4 \mathrm{~B}$ & $13 \mathrm{~B}$ & 0.735 & 0.608 & 0.621 & 0.578 & 0.615 \\
ViT B/16 & OpenAI's WIT & $0.4 \mathrm{~B}$ & $13 \mathrm{~B}$ & 0.683 & 0.559 & 0.546 & 0.527 & 0.563 \\
ViT B/16 & LAION-2B & $2.3 \mathrm{~B}$ & $34 \mathrm{~B}$ & 0.702 & 0.566 & 0.572 & 0.583 & 0.587 \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-55.jpg?height=516&width=1030&top_left_y=254&top_left_x=539)

Figure 24: We examine the percentage of texts classified as English after taking the top fraction (on the x-axis) of the large billion pool as sorted by CLIP similarity score. We see that doing CLIP filtering implicitly does some English filtering, as image-text pairs with a higher CLIP score are more frequently classified as English.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-55.jpg?height=502&width=1374&top_left_y=985&top_left_x=365)

Figure 25: Correlation between ImageNet accuracy and average performance on our suite of evaluation tasks. While ImageNet accuracy strongly correlates with the average performance (both on the clean subset and the full suite), the same is not true for all individual datasets we study, as shown in Appendix R.
![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-55.jpg?height=640&width=960&top_left_y=1705&top_left_x=580)

\begin{tabular}{|clllll}
--- & $x=y$ & $\prec$ & Basic & + & Text-based \\
$\circ$ & ImageNet models & $\times$ & CLIP score & $\curlywedge$ & Rand. subset \\
$\gamma$ & No filtering & & Image-based & $\circ$ & ImageNet dist.
\end{tabular}

Figure 26: Zero-shot CLIP models trained with various filtering strategies form a reliable trend relating accuracy on ImageNet and related distribution shifts, exhibiting higher effective robustness when compared to ImageNet-trained models from Taori et al. [139].

Table 24: Comparison at the xlarge scale between a 400M subset of CommonPool and OpenAI's WIT which also contains $400 \mathrm{M}$ samples. Our $400 \mathrm{M}$ subset is created by intersecting IN1k image clustering with English cld3 filtering, then taking the top 400M samples sorted by CLIP L14 score. Our model does better across the various evaluation groupings.

![](https://cdn.mathpix.com/cropped/2024_06_04_6f9b87be2abbbd9d58bag-56.jpg?height=1946&width=1393&top_left_y=409&top_left_x=363)

Figure 27: Zero-shot performance on other datasets is often positively correlated with that on ImageNet, but not always. In cases where ImageNet shows close to zero correlation with other datasets, performance on that dataset is often close to random chance.

Table 25: Baseline results for the filtering track, small scale.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Filtering & \begin{tabular}{c} 
Training \\
dataset size
\end{tabular} & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{l} 
Average over \\
38 datasets
\end{tabular} \\
\hline No filtering & $12.8 \mathrm{M}$ & 0.025 & 0.033 & 0.145 & 0.114 & 0.133 \\
\hline Random subset (75\%) & $9.6 \mathrm{M}$ & 0.028 & 0.037 & 0.153 & 0.110 & 0.140 \\
\hline Random subset ( $50 \%$ ) & $6.4 \mathrm{M}$ & 0.027 & 0.037 & 0.147 & 0.111 & 0.137 \\
\hline Random subset (25\%) & $3.2 \mathrm{M}$ & 0.022 & 0.032 & 0.130 & 0.099 & 0.126 \\
\hline Random subset $(10 \%)$ & $1.3 \mathrm{M}$ & 0.010 & 0.018 & 0.116 & 0.077 & 0.103 \\
\hline Random subset ( $1 \%$ ) & $128 \mathrm{~K}$ & 0.002 & 0.005 & 0.095 & 0.049 & 0.078 \\
\hline Caption length & 8.7M & 0.034 & 0.040 & 0.148 & 0.109 & 0.143 \\
\hline Image size & $7.8 \mathrm{M}$ & 0.027 & 0.036 & 0.154 & 0.119 & 0.138 \\
\hline English (fasttext) & $6.3 \mathrm{M}$ & 0.038 & 0.045 & 0.164 & 0.124 & 0.154 \\
\hline English (fasttext) and caption length & $4.8 \mathrm{M}$ & 0.041 & 0.048 & 0.159 & 0.123 & 0.154 \\
\hline English (fasttext), caption length, and image size & $3.0 \mathrm{M}$ & 0.038 & 0.043 & 0.150 & 0.118 & 0.142 \\
\hline English (cld3) & $2.6 \mathrm{M}$ & 0.032 & 0.039 & 0.143 & 0.111 & 0.142 \\
\hline English (cld3) and caption length & $2.3 \mathrm{M}$ & 0.031 & 0.038 & 0.153 & 0.111 & 0.142 \\
\hline English (cld3), caption length, and image size & $1.5 \mathrm{M}$ & 0.023 & 0.030 & 0.154 & 0.092 & 0.141 \\
\hline CLIP B32 score top 1\% & $129 \mathrm{~K}$ & 0.003 & 0.007 & 0.114 & 0.050 & 0.086 \\
\hline CLIP B32 score top $3 \%$ & $384 K$ & 0.006 & 0.014 & 0.104 & 0.055 & 0.089 \\
\hline CLIP B32 score top 10\% & $1.3 \mathrm{M}$ & 0.026 & 0.035 & 0.147 & 0.083 & 0.126 \\
\hline CLIP B32 score top $20 \%$ & $2.6 \mathrm{M}$ & 0.051 & 0.056 & 0.173 & 0.114 & 0.161 \\
\hline CLIP B32 score top $30 \%$ & $3.8 \mathrm{M}$ & 0.045 & 0.052 & 0.180 & 0.120 & 0.167 \\
\hline CLIP B32 score top $40 \%$ & $5.1 \mathrm{M}$ & 0.052 & 0.057 & 0.173 & 0.123 & 0.167 \\
\hline CLIP B32 score top $50 \%$ & $6.4 \mathrm{M}$ & 0.047 & 0.053 & 0.174 & 0.124 & 0.165 \\
\hline CLIP B32 score top 75\% & $9.6 \mathrm{M}$ & 0.033 & 0.043 & 0.161 & 0.121 & 0.151 \\
\hline CLIP B32 score top $90 \%$ & $11.5 \mathrm{M}$ & 0.028 & 0.039 & 0.140 & 0.114 & 0.136 \\
\hline CLIP B32 threshold at $0.3+$ English filter & $942 \mathrm{~K}$ & 0.022 & 0.032 & 0.138 & 0.077 & 0.122 \\
\hline CLIP B32 threshold at $0.28+$ English filter & $1.3 \mathrm{M}$ & 0.031 & 0.040 & 0.136 & 0.092 & 0.133 \\
\hline CLIP B32 threshold at 0.3 & $2.6 \mathrm{M}$ & 0.052 & 0.056 & 0.166 & 0.114 & 0.161 \\
\hline CLIP B32 score $1 \%$ to $30 \%$ & $3.7 \mathrm{M}$ & 0.053 & 0.058 & 0.185 & 0.113 & 0.170 \\
\hline CLIP B32 score $2 \%$ to $30 \%$ & $3.6 \mathrm{M}$ & 0.056 & 0.059 & 0.173 & 0.120 & 0.161 \\
\hline CLIP B32 score $5 \%$ to $30 \%$ & $3.2 \mathrm{M}$ & 0.052 & 0.055 & 0.177 & 0.115 & 0.169 \\
\hline CLIP L14 score top $1 \%$ & $128 \mathrm{~K}$ & 0.002 & 0.007 & 0.111 & 0.050 & 0.080 \\
\hline CLIP L14 score top 3\% & $386 \mathrm{~K}$ & 0.004 & 0.009 & 0.110 & 0.052 & 0.088 \\
\hline CLIP L14 score top $10 \%$ & $1.3 \mathrm{M}$ & 0.021 & 0.033 & 0.131 & 0.075 & 0.119 \\
\hline CLIP L14 score top $20 \%$ & $2.6 \mathrm{M}$ & 0.042 & 0.051 & 0.165 & 0.100 & 0.151 \\
\hline CLIP L14 score top $30 \%$ & $3.8 \mathrm{M}$ & 0.051 & 0.055 & 0.190 & 0.119 & 0.173 \\
\hline CLIP L14 score top $40 \%$ & $5.1 \mathrm{M}$ & 0.050 & 0.054 & 0.173 & 0.119 & 0.168 \\
\hline CLIP L14 score top $50 \%$ & $6.4 \mathrm{M}$ & 0.045 & 0.052 & 0.164 & 0.122 & 0.160 \\
\hline CLIP L14 score top $75 \%$ & $9.6 \mathrm{M}$ & 0.035 & 0.043 & 0.164 & 0.120 & 0.151 \\
\hline CLIP L14 score top $90 \%$ & $11.5 \mathrm{M}$ & 0.031 & 0.038 & 0.154 & 0.116 & 0.144 \\
\hline Image-based clustering (ImageNet1k) & $2.9 \mathrm{M}$ & 0.043 & 0.047 & 0.178 & 0.121 & 0.159 \\
\hline Image-based clustering (ImageNet $21 \mathrm{k}$ ) & $4.5 \mathrm{M}$ & 0.035 & 0.045 & 0.154 & 0.122 & 0.148 \\
\hline Image-based sampling, $\alpha=0$ & $12.8 \mathrm{M}$ & 0.019 & 0.030 & 0.144 & 0.095 & 0.127 \\
\hline Image-based sampling, $\alpha=0.2$ & $12.8 \mathrm{M}$ & 0.031 & 0.036 & 0.133 & 0.100 & 0.131 \\
\hline Image-based sampling, $\alpha=0.5$ & $12.8 \mathrm{M}$ & 0.032 & 0.038 & 0.129 & 0.096 & 0.125 \\
\hline Image-based sampling, $\alpha=1$ & $12.8 \mathrm{M}$ & 0.021 & 0.028 & 0.128 & 0.078 & 0.116 \\
\hline Image-based sampling, $\alpha=2$ & $12.8 \mathrm{M}$ & 0.011 & 0.017 & 0.116 & 0.065 & 0.099 \\
\hline ImageNet distance (L14, top 30\%) and English & $2.0 \mathrm{M}$ & 0.031 & 0.039 & 0.163 & 0.103 & 0.145 \\
\hline ImageNet distance (L14, top 20\%) & $2.6 \mathrm{M}$ & 0.030 & 0.035 & 0.155 & 0.102 & 0.136 \\
\hline ImageNet distance (L14, top 30\%) & $3.9 \mathrm{M}$ & 0.034 & 0.041 & 0.151 & 0.106 & 0.139 \\
\hline ImageNet distance (L14, top $40 \%)$ & $5.1 \mathrm{M}$ & 0.036 & 0.040 & 0.151 & 0.118 & 0.143 \\
\hline Text-based clustering (ImageNet $1 \mathrm{k}$ ) & $427 K$ & 0.009 & 0.016 & 0.120 & 0.056 & 0.096 \\
\hline Text-based clustering (ImageNet21k) & $3.2 \mathrm{M}$ & 0.046 & 0.052 & 0.169 & 0.125 & 0.157 \\
\hline Text-based sampling with average score, $\alpha=0$ & $12.8 \mathrm{M}$ & 0.011 & 0.020 & 0.128 & 0.079 & 0.112 \\
\hline Text-based sampling with average score, $\alpha=0.5$ & $12.8 \mathrm{M}$ & 0.023 & 0.035 & 0.127 & 0.092 & 0.128 \\
\hline Text-based sampling with average score, $\alpha=1$ & $12.8 \mathrm{M}$ & 0.040 & 0.044 & 0.163 & 0.115 & 0.155 \\
\hline Text-based sampling with average score, $\alpha=1.2$ & $12.8 \mathrm{M}$ & 0.038 & 0.045 & 0.150 & 0.112 & 0.143 \\
\hline Text-based sampling with max score, $\alpha=0$ & $12.8 \mathrm{M}$ & 0.012 & 0.020 & 0.126 & 0.074 & 0.107 \\
\hline Text-based sampling with max score, $\alpha=0.5$ & $12.8 \mathrm{M}$ & 0.025 & 0.033 & 0.134 & 0.093 & 0.129 \\
\hline Text-based sampling with max score, $\alpha=1$ & $12.8 \mathrm{M}$ & 0.040 & 0.046 & 0.159 & 0.116 & 0.150 \\
\hline Text-based sampling with max score, $\alpha=1.2$ & $12.8 \mathrm{M}$ & 0.040 & 0.050 & 0.161 & 0.113 & 0.152 \\
\hline Intersect IN1k image clustering and CLIP B32 score top 30\% & $1.4 \mathrm{M}$ & 0.049 & 0.053 & 0.150 & 0.103 & 0.148 \\
\hline Intersect IN1k image clustering and CLIP L14 score top 30\% & $1.4 \mathrm{M}$ & 0.039 & 0.045 & 0.162 & 0.094 & 0.145 \\
\hline Intersect IN21k image clustering and CLIP B32 score top 30\% & $2.1 \mathrm{M}$ & 0.052 & 0.057 & 0.179 & 0.112 & 0.167 \\
\hline Intersect IN21k image clustering and CLIP L14 score top $30 \%$ & $2.1 \mathrm{M}$ & 0.047 & 0.053 & 0.176 & 0.110 & 0.163 \\
\hline
\end{tabular}

Table 26: Baseline results for the filtering track, medium scale.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Filtering & \begin{tabular}{c} 
Training \\
dataset size
\end{tabular} & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline No filtering & $128 \mathrm{M}$ & 0.176 & 0.152 & 0.259 & 0.219 & 0.258 \\
\hline Random subset (75\%) & $96.0 \mathrm{M}$ & 0.175 & 0.154 & 0.265 & 0.219 & 0.257 \\
\hline Random subset ( $50 \%$ ) & $64.0 \mathrm{M}$ & 0.171 & 0.151 & 0.258 & 0.216 & 0.252 \\
\hline Random subset ( $25 \%$ ) & $32.0 \mathrm{M}$ & 0.155 & 0.136 & 0.246 & 0.203 & 0.240 \\
\hline Random subset (10\%) & $12.8 \mathrm{M}$ & 0.107 & 0.095 & 0.210 & 0.144 & 0.200 \\
\hline Random subset $(1 \%)$ & $1.3 \mathrm{M}$ & 0.009 & 0.017 & 0.102 & 0.065 & 0.090 \\
\hline Caption length & $87.5 \mathrm{M}$ & 0.199 & 0.172 & 0.275 & 0.236 & 0.275 \\
\hline Image size & $77.8 \mathrm{M}$ & 0.189 & 0.163 & 0.248 & 0.231 & 0.259 \\
\hline English (fasttext) & $63.0 \mathrm{M}$ & 0.214 & 0.182 & 0.290 & 0.246 & 0.285 \\
\hline English (fasttext) and caption length & $47.8 \mathrm{M}$ & 0.226 & 0.193 & 0.284 & 0.251 & 0.285 \\
\hline English (fasttext), caption length, and image size & $29.8 \mathrm{M}$ & 0.226 & 0.193 & 0.297 & 0.253 & 0.294 \\
\hline English (cld3) & $25.6 \mathrm{M}$ & 0.200 & 0.175 & 0.296 & 0.235 & 0.279 \\
\hline English (cld3) and caption length & $22.9 \mathrm{M}$ & 0.204 & 0.175 & 0.287 & 0.243 & 0.278 \\
\hline English (cld3), caption length, and image size & $14.6 \mathrm{M}$ & 0.179 & 0.159 & 0.243 & 0.216 & 0.247 \\
\hline CLIP B32 score top $1 \%$ & $1.3 \mathrm{M}$ & 0.025 & 0.037 & 0.140 & 0.076 & 0.126 \\
\hline CLIP B32 score top 3\% & $3.9 \mathrm{M}$ & 0.093 & 0.096 & 0.205 & 0.128 & 0.188 \\
\hline CLIP B32 score top $10 \%$ & $12.8 \mathrm{M}$ & 0.231 & 0.199 & 0.305 & 0.206 & 0.298 \\
\hline CLIP B32 score top $20 \%$ & $25.7 \mathrm{M}$ & 0.279 & 0.234 & 0.337 & 0.241 & 0.330 \\
\hline CLIP B32 score top $30 \%$ & $38.4 \mathrm{M}$ & 0.285 & 0.240 & 0.355 & 0.253 & 0.338 \\
\hline CLIP B32 score top $40 \%$ & $51.3 \mathrm{M}$ & 0.273 & 0.227 & 0.333 & 0.257 & 0.324 \\
\hline CLIP B32 score top $50 \%$ & $64.0 \mathrm{M}$ & 0.256 & 0.219 & 0.322 & 0.259 & 0.316 \\
\hline CLIP B32 score top $75 \%$ & $96.1 \mathrm{M}$ & 0.211 & 0.180 & 0.301 & 0.238 & 0.290 \\
\hline CLIP B32 score top $90 \%$ & $115 \mathrm{M}$ & 0.189 & 0.165 & 0.279 & 0.229 & 0.274 \\
\hline CLIP B32 threshold at $0.3+$ English filter & $9.4 \mathrm{M}$ & 0.208 & 0.184 & 0.292 & 0.210 & 0.276 \\
\hline CLIP B32 threshold at $0.28+$ English filter & $13.0 \mathrm{M}$ & 0.230 & 0.198 & 0.307 & 0.233 & 0.292 \\
\hline CLIP B32 threshold at 0.3 & $25.9 \mathrm{M}$ & 0.282 & 0.233 & 0.340 & 0.243 & 0.333 \\
\hline CLIP B32 score $1 \%$ to $30 \%$ & $37.1 \mathrm{M}$ & 0.287 & 0.238 & 0.347 & 0.253 & 0.334 \\
\hline CLIP B32 score $2 \%$ to $30 \%$ & $35.9 \mathrm{M}$ & 0.288 & 0.238 & 0.338 & 0.248 & 0.330 \\
\hline CLIP B32 score $5 \%$ to $30 \%$ & $32.0 \mathrm{M}$ & 0.281 & 0.230 & 0.352 & 0.254 & 0.339 \\
\hline CLIP L14 score top $1 \%$ & $1.3 \mathrm{M}$ & 0.014 & 0.025 & 0.136 & 0.062 & 0.109 \\
\hline CLIP L14 score top 3\% & $3.9 \mathrm{M}$ & 0.065 & 0.077 & 0.176 & 0.103 & 0.160 \\
\hline CLIP L14 score top $10 \%$ & $12.8 \mathrm{M}$ & 0.198 & 0.183 & 0.283 & 0.188 & 0.277 \\
\hline CLIP L14 score top $20 \%$ & $25.7 \mathrm{M}$ & 0.260 & 0.225 & 0.326 & 0.235 & 0.322 \\
\hline CLIP L14 score top $30 \%$ & $38.4 \mathrm{M}$ & 0.273 & 0.230 & 0.338 & 0.251 & 0.328 \\
\hline CLIP L14 score top $40 \%$ & $51.2 \mathrm{M}$ & 0.262 & 0.226 & 0.330 & 0.260 & 0.327 \\
\hline CLIP L14 score top $50 \%$ & $64.1 \mathrm{M}$ & 0.254 & 0.218 & 0.322 & 0.262 & 0.315 \\
\hline CLIP L14 score top $75 \%$ & $96.1 \mathrm{M}$ & 0.212 & 0.180 & 0.287 & 0.242 & 0.285 \\
\hline CLIP L14 score top $90 \%$ & $115 \mathrm{M}$ & 0.188 & 0.164 & 0.258 & 0.225 & 0.266 \\
\hline Image-based clustering (ImageNet1k) & $29.2 \mathrm{M}$ & 0.268 & 0.213 & 0.319 & 0.256 & 0.312 \\
\hline Image-based clustering (ImageNet $21 \mathrm{k}$ ) & $45.1 \mathrm{M}$ & 0.238 & 0.198 & 0.304 & 0.252 & 0.312 \\
\hline Image-based sampling, $\alpha=0$ & $128 \mathrm{M}$ & 0.170 & 0.150 & 0.266 & 0.209 & 0.254 \\
\hline Image-based sampling, $\alpha=0.2$ & $128 \mathrm{M}$ & 0.249 & 0.193 & 0.292 & 0.221 & 0.284 \\
\hline Image-based sampling, $\alpha=0.5$ & $128 \mathrm{M}$ & 0.269 & 0.196 & 0.301 & 0.216 & 0.284 \\
\hline Image-based sampling, $\alpha=1$ & $128 \mathrm{M}$ & 0.207 & 0.145 & 0.264 & 0.166 & 0.239 \\
\hline Image-based sampling, $\alpha=2$ & $128 \mathrm{M}$ & 0.118 & 0.082 & 0.207 & 0.110 & 0.180 \\
\hline ImageNet distance (L14, top 30\%) and English & $19.8 \mathrm{M}$ & 0.212 & 0.158 & 0.272 & 0.178 & 0.259 \\
\hline ImageNet distance (L/14, top 20\%) & $25.8 \mathrm{M}$ & 0.193 & 0.138 & 0.276 & 0.176 & 0.252 \\
\hline ImageNet distance (L/14, top $30 \%$ ) & $38.5 \mathrm{M}$ & 0.212 & 0.159 & 0.283 & 0.201 & 0.269 \\
\hline ImageNet distance (L/14, top $40 \%$ ) & $51.3 \mathrm{M}$ & 0.212 & 0.165 & 0.273 & 0.212 & 0.270 \\
\hline Text-based clustering (ImageNet $1 \mathrm{k}$ ) & $4.3 \mathrm{M}$ & 0.099 & 0.090 & 0.173 & 0.109 & 0.166 \\
\hline Text-based clustering (ImageNet $21 \mathrm{k}$ ) & $31.7 \mathrm{M}$ & 0.255 & 0.215 & 0.328 & 0.249 & 0.307 \\
\hline Text-based sampling with average score, $\alpha=0$ & $128 \mathrm{M}$ & 0.136 & 0.110 & 0.213 & 0.140 & 0.209 \\
\hline Text-based sampling with average score, $\alpha=0.5$ & $128 \mathrm{M}$ & 0.222 & 0.178 & 0.273 & 0.206 & 0.269 \\
\hline Text-based sampling with average score, $\alpha=1$ & $128 \mathrm{M}$ & 0.245 & 0.204 & 0.302 & 0.251 & 0.293 \\
\hline Text-based sampling with average score, $\alpha=1.2$ & $128 \mathrm{M}$ & 0.231 & 0.200 & 0.298 & 0.240 & 0.289 \\
\hline Text-based sampling with max score, $\alpha=0$ & $128 \mathrm{M}$ & 0.140 & 0.116 & 0.242 & 0.138 & 0.225 \\
\hline Text-based sampling with max score, $\alpha=0.5$ & $128 \mathrm{M}$ & 0.229 & 0.190 & 0.290 & 0.205 & 0.283 \\
\hline Text-based sampling with max score, $\alpha=1$ & $128 \mathrm{M}$ & 0.247 & 0.209 & 0.300 & 0.241 & 0.295 \\
\hline Text-based sampling with max score, $\alpha=1.2$ & $128 \mathrm{M}$ & 0.235 & 0.200 & 0.298 & 0.239 & 0.290 \\
\hline Intersect IN1k image clustering and CLIP B32 score top 30\% & $14.2 \mathrm{M}$ & 0.305 & 0.243 & 0.342 & 0.250 & 0.328 \\
\hline Intersect IN1k image clustering and CLIP L14 score top 30\% & $14.0 \mathrm{M}$ & 0.297 & 0.239 & 0.346 & 0.231 & 0.328 \\
\hline Intersect IN21k image clustering and CLIP B32 score top 30\% & $21.1 \mathrm{M}$ & 0.298 & 0.244 & 0.347 & 0.256 & 0.336 \\
\hline Intersect IN21k image clustering and CLIP L14 score top $30 \%$ & $20.8 \mathrm{M}$ & 0.290 & 0.241 & 0.339 & 0.244 & 0.328 \\
\hline
\end{tabular}

Table 27: Baseline results for the filtering track, large scale.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Filtering & \begin{tabular}{c} 
Training \\
dataset size
\end{tabular} & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline No filtering & $1.28 \mathrm{~B}$ & 0.459 & 0.378 & 0.426 & 0.419 & 0.437 \\
\hline Random subset (75\%) & $960 \mathrm{M}$ & 0.456 & 0.379 & 0.435 & 0.415 & 0.442 \\
\hline Random subset ( $50 \%$ ) & $640 \mathrm{M}$ & 0.453 & 0.377 & 0.427 & 0.413 & 0.433 \\
\hline Random subset (25\%) & $320 \mathrm{M}$ & 0.447 & 0.373 & 0.424 & 0.407 & 0.434 \\
\hline Random subset (10\%) & $128 \mathrm{M}$ & 0.426 & 0.350 & 0.417 & 0.396 & 0.442 \\
\hline Random subset $(1 \%)$ & $12.8 \mathrm{M}$ & 0.135 & 0.118 & 0.219 & 0.135 & 0.218 \\
\hline Caption length & $874 \mathrm{M}$ & 0.474 & 0.392 & 0.438 & 0.443 & 0.445 \\
\hline Image size & $777 \mathrm{M}$ & 0.466 & 0.375 & 0.421 & 0.438 & 0.429 \\
\hline English (fasttext) & $630 \mathrm{M}$ & 0.500 & 0.414 & 0.449 & 0.460 & 0.462 \\
\hline English (fasttext), caption length, and image size & $298 \mathrm{M}$ & 0.516 & 0.423 & 0.446 & 0.480 & 0.458 \\
\hline English (cld3) & $256 \mathrm{M}$ & 0.486 & 0.405 & 0.462 & 0.472 & 0.458 \\
\hline CLIP B32 score top $10 \%$ & $128 \mathrm{M}$ & 0.543 & 0.440 & 0.471 & 0.435 & 0.483 \\
\hline CLIP B32 score top $20 \%$ & $257 \mathrm{M}$ & 0.578 & 0.465 & 0.516 & 0.463 & 0.515 \\
\hline CLIP B32 score top $30 \%$ & $384 \mathrm{M}$ & 0.578 & 0.466 & 0.525 & 0.475 & 0.527 \\
\hline CLIP B32 score top $40 \%$ & $512 \mathrm{M}$ & 0.560 & 0.454 & 0.512 & 0.478 & 0.511 \\
\hline CLIP B32 score top $50 \%$ & $640 \mathrm{M}$ & 0.546 & 0.450 & 0.504 & 0.484 & 0.505 \\
\hline CLIP B32 threshold at $0.3+$ English filter & $94.3 \mathrm{M}$ & 0.553 & 0.447 & 0.511 & 0.482 & 0.502 \\
\hline CLIP B32 threshold at $0.28+$ English filter & $130 \mathrm{M}$ & 0.553 & 0.453 & 0.510 & 0.495 & 0.501 \\
\hline CLIP B32 threshold at 0.3 & $258 \mathrm{M}$ & 0.579 & 0.464 & 0.501 & 0.465 & 0.505 \\
\hline CLIP L14 score top $10 \%$ & $128 \mathrm{M}$ & 0.528 & 0.444 & 0.482 & 0.413 & 0.486 \\
\hline CLIP L14 score top $20 \%$ & $257 \mathrm{M}$ & 0.570 & 0.466 & 0.524 & 0.455 & 0.521 \\
\hline CLIP L14 score top $30 \%$ & $384 \mathrm{M}$ & 0.578 & 0.474 & 0.538 & 0.466 & 0.529 \\
\hline CLIP L14 score top $40 \%$ & $512 \mathrm{M}$ & 0.564 & 0.462 & 0.533 & 0.468 & 0.529 \\
\hline CLIP L14 score top $50 \%$ & $641 \mathrm{M}$ & 0.548 & 0.455 & 0.539 & 0.469 & 0.528 \\
\hline Image-based clustering (ImageNet1k) & $294 \mathrm{M}$ & 0.572 & 0.454 & 0.483 & 0.481 & 0.481 \\
\hline Image-based clustering (ImageNet $21 \mathrm{k}$ ) & $450 \mathrm{M}$ & 0.527 & 0.433 & 0.468 & 0.463 & 0.471 \\
\hline Text-based clustering (ImageNet1k) & $42.7 \mathrm{M}$ & 0.419 & 0.355 & 0.340 & 0.309 & 0.361 \\
\hline Text-based clustering (ImageNet21k) & $317 \mathrm{M}$ & 0.561 & 0.465 & 0.465 & 0.479 & 0.476 \\
\hline Intersect IN1k image clustering and CLIP B32 score top $30 \%$ & $143 \mathrm{M}$ & 0.632 & 0.498 & 0.525 & 0.504 & 0.528 \\
\hline Intersect IN1k image clustering and CLIP L14 score top $30 \%$ & $140 \mathrm{M}$ & 0.631 & 0.508 & 0.546 & 0.498 & 0.537 \\
\hline Intersect IN21k image clustering and CLIP B32 score top 30\% & $211 \mathrm{M}$ & 0.605 & 0.481 & 0.531 & 0.494 & 0.519 \\
\hline Intersect IN21k image clustering and CLIP L14 score top $30 \%$ & $208 \mathrm{M}$ & 0.506 & 0.416 & 0.466 & 0.424 & 0.471 \\
\hline
\end{tabular}

Table 28: Baseline results for the filtering track, xlarge scale.

\begin{tabular}{|c|c|c|c|c|c|c|}
\hline Filtering & \begin{tabular}{c} 
Training \\
dataset size
\end{tabular} & ImageNet & \begin{tabular}{l} 
ImageNet \\
dist. shifts
\end{tabular} & VTAB & Retrieval & \begin{tabular}{c} 
Average over \\
38 datasets
\end{tabular} \\
\hline No filtering & $12.8 \mathrm{~B}$ & 0.723 & 0.612 & 0.611 & 0.569 & 0.621 \\
\hline CLIP B32 score top $30 \%$ & $3.84 B$ & 0.764 & 0.640 & 0.628 & 0.599 & 0.638 \\
\hline CLIP B32 threshold at $0.28+$ English filter & $1.3 \mathrm{~B}$ & 0.755 & 0.637 & 0.624 & 0.620 & 0.636 \\
\hline CLIP L14 score top $20 \%$ & $2.56 \mathrm{~B}$ & 0.761 & 0.649 & 0.630 & 0.575 & 0.636 \\
\hline CLIP L14 score top $25 \%$ & $3.2 \mathrm{~B}$ & 0.768 & 0.656 & 0.621 & 0.585 & 0.637 \\
\hline CLIP L14 score top $30 \%$ & $3.84 \mathrm{~B}$ & 0.764 & 0.655 & 0.643 & 0.588 & 0.650 \\
\hline Intersect IN1k image clustering and CLIP L14 score top 30\% & $1.38 \mathrm{~B}$ & 0.792 & 0.679 & 0.652 & 0.608 & 0.663 \\
\hline
\end{tabular}

\section*{S Datasheet}

\section*{S. 1 Motivation}

Q1 For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.
- The purpose of DataComp and the associated CommonPool dataset is to enable study of what makes a strong image-text dataset, which supports a broad range of applications. Prior work mainly focuses on data curation in the context of supervised datasets and smaller scales. For a fuller treatment see Section 2. In our initial release of DataComp we focus on 38 downstream image classification and image retrieval tasks. For details see Section 3.5 and Appendix O.

Q2 Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
- DataComp and CommonPool were created by a group of researchers with the following affiliations, listed in alphabetical order: Allen Institute for Artificial Intelligence (AI2), Apple, Columbia University, Google Research, Graz University of Technology, Hebrew University, Juelich Supercomputing Center, LAION, Research Center Juelich, StabilityAI, Tel Aviv University, University of Illinois Urbana-Champaign, University of Texas at Austin, University of Washington.

Q3 Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.
- Compute for this research was generously provided by StabilityAI. For more specific acknowledgments, see the acknowledgment section at the end of the main paper.

Q4 Any other comments?
- We hope that CommonPool will help to facilitate data-centric questions in ML and AI towards the next generation of web-scale datasets, that 1) yield higher accuracy models and 2) models that are safer and more equitable.

\section*{S. 2 Composition}

Q5 What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.
- Each instance is a pair of url and corresponding image alt-text. The url points to an image that a user can then try to download. Each sample is also tagged with metadata, discussed in Q25.

Q6 How many instances are there in total (of each type, if appropriate)?
- There are 12.8B instances in CommonPool. For breakdowns and statistics see Appendix I.

Q7 Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).
- We find $\sim 88 \mathrm{~B}$ possible samples in common crawl. These samples are globally shuffled to ensure i.i.d. sampling for all sampling based parts of the downstream pipeline. Of these samples we attempt to download $\sim 40$ B samples. Due to various download issues, such as dead links and throttling, we are able to successfully download $\sim 16.8 \mathrm{~B}$ samples. After NSFW filtering and evaluation set deduplication we end up with $\sim 13.1 \mathrm{~B}$ viable samples, from which we randomly sample 12.8B for COMmONPool. For a complete treatment and visualization of our data processing funnel, see Appendix H. For each sample we also release metadata shown in Table 8.

Q8 What data does each instance consist of? "Raw" data (e.g., unprocessed text or images) or features? In either case, please provide a description.
- Each sample contains an image url for download and an associated alt-text caption. Additionally, each sample contains metadata fields shown in Table 8 (e.g., image aspect ratio and CLIP features).

Q9 Is there a label or target associated with each instance? If so, please provide a description.
- We do not provide any category labels; however, the text associated with each image can be considered a soft, noisy label for each sample. Such labels are common in modern image-text training paradigms (e.g., image-text representation alignment, image captioning objectives, text-conditional image generation objectives, etc.).

Q10 Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.
- No, each sample is an image-text pair.

Q11 Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.
- No, the dataset is released as it is with no explicit attempt to establish relationships between instances.

Q12 Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.
- No. The test tasks are existing image classification tasks. We run a deduplication model to try to prevent test set contamination in CommonPool.

Q13 Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.
- CommonPool is sourced from Common Crawl, which can be thought of as a snapshot of the internet. Hence, there can be considerable noise (e.g., alt-text being unrelated to its associated image), duplicate data, etc.

Q14 Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.
- The data is not self-contained and rather links other external resources on the internet. Links point to resources distributed across the internet. There is no guarantee that the resources will exist in perpetuity or that that the resources will not change. To mitigate against data poisoning in future CommONPool downloads, we release SHA256 hashes of images. Due to the size of the dataset, it is not possible to provide it in an archival form.

Q15 Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)? If so, please provide a description.
- The dataset is comprised of data that was readily available on the public internet at the time of our download. However, it is possible that the dataset contains confidential information (e.g., private data that is hosted publicly for nefarious reasons or out of ignorance of said data being confidential).

Q16 Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
- Considering the plurality of people and their backgrounds across the world, it is highly likely that there is content in CommonPool that may upset people. Common Crawl scrapes the internet, which has pornographic, hateful, racist, sexist, and otherwise abhorrent and toxic material While we attempt to do thorough NSFW filtering, these methods are not $100 \%$ accurate. At the 12.8B scale at which we operate, it is highly likely that there is still toxic content in the dataset. We consider the dataset as a research artifact and hope future work will look critically at COMMONPOoL in the hopes of developing even better safety filters.

Q17 Does the dataset relate to people? If not, you may skip the remaining questions in this section.
- People may appear in the dataset; however, in an effort to preserve privacy, our downloading tooling automatically blurs all detected faces in CommonPooL images.

Q18 Does the dataset identify any subpopulations (e.g., by age, gender)?
- While CommonPool does not explicitly identify subpopulations in its metadata, it is plausible to extract such information for some images using the corresponding textual caption.

Q19 Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.
- We conjecture that even with our face blurring procedure, it may still be possible to identify individuals. Face blurring relies of a face detection model, which could fail (See Appendix $\mathrm{G}$ for experimental validation of the employed detector). It is also possible to identify certain celebrities or athletes, who may wear distinctive clothing that is associated with them. It is also likely that names are contained in textual captions, though it is not guaranteed that these names correspond to people in images due to the inherent noisiness of internet captions.

Q20 Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.
- Yes. CommonPool is created using images and corresponding alt-text that are available on the public internet. Given the 12.8B scale of CommonPool, it is highly likely that there is sensitive data in the dataset. To mitigate against making sensitive content more accessible, we 1) run NSFW image filtering and 2) NSFW text filtering when generating CommOnPool, discarding all samples that are flagged. Additionally we 3) provide automatic face blurring in our ComMONPOol download scripts to blur all detected faces.

Q21 Any other comments?
- CommonPool is a research artifact, and we hope it will be useful for those studying how to make internet-scale datasets safer.

\section*{S. 3 Collection Process}

Q22 How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.
- Data is directly downloaded from the public internet.

Q23 What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?
- We iterate on the LAION-5B data collection process, making an effort to emphasize safety. We ran python based processing scripts to parse Common Crawl dumps, download images, filter our NSFW content, deduplicate samples against downstream tests sets, blur faces, and compute CLIP features. We ran processes on 100s of AWS CPU nodes for Common Crawl parsing and data download. Other steps were run on one of StabilityAI's GPU cluster. For software links see Q37. For software validation related to NSFW content filtering and face blurring see Appendices E and G respectively. In brief, for NSFW image filtering, we validate against commercial APIs and on the NSFW test set introduced in LAION-5B. For face detection (used for face blurring), we evaluate against commercial APIs. We find strong performance for both modules.

Q24 If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
- See Q7.

Q25 Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
- The researching authors were involved in the data collection as an open source effort. No researchers were compensated specifically for their involvement in this project.

Q26 Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.
- Data was downloaded between December 2022 and March 2023. The urls are collected from Common Crawl dumps between 2014 and 2022. Common Crawl dumps may include urls from the early days of the internet. Hence, the download/collection timeframe does not match the creation timeframe. Additionally, future users of CommonPool and its subsets will have to download data themselves using our tooling.

Q27 Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.
- Our dataset collection process iterates on the LAION-5B process, which found IRB review was not necessary as they "do not intervene with the people depicted in the data as well as the data being public." [129]. Additionally, the NeurIPS ethics review found no serious ethical issues with LAION-5B. We take even more stringent safety measures than the original LAION-5B dataset, in that we filter out data that is flagged as NSFW by our detection pipeline and blur detected faces in CommonPool, automatically in our released download tooling. All this being said, a formal ethics review has not been conducted to date.

Q28 Does the dataset relate to people? If not, you may skip the remaining questions in this section.
- Yes. People may appear in the dataset. Detected faces are blurred when downloading CoMMONPoOL with our tooling.

Q29 Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
- We collect data from websites across the internet.

Q30 Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.
- Individuals were not notified about the data collection.

Q31 Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.
- Following our usage of Common Crawl and https://github.com/rom1504/img2dataset for download images, we respect robots.txt files, which specify parts of websites that a crawler may access. It is, however, possible that images of people, medical images, etc. were uploaded to the internet without a person's consent. To mitigate against such safety concerns we make an effort to do rigorous NSFW filtering and blur all detected faces automatically in our download tooling.

Q32 If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).
- In conjunction with LAION, we use https://laion.ai/dataset-requests/ to monitor user takedown requests. We will also make an effort to provide a user with the url at which their sensitive content is hosted-if they do not have this information already-, so they can take further action as they see fit (e.g., contacting the host to request that the content is taken down from the internet).

Q33 Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.
- We conduct a fairness evaluation on models trained on CommonPool and its derivative. See Appendix Q for details. Birhane et al. [15] conduct an extensive study in the context of LAION$400 \mathrm{M}$, which is an image-text dataset also sourced from Common Crawl, finding a plethora of dangerous and unsafe content. Our dataset differs from LAION-400M in that we conduct NSFW preprocessing and face blurring for detected faces. CommonPoOL only contains samples that pass our NSFW safety checks and our download tooling automatically blurs detected faces. However, since CommonPool is created from the internet, it is still likely that it contains some harmful data.

Q34 Any other comments?
- We hope that future work will use CommonPool to study how to construct safer, web-scale datasets.

\section*{S. 4 Preprocessing, Cleaning, and/or Labeling}

Q35 Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.
- Yes. See Q7. For more details see Appendix H.

Q36 Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the "raw" data.
- Raw data is not available or distributed due to safety considerations. We distribute only urls that are in the dataset on HuggingFace-and not urls of images our preprocessing flagged as NSFW.

Q37 Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.
- We use the following, open-source software to aid in data processing:
- Apache Spark: https://spark.apache.org
- Ray: https://www.ray.io
- img2dataset: https://github.com/rom1504/img2dataset
- OpenAI CLIP: https://github.com/openai/CLIP
- Near dedulicate detector: https://github.com/lyakaap/ ISC21-Descriptor-Track-1st
- Face detector: https://github.com/deepinsight/insightface
- Detoxify, for detecting toxic language: https://github.com/unitaryai/detoxify
- A modified version of the following NSFW image detector: https://github.com/ LAION-AI/CLIP-based-NSFW-Detector. Specifically, we use the dataset used to train this model to train our own 4-layer MLP classifier.

Q38 Any other comments?
- CommonPool and DataComp would not be possible without tools developed by the opensource community.

\section*{S. 5 Uses}

Q39 Has the dataset been used for any tasks already? If so, please provide a description.
- The full dataset (and subsets) have been used to train several CLIP models at various scales and compute budgets as presented in our main paper. We evaluate these models zero-shot on 38 downstream image classification and retrieval tasks. See Section 3.5 and Appendix O for more details.

Q40 Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.
- No. However, there is a leaderboard associated with DataComp. Interested parties can investigate the submissions and further study publications that make use of our data. See: https://www.datacomp.ai/leaderboard.html.

\section*{Q41 What (other) tasks could the dataset be used for?}
- The dataset could also be used for training image captioning models and language-conditional image generation models. Note: generative image models trained on COMMONPooL are not expected to generate recognizable human faces as our download tooling automatically blurs detected faces. CommonPool could be used for sociological studies, for example, examining societal biases or to better understand what is on the public internet.

Q42 Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?
- CommonPool and its derivatives are not intended for production ready products, including but not limited to those related to race, gender identity or expression, ethnicity, sexual orientation, age, socioeconomic status, disability, religion, national origin or creed. COMMONPOOL is not suitable for any software that makes decisions involving people. CommonPool is collected from the internet and hence reflects many of the biases, unfairness, and stereotypes currently existing in our societies. CommONPool is intended as a research artifact to study multimodal dataset curation and the effect of data curation strategies on downstream models.

Q43 Are there tasks for which the dataset should not be used? If so, please provide a description.
- CommonPool in its current form or the subsets presented in this paper should not be used in software that makes decisions related to people. The known biases (Appendix $\mathrm{Q}$ ) make deploying software, especially widely decimated production-level products, built on CommonPoOL incredibly irresponsible. CommonPool is designed as a research artifact for academic exploration. We also do not condone the use of CoMmonPool in surveillance or military applications.

Q44 Any other comments?
- Our goal with CommonPool and DataComp was to put a benchmark in place so the community can start measuring dataset progress along many different axes (e.g., model performance on diverse tasks). We believe this is crucial to develop more performant and safer datasets.

\section*{S. 6 Distribution}

Q45 Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.
- Yes. We use HuggingFace datasets for public release.

Q46 How will the dataset be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?
- The dataset will be distributed via HuggingFace datasets at https://huggingface.co/ datasets/mlfoundations/datacomp_pools/tree/main

Q47 When will the dataset be distributed?
- DataComp will be available starting May 2023.

Q48 Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.
- We distribute the url-text sample and metadata under a standard CC-BY-4.0 licence.

Q49 Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.
- We do not copyright samples in the dataset.

Q50 Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.
- The dataset is provided as an index of url-text pairs.

Q51 Any other comments?
- We provide several subsets of CommonPool (between $12.8 \mathrm{M}$ samples and the full dataset of 12.8B samples). Hence, it is possible to download and experiment with subset of the data.

\section*{S. 7 Maintenance}

Q52 Who will be supporting/hosting/maintaining the dataset?
- HuggingFace currently hosts the url-text pairs and metadata. The DataComp team will be responsible for maintaining the dataset.

Q53 How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
- We can be contacted at contact@datacomp.ai.

Q54 Is there an erratum? If so, please provide a link or other access point.
- Currently there are no errata. If issues are discovered, we will communicate with the public via our website https://datacomp.ai.

Q55 Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?
- At the present time there is no intention to update CommonPool for scientific reasons. However, we will respond to user takedown requests (see Q56). CommonPool is inherently noisy and the purpose of releasing it is to encourage researchers in the community to study dataset cleaning in the context of image-text samples.

Q56 If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.
- We will use the following website, https://laion.ai/dataset-requests, for user takedown requests, where "Sample ID" is the sample uid.

Q57 Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.
- This is the first version of DataComp and the associated CommonPool dataset. We do not intend to maintain deprecated version of CommonPool. We will communicate deprication notices through our website: https://datacomp.ai.

Q58 If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.
- All alterations to the dataset will be handled on a case-by-case basis.

Q59 Any other comments?
- We encourage community members to contact us at contact@datacomp.ai with inquiries related to dataset maintainence.```


[^0]:    ${ }^{*}$ Equal contribution, randomly ordered. Correspondence to contact@datacomp.ai. ${ }^{1}$ University of Washington ${ }^{2}$ Columbia University ${ }^{3} \mathrm{Tel}$ Aviv University ${ }^{4}$ Apple ${ }^{5}$ UT Austin ${ }^{6}$ LAION ${ }^{7}$ AI2 ${ }^{8}$ Juelich Supercomputing Center, Research Center Juelich ${ }^{9}$ University of Illinois Urbana-Champaign ${ }^{10}$ Graz University of Technology ${ }^{11}$ Hebrew University ${ }^{12}$ Google Research ${ }^{13}$ Snorkel AI

[^1]:    ${ }^{1}$ More precisely, given a batch of data $\left\{\left(x_{1}, y_{1}\right), \ldots,\left(x_{B}, y_{B}\right)\right\}$ with images $x$ and captions $y$, we train the image encoder $g$ and text encoder $v$ with the loss $\ell=\frac{1}{2} \sum_{i=1}^{B} \frac{\sigma_{i i}}{\sum_{j=1}^{B} \sigma_{i j}}+\frac{1}{2} \sum_{i=1}^{B} \frac{\sigma_{i i}}{\sum_{j=1}^{B} \sigma_{j i}}$, where $\sigma_{i j}=\exp \left\langle g\left(x_{i}\right), h\left(y_{j}\right)\right\rangle$. We also use a learnable temperature parameter as in Radford et al. [111].

[^2]:    ${ }^{2}$ Cherti et al. [28] also observe that models rank differently on classification and retrieval tasks.

</end of paper 2>


<paper 3>
# Data Filtering Networks 

Alex Fang ${ }^{* 1,2}$ Albin Madappally Jose ${ }^{1}$ Amit Jain ${ }^{1}$<br>Ludwig Schmidt ${ }^{2}$ Alexander Toshev ${ }^{1}$ Vaishaal Shankar ${ }^{1}$<br>${ }^{1}$ Apple, ${ }^{2}$ University of Washington


#### Abstract

Large training sets have become a cornerstone of machine learning and are the foundation for recent advances in language modeling and multimodal learning. While data curation for pre-training is often still ad-hoc, one common paradigm is to first collect a massive pool of data from the Web and then filter this candidate pool down to an actual training set via various heuristics. In this work, we study the problem of learning a data filtering network (DFN) for this second step of filtering a large uncurated dataset. Our key finding is that the quality of a network for filtering is distinct from its performance on downstream tasks: for instance, a model that performs well on ImageNet can yield worse training sets than a model with low ImageNet accuracy that is trained on a small amount of high-quality data. Based on our insights, we construct new data filtering networks that induce state-of-the-art image-text datasets. Specifically, our best performing dataset DFN-5B enables us to train state-of-the-art CLIP models for their compute budgets: among other improvements on a variety of tasks, a ViT-H trained on our dataset achieves $84.4 \%$ zero-shot transfer accuracy on ImageNet, outperforming models trained on other datasets such as LAION-2B, DataComp-1B, or OpenAI's WIT. In order to facilitate further research in dataset design, we also release a new 2 billion example dataset DFN-2B and show that high performance data filtering networks can be trained from scratch using only publicly available data.


## 1 Introduction

Carefully curated datasets have driven progress in machine learning for decades, from early pattern recognition experiments in Bell Labs to recent developments like GPT-4, Stable Diffusion, and CLIP (Highleyman \& Kamentsky, 1959; LeCun et al., 1989, 1998; Deng et al., 2009; Krizhevsky et al., 2009, 2012; Radford et al., 2019, 2021, 2022; OpenAI, 2023). Despite their crucial role, datasets themselves are rarely the subject of active research (Sambasivan et al., 2021).

Current approaches to improving performance on machine learning tasks have focused on scaling model capacity or training data volume. While scaling laws (Hestness et al., 2017; Kaplan et al. 2020; Aghajanyan et al., 2023 Cherti et al. 2023) have elucidated the relationship between model size, data size, and performance, little formal guidance exists on how to scale these quantities. On the model side, experimentation is straightforward - with enough compute, permutations of width, depth, normalization and training hyperparameters can be rigorously evaluated, leading to consistent modeling improvements over the years (Touvron et al., 2023a b; Elsen et al., 2023).[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-02.jpg?height=634&width=1616&top_left_y=255&top_left_x=252)

Figure 1: Compute scaling behavior of training CLIP models on various datasets. DFN-2B, the subset of CommonPool (DataComp-12.8B) chosen by our best performing data filtering networks, out-performs all other datasets including OpenAI's WIT and the previous state-of-the-art CLIP training dataset DataComp-1B. Our ViT-L outperforms a ViT-G trained on LAION with $18 \times$ more compute. Similarly, our ViT-B/16 trained on our dataset outperforms OpenAI's ViT-L/14 trained with $4 \times$ more compute. Our ViT-H/14 achieves $84.4 \%$ on ImageNet, out-performing any model in its compute class. All DFN-trained models were trained on DFN-2B, except for the ViT-H which was trained on DFN-5B. Both datasets were induced by the same DFN. We note the cost of training DFN was omitted from this plot, which corresponds to less than $\frac{1}{50}$ th of total CLIP training cost.

The dataset side is unfortunately murkier. Most large-scale training sets are not released, leaving the community to attempt open reproductions (Schuhmann et al., 2021, 2022; Byeon et al., 2022; Gao et al., 2020); however, these are often one-off efforts without the iterative refinement that models enjoy. Recent efforts like DataPerf, DataComp and MetaCLIP Mazumder et al., 2022; Gadre et al., 2023; Xu et al. 2023) help bridge the gap by providing consistent dataset evaluation and reproduction frameworks.

We argue dataset design can leverage the same tools as model design. Almost all large-scale dataset construction can be broken down into two phases: uncurated data collection and dataset filtering. We focus our work on the latter, with the assumption that a large uncurated dataset exists. We show data filtering networks (DFNs) - neural networks designed to filter data - can induce massive, high-quality pre-training datasets. Unlike previous techniques relying on domain-specific heuristics, DFNs paired with a large unfiltered image-text pool produce billion-scale state-of-the-art datasets algorithmically. We demonstrate DFNs can be efficiently trained from scratch and improved with the same techniques as standard ML models.

The contributions of this work are as follows. First, we characterize the properties of data filtering networks that lead to high-quality datasets. We ablate properties of data filtering networks from supervision signal to training data quality. We find that a small contrastive image-text model trained on only high-quality data is sufficient to construct state-of-the-art datasets.

Second, we use these properties to train DFNs and construct datasets that induce Contrastive

Image-Text Pre-trained (CLIP) models that achieve high accuracy and present better compute accuracy tradeoff than any existing dataset in the literature as show in Figure 1. In particular we train a ViT-L/14 or 12.8B examples seen on our DFN induced dataset DFN-2B to 81.4 ImageNet zero-shot transfer accuracy, outperforming the previous best ViT-L trained on DataComp-1B by over 2 percentage points. We further train a ViT-H/14 for 39B samples seen on a larger DFN induced dataset DFN-5B to 84.4ImageNet zero-shot transfer accuracy. We show that models trained on these datasets show consistent improvements on many tasks, including zero-shot classification, retrieval, and visual question answering, and maintain the favorable robustness properties of CLIP models.

Lastly, the above insights can be used as a recipe to construct high-quality datasets from scratch by using only public data 1 thus making strides towards democratization of large high-quality datasets. In addition, we release DFN-2B for the community to enable research on large image-text models.

## 2 Background and Related Work

### 2.1 Contrastive Image Language Pre-training (CLIP)

CLIP has altered the use of cheaply available image-alt-text datasets by demonstrating the practicality of large-scale training on web-scraped image-text pairs to build state-of-the-art image representations. CLIP consists of separate vision and text encoders, and uses contrastive loss during training to push the representations of related images and text pairs together, and unrelated pairs apart. Crucial to this process is a large dataset of aligned image-text pairs - images paired with semantically relevant text. The release of CLIP was followed by several other image-text models such as ALIGN, BASIC, LiT and Open-CLIP all of which we will refer to in this work as CLIP models (Jia et al., 2021; Pham et al., 2023; Zhai et al. 2022b, Ilharco et al., 2021). CLIP models generally come in 3 canonical sizes of vision transformer: ViT-B/32, ViT-B/16 and ViT-L/14; since then, the open source community has extended these to 3 larger variants ViT-H/14, ViT-g/14 and ViT-G/14 (Dosovitskiy et al., 2020; Zhai et al. 2022a). Generally the larger models exhibit better zero-shot generalization and transfer properties. CLIP models have been trained on a variety of datasets from OpenAI's WiT, Google's WebLI and JFT-3B, LAION, COYO and DataComp-1B.

Prior work has also studied how to fine-tune CLIP models to improve performance in targeted directions. CLIP models can be fine-tuned on image classification tasks by using templates to transform labels to text (Fang et al., 2022; Goyal et al., 2022). Additionally, practitioners often use weight ensembling to preserve robustness properties of the pre-trained model while reaping the benefits of fine-tuning (Wortsman et al., 2022). We take advantage of these techniques in order to improve the filtering models we train in this work.

### 2.2 Dataset Construction

Prior to CLIP, datasets most commonly used in computer vision were supervised with human labels (Deng et al., 2009; Krizhevsky et al., 2009). Though these older dataset construction pipelines were quite intricate and did not scale beyond a few million examples, they share some similarity with[^1]current constructions. Classical datasets such as ImageNet and CIFAR started with a large roughly curated pool of images paired with metadata, and used humans to either label or filter the data.

Modern dataset pipelines have a similar procedure but at a much higher scale. The initial pool of images can contain up to 100 billion images, and the dataset filtering is purely automated, often with a set of rules and heuristic filtering stages (Jia et al., 2021). Past work in natural language processing has used binary filters as an initial step to remove low quality documents (Wenzek et al. 2019; Brown et al., 2020), but contain multiple components to their filtering pipelines.

One of the first publicly available web-scale image-text datasets is LAION. LAION-400M and LAION-2B were constructed by collecting image-text pairs from Common Crawl, filtering by English, and then keeping pairs whose image and text are well aligned. This alignment is performed using a procedure known as CLIP filtering. The procedure leverages an existing image-text model (in LAION's case OpenAI CLIP ViT-B/32), and removes samples whose cosine similarity between image and text are below some threshold. We show pseudocode of the basic CLIP filtering operation below.

```
def clip_filter(image, text, threshold=0.3):
    # compute image and text representations
    image_features = clip.encode_image(image_input)
    text_features = clip.encode_text(text_input)
    # compute alignment
    dot_product = image_features.T @ text_features
    norm_a = image_features.norm()
    norm_b = text_features.norm()
    similarity = dot_product / (norm_a * norm_b)
    # filter by alignment
    return similarity > threshold
```

While CLIP filtering is convenient it is dependent on a existing trained CLIP model, and perhaps limited on the top-line performance of any model trained using it as a filter. For example, despite LAION-2B being five times larger than OpenAI's dataset, models trained on it could only match OpenAI's ImageNet zero-shot performance with a significantly larger compute budget.

To better facilitate the study of image-text datasets, researchers created the DataComp benchmark (Gadre et al. 2023). The benchmark provides 12.8 billion image-text pairs from Common Crawl so that researchers can study the effect of various data filtering techniques. DataComp fixes the computational budget used to train the resulting models, fixing the compute budget of the largest scale to match the cost of training OpenAI's ViT-L/14 CLIP model. These models are then evaluated on a suite of 38 downstream tasks, which includes ImageNet and distribution shifts, VTAB, and retrieval tasks. We use this benchmark as our primary method of evaluating the datasets created by our data filtering networks.

The authors of DataComp also released a baseline dataset, DataComp-1B (DC-1B) that improved upon LAION-5B, by combining CLIP filtering with an ImageNet based clustering approach to improve dataset quality on a variety of benchmarks. However this dataset still relies on the OpenAI CLIP model for CLIP filtering and imposes a costly ImageNet specific clustering step in the pipeline.

Recent work (Xu et al., 2023) has demystified the CLIP dataset construction process and demonstracted high quality dataset construction is possible by simple keyword based sampling and global balancing. While their work does create competitive datasets, the reliance on sampling heuristics

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-05.jpg?height=418&width=1621&top_left_y=298&top_left_x=249)

Figure 2: A high level overview of our pipeline for constructing datasets using DFNs

from the original CLIP paper (Radford et al. 2021) allows for accurate dataset reproduction, our work focuses on improving model performance using dataset construction.

## 3 Data Filtering Networks

The core object we study in this work is a data filtering network (DFN). In this section we define DFNs and introduce their evaluation setup.

### 3.1 Definitions

Since our ultimate goal is to build functions that filter potentially trillions of examples efficiently, we restrict the scope of our study to DFNs that are only applied pointwise to elements of a larger data pool. Thus, processing a data pool with a DFN, defined in pseudocode as follows

```
def apply_dfn(dfn, data_pool):
    return [x for x in data_pool if dfn(x)]
```

lends itself to parallelization and thus efficient application. For a given DFN and pool, we refer to the data pool we train the DFN on as a filter dataset. Furthermore, we refer to the dataset constructed by filtering the pool with the DFN the induced dataset. We then refer to a model trained (only) on that dataset the induced model.

As introduced in Section 2.2 a common choice for a DFN is a CLIP trained image-text model. Thus, a DFN can not only be used to induce a dataset but also be applied to common evaluation problems such as zero-shot ImageNet classification. Inversely, a CLIP model can be both used for general recognition as well as as a DFN. When we use a CLIP model as a DFN, we define its filtering performance as the performance of the induced model, as evaluated on standard benchmarks, e.g. ImageNet top-1.

### 3.2 Data Filtering Networks Evaluation Setup

With these definitions in place, we now address how we evaluate DFNs. In our context, the quality of a DFN is determined by the strength of models it can induce. We build on the evaluation framework proposed by DataComp (Gadre et al., 2023). DataComp provides a multi-scale evaluation

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-06.jpg?height=704&width=1309&top_left_y=247&top_left_x=403)

Figure 3: Filtering strength is uncorrelated with image task performance. The models are trained using CLIP, and the number of samples seen and the training data are displayed on the right hand side. Filtering performance is measured by filtering on DataComp medium.

framework for datasets by measuring CLIP model zero-shot performance. It provides 4 nested unfiltered image-text pair pools of increasing size. In this work, we use the medium (128M datapoints), large (1.28B datapoints) and xlarge(12.8B datapoints) pools. We also follow the DataComp guidelines of model hyperparameters for each of these pools, which are ViT-B/32 for medium, ViT-B/16 for large and ViT-L/14 for XL. Exact hyperparameters can be found in Table 7. We additionally expand our DFN to a larger pool of 42B images by combining 30B non-DataComp web-scraped images with the DataComp XL pool. We denote the dataset induced using this pool and our DFN as DFN-5B, which we use to train a ViT-H/14 model.

For evaluation we use 38 zero-shot classification and retrieval tasks in the DataComp benchmark. We denote the average performance on these benchmarks simply as "Average" performance, but we also track various subsets: ImageNet performance (IN), ImageNet distribution shift performance (IN shifts), Visual Task Adapation Benchmark (VTAB), Retrieval performance (COCO, Flickr, WinoGAViL).

Our actual training runs on both Nvidia A100s and TPU v4s. We use OpenClip and AXlearn to train our CLIP models on GPUs and TPUs respectively (Ilharco et al., 2021; Apple, 2023) .

### 3.3 Understanding Data Filtering Networks

As open source CLIP-models improve on standard vision metrics such as ImageNet, the question arises whether we can replace the OpenAI CLIP model used in the dataset construction process with one of these better models. We can even hope of recursively applying this process to continuously train better models that can be used as better filtering models that once again yield even better models. Unfortunately, this does not seem to be true. Figure 3 shows that ImageNet performance of CLIP models is not correlated with filtering performance. To measure filtering performance, we create a dataset by using the CLIP model to apply CLIP filtering on DataComp's medium raw pool, and measure ImageNet performance of models trained on the induced dataset. It is especially

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-07.jpg?height=607&width=1290&top_left_y=252&top_left_x=407)

Figure 4: Data quality determines the filtering performance of models. We create these filter training datasets of various quality by having a set pool size of 10 million samples, and interpolating between CC-12M (high quality) and CommonPool (low quality). We then train models induced by the DFN filtering DataComp medium.

striking that a model with $30 \%$ less ImageNet performance than OpenAI's CLIP models can be as good when used as a filtering model.

We find that data quality is key to training good filtering models. To demonstrate this, we start with a high-quality pool of 10 million samples from Conceptual 12M (CC12M), and gradually replace it with unfiltered data from Common Crawl until this pool only contains Common Crawl. We train DFNs on these data mixes, and use these DFNs to CLIP filter a separate pool of 128 million Common Crawl samples from DataComp's medium scale. In Figure 4, we measure the ImageNet performance of both the DFNs and the induced models trained on datasets generated by each of the DFNs. While the ImageNet performance of the DFNs degrade steadily as they are trained on larger fractions of unfiltered data, their performance as filtering networks decreases immediately when the high-quality pool is "poisoned" with even a small portion of unfiltered data. Once the filtering training pool is poisoned, the dataset induced by the DFN is only slightly better than unfiltered data.

Table 1: Filtering Performance of various filtering models, after filtering DataComp medium scale (ViT-B/32, 128M samples seen). We present results on ImageNet top-1 as well as "Average" set of tasks (see Sec. 3.2 for details.)

| DFN Type | Filter Dataset | ImageNet | Average |
| :--- | :--- | :---: | :---: |
| No Filter Baseline | None | 0.176 | 0.258 |
| ResNet-34 Image Binary Filter | ImageNet | 0.242 | 0.292 |
| OpenAI ViT-B/32 Image Binary Filter | ImageNet | 0.266 | 0.295 |
| ResNet-34 Image Binary Filter | CC12M | 0.203 | 0.257 |
| OpenAI ViT-B/32 Image Binary Filter | CC12M | 0.218 | 0.276 |
| M3AE ViT-B/16 | CC12M | 0.237 | 0.297 |
| CLIP ViT-B/32 | CC12M | 0.289 | 0.335 |

Next, we explore using filtering models beyond CLIP models. While DFNs can use any model that can be reduced to a binary function, intuitively it makes sense to use CLIP models. By filtering with a similarity score between the image and text, we encourage keeping samples where the image and text are aligned.

In order to verify this intuition we consider a few other options to produce a DFN. One is to train a binary classifier that can distinguish between ImageNet or CC12M data as positives and Common Crawl as negatives. We consider both ResNet (He et al. 2016) as well as frozen OpenAI CLIP embeddings for this filter. Another option is to use M3AE (Geng et al., 2022) trained on CC12M as a DFN that takes into account both images and text. We can use reconstruction loss as the filtering criterion, as it is a reasonable proxy for how similar samples are to the high-quality data used to train the filtering model.

The filtering performance of all these options, including CLIP models, are summarized in Table 1 , where the CLIP model outperform the other backbones. A key difference between the binary classifier and CLIP filters is that the binary filter makes an explicit assumption on what qualifies as a good distribution, while CLIP filters are more flexible. Although the M3AE and CLIP filtering models both are trained on CC12M and examine both modalities, M3AE performs much worse, potentially due to a combination of CLIP encouraging image-text alignment and the difficulty of text reconstruction from just CC12M. We conclude that CLIP models are the most practical and performant models for image-text DFNs.

## 4 Creating Better Data Filtering Networks

Equipped with a better understanding of CLIP models as data filtering networks, we aim to create better data filtering networks. DFNs can be trained and modified in the same ways as standard machine learning models. We start by training a CLIP model on a high-quality dataset, and then we can fine-tune the filtering network on subsequent datasets that we want to do especially well on. We use weight ensembling to reduce overfitting on the fine-tuned datasets. Standard machine learning techniques such as augmentation, using a different initialization, and training for more steps with a larger batch size seem to improve the filtering model. We demonstrate the effect of these interventions in Table 2. On the other hand, using a different model size seems to have limited benefits, while model ensembling increases filtering costs without bringing gains. Compared to previous datasets such as DataComp-1B (DC-1B) which involved combining CLIP filtering with clustering-based heuristics, DFNs simplify the data filtering process into a single pipeline while also reducing computational costs.

To create our best DFN, we train a ViT-B/32 CLIP model on High-Quality Image-Text Pairs (HQITP-350M), which is a high-quality dataset of 357 million image-text samples with humanverified captions. This dataset is similar to the HQITP-135M used in Ranasinghe et al. (2023), but expanded to $357 \mathrm{M}$ examples. We initialize the weights with OpenAI's checkpoint. We then fine-tune on the combined MS COCO training set, Flickr30k training set, and ImageNet $1 \mathrm{k}$ with OpenAI templates as the captions. We use additional augmentation at both training and finetuning time. Additional training details can be found in Appendix B. We create our dataset DFN2B by applying this DFN on DataComp's full 12.8 billion sample CommonPool, with a threshold equivalent to taking the top $15 \%$ of samples.

Table 2: Standard interventions used to improve models can be used on DFNs to induce stronger datasets, leading to better models. DFNs are used to filter and train DataComp large (ViT-B/16, 1.28B samples seen).

| Intervention |  | IN | IN Shifts | VTAB | Retrieval | Average |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Augmentation | $\boldsymbol{x}$ | 0.620 | 0.493 | 0.534 | 0.515 | 0.536 |
|  | $\boldsymbol{\checkmark}$ | 0.626 | 0.501 | 0.534 | 0.516 | 0.542 |
| Samples Seen / Batch Size | $2.56 \mathrm{~B} / 4096$ | 0.626 | 0.506 | 0.536 | 0.511 | 0.545 |
|  | $5.12 \mathrm{~B} / 8192$ | 0.624 | 0.508 | 0.551 | 0.517 | 0.550 |
| Fine-tune | $\boldsymbol{x}$ | 0.624 | 0.508 | 0.551 | 0.517 | 0.550 |
|  | $\boldsymbol{\checkmark}$ | 0.678 | 0.540 | 0.555 | 0.534 | 0.560 |
| OAI-Init | $\boldsymbol{x}$ | 0.674 | 0.535 | 0.533 | 0.529 | 0.548 |
|  | $\boldsymbol{\checkmark}$ | 0.678 | 0.540 | 0.555 | 0.534 | 0.560 |

Our DFN induces datasets that achieve state-of-the-art results on medium, large, and xlarge scales in DataComp. In particular at xlarge, we train a ViT-L/14 on DFN-2B for 12.8B samples seen to achieve $81.4 \%$ zero-shot accuracy on ImageNet, and a 0.669 average over 38 DataComp evaluation datasets. As shown in Table 3, in terms of ImageNet zero-shot improvement, this is a $2.2 \% \mathrm{im}-$ provement over DC-1B, a $5.9 \%$ improvement over OpenAI WIT-400M, and a $8.3 \%$ improvement over LAION-2B. These improvements are beyond ImageNet, as we can see similar trends across the DataComp evaluation suite in distribution shifts, retrieval, VTAB, and average performance. Lastly, we train DFN-5B on a ViT-H/14 for 39B samples seen at $224 \times 224$ resolution, and 5B samples at $378 \times 378$ resolution - achieving $84.4 \%$ zero-shot transfer accuracy on ImageNet, and 0.710 average on the DataComp evaluation suite. We find that models trained our DFN produced datasets outperform all other models on the evaluation suite regardless of pre-training dataset: MetaClip, WebLI or DataComp-1B (Xu et al., 2023; Zhai et al., 2022a; Gadre et al. 2023), archiectural improvements such as shape-optimized ViTs (Alabdulmohsin et al., 2023), a more performant sigmoid loss (Zhai et al., 2023), or pre-training performance optimizations such as those in Li et al. (2023b).

Creating better datasets not only improves model performance, but also improves model efficiency. Performance that was once only achievable by larger models can be matched with a smaller model trained on a better dataset. Our ViT-L/14 trained on DFN-2B surpasses a ViT-G/14 trained on LAION-2B for 34B samples seen by $1.5 \%$ zero-shot accuracy on ImageNet, and by 0.002 average performance, despite using 16x less computational cost 2 . Similarly, we can train a ViT-B/16 on DFN-2B for 12.8B samples seen to achieve competitive performance with OpenAI's ViT-L/14, representing a $4 \mathrm{x}$ computational cost reduction.

The key to training good DFNs is using high-quality data for training the filtering network. Collecting verified high-quality data is expensive, as it often requires human annotations, and is thus difficult to scale to large quantities. But given a sizable high-quality dataset, we can explore if there are benefits to directly training on it instead of using it to train a DFN. In Table 4 , we compare models trained on datasets induced by our DFNs with a model trained on HQITP-350M combined[^2]

Table 3: Training on DFN-2B produces state-of-the-art CLIP models. Here we evaluate on the DataComp benchmark, comparing against LAION-2B, DC-1B, MetaCLIP and OpenAI WIT-400M. Additional comparisons can be found on the DataComp leaderboard.

| Dataset | DataComp <br> Scale | IN | IN Shifts | VTAB | Retrieval | Average |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DC-1B | medium | 0.297 | 0.239 | 0.346 | 0.231 | 0.328 |
| DFN-2B | medium | 0.371 | 0.298 | 0.388 | 0.288 | 0.373 |
| DC-1B | large | 0.631 | 0.508 | 0.546 | 0.498 | 0.537 |
| DFN-2B | large | 0.678 | 0.540 | 0.555 | 0.534 | 0.560 |
| LAION-2B | xlarge | 0.731 | 0.603 | 0.586 | 0.589 | 0.601 |
| OpenAI WIT-400M | xlarge | 0.755 | 0.649 | 0.586 | 0.543 | 0.617 |
| DC-1B | xlarge | 0.792 | 0.679 | 0.652 | 0.608 | 0.663 |
| DFN-2B | xlarge | 0.814 | 0.688 | 0.656 | 0.649 | 0.669 |
| LAION-2B | N/A, ViT-G/14-224px | 0.801 | 0.691 | 0.646 | 0.635 | 0.667 |
| DC-1B (CLIPA-v2) | N/A, ViT-G/14-224px | 0.831 | $\mathbf{0 . 7 4 0}$ | 0.645 | 0.631 | 0.684 |
| MetaCLIP | N/A, ViT-H/14-336px | 0.805 | 0.700 | 0.640 | 0.652 | 0.667 |
| WebLI | N/A, ViT-SO/400M-384px | 0.831 | 0.734 | 0.648 | $\mathbf{0 . 6 9 8}$ | 0.692 |
| DFN-5B | N/A, ViT-H/14-224px | 0.834 | 0.713 | 0.675 | 0.684 | 0.698 |
| DFN-5B | N/A, ViT-H/14-378px | $\mathbf{0 . 8 4 4}$ | 0.738 | $\mathbf{0 . 6 8 5}$ | 0.695 | $\mathbf{0 . 7 1 0}$ |

with the dataset induced by CLIP filtering CommonPool with OpenAI's ViT-B/32. Models trained on DFN induced datasets outperform the baseline on all major categories within the DataComp evaluation suite. Furthermore, training on the combination of HQITP-350M and DFN-2B seems to have little improvement when compared to just training on DFN-2B. By training a DFN instead of directly training on high-quality data, we demonstrate a successful recipe for leveraging high-quality data for creating large-scale high-quality datasets.

We can also explore the differences between fine-tuning a DFN and directly training on the finetuning dataset. In Figure 5 and Table 8, we compare models trained on a dataset induced by a baseline DFN, a dataset induced by the baseline DFN fine-tuned on ImageNet and a dataset induced by the baseline DFN without fine-tuning on ImageNet combined with ImageNet. While the model that directly trains on ImageNet has much higher performance on ImageNet and ImageNet-V2, it does not improve upon the baseline for the ObjectNet, ImageNet-Sketch, and ImageNet-R. On the other hand, the DFN fine-tuned on ImageNet induces a dataset that improves over the baseline on ImageNet and all of its distribution shifts. Fine-tuning on DFNs acts as a regularizer to induce datasets similar to the fine-tuning dataset, while maintaining strong robustness properties that come with drawing from a more distributionally diverse candidate pool.

### 4.1 Better Datasets Beyond Vision Tasks: VQA

Just like how machine learning models would ideally generalize across many tasks, we would also like our datasets to generalize across diverse tasks. We show that our datasets not only lead to better models when evaluated on vision tasks, but also lead to better visual question answering

Table 4: High-quality data is best used to train the filtering model rather than the end model. Training DFNs with HQITP-350M induces a dataset that outperforms the dataset induced by a worse DFN combined with HQITP-350M.

| Dataset | Model | IN | IN Shifts | VTAB | Retrieval | Average |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| OAI ViT-B/32 Induced Dataset | ViT-B $/ 16$ | 0.706 | 0.572 | 0.582 | 0.575 | 0.596 |
| + HQITP-350M |  |  |  |  |  |  |
| DFN without FT Induced Dataset | ViT-B 16 | 0.729 | 0.599 | 0.604 | 0.597 | 0.612 |
| DFN-2B | ViT-B/16 | 0.762 | 0.623 | 0.598 | 0.611 | 0.609 |
| OAI ViT-B/32 Induced Dataset | ViT-L/14 | 0.774 | 0.654 | 0.643 | 0.616 | 0.654 |
| + HQITP-350M | ViT-L/14 | 0.814 | 0.688 | 0.656 | 0.649 | 0.669 |
| DFN-2B | ViT-L/14 | 0.813 | 0.691 | 0.662 | 0.656 | 0.670 |
| DFN-2B + HQITP-350M |  |  |  |  |  |  |

(VQA) models. We train a BLIP2 model (Li et al., 2023a) which takes as input a CLIP visual encoder and is trained for zero-shot VQA on $\mathrm{COCO}$ and Visual Genome, to measure zero-shot VQA performance on VQVA2, GQA, and OKVQA (Goyal et al., 2017; Hudson \& Manning, 2019, Marino et al. 2019). We compare the performance on BLIP2 between the standard OpenAI ViT-L visual encoder and the ViT-L trained on DFN-2B. The DFN-2B model consistently outperforms the OpenAI ViT-L encoder and is competitive with a much larger EVA ViT-g model trained on LAION-2B3

Table 5: Performance of BLIP-2 variants with different visual encoder training datasets. The DFN2B trained ViT-L provides consistent improvements across multiple zero-shot VQA tasks.

| Visual Encoder <br> Training Dataset | Architecture | VQAv2 Acc. (\%) | GQA Acc. (\%) | OKVQ Acc. (\%) |
| :--- | :---: | :---: | :---: | :---: |
| OAI-WIT-400M | ViT-L | 45.5 | 30.0 | 19.1 |
| DFN-2B | ViT-L | 48.3 | 31.3 | 21.9 |
| LAION-2B | ViT-g | 48.7 | 31.1 | 24.5 |

### 4.2 Publicly Reproducible DFNs

Scientific research benefits from results that can be reproduced by anyone from scratch. Though OpenAI's internal dataset and HQITP-350M are not publicly accessible, we demonstrate that a competitive DFN can be trained on public data sources. We train a ViT-B/32 on Conceptual Caption12M, Conceptual Captions 3M, and Shutterstock 15M (Changpinyo et al., 2021, Sharma et al. 2018; Nguyen et al., 2023). As shown in Table 6, this DFN matches OpenAI's ViT-B/32 in terms of filtering performance at DataComp's medium and large scales. Additionally, this DFN can be modified as described in the previous section to further improve filtering performance.[^3]

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-12.jpg?height=672&width=786&top_left_y=255&top_left_x=325)
$--=x$

- ImageNet Classification
- Linear fit (ImageNet Classification)
$\star$ CLIP zero-shot
- Linear fit (CLIP zero-shot)
$\rightarrow$ Model trained on dataset induced by DFN
DFN no FT
DFN no FT Induced
DFN no FT Induced + IN1k
* DFN FT on IN1k + Retr
* DFN FT on IN1k + Retr Induced

```

Figure 5: Datasets induced by DFNs can be robust to distribution shift. DFNs can be fine-tuned to maintain robustness of induced datasets, unlike directly training on ImageNet (IN). DFNs are not performing distillation because induced datasets lead to higher performing models than the original DFN. Distribution shifts used are IN-V2, ObjectNet, IN-Sketch, IN-R, and IN-A.

Table 6: DFNs are trained with a ViT-B/32, then used to filter DataComp pools. Conceptual 12M, Conceptual Captions 3M, and Shutterstock $15 \mathrm{M}$ are publicly available datasets, demonstrating that large-scale high-quality datasets can be constructed with only publicly available resources.

\begin{tabular}{lllllll}
\hline DFN Training Data & \begin{tabular}{l} 
DataComp \\
Scale
\end{tabular} & IN & IN Shifts & VTAB & Retrieval & Average \\
\hline CC12M + CC3M + SS15M & medium & 0.307 & 0.253 & 0.359 & 0.274 & 0.349 \\
OpenAI WIT-400M & medium & 0.285 & 0.240 & 0.355 & 0.253 & 0.338 \\
\hline CC12M + CC3M + SS15M & large & 0.591 & 0.481 & 0.522 & 0.503 & 0.532 \\
OpenAI WIT-400M & large & 0.578 & 0.466 & 0.525 & 0.475 & 0.527 \\
\hline
\end{tabular}

\section*{5 Discussion}

The simplicity of the data filtering network pipeline makes it a flexible tool to integrate into existing workflows. As DFNs operates on individual samples, this approach scales linearly with candidate pool size, enabling the creation of datasets orders of magnitude larger than those that we introduce in this work. Additionally, the DFNs we train in this work are relatively small neural networks which allows for filtering to be directly integrated into training procedures of much larger networks for minimal marginal cost. DFNs can then filter batches of raw data that are then trained on, reducing the need for complex data pre-processing procedures.

As useful as DFNs are in building performant models in this work, there are still many unanswered questions to address in future work. We still do not know exactly how to optimize directly for dataset quality, and thus opt for weak proxies such as alignment. It is not even clear what that
proxy would be for other domains where DFNs could be applied such as speech, text or video data. We hope that these open questions and the bridge DFNs build between modeling work and dataset work can lead to fruitful new avenues of research.

\section*{Acknowledgements}

We would like to thank Bowen Zhang, Ruoming Pang, Brandon McKinzie, Mitchell Wortsman, Gabriel Ilharco, Ross Wightman, Achal Dave, Josh Susskind, Alaaeldin Ali, Fartash Faghri, Preetum Nakkiran, and Chen Huang for helpful feedback at various stages of the project.

Bowen and Ruoming were invaluable for helping us set up AxLearn and answering countless questions when we ran into various errors. Brandon pointed us in the direction of the HQITP datasets that were crucial for our best results, and also helped us with AxLearn issues. Mitchell's OpenClip experience helped us set hyper-parameters for our largest scale runs. Gabriel helped us debug a webdataset related dataloader bug. Ross caught a bug in our final high resolution model that led to a modest performance improvement. Achal provided hyper-parameters and instructions for training BLIP2 for the VQA experiments. Alaaeldin, Chen, Fartash, Josh, and Preetum provided helpful comments on the manuscript.

\section*{References}

Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixed-modal language models. arXiv preprint arXiv:2301.03728, 2023.

Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. arXiv preprint arXiv:2305.13035, 2023.

Apple. axlearn, July 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3558-3568, 2021.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818-2829, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition (CVPR), 2009. https://ieeexplore.ieee.org/abstract/document/5206848.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth $16 \mathrm{x} 16$ words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, and Arushi Somani. Releasing Persimmon-8B, 2023. URL https://www.adept.ai/ $\mathrm{blog} /$ persimmon-8b.

Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pretraining (clip), 2022.

Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Xinyang Geng, Hao Liu, Lisa Lee, Dale Schuurams, Sergey Levine, and Pieter Abbeel. Multimodal masked autoencoders learn transferable representations. arXiv preprint arXiv:2205.14204, 2022.

Sachin Goyal, Ananya Kumar, Sankalp Garg, Zico Kolter, and Aditi Raghunathan. Finetune like you pretrain: Improved finetuning of zero-shot vision models, 2022.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https: //arxiv.org/abs/1512.03385.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically, 2017.

W. H. Highleyman and L. A. Kamentsky. A generalized scanner for pattern- and characterrecognition studies. In Papers Presented at the the March 3-5, 1959, Western Joint Computer Conference, IRE-AIEE-ACM '59 (Western), pp. 291-294, New York, NY, USA, 1959. Association for Computing Machinery. ISBN 9781450378659. doi: 10.1145/1457838.1457894. URL https://doi.org/10.1145/1457838.1457894

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali

Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/zenodo. 5143773 .

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images, 2009. https://www.cs.toronto.edu/ kriz/learning-features-2009-TR.pdf.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2012. https://proceedings.neurips.cc/paper_files/paper/2012/file/ c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.

Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541551, 1989. doi: 10.1162/neco.1989.1.4.541.

Yann LeCun, Yann LeCun, and Yann LeCun. The mnist database of handwritten digits, 1998. http://yann.lecun.com/exdb/mnist/.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023a.

Xianhang Li, Zeyu Wang, and Cihang Xie. Clipa-v2: Scaling clip training with 81.1accuracy within a $\$ 10,000$ budget; an extra $\$ 4,000$ unlocks $81.82023 \mathrm{~b}$.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Juan Ciro, Lora Aroyo, Bilge Acun, Sabri Eyuboglu, Amirata Ghorbani, Emmett Goodman, Tariq Kane, Christine R. Kirkpatrick, Tzu-Sheng Kuo, Jonas Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen Paritosh, Ce Zhang, James Zou, Carole-Jean Wu, Cody Coleman, Andrew Ng, Peter Mattson, and Vijay Janapa Reddi. Dataperf: Benchmarks for data-centric ai development, 2022. https://arxiv.org/abs/2207.10062.

Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip, 2023.

OpenAI. Gpt-4 technical report, 2023.

Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei Yu, Jiahui Yu,

Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, Mingxing Tan, and Quoc V. Le. Combined scaling for zero-shot transfer learning, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2103.00020.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356, 2022 .

Kanchana Ranasinghe, Brandon McKinzie, Sachin Ravi, Yinfei Yang, Alexander Toshev, and Jonathon Shlens. Perceptual grouping in contrastive vision-language models, 2023.

Nithya Sambasivan, Shivani Kapania, Hannah Highfill, Diana Akrong, Praveen Paritosh, and Lora M Aroyo. "everyone wants to do the model work, not the data work": Data cascades in high-stakes ai. In proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp. 1-15, 2021.

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of clip-filtered 400 million image-text pairs, 2021. https://arxiv.org/abs/2111.02114.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation imagetext models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2022. https://openreview.net/forum?id=M3Y74vmsMcY.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. Ccnet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359, 2019.

Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs,

Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models, 2022.

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data, 2023.

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers, 2022a.

Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning, 2022b.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training, 2023.

\section*{A Training Hyperparameters}

Table 7: We follow the hyperparameter settings of the DataComp paper for the medium, large and xlarge pool

\begin{tabular}{llclllll}
\hline Dataset & Model & \begin{tabular}{c} 
Pool size and \\
\# seen samples
\end{tabular} & \begin{tabular}{l} 
Batch \\
Size
\end{tabular} & Max LR & \begin{tabular}{l} 
Weight \\
Decay
\end{tabular} & Warmup & Beta2 \\
\hline DataComp-medium & ViT-B/32 & $128 \mathrm{M}$ & 4096 & $5 \mathrm{e}-4$ & 0.2 & 500 & - \\
DataComp-large & ViT-B/16 & $1.28 \mathrm{~B}$ & 8192 & $5 \mathrm{e}-4$ & 0.2 & 500 & - \\
DataComp-xlarge & ViT-L/14 & $12.8 \mathrm{~B}$ & 90112 & $1 \mathrm{e}-3$ & 0.2 & 10000 & 0.95 \\
DFN-5B-pool & ViT-H/14 & $39 \mathrm{~B}$ & 79872 & $2 \mathrm{e}-3$ & 0.1 & 10000 & 0.95 \\
\hline
\end{tabular}

\section*{B DFN Hyperparameters}

DFNs trained for ablations use DataComp large scale hyperparameters with a ViT-B/32 instead of a ViT-B/16. Final DFNs that induce DC-2B train for 5.12B samples, 16,384 batch size, and 2,000 steps of warmup.

\section*{C Robustness of Using ImageNet at Filtering vs. Training Time}

Table 8: Fine-tuning a DFN on ImageNet induces datasets with nice robustness properties that are lost when directly training on ImageNet. Ran at DataComp large scale (ViT-B/16, 1.28B samples).

\begin{tabular}{lccccccc}
\hline Dataset & IN & IN-V2 & ObjectNet & IN-Sketch & IN-R & IN-A & VTAB \\
\hline Baseline DFN & 0.624 & 0.547 & 0.511 & 0.510 & 0.724 & 0.257 & 0.551 \\
Baseline DFN FT on ImageNet & 0.678 & 0.594 & 0.536 & 0.536 & 0.743 & 0.284 & 0.555 \\
Baseline DFN + IN & 0.757 & 0.652 & 0.509 & 0.512 & 0.703 & 0.272 & 0.543 \\
\hline
\end{tabular}

\section*{D Full Experimental Evaluation \& Model Release}

Below we provide links to checkpoints and detailed evaluation results of models in Table 3 on each of the 38 DataComp evaluation datasets

\begin{tabular}{lcc}
\hline Model Link & ImageNet & Average \\
\hline DFN5B-CLIP-ViT-H-14-378 & 0.844 & 0.710 \\
\hline DFN5B-CLIP-ViT-H-14 & 0.834 & 0.698 \\
\hline DFN2B-CLIP-ViT-L-14 \\
\hline DFN2B-CLIP-ViT-B-16 & 0.814 & 0.669 \\
\hline
\end{tabular}

Table 9: Links to checkpoints and detailed evaluation results

\section*{E Figures measuring average performance instead of ImageNet}
![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-19.jpg?height=712&width=1308&top_left_y=348&top_left_x=405)

Dataset

CC12M+CC3M+SS15M
- HQITP-135M (Ours)
- HQITP-350M (Ours)
- LAION-2B

OpenAl-WIT400M
- CommonPool
- DataComp-1B

Steps
- 1B
- 3B

13B
- ViT-L/14

Figure 6: Average accuracy version of Figure 3.

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-19.jpg?height=612&width=1287&top_left_y=1147&top_left_x=408)

Figure 7: Average accuracy version of Figure 4

\section*{F Log-Scale plot of Figure 1}

![](https://cdn.mathpix.com/cropped/2024_06_04_944d346a9b1b981697beg-20.jpg?height=634&width=1619&top_left_y=358&top_left_x=253)

Figure 8: Compute scaling behavior of training CLIP models on various datasets (log scale)

\section*{G Additional Tables}

Table 10: We can produce high-quality DFNs completely from scratch. Specifically, we do not use any OpenAI CLIP models for results in this table. We also use HQITP-135M for DFN training, a subset of HQITP-350M that we use in the rest of the paper

\begin{tabular}{lllllll}
\hline Dataset & \begin{tabular}{l} 
DataComp \\
Scale
\end{tabular} & IN & IN Shifts & VTAB & Retrieval & Average \\
\hline \begin{tabular}{l} 
Induced by DFN HQITP-135M \\
with FT on IN1k, \\
no OAI-Init, no Aug., \\
samples 1B, BS 4096
\end{tabular} & xlarge & 0.805 & 0.665 & 0.641 & 0.639 & 0.663 \\
\hline
\end{tabular}```


[^0]:    *Work done while at Apple

[^1]:    ${ }^{1}$ Most large public Image-Text datasets including LAION-5B and DataComp-1B are built using OpenAI's CLIP model

[^2]:    ${ }^{2}$ calculation does not take into account patch dropout used to train ViT-G/14 on LAION-2B

[^3]:    ${ }^{3}$ EVA's ViT-g has an additional pre-training procedure trained on ImageNet-21k, COCO, Objects365 and Conceptual Captions $12 \mathrm{M}$

</end of paper 3>


