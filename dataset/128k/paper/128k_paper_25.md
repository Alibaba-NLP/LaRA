<paper 0>
# DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection in Multivariate Time Series 

Zahra Zamanzadeh Darban ${ }^{1 *}$, Geoffrey I. Webb ${ }^{1}$, Mahsa Salehi ${ }^{1}$<br>${ }^{1}$ Monash University, Melbourne<br>\{zahra.zamanzadeh, geoff.webb, mahsa.salehi\}@ monash.edu


#### Abstract

Time series anomaly detection (TAD) faces a significant challenge due to the scarcity of labelled data, which hinders the development of accurate detection models. Unsupervised domain adaptation (UDA) addresses this challenge by leveraging a labelled dataset from a related domain to detect anomalies in a target dataset. Existing domain adaptation techniques assume that the number of anomalous classes does not change between the source and target domains. In this paper, we propose a novel Domain Adaptation Contrastive learning for Anomaly Detection in multivariate time series (DACAD) model to address this issue by combining UDA and contrastive representation learning. DACAD's approach includes an anomaly injection mechanism that introduces various types of synthetic anomalies, enhancing the model's ability to generalise across unseen anomalous classes in different domains. This method significantly broadens the model's adaptability and robustness. Additionally, we propose a supervised contrastive loss for the source domain and a self-supervised contrastive triplet loss for the target domain, improving comprehensive feature representation learning and extraction of domaininvariant features. Finally, an effective Centrebased Entropy Classifier (CEC) is proposed specifically for anomaly detection, facilitating accurate learning of normal boundaries in the source domain. Our extensive evaluation across multiple real-world datasets against leading models in time series anomaly detection and UDA underscores DACAD's effectiveness. The results validate DACAD's superiority in transferring knowledge across domains and its potential to mitigate the challenge of limited labelled data in time series anomaly detection.


[^0]
## 1 Introduction

Unsupervised Domain Adaptation (UDA) is a technique used to transfer knowledge from a labelled source domain to an unlabeled target domain, particularly useful when labelled data in the target domain are scarce or unavailable. Deep learning methods have become the predominant approach in UDA, offering advanced capabilities and significantly improved performance compared to traditional techniques. UDA's approach is crucial in situations where deep models' performance drops significantly due to the discrepancy between the data distributions in the source and target domains, a phenomenon known as domain shift [Liu et al., 2022].

Applications of UDA are diverse, ranging from image and video analysis to natural language processing and time series data analysis. In time series data analysis and anomaly detection, UDA can be particularly challenging, given that often i) the nature of time series data across different domains is complex and varied [Wilson and Cook, 2020; Liu et al., 2022], and ii) the number of anomalous classes changes between the source and target domains.

For time series data specifically, UDA methods often employ neural architectures as feature extractors. These models are designed to handle domain adaptation, primarily target time series regression and classification problems [Purushotham et al., 2016; Cai et al., 2021; Ozyurt et al., 2023; Wilson et al., 2020], focusing on aligning the major distributions of two domains. This approach may lead to negative transfer effects [Zhang et al., 2022] on minority distributions, which is a critical concern in time series anomaly detection (TAD). The minority distributions, often representing rare events or anomalies, may be overshadowed or incorrectly adapted due to the model's focus on aligning dominant data distributions, potentially leading to a higher rate of false negatives in anomaly detection. Since transferring knowledge of anomalies requires aligning minority distributions and the anomaly label spaces often have limited similarity across domains, existing methods face limitations in addressing anomaly detection in time series data. This highlights a significant gap in current UDA methodologies, pointing towards the need for novel approaches that can effectively address the unique requirements of anomaly detection in time series data.

Furthermore, in the realm of time series anomaly detection, in the most recent model called ContextDA [Lai et al., 2023],
the discriminator aligns source/target domain windows without leveraging label information in the source domain, which makes the alignment ineffective. Our model leverages source labels and anomaly injection for better feature extraction, enhancing the alignment of normal samples. This is particularly vital due to the challenges in aligning anomalies across domains with different anomalous classes.

In our study, we introduce Domain Adaptation Contrastive learning model for Anomaly Detection in time series (DACAD), a unique framework for UDA in multivariate time series anomaly detection, leveraging contrastive learning (CL). DACAD focuses on contextual representations. It forms positive pairs based on proximity and negative pairs using anomaly injection, subsequently learning their embeddings with both supervised CL in the source domain and selfsupervised CL in the target domain. To sum up, we make the following contributions:

- We introduce DACAD, a pioneering CL framework for multivariate time series anomaly detection with UDA. It is a novel deep UDA model based on contrastive representation utilising a labelled source dataset and synthetic anomaly injection to overcome the lack of labelled data in TAD, distinguishing between normal and anomalous patterns effectively.
- Extending on the deep one-class classifier (DeepSVDD) [Ruff et al., 2018], our proposed one-class classifier, the CEC, operates on the principle of spatial separation in the feature space by leveraging the existing label information in the source domain, aiming to bring "normal" sample representations closer to the centre and distancing anomalous ones. This spatial adjustment is measured using a distance metric, establishing the basis for anomaly detection.
- Our comprehensive evaluation with real-world datasets highlights DACAD's efficiency. In comparison with the recent TAD deep models and UDA models for time series classification and UDA for anomaly detection, DACAD demonstrates superior performance, emphasising the considerable importance of our study.


## 2 Related Works

In this section, we provide a brief survey of the existing literature that intersects with our research. We concentrate on two areas: UDA for time series and deep learning for TAD.

UDA for time series: In the realm of UDA for time series analysis, the combination of domain adaptation techniques with the unique properties of time series data presents both challenges and opportunities. Traditional approaches such as Maximum Mean Discrepancy (MMD) [Long et al., 2018] and adversarial learning frameworks [Tzeng et al., 2017] are geared towards minimizing domain discrepancies by developing domain-invariant features. These methods are vital in applications spanning medical, industrial, and speech data, with significant advancements in areas like sleep classification [Zhao et al., 2021], arrhythmia classification [Wang et al., 2021], and various forms of anomaly detection [Lai et al., 2023], fault diagnosis [Lu et al., 2021], and lifetime prediction [Ragab et al., 2020].
Time series specific UDA approaches like variational recurrent adversarial deep domain adaptation (VRADA) by Purushotham et al., pioneered UDA for multivariate time series, utilising adversarial learning with an LSTM networks [Hochreiter and Schmidhuber, 1997] and variational RNN [Chung et al., 2015] feature extractor. Convolutional deep domain adaptation for time series (CoDATS) by Wilson et al., built on VRADA's adversarial training but employed a convolutional neural network as the feature extractor. A metric-based method, time series sparse associative structure alignment (TS-SASA) [Cai et al., 2021] aligns intra and inter-variable attention mechanisms between domains using MMD. Adding to these, the CL for UDA of time series (CLUDA) [Ozyurt et al., 2023] model offers a novel approach, enhancing domain adaptation capabilities in time series. All these methods share a common objective of aligning features across domains, each contributing unique strategies to the complex challenge of domain adaptation in time series classification data. However, they are ineffective when applied to TAD tasks. Additionally, ContextDA, introduced by Lai et al., is a TAD model that applies deep reinforcement learning to optimise domain adaptation, framing context sampling as a Markov decision process. However, it is ineffective when the anomaly classes change between source and target.

Deep Learning for TAD: The field of Time Series Anomaly Detection (TAD) has advanced significantly, embracing a variety of methods ranging from basic statistical approaches to sophisticated deep learning techniques [Schmidl et al., 2022; Audibert et al., 2022]. Notably, deep learning has emerged as a promising approach due to its autonomous feature extraction capabilities [Darban et al., 2022]. TAD primarily focuses on unsupervised [Yue et al., 2022; Hundman et al., 2018; Audibert et al., 2020; Xu et al., 2022] and semi-supervised methods [Niu et al., 2020; Park et al., 2018] to tackle the challenge of limited labelled data availability. Unsupervised methods like OmniAnomaly [Su et al., 2019], GDN [Deng et al., 2021], and BeatGAN [Zhou et al., 2019] are especially valuable in scenarios with sparse anomaly labels, whereas semi-supervised methods leverage available labels effectively.

Advanced models such as LSTM-NDT [Hundman et al., 2018] and THOC [Shen et al., 2020] excel in minimizing forecasting errors and capturing temporal dependencies. However, models like USAD [Audibert et al., 2020] face challenges with long time series data due to error accumulation in decoding. Additionally, unsupervised representation learning, exemplified by SPIRAL [Lei et al., 2019] and TST [Zerveas et al., 2021], shows impressive performance, albeit with scalability issues in long time series. Newer models like TNC [Tonekaboni et al., 2021] and TLoss [Franceschi et al., 2019] aim to overcome these challenges using methods such as time-based negative sampling.

Furthermore, contrastive representation learning, crucial in TAD for pattern recognition, groups similar samples together while distancing dissimilar ones. It has been effectively employed in TS2Vec [Yue et al., 2022] for multi-level semantic representation and in DCdetector [Yang et al., 2023], which uses a dual attention asymmetric design for permutation invariant representations.

## 3 DACAD

Problem formulation: Given an unlabeled time series dataset $T$ (target), the problem is to detect anomalies in $T$ using a labelled time series dataset $S$ (source) from a related domain.

In this section, we present DACAD, which ingeniously uses temporal correlations in time series data and adapts to differences between source and target domains. It starts with a labelled dataset in the source domain, featuring both normal and anomalous instances. Both real and synthetic (injected) anomalies aid in domain adaptation, ensuring training and improving generalisability on a wide range of anomaly classes.

DACAD's core is CL as inspired by a recent UDA for time series work [Ozyurt et al., 2023], which strengthens its ability to handle changes between domains by improving the feature representation learning in source and target domains. As described in subsection 3.3, in the target domain, we use a self-supervised contrastive loss [Schroff et al., 2015] by forming triplets to minimise the distance between similar samples and maximise the distance between different samples. Additionally, in the source domain, we leverage label information of both anomaly and normal classes and propose an effective supervised contrastive loss [Khosla et al., 2020] named supervised mean-margin contrastive loss. A Temporal Convolutional Network (TCN) in DACAD captures temporal dependencies, generating domain-invariant features. A discriminator ensures these features are domain-agnostic, leading to consistent anomaly detection across domains. DACAD's strength lies in its dual ability to adapt to new data distributions and to distinguish between normal and anomalous patterns effectively. In DACAD, time series data is split into overlapping windows of size $W S$ with a stride of 1 , forming detailed source $(S)$ and target $(T)$ datasets. Windows in the source domain are classified based on anomaly presence: those with anomalies are marked as anomalous windows ( $S_{\text {anom }}$ ), and others as normal ( $S_{\text {norm }}$ ). Figure 1 shows the DACAD architecture, and Algorithm 1 details its steps. The following subsections explore DACAD's components and their functionalities.

### 3.1 Anomaly Injection

In the anomaly injection phase, we augment the original time series windows through a process of negative augmentation, thereby generating synthetic anomalous time series windows. This step is applied to all windows in the target domain $(T)$ and all normal windows in the source domain ( $S_{\text {norm }}$ ). We employ the anomaly injection method outlined in [Darban et al., 2023], which encompasses five distinct types of anomalies: Global, Seasonal, Trend, Shapelet, and Contextual. This procedure results in the creation of two new sets of time series windows. The first set, $S_{\mathrm{inj}}$, consists of anomaly-injected windows derived from the normal samples in the source domain. The second set, $T_{\mathrm{inj}}$, comprises anomaly-injected windows originating from the target domain data.

### 3.2 Pair Selection

In DACAD's pair selection step, appropriate triplets from source and target domains are created for CL. In the source

```
Algorithm 1 DACAD $(S, T, \alpha, \beta, \gamma, \lambda)$
Input: Source time series windows $S=\left\{w_{1}^{s}, w_{2}^{s}, \ldots, w_{|S|}^{s}\right\}$, Target
    time series windows $T=\left\{w_{1}^{t}, w_{2}^{t}, \ldots, w_{|T|}^{t}\right.$, Loss coefficients
    $\alpha, \beta \gamma, \lambda$
Output: Representation function $\phi^{R}$, classifier $\phi^{C L}$, centre $c$
    Initialise $\phi^{R}, \phi^{D}, \phi^{C L}, c$
    Split $S$ to $S_{\text {norm }}$ and $S_{\text {anom }}$
    Create $S_{\text {inj }}, T_{\text {inj }} \quad \triangleright$ Anomaly Injection (3.1)
    Form $S_{\text {triplets }}$ and $T_{\text {triplets }} \quad \triangleright$ Pair Selection (3.2)
    for each training iteration do
        Compute $\phi^{R}(S), \phi^{R}(T), \phi^{R}\left(S_{\text {Triplets }}\right), \phi^{R}\left(T_{\text {Triplets }}\right) \quad \triangleright(3.3)$
        Compute $\mathcal{L}_{\text {SupCont }}$ using Eq. (1) and $\phi^{R}\left(S_{\text {Triplets }}\right)$
        Compute $\mathcal{L}_{\text {SelfCont }}$ using Eq. (2) and $\phi^{R}\left(T_{\text {Triplets }}\right)$
        Compute $\mathcal{L}_{\text {Disc }}$ using Eq. (3), $\phi^{R}(S)$ and $\phi^{R}(T) \quad \triangleright(3.4)$
        Compute $\mathcal{L}_{\text {Cls }}$ using Eq. (4) and $\phi^{R}(S) \quad \triangleright(3.5)$
        $\mathcal{L}_{\text {DACAD }} \leftarrow \alpha \cdot \mathcal{L}_{\text {SupCont }}+\beta \cdot \mathcal{L}_{\text {SelfCont }}+\gamma \cdot \mathcal{L}_{\text {Disc }}+\lambda \cdot \mathcal{L}_{\text {Cls }}$
        Update model parameters to minimise $\mathcal{L}_{\text {DACAD }}$
    end for
    Return $\phi^{R}, \phi^{C L}, c$
```

domain, we use labels to form distinct lists of normal samples ( $S_{\text {norm }}$ ), anomalous samples ( $S_{\text {anom }}$ ), and anomaly-injected samples ( $S_{\mathrm{inj}}$ ). This allows for a supervised CL approach, enhancing differentiation between normal and anomalous samples. Here, triplets ( $S_{\text {triplets }}$ ) consist of an anchor (normal window from $S_{\text {norm }}$ ), a positive (different normal window from $S_{\text {norm }}$ ), and a negative (randomly selected from either an anomalous window from $S_{\text {anom }}$ or an anomaly-injected anchor from $S_{\text {inj }}$ ).

In the target domain, lacking ground truth labels, a selfsupervised approach shapes the triplets ( $T_{\text {triplets }}$ ). Each target triplet includes an anchor (original window from $T$ ), a positive (temporally close window to anchor, from $T$, likely sharing similar characteristics), and a negative (anomaly-injected anchor from $T_{\text {inj }}$ ).

### 3.3 Representation Layer ( $\phi^{R}$ )

In our model, we employ a TCN [Lea et al., 2016] for the representation layer, which is adept at handling both multivariate time series windows. This choice is motivated by the TCN's ability to capture temporal dependencies effectively, a critical aspect in time series analysis. The inputs to the TCN are the datasets and triplets from both domains, specifically $S$, $T, S_{\text {Triplets }}$, and $T_{\text {Triplets. }}$. The outputs of the TCN, representing the transformed feature space, are

- $\phi^{R}(S)$ : The representation of source windows.
- $\phi^{R}(T)$ : The representation of target windows.
- $\phi^{R}\left(S_{\text {Triplets }}\right)$ : The representation of source triplets.
- $\phi^{R}\left(T_{\text {Triplets }}\right)$ : The representation of target triplets.

Utilising the outputs from the representation layer $\phi^{R}$, we compute two distinct loss functions. These are the supervised mean margin contrastive loss for source domain data $\left(\mathcal{L}_{\text {SupCont }}\right)$ and the self-supervised contrastive triplet loss for target domain $\left(\mathcal{L}_{\text {SelfCont }}\right)$. These loss functions are critical for training our model to differentiate between normal and anomalous patterns in both source and target domains.

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-04.jpg?height=545&width=1778&top_left_y=194&top_left_x=171)

Figure 1: DACAD Model Overview: Involves source $(S)$ and target $(T)$ domains. Source domain uses normal ( $S_{\text {norm }}$ ) and anomalous ( $S_{\text {anom }}$ ) samples, plus synthetic anomalies ( $S_{\mathrm{inj}}$ ) for source triplets ( $S_{\text {Triplets }}$ ) and contrastive loss. Target domain similarly uses proximity-based pair selection and anomaly injection ( $T_{\text {inj }}$ ) to create target triplets ( $T_{\text {Triplets }}$ ). $\operatorname{TCN}\left(\phi^{R}\right)$ is used for feature extraction. Features from both domains are fed into discriminator ( $\phi^{D}$ ) for domain-invariant learning. Source features are classified by classifier $\phi^{C L}$.

## Supervised Mean Margin Contrastive Loss for Source Domain $\left(\mathcal{L}_{\text {SupCont }}\right)$

This loss aims to embed time series windows into a feature space where normal sequences are distinctively separated from anomalous ones. It utilises a triplet loss framework, comparing a base (anchor) window with both normal (positive) and anomalous (negative) windows [Khosla et al., 2020]. Our method diverges from traditional triplet loss by focusing on the average effect of all negatives within a batch. The formula is given by equation (1):

$$
\begin{align*}
& \mathcal{L}_{\text {SupCont }}=\frac{1}{|B|} \sum_{i=1}^{|B|} \max \\
& \left(\frac{1}{|N|} \sum_{j=1}^{|N|}\left(\left\|\phi^{R}\left(a_{i}^{s}\right)-\phi^{R}\left(p_{i}^{s}\right)\right\|_{2}^{2}-\left\|\phi^{R}\left(a_{i}^{s}\right)-\phi^{R}\left(n_{j}^{s}\right)\right\|_{2}^{2}+m\right), 0\right) \tag{1}
\end{align*}
$$

Here, $|B|$ is the batch size, and $|N|$ is the number of negative samples in the batch. The anchor $a_{i}^{s}$ is a normal time series window, and the positive pair $p_{i}^{s}$ is another randomly selected normal window. Negative pairs $n_{j}^{s}$ are either true anomalous windows or synthetically created anomalies through the anomaly injector module applied on the anchor $a_{i}^{s}$. This loss function uses true anomaly labels to enhance the separation between normal and anomalous behaviours by at least the margin $m$. It includes both genuine and injected anomalies as negatives, balancing ground truth and potential anomaly variations. The supervised mean margin contrastive loss offers several advantages for TAD: supervised learning (uses labels for better anomaly distinction), comprehensive margin (applies a margin over average distance to negatives), and flexible negative sampling (incorporates a mix of real and injected anomalies, enhancing robustness against diverse anomalous patterns).

## Self-supervised Contrastive Triplet Loss for Target Domain $\left(\mathcal{L}_{\text {SelfCont }}\right)$

For the target domain, we employ a self-supervised contrastive approach using triplet loss [Schroff et al., 2015], designed to ensure that the anchor window is closer to a positive window than to a negative window by a specified margin. The self-supervised triplet loss formula is shown in equation (2):

$$
\begin{align*}
& \mathcal{L}_{\text {Selff Cont }}= \\
& \frac{1}{|B|} \sum_{i=1}^{|B|} \max \left(\left\|\phi^{R}\left(a_{i}^{t}\right)-\phi^{R}\left(p_{i}^{t}\right)\right\|_{2}^{2}-\left\|\phi^{R}\left(a_{i}^{t}\right)-\phi^{R}\left(n_{i}^{t}\right)\right\|_{2}^{2}+m, 0\right) \tag{2}
\end{align*}
$$

In this setup, the anchor window $a_{i}^{t}$ is compared to a positive window $p_{i}^{t}$ (a randomly selected window in temporal proximity to the anchor from $T$ ) and a negative window $n_{i}^{t}$ (the anomaly-injected version of the anchor from $T_{\mathrm{inj}}$ ), ensuring the anchor is closer to the positive than the negative by at least the margin $m$.

### 3.4 Discriminator Component ( $\phi^{D}$ )

The model incorporates a discriminator component within an adversarial framework. This component is crucial for ensuring that the learned features are not only relevant for the anomaly detection task but also general enough to be applicable across different domains. The discriminator is specifically trained to distinguish between representations from the source and target domains.

Designed to differentiate between features extracted from the source and target domains, the discriminator employs adversarial training techniques similar to those found in Generative Adversarial Networks (GANs) [Creswell et al., 2018]. Discriminator $\phi^{D}$ is trained to differentiate between source and target domain features, while $\phi^{R}$ is conditioned to produce domain-invariant features.

A crucial element is the Gradient Reversal Layer [Ganin and Lempitsky, 2015], which functions normally during forward passes but reverses gradient signs during backpropagation. This setup enhances the training of $\phi^{D}$ and simultaneously adjusts $\phi^{R}$ to produce features that challenge $\phi^{D}$.

The discriminator's training involves balancing its loss against other model losses. Effective domain adaptation occurs when $\phi^{R}$ yields discriminator accuracy close to random guessing. $\phi^{D}$, taking $\phi^{R}(S)$ and $\phi^{R}(T)$ as inputs, classifies them as belonging to the source or target domain. Its loss,
a binary classification problem, minimizes classification error for source and target representations. The loss function for the discriminator $\phi^{D}$ is defined using the Binary CrossEntropy (BCE) loss, as shown in equation (3):

$$
\begin{align*}
& \left.\left.\mathcal{L}_{\text {Disc }}=-\frac{1}{|S|+|T|}\left(\sum_{i=1}^{|S|} \log \left(f\left(w_{i}^{s}\right)\right)\right)+\sum_{j=1}^{|T|} \log \left(1-f\left(w_{j}^{t}\right)\right)\right)\right)  \tag{3}\\
& \text { where } f(w)=\phi^{D}\left(\phi^{R}(w)\right)
\end{align*}
$$

Here, $|S|$ and $|T|$ are the source and target window counts. $w_{i}^{s}$ and $w_{j}^{t}$ represent the $i^{t h}$ and $j^{t h}$ windows from $S$ and $T$. The function $\phi^{D}\left(\phi^{R}(\cdot)\right)$ computes the likelihood of a window being from the source domain.

### 3.5 Centre-based Entropy Classifier ( $\phi^{C L}$ )

Extending the DeepSVDD [Ruff et al., 2018] designed for anomaly detection, the $\mathrm{CEC}\left(\phi^{C L}\right.$ ) in DACAD is proposed as an effective anomaly detection classifier in the source domain. It assigns anomaly scores to time series windows, using labelled data from the source domain $S$ for training and applying the classifier to target domain data $T$ during inference. It is centred around a Multi-Layer Perceptron (MLP) with a unique "centre" parameter crucial for classification.

The classifier operates by spatially separating transformed time series window representations $\left(\phi^{R}\left(w_{i}^{s}\right)\right)$ in the feature space relative to a predefined "centre" $c$. The MLP's aim is to draw normal sample representations closer to $c$ and push anomalous ones further away. This spatial reconfiguration is quantified using a distance metric, forming the basis for anomaly scoring. A sample closer to $c$ is considered more normal, and vice versa. The effectiveness of this spatial adjustment is measured using a loss function based on $\mathrm{BCE}$, expressed in equation (4):

$\mathcal{L}_{\mathrm{Cls}}=-\frac{1}{|S|} \sum_{i=1}^{|S|}\left[y_{i} \cdot \log \left(\left\|g\left(w_{i}^{s}\right)-c\right\|_{2}^{2}\right)+\left(1-y_{i}\right) \cdot \log \left(1-\left\|g\left(w_{i}^{s}\right)-c\right\|_{2}^{2}\right)\right]$ where $g(w)=\phi^{C L}\left(\phi^{R}(w)\right)$

In this equation, $|S|$ is the number of samples in $S, w_{i}^{s}$ is the $i^{\text {th }}$ window in $S$, and $y_{i}$ is its ground truth label, with 1 for normal and 0 for anomalous samples.

The loss function is designed to minimise the distance between normal samples and $c$ while maximising it for anomalous samples. These distances are directly used as anomaly scores, offering a clear method for anomaly detection.

### 3.6 Overall Loss in DACAD

The overall loss function in the DACAD model is the amalgamation of four distinct loss components, each contributing to the model's learning process in different aspects. These components are the Supervised Contrastive Loss ( $\mathcal{L}_{\text {SupCont }}$ ), the Self-Supervised Contrastive Loss ( $\mathcal{L}_{\text {SelfCont }}$ ), the Discriminator Loss $\left(\mathcal{L}_{\text {Disc }}\right)$, and the Classifier Loss $\left(\mathcal{L}_{\mathrm{Cls}}\right)$. The overall loss function for DACAD denoted as $\mathcal{L}_{\text {DACAD }}$, is formulated as a weighted sum of these components (with a specific weight: $\alpha$ for $\mathcal{L}_{\text {SupCont }}, \beta$ for $\mathcal{L}_{\text {SelfCont }}, \gamma$ for $\mathcal{L}_{\text {Disc }}$, and $\lambda$ for $\mathcal{L}_{\mathrm{Cls}}$ ), as shown in equation (5):

$$
\begin{equation*}
\mathcal{L}_{\text {DACAD }}=\alpha \cdot \mathcal{L}_{\text {SupCont }}+\beta \cdot \mathcal{L}_{\text {SelfCont }}+\gamma \cdot \mathcal{L}_{\text {Disc }}+\lambda \cdot \mathcal{L}_{\mathrm{Cls}} \tag{5}
\end{equation*}
$$

Table 1: Statistics of the benchmark datasets used.

| Benchmark | \# datasets | \# dims | Train size | Test size | Anomaly ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MSL [Hundman et al., 2018] | 27 | 55 | 58,317 | 73,729 | $10.72 \%$ |
| SMAP [Hundman et al., 2018] | 55 | 25 | 140,825 | 444,035 | $13.13 \%$ |
| SMD [Su et al., 2019] | 28 | 38 | 708,405 | 708,420 | $4.16 \%$ |
| Boiler [Cai et al., 2021] | 3 | 274 | 277,152 | 277,152 | $15.00 \%$ |

The overall loss function $\mathcal{L}_{\text {DACAD }}$ is what the model seeks to optimise during the training process. By fine-tuning these weights ( $\alpha, \beta, \gamma$, and $\lambda$ ), the model can effectively balance the significance of each loss component. This balance is crucial as it allows the model to cater to specific task requirements.

### 3.7 DACAD's Inference

In the inference phase of the DACAD model, the primary objective is to identify anomalies in the target domain $T$. This is accomplished for each window $w^{t}$ in $T$. The anomaly detection is based on the concept of spatial separation in the feature space, as established during the training phase. The anomaly score for each window is derived from the spatial distance between the classifier's output and a predefined centre $c$ in the feature space. This score is calculated as the squared Euclidean distance, as shown in Equation 6:

$$
\begin{equation*}
\text { Anomaly Score }\left(w^{t}\right)=\left\|\phi^{C L}\left(\phi^{R}\left(w^{t}\right)\right)-c\right\|_{2}^{2} \tag{6}
\end{equation*}
$$

Where $\phi^{C L}\left(\phi^{R}\left(w^{t}\right)\right)$ denotes the feature representation of the window $w^{t}$ after processing by the representation layer $\phi^{R}$ and the classifier layer $\phi^{C L}$. The distance measured here quantifies how much each window in the target domain deviates from the "normal" pattern, as defined by the centre $c$.

The anomaly score is a crucial metric in this process. A higher score suggests a significant deviation from the normative pattern, indicating a higher likelihood of the window being an anomaly. Conversely, a lower score implies that the window's representation is closer to the centre, suggesting it is more likely to be normal.

In practical terms, the anomaly score can be thresholded to classify windows as either normal or anomalous. By effectively utilising these anomaly scores, the DACAD model provides a robust mechanism for identifying anomalous patterns in unlabelled target domain data.

## 4 Experiments

The aim of this section is to provide a comprehensive evaluation of DACAD, covering the experimental setups (Section 4.2) and results (Section 4.3 and 4.4), to clearly understand its performance and capabilities in different contexts.

### 4.1 Datasets

We evaluate the performance of the proposed model and make comparisons of the results across the four datasets, including the three most commonly used real benchmark datasets for TAD and the dataset used for time series domain adaptation. The datasets are summarised in Table 1.

### 4.2 Experimental Setup

In our study, we evaluate SOTA TAD models including OmniAnomaly [Su et al., 2019], TS2Vec [Yue et al., 2022], THOC [Shen et al., 2020], and DCdetector [Yang et al., 2023]
and SOTA UDA models that support multivariate time series classification including VRADA [Purushotham et al., 2016] and CLUDA [Ozyurt et al., 2023] on benchmark datasets previously mentioned in section 1 using their source code and best hyper-parameters as they stated to ensure a fair evaluation. The hyper-parameters used in our implementation are as follows: DACAD consists of a 3-layer TCN architecture with three different channel sizes [128, 256, 512] to capture temporal dependencies. The dimension of the representation is 1024. We use the same hyper-parameters across all datasets to evaluate DACAD: window size $(W S)=100$, margin $m=$ 1 , and we run our model for 20 epochs.

### 4.3 Baselines Comparison

Table 2 provides a comprehensive comparison of various models' performance on multivariate time series benchmark datasets, using F1 score, AUPR (Area Under the PrecisionRecall Curve), and AUROC (Area Under the Receiver Operating Characteristic Curve). Despite Point Adjustment (PA) popularity in recent years, we do not use PA when calculating these metrics due to Kim et al.'s findings that its application leads to an overestimation of a TAD model's capability and can bias results in favour of methods that produce extreme anomaly scores. Instead, we use conventional performance metrics for anomaly detection.

Benchmark datasets like SMD contain multiple time series that cannot be merged due to missing timestamps, making them unsuitable for averaging their F1 scores. The F1 score is a non-additive metric combining precision and recall. To address this, we compute individual confusion matrices for each time series. These matrices are then aggregated into a collective confusion matrix for the entire dataset. From this aggregated matrix, we calculate the overall precision, recall, and F1 score, ensuring an accurate and undistorted representation of the dataset's F1 score. For datasets with multiple time series, we present the mean and standard deviation of AUPR and AUROC for each series.

DACAD emerges as the top performer in Table 2, consistently achieving the best results across all scenarios and metrics, as highlighted in bold. This suggests its robustness and adaptability in handling normal and anomalous representations of time series for anomaly detection. VRADA and CLUDA models often rank as the second best, with their results underlined in several instances. Other models like OmniAnomaly, THOC, TS2Vec, and DCdetector demonstrate more uniform performance across various metrics but generally do not reach the top performance levels seen in DACAD, VRADA, or CLUDA. Their consistent but lower performance could make them suitable for applications where top-tier accuracy is less critical.

To assess our model against ContextDA [Lai et al., 2023] -the only exiting UDA for TAD model-, we use the same datasets and metrics reported in ContextDA's main paper, as its code is unavailable. On the SMD dataset, ContextDA achieves an average macro F1 of 0.63 and AUROC of 0.75, whereas our model achieves a higher average macro F1 of 0.81 and AUROC of 0.86 . Similarly, on the Boiler dataset, ContextDA's average macro F1 is 0.50 and AUROC 0.65, compared to our model's superior performance with an av- erage macro F1 of 0.63 and AUROC of 0.71. Details of these results are provided in the appendix, due to page limitations.

### 4.4 Ablation Study

Our ablation study focused on the following aspects: (1) Effect of the loss components, (2) Effect of CEC, and (3) Effect of anomaly injection.

Effect of the loss components: Table 3 offers several key insights into the impact of different loss components on the MSL dataset using F-5 as a source. Removing the target self-supervised CL (w/o $\mathcal{L}_{\text {SelfCont }}$ ) leads to lower metrics (F1: 0.481, AUPR: 0.495, AUROC: 0.697). Moreover, excluding source supervised CL (w/o $\mathcal{L}_{\text {SupCont }}$ ) reduces effectiveness (F1: 0.463, AUPR: 0.427, AUROC: 0.639), highlighting its role in capturing source-specific features which are crucial for the model's overall accuracy. Similarly, omitting the discriminator component results in performance reduction (F1: 0.471, AUPR: 0.484, AUROC: 0.699). However, the most significant decline occurs without the classifier (F1: 0.299, AUPR: 0.170, AUROC: 0.503), underscoring its crucial role in effectively distinguishing between nor$\mathrm{mal} /$ anomaly classes. Overall, the best results (F1: 0.595, AUPR: 0.554, AUROC: 0.787) are achieved with all components, highlighting the effectiveness of an integrated approach.

Overall, each component within the model plays a crucial role in enhancing its overall performance. The highest performance across all metrics (F1: 0.595, AUPR: 0.554, AUROC: 0.787 ) is achieved when all components are included.

To elucidate the effectiveness of UDA within DACAD, we examine the feature representations from the MSL dataset, as illustrated in Figure 2. It presents the t-SNE 2D embeddings of DACAD feature representations $\phi^{C L}\left(\phi^{R}(w)\right)$ for MSL dataset. Each point represents a time series window, which can be normal, anomalous or anomaly-injected. These representations highlight the domain discrepancies between source and target entities and demonstrate how DACAD aligns the time series window features effectively. The comparison of feature representations with and without UDA reveals a significant domain shift when UDA is not employed, between source (entity F-5) and target (entity T-5) within MSL dataset.

Effect of CEC: Table 4 compares the performance of our CEC classifier with two BCE and DeepSVDD - on the MSL dataset using F-5 as a source. Our proposed CEC shows superior performance compared to BCE and DeepSVDD across three metrics on the MSL dataset. With the highest F1 score, it demonstrates a better balance of precision and recall. Its leading performance in AUPR indicates greater effectiveness in identifying positive classes in imbalanced datasets. Additionally, CEC's higher AUROC suggests it is more capable of distinguishing between classes.

Effect of anomaly injection: We study the impact of anomaly injection in Table 5, on MSL dataset when using F-5 as a source. It shows that anomaly injection significantly improves all metrics (F1: 0.595, AUPR: 0.554, AUROC: 0.787), enhancing the model's ability to differentiate between normal and anomalous patterns, thereby improving DACAD's overall accuracy. Without anomaly injection, there's a notable decline in performance, emphasizing its role in precision. The

Table 2: F1, AUPR, and AUROC results for various models on multivariate time series benchmark datasets (SMD, MSL, SMAP). The most optimal evaluation results are displayed in bold, while the second best ones are indicated by underline.

| $\underset{2}{2}$ | $\overline{\mathrm{g}}$ <br> $\bar{v}$ | src | SMD |  |  |  | MSL |  |  |  | SMAP |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | $1-1$ | $2-3$ | $3-7$ | $1-5$ | F-5 | P-10 | D-14 | C-1 | A-7 | P-2 | E-8 | D-7 |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=198&width=35&top_left_y=360&top_left_x=191) | $\frac{\widehat{\hat{}}}{2}$ | F1 | $\underline{0.511}$ | 0.435 | 0.268 | 0.291 | 0.302 | 0.321 | 0.310 | 0.285 | 0.244 | 0.282 | 0.349 | 0.226 |
|  |  | AUPR | $\underline{0.523 \pm 0.223}$ | $0.433 \pm 0.172$ | $0.316 \pm 0.208$ | $0.292 \pm 0.147$ | $0.201 \pm 0.133$ | $0.199 \pm 0.159$ | $\underline{0.351 \pm 0.244}$ | $0.167 \pm 0.132$ | $0.133 \pm 0.196$ | $\underline{0.265 \pm 0.263}$ | $0.261 \pm 0.247$ | $0.120 \pm 0.149$ |
|  |  | AUROC | $302 \pm 0.130$ | $0.731 \pm 0.121$ | $0.597 \pm 0.146$ | $0.651 \pm 0.088$ | $0.526 \pm 0.054$ | $0.516 \pm 0.097$ | $0.615 \pm 0.207$ | $0.506 \pm 0.024$ | $0.474 \pm 0.176$ | $\underline{0.573 \pm 0.162}$ | $0.612 \pm 0.120$ | $0.443 \pm 0.170$ |
|  | $\hat{E}$ | FI | 0.435 | $\underline{0.487}$ | 0.320 | 0.314 | $\underline{0.395}$ | $\underline{0.368}$ | $\underline{0.324}$ | 0.312 | 0.292 | 0.278 | $\underline{0.384}$ | 0.293 |
|  |  | AUPR | $0.423 \pm 0.223$ | $\underline{0.494 \pm 0.174}$ | $\underline{0.400 \pm 0.201}$ | $\underline{0.328 \pm 0.140}$ | $\underline{0.325 \pm 0.263}$ | $0.239 \pm 1617$ | $0.318 \pm 0.276$ | $0.193 \pm 1640$ | $\underline{0.250 \pm 0.290}$ | $0.242 \pm 0.217$ | $\underline{0.332 \pm 0.263}$ | $\underline{0.193 \pm 0.220}$ |
|  |  | AUROC | $0.788 \pm 0.124$ | $\underline{0.794 \pm 0.121}$ | $\underline{0.725 \pm 0.155}$ | $\underline{0.693 \pm 0.119}$ | $\underline{0.579 \pm 0.204}$ | $0.569 \pm 0.132$ | $0.558 \pm 0.222$ | $0.479 \pm 0.164$ | $\underline{0.560 \pm 0.223}$ | $0.500 \pm 0.209$ | $\underline{0.692 \pm 0.171}$ | $0.480 \pm 0.222$ |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=506&width=43&top_left_y=559&top_left_x=190) | $\bar{E}$ | F | 0.397 | 0.413 | $\underline{0.419}$ | $\underline{0.421}$ | 0.254 | 0.266 | 0.246 | 0.242 | 0.320 | 0.326 | 0.331 | 0.318 |
|  |  | AUPR | $0.296 \pm 0.155$ | $0.309 \pm 0.165$ | $0.314 \pm 0.171$ | $0.318 \pm 0.182$ | $0.154 \pm 0.184$ | $0.154 \pm 0.184$ | $0.152 \pm 0.185$ | $0.148 \pm 0.186$ | $0.111 \pm 0.129$ | $0.114 \pm 0.130$ | $0.116 \pm 0.131$ | $0.112 \pm 0.131$ |
|  |  | AUROC | $0.619 \pm 0.153$ | $0.628 \pm 0.164$ | $0.639 \pm 0.156$ | $0.620 \pm 0.155$ | $0.392 \pm 0.173$ | $0.392 \pm 0.173$ | $0.398 \pm 0.165$ | $0.386 \pm 0.175$ | $0.421 \pm 0.189$ | $0.406 \pm 0.188$ | $0.406 \pm 0.190$ | $0.414 \pm 0.181$ |
|  | O | F1 | 0.156 | 0.167 | 0.168 | 0.168 | 0.307 | 0.317 | 0.309 | 0.310 | 0.321 | 0.328 | 0.335 | 0.320 |
|  |  | AUPR | $0.090 \pm 0.095$ | $0.106 \pm 0.129$ | $0.109 \pm 0.128$ | $0.109 \pm 0.128$ | $0.241 \pm 0.278$ | $\underline{0.247 \pm 0.275}$ | $0.244 \pm 0.277$ | $\underline{0.242 \pm 2778}$ | $0.191 \pm 0.263$ | $0.197 \pm 0.264$ | $0.200 \pm 0.266$ | $0.191 \pm 0.264$ |
|  |  | AUROC | $0.646 \pm 1.550$ | $0.646 \pm 0.154$ | $0.657 \pm 0.161$ | $0.648 \pm 0.158$ | $0.631 \pm 0.177$ | $\underline{0.637 \pm 0.182}$ | $\underline{0.637 \pm 0.182}$ | $\underline{0.640 \pm 0.183}$ | $0.535 \pm 0.211$ | $0.539 \pm 0.211$ | $0.543 \pm 0.211$ | $\underline{0.539 \pm 0.213}$ |
|  | $\stackrel{0}{\Delta}$ <br> $\stackrel{2}{c}$ | F1 | 0.171 | 0.173 | 0.173 | 0.173 | 0.320 | 0.316 | 0.317 | $\underline{0.313}$ | $\underline{0.362}$ | $\underline{0.368}$ | 0.374 | $\underline{0.362}$ |
|  |  | AUPR | $0.112 \pm 0.076$ | $0.112 \pm 0.076$ | $0.116 \pm 0.075$ | $0.116 \pm 0.075$ | $0.137 \pm 0.137$ | $0.138 \pm 0.136$ | $0.135 \pm 0.138$ | $0.133 \pm 0.138$ | $0.145 \pm 0.167$ | $0.147 \pm 0.167$ | $0.149 \pm 0.168$ | $0.143 \pm 0.165$ |
|  |  | AUROC | $0.492 \pm 0.046$ | $0.493 \pm 0.045$ | $0.489 \pm 0.044$ | $0.487 \pm 0.040$ | $0.511 \pm 0.096$ | $0.514 \pm 0.096$ | $0.514 \pm 0.096$ | $0.513 \pm 0.096$ | $0.504 \pm 0.088$ | $0.506 \pm 0.088$ | $0.507 \pm 0.089$ | $0.507 \pm 0.089$ |
|  | ![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=108&width=38&top_left_y=859&top_left_x=239) | $\mathrm{Fl}$ | 0.079 | 0.085 | 0.085 | 0.083 | 0.208 | 0.215 | 0.203 | 0.201 | 0.280 | 0.283 | 0.287 | 0.278 |
|  |  | AUPR | $0.041 \pm 0.036$ | $0.044 \pm 0.037$ | $0.044 \pm 0.037$ | $0.044 \pm 0.037$ | $0.124 \pm 0.139$ | $0.125 \pm 0.138$ | $0.122 \pm 0.140$ | $0.122 \pm 0.140$ | $0.128 \pm 0.158$ | $0.129 \pm 0.159$ | $0.132 \pm 0.159$ | $0.127 \pm 0.157$ |
|  |  | AUROC | $0.496 \pm 0.008$ | $0.495 \pm 0.008$ | $0.495 \pm 0.008$ | $0.495 \pm 0.008$ | $0.501 \pm 0.011$ | $0.501 \pm 0.011$ | $0.501 \pm 0.011$ | $0.501 \pm 0.011$ | $0.516 \pm 0.082$ | $0.513 \pm 0.079$ | $0.516 \pm 0.082$ | $0.517 \pm 0.082$ |
|  | $\hat{\imath}$ <br> Ú <br> 0 | F1 | 0.600 | 0.633 | 0.598 | 0.572 | 0.595 | 0.528 | 0.522 | 0.528 | 0.532 | 0.464 | 0.651 | 0.463 |
|  |  | AUPR | $0.528 \pm 0.245$ | $0.605 \pm 0.196$ | $0.565 \pm 0.211$ | $0.535 \pm 0.219$ | $0.554 \pm 0.268$ | $0.448 \pm 0.280$ | $0.475 \pm 0.304$ | $0.514 \pm 0.273$ | $\mathbf{0 . 4 8 3} \pm 0.288$ | $0.447 \pm 0.270$ | $0.550 \pm 0.270$ | $0.388 \pm 0.270$ |
|  |  | AUROC | $0.856 \pm 0.094$ | $0.858 \pm 0.083$ | $\mathbf{0 . 8 2 2} \pm 0.093$ | $0.813 \pm 0.098$ | $0.787 \pm 0.268$ | $\mathbf{0 . 7 4 8} \pm 0.142$ | $0.719 \pm 0.191$ | $0.769 \pm 0.151$ | $\mathbf{0 . 7 6 0} \pm 0.151$ | $0.687 \pm 0.202$ | $\mathbf{0 . 7 9 0} \pm \mathbf{0 . 1 0 5}$ | $0.619 \pm 0.234$ |

Table 3: Effect of loss components on MSL dataset (source: F-5).

| Component | F1 | AUPR | AUROC |
| :---: | :---: | :---: | :---: |
| DACAD | $\mathbf{0 . 5 9 5}$ | $\mathbf{0 . 5 5 4} \pm \mathbf{0 . 2 6 8}$ | $\mathbf{0 . 7 8 7} \pm \mathbf{0 . 2 6 8}$ |
| w/o $\mathcal{L}_{\text {SelfCont }}$ | 0.481 | $0.495 \pm 0.291$ | $0.697 \pm 0.203$ |
| w/o $\mathcal{L}_{\text {SupCont }}$ | 0.463 | $0.427 \pm 0.274$ | $0.639 \pm 0.198$ |
| w/o $\mathcal{L}_{\text {Disc }}$ | 0.471 | $0.484 \pm 0.315$ | $0.699 \pm 0.229$ |
| w/o $\mathcal{L}_{\text {Cls }}$ | 0.299 | $0.170 \pm 0.127$ | $0.503 \pm 0.018$ |

Table 4: Effect of CEC classifier on MSL dataset (source: F-5).

| Classifier | F1 | AUPR | AUROC |
| :---: | :---: | :---: | :---: |
| Our proposed CEC | $\mathbf{0 . 5 9 5}$ | $\mathbf{0 . 5 5 4} \pm \mathbf{0 . 2 6 8}$ | $\mathbf{0 . 7 8 7} \pm \mathbf{0 . 2 6 8}$ |
| BCE-based | 0.467 | $0.504 \pm 0.304$ | $0.670 \pm 0.241$ |
| DeepSVDD | 0.428 | $0.400 \pm 0.250$ | $0.595 \pm 0.185$ |

Table 5: Effect of anomaly injection on MSL dataset (source: F-5).

| Approach | F1 | AUPR | AUROC |
| :---: | :---: | :---: | :---: |
| with injection | $\mathbf{0 . 5 9 5}$ | $\mathbf{0 . 5 5 4} \pm \mathbf{0 . 2 6 8}$ | $\mathbf{0 . 7 8 7} \pm \mathbf{0 . 2 6 8}$ |
| w/o injection | 0.489 | $0.461 \pm 0.276$ | $0.682 \pm 0.666$ |

higher standard deviation in AUROC scores without injection suggests more variability and less stability in the model's performance. The study underscores the vital role of anomaly injection in improving anomaly detection models. It reveals that incorporating anomaly injection not only boosts detection accuracy but also enhances the model's overall stability and reliability.

## 5 Conclusion

The DACAD model stands as a notable innovation in the field of TAD, particularly effective in environments with limited labelled data. By melding domain adaptation and contrastive learning, it applies labelled data from one domain to detect anomalies in another. Its anomaly injection mechanism, introducing a spectrum of synthetic anomalies, significantly bolsters the model's adaptability and robustness across var-

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=731&width=637&top_left_y=1109&top_left_x=1205)

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=287&width=295&top_left_y=1128&top_left_x=1224)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=306&width=290&top_left_y=1473&top_left_x=1210)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=292&width=309&top_left_y=1128&top_left_x=1512)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_22d3ce01a9980333ad6fg-07.jpg?height=303&width=306&top_left_y=1472&top_left_x=1517)

(d)
Figure 2: Impact of UDA on DACAD feature representations. It contrasts the embeddings of (a) source entity F-5 with UDA, (b) target entity T-5 with UDA, (c) source entity F-5 without UDA, and (d) target entity T-5 without UDA.

ious domains. Our evaluations on diverse real-world datasets establish DACAD's superiority in handling domain shifts and outperforming existing models. Its capability to generalise and accurately detect anomalies, regardless of the scarcity of labelled data in the target domain, represents a significant contribution to TAD. In future work, we aim to refine the anomaly injection process further, enhancing the model's ability to simulate a broader range of anomalous patterns. Additionally, we plan to evolve the model to encompass univariate time series analysis, broadening its scope and utility.

## A Datasets' Descriptions

Mars Science Laboratory (MSL) and Soil Moisture Active Passive (SMAP) ${ }^{1}$ datasets [Hundman et al., 2018] are real-world datasets collected from NASA spacecraft. These datasets contain anomaly information derived from reports of incident anomalies for a spacecraft monitoring system. MSL and SMAP comprises of 27 and 55 datasets respectively and each equipped with a predefined train/test split, where, unlike other datasets, their training set are unlabeled.

Server Machine Dataset (SMD) ${ }^{2}$ [Su et al., 2019] is gathered from 28 servers, incorporating 38 sensors, over a span of 10 days. During this period, normal data was observed within the initial 5 days, while anomalies were sporadically injected during the subsequent 5 days. The dataset is also equipped with a predefined train/test split, where the training data is unlabeled.

Boiler Fault Detection Dataset. ${ }^{3}$ [Cai et al., 2021] The Boiler dataset includes sensor information from three separate boilers, with each boiler representing an individual domain. The objective of the learning process is to identify the malfunctioning blowdown valve in each boiler. Obtaining samples of faults is challenging due to their scarcity in the mechanical system. Therefore, it's crucial to make use of both labelled source data and unlabeled target data to enhance the model's ability to generalise.

## B Baselines

Below, we will provide an enhanced description of the UDA models for time series classification and anomaly detection.

Additionally, we provide a description of the four prominent and state-of-the-art TAD models that were used for comparison with DACAD. We have selected the models from different categories of TAD, namely, unsupervised reconstruction-based (OmniAnomaly [Su et al., 2019] model, unsupervised forecasting-based (THOC [Shen et al., 2020]), and self-supervised contrastive learning (TS2Vec [Yue et al., 2022] and DCdetector [Yang et al., 2023]) TAD models.

VRADA ${ }^{4}$ combines deep domain confusion [?] with variational recurrent adversarial deep domain adaptation [Purushotham et al., 2016], which simultaneously optimises source domain label prediction, MMD and domain discrimination with the latent representations generated by the LSTM encoder. Meanwhile, the reconstruction objective of AELSTM is performed to detect anomalies.

CLUDA $^{5}$ is a novel framework developed for UDA in time series data and evaluated on time series classification. This framework utilises a contrastive learning approach to learn domain-invariant semantics within multivariate time series data, aiming to preserve label information relevant to prediction tasks. CLUDA is distinctive as it is the first UDA frame-[^1]

work designed to learn contextual representations of time series data, maintaining the integrity of label information. Its effectiveness is demonstrated through evaluations of various time series classification datasets.

ContextDA ${ }^{6}$ is a sophisticated approach for detecting anomalies in time series data. It employs context sampling formulated as a Markov decision process and uses deep reinforcement learning to optimise the domain adaptation process. This model is designed to generate domain-invariant features for better anomaly detection across various domains. It has shown promise in transferring knowledge between similar or entirely different domains.

OmniAnomaly ${ }^{7}$ is a model operating on an unsupervised basis, employing a Variational Autoencoder (VAE) to handle multivariate time series data. It identifies anomalies by evaluating the reconstruction likelihood of specific data windows.

THOC ${ }^{8}$ utilises a multi-layer dilated recurrent neural network (RNN) alongside skip connections in order to handle contextual information effectively. It adopts a temporal hierarchical one-class network approach for detecting anomalies.

TS2 $\mathbf{V e c}^{9}$ is an unsupervised model that is capable of learning multiple contextual representations of MTS and UTS semantically at various levels. This model employs contrastive learning in a hierarchical way, which provides a contextual representation. A method within TS2Vec has been proposed for application in TAD.

DCdetector. ${ }^{10}$ is distinctive for its use of a dual attention asymmetric design combined with contrastive learning. Unlike traditional models, DCdetector does not rely on reconstruction loss for training. Instead, it utilises pure contrastive loss to guide the learning process. This approach enables the model to learn a permutation invariant representation of time series anomalies, offering superior discrimination abilities compared to other methods.

## C Extended Experiments

## C. 1 UDA Comparison

Table 6, adapted from [Lai et al., 2023], now includes the results of our model in its final column. This table provides a comprehensive comparison of the performances of various models on the SMD and Boiler datasets, as measured by Macro F1 and AUROC scores.

In this comparative analysis, the DACAD model distinctly outperforms others, consistently achieving the highest scores in both Macro F1 and AUROC across the majority of test cases in both datasets. Its performance is not only superior but also remarkably stable across different scenarios within each dataset, showcasing its robustness and adaptability to diverse data challenges.

Following DACAD, the ContexTDA model frequently ranks as the runner-up, particularly in the SMD dataset, where[^2]

Table 6: Macro F1/AUROC results on SMD and Boiler dataset. The most optimal evaluation results are displayed in bold, while the second best ones are indicated by underline.

| 慈 | $\mathbf{s r c} \mapsto \operatorname{trg}$ | Models (Macro F1/AUROC) |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | AE-MLP | AE_LSTM | RDC | VRADA | SASA | ContexTDA | DACAD |
| $\sum_{k}^{1}$ | $1-1 \mapsto 1-2$ | $0.72 / 0.83$ | $0.74 / 0.90$ | $0.74 / 0.89$ | $0.74 / 0.91$ | $0.59 / 0.63$ | $\underline{0.75} / 0.91$ | $\mathbf{0 . 8 0} 0.84$ |
|  | $1-1 \mapsto 1-3$ | $0.57 / 0.70$ | $0.49 / 0.41$ | $0.57 / 0.72$ | $0.49 / 0.41$ | $\underline{0.61} / \mathbf{0 . 9 0}$ | $0.57 / 0.75$ | $0.71 / \underline{0.84}$ |
|  | $1-1 \mapsto 1-4$ | $0.55 / 0.74$ | $0.54 / 0.41$ | $0.54 / 0.75$ | $0.54 / 0.41$ | $0.55 / 0.75$ | $\underline{0.59} / \underline{0.76}$ | $0.75 / \overline{0.83}$ |
|  | $1-1 \mapsto 1-5$ | $0.54 / 0.79$ | $0.55 / 0.84$ | $0.56 / 0.85$ | $0.55 / 0.79$ | $0.65 / 0.87$ | $\underline{0.66} / \underline{0.87}$ | $0.82 / 0.96$ |
|  | $1-1 \mapsto 1-6$ | $0.71 / 0.88$ | $0.71 / 0.91$ | $0.71 / 0.87$ | $0.71 / \underline{0.91}$ | $0.44 / 0.84$ | $\underline{0.73} / \overline{0.84}$ | $0.89 / 0.92$ |
|  | $1-1 \mapsto 1-7$ | $0.48 / 0.55$ | $0.48 / 0.50$ | $0.49 / 0.54$ | $0.48 / \overline{0.50}$ | $0.31 / \underline{0.57}$ | $\underline{0.51} / 0.53$ | $0.88 / 0.91$ |
|  | $1-1 \mapsto 1-8$ | $0.55 / 0.57$ | $0.53 / \underline{0.70}$ | $0.55 / 0.58$ | $0.54 / 0.56$ | $0.52 / 0.56$ | $\underline{0.58} / 0.58$ | $0.79 / 0.72$ |
| 玄 | $1 \mapsto 2$ | $0.44 / 0.64$ | $0.43 / 0.48$ | $0.43 / 0.54$ | $0.43 / 0.48$ | $\underline{0.53 / 0.88}$ | $0.50 / 0.59$ | $0.61 / 0.66$ |
|  | $1 \mapsto 3$ | $0.40 / 0.28$ | $0.40 / 0.11$ | $0.43 / 0.49$ | $0.42 / 0.18$ | $\overline{0.41} / 0.48$ | $\underline{0.50} / \underline{0.67}$ | $0.74 / \overline{0.81}$ |
|  | $2 \mapsto 1$ | $0.39 / 0.18$ | $0.40 / 0.21$ | $0.40 / 0.36$ | $0.40 / 0.15$ | $\underline{0.53} / \mathbf{0 . 9 0}$ | $\overline{0.51} / \overline{0.66}$ | $0.58 / \underline{0.65}$ |
|  | $2 \mapsto 3$ | $0.40 / 0.38$ | $0.40 / 0.20$ | $0.45 / 0.39$ | $0.42 / 0.21$ | $\overline{0.41} / 0.49$ | $0.50 / \mathbf{0 . 6 9}$ | $\mathbf{0 . 5 9} / \overline{0.63}$ |
|  | $3 \mapsto 1$ | $0.39 / 0.20$ | $0.40 / 0.16$ | $0.39 / 0.31$ | $0.40 / 0.15$ | $0.48 / 0.67$ | $\underline{0.51} / \underline{0.67}$ | $0.72 / \overline{0.84}$ |
|  | $3 \mapsto 2$ | $0.48 / 0.54$ | $0.49 / 0.48$ | $0.49 / 0.55$ | $0.49 / 0.48$ | $0.46 / 0.31$ | $\underline{0.50} / \overline{0.57}$ | $0.58 / 0.68$ |

it often secures the second-highest scores in either or both the Macro F1 and AUROC metrics. Interestingly, certain models exhibit a degree of specialization. For example, SASA, which underperforms in the SMD dataset, demonstrates notably better results in specific scenarios within the Boiler dataset, particularly in terms of the Macro F1 score. This suggests that some models may be more suited to specific types of data or scenarios.

In conclusion, while the DACAD model emerges as the most effective across most scenarios, the varying performances of different models across scenarios and datasets emphasize the need to carefully consider the unique characteristics and capabilities of each model when selecting the most appropriate one for a specific task. This nuanced approach is crucial in leveraging the strengths of each model to achieve optimal results.

## References

[Audibert et al., 2020] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A Zuluaga. Usad: Unsupervised anomaly detection on multivariate time series. In SIGKDD, pages 3395-3404, 2020.

[Audibert et al., 2022] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A Zuluaga. Do deep neural networks contribute to multivariate time series anomaly detection? Pattern Recognition, 132:108945, 2022.

[Cai et al., 2021] Ruichu Cai, Jiawei Chen, Zijian Li, Wei Chen, Keli Zhang, Junjian Ye, Zhuozhang Li, Xiaoyan Yang, and Zhenjie Zhang. Time series domain adaptation via sparse associative structure alignment. In $A A A I$, volume 35, pages 6859-6867, 2021.

[Chung et al., 2015] Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. Advances in neural information processing systems, 28, 2015.

[Creswell et al., 2018] Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and
Anil A Bharath. Generative adversarial networks: An overview. IEEE signal processing magazine, 35(1):53-65, 2018.

[Darban et al., 2022] Zahra Zamanzadeh Darban, Geoffrey I Webb, Shirui Pan, Charu C Aggarwal, and Mahsa Salehi. Deep learning for time series anomaly detection: A survey. arXiv preprint arXiv:2211.05244, 2022.

[Darban et al., 2023] Zahra Zamanzadeh Darban, Geoffrey I Webb, Shirui Pan, Charu C. Aggarwal, and Mahsa Salehi. Carla: Self-supervised contrastive representation learning for time series anomaly detection. arXiv preprint arXiv:2308.09296, 2023.

[Deng et al., 2021] Hao Deng, Yifan Sun, Ming Qiu, Chao Zhou, and Zhiqiang Chen. Graph neural network-based anomaly detection in multivariate time series data. In COMPSAC'2021, pages 1128-1133. IEEE, 2021.

[Franceschi et al., 2019] J.-Y. Franceschi, A. Dieuleveut, and M. Jaggi. Unsupervised scalable representation learning for multivariate time series. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[Ganin and Lempitsky, 2015] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, pages 1180-1189. PMLR, 2015.

[Hochreiter and Schmidhuber, 1997] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9:1735-1780, 1997.

[Hundman et al., 2018] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In SIGKDD, pages 387-395, 2018.

[Khosla et al., 2020] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661-18673, 2020.

[Kim et al., 2022] Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon. Towards a rigorous evaluation of time-series anomaly detection. In $A A A I$, volume 36, pages 7194-7201, 2022.

[Lai et al., 2023] Kwei-Herng Lai, Lan Wang, Huiyuan Chen, Kaixiong Zhou, Fei Wang, Hao Yang, and Xia Hu. Context-aware domain adaptation for time series anomaly detection. In Proceedings of the SDM'2023, pages 676684. SIAM, 2023.

[Lea et al., 2016] Colin Lea, Rene Vidal, Austin Reiter, and Gregory D Hager. Temporal convolutional networks: A unified approach to action segmentation. In Computer Vision-ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, pages 47-54. Springer, 2016.

[Lei et al., 2019] Q. Lei, J. Yi, R. Vaculin, L. Wu, and I. S. Dhillon. Similarity preserving representation learning for time series clustering. In IJCAI, volume 19, pages 2845$2851,2019$.

[Liu et al., 2022] Xiaofeng Liu, Chaehwa Yoo, Fangxu Xing, Hyejin Oh, Georges El Fakhri, Je-Won Kang, Jonghye Woo, et al. Deep unsupervised domain adaptation: A review of recent advances and perspectives. $A P$ SIPA Transactions on Signal and Information Processing, $11,2022$.

[Long et al., 2018] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. Advances in neural information processing systems, 31, 2018.

[Lu et al., 2021] Nannan Lu, Hanhan Xiao, Yanjing Sun, Min Han, and Yanfen Wang. A new method for intelligent fault diagnosis of machines based on unsupervised domain adaptation. Neurocomputing, 427:96-109, 2021.

[Niu et al., 2020] Zijian Niu, Ke Yu, and Xiaofei Wu. Lstmbased vae-gan for time-series anomaly detection. Sensors, 20:3738, 2020.

[Ozyurt et al., 2023] Yilmazcan Ozyurt, Stefan Feuerriegel, and $\mathrm{Ce}$ Zhang. Contrastive learning for unsupervised domain adaptation of time series. ICLR, 2023.

[Park et al., 2018] Daehyung Park, Yuuna Hoshi, and Charles C Kemp. A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder. IEEE Robotics and Automation Letters, 3:1544-1551, 2018.

[Purushotham et al., 2016] Sanjay Purushotham, Wilka Carvalho, Tanachat Nilanon, and Yan Liu. Variational recurrent adversarial deep domain adaptation. In ICLR, 2016.

[Ragab et al., 2020] Mohamed Ragab, Zhenghua Chen, Min Wu, Chuan Sheng Foo, Chee Keong Kwoh, Ruqiang Yan, and Xiaoli Li. Contrastive adversarial domain adaptation for machine remaining useful life prediction. IEEE Transactions on Industrial Informatics, 17:5239-5249, 2020.

[Ruff et al., 2018] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. Deep one-class classification. In ICML, pages 4393-4402. PMLR, 2018.

[Schmidl et al., 2022] Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock. Anomaly detection in time series: a comprehensive evaluation. Proceedings of the $V L D B, 15: 1779-1797,2022$.

[Schroff et al., 2015] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815-823, 2015.

[Shen et al., 2020] Lifeng Shen, Zhuocong Li, and James Kwok. Timeseries anomaly detection using temporal hierarchical one-class network. Advances in Neural Information Processing Systems, 33:13016-13026, 2020.

[Su et al., 2019] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In SIGKDD, pages 2828-2837, 2019.

[Tonekaboni et al., 2021] S. Tonekaboni, D. Eytan, and A. Goldenberg. Unsupervised representation learning for time series with temporal neighborhood coding. In ICLR, 2021.

[Tzeng et al., 2017] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages $7167-7176,2017$.

[Wang et al., 2021] Guijin Wang, Ming Chen, Zijian Ding, Jiawei Li, Huazhong Yang, and Ping Zhang. Inter-patient ecg arrhythmia heartbeat classification based on unsupervised domain adaptation. Neurocomputing, 454:339-349, 2021.

[Wilson and Cook, 2020] Garrett Wilson and Diane J Cook. A survey of unsupervised deep domain adaptation. ACM Transactions on Intelligent Systems and Technology (TIST), 11:1-46, 2020.

[Wilson et al., 2020] Garrett Wilson, Janardhan Rao Doppa, and Diane J Cook. Multi-source deep domain adaptation with weak supervision for time-series sensor data. In SIGKDD, pages 1768-1778, 2020.

[Xu et al., 2022] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In ICLR, 2022.

[Yang et al., 2023] Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. Dcdetector: Dual attention contrastive representation learning for time series anomaly detection. In SIGKDD, 2023.

[Yue et al., 2022] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series. In AAAI, volume 36, pages 8980-8987, 2022.

[Zerveas et al., 2021] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhoff. A transformerbased framework for multivariate time series representation learning. In SIGKDD, pages 2114-2124. Association for Computing Machinery, 2021.

[Zhang et al., 2022] Wen Zhang, Lingfei Deng, Lei Zhang, and Dongrui Wu. A survey on negative transfer. IEEE/CAA Journal of Automatica Sinica, 10(2):305-329, 2022.

[Zhao et al., 2021] Ranqi Zhao, Yi Xia, and Yongliang Zhang. Unsupervised sleep staging system based on domain adaptation. Biomedical Signal Processing and Control, 69:102937, 2021.

[Zhou et al., 2019] Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. Beatgan: Anomalous rhythm detection using adversarially generated time series. In $I J$ CAI, volume 2019, pages 4433-4439, 2019.


[^0]:    ${ }^{*}$ Corresponding Author

[^1]:    ${ }^{1}$ https://www.kaggle.com/datasets/patrickfleith/ nasa-anomaly-detection-dataset-smap-msl

    ${ }^{2}$ https://github.com/NetManAIOps/OmniAnomaly/tree/master/ ServerMachineDataset

    ${ }^{3}$ https://github.com/DMIRLAB-Group/SASA-pytorch/tree/ main/datasets/Boiler

    ${ }^{4}$ https://github.com/floft/vrada

    ${ }^{5}$ https://github.com/oezyurty/CLUDA

[^2]:    ${ }^{6}$ There is no implementation available for this model, so we rely on the results claimed in the paper.

    ${ }^{7}$ https://github.com/smallcowbaby/OmniAnomaly

    ${ }^{8}$ We utilised the authors' shared implementation, as it is not publicly available.

    ${ }^{9}$ https://github.com/yuezhihan/ts2vec

    ${ }^{10}$ https://github.com/DAMO-DI-ML/KDD2023-DCdetector

</end of paper 0>


<paper 1>
# Deep Learning for Time Series Anomaly Detection: A Survey 

ZAHRA ZAMANZADEH DARBAN*, Monash University, Australia<br>GEOFFREY I. WEBB, Monash University, Australia<br>SHIRUI PAN, Griffith University, Australia<br>CHARU C. AGGARWAL, IBM T. J. Watson Research Center, USA<br>MAHSA SALEHI, Monash University, Australia


#### Abstract

Time series anomaly detection is important for a wide range of research fields and applications, including financial markets, economics, earth sciences, manufacturing, and healthcare. The presence of anomalies can indicate novel or unexpected events, such as production faults, system defects, and heart palpitations, and is therefore of particular interest. The large size and complexity of patterns in time series data have led researchers to develop specialised deep learning models for detecting anomalous patterns. This survey provides a structured and comprehensive overview of state-of-the-art deep learning for time series anomaly detection. It provides a taxonomy based on anomaly detection strategies and deep learning models. Aside from describing the basic anomaly detection techniques in each category, their advantages and limitations are also discussed. Furthermore, this study includes examples of deep anomaly detection in time series across various application domains in recent years. Finally, it summarises open issues in research and challenges faced while adopting deep anomaly detection models to time series data.


CCS Concepts: $\cdot$ Computing methodologies $\rightarrow$ Anomaly detection; $\cdot$ General and reference $\rightarrow$ Surveys and overviews.

Additional Key Words and Phrases: Anomaly detection, Outlier detection, Time series, Deep learning, Multivariate time series, Univariate time series

## ACM Reference Format:

Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, Charu C. Aggarwal, and Mahsa Salehi. 2023. Deep Learning for Time Series Anomaly Detection: A Survey. 1, 1 (May 2023), 42 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 INTRODUCTION

The detection of anomalies, also known as outlier or novelty detection, has been an active research field in numerous application domains since the 1960s [72]. As computational processes evolve, the collection of big data and its use in artificial intelligence (AI) is better enabled, contributing to time series analysis including the detection of anomalies. With greater data availability and increasing algorithmic efficiency/computational power, time series analysis is increasingly used to address business applications through forecasting, classification, and anomaly detection [57], [23]. Time series anomaly detection (TSAD) has received increasing attention in recent years, because of increasing applicability in a wide variety of domains, including urban management, intrusion detection, medical risk, and natural disasters.[^0]

Deep learning has become increasingly capable over the past few years of learning expressive representations of complex time series, like multidimensional data with both spatial (intermetric) and temporal characteristics. In deep anomaly detection, neural networks are used to learn feature representations or anomaly scores in order to detect anomalies. Many deep anomaly detection models have been developed, providing significantly higher performance than traditional time series anomaly detection tasks in different real-world applications.

Although the field of anomaly detection has been explored in several literature surveys [26], [140], [24], [17], [20] and some evaluation review papers exist [153], [101], there is only one survey on deep anomaly detection methods for time series data [37]. However, the mentioned survey [37] has not covered the vast range of TSAD methods that have emerged in recent years, such as DAEMON [33], TranAD [171], DCT-GAN [114], and Interfusion [117]. Additionally, the representation learning methods within the taxonomy of TSAD methodologies has not been addressed in this survey. As a result, there is a need for a survey that enables researchers to identify important future directions of research in TSAD and the methods that are suitable to various application settings. Specifically, this article makes the following contributions:

- Taxonomy: We present a novel taxonomy of deep anomaly detection models for time series data. These models are broadly classified into four categories: forecasting-based, reconstruction-based, representation-based and hybrid methods. Each category is further divided into subcategories based on the deep neural network architectures used. This taxonomy helps to characterise the models by their unique structural features and their contribution to anomaly detection capabilities.
- Comprehensive Review: Our study provides a thorough review of the current state-of-the-art in time series anomaly detection up to 2024 . This review offers a clear picture of the prevailing directions and emerging trends in the field, making it easier for readers to understand the landscape and advancements.
- Benchmarks and Datasets: We compile and describe the primary benchmarks and datasets used in this field Additionally, we categorise the datasets into a set of domains and provide hyperlinks to these datasets, facilitating easy access for researchers and practitioners.
- Guidelines for Practitioners: Our survey includes practical guidelines for readers on selecting appropriate deep learning architectures, datasets, and models. These guidelines are designed to assist researchers and practitioners in making informed choices based on their specific needs and the context of their work.
- Fundamental Principles: We discuss the fundamental principles underlying the occurrence of different types of anomalies in time series data. This discussion aids in understanding the nature of anomalies and how they can be effectively detected.
- Evaluation Metrics and Interpretability: We provide an extensive discussion on evaluation metrics together with guidelines for metric selection. Additionally, we include a detailed discussion on model interpretability to help practitioners understand and explain the behaviour and decisions of TSAD models.

This article is organised as follows. In Section 2, we start by introducing preliminary definitions, which is followed by a taxonomy of anomalies in time series. Section 3 discusses the application of deep anomaly detection models to time series. Different deep models and their capabilities are then presented based on the main approaches (forecastingbased, reconstruction-based, representation-based, and hybrid) and architectures of deep neural networks. Additionally, Section D explores the applications of time series deep anomaly detection models in different domains. Finally, Section 5 provides several challenges in this field that can serve as future opportunities. An overview of publicly available and commonly used datasets for the considered anomaly detection models can be found in Section 4.

Manuscript submitted to ACM

## 2 BACKGROUND

A time series is a series of data points indexed sequentially over time. The most common form of time series is a sequence of observations recorded over time [75]. Time series are often divided into univariate (one-dimensional) and multivariate (multi-dimensional). These two types are defined in the following subsections. Thereafter, decomposable components of the time series are outlined. Following that, we provide a taxonomy of anomaly types based on time series' components and characteristics.

### 2.1 Univariate Time Series

As the name implies, a univariate time series (UTS) is a series of data that is based on a single variable that changes over time, as shown in Fig. 1a. Keeping a record of the humidity level every hour of the day would be an example of this. The time series $X$ with $t$ timestamps can be represented as an ordered sequence of data points in the following way:

$$
\begin{equation*}
X=\left(x_{1}, x_{2}, \ldots, x_{t}\right) \tag{1}
\end{equation*}
$$

where $x_{i}$ represents the data at timestamp $i \in T$ and $T=\{1,2, \ldots, t\}$.

### 2.2 Multivariate Time Series

Additionally, a multivariate time series (MTS) represents multiple variables that are dependent on time, each of which is influenced by both past values (stated as "temporal" dependency) and other variables (dimensions) based on their correlation. The correlations between different variables are referred to as spatial or intermetric dependencies in the literature, and they are used interchangeably [117]. In the same example, air pressure and temperature would also be recorded every hour besides humidity level.

An example of an MTS with two dimensions is illustrated in Fig. 1b. Consider a MTS represented as a sequence of vectors over time, each vector at time $i, X_{i}$, consisting of $d$ dimensions:

$$
\begin{equation*}
X=\left(X_{1}, X_{2}, \ldots, X_{t}\right)=\left(\left(x_{1}^{1}, x_{1}^{2}, \ldots, x_{1}^{d}\right),\left(x_{2}^{1}, x_{2}^{2}, \ldots, x_{2}^{d}\right), \ldots,\left(x_{t}^{1}, x_{t}^{2}, \ldots, x_{t}^{d}\right)\right) \tag{2}
\end{equation*}
$$

where $X_{i}=\left(x_{i}^{1}, x_{i}^{2}, \ldots, x_{i}^{d}\right)$ represents a data vector at time $i$, with each $x_{i}^{j}$ indicating the observation at time $i$ for the $j^{\text {th }}$ dimension, and $j=1,2, \ldots, d$, where $d$ is the total number of dimensions.

### 2.3 Time Series Decomposition

It is possible to decompose a time series $X$ into four components, each of which express a specific aspect of its movement [52]. The components are as follows:

- Secular trend: This is the long-term trend in the series, such as increasing, decreasing or stable. The secular trend represents the general pattern of the data over time and does not have to be linear. The change in population in a particular region over several years is an example of nonlinear growth or decay depending on various dynamic factors.
- Seasonal variations: Depending on the month, weekday, or duration, a time series may exhibit a seasonal pattern. Seasonality always occurs at a fixed frequency. For instance, a study of gas/electricity consumption shows that the consumption curve does not follow a similar pattern throughout the year. Depending on the season and the locality, the pattern is different.

Manuscript submitted to ACM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-04.jpg?height=285&width=750&top_left_y=343&top_left_x=365)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-04.jpg?height=307&width=746&top_left_y=318&top_left_x=1123)

(b)

Fig. 1. (a) An overview of different temporal anomalies plotted from the NeurIPS-TS dataset [107]. Global and contextual anomalies occur in a point (coloured in blue). Seasonal, trend and shapelet can occur in a subsequence (coloured in red). (b) Intermetric and temporal-intermetric anomalies in MTS. In this figure, metric 1 is power consumption, and metric 2 is CPU usage.

- Cyclical fluctuations: A cycle is defined as an extended deviation from the underlying series defined by the secular trend and seasonal variations. Unlike seasonal effects, cyclical effects vary in onset and duration. Examples include economic cycles such as booms and recessions.
- Irregular variations: This refers to random, irregular events. It is the residual after all the other components are removed. A disaster such as an earthquake or flood can lead to irregular variations.

A time series can be mathematically described by estimating its four components separately, and each of them may deviate from the normal behaviour.

### 2.4 Anomalies in Time Series

According to [77], the term anomaly refers to a deviation from the general distribution of data, such as a single observation (point) or a series of observations (subsequence) that deviate greatly from the general distribution. A small portion of the dataset contains anomalies, indicating the dataset mostly follows a normal pattern. There may be considerable amounts of noise embedded in real-world data, and such noise may be irrelevant to the researcher [4]. The most meaningful deviations are usually those that are significantly different from the norm. In circumstances where noise is present, the main characteristics of the data are identical. In data domains such as time series, trend analysis and anomaly detection are closely related, but they are not equivalent [4]. It is possible to see changes in time series datasets owing to concept drift, which occurs when values and trends change over time gradually or abruptly [128], [3].

2.4.1 Types of Anomalies. Anomalies in UTS and MTS can be classified as temporal, intermetric, or temporal-intermetric anomalies [117]. In a time series, temporal anomalies can be compared with either their neighbours (local) or the whole time series (global), and they present different forms depending on their behaviour [107]. There are several types of temporal anomalies that commonly occur in UTS, all of which are shown in Fig. 1a. Temporal anomalies can also occur in the MTS and affect multiple dimensions or all dimensions. A subsequent anomaly may appear when an unusual pattern of behaviour emerges over time; however, each observation may not be considered an outlier by itself. As a result of a point anomaly, an unexpected event occurs at one point in time, and it is assumed to be a short sequence. Different types of temporal anomalies are as follows:

- Global: They are spikes in the series, which are point(s) with extreme values compared to the rest of the series. A global anomaly, for instance, is an unusually large payment by a customer on a typical day. Considering a threshold, it can be described as Eq. (3).

$$
\begin{equation*}
\left|x_{t}-\hat{x}_{t}\right|>\text { threshold } \tag{3}
\end{equation*}
$$

Manuscript submitted to ACM
where $\hat{x}$ is the output of the model. If the difference between the output and actual point value is greater than a threshold, then it has been recognised as an anomaly. An example of a global anomaly is shown on the left side of Fig. 1a where -6 has a large deviation from the time series.

- Contextual: A deviation from a given context is defined as a deviation from a neighbouring time point, defined here as one that lies within a certain range of proximity. These types of anomalies are small glitches in sequential data, which are deviated values from their neighbours. It is possible for a point to be normal in one context while an anomaly in another. For example, large interactions, such as those on boxing day, are considered normal, but not so on other days. The formula is the same as that of a global anomaly, but the threshold for finding anomalies differs. The threshold is determined by taking into account the context of neighbours:

$$
\begin{equation*}
\text { threshold } \approx \lambda \times \operatorname{var}\left(X_{t-w: t}\right) \tag{4}
\end{equation*}
$$

where $X_{t-w: t}$ refers to the context of the data point $x_{t}$ with a window size $w$, var is the variance of the context of data point and $\lambda$ controlling coefficient for the threshold. The second blue highlight in Fig. 1a is a contextual anomaly that occurs locally in a specific context.

- Seasonal: In spite of normal shapes and trends of the time series, their seasonality is unusual compared to the overall seasonality. An example is the number of customers in a restaurant during a week. Such a series has a clear weekly seasonality, so it makes sense to look for deviations in this seasonality and process the anomalous periods individually.

$$
\begin{equation*}
\operatorname{diss}_{S}(S, \hat{S})>\text { threshold } \tag{5}
\end{equation*}
$$

where diss $_{S}$ is a function measuring the dissimilarity between two subsequences and $\hat{S}$ denotes the seasonality of the expected subsequences. As demonstrated in the first red highlight of Fig. 1a, the seasonal anomaly changes the frequency of a rise and drop of data in the particular segment.

- Trend: The event that causes a permanent shift in the data to its mean and produces a transition in the trend of the time series. While this anomaly preserves its cycle and seasonality of normality, it drastically alters its slope. Trends can occasionally change direction, meaning they may go from increasing to decreasing and vice versa. As an example, when a new song comes out, it becomes popular for a while, then it disappears from the charts like the segment in Fig. 1a where the trend is changed and is assumed as a trend anomaly. It is likely that the trend will restart in the future.

$$
\begin{equation*}
\operatorname{diss}_{t}(T, \hat{T})>\text { threshold } \tag{6}
\end{equation*}
$$

where $\hat{T}$ is the normal trend.

- Shapelet: Shapelet means a distinctive, time series subsequence pattern. There is a subsequence whose time series pattern or cycle differs from the usual pattern found in the rest of the sequence. Variations in economic conditions, like the total demand for and supply of goods and services, are often the cause of these fluctuations. In the short-run, these changes lead to periods of expansion and recession.

$$
\begin{equation*}
\operatorname{diss}_{c}(C, \hat{C})>\text { threshold } \tag{7}
\end{equation*}
$$

where $\hat{C}$ specifies the cycle or shape of expected subsequences. For example, the last highlight in Fig. 1a where the shape of the segment changed due to some fluctuations.

Having discussed various types of anomalies, we understand that these can often be characterised by the distance between the actual subsequence observed and the expected subsequence. In this context, dynamic time warping (DTW)

[134], which optimally aligns two time series, is a valuable method for measuring this dissimilarity. Consequently, DTW's ability to accurately calculate temporal alignments makes it a suitable tool for anomaly detection applications, as evidenced in several studies [15], [161]. Moreover, MTS is composed of multiple dimensions (a.k.a, metrics [117, 163]) that each describe a different aspect of a complex entity. Spatial dependencies (correlations) among dimensions within an entity, also known as intermetric dependencies, can be linear or nonlinear. MTS would exhibit a wide range of anomalous behaviour if these correlations were broken. An example is shown in the left part of Fig. 1b. The correlation between power consumption in the first dimension (metric 1) and CPU usage in the second dimension (metric 2) usage is positive, but it breaks about 100th of a second after it begins. Such an anomaly is named the intermetric anomaly in this study.

$$
\begin{equation*}
\max _{\forall j, k \in D, j \neq k} \operatorname{diss}_{\mathrm{corr}}\left(\operatorname{Corr}\left(X^{j}, X^{k}\right), \operatorname{Corr}\left(X_{t+\delta t_{j}: t+w+\delta t_{j}}^{j}, X_{t+\delta t_{k}: t+w+\delta t_{k}}^{k}\right)\right)>\text { threshold } \tag{8}
\end{equation*}
$$

where $X^{j}$ and $X^{k}$ are different dimensions of the MTS, Corr denotes the correlation function that measures the relationship between two dimensions, $\delta t_{j}$ and $\delta t_{k}$ are time shifts that adjust the comparison windows for dimensions $j$ and $k$, accommodating asynchronous events or delays between observations, $t$ is the starting point of the time window, $w$ is the width of the time window, indicating the duration over which correlations are assessed, diss ${ }_{\text {corr }}$ is a function that quantifies the divergence in correlation between the standard, long-term measurement and the dynamic, short-term measurement within the specified window, threshold is a predefined limit that determines when the divergence in correlations signifies an anomaly, and $D$ is the set of all dimensions within the MTS, with the comparison conducted between every unique pair $(j, k)$ where $j \neq k$.

Dimensionality reduction techniques, such as selecting a subset of critical dimensions based on domain knowledge or preliminary analysis, help manage the computational complexity that increases with the number of dimensions.

Where $X^{j}$ and $X^{k}$ are two different dimensions of MTS that are correlated, and corr measures the correlations between two dimensions. When this correlation deteriorates in the window $t: t+w$, it means that the coefficient deviates more than threshold from the normal coefficient.

Intermetric-temporal anomalies introduce added complexity and challenges in anomaly detection; however, they occasionally facilitate easier detection across temporal or various dimensional perspectives due to their simultaneous violation of intermetric and temporal dependencies, as illustrated on the right side of Fig. 1b.

## 3 TIME SERIES ANOMALY DETECTION METHODS

Traditional methods offer varied approaches to time series anomaly detection. Statistical-based methods [186] aim to learn a statistical model of the normal behaviour of time series. In clustering-based approaches [133], a normal profile of time series windows is learned, and the distance to the centroid of the normal clusters is considered as an anomaly score, or clusters with a small number of members are considered as anomaly clusters. Distances-based approaches are extensively studied [188], in which the distance of a window of time series to its nearest neighbours is considered as an anomaly score. Density-based approaches [50] estimate the density of data points and time series windows with low density are detected as anomalies.

In data with complex structures, deep neural networks are powerful for modelling temporal and spatial dependencies in time series. A number of scholars have explored their application to anomaly detection using various deep architectures, as illustrated in Fig 2.

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-07.jpg?height=203&width=1076&top_left_y=316&top_left_x=454)

Fig. 2. Deep Learning architectures used in time series anomaly detection

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-07.jpg?height=168&width=1057&top_left_y=623&top_left_x=469)

Fig. 3. General components of deep anomaly detection models in time series

### 3.1 Deep Models for Time Series Anomaly Detection

An overview of deep anomaly detection models in time series is shown in Fig. 3. In our study, deep models for anomaly detection in time series are categorised based on their main approach and architectures. There are two main approaches (learning component in Fig. 3) in the TSAD literature: forecasting-based and reconstruction-based. A forecasting-based model can be trained to predict the next time stamp, whereas a reconstruction-based model can be deployed to capture the embedding of time series data. A categorisation of deep learning architectures in TSAD is shown in Fig. 2.

The TSAD models are summarised in Table 1 and Table 2 based on the input dimensions they process, which are UTS and MTS, respectively. These tables give an overview of the following aspects of the models: Temporal/Spatial, Learning scheme, Input, Interpretability, Point/Sub-sequence anomaly, Stochasticity, Incremental, and Univariate support. However, Table 1 excludes columns for Temporal/Spatial, Interpretability, and Univariate support as these features pertain solely to MTS. Additionally, it lacks an Incremental column because no univariate models incorporate an incremental approach.

3.1.1 Temporal/Spatial. With a UTS as input, a model can capture temporal information (i.e., pattern), while with a MTS as input, it can learn normality through both temporal and spatial dependencies. Moreover, if the model input is an MTS in which spatial dependencies are captured, the model can also detect intermetric anomalies (shown in Fig. 1b).

3.1.2 Learning Schemes. In practice, training data tends to have a very small number of anomalies that are labelled. As a consequence, most of the models attempt to learn the representation or features of normal data. Based on anomaly definitions, anomalies are then detected by finding deviations from normal data. There are four learning schemes in the recent deep models for anomaly detection: unsupervised, supervised, semi-supervised, and self-supervised. These are based on the availability (or lack) of labelled data points. Supervised method employs a distinct method of learning the boundaries between anomalous and normal data that is based on all the labels in the training set. It can determine an appropriate threshold value that will be used for classifying all timestamps as anomalous if the anomaly score (Section 3.1) assigned to those timestamps exceeds the threshold. The problem with this method is that it is not applicable to many real-world applications because anomalies are often unknown or improperly labelled. In contrast, Unsupervised approach uses no labels and makes no distinction between training and testing datasets. These techniques are the most flexible since they rely exclusively on intrinsic features of the data. They are useful in streaming applications because they do not require labels for training and testing. Despite these advantages, researchers may encounter difficulties evaluating anomaly detection models using unsupervised methods. The anomaly detection problem is typically treated
as an unsupervised learning problem due to the inherently unlabelled nature of historical data and the unpredictable nature of anomalies. Semi-supervised anomaly detection in time series data may be utilised in cases where the dataset only consists of labelled normal data, unlike supervised methods that require a fully labelled dataset of both normal and anomalous points. Unlike unsupervised methods, which detect anomalies without any labelled data, semi-supervised TSAD relies on labelled normal data to define normal patterns and detect deviations as anomalies. This approach is distinct from self-supervised learning, where the model generates its own supervisory signal from the input data without needing explicit labels.

3.1.3 Input. A model may take an individual point (i.e., a time step) or a window (i.e., a sequence of time steps containing historical information) as an input. Windows can be used in order, also called sliding windows, or shuffled without regard to the order, depending on the application. To overcome the challenges of comparing subsequences rather than points, many models use representations of subsequences (windows) instead of raw data and employ sliding windows that contain the history of previous time steps that rely on the order of subsequences within the time series data. A sliding window extraction is performed in the preprocessing phase after other operations have been implemented, such as imputing missing values, downsampling or upsampling of the data, and data normalisation.

3.1.4 Interpretability. In interpretation, the cause of an anomalous observation is given. Interpretability is essential when anomaly detection is used as a diagnostic tool since it facilitates troubleshooting and analysing anomalies. MTS are challenging to interpret, and stochastic deep learning complicates the process even further. A typical procedure to troubleshoot entity anomalies involves searching for the top dimension that differs most from previously observed behaviour. In light of that, it is, therefore, possible to interpret a detected entity anomaly by analysing several dimensions with the highest anomaly scores.

3.1.5 Point/Subsequence anomaly. The model can detect either point anomalies or subsequence anomalies. A point anomaly is a point that is unusual when compared with the rest of the dataset. Subsequence anomalies occur when consecutive observations have unusual cooperative behaviour, although each observation is not necessarily an outlier on its own. Different types of anomalies are described in Section 2.4 and illustrated in Fig. 1a and Fig. 1b

3.1.6 Stochasticity. As shown in Tables 1 and 2, we investigate the stochasticity of anomaly detection models as well. Deterministic models can accurately predict future events without relying on randomness. Predicting something that is deterministic is easy because you have all the necessary data at hand. The models will produce the same exact results for a given set of inputs in this circumstance. Stochastic models can handle uncertainties in the inputs. Through the use of a random component as an input, you can account for certain levels of unpredictability or randomness.

3.1.7 Incremental. This is a machine learning paradigm in which the model's knowledge extends whenever one or more new observations appear. It specifies a dynamic learning strategy that can be used if training data becomes available gradually. The goal of incremental learning is to adapt a model to new data while preserving its past knowledge.

Moreover, the deep model processes the input in a step-by-step or end-to-end fashion (see Fig. 3). In the first category (step-by-step), there is a learning module followed by an anomaly scoring module. It is possible to combine the two modules in the second category to learn anomaly scores using neural networks as an end-to-end process. An output of these models may be anomaly scores or binary labels for inputs. Contrary to algorithms whose objective is to improve representations, DevNet [141], for example, introduces deviation networks to detect anomalies by leveraging a few labelled anomalies to achieve end-to-end learning for optimizing anomaly scores. End-to-end models in anomaly Manuscript submitted to ACM

Table 1. Univariate Deep Anomaly Detection Models in Time Series

| $\mathrm{A}^{1}$ | $\mathrm{MA}^{2}$ | Model | Year | $\mathrm{Su} / \mathrm{Un}^{3}$ | Input $^{4}$ | $\mathrm{P} / \mathrm{S}^{5}$ | $\mathrm{Stc}^{6}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-09.jpg?height=330&width=43&top_left_y=440&top_left_x=557) | RNN (3.2.1) | LSTM-AD [126] | 2015 | Un | $\mathrm{P}$ | Point |  |
|  |  | DeepLSTM [28] | 2015 | Semi | P | Point |  |
|  |  | LSTM RNN [19] | 2016 | Semi | $\mathrm{P}$ | Subseq |  |
|  |  | LSTM-based [56] | 2019 | Un | $\mathrm{W}$ | - |  |
|  |  | TCQSA [118] | 2020 | $\mathrm{Su}$ | P | - |  |
|  | $\operatorname{HTM}(3.2 .4)$ | Numenta HTM [5] | 2017 | Un | - | - |  |
|  |  | Multi HTM [182] | 2018 | Un | - | - |  |
|  | $\mathrm{CNN}(3.2 .2)$ | SR-CNN [147] | 2019 | Un | $\mathrm{W}$ | Point + Subseq |  |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-09.jpg?height=164&width=46&top_left_y=771&top_left_x=552) | VAE (3.3.2) | Donut [184] | 2018 | Un | $\mathrm{W}$ | Subseq | $\checkmark$ |
|  |  | Bagel [115] | 2018 | Un | $\mathrm{W}$ | Subseq | $\checkmark$ |
|  |  | Buzz [32] | 2019 | Un | $\mathrm{W}$ | Subseq | $\checkmark$ |
|  | $\mathrm{AE}(3.3 .1)$ | EncDec-AD [125] | 2016 | Semi | $\mathrm{W}$ | Point |  |

${ }^{1}$ A: Approach, ${ }^{2}$ MA: Main Architecture, ${ }^{3}$ Su/Un: Supervised/Unsupervised $\mid$ Values: [Su: Supervised, Un: Unsupervised, Semi: Semi-supervised, Self: Self-supervised], ${ }^{4}$ Input: $P:$ point / W: window, ${ }^{5}$ P/S: Point/Sub-sequence, ${ }^{6}$ Stc: Stochastic, " - " indicates a feature is not defined or mentioned.

detection are designed to directly output the final classification of data points or subsequences as normal or anomalous, which includes the explicit labelling of these points. In contrast, step-by-step models typically generate intermediate outputs at each stage of the analysis, such as anomaly scores for each subsequence or point. These scores then require additional post-processing, such as thresholding, to determine if an input is anomalous. Common methods for establishing these thresholds include Nonparametric Dynamic Thresholding (NDT) [92] and Peaks-Over-Threshold (POT) [158], which help convert scores into final labels.

An anomaly score is mostly defined based on a loss function. In most of the reconstruction-based approaches, reconstruction probability is used, and in forecasting-based approaches, the prediction error is used to define an anomaly score. An anomaly score indicates the degree of an anomaly in each data point. Anomaly detection can be accomplished by ranking data points according to anomaly scores $\left(A_{S}\right)$ and a decision score based on a threshold value:

$$
\begin{equation*}
\left|A_{S}\right|>\text { threshold } \tag{9}
\end{equation*}
$$

Evaluation metrics that are used in these papers are introduced in Appendix A

### 3.2 Forecasting-Based Models

The forecasting-based approach uses a learned model to predict a point or subsequence based on a point or a recent window. In order to determine how anomalous the incoming values are, the predicted values are compared to their actual values and their deviations are considered as anomalous values. Most forecasting methods use a sliding window to forecast one point at a time. This is especially helpful in real-world anomaly detection situations where normal behaviour is in abundance, but anomalous behaviour is rare.

It is worth mentioning that some previous works such as [124] use prediction error as a novelty quantification rather than an anomaly score. In the following subsections, different forecasting-based architectures are explained.

3.2.1 Recurrent Neural Networks (RNN). RNNs have internal memory, allowing them to process variable-length input sequences and retain temporal dynamics [2, 167]. An example of a simple RNN architecture is shown in Fig 4a. Recurrent units take the points of the input window $X_{t-w: t-1}$ and forecast the next timestamp $x_{t}^{\prime}$. The input sequence is processed

Table 2. Multivariate Deep Anomaly Detection Models in Time Series

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-10.jpg?height=1767&width=1274&top_left_y=393&top_left_x=488)

${ }^{1}$ A: Approach, ${ }^{2}$ MA: Main Architecture, ${ }^{3}$ T/S: Temporal/Spatial $\mid$ Values: [S:Spatial, T:Temporal, ST:Spatio-Temporal], ${ }^{4}$ Su/Un: Supervised/Unsupervised $\mid$ Values: [Su: Supervised, Un: Unsupervised, Semi: Semi-supervised, Self: Self-supervised], ${ }^{5}$ Input: P: point / W: window, ${ }^{6}$ Int: Interpretability, ${ }^{7} \mathrm{P} / \mathrm{S}$ : Point/Sub-sequence, ${ }^{8}$ Stc: Stochastic, ${ }^{9}$ Inc: Incremental, ${ }^{1} 0$ US: Univarite support, ${ }^{*}$ Models with more than one main architecture.," - " indicates a feature is not defined or mentioned.

Manuscript submitted to ACM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-11.jpg?height=390&width=461&top_left_y=323&top_left_x=255)

(a) RNN

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-11.jpg?height=390&width=464&top_left_y=323&top_left_x=771)

(b) LSTM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-11.jpg?height=390&width=463&top_left_y=323&top_left_x=1292)

(c) GRU

Fig. 4. An Overview of (a) Recurrent neural network (RNN), (b) Long short-term memory unit (LSTM), and (c) Gated recurrent unit (GRU). These models can predict $x_{t}^{\prime}$ by capturing the temporal information of a window of $w$ samples prior to $x_{t}$ in the time series. Using the error $\left|x_{t}-x_{t}^{\prime}\right|$, an anomaly score can be computed.

iteratively, timestamp by timestamp. Given input $x_{t-1}$ to the recurrent unit $o_{t-2}$ and an activation function like tanh, the output $x_{t}^{\prime}$ is calculated as follows:

$$
\begin{array}{r}
x_{t}^{\prime}=\sigma\left(W_{x^{\prime}} \cdot o_{t-1}+b_{x^{\prime}}\right) \\
o_{t-1}=\tanh \left(W_{o} \cdot x_{t-1}+U_{o} \cdot o_{t-2}+b_{h}\right) \tag{10}
\end{array}
$$

where $W_{x^{\prime}}, W_{o}, U_{o}$, and $b_{h}$ are the network parameters. The network learns long-term and short-term temporal dependencies using previous outputs as inputs.

LSTM networks extend RNNs with memory lasting thousands of steps [82], enabling superior predictions through long-term dependencies. An LSTM unit, illustrated in Fig. 4b, comprises cells, input gates, output gates, and forget gates. The cell remembers values for variable time periods, while the gates control the flow of information.

In LSTM processing, the forget gate $f_{t-1}$ is calculated as:

$$
\begin{align*}
f_{t-1} & =\sigma\left(W_{f} \cdot x_{t-1}+U_{f} \cdot o_{t-2}\right)  \tag{11}\\
i_{t-1} & =\sigma\left(W_{i} \cdot x_{t-1}+U_{i} \cdot o_{t-2}\right)  \tag{12}\\
s_{t-1} & =\sigma\left(W_{s} \cdot x_{t-1}+U_{s} \cdot o_{t-2}\right) \tag{13}
\end{align*}
$$

Next, the candidate cell state $\tilde{c_{t-1}}$ is updated as:

$$
\begin{array}{r}
c_{t-1}=\tanh \left(W_{c} \cdot x_{t-1}+U_{c} \cdot o_{t-2}\right)  \tag{14}\\
c_{t-1}=i_{t-1} \cdot c_{t-1}+f_{t-1} \cdot c_{t-2}
\end{array}
$$

Finally, the hidden state $o_{t-1}$ or output is:

$$
\begin{equation*}
o_{t-1}=\tanh \left(c_{t-1}\right) \cdot s_{t-1} \tag{15}
\end{equation*}
$$

Where $W$ and $U$ are the parameters of the LSTM cell. $x_{t}^{\prime}$ is finally calculated using Equation 10 .

Experience with LSTM has shown that stacking recurrent hidden layers with sigmoidal activation units effectively captures the structure of time series data, allowing for processing at different time scales compared to other deep learning architectures [80]. LSTM-AD [126] possesses long-term memory capabilities and combines hierarchical recurrent layers
to detect anomalies in UTS without using labelled data for training. This stacking helps learn higher-order temporal patterns without needing prior knowledge of their duration. The network predicts several future time steps to capture the sequence's temporal structure, resulting in multiple error values for each point in the sequence. These prediction errors are modelled as a multivariate Gaussian distribution to assess the likelihood of anomalies. LSTM-AD's results suggest that LSTM-based models are more effective than RNN-based models, especially when it's unclear whether normal behaviour involves long-term dependencies.

As opposed to the stacked LSTM used in LSTM-AD, Bontemps et al. [19] propose a simpler LSTM RNN model for collective anomaly detection based on its predictive abilities for UTS. First, an LSTM RNN is trained with normal time series data to make predictions, considering both current states and historical data. By introducing a circular array, the model detects collective anomalies by identifying prediction errors that exceed a certain threshold within a sequence.

Motivated by promising results in LSTM models for UTS anomaly detection, a number of methods attempt to detect anomalies in MTS based on LSTM architectures. In DeepLSTM [28], stacked LSTM recurrent networks are trained on normal time series data. The prediction errors are then fitted to a multivariate Gaussian using maximum likelihood estimation. This model predicts both normal and anomalous data, recording the Probability Density Function (PDF) values of the errors. This approach has the advantage of not requiring preprocessing, and it works directly on raw time series. LSTM-PRED [66] utilises three LSTM stacks with 100 hidden units each, processing data sequences of 100 seconds to learn temporal dependencies. Instead of setting thresholds for each sensor, it uses the Cumulative Sum (CUSUM) method to detect anomalies. CUSUM calculates the cumulative sum of the sequence predictions to identify small deviations, reducing false positives. It computes the positive and negative differences between predicted and actual values, setting Upper Control Limits (UCL) and Lower Control Limits (LCL) from the validation data to determine anomalies. Moreover, this model can pinpoint the specific sensor showing abnormal behaviour.

In all three above-mentioned models, LSTMs are stacked to improve prediction accuracy by analysing historical data from MTS; however, LSTM-NDT [92] combines various techniques. LSTM-NDT model introduces a technique that automatically adjusts thresholds for data changes, addressing issues like diversity and instability in evolving data. Another model, called LGMAD [49], enhances LSTM's structure for better anomaly detection in time series. Additionally, a method combines LSTM with a Gaussian Mixture Model (GMM) for detecting anomalies in both simple and complex systems, with a focus on assessing the system's health status through a health factor. This model can only be applied in low-dimensional applications. For high-dimensional data, it's suggested to use dimension reduction methods like PCA for effective anomaly detection [88].

Ergen and Kozat [56] present LSTM-based anomaly detection algorithms in an unsupervised framework, as well as semi-supervised and fully supervised frameworks. To detect anomalies, it uses scoring functions implemented by One Class-SVM (OC-SVM) and Support Vector Data Description (SVDD) algorithms. In this framework, LSTM and OC-SVM (or SVDD) architecture parameters are jointly trained with well-defined objective functions, utilising two joint optimisation approaches. The gradient-based joint optimisation method uses revised OC-SVM and SVDD formulations, illustrating their convergence to the original formulations. As a result of the LSTM-based structure, methods are able to process data sequences of variable length. Aside from that, the model is effective at detecting anomalies in time series data without preprocessing. Moreover, since the approach is generic, the LSTM architecture in this model can be replaced by a GRU (gated recurrent neural networks) architecture [38].

GRU was proposed by Cho et al. [36] in 2014, similar to LSTM but incorporating a more straightforward structure that leads to less computing time (see Fig. 4c). Both LSTM and GRU use gated architectures to control information flow. However, GRU has gating units that inflate the information flow inside the unit without having any separate Manuscript submitted to ACM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-13.jpg?height=217&width=919&top_left_y=320&top_left_x=538)

Fig. 5. Structure of a Convolutional Neural Network (CNN) predicting the next values of an input time series based on a previous data window. Time series dependency dictates that predictions rely solely on previously observed inputs.

memory unit, unlike LSTM [47]. There is no output gate but an update gate and a reset gate. Fig. 4c shows the GRU cell that integrates the new input with the previous memory using its reset gate. The update gate defines how much of the last memory to keep [73]. The issue is that LSTMs and GRUs are limited in learning complex seasonal patterns in multi-seasonal time series. As more hidden layers are stacked and the backpropagation distance (through time) is increased, accuracy can be improved. However, training may be costly.

In this regard, the AD-LTI model is a forecasting tool that combines a GRU network with a method called Prophet to learn seasonal time series data without needing labelled data. It starts by breaking down the time series to highlight seasonal trends, which are then specifically fed into the GRU network for more effective learning. When making predictions, the model considers both the overall trends and specific seasonal patterns like weekly and daily changes. However, since it uses past data that might include anomalies, the projections might not always be reliable. To address this, it introduces a new measure called Local Trend Inconsistency (LTI), which assesses the likelihood of anomalies by comparing recent predictions against the probability of them being normal, overcoming the issue that there might be anomalous frames in history.

Traditional one-class classifiers are developed for fixed-dimension data and struggle with capturing temporal dependencies in time series data [149]. A recent model, called THOC [156], addresses this by using a complex network that includes a multilayer dilated RNN [27] and hierarchical SVDD [165]. This setup allows it to capture detailed temporal features at multiple scales (resolution) and efficiently recognise complex patterns in time series data. It improves upon older models by using information from various layers, not just the simplest features, and it detects anomalies by comparing current data against its normal pattern representation. In spite of the accomplishments of RNNs, they still face challenges in processing very long sequences due to their fixed window size.

3.2.2 Convolutional Neural Networks (CNN). Convolutional Neural Networks (CNNs) are adaptations of multilayer perceptrons designed to identify hierarchical patterns in data. These networks employ convolutional, pooling, and fully connected layers, as depicted in Fig. 5. Convolutional layers utilise a set of learnable filters that are applied across the entire input to produce 2D activation maps through dot products. Pooling layers summarise these outputs statistically.

The CNN-based DeepAnt model [135] efficiently detects small deviations in time series patterns with minimal training data and can handle data contamination under $5 \%$ in an unsupervised setup. DeepAnt is applicable to both UTS and MTS and detects various anomaly types, including point, contextual anomalies, and discords.

Despite their effectiveness, traditional CNNs struggle with sequential data due to their inherent design. This limitation has been addressed by the development of Temporal Convolutional Networks (TCN) [11], which use dilated convolutions to accommodate time series data. TCNs ensure that outputs are the same length as inputs without future data leakage. This is achieved using a 1D fully convolutional network and dilated convolutions, ensuring all computations for a

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-14.jpg?height=496&width=1046&top_left_y=321&top_left_x=602)

Fig. 6. The basic structure of Graph Neural Network (GNN) for MTS anomaly detection that can learn the relationships (correlations) between metrics and predict the expected behaviour of time series.

timestamp $t$ use only historical data. The dilated convolution operation is defined as:

$$
\begin{equation*}
x^{\prime}(t)=\left(x *_{l} f\right)(t)=\sum_{i=0}^{k-1} f(i) \cdot x_{t-l \cdot i} \tag{16}
\end{equation*}
$$

where $f$ is a filter of size $k, *_{l}$ denotes convolution with dilation factor $l$, and $x_{t-l \cdot i}$ represents past data points.

He and Zhao [78] use different methods to predict and detect anomalies in data over time. They use a TCN trained on normal data to forecast trends and calculate anomaly scores using multivariate Gaussian distribution fitted to prediction errors. It includes a skipping connection to blend multi-scale features, accommodating different pattern sizes. Ren et al. [147] combines a Spectral Residual model, originally for visual saliency detection [83], with a CNN to enhance accuracy. This method, used by over 200 Microsoft teams, can rapidly detect anomalies in millions of time series per minute. The TCN Autoencoder (TCN-AE), developed by Thill et al. [169] (2020), modifies the standard AE by using CNNs instead of dense layers, making it more effective and adaptable. It uses two TCNs for encoding and decoding, with layers that respectively downsample and upsample data.

Many real-world scenarios produce quasi-periodic time series (QTS), like the patterns seen in ECGs (electrocardiograms). A new automated system for spotting anomalies in these QTS called AQADF [118], uses a two-part method. First, it segments the QTS into consistent periods using an algorithm (TCQSA) that uses a hierarchical clustering technique and groups similar data points without needing manual help, even filtering out errors to make it more reliable. Second, it analyses these segments with an attention-based hybrid LSTM-CNN model (HALCM), which looks at both broad trends and detailed features in the data. Furthermore, HALCM is further enhanced by three attention mechanisms, allowing it to capture more precise details of the fluctuation patterns in QTS. Specifically, TAGs are embedded in LSTMs in order to fine-tune variations extracted from different parts of QTS. A feature attention mechanism and a location attention mechanism are embedded into a CNN in order to enhance the effects of key features extracted from QTSs.

TimesNet [181] is a versatile deep learning model designed for comprehensive time series analysis. It transforms 1D time series data into 2D tensors to effectively capture complex temporal patterns. By using a modular structure called TimesBlock, which incorporates a parameter-efficient inception block, TimesNet excels in a variety of tasks, including forecasting, classification, and anomaly detection. This innovative approach allows it to handle intricate variations in time series data, making it suitable for applications across different domains.

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-15.jpg?height=113&width=883&top_left_y=323&top_left_x=556)

(a) Components of anomaly detection using HTM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-15.jpg?height=255&width=761&top_left_y=474&top_left_x=617)

(b) Structure of HTM cell

Fig. 7. (a) Components of an HTM-based (Hierarchical Temporal Memory) anomaly detection system calculating prediction error and anomaly likelihood. (b) An HTM cell internal structure. Dendrites act as detectors with synapses. Context dendrites receive lateral input from other neurons. Sufficient lateral activity puts the cell in a predicted state.

3.2.3 Graph Neural Networks (GNN). In recent years, researchers have proposed extracting spatial information from MTS to form a graph structure, converting TSAD into a problem of detecting anomalies based on these graphs using GNNs. As shown in Fig. 6, GNNs use pairwise message passing, where graph nodes iteratively update their representations by exchanging information. In MTS anomaly detection, each dimension is a node in the graph, represented as $V=\{1, \ldots, d\}$. Edges $E$ indicate correlations learned from MTS. For node $u \in V$, the message passing layer outputs for iteration $k+1$ :

$$
\begin{align*}
& h_{u}^{k+1}=\operatorname{UPDATE}^{k}\left(h_{u}^{k}, m_{N(u)}^{k}\right) \\
m_{N(u)}^{k}= & \operatorname{AGGREGATE}^{k}\left(h_{i}^{k}, \forall i \in N(u)\right) \tag{17}
\end{align*}
$$

where $h_{u}^{k}$ is the embedding for each node and $N(u)$ is the neighbourhood of node $u$. GNNs enhance MTS modelling by learning spatial structures [151]. Various GNN architectures exist, such as Graph Convolution Networks (GCN) [103], which aggregate one-step neighbours, and Graph Attention Networks (GAT) [173], which use attention functions to compute different weights for each neighbour.

Incorporating relationships between features is beneficial. Deng and Hooi [45] introduced GDN, a GNN attentionbased model that captures sensor characteristics as nodes and their correlations as edges, predicting behaviour based on adjacent sensors. Anomaly detection framework GANF (Graph-Augmented Normalizing Flow) [40] augments normalizing flow with graph structure learning, detecting anomalies by identifying low-density instances. GANF represents time series as a Bayesian network, learning conditional densities with a graph-based dependency encoder and using graph adjacency matrix optimisation [189].

In conclusion, extracting graph structures from time series and modelling them with GNNs enables the detection of spatial changes over time, representing a promising research direction.

3.2.4 Hierarchical Temporal Memory (HTM). Hierarchical Temporal Memory (HTM) mimics the hierarchical processing of the neocortex for anomaly detection [65]. Fig. 7a shows the typical components of the HTM. The input $x_{t}$ is encoded and then processed through sparse spatial pooling [39], resulting in $a\left(x_{t}\right)$, a sparse binary vector. Sequence memory models temporal patterns in $a\left(x_{t}\right)$ and returns a sparse vector prediction $\pi\left(x_{t}\right)$. The prediction error is defined as:

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-16.jpg?height=377&width=1130&top_left_y=318&top_left_x=560)

Fig. 8. Transformer network structure for anomaly detection. The Transformer uses an encoder-decoder structure with multiple identical blocks. Each encoder block includes a multi-head self-attention module and a feedforward network. During decoding, cross-attention is added between the self-attention module and the feedforward network.

$$
\begin{equation*}
e r r_{t}=1-\frac{\pi\left(x_{t-1}\right) \cdot a\left(x_{t}\right)}{\left|a\left(x_{t}\right)\right|} \tag{18}
\end{equation*}
$$

where $\left|a\left(x_{t}\right)\right|$ is the number of $1 \mathrm{~s}$ in $a\left(x_{t}\right)$. Anomaly likelihood, based on the model's prediction history and error distribution, indicates whether the current state is anomalous.

HTM neurons are organised in columns within a layer (Fig. 7b). Multiple regions exist within each hierarchical level, with fewer regions at higher levels combining patterns from lower levels to recognise more complex patterns. Sensory data enters lower-level regions during learning and generates patterns for higher levels. HTM is robust to noise, has high capacity, and can learn multiple patterns simultaneously. It recognises and memorises frequent spatial input patterns and identifies sequences likely to occur in succession.

Numenta HTM [5] detects temporal anomalies of UTS in predictable and noisy environments. It effectively handles extremely noisy data, adapts continuously to changes, and can identify small anomalies without false alarms. MultiHTM [182] learns context over time, making it noise-tolerant and capable of real-time predictions for various anomaly detection challenges, so it can be used as an adaptive model. In particular, it is used for univariate problems and applied efficiently to MTS. RADM [48] proposes a real-time, unsupervised framework for detecting anomalies in MTS by combining HTM with a naive Bayesian network. Initially, HTM efficiently identifies anomalies in UTS with excellent results in terms of detection and response times. Then, it pairs with a Bayesian network to improve MTS anomaly detection without needing to reduce data dimensions, catching anomalies missed in UTS analyses. Bayesian networks help refine observations due to their adaptability and ease in calculating probabilities.

3.2.5 Transformers. Transformers [172] are deep learning models that weigh input data differently depending on the significance of different parts. In contrast to RNNs, transformers process the entire data simultaneously. Due to its architecture based solely on attention mechanisms, illustrated in Fig. 8, it can capture long-term dependencies while being computationally efficient. Recent studies utilise them to detect time series anomalies as they process sequential data for translation in text data.

The original transformer architecture is encoder-decoder-based. An essential part of the transformer's functionality is its multi-head self-attention mechanism, stated in the following equation:

$$
\begin{equation*}
Q, K, V=\operatorname{softmax}\left(\frac{Q K^{\mathrm{T}}}{\sqrt{d_{k}}}\right) V \tag{19}
\end{equation*}
$$

where $Q, K$ and $V$ are defined as the matrices and $d_{k}$ is for normalisation of attention map.

Manuscript submitted to ACM

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-17.jpg?height=87&width=303&top_left_y=325&top_left_x=694)

(a) Predictable

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-17.jpg?height=89&width=312&top_left_y=324&top_left_x=993)

(b) Unpredictable

Fig. 9. A time series may be unknown at any given moment or may change rapidly like (b), which illustrates sensor readings for manual control [125]. Such a time series cannot be predicted in advance, making prediction-based anomaly detection ineffective.

A semantic correlation is identified in a long sequence, filtering out unimportant elements. Since transformers lack recurrence or convolution, they need positional encoding for token positions (i.e. relative or absolute positions). GTA [34] uses transformers for sequence modelling and a bidirectional graph to learn relationships among multiple IoT sensors. It introduces an Influence Propagation (IP) graph convolution for semi-supervised learning of sensor dependencies. To boost efficiency, each node's neighbourhood is constrained, and then graph convolution layers model information flow. As a next step, a multiscale dilated convolution and graph convolution are fused for hierarchical temporal context encoding. They use transformers for parallelism and contextual understanding and propose multi-branch attention to reduce attention complexity. In another recent work, SAnD [160] uses a transformer with stacked encoder-decoder structures, relying solely on attention mechanisms to model clinical time series. The architecture utilises self-attention to capture dependencies with multiple heads, positional encoding, and dense interpolation embedding for temporal order. It was also extended for multitask diagnoses.

### 3.3 Reconstruction-Based Models

Many complex TSAD methods are designed around modelling the time series to predict future values, using prediction errors as indicators of anomalies. However, forecasting-based models often struggle with rapidly and continuously changing time series, as seen in Fig. 9, where the future states of a series may be unpredictable due to rapid changes or unknown elements [68]. In such cases, these models tend to generate increased prediction errors as the number of time points grows [126], limiting their utility primarily to very short-term predictions. For example, in financial markets, forecasting-based methods might predict only the next immediate step, which is insufficient in anticipating or mitigating a potential financial crisis.

In contrast, reconstruction-based models can offer more accurate anomaly detection because they have access to current time series data, which is not available to forecasting-based models. This access allows them to effectively reconstruct a complete scenario and identify deviations. While these models might cause some delay in detection, they are preferred when high accuracy is paramount, and some delay is acceptable. Thus, reconstruction-based models are better suited for applications where precision is critical, even if it results in a minor delay in response.

Models for normal behaviour are constructed by encoding subsequences of normal training data in latent spaces (low dimensions). Model inputs are sliding windows (see Section 3) that provide the temporal context. We presume that the anomalous subsequences are less likely to be reconstructed compared to normal subsequences in the test phase since anomalies are rare. As a result, anomalies are detected by reconstructing a point/sliding window from test data and comparing them to the actual values, which is called reconstruction error. In some models, the detection of anomalies is triggered when the reconstruction probability is below a specified threshold since anomalous points/subsequences have a low reconstruction probability.

3.3.1 Autoencoder (AE). Autoencoders (AEs), also known as auto-associative neural networks [105], are widely used in MTS anomaly detection for their nonlinear dimensionality reduction capabilities [150, 203]. Recent advancements in deep learning have focused on learning low-dimensional representations (encoding) using AEs [16, 81].

AEs consist of an encoder and a decoder (see Fig. 10a). The encoder converts input into a low-dimensional representation, and the decoder reconstructs the input from this representation. The goal is to achieve accurate reconstruction and minimise reconstruction error. This process is summarised as follows:

$$
\begin{equation*}
Z_{t-w: t}=\operatorname{Enc}\left(X_{t-w: t}, \phi\right), \quad \hat{X}_{t-w: t}=\operatorname{Dec}\left(Z_{t-w: t}, \theta\right) \tag{20}
\end{equation*}
$$

where $X_{t-w: t}$ is a sliding window of input data, $x_{t} \in \mathbb{R}^{d}$, Enc is the encoder with parameters $\phi$, and Dec is the decoder with parameters $\theta . Z$ represents the latent space (encoded representation). The encoder and decoder parameters are optimised during training to minimise reconstruction error:

$$
\begin{equation*}
\left(\phi^{*}, \theta^{*}\right)=\arg \min _{\phi, \theta} \operatorname{Err}\left(X_{t-w: t}, \operatorname{Dec}\left(\operatorname{Enc}\left(X_{t-w: t}, \phi\right), \theta\right)\right) \tag{21}
\end{equation*}
$$

To improve representation, techniques such as Sparse Autoencoder (SAE) [137], Denoising Autoencoder (DAE) [174], and Convolutional Autoencoder (CAE) [139] are used. The anomaly score of a window in an AE-based model is defined based on the reconstruction error:

$$
\begin{equation*}
A S_{w}=\left\|X_{t-w: t}-\operatorname{Dec}\left(\operatorname{Enc}\left(X_{t-w: t}, \phi\right), \theta\right)\right\|^{2} \tag{22}
\end{equation*}
$$

There are several papers in this category in our study. Sakurada and Yairi [150] shows how AEs can be used for dimensionality reduction in MTS as a preprocessing step for anomaly detection. They treat each data sample at each time index as independent, disregarding the time sequence. Even though AEs already perform well without temporal information, they can be further boosted by providing current and past samples. The authors compare linear PCA, Denoising Autoencoders (DAEs), and kernel PCA, finding that AEs can detect anomalies that linear PCA is incapable of detecting. DAEs further enhance AEs. Additionally, AEs avoid the complex computations of kernel PCA without losing quality in detection. DAGMM (Deep Autoencoding Gaussian Mixture Model) [203] estimates the probability of MTS input samples using a Gaussian mixture prior to the latent space. It has two major components: a compression network for dimensionality reduction and an estimation network for anomaly detection using Gaussian Mixture Modelling to calculate anomaly scores in low-dimensional representations. However, DAGMM only considers spatial dependencies and lacks temporal information. The estimation network introduced a regularisation term that helps the compression network avoid local optima and reduce reconstruction errors through end-to-end training.

EncDec-AD [125] model detects anomalies from unpredictable UTS by using the first principal component of the MTS. It can handle time series up to 500 points long but faces issues with error accumulation for longer sequences. [98] proposes two AEs ensemble frameworks based on sparsely connected RNNs: one with independent AEs and another with multiple AEs trained simultaneously, sharing features and using median reconstruction errors to detect outliers. Audibert et al. [10] propose Unsupervised Anomaly Detection (USAD) using AEs in which adversarially trained AEs are utilised to amplify reconstruction errors in MTS, distinguishing anomalies and facilitating quick learning. The input to USAD for either training or testing is in a temporal order. Goodge et al. [70] determine whether AEs are vulnerable to adversarial attacks in anomaly detection by analyzing the effects of various adversarial attacks. APAE (Approximate Projection Autoencoder) improves robustness against adversarial attacks by using gradient descent on latent representations and feature-weighting normalisation to account for variable reconstruction errors across features.

Manuscript submitted to ACM

In MSCRED [192], attention-based ConvLSTM networks capture temporal trends, and a convolutional autoencoder (CAE) reconstructs a signature matrix, representing inter-sensor correlations instead of relying on the time series explicitly. The matrix length is 16 , with a step interval of 5 . An anomaly score is derived from the reconstruction error, aiding in anomaly detection, root cause identification, and anomaly duration interpretation. In CAE-Ensemble [22], a convolutional sequence-to-sequence autoencoder captures temporal dependencies with high parallelism. Gated Linear Units (GLU) with convolution layers and attention capture local patterns, recognising recurring subsequences like periodicity. The ensemble combines outputs from diverse models based on CAEs and uses a parameter-transfer training strategy, which enhances accuracy and reduces training time and error. In order to ensure diversity, the objective function also considers the differences between basic models rather than simply assessing their accuracy.

RANSysCoders [1] outlines a real-time anomaly detection system used by eBay. The authors propose an architecture with multiple encoders and decoders, using random feature selection and majority voting to infer and localise anomalies. The decoders set reconstruction bounds, functioning as bootstrapped AE for feature-bounds construction. The authors also recommend using spectral analysis of the latent space representation to extract priors for MTS synchronisation. Improved accuracy comes from feature synchronisation, bootstrapping, quantile loss, and majority voting. This method addresses issues with previous approaches, such as threshold identification, time window selection, downsampling, and inconsistent performance for large feature dimensions.

A novel Adaptive Memory Network with Self-supervised Learning (AMSL) [198] is designed to increase the generalisation of unsupervised anomaly detection. AMSL uses an AE framework with convolutions for end-to-end training. It combines self-supervised learning and memory networks to handle limited normal data. The encoder maps the raw time series and its six transformations into a feature space. A multi-class classifier is then used to classify these features and improve generalisation. The features are also processed through global and local memory networks, which learn common and specific features. Finally, an adaptive fusion module merges these features into a new reconstruction representation. Recently, ContextDA [106] utilises deep reinforcement learning to optimise domain adaptation for TSAD. It frames context sampling as a Markov decision process, focusing on aligning windows from the source and target domains. The model uses a discriminator to align these domains without leveraging label information in the source domain, which may lead to ineffective alignment when anomaly classes differ. ContextDA addresses this by leveraging source labels, enhancing the alignment of normal samples and improving detection accuracy.

3.3.2 Variational Autoencoder (VAE). Fig. 10b shows a typical configuration of the variational autoencoder (VAE), a directional probabilistic graph model which combines neural network autoencoders with mean-field variational Bayes [102]. The VAE works similarly to AE, but instead of encoding inputs as single points, it encodes them as a distribution using inference network $q_{\phi}\left(Z_{t-w+1: t} \mid X_{t-w+1: t}\right)$ where $\phi$ is its parameters. It represents a $d$ dimensional input $X_{t-w+1: t}$ to a latent representation $Z_{t-w+1: t}$ with a lower dimension $k<d$. A sampling layer takes a sample from a latent distribution and feeds it to the generative network $p_{\theta}\left(X_{t-w+1: t} \mid Z_{t-w+1: t}\right)$ with parameters $\theta$, and its output is $g\left(Z_{t-w+1: t}\right)$, reconstruction of the input. There are two components of the loss function, as stated in Equation (23) that are minimised in a VAE: a reconstruction error that aims to improve the process of encoding and decoding and a regularisation factor, which aims to regularise the latent space by making the encoder's distribution as close to the preferred distribution as possible.

$$
\begin{equation*}
\text { loss }=\left\|X_{t-w+1: t}-g\left(Z_{t-w+1: t}\right)\right\|^{2}+K L\left(N\left(\mu_{x}, \sigma_{x}\right), N(0,1)\right) \tag{23}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-20.jpg?height=231&width=900&top_left_y=318&top_left_x=669)

(a) Auto-Encoder

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-20.jpg?height=214&width=900&top_left_y=592&top_left_x=669)

(b) Variational Auto-Encoder

Fig. 10. Structure of (a) Auto-Encoder that compresses an input window into a lower-dimensional representation (h) and then reconstructs the output $\hat{X}$ from this representation, and (b) Variational Auto-Encoder that its encoder compresses an input window of size $w$ into a latent distribution. The decoder uses sampled data from this distribution to produce $\hat{X}$, closely matching $X$.

where $K L$ is the Kullback-Leibler divergence. By using regularised training, it avoids overfitting and ensures that the latent space is appropriate for a generative process.

LSTM-VAE [143] represents a variation of the VAE that uses LSTM instead of a feed-forward network. This model is trained with a denoising autoencoding method for better representation. It detects anomalies when the log-likelihood of a data point is below a dynamic, state-based threshold to reduce false alarms. Xu et al. [184] found that training on both normal and abnormal data is crucial for VAE anomaly detection. Their model, Donut, uses a VAE trained on shuffled data for unsupervised anomaly detection. Donut's Modified ELBO, Missing Data Injection, and MCMC Imputation make it excellent at detecting anomalies in the seasonal KPI dataset. However, due to VAE's nonsequential nature and sliding window format, Donut struggles with temporal anomalies. Later on, Bagel [115] is introduced to handle temporal anomalies robustly and unsupervised. Instead of using VAE in Donut, Bagel employs conditional variational autoencoder (CVAE) [109] and considers temporal information. VAE models the relationship between two random variables, $x$ and $z$. CVAE models the relationship between $x$ and $z$, conditioned on $y$, i.e., it models $p(x, z \mid y)$.

STORNs [159], or stochastic recurrent networks, use variational inference to model high-dimensional time series data. The algorithm is flexible and generic and doesn't need domain knowledge for structured time series. OmniAnomaly [163] uses a VAE with stochastic RNNs for robust representations of multivariate data and planar normalizing flow for non-Gaussian latent space distributions. It detects anomalies based on reconstruction probability and uses POT for thresholding. InterFusion [117] uses a hierarchical Variational Autoencoder (HVAE) with two stochastic latent variables for intermetric and temporal representations, along with a two-view embedding. To prevent overfitting anomalies in training data, InterFusion employs prefiltering temporal anomalies. The paper also introduces MCMC imputation, MTS for anomaly interpretation, and IPS for assessing results.

There are a few studies on anomaly detection in noisy time series data. Buzz [32] uses an adversarial training method to capture patterns in univariate KPI with non-Gaussian noises and complex data distributions. This model links Bayesian networks with optimal transport theory using Wasserstein distance. SISVAE (smoothness-inducing sequential VAE) [112] detects point-level anomalies by smoothing before training a deep generative model using a Bayesian method. As a result, it benefits from the efficiency of classical optimisation models as well as the ability to model uncertainty with deep generative models. This model adjusts thresholds dynamically based on noise estimates,
crucial for changing time series. Other studies have used VAE for anomaly detection, assuming an unimodal Gaussian distribution as a prior. Existing studies have struggled to learn the complex distribution of time series due to its inherent multimodality. The GRU-based Gaussian Mixture VAE [74] addresses this challenge of learning complex distributions by using GRU cells to discover time sequence correlations and represent multimodal data with a Gaussian Mixture.

In [191], a VAE with two extra modules is introduced: a Re-Encoder and a Latent Constraint network (VELC). The Re-Encoder generates new latent vectors, and this complex setup maximises the anomaly score (reconstruction error) in both the original and latent spaces to accurately model normal samples. The VELC network prevents the reconstruction of untrained anomalies, leading to latent variables similar to the training data, which helps distinguish normal from anomalous data. The VAE and LSTM are integrated as a single component in PAD [30] to support unsupervised anomaly detection and robust prediction. The VAE minimises noise impact on predictions, while LSTMs help VAE capture long-term sequences. Spectral residuals (SR) [83] are also used to improve performance by assigning weights to each subsequence, indicating their normality.

TopoMAD (topology-aware multivariate time series anomaly detector) [79] is an anomaly detector in cloud systems that uses GNN, LSTM, and VAE for spatiotemporal learning. It's a stochastic seq2seq model that leverages topological information to identify anomalies using graph-based representations. The model replaces standard LSTM cells with graph neural networks (GCN and GAT) to capture spatial dependencies. To improve anomaly detection, models like VAE-GAN [138] use partially labelled data. This semi-supervised model integrates LSTMs into a VAE, training an encoder, generator, and discriminator simultaneously. The model distinguishes anomalies using both VAE reconstruction differences and discriminator results.

The recently developed Robust Deep State Space Model (RDSSM) [113] is an unsupervised density reconstructionbased model for detecting anomalies in MTS. Unlike many current methods, RDSSM uses raw data that might contain anomalies during training. It incorporates two transition modules to handle temporal dependency and uncertainty. The emission model includes a heavy-tail distribution error buffer, allowing it to handle contaminated and unlabelled training data robustly. Using this generative model, they created a detection method that manages fluctuating noise over time. This model provides adaptive anomaly scores for probabilistic detection, outperforming many existing methods.

In [177], a variational transformer is introduced for unsupervised anomaly detection in MTS. Instead of using a feature relationship graph, the model captures correlations through self-attention. The model's performance improves due to reduced dimensionality and sparse correlations. The transformer's positional encoding, or global temporal encoding, helps capture long-term dependencies. Multi-scale feature fusion allows the model to capture robust features from different time scales. The residual VAE module encodes hidden space using local features, and its residual structure improves the KL divergence and enhances model generation.

3.3.3 Generative Adversarial Networks (GAN). A generative adversarial network (GAN) is an artificial intelligence algorithm designed for generative modelling based on game theory [69], [69]. In generative models, training examples are explored, and the probability distribution that generated them is learned. In this way, GAN can generate more examples based on the estimated distribution, as illustrated in Fig. 11. Assume that we named the generator $G$ and the discriminator $D$. The generator and discriminator are trained using following minimax model:

$$
\begin{equation*}
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{x \sim p(X)}\left[\log D\left(X_{t-w+1: t}\right)\right]+\mathbb{E}_{z \sim p(Z)}\left[\log \left(1-D\left(Z_{t-w+1: t}\right)\right)\right] \tag{24}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_3ae5d776bb60d31e3f2ag-22.jpg?height=323&width=919&top_left_y=321&top_left_x=668)

Fig. 11. Overview of a Generative Adversarial Network (GAN) with two main components: generator and discriminator. The generator creates fake time series windows for the discriminator, which learns to distinguish between real and fake data. A combined anomaly score is calculated using both the trained discriminator and generator.

where $p(x)$ is the probability distribution of input data and $X_{t-w+1: t}$ is a sliding window from the training set, called real input in Fig.11. Also, $p(z)$ is the prior probability distribution of the generated variable and $Z_{t-w+1: t}$ is a generated input window taken from a random space with the same window size.

In spite of the fact that GANs have been applied to a wide variety of purposes (mainly in research), they continue to involve unique challenges and research openings because they rely on game theory, which is distinct from most approaches to generative modelling. Generally, GAN-based models take into account the fact that adversarial learning makes the discriminator more sensitive to data outside the current dataset, making reconstructions of such data more challenging. BeatGAN [200] is able to regularise its reconstruction robustly because it utilises a combination of AEs and GANs [69] in cases where labels are not available. Moreover, using the time series warping method improves detection accuracy by speed augmentation in training datasets and robust BeatGAN against variability involving time warping in time series data. Research shows that BeatGAN can detect anomalies accurately in both ECG and sensor data.

However, training the GAN is usually difficult and requires a careful balance between the discriminator and generator [104]. A system based on adversarial training is not suitable for online use due to its instability and difficulty in convergence. With Adversarial Autoencoder Anomaly Detection Interpretation (DAEMON), anomalies are detected using adversarially generated time series. DAEMON's training involves three steps. First, a one-dimensional CNN encodes MTS. Then, instead of decoding the hidden variable directly, a prior distribution is applied to the latent vector, and an adversarial strategy aligns the posterior distribution with the prior. This avoids inaccurate reconstructions of unseen patterns. Finally, a decoder reconstructs the time series, and another adversarial training step minimises differences between the original and reconstructed values.

MAD-GAN (Multivariate Anomaly Detection with GAN) [111] is a GAN-based model that uses LSTM-RNN as both the generator and discriminator to capture temporal relationships in time series. It detects anomalies using reconstruction error and discrimination loss. Furthermore, FGANomaly (Filter GAN) [54] tackles overfitting in AEbased and GAN-based anomaly detection models by filtering out potential abnormal samples before training using pseudo-labels. The generator uses Adaptive Weight Loss, assigning weights based on reconstruction errors during training, allowing the model to focus on normal data and reduce overfitting.

3.3.4 Transformers. Anomaly Transformer [185] uses an attention mechanism to spot unusual patterns by simultaneously modelling prior and series associations for each timestamp. This makes rare anomalies more distinguishable. Anomalies are harder to connect with the entire series, while normal patterns connect more easily with nearby timestamps. Prior associations estimate a focus on nearby points using a Gaussian kernel, while series associations use self-attention weights from raw data. Along with reconstruction loss, a MINIMAX approach is used to enhance Manuscript submitted to ACM
the difference between normal and abnormal association discrepancies. TranAD [171] is another transformer-based model that has self-conditioning and adversarial training. As a result of its architecture, it is efficient for training and testing while preserving stability when dealing with huge input. When anomalies are subtle, transformer-based encoder-decoder networks may fail to detect them. However, TranAD's adversarial training amplifies reconstruction errors to fix this. Self-conditioning ensures robust feature retrieval, improving stability and generalisation.

Li et al. [114] present an unsupervised method called DCT-GAN, which uses a transformer to handle time series data, a GAN to reconstruct samples and spot anomalies, and dilated CNNs to capture temporal info from latent spaces. The model blends multiple transformer generators at different scales to enhance its generalisation and uses a weightbased mechanism to integrate generators, making it suitable for various anomalies. Additionally, MT-RVAE [177] significantly benefits from the transformer's sequence modelling and VAE capabilities that are categorised in both of these architectures.

The Dual-TF [136] is a framework for detecting anomalies in time series data by utilising both time and frequency information. It employs two parallel transformers to analyze data in these domains separately, then combines their losses to improve the detection of complex anomalies. This dual-domain approach helps accurately pinpoint both point-wise and subsequence-wise anomalies by overcoming the granularity discrepancies between time and frequency.

### 3.4 Representation-Based Models

Representation-based models aim to learn rich representations of input time series that can then be used in downstream tasks such as anomaly detection and classification. In other words, rather than using the time series in the raw input space for anomaly detection, the learned representations in the latent space are used for anomaly detection. By learning robust representations, these models can effectively handle the complexities of time series data, which often contains noise, non-stationarity, and seasonality. These models are particularly useful in scenarios where labelled data is scarce, as they can often learn useful representations in an unsupervised or self-supervised learning schemes. While time series representation learning has become a hot topic in the time series community and a number of attempts have been made in recent years, only limited work has targeted anomaly detection tasks, and this area of research is still largely unexplored. In the following subsections we surveyed representation-based TSAD models.

3.4.1 Transformers. TS2Vec [190] utilises a hierarchical transformer architecture to capture contextual information at multiple scales, providing a universal representation learning approach using self-supervised contrastive learning that defines anomaly detection problem as a downstream task across various time series datasets. In TS2Vec, positive pairs are representations at the same timestamp in two augmented contexts created by timestamp masking and random cropping, while negative samples are representations at different timestamps from the same series or from other series at the same timestamp within the batch.

3.4.2 Convolutional Neural Networks (CNN). TF-C (Time-Frequency Consistency) model [196] is a self-supervised contrastive pre-training framework designed for time series data. By leveraging both time-based and frequency-based representations, the model ensures that these embeddings are consistent within a shared latent space through a novel consistency loss. Using 3-layer 1-D ResNets as the backbone for its time and frequency encoders, the model captures the temporal and spectral characteristics of time series. This architecture allows the TF-C model to learn generalisable representations that can be used for time series anomaly detection in downstream tasks. In TF-C, a positive pair consists slightly perturbed version of an original sample, while a negative pair includes different original samples or their perturbed versions.

DCdetector [187] employs a deep CNN with a dual attention mechanism. This structure focuses on both spatial and temporal dimensions, using contrastive learning to enhance the separability of normal and anomalous patterns, making it adept at identifying subtle anomalies. In this model, a positive pair consists of representations from different views of the same time series, while it does not use negative samples and relies on the dual attention structure to distinguish anomalies by maximizing the representation discrepancy between normal and abnormal samples.

In contrast, CARLA [42] introduces a self-supervised contrastive representation learning approach using a two-phase framework. The first phase, called pretext, differentiates between anomaly-injected samples and original samples. In the second phase, self-supervised classification leverages information about the representations' neighbours to enhance anomaly detection by learning both normal behaviors and deviations indicating anomalies. In CARLA, positive pairs are selected from neighbours, while negative pairs are anomaly-injected samples. In the recent work, DACAD [43] combines a TCN with unsupervised domain adaptation techniques in its contrastive learning framework. It introduces synthetic anomalies to improve learning and generalisation across different domains, using a structure that effectively identifies anomalies through enhanced feature extraction and domain-invariant learning. DACAD selects positive pairs and negative pairs similar to CARLA.

These models exemplify the advancement in using deep learning for TSAD, highlighting the shift towards models that not only detect but also understand the intricate patterns in time series data, which makes this area of research promising. Finally, while all the models in this category are based on self-supervised contrastive learning approaches, there is no work on self-prediction-based self-supervised approaches in the TSAD literature and this research direction is unexplored.

### 3.5 Hybrid Models

These models integrate the strengths of different approaches to enhance time series anomaly detection. A forecastingbased model predicts the next timestamp, while a reconstruction-based model uses latent representations of the time series. Additionally, representation-based models learn comprehensive representations of the time series. By using a joint objective function, these combined models can be optimised simultaneously.

3.5.1 Autoencoder (AE). By capturing spatiotemporal correlation in multisensor time series, the CAE-M (Deep Convolutional Autoencoding Memory network) [197] can model generalised patterns based on normalised data by undertaking reconstruction and prediction simultaneously. It uses a deep convolutional AE with a Maximum Mean Discrepancy (MMD) penalty to match a target distribution in low dimensions, which helps prevent overfitting due to noise or anomalies. To better capture temporal dependencies, it employs nonlinear bidirectional LSTMs with attention and linear autoregressive models. Neural System Identification and Bayesian Filtering (NSIBF) [60] is a new density-based TSAD approach for Cyber-Physical Security (CPS). It uses a neural network with a state-space model to track hidden state uncertainty over time, capturing CPS dynamics. In the detection phase, Bayesian filtering is applied to the state-space model to estimate the likelihood of observed values. This combination of neural networks and Bayesian filters allows NSIBF to accurately detect anomalies in noisy CPS sensor data.

3.5.2 Recurrent Neural Networks (RNN). With TAnoGan [13], they have developed a method that can detect anomalies in time series if a limited number of examples are provided. TAnoGan has been evaluated using 46 NAB time series datasets covering a range of topics. Experiments have shown that LSTM-based GANs can outperform LSTM-based GANs when challenged with time series data through adversarial training.

Manuscript submitted to ACM

3.5.3 Graph Neural Networks (GNN). In [199], two parallel graph attention (GAT) layers are introduced for selfsupervised multivariate TSAD. These layers identify connections between different time series and learn relationships between timestamps. The model combines forecasting and reconstruction approaches: the forecasting model predicts one point, while the reconstruction model learns a latent representation of the entire time series. The model can diagnose anomalous time series (interpretability). FuSAGNet [76] fused SAE reconstruction and GNN forecasting to find complex anomalies in multivariate data. It incorporates GDN [45] but embeds sensors in each process, followed by recurrent units to capture temporal patterns. By learning recurrent sensor embeddings and sparse latent representations, the GNN predicts expected behaviours during the testing phase.

### 3.6 Model Selection Guidelines for Time Series Anomaly Detection

This section provides a concise guideline for choosing a TSAD method on specific characteristics of the data and the anomaly detection task at hand for practitioners to choose architectures that will provide the most accurate and efficient anomaly detection.

- Multidimensional Data with Complex Dependencies: GNNs are suitable for capturing both temporal and spatial dependencies in multivariate time series. They are particularly effective in scenarios such as IoT sensor networks and industrial systems where intricate interdependencies among dimensions exist. GNN architectures such as GCNs and GATs are suggested to be used in such settings.
- Sequential Data with Long-Term Temporal Dependencies: LSTM and GRU are effective for applications requiring the modelling of long-term temporal dependencies. LSTM is commonly used in financial time series analysis, predictive maintenance, and healthcare monitoring. GRU, with its simpler structure, offers faster training times and is suitable for efficient temporal dependency modelling.
- Large Datasets Requiring Scalability and Efficiency: Transformers utilise self-attention mechanisms to efficiently model long-range dependencies, making them suitable for handling large-scale datasets [97], such as network traffic analysis. They are designed for robust anomaly detection by capturing complex temporal patterns, with models like the Anomaly Transformer [185] and TranAD [171] being notable examples.
- Handling Noise in Anomaly Detection: AEs and VAEs architectures are particularly adept at handling noise in the data, making them suitable for applications like network traffic, multivariate sensor data, and cyber-physical systems.
- High-Frequency Data and Detailed Temporal Patterns: CNNs are useful for capturing local temporal patterns in high-frequency data. They are particularly effective in detecting small deviations and subtle anomalies in data such as web traffic and real-time monitoring systems. TCNs extend CNNs by using dilated convolutions to capture long-term dependencies. As a result, they are suitable for applications where there exist long-range dependencies as well as local patterns [11].
- Data with Evolving Patterns and Multimodal Distributions: Combining the strengths of various architectures, hybrid models are designed to handle complex, high-dimensional time series data with evolving patterns like smart grid monitoring, industrial automation, and climate monitoring. These models, such as those integrating GNNs, VAEs, and LSTMs, are suitable for the mentioned applications.
- Capturing Hierarchical and Multi-Scale Contexts: HTM models are designed to capture hierarchical and multi-scale contexts in time series data. They are robust to noise and can learn multiple patterns simultaneously, making them suitable for applications involving complex temporal patterns and noisy data.

Table 3. Public dataset and benchmarks used mostly for anomaly detection in time series. There are direct hyperlinks to their names in the first column.

| $\overline{\text { Dataset/Benchmark }}$ | Real/Synth | MTS/UTS ${ }^{1}$ | \# Samples ${ }^{2}$ | \# Entities ${ }^{3}$ | \# Dim $^{4}$ | Domain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CalIt2 [55] | Real | MTS | 10,080 | 2 | 2 | Urban events management |
| CAP [168] [67] | Real | MTS | $921,700,000$ | 108 | 21 | Medical and health |
| CICIDS2017 [155] | Real | MTS | $2,830,540$ | 15 | 83 | Server machines monitoring |
| Credit Card fraud detection [41] | Real | MTS | 284,807 | 1 | 31 | Fraud detectcion |
| DMDS [179] | Real | MTS | 725,402 | 1 | 32 | Industrial Control Systems |
| Engine Dataset [44] | Real | MTS | NA | NA | 12 | Industrial control systems |
| Exathlon [94] | Real | MTS | 47,530 | 39 | 45 | Server machines monitoring |
| GECCO IoT [132] | Real | MTS | 139,566 | 1 | 9 | Internet of things (IoT) |
| Genesis [175] | Real | MTS | 16,220 | 1 | 18 | Industrial control systems |
| $\mathrm{GHL}[63]$ | Synth | MTS | 200,001 | 48 | 22 | Industrial control systems |
| IOnsphere [55] | Real | MTS | 351 |  | 32 | Astronomical studies |
| KDDCUP99 [51] | Real | MTS | $4,898,427$ | 5 | 41 | Computer networks |
| Kitsune [55] | Real | MTS | $3,018,973$ | 9 | 115 | Computer networks |
| $\operatorname{MBD}[79]$ | Real | MTS | 8,640 | 5 | 26 | Server machines monitoring |
| Metro $[55]$ | Real | MTS | 48,204 | 1 | 5 | Urban events management |
| MIT-BIH Arrhythmia (ECG) [131] [67] | Real | MTS | $28,600,000$ | 48 | 2 | Medical and health |
| MIT-BIH-SVDB [71] [67] | Real | MTS | $17,971,200$ | 78 | 2 | Medical and health |
| $\operatorname{MMS}[79]$ | Real | MTS | 4,370 | 50 | 7 | Server machines monitoring |
| MSL [92] | Real | MTS | 132,046 | 27 | 55 | Aerospace |
| NAB-realAdExchange [5] | Real | MTS | 9,616 | 3 | 2 | Business |
| NAB-realAWSCloudwatch [5] | Real | MTS | 67,644 | 1 | 17 | Server machines monitoring |
| NASA Shuttle Valve Data [62] | Real | MTS | 49,097 | 1 | 9 | Aerospace |
| OPPORTUNITY [55] | Real | MTS | 869,376 | 24 | 133 | Computer networks |
| Pooled Server Metrics (PSM) [1] | Real | MTS | 132,480 | 1 | 24 | Server machines monitoring |
| PUMP [154] | Real | MTS | 220,302 | 1 | 44 | Industrial control systems |
| SMAP [92] | Real | MTS | 562,800 | 55 | 25 | Environmental management |
| SMD [115] | Real | MTS | $1,416,825$ | 28 | 38 | Server machines monitoring |
| SWAN-SF [9] | Real | MTS | 355,330 | 5 | 51 | Astronomical studies |
| SWaT [129] | Real | MTS | 946,719 | 1 | 51 | Industrial control systems |
| WADI [7] | Real | MTS | 957,372 | 1 | 127 | Industrial control systems |
| NYC Bike [123] | Real | MTS/UTS | $+25 \mathrm{M}$ | NA | NA | Urban events management |
| NYC Taxi [166] | Real | MTS/UTS | $+200 \mathrm{M}$ | NA | NA | Urban events management |
| $\mathrm{UCR}[44]$ | Real/Synth | MTS/UTS | NA | NA | NA | Multiple domains |
| Dodgers Loop Sensor Dataset [55] | Real | UTS | 50,400 | 1 | 1 | Urban events management |
| KPI AIOPS [25] | Real | UTS | $5,922,913$ | 58 | 1 | Business |
| MGAB [170] | Synth | UTS | 100,000 | 10 | 1 | Medical and health |
| MIT-BIH-LTDB [67] | Real | UTS | $67,944,954$ | 7 | 1 | Medical and health |
| NAB-artificialNoAnomaly [5] | Synth | UTS | 20,165 | 5 | 1 | $-\quad 20$ |
| NAB-artificialWithAnomaly [5] | Synth | UTS | 24,192 | 6 | 1 | - |
| NAB-realKnownCause [5] | Real | UTS | 69,568 | 7 | 1 | Multiple domains |
| NAB-realTraffic [5] | Real | UTS | 15,662 | 7 | 1 | Urban events management |
| NAB-realTweets [5] | Real | UTS | 158,511 | 10 | 1 | Business |
| NeurIPS-TS [107] | Synth | UTS | NA | 1 | 1 | $-\quad 2$ |
| NormA [18] | Real/Synth | UTS | $1,756,524$ | 21 | 1 | Multiple domains |
| Power Demand Dataset [44] | Real | UTS | 35,040 | 1 | 1 | Industrial control systems |
| SensoreScope [12] | Real | UTS | 621,874 | 23 | 1 | Internet of things (IoT) |
| Space Shuttle Dataset [44] | Real | UTS | 15,000 | 15 | 1 | Aerospace |
| Yahoo [93] | $\mathrm{Real} /$ Synth | UTS | 572,966 | 367 | 1 | Multiple domains |

- Generalisation Across Diverse Datasets: Contrastive learning excels in scenarios requiring generalisation across diverse datasets by learning robust representations through positive and negative pairs. It effectively distinguishes normal from anomalous patterns in time series data, making it ideal for applications with varying conditions, such as industrial monitoring, network security, and healthcare diagnostics.


## 4 DATASETS

This section summarises datasets and benchmarks for TSAD, which provides a rich resource for researchers in TSAD. Some of these datasets are single-purpose datasets for anomaly detection, and some are general-purpose time series Manuscript submitted to ACM
datasets that we can use in anomaly detection model evaluation with some assumptions or customisation. We can characterise each dataset or benchmark based on multiple aspects and their natural features. Here, we collect 48 well-known and/or highly-cited datasets examined by classic and state-of-the-art (SOTA) deep models for anomaly detection in time series. These datasets are characterised based on the below attributes:

- Nature of the data generation which can be real, synthetic or combined.
- Number of entities, which means the number of independent time series inside each dataset.
- Type of variety for each dataset or benchmark, which can be multivariate, univariate or a combination of both.
- Number of dimensions, which is the number of features of an entity inside the dataset.
- Total number of samples of all entities in the dataset.
- The application domain of the dataset.

Note some datasets have been updated by their authors and contributors occasionally or regularly over time. We considered and reported the latest update of the datasets and their attributes. Table 3 shows all 48 datasets with all mentioned attributes for each of them. It also includes hyperlinks to the primary source to download the latest version of the datasets.

Based on our exploration, the commonly used MTS datasets in SOTA TSAD models are MSL [92], SMAP [92], SMD [115], SWaT [129], PSM [1], and WADI [7]. For UTS, the commonly used datasets are Yahoo [93], KPI [25], NAB [5], and UCR [44]. These datasets are frequently used to benchmark and compare the performance of different TSAD models.

More detailed information about these datasets can be found on this Github repository: https://github.com/zamanzadeh/tsanomaly-benchmark.

## 5 DISCUSSION AND CONCLUSION

In spite of the numerous advances in time series anomaly detection, there are still major challenges in detecting several types of anomalies (as described in Section 2.4). In contrast to the tasks relating to the majority (regular patterns), anomaly detection focuses on minority, unpredictable and unusual events, which bring about some challenges. The following are some challenges that have to be overcome in order to detect anomalies in time series data using deep learning models:

- System behaviour in the real world is highly dynamic and influenced by the prevailing environmental conditions, rendering time series data inherently non-stationary with frequently changing data distributions. This nonstationary nature necessitates the adaptation of deep learning models through online or incremental training approaches, enabling them to update continuously and detect anomalies in real-time. Such methodologies are crucial as they allow models to remain effective in the face of evolving patterns and sudden shifts, thereby ensuring timely and accurate anomaly detection.
- The detection of anomalies in multivariate high-dimensional time series data presents a particular challenge as data can become sparse in high dimension and the model requires simultaneous consideration of both temporal dependencies and relationships between dimensions.
- In the absence of labelled anomalies, unsupervised, semi-supervised or self-supervised approaches are required. Because of this, a large number of normal instances are incorrectly identified as anomalies. Hence, one of the key challenges is to find a mechanism for minimising false positives and improve recall rates of detection.
- Time series datasets can exhibit significant differences in noise existence, and noisy instances may be irregularly distributed. Thus, models are vulnerable, and their performance is compromised by noise in the input data.
- The use of anomaly detection for diagnostic purposes requires interpretability. Even so, anomaly detection research focuses primarily on detection precision, failing to address the issue of interpretability.
- In addition to being rarely addressed in the literature, anomalies that occur on a periodic basis make detection more challenging. A periodic subsequence anomaly is a subsequence that repeats over time [146]. The periodic subsequence anomaly detection technique, in contrast to point anomaly detection, can be adapted in areas like fraud detection to identify periodic anomalous transactions over time.

The main objective of this study was to explore and identify state-of-the-art deep learning models for TSAD, industrial applications, and datasets. In this regard, a variety of perspectives have been explored regarding the characteristics of time series, types of anomalies in time series, and the structure of deep learning models for TSAD. On the basis of these perspectives, 64 recent deep models were comprehensively discussed and categorised. Moreover, time series deep anomaly detection applications across multiple domains were discussed along with datasets commonly used in this area of research. In the future, active research efforts on time series deep anomaly detection are necessary to overcome the challenges we discussed in this survey.

## REFERENCES

[1] Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. 2021. Practical approach to asynchronous multivariate time series anomaly detection and localization. In $K D D$. 2485-2494.

[2] Oludare Isaac Abiodun, Aman Jantan, Abiodun Esther Omolara, Kemi Victoria Dada, Nachaat AbdElatif Mohamed, and Humaira Arshad. 2018. State-of-the-art in artificial neural network applications: A survey. Heliyon 4, 11 (2018), e00938.

[3] Charu C Aggarwal. 2007. Data streams: models and algorithms. Vol. 31. Springer.

[4] Charu C Aggarwal. 2017. An introduction to outlier analysis. In Outlier analysis. Springer, 1-34.

[5] Subutai Ahmad, Alexander Lavin, Scott Purdy, and Zuha Agha. 2017. Unsupervised real-time anomaly detection for streaming data. Neurocomputing 262 (2017), 134-147.

[6] Azza H Ahmed, Michael A Riegler, Steven A Hicks, and Ahmed Elmokashfi. 2022. RCAD: Real-time Collaborative Anomaly Detection System for Mobile Broadband Networks. In KDD. 2682-2691

[7] Chuadhry Mujeeb Ahmed, Venkata Reddy Palleti, and Aditya P Mathur. 2017. WADI: a water distribution testbed for research in the design of secure cyber physical systems. In Proceedings of the 3rd international workshop on cyber-physical systems for smart water networks. 25-28.

[8] Khaled Alrawashdeh and Carla Purdy. 2016. Toward an online anomaly intrusion detection system based on deep learning. In ICMLA. IEEE, $195-200$.

[9] Rafal Angryk, Petrus Martens, Berkay Aydin, Dustin Kempton, Sushant Mahajan, Sunitha Basodi, Azim Ahmadzadeh, Xumin Cai, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi, Micheal Schuh, and Manolis Georgoulis. 2020. SWAN-SF. https://doi.org/10.7910/DVN/EBCFKM

[10] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A Zuluaga. 2020. Usad: Unsupervised anomaly detection on multivariate time series. In $K D D$. 3395-3404.

[11] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271 (2018).

[12] Guillermo Barrenetxea. 2019. Sensorscope Data. https://doi.org/10.5281/zenodo.2654726

[13] Md Abul Bashar and Richi Nayak. 2020. TAnoGAN: Time series anomaly detection with generative adversarial networks. In 2020 IEEE Symposium Series on Computational Intelligence (SSCI). IEEE, 1778-1785.

[14] Sagnik Basumallik, Rui Ma, and Sara Eftekharnejad. 2019. Packet-data anomaly detection in PMU-based state estimator using convolutional neural network. International fournal of Electrical Power \& Energy Systems 107 (2019), 690-702.

[15] Seif-Eddine Benkabou, Khalid Benabdeslem, and Bruno Canitia. 2018. Unsupervised outlier detection for time series by entropy and dynamic time warping. Knowledge and Information Systems 54, 2 (2018), 463-486.

[16] Siddharth Bhatia, Arjit Jain, Pan Li, Ritesh Kumar, and Bryan Hooi. 2021. MSTREAM: Fast anomaly detection in multi-aspect streams. In WWW. $3371-3382$.

[17] Ane Blázquez-García, Angel Conde, Usue Mori, and Jose A Lozano. 2021. A review on outlier/anomaly detection in time series data. CSUR 54,3 (2021), 1-33.

[18] Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed Meftah, and Emmanuel Remy. 2021. Unsupervised and scalable subsequence anomaly detection in large data series. The VLDB Journal 30, 6 (2021), 909-931.

[19] Loïc Bontemps, Van Loi Cao, James McDermott, and Nhien-An Le-Khac. 2016. Collective anomaly detection based on long short-term memory recurrent neural networks. In FDSE. Springer, 141-152.

Manuscript submitted to ACM

[20] Mohammad Braei and Sebastian Wagner. 2020. Anomaly detection in univariate time-series: A survey on the state-of-the-art. arXiv preprint arXiv:2004.00433 (2020).

[21] Yin Cai, Mei-Ling Shyu, Yue-Xuan Tu, Yun-Tian Teng, and Xing-Xing Hu. 2019. Anomaly detection of earthquake precursor data using long short-term memory networks. Applied Geophysics 16, 3 (2019), 257-266.

[22] David Campos, Tung Kieu, Chenjuan Guo, Feiteng Huang, Kai Zheng, Bin Yang, and Christian S Jensen. 2021. Unsupervised time series outlier detection with diversity-driven convolutional ensembles. VLDB 15, 3 (2021), 611-623.

[23] Ander Carreño, Iñaki Inza, and Jose A Lozano. 2020. Analyzing rare event, anomaly, novelty and outlier detection terms under the supervised classification framework. Artificial Intelligence Review 53, 5 (2020), 3575-3594.

[24] Raghavendra Chalapathy and Sanjay Chawla. 2019. Deep learning for anomaly detection: A survey. arXiv preprint arXiv:1901.03407 (2019).

[25] International AIOPS Challenges. 2018. KPI Anomaly Detection. https://competition.aiops-challenge.com/home/competition/1484452272200032281

[26] Varun Chandola, Arindam Banerjee, and Vipin Kumar. 2009. Anomaly detection: A survey. CSUR 41, 3 (2009), 1-58.

[27] Shiyu Chang, Yang Zhang, Wei Han, Mo Yu, Xiaoxiao Guo, Wei Tan, Xiaodong Cui, Michael Witbrock, Mark A Hasegawa-Johnson, and Thomas S Huang. 2017. Dilated recurrent neural networks. NeurIPS 30 (2017).

[28] Sucheta Chauhan and Lovekesh Vig. 2015. Anomaly detection in ECG time signals via deep long short-term memory networks. In 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 1-7.

[29] Qing Chen, Anguo Zhang, Tingwen Huang, Qianping He, and Yongduan Song. 2020. Imbalanced dataset-based echo state networks for anomaly detection. Neural Computing and Applications 32, 8 (2020), 3685-3694.

[30] Run-Qing Chen, Guang-Hui Shi, Wan-Lei Zhao, and Chang-Hui Liang. 2021. A joint model for IT operation series prediction and anomaly detection. Neurocomputing 448 (2021), 130-139.

[31] Tingting Chen, Xueping Liu, Bizhong Xia, Wei Wang, and Yongzhi Lai. 2020. Unsupervised anomaly detection of industrial robots using sliding-window convolutional variational autoencoder. IEEE Access 8 (2020), 47072-47081.

[32] Wenxiao Chen, Haowen Xu, Zeyan Li, Dan Pei, Jie Chen, Honglin Qiao, Yang Feng, and Zhaogang Wang. 2019. Unsupervised anomaly detection for intricate kpis via adversarial training of vae. In IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 1891-1899.

[33] Xuanhao Chen, Liwei Deng, Feiteng Huang, Chengwei Zhang, Zongquan Zhang, Yan Zhao, and Kai Zheng. 2021. Daemon: Unsupervised anomaly detection and interpretation for multivariate time series. In ICDE. IEEE, 2225-2230.

[34] Zekai Chen, Dingshuo Chen, Xiao Zhang, Zixuan Yuan, and Xiuzhen Cheng. 2021. Learning graph structures with transformer for multivariate time series anomaly detection in iot. IEEE Internet of Things fournal (2021).

[35] Yongliang Cheng, Yan Xu, Hong Zhong, and Yi Liu. 2019. HS-TCN: A semi-supervised hierarchical stacking temporal convolutional network for anomaly detection in IoT. In IPCCC. IEEE, 1-7.

[36] Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014. On the Properties of Neural Machine Translation: Encoder-Decoder Approaches. In Proceedings of SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation. 103-111.

[37] Kukjin Choi, Jihun Yi, Changhwa Park, and Sungroh Yoon. 2021. Deep learning for anomaly detection in time-series data: review, analysis, and guidelines. IEEE Access (2021).

[38] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555 (2014).

[39] Yuwei Cui, Subutai Ahmad, and Jeff Hawkins. 2017. The HTM spatial pooler-a neocortical algorithm for online sparse distributed coding. Frontiers in computational neuroscience (2017), 111.

[40] Enyan Dai and Jie Chen. 2022. Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. In ICLR.

[41] Andrea Dal Pozzolo, Olivier Caelen, Reid A Johnson, and Gianluca Bontempi. 2015. Calibrating probability with undersampling for unbalanced classification. In 2015 IEEE symposium series on computational intelligence. IEEE, 159-166.

[42] Zahra Zamanzadeh Darban, Geoffrey I Webb, Shirui Pan, and Mahsa Salehi. 2023. CARLA: A Self-supervised Contrastive Representation Learning Approach for Time Series Anomaly Detection. arXiv preprint arXiv:2308.09296 (2023).

[43] Zahra Zamanzadeh Darban, Geoffrey I Webb, and Mahsa Salehi. 2024. DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection in Multivariate Time Series. arXiv preprint arXiv:2404.11269 (2024).

[44] Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, Yanping, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen, Gustavo Batista, and Hexagon-ML. 2018. The UCR Time Series Classification Archive. https://www.cs.ucr.edu/ eamonn/time_series_data_2018/.

[45] Ailin Deng and Bryan Hooi. 2021. Graph neural network-based anomaly detection in multivariate time series. In AAAI, Vol. 35. 4027-4035.

[46] Leyan Deng, Defu Lian, Zhenya Huang, and Enhong Chen. 2022. Graph convolutional adversarial networks for spatiotemporal anomaly detection. TNNLS 33, 6 (2022), 2416-2428

[47] Rahul Dey and Fathi M Salem. 2017. Gate-variants of gated recurrent unit (GRU) neural networks. In IEEE 60th international midwest symposium on circuits and systems. IEEE, 1597-1600.

[48] Nan Ding, Huanbo Gao, Hongyu Bu, Haoxuan Ma, and Huaiwei Si. 2018. Multivariate-time-series-driven real-time anomaly detection based on bayesian network. Sensors 18, 10 (2018), 3367.

[49] Nan Ding, HaoXuan Ma, Huanbo Gao, YanHua Ma, and GuoZhen Tan. 2019. Real-time anomaly detection based on long short-Term memory and Gaussian Mixture Model. Computers \& Electrical Engineering 79 (2019), 106458.

[50] Zhiguo Ding and Minrui Fei. 2013. An anomaly detection approach based on isolation forest algorithm for streaming data using sliding window. IFAC Proceedings Volumes 46, 20 (2013), 12-17.

[51] Third International Knowledge Discovery and Data Mining Tools Competition. 1999. KDD Cup 1999 Data. https://kdd.ics.uci.edu/databases/ kddcup99/kddcup99.html

[52] Yadolah Dodge. 2008. Time Series. Springer New York, New York, NY, 536-539. https://doi.org/10.1007/978-0-387-32833-1_401

[53] Nicola Dragoni, Saverio Giallorenzo, Alberto Lluch Lafuente, Manuel Mazzara, Fabrizio Montesi, Ruslan Mustafin, and Larisa Safina. 2017. Microservices: yesterday, today, and tomorrow. Present and ulterior software engineering (2017), 195-216.

[54] Bowen Du, Xuanxuan Sun, Junchen Ye, Ke Cheng, Jingyuan Wang, and Leilei Sun. 2021. GAN-Based Anomaly Detection for Multivariate Time Series Using Polluted Training Set. TKDE (2021).

[55] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository.

[56] Tolga Ergen and Suleyman Serdar Kozat. 2019. Unsupervised anomaly detection with LSTM neural networks. TNNLS 31, 8 (2019), $3127-3141$.

[57] Philippe Esling and Carlos Agon. 2012. Time-series data mining. CSUR 45,1 (2012), 1-34

[58] Okwudili M Ezeme, Qusay Mahmoud, and Akramul Azim. 2020. A framework for anomaly detection in time-driven and event-driven processes using kernel traces. TKDE (2020).

[59] Cheng Fan, Fu Xiao, Yang Zhao, and Jiayuan Wang. 2018. Analytical investigation of autoencoder-based methods for unsupervised anomaly detection in building energy data. Applied energy 211 (2018), 1123-1135.

[60] Cheng Feng and Pengwei Tian. 2021. Time series anomaly detection for cyber-physical systems via neural system identification and bayesian filtering. In $K D D$. 2858-2867.

[61] Yong Feng, Zijun Liu, Jinglong Chen, Haixin Lv, Jun Wang, and Xinwei Zhang. 2022. Unsupervised Multimodal Anomaly Detection With Missing Sources for Liquid Rocket Engine. TNNLS (2022).

[62] Bob Ferrell and Steven Santuro. 2005. NASA Shuttle Valve Data. http://www.cs.fit.edu/ pkc/nasa/data/

[63] Pavel Filonov, Andrey Lavrentyev, and Artem Vorontsov. 2016. Multivariate industrial time series with cyber-attack simulation: Fault detection using an lstm-based predictive data model. arXiv preprint arXiv:1612.06676 (2016).

[64] A Garg, W Zhang, J Samaran, R Savitha, and CS Foo. 2022. An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series. TNNLS 33, 6 (2022), 2508-2517.

[65] Dileep George. 2008. How the brain might work: A hierarchical and temporal model for learning and recognition. Stanford University.

[66] Jonathan Goh, Sridhar Adepu, Marcus Tan, and Zi Shan Lee. 2017. Anomaly detection in cyber physical systems using recurrent neural networks. In 2017 IEEE 18th International Symposium on High Assurance Systems Engineering (HASE). IEEE, 140-145.

[67] A L Goldberger, L A Amaral, L Glass, J M Hausdorff, P C Ivanov, R G Mark, J E Mietus, G B Moody, C K Peng, and H E Stanley. 2000. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. , E215-20 pages.

[68] Abbas Golestani and Robin Gras. 2014. Can we predict the unpredictable? Scientific reports 4, 1 (2014), 1-6

[69] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. NeurIPS 27 (2014).

[70] Adam Goodge, Bryan Hooi, See-Kiong Ng, and Wee Siong Ng. 2020. Robustness of Autoencoders for Anomaly Detection Under Adversarial Impact.. In IfCAI. 1244-1250.

[71] Scott David Greenwald, Ramesh S Patil, and Roger G Mark. 1990. Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using contextual information. IEEE.

[72] Frank E Grubbs. 1969. Procedures for detecting outlying observations in samples. Technometrics 11, 1 (1969), 1-21.

[73] Antonio Gulli and Sujit Pal. 2017. Deep learning with Keras. Packt Publishing Ltd.

[74] Yifan Guo, Weixian Liao, Qianlong Wang, Lixing Yu, Tianxi Ji, and Pan Li. 2018. Multidimensional time series anomaly detection: A gru-based gaussian mixture variational autoencoder approach. In Asian Conference on Machine Learning. PMLR, 97-112.

[75] James Douglas Hamilton. 2020. Time series analysis. Princeton university press.

[76] Siho Han and Simon S Woo. 2022. Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series. In KDD. 2977-2986.

[77] Douglas M Hawkins. 1980. Identification of outliers. Vol. 11. Springer

[78] Yangdong He and Jiabao Zhao. 2019. Temporal convolutional networks for anomaly detection in time series. In fournal of Physics: Conference Series, Vol. 1213. IOP Publishing, 042050.

[79] Zilong He, Pengfei Chen, Xiaoyun Li, Yongfeng Wang, Guangba Yu, Cailin Chen, Xinrui Li, and Zibin Zheng. 2020. A spatiotemporal deep learning approach for unsupervised anomaly detection in cloud systems. TNNLS (2020).

[80] Michiel Hermans and Benjamin Schrauwen. 2013. Training and analysing deep recurrent neural networks. NeurIPS 26 (2013).

[81] Geoffrey E Hinton and Ruslan R Salakhutdinov. 2006. Reducing the dimensionality of data with neural networks. science 313,5786 (2006), $504-507$.

[82] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735-1780.

[83] Xiaodi Hou and Liqing Zhang. 2007. Saliency detection: A spectral residual approach. In IEEE Conference on computer vision and pattern recognition. Ieee, $1-8$.

[84] Ruei-Jie Hsieh, Jerry Chou, and Chih-Hsiang Ho. 2019. Unsupervised online anomaly detection on multivariate sensing time series data for smart manufacturing. In IEEE 12th Conference on Service-Oriented Computing and Applications (SOCA). IEEE, 90-97.

Manuscript submitted to ACM

[85] Chia-Yu Hsu and Wei-Chen Liu. 2021. Multiple time-series convolutional neural network for fault detection and diagnosis and empirical study in semiconductor manufacturing. Journal of Intelligent Manufacturing 32, 3 (2021), 823-836.

[86] Chao Huang, Chuxu Zhang, Peng Dai, and Liefeng Bo. 2021. Cross-interaction hierarchical attention networks for urban anomaly prediction. In IनCAI. 4359-4365.

[87] Ling Huang, Xing-Xing Liu, Shu-Qiang Huang, Chang-Dong Wang, Wei Tu, Jia-Meng Xie, Shuai Tang, and Wendi Xie. 2021. Temporal Hierarchical Graph Attention Network for Traffic Prediction. ACM Transactions on Intelligent Systems and Technology (TIST) 12, 6 (2021), 1-21.

[88] Ling Huang, XuanLong Nguyen, Minos Garofalakis, Michael Jordan, Anthony Joseph, and Nina Taft. 2006. In-network PCA and anomaly detection. NeurIPS 19 (2006)

[89] Tao Huang, Pengfei Chen, and Ruipeng Li. 2022. A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems. In WWW. 1797-1806.

[90] Xin Huang, Jangsoo Lee, Young-Woo Kwon, and Chul-Ho Lee. 2020. CrowdQuake: A networked system of low-cost sensors for earthquake detection via deep learning. In $K D D$. 3261-3271.

[91] Alexis Huet, Jose Manuel Navarro, and Dario Rossi. 2022. Local evaluation of time series anomaly detection algorithms. In $K D D$. 635-645.

[92] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. 2018. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In $K D D$. 387-395.

[93] Yahoo Inc. 2021. S5-A Labeled Anomaly Detection Dataset, Version 1.0. https://webscope.sandbox.yahoo.com/catalog.php?datatype=s\&did=70

[94] Vincent Jacob, Fei Song, Arnaud Stiegler, Bijan Rad, Yanlei Diao, and Nesime Tatbul. 2021. Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series. VLDB (2021).

[95] Herbert Jaeger. 2007. Echo state network. scholarpedia 2, 9 (2007), 2330.

[96] Ahmad Javaid, Quamar Niyaz, Weiqing Sun, and Mansoor Alam. 2016. A deep learning approach for network intrusion detection system. In Proceedings of the 9th EAI International Conference on Bio-inspired Information and Communications Technologies (formerly BIONETICS). 21-26.

[97] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. 2022. Transformers in vision: A survey. CSUR 54,10 s (2022), 1-41.

[98] Tung Kieu, Bin Yang, Chenjuan Guo, and Christian S Jensen. 2019. Outlier Detection for Time Series with Recurrent Autoencoder Ensembles.. In I7CAI. $2725-2732$

[99] Dohyung Kim, Hyochang Yang, Minki Chung, Sungzoon Cho, Huijung Kim, Minhee Kim, Kyungwon Kim, and Eunseok Kim. 2018. Squeezed convolutional variational autoencoder for unsupervised anomaly detection in edge device industrial internet of things. In 2018 international conference on information and computer technologies (icict). IEEE, 67-71.

[100] Eunji Kim, Sungzoon Cho, Byeongeon Lee, and Myoungsu Cho. 2019. Fault detection and diagnosis using self-attentive convolutional neural networks for variable-length sensor data in semiconductor manufacturing. IEEE Transactions on Semiconductor Manufacturing 32, 3 (2019), 302-309.

[101] Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon. 2022. Towards a rigorous evaluation of time-series anomaly detection. In AAAI, Vol. 36. 7194-7201.

[102] Diederik P Kingma and Max Welling. 2014. Auto-Encoding Variational Bayes. stat 1050 (2014), 1.

[103] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR.

[104] Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt Kira. 2017. On convergence and stability of gans. arXiv preprint arXiv:1705.07215 (2017)

[105] Mark A Kramer. 1991. Nonlinear principal component analysis using autoassociative neural networks. AIChE journal 37, 2 (1991), 233-243

[106] Kwei-Herng Lai, Lan Wang, Huiyuan Chen, Kaixiong Zhou, Fei Wang, Hao Yang, and Xia Hu. 2023. Context-aware domain adaptation for time series anomaly detection. In SDM. SIAM, 676-684.

[107] Kwei-Herng Lai, Daochen Zha, Junjie Xu, Yue Zhao, Guanchu Wang, and Xia Hu. 2021. Revisiting time series outlier detection: Definitions and benchmarks. In NeurIPS

[108] Siddique Latif, Muhammad Usman, Rajib Rana, and Junaid Qadir. 2018. Phonocardiographic sensing using deep learning for abnormal heartbeat detection. IEEE Sensors fournal 18, 22 (2018), 9393-9400.

[109] Alexander Lavin and Subutai Ahmad. 2015. Evaluating real-time anomaly detection algorithms-the Numenta anomaly benchmark. In ICMLA. IEEE, $38-44$.

[110] Tae Jun Lee, Justin Gottschlich, Nesime Tatbul, Eric Metcalf, and Stan Zdonik. 2018. Greenhouse: A zero-positive machine learning system for time-series anomaly detection. arXiv preprint arXiv:1801.03168 (2018).

[111] Dan Li, Dacheng Chen, Baihong Jin, Lei Shi, Jonathan Goh, and See-Kiong Ng. 2019. MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In ICANN. Springer, 703-716.

[112] Longyuan Li, Junchi Yan, Haiyang Wang, and Yaohui Jin. 2020. Anomaly detection of time series with smoothness-inducing sequential variational auto-encoder. TNNLS 32, 3 (2020), 1177-1191.

[113] Longyuan Li, Junchi Yan, Qingsong Wen, Yaohui Jin, and Xiaokang Yang. 2022. Learning Robust Deep State Space for Unsupervised Anomaly Detection in Contaminated Time-Series. TKDE (2022).

[114] Yifan Li, Xiaoyan Peng, Jia Zhang, Zhiyong Li, and Ming Wen. 2021. DCT-GAN: Dilated Convolutional Transformer-based GAN for Time Series Anomaly Detection. TKDE (2021).

[115] Zeyan Li, Wenxiao Chen, and Dan Pei. 2018. Robust and unsupervised kpi anomaly detection based on conditional variational autoencoder. In IPCCC. IEEE, $1-9$.

[116] Zhang Li, Bian Xia, and Mei Dong-Cheng. 2001. Gamma-ray light curve and phase-resolved spectra from Geminga pulsar. Chinese Physics 10, 7 (2001), 662.

[117] Zhihan Li, Youjian Zhao, Jiaqi Han, Ya Su, Rui Jiao, Xidao Wen, and Dan Pei. 2021. Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding. In $K D D .3220-3230$.

[118] Fan Liu, Xingshe Zhou, Jinli Cao, Zhu Wang, Tianben Wang, Hua Wang, and Yanchun Zhang. 2020. Anomaly detection in quasi-periodic time series based on automatic data segmentation and attentional LSTM-CNN. TKDE (2020).

[119] Jianwei Liu, Hongwei Zhu, Yongxia Liu, Haobo Wu, Yunsheng Lan, and Xinyu Zhang. 2019. Anomaly detection for time series using temporal convolutional networks and Gaussian mixture model. In fournal of Physics: Conference Series, Vol. 1187. IOP Publishing, 042111.

[120] Manuel Lopez-Martin, Angel Nevado, and Belen Carro. 2020. Detection of early stages of Alzheimer's disease based on MEG activity with a randomized convolutional neural network. Artificial Intelligence in Medicine 107 (2020), 101924.

[121] Zhilong Lu, Weifeng Lv, Zhipu Xie, Bowen Du, Guixi Xiong, Leilei Sun, and Haiquan Wang. 2022. Graph Sequence Neural Network with an Attention Mechanism for Traffic Speed Prediction. ACM Transactions on Intelligent Systems and Technology (TIST) 13, 2 (2022), 1-24.

[122] Tie Luo and Sai G Nagarajan. [n.d.]. Distributed anomaly detection using autoencoder neural networks in WSN for IoT. In ICC, pages=1-6, year $=2018$, organization $=I E E E$.

[123] Lyft. 2022. Citi Bike Trip Histories. https://ride.citibikenyc.com/system-data

[124] Junshui Ma and Simon Perkins. 2003. Online novelty detection on temporal sequences. In KDD. 613-618.

[125] Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff. 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148 (2016).

[126] Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, Puneet Agarwal, et al. 2015. Long short term memory networks for anomaly detection in time series. In ESANN, Vol. 89. 89-94.

[127] Behrooz Mamandipoor, Mahshid Majd, Seyedmostafa Sheikhalishahi, Claudio Modena, and Venet Osmani. 2020. Monitoring and detecting faults in wastewater treatment plants using deep learning. Environmental monitoring and assessment 192, 2 (2020), 1-12.

[128] Mohammad M Masud, Qing Chen, Latifur Khan, Charu Aggarwal, Jing Gao, Jiawei Han, and Bhavani Thuraisingham. 2010. Addressing conceptevolution in concept-drifting data streams. In ICDM. IEEE, 929-934.

[129] Aditya P Mathur and Nils Ole Tippenhauer. 2016. SWaT: A water treatment testbed for research and training on ICS security. In 2016 international workshop on cyber-physical systems for smart water networks (CySWater). IEEE, 31-36.

[130] Hengyu Meng, Yuxuan Zhang, Yuanxiang Li, and Honghua Zhao. 2019. Spacecraft anomaly detection via transformer reconstruction error. In International Conference on Aerospace System Science and Engineering. Springer, 351-362.

[131] George B Moody and Roger G Mark. 2001. The impact of the MIT-BIH arrhythmia database. IEEE Engineering in Medicine and Biology Magazine 20, 3 (2001), 45-50.

[132] Steffen Moritz, Frederik Rehbach, Sowmya Chandrasekaran, Margarita Rebolledo, and Thomas Bartz-Beielstein. 2018. GECCO Industrial Challenge 2018 Dataset: A water quality dataset for the 'Internet of Things: Online Anomaly Detection for Drinking Water Quality' competition at the Genetic and Evolutionary Computation Conference 2018, Kyoto, Japan. https://doi.org/10.5281/zenodo. 3884398

[133] Masud Moshtaghi, James C Bezdek, Christopher Leckie, Shanika Karunasekera, and Marimuthu Palaniswami. 2014. Evolving fuzzy rules for anomaly detection in data streams. IEEE Transactions on Fuzzy Systems 23, 3 (2014), 688-700.

[134] Meinard Müller. 2007. Dynamic time warping. Information retrieval for music and motion (2007), 69-84.

[135] Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. 2018. DeepAnT: A deep learning approach for unsupervised anomaly detection in time series. Ieee Access 7 (2018), 1991-2005.

[136] Youngeun Nam, Susik Yoon, Yooju Shin, Minyoung Bae, Hwanjun Song, Jae-Gil Lee, and Byung Suk Lee. 2024. Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection. (2024).

[137] Andrew Ng et al. 2011. Sparse autoencoder. CS294A Lecture notes 72, 2011 (2011), 1-19.

[138] Zijian Niu, Ke Yu, and Xiaofei Wu. 2020. LSTM-based VAE-GAN for time-series anomaly detection. Sensors 20,13 (2020), 3738.

[139] Hyeonwoo Noh, Seunghoon Hong, and Bohyung Han. 2015. Learning deconvolution network for semantic segmentation. In ICCV. $1520-1528$.

[140] Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel. 2021. Deep learning for anomaly detection: A review. CSUR 54, 2 (2021), 1-38

[141] Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. Deep anomaly detection with deviation networks. In KDD. 353-362.

[142] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S Tsay, Aaron Elmore, and Michael J Franklin. 2022. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. VLDB 15, 11 (2022), 2774-2787

[143] Daehyung Park, Yuuna Hoshi, and Charles C Kemp. 2018. A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder. IEEE Robotics and Automation Letters 3, 3 (2018), 1544-1551.

[144] Thibaut Perol, Michaël Gharbi, and Marine Denolle. 2018. Convolutional neural network for earthquake detection and location. Science Advances 4, 2 (2018), e1700578

[145] Tie Qiu, Ruixuan Qiao, and Dapeng Oliver Wu. 2017. EABS: An event-aware backpressure scheduling scheme for emergency Internet of Things. IEEE Transactions on Mobile Computing 17, 1 (2017), 72-84.

[146] Faraz Rasheed and Reda Alhajj. 2013. A framework for periodic outlier pattern detection in time-series sequences. IEEE transactions on cybernetics 44,5 (2013), 569-582.

Manuscript submitted to ACM

[147] Hansheng Ren, Bixiong Xu, Yujing Wang, Chao Yi, Congrui Huang, Xiaoyu Kou, Tony Xing, Mao Yang, Jie Tong, and Qi Zhang. 2019. Time-series anomaly detection service at microsoft. In $K D D .3009-3017$.

[148] Jonathan Rubin, Rui Abreu, Anurag Ganguli, Saigopal Nelaturi, Ion Matei, and Kumar Sricharan. 2017. Recognizing Abnormal Heart Sounds Using Deep Learning. In IFCAI.

[149] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. 2018. Deep one-class classification. In ICML. PMLR, 4393-4402.

[150] Mayu Sakurada and Takehisa Yairi. 2014. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Workshop on Machine Learning for Sensory Data Analysis. 4-11.

[151] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. 2008. The graph neural network model. IEEE transactions on neural networks 20,1 (2008), 61-80.

[152] Udo Schlegel, Hiba Arnout, Mennatallah El-Assady, Daniela Oelke, and Daniel A Keim. 2019. Towards a rigorous evaluation of xai methods on time series. In ICCVW. IEEE, 4197-4201.

[153] Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock. 2022. Anomaly detection in time series: a comprehensive evaluation. VLDB 15, 9 (2022), 1779-1797.

[154] Pump sensor data. 2018. Pump sensor data for predictive maintenance. https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

[155] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A Ghorbani. 2018. Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSp 1 (2018), 108-116.

[156] Lifeng Shen, Zhuocong Li, and James Kwok. 2020. Timeseries anomaly detection using temporal hierarchical one-class network. NeurIPS 33 (2020), $13016-13026$.

[157] Nathan Shone, Tran Nguyen Ngoc, Vu Dinh Phai, and Qi Shi. 2018. A deep learning approach to network intrusion detection. IEEE transactions on emerging topics in computational intelligence 2, 1 (2018), 41-50.

[158] Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, and Christine Largouet. 2017. Anomaly detection in streams with extreme value theory. In KDD. 1067-1075.

[159] Maximilian Sölch, Justin Bayer, Marvin Ludersdorfer, and Patrick van der Smagt. 2016. Variational inference for on-line anomaly detection in high-dimensional time series. arXiv preprint arXiv:1602.07109 (2016).

[160] Huan Song, Deepta Rajan, Jayaraman Thiagarajan, and Andreas Spanias. 2018. Attend and diagnose: Clinical time series analysis using attention models. In $A A A I$, Vol. 32.

[161] Xiaomin Song, Qingsong Wen, Yan Li, and Liang Sun. 2022. Robust Time Series Dissimilarity Measure for Outlier Detection and Periodicity Detection. In CIKM. 4510-4514.

[162] Yanjue Song and Suzhen Li. 2021. Gas leak detection in galvanised steel pipe with internal flow noise using convolutional neural network. Process Safety and Environmental Protection 146 (2021), 736-744

[163] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. 2019. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In $K D D$. 2828-2837.

[164] Mahbod Tavallaee, Ebrahim Bagheri, Wei Lu, and Ali A Ghorbani. 2009. A detailed analysis of the KDD CUP 99 data set. In 2009 IEEE symposium on computational intelligence for security and defense applications. Ieee, 1-6.

[165] David MJ Tax and Robert PW Duin. 2004. Support vector data description. Machine learning 54 (2004), 45-66.

[166] NYC Taxi and Limousine Commission. 2022. TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

[167] Ahmed Tealab. 2018. Time series forecasting using artificial neural networks methodologies: A systematic review. Future Computing and Informatics Journal 3, 2 (2018), 334-340.

[168] M G Terzano, L Parrino, A Sherieri, R Chervin, S Chokroverty, C Guilleminault, M Hirshkowitz, M Mahowald, H Moldofsky, A Rosa, R Thomas, and A Walters. 2001. Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep Med. 2, 6 (Nov. 2001), $537-553$.

[169] Markus Thill, Wolfgang Konen, and Thomas Bäck. 2020. Time series encodings with temporal convolutional networks. In International Conference on Bioinspired Methods and Their Applications. Springer, 161-173.

[170] Markus Thill, Wolfgang Konen, and Thomas Bäck. 2020. MarkusThill/MGAB: The Mackey-Glass Anomaly Benchmark. https://doi.org/10.5281/ zenodo. 3760086

[171] Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings. 2022. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. VLDB 15 (2022), 1201-1214

[172] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. NeurIPS 30 (2017)

[173] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2018. GRAPH ATTENTION NETWORKS. stat 1050 (2018), 4 .

[174] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In ICML. 1096-1103.

[175] Alexander von Birgelen and Oliver Niggemann. 2018. Anomaly detection and localization for cyber-physical production systems with self-organizing maps. In Improve-innovative modelling approaches for production systems to raise validatable efficiency. Springer Vieweg, Berlin, Heidelberg, 55-71.

[176] Kai Wang, Youjin Zhao, Qingyu Xiong, Min Fan, Guotan Sun, Longkun Ma, and Tong Liu. 2016. Research on healthy anomaly detection model based on deep learning from multiple time-series physiological signals. Scientific Programming 2016 (2016).

[177] Xixuan Wang, Dechang Pi, Xiangyan Zhang, Hao Liu, and Chang Guo. 2022. Variational transformer-based anomaly detection approach for multivariate time series. Measurement 191 (2022), 110791.

[178] Yi Wang, Linsheng Han, Wei Liu, Shujia Yang, and Yanbo Gao. 2019. Study on wavelet neural network based anomaly detection in ocean observing data series. Ocean Engineering 186 (2019), 106129.

[179] Politechnika Warszawska. 2020. Damadics Benchmark Website. https://iair.mchtr.pw.edu.pl/Damadics

[180] Tailai Wen and Roy Keyes. 2019. Time series anomaly detection using convolutional neural networks and transfer learning. arXiv preprint arXiv:1905.13628 (2019).

[181] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. 2023. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In ICLR.

[182] Jia Wu, Weiru Zeng, and Fei Yan. 2018. Hierarchical temporal memory method for time-series-based anomaly detection. Neurocomputing 273 (2018), 535-546

[183] Wentai Wu, Ligang He, Weiwei Lin, Yi Su, Yuhua Cui, Carsten Maple, and Stephen A Jarvis. 2020. Developing an unsupervised real-time anomaly detection scheme for time series with multi-seasonality. TKDE (2020).

[184] Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, et al. 2018. Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications. In WWW. 187-196.

[185] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. 2021. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. In ICLR.

[186] Kenji Yamanishi and Jun-ichi Takeuchi. 2002. A unifying framework for detecting outliers and change points from non-stationary time series data. In $K D D .676-681$.

[187] Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. 2023. DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection. In KDD (Long Beach, CA).

[188] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh. 2016. Matrix profile I: all pairs similarity joins for time series: a unifying view that includes motifs, discords and shapelets. In ICDM. 1317-1322.

[189] Yue Yu, Jie Chen, Tian Gao, and Mo Yu. 2019. DAG-GNN: DAG structure learning with graph neural networks. In ICML. PMLR, 7154-7163.

[190] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. 2022. Ts2vec: Towards universal representation of time series. In $A A A I$, Vol. 36. 8980-8987.

[191] Chunkai Zhang, Shaocong Li, Hongye Zhang, and Yingyang Chen. 2019. VELC: A new variational autoencoder based model for time series anomaly detection. arXiv preprint arXiv:1907.01702 (2019).

[192] Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V Chawla. 2019. A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data. In AAAI, Vol. 33. $1409-1416$.

[193] Mingyang Zhang, Tong Li, Hongzhi Shi, Yong Li, Pan Hui, et al. 2019. A decomposition approach for urban anomaly detection across spatiotemporal data. In IFCAI. International Joint Conferences on Artificial Intelligence.

[194] Runtian Zhang and Qian Zou. 2018. Time series prediction and anomaly detection of light curve using lstm neural network. In fournal of Physics: Conference Series, Vol. 1061. IOP Publishing, 012012

[195] Weishan Zhang, Wuwu Guo, Xin Liu, Yan Liu, Jiehan Zhou, Bo Li, Qinghua Lu, and Su Yang. 2018. LSTM-based analysis of industrial IoT equipment. IEEE Access 6 (2018), 23551-23560.

[196] Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik. 2022. Self-supervised contrastive pre-training for time series via time-frequency consistency. NeurIPS 35 (2022), 3988-4003.

[197] Yuxin Zhang, Yiqiang Chen, Jindong Wang, and Zhiwen Pan. 2021. Unsupervised deep anomaly detection for multi-sensor time-series signals. TKDE (2021).

[198] Yuxin Zhang, Jindong Wang, Yiqiang Chen, Han Yu, and Tao Qin. 2022. Adaptive memory networks with self-supervised learning for unsupervised anomaly detection. TKDE (2022).

[199] Hang Zhao, Yujing Wang, Juanyong Duan, Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. 2020. Multivariate time-series anomaly detection via graph attention network. In ICDM. IEEE, 841-850.

[200] Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. 2019. BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series.. In IFCAI. 4433-4439.

[201] Lingxue Zhu and Nikolay Laptev. 2017. Deep and confident prediction for time series at uber. In ICDMW. IEEE, 103-110.

[202] Weiqiang Zhu and Gregory C Beroza. 2019. PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical fournal International 216, 1 (2019), 261-273.

[203] Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and Haifeng Chen. 2018. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In ICLR.

Manuscript submitted to ACM
</end of paper 1>


<paper 2>
# AI for IT Operations (AIOps) on Cloud Platforms: Reviews, Opportunities and Challenges 

Qian Cheng * ${ }^{* \dagger}$, Doyen Sahoo *, Amrita Saha, Wenzhuo Yang, Chenghao Liu, Gerald Woo,<br>Manpreet Singh, Silvio Saverese, and Steven C. H. Hoi<br>Salesforce AI


#### Abstract

Artificial Intelligence for IT operations (AIOps) aims to combine the power of AI with the big data generated by IT Operations processes, particularly in cloud infrastructures, to provide actionable insights with the primary goal of maximizing availability. There are a wide variety of problems to address, and multiple use-cases, where AI capabilities can be leveraged to enhance operational efficiency. Here we provide a review of the AIOps vision, trends challenges and opportunities, specifically focusing on the underlying AI techniques. We discuss in depth the key types of data emitted by IT Operations activities, the scale and challenges in analyzing them, and where they can be helpful. We categorize the key AIOps tasks as - incident detection, failure prediction, root cause analysis and automated actions. We discuss the problem formulation for each task, and then present a taxonomy of techniques to solve these problems. We also identify relatively under explored topics, especially those that could significantly benefit from advances in AI literature. We also provide insights into the trends in this field, and what are the key investment opportunities.


딘.

Index Terms-AIOps, Artificial Intelligence, IT Operations, Machine Learning, Anomaly Detection, Root-cause Analysis, Failure Prediction, Resource Management

## I. INTRODUCTION

Modern software has been evolving rapidly during the era of digital transformation. New infrastructure, techniques and design patterns - such as cloud computing, Software-as-aService (SaaS), microservices, DevOps, etc. have been developed to boost software development. Managing and operating the infrastructure of such modern software is now facing new challenges. For example, when traditional software transits to SaaS, instead of handing over the installation package to the user, the software company now needs to provide 24/7 software access to all the subscription based users. Besides developing and testing, service management and operations are now the new set of duties of SaaS companies. Meanwhile, traditional software development separates functionalities of the entire software lifecycle. Coding, testing, deployment and operations are usually owned by different groups. Each of these groups requires different sets of skills. However, agile development and DevOps start to obfuscate the boundaries between each process and DevOps engineers are required to take E2E responsibilities. Balancing development and operations for a DevOps team become critical to the whole team's productivity.[^0]

Software services need to guarantee service level agreements (SLAs) to the customers, and often set internal Service Level Objectives (SLOs). Meeting SLAs and SLOs is one of the top priority for CIOs to choose the right service providers [1]. Unexpected service downtime can impact availability goals and cause significant financial and trust issues. For example, AWS experienced a major service outage in December 2021, causing multiple first and third party websites and heavily used services to experience downtime [2].

IT Operations plays a key role in the success of modern software companies and as a result multiple concepts have been introduced, such as IT service management (ITSM) specifically for SaaS, and IT operations management (ITOM) for general IT infrastructure. These concepts focus on different aspects IT operations but the underlying workflow is very similar. Life cycle of Software systems can be separated into several main stages, including planning, development/coding, building, testing, deployment, maintenance/operations, monitoring, etc. [3]. The operation part of DevOps can be further broken down into four major stages: observe, detect, engage and act, shown in Figure 1. Observing stage includes tasks like collecting different telemetry data (metrics, logs, traces, etc.), indexing and querying and visualizing the collected telemetries. Time-to-observe (TTO) is a metric to measure the performance of the observing stage. Detection stage includes tasks like detecting incidents, predicting failures, finding correlated events, etc. whose performance is typically measured as the Time-to-detect (TTD) (in addition to precision/recall). Engaging stage includes tasks like issue triaging, localization, root-cause analysis, etc., and the performance is often measured by Time-to-triage (TTT). Acting stage includes immediate remediation actions such as reboot the server, scale-up / scale-out resources, rollback to previous versions, etc. Time-to-resolve (TTR) is the key metric measured for the acting stage. Unlike software development and release, where we have comparatively mature continuous integration and continuous delivery (CI/CD) pipelines, many of the postrelease operations are often done manually. Such manual operational processes face several challenges:

- Manual operations struggle to scale. The capacity of manual operations is limited by the size of the DevOps team and the team size can only increase linearly. When the software usage is at growing stage, the throughput and workloads may grow exponentially, both in scale and complexity. It is difficult for DevOps team to grow at the

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-02.jpg?height=1122&width=851&top_left_y=195&top_left_x=190)

Fig. 1. Common DevOps life cycles[3] and ops breakdown. Ops can comprise four stages: observe, detect, engage and act. Each of the stages has a corresponding measure: time-to-observe, time-to-detect, time-to-triage and time-to-resolve.

same pace to handle the increasing amount of operational workload.

- Manual operations is hard to standardize. It is very hard to keep the same high standard across the entire DevOps team given the diversity of team members (e.g. skill level, familiarity with the service, tenure, etc.). It takes significant amount of time and effort to grow an operational domain expert who can effectively handle incidents. Unexpected attrition of these experts could significantly hurt the operational efficiency of a DevOps team.
- Manual operations are error-prone. It is very common that human operation error causes major incidents. Even for the most reliable cloud service providers, major incidents have been caused by human error in recent years.

Given these challenges, fully-automated operations pipelines powered by AI capabilities becomes a promising approach to achieve the SLA and SLO goals. AIOps, an acronym of AI for IT Operations, was coined by Gartner at 2016. According to Gartner Glossary, "AIOps combines big data and machine learning to automate IT operations processes, including event correlation, anomaly detection and causality determination"[4]. In order to achieve fully- automated IT Operations, investment in AIOps technolgies is imperative. AIOps is the key to achieve high availability, scalability and operational efficiency. For example, AIOps can use AI models can automatically analyze large volumes of telemetry data to detect and diagnose incidents much faster, and much more consistently than humans, which can help achieve ambitious targets such as 99.99 availability. AIOps can dynamically scale its capabilities with growth demands and use AI for automated incident and resource management, thereby reducing the burden of hiring and training domain experts to meet growth requirements. Moreover, automation through AIOps helps save valuable developer time, and avoid fatigue. AIOps, as an emerging AI technology, appeared on the trending chart of Gartner Hyper Cycle for Artificial Intelligence in 2017 [5], along with other popular topics such as deep reinforcement learning, nature-language generation and artificial general intelligence. As of 2022, enterprise AIOps solutions have witnessed increased adoption by many companies' IT infrastructure. The AIOps market size is predicted to be $\$ 11.02 \mathrm{~B}$ by end of 2023 with cumulative annual growth rate (CAGR) of $34 \%$.

AIOps comprises a set of complex problems. Transforming from manual to automated operations using AIOps is not a one-step effort. Based on the adoption level of AI techniques, we break down AIOps maturity into four different levels based on the adoption of AIOps capabilities as shown in Figure 2.

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-02.jpg?height=445&width=852&top_left_y=1320&top_left_x=1081)

More Human Power

Fig. 2. AIOps Transformation. Different maturity levels based on adoption of AI techniques: Manual Ops, human-centric AIOps, machine-centric AIOps, fully-automated AIOps.

Manual Ops. At this maturity level, DevOps follows traditional best practices and all processes are setup manually. There is no AI or ML models. This is the baseline to compare with in AIOps transformation.

Human-centric. At this level, operations are done mainly in manual process and AI techniques are adopted to replace subprocedures in the workflow, and mainly act as assistants. For example, instead of glass watching for incident alerts, DevOps or SREs can set dynamic alerting threshold based on anomaly detection models. Similarly, the root cause analysis process requires watching multiple dashboards to draw insights, and AI can help automatically obtain those insights.

Machine-centric. At this level, all major components (monitoring, detecting, engaging and acting) of the E2E operation process are empowered by more complex AI techniques.

Humans are mostly hands-free but need to participate in the human-in-the-loop process to help fine-tune and improve the AI systems performance. For example, DevOps / SREs operate and manage the AI platform to guarantee training and inference pipelines functioning well, and domain experts need to provide feedback or labels for AI-made decisions to improve performance.

Fully-automated. At this level, AIOps platform achieves full automation with minimum or zero human intervention. With the help of fully-automated AIOps platforms, the current $\mathrm{CI} / \mathrm{CD}$ (continuous integration and continuous deployment) pipelines can be further extended to $\mathrm{CI} / \mathrm{CD} / \mathrm{CM} / \mathrm{CC}$ (continuous integration, continuous deployment, continuous monitoring and continuous correction) pipelines.

Different software systems, and companies may be at different levels of AIOps maturity, and their priorities and goals may differ with regard to specific AIOps capabilities to be adopted. Setting up the right goals is important for the success of AIOps applications. We foresee the trend of shifting from manual operation all the way to fully-automated AIOps in the future, with more and more complex AI techniques being used to address challenging problems. In order to enable the community to adopt AIOps capabilities faster, in this paper, we present a comprehensive survey on the various AIOps problems and tasks and the solutions developed by the community to address them.

## II. Contribution of This SurveY

Increasing number of research studies and industrial products in the AIOps domain have recently emerged to address a variety of problems. Sabharwal et al. published a book "Handson AIOps" to discuss practical AIOps and implementation [6]. Several AIOps literature reviews are also accessible [7] [8] to help audiences better understand this domain. However, there are very limited efforts to provide a holistic view to deeply connect AIOps with latest AI techniques. Most of the AI related literature reviews are still topic-based, such as deep learning anomaly detection [9] [10], failure management, root-cause analysis [11], etc. There is still limited effort to provide a holistic view about AIOps, covering the status in both academia and industry. We prepare this survey to address this gap, and focus more on AI techniques used in AIOps.

Except for the monitoring stage, where most of the tasks focus on telemetry data collection and management, AIOps covers the other three stages where the tasks focus more on analytics. In our survey, we group AIOps tasks based on which operational stage they can contribute to, shown in Figure 3.

Incident Detection. Incident detection tasks contribute to detection stage. The goal of these tasks are reducing meantime-to-detect (MTTD). In our survey we cover time series incident detection (Section IV-A), log incident detection (Section IV-B), trace and multimodal incident detection (Section IV-C).

Failure Prediction. Failure prediction also contributes to detection stage. The goal of failure prediction is to predict the potential issue before it actually happens so actions can be taken in advance to minimize impact. Failure prediction also contributes to reducing mean-time-to-detect (MTTD). In our survey we cover metric failure prediction (Section V-A and $\log$ failure prediction (Section V-B). There are very limited efforts in literature that perform traces and multimodal failure prediction.

Root-cause Analysis. Root-cause analysis tasks contributes to multiple operational stages, including triaging, acting and even support more efficient long-term issue fixing and resolution. Helping as an immediate response to an incident, the goal is to minimize time to triage (MTTT), and simultaneously contribute to reduction on reducing Mean Time to Resolve (MTTR). An added benefit is also reduction in human toil. We further breakdown root-cause analysis into time-series RCA (Section VI-B), logs RCA (Section VI-B) and traces and multimodal RCA (Section VI-C).

Automated Actions. Automated actions contribute to acting stage, where the main goal is to reduce mean-time-to-resolve (MTTR), as well as long-term issue fix and resolution. In this survey we discuss about a series of methods for autoremediation (Section VII-A), auto-scaling (Section VII-B) and resource management (Section VII-C).

## III. DATA FOR AIOpS

Before we dive into the problem settings, it is important to understand the data available to perform AIOps tasks. Modern software systems generate tremendously large volumes of observability metrics. The data volume keeps growing exponentially with digital transformation [12]. The increase in the volume of data stored in large unstructured Data lake systems makes it very difficult for DevOps teams to consume the new information and fix consumers' problems efficiently [13]. Successful products and platforms are now built to address the monitoring and logging problems. Observability platforms, e.g. Splunk, AWS Cloudwatch, are now supporting emitting, storing and querying large scale telemetry data.

Similar to other AI domains, observability data is critical to AIOps. Unfortunately there are limited public datasets in this domain and many successful AIOps research efforts are done with self-owned production data, which usually are not available publicly. In this section, we describe major telemetry data type including metrics, logs, traces and other records, and present a collection of public datasets for each data type.

## A. Metrics

Metrics are numerical data measured over time which provide a snapshot of the system behavior. Metrics can represent a broad range of information, broadly classified into compute metrics and service metrics. Compute metrics (e.g. CPU utilization, memory usage, disk I/O) are an indicator of the health status of compute nodes (servers, virtual machines, pods). They are collected at the system level using tools such as Slurm [14] for usage statistics from jobs and nodes, and the Lustre parallel distributed file system for I/O information. Service metrics (e.g. request count, page visits, number of errors) measure the quality and level of service of customer facing applications. Aggregate statistics of such numerical data

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-04.jpg?height=564&width=1724&top_left_y=206&top_left_x=211)

Fig. 3. AIOps Tasks. In this survey we discuss a series of AIOps tasks, categorized by which operational stages these tasks contribute to, and the observability data type it takes.

also fall under the category of metrics, providing a more coarse-grained view of system behavior.

Metrics are constantly generated by all components of the cloud platform life cycle, making it one of the most ubiquitous forms of AIOps data. Cloud platforms and supercomputer clusters can generate petabytes of metrics data, making it a challenge to store and analyze, but at the same time, brings immense observability to the health of the entire IT operation. Being numerical time-series data, metrics are simple to interpret and easy to analyze, allowing for simple thresholdbased rules to be acted upon. At the same time, they contain sufficiently rich information to be used to power more complex AI based alerting and actions.

The major challenge in leveraging insights from metrics data arises due to their diverse nature. Metrics data can exhibit a variety of patterns, such as cyclical patterns (repeating patterns hourly, daily, weekly, etc.), sparse and intermittent spikes, and noisy signals. The characteristics of the metrics ultimately depend on the underlying service or job.

In Table I, we briefly describe the datasets and benchmarks of metrics data. Metrics data have been used in studies characterizing the workloads of cloud data centers, as well as the various AIOps tasks of incident detection, root cause analysis, failure prediction, and various planning and optimization tasks like auto-scaling and VM pre-provisioning.

## B. Logs

Software logs are specifically designed by the software developers in order to record any type of runtime information about processes executing within a system - thus making them an ubiquitous part of any modern system or software maintenance. Once the system is live and throughout its lifecycle, it continuously emits huge volumes of such logging data which naturally contain a lot of rich dynamic runtime information relevant to IT Operations and Incident Management of the system. Consequently in AI driven IT-Ops pipelines, automated log based analysis plays an important role in Incident Management - specifically in tasks like Incident Detection and Causation and Failure Prediction, as have been
![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-04.jpg?height=458&width=882&top_left_y=953&top_left_x=1076)

Fig. 4. GPU utilization metrics from the MIT Supercloud Dataset exhibiting various patterns (cyclical, sparse and intermittant, noisy).

studied by multiple literature surveys in the past [15], [16], [17], [18], [19], [20], [21], [22], [23].

In most of the practical cases, especially in industrial settings, the volume of the logs can go upto an order of petabytes of loglines per week. Also because of the nature of $\log$ content, log data dumps are much more heavier in size in comparison to time series telemetry data. This requires special handling of logs observability data in form of data streams, - where today, there are various services like Splunk, Datadog, LogStash, NewRelic, Loggly, Logz.io etc employed to efficiently store and access the log stream and also visualize, analyze and query past log data using specialized structured query language.

Nature of Log Data. Typically these logs consist of semistructured data i.e. a combination of structured and unstructured data. Amongst the typical types of unstructured data there can be natural language tokens, programming language constructs (e.g. method names) and the structured part can consist of quantitative or categorical telemetry or observability metrics data, which are printed in runtime by various logging statements embedded in the source-code or sometimes generated automatically via loggers or logging agents. Depending on the kind of service the logs are dumped from, there can be
a diverse types of logging data with heterogeneous form and content. For example, logs can be originating from distributed systems (e.g. hadoop or spark), operating systems (windows or linux) or in complex supercomputer systems or can be dumped at hardware level (e.g. switch logs) or middle-ware level (like servers e.g. Apache logs) or by specific applications (e.g. Health App). Typically each logline comprises of a fixed part which is the template that had been designed by the developer and some variable part or parameters which capture some runtime information about the system.

Complexities of Log Data. Thus, apart from being one of the most generic and hence crucial data-sources in IT Ops, logs are one of the most complex forms of observability data due to their open-ended form and level of granularity at which they contain system runtime information. In cloud computing context, logs are the source of truth for cloud users to the underlying servers that running their applications since cloud providers don't grant full access to their users of the servers and platforms. Also, being designed by developers, logs are immediately affected by any changes in the sourcecode or logging statements by developers. This results in non-stationarity in the logging vocabulary or even the entire structure or template underlying the logs.

Log Observability Tasks. Log observability typically involves different tasks like anomaly detection over logs during incident detection (Section IV-B), root cause analysis over logs (Section VI-B) and log based failure prediction (Section V-B).

Datasets and Benchmarks. Out of the different log observability tasks, log based anomaly detection is one of the most objective tasks and hence most of the publicly released benchmark datasets have been designed around anomaly detection. In Table B. we give a comprehensive description about the different public benchmark datasets that have been used in the literature for anomaly detection tasks. Out of these, datasets Switch and subsets of HPC and BGL have also been redesigned to serve failure prediction task. On the other hand there are no public benchmarks on log based RCA tasks, which has been typically evaluated on private enterprise data.

## C. Traces

Trace data are usually presented as semi-structured logs, with identifiers to reconstruct the topological maps of the applications and network flows of target requests. For example, when user uses Google search, a typical trace graph of this user request looks like in Figure 6 Traces are composed system events (spans) that tracks the entire progress of a request or execution. A span is a sequence of semi-structured event logs. Tracing data makes it possible to put different data modality into the same context. Requests travel through multiple services / applications and each application may have totally different behavior. Trace records usually contains two required parts: timestamps and span_id. By using the timestamps and span_id, we can easily reconstruct the trace graph from trace logs.

$$
\text { public void NGPStatsMsgHandler() \{ }
$$

logger.warning("Invalid NGPQueue::addTask L2Stats mQueue size = \{\} Remaining capacity is \{\} \{\}", queue_size, rem_capacity, address)

logger.info("EventActions \{\} Executing alarm rules for Alerts \{\}, Event Id \{\}PNET Event Id (), First time \{\}", thread_id, alert_value, event_id, first_time)

## $\sqrt{ }$

WARNING 08/09 17:35:19 [NGPStatsMsgHandler] Invalid NGPQueue: :addTask L2Stats mQueue size $=0$ Remaining capacity is 150000 null

INFO 08/09 17:35:19 [NGPStatsMsgHandler] EventActions[Thread-956703] 302704 Executing alarm rules for Alerts (502640,163770,null): Event Id (mwcontrolm@172.25.153.37:3181.1626882403.6807769), PNet Event Id (), First time (21/08/09 17:51:14)

INFO 08/09 17:35:18 [NGPStatsMsgHandler] EventActions[Thread-956704] 302704 Executing alarm rules for Alerts (502640,163770,null): Event Id

(mwcontrolm@172.25.153.37:3181.1626882403.6807770), PNet Event Id (), First time (21/08/09 17:51:14)

INFO 08/09 17:35:17 [NGPStatsMsgHandler]

EventActions[Thread-956703] 302704 Executing alarm rules for Alerts (502640,163771, null): Event Id (mwcontrolm@172.25.153.37:3181.1626882403.6807769), PNet Event Id (), First time (21/08/09 17:51:14)

Fig. 5. An example of Log Data generated in IT Operations

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-05.jpg?height=542&width=740&top_left_y=1320&top_left_x=1148)

Fig. 6. An snapshot of trace graph of user requests when using Google Search.

Trace analysis requires reliable tracing systems. Trace collection systems such as ReTrace [24] can help achieve fast and inexpensive trace collections. Trace collectors are usually code agnostic and can emit different levels of performance trace data back to the trace stores in near real-time. Early summarization is also involved in the trace collection process to help generate fine-grained events [25].

Although trace collection is common for system observability, it is still challenging to acquire high quality trace data to train AI models. As far as we know, there are very few public trace datasets with high quality labels. Also the only few existing public trace datasets like [26] are not widely adopted in AIOps research. Instead, most AIOps related trace analysis research use self-owned production or simulation trace data,
which are generally not available publicly.

## D. Other Data

Besides the machine generated observability data like metrics, logs, traces, etc., there are other types of operational data that could be used in AIOps.

Human activity records is part of these valuable data. Ticketing systems are used for DevOps/SREs to communicate and efficiently resolve the issues. This process generates large amount of human activity records. The human activity data contains rich knowledge and learnings about solutions to existing issues, which can be used to resolve similar issues in the future.

User feedback data is also very important to improve AIOps system performance. Unlike the issue tickets where human needs to put lots of context information to describe and discuss the issue, user feedback can be as simple as one click to confirm if the alert is good or bad. Collecting real-time user feedback of a running system and designing human-in-theloop workflows are also very significant for success of AIOps solutions.

Although many companies collects these types of data and use them to improve their operation workflows, there are still very limited published research discussing how to systematically incorporate these other types of operational data in AIOps solutions. This brings challenges as well as opportunities to make further improvements in AIOps domain.

Next, we discuss the key AIOps Tasks - Incident Detection, Failure Prediction, Root Cause Analysis, and Automated Actions, and systematically review the key contributions in literature in these areas.

## IV. INCIDENT DeTECTION

Incident detection employs a variety of anomaly detection techniques. Anomaly detection is to detect abnormalities, outliers or generally events that not normal. In AIOps context, anomaly detection is widely adopted in detecting any types of abnormal system behaviors. To detect such anomalies, the detectors need to utilize different telemetry data, such as metrics, logs, traces. Thus, anomaly detection can be further broken down to handling one or more specific telemetry data sources, including metric anomaly detection, log anomaly detection, trace anomaly detection. Moreover, multi-modal anomaly detection techniques can be employed if multiple telemetry data sources are involved in the detection process. In recent years, deep learning based anomaly detection techniques [9] are also widely discussed and can be utilized for anomaly detection in AIOps. Another way to distinguish anomaly detection techniques is depending on different application use cases, such as detecting service health issues, detecting networking issues, detecting security issues, fraud transactions, etc. Usually these variety of techniques are derived from same set of base detection algorithms and localized to handle specific tasks. From technical perspective, detecting anomalies from different telemetry data sources are better aligned with the AI technology definitions, such as, metric are usually time-series, logs are text / natural language, traces are event sequences/graphs, etc. In this article, we discuss anomaly detection by different telemetry data sources.

## A. Metrics based Incident Detection

## Problem Definition

To ensure the reliability of services, billions of metrics are constantly monitored and collected at equal-space timestamp [27]. Therefore, it is straightforward to organize metrics as time series data for subsequent analysis. Metric based incident detection, which aims to find the anomalous behaviors of monitored metrics that significantly deviate from the other observations, is vital for operators to timely detect software failures and trigger failure diagnosis to mitigate loss. The most basic form of incident detection on metrics is the rule-based method which sets up an alert when a metric breaches a certain threshold. Such an approach is only able to capture incidents which are defined by the metric exceeding the threshold, and is unable to detect more complex incidents. The rulebased method to detect incidents on metrics are generally too naive, and only able to account for the most simple of incidents. They are also sensitive to the threshold, producing too many false positives when the threshold is too low, and false negatives when the threshold is too high. Due to the openended nature of incidents, increasingly complex architectures of systems, and increasing size of these systems and number of metrics, manual monitoring and rule-based methods are no longer sufficient. Thus, more advanced metric-based incident detection methods that leveraging AI capability is urgent.

As metrics are a form of time series data, and incidents are expressed as an abnormal occurrence in the data, metric incident detection is most often formulated as a time series anomaly detection problem [28], [29], [30]. In the following, we focus on the AIOps setting and categorize it based on several key criteria: (i) learning paradigm, (ii) dimensionality, (iii) system, and (iv) streaming updates. We further summarize a list of time series anomaly detection methods with a comparison over these criteria in Table IV.

## Learning Setting

a) Label Accessibility: One natural way to formulate the anomaly detection problem, is as the supervised binary classification problem, to classify whether a given observation is an anomaly or not [31], [32]. Formulating it as such has the benefit of being able to apply any supervised learning method, which has been intensely studied in the past decades [33]. However, due to the difficulty in obtaining labelled data for metrics incident detection [34] and labels of anomalies are prone to error [35], unsupervised approaches, which do not require labels to build anomaly detectors, are generally preferred and more widespread. Particularly, unsupervised anomaly detection methods can be roughly categorized into density-based methods, clustering-based methods, and reconstruction-based methods [28], [29], [30]. Densitybased methods compute local density and local connectivity for outlier decision. Clustering-based methods formulate the anomaly score as the distance to cluster center. Reconstructionbased methods explicitly model the generative process of the
data and measure the anomaly score with the reconstruction error. While methods in metric anomaly detection are generally unsupervised, there are cases where there is some access to labels. In such situations, semi-supervised, domain adaptation, and active learning paradigms come into play. The semisupervised paradigm [36], [37], [38] enables unsupervised models to leverage information from sparsely available positive labels [39]. Domain adaptation [40] relies on a labelled source dataset, while the target dataset is unlabeled, with the goal of transferring a model trained on the source dataset, to perform anomaly detection on the target.

b) Streaming Update: Since metrics are collected in large volume every minute, the model is used online to detect anomalies. It is very common that temporal patterns of metrics change overtime. The ability to perform timely model updates when receiving new incoming data is an important criteria. On the one hand, conventional models can handle the data stream via retraining the whole model periodically [31], [41], [32], [38]. However, this strategy could be computationally expensive, and bring extra non-trivial questions, such as, how often should this retraining be performed. On the other hand, some methods [42], [43] have efficient updating mechanisms inbuilt, and are naturally able to adapt to these new incoming data streams. It can also support active learning paradigm [41], which allows models to interactively query users for labels on data points for which it is uncertain about, and subsequently update the model with the new labels.

c) Dimensionality: Each metric of monitoring data forms a univariate time series, and thus a service usually contains multiple metrics, each of which describes a different part or attribute of a complex entity, constituting a multivariate time series. The conventional solution is to build univariate time series anomaly detection for each metric. However, for a complex system, it ignores the intrinsic interactions among each metric and cannot well represent the system's overall status. Naively combining the anomaly detection results of each univariate time series performs poorly for multivariate anomaly detection method [44], since it cannot model the inter-dependencies among metrics for a service.

Model A wide range of machine learning models can be used for time series anomaly detection, broadly classified as deep learning models, tree-based models, and statistical models. Deep learning models [45], [36], [46], [47], [38], [48], [49], [50] leverage the success and power deep neural networks to learn representations of the time series data. These representations of time series data contain rich semantic information of the underlying metric, and can be used as a reconstructionbased, unsupervised method. Tree-based methods leverage a tree structure as a density-based, unsupervised method [42]. Statistical models [51] rely on classical statistical tests, which are considered a reconstruction-based method.

Industrial Practices Building a system which can handle the large amounts of metric data generated in real cloud IT operations is often an issue. This is because the metric data in real-world scenarios is quite diverse and the definition of anomaly may vary in different scenarios. Moreover, almost all time series anomaly detection systems require to handle a large amount of metrics in parallel with low-latency [32]. Thus, works which propose a system to handle the infrastructure are highlighted here. EGADS [41] is a system by Yahoo!, scaling up to millions of data points per second, and focuses on optimizing real-time processing. It comprises a batch time series modelling module, an online anomaly detection module, and an alerting module. It leverages a variety of unsupervised methods for anomaly detection, and an optional active learning component for filtering alerts. [52] is a system by Microsoft, which includes three major components, a data ingestion, experimentation, and online compute platform. They propose an efficient deep learning anomaly detector to achieve high accuracy and high efficiency at the same time. [32] is a system by Alibaba group, comprising data ingestion, offline training, online service, and visualization and alarms modules. They propose a robust anomaly detector by using time series decomposition, and thus can easily handle time series with different characteristics, such as different seasonal length, different types of trends, etc. [38] is a system by Tencent, comprising of a offline model training component and online serving component, which employs active learning to update the online model via a small number of uncertain samples.

## Challenges

Lack of labels The main challenge of metric anomaly detection is the lack of ground truth anomaly labels [53], [44]. Due to the open-ended nature and complexity of incidents in server architectures, it is difficult to define what an anomaly is. Thus, building labelled datasets is an extremely labor and resource intensive exercise, one which requires the effort of domain experts to identify anomalies from time series data. Furthermore, manual labelling could lead to labelling errors as there is no unified and formal definition of an anomaly, leading to subjective judgements on ground truth labels [35].

Real-time inference A typical cloud infrastructure could collect millions of data points in a second, requiring near realtime inference to detect anomalies. Metric anomaly detection systems need to be scalable and efficient [54], [53], optionally supporting model retraining, leading to immense compute, memory, and I/O loads. The increasing complexity of anomaly detection models with the rising popularity of deep learning methods [55] add a further strain on these systems due to the additional computational cost these larger models bring about.

Non-stationarity of metric streams The temporal patterns of metric data streams typically change over time as they are generated from non-stationary environments [56]. The evolution of these patterns is often caused by exogenous factors which are not observable. One such example is that the growth in the popularity of a service would cause customer metrics (e.g. request count) to drift upwards over time. Ignoring these factors would cause a deterioration in the anomaly detector's performance. One solution is to continuously update the model with the recent data [57], but this strategy requires carefully balancing of the cost and model robustness with respect to the updating frequency.

Public benchmarks While there exists benchmarks for general anomaly detection methods and time series anomaly detection methods [33], [58], there is still a lack of benchmarking for metric incident detection in AIOps domain. Given the
wide and diverse nature of time series data, they often exhibit a mixture of different types of anomaly depends on specific domain, making it challenging to understand the pros and cons of algorithms [58]. Furthermore, existing datasets have been criticised to be trivial and mislabelled [59].

## Future Trends

Active learning/human-in-the-loop To address the problem of lacking of labels, a more intelligent way is to integrate human knowledge and experience with minimum cost. As special agents, humans have rich prior knowledge [60]. If the incident detection framework can encourage the machine learning model to engage with learning operation expert wisdom and knowledge, it would help deal with scarce and noise label issue. The use of active learning to update online model in [38] is a typical example to incorporate human effort in the annotation task. There are certainly large research scope for incorporating human effort in other data processing step, like feature extraction. Moreover, the human effort can also be integrated in the machine learning model training and inference phase.

Streaming updates Due to the non-stationarity of metric streams, keeping the anomaly detector updated is of utmost importance. Alongside the increasingly complex models and need for cost-effectiveness, we will see a move towards methods with the built-in capability of efficient streaming updates. With the great success of deep learning methods in time series anomaly detection tasks [30]. Online deep learning is an increasingly popular topic [61], and we may start to see a transference of techniques into metric anomaly detection for time-series in the near future.

Intrinsic anomaly detection Current research works on time series anomaly detection do not distinguish the cause or the type of anomaly, which is critical for the subsequent mitigation steps in AIOps. For example, even anomaly are successfully detected, which is caused by extrinsic environment, the operator is unable to mitigate its negative effect. Introduced in [50], [48], intrinsic anomaly detection considers the functional dependency structure between the monitored metric, and the environment. This setting considers changes in the environment, possibly leveraging information that may not be available in the regular (extrinsic) setting. For example, when scaling up/down the resources serving an application (perhaps due to autoscaling rules), we will observe a drop/increase in CPU metric. While this may be considered as an anomaly in the extrinsic setting, it is in fact not an incident and accordingly, is not an anomaly in the intrinsic setting.

## B. Logs based Incident Detection

## Problem Definition

Software and system logging data is one of the most popular ways of recording and tracking runtime information about all ongoing processes within a system, to any arbitrary level of granularity. Overall, a large distributed system can have massive volume of heterogenous logs dumped by its different services or microservices, each having time-stamped text messages following their own unstructured or semistructured or structured format. Throughout various kinds of IT Operations these logs have been widely used by reliability and performance engineers as well as core developers in order to understand the system's internal status and to facilitate monitoring, administering, and troubleshooting [15], [16], [17], [18], [19], [20], [21], [22], [62]. More, specifically, in the AIOps pipeline, one of the foremost tasks that log analysis can cater to is log based Incident Detection. This is typically achieved through anomaly detection over logs which aims to detect the anomalous loglines or sequences of loglines that indicate possible occurrence of an incident, from the humungous amounts of software logging data dumps generated by the system. Log based anomaly detection is generally applied once an incident has been detected based on monitoring of KPI metrics, as a more fine-grained incident detection or failure diagnosis step in order to detect which service or micro-service or which software module of the system execution is behaving anomalously.

## Task Complexity

Diversity of Log Anomaly Patterns: There are very diverse kinds of incidents in AIOps which can result in different kinds of anomaly patterns in the log data - either manifesting in the $\log$ template (i.e. the constant part of the log line) or the $\log$ parameters (i.e. the variable part of the log line containing dynamic information). These are i) keywords - appearance of keywords in log lines bearing domain-specific semantics of failure or incident or abnormality in the system (e.g. out of memory or crash) ii) template count - where a sudden increase or decrease of log templates or log event types is indicative of anomaly iii) template sequence - where some significant deviation from the normal order of task execution is indicative of anomaly iv) variable value - some variables associated with some log templates or events can have physical meaning (e.g. time cost) which could be extracted out and aggregated into a structured time series on which standard anomaly detection techniques can be applied. v) variable distribution - for some categorical or numerical variables, a deviation from the standard distribution of the variable can be indicative of an anomaly vi) time interval - some performance issues may not be explicitly observed in the logline themselves but in the time interval between specific log events.

Need for AI: Given the humongous nature of the logs, it is often infeasible for even domain experts to manually go through the logs to detect the anomalous loglines. Additionally, as described above, depending on the nature of the incident there can be diverse types of anomaly patterns in the logs, which can manifest as anomalous key words (like "errors" or "exception") in the log templates or the volume of specific event logs or distribution over log variables or the time interval between two log specific event logs. However, even for a domain expert it is not possible to come up with rules to detect these anomalous patterns, and even when they can, they would likely not be robust to diverse incident types and changing nature of $\log$ lines as the software functionalities change over time. Hence, this makes a compelling case for
employing data-driven models and machine intelligence to mine and analyze this complex data-source to serve the end goals of incident detection.

## Log Analysis Workflow for Incident Detection

In order to handle the complex nature of the data, typically a series of steps need to be followed to meaningfully analyze logs to detect incidents. Starting with the raw log data or data streams, the log analysis workflow first does some preprocessing of the logs to make them amenable to ML models. This is typically followed by log parsing which extracts a loose structure from the semi-structured data and then grouping and partitioning the log lines into log sequences in order to model the sequence characteristics of the data. After this, the logs or log sequences are represented as a machine-readable matrix on which various log analysis tasks can be performed - like clustering and summarizing the huge log dumps into a few key $\log$ patterns for easy visualization or for detecting anomalous $\log$ patterns that can be indicative of an incident. Figure 7 provides an outline of the different steps in the log analysis wokflow. While some of these steps are more of engineering challenges, others are more AI-driven and some even employ a combination of machine learning and domain knowledge rules.

i) Log Preprocessing: This step typically involves customised filtering of specific regular expression patterns (like IP addresses or memory locations) that are deemed irrelevant for the actual log analysis. Other preprocessing steps like tokenization requires specialized handling of different wording styles and patterns arising due to the hybrid nature of logs consisting of both natural language and programming language constructs. For example a log line can contain a mix of text strings from source-code data having snake-case and camelCase tokens along with white-spaced tokens in natural language.

ii) Log Parsing: To enable downstream processing, unstructured $\log$ messages first need to be parsed into a structured event template (i.e. constant part that was actually designed by the developers) and parameters (i.e. variable part which contain the dynamic runtime information). Figure 8 provides one such example of parsing a single log line. In literature there have been heuristic methods for parsing as well as AIdriven methods which include traditional ML and also more recent neural models. The heuristic methods like Drain [63], IPLoM [64] and AEL [65] exploit known inductive bias on log structure while Spell [66] uses Longest common subsequence algorithm to dynamically extract log patterms. Out of these, Drain and Spell are most popular, as they scale well to industrial standards. Amongst the traditional ML methods, there are i) Clustering based methods like LogCluster [67], LKE [68], LogSig [69], SHISO [70], LenMa [71], LogMine [72] which assume that log message types coincide in similar groups ii) Frequent pattern mining and item-set mining methods SLCT [73], LFA [74] to extract common message types iii) Evolutionary optimization approaches like MoLFI [75]. On the other hand, recent neural methods include [76] - Neural
Transformer based models which use self-supervised Masked Language Modeling to learn log parsing vii) UniParser [77] an unified parser for heterogenous $\log$ data with a learnable similarity module to generalize to diverse logs across different systems. There are yet another class of log analysis methods [78], [79] which aim at parsing free techniques, in order to avoid the computational overhead of parsing and the errors cascading from erroneous parses, especially due to the lack of robustness of the parsing methods.

iii) Log Partitioning: After parsing the next step is to partition the log data into groups, based on some semantics where each group represents a finite chunk of $\log$ lines or $\log$ sequences. The main purpose behind this is to decompose the original log dump typically consisting of millions of log lines into logical chunks, so as to enable explicit modeling on these chunks and allow the models to capture anomaly patterns over sequences of log templates or log parameter values or both. Log partitioning can be of different kinds [20], [80] - Fixed or Sliding window based partitions, where the length of window is determined by length of log sequence or a period of time, and Identifier based partitions where logs are partitioned based on some identifier (e.g. the session or process they originate from). Figure 9 illustrates these different choices of $\log$ grouping and partitioning. A log event is eventually deemed to be anomalous or not, either at the level of a log line or a $\log$ partition.

iv) Log Representation: After log partitioning, the next step is to represent each partition in a machine-readable way (e.g. a vector or a matrix) by extracting features from them. This can be done in various ways [81], [80]- either by extracting specific handcrafted features using domain knowledge or through ii) sequential representation which converts each partition to an ordered sequence of $\log$ event ids ii) quantitative representation which uses count vectors, weighted by the term and inverse document frequency information of the log events iii) semantic representation captures the linguistic meaning from the sequence of language tokens in the log events and learns a high-dimensional embedding vector for each token in the dataset. The nature of $\log$ representation chosen has direct consequence in terms of which patterns of anomalies they can support - for example, for capturing keyword based anomalies, semantic representation might be key, while for anomalies related to template count and variable distribution, quantitative representations are possibly more appropriate. The semantic embedding vectors themselves can be either obtained using pretrained neural language models like GloVe, FastText, pretrained Transformer like BERT, RoBERTa etc or learnt using a trainable embedding layer as part of the target task.

v) Log Analysis tasks for Incident Detection: Once the logs are represented in some compact machine-interpretable form which can be easily ingested by AI models, a pipeline of log analysis tasks can be performed on it - starting with Log compression techniques using Clustering and Summarization, followed by Log based Anomaly Detection. In turn, anomaly detection can further enable downstream tasks in Incident

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-10.jpg?height=355&width=1795&top_left_y=178&top_left_x=165)

Fig. 7. Steps of the Log Analysis Workflow for Incident Detection

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-10.jpg?height=277&width=696&top_left_y=680&top_left_x=259)

Fig. 8. Example of Log Parsing

![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-10.jpg?height=469&width=653&top_left_y=1056&top_left_x=278)

Fig. 9. Different types of log partitioning

Management like Failure Prediction and Root Cause Analysis. In this section we discuss only the first two log analysis tasks which are pertinent to incident detection and leave failure prediction and RCA for the subsequent sections.

v.1) Log Compression through Clustering \& Summarization: This is a practical first-step towards analyzing the huge volumes of log data is Log Compression through various clustering and summarization techniques. The objective of this analysis serves two purposes - Firstly, this step can independently help the site reliability engineers and service owners during incident management by providing a practical and intuitive way of visualizing these massive volumes of complex unstructured raw log data. Secondly, the output of log clustering can directly be leveraged in some of the log based anomaly detection methods.

Amongst the various techniques of log clustering, [82], [67], [83] employ hierarchical clustering and can support online settings by constructing and retrieving from knowledge base of representative log clusters. [84], [85] use frequent pattern matching with dimension reduction techniques like PCA and locally sensitive hashing with online and streaming support. [86], [64], [87] uses efficient iterative or incremental clustering and partitioning techniques that support online and streaming logs and can also handle clustering of rare log instances. Another area of existing literature [88], [89], [90], [91] focus on log compression through summarization - where, for example, [88] uses heuristics like log event ids and timings to summarize and [89], [21] does openIE based triple extraction using semantic information and domain knowledge and rules to generate summaries, while [90], [91] use sequence clustering using linguistic rules or through grouping common event sequences.

v.2) Log Anomaly Detection: Perhaps the most common use of $\log$ analysis is for $\log$ based anomaly detection where a wide variety of models have been employed in both research and industrial settings. These models are categorized based on various factors i) the learning setting - supervised, semisupervised or unsupervised: While the semi-supervised models assume partial knowledge of labels or access to few anomalous instances, unsupervised ones train on normal log data and detect anomaly based on their prediction confidence. ii) the type of Model - Neural or traditional statistical non-neural models iii) the kinds of log representations used iv) Whether to use $\log$ parsing or parser free methods v) If using parsing, then whether to encode only the log template part or both template and parameter representations iv) Whether to restrict modeling of anomalies at the level of individual log lines or to support sequential modeling of anomaly detection over log sequences.

The nature of $\log$ representation employed and the kind of modeling used - both of these factors influence what type of anomaly patterns can be detected - for example keyword and variable value based anomalies are captured by semantic representation of log lines, while template count and variable distribution based anomaly patterns are more explicitly modeled through quantitative representations of $\log$ events. Similarly template sequence and time-interval based anomalies need sequential modeling algorithms which can handle log sequences.

Below we briefly summarize the body of literature dedicated to these two types of models - Statistical and Neural; and In Table III we provide a comparison of a more comprehensive list of existing anomaly detection algorithms and systems.

Statistical Models are the more traditional machine learning models which draw inference from various statistics underlying the training data. In the literature there have been various statistical ML models employed for this task under
different training settings. Amongst the supervised methods, [92], [93], [94] using traditional learning strategies of Linear Regression, SVM, Decision Trees, Isolation Forest with handcrafted features extracted from the entire logline. Most of these model the data at the level of individual log-lines and cannot not explicitly capture sequence level anomalies. There are also unsupervised methods like ii) dimension reduction techniques like Principal Component Analysis (PCA) [84] iii) clustering and drawing correlations between log events and metric data as in [67], [82], [95], [80]. There are also unsupervised pattern mining methods which include mining invariant patterns from singular value decomposition [96] and mining frequent patterns from execution flow and control flow graphs [97], [98], [99], [68]. Apart from these there are also systems which employ a rule engine built using domain knowledge and an ensemble of different ML models to cater to different incident types [20] and also heuristic methods for doing contrast analysis between normal and incidentindicating abnormal logs [100].

Neural Models, on the other hand are a more recent class of machine learning models which use artificial neural networks and have proven remarkably successful across numerous AI applications. They are particularly powerful in encoding and representing the complex semantics underlying in a way that is meaningful for the predictive task. One class of unsupervised neural models use reconstruction based self-supervised techniques to learn the token or line level representation, which includes i) Autoencoder models [101], [102] ii) more powerful self-attention based Transformer models [103] iv) specific pretrained Transformers like BERT language model [104], [105], [21]. Another offshoot of reconstruction based models is those using generative adversarial or GAN paradigm of training for e.g. [106], [107] using LSTM or Transformer based encoding. The other types of unsupervised models are forecasting based, which learn to predict the next log token or next log line in a self-supervised way - for e.g i) Recurrent Neural Network based models like LSTM [108], [109], [110], [18], [111] and GRU [104] or their attention based counterparts [81], [112], [113] ii) Convolutional Neural Network (CNN) based models [114] or more complex models which use Graph Neural Network to represent log event data [115], [116]. Both reconstruction and forecasting based models are capable of handling sequence level anomalies, it depends on the nature of training (i.e. whether representations are learnt at log line or token level) and the capacity of model to handle long sequences (e.g. amongst the above, Autoencoder models are the most basic ones).

Most of these models follow the practical setup of unsupervised training, where they train only non-anomalous $\log$ data. However, other works have also focused on supervised training of LSTM, CNN and Transformer models [111], [114], [78], [117], over anomalous and normal labeled data. On the other hand, [104], [110] use weak supervision based on heuristic assumptions for e.g. logs from external systems are considered anomalous. Most of the neural models use semantic token representations, some with pretrained fixed or trainable embeddings, initialized with GloVe, fastText or pretrained transformer based models, BERT, GPT, XLM etc. vi) Log Model Deployment: The final step in the log analysis workflow is deployment of these models in the actual industrial settings. It involves i) a training step, typically over offline log data dump, with or without some supervision labels collected from domain experts ii) online inference step, which often needs to handle practical challenges like nonstationary streaming data i.e. where the data distribution is not independently and identically distributed throughout the time. For tackling this, some of the more traditional statistical methods like [103], [95], [82], [84] support online streaming update while some other works can also adapt to evolving log data by incrementally building a knowledge base or memory or out-of-domain vocabulary [101]. On the other hand most of the unsupervised models support syncopated batched online training, allowing the model to continually adapt to changing data distributions and to be deployed on high throughput streaming data sources. However for some of the more advanced neural models, the online updation might be too computationally expensive even for regular batched updates.

Apart from these, there have also been specific work on other challenges related to model deployment in practical settings like transfer learning across logs from different domains or applications [110], [103], [18], [18], [118] under semi-supervised settings using only supervision from source systems. Other works focus on evaluating model robustness and generalization (i.e. how well the model adapts to) to unstable log data due to continuous logging modifications throughout software evolutions and updates [109], [111], [104]. They achieve these by adopting domain adversarial paradigms during training [18], [18] or using counterfactual explanations [118] or multi-task settings [21] over various log analysis tasks.

## Challenges \& Future Trends

Collecting supervision labels: Like most AIOps tasks, collecting large-scale supervision labels for training or even evaluation of log analysis problems is very challenging and impractical as it involves significant amount of manual intervention and domain knowledge. For log anomaly detection, the goal being quite objective, label collection is still possible to enable atleast a reliable evaluation. Whereas, for other $\log$ analysis tasks like clustering and summarization, collecting supervision labels from domain experts is often not even possible as the goal is quite subjective and hence these tasks are typically evaluated through the downstream log analysis or RCA task.

Imbalanced class problem: One of the key challenges of anomaly detection tasks, is the class imbalance, stemming from the fact that anomalous data is inherently extremely rare in occurrence. Additionally, various systems may show different kinds of data skewness owing to the diverse kinds of anomalies listed above. This poses a technical challenge both during model training with highly skewed data as well as choice of evaluation metrics, as Precision, Recall and FScore may not perform satisfactorily. Further at inference, thresholding over the anomaly score gets particularly chal-
lenging for unsupervised models. While for benchmarking purposes, evaluation metrics like AUROC (Area under ROC curve) can suffice, but for practical deployment of these models require either careful calibrations of anomaly scores or manual tuning or heuristic means for setting the threshold. This being quite sensitive to the application at hand, also poses realistic challenges when generalizing to heterogenous logs from different systems.

Handling large volume of data: Another challenge in log analysis tasks is handling the huge volumes of logs, where most large-scale cloud-based systems can generate petabytes of logs each day or week. This calls for log processing algorithms, that are not only effective but also lightweight enough to be very fast and efficient.

Handling non-stationary log data: Along with humongous volume, the natural and most practical setting of logs analysis is an online streaming setting, involving nonstationary data distribution - with heterogenous log streams coming from different inter-connected micro-services, and the software logging data itself evolving over time as developers naturally keep evolving software in the agile cloud development environment. This requires efficient online update schemes for the learning algorithms and specialized effort towards building robust models and evaluating their robustness towards unstable or evolving log data.

Handling noisy data: Annotating log data being extremely challenging even for domain experts, supervised and semi-supervised models need to handle this noise during training, while for unsupervised models, it can heavily mislead evaluation. Even though it affects a small fraction of logs, the extreme class imbalance aggrevates this problem. Another related challenge is that of errors compounding and cascading from each of the processing steps in the log analysis workflow when performing the downstream tasks like anomaly detection.

Realistic public benchmark datasets for anomaly detection: Amongst the publicly available log anomaly detection datasets, only a limited few contain anomaly labels. Most of those benchmarks have been excessively used in the literature and hence do not have much scope of furthering research. Infact, their biggest limitation is that they fail to showcase the diverse nature of incidents that typically arise in realworld deployment. Often very simple handcrafted rules prove to be quite successful in solving anomaly detection tasks on these datasets. Also, the original scale of these datasets are several orders of magnitude smaller than the real-world use-cases and hence not fit for showcasing the challenges of online or streaming settings. Further, the volume of unique patterns collapses significantly after the typical log processing steps to remove irrelevant patterns from the data. On the other hand, a vast majority of the literature is backed up by empirical analysis and evaluation on internal proprietary data, which cannot guarantee reproducibility. This calls for more realistic public benchmark datasets that can expose the real-world challenges of aiops-in-the-wild and also do a fair benchmarking across contemporary log analysis models.
Public benchmarks for parsing, clustering, summarization: Most of the log parsing, clustering and summarization literature only uses a very small subset of data from some of the public log datasets, where the oracle parsing is available, or in-house log datasets from industrial applications where they compare with oracle parsing methods that are unscalable in practice. This also makes fair comparison and standardized benchmarking difficult for these tasks.

Better log language models: Some of the recent advances in neural NLP models like transformer based language models BERT, GPT has proved quite promising for representing logs in natural language style and enabling various $\log$ analysis tasks. However there is more scope of improvement in building neural language models that can appropriately encode the semi-structured logs composed of fixed template and variable parameters without depending on an external parser.

Incorporating Domain Knowledge: While existing log anomaly detection systems are entirely rule-based or automated, given the complex nature of incidents and the diverse varieties of anomalies, a more practical approach would involve incorporating domain knowledge into these models either in a static form or dynamically, following a humanin-the-loop feedback mechanism. For example, in a complex system generating humungous amounts of logs, which kinds of incidents are more severe and which types of logs are more crucial to monitor for which kind of incidents. Or even at the level of loglines, domain knowledge can help understand the real-world semantics or physical significance of some of the parameters or variables mentioned in the logs. These aspects are often hard for the ML system to gauge on its own especially in the practical unsupervised settings.

Unified models for heterogenous logs: Most of the log analysis models are highly sensitive towards the nature of $\log$ preprocessing or grouping, needing customized preprocessing for each type of application logs. This alludes towards the need for unified models with more generalizable preprocessing layers that can handle heterogenous kinds of log data and also different types of log analysis tasks. While [21] was one of the first works to explore this direction, there is certainly more research scope for building practically applicable models for $\log$ analysis.

## C. Traces and Multimodal Incident Detection

## Problem Definition

Traces are semi-structured event logs with span information about the topological structure of the service graph. Trace anomaly detection relies on finding abnormal paths on the topological graph at given moments, as well as discovering abnormal information directly from trace event log text. There are multiple ways to process trace data. Traces usually have timestamps and associated sequential information so it can be covered into time-series data. Traces are also stored as trace event logs, containing rich text information. Moreover, traces store topological information which can be used to reconstruct the service graphs that represents the relation
among components of the systems. From the data perspective, traces can easily been turned into multiple data modalities. Thus, we combines trace-based anomaly detection with multimodal anomaly detection to discuss in this section. Recently, we can see with the help of multi-modal deep learning technologies, trace anomaly detection can combine different levels of information relayed by trace data and learn more comprehensive anomaly detection models [119][120].

## Empirical Approaches

Traces draw more attention in microservice system architectures since the topological structure becomes very complex and dynamic. Trace anomaly detection started from practical usages for large scale system debugging [121]. Empirical trace anomaly detection and RCA started with constructing trace graphs and identifying abnormal structures on the constructed graph. Constructing the trace graph from trace data is usually very time consuming, an offline component is designed to train and construct such trace graph. Apart from , to adapt to the usage requirements to detect and locate issues in large scale systems, trace anomaly detection and RCA algorithms usually also have an online part to support real-time service. For example, Cai et al.. released their study of a real-time trace-level diagnosis system, which is adopted by Alibaba datacenters. This is one of the very few studies to deal with real large distributed systems [122].

Most empirical trace anomaly detection work follow the offline and online design pattern to construct their graph models. In the offline modeling, unsupervised or semi-supervised techniques are utilized to construct the trace entity graphs, very similar to techniques in process discovery and mining domain. For example, PageRank has been used to construct web graphs in one of the early web graph anomaly detection works [123]. After constructing the trace entity graphs, a variety of techniques can be used to detect anomalies. One common way is to compare the current graph pattern to normal graph patterns. If the current graph pattern significantly deviates from the normal patterns, report anomalous traces.

An alternative approach is using data mining and statistical learning techniques to run dynamic analysis without constructing the offline trace graph. Chen et al. proposed Pinpoint [124, a framework for root cause analysis that using coarsegrained tagging data of real client requests at real-time when these requests traverse through the system, with data mining techniques. Pinpoint discovers the correlation between success / failure status of these requests and fault components. The entire approach processes the traces on-the-fly and does not leverage any static dependency graph models.

## Deep Learning Based Approaches

In recent years, deep learning techniques started to be employed in trace anomaly detection and RCA. Also with the help of deep learning frameworks, combining general trace graph information and the detailed information inside of each trace event to train multimodal learning models become possible.

Long-short term memory (LSTM) network [125] is a very popular neural network model in early trace and multimodal anomaly detection. LSTM is a special type of recurrent neural network (RNN) and has been proved to success in lots of other domains. In AIOps, LSTM is also commonly used in metric and log anomaly detection applications. Trace data is a natural fit with RNNs, majorly in two ways: 1) The topological order of traces can be modeled as event sequences. These event sequences can easily be transformed into model inputs of RNNs. 2) Trace events usually have text data that conveys rich information. The raw text, including both the structured and unstructured parts, can be transformed into vectors via standard tokenization and embedding techniques, and feed the RNN as model inputs. Such deep learning model architectures can be extended to support multimodal input, such as combining trace event vector with numerical time series values [119].

To better leverage the topological information of traces, graph neural networks have also been introduced in trace anomaly detection. Zhang et al. developed DeepTraLog, a trace anomaly detection technique that employs Gated graph neural networks [120]. DeepTraLog targets to solve anomaly detection problems for complex microservice systems where service entity relationships are not easy to obtain. Moreover, the constructed graph by GGNN training can also be used to localize the issue, providing additional root-cause analysis capability.

## Limitations

Trace data became increasingly attractive as more applications transitioned from monolithic to microservice architecture. There are several challenges in machine learning based trace anomaly detection.

Data quality. As far as we know, there are multiple trace collection platforms and the trace data format and quality are inconsistent across these platforms, especially in the production environment. To use these trace data for analysis, researchers and developers have to spend significant time and effort to clean and reform the data to feed machine learning models.

Difficult to acquire labels. It is very difficult to acquire labels for production data. For a given incident, labeling the corresponding trace requires identifying the incident occurring time and location, as well as the root cause which may be located in totally different time and location. Obtaining such full labels for thousands of incidents is extremely difficult. Thus, most of the existing trace analysis research still use synthetic data to evaluate the model performance. This brings more doubts whether the proposed solution can solve problems in real production.

No sufficient multimodal and graph learning models. Trace data are complex. Current trace analysis simplifies trace data into event sequences or time-series numerical values, even in the multimodal settings. However, these existing model architectures did not fully leverage all information of trace data in one place. Graph-based learning can potentially be a solution but discussions of this topic are still very limited.

Offline model training. The deep learning models in existing research relies on offline model training, partially because model training is usually very time consuming and
contradicts with the goal of real-time serving. However, offline model training brings static dependencies to a dynamic system. Such dependencies may cause additional performance issues.

## Future Trends

Unified trace data Recently, OpenTelemetry leads the effort to unify observability telemetry data, including metrics, logs, traces, etc., across different platforms. This effort can bring huge benefits to future trace analysis. With more unified data models, AI researchers can more easily acquire necessary data to train better models. The trained model can also be easily plug-and-play by other parties, which can further boost model quality improvements.

Unified engine for detection and RCA Trace graph contains rich information about the system at a given time. With the help of trace data, incident detection and root cause localization can be done within one step, instead of the current two consecutive steps. Existing work has demonstrated that by simply examining the constructed graph, the detection model can reveal sufficient information to locate the root causes |120].

Unified models for multimodal telemetry data Trace data analysis brings the opportunities for researchers to create a holistic view of multiple telemetry data modality since traces can be converted into text sequence data and time-series data. The learnings can be extended to include logs or metrics from different sources. Eventually we can expect unified learning models that can consume multimodal telemetry data for incident detection and RCA.

Online Learning Modern systems are dynamic and everchanging. Current two-step solution relies on offline model training and online serving or inference. Any system evolution between two offline training cycles could cause potential issues and damage model performance. Thus, supporting online learning is critical to guarantee high performance in real production environments.

## V. FAILURE Prediction

Incident Detection and Root-Cause Analysis of Incidents are more reactive measures towards mitigating the effects of any incident and improving service availability once the incident has already occurred. On the other hand, there are other proactive actions that can be taken to predict if any potential incident can happen in the immediate future and prevent it from happening. Failures in software systems are such kind of highly disruptive incidents that often start by showing symptoms of deviation from the normal routine behavior of the required system functions and typically result in failure to meet the service level agreement. Failure prediction is one such proactive task in Incident Management, whose objective is to continuously monitor the system health by analyzing the different types of system data (KPI metrics, logging and trace data) and generate early warnings to prevent failures from occurring. Consequently, in order to handle the different kinds of telemetry data sources, the task of predicting failures can be tailored to metric based and log based failure prediction. We describe these two in details in this section.

## A. Metrics based Failure Prediction

Metric data are usually fruitful in monitoring system. It is straightforward to directly leverage them to predict the occurrence of the incident in advance. As such, some proactive actions can be taken to prevent it from happening instead of reducing the time for detection. Generally, it can be formulated as the imbalanced binary classification problem if failure labels are available, and formulated as the time series forecasting problem if the normal range of monitored metrics are defined in advance. In general, failure prediction [126] usually adopts machine learning algorithms to learn the characteristics of historical failure data, build a failure prediction model, and then deploy the model to predict the likelihood of a failure in the future.

## Methods

General Failure Prediction: Recently, there are increasing efforts on considering general failure incident prediction with the failure signals from the whole monitoring system. [127] collected alerting signals across the whole system and discovered the dependence relationships among alerting signals, then the gradient boosting tree based model was adopted to learn failure patterns. [128] proposed an effective feature engineering process to deal with complex alert data. It used multi-instance learning and handle noisy alerts, and interpretable analysis to generate an interpretable prediction result to facilitate the understanding and handling of incidents.

Specific Type Failure Prediction: In contrast, some works In contrast, 127] and [128 aim to proactively predict various specific types of failures. [129] extracted statistical and textual features from historical switch logs and applied random forest to predict switch failures in data center networks. [130] collected data from SMART [131] and system-level signals, and proposed a hybrid of LSTM and random forest model for node failure prediction in cloud service system. [132] developed a disk error prediction method via a cost-sensitive ranking models. These methods target at the specific type of failure prediction, and thus are limited in practice.

## Challenges and Future Trends

While conventional supervised learning for classification or regression problems can be used to handle failure prediction, it needs to overcome the following main challenges. First, datasets are usually very imbalanced due to the limited number of failure cases. This poses a significant challenge to the prediction model to achieve high precision and high recall simultaneously. Second, the raw signals are usually noisy, not all information before incident is helpful. How to extract omen features/patterns and filter out noises are critical to the prediction performance. Third, it is common for a typical system to generate a large volume of signals per minute, leading to the challenge to update prediction model in the streaming way and handle the large-scale data with limited computation resources. Fourth, post-processing of failure prediction is very important for failure management system to improve availability. For example, providing interpretable failure prediction can facilitate engineers to take appropriate action for it.

## B. Logs based Incident Detection

Like Incident Detection and Root Cause Analysis, Failure Prediction is also an extremely complex task, especially in enterprise level systems which comprise of many distributed but inter-connected components, services and micro-services interacting with each other asynchronously. One of the main complexities of the task is to be able to do early detection of signals alluding towards a major disruption, even while the system might be showing only slight or manageable deviations from its usual behavior. Because of this nature of the problem, often monitoring the KPI metrics alone may not suffice for early detection, as many of these metrics might register a late reaction to a developing issue or may not be fine-grained enough to capture the early signals of an incident. System and software logs, on the other hand, being an allpervasive part of systems data continuously capture rich and very detailed runtime information that are often pertinent to detecting possible future failures.

Thus various proactive log based analysis have been applied in different industrial applications as a continuous monitoring task and have proved to be quite effective for a more finegrained failure prediction and localizing the source of the potential failure. It involves analyzing the sequences of events in the $\log$ data and possibly even correlating them with other data sources like metrics in order to detect anomalous event patterns that indicate towards a developing incident. This is typically achieved in literature by employing supervised or semi-supervised machine learning models to predict future failure likelihood by learning and modeling the characteristics of historical failure data. In some cases these models can also be additionally powered by domain knowledge about the intricate relationships between the systems. While this task has not been explored as popularly as Log Anomaly Detection and Root Cause Analysis and there are fewer public datasets and benchmark data, software and systems maintainance logging data still plays a very important role in predicting potential future failures. In literature, generally the failure prediction task over log data has been employed in broadly two types of systems - homogenous and heterogenous.

## Failure Prediction in Homogenous Systems

In homogenous systems, like high-performance computing systems or large-scale supercomputers, this entails prediction of independent failures, where most systems leverage sequential information to predict failure of a single component.

Time-Series Modeling: Amongst homogenous systems, [133], [134] extract system health indicating features from structured logs and modeled this as time series based anomaly forecasting problem. Similarly [135] extracts specific patterns during critical events through feature engineering and build a supervised binary classifier to predict failures. [136] converts unstructured logs into templates through parsing and apply feature extraction and time-series modeling to predict surge, frequency and seasonality patterns of anomalies.

Supervised Classifiers Some of the older works predict failures in a supervised classification setting using traditional machine learning models like support vector machines, nearest-neighbor or rule-based classifiers [137], [93], [138], or ensemble of classifiers [93] or hidden semi-markov model based classifier [139] over features handcrafted from log event sequences or over random indexing based log encoding while [140], [141] uses deep recurrent neural models like LSTM over semantic representations of logs. [142] predict and diagnose failures through first failure identification and causality based filtering to combine correlated events for filtering through association rule-mining method.

## Failure Prediction in Heterogenous Systems

In heterogenous systems, like large-scale cloud services, especially in distributed micro-service environment, outages can be caused by heterogenous components. Most popular methods utilize knowledge about the relationship and dependency between the system components, in order to predict failures. Amongst such systems, [143] constructed a Bayesian network to identify conditional dependence between alerting signals extracted from system logs and past outages in offline setting and used gradient boosting trees to predict future outages in the online setting. [144] uses a ranking model combining temporal features from LSTM hidden states and spatial features from Random Forest to rank relationships between failure indicating alerts and outages. [145] trains trace-level and micro-service level prediction models over handcrafted features extracted from trace logs to detect three common types of micro-service failures.

## VI. Root Cause AnAlYsiS

Root-cause Analysis (RCA) is the process to conduct a series of actions to discover the root causes of an incident. RCA in DevOps focuses on building the standard process workflow to handle incidents more systematically. Without AI, RCA is more about creating rules that any DevOps member can follow to solve repeated incidents. However, it is not scalable to create separate rules and process workflow for each type of repeated incident when the systems are large and complex. AI models are capable to process high volume of input data and learn representations from existing incidents and how they are handled, without humans to define every single details of the workflow. Thus, AI-based RCA has huge potential to reform how root cause can be discovered.

In this section, we discuss a series of AI-based RCA topics, separeted by the input data modality: metric-based, log-based, trace-based and multimodal RCA.

## A. Metric-based RCA

## Problem Definition

With the rapidly growing adoption of microservices architectures, multi-service applications become the standard paradigm in real-world IT applications. A multi-service application usually contains hundreds of interacting services, making it harder to detect service failures and identify the root causes. Root cause analysis (RCA) methods leverage the KPI metrics monitored on those services to determine the root causes when a system failure is detected, helping engineers and

SREs in the troubleshooting process $^{\text {The key idea behind }}$ RCA with KPI metrics is to analyze the relationships or dependencies between these metrics and then utilize these relationships to identify root causes when an anomaly occurs. Typically, there are two types of approaches: 1) identifying the anomalous metrics in parallel with the observed anomaly via metric data analysis, and 2) discovering a topology/causal graph that represent the causal relationships between the services and then identifying root causes based on it.

## Metric Data Analysis

When an anomaly is detected in a multi-service application, the services whose KPI metrics are anomalous can possibly be the root causes. The first approach directly analyzes these KPI metrics to determine root causes based on the assumption that significant changes in one or multiple KPI metrics happen when an anomaly occurs. Therefore, the key is to identify whether a KPI metric has pattern or magnitude changes in a look-back window or snapshot of a given size at the anomalous timestamp.

Nguyen et al. [146], [147] propose two similar RCA methods by analyzing low-level system metrics, e.g., CPU, memory and network statistics. Both methods first detect abnormal behaviors for each component via a change point detection algorithm when a performance anomaly is detected, and then determine the root causes based on the propagation patterns obtained by sorting all critical change points in a chronological order. Because a real-world multi-service application usually have hundreds of KPI metrics, the change point detection algorithm must be efficient and robust. [146] provides an algorithm by combining cumulative sum charts and bootstrapping to detect change points. To identify the critical change point from the change points discovered by this algorithm, they use a separation level metric to measure the change magnitude for each change point and extract the critical change point whose separation level value is an outlier. Since the earliest anomalies may have propagated from their corresponding services to other services, the root causes are then determined by sorting the critical change points in a chronological order. To further improve root cause pinpointing accuracy, [147] develops a new fault localization method by considering both propagation patterns and service component dependencies.

Instead of change point detection, Shan et al. [148] developed a low-cost RCA method called $\epsilon$-Diagnosis to detect root causes of small-window long-tail latency for web services. $\epsilon$ Diagnosis assumes that the root cause metrics of an abnormal service have significantly changes between the abnormal and normal periods. It applies the two-sample test algorithm and $\epsilon$ statistics for measuring similarity of time series to identify root causes. In the two-sample test, one sample (normal sample) is drawn from the snapshot during the normal period while the other sample (anomaly sample) is drawn during the anomalous period. If the difference between the anomaly sample and the normal sample are statistically significant, the corresponding metrics of the samples are potential root causes.

*A good survey for anomaly detection and RCA in cloud applications [22]

## Topology or Causal Graph-based Analysis

The advantage of metric data analysis methods is the ability of handling millions of metrics. But most of them don't consider the dependencies between services in an application. The second type of RCA approaches leverages such dependencies, which usually involves two steps, i.e., constructing topology/causal graphs given the KPI metrics and domain knowledge, and extracting anomalous subgraphs or paths given the observed anomalies. Such graphs can either be reconstructed from the topology (domain knowledge) of a certain application ([149], [150], [151], [152]) or automatically estimated from the metrics via causal discovery techniques ([153], [154], [155], [156], [157], [158], [159]). To identify the root causes of the observed anomalies, random walk (e.g., [160], [156], [153]), page-rank (e.g., [150]) or other techniques can be applied over the discovered topology/causal graphs.

When the service graphs (the relationships between the services) or the call graphs (the communications among the services) are available, the topology graph of a multi-service application can be reconstructed automatically, e.g., [149], [150]. But such domain knowledge is usually unavailable or partially available especially when investigating the relationships between the KPI metrics instead of API calls. Therefore, given the observed metrics, causal discovery techniques, e.g., [161], [162], [163] play a significant role in constructing the causal graph describing the causal relationships between these metrics. The most popular causal discovery algorithm applied in RCA is the well-known PC-algorithm [161] due to its simplicity and explainability. It starts from a complete undirected graph and eliminates edges between the metrics via conditional independence test. The orientations of the edges are then determined by finding $\mathrm{V}$-structures followed by orientation propagation. Some variants of the PC-algorithm [164], [165], [166] can also be applied based on different data properties.

Given the discovered causal graph, the possible root causes of the observed anomalies can be determined by random walk. A random walk on a graph is a random process that begins at some node, and randomly moves to another node at each time step. The probability of moving from one node to another is defined in the the transition probability matrix. Random walk for RCA is based on the assumption that a metric that is more correlated with the anomalous KPI metrics is more likely to be the root cause. Each random walk starts from one anomalous node corresponding to an anomalous metric, then the nodes visited the most frequently are the most likely to be the root causes. The key of random walk approaches is to determine the transition probability matrix. Typically, there are three steps for computing the transition probability matrix, i.e., forward step (probability of walking from a node to one of its parents), backward step (probability of walking from a node to one of its children) and self step (probability of staying in the current node). For example, [153], [158], [159], [150] computes these probabilities based on the correlation of each metric with the detected anomalous metrics during the anomaly period. But correlation based random walk may not accurately localize root cause [156]. Therefore, [156] proposes to use the partial correlations instead of correlations to compute the transition
probabilities, which can remove the effect of the confounders of two metrics.

Besides random walk, other causal graph analysis techniques can also be applied. For example, [157], [155] find root causes for the observed anomalies by recursively visiting all the metrics that are affected by the anomalies, e.g., if the parents of an affected metric are not affected by the anomalies, this metric is considered a possible root cause. [167] adopts a search algorithm based on a breadth-first search (BFS) algorithm to find root causes. The search starts from one anomalous KPI metric and extracts all possible paths outgoing from this metric in the causal graph. These paths are then sorted based on the path length and the sum of the weights associated to the edges in the path. The last nodes in the top paths are considered as the root causes. [168] considers counterfactuals for root cause analysis based on the causal graph, i.e., given a functional causal model, it finds the root cause of a detected anomaly by computing the contribution of each noise term to the anomaly score, where the contributions are symmetrized using the concept of Shapley values.

## Limitations

Data Issues For a multi-service application with hundreds of KPI metrics monitored on each service, it is very challenging to determine which metrics are crucial for identifying root causes. The collected data usually doesn't describe the whole picture of the system architecture, e.g., missing some important metrics. These missing metrics may be the causal parents of other metrics, which violates the assumption of PC algorithms that no latent confounders exist. Besides, due to noises, non-stationarity and nonlinear relationships in realworld KPI metrics, recovering accurate causal graphs becomes even harder.

Lack of Domain Knowledge The domain knowledge about the monitored application, e.g., service graphs and call graphs, is valuable to improve RCA performance. But for a complex multi-service application, even developers may not fully understand the meanings or the relationships of all the monitored metrics. Therefore, the domain knowledge provided by experts is usually partially known, and sometimes conflicts with the knowledge discovered from the observed data.

Causal Discovery Issues The RCA methods based on causal graph analysis leverage causal discovery techniques to recover the causal relationships between KPI metrics. All these techniques have certain assumptions on data properties which may not be satisfied with real-world data, so the discovered causal graph always contains errors, e.g., incorrect links or orientations. In recent years, many causal discovery methods have been proposed with different assumptions and characteristics, so that it is difficult to choose the most suitable one given the observed data.

Human in the Loop After DevOps or SRE teams receive the root causes identified by a certain RCA method, they will do further analysis and provide feedback about whether these root causes make sense. Most RCA methods cannot leverage such feedback to improve RCA performance, or provide explanations why the identified root causes are incorrect.

Lack of Benchmarks Different from incident detection problems, we lack benchmarks to evaluate RCA performance, e.g., few public datasets with groundtruth root causes are available, and most previous works use private internal datasets for evaluation. Although some multi-service application demos/simulators can be utilized to generate synthetic datasets for RCA evaluation, the complexity of these demo applications is much lower than real-world applications, so that such evaluation may not reflect the real performance in practice. The lack of public real-world benchmarks hampers the development of new RCA approaches.

## Future Trends

RCA Benchmarks Benchmarks for evaluating the performance of RCA methods are crucial for both real-world applications and academic research. The benchmarks can either be a collection of real-world datasets with groundtruth root causes or some simulators whose architectures are close to real-world applications. Constructing such large-scale realworld benchmarks is essential for boosting novel ideas or approaches in RCA.

Combining Causal Discovery and Domain Knowledge The domain knowledge provided by experts are valuable to improve causal discovery accuracy, e.g., providing required or forbidden causal links between metrics. But sometimes such domain knowledge introduces more issues when recovering causal graphs, e.g., conflicts with data properties or conditional independence tests, introducing cycles in the graph. How to combine causal discovery and expert domain knowledge in a principled manner is an interesting research topic.

Putting Human in the Loop Integrating human interactions into RCA approaches is important for real-world applications. For instance, the causal graph can be built in an iterative way, i.e., an initial causal graph is reconstructed by a certain causal discovery algorithm, and then users examine this graph and provide domain knowledge constraints (e.g., which relationships are incorrect or missing) for the algorithm to revise the graph. The RCA reports with detailed analysis about incidents created by DevOps or SRE teams are valuable to improve RCA performance. How to utilize these reports to improve RCA performance is another importance research topic.

## B. Log-based RCA

## Problem Definition

Triaging and root cause analysis is one of the most complex and critical phases in the Incident Management life cycle. Given the nature of the problem which is to investigate into the origin or the root cause of an incident, simply analyzing the end KPI metrics often do not suffice. Especially in a microservice application setting or distributed cloud environment with hundreds of services interacting with each other, RCA and failure diagnosis is particularly challenging. In order to localize the root cause in such complex environments, engineers, SREs and service owners typically need to investigate into core system data. Logs are one such ubiquitous forms of systems data containing rich runtime information. Hence one of the ultimate objectives of log analysis tasks is to enable triaging of incident and localization of root cause to diagnose faults and failures.

Starting with heterogenous log data from different sources and microservices in the system, typical log-based aiops workflows first have a layer of log processing and analysis, involving log parsing, clustering, summarization and anomaly detection. The log analysis and anomaly detection can then cater to a causal inference layer that analyses the relationships and dependencies between log events and possibly detected anomalous events. These signals extracted from logs within or across different services can be further correlated with other observability data like metrics, traces etc in order to detect the root cause of an incident. Typically this involves constructing a causal graph or mining a knowledge graph over the log events and correlating them with the KPI metrics or with other forms of system data like traces or service call graphs. Through these, the objective is to analyze the relationships and dependencies between them in order to eventually identify the possible root causes of an anomaly. Unlike the more concrete problems like $\log$ anomaly detection, log based root cause analysis is a much more open-ended task. Subsequently most of the literature on $\log$ based RCA has been focused on industrial applications deployed in real-world and evaluated with internal benchmark data gathered from in-house domain experts.

## Typical types of Log RCA methods

In literature, the task of $\log$ based root cause analysis have been explored through various kinds of approaches. While some of the works build a knowledge graph and knowledge and leverage data mining based solutions, others follow fundamental principles from Causal Machine learning or and causal knowledge mining. Other than these, there are also log based RCA systems using traditional machine learning models which use feature engineering or correlational analysis or supervised classifier to detect the root cause.

Handcrafted features based methods: [169] uses handcrafted feature engineering and probabilistic estimation of specific types of root causes tailored for Spark logs. [170] uses frequent item-set mining and association rule mining on feature groups for structured logs.

Correlation based Methods: [171], [172] localizes root cause based on correlation analysis using mutual information between anomaly scores obtained from logs and monitored metrics. Similarly [173] use PCA, ICA based correlation analysis to capture relationships between logs and consequent failures. [84], [174] uses PCA to detect abnormal system call sequences which it maps to application functions through frequent pattern mining.[175] uses LSTM based sequential modeling of log templates identified through pattern matching over clusters of similar logs, in order to predict failures.

Supervised Classifier based Methods: [176] does automated detection of exception logs and comparison of new error patterns with normal cloud behaviours on OpenStack by learning supervised classifiers over statistical and neural representations of historical failure logs. [177] employs statistical technique on the data distribution to identify the fine-grained category of a performance problem and fast matrix recovery RPCA to identify the root cause. [178], [179] uses KNN or its supervised versions to identify loglines that led to a failure.
Knowledge Mining based Methods: [180], [181] takes a different approach of summarizing log events into an entityrelation knowledge graph by extracting custom entities and relationships from log lines and mining temporal and procedural dependencies between them from the overall log dump. While this gives a more structured representation of the log summary, it is also an intuitive way of aggregating knowledge from logs, it is also a way to bridge the knowledge gap developer community who creates the log data and the site reliability engineers who typically consume the log data when investigating incidents. However, eventually the end goal of constructing this knowledge graph representation of logs is to facilitate RCA. While these works do provide use-cases like case-studies on RCA for this vision, but they leave ample scope of research towards a more concrete usage of this kind of knowledge mining in RCA.

Knowledge Graph based Methods: Amongst knowledge graph based methods, [182] diagnoses and triages performance failure issues in an online fashion by continuously building a knowledge base out of rules extracted from a random forest constructed over log data using heuristics and domain knowledge. [151] constructs a system graph from the combination of KPI metrics and log data. Based on the detected anomalies from these data sources, it extracts anomalous subgraphs from it and compares them with the normal system graph to detect the root cause. Other works mine normal log patterns [183] or time-weighted control flow graphs [99] from normal executions and on estimates divergences from them to executions during ongoing failures to suggest root causes. [184], [185], [186] mines execution sequences or user actions 187] either from normal and manually injected failures or from good or bad performing systems, in a knowledge base and utilizes the assumption that similar faults generate similar failures to match and diagnose type of failure. Most of these knowledge based approaches incrementally expand their knowledge or rules to cater to newer incident types over time.

Causal Graph based Methods: [188] uses a multivariate time-series modeling over logs by representing them as error event count. This work then infers its causal relationship with KPI error rate using a pagerank style centrality detection in order to identify the top root causes. [167] constructs a knowledge graph over operation and maintenance entities extracted from logs, metrics, traces and system dependency graphs and mines causal relations using PC algorithm to detect root causes of incidents. [189] uses a Knowledge informed Hierarchical Bayesian Network over features extracted from metric and log based anomaly detection to infer the root causes. [190] constructs dynamic causality graph over events extracted from logs, metrics and service dependency graphs. [191] similarly constructs a causal dependency graph over log events by clustering and mining similar events and use it to infer the process in which the failure occurs.

Also, on a related domain of network analysis, [192], [193], [194] mines causes of network events through causal analysis on network logs by modeling the parsed log template counts as a multivariate time series. [195], [156] use causality inference on KPI metrics and service call graphs to localize
root causes in microservice systems and one of the future research directions is to also incorporate unstructured logs to such causal analysis.

## Challenges \& Future Trends Collecting supervision la-

bels Being a complex and open-ended task, it is challenging and requires a lot of domain expertise and manual effort to collect supervision labels for root cause analysis. While a small scale supervision can still be availed for evaluation purposes, reaching the scale required for training these models is simply not practical. At the same time, because of the complex nature of the problem, completely unsupervised models often perform quite poorly. Data quality: The workflow of RCA over hetero-

geneous unstructured log data typically involves various different analysis layers, preprocessing, parsing, partitioning and anomaly detection. This results in compounding and cascading of errors (both labeling errors as well as model prediction errors) from these components, needing the noisy data to be handled in the RCA task. In addition to this, the extremely challenging nature of RCA labeling task further increases the possibility of noisy data. Imbalanced class problem: RCA on

huge voluminous logs poses an additional problem of extreme class imbalance - where out of millions of log lines or log templates, a very sparse few instances might be related to the true root cause. Generalizability of models: Most of the exist-

ing literature on RCA tailors their approach very specifically towards their own application and cannot be easily adopted even by other similar systems. This alludes towards need for more generalizable architectures for modeling the RCA task which in turn needs more robust generalizable $\log$ analysis models that can handle hetergenous kinds of log data coming from different systems. Continual learning framework: One

of the challenging aspects of RCA in the distributed cloud setting is the agile environment, leading to new kinds of incidents and evolving causation factors. This kind of nonstationary learning setting poses non-trivial challenges for RCA but is indeed a crucial aspect of all practical industrial applications. Human-in-the-loop framework: While neither

completely supervised or unsupervised settings is practical for this task, there is need for supporting human-in-the-loop framework which can incorporate feedbacks from domain experts to improve the system, especially in the agile settings where causation factors can evolve over time. Realistic public

benchmarks: Majority of the literature in this area is focused on industrial applications with in-house evaluation setting. In some cases, they curate their internal testbed by injecting failures or faults or anomalies in their internal simulation environment (for e.g. injecting CPU, memory, network and Disk anomalies in Spark platforms) or in popular testing settings (like Grid5000 testbed or open-source microservice applications based on online shopping platform or train ticket booking or open source cloud operating system OpenStack). Other works evaluate by deploying their solution in realworld setting in their in-house cloud-native application, for e.g. on IBM Bluemix platform, or for Facebook applications or over hundreds of real production services at big data cloud computing platforms like Alibaba or thousands of services at e-commerce enterprises like eBay. One of the striking limitations in this regard is the lack of any reproducible opensource public benchmark for evaluating log based RCA in practical industrial settings. This can hinder more open ended research and fair evaluation of new models for tackling this challenging task.

## C. Trace-based and Multimodal RCA

Problem Definition. Ideally, RCA for a complex system needs to leverage all kind of available data, including machine generated telemetry data and human activity records, to find potential root causes of an issue. In this section we discuss trace-based RCA together with multi-modal RCA. We also include studies about RCA based on human records such as incident reports. Ultimately, the RCA engine should aim to process any data types and discover the right root causes.

## RCA on Trace Data

In previous section (Section IV-C) we discussed trace can be treated as multimodal data for anomaly detection. Similar to trace anomaly detection, trace root cause analysis also leverages the topological structure of the service map. Instead of detecting abnormal traces or paths, trace RCA usually started after issues were detected. Trace RCA techniques help ease troubleshooting processes of engineers and SREs. And trace RCA can be triggered in a more ad-hoc way instead of running continuously. This differentiates the potential techniques to be adopted from trace anomaly detection.

Trace Entity Graph. From the technical point of view, trace RCA and trace anomaly detection share similar perspectives. To our best knowledge, there are not too many existing works talking about trace RCA alone. Instead, trace RCA serves as an additional feature or side benefit for trace anomaly detection in either empirical approaches [121] [196] or deep learning approaches [120] [197]. In trace anomaly detection, the constructed trace entity graph (TEG) after offline training provides a clean relationship between each component in the application systems. Thus, besides anomaly detection, 122 implemented a real-time RCA algorithm that discovers the deepest root of the issues via relative importance analysis after comparing the current abnormal trace pattern with normal trace patterns. Their experiment in the production environment demonstrated this RCA algorithm can achieve higher precision and recall compared to naive fixed threshold methods. The effectiveness of leverage trace entity graph for root cause analysis is also proven in deep learning based trace anomaly detection approaches. Liu et al. [198] proposed a multimodal LSTM model for trace anomaly detection. Then the RCA algorithm can check every anomalous trace with the model training traces and discover root cause by localizing the next called microservice which is not in the normal call paths. This algorithm performs well for both synthetic dataset and production datasets of four large production services, according to the evaluation of this work.

Online Learning. An alternative approach is using data mining and statistical learning techniques to run dynamic analysis without constructing the offline trace graph. Traditional trace management systems usually provides basic analytical capabilities to diagnose issues and discover root causes [199]. Such analysis can be performed online without costly model training process. Chen et al. proposed Pinpoint [124], a framework for root cause analysis that using coarsegrained tagging data of real client requests at real-time when these requests traverse through the system, with data mining techniques. Pinpoint discovers the correlation between success / failure status of these requests and fault components. The entire approach processes the traces on-the-fly and does not leverage any static dependency graph models. Another related area is using trouble-shooting guide data, where [200] recommends troubleshooting guide based on semantic similarity with incident description while [201] focuses on automation of troubleshooting guides to execution workflows, as a way to remediate the incident.

## RCA on Incident Reports

Another notable direction in AIOps literature has been mining useful knowledge from domain-expert curated data (incident report, incident investigation data, bug report etc) towards enabling the final goals of root cause analysis and automated remediation of incidents. This is an open ended task which can serve various purposes - structuring and parsing unstructured or semi-structured data and extracting targeted information or topics from them (using topic modeling or information extraction) and mining and aggregating knowledge into a structured form.

The end-goal of these tasks is majorly root cause analysis, while some are also focused on recommending remediation to mitigate the incident. Especially since in most cloudbased settings, there is an increasing number of incidents that occur repeatedly over time showing similar symptoms and having similar root causes. This makes mining and curating knowledge from various data sources, very crucial, in order to be consumed by data-driven AI models or by domain experts for better knowledge reuse.

Causality Graph. [202] extracts and mines causality graph from historical incident data and uses human-in-the-loop supervision and feedback to further refine the causality graph. [203] constructs an anomaly correlation graph, FacGraph using a distributed frequent pattern mining algorithm. [204] recommends appropriate healing actions by adapting remediations retrieved from similar historical incidents. Though the end task involves remediation recommendation, the system still needs to understand the nature of incident and root cause in order to retrieve meaningful past incidents.

Knowledge Mining. [205], [206] mines knowledge graph from named entity and relations extracted from incident reports using LSTM based CRF models. [207] extracts symptoms, root causes and remediations from past incident investigations and builds a neural search and knowledge graph to facilitate a retrieval based root cause and remediation recommendation for recurring incidents.

## Future Trends

More Efficient Trace Platform. Currently there are very limited studies in trace related topics. A fundamental challenge is about the trace platforms. There are bottlenecks in collection, storage, query and management of trace data. Traces are usually at a much larger scale than logs and metrics. How to more efficiently collect, store and retrieve trace data is very critical to the success of trace root cause analysis.

Online Learning. Compared to trace anomaly detection, online learning plays a more important role for trace RCA, especially for large cloud systems. An RCA tool usually needs to analyze the evidence on the fly and correlate the most suspicious evidence to the ongoing incidents, this approach is very time sensitive. For example, we know trace entity graph (TEG) can achieve accurate trace RCA but the preassumpiton is the TEG is reflecting the current status of the system. If offline training is the only way to get TEG, the performance of such approaches in real-world production environments is always questionable. Thus, using online learning to obtain the TEG is a much better way to guarantee high performance in this situation.

Causality Graphs on Multimodal Telemetries. The most precious information conveyed by trace data is the complex topological order of large systems. Without traces, causal analysis for system operations relies on temporal and geometrical correlations to infer causal relationships, and practically very few existing causal inference can be adopted in real-world systems. However, with traces, it is very convenient to obtain the ground truth of how requests flow through the entire system. Thus, we believe higher quality causal graphs will be much easier achievable if it can be learned by multimodel telemetry data.

Complete Knowledge Graph of Systems. Currently knowledge mining has been tried for single data type. However, to reflect the full picture of a complex system, the AI models need to mining knowledge from any kind of data types, including metrics, logs, traces, incident reports and other system activity records, then construct a knowledge graph with complete system information.

## VII. AUTOMATED ACTIONS

While both incident detection and RCA capabilities of AIOps help provide information about ongoing issues, taking the right actions is the step that solve the problems. Without automation to take actions, human operators will still be needed in every single ops task. Thus, automated actions is critical to build fully-automated end-to-end AIOps systems. Automated actions contributes to both short-term actions and longer-term actions: 1) short-term remediation: immediate actions to quickly remediate the issue, including server rebooting, live migration, automated scaling, etc.; and 2) longer-term resolutions: actions or guidance for tasks such as code bug fixing, software updating, hard build-out and resource allocation optimization. In this section, we discuss three common types of automated actions: automated remediation, auto-scaling and resource management.

## A. Automated Remediation

## Problem Definition

Besides continuously monitoring the IT infrastructure, detecting issues and discovering root causes, remediating issues with minimum, or even no human intervention, is the path towards the next generation of fully automated AIOps. Automated issue remediation (Auto-remediation) is taking a series of actions to resolve issues by leveraging known information, existing workflows and domain knowledge. Auto-remediation is a concept already adopted in many IT operation scenarios, including cloud computing, edge computing, SaaS, etc.

Traditional auto-remediation processes are based on a variety of well-defined policies and rules to get which workflows to use for a given issue. While machine learning driven auto-remediation means utilizing machine learning models to decide the best action workflows to mitigate or resolve the issue. ML based auto-remediation is exceptionally useful in large scale cloud systems or edge-computing systems where it's impossible to manually create workflows for all issue categories.

## Existing Work

End-to-end auto-remediation solutions usually contain three main components: anomaly or issue detection, root cause analysis and remediation engine [208]. This means successful autoremediation solutions highly rely on the quality of anomaly detection and root cause analysis, which we've already discussed in the above sections. Besides, the remediation engine should be able to learn from the analysis results, make decisions and execute.

Knowledge learning. The knowledge here refers to a variety of categories. Anomaly detection and root cause analysis for this specific issue contributes to a majority of the learnable knowledge [208]. Remediation engine uses these information to locate and categorize the issue. Besides, the human activity records (such as tickets, bug fixing logs) of past issues are also significant for the remediation to learn the full picture of how issues were handled in history. In Sections VI-A VI-B VI-C we discussed about mining knowledge graphs from system metrics, logs and human-in-the-loop records. A high quality knowledge graph which clearly describes the relationship of system components.

Decision making and execution. Levy et al. [209] proposed Narya, a system to handle failure remediation for running virtual machines in cloud systems. For a given issue where the host is predicted to fail, the remediation engine needs to decide what is the best action to take from a few options such as live migration, soft reboot, service healing, etc. The decision on which actions to take are made via A/B testing and reinforcement learning. With adopting machine learning in their remediation engine, they see significant virtual machine interruption savings compared to the previous static strategies.

## Future Trends

Auto-remediation research and development is still in very early stages. The existing work mainly focuses on an intermediate step such as constructing a causal graph for a given scenario, or an end-to-end auto-remediation solution for very specific use cases such as virtual machine interruptions. Below are a few topics that can significantly improve the quality of auto-remediation systems.

System Integration Now there is still no unified platform that can perform all the issue analysis, learn the context knowledge, make decisions and execute the actions.

Learn to generate and update knowledge graphs Quality of auto-remediation decision making strongly depends on domain knowledge. Currently humans collect most of the domain knowledge. In the future, it is valuable to explore approaches that learn and maintain knowledge graphs of the systems in a more reliable way.

AI driven decision making and execution Currently most of the decision making and action execution are rule-based or statistical learning based. With more powerful AI techniques, the remediation engine can then consume rich information and make more complex decisions.

## B. Auto-scaling

## Problem Definition

The cloud native technologies are becoming the de facto standard for building scalable applications in public or private clouds, enabling loosely coupled systems that are resilient, manageable, and observable ${ }^{\top}$ The cloud systems such as GCP and AWS provide users on-demand resources including CPU, storage, memory and databases. Users needs to specify a limit of these resources to provision for the workloads of their applications. If a service in an application exceeds the limit of a particular resource, end-users will experience request delays or timeouts, so that system operators will request a larger limit of this resource to avoid degraded performance. But if hundreds of services are running, such large limit results in massive resource wastage. Auto-scaling aims to resolve this issue without human intervention, which enables dynamic provisioning of resources to applications based on workload behavior patterns to minimize resource wastage without loss of quality of service (QoS) to end-users.

Auto-scaling approaches can be categorized into two types: reactive auto-scaling and proactive (or predictive) auto-scaling. Reactive auto-scaling monitors the services in a application, and brings them up and down in reaction to changes in workloads.

Reactive auto-scaling. Reactive auto-scaling is very effective and supported by most cloud platforms. But it has one potential disadvantage, i.e., it won't scale up resources until workloads increase so that there is a short period in which more capacity is not yet available but workloads becomes higher. Therefore, end-users can experience response delays in this short period. Proactive auto-scaling aims to solve this problem by predicting future workloads based on historical data. In this paper, we mainly discuss proactive auto-scaling algorithms based on machine learning.

Proactive Auto-scaling. Typically, proactive auto-scaling involves three steps, i.e., predicting workloads, estimating[^1]capacities and scaling out. Machine learning techniques are usually applied to predict future workloads and estimate the suitable capacities for the monitored services, and then adjustments can be done accordingly to avoid degraded performance.

One type of proactive auto-scaling approaches applies regression models (e.g., ARIMA [210], SARIMA [211], MLP, LSTM [212]). Given the historical metrics of a monitored service, this type of approaches trains a particular regression model to learn the workload behavior patterns. For example, [213] investigated the ARIMA model for workload prediction and showed that the model improves efficiency in resource utilization with minimal impact in QoS. [214] applied a time window MLP to predict phases in containers with different types of workloads and proposed a predictive vertical autoscaling policy to resize containers. [215] also leveraged neural networks (especially MLP) for workload prediction and compared this approach with traditional machine learning models, e.g., linear regression and K-nearest neighbors. [216] applied a bidirectional LSTM to predict the number of HTTP workloads and showed that Bi-LSTM works better than LSTM and ARIMA on the tested use cases. These approaches require accurate forecasting results to avoid over- or under-allocated of resources, while it is hard to develop a robust forecastingbased approach due to the existence of noises and sudden spikes in user requests.

The other type is based on reinforcement learning (RL) that treats auto-scaling as an automatic control problem, whose goal is to learn an optimal auto-scaling policy for the best resource provision action under each observed state. [217] presents an exhaustive survey on reinforcement learning-based auto-scaling approaches, and compares them based on a set of proposed taxonomies. This survey is very worth reading for developers or researchers who are interested in this direction. Although RL looks promising in auto-scaling, there are many issues needed to be resolved. For example, model-based methods require a perfect model of the environment and the learned policies cannot adapt to the changes in the environment, while model-free methods have very poor initial performance and slow convergence so that they will introduce high cost if they are applied in real-world cloud platforms.

## C. Resource Management

## Problem Definition

Resource management is another important topic in cloud computing, which includes resource provisioning, allocation and scheduling, e.g., workload estimation, task scheduling, energy optimization, etc. Even small provisioning inefficiencies, such as selecting the wrong resources for a task, can affect quality of service (QoS) and thus lead to significant monetary costs. Therefore, the goal of resource management is to provision the right amount of resources for tasks to improve QoS, mitigate imbalance workloads, and avoid service level agreements violations.

Because of multiple tenants sharing storage and computation resources on cloud platforms, resource management is a difficult task that involves dynamically allocating resources and scheduling tenants' tasks. How to provision resources can be determined in a reactive manner, e.g., creating static rules manually based on domain knowledge. But similar to autoscaling, reactive approaches result in response delays and excessive overheads. To resolve this issue, ML-based approaches for resource management have gained much attention recently.

## ML-based Resource Management

Many ML-based resource management approaches have been developed in recent years. Due to space limitation, we will not discuss them in details. We recommend readers who are interested in this research topic to read the following nice review papers: [218], [219], [220], [221], [222]. Most of these approaches apply ML techniques to forecast future resource consumption and then do resource provisioning or scheduling based on the forecasting results. For instance, [223] uses random forest and XGBoost to predict VM behaviors including maximum deployment sizes and workloads. [224] proposes a linear regression based approach to predict the resource utilization of the VMs based on their historical data, and then leverage the prediction results to reduce energy consumption. [225] applies gradient boosting models for temperature prediction, based on which a dynamic scheduling algorithm is developed to minimize the peak temperature of hosts. [226] proposes a RL-based workload-specific scheduling algorithm to minimize average task completion time.

The accuracy of the ML model is the key factor that affects the efficiency of a resource management system. Applying more sophisticated traditional ML models or even deep learning models to improve prediction accuracy is a promising research direction. Besides accuracy, the time complexity of model prediction is another important factor needed to be considered. If a ML model is over-complicated, it cannot handle real-time requests of resource allocation and scheduling. How to make a trade-off between accuracy and time complexity needs to be explored further.

## VIII. Future of AIOpS

## A. Common AI Challenges for AIOps

We have discussed the challenges and future trends in each task sections according to how to employ AI techniques. In summary, there are some common challenges across different AIOps tasks.

Data Quality. For all AIOps task there are data quality issues. Most real-world AIOps data are extremely imbalanced due to the nature that incidents only occurs occasionally. Also, most of the real-world AIOps data are very noisy. Significant efforts are needed in data cleaning and pre-processing before it can be used as input to train ML models.

Lack of Labels. It's extremely difficult to acquire quality labels sufficiently. We need a lot of domain experts who are very familiar with system operations to evaluate incidents, root-causes and service graphs, in order to provide high-quality labels. This is extremely time consuming and require specific expertise, which cannot be handled by general crowd sourcing approaches like Mechanical Turk.

Non-stationarity and heterogeneity. Systems are everchanging. AIOps are facing non-stationary problem space. The AI models in this domain need to have mechanisms to deal with this non-stationary nature. Meanwhile, AIOps data are heterogeneous, meaning the same telemetry data can have a variety of underlying behaviors. For example, CPU utilization pattern can be totally different when the resources are used to host different applications. Thus, discovery the hidden states and handle heterogeneity is very important for AIOps solutions to succeed.

Lack of Public Benchmarking. Even though AIOps research communities are growing rapidly, there are still very limited number of public datasets for researchers to benchmark and evaluate their results. Operational data are highly sensitive. Existing research are done either with simulated data or enterprise production data which can hardly be shared with other groups and organizations.

Human-in-the-loop. Human feedback are very important to build AIOps solutions. Currently most of the human feedback are collected in ad-hoc fashion, which is inefficient. There are lack of human-in-the-loop studies in AIOps domain to automate feedback collection and utilize the feedback to improve model performance.

## B. Opportunities and Future Trends

Our literature review of existing AIOps work shows current AIOps research still focuses more on infrastructure and tooling. We see AI technologies being successfully applied in incident detection, RCA applications and some of the solutions has been adopted by large distributed systems like AWS, Alibaba cloud. While it is still in very early stages for AIOps process standardization and full automation. With these evidences, we can foresee the promising topics of AIOps in the next few years.

## High Quality AIOps Infrastructure and Tooling

There are some successful AIOps platforms and tools being developed in recent years. But still there are opportunities where AI can help enhance the efficiency of IT operations. AI is also growing rapidly and new AI technologies are invented and successfully applied in other domains. The digital transformation trend also brings challenges to traditional IT operation and Devops. This creates tremendous needs for high quality AI tooling, including monitoring, detection, RCA, predictions and automations.

## AIOps Standardization

While building the infrastructure and tooling, AIOps experts also better understand the full picture of the entire domain. AIOps modules can be identified and extracted from traditional processes to form its own standard. With clear goals and measures, it becomes possible to standardize AIOps systems, just as what has been done in domains like recommendation systems or NLP. With such standardization, it will be much easier to experiment a large variety of AI techniques to improve AIOps performance.

## Human-centric to Machine-centric AIOps

Human-centric AIOps means human processes still play critical roles in the entire AIOps eco-systems, and AI modules help humans with better decisions and executions. While in Machine-centric mode, AIOps systems require minimum human intervention and can be in human-free state for most of its lifetime. AIOps systems continuously monitor the IT infrastructure, detecting and analysis issues, finding the right paths to drive fixes. In this stage, engineers focus primarily on development tasks rather than operations.

## IX. CONCLUSION

Digital transformation creates tremendous needs for computing resources. The trend boosts strong growth of large scale IT infrastructure, such as cloud computing, edge computing, search engines, etc. Since proposed by Gartner in 2016, AIOps is emerging rapidly and now it draws the attention from large enterprises and organizations. As the scale of IT infrastructure grows to a level where human operation cannot catch up, AIOps becomes the only promising solution to guarantee high availability of these gigantic IT infrastructures. AIOps covers different stages of software lifecycles, including development, testing, deployment and maintenance.

Different AI techniques are now applied in AIOps applications, including anomaly detection, root-cause analysis, failure predictions, automated actions and resource management. However, the entire AIOps industry is still in a very early stage where AI only plays supporting roles to help human conducting operation workflows. We foresee the trend shifting from human-centric Operations to AI-centric Operations in the near future. During the shift, Development of AIOps techniques will also transit from build tools to create humanfree end-to-end solutions.

In this survey, we discovered that most of the current AIOps outcomes focus on detections and root cause analysis, while research work on automations is still very limited. The AI techniques used in AIOps are mainly traditional machine learning and statistical models.

## ACKNOWLEDGMENT

We want to thank all participants who took the time to accomplish this survey. Their knowledge and experiences about AI fundamentals were invaluable to our study. We are also grateful to our colleagues at the Salesforce AI Research Lab and collaborators from other organizations for their helpful feedback and support.

## APPENDIX A TERMINOLOGY

DevOps: Modern software development requires not only higher development quality but also higher operations quality. DevOps, as a set of best practices that combines the development (Dev) and operations (Ops) processes, is created to achieve high quality software development and after release management [3].

Application Performance Monitoring (APM): Application performance monitoring is the practice of tracking key software application performance using monitoring software
and telemetry data 227]. APM is used to guarantee high system availability, optimize service performance and improve user experiences. Originally APM was mostly adopted in websites, mobile apps and other similar online business applications. However, with more and more traditional softwares transforming to leverage cloud based, highly distributed systems, APM is now widely used for a larger variety of software applications and backends.

Observability: Observability is the ability to measure the internal states of a system by examining its outputs [228]. A system is "observable" if the current state can be estimated by only using the information from outputs. Observability data includes metrics, logs, traces and other system generated information.

Cloud Intelligence: The artificial intelligent features that improve cloud applications.

MLOps: MLOps stands for machine learning operations. MLOps is the full process life cycle of deploying machine learning models to production.

Site Reliability Engineering (SRE): The type of engineering that bridge the gap between software development and operations.

Cloud Computing: Cloud computing is a technique, and a business model, that builds highly scalable distributed computer systems and lends computing resources, e.g. hosts, platforms, apps, to tenants to generate revenue. There are three main category of cloud computing: infrastructure as a service (IaaS), platform as a service (PaaS) and software as a service (SaaS)

IT Service Management (ITSM): ITSM refers to all processes and activities to design, create, deliver, and support the IT services to customers.

IT Operations Management (ITOM): ITOM overlaps with ITSM, focusing more on the operation side of IT services and infrastructures.

## APPENDIX B

TABLES

TABLE I

TABLE OF POPULAR PUBLIC DATASETS FOR METRICS OBSERVABILITY

| Name | Description | Tasks |
| :--- | :--- | :--- |
| Azure Public <br> Dataset | These datasets contain a representative subset of first-party <br> Azure virtual machine workloads from a geographical region. | Workload characterization, VM Pre-provisioning, Workload <br> prediction |
| Google Cluster <br> Data | 30 continuous days of information from Google Borg cells. | Workload characterization, Workload prediction |
| Alibaba Cluster <br> Trace | Cluster traces of real production servers from Alibaba Group. | Workload characterization, Workload prediction |
| MIT <br> Supercloud <br> Dataset | Combination of high-level data (e.g. Slurm Workload Manager <br> scheduler data) and low-level job-specific time series data. | Workload characterization |
| Numenta <br> Anomaly <br> Benchmark (re- <br> alAWSCloud- <br> watch) | AWS server metrics as collected by the AmazonCloudwatch <br> service. Example metrics include CPU Utilization, Network <br> Bytes In, and Disk Read Bytes. | Incident detection |
| Yahoo S5 (A1) | A1 benchmark contains real Yahoo! web traffic metrics. | Incident detection |
| Server Machine <br> Dataset | A 5-week-long dataset collected from a large Internet company <br> containing metrics like CPU load, network usage, memory <br> usage, etc. | Incident detection |
| KPI Anomaly <br> Detection <br> Dataset A | A large-scale realworld KPI anomaly detection dataset, covering <br> various KPI patterns and anomaly patterns. This dataset is <br> collected from five large Internet companies (Sougo, eBay, <br> Baidu, Tencent, and Ali). | Incident detection |

TABLE II

TABLE of PopUlar PUBLIC DATASETS FOR LOG OBSERVABILITY

| Dataset | Description | Time-span | Data Size | \# logs | Anomaly <br> Labels | \# Anomalies | \# Log Templates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Distributed system logs |  |  |  |  |  |  |  |
| HDFS | Hadoop distributed file system log | 38.7 hours | $1.47 \mathrm{~GB}$ | $11,175,629$ | $\checkmark$ | 16,838(blocks) | 30 |
|  |  | N.A. | $16.06 \mathrm{~GB}$ | $71,118,073$ | $x$ |  |  |
| Hadoop | Hadoop map-reduce job log | N.A. | $48.61 \mathrm{MB}$ | 394,308 | $\checkmark$ |  | 298 |
| Spark | Spark job log | N.A. | $2.75 \mathrm{~GB}$ | $33,236,604$ | $x$ |  | 456 |
| Zookeeper | ZooKeeper service $\log$ | 26.7 days | $9.95 \mathrm{MB}$ | 74,380 | $x$ |  | 95 |
| OpenStack | OpenStack infrastructure $\log$ | N.A. | $58.61 \mathrm{MB}$ | 207,820 | $\checkmark$ | 503 | 51 |
| Supercomputer logs |  |  |  |  |  |  |  |
| BGL | Blue Gene/L supercomputer $\log$ | 214.7 days | $708.76 \mathrm{MB}$ | $4,747,963$ | $\checkmark$ | 348,460 | 619 |
| HPC | High performance cluster $\log$ | N.A. | $32 \mathrm{MB}$ | 433,489 | $x$ |  | 104 |
| Thunderbird | Thunderbird supercomputer $\log$ | 244 days | 29.6GB | $211,212,192$ | $\checkmark$ | $3,248,239$ | 4040 |
| Operating System logs |  |  |  |  |  |  |  |
| Windows | Windows event $\log$ | 226.7 days | $16.09 \mathrm{~GB}$ | $114,608,388$ | $x$ |  | 4833 |
| Linux | Linux system log | 263.9 days | $2.25 \mathrm{MB}$ | 25,567 | $x$ |  | 488 |
| Mac | Mac OS log | 7 days | 16.09MB | 117,283 | $x$ |  | 2214 |
| Mobile System logs |  |  |  |  |  |  |  |
| Android | Android framework $\log$ | N.A. | $183.37 \mathrm{MB}$ | $1,555,005$ | $x$ |  | 76,923 |
| Health App | Health app log | 10.5days | $22.44 \mathrm{MB}$ | 253,395 | $x$ |  | 220 |
| Server application logs |  |  |  |  |  |  |  |
| Apache | Apache server error logs | 263.9 days | $4.9 \mathrm{MB}$ | 56,481 | $x$ |  | 44 |
| OpenSSH | OpenSSH server logs | 28.4 days | $70.02 \mathrm{MB}$ | 655,146 | $x$ |  | 62 |
| Standalone software logs |  |  |  |  |  |  |  |
| Proxifier | Proxifier software logs | N.A. | $2.42 \mathrm{MB}$ | 21,329 | $x$ |  | 9 |
| Hardware logs |  |  |  |  |  |  |  |
| Switch | Switch hardware failures | 2 years | - | $29,174,680$ | $\checkmark$ | 2,204 | - |

TABLE III

COMPARISON OF EXISTING LOG ANOMALY DETECTION MODELS

| Reference | Learning <br> Setting | Type of Model | Log Representation | Log Tokens | Parsing | Sequence <br> modeling |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 92 | Supervised | Linear Regression, SVM, Deci- <br> sion Tree | handcrafted feature | log template | $\checkmark$ | $\bar{x}$ |
| [84] | Unsupervised | Principal Component Analysis <br> (PCA) | quantitative | log template | $\checkmark$ | $\checkmark$ |
| [67], | Unsupervised | Clustering and Correlation be- <br> tween logs and metrics | sequential, quantitative | log template | $\checkmark$ | $x$ |
| 96 | Unsupervised | Mining invariants using singu- <br> lar value decomposition | quantitative, sequential | log template | $\checkmark$ | $x$ |
| [97], 98$],$ | Unsupervised | Frequent pattern mining from <br> Execution Flow and control <br> flow graph mining | quantitative, sequential | log template | $\checkmark$ | $x$ |
| [20], 100 | Unsupervised | Rule Engine over Ensembles <br> and Heuristic contrast analysis <br> over anomaly characteristics | sequential (with tf-idf weights) | log template | $\checkmark$ | $x$ |
| 101] | Supervised | Autoencoder for log specific <br> word2vec | semantic (trainable embedding) | log template | $\checkmark$ | $\checkmark$ |
| [102] | Unsupervised | Autoencoder w/ Isolation Forest | semantic (trainable embedding) | all tokens | $x$ | $x$ |
| 114 | Supervised | Convolutional Neural Network | semantic (trainable embedding) | log template | $\checkmark$ | $\checkmark$ |
| 108】 | Unsupervised | Attention based LSTM | sequential, quantitative, semantic <br> (GloVe embedding) | $\log$ template, <br> log parameter | $\checkmark$ | $\checkmark$ |
| [81] | Unsupervised | Attention based LSTM | quantitative and semantic (GloVe em- <br> bedding) | log template | $\checkmark$ | $\checkmark$ |
| 111 | Supervised | Attention based LSTM | semantic (fastText embedding with tf- <br> idf weights) | log template | $\checkmark$ | $\checkmark$ |
| [104] | Semi- <br> Supervised | Attention based GRU with clus- <br> tering | semantic (fastText embedding with tf- <br> idf weights) | log template | $\sqrt{\checkmark}$ | $\checkmark$ |
| [112] | Unsupervised | Attention based Bi-LSTM | semantic (with trainable embedding) | all tokens | $x$ | $\checkmark$ |
| 109 | Unsupervised | Bi-LSTM | semantic (token embedding from BERT, <br> GPT, XLM) | all tokens | $x$ | $\checkmark$ |
| 113 | Unsupervised | Attention based Bi-LSTM | semantic (BERT token embedding) | log template | $\checkmark$ | $\checkmark$ |
| 110 | Semi- <br> Supervised | LSTM, trained with supervision <br> from source systems | semantic (GloVe embedding) | log template | $\checkmark$ | $\checkmark$ |
| 18 | Unsupervised | LSTM with domain adversarial <br> training | semantic (GloVe embedding) | all tokens | $x$ | $\bar{\checkmark}$ |
| 118], 18 | Unsupervised | LSTM with Deep Support Vec- <br> tor Data Description | semantic (trainable embedding) | log template | $\checkmark$ | $\checkmark$ |
| 115] | Supervised | Graph Neural Network | semantic (BERT token embedding) | log template | $\checkmark$ | $\checkmark$ |
| 116 | Semi- <br> Supervised | Graph Neural Network | semantic (BERT token embedding) | log template | $\checkmark$ | $\checkmark$ |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_cf7e6f156a8e17729816g-26.jpg?height=68&width=180&top_left_y=1865&top_left_x=184) | Unsupervised | Self-Attention Transformer | semantic (trainable embedding) | all tokens | $x$ | $\checkmark$ |
| I78 | Supervised | Self-Attention Transformer | semantic ( trainable embedding) | all tokens | $x$ | $\bar{\checkmark}$ |
| 117 | Supervised | Hierarchical Transformer | semantic (trainable GloVe embedding) | $\log$ template, <br> log parameter | $\checkmark$ | $\checkmark$ |
| [104], [105] | Unsupervised | BERT Language Model | semantic (BERT token embedding) | all tokens | $x$ | $\checkmark$ |
| 21] | Unsupervised | Unified BERT on various $\log$ <br> analysis tasks | semantic (BERT token embedding) | all tokens | $x$ | $\checkmark$ |
| [232] | Unsupervised | Contrastive Adversarial model | semantic (BERT and VAE based em- <br> bedding) and quantitative | log template | $\checkmark$ | $\checkmark$ |
| $106, \quad 107$ | Unsupervised | LSTM,Transformer based GAN <br> (Generative Adversarial) | semantic (trainable embedding) | log template | $\checkmark$ | $\checkmark$ |

Log Tokens refers to the tokens from the logline used in the log representations

Parsing and Sequence Modeling columns respectively refers to whether these models need log parsing and they support modeling log sequences

TABLE IV

CoMPARISON OF EXISTING METRIC ANOMALY DETECTION MODELS

| Reference | Label Accessibility | Machine Learning Model | Dimensionality | Infrastructure | Streaming Updates |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 31] | Supervised | Tree | Univariate | $x$ | $\checkmark$ (Retraining) |
| 41] | Active | - | Univariate | $\checkmark$ | $\sqrt{ }$ (Retraining) |
| [42] | Unsupervised | Tree | Multivariate | $x$ | $\checkmark$ |
| 43 | Unsupervised | Statistical | Univariate | $x$ | $\checkmark$ |
| \|51| | Unsupervised | Statistical | Univariate | $x$ | $x$ |
| [37] | Semi-supervised | Tree | Univariate | $x$ | $\bar{\checkmark}$ |
| [36] | Unsupervised, Semi- <br> supervised | Deep Learning | Univariate | $x$ | $x$ |
| \|[52| | Unsupervised | Deep Learning | Univariate | $\checkmark$ | $x$ |
| 40] | Domain Adaptation, <br> Active | Tree | Univariate | $x$ | $x$ |
| [46] | Unsupervised | Deep Learning | Multivariate | $x$ | $x$ |
| 49] | Unsupervised | Deep Learning | Univariate | $x$ | $x$ |
| 45 | Unsupervised | Deep Learning | Multivariate | $x$ | $x$ |
| [32] | Supervised | Deep Learning | Univariate | $\checkmark$ | $\boldsymbol{J}$ (Retraining) |
| [47] | Unsupervised | Deep Learning | Multivariate | $x$ | $x$ |
| [48] | Unsupervised | Deep Learning | Multivariate | $x$ | $x$ |
| [50] | Unsupervised | Deep Learning | Multivariate | $x$ | $x$ |
| 38] | Semi-supervised, <br> Active | Deep Learning | Multivariate | $\checkmark$ | $\checkmark$ (Retraining) |

TABLE V

COMParisON OF EXISTING TRaCe and MULTIMODAL ANOMALY DetECTION and RCA ModelS

| Reference | Topic | Deep Learning Adoption | Method |
| :---: | :---: | :---: | :---: |
| [124 | Trace RCA | $x$ | Clustering |
| $\mid 121$ | Trace RCA | $x$ | Heuristic |
| \|234 | Trace RCA | $x$ | Multi-input Differential Sum- <br> marization |
| [197 | Trace RCA | $x$ | Random forest, k-NN |
| [122] | Trace RCA | $\bar{x}$ | Heuristic |
| \|235 | Trace Anomaly Detection | $x$ | Graph model |
| [198 | Multimodal Anomaly Detection | $\bar{\checkmark}$ | Deep Bayesian Networks |
| 236 | Trace Representation | $\checkmark$ | Tree-based RNN |
| [196 | Trace Anomaly Detection | $x$ | Heuristic |
| $\mid 120$ | Multimodal Anomaly Detection | $\checkmark$ | GGNN and SVDD |

TABLE VI

COMPARISON OF SEVERAL EXISTING METRIC RCA APPROACHES

| Reference | Metric or Graph Analysis | Root Cause Score |
| :---: | :---: | :---: |
| [147] | Change points | Chronological order |
| 146 | Change points | Chronological order |
| [148 | Two-sample test | Correlation |
| 149 | Call graphs | Cluster similarity |
| $\mid 150$ | Service graph | PageRank |
| โ151] | Service graph | Graph similarity |
| [152] | Service graph | Hierarchical HMM |
| 153 | PC algorithm | Random walk |
| $\|154\|$ | ITOA-PI | PageRank |
| [155] | Service graph and PC | Causal inference |
| [156 | PC algorithm | Random walk |
| [157 | Service graph and PC | Causal inference |
| [158 | PC algorithm | Random walk |
| 159 | PC algorithm | Random walk |
| \|237| | Service graph | Causal inference |
| [168] | Service graph | Contribution-based |

## REFERENCES

[1] T. Olavsrud, "How to choose your cloud service provider," 2012. [Online]. Available: https://www2.cio.com.au/article/416752/ how_choose_your_cloud_service_provider/

[2] "Summary of the amazon s3 service disruption in the northern virginia (us-east-1) region," 2021. [Online]. Available: https://aws. amazon.com/message/41926/

[3] S. Gunja, "What is devops? unpacking the purpose and importance of an it cultural revolution," 2021. [Online]. Available: https: ///www.dynatrace.com/news/blog/what-is-devops/

[4] Gartner, "Aiops (artificial intelligence for it operations)." [Online]. Available: https://www.gartner.com/en/information-technology/ glossary/aiops-artificial-intelligence-operations

[5] S. Siddique, "The road to enterprise artificial intelligence: A case studies driven exploration," Ph.D. dissertation, 052018.

[6] N. Sabharwal, Hands-on AIOps. Springer, 2022.

[7] Y. Dang, Q. Lin, and P. Huang, "Aiops: Real-world challenges and research innovations," in 2019 IEEE/ACM 41st International Conference on Software Engineering: Companion Proceedings (ICSE-Companion), 2019, pp. 4-5.

[8] L. Rijal, R. Colomo-Palacios, and M. Sánchez-Gordón, "Aiops: A multivocal literature review," Artificial Intelligence for Cloud and Edge Computing, pp. 31-50, 2022.

[9] R. Chalapathy and S. Chawla, "Deep learning for anomaly detection: A survey," arXiv preprint arXiv:1901.03407, 2019.

[10] L. Akoglu, H. Tong, and D. Koutra, "Graph based anomaly detection and description: a survey," Data mining and knowledge discovery, vol. 29, no. 3, pp. 626-688, 2015.

[11] J. Soldani and A. Brogi, "Anomaly detection and failure root cause analysis in (micro)service-based cloud applications: A survey," 2021. [Online]. Available: https://arxiv.org/abs/2105.12378

[12] V. Davidovski, "Exponential innovation through digital transformation," in Proceedings of the 3rd International Conference on Applications in Information Technology, ser. ICAIT'2018. New York, NY, USA: Association for Computing Machinery, 2018, p. 3-5. [Online]. Available: https://doi.org/10.1145/3274856.3274858

[13] D. S. Battina, "Ai and devops in information technology and its future in the united states," INTERNATIONAL JOURNAL OF CREATIVE RESEARCH THOUGHTS (IJCRT), ISSN, pp. 2320-2882, 2021.

[14] A. B. Yoo, M. A. Jette, and M. Grondona, "Slurm: Simple linux utility for resource management," in Workshop on job scheduling strategies for parallel processing. Springer, 2003, pp. 44-60.

[15] J. Zhaoxue, L. Tong, Z. Zhenguo, G. Jingguo, Y. Junling, and L. Liangxiong, "A survey on log research of aiops: Methods and trends," Mob. Netw. Appl., vol. 26, no. 6, p. 2353-2364, dec 2021. [Online]. Available: https://doi.org/10.1007/s11036-021-01832-3

[16] S. He, P. He, Z. Chen, T. Yang, Y. Su, and M. R. Lyu, "A survey on automated $\log$ analysis for reliability engineering," ACM Comput. Surv., vol. 54, no. 6, jul 2021. [Online]. Available: https://doi.org/10.1145/3460345

[17] P. Notaro, J. Cardoso, and M. Gerndt, "A survey of aiops methods for failure management," ACM Trans. Intell. Syst. Technol., vol. 12, no. 6, nov 2021. [Online]. Available: https://doi.org/10.1145/3483424

[18] X. Han and S. Yuan, "Unsupervised cross-system log anomaly detection via domain adaptation," in Proceedings of the 30th ACM International Conference on Information \& Knowledge Management, ser. CIKM '21. New York, NY, USA: Association for Computing Machinery, 2021, p. 3068-3072. [Online]. Available: https://doi.org/ $10.1145 / 3459637.3482209$

[19] V.-H. Le and H. Zhang, "Log-based anomaly detection with deep learning: How far are we?" in Proceedings of the 44th International Conference on Software Engineering, ser. ICSE '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 1356-1367. [Online]. Available: https://doi.org/10.1145/3510003.3510155

[20] N. Zhao, H. Wang, Z. Li, X. Peng, G. Wang, Z. Pan, Y. Wu, Z. Feng, X. Wen, W. Zhang, K. Sui, and D. Pei, "An empirical investigation of practical log anomaly detection for online service systems," in Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2021. New York, NY, USA: Association for Computing Machinery, 2021, p. 1404-1415. [Online]. Available: https://doi.org/10.1145/3468264.3473933

[21] Y. Zhu, W. Meng, Y. Liu, S. Zhang, T. Han, S. Tao, and D. Pei, "Unilog: Deploy one model and specialize it for all $\log$ analysis tasks," CoRR, vol. abs/2112.03159, 2021. [Online]. Available: https://arxiv.org/abs/2112.03159
[22] J. Soldani and A. Brogi, "Anomaly detection and failure root cause analysis in (micro) service-based cloud applications: A survey," ACM Comput. Surv., vol. 55, no. 3, feb 2022. [Online]. Available: https://doi.org/10.1145/3501297

[23] L. Korzeniowski and K. Goczyla, "Landscape of automated log analysis: A systematic literature review and mapping study," IEEE Access, vol. 10, pp. 21 892-21 913, 2022

[24] M. Sheldon and G. V. B. Weissman, "Retrace: Collecting execution trace with virtual machine deterministic replay," in Proceedings of the Third Annual Workshop on Modeling, Benchmarking and Simulation (MoBS 2007). Citeseer, 2007.

[25] R. Fonseca, G. Porter, R. H. Katz, and S. Shenker, "\{X-Trace\}: A pervasive network tracing framework," in 4th USENIX Symposium on Networked Systems Design \& Implementation (NSDI 07), 2007.

[26] J. Zhou, Z. Chen, J. Wang, Z. Zheng, and M. R. Lyu, "Trace bench: An open data set for trace-oriented monitoring," in 2014 IEEE 6th International Conference on Cloud Computing Technology and Science. IEEE, 2014, pp. 519-526.

[27] S. Zhang, C. Zhao, Y. Sui, Y. Su, Y. Sun, Y. Zhang, D. Pei, and Y. Wang, "Robust KPI anomaly detection for large-scale software services with partial labels," in 32nd IEEE International Symposium on Software Reliability Engineering, ISSRE 2021, Wuhan, China, October 25-28, 2021, Z. Jin, X. Li, J. Xiang, L. Mariani, T. Liu, X. Yu, and N. Ivaki, Eds. IEEE, 2021, pp. 103-114. [Online]. Available: https://doi.org/10.1109/ISSRE52982.2021.00023

[28] M. Braei and S. Wagner, "Anomaly detection in univariate time-series: A survey on the state-of-the-art," ArXiv, vol. abs/2004.00433, 2020.

[29] A. Blázquez-García, A. Conde, U. Mori, and J. A. Lozano, "A review on outlier/anomaly detection in time series data," ACM Computing Surveys (CSUR), vol. 54, no. 3, pp. 1-33, 2021.

[30] K. Choi, J. Yi, C. Park, and S. Yoon, "Deep learning for anomaly detection in time-series data: review, analysis, and guidelines," IEEE Access, 2021.

[31] D. Liu, Y. Zhao, H. Xu, Y. Sun, D. Pei, J. Luo, X. Jing, and M. Feng, "Opprentice: Towards practical and automatic anomaly detection through machine learning," in Proceedings of the 2015 internet measurement conference, 2015, pp. 211-224.

[32] J. Gao, X. Song, Q. Wen, P. Wang, L. Sun, and H. Xu, "Robusttad: Robust time series anomaly detection via decomposition and convolutional neural networks," arXiv preprint arXiv:2002.09545, 2020.

[33] S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, "ADBench: Anomaly detection benchmark," in Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. [Online]. Available: https://openreview.net/forum?id=foA_SFQ9zo0

[34] Z. Li, N. Zhao, S. Zhang, Y. Sun, P. Chen, X. Wen, M. Ma, and D. Pei, "Constructing large-scale real-world benchmark datasets for aiops," arXiv preprint arXiv:2208.03938, 2022

[35] R. Wu and E. J. Keogh, "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress," CoRR, vol. abs/2009.13807, 2020. [Online]. Available: https://arxiv. org/abs/2009.13807

[36] H. Xu, W. Chen, N. Zhao, Z. Li, J. Bu, Z. Li, Y. Liu, Y. Zhao, D. Pei, Y. Feng et al., "Unsupervised anomaly detection via variational autoencoder for seasonal kpis in web applications," in Proceedings of the 2018 world wide web conference, 2018, pp. 187-196.

[37] J. Bu, Y. Liu, S. Zhang, W. Meng, Q. Liu, X. Zhu, and D. Pei, "Rapid deployment of anomaly detection models for large number of emerging kpi streams," in 2018 IEEE 37th International Performance Computing and Communications Conference (IPCCC). IEEE, 2018, pp. 1-8.

[38] T. Huang, P. Chen, and R. Li, "A semi-supervised vae based active anomaly detection framework in multivariate time series for online systems," in Proceedings of the ACM Web Conference 2022, 2022, pp. 1797-1806.

[39] X.-L. Li and B. Liu, "Learning from positive and unlabeled examples with different data distributions," in European conference on machine learning. Springer, 2005, pp. 218-229.

[40] X. Zhang, J. Kim, Q. Lin, K. Lim, S. O. Kanaujia, Y. Xu, K. Jamieson, A. Albarghouthi, S. Qin, M. J. Freedman et al., "Cross-dataset time series anomaly detection for cloud systems," in 2019 USENIX Annual Technical Conference (USENIX ATC 19), 2019, pp. 1063-1076.

[41] N. Laptev, S. Amizadeh, and I. Flint, "Generic and scalable framework for automated time-series anomaly detection," in Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, 2015, pp. 1939-1947.

[42] S. Guha, N. Mishra, G. Roy, and O. Schrijvers, "Robust random cut forest based anomaly detection on streams," in International conference on machine learning. PMLR, 2016, pp. 2712-2721.

[43] S. Ahmad, A. Lavin, S. Purdy, and Z. Agha, "Unsupervised real-time anomaly detection for streaming data," Neurocomputing, vol. 262, pp. 134-147, 2017.

[44] Z. Li, Y. Zhao, J. Han, Y. Su, R. Jiao, X. Wen, and D. Pei, "Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding," in $K D D$ '21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, F. Zhu, B. C. Ooi, and C. Miao, Eds. ACM, 2021, pp. 3220-3230. [Online]. Available: https://doi.org/10.1145/3447548.3467075

[45] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "Usad: Unsupervised anomaly detection on multivariate time series," in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, 2020, pp. 3395-3404.

[46] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust anomaly detection for multivariate time series through stochastic recurrent neural network," in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining, 2019, pp. 28282837.

[47] Z. Li, Y. Zhao, J. Han, Y. Su, R. Jiao, X. Wen, and D. Pei, "Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding," in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining, 2021, pp. 3220-3230.

[48] W. Yang, K. Zhang, and S. C. Hoi, "Causality-based multivariate time series anomaly detection," arXiv preprint arXiv:2206.15033, 2022.

[49] F. Ayed, L. Stella, T. Januschowski, and J. Gasthaus, "Anomaly detection at scale: The case for deep distributional time series models," in International Conference on Service-Oriented Computing. Springer, 2020, pp. 97-109.

[50] S. Rabanser, T. Januschowski, K. Rasul, O. Borchert, R. Kurle, J. Gasthaus, M. Bohlke-Schneider, N. Papernot, and V. Flunkert, "Intrinsic anomaly detection for multi-variate time series," arXiv preprint arXiv:2206.14342, 2022.

[51] J. Hochenbaum, O. S. Vallis, and A. Kejariwal, "Automatic anomaly detection in the cloud via statistical learning," arXiv preprint arXiv:1704.07706, 2017

[52] H. Ren, B. Xu, Y. Wang, C. Yi, C. Huang, X. Kou, T. Xing, M. Yang, J. Tong, and Q. Zhang, "Time-series anomaly detection service at microsoft," in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining, 2019, pp. 30093017 .

[53] J. Soldani and A. Brogi, "Anomaly detection and failure root cause analysis in (micro) service-based cloud applications: A survey," ACM Comput. Surv., vol. 55, no. 3, pp. 59:1-59:39, 2023. [Online]. Available: https://doi.org/10.1145/3501297

[54] J. Bu, Y. Liu, S. Zhang, W. Meng, Q. Liu, X. Zhu, and D. Pei, "Rapid deployment of anomaly detection models for large number of emerging KPI streams," in 37th IEEE International Performance Computing and Communications Conference, IPCCC 2018, Orlando, FL, USA, November 17-19, 2018. IEEE, 2018, pp. 1-8. [Online]. Available: https://doi.org/10.1109/PCCC.2018.8711315

[55] Z. Z. Darban, G. I. Webb, S. Pan, C. C. Aggarwal, and M. Salehi, "Deep learning for time series anomaly detection: A survey," CoRR, vol. abs/2211.05244, 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2211.05244

[56] B. Huang, K. Zhang, M. Gong, and C. Glymour, "Causal discovery and forecasting in nonstationary environments with state-space models," in Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 2019, pp. 2901-2910. [Online]. Available: http://proceedings.mlr.press/v97/huang19g.html

[57] Q. Pham, C. Liu, D. Sahoo, and S. C. H. Hoi, "Learning fast and slow for online time series forecasting," CoRR, vol. abs/2202.11672, 2022. [Online]. Available: https://arxiv.org/abs/2202.11672

[58] K. Lai, D. Zha, J. Xu, Y. Zhao, G. Wang, and X. Hu, "Revisiting time series outlier detection: Definitions and benchmarks," in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, J. Vanschoren and S. Yeung, Eds., 2021. [Online]. Available: https://datasets-benchmarks-proceedings.neurips.cc/paper/ 2021/hash/ec5decca5ed3d6b8079e2e7e7bacc9f2-Abstract-round1.html

[59] R. Wu and E. Keogh, "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress," IEEE Transactions on Knowledge and Data Engineering, 2021.
[60] X. Wu, L. Xiao, Y. Sun, J. Zhang, T. Ma, and L. He, "A survey of human-in-the-loop for machine learning," Future Gener. Comput. Syst., vol. 135, pp. 364-381, 2022. [Online]. Available: https://doi.org/10.1016/j.future.2022.05.014

[61] D. Sahoo, Q. Pham, J. Lu, and S. C. H. Hoi, "Online deep learning: Learning deep neural networks on the fly," CoRR, vol. abs/1711.03705, 2017. [Online]. Available: http://arxiv.org/abs/1711.03705

[62] Z. Chen, J. Liu, W. Gu, Y. Su, and M. R. Lyu, "Experience report: Deep learning-based system log analysis for anomaly detection," 2021. [Online]. Available: https://arxiv.org/abs/2107.05908

[63] P. He, J. Zhu, Z. Zheng, and M. R. Lyu, "Drain: An online log parsing approach with fixed depth tree," in 2017 IEEE International Conference on Web Services (ICWS), 2017, pp. 33-40.

[64] A. A. Makanju, A. N. Zincir-Heywood, and E. E. Milios, "Clustering event logs using iterative partitioning," in Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD '09. New York, NY, USA: Association for Computing Machinery, 2009, p. 1255-1264. [Online]. Available: https://doi.org/10.1145/1557019.1557154

[65] Z. M. Jiang, A. E. Hassan, P. Flora, and G. Hamann, "Abstracting execution logs to execution events for enterprise applications (short paper)," in 2008 The Eighth International Conference on Quality Software, 2008, pp. 181-186.

[66] M. Du and F. Li, "Spell: Streaming parsing of system event logs," in 2016 IEEE 16th International Conference on Data Mining (ICDM), 2016, pp. 859-864.

[67] R. Vaarandi and M. Pihelgas, "Logcluster - A data clustering and pattern mining algorithm for event logs," in 11th International Conference on Network and Service Management, CNSM 2015, Barcelona, Spain, November 9-13, 2015, M. Tortonesi, J. Schönwälder, E. R. M. Madeira, C. Schmitt, and J. Serrat, Eds. IEEE Computer Society, 2015, pp. 1-7. [Online]. Available: https: //doi.org/10.1109/CNSM.2015.7367331

[68] Q. Fu, J.-G. Lou, Y. Wang, and J. Li, "Execution anomaly detection in distributed systems through unstructured log analysis," in 2009 Ninth IEEE International Conference on Data Mining, 2009, pp. 149-158.

[69] L. Tang, T. Li, and C.-S. Perng, "Logsig: Generating system events from raw textual logs," in Proceedings of the 20th ACM International Conference on Information and Knowledge Management, ser. CIKM '11. New York, NY, USA: Association for Computing Machinery, 2011, p. 785-794. [Online]. Available: https://doi.org/10.1145/2063576.2063690

[70] M. Mizutani, "Incremental mining of system log format," in 2013 IEEE International Conference on Services Computing, 2013, pp. 595-602.

[71] K. Shima, "Length matters: Clustering system log messages using length of words," CoRR, vol. abs/1611.03213, 2016. [Online]. Available: http://arxiv.org/abs/1611.03213

[72] H. Hamooni, B. Debnath, J. Xu, H. Zhang, G. Jiang, and A. Mueen, "Logmine: Fast pattern recognition for log analytics," in Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, ser. CIKM '16. New York, NY, USA: Association for Computing Machinery, 2016, p. 1573-1582. [Online] Available: https://doi.org/10.1145/2983323.2983358

[73] R. Vaarandi, "A data clustering algorithm for mining patterns from event logs," in Proceedings of the 3rd IEEE Workshop on IP Operations \& Management (IPOM 2003) (IEEE Cat. No.03EX764), 2003, pp. 119 126.

[74] M. Nagappan and M. A. Vouk, "Abstracting log lines to log event types for mining software system logs," in 2010 7th IEEE Working Conference on Mining Software Repositories (MSR 2010), 2010, pp $114-117$.

[75] S. Messaoudi, A. Panichella, D. Bianculli, L. Briand, and R. Sasnauskas, "A search-based approach for accurate identification of $\log$ message formats," in Proceedings of the 26th Conference on Program Comprehension, ser. ICPC '18. New York, NY, USA: Association for Computing Machinery, 2018, p. 167-177. [Online]. Available: https://doi.org/10.1145/3196321.3196340

[76] S. Nedelkoski, J. Bogatinovski, A. Acker, J. Cardoso, and O. Kao, "Self-supervised log parsing," in Machine Learning and Knowledge Discovery in Databases: Applied Data Science Track, Y. Dong, D. Mladenić, and C. Saunders, Eds. Cham: Springer International Publishing, 2021, pp. 122-138.

[77] Y. Liu, X. Zhang, S. He, H. Zhang, L. Li, Y. Kang, Y. Xu, M. Ma, Q. Lin, Y. Dang, S. Rajmohan, and D. Zhang, "Uniparser: A unified log parser for heterogeneous log data," in Proceedings of the ACM Web Conference 2022, ser. WWW '22. New York, NY, USA:

Association for Computing Machinery, 2022, p. 1893-1901. [Online]. Available: https://doi.org/10.1145/3485447.3511993

[78] V.-H. Le and H. Zhang, "Log-based anomaly detection without log parsing," in 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE), 2021, pp. 492-504.

[79] Y. Lee, J. Kim, and P. Kang, "Lanobert : System log anomaly detection based on BERT masked language model," CoRR, vol. abs/2111.09564, 2021. [Online]. Available: https://arxiv.org/abs/2111.09564

[80] M. Farshchi, J.-G. Schneider, I. Weber, and J. Grundy, "Experience report: Anomaly detection of cloud application operations using log and cloud metric correlation analysis," in 2015 IEEE 26th International Symposium on Software Reliability Engineering (ISSRE), 2015, pp. 2434.

[81] W. Meng, Y. Liu, Y. Zhu, S. Zhang, D. Pei, Y. Liu, Y. Chen, R. Zhang, S. Tao, P. Sun, and R. Zhou, "Loganomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs," in Proceedings of the 28th International Joint Conference on Artificial Intelligence, ser. IJCAI'19. AAAI Press, 2019, p. 4739-4745.

[82] Q. Lin, H. Zhang, J.-G. Lou, Y. Zhang, and X. Chen, "Log clustering based problem identification for online service systems," in Proceedings of the 38th International Conference on Software Engineering Companion, ser. ICSE '16. New York, NY, USA: Association for Computing Machinery, 2016, p. 102-111. [Online]. Available: https://doi.org/10.1145/2889160.2889232

[83] R. Yang, D. Qu, Y. Qian, Y. Dai, and S. Zhu, "An online log template extraction method based on hierarchical clustering," EURASIP J. Wirel. Commun. Netw., vol. 2019, p. 135, 2019. [Online]. Available: https://doi.org/10.1186/s13638-019-1430-4

[84] W. Xu, L. Huang, A. Fox, D. A. Patterson, and M. I. Jordan, "Detecting large-scale system problems by mining console logs," in Proceedings of the 22nd ACM Symposium on Operating Systems Principles 2009, SOSP 2009, Big Sky, Montana, USA, October 11-14, 2009, J. N. Matthews and T. E. Anderson, Eds. ACM, 2009, pp. 117-132. [Online]. Available: https://doi.org/10.1145/1629575.1629587

[85] B. Joshi, U. Bista, and M. Ghimire, "Intelligent clustering scheme for log data streams," in Computational Linguistics and Intelligent Text Processing, A. Gelbukh, Ed. Berlin, Heidelberg: Springer Berlin Heidelberg, 2014, pp. 454-465.

[86] J. Liu, J. Zhu, S. He, P. He, Z. Zheng, and M. R. Lyu, "Logzip: Extracting hidden structures via iterative clustering for log compression," in Proceedings of the 34th IEEE/ACM International Conference on Automated Software Engineering, ser. ASE '19. IEEE Press, 2019, p. 863-873. [Online]. Available: https://doi.org/10.1109/ ASE.2019.00085

[87] M. Wurzenberger, F. Skopik, M. Landauer, P. Greitbauer, R. Fiedler, and W. Kastner, "Incremental clustering for semi-supervised anomaly detection applied on log data," in Proceedings of the 12th International Conference on Availability, Reliability and Security, ser. ARES '17. New York, NY, USA: Association for Computing Machinery, 2017. [Online]. Available: https://doi.org/10.1145/3098954.3098973

[88] D. Gunter, B. L. Tierney, A. Brown, M. Swany, J. Bresnahan, and J. M. Schopf, "Log summarization and anomaly detection for troubleshooting distributed systems," in 2007 8th IEEE/ACM International Conference on Grid Computing, 2007, pp. 226-234.

[89] W. Meng, F. Zaiter, Y. Huang, Y. Liu, S. Zhang, Y. Zhang, Y. Zhu, T. Zhang, E. Wang, Z. Ren, F. Wang, S. Tao, and D. Pei, "Summarizing unstructured logs in online services," CoRR, vol. abs/2012.08938, 2020. [Online]. Available: https://arxiv.org/abs/2012.08938

[90] R. Dijkman and A. Wilbik, "Linguistic summarization of event logs - a practical approach," Information Systems, vol. 67, pp. 114-125, 2017. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0306437916303192

[91] S. Locke, H. Li, T.-H. P. Chen, W. Shang, and W. Liu, "Logassist: Assisting log analysis through log summarization," IEEE Transactions on Software Engineering, pp. 1-1, 2021.

[92] P. Bodik, M. Goldszmidt, A. Fox, D. B. Woodard, and H. Andersen, "Fingerprinting the datacenter: Automated classification of performance crises," in Proceedings of the 5th European Conference on Computer Systems, ser. EuroSys '10. New York, NY, USA: Association for Computing Machinery, 2010, p. 111-124. [Online]. Available: https://doi.org/10.1145/1755913.1755926

[93] Y. Liang, Y. Zhang, H. Xiong, and R. Sahoo, "Failure prediction in ibm bluegene/1 event logs," in Seventh IEEE International Conference on Data Mining (ICDM 2007), 2007, pp. 583-588.

[94] M. Chen, A. Zheng, J. Lloyd, M. Jordan, and E. Brewer, "Failure diagnosis using decision trees," in International Conference on Autonomic Computing, 2004. Proceedings., 2004, pp. 36-43.
[95] S. He, Q. Lin, J.-G. Lou, H. Zhang, M. R. Lyu, and D. Zhang, "Identifying impactful service system problems via log analysis," in Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2018. New York, NY, USA: Association for Computing Machinery, 2018, p. 60-70. [Online]. Available: https://doi.org/10.1145/3236024.3236083

[96] J. Lou, Q. Fu, S. Yang, Y. Xu, and J. Li, "Mining invariants from console logs for system problem detection," in 2010 USENIX Annual Technical Conference, Boston, MA, USA, June 23-25, 2010, P. Barham and T. Roscoe, Eds. USENIX Association, 2010. [Online]. Available: https://www.usenix.org/conference/usenix-atc-10/ mining-invariants-console-logs-system-problem-detection

[97] A. Nandi, A. Mandal, S. Atreja, G. B. Dasgupta, and S. Bhattacharya, "Anomaly detection using program control flow graph mining from execution logs," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD '16. New York, NY, USA: Association for Computing Machinery, 2016, p. 215-224. [Online]. Available: https://doi.org/10. $1145 / 2939672.2939712$

[98] T. Jia, P. Chen, L. Yang, Y. Li, F. Meng, and J. Xu, "An approach for anomaly diagnosis based on hybrid graph model with logs for distributed services," in 2017 IEEE International Conference on Web Services (ICWS), 2017, pp. 25-32.

[99] T. Jia, L. Yang, P. Chen, Y. Li, F. Meng, and J. Xu, "Logsed: Anomaly diagnosis through mining time-weighted control flow graph in logs," in 2017 IEEE 10th International Conference on Cloud Computing (CLOUD), 2017, pp. 447-455.

[100] X. Zhang, Y. Xu, S. Qin, S. He, B. Qiao, Z. Li, H. Zhang, X. Li, Y. Dang, Q. Lin, M. Chintalapati, S. Rajmohan, and D. Zhang, "Onion: Identifying incident-indicating logs for cloud systems," in Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2021. New York, NY, USA: Association for Computing Machinery, 2021, p. 1253-1263. [Online]. Available: https://doi.org/10.1145/3468264.3473919

[101] W. Meng, Y. Liu, Y. Huang, S. Zhang, F. Zaiter, B. Chen, and D. Pei, "A semantic-aware representation framework for online $\log$ analysis," in 2020 29th International Conference on Computer Communications and Networks (ICCCN), 2020, pp. 1-7.

[102] A. Farzad and T. A. Gulliver, "Unsupervised log message anomaly detection," ICT Express, vol. 6, no. 3, pp. 229-237, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S2405959520300643

[103] S. Nedelkoski, J. Bogatinovski, A. Acker, J. Cardoso, and O. Kao, "Self-attentive classification-based anomaly detection in unstructured logs," in 2020 IEEE International Conference on Data Mining (ICDM), 2020, pp. 1196-1201.

[104] L. Yang, J. Chen, Z. Wang, W. Wang, J. Jiang, X. Dong, and W. Zhang, "Semi-supervised log-based anomaly detection via probabilistic label estimation," in 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), 2021, pp. 1448-1460.

[105] H. Guo, S. Yuan, and X. Wu, "Logbert: Log anomaly detection via bert," in 2021 International Joint Conference on Neural Networks (IJCNN), 2021, pp. 1-8

[106] B. Xia, Y. Bai, J. Yin, Y. Li, and J. Xu, "Loggan: A log-level generative adversarial network for anomaly detection using permutation event modeling," Information Systems Frontiers, vol. 23, no. 2, p. 285-298, apr 2021. [Online]. Available: https://doi.org/10.1007/s10796-020-10026-3

[107] Z. Zhao, W. Niu, X. Zhang, R. Zhang, Z. Yu, and C. Huang, "Trine: Syslog anomaly detection with three transformer encoders in one generative adversarial network," Applied Intelligence, vol. 52, no. 8, p. 8810-8819, jun 2022. [Online]. Available: https://doi.org/10.1007/ s10489-021-02863-9

[108] M. Du, F. Li, G. Zheng, and V. Srikumar, "Deeplog: Anomaly detection and diagnosis from system logs through deep learning," in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS '17. New York, NY, USA: Association for Computing Machinery, 2017, p. 1285-1298. [Online]. Available: https://doi.org/10.1145/3133956.3134015

[109] H. Ott, J. Bogatinovski, A. Acker, S. Nedelkoski, and O. Kao, "Robust and transferable anomaly detection in log data using pre-trained language models," in 2021 IEEE/ACM International Workshop on Cloud Intelligence (CloudIntelligence). Los Alamitos, CA, USA: IEEE Computer Society, may 2021, pp. 19-

24. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ CloudIntelligence52565.2021.00013

[110] R. Chen, S. Zhang, D. Li, Y. Zhang, F. Guo, W. Meng, D. Pei, Y. Zhang, X. Chen, and Y. Liu, "Logtransfer: Cross-system log anomaly detection for software systems with transfer learning," in 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE), 2020, pp. 37-47.

[111] X. Zhang, Y. Xu, Q. Lin, B. Qiao, H. Zhang, Y. Dang, C. Xie, X. Yang, Q. Cheng, Z. Li, J. Chen, X. He, R. Yao, J.-G. Lou, M. Chintalapati, F. Shen, and D. Zhang, "Robust log-based anomaly detection on unstable log data," in Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2019. New York, NY, USA: Association for Computing Machinery, 2019, p. 807-817. [Online]. Available: https://doi.org/10.1145/3338906.3338931

[112] A. Brown, A. Tuor, B. Hutchinson, and N. Nichols, "Recurrent neural network attention mechanisms for interpretable system log anomaly detection," in Proceedings of the First Workshop on Machine Learning for Computing Systems, ser. MLCS'18. New York, NY, USA: Association for Computing Machinery, 2018. [Online]. Available: https://doi.org/10.1145/3217871.3217872

[113] X. Li, P. Chen, L. Jing, Z. He, and G. Yu, "Swisslog: Robust and unified deep learning based log anomaly detection for diverse faults," in 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE), 2020, pp. 92-103.

[114] S. Lu, X. Wei, Y. Li, and L. Wang, "Detecting anomaly in big data system logs using convolutional neural network," in 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress, DASC/PiCom/DataCom/CyberSciTech 2018, Athens, Greece, August 12-15, 2018. IEEE Computer Society, 2018, pp. 151-158. [Online]. Available: https://doi.org/10.1109/ DASC/PiCom/DataCom/CyberSciTec.2018.00037

[115] Y. Xie, H. Zhang, and M. A. Babar, "Loggd:detecting anomalies from system logs by graph neural networks," 2022. [Online]. Available: https://arxiv.org/abs/2209.07869

[116] Y. Wan, Y. Liu, D. Wang, and Y. Wen, "Glad-paw: Graph-based log anomaly detection by position aware weighted graph attention network," in Advances in Knowledge Discovery and Data Mining, K. Karlapalem, H. Cheng, N. Ramakrishnan, R. K. Agrawal, P. K. Reddy, J. Srivastava, and T. Chakraborty, Eds. Cham: Springer International Publishing, 2021, pp. 66-77.

[117] S. Huang, Y. Liu, C. Fung, R. He, Y. Zhao, H. Yang, and Z. Luan, "Hitanomaly: Hierarchical transformers for anomaly detection in system log," IEEE Transactions on Network and Service Management, vol. 17, no. 4, pp. 2064-2076, 2020.

[118] H. Cheng, D. Xu, S. Yuan, and X. Wu, "Fine-grained anomaly detection in sequential data via counterfactual explanations," 2022. [Online]. Available: https://arxiv.org/abs/2210.04145

[119] S. Nedelkoski, J. Cardoso, and O. Kao, "Anomaly detection from system tracing data using multimodal deep learning," in 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), 2019, pp. $179-186$.

[120] C. Zhang, X. Peng, C. Sha, K. Zhang, Z. Fu, X. Wu, Q. Lin, and D. Zhang, "Deeptralog: Trace-log combined microservice anomaly detection through graph-based deep learning." Pittsburgh, PA, USA: IEEE, 2022, pp. 623-634.

[121] D. C. Arnold, D. H. Ahn, B. R. De Supinski, G. L. Lee, B. P. Miller, and M. Schulz, "Stack trace analysis for large scale debugging," in 2007 IEEE International Parallel and Distributed Processing Symposium. IEEE, 2007, pp. 1-10.

[122] Z. Cai, W. Li, W. Zhu, L. Liu, and B. Yang, "A real-time tracelevel root-cause diagnosis system in alibaba datacenters," IEEE Access, vol. 7, pp. 142 692-142 702, 2019.

[123] P. Papadimitriou, A. Dasdan, and H. Garcia-Molina, "Web graph similarity for anomaly detection," Journal of Internet Services and Applications, vol. 1, no. 1, pp. 19-30, 2010.

[124] M. Y. Chen, E. Kiciman, E. Fratkin, A. Fox, and E. Brewer, "Pinpoint: Problem determination in large, dynamic internet services," in Proceedings International Conference on Dependable Systems and Networks. IEEE, 2002, pp. 595-604.

[125] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[126] F. Salfner, M. Lenk, and M. Malek, "A survey of online failure prediction methods," ACM Comput. Surv., vol. 42, no. 3, pp. 10:1-
10:42, 2010. [Online]. Available: https://doi.org/10.1145/1670679. 1670680

[127] Y. Chen, X. Yang, Q. Lin, H. Zhang, F. Gao, Z. Xu, Y. Dang, D. Zhang, H. Dong, Y. Xu, H. Li, and Y. Kang, "Outage prediction and diagnosis for cloud service systems," in The World Wide Web Conference, WWW 2019, San Francisco, CA, USA, May 13-17, 2019, L. Liu, R. W. White, A. Mantrach, F. Silvestri, J. J. McAuley, R. Baeza-Yates, and L. Zia, Eds. ACM, 2019, pp. 2659-2665. [Online]. Available: https://doi.org/10.1145/3308558.3313501

[128] N. Zhao, J. Chen, Z. Wang, X. Peng, G. Wang, Y. Wu, F. Zhou, Z. Feng, X. Nie, W. Zhang, K. Sui, and D. Pei, "Real-time incident prediction for online service systems," in ESEC/FSE '20: 28th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, Virtual Event, USA, November 8-13, 2020, P. Devanbu, M. B. Cohen, and T. Zimmermann, Eds. ACM, 2020, pp. 315-326. [Online]. Available: https://doi.org/10.1145/3368089.3409672

[129] S. Zhang, Y. Liu, W. Meng, Z. Luo, J. Bu, S. Yang, P. Liang, D. Pei, J. Xu, Y. Zhang, Y. Chen, H. Dong, X. Qu, and L. Song, "Prefix: Switch failure prediction in datacenter networks," Proc. ACM Meas. Anal. Comput. Syst., vol. 2, no. 1, pp. 2:1-2:29, 2018. [Online]. Available: https://doi.org/10.1145/3179405

[130] Q. Lin, K. Hsieh, Y. Dang, H. Zhang, K. Sui, Y. Xu, J. Lou, C. Li, Y. Wu, R. Yao, M. Chintalapati, and D. Zhang, "Predicting node failure in cloud service systems," in Proceedings of the 2018 ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/SIGSOFT FSE 2018, Lake Buena Vista, FL, USA, November 04-09, 2018, G. T. Leavens, A. Garcia, and C. S Pasareanu, Eds. ACM, 2018, pp. 480-490. [Online]. Available: https://doi.org/10.1145/3236024.3236060

[131] E. Pinheiro, W. Weber, and L. A. Barroso, "Failure trends in a large disk drive population," in 5th USENIX Conference on File and Storage Technologies, FAST 2007, February 13-16, 2007, San Jose, CA, USA, A. C. Arpaci-Dusseau and R. H. ArpaciDusseau, Eds. USENIX, 2007, pp. 17-28. [Online]. Available: http://www.usenix.org/events/fast07/tech/pinheiro.html

[132] Y. Xu, K. Sui, R. Yao, H. Zhang, Q. Lin, Y. Dang, P. Li, K. Jiang, W. Zhang, J. Lou, M. Chintalapati, and D. Zhang, "Improving service availability of cloud systems by predicting disk error," in 2018 USENIX Annual Technical Conference, USENIX ATC 2018, Boston, MA, USA, July 11-13, 2018, H. S. Gunawi and B. Reed, Eds. USENIX Association, 2018, pp. 481-494. [Online]. Available: https://www.usenix.org/conference/atc18/presentation/xu-yong

[133] R. K. Sahoo, A. J. Oliner, I. Rish, M. Gupta, J. E. Moreira, S. Ma, R. Vilalta, and A. Sivasubramaniam, "Critical event prediction for proactive management in large-scale computer clusters," in Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD '03. New York, NY, USA: Association for Computing Machinery, 2003, p. 426-435. [Online]. Available: https://doi.org/10.1145/956750.956799

[134] F. Yu, H. Xu, S. Jian, C. Huang, Y. Wang, and Z. Wu, "Dram failure prediction in large-scale data centers," in 2021 IEEE International Conference on Joint Cloud Computing (JCC), 2021, pp. 1-8.

[135] J. Klinkenberg, C. Terboven, S. Lankes, and M. S. Müller, "Data mining-based analysis of hpc center operations," in 2017 IEEE International Conference on Cluster Computing (CLUSTER), 2017, pp. 766-773

[136] S. Zhang, Y. Liu, W. Meng, Z. Luo, J. Bu, S. Yang, P. Liang, D. Pei, J. Xu, Y. Zhang, Y. Chen, H. Dong, X. Qu, and L. Song, "Prefix: Switch failure prediction in datacenter networks," Proc. ACM Meas. Anal. Comput. Syst., vol. 2, no. 1, apr 2018. [Online]. Available: https://doi.org/10.1145/3179405

[137] B. Russo, G. Succi, and W. Pedrycz, "Mining system logs to learn error predictors: a case study of a telemetry system," Empir. Softw. Eng., vol. 20, no. 4, pp. 879-927, 2015. [Online]. Available: https://doi.org/10.1007/s10664-014-9303-2

[138] I. Fronza, A. Sillitti, G. Succi, M. Terho, and J. Vlasenko, "Failure prediction based on $\log$ files using random indexing and support vector machines," J. Syst. Softw., vol. 86, no. 1, p. 2-11, jan 2013. [Online]. Available: https://doi.org/10.1016/j.jss.2012.06.025

[139] F. Salfner and M. Malek, "Using hidden semi-markov models for effective online failure prediction," in 2007 26th IEEE International Symposium on Reliable Distributed Systems (SRDS 2007), 2007, pp. $161-174$

[140] A. Das, F. Mueller, C. Siegel, and A. Vishnu, "Desh: Deep learning for system health prediction of lead times to failure in hpc," in Proceedings
of the 27th International Symposium on High-Performance Parallel and Distributed Computing, ser. HPDC '18. New York, NY, USA: Association for Computing Machinery, 2018, p. 40-51. [Online]. Available: https://doi.org/10.1145/3208040.3208051

[141] J. Gao, H. Wang, and H. Shen, "Task failure prediction in cloud data centers using deep learning," in 2019 IEEE International Conference on Big Data (Big Data), 2019, pp. 1111-1116.

[142] Z. Zheng, Z. Lan, B. H. Park, and A. Geist, "System log pre-processing to improve failure prediction," in 2009 IEEE/IFIP International Conference on Dependable Systems and Networks, 2009, pp. 572-577.

[143] Y. Chen, X. Yang, Q. Lin, H. Zhang, F. Gao, Z. Xu, Y. Dang, D. Zhang, H. Dong, Y. Xu, H. Li, and Y. Kang, "Outage prediction and diagnosis for cloud service systems," in The World Wide Web Conference, ser. WWW '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 2659-2665. [Online]. Available: https://doi.org/10.1145/3308558.3313501

[144] Q. Lin, K. Hsieh, Y. Dang, H. Zhang, K. Sui, Y. Xu, J.-G. Lou, C. Li, Y. Wu, R. Yao, M. Chintalapati, and D. Zhang, "Predicting node failure in cloud service systems," in Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2018. New York, NY, USA: Association for Computing Machinery, 2018, p. 480-490. [Online]. Available: https://doi.org/10.1145/3236024.3236060

[145] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, D. Liu, Q. Xiang, and C. He, "Latent error prediction and fault localization for microservice applications by learning from system trace logs," in Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2019. New York, NY, USA: Association for Computing Machinery, 2019, p. 683-694. [Online]. Available: https://doi.org/10.1145/3338906.3338961

[146] H. Nguyen, Y. Tan, and X. Gu, "Pal: Propagation-aware anomaly localization for cloud hosted distributed applications," ser. SLAML '11. New York, NY, USA: Association for Computing Machinery, 2011. [Online]. Available: https://doi.org/10.1145/2038633.2038634

[147] H. Nguyen, Z. Shen, Y. Tan, and X. Gu, "Fchain: Toward black box online fault localization for cloud systems," in 2013 IEEE 33rd International Conference on Distributed Computing Systems, 2013, pp. 21-30.

[148] H. Shan, Y. Chen, H. Liu, Y. Zhang, X. Xiao, X. He, M. Li, and W. Ding, "e-diagnosis: Unsupervised and real-time diagnosis of smallwindow long-tail latency in large-scale microservice platforms," in The World Wide Web Conference, ser. WWW '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 3215-3222. [Online]. Available: https://doi.org/10.1145/3308558.3313653

[149] J. Thalheim, A. Rodrigues, I. E. Akkus, P. Bhatotia, R. Chen, B. Viswanath, L. Jiao, and C. Fetzer, "Sieve: Actionable insights from monitored metrics in distributed systems," in Proceedings of the 18th ACM/IFIP/USENIX Middleware Conference, ser. Middleware '17. New York, NY, USA: Association for Computing Machinery, 2017, p. 14-27. [Online]. Available: https://doi.org/10.1145/3135974.3135977

[150] L. Wu, J. Tordsson, E. Elmroth, and O. Kao, "Microrca: Root cause localization of performance issues in microservices," in NOMS 2020 - 2020 IEEE/IFIP Network Operations and Management Symposium, 2020, pp. 1-9.

[151] Álvaro Brandón, M. Solé, A. Huélamo, D. Solans, M. S. Pérez, and V. Muntés-Mulero, "Graph-based root cause analysis for service-oriented and microservice architectures," Journal of Systems and Software, vol. 159, p. 110432, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0164121219302067

[152] A. Samir and C. Pahl, "Dla: Detecting and localizing anomalies in containerized microservice architectures using markov models," in 2019 7th International Conference on Future Internet of Things and Cloud (FiCloud), 2019, pp. 205-213.

[153] P. Wang, J. Xu, M. Ma, W. Lin, D. Pan, Y. Wang, and P. Chen, "Cloudranger: Root cause identification for cloud native systems," in 2018 18th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID), 2018, pp. 492-502.

[154] L. Mariani, C. Monni, M. Pezzé, O. Riganelli, and R. Xin, "Localizing faults in cloud systems," in 2018 IEEE 11th International Conference on Software Testing, Verification and Validation (ICST), 2018, pp. 262273.

[155] P. Chen, Y. Qi, and D. Hou, "Causeinfer: Automated end-to-end performance diagnosis with hierarchical causality graph in cloud environment," IEEE Transactions on Services Computing, vol. 12, no. 2, pp. 214-230, 2019.
[156] Y. Meng, S. Zhang, Y. Sun, R. Zhang, Z. Hu, Y. Zhang, C. Jia, Z. Wang, and D. Pei, "Localizing failure root causes in a microservice through causality inference," in 2020 IEEE/ACM 28th International Symposium on Quality of Service (IWQoS), 2020, pp. 1-10.

[157] J. Lin, P. Chen, and Z. Zheng, "Microscope: Pinpoint performance issues with causal graphs in micro-service environments," in ICSOC, 2018

[158] M. Ma, W. Lin, D. Pan, and P. Wang, "Ms-rank: Multi-metric and self-adaptive root cause diagnosis for microservice applications," in 2019 IEEE International Conference on Web Services (ICWS), 2019, pp. 60-67.

[159] M. Ma, J. Xu, Y. Wang, P. Chen, Z. Zhang, and P. Wang, AutoMAP: Diagnose Your Microservice-Based Web Applications Automatically. New York, NY, USA: Association for Computing Machinery, 2020, p. 246-258. [Online]. Available: https://doi.org/10.1145/3366423. 3380111

[160] M. Kim, R. Sumbaly, and S. Shah, "Root cause detection in a service-oriented architecture," in Proceedings of the ACM SIGMETRICS/International Conference on Measurement and Modeling of Computer Systems, ser. SIGMETRICS '13. New York, NY, USA: Association for Computing Machinery, 2013, p. 93-104. [Online]. Available: https://doi.org/10.1145/2465529.2465753

[161] P. Spirtes and C. Glymour, "An algorithm for fast recovery of sparse causal graphs," Social Science Computer Review, vol. 9, no. 1, pp. $62-72,1991$

[162] D. M. Chickering, "Learning equivalence classes of Bayesian-network structures," J. Mach. Learn. Res., vol. 2, no. 3, pp. 445-498, 2002.

[163] , "Optimal structure identification with greedy search," J. Mach. Learn. Res., vol. 3, no. 3, pp. 507-554, 2003.

[164] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, "Detecting and quantifying causal associations in large nonlinear time series datasets," Science Advances, vol. 5, no. 11, p. eaau4996, 2019. [Online]. Available: https://www.science.org/doi/abs/10.1126/ sciadv.aau4996

[165] J. Runge, "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets," in UAI, 2020.

[166] A. Gerhardus and J. Runge, "High-recall causal discovery for autocorrelated time series with latent confounders," in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 12615-12625, [Online]. Available: https://proceedings.neurips.cc/paper/2020/file/ 94e70705efae423efda1088614128d0b-Paper.pdf

[167] J. Qiu, Q. Du, K. Yin, S.-L. Zhang, and C. Qian, "A causality mining and knowledge graph based method of root cause diagnosis for performance anomaly in cloud applications," Applied Sciences, vol. 10, no. 6, 2020. [Online]. Available: https://www.mdpi.com/ 2076-3417/10/6/2166

[168] K. Budhathoki, L. Minorics, P. Bloebaum, and D. Janzing, "Causal structure-based root cause analysis of outliers," in ICML 2022, 2022. [Online]. Available: https://www.amazon.science/publications/ causal-structure-based-root-cause-analysis-of-outliers

[169] S. Lu, B. Rao, X. Wei, B. Tak, L. Wang, and L. Wang, "Log-based abnormal task detection and root cause analysis for spark," in 2017 IEEE International Conference on Web Services (ICWS), 2017, pp. 389-396.

[170] F. Lin, K. Muzumdar, N. P. Laptev, M.-V. Curelea, S. Lee, and S. Sankar, "Fast dimensional analysis for root cause investigation in a large-scale service environment," Proc. ACM Meas. Anal. Comput. Syst., vol. 4, no. 2, jun 2020. [Online]. Available: https://doi.org/10.1145/3392149

[171] L. Wang, N. Zhao, J. Chen, P. Li, W. Zhang, and K. Sui, "Root-cause metric location for microservice systems via log anomaly detection," in 2020 IEEE International Conference on Web Services (ICWS), 2020, pp. $142-150$

[172] C. Luo, J.-G. Lou, Q. Lin, Q. Fu, R. Ding, D. Zhang, and Z. Wang, "Correlating events with time series for incident diagnosis," in Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD '14. New York, NY, USA: Association for Computing Machinery, 2014, p. 1583-1592. [Online]. Available: https://doi.org/10.1145/2623330.2623374

[173] E. Chuah, S.-h. Kuo, P. Hiew, W.-C. Tjhi, G. Lee, J. Hammond, M. T. Michalewicz, T. Hung, and J. C. Browne, "Diagnosing the root-causes of failures from cluster log files," in 2010 International Conference on High Performance Computing, 2010, pp. 1-10

[174] T. S. Zaman, X. Han, and T. Yu, "Scminer: Localizing systemlevel concurrency faults from large system call traces," in 2019 34th

IEEE/ACM International Conference on Automated Software Engineering (ASE), 2019, pp. 515-526

[175] K. Zhang, J. Xu, M. R. Min, G. Jiang, K. Pelechrinis, and H. Zhang, "Automated it system failure prediction: A deep learning approach," in 2016 IEEE International Conference on Big Data (Big Data), 2016, pp. 1291-1300.

[176] Y. Yuan, W. Shi, B. Liang, and B. Qin, "An approach to cloud execution failure diagnosis based on exception logs in openstack," in 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), 2019, pp. 124-131.

[177] H. Mi, H. Wang, Y. Zhou, M. R.-T. Lyu, and H. Cai, "Toward finegrained, unsupervised, scalable performance diagnosis for production cloud computing systems," IEEE Transactions on Parallel and Distributed Systems, vol. 24, no. 6, pp. 1245-1255, 2013.

[178] H. Jiang, X. Li, Z. Yang, and J. Xuan, "What causes my test alarm? automatic cause analysis for test alarms in system and integration testing," in Proceedings of the 39th International Conference on Software Engineering, ser. ICSE '17. IEEE Press, 2017, p. 712-723. [Online]. Available: https://doi.org/10.1109/ICSE.2017.71

[179] A. Amar and P. C. Rigby, "Mining historical test logs to predict bugs and localize faults in the test logs," in 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), 2019, pp. $140-151$

[180] F. Wang, A. Bundy, X. Li, R. Zhu, K. Nuamah, L. Xu, S. Mauceri, and J. Z. Pan, "Lekg: A system for constructing knowledge graphs from $\log$ extraction," in The 10th International Joint Conference on Knowledge Graphs, ser. IJCKG'21. New York, NY, USA: Association for Computing Machinery, 2021, p. 181-185. [Online]. Available: https://doi.org/10.1145/3502223.3502250

[181] A. Ekelhart, F. J. Ekaputra, and E. Kiesling, "The slogert framework for automated log knowledge graph construction," in The Semantic Web, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova, O. Corcho, P. Ristoski, and M. Alam, Eds. Cham: Springer International Publishing, 2021, pp. 631-646.

[182] C. Bansal, S. Renganathan, A. Asudani, O. Midy, and M. Janakiraman, "Decaf: Diagnosing and triaging performance issues in large-scale cloud services," in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering: Software Engineering in Practice, ser. ICSE-SEIP '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 201-210. [Online]. Available: https://doi.org/10.1145/3377813.3381353

[183] B. C. Tak, S. Tao, L. Yang, C. Zhu, and Y. Ruan, "Logan: Problem diagnosis in the cloud using log-based reference models," in 2016 IEEE International Conference on Cloud Engineering (IC2E), 2016, pp. 6267.

[184] W. Shang, Z. M. Jiang, H. Hemmati, B. Adams, A. E. Hassan, and P. Martin, "Assisting developers of big data analytics applications when deploying on hadoop clouds," in 2013 35th International Conference on Software Engineering (ICSE), 2013, pp. 402-411.

[185] C. Pham, L. Wang, B. C. Tak, S. Baset, C. Tang, Z. Kalbarczyk, and R. K. Iyer, "Failure diagnosis for distributed systems using targeted fault injection," IEEE Transactions on Parallel and Distributed Systems, vol. 28, no. 2, pp. 503-516, 2017.

[186] K. Nagaraj, C. Killian, and J. Neville, "Structured comparative analysis of systems logs to diagnose performance problems," in Proceedings of the 9th USENIX Conference on Networked Systems Design and Implementation, ser. NSDI'12. USA: USENIX Association, 2012, p. 26.

[187] H. Ikeuchi, A. Watanabe, T. Kawata, and R. Kawahara, "Root-cause diagnosis using logs generated by user actions," in 2018 IEEE Global Communications Conference (GLOBECOM), 2018, pp. 1-7.

[188] P. Aggarwal, A. Gupta, P. Mohapatra, S. Nagar, A. Mandal, Q. Wang, and A. Paradkar, "Localization of operational faults in cloud applications by mining causal dependencies in logs using golden signals," in Service-Oriented Computing - ICSOC 2020 Workshops, H. Hacid, F. Outay, H.-y. Paik, A. Alloum, M. Petrocchi, M. R. Bouadjenek, A. Beheshti, X. Liu, and A. Maaradji, Eds. Cham: Springer International Publishing, 2021, pp. 137-149.

[189] Y. Zhang, Z. Guan, H. Qian, L. Xu, H. Liu, Q. Wen, L. Sun, J. Jiang, L. Fan, and M. Ke, "Cloudrca: A root cause analysis framework for cloud computing platforms," in Proceedings of the 30th ACM International Conference on Information \& Knowledge Management, ser. CIKM '21. New York, NY, USA: Association for Computing Machinery, 2021, p. 4373-4382. [Online]. Available: https://doi.org/10.1145/3459637.3481903

[190] H. Wang, Z. Wu, H. Jiang, Y. Huang, J. Wang, S. Kopru, and T. Xie, "Groot: An event-graph-based approach for root cause analysis in industrial settings," in Proceedings of the 36th IEEE/ACM International Conference on Automated Software Engineering, ser. ASE '21. IEEE Press, 2021, p. 419-429. [Online]. Available: https://doi.org/10.1109/ASE51524.2021.9678708

[191] X. Fu, R. Ren, S. A. McKee, J. Zhan, and N. Sun, "Digging deeper into cluster system logs for failure prediction and root cause diagnosis," in 2014 IEEE International Conference on Cluster Computing (CLUSTER), 2014, pp. 103-112.

[192] S. Kobayashi, K. Fukuda, and H. Esaki, "Mining causes of network events in log data with causal inference," in 2017 IFIP/IEEE Symposium on Integrated Network and Service Management (IM), 2017, pp. $45-53$.

[193] S. Kobayashi, K. Otomo, and K. Fukuda, "Causal analysis of network logs with layered protocols and topology knowledge," in 2019 15th International Conference on Network and Service Management (CNSM), 2019, pp. 1-9.

[194] R. Jarry, S. Kobayashi, and K. Fukuda, "A quantitative causal analysis for network log data," in 2021 IEEE 45th Annual Computers, Software, and Applications Conference (COMPSAC), 2021, pp. 1437-1442.

[195] D. Liu, C. He, X. Peng, F. Lin, C. Zhang, S. Gong, Z. Li, J. Ou, and Z. Wu, "Microhecl: High-efficient root cause localization in largescale microservice systems," in Proceedings of the 43rd International Conference on Software Engineering: Software Engineering in Practice, ser. ICSE-SEIP '21. IEEE Press, 2021, p. 338-347. [Online]. Available: https://doi.org/10.1109/ICSE-SEIP52600.2021.00043

[196] Z. Li, J. Chen, R. Jiao, N. Zhao, Z. Wang, S. Zhang, Y. Wu, L. Jiang, L. Yan, Z. Wang et al., "Practical root cause localization for microservice systems via trace analysis," in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS). IEEE, 2021, pp. $1-10$.

[197] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, D. Liu, Q. Xiang, and C. He, "Latent error prediction and fault localization for microservice applications by learning from system trace logs," in Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2019, pp. 683-694.

[198] P. Liu, H. Xu, Q. Ouyang, R. Jiao, Z. Chen, S. Zhang, J. Yang, L. Mo, J. Zeng, W. Xue et al., "Unsupervised detection of microservice trace anomalies through service-level deep bayesian networks," in 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2020, pp. 48-58.

[199] B. H. Sigelman, L. A. Barroso, M. Burrows, P. Stephenson, M. Plakal, D. Beaver, S. Jaspan, and C. Shanbhag, "Dapper, a large-scale distributed systems tracing infrastructure," 2010.

[200] J. Jiang, W. Lu, J. Chen, Q. Lin, P. Zhao, Y. Kang, H. Zhang, Y. Xiong, F. Gao, Z. Xu, Y. Dang, and D. Zhang, "How to mitigate the incident? an effective troubleshooting guide recommendation technique for online service systems," in Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ser. ESEC/FSE 2020. New York, NY, USA: Association for Computing Machinery, 2020, p. 1410-1420. [Online]. Available: https://doi.org/10.1145/3368089.3417054

[201] M. Shetty, C. Bansal, S. P. Upadhyayula, A. Radhakrishna, and A. Gupta, "Autotsg: Learning and synthesis for incident troubleshooting," CoRR, vol. abs/2205.13457, 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2205.13457

[202] X. Nie, Y. Zhao, K. Sui, D. Pei, Y. Chen, and X. Qu, "Mining causality graph for automatic web-based service diagnosis," in 2016 IEEE 35th International Performance Computing and Communications Conference (IPCCC), 2016, pp. 1-8.

[203] W. Lin, M. Ma, D. Pan, and P. Wang, "Facgraph: Frequent anomaly correlation graph mining for root cause diagnose in micro-service architecture," in 2018 IEEE 37th International Performance Computing and Communications Conference (IPCCC), 2018, pp. 1-8.

[204] R. Ding, Q. Fu, J. G. Lou, Q. Lin, D. Zhang, and T. Xie, "Mining historical issue repositories to heal large-scale online service systems," in 2014 44th Annual IEEE/IFIP International Conference on Dependable Systems and Networks, 2014, pp. 311-322.

[205] M. Shetty, C. Bansal, S. Kumar, N. Rao, N. Nagappan, and T. Zimmermann, "Neural knowledge extraction from cloud service incidents," in Proceedings of the 43rd International Conference on Software Engineering: Software Engineering in Practice, ser. ICSE-SEIP '21. IEEE Press, 2021, p. 218-227. [Online]. Available: https://doi.org/10.1109/ICSE-SEIP52600.2021.00031

[206] M. Shetty, C. Bansal, S. Kumar, N. Rao, and N. Nagappan, "Softner: Mining knowledge graphs from cloud incidents," Empir.

Softw. Eng., vol. 27, no. 4, p. 93, 2022. [Online]. Available: https://doi.org/10.1007/s10664-022-10159-w

[207] A. Saha and S. C. H. Hoi, "Mining root cause knowledge from cloud service incident investigations for aiops," in 44th IEEE/ACM International Conference on Software Engineering: Software Engineering in Practice, ICSE (SEIP) 2022, Pittsburgh, PA, USA, May 22-24, 2022. IEEE, 2022, pp. 197-206. [Online]. Available: https://doi.org/10.1109/ICSE-SEIP55303.2022.9793994

[208] S. Becker, F. Schmidt, A. Gulenko, A. Acker, and O. Kao, "Towards aiops in edge computing environments," in 2020 IEEE International Conference on Big Data (Big Data), 2020, pp. 3470-3475.

[209] S. Levy, R. Yao, Y. Wu, Y. Dang, P. Huang, Z. Mu, P. Zhao, T. Ramani, N. Govindaraju, X. Li, Q. Lin, G. L. Shafriri, and M. Chintalapati, "Predictive and adaptive failure mitigation to avert production cloud VM interruptions," in 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). USENIX Association, Nov. 2020, pp. 1155-1170. [Online]. Available: https://www.usenix.org/conference/osdi20/presentation/levy

[210] J. D. Hamilton, Time Series Analysis. Princeton University Press, 1994.

[211] R. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 2nd ed. OTexts, 2018.

[212] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[213] R. N. Calheiros, E. Masoumi, R. Ranjan, and R. Buyya, "Workload prediction using arima model and its impact on cloud applications' qos," IEEE Transactions on Cloud Computing, vol. 3, no. 4, pp. 449458, 2015.

[214] D. Buchaca, J. L. Berral, C. Wang, and A. Youssef, "Proactive container auto-scaling for cloud native machine learning services," in 2020 IEEE 13th International Conference on Cloud Computing (CLOUD), 2020, pp. 475-479.

[215] M. Wajahat, A. Gandhi, A. Karve, and A. Kochut, "Using machine learning for black-box autoscaling," in 2016 Seventh International Green and Sustainable Computing Conference (IGSC), 2016, pp. 18 .

[216] N.-M. Dang-Quang and M. Yoo, "Deep learning-based autoscaling using bidirectional long short-term memory for kubernetes," Applied Sciences, vol. 11, no. 9, 2021. [Online]. Available: https://www.mdpi. com/2076-3417/11/9/3835

[217] Y. Garí, D. A. Monge, E. Pacini, C. Mateos, and C. García Garino, "Reinforcement learning-based application autoscaling in the cloud: A survey," Engineering Applications of Artificial Intelligence, vol. 102, p. 104288, 2021. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0952197621001354

[218] S. Mustafa, B. Nazir, A. Hayat, A. ur Rehman Khan, and S. A. Madani, "Resource management in cloud computing: Taxonomy, prospects, and challenges," Computers and Electrical Engineering, vol. 47, pp. 186-203, 2015. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S004579061500275X

[219] T. Khan, W. Tian, G. Zhou, S. Ilager, M. Gong, and R. Buyya, "Machine learning (ml)-centric resource management in cloud computing: A review and future directions," Journal of Network and Computer Applications, vol. 204, p. 103405, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1084804522000649

[220] F. Nzanywayingoma and Y. Yang, "Efficient resource management techniques in cloud computing environment: a review and discussion," International Journal of Computers and Applications, vol. 41, no. 3, pp. 165-182, 2019. [Online]. Available: https://doi.org/10.1080/ 1206212X.2017.1416558

[221] N. M. Gonzalez, T. C. M. D. B. Carvalho, and C. C. Miers, "Cloud resource management: Towards efficient execution of large-scale scientific applications and workflows on complex infrastructures," J. Cloud Comput., vol. 6, no. 1, dec 2017. [Online]. Available: https://doi.org/10.1186/s13677-017-0081-4

[222] R. Bianchini, M. Fontoura, E. Cortez, A. Bonde, A. Muzio, A.-M. Constantin, T. Moscibroda, G. Magalhaes, G. Bablani, and M. Russinovich, "Toward ml-centric cloud platforms," Commun. ACM, vol. 63, no. 2, p. 50-59, jan 2020. [Online]. Available: https://doi.org/10.1145/3364684

[223] E. Cortez, A. Bonde, A. Muzio, M. Russinovich, M. Fontoura, and R. Bianchini, "Resource central: Understanding and predicting workloads for improved resource management in large cloud platforms," in Proceedings of the 26th Symposium on Operating Systems Principles, ser. SOSP '17. New York, NY, USA: Association for Computing Machinery, 2017, p. 153-167. [Online]. Available: https://doi.org/10.1145/3132747.3132772
[224] K. Haghshenas and S. Mohammadi, "Prediction-based underutilized and destination host selection approaches for energy-efficient dynamic vm consolidation in data centers," The Journal of Supercomputing, vol. 76, no. 12, pp. 10240-10257, Dec 2020. [Online]. Available: https://doi.org/10.1007/s11227-020-03248-4

[225] S. Ilager, K. Ramamohanarao, and R. Buyya, "Thermal prediction for efficient energy management of clouds using machine learning," IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 5, pp. 1044-1056, 2021.

[226] H. Mao, M. Schwarzkopf, S. B. Venkatakrishnan, Z. Meng, and M. Alizadeh, "Learning scheduling algorithms for data processing clusters," in Proceedings of the ACM Special Interest Group on Data Communication, ser. SIGCOMM '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 270-288. [Online]. Available: https://doi.org/10.1145/3341302.3342080

[227] D. Anderson, "What is apm?" 2021. [Online]. Available: https: //www.dynatrace.com/news/blog/what-is-apm-2/

[228] J. Livens, "What is observability? not just logs, metrics and traces," 2021. [Online]. Available: https://www.dynatrace.com/news/ blog/what-is-observability-2/

[229] Y. Guo, Y. Wen, C. Jiang, Y. Lian, and Y. Wan, "Detecting $\log$ anomalies with multi-head attention (LAMA)," CoRR, vol. abs/2101.02392, 2021. [Online]. Available: https://arxiv.org/abs/2101. 02392

[230] S. Zhang, Y. Liu, X. Zhang, W. Cheng, H. Chen, and H. Xiong, "Cat: Beyond efficient transformer for content-aware anomaly detection in event sequences," in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 4541-4550. [Online]. Available: https://doi.org/10.1145/3534678.3539155

[231] C. Zhang, X. Wang, H. Zhang, H. Zhang, and P. Han, "Log sequence anomaly detection based on local information extraction and globally sparse transformer model," IEEE Transactions on Network and Service Management, vol. 18, no. 4, pp. 4119-4133, 2021.

[232] Q. Wang, X. Zhang, X. Wang, and Z. Cao, "Log Sequence Anomaly Detection Method Based on Contrastive Adversarial Training and Dual Feature Extraction," Entropy, vol. 24, no. 1, p. 69, Dec. 2021.

[233] J. Qi, Z. Luan, S. Huang, Y. Wang, C. J. Fung, H. Yang, and D. Qian, "Adanomaly: Adaptive anomaly detection for system logs with adversarial learning," in 2022 IEEE/IFIP Network Operations and Management Symposium, NOMS 2022, Budapest, Hungary, April 25-29, 2022. IEEE, 2022, pp. 1-5. [Online]. Available: https://doi.org/10.1109/NOMS54207.2022.9789917

[234] M. Attariyan, M. Chow, and J. Flinn, "X-ray: Automating \{RootCause \} diagnosis of performance anomalies in production software," in 10th USENIX Symposium on Operating Systems Design and Implementation (OSDI 12), 2012, pp. 307-320.

[235] X. Guo, X. Peng, H. Wang, W. Li, H. Jiang, D. Ding, T. Xie, and L. Su, "Graph-based trace analysis for microservice architecture understanding and problem diagnosis," in Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2020, pp. 1387-1397.

[236] Y. Xu, Y. Zhu, B. Qiao, H. Che, P. Zhao, X. Zhang, Z. Li, Y. Dang, and Q. Lin, "Tracelingo: Trace representation and learning for performance issue diagnosis in cloud services," in 2021 IEEE/ACM International Workshop on Cloud Intelligence (CloudIntelligence). IEEE, 2021, pp. $37-40$.

[237] M. Li, Z. Li, K. Yin, X. Nie, W. Zhang, K. Sui, and D. Pei, "Causal inference-based root cause analysis for online service systems with intervention recognition," in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 3230-3240. [Online]. Available: https://doi.org/10.1145/3534678.3539041


[^0]:    * Equal Contribution

    $\dagger$ Work done when author was with Salesforce AI

[^1]:    ${ }^{\dagger}$ https://github.com/cncf/foundation/blob/main/charter.md

</end of paper 2>


