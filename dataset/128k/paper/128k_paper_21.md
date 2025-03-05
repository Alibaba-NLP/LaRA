<paper 0>
# Extreme Compression of Large Language Models via Additive Quantization 

Vage Egiazarian ${ }^{* 12}$ Andrei Panferov ${ }^{* 12}$ Denis Kuznedelev ${ }^{23}$ Elias Frantar $^{4}$ Artem Babenko ${ }^{2}$ Dan Alistarh ${ }^{45}$


#### Abstract

The emergence of accurate open large language models (LLMs) has led to a race towards performant quantization techniques which can enable their execution on end-user devices. In this paper, we revisit the problem of "extreme" LLM compression-defined as targeting extremely low bit counts, such as 2 to 3 bits per parameter-from the point of view of classic methods in MultiCodebook Quantization (MCQ). Our algorithm, called AQLM, generalizes the classic Additive Quantization (AQ) approach for information retrieval to advance the state-of-the-art in LLM compression, via two innovations: 1) learned additive quantization of weight matrices in input-adaptive fashion, and 2) joint optimization of codebook parameters across entire layer blocks. Broadly, AQLM is the first scheme that is Pareto optimal in terms of accuracy-vs-model-size when compressing to less than 3 bits per parameter, and significantly improves upon all known schemes in the extreme compression (2bit) regime. In addition, AQLM is practical: we provide fast GPU and CPU implementations of AQLM for token generation, which enable us to match or outperform optimized FP16 implementations for speed, while executing in a much smaller memory footprint.


## 1. Introduction

The rapid advancement of generative large language models (LLMs) has led to massive industrial and popular interest, driven in part by the availability of accurate open LLMs, such as Llama 1 and 2 (Touvron et al., 2023), Falcon (TII UAE, 2023), BLOOM (Scao et al., 2022), OPT (Zhang et al., 2022), or NeoX/Pythia (Biderman et al., 2023). A key advantage of open models is that they can be inferenced or fine-tuned locally by end-users, assuming that their computational and memory costs can be reduced to be manageable on commodity hardware. This has led to several methods for[^0]

Preliminary work. To be extended with additional experiments.

![](https://cdn.mathpix.com/cropped/2024_06_04_46c218fad80e5caf3b83g-01.jpg?height=458&width=656&top_left_y=606&top_left_x=1144)

Figure 1: Comparison of AQLM (2-bit) relative to the stateof-the-art QuIP\# (2-bit) and the original 16-bit weights on LLAMA 2 7, 13, and 70B models.

inference and fine-tuning on compressed LLMs (Dettmers et al., 2022; Frantar et al., 2022a; Dettmers \& Zettlemoyer, 2022; Lin et al., 2023; Dettmers et al., 2023a). Currently, the primary approach for accurate post-training compression of LLMs is quantization, which reduces the bit-width at which model weights (and possibly activations) are stored, leading to improvements in model footprint and memory transfer.

By and large, LLM weights are compressed via "direct" quantization, in the sense that a suitable quantization grid and normalization are first chosen for each matrix subcomponent, and then weights are each mapped onto the grid either by direct rounding, e.g. (Dettmers \& Zettlemoyer, 2022), or via more complex allocations, e.g. (Frantar et al., 2022a). Quantization induces a natural compression-vsaccuracy trade-off, usually measured in terms of model size vs model perplexity (PPL). Existing approaches can achieve arguably low accuracy loss at 3-4 bits per element (Dettmers et al., 2023b; Chee et al., 2023; Kim et al., 2023), and can even stably compress models to 2 or even less bits per element, in particular, for extremely large models (Frantar \& Alistarh, 2023). Yet, in most cases, low bit counts come at the cost of significant drops in accuracy, higher implementation complexity and runtime overheads. Specifically, from the practical perspective, "extreme" quantization in the 2-bit range using current techniques is inferior to simply using a smaller base model and quantizing it to higher bitwidths, such as 3-4 bits per parameter, as the latter yields higher accuracy given the same model size in bytes (Dettmers \& Zettlemoyer, 2022; Chee et al., 2023).

Contribution. In this work, we improve the state-of-the-art in LLM compression by showing for the first time that MultiCodebook Quantization (MCQ) techniques can be extended to LLM weight compression. Broadly, MCQ is a family of information retrieval methods (Chen et al., 2010; Jegou et al., 2010; Ge et al., 2013; Zhang et al., 2014; Babenko \& Lempitsky, 2014; Martinez et al., 2016; 2018), consisting of specialized quantization algorithms to compress databases of vectors, allowing for efficient search. Unlike direct quantization, MCQ compresses multiple values jointly, by leveraging the mutual information of quantized values.

More precisely, we extend Additive Quantization (AQ) (Babenko \& Lempitsky, 2014; Martinez et al., 2016), a popular MCQ algorithm, to the task of compressing LLM weights such that the output of each layer and Transformer block are approximately preserved. Our extension reformulates the classic AQ optimization problem to reduce the error in LLM layer outputs under the input token distribution and as well as to jointly optimize codes over layer blocks, rather than only preserving the weights themselves as in standard AQ. We refer to the resulting procedure as Additive Quantization of Language Models (AQLM). Unlike some extreme LLM quantization approaches that require hybrid sparse-quantized formats which separate outlier quantization (Kim et al., 2023; Dettmers et al., 2023b), AQLM quantizes models in a simple homogeneous format, which is easy to support in practice. Our main contributions are as follows:

1. We propose the AQLM algorithm, which extends AQ to post-training compression of LLM weights, via two innovations: (1) adapting the MAP-MRF optimization problem behind AQ to be instance-aware, taking layer calibration input \& output activations into account; (2) complementing the layer-wise optimization with an efficient intra-layer tuning technique, which optimizes quantization parameters jointly over several layers, using only the calibration data.
2. We evaluate the effectiveness of this algorithm on the task of compressing accurate open LLMs from the LlamA 2 (Touvron et al., 2023) family with compression rates of 2-4 bits per parameter. We find that AQLM outperforms the previous state-of-the-art across the standard 2-4 bit compression range, with the most significant improvements for extreme 2-bit quantization (see Figure 1). We provide detailed ablations for the impact of various algorithm parameters, such as code width and number of codebooks, and extend our analysis to the recent Mixtral model (Jiang et al., 2024).
3. We show that $\mathrm{AQLM}$ is practical, by providing efficient GPU and CPU kernels implementations for specific encodings, as well as end-to-end generation ${ }^{1}$. Results[^1]

show show that our approach can match or even outperform the floating point baseline in terms of speed, while reducing the memory footprint by up to $8 x$. Specifically, AQLM can be executed with layer-wise speedups of $\sim 30 \%$ for GPUs, and of up to $4 x$ for CPU inference.

## 2. Background \& Related Work

### 2.1. LLM Quantization

Early efforts towards post-training quantization (PTQ) methods (Nagel et al., 2020; Gholami et al., 2021) that scale to LLMs such as ZeroQuant (Yao et al., 2022), LLM.int8() (Dettmers et al., 2022), and nuQmm (Park et al., 2022) employed direct round-to-nearest (RTN) projections, and adjusted quantization granularity to balance memory efficiency and accuracy. GPTQ (Frantar et al., 2022a) proposed a more accurate data-aware approach via an approximate large-scale solver for minimizing layer-wise $\ell_{2}$ errors.

Dettmers \& Zettlemoyer (2022) examined the accuracycompression trade-offs of these early methods, suggesting that 4-bit quantization may be optimal for RTN quantization, and observing that data-aware methods like GPTQ allow for higher compression, i.e. strictly below 4 bits/weight, maintaining Pareto optimality. Our work brings this Pareto frontier below 3 bits/weight, for the first time. Parallel work quantizing both weights and activations to 8 -bits, by Dettmers et al. (2022), Xiao et al. (2022), and Yao et al. (2022) noted that the "outlier features" in large LLMs cause substantial errors, prompting various mitigation strategies.

Recently, several improved techniques have focused on the difficulty of quantizing weight outliers, which have high impact on the output error. SpQR (Dettmers et al., 2023b) addresses this by saving outliers as a highly-sparse higherprecision matrix. AWQ (Lin et al., 2023) reduces the error of quantizing channels with the highest activation magnitudes by employing per-channel scaling to reduce the error on important weights. SqueezeLLM (Kim et al., 2023) uses the diagonal Fisher as a proxy for the Hessian and implements non-uniform quantization through K-means clustering.

The state-of-the-art method in terms of accuracy-to-size trade-off is QuIP (Chee et al., 2023). Concurrent to our work, an improved variant called QuIP\# (Tseng et al., 2023) was introduced. Roughly, these methods work by first "smoothening" weights by multiplying with a rotation matrix, and then mapping them onto a lattice. QuIP was the first method to obtain stable results (i.e., single-digit PPL increases) in the 2-bit per parameter compression range. At a high level, QuIP and QuIP\# aim to minimize the "worstcase" error for each layer, given initial weights and calibration data. For instance, in QuIP\#, the distribution of the rotated weights approximates a Gaussian, while the encoding lattice (E8P) is chosen to minimize "rounding" error.

By contrast, our approach uses a different weight encoding (codebooks are additive), and learned codebooks instead of a fixed codebook. Thus, our insight is that we should be able to obtain higher accuracy by direct optimization of the codebooks over the calibration set, removing the rotation. Further, we show that codebooks for different layers can co-train via joint fine-tuning over the calibration data.

### 2.2. Quantization for Nearest Neighbor Search

Our work builds on approximate nearest neighbor search (ANN) algorithms. Unlike PTQ, ANN quantization aims to compress a database of vectors to allow a user to efficiently compute similarities and find nearest neighbors relative to a set of query points. For high compression, modern ANN search algorithms employ vector quantization (VQ)-which quantizes multiple vector dimensions jointly (Burton et al., 1983; Gray, 1984). It achieves this by learning "codebooks": i.e. a set of learnable candidate vectors that can be used to encode the data. To encode a given database vector, VQ splits it into sub-groups of entries, then encodes every group by choosing a vector from the learned codebook. The algorithm efficiently computes distances or dot-products for similarity search by leveraging the linearity of dot products.

Quantization methods for ANN search generalize vector quantization and are referred to as multi-codebook quantization (MCQ). MCQ methods typically do not involve information loss on the query side, which makes them the leading approach for memory-efficient ANN (Ozan et al., 2016; Martinez et al., 2018). We briefly review MCQ below.

Product quantization (PQ) (Jegou et al., 2010) is an early version of MCQ, which encodes each vector $x \in \mathbf{R}^{D}$ as a concatenation of $M$ codewords from $M \frac{D}{M}$-dimensional codebooks $C_{1}, \ldots, C_{M}$, each containing $K$ codewords. PQ decomposes a vector into $M$ separate subvectors and applies vector quantization (VQ) to each subvector, while using a separate codebook. Thus, each vector $x$ is encoded by a tuple of codeword indices $\left[i_{1}, \ldots, i_{M}\right]$ and approximated by $x \approx\left[c_{1 i_{1}}, \ldots, c_{M i_{M}}\right]$. Fast Euclidean distance computation becomes possible using lookup tables:

$\|q-x\|^{2} \approx\left\|q-\left[c_{1 i_{1}}, \ldots, c_{M i_{M}}\right]\right\|^{2}=\sum_{m=1}^{M}\left\|q_{m}-c_{m i_{m}}\right\|^{2}$,

where $q_{m}$ is the $m$ th subvector of a query $q$. This sum can be calculated using $M$ additions and lookups if the distances from query subvectors to codewords are precomputed. Since product-based approximations work better if the $\frac{D}{M}$ dimensional components independent distributions, subsequent work has looked into finding better transformations (Ge et al., 2013; Norouzi \& Fleet, 2013). As for the other similarity functions, (Guo et al., 2016) proposes a quantization procedure for maximum inner product search (MIPS). They minimize quantization error in the inner products be- tween database and query vectors by solving a constrained optimization problem. Similarly to the formula above, this procedure allows for efficient inner product search by precomputing dot products between the query $q$ an all codes in the learned codebooks, then adding these partial dot products to recover the full similarity score.

Non-orthogonal quantizations. Follow-up work (Chen et al., 2010; Babenko \& Lempitsky, 2014; Martinez et al., 2016; Zhang et al., 2014; Ozan et al., 2016; Martinez et al., 2018) generalized the idea of Product Quantization by approximating each vector by a sum of $M$ codewords instead of concatenation. The resulting procedure is still efficient while the approximation accuracy is increased.

For this, Residual Vector Quantization (Chen et al., 2010), quantizes original vectors, and then iteratively quantizes the approximation residuals from the previous iteration. Additive Quantization (AQ) (Babenko \& Lempitsky, 2014) is more general, as it does not impose constraints on the codewords from the different codebooks. Usually, AQ provides the smallest compression errors, but is more complex to train for large $M$. We discuss this in detail in Section 3.

Finally, several recent works (Martinez et al., 2016; 2018; Zhang et al., 2014) elaborate the idea of Additive Quantization, proposing the more effective procedure for codebooks learning. Composite Quantization (CQ) (Zhang et al., 2014) learns codebooks with a fixed value of scalar product between the codewords from different codebooks. Currently, the state-of-the-art compression accuracy is achieved by the LSQ method (Martinez et al., 2018).

Vector quantization for model compression. There has been significant work on exploiting vector quantization in the context of machine learning. For instance, Zhou et al. (2017); Li et al. (2017); Chen et al. (2019) use multi-codebook quantization to compress word embeddings within deep learning models. Another line of work (Blalock \& Guttag, 2021; McCarter \& Dronen, 2022; FernándezMarqués et al., 2023) explores vector quantization for linear models, or linear layers within deep models. Similarly to $\mathrm{PQ}$ above, these techniques pre-compute inner products between inputs and all codes, then compute linear layer via look-up, which speeds up inference. However, these algorithms introduce significant prediction error that does not allow them to compress deep models. Thus, we believe we are the first to successfully adapt and scale MCQ to LLMs.

## 3. AQLM: Additive Quantization for LLMs

### 3.1. Overview

We start from the observation that additive quantization (AQ) solves a related problem to post-training quantization (PTQ) (Nagel et al., 2020; Frantar et al., 2022b): both settings assume the existence of a set of "input" vectors, i.e.
input data for AQ, and the weight matrix rows for PTQ. The goal is to compress these inputs while preserving dot product similarity, against query vectors (for AQ), and against layer input embeddings (for PTQ). The difference between the two is that AQ assumes that the distribution of queries is unknown, whereas PTQ methods, e.g. (Frantar et al., 2022b), show that it is sufficient to optimize for sample input embeddings from a set of calibration data.

At a high level, we start by solving the following problem: for a linear layer with $d_{i n}$ input and $d_{o u t}$ output features given its weights $\mathbf{W} \in \mathbb{R}^{d_{\text {out }} \times d_{\text {in }}}$ and a set of calibration inputs $\mathbf{X} \in \mathbb{R}^{d_{i n} \times n}$, one seeks for a configuration of quantized weights $\hat{W}$ that optimizes squared error between the output of the original and compressed layer:

$$
\begin{equation*}
\underset{\widehat{\mathbf{W}}}{\arg \min }\|\mathbf{W} \mathbf{X}-\widehat{\mathbf{W}} \mathbf{X}\|_{2}^{2} \tag{1}
\end{equation*}
$$

In the following, we will assume that $\widehat{\mathbf{W}}$ is quantized using AQ, and adopt standard notation (Martinez et al., 2016). AQ splits weight rows into groups of $g$ consecutive elements, and represents each group of weights as a sum of $M$ vectors chosen from multiple learned codebooks $C_{1}, \ldots C_{M}$, each containing $2^{B}$ vectors (for $\mathrm{B}$-bit codes). A weight is encoded by choosing a single code from each codebook and summing them up. We denote this choice as a one-hot vector $b_{m}$, which results in the following representation for a group: $\sum_{m=1}^{M} C_{m} b_{i j m}$. This is similar to PTQ algorithms (Frantar et al., 2022a), except for using much more complex coding per group. To represent the full weights, we simply concatenate:

$$
\begin{equation*}
\widehat{\mathbf{W}}_{i}=\sum_{m=1}^{M} C_{m} b_{i, 1, m} \oplus \ldots \oplus \sum_{m=1}^{M} C_{m} b_{i, d_{i n} / g, m} \tag{2}
\end{equation*}
$$

where $\oplus$ denotes concatenation and $b_{i j m} \in \mathbb{R}^{2^{B}}$ represents a one-hot code for the $i$-th output unit, $j$-th group of input dimensions and $m$-th codebook.

![](https://cdn.mathpix.com/cropped/2024_06_04_46c218fad80e5caf3b83g-04.jpg?height=236&width=228&top_left_y=1823&top_left_x=211)

Weight Matrix

![](https://cdn.mathpix.com/cropped/2024_06_04_46c218fad80e5caf3b83g-04.jpg?height=341&width=523&top_left_y=1800&top_left_x=451)

Figure 2: Groups of weights are represented by a sum of codes selected from codebooks by corresponding indices.

Our algorithm will learn codebooks $C_{m} \in \mathbb{R}^{g \times 2^{B}}$ and the discrete codes represented by one-hot $b \in$ $\mathbb{R}^{d_{\text {out }} \times d_{\text {in }} / g \times M \times 2^{B}}$. The resulting scheme encodes each group of $g$ weights using $M \cdot B$ bits and further requires $g \cdot 2^{B} \cdot 16$ bits for FP16 codebooks. The error becomes:

$$
\begin{equation*}
\underset{C, b}{\arg \min } \| \mathbf{W X}-\left(\text { Concat }_{i, j} \sum_{m=1}^{M} C_{m} b_{i, j, m}\right) \mathbf{X} \|_{2}^{2} \tag{3}
\end{equation*}
$$

To learn this weight representation, we initialize codebooks $C$ and codes $b$ by running residual K-means as in Chen et al. (2010). Then, we alternate between updating codes $b_{i, j, m}$ and codebooks $C_{m}$ until the loss function (3) stops improving up to the specified tolerance. Since codes are discrete and codebooks are continuous, and we are optimizing over multiple interacting layers, our approach has three phases, described in Algorithm 1 and detailed below.

### 3.2. Phase 1: Beam search for codes

First, AQLM updates the codes $b_{i, j, m}$ to minimize the MSE objective (3). Similarly to Babenko \& Lempitsky (2014); Martinez et al. (2016; 2018), we reformulate the objective in terms of a fully-connected discrete Markov Random Field (MRF) to take advantage of MRF solvers.

To simplify the derivation, let us first consider a special case of a single output unit $\left(d_{\text {out }}=1\right)$ and a single quantization group (i.e. $g=d_{i n}$ ), to get rid of the concatenation operator: $\left\|\mathbf{W X}-\sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\|_{2}^{2}$. We rewrite this objective by expanding the squared difference:

$$
\begin{align*}
& \left\|\mathbf{W X}-\sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\|_{2}^{2}=\|\mathbf{W} \mathbf{X}\|_{2}^{2}- \\
& -2\left\langle\mathbf{W X}, \sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\rangle_{F}+\left\|\sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\|_{2}^{2} \tag{4}
\end{align*}
$$

Above, $\langle\cdot, \cdot\rangle_{F}$ denotes a Frobenius inner product of two matrices. Next, let us consider the three components of Eqn. (4) in isolation. First, note that $\|\mathbf{W} \mathbf{X}\|_{2}^{2}$ is constant in $b$ and can be ignored. The second component can be expanded further into pairwise dot products:

$$
\begin{equation*}
\left\|\sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\|_{2}^{2}=\sum_{i=1}^{M} \sum_{j=1}^{M}\left\langle C_{i} b_{i} \mathbf{X}, C_{j} b_{j} \mathbf{X}\right\rangle_{F} \tag{5}
\end{equation*}
$$

Note that both the second and third components rely on Frobenius products of $C_{m} b_{m} \mathbf{X}$-like matrices. These matrices can be inconvenient in practice: since $\mathbf{X} \in \mathbb{R}^{d_{i n} \times n}$, the size of each matrix will scale with the size of calibration dataset $n$. To circumvent this, we rewrite the products as:

$$
\begin{equation*}
\left\langle C_{i} b_{i} \mathbf{X}, C_{j} b_{j} \mathbf{X}\right\rangle_{F}=\left\langle C_{i} b_{i} \mathbf{X} \mathbf{X}^{T}, C_{j} b_{j}\right\rangle_{F} \tag{6}
\end{equation*}
$$

Thus one can pre-compute $\mathbf{X} \mathbf{X}^{T} \in \mathbb{R}^{d_{i n} \times d_{i n}}$. We will denote this type of product as $\langle\mathbf{A}, \mathbf{B}\rangle_{\mathbf{X X}^{T}} \stackrel{\text { def }}{=}\left\langle\mathbf{A X X}{ }^{T}, \mathbf{B}\right\rangle_{F}$ in future derivations. Then, Eqn. (4) becomes:

$$
\begin{align*}
\left\|\mathbf{W X}-\sum_{m=1}^{M} C_{m} b_{m} \mathbf{X}\right\|_{2}^{2}=\|\mathbf{W} \mathbf{X}\|_{2}^{2}- \\
-2 \sum_{m=1}^{M}\left\langle\mathbf{W}, C_{m} b_{m}\right\rangle_{\mathbf{X} \mathbf{x}^{T}}+\sum_{i=1}^{M} \sum_{j=1}^{M}\left\langle C_{i} b_{i}, C_{j} b_{j}\right\rangle_{\mathbf{X X}^{T}} \tag{7}
\end{align*}
$$

Finally, we generalize this equation to multiple output units $\left(d_{\text {out }}>1\right)$ and quantization groups $\left(g \neq d_{\text {in }}\right)$. For $d_{\text {out }}>1$, note that the original objective (3) is additive with respect to output units: thus, we can apply (7) independently to each output dimension and sum up results. To support multiple input groups $\left(g \neq d_{i n}\right)$, we can treat each group as a separate codebook where only the codes for the active group are nonzero. Thus, we need to repeat each codebook $d_{i n} / g$ times and pad it with zeros according to the active group.

It is now evident that minimizing (4) is equivalent to MAP inference in a Markov Random Field with $\left\langle\mathbf{W}, C_{m} b_{m}\right\rangle \mathbf{X X}^{T}$ as unary potentials and $\left\langle C_{i} b_{i}, C_{j} b_{j}\right\rangle_{\mathbf{X x}^{T}}$ as pairwise potentials. While finding the exact optimum is infeasible, prior work has shown that this type of MRF can be solved approximately via beam search or ICM (Besag, 1986).

To solve this problem, we chose to adapt a beam search algorithm from Babenko \& Lempitsky (2014). This algorithm maintains a beam of $k$ (beam size) best configurations for the codes, starting from the previous solution. On each step, the algorithm attempts to replace one code by trying all $2^{B} k$ alternatives and selecting the $k$ best based on MSE (7).

Since the loss function is additive, changing one code only affects a small subset of loss components. Thus, we can compute the loss function efficiently by starting with a previous loss function (before code replacement), then adding and subtracting the components that changed during this iteration. These few loss components can be computed efficiently by multiplying with $\mathbf{X} \mathbf{X}^{T}$ ahead of beam search. The beam search runs over all $d_{o u t}$ output units in parallel. This is possible because encoding one output unit does not affect the objective (7) of other units. Note that beam search is not necessarily the best solution to this problem. AQ variants for retrieval (Martinez et al., 2016; 2018) use randomized ICM to find solutions faster. In this study, we chose beam search because it was easier to implement in ML frameworks like PyTorch/JAX.

### 3.3. Phase 2: Codebook update

In the second phase, we find the optimal codebook vectors $C_{1}, \ldots, C_{M}$ that minimize the same squared error as the beam search. If we treat the codes $b$ as constants, minimizing (3) becomes a least squares problem for $C_{m}$. The original AQ algorithm solves this problem in closed form, relying on the fact that each vector dimension can be optimized independently. Our problem is complicated due to the presence of $\mathbf{X} \mathbf{X}^{T}$ : the optimal value of one codebook coordinate depends on the values of all others. In principle, we could optimize $C_{m}$ in closed form, but it would require inverting a large matrix, or using iterative least squares solvers (e.g. conjugate gradients) specialized to this problem.

For simplicity, our current implementation defaults to using Adam (Kingma \& Ba, 2015) for approximately solving this minimization problem. In practice, this codebook tuning phase takes up a small fraction of the total compute time. We compute the objective as follows:

$$
\begin{align*}
\|\mathbf{W} \mathbf{X}-\widehat{\mathbf{W}} \mathbf{X}\|_{2}^{2} & =\|(\mathbf{W}-\widehat{\mathbf{W}}) \mathbf{X}\|_{2}^{2}= \\
& =\left\langle(\mathbf{W}-\widehat{\mathbf{W}}) \mathbf{X} \mathbf{X}^{T},(\mathbf{W}-\widehat{\mathbf{W}})\right\rangle_{F} \tag{8}
\end{align*}
$$

where $\widehat{\mathbf{W}}$ is the quantized weight matrix from 2 , and the $\mathbf{X X}^{T}$ matrix is pre-computed. We optimize this objective by iterating (non-stochastic) full-batch gradient descent.

For each update phase, our implementation runs 100 Adam steps with learning rate 1e-4. However, we found that the final result is not sensitive to either of these parameters: training with smaller number of steps or learning rate achieves the same loss, but takes longer to converge. In future work, these hyperparameters could be eliminated by switching to dedicated least squares solver for codebooks. Similarly to other algorithms, we also learn per-unit scales $s \in \mathbb{R}^{h}$ that are initialized as $s_{i}:=\left\|\mathbf{W}_{i}\right\|_{2}$ and updated alongside codebooks via the same optimizer (line 19 in Algorithm 1).

### 3.4. Phase 3: Fine-tuning for intra-layer cohesion

So far, our algorithm compresses each weight matrix independently of the rest of the model. However, in practice, quantization errors interact differently between matrices. This issue is especially relevant in the case of extreme (2bit) compression, where quantization errors are larger.

Prior work addresses this issue via quantization-aware training (QAT), e.g. (Gholami et al., 2021). Instead of compressing the entire model in a single pass, they quantize model parameters gradually and train the remaining parameters to compensate for the quantization error. Unfortunately, running QAT in our setting is infeasible, since most modern LLMs are extremely expensive to train or even fine-tune. Thus, most PTQ algorithms for LLMs only adjust model parameters within the same linear layer (Frantar et al., 2022a; Lin et al., 2023; Dettmers et al., 2023b).

Here, we opt for a middle ground by performing optimization at the level of individual transformer blocks, i.e. groups of 4-8 linear layers ${ }^{2}$ that constitute a single multi-head self-[^2]

![](https://cdn.mathpix.com/cropped/2024_06_04_46c218fad80e5caf3b83g-06.jpg?height=284&width=1705&top_left_y=197&top_left_x=188)

Figure 3: AQLM compressed weight format. Horizontal and vertical axes are input features and output units, respectively. Depth represents the codebook index. Reconstruction procedure, from left to right: i) compressed weight codes ii) zoom-in one weight group, each code is an index in its respective codebook iii) select codes from each codebook iv) add up codes as in (2) v) multiply by scales (one scale per output dimension).

```
Algorithm 1 AQLM: Additive Quantization for LLMs
Require: model, data
    $\mathbf{X}_{\text {block }}:=$ model.input_embeddings (data)
    for $i=1, \ldots$,model.num_layers do
        block := model.get_block(i)
        $\mathbf{Y}_{\text {block }}:=\operatorname{block}\left(\mathbf{X}_{\text {block }}\right)$
        for layer $\in$ linear_layers (block) do
            $\mathbf{W}:=$ layer.weight
            $\mathbf{X}:=$ layer_inputs(layer, $\mathbf{X}_{\text {block }}$ )
            $C, b, s:=$ initialize(W) // k-means
            while loss improves by at least $\tau$ do
                    $C, s:=$ train_Cs_adam $\left(\mathbf{X X}^{T}, \mathbf{W}, C, b, s\right)$
                    $b:=$ beam_search $\left(\mathbf{X X}^{T}, \mathbf{W}, C, b, s\right)$
            end while
            /* save for fine-tuning */
            layer.weight:= AQLMFormat $(C, b, s)$
        end for
        $\theta:=$ trainable_parameters(block)
        while loss improves by at least $\tau$ do
            $L:=\left\|b l o c k\left(\mathbf{X}_{\text {block }}\right)-\mathbf{Y}_{\text {block }}\right\|_{2}^{2}$
            $\theta:=\operatorname{adam}\left(\theta, \frac{\partial L}{\partial \theta}\right)$
        end while
        $\mathbf{Y}_{\text {block }}:=\operatorname{block}\left(\mathbf{X}_{\text {block }}\right)$
    end for
```

attention, followed by a single and MLP layer. Having quantized all linear layers within a single transformer block, we fine-tune its remaining parameters to better approximate the original outputs of that transformer block by backpropagating through the weight representation (2).

Concretely, we use the PyTorch autograd engine to differentiate the $\left\|\operatorname{block}\left(\mathbf{X}_{\text {block }}\right)-\mathbf{Y}_{\text {block }}\right\|^{2}$, where $\mathbf{X}_{\text {block }}$ are the inputs activations for that transformer block and $\mathbf{Y}_{\text {block }}$ are output activations of block $\left(\mathbf{X}_{\text {block }}\right)$ recorded prior to quantization. We train the codebooks $C_{m}$, scale vectors $s$ and all non-quantized parameters (RMSNorm scales and biases), while keeping the codes $b_{i, j, m}$ frozen. Similarly to Section 3.3, we train these parameters using Adam to minimize the MSE against the original block outputs (prior to quantization). This phase uses the same calibration data as for the individual layer quantization. The full procedure is summarized in Alg. 1.

While fine-tuning blocks is more expensive than individual linear layers, it is still possible to quantize billion-parameter

![](https://cdn.mathpix.com/cropped/2024_06_04_46c218fad80e5caf3b83g-06.jpg?height=46&width=813&top_left_y=2433&top_left_x=190)

models on a single GPU in reasonable time. Also, since the algorithm only modifies a few trainable parameters, it uses little VRAM for optimizer states. This fine-tuning converges after a few iterations, as it starts from a good initial guess. In practice, fine-tuning transformer layers takes a minority (10-30\% or less) of the total calibration time.

## 4. Experiments

We evaluate the AQLM algorithm in typical scenarios for post-training quantization of modern LLMs. Our evaluation is focused on the LLAMA 2 model family since it is a popular backbone for fine-tuned models or general LLM applications, e.g. (Dettmers et al., 2023a), and we also present results on Mistral-family models (Jiang et al., 2024). In Section 4.1, we evaluate the full AQ procedure for various LLAMA 2 models and quantization bit-widths; Section 4.2 presents an ablation analysis for individual AQ components and implementation details.

### 4.1. Compression quality for modern LLMs

We report perplexity on WikiText2 (Merity et al., 2016) and C4 (Raffel et al., 2020) validation sets. We also measure zero-shot accuracy on WinoGrande (Sakaguchi et al., 2021), PiQA (Tata \& Patel, 2003), HellaSwag (Zellers et al., 2019), ARC-easy and ARC-challenge (Clark et al., 2018) via the LM Eval Harness (Gao et al., 2021). We broadly follow the evaluation setup of GPTQ (Frantar et al., 2022a).

We consider three main targets in terms of compression ranges: 2-2.8 bits, 3-3.1 bits, and 4-4.1 bits per model parameter. In the results below average bits per parameter takes into account only quantized weights, we do not include parameters kept in floating precision similarly to the related work. The details on the model size estimate are provided in Appendix G. We compare AQ against GPTQ for $3 \& 4$ bits (Frantar et al., 2022a), SpQR for 3\&4 bits (Dettmers et al., 2023b), QuIP in 2,3 \& 4 bits (Chee et al., 2023) and QuIP\# for $2 \& 4$ bits (Tseng et al., 2023). While GPTQ and SpQR technically support 2-bit quantization, they perform poorly in the 2-3 bit range. For QuIP, we omit results for the 7B model, as we could not achieve competitive performance in this one scenario using the available implementations. (Currently, there is no official implementation of the origi-

Extreme LLM Compression via Additive Quantization

Table 1: Evaluation of quantized LLAMA 2 models for 2-2.8 bits per parameter, with an extra section for higher bitwidth. We report perplexity on WikiText2 (Merity et al., 2016) \& C4 (Raffel et al., 2020) and accuracy for zero-shot tasks. The Average accuracy is the mean of 5 zero-shot tasks. Primary metrics are Wiki2 (PPL), C4 (PPL) and Average accuracy.

| Size | Method | Avg bits | $\mid$ Wiki $2 \downarrow$ | $\mathrm{C} 4 \downarrow$ | $\mid$ WinoGrande $\uparrow$ | PiQA $\uparrow$ | HellaSwag $\uparrow$ | $\operatorname{ArcE} \uparrow$ | $\operatorname{ArcC} \uparrow$ | Average accuracy $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7B | - | 16 | 5.12 | 6.63 | 67.25 | 78.45 | 56.69 | 69.32 | 40.02 | 62.35 |
|  | AQLM | 2.02 | 6.64 | 8.56 | 64.17 | 73.56 | 49.49 | 61.87 | 33.28 | 56.47 |
|  | QuIP\# | 2.02 | 8.22 | 11.01 | 62.43 | 71.38 | 42.94 | 55.56 | 28.84 | 52.23 |
|  | AQLM | 2.29 | 6.29 | 8.11 | 65.67 | 74.92 | 50.88 | 66.50 | 34.90 | 58.57 |
| 13B | - | 16 | 4.57 | 6.05 | 69.61 | 78.73 | 59.72 | 73.27 | 45.56 | 65.38 |
|  | $\mathrm{AQLM}$ | 1.97 | 5.65 | 7.51 | 65.43 | 76.22 | 53.74 | 69.78 | 37.8 | 60.59 |
|  | QuIP | 2.00 | 13.48 | 16.16 | 52.80 | 62.02 | 35.80 | 45.24 | 23.46 | 43.86 |
|  | QuIP\# | 2.01 | 6.06 | 8.07 | 63.38 | 74.76 | 51.58 | 64.06 | 33.96 | 57.55 |
|  | AQLM | 2.18 | 5.41 | 7.20 | 68.43 | 76.22 | 54.68 | 69.15 | 39.42 | 61.58 |
|  | AQLM | 2.53 | 5.15 | 6.80 | 68.11 | 76.99 | 56.54 | 71.38 | 40.53 | 62.71 |
|  | AQLM | 2.76 | 4.94 | 6.54 | 68.98 | 77.58 | 57.71 | 72.90 | 43.60 | 64.15 |
| 70B | - | 16 | 3.12 | 4.97 | 76.95 | 81.07 | 63.99 | 77.74 | 51.11 | 70.17 |
|  | AQLM | 2.07 | 3.94 | 5.72 | 75.93 | 80.43 | 61.79 | 77.68 | 47.93 | 68.75 |
|  | QuIP | 2.01 | 5.90 | 8.17 | 67.48 | 74.76 | 50.45 | 62.16 | 33.96 | 57.76 |
|  | QuIP\# | 2.01 | 4.16 | 6.01 | 74.11 | 79.76 | 60.01 | 76.85 | 47.61 | 67.67 |

Table 2: Evaluation of quantized LLAMA 2 models for 3-3.1 bits per parameter, with the same metrics as in Table 1.

| Size | Method | Avg bits | $\mid$ Wiki $2 \downarrow$ | $\mathrm{C} 4 \downarrow$ | WinoGrande $\uparrow$ | PiQA $\uparrow$ | HellaSwag $\uparrow$ | $\operatorname{ArcE} \uparrow$ | $\operatorname{ArcC} \uparrow$ | Average accuracy $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7B | - | 16 | 5.12 | 6.63 | 67.25 | 78.45 | 56.69 | 69.32 | 40.02 | 62.35 |
|  | AQLM | 3.04 | 5.46 | 7.08 | 66.93 | 76.88 | 54.12 | 68.06 | 38.40 | 60.88 |
|  | GPTQ | 3.00 | 8.06 | 10.61 | 59.19 | 71.49 | 45.21 | 58.46 | 31.06 | 53.08 |
|  | $\mathrm{SpQR}$ | 2.98 | 6.20 | 8.20 | 63.54 | 74.81 | 51.85 | 67.42 | 37.71 | 59.07 |
| 13B | - | 16 | 4.57 | 6.05 | 69.61 | 78.73 | 59.72 | 73.27 | 45.56 | 65.38 |
|  | AQLM | 3.03 | 4.82 | 6.37 | 68.43 | 77.26 | 58.30 | 70.88 | 42.58 | 64.49 |
|  | GPTQ | 3.00 | 5.85 | 7.86 | 63.93 | 76.50 | 53.47 | 65.66 | 38.48 | 59.61 |
|  | SpQR | 2.98 | 5.28 | 7.06 | 67.48 | 77.20 | 56.34 | 69.78 | 39.16 | 61.99 |
|  | QuIP | 3.00 | 5.12 | 6.79 | 69.93 | 76.88 | 57.07 | 70.41 | 41.47 | 63.15 |
| 70B | - | 16 | 3.12 | 4.97 | 76.95 | 81.07 | 63.99 | 77.74 | 51.11 | 70.17 |
|  | AQLM | 3.01 | 3.36 | 5.17 | 77.19 | 81.28 | 63.23 | 77.61 | 50.00 | 69.86 |
|  | GPTQ | 3.00 | 4.40 | 6.26 | 71.82 | 78.40 | 60.00 | 72.73 | 44.11 | 65.41 |
|  | $\mathrm{SpQR}$ | 2.98 | 3.85 | 5.63 | 74.66 | 80.52 | 61.95 | 75.93 | 48.04 | 68.22 |
|  | QuIP | 3.01 | 3.87 | 5.67 | 74.59 | 79.98 | 60.73 | 73.19 | 46.33 | 66.96 |

nal QuIP (non-\#) for the LLAMA 2 model.) For QuIP\#, we focus on 2 and 4 bit because the available implementation does not yet support 3-bit compression. We calibrate each algorithm using the subset of RedPajama dataset (Computer, 2023), with a sequence length of 4096.

The exact bit-widths for each method are dictated by parameters such as the number of codebooks and code width. We report results for the $2-2.8$ and $3-3.1$ bitwidth ranges in Tables 1 and 2, respectively. Additional results for $4-4.1$ bits are deferred to Appendix E.2.

The results show that $\mathrm{AQLM}$ outperforms the previous best PTQ algorithms across all settings, often by wide margins, especially at high compression. This holds both in terms of
PPL across standard validation sets (Wiki-Text2 and C4), and accuracy across zero-shot tasks. Specifically, we observe the highest accuracy gains in the "extreme" 2-2.1 bits per parameter range, where the deviation from the uncompressed model becomes large for all methods.

Mixtral quantization. Table 3 presents results on the Mixtral MoE-type model, comparing against QuIP\# at 2-bits. (See Appendix E. 1 for full results.) AQLM outperforms QuIP\# in this case as well. Although the margins are lower compared to LLAMA 2 models, they are still significant for "harder" tasks, such as Arc Challenge ( +3 points).

Pareto optimality of AQLM. The significant error improve-

Table 3: Evaluation of quantized Mixtral (Jiang et al., 2024) models for 2 bits. The table reports perplexity on WikiText2 (Merity et al., 2016) and C4 (Raffel et al., 2020), as well as accuracy for zero-shot tasks. The Average accuracy column is the mean of 5 zero-shot task accuracies. Primary metrics are Wiki2 (PPL), C4 (PPL) and Average accuracy.

| Size | Method | Avg bits | Wiki $2 \downarrow$ | C4 $\downarrow$ | WinoGrande $\uparrow$ | PiQA $\uparrow$ | HellaSwag $\uparrow$ | ArcE $\uparrow$ | ArcC $\uparrow$ | Average accuracy $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8x7B |  |  |  |  |  |  |  |  |  |  |
|  | - | 16 | 3.46 | 5.02 | 75.45 | 82.37 | 64.65 | 83.38 | 55.80 | 72.33 |
|  | QuLM | 1.98 | $\mathbf{4 . 6 1}$ | $\mathbf{5 . 7 5}$ | $\mathbf{7 3 . 6 4}$ | $\mathbf{7 9 . 2 7}$ | 57.91 | $\mathbf{7 8 . 9 6}$ | $\mathbf{4 8 . 6 3}$ | $\mathbf{6 7 . 6 8}$ |
|  | QuIP\# | 2.01 | 4.75 | 5.89 | 71.11 | 79.05 | $\mathbf{5 8 . 2 3}$ | 77.57 | 45.73 | 66.34 |

ments raise the question of choosing the "optimal" model variant to maximize accuracy within a certain memory budget. For this, we follow Dettmers \& Zettlemoyer (2022): a quantized model is said to be Pareto-optimal if it maximizes accuracy at the same or lower total size (bytes). Despite rapid progress, prior art methods are not Pareto-optimal at 2-bits: for instance, the previous best 2-bit LLAMA 2 13B (QuIP\#, Table 1) achieves Wiki2 PPL of 6.06, but one can get much lower 5.21 PPL by using a 7B model with 4-bit quantization, which is smaller (see Appendix Table 10).

AQLM compression to strictly 2 bits for the same model is also below Pareto-optimality, as it is outperformed by 4-bit AQLM compression for LLAMA 2 7B (5.21 vs 5.65). To find the Pareto-optimal quantization bitwidth, we run experiments between 2-3 bits per parameter and report them in Table 1, below horizontal bars. Thus, the Pareto-optimal bitwidth for AQLM appears to be around 2.5 bits per parameter (Table 1), at which point we are comparable to 5-bit AQLM for LLAMA 2 7B (Appendix Table 10). In turn, the 2.76-bit AQLM on 13B outperforms the uncompressed 7B model. As such, AQLM is the first algorithm to achieve Pareto-optimality at less than 3 bits per parameter.

### 4.2. Ablation analysis

In Appendix D, we examine key design choices regarding initialization, alternating optimization, the impact of the fine-tuning protocol, distribution of codebooks vs groups, as well as other hyper-parameters. In brief, we first find that the residual $K$-means initialization is critical for fast algorithm convergence: when compared with random initialization, it needs significantly fewer training iterations. Second, to validate our calibration fine-tuning procedure, we compare it against 1) no fine-tuning, 2) fine-tuning only of non-linear layers (e.g. RMSNorm) but not of codebook parameters, and 3) fine-tuning only the codebooks (but not other layers). The results, presented in full in Appendix D, show that finetuning the codebook parameters has the highest impact on accuracy, by far, while fine-tuning the RMSNorm only has minor impact. This validates our choice of leveraging the calibration set for learned codebooks.

Further, we observe that increasing the number of sample sequences in the range 128 to 4096 leads to a gradual PPL improvement, but with diminishing returns. In this respect, AQLM benefits more from larger calibration sets (similarly to QuIP\#), as opposed to direct methods like GPTQ which saturate accuracy at around 256 input sequences. Finally, we investigate various options for investing a given bit budget, comparing e.g. longer codes (e.g. $1 \times 15$ ) vs multiple codebooks with shorter codes (e.g. $2 x 8$ ).

### 4.3. Inference Speed

Although our primary objective is to maximize accuracy for a given model size, AQLM can also be practical in terms of inference latency. To demonstrate this, we implemented efficient GPU and CPU kernels for a few hardware-friendly configurations of AQLM. The results can be found in Table 4. For GPU inference, we targeted quantized Llama 2 models with 16 -bit codebooks, corresponding to 2.07 bits for LLAMA 2 70B, 2.18 bits for 13B, and 2.29 bits for 7B models (see Table 1), as well as a $2 \times 8$-bit codebook model with perplexity 7.98 on Wiki2. For each model we benchmark the matrix-vector multiplication subroutine performance on a standard layer. The results show that AQLM can execute at speeds comparable to or better than FP16. End-to-end generative numbers on a preliminary HuggingFace integration can be found in Appendix H: for instance, we can achieve $\sim 6$ tokens/s on LLAMA 2 70B in this setting. We observe that multiple smaller codebooks allow efficient GPU cache utilization, leading to greater speedup, at the price of slightly lower accuracy.

Table 4: Speed of the FP16 gate_proj layer matrix-vector multiplication in PyTorch, and relative AQLM speedups.

| Llama 2 | $7 \mathrm{~B}$ | 13B | $70 B$ |
| :---: | :---: | :---: | :---: |
| 2 bit speedup over FP16 on Nvidia RTX 3090 GPU |  |  |  |
| Original (float16) | $129 \mu \mathrm{s}$ | $190 \mu \mathrm{s}$ | $578 \mu \mathrm{s}$ |
| AQLM (Table 1) | $\mathrm{x} 1.31$ | $\mathrm{x} 1.20$ | $\mathrm{x} 1.20$ |
| AQLM $(2 \times 8$-bit $)$ | $\mathrm{x} 1.57$ | $\mathrm{x} 1.82$ | $\mathrm{x} 3.05$ |
| 2 bit speedup over FP32 on Intel i9 CPU, 8 cores |  |  |  |
| Original (float32) | $1.83 \mathrm{~ms}$ | $3.12 \mathrm{~ms}$ | $11.31 \mathrm{~ms}$ |
| AQLM (2×8-bit) | $x 2.75$ | $\mathrm{x} 3.54$ | $x 3.69$ |
| AQLM (4×8-bit) | $x 2.55$ | $x 3.02$ | $x 4.07$ |
| AQLM (8×8-bit) | $\mathrm{x} 2.29$ | $x 2.68$ | $x 4.03$ |

Next, we explore how to leverage AQLM to accelerate CPU inference. As discussed in Section 2.2, additive quantization can compute dot products efficiently if the codebook size is small. One way to achieve it for AQLM is to replace each 16-bit codebook with a number of smaller 8-bit ones. This leads to higher quantization error, but still outperforms the baselines in terms of accuracy (see Appendix Table 9). The results in Table 4 show that this also allows for up to $4 x$ faster inference relative to $\mathrm{FP} 32$ on CPU.

## 5. Conclusion and Future Work

We presented AQLM, a new form of additive quantization (AQ) targeted to LLM compression, which significantly improved the state-of-the-art results for LLM quantization in the regime of 2 and 3 bits per weight. In terms of limitations, AQLM is more computationally-expensive than direct post-training quantization methods, such as RTN or GPTQ, specifically because of the use of a more complex coding representation. Yet, despite the more sophisticated encoding and decoding, we have shown AQLM lends itself to efficient implementation on both CPU and GPU. Overall, we find it remarkable that, using AQLM, massive LLMs can be executed accurately and efficiently using little memory.

## 6. Acknowledgements

Authors would like to thank Ruslan Svirschevski for his help in solving technical issues with AQLM and baselines. We also thank Tim Dettmers for helpful discussions on the structure of weights in modern LLMs and size-accuracy trade-offs. The authors would also like to thank Daniil Pavlov for his assistance with CPU benchmarking. Finally, authors would like to thank the communities of ML enthusiasts known as LocalLLaMA ${ }^{3}$ and Petals community on discord ${ }^{4}$ for the crowd wisdom about running LLMs on consumer devices.

## References

Babenko, A. and Lempitsky, V. Additive quantization for extreme vector compression. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 931-938, 2014.

Besag, J. On the statistical analysis of dirty pictures. Journal of the Royal Statistical Society Series B: Statistical Methodology, 48(3):259-279, 1986.

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for ana-[^3]

lyzing large language models across training and scaling. arXiv preprint arXiv:2304.01373, 2023.

Blalock, D. and Guttag, J. Multiplying matrices without multiplying. In International Conference on Machine Learning, pp. 992-1004. PMLR, 2021.

Burton, D., Shore, J., and Buck, J. A generalization of isolated word recognition using vector quantization. In ICASSP '83. IEEE International Conference on Acoustics, Speech, and Signal Processing, volume 8, pp. 1021-1024, 1983. doi: 10.1109/ICASSP.1983.1171915.

Chee, J., Cai, Y., Kuleshov, V., and Sa, C. D. Quip: 2-bit quantization of large language models with guarantees, 2023.

Chen, S., Wang, W., and Pan, S. J. Deep neural network quantization via layer-wise optimization using limited training data. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):3329-3336, Jul. 2019. doi: 10.1609/aaai.v33i01.33013329. URL https://ojs.aaai.org/index.php/AAAI / article/view/4206.

Chen, Y., Guan, T., and Wang, C. Approximate nearest neighbor search by residual vector quantization. Sensors, $10(12): 11259-11273,2010$.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai 2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Computer, T. Redpajama: an open dataset for training large language models, 2023. URL https://github.com/togethercomputer/ RedPajama-Data.

Dettmers, T. and Zettlemoyer, L. The case for 4-bit precision: k-bit inference scaling laws. arXiv preprint arXiv:2212.09720, 2022.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. LLM.int8(): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023a.

Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T., and Alistarh, D. Spqr: A sparse-quantized representation for near-lossless $11 \mathrm{~m}$ weight compression. arXiv preprint arXiv:2306.03078, 2023b.

Fernández-Marqués, J., AbouElhamayed, A. F., Lane, N. D., and Abdelfattah, M. S. Are we there yet? product quantization and its hardware acceleration. ArXiv, abs/2305.18334, 2023. URL https: //api.semanticscholar.org/CorpusID: 258967539 .

Frantar, E. and Alistarh, D. Qmoe: Practical sub-1-bit compression of trillion-parameter models. arXiv preprint arXiv:2310.16795, 2023.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pretrained transformers. arXiv preprint arXiv:2210.17323, 2022a.

Frantar, E., Singh, S. P., and Alistarh, D. Optimal Brain Compression: A framework for accurate posttraining quantization and pruning. arXiv preprint arXiv:2208.11580, 2022b. Accepted to NeurIPS 2022, to appear.

Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for fewshot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628.

Ge, T., He, K., Ke, Q., and Sun, J. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4):744-755, 2013.

Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., and Keutzer, K. A survey of quantization methods for efficient neural network inference. arXiv preprint arXiv:2103.13630, 2021.

Gray, R. Vector quantization. IEEE ASSP Magazine, 1(2): 4-29, 1984. doi: 10.1109/MASSP.1984.1162229.

Guo, R., Kumar, S., Choromanski, K., and Simcha, D. Quantization based fast inner product search. In Artificial intelligence and statistics, pp. 482-490. PMLR, 2016.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network, 2015.

Jegou, H., Douze, M., and Schmid, C. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128, 2010.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Hanna, E. B., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M. W., and Keutzer, K. Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629, 2023.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR), 2015.

Kurtic, E., Kuznedelev, D., Frantar, E., Goin, M., and Alistarh, D. Sparse fine-tuning for inference acceleration of large language models, 2023.

Li, Z., Ni, B., Zhang, W., Yang, X., and Gao, W. Performance guaranteed network acceleration via high-order residual quantization, 2017.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978, 2023.

Martinez, J., Clement, J., Hoos, H. H., and Little, J. J. Revisiting additive quantization. In Computer VisionECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pp. 137-153. Springer, 2016.

Martinez, J., Zakhmi, S., Hoos, H. H., and Little, J. J. Lsq++: Lower running time and higher recall in multi-codebook quantization. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 491-506, 2018.

McCarter, C. and Dronen, N. Look-ups are not (yet) all you need for deep learning inference. ArXiv, abs/2207.05808, 2022. URL https: //api.semanticscholar.org/CorpusID: 250491319 .

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., and Blankevoort, T. Up or down? Adaptive rounding for post-training quantization. In International Conference on Machine Learning (ICML), 2020.

Norouzi, M. and Fleet, D. J. Cartesian k-means. In Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, pp. 3017-3024, 2013.

Ozan, E. C., Kiranyaz, S., and Gabbouj, M. Competitive quantization for approximate nearest neighbor search. IEEE Transactions on Knowledge and Data Engineering, 28(11):2884-2894, 2016. doi: 10.1109/ TKDE.2016.2597834.

Park, G., Park, B., Kwon, S. J., Kim, B., Lee, Y., and Lee, D. nuQmm: Quantized matmul for efficient inference of large-scale generative language models. arXiv preprint arXiv:2206.09557, 2022.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An imperative style, high-performance deep learning library. In Conference on Neural Information Processing Systems (NeurIPS). 2019.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21 (140):1-67, 2020.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99-106, 2021. doi: 10.1145/3474381. URL https://doi.org/ $10.1145 / 3474381$.

Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., Gallé, M., et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

Sun, S., Cheng, Y., Gan, Z., and Liu, J. Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355, 2019.

Tata, S. and Patel, J. M. PiQA: An algebra for querying protein data sets. In International Conference on Scientific and Statistical Database Management, 2003.

TII UAE. The Falcon family of large language models. https://huggingface.co/tiiuae/ falcon-40b, May 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Tseng, A., Chee, J., Sun, Q., Kuleshov, V., and Sa, C. D. Quip\#: Quip with lattice codebooks, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.
Xiao, G., Lin, J., Seznec, M., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438, 2022.

Yao, Z., Aminabadi, R. Y., Zhang, M., Wu, X., Li, C., and He, Y. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. arXiv preprint arXiv:2206.01861, 2022.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Korhonen, A., Traum, D. R., and Màrquez, L. (eds.), Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 4791-4800. Association for Computational Linguistics, 2019. doi: 10.18653/v1/p19-1472. URL https:// doi.org/10.18653/v1/p19-1472.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Zhang, T., Du, C., and Wang, J. Composite quantization for approximate nearest neighbor search. In International Conference on Machine Learning, pp. 838-846. PMLR, 2014.

Zhou, S.-C., Wang, Y.-Z., Wen, H., He, Q.-Y., and Zou, Y.-H. Balanced quantization: An effective and efficient approach to quantized neural networks. Journal of Computer Science and Technology, 32(4):667-682, Jul 2017. ISSN 1860-4749. doi: 10.1007/s11390-017-1750-y. URL https://doi.org/10.1007/s11390-017$1750-\mathrm{y}$.
</end of paper 0>


<paper 1>
# LoQT: Low Rank Adapters for Quantized Training 

Sebastian Loeschcke<br>University of Copenhagen<br>IT University of Copenhagen<br>sbl@di.ku.dk

() https://github.com/sebulo/LoQT

Serge Belongie<br>University of Copenhagen<br>s.belongie@di.ku.dk

Mads Toftrup*<br>Aarhus University<br>toftrup@cs.au.dk


#### Abstract

Training of large neural networks requires significant computational resources. Despite advances using low-rank adapters and quantization, pretraining of models such as LLMs on consumer hardware has not been possible without model sharding, offloading during training, or per-layer gradient updates. To address these limitations, we propose LoQT, a method for efficiently training quantized models. LoQT uses gradient-based tensor factorization to initialize low-rank trainable weight matrices that are periodically merged into quantized full-rank weight matrices. Our approach is suitable for both pretraining and fine-tuning models, which we demonstrate experimentally for language modeling and downstream task adaptation. We find that LoQT enables efficient training of models up to 7B parameters on a consumer-grade 24GB GPU. We also demonstrate the feasibility of training a 13B parameter model using per-layer gradient updates on the same hardware.


## 1 Introduction

Training large neural networks requires substantial hardware and energy resources. Reducing these requirements is thus important for cost efficiency and environmental sustainability, while also lowering the entry barrier for researchers and practitioners. The main barriers in training large models are the compute operations required, as well as the memory needed to store those computations, in this paper we focus on the latter. Memory use during training comes primarily from the weights of the model itself as well as the optimizer states used to train the model. To target the weights, variations on lowrank adaptation (LoRA) [16, 12, 6, 22, 23] have been suggested to decrease the number of trainable parameters, in combination with the use of low precision representations. To target the optimizer, low-rank approaches for projecting gradients to a lower rank have been used [44]. Finally, various[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_10a27cf460dde279b692g-01.jpg?height=420&width=593&top_left_y=1752&top_left_x=1102)

Figure 1: Memory usage of Llama 13B, rank 1024. PL: per-layer gradient updates. A8bit: Adam 8-bit
![](https://cdn.mathpix.com/cropped/2024_06_04_10a27cf460dde279b692g-02.jpg?height=432&width=1128&top_left_y=236&top_left_x=496)

Figure 2: Overview of LoQT. (1) Low-rank factors $P$ and $B$ are periodically initialized from the gradient of the dequantized model weights $\nabla W$, (2) then only $B$ is trained while $P_{q}$ and $W_{q}$ are kept quantized and frozen, over an exponentially increasing interval until $T_{i}$, (3) the low-rank factors are merged back into the quantized model. The process is repeated until training halts.

applications of quantization [10, 26, 6] have been used to decrease memory requirements. In this work, we combine these approaches into a highly memory-efficient training configuration.

In typical training setups, the optimizer states take up larger space than the model itself, as methods such as Adam [18] need to keep track of two parameters for each weight of the model. GaLore [44] significantly reduces the number of parameters needed for storing the optimizer states by only keeping track of the optimizer state in a low-rank projection which is then projected up to be applied to the model weights. Combining this method with quantization would further shrink the footprint of the model. However, updating the weights of a highly quantized model directly in low-precision space has not been shown to work. This is mainly due to the higher-precision gradient updates being too small to have an impact on the lower-precision quantized states. Lastly, while LoRA is memory efficient for parameter-efficient fine-tuning of pre-trained models, it does not work as a pretraining method by itself [22].

To address these shortcomings, we propose a new method, LoQT. LoQT initializes two low-rank factors for each weight matrix $W: 1$ ) $P$, using a projection of $W$ 's gradients into a low-rank subspace, and 2) $B$, initialized to minimize the error of quantizing and represents the only low-rank matrix being trained. Only training $B$ means that the optimizer state can be shrunk significantly. The product $P B$ is periodically merged into the full rank matrix $W$ with exponentially increasing scheduling. As $W$ and $P$ do not receive gradient updates, they can be quantized, thus optimizing memory usage even further. We stress that it is the large accumulated updates that make it possible to update a quantized model as the addition of smaller changes would not register in the quantized state. A high-level overview is given in Fig. 2 .

We show that LoQT works well with and without a quantized model, enabling not only a lower memory footprint in the optimizer state but also over the model parameters. Our results show that we get competitive performance to prior methods using significantly less memory, in particular when using quantization of the model weights in an application such as training an LLM. We also demonstrate superior performance when fine-tuning pre-trained models, by training and evaluating on the GLUE [36] benchmark for natural language understanding. Finally, we ablate several properties of the suggested approach and we find that an exponentially increasing projection gap is beneficial, not only to our work but also for GaLore. This is particularly crucial for the training of quantized models. LoQT enables efficient training of 7B models on consumer-grade hardware with 24GB of memory, and makes it feasible to train models with up to 13 billion parameters without model parallel by making use of per-layer gradient updates [25] on the same hardware as shown in Fig. 11.

## 2 Related Work and Background

### 2.1 Neural Network Quantization and NF4

Quantization compresses neural networks by converting high-precision values into lower-precision formats, significantly reducing storage requirements [43, 33, 1, 4]. The process involves taking a datatype of high precision, such as 32 -bit, requiring 4 bytes of memory, and converting it into a
representation with increasing rounding errors but lower memory cost. In this work, we use NF4 quantization [6], since it is a 4-bit code it only contains $2^{4}$ different values. NF4 works by first normalizing values onto the interval $[-1: 1]$, these are then discretized onto quantiles of the normal distribution, $\left(q_{i}\right)_{i=1}^{16}$ (see [6] for details). The elements of a layer are divided into blocks of 64 weights. Each block $\beta$ has a scaling factor $\mathcal{M}_{\beta}=\max _{w \in \beta}\left|w_{32}\right|$.

$$
\begin{align*}
w_{\mathrm{NF} 4} & =q_{\mathrm{NF} 4}\left(w, \mathcal{M}_{\beta}\right):=\operatorname{argmin}_{q_{i}}\left|w / \mathcal{M}_{\beta}-q_{i}\right|  \tag{1}\\
w & =q_{\mathrm{NF} 4}^{-1}\left(w_{\mathrm{NF} 4}, \mathcal{M}_{\beta}\right):=\mathcal{M}_{\beta} \cdot w_{\mathrm{NF} 4} \tag{2}
\end{align*}
$$

We provide an overview of different categories of quantization techniques, and how they relate to LoQT, in Appendix A Compared to prior approaches, LoQT retains the benefits of reduced memory usage while minimizing accuracy loss, using high-precision updates on a low-rank representation. This allows for efficient model updates without the overhead of full matrix storage and re-quantization.

### 2.2 Adaptation of Pretrained Networks

Low-Rank Adaptation (LoRA) [16] enables fine-tuning of pre-trained models using low-rank adaptors, effectively reducing the memory footprint by only training weight adaptors for targeted layers. However, simple low-rank training using LoRA factor matrices has not been shown to work for pre-training [22].

LoRA employs trainable low-rank matrices $A$ and $B$ that are used to update $W$ following $W_{t}=$ $W_{t-1}+A B$, where $W_{t-1}$ is frozen to enable precise adjustments within a low-rank framework. Since LoRA only trains $A$ and $B$ and keeps $W$ fixed, QLoRA [16] explore quantizing $W$. They fine-tune a quantized model $q(W)=W_{q}$ with 4-bit precision using randomly initialized 16-bit precision factors $A$ and $B$. To address quantization errors $\mathcal{E}=\left|W_{q}-W\right|$, low-rank factors of the quantization error $\mathcal{E}$ have been used [21].

LoQT extends LoRA to both pretraining and fine-tuning. Unlike traditional LoRA, LoQT uses $A$ and $B$ to refine $W$ throughout training, with $A$ initialized from $W$ 's gradient projection and $B$ trained along this gradient path. LoQT also incorporates quantization and targeted optimization iterations similar in spirit to LoftQ [21], correcting for quantization errors in $W_{q}$, thus better aligning it with the original non-quantized $W$.

### 2.3 Memory Efficient Optimization

Optimizer memory consumption A significant portion of the memory needed to train neural networks is typically consumed by optimizer states. Notably, Adam [18], one of the most widely used optimizers, uses double the amount of memory as the gradient matrix to maintain first and second-order gradient statistics. Efforts to reduce this overhead have led to the development of adaptive optimization algorithms like Adafactor [32], which achieves sub-linear memory costs by factorizing the second-order statistics into a row-column outer product. GaLore [44] expands on this concept by using low-rank factorization and projecting low-rank gradients up to a full-rank when updating model weights.

Periodic updating of weight matrices ReLoRA [22] combines low-rank updates with initial full-rank training. They find that doing one-third of the training in full-rank, and the subsequent two-thirds in low-rank (see $\$ 2.2$ ) results in comparable performance to standard training methods.

Low-rank gradients GaLore [44], focuses on the structure of the gradients, projecting them into a low-rank space using factors $P$ and $Q$, which are derived from a truncated singular value decomposition (SVD) of the weight matrix gradient, $G_{W} \approx P_{r} \Sigma_{r} Q_{r}$. This reduces memory costs associated with storing the optimizer states and aligns with findings from recent studies which suggest that learning primarily occurs within a low-dimensional subspace at a given time [19, 11]. This can be further combined with applying per-layer gradient updates, reducing the memory needed for storing the gradients for the full model at once [25].

LoQT builds on GaLore's gradient projection ( $\$ 3.1$ to initialize LoRA factors while updating the full matrix following a schedule inspired by ReLora, while only training one low-rank matrix per layer. We achieve comparable quality to GaLore and better performance than ReLoRA while reducing tunable parameters and memory usage compared to both approaches.

## 3 Efficient Pretraining With LoQT

LoQT works by initializing and training low-rank adapters obtained by taking the SVD of a given layer's gradients. Let $W$ indicate the full weights matrix of a given layer, $P$ be the left factor constructed from the SVD decomposition of the gradients matrix: $\nabla W=U \Sigma V^{\top}$; i.e. $P$ consists of the first $r$ columns of $U$ corresponding to the singular vectors with the $r$ largest singular values of $W$. The update rule for an interval $\left[T_{i-1}, T_{i}\right]$ is then given by $W_{T_{i}}=W_{T_{i-1}}+P B$, where only the weights of $B$ are updated. $P$ and $W_{T_{i-1}}$ are kept constant over the time interval. We describe this in more detail below, followed by a discussion on periodic updating of the factor $P$, the enabling of quantized pre-training, error compensation, and exponential update intervals. Pseudo-code for LoQT is shown in Fig. 3 .

### 3.1 Background: GaLore

Zhao et al. [44] show that gradients exhibit a low-rank structure during training. They exploit this insight by projecting the gradient to a low-rank subspace and applying the Adam optimizer before projecting back to the original dimensions. By doing this, the memory-intensive optimizer states required by Adam are shrunk significantly for low enough ranks.

Definition 3.1 (Gradient Low-rank Projection, def. 3.4 in [44]). Gradient low-rank projection (GaLore) denotes the following gradient update rules, where $\eta$ is the learning rate, $\rho$ is the Adam optimizer, and $W \in R^{m \times n}$ is the weight matrix being trained, and $T$ represents the total number of training iterations between recomputing the projection matrices:

$$
W_{T}=W_{0}+\eta \sum_{t=0}^{T-1} \tilde{G}_{t}, \text { where } \quad \tilde{G}_{t}=P_{t} \rho_{t}\left(P_{t}^{\top} G_{t} Q_{t}\right) Q_{t}^{\top}
$$

where $P_{t} \in R^{m \times r}$ and $Q_{t} \in R^{n \times r}$ are are the top-r singular vectors from the SVD decomposition of the gradient matrix at each iteration $t$. In practice, this can be approximated by only applying a one-sided projection, as in

$$
W_{T}^{\prime}=W_{0}+\eta \sum_{t=0}^{T-1} P_{t} \rho_{t}\left(P_{t}^{\top} G_{t}\right) \quad \text { or } \quad W_{T}^{\prime}=W_{0}+\eta \sum_{t=0}^{T-1} \rho_{t}\left(G_{t} Q_{t}\right) Q_{t}^{\top}
$$

GaLore demonstrates that it is sufficient to keep the projection matrix fixed and only update it once every $T$ iterations, which we use in the following.

### 3.2 Low-rank Gradients as Adapters

We now describe the process by which we initialize the parameters we optimize in LoQT. We adopt the memory-performance trade-off of using only a one-sided projection. We compute $P^{\top} G$ if $m \leq n$ and $G Q$ otherwise. We want to achieve a separation between trainable weights and static weights, which we achieve by rewriting GaLore in terms of low-rank adaptors. Assume without loss of generality that $m \leq n$. Using the fact that $P_{t}$ is fixed in the interval $[0, T]$ we have that

$$
\begin{align*}
W_{T} & =W_{0}+\eta \sum_{t=0}^{T-1} P \rho_{t}\left(P^{\top} G_{t}\right)  \tag{3}\\
& =W_{0}+\eta \underbrace{P}_{\in \mathbb{R}^{m \times r}} \underbrace{\sum_{t=0}^{T-1} \rho\left(P^{\top} G_{t}\right)}_{B \in \mathbb{R}^{r \times n}} \tag{4}
\end{align*}
$$

From (4) it is clear that we can keep track of low-rank updates using rank- $r$ low-rank adaptors. We note that in the interval $[0, T]$ only $B$ is updated, creating the desired separation. If implemented directly, we would need to compute the gradient with respect to $W$ and then project it down using $P^{\top} G_{t}$. We find that this step is unnecessary; it is sufficient to train $B$ using standard gradient descent.

We now show that training the $B$ matrix using gradient descent is equivalent to training w.r.t. $W_{t}$ as in definition 3.1 Let $G^{W}$ indicate the gradient of the loss with respect to $W$, and $G^{B}$ for the gradient
of the loss with respect to $B$. Given a weight matrix $W$, an factor $P$ and a matrix $B$, when computing the forward pass $y=x W+x P B$, the gradient of a loss function $\mathcal{L}$ w.r.t. $B$ is $G^{B}=P^{\top} G^{W}$. This can be seen by applying the chain rule to get $G^{W}=x^{\top} \frac{\partial \mathcal{L}}{\partial y}$. The vector multiplied onto $B$ is $x P$ giving $G^{B}=(x P)^{\top} \frac{\partial \mathcal{L}}{\partial y}=P^{\top} x^{\top} \frac{\partial L}{\partial y}=P_{t}^{\top} G^{w}$. This shows that calculating the gradient w.r.t $B$ gives the same as projecting the gradient w.r.t $W$. It is thus clear that GaLore's low-rank gradient updates should be the same as those obtained using backpropagation through LoRA.

### 3.3 Enabling pretraining with LoRA

Previous work has shown that training low-rank weight matrices works well for fine-tuning pre-trained weights. However, it has been shown that training low-rank factors, and periodically merging them into frozen $W$, does not work when starting with a randomly initialized matrix [22]. Here we address this shortcoming to enable full training using low-rank weight matrices.

Inspired by prior work [22, 44], we periodically update a given layer $W_{T+1}=W_{T}+P_{T} B_{T}$ at fixed steps $T \in \mathcal{T}$. This approach allows $W$ to evolve as a sum of low-rank matrices aligning with GaLore's strategy of updating the gradient subspace during training:

$$
W_{t}=W_{0}+\Delta W_{T_{1}}+\Delta W_{T_{2}}+\ldots+\Delta W_{T_{n}}
$$

where $t=\sum_{i=1}^{|\mathcal{T}|} T_{i}$ and $\Delta W_{T_{i}}=P_{T_{i}} B_{T_{i}}$ represents the product of the learning from $B$ during the interval $T_{i}-T_{i-1}$ scaled by the learning rate $\eta$ and modulated by the gradient projection matrix $P_{T_{i}}$. After each update at iteration $T_{i} \in \mathcal{T}$, we reinitialize the low-rank factors $P_{T}$ and $B_{T}$. As in [44], we compute the gradient of $W_{T}$ over a single batch, focusing only on $\nabla W_{T}$ without needing full optimizer states. Not requiring optimizer states reduces the memory increase compared to full-rank training.

With the updated $W_{t}$ and reinitialized $P_{t}$ and $B_{t}$, a new gradient subspace is established for exploring the next $T_{i+1}-T_{i}$ steps. Our method treats $W_{t}$ as the full-rank repository of accumulated updates. Although it is periodically updated, $W_{t}$ is not part of the optimizer state computations and the gradients during the single forward pass are offloaded to cpu. Since the SVD calculations are done layerwise only the current layer is needed on GPU, or the SVD can be calculated on CPU. $P_{t}$ defines the general gradient subspace and trajectory for the upcoming $T_{i+1}-T_{i}$ steps, and $B_{t}$ is adjusted to navigate optimally within the direction set by $P_{t}$. As only $B_{t}$ is trained, the number of parameters needing optimizer states is drastically reduced.

### 3.4 Quantized Training

Given that $B$ is the only matrix accumulating gradients and undergoing changes, the other matrices $W$ and $P$ can be kept quantized. This approach allows storing the weights in NF4 precision without requiring high-precision gradient and weights to update $W$ and $P$. To the best of our knowledge, we are the first to enable efficient 4-bit quantized training using gradient descent without storing the weights in full precision.

We quantize weights $q_{\mathrm{NF} 4}(W)=W_{q}$ and $q_{\mathrm{NF} 4}(P)=P_{q}$ as described in $\$ 2.1$. During periodic updates at interval time steps $\left(\sum_{i=1}^{n} T_{i}\right)_{n=1}^{\max }, P_{q}$ and $W_{q}$ are dequantized using the inverse function, $P_{\mathrm{BF} 16}=q_{\mathrm{NF} 4}^{-1}\left(P_{\mathrm{NF} 4}\right)$ and $W_{B F 16}=q_{\mathrm{NF} 4}^{-1}\left(W_{\mathrm{NF} 4}\right)$. After this, $W_{T_{i}}=W_{T_{i-1}}+P_{T_{i-1}} B_{T_{i-1}}$ is computed and quantized. The quantization and dequantization processes are applied layer by layer, ensuring that not all layers are simultaneously in a non-quantized state to reduce memory usage. Moreover, the quantization state itself is re-quantized for further efficiency following [6]. We implement LoQT using weight-only quantization, meaning the quantized weights are loaded into memory and then dequantized before computing the matrix multiplications.

### 3.5 Compensating for Errors Introduced by Quantization

As the quantization process inevitably leads to a discrepancy between the non-quantized and quantized versions of $W$ we wish to reduce this effect as much as possible. While compensating for quantization errors has been done before [21], we need a tailored solution for LoQT.

During the merging phase, we first dequantize to obtain $W_{T-1}$ and $P_{T-1}$, and then compute the update $W_{T}=W_{T-1}+P_{T-1} B_{T-1}$. This is immediately followed by re-quantizing to get $Q_{T}=q_{\mathrm{NF} 4}\left(W_{T}\right)$.

```
Algorithm 1 LoQT: Low Rank Adapters for
Quantized Training
Require: $W$ : Weight, $T$ : Update steps, $\eta$ :
    $\mathrm{LR}, r$ : rank, $q_{N}(\cdot)$ and $d e q_{N}(\cdot)$ : N-bit
    quant and dequant functions.
    $G_{W} \leftarrow \nabla_{W} \mathcal{L}(W)$
    $W_{Q}, P_{Q}, B \leftarrow \operatorname{Initialize}\left(W, G_{W}\right)$
    for each $t$ in training steps do
        if $t \in T$ then
            $W \leftarrow W_{Q}+s \cdot P_{Q} \cdot B_{t}$
            $G^{W} \leftarrow \nabla_{W} \mathcal{L}(W)$
            $W_{Q}, P_{Q}, B_{t} \leftarrow \operatorname{Initialize}\left(W, G^{W}\right)$
        else
            $B_{t+1} \leftarrow B_{t}-\rho\left(G_{t}^{B}\right)$
    return $\theta$
```

```
Algorithm 2 Initialization Procedure
    Initialize $\left(W, G^{W}\right)$ :
    $U, S, V^{T} \leftarrow \operatorname{SVD}\left(G^{W}\right)$
    $P \leftarrow U[:,: r]$ \{First $r$ singular vectors \}
    $P_{q} \leftarrow q_{N}(P)$
    $B \leftarrow 0$
    $\hat{W} \leftarrow W$
    for each $c$ in compensation steps $C$ do
        $Q_{c} \leftarrow q_{N}(\hat{W})$
        $B \leftarrow P^{+}\left(\hat{W}-Q_{c}\right)$
        $\hat{W} \leftarrow W-P B$
    return $Q_{c}, B, P_{q}$
```

Figure 3: Pseudo-code for LoQT.

Table 1: Comparison of low-rank pre-training methods for LLaMA2-style language models on the C4 dataset. The table shows validation perplexity, memory estimates, and quantization states for LoQT. The rank ratio $r / d_{\text {model }}$ is relative to the largest weight matrix dimension. Perplexity values are averaged over three seeds showing mean and standard error. $\left({ }^{*}\right)$ Denotes results from GaLore [44]. Only one seed was used for the 1B experiment due to compute constraints.

|  | $\mathbf{6 0 M}$ | $\mathbf{1 3 0 M}$ | $\mathbf{3 5 0 M}$ | $\mathbf{1 B}$ |
| :--- | :---: | :---: | :---: | :---: |
| Full | $33.32 \pm 0.22(0.36 \mathrm{G})$ | $24.51 \pm 0.03(0.76 \mathrm{G})$ | $18.87 \pm 0.18(2.06 \mathrm{G})$ | $15.56^{*}(7.80 \mathrm{G})$ |
| LoQT (Ours) | $33.98 \pm 0.15(0.23 \mathrm{G})$ | $24.57 \pm 0.01(0.49 \mathrm{G})$ | $19.12 \pm 0.01(0.98 \mathrm{G})$ | $15.55(3.16 \mathrm{G})$ |
| LoQT-nq (No quant.) | $33.55 \pm 0.03(0.28 \mathrm{G})$ | $24.37 \pm 0.02(0.63 \mathrm{G})$ | $18.85 \pm 0.01(1.47 \mathrm{G})$ | $15.20(5.11 \mathrm{G})$ |
| GaLore | $34.15 \pm 0.24(0.24 \mathrm{G})$ | $24.81 \pm 0.04(0.52 \mathrm{G})$ | $19.47 \pm 0.01(1.22 \mathrm{G})$ | $15.64^{*}(4.38 \mathrm{G})$ |
| LoRA | $34.99^{*}(0.36 \mathrm{G})$ | $33.92^{*}(0.80 \mathrm{G})$ | $25.58 *(1.76 \mathrm{G})$ | $19.21^{*}(6.17 \mathrm{G})$ |
| ReLoRA | $37.04^{*}(0.36 \mathrm{G})$ | $29.37^{(0.80 G)}$ | $29.08^{*}(1.76 \mathrm{G})$ | $18.33^{*}(6.17 \mathrm{G})$ |
| $r / d_{\text {model }}$ | $128 / 256$ | $256 / 768$ | $256 / 1024$ | $512 / 2048$ |
| Training Tokens | $1.1 \mathrm{~B}$ | $2.2 \mathrm{~B}$ | $6.4 \mathrm{~B}$ | $13.1 \mathrm{~B}$ |

Our goal is thus to minimize the quantization error $\left\|\left(Q_{T}+P_{T} B_{T}\right)-W_{T}\right\|$. To achieve this, we solve for $B_{T}$ in the merging step, initializing $B_{T}$ as $B_{T}:=P_{T}^{+}\left(Q_{T}-W_{T}\right)$, where $P_{T}^{+}$is the Moore-Penrose pseudo-inverse. This approach avoids initializing $B_{T}$ as zeros and instead uses it for minimizing the quantization error $\left\|Q_{T}-W_{T}\right\|$. We then iteratively refine $B_{T}$, recomputing $Q_{T}=q_{\mathrm{NF} 4}\left(W_{T}-P_{T} B_{T}\right)$, improving the alignment between the full-precision $W$ and its quantized state.

As training advances and the learning rate decays, the magnitude of the update $B_{T-1}$ to form $W_{T}$ decreases. This leads to negligible differences between $\left|q\left(Q_{t}+P_{t} B_{t}\right)-Q_{t}\right|$, which results in the weights plateauing early, as depicted in Fig. 4a. To address this, we implement an exponentially increasing scheduler for updating $W$. Drawing from GaLore's observation on the exponential decay of gradient rank (Lemma 3.1 [44]), we start with a frequency gap $\tau$ and progressively increase the update intervals by a factor of $\psi$. The sequence of updates is then given by $\left(T_{i}\right)_{i=0}^{\infty}=\left(\tau+\psi^{i}\right)_{i=0}^{\infty}$ Each $T_{i}$ marks a training step $t$ when $W$ is updated. This scheduling ensures more frequent updates earlier in training and more well-spaced adjustments later, allowing more accumulated gradients before each update.

## 4 Experiments and Results

### 4.1 Experimental Setup

We evaluate LoQT by training LLaMA-based[34] language models on the C4 dataset [30], a collection of processed text in English that was scraped from the internet [30]. We train models of sizes of $60 \mathrm{M}, 130 \mathrm{M}, 350 \mathrm{M}$, and 1B parameters, adhering to single-epoch training cycles determined by Chinchilla Scaling Laws [15]. While LoQT is capable of training models up to 13 billion parameters on consumer GPUs, compute limits prevent us from training to convergence for sizes above 1B. We also benchmark LoQT on the GLUE test-suite for natural language understanding [37]. Runs were conducted on up to $4 \mathrm{x}$ 40GB NVIDIA A100s $2 \mathrm{x}$ 80GB NVIDIA H100s, or a single 24GB NVIDIA RTX 3090. The longest run was the training of the 1B models, taking approximately four days on the four A100s. The 3090 was used for throughput and to empirically verify memory claims.

Hyperparameters are consistent across model sizes, with experiments in BF16 format for memory efficiency. All models use a maximum sequence length of 256 , a total token batch size of $131 \mathrm{~K}$ tokens, and a learning rate warmup for the first $10 \%$ of the training steps, followed by cosine annealing to $10 \%$ of the initial learning rate. Full experimental details, including the specific hyperparameters for each task, are provided in Appendix B.

Baselines For pre-training, we compare LoQT against LoRA [16], ReLoRA [22], GaLore [44], and non-quantized version of LoQT, LoQT-nq. In our experiments, LoQT, LoRA, and ReLoRA modify attention and fully connected layers while maintaining full-rank embeddings and normalization layers. This contrasts with GaLore, which keeps weights full-rank but projects gradients to low-rank, and standard full-rank training. For fine-tuning, we benchmark LoQT against GaLore, LoftQ [21], LoRA and LoQT-nq. All models use identical update frequencies for GaLore, ReLoRA, LoQT-nq, and LoQT, starting with an update frequency of $T=100$ and then with exponentially increasing update frequencies. This means that we do more frequent updates early and fewer as the model stabilizes (see Section 4b)for more details). All models are trained using the Adam optimizer, except GaLore which uses GaLoreAdam for gradient projection.

### 4.2 Pre-training of Generative Language Models

Results and details for pretraining of language models of sizes $60 \mathrm{M}, 130 \mathrm{M}, 350 \mathrm{M}$ and 1B parameters are shown in Table 1. Model sizes are calculated based on the full models without any low-rank methods. We see that LoQT and LoQT-nq both perform very close to full rank pretraining and GaLore, while using significantly less memory by keeping most of the model weights in a quantized state. For the 60M model, full training is only slightly better than LoQT, while we see results improve or be within the standard error for the other sizes. We also notice a slight drop in performance from quantizing the original weight matrix, comparing LoQT and LoQT-nq. The key difference between the approaches are the theoretical memory estimaes, e.g. where LoQT requires $59 \%$ less memory for the 1B model in full precision and 28\% less memory than GaLore.

Table 2: Results with LoQT, LoQT-nq, and GaLore of DeBERTaV3-base models on the GLUE development set. We report mean and standard error over three seeds. The best results on each dataset are shown in bold.

| Rank | Method | MNLI <br> Acc | QNLI <br> Acc | RTE <br> Acc | SST <br> Acc | MRPC <br> f1 | CoLA <br> Matt | QQP <br> f1 | STSB <br> PCorr | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 32 | LoQT-nq | $90.0 \pm 0.10$ | $94.2 \pm 0.06$ | $\mathbf{8 4 . 8} \pm \mathbf{0 . 7 5}$ | $\mathbf{9 5 . 9} \pm \mathbf{0 . 0 6}$ | $94.1 \pm 0.25$ | $\mathbf{7 2 . 5} \pm \mathbf{0 . 4 1}$ | $\mathbf{9 0 . 0} \pm \mathbf{0 . 0 6}$ | $91.5 \pm 0.07$ | $\mathbf{8 9 . 1}$ |
| 32 | LoQT | $90.0 \pm 0.09$ | $\mathbf{9 4 . 3} \pm \mathbf{0 . 0 4}$ | $84.1 \pm 0.91$ | $95.5 \pm 0.10$ | $\mathbf{9 4 . 4} \pm \mathbf{0 . 2 0}$ | $70.5 \pm 0.35$ | $89.2 \pm 0.02$ | $\mathbf{9 1 . 5} \pm \mathbf{0 . 1 3}$ | 88.7 |
| 32 | LoRA | $89.9 \pm 0.03$ | $94.0 \pm 0.09$ | $83.6 \pm 0.12$ | $95.7 \pm 0.15$ | $93.5 \pm 0.26$ | $69.3 \pm 0.47$ | $89.8 \pm 0.11$ | $90.7 \pm 0.22$ | 88.3 |
| 32 | LoftQ | $90.4 \pm 0.09$ | $93.2 \pm 0.02$ | $83.8 \pm 0.63$ | $94.7 *$ | $93.2 \pm 0.14$ | $71.1 \pm 0.28$ | $89.6 \pm 0.12$ | $91.0 \pm 0.09$ | 88.4 |
| 32 | GaLore | $\mathbf{9 0 . 3} \pm \mathbf{0 . 0 7}$ | $94.0 \pm 0.04$ | $83.7 \pm 0.79$ | $95.6 \pm 0.07$ | $93.4 \pm 0.38$ | $70.7 \pm 0.24$ | $89.8 \pm 0.05$ | $90.6 \pm 0.01$ | 88.5 |

Table 3: Comparison of memory usage between GaLore, LoRA, and LoQT. $W \in \mathbb{R}^{m \times n}(m \leq n)$, rank $r$.

|  | GaLore | LoRA | LoQT (Ours) |
| :--- | :---: | :---: | :---: |
| Weights | $m n$ | $m n+m r+n r$ | $m n+m r+n r$ |
| Optimizer States | $m r+2 n r$ | $2 m r+2 n r$ | $2 n r$ |
| Gradients | $m n$ | $m r+n r$ | $n r$ |
| Pretraining | Yes | No | Yes |
| Fine-Tuning | Yes | Yes | Yes |
| Quantizeable | No | Yes | Yes |

### 4.3 Memory-efficient finetuning

We fine-tune the pre-trained DeBERTa-V3-base 2 [13] model on GLUE tasks using LoQT and compare its performance with a full fine-tuning baseline, LoRA, LoftQ, and GaLore. See Appendix 5 for details on hyperparameters. Results are given in Table 2 .

We find that both LoQT-nq and LoQT perform well. And somewhat surprisingly, it sometimes surpasses GaLore, LoftQ, and LoRA.This may indicate that initializing the LoRA factors with information about the gradient of $W$ could be a beneficial starting point compared to standard initialization methods. As the goal of this work is to limit memory consumption, we leave out further comparisons that could verify these findings to future work.

### 4.4 Memory and Throughput

Memory usage An overview of memory usage for GaLore, LoRA and LoQT is given in Table 3 We see that LoQT makes use of the same number of trainable parameters as LoRA for a given rank while using less memory for the optimizer states and gradients than in both LoRA and GaLore.

We compare the LoQT to the closest in memory performance approach, GaLore, for 13B in Figure 1 . and for other model-sizes in Figure 6. We compare three different use cases, using the approaches directly, combining them with an 8-bit Adam optimizer [5], and using per-layer weight updates with offloading (while still using 8-bit Adam). We see from the figures that LoQT significantly shrinks both the number of trainable parameters and optimizer states compared to GaLore.

Per-layer weight update is essential for GaLore; without it, an additional $\sim 12$ GB of VRAM is needed for gradients in a 7B model, making full-parameter fine-tuning impossible on a 24GB GPU. Additionally, the per-layer gradient updates may not work well with DDP (Distributed Data Parallel) and gradient accumulation. With our method, we can get a lower memory than GaLore even when they use per-layer gradient updates. When not using per-layer gradient updates, this difference becomes even bigger as seen for the 7B model in Figure 6.

Moreover, our method supports training 7B models without per-layer computations on 24GB GPU. This makes it possible to use multi-GPU training, a capability not possible with the current GaLore approach. Our memory advantage allows for a batch size of 1280 tokens compared to GaLore's 256 for the 7B model on the 24GB RTX3090. With per-layer gradient updates, LoQT can train a 13B model on a single GPU, pushing the limits of hardware efficiency.

Throughput We evaluate the throughput with a sample batch size of 16 with a total batch size of 512 using gradient accumulation, which is the largest power of 2 that fits on the GPU. We update the projection matrix $P$ for every 200 iterations. The per-layer gradient update algorithms apply a weight update for every mini-batch as they do not support gradient accumulation. For the evaluation, we use a 1B parameter model with rank 512. We find that LoQT can process $16 \%$ fewer tokens per second than only using AdamW, at 3996 tokens/s compared to 4782 tokens/s on the RTX3090.

![](https://cdn.mathpix.com/cropped/2024_06_04_10a27cf460dde279b692g-09.jpg?height=672&width=1423&top_left_y=228&top_left_x=362)

![](https://cdn.mathpix.com/cropped/2024_06_04_10a27cf460dde279b692g-09.jpg?height=556&width=680&top_left_y=237&top_left_x=365)

(a) EC: Error compensation, EF: Exp. decreasing update frequency.

![](https://cdn.mathpix.com/cropped/2024_06_04_10a27cf460dde279b692g-09.jpg?height=556&width=697&top_left_y=237&top_left_x=1061)

(b) Ablation for different update frequencies, with exponentially increasing at the bottom

Figure 4: Ablation results for update frequency, error-compensation, quantization, model size 130m, and rank 256 .

## 5 Ablations

Quantization Error Compensation and Initialization To assess the impact of quantization error compensation, we analyze the validation loss curves for a 130 million parameter model. Figure $4 \mathrm{a}$ shows that quantizing $W$ or both $W$ and $P$ without error compensation, or exponential frequency updates, causes the loss to stagnate early. We also note that quantizing $P$ has a much smaller effect on the loss compared to quantizing $W$. Error compensation significantly improves the model's performance, resulting in approximately 3.5 points better perplexity. Adding exponentially increasing update frequency improves perplexity by an additional 1.5 points, achieving performance close to that of models without quantization.

Without the quantization error compensation detailed in $\$ 3.5$ LoQT's performance stagnates earlier and diverges more from the other models. This demonstrates the effectiveness of our compensation approach in mitigating the quantization errors introduced during the $W$ update with $A B$ and subsequent quantization steps.

Projection update frequency Our scheduling approach ensures more frequent updates early in training to facilitate substantial weight adjustments. As training progresses, the update frequency decreases, allowing for the accumulation of larger updates to compensate for smaller updates that might be canceled out by quantization errors. Figure 4b presents an ablation study on our method of progressively increasing update frequency starting at 100 and increasing by a factor of $1.2^{T}$ up to 2500. We show the validation loss curves for fixed update frequencies $200,400,500$, and 1000 .

The results show that exponentially increasing the update gap is particularly beneficial for models employing quantization, enabling them to achieve the same perplexity as those without quantization while making use of GaLore. Conversely, the performance gains are more subtle for models that do not use quantization and rely solely on GaLore. We hypothesize that even these models might benefit from the larger projection gap intervals. This could be due to the reduction in the accumulation of errors from frequent updates of the projection factor $P$, as the influence of outdated optimizer statistics becomes less prevalent. Finally, an ablation on the ranks used for $P$ and $B$ is given in Figure 5 in the Appendix.

## 6 Discussion and Conclusion

We present LoQT, a method for memory-efficient pretraining and adaptation of quantized models. The key insights behind the approach are the benefits of initializing low-rank factors using the[^1]gradient of the weight matrix and using exponentially increasing update gaps that make updating of a quantized model possible. While our initial goal was to lower memory usage to facilitate the training of models such as LLMs on consumer-grade hardware, we are cautiously excited about the results sometimes being better than the baselines. These evaluations will be explored in more detail in future work.

Our method is general and opens up new ways of decreasing memory use as well as improving the training throughput. This could be done by implementing kernel fusion and using other quantization methods such as NF2 [6] or quantization of activations, making it possible to do the matrix multiplications using modern tensor core formats such as FP8 or INT4.

## 7 Impact and Limitations

Our work has the potential to have a significant impact on those working in hardware-constrained settings by enabling more efficient training on consumer hardware. We are particularly excited to see the method being applied in single GPU settings. We validate LoQT on several model sizes, by training over many steps and by fine-tuning on a standard benchmark for natural language understanding. While we are confident in our results, further exploration of training duration, data diversity, and hyper-parameter tuning might lead to different results in those settings.

## 8 Acknowledgements

This work is supported by the Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429) and VILLUM FONDEN (40516). Serge Belongie and Vésteinn Snæbjarnarson are supported by the Pioneer Centre for AI, DNRF grant number P1. MJK acknowledges support from the Carlsberg Foundation and the Novo Nordisk Foundation.

## References

[1] Haoli Bai, Lu Hou, Lifeng Shang, Xin Jiang, Irwin King, and Michael R. Lyu. Towards efficient post-training quantization of pre-trained language models, 2021.

[2] Ron Banner, Itay Hubara, Elad Hoffer, and Daniel Soudry. Scalable methods for 8-bit training of neural networks, 2018.

[3] Brian Chmiel, Ron Banner, Elad Hoffer, Hilla Ben Yaacov, and Daniel Soudry. Logarithmic unbiased quantization: Simple 4-bit training in deep learning, 2022.

[4] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale, 2022.

[5] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization, 2022.

[6] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms, 2023.

[7] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless $11 \mathrm{~m}$ weight compression, 2023.

[8] Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization, 2024.

[9] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023.

[10] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. CoRR, abs/2103.13630, 2021.

[11] Guy Gur-Ari, Daniel A. Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace, 2018.

[12] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. arXiv preprint arXiv:2402.12354, 2024.

[13] Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electrastyle pre-training with gradient-disentangled embedding sharing, 2023.

[14] Jung Hwan Heo, Jeonghoon Kim, Beomseok Kwon, Byeongwook Kim, Se Jung Kwon, and Dongsoo Lee. Rethinking channel dimensions to isolate outliers for low-bit weight quantization of large language models, 2024.

[15] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022.

[16] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

[17] Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Youngjun Kwak, Jae-Joon Han, Sung Ju Hwang, and Changkyu Choi. Learning to quantize deep networks by optimizing quantization intervals with task loss, 2018.

[18] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.

[19] Brett W. Larsen, Stanislav Fort, Nic Becker, and Surya Ganguli. How many degrees of freedom do we need to train deep networks: a loss landscape perspective, 2022.

[20] Jung Hyun Lee, Jeonghoon Kim, Se Jung Kwon, and Dongsoo Lee. FlexRound: Learnable rounding based on element-wise division for post-training quantization. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 18913-18939. PMLR, 23-29 Jul 2023.

[21] Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, and Tuo Zhao. Loftq: Lora-fine-tuning-aware quantization for large language models, 2023.

[22] Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, and Anna Rumshisky. Relora: Highrank training through low-rank updates, 2023.

[23] Baohao Liao and Christof Monz. Apiq: Finetuning of 2-bit quantized large language model, 2024.

[24] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023.

[25] Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo, and Xipeng Qiu. Full parameter fine-tuning for large language models with limited resources, 2023.

[26] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits, 2024.

[27] Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. Lut-gemm: Quantized matrix multiplication based on luts for efficient inference in large-scale generative language models, 2024.

[28] Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, Ruizhe Wang, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, and Peng Cheng. Fp8-lm: Training fp8 large language models, 2023.

[29] Sergio P. Perez, Yan Zhang, James Briggs, Charlie Blake, Josh Levy-Kramer, Paul Balanca, Carlo Luschi, Stephen Barlow, and Andrew William Fitzgibbon. Training and inference of large language models using 8-bit floating point, 2023.

[30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.

[31] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models, 2024.

[32] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost, 2018.

[33] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. Q-bert: Hessian based ultra low precision quantization of bert, 2019.

[34] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[35] Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip : Even better llm quantization with hadamard incoherence and lattice codebooks, 2024.

[36] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Tal Linzen, Grzegorz Chrupała, and Afra Alishahi, editors, Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355, Brussels, Belgium, November 2018. Association for Computational Linguistics.

[37] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding, 2019.

[38] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models, 2023.

[39] Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Training deep neural networks with 8-bit floating point numbers, 2018.

[40] Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, and Ludwig Schmidt. Stable and low-precision training for large-scale vision-language models, 2023.

[41] Haocheng Xi, Yuxiang Chen, Kang Zhao, Kaijun Zheng, Jianfei Chen, and Jun Zhu. Jetfire: Efficient and accurate transformer pretraining with int8 data flow and per-block quantization, 2024.

[42] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models, 2024.

[43] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. Q8bert: Quantized 8bit bert. In 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing NeurIPS Edition (EMC2-NIPS). IEEE, December 2019.

[44] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection, 2024.
</end of paper 1>


<paper 2>
# Compressing Large Language Models using Low Rank and Low Precision Decomposition 

Rajarshi Saha<br>Stanford University<br>rajsaha@stanford.edu

Naomi Sagan<br>Stanford University<br>nsagan@stanford.edu

Varun Srivastava<br>Stanford University<br>vsriva@stanford.edu

Andrea J. Goldsmith<br>Princeton University<br>goldsmith@princeton.edu

Mert Pilanci<br>Stanford University<br>pilanci@stanford.edu


#### Abstract

The prohibitive sizes of Large Language Models (LLMs) today make it difficult to deploy them on memory-constrained edge devices. This work introduces CALDERA - a new post-training LLM compression algorithm that harnesses the inherent low-rank structure of a weight matrix $\mathbf{W}$ by approximating it via a lowrank, low-precision decomposition as $\mathbf{W} \approx \mathbf{Q}+\mathbf{L R}$. Here, $\mathbf{L}$ and $\mathbf{R}$ are low rank factors, and the entries of $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$ are quantized. The model is compressed by substituting each layer with its $\mathbf{Q}+\mathbf{L R}$ decomposition, and the zero-shot performance of the compressed model is evaluated. Additionally, $\mathbf{L}$ and $\mathbf{R}$ are readily amenable to low-rank adaptation, consequently enhancing the zero-shot performance. CALDERA obtains this decomposition by formulating it as an optimization problem $\min _{\mathbf{Q}, \mathbf{L}, \mathbf{R}}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$, where $\mathbf{X}$ is the calibration data, and $\mathbf{Q}, \mathbf{L}, \mathbf{R}$ are constrained to be representable using low-precision formats. Theoretical upper bounds on the approximation error of CALDERA are established using a rank-constrained regression framework, and the tradeoff between compression ratio and model performance is studied by analyzing the impact of target rank and quantization bit budget. Results illustrate that compressing LlaMa2 7B/70B and LlaMa-3 8B models using CALDERA outperforms existing posttraining LLM compression techniques in the regime of less than 2.5 bits per parameter. The implementation is available at: https://github.com/pilancilab/caldera.


## 1 Introduction

Large Language Models (LLMs) stand out due to their remarkable ability to generate human-like text, thereby supporting a diverse range of applications ranging from writing assistance to code generation. These models leverage vast datasets and significant computational resources to achieve their impressive functionality. The architecture of LLMs typically includes multiple layers, each with weight matrices essential for encoding various aspects of the training data - from simple syntactic patterns to complex semantic relationships. However, the substantial size of these trained models leads to high computational costs and considerable energy consumption during inference, which can be challenging for deployment in resource-constrained environments. As LLMs continue to expand in scale, compression techniques to reduce the memory and computational requirements of the models are becoming crucial to ensure their broad accessibility.

Due to the correlated nature of language syntax and semantics learned during training, often, the weight matrices of LLMs exhibit redundancy, which manifests as a low-rank structure. This redundancy suggests the potential for compression without substantial loss in performance. This
work introduces CALDERA: Calibration Aware Low-Precision DEcomposition with Low-Rank Adaptation, which compresses LLMs by leveraging the approximate low rank structure inherent in these weight matrices. Given a matrix $\mathbf{W} \in \mathbb{R}^{n \times d}$, CALDERA approximates it as $\mathbf{W} \approx \mathbf{Q}+\mathbf{L R}$, where $\mathbf{Q} \in \mathbb{R}^{n \times d}, \mathbf{L} \in \mathbb{R}^{n \times k}$ and $\mathbf{R} \in \mathbb{R}^{k \times d}$. Here, the left and right low rank factors, respectively $\mathbf{L}$ and $\mathbf{R}$, are tall and wide matrices, and $k$ is the target rank. Furthermore, the entries of $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$ are represented using low-precision formats with $\mathrm{B}_{\mathrm{Q}}, \mathrm{B}_{\mathrm{L}}$ and $\mathrm{B}_{\mathrm{R}}$ bits per entry, respectively.

Since the singular value profile (aka spectrum) of the weight matrices of an LLM follow a decaying profile as shown in Fig. 1, the low-rank factors $\mathbf{L}$ and $\mathbf{R}$ capture the effect of the large singular components of $\mathbf{W}$ with high fidelity. Moreover, the backbone $\mathbf{Q}$, which is quantized aggressively - for instance, using $\mathrm{B}_{\mathrm{Q}}=2$ bits, coarsely captures the essence of the moderately decaying and low singular components of $\mathbf{W}$. CALDERA substitutes each weight matrix $\mathbf{W}$ in an LLM, with its approximate lowprecision and low-rank decomposition $\mathbf{Q}+\mathbf{L R}$, resulting in a post-training quantization strategy that delivers stateof-the-art zero-shot performance. In addition, since usu-

![](https://cdn.mathpix.com/cropped/2024_06_04_bc47e5d834c845042b4fg-02.jpg?height=352&width=555&top_left_y=496&top_left_x=1186)

Figure 1: Decaying spectrum of weight matrices (aka, "approximate low-rank") ally $k \ll \min \{n, d\}$,implying that the total number of parameters in $\mathbf{L R}$ is much smaller compared to the number of entries in $\mathbf{W}$ (i.e., $k(n+d) \ll n d$ ), CALDERA can readily fine-tune (or "adapt") the low rank factors $\mathbf{L}$ and $\mathbf{R}$ in order to boost the zero-shot results.

### 1.1 Significance and Related Works

Recent efforts have explored various avenues for compression, including but not limited to weight pruning, quantization, and the use of parameter-efficient training methods - each approach offering distinct advantages and tradeoffs. This section briefly reviews the current methodologies, highlighting the contributions and limitations of some studies closely related to this work.

LLM Compression and Outlier Mitigation: Recent studies like SmoothQuant [41], OPTQ [8], QuIP [3], AQLM [7], and QuIP\# [35] consider the challenging regime of sub-4 bit post-training LLM quantization. These works collectively emphasize the need to manage the impact of outliers, i.e., weights with unusually high magnitudes. Accommodating outliers necessitates choosing the dynamic range (or scale) of a quantizer to be high, consequently increasing the quantization error. QuIP equalizes (and reduces) the weight matrices by using a randomized matrix transform, and subsequently, QuIP\# employs E8 lattice to make the weights more amenable to vector quantization. Both QuIP and QuIP\# use a refined variant of the column-wise quantization method proposed in OPTQ, wherein error feedback from previously quantized columns of a matrix is used to compensate for the error incurred while quantizing subsequent columns. CALDERA utilizes this diverse arsenal of strategies and builds on top of QuIP\#, while capitalizing on the approximate low-rank structure of LLM weight matrices. While it is possible to obtain even more aggressively compressed LLMs [24], this approach requires training from scratch, which is computationally demanding.

Parameter Efficient Fine-Tuning (PEFT): In a related yet distinct vein of work, PEFT methods have gained significant momentum, aiming to adapt LLMs to specific tasks without extensive computational overhead. Recent studies such as QLoRA [5], LoftQ [22], and LQ-LoRA [13] have explored the intersection of PEFT and quantization, demonstrating that fine-tuning through low-rank updates, as originally proposed in LoRA [16], can mitigate the performance losses due to quantization. Given that CALDERA yields a decomposition $\mathbf{Q}+\mathbf{L R}$, the low-rank components are particularly suitable for fine-tuning with any existing PEFT methods, thereby enhancing the zero-shot capabilities.

Low Rank Approximation: The rank- $k$ approximation of a matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$ can be represented by the factorization $\mathbf{A} \approx \mathbf{L R}$, with $\mathbf{L} \in \mathbb{R}^{n \times k}$ and $\mathbf{R} \in \mathbb{R}^{k \times d}$, where $k \leq \min \{n, d\}$. Known as the Burer-Monteiro factorization, this method substantially decreases the number of parameters, thus reducing computational demands. Recent studies such as LoRD [18], ASVD [43], FWSVD [15], LASER [32], LQER [44], and ZeroQuant-V2 [42] have explored the efficacy of low-rank structures in LLM weights, treating low-rank factorization and quantization independently. In contrast, LPLR [30] approaches this by uniquely formulating a joint optimization problem for generic matrices, while simultaneously leveraging the equalization property of randomized transforms, as in [3]. CALDERA formally leverages this inherent low-rank structure for LLM compression along-
side existing frameworks such as QuIP\# [35] and LoftQ [22], providing additional flexibility for compression. Furthermore, rigorous theoretical guarantees are derived using a rank-constrained regression framework for obtaining a low precision and low-rank decomposition, thereby also analytically demonstrating its superiority over rank-agnostic strategies.

## 2 Problem Formulation

In a neural network layer, a weight matrix $\mathbf{W}$ transforms an input activation $\mathbf{x}$ into an output activation given by $\mathbf{W x}$. This transformation can be succinctly described using the matrix's singular value decomposition (SVD). For any matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$, the SVD is $\mathbf{A}=\sum_{i} \sigma_{i} \mathbf{u}_{i} \mathbf{v}_{i}$, where $\sigma_{i}, \mathbf{u}_{i}, \mathbf{v}_{i}$ are the $i^{\text {th }}$ singular value and the corresponding left and right singular vectors, respectively. The impact of each singular component $\mathbf{u}_{i} \mathbf{v}_{i}$ on the matrix's transformation is determined by the magnitude of $\sigma_{i}$. Given that weight matrices exhibit a decaying singular value profile (Fig. 1), indicating an approximate low-rank structure, lesser contributing singular components can be pruned with minimal impact on the functionality of the matrix, ensuring minimal distortion in the output activations.

CALDERA approximates the weight matrix of a neural network, $\mathbf{W}$, as a low-precision, low-rank decomposition, $\mathbf{W} \approx \mathbf{Q}+\mathbf{L R}$, with all components $\mathbf{Q}, \mathbf{L}, \mathbf{R}$ in low-precision format. Unlike previous works such as $[13,22,43,44]$, which represent the low-rank factors $\mathbf{L}$ and $\mathbf{R}$ in highprecision (16 or 32-bit floating point), this work extends their representation to low-precision. This further reduces the memory footprint while preserving performance. Alternatively, for the same memory footprint, it allows the target rank $k$ to be higher, thereby capturing the low rank structure with higher fidelity by including more of the higher singular value components. The following paragraph formalizes this as a constrained optimization problem.

For a given quantizer, let $\mathbb{Q}$ denote the set of discrete quantization points in $\mathbb{R}$. For B-bit quantization, the cardinality of $\mathbb{Q}$ satisfies $\log _{2}|\mathbb{Q}| \leq \mathrm{B}$. Consider a matrix $\mathbf{W} \in \mathbb{R}^{n \times d}$. The goal of this work is to obtain an decomposition $\mathbf{W} \approx \mathbf{Q}+\mathbf{L R}$ by approximately solving the minimization problem

$$
\begin{equation*}
\min _{\mathbf{Q}, \mathbf{L}, \mathbf{R}}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \quad \text { subject to } \quad \mathbf{Q} \in \mathbb{Q}_{\mathrm{Q}}^{n \times d}, \mathbf{L} \in \mathbb{Q}_{\mathrm{L}}^{n \times k}, \text { and } \mathbf{R} \in \mathbb{Q}_{\mathrm{R}}^{k \times d} \tag{1}
\end{equation*}
$$

Here, $\mathbb{Q}_{\mathrm{Q}}, \mathbb{Q}_{\mathrm{L}}$ and $\mathbb{Q}_{\mathrm{R}}$ denote the lattice codebooks used to quantize $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$, using $\mathrm{B}_{\mathrm{Q}}, \mathrm{B}_{\mathrm{L}}$ and $\mathrm{B}_{\mathrm{R}}$ bits, respectively. Furthermore, $\mathbf{X} \in \mathbb{R}^{m \times d}$ is a calibration matrix that aims to preserve the Frobenius norm error of the compressed layer output activations. If $\mathbf{W}$ is the first layer's weight matrix, $\mathbf{X}$ includes input embeddings from a calibration dataset, such as a subset of RedPajama [33], with the $i^{\text {th }}$ row representing the $i^{\text {th }}$ datapoint. For intermediate layers, $\mathbf{X}$ contains the input activations, which are the output activations of the preceding layer.

Using the Frobenius norm of the output of a layer as a proxy objective for quantizing the weight matrices of an LLM is a popular strategy, and was used in prior work of Nagel et al. [27] . This proxy objective function is particularly useful for post-training quantization of LLMs because their large size makes it difficult to apply sophisticated compression methods.

## 3 Proposed Algorithm: Calibration-Aware Low-Precision Decomposition with Low Rank Adaptation

This section introduces CALDERA to approximately solve (1) and get a $\mathbf{Q}+\mathbf{L R}$ decomposition of a weight matrix $\mathbf{W}$ using the calibration matrix $\mathbf{X}$. The pseudocode is provided in Alg. 1. It consists of a nested loop for alternately optimizing the variables $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$. Suppose $\mathrm{Q}_{\mathrm{Q}}, \mathrm{Q}_{\mathrm{L}}$ and $\mathrm{Q}_{\mathrm{R}}$, respectively, denote quantizers used for quantizing $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$. For instance, they can refer to uniformly dithered scalar quantizers, as described in App. G.2. Initially, the low-rank actors are set to $\mathbf{0}$, and $\mathbf{W}$ is quantized using the LDLQ quantizer proposed in [3, §3.1]. LDLQ is an adaptive quantization method that iteratively quantizes [8] each column of $\mathbf{W}$ using $\mathrm{Q}_{\mathrm{Q}}$ to get $\mathbf{Q}$ as

$$
\begin{equation*}
\mathbf{Q}^{(k)}=\mathrm{Q}_{\mathbf{Q}}\left(\mathbf{W}^{(k)}+\left(\mathbf{W}^{(1: k-1)}-\mathbf{Q}^{(1: k-1)}\right) \mathbf{a}_{k}\right) \tag{2}
\end{equation*}
$$

where $\mathbf{Q}^{(k)}, \mathbf{W}^{(k)}$ denote the $k^{\text {th }}$ column, $\mathbf{W}^{(1: k-1)}$ denotes the first $k$ columns, $\mathrm{Q}_{\mathrm{Q}}$ has a bit-budget $\mathrm{B}_{\mathrm{Q}}$, and $\mathbf{a}_{k} \in \mathbb{R}^{k-1}$ is a learnable sequence of vectors. Update Eq. (2) incorporates linear feedback from already quantized columns, it can be seen that $\mathbf{Q}$ satisfies $\mathbf{Q}=$ $\mathrm{Q}_{\mathrm{Q}}(\mathbf{W}+(\mathbf{W}-\mathbf{Q}) \mathbf{M})$, where the feedback matrix $\mathbf{M}$ is a strictly upper triangular matrix with
columns $\mathbf{a}_{k}$. Defining $\mathbf{H} \triangleq \frac{1}{m} \mathbf{X}^{\top} \mathbf{X}$ to be the (scaled) Hessian of the least squares objective in (1), [3] show that the optimal feedback matrix is the $\mathbf{M}$ obtained from the LDL decomposition of $m \mathbf{H}$, given by $m \mathbf{H}=(\mathbf{M}+\mathbf{I}) \mathbf{D}(\mathbf{M}+\mathbf{I})^{\top}$.

Subsequently, $\mathbf{Q}$ is fixed and the Low-Precision Low-Rank (LPLR) factorization of the residual, $(\mathbf{W}-\mathbf{Q})$, is computed. This is done by the LPLRFActorize submodule (Alg. 2), which is a refined version of the LPLR algorithm proposed in [30]. For a given matrix A, Alg. 2 minimizes

$$
\begin{equation*}
\min _{\mathbf{L}, \mathbf{R}}\left\|(\mathbf{L R}-\mathbf{A}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \quad \text { subject to } \quad \mathbf{L} \in \mathbb{Q}_{\mathrm{L}}^{n \times k}, \text { and } \mathbf{R} \in \mathbb{Q}_{\mathrm{R}}^{k \times d} \tag{3}
\end{equation*}
$$

where $Q_{L}$ and $Q_{R}$ use $B_{L}$ and $B_{R}$ bits, respectively. In contrast to [30], the objective in (3) is calibration-data aware. Therefore, the update equations are derived using a rank-constrained regression framework, as described in App. B. Moreover, lines 7 to 14 in LPLRFACTORIZE iteratively refine the estimates of $\mathbf{L}$ and $\mathbf{R}$, and can only yield a smaller Frobenius norm error. The left and right low-rank factor update equations are described as follows.

Initialization: In the absence of quantization constraints, a globally optimal solution to the optimization problem (3) can be found as described later in lemma 4.2. Consequently, the low-rank factors are initialized using rank-constrained regression in lines $2-4$. Since subsequent quantization disrupts optimality of the solution, the factors are iteratively updated to minimize this distortion.

Updating L: To update the left factor $\mathbf{L}$, lines 5 and 9 of Alg. 2 solves $\min _{\mathbf{Z}}\left\|(\mathbf{Z R}-\mathbf{A}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$. For a fixed $\mathbf{R}$, this is a least squares minimization, whose solution is available is closed form as $\grave{\mathbf{L}}=\left(\mathbf{A X}^{\top}\right)\left(\mathbf{R} \mathbf{X}^{\top}\right)^{\dagger}=\mathbf{A H R}^{\top}\left(\mathbf{R H R}{ }^{\top}\right)^{-1}$, as derived in App. C.1.

Updating R: Line 8 of Alg. 2, updates the right factor $\mathbf{R}$ by keeping $\mathbf{L}$ fixed and solving $\min _{\mathbf{Z}}\left\|(\mathbf{L Z}-\mathbf{A}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$. As this is an under-determined linear system, there exist multiple solutions for $\mathbf{Z}$, all attaining the same objective function value. It is shown in App. C. 1 that $\grave{\mathbf{R}}=\mathbf{L}^{\dagger} \mathbf{A H} \mathbf{H}^{\dagger}$ is a solution. The corresponding error is also obtained, which is used in the derivation of Thm. 4.1.

Computational Complexity: A high-level calculation is provided here, and detailed discussions can be found in App. D. It is worthwhile to note that the closed form expressions of $\grave{\mathbf{L}}$ and $\grave{\mathbf{R}}$, which are iteratively quantized, are functions of the Hessian $\mathbf{H}=\frac{1}{m} \mathbf{X}^{\top} \mathbf{X}$. Therefore, $\mathbf{H}$ can be computed offline initially, per LLM, by doing a single forward pass, and subsequently used for all model quantization experiments. For each layer, this pre-processing includes computing $\mathbf{H}$ and its LDL decomposition, along with computing $\mathbf{H} \mathbf{H}^{\dagger}$, requiring a total of $\mathrm{O}\left(m d^{2}+2 d^{3}\right)$ multiplications. Each outer iteration involves an LDLQ quantization. Quantizing the $k^{\text {th }}$ column has complexity $\mathrm{O}(n k)$, since feedback from $k$ already quantized columns need to be incorporated. Hence, quantizing a matrix in $\mathbb{R}^{n \times d}$ entails $\mathrm{O}\left(n^{2}+3 n\right)$ complexity. Moreover, LPLRFACTORIZE requires $\mathrm{O}\left(m^{2}(n+d)\right)$ to initialize, and subsequently, each inner iteration entails $\mathrm{O}(n d k)$. Assuming $n, d \geq m \gg k$, and keeping only the dominant terms, the total complexity of CALDERA, not including the complexity of the pre-processing discussed earlier, is $\mathrm{O}\left(\mathrm{T}_{\text {out }}\left(n^{2}+m^{2}(n+d)+n d k \mathrm{~T}_{\mathrm{in}}\right)\right)$.

Fine tuning via Low-Rank Adaptation: Once the weight matrices of each layer are replaced by its $\mathbf{Q}+\mathbf{L R}$ approximation, the zero-shot performance of (post-training) quantized model can be evaluated. $\S 5$ shows that CALDERA quantized models outperform existing strategies. Additionally, if desired, the low-rank factors $\mathbf{L}$ and $\mathbf{R}$ can be further fine-tuned using low-rank adaptation [13, 16, 22] on a small task-specific dataset. While the initialization of the fine-tuning step has quantized $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$, the fine-tuned factors are represented using 16-bits (BF16 format). Although this leads to a slight increase in the memory footprint, the performance gains from fine-tuning are substantial.

## 4 Approximation Error Analysis

The approximation error upper bounds are derived via a rank-constrained regression framework. Thm. 4.1 below (formally stated and proved in App. C.4) is an informal version of the main theoretical result of this paper, and provides an upper bound on the Frobenius norm error when CALDERA approximates a weight matrix $\mathbf{W}$ is as $\mathbf{W} \approx \mathbf{Q}+\mathbf{L R}$ by solving the optimization problem (1) using Alg. 1. For convenience of analysis, it is assumed that the dynamic range of $\mathrm{Q}_{\mathrm{Q}}$, denoted as $R$, is chosen to be high enough, ensuring it remains unsaturated. Consequently, for a scalar input the quantization error from $Q_{Q}$ has zero mean and bounded variance, given by $\frac{\Delta^{2}}{4}=\frac{R^{2}}{\left(2^{B} Q^{2}-1\right)^{2}}$.

Algorithm 1: CALDERA: Calibration Aware Low-Precision DEcomposition with Low-Rank Adaptation

```
Input: Matrix: $\mathbf{W} \in \mathbb{R}^{n \times d}$, Target rank: $k$, Calibration matrix: $\mathbf{X} \in \mathbb{R}^{m \times d}$, Outer and inner
                iterations: $\mathrm{T}_{\text {out }}, \mathrm{T}_{\text {in }}$, Quantizers: $\mathrm{Q}_{\mathrm{Q}}, \mathrm{Q}_{\mathrm{L}}, \mathrm{Q}_{\mathrm{R}}$, Flag: EnableLoRA, Fine-tune rank: $r$
Output: LPLR decomposition: $\mathbf{Q} \in \mathbb{Q}_{\mathrm{Q}}^{n \times d}, \mathbf{L} \in \mathbb{Q}_{\mathrm{L}}^{n \times k}, \mathbf{R} \in \mathbb{Q}_{\mathrm{R}}^{k \times d}$ s.t.
            $\mathbf{W} \mathbf{X}^{\top} \approx(\mathbf{Q}+\mathbf{L R}) \mathbf{X}^{\top}$
Initialize: $t \leftarrow 0, \mathbf{L}_{0} \leftarrow \mathbf{0}, \mathbf{R}_{0} \leftarrow \mathbf{0}$, MinError $\leftarrow \infty$
while $t<\mathrm{T}_{\text {out }}$ do
    Update $\mathbf{Q}: \mathbf{Q}_{t+1} \leftarrow \operatorname{LDLQ}\left(\mathbf{W}-\mathbf{L}_{t} \mathbf{R}_{t}, \mathrm{Q}_{\mathrm{Q}}\right)$
    Update low-rank factors:

```

![](https://cdn.mathpix.com/cropped/2024_06_04_bc47e5d834c845042b4fg-05.jpg?height=46&width=951&top_left_y=665&top_left_x=430)

```
    if $\left\|\left(\mathbf{Q}_{t+1}+\mathbf{L}_{t+1} \mathbf{R}_{t+1}-\mathbf{W}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}<$ MinError then
        $\mathbf{Q}_{\text {best }} \leftarrow \mathbf{Q}_{t+1}, \mathbf{L}_{\text {best }} \leftarrow \mathbf{L}_{t+1}, \mathbf{R}_{\text {best }} \leftarrow \mathbf{R}_{t+1}$,
            MinError $\leftarrow\left\|\left(\mathbf{Q}_{t+1}+\mathbf{L}_{t+1} \mathbf{R}_{t+1}-\mathbf{W}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$
    end
    $t \leftarrow t+1$
end
if EnableLoRA is TRUE then
    Further Fine-tune top- $r$ singular components of $\mathbf{L}_{\text {best }}$ and $\mathbf{R}_{\text {best }}$ to 16-bit precision using
        Low-Rank Adaptation (LoRA) (as in [13, 16, 22])
end
return $\mathbf{Q}_{\text {best }}, \mathbf{L}_{\text {best }}, \mathbf{R}_{\text {best }}$
```

Theorem 4.1. Approximation error of CAlDERA (Informal) Given $\mathbf{W} \in \mathbb{R}^{n \times d}$ and $\mathbf{X} \in \mathbb{R}^{m \times d}$ with $m \leq d$, let $\mathbf{D}$ be obtained from the LDL decomposition $\mathbf{X}^{\top} \mathbf{X}=m \mathbf{H}=(\mathbf{M}+\mathbf{I}) \mathbf{D}(\mathbf{M}+\mathbf{I})^{\top}$, and $\lambda_{\max }, \lambda_{\min }$ denote the max and min eigenvalues of $\mathbf{H}$. Additionally, let $\mathbf{Q} \triangleq \operatorname{LDLQ}\left(\mathbf{W}, \mathrm{Q}_{\mathrm{Q}}\right)$, where $\mathrm{Q}_{\mathrm{Q}}$ has dynamic range $\mathrm{R}$ and bit-budget $\mathrm{B}_{\mathrm{Q}}$, the quantization error be $\boldsymbol{\eta} \triangleq \mathrm{Q}_{\mathrm{Q}}(\mathbf{Q}+(\mathbf{W}-$ $\mathbf{Q}) \mathbf{M})-(\mathbf{Q}+(\mathbf{W}-\mathbf{Q}) \mathbf{M})$, and $\sigma_{1} \geq \ldots \geq \sigma_{k} \ldots$ be the singular values of $\mathbf{X}(\mathbf{W}-\mathbf{Q})^{\top}$. If the target rank $k$ satisfies $0.25 \lambda_{\min }^{1 / 2}\left(m \sigma_{1}\right)^{-1} \lambda_{\max }^{-3 / 2} \sum_{i>k} \sigma_{i}^{2} \leq k \leq m$, and the dynamic ranges of $\mathrm{Q}_{\mathrm{L}}$ and $\mathrm{Q}_{\mathrm{R}}$ are set as $\mathrm{R}_{\mathrm{L}}=\frac{2 \sigma_{1}}{\sigma_{k} \sqrt{m \lambda_{\min }}}$ and $\mathrm{R}_{\mathrm{R}}=\sigma_{1}$, then $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$ returned by Alg. 1 satisfy

$$
\frac{1}{n m} \mathbb{E}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \leq \frac{1}{n} \sum_{i>k} \mathbb{E} \lambda_{i}\left(\boldsymbol{\eta} \mathbf{D} \boldsymbol{\eta}^{\top}\right)+\epsilon \lesssim \frac{4 d \lambda_{\max } \mathrm{R}^{2}}{\pi\left(2^{\mathrm{B}_{\mathrm{Q}}}-1\right)^{2}}\left(1-\frac{k}{2 n}\right)^{2}+\epsilon
$$

while utilizing an average budget of $\frac{1}{2} \log _{2}\left(\frac{k \sigma_{1}^{3}}{m \epsilon \sigma_{k}} \frac{\lambda_{\max }}{\lambda_{\min }} \sqrt{d / n}\right)$ bits per parameter for the low-rank factors $\mathbf{L}$ and $\mathbf{R}$, when $n \approx d$. Here, the expectation is over the stochasticity of the quantizers.

An informal version of the main result is provided here, and the formal version including specific constant values, along with the derivation, can be found in App. C.4. The requirement $m \leq d$ is not restrictive, as $\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}=\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{H}^{1 / 2}\right\|_{\mathrm{F}}^{2}$, i.e., (1) can be rewritten to ensure $m=d$. The approximation error upper bound given by Thm. 4.1 can be directly compared with the result of Chee et al. [3, Thm. 1], which states that for vanilla LDLQ without LPLRFACTORIZE,

$$
\begin{equation*}
\mathbb{E}\left\|(\mathbf{Q}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \leq \mathbb{E}\left[\operatorname{Tr}\left(\boldsymbol{\eta} \mathbf{D} \boldsymbol{\eta}^{\top}\right)\right]=\sum_{i=1}^{n} \mathbb{E} \lambda_{i}\left(\boldsymbol{\eta} \mathbf{D} \boldsymbol{\eta}^{\top}\right) \tag{4}
\end{equation*}
$$

Evidently, Alg. 1 yields a smaller error provided, $\sum_{i>k} \mathbb{E} \lambda_{i}\left(\boldsymbol{\eta} \mathbf{D} \boldsymbol{\eta}^{\top}\right)<\sum_{i=1}^{k} \mathbb{E} \lambda_{i}\left(\boldsymbol{\eta} \mathbf{D} \boldsymbol{\eta}^{\top}\right)-\epsilon$, where $\epsilon$ can be chosen to be arbitrarily small. Furthermore, since the expression in Thm. 4.1 consists of two terms, namely, the rank-constrained regression error, which depends on the target rank $k$, and the additive quantization error of $\epsilon$, which is dictated by the bit-budgets used for $\mathbf{L}$ and $\mathbf{R}$, this upper bound can be made arbitrarily small by ensuring that the two terms are approximately equal, i.e., $\mathbb{E}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$ is upper bounded by $2 \epsilon$. This is apparent in the following regimes:

(i) $k \ll n$ : In this regime, $k$ is treated as a constant as $n$ grows. Then, if the bit-budget $\mathrm{B}_{\mathrm{Q}}$ satisfies

$$
\mathrm{B}_{\mathrm{Q}} \geq \log _{2}\left(2 \mathrm{R}(\pi \epsilon)^{-1 / 2} \sqrt{n m d \lambda_{\max }}+1\right), \quad \text { then } \quad \mathbb{E}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \leq 2 \epsilon
$$

(ii) $k=\mathrm{O}(n)$ : For a fixed $\mathrm{B}_{\mathrm{Q}}$, if $k$ is allowed to grow with dimension $n$, then choosing $k$ to satisfy $k \geq 2 n-\left(2^{\mathrm{B}_{\mathrm{Q}}}-1\right) \mathrm{R}^{-1}(\pi \epsilon)^{1 / 2}\left(m d \lambda_{\max }^{-1 / 2}\right) \sqrt{n} \quad$ ensures $\quad \mathbb{E}\left\|(\mathbf{Q}+\mathbf{L R}-\mathbf{W}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2} \leq 2 \epsilon$.

This implies that the upper bound can be made arbitrarily small by either (i) increasing the bit-budget of the backbone, i.e., $\mathrm{B}_{\mathrm{Q}}$, for a fixed rank $k$, or (ii) increasing the rank $k$ for a fixed $\mathrm{B}_{\mathrm{Q}}$, for example, $\mathrm{B}_{\mathrm{Q}}=2$. Alternatively stated, this provides a tunable knob for controlling the error by trading off the allocated bit-budget between the backbone $\mathbf{Q}$ and the low-rank factors $\mathbf{L}, \mathbf{R}$.

### 4.1 Analysis Outline

In this section, a brief proof sketch is presented, highlighting the major challenges in the proof and how they are addressed. For analysis, $\mathbf{Q}$ is assumed to be updated prior to $\mathbf{L}, \mathbf{R}$ in Alg. 1. However, in practice, the update order is inconsequential, and can be swapped, depending on whichever yields a smaller error. The complete derivation of the approximation error is provided in App. C. A key ingredient of the proof is the solution of the rank-constrained regression problem, defined as,

$$
\begin{equation*}
\min _{\operatorname{rank}(\mathbf{Z}) \leq k}\|\mathbf{X Z}-\mathbf{Y}\|_{\mathrm{F}}^{2} \tag{5}
\end{equation*}
$$

Although this problem is non-convex, it can be solved to global optimality via two SVDs [40]. The following lemma characterizes the solution to the optimization problem in (5).

Lemma 4.2. Given $\mathbf{Y} \in \mathbb{R}^{m \times n}$, and full $\operatorname{rank} \mathbf{X} \in \mathbb{R}^{m \times d}$, where $m \leq d$. Let $\mathbf{X}=\mathbf{U} \widetilde{\boldsymbol{\Sigma}} \mathbf{V}^{\top}$ and

![](https://cdn.mathpix.com/cropped/2024_06_04_bc47e5d834c845042b4fg-06.jpg?height=57&width=1279&top_left_y=1045&top_left_x=366)

$$
\mathbf{Z}_{*} \triangleq \underset{\operatorname{rank}(\mathbf{Z}) \leq k}{\arg \min }\|\mathbf{X Z}-\mathbf{Y}\|_{\mathrm{F}}^{2}=\left(\mathbf{V} \mathbf{I}_{m} \boldsymbol{\Sigma}^{-1} \mathbf{U}_{\mathbf{I}}\right)\left(\mathbf{I}_{k}^{\top} \grave{\boldsymbol{\Sigma}} \mathbf{V}^{\top}\right)
$$

where $\boldsymbol{\Sigma}:=\widetilde{\boldsymbol{\Sigma}} \mathbf{I}_{m} \in \mathbb{R}^{m \times m}$ is a diagonal matrix consisting of the non-zero singular values of $\mathbf{X}$. Moreover, denoting the non-zero singular values of $\mathbf{Y}$ as $\left\{\sigma_{i}(\mathbf{Y})\right\}_{i=1}^{m}$, the optimal value of (7) is

$$
\begin{equation*}
\min _{\operatorname{rank}(\mathbf{Z}) \leq k}\|\mathbf{X Z}-\mathbf{Y}\|_{\mathrm{F}}^{2}=\left\|\mathbf{X Z} \mathbf{Z}_{*}-\mathbf{Y}\right\|_{\mathrm{F}}^{2}=\sum_{i=k+1}^{m} \sigma_{i}^{2}(\mathbf{Y}) \tag{6}
\end{equation*}
$$

The complete lemma (with the case $m>d$ ), and the derivation, are provided in App. B. Using lemma 4.2, the approximation error of LPLRFACTORIZE is analyzed in App. C.3. Specifically,

Algorithm 2: LPLRFACTORIZE(A, $\left.k, \mathbf{X}, \mathrm{Q}_{\mathrm{L}}, \mathrm{Q}_{\mathrm{R}}, T_{\text {in }}\right)$ : LPLR factorization submodule

Input: Matrix: $\mathbf{A} \in \mathbb{R}^{n \times d}$, Target rank: $k$, Calibration matrix: $\mathbf{X} \in \mathbb{R}^{m \times d}$, Iterations: $\mathrm{T}_{\mathrm{in}}$, Quantizers: $\mathrm{Q}_{\mathrm{L}}, \mathrm{Q}_{\mathrm{R}}$

Output: Low precision Low Rank factors: $\mathbf{L} \in \mathbb{Q}^{n \times k}, \mathbf{R} \in \mathbb{Q}^{k \times d}$ s.t. $\mathbf{A X}{ }^{\top} \approx \mathbf{L R X}{ }^{\top}$

Initialize: Iteration counter: $i \leftarrow 0$

Compute SVD of $\mathbf{X}$ as $\mathbf{U} \tilde{\boldsymbol{\Sigma}} \mathbf{V}^{\top}$.

Compute SVD of $\mathbf{U}^{\top} \mathbf{X} \mathbf{A}^{\top}$ as $\mathbf{U} \grave{\boldsymbol{\Sigma}} \grave{\mathbf{V}}^{\top}$

Get right low-rank factor: $\mathbf{R}_{0} \leftarrow \mathrm{Q}_{\mathrm{R}}\left(\mathbf{I}_{k}^{\top} \grave{\boldsymbol{\Sigma}} \grave{\mathbf{V}}^{\top}\right)$

Get left low-rank factor: $\mathbf{L}_{0} \triangleq \mathrm{Q}_{\mathrm{L}}\left(\grave{\mathbf{L}}_{0}\right)$, where $\grave{\mathbf{L}}_{0}=\arg \min _{\mathbf{Z} \in \mathbb{R}^{k \times d}}\left\|\left(\mathbf{Z R} \mathbf{R}_{0}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$

$\mathbf{L}_{\text {best }} \leftarrow \mathbf{L}_{0}, \mathbf{R}_{\text {best }} \leftarrow \mathbf{R}_{0}$, MinError $\leftarrow\left\|\left(\mathbf{L}_{0} \mathbf{R}_{0}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$.

while $i<\mathrm{T}_{\text {in }}$ do

Update right: $\mathbf{R}_{i+1} \leftarrow \mathrm{Q}_{\mathrm{R}}\left(\grave{\mathbf{R}}_{i+1}\right)$, where $\grave{\mathbf{R}}_{i+1}=\arg \min _{\mathbf{Z} \in \mathbb{R}^{k \times d}}\left\|\left(\mathbf{L}_{i} \mathbf{Z}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$

Update left: $\mathbf{L}_{i+1} \leftarrow \mathrm{Q}_{\mathrm{L}}\left(\grave{\mathbf{L}}_{i+1}\right)$, where $\grave{\mathbf{L}}_{i+1}=\arg \min _{\mathbf{Z} \in \mathbb{R}^{n \times k}}\left\|\left(\mathbf{Z R} \mathbf{R}_{i}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$

if $\left\|\left(\mathbf{L}_{i+1} \mathbf{R}_{i+1}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}<$ MinError then

$\mathbf{L}_{\text {best }} \leftarrow \mathbf{L}_{i+1}, \mathbf{R}_{\text {best }} \leftarrow \mathbf{R}_{i+1}$, MinError $\leftarrow\left\|\left(\mathbf{L}_{i+1} \mathbf{R}_{i+1}-\mathbf{A}\right) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$

end

$i \leftarrow i+1$

end

return $\mathbf{L}_{\text {best }}, \mathbf{R}_{\text {best }}$
lemma C. 3 shows that for any input matrix $\mathbf{A}$, Alg. 2 with suitably chosen $\mathrm{B}_{\mathrm{L}}$ and $\mathrm{B}_{\mathrm{R}}$, ensures that $\mathbb{E}\left\|(\mathbf{L R}-\mathbf{A}) \mathbf{X}^{\top}\right\|_{\mathrm{F}}^{2}$, as in (3), can be upper bounded by twice the sum of squared trailing singular values, (ref. (6)). While proving lemma C.3, it is assumed that if $\mathrm{Q}_{\mathrm{L}}$ or $\mathrm{Q}_{\mathrm{R}}$ gets saturated, a trivial output of $\mathbf{L}=\mathbf{0}, \mathbf{R}=\mathbf{0}$ is returned. Therefore, lemmas C. 1 and C. 2 specify choosing the dynamic ranges $R_{R}$ and $R_{L}$ to be sufficiently high so that saturation happens with a very low probability. The proof of Thm. 4.1 is completed by using the LDL decomposition of $m \mathbf{H}$ as proposed in [3], along with an application of Marchenko-Pastur approximation to bound the expected eigenvalues of the quantization error, i.e., $\mathbb{E} \lambda_{i}\left(\boldsymbol{\eta} \boldsymbol{\eta}^{\top}\right)$, yielding the final inequality.

## 5 Numerical Simulations

The efficacy of CALDERA is assessed by using it to compress three popular open source LLMs from Meta AI, namely, LLaMa-2 7B, LLaMa-2 70B [34] and LLaMa-3 8B [26]. The framework is built in PyTorch on top of the QuIP\# [35] and LoftQ [22], and is available at https://github.com/pilancilab/caldera.

Baselines. The full-rank matrix $\mathbf{Q}$, also referred to as the backbone, is quantized to 2-bits using the LDLQ procedure from QuIP [3, 35], employing an E8 lattice quantizer [38]. For CALDERA, which allows even the low-rank factors, $\mathbf{L}$ and $\mathbf{R}$, to be represented in low-precision, the quantization is also performed with an E8 lattice. Prior to running Alg. 1, a randomized Hadamard transform (RHT) is applied to the left and the right of the input weight matrix, as the incoherence pre-processing step, to equalize the magnitude of the entries making them more robust to quantization. In other words, CALDERA decomposition is performed on $\widetilde{\mathbf{W}} \triangleq \mathbf{H}_{\mathrm{L}}^{\top} \mathbf{W} \mathbf{H}_{\mathrm{R}}$, where $\mathbf{H}_{\mathrm{L}}$ and $\mathbf{H}_{\mathrm{R}}$ are Hadamard matrices, right-multiplied by a diagonal matrix with i.i.id. $\{ \pm 1\}$ entries. In addition, the Hessian matrix obtained from the calibration data is substituted by $\widetilde{\mathbf{H}} \triangleq \mathbf{H}_{\mathrm{R}}^{\top} \mathbf{H} \mathbf{H}_{\mathrm{R}}$. As described in [3], this improves the quantization error incurred by LDLQ. Further details are provided in App. E.2.

Metrics. The performance of CALDERA is evaluated using perplexity on the test splits of the Wikitext2 [25] and C4 [6] datasets, as well as task-specific goodness-of-fit metrics such as zeroshot accuracy for sequence classification. Specifically, zero-shot accuracy was measured on the Winogrande [19], RTE [1, 39], PiQA [2], ARC-Easy, and ARC-Challenge [4] tasks. App. E. 3 provides more details regarding these benchmarks. Perplexity was measured using a sequence length equal to the model's maximum context length, i.e., 4096 for LLaMa-2, and 8192 for LLaMa-3. Zeroshot experiments were performed using EleutherAI's Language Model Evaluation Harness [9].

### 5.1 Zero-shot Results

Tables 1 and 2 report the perplexities and accuracies for CALDERA with varying target rank $(k)$ of $\mathbf{L}$ and $\mathbf{R}$. A smaller value is better for perplexity, which is defined as the $\exp (\cdot)$ of the training objective, while zero-shot accuracies are reported as percentages. Per-parameter bit budgets range from 2.1 (e.g., rank-64 factors in 4 -bit precision) to 2.4 bits (e.g., rank-64 factors in half precision or rank-256 factors in 4-bit precision). For comparison, the $\mathbf{Q}+\mathbf{L R}$ decomposition of weight matrices found in the QuIP\# codebase was performed on each model. For the sake of direct comparison, finetuning of the diagonal matrices in RHT was omitted. As QuIP\# does not support quantized factors, $\mathbf{L}$ and $\mathbf{R}$ are rank-64 in order to ensure that the per-parameter bit-budget remains in the $2-2.4$ range. As another baseline comparison, each model is quantized using QuIP\# without any low-rank factors. Results for the unquantized LLaMa-2 7B, LLaMa-2 70B, and LLaMa-3 8B models are also provided.

For all but the LLaMa-2 70B model, the rank-256 CALDERA decomposition with 4-bit factors had the lowest perplexity and highest accuracy. As CALDERA supports quantizing low-rank factors with minimal performance loss, more singular components can be captured compared to using halfprecision factors while employing the same number of bits. Consequently, the low-rank factors can regain the performance that was compromised when the backbone $\mathbf{Q}$ was quantized to 2 bits. Since zero-shot experiments have some inherent randomness and low-rank regularization effects [32], the zero-shot accuracies reported here are not as directly indicative of quantization performance as the perplexity results. In addition, $\S 5.3$, demonstrates that degradation in zero-shot accuracy can be recovered via LoRA fine-tuning. It is worthwhile to note these results substantiate the claims of [17],

Table 1: Zero-shot perplexities (denoted by $\downarrow$ ) and accuracies ( $\uparrow$ ) for LLaMa-2. $\mathrm{B}_{\mathrm{Q}}=2$ bits throughout.

| Method | Rank | $\mathrm{B}_{\mathrm{L}}\left(=\mathrm{B}_{\mathrm{R}}\right)$ | Avg Bits | Wiki2 $\downarrow$ | $\mathrm{C} 4 \downarrow$ | Wino $\uparrow$ | RTE $\uparrow$ | PiQA $\uparrow$ | ArcE $\uparrow$ | ArcC $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CALDERA (7B) | 64 | 16 | 2.4 | 7.36 | 9.47 | 64.6 | 66.4 | 73.7 | 60.8 | 31.7 |
| CALDERA (7B) | 64 | 4 | 2.1 | 7.37 | 9.74 | 63.7 | 62.1 | 72.3 | 60.9 | 31.7 |
| CALDERA (7B) | 128 | 4 | 2.2 | 6.76 | 8.83 | 63.8 | 59.9 | 75.1 | $\mathbf{6 5 . 1}$ | $\mathbf{3 4 . 6}$ |
| CALDERA (7B) | 256 | 4 | 2.4 | $\mathbf{6 . 1 9}$ | $\mathbf{8 . 1 4}$ | $\mathbf{6 6 . 0}$ | 60.6 | $\mathbf{7 5 . 6}$ | 63.6 | 34.0 |
| QuIP\# (7B, No FT) | 64 | 16 | 2.4 | 7.73 | 10.0 | 63.1 | $\mathbf{6 6 . 8}$ | 71.7 | 63.2 | 31.7 |
| QuIP\# (7B, No FT) | 0 | - | 2 | 8.23 | 10.8 | 61.7 | 57.8 | 69.6 | 61.2 | 29.9 |
| CALDERA (70B) | 64 | 16 | 2.2 | $\mathbf{4 . 5 0}$ | $\mathbf{6 . 3 8}$ | $\mathbf{7 5 . 4}$ | $\mathbf{7 1 . 7}$ | $\mathbf{7 9 . 2}$ | 71.8 | $\mathbf{4 3 . 9}$ |
| CALDERA (70B) | 128 | 4 | 2.1 | 5.07 | 7.10 | 72.9 | 62.1 | 78.0 | $\mathbf{7 3 . 2}$ | $\mathbf{4 3 . 9}$ |
| QuIP\# (70B, No FT) | 0 | - | 2 | 5.37 | 7.51 | 72.3 | 47.6 | 77.7 | 68.8 | 40.9 |
| Unquantized (7B) | 0 | - | 16 | 5.12 | 6.63 | 67.3 | 63.2 | 78.5 | 69.3 | 40.0 |
| Unquantized (70B) | 0 | - | 16 | 3.12 | 4.97 | 77.0 | 67.9 | 81.1 | 77.7 | 51.1 |

Table 2: Zero-shot perplexities (denoted by $\downarrow$ ) and accuracies ( $\uparrow$ ) for LLaMa-3 8B. $\mathrm{B}_{\mathrm{Q}}=2$ bits throughout.

| Method | Rank | $\mathrm{B}_{\mathrm{L}}\left(=\mathrm{B}_{\mathrm{R}}\right)$ | Avg Bits | Wiki2 $\downarrow$ | $\mathrm{C} 4 \downarrow$ | Wino $\uparrow$ | RTE $\uparrow$ | PiQA $\uparrow$ | ArcE $\uparrow$ | ArcC $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CALDERA | 64 | 16 | 2.4 | 9.22 | 10.5 | 68.9 | 63.9 | 72.9 | 69.9 | 36.5 |
| CALDERA | 64 | 4 | 2.1 | 10.6 | 11.8 | 66.9 | 58.5 | 71.8 | 68.2 | 34.3 |
| CALDERA | 128 | 4 | 2.2 | 9.21 | 10.5 | 67.6 | $\mathbf{6 9 . 7}$ | 74.4 | 71.8 | 36.3 |
| CALDERA | 256 | 4 | 2.4 | $\mathbf{8 . 2 2}$ | $\mathbf{9 . 5 6}$ | $\mathbf{6 9 . 7}$ | 65.0 | $\mathbf{7 5 . 1}$ | $\mathbf{7 3 . 2}$ | $\mathbf{4 0 . 0}$ |
| QuIP\# (No FT) | 64 | 16 | 2.4 | 10.9 | 11.8 | 66.5 | 57.0 | 69.6 | 63.8 | 31.0 |
| QuIP\# (No FT) | 0 | - | 2 | 13.8 | 15.6 | 63.2 | 52.7 | 67.6 | 57.6 | 28.2 |
| Unquantized | 0 | - | 16 | 5.54 | 7.01 | 73.5 | 68.6 | 79.7 | 80.1 | 50.2 |

which report that low-bit quantization of LLaMa-3 8B, significantly deteriorates model performance across various post-training quantization techniques, more so than with the LLaMa-2 series.

### 5.2 Fine-tuning of Randomized Hadamard Transform (RHT) Parameters

As CALDERA presents a general optimization framework for matrix decompositions of the form $\mathbf{Q}+\mathbf{L R}$, it can easily be extended with additional heuristics to improve performance. This section serves as a proof of concept, by examining one such heuristic: Fine-tuning of randomized Hadamard transform parameters. This technique, proposed in QuIP\# [35], involves fine-tuning the diagonal Rademacher matrices with $\pm 1$ entries in the RHT to minimize the cross-entropy loss between the output of the original and quantized models on the calibration dataset. Subsequently, RHT finetuning is performed on the models quantized using CALDERA in $\$ 5.1 .{ }^{1}$ Details on specific finetuning hyperparameters can be found in App. E.4.

Table 3: Zero-shot perplexities and accuracies for LLaMa-2 7B, with end-to-end fine-tuning of randomized

![](https://cdn.mathpix.com/cropped/2024_06_04_bc47e5d834c845042b4fg-08.jpg?height=43&width=951&top_left_y=1930&top_left_x=370)

| Method | Rank | $\mathrm{B}_{\mathrm{L}}\left(=\mathrm{B}_{\mathrm{R}}\right)$ | Avg Bits | Wiki2 $\downarrow$ | $\mathrm{C} 4 \downarrow$ | Wino $\uparrow$ | RTE $\uparrow$ | PiQA $\uparrow$ | ArcE $\uparrow$ | ArcC $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CALDERA | 64 | 16 | 2.4 | 6.22 | 8.23 | 64.2 | 63.2 | 76.1 | 63.4 | 34.7 |
| CALDERA | 64 | 4 | 2.1 | 6.30 | 8.32 | 64.6 | 65.7 | 75.4 | 63.3 | 35.4 |
| CALDERA | 128 | 4 | 2.2 | 6.09 | 8.06 | 65.1 | 61.0 | $\mathbf{7 6 . 5}$ | $\mathbf{6 5 . 1}$ | 35.6 |
| CALDERA | 256 | 4 | 2.4 | $\mathbf{5 . 8 4}$ | $\mathbf{7 . 7 5}$ | $\mathbf{6 5 . 7}$ | 60.6 | $\mathbf{7 6 . 5}$ | 64.6 | $\mathbf{3 5 . 9}$ |
| QuIP\#* $^{*}$ | 64 | 16 | 2.4 | 6.32 | 8.31 | 64.9 | $\mathbf{6 6 . 4}$ | 75.0 | $\mathbf{6 5 . 2}$ | 34.5 |
| QuIP\# $^{*}$ | 0 | - | 2 | 6.58 | 8.62 | 64.4 | 53.4 | 75.0 | 64.8 | 34.0 |

[^0]Table 4: Zero-shot perplexities and accuracies for LLaMa-3 8B, with end-to-end fine-tuning of randomized Hadamard transform parameters. $\mathrm{B}_{\mathrm{Q}}=2$ bits throughout. ${ }^{*}$ See Footnote 1.

| Method | Rank | $\mathrm{B}_{\mathrm{L}}\left(=\mathrm{B}_{\mathrm{R}}\right)$ | Avg Bits | Wiki2 $\downarrow$ | $\mathrm{C} 4 \downarrow$ | Wino $\uparrow$ | RTE $\uparrow$ | PiQA $\uparrow$ | ArcE $\uparrow$ | ArcC $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CALDERA | 64 | 16 | 2.4 | 7.63 | 8.9 | $\mathbf{7 0 . 3}$ | $\mathbf{7 0 . 8}$ | 75.4 | 72.4 | 39.0 |
| CALDERA | 64 | 4 | 2.1 | 8.06 | 9.34 | 69.5 | 64.3 | 76.0 | 71.5 | 40.0 |
| CALDERA | 128 | 4 | 2.2 | 7.76 | 9.02 | 69.4 | 63.9 | 76.0 | $\mathbf{7 3 . 7}$ | 41.8 |
| CALDERA | 256 | 4 | 2.4 | $\mathbf{7 . 3 4}$ | $\mathbf{8 . 6 8}$ | $\mathbf{7 0 . 3}$ | 70.4 | $\mathbf{7 6 . 5}$ | $\mathbf{7 3 . 6}$ | $\mathbf{4 2 . 3}$ |
| QuIP\#* $^{2} \mathbf{6 4}$ | 16 | 2.4 | 7.92 | 9.15 | 68.4 | 58.1 | 74.9 | 72.3 | 40.4 |  |
| QuIP\# $^{*}$ | 0 | - | 2 | 8.44 | 9.75 | 67.5 | 57.8 | 72.9 | 67.6 | 37.3 |

Perplexity and zero-shot results in Tables 3 and 4 match the trends in $\S 5.1$, i.e., CALDERA with rank-256 factors typically performs best, with the exception of RTE. In addition, perplexities are substantially lower than without the fine-tuning of randomized Hadamard transform parameters.

### 5.3 Low Rank Adaptation (LoRA) Fine-tuning Results

Once the $\mathbf{Q}+\mathbf{L R}$ decomposition with target rank $k$ is obtained, and $k$ takes values 64,128 and 256, fine-tuning the top $r(\leq k)$ singular components on a task-specific dataset can recover the performance lost due to quantization. Throughout all experiments in Table $5, r=64$ is chosen and those singular components are fine-tuned to 16 -bit precision, i.e., $\mathrm{BF} 16$ format. In other words, the approximation is written as $\mathbf{W} \approx \mathbf{Q}+\mathbf{L}_{1} \mathbf{R}_{1}+\mathbf{L}_{2} \mathbf{R}_{2}$, where $\mathbf{L}_{1} \in \mathbb{R}^{n \times r}, \mathbf{L}_{2} \in \mathbb{R}^{n \times(k-r)}$, $\mathbf{R}_{1} \in \mathbb{R}^{r \times d}, \mathbf{R}_{2} \in \mathbb{R}^{(k-r) \times d}, \mathbf{L}=\left[\mathbf{L}_{1} \mid \mathbf{L}_{2}\right], \mathbf{R}^{\top}=\left[\mathbf{R}_{1}^{\top} \mid \mathbf{R}_{2}^{\top}\right]$. The value of $r$ is set to 64 and $\mathbf{L}_{2}, \mathbf{R}_{2}$ are fined-tuned to $\mathbf{L}_{\mathrm{bf} 16}, \mathbf{R}_{\mathrm{bf16}}$ using low-rank adaptation similar to [13, 16, 22]. Doing this significantly on a small task-specific dataset like WikiText2, RTE, or Winogrande, can noticeably boost the zero-shot accuracy, as can be seen from Table $5 .{ }^{2}$

Table 5: CALDERA fine-tuning results for LLaMa-2 7B and LLaMa-3 8B. $B_{L}, B_{R}$ are the bit-budgets of $\mathbf{L}$ and $\mathbf{R}$ for the low-rank initialization. Rank-64 fine-tuned factors are represented in $\mathrm{BF} 16$ precision.

|  |  |  |  |  |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Method | Rank | $\mathrm{B}_{\mathrm{Q}}$ | $\mathrm{B}_{\mathrm{L}}\left(=\mathrm{B}_{\mathrm{R}}\right)$ | RHT FT | Avg Bits | Wiki2 $\downarrow$ | RTE $\uparrow$ | Wino $\uparrow$ | Wiki2 $\downarrow$ | RTE $\uparrow$ | Wino $\uparrow$ |
| CALDERA | 64 | 2 | 16 | No | 2.4 | 6.06 | 82.31 | 84.06 | 7.91 | 84.48 | 85.56 |
| CALDERA | 64 | 2 | 16 | Yes | 2.4 | 5.89 | 85.19 | 85.32 | 7.88 | 86.28 | 88.16 |
| CALDERA | 64 | 2 | 4 | No | 2.4 | 6.01 | 81.23 | 84.06 | 8.33 | 85.56 | 88.40 |
| CALDERA | 64 | 2 | 4 | Yes | 2.4 | 5.91 | 85.56 | 83.42 | 7.96 | $\mathbf{8 7 . 0 0}$ | 88.40 |
| CALDERA | 128 | 2 | 4 | No | 2.5 | 5.84 | 83.75 | 85.32 | 7.84 | 84.84 | 88.63 |
| CALDERA | 128 | 2 | 4 | Yes | 2.5 | 5.77 | 84.12 | 85.00 | 7.68 | 86.64 | 88.00 |
| CALDERA | 256 | 2 | 4 | No | 2.7 | 5.61 | 83.75 | $\mathbf{8 5 . 4}$ | $\mathbf{7 . 4 4}$ | 86.28 | 88.08 |
| CALDERA | 256 | 2 | 4 | Yes | 2.7 | $\mathbf{5 . 5 5}$ | $\mathbf{8 6 . 2 8}$ | 84.93 | $\mathbf{7 . 4 4}$ | 85.20 | $\mathbf{8 9 . 1 9}$ |
| LoftQ | 64 | 2 | 16 | - | 2.4 | 7.85 | - | - | - | - | - |
| LoftQ | 64 | 2.5 | 16 | - | 2.9 | 5.78 | - | - | - | - | - |
| LQ-LoRA | 64 | 2.75 | 8 | - | 2.95 | 5.67 | - | 72.4 | - | - | - |

Experimental details can be found in App. E.4. For each dataset, ten checkpoints are saved during the course of fine-tuning, and the best test performance is reported in Table 5. For datasets where test labels are not available, evaluation performance is reported instead.

For comparison, results from the LoftQ [22] and LQ-LoRA [13] papers are also reported, where available. As these papers were published before the release of LLaMa-3, only LLaMa-2 results are available. ${ }^{3}$ In each case, CALDERA achieves better performance at a lower bit budget.[^1]

## 6 Conclusions

In this work, the problem of obtaining a low-precision and low-rank decomposition of an LLM weight matrix was considered. A $\mathbf{Q}+\mathbf{L R}$ decomposition efficiently captures the high singular components of the weight matrix with sufficient fidelity, while coarsely compressing the less significant moderate-to-low singular components. An optimization-theoretically motivated algorithm was proposed to obtain this decomposition, which iteratively optimized the quantized backbone $\mathbf{Q}$ and the low-rank factors $\mathbf{L}, \mathbf{R}$. Additionally, it was shown that $\mathbf{L}$ and $\mathbf{R}$ can be efficiently fine-tuned using low-rank adaptation to boost the zero-shot performance of the quantized model. By utilizing a rankconstrained regression framework, an upper bound was established on the approximation error of the algorithm, and it was shown that this upper bound can be significantly smaller than prior bounds in the literature. Finally, the proposed method was empirically evaluated by compressing the LlaMA family of LLMs in the challenging sub-2.5 bits per parameter regime. The proposed approach can also be used to complement existing compression strategies; thereby making it efficient to distribute compressed LLMs and deploy them on regular consumer hardware, making them more accessible to researchers.

## Acknowledgements

This work was supported in part by the National Science Foundation (NSF) under Grant DMS2134248; in part by the NSF CAREER Award under Grant CCF-2236829; in part by the U.S. Army Research Office Early Career Award under Grant W911NF-21-1-0242; and in part by the Office of Naval Research under Grant N00014-24-1-2164.

## References

[1] L. Bentivogli, I. Dagan, H. T. Dang, D. Giampiccolo, and B. Magnini. The Fifth PASCAL Recognizing Textual Entailment Challenge, 2009.

[2] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

[3] J. Chee, Y. Cai, V. Kuleshov, and C. D. Sa. QuIP: 2-Bit Quantization of Large Language Models With Guarantees. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[4] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457v1, 2018.

[5] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=0UIFPHEgJU.

[6] J. Dodge, M. Sap, A. Marasović, W. Agnew, G. Ilharco, D. Groeneveld, M. Mitchell, and M. Gardner. Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus, 2021.

[7] V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko, and D. Alistarh. Extreme Compression of Large Language Models via Additive Quantization, 2024. URL https://arxiv.org/abs/2401.06118.

[8] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. OPTQ: Accurate Quantization for Generative Pre-trained Transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tcbBPnfwxS.

[9] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac'h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.

[10] G. H. Golub and C. F. van Loan. Matrix Computations. JHU Press, fourth edition, 2013. ISBN 1421407949 9781421407944. URL http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm.

[11] R. Gray and T. Stockham. Dithered quantizers. IEEE Transactions on Information Theory, 39 (3):805-812, 1993. doi: $10.1109 / 18.256489$.

[12] S. Gugger, L. Debut, T. Wolf, P. Schmid, Z. Mueller, S. Mangrulkar, M. Sun, and B. Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022.

[13] H. Guo, P. Greengard, E. P. Xing, and Y. Kim. LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning. arxiv:2311.12023, 2023. URL https://arxiv.org/abs/2311.12023.

[14] N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217-288, 2011. doi: 10.1137/090771806. URL https://doi.org/10.1137/090771806.

[15] Y.-C. Hsu, T. Hua, S. Chang, Q. Lou, Y. Shen, and H. Jin. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=uPv9Y3gmAI5.

[16] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[17] W. Huang, X. Ma, H. Qin, X. Zheng, C. Lv, H. Chen, J. Luo, X. Qi, X. Liu, and M. Magno. How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study, 2024.

[18] A. Kaushal, T. Vaidhya, and I. Rish. LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression, 2023.

[19] S. Keisuke, L. B. Ronan, B. Chandra, and C. Yejin. WinoGrande: An Adversarial Winograd Schema Challenge at Scale, 2019.

[20] A. Krishnamoorthy and D. Menon. Matrix inversion using cholesky decomposition. In 2013 Signal Processing: Algorithms, Architectures, Arrangements, and Applications (SPA), pages $70-72,2013$.

[21] H. J. Levesque, E. Davis, and L. Morgenstern. The winograd schema challenge. In Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning, KR'12, page 552-561. AAAI Press, 2012. ISBN 9781577355601.

[22] Y. Li, Y. Yu, C. Liang, P. He, N. Karampatziakis, W. Chen, and T. Zhao. LoftQ: LoRAFine-Tuning-Aware Quantization for Large Language Models. arxiv:2310.08659, 2023. URL https://arxiv.org/abs/2310.08659.

[23] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen. Dora: Weight-decomposed low-rank adaptation, 2024.

[24] S. Ma, H. Wang, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, R. Wang, J. Xue, and F. Wei. The era of 1-bit llms: All large language models are in 1.58 bits, 2024.

[25] S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer Sentinel Mixture Models, 2016.

[26] Meta AI. Introducing Meta Llama 3: The most capable openly available LLM to date. https://ai.meta.com/blog/meta-llama-3/, 2024. Accessed: 2024-05-07.

[27] M. Nagel, R. A. Amjad, M. Van Baalen, C. Louizos, and T. Blankevoort. Up or Down? Adaptive Rounding for Post-Training Quantization. In Proceedings of the 37th International Conference on Machine Learning, volume 119, pages 7197-7206, 2020.

[28] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion parameter models, 2020.

[29] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, KDD '20, page 3505-3506, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3406703. URL https://doi.org/10.1145/3394486.3406703.

[30] R. Saha, V. Srivastava, and M. Pilanci. Matrix Compression via Randomized Low Rank and Low Precision Factorization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=rxsCTtkqA9.

[31] L. Schuchman. Dither Signals and Their Effect on Quantization Noise. IEEE Transactions on Communication Technology, 12(4):162-165, 1964. doi: 10.1109/TCOM.1964.1088973.

[32] P. Sharma, J. T. Ash, and D. Misra. The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction, 2023.

[33] Together Computer. Redpajama: an open dataset for training large language models, October 2023. URL https://github.com/togethercomputer/RedPajama-Data.

[34] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov,

P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models, 2023.

[35] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, and C. D. Sa. QuIP\#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks, 2024.

[36] M. Udell and A. Townsend. Why are big data matrices approximately low rank? SIAM Journal on Mathematics of Data Science, 1(1):144-160, 2019. doi: 10.1137/18M1183480. URL https://doi.org/10.1137/18M1183480.

[37] R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018. doi: 10.1017/9781108231596.

[38] M. Viazovska. The sphere packing problem in dimension 8. Annals of Mathematics, 185(3), May 2017. ISSN 0003-486X. doi: 10.4007/annals.2017.185.3.7. URL http://dx.doi.org/10.4007/annals.2017.185.3.7.

[39] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, 2019. In the Proceedings of ICLR.

[40] S. Xiang, Y. Zhu, X. Shen, and J. Ye. Optimal exact least squares rank minimization. In Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '12, page 480-488, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450314626. doi: 10.1145/2339530.2339609. URL https://doi.org/10.1145/2339530.2339609.

[41] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In Proceedings of the 40th International Conference on Machine Learning, 2023.

[42] Z. Yao, X. Wu, C. Li, S. Youn, and Y. He. ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation, 2023.

[43] Z. Yuan, Y. Shang, Y. Song, Q. Wu, Y. Yan, and G. Sun. ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models, 2023.

[44] C. Zhang, J. Cheng, G. A. Constantinides, and Y. Zhao. LQER: Low-Rank Quantization Error Reconstruction for LLMs, 2024.
</end of paper 2>


<paper 3>
# SqueezeLLM: Dense-and-Sparse Quantization 

Sehoon Kim*1 Coleman Hooper*1 ${ }^{* 1}$ Amir Gholami ${ }^{* 1,2} \quad$ Zhen Dong $^{1}$<br>Xiuyu Li ${ }^{1} \quad$ Sheng Shen $^{1} \quad$ Michael W. Mahoney ${ }^{1,2,3}$ Kurt Keutzer ${ }^{1}$<br>${ }^{1}$ UC Berkeley $\quad{ }^{2}$ ICSI $\quad{ }^{3}$ LBNL<br>\{sehoonkim, chooper, amirgh, zhendong, xiuyu, sheng.s, mahoneymw, keutzer\} @berkeley.edu


#### Abstract

Generative Large Language Models (LLMs) have demonstrated remarkable results for a wide range of tasks. However, deploying these models for inference has been a significant challenge due to their unprecedented resource requirements. This has forced existing deployment frameworks to use multi-GPU inference pipelines, which are often complex and costly, or to use smaller and less performant models. In this work, we demonstrate that the main bottleneck for generative inference with LLMs is memory bandwidth, rather than compute, specifically for single batch inference. While quantization has emerged as a promising solution by representing weights with reduced precision, previous efforts have often resulted in notable performance degradation. To address this, we introduce SqueezeLLM, a post-training quantization framework that not only enables lossless compression to ultra-low precisions of up to 3-bit, but also achieves higher quantization performance under the same memory constraint. Our framework incorporates two novel ideas: (i) sensitivity-based non-uniform quantization, which searches for the optimal bit precision assignment based on second-order information; and (ii) the Dense-and-Sparse decomposition that stores outliers and sensitive weight values in an efficient sparse format. When applied to the LLaMA models, our 3-bit quantization significantly reduces the perplexity gap from the FP16 baseline by up to $2.1 \times$ as compared to the state-of-the-art methods with the same memory requirement. Furthermore, when deployed on an A6000 GPU, our quantized models achieve up to $2.3 \times$ speedup compared to the baseline. Our code is available at https://github.com/SqueezeAILab/SqueezeLLM.


## 1 Introduction

Recent advances in Large Language Models (LLMs) trained on massive text corpora, with up to hundreds of billions of parameters, have showcased their remarkable problem-solving capabilities across various domains [3, 7, 14, 24, 37, 38, 41- $-43,52]$. However, deploying these models for inference has been a significant challenge due to their demanding resource requirements. For instance, the LLaMA-65B model requires at least 130GB of RAM to deploy in FP16, which exceeds current GPU capacity. Even storing such large-sized models has become costly and complex.

As will be discussed in Sec. 3, the main performance bottleneck in LLM inference for generative tasks is memory bandwidth rather than compute. This means that the speed at which we can load and store parameters becomes the primary latency bottleneck for memory-bound problems, rather than arithmetic computations. However, recent advancements in memory bandwidth technology have been significantly slow compared to the improvements in computes, leading to the phenomenon known as the Memory Wall [36]. Consequently, researchers have turned their attention to exploring algorithmic methods to overcome this challenge.

One promising approach is quantization, where model parameters are stored at lower precision, instead of the typical 16 or 32-bit precision used for training. For instance, it has been demonstrated that LLM models can be stored in 8-bit precision without performance degradation [48], where 8 -bit quantization not only improves the storage requirements by half but also has the potential to improve inference latency and throughput. As a result, there has been significant research interest in quantizing models to even lower precisions. A pioneering approach is GPTQ [17] which uses a training-free quantization technique that achieves near-lossless 4-bit quantization for large LLM models with over tens of billions of parameters. However, achieving high quantization performance remains challenging, particularly with lower bit precision and for relatively smaller models (e.g., $<50 \mathrm{~B}$ parameters) such as the recent LLaMA models [43].[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-02.jpg?height=514&width=1400&top_left_y=214&top_left_x=359)

Figure 1: (Left) SqueezeLLM incorporates two key approaches: (i) sensitivity-based non-uniform quantization (Sec. 4.1), where quantization bins are allocated closer to sensitive values, and (ii) the Dense-and-Sparse decomposition (Sec. 4.2), which retains both sensitive values and outlier values as full-precision sparse format. When applied to LLaMA-7B with 3-bit quantization, our method outperforms the state-of-the-art methods [17,32] by a large perplexity margin of over 0.3 on the $\mathrm{C} 4$ benchmark. (Right) By applying our methods to LLaMA models of varying sizes, we can achieve improved trade-offs between perplexity and model size.

Contributions. In this paper, we conduct an extensive study of low-bit precision quantization and identify limitations in existing approaches. We first present performance modeling results demonstrating that the memory, rather than the compute, is the primary bottleneck in LLM inference with generative tasks (Sec. 3 and Fig. 2). Building on this insight, we introduce SqueezeLLM, a post-training quantization framework with a novel sensitivity-based non-uniform quantization and Dense-and-Sparse decomposition. These techniques enable lossless compression even at precisions as low as 3 bits with reduced model sizes and faster inference without compromising model performance. Our detailed contributions include:

- Sensitivity-based Non-Uniform Quantization: We demonstrate that uniform quantization of prior works is suboptimal for LLM inference for two reasons. First, the weight distributions in LLMs exhibit clear non-uniform patterns (Fig. 3). Second, the inference computation in prior works does not fully benefit from uniform quantization as the arithmetic is performed in FP16 precision, not in reduced precision. To address these, we propose a novel sensitivity-based non-uniform quantization method to achieve more optimal LLM quantization, which significantly improves the perplexity of 3-bit LLaMA-7B from 28.26 of uniform quantization to 7.75 on $\mathrm{C} 4$ (Sec. 4.1).
- Dense-and-Sparse Quantization: The weights in LLMs contain significant outliers, making low-bit quantization extremely challenging. To address this, we propose a simple solution that decomposes weights into dense and sparse components. The sparse part holds outlier values in full precision using efficient sparse storage methods and the dense part can have a more compact range to aid quantization. By extracting only $0.45 \%$ of the weight values as the sparse component, we further improve the perplexity of LLaMA-7B from 7.75 to 7.58 on $\mathrm{C} 4$ (Sec. 4.2).
- Evaluation: We extensively test SqueezeLLM on various models on language modeling tasks using the C4 and WikiText2 datasets as well as on the MMLU [23] and Vicuna benchmarks [6] (Sec. 5.3). Furthermore, our deployed models on A6000 GPUs also exhibit significant latency gains of up to $2.4 \times$ compared to the FP16 baseline, showcasing the effectiveness of our method in terms of both quantization performance and inference efficiency (Sec. 5.4.


## 2 Related Work

LLM Quantization. In Sec. A.1, we offer an overview and related works of Transformer quantization, with an emphasis on Post-Training Quantization (PTQ), which is the primary focus of our work. With the increasing popularity of LLMs, weight-only quantization has surfaced as a promising approach to reduce memory consumption and enhance inference efficiency. GPTQ [17] has been a pioneering work, and AWQ [32] and $\mathrm{SpQR}$ [12] have also suggested the weight-only quantization schemes concurrent to our work. Our work, however, is different in two key aspects. First, our work employs non-uniform quantization as opposed to the uniform quantization of the aforementioned works.

In particular, our sensitivity-based non-uniform quantization not only better represents non-uniform distributions of weights but also strategically reduces the impact on more sensitive values, thereby enabling more aggressive quantization without performance degradation. Second, while previous works quantize weights in a way that layer-wise output activations remain unaffected, our approach targets preserving the model's final output. This strategy of minimizing the final loss, as shown in Sec. A.4.4, leads to better quantization performance since it is a direct measure of the end-to-end performance degradation after quantization.

For low-bit LLM quantization, [11] has recently introduced the NF datatype, underscoring the importance of non-uniform quantization. However, our approach differs by offering a more dynamic non-uniform representation that accounts for both weight distributions and sensitivity of values, as opposed to the static, hard-coded NF datatype that assumes the normal distribution of the weights. While previous studies [22 47] have utilized k-means clustering in quantization, our work pioneers its application in LLM quantization. Furthermore, we introduce the novel sensitivity-based weighted k-means clustering strategy, enabling lossless sub-4-bit quantization by significantly reducing performance degradation in contrast to the sensitivity-agnostic counterpart (Fig. 11.

Among the various challenges in low-bit Transformer quantization, one key issue is the presence of outliers [29], which can unnecessarily increase the quantization range. To address this issue, outlier-aware quantization methods have been investigated [2, 10, 45, 46]. Notably, [10] keeps outlier activations in floating-point, while [46] transfers outlier factors to later layers without affecting functionality. These focus on activations, which is not a concern in our work where all activations are in floating-point. Our Dense-and-Sparse quantization instead tackles weight outliers for low-bit LLM quantization.

Concurrently to our work, $\operatorname{SpQR}$ [12] also explores outlier extraction in the context of weight quantization. While it shows a promising result on outlier extraction, SqueezeLLM, leveraging sensitivity-based non-uniform quantization, achieves precise quantization with significantly lower (e.g., $0.05 \%$ ) or even zero sparsity levels. This is critical for both reducing model size and improving inference speed, as higher sparsity often degrades latency. Furthermore, SqueezeLLM uses outlier extraction as a direct solution to prevent outliers from negatively impacting quantization performance, bypassing the need for using the grouping strategy as an indirect solution. This contrasts with $\mathrm{SpQR}$, which relies on fine-grained grouping that leads to increased model size and a more complex quantization process such as the bi-level quantization scheme.

Dense-and-Sparse Decomposition. Matrix decomposition into dense and sparse components has been explored in attention map decomposition [5, 9], leveraging the fact that attention patterns often present low-rank characteristics without a few outliers. Our research, however, is the first attempt to apply the dense-and-sparse decomposition strategy to weight matrices to improve quantization performance to the best of our knowledge. Additionally, we uniquely incorporate both outlier and sensitive values within the sparse matrix, which yields considerable improvement in post-quantization performance.

## 3 Memory Wall

Inference behavior broadly falls into two categories: compute-bound inference that is limited by computational throughput, and memory-bound inference that is bottlenecked by the rate at which data can be fed into the processing cores from memory. Arithmetic intensity, the ratio of compute to memory operations, is a typical metric used to assess this behavior. High and low arithmetic intensity indicates a compute-bound and memory-bound problem, respectively. For memory-bound problems, the speedup can be achieved by reducing the memory traffic rather than compute since the compute units in hardware are often underutilized waiting to receive data from memory.

Generative LLM inference exhibits extremely low arithmetic intensity compared to other workload 1 , [28]. This is because it consists almost entirely of matrix-vector operations, which limits the data reuse as each weight load can only process a single vector for a single token, and cannot be amortized across the multiple vectors for different tokens. This low arithmetic intensity needs to be contrasted with the compute operations on a typical GPU which is orders of magnitude higher than the memory operations ${ }^{2}$. The disparity between compute and memory bandwidth, along with the growing memory requirements of deep learning, has been termed the Memory Wall problem [20]. To further illustrate this problem, we used a simple roofline-based performance modeling approach [28] to study LLaMA-7B's runtime on[^1]![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-04.jpg?height=468&width=928&top_left_y=232&top_left_x=574)

Figure 2: Normalized runtime for LLaMA-7B when reducing the bit precision for the weights with sequence lengths of 128 (left) and 2048 (right). Results were obtained using a roofline-based performance model for an A5000 GPU. Reducing only the precision of the weights (and not the activations) is sufficient to obtain significant latency reductions.

an A5000 GPU with different bit precisions (Fig. 2). While we assume that all computations are kept at FP16, we see that the latency decreases linearly as we reduce the bit precision, indicating that the main bottleneck is memory, not compute.

In summary, in generative LLM inference, loading weights into memory is the primary bottleneck, while the cost of dequantization and FP16 computation is relatively small. Thus, by quantizing just the weights to lower precision, while leaving the activations in full precision, we can attain significant speedup as well as reduced model size. Given this insight, the appropriate strategy is to minimize the memory size even if it may add overhead to arithmetic operations.

## 4 Methodology

### 4.1 Sensitivity-Based Non-uniform Quantization

As in Fig. 3 (Top), weight distributions in LLMs demonstrate non-uniform patterns. The main task for quantization is to find an optimal way to allocate distinct quantized values (e.g., 8 for 3 bits) in a way that preserves model performance. A widely used approach in LLM quantization works is uniform quantization where the weight range is evenly divided into bins. This has two main issues. First, uniformly distributing quantized values is sub-optimal as weight distributions are typically non-uniform. Second, while the main advantage of uniform quantization is efficient integer computation, this does not lead to end-to-end latency improvement in memory-bound LLM inference. Therefore, we have chosen non-uniform quantization, which allows for a more flexible allocation of the representative values.

Finding an optimal non-uniform quantization configuration translates into solving a k-means problem. Given a weight distribution, the goal is to determine $k$ centroids that best represent the values (e.g., $k=8$ for 3-bit). This optimization problem for non-uniform quantization can be formulated as

$$
\begin{equation*}
Q(w)^{*}=\underset{Q}{\arg \min }\left\|W-W_{Q}\right\|_{2}^{2} \tag{1}
\end{equation*}
$$

where $W$ denotes the weights and $W_{Q}$ is the corresponding quantized weights (i.e., $[Q(w)$ for $w \in W]$ ), represented by $k$ distinct values $\left\{q_{1}, \cdots, q_{k}\right\}$. Here, the optimal solution $Q(w)^{*}$ can be obtained by 1-dimensional k-means clustering, which clusters the parameters into $k$ clusters and assign the centroid of each cluster as $q_{j}$ 's. While this already outperforms uniform quantization, we propose an improved sensitivity-based clustering algorithm.

Sensitivity-Based K-means Clustering. The quantization objective is to represent the model weights with low-bit precision with minimal perturbation in the model output [13]. While quantization introduces perturbations in each layer, we need to minimize the overall perturbation with respect to the final loss term, rather than focusing on individual layers, as it provides a more direct measure of the end-to-end performance degradation after quantization [30]. To achieve this, we need to place the k-means centroids closer to the values that are more sensitive w.r.t the final loss, rather than treating all weight values equally as in Eq. 1. To determine more sensitive values, we perform Taylor expansion to analyze how the loss changes in response to perturbations in the weights $W$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-05.jpg?height=580&width=829&top_left_y=217&top_left_x=645)

Figure 3: (Top) The weight distribution of one output channel in LLaMA-7B. The top-20 sensitive values are marked in red. (Bottom) Weight distributions after 3-bit quantization using uniform and sensitivity-based non-uniform quantization. In the latter case, the quantized values are clustered around the sensitive values.

$$
\begin{align*}
\mathcal{L}\left(W_{Q}\right) & \simeq \mathcal{L}(W)-g^{\top}\left(W-W_{Q}\right)  \tag{2}\\
& +\frac{1}{2}\left(W-W_{Q}\right)^{\top} H\left(W-W_{Q}\right) \tag{3}
\end{align*}
$$

where $g$ and $H=\mathbb{E}\left[\frac{\partial^{2}}{\partial W^{2}} \mathcal{L}(W)\right]$ are the gradient and Hessian of the loss at $W$. Assuming that the model has converged, the gradient $g$ can be approximated as zero which gives us the following formula for computing how much the model gets perturbed after quantization:

$$
\begin{equation*}
Q(w)^{*}=\underset{Q}{\arg \min }\left(W-W_{Q}\right)^{\top} H\left(W-W_{Q}\right) \tag{4}
\end{equation*}
$$

In the new optimization target, as compared to Eq. 1 , the perturbation of each weight after quantization, i.e., $W-W_{Q}$, is weighted by the scaling factor introduced by the second-order derivative, $H$. This highlights the importance of minimizing perturbations for weights with large Hessian values, as they have a greater impact on the overall perturbation of the final output. In other words, the second-order derivative serves as a measure of importance for each weight value.

Due to the cost of computing the Hessian, we use an approximation to the Hessian based on the Fisher information matrix $\mathcal{F}$, which can be calculated over a sample dataset $D$ as $H \simeq \mathcal{F}=\frac{1}{|D|} \sum_{d \in D} g_{d} g_{d}{ }^{\top}$. This only requires computing gradient for a set of samples, which can be calculated efficiently with existing frameworks. To make the optimization objective in Eq. 4 more feasible, we further approximate the Fisher information matrix as a diagonal matrix by assuming that the cross-weight interactions are negligible. This simplifies our objective target as follows:

$$
\begin{align*}
Q(w)^{*} & \simeq \underset{Q}{\arg \min }\left(W-W_{Q}\right)^{\top} \operatorname{diag}(\mathcal{F})\left(W-W_{Q}\right)  \tag{5}\\
& =\underset{Q}{\arg \min } \sum_{i=1}^{N} \mathcal{F}_{i i}\left(w_{i}-Q\left(w_{i}\right)\right)^{2} \tag{6}
\end{align*}
$$

An important consequence of Eq. 5 is the weighted $\mathrm{k}$-means clustering setting, where the centroids will be pulled closer to these sensitive weight values. In Fig. 3, we illustrate the top-20 sensitive values based on the Fisher information of the exemplary weight distribution. At the bottom, the quantized values assigned by uniform quantization (green) are compared to those assigned by the sensitivity-based k-means approach (purple), which achieves a better tradeoff by placing centroids near sensitive values, effectively minimizing quantization error. With 3-bit LLaMA-7B, sensitivity-based non-uniform quantization achieves a much lower perplexity of 7.75 compared to the 28.26 perplexity of round-to-nearest uniform quantization on $\mathrm{C} 4$ (Fig. 1 and Sec. 5.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-06.jpg?height=379&width=979&top_left_y=228&top_left_x=573)

Figure 4: The distributions of the (normalized) absolute weight values, for the output layers in MHA and the down layers in FFN across different layers in LLaMA-7B. Note that the distributions exhibit outlier patterns across all layers, with $99 \%$ of the values clustered within $\sim 10 \%$ of the entire range.

### 4.2 Dense-and-Sparse Quantization

Another challenge in low-bit LLM quantization is outlier values [2. 10, 45, 46]. In Fig 4, we plot the normalized weight distributions of different layers in LLaMA-7B, which demonstrate that $\sim 99.9 \%$ of the weights are concentrated in a narrow range of $\sim 10 \%$ of the entire distribution. Naively quantizing the weights with a large range will significantly degrade performance, especially at low precisions. However, this also implies opportunity as the range of the weight values can be contracted by a factor of 10 simply by removing a small number of outlier values (e.g., $0.1 \%$ ), yielding a significant improvement in quantization resolution. This will then help the sensitivity-based $\mathrm{k}$-means centroids to focus more on the sensitive values rather than a few outliers.

Motivated by this, we introduce a method to filter out outliers from the weight matrix $W$ by performing a simple yet effective decomposition into a sparse matrix $(S)$ containing the outliers and the remaining dense matrix (D) that can be quantized much more effectively thanks to its significantly reduced range of values. That is, $W=D+S$ where $D=W\left[T_{\min } \leq w \leq T_{\max }\right]$ and $S=W\left[w<T_{\min }\right.$ or $\left.w>T_{\max }\right]$. Here, $T_{\min } / \max$ are thresholds that define outliers based on the percentile of the distribution.

Importantly, the overhead of this decomposition is minimal, since the number of outlier values is small, normally less than $>0.5$ of sparsity. Therefore, the sparse matrix can be stored efficiently using methods like compressed sparse row (CSR) format. The inference is also straightforward with the decomposition as in $W X=D X+S X$, two kernels for dense and sparse multiplication can be overlapped, and the sparse part (SX) can benefit from sparse kernels (Sec. 4.3).

Sensitivity-Based Sparse Matrix. In addition to isolating outliers into a sparse matrix, we've also discovered the advantage of precisely representing a small number of highly sensitive weight matrix values. These values can be easily identified based on the Fisher information (Sec. 4.1). This not only maintains sensitive values with FP16 to avoid their impact on the model output, but also prevents the centroids of Eq 5 from skewing towards the sensitive values. We have observed that extracting only $0.05 \%$ of these sensitive values across layers substantially enhances quantization performance (Sec. A.4). Altogether, with 3-bit LLaMA-7B, extracting $0.45 \%$ of outlier and sensitive values further reduces the perplexity from 7.67 to 7.56 (Fig. 1 and Sec. 5.2.

### 4.3 Dense-and-Sparse Kernel Implementation

To efficiently process non-uniformly quantized values, we implement 3/4-bit CUDA LUT-based kernels for matrixvector multiplication between compressed weight matrices and uncompressed activation vectors. These kernels load the compressed weights and dequantize them piece-by-piece to minimize memory bandwidth utilization. The compressed matrices store 3/4-bit indices, which correspond to LUT entries containing FP16 values associated with the bins obtained from non-uniform quantization. After dequantization, all arithmetic is performed in FP16.

To optimize the handling of our Dense-and-Sparse representation, we develop kernels for sparse matrix-vector multiplication that load a matrix in CSR format and a dense activation vector, inspired by [15]. Since the non-zero entry distributions are highly skewed across rows (Sec. A.3), assigning a single thread per row can be inefficient due to uneven workload distribution among threads. Thus, we implement balanced hybrid kernels based on [16] by assigning an equal number of nonzeros per thread; this leads to additional synchronization across threads due to rows being processed by multiple threads, but leads to a more balanced work assignment. We set the number of threads such that
there were 10 nonzeros per thread. The dense non-uniform kernel and balanced sparse kernels are launched in one call to avoid overhead from summing the outputs from these separate operations.

## 5 Evaluations

### 5.1 Experiment Setup

Below is our experiment setup with more details in Sec. A.2

Models and Datasets. We have conducted comprehensive evaluations of SqueezeLLM using various models including LLaMA, LLaMA2 [43, 44], OPT [52] and Vicuna [6] (v1.1 and v1.3). We conduct language modeling evaluation using the $\mathrm{C} 4$ [37] and WikiText2 [34] datasets. We further evaluate the domain-specific knowledge and problem-solving ability using MMLU [23], and the instruction-following ability using the methodology in [6].

Baseline Methods. We compare SqueezeLLM against various PTQ methods for LLMs including RTN, GPTQ [17], AWQ [32] and SpQR [12]. To ensure a fair comparison, we use GPTQ with activation ordering throughout all experiments unless specified, which addresses the significant performance drop that would otherwise occur.

Quantization Details. For SqueezeLLM, we adopt channel-wise quantization where each output channel is assigned a separate lookup table. We use 2 different sparsity levels: $0 \%$ (dense-only) and $0.45 \%$ ( $0.05 \%$ sensitive values and $0.4 \%$ outlier values as discussed in Sec. 4.2). For measuring sensitivity, we use 100 random samples from the Vicuna training set for Vicuna models and C4 training set for the others. While grouping can also be incorporated with our method, we found it sub-optimal as compared to extracting sensitive/outlier values with sparsity (Sec. A.4.3).

Latency Profiling. We measure the latency and peak memory usage for generating 128 and 1024 tokens on an A6000 machine using the Torch CUDA profiler. As an official implementation of GPTQ (in particular, the grouped version) is not available, we implement an optimized kernel for single-batch inference based on the most active open-source codebase [21].

### 5.2 Main Results

Table 1 shows quantization results for LLaMA along with the baseline methods. The models are grouped based on their size to better compare size-perplexity trade-offs. See Fig. 5 for a visual illustration. Below we use LLaMA-7B as the main example to discuss the impact of dense-only and Dense-and-Sparse quantization, and subsequently discuss how these trends extend to larger models. We provide the full evaluation result on all LLaMA models in Tab. A.6.

Dense-only Quantization. In Tab. 1 (Top), we compare dense-only SqueezeLLM with $0 \%$ sparsity level and GPTQ without grouping. With 4-bit quantization, our method exhibits minimal degradation compared to the FP16 baseline, with only $\sim 0.1$ perplexity degradation on $\mathrm{C} 4$ and WikiText2, while reducing the model size by $3.95 \times$. Moreover, when compared to non-grouped GPTQ our method shows significant perplexity improvement of up to 0.22 .

The performance gap between the two methods becomes more pronounced with 3-bit quantization. SqueezeLLM outperforms GPTQ by a substantial margin of $1.80 / 1.22$ points on C4/WikiText2 with a $5.29 \times$ compression rate. This is only $0.67 / 0.55$ points off from the FP16 baseline. This demonstrates the effectiveness of the sensitivity-based non-uniform method for ultra-low-bit quantization.

Dense-and-Sparse Quantization. By leveraging the Dense-and-Sparse quantization, we achieve a further reduction in the perplexity gap from the FP16 baseline as shown in Tab. 1. This improvement is particularly significant with 3-bit quantization, where extracting just $0.45 \%$ of the values yields around 0.2 perplexity improvement. This enables nearly lossless compression with less than $0.1 / 0.5$ perplexity deviation from the FP16 baseline for 4/3-bit, respectively.

Both GPTQ and AWQ use a grouping strategy to enhance performance with a slight overhead in model size. However, we demonstrate that SqueezeLLM with a sparsity level of $0.45 \%$ consistently outperforms both GPTQ/AWQ with a group size of 128 in all scenarios with comparable model sizes. This is more pronounced for 3-bit quantization, where SqueezeLLM with a $0.45 \%$ sparsity level outperforms both GPTQ/AWQ with a group size of 128 by up $\sim 0.3$ perplexity points.

Table 1: Perplexity comparison of LLaMA models quantized into 3 and 4 bits using different methods including RTN, GPTQ, AWQ and SpQR on C4 and WikiText-2. We compare the performance of different methodologies by grouping them based on their model sizes. In the first group, we compare dense-only SqueezeLLM with non-grouped GPTQ. In the second group, we compare SqueezeLLM with a sparsity level of $0.45 \%$ to GPTQ and AWQ with a group size of 128 . For comparison, we add speedup and peak memory usage numbers, which we provide more details in Tab. A. 6 Further results for LLaMA-30/65B can be found in Tab. A.6, and results on other models including LLaMA-2 7/13/70B are provided in Sec. A.7.1

| $\frac{\text { LLaMA-7B }}{\text { Method }}$ | 3-bit |  |  |  |  | 4-bit |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Avg. Bits <br> (comp. rate) | PPL $(\downarrow)$ |  | $\underset{(\uparrow)}{\text { Speedup }}$ | Mem. <br> $(\mathrm{GB}, \downarrow)$ | Avg. Bits <br> (comp. rate) | PPL $(\downarrow)$ |  | $\underset{(\uparrow)}{\text { Speedup }}$ | Mem. <br> $(\mathrm{GB}, \downarrow)$ |
|  |  | C4 | Wiki |  |  |  | C4 | Wiki |  |  |
| Baseline | 16 | 7.08 | 5.68 | $1 \times$ | 12.7 | 16 | 7.08 | 5.68 | $1 \times$ | 12.7 |
| RTN | $3(5.33)$ | 28.26 | 25.61 | $2.3 \times$ | 2.9 | $4(4.00)$ | 7.73 | 6.29 | $2.0 \times$ | 3.7 |
| GPTQ | $3(5.33)$ | 9.55 | 7.55 | $2.3 \times$ | 2.9 | $4(4.00)$ | 7.43 | 5.94 | $2.0 \times$ | 3.7 |
| $\mathrm{SpQR}$ | - | . | - | - | - | $3.94(4.06)$ | 7.28 | 5.87 | $1.2 x^{+}$ | N/A |
| SqueezeLLM | $3.02(5.29)$ | 7.75 | 6.32 | $2.1 \times$ | 2.9 | $4.05(3.95)$ | 7.21 | 5.79 | $1.8 \times$ | 3.8 |
| GPTQ (g 128 , no reorder) $)^{\ddagger}$ | 3.24 (4.93) | 10.09 | 8.85 | $2.0 \times$ | 3.0 | 4.24 | 7.80 | 6.07 | $1.6 \times$ | 3.8 |
| GPTQ (g128) ‡ | 3.24 (4.93) | 7.89 | 6.27 | $0.2 \times$ | 3.0 | 4.24 (3.77) | 7.21 | 5.78 | $0.4 \times$ | 3.8 |
| AWQ (g128) | 3.24 (4.93) | 7.90 | 6.44 | $2.0 x$ | 3.0 | 4.24 (3.77) | 7.22 | 5.82 | $1.6 \times$ | 3.8 |
| SqueezeLLM (0.45\%) | 3.24 (4.93) | 7.56 | 6.13 | $1.9 \times$ | 3.1 | $4.27(3.75)$ | 7.18 | 5.77 | $1.7 \times$ | 4.0 |
| LLaMA-13B | 3-bit |  |  |  |  | 4-bit |  |  |  |  |
| Method | Avg. Bits | PPL | $(\downarrow)$ | Speedup | Mem. | Avg. Bits | PP] | $(\downarrow)$ | Speedup | Mem. |
| retiod | (comp. rate) | C4 | Wiki | $(\uparrow)$ | $(\mathrm{GB}, \downarrow)$ | (comp. rate) | $\mathrm{C} 4$ | Wiki | $(\uparrow)$   | $(\mathrm{GB}, \downarrow)$ |
| Baseline | 16 | 6.61 | 5.09 | $1 \times$ | 24.6 | 16 | 6.61 | 5.09 | $1 \times$ | 24.6 |
| RTN | $3(5.33)$ | 13.24 | 11.78 | $2.7 \times$ | 5.3 | $4(4.00)$ | 6.99 | 5.53 | $2.3 \times$ | 6.8 |
| GPTQ | $3(5.33)$ | 8.22 | 6.22 | $2.7 \times$ | 5.3 | $4(4.00)$ | 6.84 | 5.29 | $2.3 \times$ | 6.8 |
| SpQR |  |  | ![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-08.jpg?height=35&width=68&top_left_y=1201&top_left_x=923) | ![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-08.jpg?height=35&width=107&top_left_y=1201&top_left_x=999) | ![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-08.jpg?height=35&width=90&top_left_y=1201&top_left_x=1114) | $3.96(4.04)$ | 6.72 | 5.22 | $1.2 x^{+}$ | N/A |
| SqueezeLLM | $3.02(5.30)$ | 7.08 | 5.60 | $2.4 \times$ | 5.4 | 4.04 (3.96) | 6.71 | 5.18 | $2.0 \times$ | 6.9 |
| GPTQ (g128, no reorder $)^{\ddagger}$ | $3.25(4.92)$ | 7.16 | 5.53 | $2.2 \times$ | 5.7 | $4.25(3.77)$ | 6.71 | 5.18 | $1.9 \times$ | 7.2 |
| GPTQ (g128) ‡ | $3.25(4.92)$ | 7.12 | 5.47 | $0.2 \times$ | 5.6 | $4.25(3.77)$ | 6.70 | 5.17 | $0.4 \times$ | 7.0 |
| AWQ (g128) | $3.25(4.92)$ | 7.08 | 5.52 | $2.2 \times$ | 5.7 | 4.25 (3.77) | 6.70 | 5.21 | $1.9 \times$ | 7.2 |
| SqueezeLLM ( $0.45 \%)$ | 3.24 (4.94) | 6.92 | 5.45 | $2.2 x$ | 5.8 | 4.26 (3.76) | 6.68 | $\mathbf{5 . 1 7}$ | $1.9 \times$ | 7.3 |

![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-08.jpg?height=344&width=1652&top_left_y=1422&top_left_x=236)

Figure 5: Perplexity comparison PTQ methods for 3-bit LLaMA quantization, evaluated on C4. The x-axes are the relative model sizes with respect to the model size in FP16. Different size-perplexity trade-offs are achieved by adjusting the group size for GPTQ and AWQ and the sparsity level for ours. Our quantization method consistently and significantly outperforms GPTQ and AWQ across all model size regimes, with a more pronounced gap in lower-bit and smaller model sizes.

Results on Larger Models. In Tab. 11(13B) and Tab. A.6 (30/65B), we observe that the trend in 7B extends to larger models, where SqueezeLLM consistently outperforms other PTQ methods across all models and bit widths. Such a trend is also illustrated in Fig. 5 for 3-bit quantization where even dense-only SqueezeLLM achieves comparable perplexity to grouped GPTQ/AWQ\|| With sparsity, we can further improve perplexity, reducing the gap from the FP16 baseline to less than 0.1/0.4 perplexity points for 4/3-bit quantization. Notably, with 3-bit quantization, our approach achieves up to a $2.1 \times$ reduction in perplexity gap from the FP16 baseline compared to existing methods. Further ablation studies on our design choices are provided in Sec. A.4, and additional results on LLaMA2 and OPT can be[^2]

Table 2: Comparison of PTQ methods on zero-shot MMLU accuracy applied to Vicuna v1.1 and v1.3. We add peak memory usage in GB for comparison. Additional results on 5-shot MMLU evaluation can be found in Sec. A.7.2

| Method | Avg. <br> bit | Vicuna-7B (v1.1) |  | Vicuna-13B (v1.1) |  | Vicuna-7B (v1.3) |  | Vicuna-13B (v1.3) |  | Vicuna-33B (v1.3) |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $\operatorname{Acc}(\uparrow)$ | $\operatorname{Mem}(\mathrm{GB} \downarrow)$ | $\operatorname{Acc}(\uparrow)$ | $\operatorname{Mem}(\mathrm{GB} \downarrow)$ | $\operatorname{Acc}(\uparrow)$ | $\operatorname{Mem}(\mathrm{GB} \downarrow)$ | $\operatorname{Acc}(\uparrow)$ | $\operatorname{Mem}(\mathrm{GB} \downarrow)$ | $\operatorname{Acc}(\uparrow)$ | $\operatorname{Mem}(\mathrm{GB} \downarrow$ |
| Baseline | 16 | $39.1 \%$ | 12.7 | $41.2 \%$ | 24.6 | $40.2 \%$ | 12.7 | $43.3 \%$ | 24.6 | $49.5 \%$ | $\mathrm{OOM}$ |
| AWQ (g128) | 4.25 | $38.0 \%$ | 3.8 | $40.4 \%$ | 7.2 | $39.6 \%$ | 3.8 | $42.2 \%$ | 7.2 | $49.5 \%$ | 17.2 |
| SqueezeLLM | 4.05 | $38.8 \%$ | 3.8 | $39.2 \%$ | 6.9 | $39.3 \%$ | $3.8 \quad$ | $44.1 \%$ | 6.9 | $48.0 \%$ | 17.5 |
| SqueezeLLM (0.45\%) | 4.26 | $39.4 \%$ | 4.0 | $41.0 \%$ | 7.3 | $39.5 \%$ | 4.0 | $43.8 \%$ | 7.3 | $49.9 \%$ | 18.7 |
| AWQ (g128) | 3.25 | $36.5 \%$ | 3.0 | $37.6 \%$ | 5.7 | $37.4 \%$ | 3.0 | $40.7 \%$ | 5.7  | $46.4 \%$ | 13.2 |
| SqueezeLLM | 3.02 | $36.0 \%$ | 2.9 | $37.2 \%$ | 5.4 | $35.1 \%$ | 2.9 | $40.5 \%$ | 5.4 | $46.2 \%$ | 12.5 |
| SqueezeLLM ( $0.45 \%)$ | 3.24 | $37.7 \%$ | 3.1 | $39.4 \%$ | 5.8 | $37.6 \%$ | 3.1 | $\mathbf{4 0 . 8 \%}$ | 5.8 | $47.7 \%$ | 14.7 |

![](https://cdn.mathpix.com/cropped/2024_06_04_81b632f2a8dde3752ee9g-09.jpg?height=362&width=1310&top_left_y=702&top_left_x=404)

Figure 6: Comparison of PTQ methods applied to Vicuna v1.1. Blue / yellow / red represent the number of times that the quantized model won / tied / lost against the baseline FP16 model. This evaluation was performed using the methodology from Vicuna.

found in Sec. A.7.1.

### 5.3 Quantization of Instruction Following Models

Instruction tuning has emerged as a method for improving the model's ability to respond to user commands. We explore the quantization of instruction-following models to demonstrate the benefits of SqueezeLLM in terms of accuracy preservation by applying it to the Vicuna models, and evaluating the performance with the following approaches.

MMLU Evaluation. We first evaluate the baseline and quantized models on the MMLU benchmark where the weighted accuracy in the zero-shot setting is provided in Tab. 2 for Vicuna models. As we can see, 3-bit SqueezeLLM achieves higher accuracy for all models compared to AWQ and also preserves the FP16 baseline accuracy with 4-bit quantization. 5-shot results are provided in Sec. A.7.2.

Instruction-Following Ability. Another approach for evaluating instruction-following ability is to ask GPT-4 to rank the generated responses as presented in [6]. As shown in Fig. 6. SqueezeLLM without sparsity achieves near-perfect performance (i.e., 50/50 split) with 4-bit quantization for both Vicuna-7B and 13B, outperforming GPTQ with the same model size. In the case of 3-bit quantization, SqueezeLLM outperforms both GPTQ and AWQ with comparable model sizes. In the case of the Vicuna-13B model, achieving a near-perfect 50/50 split for 3-bit quantization.

### 5.4 Hardware Deployment and Profiling

We show the latency and peak GPU memory usage of SqueezeLLM in Tab. 3 on an A6000 GPU for different configurations when generating 128 tokens. We observe that the LUT-based non-uniform approach in SqueezeLLM (3rd row) shows up to $2.4 \times$ speedup compared to the FP16 baseline, and exhibits comparable latency and peak memory usage to the uniform quantization of non-grouped GPTQ (2nd row). This indicates that the overhead associated with LUT-based dequantization is small, especially considering the significant perplexity gains it enables.

Additionally, when incorporating sparsity, we still observed latency gains relative to the FP16 baseline. As shown in Tab. 3. keeping $0.45 \%$ of parameters in FP16 (4th row) only adds around $10 \%$ latency overhead relative to the dense-only implementation, while still resulting in up to $2.2 \times$ speed up compared to the FP16 baseline. In contrast, when accounting for permutation, the GPTQ runtime is degraded heavily (5th row). This latency penalty is due to

Table 3: Latency (s) and peak memory usage (GB) of 3-bit LLaMA when generating 128 tokens on an A6000 GPU. The table compares the FP16 baseline, non-grouped and grouped GPTQ with activation ordering, and SqueezeLLM with different sparsity levels. For comparison, we include bitwidth and perplexity on the C4 benchmark. See Tab. A. 4 for additional results on generating 1024 tokens, and see Tab. A.5 for additional benchmarking results on an A100 GPU.

| Method | Bit <br> width | PPL (C4) | 7B <br> Lat | Mem (G) | PPL (C4) | 13B <br> Lat (s) | Mem (G) | PPL (C4) | 30B <br> Lat (s) | Mem (G) | PPL (C4) | 65B <br> Lat (s) | Mem (G) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | 16 | 7.08 | 3.2 | 12.7 | 6.61 | 5.6 | 24.6 | 5.98 | OOM | OOM | 5.62 | OOM | OOM |
| GPTQ | 3 | 9.55 | 1.4 | 2.9 | 8.22 | 2.1 | 5.3 | 7.31 | 4.0 | 12.3 | 6.70 | 6.7 | 24.0 |
| SqueezeLLM | 3.02 | 7.75 | 1.5 | 2.9 | 7.08 | 2.4 | 5.4 | 6.37 | 4.0 | 12.5 | 5.99 | 7.6 | 24.5 |
| GPTQ (g128) | 3.25 | 7.89 | 13.7 | 3.0 | 7.12 | 24.2 | 5.6 | 6.47 | 61.9 | 12.9 | 6.01 | 117.8 | 25.1 |
| SqueezeLLM $(0.45 \%)$ | 3.24 | 7.56 | 1.7 | 3.1 | 6.92 | 2.5 | 5.8 | 6.23 | 4.4 | 14.7 | 5.84 | 8.8 | 28.0 |

permutation, which means that elements in the same channel need to be scaled using different scaling factors (which are accessed using group indices); it is challenging for these distributed memory accesses to be performed efficiently, as GPUs rely heavily on coalesced memory accesses in order to optimally utilize memory bandwidth. This shows how our Dense-and-Sparse quantization methodology allows for both higher accuracy as well as better performance relative to GPTQ. Additional evaluation results on generating 1024 tokens are provided in Tab. A.4, where we can observe a similar trend.

## 6 Conclusion

We have presented SqueezeLLM which attempts to address the Memory Wall problem associated with generative LLM inference that is memory-bound. SqueezeLLM incorporates two novel ideas that allow ultra-low precision quantization of LLMs with negligible degradation in generation performance: the sensitivity-based non-uniform quantization method; and the Dense-and-Sparse decomposition that resolves the outlier issue. We have evaluated SqueezeLLM on a wide range of models and datasets that assess language modeling, problem-solving, and instruction-following capabilities of quantized models, where we have demonstrated that our quantization method can consistently outperform the previous state-of-the-art methodologies.

## Acknowledgements

The authors would like to acknowledge Karttikeya Mangalam, Nicholas Lee, and Thanakul Wattanawong for helpful discussions and brainstorming. We acknowledge gracious support from Google Cloud, Google TRC team, and specifically Jonathan Caton, Jing Li, Jiayu Ye, and Prof. David Patterson. Prof. Keutzer's lab is sponsored by Intel corporation, Intel VLAB team, Intel One-API center of excellence, as well as gracious funding from Furiosa, Berkeley Deep Drive, and BAIR. Our conclusions do not necessarily reflect the position or the policy of our sponsors, and no official endorsement should be inferred.

## References

[1] Haoli Bai, Wei Zhang, Lu Hou, Lifeng Shang, Jing Jin, Xin Jiang, Qun Liu, Michael Lyu, and Irwin King. BinaryBERT: Pushing the limit of BERT quantization. arXiv preprint arXiv:2012.15701, 2020.

[2] Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Understanding and overcoming the challenges of efficient Transformer quantization. arXiv preprint arXiv:2109.12948, 2021.

[3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

[4] Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. ZeroQ: A novel zero shot quantization framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13169-13178, 2020.

[5] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. Advances in Neural Information Processing Systems, 34:17413-17426, 2021.

[6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality, March 2023.

[7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PALM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

[8] Insoo Chung, Byeongwook Kim, Yoonjung Choi, Se Jung Kwon, Yongkweon Jeon, Baeseong Park, Sangha Kim, and Dongsoo Lee. Extremely low bit transformer quantization for on-device neural machine translation. arXiv preprint arXiv:2009.07453, 2020.

[9] Jyotikrishna Dass, Shang Wu, Huihong Shi, Chaojian Li, Zhifan Ye, Zhongfeng Wang, and Yingyan Lin. Vitality: Unifying low-rank and sparse approximation for vision transformer acceleration with a linear taylor attention, 2022 .

[10] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems.

[11] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.

[12] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. SpQR: A sparse-quantized representation for near-lossless LLM weight compression. arXiv preprint arXiv:2306.03078, 2023.

[13] Zhen Dong, Zhewei Yao, Daiyaan Arfeen, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. HAWQ-V2: Hessian Aware trace-Weighted Quantization of neural networks. NeurIPS'19 workshop on Beyond First-Order Optimization Methods in Machine Learning., 2019.

[14] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. GLAM: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pages 5547-5569. PMLR, 2022.

[15] Georgii Evtushenko. Sparse Matrix-Vector Multiplication with CUDA. https://medium.com/analyticsvidhyalsparse-matrix-vector-multiplication-with-cuda-42d191878e8f, 2019.

[16] Goran Flegar and Enrique S Quintana-Ortí. Balanced csr sparse matrix-vector product on graphics processors. In Euro-Par 2017: Parallel Processing: 23rd International Conference on Parallel and Distributed Computing, Santiago de Compostela, Spain, August 28-September 1, 2017, Proceedings 23, pages 697-709. Springer, 2017.

[17] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

[18] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.

[19] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. arXiv preprint arXiv:2103.13630, 2021.

[20] Amir Gholami, Zhewei Yao, Sehoon Kim, Michael W Mahoney, and Kurt Keutzer. AI and Memory Wall. https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8, 2021.

[21] GPTQ-For-LLaMA. https://github.com/qwopqwop200/gptq-for-llama.

[22] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. International Conference on Learning Representations, 2016.

[23] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[24] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[25] Yafeng Huang, Huanrui Yang, Zhen Dong, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Yuan Du, Shanghang Zhang, and Kurt Keutzer. Output sensitivity-aware detr quantization. 2023.

[26] Yongkweon Jeon, Chungman Lee, Eulrang Cho, and Yeonju Ro. Mr. BiQ: Post-training non-uniform quantization based on minimizing the reconstruction error. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12329-12338, 2022.

[27] Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. I-BERT: Integer-only bert quantization. arXiv preprint arXiv:2101.01321, 2021.

[28] Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, Minwoo Kang, Ruohan Yan, Hasan Genc, Grace Dinh, Qijing Huang, Kurt Keutzer, Michael W Mahoney, Sophia Shao, and Amir Gholami. Full stack optimization of transformer inference: a survey. arXiv preprint arXiv:2302.14017, 2023.

[29] Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and Anna Rumshisky. Bert busters: Outlier dimensions that disrupt transformers. arXiv preprint arXiv:2105.06990, 2021.

[30] Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In Advances in neural information processing systems, pages 598-605, 1990.

[31] Xiuyu Li, Long Lian, Yijiang Liu, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. arXiv preprint arXiv:2302.04304, 2023.

[32] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration. 2023.

[33] Yijiang Liu, Huanrui Yang, Zhen Dong, Kurt Keutzer, Li Du, and Shanghang Zhang. NoisyQuant: Noisy bias-enhanced post-training activation quantization for vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20321-20330, 2023.

[34] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.

[35] Sangyun Oh, Hyeonuk Sim, Jounghyun Kim, and Jongeun Lee. Non-uniform step size quantization for accurate post-training quantization. In Computer Vision-ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XI, pages 658-673. Springer, 2022.

[36] David A Patterson. Latency lags bandwith. Communications of the ACM, 47(10):71-75, 2004.

[37] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020.

[38] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

[39] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Q-BERT: Hessian based ultra low precision quantization of bert. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8815-8821, 2020.

[40] Gil Shomron, Freddy Gabbay, Samer Kurzum, and Uri Weiser. Post-training sparsity-aware quantization. Advances in Neural Information Processing Systems, 34:17737-17748, 2021.

[41] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.

[42] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.

[43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[44] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[45] Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. Outlier suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling. arXiv preprint arXiv:2304.09145, 2023.

[46] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. arXiv preprint arXiv:2209.13325, 2022.

[47] Yuhui Xu, Yongzhuang Wang, Aojun Zhou, Weiyao Lin, and Hongkai Xiong. Deep neural network compression with single and multiple level quantization, 2018.

[48] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers. arXiv preprint arXiv:2206.01861, 2022.

[49] Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu Sun, Qiang Wu, Jiaxiang Wu, and Bingzhe Wu. RPTQ: Reorder-based post-training quantization for large language models. arXiv preprint arXiv:2304.01089, 2023.

[50] Ali Hadi Zadeh, Isak Edo, Omar Mohamed Awad, and Andreas Moshovos. GOBO: Quantizing attention-based nlp models for low latency and energy efficient inference. In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 811-824. IEEE, 2020.

[51] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. Q8BERT: Quantized 8bit bert. arXiv preprint arXiv:1910.06188, 2019.

[52] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

[53] Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, and Qun Liu. TernaryBERT: Distillationaware ultra-low bit bert. arXiv preprint arXiv:2009.12812, 2020.

[54] Yifan Zhang, Zhen Dong, Huanrui Yang, Ming Lu, Cheng-Ching Tseng, Yandong Guo, Kurt Keutzer, Li Du, and Shanghang Zhang. Qd-bev: Quantization-aware view-guided distillation for multi-view 3d object detection. 2023.

[55] Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Chris De Sa, and Zhiru Zhang. Improving neural network quantization without retraining using outlier channel splitting. In International conference on machine learning, pages 7543-7552. PMLR, 2019.
</end of paper 3>


<paper 4>
# QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models 

Elias Frantar $^{1}$ Dan Alistarh ${ }^{12}$


#### Abstract

Mixture-of-Experts (MoE) architectures offer a general solution to the high inference costs of large language models (LLMs) via sparse routing, bringing faster and more accurate models, at the cost of massive parameter counts. For example, the SwitchTransformer-c2048 model has 1.6 trillion parameters, requiring $3.2 \mathrm{~TB}$ of accelerator memory to run efficiently, which makes practical deployment challenging and expensive. In this paper, we present a solution to this memory problem, in form of a new compression and execution framework called QMoE. Specifically, QMoE consists of a scalable algorithm which accurately compresses trillion-parameter MoEs to less than 1 bit per parameter, in a custom format co-designed with bespoke GPU decoding kernels to facilitate efficient end-to-end compressed inference, with minor runtime overheads relative to uncompressed execution. Concretely, QMoE can compress the 1.6 trillion parameter SwitchTransformer-c2048 model to less than $160 \mathrm{~GB}$ ( $20 \mathrm{x}$ compression, 0.8 bits per parameter) at only minor accuracy loss, in less than a day on a single GPU. This enables, for the first time, the execution of a trillion-parameter model on affordable commodity hardware, like a single server with $4 x$ NVIDIA A6000 or 8x NVIDIA 3090 GPUs, at less than $5 \%$ runtime overhead relative to ideal uncompressed inference. The source code and compressed models are available at github.com/IST-DASLab/qmoe.


## 1. Introduction

Generative large language models (LLMs), e.g. (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023a;b), have garnered significant industrial and popular attention due to their surprising performance across many practical language and reasoning tasks. Yet, a major obstacle to broad deployment is given by their extremely high inference costs. One particularly promising approach for reducing these costs is the use of Mixture-of-Experts (MoE) architec-[^0]

tures, e.g. (Fedus et al., 2022; Artetxe et al., 2022), whose general idea is to replicate certain model components many times while routing each input only to a small subset of those replicas. Through expert "specialization" to input subsets, MoEs achieve faster inference for the same model quality, but with significantly higher memory costs due to components being replicated hundreds or even thousands of times, for the largest and best-performing models.

For example, the popular SwitchTransformer family (Fedus et al., 2022), which we focus on in this study, uses between 128 and 2048 experts (layer replicas) to significantly outperform standard dense T5 models (Raffel et al., 2020b) in terms of inference and training costs, at equivalent model accuracy. Artetxe et al. (2022) report similar improvements, on different tasks, for 512 experts. However, these results come at the cost of dramatic increases in model size: the largest SwitchTransformer has 1.6 trillion parameters, requiring 3.2TB of storage in standard half-precision, and correspondingly requires a hundred or more expensive (GPU or TPU) accelerators for efficient usage. This not only makes practical deployment costly and challenging, but also strongly limits research on such models.

Challenges. It is natural to ask whether the truly massive memory costs of such MoEs can be reduced via standard techniques for model compression, such as quantization (Gholami et al., 2021) or sparsity (Hoefler et al., 2021), without significant accuracy loss. Achieving this would require overcoming conceptual and technical barriers:

1. Conceptually, existing post-training compression methods, whose costs would be affordable enough to execute on such models, are currently only able to reduce precision to 3 or 4 bits per parameter (Frantar et al., 2022; Dettmers \& Zettlemoyer, 2022; Wu et al., 2023) or around $50 \%$ sparsity (Frantar \& Alistarh, 2023), before significant accuracy loss occurs. Yet, making trillion-parameter MoEs practical would require compression rates between $10 \times$ and $20 \times$ relative to 16 -bit precision, i.e., on average less than 1 bit per parameter.
2. A key practical issue is scaling: applying state-of-theart compression methods, designed for large dense models, to MoEs that are an order of magnitude larger, while maintaining affordability, runs into a plethora of memory, performance and reliability roadblocks.
3. Actually achieving sub-1-bit compression would re-
quire a non-trivial custom compression format. Such a format would also need to come with decoding algorithms that are highly-efficient on accelerators such as GPUs, in order to run inference on compressed models without major processing slowdowns.

Contribution. In this paper, we overcome these challenges, and introduce QMoE, a framework for accurate compression and fast compressed inference of massive MoEs, reducing model sizes by $10-20 \times$, to less than 1 bit per parameter. QMoE is specifically designed to compress and subsequently inference with models like the 1.6 trillion parameter SwitchTransformer-c2048, using only modest computational resources.

Our key technical contributions are a highly scalable compression algorithm implementation and a customized compression format designed together with bespoke GPUkernels for fast on-the-fly decoding. Further, we show for the first time that accurate sub-1-bit compression of trillion parameter MoEs is feasible and can be achieved via affordable retraining-free compression techniques.

Concretely, we reduce the size of SwitchTransformer-c2048, the largest openly-available model, from 3.2TB in bfloat16 to less than $160 \mathrm{~GB}$ in our customized compressed format, that is, $\approx 0.8$ bits per parameter, at only a minor increase in loss on pretraining validation and zero-shot data. Using our QMoE kernels, this compressed model can then be executed fully, without any slow offloading, on commodity hardware such as $8 \times$ NVIDIA RTX 3090 or $4 \times$ NVIDIA A6000 GPUs, with $<5 \%$ runtime overhead relative to an idealized version of uncompressed execution, which would require $\approx 20 \times$ more GPUs.

In summary, our work enables, for the first time, the performant execution of massive-scale MoE models on commodity hardware. This is illustrated by the fact that we are able to efficiently run the trillion-parameter SwitchTransformerc2048 model on a single commodity GPU server, with minor accuracy loss. This addresses one of the key limitations behind MoE architectures, and should improve their practical adoption as well as facilitate further research on understanding and improving such models.

## 2. Background

### 2.1. Mixture of Expert Models (MoEs)

The core idea behind Mixture of Expert models (MoEs) is to increase the number of parameters, and thus the network's modelling power, while at the same time keeping compute costs near-constant, relative to a standard feedforward architecture. This is typically achieved by creating many copies of certain model components, each of which is responsible for processing only a subset of all input tokens. The corresponding input-to-component assignments are generally decided by a "router" layer. Probably the most common MoE design (Fedus et al., 2022; Artetxe et al., 2022), which we also focus on in this paper, is to replicate the fully-connected module of a Transformer and route tokens to the replica, referred to as an expert, with the highest assignment score predicted by a linear routing layer; see Figure 1 for an illustration. This design enables efficient training and inference of extremely large models, using 100s or even 1000s of experts/, since each token is processed only by a small subset of the massive overall network.

![](https://cdn.mathpix.com/cropped/2024_06_04_2bf28cb8b86df30ffa9ag-02.jpg?height=325&width=729&top_left_y=683&top_left_x=1099)

Figure 1. Example of an MoE Transformer block. Each token is routed to a different fully-connected (FC) block.

MoEs have been shown to bring substantial accuracy and training speed improvements for equivalent inference speed (Clark et al., 2022; Du et al., 2022; Zoph et al., 2022). However, their current practicality is limited since they are extremely large in size and thus require massive amounts of accelerator memory to be executed efficiently.

### 2.2. Data-dependent Quantization

The currently most effective strategy for reducing model size and corresponding memory costs is quantization, i.e., converting model weights to lower numerical precision. On large models (Dettmers et al., 2022; Dettmers \& Zettlemoyer, 2022), in particular also MoEs (Kim et al., 2022b; Yi et al., 2023), just simple rounding can decrease precision to 8 or even 4 bits per weight, at minimal accuracy loss relative to the standard half (16-bit) precision employed for these models. However, some MoEs are so large that reduction rates significantly higher than $4 \times$ (accomplished by 4 -bit) would be required to render them practical. Accurately quantizing models to extremely low precision (e.g., lower than 3 bits per parameter) typically requires more sophisticated data-dependent methods (Nagel et al., 2020; Wang et al., 2020; Hubara et al., 2021).

Such data-dependent quantization methods use a small set of calibration data, which is passed through the model. As this happens, for each linear layer $\ell$ with weights $W_{\ell}$, quantized weights $Q_{\ell}$ are determined one-by-one. Specifically, one approach to do this is by solving a layer-wise quantization problem, stated with respect to $W_{\ell}$ and the observed calibration data inputs $X_{\ell}$ at the current layer:

$$
\begin{equation*}
\operatorname{argmin}_{Q_{\ell}}\left\|Q_{\ell} X_{\ell}-W_{\ell} X_{\ell}\right\| . \tag{1}
\end{equation*}
$$

Various solvers for Equation (1) have been proposed, with some optimized, in terms of speed and accuracy, particularly for extremely large models, like GPTQ (Frantar et al., 2022) or ZeroQuant (Yao et al., 2022; Wu et al., 2023). The former performs quantization using second-order information in the layer-wise Hessian matrix $X_{\ell} X_{\ell}^{\top}$, while the latter applies SGD-optimization with straight-through gradient estimation (Bengio et al., 2013).

Another noteworthy characteristic of many such methods is that per-layer quantization can be performed sequentially, using the input from the already partially quantized model up to layer $\ell-1$, when quantizing layer $\ell$, serving to reduce error accumulation. Concretely, this can be efficiently implemented by using $X_{\ell}$ to find $Q_{\ell}$ before passing on $X_{\ell+1}=Q_{\ell} X_{\ell}$ to the next layer.

### 2.3. MoE Quantization

There are several aspects which make very-low-bit, e.g. ternary (3 values) quantization promising for MoE models:

- In many architectures, almost all parameters are located in the experts, as they are 1000s of them. This means that, for size reduction, it suffices to focus on compressing just those experts and leave other layers in standard precision. This reduces error accumulation since only a subset of modules involved in a forward pass are actually quantized.
- Previous work has observed that extremely large dense models are more resistant to quantization noise than smaller ones (Frantar et al., 2022; Chee et al., 2023). Large MoEs can be much larger than some of these massive dense models, and are thus a prime target for accurate quantization.
- MoE training involves additional stochasticity through routing instabilities and strategies like token dropping (Lepikhin et al., 2020), which may inherently encourage high resistance to noise. Finetuning is also often performed with high dropout (Fedus et al., 2022).

Our experiments in Section 5.2 confirm that MoEs are indeed highly robust to extreme levels of quantization.

## 3. Scaling Data-dependent Quantization to Trillion Parameter MoEs

### 3.1. Challenges

While data-dependent quantization techniques have already been used to successfully compress large dense models up to 176 billion parameters (Frantar et al., 2022; Wu et al., 2023), applying them to sparse mixture-of-expert models another order of magnitude larger brings several new challenges.

Memory Costs. The first major problem we encounter is a large increase in the memory required to apply such techniques. Not only are the original model weights nearly $10 \times$ larger, but the quantization process itself also needs $>100 \times$ more data. The latter constraint is because accurate datadependent quantization methods require a sufficient number of input samples for each layer that is being compressed. For very large dense models, a few hundreds of thousands of "calibration tokens" typically suffice (Frantar et al., 2022; Yao et al., 2022). However, in MoEs with thousands of layers, a single expert processes only a small subset of all inputs, hence we need much more tokens overall to achieve good coverage of all experts. Further, in encoder-decoder architecture models, like SwitchTransformers, each token is processed only by half of the model, again increasing data requirements. For fast compression, we must maintain intermediate results for the full calibration dataset, which requires 100s of GBs of memory for the largest models.

GPU Utilization. The next significant challenge is that existing large-scale quantization implementations, in particular for GPTQ and related methods (Frantar et al., 2022; Chee et al., 2023), are designed to be fast and memory efficient for the massive individual layers occurring in dense models. Meanwhile, MoEs typically have smaller layers, but $100 \times$ to $1000 \times$ more of them. Current implementations have poor GPU utilization in this case, and consequently bad performance. A similar issue occurs if activations and weights have to be transferred between CPU and GPU with high frequency, which may be required to cope with the massive memory requirements discussed previously.

Reliability Requirements. Finally, another issue when compressing models with tens of thousands of layers is that running into rare edge cases, which may break the process, is highly likely. This is includes numerical problems like noninvertible layer-wise Hessians, as well as model-specific ones, e.g., extreme routing patterns on particular layers.

### 3.2. System Design \& Optimizations

In this section, we describe system-level design and optimizations to address the challenges in Section 3.1. This allows us to apply data-dependent compression to massive MoEs, while preserving the key feature of post-training compression techniques: the ability to perform effective compression using only modest computational resources, e.g., a single NVIDIA A6000 GPU and less than one day of compute. Although we focus on scaling the popular GPTQ method, most techniques described below will generalize to other data-dependent quantization approaches, like ZeroQuant (Yao et al., 2022), as well.

### 3.2.1. OPTIMIZED ACTIVATION OFFLOADING

As discussed in Section 3.1, a key challenge in compressing MoEs is that we need to maintain massive activation sets. Yet, it is possible to carefully orchestrate model execution in

![](https://cdn.mathpix.com/cropped/2024_06_04_2bf28cb8b86df30ffa9ag-04.jpg?height=347&width=1288&top_left_y=236&top_left_x=386)

Figure 2. Illustration of the offloading execution for the sparse part of a Transformer block. An expert $E_{2}$ and its corresponding input tokens $X_{E}$ are fetched to GPU memory to produce $E_{2}^{\prime}$, which together with the corresponding outputs $Y_{E}$ are written back to CPU again.

such a way that we only ever need to perform computation on a small subset of the intermediate data. This allows us to offload main storage from GPU, to much less expensive and plentiful CPU memory.

Concretely, we maintain a single large buffer $B$ which we update as follows, for the dense part of a Transformer block:

1. Fetch one "sample" $X$, containing a few hundreds of tokens, from CPU to GPU.
2. Pass it through the corresponding dense layers to obtain the result $Y$.
3. Calculate and store expert assignment for tokens in $Y$.
4. Send $Y$ back to CPU and overwrite $X$ in $B$.

and respectively for the sparse part, looping over experts:

1. Fetch all individual tokens in $B$ that have been assigned to expert $E$, denoted by $X_{E}$, from CPU to GPU.
2. Use them to produce compressed expert $E^{\prime}$ (for example, with GPTQ).
3. Run $X_{E}$ through $E^{\prime}$ to get $Y_{E^{\prime}}$.
4. Send $Y_{E^{\prime}}$ back to CPU and overwrite $X_{E}$ in $B$.

This process, which is visualized in Figure 2, minimizes both memory consumption and transfer cost: we need only a single copy of $B$ and each token is only read and written twice per Transformer block.

![](https://cdn.mathpix.com/cropped/2024_06_04_2bf28cb8b86df30ffa9ag-04.jpg?height=258&width=705&top_left_y=2137&top_left_x=249)

Figure 3. List buffer example with 3 samples, indicated by hue.

### 3.2.2. LIST BUFFER

To efficiently support per-sample access for evaluating dense model components, as well as fully-vectorized querying of expert tokens, we store $B$ as a list buffer data structure. This can be seen as a huge contiguous buffer of all token hidden states, together with delimiter indices denoting boundaries between individual samples. Figure 3 illustrates this storage format. This datastructure is crucial for efficiency; naively iterating over samples and fetching relevant tokens via masking is unusably slow for large sample counts.

### 3.2.3. LAZY WEIGHT FETCHING

Since the weights of the 1.6 trillion parameter model consume $>3 \mathrm{~TB}$ of storage, they cannot even be stored in CPU RAM. Thus, we lazily fetch them directly from disk storage as they are required. If we follow the inference procedure outlined in Section 3.2.1, this would be exactly once. Afterwards, their memory is released again.

### 3.2.4. EXPERT GROUPING

Additionally, in order to avoid GPU underutilization (see Section 3.1), we group multiple experts together and apply a joint batched variant of the GPTQ algorithm. Concretely, we extract the inputs $X_{E}$ corresponding to all experts $E \in \mathcal{E}$ in group $\mathcal{E}$ (the $X_{E}$ will generally have different sizes) and compute Hessians $H_{E}$. These matrices, together with the weight matrices $W_{E}$, are then stacked to 3-dimensional tensors, on which our modified GPTQ algorithm operates, compressing all experts simultaneously. We can also compute $H_{E}=X_{E} X_{E}^{\top}$ directly with a single matmul as the $X_{E}$ are generally small enough, avoiding the slow per-sample accumulation employed by prior implementations. Our default expert groupsize $|\mathcal{E}|$ is 16 , which brings a good trade-off between GPU memory consumption and utilization.

Table 1 demonstrates the impact of expert grouping via GPTQ batching, when compressing a sparse encoder layer of switch-base-128 using $10 \mathrm{k}$ samples; $|\mathcal{E}|=16$ yields about $\approx 6 \times$ speedup over standard per-expert computation.

| $\|\mathcal{E}\|=1$ | $\|\mathcal{E}\|=4$ | $\|\mathcal{E}\|=16$ |
| :---: | :---: | :---: |
| $174.1 \mathrm{~s}$ | $54.4 \mathrm{~s}$ | $\mathbf{2 8 . 8 s}$ |

Table 1. Sparse layer compression time for different $|\mathcal{E}|$.

### 3.2.5. RoBUSTNESS MODIFICATIONS

To achieve sufficiently high robustness for successfully quantizing trillion parameter models with tens of thousands of layers, we need to employ various numerical and memory adjustments. The most important are listed below:

- We use $10 \times$ higher relative Hessian dampening $\delta=$ 0.1 , avoiding breakdowns with inf-values.
- Very few layer Hessians are not invertible even after high dampening; we skip GPTQ for those and simply perform vanilla rounding.
- Sometimes an expert receives a number of tokens that is much larger than average, leading to out-of-memory situations when these are fetched to GPU. We avoid this by capping the maximum number of tokens used for compression at $4 \times$ the mean and use multiple iterations for computing and updating $Y_{E}$ in such cases.


### 3.3. Accuracy Improvements

In addition to implementing a highly efficient compression system, we also make new discoveries about applying GPTQ in our particular context, i.e., for models trained for maskedlanguage-modelling, MoEs and ternary quantization.

Premasking Special Tokens. First, we find that results can be improved if the various special separator tokens inserted by the masked-language-modelling task (Raffel et al., 2020b) are excluded from the calibration data used for compression. Conretely, in the encoder, we mask out those "mask-tokens" during the Hessian computation. Meanwhile, in the decoder, we skip the token directly before such a special token as this is the one used to predict the latter.

As shown in Table 2 for switch-base-128 with 10k samples, this brings noticeably lower loss at no additional compute cost. We think that because those tokens are very common during training, the model is so robust in their prediction that any error compensation on them during quantization is unnecessary, while worsening correction for other tokens.

| mask | BF16 | 2bit | tern |
| :---: | :---: | :---: | :---: |
| no | 1.73 | 1.86 | 2.16 |
| yes | 1.73 | $\mathbf{1 . 7 6}$ | $\mathbf{1 . 9 9}$ |

Table 2. Impact of special token masking; validation loss.

Ineffective Heuristics. We also evaluate two more recently proposed GPTQ enhancement heuristics: activation reorder- ing and true sequential execution (Frantar et al., 2023). However, as shown in Table 3 for ternary quantization of switchbase-128, we find the former to be actually harmful and the latter to be more or less quality neutral, for our particular use-case. We suspect that, in this highly aggressive setting, quantizing all the most sensitive columns first, leads to large changes of the entire weight matrix, and thus to overfitting.

| GPTQ | act | seq | act + seq |
| :---: | :---: | :---: | :---: |
| $\mathbf{1 . 9 9}$ | 2.23 | $\mathbf{1 . 9 9}$ | 2.28 |

Table 3. Activation reordering (act) and sequential execution (seq).

## 4. Realizing Sub-1-Bit Compression

Using our system discussed in Section 3, we can accurately quantize extremely large SwitchTransformers to very low bit-widths: 2-bit and even ternary (3 possible values). Yet, in practice, this falls still short of our compression goal of less than 1 bit per parameter. We find that compression rates can be pushed significantly further by taking advantage of the low entropy in the quantized weights. Next, we co-design an encoding scheme and a CUDA kernel which realize sub1-bit per weight compression in practice, at minimal cost in terms of GPU execution overhead for inference.

### 4.1. Natural Sparsity

We pick quantization grids in standard fashion: row-wise around the min and max weights values (Dettmers et al., 2022; Frantar et al., 2022), e.g., for ternary: $\left\{w_{\min }, 0, w_{\max }\right\}$. These rather wide grids combined with the fact that weights are typically close to normally distributed, naturally lead to high sparsity after quantization, i.e., a large number of zeros. We demonstrate this in Table 4, averaged over all layers. For ternary weights, the largest model achieves close to $90 \%$ natural sparsity; the standard deviation is also quite low, at $<5 \%$. Seen another way, the quantized weights have low entropy, meaning that, on average, significantly less bits per weight should be required for lossless storage.

| model | 2-bit | ternary |
| :---: | :---: | :---: |
| base128 | $72.2 \%$ | $85.7 \%$ |
| large128 | $73.1 \%$ | $86.4 \%$ |
| c2048 | $76.5 \%$ | $88.6 \%$ |

Table 4. Natural sparsity for different compressed models.

### 4.2. From Sparsity to Entropy

The direct way of utilizing these high zero proportions would be in form of a joint sparse \& quantized representation (Kurtic et al., 2022; Yu et al., 2023): storing only the quantized values of non-zero weights, together with
necessary position metadata. However, as our base quantization levels are already very low, standard sparsity metadata formats (Elsen et al., 2020; Lin et al., 2023) would only allow limited additional compression. A bitmask indicating non-zero locations requires 1 bit per weight, while 10-13 bit (depending on layer size) column indices are even less memory efficient at the sparsity levels we encounter. Therefore, we take a different approach: we do not utilize sparsity directly but rather the low entropy, which is implied by the fact that a single value (0) occurs very frequently, using an appropriate encoding scheme.

### 4.2.1. FASt GPU DECODing ChALLENGES

In principle, we could group multiple consecutive ternary weights into super-symbols and then apply a code which assigns variable length codewords to those super-symbols, based on their probability of occurrence, for example, via a Huffman approach (Huffman, 1952). If the quantized weight values were close to independent, this would achieve strong compression rates; in fact, for actual independence, they would be essentially Shannon-optimal (MacKay, 2003).

At the same time, our primary goal is to use compressed models for fast and space-efficient inference. Thus, it is critical not only that our encoding scheme achieves good compression, but also that it can be decoded fast on GPU hardware. This is challenging for a number of reasons:

Challenge 1: Entropy-based codes generally possess sequential decoding dependencies: symbol $i$ can only be determined if the length, which is variable, of all $(i-1)$ prior symbols is known. Hence, processing consecutive symbols simultaneously leads to high synchronization overhead.

Challenge 2: Binary words in storage (e.g., INT32 blobs) may contain different numbers of decoded symbols. Consequently, even if rows/blocks are encoded independently, parallel decoding will happen non-uniformly, while all threads in a GPU-warp must always execute the same instruction. This would result in many wasted operations.

Challenge 3: Variable-length low-bit decoding involves a large number of binary operations like shifts, which are not particularly efficient on GPUs.

Challenge 4: Individual matrices of MoEs are typically not very large, making it difficult to split them into enough separately decoded segments to achieve good GPU utilization without having to store additional data to break sequential dependencies, which would harm compression rates.

In contrast, uncompressed half-precision matrix-vector products, which are the primary operation underlying generative inference, easily achieve close to ideal memory-bandwidth utilization and thus present a very strong baseline.

### 4.3. Compression Scheme \& Kernel Co-design

To achieve our goal, we need to design a compression scheme and its GPU decoding kernel jointly, and potentially trade off compression for faster decoding. We begin with an overview of the main ideas behind our approach, followed by an in-depth discussion of key details.

### 4.3.1. OVERVIEW

Instead of a code with variable length codewords (see Section 4.2.1) mapping to fixed length data, we will use a dictionary-based code with fixed length codewords mapping to a variable number of symbols. Such LZW-based schemes (Welch, 1984) are popular for general purpose compression like ZIP, as they are particularly effective for text data with long repeated segments. While a dictionary code is not ideal in terms of compression rate for the case of almost-random data in our application, it will be key for fast GPU decoding.

First, our kernel design uses one warp, that is 32 consecutive threads, to handle a row of a weight matrix, each of which is encoded independently. This addresses Challenge 4 in Section 4.2.1, yielding reasonable GPU utilization for relevant matrix sizes, with negligible metadata overhead. Further, we use a fixed-to-variable code with a large dictionary. This allows us to use a full warp to process one codeword at-atime, extracting all data, while maintaining good efficiency, thus working around Challenges 1 and 2. This way, slow bit and base-3 operations (for ternary) can also be kept at a minimum, resolving Challenge 3.

### 4.3.2. DICTIONARY DESIGN AND IMPLEMENTATION

In general, assume that the values of a ternary weight matrix (denoted by $0,1,2$ ) are distributed close to independently according to the distribution:

$$
\begin{equation*}
P(0)=p_{0}, \quad P(1)=P(2)=\frac{1-p_{0}}{2} \tag{2}
\end{equation*}
$$

where $p_{0}$ denotes the probability of sampling 0 , e.g., 0.885 as per Table 4. Since we plan to use a rather large dictionary, it should be shared between many weight matrices, in order for the dictionary itself not to cause substantial storage overheads. We find that such a static dictionary works well enough, while simplifying memory efficient compression (see Section 3.2) as we do not have to collect statistics over many yet uncompressed experts.

Next, we consider pairs of ternary values $t=\left(t_{1}, t_{2}\right)$, whose corresponding probability is $P(t)=P\left(t_{1}\right) P\left(t_{2}\right)$. We generate the $2^{16}$ highest probability sequences containing at most 14 such pairs. This dictionary can be generated using a max-priority queue on probability, as shown by Algorithm 1.

To briefly understand the procedure, notice that upon the first iteration, it will push all individual pairs $t=\left(t_{1}, t_{2}\right)$ to the priority queue, sorting them by decreasing probability,

```
Algorithm 1 Generate decoding dictionary sequences.
    $Q \leftarrow$ max priority queue containing $(1.0,())$
    while $|D|<2^{16}$ do
        $p, s \leftarrow \operatorname{pop}(Q)$
        append $s$ to dictionary if $0<|s|<28$
        for $t \in\left\{\left(t_{1}, t_{2}\right) \mid t_{1}, t_{2} \in\{0,1,2\}\right\}$ do
            $\operatorname{push}((p \cdot P(t), \operatorname{cat}(s, t)), Q)$
        end for
    end while
```

after which they will be expanded in this order.

We have exactly $2^{16}$ codewords as this allows us to store them in the native UINT16 datatype, avoiding any slow bitextractions at this decoding level. Each of those codewords maps to two consecutive UINT32 values containing up to 7 pairs each, stored using 2 bits per ternary value, followed by the total number of pairs in the sequence; see also Figure 4. This format dictates our maximum chosen pair count of 14 . Further, we consider pairs, rather than individual weights, to fit the maximum count into 4 bits. The 2-bit-per-weight format is used as there is enough space, while a more compact ternary encoding would involve slow modulo and division operations for extraction. We store the pair-count twice so that each thread can work with only half of the data, stored in a fast INT32 type.

![](https://cdn.mathpix.com/cropped/2024_06_04_2bf28cb8b86df30ffa9ag-07.jpg?height=290&width=767&top_left_y=1378&top_left_x=213)

Figure 4. Data format of a dictionary entry; here of 24 weights.

Overall, mapping 16-bit codewords to 64-bit data blobs strikes a good balance between several goals: (a) Having codewords map to, on average, more uncompressed values than their bitwidth, a necessary condition for achieving $<1$ bit compression. (b) Minimizing the overall storage cost of the dictionary to fit into the L2-cache of the GPU, which is critical for good decoding performance. (c) Utilizing as many threads in a warp as possible for simultaneously extracting plain weights from the decoded data; usually, $>16$ will do useful work and only 4 out of 32 threads are never active in this step. (d) Avoiding as many conditionals and extra operations necessary for dealing with non-uniform data storage as possible, which slow down parallelization.

Finally, we note that while dictionary lookups are in principle random access, keeping it sorted from highest to lowest probability ensures very favorable caching behavior. Since each lookup also automatically prefetches several subse- quent elements, and most lookups are for frequently occurring codewords, there are many fast L1-cache hits.

Validation. To assess the effectiveness of our scheme, we compute achieved compression rates, both on a real ternary quantized c2048 model as well as on weight matrices sampled directly from distribution (2), yielding $20.07 \times$ and $21.11 \times$, respectively. This gap of only $\approx 5 \%$ suggests that our simplifying independence assumption is indeed quite close for large models. We also note that our rates are only $\approx 20 \%$ away from the distribution's (with $p=0.885$ ) theoretical compression limit of $25.40 \times$, which we consider a reasonable trade-off for enabling fast GPU decoding.

### 4.3.3. GPU KERNEL

Having defined the dictionary format, we can now discuss the design of the actual decoding kernel in detail. We focus on the most important operation for inference, decompression fused with a matrix-vector-product. However, our techniques can easily be adapted to other use-cases, e.g., pure decompression.

Listing 1 provides CUDA-like pseudocode for our kernel, computing the matrix-vector-product of compressed matrix w_comp (with metadata row_off and ter_minmax, using dictionary dec) and BF16 vector $x$, into output buffer $y$. The handling of various edge cases and some index calculations have been removed for readability. Please see our repository for the fully functional implementation.

```
template <int num_warps, int w_width>
_global__ void Sub1MatVec(
    int* dec,
    ushort* w_comp, int* row_off, __nv_bfloat162* ter_minmax,
    __nv_bfloat16* x, __nv_bfloat16* y
{
    __shared__ float x_shared[w_width];
    for (int i = thread; i < w width; i += 32 * num_warps)
        x_shared[i] = __bfloat162float(x[i]);
    _shared__ float deq[3][32 * num_warps]
    deq[0][thread] = 0;
    deq[1][thread] = __bfloat162float(ter_minmax[row].x);
    deq[2][thread] = __bfloat162float(ter_minmax[row].y);
    _syncthreads():
    __shared__ w_comp_block[32][num_warps];
    float res = 0;
    int idx = 0;
    for (int i = 0; i < row_off[row + 1] - row_off[row]; i += 32) (
        w_comp_block[warp][lane] = w_comp[i + lane];
        if (lane < 28) {
            for (int j = 0; j < 32; j++) {
                    int enc = w_comp_block [warp][j];
                    int wx14 = dec[2 * enc + (lane / 14)];
                    int ter = (wx14 >> (4 + 2 * (lane %14))) & 0x3;
                    int ter = (wx14 >> (4 + 2
                    float w = deq[ter][thread];
                    res += w * x_shared[idx +
    }
for (int i = 16; i > 0; i /= 2)
    res += __shfl_down_sync(0xfffffffff, res, i);
    if (lane == 0
        y[row] += __float2bfloat16(res);
}
```

Listing 1. Simplified kernel pseudocode for a fused decompress + matrix-vector-product operation.

Parallelization. Overall, each threadblock will handle multiple consecutive rows, each of which is processed by a single warp. We use exactly one thread-block per GPU Streaming Multiprocessor (SM) with min(\#rows_in_block, 32) warps; if there are more than 32 rows in a block, (some) warps sequentially process multiple rows (note that this part is omitted in Listing 1 for simplicity). This avoids any bad wave quantization effects. We find this strategy to be an effective heuristic that yields good performance for all matrix shapes we consider.

Execution. Our kernel starts by loading the entire input vector to shared memory (x_shared, lines 7-9), using all warps in a threadblock. This enables fast element access in the subsequent per-row product-sum accumulations.

Next, each warp processes its corresponding row by first fetching (up to) 32 codewords into shared memory (w_comp_block, line 23) using a single coalesced transaction. It then loops over those symbols, processing one-ata-time (lines 26-33). First, using 28 of its 32 threads (line 25), it fetches the corresponding decoding data from the dictionary where the first UINT32 is assigned to threads $0-13$ and the second to threads 14-27 (wx14, line 27). Then, each thread extracts its corresponding ternary weight (lines 29-30) and adds the corresponding input product into its own partial result accumulator (res, line 31). We note that the input reads from shared memory are contiguous and do not cause bank conflicts. Afterwards, each thread advances the offset index (idx, line 32) into the input vector by the total number of weights encoded in the current symbol.

Finally, after the full row has been scanned, a warpreduction (lines 37-38) over the partial results of each thread yields the output ( $\mathrm{y}$, lines 39-40).

Ternary decoding. Another relevant detail is that ternary weights are stored as $0,1,2$ (line 29) but need to be dequantized to $0, w_{\min }, w_{\max }$ for multiplication with inputs. We found that the most efficient way of performing this conversion is via a shared memory lookup table (lines 11-14). Crucially, this table needs to be replicated 32 times across the column-dimension to avoid very frequent bank conflicts, which would otherwise occur every time not all 28 threads dequantize the same value (line 30). Fortunately, there are only 3 input values and so its overall size is tolerable.

Encoding. So far, we have only focused on the decoding operation, but we also have to encode matrices with reasonable efficiency. In general, this is done by building a trie datastructure (of the dictionary discussed in Section 4.3.2) mapping sequences to codewords. Then, we iterate through the input while simulatenously traversing the trie to find longest prefix matches, yielding the corresponding codewords. Finally, we densely pack rows of different lengths into a contiguous buffer and record corresponding row offsets. Unlike decoding, encoding is not very latency critical and a straight-forward GPU kernel using one thread per row of the matrix to compress suffices.

## 5. Experiments

### 5.1. General Setup

Models. We focus our experiments on the SwitchTransformer (Fedus et al., 2022) family of models. Our primary target is the very largest variant, c2048, with around 1.6 trillion parameters, but we also consider the comparatively small base 128 (7B params) and large 128 (26B params) versions for testing and ablations. We chose the SwitchTransformer family as it contains the largest publicly-available model, which also features a similar or higher number of training tokens to parameters ratio than potential alternatives like Artetxe et al. (2022). Further, those models are also among the most popular massive MoEs, with several implementations across frameworks.

Framework. As accessibility is a major goal of our work, we build our code-base around the PyTorch-backend of the highly popular HuggingFace (Wolf et al., 2019) framework, rather than on the SwitchTransormer's original training environment MeshTensorflow (Shazeer et al., 2018) or its JAX-based successor T5X (Google, 2023). This brings a number of additional challenges.

First, we find that the largest model variants require a handful of bugfixes, primarily configuration and model setup changes, in order to run properly. We suspect that this is because their enormous sizes have rendered extensive testing very difficult. Second, we observed a major inefficiency in the context of generative inference for models with a large number of experts: the HuggingFace implementation will perform several (empty) CUDA calls for potentially 1000s of experts to which no token is routed, accumulating large overheads. We modify the implementation (also for baselines) to skip such unnecessary calls, leading to $>10 \times$ speedup for large models. We apply all changes to the HuggingFace framework only dynamically at runtime, so that our code can be run directly with an official installation.

HuggingFace prioritizes ease-of-use and flexibility over high performance. For that reason, we conduct inference measurements not only end-to-end, including all HuggingFace overheads, but also in isolated fashion, comparing uncompressed and compressed matrix operations directly. This is to demonstrate that our GPU kernels would also yield low overhead in more optimized inference environments.

Datasets. SwitchTransformers have been trained for a Masked-Language-Modelling (MLM) objective (Raffel et al., 2020b) on the C4 dataset (Raffel et al., 2020a). Similar to most works in the area of LLM quantization (Yao et al., 2022; Frantar et al., 2022; Dettmers \& Zettlemoyer, 2022), we focus on general upstream compression directly on this pretraining task/dataset combination. Consequently, our evaluation focuses on validation performance for C4/MLM,
where we use the public reproduction of $\mathrm{C} 4$ on HuggingFace as well as their replication of the original masking procedure. Calibration data for compression is taken, in order, from the first two shards of the training set. For efficiency, we primarily evaluate on 128 samples (corresponding to the average loss over $>10 \mathrm{~K}$ tokens, which is quite stable) from the first shard of the validation set, but we also perform some evaluations other datasets.

Hardware. All compression experiments, including those for the very largest models, can be performed in less than a day on a single NVIDIA A6000 with 48GB of GPU memory. However, efficiently compressing trillion parameter models using a large number of calibration samples requires a few 100GBs of (CPU) RAM; the original 1.6T model itself also occupies $>3$ TB disk storage. We highlight that our work is performed in a highly constrained environment for models of this size, for example, it is already infeasible to load the entire (uncompressed) 1.6T model into RAM, let alone into GPU memory. For inference on compressed models, we will also consider running on multiple NVIDIA 3090 GPUs, with $24 \mathrm{~GB}$ of memory each, in addition to A6000s.

### 5.2. Compression Results

Accuracy. We begin by quantizing all SwitchTransformer models to 2-bit and ternary precision, and evaluating their validation loss. Our default number of calibration samples is $10 \mathrm{~K}$ for 128 experts and $160 \mathrm{~K}$ for 2048 , but we also consider using $0.5 \times$ and $2 \times$ as many samples. In addition to using our efficient QMoE framework discussed in Section 3, we also consider a standard round-to-nearest (RTN) baseline (Dettmers et al., 2022). We simulate the latter by fixing Hessians to the identity matrix, thus applying precisely the same quantization settings and evaluation protocol. Table 5 summarizes our results.

| method | base128 |  | large128 |  | c2048 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | 2bit | tern | 2bit | tern | 2bit | tern |
| BF16 | 1.73 |  | 1.55 |  | 1.18 |  |
| RTN | 2.27 | 4.54 | 1.96 | 2.79 | 1.33 | 2.15 |
| QMoE 0.5x | 1.78 | 2.11 | 1.54 | 1.70 | 1.22 | 1.27 |
| QMoE 1.0x | $\mathbf{1 . 7 6}$ | 1.99 | $\mathbf{1 . 5 6}$ | 1.69 | $\mathbf{1 . 2 0}$ | $\mathbf{1 . 2 6}$ |
| QMoE 2.0x | $\mathbf{1 . 7 6}$ | $\mathbf{1 . 9 3}$ | 1.57 | $\mathbf{1 . 6 4}$ | 1.21 | $\mathbf{1 . 2 6}$ |

Table 5. Comparing $\mathrm{C} 4$ validation losses for 2-bit and ternary (tern) quantized SwitchTransformers. "QMoE $0.5 \mathrm{x}$ " indicates that only half of the default number of calibration samples are used.

Perhaps surprisingly, vanilla rounding (RTN) does not lead to a complete model collapse even at ternary precision, emphasizing the high robustness of large MoEs to quantization. Nevertheless, the loss increases are quite significant for smaller models at 2-bit and far too large to be useful at ternary precision. In contrast, using data-dependent quanti- zation, 2-bit is achievable at minimal loss (1.7\% relative on c2048) and ternary at only a small increase ( $6.7 \%$ relative on c2048). This demonstrates not only the effectiveness of such advanced quantization methods in this context, but also shows that extremely low-bit compression is indeed practical for massive MoEs.

Additionally, we conduct evaluations on Arxiv, GitHub, StackeEchange and Wikipedia data sampled from RedPajama (Computer, 2023). Even though only $<0.01 \%$ of our $\mathrm{C} 4$ calibration data originates from those websites, the compressed model still preserves performance almost as well as on the core of the distribution (see Table 6).

| bits | arxiv | github | stackexch. | wiki |
| :---: | :---: | :---: | :---: | :---: |
| BF16 | 1.31 | 0.99 | 1.15 | 1.20 |
| 2-bit | 1.34 | 1.05 | 1.17 | 1.24 |
| tern | 1.42 | 1.13 | 1.22 | 1.32 |

Table 6. Additional evaluations for the c2048 model.

In terms of calibration data, we see that increasing the amount of samples generally improves performance slightly, most noticeably for ternary quantization, but there is also some noise in the process, especially at 2-bit.

Compression. Next, we investigate the actual compression rates that are achieved by further compressing ternary models using our scheme introduced in Section 4. We consider both compression relative to just the $\mathrm{MoE}$ modules (the model parts we quantize) as well as to the full model and all its metadata. The compression rates and overall checkpoint sizes are listed in Table 7.

| model | moe-only | full | size [GB] |  |
| :---: | :---: | :---: | :---: | :---: |
|  |  |  | bf16 | ours |
| base128 | $17.06 \times$ | $11.76 \times$ | 14.9 | 1.27 |
| large128 | $18.34 \times$ | $13.32 \times$ | 52.7 | 3.96 |
| c2048 | $20.07 \times$ | $19.81 \times$ | 3142 | 158.6 |

Table 7. Compression rates and sizes for ternary models.

In general, measuring only relative to parts we compress (moe-only), all sizes achieve $>16 \times$ compression rate and thus $<1$ bits per parameter storage. On c2048, even the overall rate, including all uncompressed dense layers, remains at $19.81 \times$, corresponding to 0.807 bits per parameter reducing the checkpoint size from $3142 \mathrm{~GB}$ to $158.6 \mathrm{~GB}$. One can also observe that compression rates increase with model size, which is for two reasons: (a) natural sparsity increases while our encoding dictionary is also optimized for c2048 (see Section 4), and (b) weight distributions become closer to independent for larger layer sizes.

Runtime. Finally, we evaluate how long it takes to produce compressed models on a single A6000 GPU, for different
![](https://cdn.mathpix.com/cropped/2024_06_04_2bf28cb8b86df30ffa9ag-10.jpg?height=442&width=1634&top_left_y=213&top_left_x=213)

Figure 5. (Left) Per-layer compressed kernel performance relative to uncompressed execution. (Right) End-to-end runtimes of compressed models and estimates ( ${ }^{*}$, would require $65 / 130$ GPUs) for bloat16 baselines. c2048 is run on $4 \times$ A6000 and $8 \times 3090$ GPUs, respectively.

amounts of calibration data. The results are shown in Table 8 . Smaller models can be compressed in less than an hour and even c2048 in less than a day, confirming the high efficiency of QMoE. The runtime increase from large 128 to c2048 is roughly proportional to the difference in size, despite the latter using $16 \times$ more samples. This is because the number of samples per expert stays constant and the expert size increases only slightly. Finally, we note that simply (iteratively) loading the original $1.6 \mathrm{~T}$ model into RAM takes close to 5 hours on our slow disk storage.

| model | $5 \mathrm{~K} / 80 \mathrm{~K}$ | $10 \mathrm{~K} / 160 \mathrm{~K}$ | $20 \mathrm{~K} / 320 \mathrm{~K}$ |
| :---: | :---: | :---: | :---: |
| base128 | $8.4 \mathrm{~min}$ | $14.0 \mathrm{~min}$ | $21.6 \mathrm{~min}$ |
| large128 | $22.0 \mathrm{~min}$ | $30.2 \mathrm{~min}$ | $45.2 \mathrm{~min}$ |
| c2048 | $13.3 \mathrm{~h}$ | $16.0 \mathrm{~h}$ | $20.8 \mathrm{~h}$ |

Table 8. Compression runtime for different calibration data size.

### 5.3. Runtime Results

Individual Layers. Our kernel performance evaluation starts with a direct (isolated) comparison of our compressed matrix-vector product kernels (see Section 4) against PyTorch's standard (uncompressed) bfloat 16 cuBLAS kernels. Figure 5 (Left) shows the time taken by our compressed kernels relative to bfloat16, for the matrix shapes found in our MoEs, on two different GPUs. While our kernels have to perform a lot less slow (global) memory reads than the bfloat 16 baseline due to lower storage costs, they need to spend much more compute for complex unpacking of the heavily-compressed weights. Nevertheless, executing our compressed kernels takes less time than the close to ideal bfloat 16 baseline in all cases, with up to $35 \%$ speedup on specific matrix shapes. We note that these are very lowlatency operations, with the smallest matrix taking $<0.02$ milliseconds and the largest $<0.05$.

End-to-End Execution. Finally, we also benchmark our kernels end-to-end in HuggingFace on the real weights of our compressed MoE models. We consider an individual user application, like (Frantar et al., 2022; Leviathan et al., 2023; Park et al., 2022), where a single prompt (sampled from $\mathrm{C} 4$ ) should be processed to generate a 128 -token response. As actually running the bfloat 16 version of the c2048 model would require $>65$ A6000 and $>1303090$ GPUs (versus 4 and 8 , respectively, for sub-1-bit compressed weights) we have to estimate its runtime. We do this by having all experts in a layer point to the same weight data (completely resolving memory issues), which allows us to collect timings with precisely the same overheads as for our compressed models. However, this is a highly optimistic estimate since real execution would require close to $20 \times$ more GPUs, with corresponding communication overheads, and our numbers should thus be viewed only as a lower bound.

The results, shown in Figure 5 (Right), demonstrate that end-to-end execution of compressed models is only $<5 \%$ slower than standard (uncompressed) execution. This slight slow-down despite faster per-layer timings is due to the fact that the encoder may sometimes route multiple tokens to the same expert. Our current implementation naively executes a separate matrix-vector product for each token, while the baseline performs a much more efficient joint matrix multiplication. For applications where this is a significant bottleneck, one could easily introduce an inner loop over tokens into our kernel (Listing 1, line 30), or fully decompress first, followed by a standard matmul, for large token counts.

## 6. Related Work

Mixture-of-Expert (MoE) Models. Mixture-of-expert models are a popular research direction aimed at creating significantly more efficient large-scale models (Fedus et al., 2022; Artetxe et al., 2022; Clark et al., 2022). At the core of MoEs lie (sparse) routing mechanisms, of which many variants have been proposed. Those range from static assignment based on input tokens IDs (Roller et al., 2021), over dynamic token-to-expert matching (Zhou et al., 2022), to "soft" routing of linear input combinations (Puigcerver
et al., 2023). Since MoEs can feature rather different computational profiles from standard dense models, there is also significant research on optimizing inference and training systems (Barham et al., 2022; Gale et al., 2023; Hwang et al., 2023). Among the most critical problems in this area are data-exchanges between accelerators during routing and dealing with uneven compute-loads for different experts.

LLM Quantization. Quantization is a very popular compression technique, which has seen a vast amount of work (Gholami et al., 2021), especially in the context of LLMs. Specifically, the ability to perform accurate weight quantization for billion-parameter models has greatly boosted their accessibility: it has been shown that extremely large dense models can be quantized to 8 - or even 4 -bit precision at little accuracy loss (Dettmers et al., 2022; Yao et al., 2022; Frantar et al., 2022; Dettmers \& Zettlemoyer, 2022). Pushing towards even lower bitwidths via more sophisticated compression formats, like multi-level grouping coupled with higher-precision outliers (Dettmers et al., 2023b), or new quantization techniques, like incoherence preprocessing (Chee et al., 2023), is an active area of research. Currently, accurate quantization to 2 or less bits per parameter appears to be a major barrier for post-training quantization of standard LLMs. By contrast, in this work we show that massive MoE models appear to be significantly more compressible, as we achieve sub-1-bit compression at comparable loss increases to 3-bit or 4-bit quantization of standard LLMs with advanced techniques.

MoE Compression. There has also been work on compressing MoE models in particular. Chen et al. (2022) and Koishekenov et al. (2022) perform compression via specialization of MoEs to specific "downstream" finetuning datasets by pruning components not relevant to the particular task. In contrast, we focus on general "upstream" compression of the pretrained model, via extremely low-bit quantization. Other works (Kim et al., 2022b; Yi et al., 2023; Kim et al., 2023) also perform MoE quantization, but focus on noticeably higher bit-widths, like 8 or 4 bits per weight. This is accomplished primarily via simple rounding, which, as shown by our experiments, is not accurate enough for full 2-bit or lower compression. Kim et al. (2022a) achieve 2-bit quantization on a 5 billion parameter $\mathrm{MoE}$, which is considered relatively small in this area, by further optimization of the model via Quantization-Aware Training (Nagel et al., 2021). Applying such an approach for trillion-scale models would be extremely resource intensive. They also do not provide any mechansims for exploiting low-bit quantization and its corresponding natural sparsity in practice, which is challenging and constitutes a key contribution of our work.

We are particularly focused on scalabilty and practicalty. While existing works study models with at most tens of billions of parameters, we demonstrate the effectiveness and efficiency of our techniques at trillion parameter scale, both for the quantization process itself as well as for actual inference of compressed models.

## 7. Discussion and Limitations

We have presented QMoE, an end-to-end compression and inference framework for addressing the massive memory costs of MoE inference. We showed, for the first time, that models such as the trillion-parameter SwitchTransformerc2048 can be accurately compressed to less than 1 bit per parameter, close to $20 \times$ compression rate, in a custom format that enables the first efficient end-to-end execution of such a model on a single commodity GPU server. QMoE is fully open-source and built around the popular HuggingFace framework, making deployment and research for massive MoEs significantly cheaper and more accessible.

Our study is confined to a limited set of models, as only very few massive and accurate MoEs are available publicy. Additionaly, due to their size, most MoEs are trained and deployed in different bespoke framework, requiring complex manual integrations to use for further research. Nevertheless, we have covered some of the largest and most accurate available MoEs, specifically SwitchTransformers (Fedus et al., 2022). A natural extension of our work would be to apply our QMoE techniques to other MoE models and variants, such as Artetxe et al. (2022) or the recently-proposed SoftMoEs (Puigcerver et al., 2023).

Additionally, we have focused on direct compression of the pretrained base model. However, it would also be interesting to further finetune a compressed model for specialized downstream tasks, similar to QLoRA (Dettmers et al., 2023a). Zoph et al. (2022) report strong results when finetuning only non-expert layers, which $\mathrm{QMoE}$ leaves uncompressed, suggesting that this application could be promising. We hope to explore this in future work.

## References

Artetxe, M., Bhosale, S., Goyal, N., Mihaylov, T., Ott, M., Shleifer, S., Lin, X. V., Du, J., Iyer, S., Pasunuru, R., et al. Efficient large scale language modeling with mixtures of experts. In Empirical Methods in Natural Language Processing (EMNLP), 2022.

Barham, P., Chowdhery, A., Dean, J., Ghemawat, S., Hand, S., Hurt, D., Isard, M., Lim, H., Pang, R., Roy, S., et al. Pathways: Asynchronous distributed dataflow for ml. In Conference on Machine Learning and Systems (MLSys), 2022 .

Bengio, Y., Léonard, N., and Courville, A. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,

Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. Quip: 2-bit quantization of large language models with guarantees. arXiv preprint arXiv:2307.13304, 2023.

Chen, T., Huang, S., Xie, Y., Jiao, B., Jiang, D., Zhou, H., Li, J., and Wei, F. Task-specific expert pruning for sparse mixture-of-experts. arXiv preprint arXiv:2206.00277, 2022.

Clark, A., De Las Casas, D., Guy, A., Mensch, A., Paganini, M., Hoffmann, J., Damoc, B., Hechtman, B., Cai, T., Borgeaud, S., et al. Unified scaling laws for routed language models. In International Conference on Machine Learning (ICML), 2022.

Computer, T. RedPajama: An open source recipe to reproduce llama training dataset, 2023. URL https://github.com/togethercomputer/ RedPajama-Data.

Dettmers, T. and Zettlemoyer, L. The case for 4-bit precision: k-bit inference scaling laws. arXiv preprint arXiv:2212.09720, 2022.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. LLM.int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023a.

Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T., and Alistarh, D. SpQR: A sparse-quantized representation for near-lossless llm weight compression. arXiv preprint arXiv:2306.03078, $2023 \mathrm{~b}$.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. GLaM: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning (ICML), 2022.

Elsen, E., Dukhan, M., Gale, T., and Simonyan, K. Fast sparse convnets. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232-5270, 2022.
Frantar, E. and Alistarh, D. SparseGPT: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning (ICML), 2023.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. GPTQ: Accurate post-training compression for generative pretrained transformers. arXiv preprint arXiv:2210.17323, 2022.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. GPTQ code, 2023. URL https://github.com/ IST-DASLab/gptq.

Gale, T., Narayanan, D., Young, C., and Zaharia, M. MegaBlocks: Efficient sparse training with mixture-ofexperts. In Conference on Machine Learning and Systems (MLSys), 2023.

Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., and Keutzer, K. A survey of quantization methods for efficient neural network inference. arXiv preprint arXiv:2103.13630, 2021.

Google. T5x, 2023. URL https://github.com/ google-research/t5x.

Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., and Peste, A. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. arXiv preprint arXiv:2102.00554, 2021.

Hubara, I., Nahshan, Y., Hanani, Y., Banner, R., and Soudry, D. Accurate post training quantization with small calibration sets. In International Conference on Machine Learning (ICML), 2021.

Huffman, D. A. A method for the construction of minimumredundancy codes. Proceedings of the IRE, 40(9):10981101,1952 .

Hwang, C., Cui, W., Xiong, Y., Yang, Z., Liu, Z., Hu, H., Wang, Z., Salas, R., Jose, J., Ram, P., et al. Tutel: Adaptive mixture-of-experts at scale. In Conference on Machine Learning and Systems (MLSys), 2023.

Kim, Y. J., Fahim, R., and Awadalla, H. H. Mixture of quantized experts (MoQE): Complementary effect of lowbit quantization and robustness. OpenReview, 2022a.

Kim, Y. J., Henry, R., Fahim, R., and Awadalla, H. H. Who says elephants can't run: Bringing large scale moe models into cloud scale production. arXiv preprint arXiv:2211.10017, 2022b.

Kim, Y. J., Henry, R., Fahim, R., and Awadalla, H. H. Finequant: Unlocking efficiency with fine-grained weight-only quantization for llms. arXiv preprint arXiv:2308.09723, 2023.

Koishekenov, Y., Nikoulina, V., and Berard, A. Memoryefficient NLLB-200: Language-specific expert pruning of a massively multilingual machine translation model. arXiv preprint arXiv:2212.09811, 2022.

Kurtic, E., Campos, D., Nguyen, T., Frantar, E., Kurtz, M., Fineran, B., Goin, M., and Alistarh, D. The Optimal BERT Surgeon: Scalable and accurate secondorder pruning for large language models. arXiv preprint arXiv:2203.07259, 2022.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Gshard, Z. Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning (ICML), 2023.

Lin, B., Zheng, N., Wang, L., Cao, S., Ma, L., Zhang, Q., Zhu, Y., Cao, T., Xue, J., Yang, Y., et al. Efficient GPU kernels for n:m-sparse weights in deep learning. In Conference on Machine Learning and Systems (MLSys), 2023 .

MacKay, D. J. Information theory, inference and learning algorithms. Cambridge University Press, 2003.

Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., and Blankevoort, T. Up or down? Adaptive rounding for post-training quantization. In International Conference on Machine Learning (ICML), 2020.

Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., and Blankevoort, T. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295, 2021.

Park, G., Park, B., Kwon, S. J., Kim, B., Lee, Y., and Lee, D. nuQmm: Quantized matmul for efficient inference of large-scale generative language models. arXiv preprint arXiv:2206.09557, 2022.

Puigcerver, J., Riquelme, C., Mustafa, B., and Houlsby, N. From sparse to soft mixtures of experts. arXiv preprint arXiv:2308.00951, 2023.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21 (140):1-67, 2020a.
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research (JMLR), 21(1):5485-5551, 2020b.

Roller, S., Sukhbaatar, S., Weston, J., et al. Hash layers for large sparse models. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

Shazeer, N., Cheng, Y., Parmar, N., Tran, D., Vaswani, A., Koanantakool, P., Hawkins, P., Lee, H., Hong, M., Young, C., et al. Mesh-tensorflow: Deep learning for supercomputers. Conference on Neural Information Processing Systems (NeurIPS), 2018.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Wang, P., Chen, Q., He, X., and Cheng, J. Towards accurate post-training network quantization via bit-split and stitching. In International Conference on Machine Learning (ICML), 2020.

Welch, T. A. A technique for high-performance data compression. Computer, 17(06):8-19, 1984.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019 .

Wu, X., Yao, Z., and He, Y. ZeroQuant-FP: A leap forward in llms post-training w4a8 quantization using floatingpoint formats. arXiv preprint arXiv:2307.09782, 2023.

Yao, Z., Aminabadi, R. Y., Zhang, M., Wu, X., Li, C., and He, Y. ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers. arXiv preprint arXiv:2206.01861, 2022.

Yi, R., Guo, L., Wei, S., Zhou, A., Wang, S., and Xu, M. Edgemoe: Fast on-device inference of moe-based large language models. arXiv preprint arXiv:2308.14352, 2023.

Yu, C., Chen, T., and Gan, Z. Boost transformer-based language models with gpu-friendly sparsity and quantization. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.

Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A. M., Le, Q. V., Laudon, J., et al. Mixture-ofexperts with expert choice routing. Conference on Neural Information Processing Systems (NeurIPS), 2022.

Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., and Fedus, W. ST-MoE: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.


[^0]:    ${ }^{1}$ Institute of Science and Technology Austria (ISTA) ${ }^{2}$ Neural Magic Inc. Corresponding author: elias.frantar@ist.ac.at

</end of paper 4>


