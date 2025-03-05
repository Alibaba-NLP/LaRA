<paper 0>
# Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon 

Peitian Zhang ${ }^{1,2 *}$ Zheng Liu $^{1 \dagger}$ Shitao Xiao $^{1}$ Ninglu Shao ${ }^{1,2}$ Qiwei Ye ${ }^{1}$ Zhicheng Dou ${ }^{2}$<br>1: Beijing Academy of Artificial Intelligence,<br>2: Gaoling School of Artificial Intelligence, Renmin University of China<br>\{namespace.pt, zhengliu1026\}@gmail.com


#### Abstract

The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considerable cost at both training and inference time, and exert an unfavorable impact to the LLM's original capabilities. In this work, we propose a new method called Activation Beacon, which condenses LLM's raw activations into compact forms such that the LLM can perceive a longer context with a limited context window. Activation Beacon is introduced as a plug-in module, which fully preserves the LLM's original capability in short contexts. It works with the sliding window to streamingly process the long context, which leads to a competitive memory and time efficiency in both training and inference. Activation Beacon is trained with short-sequence data of diversified condensing ratios. Thanks to such a treatment, it can be effectively learned to support different context lengths with a small training cost. Our experiment verifies Activation Beacon's effectiveness of context extension: it can remarkably accomplish highquality extension of Llama-2-7B's context by $\times 100$ times (from $4 \mathrm{~K}$ to $400 \mathrm{~K}$ ); meanwhile, it can also achieve superior performances across a variety of longcontext language modeling and understanding tasks. The source code and model checkpoint are available at https://github.com/FlagOpen/FlagEmbedding


![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-01.jpg?height=410&width=1388&top_left_y=1730&top_left_x=366)

Figure 1: Comparison of the sliding window perplexity [22] between Activation Beacon and other context extension methods, including 1) Position Interpolation [5], 2) NTK-Aware Scaled RoPE [1], 3) LongLlama [32]. Activation Beacon leads to better long-context generation quality with higher running efficiency (memory, time).[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-02.jpg?height=279&width=683&top_left_y=240&top_left_x=366)

(A) Activation Condensing with Beacon

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-02.jpg?height=279&width=697&top_left_y=240&top_left_x=1061)

(B) AR with Condensed Activations

Figure 2: (A) The beacon token ( $\langle\mathrm{bcn}\rangle$ ) is appended to a context, which prompts the LLM to condense the raw activations into more compact forms; (B) The condensed activations are streamingly processed with the sliding window for auto-regression (AR).

## 1 Introduction

Large language models (LLMs) need to process long contexts to accomplish many important tasks, such as retrieval-augmented generation and in-context learning. However, existing LLMs are typically constrained by fixed context windows, e.g., $2 \mathrm{~K}$ for Llama-1 [29] and $4 \mathrm{~K}$ for Llama-2 [30], which is not enough to handle some real-world scenarios. Although LLMs can be fine-tuned or retrained to extend their context windows [16, 6, 5, 28, 20, 32, 18], it will result in considerable costs at both training and inference time due to the quadratic computing complexity of self attention. Besides, the continued training on long-sequence data may compromise the LLM's general capability in shorter contexts, which is unfavorable to their practical usage. In light of these challenges, it is desirable to explore new mechanisms, which can not only realize the cost-effective extension of context length, but also be compatible with the LLM's existing capabilities.

In this work, we propose Activation Beacon (shown as Figure 2) as a new method for LLM's context extension. It condenses the LLM's raw activations (i.e. keys and values from the self-attention module) into highly compact forms such that the LLM can perceive the information from a vast scope of context even with a limited context window. The above idea shares the common philosophy as sparse attention [3, 8, 38] and context compression [4, 7, 19, 22, 14]. However, it enjoys substantial advantages over the previous methods in many perspectives, including the effectiveness of context extension (especially the quality of long-context generation and the flexibility of supporting diverse context lengths), inference and training efficiency, and the compatibility with the existing LLMs, thanks to a series of crucial technical designs.

Instead of developing a new model from scratch, we argue that the LLM itself can work as a proficient activation condenser with proper adaptation given its strong and well-established context representation capability. Based on this argument, we introduce a simple but effective model architecture and running mechanism to facilitate the production and utilization of condensed activations. Particularly, we introduce special tokens, known as beacon tokens ( $\langle\mathrm{bcn}\rangle)$, which prompt the LLM to condense the contextual information into beacon tokens's activations (Figure 2). For a context of length $l$, a team of $k(k<l)$ beacon tokens are dispatched to the end of it, which leads to a condensing ratio of $\alpha(\alpha=l / k)$. We maintain another copy of the LLM's self-attention parameters, including $\left\{W_{Q}^{b}, W_{K}^{b}, W_{V}^{b}, W_{O}^{b}\right\}$. These new parameters are specialized to learn the activation condensing, while the original parameters in the LLM are fixed. Thus, Activation Beacon serves as a plug-in component for the LLM, introducing extended contextual information to the LLM without adversely affecting its existing capabilities in short contexts.

To efficiently handle long contexts, we propose stream processing with the sliding window. The long context is partitioned into multiple intervals of length $l$. A sliding window is employed to sequentially process one interval at a time. When dealing with the next interval, the raw activations of the previous interval are discarded while its condensed activations are accumulated. Therefore, the sliding window is formulated as $\left[\langle\mathrm{bcn}\rangle_{1}, \ldots,\langle\mathrm{bcn}\rangle_{m}, x_{m+1}, \ldots, x_{n}\right]$ where $\langle\mathrm{bcn}\rangle_{*}$ stands for the beacon tokens from previous intervals and $x_{*}$ is normal tokens in the current interval. The size of the sliding window is upper-bounded by the maximum window size of the LLM, e.g. $4 \mathrm{~K}$ for Llama-2, which maintains a low memory consumption and a linear time complexity. Meanwhile, it also accumulatively gathers rich contextual information from the past $(\alpha-1) \times m+n$ tokens.

The condensed activations are expected to fully encode the information within the raw activations, thereby assisting the LLM to accomplish high-quality generation of new content. With this consideration, we propose to learn Activation Beacon through the auto-regression task. In the sliding window, the generation likelihood of the normal token $x_{i}$ is maximized based on the beacon tokens and its preceding normal tokens, i.e., $\max p\left(x_{i} \mid\langle\mathbf{b c n}\rangle_{1}, \ldots,\langle\mathrm{bcn}\rangle_{m}, x_{m+1} \ldots, x_{i-1}\right)$. Considering that a dramatic extension of context calls for a large condensing ratio, while a moderate extension just needs a small condensing ratio, we perform a random sampling of $\alpha$ during the stream processing. Consequently, the generation can be conditioned on a mixture of condensed activations with diversified condensing ratios, which substantially contributes to the Activation Beacon's generalization in handling the extension of different context lengths.

Activation Beacon is applied to Llama-2-7B (chat), whose original context length is $4 \mathrm{~K}$. The training data is sampled from RedPajama [10] and LongAlpaca [6], whose length are all less than $8 \mathrm{~K}$. The training process merely takes $10 \mathrm{~K}$ steps, which can be accomplished within 9 hours on an $8 \times \mathrm{A} 800$ GPU machine. Notably, it leads to a superior quality of language modeling on the extended context lengths, like $8 \mathrm{~K}, 16 \mathrm{~K}$, and $32 \mathrm{~K}$, whose result is even better than the fine-tuned full-attention baselines. It is equally competitive on long-context understanding tasks, such as question answering and fewshot learning. Activation Beacon also shows the potential to establish super long contexts: by learning to support the condensing factor of 128 , the context length of Llama-2 can be remarkably extended to $400 \mathrm{~K}$ (Figure 1). As a compatible module, Activation Beacon can also work with other techniques, like position interpolation ( $\$$ C) and retrieval (\$D) for even longer and better context extension effect.

To summarize, we propose Activation Beacon, which realizes dramatic extension of LLM's context based on the high-quality condensing of LLM's activations. It also enjoys a high running efficiency, a high compatibility with the existing LLM, and a small cost of training thanks to its optimized designs on architecture and running mechanism. In our experiment, the effectiveness of Activation Beacon is verified given its superior performances across a wide range of long-context processing tasks.

## 2 Activation Beacon

### 2.1 Overview

The LLM exploits the contextual information while predicting the new content. The contextual information is represented by the activations, particularly the keys and values in the self-attention module. With a fixed size of context window $L$, a typical LLM can only query the recent $L$ activations for contextual information. However, we argue that the window size should simply be the upper bound of input units rather than context length. By condensing more information into each activation, i.e. the information from a larger scope rather a single token, the LLM will be able to perceive a longer context with its original context window.

### 2.2 Activation Condensing

We aim to adapt the LLM itself for activation condensing given its strong context representation capability. Particularly, we employ special tokens, called beacon tokens, which prompt the LLM to condense the contextual information into their activations. We also maintain another copy of the LLM's MHA (multi-head self-attention) parameters, denoted as MHA ${ }^{b}$, including the layerwise projection matrices for queries, keys, values, and outputs $\left\{\boldsymbol{W}_{Q}^{b}, \boldsymbol{W}_{K}^{b}, \boldsymbol{W}_{V}^{b}, \boldsymbol{W}_{O}^{b}\right\}$. These parameters are specifically learned for condensing the activations. Besides, they are lightweight, merely accounting for $1 / 3$ of the LLM's original parameters (e.g., 2B with the LLaMA-2 7B model).

The activation condensing is performed with the following operations (Figure 3 I). For the context of length $l, k$ beacon tokens are appended to the end of it. The LLM auto-regressively encodes the context as well as the beacon tokens, as a result, the raw activations of regular tokens are generated and then condensed into the beacon tokens' activations. Formally, let the input features of the beacon tokens as $\boldsymbol{H}^{b} \in \mathbb{R}^{k \times D}$, the projections for the beacon tokens' queries, keys, and values are performed in the first place:

$$
\boldsymbol{Q}^{b} \leftarrow \boldsymbol{W}_{Q}^{b} \boldsymbol{H}^{b}, \quad \boldsymbol{K}^{b} \leftarrow \boldsymbol{W}_{K}^{b} \boldsymbol{H}^{b}, \quad \boldsymbol{V}^{b} \leftarrow \boldsymbol{W}_{V}^{b} \boldsymbol{H}^{b}
$$

Then, the projection results query the keys $\left(\boldsymbol{K}^{r} \in \mathbb{R}^{l \times D}\right)$ and values $\left(\boldsymbol{V}^{r} \in \mathbb{R}^{l \times D}\right)$ of the raw activations from normal tokens to generate the condensed activations, leading to a condensing ratio

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-04.jpg?height=507&width=1179&top_left_y=234&top_left_x=473)

II. Attention Scheme of Beacons

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-04.jpg?height=314&width=1114&top_left_y=767&top_left_x=511)

Figure 3: (I) The raw activations of ordinal tokens (the blue square) are condensed into the compact activations of beacon tokens (the green squere). Future tokens are auto-regressively generated conditioned on the raw activations in the current interval and the condensed activations accumulated from previous intervals. (II) The attention schemes for activation condensing.

$\alpha=l / k:$

$$
\begin{gather*}
\boldsymbol{A} \leftarrow \operatorname{softmax}\left(\operatorname{mask}\left(\frac{\boldsymbol{Q}^{b}\left\{\boldsymbol{K}^{r} \oplus \boldsymbol{K}^{b}\right\}^{T}}{\sqrt{D}}\right)\right) \\
\boldsymbol{O}^{b} \leftarrow \boldsymbol{W}_{O}^{b} \boldsymbol{A}\left\{\boldsymbol{V}^{r} \oplus \boldsymbol{V}^{b}\right\} \tag{1}
\end{gather*}
$$

The final output of self-attention is produced by the concatenation of both raw activations from the normal tokens and the condensed activations from the beacon tokens.

To optimize the quality of activation condensing, we explore three attention schemes for the beacon tokens, i.e. the mask $(\cdot)$ operator, which are as shown in Figure 3 II. 1) Segmentation, where each beacon can attend to an equally segmented span of the context. 2) Stepwise expansion, where each beacon can attend to one more span than its predecessor, and the last beacon can attend to the entire context. 3) Full coverage, where the entire context can be attended by all beacons. For all three options, we restrict the context length $l$ to be evenly divisible by the number of beacon tokens $k$. Besides, the beacon tokens are always positioned next to the last normal token it can attend to. Although the three options are of the same computation cost, it's empirically found that the second option, i.e. the stepwise expansion, leads to the optimal performance ( $\$ 5$.

### 2.3 Stream Processing

The long context is partitioned into multiple intervals of length $l$. A sliding window is employed to sequentially process one interval at a time. When dealing with the next interval, the raw activations of the previous interval are discarded while its condensed activations are accumulated. Therefore, the sliding window consists of $m$ beacon tokens (i.e. $\langle\mathrm{bcn}\rangle$ ) from the past intervals, and the normal tokens in the current interval. With the above formulation, the next token is predicted as:

$$
\begin{equation*}
p\left(x_{n} \mid\langle\mathrm{bcn}\rangle_{1}, \ldots,\langle\mathrm{bcn}\rangle_{m}, x_{m+1}, \ldots, x_{n-1} ; \Theta, \Theta^{b}\right) \tag{2}
\end{equation*}
$$

where $\Theta$ denotes the parameters of the LLM and $\Theta^{b}$ denotes the introduced parameters for beacons. Crucially, both $\langle\mathrm{bcn}\rangle_{*}$ and $x_{*}$, are encoded by their relative positions within the sliding window, regardless of their absolute positions in the entire context. The size of the sliding window is upbounded by the context window size of the LLM, which results in a competitive running efficiency for
both training and inference. Different from the typical stream processing where the context beyond the sliding window is discarded [36], our method can accumulatively cover the information from the past $(\alpha-1) \times m+n$ tokens. Note that the above working mechanism may also benefit from the increasing of window size, as more beacon tokens can be accumulated in the sliding window to cover an even longer context. Consequently, Activation Beacon can work with strategies like NTK [1], PI [5] for further extension of the context. Detailed collaboration effect is explored in Appendix C.

### 2.4 Learning Method

Plug-in to LLM. As introduced, Activation Beacon introduces the following parameters $\left.\left(\Theta_{b}\right): 1\right)$ the beacon token's embedding $\left.e_{\langle b c n\rangle}, 2\right)$ the linear projection matrices for $\mathrm{MHA}^{b}$ : $\left\{\boldsymbol{W}_{Q}^{b}, \boldsymbol{W}_{K}^{b}, \boldsymbol{W}_{V}^{b}, \boldsymbol{W}_{O}^{b}\right\}$ in each transformer layer. Overall, it accounts for less than $1 / 3$ of the LLM's original size, e.g., 2B with the Llama-2-7B model. Activation Beacon reuses other transformer modules from the LLM (i.e., MLP and LayerNorm). This turns out to be the optimal trade-off between effectiveness and training cost. Activation Beacon is learned while all of the LLM's original parameters are frozen. Besides, it is only used to generate the condensed activations without interfering the inference process of normal tokens. Therefore, it serves as a plug-in module for the LLM, which introduces the long contextual information without affecting the LLM's existing capabilities in processing short contexts.

Auto-Regression. We train Activation Beacon by auto-regression, where the next token is predicted based on the condensed activations from the beacon tokens and the raw activations from the ordinary tokens. As mentioned in $\$ 2.2$, a training instance is partitioned into equal-sized intervals of length $l$ and streamingly processed. Afterwards, the following loss is minimized:

$$
\begin{equation*}
\min _{\Theta_{b}} . \sum_{j=1}^{\lceil|X| / / l\rceil} \sum_{i=1}^{l}-\log p\left(x_{i}^{j} \mid\langle\mathbf{b c n}\rangle_{1}, \ldots,\langle\mathbf{b c n}\rangle_{m_{j}}, x_{1}^{j}, \ldots, x_{i-1}^{j} ; \Theta, \Theta^{b}\right) \tag{3}
\end{equation*}
$$

where $x_{i}^{j}$ is the $i$-th token in the $j$-th interval of $X, m_{j}$ stands for the number of beacon tokens accumulated before the $j$-th interval, whose value depends on the condensing ratio of each preceding interval $\left(m_{j}=\sum_{z=1}^{j-1}\left(l / / \alpha_{z}\right)\right.$ ).

Step-wise randomized condensing ratio. The training is performed purely with short-sequence data, i.e. $1024<|X|<8192$, where the majority of training samples are less than $4 \mathrm{~K}$ (Table 6. Therefore, we are able to achieve superior training efficiency. To generalize Activation Beacon to support different context lengths, e.g., $16 \mathrm{~K}, 32 \mathrm{~K}, 100 \mathrm{~K}$, and even longer, the auto-regression needs to be conditioned on different amounts of beacon tokens with diversified condensing ratios. For this purpose, we randomly sample the condensing ratio for each interval within a large candidate scope: $\alpha_{j} \sim\{2,4,8, \ldots 128\}$, which will introduce dramatic diversity to the condensing ratios and amount of beacon tokens within the auto-regression process.

## 3 Experiment

Our experiments are performed for the exploration of the following issues. 1) Activation Beacon's impact on the long-context generation capabilities (measured by Perplexity). 2) Activation Beacon's impact on the long-context utilization capability (reflected by tasks like long document QA and summarization). 3) Activation Beacon's impact on efficiency in terms of GPU memory and inference time. 4) The individual contribution of different technical factors.

### 3.1 Settings

Implementation. Our method is applied to Llama-2-7B (chat) [30] for empirical studies. Our training data is a mixture of $80 \mathrm{~K}$ sampled data from RedPajama [10] and LongAlpaca [6] (70K from RedPajama and $10 \mathrm{~K}$ from LongAlpaca, respectively). The sequence length of each sample is between 1024 and 8192. The statistics of our training data is reported in Table 6 We use a single $8 \times$ A800 GPU machine for training. The training is performed for 10,000 steps (one epoch of the whole training data) with a batch size of 8 and a learning rate of 5 e-5 using the linear scheduler. The length of the context interval is set to 1024 . The condensing ratio is sampled from $\{2,4,8,16,32,64,128\}$ during training. As introduced, Llama's own parameters are freezed throughout the training process.

Table 1: Sliding window perplexity of different context window extension methods on PG19, ProofPile, and CodeParrot. Activation Beacon successfully extends the context window of Llama-2-7B model to sequences much longer than the ones seen during training.

| Method | PG19 |  |  |  | Proof-Pile |  |  |  | CodeParrot |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $4 \mathrm{~K}$ | $16 \mathrm{~K}$ | $32 \mathrm{~K}$ | $100 \mathrm{~K}$ | $4 \mathrm{~K}$ | $16 \mathrm{~K}$ | $32 \mathrm{~K}$ | $100 \mathrm{~K}$ | $4 \mathrm{~K}$ | $16 \mathrm{~K}$ | $32 \mathrm{~K}$ | $100 \mathrm{~K}$ |
| Llama-2-7B | 9.21 | $>10^{3}$ | $>10^{3}$ | OOM | 3.47 | $>10^{3}$ | $>10^{3}$ | OOM | 2.55 | $>10^{3}$ | $>10^{3}$ | OOM |
| PI | 9.21 | 19.5 | $>10^{2}$ | OOM | 3.47 | 5.94 | 33.7 | OOM | 2.55 | 4.57 | 29.33 | $\mathrm{OOM}$ |
| NTK | $9.21 \quad$ | 11.5 | 37.8 | OOM | 3.47 | 3.65 | 7.67 | $\mathrm{OOM}$ | 2.55 | 2.86 | 7.68 | $\mathrm{OOM}$ |
| StreamingLLM | 9.21 | 9.25 | 9.24 | 9.32 | 3.47 | 3.51 | 3.50 | 3.55 | 2.55 | 2.60 | 2.54 | 2.56 |
| AutoCompre.-6K | 11.8 | $>10^{2}$ | $>10^{3}$ | OOM | 4.55 | $>10^{2}$ | $>10^{3}$ | OOM | 5.43 | $>10^{2}$ | $>10^{3}$ | OOM |
| YaRN-128K | 6.68 | 6.44 | 6.38 | OOM | 2.70 | 2.47 | 2.41 | OOM | 2.17 | 2.04 | 2.00 | $\mathrm{OOM}$ |
| LongChat-32K | 9.47 | 8.85 | 8.81 | OOM | 3.07 | 2.70 | 2.65 | OOM | 2.36 | 2.16 | 2.13 | $\mathrm{OOM}$ |
| LongAlpaca-16K | 9.96 | 9.83 | $>10^{2}$ | OOM | 3.82 | 3.37 | $>10^{3}$ | OOM | 2.81 | 2.54 | $>10^{3}$ | OOM |
| LongLlama | 9.06 | 8.83 | $\mathrm{OOM}$ | $\mathrm{OOM}$ | 2.61 | 2.41 | $\mathrm{OOM}$ | $\mathrm{OOM}$ | 1.95 | 1.90 | $\mathrm{OOM}$ | OOM |
| Activation Beacon | 9.21 | 8.34 | 8.27 | 8.50 | 3.47 | 3.34 | 3.32 | 3.31 | 2.55 | 2.43 | 2.41 | 2.62 |

Baselines. The following types of baselines are chosen for comparison (all based on the LLaMA-2-7B (chat) model unless otherwise specified). 1) The basic method, i.e. LLaMA-2-7B (chat) [29] with $4 \mathrm{~K}$ context length. 2) The fine-tuning free methods, including Positional Interpolation (PI) [5], the NTKAware Scale ROPE (NTK) [1], and StreamingLLM [36]. 3) The fine-tuned full-attention methods, including LongChat-32K [16], LongAlpaca-16K [6], YaRN-128K [20]. 4) The fine-tuned methods with adapted architectures for long contexts, including AutoCompressor-6K [7] and LongLlama [32] (based on CodeLlama [24]). We enable FlashAttention-2 [11] to accelerate self-attention computation and save GPU usage for all the baselines. At present, Activation Beacon is incompatible with FlashAttention-2 due to its utilization of the customized attention scheme; thus, we use the scaled dot product attention (sdpa) from PyTorch [17] for acceleration.

### 3.2 Main Results

### 3.2.1 Long-Context Language Modeling

The experiment on long-context language modeling is performed with three datasets: PG19 [22], Proof-Pile [40], and CodeParrot [31]. Specifically, for PG19, we use its entire test set with 100 books. For Proof-Pile, we extract the arxiv papers from the test set that are longer than $32 \mathrm{~K}$, which are 79 papers in total. For CodeParrot, there is no pre-defined test set. Following previous studies [25, 39], we first concatenate code from the same repository to form long sequences, then we sample 100 sequences for evaluation. The perplexity is computed with a sliding window of size $2 \mathrm{~K}$ [21].

The evaluation results are reported in Table 1, where Activation Beacon leads to a superior longcontext language modeling performance. First of all, it not only outperforms the Llama-2-7B baseline but also results in a notably improved performance than the fine-tuning free methods. It is worth noting that with the extension of context from $4 \mathrm{~K}$ to $32 \mathrm{~K}$, the language modeling performance can be gradually improved by Activation Beacon, indicating that the expanded information from the longer context can be effectively utilized to facilitate the generation. By comparison, the language modeling performance is decreased with other fine-tuning-free methods. Most of them become ineffective after the context length goes beyond $32 \mathrm{~K}$.

Secondly, Activation Beacon's performance is comparable to or even better than the fine-tuned full-attention methods. This result is remarkable knowing that Activation Beacon runs with a much higher efficiency (to be analyzed in Section 3.3). Although there are cases where some of the fine-tuned full-attention baselines achieve better performances, their empirical advantages may not be fully resulted from the introduction of long contextual information. For example, YaRN-128K's performance has already been notably higher than Llama-2-7B at the context length of $4 \mathrm{~K}$, and so is the case with LongChat-32K on Proof-Pile and CodeParrot. Note that the update of the LLM's original parameters is not always favorable because it may not be well generalized to many other scenarios. By comparison, our method is simply a plug-in module to introduce long contextual information without affecting the LLM's existing capabilities.

Thirdly, Activation Beacon is able to achieve a much longer extension of the context than the rest of the methods. Particularly, it maintains a quality generation performance after the context length is

Table 2: Evaluation of different methods on LongBench. Activation Beacon performs on par with the fine-tuned full-attention baselines.

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-Shot | Code |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Llama-2-7B | 24.90 | 22.60 | 24.70 | 60.00 | 48.10 |
| PI | 18.98 | 17.16 | 25.03 | 49.43 | 52.73 |
| NTK | 23.21 | 23.34 | 24.40 | 59.29 | 49.28 |
| StreamingLLM | 21.47 | 22.22 | 22.20 | 50.05 | 48.00 |
| AutoCompressor-6K | 13.22 | 10.61 | 14.00 | 15.72 | 23.62 |
| YaRN-128K | 24.03 | 24.11 | 19.82 | 60.00 | $\underline{62.73}$ |
| LongChat-4K | 28.14 | 21.88 | 26.59 | 62.06 | 52.77 |
| LongChat-32K | $\mathbf{3 1 . 5 8}$ | 23.50 | 26.70 | $\mathbf{6 4 . 0 2}$ | 54.10 |
| LongAlpaca-4K | 26.81 | 24.44 | $\underline{26.93}$ | 62.92 | 55.15 |
| LongAlpaca-16K | 28.70 | $\underline{28.10}$ | $\mathbf{2 7 . 8 0}$ | $\underline{63.70}$ | 56.00 |
| LongLlama | $\underline{30.12}$ | 16.37 | 24.19 | 60.31 | $\mathbf{6 6 . 0 5}$ |
| Activation Beacon | 28.27 | $\mathbf{2 8 . 4 4}$ | 25.15 | 61.00 | 57.75 |

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-07.jpg?height=675&width=1033&top_left_y=923&top_left_x=538)

Figure 4: The evaluation of topic retrieval accuracy at different context lengths. Activation Beacon is competitive against the fine-tuned methods, like LongChat-32K and LongAlpaca-16K.

extended to $100 \mathrm{~K}$, where most of the baselines become either ineffective or out-of-memory (OOM). In fact, Activation Beacon is still effective even after the context length is further extended to $400 \mathrm{~K}$ (see Figure 1), which means a $100 \times$ extension of Llama-2-7B's maximum context length. Unlike many other methods like fine-tuning, Activation Beacon does not require any long-sequence training data to acquire such a super long-context capability, which contributes to its high usability in practice.

### 3.2.2 More Long-Context Tasks

We further study the five real-world tasks from LongBench [2], including single-doc QA, multi-doc QA, summarization, few-shot learning, and code completion, where the experiment result on each task is reported in Table 2. We also evaluate the topic retrieval task [16], whose result is shown in Figure 4. In Appendix D, we evaluate the passkey retrieval task [35]. Similar to our previous observation on long-context language modeling, Activation Beacon leads to a notable improvement over Llama-2-7B and the fine-tuning-free baselines. Meanwhile, it reaches a comparable performance with the fine-tuned full-attention methods. Because a large portion of the evaluation samples can be (almost) covered by the $16 \mathrm{~K}$ or $32 \mathrm{~K}$ context window, the fine-tuned full-attention methods indeed set a high standard on LongBench. However, knowing that the fine-tuning operation will change the LLM's original parameters, it is still interesting to investigate where the empirical advantage of the finetuned methods comes from. To figure out this problem, we benchmark the performance of

Table 3: Evaluation of inference time and GPU memory usage. Both metrics are measured by the average value of 100 forward passes (FlashAttention-2 is enabled for LongChat).

| Method | GPU Memory (GB) |  |  |  |  |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $4 \mathrm{~K}$ | $8 \mathrm{~K}$ | $16 \mathrm{~K}$ | $32 \mathrm{~K}$ | $100 \mathrm{~K}$ | $4 \mathrm{~K}$ | $8 \mathrm{~K}$ | $16 \mathrm{~K}$ | $32 \mathrm{~K}$ | $100 \mathrm{~K}$ |
| LongChat-32K | 18.5 | 24.2 | 35.6 | 58.4 | $\mathrm{OOM}$ | 0.045 | 0.089 | 0.191 | 0.460 | $\mathrm{OOM}$ |
| StreamingLLM | 19.9 | 19.9 | 19.9 | 19.9 | 19.9 | - | - | - | - | - |
| AutoCompressor-6K | 17.7 | 22.6 | 32.3 | 51.7 | OOM | 0.087 | 0.134 | 0.224 | 0.478 | OOM |
| LongLlama | 18.2 | 21.9 | 34.2 | OOM | OOM | 0.079 | 0.190 | 0.436 | OOM | OOM |
| Activation Beacon | 21.7 | 21.3 | 21.4 | 21.6 | 21.6 | 0.071 | 0.121 | 0.237 | 0.473 | 1.494 |

LongChat-32K and LongAlpaca-16K at the context length of $4 \mathrm{~K}$, where they use the same information as the Llama-2-7B baseline. Interestingly, both methods result in a substantial improvement over Llama-2-7B on every task. Especially for summarization, where both methods are already sufficiently strong at $4 \mathrm{~K}$, yet little extra improvements are made with the further extended context window. By comparison, Activation Beacon inherits Llama-2-7B's performance at the context length of $4 \mathrm{~K}$, where its performance gain over Llama-2-7B is introduced by the extended context. In this sense, its impact on utilizing the long contextual information can still be no inferior to the ones from the finetuned methods in the corresponding situations.

### 3.3 Efficiency Analysis

We evaluate the running efficiency at the inference time in terms of time cost and GPU memory usage, whose results are reported in Table 3 Compared with LongChat (full-attention) and LongLlama, Activation Beacon enjoys a much smaller GPU memory usage at the long context. Activation Beacon and StreamingLLM result in a similar memory cost because both methods are based on sliding windows. As for the inference time, Activation Beacon is faster than LongLlama, but slower than LongChat when the context is short. This is because Activation Beacon is streamingly processed while LongChat is fully paralle ${ }^{3}$ However, Activation Beacon is able to gradually catch up when the context length gets longer, as its time complexity is linear to the context length. It will ultimately become much faster than the full-attention methods if the context length is extended long enough. Finally, we compare our training cost with LongAlpaca, which is featured for its high training efficiency (shown in Table 4). Under a similar hardware condition (8×A800 GPUs vs. $8 \times$ A100 GPUs), the training of Activation Beacon can be accomplished in just 9 hours, which is even faster than the reported time cost of LongAlpaca-16K with $S^{2}$-attn 4 ( $\$ 2.4$.

### 3.4 Ablation Studies

We perform ablation studies to evaluate the impact from different technical factors, including the

![](https://cdn.mathpix.com/cropped/2024_06_04_149022b2d7d54424841fg-08.jpg?height=43&width=1385&top_left_y=2022&top_left_x=369)
parameters for beacons (\$2.4), and the composition of training data (\$3.1). The experiment results are shown in Table 5

First of all, we can find that the attention scheme exerts a substantial impact on Activation Beacon's performances on both long-context language modeling (PG19) and long-context understanding (QA). The stepwise expansion works with the gradually expanded attention scope. Therefore, it enables the beacons to acquire different levels of local and global information of each context interval, which notably improves the performance over the other two options.[^1]

Table 5: The impact of different technical factors: attention scheme of beacon token, condensing ratio, composition of training data. Performances are measured by PG19 with $32 \mathrm{~K}$ context and single-Doc QA on LongBench. Default settings are marked by *.

| Factor | Setting | PG19 | QA |
| :--- | :--- | :---: | :---: |
| Attention | Segmentation | 8.39 | 26.05 |
|  | Full coverage | 8.76 | 23.13 |
|  | Stepwise expansion* | 8.27 | 28.27 |
| Condensing | Monotonous $(\alpha=4)$ | $>10^{2}$ | 26.48 |
|  | Instance-wise randomized | 8.19 | 26.33 |
|  | Step-wise randomized* | 8.27 | 28.27 |
| Beacon | Q, K, V (1.5B) | 8.32 | 27.04 |
|  | Q, K, V, O, MLP (5.5B) | 8.81 | 23.46 |
|  | Q, K, V, O (2.0B)* | 8.27 | 28.27 |
| Data | RedPajama only | 8.24 | 24.98 |
| Composition | RedPajama+LongAlpaca* | 8.27 | 28.27 |

Secondly, the sampling of the condensing ratio is another influential factor. In this place, we compare two alternative strategies. The instance-wise option samples one condensing ratio for all context intervals of each training instance $X$ (from the same scope as the step-wise method, i.e. $\{2,4,8, \ldots, 128\}$ ). While the monotonous option makes use of one constant condensing ratio of 4 (which can support a up-to $16 \mathrm{~K}$ context length). We can observe that the step-wise sampling strategy, which introduces the most diversified condensing ratios when learning, results in competitive performance on perplexity while significantly outperforms the other two options on long-context understanding.

Thirdly, we analyze the impact by introducing different amounts of learnable parameters to the beacon module. Specifically, when we remove the output projection matrix $\boldsymbol{W}_{O}^{b}$ from the beacon parameters $\mathrm{MHA}^{b}$ ( $\$ 2.2$, the empirical performances on both tasks degrade. When we additionally include the MLP parameters of FFN, the model's performance does not improve. We conjecture that this is probably because the FFN layer is heavily loaded, which slows down the convergence of the training process. As a result, it suggests that our current formulation of the learnable parameters is a good trade-off between cost and effectiveness.

Lastly, we can also observe that only using RedPajama as the training data already leads to a competitive performance on both evaluation tasks. The introduction of more training data from LongAlpaca contributes little to the language modeling task. However, it brings an additional improvement to the empirical performance on Single-Doc QA.

## 4 Related Works

We discuss the following works which are devoted to the extension of LLM's context. First of all, a large body of methods have been proposed to increase the size of context window. For example, ALiBi [21] leverages linear-decaying attention biases to achieve the extrapolation of position encoding. Methods like Position Interpolation [5], NTK-Aware scaling [1] and ReRoPE [26] make progress on top of RoPE [27], which enable the LLM to handle unseen positions at the inference time. Although such methods can be directly applied to the well-trained LLM, they usually benefit from continual fine-tuning where the extended context can be better utilized [20]. The fine-tuning with long-sequence data is expensive. Thus, people investigate how to reduce the training cost. For example, LongLora [6] proposes $S^{2}$-Attn and leverages LoRA for cost-effective training; while PoSE [41] uses skip-wise position indices to train LLMs on $2 \mathrm{~K}$ context length as a simulation of $128 \mathrm{~K}$. However, the fine-tuning operations are still prone to big costs if super long-sequence data is presented. Finally, the fine-tuning operation may impair the LLM's existing capabilities on short contexts [20]. By comparison, our method is trained with a small cost and enjoys a high efficiency in training and inference. Besides, it serves as a plug-in module that is fully compatible with the existing LLM.

The quadratic complexity of transformer is a major bottleneck to achieve long contexts. Thus, many previous works aim to address this problem by using sparse attention [8; 3; 38; 12] or approximate attention computation [15; 33; 9; 23]. However, there are threefold challenges about these methods as analyzed in [36]: the requirement of customized GPU kernels for specific variants of matrix multiplication, the dependency on global attention patterns which are unsuitable for autoregressive language models, the incompatibility with the well-pretrained models. In contrast, our method is free from these constraints and preserves a high compatibility with the existing LLMs.

It is also plausible to find ways to process long contexts with short context windows. One popular strategy is to use sliding windows. For example, StreamingLLM [36] and LM-Infinite [13] are able to achieve an infinite context by only maintaining the activations for the very first and the latest tokens. However, they are unable to leverage the rich information from the long context because the portion beyond the sliding window will be discarded. Besides, the long contexts can also be summarized and compressed into more compact forms [4, 7, 19, 22, 14], which follow the same spirit as our work. However, the previous methods call for major changes to the original model's architecture and working process, which brings in many problems. Notably, they are prone to substantial compression losses which prevent them from making extensions for long contexts. Besides, they lack the flexibility to support different context lengths, and suffer from the incompatibility with existing LLMs.

Finally, it becomes popular to offload the long context into external memory and retrieve the useful part from it as the working context. The retrieved data can be either the chunked input [37, 39] or the cached KV activations, e.g., Memorizing Transformers [35] and LongMem [34]. This idea has been further extended by many recent works. For example, Landmark Attention [18] uses a special token to represent a chunk of activations, which enables more efficient computation of retrieval. Focused Transformers [32] proposes to use contrastive training which improves the discrimination of relevant keys from the cached data. The retrieval-based methods can be limited due to the utilization of incoherent context. However, it tackles the the problem from a different perspective which can benefit from the collaboration with our method (explored in Appendix D.

## 5 Conclusion

We introduce Activation Beacon for the extension of LLM's context length. Activation Beacon condenses the LLM's raw activations into highly compact forms, enabling the LLM to perceive a long context with a limited context window. As a plug-in component for the LLM, it brings in long contextual information while fully preserving the LLM's existing capabilities in short contexts. When dealing with long-sequence data, it resorts to a sliding window for stream processing, which leads to a superior working efficiency for both training and inference. By using short-sequence data with diversely sampled condensing ratios, it can be effectively learned to support different context lengths with a small training cost. Our experiment verifies Activation Beacon as an effective, efficient, compatible, and low-cost method to extend the context length for LLMs.

## Broader Impact

Activation Beacon establishes long-context capabilities for the large language model without affecting its original capabilities. This enhancement may benefit many long-context scenarios using LLMs, such as long document understanding/summarization, and lifelong chating with long-term memory. Therefore, it is particularly useful for AI applications like AI readers and lifelong AI chatbots. Activation Beacon is able to compress the raw activations of LLM into fewer yet more compact ones with minimal loss. As a result, it can reduce the Key-Value cache requirements for numerous AI applications, leading to significant resource savings. Moreover, compared to full attention mechanisms, Activation Beacon requires considerably fewer computational resources with competitive speed. This efficiency also contributes to environmental sustainability. As a downside, since Activation Beacon is based on the LLM, it inherits the internal biases of the LLM. Consequently, there is a risk of generating unreliable or harmful content, which underscores the need for careful monitoring the ethical usage of these AI systems.

## References

[1] Ntk-aware scaled rope, 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/ 14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/.

[2] Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023.

[3] Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. CoRR, abs/2004.05150, 2020. URLhttps://arxiv.org/abs/2004.05150

[4] Bulatov, A., Kuratov, Y., and Burtsev, M. S. Scaling transformer to $1 \mathrm{~m}$ tokens and beyond with RMT. CoRR, abs/2304.11062, 2023. doi: 10.48550/ARXIV.2304.11062. URL https: //doi.org/10.48550/arXiv. 2304.11062

[5] Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

[6] Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., and Jia, J. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023.

[7] Chevalier, A., Wettig, A., Ajith, A., and Chen, D. Adapting language models to compress contexts. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 3829-3846. Association for Computational Linguistics, 2023. URL https: //aclanthology.org/2023.emnlp-main. 232

[8] Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[9] Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH

[10] Computer, T. Redpajama: An open source recipe to reproduce llama training dataset, 2023. URL https://github.com/togethercomputer/RedPajama-Data

[11] Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. CoRR, abs/2307.08691, 2023. doi: 10.48550/ARXIV.2307.08691. URL https://doi.org/10 48550/arXiv. 2307.08691

[12] Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., and Wei, F. Longnet: Scaling transformers to $1,000,000,000$ tokens. CoRR, abs/2307.02486, 2023. doi: 10.48550/ ARXIV.2307.02486. URL https://doi.org/10.48550/arXiv.2307.02486.

[13] Han, C., Wang, Q., Xiong, W., Chen, Y., Ji, H., and Wang, S. Lm-infinite: Simple onthe-fly length generalization for large language models. CoRR, abs/2308.16137, 2023. doi: 10.48550/ARXIV.2308.16137. URLhttps://doi.org/10.48550/arXiv.2308.16137.

[14] Huang, X. and Hollenstein, N. Long-range language modeling with selective cache. In Bouamor, H., Pino, J., and Bali, K. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pp. 4838-4858. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.findings-emnlp.321.

[15] Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=rkgNKkHtvB

[16] Li, D., Shao, R., Xie, A., Sheng, Y., Zheng, L., Gonzalez, J. E., Stoica, I., Ma, X., and Zhang, H. How long can open-source llms truly promise on context length?, June 2023. URL https://lmsys.org/blog/2023-06-29-longchat.

[17] Michael Gschwind, Driss Guessous, C. P. Accelerated pytorch 2 transformers. https:// pytorch.org/blog/accelerated-pytorch-2/ 2023.

[18] Mohtashami, A. and Jaggi, M. Landmark attention: Random-access infinite context length for transformers. arXiv preprint arXiv:2305.16300, 2023.

[19] Mu, J., Li, X. L., and Goodman, N. D. Learning to compress prompts with gist tokens. CoRR, abs/2304.08467, 2023. doi: 10.48550/ARXIV.2304.08467. URL https://doi.org/10 48550/arXiv. 2304.08467 .

[20] Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

[21] Press, O., Smith, N. A., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=R8sQPpGCv0.

[22] Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P. Compressive transformers for long-range sequence modelling. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=SylKikSYDH

[23] Ren, H., Dai, H., Dai, Z., Yang, M., Leskovec, J., Schuurmans, D., and Dai, B. Combiner: Full attention transformer with sparse computation cost. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 22470-22482, 2021. URL https://proceedings.neurips. cc/paper/2021/hash/bd4a6d0563e0604510989eb8f9ff71f5-Abstract.html.

[24] Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J., et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023 .

[25] Rubin, O. and Berant, J. Long-range language modeling with self-retrieval. CoRR, abs/2306.13421, 2023. doi: 10.48550/ARXIV.2306.13421. URL https://doi.org/10 48550/arXiv. 2306.13421

[26] Su, J. Rectified rotary position embeddings. https://github.com/bojone/rerope, 2023.

[27] Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. CoRR, abs/2104.09864, 2021. URLhttps://arxiv.org/abs/2104.09864

[28] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A., Chaudhary, V., Song, X., and Wei, F. A length-extrapolatable transformer. arXiv preprint arXiv:2212.10554, 2022.

[29] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[30] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[31] Tunstall, L., Von Werra, L., and Wolf, T. Natural language processing with transformers, 2022.

[32] Tworkowski, S., Staniszewski, K., Pacek, M., Wu, Y., Michalewski, H., and Miłoś, P. Focused transformer: Contrastive training for context scaling. arXiv preprint arXiv:2307.03170, 2023.

[33] Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. CoRR, abs/2006.04768, 2020. URL https://arxiv.org/abs/2006.04768

[34] Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F. Augmenting language models with long-term memory. CoRR, abs/2306.07174, 2023. doi: 10.48550/ARXIV.2306. 07174. URLhttps://doi.org/10.48550/arXiv.2306.07174.

[35] Wu, Y., Rabe, M. N., Hutchins, D., and Szegedy, C. Memorizing transformers. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=TrjbxzRcnf-.

[36] Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

[37] Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B. Retrieval meets long context large language models. CoRR, abs/2310.03025, 2023. doi: 10.48550/ARXIV.2310.03025. URL https://doi.org/10 48550/arXiv. 2310.03025

[38] Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283-17297, 2020.

[39] Zhang, P., Xiao, S., Liu, Z., Dou, Z., and Nie, J. Retrieve anything to augment large language models. CoRR, abs/2310.07554, 2023. doi: 10.48550/ARXIV.2310.07554. URL https: //doi.org/10.48550/arXiv.2310.07554

[40] Zhangir Azerbayev, Edward Ayers, B. P. Proof-pile. https://huggingface.co/datasets/ hoskinson-center/proof-pile, 2022.

[41] Zhu, D., Yang, N., Wang, L., Song, Y., Wu, W., Wei, F., and Li, S. Pose: Efficient context window extension of llms via positional skip-wise training. CoRR, abs/2309.10400, 2023. doi: 10.48550/ARXIV.2309.10400. URL https://doi.org/10.48550/arXiv.2309.10400
</end of paper 0>


<paper 1>
# Extending Llama-3's Context Ten-Fold Overnight 

Peitian Zhang ${ }^{1,2}$, Ninglu Shao ${ }^{1,2}$, Zheng Liu ${ }^{1 *}$ Shitao Xiao ${ }^{1}$, Hongjin Qian ${ }^{1,2}$,<br>Qiwei Ye ${ }^{1}$, Zhicheng Dou ${ }^{2}$<br>${ }^{1}$ Beijing Academy of Artificial Intelligence<br>${ }^{2}$ Gaoling School of Artificial Intelligence, Renmin University of China<br>namespace.pt@gmail.com zhengliu1026@gmail.com


#### Abstract

We extend the context length of Llama-3-8B-Instruct from $8 \mathrm{~K}$ to $80 \mathrm{~K}$ via QLoRA fine-tuning ${ }^{2}$. The entire training cycle is super efficient, which takes 8 hours on one 8xA800 (80G) GPU machine. The resulted model exhibits superior performances across a broad range of evaluation tasks, such as NIHS, topic retrieval, and longcontext language understanding; meanwhile, it also well preserves the original capability over short contexts. The dramatic context extension is mainly attributed to merely $3.5 \mathrm{~K}$ synthetic training samples generated by GPT-4, which indicates the LLMs' inherent (yet largely underestimated) potential to extend its original context length. In fact, the context length could be extended far beyond $80 \mathrm{~K}$ with more computation resources. Therefore, the team will publicly release the entire resources (including data, model, data generation pipeline, training code) so as to facilitate the future research from the community: https://github.com/ FlagOpen/FlagEmbedding.


## 1 Introduction

Recently, considerable attention has been directed towards long-context large language models, where different approaches are adopted to establish long-context capabilities for large language models [4, 14, 5, 8, 9, 16, 2]. However, most of them require significant compute and resources to accomplish.

In this technical report, we propose an efficient solution for entitling the long-context capabilities for LLMs, with which we extend the context length of Llama-3-8B-Instruc ${ }^{3}$ from $8 \mathrm{~K}$ to $80 \mathrm{~K}$. Specifically, we use GPT-4 [13] to synthesize $3.5 \mathrm{~K}$ long-context training data, covering three long-context tasks:

1. Single-Detail QA: the inquiry targets on one specific detail in a long context. To construct data for this task, we slice out a short segment (e.g., a chunk with less than 4096 tokens) from a long context (e.g., a book or a long paper) and prompt GPT-4 to generate multiple question-answer pairs based on this segment.
2. Multi-Detail QA: the inquiry requires information aggregation and reasoning over multiple details in a long context. We define two types of long context. The homogeneous context contains a coherent text, such as a book or a long paper. We prompt GPT-4 to generate multiple question-answer pairs that require aggregating and analyzing information from different locations in the context. The heterogeneous context consists of multiple independent texts. Notably, we perform clustering over a large corpus then extract texts from[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_fff81c76010c19555174g-2.jpg?height=691&width=1391&top_left_y=245&top_left_x=367)

Figure 1: The accuracy score of Llama-3-8B-Instruct-80K-QLoRA on Needle-In-A-HayStack task. The blue vertical line indicates the training length, i.e. $80 \mathrm{~K}$.

the same cluster to form each heterogeneous context. Therefore, the grouped texts share some semantic similarity. We then prompt GPT-4 to ask about the similarities/dissimilarities across these texts.

3. Biography Summarization: we prompt GPT-4 to write a biography for each main character in a given book.

For all three tasks, the length of context is between $64 \mathrm{~K}$ to $80 \mathrm{~K}$. Note that longer data can also be synthesized following the same methodology. When training, we organize the question-answer pairs for the same context in one multi-turn conversation then fine-tune the LLM to correctly answer the questions given the entire long context as input. Following previous work ${ }^{4}$, we mix $5 \mathrm{~K}$ instances randomly chosen from RedPajama [6] to mitigate forgetting. We also mix LongAlpaca [5] in the training set, which contains $12 \mathrm{~K}$ instruction tuning instances with $16 \mathrm{~K}$ length at maximum. Therefore, the entire training dataset contains $20 \mathrm{~K}$ instances.

We use QLoRA [7] to efficiently fine-tune the model. We apply LoRA on all Q,K,V,O projections and additionally train the embedding layer. We set LoRA rank to 32 and alpha to 16. The learning rate is $5 \mathrm{e}-5$ with linear decay and no warmups. The batch size is 8 . Gradient checkpointing is enabled. No parallel strategy is required thanks to the efficient implementation from Unsloth [1]. We train the model for 1 epoch, which takes 8 hours to complete on a 8 xA800 (80G) machine. Importantly, we expand the RoPE base from $500 \mathrm{~K}$ to $200 \mathrm{M}$ in training.

Our contributions are highlighted as follows:

- We release Llama-3-8B-Instruct-80K-QLoRA, which extends the context length of Llama3-8B-Instruct from $8 \mathrm{~K}$ to $80 \mathrm{~K}$. The entire resources including the model, training data, and code are all publicly available, which may advance the field of training long-context LLMs.
- Our training recipe is simple and efficient, while the resulted model demonstrates remarkable performance on downstream long-context tasks. Further research can be made to improve our approach.


## 2 Experiments

We evaluate our model on popular long-context benchmarks, then compare it with the original Llama-3-8B-Instruct model and the long-context Llama-3-8B-Instruct- $262 \mathrm{~K}$ from the community ${ }^{5}$.[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_fff81c76010c19555174g-3.jpg?height=564&width=1261&top_left_y=241&top_left_x=432)

Figure 2: The accuracy of Topic Retrieval task.

| Model | Single-Doc | Multi-Doc | Summ. | Few-Shot | Synthetic | Code | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-3-8B-Instruct | 37.33 | 36.04 | 26.83 | $\mathbf{6 9 . 5 6}$ | 37.75 | 53.24 | 43.20 |
| Llama-3-8B-Instruct-262K | 37.29 | 31.20 | 26.18 | 67.25 | 44.25 | $\mathbf{6 2 . 7 1}$ | 43.73 |
| Llama-3-8B-Instruct-80K-QLoRA | $\mathbf{4 3 . 5 7}$ | $\mathbf{4 3 . 0 7}$ | $\mathbf{2 8 . 9 3}$ | 69.15 | $\mathbf{4 8 . 5 0}$ | 51.95 | $\mathbf{4 7 . 1 9}$ |

Table 1: Evaluation results on LongBench. For Llama-3-8B-Instruct, we use 8K context length.

| Model | LongBookQA Eng | LongBookSum Eng |
| :---: | :---: | :---: |
| GPT-4 | 22.22 | 14.73 |
| Llama-3-8B-Instruct | 7.00 | $\mathbf{1 6 . 4 0}$ |
| Llama-3-8B-Instruct-262K | 20.30 | 10.34 |
| Llama-3-8B-Instruct-80K-QLoRA | $\mathbf{3 0 . 9 2}$ | 14.73 |

Table 2: Evaluation results on InfBench. For Llama-3-8B-Instruct, we use 8K context length. The results of GPT-4 is copied from the paper [17].

| Model | STEM | Social | Humanities | Others | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-2-7B-Chat | 35.92 | 54.37 | 51.74 | 51.42 | 47.22 |
| Mistral-7B-v0.2-Instruct | 48.79 | 69.95 | 64.99 | 61.64 | 60.10 |
| Llama-3-8B-Instruct | $\mathbf{5 3 . 8 7}$ | $\mathbf{7 5 . 6 6}$ | $\mathbf{6 9 . 4 4}$ | 69.75 | $\mathbf{6 5 . 9 1}$ |
| Llama-3-8B-Instruct-262K | 52.10 | 73.26 | 67.15 | $\mathbf{6 9 . 8 0}$ | 64.34 |
| Llama-3-8B-Instruct-80K-QLoRA | 53.10 | 73.24 | 67.32 | 68.79 | 64.44 |

Table 3: Zero-shot performance on MMLU.

Firstly, we leverage the Needle-In-A-Haystack task, which aims to recall an irrelevant piece of information (a.k.a. needle) inserted into a lengthy context (a.k.a. haystack). The accuracy is evaluated with GPT3.5. We use the same needle and haystack as in the official repository. Our model achieves $100 \%$ accuracy over all its training context length. Besides, the model generalizes well to the unseen positions $(80 \mathrm{~K} \sim 128 \mathrm{~K})$.

Secondly, we report the Topic Retrieval [12] accuracy in Figure 2. This task synthesizes a long conversation with multiple independent discussions of a certain topic between the user and the assistant. Then the LLM is required to repeat the first topic as is in the conversation. We use the conversations made up of $[5,10,15,20,25,30,40,50,60,70]$ topics for evaluation. It can be observed that Llama-3-8B-Instruct fails to remember the topic when the context is longer than $9 \mathrm{~K}$. However, the accuracy of our model remains $100 \%$ throughout all context lengths.[^2]

Thirdly, we evaluate our model on LongBench [3], which contains a variety of real-world long-context tasks. Most context on this benchmark is shorter than $32 \mathrm{~K}$. Thus, we use $32 \mathrm{~K}$ context length by default and $8 \mathrm{~K}$ for Llama-3-8B-Instruct. The results are shown in Table 1 . Our model significantly and consistently outperforms all baselines except on the code completion task. Mixing more code data in training may mitigate this problem.

Forthly, we employ the English Long-Book QA and the Long-Book Summarization task from InfiniteBench [17] to assess the model's performance on really long context. The testing instances are usually longer than $100 \mathrm{~K}$. We truncate them to $80 \mathrm{~K}$. According to Table 2. Llama-3-8B-Instruct-80KQLoRA excels on answering the questions based on the long context. It also achieves competitive performance against GPT-4 in terms of summarization. Interestingly, Llama-3-8B-Instruct with $8 \mathrm{~K}$ context outperforms GPT- 4 with $128 \mathrm{~K}$ context on summarization. This is likely to be a metricoriented issue (currently rouge-f1 is used) since the summary may have different paraphrases, which may not necessarily overlap with the ground truth.

Lastly, in Table 3, we compare the zero-shot performance of our model and the baselines on MMLU [10] benchmark. We also include Llama-2-7B-Chat [15] and Mistral-7B-Instruct-v0.2 [11] for comparison. It can be observed that both long-context models underperform the original Llama-38B-Instruct, indicating that context extension may compromise the model's short-context capability. This observation is in line with previous research [14]. However, our model's performance is still superior to other open-source models at the same scale.

## References

[1] Unsloth.ai. https://github.com/unslothai/unsloth, 2023.

[2] S. An, Z. Ma, Z. Lin, N. Zheng, and J.-G. Lou. Make your llm fully utilize the context, 2024.

[3] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023.

[4] S. Chen, S. Wong, L. Chen, and Y. Tian. Extending context window of large language models via positional interpolation, 2023.

[5] Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia. Longlora: Efficient fine-tuning of long-context large language models, 2024.

[6] T. Computer. Redpajama: An open source recipe to reproduce llama training dataset, 2023.

[7] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. Qlora: Efficient finetuning of quantized llms, 2023.

[8] Y. Ding, L. L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang. Longrope: Extending llm context window beyond 2 million tokens, 2024.

[9] Y. Fu, R. Panda, X. Niu, X. Yue, H. Hajishirzi, Y. Kim, and H. Peng. Data engineering for scaling language models to $128 \mathrm{k}$ context, 2024.

[10] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding, 2021.

[11] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023.

[12] D. Li*, R. Shao*, A. Xie, Y. Sheng, L. Zheng, J. E. Gonzalez, I. Stoica, X. Ma, , and H. Zhang. How long can open-source llms truly promise on context length?, June 2023.

[13] OpenAI. Gpt-4 technical report, 2024.

[14] B. Peng, J. Quesnelle, H. Fan, and E. Shippole. Yarn: Efficient context window extension of large language models, 2023.

[15] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra,

I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[16] P. Zhang, Z. Liu, S. Xiao, N. Shao, Q. Ye, and Z. Dou. Soaring from 4k to 400k: Extending llm's context with activation beacon, 2024.

[17] X. Zhang, Y. Chen, S. Hu, Z. Xu, J. Chen, M. K. Hao, X. Han, Z. L. Thai, S. Wang, Z. Liu, and M. Sun. $\infty$ bench: Extending long context evaluation beyond 100k tokens, 2024.


[^0]:    ${ }^{*}$ Corresponding author.

    ${ }^{2}$ The model is noted as Llama-3-8B-Instruct-80K-QLoRA given its max context length during fine-tuning. However, users could apply the model for even longer contexts via extrapolation.

    $\sqrt[3]{\text { https://llama.meta.com/llama3/ }}$

[^1]:    ${ }^{4}$ https://www.together.ai/blog/llama-2-7b-32k

    5 https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k

[^2]:    ${ }_{\text {https://github.com/gkamradt/LLMTest_NeedleInAHaystack }}$

</end of paper 1>


<paper 2>
# PoSE: EFFICIENT CONTEXT WINDOW EXTENSION OF LLMS VIA POSITIONAL SKIP-WISE TRAINING 

Dawei Zhu * *an Yang $\diamond \quad$ Liang Wang $\diamond \quad$ Yifan Song ${ }^{\curvearrowright} \quad$ Wenhao Wu $\curvearrowright \star$<br>Furu Wei $\diamond$ Sujian Li ${ }^{\ominus} \pitchfork$<br>${ }^{\odot}$ School of Computer Science, Peking University<br>N National Key Laboratory for Multimedia Information Processing, Peking University<br>$\diamond$ Microsoft Corporation<br>https://github.com/dwzhu-pku/PoSE


#### Abstract

Large Language Models (LLMs) are trained with a pre-defined context length, restricting their use in scenarios requiring long inputs. Previous efforts for adapting LLMs to a longer length usually requires fine-tuning with this target length (Fulllength fine-tuning), suffering intensive training cost. To decouple train length from target length for efficient context window extension, we propose Positional Skip-wisE (PoSE) training that smartly simulates long inputs using a fixed context window. This is achieved by first dividing the original context window into several chunks, then designing distinct skipping bias terms to manipulate the position indices of each chunk. These bias terms and the lengths of each chunk are altered for every training example, allowing the model to adapt to all positions within target length. Experimental results show that PoSE greatly reduces memory and time overhead compared with Full-length fine-tuning, with minimal impact on performance. Leveraging this advantage, we have successfully extended the LLaMA model to $128 \mathrm{k}$ tokens using a $2 \mathrm{k}$ training context window. Furthermore, we empirically confirm that PoSE is compatible with all RoPE-based LLMs and position interpolation strategies. Notably, our method can potentially support infinite length, limited only by memory usage in inference. With ongoing progress for efficient inference, we believe PoSE can further scale the context window beyond 128k.


## 1 INTRODUCTION

Large Language Models (LLMs) have revolutionized language modeling and demonstrated impressive abilities to perform various tasks (Brown et al., 2020). However, even with their remarkable capacity, these LLMs remain restricted by pre-defined context window sizes, suffering from notable performance decline when input tokens exceeds these limits. Nevertheless, numerous application scenarios demand extremely long input sequences, including long document summarization (Huang et al., 2021), in-context learning with numerous examples (Li et al., 2023), and long document retrieval (Zhou et al., 2022), etc. This naturally poses a significant challenge of context window extension: Extending the context window of a pre-trained LLM to accommodate longer sequences.

Naively fine-tuning LLMs on inputs of target length for window extension has received limited success due to the large disruption introduced by new position indices (Chen et al., 2023a; Han et al., 2023). Addressing this, Position Interpolation (Chen et al., 2023a; kaiokendev, 2023; Peng et al., 2023) propose to down-scale the position indices to match the original window size, yielding improved results for context extension. However, these methods still rely on Full-length fine-tuning, i.e., finetuning with context of target length, which is memory and time-intensive due to the computational complexity that increases quadratically with input length. For example, Chen et al. (2023a) use 32 A100 GPUs to extend LLaMA models from $2 \mathrm{k}$ to $8 \mathrm{k}$ context, and 128 A100 GPUs for even larger context. These overhead has made it impossible to extend context window to extreme lengths.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-02.jpg?height=461&width=1390&top_left_y=276&top_left_x=365)

Figure 1: Position indices of Full-length fine-tuning v.s. PoSE fine-tuning for extending the context window size from 2,048 to 8,192 . At each iteration, the former directly takes 8,192 tokens for fine-tuning, while PoSE manipulates the position indices of 2,048 tokens to simulate longer inputs. For example, we partition the original context window of 2,048 tokens into two chunks, and adjust the position indices of the second chunk by adding a distinct skipping bias term. These bias terms, as well as the length of each chunk, are altered for each training example, so that the model can adapt to all relative positions of the target context window through fine-tuning.

In this paper, we introduce Positional Skip-wisE (PoSE) fine-tuning to decouple the fine-tuning length from the target context window length, unleashing the possibility of efficiently extending context window to an extreme size. The key idea of PoSE is to simulate long inputs by manipulating position indices within a fixed context window. As depicted in Figure 1, we partition the original context window into several chunks, and adjust the position indices of each chunk by adding a distinct skipping bias term. These bias terms, as well as the length of each chunk, are altered for each training example, so that the model can adapt to all positions (including both absolute and relative) within the target context window through fine-tuning. Meanwhile, by maintaining continuous position indices within each chunk, PoSE bears a close resemblance to pre-training. As a result, the model's pre-trained capacity for language modeling and comprehension is retained to the greatest degree.

The advantages of our PoSE are threefold: 1) Memory and Time Efficiency: By only requiring the original context size for fine-tuning, PoSE circumvents the quadratic increase in computational complexity with respect to target length during the fine-tuning stage, thereby significantly reducing memory and time overhead. 2) Potential for Extremely-Long Context: We manage to extend the context window of LLaMA (Touvron et al., 2023a) by up to 64 times $(2 \mathrm{k} \rightarrow 128 \mathrm{k}, \mathrm{k}=1,024)$ while preserving decent ability of language modeling and understanding. 3) Compatible with all RoPE-based LLMs and PI strategies: The effectiveness of PoSE has been empirically validated across several representative RoPE-based LLMs, including LLaMA, LLaMA2 (Touvron et al., 2023b), GPT-J (Wang \& Komatsuzaki, 2021), and Baichuan (Baichuan, 2023). Additionally, PoSE has been demonstrated to be compatible with a variety of position interpolation methods, including Linear (Chen et al., 2023a), NTK (Peng \& Quesnelle, 2023), and YaRN (Peng et al., 2023) interpolation.

Notably, by decoupling the fine-tuning and target length, PoSE can theoretically extend context window to an infinite length. The only constraint is the memory usage during the inference phase. Hopefully, with the continuous advancements in efficient inference techniques, including Flash Attention (Dao et al., 2022; Dao, 2023), xFormers (Lefaudeux et al., 2022), vLLM (Kwon et al., 2023), etc, we believe PoSE can promisingly push the context window size to a even larger scale.

## 2 RELATED WORK

Training Length-Extrapolatable Models. Length extrapolation requires the model to handle continually increasing input tokens, even beyond the context window size used for training (Press et al., 2021). To this end, a series of positional embedding schemes have been proposed, including ALibi (Press et al., 2021), xPos (Sun et al., 2023), NoPos (Haviv et al., 2022), etc.

Similar to our work, Ruoss et al. (2023) also attempted to simulate longer sequences during training time to mitigate out-of-distribution lengths. They proposed randomized positional encoding

(RandPos), which randomly selected an ordered subset of position indices from longer sequences. Our proposed method, PoSE, diverges from their approach in several key aspects: First, RandPos is a positional embedding scheme designed to pre-train encoder-only models from scratch for length extrapolation. In contrast, PoSE is a fine-tuning method aiming at efficiently extend the context window of pre-trained LLMs, which are majorly decoder-only models. Second, in RandPos, the position indices between adjacent tokens are not continuous. However, in PoSE, the position indices within each chunk are intentionally made continuous to resemble the pre-training phase, therefore reducing the risk of disrupting the language modeling abilities learned during pre-training.

Fine-tuning LLMs for Longer Context. Differing from length extrapolation, which primarily involves training a model from scratch to support lengths exceeding those it was initially trained for, context window extension focuses on extending the context window of a pre-trained LLM. Directly fine-tuning an existing LLM with a longer context window has been shown to progress slowly (Chen et al., 2023a). To expedite and stabilize training, Chen et al. (2023a) first down-scaled position indices to match original context size through Linear Position Interpolation. Subsequently, a range of Positional Interpolation (PI) strategies have been introduced, including NTK (Peng \& Quesnelle, 2023) and YaRN (Peng et al., 2023). More recently, LongLora (Chen et al., 2023b) propose shift short attention to approximate full attention. However, all these methods require Full-length fine-tuning, suffering computational cost that grows with target context size. By contrast, our method managed to decouple train / target length, requiring only the original context size for fine-tuning.

Memory Transformers. An alternative strategy for extremely long input sequences involves memory mechanisms. Typically, there are two lines of research for utilizing memory: the recurrencebased approach (Dai et al., 2019; Bulatov et al., 2022) and the retrieval-based approach (Wu et al., 2022; Wang et al., 2023; Tworkowski et al., 2023). The former segments long inputs and reuses the hidden states of preceding segments as memory, suffering from information loss and limited capacity for random access. The latter encodes prior sequences as (key, value) pairs and utilizes a memory retriever and reader to extract previously encoded information, primarily limited by the lack of interaction between discrete memory segments. More recently, Mohtashami \& Jaggi (2023) introduced landmark attention to facilitates random access to any chunk of the input. In contrast, our method achieves full access to the entire input without any modifications to the attention mechanism.

## 3 METHODOLOGY

### 3.1 PRELIMINARIES

Rotary Position Embedding (RoPE). The use of RoPE (Su et al., 2021) has become pervasive in contemporary LLMs, including LLaMA (Touvron et al., 2023a), GPT-J (Wang \& Komatsuzaki, 2021), etc. It encodes position information of tokens with a rotation matrix that naturally incorporates explicit relative position dependency. To elucidate, given a hidden vector $\boldsymbol{h}=\left[h_{0}, h_{1}, \ldots, h_{d-1}\right]$, where $d$ is the hidden dimension, and a position index $m$, RoPE operates as follows:

$$
f(\boldsymbol{h}, m)=\left(\begin{array}{c}
h_{0}  \tag{1}\\
h_{1} \\
h_{2} \\
h_{3} \\
\vdots \\
h_{d-2} \\
h_{d-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos m \theta_{0} \\
\cos m \theta_{0} \\
\cos m \theta_{1} \\
\cos m \theta_{1} \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right)+\left(\begin{array}{c}
-h_{1} \\
h_{0} \\
-h_{3} \\
h_{2} \\
\vdots \\
-h_{d-1} \\
h_{d-2}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_{0} \\
\sin m \theta_{0} \\
\sin m \theta_{1} \\
\sin m \theta_{1} \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$

where $\theta_{j}=10000^{-2 j / d}, j \in\{0,1, \ldots, d / 2-1\}$. Unlike previous absolute position encodings that are directly applied to the input vector $\boldsymbol{x}$, RoPE is employed on the query and key vectors at each layer. Given a query $\boldsymbol{q}$ at position $m$ and a key $\boldsymbol{k}$ at position $n$, attention score $a(\boldsymbol{q}, \boldsymbol{k})$ is defined as:

$$
\begin{align*}
a(\boldsymbol{q}, \boldsymbol{k}) & =<f(\boldsymbol{q}, m), f(\boldsymbol{k}, n)> \\
& =\sum_{j=0}^{d / 2-1}\left[\left(q_{2 j} k_{2 j}+q_{2 j+1} k_{2 j+1}\right) \cos (m-n) \theta_{j}+\left(q_{2 j} k_{2 j+1}-q_{2 j+1} k_{2 j}\right) \sin (m-n) \theta_{j}\right] \\
& :=g(\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{\theta}, m-n) \tag{2}
\end{align*}
$$

Hence, RoPE encodes position information in a relative manner, as the attention score depends on the relative distances between positions rather than their absolute position values.

Problem Formulation. Given a Large Language Model pre-trained with a context window size of $L_{c}$, our objective is to extend this context size to a target length $L_{t}$, so that the model maintains good performance when processing input sequences containing a maximum of $L_{t}$ tokens.

Position Interpolation (PI). In contrast to directly extending the position indices to $L_{t}-1$ when dealing with an input text $\boldsymbol{x}=\left\{x_{0}, x_{1}, \ldots, x_{L_{t}}\right\}$, position interpolation down-scales the position indices to align with the original context window size $L_{c}$. This approach effectively mitigates the risk of encountering extreme values and has been empirically demonstrated to enhance stability during fine-tuning. Various interpolation strategies have been proposed, with $\alpha=L_{t} / L_{c}$ denoting the scaling factor:

- Linear Interpolation. As described by Chen et al. (2023a) and kaiokendev (2023), linear interpolation involves a proportional down-scaling of the position index $m$ to $m / \alpha$. Consequently, the attention score between a query $\boldsymbol{q}$ at position $m$ and a key $\boldsymbol{k}$ at position $n$ becomes $g(\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{\theta},(m-n) / \alpha)$, as defined in Equation 2. Theoretical analysis has substantiated that the interpolated attention score exhibits significantly greater stability compared to the extrapolated counterpart.
- Neural Tangent Kernel (NTK) Interpolation. In contrast to linear interpolation, NTK Interpolation alters the base of RoPE, effectively modifying the rotational "speed" of each dimension of $\operatorname{RoPE}$ (Peng \& Quesnelle, 2023). Specifically, the original $\theta_{j}=10000^{-2 j / d}, j \in\{0,1, \ldots, d / 2-1\}$ in RoPE is transformed into $\theta_{j}^{\prime}=(10000 \lambda)^{-2 j / d}$, where $\lambda=\alpha^{d / d-2}$. It is noteworthy that the value of $\lambda$ is chosen to ensure that $m \theta_{d / 2-1}^{\prime}=(m / \alpha) \theta_{d / 2-1}$.
- YaRN Interpolation. Different from Linear and NTK interpolation that treat each dimension of RoPE equally, YaRN (Peng et al., 2023) employs a ramp function to combine Linear and NTK interpolation at varying proportions across different dimensions. Simultaneously, it introduces a temperature factor to mitigate distribution shift of attention matrix caused by long inputs.


### 3.2 Proposed Approach: PoSitional SKIP-WISE Training (POSE)

Although position interpolation effectively addresses out-of-distribution position indices, extending to an extreme length by fine-tuning on context window of this size remains impractical, owing to the quadratic growth in computational complexity of attention as sequence length increases. Instead, we explore to train within the original context window $L_{c}$ and achieve context window extension via manipulating position indices to simulate longer inputs.

There are two designing desiderata for this endeavor: First, to avoid out-of-distribution positions during inference, the relative distance of manipulated position indices should comprehensively cover the range of $\left\{1, \ldots, L_{t}-1\right\}$. Second, fine-tuning with the manipulated position indices should not harm the original abilities of LLMs, so the structure of manipulated position indices should closely adhere to the original structure to the greatest extent possible.

Initially, we randomly divide the original context window $L_{c}$ into $N$ chunks $c_{0}, c_{1}, \ldots, c_{N-1}$, each with lengths $l_{0}, l_{1}, \ldots, l_{N-1}$, where $\sum_{i=0}^{N-1} l_{i}=L_{c}$. We introduce the starting index $s t_{i}$ for each chunk $c_{i}$, which facilitates the formulation of its position indices as follows:

$$
\begin{equation*}
\operatorname{Pos}\left(c_{i}\right)=\left\{s t_{i}, s t_{i}+1, \ldots, s t_{i}+l_{i}-1\right\}, \quad s t_{i}=\sum_{j=0}^{i-1} l_{j} \tag{3}
\end{equation*}
$$

Subsequently, we employ the discrete uniform distribution $\mathcal{U}(S)$ to sample a skipping bias term $u_{i} \sim \mathcal{U}\left(\left\{u_{i-1}, \ldots, L_{t}-L_{c}\right\}\right)$ for each chunk $c_{i}$. This bias term is applied to the corresponding chunk to transform the original position indices into:

$$
\begin{equation*}
\operatorname{PoSE}\left(c_{i}\right)=\left\{u_{i}+s t_{i}, u_{i}+s t_{i}+1, \ldots, u_{i}+s t_{i}+l_{i}-1\right\} \tag{4}
\end{equation*}
$$

Note that the constraint of $u_{i} \geq u_{i-1}$ is applied to prevent position index overlaps between chunks.

Intuitively, the introduction of skipping bias terms exposes model to a more diverse range of relative positions. To achieve comprehensive coverage of the target context window, we re-sample both the
length and skipping bias term of every chunk for each training example. Moreover, the continuity of position indices within each chunk closely resembles the structure employed during pre-training. Consequently, fine-tuning the model on these new position indices for language modeling does not compromise its original capabilities.

Concerning the text contained within each chunk, a similar procedure is followed to select continuous spans of tokens from the input text $\boldsymbol{x}=\left\{x_{0}, x_{1}, \ldots, x_{L_{x}}\right\}$. To elaborate, we begin by sampling a bias term $v_{i} \sim \mathcal{U}\left(\left\{v_{i-1}, \ldots, L_{x}-L_{c}\right)\right.$ followed by assigning the content of chunk $c_{i}$ as below:

$$
\begin{equation*}
c_{i}=\boldsymbol{x}\left[v_{i}+s t_{i}: v_{i}+s t_{i}+l_{i}\right] \tag{5}
\end{equation*}
$$

Notably, we have also explored other assigning strategy of $v_{i}$, including scenarios where $v_{i}=0$, which results in genuinely continuous content for the chunks, or $v_{i}=u_{i}$, aligning the manipulated position indices with actual positions in the original text. However, we observe that these variations have relatively little impact on the outcomes of fine-tuning.

After position indices and content for each chunk are settled, we perform position interpolation for stabilized fine-tuning. For simplicity, We set the initial bias terms $u_{0}$ and $v_{0}$ to 0 . In terms of chunk number $N$, we view it as an trade-off between efficiency and effectiveness. Because an increase in the number of chunks will further deviates from the position structure of pre-training, which may harm the ability acquired during pre-training. Hence, in this paper we set $N$ to 2 , exposing the models to a wider range of relative positions, while adhering as close to the original position structure as possible. (See Appendxi A and B for further discussion of $v_{i}$ and $N$.)

## 4 EXPERIMENTS

In this section, we conduct experiments to verify the effectiveness of PoSE for context window extension. Our method demonstrates impressive results on context lengths of both $16 \mathrm{k}$ and $32 \mathrm{k}$ for language modeling as well as passkey retrieval. Other advantages of PoSE are discussed in Section 5.

### 4.1 SETUPS

Training Procedure. For each setting in the main experiments, we train LLaMA-7B with the next token prediction objective. This training process comprises 1,000 steps, employing a global batch size of 64 on 8 V100 GPUs using Deepspeed ZeRO stage 3 (Rajbhandari et al., 2020). We use learning rate $2 e^{-5}$ and a linear scheduler, with 10 warmup steps. We use AdamW optimizer with its default hyperparameters setup. The fine-tuning dataset is sourced from The Pile (Gao et al., 2020), with a minimum length requirement of 2,048 tokens. Our default choice for interpolation strategies is linear interpolation. For evaluation, we use a single A100 GPU. Flash Attention V2 (Dao, 2023) is applied, making it possible to evaluate long documents of up to $128 \mathrm{k}$ tokens $(\mathrm{k}=1,024)$

Evaluation Tasks and Datasets. We examine the ability of long text modeling on two tasks: language modeling and passkey retrieval. The language modeling task is a fundamental task that reflects the overall capability of a model in handling long text. Passkey retrieval, on the other hand, can effectively measure the maximum distance that a token can attend to during the inference stage. We evaluate language modeling on GovReport (Huang et al., 2021) and Proof-pile (Zhangir et al., 2022) datasets. For passkey retrieval, we follow Mohtashami \& Jaggi (2023) to construct synthetic prompts for evaluation.

Baseline Methods. We compare our PoSE training method against following baselines:

- Full-length fine-tuning takes input tokens of target length for fine-tuning. For this method, computation complexity scales quadratically with target context window size. Following Chen et al. (2023a) and Peng et al. (2023), we perform PI before fine-tuning LLMs on inputs of target length.
- RandPos (Ruoss et al., 2023) is initially designed to train an encoder-only model from scratch for length extrapolation. However, since it shares similar idea of simulating longer sequences via changing position indices, we include it for a comprehensive comparison. Given the original / target context window length $L_{c} / L_{t}$, it uniquely samples $L_{c}$ positions from the set $\left\{0, \ldots, L_{t}-1\right\}$, arranges them in ascending order, and employs them as new position indices for training. For fair comparison, we also apply PI for this method.

Table 1: Perplexity of models trained with different methods. We conduct evaluation on the GovReport and Proof-pile datasets, varying evaluation context window size from 2k to $32 \mathrm{k}$. Our PoSE, with a fixed training window size of $2 k$, effectively extended to a target context size of $16 k / 32 k$ for inference while receiving only minimal performance degradation compared to Full-length.

| Method | Context size <br> Train / Target | GovReport |  |  |  |  | Proof-pile |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $2 k$ | $4 \mathbf{k}$ | $8 \mathbf{k}$ | $16 \mathrm{k}$ | $32 \mathbf{k}$ | $2 \mathbf{k}$ | $4 k$ | $8 \mathbf{k}$ | 16k | 32k |
| Original | $-1-$ | 4.74 | $>10^{3}$ | $>10^{3}$ | $>10^{3}$ | $>10^{3}$ | 2.83 | $>10^{3}$ | $>10^{3}$ | $>10^{3}$ | $>10^{3}$ |
| Full-length | $16 \mathrm{k} / 16 \mathrm{k}$ | 4.87 | 4.70 | 4.61 | 4.59 | - | 2.93 | 2.71 | 2.58 | 2.53 | - |
| RandPos | $2 \mathrm{k} / 16 \mathrm{k}$ <br> $2 \mathrm{k} / 32 \mathrm{k}$ | 11.63 <br> 93.43 | 11.17 <br> 95.85 | 11.54 <br> 91.79 | 15.16 <br> 93.22 | 97.57 | 7.26 <br> 60.74 | 6.83 <br> 63.54 | 6.76 <br> 60.56 | 7.73 <br> 63.15 | - <br> 66.47 |
| PoSE (Ours) | $2 \mathrm{k} / 16 \mathrm{k}$ <br> $2 \mathrm{k} / 32 \mathrm{k}$ | 4.84 <br> 4.91 | 4.68 <br> 4.76 | 4.60 <br> 4.68 | 4.60 <br> 4.64 | 4.66 | 2.95 <br> 3.01 | 2.74 <br> 2.78 | 2.61 <br> 2.66 | 2.60 <br> 2.60 | ![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-06.jpg?height=84&width=98&top_left_y=788&top_left_x=1644) |

### 4.2 LANGUAGE MoDELING

First, we investigate the impacts of different fine-tuning methods on long sequence language modeling using the GovReport and Proof-pile datasets. GovReport is a summarization dataset comprising 19,402 reports published by the Congress and the U.S. Government, with an average document length of 7,866 tokens. We randomly select 50 reports containing more than 32,768 tokens for evaluation. Similarly, Proof-pile is a $13 \mathrm{~GB}$ mathematical dataset of long mathematical documents. In line with the approach taken for GovReport, we choose 50 samples from Proof-pile that contain more than 32,768 tokens for evaluation.

Table 1 presents the results of scaling to $16 \mathrm{k}$ and $32 \mathrm{k}$ using Full-length, RandPos, and PoSE training method, each with linear interpolation (See Appendix C for results of NTK and YaRN) . For each scaled model, as well as the Original LLaMA model, we report perplexity scores at various evaluation context window sizes, ranging from $2 \mathrm{k}$ to $32 \mathrm{k}$, employing the sliding window approach proposed by Press et al. (2021). For evaluation efficiency, we set the stride of the sliding window to 1,024 .

First, we observe an overall decreasing trend of perplexity for both models scaled to $16 \mathrm{k}$ and $32 \mathrm{k}$ via PoSE as evaluation context window size increases, proving their abilities to leverage longer context. Second, with significantly shorter context length during fine-tuning, our PoSE achieves comparable results with Full-length, consolidating its effectiveness. Third, our method achieves much stronger results than RandPos. We suppose it is because our manipulated position indices closely resembles that of pre-training, hereby preserving the pre-trained language modeling ability to the greatest extent.

We also notice that all the scaling methods suffers certain performance degradation as the supported context length increases. We perceive this as a trade-off between the quantity of tokens the model can process and the level of granularity in the attention the model can pay to each individual token.

### 4.3 PASSKEY RETRIEVAL FOR EFFECTIVE CONTEXT WINDOW

To effectively measure the maximum distance that a token can attend to during the inference stage, we adopt the passkey retrieval test proposed by Mohtashami \& Jaggi (2023). In this test, models are tasked with recovering a random passkey hidden within a lengthy document. Prompt template used for this task is presented in Figure 2a.

Specifically, we compare the original LLaMA model with the PoSE-extended versions for $16 \mathrm{k}$ and $32 \mathrm{k}$ context. For each model, we vary the prompt length from $2 \mathrm{k}$ to $32 \mathrm{k}$. For each length, we conduct the passkey retrieval test for 50 times, with a random passkey of 5 digits generated and placed at a random position inside the prompt. We also include results from Full-length, RandPos, and PI-only (position interpolation without fine-tuning). Figure 2 b illustrates the results. For the Original, PI-only, and RandPos models, their retrieval accuracy rapidly drop to 0 when the context exceeds $2 \mathrm{k}$. In contrast, both PoSE-16k / 32k models managed to maintain a high retrieval accuracy $(\geq 90 \%)$ within their respective target context window, comparable to Full-length. This indicates that models trained via PoSE genuinely possess the capability to attend to all tokens within the extended context windows.

There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat $x$ times)

The pass key is 81501 . Remember it. 81501 is the pass key.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat y times)

What is the pass key? The pass key is

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-07.jpg?height=425&width=705&top_left_y=281&top_left_x=1054)

(b)

(a)

Figure 2: (a) Prompt template used for passkey retrieval; (b) Retrieval accuracy for the PoSE-extended $16 \mathrm{k} / 32 \mathrm{k}$ models, compared with other baselines. Both PoSE-extended models maintain a high retrieval accuracy ( $\geq 90 \%$ ) within their respective context window.

## 5 ANALYSIS

In this section, we analyze the advantages of PoSE, including 1) memory and time efficiency; 2) compatibility with all RoPE-based LLMs and diverse interpolation strategies; 3) potential for extremely-long context. In Section 5.4, We also verify that model performance within the original context window only receives minimal degradation.

### 5.1 MEMORY AND TIME EFFICIENCY

We study the memory and time efficiency of PoSE compared with Full-length fine-tuning. For each method, we scale LLaMA-7B to $4 \mathrm{k} / 8 \mathrm{k} / 16 \mathrm{k}$ through 1,000 training steps with a global batch size of 16 on 8 V100 GPUs. Experiment results are demonstrated in Figure 3. Figure 3(a) and (b) respectively illustrates memory and time consumption for 1,000 steps of Full-length versus PoSE. While the training cost of Full-length increases rapidly with target window length, PoSE only requires a fixed quota of memory and time for context extension, which is significantly lower. Figure 3(c) further compares model perplexity of the two training methods at different steps on GovReport. Notably, both models achieve relatively low perplexity levels within the initial 100 training steps. Moreover, at each step, our proposed PoSE, while requiring only a training context size of $2 \mathrm{k}$ tokens, exhibits very close language modeling ability to Full-length fine-tuning, which requires an extended training context of $16 \mathrm{k}$. We did not experiment with context window of $32 \mathrm{k}$ or above, because V100 machines cannot afford full fine-tuning of these lengths. But it can be expected that the overhead ration between Full-leng and PoSE will become more exaggerated as target length increases. Consequently, we can confidently assert that our proposed approach is both memory and time-efficient.

### 5.2 COMPATIBILITY WITH RoPE-BASED LLMs AND DIVERSE INTERPOLATION STRATEGIES

We also delve into the effectiveness of PoSE when applied to different RoPE-based LLMs, as well as various interpolation strategies. Specifically, we employ PoSE on four distinct models: LLaMA-7B, LLaMA2-7B, GPT-J-6B, and Baichuan2-7B, all of which encompasses RoPE in their architectures. The original context size of LLaMA-7B and GPT-J-6B is $2 \mathrm{k}$, while that of LLaMA2-7B and Baichuan2-7B is 4k. For each model, we examine the integration with Linear, NTK, and YaRN interpolation, as well as the original version for comparative purposes. The same GovReport dataset as described in Section 4.2 is utilized. The test set is truncated to the first $1 \mathrm{k}$ to $16 \mathrm{k}$ tokens for plotting the perplexity curve, as depicted in Figure 4. First, it is evident that PoSE is effective across all four models and three interpolation strategies, as evidenced by the low perplexities achieved by all 12 combinations in comparison to the 4 original model. Second, we observe that NTK and YaRN interpolation generally yields superior results compared to Linear interpolation. However, it is noteworthy that NTK exhibits a significant increase in perplexity after a certain turning point, which occurs prior to reaching the target context length. This behavior is consistent with previous findings, indicating that for a given scaling factor $\alpha$, NTK cannot genuinely expand the context window by $\alpha$ times (Peng \& Quesnelle, 2023; Quesnelle, 2023; Peng et al., 2023).

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-08.jpg?height=477&width=1391&top_left_y=268&top_left_x=367)

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-08.jpg?height=412&width=380&top_left_y=282&top_left_x=379)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-08.jpg?height=418&width=376&top_left_y=279&top_left_x=777)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-08.jpg?height=414&width=580&top_left_y=281&top_left_x=1163)

(c)

Figure 3: Full-length fine-tuning v.s. PoSE in terms of (a) Memory and (b) Time consumption for extending LLaMA-7B from $2 \mathrm{k}$ to $4 \mathrm{k} / 8 \mathrm{k} / 16 \mathrm{k}$ context, each finishing 1000 training steps. (c) Perplexity of both $16 \mathrm{k}$-context models at every training steps. We show that PoSE takes a constantly reduced time and memory for context extension, while attaining a comparable level of PPL performance with Full-length fine-tuning at each step.
![](https://cdn.mathpix.com/cropped/2024_06_04_282b701db7febcf7af8ag-08.jpg?height=392&width=1392&top_left_y=1002&top_left_x=366)

Figure 4: Perplexity of LLaMA-7B, LLaMA2-7B, GPT-J-6B, Baichuan2-7B extended to 16k via PoSE with Linear / NTK / YaRN interpolation, along with the Original model. The consistently low perplexity observed across all nine combinations serves as an indication of the effectiveness of our method across RoPE-based LLMs and diverse interpolation strategies.

### 5.3 POTENTIAL FOR EXTREMELY-LONG CONTEXT

Because PoSE only takes a fixed context window at training stage to extend to target context window size, we can promisingly extend LLMs to support infinite input lengths using this method. In this section, we extend context window size to $96 \mathrm{k}$ and $128 \mathrm{k}$ to explore PoSE's potential for extreme context window extension. Given the need to evaluate on extremely long documents, we have opted to employ two book datasets, namely Books3 (Presser, 2020) and Gutenberg (PG-19) (Rae et al., 2019). Both of these datasets consist of extensive collections of literary works, rendering them well-suited subjects for the assessment of long-range modeling. For our evaluation, we randomly selected 20 books from each dataset, each containing more than $128 \mathrm{k}$ tokens.

Fine-tuning LLaMA models using PoSE, we experimented with Linear / NTK / YaRN interpolation for both the $96 \mathrm{k}$ and $128 \mathrm{k}$ models. To calculate perplexity, we adhere to the sliding window strategy adopted in Section 4.2, with an increased sliding window step of 16k to enhance evaluation efficiency. The outcomes of these experiments are detailed in Table 2. It is observe that, PoSE successfully extends the model's context window to $96 \mathrm{k}$ when coupled with Linear interpolation, and further extends the context window to $128 \mathrm{k}$ when paired with YaRN. These promising results consolidates the effectiveness of PoSE for extreme context window extension.

### 5.4 EVALUATION OF CAPABILITY ON ORIGINAL CONTEXT WINDOW

In this section, we examine the capabilities of the PoSE-extended models on the original context window using standard benchmarks. We combine the Hugging Face Open LLM Leaderboard (Face, 2023)

Table 2: Perplexity of models extended to extreme context size via PoSE on PG-19 and Books3. We show that our training method can effectively extend context window size to $128 \mathrm{k}$ when combined with YaRN interpolation.

| Model | Gutenberg $($ PG-19) |  |  |  |  | Books3 |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathbf{3 2 k}$ | $\mathbf{6 4 k}$ | $\mathbf{9 6 k}$ | $\mathbf{1 2 8 k}$ |  | $\mathbf{3 2 k}$ | $\mathbf{6 4 k}$ | $\mathbf{9 6 k}$ | $\mathbf{1 2 8 k}$ |
| PoSE-Linear-96k | 10.18 | 11.11 | 13.57 | - |  | 9.98 | 10.90 | 13.42 | - |
| PoSE-NTK-96k | 7.98 | 20.39 | 38.73 | - |  | 8.29 | 20.82 | 40.39 | - |
| PoSE-YaRN-96k | 8.31 | 8.65 | 9.36 | - |  | 8.90 | 9.40 | 10.38 | - |
| PoSE-Linear-128k | 16.90 | 22.47 | 26.77 | 31.18 |  | 26.20 | 43.62 | 57.08 | 70.87 |
| PoSE-NTK-128k | 8.04 | 14.84 | 29.48 | 34.80 |  | 8.34 | 16.04 | 31.42 | 37.00 |
| PoSE-YaRN-128k | 9.32 | 10.36 | 10.77 | 11.33 |  | 10.56 | 12.30 | 13.07 | 13.81 |

Table 3: Performance of PoSE-extended LLaMA model on standard benchmarks in comparison with Full-length fine-tuning and the original LLaMA. We show that PoSE-extended models exhibit only marginal performance degradation compared with Full-length fine-tuning and the original version.

| Model | Zero-Shot |  |  |  |  | Few-Shot |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | BoolQ | PIQA | WinoGrande | TruthfulQA |  | ARC-C | HellaSwag |
| Original LLaMA | 75.11 | 78.67 | 69.85 | 34.08 |  | 51.19 | 77.75 |
| Full-Linear-16k | 70.95 | 77.64 | 69.06 | 31.89 |  | 48.55 | 74.19 |
| Full-NTK-16k | 75.80 | 78.08 | 68.98 | 33.83 |  | 48.81 | 76.57 |
| Full-YaRN-16k | 73.88 | 77.64 | 68.15 | 34.12 |  | 50.60 | 77.18 |
| PoSE-Linear-16k | 74.50 | 78.13 | 68.59 | 32.05 |  | 48.29 | 75.56 |
| PoSE-NTK-16k | 74.28 | 78.24 | 68.90 | 33.89 |  | 49.83 | 76.82 |
| PoSE-YaRN-16k | 74.28 | 78.02 | 69.06 | 34.00 |  | 49.23 | 77.04 |
| PoSE-Linear-128k | 67.71 | 76.22 | 67.56 | 36.16 |  | 39.93 | 66.04 |
| PoSE-NTK-128k | 75.35 | 78.18 | 68.98 | 32.71 |  | 49.66 | 76.19 |
| PoSE-YaRN-128k | 73.61 | 77.80 | 70.01 | 34.47 |  | 48.46 | 75.54 |

with a subset of LLaMA benchmarks to assess zero-shot and few-shot performance. For zero-shot evaluation, we employ BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Keisuke et al., 2019), and TruthfulQA (Lin et al., 2022). For few-shot evaluation, we utilize 25-shot ARCChallenge (Clark et al., 2018) and 10-shot HellaSwag (Zellers et al., 2019). Our evaluation metrics are benchmark-specific: for BoolQ, PIQA, and WinoGrande, we report accuracy; for TruthfulQA, we report mc2; and for ARC-C and HellaSwag, we report normalized accuracy.

Table 3 summarizes the results. It is observed that, PoSE-extended models exhibit only marginal performance degradation compared with Full-length fine-tuning and the original LLaMA, with the only exception of the $128 \mathrm{k}$ model employing linear interpolation. This indicates that while extending context window size, PoSE effectively preserves original language comprehension ability.

## 6 CONCLUSION

In this paper, we introduce Positional Skip-wisE (PoSE) training to efficiently extend the context window of Large Language Models. PoSE simulates long inputs by manipulating position indices, thereby requiring only the original context window for fine-tuning, successfully decoupling train length and target length. Experiments have shown that, compared with fine-tuning on the full length, PoSE greatly reduces memory and time overhead. Taking advantage of this, we have managed to extend LLaMA model to $128 \mathrm{k}$ on $8 \mathrm{~V} 100$ GPUs, observing only minimal performance degradation on standard benchmarks. We have also empirically verified that PoSE is compatible with all RoPE-based LLMs and position interpolation strategies.

## 7 ACKNOWLEDGEMENT

We thank all the anonymous reviewers for their helpful comments on this paper. We thank Xueguang Ma, Yang Ouyang, Pengyun Yue, Hanyu Li, Fangwei Zhu for the thoughtful discussion. This work was partially supported by the Okawa Research Grant.

## REFERENCES

Baichuan. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023. URL https://arxiv.org/abs/2309.10305.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:11079-11091, 2022.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. ArXiv, abs/2306.15595, 2023a.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023b.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In $N A A C L$, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai 2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 2978-2988, 2019.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

Hugging Face. Open llm leaderboard. https://huggingface.co/spaces/HuggingFaceH4/open_ llm_leaderboard, 2023.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv preprint arXiv:2308.16137, 2023

Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1382-1390, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp. 99 .

Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1419-1436, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main. 112 .

kaiokendev. Things i'm learning while training superhot. https://kaiokendev.github.io/til\# extending-context-to-8k, 2023.

Sakaguchi Keisuke, Le Bras Ronan, Bhagavatula Chandra, and Choi Yejin. Winogrande: An adversarial winograd schema challenge at scale. 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023.

Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022.

Mukai Li, Shansan Gong, Jiangtao Feng, Yiheng Xu, Jun Zhang, Zhiyong Wu, and Lingpeng Kong. In-context learning with many demonstration examples. arXiv preprint arXiv:2302.04931, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3214-3252, 2022.

Amirkeivan Mohtashami and Martin Jaggi. Landmark attention: Random-access infinite context length for transformers, 2023.

Bowen Peng and Jeffrey Quesnelle. Ntk-aware scaled rope allows llama models to have extended $(8 \mathrm{k}+$ ) context size without any fine-tuning and minimal perplexity degradation. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_ rope_allows_llama_models_to_have, 2023.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models, 2023.

Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

Shawn Presser. https://twitter.com/theshawwn/status/1320282149329784833, 2020.

Jeffrey Quesnelle. Dynamically scaled rope further increases performance of long context llama with zero fine-tuning. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/ dynamically_scaled_rope_further_increases/, 2023.

Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, Chloe Hillier, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. arXiv preprint, 2019.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-16. IEEE, 2020.

Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, and Joel Veness. Randomized positional encodings boost length generalization of transformers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1889-1903, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-short. 161.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14590-14604, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: $10.18653 / v 1 / 2023$.acl-long. 816 .

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłoś. Focused transformer: Contrastive training for context scaling. arXiv preprint arXiv:2307.03170, 2023.

Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. arXiv preprint arXiv:2306.07174, 2023.

Yuhuai Wu, Markus N Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. arXiv preprint arXiv:2203.08913, 2022.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791-4800, 2019.

Azerbayev Zhangir, Ayers Edward, and Bartosz Piotrowski. Proof-pile. https://github.com/ zhangir-azerbayev/proof-pile, 2022.

Yucheng Zhou, Tao Shen, Xiubo Geng, Chongyang Tao, Guodong Long, Can Xu, and Daxin Jiang. Fine-grained distillation for long document retrieval. arXiv preprint arXiv:2212.10423, 2022.

Table 4: Comparison of different methods for choosing $v_{i}$. We report perplexity with evaluation context window ranging from $2 \mathrm{k}$ to $16 \mathrm{k}$. We show that these variations have relatively little impact on the outcomes of fine-tuning.

| Method | GovReport |  |  |  |  | Proof-pile |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathbf{2 k}$ | $\mathbf{4 k}$ | $\mathbf{8 k}$ | $\mathbf{1 6 k}$ |  | $\mathbf{2 k}$ | $\mathbf{4 k}$ | $\mathbf{8 k}$ | $\mathbf{1 6 k}$ |
| $v_{i} \sim \mathcal{U}(\ldots)$ | 4.84 | 4.68 | 4.60 | 4.60 |  | 2.95 | 2.74 | 2.61 | 2.60 |
| $v_{i}=0$ | 4.85 | 4.72 | 4.64 | 4.68 |  | 2.96 | 2.75 | 2.63 | 2.61 |
| $v_{i}=u_{i}$ | 4.84 | 4.68 | 4.60 | 4.60 |  | 2.95 | 2.73 | 2.60 | 2.56 |
</end of paper 2>


