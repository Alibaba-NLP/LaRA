<paper 0>
# ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS 

Kevin Clark<br>Stanford University<br>kevclark@cs.stanford.edu

Christopher D. Manning<br>Stanford University \& CIFAR Fellow<br>manning@cs.stanford.edu

Quoc V. Le<br>Google Brain<br>qvl@google.com

Minh-Thang Luong<br>thangluong@google.com


#### Abstract

Masked language modeling (MLM) pre-training methods such as BERT corrupt the input by replacing some tokens with [MASK] and then train a model to reconstruct the original tokens. While they produce good results when transferred to downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a more sample-efficient pre-training task called replaced token detection. Instead of masking the input, our approach corrupts it by replacing some tokens with plausible alternatives sampled from a small generator network. Then, instead of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments demonstrate this new pre-training task is more efficient than MLM because the task is defined over all input tokens rather than just the small subset that was masked out. As a result, the contextual representations learned by our approach substantially outperform the ones learned by BERT given the same model size, data, and compute. The gains are particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained using 30x more compute) on the GLUE natural language understanding benchmark. Our approach also works well at scale, where it performs comparably to RoBERTa and XLNet while using less than $1 / 4$ of their compute and outperforms them when using the same amount of compute.


## 1 INTRODUCTION

Current state-of-the-art representation learning methods for language can be viewed as learning denoising autoencoders (Vincent et al., 2008). They select a small subset of the unlabeled input sequence (typically 15\%), mask the identities of those tokens (e.g., BERT; Devlin et al. (2019)) or attention to those tokens (e.g., XLNet; Yang et al. (2019)), and then train the network to recover the original input. While more effective than conventional language-model pre-training due to learning bidirectional representations, these masked language modeling (MLM) approaches incur a substantial compute cost because the network only learns from $15 \%$ of the tokens per example.

As an alternative, we propose replaced token detection, a pre-training task in which the model learns to distinguish real input tokens from plausible but synthetically generated replacements. Instead of masking, our method corrupts the input by replacing some tokens with samples from a proposal distribution, which is typically the output of a small masked language model. This corruption procedure solves a mismatch in BERT (although not in XLNet) where the network sees artificial [MASK] tokens during pre-training but not when being fine-tuned on downstream tasks. We then pre-train the network as a discriminator that predicts for every token whether it is an original or a replacement. In contrast, MLM trains the network as a generator that predicts the original identities of the corrupted tokens. A key advantage of our discriminative task is that the model learns from all input tokens instead of just the small masked-out subset, making it more computationally efficient. Although our
![](https://cdn.mathpix.com/cropped/2024_05_26_9c8a589047e9bed49e4eg-02.jpg?height=626&width=1398&top_left_y=267&top_left_x=360)

Figure 1: Replaced token detection pre-training consistently outperforms masked language model pre-training given the same compute budget. The left figure is a zoomed-in view of the dashed box.

approach is reminiscent of training the discriminator of a GAN, our method is not adversarial in that the generator producing corrupted tokens is trained with maximum likelihood due to the difficulty of applying GANs to text (Caccia et al. 2018).

We call our approach ELECTRA for "Efficiently Learning an Encoder that Classifies Token Replacements Accurately." As in prior work, we apply it to pre-train Transformer text encoders (Vaswani et al., 2017) that can be fine-tuned on downstream tasks. Through a series of ablations, we show that learning from all input positions causes ELECTRA to train much faster than BERT. We also show ELECTRA achieves higher accuracy on downstream tasks when fully trained.

Most current pre-training methods require large amounts of compute to be effective, raising concerns about their cost and accessibility. Since pre-training with more compute almost always results in better downstream accuracies, we argue an important consideration for pre-training methods should be compute efficiency as well as absolute downstream performance. From this viewpoint, we train ELECTRA models of various sizes and evaluate their downstream performance vs. their compute requirement. In particular, we run experiments on the GLUE natural language understanding benchmark (Wang et al., 2019) and SQuAD question answering benchmark (Rajpurkar et al. 2016). ELECTRA substantially outperforms MLM-based methods such as BERT and XLNet given the same model size, data, and compute (see Figure 1). For example, we build an ELECTRA-Small model that can be trained on 1 GPU in 4 days ${ }^{2}$ ELECTRA-Small outperforms a comparably small BERT model by 5 points on GLUE, and even outperforms the much larger GPT model (Radford et al. , 2018). Our approach also works well at large scale, where we train an ELECTRA-Large model that performs comparably to RoBERTa (Liu et al. 2019) and XLNet (Yang et al. 2019), despite having fewer parameters and using $1 / 4$ of the compute for training. Training ELECTRA-Large further results in an even stronger model that outperforms ALBERT (Lan et al., 2019) on GLUE and sets a new state-of-the-art for SQuAD 2.0. Taken together, our results indicate that the discriminative task of distinguishing real data from challenging negative samples is more compute-efficient and parameter-efficient than existing generative approaches for language representation learning.

## 2 METHOD

We first describe the replaced token detection pre-training task; see Figure 2 for an overview. We suggest and evaluate several modeling improvements for this method in Section 3.2[^0]

![](https://cdn.mathpix.com/cropped/2024_05_26_9c8a589047e9bed49e4eg-03.jpg?height=360&width=1315&top_left_y=270&top_left_x=405)

Figure 2: An overview of replaced token detection. The generator can be any model that produces an output distribution over tokens, but we usually use a small masked language model that is trained jointly with the discriminator. Although the models are structured like in a GAN, we train the generator with maximum likelihood rather than adversarially due to the difficulty of applying GANs to text. After pre-training, we throw out the generator and only fine-tune the discriminator (the ELECTRA model) on downstream tasks.

Our approach trains two neural networks, a generator $G$ and a discriminator $D$. Each one primarily consists of an encoder (e.g., a Transformer network) that maps a sequence on input tokens $\boldsymbol{x}=$ $\left[x_{1}, \ldots, x_{n}\right]$ into a sequence of contextualized vector representations $h(\boldsymbol{x})=\left[h_{1}, \ldots, h_{n}\right]$. For a given position $t$, (in our case only positions where $x_{t}=[\mathrm{MASK}]$ ), the generator outputs a probability for generating a particular token $x_{t}$ with a softmax layer:

$$
p_{G}\left(x_{t} \mid \boldsymbol{x}\right)=\exp \left(e\left(x_{t}\right)^{T} h_{G}(\boldsymbol{x})_{t}\right) / \sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{T} h_{G}(\boldsymbol{x})_{t}\right)
$$

where $e$ denotes token embeddings. For a given position $t$, the discriminator predicts whether the token $x_{t}$ is "real," i.e., that it comes from the data rather than the generator distribution, with a sigmoid output layer:

$$
D(\boldsymbol{x}, t)=\operatorname{sigmoid}\left(w^{T} h_{D}(\boldsymbol{x})_{t}\right)
$$

The generator is trained to perform masked language modeling (MLM). Given an input $\boldsymbol{x}=$ $\left[x_{1}, x_{2}, \ldots, x_{n}\right]$, MLM first select a random set of positions (integers between 1 and $n$ ) to mask out $\boldsymbol{m}=\left[m_{1}, \ldots, m_{k}\right]$ The tokens in the selected positions are replaced with a [MASK] token: we denote this as $\boldsymbol{x}^{\text {masked }}=\operatorname{REPLACE}(\boldsymbol{x}, \boldsymbol{m},[$ MASK $])$. The generator then learns to predict the original identities of the masked-out tokens. The discriminator is trained to distinguish tokens in the data from tokens that have been replaced by generator samples. More specifically, we create a corrupted example $\boldsymbol{x}^{\text {corrupt }}$ by replacing the masked-out tokens with generator samples and train the discriminator to predict which tokens in $\boldsymbol{x}^{\text {corrupt }}$ match the original input $\boldsymbol{x}$. Formally, model inputs are constructed according to

$$
\begin{array}{ll}
m_{i} \sim \operatorname{unif}\{1, n\} \text { for } i=1 \text { to } k & \boldsymbol{x}^{\text {masked }}=\operatorname{REPLACE}(\boldsymbol{x}, \boldsymbol{m},[\text { MASK }]) \\
\hat{x}_{i} \sim p_{G}\left(x_{i} \mid \boldsymbol{x}^{\text {masked }}\right) \text { for } i \in \boldsymbol{m} & \boldsymbol{x}^{\text {corrupt }}=\operatorname{REPLACE}(\boldsymbol{x}, \boldsymbol{m}, \hat{\boldsymbol{x}})
\end{array}
$$

and the loss functions are

$$
\begin{aligned}
& \mathcal{L}_{\mathrm{MLM}}\left(\boldsymbol{x}, \theta_{G}\right)=\mathbb{E}\left(\sum_{i \in \boldsymbol{m}}-\log p_{G}\left(x_{i} \mid \boldsymbol{x}^{\text {masked }}\right)\right) \\
& \mathcal{L}_{\text {Disc }}\left(\boldsymbol{x}, \theta_{D}\right)=\mathbb{E}\left(\sum_{t=1}^{n}-\mathbb{1}\left(x_{t}^{\text {corrupt }}=x_{t}\right) \log D\left(\boldsymbol{x}^{\text {corrupt }}, t\right)-\mathbb{1}\left(x_{t}^{\text {corrupt }} \neq x_{t}\right) \log \left(1-D\left(\boldsymbol{x}^{\text {corrupt }}, t\right)\right)\right)
\end{aligned}
$$

Although similar to the training objective of a GAN, there are several key differences. First, if the generator happens to generate the correct token, that token is considered "real" instead of "fake"; we found this formulation to moderately improve results on downstream tasks. More importantly, the generator is trained with maximum likelihood rather than being trained adversarially to fool the discriminator. Adversarially training the generator is challenging because it is impossible to backpropagate through sampling from the generator. Although we experimented circumventing this issue[^1]by using reinforcement learning to train the generator (see Appendix F), this performed worse than maximum-likelihood training. Lastly, we do not supply the generator with a noise vector as input, as is typical with a GAN.

We minimize the combined loss

$$
\min _{\theta_{G}, \theta_{D}} \sum_{\boldsymbol{x} \in \mathcal{X}} \mathcal{L}_{\mathrm{MLM}}\left(\boldsymbol{x}, \theta_{G}\right)+\lambda \mathcal{L}_{\text {Disc }}\left(\boldsymbol{x}, \theta_{D}\right)
$$

over a large corpus $\mathcal{X}$ of raw text. We approximate the expectations in the losses with a single sample. We don't back-propagate the discriminator loss through the generator (indeed, we can't because of the sampling step). After pre-training, we throw out the generator and fine-tune the discriminator on downstream tasks.

## 3 EXPERIMENTS

### 3.1 EXPERIMENTAL SETUP

We evaluate on the General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2019) and Stanford Question Answering (SQuAD) dataset (Rajpurkar et al., 2016). GLUE contains a variety of tasks covering textual entailment (RTE and MNLI) question-answer entailment (QNLI), paraphrase (MRPC), question paraphrase (QQP), textual similarity (STS), sentiment (SST), and linguistic acceptability (CoLA). See Appendix C for more details on the GLUE tasks. Our evaluation metrics are Spearman correlation for STS, Matthews correlation for CoLA, and accuracy for the other GLUE tasks; we generally report the average score over all tasks. For SQuAD, we evaluate on versions 1.1, in which models select the span of text answering a question, and 2.0, in which some questions are unanswerable by the passage. We use the standard evaluation metrics of Exact-Match (EM) and F1 scores. For most experiments we pre-train on the same data as BERT, which consists of 3.3 Billion tokens from Wikipedia and BooksCorpus (Zhu et al., 2015). However, for our Large model we pre-trained on the data used for XLNet (Yang et al., 2019), which extends the BERT dataset to 33B tokens by including data from ClueWeb (Callan et al.. 2009), CommonCrawl, and Gigaword (Parker et al. 2011). All of the pre-training and evaluation is on English data, although we think it would be interesting to apply our methods to multilingual data in the future.

Our model architecture and most hyperparameters are the same as BERT's. For fine-tuning on GLUE, we add simple linear classifiers on top of ELECTRA. For SQuAD, we add the questionanswering module from XLNet on top of ELECTRA, which is slightly more sophisticated than BERT's in that it jointly rather than independently predicts the start and end positions and has a "answerability" classifier added for SQuAD 2.0. Some of our evaluation datasets are small, which means accuracies of fine-tuned models can vary substantially depending on the random seed. We therefore report the median of 10 fine-tuning runs from the same pre-trained checkpoint for each result. Unless stated otherwise, results are on the dev set. See the appendix for further training details and hyperparameter values.

### 3.2 Model Extensions

We improve our method by proposing and evaluating several extensions to the model. Unless stated otherwise, these experiments use the same model size and training data as BERT-Base.

Weight Sharing We propose improving the efficiency of the pre-training by sharing weights between the generator and discriminator. If the generator and discriminator are the same size, all of the transformer weights can be tied. However, we found it to be more efficient to have a small generator, in which case we only share the embeddings (both the token and positional embeddings) of the generator and discriminator. In this case we use embeddings the size of the discriminator's hidden states ${ }^{4}$ The "input" and "output" token embeddings of the generator are always tied as in BERT.

We compare the weight tying strategies when the generator is the same size as the discriminator. We train these models for 500k steps. GLUE scores are 83.6 for no weight tying, 84.3 for tying token embeddings, and 84.4 for tying all weights. We hypothesize that ELECTRA benefits from[^2]![](https://cdn.mathpix.com/cropped/2024_05_26_9c8a589047e9bed49e4eg-05.jpg?height=508&width=1390&top_left_y=266&top_left_x=366)

Figure 3: Left: GLUE scores for different generator/discriminator sizes (number of hidden units). Interestingly, having a generator smaller than the discriminator improves results. Right: Comparison of different training algorithms. As our focus is on efficiency, the x-axis shows FLOPs rather than train steps (e.g., ELECTRA is trained for fewer steps than BERT because it includes the generator).

tied token embeddings because masked language modeling is particularly effective at learning these representations: while the discriminator only updates tokens that are present in the input or are sampled by the generator, the generator's softmax over the vocabulary densely updates all token embeddings. On the other hand, tying all encoder weights caused little improvement while incurring the significant disadvantage of requiring the generator and discriminator to be the same size. Based on these findings, we use tied embeddings for further experiments in this paper.

Smaller Generators If the generator and discriminator are the same size, training ELECTRA would take around twice as much compute per step as training only with masked language modeling. We suggest using a smaller generator to reduce this factor. Specifically, we make models smaller by decreasing the layer sizes while keeping the other hyperparameters constant. We also explore using an extremely simple "unigram" generator that samples fake tokens according their frequency in the train corpus. GLUE scores for differently-sized generators and discriminators are shown in the left of Figure 3. All models are trained for 500k steps, which puts the smaller generators at a disadvantage in terms of compute because they require less compute per training step. Nevertheless, we find that models work best with generators 1/4-1/2 the size of the discriminator. We speculate that having too strong of a generator may pose a too-challenging task for the discriminator, preventing it from learning as effectively. In particular, the discriminator may have to use many of its parameters modeling the generator rather than the actual data distribution. Further experiments in this paper use the best generator size found for the given discriminator size.

Training Algorithms Lastly, we explore other training algorithms for ELECTRA, although these did not end up improving results. The proposed training objective jointly trains the generator and discriminator. We experiment with instead using the following two-stage training procedure:

1. Train only the generator with $\mathcal{L}_{\text {MLM }}$ for $n$ steps.
2. Initialize the weights of the discriminator with the weights of the generator. Then train the discriminator with $\mathcal{L}_{\text {Disc }}$ for $n$ steps, keeping the generator's weights frozen.

Note that the weight initialization in this procedure requires having the same size for the generator and discriminator. We found that without the weight initialization the discriminator would sometimes fail to learn at all beyond the majority class, perhaps because the generator started so far ahead of the discriminator. Joint training on the other hand naturally provides a curriculum for the discriminator where the generator starts off weak but gets better throughout training. We also explored training the generator adversarially as in a GAN, using reinforcement learning to accommodate the discrete operations of sampling from the generator. See Appendix Ffor details.

Results are shown in the right of Figure 3 During two-stage training, downstream task performance notably improves after the switch from the generative to the discriminative objective, but does not end up outscoring joint training. Although still outperforming BERT, we found adversarial training to underperform maximum-likelihood training. Further analysis suggests the gap is caused by two

| Model | Train / Infer FLOPs | Speedup | Params | Train Time + Hardware | GLUE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ELMo | $3.3 \mathrm{e} 18 / 2.6 \mathrm{e} 10$ | $19 \mathrm{x} / 1.2 \mathrm{x}$ | $96 \mathrm{M}$ | 14d on 3 GTX 1080 GPUs | 71.2 |
| GPT | $4.0 \mathrm{e} 19 / 3.0 \mathrm{e} 10$ | $1.6 \mathrm{x} / 0.07 \mathrm{x}$ | $117 \mathrm{M}$ | 25d on 8 P6000 GPUs | 78.8 |
| BERT-Small | $1.4 \mathrm{e} 18 / 3.7 \mathrm{e} 9$ | $45 \mathrm{x} / 8 \mathrm{x}$ | $14 \mathrm{M}$ | $4 \mathrm{~d}$ on 1 V100 GPU | 75.1 |
| BERT-Base | $6.4 \mathrm{e} 19 / 2.9 \mathrm{e} 10$ | $1 \mathrm{x} / 1 \mathrm{x}$ | $110 \mathrm{M}$ | $4 \mathrm{~d}$ on 16 TPUv3s | 82.2 |
| ELECTRA-Small | $1.4 \mathrm{e} 18 / 3.7 \mathrm{e} 9$ | $45 \mathrm{x} / 8 \mathrm{x}$ | $14 \mathrm{M}$ | 4d on 1 V100 GPU | 79.9 |
| 50\% trained | $7.1 \mathrm{e} 17 / 3.7 \mathrm{e} 9$ | $90 \mathrm{x} / 8 \mathrm{x}$ | $14 \mathrm{M}$ | 2d on 1 V100 GPU | 79.0 |
| 25\% trained | $3.6 \mathrm{e} 17 / 3.7 \mathrm{e} 9$ | $181 \mathrm{x} / 8 \mathrm{x}$ | $14 \mathrm{M}$ | 1d on 1 V100 GPU | 77.7 |
| 12.5\% trained | $1.8 \mathrm{e} 17 / 3.7 \mathrm{e} 9$ | $36 \mathrm{x} / \mathrm{x}$ | $14 \mathrm{M}$ | 12h on 1 V100 GPU | 76.0 |
| 6.25\% trained | $8.9 \mathrm{e} 16 / 3.7 \mathrm{e} 9$ | $722 \mathrm{x} / 8 \mathrm{x}$ | $14 \mathrm{M}$ | 6h on 1 V100 GPU | 74.1 |
| ELECTRA-Base | $6.4 \mathrm{e} 19 / 2.9 \mathrm{e} 10$ | $1 \mathrm{x} / 1 \mathrm{x}$ | $110 \mathrm{M}$ | 4d on 16 TPUv3s | 85.1 |

Table 1: Comparison of small models on the GLUE dev set. BERT-Small/Base are our implementation and use the same hyperparameters as ELECTRA-Small/Base. Infer FLOPs assumes single length-128 input. Training times should be taken with a grain of salt as they are for different hardware and with sometimes un-optimized code. ELECTRA performs well even when trained on a single GPU, scoring 5 GLUE points higher than a comparable BERT model and even outscoring the much larger GPT model.

problems with adversarial training. First, the adversarial generator is simply worse at masked language modeling; it achieves $58 \%$ accuracy at masked language modeling compared to $65 \%$ accuracy for an MLE-trained one. We believe the worse accuracy is mainly due to the poor sample efficiency of reinforcement learning when working in the large action space of generating text. Secondly, the adversarially trained generator produces a low-entropy output distribution where most of the probability mass is on a single token, which means there is not much diversity in the generator samples. Both of these problems have been observed in GANs for text in prior work (Caccia et al., 2018).

### 3.3 SMaLl ModeLS

As a goal of this work is to improve the efficiency of pre-training, we develop a small model that can be quickly trained on a single GPU. Starting with the BERT-Base hyperparameters, we shortened the sequence length (from 512 to 128), reduced the batch size (from 256 to 128), reduced the model's hidden dimension size (from 768 to 256), and used smaller token embeddings (from 768 to 128). To provide a fair comparison, we also train a BERT-Small model using the same hyperparameters. We train BERT-Small for $1.5 \mathrm{M}$ steps, so it uses the same training FLOPs as ELECTRA-Small, which was trained for $1 \mathrm{M}$ steps ${ }^{5}$ In addition to BERT, we compare against two less resource-intensive pre-training methods based on language modeling: ELMo (Peters et al., 2018) and GPT (Radford et al. 2018) [6 We also show results for a base-sized ELECTRA model comparable to BERT-Base.

Results are shown in Table 1 See Appendix Dfor additional results, including stronger small-sized and base-sized models trained with more compute. ELECTRA-Small performs remarkably well given its size, achieving a higher GLUE score than other methods using substantially more compute and parameters. For example, it scores 5 points higher than a comparable BERT-Small model and even outperforms the much larger GPT model. ELECTRA-Small is trained mostly to convergence, with models trained for even less time (as little as 6 hours) still achieving reasonable performance. While small models distilled from larger pre-trained transformers can also achieve good GLUE scores (Sun et al., 2019b; Jiao et al., 2019), these models require first expending substantial compute to pre-train the larger teacher model. The results also demonstrate the strength of ELECTRA at a moderate size; our base-sized ELECTRA model substantially outperforms BERT-Base and even outperforms BERT-Large (which gets 84.0 GLUE score). We hope ELECTRA's ability to achieve strong results with relatively little compute will broaden the accessibility of developing and applying pre-trained models in NLP.[^3]

| Model | Train FLOPs | Params | CoLA | SST | MRPC | STS | QQP | MNLI | QNLI | RTE | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BERT | $1.9 \mathrm{e} 20(0.27 \mathrm{x})$ | $335 \mathrm{M}$ | 60.6 | 93.2 | 88.0 | 90.0 | 91.3 | 86.6 | 92.3 | 70.4 | 84.0 |
| RoBERTa-100K | 6.4e20 (0.90x) | 356M | 66.1 | 95.6 | $\mathbf{9 1 . 4}$ | 92.2 | 92.0 | 89.3 | 94.0 | 82.7 | 87.9 |
| RoBERTa-500K | 3.2e21 (4.5x) | 356M | 68.0 | 96.4 | 90.9 | 92.1 | 92.2 | 90.2 | 94.7 | 86.6 | 88.9 |
| XLNet | 3.9e21 (5.4x) | 360M | 69.0 | $\mathbf{9 7 . 0}$ | 90.8 | 92.2 | 92.3 | 90.8 | 94.9 | 85.9 | 89.1 |
| BERT (ours) | $7.1 \mathrm{e} 20(1 \mathrm{x})$ | $335 \mathrm{M}$ | 67.0 | 95.9 | 89.1 | 91.2 | 91.5 | 89.6 | 93.5 | 79.5 | 87.2 |
| ELECTRA-400K | $7.1 \mathrm{e} 20(1 \mathrm{x})$ | $335 \mathrm{M}$ | $\mathbf{6 9 . 3}$ | 96.0 | 90.6 | 92.1 | $\mathbf{9 2 . 4}$ | 90.5 | 94.5 | 86.8 | 89.0 |
| ELECTRA-1.75M | 3.1e21 (4.4x) | 335M | 69.1 | 96.9 | 90.8 | $\mathbf{9 2 . 6}$ | $\mathbf{9 2 . 4}$ | $\mathbf{9 0 . 9}$ | $\mathbf{9 5 . 0}$ | $\mathbf{8 8 . 0}$ | $\mathbf{8 9 . 5}$ |

Table 2: Comparison of large models on the GLUE dev set. ELECTRA and RoBERTa are shown for different numbers of pre-training steps, indicated by the numbers after the dashes. ELECTRA performs comparably to XLNet and RoBERTa when using less than $1 / 4$ of their pre-training compute and outperforms them when given a similar amount of pre-training compute. BERT dev results are from Clark et al. $(2019)$.

| Model | Train FLOPs | CoLA | SST | MRPC | STS | QQP | MNLI | QNLI | RTE | WNLI | Avg.* | Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BERT | 1.9 e20 $(0.06 x)$ | 60.5 | 94.9 | 85.4 | 86.5 | 89.3 | 86.7 | 92.7 | 70.1 | 65.1 | 79.8 | 80.5 |
| RoBERTa | $3.2 \mathrm{e} 21(1.02 \mathrm{x})$ | 67.8 | 96.7 | 89.8 | 91.9 | 90.2 | 90.8 | 95.4 | 88.2 | 89.0 | 88.1 | 88.1 |
| ALBERT | $3.1 \mathrm{e} 22(10 \mathrm{x})$ | 69.1 | $\mathbf{9 7 . 1}$ | $\mathbf{9 1 . 2}$ | 92.0 | 90.5 | $\mathbf{9 1 . 3}$ | - | 89.2 | 91.8 | 89.0 | - |
| XLNet | $3.9 \mathrm{e} 21(1.26 \mathrm{x})$ | 70.2 | $\mathbf{9 7 . 1}$ | 90.5 | $\mathbf{9 2 . 6}$ | 90.4 | 90.9 | - | 88.5 | $\mathbf{9 2 . 5}$ | 89.1 | - |
| ELECTRA | $3.1 \mathrm{e} 21(1 \mathrm{x})$ | $\mathbf{7 1 . 7}$ | $\mathbf{9 7 . 1}$ | 90.7 | 92.5 | $\mathbf{9 0 . 8}$ | $\mathbf{9 1 . 3}$ | $\mathbf{9 5 . 8}$ | $\mathbf{8 9 . 8}$ | $\mathbf{9 2 . 5}$ | $\mathbf{8 9 . 5}$ | $\mathbf{8 9 . 4}$ |

Table 3: GLUE test-set results for large models. Models in this table incorporate additional tricks such as ensembling to improve scores (see Appendix B for details). Some models do not have QNLI scores because they treat QNLI as a ranking task, which has recently been disallowed by the GLUE benchmark. To compare against these models, we report the average score excluding QNLI (Avg.*) in addition to the GLUE leaderboard score (Score). "ELECTRA" and "RoBERTa" refer to the fully-trained ELECTRA-1.75M and RoBERTa-500K models.

### 3.4 Large ModeLS

We train big ELECTRA models to measure the effectiveness of the replaced token detection pretraining task at the large scale of current state-of-the-art pre-trained Transformers. Our ELECTRALarge models are the same size as BERT-Large but are trained for much longer. In particular, we train a model for 400k steps (ELECTRA-400K; roughly 1/4 the pre-training compute of RoBERTa) and one for $1.75 \mathrm{M}$ steps (ELECTRA-1.75M; similar compute to RoBERTa). We use a batch size 2048 and the XLNet pre-training data. We note that although the XLNet data is similar to the data used to train RoBERTa, the comparison is not entirely direct. As a baseline, we trained our own BERT-Large model using the same hyperparameters and training time as ELECTRA-400K.

Results on the GLUE dev set are shown in Table 2. ELECTRA-400K performs comparably to RoBERTa and XLNet. However, it took less than 1/4 of the compute to train ELECTRA-400K as it did to train RoBERTa and XLNet, demonstrating that ELECTRA's sample-efficiency gains hold at large scale. Training ELECTRA for longer (ELECTRA-1.75M) results in a model that outscores them on most GLUE tasks while still requiring less pre-training compute. Surprisingly, our baseline BERT model scores notably worse than RoBERTa-100K, suggesting our models may benefit from more hyperparameter tuning or using the RoBERTa training data. ELECTRA's gains hold on the GLUE test set (see Table 3), although these comparisons are less apples-to-apples due to the additional tricks employed by the models (see Appendix B).

Results on SQuAD are shown in Table 4. Consistent, with the GLUE results, ELECTRA scores better than masked-language-modeling-based methods given the same compute resources. For example, ELECTRA-400K outperforms RoBERTa-100k and our BERT baseline, which use similar amounts of pre-training compute. ELECTRA-400K also performs comparably to RoBERTa-500K despite using less than 1/4th of the compute. Unsurprisingly, training ELECTRA longer improves results further: ELECTRA-1.75M scores higher than previous models on the SQuAD 2.0 bench-

| Model | Train FLOPs | Params | SQuAD $1.1 \mathrm{dev}$ |  | SQuAD 2.0 dev |  | SQuAD 2.0 tesi |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | EM |  | EM | F1 | EM |  |
| BERT-Base | $6.4 \mathrm{e} 19(0.09 \mathrm{x})$ | $110 \mathrm{M}$ | 80.8 | 88.5 | - | - | - | - |
| BERT | $1.9 \mathrm{e} 20(0.27 \mathrm{x})$ | $335 \mathrm{M}$ | 84.1 | 90.9 | 79.0 | 81.8 | 80.0 | 83.0 |
| SpanBERT | $7.1 \mathrm{e} 20(1 \mathrm{x})$ | $335 \mathrm{M}$ | 88.8 | 94.6 | 85.7 | 88.7 | 85.7 | 88.7 |
| XLNet-Base | $6.6 \mathrm{e} 19(0.09 \mathrm{x})$ | $117 \mathrm{M}$ | 81.3 | - | 78.5 | - | - | - |
| XLNet | $3.9 \mathrm{e} 21(5.4 \mathrm{x})$ | $360 \mathrm{M}$ | 89.7 | 95.1 | 87.9 | 90.6 | 87.9 | 90.7 |
| RoBERTa-100K | $6.4 \mathrm{e} 20(0.90 \mathrm{x})$ | $356 \mathrm{M}$ | - | 94.0 | - | 87.7 | - | - |
| RoBERTa-500K | $3.2 \mathrm{e} 21(4.5 \mathrm{x})$ | $356 \mathrm{M}$ | 88.9 | 94.6 | 86.5 | 89.4 | 86.8 | 89.8 |
| ALBERT | $3.1 \mathrm{e} 22(44 \mathrm{x})$ | $235 \mathrm{M}$ | 89.3 | 94.8 | 87.4 | 90.2 | 88.1 | 90.9 |
| BERT (ours) | 7.1e20 (1x) | $335 \mathrm{M}$ | 88.0 | 93.7 | 84.7 | 87.5 | - | - |
| ELECTRA-Base | $6.4 \mathrm{e} 19(0.09 x)$ | $110 \mathrm{M}$ | 84.5 | 90.8 | 80.5 | 83.3 | - | - |
| ELECTRA-400K | 7.1e20 (1x) | $335 \mathrm{M}$ | 88.7 | 94.2 | 86.9 | 89.6 | - | - |
| ELECTRA-1.75M | $3.1 \mathrm{e} 21(4.4 \mathrm{x})$ | $335 \mathrm{M}$ | 89.7 | 94.9 | 88.0 | 90.6 | 88.7 | 91.4 |

Table 4: Results on the SQuAD for non-ensemble models.

mark. ELECTRA-Base also yields strong results, scoring substantially better than BERT-Base and XLNet-Base, and even surpassing BERT-Large according to most metrics. ELECTRA generally performs better at SQuAD 2.0 than 1.1. Perhaps replaced token detection, in which the model distinguishes real tokens from plausible fakes, is particularly transferable to the answerability classification of SQuAD 2.0, in which the model must distinguish answerable questions from fake unanswerable questions.

### 3.5 EFFICIENCY ANALYSIS

We have suggested that posing the training objective over a small subset of tokens makes masked language modeling inefficient. However, it isn't entirely obvious that this is the case. After all, the model still receives a large number of input tokens even though it predicts only a small number of masked tokens. To better understand where the gains from ELECTRA are coming from, we compare a series of other pre-training objectives that are designed to be a set of "stepping stones" between BERT and ELECTRA.

- ELECTRA 15\%: This model is identical to ELECTRA except the discriminator loss only comes from the $15 \%$ of the tokens that were masked out of the input. In other words, the sum in the discriminator loss $\mathcal{L}_{\text {Disc }}$ is over $i \in \boldsymbol{m}$ instead of from 1 to $n 7$
- Replace MLM: This objective is the same as masked language modeling except instead of replacing masked-out tokens with [MASK], they are replaced with tokens from a generator model. This objective tests to what extent ELECTRA's gains come from solving the discrepancy of exposing the model to [MASK] tokens during pre-training but not fine-tuning.
- All-Tokens MLM: Like in Replace MLM, masked tokens are replaced with generator samples. Furthermore, the model predicts the identity of all tokens in the input, not just ones that were masked out. We found it improved results to train this model with an explicit copy mechanism that outputs a copy probability $D$ for each token using a sigmoid layer. The model's output distribution puts $D$ weight on the input token plus $1-D$ times the output of the MLM softmax. This model is essentially a combination of BERT and ELECTRA. Note that without generator replacements, the model would trivially learn to make predictions from the vocabulary for [MASK] tokens and copy the input for other ones.

Results are shown in Table 5. First, we find that ELECTRA is greatly benefiting from having a loss defined over all input tokens rather than just a subset: ELECTRA $15 \%$ performs much worse than ELECTRA. Secondly, we find that BERT performance is being slightly harmed from the pre-train fine-tune mismatch from [MASK] tokens, as Replace MLM slightly outperforms BERT. We note that BERT (including our implementation) already includes a trick to help with the pre-train/finetune discrepancy: masked tokens are replaced with a random token $10 \%$ of the time and are kept the[^4]

| Model | ELECTRA | All-Tokens MLM | Replace MLM | ELECTRA 15\% | BERT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GLUE score | 85.0 | 84.3 | 82.4 | 82.4 | 82.2 |

Table 5: Compute-efficiency experiments (see text for details).
![](https://cdn.mathpix.com/cropped/2024_05_26_9c8a589047e9bed49e4eg-09.jpg?height=412&width=1328&top_left_y=491&top_left_x=388)

Figure 4: Left and Center: Comparison of BERT and ELECTRA for different model sizes. Right: A small ELECTRA model converges to higher downstream accuracy than BERT, showing the improvement comes from more than just faster training.

same $10 \%$ of the time. However, our results suggest these simple heuristics are insufficient to fully solve the issue. Lastly, we find that All-Tokens MLM, the generative model that makes predictions over all tokens instead of a subset, closes most of the gap between BERT and ELECTRA. In total, these results suggest a large amount of ELECTRA's improvement can be attributed to learning from all tokens and a smaller amount can be attributed to alleviating the pre-train fine-tune mismatch.

The improvement of ELECTRA over All-Tokens MLM suggests that the ELECTRA's gains come from more than just faster training. We study this further by comparing BERT to ELECTRA for various model sizes (see Figure 4, left). We find that the gains from ELECTRA grow larger as the models get smaller. The small models are trained fully to convergence (see Figure 4 , right), showing that ELECTRA achieves higher downstream accuracy than BERT when fully trained. We speculate that ELECTRA is more parameter-efficient than BERT because it does not have to model the full distribution of possible tokens at each position, but we believe more analysis is needed to completely explain ELECTRA's parameter efficiency.

## 4 RELATED WORK

Self-Supervised Pre-training for NLP Self-supervised learning has been used to learn word representations (Collobert et al., 2011; Pennington et al., 2014) and more recently contextual representations of words though objectives such as language modeling (Dai \& Le, 2015, Peters et al., 2018 , Howard \& Ruder, 2018). BERT (Devlin et al. 2019) pre-trains a large Transformer (Vaswani et al. 2017) at the masked-language modeling task. There have been numerous extensions to BERT. For example, MASS (Song et al. |2019) and UniLM (Dong et al. 2019) extend BERT to generation tasks by adding auto-regressive generative training objectives. ERNIE (Sun et al., 2019a) and SpanBERT (Joshi et al., 2019) mask out contiguous sequences of token for improved span representations. This idea may be complementary to ELECTRA; we think it would be interesting to make ELECTRA's generator auto-regressive and add a "replaced span detection" task. Instead of masking out input tokens, XLNet (Yang et al., 2019) masks attention weights such that the input sequence is autoregressively generated in a random order. However, this method suffers from the same inefficiencies as BERT because XLNet only generates $15 \%$ of the input tokens in this way. Like ELECTRA, XLNet may alleviate BERT's pretrain-finetune discrepancy by not requiring [MASK] tokens, although this isn't entirely clear because XLNet uses two "streams" of attention during pre-training but only one for fine-tuning. Recently, models such as TinyBERT (Jiao et al. 2019) and MobileBERT (Sun et al. 2019b) show that BERT can effectively be distilled down to a smaller model. In contrast, we focus more on pre-training speed rather than inference speed, so we train ELECTRA-Small from scratch.

Generative Adversarial Networks GANs (Goodfellow et al. 2014) are effective at generating high-quality synthetic data. Radford et al. (2016) propose using the discriminator of a GAN in downstream tasks, which is similar to our method. GANs have been applied to text data (Yu et al. 2017, Zhang et al. 2017), although state-of-the-art approaches still lag behind standard maximumlikelihood training (Caccia et al. 2018; Tevet et al., 2018). Although we do not use adversarial learning, our generator is particularly reminiscent of MaskGAN (Fedus et al. 2018), which trains the generator to fill in tokens deleted from the input.

Contrastive Learning Broadly, contrastive learning methods distinguish observed data points from fictitious negative samples. They have been applied to many modalities including text (Smith \& Eisner 2005), images (Chopra et al., 2005), and video (Wang \& Gupta, 2015, Sermanet et al.||2017) data. Common approaches learn embedding spaces where related data points are similar (Saunshi et al. 2019) or models that rank real data points over negative samples (Collobert et al., 2011||Bordes et al. 2013). ELECTRA is particularly related to Noise-Contrastive Estimation (NCE) (Gutmann \& Hyvärinen, 2010), which also trains a binary classifier to distinguish real and fake data points.

Word2Vec (Mikolov et al. 2013), one of the earliest pre-training methods for NLP, uses contrastive learning. In fact, ELECTRA can be viewed as a massively scaled-up version of Continuous Bagof-Words (CBOW) with Negative Sampling. CBOW also predicts an input token given surrounding context and negative sampling rephrases the learning task as a binary classification task on whether the input token comes from the data or proposal distribution. However, CBOW uses a bag-ofvectors encoder rather than a transformer and a simple proposal distribution derived from unigram token frequencies instead of a learned generator.

## 5 CONCLUSION

We have proposed replaced token detection, a new self-supervised task for language representation learning. The key idea is training a text encoder to distinguish input tokens from high-quality negative samples produced by an small generator network. Compared to masked language modeling, our pre-training objective is more compute-efficient and results in better performance on downstream tasks. It works well even when using relatively small amounts of compute, which we hope will make developing and applying pre-trained text encoders more accessible to researchers and practitioners with less access to computing resources. We also hope more future work on NLP pre-training will consider efficiency as well as absolute performance, and follow our effort in reporting compute usage and parameter counts along with evaluation metrics.

## REFERENCES

Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In NeurIPS, 2013.

Avishek Joey Bose, Huan Ling, and Yanshuai Cao. Adversarial contrastive estimation. In ACL, 2018.

Massimo Caccia, Lucas Caccia, William Fedus, Hugo Larochelle, Joelle Pineau, and Laurent Charlin. Language GANs falling short. arXiv preprint arXiv:1811.02549, 2018.

Jamie Callan, Mark Hoy, Changkuk Yoo, and Le Zhao. Clueweb09 data set, 2009. URL https: //lemurproject.org/clueweb09.php/

Daniel M. Cer, Mona T. Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. Semeval2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In SemEval@ACL, 2017.

Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. CVPR, 2005.

Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D. Manning, and Quoc V. Le. BAM! Born-again multi-task networks for natural language understanding. In ACL, 2019.

Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel P. Kuksa. Natural language processing (almost) from scratch. JMLR, 2011.

Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. In NeurIPS, 2015.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019.

William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In $I W P @ I J C N L P, 2005$.

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. In NeurIPS, 2019.

William Fedus, Ian J. Goodfellow, and Andrew M. Dai. MaskGAN: Better text generation via filling in the \$ \qquad \$ In ICLR, 2018

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and William B. Dolan. The third pascal recognizing textual entailment challenge. In ACL-PASCAL@ACL, 2007.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.

Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In AISTATS, 2010.

Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. In $A C L, 2018$.

Shankar Iyer, Nikhil Dandekar, and Kornl Csernai. First Quora dataset release: Question pairs, 2017. URL https://data.quora.com/ First-Quora-Dataset-Release-Question-Pairs.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351, 2019.

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. SpanBERT: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529, 2019.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In ICLR Workshop Papers, 2013.

Robert Parker, David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. English gigaword, fifth edition. Technical report, Linguistic Data Consortium, Philadelphia, 2011.

Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In $E M N L P, 2014$.

Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In NAACL-HLT, 2018.

Jason Phang, Thibault Févry, and Samuel R Bowman. Sentence encoders on STILTs: Supplementary training on intermediate labeled-data tasks. arXiv preprint arXiv:1811.01088, 2018.

Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. https://blog.openai.com/language-unsupervised, 2018.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy S. Liang. Squad: 100, 000+ questions for machine comprehension of text. In EMNLP, 2016.

Nikunj Saunshi, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. A theoretical analysis of contrastive unsupervised representation learning. In ICML, 2019.

Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. Time-contrastive networks: Self-supervised learning from video. ICRA, 2017.

Noah A. Smith and Jason Eisner. Contrastive estimation: Training log-linear models on unlabeled data. In $A C L, 2005$.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, 2013.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. MASS: Masked sequence to sequence pre-training for language generation. In ICML, 2019.

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. Ernie: Enhanced representation through knowledge integration. arXiv preprint arXiv:1904.09223, 2019a.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. MobileBERT: Task-agnostic compression of bert for resource limited devices, 2019b. URL https: //openreview.net/forum?id=SJxjVaNKwB.

Guy Tevet, Gavriel Habib, Vered Shwartz, and Jonathan Berant. Evaluating text gans as language models. In NAACL-HLT, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In ICLR, 2019.

Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. ICCV, 2015.

Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018.

Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In NAACL-HLT, 2018.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229-256, 1992.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. In NeurIPS, 2019.

Lantao Yu, Weinan Zhang, Jun Wang, and Yingrui Yu. SeqGAN: Sequence generative adversarial nets with policy gradient. In AAAI, 2017.

Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Dinghan Shen, and Lawrence Carin. Adversarial feature matching for text generation. In ICML, 2017.

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. ICCV, 2015.
</end of paper 0>


<paper 1>
# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer 

Colin Raffel ${ }^{*}$<br>Noam Shazeer*<br>Adam Roberts<br>Katherine Lee*<br>Sharan Narang<br>Michael Matena<br>Yanqi Zhou<br>Wei Li<br>Peter J. Liu<br>Google, Mountain View, CA 94043, USA

CRAFFEL@GMAIL.COM

NOAM@GOOGLE.COM

ADAROB@GOOGLE.COM

KATHERINELEE@GOOGLE.COM

SHARANNARANG@GOOGLE.COM

MMATENA@GOOGLE.COM

YANQIZ@GOOGLE.COM

MWEILI@GOOGLE.COM

PETERJLIU@GOOGLE.COM

Editor: Ivan Titov


#### Abstract

Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new "Colossal Clean Crawled Corpus", we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code. ${ }^{1}$


Keywords: transfer learning, natural language processing, multi-task learning, attentionbased models, deep learning

## 1. Introduction

Training a machine learning model to perform natural language processing (NLP) tasks often requires that the model can process text in a way that is amenable to downstream learning. This can be loosely viewed as developing general-purpose knowledge that allows the model to "understand" text. This knowledge can range from low-level (e.g. the spelling[^0]or meaning of words) to high-level (e.g. that a tuba is too large to fit in most backpacks). In modern machine learning practice, providing this knowledge is rarely done explicitly; instead, it is often learned as part of an auxiliary task. For example, a historically common approach is to use word vectors (Mikolov et al., 2013b,a; Pennington et al., 2014) to map word identities to a continuous representation where, ideally, similar words map to similar vectors. These vectors are often learned through an objective that, for example, encourages co-occurring words to be positioned nearby in the continuous space (Mikolov et al., 2013b).

Recently, it has become increasingly common to pre-train the entire model on a data-rich task. Ideally, this pre-training causes the model to develop general-purpose abilities and knowledge that can then be transferred to downstream tasks. In applications of transfer learning to computer vision (Oquab et al., 2014; Jia et al., 2014; Huh et al., 2016; Yosinski et al., 2014), pre-training is typically done via supervised learning on a large labeled data set like ImageNet (Russakovsky et al., 2015; Deng et al., 2009). In contrast, modern techniques for transfer learning in NLP often pre-train using unsupervised learning on unlabeled data. This approach has recently been used to obtain state-of-the-art results in many of the most common NLP benchmarks (Devlin et al., 2018; Yang et al., 2019; Dong et al., 2019; Liu et al., 2019c; Lan et al., 2019). Beyond its empirical strength, unsupervised pre-training for NLP is particularly attractive because unlabeled text data is available en masse thanks to the Internet - for example, the Common Crawl project ${ }^{2}$ produces about 20TB of text data extracted from web pages each month. This is a natural fit for neural networks, which have been shown to exhibit remarkable scalability, i.e. it is often possible to achieve better performance simply by training a larger model on a larger data set (Hestness et al., 2017; Shazeer et al., 2017; Jozefowicz et al., 2016; Mahajan et al., 2018; Radford et al., 2019; Shazeer et al., 2018; Huang et al., 2018b; Keskar et al., 2019a).

This synergy has resulted in a great deal of recent work developing transfer learning methodology for NLP, which has produced a wide landscape of pre-training objectives (Howard and Ruder, 2018; Devlin et al., 2018; Yang et al., 2019; Dong et al., 2019), unlabeled data sets (Yang et al., 2019; Liu et al., 2019c; Zellers et al., 2019), benchmarks (Wang et al., 2019b, 2018; Conneau and Kiela, 2018), fine-tuning methods (Howard and Ruder, 2018; Houlsby et al., 2019; Peters et al., 2019), and more. The rapid rate of progress and diversity of techniques in this burgeoning field can make it difficult to compare different algorithms, tease apart the effects of new contributions, and understand the space of existing methods for transfer learning. Motivated by a need for more rigorous understanding, we leverage a unified approach to transfer learning that allows us to systematically study different approaches and push the current limits of the field.

The basic idea underlying our work is to treat every text processing problem as a "text-to-text" problem, i.e. taking text as input and producing new text as output. This approach is inspired by previous unifying frameworks for NLP tasks, including casting all text problems as question answering (McCann et al., 2018), language modeling (Radford et al., 2019), or span extraction Keskar et al. (2019b) tasks. Crucially, the text-to-text framework allows us to directly apply the same model, objective, training procedure, and decoding process to every task we consider. We leverage this flexibility by evaluating performance on a wide variety of English-based NLP problems, including question answering, document[^1]

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-03.jpg?height=501&width=1505&top_left_y=316&top_left_x=302)

Figure 1: A diagram of our text-to-text framework. Every task we consider-including translation, question answering, and classification - is cast as feeding our model text as input and training it to generate some target text. This allows us to use the same model, loss function, hyperparameters, etc. across our diverse set of tasks. It also provides a standard testbed for the methods included in our empirical survey. "T5" refers to our model, which we dub the "Text-to-Text Transfer Transformer".

summarization, and sentiment classification, to name a few. With this unified approach, we can compare the effectiveness of different transfer learning objectives, unlabeled data sets, and other factors, while exploring the limits of transfer learning for NLP by scaling up models and data sets beyond what has previously been considered.

We emphasize that our goal is not to propose new methods but instead to provide a comprehensive perspective on where the field stands. As such, our work primarily comprises a survey, exploration, and empirical comparison of existing techniques. We also explore the limits of current approaches by scaling up the insights from our systematic study (training models up to 11 billion parameters) to obtain state-of-the-art results in many of the tasks we consider. In order to perform experiments at this scale, we introduce the "Colossal Clean Crawled Corpus" (C4), a data set consisting of hundreds of gigabytes of clean English text scraped from the web. Recognizing that the main utility of transfer learning is the possibility of leveraging pre-trained models in data-scarce settings, we release our code, data sets, and pre-trained models. ${ }^{1}$

The remainder of the paper is structured as follows: In the following section, we discuss our base model and its implementation, our procedure for formulating every text processing problem as a text-to-text task, and the suite of tasks we consider. In Section 3, we present a large set of experiments that explore the field of transfer learning for NLP. At the end of the section (Section 3.7), we combine insights from our systematic study to obtain state-of-the-art results on a wide variety of benchmarks. Finally, we provide a summary of our results and wrap up with a look towards the future in Section 4.

## 2. Setup

Before presenting the results from our large-scale empirical study, we review the necessary background topics required to understand our results, including the Transformer model architecture and the downstream tasks we evaluate on. We also introduce our approach for treating every problem as a text-to-text task and describe our "Colossal Clean Crawled Corpus" (C4), the Common Crawl-based data set we created as a source of unlabeled text data. We refer to our model and framework as the "Text-to-Text Transfer Transformer" $(\mathrm{T} 5)$.

### 2.1 Model

Early results on transfer learning for NLP leveraged recurrent neural networks (Peters et al., 2018; Howard and Ruder, 2018), but it has recently become more common to use models based on the "Transformer" architecture (Vaswani et al., 2017). The Transformer was initially shown to be effective for machine translation, but it has subsequently been used in a wide variety of NLP settings (Radford et al., 2018; Devlin et al., 2018; McCann et al., 2018; Yu et al., 2018). Due to its increasing ubiquity, all of the models we study are based on the Transformer architecture. Apart from the details mentioned below and the variants we explore in Section 3.2, we do not deviate significantly from this architecture as originally proposed. Instead of providing a comprehensive definition of this model, we refer the interested reader to the original paper (Vaswani et al., 2017) or follow-up tutorials ${ }^{3,4}$ for a more detailed introduction.

The primary building block of the Transformer is self-attention (Cheng et al., 2016). Self-attention is a variant of attention (Graves, 2013; Bahdanau et al., 2015) that processes a sequence by replacing each element by a weighted average of the rest of the sequence. The original Transformer consisted of an encoder-decoder architecture and was intended for sequence-to-sequence (Sutskever et al., 2014; Kalchbrenner et al., 2014) tasks. It has recently also become common to use models consisting of a single Transformer layer stack, with varying forms of self-attention used to produce architectures appropriate for language modeling (Radford et al., 2018; Al-Rfou et al., 2019) or classification and span prediction tasks (Devlin et al., 2018; Yang et al., 2019). We empirically explore these architectural variants in Section 3.2 .

Overall, our encoder-decoder Transformer implementation closely follows its originallyproposed form (Vaswani et al., 2017). First, an input sequence of tokens is mapped to a sequence of embeddings, which is then passed into the encoder. The encoder consists of a stack of "blocks", each of which comprises two subcomponents: a self-attention layer followed by a small feed-forward network. Layer normalization (Ba et al., 2016) is applied to the input of each subcomponent. We use a simplified version of layer normalization where the activations are only rescaled and no additive bias is applied. After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent's input to its output. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack. The decoder is similar in structure to the encoder except that it includes a standard attention[^2]mechanism after each self-attention layer that attends to the output of the encoder. The self-attention mechanism in the decoder also uses a form of autoregressive or causal selfattention, which only allows the model to attend to past outputs. The output of the final decoder block is fed into a dense layer with a softmax output, whose weights are shared with the input embedding matrix. All attention mechanisms in the Transformer are split up into independent "heads" whose outputs are concatenated before being further processed.

Since self-attention is order-independent (i.e. it is an operation on sets), it is common to provide an explicit position signal to the Transformer. While the original Transformer used a sinusoidal position signal or learned position embeddings, it has recently become more common to use relative position embeddings (Shaw et al., 2018; Huang et al., 2018a). Instead of using a fixed embedding for each position, relative position embeddings produce a different learned embedding according to the offset between the "key" and "query" being compared in the self-attention mechanism. We use a simplified form of position embeddings where each "embedding" is simply a scalar that is added to the corresponding logit used for computing the attention weights. For efficiency, we also share the position embedding parameters across all layers in our model, though within a given layer each attention head uses a different learned position embedding. Typically, a fixed number of embeddings are learned, each corresponding to a range of possible key-query offsets. In this work, we use 32 embeddings for all of our models with ranges that increase in size logarithmically up to an offset of 128 beyond which we assign all relative positions to the same embedding. Note that a given layer is insensitive to relative position beyond 128 tokens, but subsequent layers can build a sensitivity to larger offsets by combining local information from previous layers. To summarize, our model is roughly equivalent to the original Transformer proposed by Vaswani et al. (2017) with the exception of removing the Layer Norm bias, placing the layer normalization outside the residual path, and using a different position embedding scheme. Since these architectural changes are orthogonal to the experimental factors we consider in our empirical survey of transfer learning, we leave the ablation of their impact for future work.

As part of our study, we experiment with the scalability of these models, i.e. how their performance changes as they are made to have more parameters or layers. Training large models can be non-trivial since they might not fit on a single machine and require a great deal of computation. As a result, we use a combination of model and data parallelism and train models on "slices" of Cloud TPU Pods. ${ }^{5}$ TPU pods are are multi-rack ML supercomputers that contain 1,024 TPU v3 chips connected via a high-speed 2D mesh interconnect with supporting CPU host machines. We leverage the Mesh TensorFlow library (Shazeer et al., 2018) for ease of implementation of both model parallelism and data parallelism (Krizhevsky, 2014).

### 2.2 The Colossal Clean Crawled Corpus

Much of the previous work on transfer learning for NLP makes use of large unlabeled data sets for unsupervised learning. In this paper, we are interested in measuring the effect of the quality, characteristics, and size of this unlabeled data. To generate data sets that satisfy our needs, we leverage Common Crawl as a source of text scraped from the web. Common[^3]

Crawl has previously been used as a source of text data for NLP, for example to train an n-gram language model (Buck et al., 2014), as training data for commonsense reasoning (Trinh and Le, 2018), for mining parallel texts for machine translation (Smith et al., 2013), as a pre-training data set (Grave et al., 2018; Zellers et al., 2019; Liu et al., 2019c), and even simply as a giant text corpus for testing optimizers (Anil et al., 2019).

Common Crawl is a publicly-available web archive that provides "web extracted text" by removing markup and other non-text content from the scraped HTML files. This process produces around 20TB of scraped text data each month. Unfortunately, the majority of the resulting text is not natural language. Instead, it largely comprises gibberish or boiler-plate text like menus, error messages, or duplicate text. Furthermore, a good deal of the scraped text contains content that is unlikely to be helpful for any of the tasks we consider (offensive language, placeholder text, source code, etc.). To address these issues, we used the following heuristics for cleaning up Common Crawl's web extracted text:

- We only retained lines that ended in a terminal punctuation mark (i.e. a period, exclamation mark, question mark, or end quotation mark).
- We discarded any page with fewer than 3 sentences and only retained lines that contained at least 5 words.
- We removed any page that contained any word on the "List of Dirty, Naughty, Obscene or Otherwise Bad Words". 6
- Many of the scraped pages contained warnings stating that Javascript should be enabled so we removed any line with the word Javascript.
- Some pages had placeholder "lorem ipsum" text; we removed any page where the phrase "lorem ipsum" appeared.
- Some pages inadvertently contained code. Since the curly bracket " $\{$ " appears in many programming languages (such as Javascript, widely used on the web) but not in natural text, we removed any pages that contained a curly bracket.
- Since some of the scraped pages were sourced from Wikipedia and had citation markers (e.g. [1], [citation needed], etc.), we removed any such markers.
- Many pages had boilerplate policy notices, so we removed any lines containing the strings "terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", or "use cookies".
- To deduplicate the data set, we discarded all but one of any three-sentence span occurring more than once in the data set.

Additionally, since most of our downstream tasks are focused on English-language text, we used langdetect ${ }^{7}$ to filter out any pages that were not classified as English with a probability of at least 0.99 . Our heuristics are inspired by past work on using Common[^4]

Crawl as a source of data for NLP: For example, Grave et al. (2018) also filter text using an automatic language detector and discard short lines and Smith et al. (2013); Grave et al. (2018) both perform line-level deduplication. However, we opted to create a new data set because prior data sets use a more limited set of filtering heuristics, are not publicly available, and/or are different in scope (e.g. are limited to News data (Zellers et al., 2019; Liu et al., 2019c), comprise only Creative Commons content (Habernal et al., 2016), or are focused on parallel training data for machine translation (Smith et al., 2013)).

To assemble our base data set, we downloaded the web extracted text from April 2019 and applied the aforementioned filtering. This produces a collection of text that is not only orders of magnitude larger than most data sets used for pre-training (about $750 \mathrm{~GB}$ ) but also comprises reasonably clean and natural English text. We dub this data set the "Colossal Clean Crawled Corpus" (or C4 for short) and release it as part of TensorFlow Datasets. ${ }^{8}$ We consider the impact of using various alternative versions of this data set in Section 3.4.

### 2.3 Downstream Tasks

Our goal in this paper is to measure general language learning abilities. As such, we study downstream performance on a diverse set of benchmarks, including machine translation, question answering, abstractive summarization, and text classification. Specifically, we measure performance on the GLUE and SuperGLUE text classification meta-benchmarks; CNN/Daily Mail abstractive summarization; SQuAD question answering; and WMT English to German, French, and Romanian translation. All data was sourced from TensorFlow Datasets. ${ }^{9}$

GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019b) each comprise a collection of text classification tasks meant to test general language understanding abilities:
- Sentence acceptability judgment (CoLA (Warstadt et al., 2018))
- Sentiment analysis (SST-2 (Socher et al., 2013))
- Paraphrasing/sentence similarity (MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Iyer et al., 2017))
- Natural language inference (MNLI (Williams et al., 2017), QNLI (Rajpurkar et al., 2016), RTE (Dagan et al., 2005), CB (De Marneff et al., 2019))
- Coreference resolution (WNLI and WSC (Levesque et al., 2012))
- Sentence completion (COPA (Roemmele et al., 2011))
- Word sense disambiguation (WIC (Pilehvar and Camacho-Collados, 2018))
- Question answering (MultiRC (Khashabi et al., 2018), ReCoRD (Zhang et al., 2018), BoolQ (Clark et al., 2019))[^5]

We use the data sets as distributed by the GLUE and SuperGLUE benchmarks. For simplicity, when fine-tuning we treat all of the tasks in the GLUE benchmark (and similarly for SuperGLUE) as a single task by concatenating all of the constituent data sets. As suggested by Kocijan et al. (2019) we also include the Definite Pronoun Resolution (DPR) data set (Rahman and Ng, 2012) in the combined SuperGLUE task.

The CNN/Daily Mail (Hermann et al., 2015) data set was introduced as a questionanswering task but was adapted for text summarization by Nallapati et al. (2016); we use the non-anonymized version from See et al. (2017) as an abstractive summarization task. SQuAD (Rajpurkar et al., 2016) is a common question-answering benchmark. In our experiments, the model is fed the question and its context and asked to generate the answer token-by-token. For WMT English to German, we use the same training data as (Vaswani et al., 2017) (i.e. News Commentary v13, Common Crawl, Europarl v7) and newstest2013 as a validation set (Bojar et al., 2014). For English to French, we use the standard training data from 2015 and newstest2014 as a validation set (Bojar et al., 2015). For English to Romanian, which is a standard lower-resource machine translation benchmark, we use the train and validation sets from WMT 2016 (Bojar et al., 2016). Note that we only pre-train on English data, so in order to learn to translate a given model will need to learn to generate text in a new language.

### 2.4 Input and Output Format

In order to train a single model on the diverse set of tasks described above, we cast all of the tasks we consider into a "text-to-text" format - that is, a task where the model is fed some text for context or conditioning and is then asked to produce some output text. This framework provides a consistent training objective both for pre-training and fine-tuning. Specifically, the model is trained with a maximum likelihood objective (using "teacher forcing" (Williams and Zipser, 1989)) regardless of the task. To specify which task the model should perform, we add a task-specific (text) prefix to the original input sequence before feeding it to the model.

As an example, to ask the model to translate the sentence "That is good." from English to German, the model would be fed the sequence "translate English to German: That is good." and would be trained to output "Das ist gut." For text classification tasks, the model simply predicts a single word corresponding to the target label. For example, on the MNLI benchmark (Williams et al., 2017) the goal is to predict whether a premise implies ("entailment"), contradicts ("contradiction"), or neither ("neutral") a hypothesis. With our preprocessing, the input sequence becomes "mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity." with the corresponding target word "entailment". Note that an issue arises if our model outputs text on a text classification task that does not correspond to any of the possible labels (for example if the model outputs "hamburger" when the only possible labels for a task were "entailment", "neutral", or "contradiction"). In this case, we always count the model's output as wrong, though we never observed this behavior in any of our trained models. Note that the choice of text prefix used for a given task is essentially a hyperparameter; we found that changing the exact wording of the prefix had limited impact and so did not perform extensive experiments into different prefix choices. A diagram of our text-to-text framework with a few input/output
examples is shown in Figure 1. We provide full examples of preprocessed inputs for every task we studied in Appendix D.

Our text-to-text framework follows previous work that casts multiple NLP tasks into a common format: McCann et al. (2018) propose the "Natural Language Decathlon", a benchmark that uses a consistent question-answering format for a suite of ten NLP tasks. The Natural Language Decathlon also stipulates that all models must be multi-task, i.e. are able to simultaneously tackle all of the tasks at once. We instead allow for separately fine-tuning the model on each individual task and use short task prefixes instead of an explicit question-answer format. Radford et al. (2019) evaluate the zero-shot learning capabilities of language models by feeding some input to the model as a prefix and then autoregressively sampling an output. For example, automatic summarization is done by feeding in a document followed by the text "TL;DR:" (short for "too long, didn't read", a common abbreviation) and then the summary is predicted via autoregressive decoding. We mainly consider models that explicitly process an input with an encoder before generating an output with a separate decoder and we focus on transfer learning rather than zero-shot learning. Finally, Keskar et al. (2019b) unify many NLP tasks as "span extraction", where text corresponding to possible output choices are appended to the input and the model is trained to extract the input span corresponding to the correct choice. In contrast, our framework also allows for generative tasks like machine translation and abstractive summarization where it is not possible to enumerate all possible output choices.

We were able to straightforwardly cast all of the tasks we considered into a text-to-text format with the exception of STS-B, which is a regression task where the goal is to predict a similarity score between 1 and 5 . We found that most of these scores were annotated in increments of 0.2 , so we simply rounded any score to the nearest increment of 0.2 and converted the result to a literal string representation of the number (e.g. the floating-point value 2.57 would be mapped to the string " 2.6 "). At test time, if the model outputs a string corresponding to a number between 1 and 5 , we convert it to a floating-point value; otherwise, we treat the model's prediction as incorrect. This effectively recasts the STS-B regression problem as a 21 -class classification problem.

Separately, we also convert the Winograd tasks (WNLI from GLUE, WSC from SuperGLUE, and the DPR data set we add to SuperGLUE) into a simpler format that is more amenable to the text-to-text framework. Examples from the Winograd tasks consist of a text passage containing an ambiguous pronoun that could refer to more than one of the noun phrases in the passage. For example, the passage might be "The city councilmen refused the demonstrators a permit because they feared violence.", which contains the ambiguous pronoun "they" that could refer to "city councilmen" or "demonstrators". We cast the WNLI, WSC, and DPR tasks as text-to-text problems by highlighting the ambiguous pronoun in the text passage and asking the model to predict the noun that it refers to. The example mentioned above would be transformed to the input "The city councilmen refused the demonstrators a permit because *they* feared violence." and the model would be trained to predict the target text "The city councilmen".

For WSC, examples contain the passage, the ambiguous pronoun, a candidate noun, and a True/False label reflecting whether the candidate matches the pronoun (ignoring any articles). We only train on examples with a "True" label since we do not know the correct noun targets for examples with a "False" label. For evaluation, we assign a "True" label if
the words in the model's output are a subset of the words in the candidate noun phrase (or vice versa) and assign a "False" label otherwise. This removes roughly half of the WSC training set, but the DPR data set adds about 1,000 pronoun resolution examples. Examples from DPR are annotated with the correct referent noun, making it easy to use this data set in the format listed above.

The WNLI training and validation sets have a significant overlap with the WSC training set. To avoid leaking validation examples into our training data (a particular issue in the multi-task experiments of Section 3.5.2), we therefore never train on WNLI and never report results on the WNLI validation set. Omitting results on the WNLI validation set is standard practice (Devlin et al., 2018) due to the fact that it is "adversarial" with respect to the training set, i.e. validation examples are all slightly-perturbed versions of training examples with the opposite label. As such, we do not include WNLI in the average GLUE score whenever we report on the validation set (all sections except Section 3.7 where results are presented on the test sets). Converting examples from WNLI to the "referent noun prediction" variant described above is a little more involved; we describe this process in Appendix B.

## 3. Experiments

Recent advances in transfer learning for NLP have come from a wide variety of developments, such as new pre-training objectives, model architectures, unlabeled data sets, and more. In this section, we carry out an empirical survey of these techniques in hopes of teasing apart their contribution and significance. We then combine the insights gained to attain state-of-the-art in many of the tasks we consider. Since transfer learning for NLP is a rapidly growing area of research, it is not feasible for us to cover every possible technique or idea in our empirical study. For a broader literature review, we recommend a recent survey by Ruder et al. (2019).

We systematically study these contributions by taking a reasonable baseline (described in Section 3.1) and altering one aspect of the setup at a time. For example, in Section 3.3 we measure the performance of different unsupervised objectives while keeping the rest of our experimental pipeline fixed. This "coordinate ascent" approach might miss second-order effects (for example, some particular unsupervised objective may work best on a model larger than our baseline setting), but performing a combinatorial exploration of all of the factors in our study would be prohibitively expensive. In future work, we expect it could be fruitful to more thoroughly consider combinations of the approaches we study.

Our goal is to compare a variety of different approaches on a diverse set of tasks while keeping as many factors fixed as possible. In order to satisfy this aim, in some cases we do not exactly replicate existing approaches. For example, "encoder-only" models like BERT (Devlin et al., 2018) are designed to produce a single prediction per input token or a single prediction for an entire input sequence. This makes them applicable for classification or span prediction tasks but not for generative tasks like translation or abstractive summarization. As such, none of the model architectures we consider are identical to BERT or consist of an encoder-only structure. Instead, we test approaches that are similar in spirit-for example, we consider an analogous objective to BERT's "masked language modeling" objective in

Section 3.3 and we consider a model architecture that behaves similarly to BERT on text classification tasks in Section 3.2.

After outlining our baseline experimental setup in the following subsection, we undertake an empirical comparison of model architectures (Section 3.2), unsupervised objectives (Section 3.3), pre-training data sets (Section 3.4), transfer approaches (Section 3.5), and scaling (Section 3.6). At the culmination of this section, we combine insights from our study with scale to obtain state-of-the-art results in many tasks we consider (Section 3.7).

### 3.1 Baseline

Our goal for our baseline is to reflect typical, modern practice. We pre-train a standard Transformer (described in Section 2.1) using a simple denoising objective and then separately fine-tune on each of our downstream tasks. We describe the details of this experimental setup in the following subsections.

### 3.1.1 ModeL

For our model, we use a standard encoder-decoder Transformer as proposed by Vaswani et al. (2017). While many modern approaches to transfer learning for NLP use a Transformer architecture consisting of only a single "stack" (e.g. for language modeling (Radford et al., 2018; Dong et al., 2019) or classification and span prediction (Devlin et al., 2018; Yang et al., 2019)), we found that using a standard encoder-decoder structure achieved good results on both generative and classification tasks. We explore the performance of different model architectures in Section 3.2.

Our baseline model is designed so that the encoder and decoder are each similar in size and configuration to a "BERT ${ }_{\text {BASE" }}$ (Devlin et al., 2018) stack. Specifically, both the encoder and decoder consist of 12 blocks (each block comprising self-attention, optional encoder-decoder attention, and a feed-forward network). The feed-forward networks in each block consist of a dense layer with an output dimensionality of $d_{\mathrm{ff}}=3072$ followed by a ReLU nonlinearity and another dense layer. The "key" and "value" matrices of all attention mechanisms have an inner dimensionality of $d_{\mathrm{kv}}=64$ and all attention mechanisms have 12 heads. All other sub-layers and embeddings have a dimensionality of $d_{\text {model }}=768$. In total, this results in a model with about 220 million parameters. This is roughly twice the number of parameters of $\mathrm{BERT}_{\text {BASE }}$ since our baseline model contains two layer stacks instead of one. For regularization, we use a dropout probability of 0.1 everywhere dropout is applied in the model.

### 3.1.2 Training

As described in Section 2.4, all tasks are formulated as text-to-text tasks. This allows us to always train using standard maximum likelihood, i.e. using teacher forcing (Williams and Zipser 1989) and a cross-entropy loss. For optimization, we use AdaFactor (Shazeer and Stern, 2018). At test time, we use greedy decoding (i.e. choosing the highest-probability logit at every timestep).

We pre-train each model for $2^{19}=524,288$ steps on $\mathrm{C} 4$ before fine-tuning. We use a maximum sequence length of 512 and a batch size of 128 sequences. Whenever possible,
we "pack" multiple sequences into each entry of the batch ${ }^{10}$ so that our batches contain roughly $2^{16}=65,536$ tokens. In total, this batch size and number of steps corresponds to pre-training on $2^{35} \approx 34 \mathrm{~B}$ tokens. This is considerably less than BERT (Devlin et al., 2018), which used roughly 137B tokens, or RoBERTa (Liu et al., 2019c), which used roughly 2.2T tokens. Using only $2^{35}$ tokens results in a reasonable computational budget while still providing a sufficient amount of pre-training for acceptable performance. We consider the effect of pre-training for more steps in Sections 3.6 and 3.7. Note that $2^{35}$ tokens only covers a fraction of the entire $\mathrm{C} 4$ data set, so we never repeat any data during pre-training.

During pre-training, we use an "inverse square root" learning rate schedule: $1 / \sqrt{\max (n, k)}$ where $n$ is the current training iteration and $k$ is the number of warm-up steps (set to $10^{4}$ in all of our experiments). This sets a constant learning rate of 0.01 for the first $10^{4} \mathrm{steps}$, then exponentially decays the learning rate until pre-training is over. We also experimented with using a triangular learning rate (Howard and Ruder, 2018), which produced slightly better results but requires knowing the total number of training steps ahead of time. Since we will be varying the number of training steps in some of our experiments, we opt for the more generic inverse square root schedule.

Our models are fine-tuned for $2^{18}=262,144$ steps on all tasks. This value was chosen as a trade-off between the high-resource tasks (i.e. those with large data sets), which benefit from additional fine-tuning, and low-resource tasks (smaller data sets), which overfit quickly. During fine-tuning, we continue using batches with 128 length-512 sequences (i.e. $2^{16}$ tokens per batch). We use a constant learning rate of 0.001 when fine-tuning. We save a checkpoint every 5,000 steps and report results on the model checkpoint corresponding to the highest validation performance. For models fine-tuned on multiple tasks, we choose the best checkpoint for each task independently. For all of the experiments except those in Section 3.7, we report results in the validation set to avoid performing model selection on the test set.

### 3.1.3 Vocabulary

We use SentencePiece (Kudo and Richardson, 2018) to encode text as WordPiece tokens (Sennrich et al., 2015; Kudo, 2018). For all experiments, we use a vocabulary of 32,000 wordpieces. Since we ultimately fine-tune our model on English to German, French, and Romanian translation, we also require that our vocabulary covers these non-English languages. To address this, we classified pages from the Common Crawl scrape used in C4 as German, French, and Romanian. Then, we trained our SentencePiece model on a mixture of 10 parts of English C4 data with 1 part each of data classified as German, French or Romanian. This vocabulary was shared across both the input and output of our model. Note that our vocabulary makes it so that our model can only process a predetermined, fixed set of languages.

### 3.1.4 UNSUPERVISED ObJECTIVE

Leveraging unlabeled data to pre-train our model necessitates an objective that does not require labels but (loosely speaking) teaches the model generalizable knowledge that will be[^6]

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-13.jpg?height=337&width=914&top_left_y=314&top_left_x=603)

Figure 2: Schematic of the objective we use in our baseline model. In this example, we process the sentence "Thank you for inviting me to your party last week." The words "for", "inviting" and "last" (marked with an $\times$ ) are randomly chosen for corruption. Each consecutive span of corrupted tokens is replaced by a sentinel token (shown as $\langle\mathrm{X}\rangle$ and $\langle\mathrm{Y}\rangle$ ) that is unique over the example. Since "for" and "inviting" occur consecutively, they are replaced by a single sentinel <X>. The output sequence then consists of the dropped-out spans, delimited by the sentinel tokens used to replace them in the input plus a final sentinel token <Z>.

useful in downstream tasks. Preliminary work that applied the transfer learning paradigm of pre-training and fine-tuning all of the model's parameters to NLP problems used a causal language modeling objective for pre-training (Dai and Le, 2015; Peters et al., 2018; Radford et al., 2018; Howard and Ruder, 2018). However, it has recently been shown that "denoising" objectives (Devlin et al., 2018; Taylor, 1953) (also called "masked language modeling") produce better performance and as a result they have quickly become standard. In a denoising objective, the model is trained to predict missing or otherwise corrupted tokens in the input. Inspired by BERT's "masked language modeling" objective and the "word dropout" regularization technique (Bowman et al., 2015), we design an objective that randomly samples and then drops out $15 \%$ of tokens in the input sequence. All consecutive spans of dropped-out tokens are replaced by a single sentinel token. Each sentinel token is assigned a token ID that is unique to the sequence. The sentinel IDs are special tokens which are added to our vocabulary and do not correspond to any wordpiece. The target then corresponds to all of the dropped-out spans of tokens, delimited by the same sentinel tokens used in the input sequence plus a final sentinel token to mark the end of the target sequence. Our choices to mask consecutive spans of tokens and only predict dropped-out tokens were made to reduce the computational cost of pre-training. We perform thorough investigation into pre-training objectives in Section 3.3. An example of the transformation resulting from applying this objective is shown in Figure 2. We empirically compare this objective to many other variants in Section 3.3 .

### 3.1.5 Baseline Performance

In this section, we present results using the baseline experimental procedure described above to get a sense of what kind of performance to expect on our suite of downstream tasks. Ideally, we would repeat every experiment in our study multiple times to get a confidence interval on our results. Unfortunately, this would be prohibitively expensive due to the large

|  | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| В Baseline average | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| Baseline standard deviation | 0.235 | 0.065 | 0.343 | 0.416 | 0.112 | 0.090 | 0.108 |
| No pre-training | 66.22 | 17.60 | 50.31 | 53.04 | 25.86 | $\mathbf{3 9 . 7 7}$ | 24.04 |

Table 1: Average and standard deviation of scores achieved by our baseline model and training procedure. For comparison, we also report performance when training on each task from scratch (i.e. without any pre-training) for the same number of steps used to fine-tune the baseline model. All scores in this table (and every table in our paper except Table 14) are reported on the validation sets of each data set.

number of experiments we run. As a cheaper alternative, we train our baseline model 10 times from scratch (i.e. with different random initializations and data set shuffling) and assume that the variance over these runs of the base model also applies to each experimental variant. We don't expect most of the changes we make to have a dramatic effect on the inter-run variance, so this should provide a reasonable indication of the significance of different changes. Separately, we also measure the performance of training our model for $2^{18}$ steps (the same number we use for fine-tuning) on all downstream tasks without pre-training. This gives us an idea of how much pre-training benefits our model in the baseline setting.

When reporting results in the main text, we only report a subset of the scores across all the benchmarks to conserve space and ease interpretation. For GLUE and SuperGLUE, we report the average score across all subtasks (as stipulated by the official benchmarks) under the headings "GLUE" and "SGLUE". For all translation tasks, we report the BLEU score (Papineni et al., 2002) as provided by SacreBLEU v1.3.0 (Post, 2018) with "exp" smoothing and "intl" tokenization. We refer to scores for WMT English to German, English to French, and English to Romanian as EnDe, EnFr, and EnRo, respectively. For CNN/Daily Mail, we find the performance of models on the ROUGE-1-F, ROUGE-2-F, and ROUGE-L-F metrics (Lin, 2004) to be highly correlated so we report the ROUGE-2-F score alone under the heading "CNNDM". Similarly, for SQuAD we find the performance of the "exact match" and "F1" scores to be highly correlated so we report the "exact match" score alone. We provide every score achieved on every task for all experiments in Table 16, Appendix E.

Our results tables are all formatted so that each row corresponds to a particular experimental configuration with columns giving the scores for each benchmark. We will include the mean performance of the baseline configuration in most tables. Wherever a baseline configuration appears, we will mark it with a $\star$ (as in the first row of Table 1). We also will boldface any score that is within two standard deviations of the maximum (best) in a given experiment.

Our baseline results are shown in Table 1. Overall, our results are comparable to existing models of similar size. For example, BERT $_{\text {BASE }}$ achieved an exact match score of 80.8 on SQuAD and an accuracy of 84.4 on MNLI-matched, whereas we achieve 80.88 and 84.24, respectively (see Table 16). Note that we cannot directly compare our baseline to $\mathrm{BERT}_{\text {BASE }}$ because ours is an encoder-decoder model and was pre-trained for roughly $1 / 4$ as many steps. Unsurprisingly, we find that pre-training provides significant gains across almost all benchmarks. The only exception is WMT English to French, which is a large
enough data set that gains from pre-training tend to be marginal. We include this task in our experiments to test the behavior of transfer learning in the high-resource regime. Since we perform early stopping by selecting the best-performing checkpoint, the large disparity between our baseline and "no pre-training" emphasize how much pre-training improves performance on tasks with limited data. While we do not explicitly measure improvements in data efficiency in this paper, we emphasize that this is one of the primary benefits of the transfer learning paradigm.

As for inter-run variance, we find that for most tasks the standard deviation across runs is smaller than $1 \%$ of the task's baseline score. Exceptions to this rule include CoLA, CB, and COPA, which are all low-resource tasks from the GLUE and SuperGLUE benchmarks. For example, on CB our baseline model had an average F1 score of 91.22 with a standard deviation of 3.237 (see Table 16), which may be partly due to the fact that CB's validation set contains only 56 examples. Note that the GLUE and SuperGLUE scores are computed as the average of scores across the tasks comprising each benchmark. As a result, we caution that the high inter-run variance of CoLA, CB, and COPA can make it harder to compare models using the GLUE and SuperGLUE scores alone.

### 3.2 Architectures

While the Transformer was originally introduced with an encoder-decoder architecture, much modern work on transfer learning for NLP uses alternative architectures. In this section, we review and compare these architectural variants.

### 3.2.1 Model Structures

A major distinguishing factor for different architectures is the "mask" used by different attention mechanisms in the model. Recall that the self-attention operation in a Transformer takes a sequence as input and outputs a new sequence of the same length. Each entry of the output sequence is produced by computing a weighted average of entries of the input sequence. Specifically, let $y_{i}$ refer to the $i$ th element of the output sequence and $x_{j}$ refer to the $j$ th entry of the input sequence. $y_{i}$ is computed as $\sum_{j} w_{i, j} x_{j}$, where $w_{i, j}$ is the scalar weight produced by the self-attention mechanism as a function of $x_{i}$ and $x_{j}$. The attention mask is then used to zero out certain weights in order to constrain which entries of the input can be attended to at a given output timestep. Diagrams of the masks we will consider are shown in Figure 3. For example, the causal mask (Figure 3, middle) sets any $w_{i, j}$ to zero if $j>i$.

The first model structure we consider is an an encoder-decoder Transformer, which consists of two layer stacks: The encoder, which is fed an input sequence, and the decoder, which produces a new output sequence. A schematic of this architectural variant is shown in the left panel of Figure 4.

The encoder uses a "fully-visible" attention mask. Fully-visible masking allows a selfattention mechanism to attend to any entry of the input when producing each entry of its output. We visualize this masking pattern in Figure 3, left. This form of masking is appropriate when attending over a "prefix", i.e. some context provided to the model that is later used when making predictions. BERT (Devlin et al., 2018) also uses a fully-visible masking pattern and appends a special "classification" token to the input. BERT's output
![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-16.jpg?height=464&width=1204&top_left_y=304&top_left_x=466)

Figure 3: Matrices representing different attention mask patterns. The input and output of the self-attention mechanism are denoted $x$ and $y$ respectively. A dark cell at row $i$ and column $j$ indicates that the self-attention mechanism is allowed to attend to input element $j$ at output timestep $i$. A light cell indicates that the self-attention mechanism is not allowed to attend to the corresponding $i$ and $j$ combination. Left: A fully-visible mask allows the self-attention mechanism to attend to the full input at every output timestep. Middle: A causal mask prevents the $i$ th output element from depending on any input elements from "the future". Right: Causal masking with a prefix allows the self-attention mechanism to use fully-visible masking on a portion of the input sequence.

at the timestep corresponding to the classification token is then used to make a prediction for classifying the input sequence.

The self-attention operations in the Transformer's decoder use a "causal" masking pattern. When producing the $i$ th entry of the output sequence, causal masking prevents the model from attending to the $j$ th entry of the input sequence for $j>i$. This is used during training so that the model can't "see into the future" as it produces its output. An attention matrix for this masking pattern is shown in Figure 3, middle.

The decoder in an encoder-decoder Transformer is used to autoregressively produce an output sequence. That is, at each output timestep, a token is sampled from the model's predicted distribution and the sample is fed back into the model to produce a prediction for the next output timestep, and so on. As such, a Transformer decoder (without an encoder) can be used as a language model (LM), i.e. a model trained solely for next-step prediction (Liu et al., 2018; Radford et al., 2018; Al-Rfou et al., 2019). This constitutes the second model structure we consider. A schematic of this architecture is shown in Figure 4, middle. In fact, early work on transfer learning for NLP used this architecture with a language modeling objective as a pre-training method (Radford et al., 2018).

Language models are typically used for compression or sequence generation (Graves, 2013). However, they can also be used in the text-to-text framework simply by concatenating the inputs and targets. As an example, consider the case of English to German translation: If we have a training datapoint with input sentence "That is good." and target "Das ist gut.", we would simply train the model on next-step prediction over the concatenated input sequence "translate English to German: That is good. target: Das ist gut." If we wanted to

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-17.jpg?height=523&width=347&top_left_y=324&top_left_x=455)

Language model

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-17.jpg?height=420&width=350&top_left_y=413&top_left_x=866)

Prefix LM

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-17.jpg?height=410&width=339&top_left_y=421&top_left_x=1313)

Figure 4: Schematics of the Transformer architecture variants we consider. In this diagram, blocks represent elements of a sequence and lines represent attention visibility. Different colored groups of blocks indicate different Transformer layer stacks. Dark grey lines correspond to fully-visible masking and light grey lines correspond to causal masking. We use "." to denote a special end-of-sequence token that represents the end of a prediction. The input and output sequences are represented as $x$ and $y$ respectively. Left: A standard encoder-decoder architecture uses fullyvisible masking in the encoder and the encoder-decoder attention, with causal masking in the decoder. Middle: A language model consists of a single Transformer layer stack and is fed the concatenation of the input and target, using a causal mask throughout. Right: Adding a prefix to a language model corresponds to allowing fully-visible masking over the input.

obtain the model's prediction for this example, the model would be fed the prefix "translate English to German: That is good. target:" and would be asked to generate the remainder of the sequence autoregressively. In this way, the model can predict an output sequence given an input, which satisfies the needs of text-to-text tasks. This approach was recently used to show that language models can learn to perform some text-to-text tasks without supervision (Radford et al., 2019).

A fundamental and frequently cited drawback of using a language model in the textto-text setting is that causal masking forces the model's representation of the $i$ th entry of the input sequence to only depend on the entries up until $i$. To see why this is potentially disadvantageous, consider the text-to-text framework where the model is provided with a prefix/context before being asked to make predictions (e.g., the prefix is an English sentence and the model is asked to predict the German translation). With fully causal masking, the model's representation of a prefix state can only depend on prior entries of the prefix. So, when predicting an entry of the output, the model will attend to a representation of the prefix that is unnecessarily limited. Similar arguments have been made against using a unidirectional recurrent neural network encoder in sequence-to-sequence models (Bahdanau et al., 2015).

This issue can be avoided in a Transformer-based language model simply by changing the masking pattern. Instead of using a causal mask, we use fully-visible masking during the prefix portion of the sequence. This masking pattern and a schematic of the resulting "prefix LM" (the third model structure we consider) are illustrated in the rightmost panels of Figures 3 and 4, respectively. In the English to German translation example mentioned above, fully-visible masking would be applied to the prefix "translate English to German: That is good. target:" and causal masking would be used during training for predicting the target "Das ist gut." Using a prefix LM in the text-to-text framework was originally proposed by Liu et al. (2018). More recently, Dong et al. (2019) showed that this architecture is effective on a wide variety of text-to-text tasks. This architecture is similar to an encoder-decoder model with parameters shared across the encoder and decoder and with the encoder-decoder attention replaced with full attention across the input and target sequence.

We note that when following our text-to-text framework, the prefix LM architecture closely resembles BERT (Devlin et al., 2018) for classification tasks. To see why, consider an example from the MNLI benchmark where the premise is "I hate pigeons.", the hypothesis is "My feelings towards pigeons are filled with animosity." and the correct label is "entailment". To feed this example into a language model, we would transform it into the sequence "mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity. target: entailment". In this case, the fully-visible prefix would correspond to the entire input sequence up to the word "target:", which can be seen as being analogous to the "classification" token used in BERT. So, our model would have full visibility over the entire input, and then would be tasked with making a classification by outputting the word "entailment". It is easy for the model to learn to output one of the valid class labels given the task prefix ("mnli" in this case). As such, the main difference between a prefix LM and the BERT architecture is that the classifier is simply integrated into the output layer of the Transformer decoder in the prefix LM.

### 3.2.2 Comparing Different Model Structures

In the interest of experimentally comparing these architectural variants, we would like each model we consider to be equivalent in some meaningful way. We might say that two models are equivalent if they either have the same number of parameters or they require roughly the same amount of computation to process a given (input-sequence, target-sequence) pair. Unfortunately, it is not possible to compare an encoder-decoder model to a language model architecture (comprising a single Transformer stack) according to both of these criteria at the same time. To see why, first note an encoder-decoder model with $L$ layers in the encoder and $L$ layers in the decoder has approximately the same number of parameters as a language model with $2 L$ layers. However, the same $L+L$ encoder-decoder model will have approximately the same computational cost as a language model with only $L$ layers. This is a consequence of the fact that the $L$ layers in the language model must be applied to both the input and output sequence, while the encoder is only applied to the input sequence and the decoder is only applied to the output sequence. Note that these equivalences are approximate - there are some extra parameters in the decoder due to the encoder-decoder attention and there are also some computational costs in the attention layers that are quadratic in the sequence lengths. In practice, however, we observed nearly identical step
times for $L$-layer language models versus $L+L$-layer encoder-decoder models, suggesting a roughly equivalent computational cost. Further, for the model sizes we consider, the number of parameters in the encoder-decoder attention layers is about $10 \%$ of the total parameter count, so we make the simplifying assumption that an $L+L$-layer encoder-decoder model has the same number of parameters as an $2 L$-layer language model.

To provide a reasonable means of comparison, we consider multiple configurations for our encoder-decoder model. We will refer to the number of layers and parameters in a

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-19.jpg?height=46&width=1515&top_left_y=649&top_left_x=305)
of FLOPs required for an $L+L$-layer encoder-decoder model or $L$-layer decoder-only model to process a given input-target pair. In total, we will compare:

- An encoder-decoder model with $L$ layers in the encoder and $L$ layers in the decoder. This model has $2 P$ parameters and a computation cost of $M$ FLOPs.
- An equivalent model, but with parameters shared across the encoder and decoder, resulting in $P$ parameters and an $M$-FLOP computational cost.
- An encoder-decoder model with $L / 2$ layers each in the encoder and decoder, giving $P$ parameters and an $M / 2$-FLOP cost.
- A decoder-only language model with $L$ layers and $P$ parameters and a resulting computational cost of $M$ FLOPs.
- A decoder-only prefix LM with the same architecture (and thus the same number of parameters and computational cost), but with fully-visible self-attention over the input.


## 3.2 .3 ОвJECTIVES

As an unsupervised objective, we will consider both a basic language modeling objective as well as our baseline denoising objective described in Section 3.1.4. We include the language modeling objective due to its historic use as a pre-training objective (Dai and Le, 2015; Ramachandran et al., 2016; Howard and Ruder, 2018; Radford et al., 2018; Peters et al., 2018) as well as its natural fit for the language model architectures we consider. For models that ingest a prefix before making predictions (the encoder-decoder model and prefix LM), we sample a span of text from our unlabeled data set and choose a random point to split it into prefix and target portions. For the standard language model, we train the model to predict the entire span from beginning to end. Our unsupervised denoising objective is designed for text-to-text models; to adapt it for use with a language model we concatenate the inputs and targets as described in Section 3.2.1.

### 3.2.4 ReSULTS

The scores achieved by each of the architectures we compare are shown in Table 2. For all tasks, the encoder-decoder architecture with the denoising objective performed best. This variant has the highest parameter count $(2 P)$ but the same computational cost as the $P$-parameter decoder-only models. Surprisingly, we found that sharing parameters across the encoder and decoder performed nearly as well. In contrast, halving the number of layers in

| Architecture | Objective | Params | Cost | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ڤ Encoder-decoder | Denoising | $2 P$ | $M$ | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| Enc-dec, shared | Denoising | $P$ | $M$ | 82.81 | 18.78 | $\mathbf{8 0 . 6 3}$ | $\mathbf{7 0 . 7 3}$ | 26.72 | 39.03 | $\mathbf{2 7 . 4 6}$ |
| Enc-dec, 6 layers | Denoising | $P$ | $M / 2$ | 80.88 | 18.97 | 77.59 | 68.42 | 26.38 | 38.40 | 26.95 |
| Language model | Denoising | $P$ | $M$ | 74.70 | 17.93 | 61.14 | 55.02 | 25.09 | 35.28 | 25.86 |
| Prefix LM | Denoising | $P$ | $M$ | 81.82 | 18.61 | 78.94 | 68.11 | 26.43 | 37.98 | 27.39 |
| Encoder-decoder | LM | $2 P$ | $M$ | 79.56 | 18.59 | 76.02 | 64.29 | 26.27 | 39.17 | 26.86 |
| Enc-dec, shared | LM | $P$ | $M$ | 79.60 | 18.13 | 76.35 | 63.50 | 26.62 | 39.17 | 27.05 |
| Enc-dec, 6 layers | LM | $P$ | $M / 2$ | 78.67 | 18.26 | 75.32 | 64.06 | 26.13 | 38.42 | 26.89 |
| Language model | LM | $P$ | $M$ | 73.78 | 17.54 | 53.81 | 56.51 | 25.23 | 34.31 | 25.38 |
| Prefix LM | LM | $P$ | $M$ | 79.68 | 17.84 | 76.87 | 64.86 | 26.28 | 37.51 | 26.76 |

Table 2: Performance of the different architectural variants described in Section 3.2.2. We use $P$ to refer to the number of parameters in a 12-layer base Transformer layer stack and $M$ to refer to the FLOPs required to process a sequence using the encoderdecoder model. We evaluate each architectural variant using a denoising objective (described in Section 3.1.4) and an autoregressive objective (as is commonly used to train language models).

the encoder and decoder stacks significantly hurt performance. Concurrent work (Lan et al., 2019) also found that sharing parameters across Transformer blocks can be an effective means of lowering the total parameter count without sacrificing much performance. XLNet also bears some resemblance to the shared encoder-decoder approach with a denoising objective (Yang et al., 2019). We also note that the shared parameter encoder-decoder outperforms the decoder-only prefix LM, suggesting that the addition of an explicit encoder-decoder attention is beneficial. Finally, we confirm the widely-held conception that using a denoising objective always results in better downstream task performance compared to a language modeling objective. This observation has been previously made by Devlin et al. (2018), Voita et al. (2019), and Lample and Conneau (2019) among others. We undertake a more detailed exploration of unsupervised objectives in the following section.

### 3.3 Unsupervised Objectives

The choice of unsupervised objective is of central importance as it provides the mechanism through which the model gains general-purpose knowledge to apply to downstream tasks. This has led to the development of a wide variety of pre-training objectives (Dai and Le, 2015; Ramachandran et al., 2016; Radford et al., 2018; Devlin et al., 2018; Yang et al., 2019; Liu et al., 2019b; Wang et al., 2019a; Song et al., 2019; Dong et al., 2019; Joshi et al., 2019). In this section, we perform a procedural exploration of the space of unsupervised objectives. In many cases, we will not replicate an existing objective exactly-some will be modified to fit our text-to-text encoder-decoder framework and, in other cases, we will use objectives that combine concepts from multiple common approaches.

Overall, all of our objectives ingest a sequence of token IDs corresponding to a tokenized span of text from our unlabeled text data set. The token sequence is processed to produce a (corrupted) input sequence and a corresponding target. Then, the model is trained as usual

| Objective | Inputs | Targets |
| :--- | :--- | :--- |
| Prefix language modeling | Thank you for inviting | me to your party last week . |
| BERT-style Devlin et al. (2018) | Thank you <M <M> me to your party apple week . | (original text) |
| Deshuffling | party me for your to . last fun you inviting week Thank | (original text) |
| MASS-style Song et al. (2019) | Thank you <M> <M> me to your party <M> week. | (original text) |
| I.i.d. noise, replace spans | Thank you <X> me to your party <Y> week. | <X> for inviting <Y> last <Z> |
| I.i.d. noise, drop tokens | Thank you me to your party week . | for inviting last |
| Random spans | Thank you <X> to <Y> week | <X> for inviting me <Y> your party last <Z> |

Table 3: Examples of inputs and targets produced by some of the unsupervised objectives we consider applied to the input text "Thank you for inviting me to your party last week ." Note that all of our objectives process tokenized text. For this particular sentence, all words were mapped to a single token by our vocabulary. We write (original text) as a target to denote that the model is tasked with reconstructing the entire input text. <M> denotes a shared mask token and <X>, <Y>, and <Z> denote sentinel tokens that are assigned unique token IDs. The BERT-style objective (second row) includes a corruption where some tokens are replaced by a random token ID; we show this via the greyed-out word apple.

with maximum likelihood to predict the target sequence. We provide illustrative examples of many of the objectives we consider in Table 3.

### 3.3.1 Disparate High-Level Approaches

To begin with, we compare three techniques that are inspired by commonly-used objectives but differ significantly in their approach. First, we include a basic "prefix language modeling" objective as was used in Section 3.2.3. This technique splits a span of text into two components, one to use as inputs to the encoder and the other to use as a target sequence to be predicted by the decoder. Second, we consider an objective inspired by the "masked language modeling" (MLM) objective used in BERT (Devlin et al., 2018). MLM takes a span of text and corrupts $15 \%$ of the tokens. $90 \%$ of the corrupted tokens are replaced with a special mask token and $10 \%$ are replaced with a random token. Since BERT is an encoder-only model, its goal during pre-training is to reconstruct masked tokens at the output of the encoder. In the encoder-decoder case, we simply use the entire uncorrupted sequence as the target. Note that this differs from our baseline objective, which uses only the corrupted tokens as targets; we compare these two approaches in Section 3.3.2. Finally, we also consider a basic deshuffling objective as used e.g. in (Liu et al., 2019a) where it was applied to a denoising sequential autoencoder. This approach takes a sequence of tokens, shuffles it, and then uses the original deshuffled sequence as a target. We provide examples of the inputs and targets for these three methods in the first three rows of Table 3.

The performance of these three objectives is shown in Table 4. Overall, we find that the BERT-style objective performs best, though the prefix language modeling objective attains similar performance on the translation tasks. Indeed, the motivation for the BERT objective was to outperform language model-based pre-training. The deshuffling objective performs considerably worse than both prefix language modeling and the BERT-style objective.

| Objective | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Prefix language modeling | 80.69 | 18.94 | 77.99 | 65.27 | $\mathbf{2 6 . 8 6}$ | 39.73 | $\mathbf{2 7 . 4 9}$ |
| BERT-style (Devlin et al., 2018) | $\mathbf{8 2 . 9 6}$ | $\mathbf{1 9 . 1 7}$ | $\mathbf{8 0 . 6 5}$ | $\mathbf{6 9 . 8 5}$ | $\mathbf{2 6 . 7 8}$ | $\mathbf{4 0 . 0 3}$ | $\mathbf{2 7 . 4 1}$ |
| Deshuffling | 73.17 | 18.59 | 67.61 | 58.47 | 26.11 | 39.30 | 25.62 |

Table 4: Performance of the three disparate pre-training objectives described in Section 3.3.1.

| Objective | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT-style (Devlin et al., 2018) | 82.96 | 19.17 | $\mathbf{8 0 . 6 5}$ | 69.85 | 26.78 | $\mathbf{4 0 . 0 3}$ | 27.41 |
| MASS-style (Song et al., 2019) | 82.32 | 19.16 | 80.10 | 69.28 | 26.79 | $\mathbf{3 9 . 8 9}$ | 27.55 |
| ^ Replace corrupted spans | 83.28 | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | 39.82 | $\mathbf{2 7 . 6 5}$ |
| Drop corrupted tokens | $\mathbf{8 4 . 4 4}$ | $\mathbf{1 9 . 3 1}$ | $\mathbf{8 0 . 5 2}$ | 68.67 | $\mathbf{2 7 . 0 7}$ | 39.76 | $\mathbf{2 7 . 8 2}$ |

Table 5: Comparison of variants of the BERT-style pre-training objective. In the first two variants, the model is trained to reconstruct the original uncorrupted text segment. In the latter two, the model only predicts the sequence of corrupted tokens.

### 3.3.2 Simplifying the BERT ObJective

Based on the results in the prior section, we will now focus on exploring modifications to the BERT-style denoising objective. This objective was originally proposed as a pre-training technique for an encoder-only model trained for classification and span prediction. As such, it may be possible to modify it so that it performs better or is more efficient in our encoder-decoder text-to-text setup.

First, we consider a simple variant of the BERT-style objective where we don't include the random token swapping step. The resulting objective simply replaces $15 \%$ of the tokens in the input with a mask token and the model is trained to reconstruct the original uncorrupted sequence. A similar masking objective was used by Song et al. (2019) where it was referred to as "MASS", so we call this variant the "MASS-style" objective. Second, we were interested to see if it was possible to avoid predicting the entire uncorrupted text span since this requires self-attention over long sequences in the decoder. We consider two strategies to achieve this: First, instead of replacing each corrupted token with a mask token, we replace the entirety of each consecutive span of corrupted tokens with a unique mask token. Then, the target sequence becomes the concatenation of the "corrupted" spans, each prefixed by the mask token used to replace it in the input. This is the pre-training objective we use in our baseline, described in Section 3.1.4. Second, we also consider a variant where we simply drop the corrupted tokens from the input sequence completely and task the model with reconstructing the dropped tokens in order. Examples of these approaches are shown in the fifth and sixth rows of Table 3.

An empirical comparison of the original BERT-style objective to these three alternatives is shown in Table 5. We find that in our setting, all of these variants perform similarly. The only exception was that dropping corrupted tokens completely produced a small improvement in the GLUE score thanks to a significantly higher score on CoLA (60.04, compared to our

| Corruption rate | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $10 \%$ | $\mathbf{8 2 . 8 2}$ | 19.00 | $\mathbf{8 0 . 3 8}$ | 69.55 | $\mathbf{2 6 . 8 7}$ | 39.28 | $\mathbf{2 7 . 4 4}$ |
| $\star 15 \%$ | $\mathbf{8 3 . 2 8}$ | 19.24 | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| $25 \%$ | $\mathbf{8 3 . 0 0}$ | $\mathbf{1 9 . 5 4}$ | $\mathbf{8 0 . 9 6}$ | 70.48 | $\mathbf{2 7 . 0 4}$ | $\mathbf{3 9 . 8 3}$ | $\mathbf{2 7 . 4 7}$ |
| $50 \%$ | 81.27 | 19.32 | 79.80 | 70.33 | $\mathbf{2 7 . 0 1}$ | $\mathbf{3 9 . 9 0}$ | $\mathbf{2 7 . 4 9}$ |

Table 6: Performance of the i.i.d. corruption objective with different corruption rates.

baseline average of 53.84, see Table 16). This may be due to the fact that CoLA involves classifying whether a given sentence is grammatically and syntactically acceptable, and being able to determine when tokens are missing is closely related to detecting acceptability. However, dropping tokens completely performed worse than replacing them with sentinel tokens on SuperGLUE. The two variants that do not require predicting the full original sequence ("replace corrupted spans" and "drop corrupted spans") are both potentially attractive since they make the target sequences shorter and consequently make training faster. Going forward, we will explore variants where we replace corrupted spans with sentinel tokens and only predict the corrupted tokens (as in our baseline objective).

### 3.3.3 Varying the Corruption Rate

So far, we have been corrupting $15 \%$ of the tokens, the value used in BERT (Devlin et al., 2018). Again, since our text-to-text framework differs from BERT's, we are interested to see if a different corruption rate works better for us. We compare corruption rates of $10 \%$, $15 \%, 25 \%$, and $50 \%$ in Table 6 . Overall, we find that the corruption rate had a limited effect on the model's performance. The only exception is that the largest corruption rate we consider $(50 \%)$ results in a significant degradation of performance on GLUE and SQuAD. Using a larger corruption rate also results in longer targets, which can potentially slow down training. Based on these results and the historical precedent set by BERT, we will use a corruption rate of $15 \%$ going forward.

### 3.3.4 Corrupting Spans

We now turn towards the goal of speeding up training by predicting shorter targets. The approach we have used so far makes an i.i.d. decision for each input token as to whether to corrupt it or not. When multiple consecutive tokens have been corrupted, they are treated as a "span" and a single unique mask token is used to replace the entire span. Replacing entire spans with a single token results in unlabeled text data being processed into shorter sequences. Since we are using an i.i.d. corruption strategy, it is not always the case that a significant number of corrupted tokens appear consecutively. As a result, we might obtain additional speedup by specifically corrupting spans of tokens rather than corrupting individual tokens in an i.i.d. manner. Corrupting spans was also previously considered as a pre-training objective for BERT, where it was found to improve performance (Joshi et al., 2019).

To test this idea, we consider an objective that specifically corrupts contiguous, randomlyspaced spans of tokens. This objective can be parametrized by the proportion of tokens to be corrupted and the total number of corrupted spans. The span lengths are then chosen

| Span length | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\star$ Baseline (i.i.d.) | 83.28 | 19.24 | 80.88 | 71.36 | 26.98 | 39.82 | 27.65 |
| 2 | 83.54 | 19.39 | 82.09 | 72.20 | 26.76 | 39.99 | 27.63 |
| 3 | 83.49 | 19.62 | 81.84 | 72.53 | 26.86 | 39.65 | 27.62 |
| 5 | 83.40 | 19.24 | 82.05 | 72.23 | 26.88 | 39.40 | 27.53 |
| 10 | 82.85 | 19.33 | 81.84 | 70.44 | 26.79 | 39.49 | 27.69 |

Table 7: Performance of the span-corruption objective (inspired by Joshi et al. (2019)) for different average span lengths. In all cases, we corrupt $15 \%$ of the original text sequence.

randomly to satisfy these specified parameters. For example, if we are processing a sequence of 500 tokens and we have specified that $15 \%$ of tokens should be corrupted and that there should be 25 total spans, then the total number of corrupted tokens would be $500 \times 0.15=75$ and the average span length would be $75 / 25=3$. Note that given the original sequence length and corruption rate, we can equivalently parametrize this objective either by the average span length or the total number of spans.

We compare the span-corruption objective to the i.i.d-corruption objective in Table 7. We use a corruption rate of $15 \%$ in all cases and compare using average span lengths of 2,3 , 5 and 10. Again, we find a limited difference between these objectives, though the version with an average span length of 10 slightly underperforms the other values in some cases. We also find in particular that using an average span length of 3 slightly (but significantly) outperforms the i.i.d. objective on most non-translation benchmarks. Fortunately, the span-corruption objective also provides some speedup during training compared to the i.i.d. noise approach because span corruption produces shorter sequences on average.

### 3.3.5 DISCUSSION

Figure 5 shows a flow chart of the choices made during our exploration of unsupervised objectives. Overall, the most significant difference in performance we observed was that denoising objectives outperformed language modeling and deshuffling for pre-training. We did not observe a remarkable difference across the many variants of the denoising objectives we explored. However, different objectives (or parameterizations of objectives) can lead to different sequence lengths and thus different training speeds. This implies that choosing among the denoising objectives we considered here should mainly be done according to their computational cost. Our results also suggest that additional exploration of objectives similar to the ones we consider here may not lead to significant gains for the tasks and model we consider. Instead, it may be fortuitous to explore entirely different ways of leveraging unlabeled data.

### 3.4 Pre-training Data set

Like the unsupervised objective, the pre-training data set itself is a crucial component of the transfer learning pipeline. However, unlike objectives and benchmarks, new pre-training data sets are usually not treated as significant contributions on their own and are often not

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-25.jpg?height=463&width=916&top_left_y=300&top_left_x=602)

Figure 5: A flow chart of our exploration of unsupervised objectives. We first consider a few disparate approaches in Section 3.3.1 and find that a BERT-style denoising objective performs best. Then, we consider various methods for simplifying the BERT objective so that it produces shorter target sequences in Section 3.3.2. Given that replacing dropped-out spans with sentinel tokens performs well and results in short target sequences, in Section 3.3.3 we experiment with different corruption rates. Finally, we evaluate an objective that intentionally corrupts contiguous spans of tokens in Section 3.3.4.

released alongside pre-trained models and code. Instead, they are typically introduced in the course of presenting a new method or model. As a result, there has been relatively little comparison of different pre-training data sets as well as a lack of a "standard" data set used for pre-training. Some recent notable exceptions (Baevski et al., 2019; Liu et al., 2019c; Yang et al., 2019) have compared pre-training on a new large (often Common Crawl-sourced) data set to using a smaller preexisting data set (often Wikipedia). To probe more deeply into the impact of the pre-training data set on performance, in this section we compare variants of our $\mathrm{C} 4$ data set and other potential sources of pre-training data. We release all of the C4 data set variants we consider as part of TensorFlow Datasets. ${ }^{11}$

### 3.4.1 UnLabeled Data Sets

In creating C4, we developed various heuristics to filter the web-extracted text from Common Crawl (see Section 2.2 for a description). We are interested in measuring whether this filtering results in improved performance on downstream tasks, in addition to comparing it to other filtering approaches and common pre-training data sets. Towards this end, we compare the performance of our baseline model after pre-training on the following data sets:

C4 As a baseline, we first consider pre-training on our proposed unlabeled data set as described in Section 2.2.

Unfiltered C4 To measure the effect of the heuristic filtering we used in creating C4 (deduplication, removing bad words, only retaining sentences, etc.), we also generate an alternate version of $\mathrm{C} 4$ that forgoes this filtering. Note that we still use langdetect

11. https://www.tensorflow.org/datasets/catalog/c4
to extract English text. As a result, our "unfiltered" variant still includes some filtering because langdetect sometimes assigns a low probability to non-natural English text.

RealNews-like Recent work has used text data extracted from news websites (Zellers et al., 2019; Baevski et al., 2019). To compare to this approach, we generate another unlabeled data set by additionally filtering $\mathrm{C} 4$ to only include content from one of the domains used in the "RealNews" data set (Zellers et al., 2019). Note that for ease of comparison, we retain the heuristic filtering methods used in C4; the only difference is that we have ostensibly omitted any non-news content.

WebText-like Similarly, the WebText data set (Radford et al., 2019) only uses content from webpages that were submitted to the content aggregation website Reddit and received a "score" of at least 3. The score for a webpage submitted to Reddit is computed based on the proportion of users who endorse (upvote) or oppose (downvote) the webpage. The idea behind using the Reddit score as a quality signal is that users of the site would only upvote high-quality text content. To generate a comparable data set, we first tried removing all content from C4 that did not originate from a URL that appeared in the list prepared by the OpenWebText effort. ${ }^{12}$ However, this resulted in comparatively little content-only about $2 \mathrm{~GB}$-because most pages never appear on Reddit. Recall that C4 was created based on a single month of Common Crawl data. To avoid using a prohibitively small data set, we therefore downloaded 12 months of data from Common Crawl from August 2018 to July 2019, applied our heuristic filtering for C4, then applied the Reddit filter. This produced a 17 GB WebText-like data set, which is of comparable size to the original 40GB WebText data set (Radford et al., 2019).

Wikipedia The website Wikipedia consists of millions of encyclopedia articles written collaboratively. The content on the site is subject to strict quality guidelines and therefore has been used as a reliable source of clean and natural text. We use the English Wikipedia text data from TensorFlow Datasets, ${ }^{13}$ which omits any markup or reference sections from the articles.

Wikipedia + Toronto Books Corpus A drawback of using pre-training data from Wikipedia is that it represents only one possible domain of natural text (encyclopedia articles). To mitigate this, BERT (Devlin et al., 2018) combined data from Wikipedia with the Toronto Books Corpus (TBC) (Zhu et al., 2015). TBC contains text extracted from eBooks, which represents a different domain of natural language. BERT's popularity has led to the Wikipedia + TBC combination being used in many subsequent works.

The results achieved after pre-training on each of these data sets is shown in Table 8. A first obvious takeaway is that removing the heuristic filtering from C4 uniformly degrades performance and makes the unfiltered variant perform the worst in every task. Beyond this, we found that in some cases a pre-training data set with a more constrained domain outperformed the diverse C4 data set. For example, using the Wikipedia + TBC corpus[^7]

| Data set | Size | GLUE | CNNDM | SQuAD | SGLUE | EnDe | $\mathrm{EnFr}$ | EnRo |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{C} 4$ | $6 \mathrm{~GB}$ | 8 | 19.24 | $80.8-10$ | 71.36 | 3.98 | 39.82 | 7.65 |
|  |  |  |  |  |  | .55 |  | 7.21 |
|  | GF | 2 |  | 1: | ![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-27.jpg?height=43&width=140&top_left_y=457&top_left_x=1249) | 75 | 90 | 7.48 |
| Web | $17 \mathrm{~GB}$ | 84. |  | 81.4 | 1.40 | 26.80 | 39.74 | 27.59 |
| Wikipedia | $16 \mathrm{~GB}$ | 81.85 | 19.31 | 81.29 | 68.01 | 26.94 | 39.69 | 27.67 |
| Wikipedia + TBC | $20 \mathrm{~GB}$ | 83.65 | 19.28 | 82.08 | 73.24 | 26.77 | 39.63 | 27.57 |

Table 8: Performance resulting from pre-training on different data sets. The first four variants are based on our new $\mathrm{C} 4$ data set.

produced a SuperGLUE score of 73.24, beating our baseline's score (using C4) of 71.36. This is almost entirely attributable to a boost in performance from 25.78 (baseline, C4) to 50.93 (Wikipedia + TBC) on the Exact Match score for MultiRC (see Table 16). MultiRC is a reading comprehension data set whose largest source of data comes from fiction books, which is exactly the domain covered by TBC. Similarly, using the RealNews-like data set for pre-training conferred an increase from 68.16 to 73.72 on the Exact Match score for ReCoRD, a data set that measures reading comprehension on news articles. As a final example, using data from Wikipedia produced significant (but less dramatic) gains on $\mathrm{SQuAD}$, which is a question-answering data set with passages sourced from Wikipedia. Similar observations have been made in prior work, e.g. Beltagy et al. (2019) found that pre-training BERT on text from research papers improved its performance on scientific tasks. The main lesson behind these findings is that pre-training on in-domain unlabeled data can improve performance on downstream tasks. This is unsurprising but also unsatisfying if our goal is to pre-train a model that can rapidly adapt to language tasks from arbitrary domains. Liu et al. (2019c) also observed that pre-training on a more diverse data set yielded improvements on downstream tasks. This observation also motivates the parallel line of research on domain adaptation for natural language processing; for surveys of this field see e.g. Ruder (2019); Li (2012).

A drawback to only pre-training on a single domain is that the resulting data sets are often substantially smaller. Similarly, while the WebText-like variant performed as well or better than the $\mathrm{C} 4$ data set in our baseline setting, the Reddit-based filtering produced a data set that was about $40 \times$ smaller than C4 despite being based on $12 \times$ more data from Common Crawl. Note, however, that in our baseline setup we only pre-train on $2^{35} \approx 34 \mathrm{~B}$ tokens, which is only about 8 times larger than the smallest pre-training data set we consider. We investigate at what point using a smaller pre-training data sets poses an issue in the following section.

### 3.4.2 Pre-training Data Set Size

The pipeline we use to create $\mathrm{C} 4$ was designed to be able to create extremely large pretraining data sets. The access to so much data allows us to pre-train our models without repeating examples. It is not clear whether repeating examples during pre-training would be helpful or harmful to downstream performance because our pre-training objective is itself stochastic and can help prevent the model from seeing the same exact data multiple times.

| Number of tokens | Repeats | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ฝ Full data set | 0 | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| $2^{29}$ | 64 | $\mathbf{8 2 . 8 7}$ | $\mathbf{1 9 . 1 9}$ | $\mathbf{8 0 . 9 7}$ | $\mathbf{7 2 . 0 3}$ | $\mathbf{2 6 . 8 3}$ | $\mathbf{3 9 . 7 4}$ | $\mathbf{2 7 . 6 3}$ |
| $2^{27}$ | 256 | 82.62 | $\mathbf{1 9 . 2 0}$ | 79.78 | 69.97 | $\mathbf{2 7 . 0 2}$ | $\mathbf{3 9 . 7 1}$ | 2.37 |
| $2^{25}$ | 1,024 | 79.55 | 18.57 | 7.27 | 64.76 | 26.38 | 39.56 | 26.80 |
| $2^{23}$ | 4,096 | 76.34 | 18.33 | 70.92 | 59.29 | 26.37 | 38.84 | 25.81 |

Table 9: Measuring the effect of repeating data during pre-training. In these experiments, we only use the first $N$ tokens from C4 (with varying values of $N$ shown in the first column) but still pre-train over $2^{35}$ tokens. This results in the data set being repeated over the course of pre-training (with the number of repeats for each experiment shown in the second column), which may result in memorization (see Figure 6).

To test the effect of limited unlabeled data set sizes, we pre-trained our baseline model on artificially truncated versions of C4. Recall that we pre-train our baseline model on $2^{35} \approx 34 \mathrm{~B}$ tokens (a small fraction of the total size of C4). We consider training on truncated variants of $\mathrm{C} 4$ consisting of $2^{29}, 2^{27}, 2^{25}$ and $2^{23}$ tokens. These sizes correspond to repeating the data set $64,256,1,024$, and 4,096 times respectively over the course of pre-training.

The resulting downstream performance is shown in Table 9. As expected, performance degrades as the data set size shrinks. We suspect this may be due to the fact that the model begins to memorize the pre-training data set. To measure if this is true, we plot the training loss for each of these data set sizes in Figure 6. Indeed, the model attains significantly smaller training losses as the size of the pre-training data set shrinks, suggesting possible memorization. Baevski et al. (2019) similarly observed that truncating the pre-training data set size can degrade downstream task performance.

We note that these effects are limited when the pre-training data set is repeated only 64 times. This suggests that some amount of repetition of pre-training data might not be harmful. However, given that additional pre-training can be beneficial (as we will show in Section 3.6) and that obtaining additional unlabeled data is cheap and easy, we suggest using large pre-training data sets whenever possible. We also note that this effect may be more pronounced for larger model sizes, i.e. a bigger model may be more prone to overfitting to a smaller pre-training data set.

### 3.5 Training Strategy

So far we have considered the setting where all parameters of a model are pre-trained on an unsupervised task before being fine-tuned on individual supervised tasks. While this approach is straightforward, various alternative methods for training the model on downstream/supervised tasks have been proposed. In this section, we compare different schemes for fine-tuning the model in addition to the approach of training the model simultaneously on multiple tasks.

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-29.jpg?height=523&width=919&top_left_y=297&top_left_x=598)

Figure 6: Pre-training loss for our original $\mathrm{C} 4$ data set as well as 4 artificially truncated versions. The sizes listed refer to the number of tokens in each data set. The four sizes considered correspond to repeating the data set between 64 and 4,096 times over the course of pre-training. Using a smaller data set size results in smaller training loss values, which may suggest some memorization of the unlabeled data set.

### 3.5.1 Fine-TUNING Method

It has been argued that fine-tuning all of the model's parameters can lead to suboptimal results, particularly on low-resource tasks (Peters et al., 2019). Early results on transfer learning for text classification tasks advocated fine-tuning only the parameters of a small classifier that was fed sentence embeddings produced by a fixed pre-trained model (Subramanian et al., 2018; Kiros et al., 2015; Logeswaran and Lee, 2018; Hill et al., 2016; Conneau et al., 2017). This approach is less applicable to our encoder-decoder model because the entire decoder must be trained to output the target sequences for a given task. Instead, we focus on two alternative fine-tuning approaches that update only a subset of the parameters of our encoder-decoder model.

The first, "adapter layers" (Houlsby et al., 2019; Bapna et al., 2019), is motivated by the goal of keeping most of the original model fixed while fine-tuning. Adapter layers are additional dense-ReLU-dense blocks that are added after each of the preexisting feed-forward networks in each block of the Transformer. These new feed-forward networks are designed so that their output dimensionality matches their input. This allows them to be inserted into the network with no additional changes to the structure or parameters. When finetuning, only the adapter layer and layer normalization parameters are updated. The main hyperparameter of this approach is the inner dimensionality $d$ of the feed-forward network, which changes the number of new parameters added to the model. We experiment with various values for $d$.

The second alternative fine-tuning method we consider is "gradual unfreezing" (Howard and Ruder, 2018). In gradual unfreezing, more and more of the model's parameters are finetuned over time. Gradual unfreezing was originally applied to a language model architecture consisting of a single stack of layers. In this setting, at the start of fine-tuning only the

| Fine-tuning method | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ฝ All parameters | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| Adapter layers, $d=32$ | 80.52 | 15.08 | 79.32 | 60.40 | 13.84 | 17.88 | 15.54 |
| Adapter layers, $d=128$ | 81.51 | 16.62 | 79.47 | 63.03 | 19.83 | 27.50 | 22.63 |
| Adapter layers, $d=512$ | 81.54 | 17.78 | 79.18 | 64.30 | 23.45 | 33.98 | 25.81 |
| Adapter layers, $d=2048$ | 81.51 | 16.62 | 79.47 | 63.03 | 19.83 | 27.50 | 22.63 |
| Gradual unfreezing | 82.50 | 18.95 | 79.17 | $\mathbf{7 0 . 7 9}$ | 26.71 | 39.02 | 26.93 |

Table 10: Comparison of different alternative fine-tuning methods that only update a subset of the model's parameters. For adapter layers, $d$ refers to the inner dimensionality of the adapters.

parameters of the final layer are updated, then after training for a certain number of updates the parameters of the second-to-last layer are also included, and so on until the entire network's parameters are being fine-tuned. To adapt this approach to our encoder-decoder model, we gradually unfreeze layers in the encoder and decoder in parallel, starting from the top in both cases. Since the parameters of our input embedding matrix and output classification matrix are shared, we update them throughout fine-tuning. Recall that our baseline model consists of 12 layers each in the encoder and decoder and is fine-tuned for $2^{18}$ steps. As such, we subdivide the fine-tuning process into 12 episodes of $2^{18} / 12$ steps each and train from layers $12-n$ to 12 in the $n$th episode. We note that Howard and Ruder (2018) suggested fine-tuning an additional layer after each epoch of training. However, since our supervised data sets vary so much in size and since some of our downstream tasks are actually mixtures of many tasks (GLUE and SuperGLUE), we instead adopt the simpler strategy of fine-tuning an additional layer after every $2^{18} / 12$ steps.

A comparison of the performance of these fine-tuning approaches is shown in Table 10. For adapter layers, we report the performance using an inner dimensionality $d$ of 32,128 , 512, 2048. Pursuant with past results (Houlsby et al., 2019; Bapna et al., 2019) we find that lower-resource tasks like SQuAD work well with a small value of $d$ whereas higher resource tasks require a large dimensionality to achieve reasonable performance. This suggests that adapter layers could be a promising technique for fine-tuning on fewer parameters as long as the dimensionality is scaled appropriately to the task size. Note that in our case we treat GLUE and SuperGLUE each as a single "task" by concatenating their constituent data sets, so although they comprise some low-resource data sets the combined data set is large enough that it necessitates a large value of $d$. We found that gradual unfreezing caused a minor degradation in performance across all tasks, though it did provide some speedup during fine-tuning. Better results may be attainable by more carefully tuning the unfreezing schedule.

### 3.5.2 Multi-task Learning

So far, we have been pre-training our model on a single unsupervised learning task before fine-tuning it individually on each downstream task. An alternative approach, called "multitask learning" (Ruder, 2017; Caruana, 1997), is to train the model on multiple tasks at a time. This approach typically has the goal of training a single model that can simultaneously
perform many tasks at once, i.e. the model and most of its parameters are shared across all tasks. We relax this goal somewhat and instead investigate methods for training on multiple tasks at once in order to eventually produce separate parameter settings that perform well on each individual task. For example, we might train a single model on many tasks, but when reporting performance we are allowed to select a different checkpoint for each task. This loosens the multi-task learning framework and puts it on more even footing compared to the pre-train-then-fine-tune approach we have considered so far. We also note that in our unified text-to-text framework, "multi-task learning" simply corresponds to mixing data sets together. It follows that we can still train on unlabeled data when using multi-task learning by treating the unsupervised task as one of the tasks being mixed together. In contrast, most applications of multi-task learning to NLP add task-specific classification networks or use different loss functions for each task (Liu et al., 2019b).

As pointed out by Arivazhagan et al. (2019), an extremely important factor in multi-task learning is how much data from each task the model should be trained on. Our goal is to not under- or over-train the model - that is, we want the model to see enough data from a given task that it can perform the task well, but not to see so much data that it memorizes the training set. How exactly to set the proportion of data coming from each task can depend on various factors including data set sizes, the "difficulty" of learning the task (i.e. how much data the model must see before being able to perform the task effectively), regularization, etc. An additional issue is the potential for "task interference" or "negative transfer", where achieving good performance on one task can hinder performance on another. Given these concerns, we begin by exploring various strategies for setting the proportion of data coming from each task. A similar exploration was performed by Wang et al. (2019a).

Examples-proportional mixing A major factor in how quickly a model will overfit to a given task is the task's data set size. As such, a natural way to set the mixing proportions is to sample in proportion to the size of each task's data set. This is equivalent to concatenating the data sets for all tasks and randomly sampling examples from the combined data set. Note, however, that we are including our unsupervised denoising task, which uses a data set that is orders of magnitude larger than every other task's. It follows that if we simply sample in proportion to each data set's size, the vast majority of the data the model sees will be unlabeled, and it will undertrain on all of the supervised tasks. Even without the unsupervised task, some tasks (e.g. WMT English to French) are so large that they would similarly crowd out most of the batches. To get around this issue, we set an artificial "limit" on the data set sizes before computing the proportions. Specifically, if the number of examples in each of our $N$ task's data sets is $e_{n}, n \in\{1, \ldots, N\}$ then we set probability of sampling an example from the $m$ th task during training to $r_{m}=\min \left(e_{m}, K\right) / \sum \min \left(e_{n}, K\right)$ where $K$ is the artificial data set size limit.

Temperature-scaled mixing An alternative way of mitigating the huge disparity between data set sizes is to adjust the "temperature" of the mixing rates. This approach was used by multilingual BERT to ensure that the model was sufficiently trained on lowresource languages. ${ }^{14}$ To implement temperature scaling with temperature $T$, we raise

14. https://github.com/google-research/bert/blob/master/multilingual.md
each task's mixing rate $r_{m}$ to the power of $1 / T$ and renormalize the rates so that they sum to 1 . When $T=1$, this approach is equivalent to examples-proportional mixing and as $T$ increases the proportions become closer to equal mixing. We retain the data set size limit $K$ (applied to obtain $r_{m}$ before temperature scaling) but set it to a large value of $K=2^{21}$. We use a large value of $K$ because increasing the temperature will decrease the mixing rate of the largest data sets.

Equal mixing In this case, we sample examples from each task with equal probability. Specifically, each example in each batch is sampled uniformly at random from one of the data sets we train on. This is most likely a suboptimal strategy, as the model will overfit quickly on low-resource tasks and underfit on high-resource tasks. We mainly include it as a point of reference of what might go wrong when the proportions are set suboptimally.

To compare these mixing strategies on equal footing with our baseline pre-train-thenfine-tune results, we train multi-task models for the same total number of steps: $2^{19}+2^{18}=$ 786,432 . The results are shown in Table 11.

In general, we find that multi-task training underperforms pre-training followed by fine-tuning on most tasks. The "equal" mixing strategy in particular results in dramatically degraded performance, which may be because the low-resource tasks have overfit, the highresource tasks have not seen enough data, or the model has not seen enough unlabeled data to learn general-purpose language capabilities. For examples-proportional mixing, we find that for most tasks there is a "sweet spot" for $K$ where the model obtains the best performance, and larger or smaller values of $K$ tend to result in worse performance. The exception (for the range of $K$ values we considered) was WMT English to French translation, which is such a high-resource task that it always benefits from a higher mixing proportion. Finally, we note that temperature-scaled mixing also provides a means of obtaining reasonable performance from most tasks, with $T=2$ performing the best in most cases. The finding that a multi-task model is outperformed by separate models trained on each individual task has previously been observed e.g. by Arivazhagan et al. (2019) and McCann et al. (2018), though it has been shown that the multi-task setup can confer benefits across very similar tasks Liu et al. (2019b); Ratner et al. (2018). In the following section, we explore ways to close the gap between multi-task training and the pre-train-then-fine-tune approach.

### 3.5.3 Combining Multi-Task Learning with Fine-Tuning

Recall that we are studying a relaxed version of multi-task learning where we train a single model on a mixture of tasks but are allowed to evaluate performance using different parameter settings (checkpoints) for the model. We can extend this approach by considering the case where the model is pre-trained on all tasks at once but is then fine-tuned on the individual supervised tasks. This is the method used by the "MT-DNN" (Liu et al., 2015, 2019b), which achieved state-of-the-art performance on GLUE and other benchmarks when it was introduced. We consider three variants of this approach: In the first, we simply pre-train the model on an examples-proportional mixture with an artificial data set size limit of $K=2^{19}$ before fine-tuning it on each individual downstream task. This helps us measure whether including the supervised tasks alongside the unsupervised objective during pre-training

| Mixing strategy | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 太 Baseline (pre-train/fine-tune) | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | $\mathbf{3 9 . 8 2}$ | $\mathbf{2 7 . 6 5}$ |
| Equal | 76.13 | 19.02 | 76.51 | 63.37 | 23.89 | 34.31 | 26.78 |
| Examples-proportional, $K=2^{16}$ | 80.45 | 19.04 | 77.25 | 69.95 | 24.35 | 34.99 | 27.10 |
| Examples-proportional, $K=2^{17}$ | 81.56 | 19.12 | 77.00 | 67.91 | 24.36 | 35.00 | 27.25 |
| Examples-proportional, $K=2^{18}$ | 81.67 | 19.07 | 78.17 | 67.94 | 24.57 | 35.19 | 27.39 |
| Examples-proportional, $K=2^{19}$ | 81.42 | $\mathbf{1 9 . 2 4}$ | 79.78 | 67.30 | 25.21 | 36.30 | $\mathbf{2 7 . 7 6}$ |
| Examples-proportional, $K=2^{20}$ | 80.80 | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 3 6}$ | 67.38 | 25.66 | 36.93 | $\mathbf{2 7 . 6 8}$ |
| Examples-proportional, $K=2^{21}$ | 79.83 | 18.79 | 79.50 | 65.10 | 25.82 | 37.22 | 27.13 |
| Temperature-scaled, $T=2$ | 81.90 | $\mathbf{1 9 . 2 8}$ | 79.42 | 69.92 | 25.42 | 36.72 | 27.20 |
| Temperature-scaled, $T=4$ | 80.56 | $\mathbf{1 9 . 2 2}$ | 77.99 | 69.54 | 25.04 | 35.82 | 27.45 |
| Temperature-scaled, $T=8$ | 77.21 | 19.10 | 77.14 | 66.07 | 24.55 | 35.35 | 27.17 |

Table 11: Comparison of multi-task training using different mixing strategies. Examplesproportional mixing refers to sampling examples from each data set according to the total size of each data set, with an artificial limit $(K)$ on the maximum data set size. Temperature-scaled mixing re-scales the sampling rates by a temperature $T$. For temperature-scaled mixing, we use an artificial data set size limit of $K=2^{21}$.

gives the model some beneficial early exposure to the downstream tasks. We might also hope that mixing in many sources of supervision could help the pre-trained model obtain a more general set of "skills" (loosely speaking) before it is adapted to an individual task. To measure this directly, we consider a second variant where we pre-train the model on the same examples-proportional mixture (with $K=2^{19}$ ) except that we omit one of the downstream tasks from this pre-training mixture. Then, we fine-tune the model on the task that was left out during pre-training. We repeat this for each of the downstream tasks we consider. We call this approach "leave-one-out" multi-task training. This simulates the real-world setting where a pre-trained model is fine-tuned on a task it had not seen during pre-training. Note that multi-task pre-training provides a diverse mixture of supervised tasks. Since other fields (e.g. computer vision (Oquab et al., 2014; Jia et al., 2014; Huh et al., 2016; Yosinski et al., 2014)) use a supervised data set for pre-training, we were interested to see whether omitting the unsupervised task from the multi-task pre-training mixture still produced good results. For our third variant we therefore pre-train on an examples-proportional mixture of all of the supervised tasks we consider with $K=2^{19}$. In all of these variants, we follow our standard procedure of pre-training for $2^{19}$ steps before fine-tuning for $2^{18}$ steps.

We compare the results of these approaches in Table 12. For comparison, we also include results for our baseline (pre-train then fine-tune) and for standard multi-task learning (without fine-tuning) on an examples-proportional mixture with $K=2^{19}$. We find that fine-tuning after multi-task pre-training results in comparable performance to our baseline. This suggests that using fine-tuning after multi-task learning can help mitigate some of the trade-offs between different mixing rates described in Section 3.5.2. Interestingly, the performance of "leave-one-out" training was only slightly worse, suggesting that a model that was trained on a variety of tasks can still adapt to new tasks (i.e. multi-task pretraining might not result in a dramatic task interference). Finally, supervised multi-task pre-training performed significantly worse in every case except for the translation tasks. This

| Training strategy | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 丸 Unsupervised pre-training + fine-tuning | $\mathbf{8 3 . 2 8}$ | $\mathbf{1 9 . 2 4}$ | $\mathbf{8 0 . 8 8}$ | $\mathbf{7 1 . 3 6}$ | $\mathbf{2 6 . 9 8}$ | 39.82 | 27.65 |
| Multi-task training | 81.42 | $\mathbf{1 9 . 2 4}$ | 79.78 | 67.30 | 25.21 | 36.30 | 27.76 |
| Multi-task pre-training + fine-tuning | $\mathbf{8 3 . 1 1}$ | $\mathbf{1 9 . 1 2}$ | $\mathbf{8 0 . 2 6}$ | $\mathbf{7 1 . 0 3}$ | $\mathbf{2 7 . 0 8}$ | 39.80 | $\mathbf{2 8 . 0 7}$ |
| Leave-one-out multi-task training | 81.98 | 19.05 | 79.97 | $\mathbf{7 1 . 6 8}$ | $\mathbf{2 6 . 9 3}$ | 39.79 | $\mathbf{2 7 . 8 7}$ |
| Supervised multi-task pre-training | 79.93 | 18.96 | 77.38 | 65.36 | 26.81 | $\mathbf{4 0 . 1 3}$ | $\mathbf{2 8 . 0 4}$ |

Table 12: Comparison of unsupervised pre-training, multi-task learning, and various forms of multi-task pre-training.

could suggest that the translation tasks benefit less from (English) pre-training, whereas unsupervised pre-training is an important factor in the other tasks.

### 3.6 Scaling

The "bitter lesson" of machine learning research argues that general methods that can leverage additional computation ultimately win out against methods that rely on human expertise (Sutton, 2019; Hestness et al., 2017; Shazeer et al., 2017; Jozefowicz et al., 2016; Mahajan et al., 2018; Shazeer et al., 2018, 2017; Huang et al., 2018b; Keskar et al., 2019a). Recent results suggest that this may hold true for transfer learning in NLP (Liu et al., 2019c; Radford et al., 2019; Yang et al., 2019; Lan et al., 2019), i.e. it has repeatedly been shown that scaling up produces improved performance compared to more carefully-engineered methods. However, there are a variety of possible ways to scale, including using a bigger model, training the model for more steps, and ensembling. In this section, we compare these different approaches by addressing the following premise: "You were just given $4 \times$ more compute. How should you use it?"

We start with our baseline model, which has $220 \mathrm{M}$ parameters and is pre-trained and fine-tuned for $2^{19}$ and $2^{18}$ steps respectively. The encoder and decoder are both sized

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-34.jpg?height=46&width=1515&top_left_y=1733&top_left_x=305)
of "BERT ${ }_{\text {LARGE" }}$ Devlin et al. (2018) and use $d_{\mathrm{ff}}=4096, d_{\text {model }}=1024, d_{\mathrm{kv}}=64$ and 16-head attention mechanisms. We then generate two variants with 16 and 32 layers each in the encoder and decoder, producing models with $2 \times$ and $4 \times$ as many parameters as our original model. These two variants also have a roughly $2 \times$ and $4 \times$ the computational cost. Using our baseline and these two larger models, we consider three ways of using $4 \times$ as much computation: Training for $4 \times$ as many steps, training for $2 \times$ as many steps with the $2 \times$ bigger model, and training the $4 \times$ bigger model for the "baseline" number of training steps. When we increase the training steps, we scale both the pre-train and fine-tune steps for simplicity. Note that when increasing the number of pre-training steps, we are effectively including more pre-training data as $\mathrm{C} 4$ is so large that we do not complete one pass over the data even when training for $2^{23}$ steps.

An alternative way for the model to see $4 \times$ as much data is to increase the batch size by a factor of 4. This can potentially result in faster training due to more efficient parallelization. However, training with a $4 \times$ larger batch size can yield a different outcome than training

| Scaling strategy | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 83.28 | 19.24 | 80.88 | 71.36 | 26.98 | 39.82 | 27.65 |
| $1 \times$ size, $4 \times$ training steps | 85.33 | 19.33 | 82.45 | 74.72 | 27.08 | 40.66 | 27.93 |
| $1 \times$ size, $4 \times$ batch size | 84.60 | 19.42 | 82.52 | 74.64 | 27.07 | 40.60 | 27.84 |
| $2 \times$ size, $2 \times$ training steps | $\mathbf{8 6 . 1 8}$ | 19.66 | $\mathbf{8 4 . 1 8}$ | 77.18 | 27.52 | $\mathbf{4 1 . 0 3}$ | 28.19 |
| $4 \times$ size, $1 \times$ training steps | $\mathbf{8 5 . 9 1}$ | 19.73 | $\mathbf{8 3 . 8 6}$ | $\mathbf{7 8 . 0 4}$ | 27.47 | 40.71 | 28.10 |
| $4 \times$ ensembled | 84.77 | $\mathbf{2 0 . 1 0}$ | 83.09 | 71.74 | $\mathbf{2 8 . 0 5}$ | 40.53 | $\mathbf{2 8 . 5 7}$ |
| $4 \times$ ensembled, fine-tune only | 84.05 | 19.57 | 82.36 | 71.55 | 27.55 | 40.22 | 28.09 |

Table 13: Comparison of different methods of scaling up our baseline model. All methods except ensembling fine-tuned models use $4 \times$ the computation as the baseline. "Size" refers to the number of parameters in the model and "training time" refers to the number of steps used for both pre-training and fine-tuning.

for $4 \times$ as many steps (Shallue et al., 2018). We include an additional experiment where we train our baseline model with a $4 \times$ larger batch size to compare these two cases.

It is common practice on many of the benchmarks we consider to eke out additional performance by training and evaluating using an ensemble of models. This provides an orthogonal way of using additional computation. To compare other scaling methods to ensembling, we also measure the performance of an ensemble of 4 separately pre-trained and fine-tuned models. We average the logits across the ensemble before feeding them into the output softmax nonlinearity to obtain an aggregate prediction. Instead of pre-training 4 separate models, a cheaper alternative is to take a single pre-trained model and produce 4 separate fine-tuned versions. While this does not use our entire $4 \times$ computational budget, we also include this method to see if it produces competitive performance to the other scaling methods.

The performance achieved after applying these various scaling methods is shown in Table 13. Unsurprisingly, increasing the training time and/or model size consistently improves the baseline. There was no clear winner between training for $4 \times$ as many steps or using a $4 \times$ larger batch size, though both were beneficial. In general, increasing the model size resulted in an additional bump in performance compared to solely increasing the training time or batch size. We did not observe a large difference between training a $2 \times$ bigger model for $2 \times$ as long and training a $4 \times$ bigger model on any of the tasks we studied. This suggests that increasing the training time and increasing the model size can be complementary means of improving performance. Our results also suggest that ensembling provides an orthogonal and effective means of improving performance through scale. In some tasks (CNN/DM, WMT English to German, and WMT English to Romanian), ensembling 4 completely separately trained models significantly outperformed every other scaling approach. Ensembling models that were pre-trained together but fine-tuned separately also gave a substantial performance increase over the baseline, which suggests a cheaper means of improving performance. The only exception was SuperGLUE, where neither ensembling approach significantly improved over the baseline.

We note that different scaling methods have different trade-offs that are separate from their performance. For example, using a larger model can make downstream fine-tuning and
inference more expensive. In contrast, the cost of pre-training a small model for longer is effectively amortized if it is applied to many downstream tasks. Separately, we note that ensembling $N$ separate models has a similar cost to using a model that has an $N \times$ higher computational cost. As a result, some consideration for the eventual use of the model is important when choosing between scaling methods.

### 3.7 Putting It All Together

We now leverage the insights from our systematic study to determine how far we can push performance on popular NLP benchmarks. We are also interested in exploring the current limits of transfer learning for NLP by training larger models on large amounts of data. We start with our baseline training approach and make the following changes:

Objective We swap out the i.i.d. denoising objective in our baseline for the span-corruption objective described in Section 3.3.4, which was loosely inspired by SpanBERT (Joshi et al., 2019). Specifically, we use a mean span length of 3 and corrupt $15 \%$ of the original sequence. We found that this objective produced marginally better performance (Table 7) while being slightly more computationally efficient due to shorter target sequence lengths.

Longer training Our baseline model uses a relatively small amount of pre-training ( $1 / 4$ as much as BERT (Devlin et al., 2018), 11/16 as much as XLNet (Yang et al., 2019), $1 / 64$ as much as RoBERTa (Liu et al., 2019c), etc.). Fortunately, C4 is big enough that we can train for substantially longer without repeating data (which can be detrimental, as shown in Section 3.4.2). We found in Section 3.6 that additional pre-training can indeed be helpful, and that both increasing the batch size and increasing the number of training steps can confer this benefit. We therefore pre-train our models for 1 million steps on a batch size of $2^{11}$ sequences of length 512, corresponding to a total of about 1 trillion pre-training tokens (about $32 \times$ as many as our baseline). In Section 3.4.1, we showed that pre-training on the RealNews-like, WebText-like, and Wikipedia + TBC data sets outperformed pre-training on $\mathrm{C} 4$ on a few downstream tasks. However, these data set variants are sufficiently small that they would be repeated hundreds of times over the course of pre-training on 1 trillion tokens. Since we showed in Section 3.4.2 that this repetition could be harmful, we opted instead to continue using the $\mathrm{C} 4$ data set.

Model sizes In Section 3.6 we also showed how scaling up the baseline model size improved performance. However, using smaller models can be helpful in settings where limited computational resources are available for fine-tuning or inference. Based on these factors, we train models with a wide range of sizes:

- Base. This is our baseline model, whose hyperparameters are described in Section 3.1.1. It has roughly 220 million parameters.
- Small. We consider a smaller model, which scales the baseline down by using $d_{\text {model }}=512, d_{\mathrm{ff}}=2,048,8$-headed attention, and only 6 layers each in the encoder and decoder. This variant has about 60 million parameters.
- Large. Since our baseline uses a BERT $\mathrm{BASE}_{\text {-Sized encoder and decoder, we }}$ also consider a variant where the encoder and decoder are both similar in size and structure to $\mathrm{BERT}_{\text {LARGE }}$. Specifically, this variant uses $d_{\text {model }}=1,024$, $d_{\mathrm{ff}}=4,096, d_{\mathrm{kv}}=64,16$-headed attention, and 24 layers each in the encoder and decoder, resulting in around 770 million parameters.
- 3B and 11B. To further explore what kind of performance is possible when using larger models, we consider two additional variants. In both cases, we use $d_{\text {model }}=1024$, a 24 layer encoder and decoder, and $d_{\mathrm{kv}}=128$. For the " $3 \mathrm{~B}$ " variant, we use $d_{\mathrm{ff}}=16,384$ with 32 -headed attention, which results in around 2.8 billion parameters; for "11B" we use $d_{\mathrm{ff}}=65,536$ with 128 -headed attention producing a model with about 11 billion parameters. We chose to scale up $d_{\mathrm{ff}}$ specifically because modern accelerators (such as the TPUs we train our models on) are most efficient for large dense matrix multiplications like those in the Transformer's feed-forward networks.

Multi-task pre-training In Section 3.5.3, we showed that pre-training on a multi-task mixture of unsupervised and supervised tasks before fine-tuning worked as well as pre-training on the unsupervised task alone. This is the approach advocated by the "MT-DNN" (Liu et al., 2015, 2019b). It also has the practical benefit of being able to monitor "downstream" performance for the entire duration of training, rather than just during fine-tuning. We therefore used multi-task pre-training in our final set of experiments. We hypothesize that larger models trained for longer might benefit from a larger proportion of unlabeled data because they are more likely to overfit to smaller training data sets. However, we also note that the results of Section 3.5.3 suggest that fine-tuning after multi-task pre-training can mitigate some of the issues that might arise from choosing a suboptimal proportion of unlabeled data. Based on these ideas, we substitute the following artificial data set sizes for our unlabeled data before using standard example-proportional mixing (described in Section 3.5.2): 710,000 for Small, 2,620,000 for Base, 8,660,000 for Large, 33,500,000 for 3B, and 133,000,000 for 11B. For all model variants, we also capped the effective data set size of the WMT English to French and WMT English to German data sets to 1M examples during pre-training.

Fine-tuning on individual GLUE and SuperGLUE tasks So far, when fine-tuning on GLUE and SuperGLUE, we have concatenated all of the data sets in each benchmark so that we only fine-tune models once for GLUE and once for SuperGLUE. This approach makes our study logistically simpler, but we found that this sacrifices a small amount of performance on some tasks compared to fine-tuning on the task separately. A potential issue with fine-tuning on individual tasks, which would otherwise be mitigated by training on all tasks at once, is that we might overfit quickly to low-resource tasks. For example, our large batch size of $2^{11}$ length-512 sequences would result in the entire data set appearing multiple times in each batch for many of the low-resource GLUE and SuperGLUE tasks. We therefore use a smaller batch size of 8 length-512 sequences during fine-tuning for each GLUE and SuperGLUE task. We also save checkpoints every 1,000 steps rather than every 5,000 steps to ensure we have access to the model's parameters before it overfits.

Beam search All of our previous results were reported using greedy decoding. For tasks with long output sequences, we found improved performance from using beam search (Sutskever et al., 2014). Specifically, we use a beam width of 4 and a length penalty of $\alpha=0.6$ (Wu et al., 2016) for the WMT translation and CNN/DM summarization tasks.

Test set Since this is our final set of experiments, we report results on the test set rather than the validation set. For CNN/Daily Mail, we use the standard test set distributed with the data set. For the WMT tasks, this corresponds to using newstest2014 for English-German, newstest2015 for English-French, and newstest2016 for EnglishRomanian. For GLUE and SuperGLUE, we used the benchmark evaluation servers to compute official test set scores. ${ }^{15,16}$ For SQuAD, evaluating on the test set requires running inference on a benchmark server. Unfortunately, the computational resources on this server are insufficient for obtaining predictions from our largest models. As a result, we instead continue to report performance on the $\mathrm{SQuAD}$ validation set. Fortunately, the model with the highest performance on the SQuAD test set also reported results on the validation set, so we can still compare to what is ostensibly the state-of-the-art.

Apart from those changes mentioned above, we use the same training procedure and hyperparameters as our baseline (AdaFactor optimizer, inverse square root learning rate schedule for pre-training, constant learning rate for fine-tuning, dropout regularization, vocabulary, etc.). For reference, these details are described in Section 2.

The results of this final set of experiments are shown in Table 14. Overall, we achieved state-of-the-art performance on 18 out of the 24 tasks we consider. As expected, our largest (11 billion parameter) model performed best among our model size variants across all tasks. Our T5-3B model variant did beat the previous state of the art in a few tasks, but scaling the model size to 11 billion parameters was the most important ingredient for achieving our best performance. We now analyze the results for each individual benchmark.

We achieved a state-of-the-art average GLUE score of 90.3. Notably, our performance was substantially better than the previous state-of-the-art for the natural language inference tasks MNLI, RTE, and WNLI. RTE and WNLI are two of the tasks where machine performance has historically lagged behind human performance, which is 93.6 and 95.9 respectively (Wang et al., 2018). In terms of parameter count, our 11B model variant is the largest model that has been submitted to the GLUE benchmark. However, most of the best-scoring submissions use a large amount of ensembling and computation to produce predictions. For example, the best-performing variant of ALBERT (Lan et al., 2019) uses a model similar in size and architecture to our 3B variant (though it has dramatically fewer parameters due to clever parameter sharing). To produce its impressive performance on GLUE, the ALBERT authors ensembled "from 6 to 17" models depending on the task. This likely results in it being more computationally expensive to produce predictions with the ALBERT ensemble than it is with T5-11B.

For SQuAD, we outperformed the previous state-of-the-art (ALBERT (Lan et al., 2019)) by over one point on the Exact Match score. SQuAD is a long-standing benchmark that

15. http://gluebenchmark.com
16. http://super.gluebenchmark.com

![](https://cdn.mathpix.com/cropped/2024_05_26_744c1674610f11c9deceg-39.jpg?height=1746&width=1461&top_left_y=298&top_left_x=337)

Table 14: Performance of our T5 variants on every task we study. Small, Base, Large, 3B, and 11B refer to model configurations with 60 million, 220 million, 770 million, 3 billion, and 11 billion parameters, respectively. In the first row of each table, we report the state-of-the-art for the task (as of October 24th, 2019), with the superscript denoting its source with references listed at the end of this caption. All results are reported on the test set except for $\mathrm{SQuAD}$ where we use the validation set. ${ }^{a}$ (Lan et al., 2019) ${ }^{b}$ (Wang et al., 2019c) ${ }^{c}$ (Zhu et al., 2019) ${ }^{d}$ (Liu et al., 2019c) ${ }^{e}$ (Edunov et al., 2018) ${ }^{f}$ (Lample and Conneau, 2019) ${ }^{g}$ (Dong et al., 2019)
was created over three years ago, and most recent improvements have only increased the state-of-the-art by a fraction of a percentage point. We note that when results are reported on the test set, they are typically based on an ensemble of models and/or leverage external data sets (e.g. TriviaQA (Joshi et al., 2017) or NewsQA (Trischler et al., 2016)) to augment the small SQuAD training set. Human performance on SQuAD is estimated at 82.30 and 91.22 for the Exact Match and F1 metric respectively (Rajpurkar et al., 2016), so it is not clear if further improvements on this benchmark are meaningful.

For SuperGLUE, we improved upon the state-of-the-art by a large margin (from an average score of 84.6 (Liu et al., 2019c) to 88.9). SuperGLUE was designed to include tasks that were "beyond the scope of current state-of-the-art systems, but solvable by most college-educated English speakers" (Wang et al., 2019b). We nearly match the human performance of 89.8 (Wang et al., 2019b). Interestingly, on the reading comprehension tasks (MultiRC and ReCoRD) we exceed human performance by a large margin, suggesting the evaluation metrics used for these tasks may be biased towards machine-made predictions. On the other hand, humans achieve $100 \%$ accuracy on both COPA and WSC, which is significantly better than our model's performance. This suggests that there remain linguistic tasks that are hard for our model to perfect, particularly in the low-resource setting.

We did not achieve state-of-the-art performance on any of the WMT translation tasks. This may be in part due to our use of an English-only unlabeled data set. We also note that most of the best results on these tasks use backtranslation (Edunov et al., 2018; Lample and Conneau, 2019), which is a sophisticated data augmentation scheme. The state of the art on the low-resource English to Romanian benchmark also uses additional forms of cross-lingual unsupervised training (Lample and Conneau, 2019). Our results suggest that scale and English-language pre-training may be insufficient to match the performance of these more sophisticated methods. On a more specific note, the best results on English to German newstest2014 set use the much larger training set from WMT 2018 (Edunov et al., 2018), making direct comparison to our results difficult.

Finally, on CNN/Daily Mail we attain state-of-the-art performance, though only by a significant amount on the ROUGE-2-F score. It has been shown that improvements to the ROUGE score do not necessarily correspond to more coherent summaries (Paulus et al., 2017). Furthermore, while CNN/Daily Mail is posed as an abstractive summarization benchmark, purely extractive approaches have been shown to work well (Liu, 2019). It has also been argued that generative models trained with maximum likelihood are prone to producing repetitive summaries (See et al., 2017). Despite these potential issues, we find that our models do generate coherent and largely correct summaries. We provide some non-cherry-picked validation set examples in Appendix C.

To achieve its strong results, T5 combines insights from our experimental study with unprecedented scale. Note that in Section 3.6 we found that scaling up the pre-training amount or size of our baseline model produced substantial gains. Given this, we were interested to measure how much the "non-scaling" changes we introduced into T5 contributed to its strong performance. We therefore carried out a final experiment where we compared the following three configurations: First, the standard baseline model, which was pre-trained on $2^{35} \approx 34 \mathrm{~B}$ tokens; second, the baseline trained instead for about 1 trillion tokens (i.e. the same amount of pre-training used for T5), which we refer to as "baseline-1T"; and third, T5-Base. Note that the differences between baseline-1T and T5-Base comprise the

| Model | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A Baseline | 83.28 | 19.24 | 80.88 | 71.36 | 26.98 | 39.82 | 27.65 |
| Baseline-1T | 84.80 | 19.62 | 83.01 | 73.90 | 27.46 | 40.30 | 28.34 |
| T5-Base | $\mathbf{8 5 . 9 7}$ | $\mathbf{2 0 . 9 0}$ | $\mathbf{8 5 . 4 4}$ | $\mathbf{7 5 . 6 4}$ | $\mathbf{2 8 . 3 7}$ | $\mathbf{4 1 . 3 7}$ | $\mathbf{2 8 . 9 8}$ |

Table 15: Performance comparison of T5-Base to our baseline experimental setup used in the rest of the paper. Results are reported on the validation set. "Baseline-1T" refers to the performance achieved by pre-training the baseline model on 1 trillion tokens (the same number used for the T5 model variants) instead of $2^{35} \approx 34 \mathrm{~B}$ tokens (as was used for the baseline).

"non-scaling" changes we made when designing T5. As such, comparing the performance of these two models gives us a concrete measurement of the impact of the insights from our systematic study.

The performance of these three model configurations is shown in Table 15. Consistent with the findings in Section 3.6, we find that additional pre-training improves performance over the baseline. Nevertheless, T5-Base substantially outperforms baseline-1T on all downstream tasks. This suggests that scale is not the only factor that contributes to T5's success. We hypothesize that the larger models benefit not only from their increased size but also from these non-scaling factors.

## 4. Reflection

Having completed our systematic study, we wrap up by first recapping some of our most significant findings. Our results provide some high-level perspective on which avenues of research might be more or less promising. To conclude, we outline some topics we think might provide effective approaches for further progressing the field.

### 4.1 Takeaways

Text-to-text Our text-to-text framework provides a simple way to train a single model on a wide variety of text tasks using the same loss function and decoding procedure. We showed how this approach can be successfully applied to generative tasks like abstractive summarization, classification tasks like natural language inference, and even regression tasks like STS-B. In spite of its simplicity, we found the text-totext framework obtained comparable performance to task-specific architectures and ultimately produced state-of-the-art results when combined with scale.

Architectures While some work on transfer learning for NLP has considered architectural variants of the Transformer, we found the original encoder-decoder form worked best in our text-to-text framework. Though an encoder-decoder model uses twice as many parameters as "encoder-only" (e.g. BERT) or "decoder-only" (language model) architectures, it has a similar computational cost. We also showed that sharing the parameters in the encoder and decoder did not result in a substantial performance drop while halving the total parameter count.

Unsupervised objectives Overall, we found that most "denoising" objectives, which train the model to reconstruct randomly corrupted text, performed similarly in the text-totext setup. As a result, we suggest using objectives that produce short target sequences so that unsupervised pre-training is more computationally efficient.

Data sets We introduced the "Colossal Clean Crawled Corpus" (C4), which comprises heuristically-cleaned text from the Common Crawl web dump. When comparing C4 to data sets that use additional filtering, we found that training on in-domain unlabeled data could boost performance in a few downstream tasks. However, constraining to a single domain typically results in a smaller data set. We separately showed that performance can degrade when an unlabeled data set is small enough that it is repeated many times over the course of pre-training. This motivates the use of a large and diverse data set like $\mathrm{C} 4$ for generic language understanding tasks.

Training strategies We found that the basic approach of updating all of a pre-trained model's parameters during fine-tuning outperformed methods that are designed to update fewer parameters, although updating all parameters is most expensive. We also experimented with various approaches for training the model on multiple tasks at once, which in our text-to-text setting simply corresponds to mixing examples from different data sets when constructing batches. The primary concern in multi-task learning is setting the proportion of each task to train on. We ultimately did not find a strategy for setting mixing proportions that matched the performance of the basic approach of unsupervised pre-training followed by supervised fine-tuning. However, we found that fine-tuning after pre-training on a mixture of tasks produced comparable performance to unsupervised pre-training.

Scaling We compared various strategies for taking advantage of additional compute, including training the model on more data, training a larger model, and using an ensemble of models. We found each approach conferred a significant boost in performance, though training a smaller model on more data was often outperformed by training a larger model for fewer steps. We also showed an ensemble of models can provide substantially better results than a single model, which provides an orthogonal means of leveraging additional computation. Ensembling models that were fine-tuned from the same base pre-trained model performed worse than pre-training and fine-tuning all models completely separately, though fine-tune-only ensembling still substantially outperformed a single model.

Pushing the limits We combined our above insights and trained substantially larger models (up to 11 billion parameters) to achieve state-of-the-art results across many of the benchmarks we considered. For unsupervised training, we extracted text from our C4 data set and applied a denoising objective that corrupts contiguous spans of tokens. We pre-trained on a multi-task mixture before fine-tuning on individual tasks. Overall, our models were trained on over 1 trillion tokens. In the interest of facilitating the replication, extension, and application of our results, we release our code, the C4 data set, and pre-trained model weights for each T5 variant. ${ }^{1}$

### 4.2 Outlook

The inconvenience of large models An unsurprising but important result from our study is that larger models tend to perform better. The fact that the hardware used for running these models is continually getting cheaper and more powerful suggests that scaling up may continue to be a promising way to achieve better performance (Sutton, 2019). However, it will always be the case that there are applications and scenarios where using a smaller or less expensive model is helpful, for example when performing client-side inference or federated learning (Konečnỳ et al., 2015, 2016). Relatedly, one beneficial use of transfer learning is the possibility of attaining good performance on low-resource tasks. Low-resource tasks often occur (by definition) in settings where one lacks the assets to label more data. It follows that low-resource applications often also have limited access to computational resources which can incur additional costs. As a result, we advocate for research on methods that achieve stronger performance with cheaper models so that transfer learning can be applied where it will have the most impact. Some current work along these lines include distillation (Hinton et al., 2015; Sanh et al., 2019; Jiao et al., 2019), parameter sharing (Lan et al., 2019), and conditional computation (Shazeer et al., 2017).

More efficient knowledge extraction Recall that one of the goals of pre-training is (loosely speaking) to provide the model with general-purpose "knowledge" that improves its performance on downstream tasks. The method we use in this work, which is currently common practice, is to train the model to denoise corrupted spans of text. We suspect that this simplistic technique may not be a very efficient way to teach the model general-purpose knowledge. More concretely, it would be useful to be able to attain good fine-tuning performance without needing to train our models on 1 trillion tokens of text first. Some concurrent work along these lines improves efficiency by pre-training a model to distinguish between real and machine-generated text (Clark et al., 2020).

Formalizing the similarity between tasks We observed that pre-training on unlabeled in-domain data can improve performance on downstream tasks (Section 3.4). This finding mostly relies on basic observations like the fact that $S Q u A D$ was created using data from Wikipedia. It would be useful to formulate a more rigorous notion of the "similarity" between the pre-training and downstream tasks, so that we could make more principled choices about what source of unlabeled data to use. There is some early empirical work along these lines in the field of computer vision (Huh et al., 2016; Kornblith et al., 2018; He et al., 2018). A better notion of the relatedness of tasks could also help choose supervised pre-training tasks, which has been shown to be helpful for the GLUE benchmark (Phang et al., 2018).

Language-agnostic models We were disappointed to find that English-only pre-training did not achieve state-of-the-art results on the translation tasks we studied. We also are interested in avoiding the logistical difficulty of needing to specify which languages a vocabulary can encode ahead of time. To address these issues, we are interested in further investigating language-agnostic models, i.e. models that can perform a given NLP task with good performance regardless of the text's language. This is an especially
pertinent issue given that English is not the native language for the majority of the world's population.

The motivation for this paper was the flurry of recent work on transfer learning for NLP. Before we began this work, these advances had already enabled breakthroughs in settings where learning-based methods had not yet been shown to be effective. We are happy to be able to continue this trend, for example by nearly matching human-level performance on the SuperGLUE benchmark, a task specifically designed to be difficult for modern transfer-learning pipelines. Our results stem from the combination of a straightforward and unified text-to-text framework, our new C4 data set, and insights from our systematic study. Additionally, we provided an empirical overview of the field and a perspective on where it stands. We are excited to see continued work using transfer learning towards the goal of general language understanding.

## Acknowledgments

We thank Grady Simon, Noah Fiedel, Samuel R. Bowman, Augustus Odena, Daphne Ippolito, Noah Constant, Orhan Firat, Ankur Bapna, and Sebastian Ruder for their comments on this manuscript; Zak Stone and the TFRC team for their support; Austin Tarango for his guidance on data set creation; Melvin Johnson, Dima Lepikhin, Katrin Tomanek, Jeff Klingner, and Naveen Arivazhagan for insight into multi-task machine translation; Neil Houlsby for comments on adapter layers; Olga Wichowska, Ola Spyra, Michael Banfield, Yi Lin, and Frank Chen for assistance with infrastructure; Etienne Pot, Ryan Sepassi, and Pierre Ruyssen for collaboration on TensorFlow Datasets; Rohan Anil for help with our download pipeline for Common Crawl; Robby Neale and Taku Kudo for their work on SentencePiece; Jeffrey Li for pointing out missing details about the creation of C4; and many other members of the Google Brain team for their discussion and insight.

</end of paper 1>


<paper 2>
# ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS 

Zhenzhong Lan $^{1} \quad$ Mingda Chen $^{2 *} \quad$ Sebastian Goodman $^{1} \quad$ Kevin Gimpel $^{2}$<br>Piyush Sharma $^{1} \quad$ Radu Soricut $^{1}$<br>${ }^{1}$ Google Research $\quad{ }^{2}$ Toyota Technological Institute at Chicago<br>\{lanzhzh, seabass, piyushsharma, rsoricut\}@google.com<br>$\{m c h e n, k g i m p e l\} @ t t i c . e d u$


#### Abstract

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameterreduction techniques to lower memory consumption and increase the training speed of BERT (Devlin et al., 2019). Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large. The code and the pretrained models are available at https://github.com/google-research/ALBERT


## 1 INTRODUCTION

Full network pre-training (Dai \& Le, 2015, Radford et al., 2018; Devlin et al., 2019, Howard \& Ruder 2018) has led to a series of breakthroughs in language representation learning. Many nontrivial NLP tasks, including those that have limited training data, have greatly benefited from these pre-trained models. One of the most compelling signs of these breakthroughs is the evolution of machine performance on a reading comprehension task designed for middle and high-school English exams in China, the RACE test (Lai et al. 2017): the paper that originally describes the task and formulates the modeling challenge reports then state-of-the-art machine accuracy at $44.1 \%$; the latest published result reports their model performance at $83.2 \%$ (Liu et al. 2019); the work we present here pushes it even higher to $89.4 \%$, a stunning $45.3 \%$ improvement that is mainly attributable to our current ability to build high-performance pretrained language representations.

Evidence from these improvements reveals that a large network is of crucial importance for achieving state-of-the-art performance (Devlin et al. , 2019, Radford et al., 2019). It has become common practice to pre-train large models and distill them down to smaller ones (Sun et al., 2019, Turc et al. 2019) for real applications. Given the importance of model size, we ask: Is having better NLP models as easy as having larger models?

An obstacle to answering this question is the memory limitations of available hardware. Given that current state-of-the-art models often have hundreds of millions or even billions of parameters, it is easy to hit these limitations as we try to scale our models. Training speed can also be significantly hampered in distributed training, as the communication overhead is directly proportional to the number of parameters in the model.

Existing solutions to the aforementioned problems include model parallelization (Shazeer et al. 2018; Shoeybi et al. 2019) and clever memory management (Chen et al., 2016; Gomez et al.||2017).[^0]

These solutions address the memory limitation problem, but not the communication overhead. In this paper, we address all of the aforementioned problems, by designing A Lite BERT (ALBERT) architecture that has significantly fewer parameters than a traditional BERT architecture.

ALBERT incorporates two parameter reduction techniques that lift the major obstacles in scaling pre-trained models. The first one is a factorized embedding parameterization. By decomposing the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden layers from the size of vocabulary embedding. This separation makes it easier to grow the hidden size without significantly increasing the parameter size of the vocabulary embeddings. The second technique is cross-layer parameter sharing. This technique prevents the parameter from growing with the depth of the network. Both techniques significantly reduce the number of parameters for BERT without seriously hurting performance, thus improving parameter-efficiency. An ALBERT configuration similar to BERT-large has $18 \mathrm{x}$ fewer parameters and can be trained about $1.7 \mathrm{x}$ faster. The parameter reduction techniques also act as a form of regularization that stabilizes the training and helps with generalization.

To further improve the performance of ALBERT, we also introduce a self-supervised loss for sentence-order prediction (SOP). SOP primary focuses on inter-sentence coherence and is designed to address the ineffectiveness (Yang et al., 2019, Liu et al. 2019) of the next sentence prediction (NSP) loss proposed in the original BERT.

As a result of these design decisions, we are able to scale up to much larger ALBERT configurations that still have fewer parameters than BERT-large but achieve significantly better performance. We establish new state-of-the-art results on the well-known GLUE, SQuAD, and RACE benchmarks for natural language understanding. Specifically, we push the RACE accuracy to $89.4 \%$, the GLUE benchmark to 89.4, and the F1 score of SQuAD 2.0 to 92.2 .

## 2 RELATED WORK

### 2.1 SCALING Up REPRESENTATION LEARNING FOR NATURAL LANGUAGE

Learning representations of natural language has been shown to be useful for a wide range of NLP tasks and has been widely adopted (Mikolov et al., 2013, Le \& Mikolov, 2014; Dai \& Le, 2015, Peters et al., 2018; Devlin et al., 2019, Radford et al.,|2018; 2019). One of the most significant changes in the last two years is the shift from pre-training word embeddings, whether standard (Mikolov et al. 2013, Pennington et al., 2014) or contextualized (McCann et al., 2017, Peters et al.,| 2018), to full-network pre-training followed by task-specific fine-tuning (Dai \& Le, 2015; Radford et al. 2018; Devlin et al., 2019). In this line of work, it is often shown that larger model size improves performance. For example, Devlin et al. (2019) show that across three selected natural language understanding tasks, using larger hidden size, more hidden layers, and more attention heads always leads to better performance. However, they stop at a hidden size of 1024, presumably because of the model size and computation cost problems.

It is difficult to experiment with large models due to computational constraints, especially in terms of GPU/TPU memory limitations. Given that current state-of-the-art models often have hundreds of millions or even billions of parameters, we can easily hit memory limits. To address this issue, Chen et al. (2016) propose a method called gradient checkpointing to reduce the memory requirement to be sublinear at the cost of an extra forward pass. Gomez et al. (2017) propose a way to reconstruct each layer's activations from the next layer so that they do not need to store the intermediate activations. Both methods reduce the memory consumption at the cost of speed. Raffel et al. (2019) proposed to use model parallelization to train a giant model. In contrast, our parameter-reduction techniques reduce memory consumption and increase training speed.

### 2.2 CROSS-LAYER PARAMETER SHARING

The idea of sharing parameters across layers has been previously explored with the Transformer architecture (Vaswani et al. 2017), but this prior work has focused on training for standard encoderdecoder tasks rather than the pretraining/finetuning setting. Different from our observations, Dehghani et al. (2018) show that networks with cross-layer parameter sharing (Universal Transformer, UT) get better performance on language modeling and subject-verb agreement than the standard
transformer. Very recently, Bai et al. (2019) propose a Deep Equilibrium Model (DQE) for transformer networks and show that DQE can reach an equilibrium point for which the input embedding and the output embedding of a certain layer stay the same. Our observations show that our embeddings are oscillating rather than converging. Hao et al. (2019) combine a parameter-sharing transformer with the standard one, which further increases the number of parameters of the standard transformer.

### 2.3 SENTENCE Ordering ObJECTIVES

ALBERT uses a pretraining loss based on predicting the ordering of two consecutive segments of text. Several researchers have experimented with pretraining objectives that similarly relate to discourse coherence. Coherence and cohesion in discourse have been widely studied and many phenomena have been identified that connect neighboring text segments (Hobbs, 1979, Halliday \& Hasan 1976, Grosz et al., 1995). Most objectives found effective in practice are quite simple. Skipthought (Kiros et al.| 2015) and FastSent (Hill et al., 2016) sentence embeddings are learned by using an encoding of a sentence to predict words in neighboring sentences. Other objectives for sentence embedding learning include predicting future sentences rather than only neighbors (Gan et al. 2017) and predicting explicit discourse markers (Jernite et al. , 2017, Nie et al., 2019). Our loss is most similar to the sentence ordering objective of Jernite et al. (2017), where sentence embeddings are learned in order to determine the ordering of two consecutive sentences. Unlike most of the above work, however, our loss is defined on textual segments rather than sentences. BERT (Devlin et al. 2019) uses a loss based on predicting whether the second segment in a pair has been swapped with a segment from another document. We compare to this loss in our experiments and find that sentence ordering is a more challenging pretraining task and more useful for certain downstream tasks. Concurrently to our work, Wang et al. (2019) also try to predict the order of two consecutive segments of text, but they combine it with the original next sentence prediction in a three-way classification task rather than empirically comparing the two.

## 3 THE ELEMENTS OF ALBERT

In this section, we present the design decisions for ALBERT and provide quantified comparisons against corresponding configurations of the original BERT architecture (Devlin et al., 2019).

### 3.1 MODEL ARCHITECTURE CHOICES

The backbone of the ALBERT architecture is similar to BERT in that it uses a transformer encoder (Vaswani et al. 2017) with GELU nonlinearities (Hendrycks \& Gimpel 2016). We follow the BERT notation conventions and denote the vocabulary embedding size as $E$, the number of encoder layers as $L$, and the hidden size as $H$. Following Devlin et al. (2019), we set the feed-forward/filter size to be $4 H$ and the number of attention heads to be $H / 64$.

There are three main contributions that ALBERT makes over the design choices of BERT.

Factorized embedding parameterization. In BERT, as well as subsequent modeling improvements such as XLNet (Yang et al., 2019) and RoBERTa (Liu et al., 2019), the WordPiece embedding size $E$ is tied with the hidden layer size $H$, i.e., $E \equiv H$. This decision appears suboptimal for both modeling and practical reasons, as follows.

From a modeling perspective, WordPiece embeddings are meant to learn context-independent representations, whereas hidden-layer embeddings are meant to learn context-dependent representations. As experiments with context length indicate (Liu et al. 2019), the power of BERT-like representations comes from the use of context to provide the signal for learning such context-dependent representations. As such, untying the WordPiece embedding size $E$ from the hidden layer size $H$ allows us to make a more efficient usage of the total model parameters as informed by modeling needs, which dictate that $H \gg E$.

From a practical perspective, natural language processing usually require the vocabulary size $V$ to be large ${ }^{1}$ If $E \equiv H$, then increasing $H$ increases the size of the embedding matrix, which has size[^1]$V \times E$. This can easily result in a model with billions of parameters, most of which are only updated sparsely during training.

Therefore, for ALBERT we use a factorization of the embedding parameters, decomposing them into two smaller matrices. Instead of projecting the one-hot vectors directly into the hidden space of size $H$, we first project them into a lower dimensional embedding space of size $E$, and then project it to the hidden space. By using this decomposition, we reduce the embedding parameters from $O(V \times H)$ to $O(V \times E+E \times H)$. This parameter reduction is significant when $H \gg E$. We choose to use the same $\mathrm{E}$ for all word pieces because they are much more evenly distributed across documents compared to whole-word embedding, where having different embedding size (Grave et al. (2017); Baevski \& Auli (2018); Dai et al. (2019) ) for different words is important.

Cross-layer parameter sharing. For ALBERT, we propose cross-layer parameter sharing as another way to improve parameter efficiency. There are multiple ways to share parameters, e.g., only sharing feed-forward network (FFN) parameters across layers, or only sharing attention parameters. The default decision for ALBERT is to share all parameters across layers. All our experiments use this default decision unless otherwise specified. We compare this design decision against other strategies in our experiments in Sec. 4.5

Similar strategies have been explored by Dehghani et al. (2018) (Universal Transformer, UT) and Bai et al. (2019) (Deep Equilibrium Models, DQE) for Transformer networks. Different from our observations, Dehghani et al. (2018) show that UT outperforms a vanilla Transformer. Bai et al. (2019) show that their DQEs reach an equilibrium point for which the input and output embedding of a certain layer stay the same. Our measurement on the L2 distances and cosine similarity show that our embeddings are oscillating rather than converging.
![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-04.jpg?height=426&width=1266&top_left_y=1229&top_left_x=424)

Figure 1: The L2 distances and cosine similarity (in terms of degree) of the input and output embedding of each layer for BERT-large and ALBERT-large.

Figure 1 shows the L2 distances and cosine similarity of the input and output embeddings for each layer, using BERT-large and ALBERT-large configurations (see Table 1). We observe that the transitions from layer to layer are much smoother for ALBERT than for BERT. These results show that weight-sharing has an effect on stabilizing network parameters. Although there is a drop for both metrics compared to BERT, they nevertheless do not converge to 0 even after 24 layers. This shows that the solution space for ALBERT parameters is very different from the one found by DQE.

Inter-sentence coherence loss. In addition to the masked language modeling (MLM) loss (Devlin et al., 2019), BERT uses an additional loss called next-sentence prediction (NSP). NSP is a binary classification loss for predicting whether two segments appear consecutively in the original text, as follows: positive examples are created by taking consecutive segments from the training corpus; negative examples are created by pairing segments from different documents; positive and negative examples are sampled with equal probability. The NSP objective was designed to improve performance on downstream tasks, such as natural language inference, that require reasoning about the relationship between sentence pairs. However, subsequent studies (Yang et al., 2019; Liu et al. 2019) found NSP's impact unreliable and decided to eliminate it, a decision supported by an improvement in downstream task performance across several tasks.

We conjecture that the main reason behind NSP's ineffectiveness is its lack of difficulty as a task, as compared to MLM. As formulated, NSP conflates topic prediction and coherence prediction in a

| Model |  | Parameters | Layers | Hidden | Embedding | Parameter-sharing |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | base | $108 \mathrm{M}$ | 12 | 768 | 768 | False |
| BERT | large | $334 \mathrm{M}$ | 24 | 1024 | 1024 | False |
| ALBERT | base | $12 \mathrm{M}$ | 12 | 768 | 128 | True |
|  | large | $18 \mathrm{M}$ | 24 | 1024 | 128 | True |
|  | xlarge | $60 \mathrm{M}$ | 24 | 2048 | 128 | True |
|  | xxlarge | $235 \mathrm{M}$ | 12 | 4096 | 128 | True |

Table 1: The configurations of the main BERT and ALBERT models analyzed in this paper.

single $\operatorname{tas} \mathrm{K}^{2}$. However, topic prediction is easier to learn compared to coherence prediction, and also overlaps more with what is learned using the MLM loss.

We maintain that inter-sentence modeling is an important aspect of language understanding, but we propose a loss based primarily on coherence. That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which avoids topic prediction and instead focuses on modeling inter-sentence coherence. The SOP loss uses as positive examples the same technique as BERT (two consecutive segments from the same document), and as negative examples the same two consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence properties. As we show in Sec. 4.6 , it turns out that NSP cannot solve the SOP task at all (i.e., it ends up learning the easier topic-prediction signal, and performs at randombaseline level on the SOP task), while SOP can solve the NSP task to a reasonable degree, presumably based on analyzing misaligned coherence cues. As a result, ALBERT models consistently improve downstream task performance for multi-sentence encoding tasks.

### 3.2 MODEL SETUP

We present the differences between BERT and ALBERT models with comparable hyperparameter settings in Table 1] Due to the design choices discussed above, ALBERT models have much smaller parameter size compared to corresponding BERT models.

For example, ALBERT-large has about $18 \mathrm{x}$ fewer parameters compared to BERT-large, $18 \mathrm{M}$ versus 334 M. An ALBERT-xlarge configuration with $H=2048$ has only $60 \mathrm{M}$ parameters and an ALBERT-xxlarge configuration with $H=4096$ has $233 \mathrm{M}$ parameters, i.e., around $70 \%$ of BERTlarge's parameters. Note that for ALBERT-xxlarge, we mainly report results on a 12-layer network because a 24-layer network (with the same configuration) obtains similar results but is computationally more expensive.

This improvement in parameter efficiency is the most important advantage of ALBERT's design choices. Before we can quantify this advantage, we need to introduce our experimental setup in more detail.

## 4 EXPERIMENTAL RESULTS

### 4.1 EXPERIMENTAL SETUP

To keep the comparison as meaningful as possible, we follow the BERT (Devlin et al. , 2019) setup in using the BоокCorpus (Zhu et al. 2015) and English Wikipedia (Devlin et al.| 2019) for pretraining baseline models. These two corpora consist of around $16 \mathrm{~GB}$ of uncompressed text. We format our inputs as "[CLS] $x_{1}[\mathrm{SEP}] x_{2}[\mathrm{SEP}]$ ", where $x_{1}=x_{1,1}, x_{1,2} \cdots$ and $x_{2}=x_{1,1}, x_{1,2} \cdots$ are two segments 3 We always limit the maximum input length to 512 , and randomly generate input sequences shorter than 512 with a probability of $10 \%$. Like BERT, we use a vocabulary size of 30,000, tokenized using SentencePiece (Kudo \& Richardson, 2018) as in XLNet (Yang et al., 2019).[^2]

We generate masked inputs for the MLM targets using $n$-gram masking (Joshi et al., 2019), with the length of each $n$-gram mask selected randomly. The probability for the length $n$ is given by

$$
p(n)=\frac{1 / n}{\sum_{k=1}^{N} 1 / k}
$$

We set the maximum length of $n$-gram (i.e., $n$ ) to be 3 (i.e., the MLM target can consist of up to a 3 -gram of complete words, such as "White House correspondents").

All the model updates use a batch size of 4096 and a LAMB optimizer with learning rate 0.00176 (You et al. 2019). We train all models for 125,000 steps unless otherwise specified. Training was done on Cloud TPU V3. The number of TPUs used for training ranged from 64 to 512, depending on model size.

The experimental setup described in this section is used for all of our own versions of BERT as well as ALBERT models, unless otherwise specified.

### 4.2 EVALUATION BENCHMARKS

### 4.2.1 INTRINSIC EVALUATION

To monitor the training progress, we create a development set based on the development sets from SQuAD and RACE using the same procedure as in Sec.4.1 We report accuracies for both MLM and sentence classification tasks. Note that we only use this set to check how the model is converging; it has not been used in a way that would affect the performance of any downstream evaluation, such as via model selection.

### 4.2.2 DOWNSTREAM EVALUATION

Following Yang et al. (2019) and Liu et al. (2019), we evaluate our models on three popular benchmarks: The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018), two versions of the Stanford Question Answering Dataset (SQuAD; Rajpurkar et al.,|2016; 2018), and the ReAding Comprehension from Examinations (RACE) dataset (Lai et al. | 2017). For completeness, we provide description of these benchmarks in Appendix A.3 As in (Liu et al. 2019), we perform early stopping on the development sets, on which we report all comparisons except for our final comparisons based on the task leaderboards, for which we also report test set results. For GLUE datasets that have large variances on the dev set, we report median over 5 runs.

### 4.3 OvERALL COMPARISON BETWEEN BERT AND ALBERT

We are now ready to quantify the impact of the design choices described in Sec. 3. specifically the ones around parameter efficiency. The improvement in parameter efficiency showcases the most important advantage of ALBERT's design choices, as shown in Table 2 with only around $70 \%$ of BERT-large's parameters, ALBERT-xxlarge achieves significant improvements over BERT-large, as measured by the difference on development set scores for several representative downstream tasks: SQuAD v1.1 (+1.9\%), SQuAD v2.0 (+3.1\%), MNLI (+1.4\%), SST-2 (+2.2\%), and RACE (+8.4\%).

Another interesting observation is the speed of data throughput at training time under the same training configuration (same number of TPUs). Because of less communication and fewer computations, ALBERT models have higher data throughput compared to their corresponding BERT models. If we use BERT-large as the baseline, we observe that ALBERT-large is about 1.7 times faster in iterating through the data while ALBERT-xxlarge is about 3 times slower because of the larger structure.

Next, we perform ablation experiments that quantify the individual contribution of each of the design choices for ALBERT.

![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-06.jpg?height=41&width=783&top_left_y=2354&top_left_x=373)

Table 3 shows the effect of changing the vocabulary embedding size $E$ using an ALBERT-base configuration setting (see Table 11, using the same set of representative downstream tasks. Under the non-shared condition (BERT-style), larger embedding sizes give better performance, but not by

| Model |  | Parameters | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg | Speedup |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| berT | base | 108M | $90.4 / 83.2$ | $80.4 / 77.6$ | 84.5 | 92.8 | 68.2 | 82.3 | $4.7 \mathrm{x}$ |
|  | large | 334M | $92.2 / 85.5$ | $85.0 / 82.2$ | 86.6 | 93.0 | 73.9 | 85.2 | 1.0 |
| ALBERT | base | 12M | $89.3 / 82.3$ | $80.0 / 77.1$ | 81.6 | 90.3 | 64.0 | 80.1 | $5.6 \mathrm{x}$ |
|  | large | 18M | $90.6 / 83.9$ | $82.3 / 79.4$ | 83.5 | 91.7 | 68.5 | 82.4 | $1.7 \mathrm{x}$ |
|  | xlarge | 60M | $92.5 / 86.1$ | $86.1 / 93.1$ | 86.4 | 92.4 | 74.8 | 85.5 | $0.6 \mathrm{x}$ |
|  | xxlarge | $235 \mathrm{M}$ | $\mathbf{9 4 . 1 / 8 8 . 3}$ | $\mathbf{8 8 . 1 / 8 5 . 1}$ | $\mathbf{8 8 . 0}$ | $\mathbf{9 5 . 2}$ | $\mathbf{8 2 . 3}$ | $\mathbf{8 8 . 7}$ | $0.3 \mathrm{x}$ |

Table 2: Dev set results for models pretrained over BоокCorpus and Wikipedia for $125 \mathrm{k}$ steps. Here and everywhere else, the Avg column is computed by averaging the scores of the downstream tasks to its left (the two numbers of F1 and EM for each SQuAD are first averaged).

much. Under the all-shared condition (ALBERT-style), an embedding of size 128 appears to be the best. Based on these results, we use an embedding size $E=128$ in all future settings, as a necessary step to do further scaling.

| Model | $E$ | Parameters | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ALBERT | 64 | $87 \mathrm{M}$ | 89.99822 .9 | $80.1 / 77.8$ | 82.9 | 91.5 | 66.7 | 81.3 |
|  | 128 | $89 \mathrm{M}$ | $89.9 / 82.8$ | $80.3 / 77.3$ | 83.7 | 91.5 | 67.9 | 81.7 |
| not-shared | 256 | $93 \mathrm{M}$ | $90.2 / 83.2$ | $80.3 / 77.4$ | 84.1 | 91.9 | 67.3 | 81.8 |
|  | 768 | $108 \mathrm{M}$ | $90.4 / 83.2$ | $80.4 / 77.6$ | 84.5 | 92.8 | 68.2 | 82.3 |
| ALBERT | 64 | $10 \mathrm{M}$ | $88.7 / 81.4$ | $77.5 / 74.8$ | 80.8 | 89.4 | 63.5 | 79.0 |
|  | 128 | $12 \mathrm{M}$ | 89.3382 .3 | $80.0 / 77.1$ | 81.6 | 90.3 | 64.0 | 80.1 |
| all-shared | 256 | $16 \mathrm{M}$ | 88.88 .81 .5 | $79.1 / 76.3$ | 81.5 | 90.3 | 63.4 | 79.6 |
|  | 768 | $31 \mathrm{M}$ | $88.6 / 81.5$ | $79.2 / 76.6$ | 82.0 | 90.6 | 63.3 | 79.8 |

Table 3: The effect of vocabulary embedding size on the performance of ALBERT-base.

### 4.5 CROSS-LAYER PARAMETER SHARING

Table 4 presents experiments for various cross-layer parameter-sharing strategies, using an ALBERT-base configuration (Table 1) with two embedding sizes ( $E=768$ and $E=128$ ). We compare the all-shared strategy (ALBERT-style), the not-shared strategy (BERT-style), and intermediate strategies in which only the attention parameters are shared (but not the FNN ones) or only the FFN parameters are shared (but not the attention ones).

The all-shared strategy hurts performance under both conditions, but it is less severe for $E=128$ (1.5 on Avg) compared to $E=768$ (-2.5 on Avg). In addition, most of the performance drop appears to come from sharing the FFN-layer parameters, while sharing the attention parameters results in no drop when $E=128$ (+0.1 on Avg), and a slight drop when $E=768$ (-0.7 on Avg).

There are other strategies of sharing the parameters cross layers. For example, We can divide the $L$ layers into $N$ groups of size $M$, and each size- $M$ group shares parameters. Overall, our experimental results shows that the smaller the group size $M$ is, the better the performance we get. However, decreasing group size $M$ also dramatically increase the number of overall parameters. We choose all-shared strategy as our default choice.

| Model |  | Parameters | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ALBERT | all-shared | 31M | $88.6 / 81.5$ | $79.2 / 76.6$ | 82.0 | 90.6 | 63.3 | 79.8 |
|  | shared-attention | $83 \mathrm{M}$ | $89.9 / 82.7$ | $80.0 / 77.2$ | 84.0 | 91.4 | 67.7 | 81.6 |
| $E=768$ | shared-FFN | 57M | $89.2 / 82.1$ | $78.2 / 75.4$ | 81.5 | 90.8 | 62.6 | 79.5 |
|  | not-shared | $108 \mathrm{M}$ | $90.4 / 83.2$ | $80.4 / 77.6$ | 84.5 | 92.8 | 68.2 | 82.3 |
| ALBERT | all-shared | $12 \mathrm{M}$ | $89.3 / 82.3$ | $80.0 / 77.1$ | 82.0 | 90.3 | 64.0 | 80.1 |
|  | shared-attention | $64 \mathrm{M}$ | $89.9 / 82.8$ | $80.7 / 77.9$ | 83.4 | 91.9 | 67.6 | 81.7 |
| $E=128$ | shared-FFN | $38 \mathrm{M}$ | $88.9 / 1.6$ | $78.6 / 75.6$ | 82.3 | 91.7 | 64.4 | 80.2 |
|  | not-shared | $89 \mathrm{M}$ | $89.9 / 82.8$ | $80.3 / 77.3$ | 83.2 | 91.5 | 67.9 | 81.6 |

Table 4: The effect of cross-layer parameter-sharing strategies, ALBERT-base configuration.

### 4.6 SENTENCE ORDER PREDICTION (SOP)

We compare head-to-head three experimental conditions for the additional inter-sentence loss: none (XLNet- and RoBERTa-style), NSP (BERT-style), and SOP (ALBERT-style), using an ALBERTbase configuration. Results are shown in Table 5, both over intrinsic (accuracy for the MLM, NSP, and SOP tasks) and downstream tasks.

|  | Intrinsic Tasks |  |  |  | Downstream Tasks |  |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SP tasks | MLM | NSP | SOP | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |  |  |
| None | 54.9 | 52.4 | 53.3 | $88.6 / 81.5$ | $78.1 / 75.3$ | 81.5 | 89.9 | 61.7 | 79.0 |  |  |
| NSP | 54.5 | 90.5 | 52.0 | $88.4 / 81.5$ | $77.2 / 74.6$ | 81.6 | $\mathbf{9 1 . 1}$ | 62.3 | 79.2 |  |  |
| SOP | 54.0 | 78.9 | 86.5 | $\mathbf{8 9 . 3 / 8 2 . 3}$ | $\mathbf{8 0 . 0 / 7 7 . 1}$ | $\mathbf{8 2 . 0}$ | 90.3 | $\mathbf{6 4 . 0}$ | $\mathbf{8 0 . 1}$ |  |  |

Table 5: The effect of sentence-prediction loss, NSP vs. SOP, on intrinsic and downstream tasks.

The results on the intrinsic tasks reveal that the NSP loss brings no discriminative power to the SOP task $(52.0 \%$ accuracy, similar to the random-guess performance for the "None" condition). This allows us to conclude that NSP ends up modeling only topic shift. In contrast, the SOP loss does solve the NSP task relatively well ( $78.9 \%$ accuracy), and the SOP task even better ( $86.5 \%$ accuracy). Even more importantly, the SOP loss appears to consistently improve downstream task performance for multi-sentence encoding tasks (around $+1 \%$ for SQuAD1.1, $+2 \%$ for SQuAD2.0, $+1.7 \%$ for RACE), for an Avg score improvement of around $+1 \%$.

### 4.7 WHAT IF WE TRAIN FOR THE SAME AMOUNT OF TIME?

The speed-up results in Table 2 indicate that data-throughput for BERT-large is about 3.17x higher compared to ALBERT-xxlarge. Since longer training usually leads to better performance, we perform a comparison in which, instead of controlling for data throughput (number of training steps), we control for the actual training time (i.e., let the models train for the same number of hours). In Table 6. we compare the performance of a BERT-large model after 400k training steps (after 34h of training), roughly equivalent with the amount of time needed to train an ALBERT-xxlarge model with $125 \mathrm{k}$ training steps ( $32 \mathrm{~h}$ of training).

| Models | Steps | Time | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT-large | $400 \mathrm{k}$ | 34h | $93.5 / 87.4$ | $86.9 / 84.3$ | 87.8 | 94.6 | 77.3 | 87.2 |
| ALBERT-xxlarge | $125 \mathrm{k}$ | $32 \mathrm{~h}$ | $\mathbf{9 4 . 0 / 8 8 . 1}$ | $\mathbf{8 8 . 3 / 8 5 . 3}$ | 87.8 | $\mathbf{9 5 . 4}$ | $\mathbf{8 2 . 5}$ | $\mathbf{8 8 . 7}$ |

Table 6: The effect of controlling for training time, BERT-large vs ALBERT-xxlarge configurations.

After training for roughly the same amount of time, ALBERT-xxlarge is significantly better than BERT-large: $+1.5 \%$ better on Avg, with the difference on RACE as high as $+5.2 \%$.

### 4.8 ADDITIONAL TRAINING DATA AND DROPOUT EFFECTS

The experiments done up to this point use only the Wikipedia and BоокCORpus datasets, as in (Devlin et al., 2019). In this section, we report measurements on the impact of the additional data used by both XLNet (Yang et al., 2019) and RoBERTa (Liu et al., 2019).

Fig. 2a plots the dev set MLM accuracy under two conditions, without and with additional data, with the latter condition giving a significant boost. We also observe performance improvements on the downstream tasks in Table 7, except for the SQuAD benchmarks (which are Wikipedia-based, and therefore are negatively affected by out-of-domain training material).

|  | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No additional data | $\mathbf{8 9 . 3 / 8 2 . 3}$ | $\mathbf{8 0 . 0 / 7 7 . 1}$ | 81.6 | 90.3 | 64.0 | 80.1 |
| With additional data | $88.8 / 81.7$ | $79.1 / 76.3$ | $\mathbf{8 2 . 4}$ | $\mathbf{9 2 . 8}$ | $\mathbf{6 6 . 0}$ | $\mathbf{8 0 . 8}$ |

Table 7: The effect of additional training data using the ALBERT-base configuration.

We also note that, even after training for $1 \mathrm{M}$ steps, our largest models still do not overfit to their training data. As a result, we decide to remove dropout to further increase our model capacity. The

![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=475&width=1266&top_left_y=264&top_left_x=424)

![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=390&width=567&top_left_y=279&top_left_x=432)

(a) Adding data

![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=393&width=569&top_left_y=275&top_left_x=1122)

(b) Removing dropout

Figure 2: The effects of adding data and removing dropout during training.

plot in Fig. 2b shows that removing dropout significantly improves MLM accuracy. Intermediate evaluation on ALBERT-xxlarge at around $1 \mathrm{M}$ training steps (Table 8 ) also confirms that removing dropout helps the downstream tasks. There is empirical (Szegedy et al., 2017) and theoretical (Li et al. 2019) evidence showing that a combination of batch normalization and dropout in Convolutional Neural Networks may have harmful results. To the best of our knowledge, we are the first to show that dropout can hurt performance in large Transformer-based models. However, the underlying network structure of ALBERT is a special case of the transformer and further experimentation is needed to see if this phenomenon appears with other transformer-based architectures or not.

|  | SQuAD1.1 | SQuAD2.0 | MNLI | SST-2 | RACE | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| With dropout | $94.7 / 89.2$ | $89.6 / 86.9$ | 90.0 | 96.3 | 85.7 | 90.4 |
| Without dropout | $\mathbf{9 4 . 8 / 8 9 . 5}$ | $\mathbf{8 9 . 9 / 8 7 . 2}$ | $\mathbf{9 0 . 4}$ | $\mathbf{9 6 . 5}$ | $\mathbf{8 6 . 1}$ | $\mathbf{9 0 . 7}$ |

Table 8: The effect of removing dropout, measured for an ALBERT-xxlarge configuration.

### 4.9 CURRENT StATE-OF-THE-ART ON NLU TASKS

The results we report in this section make use of the training data used by Devlin et al. (2019), as well as the additional data used by Liu et al. (2019) and Yang et al. (2019). We report state-of-the-art results under two settings for fine-tuning: single-model and ensembles. In both settings, we only do single-task fine-tuning ${ }^{4}$ Following Liu et al. (2019), on the development set we report the median result over five runs.

| Models | MNLI | QNLI | QQP | RTE | SST | MRPC | CoLA | STS | WNLI | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-task single models on dev |  |  |  |  |  |  |  |  |  |  |
| BERT-large | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 | - | - |
| XLNet-large | 89.8 | 93.9 | 018 | 83.8 | ![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=43&width=84&top_left_y=1817&top_left_x=1090) | 807 | 63.6 |  | - | - |
| RoBE | ![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=42&width=106&top_left_y=1852&top_left_x=651) | 94.7 | 92.2 | 86.6 | 96.2 | 90.9 | 68.0 | 4 | - | - |
| LB] | ![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=43&width=106&top_left_y=1886&top_left_x=651) | 95.2 | 92.0 | 88.1 | ![](https://cdn.mathpix.com/cropped/2024_05_26_ccf341feaf3ac183e55ag-09.jpg?height=43&width=84&top_left_y=1886&top_left_x=1090) | 90.2 | 68.7 | 92.7 | - | - |
| ALBERT (1.5M) | 90.8 | 95.3 | 92.2 | 89.2 | 96.9 | 90. | 71.4 | 93.0 |  | - |
| Ensembles on test (from leaderboard as of Sept. 16, 2019) |  |  |  |  |  |  |  |  |  |  |
| ALICE | 88.2 | 95.7 | 90.7 | 83.5 | 95.2 | 92. | 69. | 91 | 80 | 87.0 |
|  | 87.9 | 96. | 80 | 86 | 96. |  | 0 |  | 89 | 87.6 |
| XLN | 90.2 | 98.6 | 90. | 86.3 | 96.8 | 93. | 67.8 | 91.6 | 90 | 88.4 |
| RoB | on | 98. | 90. | 88. | 96. | 92. | 67.8 | 92.2 | 89.0 | 88.5 |
| Adv-RoBE | 91.1 | 98.8 | 90.3 | 88.7 | 96.8 | 93.1 | 68.0 | 92.4 | 89.0 | 88.8 |
|  | 12 | 99.2 |  | 802 | 97.1 | 93.4 | 69.1 | 92.5 | 91.8 | 89.4 |

Table 9: State-of-the-art results on the GLUE benchmark. For single-task single-model results, we report ALBERT at $1 \mathrm{M}$ steps (comparable to RoBERTa) and at $1.5 \mathrm{M}$ steps. The ALBERT ensemble uses models trained with $1 \mathrm{M}, 1.5 \mathrm{M}$, and other numbers of steps.

The single-model ALBERT configuration incorporates the best-performing settings discussed: an ALBERT-xxlarge configuration (Table 1) using combined MLM and SOP losses, and no dropout.[^3]

The checkpoints that contribute to the final ensemble model are selected based on development set performance; the number of checkpoints considered for this selection range from 6 to 17, depending on the task. For the GLUE (Table 9) and RACE (Table 10) benchmarks, we average the model predictions for the ensemble models, where the candidates are fine-tuned from different training steps using the 12-layer and 24-layer architectures. For SQuAD (Table 10), we average the prediction scores for those spans that have multiple probabilities; we also average the scores of the "unanswerable" decision.

Both single-model and ensemble results indicate that ALBERT improves the state-of-the-art significantly for all three benchmarks, achieving a GLUE score of 89.4, a SQuAD 2.0 test F1 score of 92.2, and a RACE test accuracy of 89.4. The latter appears to be a particularly strong improvement, a jump of $+17.4 \%$ absolute points over BERT (Devlin et al. , 2019, Clark et al., 2019) , $+7.6 \%$ over XLNet (Yang et al., 2019) $+6.2 \%$ over RoBERTa (Liu et al. 2019), and 5.3\% over DCMI+ (Zhang et al., 2019), an ensemble of multiple models specifically designed for reading comprehension tasks. Our single model achieves an accuracy of $86.5 \%$, which is still $2.4 \%$ better than the state-of-the-art ensemble model.

| Models | SQuAD1.1 dev | SQuAD2.0 dev | SQuAD2.0 test | RACE test (Middle/High) |
| :--- | :---: | :---: | :---: | :---: |
| Single model (from leaderboard as of Sept. 23, | 2019) |  |  |  |
| BERT-large | $90.9 / 84.1$ | $81.8 / 79.0$ | $89.1 / 86.3$ | $72.0(76.6 / 70.1)$ |
| XLNet | $94.5 / 89.0$ | $88.8 / 86.1$ | $89.1 / 86.3$ | $81.8(85.5 / 80.2)$ |
| RoBERTa | $94.6 / 88.9$ | $89.4 / 86.5$ | $89.8 / 86.8$ | 83.2 (86.5/81.3) |
| UPM | - | - | $89.9 / 87.2$ | - |
| XLNet + SG-Net Verifier++ | - | - | $90.1 / 87.2$ | - |
| ALBERT (1M) | $94.8 / 89.2$ | $89.9 / 87.2$ | - | $86.0(88.2 / 85.1)$ |
| ALBERT (1.5M) | $\mathbf{9 4 . 8 / 8 9 . 3}$ | $\mathbf{9 0 . 2 / 8 7 . 4}$ | $\mathbf{9 0 . 9 / 8 8 . 1}$ | $\mathbf{8 6 . 5}(\mathbf{8 9 . 0 / 8 5 . 5 )}$ |
| Ensembles (from leaderboard as of Sept. 23, 2019) |  |  |  |  |
| BERT-large | $92.2 / 86.2$ | - | - | - |
| XLNet + SG-Net Verifier | - | - | $90.7 / 88.2$ | - |
| UPM | - | - | $90.7 / 88.2$ | - |
| XLNet + DAAF + Verifier | - | - | $90.9 / 88.6$ | - |
| DCMN+ | - | - | - | $84.1(88.5 / 82.3)$ |
| ALBERT | $\mathbf{9 5 . 5 / 9 0 . 1}$ | $\mathbf{9 1 . 4 / 8 8 . 9}$ | $\mathbf{9 2 . 2 / 8 9 . 7}$ | $\mathbf{8 9 . 4}(\mathbf{9 1 . 2 / 8 8 . 6 )}$ |

Table 10: State-of-the-art results on the SQuAD and RACE benchmarks.

## 5 DISCUSSION

While ALBERT-xxlarge has less parameters than BERT-large and gets significantly better results, it is computationally more expensive due to its larger structure. An important next step is thus to speed up the training and inference speed of ALBERT through methods like sparse attention (Child et al. 2019) and block attention (Shen et al. 2018). An orthogonal line of research, which could provide additional representation power, includes hard example mining (Mikolov et al., 2013) and more efficient language modeling training (Yang et al. 2019). Additionally, although we have convincing evidence that sentence order prediction is a more consistently-useful learning task that leads to better language representations, we hypothesize that there could be more dimensions not yet captured by the current self-supervised training losses that could create additional representation power for the resulting representations.

## REFERENCES

Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. arXiv preprint arXiv:1809.10853, 2018.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Neural Information Processing Systems (NeurIPS), 2019.

Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. In Proceedings of the second PASCAL challenges workshop on recognising textual entailment, volume 6, pp. 6-4. Venice, 2006.

Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth PASCAL recognizing textual entailment challenge. In $T A C, 2009$.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 1-14, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/S17-2001. URLhttps://www.aclweb.org/anthology/S17-2001.

Tianqi Chen, Bing $\mathrm{Xu}$, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D Manning, and Quoc V Le. Bam! born-again multi-task networks for natural language understanding. arXiv preprint arXiv:1907.04829, 2019.

Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges Workshop, pp. 177-190. Springer, 2005.

Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. In Advances in neural information processing systems, pp. 3079-3087, 2015.

Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: $10.18653 / \mathrm{v} 1 / \mathrm{N} 19-1423$. URL https: //www.aclweb.org/anthology/N19-1423.

William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005. URL https://www.aclweb.org/anthology/I05-5002.

Zhe Gan, Yunchen Pu, Ricardo Henao, Chunyuan Li, Xiaodong He, and Lawrence Carin. Learning generic sentence representations using convolutional neural networks. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2390-2400, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1254. URLhttps://www.aclweb.org/anthology/D17-1254.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, pp. 1-9, Prague, June 2007. Association for Computational Linguistics. URLhttps://www.aclweb.org/anthology/W07-1401.

Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. In Advances in neural information processing systems, pp. 2214-2224, 2017.

Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of bert by progressively stacking. In International Conference on Machine Learning, pp. 2337-2346, 2019.

Edouard Grave, Armand Joulin, Moustapha Cissé, Hervé Jégou, et al. Efficient softmax approximation for gpus. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1302-1310. JMLR. org, 2017.

Barbara J. Grosz, Aravind K. Joshi, and Scott Weinstein. Centering: A framework for modeling the local coherence of discourse. Computational Linguistics, 21(2):203-225, 1995. URL https: ///www.aclweb.org/anthology/J95-2003.

M.A.K. Halliday and Ruqaiya Hasan. Cohesion in English. Routledge, 1976.

Jie Hao, Xing Wang, Baosong Yang, Longyue Wang, Jinfeng Zhang, and Zhaopeng Tu. Modeling recurrence for transformer. Proceedings of the 2019 Conference of the North, 2019. doi: 10. 18653/v1/n19-1122. URLhttp://dx.doi.org/10.18653/v1/n19-1122

Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415, 2016.

Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1367-1377. Association for Computational Linguistics, 2016. doi: 10.18653/v1/N16-1162. URL http: //aclweb.org/anthology/N16-1162

Jerry R. Hobbs. Coherence and coreference. Cognitive Science, 3(1):67-90, 1979.

Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.

Shankar Iyer, Nikhil Dandekar, and Kornl Csernai. First quora dataset release: Question pairs, January 2017. URL https://www.quora.com/q/quoradata/ First-Quora-Dataset-Release-Question-Pairs.

Yacine Jernite, Samuel R Bowman, and David Sontag. Discourse-based objectives for fast unsupervised sentence representation learning. arXiv preprint arXiv:1705.00557, 2017.

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. SpanBERT: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529, 2019.

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Skip-thought vectors. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, NIPS'15, pp. 3294-3302, Cambridge, MA, USA, 2015. MIT Press. URL/http://dl.acm.org/citation.cfm?id= 2969442.2969607 .

Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66-71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URLhttps://www.aclweb.org/anthology/D18-2012.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785-794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URLhttps://www. aclweb.org/anthology/D17-1082

Quoc Le and Tomas Mikolov. Distributed representations of sentences and documents. In Proceedings of the 31st ICML, Beijing, China, 2014.

Hector Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012.

Xiang Li, Shuo Chen, Xiaolin Hu, and Jian Yang. Understanding the disharmony between dropout and batch normalization by variance shift. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2682-2690, 2019.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 6294-6305. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/ 7209-learned-in-translation-contextualized-word-vectors.pdf

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111-3119, 2013.

Allen Nie, Erin Bennett, and Noah Goodman. DisSent: Learning sentence representations from explicit discourse relations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4497-4510, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1442. URLhttps://www.aclweb.org/anthology/ P19-1442

Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1162. URL/https://www.aclweb.org/anthology/ D14-1162

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 2227-2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1202. URL https://www.aclweb.org/anthology/N18-1202.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. https://s3-us-west-2.amazonaws.com/ openai-assets/research-covers/language-unsupervised/language_ understanding_paper.pdf, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 2019.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383-2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URLhttps://www.aclweb. org/anthology/D16-1264.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 784-789, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124. URLhttps://www.aclweb. org/anthology/P18-2124

Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, et al. Mesh-tensorflow: Deep learning for supercomputers. In Advances in Neural Information Processing Systems, pp. 10414$10423,2018$.

Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. Bi-directional block selfattention for fast and memory-efficient sequence modeling. arXiv preprint arXiv:1804.00857, 2018.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism, 2019.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631-1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URLhttps://www.aclweb.org/anthology/D13-1170

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for BERT model compression. arXiv preprint arXiv:1908.09355, 2019.

Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: The impact of student initialization on knowledge distillation. arXiv preprint arXiv:1908.08962, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 353-355, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5446. URL https://www.aclweb.org/anthology/ W18-5446

Wei Wang, Bin Bi, Ming Yan, Chen Wu, Zuyi Bao, Liwei Peng, and Luo Si. StructBERT: Incorporating language structures into pre-training for deep language understanding. arXiv preprint arXiv:1908.04577, 2019 .

Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018.

Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North

American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112-1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1101. URL https://www.aclweb. org/anthology/N18-1101

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv: $1906.08237,2019$.

Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. Reducing BERT pre-training time from 3 days to 76 minutes. arXiv preprint arXiv:1904.00962, 2019.

Shuailiang Zhang, Hai Zhao, Yuwei Wu, Zhuosheng Zhang, Xi Zhou, and Xiang Zhou. DCMN+: Dual co-matching network for multi-choice reading comprehension. arXiv preprint arXiv:1908.11511, 2019.

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp. $19-27,2015$.
</end of paper 2>


<paper 3>
# GECToR - Grammatical Error Correction: Tag, Not Rewrite 

Kostiantyn Omelianchuk $\quad$ Vitaliy Atrasevych ${ }^{*}$ Artem Chernodub* ${ }^{*} \quad$ Oleksandr Skurzhanskyi*<br>Grammarly<br>$\{$ firstname.lastname\}@grammarly.com


#### Abstract

In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an $F_{0.5}$ of $65.3 / 66.5$ on CoNLL-2014 (test) and $F_{0.5}$ of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system. The code and trained models are publicly available ${ }^{1}$.


## 1 Introduction

Neural Machine Translation (NMT)-based approaches (Sennrich et al., 2016a) have become the preferred method for the task of Grammatical Error Correction (GEC) $)^{2}$. In this formulation, errorful sentences correspond to the source language, and error-free sentences correspond to the target language. Recently, Transformer-based (Vaswani et al., 2017) sequence-to-sequence (seq2seq) models have achieved state-of-the-art performance on standard GEC benchmarks (Bryant et al., 2019). Now the focus of research has shifted more towards generating synthetic data for pretraining the Transformer-NMT-based GEC systems (Grundkiewicz et al., 2019; Kiyono et al., 2019). NMT-based GEC systems suffer from several issues which make them inconvenient for real world deployment: (i) slow inference speed, (ii) demand for large amounts of training data[^0]

and (iii) interpretability and explainability; they require additional functionality to explain corrections, e.g., grammatical error type classification (Bryant et al., 2017).

In this paper, we deal with the aforementioned issues by simplifying the task from sequence generation to sequence tagging. Our GEC sequence tagging system consists of three training stages: pretraining on synthetic data, fine-tuning on an errorful parallel corpus, and finally, fine-tuning on a combination of errorful and error-free parallel corpora.

Related work. LaserTagger (Malmi et al., 2019) combines a BERT encoder with an autoregressive Transformer decoder to predict three main edit operations: keeping a token, deleting a token, and adding a phrase before a token. In contrast, in our system, the decoder is a softmax layer. PIE (Awasthi et al., 2019) is an iterative sequence tagging GEC system that predicts tokenlevel edit operations. While their approach is the most similar to ours, our work differs from theirs as described in our contributions below:

1. We develop custom $\mathrm{g}$-transformations: token-level edits to perform (g)rammatical error corrections. Predicting g-transformations instead of regular tokens improves the generalization of our GEC sequence tagging system.
2. We decompose the fine-tuning stage into two stages: fine-tuning on errorful-only sentences and further fine-tuning on a small, high-quality dataset containing both errorful and error-free sentences.
3. We achieve superior performance by incorporating a pre-trained Transformer encoder in our GEC sequence tagging system. In our experiments, encoders from XLNet and RoBERTa outperform three other cutting-edge Transformer encoders (ALBERT, BERT, and GPT-2).

| Dataset | \# sentences | \% errorful <br> sentences | Training <br> stage |
| :---: | :---: | :---: | :---: |
| PIE-synthetic | $9,000,000$ | $100.0 \%$ | I |
| Lang-8 | 947,344 | $52.5 \%$ | II |
| NUCLE | 56,958 | $38.0 \%$ | II |
| FCE | 34,490 | $62.4 \%$ | II |
| W\&I+LOCNESS | 34,304 | $67.3 \%$ | II, III |

Table 1: Training datasets. Training stage I is pretraining on synthetic data. Training stages II and III are for fine-tuning.

## 2 Datasets

Table 1 describes the finer details of datasets used for different training stages.

Synthetic data. For pretraining stage I, we use $9 \mathrm{M}$ parallel sentences with synthetically generated grammatical errors (Awasthi et al., 2019) 3 .

Training data. We use the following datasets for fine-tuning stages II and III: National University of Singapore Corpus of Learner English (NUCLE) ${ }^{4}$ (Dahlmeier et al., 2013), Lang-8 Corpus of Learner English (Lang-8) ${ }^{5}$ (Tajiri et al., 2012), FCE dataset ${ }^{6}$ (Yannakoudakis et al., 2011), the publicly available part of the Cambridge Learner Corpus (Nicholls, 2003) and Write \& Improve + LOCNESS Corpus (Bryant et al., 2019) ${ }^{7}$.

Evaluation data. We report results on CoNLL2014 test set (Ng et al., 2014) evaluated by official $M^{2}$ scorer (Dahlmeier and $\mathrm{Ng}, 2012$ ), and on BEA-2019 dev and test sets evaluated by ERRANT (Bryant et al., 2017).

## 3 Token-level transformations

We developed custom token-level transformations $T\left(x_{i}\right)$ to recover the target text by applying them to the source tokens $\left(x_{1} \ldots x_{N}\right)$. Transformations increase the coverage of grammatical error corrections for limited output vocabulary size for the most common grammatical errors, such as Spelling, Noun Number, Subject-Verb Agreement and Verb Form (Yuan, 2017, p. 28).

The edit space which corresponds to our default tag vocabulary size $=5000$ consists of 4971[^1]

basic transformations (token-independent KEEP, DELETE and 1167 token-dependent APPEND, 3802 REPLACE) and 29 token-independent $g$ transformations.

Basic transformations perform the most common token-level edit operations, such as: keep the current token unchanged (tag $\$ K E E P$ ), delete current token (tag \$DELETE), append new token $t_{1}$ next to the current token $x_{i}\left(\operatorname{tag} \$ A P P E N D_{-} t_{1}\right)$ or replace the current token $x_{i}$ with another token $t_{2}$ (tag \$REPLACE_ $t_{2}$ ).

g-transformations perform task-specific operations such as: change the case of the current token (CASE tags), merge the current token and the next token into a single one (MERGE tags) and split the current token into two new tokens (SPLIT tags). Moreover, tags from NOUN NUMBER and VERB FORM transformations encode grammatical properties for tokens. For instance, these transformations include conversion of singular nouns to plurals and vice versa or even change the form of regular/irregular verbs to express a different number or tense.

To obtain the transformation suffix for the VERB_FORM tag, we use the verb conjugation dictionary ${ }^{8}$. For convenience, it was converted into the following format: token $_{0}$ token $_{1}:$ tag $_{0} \_t_{a g_{1}}$ (e.g., go_goes : $V B_{-} V B Z$ ). This means that there is a transition from word $d_{0}$ and wor $_{1}$ to the respective tags. The transition is unidirectional, so if there exists a reverse transition, it is presented separately.

The experimental comparison of covering capabilities for our token-level transformations is in $\mathrm{Ta}-$ ble 2. All transformation types with examples are listed in Appendix, Table 9.

Preprocessing. To approach the task as a sequence tagging problem we need to convert each target sentence from training/evaluation sets into a sequence of tags where each tag is mapped to a single source token. Below is a brief description of our 3-step preprocessing algorithm for color-coded sentence pair from Table 3:

Step 1). Map each token from source sentence to subsequence of tokens from target sentence. [A $\mapsto$ A], [ten $\mapsto$ ten, -], [years $\mapsto$ year, -], [old $\mapsto$ old], [go $\mapsto$ goes, to], [school $\mapsto$ school, .].[^2]

| Tag <br> vocab. size | Transformations |  |
| :---: | :---: | :---: |
|  | Basic transf. | All transf. |
| 100 | $60.4 \%$ | $79.7 \%$ |
| 1000 | $76.4 \%$ | $92.9 \%$ |
| 5000 | $89.5 \%$ | $98.1 \%$ |
| 10000 | $93.5 \%$ | $100.0 \%$ |

Table 2: Share of covered grammatical errors in CoNLL-2014 for basic transformations only (KEEP, DELETE, APPEND, REPLACE) and for all transformations w.r.t. tag vocabulary's size. In our work, we set the default tag vocabulary size $=5000$ as a heuristical compromise between coverage and model size.

For this purpose, we first detect the minimal spans of tokens which define differences between source tokens $\left(x_{1} \ldots x_{N}\right)$ and target tokens $\left(y_{1} \ldots y_{M}\right)$. Thus, such a span is a pair of selected source tokens and corresponding target tokens. We can't use these span-based alignments, because we need to get tags on the token level. So then, for each source token $x_{i}, 1 \leq i \leq$ $N$ we search for best-fitting subsequence $\Upsilon_{i}=$ $\left(y_{j_{1}} \ldots y_{j_{2}}\right), 1 \leq j_{1} \leq j_{2} \leq M$ of target tokens by minimizing the modified Levenshtein distance (which takes into account that successful gtransformation is equal to zero distance).

Step 2). For each mapping in the list, find tokenlevel transformations which convert source token to the target subsequence: [A $\mathrm{A}$ ] : $\$$ KEEP, [ten $\mapsto$ ten, -]: \$KEEP, \$MERGE_HYPHEN, [years $\mapsto$ year, -]: \$NOUN_NUMBER_SINGULAR, \$MERGE_HYPHEN], [old $\mapsto$ old]: \$KEEP, [go $\mapsto$ goes, to]: \$VERB_FORM_VB_VBZ, \$APPEND_to, [school $\mapsto$ school, .]: \$KEEP, \$APPEND_\{. $\}$.

Step 3). Leave only one transformation for each source token: A $\Leftrightarrow$ \$KEEP, ten $\Leftrightarrow$ \$MERGE_HYPHEN, years $\Leftrightarrow$ \$NOUN_NUMBER_SINGULAR, old $\Leftrightarrow$ \$KEEP, go $\Leftrightarrow$ \$VERB_FORM_VB_VBZ, school $\Leftrightarrow$ \$APPEND_ $\{$.$\} .$

The iterative sequence tagging approach adds a constraint because we can use only a single tag for each token. In case of multiple transformations we take the first transformation that is not a \$KEEP tag. For more details, please, see the preprocessing script in our repository ${ }^{9}$.

## 4 Tagging model architecture

Our GEC sequence tagging model is an encoder made up of pretrained BERT-like transformer[^3]

| Iteration \# | Sentence's evolution | \# corr. |
| :---: | :---: | :---: |
| Orig. sent | A ten years old boy go school | - |
| Iteration 1 | A ten-years old boy goes school | 2 |
| Iteration 2 | A ten-year-old boy goes to school | 5 |
| Iteration 3 | A ten-year-old boy goes to school. | 6 |

Table 3: Example of iterative correction process where GEC tagging system is sequentially applied at each iteration. Cumulative number of corrections is given for each iteration. Corrections are in bold.

stacked with two linear layers with softmax layers on the top. We always use cased pretrained transformers in their Base configurations. Tokenization depends on the particular transformer's design: BPE (Sennrich et al., 2016b) is used in RoBERTa, WordPiece (Schuster and Nakajima, 2012) in BERT and SentencePiece (Kudo and Richardson, 2018) in XLNet. To process the information at the token-level, we take the first subword per token from the encoders representation, which is then forwarded to subsequent linear layers, which are responsible for error detection and error tagging, respectively.

## 5 Iterative sequence tagging approach

To correct the text, for each input token $x_{i}, 1 \leq$ $i \leq N$ from the source sequence $\left(x_{1} \ldots x_{N}\right)$, we predict the tag-encoded token-level transformation $T\left(x_{i}\right)$ described in Section 3. These predicted tagencoded transformations are then applied to the sentence to get the modified sentence.

Since some corrections in a sentence may depend on others, applying GEC sequence tagger only once may not be enough to fully correct the sentence. Therefore, we use the iterative correction approach from (Awasthi et al., 2019): we use the GEC sequence tagger to tag the now modified sequence, and apply the corresponding transformations on the new tags, which changes the sentence further (see an example in Table 3). Usually, the number of corrections decreases with each successive iteration, and most of the corrections are done during the first two iterations (Table 4). Limiting the number of iterations speeds up the overall pipeline while trading off qualitative performance.

| Iteration \# | $\mathbf{P}$ | $\mathbf{R}$ | $\mathbf{F}_{\mathbf{0} .5}$ | \# corr. |
| :---: | :---: | :---: | :---: | :---: |
| Iteration 1 | 72.3 | 38.6 | 61.5 | 787 |
| Iteration 2 | 73.7 | 41.1 | 63.6 | 934 |
| Iteration 3 | 74.0 | 41.5 | 64.0 | 956 |
| Iteration 4 | 73.9 | 41.5 | 64.0 | 958 |

Table 4: Cumulative number of corrections and corresponding scores on CoNLL-2014 (test) w.r.t. number of iterations for our best single model.

| Training <br> stage \# | CoNLL-2014 (test) |  | BEA-2019 (dev) |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathbf{P}$ | $\mathbf{R}$ | $\mathbf{F}_{\mathbf{0 . 5}}$ | $\mathbf{P}$ | $\mathbf{R}$ | $\mathbf{F}_{\mathbf{0} . \mathbf{5}}$ |
| Stage I. | 55.4 | 35.9 | 49.9 | 37.0 | 23.6 | 33.2 |
| Stage II. | 64.4 | 46.3 | 59.7 | 46.4 | 37.9 | 44.4 |
| Stage III. | 66.7 | $\mathbf{4 9 . 9}$ | 62.5 | 52.6 | $\mathbf{4 3 . 0}$ | 50.3 |
| Inf. tweaks | $\mathbf{7 7 . 5}$ | 40.2 | $\mathbf{6 5 . 3}$ | $\mathbf{6 6 . 0}$ | 33.8 | $\mathbf{5 5 . 5}$ |

Table 5: Performance of GECToR (XLNet) after each training stage and inference tweaks.

## 6 Experiments

Training stages. We have 3 training stages (details of data usage are in Table 1):

I Pre-training on synthetic errorful sentences as in (Awasthi et al., 2019).

II Fine-tuning on errorful-only sentences.

III Fine-tuning on subset of errorful and errorfree sentences as in (Kiyono et al., 2019).

We found that having two fine-tuning stages with and without error-free sentences is crucial for performance (Table 5).

All our models were trained by Adam optimizer (Kingma and Ba, 2015) with default hyperparameters. Early stopping was used; stopping criteria was 3 epochs of $10 \mathrm{~K}$ updates each without improvement. We set batch size=256 for pre-training stage I (20 epochs) and batch size=128 for finetuning stages II and III (2-3 epochs each). We also observed that freezing the encoder's weights for the first 2 epochs on training stages I-II and using a batch size greater than 64 improves the convergence and leads to better GEC performance.

Encoders from pretrained transformers. We fine-tuned BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), GPT-2 (Radford et al., 2019), XLNet (Yang et al., 2019), and ALBERT (Lan et al., 2019) with the same hyperparameters setup. We also added LSTM with randomly initialized embeddings $(\mathrm{dim}=300)$ as a baseline. As follows from Table 6, encoders from fine-tuned Transformers significantly outperform LSTMs. BERT, RoBERTa and XLNet encoders perform better than GPT-2 and ALBERT, so we used them only in our next experiments. All models were trained out-of-the-box ${ }^{10}$ which seems to not work well for GPT-2. We hypothesize that encoders from Transformers which were pretrained as a part of the entire encoder-decoder pipeline are less useful for GECToR.

| Encoder | CoNLL-2014 (test) |  | BEA-2019 (dev) |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathbf{P}$ | $\mathbf{R}$ | $\mathbf{F}_{\mathbf{0} .5}$ |  |  |  |
| LSTM | P | $\mathbf{R}$ | $\mathbf{F}_{0.5}$ |  |  |  |
| LSTBERT | 51.6 | 15.3 | 35.0 | - | - | - |
| ALBER | 59.5 | 31.0 | 50.3 | 43.8 | 22.3 | 36.7 |
| BERT | 65.6 | 36.9 | 56.8 | 48.3 | 29.0 | 42.6 |
| GTT-2 | 61.0 | 6.3 | 22.2 | 44.5 | 5.0 | 17.2 |
| RoBERTa | $\mathbf{6 7 . 5}$ | 38.3 | $\mathbf{5 8 . 6}$ | $\mathbf{5 0 . 3}$ | 30.5 | $\mathbf{4 4 . 5}$ |
| XLNet | 64.6 | $\mathbf{4 2 . 6}$ | 58.5 | 47.1 | $\mathbf{3 4 . 2}$ | 43.8 |

Table 6: Varying encoders from pretrained Transformers in our sequence labeling system. Training was done on data from training stage II only.

Tweaking the inference. We forced the model to perform more precise corrections by introducing two inference hyperparameters (see Appendix, Table 11), hyperparameter values were found by random search on BEA-dev.

First, we added a permanent positive confidence bias to the probability of $\$$ KEEP tag which is responsible for not changing the source token. Second, we added a sentence-level minimum error probability threshold for the output of the error detection layer. This increased precision by trading off recall and achieved better $F_{0.5}$ scores (Table 5).

Finally, our best single-model, GECToR (XLNet) achieves $F_{0.5}=65.3$ on CoNLL-2014 (test) and $F_{0.5}=72.4$ on BEA-2019 (test). Best ensemble model, GECToR (BERT + RoBERTa + XLNet) where we simply average output probabilities from 3 single models achieves $F_{0.5}=66.5$ on CoNLL-2014 (test) and $F_{0.5}=73.6$ on BEA-2019 (test), correspondingly (Table 7).

Speed comparison. We measured the models average inference time on NVIDIA Tesla V100 on batch size 128. For sequence tagging we don't need to predict corrections one-by-one as in autoregressive transformer decoders, so inference is naturally parallelizable and therefore runs many times faster. Our sequence tagger's inference speed is up to 10 times as fast as the state-ofthe-art Transformer from Zhao et al. (2019), beam size $=12$ (Table 8).[^4]

| GEC system | Ens. | CoNLL-2014 (test) |  |  | BEA-2019 (test) |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $\mathbf{P}$ | $\overline{\mathbf{R}}$ | $\overline{F_{0.5}}$ | $\overline{\mathbf{P}}$ | $\overline{\mathbf{R}}$ | $\overline{F_{0.5}}$ |
| Zhao et al. (2019) |  | ![](https://cdn.mathpix.com/cropped/2024_05_26_bb756a36a40aaaa21263g-5.jpg?height=38&width=71&top_left_y=284&top_left_x=1072) | 40.6 | $59.8 \quad$ | - | - | - |
| Awasthi et al. (2019) |  | ![](https://cdn.mathpix.com/cropped/2024_05_26_bb756a36a40aaaa21263g-5.jpg?height=38&width=71&top_left_y=321&top_left_x=1072) | 43.0 | 59.7 |  |  |  |
| Kiyono et al. (2019) |  | $67.9 \quad$ | 44.1 | $61.3 \quad-1-2$ | 65.5 | $59.4 \quad-1$ | 64.2 |
| Zhao et al. (2019) | $\bar{\checkmark}$ | 74.1 | 36.3 | $61.3 \quad 2 \quad-10$ | $\cdots$ | - | - |
| Awasthi et al. (2019) | $\checkmark$ | $68.3 \quad-\quad-1$ | $43.2 \quad-\quad-1<r$ | 61.2 |  | - | - |
| Kiyono | $\checkmark$ | 72.4 | 46.1 | 65.0 | 74.7 | 56.7 | 70.2 |
| Kantor et al. (2019) | $\checkmark$ | - | - | - | 78.3 | 58.0 | 73.2 |
| GECToR (BERT) |  | $72.1 \quad-1 . r$ | 42.0 | $63.0 \quad-\quad-10$ | 71.5 | $55.7 \quad 5$ | 67.6 |
| GECToR (RoBERTa) |  | $73.9 \quad-2-10$ | $41.5 \quad-5$ | $64.0 \quad-2-10$ | 77.2 | 55.1 | $71.5 \quad-\quad-r l t$ |
| GECToR (XLNet) |  | $77.5 \quad-\quad-1)$ | $40.1 \quad-\quad-1<r$ | $65.3 \quad-\quad-1-1$ | 79.2 | 53.9 | $72.4 \quad-\quad r \quad$ |
| GECToR (RoBERTa + XLN | $\checkmark$ | 76.6 | $42.3 \quad-10$ | 66.0 | $79.4 \quad 2$ | 57.2 | ![](https://cdn.mathpix.com/cropped/2024_05_26_bb756a36a40aaaa21263g-5.jpg?height=38&width=74&top_left_y=629&top_left_x=1560) |
| GECToR (BERT + RoBERTa + | $\checkmark$ | 78.2 | $41.5 \quad-\quad-10$ | $66.5 \quad-1-5$ | 78.9 | 58.2 | 73.6 |

Table 7: Comparison of single models and ensembles. The $M^{2}$ score for CoNLL-2014 (test) and ERRANT for the BEA-2019 (test) are reported. In ensembles we simply average output probabilities from single models.

| GEC system | Time (sec) |
| :--- | :---: |
| Transformer-NMT, beam size $=12$ | 4.35 |
| Transformer-NMT, beam size $=4$ | 1.25 |
| Transformer-NMT, beam size $=1$ | 0.71 |
| GECToR (XLNet), 5 iterations | 0.40 |
| GECToR (XLNet), 1 iteration | 0.20 |

Table 8: Inference time for NVIDIA Tesla V100 on CoNLL-2014 (test), single model, batch size $=128$.

## 7 Conclusions

We show that a faster, simpler, and more efficient GEC system can be developed using a sequence tagging approach, an encoder from a pretrained Transformer, custom transformations and 3 -stage training.

Our best single-model/ensemble GEC tagger achieves an $F_{0.5}$ of 65.3/66.5 on CoNLL-2014 (test) and $F_{0.5}$ of 72.4/73.6 on BEA-2019 (test). We achieve state-of-the-art results for the GEC task with an inference speed up to 10 times as fast as Transformer-based seq 2 seq systems.

## References

Abhijeet Awasthi, Sunita Sarawagi, Rasna Goyal, Sabyasachi Ghosh, and Vihari Piratla. 2019. Parallel iterative edit models for local sequence transduction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 42604270, Hong Kong, China. Association for Computational Linguistics.

Christopher Bryant, Mariano Felice, Øistein E. Andersen, and Ted Briscoe. 2019. The BEA-2019 shared task on grammatical error correction. In Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 52-75, Florence, Italy. Association for Computational Linguistics.

Christopher Bryant, Mariano Felice, and Ted Briscoe. 2017. Automatic annotation and evaluation of error types for grammatical error correction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 793-805, Vancouver, Canada. Association for Computational Linguistics.

Daniel Dahlmeier and Hwee Tou Ng. 2012. Better evaluation for grammatical error correction. In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 568-572. Association for Computational Linguistics.

Daniel Dahlmeier, Hwee Tou Ng, and Siew Mei Wu. 2013. Building a large annotated corpus of learner english: The nus corpus of learner english. In Proceedings of the eighth workshop on innovative use of NLP for building educational applications, pages $22-31$.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference
of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Roman Grundkiewicz, Marcin Junczys-Dowmunt, and Kenneth Heafield. 2019. Neural grammatical error correction systems with unsupervised pre-training on synthetic data. In Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 252-263.

Yoav Kantor, Yoav Katz, Leshem Choshen, Edo CohenKarlik, Naftali Liberman, Assaf Toledo, Amir Menczel, and Noam Slonim. 2019. Learning to combine grammatical error corrections. In Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 139-148, Florence, Italy. Association for Computational Linguistics.

Diederik P Kingma and Jimmy Ba. 2015. Adam (2014), a method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), arXiv preprint arXiv, volume 1412.

Shun Kiyono, Jun Suzuki, Masato Mita, Tomoya Mizumoto, and Kentaro Inui. 2019. An empirical study of incorporating pseudo data into grammatical error correction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pages 1236-1242, Hong Kong, China. Association for Computational Linguistics.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66-71, Brussels, Belgium. Association for Computational Linguistics.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, and Aliaksei Severyn. 2019. Encode, tag, realize: High-precision text editing. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5054-5065, Hong
Kong, China. Association for Computational Linguistics.

Hwee Tou Ng, Siew Mei Wu, Ted Briscoe, Christian Hadiwinoto, Raymond Hendy Susanto, and Christopher Bryant. 2014. The conll-2014 shared task on grammatical error correction. In Proceedings of the Eighteenth Conference on Computational Natural Language Learning: Shared Task, pages 1-14.

Diane Nicholls. 2003. The cambridge learner corpus: Error coding and analysis for lexicography and elt. In Proceedings of the Corpus Linguistics 2003 conference, volume 16, pages 572-581.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9.

Mike Schuster and Kaisuke Nakajima. 2012. Japanese and korean voice search. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5149-5152. IEEE.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016a. Edinburgh neural machine translation systems for WMT 16. In Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers, pages 371-376, Berlin, Germany. Association for Computational Linguistics.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17151725, Berlin, Germany. Association for Computational Linguistics.

Toshikazu Tajiri, Mamoru Komachi, and Yuji Matsumoto. 2012. Tense and aspect error correction for esl learners using global context. In Proceedings of the 50th Aпnual Meeting of the Association for Computational Linguistics: Short Papers-Volume 2, pages 198-202. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems, pages 5998-6008.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pages 5754-5764.

Helen Yannakoudakis, Ted Briscoe, and Ben Medlock. 2011. A new dataset and method for automatically grading esol texts. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1 , pages 180-189. Association for Computational Linguistics.

Zheng Yuan. 2017. Grammatical error correction in non-native english. Technical report, University of Cambridge, Computer Laboratory.

Wei Zhao, Liang Wang, Kewei Shen, Ruoyu Jia, and Jingming Liu. 2019. Improving grammatical error correction via pre-training a copy-augmented architecture with unlabeled data. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: $\mathrm{Hu}$ man Language Technologies, Volume 1 (Long and Short Papers), pages 156-165, Minneapolis, Minnesota. Association for Computational Linguistics.
</end of paper 3>


<paper 4>
# STAR: A Benchmark for Situated Reasoning in Real-World Videos 

Bo Wu<br>MIT-IBM Watson AI Lab

Shoubin Yu<br>Shanghai Jiao Tong University

Zhenfang Chen<br>MIT-IBM Watson AI Lab

Joshua B. Tenenbaum<br>MIT BCS, CBMM, CSAIL

Chuang Gan<br>MIT-IBM Watson AI Lab

http://star.csail.mit.edu


#### Abstract

Reasoning in the real world is not divorced from situations. How to capture the present knowledge from surrounding situations and perform reasoning accordingly is crucial and challenging for machine intelligence. This paper introduces a new benchmark that evaluates the situated reasoning ability via situation abstraction and logic-grounded question answering for real-world videos, called Situated Reasoning in Real-World Videos (STAR). This benchmark is built upon the realworld videos associated with human actions or interactions, which are naturally dynamic, compositional, and logical. The dataset includes four types of questions, including interaction, sequence, prediction, and feasibility. We represent the situations in real-world videos by hyper-graphs connecting extracted atomic entities and relations (e.g., actions, persons, objects, and relationships). Besides visual perception, situated reasoning also requires structured situation comprehension and logical reasoning. Questions and answers are procedurally generated. The answering logic of each question is represented by a functional program based on a situation hyper-graph. We compare various existing video reasoning models and find that they all struggle on this challenging situated reasoning task. We further propose a diagnostic neuro-symbolic model that can disentangle visual perception, situation abstraction, language understanding, and functional reasoning to understand the challenges of this benchmark.


## 1 Introduction

Reasoning about real-world situations is essential to human intelligence. In a specific situation like Figure 1, we are able to know how to act in situations quickly and make feasible decisions subconsciously. That means we are logically antecedent before the concrete act. "Situated Reasoning" aims at making us understand situations dynamically and reason with the present knowledge accordingly. Such ability is logic-centered but not isolated or divorced from the surrounding situations since cognition in the real world cannot be separated from the context [5].

In fact, such situated reasoning in the real world is very challenging to existing intelligent systems. Early studies about reasoning in actions [35, 42] provide formalism definitions and frameworks from logic formalism perspectives (e.g., situation calculus, etc.). They formulate situations as a set of formulae and perform calculus based on the designed logic rules [31, 36]. However, creating all possible logic rules in real scenarios is impossible, limiting their practicality. Recent studies of visual reasoning on synthetic video datasets [47] demonstrate the possibilities to connect visual perception, language understanding with symbolic reasoning. It remains unclear to what extent the model performs well on these synthetic datasets can be extended to real-world situations.

![](https://cdn.mathpix.com/cropped/2024_05_26_9ad87db506347868b4e4g-02.jpg?height=764&width=1372&top_left_y=236&top_left_x=363)

Figure 1: A representative example in the benchmark STAR. STAR aims at evaluating the skills in real-world situation recognition, abstraction, and reasoning. Q, A, and S indicate questions, answers, and situation data types with palace-holders. Answers in green (bold font) or red mean correct or incorrect. Masked situations are unseen for prediction or feasibility questions. Best viewed in color.

According to situated cognition theory $[7,5,4]$, situated reasoning relies on logical thinking and integrates naturally with the present knowledge captured from the surrounding situations. Such situated reasoning may be trivial for humans but not easy to current state-of-the-art methods. According to the experiment results in Table 2 of the paper, we find existing QA models struggle with these challenging tasks, and they mainly leveraging the correlation between the visual content and question-answer pairs instead of reasoning. To explore situated reasoning with increasing complexity, we propose STAR, a novel benchmark for real-world situated reasoning via videos that require systems to capture the present knowledge from dynamic situations as structured representation and answer questions accordingly. From our perspective, such ability is a progressive process from concrete situations to mental logic. We hope the diagnostic benchmark will help to reduce the gap by conducting bottom-up perception, structured abstraction, and explicit reasoning in real-world videos.

We take human activities or actions in daily life as an exemplary domain and build the dataset upon video clips of real-world situations. The benchmark includes four types of questions: interaction question, sequence question, prediction question, and feasibility question. Each question is associated with an action-centered situation from diverse scenes and places, and each situation involves multiple actions. In order to represent the present knowledge and their dynamic changes in situations, we abstract them into structured representations with entities and relations: situation hypergraphs. Inspired by the work [22], our benchmark designs well-controlled questions and answers by question templates and programs. We simplify the language understanding by adopting concise forms and question templates for generation since our research scope mainly focuses on diagnostics for visual reasoning ability. And we also provided an auxiliary set STAR-Humans to help the evaluation with more challenging human-written questions. The answering logics describe logical reasoning processes which were grounded to executive programs over generated situation hypergraphs. We analyzed rationality by human annotations by showing these situations and synthetic questions and choices to annotators. As summarized in Table 1, STAR complements existing visual reasoning benchmarks on various aspects. It combines both situation abstraction and diagnostic reasoning focusing on human-object interaction, temporal sequence analysis, action prediction, and feasibility inference. We evaluate various visual question answering or visual reasoning models on STAR but find none of them can achieve promising performance. We design a diagnostic model called Neuro-Symbolic Situated Reasoning (NS-SR), a neural-symbolic architecture for real-world situated reasoning. It answers questions by leveraging structured situation graphs and dynamic clues from situations to perform symbolic reasoning. Our main contributions are:

- We systematically formulate the problem of situated reasoning from real-world videos, focusing on interaction, sequence, prediction, and feasibility questions.
- We construct a well-controlled benchmark STAR for situated reasoning, where designing annotations from three perspectives: visual perception, situation abstraction and logic reasoning. Each video is grounded with a situation hyper-graph, and each question is associated with a functional program that specifies the explicit reasoning steps to answer the question.
- We evaluate various state-of-the-art methods on STAR and find that they still make many mistakes in situations that are trivial for humans.
- We design a diagnostic neuro-symbolic framework for an in-depth analysis of the challenges on STAR benchmark and provide future directions on building more powerful reasoning models.


## 2 Related Work

Visual Question Answering Visual Question Answering $[1,40]$ requires a model to answer visual related questions via understanding both visual content and question semantics. The existing visual/video question answering benchmarks [13, 50, 24, 40, 44] adopted images [1, 50, 11]/videos [40, $44,19,28,18,9,45]$ and types of visual comprehension questions. They achieved significant progress on evaluating the vision-language understanding ability of systems from multiple perspectives of perception. Differently, STAR requires systems to perform explicit reasoning in real-world situations and provides step-by-step reasoning programs.

Visual Reasoning Beyond visual question answering, several new datasets [21, 18, 47, 14, 16, 6] are designed to diagnose models' reasoning abilities. They contain questions with compositional attributes and logic programs, which require systems to perform step-by-step reasoning. It was first studied in CLEVR [21] and GQA [18] for reasoning in static images. Later, it was extended to the video domain for a more complex visual senses. MarioQA [32], COG [46], CATER [12] and CLEVRER [47] include human-annotated or generated questions and synthetic videos from simulated environments. They ask models to recognize geometric objects and their movements or collisions for understanding of compositional or spatio-temporal relations in the form of video question answering. Most of them focus on objects dynamics in synthetic scenes and it remains a doubt whether those are representative enough to reflect the complexity of real-world situations. AGQA [14] is the most recent work about reasoning in real-world videos, but it focuses on spatio-temporal relations.

Situation Formalism Early-stage work [31, 36, 25] establish formalisms for reasoning about action and change. The situation calculus represents changing scenarios as a set of first-order logic formulae. However, it is not realistic to apply such formalisms directly to real-world situations. Not all axioms are visible or detectable. Moreover, real-world situations are dynamic and have not been well-defined. It is still an open challenge to diagnose reasoning about actions for real-world situations.

| Dataset | Real-World |  | Situation |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Videos | Abstraction | Diagnostic <br> Reasoning | Interaction | Sequence | Prediction | Feasibility |
| VQA [1] | $x$ | $x$ | $x$ |  |  |  |  |
| VCR [49] | $x$ | $x$ | $x$ | $\checkmark$ | $\checkmark$ | $x$ | $x$ |
| GQA [18] | $x$ | $\checkmark$ | $x$ | $x$ | $x$ | $x$ | $x$ |
| CLEVR [21] | $x$ | $x$ | $\checkmark$ | $x$ | $x$ | $x$ | $x$ |
| COG [46] | $x$ | $x$ | $\checkmark$ | $x$ | $x$ | $x$ | $x$ |
| CLEVRER [47] | $x$ | $x$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $x$ |
| TGIF-QA [19] | $\checkmark$ | $x$ | $x$ | $\checkmark$ | $\checkmark$ | $x$ | $x$ |
| MovieQA [40] | $\checkmark$ | $x$ | $x$ | $\checkmark$ | $\checkmark$ | $x$ | $x$ |
| TVQA/TVQA+ [28, 29] | $\checkmark$ | $x$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $x$ | $x$ |
| STAR (ours) | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |

Table 1: Comparison between STAR and other benchmarks (visual reasoning or video QA). STAR is a real-world situated reasoning benchmark with situation abstraction and diagnostic reasoning. It contains a wide range of reasoning tasks about human-object interaction, temporal sequence analysis, action prediction, and feasibility inference.

## 3 Situated Reasoning Benchmark

STAR evaluates the human-like ability: situated reasoning. It requires systems to learn and perform reasoning in real-world situations to challenging questions. Building a situated reasoning benchmark via real-world data is challenging because it requires tight-controlled situation clues and well-designed question-answer pairs. We combine both situations abstraction and logical reasoning and adopt three guidelines in our benchmark construction: 1. situations are represented by hierarchical graphs based on bottom-up annotations for abstraction; 2. question and option generation for situated reasoning is grounded to formatted questions, functional programs, and shared situation data types; 3. situated reasoning can perform over the situation graphs iteratively.

STAR consists of about $60 \mathrm{~K}$ situated reasoning questions with programs and answers, $240 \mathrm{~K}$ candidate choices, and $22 \mathrm{~K}$ trimmed situation video clips. Situation video clips in our benchmark are sourced from human activity videos, which record the dynamic interaction processes of human actions and surrounding environments in daily-life scenes. We also provide about $144 \mathrm{~K}$ situation hypergraphs as structured situation abstraction. Designed questions cover four types of skills for situated reasoning. We constructed annotated questions with answers and options. Each question answering corresponds to a specific program for reasoning logic. To connect situation abstraction and reasoning diagnosis for question-answering, we provide situation hypergraphs tied with executable programs. Then situations, questions, and options are aligned with the unified data type schema, including actions, objects, humans, and relations. The STAR includes 111 action predicates, 28 objects, and 24 relationships. The benchmark is split into training/validation/test sets with a ratio of about 6:1:1. More dataset setting and data analysis details are in the supplementary material Section 2 or 3.

### 3.1 Situation Abstraction

Situations Situation is a core concept in the STAR benchmark. It describes entities, events, moments, and environments. We build up situations start from $9 \mathrm{~K}$ source videos with action annotations sampled from Charades dataset [38]. The videos describe daily-life actions or activities in 11 indoor scenes, such as the kitchen, living room, bedroom, etc.. A situation is a trimmed video with multiple consecutive or overlapped actions and interactions. According to the provided annotations, we filter source videos by their quality, stability, and video length to construct clean and unambiguous data space for situations. All situation videos in our dataset are trimmed from source videos according to question types, temporal boundaries of multiple appeared actions (from Charades), and question logic. We split each action into two action segments according to the definition in situation calculus [31]: action precondition and effect. The action precondition is the beginning frame to show an initial static scene of the environment. The action effect describes the process of a single action or multiple actions. Situations of interaction or sequence questions contain complete action segments. Situations of prediction questions (or feasibility questions) include the actions involved in questions and an incomplete action effect segment (or no other action segments) about answers.

Situation Hypergraph To distill abstract representations from situation videos, STAR benchmark defines a unified schema to describe dynamic processes in real-world situations in the form of the hypergraph. Situation hypergraphs represent actions and inner-relations and their hierarchical structures within situations. As shown in Figure 1, each situation video is a set of subgraphs with person and object nodes, and edges represent in-frame relations (person-object or object-object). Meanwhile, each action hyperedge connects multiple subgraphs. In some cases, multiple actions are overlapped, and the nodes in subgraphs are shared. The entire dynamic process in a situation can be abstracted to a set of consecutive and overlapped situation hypergraphs. Formally, the situation hypergraph $H$ is a pair $H=(X, E)$ where $X$ is a set of nodes for objects or persons that appeared in situation frames, and $E$ is a set of non-empty hyperedge subgraphs $S_{i}$ for actions. Different from spatio-temporal graphs [24, 41, 20], the hypergraph structure describes actions as hyperedges and instead of the frame-level subgraphs. Such structure naturally reflects the hierarchical abstraction from real-world situations and symbolic representations. The annotations of situation hypergraphs are as follows: We created the one-to-many connections as action hyperedges based on the annotations of action temporal duration and appeared objects. The action annotations are from Charades, personobject relationships (Rel1), objects/persons annotations are from ActionGenome [20]. We extracted object-object relationships (Rel2) by using a detector VCTree with TDE [39], and extended more person-object relations (Rel3) with relation propagation over Rel1 and Rel2. For example, if <person, on, chair> and <chair, on the left of, table> exist, the <person, on the left of, table> exists. All models
in experiments use videos as inputs, but hypergraph annotations (entities, relationships, actions, or entire graphs) can be used to learn better visual perception or structured abstraction.

### 3.2 Questions and Answers Designing

The question-answer engine generates all questions, answers, and options based on situation hypergraphs. Such design allows the question and answer generation of STAR are under control and available to be applied in situated reasoning diagnosis.

Question Generation Situated reasoning questions ask systems to provide rational answers for multiple purposes in particular situations. We design multiple types of questions that indicate distinct purposes and cover different levels of difficulty in situated reasoning. In dynamic video situations, we propose that four types of purposes are essential and close to our daily life: happened facts, temporal order, future probability, and feasibility in a specific situation.

- Interaction Question (What did a person do ...): It is a basic test for understanding interactions between humans and objects in a situation.
- Sequence Question (What did the person do before/after ...): This type evaluates the temporal relationship reasoning of systems when facing consecutive actions in dynamic situations.
- Prediction Question ( What will the person do next with...): This type investigates the forecasting about plausible actions under the current situation. Seen situations only include the beginning (1/4) of actions (the remaining situations were masked), and questions ask the future actions or results.
- Feasibility Question (What is the person able to do/Which object is possible to be ...): This type probes the ability to infer feasible actions in particular situation conditions. We use spatial and temporal prompts (e.g., spatial relationships and temporal relationships) to control the situations.

To keep the logical consistency of the question types, all types of questions are derived from welldesigned templates and data from situation hypergraphs. We design formatted question templates with shared data type placeholders to align data types in situation hypergraphs (e.g., $[\mathrm{P}],[\mathrm{O}],[\mathrm{V}],[\mathrm{R}]$ for the person, objects, action verbs or relationships, etc..). Then the generation process is consists of the following steps: (1) data extraction from situation annotations and hypergraphs; (2) question templates filling with extracted data; (3) language expansion for phrase collocation and morphology (articles, prepositions, and tenses).

Answer Generation Each question has a correct answer generated by executing a functional program (parsed from the given question) on a STAR hypergraph of a given situation video. The program shows the step-by-step reasoning process on graph structures. A valid functional program (Supplementary material Figure 5) is a set of predefined and nested functional operations that can be executed (more details refer to the work in [22]) until getting the final correct answer. Each operation takes certain entities or relationships as inputs and returns the entities, relationships, or actions as the inputs of the next reasoning step or the final output.

Distractor Generation Setting deliberate confusion forces systems to distinguish the reasoning logic behind correct answers and incorrect options instead of guessing by probability. We design three distractor strategies: compositional option, random option, and frequent option.

- Compositional Option: This option is the most challenging incorrect option since it has contraries to the given situation. It satisfies the verb-object compositionality and is also generated from the program over happened facts in the same situation.
- Random Option: This option also satisfies compositionality but was randomly selected from other situation hypergraphs.
- Frequent Option: This option is used for deceiving models by probability. It selects the most happened option in each type of question group.

Finally, all options (one correct answer and three distractors) are randomly ordered for each question.

Debiasing and Balancing Strategies In real-world situations, the data of human actions naturally have distribution bias and reasoning shortcuts because some entities or action compositions (e.g., "wear clothes" or "grasp doorknob") frequently occurred. Such frequent collocation makes questions can be easily answered even without seeing the actual situations or questions. To avoid such shortcuts, we control the compositionality of appeared verbs/nouns in questions and answers and only select the verbs or nouns which has multiple compositions in our dataset world. To deal with answer distribution bias, we balance the answer distribution for each type of question through

![](https://cdn.mathpix.com/cropped/2024_05_26_9ad87db506347868b4e4g-06.jpg?height=325&width=1350&top_left_y=298&top_left_x=366)

Answer Distribution after Debiasing
![](https://cdn.mathpix.com/cropped/2024_05_26_9ad87db506347868b4e4g-06.jpg?height=984&width=1376&top_left_y=688&top_left_x=366)

Figure 2: Top: The bar charts show answer distribution comparison for before and after debiasing. T: question templates; INT/SEQ/PRE/FEA: four our question types. Bottom Left: The bar charts show answer distribution among options, which shows STAR has a balanced distribution on each template. Bottom Right: The two Sankey figures illustrate the compositionality distribution change of the key components within QA pairs after breaking shortcuts. Flows mean the number of the co-occurred key components. The left subfigure shows a heavily unbalanced distribution because of the existing QA shortcuts, but the right subfigure break such shortcuts distribution after processing.

resampling. Figure 2 top and bottom right show the results of before and after the debiasing on answers and breaking shortcuts in action combinations. We notice the trend that the STAR dataset has more balanced distributions after the debiasing stage. As shown in Figure 2 bottom left, We control the frequency of entities and actions in options so that each option has a fair chance to be correct.

Grammar Correctness and Correlation The questions and answers in STAR are generated automatically but in the form of natural language. To validate grammar correctness, we apply grammar checkers $[3,33]$ to perform grammar checking and correction for word typos, tense issues, or syntactic structures. The initial grammar correctness of generated questions and answers is $87 \%$. After three rounds of iterative corrections, the correctness achieved the expected level (improved to 98\%).

Rationality and Consistency The real-world source videos are noisy and quality-limited because recorders captured the videos by personal phones or cameras in various indoor environments. To confirm the relevance and quality of generated situation videos, questions, and candidate choices,
we evaluate STAR through rationality and consistency by human annotation. We perform statistical analysis through a majority vote on labeled results. Three Amazon MTurk crowd-workers labeled each question. The rationality measures if a question-answer sample and the associated situation has ill-posed, semantic misaligned, or data missing issues. For each question, annotators need to label rationality by observing both questions, candidate choices, and situation videos. Here are rationality statistics in terms of four question types (from interaction to feasibility): $89.9 \%, 87.2 \%, 78.5 \%$, and $77.5 \%$. Consistency was calculated by the matching ratios between human-labeled options and generated options overall questions. If there is no matched correct or wrong option, this sample is none of the above makes sense. The consistency statistics of four types of questions (from interaction to feasibility) are the following: $82.5 \%, 85.3 \%, 80.4 \%$, and $78.5 \%$. Finally, we only keep the samples that satisfy rationality and consistency in our dataset.

## 4 Baseline Evaluation

To evaluate STAR thoroughly, we test various baseline models and analyze their strengths and weaknesses in situated reasoning. In the evaluation, a model needs to select a correct answer from the four provided candidate options for a given question. We adopt the average answer accuracy of overall questions to measure the model performance. In Table 2, we present the performances of each model individually according to the four question types. For each question, we calculate answer accuracy per question by comparing all option correctness between ground-truth and predicted results. We select representative methods for our question-answering task as competitive baselines, which include Q-type models, blind models, vision-language models, and video question-answering models. All comparison models are trained from scratch on STAR training and validation sets, and tested on the STAR test set (implementation and setting details are in the supplementary material).

| Model Name | Question Type |  |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  | Interaction | Sequence | Prediction | Feasibility |
| Q-type (Random) [21] | 25.06 | 24.93 | 24.79 | 24.81 |
| Q-type (Frequent) [21] | 19.09 | 19.45 | 12.90 | 18.31 |
| Blind Model (LSTM) [15] | 32.24 | 32.17 | 28.56 | 28.41 |
| Blind Model (BERT) [8] | 32.68 | 34.21 | 29.98 | 29.26 |
| CNN-LSTM [47] | 33.25 | 32.67 | 30.69 | 30.43 |
| CNN-BERT [30] | 33.59 | 37.16 | 30.95 | 30.84 |
| LCGN [17] | 39.01 | 37.97 | 28.81 | 26.98 |
| HCRN [26] | 39.10 | 38.17 | 28.75 | 27.27 |
| ClipBERT [27] | 39.81 | 43.59 | 32.34 | 31.42 |

Table 2: Question-answering accuracy results of four question types on STAR (average accuracy per question). Video QA models perform better, but significant headroom remains for further exploration.

Q-type (Random) [21] randomly selects a choice as answer.

Q-type (Frequent) [21] chooses the highest frequency answer of each question type in the train set.

Blind Model (LSTM or BERT) is a language-only model. We uses an LSTM [15] or transformerbased model BERT [8] to encode question and choices and a MLP to predict the answer.

CNN+LSTM [47] takes the final state of an LSTM to capture language and visual context.

CNN+BERT reimplements VL-BERT model [30] for video QA.

LCGN [17] iteratively uses location-aware GCN to model object's spatial-temporal relations.

HCRN [26] is a recent video question answering model, which involves hierarchical conditional relation networks for better representation relation learning.

ClipBERT [27] is a recent state-of-the-art framework that enables end-to-end learning for video-andlanguage tasks including video question answering by employing sparse sampling.

### 4.1 Comparison Analysis

According to Table 2, we can conclude that STAR is a challenging task since different types of models have diverse performances, and the average level overall baselines are still not good enough.

From the results of the basic models, we can observe that the benchmark has no option biases and follows the random probability distribution naturally. The Q-type (Random) provides about $25 \%$ accuracy by randomly selecting a correct answer in four options. The Q-type (Frequent) obtain a lower performance, which indicates that the design of frequent distractors successfully influences the inference probability. With external linguistic representation as knowledge, blind models perform better than basic models only. The vision-language models can grasp the course-grained visual and language representations and achieve preliminary improvements. Nevertheless, the improvements are limited. Because simple vision-language models are good at representation but not for video question answering tasks. From simple visual-language to video QA models, about $5.03 \%$ significant increases can be observed on average accuracy. The best average accuracy achieves $36.79 \%$ by the ClipBERT. Such advantages are reasonable since they explicitly extract object interactions (LCGN) or better visual representations (HCRN and ClipBERT). We notice that although these models are better, the main improvements are from easier tasks instead of complex tasks (prediction or feasibility). These models are still struggling in reasoning tasks, although capturing vision-language interactions.

## 5 Diagnostic Model Evaluation

STAR emphasizes that ideal situation reasoning relies on visual perception, situation abstraction, and logical reasoning abilities. However, exploring the challenges and characteristics of STAR from the perspectives is not trivial. To provide more insights, we design a neuro-symbolic framework Neuro-Symbolic Situated Reasoning (NS-SR) as a diagnostic model (shown in Figure 3), which can disentangle visual perception, situation abstraction, language understanding, and symbolic reasoning. More details about implementations, evaluation, and examples are in the supplementary material.

### 5.1 Model Design

Video Parser This is a visual perception module consists of a set of detectors, where we obtain human-centric/object-centric interactions from video keyframe inputs. An object detector (Faster R-CNN, X101-FPN [37]) is used to detect objects/persons and ResNeXt-50 [43] is used to extracts visual representation for each entity. We detect relationships by VCTree with TDE-sum [39]) and extract relationship representations via GloVe [34]. A pose parser (AlphaPose [10]) is used to extract skeletons of motions. For the tasks with query actions (e.g., feasibility/sequence) in questions only, we adopt a pretrained action recognizer MoViNets [23] to recognize seen actions in the situation video as preconditions. The video parser is trained on the situation video keyframes from the training set to obtain bounding box regions or visual features.

Transformers-based Action Transition Model To distill structured cues from the dynamic realworld situations for further reasoning, we propose a transition model to process and predict the present and future situations in the form of hypergraphs.

Situation Hypergraph Encoder: NS-SR performs dynamic state transitions over situation hypergraphs. The encoder constructs "initial" situation hypergraphs by connecting detected entities or relationships and encodes graphs to a structured hypergraph token sequence. Differ from existing token representations for transformers, the token sequence describes the structures of a top-down situation hypergraph and implies situation segments, subgraph segments, and entities in graphs. Suppose given $t$ situation segments $\left\langle s^{0}, \ldots, s^{T}\right\rangle$, and each situation in time $t$ comprises multiple predicate tokens and a set of triplet tokens. Each predicate denotes an appeared atomic action $a_{j}$ where exists hyper-edges relation connecting a connected situation subgraph in the situation $s_{t}$. The triplet tokens $<h_{i}, o_{i}, r_{i}>$ are human-relationship-object interactions. Each situation segment is padding with zero tokens for a fixed length. We represent multiple types of embedding vectors to represent graph entities, hyper-edges, segments, and situations and sum their embeddings as a token embedding: token embedding, type or hyperedge embedding, situation embedding, position embedding, and segment embedding. The module details are in the supplementary material Section 4.

Dynamics Transformer Model. The dynamics model is designed to dynamically predict action states or relationships by learning the relations among the input data types in given situation videos. The model architecture is a multiple-layers of stacked transformers with down-stream task predictors. We use transformer blocks (implemented like VisualBERT [30]) to calculate self-attention scores for input token sequence with multiple heads. The attentions describe the "connections" of each potential relationship between two nodes in situation graphs (e.g., action hyper-edges or human-relationship-

![](https://cdn.mathpix.com/cropped/2024_05_26_9ad87db506347868b4e4g-09.jpg?height=748&width=1306&top_left_y=249&top_left_x=407)

Figure 3: The architecture overview of NS-SR. It use a video parser to perceive entities, relationships and human-object interactions for visual situations. The present situation is sent to a transition model to learn complete situation abstraction and predict future situations in forms of a situation hypergraph. A program parser parses the question and options into a set of nested functions. The generated hypergraph fed to a symbolic program executor to get the answer. Best viewed in color.

object triplets etc..). Because the self-attention inner structures of transformers correspond with token pairs, the whole attention over input tokens performs a dynamic relation modeling. The neighbored node connections are summed into a single node. The aggregated effect will be stored in the current state in time $t$ and applied to the prediction for the missing information in the current step or the state next time $t+1$. Such dynamic attention modeling deals with all possible relations as implicit connections. It would be more robust while relationships are unknown or some of the visual clues are not reliable. Meanwhile, we also adopt this model to predict the entities in unseen situations for prediction questions or feasibility questions.

Graph Sequence Decoder We set up three self-supervision tasks: action type prediction, human-object relationship type prediction, and masked token modeling (for objects or persons). The first two tasks use classifiers to predict action hyper-edges or relationships using MLPs with pooled global representations of all states in previous situations. Although recent perception models can achieve high accuracy in some datasets, some objects or human poses in our situation videos are blurred or invisible for the STAR videos. The masked token modeling aims to enhance the representation robustness by reconstructing their embedding vectors.

Language Parser Language Parser parses each question to a functional program [22, 47] in the form of a program sentence. The functional program (Supplementary material Figure 5 and Table 6) is composed of a series of nested operations. We defined five different types of atomic operations (e.g. query function) in the benchmark to construct step-by-step reasoning programs. We use an attentionbased Seq2Seq model [2] to parse the input questions into corresponding programs. Since our dataset questions are single-select, we use two models to parse the questions and choices individually. Each model consists of a bidirectional LSTM encoder plus an LSTM decoder [48]. We use two hidden layers of 256 hidden units and an embedding layer to get 300-dimensional word vectors for both the encoder and decoder.

Program Executor We design a Program Executor to answer questions by executing programs on discrete hypergraphs (inspired by the work in [22]). It explicitly conducts the symbolic reasoning for the answering and plays the role of the reasoning engine in NS-SR. Our executor takes the program and the predicted situation hypergraph as symbolic and discrete inputs and orderly executes the mentioned functional operations in the program on the hypergraph. We implemented the predefined operations based on the entities and relations in structured situation hypergraphs (Supplementary material Table 5 and 6). Each operation inputs certain entities or relationships and outputs the predictions as the inputs of the next reasoning step or the final answer prediction. Taking hypergraphs

| NS-SR Model Variants | Question Type |  |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  | Interaction | Sequence | Prediction | Feasibility |
| w/o perfect hypergraphs (Obj GT, Rel GT, Graph Det) | $\mathbf{1 0 0 . 0 0}$ | $\mathbf{1 0 0 . 0 0}$ | $\mathbf{1 0 0 . 0 0}$ | $\mathbf{1 0 0 . 0 0}$ |
| w/o perfect visual perception (Obj GT, Rel Det, Graph Det) | 37.41 | 46.26 | 43.44 | 43.88 |
| w/o perfect visual perception (Obj Det, Rel Det, Graph Det) | 30.89 | 38.69 | 38.49 | 38.17 |
| w/o perfect language understanding (Graph GT) | 99.79 | 31.77 | 30.24 | 29.74 |
| w/o GT | 30.88 | 39.98 | 99.98 | 99.97 |

Table 3: Performance comparison on STAR via the variants of NS-SR. GT: ground-truth, Det: detection, Obj: object, Rel: relationships, and Graph: hypergraphs.

as inputs, the reasoning starts from the cues (object, motion, or other basic data types) in questions as the initial query, then passes through all the operations iteratively and outputs the answer finally.

### 5.2 Result Analysis

Due to the modularization of NS-SR, we can explore the core challenges of STAR by an outcomecontrolled evaluation under perfect/imperfect switching settings (details in the supplementary material), as shown in Table 3. Specifically, we first use all ground-truths with a symbolic reasoning module to build an oracle model, achieving the op-line accuracy ( $100 \%$ ). This is not surprising since all questions can be answered based on perfect situation hyper-graphs and programs. Then, we remove distinct perfect conditions individually by replacing each disentangled module of NS-SR for comparisons. The final row is the performance for the version without using any ground-truths.

Situation Abstraction: This setting learns situation hyper-graphs by transformer-based action transition model but adopts ground-truths of the video parser (in the form of incomplete hypergraphs) and the program parser for simulation. Although having the perfect visual perception and reasoning logic, the model without perfect structured situation abstraction dropped about $55.95 \%$. This illustrates the situation structure abstraction challenging is the bottleneck of ideal situated reasoning in STAR.

Visual Perception: The noticeable drops show that visual perception has a significant impact on situated reasoning. The accuracy gap between the model (using object and relationship detection) and the situation abstraction variant $13.39 \%$ is smaller than the oracle version but still significant. It denotes existing vision models struggle in real-world situations, although made remarkable progress in other tasks. And situated reasoning requires well-performed visual perception. Compared to the variants between the oracle variant, the degrades of removing relationship ground-truths larger than removing object ground-truths, which means the relationship detection has more difficulties.

Language Understanding: The performance without using perfect programs has slight decrease (within 1\%) that implies the language perception in STAR is not difficult. It makes sense because we simplify the linguistic complexity and pays more attentions on visually-relevant reasoning challenges.

Without Ground-Truths: This setting uses the entire architecture in NS-SR: the video parser provides detection and poses extraction results for visual perception; the program parser provides programs parsed from given questions and options. The results are not good enough now, which shows enough remaining space for further exploration. We suggest that future directions should focus on improving the visual perception and situation abstraction on real-world videos.

## 6 Conclusions

Towards reasoning in real-world situations, we introduce a new benchmark STAR to explore how to reason accordingly. Besides perception, it integrates bottom-up situation abstraction and logical reasoning. The situation abstraction provides a unified and structured abstraction for dynamic situations, and logical reasoning adopts aligned questions, programs, and data types. We design a situated reasoning task that requires systems to learn from dynamic situations and reasonable answers for the four types of questions in specific situations: interaction, sequence, prediction, and feasibility. Our experiments demonstrate that situated reasoning is still challenging to states-of-art methods. Moreover, we design a new diagnostic model with neural-symbolic architecture to explore situated reasoning. Although the situated reasoning mechanism is not fully developed, the results show the challenges of our benchmark and indicate promising future directions. We believe STAR benchmark will open up many new opportunities for real-world situated reasoning.

## References

[1] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh. Vqa: Visual question answering. In ICCV, 2015.

[2] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

[3] Bakwc. bakwc/jamspell.

[4] M. Bloch. Situated learning: Legitimate peripheral participation. Man, 29(2):487-489, 1994.

[5] J. S. Brown, A. Collins, and P. Duguid. Situated cognition and the culture of learning. Educational researcher, 18(1):32-42, 1989.

[6] Z. Chen, J. Mao, J. Wu, K.-Y. K. Wong, J. B. Tenenbaum, and C. Gan. Grounding physical concepts of objects and events through dynamic visual reasoning. In International Conference on Learning Representations, 2021.

[7] W. J. Clancey. Situated cognition: Stepping out of representational flatland. AI Communications The European Journal on Artificial Intelligence, 4(2/3):109-112, 1991.

[8] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[9] M. Ding, Z. Chen, T. Du, P. Luo, J. B. Tenenbaum, and C. Gan. Dynamic visual reasoning by learning differentiable physics models from video and language. NeurIPS, 2021.

[10] H.-S. Fang, S. Xie, Y.-W. Tai, and C. Lu. RMPE: Regional multi-person pose estimation. In $I C C V, 2017$.

[11] C. Gan, Y. Li, H. Li, C. Sun, and B. Gong. Vqs: Linking segmentations to questions and answers for supervised attention in vqa and question-focused semantic segmentation. In ICCV, pages 1811-1820, 2017.

[12] R. Girdhar and D. Ramanan. Cater: A diagnostic dataset for compositional actions and temporal reasoning. In ICLR, 2020.

[13] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

[14] M. Grunde-McLaughlin, R. Krishna, and M. Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[15] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735$1780,1997$.

[16] Y. Hong, L. Yi, J. Tenenbaum, A. Torralba, and C. Gan. Ptr: A benchmark for part-based conceptual, relational, and physical reasoning. Advances in Neural Information Processing Systems, 34, 2021.

[17] R. Hu, A. Rohrbach, T. Darrell, and K. Saenko. Language-conditioned graph networks for relational reasoning. In $I C C V, 2019$.

[18] D. A. Hudson and C. D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

[19] Y. Jang, Y. Song, Y. Yu, Y. Kim, and G. Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In CVPR, 2017.

[20] J. Ji, R. Krishna, L. Fei-Fei, and J. C. Niebles. Action genome: Actions as compositions of spatio-temporal scene graphs. In CVPR, 2020.

[21] J. Johnson, B. Hariharan, L. Van Der Maaten, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017.

[22] J. Johnson, B. Hariharan, L. Van Der Maaten, J. Hoffman, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick. Inferring and executing programs for visual reasoning. In Proceedings of the IEEE International Conference on Computer Vision, pages 2989-2998, 2017.

[23] D. Kondratyuk, L. Yuan, Y. Li, L. Zhang, M. Tan, M. Brown, and B. Gong. Movinets: Mobile video networks for efficient video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16020-16030, 2021.

[24] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 2017.

[25] G. Lakemeyer. The situation calculus: A case for modal logic. Journal of Logic, Language and Information, 19(4):431-450, 2010.

[26] T. M. Le, V. Le, S. Venkatesh, and T. Tran. Hierarchical conditional relation networks for video question answering. In CVPR, 2020.

[27] J. Lei, L. Li, L. Zhou, Z. Gan, T. L. Berg, M. Bansal, and J. Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7331-7341, 2021.

[28] J. Lei, L. Yu, M. Bansal, and T. L. Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018.

[29] J. Lei, L. Yu, T. L. Berg, and M. Bansal. Tvqa+: Spatio-temporal grounding for video question answering. arXiv preprint arXiv:1904.11574, 2019.

[30] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.

[31] J. McCarthy. Situations, actions, and causal laws. Technical report, STANFORD UNIV CA DEPT OF COMPUTER SCIENCE, 1963.

[32] J. Mun, P. Hongsuck Seo, I. Jung, and B. Han. Marioqa: Answering questions by watching gameplay videos. In $I C C V, 2017$.

[33] K. Omelianchuk, V. Atrasevych, A. Chernodub, and O. Skurzhanskyi. Gector-grammatical error correction: Tag, not rewrite. arXiv preprint arXiv:2005.12592, 2020.

[34] J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532-1543, 2014.

[35] H. Prendinger and G. Schurz. Reasoning about action and change. Journal of logic, language and information, 5(2):209-245, 1996.

[36] R. Reiter. The frame problem in the situation calculus: A simple solution (sometimes) and a completeness result for goal regression. In Artificial and Mathematical Theory of Computation, pages 359-380. Citeseer, 1991.

[37] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015.

[38] G. A. Sigurdsson, O. Russakovsky, and A. Gupta. What actions are needed for understanding human actions in videos? In Proceedings of the IEEE international conference on computer vision, pages 2137-2146, 2017.

[39] K. Tang, Y. Niu, J. Huang, J. Shi, and H. Zhang. Unbiased scene graph generation from biased training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3716-3725, 2020.

[40] M. Tapaswi, Y. Zhu, R. Stiefelhagen, A. Torralba, R. Urtasun, and S. Fidler. Movieqa: Understanding stories in movies through question-answering. In CVPR, 2016.

[41] X. Wang, A. Farhadi, and A. Gupta. Actions transformations. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2658-2667, 2016.

[42] M. S. Winslett. Reasoning about action using a possible models approach. Department of Computer Science, University of Illinois at Urbana-Champaign, 1988.

[43] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. arXiv preprint arXiv:1611.05431, 2016.

[44] D. Xu, Z. Zhao, J. Xiao, F. Wu, H. Zhang, X. He, and Y. Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645-1653, 2017.

[45] A. Yang, A. Miech, J. Sivic, I. Laptev, and C. Schmid. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1686-1697, 2021.

[46] G. R. Yang, I. Ganichev, X.-J. Wang, J. Shlens, and D. Sussillo. A dataset and architecture for visual reasoning with a working memory. In ECCV, 2018.

[47] K. Yi, C. Gan, Y. Li, P. Kohli, J. Wu, A. Torralba, and J. B. Tenenbaum. Clevrer: Collision events for video representation and reasoning. In ICLR, 2020.

[48] K. Yi, J. Wu, C. Gan, A. Torralba, P. Kohli, and J. B. Tenenbaum. Neural-symbolic vqa: Disentangling reasoning from vision and language understanding. arXiv preprint arXiv:1810.02338, 2018.

[49] R. Zellers, Y. Bisk, A. Farhadi, and Y. Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6720-6731, 2019.

[50] Y. Zhu, O. Groth, M. Bernstein, and L. Fei-Fei. Visual7w: Grounded question answering in images. In CVPR, 2016.
</end of paper 4>


