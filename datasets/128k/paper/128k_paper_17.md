<paper 0>
# WARM: On the Benefits of Weight Averaged Reward Models 

Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, Johan Ferret<br>Google DeepMind


#### Abstract

Aligning large language models (LLMs) with human preferences through reinforcement learning (RLHF) can lead to reward hacking, where LLMs exploit failures in the reward model (RM) to achieve seemingly high rewards without meeting the underlying objectives. We identify two primary challenges when designing RMs to mitigate reward hacking: distribution shifts during the $\mathrm{RL}$ process and inconsistencies in human preferences. As a solution, we propose Weight Averaged Reward Models (WARM), first finetuning multiple RMs, then averaging them in the weight space. This strategy follows the observation that fine-tuned weights remain linearly mode connected when sharing the same pre-training. By averaging weights, WARM improves efficiency compared to the traditional ensembling of predictions, while improving reliability under distribution shifts and robustness to preference inconsistencies. Our experiments on summarization tasks, using best-of- $N$ and RL methods, shows that WARM improves the overall quality and alignment of LLM predictions; for example, a policy RL fine-tuned with WARM has a $79.4 \%$ win rate against a policy RL fine-tuned with a single RM.


Keywords: Alignment, RLHF, Reward Modeling, Model Merging

## 1. Introduction

Reward modeling. Conversational assistants such as Gemini [1] or GPT-4 [2] have revolutionized the AI community and beyond. These LLMs are capable of completing novel and intricate tasks, including mathematics, coding, and tool use [3]. These advancements are underpinned by a systematic three stage training procedure: pre-training by next token prediction [4, 5, 6], supervised fine-tuning (SFT) to learn to follow instructions $[7,8,9]$, and ultimately, reinforcement learning (RL) to maximize a reward encapsulating the desired behaviors [10]. However, defining such rewards for real-world tasks is non-trivial [11]. In reinforcement learning from human feedback (RLHF) [12, 13, 14, 15], rewards are reward models (RMs), trained on binary preference datasets to emulate human judgment. The enhancement of LLM capabilities from RL is strongly tied to the quality of the RMs [16].

Reward hacking. Particularly insidious in RLHF [17, 18] is the reward hacking issue [19, 20, 21, 22] (a.k.a. reward overoptimization), arising from reward misspecification [23, 24] between the proxy $\mathrm{RM}$ and actual human preferences. While optimizing for the RM initially provides improvements, in later stages the policy (i.e., the LLM being trained) usually learns to exploit loopholes in the RM and achieves high rewards without truly fulfilling the intended objectives, as illustrated in Figure 1(b). This reward hacking phenomenon poses numerous issues. First, it degrades performances, manifesting as linguistically flawed [25] or unnecessarily verbose [26] outputs, which do not reflect true human preferences. Second, it complicates checkpoint selection due to the unreliability of the proxy RM, echoing Goodhart's Law [27]: "when a measure becomes a target, it ceases to be a good measure". Third, it can engender sycophancy $[28,29]$ or amplify social biases, reflecting the limited and skewed demographics of feedback providers [30, 31]. Lastly and most critically, misalignment [32, 33] due to reward hacking can escalate into safety risks $[19,34,35]$, in particular given the rapid integration of LLMs in everyday life and critical decision-making. Such concerns underscore the need to mitigate reward hacking to ensure the beneficial and safe deployment of LLMs.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-02.jpg?height=540&width=1062&top_left_y=341&top_left_x=220)

(a) WARM procedure with $M=3$.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-02.jpg?height=534&width=554&top_left_y=364&top_left_x=1299)

(b) WARM mitigates reward hacking.

Figure $1 \mid$ Figure 1 (a) illustrates the alignment process with WARM. From a SFT-ed LLM, we apply RL fine-tuning to optimize a proxy reward model (RM), in line with RLHF [12]. The innovation of WARM lies in the design of the proxy RM, which is the weight average (WA) of $M$ individual RMs, each fine-tuned from a shared pre-trained LLM on the same preference dataset, but with slight differences such as diverse hyperparameters. This WA approach is efficient, while enhancing the reliability under distribution shifts and robustness under inconsistent preferences. Figure 1(b) showcases the impact during RL alignment. The control reward (detailed in Section 5) initially increases but eventually deteriorates, a phenomenon called reward hacking [19]. However, when WARM serves as the proxy RM, increasing $M$ (the number of averaged RMs) significantly improves absolute results while delaying the collapse, as indicated by the control rewards maintaining higher values for longer during training. Same plot with KL as the $x$-axis in Figure 8(a) and with label corruption in Figure 18.

Challenges. Two primary challenges underlie reward hacking. The first major issue are the distribution shifts encountered by the RM [36, 37]. Indeed, the generations from the policy might deviate substantially from those in the offline preference dataset, posing an out-of-distribution (OOD) challenge. Moreover, those distribution shifts are accentuated by the policy drift during the RL procedure: the policy moves away from its SFT initialization, continually altering the distribution of predictions the RM needs to interpret reliably. Second, preferences are inconsistent: the binary labels in the preference dataset are noisy. Indeed, human labelers often rely on simpler criteria (length, bullet points, politeness) over more nuanced indicators. Moreover, errors can be exacerbated for complex tasks requiring specific expertise [38], and because of the multi-objective nature of alignment [39] requiring handling the heterogeneity of human opinions. Overall, this results in a low inter-labeler agreement ( $72.6 \%$ for InstructGPT [40]), altering the robustness of the RM.

Goal and ensembling baseline. Designing good RMs must meet a tripartite objective: guiding RL efficiently, reliably scoring generations despite the distribution shifts, and providing robust signals amidst label noise. To address these challenges, the seminal work on RLHF from Christiano et al. [12] and more recent works [41, 42] leveraged prediction ensembling (ENS) [43], averaging the rewards from multiple RMs. ENS improves the reliability of the reward and mitigates hacking risks [41, 42]. Yet, ENS suffers from memory and inference overhead causing efficiency challenges; we will also show that ENS fails to improve robustness to label noise in the preference datasets.

WARM. In this paper, we propose weight averaged reward models (WARM), a simple, efficient and scalable strategy to obtain a reliable and robust RM by combining multiple RMs. Starting from a shared pre-trained LLM, we launch multiple RM fine-tunings: in practice, the different runs have different hyperparameters (as in grid search), and see the preference data in different orders, thus
leading to diverse RMs. A key contribution is how the different RMs are merged: by linear interpolation in the weight space. This follows the findings from the linear mode connectivity (LMC) [44, 45] and weight averaging (WA) literature [46, 47, 48]: under shared pre-training, the different weights can be linearly interpolated despite the non-linearities in the architecture.

On the benefits of WARM. Firstly, WARM stands out for its efficiency and practicality. By requiring a single model at inference time, it provides a scalable approximation to the traditional, costlier ensembling of predictions, without its memory and inference burdens. Secondly, WARM improves reliability by inheriting from the generalization abilities of WA under distribution shifts, a quality well-documented in the OOD literature for supervised learning [47, 48, 49]. Lastly, WARM improves robustness to label corruption. We show that WA selects the invariant predictive mechanisms [50, 51] across different runs [52, 53], thus naturally diminishing the memorization of corrupted samples, occurring in each run in different ways. In contrast, ENS simply memorizes the corrupted samples. We also explain why reducing memorization when modeling noisy preferences enhances stability in the RL process. These multifaceted benefits of WARM are further explored in Section 4.

We summarize our contributions as follows.

1. Innovation in reward modeling. We introduce WARM, the first instance of weight averaging for reward modeling. This novel strategy efficiently mitigates reward hacking, improves reliability under distribution shifts and robustness to label corruption.
2. Theoretical and empirical insights into weight averaging. We validate linear mode connectivity for reward models trained on binary preference datasets. Moreover, we reveal a key difference between weight and prediction averaging, that appears clearly under label corruption; weight averaging only maintains the invariant predictive mechanisms across runs, thereby diminishing memorization and enhancing the focus on generalizable features.

Our experiments on summarization tasks in Section 5 confirm that WARM improves performance without any memory or inference overhead, either when used as the reward selector in best-of- $N$, or as the proxy RM in RL. WARM mitigates reward hacking, and thus provides better downstream policies; specifically, it leads to a win rate of $79.4 \%$ (according to the preference oracle metric) against a policy trained with a standard RM.

## 2. Context and challenges

### 2.1. Context

LLMs. We consider an LLM $f_{\theta}$ of a fixed non-linear architecture parameterized by $\theta$, usually a Transformer with attention layers [54]. It defines a policy by mapping prompt inputs $x$ to $f_{\theta}(x)$. Following the foundation model paradigm [55] and the success of transfer learning [56], the weights $\theta$ are first pre-trained [4] on the vast amount of web data into $\theta^{p t}$, before supervised fine-tuning (SFT) [7] to learn to follow instructions into $\theta^{\text {sft }}$. However, the high cost and limited scope of instruction data (i.e., prompts and responses) can create a misalignment [19, 32, 33] between the LLM and its intended application. Reinforcement learning (RL) as a third step in the training process of LLMs was shown to help alignment of LLMs with the intended usage [40].

RMs. A notable aspect of RL is the absence of supervised samples to be imitated by the policy; instead, the focus shifts to maximizing the reward of generated samples, that should measure their quality. The challenge is that the oracle reward, perfectly encapsulating the desired behaviors, is not given by the environment. The key innovation from RLHF [12] is that this reward is the output of a reward model (RM), trained in a supervised way to predict and thus reflect human preferences. Specifically,
an RM is an LLM $r_{\phi}$ parameterized by $\phi$, predicting a single scalar as the reward $r_{\phi}(x, y)$ for a prompt $x$ and generation $y$. The weights $\phi$ are usually initialized from $\left(\theta^{s f t}, \omega\right)$, where the final linear layer $\omega$ is added on top of the extracted features from the SFT model $\theta^{s f t}$. Then $\phi$ is trained on a preference dataset $\mathcal{D}_{\text {train }}=\left\{x_{d}, y_{d}^{+}, y_{d}^{-}\right\}_{d=1}^{D}$ where the generation $y_{d}^{+}$has been preferred over $y_{d}^{-}$to continue $x_{d}$. Usually human labelers evaluate those generations, but recent works on RLAIF [57, 58] showed that similar performances can be obtained by prompting an LLM for AI feedback. Following the Bradley-Terry [59] assumption about the distribution of preferences, and by framing the problem as binary classification, the maximum likelihood principle motivates learning $\phi$ by minimizing the following negative log-likelihood loss (where $\sigma$ is the logistic function):

$$
\begin{equation*}
\mathcal{L}_{R}\left(r_{\phi}, \mathcal{D}_{\text {train }}\right)=-\mathbb{E}_{\left(x, y^{+}, y^{-}\right) \in \mathcal{D}_{\text {train }}}\left[\log \sigma\left(r_{\phi}\left(x, y^{+}\right)-r_{\phi}\left(x, y^{-}\right)\right)\right] \tag{1}
\end{equation*}
$$

Reward inference. With this RM, the literature suggests applying any kind of RL algorithm (usually REINFORCE [60] or PPO [61]) to fine-tuned $\theta^{s f t}$ into $\theta^{r l}$, as analyzed in Section 5.2. A training-free alternative is best-of- $N$ (BoN) sampling, analyzed in Section 5.1, which returns the generation that has the highest reward among $N$ generations from $\theta^{\text {sft }}$. Both methods aim to align the policy with human preferences. Yet, the reward misspecification [23] between the proxy RM and the true human preferences can lead to reward hacking [19, 20, 21, 22], where the policy exploits loopholes in the proxy RM to artificially increase the score without matching human preferences.

### 2.2. Challenges in reward modeling

When handling rich inputs such as text, or when assessing complex behaviours, designing rewards aligned with human preferences is a complex challenge for two main reasons, described below.

Distribution shifts. The primary challenge is the distribution shifts resulting from the offline nature of preference data. Indeed, the generations in the preference dataset and those from the policy $\theta^{\text {st }}$ do not necessarily follow the same distributions, and the shifts can become even more pronounced due to model drift during RL. The OOD generalization literature has extensively analyzed the repercussions of these shifts. Firstly, they often lead to a reduction in performance [62, 63]. RMs (of limited capacity) trained on narrow data distributions may rely on spurious correlations [51] or a limited number of features [64], thus failing when encountering OOD examples [65, 66]. Secondly, they complicate the selection of RMs, as ID validation metrics may poorly correlate with real-world OOD performances [67, 68] and the ability to guide the RL [41]. Lastly, RMs can become poorly calibrated [69] in OOD scenarios [70, 71], and predict more extreme values as rewards. Such miscalibration exacerbates the problem in a negative feedback loop, further intensifying model drift and distribution shifts. In conclusion, limited data coverage during reward modeling reduces the reliability of the RM and facilitates reward hacking [36] in regions where the RM is badly specified.

Inconsistent preferences. The second major challenge is the label noise in preference datasets. Human labelers, often grappling with fatigue, misunderstandings [72, 73] and imperfect incentives [74], might default to simpler criteria such as length, bullet points, or politeness rather than more causal indicators. This tendency is exacerbated for complex tasks [38] or when considering multiple objectives, ranging from harmlessness [75] to engagement [76] and representing the heterogeneity of human opinions. Consequently, these factors lead to low inter-rater agreement, where human data appears as an imperfect representation of the underlying ground truth [77, 78]. To mitigate these issues, there has been a shift towards AI-generated preferences [57, 58], which, while reducing human labor costs, introduces its own set of noise and failure cases, such as sensitivity to prompting strategies $[79,80]$. These layers of noise and inconsistency challenge the robustness of the RM, and its ability to provide stable signals.

With this in mind, a good RM should ideally satisfy the three following properties.

Property 1: efficiency. The RM should incur no memory or inference overhead. Then the policy can be optimized efficiently.

Property 2: reliability. The RM should reliably reward predictions despite the distribution shifts. Then the policy can explore away from its initialization while relying on the RM.

Property 3: robustness. The RM should be robust to the label inconsistencies in binary preferences. Then the policy can learn from robust signals given by the RM.

### 2.3. Existing approaches

To tackle those issues, previous works have explored a few research directions, further detailed in our related work from Appendix A.2. During RL, the standard strategy is to encourage the policy to remain close to its SFT initialization with Kullback-Leibler (KL) regularization [81, 82]; KL reduces model drift [83, 84] but can cause underfitting and adds an extra hyperparameter (the regularization strength $\alpha$ ). Collecting, labelling and then training on new data (reflecting the evolving policy) can improve the reliability of the RM [16]. Yet it poses significant efficiency challenges due to the continuous requirement for human annotation and computational resources. In contrast, active learning strategies [85, 86] proactively enrich the preference dataset by seeking out a diverse set of generations and potential failure cases. Concurrent work [87] suggests applying label smoothing and flipping. Finally, and most similar to WARM, prediction ensembling (ENS) [43] strategies average the logits from $M$ RMs. From a bias-variance perspective [88], ENS reduces the variance term when members are sufficiently diverse [89], and thus favors reliability under distribution shifts where variance is the key issue [47]. From a RL perspective, ENS was shown to mitigate hacking risks [12, 41, 42]. Despite its advantages, ENS faces efficiency challenges; the memory and inference costs grow linearly with $M$, making ENS incompatible with the scaling trend in RMs, where larger architectures consistently perform better [90]. Moreover, we will also show in Section 4.2 that ENS fails to improve robustness to preference inconsistencies.

## 3. WARM

### 3.1. Weight averaging of reward models

Facing those challenges in reward modeling and the limitations from existing approaches, we propose Weight Averaged Reward Models (WARM). WARM is a simple and efficient strategy that combines multiple models without the memory and inference overheads of prediction ensembling, enhancing reward reliability (under distribution shifts) and robustness (amidst noisy preference dataset). WARM is illustrated in Figure 1(a) and described below.

1. Shared pre-trained initialization. For a given pre-trained LLM, each RM is initialized from $\left(\theta^{\text {sft }}, \omega\right)$ combining SFT weights and a linear probed [91] classifier.
2. Diverse fine-tunings. We run $M$ RM fine-tunings, optimizing Equation (1) with diverse hyperparameters (as in a grid search), yielding $M$ weights $\left\{\phi_{i}\right\}_{i=1}^{M}$.
3. Weight averaging. We average those $M$ weights together to form $\phi^{\mathrm{WARM}}=\frac{1}{M} \sum_{i=1}^{M} \phi_{i}$.

Then $r_{\phi^{\text {WARM }}}$ serves as the proxy RM to guide the RL procedure, as efficiently as an individual RM, but with the enhanced reliability and robustness provided by the WA strategy, that leverages the strengths and mitigates the weaknesses of the individual RMs.

### 3.2. Linear mode connectivity

Compared to ENS, the main difference lies in how WARM combines the different RMs: we do so through linear interpolation in the weight space. It relies on the linear mode connectivity (LMC) [44, 45] property across fine-tuned weights, i.e., the fact that the accuracy of the interpolated model is at least as good as the interpolation of the individual accuracies. Precisely, by defining the pairwise accuracy of an $\operatorname{RM} r_{\phi}$ w.r.t. a dataset $\mathcal{D}$ as $\operatorname{Acc}\left(r_{\phi}, \mathcal{D}\right)=\mathbb{E}_{\left(x, y^{+}, y^{-}\right) \in \mathcal{D}}\left[\mathbb{1}_{r_{\phi}\left(x, y^{+}\right) \geq r_{\phi}\left(x, y^{-}\right)}\right]$, the following Observation 1 underpins the success of WARM.

Observation 1 (LMC). Given two fine-tuned weights $\phi_{1}$ and $\phi_{2}$ with a shared pre-training and a test

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-06.jpg?height=52&width=591&top_left_y=945&top_left_x=207)

$$
\begin{equation*}
\operatorname{Acc}\left(r_{(1-\lambda) \cdot \phi_{1}+\lambda \cdot \phi_{2}}, \mathcal{D}_{\text {test }}\right) \geq(1-\lambda) \times \operatorname{Acc}\left(r_{\phi_{1}}, \mathcal{D}_{\text {test }}\right)+\lambda \times \operatorname{Acc}\left(r_{\phi_{2}}, \mathcal{D}_{\text {test }}\right) \tag{2}
\end{equation*}
$$

We empirically validate this LMC in Figure 3, by evaluating interpolated RMs on OOD test samples. This follows similar observations for multi-class classification in the context of computer vision [44, 45], which led to a plethora of weight averaging (WA) works such as the model soups [46, 47, 48] variants (detailed in our related work in Appendix A.1).

Remark 1 (Importance of pre-training and linear probing). The efficacy of WA can be surprising given the non-linearities [54] and permutation symmetries [92] in deep neural network architectures. WA is actually possible only because of the shared pre-training which constrains the divergence during fine-tunings [45], such as the weights remain in convex regions of the loss valley [93]. In contrast, the LMC does not hold when training weights from scratch [45], even if the random initialization is shared. For these reasons and to facilitate the LMC, we follow [47, 48] and use linear probing to initialize the classifier $\omega$; compared to random initialization, such linear probing prevents feature distortion [91].

### 3.3. Sources of diversity

On one hand, WARM requires shared pre-training so that the fine-tuned weights remain linearly connected. On the other hand, weights must not be identical: actually, the diversity across those fine-tuned weights significantly contributes to the accuracy gains observed in WA [47]. Overall, an effective WARM requires a delicate trade-off between ensuring LMC and diversity across weights.

In practice, we use the following sources of diversity [94], leading the RM fine-tunings to diverse yet linearly connected models. First, the different fine-tunings see the data samples in different orders. Second, we sample slightly different hyperparameters, notably different learning rates and dropout probabilities, as detailed in Appendix B.3. Third, we investigate a new source of diversity in initialization named Baklava, illustrated in Figure 2. Specifically, we initialize the RMs' featurizers from different checkpoints $\left\{\theta_{i}^{s f t}\right\}_{i=1}^{M}$ collected along a given SFT trajectory. Baklava relaxes the shared initialization constraint from model soups [46] to simply sharing the same pre-training: Baklava is actually an efficient alternative to model ratatouille [48] but without the need of multiple auxiliary tasks. Overall, Baklava increases diversity compared to only initializing from the last SFT checkpoint, while adhering to the shared pre-training requisite for LMC, without incurring any overhead.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-07.jpg?height=318&width=989&top_left_y=289&top_left_x=542)

Figure 2 | Baklava diversity procedure. Starting from a pre-trained LLM $\theta^{p t}$, we consider different checkpoints $\left\{\theta_{i}^{s f t}\right\}_{i=1}^{M}$ along a single SFT run (dashed arrow -- $)$ collected at different number of SFT training steps. Those checkpoints serve as initializations for $M$ RM fine-tunings on the preference dataset (thick solid arrows $\longrightarrow$ ) to learn the $\left\{\phi_{i}\right\}_{i=1}^{M}$. Finally, those RMs are weight averaged (dotted arrows $\cdots \cdots \Rightarrow$ ) into the final model $\phi^{\text {WARM }}$. Following the culinary analogy from model soups [46] and model ratatouille [48], we named this method Baklava because of its diamond geometric shape.

Remark 2 (Moving average). Following stochastic weight average [95] or moving average [96], we also tried to average checkpoints collected along a single RM fine-tuning. Though interesting because less costly for training, the lower results in Figure 3(a) suggest that the accuracy-diversity trade-off was not favorable: incorporating early checkpoints would compromise individual accuracies, and considering only later checkpoints would not bring the necessary diversity. As a result, we opted to use in WARM only the last checkpoint from each RM fine-tuning.

## 4. On the benefits of WARM

We now explore the properties and benefits from the WARM strategy, previously described in Section 3. We ground our analysis on the empirical comparison between WA and ENS for reward modeling, and a novel general theoretical comparison in Section 4.3.

Experimental setup. We leverage the TL;DR summarization benchmark [97], a standard in reward modeling for LLMs, that we briefly describe below and further detail in Appendix B. The goal of the RMs is to score summaries such as they are ranked properly. In training, we use the dataset $\mathcal{D}_{\text {train }}$ from Stiennon et al. [14] where the candidate summaries are generated by GPT-3 [6] variants. To obtain the labels, we follow the RLAIF procedure from [58], where a PaLM-L [98] is prompted with chain-of-thought [99] to generate feedback mimicking human preferences. This strategy performs similarly to human labelers with similar inter-agreement, and will be useful in Section 5 as an oracle metric. The RMs are PaLM-XXS models, pre-trained and SFT-ed on the preferred summaries from $\mathcal{D}_{\text {train }}$, on which we plug a linear probed [91] classification layer. We train the RMs for 10k steps on $\mathcal{D}_{\text {train }}$, with hyperparameters and procedure detailed in Appendix B.3. We report accuracies of those RMs on a novel out-of-distribution (OOD) test dataset $\mathcal{D}_{\text {ood }}$ with $92 \mathrm{k}$ pairwise comparisons where the summaries are generated by multiple PaLM-XS policies with high temperature, some of which are pre-trained only, others SFT-ed and others RLHF-ed.

## 4.1. $1^{\text {st }}$ order analysis: weight averaging for reliable and more efficient ensembling

Previous works $[46,47,95]$ have argued that the best way to understand WA is as an efficient approximation of ENS, as clarified in Observation 2.

Observation 2 (WA and ENS: $1^{\text {st }}$ order analysis). Weight averaging and prediction ensembling perform similarly: i.e., for all $\lambda \in[0,1]$ and a test dataset $\mathcal{D}_{\text {test }}$,

$$
\begin{equation*}
\operatorname{Acc}\left(r_{(1-\lambda) \cdot \phi_{1}+\lambda \cdot \phi_{2}}, \mathcal{D}_{\text {test }}\right) \approx \operatorname{Acc}\left((1-\lambda) \times r_{\phi_{1}}+\lambda \times r_{\phi_{2}}, \mathcal{D}_{\text {test }}\right) \tag{3}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=499&width=1656&top_left_y=276&top_left_x=200)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=362&width=394&top_left_y=293&top_left_x=220)

(a) 1 RM fine-tuning at 2 different training steps.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=366&width=392&top_left_y=294&top_left_x=632)

(b) 2 RM fine-tunings with shared config.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=360&width=391&top_left_y=297&top_left_x=1044)

(c) 2 RM fine-tunings with different learning rates.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=362&width=391&top_left_y=293&top_left_x=1461)

(d) 2 RM fine-tunings with different inits: Baklava.

Figure 3 | Experiments under distribution shifts validating Observations 1 and 2 on the TL;DR summarization benchmark [97]. We report the accuracies on $\mathcal{D}_{\text {ood }}$ when interpolating between two RM weights $\phi_{1}$ and $\phi_{2}$ with the coefficient $\lambda$ sliding between 0 and 1. WA stands for weight averaging $r_{(1-\lambda) \cdot \phi_{1}+\lambda \cdot \phi_{2}}$ while ENS combines the predictions $(1-\lambda) \times r_{\phi_{1}}+\lambda \times r_{\phi_{2}} ;$ Diag is the interpolated accuracy $(1-\lambda) \times \operatorname{Acc}\left(r_{\phi_{1}}\right)+\lambda \times \operatorname{Acc}\left(r_{\phi_{2}}\right)$. We consider sources of increasing diversity [94] between $\phi_{1}$ and $\phi_{2}$ : in Figure 3(a), they are collected at different number of training steps (8k and 10k) along a single RM fine-tuning; in Figure 3(b), they are from two independant RM fine-tunings, with the exact same config, but seeing the data in different orders; in Figure 3 (c), they have different learning rates (1e-4 and 4e-5); in Figure 3(d), they are initalized from different SFT checkpoints collected at different number of SFT steps (8k and 12k), per Baklava introduced in Figure 2.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=446&width=1665&top_left_y=1242&top_left_x=207)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=360&width=391&top_left_y=1256&top_left_x=221)

(a) Train (corrupt).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=357&width=394&top_left_y=1261&top_left_x=631)

(b) Train (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=360&width=391&top_left_y=1256&top_left_x=1044)

(c) Validation (ID).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-08.jpg?height=365&width=397&top_left_y=1254&top_left_x=1458)

(d) Test (OOD).

Figure 4 | Corruption experiment validating Observation 3. We consider $\phi_{1}$ and $\phi_{2}$, two RMs fine-tuned independently with the same config as in Figure 3(b), but this time with $25 \%$ of the training labels corrupted. We then report the performances of their WA and ENS on the different data subsets. We observe that WA reduces memorization of the corrupted labels in Figure 4(a), and still performs slightly worse than ENS on the clean training samples in Figure 4(b); yet, the performances of WA w.r.t. ENS improves as we move away from the training distribution, in particular on $\mathcal{D}_{\text {ood }}$ in Figure 4(d) where WA generalizes better.

Theoretically, a simple Taylor expansion can justify this similarity when $\left\|\phi_{1}-\phi_{2}\right\| \ll 1$. Empirically, this is validated in Figure 3 where the accuracy curves on $\mathcal{D}_{\text {ood }}$ for WA and ENS closely match. This similarity justifies that WA is a variance reduction method; then, because variance is the dominant issue under distribution shifts [47], this explains the significant gains in Figure 3 over the individual RMs $\phi_{1}$ and $\phi_{2}$ (validating Observation 1), in particular when weights are sufficiently diverse. This suggests improved reliability in WARM, with efficiency benefits over ENS: indeed, WA maintains a single set of weights, removing the memory and inference overheads from ENS.

## 4.2. $2^{\text {nd }}$ order analysis: weight averaging for more robust ensembling

A surprising fact remains unexplained. WA is slightly superior to ENS under distribution shifts, which one can see on the plots from Figure 3, and more consistently in Figure B. 1 from model soups [46] or in Figure 1 from DiWA [47]. More generally, WA is the state-of-the-art strategy for OOD generalization, consistently outperforming ENS; yet, this was not explained in previous works, thus urging for new insights about the difference between WA and ENS.

Corruption setup. To refine our understanding on the difference between WA and ENS, we propose a new setup where $25 \%$ of the binary labels are swapped in training. We then report the per-subset accuracies on Figure 4, enriched in Appendix C. 1 and aggregated in Figure 5. On the corrupted subset of training data, the accuracy curve for WA is below the expected accuracies, while it is above on all other subsets. More precisely, we make the following Observation 3.

Observation 3 (WA and ENS: $2^{\text {nd }}$ order analysis). The accuracy gains of WA over ENS grow as data moves away from the training distribution.

- WA $\ll E N S$ on train corrupt: WA is far worse than ENS on train samples with swapped labels, showing reduced memorization and improved robustness to label corruption.
- WA $\leq$ ENS on train clean: WA is worse than ENS on train samples with correct labels.
- WA $\approx E N S$ on ID val: WA is better or similar to ENS on samples without distribution shifts.
- WA $\geq E N S$ on $O O D$ test: WA is far better than ENS on test samples from new distributions, showing better reliability under distribution shifts.

Overall, this suggests that weight averaging memorizes less and generalizes better than ensembling predictions.

### 4.3. Weight averaging enforces invariance across runs

We now provide theoretical support to this Observation 3. In brief, our simplifying assumptions suggest that WA acts as a regularization towards the predictive mechanisms that are invariant across runs, i.e., learned simultaneously in each independent run. Then, in contrast with ENS, WA would improve robustness to corruption because it would underweight the run-specific features (with low probability of being learned) inducing memorization.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-09.jpg?height=497&width=534&top_left_y=1331&top_left_x=1315)

Figure 5 | Histograms of the differences in accuracy between WA and ENS on different data subsets.

Setup. We follow Lin et al. [53], and consider a simplified binary classification setup with labels $y \in\{-1,1\}$, related to $F$ features $\left\{z^{j}\right\}_{j=1}^{F}$ such as $z^{j} \in \mathbb{R}^{d}$. From inputs $x$, we train a binary classifier $r(x)=\omega^{\top} f(x)$. Following [53], we make three key assumptions. First, features orthogonality: we assume that $\left\{z^{j}\right\}_{j=1}^{F}$ are orthogonal, i.e., $\left(z^{j}\right)^{\top} z^{j^{\prime}}=0$ when $j \neq j^{\prime}$. Second, input as bag of features: we assume that the input $x=\left[x^{j}\right]_{j=1}^{F} \in \mathbb{R}^{F \times d}$ can be represented as the concatenation of $x^{j}$ generated by $x^{j} \sim \mathcal{N}\left(y \cdot z^{j}, \sigma \cdot \mathbf{I}_{d}\right)$ with $\sigma \ll 1$. Finally, the binary featurizer assumption: we consider that the featurizer $f=\left[f^{j}\right]_{j=1}^{F} \in\{0,1\}^{F}$ is a binary selector of the features that make the input. For example, if $y=1, F=3, x \approx\left[z^{1}, z^{2}, z^{3}\right]$, and $f=[1,0,1]$ learns to extract the first and third features, then $f(x) \approx z^{1}+z^{3}$. We denote $p_{j}$ the probability that the featurizer $f$ learns to use the $j$-th feature dimension (associated with $z^{j}$ ); this means $f^{j}$ is 1 with probability $p_{j}$ and 0 otherwise. Moreover, for infinite training samples and under some constraint on $\sigma$, Lemma 5 in [53] proved that, to learn $r=\omega^{\top} f$, the optimal linear fit $\omega$ on the features selected from $f$ would be $\omega=\sum_{j=1}^{F} f^{j} \cdot z^{j}$.

Results. We consider $M$ RMs $\left\{r_{i}=\omega_{i}^{\top} f_{i}\right\}_{i=1}^{M}$, and compare the limit behaviours of their prediction ensembling $r_{M}^{E N S}$ and weight averaging $r_{M}^{W A}$ when $M \rightarrow \infty$. In this limit case, the averaged prediction $r_{M}^{E N S}=\frac{1}{M} \sum_{i=1}^{M} \omega_{i}^{\top} f_{i}$ for an input $x$ from label $y$ tends towards the expected prediction $\mathbb{E}[r(x)]=\mathbb{E}\left[\omega^{\top} f(x)\right]=\mathbb{E}_{\left\{f^{j}\right\}_{j=1}^{F}}\left[\left(\sum_{j=1}^{F} f^{j} \cdot z^{j}\right)^{\top}\left(\sum_{j^{\prime}=1}^{F} f^{j^{\prime}} \cdot x^{j^{\prime}}\right)\right] \approx y \cdot \sum_{j=1}^{F} p_{j} \cdot\left|z^{j}\right|^{2}$, using $x^{j^{\prime}} \approx y \cdot z^{j^{\prime}}$ thus $\left(z^{j}\right)^{\top} x^{j^{\prime}} \approx 0$ when $j \neq j^{\prime}$, and $\left(f^{j}\right)^{2}=f^{j}$.

$$
\begin{equation*}
r_{M}^{E N S}(x) \underset{M \rightarrow \infty}{\longrightarrow} \mathbb{E}[r(x)] \approx y \cdot \sum_{j=1}^{F} p_{j} \cdot\left|z^{j}\right|^{2} \tag{4}
\end{equation*}
$$

In contrast, when considering $r_{M}^{W A}=\left(\frac{1}{M} \sum_{i=1}^{M} \omega_{i}\right)^{\top}\left(\frac{1}{M} \sum_{i=1}^{M} f_{i}\right)$ with $M \rightarrow \infty$, we have $\frac{1}{M} \sum_{i=1}^{M} f_{i} \xrightarrow[M \rightarrow \infty]{\longrightarrow} \mathbb{E}[f]=\left[p_{j}\right]_{j=1}^{F}$ and $\frac{1}{M} \sum_{i=1}^{M} \omega_{i} \xrightarrow[M \rightarrow \infty]{\longrightarrow} \mathbb{E}[\omega]=\sum_{j=1}^{F} p_{j} \cdot z^{j}$, and thus:

$$
\begin{equation*}
r_{M}^{W A}(x) \underset{M \rightarrow \infty}{\longrightarrow}\left(\sum_{j=1}^{F} p_{j} \cdot z^{j}\right)^{\top}\left(\sum_{j^{\prime}=1}^{F} p_{j^{\prime}} \cdot x^{j^{\prime}}\right) \approx y \cdot \sum_{j=1}^{F} p_{j}^{2} \cdot\left|z^{j}\right|^{2} \tag{5}
\end{equation*}
$$

Interpretation. For ENS, the coefficient for a given feature is $\boldsymbol{p}_{j}$, the same as the probability of this information being used by any individual network. In contrast, WA involves the square of the probability $\boldsymbol{p}_{j}^{2}$. Thus WA reduces the reliance on features with low probability, related to minor specific information (such as noise or context) which can be used to fit the corrupted training samples; this would reduce memorization, and thus explains the robustness of WA under label corruption. Reciprocally, WA tends to prioritize the most probable features, favoring the mechanisms that are consistently learned, in other words the mechanisms invariant across runs. Overall, WA acts as a regularization, improving robustness under label corruption by tackling run-specific mechanisms favoring memorization, and improving reliability under distribution shifts by preserving run-invariant mechanisms favoring generalization.

Remark 3 (Invariance). We argue that weight averaging only keeps the invariant predictive mechanisms across runs. This is in analogy with the invariance literature [50], popular for domain generalization [51, 100] under spurious correlations, where the key idea is that the predictive mechanisms which are invariant across domains are the causal ones that are stable under distribution shifts. This theoretically connects two key paradigms for OOD generalization, ensembling and invariance, and shows that weight averaging actually benefits from both.

Remark 4 (Extension to a deeper structure with $L$ layers). We obtain a square in $p_{j}^{2}$ due to our simplified two-layer architecture. Yet, in full generality, using a deeper structure with L layers would lead to $p_{j}^{L}$. Intuitively, WA applies an AND-mask on the information, that need to be found both in the previous feature space and the next layer weights.

Remark 5 (From reward robustness to learnability). When applied to the design of RMs in WARM, we now argue that WA facilitates WARM's stability [87] by mitigating the reliance on some non-robust features. Indeed, WA makes the WARM reward more robust to small (potentially adversarial [101]) perturbations [102], i.e., smoother [103] in the input space. This relates to the Lipschitzness property of the reward [104, 105, 106], where the difference in predicted rewards is bounded by the distance in input space. Fortunately, such smoothness is useful in RL [107], in particular for the stability of the policy gradient [108] because "sharp changes in reward value are hard to represent and internalize" [109]. This is studied in Lipschitzness is all you need [109] where the authors argue that "the local Lipschitzness of the reward is a sine qua non condition for good performance", required "to even learn anything". In summary, robustness improves stability and hinders the cascade of errors occurring when minor input variations can cause large reward differences.

In conclusion, we summarize the benefits from WARM. First, WARM is efficient, incurring no memory or computation costs, as it returns a single model. Second, WARM reduces variance while leveraging mechanisms invariant across runs, thus improving its reliability under distribution shifts. Lastly, WARM also addresses label corruption, thereby augmenting robustness to noisy preferences.

## 5. Experiments

To empirically validate WARM's benefits described in previous section, we train PaLM-XXS RMs on the TL;DR summarization benchmark [97] where preference labels are generated by a PaLM-L model prompted with chain-of-thought [99]. This AI labeling approach, increasingly common in recent research [26, 41, 110] as an efficient alternative to human assessments, is motivated by studies [57,58] indicating that it correlates well with human preferences: critically, it provides an automatic pairwise oracle preference metric to evaluate reward hacking (in a similar fashion to the distillation setup from [17], discussed in Appendix C.4). In addition, we leverage a PaLM-XS RM for pointwise

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=468&width=1654&top_left_y=1028&top_left_x=201)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=368&width=396&top_left_y=1047&top_left_x=216)

(a) PaLM (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=365&width=394&top_left_y=1051&top_left_x=631)

(b) PaLM (corrupt).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=365&width=394&top_left_y=1051&top_left_x=1045)

(c) T5 (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=365&width=394&top_left_y=1051&top_left_x=1456)

(d) T5 (corrupt)

Figure 6 | Control reward for BoN experiments: clean preference dataset in Figures 6(a) and 6(c) and 25\% corruptions in Figures 6(b) and 6(d). We consider two SFT policies to generate candidate summaries: one based on PaLM architecture [98], the other on T5 architecture [111]. The $x$-axis is the KL between the BoN policy and the SFT policy; the $y$-axis represents the control reward gains w.r.t. to an RM $\phi_{1}$, which was the best individual RM on $\mathcal{D}_{\text {ood }}$. The blue lines represent WARM with $M$ weights: WARM performs higher than the individual RMs (in yellows) or when ensembling their predictions (ENS in red). We report the absolute control rewards for those experiments in Figure 15, where the values range roughly between 3 and 7 .

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=462&width=1671&top_left_y=1882&top_left_x=198)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=365&width=394&top_left_y=1899&top_left_x=206)

(a) SFT (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=368&width=397&top_left_y=1895&top_left_x=630)

(b) SFT (corrupt)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=368&width=394&top_left_y=1895&top_left_x=1042)

(c) WARM (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-11.jpg?height=383&width=400&top_left_y=1893&top_left_x=1453)

(d) WARM (corrupt).

Figure $7 \mid$ Oracle preference metric for BoN experiments on T5 generations: clean preference dataset in Figures 7 (a) and 7 (c) and 25\% corruptions in Figures 7(b) and 7(d). We plot the win rates for different values of $N$ vs. two reference strategies: SFT (i.e., random selection or equivalently BoN with $N=1$ ), or selecting the best summary according to WARM $M=6$. We observe that all strategies beat the SFT reference (they are all above $50 \%$ win rate), but that none beat the $W A R M M=6$ reference.
control reward reaching $80.1 \%$ accuracy on the OOD dataset $\mathcal{D}_{\text {ood }}$. As verified in our experiments, this control RM also detects hacking, as it benefits from a larger architecture and a disjoint pretraining compared to the PaLM-XXS RMs of interest. Below, we explore two key scenarios: in Section 5.1, WARM reranks outputs in best-of-N (BoN); in Section 5.2, WARM guides the RL procedure.

### 5.1. Best-of- $N$ experiments

Setup. We start with best-of-N (BoN) sampling experiments in Figures 6 and 7. Given a dataset of $D$ text prompts, for each prompt we generate $N$ summaries from a SFT policy, and then returns the summary with the highest reward according to different RMs. We actually consider two SFT policies; one based on PaLM architecture [98] ( $N=8, D=15000)$, the other on T5 architecture [111] ( $N=1000, D=1000$ ). For the $x$-axis, we plot the KL between the BoN policy and the SFT policy, which can be approximated by $\log (N)-\frac{N-1}{N}[112,113]$. BoN is effective [16], especially in the low-KL regime (i.e., for small $N$ ). We consider two setups, without (clean setup) and with (corrupt setup) $25 \%$ label corruption in the preference datasets for reward modeling, and denote in each setup the weights $\left\{\phi_{i}\right\}_{i=1}^{M}$ sorted in decreasing accuracy on $\mathcal{D}_{\text {ood }}$.

Control reward. Figure 6 shows that, in terms of pointwise control reward, WARM performs consistently better than ENS (only with $M=2$ for computational reasons) and the two best individual RMs $\phi_{1}$ and $\phi_{2}$; moreover, the gains get bigger for $M=6$. As a side note, we also observe that the individual RM $\phi_{2}$ performs better in BoN in Figure 6(c) than $\phi_{1}$ though $\phi_{1}$ was better than $\phi_{2}$ on $\mathcal{D}_{\text {ood }}$, highlighting that selecting the appropriate individual RM is not trivial [41].

Oracle preference. In Figure 7, we leverage the pairwise oracle preference [58] metric to validate better performance with WARM. We observe in Figures 7(a) and 7(b) that summaries selected with WARM have a win rate of up to $92.5 \%$ against the random selection of a summary (from SFT). We also see in Figures 7(c) and 7(d) that reciprocally, all selection strategies have a win rate lower than $50 \%$ against the summaries selected by $W A R M M=6$.

### 5.2. RL experiments

Setup. For RL fine-tuning of policies, we follow [58] and use their modified version of REINFORCE [60] with a baseline value score for variance reduction, a simpler algorithm than PPO [61] yet still effective for LLMs. Both policy and value LLMs are PaLM-XS, initialized from the same SFT model. We then generate samples with the policy, compute the reward with the RMs and update the weights to optimize this reward. More details are available in Appendix B.4. To reduce forgetting and encourage the policy to remain close to its SFT initialization, we incorporate a KL regularization [81, 82] controlled by a coefficient $\alpha$, ablated in Figure 8(c), yet otherwise set to 0.003 in the clean setup and 0.01 in the corrupt setup. This KL serves as the $x$-axis in our plots to estimate model drift, as done in the literature; same curves with the number of training steps as the $x$-axis in Figures 1(b) and 18 .

Control reward. In Figure 8, we observe reward hacking; as the policy moves away from its SFT initialization, the control reward collapses. Critically, WARM improves performances: in particular, increasing $M$ pushes the Pareto front of solutions to the top left in Figures 8(a) and 8(b). In comparison, policies trained with ENS (with $M=2$ for computational reasons) are still susceptible to early reward hacking, while reaching absolute control rewards significantly worse than with WARM (even with $M=2$ ). In Figure 8(c), we confirm that the $\alpha$ hyperparameter plays a crucial role; low values of $\alpha$ such as 0.001 correspond to high $\mathrm{KL}$, while high values of $\alpha$ such as 0.01 entail low $\mathrm{KL}$ but a risk of underfitting. From a practical perspective, this highlights that the optimal value of $\alpha$ for WARM is lower than for a single RM; this is because WARM can mitigate reward hacking, and thus the optimal policies are obtained for larger values of $\mathrm{KL}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=619&width=1671&top_left_y=276&top_left_x=198)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=516&width=534&top_left_y=296&top_left_x=218)

(a) RL (clean).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=520&width=534&top_left_y=297&top_left_x=767)

(b) RL (corrupt).

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=520&width=534&top_left_y=297&top_left_x=1315)

(c) Ablating $\alpha$ for RL (clean).

Figure 8 | Control reward for RL experiments: clean preference dataset in Figures 8(a) and 8(c) and 25\% corruptions in Figure 8(b). The blue lines show the RL fine-tuning of policies when averaging $M$ weights as the RM; the darker, the higher the $M$. It performs higher than when RL fine-tuning with the individual RMs (in yellows) or when ensembling their predictions (in red). Figure 8(c) shows results of policies RL fine-tuned with WARM $M=6$ or $\phi_{1}$, for different values of $\alpha$ controlling the KL regularization strength.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=603&width=1671&top_left_y=1189&top_left_x=198)

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=502&width=534&top_left_y=1208&top_left_x=218)

(a) SFT.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=505&width=552&top_left_y=1204&top_left_x=752)

(b) WARM $M=6$.

![](https://cdn.mathpix.com/cropped/2024_06_04_4722035d5fa77853a0b7g-13.jpg?height=502&width=534&top_left_y=1208&top_left_x=1315)

(c) $\phi_{1}$ (best individual RM).

Figure 9 | Oracle preference metric for RL experiments: clean preference dataset. We plot the win rates along RL fine-tuning against three reference policies: the SFT policy, the policy RL fine-tuned with WARM $M=6$ after 3500 steps, and the policy RL fine-tuned with $\phi_{1}$ after 3000 steps. Figure 19 reports results when comparing policies at fixed number of training steps.

Oracle preference. In Figure 9, we compare the different policies according to our pairwise oracle preference AI labeler [58]. In Figure 9 (a), the reference policy is the SFT initialization; all the RL fine-tuned policies outperform this baseline, with $W A R M M=6$ reaching a win rate of $99.8 \%$ after 3500 steps (the highest win rate among all policies). We use this policy as the reference in Figure 9(b); no other policy could beat it. Interestingly, we observe that using $M=10$ rewards can delay reward hacking but does not improve the peak performance; we speculate this is related to our weight selection procedure, as the weights $\left\{\phi_{i}\right\}_{i=7}^{10}$ have lower individual accuracy on $\mathcal{D}_{\text {ood }}$ than $\left\{\phi_{i}\right\}_{i=1}^{6}$ (more details in Figure 10). Finally, in Figure 9(c), the reference policy is obtained after 3000 steps of RL fine-tuning with $\phi_{1}$ (the best individual RM on $\mathcal{D}_{\text {ood }}$ ). There is a large region of steps in which policies trained WARM (even with $M=2$ ) beat this approach; the previous reference from Figure 9(b) actually has a $79.4 \%$ win rate against it.

## 6. Discussion

Benefits. WARM represents a flexible and pragmatic method to improve the alignment of AI with human values and societal norms. This paper has detailed several of its benefits, and below, we delve into additional, more exploratory advantages. WARM follows the updatable machine learning paradigm [114], eliminating the need for inter-server communication, thus enabling embarrassingly simple parallelization [115] of RMs. This facilitates its use in federated learning scenario [116] where the data should remain private; moreover, WA would add a layer of privacy and bias mitigation by reducing the memorization of private preference [52]. Then, a straightforward extension of WARM would combine RMs trained on different datasets, for example, coming from different (clusters of) labelers. This diversity could help WARM performances, but also from a multi objective perspective [117]; by non-uniform interpolation of RMs, we could learn a set of personalized policies [39]. Furthermore, as WA has been shown to limit catastrophic forgetting [118, 119], WARM could seamlessly support iterative and evolving preferences. Finally, a promising research direction is extending WARM to direct preference optimization (DPO) strategies [120], where averaging the RMs casts back to averaging the DPO policies [121].

Limitations. WARM, while innovative, does face some limitations, notably two when compared to prediction ensembling methods; first, prediction ensembling can benefit from the diversity brought by combining RMs from various architectures and pre-trainings; second, prediction ensembling can incorporate prediction disagreement into the reward to provide uncertainty estimation and limit model drift. However, it's been noted in [41] that simple averaging of logits often performs comparably to more complex prediction aggregation functions that include uncertainty elements. Another limitation is that, while WARM effectively reduces certain types of memorization, it does not completely eradicate all forms of spurious correlations or biases inherent in the preference data. For instance, if each individual RM predominantly relies on summary length as a criterion, WARM is likely to replicate this tendency. Therefore, alternative methods (from the OOD generalization literature?) might be required, for example those based on invariance regularization [51, 100] or last layer retraining [122]. Finally, WARM only enhances reward modeling without tackling the other challenges in RLHF [18]; thus, to mitigate the safety risks [19, 34, 35] from misalignment [32, 33], WARM must be considered within the larger context of responsible AI.

## 7. Conclusion

In conclusion, we introduce Weight Averaged Reward Models (WARM) to address two critical challenges in reward modeling: reliability under distribution shifts and robustness under label corruption. By averaging the weights of multiple RMs obtained from diverse fine-tunings, WARM appears as an efficient solution to mitigate reward hacking in reinforcement learning from human feedback. Our empirical results demonstrate its effectiveness when applied to summarization. We anticipate that WARM will contribute to more aligned, transparent, and effective AI systems, encouraging further exploration in reward modeling.

## References

[1] Google Gemini Team. Gemini: A family of highly capable multimodal models. 2023. (p. 1)

[2] OpenAI. Gpt-4 technical report. 2023. (p. 1)

[3] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint, 2023. (р. 1)

[4] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. (pp. 1 and 3)

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. (p. 1)

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. (pp. 1, 7, and 27)

[7] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR, 2022. (pp. 1 and 3)

[8] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In ACL, 2022. (p. 1)

[9] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford Alpaca: An instruction-following LLaMA model, 2023. (p. 1)

[10] Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Léonard Hussenot, Orgad Keller, et al. Factually consistent summarization via reinforcement learning with textual entailment feedback. In $A C L$, 2023. (p. 1)

[11] Lev McKinney, Yawen Duan, David Krueger, and Adam Gleave. On the fragility of learned reward functions. arXiv preprint, 2023. (p. 1)

[12] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In NeurIPS, 2017. (pp. 1, 2, 3, 5, and 27)

[13] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint, 2019. (pp. 1 and 27)

[14] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. NeurIPS, 2020. (pp. 1, 7, and 27)

[15] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint, 2021. (pp. 1 and 27)

[16] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. LLaMA 2: Open foundation and fine-tuned chat models. arXiv preprint, 2023. (pp. 1, 5, 12, and 27)

[17] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In ICML, 2023. (pp. 1, 11, and 33)

[18] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR, 2023. (pp. 1 and 14)

[19] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint, 2016. (pp. 1, 2, 3, 4, and 14)

[20] Jack Clark and Dario Amodei. Faulty Reward Functions in the Wild. https://openai.com /research/faulty-reward-functions, 2016. (pp. 1 and 4)

[21] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. arXiv preprint, 2021. (pp. 1 and 4)

[22] Joar Max Viktor Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. In NeurIPS, 2022. (pp. 1 and 4)

[23] Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In ICLR, 2022. (pp. 1 and 4)

[24] Nathan Lambert and Roberto Calandra. The alignment ceiling: Objective mismatch in reinforcement learning from human feedback. arXiv preprint, 2023. (p. 1)

[25] Mike Lewis, Denis Yarats, Yann N Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end learning for negotiation dialogues. arXiv preprint, 2017. (p. 1)

[26] Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. arXiv preprint, 2023. (pp. 1 and 11)

[27] Marilyn Strathern. Improving ratings: audit in the british university system. European Review, 1997. (p. 1)

[28] Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. arXiv preprint, 2022. (p. 1)

[29] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards understanding sycophancy in language models. arXiv preprint, 2023. (p. 1)

[30] Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In ICML, 2023. (p. 1)

[31] Jochen Hartmann, Jasper Schwenzow, and Maximilian Witte. The political ideology of conversational ai: Converging evidence on chatgpt's pro-environmental, left-libertarian orientation. arXiv preprint, 2023. (p. 1)

[32] Jessica Taylor, Eliezer Yudkowsky, Patrick LaVictoire, and Andrew Critch. Alignment for advanced machine learning systems. Ethics of AI, 2016. (pp. 1, 3, and 14)

[33] Richard Ngo, Lawrence Chan, and Soren Mindermann. The alignment problem from a deep learning perspective. arXiv preprint, 2022. (pp. 1, 3, 14, and 27)

[34] Dan Hendrycks and Mantas Mazeika. X-risk analysis for AI research. arXiv preprint, 2022. (pp. 1 and 14)

[35] Dan Hendrycks. Natural selection favors AIs over humans. arXiv preprint, 2023. (pp. 1 and 14)

[36] Simon Zhuang and Dylan Hadfield-Menell. Consequences of misaligned AI. NeurIPS, 2020. (pp. 2 and 4)

[37] Daniel Shin, Anca Dragan, and Daniel S. Brown. Benchmarks and algorithms for offline preference-based reward learning. TMLR, 2023. (p. 2)

[38] Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile Lukosuite, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. arXiv preprint, 2022. (pp. 2 and 4)

[39] Alexandre Ramé, Guillaume Couairon, Mustafa Shukor, Corentin Dancette, Jean-Baptiste Gaya, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In NeurIPS, 2023. (pp. 2, 14, and 26)

[40] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. (pp. 2, 3, and 27)

[41] Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D'Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint, 2023. (pp. 2, 4, 5, 11, 12, 14, and 27)

[42] Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint, 2023. (pp. 2, 5, and 27)

[43] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In NeurIPS, 2017. (pp. 2 and 5)

[44] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In ICML, 2020. (pp. 3, 6, and 26)

[45] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? In NeurIPS, 2020. (pp. 3, 6, and 26)

[46] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, 2022. (pp. 3, 6, 7, 9, 26, and 28)

[47] Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. In NeurIPS, 2022. (pp. 3, 5, 6, 7, 8, 9, 26, and 28)

[48] Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David LopezPaz. Model Ratatouille: Recycling diverse models for out-of-distribution generalization. In ICML, 2023. (pp. 3, 6, 7, 26, and 28)

[49] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. SWAD: Domain generalization by seeking flat minima. In NeurIPS, 2021. (pp. 3 and 26)

[50] Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. Domain generalization via invariant feature representation. In ICML, 2013. (pp. 3 and 10)

[51] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint, 2019. (pp. 3, 4, 10, and 14)

[52] Kerem Zaman, Leshem Choshen, and Shashank Srivastava. Fuse to forget: Bias reduction and selective memorization through model fusion. arXiv preprint, 2023. (pp. 3, 14, and 26)

[53] Yong Lin, Lu Tan, Yifan Hao, Honam Wong, Hanze Dong, Weizhong Zhang, Yujiu Yang, and Tong Zhang. Spurious feature diversification improves out-of-distribution generalization. In ICLR, 2024. (pp. 3, 9, and 26)

[54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. (pp. 3 and 6)

[55] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint, 2021. (pp. 3 and 26)

[56] Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In CVPR, 2014. (p. 3)

[57] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint, 2022. (pp. 4 and 11)

[58] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint, 2023. (pp. 4, 7, 11, 12, 13, 27, and 28)

[59] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 1952. (p. 4)

[60] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Reinforcement learning, 1992. (pp. 4, 12, and 28)

[61] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017. (pp. 4 and 12)

[62] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In ICLR, 2021. (p. 4)

[63] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In ICML, 2021. (p. 4)

[64] Mohammad Pezeshki, Sékou-Oumar Kaba, Yoshua Bengio, Aaron Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. In NeurIPS, 2020. (р. 4)

[65] Firas Laakom, Jenni Raitoharju, Alexandros Iosifidis, and Moncef Gabbouj. Learning distinct features helps, provably. arXiv preprint, 2021. (p. 4)

[66] Niv Nayman, Avram Golbert, Asaf Noy, Tan Ping, and Lihi Zelnik-Manor. Diverse ImageNet models transfer better. arXiv preprint, 2022. (p. 4)

[67] Alexander D'Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoffman, et al. Underspecification presents challenges for credibility in modern machine learning. JMLR, 2020. (pp. 4 and 26)

[68] Damien Teney, Yong Lin, Seong Joon Oh, and Ehsan Abbasnejad. ID and OOD performance are sometimes inversely correlated on real-world datasets. In NeurIPS Workshop, 2023. (p. 4)

[69] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In ICML, 2017. (p. 4)

[70] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In NeurIPS, 2019. (p. 4)

[71] Yoav Wald, Amir Feder, Daniel Greenfeld, and Uri Shalit. On calibration and out-of-domain generalization. In NeurIPS, 2021. (p. 4)

[72] Herbert A Simon. Bounded rationality. Utility and probability, 1990. (p. 4)

[73] Rohin Shah, Noah Gundotra, Pieter Abbeel, and Anca Dragan. On the feasibility of learning, rather than assuming, human biases for reward inference. In ICML, 2019. (p. 4)

[74] Timo Kaufmann, Sarah Ball, Jacob Beck, Eyke Hüllermeier, and Frauke Kreuter. On the challenges and practices of reinforcement learning from real human feedback. 2023. (p. 4)

[75] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint, 2022. (p. 4)

[76] Robert Irvine, Douglas Boubert, Vyas Raina, Adian Liusie, Vineet Mudupalli, Aliaksei Korshuk, Zongyi Liu, Fritz Cremer, Valentin Assassi, Christie-Carol Beauchamp, et al. Rewarding chatbots for real-world engagement with millions of users. arXiv preprint, 2023. (p. 4)

[77] Condorcet. Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix. 1785. (p. 4)

[78] Silviu Pitis. Failure modes of learning reward models for llms and other sequence models. In ICML, 2023. (р. 4)

[79] Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models' sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. arXiv preprint, 2023. (p. 4)

[80] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. State of what art? a call for multi-prompt llm evaluation. arXiv preprint, 2023. (p. 4)

[81] Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In ICML, 2017. (pp. 5 and 12)

[82] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In ICML, 2019. (pp. 5 and 12)

[83] Angeliki Lazaridou, Anna Potapenko, and Olivier Tieleman. Multi-agent communication meets natural language: Synergies between functional and structural language learning. In ACL, 2020. (p. 5)

[84] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In ICML, 2020. (р. 5)

[85] Siddharth Reddy, Anca Dragan, Sergey Levine, Shane Legg, and Jan Leike. Learning human objectives by evaluating hypothetical behavior. In ICML, 2020. (pp. 5 and 27)

[86] William Saunders, Girish Sastry, Andreas Stuhlmüller, and Owain Evans. Trial without error: Towards safe reinforcement learning via human intervention. In AAMAS, 2018. (p. 5)

[87] Binghai Wang et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint, 2023. (pp. 5, 10, and 27)

[88] Ron Kohavi, David H Wolpert, et al. Bias plus variance decomposition for zero-one loss functions. In ICML, 1996. (p. 5)

[89] Naonori Ueda and Ryohei Nakano. Generalization error of ensemble estimators. In ICNN, 1996. (р. 5)

[90] Sandipan Kundu, Yuntao Bai, Saurav Kadavath, Amanda Askell, Andrew Callahan, Anna Chen, Anna Goldie, Avital Balwit, Azalia Mirhoseini, Brayden McLean, et al. Specific versus general principles for constitutional ai. arXiv preprint, 2023. (p. 5)

[91] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. In ICLR, 2022. (pp. 5, 6, 7, and 28)

[92] Samuel K. Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In ICLR, 2022. (p. 6)

[93] Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In EMNLP, 2023. (p. 6)

[94] Raphael Gontijo-Lopes, Yann Dauphin, and Ekin Dogus Cubuk. No one representation to rule them all: Overlapping features of training methods. In ICLR, 2022. (pp. 6 and 8)

[95] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In UAI, 2018. (pp. 7 and 26)

[96] Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. Ensemble of averages: Improving model selection and boosting performance in domain generalization. In NeurIPS, 2021. (pp. 7 and 26)

[97] Michael Völske, Martin Potthast, Shahbaz Syed, and Benno Stein. T1; dr: Mining reddit to learn automatic summarization. In ACL Workshop, 2017. (pp. 7, 8, and 11)

[98] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 technical report. arXiv preprint, 2023. (pp. 7, 11, 12, 27, 28, and 30)

[99] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought prompting elicits reasoning in large language models. In NeurIPS, 2022. (pp. 7, 11, and 27)

[100] Alexandre Ramé, Corentin Dancette, and Matthieu Cord. Fishr: Invariant gradient variances for out-of-distribution generalization. In ICML, 2022. (pp. 10 and 14)

[101] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint, 2013. (p. 10)

[102] Yao-Yuan Yang, Cyrus Rashtchian, Hongyang Zhang, Ruslan Salakhutdinov, and Kamalika Chaudhuri. Adversarial robustness through local lipschitzness. arXiv preprint, 2020. (p. 10)

[103] Mihaela Rosca, Theophane Weber, Arthur Gretton, and Shakir Mohamed. A case for new neural network smoothness constraints. In NeurIPS ICBINB, 2020. (p. 10)

[104] Matthias Hein and Maksym Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. NeurIPS, 2017. (p. 10)

[105] Jure Sokolić, Raja Giryes, Guillermo Sapiro, and Miguel RD Rodrigues. Robust large margin deep neural networks. IEEE Transactions on Signal Processing, 2017. (p. 10)

[106] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In ICML, 2019. (p. 10)

[107] Roland Hafner and Martin Riedmiller. Reinforcement learning in feedback control: Challenges and benchmarks from technical process control. Machine learning, 2011. (p. 10)

[108] Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Policy gradient in lipschitz markov decision processes. Machine Learning, 2015. (p. 10)

[109] Lionel Blondé, Pablo Strasser, and Alexandros Kalousis. Lipschitzness is all you need to tame off-policy generative adversarial imitation learning. Machine Learning, 2022. (p. 10)

[110] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint, 2023. (p. 11)

[111] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. (pp. 11, 12, 30, and 33)

[112] Jacob Hilton. KL divergence of max-of-n, 2023. (p. 12)

[113] Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D'Amour, Jacob Eisenstein, Chirag Nagpal, and Ananda Theertha Suresh. Theoretical guarantees on the best-of-n alignment policy. arXiv preprint, 2024. (p. 12)

[114] Colin Raffel. Building Machine Learning Models Like Open Source Software. ACM, 2023. (p. 14)

[115] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-Train-Merge: Embarrassingly parallel training of expert language models. arXiv preprint, 2022. (p. 14)

[116] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017. (p. 14)

[117] Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. In NeuriPS, 2023. (p. 14)

[118] Zafir Stojanovski, Karsten Roth, and Zeynep Akata. Momentum-based weight interpolation of strong zero-shot models for continual learning. In NeurIPS Workshop, 2022. (pp. 14 and 26)

[119] Steven Vander Eeckt et al. Weight averaging: A simple yet effective method to overcome catastrophic forgetting in automatic speech recognition. arXiv preprint, 2022. (р. 14)

[120] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint, 2023. (pp. 14 and 27)

[121] Maxime Labonne. NeuralBeagle14-7B. https://huggingface.co/mlabonne/NeuralBe agle14-7B-GGUF, 2024. (pp. 14 and 27)

[122] Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In ICLR, 2023. (p. 14)

[123] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019. (p. 26)

[124] John R. Zech, Marcus A. Badgeley, Manway Liu, Anthony B. Costa, Joseph J. Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine, 2018. (p. 26)

[125] Alex J DeGrave, Joseph D Janizek, and Su-In Lee. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 2021. (p. 26)

[126] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Hanna Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models. In CVPR, 2022. (p. 26)

[127] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. In NeurIPS, 2022. (p. 26)

[128] Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. ColD fusion: Collaborative descent for distributed multitask finetuning. In ACL, 2023. (p. 26)

[129] Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret. Pareto manifold learning: Tackling multiple tasks via ensembles of single-task models. arXiv preprint, 2022. (Not cited.)

[130] Mustafa Shukor, Corentin Dancette, Alexandre Ramé, and Matthieu Cord. Unival: Unified model for image, video, audio and language. TMLR, 2023. (p. 26)

[131] Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, and Sven Gowal. Seasoning model soups for robustness to adversarial and natural distribution shifts. In CVPR, 2023. (p. 26)

[132] Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. In ICLR, 2023. (p. 26)

[133] Evgenii Nikishin, Pavel Izmailov, Ben Athiwaratkun, Dmitrii Podoprikhin, Timur Garipov, Pavel Shvechikov, Dmitry Vetrov, and Andrew Gordon Wilson. Improving stability in deep reinforcement learning with weight averaging. 2018. (p. 26)

[134] Jean-Baptiste Gaya, Laure Soulier, and Ludovic Denoyer. Learning a subspace of policies for online adaptation in reinforcement learning. In ICLR, 2022. (p. 26)

[135] Daniel Lawson and Ahmed H Qureshi. Merging decision transformers: Weight averaging for forming multi-task policies. In ICLR RRL Workshop, 2023. (p. 26)

[136] Michael Noukhovitch, Samuel Lavoie, Florian Strub, and Aaron Courville. Language model alignment with elastic reset. In NeurIPS, 2023. (p. 26)

[137] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In ICLR, 2023. (p. 26)

[138] Nico Daheim, Nouha Dziri, Mrinmaya Sachan, Iryna Gurevych, and Edoardo M Ponti. Elastic weight removal for faithful and abstractive dialogue generation. arXiv preprint, 2023. (p. 26)

[139] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. TNNLS, 2022. (p. 26)

[140] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. ICLR, 2017. (p. 26)

[141] Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C Alexander, and Nathan Silberman. Learning from noisy labels by regularized estimation of annotator confusion. In CVPR, 2019. (p. 26)

[142] Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, et al. Neftune: Noisy embeddings improve instruction finetuning. arXiv preprint, 2023. (p. 26)

[143] Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. Robust loss functions under label noise for deep neural networks. In $A A A I, 2017$. (р. 26)

[144] Xiaobo Xia, Tongliang Liu, Bo Han, Mingming Gong, Jun Yu, Gang Niu, and Masashi Sugiyama. Sample selection with uncertainty of losses for learning with noisy labels. In ICLR, 2022. (p. 26)

[145] Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018. (p. 26)

[146] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. NeurIPS, 2018. (р. 26)

[147] Maryam Sabzevari. Ensemble learning in the presence of noise. PhD thesis, Universidad Autónoma de Madrid, 2019. (р. 26)

[148] Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In ICML, 2000. (p. 27)

[149] W Bradley Knox, Stephane Hatgis-Kessell, Sigurdur Orn Adalgeirsson, Serena Booth, Anca Dragan, Peter Stone, and Scott Niekum. Learning optimal advantage from preferences and mistaking it for reward. arXiv preprint, 2023. (p. 27)

[150] Peter Barnett, Rachel Freedman, Justin Svegliato, and Stuart Russell. Active reward learning from multiple teachers. arXiv preprint, 2023. (р. 27)

[151] Sian Gooding and Hassan Mansoor. The impact of preference agreement in reinforcement learning from human feedback: A case study in summarization. arXiv preprint, 2023. (p. 27)

[152] Lei Li, Yekun Chai, Shuohuan Wang, Yu Sun, Hao Tian, Ningyu Zhang, and Hua Wu. Toolaugmented reward modeling. In ICLR, 2023. (p. 27)

[153] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint, 2023. (p. 27)

[154] Anonymous. RIME: Robust preference-based reinforcement learning with noisy human preferences. In Submitted to ICLR, 2023. (р. 27)

[155] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint, 2023. (р. 27)

[156] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In ICML, 2018. (pp. 27 and 28)

[157] Noam Razin, Hattie Zhou, Omid Saremi, Vimal Thilak, Arwen Bradley, Preetum Nakkiran, Joshua Susskind, and Etai Littwin. Vanishing gradients in reinforcement finetuning of language models. arXiv preprint, 2023. (р. 28)

# WARM: On the Benefits of Weight Averaged Reward Models 

Supplementary material

This supplementary material is organized as follows:

- Appendix A enriches our related work.
- Appendix B clarifies some experimental details.
- Appendix C enriches our experiments.

</end of paper 0>


<paper 1>
# COURSEGPT-ZH: AN EDUCATIONAL LARGE LANGUAGE MODEL BASED ON KNOWLEDGE DISTILLATION INCORPORATING PROMPT OPTIMIZATION * 

Zheyan Qu, Lu Yin, Zitong Yu, Wenbo Wang, Xing zhang*<br>Wireless Signal Processing and Network Laboratory, Beijing University of Posts and Telecommunications, Beijing<br>Department of Computer Science, University of Aberdeen, UK<br>zhangx @ ieee.org


#### Abstract

Large language models (LLMs) have demonstrated astonishing capabilities in natural language processing (NLP) tasks, sparking interest in their application to professional domains with higher specialized requirements. However, restricted access to closed-source LLMs via APIs and the difficulty in collecting massive high-quality datasets pose obstacles to the development of large language models in education fields of various courses. Given these challenges, we propose CourseGPT-zh, a course-oriented education LLM that supports customization and low-cost deployment. To address the comprehensiveness and diversity requirements of course-specific corpora, we design a high-quality question-answering corpus distillation framework incorporating prompt optimization, which effectively mines textbook knowledge and enhances its diversity. Moreover, considering the alignment of LLM responses with user needs, a novel method for discrete prompt optimization based on LLM-as-Judge is introduced. During optimization, this framework leverages the LLM's ability to reflect on and exploit error feedback and patterns, allowing for prompts that meet user needs and preferences while saving response length. Lastly, we obtain CourseGPT-zh based on the open-source LLM using parameter-efficient fine-tuning. Experimental results show that our discrete prompt optimization framework effectively improves the response quality of ChatGPT, and CourseGPT-zh exhibits strong professional capabilities in specialized knowledge question-answering, significantly outperforming comparable open-source models.


## 1 Introduction

Large language models, such as ChatGPT [1], GPT4 [2], LLaMA [3], and ChatGLM [4], have demonstrated remarkable performance and generalization capabilities across various NLP tasks, significantly expanding the boundaries of language applications. With the increase in model parameters and pretraining corpus size, capabilities such as logical reasoning, instruction following, and In-Context Learning [5],[6], [7] have emerged. Based on these breakthroughs, the latest LLMs have shown profound understanding and professionalism in various fields, such as virtual assistants, text generation, and code annotation. Utilizing LLMs to disrupt industries has become an inevitable trend, including the field of education[8], [9].

Recently, there has been a desire to leverage the extensive knowledge of large language models to construct domainspecific LLMs in various vertical fields, which require greater expertise and accuracy. To address the issue that general-purpose LLMs cannot meet specific domain requirements, a variety of methods have been proposed. For instance, steering foundation models through role-playing or prompt engineering have been used to tap into the knowledge learned during the pre-training phase, which can unleash their deep-seated expert capabilities [10], [11]. Other approaches involve pretraining or continual pre-training with domain-specific corpus to incorporate domainspecific knowledge into large language models [8], [12], [13],[14]. In addition, to reduce the hallucination during the response generation, retrieval augmentation has also been applied to provide reliable references [8],[15]. Based on these[^0]approaches, successful implementations such as MedAgents [10], ChatLaw [15], EduChat [8], and FinGPT [16] have demonstrated the potential of LLMs to provide professional responses and insights in various vertical fields, including healthcare, law, finance, and education.

However, constructing domain-specific large language models is still labor-consuming and expensive. To begin with, for closed-source large language models like ChatGPT, the high costs of text generation and fine-tuning services are often prohibitive. As for open-source LLMs, there is a significant gap in parameter size and pre-training corpus compared to closed-source LLMs, resulting in significantly weaker general capabilities such as reasoning, and domain-specific knowledge extraction [9],[17], [18], [19]. Faced with complex professional terminology, open-source large language models often fail to meet user requirements for domain knowledge. In this context, it often requires a large amount of in-domain pre-training corpus or expertise datasets to enhance professionalism in vertical fields.

Although various existing works have developed specialized datasets and evaluation criteria for various fields such as philosophy, medicine, and law, as well as for scenarios including network operation and geospatial semantics [17], [18], [19], [20], [21], there is still a considerable demand for manual effort in constructing datasets for courses or privatized scenarios that are not covered by these datasets. This challenge is particularly pronounced when accessible corpora in the field are scarce, making it extremely difficult to construct tens of thousands of specialized instruction data. Furthermore, the majority of models are primarily pre-trained on English corpora, which may lead to a degradation in their performance in other languages [22], [23].

In addition to the challenges of constructing specialized corpora, the high cost of inference incurred by open-source large language models cannot be overlooked. Compared to the concise responses provided by humans, the responses generated by large language models, while more comprehensive, also include a significant amount of redundant information, resulting in unnecessary inference overhead. Typically, to further align the responses of large language models with specific preferences, methods such as RLHF (Reinforcement Learning from Human Feedback)[24] are introduced for fine-tuning models. However, this approach still requires a substantial amount of human-labeled preference data. Consequently, promoting alignment between the responses and human preferences, as well as reducing inference costs, is also a key factor in fostering the widespread adoption of open-source large models in specialized vertical domains.

Targeted at these issues, we propose CourseGPT-zh, an open-source education large language model, and design a pipeline for constructing high-quality question-answer pairs through mining textbook knowledge. By utilizing the constructed diverse question-answer pairs, we perform parameter-efficient fine-tuning on the open-source model to mitigate the resource constraints required for deployment. In addition, in the data construction process, we incorporate LLM-as-Judge and utilize discrete prompt optimization to generate optimal prompts, steering ChatGPT to produce high-quality training data aligned with human preferences. Through this method, we ensure high-quality responses while reducing the deployment costs associated with response length.

Our main contributions can be summarized as:

- In this paper, we propose CourseGPT-zh, an open-source education large language model, with a pipeline for constructing high-quality and diverse question-answer pairs. Based on textbooks, we guide the model to conduct thorough exploration and questioning of textbooks, extracting knowledge from both closed-source large language models and specialized texts. Additionally, we employ a method inspired by self-instruct to guide the large language models in generating related questions, further enhancing the diversity.
- Considering that although large language models can generate comprehensive answers, some content may be redundant or incorrect. Therefore, we employ prompt engineering to guide ChatGPT in generating responses that align with human preferences. To obtain the optimal prompts, we have designed an iterative discrete prompt optimization framework, which incorporates LLM-as-Judge to facilitate automatic evaluation of the quality of responses guided by prompts. Furthermore, the optimized prompt allows the large language model to achieve a balance between the quality of responses and their length, achieving information compression in responses.
- A parameter-efficient fine-tuning method of the ChatGLM3 model is conducted based on constructed highquality question-answering data, resulting in the CourseGPT-zh. Experimental evidence has shown that CourseGPT-zh exhibits improved alignment with human responses, and delivers more concise answers while maintaining a high level of response quality. On various NLP task evaluation metrics, CourseGPT-zh significantly outperforms other open-source large models.


## 2 Related-work

With fierce competition and rapid development, large language models ranging from billions to trillions of parameters have achieved remarkable performance across various NLP tasks after being pre-trained on massive amounts of text. Represented by LLMs such as ChatGPT, GPT4, and GPT4-Turbo, the OpenAI model family has successively reset the benchmarks for NLP tasks, being regarded as one of the greatest inventions in history. Concurrently, a multitude of open-source large language models, including llama-2-13b, ChatGLM3-6b, and Mistral-8x7B-MoE[25], have also shown astonishing improvements, even surpassing the level of ChatGPT on some dimensions. More importantly, they can be deployed on a single to several GPUs and can be flexibly customized through fine-tuning.

### 2.1 Domain-specific LLMs

Although general-purpose large language models have achieved exceptional performance on generic NLP tasks, they often fall short in vertical domains that necessitate extensive specialized knowledge and high accuracy requirements. The performance of zero-shot large language models in these domains is typically inadequate, thereby granting domainspecific LLMs significant attention. Closed-source large language models, while exhibiting superior performance across various capabilities, present challenges for continual pre-training and fine-tuning with private corpora. Therefore, the construction of domain-specific models based on closed-source LLMs frequently leverages role-playing or collaboration abilities to extract knowledge in the specialized field during the pre-training phase. In contrast, open-source LLMs can be further pre-trained or fine-tuned with extensive high-quality domain-specific data, and they have achieved multiple successful applications in fields such as medicine, law, education, finance, etc.

HuatuoGPT [26] employs a mixed dataset comprising distilled data from ChatGPT and real-world data provided by physicians' medical advice to fine-tune an open-source model. Furthermore, it aligns the model's response with human preferences through RLAIF (Reinforcement Learning from Artificial Intelligence Feedback). By learning from the response styles of real-world doctor-patient interactions, the fine-tuned model can engage with users in a human-like manner and significantly surpasses other models at a similar level across various metrics. MedChatZH [12] has developed a dialogue model specifically designed for Traditional Chinese Medicine, incorporating extensive Chinese medical literature for continual pre-training. After fine-tuning millions of question-answer data from the Internet and various Chinese hospitals, the model achieves state-of-the-art performance in the field of Chinese medicine. ChatLaw [15], targeting the legal domain, not only provides professional responses concerning legal knowledge but also acquires problem-solving abilities through training on multiple-choice question data. Furthermore, it employs a method combining vector database retrieval with keyword search, effectively reducing the hallucination in responses. EduChat [8] offers a range of functionalities, including open-ended question answering, paper assessment, and Socratic teaching, enhancing various skills through fine-tuning and the integration of tools. The model gains interdisciplinary knowledge through continual pre-training and strengthens its question-answering and instruction-following capabilities with large-scale instruction and open-domain dialogue datasets. FinGPT [16] adopts a data-centric approach, focusing on automated data management pipelines and lightweight adaptive technologies, establishing a comprehensive framework from data processing to feature engineering and application, while also enhancing the transparency of the overall framework. One of its strengths lies in its ability to integrate seamlessly with both open-source and closed-source large language models without the need for further training.

### 2.2 Discrete prompt engineering

Prompt engineering aims to guide large language models to fully leverage their potential through the meticulous design of prompts. Extensive research has demonstrated that well-crafted prompts can significantly enhance the ability of large language models to improve their performance across various NLP tasks [27],[28]. Prompt engineering encompasses continuous prompt learning and discrete prompt optimization. Continuous prompt learning aims to adapt large language models to various tasks by incorporating learnable parameters within the prompts [29], [30]. However, continuous prompt learning typically requires access to the gradient vectors of the LLMs, which restricts its application in closed-source models that are accessed only through APIs. For discrete prompts, traditional methods often rely on meticulous manual design, which not only demands considerable human effort but also may not necessarily maximize the model's performance. Consequently, numerous methods for automatically generating optimal discrete prompts have been explored, leveraging the large model itself as an optimizer to autonomously enhance its performance in NLP tasks.

Recently, several leading automated discrete prompt optimization frameworks have been proposed. EVOPROMPT[31] draws on the principles of evolutionary algorithms (EAs) to iteratively guide LLMs to generate new prompts through evolutionary operators. It does not require any gradient information from LLMs and can achieve a balance between exploration and exploitation. Experiments on nine datasets have shown that optimized prompts can significantly improve task performance. APE[32], inspired by program synthesis, represents discrete prompting optimization as

![](https://cdn.mathpix.com/cropped/2024_06_04_fa7ab643797b68e77915g-04.jpg?height=656&width=1217&top_left_y=236&top_left_x=451)

Figure 1: CourseGPT-zh Framework

a black-box optimization problem. It treats instructions as "programs" and optimizes them by searching through the candidate instruction pool proposed by LLMs. Furthermore, it employs an iterative Monte Carlo search to further enhance prompt performance. OPRO[33] utilizes LLMs to generate new candidate prompts from previously generated results and their scores and then evaluates the new candidate prompts for the next iteration. PROMPTAGENT[34] approaches it as a strategic planning problem, using Monte Carlo tree search to achieve a balance between exploration and exploitation. Unlike other discrete prompt optimization frameworks, it also leverages learning capabilities based on error summarization of large language models, introducing expert-level domain knowledge and guidance based on reflection. The optimal prompts obtained from these prompting optimization frameworks have achieved results significantly better than manually crafted prompts on various NLP tasks, including GSM8K and Big-Bench Hard tasks.

## 3 Data Construction

Large language models often require at least tens of thousands of high-quality instruction-tuning data to demonstrate satisfactory performance; however, the collection and processing of such data can be prohibitively labor-intensive. Unlike fields such as medicine, which benefit from a wealth of open-source question-answering datasets gathered from the internet and hospital medical databases, amassing large volumes of high-quality question-answering datasets poses significant challenges. In light of these barriers to data construction and the demand for low-cost model development, we propose a pipeline based on knowledge distillation from ChatGPT and GLM-4, which ensures the comprehensiveness and diversity of questions, as well as the professionalism and alignment of the distilled responses. The entire process consists of two components: question construction and response generation.

### 3.1 Question Generation

To ensure that the fine-tuned model can provide professional answers to users' questions about various knowledge both inside and outside the textbooks, the comprehensiveness and diversity of the questions are of great importance. Merely relying on human-generated questions as a seed pool and using methods like self-instruct [35] to generate question data may not cover all knowledge points comprehensively, especially for the understanding of various professional terms in specialized fields. In addition, the diversity of questioning methods for the same set of knowledge points also needs to be addressed. Diverse questioning and answering of the same knowledge can guide the model to learn the internal relationships of knowledge more effectively.

In response to these challenges, we have developed a diversified question generation pipeline, as depicted in the orange section of Figure 2 Initially, knowledge extraction and questioning are based on textbook paragraphs to ensure the comprehensiveness of the questions. The textbook is divided into paragraphs, which are sequentially input into a large language model to guide the generation of a list of questions targeting specific knowledge points. Concurrently, during the process, 6 questions from a seed pool comprising 50 carefully chosen human-written questions, along with two generated questions are randomly selected as in-context question examples. This approach steers the model towards generating diverse questions for a group of knowledge points, such as interpretive questions, pros and cons comparison questions, and comparative questions. Lastly, in terms of question quantity, we force the large language model to

![](https://cdn.mathpix.com/cropped/2024_06_04_fa7ab643797b68e77915g-05.jpg?height=648&width=1217&top_left_y=234&top_left_x=454)

Figure 2: Data Construction Framework

generate an excess number of questions for the limited length of paragraphs. This serves to distill the knowledge from the large model and enhance the diversity of the questions.

However, the question lists constructed based on textbook paragraphs are still limited in form and content by the textbook. To further enhance the diversity, inspired by self-instruct, we employ an iterative approach to distill new questions from a large model. Specifically, a set of questions generated in the previous round is selected as content examples, guiding the large language model to sample and generate new content-related questions. At the same time, 3 questions are randomly selected from the seed pool as style examples to enrich the diversity of question forms. After generating a new set of questions, it is then used as content examples for the next iteration. In this way, we can reduce the influence of referencing textbook paragraphs, and a large number of new questions can be obtained through distillation.

Finally, we find that different large language models exhibit significant variation in the style and distribution of the questions they generate. Consequently, we employ both ChatGPT and GLM-4 for concurrent question generation and sampling, and subsequently deduplicated the final question list. As CourseGPT-zh is an adaptable course-specific large language model, we selected Communication Principles as the focus of our experiment, which encompasses specialized domains such as signal modulation, quantization, and coding, requiring the model to provide professional and precise responses. Acknowledging the limitations of open-source models in mathematical derivation capabilities such as integration, our focus is directed towards the learning of conceptual knowledge in this field. Utilizing the textbook on Communication Principles, we separately generate approximately 10k questions using ChatGPT and GLM-4 and conduct two rounds of iterative sampling based on these questions, resulting in about $20 \mathrm{k}$ sampled questions.

### 3.2 Answer Generation

After obtaining a diverse list of questions, the next challenge is to utilize large language models to generate professional answers that meet users' preferences and needs. Generally, LLMs tend to produce comprehensive and well-structured answers that are reader-friendly. However, these lengthy responses not only significantly differ from the style of human replies but also substantially increase response latency and reasoning. Therefore, it is necessary to align the response style of LLMs with that of humans and enhance the accuracy and professionalism of the responses. As shown in the blue part of the Figure 2, we leverage role-playing prompts to guide the large language model in generating responses that meet the requirements. Taking advantage of role-playing capabilities, the large language model can focus on professional fields and generate more reliable responses. For the same group of tasks, different prompts can lead to significant differences in model performance, which is also the case in the question-answering field. It is worth noting that for questions derived from textbook paragraphs, we refer to the original text to improve the accuracy of the answers. The optimization of prompts will be introduced in the next section.

## 4 Discrete Prompt Optimization

Research indicates that the design of prompts significantly affects the performance of large language models on NLP tasks [27],[28]. Prior studies have emphasized the design of prompts at various stages, yet the prompts deemed optimal by humans may not necessarily elicit the desired performance. Therefore, optimizing the prompts is essential, particularly during the process of knowledge distillation in specialized domains, which can prompt the large language models to generate more professional responses. In our discrete prompt optimization framework, we further incorporate LLM-as-Judge to evaluate the alignment between the responses guided by prompts and human responses. By training on the distillation data guided by optimal prompts, the fine-tuned large language model obviates the need for further optimization using Reinforcement Learning from Human Feedback (RLHF).

### 4.1 LLM-as-Judge

Alignment is a crucial step that ensures the model's performance aligns with user intentions and human preferences. Generally, Reinforced Learning from AI Feedback (RLAIF) is extensively applied in alignment tasks. However, reward models can be challenging to train and are susceptible to the influence of human errors in the training data. Moreover, in contrast to the approach of first training with mixed data and then aligning, we aim to maintain alignment between LLM responses and human responses starting from the construction of the training data. This method avoids potential issues such as convergence difficulties and reward hacking in reinforcement learning models [36],[37]. Furthermore, recent research has started to introduce large models, such as ChatGPT, as judges for alignment evaluation. By comparing LLM-generated responses with human responses across multiple dimensions, LLMs have demonstrated a high degree of consistency with human judgments [38],[39]. More importantly, it requires only a subset of validation samples to effectively reflect the overall situation and can quickly adjust the evaluation dimensions according to user needs, significantly reducing costs.

In this section, we adopt the prompt design of the judge in AlignBench [39], taking into account Factual Accuracy, User Satisfaction, Clarity, and Condensability. We separately score each evaluation dimension using CoT to obtain the final score, thereby enhancing the transparency and credibility of the evaluation process and improving alignment performance. Factual Accuracy considers the accuracy of information in the LLM response and whether it aligns with the facts presented in human responses. User Satisfaction evaluates whether the LLM response adequately and appropriately meets the user's query needs in comparison to human responses. Clarity aims to assess whether the LLM response is concise and clear in comparison to human responses, enhancing user readability. Lastly, Condensability evaluates whether the LLM response is succinct and refined, considering the potential redundancy in the LLM response. Furthermore, by collecting evaluation results, we can understand the strengths and weaknesses of the current prompts across various dimensions, providing instructive feedback for further prompt improvement. Based on this feedback, we can further refine the prompts using reflection [40].

### 4.2 Discrete Prompt Optimization Framework

The application of knowledge distillation using large language models such as ChatGPT has been demonstrated to effectively train student models, thereby enhancing their performance across various NLP tasks [26], [41]. However, the quality of responses generated by ChatGPT heavily depends on the design of the prompts, which has been overlooked in previous work within specialized domains. Different prompts significantly affect the style, accuracy, and professionalism of the responses. Considering the refined language and precise factual responses typical of human replies, we aim to fine-tune the model using distillation data guided by optimal prompts. This will enable the fine-tuned model to enhance the information density of its responses while meeting user requirements, thereby avoiding unnecessary output and improving generation speed.

In this case, we have developed a discrete prompt optimization framework based on reflection [40] and LLM-as-Judge. In this framework, we task LLM-as-Judge with scoring candidate prompts on a randomly sampled set of 50 humandrafted question-answer pairs to assess the quality. The evaluation feedback provided by LLM-as-Judge offers a clear direction for the improvement of the candidate prompts. Consequently, we integrate the feedback with Reflection to guide the learning process of the LLM, thereby enhancing the quality of the prompts. It is noteworthy that LLM-as-Judge initially provides separate evaluations for each dimension and concludes with a comprehensive evaluation and scoring. To reduce the overhead associated with the reflection operation, we randomly collect 5 comprehensive evaluation results as feedback for each candidate prompt, which serve as the basis for subsequent improvements.

However, in this process, we found that although utilizing evaluation results to make improvements could enhance the performance of the prompts and make them more aligned with the evaluator's criteria, it simultaneously tends to generate overly lengthy prompts and encourages ChatGPT to produce longer responses to achieve higher scores. This

![](https://cdn.mathpix.com/cropped/2024_06_04_fa7ab643797b68e77915g-07.jpg?height=455&width=1217&top_left_y=255&top_left_x=451)

Figure 3: Discrete Prompt Optimization Framework

might lead to deviations from human-like response styles and increase the cost of inference. To address this issue, we took the following measures: Firstly, in addition to the LLM scores, we introduced a length penalty factor by referencing the length of human responses to more reasonably balance the comprehensiveness and length of the responses. Secondly, during the prompt optimization process, besides improving the prompts with feedback, we incorporated a Resample module. Specifically, after calculating the scores of candidate prompts using LLM-as-Judge, we select the top five highest-scoring prompts sorted in descending order. These sorted prompt-score pairs are then used as input to guide the large language model to automatically discover the key information within the prompts and the patterns related to the scores, thereby resampling an equivalent number of prompts. This approach allows for the retention of key information within the prompts while also achieving their conciseness.

As illustrated in Figure 3, the proposed optimization framework for discrete prompts is presented. Initially, a set of ten manually crafted prompts are employed as the preliminary candidates, and their respective scores and evaluation outcomes are obtained using LLM-as-Judge. The significance of these initial prompts lies in their capacity to inject the preliminary requirements into the optimization framework for further refinement. It is noteworthy that role-playing is incorporated into the initial prompts to guide the large model to concentrate on professional domains, thereby generating more specialized responses.

In terms of scoring, the calculation method is as described by Equation 1. where $s_{i}^{L L M}$ represents the comprehensive score given by LLM-as-Judge, $l^{\text {res }}$ denotes the length of the response generated by the LLM, and $l^{\text {ref }}$ indicates the length of the reference human response. If the LLM-generated response is shorter than the reference length, no penalty is imposed, and the evaluation is solely based on the LLM-as-Judge's assessment across various dimensions. Conversely, if the LLM-generated response exceeds the reference length, a penalty is applied to the additional length, with the degree of penalty determined by the parameter $\alpha$.

$$
s_{i}= \begin{cases}s_{i}^{L L M} & l^{\text {res }} \leq l^{\text {ref }}  \tag{1}\\ s_{i}^{L L M}-\alpha\left(\frac{l^{\text {res }}}{l^{\text {ref }}}-1\right) & l^{\text {res }}>l^{\text {ref }}\end{cases}
$$

After obtaining the scores and feedback for each candidate prompt, we iteratively generated the next generation of candidate prompts by combining reflection and resampling modules. Initially, we selected the top 5 prompts with the highest scores for subsequent operations. The reflection method effectively utilizes the evaluation results from LLM-as-Judge to refine the prompts, ensuring that the generated prompts better meet the requirements of multiple dimensions. Furthermore, by incorporating the resampling module, the LLM can identify patterns and key components, sampling a new generation of prompts. During optimization, the top 5 prompts might include candidates generated from the two aforementioned modules in the previous iteration. In such cases, the resampling module can leverage the key information from the prompts optimized by both modules.

In the experiment, we set the parameter $\alpha$ to 0.5 and conducted three iterations, and we selected the prompt term with the highest score on the validation set as the optimal prompts.

## 5 Experiment and Analysis

### 5.1 Model Training

We conducted fine-tuning based on ChatGLM3-6B as the foundational architecture. ChatGLM3-6B is a pre-trained large language model tailored for Chinese, which possesses excellent features such as conversational fluency and low deployment threshold. In the instruction fine-tuning, we employed the LoRA strategy [42], with the rank textbfk set to 64 and textbfalpha set to 128 . Additionally, the learning rate, batch size, and maximum context length were set to $1 \mathrm{e}-4$, 32, and 2048, respectively. Regarding training facilities, we distributed the model across 8 A40 GPUs using Pytorch to accelerate the training process.

### 5.2 Benchmarks

We constructed a test dataset consisting of 200 QA pairs derived from examination papers and web pages in the field of communication principles, encompassing various chapters within this domain to evaluate the model's proficiency in various knowledge. In our tests, we selected several open-source and closed-source models that support Chinese, including Qianfan-llama2-7b-chinese, Qianfan-llama2-13b-chinese, chatglm3-6B, ERNIE-Bot-turbo [43], and ChatGPT, for comparative purposes.

Among them, Qianfan-llama2-7b-chinese and Qianfan-llama2-13b-chinese are versions based on Llama-2 that have undergone enhanced pre-training with large-scale Chinese-English corpora and fine-tuning for instruction following, which has improved their performance in Chinese-English question-answering. ChatGLM3-6B is a bilingual dialogue language model released by Zhipu AI and the KEG Laboratory of Tsinghua University, demonstrating optimal performance among models below 10B parameters in semantics, mathematics, reasoning, and coding. ERNIE-Botturbo, developed by Baidu, encompasses a vast amount of Chinese data, exhibiting impressive capabilities in Chinese question-answering and Chinese content generation. Lastly, ChatGPT is recognized as one of the most advanced models. To verify the effectiveness of prompt optimization, we also conducted a comparative analysis of the response quality of GPT-3.5-turbo under the guidance of optimal prompts.

### 5.3 Evaluation Metrics

We employed the BLEU[44], GLEU[45], and ROUGE[46] metrics to evaluate the similarity between the responses generated by LMM and the reference human responses. BLEU is an accuracy-based evaluation method that assesses the similarity between LLM responses and reference responses through the overlap precision of n-grams. GLEU further takes into account factors such as lexical overlap and order, reflecting the fluency and naturalness of sentences. Unlike BLEU, ROUGE primarily focuses on the recall of n-grams, based on the comprehensiveness and coverage of the LLM responses. Finally, ROUGE-L is based on the calculation of the longest common subsequence of matches.

However, these metrics are solely based on n-grams matching and cannot evaluate the alignment of semantics between LLM responses and reference human responses. Therefore, we also incorporated LLM-as-Judge to assess the quality of responses in terms of factual accuracy, user engagement, clarity, and conciseness. Additionally, we recorded the LLM-as-Judge scores, length penalty scores, and comprehensive scores to accurately and comprehensively reflect the quality of the responses.

### 5.4 CourseGPT-zh

All models were scored on traditional Natural Language Processing (NLP) metrics as detailed in Table 1. For each comparative model, we utilized the officially provided APIs or checkpoint models to obtain responses on the test set. As depicted in the table, closed-source models such as ERNIE-Bot-turbo and ChatGPT exhibit significant advantages over open-source models due to their larger parameter counts and the benefits of pre-trained corpora. Additionally, compared to the performance of ChatGPT without the use of additional prompts, the application of optimal prompts led to effective improvements across all metrics for ChatGPT, particularly enhancing the fluency of responses by approximately $17 \%$. This validates the efficacy of the discrete prompt optimization framework we proposed. Surprisingly, the fine-tuned ChatGLM3 surpassed ChatGPT on all metrics, especially achieving a higher recall rate in the ROUGE metric. This may be attributed to its training on the data distilled under the guidance of optimal prompts.

However, these traditional metrics have clear limitations as they only focus on the calculation of n-gram-related accuracy or recall rates, lacking the ability to evaluate complex semantic information. Therefore, we continued to use LLM-as-Judge to evaluate the response quality of each model, as shown in Table 2 Firstly, by comparing the results of the open-source models, it can be observed that in the field of domain-specific question-answering, simply increasing the number of model parameters does not result in significant progress. Instead, training with specialized and high-quality

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | GLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qianfan-llama2-7b-chinese | 0.157 | 0.073 | 0.037 | 0.018 | 0.056 | 0.254 | 0.059 | 0.193 |
| Qianfan-llama2-13b-chinese | 0.176 | 0.081 | 0.040 | 0.021 | 0.060 | 0.258 | 0.062 | 0.190 |
| ChatGLM3-6B | 0.166 | 0.075 | 0.039 | 0.020 | 0.056 | 0.245 | 0.053 | 0.180 |
| ERNIE-Bot-turbo | 0.180 | 0.083 | 0.041 | 0.021 | 0.060 | 0.247 | 0.057 | 0.178 |
| ChatGPT | 0.229 | 0.108 | 0.055 | 0.028 | 0.077 | 0.274 | 0.068 | 0.202 |
| ChatGPT-prompt | $\mathbf{0 . 2 5 4}$ | 0.117 | 0.059 | 0.030 | $\mathbf{0 . 0 9 0}$ | $\mathbf{0 . 3 0 0}$ | 0.072 | $\mathbf{0 . 2 2 2}$ |
| CourseGPT-zh | 0.253 | $\mathbf{0 . 1 2 0}$ | $\mathbf{0 . 0 6 3}$ | $\mathbf{0 . 0 3 3}$ | 0.088 | 0.297 | $\mathbf{0 . 0 7 6}$ | 0.218 |

Table 1: Benchmark on QA dataset.(zero-shot)

corpora is necessary. The superior performance of ChatGLM3-6b among open-source models may be related to its broader Chinese pretraining corpus. Furthermore, by comparing the results of open-source and closed-source models, it can be seen that although both tend to generate long responses to meet user needs, there are significant differences in response quality. The model with the highest comprehensive score is ChatGPT, while ERNIE-Bot-turbo has the highest response quality, albeit with excessively long responses.

In comparison to the responses generated by ChatGPT under the guidance of optimal prompts, the reduction in length penalty is more than threefold, while achieving nearly identical response quality. This indicates that under the guidance of optimal prompts, the density of effective information in its responses has significantly increased, taking into account multiple dimensions such as accuracy and responsiveness to user needs. Furthermore, the response quality of the fine-tuned ChatGLM3 model significantly surpasses that of various open-source models, and due to its refined responses, it has obtained the second-highest overall score. Compared to the ChatGLM3-6B model without fine-tuning, it achieves better response quality while reducing the length penalty by $63 \%$. This demonstrates that fine-tuning with specialized distillation corpora in a specific style can significantly enhance the response quality of open-source models.

In comparison to the vanilla ChatGPT, the ChatGPT guided by optimal prompts exhibited a reduction in length penalty by more than threefold, while maintaining nearly equivalent response quality. This demonstrates that, under the guidance of optimal prompts, there is a significant enhancement in the density of effective information, satisfying multidimensional requirements such as accuracy and responsiveness to user needs. Furthermore, the fine-tuned ChatGLM3-6B model significantly outperformed various open-source models in terms of response quality, and it achieved the second-highest comprehensive score due to its refined answers. This indicates that fine-tuning with specialized distillation corpora of a specific style can markedly improve the response quality of open-source models.

It is worth noting that the LLM-as-Judge is tested based on ChatGPT. With the evolvement of ChatGPT, the evaluation scores might change in the future, so it is necessary to pay attention to the relative scores.

| Model | Comprehensive Score | LLM-as-Judge | Length Penalty |
| :--- | :--- | :--- | :--- |
| Qianfan-llama2-7b-chinese | 4.64 | 5.54 | 0.90 |
| Qianfan-llama2-13b-chinese | 4.92 | 5.84 | 0.92 |
| ChatGLM3-6B | 5.28 | 6.21 | 0.93 |
| ERNIE-Bot-turbo | 5.72 | $\mathbf{6 . 7 5}$ | 1.03 |
| ChatGPT | 6.14 | 6.69 | 0.55 |
| ChatGPT-prompt | $\mathbf{6 . 4 8}$ | 6.65 | $\mathbf{0 . 1 7}$ |
| CourseGPT-zh | 6.21 | 6.55 | 0.34 |

Table 2: The model scores on 200 single-turn questions, using LLM-as-Judge (ChatGPT)

### 5.5 Discrete prompt optimization framework

As shown in Table 3, three types of prompts were collected during the optimization process. The first prompt with the highest comprehensive score was selected as the optimal prompts for subsequent data construction. This prompt utilized role-playing and extracted the most critical semantic components from the optimization experience, achieving a balance between answer quality and length. The second prompt was generated through the reflection module, which effectively integrated the experience of feedback. However, this prompt was overly comprehensive, resulting in the best answer quality but at the cost of excessively long responses. Compared to the non-optimized prompt, its length penalty increased by 0.3. The last prompt, although capable of generating brief responses, failed to guide the generation of high-quality content. This demonstrates that under nearly the same response length, the design of the prompt has a significant impact on the quality of the response. Additionally, during the prompt optimization process, the quality of the prompts did not consistently improve but experienced certain fluctuations.

| Prompts | Comprehensive <br> score | LLM-as-Judge | Length <br> Penalty |
| :---: | :---: | :---: | :---: |
| 你是一位通信工程领域的专家, 你以简洁明了的方 <br> 式提供准确无误的回答。你的回答简洁明了、准确无 <br> 㻍, 避免长和琐。 <br> As an expert in the field of telecommunications engineering, <br> you provide concise and unambiguous responses that are <br> error-free. Your answers are succinct and precise, devoid of <br> superfluity and complexity. | 6.48 | 6.65 | 0.17 |
| 你是一位通信工程领域的教授, 专注于通信原理领 <br> 域。以深厚的知识和清晰的表达著称。你的回答结构 <br> 化、简明抢要、易于理解、准确无误、全面清晰。你 <br> 提供准确信息, 确保回答满足用户需求。 <br> As a professor in the field of telecommuncations engi- <br> neering, specializing in communication principles, you are <br> renowned for your profound knowledge and clear articula- <br> tion. Your responses are structured, concise, and easy to <br> understand, while also being accurate, comprehensive, and <br> clear. You provide accurate information to ensure that your <br> answers meet the user's needs. | 6.37 | 7.23 | 0.85 |
| 请用简洁明了的语言回答以下问题, 确保回答准确无 <br> 误、全面清晰。请核实事实,满足用户需求, 并尽量 <br> 使通焀的语言, 简化句子结构, 提高答的凝 <br> 炼性。 <br> Please answer the following questions with concise and <br> clear language, ensuring that the answers are accurate, com- <br> prehensive, and lucid. Verify the facts, meet the user's needs, <br> and strive to use easily understandable language. Simplify <br> the sentence structure to enhance the conciseness of the <br> responses. | 5.98 | 6.17 | 0.19 |

Table 3: Sample prompts from discrete prompt optimization.

### 5.6 Case study

Question: What are the impacts of inter-symbol interference?

As shown in Table 4, we present case studies of responses from different models to the same question. Firstly, ChatGPT guided by optimal prompts provided a refined response and was able to correctly answer the question, covering aspects such as bit error rate and communication quality. However, ChatGLM3-6B explained the concept of inter-symbol interference but failed to correctly respond to the question. Finally, CourseGPT-zh, with a similarly refined style, provided explanations regarding bit error rate, system capacity, and modulation performance, correctly addressing the user's inquiry.

## 6 Conclusion and Future work

In this paper, we introduce CourseGPT-zh, an open-source education large language model for professional courses, and the corresponding question-answering data construction framework. Based on this framework, we have implemented a communication principles specialized chatbot. Unlike previous work, we have focused on the comprehensiveness and diversity of questions, as well as the alignment of question responses with human needs. Furthermore, we have integrated discrete prompt optimization to enhance response quality, with LLM-as-judge for automatic multidimensional evaluation. Finally, we have trained CourseGPT-zh using the parameter-efficient fine-tuning method LoRA. Experimental results show that CourseGPT-zh exhibits impressive performance in question-answering for the corresponding course topics.

| Model/References | Response |
| :---: | :---: |
| References | ![](https://cdn.mathpix.com/cropped/2024_06_04_fa7ab643797b68e77915g-11.jpg?height=502&width=1324&top_left_y=303&top_left_x=542) |
| ChatGPT-prompt | 码间串扰是指不同码片序列之间相互干扰的现象, 在通信系统中会产生以下影响: <br> 1.码误差:码间串扰会号致接收端无法正确解码接收到的信号.....降低通信系统的性能。 <br> 2.信号衰减:码间串扰会引起信的衰减......致通信量下降。 <br> 3.互调干扰:码间串扰还可能引起互调干扰.....影响统的动态范围和信号的有效传输。 <br> 4.报文错误率增加:码间串扰会增加报文的错误率......码间串扰对通信系统的性能和质量都会产 <br> 生不利的影响。 <br> Inter-symbol interference refers to the phenomenon of interference between different chip sequences, <br> which has the following effects in communication systems: <br> 1. Code Error: Inter-symbol interference can cause the receiver to be unable to decode the received signal <br> correctly ... thus degrading the performance of the communication system. <br> 2. Signal Attenuation: Inter-symbol interference can cause signal attenuation ... leading to a decrease in <br> communication quality. <br> 3. Intermodulation Interference: Inter-symbol interference may also cause intermodulation interference ... <br> affecting the dynamic range of the system and the effective transmission of signals. <br> 4. Increased Message Error Rate: Inter-symbol interference can increase the error rate of messages ... <br> inter-symbol interference has an adverse effect on the performance and quality of communication systems. |
| ChatGLM3-6B | ![](https://cdn.mathpix.com/cropped/2024_06_04_fa7ab643797b68e77915g-11.jpg?height=544&width=1324&top_left_y=1378&top_left_x=542) |
| CourseGPT-zh | 码间串扰会产生以下影响: <br> 1.增加误码率码间串扰会引入额外的噪声..... 从而增加误码率。 <br> 2.降低系统量码间串扰会限制不同信号之间的距离.....而降低统的容量。 <br> 3.影响调制性能码间串扰会改交信号的频谐特性.....影响系统的传输质量。 <br> 4.增加信号功率:为了抵消码间串扰引起的码率增加....增加了功耗和成本。 <br> 因此.....需采取合适的调制方案、均衡技术和干扰抑制方法来提高系统的性能。 <br> Inter-symbol interference (ISI) results in the following effects: <br> 1. Increased bit error rate: ISI introduces additional noise ..., thereby increasing the bit error rate. <br> 2. Reduced system capacity: ISI constrains the distance between different signals ..., thus decreasing the <br> system capacity. <br> 3. Impacted modulation performance: ISI alters the frequency characteristics of the signals ..., affecting <br> the transmission quality of the system. <br> 4. Increased signal power: In order to counteract the increased bit error rate caused by ISI ..., signal power <br> and cost are increased. <br> Therefore ..., appropriate modulation schemes, equalization techniques, and interference suppression <br> methods are required to enhance system performance. |

Table 4: Sample responses from three models.

However, due to the limitations of the model's parameter size, particularly its pre-trained knowledge and reasoning capabilities, further work is needed to expand and enhance the performance of CourseGPT-zh. Future directions for development are outlined below.

Reducing Hallucinations For specialized courses, the accuracy of responses is crucial for the user experience. However, even the most advanced models, such as GPT4, struggle with the issue of hallucinations, which is more pronounced in 6B parameter models. Furthermore, the performance of the $6 \mathrm{~B}$ model in answering questions about specific professional knowledge is constrained by the limitations of its pre-trained corpus. To address these issues, we plan to construct a knowledge base of specialized knowledge based on textbooks and encyclopedias in the future and utilize Retrieval Augmented Generation (RAG) technology to provide references for response generation. This approach can reduce the model's hallucination problems and increase the accuracy of responses. Moreover, to address the potential changes in response distribution after incorporating references, we can use prompt optimization and joint adjustment with RAG to achieve the desired response style.

High-quality Professional Knowledge Base CourseGPT-zh conducts knowledge extraction and question-answer pair construction based on a high-quality professional knowledge base. The richness and comprehensiveness of the professional knowledge base have a significant impact on the quality of the constructed data. Moreover, the most up-to-date knowledge base can prompt large models to generate the latest professional content. However, in the current work, data construction is only based on the main reference books of the corresponding courses. In the future, more field-related professional knowledge bases will be introduced to further improve the quality and quantity of the data and to further tap the potential of large language models.

General Tasks and Extended Capabilities Currently, CourseGPT-zh is optimized for single-turn question-answering. However, multi-turn question-answering is also of great importance in future work. To improve the quality of multi-turn question-answering structured data in vertical domains, it is necessary to design a dedicated framework to ensure that large models can accurately understand contextual needs and refer to professional knowledge during the data construction process. Furthermore, other extended capabilities, such as Socratic teaching and problem-solving reasoning, are also very important for large language models in the field of education. These capabilities need to be further expanded in future work.

Finally, in light of the potential social risks posed by CourseGPT-zh, it is imperative to further enhance its security measures to prevent malicious utilization.

## References

[1] OpenAI ChatGPT. optimizing language models for dialogue. openai. 2022, 2023.

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[3] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[4] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022.

[5] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.

[6] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023.

[7] Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-ofthought. arXiv preprint arXiv:2210.01240, 2022.

[8] Yuhao Dan, Zhikai Lei, Yiyang Gu, Yong Li, Jianghao Yin, Jiaju Lin, Linhao Ye, Zhiyan Tie, Yougen Zhou, Yilei Wang, et al. Educhat: A large-scale language model-based chatbot system for intelligent education. arXiv preprint arXiv:2308.02773, 2023.

[9] Zhouhong Gu, Xiaoxuan Zhu, Haoning Ye, Lin Zhang, Jianchen Wang, Sihang Jiang, Zhuozhi Xiong, Zihan Li, Qianyu He, Rui Xu, et al. Xiezhi: An ever-updating benchmark for holistic domain knowledge evaluation. arXiv preprint arXiv:2306.05783, 2023.

[10] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint arXiv:2311.10537, 2023.

[11] Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, et al. Can generalist foundation models outcompete special-purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452, 2023.

[12] Yang Tan, Mingchen Li, Zijie Huang, Huiqun Yu, and Guisheng Fan. Medchatzh: a better medical adviser learns from better instructions. arXiv preprint arXiv:2309.01114, 2023.

[13] Xuanyu Zhang and Qing Yang. Xuanyuan 2.0: A large chinese financial chat model with hundreds of billions parameters. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 4435-4439, 2023.

[14] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. 2022.

[15] Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. Chatlaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092, 2023.

[16] Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. Fingpt: Open-source financial large language models. arXiv preprint arXiv:2306.06031, 2023.

[17] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36, 2024.

[18] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.

[19] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.

[20] Hongcheng Guo, Jian Yang, Jiaheng Liu, Liqun Yang, Linzheng Chai, Jiaqi Bai, Junran Peng, Xiaorong Hu, Chao Chen, Dongfeng Zhang, et al. Owl: A large language model for it operations. arXiv preprint arXiv:2309.09298, 2023.

[21] Dongyang Li, Ruixue Ding, Qiang Zhang, Zheng Li, Boli Chen, Pengjun Xie, Yao Xu, Xin Li, Ning Guo, Fei Huang, et al. Geoglue: A geographic language understanding evaluation benchmark. arXiv preprint arXiv:2305.06545, 2023.

[22] Haoyang Huang, Tianyi Tang, Dongdong Zhang, Wayne Xin Zhao, Ting Song, Yan Xia, and Furu Wei. Not all languages are created equal in llms: Improving multilingual capability by cross-lingual-thought prompting. arXiv preprint arXiv:2305.07004, 2023.

[23] Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. Extrapolating large language models to non-english by aligning languages. arXiv preprint arXiv:2308.04948, 2023.

[24] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.

[25] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

[26] Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Jianquan Li, Guiming Chen, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, et al. Huatuogpt, towards taming language model to be a doctor. arXiv preprint arXiv:2305.15075, 2023.

[27] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. AI Open, 2023.

[28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022.

[29] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.

[30] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021.

[31] Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. arXiv preprint arXiv:2309.08532, 2023.

[32] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910, 2022.

[33] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. arXiv preprint arXiv:2309.03409, 2023.

[34] Xinyuan Wang, Chenxi Li, Zhen Wang, Fan Bai, Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric P Xing, and Zhiting Hu. Promptagent: Strategic planning with language models enables expert-level prompt optimization. arXiv preprint arXiv:2310.16427, 2023.

[35] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.

[36] Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. Warm: On the benefits of weight averaged reward models. arXiv preprint arXiv:2401.12187, 2024.

[37] Lichang Chen, Chen Zhu, Davit Soselia, Jiuhai Chen, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. Odin: Disentangled reward mitigates hacking in rlhf. arXiv preprint arXiv:2402.07319, 2024.

[38] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2024.

[39] Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, et al. Alignbench: Benchmarking chinese alignment of large language models. arXiv preprint arXiv:2311.18743, 2023.

[40] Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning.(2023). arXiv preprint cs.AI/2303.11366, 2023.

[41] Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301, 2023.

[42] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[43] Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137, 2021.

[44] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311-318, 2002.

[45] Andrew Mutton, Mark Dras, Stephen Wan, and Robert Dale. Gleu: Automatic evaluation of sentence-level fluency. In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics, pages 344-351, 2007.

[46] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74-81, 2004.


[^0]:    ${ }^{*}$ Xing zhang is the corresponding author.

</end of paper 1>


<paper 2>
# MECHANISM DESIGN FOR LLM FINE-TUNING WITH MULTIPLE REWARD MODELS 

Haoran Sun<br>Peking University<br>sunhaoran0301@stu.pku.edu.cn

Wei Chen<br>Microsoft Research Asia<br>weic @microsoft.com

Yurong Chen

Peking University

chenyurong@pku.edu.cn
Siwei Wang

Microsoft Research Asia siweiwang@microsoft.com
Xiaotie Deng

Peking University

xiaotie@pku.edu.cn


#### Abstract

Recent research on fine-tuning large language models (LLMs) through the aggregation of multiple preferences has attracted considerable attention. However, the existing literature predominantly focuses on the empirical performance of aggregation algorithms, while neglecting the underlying motivation for agents to misreport their preferences. In this paper, we formalize this as a multiparameter mechanism design problem, where an LLM provider designs both training and payment rules to achieve specific objectives and promote the truthful reporting of preferences. Firstly, we claim the necessity of a payment scheme by demonstrating that without payments, truth-telling is a strictly dominated strategy under a wide range of training rules. Then, we introduce the affine maximizer payment scheme for the social welfare maximizing training rules that are widely used in practice, which ensures both dominant-strategy incentive compatibility (DSIC) and individual rationality (IR). Furthermore, we prove that under mild conditions, any other payment rule that also implements these training rules in DSIC can be converted to the affine maximizer payment by adding a factor irrelevant to the agents' own reports. We also show that this mechanism satisfies approximate DSIC when the input of the mechanism is a biased version of the reported preferences, showcasing its robustness in real-world applications.


## 1 Introduction

The pre-training and fine-tuning paradigm is fundamental in developing language models (Devlin et al. [2018], Radford et al. [2018], Liu et al. [2019], Touvron et al. [2023]). During pre-training, the model is fed with vast amounts of data to acquire a general capability to understand and generate language through self-supervised learning. The subsequent fine-tuning phase customizes these pre-trained models for specific downstream tasks using smaller, taskoriented datasets, ensuring that the model outputs are more closely aligned with particular requirements. As LLMs gain increasing popularity, there is a growing demand for fine-tuning basic LLMs, as basic models often fail to meet users' demands, especially in catering to individual preferences.

The process of fine-tuning an LLM to align with certain human preferences is challenging to achieve through supervision (Ji et al. [2023], Köpf et al. [2024], Wang et al. [2023b], Shen et al. [2023]), primarily due to the difficulty in constructing datasets with a substantial number of valid question-answer pairs for supervised training. Reinforcement learning from human feedback (RLHF) (Ouyang et al. [2022], Christiano et al. [2017]) offers a promising solution to this problem. In RLHF, a reward model is first trained to be used as a proxy for human judgment. This model then provides reward signals for the standard reinforcement learning process. This technique of fine-tuning with a reward model has proven effective in encoding human preferences into models and has become a fundamental component of the training process for most advanced LLMs. With the advancement of RLHF, numerous studies have investigated efficient methods for aggregating multiple preferences into a single fine-tuned model.

However, most of these studies focus primarily on improving empirical performance across various metrics (Ramé et al. [2024], Wu et al. [2024], Jang et al. [2023], Coste et al. [2023], Zhang et al. [2024], Wang et al.

[2024], Eisenstein et al. [2023]). They often implicitly assume that we are accessible to real preferences, neglecting the possibility of agents' misreporting their preferences. This problem becomes more crucial when we consider a real-world scenario, where different agents provide their preferences for the aggregation. In such cases, agents may engage in strategic misreporting to increase their utility. An intuitive example is that if an agent knows beforehand that the fine-tuning process aims to neutralize all preferences, it might pretend to have a more polarized preference as a beneficial strategy. These strategic behaviors can distort the final training results, even if the trained algorithm is highly effective. Nevertheless, this issue has not attracted sufficient attention in the existing literature, particularly concerning the fine-tuning process of LLMs.

Our Contribution. In this paper, we mainly study the incentive design in such scenarios. First, we formalize this as a multi-parameter mechanism design problem between a fine-tuning service provider and groups of agents seeking fine-tuning services. The provider proposes a mechanism that includes a training rule for integrating different groups' preferences into a fine-tuned model and a payment rule to charge the groups. After observing the mechanism, each group strategically reports its preference to maximize its utility. We consider that the subsequent fine-tuning process is implemented using RLHF, a standard method for aligning a model with human preference. Therefore, we abstract the preference of each group to be reward models, and term the whole scenario the RLHF Game.

Secondly, we demonstrate the profitability of misreporting a polarized preference under a wide range of mechanisms that include only a training rule (Theorem 3.3). This underscores the necessity of a payment rule to address incentive issues.

Thirdly, we focus on a representative set of training rules, termed the SW-Maximizing training rules, in which the provider aims to maximize social welfare while incorporating different regularization measures. For SW-Maximizing training rules, we propose the affine maximizer payment scheme, a weighted version of the Vickrey-Clarke-Groves (VCG) payment Vickrey [1961], Clarke [1971], Groves [1973]). We prove that agents truthfully reporting their preferences constitutes a dominant strategy in such mechanisms (Theorem 4.2). Utilizing the notion of payment equivalence, we prove that under a mild condition, any other payment rule that also implements these training rules in dominantstrategy incentive compatibility (DSIC) can be converted to the affine maximizer payment by adding a factor irrelevant to agents' own reports (Theorem4.5). We validate this condition for many commonly used regularization terms like KL-divergence (Proposition 4.4). Consequently, we derive the revenue-maximizing payment rule that implements SW-Maximizing training rules in both DSIC and individual rationality (IR) (Corollary 4.6). Furthermore, we show that this mechanism remains approximately DSIC when the input of the mechanism is a biased version of the reported preferences, which is an abstraction modeling for the inevitable errors that occur in practice. This showcases the robustness of the proposed mechanisms in real-world applications (Theorem 4.9).

Primary Related Work. Several studies have investigated similar scenarios. Among them, Duetting et al. [2023] and Soumalias et al. [2024] are most related to ours. Duetting et al. [2023] examines the problem of designing a mechanism to aggregate multiple agents' preferences based on each agent's bids and determine their payments. However, they exclude the case where preferences can be misreported, which is the primary concern in our study. The concurrent work by Soumalias et al. [2024] also considers the mechanism design for aggregating multiple preferences. Their focus is mainly on the practical implementation of SW-Maximizing training rule with KL-divergence and the payment scheme that obtains both DSIC and interpretability. However, in this scenario, we are more concerned with the theoretical properties of more general mechanisms, including the implementability and the property of payment equivalence.

Additionally, there are works studying other scenarios related to LLMs from the perspective of algorithmic game theory. Laufer et al. [2023] abstracts the fine-tuning process as a bargaining game and characterizes the perfect subgame equilibria. Dubey et al. [2024] proposes an auction where bidders compete to place their content within a summary generated by an LLM. Conitzer et al. [2024] considers incorporating social choice theory in LLM alignment. Feizi et al. [2023] explores the potential for leveraging LLMs in online advertising systems.

Paper Organization. In Section 2, we provide the preliminaries and the formal description of the RLHF Game. In Section 3, we study the incentive design for general training rules in the RLHF Game. We demonstrate the properties of mechanisms that consist of SW-Maximizing training rules and payment rules in Section4, Further related work is provided in Section 5, and we conclude in Section6.

## 2 Preliminaries and Model

### 2.1 Preliminaries

Large Language Models. Large language models (LLMs) function as mappings from a sequence of tokens to a probability distribution over the next token. The input sequence is usually constrained by a maximum length $K$, thereby making the set of all possible inputs finite. Let $T$ denote the finite set of all tokens, and let $T^{*}:=\emptyset \cup T \cup T^{2} \cup$ $\cdots \cup T^{K}$ represent the set of all possible input sequences with lengths up to $K$.

An LLM parameterized by $\theta \in \Theta$ is denoted as $g_{\theta}: T^{*} \rightarrow \Delta T$, where $\Delta T$ is the set of all probability distributions over the token set $T$. For practical purposes, the output sequence is also required to be of finite length. We assume the maximum output length is also $K$ so that the output space is also $T^{*}$. We denote $\operatorname{LLM}_{\theta}(\boldsymbol{x})$ as the probability of a sequence of tokens $\boldsymbol{x} \in T^{*}$ generated by $g_{\theta}$. Since the model generates a sequence by predicting the next token iteratively until a special ending token is encountered, the relationship between $\mathrm{LLM}_{\theta}$ and $g_{\theta}$ is given by:

$$
\operatorname{LLM}_{\theta}(\boldsymbol{x})=\prod_{t=1}^{|\boldsymbol{x}|} g_{\theta}\left(x_{t} \mid \boldsymbol{x}_{<t}\right)
$$

where $\boldsymbol{x}_{<t}$ denotes the prefix subsequence of $\boldsymbol{x}$ preceding $x_{t}$ and $\boldsymbol{x}_{<1}=\emptyset$. LLM $_{\theta}$ is a distribution over $T^{*}$ and can be represented as a $\left|T^{*}\right|$-dimensional vector, with each coordinate $\boldsymbol{x}$ the probability of $\boldsymbol{x}$ being generated under $g_{\theta}$.

Reward Modeling. Reward modeling is instrumental for aligning LLMs with human preferences, particularly within the context of RLHF. In this process, a reward model $\mathrm{rm}: T^{*} \rightarrow \mathbb{R}$ is first trained on the human-annotated preference dataset by using the Bradley-Terry model (Bradley and Terry [1952]). Essentially, the reward model is a function that maps a sequence of tokens to a real number indicative of the preference for that sequence. Similar to $\mathrm{LLM}_{\theta}$, rm can be also considered as a $\left|T^{*}\right|$-dimensional vector. Following prior empirical work for RLHF (Rafailov et al. [2023]), we consider the normalized reward models which are normalized to have the summation 1 over $T^{*}$ i.e. $\sum_{\boldsymbol{x} \in T^{*}} \operatorname{rm}(\boldsymbol{x})=1$. Furthermore, we also assume that the output rewards are all non-negative, i.e. $\operatorname{rm}(\boldsymbol{x}) \geq 0$ for all $\boldsymbol{x} \in T^{*}$. Unless otherwise stated, we use $\mathcal{R}$ to denote the domain of all reward model functions that satisfy the above conditions. In fact, the results in our paper are also applicable for some other normalization methods like $\max _{\boldsymbol{x} \in T^{*}} \operatorname{rm}(\boldsymbol{x})=1$.

### 2.2 Formulation of the RLHF Game

In this part, we present the formal description of the RLHF Game. There is one LLM provider and $n$ groups of agents, denoted by $[n]=\{1,2, \cdots, n\}$. The provider has an initial model $\operatorname{LLM}_{\theta_{\text {init }}}$ with non-negative probability for all sequences, i.e. $\operatorname{LLM}_{\theta_{\text {init }}}(\boldsymbol{x})>0$ for all $\boldsymbol{x} \in T^{*}$. Each group $i$ has $w_{i}$ agents and a joint preference represented by a reward model $\mathrm{rm}_{i}$. Let $\mathcal{R}$ and $\mathcal{W} \subseteq \mathbb{N}_{+}$denote the domains for each group's reward model and group size, respectively. We assume an upper bound $\bar{w}$ for $\mathcal{W}$. The exact reward model $\mathrm{rm}_{i}$ and the size $w_{i}$ are group $i$ 's private information. For an agent in group $i$, the valuation when it receives a model $\operatorname{LLM}_{\theta}$ is denoted by $v_{i}\left(\theta ; \mathrm{rm}_{i}\right)$. The form of the valuation function $v_{i}(\cdot ; \cdot)$ is known by both the provider and the agents.

The provider first announces the mechanism, including a training rule $\psi$ and a payment rule $p$,

$$
\psi: \mathcal{R}^{n} \times \mathcal{W}^{n} \times \Theta \rightarrow \Theta, \quad p: \mathcal{R}^{n} \times \mathcal{W}^{n} \times \Theta \rightarrow \mathbb{R}^{n}
$$

Both rules take $n$ reported reward models, $n$ reported sizes, and an initial model as input, and output the objective fine-tuned model and each group's payment, respectively. The provider can choose not to charge the users by setting $p$ always equal to 0 . In this case, the model coincides with most previous work, where agents' incentives are not considered (Ramé et al. [2024], Wu et al. [2024], Jang et al. [2023], Coste et al. [2023], Zhang et al. [2024], Wang et al. [2024], Eisenstein et al. [2023]). Specifically, the training rule seeks to find the model that maximizes a certain objective function $f$. That is,

$$
\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right) \in \arg \max _{\theta \in \Theta} f\left(v_{1}\left(\theta ; \operatorname{rm}_{1}\right), \cdots, v_{n}\left(\theta ; \operatorname{rm}_{n}\right), \vec{w}, D\left(\operatorname{LLM}_{\theta} \| \operatorname{LLM}_{\theta_{\text {init }}}\right)\right)
$$

where $D$ is a measure of the distance between $\operatorname{LLM}_{\theta}$ and $\operatorname{LLM}_{\theta_{\text {init }}}$. We assume that the function $f$ has a unique global optimal point for any possible inputs. Hence, in the rest of the paper, the " $\in$ " in the definition of $\psi$ is substituted by "=".

After observing the announced mechanism $(\psi, p)$, each group $i$ reports a reward model, $\widetilde{\mathrm{rm}}_{i}$, and its group size $\tilde{w}_{i}$. We assume all reported sizes are in $\mathcal{W}$ and therefore bounded by $\bar{w}$. Based on the reported information, the provider fine-tunes the model until the model $\operatorname{LLM}_{\theta_{\text {final }}}$ is optimal, i.e., the final parameter satisfies $\theta_{\text {final }}=\psi\left(\overrightarrow{\mathrm{rm}}, \overrightarrow{\tilde{w}}, \theta_{\text {init }}\right)$. The
provider then charges group $i$ according to the payment rule, $p_{i}\left(\stackrel{\mathrm{rm}}{\vec{w}}, \overrightarrow{\tilde{w}}, \theta_{\text {init }}\right)$. All the members in the group have access to the fine-tuned model $\theta_{\text {final }}$, so the valuation for group $i$ is $w_{i} v_{i}\left(\theta_{\text {final }} ; \operatorname{rm}_{i}\right)$. We assume all groups have quasi-linear utilities. Therefore, group $i$ 's utility is

$$
u_{i}\left(\overrightarrow{\mathrm{rm}}, \overrightarrow{\tilde{w}} ; \psi, p, \operatorname{rm}_{i}, w_{i}\right)=w_{i} v_{i}\left(\theta_{\text {final }} ; \operatorname{rm}_{i}\right)-p_{i}\left(\overrightarrow{\mathrm{rm}}, \overrightarrow{\tilde{w}}, \theta_{\text {init }}\right)
$$

The groups may strategically report, thus $\overrightarrow{\mathrm{rm}}$ and $\overrightarrow{\vec{w}}$ do not necessarily equal the true $\overrightarrow{\mathrm{rm}}$ and $\vec{w}$. The goal of the LLM provider is to achieve its training objective based on the group's true preferences, taking into account that the misreporting may distort the training outcome. To this end, it is crucial to incentivize all groups to report their information truthfully so that the provider is accessible to the groups' private information. We formally define these desiderata of a mechanism as follows.

Definition 2.1. A mechanism $(\psi, p)$ satisfies dominant-strategy incentive compatibility (DSIC) if $\forall i, \operatorname{rm}_{i}, w_{i}, \mathrm{rm}_{i}^{\prime}, w_{i}^{\prime}$, $\overrightarrow{\operatorname{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}$, we have

$$
\begin{equation*}
u_{i}\left(\left(\operatorname{rm}_{i}, \overrightarrow{\operatorname{rm}}_{-i}\right),\left(w_{i}, \vec{w}_{-i}\right) ; \psi, p, \operatorname{rm}_{i}, w_{i}\right) \geq u_{i}\left(\left(\mathrm{rm}_{i}^{\prime}, \overrightarrow{\mathrm{rm}}_{-i}\right),\left(w_{i}^{\prime}, \vec{w}_{-i}\right) ; \psi, p, \operatorname{rm}_{i}, w_{i}\right) \tag{DSIC}
\end{equation*}
$$

Definition 2.2. A mechanism $(\psi, p)$ satisfies individually rationality (IR) if $\forall i, \operatorname{rm}_{i}, w_{i}, \overrightarrow{\operatorname{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}$, we have

$$
\begin{equation*}
u_{i}\left(\left(\operatorname{rm}_{i}, \overrightarrow{\operatorname{rm}}_{-i}\right),\left(w_{i}, \vec{w}_{-i}\right) ; \psi, p, \operatorname{rm}_{i}, w_{i}\right) \geq 0 \tag{IR}
\end{equation*}
$$

DSIC means that for any group, truthfully reporting the reward model and the group size yields the highest utility, regardless of other groups' reports. IR means that truthfully reporting always yields non-negative utilities. Only when both DSIC and IR are satisfied, all groups are incentivized to participate in this game and report truthfully. When a mechanism $(\psi, p)$ satisfies DSIC, IR, or both DSIC and IR, we say that the payment rule $p$ implements $\psi$ in DSIC, IR or both DSIC and IR. Especially, when we say the implementability of a training rule, we refer to the property of DSIC.

## 3 Incentives for General Training Rules

In this section, we discuss the incentive design within the RLHF Game framework. As a warm-up, we consider a simplified scenario where all group sizes are equal to 1 , i.e., $\vec{w}=1$, and this information is public to all groups and the provider. Consequently, each group is required only to report its reward model. For convenience, we let $\vec{w} \equiv 1$ and omit the notation of $\vec{w}$. Unless stated otherwise, the results directly apply to the more general case where $\vec{w}$ is also private information.

For the valuation function in this section, we consider a reasonable form $v(\cdot ; \cdot)$ defined as follows.

Assumption 3.1. For any agent with preference represented by reward model rm, its valuation on model $\mathrm{LLM}_{\theta}$ is its expected reward on the sequences generated by $\mathrm{LLM}_{\theta}$ :

$$
v(\theta ; \mathrm{rm})=\mathbb{E}_{\boldsymbol{x} \sim \mathrm{LLM}_{\theta}} \operatorname{rm}(\boldsymbol{x})=\sum_{\boldsymbol{x} \in T^{*}} \operatorname{LLM}_{\theta}(\boldsymbol{x}) \operatorname{rm}(\boldsymbol{x})
$$

In practice, this can be obtained by averaging the reward of the sequences sampled from an LLM. We discuss the influence of possible errors in this process in Section 4

### 3.1 Necessity of Payment Rule

We begin by demonstrating the necessity of payment rules to ensure incentive compatibility for training rules under the following assumptions.

Assumption 3.2. (1) For all $i \in[n], \partial f / \partial v_{i}$ exists and $\partial f / \partial v_{i}>0$. $\partial f / \partial D$ exists and $\partial f / \partial D<0$. (2) The distance measure function $D$ satisfies that for all $\boldsymbol{x} \in T^{*}, \partial^{2} D / \partial \operatorname{LLM}_{\theta}(\boldsymbol{x})^{2}$ exists and is positive. (3) For all $\overrightarrow{\mathrm{rm}}$ and $\theta_{\text {init }}$, the fine-tuned model $\theta=\psi\left(\overrightarrow{\mathrm{rm}}, \theta_{\text {init }}\right)$ satisfies that $\operatorname{LLM}_{\theta}(\boldsymbol{x})>0$ for all $\boldsymbol{x} \in T^{*}$.

The rationale of these assumptions is as follows: (1) is that we assume the training process aims to find a model $\mathrm{LLM}_{\theta}$ that not only brings higher valuation for all agents but also remains close to the initial model $\mathrm{LLM}_{\theta_{\text {initi }}}$. (2) is like a convex condition in which we assign an increasingly large penalty on $\operatorname{LLM}_{\theta}(\boldsymbol{x})$ when it becomes farther from $\operatorname{LLM}_{\theta_{\text {init }}}(\boldsymbol{x})$. And (3) is to exclude some extreme training rules that the training outcome remains the same for most input and changes drastically. In practice, (1) is satisfied for most training functions $f$, including those aiming to
maximize social welfare and Nash social welfare. (2) and (3) depend on the choice of the regularization measure $D$ and the strength of regularization. At least, they are satisfied by the commonly used KL-divergence.

Combining these three conditions, we show that when the preference for some $\boldsymbol{x}\left(\sum_{i=1}^{n} \mathrm{rm}_{i}(\boldsymbol{x})\right)$ increases and others remain, the probability of $\boldsymbol{x}$ for the optimal model will also increase. In this case, an intuitive manipulation is that the agent reports a polarized reward model: higher reward value $\widetilde{\mathrm{rm}}(\boldsymbol{x})$ for the $\boldsymbol{x}$ it values most. We show that this strategy will give strictly higher utility to the agent unless the agent is indifferent among outcomes $\boldsymbol{x}$ in a subset $S \subseteq T^{*}$ and does not care about the outcomes outside $S$ at all.

Theorem 3.3. Under Assumption 3.1 and Assumption 3.2 when the payment rule $p \equiv 0$, for any agent $i$, truthfully reporting $r m_{i}$ is a strongly dominated strategy, except for the case: $\exists S \subseteq T^{*}$, such that $r m_{i}(\boldsymbol{x})=1 /|S|$ if $\boldsymbol{x} \in S$ and $r m_{i}(\boldsymbol{x})=0$ if $\boldsymbol{x} \notin S$.

Here, we call a strategy strongly dominated when another strategy yields strictly higher utility regardless of others' reports. Theorem 3.3 tells us that truthful reporting is strongly dominated with only training rules, and thus will not be adopted by rational agents.

### 3.2 Characteristics of Payment Rules

Having established the necessity of payment rules in this scenario, we mainly address two questions in the remainder of this section: First, given a training rule $\psi$, can we find a payment rule $p$ such that the mechanism $(\psi, p)$ satisfies DSIC? This is the so-called implementability of a training rule $\psi$. Second, for an implementable training rule $\psi$, can we identify the relationship between the payment rules ps among all DSIC mechanisms $(\psi, p)$.

We resolve the first question primarily by utilizing the notion of cycle monotonicity, first proposed by Rochet 1987. Cycle monotonicity generalizes monotonicity defined in a single-parameter scenario ([Myerson, 1981]). In the RLHF Game, we define a function as $l\left(\mathrm{rm}^{\prime}, \mathrm{rm} ; \overrightarrow{\mathrm{rm}}_{-i}, \theta_{\text {init }}\right):=v_{i}\left(\psi\left((\mathrm{rm}, \overrightarrow{\mathrm{rm}}-i), \theta_{\text {init }}\right) ; \mathrm{rm}\right)-v_{i}\left(\psi\left(\left(\mathrm{rm}^{\prime}, \overrightarrow{\mathrm{rm}}_{-i}, \theta_{\text {init }}\right)\right) ; \mathrm{rm}\right)$. $l\left(\mathrm{rm}^{\prime}, \mathrm{rm} ; \overrightarrow{\mathrm{rm}}_{-i}, \theta_{\text {init }}\right)$ measures the valuation gains from misreporting $\left(\mathrm{rm}_{i}^{\prime}\right)$ to truthfully reporting $\left(\mathrm{rm}_{i}\right)$ under $\overrightarrow{\mathrm{rm}}-i$ and $\theta_{\text {init }}$. The cycle monotonicity is defined based on this function:

Definition 3.4 (Cycle Monotonicity). The training rule $\psi$ satisfies cycle monotonicity if for any $\mathrm{rm}_{i}, \mathrm{rm}_{i}^{\prime} \in \mathcal{R}_{i}$, any

![](https://cdn.mathpix.com/cropped/2024_06_04_1fc48031c34ce8b3fe25g-05.jpg?height=63&width=1645&top_left_y=1324&top_left_x=240)
have

$$
\sum_{j=0}^{k+1} l\left(\mathrm{rm}_{i}^{j}, \mathrm{rm}_{i}^{j+1} ; \overrightarrow{\mathrm{rm}}_{-i}, \theta_{\text {init }}\right) \geq 0 \quad \mathrm{rm}_{i}^{0}=\mathrm{rm}_{i}^{k+2}:=\mathrm{rm}_{i} \text { and } \mathrm{rm}_{i}^{k+1}:=\mathrm{rm}_{i}^{\prime}
$$

For general training rules, cycle monotonicity is a sufficient and necessary condition for implementability.

Theorem 3.5 (Rochet 1987]). A training rule $\psi$ is implementable if and only if it satisfies cycle monotonicity.

In fact, the proof of Theorem 3.5 is constructive. However, for general implementable training rules, the calculation of the payment rules is too complex to be practical.

The second question is more general, so we primarily consider the concept of payment equivalence (Ashlagi et al., 2010]) for an implementable training rule.

Definition 3.6 (Payment Equivalence). An implementable training rule $\psi$ satisfies payment equivalence if for any two mechanisms $(\psi, p)$ and $\left(\psi, p^{\prime}\right)$ satisfying DSIC, there exists a function $f$ such that

$$
p_{i}^{\prime}\left(\operatorname{rm}_{i}, \overrightarrow{\mathrm{rm}}_{-i} ; \theta_{\text {init }}\right)=p_{i}\left(\mathrm{rm}_{i}, \overrightarrow{\mathrm{rm}}_{-i} ; \theta_{\text {init }}\right)+f\left(\overrightarrow{\mathrm{rm}}_{-i}, \theta_{\text {init }}\right) \quad \forall \mathrm{rm}_{i} \in \mathcal{R}_{i}
$$

Or equivalently, when fixing $\overrightarrow{\mathrm{rm}}_{-i}$ and $\theta_{\text {init }}$, there exits a constant $c$ such that $p_{i}^{\prime}\left(\mathrm{rm}_{i}\right)=p_{i}\left(\mathrm{rm}_{i}\right)+c$ for all $\mathrm{rm}_{i} \in \mathcal{R}_{i}$.

Payment equivalence indicates that the only way to modify a DSIC mechanism $(\psi, p)$ to $\left(\psi, p^{\prime}\right)$ while maintaining incentive compatibility is to add a term that is independent of $i$ 's report to agent $i$ 's payment function $p_{i}$. Thus, the payment equivalence of $\psi$ is sometimes interpreted as the uniqueness of the payment rule $p$ that implements it in DSIC. This notion is strong and useful since when a training rule $\psi$ satisfies payment equivalence and we can figure out one mechanism $(\psi, p)$ that satisfies DSIC, then all the payment rules $p^{\prime}$ that implement $\psi$ in DSIC are characterized. In particular, it is possible to find the revenue-maximizing payment rule $p^{*}$ among all these payment rules that implement $\psi$ in both DSIC and IR.

Payment equivalence is influenced by the domain of the types: reward models and group sizes in the RLHF Game. When $\vec{w} \equiv 1$, the agents only report the reward models whose domain $\mathcal{R}$ contains all normalized reward models rm. Therefore, for all $i \in[n]$, the domain of the whole private information is exactly $\mathcal{R}$, which is a connected set in the Euclidean space. Thus, we can directly apply the result in Nisan et al. [2007] and get the following theorem.

Proposition 3.7. When $\vec{w} \equiv 1$ is public information and the agents only report the reward models, all implementable training rules satisfy payment equivalence.

However, when the group sizes $\vec{w}$ is also a part of the private information for all groups, the domain of the whole private information becomes $\mathcal{W} \times \mathcal{R}$ that is no longer a connected set because $\mathcal{W} \subseteq \mathbb{N}_{+}$. Thus, payment equivalence may not be satisfied for general training rules, and we will study this for a representative set of training rules in the following section.

## 4 Social Welfare Maximizing Mechanism

In this section, we consider the scenario where group $i$ consists of $w_{i}$ agents, and each group must simultaneously report its reward model and size. Our objective is to design a mechanism $(\psi, p)$ that incentivizes each group $i$ to truthfully report both $\mathrm{rm}_{i}$ and $w_{i}$. For general training rules $\psi$, though it is possible to adopt the method used in the constructive proof for Theorem 3.5 to derive the payment rule, the resulting payment rule can be complex and impractical.

Therefore, in this section, our primary focus is on a subset of training rules designed to maximize social welfare under regularization constraints, which is commonly used in practice to aggregate various preferences (Boyd and Vandenberghe [2004], Nocedal and Wright [1999]), balancing efficiency and fairness.

Definition 4.1 (SW-Maximizing Training Rules). Given the reports $\overrightarrow{\mathrm{rm}}, \vec{w}$, and the initial model $\theta_{\text {init }}$, a SWMaximizing training rule fine-tunes the model to maximize the social welfare under a regularization penalty measured by some metric $D$. Formally, it is represented as:

$$
\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)=\arg \max _{\theta \in \Theta} \sum_{i=1}^{n} w_{i} v_{i}\left(\theta ; \operatorname{rm}_{i}\right)-\lambda D\left(\operatorname{LLM}_{\theta} \| \operatorname{LLM}_{\theta_{\text {init }}}\right)
$$

Here, $\lambda$ is a hyperparameter that adjusts regularization strength.

Note that SW-Maximizing training rules constitute a set of training rules. We use $\psi \in \Psi^{S W}$ to indicate that $\psi$ is a member of this set. Furthermore, similar to the (3) in Assumption 3.2, we also assume that for all $\overrightarrow{\mathrm{rm}}, \vec{w}$ and $\theta_{\text {init }}$, the fine-tuned model $\theta=\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)$ satisfies that $\operatorname{LLM}_{\theta}(\boldsymbol{x})>0$ for $\forall \boldsymbol{x} \in T^{*}$. One simple way to achieve it is to set a large $\lambda$ and hence the training result is close enough to $\theta_{\text {init }}$.

### 4.1 Affine Maximizer Payment

We introduce the affine maximizer payment rule (Roberts [1979]) $p^{A F F}$, a weighted version of VCG payment Vickrey [1961], Clarke [1971], Groves [1973]):

$$
\begin{aligned}
p_{i}^{A F F}\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)=A S W_{-i} & \left(\overrightarrow{\mathrm{rm}}, \vec{w}, \psi\left(\overrightarrow{\mathrm{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right) ; \theta_{\text {init }}\right) \\
& -A S W_{-i}\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right) ; \theta_{\text {init }}\right)
\end{aligned}
$$

The notations $A S W\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta ; \theta_{\text {init }}\right)$ and $A S W_{-j}\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta ; \theta_{\text {init }}\right)$ refer to the affine social welfare with and without group $j$ when the reported reward models are $\overrightarrow{\mathrm{rm}}$, the reported number of agents are $\vec{w}$, the initial model is $\operatorname{LLM}_{\theta_{\text {init }}}$, and the parameters of model is $\theta$. The affine social welfare consists of both the groups' valuations and the regularization term. Formally,

$$
\begin{gathered}
A S W\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta ; \theta_{\text {init }}\right):=\sum_{i=1}^{n} w_{i} v_{i}\left(\theta ; \operatorname{rm}_{i}\right)-\lambda D\left(\operatorname{LLM}_{\theta} \| \operatorname{LLM}_{\theta_{\text {init }}}\right) \\
A S W_{-j}\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta ; \theta_{\text {init }}\right):=\sum_{i=1, i \neq j}^{n} w_{i} v_{i}\left(\theta ; \operatorname{rm}_{i}\right)-\lambda D\left(\operatorname{LLM}_{\theta} \| \mid \mathrm{LLM}_{\theta_{\text {init }}}\right)
\end{gathered}
$$

We show that $p^{A F F}$ implements SW-Maximizing training rules in both DSIC and IR, which implies that truthfully reporting both reward models and group sizes constitutes a dominant Nash Equilibrium in this mechanism.

Theorem 4.2. For any $\psi \in \Psi^{S W}$, mechanism $\left(\psi, p^{A F F}\right)$ satisfies DSIC and IR.

Regarding payment equivalence, as we have mentioned in the previous section, the domain $\mathcal{W} \times \mathcal{R}$ is not connected in the Euclidean space since $\mathcal{W} \subseteq \mathbb{N}_{+}$, the results in Nisan et al. [2007] can not be directly applied. However, we show that under the following assumption, SW-Maximizing training rules satisfy payment equivalence.

Assumption 4.3. For any $\epsilon>0$, there exists a $\delta>0$ such that for any $\theta_{\text {init }}, \overrightarrow{\mathrm{rm}}, \overrightarrow{\mathrm{rm}^{\prime}}, \vec{w}$ and $\vec{w}^{\prime}$, if $\max _{\boldsymbol{x} \in T^{*}}\left|\sum_{i=1}^{n}\left(w_{i} \operatorname{rm}_{i}(\boldsymbol{x})-w_{i}^{\prime} \operatorname{rm}_{i}^{\prime}(\boldsymbol{x})\right)\right| \leq \delta$, then $\max _{\boldsymbol{x} \in T^{*}}\left|\operatorname{LLM}_{\theta}(\boldsymbol{x})-\operatorname{LLM}_{\theta^{\prime}}(\boldsymbol{x})\right| \leq \epsilon$, where $\theta:=$ $\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)$ and $\theta^{\prime}:=\left(\overrightarrow{\mathrm{rm}}^{\prime}, \vec{w}^{\prime}, \theta_{\text {init }}\right)$.

This assumption is reasonable for most measures $D$ in SW-Maximizing training rules as the space of $\theta$ is continuous. The continuity ensures that when the reported information $(\overrightarrow{\mathrm{rm}}, \vec{w})$ and $\left(\overrightarrow{\mathrm{rm}}^{\prime}, \vec{w}^{\prime}\right)$ are sufficiently close, the training outcomes $\theta$ and $\theta^{\prime}$ should also be close. Specifically, we validate this assumption for some widely used distance measures.

Proposition 4.4. Assumption 4.3 holds for SW-Maximizing training rules with regularizations KL-divergence, $D_{\mathrm{KL}}(p \| q)=\sum_{\boldsymbol{x} \in T^{*}} p(\boldsymbol{x}) \log p(\boldsymbol{x}) / q(\boldsymbol{x})$, and $L_{2}$ distance, $D_{2}(p \| q)=\sum_{\boldsymbol{x} \in T^{*}}(p(\boldsymbol{x})-q(\boldsymbol{x}))^{2}$.

Under this assumption, we derive the following result:

Theorem 4.5. Under Assumption 3.1 and Assumption 4.3 each training rule $\psi \in \Psi^{S W}$ satisfies payment equivalence.

With the property of payment equivalence, we further investigate the revenue-maximizing payment rule that implements SW-Maximizing training rules in both DSIC and IR. Finding the revenue-maximizing multi-parameter mechanism is a challenging problem in classic mechanism design theory. However, since we have proved the payment equivalence for SW-Maximizing training rules, we can utilize the necessary condition defined in Definition 3.6 to formulate it as a optimization problem. Solving this problem provides the optimal payment rule under the same conditions.

Corollary 4.6. Under Assumption 3.1 and Assumption 4.3 for each training rule $\psi \in \Psi^{S W}$, the revenue-maximizing payment rule $p^{*}$ that implements $\psi$ in both DSIC and IR is given by

$$
\begin{aligned}
& p_{i}^{*}\left(\left(r m_{i}, \overrightarrow{r m}_{-i}\right),\left(w_{i}, \vec{w}_{-i}\right), \theta_{\text {init }}\right)=p_{i}^{A F F}\left(\left(r m_{i}^{\prime}, \overrightarrow{r m}_{-i}\right),\left(w_{i}^{\prime}, \vec{w}_{-i}\right) ; \theta_{\text {init }}\right) \\
& +\inf _{r m_{i}^{\prime} \in \mathcal{R}, w_{i}^{\prime} \in \mathcal{W}} u_{i}\left(\left(r m_{i}^{\prime}, \overrightarrow{r m}_{-i}\right),\left(w_{i}^{\prime}, \vec{w}_{-i}\right) ; \psi, p^{A F F}, r m_{i}^{\prime}, w_{i}^{\prime}\right)
\end{aligned}
$$

The relationship between the domains $\mathcal{R} \times \mathcal{W}$, and this corollary is reflected in two aspects. First, the establishment of payment equivalence depends on the assumptions of the choice of $\mathcal{R}, \mathcal{W}$, particularly considering $\mathcal{R}$ includes all normalized reward models. Second, based on payment equivalence, finding the revenue-maximizing mechanism satisfying IR also needs information on the exact domains.

### 4.2 Approximate Valuation

In this part, we discuss the influence of error generated in practice on the incentive property in the RLHF Game. We abstract it as an approximate valuation problem (Chiesa et al. [2012]). Formally, when group $i$ reports its reward model $\mathrm{rm}_{i}$, the mechanism may not use $\mathrm{rm}_{i}$ exactly but rather a noisy reward model $\widehat{\mathrm{rm}}_{i}$ with a conditional distribution $F_{i}\left(\cdot \mid \mathrm{rm}_{i}\right)$ as the input into the mechanism. We argue that this abstraction has covered various error cases. One example is that the calculation of valuation defined in Assumption 3.1 requires sampling sequences from LLM, which may result in a deviation from the true valuation. Another example is that the fine-tuned model $\mathrm{LLM}_{\theta}$ may not be exactly optimal for the reported reward models. However, this model $\mathrm{LLM}_{\theta}$ can be considered as the optimal for the deviated reward models.

We assume that agent groups are aware of the noise when feeding preferences into the mechanism. Therefore, their utilities will take it into account and have a different form. We use the capital letter $U_{i}$ to represent agent $i$ 's revised utility. Formally, for group $i$ with reward model $\mathrm{rm}_{i}$ and group size $w_{i}$, its utility for reporting $\left(\mathrm{rm}_{i}^{\prime}, w_{i}^{\prime}\right)$ is given by

![](https://cdn.mathpix.com/cropped/2024_06_04_1fc48031c34ce8b3fe25g-07.jpg?height=61&width=1395&top_left_y=2013&top_left_x=365)

Note that in defining $U_{i}$, we implicitly assume that each group is unable to know the other group's noise information. Therefore, the expectation is not taken concerning $\overrightarrow{\mathrm{rm}}_{-i}$.

We only consider the case when the noised input to the mechanism and the reported reward models are close:

Assumption 4.7 (Bounded Error). For any profile of reported reward models $\overrightarrow{\mathrm{rm}}$, any profile of reward models $\overrightarrow{\mathrm{rm}}$ that can be generated from $F_{i}\left(\cdot \mid \mathrm{rm}_{i}\right) \mathrm{s}$ with non-zero probability satisfies

$$
\max _{\boldsymbol{x} \in T^{*}}\left|\widehat{\operatorname{rm}}_{i}(\boldsymbol{x})-\operatorname{rm}_{i}(\boldsymbol{x})\right| \leq \epsilon \quad \forall i \in[n]
$$

We first show that by directly applying results in Section 4.1 to the noised input, the loss in the social welfare is upper-bounded by $2 \epsilon \sum_{i=1}^{n} w_{i}$.

Lemma 4.8. Under Assumption 3.1 and Assumption 4.7 when the training rule $\psi \in \Psi^{S W}$, the loss in social welfare is bounded by

$$
A S W\left(\overrightarrow{r m}, \vec{w}, \psi\left(\overrightarrow{r m}, \vec{w}, \theta_{\text {init }}\right) ; \theta_{\text {init }}\right) \geq A S W\left(\overrightarrow{r m}, \vec{w}, \psi\left(\overrightarrow{r m}, \vec{w}, \theta_{\text {init }}\right) ; \theta_{\text {init }}\right)-2 \epsilon \sum_{i=1}^{n} w_{i}
$$

For training rule $\psi \in \Psi^{S W}$, a group's utility in the mechanism $\left(\psi, p^{A F F}\right)$ consists of an affine social welfare term $A S W$. Therefore, we can derive the following theorem based on Lemma4.8.

Theorem 4.9. Under Assumption 3.1 and Assumption 4.7 when the training rule $\psi \in \Psi^{S W}$, for group $i$ and any rm ${ }_{i}$, $r m_{i}^{\prime}, \overrightarrow{r m}_{-i}, w_{i}$ and $\vec{w}_{i}$, we have

$$
U_{i}\left(\left(r m_{i}, \overrightarrow{r m}_{-i}\right),\left(w_{i}, \vec{w}_{-i}\right) ; \psi, p^{A F F}, r m_{i}, w_{i}\right) \geq U_{i}\left(\left(r m_{i}^{\prime}, \overrightarrow{r m}_{-i}\right),\left(w_{i}, \vec{w}_{-i}\right) ; \psi, p^{A F F}, r m_{i}, w_{i}\right)-2 w_{i} \epsilon
$$

In other words, when $\vec{w}$ is truthfully reported, $\left(\psi, p^{A F F}\right)$ is $\max _{i \in[n]} 2 w_{i} \epsilon$-DSIC mechanism.

This means that for any group $i$, the maximum gain of misreporting is less than $2 w_{i} \epsilon$ regardless of the others' reports. Agents will tend to truthfully report in cases where finding the optimal strategy is costlier than $2 w_{i} \epsilon$.

## 5 Further Related Work

RLHF with Multiple Reward Models. Research involving multiple reward models primarily focuses on developing algorithms to enhance practical performance. Some studies design methods to simultaneously satisfy multiple preferences (Ramé et al. [2024], Wu et al. [2024], Jang et al. [2023], Park et al. [2024]). Additionally, there is a body of work that trains multiple models for a single preference and then ensembles them to improve the robustness of RLHF (Coste et al. [2023], Zhang et al. [2024]), mitigate the influence of incorrect and ambiguous preferences in the dataset (Wang et al. [2024]), and reduce reward hacking (Eisenstein et al. [2023]). Unlike these approaches, our work considers how to collect misaligned preferences truthfully from different agents.

Multi-parameter Auctions. Several studies have explored the properties relevant to our paper in various multiparameter auction scenarios, such as implementability (Rochet [1987], Miyake [1998], Conitzer and Sandholm [2004, Saks and Yu [2005], Bikhchandani et al. [2006], Ashlagi et al. [2010]) and payment equivalence (Ivanova-Stenzel and Salmon [2008], Heydenreich et al. [2009], Bergemann and Välimäki [2010], Pavan et al. [2014]). Another central topic in auction theory is to design mechanisms that satisfy DSIC and IR while maximizing the expected revenue for the auctioneer. Although the single-parameter scenario has been resolved by Myerson [1981], the optimal auction design for multi-parameter settings remains an open question. Therefore, there is a stream of research focusing on a specific subset, affine maximizer auctions, which inherently satisfy DSIC and IR (Sandholm and Likhodedov [2015], Roberts [1979], Likhodedov and Sandholm [2004], Briest et al. [2010], Tang and Sandholm [2012], Jehiel et al. [2007]), and proposes optimizations to enhance empirical performance (Curry et al. [2022], Duan et al. [2024a.b]). Compared to these works, we are the first to discuss the property of payment equivalence and the revenue-maximizing solution in the scenario of fine-tuning LLMs.

Game Theory and LLMs. Other works also explored the intersection of game theory and large language models. Some research has proposed algorithms for training LLMs inspired by concepts in game theory, such as Nash learning from human feedback (Munos et al. [2023]), consensus game (Jacob et al. [2023]), and direct Nash optimization (Rosset et al. [2024]), and Gemp et al. [2024]. Furthermore, various studies assess LLMs from a gametheoretical perspective, examining aspects such as rationality (Chen et al. [2023], Fan et al. [2023]), behavior in matrix games (Akata et al. [2023], Gandhi et al. [2023], Lorè and Heydari [2023]), and performance in strategic games like auctions (Guo et al. [2023, 2024]), Werewolf (Xu et al. [2023a] b]), and Avalon (Wang et al. [2023a]).

## 6 Discussion and Conclusion

Efficient Practical Implementation of $p^{A F F}$. In the RLHF Game with $n$ groups, calculating $p^{A F F}$ requires $n$ separate complete training processes of different $\psi$ s. This can result in inefficiency due to the costly training. To address this problem, we propose two modifications to $p^{A F F}$. Both modifications involve computing an approximate $\widehat{\psi}\left(\overrightarrow{\mathrm{mm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right)$, instead of the true optimal $\psi\left(\overrightarrow{\mathrm{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right)$ when calculating payments:

1. Calculate an approximate $\widehat{\psi}\left(\overrightarrow{\mathrm{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right)=\arg \max _{\theta \in\left\{\theta_{1}, \cdots, \theta_{K}\right\}} A S W_{-i}\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta ; \theta_{\text {init }}\right)$, where $\left\{\theta_{1}, \cdots, \theta_{K}\right\}$ are the parameters saved in the process of training $\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)$.
2. Adopt less iterations in the training process for calculating $\psi\left(\overrightarrow{\mathrm{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right)$. And thus get a result $\widehat{\psi}\left(\overrightarrow{\mathrm{rm}}_{-i}, \vec{w}_{-i}, \theta_{\text {init }}\right)$ that is not optimal.

The first method needs only one training process (for $\left.\psi\left(\overrightarrow{\mathrm{rm}}, \vec{w}, \theta_{\text {init }}\right)\right)$ but affects the property of DSIC since the saved parameters $\left\{\theta_{1}, \cdots, \theta_{K}\right\}$ are also influenced $i$ 's report. In comparison, the second approach incurs higher training costs but guarantees strict DSIC.

Conclusion and Future Work. This paper investigates incentive design in fine-tuning large language models using multiple reward models. We formalize this scenario as the RLHF Game, where a service provider proposes training and payment rules, and agents strategically report their preferences. We demonstrate the necessity of payment schemes for incentivizing truthful reporting in general training rules and provide a comprehensive characterization of payment schemes that implement SW-Maximizing training rules in dominant strategies. These findings enhance the theoretical understanding of mechanism design in LLM fine-tuning and offer guidelines for implementing effective RLHF-based systems in various contexts.

Future research in this field presents several promising directions. Firstly, investigating mechanisms integrating efficiency and incentive compatibility within the RLHF Game could significantly enhance its applicability in real-world scenarios. Secondly, modeling and examining more complex training rules, such as dynamic training rules, could deepen the understanding of this framework. Thirdly, designing mechanisms for more general cases that aggregate preferences into multiple models based on diversity considerations is crucial. Additionally, applying mechanism design theory to other scenarios related to large language models, such as API charge schemes, retrieval-augmented generation (RAG), and prompt engineering, offers valuable opportunities for further exploration.

## References

Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. arXiv preprint arXiv:2305.16867, 2023.

Itai Ashlagi, Mark Braverman, Avinatan Hassidim, and Dov Monderer. Monotonicity and implementability. Econometrica, 78(5):1749-1772, 2010.

Dirk Bergemann and Juuso Välimäki. The dynamic pivot mechanism. Econometrica, 78(2):771-789, 2010.

Sushil Bikhchandani, Shurojit Chatterji, Ron Lavi, Ahuva Mu'alem, Noam Nisan, and Arunava Sen. Weak monotonicity characterizes deterministic dominant-strategy implementation. Econometrica, 74(4):1109-1132, 2006.

Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324-345, 1952.

Patrick Briest, Shuchi Chawla, Robert Kleinberg, and S Matthew Weinberg. Pricing randomized allocations. In Proceedings of the twenty-first annual ACM-SIAM symposium on Discrete Algorithms, pages 585-597. SIAM, 2010.

Yiting Chen, Tracy Xiao Liu, You Shan, and Songfa Zhong. The emergence of economic rationality of gpt. Proceedings of the National Academy of Sciences, 120(51):e2316205120, 2023.

Alessandro Chiesa, Silvio Micali, and Zeyuan Allen Zhu. Mechanism design with approximate valuations. In Proceedings of the 3rd Innovations in Theoretical Computer Science conference, pages 34-38, 2012.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Edward H Clarke. Multipart pricing of public goods. Public choice, pages 17-33, 1971.

Vincent Conitzer and Tuomas Sandholm. Self-interested automated mechanism design and implications for optimal combinatorial auctions. In Proceedings of the 5th ACM Conference on Electronic Commerce, pages 132-141, 2004.

Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H Holliday, Bob M Jacobs, Nathan Lambert, Milan Mossé, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, et al. Social choice for ai alignment: Dealing with diverse human feedback. arXiv preprint arXiv:2404.10271, 2024.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Michael Curry, Tuomas Sandholm, and John Dickerson. Differentiable economics for randomized affine maximizer auctions. arXiv preprint arXiv:2202.02872, 2022.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Zhijian Duan, Haoran Sun, Yurong Chen, and Xiaotie Deng. A scalable neural network for dsic affine maximizer auction design. Advances in Neural Information Processing Systems, 36, 2024a.

Zhijian Duan, Haoran Sun, Yichong Xia, Siqiang Wang, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, and Xiaotie Deng. Scalable virtual valuations combinatorial auction design by combining zeroth-order and first-order optimization method. arXiv preprint arXiv:2402.11904, 2024b.

Kumar Avinava Dubey, Zhe Feng, Rahul Kidambi, Aranyak Mehta, and Di Wang. Auctions with llm summaries. arXiv preprint arXiv:2404.08126, 2024.

Paul Duetting, Vahab Mirrokni, Renato Paes Leme, Haifeng Xu, and Song Zuo. Mechanism design for large language models. arXiv preprint arXiv:2310.10826, 2023.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D'Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Caoyun Fan, Jindou Chen, Yaohui Jin, and Hao He. Can large language models serve as rational players in game theory? a systematic analysis. arXiv preprint arXiv:2312.05488, 2023.

Soheil Feizi, MohammadTaghi Hajiaghayi, Keivan Rezaei, and Suho Shin. Online advertisements with llms: Opportunities and challenges. arXiv preprint arXiv:2311.07601, 2023.

Kanishk Gandhi, Dorsa Sadigh, and Noah D Goodman. Strategic reasoning with language models. arXiv preprint arXiv:2305.19165, 2023.

Ian Gemp, Yoram Bachrach, Marc Lanctot, Roma Patel, Vibhavari Dasagi, Luke Marris, Georgios Piliouras, and Karl Tuyls. States as strings as strategies: Steering language models with game-theoretic solvers. arXiv preprint arXiv:2402.01704, 2024.

Theodore Groves. Incentives in teams. Econometrica: Journal of the Econometric Society, pages 617-631, 1973.

Shangmin Guo, Haochuan Wang, Haoran Bu, Yi Ren, Dianbo Sui, Yu-Ming Shang, and Siting Lu. Large language models as rational players in competitive economics games. arXiv preprint arXiv:2308.10032, 2023.

Shangmin Guo, Haoran Bu, Haochuan Wang, Yi Ren, Dianbo Sui, Yuming Shang, and Siting Lu. Economics arena for large language models. arXiv preprint arXiv:2401.01735, 2024.

Birgit Heydenreich, Rudolf Müller, Marc Uetz, and Rakesh V Vohra. Characterization of revenue equivalence. Econometrica, 77(1):307-316, 2009.

Radosveta Ivanova-Stenzel and Timothy C Salmon. Revenue equivalence revisited. Games and Economic Behavior, 64(1):171-192, 2008 .

Athul Paul Jacob, Yikang Shen, Gabriele Farina, and Jacob Andreas. The consensus game: Language model generation via equilibrium search. arXiv preprint arXiv:2310.09139, 2023.

Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. arXiv preprint arXiv:2310.11564, 2023.

Philippe Jehiel, Moritz Meyer-Ter-Vehn, and Benny Moldovanu. Mixed bundling auctions. Journal of Economic Theory, 134(1):494-512, 2007.

Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, et al. Ai alignment: A comprehensive survey. arXiv preprint arXiv:2310.19852, 2023.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. Advances in Neural Information Processing Systems, 36, 2024.

Benjamin Laufer, Jon Kleinberg, and Hoda Heidari. Fine-tuning games: Bargaining and adaptation for generalpurpose models. arXiv preprint arXiv:2308.04399, 2023.

Anton Likhodedov and Tuomas Sandholm. Methods for boosting revenue in combinatorial auctions. In AAAI, pages 232-237, 2004.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Nunzio Lorè and Babak Heydari. Strategic behavior of large language models: Game structure vs. contextual framing. arXiv preprint arXiv:2309.05898, 2023.

David G Luenberger, Yinyu Ye, et al. Linear and nonlinear programming, volume 2. Springer, 1984.

Mitsunobu Miyake. On the incentive properties of multi-item auctions. International Journal of Game Theory, 27: $1-19,1998$.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.

Roger B Myerson. Optimal auction design. Mathematics of operations research, 6(1):58-73, 1981.

Noam Nisan et al. Introduction to mechanism design (for computer scientists). Algorithmic game theory, 9:209-242, 2007.

Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 1999.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.

Chanwoo Park, Mingyang Liu, Kaiqing Zhang, and Asuman Ozdaglar. Principled rlhf from heterogeneous feedback via personalization and preference aggregation. arXiv preprint arXiv:2405.00254, 2024.

Alessandro Pavan, Ilya Segal, and Juuso Toikka. Dynamic mechanism design: A myersonian approach. Econometrica, 82(2):601-653, 2014.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. Warm: On the benefits of weight averaged reward models. arXiv preprint arXiv:2401.12187, 2024.

Kevin Roberts. The characterization of implementable choice rules. Aggregation and revelation of preferences, 12(2): 321-348, 1979 .

Jean-Charles Rochet. A necessary and sufficient condition for rationalizability in a quasi-linear context. Journal of mathematical Economics, 16(2):191-200, 1987.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

Michael Saks and Lan Yu. Weak monotonicity suffices for truthfulness on convex domains. In Proceedings of the 6th ACM conference on Electronic commerce, pages 286-293, 2005.

Tuomas Sandholm and Anton Likhodedov. Automated design of revenue-maximizing combinatorial auctions. Operations Research, 63(5):1000-1025, 2015.

Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, and Deyi Xiong. Large language model alignment: A survey. arXiv preprint arXiv:2309.15025, 2023.

Ermis Soumalias, Michael J Curry, and Sven Seuken. Truthful aggregation of llms with an application to online advertising. arXiv preprint arXiv:2405.05905, 2024.

Pingzhong Tang and Tuomas Sandholm. Mixed-bundling auctions with reserve prices. In AAMAS, pages 729-736, 2012.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

William Vickrey. Counterspeculation, auctions, and competitive sealed tenders. The Journal of finance, 16(1):8-37, 1961.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080, 2024.

Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, and Gao Huang. Avalon's game of thoughts: Battle against deception through recursive contemplation. arXiv preprint arXiv:2310.01320, 2023a.

Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu. Aligning large language models with human: A survey. arXiv preprint arXiv:2307.12966, 2023b.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. Advances in Neural Information Processing Systems, 36, 2024.

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. arXiv preprint arXiv:2309.04658, 2023a.

Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game. arXiv preprint arXiv:2310.18940, 2023b.

Shun Zhang, Zhenfang Chen, Sunli Chen, Yikang Shen, Zhiqing Sun, and Chuang Gan. Improving reinforcement learning from human feedback with efficient reward model ensemble. arXiv preprint arXiv:2401.16635, 2024.
</end of paper 2>


<paper 3>
# Language Model Alignment with Elastic Reset 

Michael Noukhovitch*<br>Mila, Université de Montréal

Florian Strub

Google Deepmind

Samuel Lavoie<br>Mila, Université de Montréal

Aaron Courville<br>Mila, Université de Montréal<br>Canada CIFAR AI Chair


#### Abstract

Finetuning language models with reinforcement learning (RL), e.g. from human feedback (HF), is a prominent method for alignment. But optimizing against a reward model can improve on reward while degrading performance in other areas, a phenomenon known as reward hacking, alignment tax, or language drift. First, we argue that commonly-used test metrics are insufficient and instead measure how different algorithms tradeoff between reward and drift. The standard method modified the reward with a Kullback-Lieber (KL) penalty between the online and initial model. We propose Elastic Reset, a new algorithm that achieves higher reward with less drift without explicitly modifying the training objective. We periodically reset the online model to an exponentially moving average (EMA) of itself, then reset the EMA model to the initial model. Through the use of an EMA, our model recovers quickly after resets and achieves higher reward with less drift in the same number of steps. We demonstrate that fine-tuning language models with Elastic Reset leads to state-of-the-art performance on a small scale pivot-translation benchmark, outperforms all baselines in a medium-scale RLHF-like IMDB mock sentiment task and leads to a more performant and more aligned technical QA chatbot with LLaMA-7B. Code available at github.com/mnoukhov/elastic-reset.


## 1 Introduction

Dialogue agents that can effectively interpret and use language are a long-term challenge for NLP. The rise of large pretrained language models (LMs) [Brown et al., 2020] made language model finetuning one of the most promising research directions to achieving capable dialogue agents [Bender and Koller, 2020]. Recently, reinforcement learning (RL) has become a key ingredient of finetuning large LMs for interaction with humans [Ziegler et al., 2019, Ouyang et al., 2022, Bai et al., 2022], notably shown in ChatGPT [OpenAI, 2022]. A reward model is learned on the alignment objective, such as learned human preferences [RLHF; Christiano et al., 2017, Stiennon et al., 2020], and the language model is finetuned to optimize the reward. But training on the RL objective moves the model away from its pretraining and can reduce performance on important benchmarks [Ouyang et al., 2022] and even drifting away from natural language syntax and semantics [Lazaridou et al., 2020].

"Language drift" [Lee et al., 2019, Lazaridou et al., 2020], "alignment tax" [Askell et al., 2021], "reward model overoptimization" [Gao et al., 2022], or LM-specific "reward-hacking" [Clark and Amodei, 2016] is inherent to RLHF. In the extreme case, models learn to achieve high reward by generating nonsense text that is unintelligible to humans [Lewis et al., 2017]. Methods to mitigate this issue range from re-running pretraining [Lowe* et al., 2021], grounding in other modalities [Lee et al.,[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-02.jpg?height=301&width=1247&top_left_y=240&top_left_x=431)

Figure 1: Elastic Reset. In actor-critic RL, we reset the policy but maintain the value function.

2019], masking the LM generation [Ramamurthy et al., 2022] and iterated learning [Lu et al., 2020]. But the standard, and by far most popular approach, adds a Kullback-Lieber (KL) divergence penalty to the reward in order to prevent the finetuned model from drifting too far from the pretrained model [Jaques et al., 2017, 2019, Ziegler et al., 2019]. Still, all methods are insufficient over a large-enough training horizon so models are early-stopped before reaching a catastrophic level of drift.

Gao et al. [2022] find that achieving reward is proportional to drift from the initial model, but that not all drifts are equal. We wish to make small but effective changes that achieve high reward but maintain capabilities, yet auxiliary losses such as the KL penalty don't seem improve this tradeoff and only serve to slow down training [Gao et al., 2022]. We posit that RLHF training requires a useful inductive bias that does not modify the training objective. Inspired by recent work in generalization for image classification [Zhou et al., 2022], sample-efficient RL [Nikishin et al., 2022, D'Oro et al., 2023], and inducing compositional language [Li and Bowling, 2019], we propose to use resets. Similar to iterated learning [Kirby, 2001], iteratively resetting a model has been shown to reduce overfitting in language and RL scenarios [Rita et al., 2022]. In this work, we show iteratively resetting a model also reduces drift while attaining equal or better reward than just a KL penalty.

Unlike previous work in sample-efficient RL [Nikishin et al., 2022], RLHF is typically on-policy so it does not maintain a replay buffer with which to bootstrap learning after a reset. In lieu of a replay buffer, we reset the policy but maintain the value function. Yet resetting the policy to its initial state can still cause a large drop in performance. So we propose resetting to a model in-between our online and initial state, specifically to an exponential moving average (EMA) of our online policy, as EMA has been shown to be highly performant [Caron et al., 2021]. We still expect our EMA model to slowly drift, so we add a second step where we reset the EMA model to the initial model. We call this overall method Elastic Reset and illustrate it in Figure 1. Elastic Reset is implemented on top of regular RL methods such as REINFORCE [Williams, 1992] or PPO [Schulman et al., 2017].

First, we test our method on a small scale task: pivot translation with a transformer. In this classic benchmark for drift, we outperform all previous baselines and demonstrate state-of-the-art performance. Next, we re-evaluate how performance is measured in the field and argue for a metric of how each method trades off performance vs drift. We propose the Pareto Frontier Graph, a graphical measure that illuminates the trade-off between performance and drift and demonstrate that Elastic Reset dominates the baselines against this trade-off. Then, we scale up slightly to GPT2 and work on a popular task closer to RLHF, IMDB mock sentiment. Comparing to all baseline methods, we again show state-of-the-art performance on the benchmark. Through ablations, we show that Elastic Reset is robust to choices of hyperparameters, even more so than baselines. Finally, we scale up even more to true RLHF finetuning of Llama-7B in order to create a helpful technical QA chatbot using a StackExchange dataset. We again outperform the baseline, demonstrating how Elastic Reset mitigates the alignment tax while better optimizing the human feedback reward.

## 2 Related Work

"It is often difficult or infeasible to capture exactly what we want an agent to do, and as a result we frequently end up using imperfect but easily measured proxies" [Clark and Amodei, 2016]. In RL, this proxy is how we construct our reward and the consequence can be "reward-hacking" [Clark and Amodei, 2016]; an agent optimizes the reward but does not accomplishing the meaningful task. RLHF aims to align an agent with human preferences while maintaining the capabilities of the pretrained model, but uses a learned reward model as a proxy of human preferences [Christiano et al., 2017, Ziegler et al., 2019]. This can lead to LMs that optimize a reward model but degrade in
performance on general NLP benchmarks [Askell et al., 2021], overfit the reward model and do not generalize to true human preferences [Bai et al., 2022, Gao et al., 2022], or latch onto confounding factors hidden in the reward [Stiennon et al., 2020]. These effects are exacerbated if the reward model is updated during training such as in iterated RLHF [Bai et al., 2022] or related setups such as emergent communication [Lazaridou and Baroni, 2020], end-to-end dialogue [Lewis et al., 2017], learning RL policies through latent language [Andreas et al., 2018], and pivot translation [Utiyama and Isahara, 2007]. There, the phenomenon is known as "language drift" [Lee et al., 2019] and can lead to incoherent and unnatural linguistic outputs Lewis et al. [2017].

This phenomenon is inherent to RLHF. Gao et al. [2022] show that improvement on alignment / reward is proportional to drift from the initial model, but also find that different methods and design choices achieve different proportions of performance to drift. Therefore, a major challenge of RLHF is how to learn the reward in such a way as to minimize the drift, alignment tax, and reward-hacking. The standard approach used in most RLHF is to incorporate a KL penalty between the training language model and some fixed model [Jaques et al., 2019, Ziegler et al., 2019, Stiennon et al., 2020, Ouyang et al., 2022, Steinert-Threlkeld et al., 2022, Bai et al., 2022], usually the initial, pretrained model. Less common is to add the original pretraining task to the finetuning objective (termed S2P by Lowe* et al. [2021]) but this can be compute-intensive and requires maintaining the pretraining data which may be even more expensive for larger models [Brown et al., 2020]. On small-scale pivot-translation, Lu et al. [2020] propose iterated learning with student-teacher distillation but it too is relatively compute intensive. Recently, Ramamurthy et al. [2022] propose to maintain a delayed masking model and mask the LM to output only the top- $p$ tokens. Elastic Reset takes inspiration from both of these, using an iterated process and maintaining an EMA model. Apart from better performance, our method is more space efficient and maintains the EMA on CPU whereas both other methods require maintaining an extra model on GPU. It is also more compute efficient as resetting weights and EMA updates are very cheap operations, whereas Lu et al. [2020] requires a long distillation phase and Ramamurthy et al. [2022] requires an extra forward pass with the masking model. There exist less-popular methods that have been applied to similar issues in RL: prompt-tuning [Singh et al., 2022], using a fixed model to generate many options and re-ranking using a reward model [Lazaridou et al., 2020, Meta FAIR Diplomacy Team et al., 2022], and grounding the output in a separate modality [Lee et al., 2019] but none have been used for RLHF or at scale.

Our method is inspired by recent works that leverage resets for single agent RL [Nikishin et al., 2022, D'Oro et al., 2023], image classification [Zhou et al., 2022], and emergent communication [Rita et al., 2022]. Those works generally train from scratch and reset to random initializations in order to improve generalization. Our scenario requires resetting to pretrained models and focuses on improving the tradeoff between performance and drift from this pretrained model. Elastic Reset can be seen as an on-policy alternative to Nikishin et al.'s [2022] off-policy resets; whereas they maintain the old replay buffer, we maintain the value model and an EMA of our policy.

Finally, the pretrain-then-RL-finetune setup with the goal of maintaining pretrained knowledge can be seen as a two-step, RL-specific instance of continual learning and therefore language drift has links to catastrophic forgetting [McCloskey and Cohen, 1989]. There is a clear similarity between mitigation methods: rehearsal [Robins, 1995] or experience replay [Rolnick et al., 2019] is equivalent to multitasking with the pretraining objective [Lowe* et al., 2021] and weight-update regularization [Kirkpatrick et al., 2017] has similarities to KL regularization [Jaques et al., 2019].

## 3 Elastic Reset

The standard method against drift is a KL penalty, generally between the learning policy $\theta$ and the initial, pretrained model $\theta_{0}$. It is calculated empirically over the minibatch of training inputs $x$ and outputs $y$ and used as an auxiliary reward with coefficient $\beta$ on top of the regular reward model $r$

$$
\begin{equation*}
R(x, y)=r(x, y)-\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\theta_{0}}(y \mid x)} \tag{1}
\end{equation*}
$$

For Elastic Reset, we maintain an exponential moving average $\bar{\theta}$ of our learning model $\theta$ and choose a decay hyperparameter parameter $\eta$. We initialize $\theta \leftarrow \theta_{0}$ and after every online model step, we update our EMA model $\bar{\theta} \leftarrow(1-\eta) \theta+\eta \bar{\theta}$. Every $n$ steps, Elastic Reset sets the online model to the
![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-04.jpg?height=270&width=1388&top_left_y=248&top_left_x=368)

Figure 2: The Translation Game (top left), IMDB mock sentiment task (bottom left), and StackLLaMA (right). We show all RL finetuning setups and StackLLaMA's reward modelling (top right).

EMA model $\theta \leftarrow \bar{\theta}$ and sets the EMA model to the initial model $\bar{\theta} \leftarrow \theta_{0}$. As with other methods, Elastic Reset can be easily combined with a KL penalty.

## 4 Translation Game: Careful Comparison to SOTA

Setup We first investigate the pivot-translation benchmark of Lee et al. [2019], which was previously popular for small-scale methods countering drift. Two translation models, French to English $(\mathrm{FR} \rightarrow \mathrm{EN})$ and English to German (EN $\rightarrow \mathrm{DE}$ ), are pretrained on IWSLT [Cettolo et al., 2012]. Then, the models are finetuned on translating French to German through English (FR $\rightarrow \mathrm{EN} \rightarrow \mathrm{DE}$ ) but given only paired French and German data from Multi30k [Elliott et al., 2016, 2017] as shown in Figure 2. The models are not given English at finetune-time so the challenge is optimizing FR $\rightarrow \mathrm{DE}$ while maintaining fluency in the intermediate English. Whereas larger benchmarks have only proxies for drift, we can exactly measure the performance degradation in our setup with the standard translation metric BLEU on a held-out FR $\rightarrow$ EN validation set. Similarly, we measure success on the task with the FR $\rightarrow \mathrm{EN} \rightarrow \mathrm{DE}$ BLEU score. Each model is an encoder-decoder Transformer [Vaswani et al., 2017] with 6 layers and all experimental details are available in Appendix A.

Baselines The $\mathrm{EN} \rightarrow \mathrm{DE}$ reward model is simply trained using cross-entropy between predicted and true DE. Our lower-bound baseline is FROZEN ENGLISH, we freeze the FR $\rightarrow$ EN model and only update the $\mathrm{EN} \rightarrow \mathrm{DE}$ model. This models is guaranteed not to drift, but also cannot reach the best possible performance. For that, we need to update FR $\rightarrow$ EN by backpropogating through the discrete EN tokens. We follow Lee et al. [2019] and train FR $\rightarrow$ EN using REINFORCE [Williams, 1992] to estimate the gradient. As our base model, we combine REINFORCE with an exponentially moving baseline and, as with previous work, add a loss for entropy regularization.

When both $\mathrm{FR} \rightarrow \mathrm{EN}$ and $\mathrm{EN} \rightarrow \mathrm{DE}$ are being updated, we tend to see reasonably large drift and we compare to the best previous methods that counter it on this benchmark. We follow Lee et al. [2019] to simulate the standard KL penalty method, KL PENALTY by training an LSTM LM on IWSLT English text and adding a $\mathrm{KL}$ penalty with $\beta=0.05$ to regularize the $\mathrm{FR} \rightarrow \mathrm{EN}$ model. MULTITASK learning, re-training, or S2P [Lowe* et al., 2021], adds the supervised FR $\rightarrow$ EN objective on IWSLT pretraining data as an auxiliary task for the FR $\rightarrow$ EN model. Finally, we implement Seeded Iterated Learning [SIL; Lu et al., 2020], which alternates between $n$ finetuning steps and $m$ steps of teacher-student distillation. $\mathrm{FR} \rightarrow \mathrm{EN}$ and $\mathrm{EN} \rightarrow \mathrm{DE}$ "teacher" models are finetuned on the translation game, then each distills knowledge into "student" model of itself, and finally the students are initialized as teachers for the next iteration. ELASTIC RESET is implemented on top of REINFORCE with a very minimal KL penalty $\beta=0.001$ and uses an EMA decay $\eta=0.99$. We run all models for $50 \mathrm{k}$ updates and reset every $23 \mathrm{k}$ steps to get 2 resets / 3 iterations within a run. Hyperparameters may differ between methods, e.g. $\beta$, because we used a minimal search to find the best hyperparameters for each method.

Experiments For each method, we run 5 seeds and plot the validation scores over training for the end-to-end task score, FR $\rightarrow \mathrm{EN} \rightarrow \mathrm{DE}$ BLEU, and drift score, FR $\rightarrow$ EN BLEU, in Figures 3b, 3a respectively. Following Lee et al. [2019], we also show the final validation score in Table 1. As a sanity check, FrozEN ENGLISH does not drift but also does not achieve a very high task performance. In line with previous results [Lu et al., 2020], all learning models initially improve $\mathrm{FR} \rightarrow \mathrm{EN}$ performance, likely because models are quickly, semi-supervised adapting from their pretraining (IWSLT) to the distribution of the finetune dataset (Multi30k). Afterwards, they start to overfit on their objective and FR $\rightarrow$ EN performance degrades. REINFORCE achieves the best

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-05.jpg?height=406&width=1369&top_left_y=236&top_left_x=367)

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-05.jpg?height=320&width=436&top_left_y=255&top_left_x=389)

(a) Task Performance

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-05.jpg?height=328&width=439&top_left_y=248&top_left_x=840)

(b) Language Drift

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-05.jpg?height=336&width=441&top_left_y=244&top_left_x=1292)

(c) Pareto Frontier Graph

Figure 3: Comparing Elastic Reset to all baseline methods on the Translation Game. We measure (a) Task Performance with FR $\rightarrow$ EN $\rightarrow$ DE BLEU and (b) Language Drift with FR $\rightarrow$ EN BLEU, on the validation set during finetuning. We plot the mean and standard error over 5 seeds. To compare how methods trade off the two metrics, we plot (c) the best achieved drift vs task performance.

Table 1: Translation Game final validation scores

|  | $\uparrow \mathrm{FR} \rightarrow \mathrm{EN} \rightarrow \mathrm{DE}$ | $\uparrow \mathrm{FR} \rightarrow \mathrm{EN}$ |
| :--- | :---: | :---: |
| FROZEN ENGLISH | $30.8 \pm 0.2$ | $\mathbf{3 6 . 3} \pm \mathbf{0 . 1}$ |
| REINFORCE | $\mathbf{3 3 . 2} \pm \mathbf{0 . 3}$ | $29.6 \pm 0.3$ |
| + SIL | $28.2 \pm 0.4$ | $27.3 \pm 4.4$ |
| + MULTITASK (S2P) | $32.2 \pm 0.3$ | $35.2 \pm 1.0$ |
| + KL PENALTY | $\mathbf{3 3 . 2} \pm \mathbf{0 . 2}$ | $30.8 \pm 0.4$ |
| + ELASTIC RESET | $32.9 \pm 0.1$ | $\mathbf{3 6 . 3} \pm \mathbf{0 . 1}$ |

possible task performance but drifts significantly. Despite extensive hyperparameter tuning and correspondence with the original authors, SIL does not manage to outperform the REINFORCE baseline so we exclude it from the figures for visual clarity but show values in Table 1 as well as full results in Appendix A. In line with previous work [Lee et al., 2019, Lu et al., 2020], we find that MULTITASK and KL PENALTY are both beneficial to reducing drift, but both represent a tradeoff. Whereas MULTITASK strongly reduces drift, it does not achieve a high task score. In contrast, KL PENALTY achieves a high task score but drifts quite drastically. Elastic Reset achieves nearly the best possible task score while maintaining the same drift score as the initial model. Visually, we see that our method track the baselines until the reset at $23 \mathrm{k}$ steps. After the reset, we see a slight performance drop but also a big jump back in terms of FR $\rightarrow$ EN drift. While the task performance recovers within $5 \mathrm{k}$ steps, the drift performance does not degrade to previous levels. For the second reset, the EMA model is slightly more drifted and so the reset is less pronounced for both task and drift, leading to faster task recovery but slightly more drift. Overall, Elastic Reset shows state-of-the-art results on the benchmark and outperforms all previous small-scale methods.

## 5 Pareto Frontier Graph

Simply evaluating validation curves side-by-side or looking at a table of final scores, it can be unclear which method is better if one drifts less but the other achieves a higher reward e.g. MULTITASK vs KL PENALTY in Table 1. Previous work on this [Lee et al., 2019] and other benchmarks [Ramamurthy et al., 2022] compare methods using simple point-estimates after training for a specific number of epochs. But this number of epochs is quite arbitrary as models never fully converge to a reward, they are early-stopped such that drift is not catastrophic. Since different setups may admit different levels of drift, we believe that evaluation should reflect the continuous tradeoff between task and drift. We extend Ziegler et al. [2019], and create a pareto frontier graph to plot each method's achieved task score vs drift metric on the validation set over training. We believe practioners will wish to choose the best model for some given task performance so, contrary to Ziegler et al. [2019], we plot the task score on $x$-axis and drift score on the $y$-axis. Improvement on a drift metric can either mean lower scores (perplexity) or higher scores (BLEU) so we always plot task score as increasing from bottom to top such that, graphically, a better method will functionally dominate a worse method. We plot the best achieved reward vs drift over all validation steps for the Translation Game in Figure 3c. Not only

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-06.jpg?height=410&width=1374&top_left_y=234&top_left_x=365)

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-06.jpg?height=325&width=442&top_left_y=244&top_left_x=386)

(a) Task Performance

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-06.jpg?height=336&width=444&top_left_y=241&top_left_x=838)

(b) Language Drift

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-06.jpg?height=325&width=450&top_left_y=241&top_left_x=1285)

(c) Pareto Frontier Graph

Figure 4: Plotting PPO vs Elastic Reset on IMDB but splitting the results visually between resets. We measure (a) Language Drift and (b) Task Performance via Semantic Score on the validation set over finetuning. All methods also include a KL penalty. We plot mean and standard error across 5 seeds.

Table 2: IMDB mock sentiment final test scores

|  | $\uparrow$ SENTIMENT | $\downarrow$ PERPLEXITY |
| :--- | :---: | :---: |
| ZERO-SHOT | $.489 \pm 0.01$ | $32.45 \pm 0.13$ |
| PPO | $.596 \pm 0.02$ | $33.45 \pm 0.40$ |
| NLPO | $.558 \pm 0.06$ | $33.12 \pm 0.74$ |
| ELASTIC RESET | $.611 \pm 0.02$ | $33.32 \pm 0.23$ |

does Elastic Reset outperform the baselines at the final validation score, but it functionally dominates such that it is the best method for all levels of task performance it achieves.

## 6 IMDB Mock Sentiment: Ablation Study for RLHF

Setup Next, we scale to a larger benchmark that more closely approximates the standard RLHF setup. We use the recently released GRUE benchmark for RL training of LMs [Ramamurthy et al., 2022] and use IMDB mock sentiment [Ziegler et al., 2019], the main task where language models are susceptible to reward-hacking, shown in Figure 2. The goal is to complete an IMDB movie review with as positive a sentiment as possible. The baseline LM is GPT-2 [Radford et al., 2019] with 117M parameters further pretrained on the IMDB domain [Maas et al., 2011]. We learn a DistilBERT [Sanh et al., 2020] reward model on IMDB to output a sentiment score between 0 (negative) and 1 (positive). We then train our GPT-2 LM to complete different IMDB reviews while maximizing the sentiment reward. Following Ramamurthy et al. [2022], we measure reward-hacking / drift with our model's perplexity on the true IMDB data. If we consider knowledge of the IMDB data as a useful capability, then our initial model was finetuned on IMDB to maximize log-probability, i.e. minimize perplexity, and has the maximum capabilites. We measure divergence from the initial model, and decrease in capabilities, by the increase in our trained model's perplexity on ground truth IMDB data. In contrast to the previous task, a lower perplexity score corresponds to less drift.

Baselines Our main baseline is PPO [Schulman et al., 2017] with Ziegler et al. [2019] modifications for RLHF training, specifically adding a KL penalty with the frozen initial model (equivalent to KL WITH PRETRAINED) and dynamically decaying the coefficient $\beta$ over training. To further increase stability, Generalized Advantage Estimation [Schulman et al., 2015] is used for the advantage estimator. We also compare to NLPO [Ramamurthy et al., 2022], a recent method that extends PPO with a masking model to counteract drift. The masking model is initialized to the pretrained model and recieves delayed updates; it is set to the online model every $n$ steps. During training, the online model's output probabilities are restricted to the mask model's top $p$ tokens. We use the RL4LMs library Ramamurthy et al. [2022] and their default hyperparameters for both PPO and NLPO e.g. $\beta=0.1$. We implement ElASTIC RESET on top of PPO with an EMA decay rate of 0.995 and greatly reduce the $\mathrm{KL}$ coefficient $\beta=0.001$ to allow the model to drift more, then reset every 17 epochs such that we get two resets / three iterations during our training.

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-07.jpg?height=973&width=1269&top_left_y=224&top_left_x=428)

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-07.jpg?height=414&width=559&top_left_y=243&top_left_x=447)

(a) Reset Ablation

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-07.jpg?height=412&width=555&top_left_y=713&top_left_x=449)

(c) $\mathrm{KL} \beta$ Ablation for PPO

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-07.jpg?height=412&width=566&top_left_y=241&top_left_x=1105)

(b) EMA Decay $\eta$ Ablation

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-07.jpg?height=418&width=569&top_left_y=707&top_left_x=1106)

(d) $\operatorname{KL} \beta$ Ablation for Elastic Reset

Figure 5: Ablating Elastic Reset on the IMDB mock sentiment task. We plot pareto graphs using mean and standard error across 5 seeds.

Experiments We run all experiments for 5 seeds and report mean and standard error on our validation set for our reward, DistilBERT sentiment, and our drift score, perplexity. Following Ramamurthy et al. [2022], we run for 50 epochs (equivalent to $64 \mathrm{k}$ updates) and show our results in Figure 4. To make our resets more visible, we plot validation scores every epoch for Elastic Reset. Since the benchmark provides a test set as well, we compare all final models in Table 2. The PPO baseline performs quite well because it already includes a KL with the pretrained model. We find NLPO performs similarly to PPO, so we relegate NLPO results to Appendix B. Results from the original NLPO paper were stronger [Ramamurthy et al., 2022] but our reproduced numbers and curves were confirmed by the original authors [Ammanabrolu, 2023]. Elastic Reset achieves better semantic scores much faster by using a smaller KL penalty coefficient ( 0.001 vs PPO 0.1) but also drifts more to achieve them. As with the previous task, this drift is then mitigated by the reset and we see semantic task score improve in relation to drift over the iterations. Looking at the pareto graph in Figure 8c, we see that Elastic Reset far outpaces the baselines and provides a better tradeoff of reward vs drift for every reward.


#### Abstract

Ablations We empirically investigate our method through ablations. Throughout this section we run experiments on IMDB with the same hyperparameters, unless otherwise mentioned. For brevity, we plot only the pareto graphs but include all other graphs in Appendix D. 3 along with these same ablation experiments for the Translation Game, with similar results.

To investigate the source of improvement in Elastic Reset, we ablate the two resets: online to EMA, and EMA to initial model. We discard the second reset to get Reset to EMA: our model is reset to an EMA but the EMA is never reset. We also compare to the simplest reset idea, Reset to Init, and reset our policy to the initial model. We run all methods as previously and plot the pareto graph in Figure 5a, for task and drift graphs see Appendix D.3. We find that even simple resets are already performant but the two ablations have a tradeoff: Reset to EMA is better at lower reward because it maintains performance whereas Reset to Init does better at higher reward because it doesn't drift as much. Elastic Reset combines the benefits of both and outperforms each method.


Next, we consider our method's robustness to hyperparameters. First, we search along different EMA decay rates $\eta$ and plot our results in Figure 5b finding that our method is quite robust to choice of

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-08.jpg?height=414&width=1374&top_left_y=232&top_left_x=365)

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-08.jpg?height=331&width=439&top_left_y=244&top_left_x=388)

(a) Reward

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-08.jpg?height=338&width=447&top_left_y=243&top_left_x=839)

(b) Drift

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-08.jpg?height=331&width=458&top_left_y=241&top_left_x=1278)

(c) Pareto Frontier Graph

Figure 6: Elastic Reset compared to PPO on StackLLaMA: A LLaMA-7B model RLHF finetuned on StackExchange as a helpful, technical QA chatbot

decay. Next, we investigate robustness to the choice of KL penalty coefficient $\beta$. We search across coefficients that range from $10 \mathrm{x}$ smaller to $10 \mathrm{x}$ larger than our best KL penalty coefficient for PPO $(\beta=0.1)$ and Elastic Reset $(\beta=0.001)$. For visual clarity, we plot PPO in Figure 5c and Elastic Reset in Figure 5d and only plot four points for PPO $\beta=0,0.01$ to maintain visual scale. We find that PPO is not robust to choice of $\mathrm{KL}$ and larger values correspond to better pareto curves but slower training. Results with NLPO are similar and shown in Appendix D.3. In contrast, Elastic Reset seems to be more robust to choice of KL with 0.001 producing the best curves while $10 \mathrm{x}$ larger and smaller values are similar. As opposed to PPO, Elastic Reset even works reasonably well without a KL penalty at all, $(\beta=0)$, matching PPO's best performance with a KL. This demonstrates that the expensive KL penalty may be replaced with the cheap Elastic Reset, although the combination of the two is best. This is also in line with previous work that have argued that the KL penalty may be unnecessary for RLHF [Bai et al., 2022, Gao et al., 2022]. We also ablate the frequency of resets in Appendix D.4.1 and find that pareto curves are essentially unchanged.

Finally, we provide an empirical intuition for Elastic Reset: in Appendix E. 1 we show that resets iteratively improve the value function and in Appendix E. 2 we show how EMA smoothes optimization but requires resetting in order to achieve high performance.

## 7 StackLLaMA: Practical RLHF

Setup Finally, we apply Elastic Reset to a larger-scale RLHF pipeline. We choose LLaMA [Touvron et al., 2023] as it is a prominent open-source model that has demonstrated strong performance on benchmarks. We follow Beeching et al. [2023] to finetune LLaMA-7B with RLHF on the StackExchange dataset [Lambert et al., 2023] to output helpful answers to technical questions, as judged by humans. Users ask technical questions on StackExchange and upvote the best answers. We score answers from StackExchange based on the number of upvotes they received from users, score $=\log _{2}(1+$ upvotes [Askell et al., 2021] . At most 10 answers are drawn per question, text is cleaned, and HTML is converted to Markdown to make it easier to parse. First, we finetune LLaMA-7B with language modelling on the dataset to get LLaMA-7B-SE. We then further finetune it to get a reward model by learning to predict which of two answers was more upvoted [Stiennon et al., 2020]. For a given question $x$ and two answers $y_{+,-}$(where $y_{+}$is preferred), the loss for our reward model $r_{\theta}$ is $\log \sigma\left(r_{\theta}\left(x, y_{+}\right)-r_{\theta}\left(x, y_{-}\right)\right)$. Finally, we finetune LLaMA-7B-SE with RL against the reward model by sampling questions from the dataset and learning to optimize the reward for our model's answer. All finetuning is done with a converted 8-bit model [Dettmers et al., 2022] and LoRA [Hu et al., 2021] for efficiency and to make the model training fit on our GPUs. We rely on the HuggingFace trl [von Werra et al., 2023] and peft [Mangrulka et al., 2023] libraries. All technical details are described in Appendix C.

Experiment We again compare to PPO with Ziegler et al. [2019] modifications i.e. KL penalty with a dynamically decaying coefficient. We run for 600 epochs (equivalent to 100k updates) and Elastic Reset every 260 epochs to get two resets / three iterations. Each run takes 20 hours on 4 A100s. Since only the LoRA parameters are being learned, we use Elastic Reset on those and therefore maintain only a small percentage of parameters in our EMA. We use a decay rate $\eta=0.995$ and a KL penalty coefficient $\beta=0.02$ for both methods. Calculating perplexity for each epoch is computationally

Table 3: Evaluations of the initial (zero-shot) and finetuned StackLLaMA models after 600 epochs. We measure alignment using an average over three reward model trained with three different seeds and drift with perplexity on the data. HumanEval is a programming benchmark that acts as a practical measure of drift / alignment tax.

|  | $\uparrow \Delta$ REWARD | $\downarrow$ PERPLEXITY | $\uparrow$ HUMANEVAL (PASS @ 1,PASS @ 10) |
| :--- | :---: | :---: | :---: |
| ZERO-SHOT | 0 | 4.43 | $11.0,12.7$ |
| PPO | $0.81 \pm 0.06$ | 4.62 | $7.8,10.7$ |
| ELASTIC RESET | $0.96 \pm 0.09$ | 4.57 | $11.0,13.0$ |

infeasible so we measure drift during training with the $\mathrm{KL}$ from the pretrained model over samples as done previously in other larger-scale RLHF [Bai et al., 2022, Gao et al., 2022].

Results We plot reward in Figure $6 a^{2}$ and $\mathrm{KL}$ from initial model over training in Figure 6b. As noted by Beeching et al. [2023], the task is much noisier at a larger scale and with a real HF reward model. As a sanity check, we find that Elastic Reset tracks PPO until the first reset at 260 epochs where it drops only slightly, but also doesn't lose much performance. Around the second reset at 520 epochs, we see a much sharper drop but also maintaining the same approximate reward. At the end, Elastic Reset provides a non-trivial reduction in drift while aligning just as well as PPO. The pareto curve in Figure 6c shows Elastic Reset is equal or slightly worse at low reward but shows large improvements over PPO at higher reward. Notably, Elastic Reset seems to work out-of-the-box with LoRA. To evaluate drift another way, we get the perplexity of the final models over the StackExchange validation set as in Section 6. For a more robust view of reward, we train two more reward models using different seeds and evaluate the increase in reward between initial and final models. We show mean and standard deviation across the three reward models in Table 3, we find that Elastic Reset achieves a slightly better final reward than PPO while maintaining lower perplexity on the data. To examine a true alignment tax, we run our models on HumanEval Chen et al. [2021], a programming benchmark that provides another view of drift. Answering human-written coding questions is both a useful capability for our model and also falls within a similar domain to StackExchange. The benchmark tests for functional correctness such that pass @ 1 corresponds to the percentage of problems solved by the model on the first try as shown in Table 3. Training with PPO degrades performance compared to the initial model, demonstrating a large alignment tax. In contrast, Elastic Reset achieves a similar reward but maintains performance, even slightly improving on pass @ 10, creating an alignment bonus [Askell et al., 2021] instead of tax.

## 8 Limitations

As a method, Elastic Reset is quite cheap computationally because both EMA updates and resets take negligable time compared to RLHF training and the EMA model can be stored on CPU. But our method is sensitive to the choice of reset rate; we chose heuristically based on when it seemed the model was overfitting. It is also possible to reset the policy and EMA model at different time scales, which could be a source of improvement. Our method also resets all of the trainable parameters, research in similar methods suggests that resetting larger models can benefit from resetting only part of the network [Zhou et al., 2022, Nikishin et al., 2022] or weighted-averaging instead of resets [D'Oro et al., 2023]. We leave both of these directions to future work.

Although we have thoroughly investigated our method on three different tasks, we note that none of them are ideal RLHF benchmarks. As pointed out by Gao et al. [2022], we measure our model's performance using the same reward model we optimize. This can lead to reward model overoptimization and our metric could mask overfitting and lack of generalization to the real world i.e. actual human preferences. An ideal benchmark could include a "gold" reward model as a proxy for human preference [Gao et al., 2022], but no such benchmarks are open-sourced and available.[^1]

Finally, we note that we follow all previous RLHF work and investigate only on-policy methods [Ziegler et al., 2019, Stiennon et al., 2020, Askell et al., 2021, Bai et al., 2022]. Previous work in resetting for RL has focused on off-policy methods and demonstrated strong performance [Nikishin et al., 2022, D' Oro et al., 2023]. As previously noted, our method can be seen as an adaptation of those to on-policy RL. In RLHF, PPO is by far the most popular method and on-policy is the dominant paradigm since it guarantees better local gradients. But it is possible that off-policy methods could implicitly balance performance and drift by incorporating a replay buffer with older data.

## 9 Conclusion

The problems of drift [Lee et al., 2019], alignment tax [Askell et al., 2021], reward model overoptimization [Gao et al., 2022], and reward hacking [Clark and Amodei, 2016] are inherent to RLHF and reduce its efficacy. We have introduced a simple but powerful new method, Elastic Reset, to tackle this problem and improve performance while maintaining linguistic capabilities. We have shown its ability on three different tasks and across three different scales: from 6 layer Transformers to GPT2 to LLaMA-7B. The problem of drift is currently being addressed with a standard KL penalty despite the computational cost, tradeoff with reward, and recent claims that is may be unnecessary [Bai et al., 2022, Gao et al., 2022]. Elastic Reset is a cheap and effective method to tackle the same problem, achieving a better tradeoff of reward and drift while reducing alignment tax. We hope our method leads to better RLHF and therefore models that are closer aligned with human preferences [Ziegler et al., 2019]. As well, we hope this work invigorates more research into improving the reward / drift tradeoff of RLHF with a focus on computationally efficient methods that scale.

## Acknowledgments and Disclosure of Funding

MN is supported by Fonds de recherche du Québec - Nature et technologies and Sony. MN would like to thank Issam Laradji, ServiceNow Research, Mila, and Compute Canada for providing resources used in the experiments.

## References

P. Ammanabrolu. Re: Reproducing NLPO (RL4LMs), Jan. 2023.

J. Andreas, D. Klein, and S. Levine. Learning with Latent Language. In NAACL. arXiv, 2018. doi: 10.48550/arXiv.1711.00482. URL http://arxiv.org/abs/1711.00482. arXiv:1711.00482 [cs].

A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan. A General Language Assistant as a Laboratory for Alignment, Dec. 2021. URL http://arxiv.org/abs/2112. 00861. arXiv:2112.00861 [cs].

D. Bahdanau, K. Cho, and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. In ICLR. arXiv, 2015. URL http://arxiv.org/abs/1409.0473. arXiv:1409.0473 [cs, stat].

Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. HatfieldDodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback, Apr. 2022. URL http://arxiv.org/abs/2204.05862. arXiv:2204.05862 [cs].

E. Beeching, Y. Belkada, K. Rasul, L. Tunstall, L. v. Werra, N. Rajani, and N. Lambert. StackLLaMA: An RL Fine-tuned LLaMA Model for Stack Exchange Question and Answering, 2023. URL https://huggingface.co/blog/stackllama.

E. M. Bender and A. Koller. Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5185-5198, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.463. URL https://aclanthology.org/2020.acl-main. 463.

T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language Models are Few-Shot Learners. In Neural Information Processing Systems. arXiv, July 2020. doi: 10.48550/arXiv.2005.14165. URL http://arxiv.org/abs/2005.14165. arXiv:2005.14165 [cs].

M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging Properties in Self-Supervised Vision Transformers, May 2021. URL http://arxiv.org/abs/ 2104.14294. arXiv:2104.14294 [cs].

M. Cettolo, C. Girardi, and M. Federico. WIT3: Web Inventory of Transcribed and Translated Talks. In Proceedings of the 16th Annual conference of the European Association for Machine Translation, pages 261-268, Trento, Italy, May 2012. European Association for Machine Translation. URL https://aclanthology.org/2012.eamt-1.60.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating Large Language Models Trained on Code, July 2021. URL http://arxiv.org/abs/ 2107.03374. arXiv:2107.03374 [cs].

P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep Reinforcement Learning from Human Preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://papers.nips.cc/paper/2017/ hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html.

J. Clark and D. Amodei. Faulty Reward Functions in the Wild, Dec. 2016. URL https://openai . com/blog/faulty-reward-functions/.

T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. In NeurIPS. arXiv, Nov. 2022. doi: 10.48550/arXiv.2208.07339. URL http://arxiv.org/abs/2208.07339. arXiv:2208.07339 [cs].

P. D'Oro, M. Schwarzer, E. Nikishin, P.-L. Bacon, M. G. Bellemare, and A. Courville. SampleEfficient Reinforcement Learning by Breaking the Replay Ratio Barrier. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= $4 \mathrm{GBGwVIEYJ.}$

D. Elliott, S. Frank, K. Sima' an, and L. Specia. Multi30K: Multilingual English-German Image Descriptions. In Proceedings of the 5th Workshop on Vision and Language, pages 70-74, Berlin, Germany, 2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-3210. URL http://aclweb.org/anthology/W16-3210.

D. Elliott, S. Frank, L. Barrault, F. Bougares, and L. Specia. Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description. In Proceedings of the Second Conference on Machine Translation, pages 215-233, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4718. URL https: //aclanthology.org/W17-4718.

L. Gao, J. Schulman, and J. Hilton. Scaling Laws for Reward Model Overoptimization, Oct. 2022. URL http://arxiv.org/abs/2210.10760. arXiv:2210.10760 [cs, stat].

S. Gugger, L. Debut, T. Wolf, P. Schmid, Z. Mueller, S. Mangrulkar, M. Sun, and B. Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable, 2022. URL https://github.com/huggingface/accelerate.

S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8):1735-1780, Nov. 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL https://doi.org/10. 1162/neco.1997.9.8.1735.

E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models, Oct. 2021. URL http://arxiv.org/abs/2106. 09685. arXiv:2106.09685 [cs].

J. D. Hunter. Matplotlib: A 2D Graphics Environment. Computing in Science \& Engineering, 9(3): 90-95, May 2007. ISSN 1558-366X. doi: 10.1109/MCSE.2007.55. URL https://ieeexplore. ieee.org/document/4160265. Conference Name: Computing in Science \& Engineering.

E. Jang, S. Gu, and B. Poole. Categorical Reparameterization with Gumbel-Softmax. In ICLR. arXiv, Aug. 2017. doi: 10.48550/arXiv.1611.01144. URL http://arxiv.org/abs/1611.01144. arXiv:1611.01144 [cs, stat].

N. Jaques, S. Gu, D. Bahdanau, J. M. Hernández-Lobato, R. E. Turner, and D. Eck. Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control. In International Conference on Machine Learning. arXiv, Oct. 2017. doi: 10.48550/arXiv.1611.02796. URL http://arxiv.org/abs/1611.02796. arXiv:1611.02796 [cs].

N. Jaques, A. Ghandeharioun, J. H. Shen, C. Ferguson, A. Lapedriza, N. Jones, S. Gu, and R. Picard. Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog, July 2019. URL http://arxiv.org/abs/1907.00456. arXiv:1907.00456 [cs, stat].

S. Kirby. Spontaneous evolution of linguistic structure-an iterated learning model of the emergence of regularity and irregularity. IEEE Transactions on Evolutionary Computation, 5(2):102-110, Apr. 2001. ISSN 1941-0026. doi: 10.1109/4235.918430. Conference Name: IEEE Transactions on Evolutionary Computation.

J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521-3526, Mar. 2017. doi: 10.1073/pnas.1611835114. URL https: //www.pnas.org/doi/10.1073/pnas.1611835114. Publisher: Proceedings of the National Academy of Sciences.

P. Koehn, H. Hoang, A. Birch, C. Callison-Burch, M. Federico, N. Bertoldi, B. Cowan, W. Shen, C. Moran, R. Zens, C. Dyer, O. Bojar, A. Constantin, and E. Herbst. Moses: Open Source Toolkit for Statistical Machine Translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions, pages 177-180, Prague, Czech Republic, June 2007. Association for Computational Linguistics. URL https://aclanthology.org/P07-2045.

N. Lambert, L. Tunstall, N. Rajani, and T. Thrush. HuggingFace H4 Stack Exchange Preference Dataset, 2023. URL https://huggingface.co/datasets/HuggingFaceH4/ stack-exchange-preferences.

A. Lazaridou and M. Baroni. Emergent Multi-Agent Communication in the Deep Learning Era, July 2020.

A. Lazaridou, A. Potapenko, and O. Tieleman. Multi-agent Communication meets Natural Language: Synergies between Functional and Structural Language Learning. In ACL. arXiv, May 2020. doi: 10.48550/arXiv.2005.07064. URL http://arxiv.org/abs/2005.07064. arXiv:2005.07064 $[\mathrm{cs}]$.

J. Lee, K. Cho, and D. Kiela. Countering Language Drift via Visual Grounding, Sept. 2019. URL http://arxiv.org/abs/1909.04499. arXiv:1909.04499 [cs].

M. Lewis, D. Yarats, Y. N. Dauphin, D. Parikh, and D. Batra. Deal or No Deal? End-to-End Learning for Negotiation Dialogues, June 2017. URL http://arxiv.org/abs/1706. 05125. arXiv:1706.05125 [cs] version: 1.

Q. Lhoest, A. V. del Moral, Y. Jernite, A. Thakur, P. von Platen, S. Patil, J. Chaumond, M. Drame, J. Plu, L. Tunstall, J. Davison, M. Šaško, G. Chhablani, B. Malik, S. Brandeis, T. L. Scao, V. Sanh, C. Xu, N. Patry, A. McMillan-Major, P. Schmid, S. Gugger, C. Delangue, T. Matussière, L. Debut, S. Bekman, P. Cistac, T. Goehringer, V. Mustar, F. Lagunas, A. M. Rush, and T. Wolf. Datasets: A Community Library for Natural Language Processing, Sept. 2021. URL http: //arxiv.org/abs/2109.02846. arXiv:2109.02846 [cs].

F. Li and M. Bowling. Ease-of-Teaching and Language Structure from Emergent Communication. In NeurIPS, 2019. URL http://arxiv.org/abs/1906.02403. arXiv: 1906.02403.

I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, Feb. 2022. URL https://openreview.net/forum?id= Bkg6RiCqY7.

R. Lowe*, A. Gupta*, J. Foerster, D. Kiela, and J. Pineau. On the interaction between supervision and self-play in emergent communication. In International Conference on Learning Representations, Sept. 2021. URL https://openreview.net/forum?id=rJxGLlBtwH.

Y. Lu, S. Singhal, F. Strub, O. Pietquin, and A. Courville. Countering Language Drift with Seeded Iterated Learning. In ICML. arXiv, Aug. 2020. doi: 10.48550/arXiv.2003.12694. URL http: //arxiv.org/abs/2003.12694. arXiv:2003.12694 [cs].

A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142-150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL https://aclanthology. org/P11-1015.

C. J. Maddison, A. Mnih, and Y. W. Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. In ICLR. arXiv, Mar. 2017. doi: 10.48550/arXiv.1611.00712. URL http://arxiv.org/abs/1611.00712. arXiv:1611.00712 [cs, stat].

S. Mangrulka, S. Gugger, L. Debut, Y. Belkada, and S. Paul. PEFT: State-of-the-art ParameterEfficient Fine-Tuning methods, May 2023. URL https://github.com/huggingface/peft. original-date: 2022-11-25T03:51:09Z.

M. McCloskey and N. J. Cohen. Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. In G. H. Bower, editor, Psychology of Learning and Motivation, volume 24, pages 109-165. Academic Press, Jan. 1989. doi: 10.1016/S0079-7421(08)60536-8. URL https: //www.sciencedirect.com/science/article/pii/S0079742108605368.

Meta FAIR Diplomacy Team, A. Bakhtin, N. Brown, E. Dinan, G. Farina, C. Flaherty, D. Fried, A. Goff, J. Gray, H. Hu, A. P. Jacob, M. Komeili, K. Konath, M. Kwon, A. Lerer, M. Lewis, A. H. Miller, S. Mitts, A. Renduchintala, S. Roller, D. Rowe, W. Shi, J. Spisak, A. Wei, D. Wu, H. Zhang, and M. Zijlstra. Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science, Nov. 2022. doi: 10.1126/science.ade9097. URL https: //www.science.org/doi/10.1126/science.ade9097. Publisher: American Association for the Advancement of Science.

E. Nikishin, M. Schwarzer, P. D'Oro, P.-L. Bacon, and A. Courville. The Primacy Bias in Deep Reinforcement Learning. In Proceedings of the 39th International Conference on Machine Learning, pages 16828-16847. PMLR, June 2022. URL https://proceedings.mlr.press/ v162/nikishin22a.html. ISSN: 2640-3498.

OpenAI. ChatGPT: Optimizing Language Models for Dialogue, Nov. 2022. URL https://webcache.googleusercontent.com/search?q=cache:qLONB_tyjdcJ:https: //openai.com/blog/chatgpt/\&cd=1\&hl=en\&ct=clnk\&gl=ca.

M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and M. Auli. fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North, pages 48-53, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-4009. URL http://aclweb.org/anthology/N19-4009.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, Mar. 2022. URL http://arxiv.org/abs/2203.02155. arXiv:2203.02155 [cs].

A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library, Dec. 2019. URL http://arxiv.org/abs/1912.01703. arXiv:1912.01703 [cs, stat].

M. Post. A Call for Clarity in Reporting BLEU Scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186-191, Brussels, Belgium, Oct. 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-6319. URL https://aclanthology . org/W18-6319.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language Models are Unsupervised Multitask Learners, 2019. URL https://www.semanticscholar.org/ paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/ 9405cc0d6169988371b2755e573cc28650d14dfe.

R. Ramamurthy, P. Ammanabrolu, K. Brantley, J. Hessel, R. Sifa, C. Bauckhage, H. Hajishirzi, and Y. Choi. Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization. In ICLR. arXiv, 2022. URL http://arxiv.org/abs/2210.01241. arXiv:2210.01241 [cs].

M. Rita, C. Tallec, P. Michel, J.-B. Grill, O. Pietquin, E. Dupoux, and F. Strub. Emergent Communication: Generalization and Overfitting in Lewis Games, Sept. 2022. URL http: //arxiv.org/abs/2209.15342. arXiv:2209.15342 [cs, math].

A. Robins. Catastrophic Forgetting, Rehearsal and Pseudorehearsal. Connection Science, 7 (2):123-146, June 1995. ISSN 0954-0091. doi: 10.1080/09540099550039318. URL https://doi.org/10.1080/09540099550039318. Publisher: Taylor \& Francis _eprint: https://doi.org/10.1080/09540099550039318.

D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne. Experience Replay for Continual Learning, Nov. 2019. URL http://arxiv.org/abs/1811.11682. arXiv:1811.11682 [cs, stat].

V. Sanh, L. Debut, J. Chaumond, and T. Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, Feb. 2020. URL http://arxiv.org/abs/1910.01108. arXiv:1910.01108 $[\mathrm{cs}]$.

J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. Trust Region Policy Optimization. In ICML. arXiv, 2015. doi: 10.48550/arXiv.1502.05477. URL http://arxiv.org/abs/1502 . 05477. arXiv: 1502.05477 [cs].

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimization Algorithms, Aug. 2017. URL http://arxiv.org/abs/1707.06347. arXiv:1707.06347 [cs].

A. K. Singh, D. Ding, A. Saxe, F. Hill, and A. K. Lampinen. Know your audience: specializing grounded language models with the game of Dixit, June 2022. URL http://arxiv.org/abs/ 2206 . 08349. arXiv:2206.08349 [cs].

S. Steinert-Threlkeld, X. Zhou, Z. Liu, and C. M. Downey. Emergent Communication Fine-tuning (EC-FT) for Pretrained Language Models. In Emergent Communication Workshop at ICLR 2022, June 2022. URL https://openreview.net/forum?id=SUqrM7WR7W5.

N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. Christiano. Learning to summarize from human feedback. In NeurIPS. arXiv, 2020. URL http://arxiv.org/abs/2009.01325. arXiv:2009.01325 [cs].

I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to Sequence Learning with Neural Networks. In Neural Information Processing Systems. arXiv, Dec. 2014. URL http://arxiv.org/abs/1409 . 3215. arXiv:1409.3215 [cs].

H. Q. To, N. D. Q. Bui, and M. Nguyen. CodeCapybara: Open Source LLaMA Model that Follow Instruction-Tuning for Code Generation., May 2023. URL https://github.com/ FSoft-AI4Code/CodeCapybara. original-date: 2023-04-21T10:28:53Z.

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. LLaMA: Open and Efficient Foundation Language Models, Feb. 2023. URL http://arxiv.org/abs/2302.13971. arXiv:2302.13971 [cs] version: 1.

M. Utiyama and H. Isahara. A Comparison of Pivot Methods for Phrase-Based Statistical Machine Translation. In Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Proceedings of the Main Conference, pages 484-491, Rochester, New York, Apr. 2007. Association for Computational Linguistics. URL https://aclanthology.org/N07-1061.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is All you Need. In Neural Information Processing Systems, page 11, 2017.

L. von Werra, Y. Belkada, L. Tunstall, E. Beeching, T. Thrush, and N. Lambert. TRL: Transformer Reinforcement Learning, 2023. URL https://github.com/lvwerra/trl.

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, page 28, 1992.

T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. Rush. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45, Online, Oct. 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL https: //aclanthology.org/2020.emnlp-demos.6.

H. Zhou, A. Vani, H. Larochelle, and A. Courville. Fortuitous Forgetting in Connectionist Networks. In International Conference on Learning Representations, Mar. 2022. URL https: //openreview.net/forum?id=ei3SY1_zYsE.

D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-Tuning Language Models from Human Preferences, 2019. URL http://arxiv.org/abs/ 1909 .08593. arXiv:1909.08593 [cs, stat].

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-16.jpg?height=374&width=1374&top_left_y=230&top_left_x=365)

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-16.jpg?height=282&width=439&top_left_y=255&top_left_x=388)

(a) Task Performance

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-16.jpg?height=285&width=441&top_left_y=253&top_left_x=842)

(b) Language Drift

![](https://cdn.mathpix.com/cropped/2024_06_04_f8a41fe9ad640e713859g-16.jpg?height=285&width=441&top_left_y=253&top_left_x=1292)

(c) Pareto Frontier Graph

Figure 7: All transformer model methods on the Translation Game. We measure (a) Task Performance with FR $\rightarrow \mathrm{EN} \rightarrow \mathrm{DE}$ BLEU and (b) Language Drift with FR $\rightarrow$ EN BLEU, on the validation set during finetuning. We plot the mean and show error bars for standard deviation over 5 seeds. To compare how methods do on both metrics, we plot (c) the best achieved drift vs task performance across finetuning.
</end of paper 3>


<paper 4>
# Helping or Herding? 邻 

## REWARD MoDEL ENSEMBLES MitiGATE BUT DO NOT ELIMINATE REWARD HACKING

Jacob Eisenstein ${ }^{1, *}$<br>Ahmad Beirami ${ }^{2}$<br>Katherine Heller ${ }^{2}$<br>Jonathan Berant ${ }^{1, *}$<br>1. Google DeepMind<br>2. Google Research<br>* Core contributors<br>reward-ensembles-helping-or-herding@google.com<br>Chirag Nagpal ${ }^{2, *}$<br>Alex D'Amour ${ }^{1}$<br>Stephen Pfoh ${ }^{2}$<br>Deepak Ramachandran ${ }^{2}$<br>Alekh Agarwal ${ }^{2, *}$<br>Adam Fisch ${ }^{1}$<br>Peter Shaw ${ }^{1}$


#### Abstract

Reward models play a key role in aligning language model applications towards human preferences. However, this setup creates an incentive for the language model to exploit errors in the reward model to achieve high estimated reward, a phenomenon often termed reward hacking. A natural mitigation is to train an ensemble of reward models, aggregating over model outputs to obtain a more robust reward estimate. We explore the application of reward ensembles to alignment at both training time (through reinforcement learning) and inference time (through reranking). First, we show that reward models are underspecified: reward models that perform similarly in-distribution can yield very different rewards when used in alignment, due to distribution shift. Second, underspecification results in overoptimization, where alignment to one reward model does not improve reward as measured by another reward model trained on the same data. Third, overoptimization is mitigated by the use of reward ensembles, and ensembles that vary by their pretraining seeds lead to better generalization than ensembles that differ only by their fine-tuning seeds, with both outperforming individual reward models. However, even pretrain reward ensembles do not eliminate reward hacking: we show several qualitative reward hacking phenomena that are not mitigated by ensembling because all reward models in the ensemble exhibit similar error patterns.


## 1 INTRODUCTION

To align machine learning systems with human preferences, it is common to use reward models that are finetuned on preference annotations to score potential outputs by how likely they are to be preferred by human raters (Christiano et al., 2017; Stiennon et al., 2020; Bai et al., 2022; Roit et al., 2023). There are many ways to use reward models to align policy models: they can act as training signals in reinforcement learning (Christiano et al., 2017; Stiennon et al., 2020), they can select examples for further imitation learning (Gulcehre et al., 2023; Liu et al., 2023; Dong et al., 2023; Touvron et al., 2023), or they can be applied at inference time to steer the output distribution toward higher expected reward (e.g., Yang \& Klein, 2021; Gao et al., 2023). Such procedures create a semi-adversarial dynamic in which the language model is encouraged to produce outputs that obtain high reward by exploiting errors in the reward model. Furthermore, while the reward model

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-02.jpg?height=819&width=1418&top_left_y=268&top_left_x=359)

x: context

Human: I want to make a nice steak dinner, but I don't know the difference between the various cuts of steak. Assistant: OK, l'm happy to help! I think you want to know which steaks are most tender [...] Human: What should we drink with it? Assistant:
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-02.jpg?height=656&width=634&top_left_y=430&top_left_x=366)

$\mathbf{x}$ context

Human: I'm going to the Netherlands and would like to learn a few common phrases in Dutch. Assistant: Sure, would you like to learn how to say [...] Human: Could you teach me how to say "Goodbye" and "Thank you"? Assistant

I

$(y \mid x)$

\{

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-02.jpg?height=431&width=627&top_left_y=560&top_left_x=1123)

$r^{*}=-1.04 \ll \bar{r}=+2.31$

Figure 1: Left: reward model ensembles can attenuate errors made by individual reward models, in this case the positive $r_{1}$ for this off-topic response from the policy model $\pi(y \mid x)$, which gets a low true reward $\left(r^{*}\right)$. Right: insufficiently diverse reward models unanimously rate this overly-verbose and non-responsive reply from $\pi(y \mid x)$ as positive, but it too gets a low true reward. Both examples are real outputs and rewards (here, represented as normalized Z-scores) from best-of- $n$ reranking on a dataset of dialogue turns rated for helpfulness (Bai et al., 2022); see the paper for details.

is trained on a fixed set of human preference data, the process of alignment shifts the distribution of its inputs, increasing the likelihood of such errors. This phenomenon where the policy language model exploits reward model errors is often termed reward hacking (Amodei et al., 2016), reward gaming (Skalse et al., 2022; Pang et al., 2023), or reward over-optimization (Gao et al., 2023).

Reward hacking has been investigated from several perspectives in prior work (e.g., Krakovna et al., 2020; Skalse et al., 2022; Pan et al., 2022). Bai et al. (2022) used reinforcement learning with human feedback (RLHF) and trained two reward models on non-overlapping splits of preference data, using one to drive alignment, and the other to measure the quality of the outputs. They find that RLHF increases performance according to both the driver and measurement models, but that a performance gap emerges as the policy is allowed to diverge from the initial distribution. However, both reward models were built on base models trained on the same pretraining data, which, as we will show, limits their diversity (as hypothesized by Gleave \& Irving (2022)) and thus may understate the effect of reward hacking. Other work has simulated the relationship between a "true" reward and a learned proxy, showing that it is possible to over-optimize the proxy to such an extent that the true reward starts to decrease (Gao et al., 2023; Coste et al., 2023). This has been replicated in more realistic settings by examining (and creating) spurious correlations in reward model training data (Pang et al., 2023).

In this work, we first analyze reward model distribution shift from the perspective of underspecification (D'Amour et al., 2022), which occurs when a machine learning pipeline yields reliable performance on held-out data from the training distribution, but variable performance on out-ofdistribution data. When applied to learning reward models from human preference data, we show that reward models that agree in-distribution often disagree when transferred out-of-distribution. Furthermore, such disagreements are more pronounced when the reward models are built on different pretrainings, even when that difference is induced merely by varying the pretraining random seed. These disagreements become increasingly severe when evaluated on outputs of a policy model that has been aligned to a specific reward model. This occurs both when using reward models in RLHF, as well as when using an inference-time alignment procedure, best-of- $n$ reranking, where $n$ samples are drawn from the policy and then reranked with a reward model.

Motivated by these findings, we systematically investigate reward model ensembles as a possible remedy for reward hacking. Assuming different models err in different ways, ensembling can leverage
reward uncertainty across the ensemble during alignment (see Figure 1, Left). We explore several techniques for aggregating scores across the ensemble, e.g., taking the median score as a robust estimate of the true reward of the policy. We also consider two types of ensembles: pretrain ensembles, where different members of the ensemble differ in the random seed used during the pretraining phase, and finetune ensembles, where members differ only in the random seed used during finetuning. These ensembles are then evaluated across several types of policies and preference annotations: dialogue preferences for a helpful assistant (Bai et al., 2022), summarization quality (Stiennon et al., 2020), and whether a single-document summary is grounded in its source text (Roit et al., 2023).

We find that pretrain ensembles substantially outperform finetune ensembles. Moreover, they consistently outperform single reward models, unlike finetune ensembles, which in many cases are comparable to single reward models. However, our analysis also reveals that policies trained with ensembles are still susceptible to reward hacking: different reward models sometimes share similar error patterns, which in turn propagate to the ensemble (see Figure 1, Right). This is exploited and amplified by the policy, leading, for example, to outputs that are too short when tuning for factuality, too verbose when tuning for summarization quality, or responses that follow a particular format that is often unsuitable, when training a helpful assistant. Thus, it is possible that methods that, unlike ensembles, are aware of the distance of outputs from the reward data distribution (Liu et al., 2020) could provide more reliable estimates of uncertainty.

In concurrent work, Coste et al. (2023) argue that reward model ensembles effectively mitigate reward hacking. Our work shares a similar research question, but differs in several ways, leading to more nuanced conclusions. First, we investigate the difference between pretrain and finetune ensembles, finding that pretrain ensembles are considerably more effective. Second, we use human-annotated preference data rather than synthetically-generated labels, which provides a more realistic experimental setup. Third, we perform analysis that demonstrates the limitations of reward ensembles, showing reward ensembles are still susceptible to reward hacking. Last, our experimental setup covers a wider range of tasks, larger reward models, and more extensive policy optimization.

## 2 PRELIMINARIES

Reward models have become the primary tool for aligning LMs towards user-facing applications. We now briefly review how reward models are trained (\$2.1) and how they are used for alignment (§2.2). We then describe the experimental setup that we will use for the remainder of the paper (§2.3).

### 2.1 REWARD MODEL TRAINING

We focus on the the typical setup where reward models are trained from preference data, $\left(x, y^{+}, y^{-}\right) \in$ $D$, where $y^{+}$is annotated to be preferred over $y^{-}$for prompt $x$. Under the Bradley-Terry model (Bradley \& Terry, 1952), the probability that response $y_{2}$ is preferred over $y_{1}$ given a reward function $r$ and a prompt $x$ is $p\left(y_{1} \prec y_{2} \mid x\right)=\sigma\left(r\left(x, y_{2}\right)-r\left(x, y_{1}\right)\right)$, where $\sigma(\cdot)$ is the sigmoid function. Then, we can use preference data to train a reward model by maximizing

$$
\begin{equation*}
\mathcal{J}(r)=\mathbb{E}_{\left(x, y^{+}, y^{-}\right) \sim D}\left[\log p\left(y^{-} \prec y^{+} \mid x\right)\right] \tag{1}
\end{equation*}
$$

The Bradley-Terry model is underdetermined: for any reward model $r^{*}$, we can define an equivalent reward model, $r^{\prime}(x, y)=r^{*}(x, y)+C(x)$ where $C(x)$ is a prompt-dependent constant, obtaining the same objective value as $r^{*}$, i.e., $\mathcal{J}\left(r^{*}\right)=\mathcal{J}\left(r^{\prime}\right)$. This is problematic for ensembling: if different reward models choose different values for $C(x)$, then order statistics like median and minimum are meaningless. We therefore modify the objective function by adding a regularization term to encourage the sum of reward values per preference pair to stay close to zero, i.e.,

$$
\begin{equation*}
\mathcal{J}_{\text {reg }}(r)=\mathcal{J}(r)+\eta \cdot \mathbb{E}_{\left(x, y^{+}, y^{-}\right) \sim D}\left[\left(r\left(x, y^{+}\right)+r\left(x, y^{-}\right)\right)^{2}\right] \tag{2}
\end{equation*}
$$

where $\eta$ is a small positive value, thereby resolving the issue of underdetermination.

Note that reward models can also be trained from "pointwise" data, such as toxicity or factuality annotations on individual examples (Yang \& Klein, 2021; Roit et al., 2023). Such reward models are not underdetermined and so can be aggregated without adjustment.

| Task | Prompt | Output |
| :--- | :--- | :--- | :--- |
| TL;DR | I'll try to keep this short! **Background****'ve always been an on again/off again (very <br> casual!) jogger, typically doing $3-5 \mathrm{k} *$ My knees have always been finicky, and I went to a <br> physio who thought I had "runner's knee" $[\ldots]$ I had my baby a year ago, so all in all I haven't <br> run for about 1.5 years [...] However, when I run even the tinyiest bit, or even go on a long <br> walk or a hike, my pelvis gets very sore and tight $[\ldots]$ | I think pregnancy messed with my body, now I <br> can't even run even the smallest amount with- <br> out pain in my pelvis and knees. I'm fairly <br> certain the problem isn't just that I'm com- <br> pletely out of shape. |
| Helpfulness | Human: Assistant, can you help me find books that have really popular antagonists? Assistant: <br> Sure! I know many books that feature good antagonists, such as those in the Lord of the Rings <br> books. Human: Who specifically from Lord of the Rings? Assistant: | Well, Sauron, for example. If you want, I can <br> explain who he was. |
| XSum/NLI | The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foun- <br> dation - a charity to raise money for Nigerian sport. Mr Sodje, 37, is jointly charged with elder <br> brothers Efe, 44, Bright, 50 and Stephen, 42. Appearing at the Old Bailey earlier, all four <br> denied the offence. The charge relates to offences which allegedly took place between 2008 <br> and 2014. Sam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, <br> are due to stand trial in July. They were all released on bail. | Former Premier League footballer Sam Sodje <br> accused of charity fraud. |

Table 1: Prompt-output pairs for the three benchmarks we consider. See $\S 2.3$.

### 2.2 AliGning LanguAGe ModelS USInG ReWARd ModelS

Best-of- $n$ reranking (BoN) is an inference-time alignment strategy, where given a prompt $x$, we sample $n$ generations $y_{1}, \ldots, y_{n}$ from a policy language model $\pi(y \mid x)$ and return the generation that has the highest reward according to a reward model $r$, i.e., $y *=\arg \max _{y_{k} \in\left\{y_{1}, \ldots, y_{n}\right\}} r\left(x, y_{k}\right)$. The Kullback-Leibler (KL) divergence of BoN from the initial policy is upper bounded by $\log n-\frac{n-1}{n}$. BoN tends to outperform more elaborate alignment techniques like RLHF in the low-KL regime (Gao et al., 2023), albeit with the cost of generating multiple samples at inference time.

Reinforcement Learning from Human Feedback (RLHF) is an online reinforcement learning method that trains a policy language model $\pi$ to maximize expected reward, while staying close to an initial policy, $\pi_{\mathrm{sft}}$, which is typically finetuned on supervised data (prompt-output pairs). Distance from the initial policy is measured with KL divergence, which leads to the regularized objective

$$
\begin{equation*}
\max _{\pi} \underset{\substack{x \sim \rho \\ y \sim \pi}}{\mathbb{E}}[r(x, y)]-\lambda \mathrm{KL}\left(\pi \| \pi_{\mathrm{sft}}\right) \tag{3}
\end{equation*}
$$

where $r$ is a reward model, $\rho$ is a distribution over prompts, and $\lambda$ is a hyper-parameter. Typically, this objective is optimized using PPO (Schulman et al., 2017), which we also use in this work.

### 2.3 EXPERIMENTAL SETUP

Datasets We will examine the performance of reward models (both single models and ensembles) across three tasks. An example from each task is provided in Table 1.

- TL;DR: A summarization benchmark where authors summarize their own reddit posts (Völske et al., 2017). We use the preference data created by Stiennon et al. (2020). This benchmark has been commonly used to evaluate finetuning of policy LMs (Rafailov et al., 2023; Zhao et al., 2023).
- HELPFULNESS: A helpful assistant benchmark (Bai et al., 2022), where given a partial conversation between a human and a digital assistant the goal is to complete the next turn of the assistant. This benchmark has also been commonly used for evaluating finetuned policy LMs (Bai et al., 2022; Rafailov et al., 2023). We use the base dataset (44K examples), where responses are generated from a 52B context-distilled LM, and split the training set into two: half for training the reward model, and half for training the policy model.
- XSUM/NLI: We adopt the setup of factually-consistent summarization (Roit et al., 2023), where a model trained on XSum (Narayan et al., 2018) is finetuned to generate summaries that are consistent with the source document according to a Natural Language Inference (NLI) reward model.

Training reward models To examine the effect of pretraining on reward models, we pretrain five T5 models from scratch with the base (220M parameters), large (770M), and XL (3B) architectures, using the standard denoising objective over the C4 corpus (Raffel et al., 2020). The pretrained checkpoints differ only in their random seed, which controls parameter initialization and the sample from the pretraining data. The same pretrained models are used for finetuning across all tasks.

We finetune each pretrained model five times using different random seeds across all three benchmarks. In TL;DR and HELPFULNESS we use the aforementioned preference data. For XSUM/NLI, we finetune

| Model Size | TL;DR | HELPFULNESS | XSum/NLI |
| :--- | :---: | :---: | :---: |
| T5-BASE | $65.8 \pm 0.3$ | $66.7 \pm 0.7$ | $86.7 \pm 0.9$ |
| T5-LARGE | $69.3 \pm 0.7$ | $68.5 \pm 0.4$ | $88.3 \pm 1.2$ |
| T5-XL | $71.4 \pm 0.8$ | $69.2 \pm 0.6$ | $91.3 \pm 0.5$ |
| T5-XXL | 79.5 | 71.5 | 92.9 |

Table 2: Mean in-distribution accuracy of 25 trained reward models on validation data for TL;DR, HELPFULNESS, and XSUM/NLI. Standard deviation is also reported, and observed to be small indistribution. The single T5-XXL reward model is used for evaluation purposes only.

NLI models on the ANLI dataset (Nie et al., 2020). Overall we obtain 25 reward models per task ( 5 pretrain $\times 5$ finetune). This makes it possible to evaluate the effect of pretraining and finetuning on underspecfication (§3) by constructing ensembles that differ in either pretrain or finetune seed (§4).


#### Abstract

Alignment strategy We use the publicly available T5-large model (Raffel et al., 2020) as a policy for the two summarization tasks. For helpfulness, the task requires substantial background knowledge, and thus we use the instruction-tuned PALM-2-XXS model (Anil et al., 2023). Prior to alignment, we create a finetuned policy $\pi_{\text {sft }}$ by finetuning on supervised data in the standard manner. We finetune on annotated summaries from TL;DR and XSUM/NLI for the corresponding tasks, and on the preferred responses, $\left(x, y^{+}\right)$, from the preference data in HELPFULNESS.

In BoN reranking, we rerank sampled sets of size $n \in\left\{2^{1}, 2^{2}, \ldots, 2^{5}\right\}$ for HELPFULNESS and $\left\{2^{1}, \ldots, 2^{6}\right\}$ for TL;DR. Larger sets lead to higher reward at a cost of more expensive inference and larger deviation from $\pi_{\text {sft }}$. In RLHF, we obtain a trade-off between the $\mathrm{KL}$ from $\pi_{\text {sft }}$ and the expected reward by training multiple times, varying the value of $\lambda$. Low values of $\lambda$ correspond to high KL and high reward, while high values of $\lambda$ entail low KL and low reward. For each value of $\lambda$ we train roughly to convergence using a predetermined fixed number of steps (all hyperparameter values, including $\lambda$ and the number of steps, are in Appendix C). Coste et al. (2023) trade-off KL and reward by tracking their values during training; however, for any particular value of KL the reward might still be underoptimized during training (i.e., there can exist a different policy $\pi(y \mid x)$ with better reward, but the same $\operatorname{KL}\left(\pi(y \mid x) \| \pi_{\mathrm{sft}}(y \mid x)\right)$, which can be found with longer training).


Evaluation We use two metrics to quantify generalization of reward models-reward by a larger model and win rate. Similar to past work (Gao et al., 2023; Coste et al., 2023), we use a larger reward model to evaluate the generalization of models trained with a smaller reward model. We train a T5-XXL reward model by taking the publicly available T5-XXL (Raffel et al., 2020) and finetuning it as described above. Table 2 details the performance of reward models of different sizes on the three tasks, and it can be seen that T5-XXL outperforms the best T5-XL model. We report both average reward of the $\mathrm{T} 5-\mathrm{XXL}$ evaluator as well as win rate, which is the fraction of prompts for which the response sampled from the aligned policy $\pi$ has higher reward compared to $\pi_{\text {sft }}$.

The errors of the T5-XXL autoeval model might correlate with errors of the smaller T5 models because they are trained on the same preference data. For this reason, we also evaluate win rate according to a prompted PALM-2-Large model, which was not exposed to the reward training data but was instruction-tuned on FLAN (Wei et al., 2022). Given a prompt $x$, we sample a response $y_{\text {stt }}$ from $\pi_{\text {sft }}$ and $y_{\text {rlhf }}$ from $\pi$. We then ask PALM-2 which response is better, using a hand-engineered prompt proposed by Rafailov et al. (2023). To avoid position bias we run PALM-2 on the two possible orderings $\left(y_{\mathrm{st}}, y_{\mathrm{rlhf}}\right)$ and $\left(y_{\mathrm{st}}, y_{\mathrm{rlhf}}\right)$, sample $K=8$ outputs for each order and determine the winner on this prompt through majority voting. This style of evaluation has become common recently (Dubois et al., 2023; Singhal et al., 2023) and was shown to correlate well with human judgements (Rafailov et al., 2023).

# 3 UNDERSPECIFICATION IN REWARD MODELS 

We now analyze alignment strategies that use a single reward model, and demonstrate that reward models are underspecified. First, Table 2 shows the average in-distribution accuracy across the 25 different reward models, together with the standard deviation (which is low in-distribution).
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-06.jpg?height=816&width=1390&top_left_y=278&top_left_x=365)

(a) TL;DR

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-06.jpg?height=337&width=1355&top_left_y=688&top_left_x=382)

(b) HELPFULNESS

Figure 2: Average reward of the best-of- $n$ output, as judged by: the same reward model used for ranking (self); reward models fine-tuned from the same pretrain as the ranker (same pretrain); reward models fine-tuned from different pretrains from the ranker (diff pretrain). The reward models that do not share a pretrain with the ranker regard the ranker's preferred outputs as significantly worse.

The story changes, however, when we move to out-of-distribution data. Figure 2 shows the expected reward achieved by BoN as a function of the number of sampled candidates, $n$, for three reward model scales (KL is approximately $\log n-\frac{n-1}{n}$ ). The dotted green line shows the expected reward of the top-ranked output according to the reranker itself, while the dashed orange line shows the expected reward of the same output according to reward models that share a pretrain seed. The solid blue line shows the expected reward according to reward models that do not share a pretrain seed. Unsurprisingly, the reranker scores its own top outputs more favorably than the other reward models do. However, the reranker's outputs are scored significantly less favorably by reward models which do not share a pretrain with the ranker. Reward models that share a pretrain seed with the ranker model overestimate the true reward of the top-ranked output-suggesting that finetune ensembles are not sufficiently diverse because of the shared pretraining state of each of the ensemble's members. Notably, this gap does not disappear with scale, and is present for base, large, and XL models.

Moving to alignment, differences in estimated rewards induce different policies from the BoN strategy: Figure 3 shows the effects on agreement of the top-ranked summary when reward models do (crosses) or do not (circles) share pretraining seeds. Different reward models tend to produce different 1-best outputs. Again these differences are strongly associated with the pretraining seed: for example, two reward models from different pretrains will choose a different best-of-16 output more than half the time for both TL;DR and HELPFULNESS and in all scales.

Last, Figure 4 analyzes the evolution of agreement of the estimated reward scores when performing RLHF on TL;DR for reward models of various scales. Specifically, we align a policy using a single reward model, and then measure how well pairs of reward models agree on the ranking of samples from that policy using Spearman rank correlation. To compute Spearman, we sample 5 completions for each prompt in the validation set from a policy model, at $2 \mathrm{~K}$ step intervals during RLHF. We compare the agreement between a set of 5 reward models that share the same pre-training seed and a set of 5 that do not (both sets include the reward model used to drive RLHF). For each prompt, we compute Spearman correlation across all ten pairs in each set and report the mean correlation over the pairs. The correlation of models that do not share a pretrain is lower compared to models that share a pretrain seed. Moreover, correlation goes down during RLHF, indicating that the uncertainty about the true reward increases as a result of alignment.
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-07.jpg?height=348&width=1374&top_left_y=278&top_left_x=365)

(a) TL;DR
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-07.jpg?height=354&width=1390&top_left_y=674&top_left_x=365)

(b) HELPFULNESS

Figure 3: Agreement of the top-ranked output between reward models that do (crosses) and do not (circles) share pretraining seeds. Underspecification of reward models directly affects the behavior of the aligned policy. Chance agreement is $1 / n$.
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-07.jpg?height=360&width=1390&top_left_y=1273&top_left_x=365)

Figure 4: Rank correlation of reward scores for TL;DR reward models that share a pretraining seed and models that do not. RLHF alignment increases disagreements between reward models (lower correlation), particularly at low values of $\lambda$ and for reward models that do not share a pretrain.

Overall, our analysis demonstrates that (1) different reward models tend to disagree on out-ofdistribution data, particularly when the reward models have different pretraining seeds; (2) this propagates to the trained policy model, in the sense that the resulting policy is highly tuned to the preferences of the specific reward model used to drive it; and (3) as a result, the disagreement between reward models tends to increase during alignment. These findings suggest that reward model ensembles might mitigate reward hacking, which we turn to next.

## 4 REWARD MODEL ENSEMBLES

We describe how to construct reward model ensembles (\$4.1), and evaluate their performance (\$4.2).

### 4.1 Pretrain and FinEtUNE ReWARd EnSEmbles

We showed that reward models are underspecified—as they are used more in alignment, they induce a stronger distribution shift in the outputs of the policy, which in turns leads to higher disagreement across reward models. Thus, a natural mitigation strategy is to ensemble multiple reward models, under the assumption that different models will have different errors. Aggregating over the scores
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-08.jpg?height=748&width=1390&top_left_y=279&top_left_x=365)

(a) TL;DR

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-08.jpg?height=337&width=1365&top_left_y=688&top_left_x=380)

(b) HELPFULNESS

Figure 5: In best-of- $n$ reranking, pretrain ensemble reward models significantly improve the quality of outputs in the TL;DR summarization task (top) and the HELPFULNESS task, as measured by a T5-XXL model. Full numerical results are in Appendix A.

of the ensemble members will help when some of the ensemble members erroneously assign high reward to a bad output.

Given a set of reward models $\mathcal{M}$, we define the reward of the ensemble to be $\bar{r}(x, y)=\operatorname{agg}\left(\left\{r_{m}(x, y)\right\}_{m \in \mathcal{M}}\right)$, with agg indicating an aggregation function (Dietterich, 2000; Lakshminarayanan et al., 2017; Raffel et al., 2020; Zaidi et al., 2021). Intuitively, the aggregation function should be conservative, and return a lower score when there is disagreement between the ensemble members. We consider the following simple aggregation function: MEAN, MEDIAN, and MEAN_MINUS_STD, which subtracts the standard deviation of the reward from the mean to penalize high variance. We also experiment with MIN, but overall find it to be inferior to the alternatives.

We evaluate two types of reward ensembles: pretrain ensembles, where each member was pretrained using a different random seed, ${ }^{1}$ and finetune ensembles, where all members share the same pretraining seed, but use a different seed when finetuned on the reward data (which typically includes preference pairs, where one output is preferred over another). In all cases the ensemble contains exactly 5 individual reward models. Pretrain ensembles are significantly more expensive to train, but are more diverse and hence likely to lead to a more robust reward estimate. In fact, Gleave \& Irving (2022) reported negative results when using reward ensembles and hypothesized this is due to ensemble members sharing the same underlying pretrained model.

### 4.2 EXPERIMENTS

We now evaluate reward model ensembles across all tasks. Figure 5 shows the results of ensembling in best-of- $n$ reranking, as measured by an XXL-scale fine-tuned reward model. Pretrain ensembles consistently improve performance over individual reward models, especially for higher values of $n$ for both TL;DR and HELPFULNESS. Finetune ensembles, conversely, improve performance in some cases and are comparable in others. For example, on TL;DR a pretrain ensemble with the MEAN aggregator achieves a win rate of $90 \%$ over the SFT outputs at the XL scale, while the win rate of a finetune ensemble with the same MEAN aggregator is $87.3 \%$. The win rate of the average individual XL-scale reward model is $85.3 \%$ (see Table 7). For visual clarity, in Figure 5 we show only two[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-09.jpg?height=1044&width=1400&top_left_y=275&top_left_x=362)

(a) TL;DR
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-09.jpg?height=450&width=1376&top_left_y=797&top_left_x=367)

(b) HELPFULNESS

Figure 6: In RLHF, pretrain ensemble reward models lead to significantly more favorable reward-KL tradeoffs, as judged by a T5-XXL autoeval model. Each point corresponds to training of models to convergence with a particular value of $\lambda$. We show the MEDIAN aggregator here, full numerical results are in Appendix B.

aggregators: MEAN and MEAN_MINUS_STD; see Appendix A for results with other aggregators. In general, the differences between aggregators are small, with MEAN usually performing at, or near, the top. More conservative aggregators (MIN and MEAN_MINUS_STD) come out slightly ahead of MEAN at the smaller scales on TL;DR, suggesting that high variance may be a bigger issue in this setting.

Figure 6 shows the KL-reward trade-off of ensemble reward models in RLHF for TL;DR and HELPFULNESS (evaluated with the finetuned T5-XXL model). In such plots, a better model is one that improves reward and/or reduces the value of KL from the original SFT policy (Gao et al., 2023; Coste et al., 2023). Indeed, similar to BoN, pretrain ensembles consistently outperform both finetune ensembles as well as the average individual model. We present results for the MEDIAN and MEAN aggregators for visual clarity, and report full numerical results in Appendix B. In RLHF, KL values are much higher than BoN (which is bounded by $\approx 3.17$ for $n=64$ ). Consequently, in this setting we witness explicit reward hacking, in which the T5-XXL rewards decrease even as the RLHF objective improves. This happens most prominently for individual models, in many cases for finetune ensembles, and most rarely for pretrain ensembles-where T5-XXL reward scores decrease only when RLHF uses a T5-Base reward model. Thus, our experiments on real data yield more negative conclusions than Coste et al. (2023) about the potential of ensembles to eliminate reward overoptimization.

Because the T5-XXL autoeval model is trained on the same data distribution as the reward models used for best-of- $n$ and RLHF, it may overstate their performance. For this reason, we also use a zero-shot autoeval model (PALM-2-Large), as described in Section 2.3. Because this evaluation is more computationally expensive, we apply it only to the largest-scale reward models (XL). Results are shown in Figure 7. Ensemble reward models consistently achieve higher win rates on both tasks and with both alignment techniques. For best-of- $n$, pretrain ensembles get significantly higher win rates on TL;DR at $n=64$ ( $p<.001$ by a permutation test); on HELPFULNESS the differences

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=919&width=1401&top_left_y=316&top_left_x=362)

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=406&width=618&top_left_y=331&top_left_x=382)

(a) BoN + TL;DR

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=361&width=697&top_left_y=798&top_left_x=367)

(c) RLHF + TL;DR

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=423&width=637&top_left_y=325&top_left_x=1058)

(b) BoN + HELPFULNESS

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=358&width=697&top_left_y=802&top_left_x=1061)

(d) RLHF + HELPFULNESS

Figure 7: Using a prompted autoevaluator (PALM-2-FLAN), ensemble reward models offer significantly better win rates on both TL;DR and HELPFULNESS. Here all reward models are XL-scale.

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-10.jpg?height=380&width=708&top_left_y=1393&top_left_x=706)

Figure 8: XSUM/NLI KL-reward tradeoff for pretrain ensembles, finetune ensembles, and individual models. Reward is measured with T5-XXL. Both pretrain and finetune ensembles slightly improve over individual models.

between ensembling techniques are not significant at $n=32$. On both tasks, single reward models are significantly worse, $p<.001$. For RLHF, pretrain ensembles generally achieve better or equal win rates at lower KL divergence from the reference policy, with particularly strong performance on HELPFULNESS. Overall, these results mirror the T5-XXL evaluation, with one interesting difference: the PALM-2 autoeval model reveals more reward hacking for RLHF, where win rate decreases with KL. This suggests that fine-tuned autoevaluators can overestimate performance when they are trained on the same preference data as the alignment reward models.

Figure 8 shows RLHF results for XSUM/NLI. Here we see a relatively small improvement for ensembles compared to individual models, and a very small difference between pretrain and finetune ensembles. We conjecture this is because XSUM/NLI optimizes for a particular aspect of the response, namely its factuality. This allows all models to find simple and similar strategies that lead to high reward (for example, emitting short responses with limited content), and thus ensembling does not lead to large gains in performance. We further elaborate on this when discussing limitations of ensembles in $\S 5$.

## 5 WhEN Do ReWard Model EnSEMBLes FaIL?

We saw that ensembles improve performance according to automatic evaluation metrics. We now conduct a complementary analysis that illustrates that, for some types of errors, ensembling is ineffective. When all reward models share a similar error pattern, this error propagates to the ensemble. Systematic errors across ensemble members can arise due to biases in the finite reward model training data.

To demonstrate this, we manually analyze ensemble outputs to detect frequent errors, and then perform a qualitative analysis. Figure 9 shows the results of this analysis on all three benchmarks. The x-axis corresponds to outputs of the model after training for a certain number of steps, and the y-axis is a statistic of interest (e.g., average output length). We plot the statistic value for the pretrained ensemble (using MEAN as a representative aggregation function) and for its members. In addition, for TL;DR and HELPFULNESS, where the reward model is trained on the preference data, we show the statistic value on the preference data validation set, conditioned on the label 'Preferred' or 'Rejected'.

- For HELPFULNESS (Figure 9a), outputs tend to be in a format of a list, and thus we write a regular expression that captures this format. The fraction of outputs that have this pattern increases to roughly $50 \%$ for 3 members of the ensemble and to the ensemble itself. Looking at the preference data, we do not detect a tendency to produce list outputs in the preferred responses, as the fraction of outputs that matches this format is roughly $8 \%$ for both the preferred and rejected responses.
- For TL;DR (Figure 9b), RLHF alignment leads to longer summaries (Singhal et al., 2023) and also outputs that are more extractive, i.e., copy more from the input. Summary length in characters grows substantially for the ensemble and all its members, where for the ensemble, length increases by a factor of two. On the preference data, indeed preferred responses are slightly longer than rejected responses, but much shorter than outputs post-RLHF. We also compute the longest common subsequence (in characters) between the document and the summary and find that it increases for the ensemble from 28.2 to 49.1. Again, the tendency for copying from the document already occurs in the preference data to a small degree, but is amplified by RLHF. ${ }^{2}$
- For XSUM/NLI (Figure 9c), training for factuality tends to make summaries shorter. Additionally, precise numbers are typically omitted from the summaries. Figure 9 shows how all members of the ensemble and the ensemble itself exhibit this phenomenon, with length in characters decreasing rapidly, as well as the fraction of examples that contain any numeric value whatsoever.

Overall, these qualitative findings are symptoms of the tendency for different pretrain reward models to learn to associate certain features with high reward. Policy models can then exploit this association, and use these features to produce outputs that are dramatically different from the reward training data, and that achieve (spuriously) high reward for both single reward models and the ensemble.

Why does this happen for both single reward models and reward model ensembles? As one indication, Lakshminarayanan et al. (2017) have proposed distance-awareness, i.e., the ability to quantify the distance of an example from the training set, as a necessary condition for achieving good uncertainty estimates. They showed in a synthetic binary classfication setup that deep ensembles provide good estimates when examples are on the decision boundary, but underestimate uncertainty in areas that are far from the training distribution. In LM alignment, the policy can shift the output distribution away from the decision boundary to areas where all reward models erroneously extrapolate in the same manner. While we focus on ensembles in this work, we hypothesize that the same phenomenon will occur in other approaches for uncertainty estimation that are not distance-aware, such as Monte-Carlo Dropout (Gal \& Ghahramani, 2016) and Epistemic Neural Networks (Osband et al., 2021).

## 6 CONCLUSION

In this work, we investigate reward model ensembles as a method for mitigating reward hacking. We find that diversity of the reward ensemble is crucial, and that a pretrain ensemble that contains members that do not share a pretrain seed leads to stronger generalization during alignment when compared to an ensemble whose members share a pretrain seed. However, reward ensembles are not always effective-for example, we find that they can still assign reward based on spurious correlations[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-12.jpg?height=556&width=653&top_left_y=291&top_left_x=736)

(a) HELPFULNESS. Fraction of answers containing lists (as matched by a regular expression).
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-12.jpg?height=570&width=1350&top_left_y=904&top_left_x=363)

(b) TL;DR. Left: extractiveness, as measured by average longest common substring between the summary and the context document. Right: length.
![](https://cdn.mathpix.com/cropped/2024_06_04_498da604b11aa2464a64g-12.jpg?height=564&width=1348&top_left_y=1569&top_left_x=366)

(c) XSUM/NLI. Left: length. Right: specificity, as measured by fraction of numerical tokens in the output.

Figure 9: Limitations of reward model ensembles. The x-axis is number of RLHF steps, the $y$-axis plots different statistics of the average validation output at that step, and the curves correspond to the pretrain ensemble (solid blue) and its members (dashed orange). For preference data, we plot the same statistics conditioned on the preference data label (Preferred vs. Rejected). On HELPFULNESS $(\lambda=0.05$, top), the ensemble tends to return a list of items. On TL;DR (center, $\lambda=0.01$ ), summaries become longer and copy longer spans from the original document. For XSUM/NLI $(\lambda=0.03$, bottom), responses are short and less specific, as measured by lack of numerical information. In HELPFULNESS and TL;DR, the statistics of the "aligned" outputs are far from their values in the preference data.
between the input and the label. If all members of the ensemble capture the same correlations, the ensemble will inherit the same undesirable behaviour. In such cases, the policy can exploit this vulnerability and shift the distribution towards outputs that overuse this correlation, which results in reward hacking. Consequently, reward model ensembles mitigate, but do not fully eliminate, reward hacking. Future work should examine methods for uncertainty estimation that are more robust to the type of distribution shift that occurs during alignment, particularly those that are aware of how different model policy outputs are from the preference data-such as Gaussian processes (Kuss \& Rasmussen, 2003; Chu \& Ghahramani, 2005; Liu et al., 2020) and conformal prediction under covariate shift (Tibshirani et al., 2019).

Acknowledgments Thanks to Sharat Chikkerur, Mohammad Havaei, and the anonymous reviewers for feedback on this paper. The research also benefited from feedback from David Bruns-Smith, Ming-Wei Chang, Michael Collins, Patrick Fernandez, Mandar Joshi, Rishabh Joshi, Balaji Lakshminarayanan, Kenton Lee, Kristina Toutanova, Victor Veitch, and Zihao Wang. Finally, we thank the people who built the infrastructure used in our experiments, including the T5X team and Léonard Hussenot, Johan Ferret, Robert Dadashi, Geoffrey Cideron, Alexis Jacq, Sabela Ramos, Piotr Stanczyk, Sertan Girgin, Danila Sinopalnikov, Amélie Héliou, Bobak Shahriari, Bilal Piot, Matt Hoffmann, Nikola Momchev, and Olivier Bachem.

## REFERENCES

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324-345, 1952. ISSN 00063444. URL http://www.jstor.org/stable/2334029.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Wei Chu and Zoubin Ghahramani. Preference learning with gaussian processes. In Proceedings of the 22nd international conference on Machine learning, pp. 137-144, 2005.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Alexander D' Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoffman, et al. Underspecification presents challenges for credibility in modern machine learning. The Journal of Machine Learning Research, 23(1):10237-10297, 2022.

Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classifier systems, pp. 1-15. Springer, 2000.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. AlpacaFarm: A simulation framework for methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050-1059. PMLR, 2016.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835-10866. PMLR, 2023.

Adam Gleave and Geoffrey Irving. Uncertainty estimation for language reward models. arXiv preprint arXiv: $2203.07472,2022$.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (ReST) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg. Specification gaming: the flip side of ai ingenuity. DeepMind Blog, 3, 2020.

Malte Kuss and Carl Rasmussen. Gaussian processes in reinforcement learning. Advances in neural information processing systems, 16, 2003.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, $30,2017$.

Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaji Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in Neural Information Processing Systems, 33:7498-7512, 2020.

Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial NLI: A new benchmark for natural language understanding. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.

Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Morteza Ibrahimi, Xiuyuan Lu, and Benjamin Van Roy. Epistemic neural networks. arXiv preprint arXiv:2107.08924, 2021.

Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In International Conference on Learning Representations (ICLR), 2022.

Richard Yuanzhe Pang, Vishakh Padmakumar, Thibault Sellam, Ankur Parikh, and He He. Reward gaming in conditional text generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Vlume 1: Long Papers), 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020.

Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Leonard Hussenot, Orgad Keller, Nikola Momchev, Sabela Ramos Garea, Piotr Stanczyk, Nino Vieillard, Olivier Bachem, Gal Elidan, Avinatan Hassidim, Olivier Pietquin, and Idan Szpektor. Factually consistent summarization via reinforcement learning with textual entailment feedback. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. arXiv preprint arXiv:2310.03716, 2023.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460-9471, 2022.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008-3021, 2020.

Ryan J Tibshirani, Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. Conformal prediction under covariate shift. In H. Wallach, H. Larochelle, A. Beygelzimer, F. dÁlché Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, 2019.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

Michael Völske, Martin Potthast, Shahbaz Syed, and Benno Stein. TL;DR: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pp. $59-63,2017$.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https: / openreview. net/forum? id=gEZrGCozdqR.

Kevin Yang and Dan Klein. FUDGE: Controlled text generation with future discriminators. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021.

Sheheryar Zaidi, Arber Zela, Thomas Elsken, Chris C Holmes, Frank Hutter, and Yee Teh. Neural ensemble search for uncertainty estimation and dataset shift. Advances in Neural Information Processing Systems, 34:7898-7911, 2021.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J. Liu. SLiC-HF: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.
</end of paper 4>


