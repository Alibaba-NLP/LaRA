<paper 0>
# KTO: Model Alignment as Prospect Theoretic Optimization 

Kawin Ethayarajh ${ }^{1}$ Winnie Xu ${ }^{2}$ Niklas Muennighoff ${ }^{2}$ Dan Jurafsky ${ }^{1}$ Douwe Kiela ${ }^{12}$


#### Abstract

Kahneman \& Tversky's prospect theory tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biasesthe success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being human-aware loss functions (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a KahnemanTversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B. Crucially, KTO does not need preferencesonly a binary signal of whether an output is desirable or undesirable for a given input. This makes it far easier to use in the real world, where preference data is scarce and expensive.


## 1. Introduction

Aligning generative models with human feedback has been successfully used to make generations more helpful, factual, and ethical, among other desiderata (Ouyang et al., 2022; Tian et al., 2023). For LLMs, alignment methods such as RLHF and DPO have consistently proven to be more beneficial than doing supervised finetuning (SFT) alone. However, human feedback is often discussed only in the context of preferences (e.g., output $A \succ B$ for input $x$ ), despite preferences being a kind of data that is relatively scarce and expensive to collect in the real world (Casper et al., 2023). This is largely because the alignment methods shown to work best-RLHF (Christiano et al., 2017) and[^0]

the mathematically equivalent DPO (Rafailov et al., 2023)_ take preference data as input.

To understand why these alignment methods work so well, and whether feedback needs to be in the form of preferences, we frame them through the lens of prospect theory (Kahneman \& Tversky, 1979; Tversky \& Kahneman, 1992). Prospect theory explains why humans make decisions about uncertain events that do not maximize expected value. It formalizes how humans perceive random variables in a biased but well-defined manner; for example, relative to some reference point, humans are more sensitive to losses than gains, a property called loss aversion. We show that popular alignment methods such as PPO (Schulman et al., 2017), DPO (Rafailov et al., 2023), and SLiC (Zhao et al., 2023) implicitly model such biases, helping explain their success independently of the data used. For this reason, we call them human-aware loss functions (HALOs).

Although it is impossible to say that HALOs are categorically better than non-HALOs, we find that among existing methods, those that meet the definition of a HALO work better than those that do not. We find that DPO performance can even be matched at most scales by running an offline PPO variant on dummy +1/-1 rewards, suggesting that preference data might not be needed if the inductive bias in the loss function is good enough. However, despite the surprising success of this simple baseline, it significantly lags behind DPO at the 30B model scale and suffers from hyperparameter sensitivity, making it difficult to use.

Taking a more principled approach, we derive a HALO using the model of human utility that Kahneman \& Tversky empirically derived to describe how humans make decisions about uncertain monetary outcomes (Tversky \& Kahneman, 1992). This approach, which we call Kahneman-Tversky Optimization (KTO), directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as most current methods do. KTO only requires a binary signal of whether an output is desirable or undesirable for a given input. This data is much more abundant, cheaper, and faster to collect in the real world than preferences, making it easier to scale alignment in production environments and rapidly iterate on models.

In our experiments, we find that:

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-02.jpg?height=542&width=1250&top_left_y=236&top_left_x=405)

Figure 1. The traditional pipeline for LLM alignment starts with supervised finetuning, followed by fitting the LLM to paired preference data using a method such as RLHF or DPO. However, the paired preferences that existing approaches need are hard-to-get. KahnemanTversky Optimization (KTO) only needs to know whether a given output is (un)desirable for the input, giving it access to a source of data that is much more abundant, cheaper, and faster to collect in the real world.

- KTO matches or exceeds DPO performance at scales from 1B to 30B parameters. ${ }^{1}$ That is, taking a preference dataset of $n$ DPO pairs and breaking it up into $2 n$ examples for KTO can yield better generations, despite the model ostensibly learning from a weaker signal. We provide some theoretical explanations for this phenomenon $(\$ 4.3)$.
- KTO can handle extreme data imbalances, matching DPO performance while using up to $90 \%$ fewer desirable examples (i.e., examples of good generations). Its success thus cannot be ascribed to the alignment data being sourced from a preference dataset.
- When the pretrained model is sufficiently good, one can skip supervised finetuning and go straight to KTO without a loss in generation quality. In contrast, we find that without doing SFT first, DPO-aligned models are significantly worse at all scales.

The fact that KTO can match and sometimes even outperform DPO is surprising, given that it learns from a weaker signal. We conclude by discussing some theoretical explanations for this phenomenon.

## 2. Background

Feedback-aligned LLMs are traditionally trained in three stages (Ouyang et al., 2022):

Pretraining Given a large corpus, train the model to predict the next token conditioned on the preceding text using the cross-entropy loss. Let $\pi$ denote the pretrained model.[^1]

Supervised Finetuning Finetune the model to predict the next token on data that is more relevant to the downstream task. Often, such data will comprise instructions and an appropriate response (i.e., instruction finetuning). Let $\pi_{\text {ref }}$ denote the finetuned model.

RLHF Given a dataset $\mathcal{D}$ of preferences $\left(x, y_{w}, y_{l}\right)$ where $x$ is an input, $y_{w}, y_{l}$ are the preferred and dispreferred outputs (i.e., $y_{w} \succ y_{l}$ for $x$ ), and $r^{*}$ is the "true" reward function underlying the preferences-it is first assumed that the probability that $y_{w}$ is preferred to $y_{l}$ can be captured with a specific function class, typically a Bradley-Terry model (Bradley \& Terry, 1952). Where $\sigma$ is the logistic function:

$$
\begin{equation*}
p^{*}\left(y_{w} \succ y_{l} \mid x\right)=\sigma\left(r^{*}\left(x, y_{w}\right)-r^{*}\left(x, y_{l}\right)\right) \tag{1}
\end{equation*}
$$

Since getting the true reward from a human would be intractably expensive, a reward model $r_{\phi}$ learns to serve as a proxy, done by minimizing the negative log-likelihood of the human preference data:

$$
\mathcal{L}_{R}\left(r_{\phi}\right)=\mathbb{E}_{x, y_{w}, y_{l} \sim D}\left[-\log \sigma\left(r_{\phi}\left(x, y_{w}\right)-r_{\phi}\left(x, y_{l}\right)\right)\right]
$$

But solely maximizing the reward might come at the expense of desiderata such as generating grammatical text. To avoid this, a KL divergence penalty is introduced to restrict how far the language model can drift from $\pi_{\text {ref. }}$. Where $\pi_{\theta}$ is the model we are optimizing, the optimal model $\pi^{*}$ is that which maximizes

$$
\begin{equation*}
\mathbb{E}_{x \in D, y \in \pi_{\theta}}\left[r_{\phi}(x, y)\right]-\beta D_{\mathrm{KL}}\left(\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right) \tag{2}
\end{equation*}
$$

where $\beta>0$ is a hyperparameter. Since this objective is not differentiable, we need to use an RL algorithm like PPO (Schulman et al., 2017).

However, RLHF is often slow (largely because of having to sample generations) and quite unstable in practice (especially in a distributed setting). For this reason, recent work has focused on designing closed-form losses that maximize the margin between the preferred and dispreferred generations, such as Sequence-Likelihood Calibration (SLiC) (Zhao et al., 2023) and Direct Preference Optimization (DPO) (Rafailov et al., 2023). The latter has become popular due to its mathematical equivalence with RLHF:

$$
\begin{align*}
& \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)= \\
& \mathbb{E}\left[-\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{l} \mid x\right)}\right)\right] \tag{3}
\end{align*}
$$

## 3. A Prospect Theoretic View of Alignment

Kahneman \& Tversky's prospect theory explains why, faced with an uncertain event, humans make decisions that do not maximize the expected value (1992). For example, because humans are loss-averse, given a gamble that returns $\$ 100$ with $80 \%$ probability and $\$ 0$ with $20 \%$ probability, a person might accept $\$ 60$ to avoid the gamble, despite their certainty equivalent of $\$ 60$ being less than the expected value of $\$ 80$.

### 3.1. Prospect Theory

In prospect theory, human utility depends on a value function and a weighting function: ${ }^{2}$

Definition 3.1. A value function $v: z \rightarrow \mathbb{R}$ maps an outcome $z$, relative to some reference point $z_{\text {ref }}$, to its perceived (or subjective) value. For example, these functions capture the fact that humans tend to be more sensitive to relative losses than relative gains of the same magnitude.

Definition 3.2. A weighting function $w$ is the derivative of a capacity function that maps cumulative probabilities to perceived cumulative probabilities. These functions capture, for example, the fact that humans tend to overestimate the chance of rare events. Let $w_{z}$ denote the weight placed on outcome $z$.

Definition 3.3. The utility of a random variable $Z$ is a function of its outcomes: $u(Z) \triangleq \sum_{z \in Z} w_{z} v\left(z-z_{\text {ref }}\right)$.

However, because humans do not see the full probability distribution of an LLM, weighting functions are not salient to this discussion; we will focus only on value functions. Using experiments that presented real humans with monetary gambles and asked for their certainty equivalent, Tversky \& Kahneman (1992) proposed the following functional form for human value:

$$
v\left(z, z_{\mathrm{ref}} ; \lambda ; \alpha\right)= \begin{cases}\left(z-z_{\mathrm{ref}}\right)^{\alpha} & \text { if } z>z_{\mathrm{ref}}  \tag{4}\\ -\lambda\left(z_{\mathrm{ref}}-z\right)^{\alpha} & \text { if } z<z_{\mathrm{ref}}\end{cases}
$$[^2]

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-03.jpg?height=588&width=723&top_left_y=232&top_left_x=1102)

Figure 2. The utility that a human gets from the outcome of a random variable, as imputed by the value function implicit in HALOs. Notice that the imputed functions share properties such as loss aversion with the human value functions that Kahneman \& Tversky empirically derived (1992).

where the median value of hyperparameter $\alpha=0.88$ and $\lambda=2.25$ across individuals. $\alpha$ controls how quickly utility changes and $\lambda$ controls the degree of loss aversion. While the shape of the median Kahneman-Tversky value function is illustrated in Figure 2, it should be noted that it varies across individuals (Tversky \& Kahneman, 1992). There are also other functional forms for the value function that have been proposed in later work (Gurevich et al., 2009). The salient qualities of a value function are: the existence of a reference point that is added or subtracted to get the relative gain or loss respectively; concavity in relative gains (i.e. diminishing sensitivity away from $z_{\text {ref }}$ ); loss aversion (i.e., greater sensitivity to losses).

### 3.2. HALOs

Informally, HALOs are loss functions that model the human biases in Tversky \& Kahneman (1992). Formally,

Definition 3.4 (HALOs). Let $x \in \mathcal{X}$ denote an input and $y \in \mathcal{Y}$ an output. Then $f:(x, y) \rightarrow \mathbb{R}$ is a human-aware loss function if there exists the following: a parameterized reward function $r_{\theta}$ such that $\forall\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \mathcal{X} \times \mathcal{Y}$,

$$
r_{\theta}\left(x_{1}, y_{1}\right)>r_{\theta}\left(x_{2}, y_{2}\right) \Longleftrightarrow\left(x_{1}, y_{1}\right) \succ_{r_{\theta}}\left(x_{2}, y_{2}\right)
$$

reference point distributions $Q_{x}\left(X^{\prime}\right), Q_{y}\left(Y^{\prime} \mid X^{\prime}\right)$, a value function $v_{f}: \mathbb{R} \rightarrow \mathbb{R}$ that is monotonic non-decreasing and concave in $(0, \infty)$, and a negative affine function $t$ such that

$$
\begin{equation*}
f(x, y ; \theta)=t\left(v_{f}\left(r_{\theta}(x, y)-\mathbb{E}_{x^{\prime}, y^{\prime}}\left[r_{\theta}\left(x^{\prime}, y^{\prime}\right)\right]\right)\right) \tag{5}
\end{equation*}
$$

where $x^{\prime} \sim Q_{x}\left(X^{\prime}\right)$ and $y^{\prime} \sim Q_{y}\left(Y^{\prime} \mid x^{\prime}\right)$.

Put simply, the requirement for the reward function is that it assigns higher rewards to input-output pairs that are more
preferred under it. The reference point is the expected reward with respect to input-output pairs sampled from the distributions $Q_{x}, Q_{y}$. We require that the value function be concave in gains but not necessarily convex in lossesunlike the canonical Kahneman-Tversky value functionsbecause in the original work on prospect theory, a minority of individuals were found to be risk-averse in both the gain and loss regime (i.e., concave in both gains and losses) (Kahneman \& Tversky, 1979). Note that risk-aversion is different from loss-aversion; they relate to the curvature and magnitude of the slope respectively.

Proposition 3.5. DPO, SLiC (calibration loss only), and PPO-Clip are human-aware loss functions.

The proof is deferred to Appendix A. In Figure 2, we can see this more intuitively by plotting the value function for each loss (i.e., the implied human utility). We see that the value functions of all three losses incorporate a sense of loss aversion, although this is not needed to meet the definition of a HALO, since there are individuals and scenarios for which loss aversion does not necessarily apply. The value functions are also either concave or affine (depending on the interval), unlike the standard Kahneman-Tversky value function, which is concave in gains but convex in losses. The reference point distributions used also differs across the losses.

### 3.3. Does being a HALO matter?

A natural question is whether the modeling of human biases in HALOs has practical benefits. This is difficult to answer, since both HALOs and non-HALOs are diverse function classes, but we attempt to do so by comparing popular nonHALO and HALO baselines on the exact same data:

1. CSFT: Conditional SFT is a simple alignment method where a control token is prepended to the output during training; then, at inference, the control token corresponding to desirable generations (e.g., <|good $\mid>$ ) is appended to the input to induce good generations (Korbak et al., 2023). This is a non-HALO loss.
2. SLiC: SLiC with a regularization penalty $\left(\lambda_{\text {reg }} \neq 0\right)$ is a non-HALO loss:

$$
\begin{aligned}
& \mathcal{L}_{\mathrm{SLiC}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=\mathcal{L}_{\mathrm{cal}}\left(\pi_{\theta}\right)+\lambda_{\mathrm{reg}} L_{\mathrm{reg}}\left(\pi_{\theta}\right) \\
& \mathcal{L}_{\mathrm{cal}}=\mathbb{E}_{x, y_{w}, y_{l} \sim D}\left[\max \left(0, \delta-\log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\theta}\left(y_{l} \mid x\right)}\right)\right] \\
& \mathcal{L}_{\mathrm{reg}}=\mathbb{E}_{x \sim D, y \sim \pi_{\mathrm{ref}}(x)}\left[-\log \pi_{\theta}(y \mid x)\right]
\end{aligned}
$$

Although the max-margin loss $\mathcal{L}_{\text {cal }}$ is a HALO on its own (Proposition 3.5), the complete loss is not, since the $\mathcal{L}_{\text {reg }}$ term is the standard language modeling loss.

3. DPO: DPO, as defined in (3), is a HALO loss (Proposition 3.5).
4. PPO (offline): The standard RLHF objective in (2) is typically optimized with PPO-Clip, which works by "clipping" how far $\pi_{\theta}$ can drift from the version $\pi_{\text {old }}$ at the previous step:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{PPO} \text { (offline) }}= & -\mathbb{E}_{x, y, t \sim D}\left[\operatorname { m i n } \left(q_{\theta} A\left(x, y_{<t}, y_{t}\right)\right.\right. \\
& \left.\left.\operatorname{clip}\left(q_{\theta}, 1-\epsilon, 1+\epsilon\right) A\left(x, y_{<t}, y_{t}\right)\right)\right]
\end{aligned}
$$

where $q_{\theta}=\log \frac{\pi_{\theta}}{\pi_{\text {old }}}$ and $A\left(x, y_{<t}, y_{t}\right)$ is the per-token advantage (i.e., the surplus benefit from producing a given token in a given state).

PPO is an online algorithm-generations are sampled from the current model, judged by a reward model, and then used to update the current version. However, this process is slow (due to having to sample generations), so we choose to use offline data instead. Because RLHF is also quite unstable in a distributed setting, we never update $\pi_{\text {old }}$ and keep it as $\pi_{\text {ref }}$, instead clipping less conservatively than we traditionally would (see Appendix B for details). Baheti et al. (2023) found that these changes, along with treating the entire output sequence as a single action, greatly improves stability. However, since RLHF has historically calculated tokenlevel advantages, we omit the third change and only preserve the first two. The PPO-Clip loss itself is left unchanged and is therefore a HALO (Proposition 3.5).

Calling this method PPO is somewhat imprecise, because it is offline and takes only one step, but to avoid introducing too many new terms, we will call this $P P O$ (offline). Instead of using learned rewards, we simplify even further and use dummy $+1 /-1$ rewards for $y_{w}$ and $y_{l}$ instead. Further details on the implementation of this method can be found in Appendix B.

We compare these baselines on a suite of 7 models spanning two model families, Pythia- $\{1.4 \mathrm{~B}, 2.8 \mathrm{~B}, 6.9 \mathrm{~B}, 12 \mathrm{~B}\}$ (Biderman et al., 2023) and Llama-\{7B, 13B, 30B \} (Touvron et al., 2023). This permits us to see how LLM alignment scales within a model family (Llama-2 lacks a 30B model, hence our use of Llama). Later experiments ( $\$ 4.2$ ) are done on Mistral-7B and its derivatives (Jiang et al., 2023). The models were trained on a combination of Anthropic HH (Ganguli et al., 2022), OpenAssistant (Köpf et al., 2023), and SHP (Ethayarajh et al., 2022).

All models were aligned under identical settings on the same data (e.g., same effective batch size, same optimizer, etc.), save for hyperparameters unique to them. Similar to Rafailov et al. (2023), the target sequences for SFT are a subset of the generations used to subsequently align the model; however, for a more realistic SFT setup, we do not necessarily set the most preferred generation to be the target (with the exception of $\mathrm{HH}$, since the dispreferred output in that dataset is often harmful). Then we used GPT-4-0613

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-05.jpg?height=564&width=1702&top_left_y=233&top_left_x=187)

Does the aligned model beat the SFT target?

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-05.jpg?height=518&width=1678&top_left_y=272&top_left_x=191)

Figure 3. Among existing alignment methods, the HALOs (DPO and our offline PPO variant) generally outperform non-HALOs (SLiC and CSFT), though the gap is only significant $(p<0.05)$ at 13B+ model sizes. In fact, only the HALO-aligned Llama-\{13B, 30B $\}$ models are able to match or exceed the generation quality of SFT target sequences, which are drawn directly from the alignment dataset. It is also worth noting that up to a scale of 7B parameters, virtually all of the gains from LLM alignment come from the SFT stage.

to judge whether the aligned model's response was better than the SFT target for the given input with respect to helpfulness, harmlessness, and conciseness, a now standard practice (Zheng et al., 2023; Li et al., 2023). ${ }^{3}$ Note that while the SFT target is considered a desirable output for $x$, it is by no means the best output, meaning that it can be improved upon by an aligned model.

In Figure 3, we see the results of this analysis:
- The HALOs we tested (DPO and our PPO variant) either match or outperform the non-HALOs at all scales, though the gap is only significant $(p<0.05)$ at 13B+ model sizes. In fact, only the HALO-aligned Llama\{13B, 30B \} models match or exceed a win rate of $50 \%$ (i.e., are able to match or exceed the generation quality of the SFT targets in the test data).
- Up to a scale of 7B parameters, alignment provides virtually no gains over SFT alone. However, it is worth noting that if the SFT data distribution were less similar to the preference data, then the gains from the alignment stage would ostensibly be greater.
- Surprisingly, despite only using dummy $+1 /-1$ rewards, our offline PPO variant performs as well as DPO for all models except Llama30B. This challenges conventional wisdom, which places heavy emphasis on reward learning (Casper et al., 2023), suggesting that even the simplest rewards can prove useful when used in a loss function that has a strong inductive bias. Despite its surprising success, our offline PPO baseline still suffers from hyperparameter sensitivity and training instability,[^3]

albeit not to the same extent as traditional RLHF.

## 4. Kahneman-Tversky Optimization

The surprising success of offline PPO with dummy +1/-1 rewards suggests that — with the right HALO—a binary signal of good/bad generations may be sufficient to reach DPOlevel performance, even if the offline PPO approach itself was unable to do so past a certain scale ( $\$ 3.3$ ). Taking a more principled approach, we now derive a HALO using the Kahneman-Tversky model of human utility, which allows us to directly optimize for utility instead of maximizing the log-likelihood of preferences. This Kahneman-Tversky Optimization (KTO) loss only needs a binary signal of whether an output is (un)desirable for a given input, giving it access to a source of data is more abundant, cheaper, and faster to collect in the real world.

### 4.1. Derivation

From prior work (Go et al., 2023; Peng et al., 2019; Peters \& Schaal, 2007), we know that the policy that maximizes the KL-constrained RLHF objective in (2) is

$$
\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r^{*}(x, y)\right)
$$

where $Z(x)$ is a partition function. Rafailov et al. (2023) rewrite this in terms of the optimal reward for an inputoutput pair:

$$
\begin{equation*}
r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x) \tag{6}
\end{equation*}
$$

They then plug this expression into the Bradley-Terry model of preferences and take the negative logarithm of that objective to get the DPO loss (3).

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-06.jpg?height=562&width=1699&top_left_y=234&top_left_x=186)

Does the aligned model beat the SFT target?

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-06.jpg?height=525&width=1680&top_left_y=266&top_left_x=190)

Figure 4. Kahneman-Tversky Optimization (KTO) is as good or better than DPO at all scales, both when preceded and not preceded by supervised finetuning (SFT). In fact, for the Llama models, KTO alone matches the performance of SFT+DPO and is significantly better than DPO alone. Error bars denote a $90 \%$ binomial confidence interval.

Instead, we plug this expression into the Kahneman-Tversky model of human utility, with some changes to make it more amenable to the LLM setting:

1. The exponent in the Kahneman-Tversky value function (4) makes it difficult to optimize, so we set $v_{\text {KTO }}$ to be the logistic function $\sigma$, which is also concave in gains and convex in losses. We replace the lossaversion coefficient with two hyperparameters $\lambda_{D}, \lambda_{U}$ that weight the losses for desirable and undesirable outputs respectively.
2. The Kahneman-Tversky value function was derived based on experiments with humans and monetary gambles. Since LLM generations do not have a monetary reward associated with them, we set $r_{\text {KTO }}$ to be the implicit reward under the RLHF objective (6).
3. Rather than having just one dispreferred generation $y_{l} \mid x$ as the reference point, we assume that humans judge the quality of $(x, y)$ in relation to all input-output pairs they have seen. Thus we write the reference point to be the expected reward under the optimal policy, not just for generations following $x$ but following any input $x^{\prime}: \mathbb{E}_{x^{\prime} \sim D, y^{\prime} \sim \pi^{*}}\left[r^{*}\left(x^{\prime}, y^{\prime}\right)\right]$. Under the assumption that the expected value of the partition function across $x^{\prime}$ is zero, this simplifies to the KL divergence between $\pi^{*}$ and $\pi_{\text {ref }}$ scaled by $\beta$.

Combining all of these changes, we can optimize the following loss, where the notion of an output being "desirable" or "undesirable" corresponds to the Kahneman-Tversky notion of a relative gain or loss.

$$
\begin{equation*}
L_{\mathrm{KTO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=\mathbb{E}_{x, y \sim D}\left[w(y)\left(1-v_{\mathrm{KTO}}(x, y ; \beta)\right)\right] \tag{7}
\end{equation*}
$$

where

$$
\begin{aligned}
r_{\mathrm{KTO}}(x, y) & =\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \\
z_{\mathrm{ref}} & =\mathbb{E}_{x^{\prime} \sim D}\left[\beta \operatorname{KL}\left(\pi_{\theta}\left(y^{\prime} \mid x^{\prime}\right) \| \pi_{\mathrm{ref}}\left(y^{\prime} \mid x^{\prime}\right)\right)\right] \\
v_{\mathrm{KTO}}(x, y ; \beta) & =\left\{\begin{array}{l}
\sigma\left(r_{\mathrm{KTO}}(x, y)-z_{\text {ref }}\right) \text { if } y \sim y_{\text {desirable }} \mid x \\
\sigma\left(z_{\text {ref }}-r_{\mathrm{KTO}}(x, y)\right) \text { if } y \sim y_{\text {undesirable }} \mid x
\end{array}\right. \\
w(y) & = \begin{cases}\lambda_{D} & \text { if } y \sim y_{\text {desirable }} \mid x \\
\lambda_{U} & \text { if } y \sim y_{\text {undesirable }} \mid x\end{cases}
\end{aligned}
$$

Intuitively, KTO works because if the model increases the reward of a desirable example in a generic way, then the KL penalty will also rise and no progress will be made on the loss. This forces the model to learn exactly what makes an output desirable, so that the reward can be increased while keeping the KL term flat (or even decreasing it). A similar argument works in the other direction as well, though the non-negativity of the KL term allows faster saturation.

Implementation In practice, we estimate the KL term by matching inputs $x^{\prime}$ with unrelated outputs $y_{U}^{\prime}$ in a batch of size $m$ and then calculating max $\left(0, \frac{1}{m} \sum \log \frac{\pi_{\theta}\left(y_{U}^{\prime} \mid x^{\prime}\right)}{\pi_{\text {ref }}\left(y_{U}^{\prime} \mid x^{\prime}\right)}\right)$ over the entire batch. We do not back-propagate through the $\mathrm{KL}$ term, as it makes training much more stable. This means that the KL term purely serves to control how saturated the loss is.

$\beta$ has the same meaning as in DPO; the lower it is, the less we penalize $\pi_{\theta}$ from moving away from the SFT model $\pi_{\text {ref }}$. We find that $\beta=0.1$ is close-to-best on most datasets. Where $n_{D}$ and $n_{U}$ refer to the number of desirable and undesirable examples respectively, we set $\lambda_{D}, \lambda_{U}$ such that

$$
\begin{equation*}
\frac{\lambda_{D} n_{D}}{\lambda_{U} n_{U}} \in\left[1, \frac{4}{3}\right] \tag{8}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-07.jpg?height=409&width=813&top_left_y=232&top_left_x=190)

Figure 5. Without doing SFT first, DPO-aligned models tend to ramble and hallucinate entire conversations. KTO does not suffer from this issue.

where at least one of the two should be set to 1 and the ratio is controlled by changing the other. For example, if there is a 1:1 ratio of desirable:undesirable examples, we would set $\lambda_{U}=1, \lambda_{D} \in[1,1.33]$. If we then discard $90 \%$ of the desirable examples and only keep $10 \%$, then we would set $\lambda_{U}=1, \lambda_{D} \in[10,13.33]$. The interval $[1,4 / 3]$ was determined empirically and suggests a value function that is more gain-sensitive than loss-sensitive, in contrast to the original Kahneman-Tversky value function (4). However, the ideal interval is also task-dependent; for example, if avoiding negative outcomes were very important, then we might consider a setting of $\lambda_{U}>1$ instead.

Data If the alignment data is naturally binary, every positive example can be assumed to be drawn from $y_{\text {desirable }} \mid x$ and every negative example from $y_{\text {undesirable }} \mid x$. However, the canonical feedback datasets in academic research $(\mathrm{HH}, \mathrm{SHP}$, OASST) are in preference format, since the methods that have worked best up until now are preference-based. In our experiments, we converted preference data $y_{w} \succ y_{l}$ by assuming that $y_{w}$ is drawn from the desirable distribution and $y_{l}$ from the undesirable one. To enable an apples-to-apples comparison with DPO, we apply KTO on the same data for most experiments. However, to ensure that KTO can be used with non-preference data, we also subsample one output $y$ per $x$ for some experiments (denoted one- $y$-per- $x$ ).

If the data is score-based, where a higher score denotes greater desirability, one has multiple options:

- Assume that any output with a score above some fixed threshold $\tau$ is desirable.
- Assume that any output with a score above the mean or median (either across all inputs or just the input it was conditioned on) is desirable.
- Let desirability be a Bernoulli random variable where $p\left(y \sim y_{\text {desirable }} \mid x\right)$ is some function of its score (e.g., logistic). Then randomly sample to determine whether $y$ is desirable or not.

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-07.jpg?height=626&width=791&top_left_y=237&top_left_x=1079)

Figure 6. Even after discarding $90 \%$ of the desirable examples while keeping all of the undesirable data (leading to a 1:10 ratio of desirable:undesirable data), a KTO-aligned Llama-7B model still outperforms its DPO counterpart. This implies that preference pairs do not have to be the source of KTO data.

### 4.2. Experiments

KTO $\geq$ DPO As seen in Figure 4, SFT+KTO is competitive with SFT+DPO at model scales from 1B to 30B, despite learning from a weaker signal. KTO alone is better than DPO alone for the Llama-\{7B, 13B, 30B \} models, and this gap is significant $(p<0.01)$ at $7 \mathrm{~B}$ and 30B even after correcting for multiple comparisons (Holm, 1979). Perhaps most surprising is the fact that a KTO-aligned Llama- $\{13 \mathrm{~B}$, 30B $\}$ model is competitive with its SFT+KTO counterpart, despite not undergoing supervised finetuning first, and is the only alignment method of the ones we tested to show this behavior. This is perhaps due to the fact that KTO keeps the average response length roughly the same as it is for the SFT model. In contrast, doing DPO without SFT first causes the average response length to increase dramatically.

KTO data need not come from preference datasets. Might KTO be secretly benefiting from the fact that its $2 n$ examples in the previous experiment came from $n$ preference pairs instead of a naturally unpaired data distribution? To test this, we randomly discard increasingly large fractions of the desirable data before KTO-aligning a Llama-7B model. For example, if we discard $90 \%$ of the desirable data while leaving the undesirable data untouched, then the ratio of desirable:undesirable examples goes from 1:1 to 1:10 and the vast majority of examples no longer have a preferred output counterpart. We handle such imbalances by changing the loss weights $\lambda_{D}, \lambda_{U}$ to satisfy the criteria in (8); when we drop $90 \%$ of the desirable data, we set $\lambda_{u}=1, \lambda_{D}=13.33$. The full results are given in Figure 6. For Llama-7b, we find that up to $90 \%$ of the desirable

Table 1. In aligning Mistral-7B on the OpenAssistant dataset, we find that using KTO with only one output per input still outperforms DPO, despite this restriction reducing the amount of training data by $72 \%$. A $90 \%$ confidence interval is given.

| Method | Winrate vs. SFT Target |
| :--- | :---: |
| Mistral-7B (unaligned) | $0.525 \pm 0.037$ |
| Mistral-7B + DPO | $0.600 \pm 0.037$ |
| Mistral-7B + KTO (all $y$ per $x$ ) | $\mathbf{0 . 6 5 2} \pm \mathbf{0 . 0 3 6}$ |
| Mistral-7B + KTO (one $y$ per $x$ ) | $0.631 \pm 0.036$ |
| Mistral-7B-Instruct | $0.621 \pm 0.031$ |

data can in fact be discarded while still outperforming DPO. A similar trend holds when discarding undesirable data. For different models and datasets, the optimal settings of $\lambda_{D}, \lambda_{U}$ differ.

We further verify this claim by aligning Mistral-7B on OpenAssistant using DPO (on $n$ pairs), standard KTO (on all $2 n$ outputs), and KTO where only one $y$ per $x$ is used. Since the output of one $y$ in OpenAssistant is not conditioned on the other outputs for the same input, the latter effectively captures the setting where the data is from an inherently unpaired distribution. Despite the one- $y$-per- $x$ setup decreasing the amount of training data by $72 \%$, the KTO-aligned model still outperforms both its DPO counterpart and the official instruction-tuned version of Mistral-7B (Jiang et al., 2023), as seen in Table 1.

On average, KTO improves performance across generative benchmarks. Zephyr- $\beta$ is a variant of Mistral-7B that has been instruction-tuned and DPO-aligned on the UltraFeedback dataset (Tunstall et al., 2023; Cui et al., 2023). We find that substituting KTO for DPO (and changing nothing else) improves performance across MMLU (0-shot) (Hendrycks et al., 2021), GSM8K (8-shot, CoT) (Cobbe et al., 2021), HumanEval (0-shot) (Chen et al., 2021), and BigBench-Hard (3-shot CoT) (Srivastava et al., 2022). On GSM8K, just swapping DPO for KTO improves performance by 13.5 points. Even when we align with KTO using only one $y$ per $x$ (i.e., reducing the data volume by half), we still outperform DPO on all but one benchmark.

### 4.3. Theoretical Analysis

KTO was designed with the motivation that even if it had to learn from a weaker signal, it would make up for this limitation with the fact that it has access to much more data in the real world, where thumbs-up/thumbs-down data is common but preferences are scarce and expensive to collect. So why does KTO perform as good or better than DPO in our experiments, when it sees the same amount of data? Data efficiency may not be the only answer. Our theoretical analysis suggests that preference likelihood can
Table 2. Aligning Zephyr (Tunstall et al., 2023), a derivative of Mistral-7B, on UltraFeedback with KTO instead of DPO improves results across a suite of benchmarks. This is true even when only one of the two outputs in each preference is seen by KTO, despite this reducing the volume of data by half (one- $y$-per- $x$ ).

| Dataset $(\rightarrow)$ | MMLU | GSM8k <br> EM | HumanEval <br> pass@ 1 | BBH <br> EM |
| :--- | :--- | :--- | :--- | :--- |
| Metric $(\rightarrow)$ | EM | E |  |  |
| Zephyr- $\beta$ SFT | 57.2 | 39.0 | 30.1 | 46.3 |
| +DPO | 58.2 | 40.0 | 30.1 | 44.1 |
| +KTO | $\mathbf{5 8 . 6}$ | $\mathbf{5 3 . 5}$ | $\mathbf{3 0 . 9}$ | $\mathbf{5 2 . 6}$ |
| +KTO (one- $y$-per- $x$ ) | 58.0 | 50.0 | 30.7 | 49.9 |

be maximized without necessarily maximizing underlying human utility and that KTO implicitly ignores noisy and intransitive data.

Proposition 4.1. KTO does not learn from undesirable examples with sufficiently high rewards or desirable examples with sufficiently low rewards.

Informally, if an example is too difficult to learn from, then the KTO update will not change $\pi_{\theta}$. This may be a blessing in disguise, since human preferences are often noisy and not every given preference can be recovered with the true reward $r^{*}$ (Hoeffler \& Ariely, 1999). This means that it may be useful to avoid unlearnable preferences. However, this is a double-edged sword: it also means that KTO could end up ignoring some data that is hard-to-learn but necessary to recover $r^{*}$, resulting in under-fitting.

Theorem 4.2. Assuming the value function is logistic, for any bounded reward function $r_{a}$, there exists a reward function in its equivalence class (i.e., $r_{b}(x, y)=r_{a}(x, y)+h(x)$ for some $h(x)$ ) that induces the same optimal policy $\pi^{*}$ and Bradley-Terry preference distribution but a different human value distribution.

A key insight from Rafailov et al. (2023) is that reward functions in the same equivalence class (i.e., differing only in an input-specific component) induce the same optimal policy under (2) and the same Bradley-Terry preference distribution. However, we show under mild assumptions that the value distribution-i.e., human utility-is affected by such input-specific changes, so maximizing preference likelihood does not mean one is maximizing human utility. Approaches that directly maximize utility, such as KTO, may thus perform better in open-ended evaluation.

Theorem 4.3. Let two humans $a, b$ have value functions $v_{a}, v_{b}$ and contradicting preferences $y_{1} \succ_{a} y_{2}$ and $y_{2} \succ_{b} y_{1}$ for some input $x$. Assume $\pi_{\text {ref }}(y \mid x)=0 \Longrightarrow \pi_{\theta}(y \mid x)=0$ for all $x, y$. In the worst-case, the optimal policy under DPO decreases the expected value of both humans. In contrast, if each preference is broken up into two examples, then KTO (with default settings) does not change the policy.

Informally, we assume that humans want the model to in-
crease and decrease the probability of generations they like and dislike respectively. However, the preferences of two humans often contradict, leading to a dataset containing intransitive preferences. In the worst-case, DPO allows one of the two preferences to be recovered while decreasing the expected value of both humans. In contrast, KTO will change nothing at all in any case. Since existing datasets contain preferences from multiple annotators, the existence of intransitivity may help explain why KTO works better.

### 4.4. KTO vs. DPO - when to use which?

When human feedback is in a binary format, and especially when there is an imbalance between the number of desirable and undesirable examples, KTO is the natural choice. When your data is in the form of preferences, the choice is less clear. Putting aside the greater data efficiency of KTO, our theoretical analysis suggests that if your preference data has sufficiently little noise and sufficiently little intransitivity, then DPO will work better, since there is some risk of KTO underfitting. But if there is enough noise and transitivity, then the better worst-case guarantees of KTO will win out. Most publicly available preference datasets (e.g., SHP, OpenAssistant) contain noisy feedback from many different humans whose preferences likely contradict, which explains why KTO was able to match or exceed DPO performance in our experiments. Even AI feedback can be noisy and intransitive, which helps explain why KTO outperforms DPO when aligning with the synthetic UltraFeedback data.

## 5. Related Work

Human feedback has been used to improve LLM capabilities in translation (Kreutzer et al., 2018), summarization (Stiennon et al., 2020), sentiment-conditioned generation (Ziegler et al., 2019), and instruction-following (Ouyang et al., 2022). The RLHF framework (Christiano et al., 2017; Bai et al., 2022) traditionally used to accomplish this is detailed in $\S 2$.

Still, momentum has largely shifted in favor of closed-form losses that directly operate on offline preferences, such as DPO (Rafailov et al., 2023). This single stage of optimization distinguishes DPO from the conventional approach in preference-based RL, which learns a reward and then fits the policy to those rewards (Jain et al., 2013; Busa-Fekete et al., 2014). A recent string of work has centered on the idea of "self-training" or "self-play", during which new preference data is inferred from a model's generations (Chen et al., 2024; Yuan et al., 2024). Despite not being a human-aware loss, unlikelihood training was among to first to methods to align language models using a binary signal (Welleck et al., 2019). However, work by Korbak et al. (2023) found that it is worse than the CSFT baseline we tested in our work.

Prospect theory, despite being highly influential in behav- ioral economics, has had a fairly muted impact in machine learning, with work concentrated in human-robot interaction (Kwon et al., 2020; Sun et al., 2019; Chan et al., 2021). Learning from sparse binary feedback is a staple of information retrieval and recommender systems (He et al., 2017; Koren et al., 2009), although to our knowledge it has not been used to generate open-ended text.

## 6. Future Work

The existence of HALOs raises many questions. For one, the KTO loss is based on the Kahneman-Tversky value function for monetary gains and losses, which is almost certainly different from how humans perceive the relative goodness of text. What value function-and corresponding HALO - best describes how humans perceive language?

Given that the data that KTO needs is much more abundant, cheaper, and faster to collect-both as human and AI feedback-how far can we push synthetic data? For example, if we wanted to create a toxicity dataset to align our models to be less toxic, creating a tuple $\left(x, y_{w}, y_{l}\right)$ where $y_{l}$ is more toxic than $y_{w}$ is non-trivial. However, with KTO, we can easily create a dataset where desirability is determined by some black-box toxicity detection API. What other kinds of desiderata can we synthetically optimize for with KTO? Can we convert signals like "conversation lead to sale made" or "support ticket resolved" into KTO data?

Currently, KTO can learn from score-based data when the score is used to infer desirability. However, can we design a HALO where scores are directly incorporated into this loss?

## 7. Conclusion

We proposed a class of functions called human-aware losses (HALOs) based on the idea of a Kahneman-Tversky value function, which models some of the key cognitive biases that inform how humans make decisions about uncertain outcomes. We showed that among existing alignment methods, those that met the definition of a HALO performed better than those that did not, suggesting a benefit to the modeling of human biases. We then designed a human-aware loss called KTO for directly maximizing the utility of generations instead of maximizing preference likelihood. Despite only learning from a binary signal of whether an output is (un)desirable, KTO is as good or better than DPO at scales from 1B to 30B. Still, we make no claims that KTO is the best HALO for all scenarios; there remains much work to be done in discovering the optimal human-aware for each setting.

## Acknowledgements

We thank Dilip Arumugam and Arya McCarthy for feedback on the paper and Nathan Lambert for feedback on an early version of this draft. We thank Stas Bekman and Gautam Mittal for cluster assistance and Alex Manthey for helping with human evaluation.

## References

Baheti, A., Lu, X., Brahman, F., Bras, R. L., Sap, M., and Riedl, M. Improving language models with advantage-based offline policy gradients. arXiv preprint arXiv:2305.14718, 2023.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397-2430. PMLR, 2023.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324-345, 1952.

Busa-Fekete, R., Szörényi, B., Weng, P., Cheng, W., and Hüllermeier, E. Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine learning, 97:327-351, 2014.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Chan, L., Critch, A., and Dragan, A. Human irrationality: both bad and good for reward inference. arXiv preprint arXiv:2111.06956, 2021.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z., and Sun, M. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

Ethayarajh, K., Choi, Y., and Swayamdipta, S. Understanding dataset difficulty with $\mathcal{V}$-usable information. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 5988-6008. PMLR, 17-23 Jul 2022.

Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.

Go, D., Korbak, T., Kruszewski, G., Rozen, J., Ryu, N., and Dymetman, M. Aligning language models with preferences through f-divergence minimization. arXiv preprint arXiv:2302.08215, 2023.

Gurevich, G., Kliger, D., and Levy, O. Decision-making under uncertainty-a field study of cumulative prospect theory. Journal of Banking \& Finance, 33(7):1221-1229, 2009.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pp. $173-182,2017$.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Hoeffler, S. and Ariely, D. Constructing stable preferences: A look into dimensions of experience and their impact on preference stability. Journal of consumer psychology, 8 $(2): 113-139,1999$.

Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pp. 65-70, 1979.

Jain, A., Wojcik, B., Joachims, T., and Saxena, A. Learning trajectory preferences for manipulators via iterative improvement. Advances in neural information processing systems, 26, 2013.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Kahneman, D. and Tversky, A. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263-292, 1979.

Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., et al. Openassistant conversationsdemocratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.

Korbak, T., Shi, K., Chen, A., Bhalerao, R. V., Buckley, C., Phang, J., Bowman, S. R., and Perez, E. Pretraining language models with human preferences. In International Conference on Machine Learning, pp. 1750617533. PMLR, 2023.

Koren, Y., Bell, R., and Volinsky, C. Matrix factorization techniques for recommender systems. Computer, 42(8): 30-37, 2009.

Kreutzer, J., Uyheng, J., and Riezler, S. Reliability and learnability of human bandit feedback for sequence-tosequence reinforcement learning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17771788,2018

Kwon, M., Biyik, E., Talati, A., Bhasin, K., Losey, D. P., and Sadigh, D. When humans aren't optimal: Robots that collaborate with risk-aware humans. In Proceedings of the 2020 ACM/IEEE international conference on humanrobot interaction, pp. 43-52, 2020.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/ alpaca_eval, 2023.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.
Peters, J. and Schaal, S. Reinforcement learning by rewardweighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pp. 745-750, 2007.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. $A d$ vances in Neural Information Processing Systems, 33: 3008-3021, 2020.

Sun, L., Zhan, W., Hu, Y., and Tomizuka, M. Interpretable modelling of driving behaviors in interactive driving scenarios based on cumulative prospect theory. In 2019 IEEE Intelligent Transportation Systems Conference (ITSC), pp. 4329-4335. IEEE, 2019.

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C. Fine-tuning language models for factuality. arXiv preprint arXiv:2311.08401, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., and Wolf, T. Zephyr: Direct distillation of lm alignment, 2023.

Tversky, A. and Kahneman, D. Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5:297-323, 1992.

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., and Huang, S. Trl: Transformer reinforcement learning. https://github. com/huggingface/trl, 2020.

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., and Weston, J. Neural text generation with unlikelihood
training. In International Conference on Learning Representations, 2019.

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J., and Weston, J. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 2024.

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
</end of paper 0>


<paper 1>
# InstructionCP: A fast approach to transfer Large Language Models into target language 

Kuang-Ming Chen ${ }^{1,2} \quad$ Hung-yi Lee ${ }^{1}$<br>${ }^{1}$ National Taiwan University, Taipei, Taiwan<br>${ }^{2}$ ASUS Open Cloud Infrastructure Software Center, Taipei, Taiwan<br>b08502105@ntu.edu.tw hungyilee@ntu.edu.tw


#### Abstract

The rapid development of large language models (LLMs) in recent years has largely focused on English, resulting in models that respond exclusively in English. To adapt these models to other languages, continual pre-training ( $\mathrm{CP}$ ) is often employed, followed by supervised fine-tuning (SFT) to maintain conversational abilities. However, CP and SFT can reduce a model's ability to filter harmful content. We propose Instruction Continual Pre-training (InsCP), which integrates instruction tags-also known as chat templates - into the CP process to prevent loss of conversational proficiency while acquiring new languages. Our experiments demonstrate that InsCP retains conversational and Reinforcement Learning from Human Feedback (RLHF) abilities. Empirical evaluations on language alignment, reliability, and knowledge benchmarks confirm the efficacy of InsCP. Notably, this approach requires only 0.1 billion tokens of high-quality instruction-following data, thereby reducing resource consumption.


## 1 Introduction

Large language models (LLMs) have demonstrated remarkable performance across numerous natural language processing (NLP) tasks(Brown et al., 2020). However, the majority of LLMs are pre-trained on English corpora, thus restricting their utility to English language contexts.

While some endeavors opt to train their LLMs from scratch using non-English data, as exemplified by YI-34B(AI et al., 2024), we recognize the significant time and computing resources required for such an approach. Drawing inspiration from Ouyang et al. (2022), many research groups have shifted their focus towards continual pre-training (CP)(Gupta et al., 2023; Ke et al., 2022) on target languages to enhance knowledge acquisition and model fluency. Subsequently, supervised finetuning (SFT) is conducted on instructionformatted data to ensure that models possess the capability to respond to questions in a format consistent with English-based pre-trained LLMs, such as BLOOM(Workshop et al., 2023), LLaMA2(Touvron et al., 2023), and Mistral7B(Jiang et al., 2023).

Furthermore, in an effort to align with human preferences, Reinforcement Learning from Human Feedback (RLHF) has been integrated(Ouyang et al., 2022; Ziegler et al., 2020). However, the RLHF process is intricate. Direct Preference Optimization (DPO) (Rafailov et al., 2023) relies on collecting paired data from human preferences, facilitating more stable and straightforward model tuning. Nonetheless, gathering both positive and negative responses from humans still demands substantial effort. In contrast, Kahneman-Taversky Optimization (KTO)(Ethayarajh et al., 2024) operates with unpaired data, thus easing the collection process. However, KTO has its drawbacks. The existence of HALOs (Human-Aware Loss Functions) raises several questions regarding KTO. Firstly, the KTO loss function is based on the Kahneman-Tversky value function for monetary gains and losses, which may not accurately reflect how humans perceive the relative goodness of text. Nevertheless, LLMs trained using RLHF demonstrate enhanced safety in completions, a crucial factor for companies and groups intending to open-source their models(Stiennon et al., 2022). Yet, as highlighted in Qi et al. (2023), challenges persist in maintaining RLHF capabilities when fine-tuning GPT-4(OpenAI, 2023) on non-English data. Our experiments validate similar observations with other LLMs like LLaMA2.
![](https://cdn.mathpix.com/cropped/2024_06_04_1e0ae59091a72960dc4bg-02.jpg?height=852&width=1532&top_left_y=340&top_left_x=270)

Figure 1: $\mathrm{n}$ illustration to demonstrate the difference between the traditional approach and our method. In the traditional approach, considerable effort is expended in collecting a plethora of contextual data for continual pre-training (CP), various types of instruction-following data for instruction tuning, and significant human resources are allocated to label data for reinforcement learning from human feedback (RLHF). However, with our method, Instruction Continual Pre-training (InsCP), these processes are streamlined into a single step

In this work, we propose a novel fine-tuning approach called Instruction Continual Pretraining (InsCP) for LLMs to adapt to nonEnglish languages. This process draws inspiration from merging CP and SFT into a unified one-step training process. Additionally, we investigate whether LLMs, equipped with their own templates, can recognize tags during CP. Furthermore, we hypothesize that providing a chat template during CP prevents the model from forgetting its conversational abilities, as it resembles its original training conditions. Our approach begins with CP on a specific dataset, where we augment each piece of data with special instruction tokens, such as $<\mid$ begin_of_text $\mid>$ in LLaMA3(AI@Meta, 2024). This augmentation enables the model to respond to target language inputs in the target language and effectively handle offensive input based on its original RLHF capabilities.
We evaluate the effectiveness of InsCP on LLMs, primarily focusing on the LLaMA3instruct model, across three key aspects: language alignment, reliability, and knowledge benchmarks. Language alignment tests the model's proficiency in learning the desired language, while reliability evaluates its retention of RLHF capabilities. Knowledge benchmarks gauge the pipeline's impact on the model's comprehension ability. Our primary focus for InsCP is Traditional Chinese as our target language.

The results demonstrate that the model, after undergoing InsCP on LLaMA3-instruct, effectively performs in Traditional Chinese when prompted with Traditional Chinese input, surpassing the performance of LLaMA3-instruct. Moreover, in addition to aligning with Traditional Chinese prompts, the model retains its ability to respond appropriately to English prompts. Furthermore, most language bench-
marks indicate comparable performance between the model before and after CP. Additionally, when tested on TruthfulQA(Lin et al., 2022), a benchmark assessing the model's reliability, our model exhibits consistent performance in both English and Traditional Chinese, indicating that the RLHF ability remains intact without compromising performance, which typically requires significant investment to develop.

## 2 Related Work

### 2.1 LLMs adapt in other languages

Fine-tuning has been a longstanding technique for enabling models to adapt to specific domains, particularly in the realm of large language models (LLMs). Many downstream tasks have been successfully addressed through finetuning(Howard and Ruder, 2018; Devlin et al., 2019; Radford et al., 2018). While most downstream tasks can be accomplished through instruction fine-tuning, also known as supervised fine-tuning, adapting an English-based LLM to other languages, such as in the work of Fujii et al. (2024); Zhao et al. (2024); Cui et al. (2023); Lin and Chen (2023); YuLan-Team (2023) for fine-tuning in non-English languages, typically begins with continual pre-training. This initial step is crucial for ensuring that the models possess the necessary language proficiency and knowledge. Subsequently, instruction fine-tuning allows the model to engage in conversational interactions using specific templates.

### 2.2 Fine-tuning hurts LLMs

Recently, OpenAI introduced the capability to fine-tune GPT-3.5-turbo using user-provided data. In the (Qi et al., 2023), they collected a limited number of explicitly harmful examples, identity-shifting data, and the Alpaca dataset to perform instruction fine-tuning on GPT-3.5turbo and LLaMA-2-Chat. Their study evaluated these models against 11 criteria for harmful content, assessed by GPT-4. They observed that fine-tuning on these models led to an increase in harmful content generation. Even when employing safety data for fine-tuning, the resulting impact was still negative, albeit less pronounced than direct fine-tuning.

### 2.3 Training from human feedback

Ouyang et al. (2022) introduced InstructGPT, a model built upon GPT-3 (Brown et al., 2020), which they further refined through reinforcement learning from human feedback (RLHF). In their work, Ouyang et al. (2022) formally outlined the RLHF algorithm, which comprises three key components: Supervised Fine-Tuning (SFT), Reward Model training, and reinforcement learning via Proximal Policy Optimization (PPO) (Schulman et al., 2017). The RLHF algorithm enhances the model's ability to adhere to instructions and shows promise in mitigating the generation of toxic or harmful content.

Recent studies have explored avenues for optimizing human preference without necessarily relying on learning a reward function. For instance,Rafailov et al. (2023) proposed Direct Preference Optimization (DPO), refining the policy through a loss function constructed using the Bradley-Terry reward model.Azar et al. (2023) introduced Identity Policy Optimization (IPO), advocating for direct optimization of pairwise human preferences using preference data, distinct from DPO as IPO does not presuppose a reward model.Ethayarajh et al. (2024) put forth Kahneman-Tversky Optimization (KTO), suggesting the utilization of whether a given output is desirable or undesirable for a given input as the sole criterion to align the model with human preferences.

## 3 Methodology

Continual pre-training (CP) has traditionally served as a method for enhancing the comprehensive and generative capabilities of LLMs in a target language by leveraging extensive target language corpora. The underlying principle of the CP process involves training LLMs to predict the next token based on preceding content. The loss function guiding this process lists in below.

For our method, Instruction Continual Pretraining, we adopt a similar approach to CP, but with the addition of the model's original chat template. Taking LLaMA3instruct(AI@Meta, 2024) as an example, to initiate a completion with LLaMA3-instruct, one must adhere to the following format:
$|<|$ begin_of_text $|><|$ start_header_id $\mid>$ user $<\mid$ end_header_id $\mid>\{$ \{inputs $\}\}<\mid$ eot_id $|><|$ start_header_id $\mid>$ assistant $<\mid$ end_header_id $\mid>\{$ model_response $\}\}$

The inputs in the template represent the prompts provided by the user. In our context, where the objective is to train LLMs in the target language through next token prediction tasks while retaining their chat ability, we place the CP data in the model response. This arrangement ensures that LLMs generate tokens based on the target language. The InsCP template is structured as follows:

```
$<\mid$ begin_of_text $\mid><$ start_header_id $\mid>$ user
    $<$ end_header_id $\mid><$ eot_id $\mid><1$
    start_header _id $\mid>$ assistant $<1$
    end_header_id $\mid>\{\{\operatorname{InsCP}$ _data $\}<1$
    eot_id $\mid>\}$
```

The loss function for CP:

$$
\begin{align*}
\mathcal{L}_{\text {pretrain }} & =\mathbb{E}_{x \sim \mathcal{D}_{C P}}[ \\
& \left.-\sum_{i}^{S} \log P\left(x_{i} \mid x_{0}, \ldots, x_{i-1} ; \theta_{C P}\right)\right] \tag{1}
\end{align*}
$$

The loss function for InsCP:

$$
\begin{align*}
\mathcal{L}_{\text {pretrain }} & =\mathbb{E}_{x \sim \mathcal{D}_{\text {InsCP }}}[ \\
& \left.-\sum_{i}^{S} \log P\left(x_{i} \mid x_{0}, \ldots, x_{i-1} ; \theta_{\text {InsCP }}\right)\right] \tag{2}
\end{align*}
$$

where $\theta_{C P}$ and $\theta_{\text {InsCP }}$ represents the model parameters, $\mathcal{D}_{\mathcal{C P}}$ stands for the data used in continual pre-training, $\mathcal{D}_{\mathcal{I} \backslash \int C \mathcal{P}}$ stands for the data added the chat template and used in instruct continual pre-training, S represents the length of the input token sequence, and $x_{i}$ represents the token to be predicted, while $x_{0}, x_{1}, \ldots, x_{i-1}$ make up the context.

## 4 Experimental Setup

### 4.1 Training Dataset

We utilize a high-quality dataset comprising paired instruction-following data for LLaMA3instruct 8B(AI@Meta, 2024) during the InsCP procedure. The dataset consists of Traditional Chinese text and has a total size of 0.1 billion tokens. Throughout the InsCP process, we segregate the questions and answers into two separate data points. Further details regarding the training process are provided in the Appendix A.2

Moreover, to demonstrate the generalizability of our method to other languages, we extend our approach to Japanese. We utilize a 70M tokens dataset, structured similarly to the Traditional Chinese dataset, to perform InsCP on LLaMA3-instruct 8B.

From our experiments, we discovered the critical importance of selecting appropriate data for $\mathrm{InsCP}$. We aimed to determine the most suitable type of data for InsCP. Based on our findings, we selected wiki context data with high perplexity( $\mathrm{PPL} \geq 30)$ and two different types of instruction-following data with low perplexity. We observed that all instruction-following data with low perplexity $(\mathrm{PPL} \leq 15)$ successfully facilitated InsCP. We posit that this outcome is reasonable because data characterized by instruction-following and low perplexity are likely to closely resemble the original output of LLMs, thereby minimizing any adverse effects on the models' original abilities. The function of the perplexity is shown below:

$$
P P L(D \mid \Theta)=\exp \left(-\frac{1}{M} \sum_{i=1}^{M} \log p\left(d_{i} \mid \Theta\right)\right)
$$

Here, $\Theta$ represents the parameters of the language model. Dataset perplexity can be interpreted as the average perplexity of the model when predicting the entire dataset. Lower perplexity indicates better predictive performance of the model on the dataset.

### 4.2 Evaluation Dataset

We introduce three aspects of evaluation datasets to assess our InsCP model: language alignment, reliability, and knowledge benchmarks. Furthermore, we employ MT-Bench in our evaluation, we think that MT-Bench can test the LLMs more comprehensively. Throughout our testing, we maintain uniformity in all generation strategies, as detailed in the Appendix.

Language alignment We employ the FastText language identification model (Joulin
et al., 2016a,b) to determine the language of 2000 aligned sentences extracted from the English and Traditional Chinese subset of the NeuLab-TedTalks language within the tokens generated by our model.

Reliability We employ several common benchmarks to evaluate the reliability of the model's output, including TruthfulQA(Lin et al., 2022), ToxiGen(Hartvigsen et al., 2022), and BOLD(Dhamala et al., 2021) by using lmevaluation-harness(Gao et al., 2021). In the TruthfulQA benchmark, we assess the model's ability to accurately respond to questions based on factual information. ToxiGen allows us to evaluate the model's proficiency in generating non-toxic responses by utilizing a RoBERTabased(Liu et al., 2019) approach for identification, while BOLD assesses the model's confidence and coherence in its responses.

Knowledge benchmarks We utilize C-eval$\mathrm{tw}$, which is a translation of C-eval(Huang et al., 2023), to evaluate our model. Additionally, we assess our model using TTQA(Hsu et al., 2023), which focuses on Taiwanese commonsense and knowledge by 64 expert-selected paragraphs from Wikipedia. For traditional Chinese multitask benchmarking, we employ TMMLU Plus(Tam et al., 2024). To ensure that our model's English-related knowledge does not degrade, we include ARC(Clark et al., 2018) and Hellaswag(Zellers et al., 2019), which are benchmarks for English commonsense reasoning. For multitask evaluation, MMLU(Hendrycks et al., 2020) is a suitable choice.

MT-Bench(Zheng et al., 2023) We utilize MT-Bench to evaluate the comprehensive abilities of the models, encompassing knowledge, reliability, and language alignment. Additionally, MT-Bench incorporates multi-conversation scenarios, allowing us to assess the model's ability to handle multiple interactions simultaneously. This enables us to demonstrate that InsCP does not compromise the RLHF ability of the model.

### 4.3 Evaluation Metrics

Language alignment The FastText language identification model is utilized to determine the language of the generated text. The model classifies text into three categories: Chinese and English. The results include the percentage of sentences identified as Chinese, English, and others from a set of 2000 input prompts.
Reliability TruthfulQA consists of questions accompanied by multiple true/false options. Scoring is determined by assigning points based on the normalized cumulative probability assigned to the correct answers. ToxiGen utilizes a RoBERTa-based classifier to identify toxic generations and determine the toxicity score. For BOLD, we employ the Valence Aware Dictionary and Sentiment Reasoner (VADER(Hutto and Gilbert, 2014)) to calculate the sentiment score for both the prompt and generated text when combined. We present the average and standard deviation of the sentiment scores across all subgroups.

Knowledge benchmarks In ARC and Hellaswag, we utilize length-normalized accuracy as our metric. For MMLU and TMMLU Plus, we directly calculate accuracy for each task. In C-eval-tw, we compute metrics by averaging accuracy across individual tasks. The accuracy computation involves selecting the option with the highest probabilities. In TTQA, we extract the model's output and calculate accuracy based on multiple-choice questions.

MT-Bench In MT-Bench, the GPT-4 score serves as our evaluation metric. GPT-4 now serves as a standard for assessing the generation ability of LLMs, eliminating the need for expensive human evaluations. For each completed conversation, we invoke the GPT-4 $\mathrm{API}$, which returns a score ranging from 0 to 10. This score is based on various factors, including instruction following, harmfulness, and knowledge. Besides, we add the prompt about judging language alignment in GPT-4 evaluation in order to test model's language ability.

### 4.4 Baselines

We select LLaMA-3-instruct as our baseline model. To evaluate the performance of Instruction Continual Pre-training (InsCP), we conduct InsCP using our baseline model. Furthermore, to compare with the original continual pre-training process, we also fine-tune a model using this method. However, we observed that the original method significantly impairs the model's chat ability and may cause it to lose its instruction-following capability. Consequently, it becomes challenging to assess the model's performance using certain benchmarks.

| model | EN prompt |  | TW prompt |  |
| :---: | :---: | :---: | :---: | :---: |
| response | EN\% $\%$ | TW\% $\downarrow$ | EN\% $\downarrow$ | TW\% $\uparrow$ |
| LLaMA3-instruct | 1.0 | 0.0 | 0.90 | 0.09 |
| LLaMA3-orgCP | 1.0 | 0.0 | 0.50 | 0.49 |
| LLaMA3-InsCP | 0.99 | 0.01 | 0.01 | $\mathbf{0 . 9 9}$ |

Table 1: Language alignment benchmark.

| model | TruthfulQA <br> mc2 $\uparrow$ |  | ToxiGen <br> toxicity $\downarrow$ |  | BOLD <br> sentiment $\downarrow$ |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| language | EN | TW | EN | TW | EN | TW |
| LLaMA3-instruct | 51.6 | 52.7 | 0.10 | 0.14 | 0.54 | 0.55 |
| LLaMA3-orgCP | 50.8 | 50.5 | 0.12 | 0.26 | 0.61 | 0.68 |
| LLaMA3-InsCP | $\mathbf{5 1 . 8}$ | $\mathbf{5 3 . 8}$ | $\mathbf{0 . 0 7}$ | 0.16 | 0.56 | $\mathbf{0 . 5 2}$ |

Table 2: Reliability benchmark

## 5 Experimental Result

In this section, we provide the experimental results of four aspects: language alignment, reliability, knowledge and MT-Bench. For Traditional Chinese, we provide comprehensive assessments using MT-Bench. Additionally, we introduce MT-Bench-JP to evaluate the results specifically for Japanese InsCP. LLaMA3InsCP refers to LLaMA3-instruct conducted with instruction CP, while LLaMA3-orgCP denotes LLaMA3-instruct with original CP.

### 5.1 Language alignment evaluation

We adhere to our evaluation methodology outlined in Section 4.3, presenting the percentage of responses among 2000 prompts generated by the models. The experimental findings are summarized in Table 1. Our observations are as follows: (1)LLaMA3-instruct exhibits poor language alignment: As indicated in Table 1, when provided with Taiwanese (Traditional Chinese) input prompts, LLaMA3instruct frequently generates output in English. This lack of alignment between the input and output languages can lead to language nonalignment issues during usage. (2)The same data used with the original CP method fails to achieve proper alignment: A key distinction between InsCP and the original CP lies in their respective language learning capabilities. We observed that with the same data size, InsCP enables LLMs to acquire language proficiency more rapidly. (3)LLaMA3-InsCP demonstrates remarkable language profi- ciency: Regardless of whether provided with English or Traditional Chinese input prompts, LLaMA3-InsCP consistently responds in the appropriate language.

### 5.2 Reliability evaluation

In Table 2, we present the results of the models' reliability. Our experiments were conducted in both English and Chinese to ensure that our model does not compromise its reinforcement learning from human feedback (RLHF) ability in either language. Across each benchmark, we observe that the orgCP model consistently achieves lower scores compared to the other models. We attribute this outcome to our hypothesis that the model's RLHF ability diminishes during continual pre-training (CP) and supervised fine-tuning (SFT). However, both LLaMA3-instruct and LLaMA3-InsCP retain their RLHF ability, allowing them to defend against toxic inputs and generate non-harmful context during inference.

### 5.3 Knowledge benchmark

In Table 3, we present the scores from six benchmark tests. We specifically chose three language-relevant benchmarks to demonstrate that InsCP does not significantly impact the model's original English knowledge. Additionally, in Chinese-related benchmarks, we observed that the model after InsCP exhibited some improvements compared to both orgCP and the original model. These findings indicate that InsCP can effectively preserve the LLM's inherent abilities while also enhancing

| model | ARC | Hellaswag | MMLU | C-eval-tw | TMMLU + | TTQA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | ACC $\uparrow$ | ACC $\uparrow$ | ACC $\uparrow$ | ACC $\uparrow$ | ACC $\uparrow$ | ACC $\uparrow$ |
| LLaMA3-instruct | 60.5 | 81.8 | 67.2 | 47.3 | 43.0 | 23.3 |
| LLaMA3-orgCP | 57.5 | 81.3 | 66.1 | 48.5 | 41.3 | 41.3 |
| LLaMA3-InsCP | $\mathbf{6 1 . 6}$ | 81.7 | 65.6 | $\mathbf{4 8 . 9}$ | $\mathbf{4 1 . 9}$ | $\mathbf{4 8 . 5}$ |

Table 3: Knowledge benchmark

| model | MT-Bench |  |
| :---: | :---: | :---: |
| language | EN $\uparrow$ | TW $\uparrow$ |
| LLaMA3-instruct | 7.8 | 4.1 |
| LLaMA3-orgCP | 4.3 | 4.6 |
| LLaMA3-InsCP | 7.6 | $\mathbf{6 . 7}$ |

Table 4: MT-Bench

| model | MT-Bench-JP |
| :---: | :---: |
| LLaMA3-instruct | 4.9 |
| LLaMA3-orgCP-JP | 4.8 |
| LLaMA3-InsCP-JP | 6.6 |

Table 5: MT-Bench-JP

its performance in target language domains.

### 5.4 MT-Bench

In Table 4, MT-Bench further highlights the distinctions between orgCP and InsCP. We note that outputs from orgCP often contain irrelevant text that deviates from our input prompts. Moreover, the orgCP model appears to forget how to appropriately conclude conversations. Additionally, due to the inclusion of language alignment criteria in GPT-4 evaluation, we observe a significant disparity between the InsCP model and LLaMA3-instruct. While LLaMA3instruct predominantly responds in English for most questions, the InsCP model demonstrates the ability to discern the language input by the user.

### 5.5 MT-Bench-JP

In Table 5 , we observe a distribution similar to that of Traditional Chinese MT-Bench. Both LLaMA3-instruct and LLaMA3-InsCP-JP successfully generate responses in the correct format corresponding to the input prompts. However, LLaMA3-instruct fails to align the responses with the target language. Conversely, LLaMA3-orgCP-JP notably deviates from the instruction format, producing text unrelated to the input and sometimes generating repetitive text.

### 5.6 Limitations of InsCP

As discussed in Section 4.1, the choice of data used in InsCP significantly influences its outcomes. Our experiments indicate that conducting InsCP necessitates the utilization of lowperplexity instruction-following data, which can be challenging to acquire in abundance for certain languages. Consequently, we opted to perform InsCP using small datasets, which we believe is a more generalizable approach for languages with limited resources. Nonetheless, both data size and data quality remain challenges when implementing InsCP.

## 6 Conclusion

In this work, we introduce a novel pipeline called InsCP designed to facilitate the transfer of LLMs into non-English domains. Through InsCP, LLMs can retain their inherent abilities, including reinforcement learning from human feedback (RLHF) and knowledge in the English domain, while also acquiring the capability for language alignment in the target language and gaining knowledge of the target domain. Additionally, we demonstrate that InsCP does not necessitate extensive data, thereby consuming fewer resources and less time. Remarkably, even with a small amount of data, InsCP can transform English-based LLMs into models aligned with the target language, a stark contrast to the resource-intensive traditional pipeline. InsCP paves the way for future LLMs, primarily finetuned in specific languages, to swiftly transfer their abilities to other languages.

## 7 Acknowledgements

We extend our appreciation to the ASUS Open Cloud Infrastructure Software Center for generously providing valuable resources. Special thanks to Steve Chung-Cheng Chen, TsungYing Yang, Dau-Cheng Lyu, Jen-Hao Cheng, Hsiao-Tsung Hung, Szu-Hsien Lee for their participation in insightful discussions.

## References

1. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open foundation models by 01.ai.

AI@Meta. 2024. Llama 3 model card.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023. A general theoretical paradigm to understand learning from human preferences.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding.

Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, KaiWei Chang, and Rahul Gupta. 2021. Bold:
Dataset and metrics for measuring biases in openended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21. ACM.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization.

Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, and Naoaki Okazaki. 2024. Continual pre-training for crosslingual llm adaptation: Enhancing japanese language capabilities.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. A framework for few-shot language model evaluation.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. 2023. Continual pre-training of large language models: How to re-warm your model? In Workshop on Efficient Systems for Foundation Models @ ICML2023.

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. 2022. Toxigen: A large-scale machinegenerated dataset for adversarial and implicit hate speech detection.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification.

Chan-Jan Hsu, Chang-Le Liu, Feng-Ting Liao, PoChun Hsu, Yi-Chang Chen, and Da shan Shiu. 2023. Advancing the evaluation of traditional chinese language models: Towards a comprehensive benchmark suite.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. 2023. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models arXiv preprint arXiv:2305.08322.

Clayton Hutto and Eric Gilbert. 2014. Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the international AAAI conference on web and social media, volume 8, pages 216-225.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b.

Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, and Tomas Mikolov. 2016a. Fasttext.zip: Compressing text classification models. arXiv preprint arXiv:1612.03651.

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2016b. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759.

Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu. 2022. Continual pre-training of language models. In The Eleventh International Conference on Learning Representations.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods.

Yen-Ting Lin and Yun-Nung Chen. 2023. Language models for taiwanese culture. Code and models available at https://github.com/MiuLab/Taiwan-LLaMa.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2023. Fine-tuning aligned language models compromises safety, even when users do not intend to!

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and
Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. Learning to summarize from human feedback.

Zhi-Rui Tam, Ya-Ting Pai, Yen-Wei Lee, Sega Cheng, and Hong-Han Shuai. 2024. An improved traditional chinese evaluation suite for foundation model.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models.

BigScience Workshop, :, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, and Matthias Gallé et al. 2023. Bloom: A 176bparameter open-access multilingual language model.

YuLan-Team. 2023. Yulan-chat: An opensource bilingual chatbot. https://github.com/ RUC-GSAI/YuLan-Chat.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence?

Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Finetuning language models from human preferences.
</end of paper 1>


<paper 2>
# Mistral 7B 


#### Abstract

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed


![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-1.jpg?height=366&width=1073&top_left_y=749&top_left_x=534)


#### Abstract

We introduce Mistral 7B, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B - Instruct, that surpasses Llama 2 13B - chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.

Code: https://github.com/mistralai/mistral-src

Webpage: https://mistral.ai/news/announcing-mistral-7b/


## 1 Introduction

In the rapidly evolving domain of Natural Language Processing (NLP), the race towards higher model performance often necessitates an escalation in model size. However, this scaling tends to increase computational costs and inference latency, thereby raising barriers to deployment in practical, real-world scenarios. In this context, the search for balanced models delivering both high-level performance and efficiency becomes critically essential. Our model, Mistral 7B, demonstrates that a carefully designed language model can deliver high performance while maintaining an efficient inference. Mistral 7B outperforms the previous best 13B model (Llama 2, [26]) across all tested benchmarks, and surpasses the best 34B model (LLaMa 34B, [25]) in mathematics and code generation. Furthermore, Mistral 7B approaches the coding performance of Code-Llama 7B [20], without sacrificing performance on non-code related benchmarks.

Mistral 7B leverages grouped-query attention (GQA) [1], and sliding window attention (SWA) [6, 3]. GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost, thereby alleviating a common limitation in LLMs. These attention mechanisms collectively contribute to the enhanced performance and efficiency of Mistral 7B.

Mistral 7B is released under the Apache 2.0 license. This release is accompanied by a reference implementation ${ }^{1}$ facilitating easy deployment either locally or on cloud platforms such as AWS, GCP, or Azure using the vLLM [17] inference server and SkyPilot ${ }^{2}$. Integration with Hugging Face ${ }^{3}$ is also streamlined for easier integration. Moreover, Mistral 7B is crafted for ease of fine-tuning across a myriad of tasks. As a demonstration of its adaptability and superior performance, we present a chat model fine-tuned from Mistral 7B that significantly outperforms the Llama 2 13B - Chat model.

Mistral 7B takes a significant step in balancing the goals of getting high performance while keeping large language models efficient. Through our work, our aim is to help the community create more affordable, efficient, and high-performing language models that can be used in a wide range of real-world applications.

## 2 Architectural details

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-2.jpg?height=504&width=1326&top_left_y=816&top_left_x=384)

Effective Context Length

Figure 1: Sliding Window Attention. The number of operations in vanilla attention is quadratic in the sequence length, and the memory increases linearly with the number of tokens. At inference time, this incurs higher latency and smaller throughput due to reduced cache availability. To alleviate this issue, we use sliding window attention: each token can attend to at most $W$ tokens from the previous layer (here, $W=3$ ). Note that tokens outside the sliding window still influence next word prediction. At each attention layer, information can move forward by $W$ tokens. Hence, after $k$ attention layers, information can move forward by up to $k \times W$ tokens.

Mistral 7B is based on a transformer architecture [27]. The main parameters of the architecture are summarized in Table 1. Compared to Llama, it introduces a few changes that we summarize below.

Sliding Window Attention. SWA exploits the stacked layers of a transformer to attend information beyond the window size $W$. The hidden state in position $i$ of the layer $k, h_{i}$, attends to all hidden states from the previous layer with positions between $i-W$ and $i$. Recursively, $h_{i}$ can access tokens from the input layer at a distance of up to $W \times k$ tokens, as illustrated in Figure 1. At the last layer, using a window size of $W=4096$, we have a theoretical attention span of approximately $131 K$ tokens. In practice, for a sequence length of $16 \mathrm{~K}$ and $W=4096$, changes made to FlashAttention [11] and xFormers [18] yield a 2x speed improvement over a vanilla attention baseline.

Rolling Buffer Cache. A fixed attention span means that we can limit our cache size using a rolling buffer cache. The cache has a fixed size of $W$, and the keys and values for the timestep $i$ are stored in position $i \bmod W$ of the cache. As a result, when the position $i$ is larger than $W$, past values in the cache are overwritten, and the size of the cache stops increasing. We provide an illustration in Figure 2 for $W=3$. On a sequence length of $32 \mathrm{k}$ tokens, this reduces the cache memory usage by $8 \mathrm{x}$, without impacting the model quality.[^0]

| Parameter | Value |
| :--- | ---: |
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 |
| window_size | 4096 |
| context_len | 8192 |
| vocab_size | 32000 |

Table 1: Model architecture.

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-3.jpg?height=225&width=1396&top_left_y=251&top_left_x=362)

Figure 2: Rolling buffer cache. The cache has a fixed size of $W=4$. Keys and values for position $i$ are stored in position $i \bmod W$ of the cache. When the position $i$ is larger than $W$, past values in the cache are overwritten. The hidden state corresponding to the latest generated tokens are colored in orange.

Pre-fill and Chunking. When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones. However, the prompt is known in advance, and we can pre-fill the $(k, v)$ cache with the prompt. If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk. For this purpose, we can select the window size as our chunk size. For each chunk, we thus need to compute the attention over the cache and over the chunk. Figure 3 shows how the attention mask works over both the cache and the chunk.

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-3.jpg?height=393&width=963&top_left_y=928&top_left_x=581)

Figure 3: Pre-fill and chunking. During pre-fill of the cache, long sequences are chunked to limit memory usage. We process a sequence in three chunks, "The cat sat on", "the mat and saw", "the dog go to". The figure shows what happens for the third chunk ("the dog go to"): it attends itself using a causal mask (rightmost block), attends the cache using a sliding window (center block), and does not attend to past tokens as they are outside of the sliding window (left block).

## 3 Results

We compare Mistral 7B to Llama, and re-run all benchmarks with our own evaluation pipeline for fair comparison. We measure performance on a wide variety of tasks categorized as follow:

- Commonsense Reasoning (0-shot): Hellaswag [28], Winogrande [21], PIQA [4], SIQA [22], OpenbookQA [19], ARC-Easy, ARC-Challenge [9], CommonsenseQA [24]
- World Knowledge (5-shot): NaturalQuestions [16], TriviaQA [15]
- Reading Comprehension (0-shot): BoolQ [8], QuAC [7]
- Math: GSM8K [10] (8-shot) with maj@ 8 and MATH [13] (4-shot) with maj@4
- Code: Humaneval [5] (0-shot) and MBPP [2] (3-shot)
- Popular aggregated results: MMLU [12] (5-shot), BBH [23] (3-shot), and AGI Eval [29] (3-5-shot, English multiple-choice questions only)

Detailed results for Mistral 7B, Llama 27B/13B, and Code-Llama 7B are reported in Table 2. Figure 4 compares the performance of Mistral 7B with Llama 2 7B/13B, and Llama $134 \mathrm{~B}^{4}$ in different categories. Mistral 7B surpasses Llama 2 13B across all metrics, and outperforms Llama 1 34B on most benchmarks. In particular, Mistral 7B displays a superior performance in code, mathematics, and reasoning benchmarks.[^1]![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-4.jpg?height=440&width=1390&top_left_y=257&top_left_x=365)

Figure 4: Performance of Mistral 7B and different Llama models on a wide range of benchmarks. All models were re-evaluated on all metrics with our evaluation pipeline for accurate comparison. Mistral 7B significantly outperforms Llama 2 7B and Llama 2 13B on all benchmarks. It is also vastly superior to Llama 1 34B in mathematics, code generation, and reasoning benchmarks.

| Model | Modality | MMLU | HellaSwag | WinoG | PIQA | Arc-e | Arc-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaMA 2 7B | Pretrained | $44.4 \%$ | $77.1 \%$ | $69.5 \%$ | $77.9 \%$ | $68.7 \%$ | $43.2 \%$ | $24.7 \%$ | $63.8 \%$ | $11.6 \%$ | $26.1 \%$ | $3.9 \%$ | $16.0 \%$ |
| LLaMA 2 13B | Pretrained | $55.6 \%$ | $\mathbf{8 0 . 7 \%}$ | $72.9 \%$ | $80.8 \%$ | $75.2 \%$ | $48.8 \%$ | $\mathbf{2 9 . 0 \%}$ | $69.6 \%$ | $18.9 \%$ | $35.4 \%$ | $6.0 \%$ | $34.3 \%$ |
| Code-Llama 7B | Finetuned | $36.9 \%$ | $62.9 \%$ | $62.3 \%$ | $72.8 \%$ | $59.4 \%$ | $34.5 \%$ | $11.0 \%$ | $34.9 \%$ | $31.1 \%$ | $\mathbf{5 2 . 5 \%}$ | $5.2 \%$ | $20.8 \%$ |
| Mistral 7B | Pretrained | $60.1 \%$ | $81.3 \% \quad$ | $75.3 \%$ | $\mathbf{8 3 . 0 \%}$ | $\mathbf{8 0 . 0 \%}$ | $\mathbf{5 5 . 5 \%}$ | $\mathbf{2 8 . 8 \%}$ | $69.9 \%$ | $30.5 \%$ | $47.5 \%$ | $13.1 \%$ | $\mathbf{5 2 . 2 \%}$ |

Table 2: Comparison of Mistral 7B with Llama. Mistral 7B outperforms Llama 2 13B on all metrics, and approaches the code performance of Code-Llama 7B without sacrificing performance on non-code benchmarks.

Size and Efficiency. We computed "equivalent model sizes" of the Llama 2 family, aiming to understand Mistral 7B models' efficiency in the cost-performance spectrum (see Figure 5). When evaluated on reasoning, comprehension, and STEM reasoning (specifically MMLU), Mistral 7B mirrored performance that one might expect from a Llama 2 model with more than $3 x$ its size. On the Knowledge benchmarks, Mistral 7B's performance achieves a lower compression rate of 1.9x, which is likely due to its limited parameter count that restricts the amount of knowledge it can store.

Evaluation Differences. On some benchmarks, there are some differences between our evaluation protocol and the one reported in the Llama 2 paper: 1) on MBPP, we use the hand-verified subset 2) on TriviaQA, we do not provide Wikipedia contexts.

## 4 Instruction Finetuning

To evaluate the generalization capabilities of Mistral 7B, we fine-tuned it on instruction datasets publicly available on the Hugging Face repository. No proprietary data or training tricks were utilized: Mistral 7B - Instruct model is a simple and preliminary demonstration that the base model can easily be fine-tuned to achieve good performance. In Table 3, we observe that the resulting model, Mistral 7B - Instruct, exhibits superior performance compared to all 7B models on MT-Bench, and is comparable to $13 \mathrm{~B}$ - Chat models. An independent human evaluation was conducted on https://llmboxing.com/leaderboard.

| Model | Chatbot Arena <br> ELO Rating | MT Bench |
| :--- | :---: | :--- |
| WizardLM 13B v1.2 | 1047 | 7.2 |
| Mistral 7B Instruct | $\mathbf{1 0 3 1}$ | $\mathbf{6 . 8 4}+/ \mathbf{0 . 0 7}$ |
| Llama 2 13B Chat | 1012 | 6.65 |
| Vicuna 13B | 1041 | 6.57 |
| Llama 2 7B Chat | 985 | 6.27 |
| Vicuna 7B | 997 | 6.17 |
| Alpaca 13B | 914 | 4.53 |

Table 3: Comparison of Chat models. Mistral 7B Instruct outperforms all 7B models on MT-Bench, and is comparable to $13 \mathrm{~B}-$ Chat models.

In this evaluation, participants were provided with a set of questions along with anonymous responses from two models and were asked to select their preferred response, as illustrated in Figure 6. As of October 6, 2023, the outputs generated by Mistral 7B were preferred 5020 times, compared to 4143 times for Llama 2 13B.
![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-5.jpg?height=682&width=1000&top_left_y=244&top_left_x=560)

Figure 5: Results on MMLU, commonsense reasoning, world knowledge and reading comprehension for Mistral 7B and Llama 2 (7B/13B/70B). Mistral 7B largely outperforms Llama 2 13B on all evaluations, except on knowledge benchmarks, where it is on par (this is likely due to its limited parameter count, which limits the amount of knowledge it can compress).

## 5 Adding guardrails for front-facing applications

The ability to enforce guardrails when it comes to AI generation is important for front-facing applications. In this section, we highlight how to leverage system prompting to optionally enforce output constraints on top of our models. Additionally, we showcase the ability of Mistral 7B to perform fine-grained content moderation, which can be useful to enforce quality content in applications.

### 5.1 System prompt to enforce guardrails

We introduce a system prompt (see below) to guide the model to generate answers within specified guardrails, similar to the work done with Llama 2. Using this prompt allows the user to move on the Pareto front of model utility / guardrails enforcement, as indicated in Table 4.

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

We use a set of 175 unsafe prompts for evaluating safety. With the recommended system prompt, the model properly declines to answer $100 \%$ of the harmful questions.

As an illustration, we provide in Table 5 the answers of both Mistral 7B - Instruct and Llama 2 Chat 13B to the question How to kill a linux process with system prompts activated. We observe that Mistral 7B provides a correct response while Llama 2 declines to answer. Note that on this specific question, both models answer correctly when system prompts are deactivated.

| Guardrails | MT Bench |
| :---: | :---: |
| No system prompt | $6.84 \pm 0.07$ |
| Llama 2 system prompt | $6.38 \pm 0.07$ |
| Mistral system prompt | $6.58 \pm 0.05$ |

Table 4: System prompts. Mean official MT Bench score over 10 iterations with standard deviation for Mistral 7B - Instruct. For reference, Llama 2 13B - Chat reports official results of 6.65 .

### 5.2 Content moderation with self-reflection

Mistral 7B - Instruct can be used as a content moderator: the model itself is able to accurately classify a user prompt or its generated answer as being either acceptable or falling into one of the following categories: Illegal activities such as terrorism, child abuse or fraud; Hateful, harassing or violent content such as discrimination, self-harm or bullying; Unqualified advice for instance in legal, medical or financial domains.

| Model | Answer |
| :--- | :--- | | To kill a Linux process, you can use the ` kill` command followed by the process ID (PID) |
| :--- |
| of the process you want to terminate. For example, to kill process with PID 1234, you |
| would run the command `kill 1234`. It's important to note that killing a process can |
| have unintended consequences, so it's generally a good idea to only kill processes that |
| you are certain you want to terminate. Additionally, it's generally a good idea to use the |
| `kill`command with caution, as it can be used to terminate critical system processes that |
| are necessary for the operation of the system. |

Table 5: Comparison between Mistral and Llama system prompts. Mistral provides the right answer whereas Llama 2 declines to answer to the question How to kill a linux process.

To do so, we designed a self-reflection prompt that makes Mistral 7B classify a prompt or a generated answer. We evaluated self-reflection on our manually curated and balanced dataset of adversarial and standard prompts and got a precision of $99.4 \%$ for a recall of $95.6 \%$ (considering acceptable prompts as positives).

The use cases are vast, from moderating comments on social media or forums to brand monitoring on the internet. In particular, the end user is able to select afterwards which categories to effectively filter based on their particular use-case.

## 6 Conclusion

Our work on Mistral 7B demonstrates that language models may compress knowledge more than what was previously thought. This opens up interesting perspectives: the field has so far put the emphasis on scaling laws in 2 dimensions (directly associating model capabilities to training cost, as in [14]); the problem is rather 3 dimensional (model capabilities, training cost, inference cost), and much remains to be explored to obtain the best performance with the smallest possible model.

## Acknowledgements

We are grateful to CoreWeave for their $24 / 7$ help in marshalling our cluster. We thank the CINECA/EuroHPC team, and in particular the operators of Leonardo, for their resources and help. We thank the maintainers of FlashAttention, vLLM, xFormers, Skypilot for their precious assistance in implementing new features and integrating their solutions into ours. A huge thanks to Tri Dao and Daniel Haziza for helping include Mistral related changes to FlashAttention and xFormers on a tight schedule. We thank the teams of Hugging Face, AWS, GCP, Azure ML for their intense help in making our model compatible everywhere.

# Can you recommend a foundational and accessible book about quantum physics? 

\author{

| Copy Link to Question | Next Round (Enter) $\rightarrow$ |
| :--- | :--- |

}

Round Winner: mistral-7b-instruct-v0.1

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-7.jpg?height=309&width=301&top_left_y=995&top_left_x=923)

mistral-7b-instruct-v0.1 Round Winner

A great book for a foundational understanding of quantum physics is "The Quantum Universe: Everything that Can Happen Does Happen" by Brian Cox and Jeff Forshaw. This book is accessible and easy to understand, making it a great choice for beginners. It covers all the basics of quantum mechanics, including superposition, entanglement, and the uncertainty principle. The authors also explore the implications of quantum physics for our understanding of the universe, including the nature of time and space. Overall, "The Quantum Universe" is an excellent choice for anyone looking to gain a foundational understanding of this fascinating field
Llama-2-13b-chat

Certainly! For a foundational and accessible introduction to quantum physics, I highly recommend "The Feynman Lectures on Physics" by Richard Feynman. This classic text provides a clear and engaging overview of the principles and concepts of quantum mechanics, with minimal mathematical formalism and plenty of illustrations and examples. It's a great starting point for anyone looking to gain a deeper understanding of quantum physics without getting bogged down in too much technical detail.

Figure 6: Human evaluation of Mistral 7B - Instruct vs Llama 2 13B - Chat Example. An example of human evaluation from llmboxing. com. The question asks for recommendations of books in quantum physics. Llama 2 13B - Chat recommends a general physics book, while Mistral 7B - Instruct recommends a more relevant book on quantum physics and describes in the contents in more detail.

## References

[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[4] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, 2020.

[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[7] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. Quac: Question answering in context. arXiv preprint arXiv:1808.07036, 2018 .

[8] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

[9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[14] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems, volume 35, 2022.

[15] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.

[16] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466, 2019.

[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[18] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022.

[19] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

[20] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99-106, 2021.

[22] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

[23] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

[24] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.

[25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[28] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[29] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.


[^0]:    ${ }^{1}$ https://github.com/mistralai/mistral-src

    ${ }^{2}$ https://github.com/skypilot-org/skypilot

    ${ }^{3}$ https://huggingface.co/mistralai

[^1]:    ${ }^{4}$ Since Llama 2 34B was not open-sourced, we report results for Llama 1 34B.

</end of paper 2>


