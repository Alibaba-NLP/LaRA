<paper 0>
# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models 

Zixiang Chen* ${ }^{* \dagger}$ Yihe Deng*‡ Huizhuo Yuan*§ Kaixuan Ji ${ }^{* \S}$ Quanquan Gu ${ }^{*}$


#### Abstract

Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional humanannotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM's performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents. Codes are available at https://github.com/uclaml/SPIN.


## 1 Introduction

Large Language Models (LLMs) have began a groundbreaking era in artificial general intelligence (AGI), demonstrating extraordinary capabilities across a wide range of domains that require intricate reasoning and specialized knowledge. These models excel in areas such as mathematical reasoning/problem solving (Cobbe et al., 2021; Wei et al., 2022; Lewkowycz et al., 2022), code generation/programming (Chen et al., 2021; Austin et al., 2021; Li et al., 2022), text generation (Bubeck[^0]et al., 2023; Anil et al., 2023; Touvron et al., 2023), summarization and creative writing, among others. A significant advancement in LLMs is the post-pretraining alignment with the more desirable behaviors (Mishra et al., 2021; Victor et al., 2022; Chung et al., 2022; Thoppilan et al., 2022), a process often reliant on the costly human-annotated data. Typical alignment methods include Supervised Fine-Tuning (SFT) (Ouyang et al., 2022; Tunstall et al., 2023a) based on human demonstrations, and Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020; Bai et al., 2022a) based on human preferences.

All the aforementioned alignment methods require a substantial volume of human annotated data. Therefore, there is increasing interest in developing fine-tuning methods that can effectively utilize human data, thereby streamlining the alignment process. This motivates us to study fine-tuning LLMs without the need for additional human-annotated data beyond the fine-tuning dataset. Our study is also related to the broader goal of converting weak models to strong models without the requirement for extra training data, which is of central interest in machine learning that can be traced back to the boosting algorithms (Kearns and Valiant, 1994; Schapire, 1990; Freund, 1995; Freund and Schapire, 1997). The self-training algorithm (Vapnik, 1999; Grandvalet and Bengio, 2004; Lee, 2013) has also been proved to be able to convert weak learners to strong learners in mixture models without the need for additional labeled data (Frei et al., 2022; Kou et al., 2022). However, the pursuit of autonomously enhancing a weak LLM without external guidance is both intriguing and understudied. This raises the following question:

Can we empower a weak LLM to improve itself without acquiring additional human annotated data?

In this paper, we answer this question affirmatively. Inspired by the success of self-play mechanisms (Samuel, 2000) in games, exemplified by AlphaGo Zero (Silver et al., 2017b), AlphaZero (Silver et al., 2017a), with historical roots traced back to TD-Gammon (Tesauro et al., 1995), we propose to convert a weak LLM to a strong one through the lens of self-play, where the model is enhanced by playing against itself without requiring any direct supervision. In particular, we propose a novel fine-tuning method called Self-Play fIne-tuNing (SPIN), which begins from a supervised fine-tuned model. SPIN allows the LLM to engage in self-play, eliminating the need for an expert annotator such as a human or more advanced LLMs like GPT-4. In detail, with the LLM from previous iteration $t$ denoted by $p_{\boldsymbol{\theta}_{t}}$, we employ it to generate responses $\mathbf{y}^{\prime}$ to the prompts $\mathbf{x}$ in the human-annotated SFT dataset. The subsequent objective is to find a new LLM $p_{\theta_{t+1}}$, capable of distinguishing the responses $\mathbf{y}^{\prime}$ generated by $p_{\boldsymbol{\theta}_{t}}$ from the responses $\mathbf{y}$ generated by humans. This process can be seen as a two-player game: the main player, or the new LLM $p_{\boldsymbol{\theta}_{t+1}}$, seeks to discern between the responses of the opponent player $p_{\theta_{t}}$ and human-generated responses, while the opponent, or the old LLM $p_{\theta_{t}}$, generates responses as similar as possible to those in the human-annotated SFT dataset. The new LLM $p_{\boldsymbol{\theta}_{t+1}}$ is obtained by fine-tuning the old one $p_{\boldsymbol{\theta}_{t}}$ to prefer responses from $p_{\text {data }}$ over $p_{\boldsymbol{\theta}_{t}}$, resulting in a distribution $p_{\boldsymbol{\theta}_{t+1}}$ that is more aligned with $p_{\text {data }}$. In the next iteration, the newly obtained LLM $p_{\boldsymbol{\theta}_{t+1}}$ becomes the opponent for response generation, with the self-play process aiming for the LLM to eventually converge to $p_{\boldsymbol{\theta}^{*}}=p_{\text {data }}$, so that the strongest possible LLM can no longer differentiate the responses generated by its previous version and those generated by the human.

Interestingly, our method exhibits similarity with the recently introduced direct preference optimization (DPO) method (Rafailov et al., 2023), with the notable distinction being the self-play nature of our method. Consequently, our approach stands out by eliminating the need for extra human preference data, a requirement present in the DPO method. Additionally, the self-play mechanism in our method resembles the idea of generative adversarial networks (GAN) (Goodfellow
et al., 2014; Arjovsky et al., 2017), albeit that both the discriminator (main player) and the generator (the opponent) in our method are instances of the same LLM from different iterations. Theoretically, we prove that our method converges when the distribution of the LLM is identical to the target data distribution, i.e., $p_{\boldsymbol{\theta}_{t}}=p_{\text {data }}$. Our experimental results on zephyr-7b-sft-full (Tunstall et al., 2023a), a fine-tuned LLM based on Mistral-7B (Jiang et al., 2023), show that while continued training using SFT on its own SFT dataset Ultrachat200k (Ding et al., 2023) reaches a performance plateau or even diminished evaluation scores, our method consistently improves zephyr-7b-sft-full across successive iterations while leveraging only a $50 \mathrm{k}$ subset of Ultrachat200k dataset. Ultimately, SPIN effectively improves the base model's average score from 58.14 to $\mathbf{6 3 . 1 6}$ on the HuggingFace Open LLM Leaderboard (Beeching et al., 2023) with remarkable $10 \%+$ improvement in scores on GSM8k and TruthfulQA, and from 5.94 to $\mathbf{6 . 7 8}$ on MT-Bench (Zheng et al., 2023). Notably, SPIN achieves results that are even comparable to models trained on additional $62 \mathrm{k}$ preference dataset (Tunstall et al., 2023a) on Open LLM leaderboard and MT-Bench.

Concurrent to our work, Singh et al. (2023) proposed the use of synthetic data with binary feedback in self-training, reducing the reliance on human data. In contrast, our approach eliminates the need for additional binary feedback from humans or an extra reward model thanks to the self-play mechanism. Additionally, Burns et al. (2023) employed a weak LLM model as the guidance to train stronger LLMs in a fashion of weak-to-strong generation. Unlike Burns et al. (2023), which necessitates both a weak supervisor and a strong model, our SPIN operates effectively with a single LLM.

Notation. We use lowercase letters and lowercase boldface letters to denote scalars and vectors, respectively. We use $[N]$ to denote the index set $\{1, \ldots, N\}$. In the function space, let $\mathcal{F}$ be the function class. The symbol $q_{\text {data }}$ designates the target data distribution, while $p$ represents the conditional probability of LLM's response (i.e., LLM policy).

## 2 Related Work

Self-Play. Self-play (Samuel, 1959; Tesauro et al., 1995), where the algorithm learns by playing against itself, has gained notable attention due to its effectiveness in multi-agent reinforcement learning (MARL). This method involves agents engaging in interactions with copies of themselves, enabling an increasing level of challenge and complexity within the learning environment. A fundamental work in the field of self-play is AlphaGo Zero (Silver et al., 2017b), which demonstrated exceptional performance against human players using a self-play learning scheme. Subsequent research has expanded upon the concept of self-play, exploring various adaptations and implementations (Anthony et al., 2017; Lanctot et al., 2017; Bansal et al., 2018; Hernandez-Leal et al., 2018; Muller et al., 2019; Vinyals et al., 2019). Our method takes the self-play approach akin to AlphaGo Zero, which can convert a weak model to a strong one without additional human-annotated data. While the effectiveness of self-play in MARL is well-established, to our knowledge, our work is the first to apply this approach to the enhancement of LLMs.

Synthetic Data for LLMs. In the context of supervised fine-tuning (SFT) of LLMs, humancrafted data has proven to be a remarkably effective source that enhances the performance of LLMs on tasks such as code generation (Roziere et al., 2023; Yang et al., 2023) and mathematical reasoning (Yuan et al., 2023; Luo et al., 2023). While human data typically exhibits high quality, acquiring sufficient amount of such data poses a challenge in cost. In light of this consideration, the use of synthetic data has become increasingly popular and considered as a proxy for human data. This approach primarily leverages advanced LLMs such as the GPT series (Radford et al., 2019;

Brown et al., 2020; OpenAI, 2023) as the guidance to generate high-quality data (Josifoski et al., 2023; Taori et al., 2023; Chiang et al., 2023; Li et al., 2023). Recent research has also highlighted the rephrasing capability of LLMs in prompting for better LLM response (Deng et al., 2023; Prasad et al., 2023) as well as augmenting synthetic data for more effective SFT (Yu et al., 2023; Liu et al., 2023). In contrast to prior studies that utilized more advanced models for synthetic data generation when pre-training or fine-tuning a target model, our approach directly generates synthetic data from the target model itself.

## 3 Problem Setting and Preliminaries

We consider a Large Language Model (LLM) parameterized by $\boldsymbol{\theta}$ and denoted by $p_{\boldsymbol{\theta}}$. The model takes as input a sequence $\mathbf{x}=\left[x_{1}, \ldots, x_{n}\right]$, commonly referred to as the prompt, to generate the corresponding response $\mathbf{y}=\left[y_{1}, \ldots, y_{m}\right]$. The response $\mathbf{y}$ is therefore considered as a sample from the conditional probability distribution $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$. In LLMs, $x_{i}$ and $y_{j}$ represent individual tokens from a predetermined vocabulary within the sequences $\mathbf{x}$ and $\mathbf{y}$, respectively. The autoregressive model $p_{\boldsymbol{\theta}}$ generates tokens sequentially for a given position, leveraging only the sequence of previously generated tokens. This model therefore constitutes a Markov process, where the conditional probability distribution $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})$ can be expressed through a decomposition as follows:

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\prod_{j=1}^{m} p_{\boldsymbol{\theta}}\left(y_{j} \mid \mathbf{x}, \mathbf{y}_{<j}\right)
$$

where $\mathbf{y}_{<1}$ is null and $\mathbf{y}_{<j}=\left[y_{1}, \ldots, y_{j-1}\right]$ for $j=2, \ldots, m$. In the following, we review two major fine-tuning methods for LLMs: supervised fine-tuning and reinforcement learning (RL) fine-tuning.

### 3.1 Supervised Fine-Tuning

Supervised fine-tuning (SFT) is employed to tailor a pre-trained LLM to specific downstream tasks, leveraging relatively smaller dataset of labeled examples in comparison to the large-scale pre-training data (Ouyang et al., 2022; Yu et al., 2023). In this context, we consider a specific task where the prompts, denoted by $\mathbf{x}$, are derived from a specified distribution $q(\cdot)$. The notation $p_{\text {data }}(\cdot \mid \mathbf{x})$ then represents the probability distribution of the associated high-quality responses $\mathbf{y}$ from the training data. Consequently, SFT involves training the LLM to minimize the following negative log-likelihood loss associated with these distributions,

$$
\begin{equation*}
L_{\mathrm{SFT}}(\boldsymbol{\theta})=-\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})\right] \tag{3.1}
\end{equation*}
$$

It should be noted that excluding $\mathbf{x} \sim q(\cdot)$ from the expectation term yields the typical crossentropy loss, expressed as $-\mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})\right] . L_{\mathrm{SFT}}(\boldsymbol{\theta})$ attains its minimum when the model's predictive distribution $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})$ aligns perfectly with the distribution of the labeled high-quality responses $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$.

Consequently, the LLM after SFT is anticipated to generate responses that closely resemble those from $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. This procedure is therefore expected to significantly enhance the model's performance in generating appropriate responses for a specific task.

### 3.2 RL Fine-Tuning

RL fine-tuning (Christiano et al., 2017; Bai et al., 2022a; Gao et al., 2023a) offers another method for enhancing the specific capabilities of general-purpose pre-trained models. Typically, RL fine-tuning is employed subsequent to SFT to achieve improved alignment for LLMs (Tunstall et al., 2023a).

For a given sequence pair $(\mathbf{x}, \mathbf{y})$, RL fine-tuning necessitates a deterministic reward function $r(\mathbf{x}, \mathbf{y})$. The higher the reward $r(\mathbf{x}, \mathbf{y})$, the better the response $\mathbf{y}$ is to the given prompt $\mathbf{x}$. The objective of the RL fine-tuning process is then to maximize the following objective function:

$$
L_{\mathrm{RL}}(\boldsymbol{\theta})=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})}[r(\mathbf{x}, \mathbf{y})]-\lambda \mathbb{E}_{\mathbf{x} \sim q(\cdot)} \mathrm{KL}\left(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}) \| p_{\mathrm{ref}}(\cdot \mid \mathbf{x})\right)
$$

where the Kullback-Leibler (KL) regularization enforces the new model $p_{\boldsymbol{\theta}}$ to be close to the reference model $p_{\text {ref }}$, and $\lambda>0$ is the regularization parameter to control the deviation of the new model $p_{\boldsymbol{\theta}}$ from the reference model $p_{\text {ref }}$. In practice, the reference model $p_{\text {ref }}$ is often initialized as the supervised fine-tuned model. The inclusion of KL regularization is vital for preventing excessive deviation from the reference model, which in turn reduces the risk of mode collapse.

Meanwhile, the primary challenge in RL fine-tuning lies in finding a good reward function. Typically, this function requires training on a preference dataset. The compilation of such a dataset demands significant resources, often involving comprehensive evaluations either by human annotators, i.e., reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Bai et al., 2022a) or strong AI agents, i.e., reinforcement learning from AI feedback (RLAIF) (Bai et al., 2022b).

## 4 Method

In this section, we introduce a new fine-tuning method for enhancing the performance of LLMs without relying on additional human or AI feedback. Consider a high-quality supervised fine-tuning (SFT) dataset $S_{\mathrm{SFT}}=\{(\mathbf{x}, \mathbf{y})\}_{i=1}^{n}$, which are sampled from the marginal distribution $q(\mathbf{x})$ and $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Given a supervised fine-tuned LLM $p_{\boldsymbol{\theta}_{0}}$, further application of the SFT approach in (3.1) with $S_{\mathrm{SFT}}$ will be ineffective and potentially lead to worse performance. In addition, without human and/or AI feedback, it becomes infeasible to acquire a preference dataset for RL fine-tuning (e.g., RLHF and RLAIF). This hinders the application of RL fine-tuning techniques.

We evaluate $p_{\boldsymbol{\theta}_{0}}$ against $S_{\mathrm{SFT}}$, where $p_{\boldsymbol{\theta}_{0}}$ is the LLM achieved by SFT using (3.1). We notice a persistent quality gap between the groundtruth response $\mathbf{y}$ from $S_{\mathrm{SFT}}$ and the LLM-generated response $\mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ (refer to Figure 1). This disparity indicates that there is still room for improvement over $p_{\boldsymbol{\theta}_{0}}$. Therefore, we propose to use the synthetic data generated by the LLM to enhance LLM's performance starting from $p_{\boldsymbol{\theta}_{0}}$ iteratively.

### 4.1 Self-Play Fine-Tuning (SPIN)

Let us consider a two-player game, where the main player's objective is to distinguish the responses generated by the LLM and those generated by the human. Meanwhile, the opponent's role is to generate responses that are indistinguishable from the human's responses. The core of our method is the self-play mechanism, where both the main player and the opponent are the same LLM, but from different iterations. More specifically, the opponent is the old LLM from the previous iteration, and the main player is the new LLM to be learned in the current iteration.

In iteration $t+1$, the opponent is the LLM from the previous iteration, denoted by $p_{\theta_{t}}$, which generates responses $\mathbf{y}^{\prime}$ for those prompts $\mathbf{x}$ in the SFT dataset according to $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$. Our method, therefore, consists of the following two steps at iteration $t+1$ : (1) training the main player, and (2) updating the opponent player.

Training the Main Player. We begin with illustrating how we expect a main player is trained to distinguish LLM responses from human responses. Motivated by integral probability metric (IPM) (Müller, 1997), we formulate our objective function such that the main player $f_{t+1}$ maximizes the

## Chosen: ground truth completion

In Southampton, what is the most popular form of transportation for commuters?

I don't have access to current data. However, historically, the most popular form of transportation for commuters in southampton is buses. The city also has a good network of trains and taxis.

## Rejected: model generation @Iter 0

In Southampton, what is the most popular form of transportation for commuters?

$55 \%$ of commuters in Southampton use cars as their primary mode of transportation, while $23 \%$ use public transportation, $12 \%$ use bikes, and $10 \%$ walk.

## Model generation @Iter 1

I do not have access to the latest updates and statistics of southampton. However, based on previous reports, the most popular form of transportation for commuters in southampton is the public transport network, which includes buses, trains, and ferries. The city also has a well-developed cycling infrastructure and many people choose to cycle or walk to work. Additionally, private cars and taxis are also commonly used for commuting.

Figure 1: Example of ground truth completion compared to the fine-tuned model generation at iteration 0 and 1 . We can observe that the model generation at iteration 0 , although fluent, incorrectly quantifies transportation preferences with specific percentages that are potentially hallucinations. The model generation at iteration 1 provides a qualitative summary of the transportation forms at Southampton without specific percentage, aligning more closely with the ground truth while adding more details.

expected value gap between the target data distribution $p_{\text {data }}$ and the opponent player's distribution $p_{\boldsymbol{\theta}_{t}}$ :

$$
\begin{equation*}
f_{t+1}=\underset{f \in \mathcal{F}_{t}}{\operatorname{argmax}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right] \tag{4.1}
\end{equation*}
$$

where $\mathcal{F}_{t}$ is a sequence of highly expressive function classes that we will determine in later deduction. The subscript $t$ in $\mathcal{F}_{t}$ is due to that the function class is dependent on $p_{\boldsymbol{\theta}_{t}}$. Given such a $f_{t+1}$ and a response sequence $\mathbf{y}$ to the prompt $\mathbf{x}$, the value of $f_{t+1}(\mathbf{x}, \mathbf{y})$ reflects the main player's degree of belief that $\mathbf{y}$ originates from $p_{\text {data }}$ rather than $p_{\boldsymbol{\theta}_{t}}$. Ideally, the main player $f_{t+1}$ should yield a high value when $\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})$ and a low value when $\mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$, where $p_{\boldsymbol{\theta}_{t}}$ is the opponent's distribution. Instead of solving (4.1), we can also solve the following more general optimization problem,

$$
\begin{equation*}
f_{t+1}=\underset{f \in \mathcal{F}_{t}}{\operatorname{argmin}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \tag{4.2}
\end{equation*}
$$

where $\ell(\cdot)$ is a loss function that is both monotonically decreasing and convex. For example, a linear loss function $\ell(t)=-t$ reduces (4.2) to the minimization version of (4.1). However, the use of a linear loss function results in an unbounded objective value, which, during continuous training, leads to a negative infinite value of $f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$ on the opponent player's responses. Therefore, in our work, we choose the logistic loss function $\ell(t):=\log (1+\exp (-t))$ for its non-negativity, smoothness, and exponentially decaying tail as $t \rightarrow \infty$. Such a choice of loss function aids in preventing the excessive growth in the absolute value of $f$.

Updating the Opponent Player. Previously we have discussed the training of $f_{t+1}$ given the opponent player's distribution $p_{\boldsymbol{\theta}_{t}}$. Now suppose we have optimized our main player $f_{t+1}$ that can distinguish $p_{\text {data }}$ from $p_{\boldsymbol{\theta}_{t}}$, within a certain function class $\mathcal{F}_{t}$, we elaborate how we get parameter $\boldsymbol{\theta}_{t+1}$
of the opponent player. Specifically, when presented with two responses $\mathbf{y}$ and $\mathbf{y}^{\prime}$ to the same prompt $\mathbf{x}, f_{t+1}$ assesses the values $f_{t+1}(\mathbf{x}, \mathbf{y})$ and $f_{t+1}\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$. It then infers that the response with the higher value is from the real data distribution $p_{\text {data }}$ and the response with lower value is attributed to the LLM $p_{\boldsymbol{\theta}_{t}}$. Subsequently, the objective of the opponent player is to find a better LLM that generates responses indistinguishable from $p_{\text {data }}$ for the main player. This is achieved by maximizing the expected value $\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p(\cdot \mid \mathbf{x})}\left[f_{t+1}(\mathbf{x}, \mathbf{y})\right]$. In addition, to prevent excessive deviation of $p_{\boldsymbol{\theta}_{t+1}}$ from $p_{\theta_{t}}$ and stabilize the self-play, we incorporate a Kullback-Leibler (KL) regularization term. Putting these together gives rise to the following optimization problem:

$$
\begin{equation*}
\underset{p}{\operatorname{argmax}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p(\cdot \mid \mathbf{x})}\left[f_{t+1}(\mathbf{x}, \mathbf{y})\right]-\lambda \mathbb{E}_{\mathbf{x} \sim q(\cdot)} \mathrm{KL}\left(p(\cdot \mid \mathbf{x}) \| p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})\right), \tag{4.3}
\end{equation*}
$$

where $\lambda>0$ is the regularization parameter. Notably, (4.3) has a closed-form solution $\widehat{p}(\cdot \mid \mathbf{x})$ :

$$
\begin{equation*}
\widehat{p}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \exp \left(\lambda^{-1} f_{t+1}(\mathbf{x}, \mathbf{y})\right) \tag{4.4}
\end{equation*}
$$

It is worth noting that $\widehat{p}(\cdot \mid \mathbf{x})$ is not guaranteed to be belong to the LLM space $\left\{p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}) \mid \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\}$. Since we hope that the closed-form solution $\widehat{p}$ in the probability space can be realized by an LLM with parameter $\boldsymbol{\theta}$, i.e., $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\widehat{p}(\mathbf{y} \mid \mathbf{x})$, solving for $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \exp \left(\lambda^{-1} f_{t+1}(\mathbf{x}, \mathbf{y})\right)$ gives $f_{t+1}(\mathbf{x}, \mathbf{y})=\lambda \cdot \log \frac{p_{\theta}(\cdot \mid \mathbf{x})}{p_{\theta_{t}}(\cdot \mid \mathbf{x})}$. This suggests the following function class $\mathcal{F}_{t}$ for $f_{t+1}$ :

$$
\begin{equation*}
\mathcal{F}_{t}=\left\{\left.\lambda \cdot \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{\mathrm{t}}(\mathbf{y} \mid \mathbf{x})}} \right\rvert\, \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\} \tag{4.5}
\end{equation*}
$$

where $\boldsymbol{\Theta}$ is the parameter space of LLMs being considered. Given the choice of $\mathcal{F}_{t}$ in (4.5), optimizing (4.2) gives $f_{t+1}$ parameterized by $\boldsymbol{\theta}_{t+1}$ in the following form:

$$
\begin{equation*}
f_{t+1}(\mathbf{x}, \mathbf{y})=\lambda \cdot \log \frac{p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{\mathrm{t}}}(\mathbf{y} \mid \mathbf{x})} \tag{4.6}
\end{equation*}
$$

Substituting (4.6) into (4.4) yields $\widehat{p}(\mathbf{y} \mid \mathbf{x})=p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$. In other words, $\boldsymbol{\theta}_{t+1}$ learned from (4.2) is exactly the LLM parameter for our ideal opponent selection.

End-to-end Training Objective. We integrate the previously discussed two steps into a single end-to-end training objective with an update rule of $\boldsymbol{\theta}_{t+1}$. Specifically, plugging (4.5) into (4.2) arrives at the update rule $\boldsymbol{\theta}_{t+1}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$, where $L_{\mathrm{SPIN}}$ is the training objective defined as follows

$$
\begin{equation*}
L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\mathrm{data}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \tag{4.7}
\end{equation*}
$$

We summarize the iterative self-play process of our method SPIN as follows,

$$
\cdots \rightarrow \underbrace{p_{\theta_{t}}(\cdot \mid \mathbf{x})}_{\text {Opponent Player at } t} \rightarrow \underbrace{\lambda \cdot \log \frac{p_{\theta_{t+1}}(\cdot \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\cdot \mathbf{x})}}_{\text {Main Player at } t+1} \rightarrow \underbrace{p_{\boldsymbol{\theta}_{t+1}}(\cdot \mid \mathbf{x})}_{\text {Opponent Player at } t+1} \rightarrow \cdots
$$

Namely, the opponent player chosen from the previous iteration $t$ is employed to train the main player at iteration $t+1$, resulting in the LLM parameterized by $\boldsymbol{\theta}_{t+1}$. Then we determine the next opponent player at iteration $t+1$ by directly copying the LLM parameter $\boldsymbol{\theta}_{t+1}$, which is then used in training the main player at iteration $t+2$. The detailed algorithm is presented in Algorithm 1 .

```
Algorithm 1 Self-Play Fine-Tuning (SPIN)
    Input: $\left\{\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)\right\}_{i \in[N]}$ : SFT Dataset, $p_{\boldsymbol{\theta}_{0}}$ : LLM with parameter $\boldsymbol{\theta}_{0}, T$ : Number of iterations.
    for $t=0, \ldots, T-1$ do
        for $i=1, \ldots N$ do
            Generate synthetic data $\mathbf{y}_{i}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{x}_{i}\right)$.
        end for
        Update $\boldsymbol{\theta}_{t+1}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} \sum_{i \in[N]} \ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{i} \mid \mathbf{x}_{i}\right)}{p_{\boldsymbol{\theta}_{+}}\left(\mathbf{y}_{i} \mid \mathbf{x}_{i}\right)}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{i}^{\prime} \mid \mathbf{x}_{i}\right)}{p_{\boldsymbol{\theta}_{+}}\left(\mathbf{y}_{i}^{\prime} \mid \mathbf{x}_{i}\right)}\right)$
    end for
    Output: $\boldsymbol{\theta}_{T}$.
```

Remark 4.1. (4.7) bears resemblance to direct preference optimization (DPO) (Rafailov et al., 2023) for RL fine-tuning. However, SPIN exhibits significant distinctions with DPO. Specifically, SPIN is applied to supervised fine-tuning (SFT) and relies solely on the SFT dataset, represented by pairs $(\mathbf{x}, \mathbf{y})$. In sharp contrast, DPO is designed for RL fine-tuning and necessitates a preference dataset, represented by $\left(\mathbf{x}, \mathbf{y}_{w}, \mathbf{y}_{l}\right)$, where $\mathbf{y}_{w}$ and $\mathbf{y}_{l}$ denote the winner (chosen) and loser (rejected) responses, respectively. DPO demands that, at the instance level, $\mathbf{y}_{w}$ is superior to $\mathbf{y}_{l}$. In comparison, our method requires that, at the distribution level, the target $p_{\text {data }}$ should be distinguishable from the weak LLM $p_{\boldsymbol{\theta}}$ before it becomes a strong one. In terms of algorithm design, DPO implements a single-iteration approach, while our method facilitates an iterative self-play strategy, as outlined in Algorithm 1.

## 5 Theoretical Analysis

In this section, we provide a theoretical analysis for Algorithm 1 in Section 4. Under monotonicity and convexity assumption of the objective function $\ell$, we show that the global optimum is obtained if and only if parameter $\boldsymbol{\theta}_{t}$ generates data distribution. We summarize our assumptions as follows:

Assumption 5.1. The loss function $\ell(t): \mathbb{R} \rightarrow \mathbb{R}$ is monotonically decreasing, i.e., $\forall t, \ell^{\prime}(t) \leq 0$ and satisfies $\ell^{\prime}(0)<0$. In addition, $\ell(t)$ is a convex function.

Assumption 5.1 holds for a wide range of loss functions commonly used in machine learning, including correlation loss $\ell(t)=1-t$, hinge loss $\ell(t)=\max (0,1-t)$, exponential loss $\ell(t)=\exp (-t)$ and logistic loss $\ell(t)=\log (1+\exp (-t))$. Under Assumptions 5.1, we present the following theorem, which is pivotal in understanding the optimization dynamics of our method.

Theorem 5.2. Under Assumption 5.1, suppose there exists $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, then we have that

- (Sufficiency) If $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, then $\boldsymbol{\theta}_{t}$ is the global minimum of (4.7) for any $\lambda \geq 0$.
- (Necessity) If $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x}) \neq p_{\text {data }}(\cdot \mid \mathbf{x})$, there exists an appropriately chosen $\lambda$, such that $\boldsymbol{\theta}_{t}$ is not the global minimum of (4.7).

Remark 5.3. Theorem 5.2 suggests that under certain conditions, the optimization process of our method naturally stops at the point $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, implying the effectiveness of our approach in aligning the LLM's distribution with the target data distribution. Moreover, Theorem 5.2 also indicates that the optimization process only stops when the global optimality is achieved, i.e., the LLM's distribution aligns with the target data distribution.

For the logistic loss function $\ell(t)=\log (1+\exp (-t))$, the following theorem gives a more precise characterization of the opponent player, enabling a better understanding of SPIN.

Theorem 5.4. Consider the choice of logistic loss $\ell(t)=\log (1+\exp (-t))$ in SPIN. Suppose that $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$ lies in the LLM space $\left\{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \mid \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\}$ and $\boldsymbol{\theta}_{t+1}$ is global minimum of $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$, then the opponent player at iteration $t+1$ satisfies

$$
p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}
$$

Remark 5.5. According to Theorem 5.4, the model update from $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ to $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$ tends to increase the probability $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$ when $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ is less than $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$, and decrease it when $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ is greater than $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Thus, Theorem 5.4 further confirms that our method's optimization process naturally converges to the point where $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ equals $p_{\text {data }}(\cdot \mid \mathbf{x})$. The update of the opponent player is controlled by $\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$, which is regulated by the factor $1 / \lambda$. A smaller $\lambda$ results in a larger change of the opponent player, while a larger $\lambda$ leads to a smaller change. Therefore, as $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ approaches $p_{\mathrm{data}}(\cdot \mid \mathbf{x})$, increasing $\lambda$ enhances the stability of LLM training. This observation aligns with (4.3), where $\lambda$ is the regularization parameter of the KL regularization that is employed to control the deviation of the opponent player.

## 6 Experiments

This section provides a detailed empirical analysis of SPIN. Our findings highlight several key points: (1) SPIN markedly enhances model performance across a wide range of evaluation benchmarks by breaking the limit of SFT; (2) even without introducing new human annotated data, SPIN at iteration 0 achieves performance on par to DPO training that utilizes even more data; (3) iterative training is a necessary component in SPIN as it breaks the limit of multi-epoch training.

### 6.1 Experiment Setup

Model and Datasets. In this study, we adopt zephyr-7b-sft-full as our base model. This model derives from the pre-trained Mistral-7B (Jiang et al., 2023) and has been further fine-tuned on the SFT dataset Ultrachat $200 \mathrm{k}^{1}$ by HuggingFace. Ultrachat200k represents a high-quality $200 \mathrm{k}$ subset of the larger UltraChat (Ding et al., 2023) corpus, which comprises approximately $1.4 \mathrm{M}$ dialogues produced using OpenAI's Turbo APIs. From UltraChat200k, We randomly sample 50k prompts and use the base model to generate the synthetic responses. We subsequently follow the optimization method described in Section 4.1 for further training. In multiple iterations, we leverage the synthetic data from the most recent iteration and add to the newly generated synthetic data, therefore resulting in a synthetic dataset size of $50 \mathrm{k}$ at iteration 0 and $100 \mathrm{k}$ at iteration 1,2 and 3 . At each iteration, we train our model for 2 epochs.

Evaluation. We employed the widely used Huggingface Open LLM Leaderboard (Beeching et al., 2023) as our evaluation benchmark, using the same Language Model Evaluation Harness library (Gao et al., 2023b). This leaderboard encompasses 6 different datasets, each focusing on a a specific capability of LLMs. Collectively, these datasets provide a thorough assessment framework, evaluating LLMs on commonsense reasoning (Arc (Clark et al., 2018), HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2021)), multi-task language understanding (MMLU(Hendrycks et al., 2020)), human falsehood mimic (TruthfulQA (Lin et al., 2021)) and math problem solving[^1]

(GSM8k (Cobbe et al., 2021)). In evaluation, the language models are prompted with few-shot in-context examples and the question. We follow the standard approach and report the average score across all datasets. In Table 1, we detail the evaluation setting adopted by both the leaderboard and our experiments. We leave further implementation details to Appendix B.

Table 1: Detailed information of HuggingFace Open LLM Leaderboard. For each evaluation dataset, we present the number of few-shot examples and metric adopted for evaluation.

| Datasets | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| \# few-shot | 25 | 0 | 5 | 5 | 10 | 5 |
| Metric | acc_norm | mc2 | acc | acc | acc_norm | acc |

### 6.2 SPIN Effectively Improves Benchmark Performance

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-10.jpg?height=653&width=683&top_left_y=926&top_left_x=713)

Figure 2: The average score of SPIN at different iterations on the HuggingFace Open LLM leaderboard datasets. For "SFT", we report the performance of our base model zephyr-7b-sft-full, which has been fine-tuned on the same dataset we use to generate synthetic data.

We demonstrate the effectiveness of SPIN using HuggingFace Open LLM Leaderboard as a wide range of evaluation. In Table 2, we compare the performance of our fine-tuned model by SPIN after iterations 0 to 3 with the base model zephyr-7b-sft-full. We can observe that SPIN exhibits remarkable effectiveness in improving the model's performance by further leveraging the SFT dataset, on which the base model has already been fully fine-tuned. At iteration 0 , where model responses are generated from zephyr-7b-sft-full, we observe an overall improvement of $2.66 \%$ on the average score. The improvement is particularly significant on the TruthfulQA and GSM8k benchmarks, with improvement exceeding $5 \%$ and $10 \%$ respectively. At iteration 1, we employ the LLM model from iteration 0 to generate new responses for SPIN, adhering to the procedure outlined in Algorithm 1. This iteration yields further enhancements of $1.32 \%$ on average, and especially significant on the Arc Challenge and TruthfulQA benchmarks. Subsequent iterations continue this trend of incremental improvement across various tasks. Meanwhile, the improvement at iteration $t+1$ is naturally smaller than that at iteration $t$. As the iterative training progresses, the degree of improvement gradually
approaches zero, suggesting that the model has reached a limiting point in the last iteration.

Table 2: Test performance of SPIN based on zephyr-7b-sft-full across HuggingFace Open LLM Leaderboard datasets. We also denote the average improvement over last iteration in the Average column.

| Model | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 58.14 |
| SPIN iteration 0 | 63.40 | 49.18 | 72.69 | 35.10 | 84.38 | 60.03 | $60.80_{(+2.66)}$ |
| SPIN iteration 1 | 65.19 | 55.17 | 72.30 | 35.78 | 84.96 | 59.34 | $62.12_{(+1.32)}$ |
| SPIN iteration 2 | 65.96 | 54.91 | 73.56 | 38.06 | 85.41 | 59.93 | $62.97_{(+0.85)}$ |
| SPIN iteration 3 | 65.87 | 54.90 | 73.72 | 38.97 | 85.54 | 59.99 | $63.16_{(+0.19)}$ |

Comparison with DPO. zephyr-7b-beta is a model derived from zephyr-7b-sft-full, trained with DPO on approximately $62 \mathrm{k}$ preference data. This data, the UltraFeedback Binarized dataset(Cui et al., 2023) ${ }^{2}$, comprises both chosen and rejected completions evaluated by GPT-4. We note that, DPO requires either human input or advanced language model feedback to determine the preference, making data generation a rather expensive procedure. In contrast, our SPIN only requires the initial model itself. Moreover, unlike DPO which requires new data source, our method exclusively leverages the existing SFT dataset. In Figure 3, we show the performance comparison of SPIN at iterations 0 and 1 (employing 50k SFT data) with DPO training, from the same SFT checkpoint. We can observe that, while DPO leverages more data from new sources, SPIN based on the existing SFT data can already achieve comparable average performance to DPO training at iteration 0. From iteration 1, SPIN even surpasses the performance of DPO on the leaderboard benchmark.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-11.jpg?height=496&width=1526&top_left_y=1411&top_left_x=294)

Figure 3: Performance comparison with DPO training across the six benchmark datasets. Self-play at iteration 0 achieves comparable performance to DPO training with $62 \mathrm{k}$ new data. At iteration 1 , self-play has already surpassed DPO training on the majority of datasets.

### 6.3 Ablation Studies

In this subsection, we examine the effect of synthetic dataset size and training epochs within an iteration. Our analysis demonstrates the effectiveness of the synthetic data used by SPIN compared to the SFT data, as well as the necessity of iterative training in SPIN. Furthermore, to comprehensively[^2]assess the performance improvements of SPIN, we perform additional evaluations on benchmark tasks distinct from those in the Open LLM leaderboard.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-12.jpg?height=647&width=967&top_left_y=365&top_left_x=579)

Figure 4: The scaling effect of training size of SPIN compared to SFT on the average score of Open LLM Leaderboard. For SPIN, we consider training data of sizes $14 \mathrm{k}, 26 \mathrm{k}$ and $50 \mathrm{k}$ where the larger dataset contains the smaller dataset. The starting point for SPIN (with x-axis 0) is the zephyr-7b-sft-full checkpoint, which has been fine-tuned on Ultrachat200k for 1 epoch. We report the model performance trained for 1 epoch with SPIN on the varying sizes of dataset. We additionally compare with SFT, where we fine-tune Mistral-7B on Ultrachat200k for 3 consecutive epochs and report the model performance at the first epoch as the starting point (with x-axis 0 ).

Training Size. We investigate the effect of varying training data size on the performance of SPIN. In Figure 4, we demonstrate the effect of training size for SPIN during iteration 0 and additionally compare with SFT with the full original dataset. Specifically, for the SFT baseline, we fully fine-tune Mistral-7B on Ultrachat200k for three epochs and report first epoch performance as the starting point (with x-axis 0) in the figure for SFT. For SPIN, we report the zephyr-7b-sft-full checkpoint as the starting point, which has also been fine-tuned on Ultrachat200k for one epoch. We select the training size of SPIN at iteration 0 to be $14 \mathrm{k}, 26 \mathrm{k}$, and $50 \mathrm{k}$ and generate the data accordingly, ensuring that the larger dataset encompasses the smaller dataset. The performance of SPIN was then evaluated after 1 epoch of self-play fine-tuning for each training size. We can observe that, while SPIN results in notable improvement with increasing training sizes, SFT on further epochs 2 and 3 fails to yield more than $1 \%$ improvement. Lastly, in Table 3, we also show the performance of SFT from zephyr-7b-sft-full on Ultrachat200k for one epoch. While self-play fine-tuning with synthetic data from zephyr-7b-sft-full effectively improves its performance, simply fine-tuning it again on the SFT data leads to degraded performance, as similarly observed in Figure 4.

Iterative Training v.s. Training for More Epochs. We further study the training within iteration 0 and compare with the performance achieved in iteration 1 , particularly contrasting the test performance obtained from extended training duration with that from next iteration. Figure 5 depicts the performance trajectory of the model trained using SPIN over multiple epochs at iteration 0 . It is evident that the most substantial improvement occurs during the first two epochs, followed by only modest gains in subsequent epochs. Notably, SPIN exhibits robustness and stability; extending

Table 3: Test performance of zephyr-7b-sft-full fine-tuned on Ultrachat200k for 1 more epoch across HuggingFace Open LLM benchmark datasets. SFT fails to further leverage the fine-tuning data for performance enhancement and even results in degraded performance.

| Model | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 58.14 |
| SFT epoch 1 | 57.76 | 44.39 | 75.77 | 25.85 | 81.69 | 57.89 | 57.23 |

the training duration does not diminish performance but rather maintains a rather consistent level. Nevertheless, the observation suggests an inherent limitation to the performance achievable within a single iteration, thereby underscoring the necessity for iterative training. As shown by the test performance achieved at iteration 1 in the figures, extending the training in iteration 0 fails to reach the performance comparable to iteration 1 .

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=505&width=1529&top_left_y=932&top_left_x=295)

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=414&width=504&top_left_y=953&top_left_x=304)

(a) Arc Challenge accuracy.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=414&width=512&top_left_y=953&top_left_x=801)

(b) TruthfulQA score.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=420&width=509&top_left_y=950&top_left_x=1312)

(c) Average score.

Figure 5: The SPIN training dynamics of zephyr-7b-sft-full on the $50 \mathrm{k}$ synthetic data with regard to the number of training epochs during iteration 0 . We can observe that iterative training is pivotal as training for more epochs during iteration 0 reaches a limit and cannot surpass iteration 1.

Further Investigation on More Tasks. Here, we further investigate the performance of SPIN on a broader variety of tasks, including MT-Bench (Zheng et al., 2023), Big-Bench (bench authors, 2023) and OpenBookQA (Mihaylov et al., 2018) in addition to the Open LLM Leaderboard tasks. Specifically, we use the following tasks from Big-Bench-Hard for a more comprehensive evaluation, including Causal Judgment (causal reasoning), Sports Understanding (commonsense reasoning) and Formal Fallacies (logical reasoning). In Table 4, we show the resulting scores of SPIN on MT-Bench as well as those tasks from Big-Bench. In Figure 6, we detail the model performances on MT-Bench with regard to different types of questions. We can see a notably robust improvement in the performance of SPIN on various tasks besides the HuggingFace Benchmark, without major degradation. Notably, on MT-Bench, the model fine-tuned by SPIN has surpassed the performance of vicuna-13b-v1.5 (Chiang et al., 2023) with a score of 6.57 .

## 7 Conclusion and Discussion

This paper introduces a novel fine-tuning method SPIN, to convert a weak LLM to a strong LLM by unleashing the full power of human-annotated data. Central to this method is a self-play mechanism,

Table 4: Test performance on other reasoning benchmark datasets for SPIN at different iterations and zephyr-7b-sft-full. We report the average score for MT-Bench and the accuracy score for Big Bench datasets under standard few-shot CoT evaluation. On OpenBookQA, we report acc_norm with 1-shot example as used in Anil et al. (2023). As similar to Open LLM Leaderboard evaluation, we observe a steady improvement in performance on the other benchmark tasks, with no significant degradation.

| Model | MT-Bench | BB-causal | BB-formal | BB-sports | OpenBookQA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 5.94 | 56.15 | 49.6 | 96.0 | 45.4 |
| SPIN iteration 0 | $6.46_{(+0.52)}$ | 57.75 | 51.6 | 95.2 | 46.8 |
| SPIN iteration 1 | $6.65_{(+0.19)}$ | 58.82 | 51.2 | 95.2 | 47.2 |
| SPIN iteration 2 | $6.78_{(+0.13)}$ | 59.36 | 51.2 | 94.4 | 47.6 |

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-14.jpg?height=623&width=1003&top_left_y=865&top_left_x=558)

Figure 6: Model performance on MT-Bench. We compare SPIN across different iterations with the base SFT model. Starting from iteration 1, our fine-tuned model by SPIN robustly outperforms the SFT checkpoint on all evaluation aspects.

wherein a main player (the LLM) is fine-tuned to differentiate the responses of opponent player (the LLM from previous iteration) from the target data distribution, and the LLM is iteratively aligned with the target data distribution. Therefore, SPIN facilitates the LLM's iterative self-evaluation and enhancement through self-play. In comparison to supervised fine-tuning and RL fine-tuning methods, SPIN enables the LLM to self-improve without additional human data or feedback from stronger LLMs. Empirical results demonstrate that SPIN significantly enhances LLM performance across diverse benchmarks, even outperforming models trained with additional human data or AI feedback.

Limitation and Future Work. Our theoretical results demonstrate that the optimization process of SPIN converges if and only if the LLM's distribution aligns with $p_{\text {data }}$. Therefore, our study focuses on a fixed target data distribution generated by humans, which inherently imposes a ceiling on the performance of fine-tuned LLM. Exploring the dynamically changing target data distribution is an important direction to overcome this limitation and elevate the LLM's performance beyond this ceiling or even to a super-human level. Moreover, considering the resource demands of synthetic data generation, another promising avenue for further exploration is to reduce the volume of required
synthetic data.

## A Further Related Work

Curriculum Learning. In deep learning, it has been observed that training models using data samples arranged in a strategically meaningful order can lead to improved performance compared to training on randomly shuffled data. This approach is commonly known as curriculum learning (Bengio et al., 2009; Soviany et al., 2022). Initial studies in curriculum learning introduced efficient algorithms that adhere to an 'easy-to-hard' progression (Spitkovsky et al., 2009; Kumar et al., 2010; Lee and Grauman, 2011; Zhang et al., 2015). In the field of Natural Language Processing (NLP), criteria such as sentence length and term frequency are commonly utilized (Cirik et al., 2016; Zhang et al., 2018; Liu et al., 2018). More recent developments include the application of curriculum learning algorithms in multi-modal learning (Liu et al., 2021; Wu et al., 2022). Our work shares a similar idea to curriculum learning, wherein the training data evolves iteratively-beginning with responses that are easy to distinguish from human-annotated data and gradually progressing to more challenging instances.

Generative Adversarial Learning. Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) represent a distinct class of generative models, characterized by their unique adversarial process. To enhance training stability and data quality, Mao et al. (2017) introduced the Least Squares GAN, employing a least squares loss function for the discriminator. A significant advancement in GANs involves the use of Integral Probability Metrics (IPM) (Müller, 1997), particularly highlighted in the development of Wasserstein GAN by Arjovsky et al. (2017). This model employs IPM in its loss design, enhancing training stability. Since then, IPMs have become popular in the design of GANs (Mroueh and Sercu, 2017; Gulrajani et al., 2017), particularly in constraining the discriminator to a specific function class, thereby preventing it from overpowering the generator. Furthermore, Jolicoeur-Martineau (2018) generalized IPM-based GANs by introducing relativistic discriminator and proposed Relativistic GAN. It is worth noting that the objective function defined in our (4.2) is similar to Relativistic GAN (Jolicoeur-Martineau, 2018) and reduces to an IPM framework in Wasserstein GAN (Arjovsky et al., 2017) with a linear loss. However, our approach differs in both the choice of the function class and the training procedure. Inspired by GAN, Cheng et al. (2023) proposed an adversarial learning framework named Adversarial Preference Optimization (APO) that trains the LLM and a reward model in an adversarial game. Our method is also related to Generative Adversarial Imitation Learning (GAIL) (Ho and Ermon, 2016), which trains separate discriminator and policy networks in each iteration for imitation learning. In contrast to the above methods, SPIN relies on self-play where both the main player and the opponent player are the same LLM from two consecutive iterations.

## B Experiment Details

## B. 1 Hyperparameters and Implementation Details

We use the Alignment Handbook library (Tunstall et al., 2023b) as the codebase for our selfplay fine-tuning method SPIN, which includes DeepSpeed ZeRO-3 (Rajbhandari et al., 2020) and FlashAttention-2 (Dao, 2023) to reduce training cost. We train our models with RMSProp (Hinton et al., 2012) optimizer with no weight decay for all iterations as commonly used in fine-tuning LLMs for alignment, with a global batch size of $64,10 \%$ warmup steps and bfloat16 precision. We set the peak learning rate to be $5 \mathrm{e}-7$ for iterations 0 and 1 , and decay this peak learning rate to $1 \mathrm{e}-7$ for
iteration 2 and 3 as we are approaching the end of self-play fine-tuning. Lastly, we choose $\beta=0.1$ and max sequence length to be 2048 tokens as in Tunstall et al. (2023b). We note that at the last iteration (iter-3) where the model is close to convergence, we increase the value of $\beta$ to 5.0. We use the Accelerate library (Gugger et al., 2022) to generate our synthetic data using distributed inference with multiple GPUs with a global batch size of 64 . We consider the prompting template "\#\#\# Instruction: $\{$ prompt\} $\} \mathrm{n} \backslash \mathrm{n} \# \# \#$ Response: " as commonly used in Taori et al. (2023). For Ultrachat200k containing multi-round conversations, we only sample the first round as our prompt and ground truth completion pairs.

## B. 2 Generation Examples

In Tables 5 and 6, we further provide the generation examples of our fine-tuned model by SPIN from different iterations. We can observe an improvement in response quality as compared to the generation of the SFT checkpoint. Meanwhile, the model generations at higher iterations typically becomes more concise than iteration 0 and resemble the ground truth completion better.

## C Proof of Theorems in Section 5

## C. 1 Proof of Theorem 5.2

Proof of Theorem 5.2. To begin with, we prove the "Sufficiency" in Theorem 5.2. Since $p_{\text {data }}(\cdot \mid \mathbf{x})=$ $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$, by symmetry property of $\mathbf{y}$ and $\mathbf{y}^{\prime}$, we have for any $\boldsymbol{\theta} \in \boldsymbol{\Theta}$ that

$$
\begin{aligned}
2 L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)= & \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& +\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y}^{\prime} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
= & \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right. \\
& \left.\quad+\left(\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}-\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)\right] \\
\geq & 2 \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}} \cdot(\cdot \mathbf{x})}\left[\ell \left(\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right.\right. \\
& \left.\left.\quad+\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}-\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)\right] \\
= & 2 \ell(0),
\end{aligned}
$$

where the inequality is due to Jensen's inequality (recalling that $\ell$ is convex in Assumption 5.1). Therefore, we have that $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right) \geq \ell(0)=L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}_{t}, \boldsymbol{\theta}_{t}\right)$, which means that $\boldsymbol{\theta}_{t}$ is the global optimum of (4.7). As a consequence, the gradient at the point $\boldsymbol{\theta}_{t}$ is zero, which concludes $\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_{t}$.

Next, we prove the "Necessity". Define $g(\lambda)$ as follows:

$$
g(\lambda)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\mathrm{data}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right]
$$

Then we have $g(0)=\ell(0)$ and

$$
g^{\prime}(0)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell^{\prime}(0)\left(\log \frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}-\log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right]
$$

Table 5: Generation example of our fine-tuned model by SPIN at different iterations.

| Prompt | How does the economic growth of Horsham compare to nearby towns? |
| :--- | :--- |
| Ground truth | I do not have access to the latest economic data on horsham and nearby <br> towns. However, you can check the latest economic statistics and reports <br> provided by the local government or the official websites of horsham and <br> nearby towns for a detailed comparison of their economic growth. |

zephyr-7b=sft-full $\mid$ According to the given material, Horsham was growing "significantly" faster than nearby towns, indicating that it was economically thriving. $\backslash \mathrm{n}$ However, the given material does not specify the exact rates or levels of economic growth of Horsham compared to nearby towns. Without this information, it is challenging to determine whether the differences in economic growth between Horsham and nearby towns were significant or minimal.

| Iteration 0 | I do not have access to the latest economic data for horsham and nearby |
| :--- | :--- |

towns. However, horsham is known to be a prosperous town in west sussex, uk, with a diverse economy that includes agriculture, manufacturing, and services. The town has a strong retail sector, and it is also home to several large companies, including the global pharmaceutical company novartis. In terms of comparison with nearby towns, horsham is generally considered to be one of the more affluent towns in the region. However, the economic growth of horsham and its neighboring towns may vary depending on various factors such as industry, investment, and government policies.

Iteration 1

I do not have access to real-time data or information about the most recent economic growth of horsesham and nearby towns. However, according to the recent reports from various sources such as the office for national statistics, horsesham's economy primarily relies on retail, healthcare, education, and public administration industries. Compared to nearby towns such as lewes, chichester, and bognor regis, horsesham's economy may have a different composition and growth rate based on their specific industries and economic policies.

Iteration 2

I don't have access to the most recent economic data. However, according to a 2019 report by pwc, horsham's expected economic growth up to 2025 was projected to be around $2.3 \%$ per annum, which is slightly below the average for the south east of england (2.8\%). Neighbouring towns like crawley and bognor regis were projected to experience similar growth rates. note: this may not reflect the impact of the covid-19 pandemic that occurred in 2020 .

$$
=\ell^{\prime}(0)\left(\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log \frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right]-\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\log \frac{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right]\right)
$$

Table 6: Another generation example of our fine-tuned model by SPIN at different iterations.

| Prompt | Can you clarify if Poppelsdorfer Allee 45 and the old site of the Mathe- <br> matical Institute are the same location? Answer according to: Go straight <br> to Poppelsdorfer Allee 45 and sign-in. $\$ n There is a big lecture hall at the <br> old site of the Mathematical Institute (Wegelerstr. 10, Bonn). |
| :--- | :--- |
| Ground truth | No, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are not the same location. To sign-in, one should <br> go straight to Poppelsdorfer Allee 45. However, there is a big lecture hall <br> at the old site of the Mathematical Institute. |
| zephyr-7b=sft-full | Yes, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> are the same location. |
| Iteration 0 | Yes, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are the same location. The Mathematical Institute <br> has been relocated and renamed as the Institute for Mathematics and <br> Theoretical Physics, but the building where it used to be is still referred <br> to as the old site. The big lecture hall at the old site still exists and may <br> be used for events or classes. |
| Iteration 1 | No, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> are not the same location. While Poppelsdorfer Allee 45 is where attendees <br> should sign-in, the old site of the Mathematical Institute (located at <br> Wegelerstr. 10, Bonn) has a large lecture hall mentioned in the statement. <br> Therefore, it can be assumed that the two locations are separate and not <br> the same. |
| Iteration 2 | Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are two different locations. As per the instruction, <br> you need to go straight to Poppelsdorfer Allee 45 and sign-in. However, <br> there is a lecture hall mentioned at the old site of the Mathematical <br> Institute (Wegelerstr. 10, Bonn). |

$$
\begin{aligned}
& =\ell^{\prime}(0)\left[\mathrm{KL}\left(p_{\text {data }}(\cdot \mid \mathbf{x})|| p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})\right)+\mathrm{KL}\left(p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})|| p_{\text {data }}(\cdot \mid \mathbf{x})\right)\right] \\
& <0
\end{aligned}
$$

where the last inequality is due to the condition that $\ell^{\prime}(0)<0$. Therefore, there exist a $\lambda_{0}$ such that for all $0<\lambda<\lambda_{0}$, we have $g(\lambda)<\ell(0)$. Choose $\boldsymbol{\theta}^{*}$ such that $p_{\boldsymbol{\theta}^{*}}(\mathbf{y} \mid \mathbf{x})=p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. For those $0<\lambda<\lambda_{0}$, we have that

$$
\begin{aligned}
L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}^{*}, \boldsymbol{\theta}_{t}\right) & =\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\boldsymbol{\theta}^{*}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}^{*}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\boldsymbol{\theta}^{*}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& =\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& =g(\lambda)
\end{aligned}
$$

$$
\begin{aligned}
& <g(0) \\
& =L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}_{t}, \boldsymbol{\theta}_{t}\right)
\end{aligned}
$$

where the second equality holds by the choice of $p_{\boldsymbol{\theta}^{*}}(\cdot \mid \mathbf{x})$, and the inequality holds due to the choice of $\lambda$. Therefore, we conclude that $\boldsymbol{\theta}_{t}$ is not the global optimum of (4.7) if $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x}) \neq p_{\text {data }}(\cdot \mid \mathbf{x})$.

## C. 2 Proof Theorem 5.4

We need the following auxiliary lemma before we prove Theorem 5.4.

Lemma C.1. Suppose that $\ell(t)=\log (1+\exp (-t))$ and for $a, b>0$, the following inequality holds

$$
a \ell(t)+b \ell(-t) \geq a \log (1+b / a)+b \log (1+a / b)
$$

the equality holds if and only if $t=\log (a / b)$.

Proof of Lemma C.1. Define $g(t)=a \ell(t)+b \ell(-t)=a \log (1+\exp (-t))+b \log (1+\exp (t))$, then we have

$$
g^{\prime}(t)=-\frac{a \exp (-t)}{1+\exp (-t)}+\frac{b \exp (t)}{1+\exp (t)}=\frac{-a+b \exp (t)}{1+\exp (t)}
$$

Therefore, $g^{\prime}(t)<0$ when $t<\log (a / b), g^{\prime}(t)>0$ when $t>\log (a / b)$, which indicates that $g$ achieves it minimum at $t=\log (a / b)$ which concludes the proof.

Lemma C. 1 shows that the global minimum of $a \ell(t)+b \ell(-t)$ is achieved when $t=\log (a / b)$. Based on Lemma C.1, we can further prove that (4.2) with the logistic loss function has a closed-form solution if we ignore the constraint set $\mathcal{F}_{t}$.

Lemma C.2. Denote $p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime}, \mathbf{x}\right)=q(\mathbf{x}) \cdot p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)$ and $p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime}, \mathbf{x}\right)=q(\mathbf{x}) \cdot p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)$. $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$,

$$
\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \geq \log 2-\operatorname{JSD}\left(p_{+} \| p_{-}\right)
$$

where $\operatorname{JSD}\left(p_{+} \| p_{-}\right)$represents the Jensen-Shannon divergence which is defined as follows

$$
\operatorname{JSD}(p \| q)=\frac{1}{2} \mathrm{KL}\left(p \| \frac{p+q}{2}\right)+\frac{1}{2} \mathrm{KL}\left(q \| \frac{p+q}{2}\right)
$$

where $\mathrm{KL}(\cdot \| \cdot)$ is KL-divergence. JSD is always non-negative and equals zero if and only if $p_{+}$and $p_{-}$are identical. Moreover, the global minimum value $\log 2-\operatorname{JSD}\left(p_{+} \| p_{-}\right)$is achieved by $f^{*}$ if and only if,

$$
f^{*}(\mathbf{x}, \mathbf{y})=Z(\mathbf{x})+\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

where $Z(\mathbf{x})$ is any function that is possibly dependent on $\mathbf{x}$.

Proof of Lemma C.2. We rewrite the objective function in the following formula,

$$
2 \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right]
$$

$$
\begin{aligned}
& =\int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] d \mathbf{y} d \mathbf{y}^{\prime} \\
& +\int q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left[\ell\left(f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-f(\mathbf{x}, \mathbf{y})\right)\right] d \mathbf{y} d \mathbf{y}^{\prime} \\
& =\int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \ell\left(f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-f(\mathbf{x}, \mathbf{y})\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& \stackrel{(i)}{\geq} \int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \log \left(1+\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) d \mathbf{y} d \mathbf{y}^{\prime}
\end{aligned}
$$

where the inequality is due to $a \ell(t)+b \ell(-t) \geq a \log (1+b / a)+b \log (1+a / b)$ in Lemma C. 1 with $a=q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right), b=q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}), t=f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$. The equality (i) holds if and only if the following equation holds almost surely for any $\mathbf{x}, \mathbf{y}, \mathbf{y}^{\prime}$,

$$
\begin{equation*}
f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)=\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) \tag{C.1}
\end{equation*}
$$

Equation (C.1) is equivalent to

$$
f(\mathbf{x}, \mathbf{y})-\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-\log \left(\frac{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)
$$

holds almost surely for any $\mathbf{x}, \mathbf{y}, \mathbf{y}^{\prime}$. Therefore, the equality (i) holds if and only if there exists some $Z(\mathbf{x})$ such that

$$
f(\mathbf{x}, \mathbf{y})=Z(\mathbf{x})+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

Recall that $p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)=p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ and $p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)=p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Then, the right-hand side of (i) can be written as

$$
\begin{aligned}
& \int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \log \left(1+\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =\int p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)+p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =2 \log 2+\int p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(\frac{1 / 2\left[p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)+p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)\right]}{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(\frac{1 / 2\left[p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)+p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)\right]}{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =2 \log 2-\mathrm{KL}\left(p_{+} \| \frac{p_{+}+p_{-}}{2}\right)-\mathrm{KL}\left(p_{-} \| \frac{p_{+}+p_{-}}{2}\right) \\
& =2 \log 2-2 \cdot \operatorname{JSD}\left(p_{+} \| p_{-}\right)
\end{aligned}
$$

where the last equality is by the definition of JSD. This concludes the proof.

Lemma C. 2 provides a closed-form solution to (4.2) if we ignore the constraint set $\mathcal{F}_{t}$. If this closed-form solution belongs to $\mathcal{F}_{t}$, then it should also be the solution to (4.2). This observation is the key to the proof of Theorem 5.4.

Proof of Theorem 5.4. Under the condition of Theorem 5.4, there exists a $p_{\boldsymbol{\theta}}$ such that

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}
$$

Therefore, there exists a function $\widehat{Z}(\mathbf{x})$ such that

$$
\begin{equation*}
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\widehat{Z}(\mathbf{x}) \cdot p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda} \tag{C.2}
\end{equation*}
$$

Applying logarithm function on both side of (C.2) yields

$$
\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=\lambda \log \left(\frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) \in \mathcal{F}_{t}
$$

By Lemma C.2, $f^{*}(\mathbf{x}, \mathbf{y})=\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ is the global minimum of the following minimization problem,

$$
\begin{equation*}
\underset{f}{\operatorname{argmin}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] . \tag{C..3}
\end{equation*}
$$

Since $f^{*} \in \mathcal{F}_{t}, f^{*}(\mathbf{x}, \mathbf{y})=\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ is also the global optimum of the optimization problem (4.2),

$$
\underset{f \in \mathcal{F}_{t}}{\operatorname{argmin}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right]
$$

Therefore, we have proved that

$$
\begin{align*}
& \min _{f} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \\
& =\min _{f \in \mathcal{F}_{t}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \\
& =\min _{\boldsymbol{\theta} \in \boldsymbol{\Theta}} L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right) \tag{C.4}
\end{align*}
$$

Since $\boldsymbol{\theta}_{t+1}$ is the global minimum of $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$. Then by (C.4), $\lambda \log \left(\frac{p_{\boldsymbol{\theta}_{+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ should be the global minimum of problem (C.3). By Lemma C.2, there exists $Z(\mathbf{x})$ such that

$$
\lambda \log \left(\frac{p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=Z(\mathbf{x})+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

which leads to the result that $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$.

## References

Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z. et al. (2023). Palm 2 technical report. arXiv preprint arXiv:2305.10403 .

Anthony, T., Tian, Z. and Barber, D. (2017). Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems 30.

Arjovsky, M., Chintala, S. and Bottou, L. (2017). Wasserstein generative adversarial networks. In International conference on machine learning. PMLR.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q. Et al. (2021). Program synthesis with large language models. arXiv preprint arXiv:2108.07732 .

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T. et al. (2022a). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 .

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C. et al. (2022b). Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 .

Bansal, T., Pachocki, J., Sidor, S., Sutskever, I. and Mordatch, I. (2018). Emergent complexity via multi-agent competition. In International Conference on Learning Representations.

Beeching, E., Fourrier, C., Habib, N., Han, S., Lambert, N., Rajani, N., Sanseviero, O., Tunstall, L. and Wolf, T. (2023). Open llm leaderboard.

BENCH AUThors, B. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research .

Bengio, Y., Louradour, J., Collobert, R. and Weston, J. (2009). Curriculum learning. In Proceedings of the 26th annual international conference on machine learning.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. et al. (2020). Language models are few-shot learners. Advances in neural information processing systems 33 1877-1901.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S. et al. (2023). Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 .

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J. et al. (2023). Weak-to-strong generalization: Eliciting strong capabilities with weak supervision .

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G. et al. (2021). Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Cheng, P., Yang, Y., Li, J., Dai, Y. and Du, N. (2023). Adversarial preference optimization.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I. and Xing, E. P. (2023). Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality.

Christiano, P. F., Leike, J., Brown, T., Martic, M., LegG, S. and Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems 30.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S. et al. (2022). Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 .

Cirik, V., Hovy, E. and Morency, L.-P. (2016). Visualizing and understanding curriculum learning for long short-term memory networks. arXiv preprint arXiv:1611.06204 .

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C. and Tafjord, O. (2018). Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 .

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z. and Sun, M. (2023). Ultrafeedback: Boosting language models with high-quality feedback.

DAO, T. (2023). Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 .

Deng, Y., Zhang, W., Chen, Z. and Gu, Q. (2023). Rephrase and respond: Let large language models ask better questions for themselves. arXiv preprint arXiv:2311.04205 .

Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M. and Zhou, B. (2023). Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233 .

Frei, S., Zou, D., Chen, Z. and Gu, Q. (2022). Self-training converts weak learners to strong learners in mixture models. In International Conference on Artificial Intelligence and Statistics. PMLR.

Freund, Y. (1995). Boosting a weak learning algorithm by majority. Information and computation $121256-285$.

Freund, Y. and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences 55 119-139.

Gao, L., Schulman, J. and Hilton, J. (2023a). Scaling laws for reward model overoptimization. In International Conference on Machine Learning. PMLR.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'H, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., WANG, K. and Zou, A. (2023b). A framework for few-shot language model evaluation.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems 27 .

Grandvalet, Y. and Bengio, Y. (2004). Semi-supervised learning by entropy minimization. Advances in neural information processing systems 17.

Gugger, S., Debut, L., Wolf, T., Schmid, P., Mueller, Z., Mangrulkar, S., Sun, M. and Bossan, B. (2022). Accelerate: Training and inference at scale made simple, efficient and adaptable.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems $\mathbf{3 0}$.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J. (2020). Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 .

Hernandez-Leal, P., Kartal, B. and Taylor, M. E. (2018). Is multiagent deep reinforcement learning the answer or the question? a brief survey. learning 2122.

Hinton, G., Srivastava, N. and Swersky, K. (2012). Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on 142 .

Ho, J. and Ermon, S. (2016). Generative adversarial imitation learning. Advances in neural information processing systems 29.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. D. L., Bressand, F., Lengyel, G., Lample, G., Saulnier, L. et al. (2023). Mistral 7b. arXiv preprint arXiv:2310.06825 .

Jolicoeur-Martineau, A. (2018). The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734.

Josifoski, M., Sakota, M., Peyrard, M. and West, R. (2023). Exploiting asymmetry for synthetic training data generation: Synthie and the case of information extraction. arXiv preprint arXiv:2303.04132 .

Kearns, M. and Valiant, L. (1994). Cryptographic limitations on learning boolean formulae and finite automata. Journal of the ACM (JACM) 41 67-95.

Kou, Y., Chen, Z., Cao, Y. and Gu, Q. (2022). How does semi-supervised learning with pseudo-labelers work? a case study. In The Eleventh International Conference on Learning Representations.

Kumar, M., Packer, B. and Koller, D. (2010). Self-paced learning for latent variable models. Advances in neural information processing systems 23.

Lanctot, M., Zambaldi, V., Gruslys, A., Lazaridou, A., Tuyls, K., Pérolat, J., Silver, D. and Graepel, T. (2017). A unified game-theoretic approach to multiagent reinforcement learning. Advances in neural information processing systems $\mathbf{3 0}$.

LeE, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In ICML Challenges in Representation Learning Workshop.

Lee, Y. J. and Grauman, K. (2011). Learning the easy things first: Self-paced visual category discovery. In CVPR 2011. IEEE.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T. et al. (2022). Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems 35 3843-3857.

Li, Y., Bubeck, S., Eldan, R., Giorno, A. D., Gunasekar, S. and Lee, Y. T. (2023). Textbooks are all you need ii: phi-1.5 technical report.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A. et al. (2022). Competition-level code generation with alphacode. Science 378 1092-1097.

Lin, S., Hilton, J. and Evans, O. (2021). Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958 .

Liu, B., Bubeck, S., Eldan, R., Kulkarni, J., Li, Y., Nguyen, A., Ward, R. and Zhang, Y. (2023). Tinygsm: achieving $>80 \%$ on gsm8k with small language models. arXiv preprint arXiv:2312.09241 .

Liu, C., He, S., Liu, K., Zhao, J. et al. (2018). Curriculum learning for natural answer generation. In $I J C A I$.

LiU, F., Ge, S. and Wu, X. (2021). Competence-based multimodal curriculum learning for medical report generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers).

Luo, H., Sun, Q., Xu, C., Zhao, P., lou, J., Tao, C., Geng, X., Lin, Q., Chen, S. and Zhang, D. (2023). Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583 .

Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z. and Paul Smolley, S. (2017). Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision.

Mihaylov, T., Clark, P., Khot, T. and Sabharwal, A. (2018). Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

Mishra, S., Khashabi, D., Baral, C. and Hajishirzi, H. (2021). Cross-task generalization via natural language crowdsourcing instructions. arXiv preprint arXiv:2104.08773 .

Mroueh, Y. and Sercu, T. (2017). Fisher gan. Advances in neural information processing systems 30 .

MÜLLER, A. (1997). Integral probability metrics and their generating classes of functions. Advances in applied probability 29 429-443.

Muller, P., Omidshafiei, S., Rowland, M., Tuyls, K., Perolat, J., Liu, S., Hennes, D., Marris, L., Lanctot, M., Hughes, E. et al. (2019). A generalized training approach for multiagent learning. arXiv preprint arXiv:1909.12823 .

OpenAI (2023). Gpt-4 technical report.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 27730-27744.

Prasad, A., Stengel-Eskin, E. and Bansal, M. (2023). Rephrase, augment, reason: Visual grounding of questions for vision-language models. arXiv preprint arXiv:2310.05861 .

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. et al. (2019). Language models are unsupervised multitask learners. OpenAI blog 19 .

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290 .

Rajbhandari, S., Rasley, J., Ruwase, O. and He, Y. (2020). Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE.

Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J. et al. (2023). Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950 .

Sakaguchi, K., Bras, R. L., Bhagavatula, C. and Choi, Y. (2021). Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM 64 99-106.

SamulL, A. L. (1959). Some studies in machine learning using the game of checkers. IBM Journal of research and development $\mathbf{3}$ 210-229.

Samuel, A. L. (2000). Some studies in machine learning using the game of checkers. IBM Journal of research and development 44 206-226.

Schapire, R. E. (1990). The strength of weak learnability. Machine learning 5 197-227.

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T. et al. (2017a). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815 .

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A. et al. (2017b). Mastering the game of go without human knowledge. nature 550 354-359.

Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Liu, P. J., Harrison, J., Lee, J., Xu, K., Parisi, A. et al. (2023). Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585 .

Soviany, P., Ionescu, R. T., Rota, P. and Sebe, N. (2022). Curriculum learning: A survey. International Journal of Computer Vision 130 1526-1565.

Spitkovsky, V. I., Alshawi, H. and Jurafsky, D. (2009). Baby steps: How "less is more" in unsupervised dependency parsing. In NIPS 2009 Workshop on Grammar Induction, Representation of Language and Language Learning.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D. and Christiano, P. F. (2020). Learning to summarize with human feedback. Advances in Neural Information Processing Systems 33 3008-3021.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. and Hashimoto, T. B. (2023). Stanford alpaca: An instruction-following llama model.

Tesauro, G. et al. (1995). Temporal difference learning and td-gammon. Communications of the ACM $3858-68$.

Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y. et al. (2022). Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239 .

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S. et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N. et al. (2023a). Zephyr: Direct distillation of $1 \mathrm{~m}$ alignment. arXiv preprint arXiv:2310.16944 .

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rush, A. M. and Wolf, T. (2023b). The alignment handbook.

VAPniK, V. (1999). The nature of statistical learning theory. Springer science \& business media.

Victor, S., Albert, W., Colin, R., Stephen, B., Lintang, S., Zaid, A., Antoine, C., Arnaud, S., Arun, R., Manan, D. et al. (2022). Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Vinyals, O., Babuschkin, I., Chung, J., Mathieu, M., Jaderberg, M., Czarnecki, W., Dudzik, A., Huang, A., Georgiev, P., Powell, R., Ewalds, T., Horgan, D., Kroiss, M., Danihelka, I., Agapiou, J., Oh, J., Dalibard, V., Choi, D., Sifre, L., Sulsky, Y., Vezhnevets, S., Molloy, J., Cai, T., Budden, D., Paine, T., Gulcehre, C., Wang, Z., Pfaff, T., Pohlen, T., Yogatama, D., Cohen, J., McKinney, K., Smith, O., Schaul, T., Lillicrap, T., Apps, C., Kavukcuoglu, K., Hassabis, D. and Silver, D. (2019). AlphaStar: Mastering the Real-Time Strategy Game StarCraft II.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D. eT AL. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35 24824-24837.

Wu, J., Liang, Y., Akbari, H., Wang, Z., Yu, C. et al. (2022). Scaling multimodal pre-training via cross-modality gradient harmonization. Advances in Neural Information Processing Systems $3536161-36173$.

Yang, Y., Singh, A. K., Elhoushi, M., Mahmoud, A., Tirumala, K., Gloeckle, F., Rozière, B., Wu, C.-J., Morcos, A. S. and Ardalani, N. (2023). Decoding data quality via synthetic corruptions: Embedding-guided pruning of code data. arXiv preprint arXiv:2312.02418 .

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A. and LiU, W. (2023). Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284 .

Yuan, Z., Yuan, H., Li, C., Dong, G., Tan, C. and Zhou, C. (2023). Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825 .

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A. and Choi, Y. (2019). Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830 .

Zhang, D., Meng, D., Li, C., Jiang, L., Zhao, Q. and Han, J. (2015). A self-paced multipleinstance learning framework for co-saliency detection. In Proceedings of the IEEE international conference on computer vision.

Zhang, X., Kumar, G., Khayrallah, H., Murray, K., Gwinnup, J., Martindale, M. J., McNamee, P., Duh, K. and Carpuat, M. (2018). An empirical exploration of curriculum learning for neural machine translation. arXiv preprint arXiv:1811.00739 .

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. ET AL. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685 .

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P. and Irving, G. (2019). Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 .


[^0]:    *Equal contribution

    ${ }^{\dagger}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: chenzx19@cs.ucla.edu

    ${ }^{\ddagger}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: yihedeng@cs.ucla.edu

    ${ }^{8}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: hzyuan@cs.ucla.edu

    ${ }^{I}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: kaixuanji@cs.ucla.edu

    ${ }^{\|}$Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: qgu@cs.ucla.edu

[^1]:    ${ }^{1}$ https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

[^2]:    ${ }^{2}$ https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized

</end of paper 0>


<paper 1>
# Boost Your Own Human Image Generation Model via Direct Preference Optimization with AI Feedback 

Sanghyeon Na*<br>Kakao Brain<br>orca.na@kakaobrain.com

Yonggyu Kim<br>Kakao Brain<br>arthur.kyg@kakaobrain.com

Hyunjoon Lee ${ }^{\dagger}$<br>Kakao Brain<br>malfo.lee@kakaobrain.com


#### Abstract

The generation of high-quality human images through text-to-image (T2I) methods is a significant yet challenging task. Distinct from general image generation, human image synthesis must satisfy stringent criteria related to human pose, anatomy, and alignment with textual prompts, making it particularly difficult to achieve realistic results. Recent advancements in T2I generation based on diffusion models have shown promise, yet challenges remain in meeting human-specific preferences. In this paper, we introduce a novel approach tailored specifically for human image generation utilizing Direct Preference Optimization (DPO). Specifically, we introduce an efficient method for constructing a specialized DPO dataset for training human image generation models without the need for costly human feedback. We also propose a modified loss function that enhances the DPO training process by minimizing artifacts and improving image fidelity. Our method demonstrates its versatility and effectiveness in generating human images, including personalized text-to-image generation. Through comprehensive evaluations, we show that our approach significantly advances the state of human image generation, achieving superior results in terms of natural anatomies, poses, and text-image alignment.


## 1 Introduction

Recently, text-to-image (T2I) generation [44, 19, 40, 34, 41, 36] has made remarkable advancements with the emergence of diffusion models [21, 49, 16]. These advancements have enabled the generation of high-quality images from textual descriptions, enhancing the practicality and versatility of T2I applications. Among the various tasks achievable through T2I generation, human image generation (referred to as human generation) stands out as one of the most practical and in-demand tasks due to its significant applications in fields such as entertainment, virtual reality, and personalized media. For instance, users can utilize personalized T2I models $44,46,55,33,8,7]$ to create profile pictures that reflect their desired identity, demonstrating the practicality and appeal of these models.

Despite the impressive capabilities of current T2I models in image synthesis, generating human images still poses significant challenges. Often, the generated human images exhibit unrealistic anatomical structures and unnatural poses that fail to meet user preferences. A common approach to mitigate these issues is to fine-tune the base model using high-quality human images to create a model specialized for human generation. Although this strategy can improve the performance[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-02.jpg?height=505&width=1010&top_left_y=233&top_left_x=362)

(a) Samples of HG-DPO

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-02.jpg?height=499&width=358&top_left_y=236&top_left_x=1401)

(b) Various random samples

Figure 1: Human images generated using HG-DPO. In Figure 1a, HG-DPO enables the generation of human images with natural anatomies, poses, and alignment with prompts. It demonstrates its stability through various samples generated using multiple random seeds in Figure 1b. The prompts used for generating the images and additional samples are in the appendices.
![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-02.jpg?height=490&width=420&top_left_y=978&top_left_x=365)

(a) Improvements resulting from HG-DPO.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-02.jpg?height=491&width=569&top_left_y=974&top_left_x=1190)

(b) Personalized T2I with HG-DPO.

Figure 2: Effectiveness of HG-DPO. In Figure 2a. HG-DPO improves the target model by preventing it from generating (i) collapsed images, unnatural (ii) anatomies and (iii) poses, and (iv) images misaligned with text (red boxes). In Figure 2b, we generate an image with a more natural pose and the identity of the concept image at the top right by applying HG-DPO to a personalized T2I model [46].

of model to some extent, it is insufficient to fully satisfy the complex range of human preferences involved in human generation through the ine-tuning alone.

In this regard, Direct Preference Optimization (DPO) [39, 51], which directly aligns model outputs with human preferences, is a promising method to further improve the quality of human generation. DPO trains a model by providing both winning (preferred) and losing (not preferred) samples. This process guides the model to generate outputs that look more like the winning samples and less like the losing samples. Through this guidance of DPO, the model can be further improved by learning the human-related semantic differences between the winning and losing samples (e.g., the unnatural poses of the losing samples in contrast to the natural poses of the winning samples).

While adopting DPO for human generation shows promise, it presents significant challenges. One major hurdle is the necessity to construct a DPO dataset. For the T2I task, this dataset must include triplets, each consisting of a prompt and two images generated conditioned on that prompt. These images must be labeled as preferred or not preferred by humans. Given the high cost of human labeling, an alternative could be to utilize publicly available human preference datasets [23, 53], as done by the previous method that adopts DPO to the diffusion model [51].

However, we argue that utilizing the public datasets is not effective for improving human generation, particularly for models specifically tailored to this task. The public datasets generally comprise images of general subjects and not specifically human-focused images. Since our primary goal is human generation, it is essential to use a DPO dataset containing various prompts specifically related
to human subjects. Moreover, the images in the public datasets are typically generated with models different from the tailored model we aim to train (target model). Such images might feature distinct characteristics from those produced by the target model, potentially disrupting the training process by forcing the target model to mitigate these differences rather than enhancing human-specific attributes like anatomy and pose. This issue becomes more evident when considering practical scenarios for deploying human generation models. As frequently observed in user communities of generative models [1, 3], users often seek models that generate human images with specific styles suited to their needs (e.g., profile pictures of Asians with a particular resolution). In these scenarios, the tailored model produces images that are markedly different from those found in the public datasets. Given these practical considerations, it is crucial to construct the DPO dataset with images generated by the tailored model (in-distribution dataset), rather than relying on the public datasets (out-distribution dataset), to enhance the capabilities of the human generation model.

Motivated by this, we propose a novel method for constructing the in-distribution dataset. Instead of manual labeling, which requires expensive human involvement, we automatically generate the winning and losing images by leveraging the existing image preference metric [23]. This efficient dataset construction method allows for the creation of large-scale DPO datasets with meaningful human-related semantic differences between the winning and losing images without the need for expensive human labeling. Additionally, we propose a modified objective function for DPO that minimizes artifacts caused by unintended differences between the winning and losing images of our dataset. Consequently, we propose a method called HG-DPO to improve Human Generation through DPO, which incorporates the novel dataset construction method and the modified objective function. With HG-DPO, as shown in Figure 1, the T2I model can generate high-quality human images with natural anatomies and poses that align with the given texts. These results are achievable due to HG-DPO enhancing the target model in terms of (i) avoiding collapsed images, (ii) producing more natural anatomies, (iii) creating more natural poses, and (iv) generating images better aligned with the text as shown in Figure 2a.

Furthermore, HG-DPO can also be easily adapted to improve the image quality of applications related to human generation. For instance, we can generate personalized human images with the desired identity and better quality, as illustrated in Figure 2b, by adapting HG-DPO to a personalized T2I model [46]. This adaptiveness further enhances the practicality of HG-DPO.

## 2 Preliminaries

### 2.1 Reinforcement Learning from Human Feedback (RLHF)

RLHF [35] is widely employed in LLMs [31, 24, 54, 47, 29, 61, 10, 30, 12, 9] to align a model with human preferences. It consists of three stages. The first stage involves fine-tuning the pre-trained model on a dataset for a downstream task and obtain the fine-tuned model, $p_{s f t}$. The second stage involves training a reward model $r_{\phi}$ using a human preference dataset. It is constructed by using $p_{s f t}$ to generate a pair of samples for a condition $c$ and then annotating which sample is preferred by humans. Then, $r_{\phi}$ is trained to assign a higher score to an input that align well with human preferences. Finally, $r_{\phi}$ is utilized to train a model, $p_{\theta}$, using the following objective function:

$$
\begin{equation*}
\max _{p_{\theta}} \mathbb{E}_{c, x \sim p_{\theta}(x \mid c)}\left[r_{\phi}(x, c)\right]-\beta D_{K L}\left(p_{\theta}(x \mid c) \| p_{r e f}(x \mid c)\right) \tag{1}
\end{equation*}
$$

Both $p_{\theta}$ and $p_{\text {ref }}$ are initialized with $p_{s f t}$, but only $p_{\theta}$ is trained while $p_{\text {ref }}$ remains frozen. The first term of Eq. (1) encourages $p_{\theta}$ to maximize the reward, while the second term prevents $p_{\theta}$ from deviating too far from $p_{r e f}$, which is a regularization term.

### 2.2 Direct Preference Optimization (DPO)

DPO [39] is a method that directly trains $p_{\theta}$ using the human preference dataset without explicit reward modeling with the following loss function derived from Eq. (1):

$$
\begin{equation*}
\mathcal{L}_{D P O}\left(p_{\theta} ; p_{r e f}, \mathcal{D}\right)=-\mathbb{E}_{\left(c, x^{w}, x^{l}\right) \sim \mathcal{D}} \log \sigma\left\{\beta \log \frac{p_{\theta}\left(x^{w} \mid c\right)}{p_{r e f}\left(x^{w} \mid c\right)}-\beta \log \frac{p_{\theta}\left(x^{l} \mid c\right)}{p_{r e f}\left(x^{l} \mid c\right)}\right\} \tag{2}
\end{equation*}
$$

where $x^{w}$ and $x^{l}$ denote winning (more preferred) and losing (less preferred) samples with a condition $c$ for generating them. $\sigma$ denotes a sigmoid function. Diffusion-DPO [51] adpats DPO to the diffusion
model by deriving the loss function from Eq. (2) as

$$
\begin{align*}
& \mathcal{L}_{\text {Diff-DPO }}\left(\epsilon_{\theta} ; \epsilon_{r e f}, \mathcal{D}\right)=-\mathbb{E}_{\left(c, x^{w}, x^{l}\right) \sim \mathcal{D}, t \sim \mathcal{U}(0, T)} \log \sigma\{-\beta  \tag{3}\\
& \left.\left.\left\|\epsilon^{w}-\epsilon_{\theta}\left(x_{t}^{w}, c, t\right)\right\|_{2}^{2}-\left\|\epsilon^{w}-\epsilon_{r e f}\left(x_{t}^{w}, c, t\right)\right\|_{2}^{2}-\left(\left\|\epsilon^{l}-\epsilon_{\theta}\left(x_{t}^{l}, c, t\right)\right\|_{2}^{2}-\left\|\epsilon^{l}-\epsilon_{r e f}\left(x_{t}^{w}, c, t\right)\right\|_{2}^{2}\right)\right)\right\}
\end{align*}
$$

where $x_{t}^{*}$ is a noisy version of $x^{*}$ obtained by adding $\epsilon^{*} \sim \mathcal{N}(0, I)$ in the diffusion process of timestep t. $\epsilon_{\theta}$ and $\epsilon_{r e f}$ are initialized with pre-trained $\epsilon_{s f t}$, but only $\epsilon_{\theta}$ is trained. Eq. (3) encourages $\epsilon_{\theta}$ to generate images like $x^{w}$ while avoiding generating images like $x^{l}$ [51] by learning the differences between $x^{w}$ and $x^{l}$. Simultaneously, it regularizes $\epsilon_{\theta}$ to not deviate too far from $\epsilon_{r e f}$. Here, $\beta$ is a regularization weight which corresponds to the weight of the KL divergence term in Eq. (1).

## 3 HG-DPO: Improvement of Human Generation through DPO

Our objective is to improve the capabilities of a diffusion model, specifically Stable Diffusion 1.5 [41], in generating high-quality human portrait images. Since this model is not initially optimized for human portraits, we first fine-tune it using a dataset of high-resolution $(704 \times 1024)$ Asian portrait images. This results in a tailored model, $\epsilon_{s f t}$, which is capable of generating human images at an acceptable quality level. In addition, note that $\epsilon_{s f t}$ can be considered as a model customized to fulfill the specific requirements of users aiming to produce $704 \times 1024$ Asian portraits, aligning with the practical application scenarios for human generation discussed in Section 1 . We then apply our HG-DPO method to further enhance the performance of this fine-tuned target model, $\epsilon_{s f t}$.

### 3.1 Construction of a Human Dataset for DPO Using AI Feedback

In this section, we explain how to construct the dataset, $\mathcal{D}_{H G-D P O}$, for HG-DPO. We follow the approach used to construct Pick-a-Pic dataset [23] but with the following differences: (1) instead of using multiple backbone models [41, 36] to generate images as in Pick-a-Pic dataset [23], we use our tailored model $\epsilon_{s f t}$; (2) for each prompt, we generate $N$ images instead of two, and then select the most preferred and the least preferred images among them; (3) instead of using human feedback to select images, we use a model-based image preference metric. Below, we provide a detailed explanation of the dataset construction method and its motivation.

Step 1. Image pool generation. The first step is to generate images using a prompt set $\mathcal{P}=\left\{p_{i}\right\}_{i=1}^{D}$, where $p_{i}$ denotes a prompt and $D$ denotes the size of $\mathcal{D}_{H G-D P O}$. For $\mathcal{P}$, we use a subset of prompts from our Asian portrait images dataset. As explained in Section 2.2, DPO requires two images for each prompt, namely the winning and losing images. However, we propose to create an image pool by generating $N$ different images for each prompt, instead of generating precisely two images for each prompt. This approach increases the variety of images in the image pool, thus increasing the likelihood of selecting winning and losing images with more meaningful differences. Since DPO encourages the model to learn the differences between the winning and losing images, having meaningful differences between them is important. Formally, for a prompt $p_{i} \in \mathcal{P}$, we generate an image pool $\mathcal{X}_{i}$ of size $N$, which is defined as

$$
\begin{equation*}
\mathcal{X}_{i}=\left\{x_{i_{j}}\right\}_{j=1}^{N} \text { where } x_{i_{j}}=\mathrm{T} 2 \mathrm{I}\left(\epsilon_{s f t}, p_{i}, r_{i_{j}}\right) \tag{4}
\end{equation*}
$$

Here, T2I is a text-to-image generator with a random seed $r_{i_{j}}$ used to generate an image $x_{i_{j}}$. To generate $N$ different images, we employ $N$ different random seeds $\left\{r_{i_{j}}\right\}_{j=1}^{N}$.

Step 2. Model-based preference scoring. We then score the generated images with a given prompt $p_{i}$ using an image preference estimator as follows:

$$
\begin{equation*}
\mathcal{S}_{i}=\left\{s_{i_{j}}\right\}_{j=1}^{N} \text { where } s_{i_{j}}=f\left(x_{i_{j}}, p_{i}\right) \text { for } x_{i_{j}} \in \mathcal{X}_{i} \tag{5}
\end{equation*}
$$

where $\mathcal{S}_{i}$ represents the preference scores of the image pool $\mathcal{X}_{i}$, and $f$ represents the image preference estimator. In this case, we use PickScore [23] as the image preference estimator.

Step 3. Selection of winning and losing samples. Finally, we select the image with the highest preference score as the winning image $x_{i}^{w}$ and the image with the lowest preference score as the losing image $x_{i}^{l}$ from the image pool $\mathcal{X}_{i}$, then create a triplet $\left(p_{i}, x_{i}^{w}, x_{i}^{l}\right)$. By selecting images with

Table 1: Quantitative comparison to the baselines. It shows the win rates (\%) for HG-DPO compared to the baselines. HG-DPO demonstrates superiority over the baselines in almost all metrics.

| Baseline | PickScore | HPS | ImageReward | Aesthetic | CLIP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| vs. Pick-a-Pic-v2 + DPO | 83.05 | 80.68 | 75.71 | 62.71 | 68.93 |
| vs. Pick-a-Pic-v2 (H) + DPO | 85.31 | 78.53 | 75.14 | 63.28 | 68.36 |
| vs. HPD-v2 + DPO | 88.14 | 82.95 | 76.27 | 63.28 | 71.19 |
| vs. HPD-v2 (H) + DPO | 86.44 | 79.89 | 77.97 | 62.15 | 72.32 |
| vs. $\mathcal{D}_{H G-D P O}+\mathrm{SFT}$ | 74.58 | 78.16 | 70.06 | 50.85 | 67.23 |
| vs. $\mathcal{D}_{H G-D P O}+$ DPO $(\beta \times 4)$ | 76.27 | 74.71 | 70.62 | 46.89 | 63.84 |
| vs. $\mathcal{D}_{H G-D P O}+\mathrm{DPO}(\beta \times 20)$ | 81.36 | 82.29 | 75.14 | 63.84 | 66.10 |
| vs. AlignProp | 39.55 | 65.91 | 64.41 | 16.95 | 74.01 |

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-05.jpg?height=112&width=1374&top_left_y=730&top_left_x=381)

Figure 3: User study. It shows the win rates (\%) of HG-DPO against the target model and several baselines on human evaluations. HG-DPO outperforms them, demonstrating its effectiveness.

the highest and lowest preference scores, we can maximize the semantic differences between the two images that affect image preference, thereby enabling the model to better learn the features of the preferred images. Specifically, $x_{i}^{w}$ and $x_{i}^{l}$ are defined as follows:

$$
\begin{equation*}
\left.\left(x_{i}^{w}, x_{i}^{l}\right)=\left(\mathcal{X}_{i}\left[j_{w}\right], \mathcal{X}_{i}\left[j_{l}\right]\right) \text { where }\left(j_{w}, j_{l}\right)=\underset{j_{w} \in[1 \ldots N]}{\operatorname{argmax}} S_{i}, \underset{j_{l} \in[1 \ldots N]}{\operatorname{argmin}} S_{i}\right) \tag{6}
\end{equation*}
$$

By creating triplets for the all prompts of $\mathcal{P}$, we complete the dataset $\mathcal{D}_{H G-D P O}=\left\{\left(p_{i}, x_{i}^{w}, x_{i}^{l}\right)\right\}_{i=1}^{D}$.

### 3.2 DPO with Statistic Matching

With the dataset $\mathcal{D}_{H G-D P O}$, we can apply DPO to $\epsilon_{s f t}$ using Eq. (3) to obtain the updated model, $\epsilon_{\theta}$. Compared to $\epsilon_{s f t}, \epsilon_{\theta}$ shows significant improvement in terms of human pose, anatomy, and textural prompt following. However, $\epsilon_{\theta}$ generates images with the color shift artifact, as shown in Figure 4 ( $\mathrm{N}-20$ ). We assume that the color is one of the low-level styles of an image and can be captured by the channel-wise statistics of latent features. Under this assumption, we hypothesize that the gaps between the channel-wise statistics of the latents sampled using $\epsilon_{\theta}$ and $\epsilon_{s f t}$ is the direct cause of the color shift. More detailed discussion can be found in Section 5

Latent adaptive normalization (LAN). To verify our hypothesis, we designed an inferencetime statistics matching approach called latent adaptive normalization (LAN). If the gaps in the channel-wise statistics of the latents cause the color shift, then eliminating those gaps should resolve it. Let $h_{t g t}^{t-1}$ and $h_{s f t}^{t-1}$ denote the latents sampled from the same random noise using $\epsilon_{\theta}$ and $\epsilon_{s f t}$ at inference time with timestep $t$, respectively (i.e., $h_{t g t}^{t-1}=\operatorname{sampler}\left(\hat{h}_{t g t}^{t}, p, t, \epsilon_{\theta}\right)$ and $h_{s f t}^{t-1}=$ sampler $\left(h_{s f t}^{t}, p, t, \epsilon_{s f t}\right)$ where sampler denotes a inference-time latent sampler and $p$ denotes an inference prompt). We define LAN as follows:

$$
\begin{equation*}
\hat{h}_{t g t}^{t-1}=\left(\frac{h_{t g t}^{t-1}-\mu\left(h_{t g t}^{t-1}\right)}{\sigma\left(h_{t g t}^{t-1}\right)}\right) \sigma\left(h_{s f t}^{t-1}\right)+\mu\left(h_{s f t}^{t-1}\right) \tag{7}
\end{equation*}
$$

where $\mu$ and $\sigma$ calculate the channel-wise mean and standard deviation from the input, respectively. As shown in Figure 4 and Table 2 (N-20-LAN), LAN can effectively address the color shift artifact, empirically verifying our hypothesis.

Statistic matching loss. By aligning the statistics of latents, we can effectively remove the color shifts from the generated images. However, the problem with LAN is that it incurs additional costs since it requires sampling from both $\epsilon_{\theta}$ and $\epsilon_{s f t}$. To address this issue, we propose a method to train the model such that the latent statistics are aligned during training time. Let $l^{t}$ denote a noisy latent of a winning image generated by the forward diffusion process at timestep $t$ during training. Then, we sample $l_{t g t}^{t-1}$ and $l_{r e f}^{t-1}$, which represent the latents sampled using $\epsilon_{\theta}$ and $\epsilon_{r e f}$, respectively (i.e., $l_{t g t}^{t-1}=\operatorname{sampler}\left(l^{t}, p, t, \epsilon_{\theta}\right)$ and $l_{r e f}^{t-1}=\operatorname{sampler}\left(l^{t}, p, t, \epsilon_{r e f}\right)$ where $p$ denotes a prompt paired with

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-06.jpg?height=547&width=1395&top_left_y=244&top_left_x=365)

Prompt : Three people in the forest, sunset, wearing orange suits, intense lighting

Figure 4: Qualitative results. The images generated by our model, HG-DPO, in the rightmost column have the natural poses while aligning well with the given texts compared to other models.

Table 2: Degree of color shift. Percentage change (\%) of color in terms of hue, saturation and value of each model relative to the target model $\left(\epsilon_{s f t}\right)$. The higher this value, the more severe the color shift is. We can observe that HG-DPO ( $\mathrm{N}-20$-SML) effectively resolves the color shift occurring in $\mathrm{N}-20$.

|  | $\mathrm{N}-2$ | $\mathrm{~N}-10$ | $\mathrm{~N}-20$ | $\mathrm{~N}-20(\beta \times 4)$ | $\mathrm{N}-20(\beta \times 20)$ | $\mathrm{N}-20-\mathrm{LAN}$ | $\mathrm{N}-20-\mathrm{SML}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Hue | 6.59 | 20.94 | 24.93 | 17.62 | 1.39 | 0.20 | 1.67 |
| Saturation | 4.50 | 5.54 | 12.52 | 7.99 | 2.86 | 2.03 | 0.74 |
| Value | 1.32 | 3.94 | 3.43 | 3.01 | 0.69 | 0.61 | 1.05 |

$l^{t}$ ). Here, $\epsilon_{r e f}$ is initialized with $\epsilon_{s f t}$ and is frozen. Finally, we define the statistic matching loss as:

$$
\begin{equation*}
\mathcal{L}_{\text {stat }}\left(\epsilon_{\theta} ; \epsilon_{r e f}, \mathcal{D}\right)=\mathbb{E}_{\left(p, x^{w}\right) \sim D, t \sim \mathcal{U}(0, T)}\left[\left\|\mu\left(l_{\text {tgt }}^{t-1}\right)-\mu\left(l_{\text {ref }}^{t-1}\right)\right\|_{2}^{2}\right] \tag{8}
\end{equation*}
$$

which resolves the color shift without incurring additional costs at inference time. Although LAN matches both the mean and the standard deviation to verify the assumption that gaps in the channelwise statistics cause the color shift, $\mathcal{L}_{\text {stat }}$ does not match the standard deviation, as we found that the mean matching sufficiently resolves the color shift ( $\mathrm{N}-20$-SML in Figure 4 and Table 2).

In conclusion, to enhance Human Generation via DPO, we propose a method called HG-DPO which incorporates i) constructing $\mathcal{D}_{H G-D P O}$ and ii) training $\epsilon_{\theta}$, which is initialized with $\epsilon_{s f t}$, with the following loss function:

$$
\begin{equation*}
\mathcal{L}_{H G-D P O}\left(\epsilon_{\theta} ; \epsilon_{r e f}, \mathcal{D}_{H G-D P O}\right)=\mathcal{L}_{\text {Diff-DPO }}\left(\epsilon_{\theta} ; \epsilon_{r e f}, \mathcal{D}_{H G-D P O}\right)+\lambda_{\text {stat }} \mathcal{L}_{\text {stat }}\left(\epsilon_{\theta} ; \epsilon_{r e f}, \mathcal{D}_{H G-D P O}\right) \tag{9}
\end{equation*}
$$

### 3.3 Personalized T2I with HG-DPO

Instead of training the entire U-Net [42], we attach LoRA [22] layers to U-Net and only train them. By attaching the pre-trained LoRA layers to various applications related to human generation, we can improve those applications. For instance, given the pre-trained LoRA layers and InstantBooth [46] which is the personalized T2I model, we can improve the image quality of InstantBooth by attaching the pre-trained LoRA layers to it without additional training. It enables us to generate high-quality human images reflecting the desired identity, which enhances the practicality of HG-DPO.

## 4 Experimental Settings

Implementation details. We set $D=100000, N=20$, and $\lambda_{\text {stat }}=10000$. More implementation details are described in the appendices.

Baselines. To demonstrate the effectiveness of our contributions, we compare our method with several baselines created using existing techniques. First, to confirm the importance of $\mathcal{D}_{H G-D P O}$, we train DPO models using the public datasets (Pick-a-pic-v2 [23] and HPD-v2 [53]). Additionally, we train DPO models using filtered versions of these datasets that contain only human images, referred

Table 3: Quantitative results of ablation study. Win rates (\%) of each model against the target model, $\epsilon_{s f t}$. In the last row, our final proposed model, HG-DPO, outperforms the target model.

| Configuration | PickScore | HPS | ImageReward | Aesthetic | CLIP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{N}-2$ | 66.10 | 67.63 | 63.84 | 55.37 | 54.24 |
| $\mathrm{~N}-5$ | 81.92 | 77.14 | 66.67 | 68.36 | 72.32 |
| $\mathrm{~N}-10$ | 84.18 | 84.57 | 74.01 | 75.14 | 69.49 |
| $\mathrm{~N}-20$ | 88.70 | 89.77 | 76.84 | 75.14 | 74.01 |
| $\mathrm{~N}-20$-LAN | 89.27 | 86.86 | 76.84 | 66.10 | 72.88 |
| $\mathrm{~N}-20-$ SML (HG-DPO) | 85.31 | 85.71 | 79.10 | 63.84 | 68.36 |

to as Pick-a-pic-v2 (H) and HPD-v2 (H). To validate the necessity of $\mathcal{L}_{\text {sta }}$, we train DPO models where the regularization weight $\beta$ in Eq. (3) is increased to mitigate the color shift instead of using $\mathcal{L}_{\text {sta }}$. Finally, to compare our method with the existing alignment approaches, we train two additional models using traditional supervised fine-tuning (SFT) and AlignProp [37].

Metrics. Similar to a previous study [51], we compare the performance of our model against a baseline model. Specifically, we generate images with both models for each prompt in a set of test prompts and then compute the preference scores. The performance of our model is assessed by determining the proportion of prompts for which our model achieves a higher preference score compared to the baseline model. This process is conducted separately for each baseline model. For the test prompts, we utilize a subset of PartiPrompts [59] that are categorized as people. To assess promptaware image preferences, we employ PickScore [23], HPS-v2 [53], and ImageReward [56]. For evaluations that do not consider the prompt, we use the AestheticScore estimator [45]. Additionally, CLIP [38] measures the alignment between the generated images and the prompts. Furthermore, we conduct user studies where participants choose their preferred images from result pairs generated by our model and a baseline model to quantify how much more frequently our model is preferred over the baseline models. Additionally, to quantify the color shifts, we convert RGB images to HSV and calculate the mean value of each channel. When evaluating the personalized T2I, we use ArcFace [14] and VGGFace [5] to measure the identity distance between the concept and generated images.

## 5 Analysis on HG-DPO

We compare HG-DPO with the baselines listed in Section 4, followed by ablations and additional analysis. We also demonstrate the effectiveness of HG-DPO on the personalized T2I. Further discussions on HG-DPO including the limitations are available in the appendices.

### 5.1 Comparison with the Baselines

In Table 1 and Figure 3 b-c, HG-DPO exhibits significantly higher win rates against models trained using Pick-a-Pic-v2, Pick-a-Pic-v2 (H), HPD-v2, and HPD-v2 (H). This corresponds with the qualitative findings shown in Figure 4, where HG-DPO produces images with more natural poses and superior text-image alignment compared to the baseline models. Furthermore, when utilizing $\mathcal{D}_{H G-D P O}$, even the baseline model trained with SFT outperforms those trained with DPO on public datasets, as evidenced by the lower win rates of HG-DPO against $\mathcal{D}_{H G-D P O}+\mathrm{SFT}$ compared to those against public datasets + DPO. This underscores the significance of our dataset.

As demonstrated in Table 2, adjusting the regularization weight $\beta$ of DPO could potentially mitigate the color shift instead of using the statistic matching loss. However, Table 1 and Figure 3 e reveals that $\mathrm{N}-20$-SML significantly outperforms $\mathcal{D}_{H G-D P O}+\mathrm{DPO}(\beta \times 20)$. This suggests that while increasing $\beta$ can reduce color shifts, it also significantly reduces the overall effectiveness of DPO. In contrast, $\mathcal{L}_{\text {stat }}$ effectively alleviates the color shift with much less impact on the performance of the model.

One of the key differences between the $\mathcal{D}_{H G-D P O}+\mathrm{SFT}$ model and our model lies in whether the training utilize only winning images or both winning and losing images. As demonstrated in Table 1 and Figure $3 \mathrm{~d}$, our model outperforms the SFT approach both quantitatively and qualitatively, highlighting the importance of including losing images in the training set. AlignProp shows better results in terms of PickScore and Aesthetic compared to our model. However, as shown in Figure 4 and in the appendices, when we fine-tune $\epsilon_{s f t}$ with AlignProp, the model loses the ability to generate Asian portraits. This is likely because the training objective of AlignProp is designed to optimize PickScore alone, neglecting the unique characteristics of the model.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-08.jpg?height=645&width=827&top_left_y=244&top_left_x=367)

(a) Winning and Losing images in $\mathcal{D}_{H G-D P O}$

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-08.jpg?height=645&width=534&top_left_y=244&top_left_x=1202)

(b) Statistics distances

Figure 5: Properties of $\mathcal{D}_{H G-D P O}$. Figure 5a shows the semantic differences between $x^{w}$ and $x^{l}$ as well as human preferences for them. Figure $5 \mathrm{~b}$ shows the cosine distance of channel-wise statistics of encoded latent (i.e., mean and standard deviation) between the images with the highest PickScore and the $n$-th highest PickScore (x-axis) in the image pool with the size of 20 . Note that $n=20$ corresponds to the losing images of our dataset, $\mathcal{D}_{\text {HG-DPO }}$. Thus, the distances for $n=20$ correspond to the distances between the winning and losing images in $\mathcal{D}_{H G-D P O}$.

### 5.2 Ablation Study

Importance of image pool in dataset construction. To demonstrate the importance of the image pool introduced in Section 3.1, we increased the value of $N$ and created datasets to train DPO models, subsequently comparing their performance. In Table 3, the notation $N-N$ denotes the model $\epsilon_{\theta}$ trained with $\mathcal{L}_{\text {Diff-DPO }}$ (see Eq. (3)) using the dataset $\mathcal{D}_{H G-D P O}$, where $N$ represents the size of the image pool. As shown in Table 3 , increasing $N$ correlates with higher win rates against the target model (ranging from $\mathrm{N}-2$ to $\mathrm{N}-20$ ). This trend is supported by Figure 4 , where $\mathrm{N}-20$ displays images with more natural poses and superior text-image alignment compared to both the target model and $\mathrm{N}-2$, underscoring the importance of a sizable image pool.

Color shift and statistics matching. As depicted in Figure 4, N-20 shows noticeable color shifts that make the images look unnatural. This is consistent with the result presented in Table 2, which indicates a significant hue shift for $\mathrm{N}$-20. In contrast, by using Latent Adaptive Normalization (LAN) or Statistics Matching Loss (SML) to align the latent statistics, we can effectively remove the color shifts and produce high-quality images with natural colors, poses, and precise text-image alignment, as illustrated in Figure 4 (N-20-LAN and N-20-SML). Comparing N-20-LAN and N-20-SML, we observe that the results from $\mathrm{N}-20$-SML are slightly inferior to those from $\mathrm{N}-20$-LAN as shown in Table 3 . We attribute this to the fact that while $\mathcal{L}_{\text {stat }}$ primarily targets color shift prevention, it also slightly regularizes the DPO process. Despite this minor performance drop, we recommend $\mathrm{N}-20$-SML as our final model due to its remarkable quality and sampling efficiency.

### 5.3 Further Analysis

Comparison of model-based preference scoring and human feedback. To evaluate whether model-based preference scoring can effectively replace manual labeling, we conduct a user study. We select a subset of data triplets $\left(p_{i}, x_{i}^{w}, x_{i}^{l}\right)$ from $\mathcal{D}_{H G-D P O}$ and ask participants to choose their preferred image between $x_{i}^{w}$ and $x_{i}^{l}$. The results indicate that $x_{i}^{w}$ is significantly more preferred by users, suggesting that model-based evaluations align well with human preferences (Figure 5 ).

Why the color shift occurs. The color shift arises from the deviation of the channel-wise statistics of latents sampled using $\epsilon_{\theta}$ from those sampled using $\epsilon_{s f t}$ as demonstrated by the effectiveness of LAN in Section 5.2. Here, we analyze why such deviation occurs. To find the cause of the difference in the channel-wise statistics of latents, we analyzed the dataset and found that there is a difference in

Table 4: Quantitative results for personalized T2I. Win rates (\%) of InstantBooth with HG-DPO against InstantBooth in terms of human preferences, text-image alignment and identity similarity.

| PickScore | HPS | ImageReward | Aesthetic | CLIP | Arcface | VGGFace |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 89.25 | 88.07 | 91.75 | 65.75 | 85.0 | 49.5 | 50.5 |

the channel-wise statistics, mean values in specific, of latents between the winning images $x^{w}$ and the losing images $x^{l}$ (see Figure 5b). Since DPO trains the model to learn the differences between winning and losing images, it can be inferred that the differences in the channel-wise mean values of latents present in the dataset were also learned by the model. This can encourage the model to shift the mean of the sampled latents far from that of the losing image and close to that of the winning image. $\mathcal{L}_{\text {stat }}$ mitigates the color shift by preventing this mean shift. Interestingly, we can observe that the difference of standard deviation between the latents of $x_{w}$ and $x_{l}$ is close to zero. We believe this is why matching only the mean in $\mathcal{L}_{\text {stat }}$ is sufficient to prevent the color shift.

Importance of the size of dataset. One of the key advantages of our approach, which utilizes AI feedback instead of costly human feedback for dataset construction, is the ease of creating a large-scale dataset. In the appendices, we demonstrate that having a sufficiently large dataset is crucial for achieving good performance. This highlights the effectiveness of our method in facilitating the creation of the large-scale dataset, much more efficiently than using human feedback.

### 5.4 Personalized T2I with HG-DPO

To demonstrate that our model also improves the performance of personalized T2I, we applied our model to InstantBooth [46], a personalized T2I method. Specifically, we train a personalization model on top of $\epsilon_{s f t}$ using InstantBooth. Since InstantBooth does not modify the existing network parameters, we simply switch $\epsilon_{s f t}$ to $\epsilon_{\theta}$ to incorporate HG-DPO into the personalization model. In Table 4, InstantBooth with HG-DPO outperforms the baseline in terms of human preferences (i.e., PickScore, HPS, ImageReward, and Aesthetic) and text-image alignment (i.e., CLIP). Additionally, it achieves win rates close to $50 \%$ in terms of identity similarity, demonstrating that InstantBooth with HG-DPO can generate images reflecting the identity of the given concept images as effectively as the base InstantBooth model. The qualitative results are provided in the appendices.

## 6 Related Work

To align the diffusion model with human preferences, several methods have been proposed based on reward maximization during training [4, 18, 6, 25, 37, 11] and inference time [52]. Furthermore, DPO [39]-based [51, 57], SPIN [9]-based [60], and KTO [17]-based [27] methods have been proposed. Additionally, a sampling method [58], which is orthogonal with them, has been also suggested. Unlike the aforementioned methods proposing the alignment algorithms, we propose the method to improve the human generation model based on one of these algorithms, Diffusion-DPO [51]. Furthermore, we propose a method that replaces human feedback with efficient AI feedback for the dataset construction, which distinguishes HG-DPO from these algorithms. Recently, a method for constructing a highquality real image dataset has been proposed [13, 26]. Furthermore, CosmicMan [26] proposed a method to learn a foundation model for the human generation. Although we also propose a dataset construction method, we propose a orthogonal method for constructing a synthesized dataset for DPO that consists of the winning and losing images, rather than a real dataset which is costly to construct.

## 7 Conclusion

In this work, we present HG-DPO, a method that enhances the performance of human image generation via DPO. Initially, we propose a technique to construct a large-scale DPO dataset without the need for costly manual labeling. Furthermore, we introduce a method designed to train models capable of producing high-quality human images without artifacts. HG-DPO demonstrates significantly improved results over existing methods, particularly in aspects such as human pose, anatomy, and adherence to text prompts. Additionally, it can be easily adapted to human-related applications such as personalized T2I, further illustrating its practical utility.

## Appendices

## A Additional Analysis on HG-DPO

In this section, we aim to conduct additional analysis on HG-DPO. In this section, we present additional analyses of HG-DPO that were not covered in our manuscript. First, we analyze the importance of the substantial differences between the winning and losing images produced by Eq. 6) during the dataset construction (Section A.1). Second, we explore the significance of creating a large-scale dataset enabled by using AI feedback instead of human feedback (Section A.2). Finally, we examine the effects of altering the LoRA [22] weight during inference time (Section A.3).

## A. 1 Importance of Large Differences between the Winning and Losing Images

In Section 3.1, we proposed a method for selecting the winning and losing images from the image pool (Eq. (6)). This method is based on the assumption that a larger PickScore difference between the winning and losing images indicates greater semantic differences, and they are crucial for enhancing the target model through DPO. As shown in Figure 6, comparing the image with the 1st highest PickScore to the image with the $l$-th highest PickScore shows that the semantic differences between the two images (e.g., anatomy, pose, and text-image alignment) become more pronounced as $l$ increases. By choosing the images with the 1st highest Pickscore and 20th highest PickScore as the winning and losing images, respectively, we accentuate the semantic differences between the them.

This design is important in improving the human generation model, as can be seen in Table 5 Table 5 presents the results of models trained by selecting the winning image with the highest PickScore from the image pool while varying the losing image. Specifically, $\mathrm{N}-20-\mathrm{L}-l$ in Table 5 refers to a model trained by choosing the image with the $l$-th highest PickScore from the image pool with the size of 20 as the losing image. Our proposed method (Eq. (6)) can be described as N-20-L-20, which is the same with $\mathrm{N}-20$ in our manuscript. Consequently, as seen in Table 5 , the win rates of $\mathrm{N}-20-\mathrm{L}-l$ against the target model increases as $l$ increases. This is consistent with Figure 7, where the image generated by $\mathrm{N}-20-\mathrm{L}-2$ shows minimal improvement from the result of the target model. This is because, when $l=2$, there are no significant semantic differences between the winning and losing images compared to when $l=20$. Note that when $l=2$, the differences between the winning and losing images are smaller compared to when $l=5$ in Figure 6 As a result, the model, which needs to learn by capturing the differences between the winning and losing images, faces ambiguity in its learning objectives. We believe this is why N-20-L-l performs better as $l$ increases.

This underscores the importance of having significant semantic differences between the winning and losing images. Our dataset construction method effectively creates these significant semantic differences without human feedback, leading to substantial performance improvements.

## A. 2 Importance of the Substantially Large-Scale Dataset

One of the key advantages of our method, which utilizes AI feedback for automatic preference labeling instead of human feedback, is its ability to facilitate the construction of large-scale dataset. When using human feedback for preference labeling, its high cost can make it difficult to construct the large-scale dataset.

Table 6 shows the performance changes when the dataset scale is reduced. In Table $6, \mathrm{~N}-20-D$ refers to a model trained with the same configuration as $\mathrm{N}-20$ in our manuscript, but with a dataset size of $D$. Here, $\mathrm{N}-20-100 \mathrm{k}$ is the configuration we used by default in all experiments, which is the same with $\mathrm{N}-20$ in our manuscript, while $\mathrm{N}-20-50 \mathrm{k}$ and $\mathrm{N}-20-2 \mathrm{k}$ are models intentionally trained with reduced dataset sizes to observe the effect of the dataset size. In Table 6, the win rates of $\mathrm{N}-20-D$ against the target model tend to increase with $D$. It is consistent with Figure 7, where the image generated by $\mathrm{N}-20-2 \mathrm{k}$ is not aligned with the text compared to the results of $\mathrm{N}-20$ and HG-DPO.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-11.jpg?height=1613&width=1396&top_left_y=459&top_left_x=362)

Figure 6: Visualization of the image pool. This figure shows the image pool with the size of 20 for the prompt in the leftmost column. The column labeled as 1st contains images with the highest PickScore, while the column labeled as 20th contains images with the 20th highest PickScore, i.e., the lowest PickScore, in the image pool. By selecting the image with the highest PickScore from this image pool as the winning image and the image with the 20th highest PickScore as the losing image, we magnify the semantic differences between the winning and losing images.

Table 5: Quantitative results according to the AI feedback ranking of the losing images. It shows win-rates (\%) of each model against the target model, $\epsilon_{s f t}$. The superior results of $\mathrm{N}-20-\mathrm{L}-20$ demonstrate that our method of maximizing the PickScore difference between the winning and losing images is effective.

| Configuration | PickScore | HPS | ImageReward | Aesthetic | CLIP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| N-20-L-2 | 66.10 | 67.63 | 63.84 | 55.37 | 54.24 |
| N-20-L-5 | 81.92 | 77.14 | 66.67 | 68.36 | 72.32 |
| N-20-L-10 | 84.18 | 84.57 | 74.01 | 75.14 | 69.49 |
| N-20-L-20 | 88.70 | 89.77 | 76.84 | 75.14 | 74.01 |

Table 6: Quantitative results according to the dataset size. It shows win-rates (\%) of each model against the target model, $\epsilon_{s f t}$. The superior results of $\mathrm{N}-20-100 \mathrm{k}$ demonstrate that a sufficiently large dataset is crucial for achieving good performance. Additionally, this underscores the effectiveness of our AI feedback-based dataset construction method, which allowed us to create such a large dataset, in contrast to relying on human feedback.

| Configuration | PickScore | HPS | ImageReward | Aesthetic | CLIP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{N}-20-2 \mathrm{k}$ | 74.58 | 79.66 | 73.45 | 74.01 | 58.19 |
| $\mathrm{~N}-20-50 \mathrm{k}$ | 85.88 | 86.29 | 76.27 | 72.88 | 70.06 |
| $\mathrm{~N}-20-100 \mathrm{k}$ | 88.70 | 89.77 | 76.84 | 75.14 | 74.01 |

This highlights the importance of the large-scale dataset for achieving superior performance. Since it is difficult to increase the dataset size when using human feedback, these results demonstrate the superiority of our dataset construction method using AI feedback.

## A. 3 Impact of the Lora Weight on Results

Figure 8 shows images generated by adjusting the LoRA weight of HG-DPO across various random seeds. When $\alpha_{\text {LoRA }}=0$, the images exhibit unnatural poses. As the value increases, the poses become more natural. However, when $\alpha_{\text {LoRA }}=0.8$, while the poses are the most natural, there is a noticeable decrease in diversity, and the background becomes too blurred. To balance between quality and diversity, we use $\alpha_{\text {LoRA }}=0.5$.

## B Limitations of HG-DPO

We have demonstrated that HG-DPO enables the target model to generate images with more natural anatomies and poses, as well as better alignment with text input. However, HG-DPO has some limitations, which we will discuss in this section. We believe that addressing these limitations could be a valuable direction for future research to further enhance HG-DPO.

## B. 1 Trade-off between the Diversity and Quality

We have demonstrated that HG-DPO is effective at generating images with more natural poses and anatomies. However, we also found that as image quality increases, diversity decreases. Each row in Figure 8 shows images generated with the same LoRA weight but different random seeds. In the first row, where the LoRA weight is 0 , the images display the lowest quality but the highest diversity. As the LoRA weight increases, the image quality continuously improves, but the diversity simultaneously decreases. While this is not a perfect solution, users can achieve satisfactory human images by selecting an optimal LoRA weight that balances both quality and diversity.

## B. 2 Limitations in Enhancing Fine Anatomical Details

While HG-DPO significantly enhances human generation in terms of overall anatomy and pose, its impact on fine anatomical details, such as fingers, is relatively limited. In Figure 8, the results with $\alpha_{\text {LoRA }}>0$ still show the unnatural fingers. We believe this is because we use PickScore [23] to

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-13.jpg?height=419&width=1137&top_left_y=235&top_left_x=494)

Prompt : Man and woman, background as ocean view with the blue sky, wearing white suits, trending on instagram

Figure 7: Qualitative results. The images generated by our model, HG-DPO, in the rightmost column have natural poses while aligning well with the given texts compared to other models.

select the winning and losing images, but PickScore does not effectively capture the fine anatomical details. In other words, since the winning image does not exhibit clear superiority over the losing image in terms of the fine anatomical features, HG-DPO is less encouraged to improve these aspects. If an estimator capable of capturing these details were used instead of PickScore, HG-DPO could be further refined to enhance these details, which is a promising avenue for future research.

## B. 3 Effect of $L_{\text {stat }}$ on Improvements Achieved through DPO

In our manuscript, we explained that although $\mathcal{L}_{\text {stat }}$ is effective in resolving the color shift, it slightly diminishes the improvement of DPO. We consider this to be one of the limitations of HG-DPO. However, note that attempting to resolve the color shift using the conventional regularization of Diffusion-DPO results in even greater harm to the improvement effects of DPO.

## C Additional Results of HG-DPO

Figures 9, 10, 11, 12, 13, and 14 show the improvements through HG-DPO. Furthermore, Figures 15 and 16 demonstrates that HG-DPO can be effectively adapted to the personalized T2I model [46]. In Figures 17, we compare HG-DPO and AlignProp [37].

## D Implementation Details

In this section, we provide implementation details on training and inference.

## D. 1 Details on Supervised Fine-Tuning

First, we introduce the method for obtaining $\epsilon_{s f t}$ through supervised fine-tuning.

Fine-tuning dataset. To obtain $\epsilon_{s f t}$, the target model for DPO, we collected 322,461 high-quality Asian images. Each image has a resolution of $704 \times 1024$. We use LLaVa [28] to generate text prompts for all the collected images.

Architecture. Using this dataset, we fine-tuned majicmix-v7 [2], a fine-tuned model of Stable Diffusion 1.5 (SD1.5) specialized in human generation. To maximize the performance of $\epsilon_{s f t}$ in human generation, we fine-tuned majicmix-v7 instead of SD1.5.

Loss function. For fine-tuning, we used the noise prediction loss [21]. Also, we used DDPM noise scheduler [21] for the forward diffusion process during training.

## D. 2 Details on HG-DPO Training

In this section, we provide details on how to improve $\epsilon_{\text {sft }}$ using HG-DPO. This involves training $\epsilon_{\text {sft }}$ with $\mathcal{L}_{H G-D P O}$ using $\mathcal{D}_{H G-D P O}$ as described in our manuscript.

Architecture. Instead of training the all parameters of $\epsilon_{s f t}$ through HG-DPO, we attached LoRA [22] layers to the all linear layers in the attention modules and only train them. We set LoRA rank as 8 .

Multiple random seeds

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-14.jpg?height=1575&width=1395&top_left_y=546&top_left_x=365)

Prompt : A photo of woman playing a guitar in the forest

Figure 8: Qualitative results. This figure shows images generated by varying the LoRA weight, $\alpha_{\text {LoRA }}$, of HG-DPO across multiple random seeds. As $\alpha_{\text {LoRA }}$ increases, the naturalness of the poses in the images improves, but the diversity decreases. To balance quality and diversity, we use $\alpha_{\text {LoRA }}=0.5$

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-15.jpg?height=998&width=1358&top_left_y=260&top_left_x=378)

Prompt: (bust shot:1.2), This is a photography of a female figure skater performing. (arms stretched out overhead), (grinning joyfully:1.2), (wearing a pale blue hue figure skating dress), (adorned with sparkling array of red and blue crystals), (it densely packed at the neckline and scatter out to downwards), (a figure skater's typical hairstyle), neat low bun, (in the ice rink), (Depth of field), professional photo

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-15.jpg?height=992&width=1357&top_left_y=1339&top_left_x=384)

Prompt: (medium close up frontal portrait shot), (horizontal composition:1.2), centered composition, (wearing a white sleeveless dress:1.2), (immerse in a sunflower field:1.2), (bright face:1.83), Depth of Field, a soft focus and muted colors, (analog film photograph:1.6), lifelike photo, natural long dark hair

Figure 9: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO, which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-16.jpg?height=1016&width=1391&top_left_y=240&top_left_x=367)

Prompt: (frontal portrait:1.2) of a girl, (wavy dark hair with white fragipani flower tucked in her ear:1.2), background as (hawaiian beach:1.3), wearing (a simple colored strappy light green dress:1.1), trending on instagram, bright weather, sunny day, natural setting

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-16.jpg?height=1014&width=1388&top_left_y=1336&top_left_x=366)

Prompt: (frontal portrait:1.2), (wearing a white knit with a scoop neckline:1.3), dreamlike (medium shot:1.2) instagram photo, looking back, soft smile, light brown (middle length hair:1.3), natural light, (dark pink tulips flower field:1.3) as background, aesthetic, (intricate detail:1.2), Perfect facial expression

Figure 10: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO, which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-17.jpg?height=1005&width=1377&top_left_y=251&top_left_x=366)

Prompt: (frontal upper body portrait of a woman, straight posture, looking front:1.6), (indoor, huge window with a beautiful scenery of winter landscape, decorated moody christmas lights and garments:1.4), (long wavy hair: 1.2), pretty bright face, ultra high quality, (wearing blue knitted sweater in female trendy wear collections:1.3), hyperrealistic, (trending on instagram:1.2), (wavy hair:1.2), (Christmas vibe:1.2), (cozy:1.2), (warm lighting on face:1.2), beauty filters, UHD, HDR, studio lighting, (centered composition:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-17.jpg?height=1013&width=1377&top_left_y=1361&top_left_x=369)

Prompt: (medium close up frontal portrait shot), (dynamic amusement park as background:1.2), (wearing a white sweatshirt:1.3), (holding an huge soft ice cream cone in one hand:1.2), (lush and long wavy ponytail:1.2), (Depth of field), lifelike photo, shallow low angle

Figure 11: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-18.jpg?height=1022&width=1392&top_left_y=240&top_left_x=364)

Prompt: (upper body shot:1.2), (frontal portrait:1.2), This is a photograph of a female volleyball player. (simple solid bright color backdrop), wearing (Premier league volleyball uniform:1.2), long hair, (holding a volleyball in the front:1.2), (centered composition), professional photograph, (front lighting)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-18.jpg?height=1005&width=1385&top_left_y=1335&top_left_x=367)

Prompt: (frontal upper body portrait of a woman holding a whole cake:1.4), wearing a (brown vintage plaid Pinafore overall dress, lace trimmed white collar academia blouse under:1.4), (neat beautiful long brunette wavy hair:1.5), (bright porcelain face:1.2), sharp focus, (face focused), front on, softly smiling, (cozy room background with holiday settings:1.2), holiday mood, (high color temperature, blurry background with bokeh:1.5), anatomically correct features

Figure 12: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-19.jpg?height=1022&width=1396&top_left_y=251&top_left_x=359)

Prompt: (bust shot:1.2) a young asian man is holding a baseball bat (wearing black suit:1.2) at a baseball stadium, professional photograph, looking front, (natural hairstyle:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-19.jpg?height=1014&width=1391&top_left_y=1339&top_left_x=367)

Prompt: A man is in a shooting stance with a pistol in the library wearing knit sweater, professional photograph, looking front, (bust shot:1.2)

Figure 13: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-20.jpg?height=1009&width=1377&top_left_y=257&top_left_x=366)

Prompt: A young asian man is riding a white horse in the forest wearing black bomber leather jumper with vintage patches

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-20.jpg?height=1006&width=1377&top_left_y=1340&top_left_x=366)

Prompt: A man is sitting on a white chair wearing a yellow cardigan and blue jean. He has his hand on his chin. There is bouquet of flowers on the floor next to him. professional photograph, looking front, (bust shot:1.2)

Figure 14: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-21.jpg?height=1019&width=1391&top_left_y=236&top_left_x=367)

Prompt: A woman is wearing a white coat, blue jean, and black knit sweater next to a tree with lush foliage standing on a snow-covered plain, (bust shot:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-21.jpg?height=1009&width=1388&top_left_y=1325&top_left_x=366)

Prompt: A woman wearing a blue long-sleeve shirt and white pants is on a yacht with an island visible in the background. she has long wave hair. professional photograph

Figure 15: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively. The image at the top right is the concept image of the desired identity.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-22.jpg?height=1017&width=1390&top_left_y=234&top_left_x=365)

Prompt: A man is holding a dog in the house. He is wearing gray sweatshirt and black pants, sitting on the sofa with the sunset reflecting through the window. professional photograph, (bust shot:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-22.jpg?height=1011&width=1388&top_left_y=1321&top_left_x=366)

Prompt: A man is standing in an amusement park at night, wearing a navy suit, holding big red wine glass. Fireworks are going off in the background. professional photograph, (bust shot:1.2)

Figure 16: HG-DPO vs without HG-DPO. The left and right columns show the generated images with HG-DPO and without HG-DPO which corresponds to the target model, respectively. The image at the top right is the concept image of the desired identity.

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-23.jpg?height=531&width=677&top_left_y=241&top_left_x=385)
is riding a camel in the desert, wearing a
professional photograph, (bust shot:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-23.jpg?height=513&width=693&top_left_y=795&top_left_x=377)

Prompt: A little girl is holding a teddy bear at home. professional photograph

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-23.jpg?height=536&width=694&top_left_y=239&top_left_x=1060)

professional photograph, (bust shot:1.2), (frontal portrait:1.2)

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-23.jpg?height=509&width=697&top_left_y=797&top_left_x=1061)

Prompt: A little boy is holding a camera at the amusement park.

Figure 17: HG-DPO vs AlignProp. The left and right images in each subfigure show the generated images with HG-DPO and AlignProp, respectively, by using each prompt. In the case of images generated by AlignProp, they lose its original styles (e.g., realistic images of Asians) and are altered to distinct styles (e.g., artistic images of Westerners). As we described in Section 1 , in the practical scenarios where human generation models are deployed, the transformation in the styles of generated images can be critical because it can prevent the user from generating images in their target styles.

Loss function. For $\mathcal{L}_{H G-D P O}$, we set $\beta=2500$ and $\lambda_{\text {stat }}=10000$. Also, we used DDPM noise scheduler [21] for the forward diffusion process during training. Note that $\mathcal{L}_{H G-D P O}$ also requires the forward diffusion process as the noise prediction loss [21]. Also, for the latent sampling in $\mathcal{L}_{\text {stat }}$, we used DDPM sampler [21]. We tried DDIM sampler [48], but there was no significant difference. In addition, classifier-free guidance [20] is not used during the latent sampling in $\mathcal{L}_{H G-D P O}$.

Optimization. For the optimization, we set the local batch size to four, which corresponds to the total batch size to 16 because we used four NVIDIA A100 GPUs. As an optimizer, we use the 8 -bit Adam optimizer [15] with $\beta_{1}$ and $\beta_{2}$ of the Adam optimizer to 0.9 and 0.999 , respectively, and the learning rate to $1 e-5$. Additionally, we utilize mixed precision for efficient training.

## D. 3 How to Adapt HG-DPO to Personalized T2I model

To adapt HG-DPO to the personalized T2I model, we firstly trained InstantBooth [46] using $\epsilon_{\text {sft }}$ as the backbone. After training InstantBooth, we can seamlessly adapt the pre-trained HG-DPO LoRA layers to InstantBooth because they share the same backbone, $\epsilon_{s f t}$.

## D. 4 Image Sampling

Prompts of Figure 1 in our manuscript. The prompts listed below are the ones used to generate the images in Figure 1, in order from the top left to the bottom right.

A man in a black suit is walking on the dark street

![](https://cdn.mathpix.com/cropped/2024_06_04_df9dd187a2912405a92ag-24.jpg?height=515&width=434&top_left_y=309&top_left_x=1320)

Figure 18: User study interface. We conduct the user study by providing a prompt and two images, asking users to choose the one that appeared more natural.

A woman in a futuristic cyberpunk world, with a neon-lit city background, backlit by vibrant city glow A woman singing on the stage, $8 k$, photo realistic

Asian man in dark cathedral, magnificent, medieval armor with complicated decorations, light from the stained-glass windows flashes him

High-fashion photography of an asian man wearing a beige coat in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, $8 k$

A woman, film grain, overexposed, long grass, wind, white sundress, fresh, outdoor photography, large aperture, highres, realistic photography

A close-up shot of woman with a short black hair and beautiful smile in the cafe. She is wearing a blue dress

A portrait photo of a girl, centered, highly detailed face, depth of field, moody light, golden hour, sunset, faint, dim, idyllic

Closeup portrait photo of goth asian man, makeup, black jacket, $8 k$

Sampling hyperparameters. DPMSolverMultistepScheduler [32] in diffusers [50] is used with the step size of 50 for sampling the images. In addition, Classifier-free guidance [20] is used, with the guidance scale of 5.0 .

## E How to conduct user study

We conduct the user study to demonstrate the superiority of HG-DPO in generating more natural images compared to other baselines. As shown in Figure 18, we create a web user interface that ask participants to choose the more natural image between two options.

## F Broader Impacts

We recognize the potential negative societal impacts of our work. Since our method can generate highquality human images, it could be misused to create malicious fake images, especially when combined with personalized T2I models. It can cause significant harm to specific individuals. However, our work can also have positive impacts on society when used beneficially, such as in the entertainment or film industries. For instance, users can create desired high-quality profile pictures using text input. It highlights the beneficial uses of our work.

## References

[1] Civitai. https://civitai.com/

[2] majicmix realistic. https://civitai.com/models/43331/majicmix-realistic

[3] Pixai. https://pixai.art/.

[4] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

[5] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face \& gesture recognition (FG 2018), pages 67-74. IEEE, 2018.

[6] C. Chen, A. Wang, H. Wu, L. Liao, W. Sun, Q. Yan, and W. Lin. Enhancing diffusion models with text-encoder reinforcement learning. arXiv preprint arXiv:2311.15657, 2023.

[7] H. Chen, Y. Zhang, S. Wu, X. Wang, X. Duan, Y. Zhou, and W. Zhu. Disenbooth: Identitypreserving disentangled tuning for subject-driven text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.

[8] X. Chen, L. Huang, Y. Liu, Y. Shen, D. Zhao, and H. Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023.

[9] Z. Chen, Y. Deng, H. Yuan, K. Ji, and Q. Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

[10] P. Cheng, Y. Yang, J. Li, Y. Dai, and N. Du. Adversarial preference optimization. arXiv preprint arXiv:2311.08045, 2023.

[11] K. Clark, P. Vicol, K. Swersky, and D. J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

[12] J. Dai, X. Pan, R. Sun, J. Ji, X. Xu, M. Liu, Y. Wang, and Y. Yang. Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023.

[13] X. Dai, J. Hou, C.-Y. Ma, S. Tsai, J. Wang, R. Wang, P. Zhang, S. Vandenhende, X. Wang, A. Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.

[14] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690-4699, 2019.

[15] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer. 8-bit optimizers via block-wise quantization. arXiv preprint arXiv:2110.02861, 2021.

[16] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780-8794, 2021.

[17] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

[18] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

[19] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, and B. Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696-10706, 2022.

[20] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[21] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840-6851, 2020.

[22] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[23] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

[24] T. Korbak, K. Shi, A. Chen, R. V. Bhalerao, C. Buckley, J. Phang, S. R. Bowman, and E. Perez. Pretraining language models with human preferences. In International Conference on Machine Learning, pages 17506-17533. PMLR, 2023.

[25] K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

[26] S. Li, J. Fu, K. Liu, W. Wang, K.-Y. Lin, and W. Wu. Cosmicman: A text-to-image foundation model for humans. arXiv preprint arXiv:2404.01294, 2024.

[27] S. Li, K. Kallidromitis, A. Gokul, Y. Kato, and K. Kozuka. Aligning diffusion models by optimizing human utility. arXiv preprint arXiv:2404.04465, 2024.

[28] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In NeurIPS, 2023.

[29] T. Liu, Y. Zhao, R. Joshi, M. Khalman, M. Saleh, P. J. Liu, and J. Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.

[30] T. Liu, Z. Qin, J. Wu, J. Shen, M. Khalman, R. Joshi, Y. Zhao, M. Saleh, S. Baumgartner, J. Liu, et al. Lipo: Listwise preference optimization through learning-to-rank. arXiv preprint arXiv:2402.01878, 2024.

[31] W. Liu, X. Wang, M. Wu, T. Li, C. Lv, Z. Ling, J. Zhu, C. Zhang, X. Zheng, and X. Huang. Aligning large language models with human preferences through representation engineering. arXiv preprint arXiv:2312.15997, 2023.

[32] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.

[33] J. Ma, J. Liang, C. Chen, and H. Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023.

[34] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[35] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.

[36] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

[37] M. Prabhudesai, A. Goyal, D. Pathak, and K. Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR, 2021.

[39] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

[40] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

[41] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684-10695, 2022.

[42] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention-MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234-241. Springer, 2015.

[43] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500-22510, 2023.

[44] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: $36479-36494,2022$.

[45] C. Schuhmann. Laion-aesthetics. https://laion.ai/blog/laion-aesthetics/ 2022.

[46] J. Shi, W. Xiong, Z. Lin, and H. J. Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411, 2023.

[47] F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang. Preference ranking optimization for human alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18990-18998, 2024.

[48] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[49] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020 .

[50] P. von Platen, S. Patil, A. Lozhkov, P. Cuenca, N. Lambert, K. Rasul, M. Davaadorj, D. Nair, S. Paul, W. Berman, Y. Xu, S. Liu, and T. Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, 2022.

[51] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.

[52] B. Wallace, A. Gokul, S. Ermon, and N. Naik. End-to-end diffusion latent optimization improves classifier guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7280-7290, 2023.

[53] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[54] Z. Wu, Y. Hu, W. Shi, N. Dziri, A. Suhr, P. Ammanabrolu, N. A. Smith, M. Ostendorf, and H. Hajishirzi. Fine-grained human feedback gives better rewards for language model training. Advances in Neural Information Processing Systems, 36, 2024.

[55] G. Xiao, T. Yin, W. T. Freeman, F. Durand, and S. Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.

[56] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

[57] K. Yang, J. Tao, J. Lyu, C. Ge, J. Chen, Q. Li, W. Shen, X. Zhu, and X. Li. Using human feedback to fine-tune diffusion models without any reward model. arXiv preprint arXiv:2311.13231, 2023.

[58] T. Yoon, K. Myoung, K. Lee, J. Cho, A. No, and E. Ryu. Censored sampling of diffusion models using 3 minutes of human feedback. Advances in Neural Information Processing Systems, 36, 2024.

[59] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

[60] H. Yuan, Z. Chen, K. Ji, and Q. Gu. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv preprint arXiv:2402.10210, 2024.

[61] Y. Zhao, R. Joshi, T. Liu, M. Khalman, M. Saleh, and P. J. Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.


[^0]:    *Equal contribution

    ${ }^{\dagger}$ Corresponding author

</end of paper 1>


<paper 2>
# Self-Augmented Preference Optimization: Off-Policy Paradigms for Language Model Alignment 

Yueqin Yin ${ }^{1, *}$, Zhendong Wang ${ }^{1,2, *}$, Yujia Xie ${ }^{2}$,<br>Weizhu Chen ${ }^{2}$, and Mingyuan Zhou ${ }^{1, \star}$<br>\{yueqin.yin, zhendong.wang\}@utexas.edu<br>\{yujiaxie,wzchen\}@microsoft.com, mingyuan.zhou@mccombs.utexas.edu<br>${ }^{1}$ The University of Texas at Austin<br>${ }^{2}$ Microsoft Azure AI

June 3, 2024


#### Abstract

Traditional language model alignment methods, such as Direct Preference Optimization (DPO), are limited by their dependence on static, pre-collected paired preference data, which hampers their adaptability and practical applicability. To overcome this limitation, we introduce Self-Augmented Preference Optimization (SAPO), an effective and scalable training paradigm that does not require existing paired data. Building on the self-play concept, which autonomously generates negative responses, we further incorporate an off-policy learning pipeline to enhance data exploration and exploitation. Specifically, we employ an Exponential Moving Average (EMA) model in conjunction with a replay buffer to enable dynamic updates of response segments, effectively integrating real-time feedback with insights from historical data. Our comprehensive evaluations of the LLaMA3-8B and Mistral-7B models across benchmarks-including the Open LLM Leaderboard, IFEval, AlpacaEval 2.0, and MT-Bench-demonstrate that SAPO matches or surpasses established offline contrastive baselines, such as DPO and Odds Ratio Preference Optimization, and outperforms offline self-play methods like SPIN. Our code is available at https://github.com/ yinyueqin/SAPO.


## 1 Introduction

In the rapidly evolving field of artificial intelligence, aligning Large Language Models (LLMs) with human preferences has emerged as a critical area of research Agrawal et al., 2023, Shi et al., 2023, Kadavath et al., 2022, Liang et al. 2021, Sheng et al. 2019, Christiano et al., 2017. Classical methods, such as Reinforcement Learning (RL) from Human Feedback (RLHF) Ziegler et al., 2019, Ouyang et al., 2022], have progressed by training models to optimize responses via a reward model that reflects human preferences. However, the necessity of a separate reward model introduces additional complexity and computational demands. To streamline this, Direct Preference Optimization (DPO) Rafailov et al. 2023 directly utilizes preference data to optimize language models, eliminating the need for an auxiliary reward model. Odds Ratio Preference Optimization (ORPO) Hong et al. 2024 further streamlines the alignment process by removing the reference model entirely. ORPO employs an odds ratio to directly evaluate preferences between different responses during Supervised Fine-Tuning (SFT), thus simplifying the alignment process. However, despite enhancements[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_a5fc7ba917d83ce5a449g-02.jpg?height=756&width=1635&top_left_y=243&top_left_x=234)

Figure 1: Given a prompt $x_{1}$ and chosen response $y_{1}^{+}$. This response is segmented into $A, B$, and $C$. Using the prompt with segment $A$, the EMA model generates a new segment $B^{\prime}$. Together, segments $A, B^{\prime}$, and $C$ form the rejected response $y_{1}^{-}$, which is appended to the replay buffer. Random tuples are sampled from this buffer to train the policy network, subsequently updating the EMA weights.

with DPO, ORPO, and many other offline contrastive preference learning algorithms Hong et al., 2024, Ethayarajh et al., 2024, Zhao et al. 2023, their reliance on static, pre-collected preference datasets poses challenges, especially in sensitive domains where privacy concerns and the scarcity of expert input limit adaptability and application scope.

The Self-Play Fine-Tuning (SPIN) method [Chen et al. 2024] tackles the challenge of data collection by using a self-play approach, where the model autonomously generates its own responses to serve as rejected inputs. This strategy enables SPIN to function with minimal data requirements-only requiring prompts and selected responses - thereby alleviating the difficulty of gathering paired preference datasets. However, SPIN's methodology comes with significant limitations. Its primary drawback is the reliance on offline, pre-generated responses, which hampers the model's ability to dynamically adjust training data in real-time. Additionally, this dependency necessitates a rigid training procedure, where complete data generation must precede the start of training, introducing significant delays.

Recent work has underscored the significance of online training in enhancing the alignment performance of Large Language Models (LLMs) Guo et al., 2024, Rosset et al., 2024. However, these methods largely rely on reward models or powerful teacher models like GPT-4 to provide guidance signals. We introduce Self-Augmented Preference Optimization (SAPO), as depicted in Figure 1, a general self-play preference learning algorithm without depending on any external reward models or teacher models. We derive the motivation for SAPO from the principles of off-policy RL training. Lillicrap et al., 2015, Haarnoja et al., 2018, Wang et al. 2022. SAPO consists of three main components: the current policy, an Exponential Moving Average (EMA) model of the current policy, and a first-in-first-out replay buffer.

The learning of SAPO involves two stages at each iteration: sampling and training. During the sampling stage, the EMA model is used to generate responses, creating self-augmented rejected samples. Both the original responses and these generated samples, along with the prompts, are stored in the replay buffer. In the training stage, we randomly select a batch of tuples from the replay buffer and employ typical preference learning methods, such as DPO Rafailov et al. 2023 and ORPO Hong et al., 2024, to train the current policy. After training, we update the EMA model at a fixed rate. These two stages work in tandem to
incrementally improve the policy. By using the EMA model and replay buffer, we reduce the impact of volatility from any single training iteration and ensure more consistent learning signals. The progression of training data within the replay buffer adopts principles akin to curriculum learning $\mathrm{Xu}$ et al. 2020, Wang et al., 2024b, Pattnaik et al., 2024, starting with simpler training pairs and gradually incorporating more complex training samples, allowing the model to build competency progressively.

SAPO employs a teacher-forcing segment-level supervision strategy, truncating a chosen response at a random point to create a supervision segment. As illustrated in Figure 1, with the prompt "Explain the role of ATP in cellular processes," a chosen response segment $B$ might be "When ATP is hydrolyzed, it releases energy, which is then harnessed to perform cellular work." Conversely, the EMA model might generate a less accurate rejected segment $B^{\prime}$, such as "providing energy only when other sources are depleted," positioning ATP erroneously as a secondary reserve. By focusing on generating tailored segments instead of entire rejected responses from scratch, SAPO is more likely to produce meaningful outputs. This method facilitates more tailored adjustments to the training data.

Experimental evaluations across four benchmarks-Open LLM Leaderboard Beeching et al., 2023], which tests question answering ability; IFEval Zhou et al. 2023, measuring instruction-following ability; MT-Bench [Zheng et al., 2024] and AlpacaEval 2.0 [Dubois et al. 2024], assessing conversational abilitydemonstrate that SAPO can either match or exceed the performance of existing offline contrastive learning methods like DPO Rafailov et al. 2023 and ORPO Hong et al. 2024, despite our method solely utilizing chosen responses. Furthermore, SAPO outperform purely offline self-play methods such as SPIN Chen et al. 2024 , which require longer training times.

## 2 Preliminaries

### 2.1 Definition of Learning Paradigms in LLM Alignment

For ease of understaning RL concepts in LLMs, we provide the following definitions:

- Offline Learning: Offline learning involves the LLM being trained on a pre-collected dataset without further interaction with additional reward models or acquiring new human annotated data during training.
- On-Policy Learning: This learning approach ensures that the training data is generated and utilized under the current policy of the LLM. It implies that the model is refined using data derived from the decisions it would make under its current strategy.
- Off-Policy Learning: Off-policy learning enables the use of data generated from a different policy than the one currently being trained. This approach allows the model to leverage data from past policies or alternative strategies, providing a broader range of experiences for the model to learn from, which may not necessarily align with its current operational policy.


### 2.2 Offline Off-Policy Contrastive Preference Learning Algorithms

DPO Rafailov et al. 2023 is designed to optimize a policy $\pi_{\theta}$, based on a reference model $\pi_{\text {ref }}$ that is typically the SFT baseline. DPO utilizes an off-policy, offline training approach, employing a pre-collected dataset of triplets $\left(x, y^{+}, y^{-}\right)$to enable preference learning. Within this dataset, $x$ serves as the input prompt, $y^{+}$as the chosen response, and $y^{-}$as the rejected response. The DPO loss function is formulated as follows:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y^{+}, y^{-}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y^{+} \mid x\right)}{\pi_{\mathrm{ref}}\left(y^{+} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y^{-} \mid x\right)}{\pi_{\mathrm{ref}}\left(y^{-} \mid x\right)}\right)\right] \tag{1}
\end{equation*}
$$

Here, $\beta$ is a hyperparameter that controls the degree of KL regularization.

ORPO Hong et al. 2024 provides a unique method for optimizing policy models without the need for a reference model. By employing log odds ratios, ORPO directly contrasts favored and disfavored responses during SFT, simplifying model alignment. Utilizing the odds defined as:

$$
\begin{equation*}
\operatorname{odds}\left(y^{+} \mid x\right)=\frac{\pi_{\theta}\left(y^{+} \mid x\right)}{1-\pi_{\theta}\left(y^{+} \mid x\right)}, \quad \operatorname{odds}\left(y^{-} \mid x\right)=\frac{\pi_{\theta}\left(y^{-} \mid x\right)}{1-\pi_{\theta}\left(y^{-} \mid x\right)} \tag{2}
\end{equation*}
$$

the ORPO algorithm compute the log odds ratio, effectively balancing the enhancement and penalization of responses, as:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{ORPO}}=\mathbb{E}_{\left(x, y^{+}, y^{-}\right) \sim \mathcal{D}}\left[\mathcal{L}_{\mathrm{SFT}}-\lambda \cdot \log \sigma\left(\log \frac{\operatorname{odds}\left(y^{+} \mid x\right)}{\operatorname{odds}\left(y^{-} \mid x\right)}\right)\right] \tag{3}
\end{equation*}
$$

Here, $\mathcal{L}_{\mathrm{SFT}}$ is the supervised fine-tuning loss aimed at maximizing the likelihood of generating the chosen responses, and $\lambda$ is a hyperparameter that weights the relative importance of the odds ratio term in the overall loss function.

### 2.3 Self-Play Fine-Tuning

The Self-Play Fine-Tuning (SPIN) method Chen et al. 2024 introduces a self-play mechanism that reduces the reliance on pre-collected paired preference data. Unlike traditional approaches, SPIN necessitates only annotated chosen data, as it autonomously generates its own rejected data from its previous iterations. SPIN then utilizes existing RLHF methods, e.g., DPO, to iteratively refine model responses by discerning these self-generated responses from those obtained from human-annotated data, a process inspired by principles of game theory [Samuel, 1959]. While SPIN effectively eliminates the reliance on offline contrastive learning's need for paired data by autonomously generating rejected responses, it utilizes pre-generated training data from the model's prior iterations. This data remains unchanged throughout subsequent training cycles, which may limit the model's ability to adapt to new information or evolving training conditions.

## 3 Self-Augmented Preference Optimization

Recent advancements in offline contrastive preference learning algorithms Rafailov et al., 2023, Ethayarajh et al. 2024, Hong et al. 2024 have shown promising results in aligning LLMs with human preferences. However, these methods typically rely on pre-collected paired preference datasets. Our goal is to create a robust and efficient self-augmented algorithm that eliminates the requirement for paired data. This new algorithm autonomously generates high-quality rejected responses which, combined with chosen responses from SFT datasets, form the necessary pairs for preference learning. Recent initiatives like SPIN Chen et al., 2024 utilize an iterative training method involving self-play, where sampling and training occur in separate phases. Initially, rejected responses are sampled across the entire dataset, followed by a distinct training phase. This iterative paradigm tends to be slow because it alternates between sampling for the entire dataset and training phases. As the training progresses, the effectiveness of the generated preference learning pairs may diminish. This is because the model continues to evolve, whereas the sampled dataset remains static for each iteration and cannot adapt to the latest model updates.

Instead, we start from a standard off-policy RL framework. To efficiently sample high-quality preference pairs, we introduce segment-level supervision. This approach involves replacing a segment of rejected responses with outputs generated by LLM, thereby naturally creating challenging negative sample pairs through targeted local modifications. Furthermore, we integrate an EMA model and a replay buffer-techniques well-established in RL-to facilitate standard off-policy training. This strategy ensures timely feedback and updates to the training data, operating independently of external feedback mechanisms. The implementation of SAPO is summarized in Algorithm 1. and a comparison with on-policy training is provided in Section 4.3

Segment-Level Supervision in SAPO. Unlike SPIN Chen et al. 2024, which generates rejected responses from scratch, SAPO utilizes a teacher-forcing segment-level supervision method to refine the learning process. Consider a SFT dataset $\mathcal{D}$ consisting of tuples $\left(x, y^{+}\right)$, where $x$ is a prompt and $y^{+}$is the selected response. During each training iteration, the model randomly selects a truncation point within each response $y^{+}$, defining segment $B$ of length $N_{\text {seg }}$ starting from this point. Segment $A$ comprises the tokens preceding $B$, and segment $C$ includes the tokens following $B$. The original response $y^{+}$can thus be expressed as:

$$
\begin{equation*}
y^{+}=A \oplus B \oplus C \tag{4}
\end{equation*}
$$

The model attempts to regenerate $B$ as $B^{\prime}$ based on the prompt and segment $A$. For continuity and to maintain the contextual integrity of the response, segment $C$ is concatenated, resulting in:

$$
\begin{equation*}
y^{-}=A \oplus B^{\prime} \oplus C \tag{5}
\end{equation*}
$$

where $\oplus$ denotes concatenation. This segmentation strategy not only improves supervision granularity by focusing on specific response segments - by regenerating only the middle segment $B^{\prime}$, the model can concentrate its learning efforts on specific parts of the response that may be problematic or less accurate. In addition, sampling segments is more time efficient than sampling the complete sentences.

Off-Policy Learning Setting. Inspired by off-policy RL training methods Lillicrap et al., 2015, Haarnoja et al., 2018, Wang et al., 2022, the proposed SAPO framework incorporates several fundamental components: the current policy, an EMA model of this policy, and a first-in-first-out replay buffer. In this context, the replay buffer $\mathcal{B}$ plays a crucial role, mirroring curriculum learning principles $\mathrm{Xu}$ et al., 2020, Wang et al., 2024b, Pattnaik et al., 2024. The recent Curry-DPO study Pattnaik et al., 2024 also highlights the effectiveness of sequencing preference pairs from simpler to more complex throughout training, which gradually increases task complexity and enhances learning efficiency. This approach prevents the model from being overwhelmed by challenging tasks at early stages, thus improving learning outcomes and enhancing model robustness. However, the Curry-DPO approach requires an additional reward model to manually order training pairs, which can introduce biases and require extra resources. Our method automates this process, naturally achieving a curriculum learning effect and reducing dependency on supplementary models.

Initially, our replay buffer is populated with simple training pairs-where the rejected responses $\left(y^{-}\right)$are distinctly different from the chosen responses $\left(y^{+}\right)$, and these responses are generated by the early iterations of the model. As the model evolves and its capacity for generating better responses increases, the quality of newly generated $y^{-}$responses also improves. These $y^{-}$responses are not generated from the current fine-tuning policy model but from the EMA model, denoted as $\pi_{\text {EMA }}$. This mechanism is designed to stabilize training by utilizing a less variable model state to generate responses, thereby reducing the impact of any single training iteration's volatility on the overall learning process. Our replay buffer operates as a queue, adhering to the FIFO (First In, First Out) principle, which facilitates the gradual replacement of simpler, initial training examples with more complex ones, embodying offline learning by reusing accumulated past data. This progression naturally mirrors curriculum learning, where tasks that start simply become progressively more challenging, thereby enhancing the model's training stability. Additionally, we have implemented a sampling mechanism within the replay buffer where each entry is equipped with a counter that increases each time that entry is sampled. This setup ensures that the sampling weight for each entry becomes inversely proportional to its frequency of use, preventing over-sampling of certain data and promoting a more even distribution of sample usage. This balanced approach is crucial for ensuring that both historical insights and fresh perspectives are consistently integrated into the model's learning, thereby enhancing the robustness of the training process.

The EMA model further supports this off-policy learning setting by stabilizing the learning process through the averaging of policy parameters $\theta$, updated as:

$$
\begin{equation*}
\theta_{\mathrm{EMA}} \leftarrow \alpha \theta_{\mathrm{EMA}}+(1-\alpha) \theta \tag{6}
\end{equation*}
$$

```
Algorithm 1 Self-Augmented Preference Optimization (SAPO)
    Input: Dataset with prompts and responses, base model $\pi_{\theta}$, total number of iterations $T$, learning rate
    $l r$, EMA coefficient $\alpha$
    Initialize replay buffer $\mathcal{B}$
    Initialize EMA model parameters $\theta_{\text {EMA }}$ with $\theta$
    for each iteration $i$ from 1 to $T$ do
        \# Sampling Stage
        Sample a mini-batch of $\left(x, y^{+}\right)$tuples from the dataset, each batch containing $N$ samples.
        for each $\left(x, y^{+}\right)$in the batch do
            Randomly truncate $y^{+}$to obtain segments $A, B$, and $C$.
            Combine $x$ with segments $A$ and $B$ as input to the EMA model to generates segment $B^{\prime}$.
            Concatenate $A, B^{\prime}$, and $C$ to form the rejected response $y^{-}$.
            Store $\left(x, y^{+}, y^{-}\right)$in the replay buffer $\mathcal{B}$.
        end for
        \# Training Stage
        Sample a mini-batch of tuples $\left(x, y^{+}, y^{-}\right)$from $\mathcal{B}$
        Compute loss $\mathcal{L}$ using DPO/ORPO formulas (Eq. 1 and Eq. 3 based on tuples $\left(x, y^{+}, y^{-}\right)$
        Update policy parameters $\theta$ using gradient descent: $\theta \leftarrow \theta-\operatorname{lr} \nabla_{\theta} \mathcal{L}\left(x, y^{+}, y^{-}, \theta\right)$
        Update EMA model parameters: $\theta_{\text {EMA }} \leftarrow \alpha \theta_{\text {EMA }}+(1-\alpha) \theta$
    end for
```

where $\alpha$ is a decay factor. This approach exemplifies the off-policy nature of the learning process, as the data used is not directly generated from the currently fine-tuned policy. Our experiments have shown that this off-policy method yields better results compared to directly using on-policy data for model fine-tuning. Such a comprehensive strategy ensures that SAPO remains adaptive and effective, refining the model's response generation capabilities throughout its training process.

## 4 Experiments

### 4.1 Experimental Setup.

Baselines. We compare two offline contrastive preference learning algorithms: DPO Rafailov et al. 2023 and ORPO Hong et al., 2024. Additionally, we adapt the SPIN algorithm Chen et al., 2024 for both DPO and ORPO. We have implemented two versions of our SAPO algorithm, each utilizing the loss functions from DPO and ORPO, respectively. Both DPO and ORPO require paired data for training, doubling the dataset size compared to SPIN and SAPO. Notably, SPIN's sequential process of generating and training on data not only introduces considerable delays, as training cannot begin until the entire dataset has been generated, but also adds complexity to the workflow. This results in higher latency in model readiness when compared to SAPO, which utilizes a more streamlined and efficient approach.

Datasets and Base Models. We utilize the Distilabel-Capybara dataset 1 . a multi-turn dialogue preference dataset comprising $7.6 \mathrm{k}$ entries, designed to enhance the conversational abilities of open-source LLMs. Each entry consists of multiple dialogue turns between a user and an assistant, with only the final message from the assistant considered as a response, while the preceding interactions serve as the prompt. Responses are generated by various LLMs, and then assessed using gpt-4-turbo. Although the dataset is relatively small in size, it is of high quality. For contrastive offline preference learning algorithms like DPO and ORPO, both[^1]chosen and rejected responses are required as training data. In contrast, for SPIN and our SAPO, only the prompts and chosen responses from the Distilabel-Capybara dataset are necessary.

We experiment with two types of models, Mistral and LLaMA, for DPO and ORPO-based algorithms. For DPO-based models utilizing the Mistral architecture, we employ mistral-7b-zephyr-sft ${ }^{2}$ as the base model, which undergoes supervised fine-tuning on the Deita 10k dataset Liu et al., 2023. For the DPO-based LLaMA model, we use Meta-LLaMA-3-8B as the base model, following the Mistral SFT protocol by conducting supervised fine-tuning on the Deita 10k dataset to produce the llama-3-8b-sft model. For the ORPO algorithm, our base models include Mistral-7B-v0.1 and Meta-LLaMA-3-8B. These models are then utilized in various preference learning algorithms to achieve preference alignment.

Evaluation Benchmarks. Following previous studies Hong et al. 2024, Chen et al. 2024, we assess the model performance using four established benchmarks, including the Open LLM Leaderboard Beeching et al., 2023], IFEval Zhou et al., 2023, MT-Bench Zheng et al., 2024, and AlpacaEval 2.0 Dubois et al., 2024 . These benchmarks enable us to comprehensively evaluate our approach and baseline methods across various aspects, including question answering, instruction-following, and conversational ability.

- Open LLM Leaderboard |Beeching et al., 2023|: A comprehensive benchmark suite aggregating six popular datasets: ARC [Clark et al., 2018, GSM8K Cobbe et al., 2021, HellaSwag Zellers et al., 2019, MMLU Hendrycks et al., 2020, TruthfulQA Lin et al. 2021, and Winogrande Sakaguchi et al. 2021. This leaderboard assesses diverse aspects of language model performance including reasoning, language understanding, and problem-solving through few-shot prompting on these test sets.
- IFEval Zhou et al., 2023]: IFEval benchmark evaluates language models on their ability to follow instructions, featuring 541 prompts with verifiable directives such as length constraints and specific formats. This benchmark assesses models using 25 different types of instructions in a zero-shot evaluation, focusing on the accuracy and compliance of models in executing detailed instructions.
- MT-Bench [Zheng et al., 2024]: MT-Bench tests language models with 80 multi-turn questions across domains like writing and coding. Each question set challenges models to maintain context over two turns. GPT-4 OpenAI, 2023 rates responses from 1 to 10, and the overall score is averaged to evaluate the model's conversational skill and understanding across subjects.
- AlpacaEval 2.0 Dubois et al., 2024]: AlpacaEval 2.0 employs a dataset of 805 input prompts for a win-rate comparison against GPT-4-Turbo. This benchmark focuses on evaluating response quality through a win-rate mechanism that has been enhanced in its latest version to control for length bias. This adjustment ensures that evaluations accurately reflect the substantive quality of the responses rather than their length.

Training Details. All training was conducted on 8 Nvidia H100 GPUs. For specific training hyperparameters of the baseline experiments, please see Appendix A. We utilized the foundational settings consistent with those for DPO Rafailov et al., 2023 and ORPO Hong et al. 2024. For our SAPO method, the maximum length for prompts was set at 1792 , with the total maximum length for prompts and responses capped at 2048. Training was carried out over four epochs. The segment length for teacher-forcing supervision was 256, the replay buffer was sized at 2000. For each combination of prompt and chosen response, we sampled a single corresponding rejected response. The update coefficient $\alpha$ for the EMA model was set to 0.5. The EMA model was updated every two steps during our training process.[^2]

### 4.2 Benchmark Performance

In our comprehensive analysis of the SAPO framework, detailed in Tables 1 and 2 , we systematically evaluate and compare the performance of various models, focusing on multiple dimensions crucial for the performance of LLMs. Table 1 presents a detailed comparative performance analysis on the Open LLM Leaderboard benchmark across various tasks tailored to assess reasoning, language understanding, and problem-solving capabilities. Notably, under DPO and ORPO-based algorithms, SAPO implemented with LLaMA and Mistral architectures, demonstrate superior performance across most datasets, achieving higher average scores. The improvement is particularly notable in the LLaMA models, with the ORPO-enhanced LLaMA reaching an average score of 67.36 .

Table 2 evaluates language model alignment performance on benchmarks such as instruction-following (IFEval) and conversational ability (MT-Bench, AlpacaEval 2.0). Here, SAPO achieves high scores in IFEval, indicating its exceptional capability in instruction-following. Additionally, in assessing conversational ability, particularly for the two-turn conversation benchmark MT-Bench, we present scores for each turn and their average score evaluted by GPT-4. SAPO consistently outperforms other models across most settings in MT-Bench, achieving an average score of 7.45 in the ORPO-based LLaMA setting, which highlights its robust multi-turn conversational capabilities. Furthermore, in AlpacaEval 2.0, we present both the length control win rate and the base win rate without length control, alongside the average response length. This data highlights how performance in AlpacaEval is influenced by response length. While SAPO underperforms compared to the LLaMA-based SPIN in single-turn tasks, it shows better performance on the Mistral model. It is important to note that our training was conducted on the multi-turn conversation dataset Distilabel-Capybara, and since AlpacaEval 2.0 primarily consists of single-turn dialogue tasks, we suggest using MT-Bench as a more appropriate metric for evaluating the model's conversational abilities.

Table 1: Open LLM Leaderboard Evaluation.

| Cases | Method | Arc Challenge | TruthfulQA | Winogrande | GSM8k | Hellaswag | MMLU | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ORPO-Based | meta-llama/Meta-Llama-3-8B | 58.02 | 43.92 | 77.43 | 51.48 | 82.10 | 65.13 | 63.01 |
|  | ORPO-Llama-3-8B | 60.41 | 57.69 | 77.9 | 55.88 | 82.62 | 64.93 | 66.57 |
|  | SPIN-ORPO-Llama-3-8B-Iter3 | 61.09 | 56.87 | 75.22 | 50.80 | 84.31 | 63.12 | 65.24 |
|  | SAPO-ORPO-Llama-3-8B | 61.95 | 59.00 | 79.08 | 56.33 | 83.48 | 64.31 | 67.36 |
|  | mistralai/Mistral-7B-v0.1 | 61.52 | 42.58 | 77.58 | 37.53 | 83.44 | 62.36 | 60.84 |
|  | ORPO-Mistral-7B | 62.80 | 54.41 | 77.90 | 45.26 | 84.16 | 60.82 | 64.23 |
|  | SPIN-ORPO-Mistral-7B-Iter3 | 56.48 | 52.91 | 70.80 | 39.88 | 77.65 | 59.35 | 59.51 |
|  | SAPO-ORPO-Mistral-7B | 63.14 | 55.00 | 79.16 | 46.70 | 85.02 | 61.42 | 65.07 |
| DPO-Based | Meta-Llama-3-8B-SFT | 54.86 | 51.73 | 76.72 | 44.81 | 81.01 | 63.57 | 63.95 |
|  | DPO-Llama-3-8B | 57.00 | 53.58 | 77.11 | 46.25 | 81.81 | 63.82 | 63.26 |
|  | SPIN-DPO-Llama-3-8B-Iter3 | 56.40 | 55.80 | 77.98 | 50.34 | 82.06 | 63.73 | 64.38 |
|  | SAPO-DPO-Llama-3-8B | 57.76 | 55.65 | 78.85 | 52.39 | 82.83 | 63.75 | 65.21 |
|  | wandb/mistral-7b-zephyr-sft | 62.63 | 54.00 | 76.32 | 44.88 | 84.77 | 60.93 | 63.92 |
|  | DPO-Mistral-7B | 63.14 | 55.81 | 75.69 | 41.02 | 85.16 | 60.97 | 63.63 |
|  | SPIN-DPO-Mistral-7B-Iter3 | 64.42 | 55.46 | 76.95 | 44.66 | 85.00 | 60.93 | 64.57 |
|  | SAPO-DPO-Mistral-7B | 63.99 | 57.47 | 76.32 | 45.11 | 85.42 | 59.79 | 64.68 |

### 4.3 Ablation Study

In Table 3, our ablation study of the LLaMA-3-8B model using the ORPO algorithm showed that on-policy sampling led to notable declines in performance on benchmarks like IFEval and the Open LLM Leaderboard, which test instruction-following and question-answering capabilities, respectively. This underperformance is likely due to the inherent volatility of on-policy sampling, where rapid shifts in model parameters and fluctuations in training data contribute to inconsistent training outcomes. Conversely, our off-policy strategy using an EMA model with a replay buffer produces more stable and representative data, especially useful when the policy model frequently updates. This approach prevents deviations in behavior that could arise

Table 2: Evaluation on IFEval, MT-Bench, AlpacaEval 2.0.

| Cases | Method | IFEval | MT Bench |  |  | AlpacaEval 2.0 |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | First Turn | Second Turn | Average | LC Win-Rate | Win-Rate | Length |
| ORPO-Based | ORPO-Llama-3-8B | 49.69 | 7.58 | 7.16 | 7.37 | 9.65 | 8.11 | 1599 |
|  | SPIN-ORPO-Llama-3-8B-Iter3 | 48.34 | 7.38 | 6.54 | 6.96 | 14.85 | 13.58 | 1725 |
|  | SAPO-ORPO-Llama-3-8B | 50.39 | 7.76 | 7.14 | 7.45 | 9.72 | 8.37 | 1507 |
|  | ORPO-Mistral-7B | 57.78 | 7.52 | 6.81 | 7.17 | 13.63 | 10.44 | 1358 |
|  | SPIN-ORPO-Mistral-7B-Iter3 | 44.68 | 7.17 | 6.51 | 6.84 | 12.55 | 11.11 | 1610 |
|  | SAPO-ORPO-Mistral-7B | 57.60 | 7.43 | 6.86 | 7.15 | 15.56 | 11.41 | 1333 |
| DPO-Based | Meta-Llama-3-8B-SFT | 41.43 | 7.12 | 6.48 | 6.80 | 7.22 | 5.27 | 1200 |
|  | DPO-Llama-3-8B | 45.50 | 7.40 | 6.68 | 7.04 | 8.94 | 6.67 | 1246 |
|  | SPIN-DPO-Llama-3-8B-Iter3 | 45.90 | 7.66 | 6.99 | 7.32 | 10.35 | 13.19 | 2289 |
|  | SAPO-DPO-Llama-3-8B | 48.28 | 7.61 | 7.16 | 7.38 | 9.73 | 9.66 | 1833 |
|  | wandb/mistral-7b-zephyr-sft | 35.35 | 7.47 | 6.65 | 7.06 | 5.47 | 21.94 | 8193 |
|  | DPO-Mistral-7B | 35.33 | 7.57 | 6.78 | 7.18 | 5.41 | 21.17 | 7937 |
|  | SPIN-DPO-Mistral-7B-Iter3 | 38.65 | 7.25 | 6.75 | 7.00 | 4.15 | 8.58 | 6683 |
|  | SAPO-DPO-Mistral-7B | 44.60 | 7.72 | 7.04 | 7.38 | 11.20 | 15.08 | 2789 |

Table 3: Ablation of training paradigm.

|  | Open LLM LeaderBoard | IFEval | MT-Bench |
| :--- | :---: | :---: | :---: |
| on-policy | 65.97 | 36.73 | $\mathbf{7 . 4 9}$ |
| no segment | 67.18 | $\mathbf{5 2 . 0 0}$ | 7.32 |
| Ours | $\mathbf{6 7 . 3 6}$ | 50.39 | 7.45 |

Table 4: Ablation of Reference Model Update Strategies.

| Model | Variant | Open LLM LeaderBoard | IFEval |
| :--- | :--- | :---: | :---: |
| LLaMA | fix-ref | 64.45 | 44.34 |
|  | policy-ref | 64.81 | 47.05 |
|  | ema-ref | $\mathbf{6 5 . 2 1}$ | $\mathbf{4 8 . 2 8}$ |
| Mistral | fix-ref | 64.50 | 41.14 |
|  | policy-ref | $\mathbf{6 4 . 7 7}$ | 41.47 |
|  | ema-ref | 64.68 | $\mathbf{4 4 . 6 0}$ |

from sampling with an unstable policy model, enhancing training consistency. Meanwhile, generating rejected responses from scratch demonstrated improved performance on IFEval, as it was not constrained by previously chosen responses and could more freely align with the prompt's instructions. However, this approach underperformed on other benchmarks, indicating that completely unrestricted generation may yield lower-quality responses that negatively impact the model's abilities in question-answering and dialogue tasks. Overall, our approach achieved better results across a range of metrics, validating the effectiveness of our training paradigm in promoting stable and consistent responses.

Table 4 presents an ablation on reference model updating strategies based on DPO for LLaMA and Mistral models. It evaluates three approaches: fixed reference (fix-ref), where the reference model remains static; policy reference (policy-ref), updated at intervals with the current policy's weights; and EMA reference (ema-ref), updated periodically with weights from an EMA model. The results highlight the importance of regularly updating the reference model; if the reference model remains unchanged, the learned model might be overly regularized towards the initial SFT model, potentially degrading performance on more complex tasks. The ema-ref update strategy shows the best performance, indicating that smoother updates significantly enhance model stability during training. For our DPO-based experiments, we implemented the ema-ref update strategy.

The ablation study shown in Figure 2 demonstrates how varying training epochs impact the LLaMA-3-8B model's performance under the ORPO algorithm across different benchmarks. As epochs increases, Open LLM Leaderboard scores decrease, signaling potential overfitting and aligning with the "alignment tax" phenomenon Askell et al. 2021 where excessive alignment with human preferences can harm general question-answering abilities. Conversely, IFEval scores improve, indicating enhanced instruction-following skills. MT-Bench scores peak at four epochs, suggesting this as the optimal training length to prevent overfitting. Notably, the Alpaca LC Win Rate also climbs, particularly after the fifth epoch. Given that we are training on the
![](https://cdn.mathpix.com/cropped/2024_06_04_a5fc7ba917d83ce5a449g-10.jpg?height=430&width=1654&top_left_y=234&top_left_x=232)

Figure 2: Ablation of training epochs on LLaMA-3-8B using ORPO across multiple benchmarks.

Distilabel-Capybara multi-turn dialogue dataset, our primary focus is on the MT-Bench metrics. After considering the overall performance across various benchmarks, we set the training epoch to 4 . More ablation results can be found in Appendix B.

## 5 Related Work

### 5.1 Reinforcement Learning from Human Feedback (RLHF)

RLHF Methods can be categorized into two primary approaches: reward-based methods and reward-free methods. Reward-based methods like Proximal Policy Optimization (PPO) Schulman et al. 2017 utilize a trained reward model [Ziegler et al., 1909, Stiennon et al., 2020, Ouyang et al., 2022] to provide feedback signals for online RL algorithms. The training of multiple models (policy, reward, and advantage model) increases computational demands and can lead to instability during the training process Gao et al., 2023, Wang et al. 2024a. In contrast, reward-free methods simplify the training process by eliminating the need for a separate reward model. DPO Rafailov et al. 2023 integrates the reward modeling stage directly into the preference learning stage. This method, based on a closed-form solution derived from the Bradley-Terry model Bradley and Terry 1952, is noted for its efficiency and stability. Zhao et al., 2023 introduce Sequence Likelihood Calibration with Human Feedback (SLiC-HF), which employs a contrastive ranking calibration loss combined with a regularization loss to refine the scoring of responses. The Kahneman-Tversky Optimization (KTO) algorithm Ethayarajh et al. 2024 leverages human utility principles to optimize language models using unpaired data, moving away from the dependency on pairwise preference datasets. Relative Preference Optimization (RPO) Yin et al., 2024 utilizes a contrastive weighting mechanism that evaluates preferences across not only identical but also semantically similar prompts, allowing for both paired and unpaired data scenarios. Recently, the ORPO algorithm Hong et al. 2024 simplifies preference alignment by integrating supervised fine-tuning and preference optimization into a single training stage without requiring a reference model. However, a major issue with offline contrastive preference learning approaches is their dependence on static, pre-collected paired preference datasets, typically involving a single optimization procedure. This reliance can lead to a distribution shift between the offline training data and the fine-tuned model, potentially impacting the model's effectiveness and adaptability.

### 5.2 Iterative Fine-Tuning LLMs

Iterative fine-tuning enhances language models by using outputs from the model itself or external models as inputs for subsequent training iterations, aiming to improve performance from each training cycle. A family of iterative methods Li et al., 2023, Gulcehre et al., 2023, Hu et al., 2023, Mukobi et al., 2023 involves continuously refining language models by supervised fine-tuning models on carefully selected, preferred responses. This iterative approach is further applied within the DPO framework, as demonstrated by a
number of recent works Xu et al., 2023, Xiong et al., 2024, Yuan et al., 2024, Chen et al., 2024. Utilizing iterative DPO-type training, updated models generate new preference pairs at each iteration Xu et al., 2023, Xiong et al., 2024, Guo et al., 2024. These pairs are then scored using feedback from additional reward models or human evaluations. Yuan et al. 2024 introduce Self-Rewarding Language Models, where the model annotates its own responses. Integrated into the iterative DPO framework, this allows the model to autonomously generate and assess preference pairs, streamlining fine-tuning by reducing external feedback reliance. However, this self-judging approach heavily relies on the model's own evaluative capabilities, making it more suitable for larger parameter language models. The SPIN algorithm Chen et al., 2024 uses a self-play framework for iterative DPO-style fine-tuning, labeling human-generated responses as winners and model-generated ones as losers. However, SPIN generates datasets for the next cycle offline, which limits the incorporation of fresh outputs from updated models. Additionally, its reliance on offline learning can lead to a shift problem as the fine-tuned model increasingly diverges from the one used to generate the preference dataset. To address these challenges, we have integrated real-time data sampling within the self-play framework, facilitating immediate updates to the training data. Some recent works Rosset et al., 2024, Wu et al. 2024 have also highlighted the importance of online iteration in preference learning. Unlike these approaches, which often rely on additional reward models or more advanced teacher models like GPT-4, our method seeks to develop a general and effective self-augmented algorithm that functions independently of external supervision.

Some research focuses on domain-specific applications to self-improve language models. For instance, Lee et al. 2024 target low-data regime tasks, optimizing language models with limited initial datasets. Other works concentrate on enhancing reasoning capabilities, as seen in Pang et al. 2024. In contrast, our research is primarily centered on enhancing language models for general instruction-following tasks.

## 6 Conlusion

In this paper, we introduce the Self-Augmented Preference Optimization (SAPO) framework, a dynamic off-policy learning paradigm that updates training data in real-time. Leveraging an Exponential Moving Average (EMA) model and a replay buffer, SAPO ensures stable and consistent performance, drastically reducing dependence on large pre-collected datasets. Through extensive evaluations across diverse Large Language Model (LLM) architectures such as LLaMA-3-8B and Mistral-7B, and using contrastive preference learning algorithms like DPO and ORPO, our method demonstrates superior performance on benchmarks including the Open LLM Leaderboard, IFEval, MT-Bench, and AlpacaEval 2.0. Furthermore, our method's

independence from annotated paired data and freedom from iterative training as seen in SPIN, positions it for broader applicability in diverse large-scale post-training tasks, pointing towards promising future directions.

## References

Ayush Agrawal, Lester Mackey, and Adam Tauman Kalai. Do language models know when they're hallucinating references? arXiv preprint arXiv:2305.18248, 2023.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open LLM leaderboard. Hugging Face, 2023.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39:324-345, 1952.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. ICML, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. NeurIPS, 30, 2017.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In ICML, pages $10835-10866,2023$.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. Direct language model alignment from online AI feedback. arXiv preprint arXiv:2402.04792, 2024.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861-1870. PMLR, 2018.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Jiwoo Hong, Noah Lee, and James Thorne. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691, 2024.

Jian Hu, Li Tao, June Yang, and Chandler Zhou. Aligning language models with offline reinforcement learning from human feedback. arXiv preprint arXiv:2308.12050, 2023.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Nicholas Lee, Thanakul Wattanawong, Sehoon Kim, Karttikeya Mangalam, Sheng Shen, Gopala Anumanchipali, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. LLM2LLM: Boosting llms with novel iterative data enhancement. arXiv preprint arXiv:2403.15042, 2024.

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. arXiv preprint arXiv:2308.06259, 2023.

Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. Towards understanding and mitigating social biases in language models. In ICML, pages 6565-6576, 2021.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.

Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Gabriel Mukobi, Peter Chatain, Su Fong, Robert Windesheim, Gitta Kutyniok, Kush Bhatia, and Silas Alberti. SuperHF: Supervised iterative learning from human feedback. arXiv preprint arXiv:2310.16763, 2023.

OpenAI. GPT-4 technical report. ArXiv, abs/2303.08774, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 35:27730-27744, 2022.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024.

Pulkit Pattnaik, Rishabh Maheshwary, Kelechi Ogueji, Vikas Yadav, and Sathwik Tejaswi Madhusudhan. Curry-DPO: Enhancing alignment using curriculum learning \& ranked preferences. arXiv preprint arXiv:2403.07230, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 2023.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, pages 99-106, 2021.

Arthur L Samuel. Some studies in machine learning using the game of checkers. IBM Journal of research and development, 3(3):210-229, 1959 .

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. The woman worked as a babysitter: On biases in language generation. arXiv preprint arXiv:1909.01326, 2019.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In ICML, pages 31210-31227, 2023.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. NeurIPS, 33:3008-3021, 2020.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of RLHF in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080, 2024a.

Haoyu Wang, Guozheng Ma, Ziqiao Meng, Zeyu Qin, Li Shen, Zhong Zhang, Bingzhe Wu, Liu Liu, Yatao Bian, Tingyang $\mathrm{Xu}$, et al. Step-on-feet tuning: Scaling self-alignment of LLMs via bootstrapping. arXiv preprint arXiv:2402.07610, 2024b.

Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv preprint arXiv:2208.06193, 2022.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675, 2024.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. In ICML, 2024.

Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan Wang, Hongtao Xie, and Yongdong Zhang. Curriculum learning for natural language understanding. In $A C L$, pages 6095-6104, 2020.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

Yueqin Yin, Zhendong Wang, Yi Gu, Hai Huang, Weizhu Chen, and Mingyuan Zhou. Relative preference optimization: Enhancing LLM alignment through contrasting responses across identical and diverse prompts. arXiv preprint arXiv:2402.10958, 2024.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Selfrewarding language models. ICML, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-HF: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. NeurIPS, 2024.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arxiv 2019. arXiv preprint arXiv:1909.08593, 1909 .

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
</end of paper 2>


<paper 3>
# Decoding Data Quality via Synthetic Corruptions: Embedding-guided Pruning of Code Data 

Yu Yang ${ }^{1,2 *}$<br>yuyang@cs.ucla.edu

Aaditya K. Singh ${ }^{2}$<br>aaditya.singh.21@ucl.ac.uk

Anas Mahmoud ${ }^{2}$<br>nas.mahmoud@mail.utoronto.ca

Kushal Tirumala ${ }^{2}$<br>ktirumala@meta.com

Mostafa Elhoushi ${ }^{2}$<br>melhoushi@meta.com

Fabian Gloeckle ${ }^{2}$

fgloeckle@meta.com

Baptiste Rozière $^{2} \quad$ Carole-Jean Wu ${ }^{2} \quad$ Ari S. Morcos ${ }^{3 \dagger} \quad$ Newsha Ardalani $^{2}$<br>broz@meta.com carolejeanwu@meta.com<br>new@meta.com<br>${ }^{1}$ UC Los Angeles $\quad{ }^{2}$ FAIR at Meta ${ }^{3}$ DatologyAI


#### Abstract

Code datasets, often collected from diverse and uncontrolled sources such as GitHub, potentially suffer from quality issues, thereby affecting the performance and training efficiency of Large Language Models (LLMs) optimized for code generation. Previous studies demonstrated the benefit of using embedding spaces for data pruning, but they mainly focused on duplicate removal or increasing variety, and in other modalities, such as images. Our work focuses on using embeddings to identify and remove "low-quality" code data. First, we explore features of "low-quality" code in embedding space, through the use of synthetic corruptions. Armed with this knowledge, we devise novel pruning metrics that operate in embedding space to identify and remove low-quality entries in the Stack dataset. We demonstrate the benefits of this synthetic corruption informed pruning (SCIP) approach on the well-established HumanEval and MBPP benchmarks, outperforming existing embedding-based methods. Importantly, we achieve up to a $3 \%$ performance improvement over no pruning, thereby showing the promise of insights from synthetic corruptions for data pruning.


## 1 Introduction

Machine learning, and in particular Large Language Models (LLMs), are transforming a wide range of industries. Their capabilities extend even to specialized tasks like code generation and medical diagnostics, thus amplifying their societal and economic impact [1]. In this race for higher performance, some training datasets have swelled to petabyte size, sourced from extensive repositories like the Common Crawl. While significant effort has gone into optimizing the computational aspects of training LLMs, such as hardware acceleration and algorithmic improvements [2], the question of data efficiency is still relatively under-explored. Data efficiency is not merely a computational concern but is intrinsically tied to the quality of the training data. The use of large, but ineffective, datasets can result in protracted training times, higher energy consumption, and ultimately, models that are expensive to deploy and maintain [3].[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_7be1f7b7784d00fe6fcfg-02.jpg?height=551&width=1395&top_left_y=234&top_left_x=365)

Figure 1: Schematic of SCIP. First, we synthetically corrupt code data, which tends to move code embeddings to smaller clusters or further from cluster centroids. Then, we use this insight to propose a new pruning metric, resulting in improved training efficiency and better end performance.

Code datasets, usually compiled from diverse, open-source platforms like GitHub, are often riddled with inconsistencies, errors, or low-quality code snippets. These issues not only undermine the model's final performance but also affect the efficiency and effectiveness of the training process. The presence of such low-quality data essentially "pollutes" the learning environment, leading to suboptimal results. Therefore, improving data quality is not merely an ancillary task but a fundamental requirement for achieving the full potential of code-generating LLMs. A recent study [4] showcased the benefits of so-called "textbook-quality" data in enhancing model efficiency for code-generation tasks. However, their strategy relies heavily on generating closed-source data with GPT-3.5 and then filtering it based on GPT-4 [5] predictions, both of which are proprietary models, thus making this approach less accessible for many researchers due to high costs and difficulty of reproducibility. Furthermore, another study [6] highlighted potential issues with training on generated outputs. This emphasizes the need for open-source techniques to identify valuable data in existing, large-scale, natural corpora.

Building upon these identified challenges and gaps in existing research, we focus on easy-to-use, accessible pruning methods for the large open-source Stack dataset [7]. To this end, we take inspiration from recent approaches to data pruning in the domains of image [3] and multimodal models [8], which make use of pre-trained embedding spaces to identify useful or duplicate data, to keep or prune, respectively. In the hitherto unexplored domain of code, we introduce synthetic corruption informed pruning (SCIP): First, we identify what constitutes "low-quality" data in embedding space through controlled corruption of existing data, and find that corrupted code tends to reside in smaller clusters and often be farther from cluster centroids. Then, we introduce a pruning strategy, based on these insights, that ranks data points based on their cluster size and distance to the nearest centroid, aiming to remove a predefined fraction of the data. Using these embedding-based methods for pruning low-quality code, we demonstrate improvements in performance and training efficiency on widely used benchmarks [9, 10].

## 2 What Does Low-Quality Mean for Code Data?

### 2.1 Definition of Low-Quality Data

Let $\mathcal{D}$ be the original dataset, $\mathcal{Q} \subseteq \mathcal{D}$ be a subset, and $\mathcal{D}_{\text {test }}$ be the test set. Let $x_{\text {test }, i}$ be the $i$-th test example in $\mathcal{D}_{\text {test }}$. First, we define a general metric $M$, which could potentially be pass $@ \mathrm{k}$ [9] or any other quality metric. We then define $M\left(\theta(\mathcal{D}), \mathcal{D}_{\text {test }}\right.$ ) as the expectation of a particular metric (for example, pass $@ \mathrm{k}_{i}$ ) over all $x_{\text {test }, i}$ in $\mathcal{D}_{\text {test }}$ when training on dataset $\mathcal{D}$ with model parameters $\theta$ :

$$
M\left(\theta(\mathcal{D}), \mathcal{D}_{\text {test }}\right)=\mathbb{E}_{x_{\text {test }, i} \in \mathcal{D}_{\text {test }}}\left[\text { pass } @ \mathrm{k}_{i}\right]
$$

The set $\mathcal{Q}$ is defined as "low-quality" if the following inequality holds:
![](https://cdn.mathpix.com/cropped/2024_06_04_7be1f7b7784d00fe6fcfg-03.jpg?height=626&width=1266&top_left_y=234&top_left_x=424)

Figure 2: Corrupted data tends to reside in smaller clusters (top row) and farther from centroids (bottom row) when compared to the original, uncorrupted data. The effects are more pronounced for syntax errors (left two columns) as compared to content errors (right two columns). Red dotted line indicates mean, black dotted line indicates 0. More details and analysis can be found in Appendix B. 2 .

$$
M\left(\theta(\mathcal{D}), \mathcal{D}_{\text {test }}\right)<M\left(\theta(\mathcal{D} \backslash \mathcal{Q}), \mathcal{D}_{\text {test }}\right)
$$

In simpler terms, $\mathcal{Q}$ is considered "low-quality" data if removing it from $\mathcal{D}$ improves the score of the general metric $M$ on $\mathcal{D}_{\text {test }}$.

### 2.2 SCIP: Two-Step Framework for Identifying Low-Quality Data

To systematically identify low-quality data, we propose a two-step framework, illustrated in Figure 1. The first step involves the creation of data with known errors, serving as markers for low-quality data. From this first step, we gather insights on how corruption affects embeddings (obtained with a pretrained model), and use this knowledge to prune data with similar embedding properties.

Synthetic Corruption Generation To identify and prune "low-quality" code data, it's important to understand its possible forms. We consider two main domains: syntax errors and content errors. Synthetic corruption has the benefit of creating matched pairs of higher and lower quality data, making it more controlled than alternative approaches which could be confounded by style.

- Data with Syntax Errors: Syntax errors are clear indicators of bad code, preventing a file from executing successfully. Such issues can be as common as unmatched parentheses or as nuanced as referencing undeclared variables. To intentionally introduce these errors for the sake of our experiments, we employ two main corruptions: removing closed brackets (specifically, ') ', ' $]$ ', '\}') and renaming variables to syntactically invalid names.
- Data with Content Errors: Although such code may run without immediate issues, its output might diverge from the intended result due to underlying logical errors. To simulate this, we either alter conditional operators (through negation) or offset array indices (changing ' $i$ ' to ' $i+1$ ') to disrupt data access patterns.

More specifics can be found in Appendix B. Through these synthetic corruptions, we ensure a systematic introduction of both syntax and content errors, aiding in a more comprehensive identification of "low-quality" data. By focusing on a representative sample of errors, we effectively set the stage for the next step: identifying and pruning "low-quality" data in large-scale datasets.

Data Pruning Informed by Synthetic Corruptions In the embedding space of a pre-trained code embedding model, StarEncoder [11], we see that synthetic corruption exhibits a distinct change: corruption moves points to smaller clusters or further out from centroids, as compared to the original, uncorrupted code (Fig. 22). These insights shape our pruning strategy. By focusing on data in smaller

Table 1: Pass@ 1 performance on HumanEval and MBPP for different pruning methods with $20 \%$ files pruned.

|  | No <br> pruning | Random <br> Pruning | SSL <br> Prototype | SemDeDup | D4 | Small <br> Clusters | Far from <br> Centroids | Combined <br> Small+Far |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HumanEval | $25.0 \%$ | $24.0 \%$ | $23.8 \%$ | $20.7 \%$ | $23.2 \%$ | $23.2 \%$ | $\underline{26.8 \%}$ | $\mathbf{2 8 . 0 \%}$ |
| MBPP | $\underline{33.4 \%}$ | $31.9 \%$ | $32.2 \%$ | $32.4 \%$ | $31.2 \%$ | $\mathbf{3 5 . 0 \%}$ | $30.8 \%$ | $33.0 \%$ |

clusters and distant from centroids, we aim to efficiently identify and remove low-quality data from the original dataset. A formal version of the algorithm, with pseudocode can be found in Appendix $\mathrm{C}$

## 3 Pruning Low-quality Data for More Efficient Training

### 3.1 Experiment Setup

Dataset. Our experiments utilize the Stack v1.1 dataset [7], which is sourced from GitHub repositories published from 2015 to 2022, and specifically designed for code generation tasks. Although the dataset includes code from 358 different programming languages, we narrow our focus solely to Python to ensure a more controlled study. This results in a dataset of $12.6 \mathrm{M}$ files and $20.4 \mathrm{~B}$ tokens.

Model and Training Details. Following the methodology of the current state-of-the-art open-source model, Code Llama [12], we fine-tune a 1.5B LLaMA [13] model instead of training from scratch. The model has 48 layers, 24 heads per layer, and inner dimension of 1536 . All experiments are run on 32 NVIDIA A100 GPUs with fully-sharded data parallel [14]. We use a learning rate of 3e-4, a batch size of 576, a sequence length of 2048, and train for 56,000 steps ( $\sim 67 \mathrm{~B}$ tokens).

### 3.2 Evaluation

Our evaluation employs two well-established benchmarks in the code generation field: HumanEval [9] and MBPP [10]. The primary metric for evaluation across these benchmarks is "pass @k," which measures the percentage of test cases that are correctly solved within the top-k generated code snippets. For baselines, we compare to no pruning, random pruning (averaged over 3 seeds), and three other pruning methods using embeddings, based on prior work in other modalities: SSL-prototypes [3], SemDeDup [8], and D4 [15]. Additional details can be found in Appendix D

### 3.3 Results

In Table 1, our proposed methods - pruning data that are "Far from Centroid" and within "Small Clusters" - yield clear performance improvements on HumanEval and MBPP, respectively. However, better performance on one benchmark often comes at the expense of the other, perhaps due to the different natures of these tasks. Motivated by the strong performance of our two suggested methods, we experimented with a combined method: first pruning files from small clusters, then files far from centroids, with the ratio between these defined by a parameter $\alpha$. We found that $\alpha=0.8$ performed best (see Appendix C). Impressively, this combined method achieves the best performance of all methods tried on HumanEval, a full $3 \%$ above no pruning and better than all prior work on embedding-based pruning, while also remaining competitive with no pruning on MBPP.

We also observe in Fig. 1 that "Far from Centroid" and "Small Clusters" both achieve an efficiency speedup (both methods achieve the baseline pass @ 1 rate in fewer training steps). Further insights into the qualitative attributes of pruned data are presented in Fig. 44."

## 4 Conclusions

We introduce SCIP, a systematic method to identify and remove "low-quality" code data from large datasets. Building on the insights of the value of high-quality data presented in earlier studies [4], our work goes further by offering accessible, open-source, and cost-effective pruning techniques through the use of embedding spaces. We go beyond prior work in embedding-based pruning [3, 8, 15] by motivating heuristics through identification of "low-quality" data via synthetic corruptions: we
systematically create code discrepancies, both in syntax and content, to understand their influence on the embedding space. Our findings reveal that syntax errors lead to significant shifts away from cluster centroids and into smaller clusters. Leveraging these observations, we designed pruning methods that consider both distances to centroids and cluster sizes to effectively identify and remove low-quality data. Applying these pruning methods leads to better performance on code generation benchmarks, showing the promise of insights from synthetic corruptions for improving pruning techniques.

More broadly, our results underscore the significance of rigorous data curation. Beyond just code, more rigorously examining "low-quality" data could lead to more informed pruning techniques. Similar to how code can have both syntax and content discrepancies, natural language data too can have structural (e.g., grammatical) and semantic (e.g., factually incorrect) anomalies. In future work, the strategies and methodologies established here of using synthetically corrupted data as a pruning signal could be extended and adapted to general natural language datasets, ensuring models trained on them produce more accurate, reliable, and coherent outputs.

## Acknowledgments

We would like to sincerely thank Jack Lanchantin for the insightful discussions, and Shubham Toshniwal, Koustuv Sinha, and Alberto Bietti for generously sharing their valuable insights drawn from their previous research.

## References

[1] Tyna Eloundou, Sam Manning, Pamela Mishkin, and Daniel Rock. Gpts are gpts: An early look at the labor market impact potential of large language models, 2023.

[2] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R'e. Flashattention: Fast and memory-efficient exact attention with io-awareness. ArXiv, abs/2205.14135, 2022.

[3] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523-19536, 2022.

[4] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.

[5] OpenAI. Gpt-4 technical report, 2023.

[6] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The curse of recursion: Training on generated data makes models forget, 2023.

[7] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code. Preprint, 2022.

[8] Amro Kamal Mohamed Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S. Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. In ICLR 2023 Workshop on Multimodal Representation Learning: Perks and Pitfalls, 2023.

[9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[10] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[11] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João

Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you! 2023.

[12] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[13] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[14] FairScale authors. Fairscale: A general purpose modular pytorch library for high performance and large scale training. https://github.com/facebookresearch/fairscale, 2021.

[15] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari S Morcos. D4: Improving llm pretraining via document de-duplication and diversification. arXiv preprint arXiv:2308.12284, 2023.

[16] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in neural information processing systems, 33:9912-9924, 2020.

[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR, 2021.

[18] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
</end of paper 3>


<paper 4>
# Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models 

Avi Singh ${ }^{1, *}$, John D Co-Reyes ${ }^{1, *}$, Rishabh Agarwal ${ }^{1,2, *}$,<br>Ankesh Anand ${ }^{1}$, Piyush Patil ${ }^{1}$, Xavier Garcia ${ }^{1}$, Peter J. Liu ${ }^{1}$, James Harrison ${ }^{1}$, Jaehoon Lee ${ }^{1}$, Kelvin Xu ${ }^{1}$,<br>Aaron Parisi ${ }^{1}$, Abhishek Kumar ${ }^{1}$, Alex Alemi ${ }^{1}$, Alex Rizkowsky ${ }^{1}$, Azade Nova ${ }^{1}$, Ben Adlam ${ }^{1}$, Bernd Bohnet ${ }^{1}$,<br>Gamaleldin Elsayed ${ }^{1}$, Hanie Sedghi ${ }^{1}$, Igor Mordatch ${ }^{1}$, Isabelle Simpson ${ }^{1}$, Izzeddin Gur ${ }^{1}$, Jasper Snoek ${ }^{1}$,<br>Jeffrey Pennington ${ }^{1}$, Jiri Hron ${ }^{1}$, Kathleen Kenealy ${ }^{1}$, Kevin Swersky ${ }^{1}$, Kshiteej Mahajan ${ }^{1}$, Laura Culp ${ }^{1}$, Lechao<br>Xiao ${ }^{1}$, Maxwell L Bileschi ${ }^{1}$, Noah Constant ${ }^{1}$, Roman Novak ${ }^{1}$, Rosanne Liu ${ }^{1}$, Tris Warkentin ${ }^{1}$, Yundi Qian ${ }^{1}$,<br>Yamini Bansal ${ }^{1}$, Ethan Dyer ${ }^{1}$, Behnam Neyshabur ${ }^{1}$, Jascha Sohl-Dickstein ${ }^{1}$, Noah Fiedel ${ }^{1}$<br>"Contributed equally, ${ }^{1}$ Google DeepMind, ${ }^{2}$ Mila

Fine-tuning language models (LMs) on human-generated data remains a prevalent practice. However, the performance of such models is often limited by the quantity and diversity of high-quality human data. In this paper, we explore whether we can go beyond human data on tasks where we have access to scalar feedback, for example, on math problems where one can verify correctness. To do so, we investigate a simple self-training method based on expectation-maximization, which we call ReST ${ }^{E M}$, where we (1) generate samples from the model and filter them using binary feedback, (2) fine-tune the model on these samples, and (3) repeat this process a few times. Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, we find that $\operatorname{ReST}^{E M}$ scales favorably with model size and significantly surpasses fine-tuning only on human data. Overall, our findings suggest self-training with feedback can reduce dependence on human-generated data.

Keywords: RL from external feedback, EM for RL, Language, LLMs, Reasoning, Coding, Self-Improvement

## 1. Introduction

Large Language Models (LLMs) are revolutionizing the landscape of deep learning, showcasing remarkable capabilities in generating human-quality text and tackling diverse language tasks (Google et al., 2023; OpenAI, 2023). While supervised fine-tuning (SFT) on human-collected data further boosts their performance on tasks of interest, acquiring high-quality human data poses a significant bottleneck. This is particularly demanding for complex problem-solving tasks, requiring significant resources and expert knowledge. To address this hurdle, model-generated synthetic data emerges as a promising alternative, offering scalability and cost-effectiveness, provided its quality can be ensured. While LLMs hold the potential to self-evaluate generated data, this paper explores a simpler setting where an external, scalar feedback signal serves as a quality indicator for each generated sample.

To investigate training on model-generated data, we consider a simple yet powerful self-training approach for language models that requires only two capabilities: 1) generating samples from the model and 2) evaluating these samples with a scoring mechanism. This approach shares similarities with Reinforced Self-Training (ReST) proposed by Gulcehre et al. (2023). We make some modifications to ReST (detailed in Section 3), and call our approach ReST ${ }^{E M}$. We show that $\operatorname{ReST}^{E M}$ can be viewed as applying expectation-maximization for reinforcement learning (Dayan and Hinton, 1997; Peters and Schaal, 2007), which we present formally in Section 3. Specifically, $\operatorname{ReST}^{E M}$ alternates between the expectation and maximization steps:
![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-02.jpg?height=496&width=1642&top_left_y=288&top_left_x=207)

Figure 1 | Self-training with ReST ${ }^{E M}$ substantially improves test performance of PaLM 2 models on two challenging benchmarks: MATH and HumanEval. Results for other models are shown for general progress on these tasks and are typically not comparable due to difference in model scales. GPT-4 results are taken from Bubeck et al. (2023). The x-axis approximately denotes release time (not to scale).

1. Generate (E-step): The language model generates multiple output samples for each input context. Then, we filter these samples using a binary reward to collect the training dataset.
2. Improve (M-step): The original language model is supervised fine-tuned on the training dataset from the previous Generate step. The fine-tuned model is then used in the next Generate step.

ReST ${ }^{E M}$, with its various adaptations (Section 4), has demonstrated success in enhancing language models across diverse domains, including machine translation (Gulcehre et al., 2023; Norouzi et al., 2016), semantic parsing (Agarwal et al., 2019), preference alignment (Dong et al., 2023), and elementary reasoning (Yuan et al., 2023; Zelikman et al., 2022). However, prior works primarily applied training with self-generated data to relatively small language models (up to 7B parameters), with limited scalability observed for larger models (Yuan et al., 2023). Complementing these efforts, our work aims to investigate the effectiveness and scalability of model-generated synthetic data compared to human-generated data in two challenging, less explored domains: competition-level mathematical problem-solving (MATH) (Hendrycks et al., 2021b) and code generation (APPS) (Hendrycks et al., 2021a).

Our empirical findings reveal significant advancements in both mathematical reasoning and code generation capabilities when applying $\operatorname{ReST}^{E M}$ to PaLM 2 models of varying scales (Figure 1). Notably, models fine-tuned on model-generated synthetic data exhibit remarkably larger performance gains compared to those trained on human-written data (Figure 2, 3). Interestingly, exceeding a couple of iterations of $\operatorname{ReST}^{E M}$ leads to diminishing improvement, indicating potential overfitting on small amount of training problems (Figure 4). Additionally, models fine-tuned using ReST ${ }^{E M}$ improve pass@k as well as majority voting performance. Furthermore, these fine-tuned models demonstrate enhanced performance on related but held-out benchmarks, including math problems (GSM8K and Hungarian HS finals), coding (HumanEval), and Big-Bench Hard tasks. We also perform ablation studies to investigate the effect of number of model-generated solutions, training problems, and iterations for $\operatorname{ReST}^{E M}$ fine-tuning. Overall, our findings suggest self-training with feedback as a promising approach to reduce dependence on human data.

The key contributions of this work are:

- We introduce ReST ${ }^{E M}$ that enables learning from self-generated data for LLMs, employing a
principled expectation-maximization approach within a reinforcement learning framework.
- We demonstrate that training on self-generated solutions surpasses training on human-generated solutions in problem-solving domains, such as mathematics and code generation.
- Through comprehensive ablation studies, we pinpoint the crucial elements necessary for attaining optimal performance.
- LLMs fine-tuned with ReST ${ }^{E M}$ exhibit robust transfer capabilities across various held-out tasks.


## 2. Preliminaries

An autoregressive language model produces an output sequence $\boldsymbol{y}=\left(y_{1}, y_{2}, \ldots . y_{T}\right)$ given a context (or source input) $x=\left(x_{1}, x_{2}, \ldots x_{L}\right)$, where the tokens $x_{l}, y_{t}$ belong to a fixed vocabulary. Auto-regressive generation involves predicting tokens one at a time, based on the previously generated tokens. Assuming that the model is parameterized by $\theta$, the conditional probability distribution of generating a sequence $y$ given $x$ is

$$
p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})=\prod_{t=1}^{T} p_{\theta}\left(y_{t} \mid \boldsymbol{y}_{<t}, \boldsymbol{x}\right)
$$

with the convention $\boldsymbol{y}_{1: 0}=\emptyset$ and $y_{1: t-1}=\left(y_{1}, y_{2}, \ldots . y_{t-1}\right)$. For ease of notation, we define $p\left(y_{t} \mid x\right):=$ $p\left(y_{t} \mid y_{<t}, x\right)$. The probability of predicting $t^{\text {th }}$ token $y_{t}, p\left(y_{t} \mid x\right)$, is determined using a softmax with temperature $\gamma: p\left(y_{t} \mid x\right)=\frac{\exp \left(z_{t} / \gamma\right)}{\sum_{i=1}^{M} \exp \left(z_{i} / \gamma\right)}$, where $z_{t}$ is the logit score for the token $y_{t}$. Higher values of temperature $\gamma$ introduces more randomness, while a lower value makes the output more deterministic by favoring the most probable words.

Given a dataset $\mathcal{D}$ of inputs $x$ and human-generated outputs $y$, supervised fine-tuning (SFT) trains the policy by minimizing the negative log likelihood loss:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{SFT}}(\theta)=-\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\sum_{t=1}^{T} \log p_{\theta}\left(y_{t} \mid y_{1: t-1}, \boldsymbol{x}\right)\right] \tag{1}
\end{equation*}
$$

We also assume access to a deterministic sequence-level (or terminal) reward $r(\boldsymbol{x}, \boldsymbol{y})$. Then, the reinforcement learning (RL) objective corresponds to:

$$
\mathcal{L}_{\mathrm{RL}}(\theta)=\mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{\boldsymbol{y} \sim p_{\theta}(y \mid x)}[r(\boldsymbol{x}, \boldsymbol{y})]\right] .
$$

Optimizing $\mathcal{L}_{\mathrm{RL}}$ loss directly using online RL methods, such as policy gradients, requires updating and sampling from the policy numerous times during training. However, the computational cost of fine-tuning on a continual flow of new samples becomes a limitation of online methods, especially when the sizes of the policy network grow to tens or hundreds of billion parameters. We discuss an alternative to such online RL approaches in the next section.

## 3. Expectation-Maximization for Reinforced Self-Training

Expectation-Maximization (EM) for RL We first describe the EM-based framework for RL with language models, building upon the prior work by Dayan and Hinton (1997). Let's define a binary optimality variable $\mathrm{O}$, such that $p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) \propto f(r(\boldsymbol{x}, \boldsymbol{y}))$, for some non-decreasing non-negative function $f: \mathbb{R} \rightarrow \mathbb{R}^{+}$. We want to maximize the log-likelihood of observing $O=1$ (obtaining high reward):

$$
\log p(O=1 \mid \boldsymbol{x}):=\log \sum_{y} p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x}) p(O=1 \mid \boldsymbol{x}, \boldsymbol{y})
$$

However, the sum over all possible sequences $y$ is typically intractable. Instead of maximizing $\log p(O=1 ; \boldsymbol{x})$, one can consider maximizing its $\operatorname{ELBO} L\left(p_{\theta}, q\right)$ with respect to parameters $\theta$ and variational distribution $q(y \mid x)$. Specifically,

$$
\begin{align*}
\log p(O=1 \mid \boldsymbol{x}) & =\log \mathbb{E}_{q(\boldsymbol{y} \mid \boldsymbol{x})}\left[\frac{p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})}{q(\boldsymbol{y} \mid \boldsymbol{x})}\right] \\
& \geq \mathbb{E}_{q(\boldsymbol{y} \mid \boldsymbol{x})}\left[\log \frac{p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})}{q(\boldsymbol{y} \mid \boldsymbol{x})}\right] \quad \text { (Jensen's inequality) } \\
& =\mathbb{E}_{q(\boldsymbol{y} \mid \boldsymbol{x})}[\log p(O=1 \mid \boldsymbol{x}, \boldsymbol{y})]-\mathrm{KL}\left[q(\boldsymbol{y} \mid \boldsymbol{x}) \| p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right] \\
& =: L\left(p_{\theta}, q\right) \tag{2}
\end{align*}
$$

The EM algorithm (Dempster et al., 1977) for Equation 2 alternates between an E-step and M-step: at iteration $t$, denote the language model parameter to be $\theta^{t}$ and the variational distribution to be $q^{t}$.

- E-step: $q^{t+1}=\arg \max _{q} L\left(p_{\theta^{t}}, q\right)$. Since $L\left(p_{\theta^{t}}, q\right)$ can be written as $-K L\left[q(\boldsymbol{y} \mid \boldsymbol{x}) \| q^{*}(\boldsymbol{y} \mid \boldsymbol{x})\right], q^{t+1}(\boldsymbol{y} \mid$ $\boldsymbol{x}) \propto q^{*}(\boldsymbol{y} \mid \boldsymbol{x}):=p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) p_{\theta^{t}}(\boldsymbol{y} \mid \boldsymbol{x})$. Thus, this step is equivalent to weighting the output samples from conditional language model distribution based on their likelihood of obtaining high rewards.
- M-step: $\theta^{t+1}:=\arg \max _{\theta} L\left(p_{\theta}, q^{t+1}\right)=\arg \min _{\theta} \operatorname{KL}\left[q^{t+1}(y \mid x) \| p_{\theta}(y \mid x)\right]=\arg \min _{\theta} \sum_{y}-q^{t+1}(y \mid$ $\boldsymbol{x}) \log p_{\theta}(y \mid x)$. As such, this step corresponds to maximizing a weighted negative $\log$-likelihood loss.

Alternating between above steps ensures a monotonic improvement in the ELBO: $L\left(p_{\theta^{t+1}}, q^{t+1}\right) \geq$ $L\left(p_{\theta^{t}}, q^{t+1}\right) \geq L\left(p_{\theta^{t}}, q^{t}\right)$.

EM with non-negative rewards. If the rewards are non-negative and $f$ is set to the identity function, then $p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) \propto r(\boldsymbol{x}, \boldsymbol{y})$ which implies $q^{t+1}(\boldsymbol{y} \mid \boldsymbol{x}) \propto r(\boldsymbol{x}, \boldsymbol{y}) p_{\theta^{t}}(\boldsymbol{y} \mid \boldsymbol{x})$. In this scenario, the updated policy parameters $\theta^{t+1}$ resulting from the M-step at iteration $t$ are given by:

$$
\begin{equation*}
\theta^{t+1}:=\arg \max _{\theta} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\mathbb{E}_{\boldsymbol{y} \sim p_{\theta}^{t}(\boldsymbol{y} \mid \boldsymbol{x})}\left[r(\boldsymbol{x}, \boldsymbol{y}) \log p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right]\right] \tag{3}
\end{equation*}
$$

Comparing the above equation with the typical RL objective ( $\mathcal{L}_{\mathrm{RL}}$ ) reveals the key distinction between standard RL and EM-based RL: how output data is sampled. Standard RL continuously updates the policy and uses this latest policy to collect data. In contrast, EM-based RL employs a fixed sampling policy from the previous iteration, decoupling data collection from policy optimization. This decoupling in EM-based approaches enables easier scaling to large policy networks, such as LLMs.

ReST ${ }^{E M}$ Motivated by the EM framework, we now discuss a simplified version of Reinforced SelfTraining (ReST) approach by Gulcehre et al. (2023). This approach, which we call ReST ${ }^{E M}$, decouples data collection (E-step) and policy optimization (M-step) in a typical RL pipeline. Algorithm 1 outlines the $\operatorname{ReST}^{E M}$ algorithm with multiple iterations, where each iteration corresponds to one Generate and Improve step. We describe these steps in detail below.

- Generate (E-step): In this step, we generate a dataset $\mathcal{D}_{i}$ by sampling many output sequences from the current policy $p_{\theta}: \mathcal{D}_{i}=\left\{\left.\left(\boldsymbol{x}^{j}, \boldsymbol{y}^{j}\right)\right|_{j=1} ^{N}\right.$ s.t. $\left.\boldsymbol{x}^{j} \sim \mathcal{D}, \boldsymbol{y}^{j} \sim p_{\theta}\left(\boldsymbol{y} \mid \boldsymbol{x}^{j}\right)\right\}$. Here, the inputs are resampled from the original dataset $x^{j} \sim \mathcal{D}$. The output sequences in $\mathcal{D}_{i}$ are then scored with a binary reward function $r(\boldsymbol{x}, \boldsymbol{y})$. In our experiments, we condition the language model using a few-shot prompt with programs for code generation and step-by-step solutions for math problems.

```
Algorithm 1: ReST (Expectation-Maximization). Given a initial policy (e.g., pre-trained
$\mathrm{LM}), \operatorname{ReST}{ }^{E M}$ iteratively applies Generate and Improve steps to update the policy.
    Input: $\mathcal{D}$ : Training dataset, $\mathcal{D}_{\text {val }}$ : Validation dataset, $\mathcal{L}(\boldsymbol{x}, \boldsymbol{y} ; \theta)$ : loss, $r(\boldsymbol{x}, \boldsymbol{y})$ : Non-negative
            reward function, $I$ : number of iterations, $N$ : number of samples per context
    for $i=1$ to $I$ do
        // Generate (E-step)
        Generate dataset $\mathcal{D}_{i}$ by sampling: $\mathcal{D}_{i}=\left\{\left.\left(\boldsymbol{x}^{j}, \boldsymbol{y}^{j}\right)\right|_{j=1} ^{N}\right.$ s.t. $\left.\boldsymbol{x}^{j} \sim \mathcal{D}, \boldsymbol{y}^{j} \sim p_{\theta}\left(\boldsymbol{y} \mid \boldsymbol{x}^{j}\right)\right\}$
            Annotate $\mathcal{D}_{i}$ with the reward $r(\boldsymbol{x}, \boldsymbol{y})$.
        // Improve (M-step)
        while reward improves on $\mathcal{D}_{\text {val }}$ do
            Optimise $\theta$ to maximize objective: $J(\theta)=\mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \mathcal{D}_{i}}\left[r(\boldsymbol{x}, \boldsymbol{y}) \log p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right]$
        end
    end
    Output: Policy $p_{\theta}$
```

- Improve (M-step): In the $i^{\text {th }}$ iteration, we use the new dataset $\mathcal{D}_{i}$ from Generate step to fine-tune the policy $p_{\theta}$. To mitigate task-specific over-fitting, we minimize drift from the base model by always fine tuning the base pretrained language model. For fine-tuning, we minimize the reward-weighted negative log-likelihood loss $J(\theta)=\mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \mathcal{D}_{i}}\left[r(\boldsymbol{x}, \boldsymbol{y}) \log p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right]$. Once the policy is improved, a new dataset of better quality samples can be created once again.

Differences with ReST (Gulcehre et al., 2023). Unlike ReST, we refrain from augmenting $\mathcal{D}_{i}$ in Generate step with human-generated outputs as such data may not always be optimal for learning or it might not be easily available. Furthermore, each Improve step fine-tunes the base model instead of the model obtained from the previous ReST iteration. This results in comparable task-specific performance but much better transfer performance on held-out tasks (see Figure 7).

Remark. Our experiments focus on problem-solving settings with binary rewards (either 0 or 1 ), unlike the bounded real-valued rewards assumed by Gulcehre et al. (2023). Specifically, for each Generate step, Gulcehre et al. (2023) perform multiple Improve steps, where each Improve step can be viewed as an M-step with the function $f(r(\boldsymbol{x}, \boldsymbol{y}))=r(\boldsymbol{x}, \boldsymbol{y})>\tau$, where $\tau \in \mathbb{R}^{+}$increases in successive M-steps. However, with binary rewards, any value of $\tau \in(0,1)$ corresponds to the identical Improve steps.

## 4. Related work

Several prior methods can be instantiated using the expectation-maximization framework presented in Section 3. We discuss methods and their relation to $\operatorname{ReST}^{E M}$ in this section.

- Expert Iteration (ExiT) (Anthony et al., 2017) alternates between two steps: expert improvement and policy distillation. During the expert improvement step (E-step), we combine a base policy with a search procedure to generate samples from a better policy, called the expert policy. Then, in the policy distillation step (M-step), we use these expert samples to train the base policy in a supervised way, effectively improving it to match the expert policy. While ExiT used monte-carlo tree-search, we simply use temperature sampling for collecting samples from the expert policy in ReST. That said, improving the E-step in ReST using the ExIT framework via search and planning procedures with language models would be interesting for future work. For example, Huang et al. (2022) implement a single iteration of $\operatorname{ReST}^{E M}$ on simple math reasoning
problems. However, unlike our setup, they do not assume access to a correctness reward and instead employ majority-voting (Wang et al., 2023) as a search procedure within the E-step.
- Self-Taught Reasoner (STaR) (Zelikman et al., 2022) employed greedy decoding instead of temperature sampling for the E-step in $\operatorname{ReST}^{E M}$, which is restricted to one model-generated solution per problem during data collection. Additionally, STaR proposed rationalization as an alternative to temperature sampling, where the language model is provided with the correct answer as part of the input to generate correct solutions for difficult problems. However, in our preliminary experiments, rationalization leads to substantial increase in false positive solutions that result in correct answer but with incorrect reasoning.
- Rejection Sampling Fine-tuning (RFT) (Yuan et al., 2023) improves reasoning performance on GSM8K and corresponds to running a single generate (E-step) and improve (M-step) of ReST ${ }^{E M}$. While RFT demonstrated limited performance improvements on GSM8K with increasing language model capacity, ReST ${ }^{E M}$ achieves larger gains on more challenging APPS and MATH benchmarks when scaling PaLM 2 model capacity. Moreover, we observe that using multiple iterations of $\operatorname{ReST}^{E M}$ result in larger performance gains.
- Iterative Maximum Likelihood (IML) optimizes a policy using a reward-weighted log-likelihood objective on self-collected data. IML has been shown to perform well with relatively small-scale language models for semantic parsing (Agarwal et al., 2019; Liang et al., 2016), machine translation (Wu et al., 2016) and simple math reasoning (Ni et al., 2022). Each E-step and M-step in IML is performed over a mini-batch of training examples instead of the entire training dataset, as done in $\operatorname{ReST}{ }^{E M}$. In IML, the learned policy can significantly diverge from the initial pretrained model, which can manifest as task-specific overfitting, where the model performs well on the target task but loses its ability to generalize to other tasks or domains. Additionally, the tightly coupled nature of data collection and policy optimization in IML leads to high computational cost with large LMs, making it significantly more expensive than $\operatorname{ReST}^{E M}$.
- Reward weighted regression (RWR) (Peters and Schaal, 2007) corresponds to EM where we set $p(O=1 \mid \boldsymbol{x}, \boldsymbol{y}) \propto \exp (r(\boldsymbol{x}, \boldsymbol{y}))$ in Section 3. RWR has been previously applied to robotic control, as it can be easily applied to non-binary reward functions. Norouzi et al. (2016) build on RWR to propose a general variant of IML for machine translation.
- Reward ranked fine-tuning (RAFT) (Dong et al., 2023) can be interpreted as alternating between E-step and M-step over mini-batches, where E-step uses the the output sample with maximum reward for each input context. For binary reward functions, RAFT is analogous to IML and as such, can be viewed as an instantiation of $\operatorname{ReST}^{E M}$.

Other related works: TRICE (Phan et al., 2023) proposes an EM-based approach to maximize the marginal log-likelihood (MML) of generating a correct answer for a reasoning problem, where the chain-of-thought rationale is treated as a latent variable. While E-step in $\operatorname{ReST}^{E M}$ simply corresponds to sampling from the model and filtering with a binary reward, TRICE uses Markov-chain Monte Carlo with a control variate to approximate the MML gradient. Sordoni et al. (2023) propose a gradient-free EM-based approach, similar to RAFT, for prompt-optimization for frozen LLMs.

Inspired by an earlier version of this manuscript, Agarwal et al. (2024) investigated if modelgenerated data can outperform human data for few-shot and many-shot prompting. They found that this is indeed the case, especially for few-shot prompting.

|  | ReST $^{E M}$ | ReST | STaR | RFT |
| :--- | :---: | :---: | :---: | :---: |
| Starts from fine-tuned model | $X$ | $\checkmark$ | $x$ | $x$ |
| Finetunes from base model in each iteration | $\checkmark$ | $x$ | $\checkmark$ | N/A |
| Uses rationalizations for unsolved questions | $x$ | $x$ | $\checkmark$ | $x$ |
| Temperature sampling for exploration | $\checkmark$ | $\checkmark$ | $x$ | $\checkmark$ |
| Experiments with Large LMs | $\checkmark$ | $x$ | $x$ | $\checkmark$ |
| Multiple iterations | $\checkmark$ | $\checkmark$ | $\checkmark$ | $x$ |
| Larger gains on bigger models | $\checkmark$ | N/A | N/A | $x$ |
| Evaluation on held out tasks | $\checkmark$ | $x$ | $x$ | $x$ |

Table 1 | Differences between $\operatorname{ReST}^{E M}$ and other closely related approaches utilizing synthetic data for advancing language model capabilities.

## 5. Experiments and analysis

The goal of our experiments is to answer the following questions:

1. How effective is $\operatorname{ReST}^{E M}$ compared to fine-tuning on human-generated data?
2. How many iterations are needed for optimal performance? How quickly does ReST ${ }^{E M}$ leads to overfitting on training set?
3. How does ReST ${ }^{E M}$ affect pass@k and majority voting performance?
4. If we fine-tune using model-generated data on a specific task, do we see positive transfer to related tasks? Is there any performance degradation compared to the base model when evaluating our fine-tuned models on a broad suite of tasks?
5. How much input data do we need to get most of the performance gains from $\operatorname{ReST}^{E M}$ ? Is one iteration of ReST ${ }^{E M}$ sufficient?

Training Datasets. We evaluate ReST ${ }^{E M}$ primarily on mathematical problem solving using the Hendrycks' MATH dataset (Hendrycks et al., 2021b) and code generation using the APPS (Introductory) dataset (Hendrycks et al., 2021a). MATH and APPS (Introductory) contain 7500 and 2342 training problems respectively. We select these tasks because the model outputs can be automatically evaluated as correct or incorrect, perfectly suited for ReST ${ }^{E M}$. Both these datasets offer binary rewards: on MATH, model-generated answers can be easily verified for correctness using the ground-truth answer, while on APPS, test cases determine whether the generated code is correct.

Models. We use the PaLM 2 models (Google et al., 2023) with public APIs on Google Cloud for experiments, including PaLM 2-S (Bison), PaLM 2-S* (Codey), and PaLM 2-L (Unicorn).

Evaluation. We report generalization performance using the test splits of the MATH and APPS (Introductory) datasets. For measuring transfer performance, we look at GSM8K (Cobbe et al., 2021), Hungarian HS finals (Paster, 2023), and HumanEval (Chen et al., 2021) datasets. We also evaluate our models using the Big-Bench Hard (Suzgun et al., 2022) benchmark to evaluate general capabilities. All evaluations follow the settings from Google et al. (2023), unless specified otherwise.

Implementation Details. During each iteration of $\operatorname{ReST}^{E M}$, we generated a fixed number of solutions per problem for the E-step: 32 for the MATH dataset and 64 for the APPS dataset. For generating solutions, we sample from the language model using top-K sampling with $\mathrm{K}=40$ and temperature of 0.7. However, directly using all these model-generated solutions can lead to an imbalanced dataset, as we will have a lot more correct solutions for the easier problems. To mitigate this, we introduced a cut-off threshold for the maximum number of solutions per problem, a design

![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-08.jpg?height=597&width=1328&top_left_y=284&top_left_x=364)

Figure $2 \mid \operatorname{ReST}^{E M}$ for math problem-solving. Test performance on MATH and GSM8K (transfer) for PaLM 2-S* and PaLM 2-L as a function of ReST ${ }^{E M}$ iterations. We also report performance of models fine-tuned via SFT on human-generated data as a baseline. Iteration 0 corresponds to pre-trained model performance. Following Google et al. (2023), we use greedy decoding for evaluation.

![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-08.jpg?height=594&width=1313&top_left_y=1136&top_left_x=377)

Figure $3 \mid \operatorname{ReST}^{E M}$ for code-generation. Test performance on APPS (introductory) and HumanEval (transfer) for PaLM 2-S* and PaLM 2-L as a function of $\operatorname{ReST}^{E M}$ iterations.

choice also used by Zelikman et al. (2022), included in the fine-tuning dataset: 10 for both MATH and APPS. This approach ensures diversity in the training data and safeguards against overfitting on easier problems. For fine-tuning, we use the few-shot prompt (and the question) as input to the model, and use the model-generated solutions as targets. We only apply the next token prediction loss (Equation 1) on the targets.

## 5.1. $\operatorname{ReST}^{E M}$ on MATH and APPS

Figures 2 and 3 show the performance of $\operatorname{ReST}^{E M}$ when trained on the MATH and APPS datasets, respectively. We see that MATH benefits from performing multiple iterations of ReST ${ }^{E M}$, both in terms of performance on the MATH test set, as well as transfer to GSM8K. On the other hand, we see that most of the gains for APPS come from the first iteration, and more iterations lead to a regression on both APPS and HumanEval.

Interestingly, Figures 2 and 3 demonstrate that fine-tuning on model-generated solutions substan-
![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-09.jpg?height=564&width=1242&top_left_y=280&top_left_x=407)

Figure 4 | Train-test performance gap on (left) MATH with PaLM-2-L, and (right) APPS with PaLM2 -S*, as a function of $\operatorname{ReST}^{E M}$ iterations.

tially outperforms using human-written solutions, especially for the PaLM 2-L model. This aligns with findings of Yuan et al. (2023) and recent work on distilling LLMs using model-generated data (Agarwal et al., 2023; Gu et al., 2023). However, unlike Yuan et al. (2023), who observed diminishing returns from model-generated data on GSM8K when scaling model capacity, our results suggest an opposite trend: $\operatorname{ReST}{ }^{E M}$ leads to larger performance gains as model capacity increases. On the MATH dataset, the test accuracy improvement with $\operatorname{ReST}^{E M}$ is $5.94 \%$ for PaLM 2-S compared to $6.34 \%$ for the larger PaLM 2-L model. Similarly, on the APPS dataset, improvements are 5.6\% for PaLM 2-S* compared to $6.4 \%$ for PaLM 2-L. This is in addition to the fact that the larger models start with a much stronger initial performance, and improvements on these benchmarks generally get harder as the baseline performance goes up.

Train-test performance gap. Figure 4 shows that while training performance increases linearly with the number of $\operatorname{ReST}^{E M}$ iterations, test set performance does not. For MATH, test performance improvements are small after the first iteration, and for APPS, we observe a regression in performance in the $2^{\text {nd }}$ iteration. We suspect that the regression in performance is likely due to overfitting on the small set of training problems. Since the APPS dataset is about a third of the size of the MATH dataset, it suffers more from this problem.

### 5.2. Impact on Pass@K and Majority-Voting Performance

To investigate the impact of fine-tuning with $\operatorname{ReST}^{E M}$ on the diversity of the final model's generated outputs, we evaluate pass $@ \mathrm{k}$ (Chen et al., 2021) and majority voting (Wang et al., 2023) performance of the fine-tuned PaLM 2-L model relative to the base model.

Pass@K measures the probability that at least one of the $\mathrm{K}$ generated solutions for a problem is correct, that is, outputs the correct answer for math problems or passes all the unit tests for code generation. Figure 5 shows the performance of Palm-2-L on the pass@K metric. We see that model obtained after ReST ${ }^{E M}$ fine-tuning is stronger for all values of $\mathrm{K}$, with the performance gap typically being the highest for $\mathrm{K}=1$.

Majority voting first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. For Hendrycks MATH, it is possible to use majority voting to maximize Pass@1 performance, and we find that when using 64 samples per question, the PaLM 2-L fine-tuned with $\operatorname{ReST}^{E M}$ obtains a test accuracy of 48.82 , while the base model gets 44.02 .
![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-10.jpg?height=436&width=1568&top_left_y=284&top_left_x=244)

Figure 5 | Pass@K results for PaLM-2-L pretrained model as well as model fine-tuned with ReST ${ }^{E M}$. For a fixed number of samples $\mathrm{K}$, fine-tuning with $\mathrm{ReST}^{E M}$ substantially improves Pass@K performance. We set temperature to 1.0 and use nucleus sampling with $p=0.95$.

### 5.3. Ablation Studies

Impact of multiple iterations Our results show that multiple iterations can sometimes lead to over-fitting on the train set (Figure 4). This raises the question of whether multiple iterations are really necessary. Is it better to collect a larger dataset and perform just a single iteration of $\operatorname{ReST}^{E M}$ ? To investigate this, we collect a dataset with the base PaLM-2-L model on Hendrycks MATH that is $3 \times$ as many solutions per problem as used in a single iteration of $\mathrm{ReST}^{E M}$ for the E-step. Fine-tuning with this dataset results in pass@1 performance of $40.3 \%$, which is lower than the $41 \%$ in second and $41.9 \%$ in third iteration, as shown in Figure 2. These results indicate that performing multiple iterations of $\operatorname{ReST}^{E M}$ leads to higher performance compared a single iteration with $3 \mathrm{x}$ the data.

Comparing model-generated data with human data A key strength of $\operatorname{ReST}^{E M}$ is its ability to generate multiple correct solutions for each problem. This provides valuable additional training data compared to human-generated data, which typically offers only a single solution per problem. While this makes a comparison in Figures 2 and 3 not entirely fair, it also highlights the potential of ReST ${ }^{E M}$ to boost performance with diverse and correct solutions.

In order to enable an apples-to-apples comparison, we conduct the following study: we select all Hendrycks MATH questions for which we have at least one correct model-generated solution, resulting in about $5 \mathrm{~K}$ questions. For these $5 \mathrm{~K}$ questions, we run two fine-tuning experiments: $\mathrm{SFT}(5 \mathrm{~K}$ ) where we fine-tune on human-written solutions (one per question), and $\operatorname{ReST}^{*}(5 \mathrm{~K})$ where we fine-tune on model-generated solutions (also one per question, selected at random).

The results in Figure 6 (right), show that ReST ${ }^{E M}$ outperforms fine-tuning on human data even in this much more restricted setting. Furthermore, the efficacy of $\operatorname{ReST}(5 \mathrm{~K})$ over $\operatorname{ReST}^{*}(5 \mathrm{~K})$ highlights the additional gain in performance that we can obtain by spending more compute on sampling a large number of solutions and performing multiple iterations of $\operatorname{ReST}^{E M}$.

Distillation with $\operatorname{ReST}^{E M}$-generated data The above results indicate that self-generated data can be better than human data for fine-tuning language models. We hypothesize this may be because model-generated solutions are more in-distribution compared to human-written solutions. This raises the question of whether $\operatorname{ReST}^{E M}$-generated data can benefit different models than the one generating the data.

To answer this question, we consider a distillation setup on MATH where we fine-tune PaLM 2-S using data generated by PaLM 2-L, resulting in solutions for about $5 \mathrm{~K}$ questions. Specifically, we ran
![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-11.jpg?height=546&width=1654&top_left_y=281&top_left_x=201)

Figure $6 \mid$ Left. Comparing $\operatorname{ReST}^{E M}$ with SFT on MATH. SFT refers to fine-tuning on human data, while $\mathrm{ReST}^{*}$ refers to a version of $\mathrm{ReST}^{E M}$ with one iteration that uses only one correct sample per problem. Here, ReST denotes ReST ${ }^{E M}$ with 3 iterations. For each method, we denote the number of questions in parenthesis. Right. Impact of Model-Generated Data for Distillation.

two distillation experiments: Distill* (2-L) where we fine-tune on teacher-generated solutions (one per question), similar to ReST (5K), and Distill (2-L), which includes multiple solutions per problem, generated during the final iteration of $\operatorname{ReST}^{E M}$ with PaLM 2-L.

Our results, shown in Figure 6 (right), reveal that Distill* surpasses the performance achieved by fine-tuning on human-written solutions, despite having smaller number of training questions. Additionally, fine-tuning PaLM 2-S with multiple solutions from PaLM 2-L, namely Distill (2-L), is superior than using self-generated solutions via ReST ${ }^{E M}$. This improvement is likely due to the larger number of training questions with solutions in PaLM 2-L generated data compared to 2-S. Overall, these results indicate that model-generated data can be more effective for fine-tuning smaller models than relying on human-generated data.

$\operatorname{ReST} v$ seST ${ }^{E M}$ A major difference between $\operatorname{ReST}{ }^{E M}$ and ReST is that while ReST ${ }^{E M}$ always finetunes the base model for each iteration, ReST continues to finetune the model from the last iteration. We run an ablation comparing these options using PaLM 2-S* in Figure 7 and observe that while ReST and ReST ${ }^{E M}$ have similar performance on APPS, the transfer performance to HumanEval is substantially better with $\operatorname{ReST}^{E M}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-11.jpg?height=302&width=782&top_left_y=1685&top_left_x=1071)

...... Palm-2-S*-SFT

Figure 7| $\operatorname{ReST}^{E M}$ vs ReST using PaLM 2-S*.

Impact of dataset size Since one of the main ingredients needed for ReST ${ }^{E M}$ is a dataset of input contexts (e.g., questions for MATH), we are interested in evaluating the effect of number of input problems. The results from our dataset ablations using the PaLM-2-L model on Hendrycks MATH, Figure 8 (left), show that utilizing just 1000 MATH questions results in significant gains, implying that the method is very efficient in the number of prompts needed. However, we noted a slight decrease in performance when using 4,000 questions compared to 2,000 , indicating potential variance in the fine-tuning process. Ideally, conducting this experiment multiple times would help quantify this variance, but this is prohibitively resource-intensive. Overall, we find that $\operatorname{ReST}^{E M}$ is quite sample efficient and performance gains from $\operatorname{ReST}^{E M}$ improve as we increase the dataset size.
![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-12.jpg?height=500&width=1448&top_left_y=290&top_left_x=310)

Figure $8 \mid$ Left. Performance for a single iteration of ReST ${ }^{E M}$ as a function of dataset size (number of questions) on MATH. Right. Improvement from ReST ${ }^{E M}$ based on the difficulty level of the question.

Which Questions Benefit Most from ReST ${ }^{E M} \quad$ We evaluate the performance enhancement of ReST ${ }^{E M}$ across different question difficulties in the Hendrycks MATH dataset. Questions are classified based on success rates from the base model at a temperature setting of $\mathrm{T}=1.0$ into four categories: "easy" (answered correctly $75 \%-100 \%$ of the time), "medium" (50\%-75\%), "hard" (25\%-50\%), and "very hard" (below $25 \%$ ). Figure 8 (right) presents the average success rates for these categories, comparing the base model to the $\operatorname{ReST}{ }^{E M}$-finetuned model. The results demonstrate that $\operatorname{ReST}^{E M}$ consistently improves performance across all difficulties, with the highest gains coming for questions categorized as medium and hard.

### 5.4. Impact on Reasoning capabilities

![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-12.jpg?height=802&width=1542&top_left_y=1529&top_left_x=266)

Figure 9 | Comparing the $\operatorname{ReST}^{E M}$ models to the base model on the Big-Bench Hard suite of tasks. Evaluations were conducted across multiple checkpoints, and the vertical black lines denote standard deviation.

General capabilities. BIG-Bench provides a suite of over 200 tasks that can be used to probe LLMs' performance across a range of fields and capabilities. BIG-Bench Hard (BBH) (Suzgun et al.,

2022) is a subset of 23 BIG-Bench tasks where the previous generation of LLMs, such as Codex and PaLM 540B, performed below the average human rater. We follow the protocol of Google et al. (2023) and evaluate on BBH using both few-shot and chain-of-thought prompting. Figure 9 shows the performance of ReST ${ }^{E M}$-finetuned models, and compares them against the base PaLM-2 model. We see no major degradation on any of the BBH tasks. Furthermore, the model fine-tuned on Hendrycks MATH outperforms the base model on this suite when using chain-of-thought prompting, and the model fine-tuned on APPS also shows slight performance gains. When using direct prompting, all three models perform similarly.

Problem-solving. To stress test the math problem-solving capabilities on a held-out "real-world" evaluation set, we evaluate our model on the 2023 Hungarian high school finals exam in mathematics, following the evaluation protocol from Paster (2023). Specifically, we evaluate the PaLM 2-L model, fine-tuned with $\operatorname{ReST}{ }^{E M}$ on Hendrycks MATH, using the 1-shot prompt from Grok, sample solutions using temperature 0.1 , and manually grade the outputs using the rubric provided by the examiners. The results from evaluation are shown in Figure 10. We find that PaLM-2-L fine-tuned with ReST ${ }^{E M}$ performs well on this exam, surpassing the performance of all existing models except GPT-4.

![](https://cdn.mathpix.com/cropped/2024_06_04_379d925a7eca23c9c6deg-13.jpg?height=631&width=1110&top_left_y=1061&top_left_x=473)

Figure 10 | Transfer results on Hungarian HS Finals Exam. Results for models other than PaLM-2-L finetuned with $\operatorname{ReST}^{E M}$ are taken from Paster (2023). Several models specialized for mathematics perform well on the widely-used GSM8K benchmark but perform poorly on the Hungarian exam. In contrast, PaLM 2-L model fine-tuned with $\operatorname{ReST}^{E M}$ performs well on both these benchmarks.

## 6. Discussion

In this paper, we propose training on model-generated data combined with a reward function, via $\operatorname{ReST}{ }^{E M}$, for improving the performance of LLMs on problem-solving tasks. Furthermore, we demonstrate that ReST ${ }^{E M}$ is theoretically grounded in the application of expectation-maximization to RL. We evaluate $\operatorname{ReST}^{E M}$ on mathematical problem solving and code generation, and show that $\operatorname{ReST}{ }^{E M}$ offers significant performance gains at a relatively low computational cost, especially when compared to the cost of pre-training. Our experiments also show that $\operatorname{ReST}^{E M}$ does not lead to regression on other tasks. We conduct a number of ablations to better understand the strengths and weaknesses of this method, and find that it is data-efficient, but also requires some vigilance to avoid over-fitting.

There are a number of limitations associated with $\operatorname{ReST}^{E M}$. First, this method requires a moderatelysized training set of problems or prompts, which would need to be collected (from humans) for any new task of interest. Second, ReST ${ }^{E M}$ also requires access to a manually-designed or learned reward
function, ideally one that can be computed automatically. Finally, while $\operatorname{ReST}^{E M}$ allows significant performance improvements in pass@1 performance, it may not quite close the gap to pass@K performance for the same task (with a sufficiently large K). Future research in self-improvement in language models should focus on automating manual parts of the pipeline (likely through language models as well), and explore algorithmic improvements that reduce the gap to pass@K performance.

## Acknowledgements

We would like to thank Tom Le Paine for providing feedback to an early draft. We also acknowledge Benjamin Anderson, Sridhar Thiagarajan, Feryal Behbahani, Aleksandra Faust, Doina Precup, Olivier Bachem, and Slav Petrov for helpful discussions.

## Author Contributions

Avi, Rishabh, and JD jointly led the project. Avi was responsible for training and evaluation infrastructure, ablations and experiments on MATH, JD led the experiments on APPS, Rishabh was responsible for the paper writing, evaluations, and distillation ablations.

Ankesh, Piyush, Ethan, and Behnam observed preliminary findings about efficacy of modelgenerated data on MATH for Minerva models and motivated this research. Piyush also helped Avi in setting up infrastructure. Xavier, Peter, James, Jaeheoon, Kelvin and Yamini took part in project discussions. Jascha and Noah sponsored and advised the project. All other authors provided feedback on this work.

## References

R. Agarwal, C. Liang, D. Schuurmans, and M. Norouzi. Learning to generalize from sparse and underspecified rewards. In International conference on machine learning, pages 130-140. PMLR, 2019 .

R. Agarwal, N. Vieillard, P. Stanczyk, S. Ramos, M. Geist, and O. Bachem. Gkd: Generalized knowledge distillation for auto-regressive sequence models. arXiv preprint arXiv:2306.13649, 2023.

R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, S. Chan, A. Anand, Z. Abbas, A. Nova, J. D. Co-Reyes, E. Chu, F. Behbahani, A. Faust, and H. Larochelle. Many-shot in-context learning, 2024.

T. Anthony, Z. Tian, and D. Barber. Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems, 30, 2017.

S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. M. Lundberg, H. Nori, H. Palangi, M. T. Ribeiro, and Y. Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4. CoRR, abs/2303.12712, 2023. doi: 10.48550/ARXIV.2303.12712. URL https://doi.org/10.48550/arXiv.2303.12712.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer,

P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

P. Dayan and G. E. Hinton. Using expectation-maximization for reinforcement learning. Neural Computation, 9(2):271-278, 1997.

A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B (methodological), 39(1):1-22, 1977.

H. Dong, W. Xiong, D. Goyal, R. Pan, S. Diao, J. Zhang, K. Shum, and T. Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.

Google, R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Y. Gu, L. Dong, F. Wei, and M. Huang. Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543, 2023.

C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, A. Sharma, A. Siddhant, A. Ahern, M. Wang, C. Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns, S. Puranik, H. He, D. Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021a.

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021b.

J. Huang, S. S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han. Large language models can self-improve. CoRR, abs/2210.11610, 2022. doi: 10.48550/ARXIV.2210.11610. URL https://doi.org/10 . 48550/arXiv. 2210.11610.

C. Liang, J. Berant, Q. Le, K. D. Forbus, and N. Lao. Neural symbolic machines: Learning semantic parsers on freebase with weak supervision. arXiv preprint arXiv:1611.00020, 2016.

A. Ni, J. P. Inala, C. Wang, A. Polozov, C. Meek, D. Radev, and J. Gao. Learning math reasoning from self-sampled correct and partially-correct solutions. In The Eleventh International Conference on Learning Representations, 2022.

M. Norouzi, S. Bengio, N. Jaitly, M. Schuster, Y. Wu, D. Schuurmans, et al. Reward augmented maximum likelihood for neural structured prediction. Advances In Neural Information Processing Systems, 29, 2016.

OpenAI. Gpt-4 technical report, 2023.

K. Paster. Testing language models on a held-out high school national finals exam. https:// huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam, 2023.

J. Peters and S. Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pages 745-750, 2007.

D. Phan, M. D. Hoffman, D. Dohan, S. Douglas, T. A. Le, A. Parisi, P. Sountsov, C. Sutton, S. Vikram, and R. A. Saurous. Training chain-of-thought via latent-variable inference. arXiv preprint arXiv:2312.02179, 2023.

A. Sordoni, X. Yuan, M.-A. Côté, M. Pereira, A. Trischler, Z. Xiao, A. Hosseini, F. Niedtner, and N. Le Roux. Joint prompt optimization of stacked llms using variational inference. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=1PL1NIMMrw.

Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476-15488, 2022.

</end of paper 4>


