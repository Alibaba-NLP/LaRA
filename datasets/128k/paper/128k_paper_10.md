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
# Enhancing Large Vision Language Models with Self-Training on Image Comprehension 

Yihe Deng ${ }^{* 1}$, Pan $\mathbf{L u}^{* 1,3}$, Fan Yin ${ }^{1}$, Ziniu Hu ${ }^{1}$, Sheng Shen ${ }^{2}$<br>James Zou ${ }^{3}$, Kai-Wei Chang ${ }^{1}$, Wei Wang ${ }^{1}$<br>${ }^{1}$ University of California, Los Angeles<br>${ }^{2}$ University of California, Berkeley ${ }^{3}$ Stanford University<br>https://stic-lvlm.github.io/

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-01.jpg?height=502&width=832&top_left_y=936&top_left_x=359)

Query: How many gallons of supreme gasoline can I get with $\$ 50$ ?

Base (LLaVA-v1.6 7B):

Based on the current gas prices displayed on the sign, you can get approximately 3.65 gallons of supreme gasoline with $\$ 50$.

## STIC (LLaVA-v1.6 7B):

With $\$ 50$, you can get approximately 13.69 gallons of supreme gasoline, as indicated by the price of $\$ 3.65$ per gallon on the sign.

Figure 1: Left: Accuracy improvement of our method, STIC, compared to the original LLaVA-v1.6 (Liu et al., 2024) on seven benchmarks. Right: Response examples from the original LLaVA-v1.6 and STIC (LLaVA-v1.6), which enhances image comprehension and subsequent reasoning capabilities.


#### Abstract

Large vision language models (LVLMs) integrate large language models (LLMs) with pre-trained vision encoders, thereby activating the perception capability of the model to understand image inputs for different queries and conduct subsequent reasoning. Improving this capability requires high-quality vision-language data, which is costly and labor-intensive to acquire. Self-training approaches have been effective in single-modal settings to alleviate the need for labeled data by leveraging model's own generation. However, effective self-training remains a challenge regarding the unique visual perception and reasoning capability of LVLMs. To address this, we introduce Self-Training on Image Comprehension (STIC), which emphasizes a self-training approach specifically for image comprehension. First, the model self-constructs a preference dataset for image descriptions using unlabeled images. Preferred responses are generated through a step-by-step prompt, while dis-preferred responses are generated from either corrupted images or misleading prompts. To further self-improve reasoning on the extracted visual information, we let the model reuse a small portion of existing instruction-tuning data and append its self-generated image descriptions to the prompts. We validate the effectiveness of STIC across seven different benchmarks, demonstrating substantial performance gains of $4.0 \%$ on average while using $70 \%$ less supervised fine-tuning data than the current method. Further studies investigate various components of STIC and highlight its potential to leverage vast quantities of unlabeled images for self-training. Code and data are made publicly available.


[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-02.jpg?height=563&width=1379&top_left_y=239&top_left_x=362)

Figure 2: Framework overview of STIC, a two-stage self-training algorithm focusing on the image comprehension capability of the LVLMs. In Stage 1, the base LVLM self-constructs its preference dataset for image description using well-designed prompts, poorly-designed prompts, and distorted images. In Stage 2, a small portion of the previously used SFT data is recycled and infused with model-generated image descriptions to further fine-tune the base LVLM.

## 1 Introduction

In recent years, we have witnessed remarkable advancements in large language models (LLMs), such as GPT-4 (OpenAI, 2023a) and the LLaMA family (Touvron et al., 2023a,b). The increasing importance of processing multimodal inputs, including images and text, has significantly driven progress in vision language models (Radford et al., 2021; Jia et al., 2021b; Goel et al., 2022). Leveraging the powerful language understanding and generation capabilities of LLMs, researchers have advanced vision language models into large vision language models (LVLMs). This enhancement is achieved by integrating LLMs with image encoders (Radford et al., 2021; Li et al., 2023a), which were pre-trained on large-scale image-text pairs to ensure alignment between the two domains. For instance, LLaVA (Liu et al., 2023b) integrates a vision encoder from CLIP (Radford et al., 2021) with the LLM Vicuna (Chiang et al., 2023b), which is further fine-tuned on carefully constructed vision-language instructional datasets to activate the model's perception capability of capturing the vision information according to different queries. This recent development has substantially expanded the requirement for large-scale instruction fine-tuning data for LVLMs (Gao et al., 2023b; Bai et al., 2023; Chen et al., 2023b; Gao et al., 2024; Anthropic, 2024; McKinzie et al., 2024).

While LVLMs have shown promising results, a key challenge lies in the acquisition of high-quality fine-tuning data. Obtaining human-curated content at scale is often prohibitively expensive, especially for multi-modal data. Many recent studies resort to GPT-4V (OpenAI, 2023b) for generating or labeling high-quality vision-language fine-tuning data. However, this approach does not significantly reduce the cost (Liu et al., 2023b; Wu et al., 2024). For instance, using GPT-4V to generate $6 k$ image descriptions with $1 k$ tokens per output would cost approximately $\$ 200$. There remains a pressing need for cost-effective methods to gather fine-tuning data to further enhance LVLMs.

To tackle the data acquisition bottleneck in multi-modality, we propose Self-Training on Image Comprehension (STIC). Our method is inspired by the recent success of self-training (Chen et al., 2024; Yuan et al., 2024; Fränken et al., 2024; Rosset et al., 2024) in LLMs, which leverages selfgenerated data to improve their downstream performance. However, different from the text-only domain, the unique vision modality of LVLMs introduces new challenges, as LVLMs must understand the input image content before reasoning and responding to any related textual queries about the image. Therefore, the proposed STIC approach is a novel two-stage self-training method that targets both image perception and reasoning over images and texts.

The overall framework is summarized in Figure 2. STIC specifically emphasizes the image comprehension self-training of LVLMs where the model generates its own preference dataset focused on image description. The self-generated dispreferred response is obtained by gathering model responses from either (1) prompts likely to elicit inaccurate responses or (2) corrupted images. The preferred responses are collected via a detailed prompt that guides the model through a step-by-step image description process. Figure 3 shows examples of such generated responses. During fine-tuning, we consider a DPO loss (Rafailov et al., 2023) with an additional regularized term explicitly emphasizing

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-03.jpg?height=288&width=266&top_left_y=249&top_left_x=366)

Task: Explain what is depicted in the photograph.

Preferred data (Good prompting: Provide a detailed description of the image, focusing on ...) The image depicts a dynamic winter sports scene. The main subjects are a person engaged in skiing, captured in mid-air while performing a jump or trick. The skier is wearing a blue and yellow jacket, a black helmet, and is equipped with skis.....

(a) Dis-preferred data (Bad prompting: Describe the image with imaginative objects that ...) ...... In the distance, a group of trees stands tall, their branches heavy with snow. ...... Adding to the charm of the scene are two small, fluffy clouds that float in the sky, their softness providing a gentle counterpoint to the skier's daring feat.

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-03.jpg?height=298&width=268&top_left_y=610&top_left_x=365)

(b) Dis-preferred data (Normal prompting + Image corruption: Color jitter.)

The skier, clad in a vibrant purple jacket and blue pants, ..... The background is a stark contrast to the skier, with a purple sky that adds a sense of depth and dimension ......

(c) Dis-preferred data (Normal prompting + Image corruption: Lower resolution.)

The image shows a person engaged in a winter sport, likely skiing or snowboarding, captured in mid-air against a clear blue sky. The individual is wearing a blue and yellow suit, ..... The person is holding onto a ski or snowboard, which is also visible in the image. The motion blur effect ......

Figure 3: Examples of the self-constructed preference data in STIC.

the preferred response. Lastly, we allow the model to self-improve its reasoning ability based on its own extracted image information by reusing a small amount of existing instruction fine-tuning data and appending its self-generated image descriptions to the prompts. We refer to this second stage as description-infused fine-tuning. Notably, our STIC approach does not require pre-labeled information of the images, which contrasts to the recent works that rely on such information for constructing vision-language preference data (Zhou et al., 2024).

To demonstrate the effectiveness of STIC, we conduct extensive experiments on seven vision-language benchmarks, including ScienceQA (Lu et al., 2022), TextVQA (Singh et al., 2019), ChartQA (Masry et al., 2022), LLaVA-Bench (Liu et al., 2023a), MMBench (Liu et al., 2023c), MM-Vet (Yu et al., 2023), and MathVista (Lu et al., 2024). These benchmarks encompass scientific reasoning, math reasoning, optical character recognition (OCR), and conversation capabilities based on vision inputs, spanning various image sources such as natural, chart, and text-rich images. We employ LLaVAv1.6 (Liu et al., 2024) as the primary base LVLM for our experiments and unitize $6 k$ images from MSCOCO (Lin et al., 2014) to construct the image description preference data. As depicted in Figure 1, STIC achieves consistent and significant performance improvements across these benchmarks, with an average accuracy gain of $\mathbf{4 . 0 \%}$ over the base LVLM and a notable gain of $\mathbf{6 . 4 \%}$ on ScienceQA. We also provide an example of the different responses from the original LVLM and STIC in Figure 1, where STIC successfully identifies the key visual information and accurately reason with it. These results demonstrate the remarkable effectiveness of our image comprehension self-training approach in enhancing the visual perception capabilities of LVLMs.

In addition, we explore the benefits of the various components of STIC. First, based on the descriptioninfused fine-tuning stage that enhances the model's reasoning ability with self-generated description, we show that further letting the model describe the image before responding to a query provides further improved reasoning capability. This results in a notable improvement of $2.8 \%$ on ScienceQA and $1.1 \%$ on average as compared to direct responses to queries (Table 2). Moreover, we examine the impact of self-generated dispreferred responses, from either bad prompting or image corruption. By excluding these dispreferred responses and conducting SFT solely with preferred responses, we observed a performance decrease of $2.5 \%$ on average across three benchmarks as compared to STIC with the preference data (Table 3). This highlights the importance of the negative samples in the self-constructed preference data by STIC. We also assess the scalability of our self-training scheme. By increasing the amount of generated preference data from $6 k$ to $12 k$, we show an even further improvement of STIC from $1.9 \%$ to $3.1 \%$ on LLaVA-Bench (Figure 7). This result suggests that STIC holds considerable potential for leveraging vast quantities of unlabeled images for self-training, given the immense availability of unlabeled image data. Lastly, our t-SNE visualization analysis shows that the closer the distribution between MSCOCO images, which we use for preference data construction, to images in downstream tasks, the more likely STIC results in higher performance gains (Figure 8).

The main contributions of this work are summarized as follows:

- We propose STIC, a novel two-stage self-training approach for LVLMs that focuses on enhancing their image comprehension capabilities by generating a preference dataset for image description without relying on pre-labeled image information.
- Through extensive experiments on seven diverse benchmarks, STIC demonstrates significant performance gains over the base LVLM, achieving an average accuracy gain of $4.0 \%$.
- We explore the benefits of various components of STIC, highlighting its potential to leverage vast quantities of unlabeled images for self-training.


## 2 Related Work

Vision language models (VLMs). VLMs (Tan and Bansal, 2019; Li et al., 2019, 2020; Kim et al., 2021; Wang et al., 2022b; Bao et al., 2022; Wang et al., 2022a; Alayrac et al., 2022; Li et al., 2023b; Chen et al., 2022; Jia et al., 2021a; Shen et al., 2022; Singh et al., 2021), processing both images and text, are pivotal in a wide range of multimodal understanding and reasoning tasks, capable of generating text or encoding multimodal representations. These models have shown increasing proficiency in visual perception and textual reasoning, and are also capable of following complex instructions (OpenAI, 2023b; Team et al., 2023). Recent advancements in the field have been propelled by the availability of open-source large language models (LLMs) (Touvron et al., 2023a,b; Jiang et al., 2023) and innovative image encoders (Radford et al., 2021; Li et al., 2022). For instance, LLaVA (Liu et al., 2023b) combines a vision encoder from CLIP (Radford et al., 2021) with the Vicuna LLM (Chiang et al., 2023b), and has been further fine-tuned on vision-language instructionfollowing datasets. The recent development of LVLMs has significantly expanded the scale and diversity of VL instruction-following data, including models such as LLaMA-Adapter-V2 (Gao et al., 2023b), Qwen-VL (Bai et al., 2023), InternVL (Chen et al., 2023b), InstructBLIP (Dai et al., 2024), SPHINX-X (Gao et al., 2024), Claude-3 (Anthropic, 2024), MM1 (McKinzie et al., 2024), and Grok-1.5V (xAI, 2024). In this work, we focus on enhancing the visual perception and mathmatical reasoning capabilities of LVLMs by efficiently aligning them with purely unsupervised data.

Alignment fine-tuning. Subsequent to supervised fine-tuning (SFT), alignment fine-tuning has emerged as a prominent approach to further enhance the performance of LLMs by aligning them with human preferences (Ouyang et al., 2022; Casper et al., 2023). Early efforts utilized on-policy reinforcement learning (RL) methods, such as proximal policy optimization (PPO) (Schulman et al., 2017), to train a reward model based on preference data (Bai et al., 2022; Touvron et al., 2023a). With the notable introduction of direct policy optimization (DPO) (Rafailov et al., 2023), a new line of research emphasizes direct learning from human preferences without relying on an explicit reward model (Zhao et al., 2023; Azar et al., 2024; Ethayarajh et al., 2024; Zheng et al., 2024). Another prominent direction is iterative preference fine-tuning, which has proven effective in enhancing model performance by repeatedly optimizing on newly generated preference pairs in each iteration (Adolphs et al., 2023; Xu et al., 2023; Xiong et al., 2023; Pang et al., 2024). While substantial research has focused on alignment fine-tuning for LLMs, efforts to adapt these techniques for LVLMs have been significantly limited. Initial attempts involve constructing preference datasets using human-labeled data (Sun et al., 2023) or GPT-4 generations for fine-tuning with a DPO loss (Zhou et al., 2024).

Self-improving language models. High-quality data, including human-crafted and advanced AI generated content, has been demonstrated to significantly enhance the performance of LLMs on various tasks (Josifoski et al., 2023; Taori et al., 2023; Chiang et al., 2023a; Li et al., 2023c). Although, acquiring such high-quality data is often prohibitively expensive. To circumvent the costs associated with obtaining human-annotated or expertly curated data, researchers have shifted their focus to leveraging data generated by the target model itself, exploring ways of self-improvement (Chen et al., 2024; Yuan et al., 2024; Fränken et al., 2024; Rosset et al., 2024). Recent studies have also emphasized the rephrasing capabilities of LLMs, which either enhance their own response quality (Deng et al., 2023; Prasad et al., 2023) or augment synthetic data for self-supervised fine-tuning (Kim et al., 2023). To the best of our knowledge, our work is the first to explore the potential for self-improvement in LVLMs, specifically focusing on the vision modality and emphasizing the self-improvement of image comprehension capabilities.

## 3 Problem Setting and Preliminaries

Notation. We use lower case letters and lower case bold face letters to denote scalars and vectors. We use the symbol $p$ to represent the probability of an LLM's response. And we denote the sequence of tokens generated from the LLM before the $t$-th token as $\mathbf{y}_{<t}=\left[y_{1}, \ldots, y_{t-1}\right]$ for $t>1$.

Generative vision language models. LVLM typically consists of three components: a vision encoder $f(\cdot)$, a projection network $g(\cdot)$, and an LLM $p_{\boldsymbol{\theta}}$ parameterized by $\boldsymbol{\theta}$. The model processes an image input $\mathbf{e}$ along with a text sequence $\mathbf{x}=\left[x_{1}, \ldots, x_{n}\right]$ as the prompt to generate a corresponding response $\mathbf{y}=\left[y_{1}, \ldots, y_{m}\right]$, where $x_{i}$ and $y_{j}$ represent individual tokens from the vocabulary of the LLM. The image is therefore converted into visual tokens within the language token space by the vision encoder and the projection network, producing $\mathbf{v}=\left[v_{1}, \ldots, v_{k}\right]=f \circ g(\mathbf{e})$. The response $\mathbf{y}$ is then considered as a sample from the conditional probability distribution $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{v}, \mathbf{x})$. As a Markov process, the conditional probability distribution $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{v}, \mathbf{x})$ can be decomposed as

$$
\begin{equation*}
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{v}, \mathbf{x})=\prod_{j=1}^{m} p_{\boldsymbol{\theta}}\left(y_{j} \mid \mathbf{v}, \mathbf{x}, \mathbf{y}_{<j}\right) \tag{3.1}
\end{equation*}
$$

Alignment fine-tuning. To improve LLM alignment with human preferences, RL fine-tuning (Bai et al., 2022; Gao et al., 2023a) is typically employed after supervised fine-tuning (SFT). This process involves a reward function $r(\mathbf{x}, \mathbf{y})$ for a given sequence pair $(\mathbf{x}, \mathbf{y})$. The more preferred response $\mathbf{y}$ is expected to result in a higher reward $r(\mathbf{x}, \mathbf{y})$, where the corresponding objective is to maximize the following:

$$
\begin{equation*}
L(\boldsymbol{\theta})=\mathbb{E}_{\mathbf{x} \sim \mathcal{D}, \mathbf{y} \sim p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})}[r(\mathbf{x}, \mathbf{y})]-\lambda \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathrm{KL}\left(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}) \| p_{\mathrm{ref}}(\cdot \mid \mathbf{x})\right) \tag{3.2}
\end{equation*}
$$

where $\mathbf{x} \sim \mathcal{D}$ is sampled from a given distribution $\mathcal{D}$ and the $\mathrm{KL}$ regularization term prevents the new model $p_{\theta}$ from deviating too much from the reference model $p_{\text {ref }}$, with $\lambda>0$ as the regularization parameter. Training the reward function is challenging in practice, but direct preference optimization (DPO) (Rafailov et al., 2023) simplifies this process using a predefined preference dataset $S_{\text {pref }}=\left\{\left(\mathbf{x}^{(i)}, \mathbf{y}_{w}^{(i)}, \mathbf{y}_{l}^{(i)}\right)\right\}_{i \in[N]}$, where $\mathbf{y}_{w}^{(i)}$ denotes the preferred response and $\mathbf{y}_{l}^{(i)}$ denotes the dispreferred response given the same prompt $\mathbf{x}^{(i)}$. The objective function is then formulated as

$$
\begin{equation*}
L_{\mathrm{DPO}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{\mathrm{ref}}\right)=\mathbb{E}_{\left(\mathbf{x}, \mathbf{y}_{w}, \mathbf{y}_{l}\right) \sim S_{\mathrm{pref}}}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{w} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{\mathrm{ref}}}\left(\mathbf{y}_{w} \mid \mathbf{x}\right)}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{l} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{\mathrm{ref}}}\left(\mathbf{y}_{l} \mid \mathbf{x}\right)}\right)\right] \tag{3.3}
\end{equation*}
$$

where $\ell(t)=\log (1+\exp (-t))$ is the logistic loss function and $\boldsymbol{\theta}_{\text {ref }}$ is the reference model.

Self-play fine-tuning. Notably, SPIN (Chen et al., 2024) shares a similar objective formulation while eliminating the need for a preference dataset by considering the model's own generation as a "dispreferred" response with an iterative self-play mechanism:

$$
\begin{equation*}
L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{\mathrm{ref}}\right)=\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim S_{\mathrm{SFT}}, \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \tag{3.4}
\end{equation*}
$$

given a SFT dataset $S_{\mathrm{SFT}}=\{(\mathbf{x}, \mathbf{y})\}_{i=1}^{n}$. Here, $p_{\boldsymbol{\theta}_{t}}$ represents the LLM in SPIN's previous iteration.

## 4 Our Method: STIC

In this section, we introduce STIC, a two-stage self-training algorithm designed to enhance image comprehension capabilities. The first stage constructs its own preference dataset and the second stage infuses the used SFT data with self-generated image descriptions for fine-tuning. Figure 2 presents the general framework of STIC. Notably, unlike recent work on fine-tuning algorithms (Sun et al., 2023; Zhou et al., 2024), STIC enables a base LVLM, such as LLaVA-v1.6 (Liu et al., 2024), to evolve from self-generated image captions, thus eliminating the need for additional supervised and preference data from human annotators or advanced teacher models. This approach fundamentally enhances image comprehension abilities and can be seamlessly applied to a wide range of vision-language reasoning tasks. We summarize STIC in Algorithms 1 and 2, and detail the process below.

Stage 1: Image comprehension self-training. The process begins with a self-constructed preference dataset from the base LVLM, which we aim to improve through fine-tuning. The dataset contains paired preference data for image descriptions:

- Preferred response: Model-generated image descriptions derived from well-crafted prompts with explicit reasoning steps.
- Dispreferred response: Model-generated descriptions resulting from either (1) corrupted image with low resolution or distorted color, or (2) "bad" prompts that cause the base model to hallucinate and describe elements that may not logically exist in the image.

```
Algorithm 1 STIC (Stage 1: image comprehension self-training)
    Input: Unlabeled image dataset: $\left\{\mathbf{v}^{(i)}\right\}_{i \in[N]}$. Image captioning prompt set: $P=\left\{\mathbf{x}^{(i)}\right\}_{i \in\left[M_{1}\right]}$.
    Hallucination prompt set: $P_{\text {hallu }}=\left\{\mathbf{x}_{\text {hallu }}^{(i)}\right\}_{i \in\left[M_{2}\right]}$. Image corruption $h(\cdot)$. Well-curated caption-
    ing prompt: $\mathbf{x}_{g}$. LVLM parameterized by $\boldsymbol{\theta}_{0}: p_{\boldsymbol{\theta}_{0}}$.
    Let self-training dataset $D=\{\}$.
    for $i=1, \ldots N$ do
        Randomly sample a number $n \in(0,1)$.
        Randomly sample $\mathbf{x} \sim\left\{\mathbf{x}^{(i)}\right\}_{i \in[M]}$.
        Generate preferred response $\mathbf{y}_{g} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{v}^{(i)}, \mathbf{x}_{g}\right)$.
        if $n<0.5$ then
            Randomly sample bad prompt $\mathbf{x}_{b} \sim P_{\text {hallu }}$.
            Generate dispreferred response $\mathbf{y}_{b} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{v}^{(i)}, \mathbf{x}_{b}\right)$.
        else
            Corrupt the image input $\mathbf{v}_{b}^{(i)}=h\left(\mathbf{v}^{(i)}\right)$.
            Generate dispreferred response $\mathbf{y}_{b} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{v}_{b}^{(i)}, \mathbf{x}^{(i)}\right)$.
        end if
        Add $\left(\mathbf{x}, \mathbf{y}_{g}, \mathbf{y}_{b}\right)$ to $D$.
    end for
    Update $\boldsymbol{\theta}_{1}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} \sum_{\left(\mathbf{x}, \mathbf{y}_{g}, \mathbf{y}_{b}\right) \in D}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{g} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{0}}\left(\mathbf{y}_{g} \mid \mathbf{x}\right)}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{b} \mid \mathbf{x}\right)}{p_{\theta_{0}}\left(\mathbf{y}_{b} \mid \mathbf{x}\right)}\right)-\alpha \log p_{\boldsymbol{\theta}}\left(\mathbf{y}_{g} \mid \mathbf{x}\right)\right]$.
```

Output: $\theta_{1}$.

The self-constructed preference dataset is used for the first-stage self-training using DPO (Rafailov et al., 2023) with an additional regularization term to further emphasize the preferred response, controlled by the hyperparameter $\alpha$. The regularized loss function is as follows:

$$
\begin{equation*}
L\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{\mathrm{ref}}\right)=\mathbb{E}_{\left(\mathbf{x}, \mathbf{y}_{w}, \mathbf{y}_{l}\right) \sim S}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{w} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{\mathrm{ref}}}\left(\mathbf{y}_{w} \mid \mathbf{x}\right)}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{l} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{\mathrm{ref}}}\left(\mathbf{y}_{l} \mid \mathbf{x}\right)}\right)-\alpha \log p_{\boldsymbol{\theta}}\left(\mathbf{y}_{w} \mid \mathbf{x}\right)\right] \tag{4.1}
\end{equation*}
$$

The use of an explicit loss term for positive examples can be similarly found in previous studies on contrastive learning (Chen et al., 2021; Chen and He, 2021; Chen et al., 2023a) and more recently in preference fine-tuning (Pang et al., 2024). Specifically, Chen et al. (2023a) demonstrated in the context of contrastive learning that a regularization term applied to positive samples provably enhances the model's ability to differentiate between positive and negative samples. As demonstrated in our experiments in Section 6, the LVLM after Stage 1 has shown notable improvement in downstream vision-language reasoning tasks, confirming that the enhanced visual comprehension ability directly benefits the model performance and its multimodal reasoning ability.

Stage 2: Description-infused fine-tuning. In the second stage, we further fine-tune the self-trained LVLM to leverage self-generated high-quality image descriptions for instruction-following tasks, and thus help ground its reasoning ability on self-generated descriptions. To achieve this, we randomly select a small subset of data from the model's instruction fine-tuning dataset already used during SFT. We then infuse the instructions in this subset with model-generated image descriptions as follows:

```
Image description: {model description}
<original instruction>
```

The original ground-truth completions remain unchanged. We then fine-tune the LVLM for one epoch on this small description-infused subset. This fine-tuning step ensures that the model effectively integrates visual information into its responses, thereby enhancing its ability to handle a variety of vision-language reasoning tasks. During inference, optionally, we can let the model self-augment its prompt for downstream vision-language reasoning tasks by describing the image before answering the question.

## 5 Experiments

In this section, we present the experiment results of STIC across seven visual question answering (VQA) benchmarks. We demonstrate that STIC effectively and substantially improves LVLM's performance across different VQA tasks using a self-constructed preference dataset without external labels.

```
Algorithm 2 STIC (Stage 2: description-infused fine-tuning)
    Input: Instruction-following dataset already used for fine-tuning the target LVLM model:
    $\left\{\mathbf{v}^{(i)}, \mathbf{x}^{(i)}, \mathbf{y}^{(i)}\right\}_{i \in[m]}$. Image description prompt set: $P=\left\{\mathbf{x}_{\text {des }}^{(i)}\right\}_{i \in\left[M_{1}\right]}$. LVLM parameterized by
    $\boldsymbol{\theta}_{1}$ after self-training: $p_{\boldsymbol{\theta}_{1}}$.
    Let description-infused dataset $D_{\text {des }}=\{\}$.
    for $i=1, \ldots m$ do
        Randomly sample $\mathbf{x}_{\text {des }} \sim\left\{\mathbf{x}_{\text {des }}^{(i)}\right\}_{i \in[M]}$.
        Generate model image description $\mathbf{y}_{\text {des }} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{v}^{(i)}, \mathbf{x}_{\text {des }}\right)$.
        Add $\left(\left[\mathbf{x}_{\mathrm{des}}, \mathbf{x}^{(i)}\right], \mathbf{y}^{(i)}\right)$ to $D_{\text {des }}$.
    end for
    Update $\widehat{\boldsymbol{\theta}}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} \sum_{(\mathbf{x}, \mathbf{y}) \in D_{\mathrm{des}}} \ell\left(\log p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})\right)$.
    Output: $\widehat{\theta}$.
```


### 5.1 Experiment Setup

Model and datasets. In experiments, we consider llava-v1.6-mistral-7b (Liu et al., 2023a) as our base model for self-training with model generated preference data. We additionally consider llava-v1.5-7b (Liu et al., 2023a) based on Vicuna-7B (Chiang et al., 2023b) to directly compare with one concurrent baseline POVID (Zhou et al., 2024). We follow the optimization process described in Section 4 for self-training on image description in Algorithm 1 and description-infused fine-tuning in Algorithm 2 to achieve improved downstream performances. For the self-constructed preference dataset, we gather $6 k$ unlabeled image data randomly sampled from the MSCOCO dataset (Lin et al., 2014) and specifically the train2014 split for its high-quality images popularly used for pre-training and fine-tuning. In the second stage, we randomly subsample $5 \mathrm{k}$ used instruction fine-tuning data from LLaVA's SFT data to construct the description-infused fine-tuning data with model-generated image descriptions. Lastly, we use low-rank adaptation (LoRA) fine-tuning (Hu et al., 2021) instead of full fine-tuning for efficient computation. We defer the detailed prompts and corruptions to Appendix A.

Evaluation. We consider the widely used benchmarks for LVLM evaluation across different domains including: ScienceQA (Lu et al., 2022), TextVQA (Singh et al., 2019), ChartQA (Masry et al., 2022), LLaVA-Bench (Liu et al., 2023a), MMBench (Liu et al., 2023c), MM-Vet (Yu et al., 2023), and MathVista (Lu et al., 2024). Specifically, ScienceQA focuses on scientific question answering and MathVista focuses on math reasoning with visual information. TextVQA consists of images with text-rich contents and ChartQA with visual charts. Lastly, LLaVA-Bench, MMBench, and MM-Vet are three recent benchmarks to comprehensively evaluate a model's capabilities in a wide range of tasks and evaluation criteria. We use the evaluation scripts provided by LLaVA (Liu et al., 2023a) to obtain the results for both our base model and after using STIC to ensure a fair comparison.

### 5.2 Main Results

Table 1: Performance of STIC compared with the original LVLM model across vision-language reasoning tasks. For LLaVA-v1.5 (Vicuna 7B), we directly report the values in the paper of POVID, and "-" indicates an unreported value.

| Model | ScienceQA | TextVQA | ChartQA | LLaVA-Bench | MMBench | MM-Vet | MathVista |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| InstructBLIP (7B) | 60.5 | 50.1 | - | 60.9 | 36.0 | 26.2 | 25.3 |
| mPLUG-OWL2 (7B) | 64.5 | 54.3 | - | 59.9 | 64.5 | 36.2 | 22.2 |
| LLaVA-v1.5 (7B) | 66.8 | 58.2 | 6.32 | 65.4 | 64.3 | 31.1 | 25.1 |
| w/ POVID | 68.8 | - | - | 68.7 | 64.9 | 31.8 | - |
| w/ STIC | $\mathbf{6 9 . 5}$ | $\mathbf{6 1 . 4}$ | $\mathbf{6 . 6 4}$ | $\mathbf{6 8 . 9}$ | $\mathbf{6 5 . 3}$ | $\mathbf{3 2 . 6}$ | $\mathbf{2 7 . 2}$ |
| LLaVA-v1.6 (7B) | 68.9 | 60.3 | 36.4 | 77.3 | 63.7 | 42.2 | 34.6 |
| w/ STIC | $\mathbf{7 5 . 3}$ | $\mathbf{6 5 . 2}$ | $\mathbf{4 1 . 5}$ | $\mathbf{7 9 . 2}$ | $\mathbf{6 7 . 8}$ | $\mathbf{4 5 . 0}$ | $\mathbf{3 7 . 0}$ |

We present our main results in Table 1 and detail the benchmark performances of STIC (LLaVA-v1.6 7B) on MMBench and MM-Vet in Figure 4. In Appendix A, we present detailed results for MMBench in Table 5 and MM-Vet in Table 6. Our results show a consistent and significant improvement of STIC over the original models (LLaVA-v1.5 and LLaVA-v1.6) across all seven datasets. This improvement
is achieved using only self-constructed preference data and a small portion of the model's SFT dataset, which had already been used for fine-tuning the original model.

On average, STIC improves LLaVA-v1.5 by $1.7 \%$, increasing from $45.3 \%$ to $47.0 \%$. and LLaVA-v1. 6 by a notable score of $4.0 \%$, increasing from $54.7 \%$ to $58.7 \%$. The improvement is comprehensive, as detailed in Tables 5 and 6, where STIC consistently enhances performance across all evaluation tasks and targets. Moreover, while STIC improves both LLaVA-v1.5 and LLaVA-v1.6, a more significant improvement is observed in the more advanced model, LLaVA-v1.6. This trend suggests that the extent of self-improvement could be correlated with the model's inherent capabilities.
![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-08.jpg?height=424&width=1378&top_left_y=568&top_left_x=363)

Figure 4: Accuracy improvement of STIC compared to the base LLaVA-v1.6 model across different tasks in Left: MMBench, where the original performances are re-scaled to 60 in plotting and STIC accordingly with the same coefficient for each task. Middle: MM-Vet, where the performances of the original model are re-scaled to 40 and STIC accordingly. Right: LLaVA-Bench, where we report the error bars over three independent runs due to the randomness of GPT-4 evaluation.

## 6 Ablation Studies and Discussions

In this section, we outline the differences of our method with current alignment fine-tuning methods focusing on the vision-language setting. Furthermore, we conduct ablation studies on the key components of STIC to demonstrate their importance and effectiveness. Additionally, we examine the image distribution of our self-training data (MSCOCO) alongside the image distributions of benchmark datasets, revealing a positive correlation between performance gains and similarity in image distributions.

Discussion with POVID. We detail the differences between STIC and POVID. In POVID, the dispreferred response is generated either by adding Gaussian noise to the original image or by manually injecting hallucinations into the ground truth completion, using the labeled object information of the images. In contrast, STIC (1) specifically targets the image description task, (2) constructs preference datasets exclusively from unlabeled images using selfgenerated content for both preferred and dispreferred responses, (3) employs an automatic model generation process without manual injections or modifications, and (4) utilizes only a small portion of SFT data for instruction-following fine-tuning with uniquely infused model descriptions. Lastly, we compare the data types and scales used in POVID and STIC in Figure 5.

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-08.jpg?height=431&width=409&top_left_y=1582&top_left_x=1335)

Figure 5: Data comparison.

Effectiveness of describe-and-respond (DaR) prompting. We assess the significance of the fine-tuning process in STIC by comparing it to the approach of directly allowing the base LVLM to describe an image and then respond to a query with a self-augmented prompt, which we refer to as the describe-and-respond (DaR) prompting method. As indicated in Table 2, applying DaR to the base LVLM yields mixed results, with performance improvements on some datasets and degradation on others, resulting in an overall average drop of $2.2 \%$. In contrast, when DaR is combined with the fine-tuning process of STIC, it leads to a further average enhancement of $1.1 \%$ and a notable $2.8 \%$ on ScienceQA. This demonstrates the synergistic effect of DaR and the fine-tuning process in STIC. Additionally, it is worth noting that STIC achieves a substantial average improvement of $2.9 \%$ even without the DaR prompting method, compared to the base LVLM.

Table 2: Test performance of STIC based on llava-v1.6-mistral-7b. We investigate the benefit of DaR as a prompting method toward the base LVLM model as compared to the benefit on STIC.

| Method | DaR | ScienceQA | TextVQA | ChartQA | LLaVA-Bench | MMBench | MM-Vet | MathVista | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original | $X$ | 68.9 | 60.3 | 36.4 | 77.3 | 63.7 | 42.2 | 34.6 | 54.8 |
|  | $\checkmark$ | 69.9 | 56.6 | 34.6 | 78.5 | 50.7 | 42.3 | 34.7 | 52.5 |
| w/ STIC | $X$ | 72.5 | 63.4 | 39.3 | 78.4 | $\mathbf{6 8 . 7}$ | $\mathbf{4 5 . 7}$ | 35.2 | 57.6 |
|  | $\checkmark$ | $\mathbf{7 5 . 3}$ | $\mathbf{6 5 . 2}$ | $\mathbf{4 1 . 5}$ | $\mathbf{7 9 . 2}$ | 67.8 | 45.2 | $\mathbf{3 7 . 0}$ | $\mathbf{5 8 . 7}$ |

Table 3: Test performance of STIC if we remove negative examples and use positive ones to perform SFT in Stage 1.

| Model | ScienceQA | TextVQA | LLaVA-Bench |
| :---: | :---: | :---: | :---: |
| Original | 68.9 | 60.3 | 77.3 |
| w/ STIC (positive) | 71.8 | 63.7 | 76.7 |
| w/ STIC | $\mathbf{7 5 . 3}$ | $\mathbf{6 5 . 2}$ | $\mathbf{7 9 . 2}$ |

The role of dispreferred samples in STIC. To understand the importance of dispreferred samples in STIC, we conduct an ablation study using llava-v1.6-mistral-7b as the base LVLM. We remove the negative examples from the preference data and only use the positive samples for supervised fine-tuning (SFT), effectively creating an SFT version of STIC. Table 3 shows that omitting the dispreferred samples evev leads to a performance drop of $0.6 \%$ on LLaVA-Bench, while failing to provide equally significant improvement as STIC with preference data. This highlights the crucial role of negative examples in aligning preferences and enabling the model to distinguish between high-quality and low-quality responses. By leveraging both positive and negative examples, STIC effectively improves the model's performance and generates more preferred outputs.

Progression of stages. In Figure 6, we illustrate the sequential improvement in performance of STIC on ScienceQA. While stage 1 focuses exclusively on enhancing the perception capabilities of the LVLM, it still notably improves performance on downstream VQA tasks. Building on the improved image comprehension achieved in stage 1 , stage 2 introduces an enhanced reasoning process that utilizes the model's self-generated image descriptions and results in an even more significant gain. This enhancement further enables the model to self-augment its prompts with Describe and Respond (DaR), resulting in total the substantial performance gains of $6.4 \%$ observed.

Scaling law of STIC. We explore the scaling law of STIC by expanding the preference data in Stage 1. Using the LLaVA-Bench benchmark as a case study, we scale up the preference data from $6 k$ to $12 k$ MSCOCO images. As depicted in Figure 7, there is an obvious gain on the LLaVA-Bench from $1.9 \%$ to $3.1 \%$. This finding demonstrates that STIC can effectively leverage larger amounts of unlabeled image data and presents a cost-effective solution to the challenge of acquiring high-quality vision-language data.

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-09.jpg?height=288&width=415&top_left_y=1233&top_left_x=1321)

Figure 6: Progression of stages in STIC.

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-09.jpg?height=266&width=420&top_left_y=1680&top_left_x=1319)

Figure 7: Scaling law in STIC.

Correlation between image distribution and performance gains. To gain further insight into the effectiveness of STIC across different benchmarks, we conducted a t-SNE visualization analysis comparing the image distributions of MSCOCO, which we used for preference data construction, with those of four benchmarks: ScienceQA, TextVQA, MathVista, and ChartQA (Figure 8). Our analysis revealed a general trend: the greater the overlap between the MSCOCO image distribution and that of a benchmark, the higher the performance gain achieved by STIC on that benchmark. This observation held true for ScienceQA and TextVQA, which exhibited substantial distributional overlap with MSCOCO and yielded the highest performance gains of $6.4 \%$ and $4.9 \%$, respectively. Conversely, MathVista, with its diverse image types and limited overlap with MSCOCO, saw a more modest gain of $2.4 \%$. Interestingly, ChartQA was an outlier, achieving a high gain of $5.1 \%$ despite minimal overlap with MSCOCO, suggesting that the improved image comprehension from STIC played a fundamental role in understanding and reasoning about the charts. Detailed per-benchmark visualizations and discussions are provided in Appendix B.2.

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-10.jpg?height=650&width=884&top_left_y=236&top_left_x=618)

Figure 8: t-SNE visualization of images from MSCOCO and four benchmarks, each sampling $1 k$ images.

## 7 Conclusion

We introduce Self-Training on Image Comprehension (STIC), a novel self-training approach designed to enhance the image comprehension capabilities of large vision language models (LVLMs). Our method leverages a two-stage self-training process, creating a preference dataset for image descriptions from unlabeled images and refining reasoning abilities through description-infused fine-tuning. Extensive experiments across seven vision-language benchmarks demonstrated significant performance improvements, with an average accuracy gain of $4.0 \%$, while reducing the need for supervised fine-tuning data by $70 \%$. Our findings underscore the potential of STIC to harness vast quantities of unlabeled images, offering a cost-effective solution for advancing LVLMs.

The promising results demonstrated by STIC in enhancing the capabilities of 7B LVLMs suggest its potential applicability to larger models, such as those with 13B, 40B, and 100B parameters, if computational resources permit. Additionally, it would be beneficial to investigate the impact of the image distribution used in self-training on STIC, aiming to further refine its effectiveness with curated image datasets. Lastly, an examination of the effects of various image corruptions and "bad" prompts on STIC could provide valuable insights into the effective generation of dispreferred samples.

## References

Adolphs, L., Gao, T., Xu, J., Shuster, K., Sukhbaatar, S. and Weston, J. (2023). The cringe loss: Learning what language not to model. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., BarR, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M. et al. (2022). Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems.

ANTHROPIC (2024). The claude 3 model family: Opus, sonnet, haiku.

azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M. and CalanDRIELLO, D. (2024). A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics. PMLR.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C. and Zhou, J. (2023). Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 .

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T. et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 .

Bao, H., Wang, W., Dong, L., Liu, Q., Mohammed, O. K., Aggarwal, K., Som, S., Piao, S. and WeI, F. (2022). Vlmo: Unified vision-language pre-training with mixture-of-modality-experts. In Advances in Neural Information Processing Systems.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., KorbaK, T., Lindner, D., Freire, P. et al. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217 .

Chen, S., Niu, G., Gong, C., Li, J., Yang, J. and Sugiyama, M. (2021). Large-margin contrastive learning with distance polarization regularizer. In International Conference on Machine Learning. PMLR.

CHEN, X. and He, K. (2021). Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L. et al. (2022). Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794 .

CHEN, Z., Deng, Y., Li, Y. and Gu, Q. (2023a). Understanding transferable representation learning and zero-shot transfer in clip. arXiv preprint arXiv:2310.00927 .

Chen, Z., Deng, Y., YuAn, H., JI, K. and Gu, Q. (2024). Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335 .

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Muyan, Z., Zhang, Q., Zhu, X., LU, L. ET AL. (2023b). Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238 .

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., GonZaleZ, J. E., Stoica, I. and Xing, E. P. (2023a). Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., ZHUANG, Y., GonZALEZ, J. E. ET AL. (2023b). Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023) 26 .

Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., Li, B., Fung, P. N. and Hoi, S. (2024). Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems 36.

Deng, Y., Zhang, W., ChEn, Z. and Gu, Q. (2023). Rephrase and respond: Let large language models ask better questions for themselves. arXiv preprint arXiv:2311.04205 .

Ethayarajh, K., Xu, W., Muennighoff, N., JurafSKY, D. and Kiela, D. (2024). Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306 .

Fränken, J.-P., Zelikman, E., Rafailov, R., Gandhi, K., Gerstenberg, T. and Goodman, N. D. (2024). Self-supervised alignment with mutual information: Learning to follow principles without preference labels. arXiv preprint arXiv:2404.14313 .

Gao, L., Schulman, J. and Hilton, J. (2023a). Scaling laws for reward model overoptimization. In International Conference on Machine Learning. PMLR.

Gao, P., Han, J., Zhang, R., Lin, Z., Geng, S., Zhou, A., Zhang, W., Lu, P., He, C., Yue, X., Li, H. and QIAO, Y. (2023b). Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010 .

Gao, P., Zhang, R., Liu, C., Qiu, L., Huang, S., Lin, W., Zhao, S., Geng, S., Lin, Z., Jin, P., Zhang, K., Shao, W., Xu, C., He, C., He, J., Shao, H., Lu, P., Li, H. and Qiao, Y. (2024). Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. In International Conference on Machine Learning (ICML).

Goel, S., Bansal, H., Bhatia, S., Rossi, R., Vinay, V. and Grover, A. (2022). Cyclip: Cyclic contrastive language-image pretraining. Advances in Neural Information Processing Systems 35 $6704-6719$.

Hu, E. J., WAllis, P., Allen-Zhu, Z., Li, Y., Wang, S., WAng, L., Chen, W. et al. (2021). Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Jia, C., Yang, Y., Xia, Y., Chen, Y., Parekh, Z., Pham, H., Le, Q. V., Sung, Y., Li, Z. and DUERIG, T. (2021a). Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (M. Meila and T. Zhang, eds.), vol. 139 of Proceedings of Machine Learning Research. PMLR.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z. and DuERIG, T. (2021b). Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning. PMLR.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. D. L., BreSSAND, F., LenGYel, G., Lample, G., SAULNiER, L. ET AL. (2023). Mistral 7b. arXiv preprint arXiv:2310.06825 .

Josifoski, M., Sakota, M., Peyrard, M. and West, R. (2023). Exploiting asymmetry for synthetic training data generation: Synthie and the case of information extraction. arXiv preprint arXiv:2303.04132 .

Kim, D., Park, C., Kim, S., Lee, W., Song, W., Kim, Y., Kim, H., Kim, Y., Lee, H., Kim, J. ET AL. (2023). Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling. arXiv preprint arXiv:2312.15166.

KIM, W., Son, B. and KIM, I. (2021). ViLT: Vision-and-language transformer without convolution or region supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (M. Meila and T. Zhang, eds.), vol. 139 of Proceedings of Machine Learning Research. PMLR.

Li, J., Li, D., SaVarese, S. and Hoi, S. (2023a). Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning. PMLR.

Li, J., Li, D., Savarese, S. and Hoi, S. (2023b). Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597

Li, L. H., YatsKar, M., Yin, D., Hsieh, C. and Chang, K. (2019). Visualbert: A simple and performant baseline for vision and language. CoRR abs/1908.03557.

Li, L. H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., YuAn, L., Zhang, L., Hwang, J.-N. ET AL. (2022). Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., Choi, Y. and Gao, J. (2020). Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXX (A. Vedaldi, H. Bischof, T. Brox and J. Frahm, eds.), vol. 12375 of Lecture Notes in Computer Science. Springer.

li, Y., Bubeck, S., Eldan, R., Giorno, A. D., Gunasekar, S. and Lee, Y. T. (2023c). Textbooks are all you need ii: phi-1.5 technical report.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and ZitniCK, C. L. (2014). Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer.

LiU, H., LI, C., LI, Y. and LEE, Y. J. (2023a). Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 .

LiU, H., Li, C., Li, Y., Li, B., ZHAnG, Y., Shen, S. and LeE, Y. J. (2024). Llava-next: Improved reasoning, ocr, and world knowledge.

LiU, H., Li, C., WU, Q. and LeE, Y. J. (2023b). Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS) 36.

LiU, Y., Duan, H., ZHang, Y., Li, B., Zhang, S., ZhaO, W., Yuan, Y., Wang, J., He, C., LiU, Z. ET AL. (2023c). Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 .

lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M. and GaO, J. (2024). Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR).

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P. and KALYAN, A. (2022). Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35 2507-2521.

MasrY, A., Do, X. L., Tan, J. Q., JotY, S. and Hoque, E. (2022). Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022.

McKinZie, B., Gan, Z., FauconniER, J.-P., Dodge, S., ZHAnG, B., Dufter, P., ShAH, D., Du, X., Peng, F., Weers, F. et al. (2024). Mm1: Methods, analysis \& insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611 .

OPENAI (2023a). Gpt-4 technical report.

OPENAI (2023b). Gpt-4v(ision) system card.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 2773027744.

Pang, R. Y., Yuan, W., Cho, K., He, H., Sukhbaatar, S. and Weston, J. (2024). Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733 .

Prasad, A., Stengel-Eskin, E. and Bansal, M. (2023). Rephrase, augment, reason: Visual grounding of questions for vision-language models. arXiv preprint arXiv:2310.05861 .

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., AsKell, A., MishKin, P., ClarK, J. ET AL. (2021). Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML). PMLR.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. arXiv preprint

![](https://cdn.mathpix.com/cropped/2024_06_04_18d42068839193d4cf89g-13.jpg?height=40&width=271&top_left_y=2013&top_left_x=407)

Rosset, C., Cheng, C.-A., Mitra, A., Santacroce, M., Awadallah, A. and Xie, T. (2024). Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715 .

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .

Shen, S., Li, L. H., Tan, H., Bansal, M., Rohrbach, A., Chang, K.-W., Yao, Z. and KeUTZER, K. (2022). How much can clip benefit vision-and-language tasks? In ICLR.

Singh, A., Hu, R., Goswami, V., Couairon, G., Galuba, W., Rohrbach, M. and Kiela, D. (2021). FLAVA: A foundational language and vision alignment model. CoRR abs/2112.04482.

Singh, A., Natarjan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D. and RoHrbach, M. (2019). Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L.-Y., Wang, Y.-X., YANG, Y. ET AL. (2023). Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525 .

TAN, H. and BANSAL, M. (2019). LXMERT: Learning cross-modality encoder representations from transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 (K. Inui, J. Jiang, V. Ng and X. Wan, eds.). Association for Computational Linguistics.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. and Hashimoto, T. B. (2023). Stanford alpaca: An instruction-following llama model.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., SchalkWYK, J., DAI, A. M., HaUTH, A. ET AL. (2023). Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 .

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., BhargaVa, P., Bhosale, S. et AL. (2023a). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., BhargaVa, P., Bhosale, S. et AL. (2023b). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., MA, J., Zhou, C., Zhou, J. and Yang, H. (2022a). Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. CoRR abs/2202.03052.

Wang, Z., Yu, J., Yu, A. W., Dai, Z., TsvetKov, Y. and Cao, Y. (2022b). SimVLM: Simple visual language model pretraining with weak supervision. In ICLR.

Wu, T., Yang, G., Li, Z., Zhang, K., LiU, Z., Guibas, L., Lin, D. and WetZStein, G. (2024). Gpt-4v (ision) is a human-aligned evaluator for text-to-3d generation. arXiv preprint arXiv:2401.04092 .

XAI (2024). Grok-1.5 vision preview.

Xiong, W., Dong, H., Ye, C., Zhong, H., Jiang, N. and Zhang, T. (2023). Gibbs sampling from human feedback: A provable kl-constrained framework for rlhf. arXiv preprint arXiv:2312.11456 .

Xu, J., Lee, A., Sukhbaftar, S. and Weston, J. (2023). Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682 .

Yu, W., YanG, Z., Li, L., WanG, J., Lin, K., LiU, Z., WanG, X. and WAnG, L. (2023). Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 .

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J. and Weston, J. (2024). Selfrewarding language models. arXiv preprint arXiv:2401.10020 .

Zhao, Y., Joshi, R., Liu, T., Khalman, M., SaleH, M. and Liu, P. J. (2023). Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425 .

ZHEng, C., WAng, Z., Ji, H., HuAng, M. and Peng, N. (2024). Weak-to-strong extrapolation expedites alignment. arXiv preprint arXiv:2404.16792 .

Zhou, Y., Cui, C., Rafailov, R., Finn, C. and YaO, H. (2024). Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411 .
</end of paper 1>


<paper 2>
# TS-Align: A Teacher-Student Collaborative Framework for Scalable Iterative Finetuning of Large Language Models 

Chen Zhang ${ }^{1}$ Chengguang Tang ${ }^{2}$ Dading Chong ${ }^{3}$<br>Ke Shi $^{2} \quad$ Guohua Tang $^{2} \quad$ Feng Jiang $^{4}$ Haizhou $\mathbf{L i}^{1,4}$<br>${ }^{1}$ National University of Singapore, Singapore<br>${ }^{2}$ Tencent AI Lab, China ${ }^{3}$ Peking University, China<br>${ }^{4}$ The Chinese University of Hong Kong, Shenzhen, China<br>chen_zhang@u.nus.edu; jeffreyjiang@cuhk.edu.cn


#### Abstract

Mainstream approaches to aligning large language models (LLMs) heavily rely on human preference data, particularly when models require periodic updates. The standard process for iterative alignment of LLMs involves collecting new human feedback for each update. However, the data collection process is costly and challenging to scale. To address this issue, we introduce the "TS-Align" framework, which fine-tunes a policy model using pairwise feedback data automatically mined from its outputs. This automatic mining process is efficiently accomplished through the collaboration between a large-scale teacher model and a smallscale student model. The policy fine-tuning process can be iteratively repeated using onpolicy generations within our proposed teacherstudent collaborative framework. Through extensive experiments, we demonstrate that our final aligned policy outperforms the base policy model with an average win rate of $69.7 \%$ across seven conversational or instruction-following datasets. Furthermore, we show that the ranking capability of the teacher is effectively distilled into the student through our pipeline, resulting in a small-scale yet effective reward model for policy model alignment.


## 1 Introduction

General-purpose conversational AI assistants, such as GPT-4 (Achiam et al., 2023) and Gemini (Google et al., 2023), are empowered by aligning large pretrained language models with humanpreferred behaviors (Stiennon et al., 2020a; Ouyang et al., 2022; Bai et al., 2022a). These aligned LLMs showcase exceptional capabilities in instruction following (Touvron et al., 2023; Tunstall et al., 2023), natural conversation (Thoppilan et al., 2022; Ding et al., 2023), safety (Ganguli et al., 2022; Dai et al., 2023), reasoning (Wei et al., 2022b; Kojima et al., 2022), among others. Commonly-used LLM alignment techniques include instruction tuning (Wei et al., 2022a; Chung et al., 2022), reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Ziegler et al., 2019), and direct preference optimization (DPO) (Rafailov et al., 2023).

While recent research has focused significantly on the development of more sophisticated alignment techniques (Song et al., 2023; Yuan et al., 2023; Liu et al., 2023; Xu et al., 2023b; Ethayarajh et al., 2024; Meng et al., 2024), it is worth noting that LLM alignment is not a one-time process and the model requires continuous refinement to adapt to evolving user needs and changing linguistic patterns. The standard practice for iterative alignment of the LLMs is to gather new human preference data for every subsequent update to the model. For instance, Touvron et al. (2023) performs five iterations of RLHF finetuning on the base SFT LLaMA2 model. For each iteration, they update the reward model with newly collected human preference data. This process poses challenges regarding scalability and resource requirements.

To alleviate the issue, existing research adopts self-evolution (Li et al., 2023a; Yuan et al., 2024; Chen et al., 2024) or external model supervision (Xu et al., 2023b; Singh et al., 2023; Guo et al., 2024). The effectiveness of self-evolution is highly dependent on the quality of the base model as it operates without the introduction of external supervision or knowledge during refinement. For instance, in their study, Yuan et al. (2024) utilize a sophisticated 70B LLaMA-2 model to demonstrate the potential of their iterative self-rewarding procedure. When employing external model supervision, it is crucial to utilize a robust model that can effectively generalize to new data. Typically, these models are substantially large to avoid reward overoptimization (Gao et al., 2023). Despite being reliable, labeling abundant data with a large-scale model is still very costly and time-consuming.

In this paper, we aim to balance reliability and efficiency in the data labeling process during the it-
erative fine-tuning of the policy model. To achieve this, we propose TS-Align, a teacher-student collaborative framework that leverages the reliability of the large-scale teacher model without requiring it to process all the candidates. Specifically, TS-Align uses a base supervised fine-tuned policy model to generate response candidates for a diverse set of instruction prompts sampled from public instruction-tuning datasets. A small-scale student reward model (RM) provides coarse-grained annotations, allowing for the quick processing of abundant unlabeled data and the selection of preference pairs from the candidates. Next, the strong teacher helps re-rank the selected pairs reliably. The policy model is then fine-tuned on the re-ranked preference data using DPO. This process is repeated in several iterations. Given that the student RM, with its smaller parameter size, is not as robust as the teacher model, we iteratively update the student using an adapter-based multi-task training setup (Pfeiffer et al., 2021). This training uses the same model-labeled preference data to enhance the student's reliability, which can be perceived as distilling new knowledge from the large teacher model to the small student RM.

Our contributions are three-fold: (1) We introduce "TS-Align", an efficient and reliable pipeline for the iterative alignment of large language models. This approach circumvents the need for costly human annotations by employing a teacher-student model collaboration to automatically extract preference data from the policy model's own outputs. (2) We demonstrate that the teacher-student collaborative mechanism produces a strong aligned policy model with an average win rate of $69.7 \%$ over the base policy on 7 conversational or instructionfollowing datasets, while also being efficient. (3) Through our pipeline, the response ranking capability of the teacher model is progressively distilled into the student model. We demonstrate that the enhanced capability of the final student model can be transferred to align other policy models.

## 2 Preliminaries

This section presents the background information. We also provide the notation list in Table 1 for better illustration.

Supervised Finetuning The base policy model should possess basic instruction-following and natural conversational capabilities. Hence, the initial step involves supervised finetuning of a pretrained

| Symbol | Definition |
| :--- | :--- |
| $\pi$ | A general notation for the policy model. |
| $\pi_{0}$ | The supervised fine-tuned base policy model. |
| $\pi_{t}$ | The policy model to be aligned at the t-th iteration |
| $r$ | A general notation for reward model. |
| $\mathcal{S}_{0}$ | The base student reward model. |
| $\mathcal{S}_{t}$ | The student reward model to be updated at the t-th iteration. |
| $\mathcal{M}$ | The teacher reward model. |
| $\mathcal{X}$ | The source of prompt instructions. |
| $\mathcal{D}_{S F T}$ | The supervised fine-tuning dataset. |
| $\mathcal{D}_{\text {pref }}$ | The offline human preference dataset. |
| $x$ | A single instruction prompt. |
| $\mathbf{y}$ | A set of completion candidates of $x$. |
| $y$ | The completion of $x$. |
| $y^{+}$ | The favored completion of $x$. |
| $y^{-}$ | The unfavored completion of $x$. |
| $\mathcal{D}_{\text {ins }}^{t}$ | The batch of instruction prompts at the t-th iteration. |
| $\mathcal{D}_{\text {auto }}^{t}$ | The model-annotated preference dataset derived from $\mathcal{D}_{\text {ins }}^{t}$ |

Table 1: The list of notations.

language model:

$$
\mathcal{L}_{\mathrm{SFT}}\left(\pi_{0}, \mathcal{D}_{\mathrm{SFT}}\right)=-\mathbb{E}_{(x, y) \sim \mathcal{D}_{\mathrm{SFT}}}\left[\log P_{\pi}(y \mid x)\right]
$$

Direct Preference Optimization DPO is derived from the Bradley-Terry model of human preferences (Bradley and Terry, 1952), which defines the human preference distribution as:

$$
\begin{equation*}
P^{*}\left(y^{+}>y^{-} \mid x\right)=\frac{\exp \left(r^{*}\left(x, y^{+}\right)\right)}{\exp \left(r^{*}\left(x, y^{+}\right)\right)+\exp \left(r^{*}\left(x, y^{-}\right)\right)} \tag{1}
\end{equation*}
$$

where $r^{*}$ represents a latent reward model that captures the true preferences and it is parameterized by $r_{\phi}$, which is trained via the following binary classification objective on $\mathcal{D}_{\text {pref }}$ :

$$
\begin{aligned}
\mathcal{L}_{\mathrm{RM}}\left(r_{\phi}, \mathcal{D}_{\text {pref }}\right)= & -\mathbb{E}_{\left(x_{j}, y_{j}^{+}, y_{j}^{-}\right) \sim \mathcal{D}_{\text {pref }}}\left[\operatorname { l o g } \sigma \left(r_{\phi}\left(x_{j}, y_{j}^{+}\right)\right.\right. \\
& \left.\left.-r_{\phi}\left(x_{j}, y_{j}^{-}\right)\right)\right]
\end{aligned}
$$

Instead of modeling $r_{\phi}$, DPO utilizes a reparameterization trick on $r^{*}(x, y)$, effectively converting the objective 1 to rely solely on the optimal policy $\left(\pi^{*}\right)$ and reference policy $\left(\pi_{\text {ref }}\right.$ ) models:

$$
P^{*}\left(y^{+}>y^{-} \mid x\right)=\frac{1}{1+\exp \left(\beta \log \frac{\pi^{*}\left(y^{-} \mid x\right)}{\pi_{\text {ref }}\left(y^{-} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y^{+} \mid x\right)}{\pi_{\text {ref }}\left(y^{+} \mid x\right)}\right)}
$$

where $\beta$ is a hyperparameter. $\pi^{*}$ is estimated with a parameterized policy $\pi_{\theta}$, which is learned with the maximum likelihood objective:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x_{j}, y_{j}^{+}, y_{j}^{-}\right) \sim \mathcal{D}_{\text {pref }}} & {\left[\operatorname { l o g } \sigma \left(\beta \log \frac{\pi_{\theta}\left(y_{j}^{+} \mid x_{j}\right)}{\pi_{\text {ref }}\left(y_{j}^{+} \mid x_{j}\right)}\right.\right.} \\
& \left.\left.-\beta \log \frac{\pi_{\theta}\left(y_{j}^{-} \mid x_{j}\right)}{\pi_{\text {ref }}\left(y_{j}^{-} \mid x_{j}\right)}\right)\right]
\end{aligned}
$$

Both $\pi_{r e f}$ and $\pi_{\theta}$ are initialized as $\pi_{0}$. During training, $\pi_{r e f}$ is frozen while $\pi_{\theta}$ is optimized.

## 3 The TS-Align Pipeline

The overall workflow of TS-Align is summarized in Algorithm 1. The key idea is to align the policy model through multiple alignment iterations and the procedure for each iteration is depicted in Figure 1. Sections $\S 3.1$ to $\S 3.3$ discuss the essential details of this pipeline.

```
Algorithm 1 TS-Align
Require: $\pi_{0}, \mathcal{S}_{0}, \mathcal{M}, \mathcal{X}$
    for $t \leftarrow 0$ to $T$ do
        Sample prompts from $\mathcal{X}$ to form $\mathcal{D}_{\text {ins }}^{t}$.
        Initialize empty set $\mathcal{D}_{\text {auto }}^{t}$.
        for $x$ in $\mathcal{D}_{\text {ins }}^{t}$ do
            $\mathbf{y} \leftarrow \operatorname{Generate}\left(\pi_{t}, x\right)$.
            $\left\{x, y_{j}^{+}, y_{j}^{-}\right\} \leftarrow \operatorname{Score}\left(\mathcal{S}_{t}, x, \mathbf{y}\right)$.
            Rerank $\left(x, y_{i}^{+}, y_{i}^{-}\right)$with $\mathcal{M}$.
            Add re-ranked $\left(x, y_{i}^{+}, y_{i}^{-}\right)$to $\mathcal{D}_{\text {auto }}^{t}$
        end for
        $\mathcal{S}_{t+1} \leftarrow \operatorname{Update}\left(\mathcal{S}_{t}, \mathcal{D}_{\text {auto }}^{t}\right)$
        $\pi_{t+1} \leftarrow \operatorname{DPO}\left(\pi_{t}, \mathcal{D}_{\text {auto }}^{t}\right)$
    end for
```


### 3.1 Automatic Preference Pair Construction

We construct a prompt source $\mathcal{X}$ that contains instruction prompts from diverse public instructiontuning datasets (described in §4.1). For each alignment iteration $t$, we sample an abundant amount of instructions from $\mathcal{X}$ to form $\mathcal{D}_{\text {ins }}^{t}$ for preference pair construction. For each $x \in \mathcal{D}_{\text {ins }}^{t}, K$ response candidates, $\mathbf{y}=\left\{y_{1}, y_{2}, \ldots, y_{k}\right\}$, is generated from $\pi_{t}$. $\mathcal{S}_{t}$ is applied to score the candidates. A preference pair, $\left(y^{+}, y^{-}\right)$, is formed using the candidates with the highest $\left(y^{\text {best }}\right.$ ) and lowest scores $\left(y^{\text {worst }}\right.$ ) respectively.

Given the potential unreliability of annotations from $\mathcal{S}_{t}$, we utilize a strong teacher model, $\mathcal{M}$, to refine the ranking of $\left(y^{+}, y^{-}\right)$. Subsequently, we add the pair to the model-annotated preference dataset $\mathcal{D}_{\text {auto }}^{t}$. The benefits of this teacher-student collaborative mechanism are the efficiency in data annotation and the continuous improvement of the student reward model through knowledge distillation in each alignment iteration.

### 3.2 The Student Reward Model

Initial Base Version $\mathcal{S}_{0}$ is initially pre-trained on a predefined human-labeled preference dataset, $\mathcal{D}_{\text {pref }}=\left\{y_{j}^{+}>y_{j}^{-} \mid x_{j}\right\}_{j=1}^{\left|\mathcal{D}_{\text {pref }}\right|}$. We implement $\mathcal{S}_{0}$ as a RoBERTa-based scoring model, which is first trained on concatenated text sequences $\left(x_{j}, y_{j}\right)$ for faster convergence and domain adaptation, utilizing the masked language modeling (MLM) objective. Next, $\mathcal{S}_{0}$ learns to predict a higher score for $y_{j}^{+}$ than $y_{j}^{-}$by minimizing the following margin ranking loss:

$\mathcal{L}_{\mathrm{RM}}\left(\mathcal{S}, \mathcal{D}_{\text {pref }}\right)=\frac{1}{\left|\mathcal{D}_{\text {pref }}\right|} \sum_{j=1}^{\left|\mathcal{D}_{\text {pref }}\right|} \max \left(0, s_{j}^{-}-s_{j}^{+}+0.1\right)$

where $s_{j}^{-}$and $s_{j}^{+}$represent the output scores for $y_{j}^{-}$ and $y_{j}^{+}$respectively.

Subsequent Update After constructing the modelannotated preference dataset $\mathcal{D}_{\text {auto }}^{t}$ using the procedure outlined in $\S 3.1$, we adapt the student reward model to the new data using adapter-based multitask learning (Pfeiffer et al., 2021). Specifically, the student is re-trained with preference data batches from previous iterations, along with those from the current iteration, $\left\{\mathcal{D}_{\text {pref }}, \mathcal{D}_{\text {auto }}^{0}, \ldots, \mathcal{D}_{\text {auto }}^{t}\right\}$. Each adapter is fine-tuned with one data batch using the above-mentioned margin ranking loss function, while the shared RoBERTa encoder is fine-tuned on all the data. This approach not only facilitates the distillation of the new knowledge from the teacher into the student but also mitigates the forgetting of previously learned knowledge. Motivated by previous research on model weight averaging (Wortsman et al., 2022; Rame et al., 2022), we average the weights of all the injected adapters from different alignment iterations for faster inference.

### 3.3 Aligning Policy Model

We adopt DPO to align the base policy model $\pi_{0}$. To stabilize the training process, we add the supervised finetuning loss term to the DPO objective:

$$
\mathcal{L}_{\text {final }}\left(\pi_{\theta}\right)=\alpha \mathcal{L}_{\mathrm{SFT}}+\mathcal{L}_{\mathrm{DPO}}
$$

where alpha is a hyperparameter set to 0.05 . The SFT objective is optimized with the positive responses $\left\{x_{j}, y_{j}^{+}\right\}$in $\mathcal{D}_{\text {auto }}^{t}$.

## 4 Experiment Setup

### 4.1 Datasets

Prompt Source We sample new instruction prompts from a diverse array of open-source instruction-tuning datasets, which are summarized in table 8. For each alignment iteration, we sample

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-04.jpg?height=457&width=1587&top_left_y=223&top_left_x=226)

Figure 1: The figure depicts one alignment iteration of TS-Algin. The process can be repeated multiple times on the updated policy model and student reward model.

| Test Datasets | Size | Avg. \#Prompt <br> Words | Avg. \#Turns | Purpose |
| :--- | :---: | :---: | :---: | :---: |
| HH-RLHF | 8,550 | 93.05 | 2.38 | $\mathrm{P}, \mathrm{R}$ |
| PKU-BeaverTails | 2,985 | 13.17 | 1.00 | $\mathrm{P}, \mathrm{R}$ |
| Alpaca-Eval | 805 | 28.56 | 1.00 | $\mathrm{P}$ |
| IFEval | 541 | 37.07 | 1.00 | $\mathrm{P}$ |
| SHP | 18,409 | 148.79 | 1.00 | $\mathrm{R}$ |
| Alpaca-Farm | 17,701 | 28.57 | 1.00 | $\mathrm{R}$ |

Table 2: Statistics of the test data. In the purpose column, "P" stands for policy model evaluation, and "R" denotes reward model evaluation.

$5 \mathrm{~K}$ prompts from each dataset. In total, we use $30 \mathrm{~K}$ prompts per alignment iteration.

Test Datasets We evaluate the policy models on four conversational or instruction-following test datasets: (1) Anthropic HH-RLHF Test ${ }^{1}$ (Bai et al., 2022a), (2) PKU-BeaverTails Test (Ji et al., 2023), (3) Alpaca-Eval (Li et al., 2023b), and (4) IFEval (Zhou et al., 2023). All the datasets measure the model's ability to follow instructions and provide helpful responses. HH-RLHF and PKUBeaverTails also examine the models' abilities to handle harmful user input.

The reward models are assessed on four offline human preference test datasets: (1) Anthropic HHRLHF Test, (2) PKU-BeaverTails Test, (3) the Standford Human Preference (SHP) Test (Ethayarajh et al., 2022), and (4) Alpaca-Farm (Dubois et al., 2023). The statistics of test datasets are presented in table 2 .

### 4.2 Implementation Details

Policy Models We use the LLaMA Factory library (Zheng et al., 2024) for all the finetuning[^0]

experiments. Low-rank adaptation (LoRA) (Hu et al., 2022) is used with rank set to 8 and alpha set to 16. The target modules are the query and key projection matrices. Each finetuning experiment is performed on a single 40GB NVIDIA A100 GPU card. We use a batch size of 8 and 2 gradient accumulation steps and employ a cosine learning rate schedule. The off-the-shelf Alpaca-7B (Taori et al., 2023) is adopted as $\pi_{0}$ in algorithm 1 and the number of responses sampled from the policy model in the "Generate" step is set to 16. In total, two alignment iterations are performed.

Reward Model We implement the student RM using the adapter-transformers library (Pfeiffer et al., 2020). The base student RM is a RoBERTa-Large encoder followed by a linear layer with output dimension 1 and a sigmoid activation function. It is fine-tuned on $40 \mathrm{~K}$ human preference data, employing a learning rate of $5 e^{-6}$ and a batch size of 8 . The human preference data consist of instances sampled equally from the training splits of Anthropic HH-RLHF, Stanford SHP, PKUBeaverTails, and UltraFeedback (Cui et al., 2023). In the subsequent alignment iterations, the adapters are finetuned with a learning rate of $1 e^{-5}$ and a batch size of 8 . The adapters are configured to the variant introduced in Houlsby et al. (2019).

For the teacher model, we experiment with both the UltraRM-13B model (Cui et al., 2023), which is initialized from LLaMA2-13B and fine-tuned on a mixture of UltraFeedback and an equal-size sample from three open-source datasets including Anthropic HH-RLHF, Standford SHP, and OpenAI Summarization (Stiennon et al., 2020b).

|  | Harmless Base | Helpful Base | Helpful Online | Helpful Rejection | Beavertails | Alpaca-Eval | IFEval | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Direct DPO | $57.66(0.91)$ | $67.74(0.87)$ | $64.09(1.30)$ | $67.97(0.81)$ | $57.73(0.74)$ | $54.89(1.54)$ | $52.74(1.74)$ | 60.40 |
| BoN | $55.41(0.93)$ | $61.60(0.92)$ | $60.54(1.33)$ | $63.13(0.85)$ | $54.48(0.76)$ | 47.04 (1.58) | $43.71(1.78)$ | 55.13 |
| OAIF (iter1) | $53.58(0.92)$ | $69.71(0.86)$ | $64.12(1.29)$ | $70.44(0.80)$ | $59.27(0.73)$ | $56.22(1.54)$ | $51.41(1.77)$ | 60.68 |
| OAIF (iter2) | $56.60(0.93)$ | $70.61(0.85)$ | $66.88(1.27)$ | $71.12(0.79)$ | $60.03(0.73)$ | $56.45(1.55)$ | $53.31(1.75)$ | 62.14 |
| Student RM on | $62.50(0.91)$ | $73.91(0.83)$ | 69.87 (1.24) | $74.47(0.76)$ | $65.01(0.70)$ | 57.26 (1.57) | $52.32(1.76)$ | 65.05 |
| Student RM only (iter2) | $64.47(0.86)$ | $77.57(0.78)$ | $71.66(1.21)$ | $76.52(0.73)$ | $63.48(0.69)$ | $59.63(1.52)$ | $54.90(1.79)$ | 66.89 |
| Teacher RM only (iter1) | $61.96(0.92)$ | $77.26(0.79)$ | $73.04(1.19)$ | $77.14(0.72)$ | $63.00(0.72)$ | 62.54 | $57.92(1.73)$ | 67.55 |
| Teacher RM only (iter2) | $64.57(0.89)$ | $82.92(0.70)$ | 78.04 (1.10) | $82.68(0.64)$ | $70.08(0.66)$ | $\overline{67.65}(1.44)$ | $\underline{58.67}(1.74)$ | 72.09 |
| TS-Align (iter1) | $\overline{60.70}(0.91)$ | $75.66(0.80)$ | $69.68(1.24)$ | $76.03(0.74)$ | $62.54(0.71)$ | $60.06(1.53)$ | $\overline{55.20}(1.77)$ | 65.70 |
| TS-Align (iter2) | $64.82(0.89)$ | $79.22(0.75)$ | 73.70 (1.18) | $79.46(0.69)$ | $69.45(0.66)$ | $62.11(1.50)$ | $\mathbf{5 9 . 1 2 ( 1 . 7 7 )}$ | 69.70 |

Table 3: Win rate (\%) of the aligned policy models against the base Alpaca-7B model as judged by GPT-4-Turbo. The standard errors are displayed in the bracket. All the methods went through two alignment iterations except "Direct DPO" and "BoN". Iter1 and Iter2 represent the first and the second alignment iterations respectively. The best score is highlighted in bold while the second best is underlined.

| Annotator | Speed | Cost | \#Parameters |
| :--- | :---: | :---: | :---: |
| RoBERTa RM | $23.19 \mathrm{it} / \mathrm{s}$ | - | $\sim 370 \mathrm{M}$ |
| UltraRM | $14.60 \mathrm{it} / \mathrm{s}$ | - | $\sim 13 \mathrm{~B}$ |
| GPT-3.5-turbo | $0.55 \mathrm{it} / \mathrm{s}$ | $4.6 \mathrm{e}-4 \$ \mathrm{it}$ | - |
| Human | $0.027 \mathrm{it} / \mathrm{s}$ | $0.3 \$ / \mathrm{it}$ | - |

Table 4: Cost analysis of different annotators used in our experiments. "it/s" denotes the average number of instances per second and "\$/it" denotes the average USD per instance. The human annotation information is obtained from (Li et al., 2023b).

### 4.3 Evaluation \& Baselines

Metrics We use the accuracy metric to evaluate the reward model. To evaluate the policy model, we employ both automatic and human evaluation. For automatic evaluation, we utilize the pairwise comparison framework from AlpacaEval (Li et al., 2023b), setting the base policy model as the reference and "weighted_alpaca_eval_gpt4_turbo" as the LLM annotator, which is reported to have the highest agreement with the human evaluation. The models are compared based on their win rate against the reference model.

For human evaluation, we apply an identical pairwise comparison method, focusing on a subset of 200 data instances from Alpaca-Eval and IFEval respectively. The setup details of human evaluation are presented in Appendix C.

Baselines We benchmark our final aligned policy model against the following baselines: (1) Iterative DPO alignment with the fixed student model, (2) Best-of-N (BoN) sampling (Touvron et al., 2023) using the teacher model annotations, (3) Iterative DPO alignment with the fixed teacher model, (4) Iterative DPO alignment using online AI Feedback ${ }^{2}$ (Guo et al., 2024) (OAIF), and (5) direct[^1]

DPO alignment using the $40 \mathrm{~K}$ human preference data, which is also used to train the base student RM. Additional descriptions of the baselines are presented in Appendix D. We excluded the Iterative RLHF (Touvron et al., 2023) baseline due to the unstable training associated with LoRA-based proximal policy optimization, and the insufficient computational resources for full model training.

## 5 Results \& Analysis

### 5.1 Alignment Performance

In this section, we discuss the results of various iterative alignment strategies. Table 3 presents the win rate of the final aligned policy model compared to the base Alpaca-7B SFT model, as evaluated by GPT-4-Turbo. Firstly, we observe that even after the initial alignment iteration, the average win rates of on-policy iterative alignment methods, which use preference data derived from policy model outputs, exceed the direct DPO method which utilizes human-labeled preference data. This observation aligns with recent research on using on-policy data for preference fine-tuning (Tajwar et al., 2024; Yuan et al., 2024) and supports the feasibility of using the model-in-the-loop data annotation procedure as an efficient alternative to the human preference data collection method. Additionally, as shown in Table 4, human annotation is much more expensive than using models.

Secondly, we also observe that SFT with bestof-N sampling is less effective compared to direct DPO and "UltraRM-13B only (iter1)." Notably, "UltraRM-13B only (iter1)", which utilizes the same annotated preference data as BoN, outperforms BoN by an average win rate of $\sim 11 \%$. These results highlight the advantage of DPO, which provides both positive and negative responses for the

|  | Harmless Base | Helpful Base | Helpful Online | Helpful Rejection | Beavertails | Alpaca-Eval | IFEval | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SPIN (iter2) | $61.51(0.91)$ | $67.90(0.88)$ | $66.26(1.25)$ | $68.90(0.80)$ | $62.39(0.70)$ | $73.50(1.37)$ | $\mathbf{6 9 . 2 2}(1.75)$ | 67.10 |
| Zephyr-7B-Beta | $63.73(0.91)$ | $75.11(0.81)$ | $72.83(1.17)$ | $75.33(0.75)$ | $\mathbf{6 8 . 6 6}(0.67)$ | $70.97(1.45)$ | $67.64(1.75)$ | $\mathbf{7 0 . 6 1}$ |
| Initial Student RM | $\mathbf{6 5 . 8 7}(0.83)$ | $78.76(0.72)$ | $72.15(1.16)$ | $77.00(0.68)$ | $63.87(0.85)$ | $72.82(1.39)$ | $56.95(1.82)$ | 69.63 |
| Final Student RM | $60.42(0.90)$ | $\mathbf{7 9 . 9 0}(0.74)$ | $\mathbf{7 3 . 6 1}(1.15)$ | $\mathbf{8 0 . 0 4}(0.67)$ | $61.23(0.89)$ | $\mathbf{7 6 . 2 1}(1.34)$ | $61.26(1.84)$ | 70.38 |

Table 5: Win rate (\%) of the final aligned models vs the base "Mistral-7B-SFT-Beta" as judged by GPT-4-Turbo.

policy model to learn from, supporting our decision to use DPO for iterative alignment.

Furthermore, the iterative OAIF approach does not perform as well as the iterative DPO, which utilizes either the fixed RoBERTa student RM or the fixed UltraRM-13B teacher RM. A key reason is that OAIF samples only two responses per instruction prompt and relies on external API to rank them, whereas using an RM allows for the simultaneous scoring of multiple responses and the identification of preference pairs with a large score margin, which are beneficial for DPO finetuning (Tajwar et al., 2024). Although API-based prompting could also rank or score multiple responses, this process is considerably slower than using an RM, as demonstrated by the annotation speed comparison in Table 4 between GPT-3.5-Turbo and the RMs.

Additionally, the win rate of our proposed student-teacher collaboration approach falls between the results achieved using solely the student RM and those using only the teacher RM across both iterations. These results are in line with our goal of achieving a good balance between efficiency and alignment performance, especially when the number of instruction prompts and the size of response candidates are large. The collaborative mechanism effectively distills the teacher's ranking capabilities into the student RM, as evidenced in subsequent sections, where we demonstrate that the refined student RM facilitates strong alignment with other base SFT models (\$5.2) and shows improvement in preference annotation on offline human preference test data (\$5.3).

Finally, the policy models demonstrate improved performance after two alignment iterations compared to just a single iteration. For example, our proposed pipeline leads to a $4 \%$ win rate improvement on average. This highlights the effectiveness of leveraging on-policy model generations for continuous updates of the policy model.

### 5.2 Transfer to Another Base Policy

In this section, we try to answer the question: Does the final student RM $\left(\mathcal{S}_{T}\right)$ help with the alignment of other base SFT models? Specifically, we experi- ment with a "Mistral-7B-SFT-Beta" (Tunstall et al., 2023) base policy model and compare the aligned model after one alignment iteration to Zephyr-7BBeta, SPIN ${ }^{3}$ (Chen et al., 2024), and a DPO baseline using the initial student $\mathrm{RM}\left(\mathcal{S}_{0}\right)$. All are based on the same Mistral (Jiang et al., 2023) backbone. Table 5 presents the win rate (\%) of various aligned policy models against the base "Mistral-7B-SFTBeta" model. Our method surpasses SPIN (two alignment iterations) by an average win rate of $3.28 \%$. SPIN is a strong self-evolution alignment method at the 7B scale, utilizing iterative supervised fine-tuning. The results demonstrate the effectiveness of DPO alignment with our student RM.

Additionally, our approach matches the performance of Zephyr-7B-Beta, a strong DPO-aligned model using $64 \mathrm{k}$ high-quality GPT-4 annotated preference data. Although our student RM is significantly smaller than GPT-4, it effectively leverages the distilled knowledge from the teacher model, enabling policy models to achieve comparable results. The performance of Zephyr-7B-Beta and our model complement each other, as each model excels on different datasets. This suggests a promising future exploration of combining offline with online preference data for policy model alignment.

Furthermore, we observe that the updated student RM outperforms the base student RM, indicating that the teacher's ranking capabilities have been effectively distilled into the student RM through our teacher-student collaborative mechanism. However, we also observe that DPO alignment with the initial student RM outperforms that with the final student RM on Harmless Base and Beavertails. This is because the initial student RM is trained on human data that includes both helpfulness and harmlessness preferences, while the teacher RM is optimized solely for helpfulness. Throughout the alignment iterations, the teacher's strengths in identifying helpful responses and its weaknesses in recognizing safe responses are gradually transferred to the students. Since helpfulness and harm-[^2]lessness are conflicting objectives, balancing them is outside the scope of this paper (Dai et al., 2023; Touvron et al., 2023). Future research may focus on better controlling the type of knowledge transferred from the teacher to the student. Nonetheless, the costs of maintaining the student RM in sync with the policy model are relatively low in TS-Align pipeline, and this efficient setup allows for scalable and continuous refinement of the policy models.

### 5.3 Performance of the Student

Table 6 shows the performance of various reward models on human preference test datasets. It is evident that the student RM's performance increasingly aligns with the teacher RM's after the iterative alignments, i.e., the performance of the student RM on the helpfulness preference datasets is increasingly better while that on harmless base is becoming worse. OpenAssistant's OASST Pythia and OASST DeBERTa reward models are fine-tuned using a diverse mixture of human-annotated preference data, including samples from the HH-RLHF training split, SHP training split, OpenAI's WebGPT (Nakano et al., 2021), and summarization comparisons (Stiennon et al., 2020b). The close average accuracy of our final student RM to both models indicates the effectiveness of our automatic preference data annotation pipeline.

Agreement with the Teacher To further validate the increasing agreement between the student RM and the teacher throughout our TS-Align pipeline, we compute the scores of $\mathcal{S}_{0}, \mathcal{S}_{1}, \mathcal{S}_{2}$, and $\mathcal{M}$ on three batches of on-policy data derived from $\pi_{0}, \pi_{1}$, and $\pi_{2}$ respectively. Here, $\pi_{0}$ represents the base policy "Mistral-7B-SFT-Beta" or "Alpaca-7B", $\pi_{1}$ is the policy model (iter1) with the teacher as the RM, and $\pi_{2}$ is the policy model (iter2) with the teacher as the RM. Each batch of on-policy preference data consists of approximately $30 \mathrm{~K}$ instruction prompts and a total of around $480 \mathrm{~K}$ candidate responses. The agreement between the students and the teacher is quantified using the Pearson correlation of their respective scores. We observe a clear increasing trend in the Pearson correlation coefficients for the base student $\left(\mathcal{S}_{0}\right)$, student iteration $1\left(\mathcal{S}_{1}\right)$, and student iteration $2\left(\mathcal{S}_{2}\right)$ with the teacher $(\mathcal{M})$, across different batches of on-policy data (generation from the base policy, policy iteration 1, and policy iteration 2), for both Mistral-7B-SFTBeta and Alpaca-7B as the base policy, suggesting the effectiveness of the student model in mimicking the teacher through the iterative alignment process.

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-07.jpg?height=736&width=748&top_left_y=326&top_left_x=1065)

Figure 2: Agreements between the teacher and students on various batches of on-policy data generated by policy models across different alignment iterations.

### 5.4 Additional Analysis

Human Evaluation Table 7 presents the pairwise human judgments on a randomly sampled subset of Alpaca-Eval and IFEval. The results show an increase in the win rate of policy models after the first and second alignment iterations using our TS-Align pipeline, which agrees with the GPT-4 judgments shown in Table 3 and validates the effectiveness of TS-Align. Additional analysis of the human evaluation is included in Appendix C.

Number of Sampled Responses We assess the alignment performance of the policy model with varying values of $K=\{2,4,8,16\}$ and conduct a single alignment iteration using the UltraRM-13B teacher as the reward model and Alpaca-7B as the base policy. The win rates of the aligned policy model compared to the base Alpaca-7B model on Alpaca-Eval, IFEval, Helpful Base, and Helpful Online are shown in Figure 3. Results for Helpful Rejection, Beavertails, and Harmless Base are detailed in Appendix E.1.

Generally, alignment performance improves with increasing $K$. A notable improvement is observed when $K$ increases from 8 to 16 across most datasets, supporting our chosen value of $K$ in prior experiments. Ideally, we should sample a highly diverse set of candidate responses, potentially setting $K>100$. Our teacher-student alignment pipeline will work much more efficiently than solely us-

|  | Harmless Base | Helpful Base | Helpful Online | Helpful Rejection | Beavertails | SHP | Alpaca-Farm | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OASST Pythia | 60.03 | 65.76 | 56.04 | 61.84 | 60.57 | 68.62 | 56.32 | 61.31 |
| OASST DeBERTa | 64.14 | 68.39 | 57.80 | 61.99 | 61.01 | 53.83 | 54.68 | 60.26 |
| UItraRM-13B (Teacher) | 39.40 | 71.79 | 62.20 | 67.08 | 64.05 | 71.57 | 61.65 | 62.53 |
| RoBERTa RM (Student Base) | 57.10 | 56.63 | 50.48 | 56.71 | 64.32 | 50.70 | 59.40 | 56.48 |
| RoBERTa RM (Student Iter1) | 54.89 | 61.43 | 53.57 | 61.73 | 65.56 | 55.87 | 61.48 | 59.97 |
| RoBERTa RM (Student Iter2) | 48.62 | 64.57 | 57.89 | 63.44 | 65.83 | 57.19 | 62.29 | 59.98 |

Table 6: Accuracy scores (\%) of different reward models on seven human preference test datasets.

|  | Alpaca-Eval |  |  |  | IFEval |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pairwise (\%) | Win | Tie | Loss |  | Win | Tie | Loss |
| Iter1 vs SFT | 61.50 | 3.50 | 35.00 |  | 56.50 | 2.00 | 41.50 |
| Iter2 vs SFT | 70.00 | 3.00 | 27.00 |  | 63.00 | 1.00 | 36.00 |

Table 7: Human evaluation of pairwise comparisons of TS-Algined policy models vs the base Alpaca-7B SFT model on subsets of Alpaca-Eval and IFEval.
![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=468&width=744&top_left_y=1028&top_left_x=244)

Figure 3: Win rates (\%) of different numbers of $\mathrm{K}$.

ing the teacher in such a scenario. However, due to limited computational resources, we defer this exploration to future work.

Size of On-Policy Data We assess the impact of the on-policy data size by conducting a single alignment iteration using the UltraRM-13B teacher as the reward model and Alpaca-7B as the base policy. We compute the win rates of the aligned model versus the base policy on Alpaca-Eval, Helpful Base, Helpful Online, and Beavertails. As shown in Table 4, performance generally improves with increasing size of on-policy preference data. The differences from $18 \mathrm{~K}$ to $30 \mathrm{~K}$ are not significant on most datasets, suggesting that further increasing the size of instruction data may not bring performance gain. Hence, our choice of $30 \mathrm{~K}$ instruction data is reasonable.

## 6 Related Work

Iterative LLM Alignment can be broadly divided into two main approaches: The first focuses on selfevolution without relying on an external reward

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=486&width=771&top_left_y=622&top_left_x=1059)

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=226&width=346&top_left_y=641&top_left_x=1072)

Helpful Online

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=186&width=340&top_left_y=901&top_left_x=1075)

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=232&width=356&top_left_y=635&top_left_x=1455)

Beavertails

![](https://cdn.mathpix.com/cropped/2024_06_04_97113f0bb7a37d05a72ag-08.jpg?height=200&width=353&top_left_y=894&top_left_x=1434)

Figure 4: Win rates (\%) of different on-policy data size.

model (Li et al., 2023a; Yuan et al., 2024; Chen et al., 2024). For example, Yuan et al. (2024) proposes self-rewarding language models, where the process begins by bootstrapping instructions from the policy model, which then creates candidate responses based on these instructions. The model employs "LLM-as-a-Judge" prompting (Zheng et al., 2023) to evaluate and reward its own outputs. This approach allows the model to align itself through directed preference optimization using the selfcurated data. Li et al. (2023a) introduces instruction back-translation. This involves using the policy model to generate new instructions from text spans within the Clueweb corpus. The model then produces responses given the newly generated instructions. The resulting instruction-response pairs serve as a basis for further fine-tuning the policy model, enhancing its alignment through continuous refinement. However, these approaches heavily rely on the scale of the LLMs as the "LLM-as-aJudge" may not work well on smaller language models. Additionally, the self-rewarding mechanism tends to bias towards their generations.

The second approach, in contrast, relies on an external reward model to guide the alignment process (Touvron et al., 2023; Xu et al., 2023b; Singh et al., 2023; Guo et al., 2024; Dong et al., 2024). Touvron et al. (2023) uses human annotations of policy generations during each alignment iteration and employs rejection sampling to guide
the policy model to produce human-favored outputs. The rest adopt a similar pipeline to ours, using an external reward model to annotate policy model generations and derive pseudo-labeled preference data for alignment.

The key difference between TS-Align and other approaches is the teacher-student collaboration mechanism, which enables reliable and efficient annotation of large-scale preference data for policy model alignment. Our approach is also more practically feasible under conditions of limited budget and resources.

Synthetic Preference Data Several recent approaches propose to curate preference data through AI feedback (Bai et al., 2022b; Lee et al., 2023; Pace et al., 2024; Guo et al., 2024), which is an efficient way to obtain large-scale preference data than using human annotators. Bai et al. (2022b); Lee et al. (2023); Guo et al. (2024) propose to annotate model generations by prompting large language models while Pace et al. (2024) relies on a semi-supervised self-training setup (Scudder, 1965). Kim et al. (2023) employs a series of heuristic rules to generate preference data for reinforcement learning. For example, one of their assumptions is that models with larger sizes typically yield better responses than their smaller counterparts. Yang et al. (2023) leverages contrasting positive and negative prompts to create high- and low-quality response pairs. Our method aligns with the approach of using on-policy model generations for preference data collection and employs an efficient and reliable teacher-student collaborative framework for annotations. Additionally, we focus on enhancing a small-scale student reward model by distilling the ranking capabilities of a strong teacher model into the student through iterative alignment.

## 7 Conclusion

In this paper, we introduce TS-Align, a teacherstudent collaborative framework designed to balance reliability and efficiency in the data labeling process for iterative fine-tuning of policy models. By leveraging the strengths of a large-scale teacher model without requiring it to process all candidates, TS-Align combines the efficiency of a smaller student reward model with the reliability of a robust teacher model. This iterative alignment process results in a highly aligned policy model with an impressive average win rate of $69.7 \%$ over the base policy, as judged by GPT-4. Human evaluations also confirm the effectiveness of TS-Align. Additionally, we demonstrate that the teacher's knowledge is effectively distilled into the student, and the final student reward model, after iterative alignment, can be transferred to align other base policy models.

## Limitation

The effectiveness of TS-Align relies on the quality and robustness of the teacher model. If the teacher model is not sufficiently strong, the knowledge distilled into the student model may be suboptimal, affecting the overall performance of the alignment process. Additionally, while our approach is efficient for the current scale of models used, its scalability to even larger models or more complex tasks remains to be validated. Lastly, the applicability and effectiveness of TS-Align across a wide range of domains and tasks also need further exploration. The current results are promising, but additional testing is required to ensure that the approach generalizes well to various types of data and instructions.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022b. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324345 .

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv: 2401.01335.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo $\mathrm{Xu}$, Mickel Liu, Yizhou Wang, and Yaodong Yang. 2023. Safe RLHF: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. 2024. RLHF workflow: From reward modeling to online rlhf. arXiv preprint arXiv: 2405.07863.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori Hashimoto. 2023. AlpacaFarm: A simulation framework for methods that learn from human feedback. In Thirty-seventh Conference on Neural Information Processing Systems.

Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. 2022. Understanding dataset difficulty with $\mathcal{V}$-usable information. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5988-6008. PMLR.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. KTO: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835-10866. PMLR.
Gemini Team Google, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. 2024. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790-2799. PMLR.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. 2023. BeaverTails: Towards improved safety alignment of llm via a humanpreference dataset. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. arXiv preprint arXiv: 2310.06825 .

Sungdong Kim, Sanghwan Bae, Jamin Shin, Soyoung Kang, Donghyun Kwak, Kang Yoo, and Minjoon Seo. 2023. Aligning large language models through synthetic feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13677-13700, Singapore. Association for Computational Linguistics.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:2219922213 .

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Minh Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Alexandrovich Glushkov, Arnav Varma Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu

Nguyen, and Alexander Julian Mattick. 2023. Openassistant conversations - democratizing large language model alignment. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2023. RLAIF: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv: 2309.00267.

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023a. Self-alignment with instruction backtranslation. arXiv preprint arXiv: 2308.06259.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b. AlpacaEval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval.

Hao Liu, Carmelo Sferrazza, and Pieter Abbeel. 2023. Chain of hindsight aligns language models with feedback. arXiv preprint arXiv: 2302.02676 .

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, and Adam Roberts. 2023. The flan collection: Designing data and methods for effective instruction tuning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 22631-22648. PMLR.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. SimPO: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730-27744. Curran Associates, Inc.

Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. 2024. West-of-n: Synthetic preference generation for improved reward modeling. arXiv preprint arXiv: 2401.12086.
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 487-503, Online. Association for Computational Linguistics.

Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, and Iryna Gurevych. 2020. AdapterHub: A framework for adapting transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 46-54, Online. Association for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, patrick gallinari, and Matthieu Cord. 2022. Diverse weight averaging for out-of-distribution generalization. In Advances in Neural Information Processing Systems.

Henry Scudder. 1965. Probability of error of some adaptive pattern-recognition machines. IEEE Transactions on Information Theory, 11(3):363-371.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. 2023. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. 2023. Preference ranking optimization for human alignment. arXiv preprint arXiv:2306.17492.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020a. Learning to summarize with human feedback. In $A d$ vances in Neural Information Processing Systems, volume 33, pages 3008-3021. Curran Associates, Inc.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020b. Learning to summarize with human feedback. In $A d$ vances in Neural Information Processing Systems, volume 33, pages 3008-3021. Curran Associates, Inc.

Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. 2024. Prefer-
ence fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An instruction-following LLaMa model. https:// github.com/tatsu-lab/stanford_alpaca.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. 2023. Zephyr: Direct distillation of $1 \mathrm{~m}$ alignment. arXiv preprint arXiv:2310.16944.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085-5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022a. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 23965-23998. PMLR.

Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. 2023a. Baize: An open-source chat model with parameter-efficient tuning on self-chat data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 62686278, Singapore. Association for Computational Linguistics.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. 2023b. Some things are more CRINGE than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv: 2312.16682.

Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. 2023. RLCD: Reinforcement learning from contrast distillation for language model alignment. arXiv preprint arXiv: 2307.12950.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-rewarding language models. arXiv preprint arXiv: 2401.10020 .

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023. RRHF: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv: 2304.05302 .

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, et al. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, and Yongqiang Ma. 2024. LlamaFactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv: Arxiv-1909.08593.
</end of paper 2>


<paper 3>
# TinyGSM: achieving $>80 \%$ on GSM8k with small language models 

Bingbin Liu ${ }^{1}$, Sebastien Bubeck ${ }^{2}$, Ronen Eldan ${ }^{2}$, Janardhan Kulkarni ${ }^{2}$,<br>Yuanzhi $\mathrm{Li}^{2}$, Anh Nguyen ${ }^{2}$, Rachel Ward ${ }^{2}$, Yi Zhang ${ }^{2}$<br>${ }^{1}$ Carnegie Mellon University<br>${ }^{2}$ Microsoft Research


#### Abstract

Small-scale models offer various computational advantages, and yet to which extent size is critical for problemsolving abilities remains an open question. Specifically for solving grade school math, the smallest model size so far required to break the $80 \%$ barrier on the GSM8K benchmark remains to be 34B. Our work studies how high-quality datasets may be the key for small language models to acquire mathematical reasoning. We introduce TinyGSM, a synthetic dataset of $12.3 \mathrm{M}$ grade school math problems paired with Python solutions, generated fully by GPT-3.5. After finetuning on TinyGSM, we find that a duo of a 1.3B generation model and a 1.3B verifier model can achieve $81.5 \%$ accuracy, outperforming existing models that are orders of magnitude larger. This also rivals the performance of the GPT-3.5 "teacher" model (77.4\%), from which our model's training data is generated. Our approach is simple and has two key components: 1) the high-quality dataset TinyGSM, 2) the use of a verifier, which selects the final outputs from multiple candidate generations.


## 1 Introduction

One fascinating phenomenon regarding large language models (LLMs) is the emergence of capbilities as both the model and dataset sizes scale up (Wei et al., 2022a; Chan et al., 2022). Among many capabilities, mathematical reasoning is one particular aspect that has received tremendous attention Lewkowycz et al. (2022); Lightman et al. (2023). However, it is unclear to what extend scale is a necessity for mathematical reasoning, and the potential of small language models (SLMs) remains largely under-explored.

In this work, we push the boundaries of SLMs' math reasoning capabilities. As a first step towards general mathematics, our primary testing ground is grade school math problems, to solve which require both mathematical understanding and language comprehension. The gold-standard benchmark in this regime is GSM8K (Cobbe et al., 2021), a collection of $8.8 \mathrm{~K}$ grade-school math word problems (with a $7 \mathrm{k}-1 \mathrm{k}$ train-test split) that involve 2 to 11 reasoning steps. GSM8K has been widely regarded to be challenging for LLMs. Even though the questions appear rather simple for humans, there have been few models that achieve $>80 \%$, and they are commonly of prohibitive sizes, e.g. 34B and above (see Table 1).

Our goal is to break the $80 \%$ barrier on GSM8K while keeping the model size friendly. As previous work shows (Gunasekar et al., 2023; Li et al., 2023; Eldan \& Li, 2023), training data quality is one of the most important factors for enhancing performance of small models. In particular, prompt-engineered synthetic data generation from gigantic models such as GPT-3.5/4 enjoys the clear advantage of desirable data hygiene and controllable diversity. This constituents a teacher-student scenario where the student learns from teacher's generations. On the tasks that the model model already excels at, their guided generations remain one of the highest quality data one can collect for training significantly smaller student models. It is also understood that the student model's performance likely ends up inferior than the teacher, and may fall far short especially when the student is considerably smaller than the teacher (Mirzadeh et al., 2019; Gudibande et al., 2023) —after all, the teacher places an information-theoretic bottleneck on the student.

To our surprise, in the case of GSM8K, we are able to bridge the performance gap between the student and teacher, by utilizing a tiny amount of labeled real data (the original GSM8K training set of $7 \mathrm{k}$ questions) to train an independent verifier model. At test time, the verifier score and select among multiple candidate answers generated from the student, and then we output the highest score generation as the final submission. Note the idea of using

![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-02.jpg?height=707&width=1396&top_left_y=186&top_left_x=359)

Figure 1: Our results on GSM8K. Please refer to Table 1 for details.

a verifier is proposed by the seminal GSM8K paper (Cobbe et al., 2021), and here we demonstrate its power of bridging the teacher-student gap, and we conduct a more thorough examination of factors affecting its efficacy.

The contributions of our work are the following:

- We introduce TinyGSM, a synthetic dataset containing GSM8K-style math word problems paired with Python solutions, generated fully by GPT-3.5-turbo. TinyGSM consists of $12.3 \mathrm{M}$ questions which amount to 1.8B tokens. We demonstrate TinyGSM's high-quality by finetuning the Phi-1.5 1.3B model (before the use of verifiers) which improves its accuracy from $44.6 \%$ to $\mathbf{6 8 . 2 \%}$ on the GSM8K test set. Notably, our smallest $125 \mathrm{M}$ model can also achieve $\mathbf{6 3 . 1 \%}$ after finetuning on TinyGSM.
- We demonstrate the power of verifiers on small-scale models. When integrated with a verifier for scoring generations, our models, named Phi-GSM models, achieve performance on par with other open source models that are orders of magnitude larger. In particular, our 1.3B model achieves $81.5 \%$ accuracy on GSM8K, as shown in Figure 1. This marks a new state-of-the-arts on billion-parameter-scale models, significantly outperforming existing open-source models and even rivaling the $77.4 \%$ accuracy of GPT-3.5, from which TinyGSM is generated. For verifier training, we identify data diversity as a crucial element for a verifier's success, and find that the scaling of the verifier may be more effective than scaling of the generator: while scaling up from a $125 \mathrm{M}$ generator to a $1.3 \mathrm{~B}$ generator only gives a $5.1 \%$ increase in performance (Table 1), scaling up the verifier from $125 \mathrm{M}$ to $1.3 \mathrm{~B}$ leads to a $7.2 \%$ performance boost (Figure 4 ).


## 2 Related works

Distilling from synthetic data: While scaling up has been a useful strategy, it is possible to outpace conventional scaling laws by better use of data (Sorscher et al., 2022). In the data-scarce case, quality synthetic data serves as an effective workaround (Eldan \& Li, 2023; Gunasekar et al., 2023), and the scaling in dataset size can compensate for a small model size (Edelman et al., 2023). Additionally, our work uses samples in the true distribution (i.e. the GSM8K train set) differently: given the small dataset size, we believe that the most sample-efficient way to utilize the true train set is to train a verifier-while the $7.4 \mathrm{k}$ samples in the GSM8K training set is too small for language model finetuning, it is sufficient for training a good quality verifier that provides $10 \%$ performance boost. While there have been potential concerns of learning from synthetic data such as loosing diversity or having a drifted distribution mean (Alemohammad et al., 2023; Shumailov et al., 2023), Alemohammad et al. (2023) showed that such degradation can be avoided by including fresh samples from the true distribution during training.

| $\overline{\text { Model }}$ | Base model | $\overline{\text { Model size }}$ | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=227&top_left_y=453&top_left_x=1207) | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=196&top_left_y=453&top_left_x=1454) | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=189&top_left_y=453&top_left_x=1673) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-2 (Touvron et al., 2023) | - | $\overline{7 \mathrm{~B}}$ | nlp | pass@1 | 14.6 |
|  |  | 13B |  |  | 28.7 |
|  |  | $34 \mathrm{~B}$ |  |  | 42.2 |
|  |  | 70B |  |  | 56.8 |
| MetaMath (Yu et al., 2023b) | Llama-2 | 7B | nlp | pass@1 | 66.5 |
|  |  | 13B |  |  | 72.3 |
|  |  | 70B |  |  | 82.3 |
| WizardMath (Luo et al., 2023) | Llama-2 | 7B | nlp | pass@1 | 54.9 |
|  |  | 13B |  |  | 63.9 |
|  |  | $70 \mathrm{~B}$ |  |  | 81.6 |
| MAmmoTH (Yue et al., 2023) | Code-Llama | 7B | code | pass@1 | 59.4 |
|  | Code-Llama | 12B |  |  | 64.7 |
|  | Code-Llama | $34 \mathrm{~B}$ |  |  | 72.7 |
|  | Llama-2 | 70B |  |  | 76.9 |
| Mistral (Jiang et al., 2023) | - | 7B | nlp | maj1@8 | 52.2 |
|  | - | $8 \times 7 \mathrm{~B}$ |  |  | 58.4 |
| OVM (Yu et al., 2023a) | Llama-2 | $7 \mathrm{~B}+7 \mathrm{~B}$ | nlp | verify100@1 | 73.7 |
|  | Mistral | $7 \mathrm{~B}+7 \mathrm{~B}$ |  |  | $84.7 \quad-1$ |
| Llemma (Azerbayev et al., 2023) | Llama-2 | 7B | nlp | pass@1 | 36.4 |
|  |  | $34 \mathrm{~B}$ |  |  | 51.5 |
| ToRA-Code (Gou et al., 2023) | Llama-2 | 7B |  |  | 72.6 |
|  |  | 13B | code. | COT@1 | 75.8 |
|  |  | $34 \mathrm{~B}$ | code |  | 80.7 |
|  |  | $70 \mathrm{~B}$ |  |  | 84.3 |
| Orca 2 (Mitra et al. 202.3) | Ulama-2 | 7B | nln | nacs@1 | 55.72 |
|  |  | 13B |  | passe 1 | 65.73 |
| Gemini Pro |  |  | $n \ln$. | moi1०20 | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=43&width=189&top_left_y=1504&top_left_x=1673) |
| Gemini Ultra (Gemini Team) | ${ }^{-}$ | ${ }^{-}$ | nIp | majल | 94.4 |
| GPT-3.5-0613 |  |  | ando | $\operatorname{mocol} 1$ | $77.4^{*}$ |
| GPT-4-0613 (OpenAI, 2023) | - | - | code | passલl 1 | $97.0^{*}$ |
| Phi-1.5 (Li et al., 2023) | - | $1.3 \mathrm{~B}$ | code | $\overline{\text { pass@1 }}$ | 44.6 |
|  | Phi-1.5-tiny | $125 \mathrm{M}$ |  |  | 63.1 |
| Phi-GSM | Phi-1.5-small | $350 \mathrm{M}$ | ordo | maca@1 | 65.9 |
|  | Phi-1.5 | $1.3 \mathrm{~B}$ | code | pass@ 1 | 68.2 |
|  | Phi-2 | $2.7 \mathrm{~B}$ |  |  | 74.3 |
|  | Phi-1.5-tiny | $125 \mathrm{M}+125 \mathrm{M}$ |  |  | 68.9 |
| Phi-GSM $+V$ | Phi-1.5-small | $350 \mathrm{M}+350 \mathrm{M}$ | code | verify48@1 | 71.3 |
|  | Phi-1.5 | $1.3 \mathrm{~B}+1.3 \mathrm{~B}$ |  |  | 81.5 |

Table 1: Results on GSM8K. * denotes results measured by ourselves. Accuracies above $\mathbf{8 0 \%}$ are labeled in bold. ' $8 \times 7 \mathrm{~B}$ ' stands for mixture of 8 experts, and each expert is of $7 \mathrm{~B}$ parameters. ' $7 \mathrm{~B}+7 \mathrm{~B}$ ' means a combination of a $7 \mathrm{~B}$ generation model plus a $7 \mathrm{~B}$ verifier model. ' $+\mathrm{V}$ ' denotes the use of verifier models.

Math word problem datasets GSM8K (Cobbe et al., 2021) has been the most common used math word problem dataset for its quality and size. In comparison, earlier datasets such as MAWPS (Koncel-Kedziorski et al., 2016), ASDiv (Miao et al., 2020) and SVAMP (Patel et al., 2021) are either much smaller in size or of less difficulty. However, GSM8K questions are too clean to test for robustness. Motivated by the observation that language models are not robust to the presence of irrelevant context, Shi et al. (2023a) proposed GSM-IC (irrelevant context). Another problem is the GSM8K dataset itself is still too small for training language models. (Ni et al., 2023b) addressed this with self-sampled data. In a work concurrent to ours, Yu et al. (2023b) bootstraps an original dataset using various augmentation techniques, such as generating multiple answers for the solution, question rephrasing, and backward reasoning. The proposed MetaMath dataset consists of 40000 questions from GSM8K and MATH (Hendrycks et al., 2021). In comparison, TinyGSM is significantly larger, encompassing 12.3M questions (or equivalently 1.8B tokens).

Leveraging multiple generations: An important component of our method is to leverage multiple generation. This idea has been proven successful in many prior works. A notable example is "self-consistency" (Wang et al., 2022), which selects the most frequent response among candidates and integrates well with other methods such as progressive-hint prompting (Zheng et al., 2023) and model selection (Zhao et al., 2023). However, self-consistency was not particularly helpful in our setup as mentioned in Section 4.2. More related to and a direct inspiration of our work is Cobbe et al. (2021), which uses a verifier to select the best response among 100 candidates, leading to an $20 \%$ accuracy boost. Our work conducts a more thorough study on the design choices of the verifier, including data diversity and the effect of verifier sizes. Another design choice orthogonal to ours is the supervision signals, such as outcome-based supervision versus process supervision (Lightman et al., 2023).

Learning from partial or process supervision: In our experiments, we evaluate on the final accuracy only but train on full programs. Prior work has studied the effect of process versus outcome based supervision. Process-based supervision is shown to be particularly helpful for complex math problems (Lightman et al., 2023), though for general problems one needs to consider a cost-efficacy tradeoff (Uesato et al., 2022). When process supervision is not available, $\mathrm{Ni}$ et al. (2023b) proposed to learn from "self-sampled" solutions, which allows the model to learn from partially correct self-generated solutions selected based on the execution trace.

Self-improvement: Several works have explored the idea of "self-improvement" where a model evaluates and corrects its own generations, mostly relying on the self-debugging ability of GPT4. Examples include "selfrefine" (Madaan et al., 2023) and "self-verify" (Weng et al., 2022; Zhou et al., 2023), both of which ask the model to iteratively provide feedback or verifications on its own generations and refine if needed. However, such self-improvement abilities have not been discovered in small language models. This motivated our use of a separate verifier model, which is initialized from the generative model but needs to be fully finetuned for verification.

Prompt-based methods: Prompt-based methods, which find prompts to improve the later conditional generations, are particularly effective for large models. Examples include in-context learning (Brown et al., 2020), where the model learns to perform novel tasks from few-shot examples provided in the prompt, as well as Chain-ofThought (Wei et al., 2022b), which shows that explicitly generating intermediate steps can significantly help with reasoning abilities. However, similar to self-improvements, prompting is targeted at large language models and do not apply for SLMs.

## 3 The TinyGSM dataset

Our objective is to assess the capability of a small language model (SLM) on mathematical reasoning. Ideally, enhancing this mathematical reasoning ability should not compromise the model's competence in language comprehension. This makes math word problems, which necessitate both mathematical and language understanding, a suitable test ground. We focus on the GSM8K dataset (Cobbe et al., 2021), consisting of around $8 \mathrm{k}$ grade-school math word problems. The math concepts in the dataset are elementary and within standard grade-school curricula, but the challenges posed by the natural language problem statement introduce an additional layer of complexity to the task.

TinyGSM: augmenting GSM8K with synthetic generations Despite the high quality, the GSM8K training set only contains 7473 problems, which is too small for training a reasonably sized language model (Ni et al., 2023a). To alleviate the size issue, we augment the GSM8K training set using GPT-3.5-turbo generated synthetic problems.

We prompt GPT-3.5-turbo to generate problem variants similar to a given question (but not the solution) randomly

```
def simple_math_problem() -> int:
    "n"
    In preparation for her party, Sarah buys 10 trays
    of food and 8 cases of beverages.
    Each tray costs $50 and each case of beverages
    costs $20.
    What is the total cost of the trays and beverages?
    n""
    trays = 10
    tray_cost = 50
    cases = 8
    case_cost = 20
    tray_total = trays * tray_cost
    case_total = cases * case_cost
    total_cost = tray_total + case_total
    result = total_cost
    return result
```

Figure 2: Examples from TinyGSM. The question is given as the docstring of a function, and the solution is the code in the function body.

sampled from the GSM8K training set. Each problem variant contains both a question and the corresponding solution written in Python, as shown in Figure 2. ${ }^{1}$ Using code allows us to leverage a Python interpreter, circumventing language models' known limitation regarding numerical calculations and code execution.

To enhance robustness, we also generated synthetic problems whose questions contain irrelevant information. This is achieved by augmenting the GSM-IC dataset (Shi et al., 2023a), which is an augmentation of GSM8K specifically designed to introduce irrelevant context (IC) to the question statement. These GSM-IC variants constitute to approximately one third of TinyGSM.

The resulting synthetic dataset contains $12.3 \mathrm{M}$ problems (i.e. question-solution pairs) ${ }^{2}$ with, based on the original $7.4 \mathrm{k}$ training set questions and their IC variants. For each question in the GSM8K train set, the prompt based on this question is shared across API calls, and the source of randomness comes entirely from the generation process. To encourage diversity, we use temperature sampling and specify in the prompt to encourage the problem variants to be grammatically diverse and contain multiple steps; the exact prompts are provided in Figure 3 and in Appendix A.1.

Filtering To ensure the quality of the synthetic data in TinyGSM, we filter out problems that are too short or do not contain numbers, as well as code solutions which are not executable. Note that we do not check for the correctness of the question or the generated solutions, since the "ground truth" solution is not available. Given the effectiveness of self-consistency (Wang et al., 2022), one might want to filter the problems by keeping the ones which have majority vote only. We did not adopt this strategy since we find that GPT-3.5-turbo's generations are only consistent on easy problems ${ }^{3}$, hence such consistency filtering will remove challenging problems, resulting in a dataset that is too easy to be useful.

## 4 Solving grade school math with small language models

The 1.3B version of our phi-GSM models is able to achieve $81.5 \%$ accuracy on GSM8K, a dataset that remains challenging for small-scale models. The performance comes from sufficient good quality synthetic data and the use[^0]

Consider the following grade-school math problem: \{\{question\}\}

Generate 10 different math problems similar to this math problem.

- Make sure each question uses diverse NLP and includes multiple logical steps.
- After each generated problem, write down a **detailed and complete Python program** to solve the question **step by step** (do NOT give the result directly, **DO NOT write calculations in the comments**).
- The program should contain multiple lines of code and end with 'result = XXX' (Make sure to replace XXX with the actual result of the python program).
- Make sure your Python program is complete and solves the problem. Do **NOT** write things like 'solution to be completed', result = ?, insert your code here etc.
- Give the complete solution to solve the problem, written in Python. Do not write things like 'insert your code here'.
- In each new question, **first end with <lendofquestion|>**, and then start writing the program. Each program should end with <lendofprogram|>.
- Example format: Question X: New question (at least 4 sentences long and use diverse NLP) (without the solution)

<|endofquestion|> Complete python code with entire solutions and the correct indent (<|endofprogram|>])

Figure 3: The prompt template for generating TinyGSM.

of a verifier, which we describe in this section.

### 4.1 Learning from synthetic data

We finetune the Phi-1.5 125M, 350M and 1.3B models on our TinyGSM from Section 3, and in particular, the 1.3B model reaches $\mathbf{6 8 . 2 \%}$ accuracy. ${ }^{4}{ }^{5}$ We use the Adam optimizer with FP16 during training, with a linear warm-up and a maximum learning rate of 1e-4, a weight decay of 0.01 , and an effective batch size of 1024 . The finetuning phase takes up to $20 \mathrm{k}$ steps in total. As shown in Figure 1, even without verifiers, our models are already competitive to models of size from 7B and larger. As an anecdote, an earlier and worse performing version of our Phi-GSM 1.3B model gets $94 \%$ (or $82.5 \%$ from $350 \mathrm{M}$ at pass $@ 32$, whereas the $750 \mathrm{M}$ CodeT5+ model (Wang et al., 2023) gets $73.8 \%$ (or $70.5 \%$ from $220 \mathrm{M}$ ) at pass $@ 100$.

### 4.2 Improving small models with a verifier

While sufficient synthetic data can significantly boost model performance, the performance is still below $70 \%$. Does further improvement necessitate larger model and more data then? There may be two concerns: First, there may be a diminishing return in adding extra parameters and data; for instance, while there is a $10 \%$ increase in performance when increasing from around one third of the final size of TinyGSM to two thirds, the final one third of the data provided only marginal gain. Moreover, even if the small language model is able to fully match the quality of the synthetic data, GPT-3.5-turbo itself can only achieves $77.4 \%$ test accuracy on GSM8K, which seemingly poses a limit on the performance of any models distilling from its generations.

In this section, we show that the use of a verifier can be an effective strategy orthogonal to introducing more and better data, and can even help SLMs exceed the accuracy of GPT-3.5-turbo generations. The main observation that the best of multiple generations significantly outperforms a single generation. These generations could be low-temperature generations from different checkpoints of a single run, where taking the best out of generations from 5 checkpoints of (an early version of) our 350M model reaches $75 \%$ accuracy, similar to findings in temporal ensembling (Laine \& Aila, 2016) and snapshot ensembles (Huang et al., 2017). ${ }^{6}$ The generations could also be from high-temperature generations based on a single checkpoint; for instance, the pass@32 accuracy of our 1.3B model is $94 \%$.

This suggests a promising direction of leveraging multiple generations: we can obtain a great performance boost if we are able to identify the best generation. This idea is effective yet natural: The probabilistic nature of the[^1]generative process naturally leads to the fact that multiple generations of a language model are more likely to contain a correct solution than a single one. Empirically, it has been widely observed that pass@ $k$ accuracy, namely, the accuracy by taking the best of $k$ generations, is often much higher than pass@1. The main challenge is that without knowing the labels, the definition of "best" is usually unclear. A workaround is to apply some form of self-selection, such as by choosing the one with the highest logit or the most consistent solution (Wang et al., 2022; Li et al., 2022). There is, however, a notable limitation: generations can be consistent and confident yet inaccurate, making the self-consistency approach through majority voting less effective (Li et al., 2022).

Given these observations and inspired by findings in (Cobbe et al., 2021), we propose to use a separate verifier for selecting candidate generations. For each base generation SLM, we train a verifier to predict whether a generation is a correct solution to the given question. During inference, we generate multiple candidate generations using temperature sampling, and select the one with the highest verifier score.

Training data The training data consists of the SLM's generations on the labele GSM8K training set questions, paired with the binary labels indicating whether a generation leads to the correct numerical answer. We sample 48 generations for each training set question. The binary label for each generation is based on the final execution result and the ground truth label only, and we do not verify the correctness of intermediate steps. Note that this is the only time where the GSM8K training set is directly utilized in training.

| Verfier model size | Base generation model size |  |  |
| :---: | :---: | :---: | :---: |
|  | $125 \mathrm{M}$ | $350 \mathrm{M}$ | $1.3 \mathrm{~B}$ |
| $125 \mathrm{M}$ | 68.9 | 68.8 | 71.7 |
| $350 \mathrm{M}$ | 67.3 | 71.3 | 78.3 |
| $1.3 \mathrm{~B}$ | 76.1 | 79.2 | $\mathbf{8 1 . 5}$ |

![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-07.jpg?height=268&width=830&top_left_y=1037&top_left_x=1081)

Figure 4: Pass@1 results on GSM8K test set with verifiers. For each test question, we sample 48 candidate answers from the base generation model, from which we submit the one with highest verifier score as the final answer. The verifier's score on a candidate answer is determined using its score on the last token.

Training setup The verifier is trained with a sequence-to-sequence task, where we use the binary label on the entire sequence to supervise each token. We find this approach improves consistently over training with a sequence classification task (i.e. only predicting a binary label on the entire sequence). The verifier model is initialized to be the same as the SLM, with an additional prediction head shared across all positions. All network parameters are updated during verifier training, which significantly outperforms alternatives where only the prediction head is updated, or where the network is trained from scratch.

Checkpoint and data diversity The diversity of the training data proves to be important, even if this may result in slightly sacrificing the data quality ${ }^{7}$ : we find that including data generated using various temperatures and different checkpoints improves the performance of the verifier. In particular, the training data for the verifier is generated from checkpoints at $1 \mathrm{k}, 6 \mathrm{k}$ and $12 \mathrm{k}$ steps, and both training and testing data use a mixture of data generated with temperature 0.5 and 0.7 . Compared to training with generations from a single checkpoint, the use of multiple checkpoints improves the performance from $78.9 \%$ to $81.5 \%$.

Note also that in general, while we train one verifier for each generative model, verifiers transfer reasonably well across models. For example, our 1.3B model was able to reach $78.1 \%$ with a $350 \mathrm{M}$ verifier trained on generations from a $350 \mathrm{M}$ model.[^2]

## Question:

Sophia is thinking of taking a road trip in her car, and would like to know how far she can drive on a single tank of gas. She has traveled 100 miles since last filling her tank, and she needed to put in 4 gallons of gas to fill it up again. The owner's manual for her car says that her tank holds 12 gallons of gas. How many miles can Sophia drive on a single tank of gas?

## Generations:

## 1) correct:

distance_traveled $=100$

gas_used $=4$

tank_size $=12$

miles_per_gallon= distance_traveled / gas_used

result = miles_per_gallon * tank_size

## 2) wrong:

```
distance_traveled = 100
gas_put_in = 4
gas_tank_size = 12
result = (distance_traveled
    + gas_put_in) / (gas_tank_size -
    gas_put_in)
```


## 3) correct:

```
miles_traveled = 100
gallons_added = 4
total_gallons = 12
miles_per_gallon = miles_traveled //
    gallons_added
total_miles = (total_gallons -
    gallons_added) * miles_per_gallon +
    miles_traveled
result = total_miles
```

Figure 5: Visualization of the verifier's token-level predictions. Text colors denote the verifier's prediction scores: correct, potentially correct, potentially wrong, and wrong. In all the three examples, the verifier's final prediction (on the last token) aligns with the groundtruth labels. In generation 1) and 2) the verifier's token-level scores appear to be interpretable and aligned with human assessment. However, in generation 3), the scores are rather strange. This suggests the verifier relies on special patterns of the model generations that may not be unversaly generalizable, even though its final predictions are fairly reliable.

Generation model size vs verifier size In Figure 4, we present results from a cross-examination of various generation model sizes + verifier model sizes. Interestingly, while the best accuracy is achieved with configuration with largest sizes, the verifier size seems to play a bigger role than the generation model size. The effect of model size scaling is surprisingly mild: as shown in Table 1, increasing the base generation model from 125M (Phi-1.5-tiny) to 1.3B (Phi-1.5) only gives a $6 \%$ boost. On the other hand, the verifier seems to be much more parameter efficient. For example, $125 \mathrm{M}$ generation model $+1.3 \mathrm{~B}$ verifier can achieve $76.1 \%$, while $1.3 \mathrm{~B}$ generation model $+125 \mathrm{M}$ verifier gets only $71.7 \%$ Figure 4 .

## 5 Robustness and decontamination

### 5.1 Contamination test

While we never use the GSM8K test set during training, TinyGSM consists entirely of synthetic data generated by GPT models, which may be contaminated since GPT-3.5-turbo may have been exposed to the test set during its own training, which would have led to some generated synthetic samples being replicating part of the test set. To prevent contamination, we decontaminate TinyGSM by checking for n-gram matches. We use $n=13$ following standard practices (Brown et al., 2020; Wei et al., 2021; Du et al., 2022), ${ }^{8}$ and remove punctuation and numbers before computing the matching. Out of the $11.0 \mathrm{M}$ unique synthetic questions ${ }^{9}, 22$ questions have a nonzero 13 -gram match with the test set, and $38 \mathrm{k}$ questions (i.e. around $0.35 \%$ of the full set) have non-zero 8 -gram matches. Examples of 13 -gram matches are provided in Appendix A.2.[^3]

### 5.2 Evaluation on SVAMP

| Verfier model size | Base generation model size |  |  |
| :---: | :---: | :---: | :---: |
|  | $125 \mathrm{M}$ | $350 \mathrm{M}$ | $1.3 \mathrm{~B}$ |
| $125 \mathrm{M}$ | 63.2 | 70.0 | 72.2 |
| $350 \mathrm{M}$ | 64.6 | 68.7 | 72.3 |
| $1.3 \mathrm{~B}$ | 74.1 | 79.0 | 75.6 |

Figure 6: SVAMP test accuracies.

For evaluating robustness of our models, we test on the SVAMP (Simple Variations on Arithmetic Math word Problems) dataset (Patel et al., 2021), consisting of 1000 math word problem questions with a focus on arithmetics. SVAMP constructed by applying certain types of variations to a set of base questions. Even though the base questions are generally considered easier than GSM8K ${ }^{10}$, the variations may often confuse LLMs, thus making it a challenging benchmark for robustness. Our 1.3B model achieves $75.6 \%$ on SVAMP without further finetuning, indicating the robustness of the model.

## 6 Discussions

In this work, we showed a simple approach that enabled a 1.3B generation model to achieve $81.5 \%$ on the GSM8K dataset, setting a new state-of-the-art for small language models and raising the performance curve for scaling. Our approach consists of two simple steps: 1) collecting TinyGSM, a GPT-3.5 generated synthetic dataset which we will fully release, and 2) using a verifier that scores how likely a generation is correct, whose quality is boosted by utilizing diverse generations. Our results provide positive evidence that small language models have more potentials to be unlock and can be used for efficient. For future directions,

- Leveraging different formats: TinyGSM uses Python code as solutions, inspired by the observation that language models tend to struggle at calculations. However, we found that different solution formats, i.e. code versus natural language, can be complementary: while code helps circumvent errors related to execution or calculation, it tends to perform worse at questions that require equation solving, likely due to the fact that the Python syntax does not naturally support equations. Properly combining both formats has the potential to further boost performance.
- The effect of verifier size: Our results show that given a budget on the model size, scaling the verifier may be a more efficient use of the parameters. This counters our intuition that verification is an easier task than generation (which involves search), though there might be connections to findings in GAN training where the size of discriminator (Arora et al., 2018). Exploring the parameter efficiency in a generation model versus a verifier is an interesting future direction.


## References

Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G. Baraniuk. Self-consuming generative models go mad. arXiv preprint arXiv: $2307.01850,2023$.

Sanjeev Arora, Andrej Risteski, and Yi Zhang. Do GANs learn the distribution? some theory and empirics. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id= BJehNfW0-.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv: $2310.10631,2023$.[^4]

Adam Block, Dylan J. Foster, Akshay Krishnamurthy, Max Simchowitz, and Cyril Zhang. Butterfly effects of sgd noise: Error amplification in behavior cloning and autoregression. arXiv preprint arXiv: $2310.11428,2023$.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract. $h t m l$.

Stephanie C. Y. Chan, Adam Santoro, Andrew Kyle Lampinen, Jane X. Wang, Aaditya K Singh, Pierre H. Richemond, J. Mcclelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. Neural Information Processing Systems, 2022. doi: 10.48550/arXiv.2205.05055.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv: Arxiv-2110.14168, 2021.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen S. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, ICML 2022, 17 -23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 5547-5569. PMLR, 2022. URL https://proceedings.mlr.press/v162/du22c.html.

Benjamin L. Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Pareto frontiers in neural feature learning: Data, compute, width, and luck. arXiv preprint arXiv: 2309.03800, 2023.

Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv: Arxiv-2305.07759, 2023.

Google Gemini Team. Gemini: A family of highly capable multimodal models.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv: 2309.17452, 2023.

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv preprint arXiv: 2305.15717, 2023.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need. arXiv preprint arXiv: $2306.11644,2023$.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS Datasets and Benchmarks, 2021.

Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, J. Hopcroft, and Kilian Q. Weinberger. Snapshot ensembles: Train 1, get m for free. International Conference on Learning Representations, 2017.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv: $2310.06825,2023$.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies, pp. 1152-1157, 2016.

S. Laine and Timo Aila. Temporal ensembling for semi-supervised learning. International Conference on Learning Representations, 2016.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier. arXiv preprint arXiv: 2206.02336, 2022.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step, 2023.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv: 2308.09583, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv: 2303.17651, 2023 .

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. Annual Meeting of the Association for Computational Linguistics, 2020. doi: $10.18653 / \mathrm{v} 1 / 2020$. acl-main. 92.

Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. arXiv preprint arXiv: 1902.03393, 2019.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason, 2023.

Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partially-correct solutions. In The Eleventh International Conference on Learning Representations, 2023a. URL https://openreview.net/forum? id=4D4TSJE6-K.

Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partially-correct solutions. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview. net/forum? id=4D4TSJE6-K.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv: $2203.13474,2022$.

OpenAI. Gpt-4 technical report, 2023.

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori B. Hashimoto. Proving test set contamination in black box language models. arXiv preprint arXiv: 2310.17623, 2023.

Arkil Patel, S. Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? North American Chapter Of The Association For Computational Linguistics, 2021. doi: 10.18653/V1/2021. NAACL-MAIN. 168.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, E. Chi, Nathanael Scharli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. International Conference on Machine Learning, 2023a. doi: $10.48550 /$ arXiv. 2302.00093.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. arXiv preprint arXiv: 2310.16789, 2023b.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The curse of recursion: Training on generated data makes models forget. arXiv preprint arXiv: 2305.17493, 2023.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523-19536, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: $2307.09288,2023$.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv: 2211.14275, 2022.

Xuezhi Wang, Jason Wei, D. Schuurmans, Quoc Le, E. Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. International Conference on Learning Representations, 2022. doi: $10.48550 /$ arXiv. 2203.11171.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D. Q. Bui, Junnan Li, and Steven C. H. Hoi. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv: 2305.07922, 2023.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, A. Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. International Conference on Learning Representations, 2021.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, E. Chi, Tatsunori Hashimoto, Oriol Vinyals, P. Liang, J. Dean, and W. Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022a. doi: 10.48550/arXiv.2206.07682.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, E. Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. Neural Information Processing Systems, 2022b.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. arXiv preprint arXiv: 2212.09561, 2022.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Kamalika Chaudhuri,

Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 23965-23998. PMLR, 2022. URL https://proceedings.mlr.press/v162/wortsman22a. html.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. Decomposition enhances reasoning via self-evaluation guided decoding. arXiv preprint arXiv: 2305.00633, 2023.

Fei Yu, Anningzhe Gao, and Benyou Wang. Outcome-supervised verifiers for planning in mathematical reasoning. arXiv preprint arXiv: 2311.09724, 2023a.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv: $2309.12284,2023 \mathrm{~b}$.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv: 2309.05653, 2023.

Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Qizhe Xie. Automatic model selection with large language models for reasoning. arXiv preprint arXiv: 2305.14333, 2023.

Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. ARXIV.ORG, 2023. doi: 10.48550/arXiv.2304.09797.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv: 2308.07921, 2023.
</end of paper 3>


<paper 4>
## WEAK-TO-STRONG GENERALIZATION: ELICITING STRONG CAPABILITIES WITH WEAK SUPERVISION

Collin Burns* Pavel Izmailov* Jan Hendrik Kirchner* Bowen Baker* Leo Gao*

Leopold Aschenbrenner* Yining Chen Adrien Ecoffet* Manas Joglekar*

Jan Leike Ilya Sutskever Jeff Wu*

$$
\mathrm{O p e n A I}
$$

## ABSTRACT

Widely used alignment techniques, such as reinforcement learning from human feedback (RLHF), rely on the ability of humans to supervise model behavior- -for example, to evaluate whether a model faithfully followed instructions or generated safe outputs. However, future superhuman models will behave in complex ways too difficult for humans to reliably evaluate; humans will only be able to weakly supervise superhuman models. We study an analogy to this problem: can weak model supervision elicit the full capabilities of a much stronger model? We test this using a range of pretrained language models in the GPT-4 family on natural language processing (NLP), chess, and reward modeling tasks. We find that when we naively finetune strong pretrained models on labels generated by a weak model, they consistently perform better than their weak supervisors, a phenomenon we call weak-to-strong generalization. However, we are still far from recovering the full capabilities of strong models with naive finetuning alone, suggesting that techniques like RLHF may scale poorly to superhuman models without further work. We find that simple methods can often significantly improve weak-to-strong generalization: for example, when finetuning GPT-4 with a GPT-2-level supervisor and an auxiliary confidence loss, we can recover close to GPT-3.5-level performance on NLP tasks. Our results suggest that it is feasible to make empirical progress today on a fundamental challenge of aligning superhuman models.

## 1INTRODUCTION

We mainly steer or align today's models with reinforcement learning from human feedback (RLHF): we reinforce behaviors that human evaluators rate highly and penalize behaviors that evaluators rate poorly (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Glaese et al., 2022; Bai et al., 2022a). This procedure is very effective when human evaluators can tell if model behavior is good or bad and is a core part of training modern language model assistants such as ChatGPT.

However, superhuman models will be capable of complex and creative behaviors that humans cannot fully understand.. For example, if a superhuman assistant model generates a million lines of extremely complicated code, humans will not be able to provide reliable supervision for key alignment-relevant tasks, including: whether the code follows the user's intentions, whether the assistant model answers questions about the code honestly, whether the code is safe or dangerous to execute, and so on. As a result, if we finetune a superhuman model with human supervision on a reward modeling (RM) or safety classification task, it is unclear how that model will generalize to complicated behaviors that humans could not reliably supervise themselves.

This leads to a fundamental technical challenge of aligning superhuman models (superalignment): how can weak supervisors control models much smarter than them? Despite the importance of Traditional ML Superalignment our Analogy

![](figures/1-0-FIGURE.jpg)

Figure 1: An illustration of our methodology. Traditional ML focuses on the setting where humans supervise models that are weaker than humans. For the ultimate superalignment problem, humans will have to supervise models much smarter than them. We study an analogous problem today: using weak models to supervise strong models.

this problem, it is difficult to empirically study today. Most prior work on alignment has either confronted this core challenge head-on--but been restricted to primarily theoretical frameworks and toy problems (Irving et al., 2018; Christiano et al., 2018; Leike et al., 2018; Demski & Garrabrant, 2019; Hubinger et al., 2019), or empirically studied humans supervising today's models--without addressing the core challenges that may arise with superhuman models (Christiano et al., 2017; Wu et al., 2021; Ouyang et al., 2022; Bowman et al., 2022; Saunders et al., 2022). In contrast, we would ideally like to have a setup that captures core challenges of aligning future superhuman models while also being able to make iterative empirical progress today.

We propose a simple setup for studying the problem of humans supervising superhuman models by considering an analogy: can we use weak models to supervise strong models? We can empirically test this by finetuning large (strong) pretrained models on labels generated by small (weak) models and observing how they generalize. Just like the problem of humans supervising superhuman models, our setup is an instance of what we call the weak-to-strong learning problem.

Why should weak-to-strong learning be possible? On the one hand, the strong model could simply learn to imitate the weak supervisor, including its errors, since that is what we would naively train it to do. On the other hand, strong pretrained models should already have good representations of the alignment-relevant tasks we care about. For example, if a model can generate complicated code, then it should intuitively also know whether that code faithfully adheres to the user's instructions. As a result, for the purposes of alignment we do not need the weak supervisor to teach the strong model new capabilities; instead, we simply need the weak supervisor to elicit what the strong model already knows. This gives us hope that the strong model can generalize beyond the weak supervision, solving even hard problems for which the weak supervisor can only give incomplete or flawed training labels. We call this phenomenon weak-to-strong generalization.

We study our weak-to-strong learning setup (Section 3) by finetuning base (i.e. pretrained-only) language models from the GPT-4 family (OpenAI, 2023), spanning 7 orders of magnitude (OOMs) of pretraining compute, across three settings: a large set of popular natural language processing (NLP) benchmarks, chess puzzles, and our internal ChatGPT reward modeling dataset. Our main findings include:

lThese models share the same general architecture and pretraining datast as GPT-4. However, this model series does not include the models known as GPT-2, GPT-3, and GPT-3.5.

tnlit atan-

![](figures/2-0-FIGURE.jpg)

Figure 2: Strong models trained with weak supervision generalize beyond their supervisor, and improving weak-to-strong generalization is tractable. We show test accuracy on a representative NLP task (left), chess puzzles (middle) and the ChatGPT reward modeling task (right). We show the weak supervisor trained on ground truth labels (light grey) and the strong student trained with weak supervision naively (green), with the best method in each setting (purple), or with ground truth supervision (dark grey). For NLP and chess we supervise GPT-4 using GPT-2-level supervision, while for reward modeling we supervise a 3.5-level model using GPT-2-level supervision. The best method is the auxiliary confidence loss for the NLP task (Section 4.3.2), bootstrapping for Chess puzzles (Section 4.3.1), and unsupervised generative finetuning for reward modeling (Section 5.2.2; generative-finetuning is also used for the strong ceiling performance).

l. Strong pretrained models naturally generalize beyond their weak supervisors. If we naively finetune strong models with labels generated by weak models, they consistently outperform their weak supervisors (Section 4.2). For example, on NLP tasks, if we fine-tune GPT-4 with labels from a GPT-2-level model, we typically recover about half of the performance gap between the two models.

2. Naively finetuning on weak supervison is not enough. Despite positive weak-to-strong generalization, there still remains a substantial gap between strong models finetuned with weak supervision and strong models finetuned with ground truth supervision. Weak-to-strong generalization is particularly poor for ChatGPT reward modeling. Collectively, our results provide empirical evidence that naive RLHF will likely scale poorly to superhuman models without additional work.

3. Improving weak-to-strong generalization is tractable. We find that we can improve performance by encouraging strong models to have confident predictions with an auxiliary loss, bootstrapping supervision with intermediate models, and improving model representations with unsupervised finetuning. For example, when supervising GPT-4 with a GPT-2-level model on NLP tasks using the auxiliary confidence loss, we typically recover nearly 80% of the performance gap between the weak and strong models.

Our work has important limitations. None of our methods work consistently in all settings, and especially in the RM setting we are still far from recovering the full performance gap between weak and strong models. Thus our methods serve more as proofs-of-concept that weak-to-strong generalization is tractable, rather than practical solutions we recommend deploying today. Furthermore, there are still important disanalogies between our empirical setup and aligning superhuman models that we did not address (Section 6); continuously refining our basic setup will be important for ensuring that research today continues to make real progress toward aligning the superhuman models we develop in the future.

Despite the limitations of our work, we find our results to be highly encouraging. We show that substantial weak-to-strong generalization is not only possible, but actually a widespread phenomenon. We also show that with very simple methods, we can drastically improve the ability of weak supervisors to elicit knowledge from strong models. With much more progress in this direction, we could get to the point where we can use weak supervisors to reliably elicit knowledge from much stronger models, at least for some key tasks that we care about. This may allow us to develop superhuman reward models or safety classifiers, which we could in turn use to align superhuman models.

Aligning superhuman models is essential for making them safe; there is increasing recognition that failing to align such powerful models has the potential to be catastrophic, making this one of the most important unsolved technical problems in the world (CAIS). We think it is now more tractable than ever to make rapid iterative empirical progress toward solving this problem.

## 2 RELATED WORK

We study how we can leverage the generalization properties of deep neural networks to solve weak-to-strong learning. Our problem setting and methods are closely connected to many existing research areas.

Weakly-supervised learning. Weak-to-strong learning is a special type of weakly supervised learning--a setting in which models are trained using unreliable labels (Bach et al., 2017; Rat-ner et al., 2017; Guo et al., 2018). There is also a rich literature on the related problem of learning from noisy labels (Song et al., 2022). Common methods include bootstrapping (Reed et al., 2014; Han et al., 2018; Li et al., 2020), noise-robust losses (Zhang & Sabuncu, 2018; Hendrycks et al., 2018; Ma et al., 2020), and noise modeling (Yi & Wu, 2019). Unlike most work on label noise, the errors in our weak supervision are much harder to address than uniform label noise, instead having "instance-dependent'" errors (Frénay & Verleysen, 2013). Semi-supervised learning, in which labels are only available for a subset of the data, is also closely related (Kingma et al., 2014; Laine & Aila, 2016; Berthelot et al., 2019). We could also study our problem in a semi-supervised setting by having an "easy" subset of examples that weak supervisors provide reliable labels for and a subset of unlabeled hard" examples that the weak supervisor can't reliably label, a problem which we call "easy-to-hard generalization'" (see Appendix C).

Student-teacher training. The framework of first training a teacher and then training a student on teacher's pseudo-labels is widely used in semi-supervised learning (Laine & Aila, 2016; Tarvainen & Valpola, 2017; Xie et al., 2020), domain adaptation (French et al., 2017; Shu et al., 2018), and knowledge distillation (Hinton et al., 2015; Gou et al., 2021; Stanton et al., 2021 ; Beyer et al., 2022). In contrast to most prior work, we focus on the setting where the student is much more capable than the teacher.

Furlanelo et al. (2018) and Xie et al. (2020) also consider cases where the student is at least as capable as the teacher. However in their settings the student is randomly initialized and has access to ground truth labels. Moreover, compared to most past work we are focused on qualitatively very weak supervision. For example, we are interested in huge leaps in generalization, similar to going from "3rd grade-level'' supervisors to "12th grade-level'' student models. Despite these differences with past work, we expect many methods from semi-supervised learning and domain adaptation to translate to our setting. For example, we found that a type of confidence auxiliary loss similar to past work (Grandvalet & Bengio, 2004) improves weak-to-strong generalization in Section 4.3.

Robustness of pretraining and finetuning. Many papers have shown that pretraining on massive, diverse data leads to more robust representations that generalize better out-of-distribution (Hendrycks et al., 2019; 2020b; Radford et al., 2021; Liu et al., 2022). Finetuning typically improves in-distribution generalization, but often performs poorly out-of-distribution, sometimes even degrading performance relative to zero-shot prompting (Kumar et al., 2022; Wortsman et al., 2022b; Awadalla et al., 2022). Recent approaches to mitigating this problem include weight ensembling (Wortsman et al., 2022b;a), finetuning only a subset of layers (Kirichenko et al., 2023, Lee et al., 2022a), or mitigating the distortion effects that finetuning has on pretrained features (Ku-mar et al., 2022). We did not find strong results in preliminary explorations of approaches similar to these (Appendix B), but we expect that with more thorough explorations one may be able to attain much stronger results with these or other ideas from the robust finetuning literature.

Debiasing. In weak-to-strong generalization, the weak labels contain a specific form of bias, which results from the weak models' lack of capability. There is a substantial literature on learning from biased training data (Bellamy et al., 2018). However, most work focuses on known biases, for example where we know that the models perform worse on minority groups. For known biases, common methods include Group Distributionally Robust Optimization (Sagawa et al., 2019), adversarial training (Zhang et al., 2018), and model editing (Santurkar et al., 2021; Meng et al., 2022). In contrast, our setting can be viewed as a particularly difficult debiasing problem where the bias is unknown. Some methods that automatically discover and mitigate biases include clustering (Sohoni et al., 2020), 1oss variance reduction (Khani et al., 2019), and auditing and re-training on high-loss group (Kim et al., 2019; Liu et al., 2021).

Imitation and preference learning. The goal of alignment is to steer already-capable models to do what we want them to do. For example, the base GPT-4 model is good at generating text following its pretraining distribution, but does not readily follow instructions. To align pretrained language models today, we finetune them using imitation learning on human demonstrations (Bain & Sammut, 1995; Atkeson & Schaal, 1997) or by using methods such as reinforcement learning from human feedback (RLIE) (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Glaese et al., 2022; Bai et al., 2022a). Constitutional AI (Bai et al., 2022b; Lee et al., 2023) leverages AI feedback to align language models, but still uses an initial RLHF phase. However, both imitation learning and preference learning assume high-quality human supervision, making it unclear if they will work for superhuman models.

Scalable oversight. Scalable oversight techniques aim to improve the ability of humans to supervise models. For example, humans may ask models to critique the outputs of other models (Irving et al., 2018; Saunders et al., 2022) or use models to help decompose a problem into simpler sub-problems (Leike et al., 2018; Christiano et al., 2018; Lightman et al., 2023). Scalable oversight methods typically take advantage of special problem structure, like decomposability or the fact that evaluation is easier than generation. In contrast to improving human supervision, we focus on generalizing beyond human supervision such that models perform well even in settings we cannot reliably supervise. That said, our weak-to-strong learning setup can be used to compare scalable oversight methods, generalization-based methods, and more. Our setup also resembles a proposal for measuring progress on scalable oversight known as ""sandwiching', which uses weak and strong humans (Cotra, 2021; Bowman, 2022).

Knowledge elicitation and honesty. Christiano et al. (2022) introduced a theoretical problem called Eliciting Latent Knowledge (ELK), in which the goal is to elicit latent knowledge from a superhuman machine learning model even under worst case assumptions. For example, a special case of ELK is honesty (Evans et al., 2021), where the goal is for the models to report their true beliefs?. Wentworth (2020) hypothesizes a tendency for neural networks to develop ""natural abstractions" that are easier to elicit. Recent empirical work on ELK includes a benchmark for measurement tampering (Roger et al., 2023), methods for discovering latent knowledge (Burns et al., 2023), and studies of honesty (Li et al., 2023; Pacchiardi et al., 2023). Our setting can be viewed as a general methodology for empirically studying problems like ELK and honesty across a wide range of tasks.

## 3 METHODOLOGY

A core challenge of superalignment is that humans will need to supervise models much smarter than us. This is a special case of what we call the weak-to-strong learning problem: how can a weak supervisor oversee a model much smarter than it? In this paper, we study a simple analogy, in which we replace the weak human supervisor with a weak model supervisor.

For a given task of interest, consisting of a dataset and a performance metric, we:

l. Create the weak supervisor. Throughout most of this work, we create weak supervisors by finetuning small pretrained models on ground truth labels.3 We call the performance of the weak supervisor the weak performance, and we generate weak labels by taking the weak model's predictions on a held-out set of examples.

2. Train a strong student model with weak supervision. We finetune a strong model with the generated weak labels. We call this model the strong student model and its resulting performance the weak-to-strong performance.

3. Train a strong model with ground truth labels as a ceiling. Finally, for comparison, we finetune a strong model with ground truth labels. We call this model's resulting performance the strong ceiling performance. Intuitively, this should correspond to "everything the strong model knows.," i.e. the strong model applying its full capabilities to the task..

For more details on how we train each model, see Appendix A.

Typically, weak-to-strong performance will be between weak performance and strong ceiling performance. We define the performance gap recovered (PGR) as a function of the above three performances (weak, weak-to-strong, and strong ceiling) as shown in the illustration below.

$$
\mathrm{P G R}={\frac{\mathrm{w e a k-t o-s t r o n g ~-~ w e a k}} {\mathrm{~ s t r o n g ~ c e i l i n g ~-~ w e a k}}} ~={\frac{\mathrm{~ \Lambda~}} {\ldots\ldots}}
$$

performance performance
weak wealk-to-strong strong ceiling
performance

PGR measures the fraction of the performance gap (the difference in performance between the weak and strong ceiling models) that we can recover with weak supervision. If we achieve perfect weak-to-strong generalization, PGR is 1. If the weak-to-strong model does no better than the weak supervisor, then PGR is 0.

Advantages. Our setup has a number of advantages, including:

1. It can be studied with any pair of weak and strong models, making it easy to study scaling laws and not requiring access to expensive state-of-the-art models. Moreover, it does not require working with humans, so feedback loops are fast.

2.. It can be studied for any task of interest, making it easy to empirically test across a wide range of settings.

3. Success will be practically useful even before we develop superhuman models: for example, if we find ways to align GPT-4 with only weak human supervision or with only GPT-3-level supervision, that would make it more convenient to align models today.

Limitations. Our setup still has important disanalogies to the ultimate problem of aligning superhuman models. We view our setup as removing one of the main disanalogies in prior work, not as providing a final, perfectly analogous setup. Two remaining disanalogies include:

1. Imitation saliency. Future superhuman models will likely have salient representations of human behaviors, but our strong models may not have learned features relevant for imitating weak model predictions; simply imitating the weak supervisor may thus be an easier failure mode to avoid in our setting than it will be in the future. More generally, the types of errors weak models make today may be different from the types of errors humans will make when attempting to supervise superhuman models.

2. Pretraining leakage. Our pretraining data implicitly contains supervision from humans. It may thus be artificially easy to elicit strong models' capabilities in our setting, since they were directly pretrained to observe strong (human-level) performance. Superhuman-level performance may not be directly observed in the same way- -superhuman knowledge might be more latent, e.g. because it was learned from self-supervised learning- -and thus might be harder to elicit from superhuman models in the future.

More generally, we do not yet know how superhuman models will be built, but they could develop new inductive biases that are qualitatively different from today's models. We view iterating on our methodology to produce even more analogous setups as a key priority for future work, as we discuss in more detail in Section 6.

## 4MAIN RESULTS

In this section, we report our main empirical results, including baselines and promising methods.

## 4.1 TASKS

Popular natural language processing benchmarks. We consider 22 popular NLP classification datasets covering ethics, commonsense reasoning, natural language inference, sentiment analysis, and other domains. We convert all datasets to binary classification tasks and approximately balance the classes. We produce soft labels from the weak model. See a full list of the datasets and their sources in Table 1.

Chess puzzles. We use the dataset originally introduced in Schwarzschild et al. (2021b), which contains chess puzzles from the lichess . org website (Lichess Team, 2023). Each puzzle consists of a chess position, and a sequence of optimal moves to play to solve the puzzle. For our evaluation, we predict the first move played, which is the best move in the given chess position. We illustrate the data format in Appendix Figure 14. For weak labels, we sample from the weak model with temperature 0. Note that unlike the other binary classification tasks we study in this paper, this is a generative task.

ChatGPT reward modeling. The standard approach to aligning models today is reinforcement learning from human feedback (RLHF). A critical step of RLHF is to train a reward model (RM) to predict human preferences between model responses. Specifically, a reward model is trained on a dataset consisting of dialogs between a human and an assistant model. For each query, the humans compare multiple possible responses (completions) from the assistant, providing human preference data. Then, a reward model is trained to predict the results of pairwise comparisons between completions. Finally, the assistant model is trained by optimizing against the reward model with reinforcement learning (RL). In our work, we do not study the RL step, and instead assume the goal is to maximize reward model accuracy. For more details on reward models, see e.g. Ouyang et al. (2022). We use a proprietary dataset used to train ChatGPT reward models.

For more details about our tasks and setup, see Appendix A.

## 4.2 NAIVELY FINETUNING ON WEAK LABELS

In each of these 3 settings (NLP tasks, chess puzzles, and reward modeling) we evaluate how well strong students generalize when naively finetuned on labels generated by weak supervisors. We study pretrained language models from the GPT-4 family (OpenAI, 2023), which allow us to study student-supervisor compute disparities of many orders of magnitude. We find that PGRs are almost universally positive--in virtually all settings that we studied, and across almost all student and supervisor sizes, students outperform their supervisors (Figure 3).

On the popular NLP benchmarks, we find especially promising weak-to-strong generalization: strong models trained with weak supervision can often generalize to a substantially higher performance than the weak model itself. Even with very weak supervisors and strong models with many orders of magnitude more compute, we recover more than 20% of the performance gap. The PGR increases both with weak supervisor size and with strong student size; for the largest students, the PGR is often above 50%.

We see more mixed results in the chess puzzle setting. In particular, when using the smallest weak models, the PGR is close to zero and the test accuracy curves appear flat. However, as the size of the weak supervisor increases, the PGR increases substantially; for small supervisor-student gaps, PGR can be above 40%. Unlike in the NLP setting, where PGR improves with the strong student size, PGR decreases with the strong student size for a given weak supervisor on chess puzzles. The cor-strong ceiling performance weak-to-strong performance (g.t. supervision) (weak supervision)

![](figures/7-0-FIGURE.jpg)

rigure 3: rromisng weak-to-strong generallzation witn nave nnetunng on INLr tasks and chess, but poor generalization on the ChatGPT reward modeling task. (a,b,c) Test accuracy as a function of strong student size on (a) NLP tasks, (b) chess puzzles, and (c) the ChatGPT reward modeling task. Accuracy of strong students trained with ground truth in black, accuracy of strong students trained with weak supervision shown with colored lines (hue indicates size of weak supervisor). (d,e,f) Same as panels a,b,c but for performance gap recovered (see Section 3 for details). For NLP settings, we compute the median across tasks (see Figure 12 for full details). We find decent weak-to-strong generalization and even positive PGR scaling on NLP tasks, decent generalization for small supervisor-student gaps but negative PGR scaling on chess puzzles, and both poor generalization and scaling for ChatGPT reward modeling.

responding test accuracy curves appear concave, potentially exhibiting inverse scaling (McKenzie et al., 2023) in strong student size.

Finally, we find that weak-to-strong generalization is poor by default in the ChatGPT reward model setting. We are usually only able to recover roughly 10% of the performance gap between the weak supervisor and the strong student. Even for relatively small gaps in compute between the weak and strong models, PGR almost never exceeds 20%.

In general, across all our settings, we observe weak-to-strong generalization: strong students consistently outperform their weak supervisors. It is not obvious why this should happen at all--especially from naive finetuning alone--and it gives us hope that weak-to-strong learning is a tractable problem. At the same time, our results suggest that naively using weak, human-level supervision will be insufficient to align strong, superhuman models; we will need qualitatively new techniques to solve superalignment.

## 4.3 IMPROVING WEAK-TO-STRONG GENERALIZATION IS TRACTABLE

We now show that we can use simple methods to substantially improve weak-to-strong generalization. While none of the methods we test works universally, these methods are proofs-of-concept that across many different tasks we can substantially improve generalization.

## 4.3.1 BOOTSTRAPPING WITH INTERMEDIATE MODEL SIZES

Bootstrapping is a long-standing idea in alignment: instead of directly aligning very superhuman models, we could first align an only slightly superhuman model, use that to align an even smarter model, and so on (Christiano, 2019; 2018; Leike & Sutskever, 2023; Worley, 2021). Our setting allows us to empirically test this idea.

wah.tn.otrn with hnn

![](figures/8-0-FIGURE.jpg)

$$
\begin{array} {l l l l l l l l l l l l l} {} & {} & {\mid\mathrm{\tiny~ \begin{array} {l l l l l l l l l l l l l l l} {{~}} & {~} & {~} & {~} & {.} & {~} & {.} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} \\ {~} & {~} & {~} & {~} & {.} & {~} & {~} & {.} & {~} & {~} & {~} & {~} & {~} & {~} \\ {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {.} & {~} & {~} & {~} & {~} & {~} \\ {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} \\ {.} & {~.} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} \\ {.} & {.} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~} & {~ ~} \\ \end{array}
$$

Higure 4: bootstrapping mproves weak-to-strong generalzation on cness puzzles. (a) lest accuracy as a function of strong student size. Accuracy of students trained with ground truth in black, accuracy of students naively trained with weak supervision shown with dotted lines (hue indicates size of weak supervisor). Accuracies of students trained via bootstrapping shown with colored squares (including both the final weak-to-strong performance and the performance of the intermediate models during bootstrapping). (b) Same as a with PGR. By taking multiple small steps instead of one big step we see substantially improved generalization, especially for larger student models.

Specifically, we can construct a sequence of model sizes $\mathcal{M}_{1} \to\mathcal{M}_{2} \to\ldots\to\mathcal{M}_{n}$ of increasing sizes. Then, we use the weak labels from $\mathcal{M}_{1}$ to finetune /l2, use /M2 to generate new weak labels that we can use to finetune the next model in the sequence, M3, and so on.

We evaluate bootstrapping in the chess puzzle etting. When we naively finetune on weak labels for chess (Section 4.2), we see high PGR when we cross small supervisor-student gaps, but low PGR for larger gaps. As a result, in this setting it may help to take multiple small steps -steps where PGR should be high-instead of one big step.

For each round of bootstrapping, we run three iterations of weak-to-strong learning, i.c. we bootstrap the weak supervision using two intermediate model sizes before finally finetuning the largest model in the sequence. We report the results (including all intermediate weak-to-strong models within each bootstrap) in Figure 4. Bootstrapping improves PGR compared to the baseline, especially for larger student models. With the naive method, transfer accuracy curves flatten as the weak-strong gap grows larger; with bootstrapping, the accuracy continues to monotonically improve.

While the results in the chess setting are promising, in preliminary experiments we observed only small improvements with bootstrapping on NLP tasks and no improvements in the RM setting. This makes sense intuitively: unlike in the chess setting where naive PGR decreased with larger supervisor-student gaps, naive PGR increased or was rougly constant for larger supervisor-student gaps in the NLP and reward modeling settings. Overall, these results suggest bootstrapping is a plausible avenue to investigate for improving weak-to-strong generalization and can be helpful in some settings, but that naive bootstrapping alone will not be enough to align models much smarter than their supervisors.

## 4.3.2 AN AUXILIARY CONFIDENCE LOSS CAN DRAMATICALLY IMPROVE GENERALIZATION ON NLP TASKS

In our baseline results (Section 4.2), we naively finetune the strong student on the labels provided by the weak supervisor. Because we are directly training the strong student to imitate the weak supervisor, it may also learn to imitate the errors of the supervisor (see Section 5.1 for more discussion). Intuitively, we want to avoid this failure mode and provide additional regularization towards what the strong pretrained model already internally knows: we want the student to learn the intent of the supervisor, but not to imitate its mistakes.

waak-tn-strnn narformanra $----$ with aliv Inee -A

![](figures/9-0-FIGURE.jpg)

Figure S: Substantially improved generalization on NLP datasets with a simple auxiliary loss

(a) Test accuracy as a function of strong student size. Accuracy of a student trained with ground truth in black, accuracy of students naively trained with weak supervision shown with dotted lines. Accuracies of students trained with auxiliary confidence loss shown with colored triangles. Median computed across 22 NLP tasks (hue indicates size of weak supervisor), see Figure 6 for individual datasets. (b) Same as a with PGR. The confidence loss can improve generalization drastically, especially for large supervisor-student gaps.

We operationalize this intuittion by adding an auxiliary confidence loss term to the standard cross entropy objective. This method is closely related to conditional entropy minimization (Grandvalet & Bengio, 2004) which is a prominent technique in semi-supervised learning. Specifically, we add an additional loss term which reinforces the strong model's confidence in its own predictions-even when they disagree with the weak labels. We provide a detailed description of the method in Appendix A.4.

In Figure 5, we plot accuracy and PGR curves with this method on our NLP tasks. We find that while it performs slightly worse than the naive baseline for smaller strong students, it dramatically improves generalization for large gaps in compute between weak and strong models. With the smallest weak supervisor and largest strong student, the confidence loss increases median PGR from about 25% to nearly 80%.

In addition, we also plot generalization curves for a representative subset of NLP datasets in Figure 6, as well as the full panel of datasets in Figure 12. There are some settings in which the confidence loss does not help much or degrades performance, e.g. when the gap between the weak supervisor and strong student is small or when the dataset features inverse scaling even with ground truth supervision. But the confidence loss improves performance on most NLP datasets dramatically, and for many datasets we get almost perfect generalization, recovering nearly all the performance of the strong model, even when using the smallest weak supervisors.

Finally, we find evidence consistent with our motivating intuition for the confidence loss (allowing the strong student to confidently disagree with its weak supervisor): the auxiliary loss reduces the strong student's imitation of weak errors and mitigates weak label overfitting (see Section 5.1).

## 5 UNDERSTANDING WEAK-TO-STRONG GENERALIZATION

Strong methods will be essential for solving superalignment, but to trust those methods it is also important to understand when and why they work. A better understanding of weak-to-strong generalization could help us trust that generalization will continue working even in the future high-stakes settings we care most about, and could help us develop better methods along the way. In this section, we study two phenomena relevant to weak-to-strong generalization: imitation of supervisor mistakes and salience of the tasks to the strong student model.

![](figures/10-0-FIGURE.jpg)

Figure 6: Simple auxiliary loss improves generalization across most datasets. Test accuracy as a function of strong student compute for a representative sample of NLP tasks. See Table 1 for dataset details and Appendix Figure 12 for results on all 22 NLP tasks. Auxiliary loss is shown with triangles, and the baseline with dotted lines. Weak supervisor model size shown in varying colors, with ground truth supervision shown in black.

## 5.1 UNDERSTANDING IMITATION

When we train a strong model with weak supervision on some task, our hope is that the strong model will perform that desired task as well as possible, leveraging the latent capabilities it learned from pretraining to significantly outperform the weak supervisor. A salient way in which we could fail to achieve that desired generalization is if the strong model instead learns to imitate the weak supervisor- -predicting how the weak supervisor would have classified each example. In particular, if the weak labels contain systematic errors that are easy to learn, the strong model could learn to imitate those errors. This is also a concern raised in theoretical work on superalignment, which has argued that the human simulator failure mode could be important: naive human supervision might result in superhuman models learning to imitate what a human would say, rather outputting its best predictions (Christiano et al., 2022).

## 5.1.1  OVERFITTING TO WEAK SUPERVISION

The failure mode of imitating weak supervision is especially relevant to our naive baseline in Section 4.2, which directly trains the student to imitate the supervisor. In the case of infinite training data, naively fitting the weak labels should result in perfect imitation, and a PGR of zero. In practice, we train on finite data for a small number of epochs. Unlike typical ML settings, however, we could expect to observe overfitting even when training for less than a single epoch: the strong model might overfit to the weak supervisor labels and its errors, degrading ground truth test accuracy over training even without classic overfitting to any specific training examples.

Empirically, we see that the strong student indeed appears to overfit to the weak supervisor's errors. In Figure 7(a) we show ground truth test accuracy curves over the course of training for the ChatGPT RM task, and in Figure 7(b) and (c) we compare the best" and final ground truth test accuracies (median across all weak-strong model pairs). We find overfitting for large weak-strong gaps. For small weak-strong gaps, weak-to-strong performance typically monotonically increases over the course of training. For larger gaps, weak-to-strong performance often increases initially, but then starts dropping well before a single epoch has elapsed. Ground truth early stopping, which "cheats" weakto-strona

![](figures/11-0-FIGURE.jpg)

Figure 7: Strong models overfit to the weak labels. In all figures, we show data for the ChatGPT Reward Modeling task. (a) Weak-to-strong performance over the course of training. Hues indicate the student-supervisor gap. (b) Best weak-to-strong performance during training (stars) and weak-to-strong performance at the end of training (dashed). Weak performance in black. Hue indicates the size of the weak supervisor. (c) Median best and final performance gap recovered (PGR) aggregated across all supervisor-student pairs. We see overitting to weak labels for large weak-strong gaps, even within one epoch. In these cases, the best test accuracy achieved over training can be substantially better than the test accuracy at the end of training. See Figure 13 for the corresponding analysis of a representative subset of NLP tasks.

by evaluating against ground truth and stopping at an optimal step with respect to ground truth test labels, typically gives a PGR improvement of around 5 percentage points.

We see the same phenomenon for NLP tasks in Figure 13. In the NLP setting, we find that t"cheating" early stopping on ground truth gives a 15 percentage point boost in PGR over the model at the end of training, and a 10 percentage point boost in PGR compared to "non-cheating'" carly stopping with respect to weak labels.

Unfortunately, an early stopping criterion that uses ground truth labels does not constitute a valid method. Nevertheless, the results above suggest that imitating weak supervisor errors may be an important phenomenon in our setting.

Moreover, these results suggest that better early stopping or regularization strategies may be able to substantially improve weak-to-strong generalization, by reducing overfitting to the weak labels and their errors. Indeed, we see in Figure 13 that the auxiliary confidence loss introduced in Section 4.3.2 reduces overfitting to weak labels on NLP tasks substantially. For large weak-strong gaps, early stopping on ground truth (compared to early stopping on weak labels) gives a 15% PGR boost when using the naive method, but only a roughly 5% PGR boost when using the confidence loss.

## 5.1.2 STUDENT-SUPERVISOR AGREEMENT

Another way to measure imitation is to directly measure the agreement between the student and the supervisor: the fraction of test inputs where the strong student makes the same prediction as the weak supervisor. Note that if agreement were 100%, then weak-to-strong accuracy would be equal to supervisor accuracy, and PGR would be 0.

In general, we notice that for our naive finetuning baseline, student-supervisor agreement is consistently high--often noticeably higher than weak supervisor accuracy. This indicates that the student is imitating some of the supervisor's errors. These phenomena hold across all tasks (NLP tasks, chess, and reward modeling) and all model sizes, for the naive method.

The confidence loss in Section 4.3.2 reduces student-supervisor agreements significantly (Figure 8), primarily by imitating supervisor mistakes less (Figure 8c). The loss encourages the strong student to make confident predictions, including when they contradict the weak supervisor. In a handful of the settings where it is most successful, the confidence loss reduces student-supervisor agreement

![](figures/12-0-FIGURE.jpg)

1gure 8: Student-supervisor agreement aecreases wiltn larger student-supervisor gaps; tne confidence loss reduces imitation of supervisor mistakes. (a) Student-supervisor agreement as a function of strong student size on NLP tasks, (b) a but only on samples where the supervisor is correct, (c) a but only on samples where the supervisor is mistaken. Dotted lines indicate naive finetuning on weak labels, and triangles indicate results with the auxiliary confidence loss results (see Section 4.3). Hue of line indicates size of weak supervisor. For results on reward models, see Figure 16.

below strong student test accuracy (weak-to-strong performance) - i., the resulting model is fitting the ground truth concept better than it is fitting the weak labels it was trained with.

## 5.1.3 INVERSE SCALING FOR IMITATING THE SUPERVISOR

Next, we study student-supervisor agreement as a function strong model size (see Figure 8 and Figure 16). Surprisingly, we find inverse scaling (McKenzie et al., 2023): larger student models consistently agree less with the errors of the supervisor than smaller student models, despite being trained to imitate the supervisor, not using early stopping, and having larger capacity than smaller student models.

This trend is especially strong if we evaluate agreement only on datapoints where the supervisor is wrong (Figure 8c), and the trend persists if looking at cross entropy loss instead of accuracy.

These results suggest that pretrained models may have a hard time fitting errors of other (smaller) pretrained models, at least in finetuning settings with relatively limited data. Stanton et al. (2021) and Furlanello et al. (2018) report a related observation in the context of knowledge distillation: it is surprisingly hard for models to fit the predictions of other models, even when they have sufficient capacity to do so.

One natural hypothesis is that the nature of (especially naive) weak-to-strong generalization depends heavily on the error structure of the weak supervisors and how easy those errors are to imitate. In Appendix E, we show initial experiments that test how different types of weak supervision errors impact what the strong student learns. Our results suggest that errors that are more difficult for the student to imitate result in stronger naive weak-to-strong generalization, but that even when they are easy to imitate, the confidence loss can help.

## 5.2 SALIENCY IN THE STRONG MODEL REPRESENTATIONS

One intuition for when weak-to-strong generalization might be feasible is when the task or concept we want to elicit is internally "salient'" to the strong model. In this section, we study several phenomena related to the saliency of the concepts we are trying to elicit from the student model.

## 5.2.1 ELICITING STRONG MODEL KNOWLEDGE WITH PROMPTING

One possible reason for the high PGR we observe in Section 4 could be that eliciting what the strong model knows is easy. In particular, it is possible that strong pretrained models can solve many relevant tasks zero-shot with a simple prompt.

In Figure 9a, we consider 7 representative NLP tasks and compare finetuning, zero-shot prompting, and 5-shot prompting; for this initial experiment, we use ground truth labels rather than weak labels fewshot a.t. fewshot.

![](figures/13-0-FIGURE.jpg)

Fieure 9: Few-shot prompting becomes competitive with finetunine for laree models: weak-to-

 $\begin{array} {c c c c c c c c c c c} {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & {\cdots} & \end{array}$ 
str qualitatively similar in the prompting setting. (a) Average zero-shot (single dashed), 5-shot (double dashed) and finetuning (solid) accuracy with ground truth labels as a function of strong student size. (b) Average 5-shot with weak labels (colored dashed) accuracy as a function of student model size. Hue of line indicates size of weak supervisor. Zero-shot and 5-shot same as in panel a. (c) Average weak-to-strong performance for 5-shot prompting (dashed with crosses), naive finetuning (dashed thin) and finetuning with the confidence loss (solid with triangle) as a function of student model compute. Results are averaged across 7 NLP tasks. Few-shot weak-to-strong performance becomes competitive with or outperforms finetuning for the largest strong students, though finetuning with the confidence loss does better.

for finetuning and 5-shot. For both the zero-shot and 5-shot baseline we use task-specific prompts summarized in Table 2. We find that zero-shot and 5-shot test accuracy is poor for most model sizes but, consistent with Brown et al. (2020), improves drastically for larger model sizes. In particular, for the largest models, 5-shot prompting becomes competitive with finetuning on many tasks, indicating that eliciting the task-relevant knowledge of these very large models is relatively straightforward.

We are also interested in weak-to-strong learning in the context of few-shot prompting. To study this setting, we construct a few-shot prompt where the labels are provided by the weak supervisor. We report the results in Figure 9b. Consistent with our findings in the finetuning setting, we get worse performance when we few-shot prompt with weak labels than we do few-shot prompting with ground truth labels. This suggests that weak-to-strong learning is a nontrivial problem in the prompting setting as well.

Similar to the finetuning setting, few-shot weak-to-strong performance improves for stronger supervisors. Compared to our weak-to-strong finetuning baseline (Figure 9c), weak-to-strong performance of few-shot prompting is poor for smaller student models, but becomes competitive or even outperforms finetuning for the largest strong students. However, weak-to-strong finetuning with the confidence loss stil enrally outperforms weak-to-strong few-shot prompting.

Overall, these results provide an important reference for our results on weak-to-strong generalization. They suggest that for the largest model sizes, the knowledge needed to solve many task can be elicited fairly easily with prompting. However, our current setup may be more disanalogous for prompting than for finetuning; many of our NLP tasks may have been implicitly observed during pretraining, which we conjecture benefits prompting more than finetuning. We discuss this potential disanalogy much more in Section 6.1.

## 5.2.2 GENERATIVE SUPERVISION IMPROVES RM WEAK-TO-STRONG GENERALIZATION

If salient representations of the desired task is useful for weak-to-strong generalization, then we may be able to improve generalization by increasing the salience of the task to the strong model. One way to increase the salience of a task without needing ground truth labels is to perform unsupervised finetuning with the language modeling objective on data relevant to that task (Dai & Le, 2015). For example, by finetuning a language model in an unsupervised way on online reviews, sentiment becomes saliently represented to models internally (Radford et al., 2017).

![](figures/14-0-FIGURE.jpg)

Figure 10: Generative finetunine on reward modeling data improves weak-to-strone perfor-

mance and PCR. (a) Weak-to-strong performance on the reward modeling task, with (solid lines) and without (dashed lines) an extra step of generative finetuning for the strong student model. Solid black line shows a strong ceiling reward model that was also trained with the generative finetuning step; dashed black line show a weak supervisor reward model trained without the generative fine-tuning step. (b) PGR with and without generative finetuning. For generative finetuning PGR, we use the strong ceiling performance that also had this extra generative finetuning step. Even with this ceiling adjustment, PGR is higher with an extra generative finetuning step.

We test this idea in our reward modeling setting, where it is standard practice to initialize the model with a baseline finetuned on demonstrations of desired behaviors (Stiennon et al., 2020). In our case, we re-use the ChatGPT comparison data instead of introducing a new supervision dataset. Comparisons are comprised of a prefix (a single request or conversation between the user and assistant) and at least two candidate completions. We finetune the base models with a language modeling loss on all prefix-completion pairs, ignoring the human preferences between those completions.

Note that these pairs include completions ranked worst by human raters, so this procedure should not in principle leak any information about the ground truth preference labels that the weak-to-strong models should not have access to. On the other hand, since the completions can come from humans or stronger models, there may be some leakage similar in kind to the pretraining leakage that we discuss as a disanalogy in Section 6.1. Even in this setup, the reward modeling task is highly nontrivial, and we leave addressing this disanalogy (e.g. by collecting completions only from weaker models) for future work.

We found that the additional generative finetuning on the RM data leads to better weak-to-strong performance. Because this procedure also improves the performance of models trained on ground truth RM data, we compare our new weak-to-strong performance to strong "ceiling' models that were also first generatively finetuned in the same way. Even with this adjusted ceiling, we find that generative supervision improves PGR by approximately 10-20%. We report the results in Figure 10.

Furthermore, the improvement from generative finetuning stacks with the improvement from ground truth early-stopping (a cheating" method to illustrate potential performance if we could optimally early stop, see Section 5.1.1). When we combine these two techniques, we can achieve PGR of approximately 30-409%, which would make the results on the RM task competitive with the weak-to-strong generalization we observe on NLP and chess puzzle tasks.

We can apply the idea of fimproving task saliency with generative finetuning on relevant data to all settings, and we believe this could be a promising direction for future work.

## 5.2.3 FINETUNING ON WEAK SUPERVISION TO INCREASE CONCEPT SALIENCY

One possible measure of concept saliency is how linearly represented a task is. In particular, we can measure the performance of a linear probe (logistic regression classifier) trained from frozen activations of the model. If the optimal solution can be approximately recovered with a linear probe, that

![](figures/15-0-FIGURE.jpg)

lesl dllUIdty (o)

Figure 11: Finetuning on weak supervisor labels makes the desired generalization more linearly represented. We plot test accuracy for five different strategies, averaged across a subset of NLP tasks. Ip(weak): training a linear probe on the base model using weak labels, lp(gt): training a linear probe on the base models using ground truth labels, ft(weak): finetuning the model on weak labels, ft(weak) + lp(gt): finetuning the model on weak labels then training a linear probe on ground truth labels, ft(gt): finetuning the model on ground truth labels. Finetuning on the weak labels significantly increases the linearity of the ground truth concept.

could simplify our problem greatly; we could focus on linear probing methods instead of finetuning methods, which could greatly reduce the search space we need to consider to elicit the desired generalization. In our work, we focus only on how linearly represented a task is in the final activations, prior to the unembedding layer.

In Figure 11, we plot average test accuracy on a subset of our NLP datasets for several different combinations of (1) finetuning or linear probing, using (2) weak or ground truth labels. First, we show linear probes trained with ground truth labels (72% accuracy on average) perform worse than finetuning with ground truth labels (829% on average), indicating that the optimal solution to most tasks is not represented completely linearly in the strong model's final activations. For comparison, we also report the results for linear probing and finetuning using weak labels, which we verify are worse than using ground-truth labels.

However, we find that we can achieve substantially better performance by first finetuning the model on the weak labels, and then linear probing using the ground truth labels. In other words, when we finetune the strong model with weak labels, the representations become more linear even with respect to ground truth labels. In fact, finetuning on weak labels then linear probing on ground truth labels results in an accuracy of 78%%, closing 60% of the gap between ground truth linear probing and finetuning. This also noticeably outperforms the naive weak-to-strong finetuning baseline.

This phenomenon is closely related to a recent finding reported by Kirichenko et al. (2023) in the spurious cues literature. They find that finetuning a model on biased supervision can result in models with very biased outputs, but surprisingly strong linear representations of the desired concepts. These results suggest an alternative approach to improving weak-to-strong generalization. We could first "linearize" the desired concept, e.g. by naively finetuning on weak labels. Then we could use simpler linear probe-based weak-to-strong methods to elicit the desired concept.

## 6DISCUSSION

In this paper, we proposed a simple analogy for studying a core challenge of aligning superhuman models and showed that it is feasible to make significant progress on this problem. However, our setup still has important disanalogies, which we now elaborate on. We then outline a number of promising avenues for future work.

## 6.1 REMAINING DISANALOGIES

Imitation saliency: superhuman models may easily imitate weak errors. Future models wil likely be very good at predicting what humans will think and say, especially if they are trained on human data in a similar manner to current models. Consequently, if we naively train such a superhuman model with human supervision, it might simply imitate the weak supervisor, outputting human-level capabilities rather than its latent superhuman capabilities (Christiano et a., 2022).

This problem is only partially captured by our setup. While our strong pretrained models do imitate weak supervisors to some extent, they are not explicitly pretrained to imitate weak models, and our results from Section 5.1.3 suggest that larger strong models may even have more difficulty doing this imitation. As such, "imitating the weak supervisor'" may not be as much of a problem in our setup as it will be for the ultimate superalignment problem. This may inflate generalization performance today. We believe a more thorough investigation of this problem is an important area for future work.

Pretraining leakage: superhuman knowledge may be latent, not observable. Many of the tasks we consider in this work may have been observed in pretraining at least indirectly, for example through questions on online forums or through slight reframings of the task. For example, it is highly likely that simple science questions similar to those in the SciQ NLP task are present in our GPT-4 series pretraining dataset at least implicitly in some form. However future superhuman models may never directly observe superhuman alignment-relevant capabilities; these capabilities may be predominantly "latent", e.g. learned through self-supervised learning or reinforcement learning rather than through imitation learning. Intuitively, latent capabilities may be harder to elicit than capabilities that models could have observed in their pretraining data.

This disanalogy could cause our results to be overly optimistic. We conjecture that this disanalogy also increases prompting performance (Section 5.2.1) more than it increases finetuning performance; intuitively prompting may work especially well on tasks that the model assigns high probability to observing. If so, this would make prompting more disanalogous in our setup than finetuning. We hope to test this conjecture in future work.

In Appendix D.1, we show a proof of concept that weak-to-strong generalization can still elicit latent capabilities that were never explicitly observed during pretraining, and even when prompting is not possible. In particular, we use AlexNet (Krizhevsky et al., 2012) to supervise models pretrained with DINO (Caron et al., 2021), a self-supervised method in computer vision that learns strong representations. We find that the strong student generalizes significantly beyond AlexNet's performance, even though the student never observed any classification labels during pretraining. Future work should study and mitigate this pretraining leakage disanology more systematically.

## 6.2 FUTURE WORK

What would convince us that we have a "solution' to superalignment? This is a complicated question and we do not claim to have a complete answer. However, we expect substantial progress in at least the following three areas will be necessary: analogous setups, scalable methods, and strong scientific understanding. We now sketch out concrete problems for each of these areas.

## 6.2.1 CONCRETE PROBLEMS: ANALOGOUS SETUPS

Having strong measurements and a reliable methodology is extremely important for making empirical progress in any field. In particular, it is important that we have metrics which provide strong signal about whether we are making real progress toward the problem we ultimately care about. Important directions for follow-up work include:

. Making our setup more analogous by fixing the main remaining disanalogies described in Section 6.1. Analogous setups are essential to ensure that methods that work today will continue to work for superhuman models.

. Validating that disanalogies are not severe, for example by checking that results are qualitatively similar to using e.g. 3rd grade humans to supervise our strongest models today.

. Relaxing some of the simplifications we made, e.g. by generalizing our methods and results to complicated generative tasks.

. Testing how robust our weak-to-strong classifiers are to optimization pressure when we attain high PGR; for example, if we attain good weak-to-strong generalization with RMs, can we optimize the learned RM using RL?

. Testing our conjecture that prompting-based methods in our current setup will not be as indicative of future results relative to finetuning-based methods (Section 5.2.1), and improving our setup to fix this.

. Identifying new or more specific disanalogies with our setup and fixing them.

Additionally, we do not yet know what future models will look like. We should update our setup over time as we learn more about how broadly superhuman models will be built.

## 6.2.2 CONCRETE PROBLEMS: SCALABLE METHODS

One intuition for why major progress on weak-to-strong generalization seems possible is because all we need to do is extract everything the strong model already "knows" about the task of interest-the strong model should intuitively already understand the task, and should hopefully have salient representations of that task. This suggests a number of properties that should be satisfied by the desired generalization, and which we may be able to measure without access to ground truth.

. The desired generalization should be able to disagree with the weak supervision when the weak supervision is wrong. This is a property our auxiliary confidence loss may capture.

. The desired generalization should be "natural'" or "salient" to the model. For example, we should not need to change the model too much to elicit the desired concept.

. The desired generalization should be consistent. Consistency properties range anywhere from basic logical consistency to complicated forms of consistency between many prompts (e.g. cycle consistency, cross examination, etc.).

Future work should identify additional unsupervised properties that can be used to specify the desired generalization. More generally, there are very likely existing methods in the machine learning literature (e.g. in semi-supervised learning or robust finetuning), which would be natural to try and which could also lead to substantial gains in weak-to-strong generalization. Generalization-based approaches to weak-to-strong learning are complementary to scalable oversight methods, in which the weak supervisor interacts with the strong model to improve the quality of the weak supervision.

6.2.3 CONCRETE PROBLEMS: SCIENTIFIC UNDERSTANDING

We will need an extremely high degree of trust and reliability in our methods for aligning superhuman models in high-stakes settings. We will not get this from strong benchmark performance alone. Instead, we also need a thorough understanding of precisely when and why our methods work. Example questions of interest include:

. What explains the difference between the relatively strong results on NLP datasets and the relatively poor results with reward models when using naive finetuning?

. What makes a concept easy or hard to elicit? What is a good definiton of 'salience"?

. Can we reliably estimate generalization error at test time without any labels? For example. can we measure the degree of weak-to-strong underspecification (Lee et al., 2022b)?

. Can we reliably extrapolate generalization error across many orders of magnitude using scaling laws?

. How important are the errors in the weak supervision, precisely? How do different kinds of weak label biases affect generalization?

. How robust are our proposed methods to optimization pressure?

In Section 5 we only scratched the surface for understanding weak-to-strong generalization, but future work will need to go much further. An advantage of our setup is that it makes it easy to run simple experiments to scientifically study generalization phenomena across a wide range of settings.

## 6.3 CONCLUSION

Recent progress in AI has been faster than almost anyone anticipated (Steinhardt, 2022; Bengio et al.). For an increasing number of researchers, the possibility of superhuman models being developed this decade has become increasingly plausible. Broadly superhuman models would be extraordinarily powerful and, if misused or misaligned with humans values, could potentially cause catastrophic harm (CAIS). Given the stakes, we need to establish extremely high reliability in the alignment of these systems ahead of time. But for years it has been unclear how to empirically study superhuman model alignment. We believe it is now easier to make progress on this problem than ever before.

## 7 ACKNOWLEDGEMENTS

We would like to thank Boaz Barak, Paul Christiano, Jacob Steinhardt, Ananya Kumar, Jakub Pa-chocki, John Schulman, Wojciech Zaremba, Alec Radford, Nat McAleese, and William Saunders for valuable technical insights and discussions. We are grateful to Mia Glaese, Boaz Barak, Kush Bhatia, Jean-Stanislas Denain, Erik Jones, Polina Kirichenko, Daniel Kokotajlo, Yoonho Lee, Jessy Lin, Richard Ngo, John Schulman, Peter Tong, Fred Zhang, Ruigi Zhong, Ryan Greenblatt, Fabien Roger, Paul Christiano, Steven Adler, Rai Pokorny, Adam Kalai, Jacob Hilton, Roger Grosse, Dan Hendrycks, Alec Radford, and Scott Aaronson for helpful feedback on earlier drafts of this paper. We also thank Shantanu Jain, Avital Oliver, Suchir Balaji, Cathy Yeh, and the Platform team for infrastructure help. CB is also grateful to Dan Hendrycks, Jacob Steinhardt, and Paul Christiano for many formative discussions over the years.

## REFERENCES

Eric Arazo, Diego Ortego, Paul Albert, Noel O'Connor, and Kevin McGuinness. Unsupervised label noise modeling and loss correction. In International conference on machine learning, pp. 312-321. PMLR, 2019. (Cited on page 33)

Christopher G Atkeson and Stefan Schaal. Robot learning from demonstration. In ICML, volume 97, pp. 12-20. Citeseer, 1997. (Cited on page 5)

Anas Awadalla, Mitchell Wortsman, Gabriel Ilharco, Sewon Min, Ian Magnusson, Hannaneh Ha-jishirzi, and Ludwig Schmidt. Exploring The Landscape of Distributional Robustness for Question Answering Models. arXiv preprint arXiv:2210.12517, 2022. (Cited on page 4)

Stephen H Bach, Bryan He, Alexander Ratner, and Christopher Ré. Learning the structure of generative models without labeled data. In International Conference on Machine Learning, pp. 273-282. PMLR, 2017. (Cited on page 4)

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022a. (Cited on page 1, 5)

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022b. (Cited on page 5, 47)

Michael Bain and Claude Sammut. A Framework for Behavioural Cloning. In Machine Intelligence 15, pp. 103-129, 1995. (Cited on page 5)

Rachel KE Bellamy, Kuntal Dey, Michael Hind, Samuel C Hoffman, Stephanie Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta, Aleksandra Mojsilovic, et al. AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. arXiv preprint arXiv:1810.01943, 2018. (Cited on page 4)

Yoshua Bengio, Geoffrey Hinton, Andrew Yao, Dawn Song, Pieter Abbeel, Yuval Noah Harari, Ya-Qin Zhang, Lan Xue, Shai Shalev-Shwartz, Gillian Hadfield, et al. Managing AI risks in an era of rapid progress. arXiv preprint arXiv:2310.17688. (Cited on page 18)

David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. Advances in neural information processing systems, 32, 2019. (Cited on page 4)

Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10925-10934, 2022. (Cited on page 4)

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, lya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models. OpenAl Blog, 2023. (Cited on page 47)

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. In Thirty-Fourth AAAl Conference on Artifi cial Intelligence, 2020. (Cited on page 29)

Sam Bowman. Artificial Sandwiching: When can we test scalable alignment protocols without humans? Al Alignment Forum, 2022. (Cited on page 5)

Samuel Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile Lukosuite, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540, 2022. (Cited on page 2, 47)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020. (Cited on page 14)

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision. In The Eleventh International Conference on Learning Representations, 2023. (Cited on page 5)

CAIS. Statement on AI risk. (Cited on page 4, 19, 47)

Joe Carlsmith. Scheming AIs: Will AIs fake alignment during training in order to get power? arXiv preprint arXiv:2311.08379. (Cited on page 48)

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650-9660, 2021. (Cited on page 17, 35, 40)

Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by secking flat minima. Advances in Neural Information Processing Systems, 34:22405-22418, 2021. (Cited on page 36)

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems, 33:22243-22255, 2020a. (Cited on page 35)

Yining Chen, Colin Wei, Ananya Kumar, and Tengyu Ma. Self-training avoids using spurious features under domain shift. Advances in Neural Information Processing Systems, 33:21061-21071, 2020b. (Cited on page 33)

Paul Christiano. Approval-directed bootstrapping. AI Alignment Forum, 2018. (Cited on page 8

Paul Christiano. Capability amplification. AI Alignment Forum, 2019. (Cited on page 8)

Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. (Cited on page 1, 2, 5, 47)

Paul Christiano, Buck Shlegeris, and Dario Amodei. Supervising strong learners by amplifying weak experts. arXiv preprint arXiv.:1810.08575, 2018. (Cited on page 2, 5)

Paul Christiano, Ajeya Cotra, and Mark Xu. Eliciting latent knowledge. Technical report, Alignment Research Center (ARC), 2022. (Cited on page 5, 11, 17, 44)

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In NAACL, 2019. (Cited on page 29)

Ajeya Cotra. The case for aligning narrowly superhuman models. AI Alignment Forum, 2021. (Cited on page 5)

Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. Advances in neural information processing systems, 28, 2015. (Cited on page 14)

Abram Demski and Scott Garrabrant. Embedded agency. arXiv preprint arXiv:1902.09469, 2019. (Cited on page 2)

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010. $\mathit{I}$ 1929, 2020. (Cited on page 40)

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html. (Cited on page 42)

Owain Evans, Owen Cotton-Barratt, Lukas Finnveden, Adam Bales, Avital Balwit, Peter Wills, Luca Righetti, and William Saunders. Truthful AI: Developing and governing AI that does not lie. arXiv preprint arXiv:2110.06674, 2021. (Cited on page 5)

Geoffrey French, Michal Mackiewicz, and Mark Fisher. Self-ensembling for visual domain adaptation. arXiv preprint arXiv:1706.05208, 2017. (Cited on page 4)

Benont Frénay and Michel Verleysen. Classification in the presence of label noise: a survey. IEEE transactions on neural networks and learning systems, 25(5):845-869, 2013. (Cited on page 4)

Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. Born again neural networks. In International Conference on Machine Learning, pp. 1607-1616. PMLR, 2018. (Cited on page 4, 13)

Amelia Glaese, Nat McAleese, Maja Trebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022. (Cited on page 1, 5)

Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129:1789-1819, 2021. (Cited on page 4)

Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. Advances in neural information processing system.s, 17, 2004. (Cited on page 4, 10, 33, 34)

Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, and Dinglong Huang. CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. (Cited on page 4)

Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31, 2018. (Cited on page 4)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016. (Cited on page 40)

Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. Using trusted data to train deep networks on labels corrupted by severe noise. Advances in neural information processing system.s, 31, 2018. (Cited on page 4)

Dan Hendrycks, Kimin Lee, and Mantas Mazeika. Using pre-training can improve model robustness and uncertainty. In International conference on machine learning, pp. 2712-2721. PMLR, 2019. (Cited on page 4)

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning AI with shared human values. arXiv preprint arXiv:2008.02275, 2020a. (Cited on page 29)

Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and Dawn Song. Pretrained transformers improve out-of-distribution robustness. arXiv preprint arXiv:2004.06100, 2020b. (Cited on page 4)

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.. Measuring Mathematical Problem Solving With the MATH Dataset. Sort, 2(4):0-6, 2021. (Cited on page 40)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. (Cited on page 4)

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations, 2022. (Cited on page 35)

Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Cosmos QA: Machine reading comprehension with contextual commonsense reasoning. arXiv preprint arXiv.: 1909.00277, 2019. (Cited on page 29)

Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from learned optimization in advanced machine learning systems. arXiv preprint arXiv:1906.01820, 2019. (Cited on page 2, 48)

Geoffrey Irving, Paul Christiano, and Dario Amodei. AI safety via debate. arXiv preprint arXiv, $\boldsymbol{I}$ 805.00899, 2018. (Cited on page 2, 5)

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wil-son. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407, 2018. (Cited on page 36)

Fereshte Khani, Aditi Raghunathan, and Percy Liang. Maximum weighted loss discrepancy. arXiv prteprint arXiv:1906.03518, 2019. (Cited on page 5)

Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Langwage Technologies, Volume 1 (Long Papers), pp. 252-262, 2018. (Cited on page 29)

Michael P Kim, Amirata Ghorbani, and James Zou. Multiaccuracy: Black-box post-processing for fairness in classification. In Proceedings of the 2019 AAAIACM Conference on Al, Ethics, and Society, pp. 247-254, 2019. (Cited on page 5)

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. (Cited on page 40, 41)

Durk P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. Advances in neural information processing systems, 27, 2014. (Cited on page 4)

Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. In The Eleventh International Conference on Learning Representation.s, 2023. (Cited on page 4, 16)

Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. Imagenet classification with deep convolu-tional neural networks. Advances in neural information processing systems, 25, 2012. (Cited on page 17, 40)

Anders Krogh and John Hertz. A simple weight decay can improve generalization. Advances in neural information processing systems, 4, 1991. (Cited on page 35)

Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. In International Conference on Learming Representations, 2022. (Cited on page 4, 35)

Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint arXiv:1610.02242, 2016. (Cited on page 4)

Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, volume 3, pp. 896. Atlanta, 2013. (Cited on page 33)

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with Al feedback. arXiv preprint arXiv:2309.00267, 2023. (Cited on page 5)

Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. In The Eleventh International Conference on Learning Representations, 2022a. (Cited on page 4)

Yoonho Lee, Huaxiu Yao, and Chelsea Finn. Diversify and disambiguate: Learning from under-specified data. arXiv preprint arXiv:2202.03418, 2022b. (Cited on page 18)

Jan Leike and Ilya Sutskever. Introducing Superalignment. OpenAI Blog, 2023. (Cited on page 8, 47）

Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018. (Cited on page 2, 5)

Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. arXiv preprint arXiv:2002.07394, 2020. (Cited on page 4)

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. arXiv preprint arXiv:2306.03341, 2023. (Cited on page 5, 47)

Lichess Team. Lichess Database. https : //github .com/lichess-org/database, 2023. Accessed: 2023. (Cited on page 7)

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's Verify Step by Step. arXiv preprint arXiv:2305.20050, 2023. (Cited on page 5)

Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just train twice: Improving group robustness without training group information. In International Conference on Machine Learning, pp. 6781-6792. PMLR, 2021. (Cited on page 5)

Ziquan Liu, Yi Xu, Yuanhong Xu, Qi Qian, Hao Li, Rong Jin, Xiangyang Ji, and Antoni B Chan. An empirical study on distribution shift robustness from the perspective of pre-training and data augmentation. arXiv preprint arXiv:2205.12753, 2022. (Cited on page 4)

Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, and James Bailey. Normalized loss functions for deep learning with noisy labels. In International conference on machine learning, pp. 6543-6553. PMLR, 2020. (Cited on page 4)

Ian R McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, et al. Inverse Scaling: When Bigger Isn't Better. arXiv preprint arXiv:2306.09479, 2023. (Cited on page 8, 13)

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35:17359-17372, 2022. (Cited on page 5)

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In EMNLP, 2018. (Cited on page 29)

Richard Ngo, Lawrence Chan, and Soren Mindermann. The alignment problem from a deep learning perspective. arXiv preprint arXiv:2209.00626, 2022. (Cited on page 48)

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial NLI: A new benchmark for natural language understanding. arXiv preprint arXiv: 1910.14599, 2019. (Cited on page 29)

Chris Olah, Arvind Satyanarayan, Ian Johnson, Shan Carter, Ludwig Schubert, Katherine Ye, and Alexander Mordvintsev. The Building Blocks of Interpretability. Distill, 2018. https://distill.pub/2018/building-blocks. (Cited on page 47)

OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. (Cited on page 2, 7, 28

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730-27744, 2022. (Cited on page 1, 2, 5, 7, 32, 47)

Lorenzo Pacchiardi, Alex J Chan, Soren Mindermann, Ilan Moscovitz, Alexa Y Pan, Yarin Gal, Owain Evans, and Jan Brauner. How to catch an AI liar: Lie detection in black-box llms by asking unrelated questions. arXiv preprint arXiv:2309.15840, 2023. (Cited on page 5)

Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Edouard Duch-esnay. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12 (85):2825-2830, 2011. (Cited on page 42)

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. Red teaming language models with language models. arXiv preprint arXiv:2202.03286, 2022a. (Cited on page 47)

Ethan Perez, Sam Ringer, Kamile Lukosite, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. arXiv preprint arXiv:2212.09251, 2022b. (Cited on page 47)

Mohammad Taher Pilehvar and Jose Camacho-Collados. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. arXiv preprint arXiv.:1808.09121, 2018. (Cited on page 29)

Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017. (Cited on page 14)

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748-8763. PMLR, 2021. (Cited on page 4)

Alexander Ratner, Stephen H Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. Snorkel: Rapid training data creation with weak supervision. In Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases, volume 1l, pp. 269. NIH Public Access, 2017. (Cited on page 4)

Scott Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. arXiv preprint arXiv:1412.6596, 2014. (Cited on page 4, 33)

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135-1144, 2016. (Cited on page 47)

Fabien Roger, Ryan Greenblatt, Max Nadeau, Buck Shlegeris, and Nate Thomas. Measurement tampering detection benchmark. arXiv preprint arXiv:2308.15605, 2023. (Cited on page 5)

Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky. Getting closer to AI complete question answering: A set of prerequisite real tasks. In Proceedings of the AAAl conference on artificial intelligence, volume 34, pp. 8722-8731, 2020. (Cited on page 29)

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211-252, 2015. (Cited on page 40)

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019. (Cited on page 4)

Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Alek-sander Madry. Editing a classifier by rewriting its prediction rules. Advances in Neural Information Processing Systems, 34:23359-23373, 2021. (Cited on page 5)

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019. (Cited on page 29)

Wiliam Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022. (Cited on page 2, 5, 47)

Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Arpit Bansal, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. Datasets for studying generalization from easy to hard examples. arXiv preprint arXiv:2108.06011, 2021a. (Cited on page 29)

Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, and Tom Goldstein. Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks. Advances in Neural Information Processing Systems, 34:6695-6706, 2021b. (Cited on page 7, 29)

Rui Shu, Hung Bui, Hirokazu Narui, and Stefano Ermon. A DIRT-T Approach to Unsupervised Domain Adaptation. In International Conference on Learning Representations, 2018. (Cited on page 4, 33)

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631-1642, 2013. (Cited on page 29)

Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. Advances in Neural Information Processing Systems, 33:19339-19352, 2020. (Cited on page 5)

Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2022. (Cited on page 4)

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overitting. The journal of machine learning research, 15(1):1929-1958, 2014. (Cited on page 35)

Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A Alemi, and Andrew G Wilson. Does knowledge distillation really work? Advances in Neural Information Processing Systems, 34:6906-6919, 2021. (Cited on page 4, 13)

Jacob Steinhardt. AI Forecasting: One Year In, 2022. (Cited on page 18)

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008-3021, 2020. (Cited on page 1, 5, 15, 32)

Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. Dream: A challenge data set and models for dialogue-based reading comprehension. Transactions of the Association for Computational Linguistics, 7:217-231, 2019. (Cited on page 29)

Oyvind Tafjord, Matt Gardner, Kevin Lin, and Peter Clark. Quartz: An open-domain dataset of qualitative relationship questions. arXiv preprint arXiv:1909.03553, 2019. (Cited on page 29)

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017. (Cited on page 4)

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018. (Cited on page 29)

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32, 2019. (Cited on page 29)

Alex Warstadt, Amanpreet Singh, and Samuel Bowman. Neural network acceptability judgments. Transactions of the Association for Computational Linguistics, 7:625-641, 2019. (Cited on page 29)

Colin Wei, Kendrick Shen, Yining Chen, and Tengyu Ma. Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data. In International Conference on Learning Representations, 2020. (Cited on page 33)

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned Language Models are Zero-Shot Learners. In International Conference on Learning Representation.s, 2021. (Cited on page 28, 29)

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yo-gatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022. (Cited on page 46)

Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017. (Cited on page 29)

John Wentworth. Alignment by Default. Al Alignment Forum, 2020. (Cited on page 5)

Gordon Seidoh Worley. Bootstrapped Alignment. A Alignment Forum, 2021. (Cited on page 8) Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, pp. 23965-23998. PMLR, 2022a. (Cited on page 4, 36)

Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7959-7971, 2022b. (Cited on page 4, 36)

Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Chris-tiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021. (Cited on page 2)

Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10687-10698, 2020. (Cited on page 4, 33)

Kun Yi and Jianxin Wu. Probabilistic End-To-End Noise Correction for Learning With Noisy Labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. (Cited on page 4)

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019. (Cited on page 29)

Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAIACM Conference on AI, Ethics, and Society, pp. 335-340, 2018. (Cited on page 5)

Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase Adversaries from Word Scrambling. In Proc. of NAACL, 2019. (Cited on page 29)

Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018. (Cited on page 4, 36)

Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. "Going on a vacation'' takes longer than "Going for a walk": A Study of Temporal Commonsense Understanding. In EMNLP, 2019. (Cited on page 29)

APPENDIX OUTLINE

. In Appendix A, we provide additional details on our setup and experiments.

. In Appendix B, we describe additional results, including negative results and methods that did not work well in our experiments.

. In Appendix C, we report results on easy-to-hard generalization, where we only provide supervision on easy examples.

. In Appendix D, we provide results in two more weak-to-strong learning settings: a self-supervised computer vision setting on ImageNet, and a pure linear probing setting.

. In Appendix E, we provide additional results and discussion on the effect of weak supervisor error simulation.

. In Appendix F, we discuss how we believe methodological progress should be made on superalignment.

. In Appendix G, we describe how our work fits into the bigger picture of alignment.
</end of paper 4>


