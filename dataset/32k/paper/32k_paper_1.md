<paper 0>
# Contrastive Decoding: Open-ended Text Generation as Optimization 

Xiang Lisa Li ${ }^{1}$, Ari Holtzman ${ }^{2}$, Daniel Fried ${ }^{3}$, Percy Liang ${ }^{1}$, Jason Eisner ${ }^{4}$,<br>Tatsunori Hashimoto ${ }^{1}$, Luke Zettlemoyer ${ }^{2,5}$, Mike Lewis ${ }^{5}$<br>Stanford University ${ }^{1}$, University of Washington ${ }^{2}$, Carnegie Mellon University ${ }^{3}$,<br>Johns Hopkins University ${ }^{4}$, FAIR $^{5}$<br>xlisali@stanford.edu, ahai@cs.washington.edu, dfried@cs.cmu.edu,<br>pliang@stanford.edu, jason@cs. jhu.edu, thashim@stanford.edu,<br>lsz@cs.washington.edu, mikelewis@meta.com


#### Abstract

Given a language model (LM), maximum probability is a poor decoding objective for open-ended generation, because it produces short and repetitive text. On the other hand, sampling can often produce incoherent text that drifts from the original topics. We propose contrastive decoding (CD), a reliable decoding approach that optimizes a contrastive objective subject to a plausibility constraint. The contrastive objective returns the difference between the likelihood under a large LM (called the expert, e.g. OPT-13B) and a small LM (called the amateur, e.g. OPT-125M), and the constraint ensures that the outputs are plausible. CD is inspired by the fact that the failures of larger LMs (e.g., repetition, incoherence) are even more prevalent in smaller LMs, and that this difference signals which texts should be preferred. CD requires zero additional training, and produces higher quality text than decoding from the larger LM alone. It also works across model scales (OPT-13B and GPT2-1.5B) and significantly outperforms four strong decoding algorithms (e.g., nucleus, top-k) in automatic and human evaluations across wikipedia, news and story domains. ${ }^{1}$


## 1 Introduction

Open-ended text generation aims to craft fluent and coherent textual continuations of given prompts, laying foundations for various downstream applications such as writing assistance and story generation (Brown et al., 2020). The canonical approaches often sample from large pre-trained language models (Holtzman et al., 2020; Fan et al., 2018; Radford et al., 2019), but the generated text is prone to incoherence and topic drift as unlucky sampling choices compound over long sequences (Eikema and Aziz, 2020; Maynez et al., 2020). On the other hand, searching for the most likely se-[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_1428cbe356819d533a4fg-01.jpg?height=640&width=766&top_left_y=725&top_left_x=1062)

Figure 1: Contrastive decoding exploits the contrasts between expert and amateur LM of different sizes by choosing tokens that maximize their log-likelihood difference. CD produces high-quality text that amplifies the good expert behavior and diminishes the undesired amateur behavior.

quences often results in short, repetitive and tedious text (Holtzman et al., 2020), indicating that maximizing probability is a wrong decoding objective.

We propose a new search-based approach, contrastive decoding (CD), that can generate fluent and lexically diverse text without compromising coherence. As shown in Figure 1, contrastive decoding takes an off-the-shelf large language model such as OPT-13B (that we call the expert) and an off-the-shelf smaller language model such as OPT-125M (that we call the amateur). CD searches for text that maximizes the difference between expert log-probabilities and amateur log-probabilities, subject to plausibility constraints which restrict the search space to tokens with sufficiently high probability under the expert LM.

Contrastive Decoding works because many failure modes of language models (short, repetitive, irrelevant or uninteresting strings) are more common
under smaller LMs than under larger LMs. Such outputs are further deemphasized by taking the difference between model log-probabilities. Conversely, stronger models tend to put more probability mass on desirable outputs, such as those with factual knowledge that has not been learnt by the weaker model, and these strings are emphasized by contrastive decoding.

Taking Figure 1 as an example, the expert model places significant probability mass on previous tokens such as "Hawaii" and "Honolulu", leading to a highly repetitive continuation from greedy search; and nonsensical tokens such as "Washington" may be sampled, leading to an incoherent continuation. A correct continuation "1961" is strongly preferred by contrastive decoding, despite only having a probability of 0.1 , and the continuation includes more correct facts. This example suggests that contrastive decoding generates outputs that emphasize the best of the expert LM and remove its amateur tendencies. Moreover, we provide a pragmatic interpretation of contrastive decoding in §4.

Compared to recent training-based methods that improve generation quality such as unlikelihood training (Welleck et al., 2020) and contrastive learning (Su et al., 2022; An et al., 2022), contrastive decoding requires zero additional training. We find that by simply contrasting two frozen language models of different sizes, we are able to decode higher quality text than from the larger LM alone. Furthermore, we find that better performance is achieved when the scale difference between expert and amateur is larger (§7.1). As a result, the optimal amateur model is also cheap to run and incurs very little inference time overhead.

We evaluate our contrastive decoding approach for open-ended text generation in three domains: Wikipedia, stories, and news, and we evaluate using different teacher-student combinations, including (GPT2-XL v.s. GPT2-small, OPT-13B v.s. OPT-125M). Compared to four decoding baselines (nucleus sampling, top-k, typical decoding and SimCTG) our contrastive decoding method significantly improves the coherence of generated text, and improves or maintains the same fluency levels, according to both human evaluation and automatic metrics.

## 2 Problem Statement

We consider decoding approaches for open-ended language generation, where the language models receive an input prompt and aim to generate a fluent and coherent continuation. Specifically, we consider a relatively short prompt of length $n$, denoted as $\mathrm{x}_{\text {pre }}=x_{1} \cdots x_{n}$, where $x_{i}$ is a token in the vocabulary $\mathcal{V}$. The decoder must generate continuations of length $m$, denoted as $\mathrm{x}_{\text {cont }}=x_{n+1}, \cdots, x_{n+m}$.

We generate text from a pre-trained autoregressive language model $p_{\mathrm{LM}}$. At decoding time, we iteratively decode one token at a time by conditioning on the preceding context:

$$
p_{\mathrm{LM}}\left(\mathrm{x}_{\text {cont }} \mid \mathrm{x}_{\text {pre }}\right)=\prod_{i=n+1}^{n+m} p_{\mathrm{LM}}\left(x_{i} \mid x_{<i}\right)
$$

where $p_{\mathrm{LM}}\left(x_{i} \mid x_{<i}\right)$ is the next token distribution. We use different subscripts to denote different LMs: $p_{\text {AMA }}$ is the amateur LM (e.g., GPT-2 small), and $p_{\text {EXP }}$ is the expert LM (e.g., GPT-2 XL).

One canonical decoding approach is to sample from a truncated next token distribution at each time step. For example, nucleus sampling (Holtzman et al., 2020) draws from the top $p$ percentile of the next token distribution; top-k sampling (Fan et al., 2018) draws from the top $k$ candidates in the next token distribution. Another common approach is to search for the most likely text sequence via greedy decoding or beam search (Wu et al., 2016); but this leads to repetition and tedious outputs.

## 3 Contrastive Decoding

We propose contrastive decoding as a search-based decoding method that optimizes a novel contrastive objective subject to our plausibility constraint. We first provide intuition and define the constrastive objective (§3.1). Second, we discuss the potential weakness of this objective alone, and introduce the plausibility constraint to correct for the weakness (§3.2). Then we define the full contrastive decoding method as our contrastive objective subject to the plausibility constraint (\$3.3). Finally, we elaborate on the design spaces by discussing the choices of amateurs (\$3.4).

### 3.1 Contrastive Objective

Smaller LMs demonstrate stronger tendencies to produce undesirable patterns (e.g., repetition, topic drift, and self contradiction) than larger LMs. For
example, when both expert (larger LM) and amateur (smaller LM) assign highest probability to a repetitive token, the expert LM is often less confident about this decision and assigns non-trivial probability mass to other good, non-repetitive continuations. Contrastive decoding is inspired by these observations. The goal is to factor out undesired behaviors highlighted by the smaller amateur LMs, and generate text from the remaining good behaviors of larger expert LMs.

To operationalize this intuition, we propose the contrastive objective $\mathcal{L}_{\mathrm{CD}}\left(\mathrm{x}_{\text {cont }}, \mathrm{x}_{\text {pre }}\right)$ :

$$
\log p_{\text {EXP }}\left(\mathrm{x}_{\text {cont }} \mid \mathrm{X}_{\text {pre }}\right)-\log p_{\text {AMA }}\left(\mathbf{x}_{\text {cont }} \mid \mathrm{x}_{\text {pre }}\right)
$$

The CD objective rewards text patterns favored by the large expert LMs and penalizes patterns favored by the small amateur LMs. However, amateur LMs are not always mistaken: small language models still capture many simple aspects of English grammar and common sense (e.g., subject verb agreement). Thus, penalizing all behaviors from amateur LMs indiscriminately would penalize these simple aspects that are correct (False negative), and conversely reward implausible tokens (False positive). To tackle this issue, we introduce the plausibility constraint, which complements our CD objective and avoids these failure modes.

## $3.2 \mathcal{V}_{\text {head }}$ : Adaptive Plausibility Constraint

To tackle the aforementioned issue, we propose an adaptive plausibility constraint $\left(\mathcal{V}_{\text {head }}\right)$ that exploits the confidence level of the expert $\mathrm{LM}$ to restrict the effect of the contrastive objective when the expert LM is highly confident:

$$
\begin{aligned}
& \mathcal{V}_{\text {head }}\left(x_{<i}\right)= \\
& \left\{x_{i} \in \mathcal{V}: p_{\mathrm{EXP}}\left(x_{i} \mid x_{<i}\right) \geq \alpha \max _{w} p_{\mathrm{EXP}}\left(w \mid x_{<i}\right)\right\}
\end{aligned}
$$

Here, $\alpha$ is a hyperparameter in $[0,1]$ that truncates the next token distribution of $p_{\text {EXP }}$. Larger $\alpha$ entails more aggressive truncation, keeping only high probability tokens, whereas smaller $\alpha$ allows tokens of lower probabilities to be generated. We set $\alpha=0.1$ throughout the paper.

This adaptive plausibility constraint corrects for both false positive and false negative failures of the contrastive objective:

False positives. An implausible token may be rewarded with a high score under our unconstrained contrastive objective. For example, the token "NetMessage" is highly implausible under the context of Figure 1, with $3 \times 10^{-9}$ of $p_{\text {EXP }}$ and $8 \times 10^{-14}$ of $p_{\text {AMA }}$; however, it attains the highest contrast of $\log p_{\text {EXP }}-\log p_{\text {AMA }}=10.6$, which is much higher than plausible tokens "1961" and "Hawaii". To handle the false positive problem, $\mathcal{V}_{\text {head }}$ filters out low probability tokens and only keeps high probability tokens in the candidate pool.

False negatives. When confronting an easy decision, the correct token that achieves high probability under both amateur LM and expert LM may receive a low score under the contrastive objective. For example, due to tokenization, the word "unicorn" consists of two subwords: "unic" and "\#orn", and the probability of "\#orn" given the prefix "unic" is close to 0.99 under both LMs, but the contrast $\log p_{\mathrm{EXP}}-\log p_{\mathrm{AMA}}$ is only $6 \times 10^{-4}$, which is much lower than bad continuations.

Here, $\mathcal{V}_{\text {head }}$ uses the expert LM's confidence (as defined by the $\alpha$ ratio with the max probability token in the given timestep) to avoid these false negative cases. The expert LM assigns high confidence to easy decisions, but not to tokens that reflect the undesired behaviors of the amateur, since probability mass is taken up by other candidate tokens the expert is able to consider. Our constraint keeps as few as one token in the candidate pool when the expert is highly confident about this token, which removes the impact of the contrastive objective, because the single token would always be highest ranked regardless of the $\mathrm{CD}$ objective.

### 3.3 Full Method

Combining the contrastive objective and the adaptive plausibility constraint, we obtain the full contrastive decoding formulation:

$$
\begin{align*}
& \max _{\mathrm{x}_{\text {cont }}} \mathcal{L}_{\mathrm{CD}}\left(\mathrm{x}_{\text {cont }}, \mathrm{X}_{\text {pre }}\right)  \tag{2}\\
& \text { subject to } \quad x_{i} \in \mathcal{V}_{\text {head }}\left(x_{<i}\right), \forall x_{i} \in \mathrm{x}_{\text {cont }}
\end{align*}
$$

The above objective is defined at the sequence level, which is intractable to optimize. Thus, we factor the objective to token level scores:

$$
\begin{align*}
& \operatorname{CD}-\operatorname{score}\left(x_{i} ; x_{<i}\right)  \tag{3}\\
& = \begin{cases}\log \frac{p_{\mathrm{EXP}}\left(x_{i} \mid x_{<i}\right)}{p_{\mathrm{AMA}}\left(x_{i} \mid x_{<i}\right)}, & \text { if } x_{i} \in \mathcal{V}_{\text {head }}\left(x_{<i}\right) \\
-\inf , & \text { otherwise }\end{cases}
\end{align*}
$$

We apply beam search to optimize CD-score, by first filtering tokens based on plausibility constraints $\mathcal{V}_{\text {head }}\left(x_{<i}\right)$, eliminating tokens that fail to
achieve sufficiently high probabilities under the expert LM. Then we score the remaining tokens based on the amount of contrast they demonstrate, according to $\log p_{\text {EXP }}\left(x_{i} \mid x_{<i}\right)-\log p_{\text {AMA }}\left(x_{i} \mid x_{<i}\right)$. As a result, we end up selecting plausible tokens under the expert LM that least resemble the amateur LM.

### 3.4 Choice of Amateur

The choice of amateur LM is an important decision for contrastive decoding. As discussed in §3.1, we should choose amateur LMs that exhibit the behaviors we would like to downweight from the expert LM. Here, we consider three aspects:

Scale. Smaller LMs have lower modeling capacity and are more prone to errors. Therefore, we choose the amateur LM to be the smallest model in the same family of the expert LM. For example, for OPT-13B expert, we choose OPT-125M as the amateur; for GPT-2 XL expert, we choose GPT-2 small as the amateur. We verify this design choice in $\S 7.1$. On the extreme end, employing n-gram models yields an amateur LM of extremely low capacity. But this choice hurts generation quality, because n-gram LMs incur too many errors to identify similar failure modes of the expert LM.

Temperature. We can manipulate the amateur LM behavior by tuning its temperature $\tau$. For example, applying a high temperature $(\tau>1)$ to the amateur LM results in flatter distributions; applying a low temperature ( $\tau$ close to 0 ) highlights the mode of the amateur distribution, which is more prone to errors (e.g. repetition). Therefore, we manipulate the temperature of the amateur LM to adjust the amateur behavior that will be penalized in contrastive decoding. In $\S 7.2$, we study the impact of $\tau$ to generation quality and set $\tau$ to 0.5 or 1.0 for our main experiments.

Context window. We can also weaken capacity by restricting the context window of the amateur LM (Li et al., 2016). For instance, we can only allow the amateur $\mathrm{LM}$ to condition on the last token of $x_{\text {pre }}$, but we allow the expert $\mathrm{LM}$ to condition on the entire $x_{\text {pre }}$. In other words, we decode from $\log \frac{p_{\mathrm{ExP}}\left(\mathrm{x}_{\text {cont }} \mid x_{1: n}\right)}{p_{\mathrm{AMA}}\left(\mathrm{x}_{\text {cont }} \mid x_{n}\right)}$. By conditioning the amateur LM only on partial prompts, the coherence of the amateur LM is weakened, and contrastive decoding produces more coherent text by highlighting the coherence nature of the expert LM. In $\S 7.5$, we study the impact of this design choice.

## 4 CD as Pragmatic Communication

Having formally described contrastive decoding, we now provide a pragmatic interpretation, justifying its validity through pragmatic communication goals .

A line of work in pragmatics (Grice, 1975) characterizes communication as a cooperative process between speakers and listeners. Several of these formalisms (Horn, 1984; Levinson, 2000) describe a tradeoff between speakers and listeners, where a speaker should generally produce language that is high quality (e.g. truthful, fluent, and relevant) while also being informative to a listener.

Our contrastive objective can be motivated by this tradeoff, with our expert and amateur LMs modeling a knowledgable speaker and a lessinformed listener: (1) Upweighting tokens by $p_{\mathrm{EXP}}$ and using our expert-based plausibility constraints generates tokens that have high probability under the expert LM, encouraging generated text to be fluent and relevant (e.g. upweighting '1961' in Figure 1). (2) Downweighting tokens by $p_{\mathrm{AMA}}$ suppresses language that is predictable by (i.e. less informative to) the amateur LM (e.g. downweighting 'Honolulu' and 'Washington'), and by proxy encourages the language to be informative to a listener in context. By combining these two criteria, our contrastive decoding method produces high quality text that satisfies the communicative goal of transferring relevant but not predictable information.

### 4.1 Special Cases of Contrastive Decoding

Maximum probability. Setting the amateur LM to a uniform distribution reduces $\mathrm{CD}$ to maximize log-probabilities under the expert LM.

N-gram blocking. If we set the amateur LM as an n-gram model whose n-gram counts are updated to fit the generated prefix, this yields a decoding algorithm with soft n-gram blocking. If we also set the amateur temperature to be very small, then it approaches the canonical heuristic of forbidding repeated n-grams (Paulus et al., 2018).

Diverse decoding. If we use the same LM as both amateur and expert and restrict the context window of the amateur LM (§3.4), our method is equivalant to the MMI decoding objective ( $\mathrm{Li}$ et al., 2016) sometimes used in dialog systems, which explicitly maximizes the pointwise mutual information between the $\mathrm{x}_{\text {pre }}$ and $\mathrm{x}_{\text {cont }}$.

## 5 Experimental Setup

### 5.1 Datasets and Metrics

We evaluate on three domains for open-ended text generation: news, Wikipedia, and story domains. For the news domain, we use news articles from Wikinews; ${ }^{2}$ for the Wikipedia domain, we use the WikiText-103 dataset (Merity et al., 2017); and for story domains, we use the BookCorpus (Zhu et al., 2015) (Project Gutenberg split).

We use the first 32 words in the passage as the prompt, and decode for 256 tokens for the continuations. We evaluate generated text with both automatic and human evaluation.

Diversity. This metrics aggregate n-gram repetition rates: DIV $=\prod_{n=2}^{4} \frac{\mid \text { unique } n \text {-grams }\left(\mathrm{x}_{\text {cont }}\right) \mid}{\text { total n-grams }\left(\mathrm{x}_{\text {cont }}\right) \mid}$. A low diversity score suggests the model suffers from repetition, and a high diversity score means the model generated text is lexically diverse.

MAUVE. MAUVE (Pillutla et al., 2021) score (the higher the better) measures the distribution similarity between the set of generated text and the set of gold reference.

Coherence. We follow $\mathrm{Su}$ et al. (2022) and approximate coherence by cosine similarity between the sentence embeddings of prompt $\mathrm{x}_{\text {pre }}$ and generated continuation $\mathrm{x}_{\text {cont }}$ : $\operatorname{COH}\left(\mathrm{x}_{\text {cont }}, \mathrm{x}_{\text {pre }}\right)=\frac{\operatorname{Emb}\left(\mathrm{x}_{\text {pre }}\right) \cdot \operatorname{Emb}\left(\mathrm{x}_{\text {cont }}\right)}{\left\|\operatorname{Emb}\left(\mathrm{x}_{\text {pre }}\right)\right\| \cdot\left\|\operatorname{EmB}\left(\mathrm{x}_{\text {cont }}\right)\right\|}$, where $\operatorname{Emb}(x)$ is the pre-trained SimCSE sentence embedding (Gao et al., 2021).

Human Eval. In order to evaluate the quality of the generated text, we consider two critical aspects: fluency and coherence. A fluent piece of text is written in grammatical English and has a natural flow (e.g. excluding unnatural repetition or web formatting). A coherent piece of text should stay on topic with the prompt and avoid unnatural topic drift. We ask Amazon Mechanical Turkers to read two continuations (A and B) of the same prompt, and choose the more fluent/coherent continuation or decide they are similar.

### 5.2 Baselines

We compare contrastive decoding with three sampling methods, each with the recommended hyperparameters: nucleus sampling ( $p=0.95)$, top-k sampling $(k=50)$, typical decoding (Meister et al., 2022) $(\tau=0.95)$; and two search-based methods:[^1]

greedy (max prob) decoding that uses $\log p_{\text {EXP }}$ as the objective, and contrastive search (CS) (Su et al., 2022; Su and Collier, 2022). Among them, nucleus sampling is the standard approach for open-ended text generation whose performance has been verified in various domains (Holtzman et al., 2020; DeLucia et al., 2020), and typical decoding is a recently proposed approach that excels in lexical diversity (Meister et al., 2022). We therefore conduct human evaluation by comparing CD against these two methods.

### 5.3 Models and Hyperparameters

In order to demonstrate that our approach generalizes across various LM families and sizes, we consider GPT-2 XL (1.5B), OPT (6.7B) and OPT (13B) as expert LMs and employ the smallest LM in their respective family as the amateurs: GPT-2 small (100M) and OPT (125M).

Recall that contrastive decoding introduces two hyperparameters: $\alpha$ is the parameter to adjust the plausibility threshold, and $\tau$ is the temperature of the amateur LM. We always set $\alpha=0.1$ for the main results in the paper - we find that this setting is quite robust and generalizes across various domains. For OPT experiments, we set the amateur temperature to 1.0 and for GPT-2 experiments, we set the amateur temperature to 0.5 . We use a beam size of 5 . We also study the impact of these hyperparameters in the ablation study $\S 7.2$, and we find that our method is robust to various hyperparameter values.

## 6 Main Results

### 6.1 Automatic Evaluation

As shown in Table 1, contrastive decoding outperforms all other decoding baselines in MAUVE score and coherence score $(\mathrm{COH})$ across three different domains (news, Wikipedia, stories) and two model sizes (1.5B, 13B). Contrastive decoding achieves comparable or slightly worse diversity compared to nucleus and typical sampling, but it achieves substantially better diversity than other search based methods.

Typical decoding and nucleus sampling produce lexically diverse text by choosing low probability tokens, at the expense of topic drift. For instance, in the story domain we observe the largest diversity gap between contrastive decoding and nucleus sampling ( 0.83 v.s. 0.94 ) in the 1.5B model, but we find that the gap shrinks ( 0.89 v.s. 0.93 ) as
the model size increases to 13 billion, suggesting that our decoding method would continue to improve as expert models continue to scale.

CD outperforms all the baselines in coherence scores by a large margin, followed by greedy decoding. Greedy decoding achieves good coherence despite being highly repetitive, because always repeating the same sentence is a degenerate way to circumvent topic drift. We believe our gain in coherence comes from three aspects: (1) CD searches to optimize our objective, avoiding the topic drift that can happen by chance in sampling-based generation techniques. (2) Our contrastive objective implicitly rewards coherence, because large LMs are typically more coherent than smaller LMs. (3) Finally, we restrict the context length of the amateur $\mathrm{LM}$ (§3.4), further encouraging CD to reward text that is connected with the prompt (Li et al., 2016).

### 6.2 Human Evaluation

We conduct human evaluation to compare our contrastive decoding approach against nucleus sampling (the canonical method that scores high under MAUVE) and typical decoding (the winning method for diversity metrics). ${ }^{3}$

As shown in Table 2, contrastive decoding generates significantly more coherent text compared to nucleus and typical decoding across three domains and two models: on average across settings, evaluators preferred CD 2.6x more than nucleus sampling and $6.4 \mathrm{x}$ more than typical decoding when evaluating coherence. As for fluency, CD is preferred $1.4 \mathrm{x}$ more than nucleus sampling and $3.5 x$ more than typical decoding.

### 6.3 Qualitative Examples

We include a truncated qualitative example in Table 3. The nucleus sampling output shows a topic drift from a video game to music, and part of the generated text includes the format of an email; moreover, there is a style shift from third person narrative style to first person conversational style. These features match the noisy pre-training distribution of internet data, but are not desirable in the context of this prompt. Contrastive decoding output stays on topic with the prompt and elaborates on various aspects of the game, making it more coherent in both content and style. We include more qualitative examples in the appendix.[^2]![](https://cdn.mathpix.com/cropped/2024_06_04_1428cbe356819d533a4fg-06.jpg?height=378&width=782&top_left_y=250&top_left_x=1022)

Figure 2: Generation quality when applying contrastive decoding to expert and amateur LMs of different scales (\$7.1). We explore the expert-amateur combination within GPT-2 family (OPT family results in the appendix). We find the larger scale gap between the expert and the amateur LMs, the more text quality improves.

## 7 Ablation Studies

### 7.1 Size of Amateur and Expert LMs

Recall in $\S 3.4$, we provide intuition that choosing smaller LMs as the amateur should improve contrastive decoding results. We empirically verify this in Figure 2.

The diagonal entries use the same model as expert and amateur, yielding highly repetitive text (low diversity score), because we cannot exploit any contrast between two identical LMs. The upper triangular entries use an expert LM that is smaller than the amateur LM, and this counter-intuitive setup leads to inferior text quality. The lower triangular entries use an expert LM that is larger than the amateur $\mathrm{LM}$, resulting in higher quality text, as measured by both diversity and MAUVE. In particular, the optimal design is to select the largest LM as the expert and the smallest one as the amateur (lower left corner).

Does this trend generalize to extremely low capacity LMs like n-gram models? We find that employing a trigram LM as the amateur produces low quality text with a MAUVE score of only 0.73 . Our findings indicate that contrastive decoding benefits most with an amateur LM that can emphasize the failure modes of the expert $\mathrm{LM}$, and the mistakes of a low-capacity n-gram model do not highlight failure modes of an expert LM.

### 7.2 The Impact of Amateur Temperature

Recall in $\S 3.3$, we introduced the amateur LM temperature $\tau$ as a hyperparameter. We study how sensitive our method is to $\tau$ as shown in Figure 3.

Large $\tau$ brings the amateur distribution closer to the uniform distribution, which makes contrastive

![](https://cdn.mathpix.com/cropped/2024_06_04_1428cbe356819d533a4fg-07.jpg?height=536&width=1265&top_left_y=246&top_left_x=401)

Table 1: Automatic evaluation results for wikipedia, wikinews, story datasets. The best scores for each (model, domain) setting are boldfaced. Contrastive decoding outperforms all other decoding baselines in MAUVE score and coherence score $(\mathrm{COH})$ for different model scales (1.5B, 6.7B, 13B). CD achieves comparable or slightly worse diversity compared to nucleus and typical sampling.

|  |  |  |  | coher |  |  | fluen |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | CD | Baseline | CD is better | same | Baseline is better | CD is better | same | Baseline is better |
|  | CD (GPT-2 XL) | nucleus (GPT-2 XL) | $0.714^{*}$ | 0.083 | 0.202 | 0.548 | 0.083 | 0.369 |
| 犬्: | CD (GPT-2 XL) | typical (GPT-2 XL) | $0.887^{*}$ | 0.046 | 0.067 | $\mathbf{0 . 7 0 3}^{*}$ | 0.082 | 0.215 |
| $\frac{\frac{2}{3}}{3} \quad$ | CD (OPT-13B) | nucleus (OPT-13B) | 0.556 | 0.202 | 0.242 | 0.419 | 0.197 | 0.384 |
|  | CD (OPT-13B) | typical (OPT-13B) | $0.773^{*}$ | 0.106 | 0.121 | $0.687^{*}$ | 0.152 | 0.162 |
|  | CD (GPT-2 XL) | nucleus (GPT-2 XL) | $\mathbf{0 . 7 0 8}^{*}$ | 0.042 | 0.25 | $\mathbf{0 . 5 8 3}^{*}$ | 0.12 | 0.297 |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_1428cbe356819d533a4fg-07.jpg?height=42&width=48&top_left_y=1289&top_left_x=255) | CD (GPT-2 XL) | typical (GPT-2 XL) | $0.771^{*}$ | 0.151 | 0.078 | $0.755^{*}$ | 0.151 | 0.094 |
| 글 | CD (OPT-13B) | nucleus (OPT-13B) | $0.585^{*}$ | 0.221 | 0.195 | 0.518 | 0.123 | 0.359 |
|  | CD (OPT-13B) | typical (OPT-13B) | $0.693^{*}$ | 0.099 | 0.208 | 0.49 | 0.297 | 0.214 |
|  | CD (GPT-2 XL) | nucleus (GPT-2 XL) | $0.636^{*}$ | 0.045 | 0.318 | 0.404 | 0.106 | 0.49 |
| $D$ | CD (GPT-2 XL) | typical (GPT-2 XL) | 0.506 | 0.256 | 0.238 | 0.387 | 0.363 | 0.25 |
| 导 | CD (OPT-13B) | nucleus (OPT-13B) | $0.616^{*}$ | 0.101 | 0.283 | 0.449 | 0.293 | 0.258 |
|  | CD (OPT-13B) | typical (OPT-13B) | $0.626^{*}$ | 0.202 | 0.172 | 0.52 | 0.212 | 0.268 |

Table 2: Human evaluation results for wikipedia, wikinews, story datasets. We boldfaced the max column and * the numbers that are significantly better than the two other columns combined. Contrastive decoding generates significantly more coherent text compared to nucleus and typical decoding across three domains and two models. CD also generates better or comparably fluent text compared to two baselines.

Prompt A version of Sonic the Hedgehog was developed by Ancient and released in 1991 for Sega's 8 - bit consoles, the Master System and Game

CD A version of Sonic the Hedgehog was developed by Ancient and released in 1991 for Sega's 8 - bit consoles, the (1.5B) Master System and Game Gear. It featured a new level editor called the Sonic Editor and was released as an expansion pack to Sonic CD in 1992, which was itself a port from the Genesis version. The editor allowed the creation and playtesting of custom levels for Sonic CD, and the expansion was also bundled with Sonic CD. It is also available for the PC as Sonic Editor. The game is considered a spiritual successor to the Genesis game Sonic 3 \& Knuckles and the Genesis port Sonic 3D Blast, as it features the same gameplay elements, including the Spin Dash, and also shares many levels and music with those games...

Nucleus A version of Sonic the Hedgehog was developed by Ancient and released in 1991 for Sega's 8 - bit consoles, the

(1.5B) Master System and Game Gear. It was set in the garden city of Sky Sanctuary and centered around Sonic's new helper, the pre-historic pillar-crusher-man Waddle Dee. Sonic the Hedgehog all the way was released for Game Gear in 1992. The author of the story speaks only for himself, but here it is.

TO: Sonic the Hedgehog

FROM: Anti-Socials

SUBJECT: Argyle DeBrosse

Contents show]

What's that? You want me to tell you what I did, huh? Well, some things never change.

Yeah. Basically, you. Now listen: I saw you on the NWA charts for ages. I heard about how great you were, and that you were a great singer. And then... later, after hearing it five times in a row, that's when I realized that I couldn't win this particular battle. You and your music have the edge...

Table 3: Qualitative example of contrastive decoding versus nucleus sampling. CD produces more coherent text both in content and style, whereas nucleus sampling produces text that suffers from topic and style drifts.

![](https://cdn.mathpix.com/cropped/2024_06_04_1428cbe356819d533a4fg-08.jpg?height=334&width=537&top_left_y=261&top_left_x=340)

Figure 3: Ablation studies for amateur temperature $\tau$ (§7.2). The figure shows how MAUVE and diversity score change as we vary the $\tau$ values, labeled next to each dot. We find that $\tau \in[0.5,1.0]$ robustly result in high generation quality. For main results we use $\tau=0.5$ for GPT-2 and $\tau=1.0$ for OPT.

|  | name | DIV | MAUVE | COH | PPL |
| :--- | :--- | :---: | :---: | :---: | :---: |
|  | CD (search) | $\mathbf{0 . 8 9}$ | $\mathbf{0 . 9 2}$ | $\mathbf{0 . 6 9}$ | $\mathbf{1 7 . 7 7}$ |
| $\stackrel{\sim}{\sim}$ | CD (sample) | 0.81 | 0.85 | 0.68 | 18.48 |
|  | CD (full) | 0.89 | $\mathbf{0 . 9 2}$ | $\mathbf{0 . 6 9}$ | $\mathbf{1 7 . 7 7}$ |
| $\stackrel{\sim}{\sim}$ | CD $\left(-\mathcal{V}_{\text {head }}\right)$ | $\mathbf{1 . 0}$ | 0.01 | 0.23 | $2 \mathrm{e} 5$ |

Table 4: Automatic evaluation for the ablation studies of search v.s. sampling the contrastive objective (\$7.3) and the importance of the plausibility constraint $\mathcal{V}_{\text {head }}$ (\$7.4).

decoding generate repetitive text, as repetition is no longer penalized. Small $\tau$ makes the amateur LM more spiky and emphasizes undesired amateur behaviors, leading to better outputs from contrastive decoding. As shown in Figure 3, we find that setting $\tau$ in $[0.5,1.5]$ attains good and robust performance in coherence and fluency.

### 7.3 Sampling v.s. Search

Recall that contrastive decoding is a search-based approach that maximizes the contrastive objective subject to plausibility constraints. We explore a sampling alternative based on the same objective. Specifically, we normalize the CD-score $\left(x_{i} ; x_{<i}\right)$ (defined in §3.3) via softmax into a probability distribution from which we sample the next token.

|  |  |  | coherence |  |  | fluency |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | A | B | A | same | B | A | same | B |  |
| 1.5b | CD (search) | CD (sample) | $\mathbf{0 . 5 3 5}$ | 0.04 | 0.424 | $\mathbf{0 . 4 3 4}$ | 0.333 | 0.232 |  |
| 13b | CD (search) | CD (sample) | $\mathbf{0 . 4 6 5}$ | 0.162 | 0.374 | $\mathbf{0 . 4 7 5}$ | 0.131 | 0.394 |  |
| 1.5b | CD (full) | CD (-context) | $\mathbf{0 . 4 2 4}$ | 0.172 | 0.404 | $\mathbf{0 . 3 6 4}$ | 0.283 | 0.354 |  |

Table 5: Human evaluation for the ablation studies of search v.s. sampling the contrastive objective (\$7.3) and ignoring prefix v.s. including prompt to the amateur $\mathrm{LM}$ (§7.5). CD (-context) denotes the ablation experiments where we condition on the entire context for both amatuer and expert, and $\mathrm{CD}$ (full) conditions the amateur only on the last context token.
As shown in Table 4 and Table 5, we find that sampling from this objective produces lower quality text than searching under the objective. According to automatic and human evaluations, CD (sample)'s fluency and coherence rating consistently falls behind $\mathrm{CD}$ (search), but sampling still yields reasonably good outputs.

### 7.4 Plausibility Constraints

In $\S 3.2$, we describe why including the feasibility constraints is critical. Here, we conduct an ablation study verifying this claim by removing the plausibility constraints $\mathcal{V}_{\text {head. }}$. We find that the generation outputs suffers from severe fluency issues, as easily shown by its MAUVE score of 0.01 in the $C D\left(-\mathcal{V}_{\text {head }}\right)$ row of Table 4.

### 7.5 Prompt Inclusion

We further experiment with ablating the prompt context on the amateur $\mathrm{LM}$ (§3.4), by letting the expert LM and amateur LM both condition on the entire $x_{\text {pre. }}$. Table 5 shows that the ablation slightly hurts coherence and fluency.

## 8 Related Work

Decoding Methods. Decoding algorithms can be broadly classified as either search or sampling algorithms. Current search methods (e.g. greedy and beam search) attain accurate generation in goaldriven tasks (e.g. summarization), but suffers from tedious and repetitive outputs in open-ended settings (e.g. story generation). Current sampling methods (e.g. nucleus (Holtzman et al., 2020), top$\mathrm{k}$ (Fan et al., 2018), and typical decoding (Meister et al., 2022)) produces more diverse and interesting text in open-ended settings, but suffers from unnatural topic drift. Contrastive decoding avoids topic drift by using search, and outperforms nucleus and top-k sampling in coherence while maintaining or improving fluency and lexical diversity.

Contrast in Text Generation. The idea of contrast for text generation has been explored in diverse settings (He et al., 2019; Li et al., 2016; Su et al., 2022). The closest work to ours is DExpert (Liu et al., 2021), which studies controllable text generation by contrasting an trained expert model (on non-toxic data) and a trained anti-expert model (on toxic data) to produce text that is non-toxic. In this work, we focus on open-ended text generation and show that it is possible to get domainand task-agnostic anti-experts simply by using a
smaller LM. Contrastive decoding contrasts offthe-shelf LMs of different scales to produce high quality text, without any training.

## 9 Conclusion and Future Work

We propose contrastive decoding, a search-based decoding approach that contrasts LMs of different scales. We evaluate our approach on open-ended text generation, and find that it improves over the prevalent methods like nucleus sampling in both fluency and coherence.

As future work, the idea of contrasting an expert (larger LM) and an amateur (smaller LM) can be expanded to myriad setups, for instance, contrasting an early checkpoint of an LM and a later checkpoint of the LM. We hope that this paper can encourage more exploration of how to use contrasting language models.

## Limitations

In this paper, we focus on open-ended text generation and demonstrate the effectiveness of contrastive decoding. We would like contrastive decoding to also work well for task-oriented generation settings such as summarization and machine translation. However, the idea of contrasting models across different scales (larger expert LM and smaller amateur $\mathrm{LM}$ ) is not directly applicable, because the modes of both amateur LM and expert LM are of high quality. Empirically, having a smaller summaization model (BART-small finetuned on summarization data) as the amateur LM yields lower ROUGE score than employing a uniform distribution as the amateur LM, which is equivalent to beam search based on log-probabilities. As future work, we aim to study the necessary properties of amateur LM to empower task-oriented generation (e.g. summarization, table-to-text).

## References

Chen An, Jiangtao Feng, Kai Lv, Lingpeng Kong, Xipeng Qiu, and Xuanjing Huang. 2022. Cont: Contrastive neural text generation. ArXiv, $\mathrm{abs} / 2205.14690$.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler,
Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc.

Alexandra DeLucia, Aaron Mueller, Xiang Lisa Li, and João Sedoc. 2020. Decoding methods for neural narrative generation. CoRR, abs/2010.07375.

Bryan Eikema and Wilker Aziz. 2020. Is map decoding all you need? the inadequacy of the mode in neural machine translation. In COLING, pages 4506-4520.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889-898, Melbourne, Australia. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Empirical Methods in Natural Language Processing (EMNLP).

H. Paul Grice. 1975. Logic and conversation. In Peter Cole and Jerry L. Morgan, editors, Speech Acts, volume 3 of Syntax and Semantics.

He He, Nanyun Peng, and Percy Liang. 2019. Pun generation with surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1734-1744, Minneapolis, Minnesota. Association for Computational Linguistics.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Laurence Horn. 1984. Toward a new taxonomy for pragmatic inference: Q-based and r-based implicature. Meaning, form, and use in context: Linguistic applications, 11:42.

Alex M Lamb, Anirudh Goyal ALIAS PARTH GOYAL, Ying Zhang, Saizheng Zhang, Aaron C Courville, and Yoshua Bengio. 2016. Professor forcing: A new algorithm for training recurrent networks. In $A d$ vances in Neural Information Processing Systems, volume 29. Curran Associates, Inc.

Stephen C Levinson. 2000. Presumptive Meanings: The Theory of Generalized Conversational Implicature. MIT Press.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North

American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110-119, San Diego, California. Association for Computational Linguistics.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 45824597, Online. Association for Computational Linguistics.

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, and Yejin Choi. 2021. DExperts: Decoding-time controlled text generation with experts and anti-experts. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6691-6706, Online. Association for Computational Linguistics.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906-1919, Online. Association for Computational Linguistics.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. 2022. Typical decoding for natural language generation. CoRR, abs/2202.00666.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

Romain Paulus, Caiming Xiong, and Richard Socher. 2018. A deep reinforced model for abstractive summarization. In International Conference on Learning Representations.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. https://openai.com/blog/better-language-models/.

Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence level training with recurrent neural networks. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.
Yixuan Su and Nigel Collier. 2022. Contrastive search is what you need for neural text generation. arXiv preprint arXiv:2210.14140.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. Neurips, $\mathrm{abs} / 2202.06417$.

Arun Venkatraman, Martial Hebert, and J.. Bagnell. 2015. Improving multi-step prediction of learned time series models. Proceedings of the AAAI Conference on Artificial Intelligence, 29(1).

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2020. Neural text generation with unlikelihood training. In International Conference on Learning Representations.

Sam Wiseman and Alexander M. Rush. 2016. Sequenceto-sequence learning as beam-search optimization. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1296-1306, Austin, Texas. Association for Computational Linguistics.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In arXiv preprint arXiv:1506.06724.
</end of paper 0>


<paper 1>
# An Empirical Study On Contrastive Search And Contrastive Decoding For Open-ended Text Generation 

Yixuan Su ${ }^{\dagger}$<br>${ }^{\oplus}$ University of Cambridge $\quad{ }^{\rho}$ Independent Researcher<br>ys484@cam.ac.uk


#### Abstract

In the study, we empirically compare the two recently proposed decoding methods, i.e. Contrastive Search (CS) and Contrastive Decoding (CD), for open-ended text generation. The automatic evaluation results suggest that, while CS performs worse than CD on the MAUVE metric, it substantially surpasses CD on the diversity and coherence metrics. More notably, extensive human evaluations across three different domains demonstrate that human annotators are universally more in favor of CS over CD with substantial margins.

The contradicted results between MAUVE and human evaluations reveal that MAUVE does not accurately reflect human preferences. Therefore, we call upon the research community to develop better evaluation metrics for open-ended text generation. To ensure the reproducibility of our work, we have open-sourced all our code, evaluation results, as well as human annotations at https://github. com/yxuansu/Contrastive_Search_versus_Contrastive_Decoding.


## 1 Introduction

Open-ended text generation aims at generating coherent as well as informative text continuation based on the given prompt, and it is the core component in various NLP applications [9, 11]. In this study, we compare the two recently proposed decoding methods for open-ended text generation, i.e. (i) contrastive decoding (CD) [3] and (ii) contrastive search (CS) $[10,8]$.

For a comprehensive comparison, we follow Li et al. [3] and conduct experiments on three benchmarks across different domains. On the one hand, the automatic evaluations (\$3.1) indicate that CD performs notably better than CS on the MAUVE metric. However, CS achieves substantially better results on the diversity and coherence metrics. On the other hand, extensive human evaluations (\$3.2) on three benchmarks validate that the human annotators are universally more in favor of the texts produced by CS than the ones produced by CD with substantial margins.

Given the contradicted results of MAUVE and human evaluations, we argue that MAUVE does not accurately reflect human preferences. In $\S 4$, we show that the human preferences better correlate with the balance between the diversity and the coherence aspects of the generated texts. Thereby, we suggest future research on better evaluation metrics for open-ended text generation to take into account these two aspects.

In summary, our contributions are:

- We conduct comprehensive experiments to compare the two recently proposed decoding methods, i.e. CD and CS, for open-ended text generation.
- We demonstrate that MAUVE does not accurately reflect the human preferences on different methods for open-ended text generation. Moreover, we suggest a plausible direction for future research on better evaluation metrics of open-ended text generation.


## 2 Preliminaries

### 2.1 Contrastive Decoding

Contrastive decoding (CD) is introduced by Li et al. [3]. Given a prompt text $\boldsymbol{x}_{<t}$, the selection of the output token $x_{t}$ is decided by comparing two separate language models (LM) as

$$
\begin{equation*}
x_{t}=\underset{\left.v \in \mathcal{V}_{\text {head }} \boldsymbol{x}_{<t}\right)}{\arg \max }\left\{\log p_{\mathrm{EXP}}\left(v \mid \boldsymbol{x}_{<t}\right)-\log p_{\mathrm{AMA}}\left(v \mid \boldsymbol{x}_{<t}, \tau\right)\right\} \tag{1}
\end{equation*}
$$

where $p_{\mathrm{EXP}}\left(\cdot \mid \boldsymbol{x}_{<t}\right)$ is the probability distribution produced by an expert LM. The $p_{\mathrm{AMA}}\left(\cdot \mid \boldsymbol{x}_{<t}, \tau\right)$ is the probability distribution produced by an amateur LM scaled with a predefined temperature $\tau$. Typically, the expert LM (e.g. GPT2-XL) is larger than the amateur LM (e.g. GPT2-Small). The candidate set $\mathcal{V}_{\text {head }}\left(\boldsymbol{x}_{<t}\right)$ is defined as

$$
\begin{equation*}
\mathcal{V}_{\text {head }}\left(\boldsymbol{x}_{<t}\right)=\left\{v \in \mathcal{V}: p_{\operatorname{EXP}}\left(v \mid \boldsymbol{x}_{<t}\right) \geq \alpha \times \max _{w} p_{\mathrm{EXP}}\left(w \mid \boldsymbol{x}_{<t}\right)\right\} \tag{2}
\end{equation*}
$$

where $\alpha$ is a hyperparameter.

### 2.2 Contrastive Search

In contrast to CD, contrastive search (CS) $[10,8]$ only requires a single LM to generate the text continuation conditioned on the prompt. Formally, given the prompt text $\boldsymbol{x}_{<t}$, the selection of the output token $x_{t}$ follows

$$
\begin{equation*}
x_{t}=\underset{v \in V^{(k)}}{\arg \max }\{(1-\alpha) \times \underbrace{p_{\theta}\left(v \mid \boldsymbol{x}_{<t}\right)}_{\text {model confidence }}-\alpha \times \underbrace{\left(\max \left\{s\left(h_{v}, h_{x_{j}}\right): 1 \leq j \leq t-1\right\}\right)}_{\text {degeneration penalty }}\} \tag{3}
\end{equation*}
$$

where $V^{(k)}$ is the set of top- $k$ predictions from the LM's probability distribution $p_{\theta}\left(\cdot \mid \boldsymbol{x}_{<t}\right)$. In Eq. (3), the first term, model confidence, is the probability of the candidate $v$ predicted by the LM. The second term, degeneration penalty, measures how discriminative of the candidate $v$ with respect to the previous context $\boldsymbol{x}_{<t}$ and $s(\cdot, \cdot)$ computes the cosine similarity between token representations. More specifically, degeneration penalty is defined as the maximum cosine similarity between the representation of the candidate $v$ and that of all tokens in $\boldsymbol{x}_{<t}$. Here, the candidate representation $h_{v}$ is computed by the LM given the concatenation of $\boldsymbol{x}_{<t}$ and $v$. Intuitively, a larger degeneration penalty of $v$ means it is more similar to the context, therefore more likely leading to the undesirable repetitions in the generated output. The hyperparameter $\alpha \in[0,1]$ regulates the importance of these two components.

## 3 Experiment

Evaluation Benchmarks. Following Li et al. [3], we conduct experiments on three benchmarks from different domains, including (i) articles from Wikinews ${ }^{1}$ in the news domain; (ii) Wikitext-103 dataset [5] from the Wikipedia domain; (iii) and BookCorpus [13] from the story domain.

Same as in Li et al. [3], the generation of LM is conditioned on the test prompts with a fixed length of 32. And the generation of the text ends upon reaching an end-of-document token or a maximum length of 256 tokens. To ensure our experiments are aligned with Li et al. [3], we directly use the data provided in the authors' released repository ${ }^{2}$.

Model and Baselines. We compare different decoding methods using the GPT2-XL model [7]. (i) Following Li et al. [3], in contrastive decoding (CD), the expert and amateur LM are set as GPT2-XL[^0]and GPT2-Small, respectively; And the $\alpha$ (see Eq. (2)) and $\tau$ (see Eq. (1)) for CD are set as 0.1 and 0.5 , respectively. (ii) For contrastive search (CS), we set $\alpha$ (see Eq. (3)) as a constant 0.6 ; And $k$ (see Eq. (3)) for news, Wikipedia, and story benchmarks is set as 5,5 , and 6 , respectively.

In addition to $\mathrm{CD}$ and $\mathrm{CS}$, in the experiments, we also report the results of other baseline methods, including (i) greedy search; (ii) top- $k$ sampling $(k=50)$ [1]; (iii) nucleus sampling $(p=0.95)$ [2]; and (iv) typical sampling $(\tau=0.95)[4]$.

Note that, for a fair comparison with Li et al. [3], we report the performance of the baseline methods (i.e. greedy search, top- $k$ sampling, nucleus sampling, typical sampling, and contrastive decoding (CD)) using the generated texts provided in the authors' released repository ${ }^{3}$. However, for contrastive search (CS), the reported numbers in Li et al. [3] are different from our reproduced numbers. Therefore, we re-implement the results of CS using the same benchmark data provided by Li et al. [3] in their official repository ${ }^{4}$.

### 3.1 Automatic Evaluation

Following previous studies $[10,8,3]$, we use the following metrics for automatic evaluation.

(i) Diversity takes into account the generation repetition at different $n$-gram levels and it is defined as: diversity $=\prod_{n=2}^{4}\left(1.0-\frac{\text { rep-n }}{100}\right)$, where rep-n $=100 \times\left(1.0-\frac{\mid \text { unique n-grams }(\hat{\boldsymbol{x}}) \mid}{\mid \operatorname{total} \text { n-grams }(\hat{\boldsymbol{x}}) \mid}\right)$ and $\hat{\boldsymbol{x}}$ is the text generated by the LM.

(ii) MAUVE [6] is designed for measuring the token distribution closeness between the generated text and the human-written text over the whole test set. Note that, while the maximum length of generation in the experiments is 256 , we follow $\mathrm{Li}$ et al. [3] and measure the MAUVE score by truncating the generated text to its first 128 tokens.

(iii) Coherence is recently introduced by $\mathrm{Su}$ and Collier [8] and it automatically measures the semantic coherence between the prompt and the generated text. Formally, given the prompt $\boldsymbol{x}$ and the generated text $\hat{\boldsymbol{x}}$, coherence is defined as the averaged log-likelihood of $\hat{\boldsymbol{x}}$ conditioned on $\boldsymbol{x}$ as

$$
\begin{equation*}
\operatorname{coherence}(\hat{\boldsymbol{x}}, \boldsymbol{x})=\frac{1}{|\hat{\boldsymbol{x}}|} \sum_{i=1}^{|\hat{\boldsymbol{x}}|} \log p_{\mathcal{M}}\left(\hat{\boldsymbol{x}}_{i} \mid\left[\boldsymbol{x}: \hat{\boldsymbol{x}}_{<i}\right]\right) \tag{4}
\end{equation*}
$$

where [:] is the concatenation operation and $\mathcal{M}$ is a massively pre-trained LM. In our experiments, we follow Su and Collier [8] and set $\mathcal{M}$ as the OPT-2.7B model [12].

| Method | Wikinews |  |  | Wikitext |  |  | Story |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | div.(\%) | MAUVE(\%) | coh. | $\operatorname{div} .(\%)$ | MAUVE(\%) | coh. | div.(\%) | $\operatorname{MAUVE}(\%)$ | coh. |
| Greedy Search ${ }^{*}$ | 3.55 | 13.96 | -0.47 | 1.77 | 4.91 | -0.41 | 0.86 | 2.65 | -0.34 |
| Top- $k$ Sampling ${ }^{*}$ | 91.56 | 89.86 | -2.22 | 87.49 | 81.00 | -2.37 | 91.22 | 87.49 | -2.45 |
| Nucleus Sampling* | 93.54 | 89.45 | -2.61 | 92.16 | 86.54 | -3.03 | 94.50 | 91.47 | -3.02 |
| Typical Sampling* | 95.37 | 90.97 | -3.26 | 94.82 | 86.07 | -3.71 | 96.29 | 88.58 | -3.68 |
| Contrastive Decoding ${ }^{*}$ | 91.57 | 92.20 | -2.16 | 88.02 | 91.46 | -2.19 | 86.41 | 93.17 | -2.09 |
| Contrastive Search | 93.72 | 84.14 | -1.39 | 89.35 | 77.97 | -1.56 | 93.06 | 84.74 | -1.61 |

Table 1: Automatic evaluation results, where div. and coh. denote diversity and coherence. The numbers marked with * are obtained using the generated texts originally released by Li et al. [3].

Evaluation Results. Table 1 presents the automatic evaluation results. On the one hand, we see that CD achieves the best MAUVE score on all evaluated benchmarks. On the other hand, CS yields competitive performances on the diversity metric and achieves substantially better results on the coherence metric than $\mathrm{CD}$ and other sampling methods.[^1]

### 3.2 Human Evaluation

To further compare contrastive decoding (CD) with contrastive search (CS), we conduct a human evaluation with 4 native-speaker graders from a third-party grading platform. We randomly select 150 test prompts from the benchmarks across different domains, and evaluate CD and CS through pairwise comparison. Specifically, for each test prompt, the annotators are given two texts, with random order, that are generated by $\mathrm{CD}$ and $\mathrm{CS}$. The annotators then decide which one is more likely written by humans considering the following aspects of the generated text:

- Coherence: Whether the generated text is semantically coherent.
- Fluency: Whether the generated text is fluent and easy to understand.
- Informativeness: Whether the generated text is diverse and contains interesting content.

| 领 | Method A is better |  | Neutral <br> $4.2 \%$ <br> $15.1 \%$ <br> $20 \%$ | Method B is better |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  | Nucleus Sampling* <br> Typical Sampling* <br> Contrastive Search | $25.0 \%$ <br> $7.8 \%$ <br> $\mathbf{6 8 . 5} \%^{\dagger}$ |  | $\mathbf{7 0 . 8} \%^{\dagger}$ <br> $\mathbf{7 7 . 1} \%^{\dagger}$ <br> $29.5 \%$ | Contrastive Decoding <br> Contrastive Decoding <br> Contrastive Decoding |
| 类 | Method A is better |  | Neutral | Method B is better |  |
|  | Nucleus Sampling* <br> Typical Sampling* <br> Contrastive Search | $20.2 \%$ <br> $6.7 \%$ <br> $\mathbf{6 5 . 0} \%^{\dagger}$ | $8.3 \%$ <br> $4.6 \%$ <br> $2.0 \%$ | $\mathbf{7 1 . 4} \%^{\dagger}$ <br> $\mathbf{8 8 . 7} \%^{\dagger}$ <br> $33.0 \%$ | Contrastive Decoding <br> Contrastive Decoding <br> Contrastive Decoding |
| $\stackrel{d}{d}$ | Method A is better |  | Neutral | Method B is better |  |
|  | Nucleus Sampling* <br> Typical Sampling* <br> Contrastive Search | $31.8 \%$ <br> $23.8 \%$ <br> $\mathbf{6 7 . 0} \%^{\dagger}$ | $4.5 \%$ <br> $25.6 \%$ <br> $1.0 \%$ | $\mathbf{6 3 . 6} \%^{\dagger}$ <br> $\mathbf{5 0 . 6} \%^{\dagger}$ <br> $32.0 \%$ | Contrastive Decoding <br> Contrastive Decoding <br> Contrastive Decoding |

Table 2: Human evaluation results. ${ }^{\dagger}$ means one method performs significantly better than the other as judged by Sign Test with $p$-value $<0.05$. * the pairwise evaluation results between (i) nucleus sampling and contrastive decoding as well as (ii) typical sampling and contrastive decoding are directly cited from Li et al. [3].

Table 2 presents the human evaluation results which validate that contrastive search (CS) significantly outperforms contrastive decoding (CD) and other sampling methods ${ }^{5}$ in all evaluated benchmarks from different domains. These results clearly demonstrate the superiority of contrastive search over other existing decoding strategies.

It is worth emphasizing that, as shown in Table 1, contrastive search yields notably lower MAUVE scores than CD and other sampling methods. Given this clear contradiction between MAUVE and human evaluations, we argue that MAUVE does not accurately reflect human preferences. Therefore, we call upon the research community to develop better evaluation metrics, for open-ended text generation, that more correlates with human judgements.

### 3.3 Case Study

Table 3 presents a qualitative example, from the news domain, comparing contrastive decoding and contrastive search. We see that the text generated by contrastive decoding contains excessive repetitions both on the lexical and phrasal levels, e.g. "The Pentagon", "The drones would likely", and etc. In contrast, the text generated by contrastive search is semantically coherent as well as grammatically fluent. It elaborates on the reasons of the military strike and provides diverse details of the incident. In Appendix A, we provide more qualitative examples for the comparison between these two decoding methods.[^2]

![](https://cdn.mathpix.com/cropped/2024_06_04_291f869615666516f328g-5.jpg?height=1821&width=1391&top_left_y=236&top_left_x=367)

Figure 1: Wikitext - Coherence versus MAUVE. Figure 2: Wikitext - Coherence versus Diversity.

## 4 Further Analysis

In this section, we provide in-depth comparison between contrastive search and other decoding methods. Specifically, we vary the $k$ (see Eq. (3)), from 2 to 10 , in contrastive search ${ }^{6}$ to generate texts using the benchmark from the Wikipedia domain. The generated texts are evaluated from three aspects, i.e. (i) coherence; (ii) diversity; and (iii) MAUVE, which are described in §3.1.[^3]

The evaluated results are presented in Figure 1 and Figure 2, respectively. ${ }^{7}$ On the one hand, Figure 1 indicates that the MAUVE score of contrastive search lags behind other decoding methods (except for greedy search) with clear margins, which obviously contradicts to the human judgements as presented in §3.2. Even by jointly considering the coherence and MAUVE metrics, it is hard to identify the better decoding method. On the other hand, from Figure 2, we see that contrastive search notably outperforms other methods on the balance between the coherence and diversity metrics, better correlating with human judgements.

Our results demonstrate that MAUVE does not accurately reflect human preferences on different methods. Moreover, we suggest future research on better evaluation metrics, for open-ended text generation, to take into account both the coherence and the diversity aspects of the generated text.

## 5 Conclusion

In this work, we empirically compare the two recently proposed decoding methods, i.e. contrastive decoding (CD) and contrastive search (CS). We conduct extensive experiments on three benchmarks from different domains. The automatic evaluation results suggest that $\mathrm{CD}$ achieves better results on MAUVE while CS performs better on diversity and coherence. Moreover, through extensive human evaluations, we show that the human annotators are universally more in favor of CS over CD with substantial margins. Given the contradicted results between MAUVE and human evaluations, we provide in-depth analysis which reveals that the balance between the diversity and coherence metrics better correlates with human judgements. Our observation provides a plausible path for future research on better evaluation metrics for open-ended text generation.

## References

[1] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. arXiv preprint arXiv:1805.04833, 2018.

[2] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.

[3] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. arXiv preprint arXiv:2210.15097, 2022.

[4] Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Typical decoding for natural language generation. arXiv preprint arXiv:2202.00666, 2022.

[5] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

[6] Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. Mauve: Measuring the gap between neural text and human text using divergence frontiers. Advances in Neural Information Processing Systems, 34:4816-4828, 2021.

[7] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[8] Yixuan Su and Nigel Collier. Contrastive search is what you need for neural text generation. arXiv preprint arXiv:2210.14140, 2022.

[9] Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: Plugging visual controls in text generation. arXiv preprint arXiv:2205.02655, 2022.

[10] Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. A contrastive framework for neural text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022 .[^4]

[11] Yixuan Su, Yan Wang, Deng Cai, Simon Baker, Anna Korhonen, and Nigel Collier. Prototypeto-style: Dialogue generation with style-aware editing on retrieval memory. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:2152-2161, 2021.

[12] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

[13] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages $19-27,2015$.
</end of paper 1>


<paper 2>
# On the performativity of SDG classifications in large bibliometric databases 

Matteo Ottaviani ${ }^{* 1}$ and Stephan Stahlschmidt ${ }^{1,2}$<br>${ }^{1}$ German Centre for Higher Education Research and Science Studies (DZHW),<br>Schützenstr. 6a, 10117 Berlin (Germany)<br>${ }^{2}$ Unit of Computational Humanities and Social Sciences (U-CHASS), EC3 Research<br>Group, University of Granada, Granada, Spain

May 7, 2024


#### Abstract

Large bibliometric databases, such as Web of Science, Scopus, and OpenAlex, facilitate bibliometric analyses, but are performative, affecting the visibility of scientific outputs and the impact measurement of participating entities. Recently, these databases have taken up the UN's Sustainable Development Goals (SDGs) in their respective classifications, which have been criticised for their diverging nature. This work proposes using the feature of large language models (LLMs) to learn about the "data bias" injected by diverse SDG classifications into bibliometric data by exploring five SDGs. We build a LLM that is fine-tuned in parallel by the diverse SDG classifications inscribed into the databases' SDG classifications. Our results show high sensitivity in model architecture, classified publications, fine-tuning process, and natural language generation. The wide arbitrariness at different levels raises concerns about using LLM in research practice.


Keywords: Sustainable Development Goals, Classifications, Bibliometric databases, Large Language Models, Text Analysis, Noun Phrases, OpenAlex, Web of Science, Scopus.[^0]

## 1 Introduction

Bibliometric databases play a critical role as digital infrastructures that enable bibliometric analyses and impact assessments within the scientific community. However, it is essential to acknowledge that these databases are not impartial; instead, they have a performative nature, as they are constructed based on specific understandings of the science system and value attributions (Whitley, 2000; Vinkler, 1988). Recently, there has been significant attention given to the contribution of the science system and its entities to the United Nations' Sustainable Development Goals (SDGs) in the bibliometric impact debate (Mishra et al., 2023: Meschede, 2020). The SDGs provide a comprehensive framework for addressing global challenges and promoting sustainable development in various domains. The latter is a global framework that includes monitoring mechanisms and indicators. European countries have adapted their national sustainability indicator systems to align with the UN Agenda 2030. In the context of bibliometrics, SDG classifications are promoted to assess the societal relevance and impact of scientific research (Armitage et al., 2020).

Major bibliometric databases, including Web of Science, Scopus, and OpenAlex, have introduced bibliometric classifications aligning publications with specific SDGs to facilitate the measurement of scientific contributions towards the SDGs. Armitage et al. (2020) carry out a bibliometric study aimed at comparing the Bergen and Elsevier approaches to finding scholarly publications related to the United Nations' SDGs. They show that the overlap in publications retrieved by the latter two approaches is small. Different search terms, combinations, and query structures significantly impact the retrieved publications, affecting the country rankings. The latter inconsistencies are not due to technical issues but rather different interpretations of SDGs. Understanding the reasons behind these differences is crucial for comprehending the performative aspects of bibliometric classifications and their impact on scientific outputs. We propose the application of Large Language Models (LLMs) for this purpose.

LLMs are pre-trained models utilizing deep learning techniques to generate human-like responses based on given prompts (Radford et al., 2019). These models are trained on vast amounts of text data and have shown remarkable language generation capabilities. Nevertheless, concerns have been raised about the objectivity and potential biases embedded within the generated answers (Bender and Friedman, 2018; Lipton, 2018).

In this work, we propose using LLMs, specifically the DistilGPT-2 model, a variant of GPT-2 particularly fitting our purposes (Radford et al., 2019; Sanh et al., 2019), to gain insights into the qualitative biases introduced by diverse SDG classifications. The choice of this LLM is due to its great compromise between embedding no prior knowledge about SDGs ( DistilGPT-2 has been trained on a
small dataset and, then, incorporated a significantly lower structural data bias compared to other renown LLMs) and serving basic LLM functions. We examine the following five SDGs: SDG 4. Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all; SDG 5. Achieve gender equality and empower all women and girls; SDG 8. Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all; SDG 9. Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation; SDG 10. Reduce inequality within and among countries.

Our research design involves three main steps: data collection and analysis, fine-tuning the LLM for each bibliometric database and each SDG, and employing text analysis techniques to explore the biases and variations in the generated responses. We identify a jointly indexed publication dataset of Web of Science, OpenAlex, and Scopus, counting 15,471,336 publications from 2015 to July 2023, which serves as the common ground for this research to identify the impact of the SDG classifications irrespective of the varying coverage of the underlying databases. From this dataset, we collect the varying publications attributed to the 5 SDGs mentioned above by each databases' SDG classification, creating distinct publication subsets from the common ground set for the fine-tuning process. For each SDG, three fine-tuned LLMs are then administered the same collection of prompts, allowing us to compare and analyse the generated responses. That is we condense the publication-level differences inflicted by the diverse SDG classifications among millions of publications into an aggregate textual summary to uncover structural differences between the SDG classifications. The responses generated by the LLMs allow us to observe general patterns on the classification's substance that are otherwise not observable due to the sheer amount of classified publications. The text analysis reveals distinct linguistic features and varying perspective associated with each SDG classification.

## 2 Data Collection

The data collection process for this study entails gathering publications from three data providers: Web of Science, OpenAlex and Scopus. All the data was obtained from the German Kompetenznetzwerk Bibliometrie. The WoS and Scopus data are snapshots taken in July 2023 while the OpenAlex database is the version that was made available in December 2023. As sketched in Fig.1, we have created a jointly indexed publication subset with records common to all three datasets based on an exact DOI match, which have been published between 2015 and July 2023, and where the DOI is unique to the record in all three databases. This allows for a comparison among OpenAlex, WoS, and Scopus controlling for the varying

![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-04.jpg?height=540&width=642&top_left_y=387&top_left_x=707)

Figure 1: The jointly indexed publication dataset of Web of Science, OpenAlex and Scopus, counts 15471336 publications. It is obtained on an exact DOI match, when publications are published between 2015 and July 2023, and where the DOI is unique. Moreover, only article or review in journal are accounted.

coverage. The publication time window has been chosen as follows. The lower bound of the data set time is due to when the UN Sustainable Development Summit in New York approved the 2030 Agenda for Sustainable Development. Regarding the upper bound, the reason depends on our current resources' availability (i.e., the limitation due to classified publications provided by WoS, updated to July 2023). Overall, we include those publications classified by Web of Science as either article or review in journal, and provided of an abstract (pivotal to pursue our analysis). According to the settings above, the jointly indexed publications dataset - independently of any SDG classification - of WoS, OpenAlex and Scopus, counts 15471336 items.

For each bibliometric database, the classified publications have been collected as follows: Clarivate directly delivered to us the WoS unique identifiers and the related SDG classifications. The OpenAlex snapshot present in our infrastructure is given of SDG labels associated with a score between 0 and 1 ; in line with their guidelines, a publication with a score above 0.4 is considered classified with respect to the SDG in question. Regarding Scopus, Elsevier provides long lists of search queries (coded in SciVal's language) to classify publications, involving fields like: author + keywords, abstracts, journal sources, etc. We translated them into SQL and retrieved those publications by means of our infrastructure. In Fig. 2 we show Venn diagrams to quantify the eventual overlap of the diverse SDG classifications, reporting number of publications and percentage with respect to the total number of classified publications for each SDG. Less than $8 \%$ of publications are
![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-05.jpg?height=998&width=1432&top_left_y=356&top_left_x=311)

Figure 2: Venn Diagrams of SDGs 4, 5, 8, 9, and 10, for Web of Science, OpenAlex, and Scopus.

uniformly assigned to a SDG validating former observations of highly disagreeing classifications.

## 3 Methods

The methodology employed in this research consists of two main steps: fine-tuning the DistilGPT-2 language model for each data provider and utilizing text analysis techniques to measure discrepancies among the data providers' responses to the same prompts. DistilGPT-2 is an English speaking, faster, and lighter variant of GPT-2; the aim of its development was to provide researchers with a playground to better understand larger generative language models (Hugging Face, 2020). And for that reason, it had been trained on a very limited dataset to embed the least possible prior knowledge in either content or instructions. It means that the structural data-bias of DistilGPT-2 is reduced to the minimum, letting us measure what comes from the different classifications. For each SDG classification
and each bibliometric DB, a blank copy of the DistilGPT-2 model is fine-tuned by the abstracts of those publications classified under that given SDG. This exposure allows the model to familiarize itself with the language and concepts present in the dataset, enabling it to generate responses that reflect the inherent perspectives within the data (Whitley, 2000).

As outlined in Fig, for each SDG the same set of prompts is administered to the three fine-tuned LLM models. In particular, for each SDG, a set of prompts has been generated asking Open AI ChatGPT to produce them, by relying on the list of the official UN SDG targets. The latter methodology has been chosen because OpenAI ChatGPT might reasonably mirror an average sample of high educated individuals / experts on SDGs formulating questions. The models generate responses based on these prompts, which serve as the input for the subsequent text analysis. In order to ensuring even conditions, three decoding strategies are employed, bringing to three different responses each prompt. The latter are: top- $k$, nucleus, and contrastive search; the first two are usually favoured by automatic evaluation while the third one by human evaluation (Su and Xu, 2022; $\mathrm{Su}$ and Collier, 2022). We analyse the responses through noun phrases analysis and topic modelling (i.e., LDA). The former resulted to be more informative and interpretable. Therefore, we extract noun phrases for each fine-tuned LLM. Once we obtain for each LLM a set of noun phrases alongside their occurrence, we proceed as follows:

1 For each pair of SDG and bibliometric DB, we aggregate the noun phrases of the 3 subsets corresponding to the different decoding strategies. For each strategy, the noun phrases with a frequency higher than $10 \%$ are selected. The outcome is a set of noun phrases for a given SDG classified by a given bibliometric DB.

2 For each SDG, we compare the three sets of noun phrases belonging to Web of Science, OpenAlex and Scopus. We issue a common subset, where we collect those noun phrases common to the three bibliometric DB's, which cannot differentiate between databases. After this filtering process, what is left is a subset of noun phrases unique to one or any two DBs.

## 4 Results

Overall, we fine-tune 15 LLMs to specific collections of abstracts. Each of them undergoes 500 prompts (drafted specifically to each SDG) through three decoding strategies, hence 1500 responses. We totally explore $1500 \times 3$ (DB's) x 5 (SDGs) $=22500$ responses through noun phrases analysis. In Fig 3 it is schematised the process that brings one of the 15 fine-tuned LLMs to its corresponding set of

![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-07.jpg?height=954&width=1054&top_left_y=428&top_left_x=501)

For each SDG\#

![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-07.jpg?height=337&width=1028&top_left_y=1528&top_left_x=523)

Figure 3: Schematic illustration of the research design followed in this paper. We fine-tune a blank large language model based on the architecture DistilGPT-2 to the subset of publication abstracts classified to a given SDG by a given bibliometric DB. Once obtained the fine-tuned LLM, we administrate to it a set of prompts (tailored on the SDG) through three different decoding strategies. Then, we collect into the same set the noun phrases extracted from the three response sets according to a minimum threshold. For each SDG, once obtained the latter sets for all the DBs involved, we filter out the common words, gathering them into another set.

![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-08.jpg?height=862&width=1468&top_left_y=343&top_left_x=297)

Figure 4: For each SDG, the common set is the collection of those noun phrases which emerge from LLM responses in all the three databases.

noun phrases, and then, how the outcomes from WoS, OpenAlex, and Scopus are compared for each SDG. As the prompt set counts 500 questions, and then 500 answers for each decoding strategy, then a noun phrase is considered valid in case its frequency (i.e. its occurrence among the responses) within a given strategy is greater that $50(10 \%)$. For each SDG, we finally obtain a set of those noun phrases found in common among the bibliometric DB's (see Fig.4), and three sets of noun phrases belonging to each bibliometric DB, respectively assigned to any two of the three databases (see Fig.5). Issuing "unique" and "common" sets is particularly helpful towards the final aim of this work, i.e., assessing the data perspective of bibliometric databases in classifying SDGs.

## 5 Discussion of the results

The aim of this work is identifying data biases within the SDG classification operated by bibliometric databases through the emerging technology of large language modeling. This approach is pretty promising as of LLM capabilities, and the increasing usage in research practice and society at large lets us figuring these data biases definitely relevant to assess. The first direct assessment of our contribution relates to identifying the publications each bibliometric database classified as
![](https://cdn.mathpix.com/cropped/2024_06_04_5705181dfa05bbf39758g-09.jpg?height=1976&width=1258&top_left_y=331&top_left_x=410)

Figure 5: For each SDG and bibliometric DB, frequency bar charts of the "unique" sets; i.e., noun phrases sets except those items that are in common among the three databases.
a given SDG. We examine 5 SDGs. After having built a jointly indexed publication set to provide even conditions for the comparison, we have collected the publications and quantified the overlapping of the diverse classifications.

As shown in Fig.2, for each SDG, the overlaps of classified publications might be extremely small, ranging from a minimum of $1.3 \%$ (SDG 10) to a maximum of $7.2 \%$ (SDG 4) when considering the simultaneous overlap of all three DBs. Even considering additionally the pairwise overlapping, the uncovered publications (i.e., those ones classified by only one DB) are at least $73 \%$ of the total in the "most agreeing classification" (SDG 4), and around $90 \%$ in the most misaligned case (i.e. SDG 10). Such a wide difference in the publication subsets is fairly expected to produce significantly diverse outcomes among the three bibliometric DBs and highlights the performativity of the SDG classification.

Performing fine-tuning process has contributed as first to raise our awareness of LLM usage. We have observed that different settings and hyper-parameter choices (e.g., batch size, epochs, token size, etc.) result in high sensitivity of LLM response. Once a LLM is fine-tuned on the classification of a given SDG by a certain DB, it is then questioned through a set of prompts tailored to the corresponding SDG.

The production of responses goes through the so-called decoding strategy. The latter determines the method by which a language model chooses the subsequent token (e.g., a word) in a sequence, based on the predicted probability for all potential tokens. Its choice has a substantial influence on the quality and diversity of the text produced.

We clearly observe how different strategies produce volatile content and text structure while maintaining high-quality responses. This remarkably enriches our awareness regarding the usage of LLM, either for generic purposes or in the research practice. This finding poses relevant ethical concerns as to this actual arbitrariness in the settings which are often thought as neutral.

We have gathered in Fig. 4 frequency bar charts of the common words found each SDG explored. The most populated one is SDG 9 (33 words), while the least is SDG 5 (22 words). Despite this variation, we assess a similar shape in the distribution to some extent, somehow drawing a concave triangular matrix. This pattern, almost repeated each SDG, lets figuring a certain regularity deriving from the relationship between LLM and the fine-tuned data. This assures that the research design brings the three bibliometric DB. Conversely, in Fig. 5 we show frequency bar charts of the unique sets, for each SDG and bibliometric DB. Each row corresponds to a given SDG and each column to a certain bibliometric DBs. Quantitatively speaking, SDG9 is the case we find highest agreement among the DB's. The number of words found in common is the highest and the unique word sets are the smallest. An interesting case is SDG 5, where OpenAlex identifies the double of noun phrases compared to WoS and Scopus, and the number of common
words is the least.

## 6 Conclusions

The SDG classifications of WoS, OpenAlex and Scopus provide each a different perspective on what constitutes SDG. Bibliometric classifications, while striving to offer objective measures, seem to present a specific focus, which is crucial in the attribution of social relevance via SDG classifications. Depending on the applied classification scientists and institutions working in the aforementioned fields might, or might not, be able to empirically underline their impact to policy makers.

LLMs have been instrumental in unearthing and understanding these perspectives. More important, assessing LLMs responses w.r.t. SDG classifications, lets us imagine what might lead informed decisions in the realm of policy making. The pre-trained DistilGPT-2 model, while smaller, offers several computational advantages; it is suitable for fine-tuning and generating scientific text. Moreover, contrary to more complex LLMs (e.g. Falcon), it owns no knowledge about SDGs in general by default. That is an essential feature for informing about data biases after fine-tuning.

The key findings of this work might be wrap up as follows. Large Language Models are sensitive to differences in SDG classifications and provide insights into the varied perspectives inscribed into the classifications. They further show high sensitivity to model architecture, fine-tuning process and decoding strategy.

Our results clearly show how decisive is an apparently objective science-informed practice as the bibliometric classification of SDGs. A variation of classified publications and/or technical settings at various stages might widely influence the attention of society.

## Funding Acknowledgments

Data for WoS, OpenAlex, and Scopus in this study were obtained from the German Kompetenznetzwerk Bibliometrie (https://bibliometrie.info/), funded by the German Federal Ministry for Education and Research (BMBF) with grant number 16WIK2101A. The study is funded by the German Federal Ministry for Education and Research (BMBF) with grant number 01PH20006B.

## References

Armitage, C. S., M. Lorenz, and S. Mikki (2020). Mapping scholarly publications related to the sustainable development goals: Do independent bibliometric ap-
proaches get the same results? Quantitative Science Studies 1(3), 1092-1108.

Bender, E. M. and B. Friedman (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics 6, 587-604.

Hugging Face (2020). Distilgpt-2 model card. https://huggingface.co/ distilgpt2. Accessed: 21 April 2024.

Lipton, Z. C. (2018). The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue 16(3), $31-57$.

Meschede, C. (2020). The sustainable development goals in scientific literature: A bibliometric overview at the meta-level. Sustainability 12(11), 4461.

Mishra, M., S. Desul, C. A. G. Santos, S. K. Mishra, A. H. M. Kamal, S. Goswami, A. M. Kalumba, R. Biswal, R. M. da Silva, C. A. C. Dos Santos, et al. (2023). A bibliometric analysis of sustainable development goals (sdgs): a review of progress, challenges, and opportunities. Environment, Development and Sustainability, 1-43.

Radford, A., J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. (2019). Language models are unsupervised multitask learners. OpenAI blog 1 (8), 9.

Sanh, V., L. Debut, J. Chaumond, and T. Wolf (2019). Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. In NeurIPS EMC ${ }^{2}$ Workshop.

$\mathrm{Su}, \mathrm{Y}$. and N. Collier (2022). Contrastive search is what you need for neural text generation. arXiv preprint arXiv:2210.14140.

$\mathrm{Su}, \mathrm{Y}$. and J. Xu (2022). An empirical study on contrastive search and contrastive decoding for open-ended text generation. arXiv preprint arXiv:2211.10797.

Vinkler, P. (1988). Bibliometric features of some scientific subfields and the scientometric consequences therefrom. Scientometrics 14(5-6), 453-474.

Whitley, R. (2000). The intellectual and social organization of the sciences. Oxford University Press, USA.


[^0]:    *Corresponding author, email: ottaviani@dzhw.eu

</end of paper 2>


