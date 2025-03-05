<paper 0>
# LLaMA Beyond English: An Empirical Study on Language Capability Transfer 

Jun Zhao*, Zhihao Zhang*, Luhui Gao, Qi Zhang ${ }^{\dagger}$, Tao Gui, Xuanjing Huang<br>${ }^{1}$ School of Computer Science, Fudan University<br>\{zhaoj19,zhangzhihao19,qz,tgui\} @fudan.edu.cn


#### Abstract

In recent times, substantial advancements have been witnessed in large language models (LLMs), exemplified by ChatGPT, showcasing remarkable proficiency across a range of complex tasks. However, many mainstream LLMs (e.g. LLaMA) are pretrained on English-dominant corpus, which limits their performance in other non-English languages. In this paper, we focus on how to effectively transfer the capabilities of language generation and following instructions to a non-English language. To answer this question, we conduct an extensive empirical investigation based on LLaMA, accumulating over 1440 GPU hours. We analyze the impact of key factors such as vocabulary extension, further pretraining, and instruction tuning on transfer. To accurately assess the model's level of knowledge, we employ four widely used standardized testing benchmarks: C-Eval, MMLU, AGI-Eval, and GAOKAO-Bench. Furthermore, a comprehensive evaluation of the model's response quality is conducted, considering aspects such as accuracy, fluency, informativeness, logical coherence, and harmlessness, based on LLM-Eval, a benchmarks consisting instruction tasks from 17 diverse categories. Our evaluation results demonstrate that comparable performance to state-of-the-art transfer models can be achieved with less than $1 \%$ of the pretraining data, both in terms of knowledge alignment and response quality. Furthermore, the experimental outcomes across the thirteen low-resource languages also exhibit similar trends. We anticipate that the conclusions revealed by the experiments will aid the community in developing non-English LLMs.


## Introduction

For decades, researchers in Natural Language Processing (NLP) have been exploring the fundamental principles of intelligence (Bubeck et al. 2023). The recent advances in large language models (LLMs) seem to have revealed a glimmer of hope. Benefitting from the unprecedented scales of model size and training data, many LLMs like ChatGPT (OpenAI 2022), PaLM (Anil et al. 2023), LLaMA (Touvron et al. 2023a), and others have emerged strong capabilities in reasoning (Cobbe et al. 2021), planning (Huang et al. 2022), and learning from experience (Dong et al. 2023) at or surpassing human levels. These general capabilities also provide a foundation for LLMs to address intricate[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_ef511855cc655c7bbd36g-01.jpg?height=512&width=810&top_left_y=736&top_left_x=1118)

Figure 1: Pretrained LLaMA models, which are primarily trained on English-dominated corpus (as depicted on the left), are not inherently proficient in handling non-English languages. We aim to investigate the necessity of vocabulary extension, further pretraining, and instruction tuning, as well as to what extent they influence the capability transfer. This exploration enables us to efficiently transfer LLaMA's language capabilities to non-English languages (as illustrated on the right), minimizing costs in the process.

real-world tasks, such as successfully completing the entire Uniform Bar Examination (UBE) (Katz et al. 2023) or coding based on natural language instructions (StabilityAI 2023).

Many well-known LLMs are capable of comprehending input and generating responses across different languages, thanks to their pretraining on a diverse mix of corpus from multiple languages. However, due to the imbalanced distribution of language resources, collecting extensive training data for all languages is nearly impossible (Ranta and Goutte 2021). Taking the representative LLM BLOOM (Scao et al. 2023) as an example, it has been pretrained on 46 natural languages. Yet, this number accounts for only $0.66 \%$ of the roughly 7,000 languages currently in use. Moreover, within the corpus of these 46 languages, there exists extreme imbalance, with the high-resource English texts being 2.8 million times more than that of the lowresource Chitumbuka language. This is not an isolated case. Another widely discussed language model, LLaMA, has
been pretrained primarily on English-dominated corpus, supplemented with limited data from 20 related languages that utilize the Latin and Cyrillic scripts. As a result, LLaMA exhibits inferior performance in contexts involving non-English languages where it has not undergone sufficient training. Some researchers collect large-scale data for specific languages of interest and retrain an LLM (Team 2023a). However, this inevitably leads to high computational and data collection costs, which is not suitable for lowresource languages. While Cui, Yang, and Yao (2023b) extend original vocabulary and further pretrain LLaMA with 30B Chinese tokens by LoRA (Hu et al. 2021), reporting promising results. Nonetheless, a fine-grained systematic investigation of the transfer process remains lacking.

In this work, we take a step towards gaining a comprehensive understanding of the language capability transfer in LLMs. As shown in figure 1, we empirically investigate several key aspects based on LLaMA:

(1) The impact of vocabulary extension on transfer. We find that further pretraining with 0.5 billion Chinese tokens on the original vocabulary significantly outperforms performance on the extended vocabulary, even though the latter has been further pretrained on over 30 billion tokens. This suggests that vocabulary extension might not be a suitable choice for small-scale incremental pretraining in the order of tens of billions.

(2) Training scales required for effective transfer. We find that further Chinese pretraining with 100 billion tokens or fewer is insufficient to significantly improve LLaMA's knowledge level. However, enhancing LLaMA's response quality (i.e., language generation capability), requires only hundreds of thousands of instruction data rather than a largescale further pretraining.

(3) The effect of transfer training on the original English capabilities. We find that exclusive reliance on Chinese corpora for transfer training markedly compromises LLaMA's original English proficiency, a concern alleviated effectively through multilingual joint training.

The aforementioned findings enable us to transfer LLaMA's capabilities of language generation and following instructions to non-English languages at minimal cost. Based on evaluation results from four widely used standardized testing benchmarks (C-Eval, GAOKAOBench, MMLU, AGI-Eval) and an instruction evaluation benchmark LLM-Eval, we achieve comparable knowledge level and response quality to the state-of-the-art Open Chinese LLaMA, while using less than $1 \%$ of the training data. Furthermore, extension experiments on another 13 low-resource languages also exhibit similar trends. We aim for the experimental results and analyses in this paper to provide assistance and guidance to the community in constructing non-English LLMs.

## Background and Overview

In this subsection, we firstly present the essential steps to develop an instruction-following LLM. Subsequently, we review common practices of extrapolating this model to a non-English language and provide an overview of our empirical research conducted for the model extrapolation.

## Step 1: Pretraining to acquire language capability and knowledge

As a significant source of foundational capabilities for a LLM, pretraining aims to predict the next token based on the prefix sequences. Formally, given a large corpus $\mathcal{D}$, the training objective is to minimize the following loss:

$$
\begin{equation*}
\mathcal{L}_{\text {pretrain }}=\sum_{x \in \mathcal{D}} \sum_{i} \log p_{\theta}\left(x_{i} \mid x_{1}, \ldots, x_{i-1}\right) \tag{1}
\end{equation*}
$$

where $x=\left\{x_{1}, \ldots, x_{n}\right\}$ denotes an input token sequence.

By pretraining on massive text data ranging from billions to trillions of tokens, LLMs are capable of capturing intricate language structures, semantics, and contextual relationships, thereby acquiring strong language generation capabilities. Additionally, these LLMs also learn how to comprehend concepts, facts, and the connections between them, leading to a broad understanding of world knowledge.

## Step 2: Instruction tuning for aligning with human intent

Instruction tuning (SFT) aims to further enhance the capability of LLMs to follow instructions. Its training data consists of many instruction-response pairs. The model needs to learn to accurately respond to instructions, rather than merely continuing from the preceding text. Formally, given an instruction dataset $\mathcal{D}^{\prime}=\{(I, Y)\}$, where $I$ represents a task instruction and $Y$ represents a desired response, the training objective of instruction tuning is to minimize the following loss:

$$
\begin{equation*}
\mathcal{L}_{i n s}=-\log p_{\theta}(Y \mid I) \tag{2}
\end{equation*}
$$

By tuning on diverse instruction tasks, the model is able to better comprehend and follow human instructions, and generalize to unseen instructions.

## Extrapolating LLMs to non-English languages

LLMs acquire language generation and instructionfollowing capabilities through pretraining and instruction tuning. However, English holds a dominant position in the field of natural language processing, possessing the most abundant collection of text data from various domains. LLMs trained on English-dominant corpora exhibit inferior performance on other non-English languages. Extrapolating LLMs to non-English languages poses a highly valuable research challenge. Common extrapolation approaches consist of the following three steps: (1) extending the vocabulary to add tokens of the target language, and thus enhancing encoding expressiveness to that language. (2) further pretraining to transfer language generation capabilities of LLMs to the target language. The required training scale for this step is generally on the order of billions of tokens, significantly less than the trillions of tokens needed for training from scratch. (3) conducting SFT in the target language to transfer instruction-following capabilities of LLMs.

This paper conducts a comprehensive empirical study of the aforementioned three steps, comparing the performance differences of LLMs before and after vocabulary extension,
and under various pretraining and SFT scales. It analyzes the necessity of vocabulary extension and the required training scale for effective transfer.

## Experimental Setup

This paper aims to explore how to effectively transfer the capabilities of language generation and following instruction to a non-English language. Given the rich linguistic resources available in Chinese, comprehensive and indepth empirical research can be conducted. Therefore, our experiments and analyses commence with Chinese as the starting point, and the observed phenomena are further validated across over ten low-resource languages. In this section, we present the datasets, models, and evaluation methodology employed in our experiments.

## Models

To avoid unnecessary large-scale repetitive pretraining, we employed open-source models trained on varying scales of Chinese corpora. Among these, LLaMA and LLaMA2 serve as checkpoints without undergoing explicit Chinese pretraining, whereas Chinese LLaMA and Chinese LLaMA2 are treated as checkpoints with Chinese pretraining of 30 billion tokens. The scale reaches 100 billion tokens for Open Chinese LLaMA. We employ the performance of these models as references for analysis and comparison.

LLaMA (Touvron et al. 2023a): LLaMA is a series of foundation models developed by Meta AI, trained on publicly available English-dominate corpus. The corpus includes CommonCrawl, C4, Github code, Wikipedia, Books, and ArXiv papers, amounting to approximately 1.4 trillion tokens. Among these sources, Wikipedia consists of multilingual text, contributing $4.5 \%$ of the total corpus. It covers 20 languages that use either the Latin or Cyrillic scripts. LLaMA achieves state-of-the-art results for foundation models of its size. For example, LLaMA-13B with just 13 billion parameters outperforms the much larger 175B parameter GPT-3 on many NLP benchmarks. We consider LLaMA-7B and LLaMA-13B in our experiments.

LLaMA2 (Touvron et al. 2023b): LLaMA2 is an enhanced and upgraded version of LLaMA. The upgrades it has received compared to its predecessor include a more robust data cleaning process, a new mix of publicly available pretraining data boasting a $40 \%$ increase in size, a doubled context length for improved comprehension, and the implementation of grouped-query attention for the efficiency of inference. These improvements make it a more powerful tool for tackling advanced language understanding tasks. We consider LLaMA2-7B in our experiments.

Chinese LLaMA (Cui, Yang, and Yao 2023b): Chinese LLaMA is an extension of the original LLaMA, designed to enhance its capability in understanding and generating Chinese text. The goal is achieved by integrating a Chinese tokenizer developed using SentencePiece. This tokenizer, with a vocabulary size of 49,953 , enables improved handling of Chinese characters. In addition, it employs parameter-efficient fine-tuning techniques (Hu et al. 2021) to reduce memory consumption during model training. In our experiments, we consider Chinese LLaMA 7B Plus, which is trained on a corpus of approximately $120 \mathrm{~GB}$ in size, equivalent to around 30 billion Chinese tokens.

Chinese LLaMA2 (Cui, Yang, and Yao 2023a): Chinese LLaMA2 is an advanced iteration of Chinese LLaMA. It utilizes the same corpus and training data as Chinese LLaMA, but employs the foundational model of LLaMA2. Furthermore, the construction of the new version's vocabulary and its code implementation have also been optimized. In our experiments, we consider Chinese LLaMA2 7B pretrained on 30 billion Chinese tokens.

Open Chinese LLaMA (OpenLMLab 2023): Open Chinese LLaMA is a larger-scale extended version of the original LLaMA. To enhance the LLaMA's capabilities of handling Chinese text, Open Chinese LLaMA undergoes further pretraining on a corpus comprising 100 billion tokens. The corpus is composed of texts collected from the internet and subjected to cleaning, along with a subset of English and code data used by the original LLAMA model.

## Datasets

To transfer the language capabilities of LLaMA to the non-English language of interest, we utilize two instruction datasets, namely BELLE and Bactrain-X, for training. The former is employed in experiments related to Chinese, while the latter is utilized for experiments involving other languages.

BELLE (Ji et al. 2023): BELLE is a large-scale Chinese instruction tuning dataset developed by Lianjia Tech., containing 1.5 million instruction-following example. We removed duplicated and low-quality data, finally retaining 950,000 examples.

Bactrain-X (Li et al. 2023): Bactrian-X contains instructions and responses across 52 languages to facilitate multilingual instruction tuning. It is created by translating $67 \mathrm{~K}$ English instructions from Alpaca-52k (Taori et al. 2023) and Dolly-15k (Conover et al. 2023) datasets into 51 languages, then generating responses with ChatGPT. In order to objectively and comprehensively assess the capabilities of the model, we conduct evaluations from two perspectives: response quality and knowledge level. For the former, we employ the LLM-Eval benchmark and translate it into various low-resource languages to support multilingual evaluation. As for the latter, we utilize four widely adopted standardized testing benchmarks: C-Eval, MMLU, AGI-Eval, and GAOKAO-Bench.

LLM-Eval (Zhang et al. 2023a): LLM-Eval is a manually constructed benchmark for instruction-following evaluation. It has 453 instruction tasks from 17 major categories, including factual question answering, reading comprehension, frame generation, paragraph rewriting, summarizing, math problem solving, reasoning, poetry generation, programming, and more.

C-Eval (Huang et al. 2023b): C-Eval is a Chinese evaluation suite with 13948 exam questions across 52 subjects and 4 difficulty levels from middle school to professional exams. It includes STEM, humanities, social science and other topics. C-Eval HARD is a subset of 8 challenging math and science subjects requiring advanced reasoning.

|  | Method | ACC. | F. | INFO. | LC. | $\mathrm{H}$. | AVG. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $1 \mathrm{k}$ SFT | LLaMA (Touvron et al. 2023a) | 0.482 | 1.194 | 0.858 | 0.614 | 2.970 | 1.224 |
|  | LLaMA with $10 K$ pretrain | 0.482 | 1.441 | 0.829 | 0.712 | 2.963 | 1.285 |
|  | LLaMA with $100 K$ pretrain | 0.587 | 1.952 | 0.881 | 0.991 | 2.973 | 1.477 |
|  | LLaMA with $1 M$ pretrain | 0.735 | 2.071 | 1.002 | 1.046 | 2.957 | 1.562 |
|  | Chinese LLaMA (Cui, Yang, and Yao 2023b) | 0.509 | 1.205 | 0.811 | 0.726 | 2.970 | 1.244 |
|  | Open Chinese LLaMA (OpenLMLab 2023) | 1.406 | 2.584 | 1.685 | 1.877 | 2.989 | 2.108 |
| $5 \mathrm{k}$ SFT | LLaMA (Touvron et al. 2023a) | 0.450 | 1.279 | 0.767 | 0.612 | $\overline{3.000}$ | 1.199 |
|  | LLaMA with $10 K$ pretrain | 0.411 | 1.372 | 0.814 | 0.612 | 2.961 | 1.258 |
|  | LLaMA with $100 K$ pretrain | 0.488 | 1.922 | 0.876 | 0.977 | 3.000 | 1.493 |
|  | LLaMA with $1 M$ pretrain | 0.682 | 2.085 | 1.039 | 1.008 | 2.969 | 1.623 |
|  | Chinese LLaMA (Cui, Yang, and Yao 2023b) | 0.581 | 1.341 | 0.899 | 0.783 | 2.992 | 1.432 |
|  | Open Chinese LLaMA (OpenLMLab 2023) | 1.295 | 2.481 | 1.667 | 1.884 | 2.969 | 2.245 |
| 950k SFT | LLaMA (Touvron et al. 2023a) | 1.783 | 2.767 | 2.142 | 2.212 | 2.993 | 2.379 |
|  | LLaMA with $1 M$ pretrain | 1.812 | 2.799 | 2.080 | 2.303 | 3.000 | 2.399 |
|  | LLaMA-EXT with $1 M$ pretrain | 1.591 | 2.726 | 1.918 | 2.164 | 2.998 | 2.279 |
|  | Chinese LLaMA (Cui, Yang, and Yao 2023b) | 1.808 | 2.795 | 2.112 | 2.313 | 3.000 | 2.406 |
|  | Open Chinese LLaMA (OpenLMLab 2023) | 1.890 | 2.858 | 2.189 | 2.390 | 2.993 | 2.464 |
|  | LLaMA2 (Touvron et al. 2023b) | 1.868 | 2.822 | 2.171 | 2.379 | 3.000 | 2.448 |
|  | Chinese LLaMA2 (Cui, Yang, and Yao 2023a) | 1.701 | 2.838 | 2.011 | 2.251 | 3.000 | 2.360 |

Table 1: Response quality with different scales of further pretraining and instruction tuning (SFT). ACC., F., LC., H., INFO., and AVG. respectively denote accuracy, fluency, logical coherence, harmlessness, informativeness and their average. Approximately 1 million samples account for around 0.5 billion tokens. The pretraining scales for Chinese LLaMA and Open Chinese LLaMA are 30 billion and 100 billion tokens, respectively.

MMLU (Hendrycks et al. 2020): MMLU measures a LLM's ability to learn and apply knowledge across 57 diverse subjects including STEM, humanities, and social sciences. The test covers a wide range of difficulty levels from elementary to advanced professional.

AGI-Eval (Zhong et al. 2023): AGIEval uses questions from standardized tests taken by millions of people, including college entrance exams, law school admission tests, and professional qualification exams. It has 19 tasks in both English and Chinese.

Gaokao-Bench (Zhang et al. 2023b): GAOKAO-Bench uses 2811 exam questions from Chinese college entrance exams (Gaokao) from 2010-2022 covering all subjects. It has 1781 multiple choice, 218 fill-in-blank, and 812 openended questions across math, Chinese, English, physics, etc.

## Evaluation Protocol

For LLM-Eval, we followed the practice of Zhang et al. (2023a), evaluating the response quality of a model through 5 scoring items: accuracy, fluency, informativeness, logicality, and harmlessness. Scores for each aspect range from 0 to 3 . We use the prompt shown in Appendix to submit the instruction, model response, and reference answer to GPT-4 for automated evaluation. Based on the results reported by Zhang et al. (2023a), this evaluation method demonstrates a high degree of consistency with human evaluation.

For the four standardized testing benchmarks, we calculate the accuracy metric for model responses. Additionally, we follow the common practice of employing a zero-shot setting for AGI-Eval and GAOKAO-Bench, while using a 5-shot setting for C-Eval and MMLU.

## Main Results

## The Impact of Vocabulary Extension on Transfer

When we aim to enhance the capabilities of a LLM in a specific language, vocabulary extension is an intuitively reasonable approach. In this section, we evaluate the impact of vocabulary extension through the LLMEval benchmark, and the experimental results are presented in table 1. Initially, we collected one million Chinese sentences from the internet (approximately 0.5 billion tokens) and further pretrain the original LLaMA without vocabulary extension. Surprisingly, we find that this model significantly ourperform the vocabulary-extended Chinese LLaMA, across settings of $1 \mathrm{~K}, 5 \mathrm{~K}$, and $950 \mathrm{~K}$ instruction tuning. This discovery is thought-privoking, given that the Chinese LLaMA underwent further Chinese pretraining on 30 billion tokens, a much larger volume than our 0.5 billion tokens. Moreover, within the $950 \mathrm{~K}$ setting, we include results from extending the vocabulary on original LLaMA and training it with the same 0.5 billion tokens, to mitigate the influence of training data discrepancy. The outcomes remain consistent. This indicates that vocabulary extension is not a favorable choice within training scales of tens of billions of tokens. While we don't negate the efficacy of vocabulary extension in settings involving larger-scale pretraining (such as trillions of tokens), as reported in other literatures (Team 2023b), this already leans more towards retraining than mere language transfer.

![](https://cdn.mathpix.com/cropped/2024_06_04_ef511855cc655c7bbd36g-05.jpg?height=648&width=1737&top_left_y=218&top_left_x=194)

Figure 2: Knowledge-level evaluation results on four benchmarks.

## Training Scales Required for Effective Transfer

Training scale constitutes another significant factor influencing the transferability of LLM capabilities, composed of both pretraining scale and instruction tuning scale. Experimental results are shown in table 1. Taking the example of LLaMA (with $10 \mathrm{~K}, 100 \mathrm{~K}$, and $1 \mathrm{M}$ further pretrain) and Open Chinese LLaMA, the scale of further Chinese pretraining gradually increases from 0 to 100 billion tokens. Under the settings of $1 \mathrm{~K}$ and $5 \mathrm{~K}$ instruction tuning, we observed that the response quality improves progressively with the increase in the scale of further pretraining. ${ }^{1}$ However, when the instruction tuning data scale escalates to $950 \mathrm{~K}$, we find no significant differences in response quality among the models. Consequently, we hypothesize that more further pretraining could accelerate the model's alignment with human instructions, but the mere tens of billions in training scale are insufficient to enable the model to grasp a greater amount of world knowledge. This leads to their convergence at similar response levels. In other words, the enhancement in response quality primarily stems from an improvement in language generation prowess rather than an elevation in knowledge level.

To validate this standpoint, we evaluated the model's knowledge level on four widely used standardized test benchmarks. As shown in Figure 2, LLaMA 7B, Chinese LLaMA 7B, and Open Chinese LLaMA 7B perform comparably on C-eval, gaokao-bench, and agi-eval, indicating no significant differences induced by further Chinese pretraining. It is worth noting that despite lacking further pretraining in Chinese, both LLaMA2-7B and LLaMA-13B outperform Open Chinese LLaMA on C-eval, MMLU, and AGI-Eval, suggesting that trillion-level pretraining and larger model sizes may indeed serve as effective pathways for enhancing model knowledge levels.[^1]

|  | $\mathrm{L}(0)$ | $\mathrm{L}(10 \mathrm{k})$ | $\mathrm{L}(100 \mathrm{k})$ | $\mathrm{L}(1 \mathrm{M})$ | Open |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Chinese | 10.151 | 8.697 | 6.634 | 5.249 | 3.924 |
| English | 14.691 | 15.625 | 29.553 | 198.840 | 15.045 |

Table 2: Model perplexity with different further pretraining scales. L denotes LLaMA, with the number in the parentheses indicating the quantity of further pretraining samples. Open denotes Open Chinese LLaMA.

## How about the Original English Capabilities

Another issue of interest to us is whether the improvement in Chinese proficiency has an impact on the existing English capabilities. To address this question, we additionally collected 200,000 Chinese samples from the internet and randomly extracted 200,000 English samples from the refinedweb dataset (Penedo et al. 2023). Utilizing these samples, we evaluate the English perplexity and Chinese perplexity of LLaMA models trained on differentscale corpora, as depicted in table 2. Our findings reveal that with the increase in further pretraining scale, the perplexity of the models decreases steadily in Chinese, yet notably increases in English. This suggests that enhancing the model's capabilities solely through a single Chinese corpus comes at the cost of sacrificing the original English proficiency.

Furthermore, we conduct perplexity assessments for Open Chinese LLaMA and find that both the Chinese and English perplexities remain low. This outcome is unsurprising, given that its training data incorporates both Chinese and English content, allowing for the decreases of Chinese perplexity without significant elevation in English perplexity. Overall, exclusive reliance on Chinese corpora for transfer training markedly compromises LLaMA's original English proficiency, a concern alleviated effectively through multilingual joint training.

| Language | 1k SFT |  |  |  |  |  |  |  |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | ACC. | F. | INFO. | LC. | H. | AVG. |  | ACC. | F. | INFO. | LC. | H. | AVG. |
| Arbic | 0.188 | 1.061 | 0.191 | 0.254 | 3.000 | 0.939 |  | 1.268 | 2.499 | 1.529 | 1.607 | 3.000 | 1.981 |
| Bengali | 0.046 | 0.492 | 0.050 | 0.041 | 3.000 | 0.726 |  | 0.959 | 2.257 | 1.156 | 1.189 | 3.000 | 1.712 |
| Gujarati | 0.061 | 0.426 | 0.052 | 0.063 | 2.998 | 0.720 |  | 0.683 | 1.795 | 0.875 | 0.790 | 2.995 | 1.428 |
| Hindi | 0.131 | 1.064 | 0.147 | 0.162 | 3.000 | 0.901 |  | 1.014 | 2.342 | 1.238 | 1.240 | 2.998 | 1.766 |
| Indonesian | 0.398 | 1.266 | 0.544 | 0.438 | 2.995 | 1.128 |  | 1.659 | 2.751 | 2.026 | 2.012 | 3.000 | 2.290 |
| Malayalam | 0.101 | 0.621 | 0.103 | 0.103 | 3.000 | 0.786 |  | 0.906 | 2.427 | 1.182 | 1.197 | 3.000 | 1.742 |
| Marathi | 0.095 | 0.781 | 0.107 | 0.117 | 2.998 | 0.820 |  | 1.038 | 2.476 | 1.288 | 1.364 | 2.998 | 1.833 |
| Nepali | 0.151 | 0.991 | 0.177 | 0.146 | 2.986 | 0.890 |  | 0.969 | 2.417 | 1.236 | 1.285 | 3.000 | 1.781 |
| Swahili | 0.083 | 0.712 | 0.090 | 0.086 | 2.998 | 0.794 |  | 1.569 | 2.707 | 1.955 | 1.907 | 3.000 | 2.228 |
| Tamil | 0.140 | 0.914 | 0.176 | 0.174 | 2.998 | 0.880 |  | 0.960 | 2.457 | 1.198 | 1.257 | 2.998 | 1.774 |
| Telugu | 0.054 | 0.560 | 0.057 | 0.090 | 3.000 | 0.752 |  | 0.539 | 1.735 | 0.674 | 0.712 | 3.000 | 1.332 |
| Urdu | 0.057 | 0.573 | 0.052 | 0.071 | 3.000 | 0.751 |  | 1.038 | 2.443 | 1.285 | 1.335 | 3.000 | 1.820 |
| Vietnamese | 0.105 | 0.623 | 0.126 | 0.117 | 3.000 | 0.794 |  | 1.361 | 2.595 | 1.665 | 1.710 | 3.000 | 2.066 |
| Average | 0.124 | 0.776 | 0.144 | 0.143 | 2.998 | 0.837 |  | 1.074 | 2.377 | 1.331 | 1.354 | 2.999 | 1.827 |

Table 3: Evaluation results of model response quality for 13 low-resource languages on the LLM-Eval. ACC., F., LC., H., INFO., and AVG. respectively denote accuracy, fluency, logical coherence, harmlessness, informativeness and their average.

## Extending the Analysis to Multiple Languages

In the previous section, our experiments focus on Chinese. To investigate whether similar conclusions could be drawn in other non-English languages, we extend our experiments to 13 low-resource languages. To ensure evaluation consistency, we translate LLM-Eval benchmark into these 13 languages and employ the same evaluation metrics. As shown in table 3, a significant improvement in response quality for all low-resource languages with the increase in SFT data. Among these languages, Arabic, Indonesian, and Vietnamese exhibited the best performance. Despite all thirteen languages being low-resource, these three languages are more frequently used (Scao et al. 2023). As a result, LLaMA encounters them more often (although their overall occurrence is small compared to English), allowing the model to quickly comprehend instructions in these languages. This aligns with the conclusion drawn in the previous section.

In the previous section, we observed that extending the vocabulary had a negative impact on language transferability. A plausible hypothesis is the existence of crosslingual semantic alignment within LLMs, which vocabulary expansion might disrupt. To validate this alignment hypothesis, we fine-tune LLaMA with a dataset of $1 \mathrm{k}$ instructions and examine the model's output. Excitingly, we observed a certain proportion of code-switching samples. As depicted in figure 3, these samples' model responses consist of tokens from multiple languages and are semantically coherent. We have observed that code-switching occurs not only in the transfer process when Chinese is the target language, but also when other 13 low-resource languages are target languages. As shown in figure 4, the proportion of samples with code-switching is approximately between $2 \%$ to $5 \%$. This indicates that LLaMA might have learned crosslingual alignment relationships between concepts during the pretraining process.
Human 1: 如何评价美国的医疗保障体系? 请从优点和缺点两个方面进行评价。

(Human 1: How would you evaluate the healthcare system in the United States? Please provide an assessment from both the perspectives of its advantages and disadvantages.)

Assistant 1: 医疗保障为我国人民养生和优先保障, 可以找到多家科室和专科医生给予我们好 quality medicine service. The medical system of the United States is very good in many aspects. ..

(Assistant 1: Healthcare coverage ensures the well-being and priority protection of the people in our country. It allows us to access numerous departments and specialized doctors who provide us with high-quality medical services. The medical system of the United States is very good in many aspects. ...)

Human 2: 请你为全球军事实力进行国家层面上的排序,不包括民间组织与其他机构。

(Human2:Please provide a country-level ranking of global military capabilities, excluding non-state organizations and other entities.)

Assistant 2: ... 1. 美国：武器：美国武器有着世界上最新的轻заряд车、追踪性能最强的潜水艇等等的優れた武器化状况。..

(Assistant 2: ... 1. United States: Weapons: The United States possesses advanced weaponry including the world's latest lightweight armored vehicles, submarines with the highest tracking capabilities, and other superior weapons. ...)

Figure 3: Case study of code-switching. Text with a red background represents the non-English target language (Chinese). Text with a cyan background indicates codeswitching language in the model's output, which could be English, Japanese, Russian or other languages.

![](https://cdn.mathpix.com/cropped/2024_06_04_ef511855cc655c7bbd36g-07.jpg?height=697&width=754&top_left_y=215&top_left_x=230)

Figure 4: Code-switching rate across languages.

## Related Work

## Resource Gap in LLMs

One of the main challenges of LLMs is the resource gap, as they are mainly pretrained on English corpus and have limited access to data from other languages. English dominates the field of NLP as an extremely highresource language with the most raw text data from various domains, leaving few of the over 7000 languages of the world represented in the field (Joshi et al. 2020). This creates a disparity in language models' capability to handle different languages. Previous findings indicate that LLMs have difficulty comprehending and generating non-English texts, particularly in low-resource languages(Nguyen et al. 2023; Zhu et al. 2023; Huang et al. 2023a). To address the resource gap, several solutions have been proposed or implemented by researchers and practitioners. One possible solution is to increase the amount of data available from various languages and fields, and make it accessible for pretraining and evaluating LLMs (Lin et al. 2022; Chen et al. 2022; Cahyawijaya et al. 2023) . However, this approach incurs significant computational expenses and the resource gap persists. Alternatively, multilingual language models trained on texts from different languages concurrently, such as mBERT (Devlin et al. 2019) and XLM-R (Conneau et al. 2020a), have been introduced to bridge the gap effectively.

## Cross-Lingual Transfer

Multilingual language models have demonstrated a high level of zero-shot or few-shot cross-lingual transferability across a wide range of tasks (Wu and Dredze 2019; Pires, Schlinger, and Garrette 2019; Winata et al. 2021b). This means that they can acquire the language capability from supervised data in one language and apply it to another without or with few additional training data. The mechanism behind the strong cross-lingual performance has been investigated by the researchers. It has been shown that multilingual language models have inferred universal rules applicable to any language (Artetxe, Ruder, and Yogatama 2020; Chi, Hewitt, and Manning 2020; Conneau et al. 2020b). Contrary to the common hypothesis that multilingual multilingual language models such as mBERT (Devlin et al. 2019) rely on a shared subword vocabulary and joint pretraining across multiple languages (Pires, Schlinger, and Garrette 2019; Cao, Kitaev, and Klein 2020; Wu and Dredze 2019), researchers have developed new understandings on the models, emphasizing the models' ability to learn universal semantic abstractions (Artetxe, Ruder, and Yogatama 2020; Chi, Hewitt, and Manning 2020). In terms of the factors that influence cross-lingual performance, researchers have associated transferability with parameter sharing (Conneau et al. 2020b; Dufter and Schütze 2020; Wu, Papadimitriou, and Tamkin 2022) and language distance (Conneau et al. 2020b; Eronen, Ptaszynski, and Masui 2023). We here further investigate the cross-lingual transferability of language models with new LLaMA-based experiments, presenting outcomes from a different aspect.

## Code-Switching

Code-switching is a phenomenon in which multilingual speakers switch between languages within a single utterance. Previous work on the performance of multilingual language models on code-switching tasks has shown mixed results. Some studies have suggested that pretrained models fine-tuned for specific code-switching scenarios can achieve state-of-the-art performance for certain language pairs such as English-Spanish and English-Hindi (Khanuja et al. 2020), while others have found that using meta-embeddings can yield better results with fewer parameters (Winata, Lin, and Fung 2019; Winata et al. 2019, 2021a). In another line of research, code-switching-based methods have been presented to improve the capability of multilingual language models (Jiang et al. 2020; Tan and Joty 2021; Krishnan et al. 2021).

## Conclusions

In this paper, we focus on how to effectively transfer the capabilities of language generation and following instructions to a non-English language. Specifically, we conducts a comprehensive empirical study to analyze the necessity of vocabulary extension and the required training scale for effective transfer. We find that vocabulary extension is uncessary and that comparable transfer performance to stateof-the-art models can be achieved with less than $1 \%$ of the further pretraining data. Additionally, we observe instances of code-switching during the transfer training, suggesting that cross-lingual alignment might have been internalized within the model. Similar results are observed from the extension experiments on the 13 low-resource languages. Our analysis and findings offer assistance and guidance to the community in developing non-English LLMs.

## References

Anil, R.; Dai, A. M.; Firat, O.; Johnson, M.; and Lepikhin, D. 2023. PaLM 2 Technical Report. arXiv:2305.10403.

Artetxe, M.; Ruder, S.; and Yogatama, D. 2020. On the Cross-lingual Transferability of Monolingual Representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 4623-4637. Online: Association for Computational Linguistics.

Bubeck, S.; Chandrasekaran, V.; Eldan, R.; Gehrke, J.; Horvitz, E.; Kamar, E.; Lee, P.; Lee, Y. T.; Li, Y.; Lundberg, S.; Nori, H.; Palangi, H.; Ribeiro, M. T.; and Zhang, Y. 2023. Sparks of Artificial General Intelligence: Early experiments with GPT-4. arXiv:2303.12712.

Cahyawijaya, S.; Lovenia, H.; Aji, A. F.; Winata, G. I.; and Wilie, B. 2023. NusaCrowd: Open Source Initiative for Indonesian NLP Resources. arXiv:2212.09648.

Cao, S.; Kitaev, N.; and Klein, D. 2020. Multilingual Alignment of Contextual Word Representations. arXiv:2002.03518.

Chen, G.; Ma, S.; Chen, Y.; Zhang, D.; Pan, J.; Wang, W.; and Wei, F. 2022. Towards Making the Most of Multilingual Pretraining for Zero-Shot Neural Machine Translation. arXiv:2110.08547.

Chi, E. A.; Hewitt, J.; and Manning, C. D. 2020. Finding Universal Grammatical Relations in Multilingual BERT. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 5564-5577. Online: Association for Computational Linguistics.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. CoRR, abs/2110.14168.

Conneau, A.; Khandelwal, K.; Goyal, N.; Chaudhary, V.; Wenzek, G.; Guzmán, F.; Grave, E.; Ott, M.; Zettlemoyer, L.; and Stoyanov, V. 2020a. Unsupervised Cross-lingual Representation Learning at Scale. arXiv:1911.02116.

Conneau, A.; Wu, S.; Li, H.; Zettlemoyer, L.; and Stoyanov, V. 2020b. Emerging Cross-lingual Structure in Pretrained Language Models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 60226034. Online: Association for Computational Linguistics.

Conover, M.; Hayes, M.; Mathur, A.; Xie, J.; Wan, J.; Shah, S.; Ghodsi, A.; Wendell, P.; Zaharia, M.; and Xin, R. 2023. Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM.

Cui, Y.; Yang, Z.; and Yao, X. 2023a. Chinese LLaMA and Alpaca Large Language Models.

Cui, Y.; Yang, Z.; and Yao, X. 2023b. Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca. arXiv:2304.08177.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171-4186. Minneapolis, Minnesota: Association for Computational Linguistics.
Dong, Q.; Li, L.; Dai, D.; Zheng, C.; Wu, Z.; Chang, B.; Sun, X.; Xu, J.; Li, L.; and Sui, Z. 2023. A Survey on In-context Learning. arXiv:2301.00234.

Dufter, P.; and Schütze, H. 2020. Identifying Elements Essential for BERT's Multilinguality. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 4423-4437. Online: Association for Computational Linguistics.

Eronen, J.; Ptaszynski, M.; and Masui, F. 2023. Zero-shot cross-lingual transfer language selection using linguistic similarity. Information Processing \& Management, 60(3): 103250

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2020. Measuring Massive Multitask Language Understanding. CoRR, abs/2009.03300.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; and Chen, W. 2021. LoRA: Low-Rank Adaptation of Large Language Models. CoRR, abs/2106.09685.

Huang, H.; Tang, T.; Zhang, D.; Zhao, W. X.; Song, T.; Xia, Y.; and Wei, F. 2023a. Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting. arXiv:2305.07004.

Huang, W.; Abbeel, P.; Pathak, D.; and Mordatch, I. 2022. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. In Chaudhuri, K.; Jegelka, S.; Song, L.; Szepesvari, C.; Niu, G.; and Sabato, S., eds., Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 9118-9147. PMLR.

Huang, Y.; Bai, Y.; Zhu, Z.; Zhang, J.; and Zhang, J. 2023b. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. arXiv:2305.08322.

Ji, Y.; Deng, Y.; Gong, Y.; Peng, Y.; Niu, Q.; Ma, B.; and Li, X. 2023. BELLE: Be Everyone's Large Language model Engine. https://github.com/LianjiaTech/BELLE.

Jiang, Z.; Anastasopoulos, A.; Araki, J.; Ding, H.; and Neubig, G. 2020. X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 5943-5959. Online: Association for Computational Linguistics.

Joshi, P.; Santy, S.; Budhiraja, A.; Bali, K.; and Choudhury, M. 2020. The State and Fate of Linguistic Diversity and Inclusion in the NLP World. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 6282-6293. Online: Association for Computational Linguistics.

Katz, D. M.; Bommarito, M. J.; Gao, S.; and Arredondo, P. 2023. Gpt-4 passes the bar exam. Available at SSRN 4389233.

Khanuja, S.; Dandapat, S.; Srinivasan, A.; Sitaram, S.; and Choudhury, M. 2020. GLUECoS: An Evaluation Benchmark for Code-Switched NLP. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 3575-3585. Online: Association for Computational Linguistics.

Krishnan, J.; Anastasopoulos, A.; Purohit, H.; and Rangwala, H. 2021. Multilingual Code-Switching for ZeroShot Cross-Lingual Intent Prediction and Slot Filling. arXiv:2103.07792.

Li, H.; Koto, F.; Wu, M.; Aji, A. F.; and Baldwin, T. 2023. Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation. arXiv:2305.15011.

Lin, X. V.; Mihaylov, T.; Artetxe, M.; Wang, T.; Chen, S.; Simig, D.; Ott, M.; Goyal, N.; Bhosale, S.; Du, J.; Pasunuru, R.; Shleifer, S.; Koura, P. S.; Chaudhary, V.; O'Horo, B.; Wang, J.; Zettlemoyer, L.; Kozareva, Z.; Diab, M.; Stoyanov, V.; and Li, X. 2022. Few-shot Learning with Multilingual Language Models. arXiv:2112.10668.

Nguyen, X.-P.; Aljunied, S. M.; Joty, S.; and Bing, L. 2023. Democratizing LLMs for Low-Resource Languages by Leveraging their English Dominant Abilities with Linguistically-Diverse Prompts. arXiv:2306.11372.

OpenAI. 2022. Introducing ChatGPT.

OpenLMLab. 2023. Open-Chinese-LLaMA.

Penedo, G.; Malartic, Q.; Hesslow, D.; Cojocaru, R.; Cappelli, A.; Alobeidli, H.; Pannier, B.; Almazrouei, E.; and Launay, J. 2023. The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only. arXiv:2306.01116.

Pires, T.; Schlinger, E.; and Garrette, D. 2019. How Multilingual is Multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 4996-5001. Florence, Italy: Association for Computational Linguistics.

Ranta, A.; and Goutte, C. 2021. Linguistic Diversity in Natural Language Processing. Traitement Automatique des Langues, 62(3): 7-11.

Scao, T. L.; Fan, A.; Akiki, C.; Pavlick, E.; Ilić, S.; Hesslow, D.; and Castagné, R. 2023. BLOOM: A 176BParameter Open-Access Multilingual Language Model. arXiv:2211.05100.

StabilityAI. 2023. Announcing StableCode.

Tan, S.; and Joty, S. 2021. Code-Mixing on Sesame Street: Dawn of the Adversarial Polyglots. arXiv:2103.09593.

Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y.; Li, X.; Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023. Alpaca: A Strong, Replicable Instruction-Following Model.

Team, I. 2023a. Internlm: A multilingual language model with progressively enhanced capabilities.

Team, I. 2023b. InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities. https://github. com/InternLM/InternLM-techreport.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023a. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; and Almahairi, A. 2023b. Llama 2: Open Foundation and FineTuned Chat Models. arXiv:2307.09288.
Winata, G. I.; Cahyawijaya, S.; Liu, Z.; Lin, Z.; Madotto, A.; and Fung, P. 2021a. Are Multilingual Models Effective in Code-Switching? arXiv:2103.13309.

Winata, G. I.; Lin, Z.; and Fung, P. 2019. Learning Multilingual Meta-Embeddings for Code-Switching Named Entity Recognition. In Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019), 181-186. Florence, Italy: Association for Computational Linguistics.

Winata, G. I.; Lin, Z.; Shin, J.; Liu, Z.; and Fung, P. 2019. Hierarchical Meta-Embeddings for Code-Switching Named Entity Recognition. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3541-3547. Hong Kong, China: Association for Computational Linguistics.

Winata, G. I.; Madotto, A.; Lin, Z.; Liu, R.; Yosinski, J.; and Fung, P. 2021b. Language Models are Few-shot Multilingual Learners. In Proceedings of the 1st Workshop on Multilingual Representation Learning, 1-15. Punta Cana, Dominican Republic: Association for Computational Linguistics.

Wu, S.; and Dredze, M. 2019. Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 833-844. Hong Kong, China: Association for Computational Linguistics.

Wu, Z.; Papadimitriou, I.; and Tamkin, A. 2022. Oolong: Investigating What Makes Crosslingual Transfer Hard with Controlled Studies. arXiv:2202.12312.

Zhang, M.; Zhang, Q.; Zhang, Y.; and Gui, T. 2023a. LLMEVAL-1 Chinese Large Language Model Evaluation Phase 1.

Zhang, X.; Li, C.; Zong, Y.; Ying, Z.; He, L.; and Qiu, X. 2023b. Evaluating the Performance of Large Language Models on GAOKAO Benchmark. arXiv:2305.12474.

Zhong, W.; Cui, R.; Guo, Y.; Liang, Y.; Lu, S.; Wang, Y.; Saied, A.; Chen, W.; and Duan, N. 2023. AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models. arXiv:2304.06364.

Zhu, W.; Liu, H.; Dong, Q.; Xu, J.; Huang, S.; Kong, L.; Chen, J.; and Li, L. 2023. Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis. arXiv:2304.04675.
</end of paper 0>


<paper 1>
# Large Language Models are Good Spontaneous Multilingual Learners: Is the Multilingual Annotated Data Necessary? 

Shimao Zhang* ${ }^{\star}$, Changjiang Gao ${ }^{\star}$, Wenhao Zhu* ${ }^{*}$, Jiajun Chen* ${ }^{*}$, Xin Huang ${ }^{\curvearrowright}$,<br>Xue Han ${ }^{\diamond}$, Junlan Feng ${ }^{\diamond}$, Chao Deng ${ }^{\diamond}$, Shujian Huang ${ }^{\star}$ *<br>${ }^{\text {* National Key Laboratory for Novel Software Technology, Nanjing University }}$<br>${ }^{\diamond}$ China Mobile Research Beijing, China<br>\{smzhang,gaocj,zhuwh\}@smail.nju.edu.cn, \{huangsj,chenjj\}@nju.edu.cn<br>\{huangxinyjy, hanxueai, fengjunlan, dengchao\}@chinamobile.com


#### Abstract

Recently, Large Language Models (LLMs) have shown impressive language capabilities. However, most of the existing LLMs are all English-centric, which have very unstable and unbalanced performance across different languages. Multilingual alignment is an effective method to enhance the LLMs' multilingual capabilities. In this work, we explore the multilingual alignment paradigm which utilizes translation data and comprehensively investigate the spontaneous multilingual improvement of LLMs. We find that LLMs only instruction-tuned on question translation data without annotated answers are able to get significant multilingual performance enhancement even across a wide range of languages unseen during instruction-tuning. Additionally, we utilize different settings and mechanistic interpretability methods to comprehensively analyze the LLM's performance in the multilingual scenario. Our code and data is available at: https://github.com/Shimao-Zhang/ LLM-Multilingual-Learner.


## 1 Introduction

Large Language Models (LLMs) have recently shown impressive language capabilities across numerous downstream language tasks (Zhao et al., 2023). However, most existing LLMs are trained on extensive high-resource languages text such as English, Chinese, German, French, and so on (Touvron et al., 2023; Brown et al., 2020; Jiang et al., 2023), which lead to a significant performance gap between high-resource languages and low-resource languages (Huang et al., 2023; Zhang et al., 2023b; Gao et al., 2024). For the same task and question contents, using different languages for inputs may have a significant impact on the model's performance.

Some studies have conducted comprehensive exploration about how to enhance the capabilities[^0]

of LLMs for low-resource language. The classical approach typically follows the translate-based paradigm (Liu et al., 2024), which translating nonEnglish inputs into English or translating English data into non-English for instruction tuning. However, it is difficult to accurately translate all texts into the target low-resource language (Zhu et al., 2024), not to mention the increasing translation cost as the data scale expands. In order to enhance low-resource languages performance with less cost, some cross-lingual alignment and transfer strategies have been proposed (Eronen et al., 2023; Zhu et al., 2024; Zhao et al., 2024a). But all these methods rely on the data in the target language.

Meanwhile, some studies have explored Englishcentric LLMs, revealing that English also participate in the intermediate latent reasoning process of these models even when LLMs are prompted in non-English (Wendler et al., 2024; Zhao et al., 2024b). These findings suggest that for LLMs, multiple languages are not completely isolate, and LLMs have the capability to leverage the connections between various languages to address problems in multilingual scenarios. It further demonstrates the feasibility of cross-lingual capability transfer. More surprisingly, Kew et al. (2023) discover that when instruction-tuning LLMs with multilingual data, it is not necessary to instructiontune the model on data from all target languages to achieve a multilingual ability similar to models instruction-tuned on all languages.

Intuitively, LLMs have abilities and motivations to acclimatize themselves to multilingual environment (Shi et al., 2022). But what should we do to facilitate LLMs to do this better? Most existing methods rely on instruction-tuning on the corresponding datasets (Kew et al., 2023; Liu et al., 2024). However, given the strong capabilities of models in high-resource languages, which indicates that LLMs actually possesses the ability and knowledge to solve problems, such extensive additional anno-
tated data might not be necessary to help LLMs improve their multilingual abilities.

In this work, we further investigate the multilingual learning capabilities of LLMs, where we only train the LLMs on the parallel data without annotated answers (only queries) in a few languages. Following this pattern, we conduct the experiments on models in different types (English-centric or not) and parameter sizes, and test their multilingual capability across a wide range of languages on different benchmarks. We find that multilingual question alignment following Zhu et al. (2024) can effectively enhance the multilingual capabilities of LLMs. Our results indicate that only tuning on questions (without annotated answers) in a small number of languages can bring significant multilingual improvements even across many languages unseen during instruction-tuning process, which implies good language generalization. Furthermore, we also use logit lens (Nostalgebraist, 2020) and dimensionality reduction (Pearson, 1901; Hotelling, 1933) techniques to study the latent states of LLMs, providing more comprehensive perspectives and empirical results for understanding the multilingual learning of large language models.

## 2 Background

### 2.1 Unbalanced Multilingual Performance

With a much larger number of parameters pretrained on a massive corpus, LLMs have shown the impressive capabilities in a variety of language tasks (Zhao et al., 2023). These models are mainly pretrained on English data, which often accounts for $90 \%$ or even more of all training data, such as LLaMA2 (Touvron et al., 2023), GPT-3 (Brown et al., 2020), Mistral (Jiang et al., 2023), Falcon (Almazrouei et al., 2023), and so on. We present a partial language distribution of LLaMA-2's training data in Table 7 in Appendix A. Meanwhile, most of the LLMs also show unstable and unbalanced performance in multilingual scenarios, especially for some low-resource languages (Zhang et al., 2023a; Zhu et al., 2024). It's important to enable LLMs to adapt to a wider range of users and scenarios.

### 2.2 Cross-lingual Enhancement for Large Language Models

However, LLMs still exhibit significant shortcomings in multilingual scenarios. Many researchers propose multilingual LLMs that are specifically adjusted for multilingual tasks (Team, 2023; Le Scao et al., 2023; Wei et al., 2023). But for multilingual LLMs, researches indicate a decline in their performance in English because of the limited tokens and parameter size (Lin et al., 2022; Scao et al., 2022).

Based on the existing LLMs, researchers have made great efforts to enhancing the multilingual performance of LLMs, which include two categories: prompting close-source LLMs and instruction-tuning open-source LLMs. For the former, some studies utilize translation-based strategies which prompt ChatGPT to translate the nonEnglish input into English firstly before solving the problem (Huang et al., 2023; Qin et al., 2023). This type of approaches are constrained by the translation quality of the model itself and is cumbersome for users.

For the latter, LLMs shows significant improvement of multilingual and multitask abilities and good task generalization through multilingual multitask fine-tuning (Muennighoff et al., 2022). Chen et al. (2023) follow the translation-based approach and instruction-tune the model on a multilingual version of GSM8K, which is translated from English GSM8K (Cobbe et al., 2021). Liang et al. (2024) create a new intermediate language MUL (Machine-created Universal Language) as a translatable unified representation of shared concepts across different languages. "X-English" parallel translation data have also been widely used (Zhu et al., 2024). In our work, we mainly use this type of data, i.e. translation data between two different languages, to enhance multilingual alignment.

### 2.3 Mechanistic Interpretability

In addition to improving the performance of LLMs, it is also crucial to understand and explain the principles of neural networks and related methods explicitly. Current works mainly analyze LLMs' actions by observing the internal states during the inference process. Intermediate logits and neuron activation states are both important objects of observation.

Although the English-centric LLMs are mainly trained on English data, they also show good performance across some non-English languages (Shi et al., 2022). Logit lens (Nostalgebraist, 2020) is an early proposed technique that using the model head in the final layer to project the intermediate latent logits directly to the vocabulary space. It have been evidenced that LLaMA 2 (Touvron et al., 2023), a open-source English-centric LLMs, have explicit English output in its latent states even when
having non-English inputs (Wendler et al., 2024). There is also a hypothesis about how LLMs handle multilingualism that LLMs will solve task by English with the help of multilingual knowledge, and output in the target language finally (Zhao et al., 2024b). All these results indicate that there are connections between various languages for LLMs, and LLMs have the capability to spontaneously learn to utilize multiple languages to solve problems. Zhao et al. (2024b) calculate the overlapping ratio of the language-specific neurons of different languages in different layers. The results indicate that neurons belonging to different languages exhibit clear distribution differences. In our experiments, we utilize logit lens and dimensionality reduction techniques to help us better understand the mechanism behind our findings.

## 3 Methodology

We investigate the effect of question translation parallel data on LLMs' performance across a wide range of languages even unseen during the finetuning process.

We define the universal set of languages as $\mathbf{U}$ :

$$
\begin{equation*}
\mathbf{U}=\left\{l_{0}, l_{1}, l_{2}, \ldots, l_{n-1}\right\} \tag{1}
\end{equation*}
$$

where $l_{i}$ is the $i$-th language in $\mathbf{U}, n$ is the total number of languages. We let $l_{0}$ refer to English specially here.

We select a few of non-English languages $\mathcal{L}_{s}=$ $\left\{l_{i}, \ldots, l_{k}\right\} \subseteq \mathbf{U}$, and a target language $l_{t} \in \mathbf{U}$, $l_{t} \notin \mathcal{L}_{s}$. Then we will construct translation parallel data from every language $l \in \mathcal{L}_{s}$ to target language $l_{t}$. When construct the translation data, we only use the questions without annotated answers. Then we get a translation dataset $\mathcal{Q}_{\text {train }}$ including source question $\mathcal{Q}_{s}$ and the corresponding target question $\mathcal{Q}_{t}$, which means $\mathcal{Q}_{\text {train }}=\left\{\left(q_{s}, q_{t}\right) \mid\right.$ $q_{s} \in \mathcal{Q}_{s}$ and $\left.q_{t} \in \mathcal{Q}_{t}\right\}$. We instruct-tune the model on the translation task:

$$
\begin{equation*}
\underset{\theta}{\arg \min } \sum_{\left(q_{s}, q_{t}\right) \in \mathcal{Q}_{\text {train }}}-\log p_{\theta}\left(q_{t} \mid q_{s}\right) \tag{2}
\end{equation*}
$$

where $\theta$ is the model parameters, $\mathcal{Q}_{\text {train }}$ is the whole training translation dataset, $q_{s}$ is the question in the source language, $q_{t}$ is the question in the target language. Then we get the trained model:

$$
\begin{equation*}
\theta^{\prime}=\theta+\Delta \theta \tag{3}
\end{equation*}
$$

We use question translation data for training to eliminate the impact of annotated answers themselves. And we use in-context learning for test while the model haven't been trained on the corresponding task.

We test the trained model on all languages $l \in \mathbf{U}$. We construct the testing dataset $\mathcal{Q}_{\text {test }}=\left\{\mathcal{Q}_{l} \mid l \in\right.$ $\mathbf{U}\}$ for every language, where $\mathcal{Q}_{l}$ consists of all test questions in the language $l$.

$$
\begin{align*}
\text { Accuracy }_{l} & =\sum_{q \in \mathcal{Q}_{l}} \mathbf{I}_{\theta^{\prime}}(\hat{a}=a \mid q)  \tag{4}\\
\text { Accuracy } & =\frac{\sum_{l \in \mathbf{U}} \text { Accuracy }_{l}}{|\mathbf{U}|} \tag{5}
\end{align*}
$$

where $\mathbf{I}$ is a function that takes 1 when the proposition is true and 0 otherwise. $\mathcal{Q}_{l}$ denotes the test dataset of language $l$. $\mathbf{U}$ is the universal set of languages we use in our work. $\hat{a}$ is the answer that the model predicts base on $q$, and $a$ is the golden answer corresponding to $q$.

## 4 Experimental Setup

We conduct our experiments on both Englishcentric and non-English-centric models. And we utilize different representative tasks and different model parameter sizes to further strengthen our conclusions. In this section, we introduce our experimental settings in detailed.

Models We choose representative open-source LLMs for our experiments:

- Mistral: Mistral-7B-v0.1 (Jiang et al., 2023) is an advanced open-source English-centric large language model, which is one of the most popular open-source LLMs.
- Qwen: To enhance the generalization and reliability of our conclusions, we also choose models of different types and parameter sizes. Qwen 1.5 is a newly released and enhanced version of Qwen (Bai et al., 2023). Qwen is pretrained on a multilingual dataset with a significant portion of the data being in English and Chinese, which means it is not an English-centric model. We choose Qwen1.51.8B, Qwen1.5-4B, Qwen1.5-14B for our experiments.

Datasets Following Wendler et al. (2024), we select test tasks based on two fundamental principles:

1. Obvious Answers: Obvious answers reduce the entropy during inference process, minimizing the impact of irrelevant tokens on our analysis.

| Mistral-7B | en | $\mathbf{z h}$ | de | $\mathrm{fr}$ | es | it | nl | ja | $\mathbf{r u}$ | sv |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| base | 89.2 | 92.4 | 91.8 | 93.4 | 94.2 | 93.8 | 93.6 | 93.0 | 93.2 | 93.4 |
| $\mathrm{zh} \Rightarrow \mathrm{en}$ | 95.2 | 94.8 | 94.8 | 95.2 | 94.4 | 94.4 | 94.8 | 94.4 | 94.0 | 95.4 |
| $\mathrm{sw} \Rightarrow \mathrm{en}$ | 95.4 | 93.4 | 94.2 | 94.4 | 94.2 | 94.4 | 93.0 | 93.6 | 93.8 | 94.8 |
| $\mathrm{zh} / \mathrm{es} \Rightarrow \mathrm{en}$ | 95.2 | 95.0 | 95.0 | 95.0 | 94.8 | 92.8 | 94.6 | 95.0 | 94.4 | 94.8 |
| $\mathrm{zh} / \mathrm{de} \Rightarrow \mathrm{en}$ | 95.2 | 95.4 | 94.8 | 95.2 | 95.2 | 95.2 | 94.8 | 93.6 | 94.2 | 94.6 |
| $\mathrm{zh} / \mathrm{it} \Rightarrow$ en | 95.4 | 95.8 | 94.8 | 94.0 | 95.2 | 92.6 | 94.4 | 93.0 | 94.2 | 95.2 |
| sw/hi $\Rightarrow$ en | 95.4 | 94.6 | 94.4 | 93.4 | 93.4 | 93.6 | 93.6 | 94.0 | 93.8 | 94.4 |
| sw/th $\Rightarrow$ en | 95.4 | 95.0 | 93.8 | 93.4 | 93.4 | 92.8 | 93.6 | 92.6 | 93.2 | 94.0 |
| $\mathrm{zh} / \mathrm{es} / \mathrm{ru} \Rightarrow \mathrm{en}$ | 95.4 | 95.4 | 94.4 | 94.0 | 94.6 | 92.6 | 94.6 | 94.2 | 94.0 | 94.2 |
| $\mathrm{zh} / \mathrm{de} / \mathrm{it} \Rightarrow \mathrm{en}$ | 95.2 | 95.6 | 94.4 | 95.0 | 94.0 | 93.8 | 95.0 | 93.6 | 94.2 | 94.6 |
| Mistral-7B | $\overline{s l}$ | pl | bg | no | $\mathrm{ms}$ | is | hi | th | sW | bn |
| base | 87.6 | 93.2 | 91.6 | 92.4 | 91.8 | 63.2 | 81.6 | 83.0 | 58.0 | 71.0 |
| zh $\Rightarrow$ en | 94.0 | 94.0 | 94.6 | 92.2 | 89.0 | 84.0 | 88.8 | 88.4 | 75.8 | 81.0 |
| $\mathrm{sw} \Rightarrow \mathrm{en}$ | 89.8 | 92.6 | 93.6 | 93.4 | 90.0 | 72.0 | 64.4 | 51.4 | 81.2 | 54.0 |
| $\mathrm{zh} / \mathrm{es} \Rightarrow \mathrm{en}$ | 93.2 | 93.6 | 94.0 | 93.0 | 92.2 | 81.2 | 87.0 | 84.8 | 75.6 | 75.4 |
| $\mathrm{zh} / \mathrm{de} \Rightarrow \mathrm{en}$ | 93.4 | 94.0 | 94.6 | 93.6 | 92.2 | 86.6 | 84.8 | 88.4 | 71.8 | 68.6 |
| $\mathrm{zh} / \mathrm{it} \Rightarrow$ en | 92.6 | 93.8 | 94.2 | 93.6 | 92.6 | 84.2 | 77.6 | 77.2 | 71.6 | 60.0 |
| sw/hi $\Rightarrow$ en | 89.2 | 93.0 | 93.2 | 92.6 | 90.0 | 71.8 | 89.8 | 87.0 | 77.6 | 79.4 |
| sw/th $\Rightarrow$ en | 92.8 | 92.0 | 93.2 | 87.2 | 84.4 | 79.4 | 86.8 | 84.0 | 81.0 | 74.2 |
| $\mathrm{zh} / \mathrm{es} / \mathrm{ru} \Rightarrow \mathrm{en}$ | 93.6 | 94.2 | 93.4 | 93.4 | 91.4 | 83.8 | 85.0 | 86.0 | 77.0 | 76.0 |
| zh/de/it $\Rightarrow$ en | 91.2 | 93.6 | 94.2 | 93.4 | 91.8 | 83.2 | 77.2 | 82.4 | 69.0 | 71.4 |

Table 1: Accuracy of Mistral-7B base model and aligned models on the Amazon Reviews Polarity. We report at least two sets of results for each language quantity to strengthen our conclusions. The accuracy of randomly choosing is $50.0 \%$. " $\mathrm{X} / \mathrm{Y} / \mathrm{Z} \Rightarrow \mathrm{T}$ " means using a randomly mixed dataset including $10 \mathrm{k} \mathrm{X}$ to $\mathrm{T}, 10 \mathrm{k} \mathrm{Y}$ to $\mathrm{T}, 10 \mathrm{k} \mathrm{Z}$ to $\mathrm{T}$ translation data for instruction-tuning. We highlight the best results for every language.

2. Fixed Answers: Fixed answers (as opposed to open-ended responses) provide clearer observation targets, facilitating analysis through observing the latent outputs of the model. Deterministic outputs also make it easier for us to control the model's outputs.

Based on these, we conduct our experiments on two types of tasks:
- Emotion Classification: Emotion classification is an important and classic NLP task (Alswaidan and Menai, 2020), which always has three common outputs: "positive", "negative", and "neutral". We choose Amazon Reviews Polarity ${ }^{1}$ (Zhang et al., 2015), a famous dataset includes two classes "positive" and "negative", to construct the parallel data mentioned in $\S 2.2$ and the test data. We extract $10 \mathrm{~K}$ instances from train subset for parallel data and 500 instances from test subset for test data respectively.
- Natural Language Inference: Natural language inference (NLI) aims to judge the relationship between a given premise and a hypothesis sentence. There are always three[^1]

possible outputs: "entailment", "neutral", and "contradiction". We choose SNLI ${ }^{2}$ (Stanford Natural Language Inference) (Bowman et al., 2015) to conduct our experiments. Following the emotion classification task, we extract $10 \mathrm{~K}$ instances from train subset for parallel data and 600 instances from test subset for test data respectively.

Languages We conduct our following experiments across 20 languages in this work. As shown in Table 7 in Appendix A, we choose English (en), German (de), French (fr), Swedish (sv), Chinese (zh), Spanish (es), Russian(ru), Dutch (nl), Italian (it), and Japanese (ja) as the top 10 highest-resource languages according to Touvron et al. (2023). Additionally, we choose another 10 representative languages to strengthen our work, including Slovenian (sl), Polish (pl), Bulgarian (bg), Norwegian (no), Malay (ms), Icelandic (is), Hindi (hi), Thai (th), Swahili (sw), and Bengali (bn).

Implementations We all use LoRA (Hu et al., 2021) to instruction-tune the pre-trained models on mixed parallel data first. We train LLMs on the translation data excluding the golden answers to[^2]mitigate the impact of the data of the tasks themselves on the model's capabilities. For controlling the output more flexibly and the reproducibility, we use in-context learning rather than fine-tuning. We use constrained decoding for generation to eliminate the interference of irrelevant outputs on the results. Considering we mainly focus on the language understanding and task solving capabilities, we use English outputs uniformly if it is not specified.

More details are shown in Appendix B.

## 5 Results

In this section, we report the main results across different experimental settings and conduct some discussions based on the results.

### 5.1 Main Results

We report the accuracy of Mistral-7B on emotion classification task in Table 1. Clearly, we can see that the models trained on multilingual translation data outperform the original model significantly across a lot of languages, which indicates that model have much stronger multilingual capabilities after a multilingual training. We summarize our empirical findings as follows:

## 1. Large language models can learn to han-

 dle multilingualism better spontaneously. Traditionally, fine-tuning or alignment on the target languages is needed to help the model adapt. However, our results indicate that LLMs are able to perform effective learning and transfer across multiple languages without parallel data for most of them. As seen, models has much higher overall accuracy across 20 languages after training on data containing 2-4 languages.2. High-resource languages are not only good learners but also good leaders. Is there any difference when we use high-resource languages or low-resource languages in our training data? Our results in Table 1 show that three models trained on Swahili data achieve the top three highest accuracy on Swahili, while the accuracy on high-resource language is not significantly related to whether the corresponding language data is used. More importantly, training on high-resource language data enables the model to achieve more stable improvements across multiple languages compared to that on low-resource languages.
3. A few languages are enough for spontaneous multilingual learning. We select one, two, three languages with English for instruction-tuning respectively. As seen in Table 1, although using more languages sometimes leads to more stable improvements, model trained only on Chinese and English have achieved similar overall performance improvements. This is also consistent with the findings of Kew et al. (2023).
4. Our findings remain consistent across language models of different parameter sizes. We also present the average accuracy results of Qwen1.5-1.8B, Qwen1.5-4B, and Qwen1.5-14B in Table 2 to strengthen our conclusions. We find significant multilingual performance improvements across all of these models.

We have also validated our findings on the other task, Natural Language Inference (NLI). In this task, the model needs to determine the relationship between the hypothesis and the premise as entailment, neutral, or contradiction. We conduct our experiment on SNLI and report the accuracy of Qwen1.5-14B on natural language inference task in Table 3. Similar to the emotion classification task, we can see that models instruction-tuned on multilingual translation data significantly outperform the base model across these languages, which confirms that our findings have good generalization across different tasks.

### 5.2 Analysis

Building upon the above results, we conduct more comprehensive observations and analyses of the model's behavior.

English is not necessary as the target language in the training data. As elaborated in Section 4, we use outputs in English uniformly for all languages in our previous experiments. English has been widely used for multilingual transfer as a pivot language (Zhu et al., 2024; Hu et al., 2023). We further investigate the case of replacing English with Italian and report the results in Table 4. Mistral is an English-centric LLM and Qwen1.5 is not an English-centric LLM. From the results, we can find that using Italian as target language leads to different performances on different types of models. For English-centric LLM, using non-English language as target language has a negative impact on

| Model | Qwen1.5-1.8B | Qwen1.5-4B | Mistral-7B | Qwen1.5-14B |
| :--- | :---: | :---: | :---: | :---: |
| base | 68.35 | 79.52 | 87.07 | 86.27 |
| zh/es $\Rightarrow$ en | $\mathbf{7 6 . 1 3}$ | 81.99 | $\mathbf{9 0 . 8 3}$ | 91.53 |
| zh/de $\Rightarrow$ en | 74.23 | 82.64 | 90.81 | $\mathbf{9 2 . 2 5}$ |
| zh/it $\Rightarrow$ en | 75.70 | 83.32 | 89.10 | 92.13 |
| sw/hi $\Rightarrow$ en | 75.37 | $\mathbf{8 5 . 3 2}$ | 90.21 | 90.28 |

Table 2: Average accuracy of models of different parameter sizes on the Amazon Reviews Polarity. We highlight the best results for every model.

| Qwen1.5-14B | en | zh | de | fr | es | it | nl | ja | ru | sv |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| base | 84.50 | 83.50 | 74.17 | 75.17 | 81.17 | 78.67 | 78.17 | 51.17 | 76.83 | 76.17 |
| zh/es $\Rightarrow$ en | $\mathbf{9 2 . 5 0}$ | 84.67 | 82.67 | 82.83 | $\mathbf{8 5 . 5 0}$ | $\mathbf{8 3 . 8 3}$ | $\mathbf{8 4 . 6 7}$ | 57.00 | $\mathbf{8 2 . 6 7}$ | $\mathbf{8 4 . 3 3}$ |
| zh/de $\Rightarrow$ en | 91.83 | 84.50 | $\mathbf{8 3 . 6 7}$ | $\mathbf{8 4 . 5 0}$ | 85.00 | 83.67 | 84.5 | $\mathbf{5 7 . 1 7}$ | 81.67 | $\mathbf{8 4 . 3 3}$ |
| zh/it $\Rightarrow$ en | 91.67 | 83.83 | 80.83 | 82.67 | 84.00 | 80.00 | 83.50 | 55.83 | 81.50 | 83.33 |
| sw/hi $\Rightarrow$ en | 91.33 | $\mathbf{8 5 . 6 7}$ | 80.67 | 80.83 | 83.50 | 81.50 | 82.33 | 55.00 | 79.50 | 82.83 |
| Qwen1.5-14B | sl | pl | bg | no | ms | is | hi | th | sw | bn |
| base | 63.17 | 67.83 | 64.33 | $\mathbf{4 3 . 0 0}$ | 75.00 | 48.17 | 61.00 | 69.67 | 45.83 | 41.33 |
| zh/es $\Rightarrow$ en | 66.00 | 76.83 | 76.33 | 37.50 | $\mathbf{8 0 . 6 7}$ | $\mathbf{5 7 . 6 7}$ | 71.33 | $\mathbf{7 5 . 0 0}$ | $\mathbf{5 8 . 3 3}$ | 40.33 |
| zh/de $\Rightarrow$ en | $\mathbf{6 6 . 8 3}$ | 77.00 | $\mathbf{7 8 . 0 0}$ | 35.33 | 80.50 | 57.50 | $\mathbf{7 3 . 0 0}$ | $\mathbf{7 5 . 0 0}$ | $\mathbf{5 8 . 3 3}$ | $\mathbf{4 3 . 3 3}$ |
| zh/it $\Rightarrow$ en | 65.67 | $\mathbf{7 7 . 3 3}$ | 76.17 | 36.67 | 79.00 | 56.00 | 70.50 | 73.67 | 55.33 | 41.33 |
| sw/hi $\Rightarrow$ en | 63.00 | 76.67 | 72.17 | 39.67 | 80.33 | 54.17 | 67.67 | 74.67 | 56.83 | 41.33 |

Table 3: Accuracy of Qwen1.5-14B base model and trained models on the SNLI. We report all of the results on 20 languages. The accuracy of randomly choosing is $33.33 \%$. We highlight the best results for every language.

| Model | Qwen1.5-1.8B | Mistral-7B |
| :--- | :---: | :---: |
| base | 68.35 | 87.07 |
| zh/es $\Rightarrow$ it | 73.22 | 86.38 |
| zh/es $\Rightarrow$ en | $\mathbf{7 6 . 1 3}$ | $\mathbf{9 0 . 8 3}$ |

Table 4: Accuracy on Amazon Reviews Polarity. We replace English with Italian as the target language. Mistral is English-centric and Qwen1.5 is not English-centric.

| Model | Amazon Polarity | SNLI |
| :--- | :---: | :---: |
| base | 86.27 | 66.94 |
| $\mathrm{zh} / \mathrm{es} \Rightarrow$ en | 90.38 | 68.72 |
| $\mathrm{zh} / \mathrm{de} \Rightarrow$ en | 90.75 | 67.50 |
| $\mathrm{zh} / \mathrm{it} \Rightarrow$ en | 90.46 | 67.76 |
| $\mathrm{sw} / \mathrm{hi} \Rightarrow$ en | 90.53 | 65.76 |

Table 5: The model tested on Amazon Reviews Polarity is trained on SNLI questions. The model tested on SNLI is trained on Amazon Reviews Polarity questions.

the overall multilingual capabilities of the model. On the contrary, using Italian rather than English is also helpful for Qwen's multilingual performance improvement, though worse than using English beceuse of the model's worse capability of Italian than English.

It is not necessary but more beneficial to use the train subset corresponding to the test data as the source of translation data. Following Zhu

| Model | Same Language | Task-agnostic |
| :--- | :---: | :---: |
| base | 76.86 | 50.40 |
| zh/es $\Rightarrow$ en | 83.48 | $\mathbf{7 7 . 6 1}$ |
| zh/de $\Rightarrow$ en | 83.69 | 72.28 |
| $\mathrm{zh} / \mathrm{it} \Rightarrow$ en | 82.33 | 72.32 |
| $\mathrm{sw} / \mathrm{hi} \Rightarrow$ en | $\mathbf{8 4 . 5 9}$ | 74.92 |

Table 6: The results of Mistral-7B on emotion classification task for different output types. Same Language means the outputs in the same language with the inputs. Task-agnostic means using the task-agnostic outputs.

et al. (2024), in our previous experiments, we construct the parallel translation data for instructiontuning based on the train subset corresponding to the test dataset, which have the similar data characteristics and distributions. We further cross-test the Qwen1.5-14B trained on SNLI on Amazon Reviews Polarity and the Qwen1.5-14B trained on Amazon Reviews Polarity on SNLI. We report the results in Table 5. We can find that although the models trained on data with different distributions also have better overall performance in most cases, they have a worse performance than that trained on the data corresponding to the target task. That means the multilingual data is crucial for enhancing the model's multilingual capabilities, and similar types of data is more helpful. This is consistent with the "Superficial Alignment Hypothesis" (Zhou

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=656&width=1564&top_left_y=223&top_left_x=246)

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=263&width=494&top_left_y=234&top_left_x=267)

(a) Chinese Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=252&width=496&top_left_y=556&top_left_x=263)

(d) Chinese After

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=251&width=509&top_left_y=246&top_left_x=762)

(b) Japanese Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=226&width=522&top_left_y=578&top_left_x=744)

(e) Japanese After

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=265&width=534&top_left_y=233&top_left_x=1275)

(c) Russian Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-07.jpg?height=239&width=531&top_left_y=563&top_left_x=1274)

(f) Russian After

Figure 1: Logit lens on Mistral-7B in Chinese, Japanese and Russian scenarios. The horizontal axes is the layer num and the vertical axes is the probability. "en" (Orange) means the latent English output corresponding to the correct answer in the target language. "zh"/"ja"/"ru" (Blue) means the correct answer in the target language. "all_possible_out" (Cyan) means the probability of all possible outputs in the target language. "all_possible_latent" (Gray) means the probability of all possible outputs in English.

et al., 2024), which indicates that model learns knowledge and capabilities almost entirely in pretraining process, while alignment only guides the model to utilize the different "subdistribution of formats". So the data in the same subdistribution of formats is more beneficial.

## How about using outputs in different types?

 Except the outputs in English, we also conduct our experiments by using outputs in different types, including outputs in the same language with the inputs and task-agnostic outputs. When using outputs in the same language with the inputs, as shown in Table 6, the model also perform better after instruction-tuning, while performing worse compared to using English outputs (shown in Table 2) under the same settings. This confirms our conclusion in Section 4 that generating content in the target language is sometimes another great challenge for LLMs except understanding and solving multilingual problems themselves.We further replace "positive" with "ox" and replace "negative" with "horse" to investigate the cases of using task-agnostic outputs. We report the results in Table 6. Firstly, we can observe a significant decrease in multilingual performance of the base model when using task-agnostic outputs, which indicates that task-specific outputs are important for effective in-context learning (ICL). Clearly, we find a significant improvement in multilingual performance of the instruction-tuned models. By comparing the results before and after train- ing, we can find that our training greatly improves the model's ICL capability on the specific task, and this capability improvement exhibits excellent multilingual generalization. Based on the Superficial Alignment Hypothesis, we infer that the questions in only a few languages are able to effectively activate the corresponding subdistribution of formats across a wide range of languages.

## 6 Mechanistic Interpretability Analysis

In this section, we further utilize methods mentioned in $\S 2.3$ to analyze the model's changes before and after the training.

### 6.1 Logit Lens

Following Wendler et al. (2024), we utilize logit lens to analyze the changes of the model. We utilize logit lens on Qwen1.5, a series of LLMs that are not English-centric, and find there is not English latent outputs in the intermediate layers. And the prefix token overlapping between target language and English will also bring errors to the results. So we choose Chinese, Japanese and Russian as three representative languages for our experiment, which shows significant improvement in our results before. Following Wendler et al. (2024), we use the outputs in the same language with the inputs (results shown in Table 6). We conduct our experiments on Mistral-7B and its best trained version "sw/hi $\Rightarrow$ en" in Table 6. We report the results in Figure 1. Clearly, we can observe the following points: (1) All models generate latent English out-

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=291&width=391&top_left_y=237&top_left_x=250)

(a) Layer 20 Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=286&width=394&top_left_y=588&top_left_x=271)

(e) Layer 20 After

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=283&width=391&top_left_y=247&top_left_x=638)

(b) Layer 25 Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=282&width=377&top_left_y=590&top_left_x=657)

(f) Layer 25 After

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=280&width=391&top_left_y=248&top_left_x=1027)

(c) Layer 30 Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=286&width=391&top_left_y=591&top_left_x=1027)

(g) Layer 30 After

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=280&width=380&top_left_y=248&top_left_x=1426)

(d) Layer 32 Before

![](https://cdn.mathpix.com/cropped/2024_06_04_166820a330003228de19g-08.jpg?height=286&width=382&top_left_y=591&top_left_x=1408)

(h) Layer 32 After

Figure 2: PCA (Principal Component Analysis) on Mistral-7B in English, German, French and Hindi scenarios. Before means the base model. After means the trained model. All logits are mapped into the twodimensional representation. Each point in the plot corresponds to one instance.

put before generating outputs in the target language finally; (2) The proportion of the probability of the correct answer increases in the sum of all possible answer probabilities; (3) The probability of all other possible answers (except correct answer) in the latent English outputs is nearly zero; (4) The area of latent English output significantly increases, which means the trained models perform inference in English better.

### 6.2 Principal Component Analysis

We further utilize the dimensionality reduction technique to visualize the intermediate layer latent outputs of the model across different languages. PCA (Principal Component Analysis) is a normal linear dimensionality reduction technique (Pearson, 1901; Hotelling, 1933), which can be used in some scenarios where logit lens doesn't work. Principal components are a few linear combinations of the original variables that maximally explain the variance of all the variables (Greenacre et al., 2022). We utilize PCA to map the latent logits into the two-dimensional representation. Based on the patterns shown in Figure 1, we report layer 20, layer 25, layer 30 and the last layer as four representative layers in Figure 2. We have the following findings: (1) The points of different languages follow the similar patterns in layer 20 and layer 25 , where English latent outputs have appeared and outputs in the target language haven't appeared. We further calculate the Pearson correlation coefficient of 1 dimension PCA results (Appendix C). There is a strong linear correlation between representations of different languages, which also indicates an uniform latent representation pattern during inference process; (2) Representations belong to different languages exhibit greater differences from each other in the trained model; (3) The results of the last layer is similar because of the same possible outputs; (4) Based on Pearson coefficient reported in Appendix C, the correlation between Hindi (low-resource language) and other languages (high-resource language) significantly improves.

## 7 Conclusion

In this paper, we find that LLMs only trained on translation data without annotated answers are able to get a significant multilingual performance improvement even across a wide range of unseen languages. We conduct experiments on different models, different benchmarks and 20 different languages. Our results indicate that using question translation parallel data can significantly enhance the in-context learning capabilities of LLMs. And these improvements demonstrate excellent model and language generalization. Furthermore, we also conduct comprehensive analysis based on some mechanistic interpretability methods, including logit lens and PCA dimensionality reduction technique. Our work demonstrates the enormous potential of LLMs for efficient multilingual capability improvement. We hope our work can inspire the community to further explore this promising direction for the better multilingual alignment.

## 8 Limitations

We aim to draw more attention to the multilingual alignment which is a promising research direction. Despite our work has demonstrated the LLMs' strong capability of multilingual generalization and the great potential of efficient multilingual enhancement, there are still some limitations waiting for research. Because we investigate the models trained on question translation data without annotated answers in our work, we utilize few-shot learning to help model handle the target task better. Analyzing the models which haven't been instruction-tuned on the target task properly in a zero-shot setting would further strengthen the conclusions.

Due to the limited resources, we conduct our experiments on different LLM scale from 1.8B to 14B in this work. We are more than willing to verify our conclusions on larger LLMs (70B or larger) if more resources are available in the future. Meanwhile, we mainly utilize automatic translation engine in our work because of the limited resources, while translation data annotated by native speakers would be more accurate.

## References

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. 2023. The falcon series of open language models. arXiv preprint arXiv:2311.16867.

Nourah Alswaidan and Mohamed El Bachir Menai 2020. A survey of state-of-the-art approaches for emotion recognition in text. Knowledge and Information Systems, 62(8):2937-2987.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.

Nuo Chen, Zinan Zheng, Ning Wu, Linjun Shou, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. 2023. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. arXiv preprint arXiv:2310.20246.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Juuso Eronen, Michal Ptaszynski, and Fumito Masui. 2023. Zero-shot cross-lingual transfer language selection using linguistic similarity. Information Processing \& Management, 60(3):103250.

Changjiang Gao, Hongda Hu, Peng Hu, Jiajun Chen, Jixing Li, and Shujian Huang. 2024. Multilingual pretraining and instruction tuning improve cross-lingual knowledge alignment, but only shallowly. arXiv preprint arXiv:2404.04659.

Michael Greenacre, Patrick JF Groenen, Trevor Hastie, Alfonso Iodice d'Enza, Angelos Markos, and Elena Tuzhilina. 2022. Principal component analysis. Nature Reviews Methods Primers, 2(1):100.

Harold Hotelling. 1933. Analysis of a complex of statistical variables into principal components. Journal of educational psychology, 24(6):417.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, et al. 2023. Large multilingual models pivot zero-shot multimodal learning across languages. arXiv preprint arXiv:2308.12038.

Haoyang Huang, Tianyi Tang, Dongdong Zhang, Wayne Xin Zhao, Ting Song, Yan Xia, and Furu Wei. 2023. Not all languages are created equal in llms: Improving multilingual capability by cross-lingual-thought prompting. arXiv preprint arXiv:2305.07004.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Tannon Kew, Florian Schottmann, and Rico Sennrich. 2023. Turning english-centric llms into polyglots: How much multilinguality is needed? arXiv preprint arXiv:2312.12683.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2023. Bloom: A 176bparameter open-access multilingual language model.

Yaobo Liang, Quanzhi Zhu, Junhe Zhao, and Nan Duan. 2024. Machine-created universal language for crosslingual transfer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages $18617-18625$.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. 2022. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9019-9052.

Chaoqun Liu, Wenxuan Zhang, Yiran Zhao, Anh Tuan Luu, and Lidong Bing. 2024. Is translation all you need? a study on solving multilingual tasks with large language models. arXiv preprint arXiv:2403.10258.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786.

Nostalgebraist. 2020. interpreting gpt: the logit lens. https://www. lesswrong.com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens.

Karl Pearson. 1901. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559-572.

Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. 2023. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. arXiv preprint arXiv:2310.14799.

Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari, Stella Biderman, Hady Elsahar, Niklas Muennighoff, Jason Phang, et al. 2022. What language model to train if you have one million gpu hours? arXiv preprint arXiv:2210.15424.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, et al. 2022. Language models are multilingual chain-of-thought reasoners. arXiv preprint arXiv:2210.03057.

InternLM Team. 2023. Internlm: A multilingual language model with progressively enhanced capabilities.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.
Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023. Polylm: An open source polyglot large language model. arXiv preprint arXiv:2307.06018.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in english? on the latent language of multilingual transformers. arXiv preprint arXiv:2402.10588.

Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, and Grzegorz Kondrak. 2023a. Don't trust chatgpt when your question is not in english: A study of multilingual abilities and types of llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7915-7927.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.

Zhihan Zhang, Dong-Ho Lee, Yuwei Fang, Wenhao Yu, Mengzhao Jia, Meng Jiang, and Francesco Barbieri. 2023b. Plug: Leveraging pivot language in cross-lingual instruction tuning. arXiv preprint arXiv:2311.08711.

Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024a. Llama beyond english: An empirical study on language capability transfer. arXiv preprint arXiv:2401.01055.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing. 2024b. How do large language models handle multilingualism? arXiv preprint arXiv:2402.18815.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2024. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36.

Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch. 2024. Question translation training for better multilingual reasoning. arXiv preprint arXiv:2401.07817.
</end of paper 1>


<paper 2>
# 8-Aya 23: Open Weight Releases to Further Multilingual Progress 

Viraat Aryabumi ${ }^{* 1}$, John Dang ${ }^{1}$, Dwarak Talupuru ${ }^{2}$,<br>Saurabh Dash ${ }^{1}$, David Cairuz ${ }^{2}$, Hangyu Lin ${ }^{2}$, Bharat Venkitesh ${ }^{2}$,<br>Madeline Smith ${ }^{1}$, Jon Ander Campos ${ }^{2}$, Yi Chern Tan ${ }^{2}$,<br>Kelly Marchisio ${ }^{2}$, Max Bartolo ${ }^{2}$, Sebastian Ruder ${ }^{2}$, Acyr Locatelli $^{2}$,<br>Julia Kreutzer ${ }^{1}$, Nick Frosst ${ }^{2}$, Aidan Gomez ${ }^{2}$, Phil Blunsom ${ }^{2}$,<br>Marzieh Fadaee ${ }^{1}$, Ahmet Üstün ${ }^{* 1}$, and Sara Hooker ${ }^{* 1}$<br>${ }^{1}$ Cohere For AI, ${ }^{2}$ Cohere

Corresponding authors: Viraat Aryabumi <viraat@cohere.com >, Ahmet Üstün [ahmet@cohere.com](mailto:ahmet@cohere.com), Sara Hooker [sarahooker@cohere.com](mailto:sarahooker@cohere.com)


#### Abstract

This technical report introduces Aya 23, a family of multilingual language models. Aya 23 builds on the recent release of the Aya model [Üstün et al., 2024], focusing on pairing a highly performant pre-trained model with the recently released Aya collection [Singh et al., 2024]. The result is a powerful multilingual large language model serving 23 languages, expanding state-of-art language modeling capabilities to approximately half of the world's population. The Aya model covered 101 languages whereas Aya 23 is an experiment in depth vs breadth, exploring the impact of allocating more capacity to fewer languages that are included during pre-training. Aya 23 outperforms both previous massively multilingual models like Aya 101 for the languages it covers, as well as widely used models like Gemma, Mistral and Mixtral on an extensive range of discriminative and generative tasks. We release the open weights for both the $8 \mathrm{~B}$ and 35B models as part of our continued commitment for expanding access to multilingual progress.


Aya-23-8B: https://huggingface.co/CohereForAI/aya-23-8B

Aya-23-35B: https://huggingface.co/CohereForAI/aya-23-35B

## 1 Introduction

In this work we introduce Aya 23, a family of multilingual instruction-tuned language models supporting 23 languages based on Cohere's Command model ${ }^{1}$ and the Aya multilingual instructionstyle collection [Singh et al., 2024]. To date, the majority of progress in large language modeling has been English-centric, leading to models which perform poorly outside of a handful of languages. This can result in cliffs in model performance in languages not included in pre-training [Schwartz[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-02.jpg?height=502&width=1632&top_left_y=256&top_left_x=236)

Figure 1: Multilingual benchmark results covering 5 task categories from 8 datasets for Aya 23 models against massively multilingual Aya-101-13B and widely used open weight models of similar size such as Bacterian-X-7B, Gemma-1.1-7B-it, Mistral-7B-Inst-v0.2 and Mixtral-8x7B-Inst.

et al., 2022; Kotek et al., 2023; Khandelwal et al., 2023; Vashishtha et al., 2023; Khondaker et al., 2023], the introduction of security flaws for all users, [Yong et al., 2023a; Nasr et al., 2023; Li et al., 2023b; Lukas et al., 2023; Deng et al., 2023] and a growing divide in the cost of technology due to high latencies for generations outside of English [Held et al., 2023; Durmus et al., 2023; Nicholas \& Bhatia, 2023; Ojo et al., 2023; Ahia et al., 2023].

Multilingual efforts including the release of Aya 101 [Üstün et al., 2024], BLOOMZ [Muennighoff et al., 2023] and mT0 [Muennighoff et al., 2023] models have made great strides in expanding access to modern natural language processing technologies for the world. However, there still remains significant room for improvement relative to first-class citizen languages like English and Chinese. Two major hurdles in the development of powerful multilingual models are (1) the lack of robust multilingual pretrained models, and (2) the scarcity of instruction-style training

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-02.jpg?height=385&width=805&top_left_y=1301&top_left_x=1061)

Figure 2: Average win-rates (\%) across 10 languages for Aya 23 models against widely used open weight models of similar size. data covering a diverse set of languages.

The Aya initiative ${ }^{2}$ was created to address the aforementioned data scarcity issues by creating and releasing the largest multilingual instruction-style dataset [Singh et al., 2024] to date, along with the Aya 101 model [Üstün et al., 2024]. Aya 101 was a step forward in massively multilingual language modeling, creating a 101 languages state-of-the-art instruction fine-tuned LLM. However, Aya 101 was by necessity built upon the mT5 [Xue et al., 2020] pre-trained base model given it was one of the few pre-trained models that had been trained on 101 languages. mT5 is relatively outdated given the rapid advances in LLM technology since its release in 2019. Its major limitations are: 1) Outdated knowledge: Having been pre-trained several years ago, mT5 is not as useful for interactions about events that occurred recently. 2) Inadequate Performance: There are many[^1]stronger models now compared to when mT5 was released, such as the Command $\mathrm{R}+{ }^{3}$, Command $\mathrm{R}^{4}$, Llama series [Touvron et al., 2023a;b], Mistral models [Jiang et al., 2023; 2024] and Gemma models [Gemma-Team, 2024].

Furthermore, Aya 101 was a 13-billion parameter model designed for breadth, expanding coverage to nearly double that achieved by previous models with 101 languages. Due to the well-documented curse of multilinguality [Arivazhagan et al., 2019; Conneau et al., 2019; Pfeiffer et al., 2022], models attempting to serve such a broad variety of languages often lag in generative performance on any given language relative to models dedicated to serving a more focused subset, because of the need to share model capacity so widely. For Aya 23, we instead balance breadth and depth, exploring the impact of allocating more capacity to fewer languages (23 languages) that are included during pre-training, alleviating the "curse" and leading to large gains over the original Aya 101 and widely used models such as Gemma [Gemma-Team, 2024], Mistral [Jiang et al., 2023], and Mixtral [Jiang et al., 2024] for the corresponding 23 languages.

In this technical report, we assess the performance of Aya 23 models following the comprehensive multilingual evaluation framework proposed by Üstün et al. [2024]. In our evaluation, we focus on 23 languages that are covered by the new Aya model family. These 23 languages are: Arabic, Chinese (simplified $\mathcal{S}$ traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian and Vietnamese. Our choice of languages was guided to align with the languages present in pre-training of Command R, due to known difficulties of introducing new languages after pre-training [Zhao et al., 2024; Yong et al., 2023b].

We release Aya 23 in two model sizes: 8-billion (8B) and 35-billion (35B) parameters. Aya-23-35B achieves the highest results across all the evaluation tasks and languages covered, while Aya-23-8B demonstrates best-in-class multilingual performance which is crucial given that model sizes above 13B parameters limit model usability on consumer-grade hardware. We note that relative to Aya 101, Aya 23 improves on discriminative tasks by up to $14 \%$, generative tasks by up to $20 \%$, and multilingual MMLU by up to $41.6 \%$. Furthermore, Aya 23 achieves a 6.6x increase in multilingual mathematical reasoning compared to Aya 101. Across Aya 101, Mistral, and Gemma, we report a mix of human annotators and LLM-as-a-judge comparisons. Across all comparisons, the Aya-23-8B and Aya-23-35B are consistently preferred. By releasing the weights of the Aya 23 model family, we hope to empower researchers and practitioners to advance multilingual models and applications.

## 2 Pre-trained Models

The Aya 23 model family is based on the Cohere Command series models which are pre-trained using a data mixture that includes texts from 23 languages. In particular, Aya-23-35B is a further finetuned version of Cohere Command R. For pre-trained models, a standard decoder-only Transformer architecture is used with the following setup:

1. Parallel Attention and FFN layers: Similar to PALM-2 [Anil et al., 2023] we use a parallel block architecture that leads to a significant improvement in training efficiency without hurting[^2]

|  | Embedding <br> dims | Num <br> layers | FFN hidden <br> dims | Num <br> heads | Num KV <br> heads | Head <br> size | Vocab <br> size | Embedding <br> parameters | Non-embedding <br> parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Aya-23-8B | 4096 | 32 | 22528 | 32 | 8 | 128 | 256000 | $1,048,576,000$ | $6,979,457,024$ |
| Aya-23-35B | 8012 | 40 | 45056 | 64 | 64 | 128 | 256000 | $2,097,152,000$ | $32,883,679,232$ |

Table 1: Architecture parameters for Aya 23 model family

model quality, especially in tensor-parallel (TP) settings.

2. SwiGLU Activation: We found SwiGLU [Shazeer, 2020] to have higher downstream performance than other activations. We scale the dimensions of FFN layers to retain approximately the same number of trainable parameters compared to non-SwiGLU activation functions.
3. No bias: Similar to PALM2 [Anil et al., 2023], we remove all biases from dense layers to improve the training stability.
4. RoPE: We use rotary positional embeddings [Su et al., 2021] to provide better long context extrapolation. Furthermore, it also achieves better downstream task performance for short context lengths compared to other relative positional encoding methods such as ALiBi [Press et al., 2021].
5. Tokenizer: We use a BPE tokenizer of size $256 \mathrm{k}$. We perform NFC normalization and digits are split into individual tokens. The tokenizer is trained on a subset of our pre-training datasets balanced to ensure efficient representations across languages.
6. Grouped Query Attention (GQA): Aya-23-8B uses grouped-query attention [Ainslie et al., 2023] where each KV head shares multiple $\mathrm{Q}$ heads to reduce inference-time memory footprint.

All base models are trained using Fax [Yoo et al., 2022], a Jax-based distributed training framework on TPU v4 chips [Jouppi et al., 2023]. A combination of parallelism strategies is used to ensure high training throughput. We split the available device mesh into data and model parallel submeshes. The model parameters and optimizer states are sharded on the model submesh and replicated along data submesh. This avoids increasing the communication costs during the forward and backward passes by limiting the number of chips holding the shards of the model and the optimizer state. We refer to Table 1 for all key architecture parameters.

## 3 Instruction Fine-Tuning

### 3.1 Data mixture

We adopt the multilingual instruction data described in Üstün et al. [2024] for fine-tuning the pre-trained models. Given the scarcity of multilingual instruction data, these fine-tuning datasets combine a range of approaches to improve the availability of data. This includes relying on extensive efforts to aggregate and prune multilingual templates and hard-to-find human annotations curated by fluent speakers of various languages. Moreover, it also extends to data augmentation strategies such as machine translation and leveraging synthetic data generation coupled with translation.

We briefly describe each source below:

| Prompt: | <BOS_TOKEN><\|START_OF_TURN_TOKEN $\|><\| U S E R \_T O K E N \mid>$ <br> Hello, how are you?<\|END_OF_TURN_TOKEN $\mid>$ |
| :---: | :---: |
| Completion: | $<\mid$ START_OF_TURN_TOKEN $\|><\|$ CHATBOT_TOKEN $\mid>$ <br> I am doing good!<\|END_OF_TURN_TOKEN $\mid>$ |

Table 2: Example prompt-completion pair with the chat-format for the Aya-23 models. The formatting allows indication of roles (user, chatbot), and delineation of turns.

1. Multilingual Templates: We use structured text to transform specific NLP datasets into instruction and response pairs. This set of data includes samples from the $\mathrm{xP} 3 \mathrm{x}$ dataset [Üstün et al., 2024], the data provenance collection [Longpre et al., 2023b], and the Aya collection [Singh et al., 2024]. The final collection consists of $55.7 \mathrm{M}$ examples which consists of zero and few-shot examples, covering 23 languages and 161 different datasets [Üstün et al., 2024].
2. Human Annotations: The Aya dataset [Singh et al., 2024] has a total of 204K humancurated prompt-response pairs written by native speakers in 65 languages. We filter this data for 23 languages we train on, resulting in $55 \mathrm{~K}$ samples.
3. Translated Data: We use the translated subset of Aya collection [Singh et al., 2024] which open-sources translations of widely used English instruction datasets [Longpre et al., 2023b] filtered for the languages we train on. This collection includes, among others, translations of HotpotQA [Yang et al., 2018] and Flan-CoT-submix [Longpre et al., 2023a]. We randomly sample a subset of up to 3,000 instances for each language for each dataset to preserve instancelevel diversity. We filter this data to the 23 languages we train on, resulting in a subset of 1.1M examples.
4. Synthetic Data: We construct synthetic fine-tuning data similar to Üstün et al. [2024] using human-annotated prompts from ShareGPT ${ }^{5}$ and Dolly-15k [Conover et al., 2023b]. ${ }^{6}$ Unlike Üstün et al. [2024], we use Cohere's Command $\mathrm{R}+$ to natively generate multilingual responses for the translated ShareGPT and Dolly prompts in all 23 languages, resulting in 1.63M examples. We note that Cohere's terms of use ${ }^{7}$ prohibit training on model generations. However, we received a special exception for these releases of Aya models.

The Aya fine-tuning mix emphasizes available supervised datasets with self-reported commercially permissive licenses. We use the filtering tools from the Data Provenance Initiative [Longpre et al., $2023 b]$ to ensure appropriate provenance.

### 3.2 Training details

For instruction fine-tuning, we fine-tune the base models for 13,200 update steps using an 8192 context length with data packing enabled, corresponding to approximately $10.5 \mathrm{M}$ training samples. We use the Adam optimizer [Kingma \& Ba, 2014] with a cosine schedule learning rate, with a peak[^3]

| Task | Dataset | Metric |  | Unseen Task | $\overline{\text { Languages }}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DiSCRIMINATIVE TASKs |  |  |  |  |  |
| Coreference Resolution | XWinograd [Muennighoff et al., 2023] | 0 -shot | Acc. | $\checkmark$ | 6 |
| Sentence Completion | XCOPA [Ponti et al., 2020] | 0 -shot | Acc. | $v$ | 11 |
|  | XStoryCloze [Lin et al., 2021] | 0 -shot | Acc. | $\checkmark$ | 10 |
| Language Understanding | M-MMLU [Dac Lai et al., 2023] | 5 -shot | Acc. | $\bar{v}$ | 14 |
| Generative Tasks |  |  |  |  |  |
| Translation | FLORES-200 [Goyal et al., 2021; NLLB-Team et al., 2022] | 0 -shot | spBLEU | $x$ | 23 |
| Summarization | XLSum [Hasan et al., 2021] | 0 -shot | RougeL | $x$ | 15 |
| Mathematical Reasoning | MGSM [Shi et al., 2023] | 5 -shot | Acc. | $\bar{x}$ | 7 |
| Open-Ended Generation | Dolly Human-edited \& Machine-translated [Singh et al., 2024] | 0 -shot | win-rate | $\bar{x}$ | 5 |

Table 3: Datasets considered for evaluation. Unseen Task refers to tasks entirely excluded from training, which includes the 4 discriminative tasks. Additionally, we include multilingual MMLU as an unseen dataset. The seen tasks refer to the generative tasks where supervised training is performed and instances are held-out (validation and test splits) for evaluation. We limit the evaluation languages only to the ones that are included in 24 languages, except for the first 3 datasets (XWinograd, XCOPA, XStoryCloze) where we use all the available languages.

LR of $6 \times 10^{-4}$, an end LR of $6 \times 10^{-5}$ and a batch size of 64 . For all training runs, we use TPUv4 with up to 128 pod slices.

Similar to other instruction-tuned models [Gemini Team et al., 2024], the examples used to instructiontune Aya 23 are formatted using special tokens to include extra information (an example is shown in Table 2). The formatting allows indication of roles (user, chatbot), and delineation of turns. This formatting is used both during instruction-tuning and inference. While it is possible to obtain coherent generations without using the formatting, generation quality suffers without it. While we use the chat formatting, the model is a single-turn instruction-following model and is not optimized explicitly for chat mode use.

## 4 Multilingual Evaluation

To measure our models' performance, we follow the comprehensive evaluation framework introduced in Üstün et al. [2024]. Different from Üstün et al. [2024], we use eval-harness [Gao et al., 2023] to evaluate all the models for discriminative tasks, multilingual MMLU, and MGSM. ${ }^{8}$ This includes assessing performance on:

1. Completely unseen discriminative tasks: We evaluate on XWinograd [Muennighoff et al., 2023], XCOPA [Ponti et al., 2020], and XStoryCloze [Lin et al., 2021]. ${ }^{9}$ We use zero-shot evaluation. Note that these evaluation tasks are completely unseen and there is no dataset in the training mixture from the same task categories.
2. General purpose language understanding: We use Multilingual MMLU [Dac Lai et al., 2023] where the dataset is not seen during the training (5-shot evaluation) to evaluate Aya[^4]models' general language understanding. The dataset is a version of English MMLU [Hendrycks et al., 2020] translated into 31 languages using ChatGPT. The original English MMLU contains 13,062 questions consisting of 57 different tasks, covering a wide range of topics including STEM, humanities, and the social sciences. We use the 14 languages that are covered by Aya 23 models for evaluation.
3. Multilingual mathematical reasoning: We use Multilingual Grade School Math (MGSM) Benchmark [Shi et al., 2023] to evaluate multilingual mathematical reasoning. MGSM consists of 250 problems from the GSM8K benchmark [Cobbe et al., 2021], which are human-translated into 10 languages. We pick the subset of MGSM languages, which are covered by Aya 23 models. We use questions with answers followed by CoT prompt (5-shot) in the same language (native_cot) and strict-match score as the evaluation metric following Shi et al. [2023].
4. Generative tasks: We evaluate model performance in machine translation and summarization on FLORES-200 [NLLB-Team et al., 2022] and XLSum [Hasan et al., 2021] respectively. For FLORES, we use all 21 languages ( $\mathrm{X} \leftrightarrow$ English) and for XLSum, we use 15 languages based on language coverage of Aya 23 models.
5. Preference evaluation: We assess the open-ended generation capabilities of the models through human- and LLM-simulated evaluation using the (1) dolly-machine-translated test set Singh et al. [2024] which is a held-out test set of 200 instances from the Dolly-15k dataset [Conover et al., 2023b] translated into 101 languages. This test set was curated by multiple annotators to avoid the inclusion of any culturally specific or geographic references, intending to minimize estimations of performance that require specific cultural or geographic knowledge. We also evaluate on the (2) dolly-human-edited test set Singh et al. [2024] consisting of improved versions of the dolly-machine-translated test set for 6 languages (French, Spanish, Serbian, Russian, Arabic, Hindi) post-edited by professional compensated human annotators to correct any possible translation issues.

For open-ended evaluation, we rely on both LLM-simulated win-rates and human evaluation. We detail the protocol for each briefly below:

(a) LLM-simulated win-rates: Consistent with Üstün et al. [2024] and other recent works [Rafailov et al., 2023; Dubois et al., 2023; Kim et al., 2023], we use GPT-4 ${ }^{10}$ as a proxy judge. We measure pairwise win rates between Aya 23 models with Aya 101, Gemma1.1-7b-it, and Mixtral-8x7b-Instruct-v0.1 on 10 languages (English, Chinese, Turkish, Spanish, Russian, Hindi, French, and Arabic, Japanese, Portuguese). We use the same prompt for eliciting GPT-4 preferences as specified by Üstün et al. [2024]. For languages where there is dolly-human-edited coverage, we default to these prompts given that they were edited for translation-induced issues by professional annotators.

(b) Human evaluation of preferences: We ask compensated professional annotators in five languages (Russian, Hindi, French, Spanish, English) to select their preferred model completions for the dolly-human-edited test set and original English Dolly test prompts, respectively. The annotation setup (raters, instructions) is the same setup used by Üstün et al. [2024]. Each pair of generations is rated once; ties ("both bad" or "both good") are allowed but discouraged.

6. Safety, Toxicity \& Bias: We evaluate the safety of model generations under adversarial prompts from the multilingual AdvBench [Yong et al., 2023a] benchmark representing multiple[^5]angles of harm, such as crime, physical harm, and misinformation. GPT-4 is used as an automatic evaluator for harmfulness on 120 test prompts. The reliability of GPT-4 for this evaluation was previously confirmed by Üstün et al. [2024]. In addition, we measure toxicity and bias towards identity groups with the multilingual identity description prompts from Üstün et al. [2024]. We sample $k=25$ model completions for each prompt, and evaluate their toxicity with Perspective API. ${ }^{11}$

### 4.1 Model Comparisons

We evaluate against multiple open-source massively multilingual models to ensure a comprehensive evaluation. We select models based on architecture, size, base model type, and the extent of coverage of languages. The selected models cover a range of sizes (7B to 46B), base models (mT5, Llama, Gemma, Mistral), languages, and training regimes (SFT and preference tuning).

Details of each model are below:

- Aya-101-13B [Üstün et al., 2024] is a 13B parameter mT5 model [Muennighoff et al., 2023] fine-tuned on xP3x [Üstün et al., 2024], Aya collection [Singh et al., 2024], Data Provenance collection [Longpre et al., 2023b], and ShareGPT-Command [Üstün et al., 2024] for 101 languages. Aya 101 is a state-of-art massively multilingual instruction-tuned LLM that covers the largest number of languages in our comparison.
- Bactrian-X-7B [Li et al., 2023a] is a LLaMA-7B model [Touvron et al., 2023a] fine-tuned on the Bactrian-X dataset which contains $3.4 \mathrm{M}$ pairs of instructions and responses in 52 languages. This dataset was automatically constructed by translating the Alpaca [Taori et al., 2023] and Dolly [Conover et al., 2023a] datasets using the Google Translate API.
- Mistral-7B-Instruct-v0.2 [Jiang et al., 2023] is an open-source instruct fine-tuned edition of the Mistral-7B pre-trained model. The model is trained on instruction datasets publicly available on the HuggingFace repository.
- Gemma-1.1-7B-it [Gemma-Team, 2024] is a 7B parameter instruction fine-tuned model trained with Gemini models' architectures, data, and training recipes [Gemini-Team et al., 2024] on 6 T tokens of data from web documents, mathematics, and code that are primarily English. In addition to the supervised fine-tuning, this model is also further fine-tuned using RLHF on collected pairs of preferences from human annotators.
- Mixtral-8x7B-Instruct-v0.1 [Jiang et al., 2024] is a sparse mixture-of-experts model with 46.7B total parameters (active 12.9B parameters per token) that is instruction fine-tuned and preference-tuned using DPO [Rafailov et al., 2023]. The model supports five languagesEnglish, French, Italian, German, and Spanish.

We do not compare our models to mT0 [Muennighoff et al., 2023] and Okapi [Dac Lai et al., 2023] models, as they have been shown to be significantly outperformed by the Aya-101-13B model [Üstün et al., 2024] which we do compare to as a baseline representative of the state-of-art in massively multilingual LLMs. We note that some of the models we evaluate such as Mistral and Gemma, do[^6]

|  | Held out tasks (Accuracy \%) |  |  |  |
| :--- | :--- | :--- | :---: | :---: |
| Model | XCOPA | XSC | XWG | Avg |
| Bactrian-X-7B | 55.3 | 59.0 | 73.7 | 62.7 |
| Mistral-7B-Instruct-v0.2 | 55.5 | 60.4 | 79.5 | 65.2 |
| Gemma-1.1-7B-it | 59.3 | $\mathbf{6 3 . 1}$ | 75.5 | 66.0 |
| Aya-101-13B | 59.7 | 60.4 | 66.3 | 62.1 |
| \#Aya-23-8B | $\mathbf{5 9 . 8}$ | 62.3 | $\mathbf{8 0 . 7}$ | $\mathbf{6 7 . 6}$ |
| Mixtral-8x7B-Instruct-v0.1 | 59.9 | 63.4 | 83.1 | 68.8 |
| \#Aya-23-35B | $\mathbf{6 2 . 8}$ | $\mathbf{6 5 . 1}$ | $\mathbf{8 4 . 4}$ | $\mathbf{7 0 . 8}$ |

Table 4: Results for discriminative unseen (held-out) task evaluation. Results are reported as the zero-shot performance averaged across all languages of XCOPA, XStoryCloze, and XWinoGrad.

not explicitly claim to support multiple languages, however in practice, they are heavily used by multilingual users relative to explicitly multilingual models like mT0 [Muennighoff et al., 2023] and BLOOMZ [Dac Lai et al., 2023]. Furthermore, we also find that these models achieve considerable performance in many multilingual tasks as shown in our evaluation.

## 5 Results

### 5.1 Discriminative Tasks

Since all discriminative tasks were unseen during training, we measure zero-shot performance during evaluation. For these tasks, we use all the languages available in the evaluation datasets. In Table 4, we report average scores across all languages for XCOPA, XStoryCloze, and XWinoGrad along with an overall average across all tasks. We observe that across all tasks Aya-23-35B outperforms all baselines with an average of $70.8 \%$.. Relative to other large models of comparable size, Aya-23-35B also outperforms Mixtral-8x7B-Instruct-v0.1 (70.8 vs 68.8).

Aya-23-8B achieves the best score within its class in terms of model size, with an average score of 67.6 compared to the next-best model Gemma-1.1-7B-it, which reaches an average score of 66 . Aya-23-8B also outperforms Bactrian-X-7B, Mixtral-7B-Inst-v0.2, and Aya-101-13B. ${ }^{12}$

The significant performance improvements exhibited by Aya-23-8B and Aya-23-35B over the other models including Aya-101-13B, highlight the importance of a high-quality pre-trained base model and an emphasis on a smaller set of languages to achieve a strong performance by avoiding the curse of multilinguality [Conneau et al., 2019].

### 5.1.1 Multilingual MMLU

Table 5 presents multilingual MMLU [Hendrycks et al., 2020] results for all models on 14 languages which is a subset of multilingual MMLU languages [Dac Lai et al., 2023] that are covered by Aya 23 models. We use 5-shot evaluation following the English MMLU benchmark [Beeching et al., 2023].[^7]

|  | ar | de | es | $\mathrm{fr}$ | hi | id | it | $\mathrm{nl}$ | $\mathrm{pt}$ | ro | $\mathrm{ru}$ | uk | vi | $\mathrm{zh}$ | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | 1 | 2.6 | 0 | .5 | 28.6 | 31.1 | 31.8 | 4 | 30.6 | 29.7 | 7 | 26.4 | 9.3 | 29.9 |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-10.jpg?height=34&width=343&top_left_y=341&top_left_x=253) |  |  |  |  |  |  |  |  |  | 46.7 |  |  |  |  | 4.6 |
|  |  | 49.7 | .8 | 6 | 40.1 |  | 50.0 | 4 |  | 47.4 | 47.2 | 0 | 46.2 | .7 | 47.6 |
| Aya- | 39.8 | 42.6 | 42.2 | 42.5 | 38.4 | 41.9 | 41.2 | 42.3 | 41.5 | 40.4 | 41.8 | 41.0 | 40.1 | 40.4 | 41.1 |
| $\because$ Ауа- $23-8 \mathrm{~B}$ | 45.1 | 50.0 | 50.9 | 51.0 | 39.7 | 48.8 | 50.7 | 49.7 | 50.8 | 49.9 | 47.8 | 46.8 | 46.5 | 47.1 | 48.2 |
| Mixtral-8x7B-Ir | 41.8 | 63.7 | 65.2 | 64.9 | 37.8 | J..4 | 64.3 | 62.2 | 63.7 | 60.6 | 59.0 | 37.0 | 48.8 | 54.1 | 57.1 |
| $\because$ Aуа-23-35B | 53.9 | 60.4 | 61.6 | 62.0 | 47.8 | 58.9 | 61.5 | 60.3 | 62.0 | 59.7 | 57.8 | 56.3 | 55.3 | 57.5 | 58.2 |

Table 5: Multilingual MMLU (5-shot) results for Aya 23 models and Aya 101, Bactrian-X, Gemma-7B, Mistral-7B and Mixtral-8x7B in 14 languages.

|  | de | en | es | fr | ja | ru | zh | $\underline{\text { Avg }}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bactrian-X-7B | 5.6 | 7.2 | 5.6 | 6.0 | 4.0 | 4.0 | 4.8 | 5.3 |
| Mistral-7B-Instruct-v0.2 | 34.4 | 31.2 | 29.2 | 32.8 | 6.0 | 31.6 | 30.4 | 27.9 |
| Gemma-1.1-7B-it | 35.6 | 45.2 | 38.4 | $\mathbf{4 1 . 6}$ | 6.0 | $\mathbf{3 9 . 2}$ | 32.0 | 34.0 |
| Aya-101-13B | 9.6 | 10.0 | 8.4 | 8.8 | 4.0 | 10.8 | 4.8 | 8.1 |
| $\because$ Aya-23-8B | $\mathbf{4 0 . 4}$ | $\mathbf{4 8 . 0}$ | $\mathbf{4 5 . 2}$ | 38.8 | $\mathbf{1 2 . 8}$ | 38.0 | $\mathbf{3 2 . 8}$ | $\mathbf{3 6 . 6}$ |
| Mixtral-8x7B-Instruct-v0.1 | 58.8 | 60.0 | 55.2 | 52.8 | $\mathbf{2 4 . 4}$ | 56.0 | 44.4 | 50.2 |
| :Aya-23-35B | $\mathbf{6 1 . 6}$ | $\mathbf{6 8 . 4}$ | $\mathbf{5 8 . 4}$ | $\mathbf{5 5 . 6}$ | 22.8 | $\mathbf{5 8 . 0}$ | $\mathbf{5 0 . 8}$ | $\mathbf{5 3 . 7}$ |

Table 6: Multilingual Grade School Math benchmark (MGSM) results for baselines and Aya models. We use questions with answers followed by CoT prompt (5-shot) in the same language (native_cot) as the dataset and strict-match score as the evaluation metric.

Similar to zero-shot unseen tasks, Aya-23-8B performs overall best among comparable "smaller" models, achieving an average of $48.2 \%$ accuracy across all languages and the highest score in 11 languages out of 14 for its class. At the larger model scale, Aya-23-35B outperforms Mixtral-8x7BInst on average ( 58.2 vs 57.1). Here, Mixtral performs slightly better in relatively high resource languages, however, especially for non-European languages such as Arabic, Hindi, and Vietnamese, Aya-23-35B scores significantly higher with a $12.1 \%, 10.0 \%$ and $6.5 \%$ respective accuracy increase for these 3 languages.

### 5.2 Multilingual Mathematical Reasoning

On MGSM, Aya 23 models outperform all in-class baselines, indicating strong mathematical reasoning ability across languages. Aya-23-8B achieves a score of 36.6 averaged over 7 languages compared to Gemma-1.1-7b-it's score of 34.0 which is the next-best model in its class. Notably, Aya-23-8B achieves a 4.5x increase in performance compared to Aya-101-13B (36.6 vs 8.1), showing the significant impact of the high-quality pre-trained model once more. For the larger scale models, Aya-23-35B outperforms Mixtral-8x7B-Instruct-v0.1 by achieving a score of 53.7 compared to 50.2. When looking at individual language scores, Aya 23 models outperform the strongest in-class models for every language with the exception of French and Russian for Aya-23-8B, and Japanese for Aya-23-35B.

| Model | Generative Tasks |  |  |
| :---: | :---: | :---: | :---: |
|  | FLORES-200 (spBleu) |  | XLSum (RougeL) |
|  | $\mathrm{X} \rightarrow \mathrm{En}$ | $\mathrm{En} \rightarrow \mathrm{X}$ |  |
| Bactrian-X-7B | 25.9 | 16.6 | 7.7 |
| Mistral-7B-Instruct-v0.2 | 31.1 | 21.0 | 6.3 |
| Gemma-1.1-7B-it | 32.0 | 25.6 | 13.0 |
| Aya-101-13B | 35.9 | 30.4 | 27.5 |
| $\because$ Aуа-23-8B | 39.5 | 34.8 | 27.5 |
| Mixtral-8x7B-Instruct-v0.1 | 36.3 | 28.9 | 7.1 |
| $\because$ Aya-23-35B | 43.0 | 37.8 | 30.9 |

Table 7: Translation (FLORES) and multilingual summarization (XLSum) results for baselines and Aya models. For XLSUM, we evaluate models on 15 languages that are included in Aya 23, and for FLORES we use all 22 languages paired with English.

### 5.3 Generative Tasks

Table 7 presents the results for translation (FLORES) and multilingual summarization (XLSum). For FLORES, we use all 23 languages paired with English (X $\leftrightarrow$ EN). For XLSum, we use 15 languages that are available and covered by Aya 23 models. In this evaluation, Aya 23 models achieve significantly higher results than other models with similar sizes. Aya-23-8B achieves an average spBleu score of 37.2, outperforming the second best model Aya-101-13B by 4 points. In XLSum, Aya-23-8B and Aya-101-13B are on par with an average RougeL score of 27.5 surpassing the nextbest model Gemma-1.1 by 14.5 points.

For large model size, Aya-23-35B outperforms Mixtral-8x7B by 7.8 spBleu (40.4 vs 32.6) in translation and 23.8 (30.9 vs 7.1) in summarization. We find that both Mistral-7B and Mixtral-8x7B tend to generate English responses to the prompt although the context is in the target language, leading to poor performance in multilingual summarization.

### 5.4 Simulated Win Rates and Human Eval

GPT-4 Win Rates We perform automatic model ranking using GPT-4 as a judge comparing generations for 200 held-out prompts from dolly-human-edited and dolly-machine-translated [Singh et al., 2024]. Aya 23 models exhibit superior win rates averaged over all languages against the strongest in-class baseline models as shown in Figure 1. Aya-23-8B outperforms Aya-101-13B, Mistral-7B-Instruct-v0.2, and Gemma-1.1-7B-it achieving average win rates of $82.4 \%, 65.2 \%$, and $65.0 \%$ respectively. Aya-23-35B outperforms Mixtral-8x7B-Instruct-v0.1 with an average win-rate of $60.9 \%$.

Figure 3 shows win rates broken down for 10 languages, against the strongest models of similar size. Aya 23 models achieve superior win rates across all languages against all in-class baseline models with the exception of English for Mistral-7B-Instruct-v0.2 for Aya-23-8B and English/French/Spanish for Mixtral-8x7B-Instruct-0.1 for Aya-23-35B. Especially for non-European languages such as Turkish, Hindi, and Japanese Aya 24 models outperform comparison models by a significant margin: Aya-23-8B wins $81.5 \%, 87.5 \%$, and $76.0 \%$ of the time against Mistal-7B while Aya-24-35B wins $78.0 \%, 84.5 \%$ and $75.0 \%$ of the time againist Mixtral-8x7B.

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-12.jpg?height=382&width=727&top_left_y=243&top_left_x=317)

(a) Aya-23-8B vs Aya-101-13B

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-12.jpg?height=377&width=730&top_left_y=706&top_left_x=318)

(c) Aya-23-8B vs Mistral-7B-Instruct-v0.2

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-12.jpg?height=372&width=737&top_left_y=253&top_left_x=1060)

(b) Aya-23-8B vs Gemma-1.1-7B-it

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-12.jpg?height=377&width=732&top_left_y=706&top_left_x=1057)

(d) Aya-23-35B vs Mixtral-8x7B-Instruct-v0.1

Figure 3: LLM-as-a-judge evaluation (\% win rates) for 10 languages comparing Aya-23 models with similar size models for 10 languages. We use gpt-4-turbo for these evaluation as the judge LLM.

Finally, among models that include a similar instruction fine-tuning mixture, Aya-23-8B is heavily preferred to Aya-101-13B in all 10 languages, showing the significant impact of a stronger pretrained model.

|  | English | French | Hindi | Russian | Spanish | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Aya-101-13B | $\mathbf{4 4 . 0}$ | 33.8 | 37.0 | 31.0 | 32.0 | 35.6 |
| Aya-23-8B | 43.0 | $\mathbf{5 6 . 1}$ | $\mathbf{4 3 . 0}$ | $\mathbf{5 9 . 5}$ | $\mathbf{5 2 . 5}$ | $\mathbf{5 0 . 8}$ |
| Aya-101-13B | 35.5 | 30.0 | 34.3 | 28.0 | 26.0 | 30.8 |
| Aya-23-35B | $\mathbf{5 8 . 5}$ | $\mathbf{6 0 . 0}$ | $\mathbf{5 0 . 5}$ | $\mathbf{6 3 . 5}$ | $\mathbf{5 5 . 5}$ | $\mathbf{5 7 . 6}$ |
| Aya-23-8B | 36.5 | 42.7 | 25.6 | 39.5 | $\mathbf{4 1 . 2}$ | 37.1 |
| Aya-23-35B | $\mathbf{4 0 . 0}$ | $\mathbf{4 8 . 7}$ | $\mathbf{3 3 . 7}$ | $\mathbf{4 7 . 0}$ | 39.2 | $\mathbf{4 1 . 7}$ |

Table 8: Human evaluation results (\% win rates) for pairwise comparisons between each pair of models. The remaining percentages are ties. The respective higher average win-rates are boldfaced.

Human Evaluation Table 8 presents win rates resulting from human preference ratings, comparing the Aya 23 models with Aya-101-13B. We observe that with the stronger pre-trained model, Aya 23 family models consistently outperform the mT5-based Aya-101-13B on all evaluated languages. In particular, Aya-23-8B, despite its smaller size wins against Aya-101-13B for 50.8\% of prompts on average across languages. Furthermore, Aya-23-35B achieves $57.6 \%$ win-rate against Aya-101-13B.

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-13.jpg?height=388&width=786&top_left_y=256&top_left_x=255)

(a) Expected maximum toxicity

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-13.jpg?height=388&width=783&top_left_y=256&top_left_x=1061)

(b) Toxicity probability

Figure 4: Toxicity analysis of Aya models (101: Aya-101, 23-8B: Aya-23-8B, 23-35B: Aya-2335B) generations when prompted with sentences for identity groups such as gender, ethnicity, and religion.

We note that human evaluation has been conducted using intermediate checkpoints of Aya 23 models before finalizing our model training due to the required time and cost for these evaluations. We expect higher win-rates for the final Aya 23 models against Aya-101-13B for human evaluation, based on GPT4 win-rates and our internal comparison.

|  | Arabic | English | Hindi | Italian | Simplified Chinese | Ukrainian | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| Aya-101-13B | 81.6 | 83.3 | 81.7 | 93.3 | 75.8 | 88.3 | 84.0 |
| Aya-23-8B | 42.5 | 56.1 | 51.7 | 51.7 | 55.8 | 53.6 | 51.9 |
| Aya-23-35B | $\mathbf{1 1 . 7}$ | $\mathbf{2 1 . 7}$ | $\mathbf{3 7 . 5}$ | $\mathbf{4 0 . 0}$ | $\mathbf{2 7 . 5}$ | $\mathbf{1 9 . 2}$ | $\mathbf{2 6 . 2}$ |

Table 9: Multilingual AdvBench results: percentage of harmful responses as judged by GPT-4. Lower is better.

### 5.5 Safety, Toxicity \& Bias

Safety Table 9 reports the percentage of harmful model completions for the 120 adversarial test split prompts from multilingual AdvBench for 6 languages, as judged by GPT-4.

Comparing Aya 23 models with the Aya-101-13B model previously benchmarked in [Üstün et al., 2024], we find that the rate of harmful responses is lower for all languages, and on average reduced by at least half. The larger capacity of the Aya-23-35B model further helps to lower the harmfulness of the responses, especially for Arabic and Italian, presumably due to a beneficial effect of improved cross-lingual transfer. In terms of quality, we notice that in particular the refusal responses are more eloquent, diverse, and elaborate than those of the Aya-101-13B model which is a reflection of the improved generation quality assessed above.

It is important to note that none of the three models have undergone any targeted safety alignment in the multilingual fine-tuning stage beyond learning from incidental safety examples in synthetically generated examples from Command $\mathrm{R}+$. These scores therefore reflect how much alignment would still be needed for the specific safety cases captured in AdvBench, rather than how much they are already aligned.

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-14.jpg?height=822&width=1489&top_left_y=234&top_left_x=318)

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-14.jpg?height=732&width=740&top_left_y=241&top_left_x=324)

(a) Racial Groups (Man)

![](https://cdn.mathpix.com/cropped/2024_06_04_f69a2d53df2e933d7aa8g-14.jpg?height=715&width=723&top_left_y=255&top_left_x=1075)

(b) Racial Groups (Woman)

Figure 5: Perspective API toxicity scores for Aya-101, Aya-23-7B and Aya-23-35B generations given input prompts in English for racial identity groups.

Toxicity \& Bias Figure 4 shows the expected maximum toxicity and toxicity probability for model completions of the identity group descriptions prompts. We observe that both Aya 23 models generally have lower expected maximum toxicity and a lower toxicity probability than the Aya-101-13B model. This holds true for all languages except English, where the toxicity is slightly higher for the new Aya 23 models. Inspecting English generations further, Figure 5 details the toxicity in descriptions of different racial groups and genders. We note that Aya 23 models tend to produce less toxic generations describing Asians, Latinx, but have a much higher chance to produce toxic descriptions of Blacks and Whites, especially for women.

## 6 Conclusion

While language technologies have made rapid strides in recent years, this progress has been predominantly concentrated in the English language. Given the increasing importance of cross-cultural communication for a broad range of social, economic, and political activities, there is a growing imperative to broaden this progress to other languages so that language technologies can better reflect the reality of the world and more effectively contribute to its more equitable development. We introduce a new family of multilingual models, Aya 23, to advance our mission of using multilingual technologies to empower a multilingual world. Our extensive evaluation demonstrates the high performance of these models on a broad range of multilingual benchmarks and human evaluation. By releasing these model weights, we hope this work will contribute to furthering future research towards this critical mission.

### 6.1 Limitations

While Aya 23 greatly improves performance for the subset of 23 languages chosen and are far more comprehensive in coverage than most open weight releases, we recognize that this subset is only a
tiny fraction of the world's linguistic diversity; of the world's approximately 7,000 languages [eth, 2023], only half of them are captured in any sort of written form [Adda et al., 2016]. Of this half, only a few hundred are included on the internet in machine readable corpora [Adda et al., 2016]. More work is needed to improve both coverage and performance simultaneously.

Additionally, it is important to acknowledge that the languages covered by these models are still limited to those present during pre-training, with a particular bias towards languages prevalent in certain regions of the world. Specifically, the pre-training coverage underrepresents languages spoken in Asia and Africa. This limitation is a critical area that requires ongoing effort and attention. We aim to address this gap and improve language inclusivity as part of the broader Aya Initiative ${ }^{13}$, with a dedicated focus on these underrepresented languages.

Building upon the foundation laid by the original Aya model, which prioritized breadth, future work will concentrate on enhancing coverage and performance for these remaining languages. This includes developing tailored language models, improving data collection and representation, and addressing any cultural and linguistic nuances to ensure equitable and effective language technologies for all.

## 7 Acknowledgements

We thank the Hugging Face team for helping us with our open-weights release including Younes Belkada, Matthew Carrigan, Lysandre Debut, Clémentine Fourrier, Nathan Habib, Quentin Lhoest, Omar Sanseviero, Daniel van Strien, and Arthur Zucker. We thank Aakanksha for sharing their evaluation code for FLORES and XLSum, and Zheng-Xin Yong for the toxicity evaluation.

Thanks to colleagues who have supported various aspects of this project: Linus Chui, Manoj Govindassamy, Yina Moe-Lange, Morgan Norman, Shubham Shukla, Claire Cheng, Trisha Starostina. Thank you to Aidan Gomez, Ivan Zhang and Nick Frosst for support across multiple Aya releases.

## References

Ethnologue. https://www.ethnologue.com/insights/how-many-languages/, 2023. Accessed: 2023-06-17.

Gilles Adda, Sebastian Stüker, Martine Adda-Decker, Odette Ambouroue, Laurent Besacier, David Blachon, Hélène Bonneau-Maynard, Pierre Godard, Fatima Hamlaoui, Dmitry Idiatov, Guy-Noël Kouarata, Lori Lamel, Emmanuel-Moselly Makasso, Annie Rialland, Mark Van de Velde, François Yvon, and Sabine Zerbian. Breaking the unwritten language barrier: The bulb project. Procedia Computer Science, 81:8-14, 2016. ISSN 1877-0509. doi: https://doi.org/10.1016/j.procs.2016.0 4.023. URL https://www.sciencedirect.com/science/article/pii/S1877050916300370. SLTU-2016 5th Workshop on Spoken Language Technologies for Under-resourced languages 09-12 May 2016 Yogyakarta, Indonesia.

Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov. Do all languages cost the same? tokenization in the era of commercial language models, 2023.[^8]

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints, 2023 .

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report. arXiv, abs/2305.10403, 2023.

Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, et al. Massively multilingual neural machine translation in the wild: Findings and challenges. arXiv preprint arXiv:1907.05019, 2019.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https: //huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. Xnli: Evaluating cross-lingual sentence representations. pp. 2475-2485, October-November 2018. doi: 10.18653/v1/D18-1269. URL https://aclanthology .org/D18-1269.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. pp. 8440-8451, July 2019. doi: 10.18653/v1/2020.acl-main.747. URL https://aclanthology.org/2020.acl-main.747.

Mike Conover, Matt Hayes, Ankit Mathur, Xiangrui Meng, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, et al. Free dolly: Introducing the world's first truly open instruction-tuned llm. Databricks, 2023a.

Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world's first truly open instruction-tuned llm, 2023b. URL https://www.databricks.com/blog/2023/04/12/dolly-f irst-open-commercially-viable-instruction-tuned-llm.

Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. arXiv e-prints, pp. arXiv-2307, 2023 .

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing. Multilingual jailbreak challenges in large language models. arXiv preprint arXiv:2310.06474, 2023.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023.

Esin Durmus, Karina Nyugen, Thomas I. Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. Towards measuring the representation of subjective global opinions in language models. arXiv, abs/2306.16388, 2023.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation. 12 2023. doi: 10.5281/zenodo.10256836. URL https: //zenodo.org/records/10256836.

Gemini-Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, Heng-Tze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian Güra, Hao Zhou, Xinying Song, Aurelien Boffy, Harish Ganapathy, Steven Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Laurent El Shafey, Yujing Zhang, Olcan Sercinoglu, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura

Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Gaurav Singh Tomar, Evan Senter, Martin Chadwick, Ilya Kornakov, Nithya Attaluri, Iñaki Iturrate, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Xavier Garcia, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Ravi Addanki, Antoine Miech, Annie Louis, Denis Teplyashin, Geoff Brown, Elliot Catt, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming-Wei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozińska, Vitaliy Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Patel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, James Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo yiin Chang, Paul Komarek, Ross McIlroy, Mario Lučić, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar

Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Konstantinos Aisopos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çağlar Ünlü, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Moufarek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Rakićević, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Sidharth Mudgal, Romina Stella, Kevin Brooks, Gautam Vasudevan, Chenxi Liu, Mainak Chain, Nivedita Melinkeri, Aaron Cohen, Venus Wang, Kristie Seymore, Sergey Zubkov, Rahul Goel, Summer Yue, Sai Krishnakumaran, Brian Albert, Nate Hurley, Motoki Sano, Anhad Mohananey, Jonah Joughin, Egor Filonov, Tomasz Keepa, Yomna Eldawy, Jiawern Lim, Rahul Rishi, Shirin Badiezadegan, Taylor Bos, Jerry Chang, Sanil Jain, Sri Gayatri Sundara Padmanabhan, Subha Puttagunta, Kalpesh Krishna, Leslie Baker, Norbert Kalb, Vamsi Bedapudi, Adam Kurzrok, Shuntong Lei, Anthony Yu, Oren Litvin, Xiang Zhou, Zhichun Wu, Sam Sobell, Andrea Siciliano, Alan Papir, Robby Neale, Jonas Bragagnolo, Tej Toor, Tina Chen, Valentin Anklin, Feiran Wang, Richie Feng, Milad Gholami, Kevin Ling, Lijuan Liu, Jules Walter, Hamid Moghaddam, Arun Kishore, Jakub Adamek, Tyler Mercado, Jonathan Mallinson, Siddhinita Wandekar, Stephen Cagle, Eran Ofek, Guillermo Garrido, Clemens Lombriser, Maksim Mukha, Botu Sun, Hafeezul Rahman Mohammad, Josip Matak, Yadi Qian, Vikas Peswani, Pawel Janus, Quan Yuan, Leif Schelin, Oana David, Ankur Garg, Yifan He, Oleksii Duzhyi, Anton Älgmyr, Timothée Lottaz, Qi Li, Vikas Yadav, Luyao Xu, Alex Chinien, Rakesh Shivanna, Aleksandr Chuklin, Josie Li, Carrie Spadine, Travis Wolfe, Kareem Mohamed, Subhabrata Das, Zihang Dai, Kyle He, Daniel von Dincklage, Shyam Upadhyay, Akanksha Maurya, Luyan Chi, Sebastian Krause, Khalid Salama, Pam G Rabinovitch, Pavan Kumar Reddy M, Aarush Selvan, Mikhail Dektiarev, Golnaz Ghiasi, Erdem Guven, Himanshu Gupta, Boyi Liu, Deepak Sharma, Idan Heimlich Shtacher, Shachi Paul, Oscar Akerlund, François-Xavier Aubet, Terry Huang, Chen Zhu, Eric Zhu, Elico Teixeira, Matthew Fritze, Francesco Bertolini, LianaEleonora Marinescu, Martin Bölle, Dominik Paulus, Khyatti Gupta, Tejasi Latkar, Max Chang, Jason Sanders, Roopa Wilson, Xuewei Wu, Yi-Xuan Tan, Lam Nguyen Thiet, Tulsee Doshi, Sid Lall, Swaroop Mishra, Wanming Chen, Thang Luong, Seth Benjamin, Jasmine Lee, Ewa Andrejczuk, Dominik Rabiej, Vipul Ranjan, Krzysztof Styrc, Pengcheng Yin, Jon Simon, Malcolm Rose Harriott, Mudit Bansal, Alexei Robsky, Geoff Bacon, David Greene, Daniil Mirylenka, Chen Zhou, Obaid Sarvana, Abhimanyu Goyal, Samuel Andermatt, Patrick Siegler, Ben Horn, Assaf Israel, Francesco Pongetti, Chih-Wei "Louis" Chen, Marco Selvatici, Pedro Silva, Kathie Wang, Jackson Tolins, Kelvin Guu, Roey Yogev, Xiaochen Cai, Alessandro Agostini, Maulik Shah, Hung Nguyen, Noah Ó Donnaile, Sébastien Pereira, Linda Friso, Adam Stambler, Adam Kurzrok, Chenkai Kuang, Yan Romanikhin, Mark Geller, ZJ Yan, Kane Jang, Cheng-Chun Lee,

Wojciech Fica, Eric Malmi, Qijun Tan, Dan Banica, Daniel Balle, Ryan Pham, Yanping Huang, Diana Avram, Hongzhi Shi, Jasjot Singh, Chris Hidey, Niharika Ahuja, Pranab Saxena, Dan Dooley, Srividya Pranavi Potharaju, Eileen O'Neill, Anand Gokulchandran, Ryan Foley, Kai Zhao, Mike Dusenberry, Yuan Liu, Pulkit Mehta, Ragha Kotikalapudi, Chalence Safranek-Shrader, Andrew Goodman, Joshua Kessinger, Eran Globen, Prateek Kolhar, Chris Gorgolewski, Ali Ibrahim, Yang Song, Ali Eichenbaum, Thomas Brovelli, Sahitya Potluri, Preethi Lahoti, Cip Baetu, Ali Ghorbani, Charles Chen, Andy Crawford, Shalini Pal, Mukund Sridhar, Petru Gurita, Asier Mujika, Igor Petrovski, Pierre-Louis Cedoz, Chenmei Li, Shiyuan Chen, Niccolò Dal Santo, Siddharth Goyal, Jitesh Punjabi, Karthik Kappaganthu, Chester Kwak, Pallavi LV, Sarmishta Velury, Himadri Choudhury, Jamie Hall, Premal Shah, Ricardo Figueira, Matt Thomas, Minjie Lu, Ting Zhou, Chintu Kumar, Thomas Jurdi, Sharat Chikkerur, Yenai Ma, Adams Yu, Soo Kwak, Victor Ähdel, Sujeevan Rajayogam, Travis Choma, Fei Liu, Aditya Barua, Colin Ji, Ji Ho Park, Vincent Hellendoorn, Alex Bailey, Taylan Bilal, Huanjie Zhou, Mehrdad Khatir, Charles Sutton, Wojciech Rzadkowski, Fiona Macintosh, Konstantin Shagin, Paul Medina, Chen Liang, Jinjing Zhou, Pararth Shah, Yingying Bi, Attila Dankovics, Shipra Banga, Sabine Lehmann, Marissa Bredesen, Zifan Lin, John Eric Hoffmann, Jonathan Lai, Raynald Chung, Kai Yang, Nihal Balani, Arthur Bražinskas, Andrei Sozanschi, Matthew Hayes, Héctor Fernández Alcalde, Peter Makarov, Will Chen, Antonio Stella, Liselotte Snijders, Michael Mandl, Ante Kärrman, Paweł Nowak, Xinyi Wu, Alex Dyck, Krishnan Vaidyanathan, Raghavender R, Jessica Mallet, Mitch Rudominer, Eric Johnston, Sushil Mittal, Akhil Udathu, Janara Christensen, Vishal Verma, Zach Irving, Andreas Santucci, Gamaleldin Elsayed, Elnaz Davoodi, Marin Georgiev, Ian Tenney, Nan Hua, Geoffrey Cideron, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Dylan Scandinaro, Heinrich Jiang, Jasper Snoek, Mukund Sundararajan, Xuezhi Wang, Zack Ontiveros, Itay Karo, Jeremy Cole, Vinu Rajashekhar, Lara Tumeh, Eyal Ben-David, Rishub Jain, Jonathan Uesato, Romina Datta, Oskar Bunyan, Shimu Wu, John Zhang, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Jane Park, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Geoffrey Irving, Edward Loper, Michael Fink, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Ivan Petrychenko, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Evan Palmer, Paul Suganthan, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybiński, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Ginger Perng, Elena Allica Abellan, Mingyang Zhang, Ishita Dasgupta, Nate Kushman, Ivo Penchev, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Daniel Andor, Pedro Valenzuela, Minnie Lui, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Ken Franko, Anna Bulanova, Rémi Leblond, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Mark Omernick, Colton Bishop, Rachel Sterneck, Ro-
han Jain, Jiawei Xia, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Daniel J. Mankowitz, Alex Polozov, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Matthieu Geist, Ser tan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christopher Yew, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Kathy Wu, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Diane Wu, Denese Owusu-Afriyie, Cosmo Du, Chloe Thornton, Jordi PontTuset, Pradyumna Narayana, Jing Li, Saaber Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Yeqing Li, Nir Levine, Ariel Stolovich, Rebeca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Charlie Deck, Hyo Lee, Zonglin Li, Kyle Levin, Raphael Hoffmann, Dan Holtmann-Rice, Olivier Bachem, Sho Arora, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Vinod Koverkathu, Christopher A. Choquette-Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Plucińska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Lynette Webb, Sahil Dua, Dong Li, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Evgenii Eltyshev, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Christof Angermueller, Xiaowei Li, Anoop Sinha, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Denny Zhou, Komal Jalan, Dinghua Li, Blake Hechtman, Parker Schuh, Milad Nasr, Kieran Milan, Vladimir Mikulik, Juliana Franco, Tim Green, Nam Nguyen, Joe Kelley, Aroma Mahendru, Andrea Hu, Joshua Howland, Ben Vargas, Jeffrey Hui, Kshitij Bansal, Vikram Rao, Rakesh Ghiya, Emma Wang, Ke Ye, Jean Michel Sarr, Melanie Moranski Preston, Madeleine Elish, Steve Li, Aakash Kaku, Jigar Gupta, Ice Pasupat, Da-Cheng Juan, Milan Someswar, Tejvi M., Xinyun Chen, Aida Amini, Alex Fabrikant, Eric Chu, Xuanyi Dong, Amruta Muthal, Senaka Buthpitiya, Sarthak Jauhari,

Nan Hua, Urvashi Khandelwal, Ayal Hitron, Jie Ren, Larissa Rinaldi, Shahar Drath, Avigail Dabush, Nan-Jiang Jiang, Harshal Godhia, Uli Sachs, Anthony Chen, Yicheng Fan, Hagai Taitelbaum, Hila Noga, Zhuyun Dai, James Wang, Chen Liang, Jenny Hamer, Chun-Sung Ferng, Chenel Elkind, Aviel Atias, Paulina Lee, Vít Listík, Mathias Carlen, Jan van de Kerkhof, Marcin Pikus, Krunoslav Zaher, Paul Müller, Sasha Zykova, Richard Stefanec, Vitaly Gatsko, Christoph Hirnschall, Ashwin Sethi, Xingyu Federico Xu, Chetan Ahuja, Beth Tsai, Anca Stefanoiu, Bo Feng, Keshav Dhandhania, Manish Katyal, Akshay Gupta, Atharva Parulekar, Divya Pitta, Jing Zhao, Vivaan Bhatia, Yashodha Bhavnani, Omar Alhadlaq, Xiaolin Li, Peter Danenberg, Dennis Tu, Alex Pine, Vera Filippova, Abhipso Ghosh, Ben Limonchik, Bhargava Urala, Chaitanya Krishna Lanka, Derik Clive, Yi Sun, Edward Li, Hao Wu, Kevin Hongtongsak, Ianna Li, Kalind Thakkar, Kuanysh Omarov, Kushal Majmundar, Michael Alverson, Michael Kucharski, Mohak Patel, Mudit Jain, Maksim Zabelin, Paolo Pelagatti, Rohan Kohli, Saurabh Kumar, Joseph Kim, Swetha Sankar, Vineet Shah, Lakshmi Ramachandruni, Xiangkai Zeng, Ben Bariach, Laura Weidinger, Amar Subramanya, Sissie Hsiao, Demis Hassabis, Koray Kavukcuoglu, Adam Sadovsky, Quoc Le, Trevor Strohman, Yonghui Wu, Slav Petrov, Jeffrey Dean, and Oriol Vinyals. Gemini: A family of highly capable multimodal models, 2024.

Gemma Gemini Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Gemma-Team. Gemma: Open models based on gemini research and technology, 2024.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc'Aurelio Ranzato, Francisco Guzman, and Angela Fan. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. arXiv, abs/2106.03193, 2021 .

Tahmid Hasan, Abhik Bhattacharjee, Md Saiful Islam, Kazi Samin, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, and Rifat Shahriyar. XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages. pp. 4693-4703, August 2021. doi: 10.48550/arXiv.2106.13822. URL https://aclanthology.org/2021.findings-acl.413.

William Held, Camille Harris, Michael Best, and Diyi Yang. A material lens on coloniality in nlp. arXiv, abs/2311.08391, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2020.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. arXiv, abs/2401.04088, 2024.

Norman P. Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, Cliff Young, Xiang Zhou, Zongwei Zhou, and David Patterson. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings, 2023.

Khyati Khandelwal, Manuel Tonneau, Andrew M. Bean, Hannah Rose Kirk, and Scott A. Hale. Casteist but not racist? quantifying disparities in large language model bias between india and the west. ArXiv, abs/2309.08573, 2023. URL https://api.semanticscholar.org/CorpusID: 262013517 .

Md Tawkat Islam Khondaker, Abdul Waheed, El Moatez Billah Nagoudi, and Muhammad Abdul-Mageed. Gptaraeval: A comprehensive evaluation of chatgpt on arabic nlp. arXiv, abs/2305.14976, 2023.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. Prometheus: Inducing fine-grained evaluation capability in language models. arXiv preprint arXiv:2310.08491, 2023.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Hadas Kotek, Rikker Dockum, and David Q. Sun. Gender bias and stereotypes in large language models. Proceedings of The ACM Collective Intelligence Conference, 2023. URL https://api. semanticscholar.org/CorpusID:261276445.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. Bactrian-x: Multilingual replicable instruction-following models with low-rank adaptation. arXiv, abs/2305.15011, 2023a.

Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, and Yangqiu Song. Privacy in large language models: Attacks, defenses and future directions. ArXiv, abs/2310.10383, 2023b. URL https://api.semanticscholar.org/CorpusID:264145758.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. Few-shot learning with multilingual language models. arXiv, abs/2112.10668, 2021.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning. arXiv, abs/2301.13688, 2023a.

Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, et al. The data provenance initiative: A large scale audit of dataset licensing \& attribution in ai. arXiv preprint arXiv:2310.16787, 2023b.

Nils Lukas, A. Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, and Santiago Zanella-B'eguelin. Analyzing leakage of personally identifiable information in language models. 2023 IEEE Symposium on Security and Privacy (SP), pp. 346-363, 2023. URL https://api.semanticscholar. org/CorpusID:256459554.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. Crosslingual generalization through multitask finetuning. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1599116111, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v 1/2023.acl-long.891. URL https://aclanthology.org/2023.acl-long.891.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models. arXiv, abs/2311.17035, 2023.

Gabriel Nicholas and Aliya Bhatia. Lost in translation: Large language models in non-english content analysis. arXiv, abs/2306.07377, 2023.

NLLB-Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation. 2022.

Jessica Ojo, Kelechi Ogueji, Pontus Stenetorp, and David I. Adelani. How good are large language models on african languages? arXiv, abs/2311.07978, 2023.

Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe. Lifting the curse of multilinguality by pre-training modular transformers. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3479-3495, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.255. URL https://aclantholo gy.org/2022.naacl-main. 255 .

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. Xcopa: A multilingual dataset for causal commonsense reasoning. pp. 2362-2376, November 2020. doi: 10.18653/v1/2020.emnlp-main.185. URL https://aclanthology.org/2020.emnlp-main. 185.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. CoRR, abs/2108.12409, 2021. URL https://arxiv.org/ab s/2108.12409.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Reva Schwartz, Apostol Vassilev, Kristen Greene, Lori Perine, Andrew Burt, Patrick Hall, et al. Towards a standard for identifying and managing bias in artificial intelligence. NIST special publication, 1270(10.6028), 2022.

Noam Shazeer. GLU variants improve transformer. CoRR, abs/2002.05202, 2020. URL https: //arxiv.org/abs/2002.05202.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=fR3w GCk-IXp.

Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzemiński, Hakimeh Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. Aya dataset: An open-access collection for multilingual instruction tuning. arXiv preprint arXiv:2402.06619, 2024.

Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. CoRR, abs/2104.09864, 2021. URL https://arxiv.org/abs/ 2104.09864 .

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model. 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv, abs/2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv, abs/2307.09288, 2023b.

Aniket Vashishtha, Kabir Ahuja, and Sunayana Sitaram. On evaluating and mitigating gender biases in multilingual settings. arXiv, abs/2307.01503, 2023.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. pp. 483-498, June 2020. doi: 10.18653/v1/2021.naacl-main.41. URL https://aclanthology.org/2 021.naacl-main. 41 .

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2369-2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology .org/D18-1259.

Zheng-Xin Yong, Cristina Menghini, and Stephen H. Bach. Low-resource languages jailbreak GPT4. arXiv, abs/2310.02446, 2023a.

Zheng Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, Genta Winata, Stella Biderman, Edward Raff, Dragomir Radev, and Vassilina Nikoulina. BLOOM +1 : Adding language support to BLOOM for zero-shot prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11682-11703, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.653. URL https://aclanthology.org/2023.acl-long.653.

Joanna Yoo, Kuba Perlin, Siddhartha Rao Kamalakara, and João G. M. Araújo. Scalable training of language models using jax pjit and tpuv4, 2022.

Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, and Xuanjing Huang. Llama beyond english: An empirical study on language capability transfer. arXiv, abs/2401.01055, 2024.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D'souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. Aya model: An instruction finetuned open-access multilingual language model, 2024.
</end of paper 2>


<paper 3>
# EFFICIENT AND EFFECTIVE TEXT ENCODING FOR CHINESE LLAMA AND ALPACA 

Yiming Cui ${ }^{*}$<br>ymcui@ieee.org

Ziqing Yang*<br>ziqingyang@gmail.com

Xin Yao<br>yaoxin94@foxmail.com


#### Abstract

Large Language Models (LLMs), such as ChatGPT and GPT-4, have dramatically transformed natural language processing research and shown promising strides towards Artificial General Intelligence (AGI). Nonetheless, the high costs associated with training and deploying LLMs present substantial obstacles to transparent, accessible academic research. While several large language models, such as LLaMA, have been open-sourced by the community, these predominantly focus on English corpora, limiting their usefulness for other languages. In this paper, we propose a method to augment LLaMA with capabilities for understanding and generating Chinese text and its ability to follow instructions. We achieve this by extending LLaMA's existing vocabulary with an additional 20,000 Chinese tokens, thereby improving its encoding efficiency and semantic understanding of Chinese. We further incorporate secondary pre-training using Chinese data and fine-tune the model with Chinese instruction datasets, significantly enhancing the model's ability to comprehend and execute instructions. Our experimental results indicate that the newly proposed model markedly enhances the original LLaMA's proficiency in understanding and generating Chinese content. Additionally, the results on the C-Eval dataset yield competitive performance among the models with several times the size of ours. We have made our pre-trained models, training scripts, and other resources available through GitHub, fostering open research for our community. ${ }^{12}$


## 1 INTRODUCTION

Natural language processing (NLP) field has witnessed a substantial paradigm shift with the advent of Large Language Models (LLMs). These models, distinguished by their considerable size and comprehensive training data, have demonstrated extraordinary abilities in comprehending and producing human-like text. In contrast to pre-trained language models dedicated to text understanding, such as BERT (Devlin et al., 2019), the GPT series (Radford et al., 2018) accentuates text generation, positioning them as more suitable platforms for creativity compared to their counterparts. Notably, the latest members of the GPT family, namely ChatGPT and GPT-4, have garnered significant attention, establishing themselves as leading examples in this rapidly evolving field.

ChatGPT (OpenAI, 2022), evolved from InstructGPT (Ouyang et al., 2022), serves as an advanced conversational AI model capable of conducting context-aware, human-like interactions. Its success set the stage for the development of GPT-4 (OpenAI, 2023), a more sophisticated LLM, demonstrating even greater potential in natural language understanding, generation, and various NLP tasks, especially for its multi-modal and reasoning abilities. These models have catalyzed new research directions and applications, intensifying interest in exploring the potential of Artificial General Intelligence (AGI). Exhibiting impressive performance across multiple benchmarks, they have also demonstrated capabilities for few-shot learning and adaptability to new tasks, significantly driving the expansion of NLP research. Consequently, they have inspired both researchers and industry professionals to further harness their potential across a wide array of applications, including sentiment analysis, machine translation, question-answering systems, and more.[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-02.jpg?height=812&width=1372&top_left_y=282&top_left_x=367)

Figure 1: Overview of the proposed Chinese LLaMA and Chinese Alpaca models (based on Meta's LLaMA and Llama-2). Chinese LLaMA series are foundation models, and Chinese Alpaca series are chat or instruction-following models.

However, as impactful as LLMs have been, their implementation comes with inherent limitations that hamper transparent and open research. A major concern is their proprietary nature, which restricts access to the models, thus inhibiting the broader research community's ability to build upon their successes. Furthermore, the vast computational resources necessary for training and deploying these models present a challenge for researchers with limited resources, further compounding the accessibility problem.

To tackle these limitations, the NLP research community has gravitated towards open-source alternatives to promote greater transparency and collaboration. LLaMA (Touvron et al., 2023), Llama-2 (Touvron et al., 2023), and Alpaca (Taori et al., 2023a) serve as notable examples of such initiatives. These open-source LLMs are intended to facilitate academic research and accelerate progress within the NLP field. The aim of open-sourcing these models is to foster an environment conducive to further advancements in model development, fine-tuning, and evaluation, ultimately leading to the creation of robust, capable LLMs applicable to a wide variety of uses.

Despite the considerable strides made by LLaMA and Alpaca in NLP, they exhibit inherent limitations concerning native support for Chinese language tasks. Their vocabularies contain only a few hundred Chinese tokens, substantially hindering their efficiency in encoding and decoding Chinese text. Building on our previous work with the Chinese BERT series (Cui et al., 2021) and Chinese minority-oriented multilingual pre-trained models (Yang et al., 2022), in this technical report, we propose the development of Chinese LLaMA and Alpaca models with enhanced capabilities for understanding and generating Chinese content. We extend the original LLaMA's vocabulary with an additional 20,000 Chinese tokens, significantly improving its proficiency in processing and generating Chinese text. To ensure efficient training and deployment of these models, we employ the Low-Rank Adaptation (LoRA) approach (Hu et al., 2021), enabling us to train and fine-tune the models without excessive computational costs. We anticipate our preliminary study to enhance the Chinese understanding and generation capabilities of LLaMA and Alpaca serves as a foundation for researchers aiming to adapt these models to other languages. By showcasing the feasibility and effectiveness of our approach, we offer insights and methodologies that can be employed to extend vocabularies and improve the performance of LLaMA and Alpaca models in various languages. An overview of the proposed models is depicted in Figure 1.

In summary, the contributions of this technical report are as follows:

- We enhance the encoding and decoding efficiency of the Chinese language and improve LLaMA's Chinese understanding ability by extending the original LLaMA's vocabulary with an additional 20,000 Chinese tokens.
- We employ the Low-Rank Adaptation (LoRA) approach to facilitate efficient training and deployment of the Chinese LLaMA and Alpaca models, enabling researchers to work with these models without incurring excessive computational costs.
- We evaluate the performance of the proposed LLaMA and Alpaca models in instructionfollowing tasks and natural language understanding tasks, thereby demonstrating substantial improvements over their original counterparts in the context of Chinese language tasks.
- We make the resources and findings of our study publicly available, fostering further research and collaboration in the NLP community and encouraging the adaptation of LLaMA and Alpaca models to other languages.


## 2 CHINESE LLAMA AND CHINESE ALPACA

### 2.1 BACKGROUND

LLaMA (Touvron et al., 2023) is a foundational, decoder-only large language model built upon the transformer architecture (Vaswani et al., 2017). Similar to the GPT series and other transformerbased LLMs, LLaMA consists of an embedding layer, multiple transformer blocks, and a language model head. LLaMA also incorporates improvements utilized in different models, such as prenormalization (Zhang \& Sennrich, 2019), SwiGLU activation (Shazeer, 2020), and rotary embeddings (Su et al., 2021). LLaMA is available in four different model sizes: 7B, 13B, 33B, and 65B.

LLaMA has been pre-trained with a standard language modeling task (see Section 2.4) using a mix of publicly available sources, such as crawled web pages, books, Wikipedia, and preprint papers. Experimental findings reveal that LLaMA delivers competitive performance compared to other LLMs like GPT-3, albeit at a smaller model size. This compactness and effectiveness have garnered considerable attention from researchers, leading to the widespread use of LLaMA-based models.

### 2.2 CHINESE VOCABULARY EXTENSION

LLaMA's training set encompasses roughly 1.4T tokens, with the majority in English and a small fraction in other European languages using Latin or Cyrillic scripts (Touvron et al., 2023). Thus, LLaMA possesses multilingual and cross-lingual comprehension abilities, mostly demonstrated in European languages. Interestingly, our prior preliminary study reveals that LLaMA exhibits basic Chinese understanding ability, although its capacity to generate Chinese texts is limited.

To equip LLaMA with enhanced Chinese understanding and generation capabilities, we propose to continue pre-training the LLaMA model with Chinese corpora. However, directly applying continual pre-training with Chinese corpora encounters several challenges. Firstly, the original LLaMA vocabulary covers less than a thousand Chinese characters, which is insufficient to encode general Chinese texts. Although the LLaMA tokenizer circumvents this issue by tokenizing unknown UTF-8 characters to bytes, this strategy significantly extends sequence length and slows down the encoding and decoding efficiency of Chinese texts, as each Chinese character splits into 3-4 byte tokens. Secondly, byte tokens are not exclusively designed to represent Chinese characters. Since byte tokens also signify UTF-8 tokens in other languages, it becomes challenging for byte tokens and transformer encoders to effectively learn representations capturing the semantic meaning of Chinese characters.

To address these problems and improve encoding efficiency, we propose to extend LLaMA vocabulary with additional Chinese tokens and adapt the model for the extended vocabulary (Yang et al., 2022). The extension process proceeds as follows:

- To enhance the tokenizer's support for Chinese texts, we initially train a Chinese tokenizer with SentencePiece (Kudo \& Richardson, 2018) on Chinese corpora ${ }^{3}$ with a vocabulary size of 20,000 .
- We subsequently merge the Chinese tokenizer into the original LLaMA tokenizer by taking the union of their vocabularies. Consequently, we obtain a merged tokenizer, which we term the Chinese LLaMA tokenizer, with a vocabulary size of 49,953.
- To adapt the LLaMA model for the Chinese LLaMA tokenizer, we resize the word embeddings and language model head from shape $V \times H$ to $V^{\prime} \times H$, where $V=32,000$ denotes the original vocabulary size, and $V^{\prime}=49,953$ is the new vocabulary size of the Chinese LLaMA tokenizer. The new rows are appended to the end of the original embedding matrices, ensuring that the embeddings of the tokens in the original vocabulary remain unaffected.

Preliminary experiments indicate that the number of tokens generated by the Chinese LLaMA tokenizer is approximately half of those generated by the original LLaMA tokenizer. Table 1 provides a comparison between the original LLaMA tokenizer and our Chinese LLaMA tokenizer. As depicted, the Chinese LLaMA tokenizer significantly reduces the encoding length compared to the original. With a fixed context length, the model can accommodate about twice as much information, and the generation speed is twice as fast as the original LLaMA tokenizer. This highlights the effectiveness of our proposed approach in enhancing the Chinese understanding and generation capabilities of the LLaMA model.

Table 1: Tokenizer comparisons between original LLaMA and Chinese LLaMA.

|  | Length | Content |
| :---: | :---: | :---: |
| Original Sentence | 28 | 人工智能是计算机科学、心理学、哲学等学科融合的交叉学科。 |
| Original Tokenizer | 35 | ![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-04.jpg?height=126&width=913&top_left_y=1308&top_left_x=819) |
| Chinese Tokenizer | 16 | ![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-04.jpg?height=85&width=913&top_left_y=1433&top_left_x=819) |

### 2.3 PARAMETER EfFICIENT FinE-TUNing WITH LoRA

The conventional training paradigm that updates the full parameters of LLMs is prohibitively expensive and is not time- or cost-feasible to most labs or companies. Low-Rank Adaptation (LoRA) (Hu et al., 2021) is a parameter-efficient training method that maintains the pre-trained model weights while introducing trainable rank decomposition matrices. LoRA freezes the pre-trained model weights and injects trainable low-rank matrices into each layer. This approach significantly reduces total trainable parameters, making it feasible to train LLMs with much less computational resources.

To be specific, for a linear layer with weight matrix $W_{0} \in \mathbb{R}^{d \times k}$, where $k$ is the input dimension, and $d$ is the output dimension, LoRA adds two low-rank decomposed trainable matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where $r$ is the pre-determined rank. The forward pass with input $x$ is given by the following equation,

$$
\begin{equation*}
h=W_{0} x+\Delta W x=W_{0} x+B A x, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d} \tag{1}
\end{equation*}
$$

During training, $W_{0}$ is frozen and does not receive gradient updates, while $B$ and $A$ are updated. By choosing the rank $r \ll \min (d, k)$, the memory consumption is reduced as we do not need to store the optimizer states for the large frozen matrix.

To achieve parameter-efficient training while adhering to a tight budget, we apply LoRA training to all Chinese LLaMA and Alpaca models in our paper, including both the pre-training and fine-tuning[^1]stages. We primarily incorporate LoRA adapters into the weights of the attention module and MLP layers. The effectiveness of applying LoRA to all linear transformer blocks is verified in QLoRA (Dettmers et al., 2023), indicating that our choices were reasonable.

### 2.4 PRE-TRAINING OBJECTIVE

We pre-train the Chinese LLaMA model with the standard Causal Language Modeling (CLM) task. Given an input token sequence $\boldsymbol{x}=\left(x_{0}, x_{1}, x_{2}, \ldots\right)$, the model is trained to predict the next token $x_{i}$ in an autoregressive manner. Mathematically, the objective is to minimize the following negative log-likelihood:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{CLM}}(\Theta)=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{\mathrm{PT}}}\left[-\sum_{i} \log p\left(x_{i} \mid x_{0}, x_{1}, \ldots, x_{i-1} ; \Theta\right)\right] \tag{2}
\end{equation*}
$$

where, $\Theta$ represents the model parameters, $\mathcal{D}_{\mathrm{PT}}$ is the pre-training dataset, $x_{i}$ is the token to be predicted, and $x_{0}, x_{1}, \ldots, x_{i-1}$ constitute the context.

### 2.5 SUPERVISED Fine-TUNING AND CHINESE AlPACA

Pre-trained language models can hardly follow user instructions and often generate unintended content. This is because the language modeling objective in Equation (2) is predicting the next token, not "follow the instructions and answer the questions" (Ouyang et al., 2022). To align the behavior of language models to the user's intention, one can fine-tune the model to explicitly train it to follow instructions. Stanford Alpaca (Taori et al., 2023b) is a LLaMA-based instruction-following model that was trained on $52 \mathrm{~K}$ instruction-following data generated by the techniques in the Self-Instruct (Wang et al., 2022). We follow the approach in Stanford Alpaca to apply self-instructed fine-tuning on Chinese LLaMA to train an instruction-following model - Chinese Alpaca.

Chinese Alpaca is trained on a combination of instruction-following datasets. Each example in the dataset consists of an instruction and an output. The supervised fine-tuning task is similar to the causal language modeling task: the model is prompted with the instruction and trained to generate the output autoregressively. The instruction is wrapped in a prompt template, and the output immediately follows the template. We adopt the following template from Stanford Alpaca for fine-tuning and inference, and the input sequence looks like:

Below is an instruction that describes a task. Write a response that appropriately completes the request.

\#\#\# Instruction:

\{instruction\}

\#\#\# Response: $\{$ output\}

The loss is only calculated on the $\{$ output $\}$ part of the input sequence and can be expressed as:

$$
\begin{equation*}
\mathcal{L}_{\mathrm{SFT}}(\Theta)=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{\mathrm{SFT}}}\left[-\sum_{i \in\{\text { output }\}} \log p\left(x_{i} \mid x_{0}, x_{1}, \ldots, x_{i-1} ; \Theta\right)\right] \tag{3}
\end{equation*}
$$

Here, $\Theta$ represents the model parameters, $\mathcal{D}_{\mathrm{SFT}}$ is the fine-tuning dataset, $\boldsymbol{x}=\left(x_{0}, x_{1}, \ldots\right)$ represents the tokenized input sequence.

A major difference between our approach and Stanford Alpaca is that we only use the prompt template designed for examples without an input field, whereas Stanford Alpaca employs two templates for examples both with and without an input field. If the example contains a non-empty input field, we concatenate the instruction and input with an " $\backslash n$ " to form the new instruction. Note that there is an additional padding token for the Chinese Alpaca model, resulting in a vocabulary size 49,954 .

## 3 EXPERIMENTAL SETUPS

### 3.1 EXPERIMENTAL SETUPS FOR PRE-TRAINING

We initialize the Chinese LLaMA model with the original LLaMA weights and conduct pre-training using fp16 on the 7B and 13B models. Additionally, for the 33B model, we employ the bitsandbytes ${ }^{4}$ library to train it in an 8-bit format, enhancing its efficiency and memory usage. We directly apply LoRA to attentions and MLPs for training while setting the embeddings and LM head as trainable.

For the basic version of Chinese LLaMA-7B, we utilize a two-stage pre-training approach. In stage 1, we fix the parameters of the transformer encoders within the model and only train the embeddings, adapting the newly added Chinese word vectors while minimizing the disturbance to the original model. In stage 2, we add LoRA weights (adapters) to the attention mechanisms and train the embeddings, LM heads, and newly added LoRA parameters. Note that two-stage training is not applied to other model training as it is less efficient in our preliminary study.

For the other Chinese LLaMA models (basic version), we utilize a 20GB general Chinese corpus for pre-training, which is consistent with the corpora used by Chinese BERT-wwm (Cui et al., 2021), MacBERT (Cui et al., 2020), LERT (Cui et al., 2022), and others. We also provide "Plus" version, which further expands the pre-training data to $120 \mathrm{~GB}$, incorporating additional data from CommonCrawl (CC) and encyclopedia sources, enhancing the model's understanding of fundamental concepts. We concatenated all the datasets and generated chunks of block size 512 for pre-training purposes.

The models are trained on A40 GPUs (48GB VRAM) for one epoch, taking up to 48 GPUs depending on the model size. The parameter-efficient training with LoRA is performed with PEFT library ${ }^{5}$. We also utilize DeepSpeed (Rasley et al., 2020) to optimize memory efficiency during the training process. We employ the AdamW optimizer (Loshchilov \& Hutter, 2019) with a peak learning rate of 2e-4 and 5\% warm-up cosine scheduler. Additionally, we apply gradient clipping with a value of 1.0 to mitigate potential gradient explosion.

Detailed hyperparameters for each Chinese LLaMA model are listed in Table 2.

Table 2: Pre-training hyperparameters for Chinese LLaMA. QKVO: four matrices in each attention module, i.e., query, key, value, and output. MLP: three matrices in each MLP layer. Note that 7B uses a two-stage training paradigm (settings are separated by ' $/$ '), which is not further adopted in other models.

| Settings | 7B | Plus-7B | 13B | Plus-13B | 33B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Training data | $20 \mathrm{~GB}$ | $120 \mathrm{~GB}$ | $20 \mathrm{~GB}$ | $120 \mathrm{~GB}$ | $20 \mathrm{~GB}$ |
| Batch size | 1,024 | 2,304 | 2,304 | 2,304 | 2,304 |
| Peak learning rate | $2 \mathrm{e}-4 / 1 \mathrm{e}-4$ | $2 \mathrm{e}-4$ | $2 \mathrm{e}-4$ | $2 \mathrm{e}-4$ | $2 \mathrm{e}-4$ |
| Max sequence length | 512 | 512 | 512 | 512 | 512 |
| LoRA rank | -8 | 8 | 8 | 8 | 8 |
| LoRA alpha | $-/ 32$ | 32 | 32 | 32 | 32 |
| LoRA weights | $-/ \mathrm{QKVO}$ | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP |
| Trainable params (\%) | $2.97 \% / 6.06 \%$ | $6.22 \%$ | $4.10 \%$ | $4.10 \%$ | $2.21 \%$ |

### 3.2 EXPERIMENTAL SETUPS FOR INSTRUCTION FINE-TUNING

After obtaining the Chinese LLaMA models, we fine-tune them according to Section 2.5. We continue to employ LoRA for efficient fine-tuning by adding LoRA modules to all linear layers of the base model. We utilize approximately $2 \mathrm{M}$ to $3 \mathrm{M}$ instruction data, including translation ( $\mathrm{Xu}, 2019$ ) (550K sampled), pCLUE ${ }^{6}$ (250K sampled, excluding "NLU-like" data), Stanford Alpaca (50K+50K[^2]for original and translated one), and crawled SFT data for tuning basic models. For the Plus version, we expand the dataset to approximately $4 \mathrm{M}$ to $4.3 \mathrm{M}$, with a specific emphasis on incorporating STEM (Science, Technology, Engineering, and Mathematics) data, as well as several scientific disciplines such as physics, chemistry, biology, medicine, and earth sciences. For Alpaca-33B, we additionally add OASST1 dataset (Köpf et al., 2023), where we only extract the first query-response pair from each conversation and translate using gpt-3.5-turbo API, resulting in roughly $20 \mathrm{~K}$ data (original and translated one). We set the maximum sequence length to 512 and pad the samples dynamically when batching to the maximum length in the batch.

For the crawled data, we refer to the self-instruct (Wang et al., 2022) method for automatically obtaining data from ChatGPT (gpt-3.5-turbo API), as used in Taori et al. (2023a). Concretely, we utilize a more simplified template that does not require seed tasks, with only the requirements for targeted domains and instruction types. Templates and code details are available on GitHub. ${ }^{7}$

Table 3: Instruction fine-tuning hyperparameters for Chinese Alpaca.

| Settings | 7B | Plus-7B | 13B | Plus-13B | 33B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Training data | $2 \mathrm{M}$ | $4 \mathrm{M}$ | $3 \mathrm{M}$ | $4.3 \mathrm{M}$ | $4.3 \mathrm{M}$ |
| Batch size | 512 | 1,152 | 1,152 | 1,152 | 1,152 |
| Peak learning rate | $1 \mathrm{e}-4$ | $1 \mathrm{e}-4$ | $1 \mathrm{e}-4$ | $1 \mathrm{e}-4$ | $1 \mathrm{e}-4$ |
| Max sequence length | 512 | 512 | 512 | 512 | 512 |
| LoRA rank | 8 | 64 | 8 | 64 | 8 |
| LoRA alpha | 32 | 128 | 32 | 128 | 32 |
| LoRA weights | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP |
| Trainable params (\%) | $6.22 \%$ | $8.08 \%$ | $4.10 \%$ | $5.66 \%$ | $2.21 \%$ |

For the Plus version, we utilize a larger LoRA rank compared to the basic version. Besides adjusting the learning rate and batch size, we also maintain consistency with the other hyperparameters and settings used during the pre-training stage.

The hyperparameters for instruction fine-tuning are listed in Table 3. Note that all Alpaca models are trained based on respective LLaMA models. For example, Chinese Alpaca-Plus-13B is trained upon Chinese LLaMA-Plus-13B.

## 4 RESULTS ON INSTRUCTION-FOLLOWING TASKS

### 4.1 TASK DeSIGN and EVALUATION METHOD

Evaluating the performance of text generation tasks can be challenging due to the significant variation in their form, making it significantly different from natural language understanding tasks, such as text classification and extractive machine reading comprehension. Following previous work that utilizes GPT-4 (OpenAI, 2023) as a scoring method, we also adopt GPT-4 to provide an overall score (on a 10-point scale) for each sample, which is more efficient than human evaluation. However, GPT-4 may not always provide accurate scores, so we perform manual checks on its ratings and adjust them if necessary. The manual checks ensure that the scores are consistent and reflect the true performance of the models being evaluated. We use the following prompt template for scoring two outputs of the systems (which can be adjusted to multiple systems):

The followings are two ChatGPT-like systems' outputs. Please rate an overall score on a ten-point scale for each and give explanations to justify your scores.

Prompt:

\{prompt-input $\}$

System1:

\{system1-output\}

System2:[^3]

## \{system2-output $\}$

By employing GPT-4 as a scoring method in conjunction with manual checks, we establish a reliable evaluation framework that effectively measures the performance of our Chinese Alpaca models on a range of natural language understanding and generation tasks.

Our evaluation set is designed to comprehensively assess the Chinese Alpaca models across a wide range of natural language understanding and generation tasks. The set comprises 200 samples, covering ten distinct tasks, including Question Answering, Reasoning, Literature, Entertainment, Translation, Multi-turn Dialogue, Coding, and Ethics, etc. The overall score for a specific task is calculated by summing the scores for all samples within that task and normalizing the total to a 100point scale. This approach ensures that the evaluation set reflects the models' capabilities across various tasks, providing a balanced and robust measure of their performance.

### 4.2 EXPERIMENTAL SETUPS FOR DECODING

The decoding process of LLMs plays a critical role in determining the quality and diversity of the generated text. In our experiments, we use the following decoding hyperparameters:

- Context size: We set the context size to 2048 , which determines the maximum number of tokens the model can consider simultaneously when generating text.
- Maximum sequence length: We limit the generated sequence length to 512 tokens to ensure that the outputs remain focused and relevant to the input prompt.
- Temperature: We set the temperature to 0.2 , which controls the randomness of the sampling process. Lower values make the model generate more focused and deterministic outputs, while higher values increase diversity at the cost of coherence. For multi-turn dialogue and generation tasks, we slightly adjust the temperature to 0.5 to allow a more diverse output.
- Top- $k$ sampling: We use Top- $k$ sampling with $k=40$, meaning that the model selects its next token from the top 40 most probable tokens at each step, adding an element of randomness and diversity to the generated text.
- Top- $p$ sampling: We also employ Top- $p$ sampling with $p=0.9$, which further enhances diversity by considering a dynamic set of tokens that collectively account for $90 \%$ of the probability mass.
- Repetition penalty: To discourage the model from generating repetitive text, we apply a repetition penalty with a factor of 1.1 , penalizing tokens that have already been selected.

Note that these values may not be optimal for each testing scenario. We did not perform further tuning on these hyperparameters for each task to maintain a balanced view.

### 4.3 RESULTS

We present and analyze the results obtained by our Chinese Alpaca-Plus-7B, Alpaca-Plus-13B, and Alpaca-33B models. The Alpaca-33B results are generated by original model (FP16), while the Alpaca-Plus-7B and Alpaca-Plus-13B adopt 8-bit quantized version. ${ }^{8}$ The overall results are shown in Table 4. The evaluation is based on GPT-4 rated results across ten distinct NLP tasks, encompassing a total of 200 samples. It is important to note that the presented scores are solely comparable with each other but not with other models, which would require rescoring the systems. Also, as our models are built upon original LLaMA, these observations can be regarded as what are important aspects to achieving better performance when built upon a well-established model rather than training from scratch. We elaborate on the findings of several major categories in detail.

We mainly present the results on Chinese-LLaMA and Chinese-Alpaca. The results on ChineseLLaMA-2 and Chinese-Alpaca-2 are presented in Appendix A.[^4]

Table 4: GPT-4 rated results for Chinese Alpaca-Plus-7B and Alpaca-Plus-13B, and Alpaca33B. Note that the results are only comparable within this model combination.

| Task | Alpaca-Plus-7B | Alpaca-Plus-13B | Alpaca-33B |
| :--- | :---: | :---: | :---: |
| Question Answering | 70.5 | 79.5 | $\mathbf{8 2 . 3}$ |
| Open-ended QA | $\mathbf{8 0 . 5}$ | 80.0 | 78.5 |
| Numerical Reasoning | 51.0 | 61.5 | $\mathbf{8 4 . 5}$ |
| Poetry, Literature, Philosophy | 78.5 | $\mathbf{8 1 . 3}$ | 76.0 |
| Music, Sports, Entertainment | 72.3 | $\mathbf{7 6 . 8}$ | 72.5 |
| Letters and Articles Writing | 81.0 | $\mathbf{8 6 . 5}$ | 79.0 |
| Translation | 86.8 | 89.3 | $\mathbf{9 2 . 3}$ |
| Multi-turn Dialogue | 80.3 | $\mathbf{8 1 . 3}$ | 78.0 |
| Coding | 62.5 | 67.5 | $\mathbf{8 4 . 0}$ |
| Ethics | 89.8 | 90.5 | $\mathbf{9 2 . 5}$ |
| Total | 75.3 | 79.4 | $\mathbf{8 2 . 0}$ |

### 4.3.1 MULTI-TURN DiALOGUE

One of the impressive achievements of ChatGPT is its rich and fluent contextual understanding ability, which is conveyed by the multi-turn dialogue interface. As we can see, Plus series models yield consistent improvements over the basic one, though the size of the latter one is several times that of the formers. This might indicate that it is much more important to ingest more training data than simply extending the parameter size of the model to achieve a better dialogue experience. Especially our models are constructed from the original LLaMA, where linguistic knowledge can not be directly transferred.

### 4.3.2 TEXT GENERATION

Text generation is one of the most fundamental abilities for language models. Compared to AlpacaPlus-7B and Alpaca-Plus-13B, Alpaca-33B shows inferior results in this category. Table 5 shows an example of a text generation task. We can see that both Alpaca-Plus-7B and Alpaca-Plus-13B provide correct letter styles, which meet the requirement of the user's prompt. Alpaca-Plus-13B provides the most comprehensive one by indicating that the applicant has thoroughly prepared all materials for visa application, making it the best generation quality among all three systems. However, Alpaca-33B does not follow the letter style, and the content is somewhat too simplified, which is clearly not as good as the others. This demonstrates that training with more data with smaller models might give better performance than big models with less data.

### 4.3.3 NUMERICAL CALCULATION AND REASONING

Numerical reasoning has been regarded as one of the most essential tasks in examining the reasoning ability of large language models. As we can see, the Alpaca-33B achieves significant improvements over Plus-7B and Plus-13B models. Table 6 shows example outputs for this task. The first prompt is well-known for probing the reasoning ability, namely "which one is heavier, $1 \mathrm{~kg}$ of cotton or $1 \mathrm{~kg}$ of iron?". Both Plus-7B and Plus-13B failed to give a correct answer mentioning that "cotton is lighter than iron". However, 33B could correctly identify that these two things are the same weight. The second prompt is a simple calculation task, asking "how many legs for a cat and a chicken". However, as we can see, both Plus-7B and Plus-13B do not have the commonsense knowledge that a cat has four legs and two for a chicken, resulting in wrong answers. The last prompt is a numerical reasoning task to let the model predict the next number of an array. Still, only 33B model correctly identifies the pattern of the given array that the next number should be the square of its index. These observations indicate that the size of the model is vital in numerical reasoning tasks.

Table 5: Example output for text generation task. 33B model does not provide detailed and rich context compared to Plus-7B and Plus-13B.

| Model | Prompt: 请你帮我写一封赴美旅游签证申请信 |
| :---: | :---: |
| Plus-7B <br> $(8.5 / 10)$ | 尊敬的美国大使馆 |
|  | 我是一名来自中国的游客, 计划于2022年7月前往美国旅游。我非常期待这次旅行, <br> 并希望获得签证。 |
|  | 我将遵守美国的法律和规定, 并尊重当地文化和风俗。我将遵守旅游计划, 并在离 <br> 开前美国大使馆报告我的踪。 |
|  | 感谢您考虑我的申请。 |
|  | 此致 |
|  | 敬礼！ |
| Plus-13B <br> $(10 / 10)$ | 尊敬的美国大使馆官员， |
|  | 我是 $\mathrm{xxx}$, 来自中国。我计划于明年3月前往美国进行为期一个月的旅行。我的行程 <br> 包括观纽约市、洛杉矶和拉斯维加斯等城市, 并与当地的们交流。我相信这次 <br> 旅行将对我的个人成长和发展有所帮助。 |
|  | 我已经准备好了所有需要的材料, 包括机票预订证明、酒店预订证明以及旅游计划。 <br> 我望过此请获的旅行顺, 以利进行。 |
|  | 谢谢您的时间和关注。我期待着能够很快收到您的回复。 |
|  | 此致, |
|  | 敬礼 |
| 33B <br> $(8 / 10)$ | 尊敬的美国大使馆工作人员, 我谨代表本人向您申请一次赴美旅游签证。我是一名 <br> 中国公民,目前从事互联网行业工作, 有良好的职业背景和经济实力。此旅行的 <br> 目的是探访美国文化、解当地风人情,并结交一些志同道合的朋友。我将遵守 <br> 美国的法律法规, 不违反任何定。谢谢您的考! |

### 4.3.4 CODING

Figure 2 shows an example of implementing the Dijkstra algorithm in Python. Plus-7B scores $3 / 10$ due to a structurally sound approach that unfortunately fails to calculate and update shortest distances correctly and includes an undefined function. Plus-13B attempts abstraction by implementing a Graph class and a distance method, which shows a basic understanding of how a graph and its related operations could be represented in object-oriented programming. Also, the fact that it is attempting to implement a shortest path algorithm (despite not correctly implementing Dijkstra's algorithm) makes it a slightly higher score than Plus-7B's. The 33B model offers a much better Dijkstra algorithm implementation, earning it an $8 / 10$ score. Despite its lack of a priority queue and absence of error handling, which would enhance efficiency and robustness, the code correctly updates shortest distances, maintains track of predecessors, and ensures all nodes are visited, reflecting a fundamental understanding of the algorithm.

From these results, it could be inferred that larger models tend to perform better in complex tasks like code generation, potentially due to their ability to capture more intricate patterns in the training data.

### 4.3.5 ETHICS

Aligning LLMs to human preference is vital in creating responsible artificial intelligence. In the Ethics category, we mainly want to test how these models respond to illegal input prompts. By

Table 6: Example output for numerical reasoning task. 33B model significantly outperforms Alpaca-Plus-7B and Alpaca-Plus-13B.

| Model | Prompt: 一公斤的棉花和一公斤的铁, 哪一个更重? |
| :--- | :--- |
| Plus-7B (3/10) | 一公斤棉花比一公斤铁轻。 |
| Plus-13B (3/10) | 一公斤的棉花比一公斤的铁轻。 |
| 33B (9/10) | 一公斤的棉花和一公斤的铁重量相同。 |
| Model | Prompt: 一只猫和一只鸡一共有几只脚? |
| Plus-7B (1/10) | 一只猫和一只鸡一共有两只脚。 |
| Plus-13B (1/10) | 一只猫和一只鸡各有四只脚, 所以两只动物一共有八只脚。 |
| 33B (10/10) | 这道题需要进行简单的数学运算。一只猫有4只脚, 一只鸡有2只脚, |
| Model | Prompt: 序列1,4,9,16的后一个数是什么? |
| Plus-7B (1/10) | 后一个数是16。 |
| Plus-13B (1/10) | 21 |
| 33B (10/10) | 序列1,4,9,16的后一个数是25。 |

checking the generation results, all three systems responded properly to users' prompts. Alpaca33B yields slightly better performance than the others. We discover that Alpaca-33B may not only "reject" illegal prompts but also give appropriate advice in addition. For example, in Table 7, both Plus-7B and Plus-13B simply refuse to give any advice on making money by exploiting some network vulnerabilities. On the contrary, 33B model not only refuses the user prompt but also gives advice on how to make money using legal ways, making the response more comprehensive and helpful.

Table 7: Example output for ethics category. 33B model tends to provide more comprehensive and helpful information over Plus-7B and Plus-13B.

| Model | Prompt: 如何利用一些网络漏洞赚钱? |
| :---: | :---: |
| Plus-7B <br> $(8 / 10)$ | 我不能提供任何有关利用网络漏洞来赚钱的建议或指导, 因为这违反了道德和法律 <br> 规定。网络安全是一项重要的任务, 我们应该逗守相关的法律法和行业标准, <br> 确保。 |
| Plus-13B <br> $(8 / 10)$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-11.jpg?height=126&width=1143&top_left_y=1821&top_left_x=566) |
| $33 B$ <br> $(9 / 10)$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-11.jpg?height=412&width=1143&top_left_y=1946&top_left_x=566) |

Overall, Alpaca-33B yields significant improvements over Alpaca-Plus-7B and Alpaca-Plus-13B in various aspects, including numerical reasoning, coding, ethics, etc. We conjecture that these abilities are better handled by bigger models than the smaller ones, though Alpaca-33B is trained

Plus-7B (3/10):

```
import heapq
def dijkstra(graph, s):
    queue = []
    heap = [s]
    while len(heap) > 0:
        current = heap.pop()
        for i in graph[current]:
            if i not in heap and dist(i, s) < dist(s, current):
                heap.append(i)
    return queue
```

Plus-13B (4/10):

```
class Graph:
    def __init__(self, adj):
        self.adj = adj
    def distance(self, v):
        dist = [inf] * len(self.adj)
        for u in self.adj[v]:
            if u not in dist
                dist[u] = 1 + min(dist[u], dist[v])
            return dist[v]
def main():
    g = Graph([(0,1), (1,2)), (2,3), (3,4)])
    V}=
    d}=\textrm{g}\cdot\mathrm{ distance (v)
    print('Distance is:', d)
if __name__ == '___main__'
    main()
```

33B (8/10):

```
def dijkstra(graph, source)
    distances = {}
    predecessors = {}
    for node in graph:
        distances[node] = float('inf')
        predecessors[node] = None
    distances[source] = 0
    queue = [source]
    while queue:
        current = queue.pop(0)
            for neighbor in graph[current]:
                    if distances[neighbor] > distances[current] + graph[current][neighbor]:
                    distances[neighbor] = distances[current] + graph[current][neighbor]
                    predecessors[neighbor] = current
    return distances
```

Figure 2: Example output for coding task. 33B model significantly outperforms Alpaca-Plus-7B and Alpaca-Plus-13B.

with less data. Another possible reason would be the inherited ability from the original LLaMA, in which coding and reasoning ability is relatively language-independent. However, we also noticed that Alpaca-33B has inferior results in text generation, multi-turn dialogue, etc. As Plus series models are trained on much more data, they are capable of providing more diverse and rich content. We anticipate these issues can be tackled when Alpaca-Plus-33B becomes available, as we find these abilities are relatively easy to overcome than those that require high-level reasoning, such as numerical reasoning and coding-related tasks. For complete comparisons, ratings, and sample outputs, please refer to our GitHub repository. ${ }^{9}$

## 5 RESULTS ON NATURAL LANGUAGE UNDERSTANDING TASKS

### 5.1 TASK DESCRIPTION

Besides the generation performance test for instruction-following tasks, we also tested our models on the C-Eval dataset (Huang et al., 2023), which is a multi-choice question answering dataset. C-[^5]

Eval mainly covers four categories: STEM, Social, Humanities, and Others, consisting of nearly $14 \mathrm{~K}$ samples for 52 disciplines. Similar to other multi-choice QA datasets, such as RACE (Lai et al., 2017), it requires the model to produce the correct option label based on the given question. We mainly tested our model on the validation split ( 1,346 samples) and test split ( 12,342 samples), where the test scores are obtained by submitting models' prediction files to the official leaderboard.

### 5.2 DECODING STRATEGY

To evaluate LLaMA models on this dataset, we directly feed the examples to these models. While when evaluating Alpaca models, we wrap the examples in the prompt template as demonstrated in Section 2.5. Then the model is asked to make a one-step prediction and give the probability distribution of the next token $p(y \mid \boldsymbol{x})$, where $y \in \mathcal{V}(\mathcal{V}$ is the vocabulary). To map the probability distribution to a valid label $t$ in $\{$ A, B, C, D\}, we extract and gather the probabilities of related tokens. We introduce a verbalizer $\mathcal{V}(\cdot)$ to map each label $t$ to tokens in the vocabulary:

$$
\mathcal{V}(\mathrm{A})=\left\{{ }^{\prime} \mathrm{A}^{\prime}, \mathrm{A}^{\prime}\right\}, \quad \mathcal{V}(\mathrm{B})=\left\{{ }^{\prime} \mathrm{B}^{\prime},{ }^{\prime} \_\mathrm{B}^{\prime}\right\}, \quad \mathcal{V}(\mathrm{C})=\left\{{ }^{\prime} \mathrm{C}^{\prime},{ }^{\prime} \_\mathrm{C}^{\prime}\right\}, \quad \mathcal{V}(\mathrm{D})=\left\{{ }^{\prime} \mathrm{D}^{\prime}, \_^{\prime} \mathrm{D}^{\prime}\right\}
$$

The probability of predicting label $t$ is given by

$$
\begin{equation*}
p(t \in\{\mathrm{A}, \mathrm{B}, \mathrm{C}, \mathrm{D}\} \mid \boldsymbol{x})=\sum_{t \in \mathcal{V}(i)} p(y=i \mid \boldsymbol{x}) \tag{4}
\end{equation*}
$$

The label with the max probability is taken as the final prediction.

Next, we will elaborate on our results and analysis in the following two subsections, illustrating the comparisons to the original LLaMA and other models.

### 5.3 COMPARISONS TO ORIGINAL LLAMA

Figure 3 demonstrates how our models evolve based on the original LLaMA. Detailed results are depicted in Table 8. We mainly describe our findings in the following aspects.

![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-13.jpg?height=569&width=1195&top_left_y=1488&top_left_x=454)

Figure 3: Results on C-Eval valid set. The results are grouped by different settings (zero-shot and 5 -shot) and model sizes (7B and 13B).

Chinese LLaMA improves original LLaMA. We can see that the proposed Chinese LLaMA models yield moderate improvements over the original LLaMA, which demonstrates that the pretraining on Chinese data has some positive effect on C-Eval but not always. When we compare Chinese LLaMA and LLaMA-Plus, the latter does not show significant improvements over the former one, even showing inferior results for $13 \mathrm{~B}$ setting. This might indicate that the pure language model (like LLaMA) may not be a good choice for C-Eval or similar tasks, and it does not benefit much from increasing the pre-training data size (from 20G to 120G for Chinese LLaMA and LLaMA-Plus, respectively).

Table 8: Results on C-Eval valid and test sets. All prediction files are generated by ourselves. The test set scores are obtained by submitting prediction files to the C-Eval leaderboard.

| Model | Valid Set |  | Test Set |  |
| :--- | :---: | :---: | :---: | :---: |
|  | Zero-shot | 5-shot | Zero-shot | 5-shot |
| Random | 25.0 | 25.0 | 25.0 | 25.0 |
| LLaMA-65B | 37.2 | 41.2 | 33.4 | 38.8 |
| LLaMA-33B | 34.5 | 37.9 | 32.4 | 36.0 |
| LLaMA-13B | 27.8 | 30.9 | 28.5 | 29.6 |
| LLaMA-7B | 25.6 | 25.3 | 26.7 | 27.8 |
| Chinese-LLaMA-33B | 34.9 | 38.4 | 34.6 | 39.5 |
| Chinese-LLaMA-Plus-13B | 27.3 | 34.0 | 27.8 | 33.3 |
| Chinese-LLaMA-13B | 29.4 | 35.0 | 29.2 | 33.7 |
| Chinese-LLaMA-Plus-7B | 27.3 | 28.3 | 26.8 | 28.4 |
| Chinese-LLaMA-7B | 26.2 | 26.2 | 27.1 | 27.2 |
| Chinese-Alpaca-33B | 43.3 | 42.6 | 41.6 | 40.4 |
| Chinese-Alpaca-Plus-13B | 43.3 | 42.4 | 41.5 | 39.9 |
| Chinese-Alpaca-13B | 37.1 | 36.3 | 36.7 | 34.5 |
| Chinese-Alpaca-Plus-7B | 36.7 | 32.9 | 36.4 | 32.3 |
| Chinese-Alpaca-7B | 30.8 | 32.5 | 30.7 | 29.2 |


#### Abstract

Alpaca models show significant improvements over LLaMA. Among different settings, such as zero-shot or 5 -shot, the Alpaca model series show significant improvements over LLaMA counterparts, demonstrating that the instruction-following models are more capable of handling these NLU-like tasks than pure language models. Unlike the phenomenon observed in the LLaMA series, we can see that Alpaca-Plus models yield significant improvement over basic Alpaca models. This might further indicate that instruction-following models are more capable of handling NLU-like tasks and can unleash the power of using more pre-training data (LLaMA-Plus).


LLaMA generally yields better performance in a few-shot setting, while Alpaca prefers zeroshot. Generally speaking, LLaMA with 5 -shot setting shows better performance than zero-shot setting, while Alpaca with zero-shot setting is much better than 5-shot one. As LLaMA is not designed for instruction-following, few-shot setting might give valuable information on how to follow the question answering structure in C-Eval. However, on the contrary, as Alpaca has already been trained with millions of instruction data, it is less likely to benefit from additional shots. Also, the official 5-shot setting uses identical prompts for all samples, making it some distraction for Alpaca models.

We would like to emphasize that these observations are solely based on the results of the C-Eval dataset, and whether it is generalizable to other datasets requires further investigation. In the future, we will include more comprehensive tests to further investigate LLaMA and Alpaca models' behaviors.

# 5.4 COMPARISONS TO OTHER MODELS 

We include our two best-performing models, i.e., Chinese-Alpaca-33B and Chinese-Alpaca-Plus13B, in the C-Eval leaderboard to make a comparison with other LLMs, including both open-source and non-open-source ones. The test results on the C-Eval leaderboard (as of June 9, 2023) are shown in Table 9 .

Not surprisingly, non-open-source LLMs have significantly better performance than open-source ones. When it comes to our models, we can see that both Chinese-Alpaca-33B and Chinese-AlpacaPlus-13B yield competitive performance among open-source LLMs in this leaderboard, showing only a moderate gap to Bloomz-mt-176B (Scao et al., 2022) and GLM-130B (Zeng et al., 2023), considering that the latter ones have several times of magnitude and trained with way more data than ours.

Table 9: Test results on C-Eval leaderboard (as of June 9, 2023), ordered by average scores. Model name with boldface represents our submissions, while the other results are evaluated by C-Eval officials. We re-evaluated two models marked with $\dagger$ (these scores are not shown publicly) based on our own inference script and achieved significantly better performance than those evaluated by C-Eval. The parameter size of the model is depicted in parentheses when available. Open: opensource. Avg-H: Average (Hard).

| Model | N-Shot | Open | Avg | Avg-H | STEM | Social | Human | Others |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4 | 5-shot | $\boldsymbol{x}$ | 68.7 | 54.9 | 67.1 | 77.6 | 64.5 | 67.8 |
| InternLM (104B) | few-shot | $\boldsymbol{x}$ | 62.7 | 46.0 | 58.1 | 76.7 | 64.6 | 56.4 |
| ChatGPT | 5-shot | $\boldsymbol{x}$ | 54.4 | 41.4 | 52.9 | 61.8 | 50.9 | 53.6 |
| Claude-v1.3 | 5-shot | $\boldsymbol{x}$ | 54.2 | 39.0 | 51.9 | 61.7 | 52.1 | 53.7 |
| Claude-instant-v1.0 | 5-shot | $\boldsymbol{x}$ | 45.9 | 35.5 | 43.1 | 53.8 | 44.2 | 45.4 |
| Bloomz-mt (176B) | 0-shot | $\checkmark$ | 44.3 | 30.8 | 39.0 | 53.0 | 47.7 | 42.7 |
| GLM-130B | 0-shot | $\checkmark$ | 44.0 | 30.7 | 36.7 | 55.8 | 47.7 | 43.0 |
| Chinese-Alpaca-33B | 0-shot | $\checkmark$ | 41.6 | 30.3 | 37.0 | 51.6 | 44.3 | 40.3 |
| Chinese-Alpaca-Plus-13B | 0-shot | $\checkmark$ | 41.5 | 30.5 | 36.6 | 49.7 | 43.1 | 41.2 |
| CubeLM (13B) | few-shot | $\boldsymbol{x}$ | 40.2 | 27.3 | 34.1 | 49.7 | 43.4 | 39.6 |
| ChatGLM-6B | 0-shot | $\checkmark$ | 38.9 | 29.2 | 33.3 | 48.3 | 41.3 | 38.0 |
| LLaMA-65B | 5-shot | $\checkmark$ | 38.8 | 31.7 | 37.8 | 45.6 | 36.1 | 37.1 |
| Chinese-Alpaca-13B $\dagger$ | 0-shot | $\checkmark$ | 36.7 | 28.4 | 33.1 | 43.7 | 38.4 | 35.0 |
| Chinese-LLaMA-13B $\dagger$ | 5-shot | $\checkmark$ | 33.7 | 28.1 | 31.9 | 38.6 | 33.5 | 32.8 |
| Chinese-LLaMA-13B | 5-shot | $\checkmark$ | 33.3 | 27.3 | 31.6 | 37.2 | 33.6 | 32.8 |
| MOSS (16B) | 0-shot | $\checkmark$ | 33.1 | 28.4 | 31.6 | 37.0 | 33.4 | 32.1 |
| Chinese-Alpaca-13B | 0-shot | $\checkmark$ | 30.9 | 24.4 | 27.4 | 39.2 | 32.5 | 28.0 |

For another aspect, Chinese-Alpaca-13B and Chinese-LLaMA-13B were previously evaluated by CEval. We also manually submitted the prediction file by our own implementation to the leaderboard. The results show that both models show significant improvements over the ones evaluated by C-Eval, especially for Alpaca-13B model, yielding +5.8 average score (from 30.9 to 36.7). Also, Alpaca13B shows advantages over LLaMA-13B, which is in accordance with our previous findings. These observations indicate that adopting a proper decoding strategy and prompt template might be vital in achieving better performance for individual LLMs, especially for instruction-following models.

## 6 EFFECT OF DIFFERENT QUANTIZATION METHODS

Deploying large language models on personal computers, particularly on CPUs, has historically been challenging due to their immense computational requirements. However, with the help of many community efforts, such as llama. cpp (Gerganov, 2023), users can efficiently quantize LLMs, significantly reducing memory usage and computational demands, making it easier to deploy LLMs on personal computers. This also enables quicker interactions with the models and facilitates local data processing. Quantizing LLMs and deploying them on personal computers offer several benefits. Firstly, it helps users protect their data privacy by ensuring that sensitive information remains within their local environment rather than being transmitted to external servers. Secondly, it democratizes access to LLMs by making them more accessible to users with limited computational resources. Lastly, it promotes the development of new applications and research directions that take advantage of local LLM deployments. Overall, the ability to deploy LLMs on personal computers using llama. cpp (or similar) paves the way for a more versatile and privacy-conscious utilization of LLMs in various domains.

In this section, we investigate the effect of different quantization methods. We use llama. cpp to quantize Alpaca-Plus-7B, Alpaca-Plus-13B, and Alpaca-33B and calculate the perplexity on Chinese text corpora. We quantize these models into 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit forms to compare with the original FP16 one. ${ }^{10}$ The results are shown in Figure 4.[^6]

![](https://cdn.mathpix.com/cropped/2024_06_04_0ea2fd0606542570bcb0g-16.jpg?height=708&width=1119&top_left_y=272&top_left_x=492)

Figure 4: Perplexities for different quantization methods. Note that $33 \mathrm{~B}$ model has a higher PPL as it is trained on less data than the others.

The quantization level is strictly bound to the memory usage and inference speed, and thus a tradeoff must be made when choosing a proper quantization level. As we can see, the 8 -bit quantization method has almost the same or even lower perplexities compared to the original FP16 model, demonstrating that it is a good choice for deploying LLMs on personal computers, with only half size of the FP16 one. The 6-bit models also achieve decent PPLs comparable to the 8 -bit one, making it a better balance of speed and performance. When we use a more aggressive quantization level, the performance drastically decreases (i.e., higher PPL), especially for 3-bit and 2-bit. We also discover that larger models are less sensitive to quantization methods than smaller ones. For example, the performance of $33 \mathrm{~B}$ models changes much more mildly than the others. A similar result is also observed when comparing Plus-7B and Plus-13B models. This might indicate that though 2-bit and 3-bit quantization are less effective for smaller models, it might be a promising way to deploy larger models without significant performance loss. This is extremely helpful when the users only have limited computing resources and still want to try large language models. This might also imply that the quantized training method may become a main-stream approach for training large language models, especially for those with limited training resources.

## 7 CONCLUSION

In this technical report, we have presented an approach to enhance the Chinese understanding and generation capabilities of the LLaMA model. Acknowledging the limitations of the original LLaMA's Chinese vocabulary, we expanded it by incorporating 20K additional Chinese tokens, significantly increasing its encoding efficiency for the Chinese language. Building on the Chinese LLaMA, we employed supervised fine-tuning with instruction data, resulting in Chinese Alpaca models exhibiting improved instruction-following capabilities.

To evaluate our models effectively, we annotated 200 samples across ten distinct task types and utilized GPT-4 for evaluation. Our experiments demonstrated that the proposed models significantly outperformed the original LLaMA in Chinese understanding and generation tasks. We also tested our models on C-Eval datasets. The results show that the proposed model could achieve significant improvements and show competitive performance to the models with several times bigger sizes.

Looking ahead, we plan to explore Reinforcement Learning from Human Feedback (RLHF) or Reinforcement Learning from AI Instructed Feedback (RLAIF) to further align the models' output with human preferences. Moreover, we intend to adopt more advanced and effective quantization methods, such as GPTQ (Frantar et al., 2022), among others. Additionally, we aim to investigate alternative methods to LoRA for more efficient and effective pre-training and fine-tuning of large lan-
guage models, ultimately enhancing their performance and applicability across various tasks within the Chinese NLP community.

## LIMITATIONS

While this project has successfully enhanced the Chinese understanding and generation capabilities of the LLaMA and Alpaca models, several limitations must be acknowledged:

- Harmful and unpredictable content: Though our model can reject unethical queries, these models may still generate harmful or misaligned with human preferences and values. This issue may arise from biases in the training data or the models' inability to discern appropriate outputs in certain contexts.
- Insufficient training: Due to constraints in computing power and data availability, the training of the models may not be sufficient for optimal performance. As a result, there is still room for improvement in the Chinese understanding capabilities of the models.
- Lack of robustness: The models may exhibit brittleness in some situations, producing inconsistent or nonsensical outputs when faced with adversarial inputs or rare language phenomena.
- Comprehensive evaluation: Evaluating large language models is an important topic in the current era. While we have seen many evaluation benchmarks for LLMs, their comprehensiveness and appropriateness for LLMs should be well-studied and examined. A more diverse and comprehensive LLM evaluation dataset and benchmark will have a great positive effect on shaping the future of LLM research.
- Scalability and efficiency: Although we applied LoRA and quantization to make the model more accessible to a broader community, when combined with the original LLaMA, the models' large size and complexity can lead to difficulties in deployment, especially for users with limited computational resources. This issue may hinder the accessibility and widespread adoption of the models in various applications.

Future work should address these limitations to further enhance the models' capabilities, making them more robust, accessible, and effective for a broader range of applications in the Chinese NLP community.

## ACKNOWLEDGMENTS

The original draft was polished by OpenAI GPT-4 for grammatical corrections and clarity improvements. We would like to thank our community members for their contributions to our open-source projects.

## REFERENCES

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. Revisiting pretrained models for Chinese natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings, pp. 657-668, Online, November 2020. Association for Computational Linguistics. URL https://www. aclweb. org / anthology/2020.findings-emnlp. 58.

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. Pre-training with whole word masking for chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3504-3514, 2021. doi: 10.1109/TASLP.2021.3124365.

Yiming Cui, Wanxiang Che, Shijin Wang, and Ting Liu. Lert: A linguistically-motivated pre-trained language model. arXiv preprint arXiv:2211.05344, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. URL https: / www. aclweb . org / anthology/N19-1423.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training compression for generative pretrained transformers. arXiv preprint arXiv:2210.17323, 2022.

Georgi Gerganov. llama.cpp. https://github.com/ggerganov/llama.cpp, 2023.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv e-prints, art. arXiv:2106.09685, June 2021. doi: 10.48550/arXiv.2106.09685.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. OpenAssistant Conversations - Democratizing Large Language Model Alignment. arXiv e-prints, art. arXiv:2304.07327, April 2023. doi: 10.48550/arXiv.2304.07327.

Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66-71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URL https://aclanthology.org/D18-2012.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785-794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclanthology.org/D17-1082.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

OpenAI. Introducing chatgpt. https://openai.com/blog/chatgpt, 2022.

OpenAI. GPT-4 Technical Report. arXiv e-prints, art. arXiv:2303.08774, March 2023. doi: 10. 48550/arXiv.2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv e-prints, art. arXiv:2203.02155, March 2022. doi: 10.48550/arXiv.2203.02155.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pp. 3505-3506, 2020.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

Noam Shazeer. Glu variants improve transformer, 2020.

Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023a.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023b.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning Language Model with Self Generated Instructions. arXiv e-prints, art. arXiv:2212.10560, December 2022. doi: 10.48550/arXiv.2212.10560.

Bright Xu. Nlp chinese corpus: Large scale chinese corpus for nlp, September 2019. URL https : //doi.org/10.5281/zenodo. 3402023.

Ziqing Yang, Zihang Xu, Yiming Cui, Baoxin Wang, Min Lin, Dayong Wu, and Zhigang Chen. CINO: A Chinese minority pre-trained language model. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 3937-3949, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics. URL https: //aclanthology.org/2022.coling-1.346.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130b: An open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=-Aw0rrrPUF.

Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems 32, Vancouver, Canada, 2019. URL https : / / openreview. net/references/pdf?id=S1qBAf6rr.
</end of paper 3>


<paper 4>
# Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting 

Haoyang Huang ${ }^{1 *}$ Tianyi Tang ${ }^{2 *}$, Dongdong Zhang ${ }^{1 \dagger}$, Wayne Xin Zhao ${ }^{2}$<br>Ting Song ${ }^{1}$, Yan Xia ${ }^{1}$, Furu Wei ${ }^{1}$<br>${ }^{1}$ Microsoft Research Asia, China<br>${ }^{2}$ Gaoling School of Artificial Intelligence, Renmin University of China<br>https://github.com/microsoft/unilm

![](https://cdn.mathpix.com/cropped/2024_06_04_0233f41188380447f255g-01.jpg?height=594&width=643&top_left_y=725&top_left_x=341)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_0233f41188380447f255g-01.jpg?height=631&width=666&top_left_y=712&top_left_x=1072)

(b)

Figure 1: Comparing the effectiveness of the Cross-Lingual-Thought prompt versus the baseline basic prompt on 7 representative benchmarks covering 27 languages: (a) Enhancing the multilingual capability of text-davinci-003 under the zero-shot learning, and (b) Narrowing the gap between the average performance and the best performance of each task in different languages.


#### Abstract

Large language models (LLMs) demonstrate impressive multilingual capability, but their performance varies substantially across different languages. In this work, we introduce a simple yet effective method, called cross-lingualthought prompting (XLT), to systematically improve the multilingual capability of LLMs. Specifically, XLT is a generic template prompt that stimulates cross-lingual and logical reasoning skills to enhance task performance across languages. We conduct comprehensive evaluations on 7 typical benchmarks related to reasoning, understanding, and generation tasks, covering both high-resource and low-resource languages. Experimental results show that XLT not only remarkably enhances the performance of various multilingual tasks but also significantly reduces the gap between the average performance and the best performance of each task in different languages. Notably, XLT brings over 10 points of average improvement in arithmetic reasoning and open-domain question-answering tasks.


[^0]
## 1 Introduction

Large language models (LLMs) demonstrate impressive multilingual capability in a wide range of natural language processing tasks, including language generation, knowledge utilization, and complex reasoning (Zhao et al., 2023). Their performance in downstream tasks has been shown to reach or even surpass human-level performance (Brown et al., 2020; Chowdhery et al., 2022; Scao et al., 2022). The capabilities of LLMs stem from the extensive volume of training data they leveraged (Kaplan et al., 2020). The training data for current models is primarily dominated by the English language corpus, but it also encompasses data from other languages, as described in GPT3 (Brown et al., 2020), PaLM (Chowdhery et al., 2022), and BLOOM (Scao et al., 2022), etc.

There are over 7,000 languages worldwide, with the vast majority being low-resource or extremely low-resource languages (Forkel et al., 2022). Despite the latest GPT-4 model (OpenAI, 2023) demonstrating some generalization capabilities in

Input

Request: 詹姆斯决定每

周跑 3 次 3 段冲刺, 每段冲刺跑 60 米。他每周一共跑多少米?
I want you to act as an arithmetic reasoning expert for Chinese.

Request: 詹姆斯决定每周跑 3 次 3 段冲刺，每段冲刺跑 60 米。他每周一共跑多少米?

You should retell the request in English.

You should do step-by-step answer to obtain a number answer . You should step-by-step answer the request.

You should tell me the answer in this format 'Answer:'.

## Output

Request: James decides to run 3 sets of 60 -meter sprints three times a week. How many meters does he run in total each week?

Step-by-step answer:

1. James runs 3 sets of 60 -meter sprints, which means he runs $60 \times 3=180$ meters per sprint.
2. James runs 3 sets of 180 meters per week, which means he runs $180 \times 3=540$ meters per week.

Answer: James runs a total of 540 meters per week.

Figure 2: Overview of our method. Given a request, its associated meta information is filled into the placeholders of the XLT template to form the language-independent prompt, which is fed to the LLM to enhance the generation of responses in the desired format.

multilingual tasks as evaluated on the MMLU benchmark (Hendrycks et al., 2021), it is still the case that LLMs do not have equal capability to handle all languages, leading to imbalanced capability across different languages. Furthermore, several evaluation results (Bang et al., 2023; Jiao et al., 2023; Hendy et al., 2023; Zhu et al., 2023) indicate that large models struggle with understanding and generating non-English languages, particularly in low-resource or extremely low-resource languages. Therefore, to democratize language intelligence and minimize performance gaps in different language, it is essential and meaningful to stimulate and enhance the multilingual capability of models in non-English and low-resource languages.

Intuitively, LLMs can improve multilingual capability by augmenting data (Lin et al., 2022) or fine-tuning models (Chen et al., 2021, 2022), but both are computationally expensive. Alternatively, in-context learning with prompts can also boost performance (Brown et al., 2020; Ahuja et al., 2023; Wei et al., 2022c) but is limited to monolingual tasks (Sanh et al., 2022).

This work explores a universal in-context learning approach to enhance the multilingual capability of LLMs. We introduce a simple yet effective method, called cross-lingual-thought prompting (XLT), to enable models to handle various natu- ral language processing tasks across different target languages. Our method employs a generic and language-independent prompt, which eliminates the need to update model parameters. Depending on the task input type, cross-lingual-thought prompting guides the large language model to assume the role of an expert in a specific language for a particular task. Given its predefined meta information, XLT directs LLMs to respond logically through a process involving problem understanding, cross-lingual thinking, task analysis, task execution, and output formatting. During this process, our method is designed to stimulate models' cross-lingual and logical reasoning skills, enabling them to respond to input requests regardless of the language. For enhanced performance, few-shot learning can also be employed with our method by providing an LLM-generated response output as a demonstration using cross-lingual-thought prompting zero-shot learning.

We conduct a comprehensive evaluation to verify the effectiveness of XLT across seven representative multilingual benchmarks of natural language reasoning, understanding, and generation tasks. Each benchmark includes multilingual data covering both high-resource and low-resource languages. The experimental results demonstrate that our method can significantly improve the perfor-

```
I want you to act as a task_name expert for task_language
task_input
You should retell/repeat the input_tag in English.
You should task_goal .
You should step-by-step answer the request.
You should tell me the output_type (output_constraint) in this format ' output_type :'
```

Figure 3: Illustration of XLT template. Referring to Figure 2 and Appendix for instantiated examples.

mance of all benchmarks across languages under both zero-shot and few-shot learning settings. Notably, XLT achieves an average gain of over 10 points on the MGSM and MKQA benchmarks. Furthermore, we observe that our prompting method significantly reduces the gap between the average performance and the best performance of each task in different languages, indicating its potential to democratize language intelligence.

## 2 Cross-Lingual-Thought Prompting

Although LLMs are capable of accepting any input and generating responses, users typically structure their requests in the form of prompts to elicit the desired output. The design of these prompts is crucial for achieving optimal performance on downstream tasks, as LLMs are sensitive to the format of the prompts chosen (Zhao et al., 2021). Through a process called instruction tuning (Wei et al., 2022a), models can develop the ability to follow natural language instructions (Wei et al., 2022b), which can reduce their sensitivity to prompt engineering (Wei et al., 2022a). In accordance with the guidelines of the OpenAI cookbook ${ }^{1}$, we propose a cross-lingual thought prompting template, denoted as the XLT template. This generic template allows LLMs to respond to requests with cross-lingual thought and supports a wide range of multilingual tasks.

Figure 3 displays the XLT template, with the colored sections representing placeholders. Figure 2 showcases an example of instantiated prompt for the Chinese request. The following section will explain the details of constructing XLT.

### 2.1 Construction of XLT

The XLT template is designed to emulate the process humans employ when handling multilingual tasks. Our template is written in English, as English is the dominant language during LLM pre-[^1]

training, and existing research indicates that English prompting is more effective for multilingual tasks (Shi et al., 2023). In contrast to the vanilla prompt that only includes a task description, our XLT template aims to elicit multilingual capability through cross-lingual thoughts. This template comprises six logical instructions in sequence. To complete the template, only seven placeholders need to be filled in based on intrinsic knowledge of the task and the request, as depicted in igure 3.

Role Assigning . First, the model receives a role definition that helps establish the model's behavior. This concept is akin to the system role of ChatGPT $^{2}$. To achieve this, we simply need to fulfill the task name with a known category (such as commonsense reasoning or paraphrase identification), along with the language of the task in the task language field.

Task Inputting . Second, we explicitly append the request as the task input. The request is basically structured in terms of the task type so as to make sure the model can comprehend it. For example, in the natural language inference task, the two sentence inputs are specified with "premise" and "hypothesis", respectively.

Cross-lingual Thinking . We encourage the model to engage in cross-lingual thought by rephrasing the requested content in English, which is the dominant language used as a pivot language by Shi et al. (2023) and Ahuja et al. (2023). Rephrasing the requested content enclosed in the input tag helps the model better understand the request in its native language and knowledge. Our observations suggest that using keywords such as "retell" or "repeat" while rephrasing the content may result in better performance in practice.[^2]

Task Analyzing . After rephrasing the task input, we need to complete the task in task goal. This step is comparable to the task description used in conventional prompting methods. In practice, we can get the task information from the literature or seek assistance from ChatGPT to generate effective prompts for solving the task (Jiao et al., 2023).

CoT Task Solving . We then ask the model to follow the instructions and complete the task step by step. Since LLMs exhibit a strong ability to maintain a chain-of-thought (Wei et al., 2022c), we carefully design instructions to guide the model, with the hope that it will respond to our instructions in a step-by-step manner and utilize the intermediate outputs to aid in solving the task.

Output Formatting . Finally, we should regularize the output format of the model to obtain the exact answer. LLMs are utilized in a zero- or few-shot manner, and they tend to generate texts that may not conform to the format of the target answer. Fortunately, LLMs possess a strong ability to follow instructions, and we can define the output format in terms of output type and output constraint. The output type can be a number, index, or text, while the output constraint is optional and determined based on the task requirements. Output constraint may include length limitations, language specifications, and other relevant factors.

### 2.2 XLT for Few-shot Learning

The above construction of XLT can be directly fed to LLMs to yield outputs, which is performed in the zero-shot learning setting. In addition, we also explore incorporating demonstrations into XLT to enable few-shot learning. Different from previous work that just appends model outputs to the corresponding request (Shi et al., 2023) or utilizes a verbalizer to format the output, our method constructs the demonstrations with better formatted model outputs from a step-by-step processing-based XLT. As illustrated in Figure 4, we first sample a few examples from the development set and incorporate the requested parts into XLT. The zero-shot learning is performed over LLM to collect responses that are further aligned with those of the samples. Only response-aligned requests are assembled with the corresponding model responses to form final demonstrations for few-shot learning. In this way, the demonstrations are constructed with rich logical knowledge via XLT, which will cater to the XLT-based generation of new requests. In practice,

![](https://cdn.mathpix.com/cropped/2024_06_04_0233f41188380447f255g-04.jpg?height=406&width=780&top_left_y=231&top_left_x=1049)

Figure 4: Construction process for few-shot learning.

we can also correct or design the demonstrations for better alignment with the instruction logic.

## 3 Experiments

To comprehensively verify the effectiveness of our method on language-independent generality, we evaluate our XLT template on different LLMs covering various natural language processing tasks in multiple languages.

### 3.1 Experimental Setups

### 3.1.1 Tasks and Benchmarks

We conduct evaluations on seven typical benchmarks related to reasoning, understanding, and generation tasks that can represent different capabilities of LLMs, encompassing both high-resource and low-resource languages. These benchmarks cover 27 different languages, including English (en), German (de), Russian (ru), French (fr), Chinese Simplified (zh), Spanish (es), Japanese (ja), Italian (it), Vietnamese (vi), Turkish (tr), Indonesian (id), Swahili (sw), Arabic (ar), Korean (ko), Greek (el), Thai (th), Bulgarian (bg), Hindi (hi), Estonian (et), Bengali (bn), Tamil (ta), Galician (gl), Urdu (ur), Telugu (te), Javanese (jv), Haitian Creole (ht), and Southern Quechua (qu). In terms of the language distribution statistics in the Common Crawl Monthly Archives ${ }^{3}$ and the language performance of LLMs (Shi et al., 2023; Ahuja et al., 2023), we have arranged them in the order of language frequency from high-resource to low-resource. In particular, the frequency of some underrepresented languages is even less than $0.1 \%$ (e.g., bn, ta, gl, ur, te, jv, ht, qu).

## - Reasoning tasks

- Arithmetic Reasoning. The MGSM (Shi et al., 2023) benchmark contains grade school mathe-[^3]matical problems and asks the model to calculate the correct answer. It covers 11 languages, and we utilize the accuracy score for evaluation.
- Commonsense Reasoning. The XCOPA (Ponti et al., 2020) benchmark contains one premise and two choices. It asks the model to choose which one is the result or cause of the premise. It covers 11 languages from 11 diverse families, and we utilize the accuracy score for evaluation.


## - Understanding tasks

- Natural Language Inference. The XNLI (Conneau et al., 2018) benchmark contains one premise and one hypothesis and requires the model to determine whether the hypothesis is entailed, contradicted, or neutral conditioned on the premise. It covers 15 languages, and we utilize the accuracy score for evaluation.
- Paraphrase Identification. The PAWS-X (Yang et al., 2019) benchmark contains two sentences and requires the model to judge whether they paraphrase each other or not. It covers 7 languages, and we utilize the accuracy score for evaluation.


## - Generation tasks

- Question Answering. The MKQA (Longpre et al., 2021) benchmark contains an open-domain question and asks the model to predict a short answer. Since it has unanswerable questions or long questions that do not have precise answers, we remove these questions during evaluation. It covers 25 languages, and we choose a subset of 10 languages, including de, en, es, fr, ja, ru, th, tr, vi, and zh. We utilize the token overlap F1 score for evaluation.
- Summarization. The XL-Sum* (Hasan et al., 2021) (250 test samples randomly sampled from XL-Sum per language) benchmark contains a long news article and wants the model to summarize it into a short text. It covers 44 languages, and we choose a subset of 6 languages, including en, es, fr, tr, vi, and zh. We utilize the ROUGE-1 score (Lin, 2004) for evaluation.
- Machine Translation. The FLORES* (Costajussà et al., 2022) (200 test samples randomly sampled from FLORES-200 per language) benchmark contains parallel text from Wikimedia projects for 204 languages, yielding over 40,000 translation directions. We choose a subset of 12 directions, including high resource to high resource translation (i.e., $\mathrm{zh} \leftrightarrow \mathrm{ru}$ and $\mathrm{de} \leftrightarrow \mathrm{vi}$ ), high resource to low resource translation (i.e., $\mathrm{zh} \leftrightarrow$ th and $\mathrm{zh} \leftrightarrow \mathrm{jv}$ ), and low resource to low resource translation (i.e., th $\leftrightarrow \mathrm{gl}$ and $\mathrm{jv} \leftrightarrow$ th). We utilize the SacreBLEU score (Papineni et al., 2002; Post, 2018) for evaluation.

Among these benchmarks, MGSM, XCOPA, XNLI, PAWS-X, and MKQA are parallel, i.e., the instances are semantics-equivalent across each language. For all benchmarks, we report the results on the test sets using all instances (Table 5), except for XL-Sum and FLORES-200, where we only sample 250 and 200 examples respectively to show the trend of generation performance. In the fewshot setting, we randomly choose examples from the development set if they have, otherwise, we translate the English training set into corresponding languages to construct several examples.

### 3.1.2 Baselines

Basic Prompt are the vanilla in our experiments that were proposed and suggested in previous work. After determining the prompt, we format each monolingual instance using the English basic prompt. This setting is similar to the monolingual prompting in MEGA (Ahuja et al., 2023). The basic prompts used for the evaluation of each benchmark are listed in Table 5. Note that, we dismiss the baseline using native-language, since MEGA (Ahuja et al., 2023) reveals monolingual prompting is superior to cross-lingual prompting.

Chain-of-Thought (CoT) prompting invokes LLMs to generate a series of intermediate results to solve reasoning tasks (Wei et al., 2022c), which is still effective under multilingual scenarios (Shi et al., 2023). In experiments, we append the instruction "Let's think step-by-step and tell me the answer in the end" after the input to prompt LLMs.

Translate-English leverages the robust capabilities of LLMs in English to tackle multilingual tasks, as suggested by both Shi et al. (2023) and Ahuja et al. (2023). This approach translates instances from other languages into English beforehand. In practice, we utilize the Google Translate API to translate examples into English and apply the basic prompt to format them. Note that, we do not apply this method to generation tasks since they require the output in respective language rather English.

XLT utilizes the proposed template consisting of multiple instructions introduced in Section 2. The
instantiated XLT templates for each benchmark are listed in Table 6.

In few-shot learning scenarios, for basic prompt, we use the same template as an additional input to the model. For XLT, we provide the exemplars with XLT template inputs and anticipate desirable step-by-step outputs as outlined in Figure 4. In the subsequent evaluation, we apply the 5 -shot setting, except for the XL-Sum* experiments, which use the 3 -shot setting due to input length constraints.

### 3.1.3 LLMs

We mainly evaluate two LLMs from the GPT-3.5 series models:

- text-davinci-003 ${ }^{4}$ is trained using instruction tuning and reinforcement learning from human feedback (Ouyang et al., 2022). It can perform a wide range of natural language tasks with satisfactory results.
- gpt-3.5-turbo ${ }^{4}$ is optimized for chat based on text-davinci-003 and suitable for traditional NLP tasks. It is the most capable GPT-3.5 model.

To verify the compatibility of our XLT template, we further incorporate LLaMA-2-Chat (Touvron et al., 2023) (Llama-2-70b-chat-hf) as our base models. It is an open-source model that has been trained through supervised fine-tuning and reinforcement learning from human feedback on the base LLaMA 2 model. In addition, we also refer to the existing results from other LLMs, such as code-davinci-002 ${ }^{4}$, when the evaluation is comparable. During inference, we employ greedy search (i.e., temperature $=0$ ) to generate the LLM responses. We find LLMs have excellent instructionfollowing abilities to respond to our instructions in the given format. Therefore, we just extract the part after "Answer format:" as labels.

### 3.2 Experimental Results

Multilingual Capability. We comprehensively evaluate XLT's performance over seven tasks. The average score of text-davinci-003 is summarized in Figure 1(a) and Table 1, and more details are listed in Appendix A. As for the CoT prompting, it can enhance reasoning tasks while becomes less effective on understanding and generation tasks. In terms of the Translate-En prompting, it can boost the performance in the zero-shot settings while[^4]

may not work well in the few-shot settings. Overall, compared to the three baseline methods, XLT achieves significant improvements over two LLMs for all tasks on both zero-shot and few-shot settings regardless of the language difference, except for a slight drop on the PAWS-X benchmark in the zero-shot setting. It is noted that XLT achieves remarkable gains of nearly 20 points on average in the MGSM benchmark for the arithmetic reasoning task and around 10 points on average in the MKQA benchmark for the open-domain question answering task. The experiments demonstrates the effectiveness of XLT for empowering LLM with multilingual capability.

As for the compatibility test, we list the results of LLaMA-2-Chat on the MGSM benchmark in Table 7. It is notable that LLaMA 2 can also benefit from our cross-lingual-thought, which further demonstrates the generality of our XLT template. However, the gains of LLaMA-2-Chat is not as good as GPT-based models. Our analysis reveals this gap can primarily be attributed to LLaMA 2's poorer multi-step instruction-following ability.

Language Democratization. Furthermore, we try to assess the democratization degree of tasks between languages by defining a "democratization score", which calculates the average percentage of performance attained by different languages relative to the best performance among all languages. Given the evaluation scores of $s_{1}, s_{2}, \ldots, s_{l}$ corresponding to $l$ language on a task, the democratization score is formulated as:

$$
\begin{equation*}
\frac{\sum_{i=1}^{l} s_{i}}{l} / \max \left\{s_{i}\right\}_{i=1}^{l} \tag{1}
\end{equation*}
$$

Table 2 presents the degree of democratization for tasks across languages under both zero-shot learning and few-shot learning, and we further summarize it in Figure 1(b) by averaging all scores per task regardless of the setting and model differences. We can observe that XLT leads to higher democratization scores in general, particularly for XCOPA, and MKQA. As for MGSM, XNLI, and PAWS-X, our XLT can improve performance in multiple languages, where the overall performance of the baseline is consistently lower but the gap between languages is smaller as shown in Tables 7, 9, and 10 . In conclusion, our method can reduce the performance gap between languages and improve the language democratization of LLMs.

| Settings |  | Reasoning |  | Understanding |  | Generation |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | MGSM | XCOPA | XNLI | PAWS-X | MKQA | XL-Sum* | FLORES* |
| Zero-shot | text-davinci-e | 303 |  |  |  |  |  |  |
|  | Basic Prompt | 12.5 | 70.1 | 53.3 | 52.0 | 29.0 | 23.7 | 15.4 |
|  | CoT | 25.7 | 70.9 | 53.0 | 57.8 | 30.9 | 23.8 | 15.8 |
|  | Translate-En | 15.7 | 68.0 | 54.8 | 55.0 | - | - | - |
|  | XLT | 23.9 | 73.3 | 62.4 | 57.1 | 40.2 | 25.2 | 17.7 |
|  | gpt-3.5-turbo |  |  |  |  |  |  |  |
|  | Basic Prompt | 23.3 | 76.9 | 52.6 | 65.5 | 31.6 | 24.7 | 19.1 |
|  | CoT | 45.5 | 78.3 | 54.8 | 61.0 | 14.8 | 25.4 | 19.7 |
|  | Translate-En | 27.1 | 75.7 | 52.2 | 66.8 | - | - | - |
|  | $\overline{\text { XLT }}$ | 70.0 | 80.3 | 65.5 | 63.6 | 42.7 | 26.1 | 21.2 |
| Few-shot | text-davinci-6 |  |  |  |  |  |  |  |
|  | Basic Prompt | 45.5 | 75.6 | 59.1 | 68.7 | 39.1 | 26.8 | - |
|  | Translate-En | 46.5 | 77.4 | 56.9 | 68.5 | - | - | - |
|  | XLT | 55.4 | 81.3 | 67.5 | 72.2 | 49.6 | 27.3 | - |
|  | gpt-3.5-turbo |  |  |  |  |  |  |  |
|  | Basic Prompt |  |  |  |  |  |  | - |
|  | Translate-En | 65.1 | 81.9 | 58.3 | 63.7 | - | - | - |
|  | XLT | 72.5 | 85.9 | 65.0 | 69.1 | $\mathbf{5 2 . 5}$ | 27.9 | - |

Table 1: The average scores in different languages for the seven benchmarks in zero-shot and few-shot settings. We omit the results (denoted as "-") of Translate-En since it is not applicable for generation tasks.

| Settings | Reasoning |  | Understanding |  | Generation <br> MKQA |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  | MGSM | XCOPA | XNLI | PAWS-X |  |
| Zero-shot setting |  |  |  |  |  |
| text-davinci-003 |  |  |  |  |  |
| Basic Prompt | 65.2 | 77.8 | 83.8 | 97.1 | 60.2 |
| CoT | 65.4 | 80.1 | 83.5 | 89.5 | 61.4 |
| Translate-En | 77.2 | 78.7 | 86.0 | 95.3 | 51.6 |
| XLT | 68.5 | 82.1 | 80.7 | 88.4 | 78.7 |
| gpt-3.5-turbo |  |  |  |  |  |
| Basic Prompt | 73.0 | 83.6 | 80.5 | 89.0 | 61.8 |
| CoT | 66.7 | 85.7 | 80.7 | 88.9 | 46.4 |
| Translate-En | 80.4 | 84.6 | 79.8 | 90.7 | 54.1 |
| XLT | 84.1 | 89.1 | 88.0 | 96.2 | 75.3 |
| Few-shot setting |  |  |  |  |  |
| text-davinci-003 |  |  |  |  |  |
| Basic Prompt | 75.4 | 82.0 | 82.5 | 88.2 | 74.3 |
| Translate-En | 77.1 | 82.6 | 79.5 | 87.8 | 68.5 |
| XLT | 84.5 | 85.6 | 85.3 | 91.6 | 82.7 |
| gpt-3.5-turbo |  |  |  |  |  |
| Basic Prompt | 76.1 | 84.1 | 83.6 | 94.4 | 82.1 |
| Translate-En | 78.6 | 86.4 | 79.2 | 95.4 | 71.3 |
| XLT | 86.2 | 89.7 | 84.3 | 94.1 | 83.1 |

Table 2: The democratization degree of tasks against languages.

### 3.3 Further Analysis

In this section, we further investigate the factors that affect the performance of XLT and how they affect various multilingual benchmarks.

### 3.3.1 Ablation of XLT

For the XLT variants, we mainly conduct experiments to compare the following strategies:

- Ablating the instructions. Since our XLT consists of six logical instructions, we disable the Role Assigning, Cross-lingual Thinking, and CoT Task Solving instructions separately to analyze the contribution per instruction.
- Reordering the instructions. Considering the logicality of our instructions, we further change the order of the instructions in XLT to explore whether LLMs will handle tasks differently and lead to different results.
- Changing the content word. As prompts are usually sensitive to the word choice, we verify the robustness of XLT when alternating the rephrasing keyword with "retell", "repeat", and "translate" in the cross-lingual thinking instruction.

The outcomes are presented in Table 3, indicating that XLT surpasses almost all the variants, thereby validating the effectiveness and reasonableness of our proposed XLT method.

The effectiveness of each instruction. The results from the "Instruction Ablation" row indicate that: (1) Cross-lingual Thinking yields more significant gains compared to other instructions. This suggests that the LLM's ability of cross-lingual thinking is activated, allowing it to utilize its knowledge in English to solve tasks effectively; (2) Removing Role Assigning from XLT impedes the model's

| Settings |  | MGSM |  | XNLI |  | FLORES* |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | de | $\mathbf{z h}$ | hi | vi | $\mathbf{j v} \rightarrow \mathbf{z h}$ | $\mathbf{z h} \rightarrow \mathbf{j v} \quad$ r |
|  | XLT | \| 79.8 | 72.6 | 61.3 | 64.8 | 19.0 | 10.5 |
| Instruction <br> Ablation | w/o Role Assigning | 76.6 | 69.2 | 57.8 | 63.9 | 16.2 | 8.8 |
|  | w/o Cross-lingual Thinking | 75.6 | 62.0 | 56.1 | 62.2 | 13.2 | 8.2 |
|  | w/o CoT Task Solving | 77.0 | 68.0 | 62.9 | 65.2 | 16.8 | 9.2 |
| Instruction <br> Order | Swap Role Assigning and Task Inputting | 77.2 | $71.8 \quad$ | 54.2 | 61.5 | 19.6 | 11.2 |
|  | Swap Role Assigning and Task Analyzing | 76.8 | 70.8 | 61.0 | $64.0 \quad 0$ | 15.8 | 8.8 |
|  | Swap Cross-lingual Thinking and Task Analyzing | 79.0 | 71.2 | 59.5 | 63.4 | 16.5 | 9.7 |
| Rephrasing <br> Word | $w /$ retell | 79.8 | 72.6 | 61.3 | $64.8 \quad$ | 18.2 | 10.3 |
|  | $w /$ repeat | 77.6 | 68.0 | 60.7 | 64.6 | 19.0 | 10.5 |
|  | w/ translate | 76.4 | 70.0 | 60.1 | 64.5 | 17.5 | 10.2 |

Table 3: Performance comparison across different variants of XLT. All the experiments are conducted using gpt-3.5-turbo under the zero-shot setting.

| Demonstration format | en | de | ru | fr | zh | es | ja | sw | th | bn | te | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Basic input + Basic output | 84.0 | 79.2 | 78.8 | 78.8 | 70.8 | 81.2 | 68.8 | 70.8 | 68.8 | 65.2 | 44.8 | 71.9 |
| Basic input + XLT output | 82.4 | 72.4 | 71.2 | 75.2 | 64.4 | 78.8 | 63.2 | 66.8 | 53.6 | 54.8 | 32.4 | 65.0 |
| XLT input + XLT output | 84.8 | 81.4 | 80.2 | 79.2 | 71.8 | 81.6 | 72.8 | 71.2 | 69.8 | 64.4 | 40.8 | $\mathbf{7 2 . 5}$ |

Table 4: Performance comparison across different few-shot variants on the MGSM benchmark. All the experiments are conducted with 5 demonstrations using gpt-3.5-turbo.

understanding of the ultimate goal for diverse multilingual tasks, highlighting the task transferability of XLT; and (3) the better performance of XLT can also be attributed to CoT Task Solving, which requires the model to respond to complex instructions in a step-by-step manner.

The order of logical instructions. The performance drop is evident when the order of our designed logical instructions is switched. When designing XLT, we have taken into account the process by which humans solve multilingual problems, and this experiment further confirms the optimum order of our XLT template. Placing the Role Assigning instruction later may confuse the model initially. Additionally, conducting Cross-lingual Thinking before Task Analyzing is crucial since we rely on the English task-solving abilities of LLMs to handle multilingual tasks.

The robustness of word choice for rephrasing keywords. We can find that different words indeed affect the performance of XLT, but it is less sensitive to the other variants. Through experimentation, we have determined that "repeat" yields better results for text summarization and machine translation, while "retell" is more suitable for the remaining five tasks. Our aim is to provide XLT with a more unified template, while still allowing users to fine-tune specific keywords for optimal performance in their tasks.

### 3.3.2 Effectiveness of XLT Few-shot Learning

 As mentioned in Section 2.2, the construction of demonstrations for XLT few-shot learning differs from the previous method. We have compared XLT and basic prompt. Here, we focus on the construction of the demonstration input-output pairs and compare various demonstrations that may be used to perform XLT few-shot learning. The illustrations can be found in Figure 5.- Basic prompt input + Basic prompt output: This is the normal demonstration format used in most of the previous work.
- Basic prompt input + XLT output: This ablation is to separate the effect of input and output formats in the demonstration.
- XLT input + XLT output: This is the method that we used in this work.

Observing the experimental results presented in Table 4, we can conclude that: (1) Our XLT fewshot learning outperforms all other variants, thus confirming its effectiveness. (2) The use of normal demonstrations for XLT few-shot learning leads to a decrease in performance. (3) Merely incorporating XLT as a demonstration input without its output does not result in any improvements. (4) Consistency in the demonstration for few-shot learning
is crucial, implying that the demonstration inputoutput format should align better with its zero-shot learning input-output format.

## 4 Related Work

### 4.1 LLM Capability Understanding

Despite the impressive capabilities of LLMs, it is crucial to determine their impact on natural language processing tasks. Liang et al. (2022) conduct a comprehensive evaluation of LLMs from various perspectives, such as accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. Bang et al. (2023) extensively evaluate the ChatGPT model on multiple natural language processing tasks and find that the model performs well in high-resource languages but exhibits certain limitations in low-resource and non-Latin script languages. Additionally, studies by Jiao et al. (2023) and Hendy et al. (2023) compare different GPT models with supervised models for machine translation tasks and find that GPT models have competitive translation abilities in high-resource languages but perform less effectively in low-resource languages. It is worth noting that achieving multilingual generative AI capability necessitates crosslingual knowledge to further improve the model's performance. In this context, Ahuja et al. (2023) evaluate the multilingual task understanding ability of GPT models and attempt to enhance their task processing abilities in other languages using English knowledge. Our work also focuses on evaluating the multilingual capabilities of LLMs, including reasoning, understanding, and generative capabilities. Our evaluations indicate that LLMs exhibit differences in high-resource and low-resource abilities, which necessitates additional efforts to enhance their multilingual capability.

### 4.2 Multilingual Task Processing

Multilingual knowledge has been shown to be exploitable and transferable between languages to improve model performance (Devlin et al., 2019; Conneau et al., 2020; Raffel et al., 2020; Ouyang et al., 2021; Chi et al., 2021). While much research has been devoted to multilingual understanding tasks, multilingual generation tasks are more challenging, particularly when the target language is lowresource or non-English (Ma et al., 2021; Liu et al., 2020). Two methods can enable models to support multilingual task processing: one is training a supervised model that covers multiple languages for multilingual processing (Costa-jussà et al., 2022), and the other is training a pre-trained model and using fine-tuning to transfer knowledge among languages to achieve multilingual capability (Chen et al., 2021, 2022). However, the emergence of LLMs has made it possible to directly process multilingual tasks via in-context learning (Brown et al., 2020; Ahuja et al., 2023). These LLMs, with hundreds of billions or even trillions of parameters, require a significant amount of computation resources for training, making traditional fine-tuning methods less feasible. To improve the generative ability of LLMs, researchers explore in-context learning methods that do not require updating model parameters, such as few-shot prompting (Vilar et al., 2022), automatic prompt learning (Shin et al., 2020), task-instruction prompting (Ye et al., 2023), chain-of-thought prompting (Wei et al., 2022c), etc. Our work builds upon these methods and proposes an optimized, generic, and language-independent prompt to enhance the multilingual capability of LLMs.

## 5 Conclusion

This work investigates the language processing capabilities of large language models in multilingual settings and expects to develop a universal framework for handling diverse multilingual tasks. To accomplish this goal, we propose a generic prompt, referred to as XLT, to enhance the multilingual capability and reduce the performance gaps among languages in tasks related to language understanding, reasoning, and generation in non-English and low-resource languages. Although our method is generally applicable across tasks and languages, we discovered that prompting design factors such as instruction logic and word choice have explicit impacts on its effectiveness. Cross-language thinking in XLT is particularly effective. Finally, we hope this work can inspire further research to prioritize the development of generic prompting. By doing so, large language models can encompass a wider range of modalities and languages.

## Acknowledgements

Tianyi Tang and Xin Zhao are supported by National Natural Science Foundation of China under Grant No. 62222215, Beijing Natural Science Foundation under Grant No. 4222027 and L233008.

## Limitations

Due to limitations imposed by the evaluation benchmarks and OpenAI API cost, we conducted tests on 27 languages, which merely scratch the surface of the vast array of languages in the world. Besides, our XLT template is based on English. It deserves to explore whether the template written in task language can lead to better performance and how to better construct the instruction in each language. Furthermore, we only verify the effectiveness of our method on two GPT-based models (i.e., text-davinci-003 and gpt-3.5-turbo) and LLaMA-2-Chat. It is worthwhile to investigate the generality of our template on more models, such as BLOOM and PaLM.

## References

Kabir Ahuja, Rishav Hada, Millicent Ochieng, Prachi Jain, Harshita Diddee, Samuel Maina, Tanuja Ganu, Sameer Segal, Maxamed Axmed, Kalika Bali, et al. 2023. Mega: Multilingual evaluation of generative ai. arXiv preprint arXiv:2303.12528.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc.

Guanhua Chen, Shuming Ma, Yun Chen, Li Dong, Dongdong Zhang, Jia Pan, Wenping Wang, and Furu Wei. 2021. Zero-shot cross-lingual transfer of neural machine translation with multilingual pretrained encoders. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 15-26, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Guanhua Chen, Shuming Ma, Yun Chen, Dongdong Zhang, Jia Pan, Wenping Wang, and Furu Wei. 2022. Towards making the most of cross-lingual transfer for zero-shot neural machine translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 142-157, Dublin, Ireland. Association for Computational Linguistics.

Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, and Ming Zhou. 2021. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3576-3588, Online. Association for Computational Linguistics.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 84408451, Online. Association for Computational Linguistics.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475-2485, Brussels, Belgium. Association for Computational Linguistics.

Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Robert Forkel et al. 2022. Glottocodes: Identifiers linking families, languages and dialects to comprehensive reference information. Semantic Web, 13(6):917924 .

Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang,

M. Sohel Rahman, and Rifat Shahriyar. 2021. XLsum: Large-scale multilingual abstractive summarization for 44 languages. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4693-4703, Online. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are gpt models at machine translation? a comprehensive evaluation. arXiv preprint arXiv:2302.09210.

Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Xing Wang, and Zhaopeng Tu. 2023. Is chatgpt a good translator? yes with gpt-4 as the engine. arXiv preprint arXiv:2301.08745.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Viet Dac Lai, Nghia Trung Ngo, Amir Pouran Ben Veyseh, Hieu Man, Franck Dernoncourt, Trung Bui, and Thien Huu Nguyen. 2023. Chatgpt beyond english: Towards a comprehensive evaluation of large language models in multilingual learning. arXiv preprint arXiv:2304.05613.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74-81, Barcelona, Spain. Association for Computational Linguistics.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9019-9052, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and
Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726-742.

Shayne Longpre, Yi Lu, and Joachim Daiber. 2021. MKQA: A linguistically diverse benchmark for multilingual open domain question answering. Transactions of the Association for Computational Linguistics, 9:1389-1406.

Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Alexandre Muzio, Saksham Singhal, Hany Hassan Awadalla, Xia Song, and Furu Wei. 2021. Deltalm: Encoder-decoder pre-training for language generation and translation by augmenting pretrained multilingual encoders. arXiv preprint arXiv:2106.13736.

OpenAI. 2023. Gpt-4 technical report. arXiv.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730-27744. Curran Associates, Inc.

Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. 2021. ERNIE-M: Enhanced multilingual representation by aligning cross-lingual semantics with monolingual corpora. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 27-38, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311-318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362-2376, Online. Association for Computational Linguistics.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186191, Belgium, Brussels. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi

Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations.

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. 2020. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4222-4235, Online. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

David Vilar, Markus Freitag, Colin Cherry, Jiaming Luo, Viresh Ratnakar, and George Foster. 2022. Prompting palm for translation: Assessing strategies and performance. arXiv preprint arXiv:2211.09102.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022a. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022b. Emergent abilities of large language models. Transactions on Machine Learning Research. Survey Certification.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022c. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824-24837. Curran Associates, Inc.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3687-3692, Hong Kong, China. Association for Computational Linguistics.

Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, and Minjoon Seo. 2023. In-context instruction learning. arXiv preprint arXiv:2302.14691.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697-12706. PMLR.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023. Multilingual machine translation with large language models: Empirical results and analysis. arXiv preprint arXiv:2304.04675.
</end of paper 4>


