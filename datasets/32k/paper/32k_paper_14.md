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
# Vikhr: The Family of Open-Source Instruction-Tuned Large Language Models for Russian 

Aleksandr Nikolich<br>ITMO University<br>Konstantin Korolev<br>HSE University<br>Artem Shelmanov<br>MBZUAI<br>alexdragannikolich@gmail.com korolevko@icloud.com artem.shelmanov@mbzuai.ac.ae


#### Abstract

There has been a surge in the development of various Large Language Models (LLMs). However, text generation for languages other than English often faces significant challenges, including poor generation quality and the reduced computational performance due to the disproportionate representation of tokens in model's vocabulary. In this work, we address these issues and introduce Vikhr, a new state-of-the-art open-source instruction-tuned LLM designed specifically for the Russian language. "Vikhr" refers to the name of the Mistral LLM series and means "strong gust of wind." Unlike previous efforts for Russian that utilize computationally inexpensive LoRA adapters on top of English-oriented models, Vikhr features an adapted tokenizer vocabulary and undergoes the continued pre-training and instruction tuning of all weights. This approach not only enhances the model's performance but also significantly improves its computational and contextual efficiency. The remarkable performance of Vikhr across various Russian-language benchmarks can also be attributed to our efforts in expanding instruction datasets and corpora for continued pre-training. Vikhr not only sets the new state of the art among open-source LLMs for Russian, but even outperforms some proprietary closed-source models on certain benchmarks. The model weights, instruction sets, and code are publicly available ${ }^{1}$.


## 1 Introduction

Instruction tuning has unlocked in Large Language Models (LLMs) vast zero-shot capabilities without the need of careful prompt engineering (Ouyang et al., 2022). The most rapid research and development efforts are currently devoted to English LLMs. There has been a surge in English open-source models: Llama series (Touvron et al., 2023a,b), Mistral series (Jiang et al., 2023), Vicuna series (Chiang et al., 2023), etc. This growth is driven[^0]

by the abundance of raw training data in English and dedicated efforts to create comprehensive sets of instruction-output pairs. Despite the fact that LLMs oriented on English have some multilingual capabilities (Zhao et al., 2024) due to small portions of texts in various languages leaked into their training datasets (Touvron et al., 2023a), their overall performance in these languages remains relatively low. Although they can usually generate portions of coherent texts, these models struggle with reasoning in non-English languages, lack culturespecific knowledge, and are highly inefficient in terms of tokenization. This inefficiency arises due to the way bite-pair tokenization algorithms work: they split the infrequent words into multiple tokens. Since multilingual data typically represents a small portion of the training dataset, non-English words are often split in many pieces. This leads to more steps during prompt processing and text generation, shorter effective context windows, and ultimately lower quality (Tikhomirov and Chernyshev, 2023; Petrov et al., 2024). This disparity places non-English languages at a disadvantage.

There is a research direction focused on developing multilingual LLMs that work well for multiple popular languages: BLOOMz (Muennighoff et al., 2023), mGPT (Shliazhko et al., 2022), Bactrian-X (Li et al., 2023), PALO (Maaz et al., 2024), Aya101 from CohereAI (Üstün et al., 2024), etc. These models are typically trained on rich multilingual datasets and are less skewed towards English. However, when aiming to perform well across multiple languages simultaneously, these models must still share their vocabulary and parameters. This often hinders their performance for each particular language in isolation, especially for the popular smaller model sizes, such as 7B and 13B.

The goal of maximizing the LLM performance for a specific language within a certain number of parameters has led researchers to develop bi-lingual LLMs. For example, Jais (Sengupta et al., 2023)
focus only on English and Arabic. The inclusion of English data in pre-training alongside Arabic data is motivated by the significantly larger volume of English data available. This helps LLMs substantially enhance skills such as logical and common sense reasoning, which are also applied when generating text in Arabic.

Russian is one of the high-resource languages and is typically represented in multilingual LLMs. Additionally, there are several proprietary closedsource LLMs, such as MTS AI, GigaChat, and YandexGPT, that meet or even surpass their Englishoriented flagship competitors when it comes to text processing and generation in Russian. However, controllable research often requires white-box access to LLM logits and layer outputs, the ability to modify weights and a model architecture, and consistent answers for reproducibility, which is often impossible in closed-source LLMs due to their constant development and retirement. There are only a few open-source LLMs designed for Russian (such as Saiga (Gusev, 2023), ruGPT (AI Forever, 2022), ruadapt (Tikhomirov and Chernyshev, 2023)). Of these, only Saiga and ruadapt are instruction-tuned.

This work aims to build an efficient and effective open-source instruction-following LLM for Russian facilitating multilingual natural language processing research. Building even a small LLM that targets a particular language from scratch requires a lot of computational resources. Consequently, many researchers simply fine-tune LoRA adapters (Hu et al., 2021) for English-oriented LLMs on some language-specific data. While this approach can improve model generation quality, it does not address computational inefficiency because the tokenizer and model vocabulary remain unchanged. In contrast, our approach not only fine-tunes a base LLM on Russian language data but also reconstructs its underlying tokenizer and vocabulary, alongside suggesting an improved method for continued pre-training. Additionally, we have significantly expanded the available Russian datasets for instruction tuning. The developed LLM achieves state-of-the-art results for the Russian language among other open-source counterparts across a wide range of benchmarks.

Contributions of the paper are the following:

- We have constructed Vikhr - a state-of-theart open-source instruction-following LLM oriented on the Russian language. In addition to its high generation quality, Vikhr features an efficient tokenizer that enables rapid text generation and good context utilization.
- We have developed a pipeline for adapting English-oriented LLMs to the Russian language. The pipeline implements vocabulary adaptation, continued pre-training with regularization to prevent "catastrophic forgetting", and instruction tuning.
- We have expanded the datasets for continued pre-training of Russian language models and previously available instruction datasets.
- We conducted an extensive evaluation of several open-source LLMs on evaluation benchmarks for Russian, demonstrating that Vikhr achieves new state-of-the-art results.


## 2 Related Work

One of the first notable series of generative LLMs for Russian is ruGPT (AI Forever, 2022; Zmitrovich et al., 2023). The authors created several models trained for the vanilla language modelling task with the sizes of up to $13 \mathrm{~b}$. The models were created from the scratch and trained on large Russian corpora. They are able to handle the linguistic nuances of Russian more effectively than multilingual models (Muennighoff et al., 2022). Since the training data was mostly in Russian, these models have efficient tokenization, but the lack of multilingual data (e.g. in English) limits their performance. ruGPT models are not instruction tuned.

Gusev (2023) suggests to leverage reasoning capabilities of existing English-oriented LLMs and adapt them to the Russian language by training LoRA adapters. They also create an Alpaca-like set of Russian instruction-output pairs and performed instruction tuning. They have established the Saiga model series, which has a competitive performance and used to be a reasonable choice for off-the-shelf open-source Russian LLM for the past year. However, the tokenizer in theses models is not adapted, so they experience issues with context and computational efficiency.

Tikhomirov and Chernyshev (2023) address these issues in Saiga. In addition to model tuning on Russian data, they also adapt the model tokenizer. They note that improving tokenization helps to both improve the efficiency of the model and its performance while reducing memory consumption. However, during continued pre-training,

| Content | Length | Tokenization Result |
| :--- | :--- | :--- |
| Original <br> Sentence | 31 | Машинное обучение изме- <br> няет мир |
| Mistral Tok- <br> enizer | 13 | ''Ма', 'шин', 'ное', 'об', 'у', <br> 'чение', 'из', 'мен', 'я', 'ет' <br> 'ми', 'р'] |
| Vikhr Tok- <br> enizer | ['Ма', 'шин', 'ное', 'обуче- <br> ние', 'изменяет', 'мир'] |  |

Table 1: Tokenizer comparisons between the original Mistral model and Vikhr

the authors freeze the model weights except LM heads and token embeddings, which probably results in the suboptimal performance.

In this work, we take advantage of pre-trained English-oriented LLMs, adapt LLM tokenizer for better computational efficiency, leverage continued pre-training on vast Russian-language corpora with regularization for preventing "catastrophic forgetting", construct a novel extended set of Russian instruction-output pairs, and perform instruction tuning. The created LLM adaptation pipeline along with the data for continued pre-training and instruction tuning enables Vikhr to achieve new state-ofthe-art results for Russian, maintain high performance for English, and demonstrate high computational efficiency.

## 3 LLM Construction Pipeline

The construction of Vikhr starts from one of English-oriented LLMs. In this work, we discuss the Vikhr model based on Mistral 7B. The strong logical and common reasoning capabilities, as well as the extensive world knowledge present in these LLMs provide an excellent starting point for our model. These features partially transfer to Vikhr, enhancing its performance in generating text in Russian. The process of LLM adaptation to Russian starts with the vocabulary adaptation. Then we perform continued pre-training of the LLM on large Russian datasets to mitigate the vocabulary shift and introduce culture specific knowledge. Finally, we perform fine-tuning of Vikhr on a set of instruction-output pairs in Russian.

### 3.1 Vocabulary Adaptation

The big drawback of English-oriented LLMs is that each Russian word would be split into multiple tokens: a common case is when symbols in the word become an individual tokens (see example in Table 1). This slows down the generation by

![](https://cdn.mathpix.com/cropped/2024_06_04_08cc0a1c12d55c45a371g-3.jpg?height=500&width=742&top_left_y=247&top_left_x=1068)

Figure 1: The Vikhr tokenizer efficiency in comparison to tokenizers of other models.

| Data Source | Approx. size <br> (GB) | Tokens <br> (Billion) |
| :--- | :---: | :---: |
| Scientific papers | 20 | 2.5 |
| News articles | 4 | 1 |
| Wikipedia | 25 | 4 |
| Habr | 6 | 1 |
| Other sources | 20 | 2.5 |

Table 2: The statistics of the Russian-language datasets for continued pre-training.

multiple times, reduces the amount of information that could be stored in the context, and drastically hurts the generation quality.

To mitigate this problem in Vikhr, we adopt the approach suggested in (Cui et al., 2023; Tikhomirov and Chernyshev, 2023), where authors rebuild the tokenizer using a language-specific corpus. In particular, we trained a SentencePiece tokenizer (Kudo and Richardson, 2018) with a 40k vocabulary on the RuLM dataset (Gusev, 2023). As can be seen from Figure 1, the resulting tokenizer for Russian is much more efficient than the tokenizer of the original English-oriented model.

### 3.2 Continued Pre-training

The new vocabulary requires also new embedding matrices and LM heads. The tokens that were present in the original vocabulary are initialized with the old embeddings, the new tokens are initialized by averaging the embeddings of their pieces in the original embedding matrix (Hewitt, 2021). The similar approach is also applied to LM heads. Training model with these modifications requires much more computational resources than the mainstream technique for adaptation of LLMs to new languages based on LoRA adapters ( $\mathrm{Hu}$ et al., 2021), as it requires to perform continued

| Hyperparam. | Value |
| :--- | :---: |
| LR | $1 \times 10^{-3}$ |
| AdamW eps | $1 \times 10^{-8}$ |
| Num warmup steps | 10 |
| AdamW betas | $0.99,0.95$ |
| Accumulation steps | 128 |
| Batch size | 3 |
| Epochs | 1 |
| Sequence length | 1024 |

Table 3: The hyperparameters for continued pretraining.

pre-training of the whole model and on much more language-specific data to mitigate the shift in the vocabulary.

The dataset for continued pre-training is constructed from Russian Wikipedia, news articles, scientific papers, top $100 \mathrm{k}$ up-voted posts on Habr, and some other sources. The statistics of these datasets is presented in Table 2. The total number of tokens used for this step is 11 billion.

We note that the continued pre-training of a LLM might partially eliminate the reasoning capabilities present in the original English-oriented model. This drastically affects the model performance. In our preliminary experiments, continued pre-training may result even in worse performance on Russian benchmarks compared to the original model. To alleviate the "catastrophic forgetting", we use the loss regularization with KL penalty between the probability distribution of Vikhr and the reference English-oriented original LLM:

$$
\begin{equation*}
L_{\mathrm{Vikhr}}=L_{\mathrm{CE}}+K L\left(P_{\mathrm{Vikhr}} \| P_{\mathrm{Ref}}\right) \tag{1}
\end{equation*}
$$

In practice, we implement this approach using the SLERP interpolation of model losses (Goddard et al., 2024).

To speed up the process of continued pretraining, we use an optimized Flash attention implementation ${ }^{2}$. As an optimization algorithm, we leverage AdamW as it trades some memory efficiency in favor of robustness to the hyperparameter choice. The hyperparameters used for continued pre-training are presented in Table 3.

### 3.3 Instruction Tuning

Instruction tuning is an essential step in reaching high zero-shot performance with LLMs. It also allows to obtain more natural communication[^1]

| Instruction Set | Language | \# instances |
| :--- | :---: | :---: |
| Veles | Russian | $30 \mathrm{k}$ |
| Nectar | English | $50 \mathrm{k}$ |
| Saiga | Russian | $100 \mathrm{k}$ |
| ruFLAN | Russian | $500 \mathrm{k}$ |

Table 4: The statistics of instruction datasets.

with the model without complex prompting. Further fine-tuning techniques such as RLHF (Ouyang et al., 2022), which require input from the assessors, are also crucial for such tasks as multicriteria alignment. However, the most significant performance gains are still achieved through instruction tuning (Jha et al., 2023).

Previously, Gusev (2023) constructed an opensource set of instruction-output pairs for the Russian language (Saiga). The core Saiga dataset was created similar to Alpaca by querying ChatGPT (gpt-3.5-turbo) (Taori et al., 2023). In this work, we extend this set by translating two English instruction datasets. First, we translated instructions for the FLAN model (Wei et al., 2021) and generated answers in Russian using ChatGPT. Originally, FLAN instructions were constructed automatically from annotated datasets using templates to facilitate multitask and zero-shot capabilities of seq2seq models. Later, it was shown that this data also helps to improve decoder-only chat-oriented models as well. Second, we construct Veles ${ }^{3}$ by translating the English OpenHermes (Teknium, 2023) instruction dataset. We also include without translation Nectar ${ }^{4}$ (Zhu et al., 2023) - the English instruction dataset. It helps to keep the performance of Vikhr high also for English. Since the majority of the outputs were machine generated there are many low quality outputs. To mitigate this problem, we filtered out low quality pairs using a reward model trained on human data. The statistics of the Vikhr instruction datasets is presented in Table 4.

Contrary to Saiga, we do not use LoRA adapters and just as in the phase of continued pre-training, we update all model parameters. The hyperparameters for the instruction tuning phase are presented in Table 5.[^2]

| Hyperparam. | Value |
| :--- | :---: |
| LR | $1 \times 10^{-5}$ |
| AdamW, eps | $1 \times 10^{-8}$ |
| Num warmup steps | 10 |
| AdamW, betas | $0.99,0.95$ |
| Accumulation steps | 64 |
| Batch size | 3 |
| Num epochs | 3 |
| Sequence length | 1024 |

Table 5: The hyperparameters for instruction tuning.

### 3.4 Hardware

Vikhr was trained on eight NVIDIA A100 GPUs 80GB. We spend approximately 1,000 GPU hours for the continued pre-training phase and 60 hours for instruction tuning.

## 4 Experiments

### 4.1 Experimental Setup

Benchmarks. The evaluation was performed on MMLU (Hendrycks et al., 2021), Ru-MMLU ${ }^{5}$, CheGeKa, Russian SuperGLUE (Shavrina et al., 2020), and MERA (Fenogenova et al., 2024). MMLU (En-MMLU) evaluates LLMs across 57 subjects with multiple-choice questions, assessing a model's broad knowledge and reasoning abilities. We use this benchmark to verify that the model retains bi-lingual capabilities. In the results, we report the accuracy @ 1 score. RuMMLU is a translation of MMLU with GPT-3.5 to Russian. Just as for MMLU, we report the accuracy @ 1 score. CheGeKa is based on questions from the game "What? Where? When?". This benchmark contains challenging open-ended questions, requiring logical reasoning and world knowledge. It includes 29,376 training and 416 test instances. The reported evaluation metric is the F1 score. Russian SuperGLUE is a benchmark similar to well-known English SuperGLUE (Wang et al., 2019). It tests LLMs on various natural language understanding tasks like reading comprehension and textual entailment. The metric reported in the results is accuracy@ 1. The MERA benchmark encompasses 21 evaluation tasks for generative LLMs in 11 skill domains. Note that among other tasks MERA also includes CheGeKa, RuMMLU, and one of the subtasks of SuperGLUE (RWSD). The reported evaluation metric is the total score, which is the average of scores across all non-diagnostic tasks.[^3]

Baselines. We compare Vikhr to six open-source and two proprietary closed-source competitors of the similar size. Open-source models: aya101 - a massively multilingual LLM from CohereAI that follows instructions in 101 languages $^{6}$, it shows state-of-the-art results among massively multilingual LLMs; Mistral-7B-0.2-instruct - an Englishoriented LLM that was used as the base model for Vikhr; rccmsu/ruadapt_mistral_saiga_7b_v0.1 - a Russian-oriented LLM that was constructed from the Mistral model using similar adaptations of the tokenizer, token embeddings, and the LM head (Tikhomirov and Chernyshev, 2023); saiga-mistral7b-lora and saiga-llama3-8b - two versions of the Saiga models based on English-oriented LLMs and obtained by fine-tuning LoRA adapters on the Saiga instruction dataset ${ }^{7}$. Closed-source proprietary models for Russian: MTS AI Chat ${ }^{8}$ and GigaChat-7b. The access to GigaChat weights is closed, so the reported results are taken from the leaderboards ${ }^{9}$. The results of MTS AI Chat are also taken from the leaderboard ${ }^{10}$.

### 4.2 Results

The evaluation results are presented in Table 6. As we can see, Vikhr outperforms all open-source models, including the ones that were built specifically for Russian. It also slightly outperforms its parent model Mistral on the En-MMLU benchmark, which might be the result of longer pre-training. The second place with close scores for all 4 Russian language benchmarks is obtained by the Saiga model based on recently released Llama-3. The high scores of this model probably are the result of the transfer of the outstanding performance of Llama-3. Since Saiga based on Llama-3 outperforms Saiga based on Mistral, we expect that applying our adaptation pipeline to Llama-3 would also help further improving the state of the art.

We note that the original Mistral-7B-0.2-instruct, despite being an English-oriented model, demonstrates competitive performance in 3 out of 4 Russian benchmarks. This indicates demonstrates that such models could be viable alternatives. The only dataset, where its performance is very low is CheGeKa, which is related to open-ended question-[^4]

| LLM | Pre-train on <br> Russian | Training <br> Method | En-MMLU | Ru-MMLU | CheGeKa | Russian <br> SuperGLUE | MERA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MTS AI Chat 7B (closed-source) $\diamond$ | false | sft+dpo | - | 0.689 | 0.083 | 0.56 | 0.479 |
| GigaChat-7B (closed-source) $\diamond$ | true | sft+dpo | - | 0.67 | $0.451^{*}$ | $0.71^{*}$ | 0.479 |
| aya101 | false | pt+sft | 0.41 | 0.37 | 0.005 | 0.36 | 0.320 |
| Mistral-7B-Instruct-v0.2 | false | none | 0.60 | $\underline{0.78}$ | 0.005 | 0.57 | 0.400 |
| rccmsu/ruadapt-mistral-7b-v0.1 | false | pt+sft | 0.61 | 0.72 | 0.005 | 0.64 | 0.421 |
| rugpt13b | true | none | 0.25 | 0.25 | 0.132 | 0.52 | 0.208 |
| saiga-mistral-7b-lora | false | sft | 0.60 | 0.76 | 0.223 | 0.64 | 0.442 |
| saiga-llama3-8b | false | sft | 0.59 | $\underline{0.78}$ | $\underline{0.225}$ | $\underline{0.66}$ | $\underline{0.476}$ |
| Vikhr-7B-instruct_0.2 | true | pt+sft | $\mathbf{0 . 6 2}$ | $\underline{\mathbf{0 . 8 0}}$ | $\mathbf{0 . 2 3 1}$ | $\mathbf{0 . 6 7}$ | $\mathbf{0 . 4 8 5}$ |

Table 6: Evaluation results for Russian and multilingual LLMs. Pre-train on Russian means that the model underwent (continued) pre-training on Russian data. The following abbreviations are used: sft - instruction tuning, $\mathrm{pt}-$ (continued) pre-training; dpo - direct preference optimization. $\diamond$ The results for GigaChat and MTS AI are taken from the leaderboards. The best result among open-source models is highlighted with bold, the second best is underscored. The best result among closed-source proprietary models is marked with *.

answering. This may be due to the lack of culturespecific knowledge, as the English-oriented model has not seen much Russian texts. Note that the MTS AI Chat also shows very low results on CheGeKa, which might also indicate the lack of culture-specific knowledge.

The proprietary model GigaChat substantially outperforms Vikhr on CheGeKa and notably on Russian SuperGLUE. We assume this is due to the use of much larger Russian datasets for pre-training. However, surprisingly, it falls behind Vikhr on RuMMLU. On all benchmarks, Vikhr outperforms the the proprietary competitor from MTS AI.

## 5 Conclusion

We have presented Vikhr - a new state-of-the-art open-source instruction-following LLM oriented on the Russian language. To create Vikhr, we developed a comprehensive pipeline for adapting English-oriented LLMs to Russian. The pipeline includes the adaptation of the tokenizer vocabulary, continued pre-training of the entire model, and instruction tuning. We have also constructed a new dataset for instruction tuning by expanding the Saiga dataset with automatically translated and cleaned English instruction datasets. Our extensive work enabled Vikhr to outperform the known baselines, while maintaining computational efficiency.

We hope that the published models will foster the research on LLMs and enhance the diversity of languages incorporated into research agendas.

## Limitations

We do not introduce additional restrictions to the usage of our models. However, the users must comply with the license of the base model and instruction datasets.

We do not implement RLHF / DPO fine-tuning of Vikhr due to the lack of the resources for human annotation. We expect further performance improvements from these techniques.

We do not introduce additional instructionoutput pairs to facilitate LLM alignment. However, we note that the majority of the data for supervised fine-tuning of Vikhr are obtained from the ChatGPT model series, so our model partially inherits its alignment.

## References

AI Forever. 2022. ru-gpts: Generative pre-trained transformer models for russian. https://github.com/ ai-forever/ru-gpts.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with $90 \% *$ chatgpt quality.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177.

Alena Fenogenova, Artem Chervyakov, Nikita Martynov, Anastasia Kozlova, Maria Tikhonova, Albina Akhmetgareeva, Anton Emelyanov, Denis Shevelev, Pavel Lebedev, Leonid Sinev, et al. 2024. Mera: A comprehensive llm evaluation in russian. arXiv preprint arXiv:2401.04531.

Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. 2024. Arcee's mergekit: A toolkit for merging large language models. arXiv preprint arXiv:2403.13257.

Ilya Gusev. 2023. rulm: A toolkit for training neural language models. https://github.com/IlyaGusev/ rulm.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR).

John Hewitt. 2021. Initializing new word embeddings for pretrained language models.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Aditi Jha, Sam Havens, Jeremy Dohmann, Alex Trott, and Jacob Portes. 2023. Limit: Less is more for instruction tuning across evaluation paradigms. arXiv preprint arXiv:2311.13133.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66-71.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023. Bactrian-x: A multilingual replicable instruction-following model with lowrank adaptation. arXiv preprint arXiv:2305.15011.

Muhammad Maaz, Hanoona Rasheed, Abdelrahman Shaker, Salman Khan, Hisham Cholakal, Rao M Anwer, Tim Baldwin, Michael Felsberg, and Fahad S Khan. 2024. Palo: A polyglot large multimodal model for $5 \mathrm{~b}$ people. arXiv preprint arXiv:2402.14818.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, et al. 2023. Crosslingual generalization through multitask finetuning. In The 61st Annual Meeting Of The Association For Computational Linguistics.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744.

Aleksandar Petrov, Emanuele La Malfa, Philip Torr, and Adel Bibi. 2024. Language model tokenizers introduce unfairness between languages. Advances in Neural Information Processing Systems, 36.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, et al. 2023. Jais and jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models. arXiv preprint arXiv:2308.16149.

Tatiana Shavrina, Alena Fenogenova, Emelyanov Anton, Denis Shevelev, Ekaterina Artemova, Valentin Malykh, Vladislav Mikhailov, Maria Tikhonova, Andrey Chertok, and Andrey Evlampiev. 2020. Russiansuperglue: A russian language understanding evaluation benchmark. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4717-4726.

Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Vladislav Mikhailov, Anastasia Kozlova, and Tatiana Shavrina. 2022. mgpt: Few-shot learners go multilingual. arXiv preprint arXiv:2204.07580.

Rohan Taori, Ishaan Shum, Pieter Abbeel, Carlos Guestrin, and Percy Liang. 2023. Stanford alpaca: An instruction-following language model. GitHub.

Teknium. 2023. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants.

Mikhail Tikhomirov and Daniil Chernyshev. 2023. Impact of tokenization on llama russian adaptation. arXiv preprint arXiv:2312.02598.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32 .

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. arXiv preprint arXiv:2401.01055.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. 2023. Starling-7b: Improving $11 \mathrm{~m}$ helpfulness \& harmlessness with rlaif.

Dmitry Zmitrovich, Alexander Abramov, Andrey Kalmykov, Maria Tikhonova, Ekaterina Taktasheva, Danil Astafurov, Mark Baushenko, Artem Snegirev, Tatiana Shavrina, Sergey Markov, et al. 2023. A family of pretrained transformer language models for russian. arXiv preprint arXiv:2309.10931 .

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, WeiYin Ko, Daniel D'souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827.


[^0]:    ${ }^{1}$ https://huggingface.co/Vikhrmodels

[^1]:    ${ }^{2}$ https://huggingface.co/docs/optimum/ bettertransformer/tutorials/convert

[^2]:    ${ }^{3}$ https://huggingface.co/datasets/Vikhrmodels/ Veles-2.5

    ${ }^{4}$ https://huggingface.co/datasets/ berkeley-nest/Nectar

[^3]:    ${ }^{5}$ https://github.com/NLP-Core-Team/mmlu_ru

[^4]:    ${ }^{6}$ https://huggingface.co/CohereForAI/aya-101

    ${ }^{7}$ https://huggingface.co/collections/IlyaGusev

    ${ }^{8}$ https://huggingface.co/MTSAIR/multi_verse_ model

    ${ }^{9}$ https://mera.a-ai.ru/ru/submits/10257

    ${ }^{10}$ https://mera.a-ai.ru/ru/submits/10290

</end of paper 1>


<paper 2>
# Evaluating the Performance of Large Language Models on GAOKAO Benchmark 

Xiaotian Zhang ${ }^{1, *}$, Chunyang Li $^{1, *}$, Yi Zong ${ }^{1, *}$, Zhengyu Ying ${ }^{2}$, Liang $\mathbf{H e}^{\dagger}$, Xipeng Qiu ${ }^{\dagger}$<br>Tianxiang Sun, Peng Li, Shiqiao Meng, Yanjun Zheng, Jun Zhan,<br>Zhangyue Yin, Xiannian Hu, Guofeng Quan<br>${ }^{1}$ School of Computer Science, Fudan University<br>${ }^{2}$ School of Computer Science and Technology, East China Normal University<br>\{xiaotianzhang21, yzong22\} @m.fudan.edu.cn, $\{19307110196$, xpqiu $\}$ fudan.edu.cn<br>\{zyying, lhe \} @cs.ecnu.edu.cn


#### Abstract

Large Language Models(LLMs) have demonstrated remarkable performance across various natural language processing tasks; however, how to comprehensively and accurately assess their performance becomes an urgent issue to be addressed. This paper introduces GAOKAOBench, an intuitive benchmark that employs questions from the Chinese GAOKAO examination as test samples, including both subjective and objective questions. To align with human examination methods, we design a method based on zero-shot settings to evaluate the performance of LLMs. With human evaluation, we obtain the converted total score of LLMs, including GPT-4, ChatGPT and ERNIE-Bot. Our findings reveal that LLMs have achieved competitive scores in Chinese GAOKAO examination, while they exhibit significant performance disparities across various subjects. We also use LLMs to grade the subjective questions, and find that model scores achieve a moderate level of consistency with human scores. In conclusion, this research contributes a robust evaluation benchmark for future large language models and offers valuable insights into the advantages and limitations of such models. ${ }^{1}$


## 1 Introduction

LLMs have demonstrated great abilities in handling diverse applications. The LLMs (Brown et al., 2020; Ouyang et al., 2022, OpenAI, 2023; Bubeck et al., 2023; Wei et al., 2022) indicate they possess abundant intrinsic knowledge, the ability to follow instructions and reasoning capabilities, which in certain areas are on par with or even surpass human abilities. To better measure the capabilities of LLMs, researchers have proposed more comprehensive and challenging benchmarks.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_f81b7c51bf4c548ca614g-01.jpg?height=642&width=877&top_left_y=821&top_left_x=978)

Figure 1: Converted Total Score of LLMs in GAOKAO-Bench. The converted total score for subjects in both the sciences and the humanities is 750 points.

MMLU (Hendrycks et al., 2021) aims to measure a text model's multitask accuracy, covering 57 tasks such as elementary mathematics, US history, computer science, and more. BIG-Bench (Srivastava et al., 2022) introduces a comprehensive evaluation framework encompassing more than 204 subtasks, including linguistics, child development, among others. AGIEval (Zhong et al., 2023) evaluates the performance of LLMs in the context of humancentric standardized examinations and contains both Chinese and English tasks. Huang et al. (2023) propose C-Eval, a comprehensive Chinese evaluation suite covering four difficulty levels. However, the benchmark mentioned above only consists of objective questions and lacks subjective questions that are more closely related to generative abilities. Besides, due to the absence of real-world test samples, individuals often underestimate the complexity of these tasks and the abilities of the models, particularly in the context of the rapid development

![](https://cdn.mathpix.com/cropped/2024_06_04_f81b7c51bf4c548ca614g-02.jpg?height=600&width=1464&top_left_y=240&top_left_x=296)

Figure 2: Scoring Rate of LLMs on objective and subjective questions across the subjects.

of LLMs. Consequently, there is a need for an intuitive and practical evaluation method.

We propose using the Chinese College Entrance Examination (GAOKAO) questions. These questions include computational, reasoning, knowledge assessment and writing tasks (Tan et al., 2021). Previous benchmarks based on the GAOKAO mainly focus on English (Yuan and Liu, 2022), especially English Reading and Comprehension Questions (Zhang et al., 2022). To this end, we introduce the GAOKAO-Benchmark (GAOKAO-Bench), a benchmark specifically tailored to LLMs evaluation that covers the GAOKAO questions from 2010 to 2022. The GAOKAO-Bench consists of 9 subjects with 1781 objective questions and 1030 subjective questions. The question types include singlechoice, cloze, correction, open-ended questions, and more.

We conduct experiments on some currently bestperforming LLMs. To more accurately measure their generative capabilities, we use human scoring evaluation to judge subjective questions. The results in Figure1 show that LLMs have achieved competitive scores in the GAOKAO. Meanwhile, we find that all of the LLMs exhibit obvious signs of subject bias, which informs the future development of LLMs.

Due to the high cost of human evaluation, we provide human-annotated marking criteria of subjective questions. And we use LLM as a judge to evaluate LLMs on subjective questions. The results indicate that equipped with the detailed marking criteria, LLMs exhibit high consistency with human teachers, making the large-scale assessment of subjective questions feasible.

## 2 GAOKAO-Bench

### 2.1 Introduction to the GAOKAO

The Chinese College Entrance Examination, also known as the GAOKAO, is a nationwide examination designed to assess the academic abilities of high school students applying to universities in China. Known as a rigorous and comprehensive examination, the GAOKAO is differentiated into two distinct streams: the sciences and the humanities: the sciences include Chinese, sciences mathematics, English, physics, chemistry and biology; the humanities include Chinese, humanities mathematics, English, politics, history and geography. The examination encompasses a variety of question types that include logical reasoning, computational analysis, knowledge-based quizzes and written expression among other aspects.

### 2.2 Dataset Description

The GAOKAO-Bench established in this paper includes the content of all national exams in the GAOKAO of all subjects from 2010 to 2022, providing an intuitive and human-aligned evaluation benchmark for LLMs.

We obtain the questions and transform them from PDF into JSON file format using a combination of automated scripting and manual annotation. Mathematical formulas within the questions were converted into $\mathrm{IAT}_{\mathrm{E}} \mathrm{X}$ format. Appendix A. 1 provides an example of a mathematical single-choice question.

The questions are divided into subjective and objective categories, depending on whether they re-

| Models | Overall | Chinese | Eng. | Sci. <br> Math | Hum. <br> Math | Phys. | Chem. | Biol. | Poli. | Hist. | Geog. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaMA-7b | $21.1 \%$ | $16.2 \%$ | $20.5 \%$ | $24.3 \%$ | $26.1 \%$ | $0.0 \%$ | $22.6 \%$ | $22.7 \%$ | $22.2 \%$ | $19.2 \%$ | $24.2 \%$ |
| Vicuna-7b | $21.0 \%$ | $12.0 \%$ | $19.6 \%$ | $23.8 \%$ | $23.4 \%$ | $7.0 \%$ | $27.4 \%$ | $20.0 \%$ | $20.9 \%$ | $23.0 \%$ | $23.2 \%$ |
| Baichuan2-7b-Base | $27.2 \%$ | $16.2 \%$ | $21.2 \%$ | $24.8 \%$ | $24.8 \%$ | $0.0 \%$ | $23.4 \%$ | $24.0 \%$ | $55.3 \%$ | $32.1 \%$ | $24.2 \%$ |
| Baichuan2-7b-Chat | $40.5 \%$ | $31.7 \%$ | $33.0 \%$ | $26.6 \%$ | $28.4 \%$ | $18.0 \%$ | $26.6 \%$ | $48.0 \%$ | $69.7 \%$ | $57.8 \%$ | $49.5 \%$ |
| Baichuan2-13b-Chat | $43.9 \%$ | $26.9 \%$ | $34.7 \%$ | $23.8 \%$ | $31.7 \%$ | $25.0 \%$ | $40.3 \%$ | $53.3 \%$ | $75.3 \%$ | $59.9 \%$ | $61.1 \%$ |
| ChatGLM-6b | $30.8 \%$ | $18.6 \%$ | $17.0 \%$ | $25.2 \%$ | $25.7 \%$ | $12.5 \%$ | $30.6 \%$ | $24.7 \%$ | $54.1 \%$ | $59.9 \%$ | $25.3 \%$ |
| ChatGLM2-6b | $42.7 \%$ | $31.1 \%$ | $30.6 \%$ | $29.0 \%$ | $35.8 \%$ | $24.2 \%$ | $46.0 \%$ | $71.3 \%$ | $55.0 \%$ | $59.2 \%$ | $41.1 \%$ |
| GPT-4-0613 | $71.6 \%$ | $52.1 \%$ | $\mathbf{9 3 . 2} \%$ | $\mathbf{5 4 . 5} \%$ | $\mathbf{6 4 . 0} \%$ | $50.8 \%$ | $43.6 \%$ | $\mathbf{8 3 . 0} \%$ | $72.5 \%$ | $74.2 \%$ | $\mathbf{8 1 . 1 \%}$ |
| GPT-4-0314 | $\mathbf{7 2 . 2 \%}$ | $\mathbf{5 3 . 9 \%}$ | $93.1 \%$ | $53.7 \%$ | $63.3 \%$ | $\mathbf{5 5 . 5} \%$ | $44.4 \%$ | $80.7 \%$ | $75.9 \%$ | $75.6 \%$ | $80.0 \%$ |
| GPT-3.5-turbo-0301 | $53.2 \%$ | $34.7 \%$ | $76.6 \%$ | $38.8 \%$ | $47.8 \%$ | $41.1 \%$ | $38.7 \%$ | $56.9 \%$ | $45.3 \%$ | $53.9 \%$ | $54.0 \%$ |
| ERNIE-Bot-0615 | $56.6 \%$ | $46.7 \%$ | $31.0 \%$ | $38.3 \%$ | $49.1 \%$ | $35.9 \%$ | $\mathbf{6 6 . 1} \%$ | $79.3 \%$ | $\mathbf{8 6 . 9 \%}$ | $\mathbf{7 9 . 1 \%}$ | $68.4 \%$ |
| ERNIE-Bot-turbo-0725 | $45.6 \%$ | $35.3 \%$ | $26.6 \%$ | $34.1 \%$ | $36.2 \%$ | $32.0 \%$ | $51.6 \%$ | $64.0 \%$ | $72.2 \%$ | $63.4 \%$ | $44.2 \%$ |

Table 1: Scoring Rate of Objective Questions. Models above the line are open-source LLMs; models below the line are closed-source LLMs.

quire human scoring. In total, we select 2811 questions, including 1030 objective questions and 1781 objective questions. Table 3 provides a breakdown of the specific types of questions and the corresponding number of questions in each type. MultiQuestion Choice refers to a format where a single question is followed by multiple sub-questions and Multi-Choice refers to a format where a single question corresponds to multiple correct answers.

## 3 Experiments

### 3.1 Methodology

Prompt Design In order to emulate the format in which humans partake in examinations, we utilize a zero-shot settings strategy (Ouyang et al., 2022) and create prompts tailored to different question types. The prompts not only require the model to complete the task, but also explicitly specify the format of the output as we contend that the intrinsic knowledge level of the model and its ability to follow instructions are equally important. The specific prompt examples we use are illustrated in Appendix A.1.

Models We evaluate several current bestperforming LLMs that support both Chinese and English:

1. GPT-4: We test on 2 checkpoints: GPT-40613 and GPT-4-0314.
2. ChatGPT: We test on GPT-3.5-turbo-0301 checkpoint.
3. ERNIE-Bot: A Chinese LLM published by Baidu. We test on ERNIE-Bot-0615 checkpoint.
4. ERNIE-Bot-turbo: We test on ERNIE-Botturbo-0725 checkpoint.

We set the sampling temperature to 0.3 in order to achieve a balance between stability and diversity.

Metric When evaluating objective and subjective questions separately, we use the scoring rate $R_{i, \text { obj }}$ and $R_{i, \text { subj }}$ for each subject $i$.

To evaluate the overall performance, we convert the scoring rates of subjective and objective questions into a total score $S_{\text {total }}$. We mimic the subjective question scores as $M_{i \text {,subj }}$ and objective question scores $M_{i, \text { obj }}$ for each subject $i$ in the GAOKAO. The converted total score can be formulated as:

$$
\begin{aligned}
S_{\text {total, } \mathcal{S}} & =\sum_{i \in \mathcal{S}}\left(R_{i, \mathrm{obj}} \cdot M_{i, \mathrm{obj}}+R_{i, \mathrm{subj}} \cdot M_{i, \mathrm{subj}}\right) \\
S_{\text {total, } \mathcal{H}} & =\sum_{i \in \mathcal{H}}\left(R_{i, \mathrm{obj}} \cdot M_{i, \mathrm{obj}}+R_{i, \mathrm{subj}} \cdot M_{i, \mathrm{subj}}\right)
\end{aligned}
$$

| Models | Overall | Chinese | Eng. | Sci. <br> Math | Hum. <br> Math | Phys. | Chem. | Biol. | Poli. | Hist. | Geog. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4-0613 | $50.8 \%$ | $50.3 \%$ | $87.6 \%$ | $\mathbf{2 4 . 6} \%$ | $27.5 \%$ | $47.1 \%$ | $28.5 \%$ | $\mathbf{8 5 . 6 \%}$ | $49.9 \%$ | $59.9 \%$ | $71.5 \%$ |
| GPT-4-0314 | $\mathbf{5 1 . 9 \%}$ | $51.5 \%$ | $\mathbf{8 8 . 3} \%$ | $24.1 \%$ | $\mathbf{2 7 . 9} \%$ | $\mathbf{5 6 . 7 \%}$ | $\mathbf{3 5 . 0 \%}$ | $\mathbf{8 5 . 6} \%$ | $50.0 \%$ | $\mathbf{6 3 . 1 \%}$ | $70.0 \%$ |
| GPT-3.5-turbo-0301 | $35.8 \%$ | $33.9 \%$ | $75.4 \%$ | $15.2 \%$ | $15.9 \%$ | $16.9 \%$ | $21.4 \%$ | $36.3 \%$ | $42.3 \%$ | $58.4 \%$ | $62.1 \%$ |
| ERNIE-Bot-0615 | $48.4 \%$ | $\mathbf{5 7 . 1 \%}$ | $45.0 \%$ | $17.0 \%$ | $25.6 \%$ | $33.5 \%$ | $30.8 \%$ | $84.9 \%$ | $\mathbf{5 3 . 0 \%}$ | $60.0 \%$ | $\mathbf{7 2 . 7 \%}$ |
| ERNIE-Bot-turbo-0725 | $39.2 \%$ | $42.5 \%$ | $28.8 \%$ | $14.6 \%$ | $15.6 \%$ | $23.2 \%$ | $25.0 \%$ | $85.1 \%$ | $45.3 \%$ | $47.0 \%$ | $61.8 \%$ |

Table 2: Scoring Rate of Subjective Questions. The results are scored by human teachers.

| Question Type |  | Number | Percentage |
| :--- | :--- | :---: | :---: |
|  | Single Choice | 1418 | $50.5 \%$ |
| Objective | Multi-Question Choice | 273 | $9.7 \%$ |
|  | Multi-Choice | 64 | $2.3 \%$ |
|  | Five out of Seven | 26 | $0.9 \%$ |
| Subjective | Open-ended Question | 786 | $28.0 \%$ |
|  | Cloze | 218 | $7.8 \%$ |
|  | Correction | 26 | $0.9 \%$ |

Table 3: Distribution of Question Types.

where $\mathcal{S}$ stands for the set of the sciences subjects, and $\mathcal{H}$ stands for the set of the humanities subjects. The total scores of sciences and humanities are both 750 points. Detailed total score for each subject is shown in Appendix B.

### 3.2 Objective Questions

Each item $i$ in the GAOKAO-Bench comprises the question $q_{i}$, the standard answer $a_{i}$, the score $s_{i}$, the analysis $n_{i}$. For objective questions, the input includes the question $q_{i}$ and the LLMs need to output $\left(r_{i}, o_{i}\right)$, where $r_{i}$ denotes the corresponding reasoning process and $o_{i}$ denotes the outcome. Points are awarded only if the outcome $o_{i}$ is consistent with the standard answer $a_{i}$. Following the technical report for OpenAI's GPT-4 (OpenAI, 2023), we score the objective questions using regular matching. In addition to the LLMs mentioned above, we evaluate several open-source LLMs on GAOKAOBench, including LLaMA (Touvron et al., 2023), Baichuan (Yang et al., 2023)and ChatGLM (Zeng et al., 2023).

### 3.3 Subjective Questions

The input and output formats of the subjective questions are similar to those of objective questions. During the grading process, evaluators take into account both the reasoning process $r_{i}$ and the outcome $o_{i}$. We assess the subjective questions using human scoring, in order to more precisely evaluate the performance of LLMs. Each subjective ques- tion is evaluated by two teachers, and the average of these scores was adopted as the final score for that question.

### 3.4 LLM as a Judge

Due to the high cost of manual evaluation, it is a natural progression to consider the use of LLMs for grading subjective questions. To better align with the teachers, we solicit teachers to provide detailed marking criteria $m_{i}$, breaking down the answers into specific scoring points for each item $i$. We design prompts in zero-shot settings and utilize GPT-4-turbo (GPT-4-1106-preview) as a judge. For each input $\left(q_{i}, a_{i}, s_{i}, n_{i}, m_{i}, r_{i}, o_{i}\right)$, the LLM need to output $\left(g_{i}, f_{i}\right)$, where $g_{i}$ denotes the process of grading and $f_{i}$ denotes the final score. The sampling temperature is set to 0 to obtain deterministic scores. We calculate the converted total score and Spearman and Kendall-Tau correlations between predicted scores and human scores following Jain et al. (2023) and Zhong et al. (2022).

### 3.5 Results

Overall Performance Figure 1 shows the converted total score of LLMs on GAOKAO-Bench. GPT-4 achieves scores exceeding 400 points and ERNIE-Bot surpasses ChatGPT. Every LLM obtains higher scores in humanities than in sciences. In the GAOKAO, the sciences require more advanced logical reasoning and computational steps than the humanities; and the humanities require a greater amount of knowledge than the sciences. The result indicates the reasoning and calculation abilities of LLMs still need further improvement.

Performance on Objective Questions Table 1 reflects the performance of LLMs on objective questions in different subjects. Open-source models pre-trained on Chinese language data and
aligned with human perform better in all subjects. And the performance of the models improves with the increase in their scale. For closed-source LLMs, GPT-4 maintains a lead in the majority of subjects, but ERNIE-Bot performs better in chemistry, politics and history.

Performance on Subjective Questions Table 2 indicates the human evaluation of subjective questions. GPT-4 obtains the highest scoring rate ( $51.9 \%$ ) and ERNIE-Bot achieves a comparably close level (48.4\%). GPT-4 and ChatGPT exhibit superior performance in English compared to Chinese, whereas ERNIE-Bot and ERNIE-Bot-turbo demonstrate the opposite trend, excelling more in Chinese than in English.

LLM as a Judge Table 4 shows the results of using GPT-4-turbo to grade subjective questions. The Question-level Spearman and Kendall-Tau correlations show a markedly strong positive correlation between model judging and human scoring.

## 4 Analysis

### 4.1 Difference in Subjects

We analyze the scoring rate of subjective questions and objective questions in different subjects of LLMs, and find that there are large differences in the ability of the model in different subjects both in objective questions and subjective questions.

GPT-4 excels in English, biology and geography with scoring rates greater than $70 \%$ both in subjective and objective questions. However, they demonstrate poor performance in mathematics and physics with scoring rates less than $40 \%$. ERNIEBot performs better in biology, history, politics in subjective questions with scoring rates greater than $60 \%$, but the scoring rate of mathematics is less than $30 \%$.

We posit that the substantial disparities across subjects can be attributed to two primary factors: firstly, the distinct competencies evaluated by each subject, for instance, language comprehension and summarization abilities in Chinese and English, and logical reasoning and computational skills in mathematics and physics; secondly, aspects related to the training of the model, including the richness of the pre-training corpus and the inclinations towards human alignment.

### 4.2 Difference between Sujective and Objective Questions

For a given subject, the scoring rate of subjective questions is generally lower than that of objective questions. For example, the scoring rate of subjective mathematics questions of GPT-4 is significantly lower than that on subjective mathematics questions. We hypothesize that subjective mathematics questions distinctly require the application of correct formulas, as well as more extensive computational and reasoning steps, which poses a significant challenge for LLMs. And compared to objective questions, the subjective questions of humanities necessitate students' mastery of more precise knowledge points, as well as their abilities in induction, summarization and categorical organization.

![](https://cdn.mathpix.com/cropped/2024_06_04_f81b7c51bf4c548ca614g-05.jpg?height=825&width=711&top_left_y=1181&top_left_x=1095)

Figure 3: The Annual Trends of LLMs on GAOKAOBench.

### 4.3 Stable Annual Trends on the GAOKAO

We categorize the examination questions based on their respective years and compute the model's converted total scores from 2013 to 2022 in Figure 3. We observe that the converted total score of LLMs are stable across the last decade. It indicates a relative stability in the difficulty level of the GAOKAO questions.

| Models | Sciences |  | Humanities |  | $\rho$ | $\tau$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Human | GPT-4-turbo | Human | GPT-4-turbo |  |  |
| GPT-4-0314 | 434 | 428 | 480 | 523 | 0.854 | 0.710 |
| GPT-3.5-turbo-0301 | 447 | 440 | 485 | 535 | 0.852 | 0.709 |
| ERNIE-Bot-0615 | 300 | 314 | 368 | 408 | 0.845 | 0.710 |
| ERNIE-Bot-turbo-0725 | 333 | 348 | 397 | 436 | 0.864 | 0.740 |

Table 4: Converted total score, Question-level Spearman and Kendall-Tau correlations of Human and GPT-4-turbo grading methods.

### 4.4 LLM as a Judge

We observe that the converted total score of sciences is much closer to human scoring than that of humanities. The deviation in scores for science subjects is less than $2 \%$ of the total score, and for humanities subjects, the deviation is around $5 \%$ of the total score. We posit that the answers and scoring criteria in the sciences are relatively explicit, whereas in the humanities, scoring depends on the alignment of semantics with designated points. This necessitates a fine-grained semantic understanding by the models, presenting a significant challenge for LLMs.

## 5 Avoid Benchmark Leakage

Benchmark leakage means the data related to evaluation sets is occasionally used for model training (Zhou et al., 2023). And it is plausible that the GAOKAO questions may be included in the training corpus of LLMs. The zero-shot settings and human evaluation used in this paper can alleviate the unfair phenomenon. Given that the GAOKAO is conducted annually in June, we plan to incorporate each year's new GAOKAO questions into the GAOKAO-Bench as a supplement, aiming to mitigate the issue of dataset leakage in evaluations. We have released the GAOKAO-Bench$2023^{2}$ which includes the objective questions in the 2023 GAOKAO. And we compare the scoring rate of objective questions in GAOKAO-Bench and GAOKAO-Bench-2023 in Table 5. We contend that these variations are within the normal range of difficulty fluctuations.

## 6 Ablation Study

We investigate the impact of manually annotated marking criteria on the accuracy of the LLM's grad-[^1]

| Models | GAOKAO-Bench | GAOKAO-Bench-2023 | $\Delta$ |
| :--- | :---: | :---: | :---: |
| ChatGLM-6b | $30.8 \%$ | $24.1 \%$ | $-6.7 \%$ |
| ChatGLM2-6b | $42.7 \%$ | $36.9 \%$ | $-5.8 \%$ |
| Baichuan2-7b-chat | $40.5 \%$ | $37.9 \%$ | $-2.6 \%$ |
| Baichuan2-13b-chat | $43.9 \%$ | $41.3 \%$ | $-2.6 \%$ |
| GPT-4-0613 | $71.6 \%$ | $71.0 \%$ | $-0.6 \%$ |
| GPT-4-0314 | $72.2 \%$ | $69.8 \%$ | $-2.4 \%$ |

Table 5: Scoring Rate of Objective Questions on GAOKAO-Bench-2023. The GAOKAO-Bench covers questions from 2010 to 2022.

ing of subjective questions. We use the GPT-4turbo to evaluate the performance of GPT-4, ChatGPT and ERNIE-Bot-turbo with or without marking criteria. Tabel 6 indicates that provided with marking criteria, LLMs can better align with human preferences.

| Methods |  | GPT-4-0613 | GPT-3.5-turbo-0301 | ERNIE-Bot-turbo-0725 |
| :--- | :--- | :---: | :---: | :---: |
| w marking criterion | $\rho$ | 0.854 | 0.845 | 0.825 |
|  | $\tau$ | 0.710 | 0.710 | 0.685 |
| w/o marking criterion | $\rho$ | 0.820 | 0.820 | 0.803 |
|  | $\tau$ | 0.659 | 0.674 | 0.654 |

Table 6: Spearman and Kendall-Tau Correlations of LLM grading and human judgement.

## 7 Related Work

Benchmark for LLMs The flourishing development of LLMs has also raised higher demands for benchmarks. Benchmarks for traditional tasks in NLP, such as GLUE (Wang et al., 2018) for natural language understanding, SQuAD (Rajpurkar et al., 2016) for reading comprehension, cannot measure the comprehensive capabilities of LLMs. Consequently, researchers have proposed new benchmarks to evaluate the advanced abilities of LLMs. MMLU (Hendrycks et al., 2021) provides a multi-task test across a diverse set of subjects. BIG-Bench (Srivastava et al., 2022) covers a diverse range of topics and languages, including auto debugging, know unknowns, logical deduction. HELM (Liang et al., 2023)
taxonomies the design space of language model evaluation into scenarios and metrics. In the field of Chinese language benchmarks, C-Eval (Huang et al., 2023) selects multiple-choice questions across four difficulty levels: middle school, high school, college, and professional. AGIEval (Zhong et al., 2023) assesses LLMs in the context of human-centric standardized exams. CMMLU (Li et al., 2023) includes subjects that may not typically appear in standard exams but are relevant to people's daily life, such as Chinese food culture, Chinese driving rule.

Human evaluation for LLMs Compared to automatic evaluation, human evaluation is more aligned with real-world application scenarios and can offer more comprehensive and precise feedback (Chang et al., 2023). Chatbot Arena (Zheng et al., 2023) provides a platform to assess and compare diverse chatbot models through user engagement and voting. Ziems et al. (2023) adopts human scoring evaluation on generation tasks. Liang et al. (2023) conduct human evaluations on 6 LLMs on summarization and disinformation scenarios.

## 8 Limitations

While we evaluate and analyze the performance of LLMs on GAOKAO-Bench, there are some limitations in this work. Firstly, due to the constraints in time and resources, this paper does not delve into the detailed analysis of the errors made by LLMs on the GAOKAO-Bench, such as model hallucinations and reasoning mistakes. Secondly, due to the rapid developments of LLMs and high cost of human evaluation, we are unable to conduct experiments on every model using human scoring. We hope to enhance the evaluation and analysis of the models' reasoning process and utilize LLMs as a replacement for human scoring in future work.

## 9 Conclusion

In this paper, we introduce the GAOKAO-Bench dataset, which serves as an evaluation standard for large language models. The dataset includes Chinese College Entrance Examination questions from 2010 to 2022, covering various subjects and question types, with an overall high level of difficulty. By testing large language models on the GAOKAO-
Bench, we can analyze the gap and advantages of these models compared to humans in a reasonable and intuitive manner.

In addition, we evaluate the ability of large language models to answer Chinese College Entrance Examination questions using zero-shot prediction approach and human evaluation. Our results show that the models perform well on knowledge-based questions, but struggle with certain types of logical reasoning and mathematical problems, as well as with reading comprehension of longer texts in Chinese.

We also use the LLMs to evaluate subjective questions, which is called LLM-as-a-Judge. We observe that equipped with human-annotated marking criteria, the LLM evaluation is consistent to human preference.

These findings suggest that large language models have potential applications in education and language assessment, but there is still room for improvement in certain areas. Future work could focus on developing approaches to enhance the model's performance on longer text reading comprehension tasks, logical reasoning and calculation problems.

## References

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A survey on evaluation of large language models.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu,

Maosong Sun, and Junxian He. 2023. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models.

Sameer Jain, Vaishakh Keshava, Swarnashree Mysore Sathyendra, Patrick Fernandes, Pengfei Liu, Graham Neubig, and Chunting Zhou. 2023. Multidimensional evaluation of text summarization with in-context learning.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023. Cmmlu: Measuring massive multitask language understanding in chinese.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic evaluation of language models.

OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.

Hongye Tan, Xiaoyue Wang, Yu Ji, Ru Li, Xiaoli Li, Zhiwei Hu, Yunxiao Zhao, and Xiaoqi Han. 2021. Gcrc: A new challenging mrc dataset from gaokao chinese for explainable evaluation. In Findings of the Association for Computational Linguistics: ACLIJCNLP 2021, pages 1319-1330.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355, Brussels, Belgium. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, JunTao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. 2023. Baichuan 2: Open large-scale language models.

Weizhe Yuan and Pengfei Liu. 2022. restructured pretraining. arXiv preprint arXiv:2206.11147.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. Glm-130b: An open bilingual pre-trained model.

Cheng Zhang, Hao Zhang, and Jie Wang. 2022. Downstream transformer generation of questionanswer pairs with preprocessing and postprocessing pipelines.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena.

Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022. Towards a unified multidimensional evaluator for text generation.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models.

Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong

Wen, and Jiawei Han. 2023. Don't make your 1lm an evaluation benchmark cheater.

Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. 2023. Can large language models transform computational social science?
</end of paper 2>


