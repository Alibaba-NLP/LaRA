<paper 0>
# Adapting Large Language Models for Document-Level Machine Translation 

Minghao $\mathbf{W u}{ }^{\ominus}$ Thuy-Trang $\mathbf{V u}^{\ominus}$ Lizhen $\mathbf{Q u}^{\triangleright}$ George Foster ${ }^{\oplus}$ Gholamreza Haffari ${ }^{\ominus}$<br>${ }^{\ominus}$ Monash University ${ }^{\top}$ Google Research<br>\{firstname.lastname\}@monash.edu fosterg@google.com


#### Abstract

Large language models (LLMs) have made significant strides in various natural language processing (NLP) tasks. Recent research shows that the moderately-sized LLMs often outperform their larger counterparts after task-specific fine-tuning. In this work, we delve into the process of adapting LLMs to specialize in document-level machine translation (DocMT) for a specific language pair. Firstly, we explore how prompt strategies affect downstream translation performance. Then, we conduct extensive experiments with two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Our findings indicate that in some cases, these specialized models even surpass GPT-4 in translation performance, while they still significantly suffer from the off-target translation issue in others, even if they are exclusively fine-tuned on bilingual parallel documents. Furthermore, we provide an in-depth analysis of these LLMs tailored for DocMT, exploring aspects such as translation errors, discourse phenomena, training strategy, the scaling law of parallel documents, additional evaluation on recent test sets, and zero-shot crosslingual transfer. Our findings not only shed light on the strengths and limitations of LLM-based DocMT models but also provide a foundation for future research.


## 1 Introduction

Large language models (LLMs) demonstrate impressive proficiency in a wide range of applications (Ouyang et al., 2022; Wei et al., 2022a; Sanh et al., 2022; Chung et al., 2022; OpenAI, 2023; Anil et al., 2023; Touvron et al., 2023a,b; Jiang et al., 2023). However, in the realm of translation tasks, only few very large models, such as GPT-3.5-TURBO and GPT-4-TURBO, can match or surpass the performance of state-of-the-art supervised encoderdecoder models like NLLB (Costa-jussà et al., 2022), while they still under-perform in translating low-resource languages (Robinson et al., 2023;
Jiao et al., 2023; Hendy et al., 2023). Consequently, a number of recent works attempt to bridge the gap between LLMs and supervised encoder-decoder models in translation tasks (Zhu et al., 2023; Yang et al., 2023; Zhang et al., 2023; Moslem et al., 2023; Xu et al., 2023; Kudugunta et al., 2023). Recently, research suggests that smaller, specialized models can outperform larger, general-purpose models in specific tasks (Gunasekar et al., 2023; Luo et al., 2023; Azerbayev et al., 2023). Therefore, we explore adapting LLMs for document-level machine translation (DocMT) in this study.

In this study, we analyze moderately-sized LLMs (with $7 B$ parameters) across 18 translation tasks involving nine language pairs. We fine-tune three LLMs using Parameter-Efficient Fine-Tuning (PEFT) and Fully Fine-Tuning (FFT). Comparisons with state-of-the-art translation models, using metrics like $s \mathrm{BLEU}, d \mathrm{BLEU}$, and COMET, confirm the superior translation capabilities of LLMs after fine-tuning. However, we identify a significant issue of off-target translations, observed even after exclusive fine-tuning on bilingual corpora. Additionally, we present an in-depth analysis of our LLM-based DocNMT models from various perspectives: translation error distribution, discourse phenomena, training strategy, the scaling law of parallel documents, additional evaluations on WMT2023 test sets, and zero-shot cross-lingual transfer, aiming to enhance understanding and efficacy of LLMs in DocMT tasks.

We present extensive empirical evidence that highlights both the superior translation capabilities and limitations of the LLM-based DocMT models in this study, making several significant discoveries. Here are the main takeaways:

- Selective Excellence in Translation Tasks: Our findings show that our moderately-sized LLMs outperform GPT-4-TURBO in certain translation tasks, but struggle in others due to the off-target translation issue. Despite this,
our DocMT models exhibit better context awareness and fewer errors, while maintaining comparable performance.
- Fine-Tuning Strategies: Our research indicates that the PEFT approach outperforms the FFT approach overall. However, the FFT approach shows greater data efficiency, needing only about $1 \%$ of the total dataset to reach the performance level of models trained on the entire dataset. In contrast, the PEFT approach requires $10 \%$ of the total dataset for comparable results.
- Evaluation on Recent Test Sets: We evaluate our models on recent test sets between English and German from WMT2023 (Koehn et al., 2023). Our empirical results show that, when the data leakage risks are mitigated, the LLM-based DocMT models generalize better on out-of-domain text, compared to the conventional DocMT models.
- Advantage of Base LLMs for Task-Specific Supervised Fine-Tuning: Our study shows that base LLMs, when used as the backbone for task-specific supervised fine-tuning, perform better than instruction-tuned LLMs. They demonstrate more effective zero-shot cross-lingual transfer.


## 2 Related Work

Document-Level Machine Translation In recent years, numerous approaches have been proposed for document-level machine translation (DocMT). There exist other approaches to DocMT, including document embedding (Macé and Servan, 2019; Huo et al., 2020), multiple encoders (Wang et al., 2017; Bawden et al., 2018; Voita et al., 2018; Zhang et al., 2018), attention variations (Miculicich et al., 2018; Zhang et al., 2020; Maruf et al., 2019; Wong et al., 2020; Wu et al., 2023), and translation caches (Maruf and Haffari, 2018; Tu et al., 2018; Feng et al., 2022). Furthermore, Maruf et al. (2022) present a comprehensive survey of DocMT.

Large Language Models Large language models (LLMs) have demonstrated remarkable proficiency across a wide range of Natural Language Processing (NLP) tasks (Brown et al., 2020; Chowdhery et al., 2022; Scao et al., 2022; Anil et al., 2023; Touvron et al., 2023a,b). Furthermore, recent research has shown that supervised finetuning (SFT) and Reinforcement Learning from
Human Feedback (RLHF) can significantly enhance their performance when following general language instructions (Weller et al., 2020; Mishra et al., 2022; Wang et al., 2022; Shen et al., 2023; Li et al., 2023; Wu and Aji, 2023). More recently, there is a growing body of work exploring the translation capabilities of LLMs (Lu et al., 2023; Zhang et al., 2023; Xu et al., 2023; Robinson et al., 2023). However, it is important to note that these efforts have primarily focused on sentencelevel machine translation (SENMT) and have not delved into document-level machine translation (DocMT). A noteworthy study in DocMT is conducted by Wang et al. (2023b), where they investigate the document-level translation capabilities of GPT-3.5-TURBO, making it the most closely related work to our work.

Ours In contrast to the work of Wang et al. (2023b), who primarily investigate the use of GPT-3.5-TURBO for DOCMT through prompting techniques, our study concentrates on analyzing the effectiveness of parameter-efficient fine-tuning (PEFT) and full fine-tuning (FFT) methods on moderately-sized LLMs in the context of DocMT.

## 3 Experimental Setup

In this study, we aim to adapt multilingual pre-trained large language models (LLMs) into a bilingual document-level machine translation (DocMT) model. In this section, we describe our experimental setup of this work, including training strategy (Section 3.1), datasets (Section 3.2), models (Section 3.3), and evaluation (Section 3.4).

### 3.1 Two-Stage Training

DocMT approaches typically begin by pretraining the translation model on sentence-level parallel corpora, subsequently refining it through finetuning on document-level parallel corpora (Voita et al., 2019; Maruf et al., 2019; Ma et al., 2020; Sun et al., 2022; Wu et al., 2023). More recently, Xu et al. (2023) propose a two-stage training strategy, which initially involves fine-tuning a LLM on monolingual text, followed by a second finetuning phase on parallel text. Given that most state-of-the-art open-sourced LLMs are trained on English-centric corpora, our approach begins with the fine-tuning of a LLM on monolingual documents, followed by fine-tuning on parallel documents. Following Xu et al. (2023), we omit the step of fine-tuning on sentence-level parallel datasets.

Fine-tuning on Monolingual Documents Existing LLMs are typically pre-trained on Englishcentric corpora. Recent research highlights that these LLMs often exhibit sub-optimal performance on multilingual benchmarks (Li et al., 2023; Chen et al., 2023; Scao et al., 2022). To address this limitation, our initial step involves fine-tuning all the parameters of LLMs using monolingual data from the target languages.

Fine-tuning on Parallel Documents We finetune the model on document-level parallel corpora in this stage. Following Wang et al. (2023a), we condition each sentence pair on its context, consisting of the three preceding consecutive sentence pairs. As demonstrated by Wang et al. (2023b), the prompting strategy plays a significant role in translating documents using LLMs. However, they only investigate how the prompting strategies affect GPT-3.5-TURBO and GPT-4-TURBO at the inference stage. In our study, we first delve into how these prompting strategies impact the fine-tuning process, as shown in Figure 1, and we present our findings in Section 4.

### 3.2 Datasets

Parallel Documents Following Zhang et al. (2022), we conduct experiments on IWSLT2017 translation tasks (Cettolo et al., 2017). IWSLT2017 comprises translation datasets sourced from TED talks, encompassing translations between English and nine other languages, including Arabic, German, French, Italian, Japanese, Korean, Dutch, Romanian, and Chinese. There are approximately $1.9 \mathrm{~K}$ sentence-aligned parallel documents with about $240 K$ sentences for each language pair. The dataset statistics can be found in Appendix A.

Monolingual Documents We gather monolingual documents for all the target languages in our translation tasks, totaling ten languages. To manage computational limitations and address concerns about catastrophic forgetting that might result from excessive continued training, we leverage the data pruning technique suggested by Marion et al. (2023) to select $100 \mathrm{M}$ tokens for each language, including English, from the CulturaX corpus (Nguyen et al., 2023), totaling $1 B$ tokens.

### 3.3 Models

Baselines The baseline models in this study can be classified into three categories, including state- of-the-art LLMs and SENMT models, and our reimplemented DocMT models:

- State-of-the-art SENMT models: Our selection includes models such as NLLB, which are available with three different sets of parameters: $600 \mathrm{M}, 1.3 \mathrm{~B}$, and 3.3B. ${ }^{1}$ We also incorporate the widely-used commercial translation system, Google Translate.
- State-of-the-art LLMs: For our baseline LLMs in the context of DocMT, we utilize GPT-3.5-TURBO and GPT-4-TURBO. ${ }^{2}$ We use the Prompt 4 as detailed in Figure 1d during the translation process.
- Our re-implemented DocMT models: We conduct full fine-tuning on the concatenationbased DocMT model (Tiedemann and Scherrer, 2017), as well as several recent DocMT baselines (Sun et al., 2022; Wu et al., 2023, 2024), initialized with MT5 (Xue et al., 2021). These models are available with parameters of $300 \mathrm{M}, 580 \mathrm{M}$, and $1.2 \mathrm{~B}$, representing the strong DocMT baseline.

Ours In this work, we utilize Llama2-7B, BLOOM-7B, and VICUNA-7B, as our backbones. ${ }^{3}$ The Llama2 models are predominantly pretrained on English text, while the BLOOM models are pre-trained on multilingual text. The use of VICUNA models allows us to compare the differences between base models and instruction-tuned models (Llama2 vs. ViCUNA). We denote those fully fine-tuned models as L-7B-FFT, B-7B-FFT, and V-7B-FFT. We denote those models fine-tuned with LoRA (Hu et al., 2022) as L-7B-LoRA, B7B-LoRA, and V-7B-LoRA. The optimization details can be found in Appendix B.

### 3.4 Evaluation

Evaluation Metrics We evaluate the translation quality using sentence-level BLEU (Papineni et al., 2002) and document-level BLEU (Liu et al., 2020) using SacreBLEU (Post, 2018), denoted as $s$ BLEU and $d \mathrm{BLEU} .{ }^{4}$ Furthermore, as conventional MT[^0]

## (a) Prompt 1

[<src_lang> Context]: <src1> <src2> <src3>

[<tgt_lang> Context]: <tgt1> <tgt2> <tgt3>

Given the provided parallel context, translate the following $\hookrightarrow$ <src_lang> sentence to <tgt_lang>:

[<src_lang> Sentence]: <src4> (b) Prompt 2

[<src_lang>]: <src1> [<tgt_lang>]: <tgt1> [<src_lang>]: <src2> [<tgt_lang>]: <tgt2> [<src_lang>]: <src3> [<tgt_lang>]: <tgt3> Given the provided parallel sentence pairs, translate the following $\hookrightarrow$ <src_lang> sentence to <tgt_lang>.

[<src_lang>]: <src4> [<tgt_lang>]: <tgt4>

(c) Prompt 3

(d) Prompt 4

Figure 1: Prompt types used in the preliminary study. <src_lang> and <tgt_lang> indicate the source and target languages. <src*> and <tgt*> indicate the source and target sentences. Note that the target sentences <tgt*> are only used during training and are replaced with the hypotheses <hyp*> generated by the model during inference. Concrete examples for each prompt variation can be found in Appendix C.

|  | PID | $\mu_{s B L E U}$ | $\mu_{d \text { BLEU }}$ | $\mu_{\text {COMET }}$ |
| :---: | :---: | :---: | :---: | :---: |
| L-7B-LORA | 1 | 15.5 | 18.2 | 67.5 |
|  | 2 | 19.0 | 21.9 | 70.7 |
|  | 3 | 15.8 | 18.3 | 69.8 |
|  | 4 | $\mathbf{2 0 . 2}$ | $\mathbf{2 3 . 4}$ | $\mathbf{7 2 . 7}$ |
| B-7B-LoRA | 1 | 19.3 | 20.5 | 70.5 |
|  | 2 | 20.6 | 23.5 | 73.6 |
|  | 3 | 19.8 | 20.8 | 73.9 |
|  | 4 | $\mathbf{2 3 . 1}$ | $\mathbf{2 7 . 3}$ | $\mathbf{7 6 . 8}$ |
| V-7B-LORA | 1 | 19.0 | 22.4 | 74.2 |
|  | 2 | 20.4 | 23.5 | 71.6 |
|  | 3 | 18.3 | 21.4 | 70.0 |
|  | 4 | $\mathbf{2 2 . 4}$ | $\mathbf{2 5 . 7}$ | $\mathbf{7 6 . 2}$ |

Table 1: Overall performance given by L-7B-LoRA, B7B-LoRA, and V-7B-LoRA on different prompt variations, across four English-centric translation tasks involving German and Chinese. PID indicates the prompt ID in Figure 1. Best results are highlighted in bold.

metrics like BLEU demonstrate poor correlation to human judgments (Freitag et al., 2022), we also evaluate the translation quality with the state-of-theart neural evaluation metric COMET (Rei et al., 2020). ${ }^{5}$ Moreover, we use the average sentencelevel BLEU $\mu_{s \mathrm{BLEU}}$, the average document-level BLEU $\mu_{d \mathrm{BLEU}}$, and the average COMET $\mu_{\mathrm{COMET}}$ for the overall performance.

Inference We use beam search with the beam size of 5 during translation. As shown in Figure 1d, previous translations serve as the context for the current translation, so the test examples are translated in their original order, beginning with the first sentence free from context.[^1]

## 4 A Preliminary Study on Prompts

The prompt plays a crucial role in LLM research. Recent studies show that an optimal prompt can greatly enhance model performance and reveal unexpected model capabilities (Kojima et al., 2022; Wei et al., 2022b). Hence, our initial focus is on investigating the prompt's impact during fine-tuning.

Prompt Variations Displayed in Figure 1, our preliminary study features four prompt types. These designs aim to tackle two research questions: How does context structure impact translation quality? (Prompt 1 vs. Prompt 2) and How do natural language instructions influence translation quality? (Prompt 1 vs. Prompt 3). We also investigate the combined effect of these aspects in Prompt 4.

Results Our investigation analyzes prompt variations using three PEFT models (L-7B-LoRA, B-7B-LoRA, and V-7B-LoRA) on four Englishcentric translation tasks involving German and Chinese. Overall results are presented in Table 1. Comparing Prompt 1 (Figure 1a) and Prompt 2 (Figure 1b), we find that models fine-tuned with Prompt 2 generally outperform those with Prompt 1 , indicating Prompt 2's effectiveness in enhancing LLM performance. Regarding our second research question (Figure 1a vs. Figure 1c), we observe varied performance. L-7B-LoRA and B-7B-LoRA perform better with Prompt 3, while V-7B-LoRA performs better with Prompt 1. These results highlight varying impacts of prompt variations across models and suggest natural language instructions are less effective when using instruction-tuned language models as model backbones. Finally, LLMs with Prompt 4 (Figure 1d) achieve the best over-

|  | \# of param. | \# of train. <br> param. | En-X |  |  | X-En |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | $\mu_{s \mathrm{BLEU}}$ | $\mu_{d \mathrm{BLEU}}$ | $\mu_{\text {COMET }}$ | $\mu_{s \mathrm{BLEU}}$ | $\mu_{d \mathrm{BLEU}}$ | $\mu_{\mathrm{COMET}}$ |
| State-of-the-art SENMT baselines |  |  |  |  |  |  |  |  |
| NLLB | $600 \mathrm{M}$ | - | 23.6 | 27.3 | 82.3 | 18.2 | 22.1 | 72.8 |
|  | $1.3 \mathrm{~B}$ | - | 25.7 | 29.5 | 83.5 | 25.0 | 28.7 | 78.1 |
|  | 3.3B | - | 26.8 | 30.5 | 84.3 | 25.8 | 29.4 | 78.9 |
| GOOGLETRANS | - | - | $\frac{24.5}{24}$ | $\overline{28.4}$ | $\overline{81.6}$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_3d3172b2801158f54e3fg-05.jpg?height=45&width=118&top_left_y=494&top_left_x=1406) | 28.5 | 81.2 |
| State-of-the-art LLMs |  |  |  |  |  |  |  |  |
| GPT-3.5-TURBO | - | - | 26.3 | 30.1 | 85.3 | 30.7 | 34.1 | 85.5 |
| GPT-4-TURBO | - | - | $\underline{27.0}$ | 30.7 | $\underline{86.3}$ | 31.7 | $\underline{35.1}$ | $\underline{86.0}$ |
| LLM backbones |  |  |  |  |  |  |  |  |
| LLAMA2-7B | - | - | 2.7 | 3.5 | 40.1 | 4.2 | 4.4 | 52.2 |
| BLOOM-7B | - | - | 2.5 | 2.9 | 35.5 | 6.7 | 7.3 | 49.4 |
| VICUNA-7B | - | - | 10.2 | 12.4 | 64.7 | 9.5 | 9.8 | 62.9 |
| Re-implemented DocMT baselines |  |  |  |  |  |  |  |  |
| Doc2Doc-MT5 (2017) | $300 \mathrm{M}$ | $300 \mathrm{M}$ | 17.2 | 20.2 | 75.1 | 19.4 | 21.2 | 75.1 |
|  | $580 \mathrm{M}$ | $580 \mathrm{M}$ | 18.6 | 21.5 | 78.3 | 20.7 | 22.5 | 77.4 |
|  | $1.2 \mathrm{~B}$ | $1.2 \mathrm{~B}$ | 18.4 | 21.4 | 79.2 | 21.5 | 23.4 | 78.7 |
| MR-DOC2SEN-MT5 (2022) | $1.2 \mathrm{~B}$ | $1.2 \mathrm{~B}$ | 18.8 | 21.9 | 79.9 | 22.0 | 23.8 | 79.3 |
| MR-DOc2DOC-MT5 (2022) | $1.2 \mathrm{~B}$ | $1.2 \mathrm{~B}$ | - | $\underline{22.5}$ | - | - | 24.0 | - |
| DOCFLAT-MT5 (2023) | $1.2 \mathrm{~B}$ | $1.2 \mathrm{~B} \quad$ | 19.2 | $\overline{22.4}$ | 80.2 | 22.2 | 24.3 | 79.3 |
| IADA-MT5 (2024) | $1.2 \mathrm{~B}$ | $1.2 \mathrm{~B}$ | $\underline{19.3}$ | 22.4 | $\underline{80.4}$ | 22.1 | 24.0 | $\underline{79.5}$ |
| LLM-based DocMT models (Ours) |  |  |  |  |  |  |  |  |
| L-7B-LoRA | 7B | $8 \mathrm{M}$ | 17.2 | 20.2 | 70.8 | 23.8 | 25.7 | 73.7 |
| L-7B-FFT | $7 \mathrm{~B}$ | 7B | 13.7 | 16.2 | $\overline{67.4}$ | 22.4 | 24.1 | 74.0 |
| B-7B-LoRA | 7B | $8 \mathrm{M}$ | $\underline{17.7}$ | $\underline{20.5}$ | 68.5 | 29.9 | 33.6 | $\underline{81.4}$ |
| B-7B-FFT | 7B | 7B | $\overline{12.0}$ | $\overline{13.8}$ | 59.6 | 22.3 | $\overline{24.5}$ | $\overline{69.9}$ |
| V-7B-LoRA | 7B | $8 \mathrm{M}$ | 15.8 | 18.6 | 68.8 | 21.6 | 23.3 | 71.4 |
| V-7B-FFT | 7B | 7B | 14.3 | 16.8 | 65.0 | 21.8 | 23.5 | 74.3 |

Table 2: Overall performance on IWSLT2017. \# of param. indicates the number of parameters of the model. \# of train. param. indicates the number of trainable parameters of the model. All the LLM approaches use Prompt 4 (Figure 1d) during inference. Best results are highlighted in bold. Best results in each group are underlined.

all performance, suggesting a positive compound effect of context structure and instructions.

Conclusion As expected, the prompt plays a significant role in LLM performance. A wellstructured prompt, which combines an appropriate context structure and natural language instructions, can significantly boost model performance. In this work, we use Prompt 4 (Figure 1d) in our other experiments, unless otherwise mentioned.

## 5 Main Results

Overall Performance In our results presented in Table 2, we observe that GPT-4-TURBO and GPT-3.5-TURBO significantly outshine all other models in performance. Notably, the NLLB variants, which are trained on vast amount of parallel sentence pairs, also demonstrate superior performance among specialized machine translation (MT) models. In the context of DocMT, conventional DocMT models still outperform our LLM-based DocMT models for translations from English to other languages when evaluated using standard MT metrics. Conversely, for translations from other languages to English, our LLM-based DocMT models perform on par or better than conventional DocMT models in $\mu_{s \mathrm{BLEU}}$ and $\mu_{d \mathrm{BLEU}}$ metrics, while those conventional DocMT models maintain superior performance in $\mu_{\mathrm{COMET}}$.

LLM-based DocMT Models As indicated in Table 2, our models incorporating LoRA typically outperform fully fine-tuned (FFT) LLMs. However, an exception is observed where V-7B-FFT outperforms V-7B-LoRA in translating from other languages to English. This discrepancy is likely attributable to overfitting. In scenarios of extensive fine-tuning with a large corpus of parallel documents, the full fine-tuning of all parameters often leads to rapid overfitting on the training dataset. In contrast, the parameter-efficient fine-tuning approach, exemplified by LoRA, updates only a select number of parameters, effectively preventing the models from overfitting the training set. Furthermore, we observe that the L-7B and V-7B models exhibit comparable performance, suggest-

![](https://cdn.mathpix.com/cropped/2024_06_04_3d3172b2801158f54e3fg-06.jpg?height=574&width=1420&top_left_y=244&top_left_x=318)

Figure 2: Breakdown results on $s$ BLEU, $d$ BLEU, and COMET given by L-7B-LoRA, V-7B-LoRA, B-7BLoRA, DOc2DOC-MT5-1.2B, and GPT-4-TURBO for the translation tasks from other languages to English.

ing that initializing with instruction-tuned models does not always enhance task-specific performance.

Breakdown Performance We present the results for the translation tasks from other languages to English in Figure 2. Regarding the readability of the figures, we present only the results provided by our models using LoRA. Our LLM-based DocMT models exhibit superior performance, sometimes even surpassing GPT-4-TURBO in certain translation tasks. However, they fail completely in others. A manual review of translation tasks where our LLM-based DocMT models fail reveals that the primary cause of failure is off-target translation. We provide an in-depth analysis of the off-target translation problem in Section 6. A complete breakdown of the results is in Appendix E.

## 6 Analyses

In this section, we investigate the off-target problem and leverage GPT-4-TURBO to analyze the translation errors. We also explore discourse phenomena, the training strategy, and the scaling law of parallel documents. Furthermore, we conduct additional evaluations on recent test sets from WMT2023 and examine crosslingual transfer.

Off-Target Translation In Figure 2, our LLMbased DocMT models excel in some translation tasks but struggle in others due to off-target translation issues. We investigate this problem using the fasttext library (Bojanowski et al., 2017) to identify translation languages and quantify off-target rates, which represent the proportion of translations that are off-target. Results are presented in Table 3, with off-target rates reaching up to $98.3 \%$

![](https://cdn.mathpix.com/cropped/2024_06_04_3d3172b2801158f54e3fg-06.jpg?height=611&width=691&top_left_y=1011&top_left_x=1094)

Figure 3: Error type analysis given by GPT-4-TURBO for translations from English to German, Romanian, and Chinese. The error types in orange are contextdependent. We omit those error types that are rare or almost never occur.

in failing tasks. Notably, only B-7B-LoRA consistently maintains low off-target rates, likely due to BLOOM-7B's multilingual pre-training. These findings shed light on the main reason of translation failures in LLM-based DocMT models, offering insights for future research. Detailed off-target rates are provided in Appendix F.

Translation Errors To comprehensively understand the translation capabilities of our LLM-based DocMT models, we select specific error types from the Multidimensional Quality Metrics (MQM) framework (Burchardt, 2013). Kocmi and Federmann (2023) demonstrate GPT-4 is capable of identifying error spans and achieving state-of-theart MT evaluation accuracy, so we leverage GPT-

|  | $\mu_{\%}$ | $\mathrm{Ar}$ | $\mathrm{Ja}$ | Ko | Zh |
| :--- | ---: | ---: | ---: | ---: | ---: |
| L-7B-LoRA | 29.2 | 87.9 | 25.5 | 44.2 | 93.1 |
| L-7B-FFT | 40.2 | 87.9 | 75.5 | 92.3 | 93.6 |
| B-7B-LoRA | 2.8 | 2.9 | 4.0 | 8.4 | 1.6 |
| B-7B-FFT | 28.0 | 54.1 | 43.8 | 70.4 | 76.4 |
| V-7B-LoRA | 32.3 | 88.2 | 40.4 | 35.7 | 90.5 |
| V-7B-FFT | 44.7 | 94.1 | 98.3 | 96.6 | 94.6 |

Table 3: Off-target rate (\%) provided by our LLM-based DocMT models for translation tasks from selective languages to English. $\mu_{\%}$ indicates the average offtarget rate across all nine language pairs. A lower offtarget rate indicates better performance.

|  | Acc. | er | es | sie |
| :--- | :---: | :---: | :---: | :---: |
| Doc2DOC-MT5-1.2B | 77.0 | 68.7 | 89.0 | 73.5 |
| MR-DOC2SEN-MT5 | 59.9 | 48.9 | 91.4 | 39.4 |
| MR-DOC2DOC-MT5 | 78.2 | 67.5 | 91.1 | 76.1 |
| DOCFLAT-MT5 | 78.0 | 68.9 | 90.1 | 75.1 |
| IADA-MT5 | 79.1 | 70.0 | 89.8 | 77.6 |
| L-7B-LoRA | 83.1 | 77.2 | $\mathbf{9 6 . 6}$ | 75.4 |
| L-7B-FFT | 81.1 | 70.2 | 96.9 | 76.2 |
| B-7B-LoRA | 75.5 | 56.2 | 95.1 | 75.1 |
| B-7B-FFT | 68.3 | 50.8 | 95.5 | 58.5 |
| V-7B-LoRA | $\mathbf{8 4 . 9}$ | $\mathbf{7 8 . 4}$ | 96.2 | 80.1 |
| V-7B-FFT | 84.4 | 76.3 | 96.4 | $\mathbf{8 0 . 5}$ |

Table 4: Accuracy (in \%) on the English-German contrastive test set. Best results are highlighted in bold.

4-TURBO to analyze the translation errors of the text translated by these models. We focus on four models due to resource constraints: L-7B-LoRA, L-7B-FFT, Doc2Doc-MT5-1.2B, and GoogleTRANS, assessing translations from English to German, Romanian, and Chinese. The error identification prompt is detailed in Appendix D, and we present the frequency of error types in Figure 3. Notably, most errors are limited to individual sentences. Despite similar scores in metrics such as $s \mathrm{BLEU}, d \mathrm{BLEU}$, and COMET among the models, our LLM-based DocMT models (L7B-LoRA and L-7B-FFT) exhibit fewer contextindependent and context-dependent errors. This highlights a limitation in current evaluation metrics, suggesting they may not sufficiently assess document-level translations. It also indicates that fine-tuning LLMs for machine translation holds promise for enhancing DocMT performance.

Discourse Phenomena To evaluate our LLMbased DocMT model's ability to leverage contextual information, we assessed it using the EnglishGerman contrastive test set by Müller et al. (2018). This evaluation tests the model's accuracy in selecting the correct German pronoun ("er", "es",

|  | $s \mathrm{BLEU}$ | $d \mathrm{BLEU}$ | COMET |
| :---: | :---: | :---: | :---: |
| Two-Stage |  |  |  |
| Nl-En | 38.9 | 41.9 | 87.0 |
| Ro-En | 38.2 | 41.4 | 87.3 |
| Ar-En | 2.5 | 2.6 | 51.6 |
| Zh-En | 0.1 | 0.1 | 67.1 |
| Three-Stage |  |  |  |
| Nl-En | 39.1 | 42.1 | 87.0 |
| Ro-En | 38.4 | 41.6 | 87.3 |
| Ar-En | 2.3 | 2.4 | 52.4 |
| Zh-En | 0.3 | 0.3 | 67.4 |

Table 5: Comparison between two-stage and three-stage training strategies. The results of the two-stage strategy are given by L-7B-FFT. For the three-stage training strategy, we fine-tune all the model parameters of LLAMA2-7B in all three stages.

![](https://cdn.mathpix.com/cropped/2024_06_04_3d3172b2801158f54e3fg-07.jpg?height=405&width=691&top_left_y=968&top_left_x=1094)

Figure 4: COMET-Percentage (\%) of training data for the translations from English to German.

and "sie") from multiple translation options. Results, shown in Table 4, reveal that models initialized with LLAMA2-7B and VICUNA-7B outperform Doc2Doc-MT5-1.2B, while BLOOM-7Binitialized models perform worse, indicating that contextual understanding is mostly acquired during pre-training, as detailed by Scao et al. (2022) due to the lack of German text in BLOOM pre-training.

Training Strategy In this study, we follow the two-stage approach of Xu et al. (2023). Unlike traditional DocMT methods, which typically start with parallel sentence training, we explore the effectiveness of this conventional training strategy on LLM-based DocMT models. In this section, we introduce a three-stage training strategy, involving: (1) monolingual document fine-tuning, (2) parallel sentence fine-tuning, and (3) parallel document fine-tuning, for all parameters of the Llama27B. The results in Table 5 indicate that the threestage training strategy is unnecessary for both highperforming languages (Dutch and Romanian) and low-performing languages (Arabic and Chinese) with LLM-based DocMT models.

|  | $\mu_{\Delta}$ | $\mathrm{Ar}$ | $\mathrm{De}$ | $\mathrm{Fr}$ | $\mathrm{It}$ | $\mathrm{Ja}$ | $\mathrm{Ko}$ | $\mathrm{Nl}$ | $\mathrm{Ro}$ | $\mathrm{Zh}$ |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| L-7B-LoRA | +29.4 | +36.3 | +38.8 | +37.2 | +32.1 | +15.9 | +17.1 | +21.7 | +35.8 | +29.5 |
| L-7B-FFT | +29.0 | +41.2 | +40.5 | +37.1 | +18.0 | +27.7 | +29.4 | +11.2 | +18.5 | +37.5 |
| B-7B-LoRA | +20.3 | +7.5 | +40.7 | +20.7 | +21.9 | +17.5 | +15.9 | +23.7 | +25.3 | +9.8 |
| B-7B-FFT | +27.3 | +14.8 | +37.8 | +28.9 | +43.3 | +13.1 | +15.3 | +38.5 | +34.7 | +19.5 |
| V-7B-LoRA | -8.9 | -12.6 | +22.1 | +18.9 | -28.6 | -27.8 | -18.7 | -11.8 | +12.1 | -34.1 |
| V-7B-FFT | -1.4 | +7.3 | +25.2 | +17.7 | -14.6 | -24.7 | -5.3 | -21.8 | +7.6 | -3.5 |

Table 6: The difference $(\Delta)$ in COMET scores on the test sets from English to other languages between our English-German LLM-based DocMT models and their backbones. $\mu_{\Delta}$ indicates the average difference across all the languages in this table.

|  | En-De |  |  | De-En |  |
| :--- | :---: | :---: | :---: | :---: | :---: |
|  | $d$ BLEU COMET | BLEU COMET |  |  |  |
| DoC2DoC-MT5-1.2B | 20.2 | 74.4 |  | 20.0 | 76.5 |
| MR-DOC2SEN-MT5 | 20.5 | 74.9 |  | 21.0 | 76.5 |
| MR-Doc2DoC-MT5 | 21.2 | 75.6 |  | 21.5 | 76.5 |
| DocFLAT-MT5 | 20.9 | 75.1 | 21.8 | 76.5 |  |
| IADA-MT5 | 21.2 | 75.4 | 22.0 | 76.5 |  |
| L-7B-LoRA | 28.9 | 76.4 | 35.5 | 83.2 |  |
| L-7B-FFT | $\mathbf{2 9 . 0}$ | $\mathbf{7 7 . 0}$ | $\mathbf{3 6 . 1}$ | $\mathbf{8 4 . 0}$ |  |
| B-7B-LoRA | 23.7 | 73.0 | 30.5 | 80.8 |  |
| B-7B-FFT | 21.0 | 69.0 | 30.0 | 80.5 |  |
| V-7B-LoRA | 20.5 | 63.8 | 33.9 | 81.8 |  |
| V-7B-FFT | 27.8 | 75.0 | 34.7 | 83.1 |  |

Table 7: $d$ BLEU and COMET on WMT2023 test sets. Best results are highlighted in bold.

Scaling Law of Parallel Documents In this section, we explore the scaling law for fine-tuning parallel documents. We focus on English to German, Romanian, and Chinese translations due to our models' proficiency. Results for EnglishGerman translation are presented in Figure 4, and for English-Romanian and English-Chinese in Appendix G. While LLMs typically excel with minimal training data, different fine-tuning strategies show distinct scaling behaviors. Our LoRA models match full training set performance with just $10 \%$ of the data (around $20 K$ examples), while fully fine-tuned models achieve near-equivalent performance with only about $1 \%$ of the data (approximately $2 K$ examples). These insights are crucial for low-resource languages, as recent LLMs are predominantly pre-trained on English text.

Evaluation on Recent Test Sets Given their pretraining on extensive text corpora, LLMs may be susceptible to data leakage risks. We evaluate our models using recent test sets from WMT2023 (Koehn et al., 2023). These tests, conducted between English and German, not only evaluate the out-of-domain generalization of our models but also help mitigate the risks associated with data leakage. We use spaCy to segment documents and and discard any parallel documents where the source and target sides have a differing number of sentences. Our findings, presented in Table 7, reveal that while Doc2Doc-MT5 models outperform LLM-based models in Table 2, LLM-based models excel in translating out-of-domain text on the WMT2023 test sets. These findings highlight the ability of LLM-based DocMT to generalize well to out-of-domain translation tasks.

Zero-Shot Crosslingual Transfer In this section, we explore the transferability of translation capabilities acquired from one language pair to others. We assess our English-German LLM-based DocMT models on English-to-other-language test sets, comparing their COMET scores to their base models in Table 6. Our results indicate that models with fine-tuned instructions (LLAMA2-7B and BLOOM-7B) consistently exhibit positive transfer effects across all language pairs, while those with instruction-tuned backbones (VICUNA-7B) benefits only a few languages. These findings suggest that LLMs are more likely to activate their inherent translation abilities during fine-tuning rather than developing new ones.

## 7 Conclusion

This study investigates the adaptation of large language models (LLMs) for document-level machine translation (DocMT) through extensive experimentation with two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Results demonstrate that taskspecific supervised fine-tuning on parallel documents significantly boosts the performance of moderately-sized LLM-based models (with 7B parameters) in DocMT, surpassing GPT-4-TURBO in some cases. Our analysis offers insights into LLMbased DocMT models, providing a foundation for future advancements in the field of DocMT.

## 8 Limitations

Constraints on Model Scale Our research is confined to language models of a moderate size, specifically those with $7 B$ parameters. This limitation is due to the constraints of our available resources. Consequently, it is crucial to acknowledge that the outcomes of our study might vary if conducted with larger models.

Instability in Training The process of supervised fine-tuning for LLMs shows instability in our observations. As detailed in Figure 4, there are noticeable inconsistencies in performance. These variations are too significant to attribute solely to the randomness inherent in training. In some cases, the fine-tuning of LLMs fails to reach convergence Unfortunately, our limited resources restrict us from investigating these failures in depth or devising potential remedies.

Influence of Prompting Techniques Section 4 of our study highlights the significant role of prompting methods in fine-tuning. We experiment with four different prompting techniques. It is important to note that the prompt we recommend may not be the most effective, potentially leading to suboptimal performance of our models.

We acknowledge these limitations and leave them to the future work.

## References

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernández Ábrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan A. Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vladimir Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, and et al. 2023. Palm 2 technical report. CoRR, $\mathrm{abs} / 2305.10403$.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. CoRR, abs/2310.10631.
Rachel Bawden, Rico Sennrich, Alexandra Birch, and Barry Haddow. 2018. Evaluating discourse phenomena in neural machine translation. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1304-1313, New Orleans, Louisiana. Association for Computational Linguistics.

Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5:135-146.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In $A d$ vances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc.

Aljoscha Burchardt. 2013. Multidimensional quality metrics: a flexible system for assessing translation quality. In Proceedings of Translating and the Computer 35, London, UK. Aslib.

Mauro Cettolo, Marcello Federico, Luisa Bentivogli, Jan Niehues, Sebastian Stüker, Katsuhito Sudoh, Koichiro Yoshino, and Christian Federmann. 2017. Overview of the IWSLT 2017 evaluation campaign. In Proceedings of the 14th International Conference on Spoken Language Translation, pages 2-14, Tokyo, Japan. International Workshop on Spoken Language Translation.

Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Barry Haddow, and Kenneth Heafield. 2023. Monolingual or multilingual instruction tuning: Which makes a better alpaca. CoRR, abs/2309.08958.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira,

Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. CoRR, abs/2204.02311.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.

Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loïc Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation. CoRR, abs/2207.04672 .

Yukun Feng, Feng Li, Ziang Song, Boyuan Zheng, and Philipp Koehn. 2022. Learn to remember: Transformer with recurrent memory for document-level machine translation. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1409-1420, Seattle, United States. Association for Computational Linguistics.

Markus Freitag, Ricardo Rei, Nitika Mathur, Chi-kiu Lo, Craig Stewart, Eleftherios Avramidis, Tom Kocmi, George Foster, Alon Lavie, and André F. T. Martins. 2022. Results of WMT22 metrics shared task: Stop using BLEU - neural metrics are better and more robust. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 46-68, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. 2023. Textbooks are all you need. CoRR, $\mathrm{abs} / 2306.11644$.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita,
Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are GPT models at machine translation? A comprehensive evaluation. CoRR, abs/2302.09210.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Jingjing Huo, Christian Herold, Yingbo Gao, Leonard Dahlmann, Shahram Khadivi, and Hermann Ney. 2020. Diving deep into context-aware neural machine translation. In Proceedings of the Fifth Conference on Machine Translation, pages 604-616, Online. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. CoRR, abs/2310.06825.

Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. 2023. Is chatgpt A good translator? A preliminary study. CoRR, abs/2301.08745.

Tom Kocmi and Christian Federmann. 2023. GEMBAMQM: Detecting translation quality error spans with GPT-4. In Proceedings of the Eighth Conference on Machine Translation, pages 768-775, Singapore. Association for Computational Linguistics.

Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz, editors. 2023. Proceedings of the Eighth Conference on Machine Translation. Association for Computational Linguistics, Singapore.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In NeurIPS.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. 2023. MADLAD400: A multilingual and document-level large audited dataset. CoRR, abs/2309.04662.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023. Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation. CoRR, abs/2305.15011.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726-742.

Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Haoran Yang, Wai Lam, and Furu Wei. 2023. Chainof-dictionary prompting elicits translation in large language models. CoRR, abs/2305.06575.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023. Wizardcoder: Empowering code large language models with evolinstruct. CoRR, abs/2306.08568.

Shuming Ma, Dongdong Zhang, and Ming Zhou. 2020. A simple and effective unified encoder for documentlevel machine translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3505-3511, Online. Association for Computational Linguistics.

Valentin Macé and Christophe Servan. 2019. Using whole document context in neural machine translation. In Proceedings of the 16th International Conference on Spoken Language Translation, Hong Kong. Association for Computational Linguistics.

Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. 2023. When less is more: Investigating data pruning for pretraining llms at scale. CoRR, abs/2309.04564.

Sameen Maruf and Gholamreza Haffari. 2018. Document context neural machine translation with memory networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1275-1284, Melbourne, Australia. Association for Computational Linguistics.

Sameen Maruf, André F. T. Martins, and Gholamreza Haffari. 2019. Selective attention for context-aware neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3092-3102, Minneapolis, Minnesota. Association for Computational Linguistics.

Sameen Maruf, Fahimeh Saleh, and Gholamreza Haffari 2022. A survey on document-level neural machine translation: Methods and evaluation. ACM Comput. Surv., 54(2):45:1-45:36.

Lesly Miculicich, Dhananjay Ram, Nikolaos Pappas, and James Henderson. 2018. Document-level neural machine translation with hierarchical attention networks. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2947-2954, Brussels, Belgium. Association for Computational Linguistics.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470-3487, Dublin, Ireland. Association for Computational Linguistics.
Yasmin Moslem, Rejwanul Haque, John D. Kelleher, and Andy Way. 2023. Adaptive machine translation with large language models. In Proceedings of the 24th Annual Conference of the European Association for Machine Translation, pages 227-237, Tampere, Finland. European Association for Machine Translation.

Mathias Müller, Annette Rios, Elena Voita, and Rico Sennrich. 2018. A large-scale test set for the evaluation of context-aware pronoun translation in neural machine translation. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 61-72, Brussels, Belgium. Association for Computational Linguistics.

Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, and Thien Huu Nguyen. 2023. Culturax: A cleaned, enormous, and multilingual dataset for large language models in 167 languages. CoRR, $\mathrm{abs} / 2309.09400$.

OpenAI. 2023. GPT-4 technical report. CoRR, $\mathrm{abs} / 2303.08774$.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311-318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186191, Brussels, Belgium. Association for Computational Linguistics.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685-2702, Online. Association for Computational Linguistics.

Nathaniel Robinson, Perez Ogayo, David R. Mortensen, and Graham Neubig. 2023. ChatGPT MT: Competitive for high- (but not low-) resource languages. In Proceedings of the Eighth Conference on Machine Translation, pages 392-418, Singapore. Association for Computational Linguistics.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey,

M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. 2022. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. 2022. BLOOM: A 176b-parameter open-access multilingual language model. CoRR, abs/2211.05100.

Sheng Shen, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre, Jason Wei, Hyung Won Chung, Barret Zoph, William Fedus, Xinyun Chen, Tu Vu, Yuexin Wu, Wuyang Chen, Albert Webson, Yunxuan Li, Vincent Zhao, Hongkun Yu, Kurt Keutzer, Trevor Darrell, and Denny Zhou. 2023. Flan-moe: Scaling instruction-finetuned language models with sparse mixture of experts. CoRR, abs/2305.14705.

Zewei Sun, Mingxuan Wang, Hao Zhou, Chengqi Zhao, Shujian Huang, Jiajun Chen, and Lei Li. 2022. Rethinking document-level neural machine translation. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3537-3548, Dublin, Ireland. Association for Computational Linguistics.

Jörg Tiedemann and Yves Scherrer. 2017. Neural machine translation with extended context. In Proceedings of the Third Workshop on Discourse in Machine Translation, pages 82-92, Copenhagen, Denmark. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Zhaopeng Tu, Yang Liu, Shuming Shi, and Tong Zhang. 2018. Learning to remember translation history with a continuous cache. Transactions of the Association for Computational Linguistics, 6:407-420.

Elena Voita, Rico Sennrich, and Ivan Titov. 2019. When a good translation is wrong in context: Context-aware machine translation improves on deixis, ellipsis, and lexical cohesion. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1198-1212, Florence, Italy. Association for Computational Linguistics.

Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. 2018. Context-aware neural machine translation learns anaphora resolution. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1264-1274, Melbourne, Australia. Association for Computational Linguistics.

Longyue Wang, Siyou Liu, Mingzhou Xu, Linfeng Song, Shuming Shi, and Zhaopeng Tu. 2023a. A survey on zero pronoun translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3325-3339, Toronto, Canada. Association for Computational Linguistics.

Longyue Wang, Chenyang Lyu, Tianbo Ji, Zhirui Zhang, Dian Yu, Shuming Shi, and Zhaopeng Tu. 2023b. Document-level machine translation with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16646-16661, Singapore. Association for Computational Linguistics.

Longyue Wang, Zhaopeng Tu, Andy Way, and Qun Liu. 2017. Exploiting cross-sentence context for neural machine translation. In Proceedings of the 2017

Conference on Empirical Methods in Natural Language Processing, pages 2826-2831, Copenhagen, Denmark. Association for Computational Linguistics.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085-5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022a. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.

Orion Weller, Nicholas Lourie, Matt Gardner, and Matthew E. Peters. 2020. Learning from task descriptions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1361-1375, Online. Association for Computational Linguistics.

KayYen Wong, Sameen Maruf, and Gholamreza Haffari. 2020. Contextual neural machine translation improves translation of cataphoric pronouns. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 59715978, Online. Association for Computational Linguistics.

Minghao Wu and Alham Fikri Aji. 2023. Style over substance: Evaluation biases for large language models. CoRR, abs/2307.03025.

Minghao Wu, George Foster, Lizhen Qu, and Gholamreza Haffari. 2023. Document flattening: Beyond concatenating context for document-level neural machine translation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 448-462, Dubrovnik, Croatia. Association for Computational Linguistics.
Minghao Wu, Yufei Wang, George Foster, Lizhen Qu, and Gholamreza Haffari. 2024. Importance-aware data augmentation for document-level neural machine translation. arXiv preprint arXiv:2401.15360.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2023. A paradigm shift in machine translation: Boosting translation performance of large language models. CoRR, abs/2309.11674.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483-498, Online. Association for Computational Linguistics.

Wen Yang, Chong Li, Jiajun Zhang, and Chengqing Zong. 2023. Bigtrans: Augmenting large language models with multilingual translation capability over 100 languages. CoRR, abs/2305.18098.

Biao Zhang, Ankur Bapna, Melvin Johnson, Ali Dabirmoghaddam, Naveen Arivazhagan, and Orhan Firat. 2022. Multilingual document-level translation enables zero-shot transfer from sentences to documents. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4176-4192, Dublin, Ireland. Association for Computational Linguistics.

Jiacheng Zhang, Huanbo Luan, Maosong Sun, Feifei Zhai, Jingfang Xu, Min Zhang, and Yang Liu. 2018. Improving the transformer translation model with document-level context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 533-542, Brussels, Belgium. Association for Computational Linguistics.

Pei Zhang, Boxing Chen, Niyu Ge, and Kai Fan. 2020. Long-short term masking transformer: A simple but effective baseline for document-level neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1081-1087, Online. Association for Computational Linguistics.

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. 2023. Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models. CoRR, $\mathrm{abs} / 2306.10968$.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023. Multilingual machine translation with large language models: Empirical results and analysis. CoRR, abs/2304.04675.

|  | Train |  |  | Validation |  |  | Test |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
|  | \# of sen.\# of doc.\# of sen.\# of doc.\# of sen.\# of doc. |  |  |  |  |  |  |  |
| En-Ar | $232 \mathrm{~K}$ | 1907 | 2453 | 19 | 1460 | 12 |  |  |
| En-De | $206 \mathrm{~K}$ | 1705 | 2456 | 19 | 1138 | 10 |  |  |
| En-Fr | $233 \mathrm{~K}$ | 1914 | 2458 | 19 | 1455 | 12 |  |  |
| En-It | $232 \mathrm{~K}$ | 1902 | 2495 | 19 | 1147 | 10 |  |  |
| En-Ja | $223 \mathrm{~K}$ | 1863 | 2420 | 19 | 1452 | 12 |  |  |
| En-Ko | $230 \mathrm{~K}$ | 1920 | 2437 | 19 | 1429 | 12 |  |  |
| En-N1 | 237K | 1805 | 2780 | 19 | 1181 | 10 |  |  |
| En-Ro | $221 \mathrm{~K}$ | 1812 | 2592 | 19 | 1129 | 10 |  |  |
| En-Zh | $231 \mathrm{~K}$ | 1906 | 2436 | 19 | 1459 | 12 |  |  |

Table 8: Dataset statistics of parallel documents.
</end of paper 0>


<paper 1>
# Context-Aware Machine Translation with Source Coreference Explanation 

Huy Hien Vu Hidetaka Kamigaito Taro Watanabe<br>Nara Institute of Science and Technology, Japan.<br>\{vu.huy_hien.va9, kamigaito.h, taro\}@is.naist.jp


#### Abstract

Despite significant improvements in enhancing the quality of translation, context-aware machine translation (MT) models underperform in many cases. One of the main reasons is that they fail to utilize the correct features from context when the context is too long or their models are overly complex. This can lead to the explain-away effect, wherein the models only consider features easier to explain predictions, resulting in inaccurate translations. To address this issue, we propose a model that explains the decisions made for translation by predicting coreference features in the input. We construct a model for input coreference by exploiting contextual features from both the input and translation output representations on top of an existing MT model. We evaluate and analyze our method in the WMT document-level translation task of English-German dataset, the English-Russian dataset, and the multilingual TED talk dataset, demonstrating an improvement of over 1.0 BLEU score when compared with other context-aware models.


## 1 Introduction

With the rapid development of machine learning techniques, the Machine Translation (MT) field has witnessed changes from exclusively probabilistic models (Brown et al., 1990; Koehn et al., 2003) to neural network based models, such as simplistic Recurrent Neural Network (RNN) based encoder-decoder models (Sutskever et al., 2014) or higher-level attention-based models (Bahdanau et al., 2015; Luong et al., 2015), and finally turn to the current state-of-the-art Transformer model (Vaswani et al., 2017) and its variations.

The quality of MT models, including RNNbased, attention-based, and Transformer models, has been improved by incorporating contextual information (Voita et al., 2018; Wang et al., 2017; and others.), or linguistic knowledge (Bugliarello and Okazaki, 2020; Sennrich and Haddow, 2016; and others). In the former context-aware methods, many successful approaches focus on context selection from previous sentences (Jean et al., 2017; Wang et al., 2017) using multiple steps of translation, including additional module to refine translations produced by context-agnostic MT system, to utilize contextual information (Voita et al., 2019; Xiong et al., 2019), and encoding all context information as end-to-end frameworks (Zhang et al., 2020; Bao et al., 2021). Although they have demonstrated improved performance, there are still many cases in which their models perform incorrectly for handling, i.e., the ellipsis phenomenon in a long paragraph. One of the reasons is that their models are still unable to select the right features from context when the context is long, or the model is overly complex. Therefore, the model will easily suffer from an explain-away effect (Klein and Manning, 2002; Yu et al., 2017; Shah et al., 2020; Refinetti et al., 2023) in which a model is learned to use only features which are easily exploited for prediction by discarding most of the input features.

In order to resolve the problem of selecting the right context features in the context-aware MT, we propose a model which explains decisions of translation by predicting input features. The input prediction model employs the representations of translation outputs as additional features to predict contextual features in the inputs. In this work, we employ coreference as the prediction task since it captures the relation of mentions that are necessary for the context-aware model. The prediction model is constructed on top of an existing MT model without modification in the same manner as done in multi-task learning, but it fuses information from representations used for the decisions of translation in the MT model.

Under the same settings of the English-Russian (En-Ru) dataset and the WMT document-level
translation task of the English-German (En-De) dataset, our proposed technique outperforms the standard transformer-based neural machine translation (NMT) model in both sentence and contextaware models, as well as the state-of-the-art context-aware model measured by BLEU (Post, 2018), BARTScore (Yuan et al., 2021) and COMET (Rei et al., 2020), and the human-annotated test set in a paragraph (Voita et al., 2019). Additionally, in the multilingual experiments, our method shows consistent results, paralleling those in the En-Ru and En-De datasets, and proving its versatility across languages.

Further analysis shows that our coreference explanation sub-model consistently enhances the quality of translation, regardless of type of dataset size. Notably, the model demonstrates consistent improvement when additional context is incorporated, highlighting its effectiveness in handling larger context sizes. Additionally, the analysis highlights a strong correlation between the selfattention heat map and coreference clusters, underscoring the significance of our coreference prediction sub-model in capturing coreference information during the translation process. Moreover, our proposed training method proves to be effective in the coreference prediction task. We also provide a suggestion to finetune the contribution of the submodel to optimize its impact within the overall MT system. We release our code and hyperparameters at https://github.com/hienvuhuy/TransCOREF.

## 2 Backgrounds

### 2.1 Transformer-based NMT

Given an input single sentence $\boldsymbol{x}=\left(x_{1}, \ldots, x_{|\boldsymbol{x}|}\right)$ and its corresponding translation $\boldsymbol{y}=\left(y_{1}, \ldots, y_{|\boldsymbol{y}|}\right)$, an MT system directly models the translation probability

$$
\begin{equation*}
p(\boldsymbol{y} \mid \boldsymbol{x} ; \theta)=\prod_{t=1}^{|\boldsymbol{y}|} p\left(y_{t} \mid \boldsymbol{y}_{<t}, \boldsymbol{x} ; \theta\right) \tag{1}
\end{equation*}
$$

where $t$ is the index of target tokens, $\boldsymbol{y}_{<t}$ is the partial translation before $y_{t}$, and $\theta$ is the model parameter. At inference time, the model will find the most likely translation $\hat{\boldsymbol{y}}$ for a given source input

$$
\begin{equation*}
\hat{\boldsymbol{y}}=\underset{\boldsymbol{y}}{\operatorname{argmax}} \prod_{t=1}^{|\boldsymbol{y}|} p\left(y_{t} \mid \boldsymbol{y}_{<t}, \boldsymbol{x} ; \theta\right) \tag{2}
\end{equation*}
$$

To model the translation conditional probability $p(\boldsymbol{y} \mid \boldsymbol{x} ; \theta)$, many encoder-decoder architectures have been proposed based either on CNNs (Gehring et al., 2017) or self-attention (Vaswani et al., 2017), and we focus on the Transformer (Vaswani et al., 2017) as our building block, given its superior ability to model long-term dependencies and capture context information features. The encoder of the Transformer comprises $l_{e}$ stacked layers which transforms the input $\boldsymbol{x}$ into hidden representations $\mathbf{H}_{e n c}^{l_{e}} \in \mathbb{R}^{|\boldsymbol{x}| \times d}$ where $d$ is a dimension for hidden vector representation. Similarly, the decoder of the Transformer comprises $l_{d}$ stacked layers which consumes the translation prefix $\boldsymbol{y}_{<t}$ and $\mathbf{H}_{e n c}^{l_{e}}$ to yield the final representation $\mathbf{H}_{d e c}^{l_{d}} \in \mathbb{R}^{|\boldsymbol{y}| \times d}$. The two processes can be formally denoted as

$$
\begin{align*}
& \mathbf{H}_{e n c}^{i}=\operatorname{ENC}\left(\mathbf{H}_{e n c}^{i-1}\right)  \tag{3}\\
& \mathbf{H}_{d e c}^{i}=\operatorname{DEC}\left(\mathbf{H}_{d e c}^{i-1}, \mathbf{H}_{e n c}^{l_{e}}\right) \tag{4}
\end{align*}
$$

Note that $\mathbf{H}_{e n c}^{0}$ is the representation of $\boldsymbol{x}$ from the embedding layer, and $\mathbf{H}_{d e c}^{0}$ is the representation of $\boldsymbol{y}$ after embedding lookup with shifting by the begin-of-sentence token. $\operatorname{ENC}(\cdot)$ and $\operatorname{DEC}(\cdot)$ denote the function of the single Transformer encoder and decoder layer, respectively.

The output target sequence is predicted based on the output hidden state $\mathbf{H}_{d e c}^{l_{d}}$ from the top layer of the decoder

$$
\begin{align*}
& p\left(y_{t} \mid \boldsymbol{y}_{<t}, \boldsymbol{x} ; \theta\right) \\
& \quad=\operatorname{SoFTMAx}\left(\mathbf{W}_{d e c} \mathbf{H}_{d e c}^{l_{d}}[t]\right)\left[y_{t}\right] \tag{5}
\end{align*}
$$

where $\mathbf{W}_{\text {dec }} \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the projection weight matrix which maps the hidden state to the probability in the output vocabulary space $\mathcal{V}$, and $[\cdot]$ denotes an index/slice to a vector/matrix.

The standard training objective is to minimize the cross-entropy loss function

$$
\begin{equation*}
\mathcal{L}_{\mathrm{MT}}=-\sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{D}} \sum_{t=1}^{|\boldsymbol{y}|} \log p\left(y_{t} \mid \boldsymbol{y}_{<t}, \boldsymbol{x} ; \theta\right) \tag{6}
\end{equation*}
$$

given a parallel corpus $\mathcal{D}=\left\{\left(\boldsymbol{x}^{w}, \boldsymbol{y}^{w}\right)\right\}_{w=1}^{|\mathcal{D}|}$ which contains $|\mathcal{D}|$ pairs of single sentence and its corresponding translation.

### 2.2 Context-Aware Transformer-base NMT

A context-aware MT model can be regarded as a model which takes a document, i.e., multiple sentences, as an input and generates multiple sentences
as its corresponding translation. We assume that each sentence is translated into a single sentence, and define the source document $\underline{x}=\left(\boldsymbol{x}^{1}, \ldots, \boldsymbol{x}^{n}\right)$ with $n$ sentences and its corresponding target language document $\boldsymbol{y}=\left(\boldsymbol{y}^{1}, \ldots, \boldsymbol{y}^{n}\right)$. A contextaware MT system directly models the translation probability

$$
\begin{equation*}
p(\underline{\boldsymbol{y}} \mid \underline{\boldsymbol{x}} ; \theta)=\sum_{k=1}^{n} p\left(\boldsymbol{y}^{k} \mid \boldsymbol{y}^{<k}, \underline{\boldsymbol{x}} ; \theta\right) \tag{7}
\end{equation*}
$$

where $k$ is an index to a sentence in $\underline{y}, \boldsymbol{y}^{<k}$ is the partial translation before the sentence $\boldsymbol{y}^{k}$. In this model, we assume that $\langle\underline{\boldsymbol{x}}, \boldsymbol{y}\rangle$ constitute a parallel document and each $\left\langle\boldsymbol{x}^{k}, \boldsymbol{y}^{k}\right\rangle$ forms a parallel sentence.

Several approaches can be used to produce a translated document, i.e., keeping a sliding window of size $m$ (Tiedemann and Scherrer, 2017), joining these $m$ sentences as a single input, translating these $m$ sentences and selecting the last sentence as an output ( $m$-to- $m$ ) (Zhang et al., 2020), or joining whole sentences in a document as a very long sequence and translating this sequence (Bao et al., 2021), amongst other methods. To simplify the definition of the context-aware NMT model, we opt for the $m$-to- $m$ method and use a special character (_eos) between sentences when feeding these $m$ sentences to the model. In this way, the contextaware translation model can still be defined as a standard sentence-wise translation in $\$ 2.1$.

### 2.3 Coreference Resolution task

Coreference Resolution is the task of identifying and grouping all the mentions or references of a particular entity within a given text into a cluster, i.e., a set of spans. This task has progressed significantly from its earlier approaches, which were based on hand-crafted feature systems (McCarthy and Lehnert, 1995; Aone and William, 1995), to more advanced and effective deep learning approaches based on spans-ranking (Lee et al., 2017, 2018; Kirstain et al., 2021) and for multilingual languages (Zheng et al., 2023).

It is typically formulated as explicitly identifying an antecedent span to the left of a mention span in the same cluster. More formally, a set of clusters $\mathcal{C}=\left\{\ldots, \mathcal{C}_{k}, \ldots\right\}$ is predicted for an input sequence $\boldsymbol{x}$, either a document or a single sentence, with each cluster comprising a set of non-overlapping spans $\mathcal{C}_{k}=\{(i, j): 1 \leq i \leq j \leq|\boldsymbol{x}|\}(1 \leq k \leq|\mathcal{C}|)$. We introduce an alternative view using a variable $\mathcal{A}$ which represents mapping for all possible mention spans $\mathcal{S}=\{(i, j): \forall 1 \leq i \leq j \leq|x|\}$ of $\boldsymbol{x}$ to its antecedent span within the sample cluster $\mathcal{C}_{k}$, i.e., $\mathcal{A}=\{\ldots, s \rightarrow c, \ldots\}$, where $c \in \mathcal{C}_{k}$ is an antecedent to the left of $s \in \mathcal{C}_{k}$ and $c=\epsilon$, i.e., an empty span, when $s$ is not a member of any clusters $\mathcal{C}_{k}$. Note that we can derive a unique $\mathcal{C}$ given a single derivation of $\mathcal{A}$ by forming a cluster of spans connected by antecedent links, but there are multiple derivations of $\mathcal{A}$ for $\mathcal{C}$ when there exists a cluster $\left|\mathcal{C}_{k}\right|>2$. The task is modeled by the conditional probability distribution of independently predicting any possible antecedents of a mention span in the same cluster

$$
\begin{align*}
& p(\mathcal{C} \mid \boldsymbol{x})=\sum_{\mathcal{A} \in a(\mathcal{C})} p(\mathcal{A} \mid \boldsymbol{x}) \\
& =\prod_{s \in \mathcal{S}} \sum_{\mathcal{A} \in a(\mathcal{C})} p\left(\mathcal{A}_{s} \mid s, \boldsymbol{x}\right) \\
& =\prod_{s \in \mathcal{S}} \sum_{\mathcal{A} \in a(\mathcal{C})} \frac{\exp \left(f\left(\mathcal{A}_{s}, s ; \mathbf{H}_{\text {coref }}\right)\right)}{\sum_{c \in \mathcal{M}_{s}} \exp \left(f\left(c, s ; \mathbf{H}_{\text {coref }}\right)\right)} \\
& \triangleq \prod_{s \in \mathcal{S}} \sum_{\mathcal{A} \in a(\mathcal{C})} \operatorname{CoreF}\left(\mathcal{A}_{s}, s ; \mathbf{H}_{\text {coref }}\right) \tag{8}
\end{align*}
$$

where $a(\cdot)$ is a function that returns all possible derivations for clusters and $\mathcal{M}_{s}$ is a set of all possible spans to the left of $s$ including $\epsilon . f(\cdot, \cdot)$ is a score function (Kirstain et al., 2021) to compute both mention and antecedent scores and $\mathbf{H}_{\text {coref }} \in$ $\mathbb{R}^{|x| \times d}$ is contextualized representation of the input sequence $\boldsymbol{x}$, i.e., BERT (Devlin et al., 2019). We denote the final function as $\operatorname{CoreF}(\cdot, \cdot)$ for brevity.

We adopt the training scheme proposed by Kirstain et al. (2021), which filters spans to avoid the explicit enumeration of all possible mention spans, and represents antecedent relations using only the endpoints of the retained spans with a biaffine transformation. At the training stage, we minimize the negative log-likelihood of predicting clusters

$$
\begin{equation*}
\mathcal{L}_{\text {CoREF }}=-\sum_{(\mathcal{C}, \boldsymbol{x}) \in \mathcal{D}_{\text {CoREF }}} \log \prod_{s \in \mathcal{S}} \sum_{\mathcal{A} \in a(\mathcal{C})} p\left(\mathcal{A}_{s} \mid s, \boldsymbol{x}\right) \tag{9}
\end{equation*}
$$

where $\mathcal{D}_{\text {Coref }}$ is a training data for coreference resolution.

## 3 Context-Aware MT with Coreference Information

Our motivation stems from the observation that when translating a paragraph, translators are able
to pick up precise words and explain why a particular choice of word is better given the context especially by relying on linguistic cues such as discourse structure, verb equivalence, etc. Thus, instead of modeling a translation by directly relying on an additional conditional variable of coreference clusters $\mathcal{C}$ for $\boldsymbol{x}$, i.e., $p(\boldsymbol{y} \mid \mathcal{C}, \boldsymbol{x})$, we propose a model that is akin to the noisy channel framework (Yee et al., 2019), to explain the decision $y$ made by the translation model :

$$
\begin{align*}
p(\boldsymbol{y} \mid \mathcal{C}, \boldsymbol{x}) & =\frac{p(\boldsymbol{y}, \mathcal{C} \mid \boldsymbol{x})}{p(\mathcal{C} \mid \boldsymbol{x})}=\frac{p(\boldsymbol{y} \mid \boldsymbol{x})(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})}{p(\mathcal{C} \mid \boldsymbol{x})} \\
& \propto p(\boldsymbol{y} \mid \boldsymbol{x}) p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x}) \tag{10}
\end{align*}
$$

where $p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})$ is a model to predict coreference clusters given both an input sentence and its translation. Note that we can omit the denominator $p(\mathcal{C} \mid \boldsymbol{x})$ given that it is a constant when predicting $\boldsymbol{y}$, similar to the noisy channel modeling, since both $\boldsymbol{x}$ and $\mathcal{C}$ are input to our model. The direct model $p(\boldsymbol{y} \mid \mathcal{C}, \boldsymbol{x})$ is prone to ignore features in $\mathcal{C}$, especially when the context is long, since the information in $\boldsymbol{x}$ has direct correspondence with $\boldsymbol{y}$. In contrast, the model for coreference resolution task, $p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})$, will explain the coreference cluster information in $\boldsymbol{x}$ not only by the features from $x$ but additional features from $\boldsymbol{y}$ and, thus, the higher $p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})$, the more likely $\boldsymbol{y}$ will be a translation for $\boldsymbol{x}$. When coupled with the translation model $p(\boldsymbol{y} \mid \boldsymbol{x})$ especially when jointly trained together, our formulation will be able to capture long-distance relations in coreference clusters since the coreference resolution task needs to predict it given $\boldsymbol{x}$ and $\boldsymbol{y}$.

Architecture The two sub-models, i.e., $p(\boldsymbol{y} \mid \boldsymbol{x})$ and $p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})$, could be trained separately as done in a noisy channel modeling approach of MT (Yee et al., 2019). This work formulates it as a multitask setting by predicting two tasks jointly, i.e., translation task and coreference resolution task, by using the representations of the encoder and decoder of Vaswani et al. (2017). More specifically, we do not alter translation task $p(\boldsymbol{y} \mid \boldsymbol{x})$, but obtain the representation for the coreference task by fusing the representations of the encoder and decoder as follows

$$
\begin{align*}
p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x}) & =\prod_{s \in \mathcal{S}} \sum_{\mathcal{A} \in a(\mathcal{C})} \operatorname{CoREF}\left(\mathcal{A}_{s}, s ; \mathbf{H}_{\text {coref }}^{\prime}\right) \\
\mathbf{H}_{\text {coref }}^{\prime} & =\operatorname{DEC}\left(\boldsymbol{H}_{e n c}^{l_{e}}, \boldsymbol{H}_{\text {dec }}^{l_{d}}\right) . \tag{11}
\end{align*}
$$

Note that we obtain $\mathbf{H}_{\text {coref }}^{\prime}$ from an additional decoder layer for the encoder representation $\boldsymbol{H}_{e n c}^{l_{e}}$

|  | Avg. \#Coref. Clusters <br> train/valid/test | \#Samples <br> train/valid/test |
| :---: | :---: | :---: |
| En-Ru | $3.1 / 3.0 / 2.9$ | $1.5 \mathrm{M} / 10 \mathrm{k} / 10 \mathrm{k}$ |
| $\mathrm{En}-\mathrm{De}$ | $4.4 / 4.4 / 4.4$ | $206 \mathrm{k} / 8 \mathrm{k} / 2 \mathrm{k}$ |

Table 1: Statistics of En-De and En-Ru datasets.

with cross attention for $\boldsymbol{H}_{d e c}^{l_{d}}$.

Training We jointly train our two sub-models using the label-smoothing variant of the crossentropy loss function in Equation 6 and the marginal log-likelihood loss function in Equation 9, but using $\mathbf{H}_{\text {coref }}^{\prime}$ in Equation 11 as follows

$$
\begin{equation*}
\mathcal{L}=\mathcal{L}_{\mathrm{MT}}+\alpha \mathcal{L}^{\prime}{ }_{\mathrm{COREF}} \tag{12}
\end{equation*}
$$

where $\alpha$ is a hyperparameter that controls a contribution of the coreference resolution task. During the training step, we feed pairs of sentences together with coreference cluster information generated by an external coreference resolution framework since human annotation is not available in MT tasks.

Inference Inference is complex in that the model for the coreference resolution task has to be evaluated every time a target token is generated by the translation model as done in the noisy channel approach of MT (Yee et al., 2019). We resort to a simpler approach of ignoring the term for coreference clusters, i.e., $p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x})$, and using only the token prediction, i.e., $p(\boldsymbol{y} \mid \boldsymbol{x})$; alternatively, we generate a large set of $N$-best translations from $p(\boldsymbol{y} \mid \boldsymbol{x})$ and rerank them using the joint probabilities

$$
\begin{equation*}
\log p(\boldsymbol{y} \mid \boldsymbol{x})+\beta \log p(\mathcal{C} \mid \boldsymbol{y}, \boldsymbol{x}) \tag{13}
\end{equation*}
$$

where $\beta$ is a hyperparameter to control the strength of the coreference resolution task.

## 4 Experiments

### 4.1 Dataset

We utilized the En-Ru dataset (Voita et al., 2019) and the widely adopted En-De benchmark dataset IWSLT 2017, as used in Maruf et al. (2019), with details provided in Table 1. We also used the multilingual TED talk dataset (Qi et al., 2018) to assess the efficacy of our proposed method across a variety of language types, including different characteristics in pronouns, word order and gender assignment with specifics delineated in Table 2.

|  | Family | WO | PP | GP | GA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| English | IE | SVO | $\diamond$ | 3SG | SEM |
| Russian | IE | SVO | $\uparrow$ | 3SG | S-F |
| German | IE | SOV/SVO | $\bigcirc$ | 3SG | S-F |
| Spanish | IE | SVO | $\bigcirc$ | $1 / 2 / 3 P$ | SEM |
| French | IE | SVO | $\bigcirc$ | $3 S G$ | S-F |
| Japanese | JAP | SOV | $\bullet$ | $3 P$ | $\diamond$ |
| Romanian | IE | SVO | $\uparrow$ | 3SG | S-F |
| Mandarin | ST | SVO | $\bigcirc$ | 3SG | $\diamond$ |
| Vietnamese | AA | SVO | $\uparrow$ | $\diamond$ | $\diamond$ |

Table 2: Properties of Languages in Our Experiments: WO (Word Order), PP (Pronouns Politeness), GP (Gendered Pronouns), and GA (Gender Assignment) denote language structural properties. IE (Indo-European), JAP (Japonic), ST (Sino-Tibetan), and AA (Austroasiatic) represent language families. Symbols $\diamond, \oslash, \boldsymbol{\&}$, and $\boldsymbol{\uparrow}$ correspond to 'None', 'Binary', 'Avoided', and 'Multiple', respectively. The terms 3SG (Third Person Singular), 1/2/3P (First, Second, and Third Person), and 3P (Third Person) are used for pronoun references. SEM and S-F stand for Semantic and Semantic-Formal, respectively, in Gender Assignment.

The En-Ru dataset comes from OpenSubtitle 2018 (Lison et al., 2018) by sampling training instances with three context sentences after tokenization and, thus, no document boundary information is preserved. In the En-De and multilingual datasets, document boundaries are provided. To maintain consistency in our translation settings during experiments, we tokenize all texts by using MeCab ${ }^{1}$ for Japanese, Jieba ${ }^{2}$ for Chinese, VnCoreNLP (Vu et al., 2018) for Vietnamse and the spaCy framework ${ }^{3}$ for all other languages. We also apply a sliding window with a size of $m$ sentences ( $m=4$ ) to each document to create a similar format to that of the En-Ru dataset. For the first $m-1$ sentences, which do not have enough $m-1$ context sentences in the $m$-to- $m$ translation settings, we pad the beginning of these sentences with empty sentences, ensuring $m-1$ context sentences for all samples in the dataset. For preprocessing, we apply the BPE (Byte-Pair Encoding) technique from Sennrich et al. (2016) with $32 \mathrm{~K}$ merging operations to all datasets. To identify coreference clusters for the source language, i.e., English, we leveraged the AllenNLP framework ${ }^{4}$ and employed the SpanBERT large model (Lee et al., 2018). After generating sub-word units, we adjust the word-wise indices[^0]

of all members in coreference clusters using the offsets for sub-word units.

### 4.2 Experiment Settings

Translation setting In our experiments, we adopt the context-aware translation settings ( $m$-to$m$ with $m=4$ ) utilized in previous works (Zhang et al., 2020). For the context-agnostic setting, we translate each sentence individually.

Baselines systems We adopt the Transformer model (Vaswani et al., 2017) as our two baselines: Base Sent, which was trained on source and target sentence pairs without context, and Base Doc, which was trained with contexts in the $m$-to- $m$ setting as described in $\S 2.2$. To make a fair comparison with previous works that use similar contextaware translation settings and enhance MT system at the encoder side, we employ the G-Transformer (Bao et al., 2021), Hybrid Context (Zheng et al., 2020) and MultiResolution (Sun et al., 2022). We also compare our approach with the CoDoNMT (Lei et al., 2022) model, which also integrates coreference resolution information to improve translation quality. Note that all aforementioned baselines utilize provided open-source code. Additionally, we trained a simple variant of a context-aware Transformer model similar to Base Doc, but differ in that it incorporated a coreference embedding, alongside the existing positional embedding, directly in to the encoder side of the model (Trans+CEmbedding). This coreference embedding is derived from the original positional embedding in the encoder with the modification that all tokens within a coreference cluster share the same value as the left-most token in the same cluster. Note that it is intended as a simple baseline for a direct model as discussed in $\S 3$.

Our systems We evaluate our proposed inference methods, including the original inference method in Transformer without reranking (Trans+COREF) or with reranking with the score from our submodel (Trans+CorEF+RR) using the coreference resolution task as denoted in Equation 13.

Hardwares All models in our experiments were trained on a machine with the following specifications: an AMD EPYC 7313P CPU, 256GB RAM, a single NVIDIA RTX A6000 with 48GB VRAM, and CUDA version 11.3. For multilingual experiments, we used a single NVIDIA RTX 3090 GPU,

|  | $\mathrm{En}-\mathrm{Ru}$ |  |  | En - De |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | BL $\uparrow$ | $\mathrm{BS} \uparrow$ | $\mathrm{CM} \uparrow$ | BL $\uparrow$ | $\mathrm{BS} \uparrow$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-06.jpg?height=56&width=110&top_left_y=290&top_left_x=1609) |
| Base Sent | 29.46 | -9.695 | $82.87 \quad$ | 22.76 | -6.178 | ![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-06.jpg?height=56&width=110&top_left_y=354&top_left_x=1609) |
| Base Doc | 29.91 | -9.551 | 83.40 | 21.54 | -6.200 | 66.91 |
| Hybrid Context (Zheng et al., 2020) | 29.96 | -9.590 | $83.45 \quad$ | 22.05 | -6.236 | $66.97 \quad$ |
| G-Transformer (Bao e | 30.15 | -9.691 | 83.13 | 22.61 | -6.090 | ![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-06.jpg?height=56&width=110&top_left_y=495&top_left_x=1609) |
| MultiResolution (Sun et al., 2022) | 29.85 | -9.763 | 81.76 | 22.09 | -6.099 | $67.99 \quad$ |
| DoCoNMT (Lei et al., 2022) | 29.92 | -9.552 | 83.03 | 22.55 | -6.197 | $67.93 \quad$ |
| nbedding | 30.13 | -9.522 | $83.43 \quad$ | 22.54 | -6.092 | $68.80 \quad$ |
| Trans+COREF | $30.39^{*} \quad$ | $-9.501^{\dagger}$ | $83.56^{\circ}$  | $23.57^{* *} \quad x \quad y=1$ | $-6.088^{\dagger}$ | $69.17 \quad$ |
| Trans+COREF+RR | $30.43^{*} \quad$ | $-9.500^{\dagger}$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-06.jpg?height=55&width=129&top_left_y=746&top_left_x=1208) | $23.60^{* *} \quad-\quad-x-2$ | $-6.086^{\dagger}$ | $69.21 \quad$ |

${ }^{(*)}$ and ${ }^{(* *)}$ indicate statistical significance (Koehn, 2004) at $p<0.02$ and $p<0.01$, respectively, compared to the Base Doc system and all other baseline systems. ${ }^{(\diamond)},{ }^{(\dagger)}$, and ${ }^{(\bullet)}$ signify statistical significance at $p<0.05$ compared to all baselines, all except Trans+C-Embedding and G-Transformer, and all except Trans+C-Embedding, Hybrid Context, and G-Transformer, respectively.

Table 3: The results of all main experiments. BL, BS and CM are abbreviations for BLEU, BARTScore and COMET, respectively. The best performance per metric are in bold text.

Intel i9-10940X, 48GB VRAM and CUDA version 12.1.

Hyperparameters We use the same parameters, including the number of training epochs, learning rate, batch size, etc., for all models in our experiments. Specifically, we train all models for 40 epochs when both losses of coreference and translation in the valid set show unchanging or no improvements.

For translation tasks, we use the Adam optimizer with $\beta_{1}=0.9, \beta_{2}=0.98$ and $\epsilon=1 e-9$, along with an inverse square root learning rate scheduler. All dropout values are set to 0.1 , and the learning rate is set to $7 e-5$. We use a batch size of 128 and 32 for experiments on the English-Russian and English-German datasets, respectively. Other parameters follow those in Vaswani et al. (2017).

For coreference tasks, we adopt parameters from Kirstain et al. (2021), with some modifications to accommodate our GPU memory. We use the Adam optimizer with $\beta_{1}=0.9, \beta_{2}=0.98$ and $\epsilon=1 e-9$, with a learning rate of $7 e-5$. Dropout value is set to 0.3 , top lambda (the percentage of all spans to keep after filtering) is set to 0.4 , hidden size is set to 512 , and the maximum span length is set to 10 . The maximum cluster values are set to 8 and 20 for the English-Russian and English-German datasets, respectively. To rerank the $\mathrm{N}$-best translations, we use Equation 13 and perform a grid search on the validation set with a step size of 0.0001 to select the optimal value for $\beta$ from -2 to 2 .

### 4.3 Metrics

BLEU We employ SacreBLEU (Post, 2018) as an automated evaluation metric to assess the quality of translations in our experiments.

BARTScore We follow Yuan et al. (2021) and use the mbart-large-50 model (mBART) ${ }^{5}$ to compute the average BARTScore of all translation to measure semantic equivalence and coherence between references and translations. In this metric, the higher value, the better semantic equivalence and coherence.

COMET We also utilize the COMET $^{6}$ metric (Rei et al., 2020), a neural network-based measure, since it is highly correlated to human judgement in prior work by Freitag et al. (2022).

### 4.4 Results

The main results of our experiments are presented in Table 3. Our results indicate that training the baseline Transformer model with both context and target sentences (Base Doc) results in better performance than training with only target sentences (Base Sent) in the En-Ru dataset. This finding is consistent with those reported by Voita et al. (2019), in which more contextual information is helpful to achieve better translation. However, in the En-De dataset, the Base Doc system performs worse com-[^1]

|  | Es $\uparrow$ | $\mathrm{Fr} \uparrow$ | $\mathrm{Ja} \uparrow$ | $\mathrm{Ro} \uparrow$ | $\mathrm{Zh} \uparrow$ | $\mathrm{Vi} \uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base Sent | 37.23 | 37.75 | 12.11 | 24.35 | 12.38 | 31.74 |
| Base Doc | 36.22 | 36.89 | 10.13 | 23.27 | 11.66 | 31.22 |
| G-Transformer | 36.46 | 37.88 | 12.27 | 24.63 | 12.07 | 32.69 |
| Trans+CorEF | $\mathbf{3 8 . 1 3}^{*}$ | $\mathbf{3 9 . 0 1}^{*}$ | $\mathbf{1 2 . 9 3}^{*}$ | $\mathbf{2 5 . 5 6}^{*}$ | $\mathbf{1 3 . 1 8}^{*}$ | $\mathbf{3 3 . 5 1}^{*}$ |

${ }^{*}$ With statistically significant (Koehn, 2004) at $p<0.01$ compared to other systems.

Table 4: The results of multilingual dataset in the BLEU metric. The highest results are in bold text.

pared to the Base Sent system. This discrepancy can be explained by the different methodologies used in constructing the En-De and En-Ru datasets. For the En-De datasets, both context-aware and context-agnostic datasets are compiled from the same pool of non-duplicate sentences. However, for the En-Ru datasets, the context-agnostic dataset is created by removing context sentences from the context-aware dataset (Voita et al., 2019), which results in varying numbers of non-duplicate sentences between these context-agnostic and context-aware datasets.

When comparing our systems with the Transformer model (Base Doc), our approaches, both Trans+CoreF and Trans+CoreF+RR, have proven effective in enhancing translation quality by explaining the decision of translation through predicting coreference information. This is demonstrated by the superior BLEU scores ( +0.52 in En-Ru and +2.06 in En-De for the Trans+COREF+RR), BARTScore and COMET observed when comparing across different settings and language pairs.

Compared to the G-Transformer system described in Bao et al. (2021), our system shows an improvement in both inference approaches (Trans+CoreF and Trans+Coref+RR). In the EnRu dataset, our system achieves a higher BLEU score by +0.24 , while in the En-De dataset, it demonstrates a larger improvement of +1.14 in the same metric (Trans+Coref). Additionally, our method outperforms the G-Transformer in terms of the BARTScore and COMET for both the En$\mathrm{Ru}$ and En-De datasets. One possible explanation for these results is that the G-Transformer is specifically designed to map each sentence in the source language to only a single sentence in the target language during both training and inference steps. This design choice helps mitigating issues related to generating very long sequences. However, when the dataset size is small, as in the case of the En-De dataset, the G-Transformer en- counters difficulties in selecting useful information. In contrast, our approach effectively selects useful information indirectly through the coreference explanation sub-model, especially when dealing with small-sized datasets, which allows our system to outperform under the scenarios with limited dataset size. Our method also surpasses the Transformer model with additional position embedding (Trans+C-Embedding), which relied on coreference information using a direct modeling approach.

In the results of the multilingual TED talk dataset in Table 4, where we compare our proposed method to Transformer models and the best baselines in $\mathrm{Ta}$ ble 3 , our method also surpasses other baselines within +1.0 to +2.3 BLEU scores. These findings provide further evidence that our approach is effective in improving translation quality and can be applied to diverse language types.

We provide an example of translations from our systems as well as other baseline systems in Table 5. In this example, the correct translation of the phrase in the last sentence, моей командой (mу team), is determined by identifying which word refers to "my", in this case, $i$ and $m e$. Both the G-Transformer and Trans+C-Embedding systems fail to capture these mentions and consequently produce an incorrect translation, мою команду. Despite correctly translating моей, the Base Doc system's phrase встретимся в моей команде is grammatically incorrect and deviates from the original English "meet my team". Conversely, our systems capture this reference accurately, yielding a translation consistent with the reference.

## 5 Analysis

Contribution of Coreference Explanation We conducted experiments by adjusting the value of $\alpha$ in Equation 12 during the training of the Trans+Coref without reranking. The experimental results in Table 6 indicate that for medium-sized

| iput | but $\underline{i}$ 'm different. _eos do me just one favor . _eos before you make any <br> decision ... _eos meet my team . |
| :---: | :---: |
| 3ase Doc | но я другой . _еоs сделай мне одолжение . _еоs прежде чем ть примешь <br> решение ... _еоs встретимся в моей команде . |
| $\mathrm{J}-\mathrm{Tra}$ | но я другой . _еоs сделай мне одолжжение . _еоs прежде чем ты примешь <br> решение ... _еоs встреть мою команду . |
| Tran | но я другой . _еоs сделай мне одолжжение . _eos прежде чем принять <br> решение ... _еоs встретить мою команду . |
| Trar | но я другой . _еоs сделай мне одолжение . _еоs прежде чем принять <br> решение ... _еоs познакомься с моей командой . |
| eference | но я изменился . _еоs выполни только одну мою просьбу . _еоs прежде <br> чем ть решишь что-то ... _еоs познакомься с моей командой . |

Table 5: Example of translations. The context and target sentences are highlighted in italics and bold, respectively. Translations of the Trans+DEC+RR and Trans+DEC are identical. Underline words indicate the same mention entity.

|  | $\alpha$ | BLEU $\uparrow$ |
| :--- | :---: | :---: |
| Base Sent | - | 29.46 |
| Base Doc | - | 29.91 |
|  | 0.8 | 30.36 |
|  | 1.0 | 30.31 |
| Trans+COREF | 2.0 | $\mathbf{3 0 . 3 9}$ |
|  | 3.0 | 30.27 |
|  | 4.0 | 30.15 |
|  | 10.0 | 30.00 |

Table 6: Ablation results on the En-Ru dataset with different weight $\alpha$. The highest result is in bold text.

corpora, selecting a value of $\alpha$ that is either too small or too large negatively impacts translation quality. The optimal range for $\alpha$ is $0.8 \leq \alpha \leq 2$.

## Conditioning on Source and Target language

 We conducted a study on coreference explanation on the En-De dataset (Maruf et al., 2019) with coreference cluster information as in $\S 4.1$ by ablating the information from the translation so that it conditions only on the input information (Trans+ENC). This setting can be regarded as a conventional multi-task setting in which coreference on the input-side is predicted together with its translation. Specifically, we replace the input representation for coreference resolution sub-model from $\operatorname{DEC}(\cdot)$ in Equation 11 to $\operatorname{ENC}(\cdot)$ in Equation 14 as follows$$
\begin{equation*}
\mathbf{H}_{\text {coref }}^{\prime \prime}=\operatorname{ENC}\left(\boldsymbol{H}_{e n c}^{l_{e}}\right) \tag{14}
\end{equation*}
$$

As shown in Table 7, the conventional multi-task learning setting of Trans+ENC performed lower than Trans+COREF, which indicates the benefits

|  | BLEU $\uparrow$ | $\mathrm{P} \uparrow$ | $\mathrm{R} \uparrow$ | $\mathrm{F} 1 \uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Trans+ENC | 22.98 | $\mathbf{8 5 . 0 2}$ | 75.69 | 80.08 |
| Trans+CoREF | $\mathbf{2 3 . 5 7}$ | 82.63 | $\mathbf{7 8 . 3 1}$ | $\mathbf{8 0 . 4 1}$ |

Table 7: Evaluation of Trans+Enc and Trans+CoreF systems using BLEU and MUC* metrics on the validating set of the En-De dataset. The highest results are in bold text.

* The MUC metric counts the changes required to align the system's entity groupings with the gold-standard, focusing on adjustments to individual references.

of fusing information from the predicted translation. We further examine the entity heat maps derived from self-attention weights of both the translation and coreference sub-models in Base Doc, Trans+EnC, and Trans+CoreF systems for the input "I_o hate that you_o 're leaving. Well, Grandma 's not doing well. So you_1 have to drop everything to go take care of her? Yes, William , I_l do ." from the human annotated test set from Voita et al. (2019). In this particular example, the coreference clusters are defined as [I_0, William], [you_0, you_1, I_ı], and [Grandma, her]. To provide visual representations, we depict the average self-attention values from the last encoder layer of these three systems. This choice is based on their tendency to focus more on semantic or abstract information (Clark et al., 2019).

Figure 1 displays the entity heat maps, which illustrate the behavior of self-attention in different systems in the translation sub-model. In the Base Doc system, self-attention primarily concentrates on local sentences while disregarding information

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-09.jpg?height=352&width=1088&top_left_y=201&top_left_x=495)

Figure 1: Entity heat maps of self-attentions: (a) Base Doc, (b) Trans+ENC and (c) Trans+CoreF.

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-09.jpg?height=322&width=754&top_left_y=707&top_left_x=251)

Figure 2: Entity heat maps of self-attentions in the coreference resolution sub-model.

between sentences. In contrast, the Trans+ENC system exhibits the ability to focus on inter-sentences. However, when it comes to tokens within coreference clusters, the focused values are incorrect for certain clusters, such as [I_0, William]. On the other hand, the Trans+CorEF system not only exhibits inter-sentential focus in its self-attention heat map but also accurately depicts the focused values for tokens within coreference clusters.

Figure 2 demonstrates the entity heat maps in the coreference sub-model. In the Trans+ENC system, self-attention mainly concentrates on entities within the local scope and immediate adjacent sentences. However, when comparing these high attention values with links in the coreference clusters, a significant proportion is found to be incorrect, i.e., [Grandma; I_1]. On the other hand, the selfattention in Trans+COREF exhibits a more balanced distribution of focus across all entities within the input. This balanced distribution results in considerably fewer errors when compared to self-attention in the Trans+Enc system. These findings align with the MUC metric (Vilain et al., 1995), which is based on the minimum number of missing links in the response entities compared to the key entities, with details, particularly the F1 score, provided in Table 7. Note that we use reference translations to form $\boldsymbol{H}_{d e c}^{l_{d}}$ in Equation 11 for identifying coreference clusters. Additionally, we generate gold label coreference clusters using the AllenNLP frame-

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-09.jpg?height=442&width=671&top_left_y=727&top_left_x=1109)

Figure 3: Translation results on En-De datasets with different $m$-to- $m$ translation settings from $m=2$ to $m=4$. The result in the $m=1$ setting serves as the Base Sent reference. The $\alpha$ in Equation 12 is set to 4.0.

|  | $\mathbf{D} \uparrow$ | $\mathbf{E I} \uparrow$ | $\mathbf{E V} \uparrow$ | $\mathbf{L} \uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Base Doc | 83.32 | 70.20 | 62.20 | 46.0 |
| Trans+COREF | $\mathbf{8 5 . 6 4}$ | $\mathbf{7 1 . 2 0}$ | $\mathbf{6 5 . 2}$ | $\mathbf{4 6 . 4}$ |

Table 8: Experimental results on the contrastive test (Voita et al., 2019). D, EI, EV and L are abbreviations for Deixis, Ellipsis Infl, Ellipsis Vp and Lexical Cohesion, respectively. Note that we only utilized the text described in $\S 4.1$, while other studies may incorporate additional sentence-level bilingual and monolingual texts associated with Voita et al. (2019).

work, as discussed in Section 4.1.

Impact of the context size We conducted experiments with the COREF (Trans+COREF) and the Transformer (Base Doc) systems by exploring different context sizes in $m$-to- $m$ settings ranging from 2 to 4 . The experimental results in Figure 3 demonstrate that the Base Doc system significantly drops the translation quality when the context gets longer, while Trans+COREF consistently achieves gains as we incorporate more contexts. This result also indicates the use of the coreference sub-model is able to capture contextual information better than the baseline.

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-10.jpg?height=483&width=694&top_left_y=204&top_left_x=241)

Figure 4: The results with N-best variants on the En-Ru dataset (Voita et al., 2019).

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-10.jpg?height=488&width=677&top_left_y=841&top_left_x=244)

Figure 5: The results with N-best variants using the oracle BLEU metric on the En-Ru dataset (Voita et al., 2019).

Impact of Coreference Explanation We conduct experiments by reranking all translation hypotheses with varying beam sizes during inference by the Equation 13 to assess the impact of coreference explanation sub-model on the En-Ru dataset (Voita et al., 2019). Figure 4 illustrates the results of our experiments measured by BLEU score. Our findings indicate that reranking with the sub-model COREF yields improved results, with differences ranging from 0.02 to 0.09 . We also report oracle BLEU score in Figure 5, which is measured by selecting a hypothesis sentence that gives the maximum sentence-BLEU scores among potential hypotheses, to verify the potentially correct translations in an $N$-best list. The results of this experiment with differences ranging from 0.2 to 0.4 suggest that using the sub-model COREF has more potential to generate correct translations. Despite the relatively minor difference in the oracle BLEU score between the Trans+Coref and the Base Doc systems, indicating a similarity in their candidate space, the beam search process yields better results with the Trans+Coref when compared with the Base Doc system. This reflects the differences in BLEU scores between Trans+CoreF and Base Doc. The performance gap in the BLEU score between the Trans+CoreF and Trans+Coref+RR could potentially be further maximized by incorporating the coreference resolution during the beam search at the expense of more computational costs. We intend to explore this possibility in our future research endeavors.

To further understand the impact of the coreference explanation sub-model on translation results, we perform an experiment on the contrastive test in Voita et al. (2019), which contains human-labeled sentences to evaluate discourse phenomena and relies on the source text only, to verify whether our method can solve phenomena at the document level. Table 8 presents the results this experiment, which indicate that our system outperforms the Base Doc system in all aspects. These results demonstrate the significant contribution of the coreference explanation sub-model to the MT system.

Impact of Coreference Accuracy We carried out experiments to assess the impact of varying accuracies within the external coreference framework, which was reported in $80.4 \%$ of the F1 score on MUC metric for English CoNLL-2012 shared task in Lee et al. (2018), on the overall translation quality. This was achieved by randomly omitting members from coreference clusters while ensuring that each valid cluster retained a minimum of two members, i.e., removing you_l from the cluster $\left[y o u \_0\right.$, you_ $\left.1, I_{-}\right]$in Figure 1.

Table 9 presents the outcomes of these experiments, where a slight reduction in translation quality is observed as members of coreference clusters are randomly dropped. Remarkably, even with the omission of up to half of the cluster members, the results continue to exceed the performance of the Base Doc system. This implies that our method could be robust and effective, particularly for languages with limited accuracy in coreference resolution tasks.

Impact of the corpus size We randomly sampled training instances from the En-Ru dataset and varied the sample sizes to 200,000 (comparable size to the En-De dataset), 500,000, and 1,000,000. Subsequently, we evaluated the contribution of the CorEF sub-model (Trans+CoreF) and the Transformer (Base Doc) on these datasets of different sample sizes. Figure 6 illustrates the results of

|  | Pruning (\%) | BLEU $\uparrow$ |  |
| :--- | :---: | :---: | :---: |
|  |  | $+\mathrm{RR}$ |  |
| Base Doc | - | 21.54 | - |
|  | 0 | 23.57 | 23.60 |
| Trans+CorEF | 10 | 23.43 | 23.44 |
|  | 20 | 23.40 | 23.41 |
|  | 30 | 23.29 | 23.29 |
|  | 50 | 22.86 | 22.86 |

Table 9: Experimental results on dropping coreference clusters on the En-De dataset. RR means reranking with the coreference sub-model using Equation 13.

|  | Accuracy (\%) $\uparrow$ |
| :--- | :---: |
| Base Doc | 12.71 |
| G-Transformer | 14.45 |
| Trans+CorEF | 18.50 |

Table 10: Accuracy of translating the word we into Vietnamese ( 173 samples).

these experiments. Our proposed system outperforms the Transformer model (Base Doc) across all sample sizes in the test set. Notably, this improvement is not limited to the small dataset size setting but similar trends are observed for medium-sized datasets. These results indicate that our system consistently outperforms the transformer model and achieves improved translation qualities regardless of the dataset size.

Remaining Challenges and Unresolved Questions While our proposed method and existing works enhance translation accuracy for certain linguistic phenomena, challenges persist, particularly in handling deixis. Unlike straightforward scenarios where additional context aids in accurately translating deictic terms (e.g., determining the speakers in a conversation to correctly translate the words $I$ and You), some instances require a comprehensive understanding of the provided text's content to achieve correct pronoun translation. Consider the following example from the test data of the English-Vietnamese dataset (Qi et al., 2018): "Oh my god! you're right!' who can we [chúng ta] sue? Now Chris is a really brilliant lawyer, but he knew almost nothing about patent law and certainly nothing about genetics. I knew something about genetics, but I wasn't even a lawyer, let alone a patent lawyer. So clearly we [chúng tôi] had a lot to learn before we [chúng ta] could file a lawsuit." In this con-

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-11.jpg?height=491&width=579&top_left_y=203&top_left_x=1047)

Figure 6: Translation results on the En-Ru dataset (Voita et al., 2019) with different sample sizes.

text, the English word we is translated as either

![](https://cdn.mathpix.com/cropped/2024_06_04_7cd2849d4b45a1bcdfe9g-11.jpg?height=54&width=780&top_left_y=927&top_left_x=1049)
reflecting the exclusion or inclusion of the listener. This example underscores the importance of contextual nuances in translating pronouns like we or us from English to Vietnamese, where the choice between chúng tôi and chúng ta is critical.

Building on the insights from the described example, we extracted all samples that presented similar linguistic challenges, in which a correctly translated sample must ensure that every instance of the word we is accurately translated. Table 10 presents the accuracy of translating the word we into the correct Vietnamese. While our method surpasses other baseline models in performance, it still exhibits lower accuracy in comparison to the deixis-related outcomes of the contrastive test for Russian (Voita et al., 2019). This discrepancy highlights the phenomenon as a significant challenge that warrants further investigation.

Computational Cost We present a detailed comparison of the parameter count and training time per epoch for our proposed method alongside other baselines in Table 11. When compared to the GTransformer, our method uses fewer parameters, takes less time to train, and yet achieves better performance. On the other hand, the Base Doc system uses the fewest parameters and trains the quickest, but its results are notably underperforming.

## 6 Related Works

Multi-task learning has primarily been utilized in MT tasks to integrate external knowledge into MT systems. Luong et al. (2016); Niehues and Cho (2017); Eriguchi et al. (2017) have employed multi-task learning with different variations of shared weights of encoders, decoders, or attentions

|  | No. of <br> Params | Training <br> Time | En-De $\uparrow$ |
| :--- | :---: | :---: | :---: |
| Base Doc | 92.03 | 407 | 21.54 |
| MultiResolution | 92.03 | 610 | 22.09 |
| G-Transformer | 101.48 | 566 | 22.61 |
| Hybrid Context | 65.78 | 1,776 | 22.05 |
| CoDoNMT | 92.03 | 638 | 22.55 |
| Trans+CorEF | 98.59 | 503 | 23.57 |

Table 11: Number of parameters (in million), training time for one epoch (in seconds) and results of systems (in the BLEU metric) on the En-De dataset.

between tasks to effectively incorporate parsing knowledge into sequence-to-sequence MT systems.

For incorporating coreference cluster information, Ohtani et al. (2019), Xu et al. (2021) and Lei et al. (2022) incorporate coreference cluster information to improve their NMT models. Ohtani et al. (2019) integrates coreference cluster information into a graph-based NMT approach to enhance the information. Similarly, Xu et al. (2021) uses the information to connect words across different sentences and incorporates other parsing information to construct a graph at the document-level, resulting in an improvement in translation quality. Lei et al. (2022) employs coreference information to construct cohesion maskings and fune-tunes sentence MT systems to produce more cohesive outputs. On the other hand, Stojanovski and Fraser (2018); Hwang et al. (2021) leverage coreference cluster information through augmented steps. They either add noise to construct a coreference-augmented dataset or use coreference information to create a contrastive dataset and train their MT systems on these enhanced datasets to achieve better translation performance. For context-aware MT, Kuang et al. (2018) and Tu et al. (2018) focus on utilizing memory-augmented neural networks, which store and retrieve previously translated parts in NMT systems. These approaches help unify the translation of objects, names, and other elements across different sentences in a paragraph. In contrast, Xiong et al. (2019); Voita et al. (2019) develop a multiplepass decoding method inspired by the Deliberation Network (Xia et al., 2017) to address coherence issues, i.e., deixis and ellipsis in paragraphs. They first translate the source sentences in the first pass and then correct the translations to improve coherence in the second pass. Mansimov et al. (2020) introduce a self-training technique, similar to domain self-adaptation, to develop a document-level
NMT system. Meanwhile, various methods aim to encapsulate contextual information, i.e., hierachical attention (Maruf et al., 2019), multiple-attention mechanism (Zhang et al., 2020; Bao et al., 2021), recurrent memory unit (Feng et al., 2022) ${ }^{7}$. In a data augmentation approach, Bao et al. (2023) diversify training data for the target side language, rather than only using a single human translation for each source document.

Recently, Wang et al. (2023) has shown that state-of-the-art Large Language Models (LLMs), i.e. GPT-4 (OpenAI et al., 2024), outperform traditional translation models in context-aware MT. In other approaches, Wu et al. (2024) and Li et al. (2024) have developed effective fine-tuning and translation methods for lightweight LLMs; however, the efficacy of NMT models can exceed that of lightweight LLMs, varying by language pair.

## 7 Conclusion

This study presents a context-aware MT model that explains the translation output by predicting coreference clusters in the source side. The model comprises two sub-models, a translation sub-model and a coreference resolution sub-model, with no modifications to the translation model. The coreference resolution sub-model predicts coreference clusters by fusing the representation from both the encoder and decoder to capture relations in the two languages explicitly. Under the same settings of the En-Ru, En-De and the multilingual datasets, and following analyses on the coreference sub-model's contributions, the impacts of context and corpus size, as well as the type of information utilized in the sub-model, our proposed method has proven effective in enhancing translation quality.

## Limitations

In this study, the hidden dimension size in the coreference resolution sub-model is smaller than typical state-of-the-art systems, i.e., 512 vs. 2048, potentially limiting its accuracy and negatively impacting the quality of translation. Additionally, this study requires fine-tuning for a certain hyperparameter that combines the coreference resolution sub-model and the translation model to achieve satisfactory results.[^2]

## Acknowledgements

The authors are grateful to the anonymous reviewers and the Action Editor who provided many insightful comments that improve the paper. This work was supported by JSPS KAKENHI Grant Number JP21H05054.

## References

Chinatsu Aone and Scott William. 1995. Evaluating automated and manual acquisition of anaphora resolution strategies. In 33rd Annual Meeting of the Association for Computational Linguistics, pages 122-129, Cambridge, Massachusetts, USA. Association for Computational Linguistics.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Guangsheng Bao, Zhiyang Teng, and Yue Zhang. 2023. Target-side augmentation for documentlevel machine translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10725-10742, Toronto, Canada. Association for Computational Linguistics.

Guangsheng Bao, Yue Zhang, Zhiyang Teng, Boxing Chen, and Weihua Luo. 2021. G-transformer for document-level machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 3442-3455. Association for Computational Linguistics.

Peter F. Brown, John Cocke, Stephen Della Pietra, Vincent J. Della Pietra, Frederick Jelinek, John D. Lafferty, Robert L. Mercer, and Paul S. Roossin. 1990. A statistical approach to machine translation. Comput. Linguistics, 16(2):7985 .

Emanuele Bugliarello and Naoaki Okazaki. 2020. Enhancing machine translation with dependencyaware self-attention. In Proceedings of the 58th
Annual Meeting of the Association for Computational Linguistics, pages 1618-1627, Online. Association for Computational Linguistics.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. 2019. What does BERT look at? an analysis of BERT's attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 276-286, Florence, Italy. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171-4186. Association for Computational Linguistics.

Akiko Eriguchi, Yoshimasa Tsuruoka, and Kyunghyun Cho. 2017. Learning to parse and translate improves neural machine translation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 72-78, Vancouver, Canada. Association for Computational Linguistics.

Yukun Feng, Feng Li, Ziang Song, Boyuan Zheng, and Philipp Koehn. 2022. Learn to remember: Transformer with recurrent memory for document-level machine translation. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1409-1420, Seattle, United States. Association for Computational Linguistics.

Markus Freitag, Ricardo Rei, Nitika Mathur, Chikiu Lo, Craig Stewart, Eleftherios Avramidis, Tom Kocmi, George Foster, Alon Lavie, and André F. T. Martins. 2022. Results of WMT22 metrics shared task: Stop using BLEU - neural metrics are better and more robust. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 46-68, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. Con-
volutional sequence to sequence learning. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 1243-1252. PMLR.

Yongkeun Hwang, Hyeongu Yun, and Kyomin Jung. 2021. Contrastive learning for contextaware neural machine translation using coreference information. In Proceedings of the Sixth Conference on Machine Translation, WMT@EMNLP 2021, Online Event, November 10-11, 2021, pages 1135-1144. Association for Computational Linguistics.

Sébastien Jean, Stanislas Lauly, Orhan Firat, and Kyunghyun Cho. 2017. Does neural machine translation benefit from larger context? CoRR, abs/1704.05135v1. Version 1.

Yuval Kirstain, Ori Ram, and Omer Levy. 2021. Coreference resolution without span representations. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 14-19. Association for Computational Linguistics.

Dan Klein and Christopher D. Manning. 2002. Conditional structure versus conditional estimation in NLP models. In Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing, EMNLP 2002, Philadelphia, PA, USA, July 6-7, 2002, pages 9-16.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, pages 388395, Barcelona, Spain. Association for Computational Linguistics.

Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics, HLT-NAACL 2003, Edmonton, Canada, May 27 - June 1, 2003. The Association for Computational Linguistics.

Shaohui Kuang, Deyi Xiong, Weihua Luo, and Guodong Zhou. 2018. Modeling coherence for neural machine translation with dynamic and topic caches. In Proceedings of the 27th International Conference on Computational Linguistics, pages 596-606, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017. End-to-end neural coreference resolution. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 188-197, Copenhagen, Denmark. Association for Computational Linguistics.

Kenton Lee, Luheng He, and Luke Zettlemoyer. 2018. Higher-order coreference resolution with coarse-to-fine inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 687-692, New Orleans, Louisiana. Association for Computational Linguistics.

Yikun Lei, Yuqi Ren, and Deyi Xiong. 2022. CoDoNMT: Modeling cohesion devices for document-level neural machine translation. In Proceedings of the 29th International Conference on Computational Linguistics, pages 52055216, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Yachao Li, Junhui Li, Jing Jiang, and Min Zhang. 2024. Enhancing document-level translation of large language model via translation mixed-instructions. arXiv preprint arXiv:2401.08088v1. Version 1.

Pierre Lison, Jörg Tiedemann, and Milen Kouylekov. 2018. OpenSubtitles2018: Statistical rescoring of sentence alignments in large, noisy parallel corpora. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. 2016. Multitask sequence to sequence learning. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.

Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective approaches to attentionbased neural machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1412-1421, Lisbon, Portugal. Association for Computational Linguistics.

Elman Mansimov, Gábor Melis, and Lei Yu. 2020. Capturing document context inside sentencelevel neural machine translation models with self-training. CoRR, abs/2003.05259v1. Version 1

Sameen Maruf, André F. T. Martins, and Gholamreza Haffari. 2019. Selective attention for context-aware neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3092-3102, Minneapolis, Minnesota. Association for Computational Linguistics.

Joseph F. McCarthy and Wendy G. Lehnert. 1995. Using decision trees for coreference resolution. In Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence, IJCAI 95, Montréal Québec, Canada, August 2025 1995, 2 Volumes, pages 1050-1055. Morgan Kaufmann.

Jan Niehues and Eunah Cho. 2017. Exploiting linguistic resources for neural machine translation using multi-task learning. In Proceedings of the Second Conference on Machine Translation, pages 80-89, Copenhagen, Denmark. Association for Computational Linguistics.

Takumi Ohtani, Hidetaka Kamigaito, Masaaki Nagata, and Manabu Okumura. 2019. Contextaware neural machine translation with coreference information. In Proceedings of the Fourth Workshop on Discourse in Machine Translation (DiscoMT 2019), pages 45-50, Hong Kong, China. Association for Computational Linguistics.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming
Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Ra-
jeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. Gpt-4 technical report. arXiv preprint arXiv:2303.08774v6. Version 6.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186-191, Belgium, Brussels. Association for Computational Linguistics.

Ye Qi, Devendra Sachan, Matthieu Felix, Sarguna Padmanabhan, and Graham Neubig. 2018. When and why are pre-trained word embeddings useful for neural machine translation? In Proceedings of the 2018 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 529535, New Orleans, Louisiana. Association for Computational Linguistics.

Maria Refinetti, Alessandro Ingrosso, and Sebastian Goldt. 2023. Neural networks trained with SGD learn distributions of increasing complexity. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 28843-28863. PMLR.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 26852702, Online. Association for Computational Linguistics.

Rico Sennrich and Barry Haddow. 2016. Linguistic input features improve neural machine translation. In Proceedings of the First Conference on Machine Translation: Volume 1, Research Papers, pages 83-91, Berlin, Germany. Association for Computational Linguistics.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715-1725, Berlin, Germany. Association for Computational Linguistics.

Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. 2020. The pitfalls of simplicity bias in neural networks. In Advances in Neural Information Processing Systems, volume 33, pages 95739585. Curran Associates, Inc.

Dario Stojanovski and Alexander Fraser. 2018. Coreference and coherence in neural machine translation: A study using oracle experiments. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 4960, Brussels, Belgium. Association for Computational Linguistics.

Zewei Sun, Mingxuan Wang, Hao Zhou, Chengqi Zhao, Shujian Huang, Jiajun Chen, and Lei Li.

2022. Rethinking document-level neural machine translation. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3537-3548, Dublin, Ireland. Association for Computational Linguistics.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pages 3104-3112.

Jörg Tiedemann and Yves Scherrer. 2017. Neural machine translation with extended context. In Proceedings of the Third Workshop on Discourse in Machine Translation, pages 82-92, Copenhagen, Denmark. Association for Computational Linguistics.

Zhaopeng Tu, Yang Liu, Shuming Shi, and Tong Zhang. 2018. Learning to remember translation history with a continuous cache. Transactions of the Association for Computational Linguistics, 6:407-420.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998-6008.

Marc B. Vilain, John D. Burger, John S. Aberdeen, Dennis Connolly, and Lynette Hirschman. 1995. A model-theoretic coreference scoring scheme. In Proceedings of the 6th Conference on Message Understanding, MUC 1995, Columbia, Maryland, USA, November 6-8, 1995, pages 4552. ACL.

Elena Voita, Rico Sennrich, and Ivan Titov. 2019. When a good translation is wrong in context: Context-aware machine translation improves on deixis, ellipsis, and lexical cohesion. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 11981212, Florence, Italy. Association for Computational Linguistics.

Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. 2018. Context-aware neural machine translation learns anaphora resolution. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1264-1274, Melbourne, Australia. Association for Computational Linguistics.

Thanh Vu, Dat Quoc Nguyen, Dai Quoc Nguyen, Mark Dras, and Mark Johnson. 2018. VnCoreNLP: A Vietnamese natural language processing toolkit. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics. Demonstrations, pages 56-60, New Orleans, Louisiana. Association for Computational Linguistics.

Longyue Wang, Chenyang Lyu, Tianbo Ji, Zhirui Zhang, Dian Yu, Shuming Shi, and Zhaopeng Tu. 2023. Document-level machine translation with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16646-16661, Singapore. Association for Computational Linguistics.

Longyue Wang, Zhaopeng Tu, Andy Way, and Qun Liu. 2017. Exploiting cross-sentence context for neural machine translation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 28262831, Copenhagen, Denmark. Association for Computational Linguistics.

Minghao Wu, Thuy-Trang Vu, Lizhen Qu, George Foster, and Gholamreza Haffari. 2024. Adapting large language models for documentlevel machine translation. arXiv preprint arXiv:2401.06468v2. Version 2.

Yingce Xia, Fei Tian, Lijun Wu, Jianxin Lin, Tao Qin, Nenghai Yu, and Tie-Yan Liu. 2017. Deliberation networks: Sequence generation beyond one-pass decoding. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Hao Xiong, Zhongjun He, Hua Wu, and Haifeng Wang. 2019. Modeling coherence for discourse neural machine translation. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019,

The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 7338-7345. AAAI Press.

Mingzhou Xu, Liangyou Li, Derek F. Wong, Qun Liu, and Lidia S. Chao. 2021. Document graph for neural machine translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8435-8448, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kyra Yee, Yann Dauphin, and Michael Auli. 2019. Simple and effective noisy channel modeling for neural machine translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5696-5701, Hong Kong, China. Association for Computational Linguistics.

Lei Yu, Phil Blunsom, Chris Dyer, Edward Grefenstette, and Tomás Kociský. 2017. The neural noisy channel. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021. Bartscore: Evaluating generated text as text generation. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 27263-27277.

Pei Zhang, Boxing Chen, Niyu Ge, and Kai Fan. 2020. Long-short term masking transformer: A simple but effective baseline for document-level neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 10811087. Association for Computational Linguistics.

Boyuan Zheng, Patrick Xia, Mahsa Yarmohammadi, and Benjamin Van Durme. 2023. Multilingual coreference resolution in multiparty dialogue. Transactions of the Association for Computational Linguistics, 11:922-940.
Zaixiang Zheng, Xiang Yue, Shujian Huang, Jiajun Chen, and Alexandra Birch. 2020. Towards making the most of context in neural machine translation. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020, pages 3983-3989. ijcai.org.


[^0]:    ${ }^{1}$ https://taku910.github.io/mecab/

    ${ }^{2}$ https://github.com/fxsjy/jieba

    ${ }^{3}$ https://spacy.io

    ${ }^{4}$ https://allenai.org

[^1]:    ${ }^{5}$ https://huggingface.co/facebook/ mbart-large-50

    ${ }^{6}$ COMET-20 model $(w m t 20-C O M E T-d a)$

[^2]:    ${ }^{7}$ In Feng et al. (2022), they provided source code without instructions. We tried to reuse and reimplement their method however, we can not reproduce their results in any efforts. They did not reply our emails for asking training details. We therefore decide not to include their results in Table 3.

</end of paper 1>


<paper 2>
# (PERHAPs) BEYOND Human TranSlation: HarNESSING MULTI-AGENT COLLABORATION FOR TRANSLATING ULTRA-LONG LITERARY TEXTS 

Minghao Wu ${ }^{1}$, Yulin Yuan ${ }^{2}$, Gholamreza Haffari ${ }^{1}$, Longyue Wang ${ }^{3 *}$<br>${ }^{1}$ Monash University $\quad{ }^{2}$ University of Macau ${ }^{3}$ Tencent AI Lab


#### Abstract

Recent advancements in machine translation (MT) have significantly enhanced translation quality across various domains. However, the translation of literary texts remains a formidable challenge due to their complex language, figurative expressions, and cultural nuances. In this work, we introduce a novel multi-agent framework based on large language models (LLMs) for literary translation, implemented as a company called TRANSAGENTS, which mirrors traditional translation publication process by leveraging the collective capabilities of multiple agents, to address the intricate demands of translating literary works. To evaluate the effectiveness of our system, we propose two innovative evaluation strategies: Monolingual Human Preference (MHP) and Bilingual LLM Preference (BLP). MHP assesses translations from the perspective of monolingual readers of the target language, while BLP uses advanced LLMs to compare translations directly with the original texts. Empirical findings indicate that despite lower $d$-BLEU scores, translations from TRANSAGENTS are preferred by both human evaluators and LLMs over human-written references, particularly in genres requiring domain-specific knowledge. We also highlight the strengths and limitations of TRANSAGENTS through case studies and suggests directions for future research.


![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-01.jpg?height=648&width=1326&top_left_y=1614&top_left_x=386)

Traditional MT

Human Translator
Machine Translator
Our Method

Human Translator

TransAgents

Figure 1: An illustration of our method. Traditional machine translation (MT) systems often underperform compared to human translators. In this study, we demonstrate that the translations produced by our TRANSAGENTS are more preferred by humans than those from conventional MT systems.[^0]

## 1 INTRODUCTION

Machine translation (MT) has achieved remarkable advancements in recent years, driven by breakthroughs in deep learning and neural networks (Cho et al., 2014; Sutskever et al. 2014; Vaswani et al. 2017, Gu et al. 2019b, Liu et al., 2020, Fan et al., 2021). Despite these technological strides, literary translation remains an unresolved challenge for MT systems. Literary texts, characterized by their complex language, figurative expressions, cultural nuances, and unique stylistic elements, pose significant hurdles that are hard for machines to overcome (Voigt \& Jurafsky, 2012). This complexity makes literary translation one of the most challenging areas within machine translation, often referred to as "the last frontier of machine translation" (Klemin, 2024).

In response to complex challenges across various domains, recent research in multi-agent systems, particularly those powered by large language models (LLMs), has shown significant promise (Yao et al., 2023, Wang et al., 2023e, Dong et al. 2023). These systems leverage the collective intelligence of multiple agents, enabling superior problem-solving capabilities compared to individual model approaches. Multi-agent systems excel in dynamic environments where intricate problem-solving and collaborative efforts are required.

Given the nature of literary translation, we harness the superior capabilities of multi-agent systems and establish a novel multi-agent translation company for literary translation, called TransAGEnTS. At TransAGEnTS, the translation process is organized into two main stages, each consisting of several sub-stages. The process begins with the selection of a Senior Editor by our pre-defined CEO agent, who chooses based on the specific requirements of each client. The selected Senior Editor then assembles a team from our roster, which includes roles such as Junior Editor, Translator, Localization Specialist, and Proofreader. Each team member collaborates through multiple sub-stages, employing strategies like Addition-by-Subtraction Collaboration and Trilateral Collaboration to refine and enhance the translation output.

Furthermore, evaluating the accuracy and quality of literary translations presents a particularly challenging task due to the subjective nature of literature and the potential imperfections in reference translations (Thai et al., 2022; Freitag et al., 2023). To effectively address these challenges, we propose two innovative evaluation strategies: Monolingual Human Preference (MHP) and Bilingual LLM Preference (BLP). Both strategies involve comparing a pair of translations from two different translation systems to determine which one is superior. The Monolingual Human Preference strategy simulates the realistic scenario of reading a translated work. It engages human evaluators from the target audience who assess translations without the influence of the original text. This approach focuses on how well the translation resonates with the readers in terms of fluidity, readability, and cultural appropriateness, mirroring the real-world consumption of literature. Conversely, the Bilingual LLM Preference leverages the capabilities of advanced LLMs, specifically GPT-4-0125-PREVIEW. In this strategy, the LLMs are provided with the original texts to facilitate a direct comparison. This method aims to harness the superior translation capabilities of advanced LLMs, mitigating the impact of imperfect reference translations.

Our empirical findings reveal that TRANSAgEnTs consistently delivers the poorest performance in terms of $d$-BLEU scores. However, it is preferred over both human-written references and GPT-4 translations by human evaluators and an LLM evaluator. In-depth analysis shows that TRANSAGENTS excels over human-written references in genres that demand domain-specific knowledge, such as historical contexts and cultural nuances, but it falls short in contemporary genres. Additionally, we observe that TransAGEnTS is capable of generating translations with more diverse and vivid descriptions. Our cost analysis indicates that using TransAgents for literary text translation can result in an $80 \times$ reduction in costs compared to employing professional human translators. Nonetheless, we also identify significant limitations in LLM-based translation systems, including both GPT-4 and TRANSAGENTS, particularly with issues related to significant content omission

In this work, our contributions can be summarized as follows:

- We introduces TransAgents, a novel multi-agent system for literary translation, which mirrors the traditional translation publication process. By employing a multi-agent approach, this approach addresses the complex nuances of literary works.
- We propose two novel evaluation strategies, Monolingual Human Preference (MHP) and Bilingual LLM Preference (BLP) to assess the quality of translations. MHP focuses on the translation's impact on target audience readers, emphasizing fluidity and cultural appropriateness, while BLP uses advanced LLMs to compare translations directly with the original texts.
- Despite lower $d$-BLEU scores, our empirical findings highlight that translations from TRANSAGENTS are preferred by both human evaluators and language models over humanwritten references. We also present in-depth analyses about the strengths and weaknesses of TRANSAGENTS.


## 2 RELATED WORK

Large Language Models Large language models (LLMs) have revolutionized the field of artificial intelligence (AI). These models are typically pretrained on a vast corpus of text data, learning to predict the next word in a sentence (Brown et al., 2020; Chowdhery et al., 2022; Scao et al., 2022; Anil et al., 2023b, Touvron et al., 2023a b; Bai et al., 2023a; Anil et al., 2023a). After pretraining, the models are fine-tuned with instructions. This process, known as supervised fine-tuning (SFT) or instruction tuning (IT), allows the model to adapt its general language understanding to follow and implement instructions from humans (Sanh et al., 2022; Wei et al., 2022, Chung et al., 2022, Wang et al., 2022; Tay et al., 2023, Longpre et al., 2023; Shen et al., 2023). Thanks to the superior capabilities of large language models, recent works demonstrate that synthetic datasets generated by these models can also be used in this step (Wang et al., 2023c, Wu et al. 2023b, Li et al. 2023a; Luo et al., 2023; Lyu et al., 2023, Yue et al. 2023; Wang et al., 2023d). Furthermore, reinforcement learning from human feedback (RLHF) is used to further improve the performance of these models. In this approach, the model is fine-tuned based on feedback from humans or other large language models, who rate the quality of the model's outputs (Ouyang et al. 2022, Rafailov et al., 2023, Hejna et al. 2023; Ethayarajh et al., 2024, Hong et al., 2024). Moreover, evaluating these large language models is a complex task, often involving both automated metrics and human judgment (Hendrycks et al., 2021; Liang et al., 2022; Wu \& Aji, 2023; Jiang et al., 2023; Lyu et al., 2024). Additionally, these models pose challenges in terms of efficient training (Hu et al., 2022; Dettmers et al., 2023, Liu et al. 2024), fairness (Li et al., 2023c), hallucination (Zhang et al., 2023c), and other issues, which are also active areas of research. In this work, we leverage the state-of-the-art LLM as the backbone of our multi-agent system for translating the literary texts.

Multi-Agent Systems Intelligent agents are designed to understand their environments, make informed decisions, and respond with appropriate actions (Wooldridge \& Jennings, 1995). The capabilities of large language models (LLMs) align well with these expectations. The emergence of LLMs has significantly advanced research on multi-agent systems across various contexts. Multiagent systems, compared to single-agent setups, are generally expected to either leverage collaboration among multiple agents to tackle complex problems or use diverse agents to effectively simulate complex real-world environments (Guo et al. 2024). Recent studies have shown promising outcomes in complex problem-solving areas such as software development (Qian et al. 2023, Hong et al. 2023), multi-robot collaboration (Mandi et al., 2023, Zhang et al., 2023a), evaluation (Chan et al. 2023), and fact-checking (Du et al. (2023a). Additionally, there is extensive research on using multiple agents to simulate real-world environments, including societal, economic, and gaming simulations (Park et al. 2022; 2023; Xu et al., 2023b; Li et al., 2023b; Mukobi et al. 2023). Liang et al. (2023) propose leveraging multi-agent debate for machine translation. However, their approach is limited to the sentence level. In this work, we focus on the first category, specifically on the translation of literary texts. Literary translation is considered one of the most complex and challenging translation tasks, and we aim to address this challenge using a multi-agent system powered by LLMs.

Machine Translation Machine translation (MT) has achieved significant advancements in recent years, with developments spanning general-purpose MT (Cho et al., 2014; Sutskever et al., 2014, Vaswani et al., 2017; Gehring et al. 2017; Shen et al., 2019), low-resource MT (Zoph et al.|| 2016; Gu et al.|| 2018; Haddow et al.| 2022), multilingual MT (Liu et al., 2020; Fan et al., 2021; Wu et al., 2021; Li et al., 2022; Costa-jussà et al., 2022, Communication et al., 2023), and non-autoregressive MT (Gu et al., 2017; 2019a; Ghazvininejad et al., 2019), among others. However, these advancements are predominantly focused at the sentence level. Recently, efforts are made to enhance translation
quality by integrating contextual information into the translation process (Wang et al., 2017, Ding et al., 2020, Sun et al., 2022; Feng et al., 2022; Wu et al., 2023a; Herold \& Ney, 2023, Wu et al., 2024b), aiming to achieve more accurate and coherent translations that extend beyond individual sentences. More recently, large language models (LLMs) have demonstrated superior capabilities in various applications, including MT (Lu et al., 2023, Zhang et al., 2023b; Xu et al., 2023a; Robinson et al., 2023, Wang et al., 2023a, Wu et al. 2024a). Given the remarkable progress in machine translation (MT), the performance of MT seems to be saturating in the general domain. There is growing interest in literary translation, which is considered one of the more challenging translation tasks because it requires not only accuracy in meaning but also the conveyance of vivid expressions and cultural nuances (Thai et al., 2022, Wang et al., 2023b). Additionally, evaluating MT accurately remains a critical aspect of research in this field. While traditional metrics like BLEU are commonly used (Papineni et al. 2002), newer approaches involve utilizing pretrained language models to assess translation quality more effectively (Rei et al., 2020; Sellam et al., 2020, Juraska et al., 2023, Guerreiro et al., 2023). Kocmi \& Federmann(2023) employ the state-of-the-art LLM, GPT-4, to estimate translation quality and achieve state-of-the-art quality estimation performance at WMT 2023 (Freitag et al. 2023). In this work, we establish a novel multi-agent virtual company TrANSAGENTS for translating literary texts. We also propose two evaluation strategies for assessing the quality of the translated literary texts.

## 3 TRAnSAGENTS: A MULTI-AGENT VIRTUAL COMPANY FOR LITERARY TRANSLATION

![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-04.jpg?height=827&width=959&top_left_y=1251&top_left_x=583)

Figure 2: TRANSAGENTS, a multi-agent virtual company for literary translation.

We establish a virtual multi-agent translation company, TRANSAGENTS, featuring a diverse range of employees including a CEO, senior editors, junior editors, translators, localization specialists, and proofreaders. When a human client assigns a book translation task, a team of selected agents from TRAnSAGENTS collaborates to translate the book. This paradigm simulates the entire book translation process, where agents with different roles work together to ensure that the translation maintains high quality and consistency throughout. In this section, we describe the company overview of TransAGEnTS in Section 3.1, the core collaboration strategies of TransAGENTS in Section 3.2. and the translation workflow in Section 3.3

### 3.1 COMPANY OVERVIEW

To simulate the entire book translation process, in addition to the designated CEO, we have a diverse array of roles, including senior editors, junior editors, translators, localization specialists, and proofreaders in our company TRANSAGENTS. Each of these roles carries its own set of responsibilities:

- Senior Editors: Senior editors are responsible for overseeing the content production process. Their primary duties encompass setting editorial standards, guiding junior editors, and ensuring that the content aligns with the company's objectives.
- Junior Editors: Junior editors work closely under the guidance of senior editors. Their responsibilities typically include managing the day-to-day editorial workflow, editing content, and assisting in content planning. They also handle communications with various other roles within the organization.
- Translators: Translators are tasked with converting written material from one language to another while preserving the tone, style, and context of the original text. Translators must possess a profound understanding of both the source and target languages, as well as a familiarity with the subject matter they are translating.
- Localization Specialists: Localization specialists go beyond simple translation; they adapt content for specific regions or markets. This role involves not only translating language but also adjusting cultural references, idioms, and images to resonate with local audiences.
- Proofreaders: Proofreaders perform final checks for grammar, spelling, punctuation, and formatting errors. Their role is crucial in ensuring that content is polished and adheres to high-quality standards before publication.

To enhance the realism and efficacy of our simulation in the translation process, we strategically utilize GPT-4-TURBO to generate a diverse set of 30 virtual agent profiles for each distinct role. As illustrated in Figure 3, these profiles are comprehensively designed to include a wide array of attributes that extend well beyond language skills. Key characteristics such as gender, nationality, rate per word, educational background, years of experience, and areas of specialization are thoughtfully incorporated. This detailed and personalized approach not only enriches the authenticity of the translation process simulation but also mir-

```
Name: Sofia Chang
Languages: English, Mandarin, Spanish, French
Nationality: Canadian
Gender: Female
Age: 47
Education: Ph.D. in Comparative Literature
Personality: meticulous, introverted,
\hookrightarrow \mp@code { p e r f e c t i o n i s t , ~ c r i t i c a l , ~ t h o u g h t f u l }
Hobbies: gardening, chess, watercolor painting
Rate per word: 0.12
Years of working: 22
Profession: Senior Editor
Role prompt: You are Sofia Chang, a highly esteemed
\hookrightarrow Senior Editor [TRUNCATED]
```

Figure 3: An example profile of Senior Editor. rors the complexity and diversity found in realworld translation settings. The inclusion of such rich, detailed metadata about the agents not only enhances current simulation strategies but is also designed to support and inspire future research.

### 3.2 Agent Collaboration Strategies

In this section, we introduce two collaboration strategies used in this work, including Addition-bySubtraction Collaboration Algorithm 1, and Trilateral Collaboration Algorithm 2.

Addition-by-Subtraction Collaboration In our framework, we propose the Addition-bySubtraction Collaboration between two agents. Unlike the debate-style strategy (Liang et al., 2023, Du et al. 2023a, Chan et al. 2023), where multiple agents propose their own answers and a thirdparty agent concludes the discussion, our strategy involves only two agents. One acts as an Addition agent, responsible for extracting as much relevant information as possible, while the other agent serves as a Subtraction agent, tasked with reviewing the extracted information, eliminating redundant details, and providing feedback to the Addition agent. We present the details of our collaboration strategy in Algorithm 1. The Addition agent $\mathbf{A}$ first generates the initial response, aiming to include as much informative content as possible. Subsequently, the Subtraction agent $\mathbf{S}$ reviews the response and removes any redundant information. The conversation iterates until no further revisions are needed for the response.

```
Algorithm 1: Addition-by-Subtraction Collaboration
Input : Context C; Instruction I; Maximum number of iterations M; Addition agent A;
    Subtraction agent $\mathbf{S}$;
Output: The final response $\mathbf{R}$ that both agents agree upon.
$\mathbf{H} \leftarrow[\mathbf{C} ; \mathbf{I}] \quad \triangleright$ Initialize the conversation history;
$\mathbf{R} \leftarrow \emptyset \quad \triangleright$ Initialize the response;
$m \leftarrow 0 \quad \triangleright$ Current round;
while $m \leq \mathrm{M}$ do
    $m \leftarrow m+1 ;$
    $\mathbf{R}^{\prime} \leftarrow \mathbf{A}(\mathbf{H}) \quad \triangleright$ Generate detailed response;
    $\mathbf{F} \leftarrow \mathbf{S}\left(\mathbf{H}, \mathbf{R}^{\prime}\right) \quad \triangleright$ Review and remove redundant information;
    $\mathbf{H} \leftarrow \mathbf{H}+\left[\mathbf{R}^{\prime} ; \mathbf{F}\right] \quad \triangleright$ Append $\mathbf{R}^{\prime}$ and $\mathbf{F}$ to the conversation history $\mathbf{H}$;
    if $\mathbf{R}=\mathbf{R}^{\prime}$ then
        Break $\triangle$ Stop iterating as no further revisions are needed;
    $\mathbf{R} \leftarrow \mathbf{R}^{\prime} ;$
Return the final response $\mathbf{R}$;
```

```
Algorithm 2: Trilateral Collaboration
Input : Context C; Instruction I; Maximum number of iterations $\mathbf{M}$; Action agent $\mathbf{P}$;
        Critique agent $\mathbf{Q}$; Judgment agent $\mathbf{J}$;
Output: The final response $\mathbf{R}$ that is approved by the Judgment agent $\mathbf{J}$;
$\mathbf{H} \leftarrow[\mathbf{C} ; \mathbf{I}] \quad \triangleright$ Initialize the conversation history;
$m \leftarrow 0 \quad \triangleright$ Current round;
while $m \leq \mathrm{M}$ do
    $m \leftarrow m+1 ;$
    $\mathbf{R} \leftarrow \mathbf{P}(\mathbf{H}) \quad \triangleright$ Generate response;
    $\mathbf{F} \leftarrow \mathbf{Q}(\mathbf{H}, \mathbf{R}) \quad \triangleright$ Generate critiques;
    $\mathbf{H} \leftarrow \mathbf{H}+[\mathbf{R} ; \mathbf{F}] \quad \triangleright$ Append $\mathbf{R}^{\prime}$ and $\mathbf{F}$ to the conversation history $\mathbf{H}$;
    if $m>1$ then
    $\mathbf{D} \leftarrow \mathbf{J}(\mathbf{C}, \mathbf{I}, \mathbf{R}) \quad \triangleright$ The Judgment agent $\mathbf{J}$ evaluate the response quality;
    if $\mathbf{D}=T R U E$ then
        Break $\triangleright$ Stop iterating if the Judgment agent $\mathbf{J}$ thinks the response is of high
        quality;
```

Return the final response $\mathbf{R}$;

Trilateral Collaboration We divide the collaboration into three branches in TRANSAGENTS, referring to as Trilateral Collaboration:

- Action: The power to follow the instruction and implement the required actions.
- Critique: The power to review the generated response and provide constructive feedback to the Action branch.
- Judgment: The power to make the final decision on whether the response is satisfactory or requires further revision.

We assign one agent for each branch and present the details of the collaboration among these agents in Algorithm 2. The Action agent $\mathbf{P}$ generates a response $\mathbf{R}$ given the context $\mathbf{C}$ and instruction $\mathbf{I}$. The Critique agent $\mathbf{Q}$ then writes critiques $\mathbf{F}$ against the response $\mathbf{R}$. The Action agent $\mathbf{P}$ has the option to either accept the critiques and update the response or maintain the original response. At the end of the iteration, the Judgment agent $\mathbf{J}$ evaluates the response $\mathbf{R}$ to determine if the discussion can be concluded or if further deliberation is required.

### 3.3 TRANSLATION WORKFLOW

In this section, we introduce the book translation workflow in our company TRANSAGENTS, including two main stages: preparation (Section 3.3.1) and execution Section 3.3.2).

### 3.3.1 PREPARATION

Project Members Selection System prompts or messages are used to assign roles to individual agents during the role-playing process. In our company's setup, we create 30 agent profiles, each accompanied by a unique role assignment prompt, as illustrated in Figure 3. These prompts are essential for assigning specific roles to the agents before the dialogues begin. Within our framework, the initial step involves the CEO selecting a Senior Editor for the book translation project. This selection process takes into account both the client's requirements and the qualifications of potential Senior Editors. Once the Senior Editor is chosen, they work closely with the CEO to assemble the rest of the project team, carefully considering the skill sets and backgrounds of the candidates. Furthermore, we introduce a self-reflection strategy (Yao et al., 2023, Shinn et al., 2023, Qian et al. 2023). This strategy involves incorporating a "ghost agent" whose task is to prompt the CEO to reconsider their decision, as we observe that they sometimes struggle to select a Senior Editor with the desired language skills.

Translation Guideline Documentation To maintain consistency throughout the entire translation workflow, which involves multiple agents, we need to have a translation guideline. In TRANSAGENTS, there are five components: the glossary, the book summary, the tone, the style, and the target audience. We have designed different strategies to process them:

- Glossary: The primary purpose of a glossary in book translation is to compile essential terms from the source language and provide their corresponding translations in the target language. This ensures consistency and accuracy in the usage of these terms throughout the book, especially since some terms may have multiple acceptable translations. In our process, we leverage the Addition-by-Subtraction Collaboration, as described in Algorithm 1. for collecting the key terms. For each chapter, the Junior Editor, serving as the Addition agent A, makes an exhaustive attempt to identify all potential key terms initially. Subsequently, the Senior Editor, serving as the Subtraction agent $\mathbf{S}$, reviews the identified key terms and removes any that are generic. The conversation continues until the list of collected key terms does not need further revision. Next, the collected key terms are translated by the Senior Editor, with consideration of their context.
- Book Summary: Generating a book summary is crucial to provide a comprehensive overview of the narrative. This task is facilitated by the collaboration between the Junior Editor (Addition Agent A) and the Senior Editor (Subtraction Agent S), employing the Addition-by-Subtraction Collaboration as depicted in Algorithm 1. In this process, the Junior Editor aims to retain as much detail as possible in the chapter summaries, while the Senior Editor focuses on removing superfluous information. Following the compilation of chapter summaries, the Senior Editor then crafts the book summary, mirroring the process of gathering a glossary.
- Tone, Style, and Target Audience: The translation of a book is more than just a word-forword conversion; it's a delicate process of adapting tone, style, and content to resonate with the target audience while staying true to the original text's essence. In TransAGENTS, the Senior Editor defines the tone, the style, and the target audience of the translated book based on a randomly selected chapter.

Overall, the glossary, book summary, tone, style, and target audience collectively constitute the comprehensive translation guidelines. These guidelines serve as an essential part of the prompts for all roles involved in the book translation process, ensuring consistency and coherence throughout the entire work.

### 3.3.2 EXECUTION

In the execution phase, the process is divided into four distinct sub-stages: translation, cultural adaptation, proofreading, and final review. During the first three sub-stages, our approach utilizes the collaborative strategy as illustrated in Algorithm 2 Within this framework, the roles of Action agents $\mathbf{P}$ are assigned to the Translator, the Localization Specialist, and the Proofreader, in that order. Meanwhile, the responsibilities of the Critique agent $\mathbf{Q}$ and the Judgment agent $\mathbf{J}$ are fulfilled by the Junior Editor and the Senior Editor, respectively. Finally, the Senior Editor performs the final checks before publication.

Translation, Localization, and Proofreading The translation stage involves three key roles: the Translator, the Junior Editor, and the Senior Editor. These roles collaborate to translate the book from the source language to the target language on a chapter-by-chapter basis. The translation process begins with the Translator (the Action agent $\mathbf{P}$ ) initially translating the chapter content from the source language to the target language. Next, the Junior Editor (the Critique agent Q) undertakes a thorough review of the translation, ensuring it adheres to the guidelines while also identifying any potential errors or areas for improvement. Lastly, the Senior Editor (the Judgment agent $\mathbf{J}$ ) evaluates the translation and determines if further revision is needed. Following the translation, the cultural adaptation process begins. The Localization Specialist tailors the translated content to fit the cultural context of the target audience, ensuring that it resonates well and maintains the intended meaning. Next, the Proofreader perform the checks for language errors. Throughout the cultural adaptation and proofreading stages, both the Junior Editor and the Senior Editor continue to offer critiques and evaluations to refine the content further.

Final Review The final review is the concluding step in the editorial process. At this point, the $\mathrm{Se}-$ nior Editor evaluates the translation quality of each chapter and also examines how pairs of adjacent chapters flow into each other. The Senior Editor not only verifies that each chapter is internally coherent and meets quality standards on its own but also ensures that the transitions between chapters are smooth, thereby maintaining narrative consistency.

On the Importance of the Judgment Agent We introduce the Judgment Agent in Algorithm 2. which is responsible for evaluating the quality of the response and determining whether further revision is needed, without requiring the conversation history. Owing to the nature of web novels, each turn of dialogue is likely to contain a few thousand words. Although recent advances in large language models (LLMs) claim that LLMs are capable of processing extremely lengthy sequences of up to millions of tokens, we still observe that our agents are not able to effectively leverage the information in the context as the conversation expands. Additionally, we observe that the meaning of translations tends to deviate from the original text after several iterations of revision. Therefore, it is critical to have the Judgment agent within the Trilateral Collaboration to ensure the overall quality of the response.

## 4 EXPERIMENTAL SETUP

In this work, our experimental setup primarily follows the WMT2023 shared task on discourse-level literary translation (DLLT) (Wang et al. 2023b). The following sections introduce the baselines Section 4.1, datasets Section 4.2, and evaluation approaches Section 4.3 used in our study.

### 4.1 BASELINES

We leverage the state-of-the-art LLM GPT-4-TURBO as the backbone of our agents ${ }^{1}$ and compare our approach with the unconstrained systems in WMT2023 shared task on DLLT:
- Llama-MT: Du et al. (2023b) fine-tune Llama-7B for literary translation. The finetuned LLAMA-MT model translates 2,048 consecutive tokens at a time.
- GPT-4: While recent versions of GPT-4 models claim to support a context size of up to $128 \mathrm{~K}$ tokens, they are restricted to generating a maximum of 4,096 tokens per response (OpenAI 2023). Therefore, we employ the GPT-4-0613 and GPT-4-1106-PREVIEW models to translate the documents on a chapter-by-chapter basis.
- Google: We employ the Google Translate system to translate the documents on a sentence-by-sentence basis.
- DUT: Zhao et al. (2023) explore several techniques to enhance the performance of large language models (LLMs) in discourse-level translation tasks.
- HW-TSC: Xie et al. (2023) initially train a sentence-level Transformer to establish a baseline, subsequently enhancing its discourse-level capabilities through domain adaptation and discourse modeling, employing a variety of techniques.[^1]

### 4.2 DATASETS

In this work, we do not need to train new models and all the agents is GPT-4-TURBO with various roles. Hence, we only leverage the official test set of WMT2023 shared task on DLLT. The official test set is collected from 20 web novels, each of which consists 20 consecutive chapters, totaling 240 chapters. The test set contains two references: REFERENCE 1 is translated by human translators and REFERENCE 2 is built by manually aligning bilingual text in web page.

### 4.3 EVALUATION

Translating literary works differs significantly from translating standard machine translation (MT) corpora, such as news articles or parliamentary proceedings. Thai et al. (2022) present a comprehensive list of techniques employed by literary translators, which largely differ from those used in common MT domains. Furthermore, literary translators have the freedom and the burden of both semantic and critical interpretation, resulting in the absence of a single, unique best translation for literary texts. In this work, we employ two evaluation approaches:

- Standard Evaluation: Following Wang et al. (2023b), we use $d$-BLEU (Papineni et al. 2002, Post. 2018, Liu et al. 2020) to evaluate the translation quality ${ }^{2}$, as the translations may not strictly align with the source text on a sentence-by-sentence basis. To compute the $d$-BLEU score, we concatenate all the chapter translations into a single document for evaluation. We present the results in Section 5.
- Preference Evaluation: Acknowledging the concern that there is no single, universally preferred translation for literary texts, we ask human raters or LLMs to select their preferred translation without giving them a reference translation. Further details regarding this novel evaluation approach are discussed in Section 6


## 5 STANDARD EVALUATION

We present the automatic evaluation results in Table 1. Interestingly, our approach performs poorly in terms of the $d$-BLEU metric, achieving the lowest scores among the compared methods. However, it is important to consider that $d$-BLEU has limitations and may not fully capture the quality and coherence of the generated text. As pointed out by Freitag et al. (2020), typical references used for calculating $d$-BLEU scores often exhibit poor diversity and tend to concentrate around translationese language. This suggests that a low $d$-BLEU score does not necessarily imply poor performance of our approach.

|  | $d$-BLEU $\uparrow$ |
| :---: | :---: |
| LLAMA-MT (Du et al. $\sqrt[2023 b]{2}$ | 43.1 |
| GPT-4-0613 (OpenAI 2023) | 43.7 |
| GPT-4-1106-PREVIEW (OpenAI 2023) | 47.8 |
| GOOGLE | 47.3 |
| DUT (Zhao et al. 2023) | 50.2 |
|  | 52.2 |
| TRANSAGENTS (Ours) | 25.0 |

Table 1: Automatic evaluation ( $d$-BLEU) results on WMT2023 DLLT test set. $\uparrow$ indicates higher is better. The worst result is highlighted in bold.

Our results align with the findings from Thai

et al. (2022), who argue that automatic metrics cannot accurately reflect human preference in the context of literary translation. Furthermore, while automatic metrics are typically highly correlated with human judgments based on the Multidimensional Quality Metrics (MQM) framework (Burchardt. 2013), this framework may not be suitable for assessing translation quality in the context of literary translation ${ }^{3}$ The unique characteristics and creative aspects of literary texts require a more nuanced evaluation approach that goes beyond the scope of standard automatic metrics and MQM-based human assessments.[^2]

```
Q: Which of the following writing style do you prefer?
[x] Chapter 455: Turnaround 3 "Allow me to demonstrate the sensing of Formless Fluctuation; it's remarkably
\hookrightarrow \mp@code { s t r a i g h t f o r w a r d , " ~ i n t e r j e c t e d ~ a n o t h e r ~ s o r c e r e r , ~ a ~ s m i l e ~ e v i d e n t ~ i n ~ h i s ~ v o i c e . ~ " Y o u r ~ a s s i s t a n c e ~ i s }

```

![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-10.jpg?height=24&width=1331&top_left_y=388&top_left_x=377)

```
\hookrightarrow \text { remaining Fragments. He had initially planned to conquer an array of Great Evil Spirits to amass}
~substantial reserves of pure soul power. Yet, the present opportunity necessitated an immediate and
\hookrightarrow \mp@code { d e c i s i v e ~ a c q u i s i t i o n . ~ P r o m p t l y , ~ t h e ~ s o r c e r e r ~ l e a d e r ~ b r o u g h t ~ L i n ~ S h e n g ~ t o ~ a ~ d a u n t i n g ~ E v i l ~ S p i r i t ~ G a t e . }
\hookrightarrow \mp@code { B o t h ~ e x t e n d e d ~ t h e i r ~ h a n d s , ~ g e n t l y ~ t o u c h i n g ~ t h e ~ g a t e ' s ~ e n i g m a t i c ~ f r a m e , ~ e y e s ~ c l o s e d ~ a s ~ o n e . ~ T h e ~ l e a d e r }

```

![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-10.jpg?height=32&width=1336&top_left_y=510&top_left_x=374)

```
[ ] Chapter 455 Reversion 3 "This is to let you feel the fluctuation of aura. It's really simple." Another
\hookrightarrow \text { Warlock couldn't help but interrupt with a smile. "Then I'll have to trouble you." Lin Sheng nodded. He}
\hookrightarrow \text { needed to find the other fragments as soon as possible. Originally, he had planned to conquer more evil}
\hookrightarrow \text { spirits and obtain more pure soul power. But now that he encountered such an opportunity, the most}
\hookrightarrow \mp@code { @ m p o r t a n t ~ t h i n g ~ f o r ~ h i m ~ w a s ~ t o ~ g e t ~ i t ~ a s ~ s o o n ~ a s ~ p o s s i b l e . ~ S o o n , ~ t h e ~ W a r l o c k ~ C o m m a n d e r ~ l e d ~ L i n ~ S h e n g ~ t o }
\hookrightarrow \mp@code { a n ~ E v i l ~ S p i r i t ~ G a t e . ~ T h e ~ t w o ~ r e a c h e d ~ o u t , ~ t o u c h e d ~ t h e ~ f r a m e ~ o f ~ t h e ~ E v i l ~ S p i r i t ~ G a t e ~ a t ~ t h e ~ s a m e ~ t i m e }
\hookrightarrow \text { and closed their eyes. The Warlock Commander quickly used his ability to build the space base as a}
\hookrightarrow \text { coordinate}
[ ] No Preference
```

Figure 4: The user interface for Monolingual Human Preference (MHP). [ $\mathrm{x}$ ] indicates the selection of human evaluator.

## 6 PREFERENCE EVALUATION

It is crucial to acknowledge that a literary text does not possess a single, universal translation. Conventional translation evaluation methodologies, which typically rely on direct comparisons to a standard reference translation, fail to accommodate the multifaceted and subjective nature of literary texts. Following Thai et al. (2022), we engage both human evaluators and large language models (LLMs) to assess translations based on their preferences. In this section, we describe our methods for preference evaluation in Section 6.1 and present our results in Section 6.2

### 6.1 EVALUATION METHODS

In this section, we propose two preference evaluation methods, monolingual human preference (MHP, Section 6.1.1) and bilingual LLM preference (BLP, Section 6.1.2). For both methods, we use the winning rate (\%), which is the percentage of instances where a model's generated chapter is preferred by either the human evaluators (in MHP) or the LLM (in BLP), to measure the model performance.

### 6.1.1 MoNOLINGUAL HUMAN PREFERENCE

When reading a translated book, it is not necessary for the reader to understand the original language. Therefore, a better translation should naturally be preferred by readers without needing to refer to the text in its original language.

Preprocessing In this work, the translations of each chapter are first manually split into several segments containing approximately 150 words each, based on the story's plot. This translation segmentation step is necessary because the full translations contain thousands of words, and human evaluators may struggle to stay focused when evaluating such long passages at once.

Evaluation The human evaluators are tasked with comparing pairs of translation segments describing the same part of the story and selecting their preferred translation for each segment pair, with the user interface shown in Figure 4 To ensure evaluations consider the full context, each evaluator is required to evaluate all the segments within a chapter in their original order, as segments may depend on information from previous segments.

Implementation In this study, we collect human preferences on translations through SurveyMonkey ${ }^{4}$ To ensure the evaluators are from the target audience, we ask if they are interested in Chinese[^3]web novels before starting the evaluation ${ }^{5}$ We only recruit evaluators from the United States to minimize potential impacts of demographics. Each translation pair is evaluated by at least 10 people and costs us $\$ 0.30$ USD per annotation. We filter out possible low-quality responses or human evaluators based on following criteria:
- Being labeled as low quality by SurveyMonkey's response quality model;
- Giving "No Preference" for all selections;
- Taking less than 20 seconds for the evaluation.

After filtering, we collect at least 5 responses per segment pair.

Mitigating Positional Bias Human evaluators may exhibit a positional bias when evaluating response quality. To mitigate this bias in our translation evaluations, the positions of the translation segments being compared are randomly swapped for each selection, as shown in Figure 4 . Furthermore, the "No Preference" (Tie) option, indicating that the evaluator does not prefer one translation over the other, is always presented as the third option.

Response Aggregation We aggregate the human evaluations using majority voting, where the most selected option is considered the final preference. If two translation systems receive the same number of votes, we record the final preference as "No Preference" (Tie).

### 6.1.2 BILINGUAL LLM PREFERENCE

The nature of literary texts, with their inherent complexities, artistic expression, and cultural nuances, makes it virtually impossible to produce a single, universally correct translation. As a result, multiple translations of the same literary text can coexist, each offering a unique perspective and interpretation. Recent works demonstrate that the reference translations are likely to be of low quality (Freitag et al., 2023, Xu et al. 2024). Kocmi \& Federmann (2023) demonstrate that GPT-4 is capable of accurately estimating translation quality without the need for human reference translations. Their proposed GEMBA-MQM metric achieves state-

```
[The start of source]
[$src_lang]: $src
The end of source]
The start of assistant 1's translation]
$tgt lang]: $asst1
[The end of assistant 1's translation]
[The start of assistant 2's translation]
[$tgt_lang]: $asst2
The end of assistant 2's translation
We would like to request your feedback [TRUNCATED]
```

Figure 5: The prompt used for bilingual LLM preference evaluation. of-the-art performance in WMT 2023 Metric Shared task (Freitag et al., 2023).

Motivated by Kocmi \& Federmann (2023), we evaluate the translation segment pairs using GPT4-0125-PREVIEW without providing the reference translations. Recent research demonstrates that even state-of-the-art LLMs may struggle to process extremely long sequences (Bai et al. 2023b, Song et al. 2024; Li et al., 2024). Therefore, we require GPT-4-0125-PREVIEW to determine which translation segment is better as described in Section 6.1.1, using the prompt shown in Figure 5. instead of directly comparing the quality of two entire chapters. We employ a different variant of GPT-4 for evaluation to avoid the potential bias. Given concerns about positional bias in LLM evaluation raised by recent studies (Wu \& Aji, 2023, Zheng et al., 2023a; Dubois et al., 2024), we evaluate each translation segment pair in both forward and reversed directions.

### 6.2 EXPERIMENTS

Setup As described in Section 4.2, there are 12 web novels consisting of 240 chapters in our test set. Due to the high cost of human evaluation, we only compare our TransAgEnTS with the REFERENCE 1 and GPT-4-1106-PREVIEW models. We evaluate the first two chapters of each of the 12 web novels in our test set using both of our preference evaluation methods.[^4]

![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-12.jpg?height=299&width=610&top_left_y=276&top_left_x=432)

Figure 6: Monolingual Human Preference evaluation results. GPT-4 indicates GPT-41106-PREVIEW.

![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-12.jpg?height=222&width=501&top_left_y=274&top_left_x=1083)

$\square$ TRANSAGENTS wins $\square$ Tie $\square$ TRANSAGENTS loses

Figure 7: Bilingual LLM Preference evaluation results. GPT-4 indicates GPT-4-1106PREVIEW.

|  | Overall | VG | EF | SR | CR | F | SF | HT | FR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Monolingual Human Preference |  |  |  |  |  |  |  |  |  |
| GPT-4-1106-PREVIEW | 55.6 | 64.5 | 68.2 | 63.3 | 44.6 | 68.2 | $\underline{39.1}$ | 48.0 | $\mathbf{7 7 . 8}$ |
| REFERENCE 1 | 52.1 | $\mathbf{6 7 . 7}$ | 63.6 | 56.7 | 42.9 | 63.6 | $\underline{37.0}$ | 40.0 | 66.7 |
| Bilingual LLM Preference |  |  |  |  |  |  |  |  |  |
| GPT-4-1106-PREVIEW | 55.9 | $\mathbf{7 4 . 1}$ | 56.8 | 58.3 | 47.3 | 70.5 | 47.8 | $\underline{34.0}$ | 66.7 |
| REFERENCE 1 | 66.2 | $\mathbf{8 8 . 7}$ | 59.1 | 70.0 | 54.5 | 83.0 | $\underline{53.3}$ | 62.0 | 61.1 |

Table 3: The breakdown winning rate of TRANSAGENTS against GPT-4-1106-PREVIEW and REFERENCE 1. The best results are highlighted in bold. The worst results are highlighted in underline.

Results We compare the performance of our TransAgents with REFerence 1 and GPT-41106-PREVIEW using monolingual human preference evaluations. The results, presented as winning rates, are shown in Figure 6. The translations produced by TransAGENTS are marginally preferred by human evaluators compared to both REFERENCE 1 and GPT-4-1106-PREVIEW. Additionally, we evaluate the models using bilingual LLM preference, with the results presented in Figure 7 The translations generated by TRANSAGENTS are also more preferred by GPT-4-0125-PREVIEW compared to the other models. Referring to the results in Table 4, we observe that GPT-4-0125PREVIEW appears to have a strong preference for diverse and vivid descriptions when evaluating literary translations. We leave the further investigation to the future work.

## 7 ANALYSIS

What Causes TransAgents to "Fail" in Terms of $d$-BLEU? As shown in Table 1 . the translation produced by TransAGENTS achieves the lowest $d$-BLEU score among the compared methods. To investigate the reasons behind this, we evaluate the output of each stage in the TransAGENTS workflow using the official references from the WMT2023 DLLT test set. The results, presented in Table 2. reveal that, although the backbone of the agents in TransAGENTS is GPT-4-1106PREVIEW, the initial translation produced by TransAGENTS achieves a significantly lower $d$-BLEU score. This suggests that the translation guideline is the main contributor to the final translation quality. Moreover, the localization step further reduces the $d$-BLEU score, while the proofreading step only minimally modifies the translation.

Strengths and Weaknesses of TransAgents The original texts of the test examples are publicly accessible online and span a variety of genres, including Video Games (VG), Eastern Fantasy (EF), Sci-fi Romance (SR), Contemporary Romance (CR), Fantasy (F), Science Fiction (SF), Hor-
ror \& Thriller (HT), and Fantasy Romance (FR). We present a detailed analysis of the performance of our model TransAgents, across these categories in Table 3 Our observations indicate that TransAGENTS excels in domains that demand extensive domain-specific knowledge, such as historical contexts and cultural nuances. These areas often pose significant challenges for translators. On the other hand, TRansAGENTS tends to underperform in contemporary domains, which may not require as much specialized knowledge. This performance trend underscores the model's strengths and weaknesses.

Linguistic Diversity Linguistic diversity in literary texts is critical for enriching the reading experience. To quantify the linguistic diversity of the translation, we leverage two metrics: the Moving-Average Type-Token Ratio (MATTR) (Covington \& McFall, 2010) and the Measure of Textual Lexical Diversity (MTLD) (McCarthy \& Jarvis, 2010). As shown in Table 4, assisted by our translation guidelines, our initial translation significantly improves linguistic diversity compared to the source text. Moreover, the localization step further enhances linguistic diversity, while the proofreading step does not

|  | MATTR $\uparrow$ | MTLD $\uparrow$ |
| :--- | :---: | :---: |
| REFERENCE 1 | 80.9 | 89.1 |
| GPT-4-1106-PREVIEW | 81.5 | 94.9 |
| TRANSAGENTS |  |  |
| - translation | 83.5 | 117.0 |
| - localization | 83.6 | 119.4 |
| - proofreading | 83.6 | 119.4 |

Table 4: Linguistic diversity in terms of MATTR (up-scaled by $\times 100$ ) and MTLD. $\uparrow$ indicates higher is better. affect it. These results demonstrate the effectiveness of our approach in preserving and enhancing the richness of language in the translated literary work.

Cost Analysis The cost of human translation services can be influenced by several factors, including the genre of the text, the translator's location, and their level of experience. The American Translators Association recommends a minimum rate of US $\$ 0.12$ per word for professional translation services ${ }^{6}$ The REFERENCE 1 from the WMT2023 DLLT test set contains an average of 1,404 English words per chapter, resulting in a translation cost of $\$ 168.48$ USD per chapter. In comparison, translating using TRanSAGENTS costs approximately \$500 USD for the entire test set, which is equivalent to $\$ 2.08$ USD per chapter. Translating literary text using TransAgents can lead to an $80 \times$ reduction in translation costs.

## 8 CASE STUDY

In this section, we explore two case studies with regard to cultural adaptation and content omission, shedding light on both the strengths and weaknesses of our approach. Additionally, we enrich our analysis by incorporating insights from interviews with two experienced professional translators.

Cultural Adaptation In Chinese, job titles are typically placed before a person's name, whereas in English, job titles usually come after the person's name. This order reflects differing linguistic and cultural conventions regarding the structuring of personal information in the two languages. As demonstrated in Table 5. TransAGENTS is the only system that accurately reflects this cultural context in its translations. In contrast, both REFERENCE 1 and GPT-4-1106-PREVIEW fail to correctly adjust the order of names and job titles, thus not adhering to the cultural norms expected in the target language. The ability to produce translations that are not only linguistically accurate but also culturally appropriate is crucial. This emphasizes the capability of TRANSAGENTS to provide translations that are culturally appropriate, ensuring an immersive reading experience for readers in the target language.

Global Consistency It is important to maintain consistency throughout the book translation from the start to the end. As shown in Table 6, the chapter titles are consistent, with the exception of the index. Both ReFerence 1 and TransAgents successfully produce consistent translations.[^5]

| Original Text | 罗德抬起头来, 正好看见一个中年男子推门走进来,他穿着冒险者的皮甲, 一头鲜红的长发随意的向后梳理着, 看见罗 <br> 德, 男子微微一“你好, 先生, 我是星月佣兵团的团长, 卡特。” |
| :---: | :---: |
| REFERENCE 1 | Rhode looked up and saw a middle-aged man pushing the door open. He wore an adventurer's leather armor and casually combed, <br> back his long red hair. The man smiled when he saw Rhode. "Hello, sir. I am the Guild Leader of the Star Moon Mercenaries, Carter." |
| GPT-4-1106-PREVIEW | Rhode looked up and saw a middle-aged man entering through the door. He was dressed in an adventurer's leather armor, with a <br> head of bright red hair casually combed back. Seeing Rhode, the man smiled slightly. "Hello, sir, I am the leader of the Star Moon <br> Mercenary Group, Carte." |
| TRANSAGENTS | Rhode looked up to see a middle-aged man entering. The man was dressed in the leather armor typical of adventurers, his fiery red <br> hair casually swept back. Spotting Rhode, the man offered a modest smile. "Hello, sir. I am Carter, the leader of the Star Moon <br> Mercenary Corps." |

Table 5: Case study for cultural adaptation. The text highlighted in red indicates that the translation is accurate in meaning but not in cultural context. The text highlighted in blue indicate that the translation is accurate both in meaning and in cultural context.

| Original Text | 第1906章不思量, 自难忘 (十二) [OMITTED] 第1907章不思量, 自难忘 (十三) [OMITTED] |
| :--- | :--- |
| REFERENCE 1 | Chapter 1906: Unforgettable Memories (12) [OMITTED] Chapter 1907: Unforgettable Memories (13) |
| GPT-4-1 106-PREVIEW | Chapter 1906: It's Hard to Forget Without Thinking (Twelve) [OMITTED] Chapter 1907: Without Intention, Unforgettable (Thirteen) |
| TRANSAGENTS | Chapter 1906: Without Intention, Unforgettable (Twelve) [OMITTED] Chapter 1907: Without Intention, Unforgettable (Thirteen) |

Table 6: Case study for global consistency. The text highlighted in red indicates that GPT-4-1106PREVIEW generates inconsistent translations across different chapters.

However, GPT-4-1106-PREVIEW struggles with maintaining consistency across different chapters. This demonstrates that TranSAGENTS is capable of maintaining consistency throughout the entire translation process, similar to human translators.

Content Omission Our TransAgents is generally preferred over both REFERENCe 1 and GPT4-1106-PREVIEW according to evaluations by human judges and large language models (LLMs) Figure 6 and Figure 7). However, despite its higher preference, the translations produced by TranSAGENTS are not without flaws. A detailed analysis of the translated chapters, when divided into smaller segments, reveals that both GPT-4-1106-PREVIEW and TRANSAGENTS exhibit significant issues with content omission, as illustrated in Table 7. While these omissions do not seem to impact the overall development of the story plot, they could potentially influence other critical aspects of the narrative. For example, missing content could diminish the depth of character development or alter the intended emotional impact of the text. Such omissions, therefore, raise concerns about the completeness and fidelity of the translation in preserving the nuanced expressions and thematic elements of the original texts.

Comments from Professional Translators We anonymize the translations from TRANSAGENTS, REFERENCE 1, and GPT-4-1106-PREVIEW for a randomly selected chapter and present both the original text and the translations to two experienced professional translators. We request that they assess and rank the quality of each translation and provide their comments on the translations. As shown in Table 8, both Translator A's and Translator B's comments highlight the novel-like, expressive translation style of TraNSAGENTS, which uses sophisticated language, though it sometimes omits parts of the original text. REFERENCE 1, and GPT-4-1106-PREVIEW stick closer to the original text. Overall, TransAGENTS's translations are viewed as the most expressive and engaging, REFERENCE 1's as straightforward, and GPT-4-1106-PREVIEW's as the most traditional. These comments confirm that TRANSAGENTS is capable of producing more expressive and engaging translations, compared to REFERENCE 1 and GPT-4-1106-PREVIEW.

## 9 LIMITATIONS

The primary limitation of our study centers on the evaluation methods used. Extensive literature has highlighted the issues in conventional machine translation (MT) evaluation techniques, such as poor evaluation metrics and the reliability of reference translations (Papineni et al., 2002, Post, 2018; Rei et al. 2020; Freitag et al. 2020, 2021, 2022; Kocmi et al., 2023; Freitag et al. 2023). Beyond traditional MT evaluation metrics such as $d$-BLEU, we propose additional methods, namely

| Original Text | ![](https://cdn.mathpix.com/cropped/2024_06_04_00d83d196fa16bc38932g-15.jpg?height=254&width=1113&top_left_y=290&top_left_x=621) |
| :---: | :---: |
| REFERENCE 1 | She called a maid to take Ye Chen and Cheng Anya downstairs to freshen up. Little Cutie really wanted to go with them, but he <br> stayed behind as Bai Ye seemed ready to start fighting someone, so Little Cutie quickly grabbed his sleeve. "Bai Ye, can you save my <br> Mommy and Daddy?" The child's eyes were bright, like two black grapes embedded in his fair face, full of expectation. It seemed <br> that if Bai Ye shook his head, the light in his eyes would dim. Jason grabbed Little Cutie and hugged him fiercely, feeling quite <br> confident. "Darling, don't worry. Little Bai can save even the dead, not to mention living people. Stop worrying about them and <br> come help me take down the Mafia's line of defense when you have time." "Who was questioning Bai Ye's medical skills just now?" <br> Black Jack expressed doubt about this. Jason threw a punch at him, but he agilely dodged it. Being carried by the tall man, Little <br> Cutie felt extremely conflicted and kicked Jason. "Put me down." "I also want a son like you. Ningning, will you be my son? I've <br> been coveting you for a long time." Jason's azure eyes burst with enthusiasm as if Little Cutie was a rare piece of jewelry. Everyone <br> was speechless... Bai Ye smiled and said, "Jason, let your mother help you improve your Chinese. Don't always complain that the <br> commander doesn't know how to use words. You're not much better." "I'm not on the same level as the commander, okay?" Jason <br> was very dissatisfied that someone had compared him to the commander. It was a matter of principle that officials and bandits were <br> not on the same side. Little Cutie struggled a little, and Jason put him down. Little Cutie asked, "Bai Ye..." "Ningning, don't worry. <br> I'll check them up before making a conclusion." Bai Ye said, rubbing the head of Little Cutie, "I promise I will do my best." |
| GPT-4-1106-PREVIEW | She summoned the maids to take Ye Chen and Cheng Anya to wash up. Xiao Nai Bao wanted to follow, but he stayed here, as Bai <br> Ye looked like he was about to beat someone up, so Xiao Nai Bao quickly grabbed his sleeve. Bai Ye, can you save my daddy and <br> mommy? Xiao Nai Bao's eyes were bright, like two black grapes embedded in his tender face, full of hope, as if the light in his eyes <br> would dim if Bai Ye shook his head. Jason grabbed Xiao Nai Bao and hugged him, full of pride, "Baby, don't worry, Xiao Bai can <br> even save the dead, let alone the living. Stop worrying about nothing, come help me bombard the mafia's defenses when you're free." <br> Who was questioning Bai Ye's medical skills just now? Black Jack expressed doubt, and with a palm strike, Jason swiftly dodged. <br> Xiao Nai Bao struggled a bit, and Jason put him down. Xiao Nai Bao asked, "Bai Ye..." Ningning, wait until I've examined them <br> before I can be sure, don't be anxious, Bai Ye said, ruffling Xiao Nai Bao's head, "I promise to do my best." |
| TRANSAGENTS | At her behest, maids ushered Ye Chen and Cheng An Ya away for a bath. Xiao Naibao, torn between following and staying, ultimately <br> remained as Bai Ye appeared poised for a scuffle. Tugging at Bai Ye's sleeve, Xiao Naibao looked up at him with hopeful eyes. "Bai <br> Ye, is there a way to cure my daddy and mommy?" "Ning Ning, let me conduct a thorough examination first. Stay calm," Bai Ye <br> soothed, patting the boy's head. "I'll do everything in my power." |

Table 7: Case study for content omission. The text highlighted in red and blue indicates the content that is omitted by TransAgents. The text highlighted in red represents the content that is omitted by GPT-4-1106-PREVIEW.

| ranslator $\mathrm{A}$ | TRANSAGENTS's translation style is similar to that of a novel, with sophisticated wording and personal flair. Despite some omis- <br> sions, it makes the text more concise and effectively conveys the original text's mood and meaning. REFERENCE 1 and GPT-4- <br> 1106 -PREVIEW's translations are more conventional, adhering strictly to the original text word for word. However, GPT-4-1106- <br> PREVIEW's translation is more grammatically precise than REFERENCE 1's, and its wording is slightly better, making its translation <br> aesthetically superior to REFERENCE 1's but still not reaching the literary expressiveness of TRANSAGENTS. From their translation <br> habits, TRANSAGENTS appears to have a solid foundation in English, REFERENCE 1 seems to rely on machine translation, and <br> GPT-4-1106-PREVIEW behaves like a standard, rule-abiding translator. |
| :---: | :---: |
|  | RANSAGENTS's translation breaks away from the constraints of the original language, using the language freely with ample addi- <br> ons and expansions, and the choice of vocabulary also demonstrates a deeper understanding of the language. REFERENCE 1 remains <br> ithful to the original text, translating directly and succinctly without adding personal interpretations. GPT-4-1106-PREVIEW's trans- <br> tion style is similar to REFERENCE 1's, both strictly adhering to the original without much personal interpretation or embellishment. <br> verall, TRANSAGENTS's translation shows the greatest depth and sophistication, followed by REFERENCE 1, while GPT-4-1106- <br> REVIEW performs most ordinarily among the three. |

Table 8: Comments from two experienced professional translators on the translations from TransAgEnTS, REFERENCE 1, and GPT-4-1106-PREVIEW. We present both the original text and the anonymized translations to two experienced professional translators. The original comments are written in Chinese, and we make adaptations while preserving their original meaning. We replace the anonymized system names with the actual system names to improve readability. The translation systems are highlighted in red.

Monolingual Human Preference and Bilingual LLM Preference, to assess translation quality. However, the implementation of these novel evaluation strategies introduces several challenges that may undermine the validity of our findings:

- Document Segmentation: Evaluating ultra-long texts introduces distinct challenges in human evaluation. In our preliminary study, we observe that human evaluators often struggle to maintain focus when reading documents containing thousands of words, which could potentially compromise the accuracy of their evaluations. Moreover, while segmenting these lengthy texts into smaller, content-based portions may simplify the task, this method risks disrupting the narrative flow and connections between different sections, potentially resulting in a loss of overall coherence. We strategically segmented the documents for this
study. However, developing more effective methods for human evaluation of ultra-long texts remains an area for future research.
- Target Audience: Literary texts are crafted with specific target audiences in mind. In our study, we initially aim to distribute our questionnaires through an online forum dedicated to web novels, intending to gather feedback directly from the target audience. However, this approach faced challenges, either due to community regulations or the slow pace of feedback collection. Additionally, although we confirm the interest of human evaluators in Chinese web novels before they participate in the evaluation, there is a possibility that evaluators might claim interest simply to qualify for the job, regardless of their true preferences. Consequently, this could mean that our evaluation results might not accurately reflect the true preferences of the target audience.
- Evaluation Scale: Due to constrained resources, the scope of our evaluation scale may be inadequate. We segment only the first two chapters of each book in the test set and gather a minimum of five valid responses per segment. Recent studies highlight the significant diversity in human preferences (Zheng et al., 2023b, Wu \& Aji, 2023, Hosking et al., 2023). Consequently, the limited scale of our evaluation could affect the outcomes.
- Human-Written References: Although the reference translations are said to be authored by professional human translators, there is a likelihood that these translators may use commercial machine translation systems, such as Google TRANSLATE, to reduce their workload. Unfortunately, we cannot verify whether the reference translations are genuinely created by humans.

We acknowledge these limitations and leave them to the future studies.

## 10 CONCLUSION

In this paper, we introduce TransAgEnTS, a novel multi-agent virtual company designed for literary translation that reflects the traditional translation publication process. Utilizing a multi-agent approach, this system effectively tackles the intricate nuances inherent in literary texts. We propose two innovative evaluation strategies: Monolingual Human Preference (MHP) and Bilingual LLM Preference (BLP), to assess the quality of the translations. MHP evaluates how the translation resonates with the target audience, focusing on fluidity and cultural appropriateness, whereas BLP employs advanced language models to directly compare the translations with the original texts. Although the $d$-BLEU scores are lower, our empirical results demonstrate that translations produced by TRANSAGENTS are favored by both human evaluators and language models over human-written references. We also provide detailed analyses of the strengths and weaknesses of TRANSAGENTS, highlighting possible directions for future research.

## REFERENCES

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023a. doi: 10.48550/ARXIV. 2312.11805. URLhttps://doi.org/10.48550/arXiv.2312.11805.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernández Ábrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan A. Botha, James

Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vladimir Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, and et al. Palm 2 technical report. CoRR, abs/2305.10403, 2023b. doi: 10.48550/ARXIV.2305. 10403. URL/https://doi.org/10.48550/arXiv.2305.10403.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. CoRR, abs/2309.16609, 2023a. doi: 10.48550/ARXIV.2309.16609. URL https://doi.org/10.48550/arXiv. 2309.16609 .

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. CoRR, abs/2308.14508, 2023b. doi: 10. 48550/ARXIV.2308.14508. URLhttps://doi.org/10.48550/arXiv.2308.14508.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877-1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Aljoscha Burchardt. Multidimensional quality metrics: a flexible system for assessing translation quality. In Proceedings of Translating and the Computer 35, London, UK, November 28-29 2013. Aslib. URLhttps://aclanthology.org/2013.tc-1.6.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. CoRR, abs/2308.07201, 2023. doi: 10.48550/ARXIV.2308.07201. URL https://doi.org/10. 48550/arXiv.2308.07201.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1179. URL https://aclanthology.org/D14-1179

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern,

Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. CoRR, abs/2204.02311, 2022. doi: 10.48550/arXiv.2204.02311. URL https://doi.org/10.48550/arXiv.2204.02311.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instructionfinetuned language models. CoRR, abs/2210.11416, 2022. doi: 10.48550/ARXIV.2210.11416. URL/https://doi.org/10.48550/arXiv.2210.11416.

Seamless Communication, Loїc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Y. Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. Seamlessm4t-massively multilingual \& multimodal machine translation. CoRR, abs/2308.11596, 2023. doi: 10.48550/ARXIV. 2308.11596. URLhttps://doi.org/10.48550/arXiv.2308.11596.

Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loïc Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation. CoRR, abs/2207.04672, 2022. doi: 10.48550/ARXIV.2207. 04672. URLhttps://doi.org/10.48550/arXiv.2207.04672.

Michael A Covington and Joe D McFall. Cutting the gordian knot: The moving-average type-token ratio (mattr). Journal of quantitative linguistics, 17(2):94-100, 2010.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 1 feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

Liang Ding, Longyue Wang, Di Wu, Dacheng Tao, and Zhaopeng Tu. Context-aware cross-attention for non-autoregressive translation. In Donia Scott, Nuria Bel, and Chengqing Zong (eds.), Proceedings of the 28th International Conference on Computational Linguistics, pp. 4396-4402, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.389. URLhttps://aclanthology .org/2020 . coling-main. 389 .

Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via chatgpt. CoRR, abs/2304.07590, 2023. doi: 10.48550/ARXIV.2304.07590. URL https://doi.org/10. 48550/arXiv. 2304.07590

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. CoRR, abs/2305.14325,
2023a. doi: 10.48550/ARXIV.2305.14325. URL https://doi.org/10.48550/arXiv. 2305.14325 .

Zefeng Du, Wenxiang Jiao, Longyue Wang, Chenyang Lyu, Jianhui Pang, Leyang Cui, Kaiqiang Song, Derek F Wong, Shuming Shi, and Zhaopeng Tu. On extrapolation of long-text translation with large language models. 2023b.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: model alignment as prospect theoretic optimization. CoRR, abs/2402.01306, 2024. doi: 10.48550/ ARXIV.2402.01306. URL/https://doi.org/10.48550/arXiv.2402.01306.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Michael Auli, and Armand Joulin. Beyond englishcentric multilingual machine translation. J. Mach. Learn. Res., 22:107:1-107:48, 2021. URL http://jmlr.org/papers/v22/20-1307.html.

Yukun Feng, Feng Li, Ziang Song, Boyuan Zheng, and Philipp Koehn. Learn to remember: Transformer with recurrent memory for document-level machine translation. In Marine Carpuat, MarieCatherine de Marneffe, and Ivan Vladimir Meza Ruiz (eds.), Findings of the Association for Computational Linguistics: NAACL 2022, pp. 1409-1420, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-naacl.105. URL https://aclanthology.org/2022.findings-naacl.105.

Markus Freitag, David Grangier, and Isaac Caswell. BLEU might be guilty but references are not innocent. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 61-71, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. emnlp-main.5. URLhttps://aclanthology.org/2020.emnlp-main. 5 .

Markus Freitag, George Foster, David Grangier, Viresh Ratnakar, Qijun Tan, and Wolfgang Macherey. Experts, errors, and context: A large-scale study of human evaluation for machine translation. Transactions of the Association for Computational Linguistics, 9:1460-1474, 2021. doi: 10.1162/tacl_a_00437. URLhttps://aclanthology.org/2021.tacl-1.87.

Markus Freitag, Ricardo Rei, Nitika Mathur, Chi-kiu Lo, Craig Stewart, Eleftherios Avramidis, Tom Kocmi, George Foster, Alon Lavie, and André F. T. Martins. Results of WMT22 metrics shared task: Stop using BLEU - neural metrics are better and more robust. In Philipp Koehn, Loïc Barrault, Ondřej Bojar, Fethi Bougares, Rajen Chatterjee, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Alexander Fraser, Markus Freitag, Yvette Graham, Roman Grundkiewicz, Paco Guzman, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Tom Kocmi, André Martins, Makoto Morishita, Christof Monz, Masaaki Nagata, Toshiaki Nakazawa, Matteo Negri, Aurélie Névéol, Mariana Neves, Martin Popel, Marco Turchi, and Marcos Zampieri (eds.), Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 46-68, Abu Dhabi, United Arab Emirates (Hybrid), December 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.wmt-1.2.

Markus Freitag, Nitika Mathur, Chi-kiu Lo, Eleftherios Avramidis, Ricardo Rei, Brian Thompson, Tom Kocmi, Frederic Blain, Daniel Deutsch, Craig Stewart, Chrysoula Zerva, Sheila Castilho, Alon Lavie, and George Foster. Results of WMT23 metrics shared task: Metrics might be guilty but references are not innocent. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 578-628, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.51. URLhttps://aclanthology.org/2023.wmt-1.51

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 1243-1252. PMLR, 2017. URLhttp://proceedings.mlr.press/v70/gehring17a.html.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 6112-6121, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1633. URL https://aclanthology.org/ D19-1633

Jiatao Gu, James Bradbury, Caiming Xiong, Victor O. K. Li, and Richard Socher. Nonautoregressive neural machine translation. CoRR, abs/1711.02281, 2017. URL http:// arxiv.org/abs/1711.02281

Jiatao Gu, Yong Wang, Yun Chen, Victor O. K. Li, and Kyunghyun Cho. Meta-learning for lowresource neural machine translation. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3622-3631, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1398. URL https://aclanthology . org/D18-1398

Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 11179-11189, 2019a. URL https://proceedings.neurips.cc/paper/ 2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html.

Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 11179-11189, 2019b. URL https://proceedings.neurips.cc/paper/ 2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html.

Nuno Miguel Guerreiro, Ricardo Rei, Daan van Stigt, Luísa Coheur, Pierre Colombo, and André F. T. Martins. xcomet: Transparent machine translation evaluation through fine-grained error detection. CoRR, abs/2310.10482, 2023. doi: 10.48550/ARXIV.2310.10482. URL https: //doi.org/10.48550/arXiv.2310.10482.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. CoRR, abs/2402.01680, 2024. doi: 10.48550/ARXIV.2402.01680. URL https: //doi.org/10.48550/arXiv.2402.01680.

Barry Haddow, Rachel Bawden, Antonio Valerio Miceli Barone, Jindrich Helcl, and Alexandra Birch. Survey of low-resource machine translation. Comput. Linguistics, 48(3):673-732, 2022. doi: 10.1162/COLI \A\_00446. URLhttps://doi.org/10.1162/coli_a_00446

Joey Hejna, Rafael Rafailov, Harshit Sikchi, Chelsea Finn, Scott Niekum, W. Bradley Knox, and Dorsa Sadigh. Contrastive preference learning: Learning from human feedback without RL. CoRR, abs/2310.13639, 2023. doi: 10.48550/ARXIV.2310.13639. URLhttps://doi.org/ $10.48550 / a r X i v .2310 .13639$

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URLhttps://openreview.net/forum?id=d7KBjmI3GmQ

Christian Herold and Hermann Ney. Improving long context document-level machine translation. In Michael Strube, Chloe Braud, Christian Hardmeier, Junyi Jessy Li, Sharid Loaiciga, and Amir Zeldes (eds.), Proceedings of the 4th Workshop on Computational Approaches to Discourse (CODI 2023), pp. 112-125, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.codi-1.15. URL https://aclanthology.org/2023. codi-1.15

Jiwoo Hong, Noah Lee, and James Thorne. ORPO: monolithic preference optimization without reference model. CoRR, abs/2403.07691, 2024. doi: 10.48550/ARXIV.2403.07691. URL https://doi.org/10.48550/arXiv.2403.07691.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, and Chenglin Wu. Metagpt: Meta programming for multi-agent collaborative framework. CoRR, abs/2308.00352, 2023. doi: 10.48550/ARXIV.2308.00352. URL https://doi.org/10.48550/arXiv. 2308.00352 .

Tom Hosking, Phil Blunsom, and Max Bartolo. Human feedback is not gold standard. CoRR, abs/2309.16349, 2023. doi: 10.48550/ARXIV.2309.16349. URL https://doi.org/10. 48550/arXiv.2309.16349.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URLhttps://openreview.net/forum?id=nZeVKeeFYf9.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. Followbench: A multi-level fine-grained constraints following benchmark for large language models. CoRR, abs/2310.20410, 2023. doi: 10.48550/ARXIV. 2310.20410. URLhttps://doi.org/10.48550/arXiv.2310.20410.

Juraj Juraska, Mara Finkelstein, Daniel Deutsch, Aditya Siddhant, Mehdi Mirzazadeh, and Markus Freitag. MetricX-23: The Google submission to the WMT 2023 metrics shared task. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 756-767, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.63. URL https://aclanthology. org/2023.wmt-1.63.

Jeremy Klemin. The last frontier of machine translation. The Atlantic, 2024. URL https://www.theatlantic.com/technology/archive/2024/01/ literary-translation-artificial-intelligence/677038/.

Tom Kocmi and Christian Federmann. GEMBA-MQM: Detecting translation quality error spans with GPT-4. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 768-775, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.64. URL https://aclanthology.org/2023.wmt-1.64.

Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, Barry Haddow, Philipp Koehn, Benjamin Marie, Christof Monz, Makoto Morishita, Kenton Murray, Makoto Nagata, Toshiaki Nakazawa, Martin Popel, Maja Popović, and Mariya Shmatova. Findings of the 2023 conference on machine translation (WMT23): LLMs are here but not quite there yet. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 1-42, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.1. URL https://aclanthology.org/2023.wmt-1.1.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation. CoRR, abs/2305.15011, 2023a. doi: 10.48550/ARXIV.2305.15011. URL https://doi.org/10.48550/arXiv. 2305.15011 .

Nian Li, Chen Gao, Yong Li, and Qingmin Liao. Large language model-empowered agents for simulating macroeconomic activities. CoRR, abs/2310.10436, 2023b. doi: 10.48550/ARXIV. 2310.10436. URLhttps://doi.org/10.48550/arXiv.2310.10436.

Pengfei Li, Liangyou Li, Meng Zhang, Minghao Wu, and Qun Liu. Universal conditional masked language pre-training for neural machine translation. In Smaranda Muresan, Preslav Nakov,
and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6379-6391, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.442. URL https://aclanthology.org/2022.acl-long.442.

Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060, 2024.

Yingji Li, Mengnan Du, Rui Song, Xin Wang, and Ying Wang. A survey on fairness in large language models. CoRR, abs/2308.10149, 2023c. doi: 10.48550/ARXIV.2308.10149. URL https://doi.org/10.48550/arXiv.2308.10149.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert Yüksekgönül, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models. CoRR, abs/2211.09110, 2022. doi: 10.48550/ARXIV.2211.09110. URL https: //doi.org/10.48550/arXiv.2211.09110.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multiagent debate. CoRR, abs/2305.19118, 2023. doi: 10.48550/ARXIV.2305.19118. URL https: //doi.org/10.48550/arXiv.2305.19118.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, KwangTing Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. CoRR, abs/2402.09353, 2024. doi: 10.48550/ARXIV.2402.09353. URL https://doi.org/10. $48550 / a r X i v .2402 .09353$.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726-742, 2020. doi: 10.1162/tacl_a_00343. URLhttps://aclanthology.org/2020.tacl-1.47

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 22631-22648. PMLR, 2023. URL https://proceedings.mlr.press/v202/longpre23a.html

Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Haoran Yang, Wai Lam, and Furu Wei. Chain-ofdictionary prompting elicits translation in large language models. CoRR, abs/2305.06575, 2023. doi: 10.48550/ARXIV.2305.06575. URL https://doi.org/10.48550/arXiv. 2305. 06575

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. CoRR, abs/2306.08568, 2023. doi: 10.48550/ARXIV.2306.08568. URL https://doi.org/10.48550/arXiv.2306.08568.

Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. CoRR, abs/2306.09093, 2023. doi: 10.48550/ARXIV.2306.09093. URL https://doi.org/10.48550/arXiv.2306.09093.

Chenyang Lyu, Minghao Wu, and Alham Fikri Aji. Beyond probabilities: Unveiling the misalignment in evaluating large language models. CoRR, abs/2402.13887, 2024. doi: 10.48550/ARXIV. 2402.13887. URLhttps://doi.org/10.48550/arXiv.2402.13887.

Zhao Mandi, Shreeya Jain, and Shuran Song. Roco: Dialectic multi-robot collaboration with large language models. CoRR, abs/2307.04738, 2023. doi: 10.48550/ARXIV.2307.04738. URL https://doi.org/10.48550/arXiv.2307.04738.

Philip M McCarthy and Scott Jarvis. Mtld, vocd-d, and hd-d: A validation study of sophisticated approaches to lexical diversity assessment. Behavior research methods, 42(2):381-392, 2010.

Gabriel Mukobi, Hannah Erlebach, Niklas Lauffer, Lewis Hammond, Alan Chan, and Jesse Clifton. Welfare diplomacy: Benchmarking language model cooperation. CoRR, abs/2310.08901, 2023. doi: 10.48550/ARXIV.2310.08901. URL https://doi.org/10.48550/arXiv. 2310. 08901

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/arXiv.2303.08774. URLhttps://doi.org/10.48550/arXiv.2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin (eds.), Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311-318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URLhttps://aclanthology.org/P02-1040.

Joon Sung Park, Lindsay Popowski, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Social simulacra: Creating populated prototypes for social computing systems. In Maneesh Agrawala, Jacob O. Wobbrock, Eytan Adar, and Vidya Setlur (eds.), The 35th Annual ACM Symposium on User Interface Software and Technology, UIST 2022, Bend, OR, USA, 29 October 2022 - 2 November 2022, pp. 74:1-74:18. ACM, 2022. doi: 10.1145/3526113.3545616. URL/https://doi.org/10.1145/3526113.3545616

Joon Sung Park, Joseph C. O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Sean Follmer, Jeff Han, Jürgen Steimle, and Nathalie Henry Riche (eds.), Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST 2023, San Francisco, CA, USA, 29 October 2023- 1 November 2023, pp. 2:1-2:22. ACM, 2023. doi: 10.1145/3586183. 3606763. URLhttps://doi.org/10.1145/3586183.3606763.

Matt Post. A call for clarity in reporting BLEU scores. In Ondřej Bojar, Rajen Chatterjee, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Christof Monz, Matteo Negri, Aurélie Névéol, Mariana Neves, Matt Post, Lucia Specia, Marco Turchi, and Karin Verspoor (eds.), Proceedings of the Third Conference on Machine Translation: Research Papers, pp. 186-191, Brussels, Belgium, October 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-6319. URL https://aclanthology.org/W18-6319

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. CoRR, abs/2307.07924, 2023. doi: 10.48550/ARXIV.2307.07924. URL https://doi.org/10.48550/arXiv. 2307 . 07924

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine
(eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023, 2023. URLhttp://papers.nips.cc/paper_files/paper/2023/hash/ a85b405ed65c6477a4fe8302b5e06ce7-A.bstract-Conference.html.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. COMET: A neural framework for MT evaluation. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 26852702, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.emnlp-main.213. URL https://aclanthology.org/2020.emnlp-main. 213.

Nathaniel Robinson, Perez Ogayo, David R. Mortensen, and Graham Neubig. ChatGPT MT: Competitive for high- (but not low-) resource languages. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 392-418, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.40. URLhttps://aclanthology.org/2023.wmt-1.40.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. BLOOM: A 176bparameter open-access multilingual language model. CoRR, abs/2211.05100, 2022. doi: 10. 48550/ARXIV.2211.05100. URLhttps://doi.org/10.48550/arXiv.2211.05100.

Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7881-7892, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main. 704. URLhttps://aclanthology.org/2020.acl-main.704.

Sheng Shen, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre, Jason Wei, Hyung Won Chung, Barret Zoph, William Fedus, Xinyun Chen, Tu Vu, Yuexin Wu, Wuyang Chen, Albert Webson, Yunxuan Li, Vincent Zhao, Hongkun Yu, Kurt Keutzer, Trevor Darrell, and Denny Zhou. Flan-moe: Scaling instruction-finetuned language models with sparse mixture of experts. CoRR, abs/2305.14705, 2023. doi: 10.48550/ARXIV.2305.14705. URL https://doi.org/10.48550/arXiv. 2305.14705 .

Tianxiao Shen, Myle Ott, Michael Auli, and Marc'Aurelio Ranzato. Mixture models for diverse machine translation: Tricks of the trade. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 915 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 5719-5728. PMLR, 2019. URL http://proceedings.mlr.press/v97/ shen19c.html.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on

Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=vAElhFcKW6.

Mingyang Song, Mao Zheng, and Xuan Luo. Counting-stars: A simple, efficient, and reasonable strategy for evaluating long-context large language models. CoRR, abs/2403.11802, 2024. doi: 10.48550/ARXIV.2403.11802. URL https://doi.org/10.48550/arXiv. 2403. 11802

Zewei Sun, Mingxuan Wang, Hao Zhou, Chengqi Zhao, Shujian Huang, Jiajun Chen, and Lei Li. Rethinking document-level neural machine translation. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Findings of the Association for Computational Linguistics: ACL 2022, pp. 3537-3548, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.279. URL https://aclanthology .org/2022 . findings-acl. 279

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In Zoubin Ghahramani, Max Welling, Corinna Cortes, Neil D. Lawrence, and Kilian Q. Weinberger (eds.), Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pp.3104-3112, 2014. URLhttps://proceedings.neurips.cc/paper/ 2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html.

Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, and Donald Metzler. UL2: unifying language learning paradigms. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URLhttps://openreview.net/pdf?id=6ruVLB727MC

Katherine Thai, Marzena Karpinska, Kalpesh Krishna, Bill Ray, Moira Inghilleri, John Wieting, and Mohit Iyyer. Exploring document-level literary machine translation with parallel paragraphs from world literature. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 98829902, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.672. URL https://aclanthology.org/ 2022.emnlp-main. 672.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023a. doi: 10.48550/ARXIV.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023b. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on

Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998-6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3£5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Rob Voigt and Dan Jurafsky. Towards a literary machine translation: The role of referential cohesion. In David Elson, Anna Kazantseva, Rada Mihalcea, and Stan Szpakowicz (eds.), Proceedings of the NAACL-HLT 2012 Workshop on Computational Linguistics for Literature, pp. 18-25, Montréal, Canada, June 2012. Association for Computational Linguistics. URL https://aclanthology.org/W12-2503.

Longyue Wang, Zhaopeng Tu, Andy Way, and Qun Liu. Exploiting cross-sentence context for neural machine translation. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2826-2831, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10. 18653/v1/D17-1301. URLhttps://aclanthology.org/D17-1301.

Longyue Wang, Chenyang Lyu, Tianbo Ji, Zhirui Zhang, Dian Yu, Shuming Shi, and Zhaopeng Tu. Document-level machine translation with large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 16646-16661, Singapore, December 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.1036. URL https: //aclanthology.org/2023.emnlp-main.1036.

Longyue Wang, Zhaopeng Tu, Yan Gu, Siyou Liu, Dian Yu, Qingsong Ma, Chenyang Lyu, Liting Zhou, Chao-Hong Liu, Yufeng Ma, Weiyu Chen, Yvette Graham, Bonnie Webber, Philipp Koehn, Andy Way, Yulin Yuan, and Shuming Shi. Findings of the WMT 2023 shared task on discourselevel literary translation: A fresh orb in the cosmos of LLMs. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 55-67, Singapore, December 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.3. URL https://aclanthology.org/2023.wmt-1.3.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085-5109, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.340. URL https://aclanthology.org/2022.emnlp-main. 340 .

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1348413508, Toronto, Canada, July 2023c. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.754. URLhttps://aclanthology.org/2023.acl-long.754.

Zhanyu Wang, Longyue Wang, Zhen Zhao, Minghao Wu, Chenyang Lyu, Huayang Li, Deng Cai, Luping Zhou, Shuming Shi, and Zhaopeng Tu. Gpt4video: A unified multimodal large language model for lnstruction-followed understanding and safety-aware generation. CoRR, abs/2311.16511, 2023d. doi: 10.48550/ARXIV.2311.16511. URL/https://doi.org/10. $48550 / a r X i v .2311 .16511$.

Zihao Wang, Shaofei Cai, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. CoRR, abs/2302.01560, 2023e. doi: 10.48550/ARXIV.2302.01560. URLhttps://doi.org/ $10.48550 / a r X i v .2302 .01560$

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URLhttps://openreview.net/forum?id= gEZrGCozdqR.

Michael J. Wooldridge and Nicholas R. Jennings. Intelligent agents: theory and practice. Knowl. Eng. Rev., 10(2):115-152, 1995. doi: 10.1017/S0269888900008122. URL https://doi. org/10.1017/S0269888900008122.

Minghao Wu and Alham Fikri Aji. Style over substance: Evaluation biases for large language models. CoRR, abs/2307.03025, 2023. doi: 10.48550/ARXIV.2307.03025. URL https:// doi.org/10.48550/arXiv.2307.03025.

Minghao Wu, Yitong Li, Meng Zhang, Liangyou Li, Gholamreza Haffari, and Qun Liu. Uncertaintyaware balancing for multilingual and multi-domain neural machine translation training. In MarieFrancine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7291-7305, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.580. URL https://aclanthology .org/ 2021.emnlp-main. 580 .

Minghao Wu, George Foster, Lizhen Qu, and Gholamreza Haffari. Document flattening: Beyond concatenating context for document-level neural machine translation. In Andreas Vlachos and Isabelle Augenstein (eds.), Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 448-462, Dubrovnik, Croatia, May 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.33. URL https://aclanthology.org/2023.eacl-main.33.

Minghao Wu, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, and Alham Fikri Aji. Lamini-lm: A diverse herd of distilled models from large-scale instructions. CoRR, abs/2304.14402, 2023b. doi: 10.48550/ARXIV.2304.14402. URL/https://doi.org/10 . $48550 / a r X i v .2304 .14402$.

Minghao Wu, Thuy-Trang Vu, Lizhen Qu, George F. Foster, and Gholamreza Haffari. Adapting large language models for document-level machine translation. CoRR, abs/2401.06468, 2024a. doi: 10.48550/ARXIV.2401.06468. URL https://doi.org/10.48550/arXiv. 2401. 06468

Minghao Wu, Yufei Wang, George Foster, Lizhen Qu, and Gholamreza Haffari. Importance-aware data augmentation for document-level neural machine translation. In Yvette Graham and Matthew Purver (eds.), Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 740-752, St. Julian's, Malta, March 2024b. Association for Computational Linguistics. URL https://aclanthology .org/ 2024.eacl-long.44.

Yuhao Xie, Zongyao Li, Zhanglin Wu, Daimeng Wei, Xiaoyu Chen, Zhiqiang Rao, Shaojun Li, Hengchao Shang, Jiaxin Guo, Lizhi Lei, Hao Yang, and Yanfei Jiang. HW-TSC's submissions to the WMT23 discourse-level literary translation shared task. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 302-306, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.32. URLhttps://aclanthology.org/2023.wmt-1.32.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. A paradigm shift in machine translation: Boosting translation performance of large language models. CoRR, abs/2309.11674, 2023a. doi: 10.48550/ARXIV.2309.11674. URL https://doi.org/10.48550/arXiv. 2309.11674 .

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of LLM performance in machine translation. CoRR, abs/2401.08417, 2024. doi: 10.48550/ARXIV. 2401.08417. URLhttps://doi.org/10.48550/arXiv.2401.08417.

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. CoRR, abs/2309.04658, 2023b. doi: 10.48550/ARXIV.2309.04658. URL https: //doi.org/10.48550/arXiv.2309.04658.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URLhttps://openreview.net/pdf?id=WE_vluYUL-X.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. CoRR, abs/2309.05653, 2023. doi: 10.48550/ARXIV.2309.05653. URL/https://doi.org/10. 48550 /arXiv.2309.05653.

Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. CoRR, abs/2307.02485, 2023a. doi: 10.48550/ARXIV.2307.02485. URL https://doi.org/10.48550/arXiv.2307.02485.

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models. CoRR, abs/2306.10968, 2023b. doi: 10.48550/ARXIV.2306.10968. URL https://doi.org/10.48550/arXiv.2306.10968.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. Siren's song in the AI ocean: A survey on hallucination in large language models. CoRR, abs/2309.01219, 2023c. doi: 10.48550/ARXIV.2309.01219. URL/https://doi.org/10 . 48550/arXiv.2309.01219.

Anqi Zhao, Kaiyu Huang, Hao Yu, and Degen Huang. DUTNLP system for the WMT2023 discourse-level literary translation. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz (eds.), Proceedings of the Eighth Conference on Machine Translation, pp. 296-301, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1. 31. URLhttps://aclanthology.org/2023.wmt-1.31.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/ hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_ Benchmarks.html.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. CoRR, abs/2306.05685, 2023b. doi: 10.48550/arXiv.2306.05685. URLhttps://doi.org/10.48550/arXiv. 2306.05685 .

Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. Transfer learning for low-resource neural machine translation. In Jian Su, Kevin Duh, and Xavier Carreras (eds.), Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1568-1575, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1163. URLhttps://aclanthology.org/D16-1163.


[^0]:    *Longyue Wang is the corresponding author: vinnylywang@tencent.com.

[^1]:    ${ }^{1}$ Model signature: gpt-4-1106-preview

[^2]:    ${ }^{2}$ Model signature: nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1

    ${ }^{3}$ In our preliminary study, we conduct small-scale MQM-based human evaluation and also observe that our approach, TRANSAGENTS, receives a low MQM score.

[^3]:    ${ }^{4}$ https://www.surveymonkey.com/

[^4]:    ${ }^{5}$ We initially attempt to collect responses directly from web novel forums, such as the $r$ /WebNovels subreddit on Reddit. However, this approach proves to be too slow and sometimes violates the community rules of these platforms.

[^5]:    ${ }^{6}$ We could not find the direct source of this information from the American Translators Association. Our source of information is available at https://tinyurl.com/bdze $92 \times r$. We assume that the recommended rate of $\$ 0.12$ USD per word is based on the number of words in the English language text.

</end of paper 2>


<paper 3>
# Large Language Models Meet NLP: A Survey 

Libo Qin ${ }^{\star}$ Qiguang Chen ${ }^{\wedge} \quad$ Xiachong Feng $\diamond$ Yang Wu ${ }^{\curvearrowright} \quad$ Yongheng Zhang ${ }^{\star}$<br>Yinghui Li ${ }^{\natural}$ Min Li ${ }^{\star} \quad$ Wanxiang Che ${ }^{\star} \quad$ Philip S. Yu ${ }^{\ominus}$<br>* Central South University * Harbin Institute of Technology $\diamond$ University of Hong Kong<br>${ }^{\natural}$ Tsinghua University ${ }^{\ominus}$ University of Illinons at Chicago<br>lbqin@csu.edu.cn, \{qgchen, car\}@ir. hit.edu.cn


#### Abstract

While large language models (LLMs) like ChatGPT have shown impressive capabilities in Natural Language Processing (NLP) tasks, a systematic investigation of their potential in this field remains largely unexplored. This study aims to address this gap by exploring the following questions: (1) How are LLMs currently applied to NLP tasks in the literature? (2) Have traditional NLP tasks already been solved with LLMs? (3) What is the future of the LLMs for NLP? To answer these questions, we take the first step to provide a comprehensive overview of LLMs in NLP. Specifically, we first introduce a unified taxonomy including (1) parameter-frozen application and (2) parameter-tuning application to offer a unified perspective for understanding the current progress of LLMs in NLP. Furthermore, we summarize the new frontiers and the associated challenges, aiming to inspire further groundbreaking advancements. We hope this work offers valuable insights into the potential and limitations of LLMs in NLP, while also serving as a practical guide for building effective LLMs in NLP.


## 1 Introduction

Recently, large language models (LLMs) represent a significant breakthrough in AI through scaling up language models (Zhao et al., 2023a; Kaddour et al., 2023; Yang et al.; Hadi et al., 2023; Zhuang et al., 2023). Current studies on LLMs, such as GPT-series (Brown et al., 2020; Ouyang et al., 2022), PaLM-series (Chowdhery et al., 2022), OPT (Zhang et al., 2022a), and LLaMA (Touvron et al., 2023), have shown impressive zero-shot performance. In addition, LLMs also bring some emergent abilities including instruction following (Wei et al., 2022a), chain-of-thought reasoning (Wei et al., 2022c) and in-context learning (Min et al., 2022), which attract increasing attention (Wei et al., 2022b).

![](https://cdn.mathpix.com/cropped/2024_06_04_b903597effb1e7fa33d1g-01.jpg?height=520&width=786&top_left_y=754&top_left_x=1049)

Figure 1: The example of applying LLMs for NLP tasks (e.g., mathematical reasoning, machine translation, information extraction and sentiment analysis).

With the advancement of large language models, as shown in Figure 1, LLMs allow various natural language processing (NLP) tasks (e.g., zero-shot mathematical reasoning, text summarization, machine translation, information extraction and sentiment analysis) to be achieved through a unified generative paradigm, which has achieved remarkable success (Wei et al., 2022c, 2023a; Qin et al., 2023a; Wang et al., 2023a,d,h,j; Wan et al., 2023b; Peng et al., 2023; Huang et al., 2023a). Additionally, some LLMs in NLP work without needing any additional training data and can even surpass traditional models fine-tuned with supervised learning. This advancement significantly contributes to the development of NLP literature. As a result, the community has witnessed an exponential growth of LLMs for NLP studies, which motivates us to investigate the following questions: (1) How are LLMs currently applied to NLP tasks in the literature? (2) Have traditional NLP tasks already been solved with LLMs? (3) What is the future of the LLMs for NLP?

To answer the above questions, we make the first attempt to present a comprehensive and detailed analysis on LLMs for NLP. The overarching
(a) Parameter-Frozen Paradigm
![](https://cdn.mathpix.com/cropped/2024_06_04_b903597effb1e7fa33d1g-02.jpg?height=326&width=1476&top_left_y=277&top_left_x=310)

Zero-shot Learning

Few-shot Learning

(b) Parameter-Tuning Paradigm
![](https://cdn.mathpix.com/cropped/2024_06_04_b903597effb1e7fa33d1g-02.jpg?height=298&width=1468&top_left_y=654&top_left_x=317)

Parameter-efficient Tuning

Full-parameter Tuning

Figure 2: The taxonomy of LLMs for NLP, including parameter-frozen (a) and parameter-tuning paradigm (b), where blue module with ice denotes that the parameters are kept unchanged, and orange module with fire represents the fine-tuning of full or selected parameters.

goal of this work is to explore the current developments in LLMs for NLP. To this end, in this paper, we first introduce the relevant background and preliminary. Furthermore, we introduce a unified paradigm on LLMs for NLP: (1) parameter-frozen application including (i) zero-shot learning and (ii) few-shot learning; (2) parameter-tuning application containing (i) full-parameter tuning and (ii) parameter-efficient tuning, aiming to provide a unified perspective to understand the current progress of LLMs for NLP:

- Parameter-frozen application directly applies prompting approach on LLM for NLP tasks without the need for parameter tuning. This category includes zero-shot and few-shot learning, depending on whether the few-shot demonstrations is required.
- Parameter-tuning application refers to the need for tuning parameters of LLMs for NLP tasks. This category includes both fullparameter and parameter-efficient tuning, depending on whether fine-tuning is required for all model parameters.

Finally, we conclude by identifying potential frontier areas for future research, along with the associated challenges to stimulate further exploration.

In summary, this work offers the following contributions:

(1) First survey: We present the first comprehensive survey of Large Language Models
(LLMs) for Natural Language Processing (NLP) tasks.

(2) New taxonomy: We introduce a new taxonomy including (1) parameter-frozen application and (2) parameter-tuning application, which provides a unified view to understand LLMs for NLP tasks.

(3) New frontiers: We discuss emerging areas of research in LLMs for NLP and highlight the challenges associated with them, aiming to inspire future breakthroughs.

(4) Abundant resources: We create the first curated collection of LLM resources for NLP, including open-source implementations, relevant corpora, and a list of research papers. These resources are available at https://github.com/LightChen233/ Awesome-LLM-for-NLP.

We expect this work will be a valuable resource for researchers and spur further advancements in the field of LLM-based NLP.

## 2 Background

As shown in Figure 2, this section describes the background of parameter-frozen paradigm (\$2.1) and parameter-tuning paradigm ( $\$ 2.2$ ).

### 2.1 Parameter-Frozen Paradigm

Parameter-frozen paradigm can directly apply prompting for NLP tasks without any parameter
tuning. As shown in Figure 2 (a), this category encompasses zero-shot learning and few-shot learning (Brown et al., 2020; Kojima et al., 2022).

Zero-shot Learning In zero-shot learning, LLMs leverage the instruction following capabilities to solve NLP tasks based on a given instruction prompt, which is defined as:

$$
\begin{equation*}
\mathcal{P}=\operatorname{Prompt}(\mathcal{I}) \tag{1}
\end{equation*}
$$

where $\mathcal{I}$ and $\mathcal{P}$ denote the input and output of prompting, respectively.

Few-shot Learning Few-shot learning uses incontext learning capabilities to solve the NLP tasks imitating few-shot demonstrations. Formally, given some demonstrations $\mathcal{E}$, the process of few-shot learning is defined as:

$$
\begin{equation*}
\mathcal{P}=\operatorname{Prompt}(\mathcal{E}, \mathcal{I}) \tag{2}
\end{equation*}
$$

### 2.2 Parameter-Tuning Paradigm

As shown in Figure 2 (b), the parameter-tuning paradigm involves adjusting LLM parameters for NLP tasks, covering both full-parameter and parameter-efficient tuning.

Full-parameter Tuning In the full-parameter tuning approach, all parameters of the model $\mathcal{M}$ are fine-tuned on the training dataset $\mathcal{D}$ :

$$
\begin{equation*}
\hat{\mathcal{M}}=\operatorname{Fine-tune}(\mathcal{M} \mid \mathcal{D}) \tag{3}
\end{equation*}
$$

where $\hat{\mathcal{M}}$ is the fine-tuned model with the updated parameters.

Parameter-efficient Tuning Parameter-efficient tuning (PET) involves adjusting a set of existing parameters or incorporating additional tunable parameters (like Bottleneck Adapter (Houlsby et al., 2019), Low-Rank Adaptation (LoRA) (Hu et al., 2021), Prefix-tuning (Li and Liang, 2021a), and QLoRA (Dettmers et al., 2023)) to efficiently adapt models for specific NLP tasks. Formally, parameter-efficient tuning first tunes a set of parameters $\mathcal{W}$, denoting as:

$$
\begin{equation*}
\hat{\mathcal{W}}=\text { Fine-tune }(\mathcal{W} \mid \mathcal{D}, \mathcal{M}) \tag{4}
\end{equation*}
$$

where $\hat{\mathcal{W}}$ stands for the trained parameters.

## 3 Natural Language Understanding

As shown in Figure 3, we first describe some typical NLP understanding tasks, which consists of Semantic Analysis (\$3.1), Information Extraction (§3.2), Dialogue Understanding (§3.3), and Table Understanding (§3.4).

### 3.1 Sentiment Analysis

Sentiment analysis, a key function in natural language processing, identifies the emotional tone of a text, like positive opinions or criticisms (Wankhade et al., 2022).

### 3.1.1 Parameter-Frozen Paradigm

Zero-shot Learning With the help of instruction tuning, LLMs have been equipped with excellent zero-shot learning ability (Belkhir and Sadat, 2023). Recent studies (Zhang et al., 2023g) find that using simple instructions can elicit ChatGPT's strong capabilities on a series of sentiment analysis tasks such as sentiment classification and aspect-based sentiment analysis. Moreover, current mainstream LLMs (Koto et al., 2024) possess the ability of multilingual understanding to analyze the sentiment conveyed by different languages based on sentiment lexicons (Koto et al., 2024).

Few-shot Learning Few-shot prompting not only elicits in-context learning in LLMs but also elaborates the intent of users more clearly. According to the findings presented by previous studies (Zhang et al., 2023g; Zhao et al., 2023b; $\mathrm{Xu}$ et al., 2023c), incorporating exemplars to the prompts significantly boosts LLMs' performance on aspect-based sentiment analysis and emotion recognition tasks. Furthermore, Sun et al. (2023b) introduce few-shot learning on more complex procedures, incorporating multi-LLM negotiation framework for sentiment analysis.

### 3.1.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-parameter instruction tuning has been shown to be an effective approach to bridge the gap between task-agnostic pre-training and task-specific inference. Specifically, Wang et al. (2022) design unified sentiment instruction for various aspect-based sentiment analysis tasks to elicit the LLMs. Varia et al. (2022) utilize task-specific sentiment instructions to finetune LLMs for the inter-task dependency. Yang and $\mathrm{Li}$ (2023) transform the visual input into plain text during prompt construction for instruction tuning. These works demonstrate the potential of tuning LLMs for advanced sentiment analysis.

Parameter-Efficient Tuning Sentiment analysis techniques have numerous real-world applications such as opinion mining (Zhao et al., 2016). Therefore, efficiency is a vital dimension for evaluating

![](https://cdn.mathpix.com/cropped/2024_06_04_b903597effb1e7fa33d1g-04.jpg?height=1188&width=1585&top_left_y=217&top_left_x=241)

Figure 3: Taxonomy of LLMs for NLP including Parameter-Frozen Paradigm and Parameter-Tuning Paradigm.

sentiment analysis methods. Qiu et al. (2023) utilize LoRA to tune LLMs on the empathy multi-turn conversation dataset namely SMILECHAT to develop emotional support systems.

### 3.2 Information Extraction

Information Extraction (IE) tasks aim at extracting structural information from plain text, which typically includes relation extraction (RE), named entity recognition (NER), and event extraction (EE) (Xu et al., 2023a).

### 3.2.1 Parameter-Frozen Paradigm

Zero-shot Learning Inspired by the impressive capabilities of LLMs on various tasks, recent studies (Zhang et al., 2023c; Wei et al., 2023a) begin to explore zero-shot prompting methods to solve IE tasks by leveraging knowledge embedded in LLMs Wei et al. (2023a), Xie et al. (2023) and Zhang et al. (2023c) propose a series of methods to decompose question-answering tasks by breaking down NER into smaller, simpler subproblems, which improves the overall process. In addition, Xie et al. (2023) further introduce two methods, syntactic prompting and tool augmentation, to improve LLMs' perfor- mance by incorporating the syntactic information.

Few-shot Learning Considering the gap between sequence labeling and text generation, providing exemplars could help LLMs better understand the given task and follow the problem-solving steps. To select pertinent demonstrations, $\mathrm{Li}$ and Zhang (2023) deploy the retrieval module to retrieve the most suitable examples for the given test sentence. Instead of using natural language for structured output, Li et al. (2023e) and Bi et al. (2023) propose reformulating IE tasks as code with code-related LLMs such as Codex.

### 3.2.2 Parameter-Tuning Paradigm

Full-Parameter Tuning A common practice to customize LLMs is fine-tuning LLMs on the collected dataset. There typically are three tuning paradigms adopted to enhance LLMs' abilities. The first one is tuning LLMs on a single dataset to strengthen a specific ability. The second one is standardizing data formats across all IE subtasks, thus enabling a single model to efficiently handle diverse tasks (Lu et al., 2023a; Gan et al., 2023). The last one is tuning LLMs on a mixed dataset
and testing on the unseen tasks (Sainz et al., 2023; Wang et al., 2023f), which is always used to improve the generalization ability of LLMs.

Parameter-Efficient Tuning Tuning huge parameters of LLMs poses a significant challenge to both research and development. To address this challenge, Das et al. (2023b) propose a method for dynamic sparse fine-tuning that focuses on a specific subset of parameters during the IE training process. This approach is particularly useful when dealing with limited data. Meanwhile, Liang et al. (2023) introduce Lottery Prompt Tuning (LPT), a method that efficiently tunes only a portion of the prompt vectors used for lifelong information extraction. This technique optimizes both parameter efficiency and deployment efficiency.

### 3.3 Dialogue Understanding

Dialogue understanding typically consists of spoken language understanding (SLU) (Tur and De Mori, 2011; Qin et al., 2019, 2021) and dialogue state tracking (DST) (Sarikaya et al., 2016; Jacqmin et al., 2022).

### 3.3.1 Parameter-Frozen Paradigm

Zero-shot Learning Recent studies highlight the effectiveness of LLMs in dialogue understanding through zero-shot prompting (Pan et al., 2023; He and Garner, 2023; Hudeček and Dušek, 2023; Heck et al., 2023). Gao et al. (2023a) and Addlesee et al. (2023) introduce zero-shot chain-ofthought prompting strategies in LLMs, enhancing understanding by step-by-step reasoning. Moreover, Zhang et al. (2023i) and Wu et al. (2023c) treat SLU and DST as agent systems and code generation tasks to effectively improve task performance. Further, Chung et al. (2023), Chi et al. (2023) and Zhang et al. (2023h) extend the task to actual scenarios and understand the dialog by zero-shot prompting for efficient interaction and dialog management.

Few-shot Learning Limited by the instruction following ability of the LLMs, recent studies have focused on improving model performance in dialogue understanding through the relevant few-shot demonstrations (Hudeček and Dušek, 2023). To address "overfitting" in the given few-shot demonstrations, Hu et al. (2022b), King and Flanigan (2023), Das et al. (2023a), Li et al. (2022b), Lee et al. (2023), King and Flanigan (2023) and Addlesee et al. (2023) further introduce some methods for retrieving diverse few-shot demonstrations to improve understanding performance. Lin et al. (2023) and Cao (2023) integrate DST tasks with an agent through in-context-learning, enhancing dialogue understanding capabilities.

### 3.3.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-parameter tuning involves not freezing any parameters and using all parameters to train dialogue understanding tasks (Yu et al., 2022). Specifically, Xie et al. (2022); Zhao et al. (2022a) unifies structured tasks into a textual format by training full parameters demonstrating significant improvement and generalization. Gupta et al. (2022) utilize input with some demonstrations as a new DST representation format to train LLM with full parameters and achieve great results.

Parameter-Efficient Tuning Limited by the huge cost of full-parameter fine-tuning, a lot of work begins to focus more on Parameter-Efficient Tuning (PET) for lower-cost dialogue understanding task training. Specifically, Feng et al. (2023b) present LDST, a LLaMA-driven DST framework that leverages LoRA technology for parameterefficient fine-tuning, achieving performance comparable to ChatGPT. Liu et al. (2023b) provide a key-value pair soft-prompt pool, selecting softprompts from the prompting pool based on the conversation history for better PET.

### 3.4 Table Understanding

Table understanding involves the comprehension and analysis of structured data presented in tables, focusing on interpreting and extracting meaningful information, like Table Question Answering (Jin et al., 2022).

### 3.4.1 Parameter-Frozen Paradigm

Zero-shot Learning Recently, the advancements for LLMs have paved the way for exploring zeroshot learning capabilities in understanding and interpreting tabular data (Singha et al., 2023; Patnaik et al., 2024; Ye et al., 2024). Ye et al. (2023) and Sui et al. (2023a) concentrate on breaking down large tables into smaller segments to reduce irrelevant data interference during table understanding. Further, Patnaik et al. (2024) introduce CABINET, a framework that includes a module for generating parsing statements to emphasize the data related to a given question. Sui et al. (2023b) develop TAP4LLM, enhancing LLMs' table understanding
abilities by incorporating reliable information from external knowledge sources into prompts. Additionally, Ye et al. (2024) propose a DataFrameQA framework to utilize secure Pandas queries to address issues of data leakage in table understanding. These efforts signify a significant stride towards leveraging LLMs for more effective and efficient zero-shot learning in table data comprehension.

Few-shot Learning Few-shot learning has been an increasingly focal point for researchers to address the limitations of LLMs, particularly in the context of table understanding and instruction following ability (Chen, 2023; Zhang et al., 2024). Luo et al. (2023b) propose a hybrid prompt strategy coupled with a retrieval-of-thought to further improve the example quality for table understanding tasks. Cheng et al. (2022) introduce Binder to redefine the table understanding task as a coding task, enabling the execution of code to derive answers directly from tables. Furthermore, $\mathrm{Li}$ et al. (2023b), Jiang et al. (2023) and Zhang et al. (2023k,f) conceptualize the table understanding as a more complex agent task, which utilizes external tools to augment LLMs in table tasks. Building upon these developments, ReAcTable (Zhang et al., 2023j) integrates additional actions into the process, such as generating SQL queries, producing Python code, and directly answering questions, thereby further enriching the few-shot learning landscape for LLMs.

### 3.4.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Leveraging the existing capabilities of LLMs, Full-Parameter Tuning optimizes these models for specific table understanding tasks. Li et al. (2023d) and Xie et al. (2022) adapt a substantial volume of table-related data for table instruction tuning, which leads to better generalization in table understanding tasks. Additionally, Xue et al. (2023) introduce DB-GPT to enhance LLMs by fine-tuning them and integrating a retrieval-augmented generation component to better support table understanding.

Parameter-Efficient Tuning Xie et al. (2022) utilize prompt-tuning for efficient fine-tuning within a unified framework of table representation instructions. Moreover, Zhang et al. (2023a), Zhu et al. (2024) and Bai et al. (2023) adapt Low-Rank Adaptation (LoRA) during instruction-tuning for better table understanding and further table cleaning. Furthermore, Zhang et al. (2023d) address challenges related to long table inputs by implementing LongLoRA, demonstrating its efficacy in managing long-context issues in table understanding tasks.

## 4 Natural Language Generation

This section presents the LLMs for classific NLP generation tasks containing Summarization (§4.1), Code Generation (§4.2), Machine Translation (\$4.3), and Mathematical Reasoning (§4.4), which are illustrated in Figure 3.

### 4.1 Summarization

Summarization aims to distill the most essential information from a text document, producing a concise and coherent synopsis that retains the original content's primary themes (Shi et al., 2018).

### 4.1.1 Parameter-Frozen Paradigm

Zero-shot Learning In the exploration of zeroshot learning for text summarization, LLMs such as GPT-3 have demonstrated amazing and superior performance in generating concise and factually accurate summaries, challenging the need for traditional fine-tuning approaches (Goyal et al., 2022; Bhaskar et al., 2022; Wang et al., 2023b). Zhang et al. (2023e) highlight instruction tuning as pivotal for LLMs' summarization success. Ravaut et al. (2023b) scrutinize LLMs' context utilization, identifying a bias towards initial document segments in summarization tasks. These studies collectively underscore the versatility and challenges of deploying LLMs in zero-shot summarization.

Few-shot Learning For few-shot learning, LLMs like ChatGPT are scrutinized for their summarization abilities. Zhang et al. (2023b) and Tang et al. (2023) demonstrate that leveraging in-context learning and a dialog-like approach can enhance LLMs' extractive summarization, particularly in achieving summary faithfulness. Adams et al. (2023) introduce a "Chain of Density" prompting technique, revealing a preference for denser, entityrich summaries over sparser ones. Together, these studies reveal the evolving strategies to optimize LLMs for summarization tasks.

### 4.1.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-Parameter Tuning for text summarization leverages the power of LLMs, optimizing them for specific summarization tasks. DIONYSUS (Li et al., 2022a) adapts
to new domains through a novel pre-training strategy tailored for dialogue summarization. Socratic Pretraining (Pagnoni et al., 2022) introduces a question-driven approach to improve the summarization process. This allows the model to be easily adapted for different summarization tasks, resulting in more controllable and relevant summaries.

Parameter-Efficient Tuning PET strategies have revolutionized the adaptability of large pretrained models for specific summarization tasks, demonstrating the power of fine-tuning with minimal parameter adjustments (Feng et al., 2023a). Zhao et al. (2022b) and Yuan et al. (2022) adapt prefix-tuning (Li and Liang, 2021b) for dialogue summarization, enhancing model knowledge and generalization across domains. Ravaut et al. (2023a) develop PromptSum to combine prompt tuning with discrete entity prompts for controllable abstractive summarization. These approaches collectively show the efficacy of PET in enabling robust, domain-adaptive, and controllable summarization with minimal additional computational costs.

### 4.2 Code Generation

Code generation involves the automatic creation of executable code from natural language specifications, facilitating a more intuitive interface for programming (Chen et al., 2021).

### 4.2.1 Parameter-Frozen Paradigm

Zero-shot Learning Recent advancements in code generation have been significantly propelled by the development of LLMs, with studies showcasing their proficiency in generating code in a zeroshot manner. Code LLMs, trained on both code and natural language, have a robust and amazing zeroshot learning capability for programming tasks (Nijkamp et al., 2022; Roziere et al., 2023). Moreover, CodeT5+ enriches the landscape by proposing a flexible encoder-decoder architecture and a suite of pretraining objectives, leading to notable improvements (Wang et al., 2023i). These models collectively push the boundary of what is achievable in code generation, offering promising avenues for zero-shot learning.

Few-shot Learning Code generation is being revolutionized by few-shot learning. This technique allows models to create precise code snippets by learning from just minimal examples (Lu et al., 2021). Chen et al. (2021), Allal et al. (2023), Li et al. (2023f), Luo et al. (2023c) and Christopoulou et al. (2022) illustrate the efficacy of few-shot learning, demonstrating an adeptness at code generation that surpasses its predecessors. The development of smaller, yet powerful models (Li et al., 2023g; Guo et al., 2024), further highlights accessibility of few-shot code generation technologies, making them indispensable tools in the arsenal of modern developers.

### 4.2.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-parameter tuning represents a pivotal strategy in enhancing code generation models, allowing comprehensive model optimization. Specifically, CodeT series (Wang et al., 2021, 2023i) epitomize this approach by incorporating code-specific pre-training tasks and architecture flexibility, respectively, to excel in both code understanding and generation. CodeRL (Le et al., 2022) and PPOCoder (Shojaee et al., 2023) introduce deep reinforcement learning, leveraging compiler feedback and execution-based strategies for model refinement, whereas StepCoder (Shojaee et al., 2023) advances this further by employing reinforcement learning, curriculum learning and finegrained optimization techniques. These models collectively demonstrate significant improvements across a spectrum of code-related tasks, embodying the evolution of AI-driven programming aids.

Parameter-Efficient Tuning PET emerges as a pivotal adaptation in code tasks, striking a balance between performance and computational efficiency (Weyssow et al., 2023). Studies (Ayupov and Chirkova, 2022; Zhuo et al., 2024) exploring adapters and LoRA showcase PET's viability on code understanding and generation tasks, albeit with limitations in generative performance.

### 4.3 Machine Translation

Machine translation is a classical task that utilize computers to automatically translate the given information from one language to another, striving for accuracy and preserving the semantic essence of the original material (Bahdanau et al., 2014).

### 4.3.1 Parameter-Frozen Paradigm

Zero-shot Learning In the realm of zero-shot learning, Zhu et al. (2023a) and Wei et al. (2023b) enhance LLMs' multilingual performance through cross-lingual and multilingual instruction-tuning, significantly improving translation tasks. OpenBA contributes to the bilingual model space, demonstrating superior performance in Chinese-oriented
tasks with a novel architecture (Li et al., 2023c). These advancements highlight the potential of LLMs in aligning language in zero-shot settings.

Few-shot Learning In the exploration of fewshot learning for machine translation (MT), recent studies present innovative strategies to enhance the capabilities of LLMs (Li et al., 2023a; Huang et al., 2024). Lu et al. (2023b) introduce Chain-ofDictionary Prompting $(\mathrm{CoD})$ to improve the MT of rare words by in-context-learning in low-resource languages. Raunak et al. (2023) investigate the impact of demonstration attributes on in-context learning, revealing the critical role of output text distribution in translation quality. Together, these works illustrate the significant potential of few-shot learning and in-context strategies in advancing the field of MT with LLMs.

### 4.3.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-parameter tuning in machine translation with LLMs represents a frontier for enhancing translation accuracy and adaptability (Xu et al., 2023b). Iyer et al. (2023) demonstrate the potential of LLMs in disambiguating polysemous words through in-context learning and fine-tuning on ambiguous datasets, achieving superior performance in multiple languages. Moslem et al. (2023) and Wu et al. (2024) focus on exploring fine-tuning methods that enhance realtime and context-aware translation capabilities. $\mathrm{Xu}$ et al. (2024) propose Contrastive Preference Optimization (CPO) to refine translation quality further, pushing LLMs towards better performance. These studies reveal the efficacy and necessity of finetuning approaches in realizing the full potential of LLMs for complex machine translation tasks.

Parameter-Efficient Tuning PET is emerging as a transformative approach for integrating LLMs into machine translation (MT), balancing performance and efficiency. Ustun and Stickland (2022) empirically assess PET's efficacy across different languages and model sizes, highlighting adapters' effectiveness with adequate parameter budgets. Alves et al. (2023) optimize the finetuning process with adapters, striking a balance between few-shot learning and finetuning efficiency. These studies collectively underline PET's potential to revolutionize MT by making LLMs more adaptable and resource-efficient.

### 4.4 Mathematical Reasoning

Mathematical reasoning tasks in NLP involve the use of NLP techniques to understand information from mathematical text, perform logical reasoning, and generate answers (Lu et al., 2023e).

### 4.4.1 Parameter-Frozen Paradigm

Zero-shot Learning Mathematics serves as a testbed to investigate the reasoning capabilities of LLMs (OpenAI, 2023; Touvron et al., 2023). The vanilla prompting method asks LLMs to directly arrive at the final answer to a given mathematical problem. It is very challenging and the reasoning process is not transparent to humans. To address it, Kojima et al. (2022) develop a zeroshot chain-of-thought technique, which utilizes the simple prompt "Let's think step by step" to elicit mathematical reasoning in LLMs. By doing this, the LLM can break down the problem into smaller, easier-to-solve pieces before arriving at a final answer. Further, Wang et al. (2023g) propose a new decoding strategy, called self-consistency. This approach integrates a series of prompting results to boost the mathematical performance.

Few-shot Learning Recent studies explore constructing more suitable exemplars for LLMs to improve mathematical reasoning. Wei et al. (2022c) introduce chain-of-thought prompting, which presents a few chain-of-thought demonstrations to teach LLMs to think step by step. However, manually constructing the demonstrations in fewshot learning is time- and labor-consuming. To solve this problem, Zhang et al. (2022b) and $\mathrm{Lu}$ et al. (2023d) propose to select in-context examples automatically. Even given detailed examples, it is still hard for LLMs to calculate the numbers precisely. To address this issue, PAL (Gao et al., 2023b) directly generates programs as intermediate reasoning steps. These programs are then executed using a runtime environment, like a Python interpreter, to find the better and robust solution.

### 4.4.2 Parameter-Tuning Paradigm

Full-Parameter Tuning Full-parameter tuning is a common way to specify LLMs' behaviors on mathematical reasoning tasks. Luo et al. (2023a) apply their proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math to improve the mathematical reasoning abilities of LLMs. Yue et al. (2023) introduce the MathInstruct dataset to enhance the general

![](https://cdn.mathpix.com/cropped/2024_06_04_b903597effb1e7fa33d1g-09.jpg?height=925&width=1374&top_left_y=226&top_left_x=338)

Figure 4: The future work and new frontier for LLM in NLP tasks.

math problem-solving ability of LLMs through indomain instruction tuning. Ho et al. (2023) teach the small language models to perform mathematical reasoning by distilling the generated intermediate rationales by large language models. Schick et al. (2023) present ToolFormer, which can use the calculator to perform simple numeric calculations when solving math problems.

Parameter-Efficient Tuning Fine-tuning LLMs with full parameter updates incurs significant memory overhead, limiting accessibility for many users. Parameter-efficient tuning techniques, such as LoRA (Hu et al., 2022a), offer a promising alternative. Additionally, Hu et al. (2023b) propose a user-friendly framework for integrating various adapters into LLMs, enabling them to tackle tasks like mathematical reasoning.

Takeaways (1) LLMs offer a unified generative solution paradigm for various NLP tasks. (2) LLMs in NLP tasks still have a certain gap from smaller supervised learning models. (3) Continuing to fine-tune LLMs on NLP tasks bring substantial improvements.

## 5 Future Work and New Frontier

In this section, as shown in Figure 4, we highlight some new frontiers, hoping to spur more breakthroughs in the future.

### 5.1 Multilingual LLMs for NLP

Despite the significant success of LLMs in English NLP tasks, there are over 7,000 languages worldwide. How to extend the success of English-centric LLMs to NLP tasks in other languages is an important research question (Qin et al., 2024). Inspired by this, recent research has increasingly focused on using multilingual LLMs to solve NLP tasks in multilingual scenarios (Xue et al., 2021; Workshop et al., 2022; Shi et al., 2022; Qin et al., 2023a; Winata et al., 2023).

Two main challenges in this direction are as follows: (1) Enhancing Low-Resource Language Performance: Due to poor performance in lowresource languages, how to build universal multilingual LLMs that achieve promising performance in NLP tasks across languages is a direction worth exploring. (2) Improving Cross-lingual Alignment: The key to multilingual LLMs is improving the alignment between English and other languages. Effectively achieving cross-lingual alignment in cross-lingual NLP tasks is a challenge.

### 5.2 Multi-modal LLMs for NLP

The current LLMs achieve excellent performance in text modality. However, integrating more modalities is one of the key ways to achieve artificial general intelligence (AGI). Therefore, a lot of work has begun to explore multi-modal LLMs for multi-
modal NLP tasks (Lu et al., 2022, 2023c; Yang et al., 2023a,b; Zhang et al., 2023l).

The primary challenges in this field are: (1) Complex Multi-modal Reasoning: Currently, most multi-modal LLMs focus on simple multi-modal reasoning, like recognition (Wang et al., 2023e; Liu et al., 2023a), while neglecting complex multimodal reansoning (Yang et al., 2023b; Lu et al., 2023c). Therefore, how to effectively explore complex multi-modal reasoning for NLP is a crucial topic. (2) Effective Multi-modal Interaction: Existing methods often simply focus on adding direct multi-modal projection or prompting to LLM for bridge multi-modality gap (Wang et al., 2023e; Liu et al., 2023a; Wu et al., 2023b; Mitra et al., 2023). Crafting a more effective multi-modal interaction mechanism in multi-modal LLMs to solve NLP tasks is an essential problem.

### 5.3 Tool-usage in LLMs for NLP

While LLMs have shown success in NLP tasks, they can still face challenges when applied in realworld scenarios (Qin et al., 2023b). Therefore, a lot of work explores utilizing LLMs as central controllers to enable the usage or construction of tools and agents to solve more practical NLP tasks (Shinn et al., 2023; Wang et al., 2023c; Zhu et al., 2023b; Hu et al., 2023a).

The primary concerns are: (1) Appropriate Tool Usage: Current works always consider static tool usage, neglecting to choose appropriate tools to use. Identifying the correct tools and using them accurately is a key issue in solving NLP tasks efficiently. (2) Efficient Tool Planning: Current works still focus on the usage of a single tool for NLP tasks. Motivated by this, there is a pressing need for NLP tasks to achieve an efficient tool chain that leverages multiple tools in a coordinated manner. For example, when facing Task-oriented Dialogue tasks, we can use three tools: booking flight tickets, booking train tickets, and booking bus tickets. Then, how to collaborate to make the trip time as short as possible and the cost as low as possible is a typical problem in effective tool planning.

### 5.4 X-of-thought in LLMs for NLP

When LLMs solve complex NLP problems, they often cannot directly give correct answers and require complex thinking. Therefore, some works adapt Xof-thought (XoT) for advanced logical reasoning. XoT primarily aims to refine logical processing for better NLP task solution (Kojima et al., 2022; Zhang et al., 2022b; Qin et al., 2023a; Yao et al., 2023; Chen et al., 2022; Lei et al., 2023).

Key challenges in this direction include: (1) Universal Step Decomposition: How to develop a method for universally applicable step decomposition to generalize LLMs to various NLP tasks is the core challenge of XoT. (2) Prompting Knowledge Integration: Diverse promptings enhance model performance across various scenarios. How to better integrate the knowledge of different XoT to solve NLP problems is an important direction.

### 5.5 Hallucination in LLMs for NLP

During solving the NLP tasks, LLMs inevitably suffer from the hallucinations where LLMs produce outputs that deviate from world knowledge (Muhlgay et al., 2023; Min et al., 2023), user request (Adlakha et al., 2023), or self-generated context (Liu et al., 2022). This deviation harms the reliability of LLMs in practical scenarios.

The primary challenges in hallucination are: (1) Efficient Hallucination Evaluation: How to find appropriate and unified evaluation benchmarks and metrics for LLMs in various NLP tasks is a key challenge. (2) Leveraging Hallucinations for Creativity: Hallucinations can often stimulate certain creative abilities. How to leverage hallucination to stimulate creativity and generate better innovative knowledge is an interesting topic.

### 5.6 Safety in LLMs for NLP

Applying large models to downstream NLP tasks also raises inevitable safety concerns, including copyright issues (Chang et al., 2023), hate toxicity (Hartvigsen et al., 2022), social bias (Wan et al., 2023a; Dhamala et al., 2021) and psychological safety (Huang et al., 2023b). Inspired by this, a series of works focus on the research on the safety of LLMs for diverse NLP tasks (Ganguli et al., 2022; Sun et al., 2023a).

The main challenges to safety in LLMs are: (1) Safety Benchmark Construction: Currently, there are few security-related benchmarks for LLM on various NLP tasks. Establishing effective safety benchmarks is a critical objective in this area. (2) Multilingual Safety Risks: LLM suffers more safety risks across languages and cultures. Identifying and mitigating these risks in a multilingual context is a significant challenge.

## 6 Conclusion

In this work, we make the first attempt to offer a systemic overview of LLMs in NLP, introducing a unified taxonomy about parameter-frozen applications and parameter-tuning applications. Besides, we highlight new research frontiers and challenges, hoping to facilitate future research. Additionally, we maintain a publicly available resource website to track the latest developments in the literature. We hope this work can provide valuable insights and resources to build effective LLMs in NLP.

## References

Griffin Adams, Alexander R. Fabbri, Faisal Ladhak, Eric Lehman, and Noémie Elhadad. 2023. From sparse to dense: Gpt-4 summarization with chain of density prompting. ArXiv, abs/2309.04269.

Angus Addlesee, Weronika Sieińska, Nancie Gunson, Daniel Hernández Garcia, Christian Dondrup, and Oliver Lemon. 2023. Multi-party goal tracking with llms: Comparing pre-training, fine-tuning, and prompt engineering. In Proceedings of the 24th Meeting of the Special Interest Group on Discourse and Dialogue, pages 229-241.

Vaibhav Adlakha, Parishad BehnamGhader, Xing Han Lu, Nicholas Meade, and Siva Reddy. 2023. Evaluating correctness and faithfulness of instructionfollowing models for question answering. arXiv preprint arXiv:2307.16877.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. 2023. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988.

Duarte M. Alves, Nuno M. Guerreiro, Joao Alves, José P. Pombal, Ricardo Rei, Jos'e G. C. de Souza, Pierre Colombo, and André Martins. 2023. Steering large language models for machine translation with finetuning and in-context learning. In Conference on Empirical Methods in Natural Language Processing.

Shamil Ayupov and Nadezhda Chirkova. 2022. Parameter-efficient finetuning of transformers for source code. ArXiv, abs/2212.05901.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473.

Fan Bai, Junmo Kang, Gabriel Stanovsky, Dayne Freitag, and Alan Ritter. 2023. Schema-driven information extraction from heterogeneous tables. arXiv preprint arXiv:2305.14336.

Ahmed Belkhir and Fatiha Sadat. 2023. Beyond information: Is chatgpt empathetic enough? In Proceedings of the 14th International Conference on Recent
Advances in Natural Language Processing, pages $159-169$.

Adithya Bhaskar, Alexander R. Fabbri, and Greg Durrett. 2022. Prompted opinion summarization with gpt-3.5. In Annual Meeting of the Association for Computational Linguistics.

Zhen Bi, Jing Chen, Yinuo Jiang, Feiyu Xiong, Wei Guo, Huajun Chen, and Ningyu Zhang. 2023. Codekgc: Code language model for generative knowledge graph construction. arXiv preprint arXiv:2304.09048.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.

Lang Cao. 2023. Diaggpt: An llm-based chatbot with automatic topic management for task-oriented dialogue. arXiv preprint arXiv:2308.08043.

Kent K Chang, Mackenzie Cramer, Sandeep Soni, and David Bamman. 2023. Speak, memory: An archaeology of books known to chatgpt/gpt-4. arXiv preprint arXiv:2305.00118.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Wenhu Chen. 2023. Large language models are few (1)shot table reasoners. In Findings of the Association for Computational Linguistics: EACL 2023, pages $1090-1100$

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588.

Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, et al. 2022. Binding language models in symbolic languages. In The Eleventh International Conference on Learning Representations.

Ryan A Chi, Jeremy Kim, Scott Hickmann, Siyan Li, Gordon Chi, Thanawan Atchariyachanvanit, Katherine Yu, Nathan A Chi, Gary Dai, Shashank Rammoorthy, et al. 2023. Dialogue distillery: Crafting interpolable, interpretable, and introspectable dialogue from llms. Alexa Prize SocialBot Grand Challenge, 5 .

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton,

Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Fenia Christopoulou, Gerasimos Lampouras, Milan Gritta, Guchun Zhang, Yinpeng Guo, Zhongqi Li, Qi Zhang, Meng Xiao, Bo Shen, Lin Li, et al. 2022. Pangu-coder: Program synthesis with function-level language modeling. arXiv preprint arXiv:2207.11280.

Willy Chung, Samuel Cahyawijaya, Bryan Wilie, Holy Lovenia, and Pascale Fung. 2023. Instructtods: Large language models for end-to-end task-oriented dialogue systems. arXiv preprint arXiv:2310.08885.

Sarkar Snigdha Sarathi Das, Chirag Shah, Mengting Wan, Jennifer Neville, Longqi Yang, Reid Andersen, Georg Buscher, and Tara Safavi. 2023a. S3dst: Structured open-domain dialogue segmentation and state tracking in the era of llms. arXiv preprint arXiv:2309.08827.

Sarkar Snigdha Sarathi Das, Haoran Zhang, Peng Shi, Wenpeng Yin, and Rui Zhang. 2023b. Unified lowresource sequence labeling by sample-aware dynamic sparse finetuning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6998-7010, Singapore. Association for Computational Linguistics.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314.

J. Dhamala, Tony Sun, Varun Kumar, Satyapriya $\mathrm{Kr}-$ ishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. 2021. Bold: Dataset and metrics for measuring biases in open-ended language generation. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.

Xiachong Feng, Xiaocheng Feng, Xiyuan Du, MingSung Kan, and Bing Qin. 2023a. Adapterbased selective knowledge distillation for federated multi-domain meeting summarization. ArXiv, abs/2308.03275.

Yujie Feng, Zexin Lu, Bo Liu, Liming Zhan, and XiaoMing Wu. 2023b. Towards llm-driven dialogue state tracking. arXiv preprint arXiv:2310.14970.

Chengguang Gan, Qinghao Zhang, and Tatsunori Mori. 2023. Giellm: Japanese general information extraction large language model utilizing mutual reinforcement effect. arXiv preprint arXiv:2311.06838.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858.
Haoyu Gao, Ting-En Lin, Hangyu Li, Min Yang, Yuchuan Wu, Wentao Ma, and Yongbin Li. 2023a. Self-explanation prompting improves dialogue understanding in large language models. arXiv preprint arXiv:2309.12940.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023b. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764-10799. PMLR.

Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2022. News summarization and evaluation in the era of gpt-3. ArXiv, abs/2209.12356.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. 2024. Deepseek-coder: When the large language model meets programming-the rise of code intelligence. arXiv preprint arXiv:2401.14196.

Raghav Gupta, Harrison Lee, Jeffrey Zhao, Yuan Cao, Abhinav Rastogi, and Yonghui Wu. 2022. Show, don't tell: Demonstrations outperform descriptions for schema-guided task-oriented dialogue. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4541-4549.

Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, Muhammad Bilal Shaikh, Naveed Akhtar, Jia Wu, Seyedali Mirjalili, et al. 2023. Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects. Authorea Preprints.

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. 2022. ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3309-3326, Dublin, Ireland. Association for Computational Linguistics.

Mutian He and Philip N Garner. 2023. Can chatgpt detect intent? evaluating large language models for spoken language understanding. arXiv preprint arXiv:2305.13512.

Michael Heck, Nurul Lubis, Benjamin Ruppik, Renato Vukovic, Shutong Feng, Christian Geishauser, Hsien-Chin Lin, Carel van Niekerk, and Milica Gašić 2023. Chatgpt for zero-shot dialogue state tracking: A solution or an opportunity? arXiv preprint arXiv:2306.01386.

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14852-14882, Toronto, Canada. Association for Computational Linguistics.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790-2799. PMLR.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022a. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Mengkang Hu, Yao Mu, Xinmiao Yu, Mingyu Ding, Shiguang Wu, Wenqi Shao, Qiguang Chen, Bin Wang, Yu Qiao, and Ping Luo. 2023a. Tree-planner: Efficient close-loop task planning with large language models. arXiv preprint arXiv:2310.08582.

Yushi Hu, Chia-Hsuan Lee, Tianbao Xie, Tao Yu, Noah A Smith, and Mari Ostendorf. 2022b. Incontext learning for few-shot dialogue state tracking. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2627-2643.

Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, EePeng Lim, Lidong Bing, Xing Xu, Soujanya Poria, and Roy Ka-Wei Lee. 2023b. Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models.

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael R. Lyu. 2023a. Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench. ArXiv:2308.03656 [cs].

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael R Lyu. 2023b. Emotionally numb or empathetic? evaluating how llms feel using emotionbench. ArXiv, abs/2308.03656.

Yi-Chong Huang, Xiaocheng Feng, Baohang Li, Chengpeng Fu, Wenshuai Huo, Ting Liu, and Bing Qin. 2024. Aligning translation-specific understanding to general understanding in large language models. ArXiv, abs/2401.05072.

Vojtěch Hudeček and Ondřej Dušek. 2023. Are llms all you need for task-oriented dialogue? arXiv preprint arXiv:2304.06556.

Vivek Iyer, Pinzhen Chen, and Alexandra Birch. 2023. Towards effective disambiguation for machine translation with large language models. In Conference on Machine Translation.
Léo Jacqmin, Lina M. Rojas Barahona, and Benoit Favre. 2022. "do you follow me?": A survey of recent approaches in dialogue state tracking. In Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 336-350, Edinburgh, UK. Association for Computational Linguistics.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen. 2023. StructGPT: A general framework for large language model to reason over structured data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9237-9251, Singapore. Association for Computational Linguistics.

Nengzheng Jin, Joanna Siebert, Dongfang Li, and Qingcai Chen. 2022. A survey on table question answering: recent advances. In China Conference on Knowledge Graph and Semantic Computing, pages 174186. Springer.

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169.

Brendan King and Jeffrey Flanigan. 2023. Diverse retrieval-augmented in-context learning for dialogue state tracking. In Findings of the Association for Computational Linguistics: ACL 2023, pages 55705585 .

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:2219922213.

Fajri Koto, Tilman Beck, Zeerak Talat, Iryna Gurevych, and Timothy Baldwin. 2024. Zero-shot sentiment analysis in low-resource languages using a multilingual sentiment lexicon. arXiv preprint arXiv:2402.02113.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. 2022. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314-21328.

Chia-Hsuan Lee, Hao Cheng, and Mari Ostendorf. 2023. Orchestrallm: Efficient orchestration of language models for dialogue state tracking. arXiv preprint arXiv:2311.09758.

Bin Lei, Chunhua Liao, Caiwen Ding, et al. 2023. Boosting logical reasoning in large language models through a new framework: The graph of thought. arXiv preprint arXiv:2308.08614.

Chunyou Li, Mingtong Liu, Hongxiao Zhang, Yufeng Chen, Jinan Xu, and Ming Zhou. 2023a. Mt2: Towards a multi-task machine translation model with translation-specific in-context learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8616-8627.

Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, and Zhaoxiang Zhang. 2023b. Sheetcopilot: Bringing software productivity to the next level through large language models. arXiv preprint arXiv:2305.19308.

Juntao Li, Zecheng Tang, Yuyang Ding, Pinzheng Wang, Peiming Guo, Wangjie You, Dan Qiao, Wenliang Chen, Guohong Fu, Qiaoming Zhu, Guodong Zhou, and M. Zhang. 2023c. Openba: An open-sourced 15b bilingual asymmetric seq2seq model pre-trained from scratch. ArXiv, abs/2309.10706.

Mingchen Li and Rui Zhang. 2023. How far is language model from $100 \%$ few-shot named entity recognition in medical domain. arXiv preprint arXiv:2307.00186.

Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2023d. Table-gpt: Table-tuned gpt for diverse table tasks. arXiv preprint arXiv:2310.09263.

Peng Li, Tianxiang Sun, Qiong Tang, Hang Yan, Yuanbin Wu, Xuanjing Huang, and Xipeng Qiu. 2023e. Codeie: Large code generation models are better few-shot information extractors. arXiv preprint arXiv:2305.05711.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023f. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161.

Xiang Lisa Li and Percy Liang. 2021a. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 45824597.

Xiang Lisa Li and Percy Liang. 2021b. Prefix-tuning: Optimizing continuous prompts for generation. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), abs/2101.00190.

Yu Li, Baolin Peng, Pengcheng He, Michel Galley, Zhou Yu, and Jianfeng Gao. 2022a. Dionysus: A pre-trained model for low-resource dialogue summarization. ArXiv, abs/2212.10018.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023g. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463.

Zekun Li, Wenhu Chen, Shiyang Li, Hong Wang, Jing Qian, and Xifeng Yan. 2022b. Controllable dialogue simulation with in-context learning. In Findings of the Association for Computational Linguistics. EMNLP 2022, pages 4330-4347.
Zujie Liang, Feng Wei, Yin Jie, Yuxi Qian, Zhenghong Hao, and Bing Han. 2023. Prompts can play lottery tickets well: Achieving lifelong information extraction via lottery prompt tuning. In Proceedings of the 61 st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 277-292, Toronto, Canada. Association for Computational Linguistics.

Eleanor Lin, James Hale, and Jonathan Gratch. 2023. Toward a better understanding of the emotional dynamics of negotiation with large language models In Proceedings of the Twenty-fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing, pages 545-550.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.

Hong Liu, Yucheng Cai, Yuan Zhou, Zhijian Ou, Yi Huang, and Junlan Feng. 2023b. Prompt pool based class-incremental continual learning for dialog state tracking. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages $1-8$. IEEE.

Tianyu Liu, Yizhe Zhang, Chris Brockett, Yi Mao, Zhifang Sui, Weizhu Chen, and William B Dolan. 2022. A token-level reference-free hallucination detection benchmark for free-form text generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6723-6737.

Di Lu, Shihao Ran, Joel Tetreault, and Alejandro Jaimes. 2023a. Event extraction as question generation and answering. arXiv preprint arXiv:2307.05567.

Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Haoran Yang, Wai Lam, and Furu Wei. 2023b. Chainof-dictionary prompting elicits translation in large language models. ArXiv, abs/2305.06575.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023c. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507-2521.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023d. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations.

Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and KaiWei Chang. 2023e. A survey of deep learning for mathematical reasoning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1460514631.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023a. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.

Tongxu Luo, Fangyu Lei, Jiahe Lei, Weihao Liu, Shihu He, Jun Zhao, and Kang Liu. 2023b. Hrot: Hybrid prompt strategy and retrieval of thought for table-text hybrid question answering. arXiv preprint arXiv:2309.12669.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023c. Wizardcoder: Empowering code large language models with evolinstruct. arXiv preprint arXiv:2306.08568.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023 Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048-11064.

Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. 2023. Compositional chain-of-thought prompting for large multimodal models. arXiv preprint arXiv:2311.17076.

Yasmin Moslem, Rejwanul Haque, and Andy Way. 2023. Fine-tuning large language models for adaptive machine translation. ArXiv, abs/2312.12740.

Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. 2023. Generating benchmarks for factuality evaluation of language models. arXiv preprint arXiv:2307.06908.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474.
OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744.

Artidoro Pagnoni, Alexander R. Fabbri, Wojciech Kryscinski, and Chien-Sheng Wu. 2022. Socratic pretraining: Question-driven pretraining for controllable summarization. ArXiv, abs/2212.10449.

Wenbo Pan, Qiguang Chen, Xiao Xu, Wanxiang Che, and Libo Qin. 2023. A preliminary evaluation of chatgpt for zero-shot dialogue understanding. arXiv preprint arXiv:2304.04256.

Sohan Patnaik, Heril Changwal, Milan Aggarwal, Sumita Bhatia, Yaman Kumar, and Balaji Krishnamurthy. 2024. Cabinet: Content relevance based noise reduction for table question answering. arXiv preprint arXiv:2402.01155.

Keqin Peng, Liang Ding, Qihuang Zhong, Li Shen, Xuebo Liu, Min Zhang, Yuanxin Ouyang, and Dacheng Tao. 2023. Towards making the most of chatgpt for machine translation. arXiv preprint arXiv:2303.13780.

Libo Qin, Wanxiang Che, Yangming Li, Haoyang Wen, and Ting Liu. 2019. A stack-propagation framework with token-level intent detection for spoken language understanding. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2078-2087, Hong Kong, China. Association for Computational Linguistics.

Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. 2023a. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2695-2709.

Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S Yu. 2024. Multilingual large language model: A survey of resources, taxonomy and frontiers. arXiv preprint arXiv:2404.04925.

Libo Qin, Tianbao Xie, Wanxiang Che, and Ting Liu. 2021. A survey on spoken language understanding: Recent advances and new frontiers. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pages 4577-4584. International Joint Conferences on Artificial Intelligence Organization. Survey Track.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023b. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Huachuan Qiu, Hongliang He, Shuai Zhang, Anqi Li, and Zhenzhong Lan. 2023. Smile: Singleturn to multi-turn inclusive language expansion via chatgpt for mental health support. arXiv preprint arXiv:2305.00450.

Vikas Raunak, Hany Hassan Awadalla, and Arul Menezes. 2023. Dissecting in-context learning of translations in gpts. ArXiv, abs/2310.15987.

Mathieu Ravaut, Hailin Chen, Ruochen Zhao, Chengwei Qin, Shafiq R. Joty, and Nancy F. Chen. 2023a. Promptsum: Parameter-efficient controllable abstractive summarization. ArXiv, abs/2308.03117.

Mathieu Ravaut, Shafiq R. Joty, Aixin Sun, and Nancy F. Chen. 2023b. On context utilization in summarization with large language models.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Oscar Sainz, Iker García-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, and Eneko Agirre. 2023. Gollie: Annotation guidelines improve zero-shot information-extraction. arXiv preprint arXiv:2310.03668.

Ruhi Sarikaya, Paul A Crook, Alex Marin, Minwoo Jeong, Jean-Philippe Robichaud, Asli Celikyilmaz, Young-Bum Kim, Alexandre Rochette, Omar Zia Khan, Xiaohu Liu, et al. 2016. An overview of endto-end language understanding and dialog management for personal digital assistants. In 2016 ieee spoken language technology workshop (slt), pages 391-397. IEEE.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, et al. 2022. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations.

Tian Shi, Yaser Keneshloo, Naren Ramakrishnan, and Chandan K. Reddy. 2018. Neural abstractive text summarization with sequence-to-sequence models. ACM Transactions on Data Science, 2:1 - 37.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.
Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, and Chandan K Reddy. 2023. Execution-based code generation using deep reinforcement learning. arXiv preprint arXiv:2301.13816.

Ananya Singha, José Cambronero, Sumit Gulwani, Vu Le, and Chris Parnin. 2023. Tabular representation, noisy operators, and impacts on table structure understanding tasks in llms. In NeurIPS 2023 Second Table Representation Learning Workshop.

Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang. 2023a. Gpt4table: Can large language models understand structured table data? a benchmark and empirical study.

Yuan Sui, Jiaru Zou, Mengyu Zhou, Xinyi He, Lun Du, Shi Han, and Dongmei Zhang. 2023b. Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning. arXiv preprint arXiv:2312.09039.

Hao Sun, Zhexin Zhang, Jiawen Deng, Jiale Cheng, and Minlie Huang. 2023a. Safety assessment of chinese large language models. arXiv preprint arXiv:2304.10436.

Xiaofei Sun, Xiaoya Li, Shengyu Zhang, Shuhe Wang, Fei Wu, Jiwei Li, Tianwei Zhang, and Guoyin Wang. 2023b. Sentiment analysis through llm negotiations. arXiv preprint arXiv:2311.01876.

Yuting Tang, Ratish Puduppully, Zhengyuan Liu, and Nancy Chen. 2023. In-context learning of large language models for controlled dialogue summarization: A holistic benchmark and empirical analysis. In Proceedings of the 4th New Frontiers in Summarization Workshop, pages 56-67, Singapore. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Gokhan Tur and Renato De Mori. 2011. Spoken language understanding: Systems for extracting semantic information from speech. John Wiley \& Sons.

A. Ustun and Asa Cooper Stickland. 2022. When does parameter-efficient transfer learning work for machine translation? In Conference on Empirical Methods in Natural Language Processing.

Siddharth Varia, Shuai Wang, Kishaloy Halder, Robert Vacareanu, Miguel Ballesteros, Yassine Benajiba, Neha Anna John, Rishita Anubhai, Smaranda Muresan, and Dan Roth. 2022. Instruction tuning for fewshot aspect-based sentiment analysis. arXiv preprint arXiv:2210.06629.

Yuxuan Wan, Wenxuan Wang, Pinjia He, Jiazhen Gu, Haonan Bai, and Michael R. Lyu. 2023a. Biasasker:

Measuring the bias in conversational ai system. Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering.

Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu, Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023b. Gpt-re: In-context learning for relation extraction using large language models. arXiv preprint arXiv:2305.02105.

Jiaan Wang, Yunlong Liang, Fandong Meng, Beiqi Zou, Zhixu Li, Jianfeng Qu, and Jie Zhou. 2023a. Zeroshot cross-lingual summarization via large language models.

Jiaan Wang, Yunlong Liang, Fandong Meng, Beiqi Zou, Zhixu Li, Jianfeng Qu, and Jie Zhou. 2023b. Zeroshot cross-lingual summarization via large language models. In Proceedings of the 4th New Frontiers in Summarization Workshop, pages 12-23, Singapore. Association for Computational Linguistics.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, $\mathrm{Xu}$ Chen, Yankai Lin, et al. 2023c. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432.

Longyue Wang, Chenyang Lyu, Tianbo Ji, Zhirui Zhang, Dian Yu, Shuming Shi, and Zhaopeng Tu. 2023d. Document-level machine translation with large language models. arXiv preprint arXiv:2304.02210.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023e. Cogvlm: Visual expert for pretrained language models. ArXiv.

Xiao Wang, Weikang Zhou, Can Zu, Han Xia, Tianze Chen, Yuansen Zhang, Rui Zheng, Junjie Ye, Qi Zhang, Tao Gui, et al. 2023f. Instructuie: Multitask instruction tuning for unified information extraction. arXiv preprint arXiv:2304.08085.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023g. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Yiming Wang, Zhuosheng Zhang, and Rui Wang. 2023h. Element-aware summarization with large language models: Expert-aligned evaluation and chain-ofthought method. arXiv preprint arXiv:2305.13412.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. 2023i. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922.
Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. 2021. Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. arXiv preprint arXiv:2109.00859.

Zengzhi Wang, Rui Xia, and Jianfei Yu. 2022. Unifiedabsa: A unified absa framework based on multi-task instruction tuning. arXiv preprint arXiv:2211.10986.

Zengzhi Wang, Qiming Xie, Zixiang Ding, Yi Feng, and Rui Xia. 2023j. Is chatgpt a good sentiment analyzer? a preliminary study. arXiv preprint arXiv:2304.04339.

Mayur Wankhade, Annavarapu Chandra Sekhara Rao, and Chaitanya Kulkarni. 2022. A survey on sentiment analysis methods, applications, and challenges. Artificial Intelligence Review, 55(7):5731-5780.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022a. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022b. Emergent abilities of large language models. Transactions on Machine Learning Research.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny Zhou, et al. 2022c. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, et al. 2023a. Zeroshot information extraction via chatting with chatgpt. arXiv preprint arXiv:2302.10205.

Xiangpeng Wei, Hao-Ran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, Tianxiang Hu, Shangjie Li, Binyuan Hui, Yu Bowen, Dayiheng Liu, Baosong Yang, Fei Huang, and Jun Xie. 2023b. Polylm: An open source polyglot large language model. ArXiv, abs/2307.06018.

Martin Weyssow, Xin Zhou, Kisub Kim, David Lo, and Houari Sahraoui. 2023. Exploring parameterefficient fine-tuning techniques for code generation with large language models. arXiv preprint arXiv:2308.10462.

Genta Winata, Alham Fikri Aji, Zheng Xin Yong, and Thamar Solorio. 2023. The decades progress on codeswitching research in NLP: A systematic survey on trends and challenges. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2936-2978, Toronto, Canada. Association for Computational Linguistics.

BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Bohong Wu, Fei Yuan, Hai Zhao, Lei Li, and Jingjing Xu. 2023a. Extrapolating multilingual understanding models as multilingual generators. In Conference on Empirical Methods in Natural Language Processing.

Minghao Wu, Thuy-Trang Vu, Lizhen Qu, George Foster, and Gholamreza Haffari. 2024. Adapting large language models for document-level machine translation. ArXiv, abs/2401.06468.

Yifan Wu, Pengchuan Zhang, Wenhan Xiong, Barlas Oguz, James C Gee, and Yixin Nie. 2023b. The role of chain-of-thought in complex vision-language reasoning task. arXiv preprint arXiv:2311.09193.

Yuxiang Wu, Guanting Dong, and Weiran Xu. 2023c. Semantic parsing by large language models for intricate updating strategies of zero-shot dialogue state tracking. arXiv preprint arXiv:2310.10520.

Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I Wang, et al. 2022. Unifiedskg: Unifying and multi-tasking structured knowledge grounding with text-to-text language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 602-631.

Tingyu Xie, Qi Li, Jian Zhang, Yan Zhang, Zuozhu Liu, and Hongwei Wang. 2023. Empirical study of zero-shot NER with ChatGPT. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7935-7956, Singapore. Association for Computational Linguistics.

Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, and Enhong Chen. 2023a. Large language models for generative information extraction: A survey. arXiv preprint arXiv:2312.17617.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2023b. A paradigm shift in machine translation: Boosting translation performance of large language models. ArXiv, abs/2309.11674.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. ArXiv, $\mathrm{abs} / 2401.08417$.

Xiancai Xu, Jia-Dong Zhang, Rongchang Xiao, and Lei Xiong. 2023c. The limits of chatgpt in extracting aspect-category-opinion-sentiment quadruples: A comparative analysis. arXiv preprint arXiv:2310.06502.
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mt5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483-498.

Siqiao Xue, Caigao Jiang, Wenhui Shi, Fangyin Cheng, Keting Chen, Hongjun Yang, Zhiping Zhang, Jianshan He, Hongyang Zhang, Ganglin Wei, et al. 2023. Db-gpt: Empowering database interactions with private large language models. arXiv preprint arXiv:2312.17449.

Bin Yang and Jinlong Li. 2023. Visual elements mining as prompts for instruction learning for target-oriented multimodal sentiment classification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6062-6075.

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. ACM Transactions on Knowledge Discovery from Data.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. 2023a. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023b. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.

Junyi Ye, Mengnan Du, and Guiling Wang. 2024. Dataframe qa: A universal llm framework on dataframe question answering without data exposure. arXiv preprint arXiv:2401.15463.

Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. 2023. Large language models are versatile decomposers: Decomposing evidence and questions for table-based reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 174-184.

Dian Yu, Mingqiu Wang, Yuan Cao, Laurent El Shafey, Izhak Shafran, and Hagen Soltau. 2022. Knowledgegrounded dialog state tracking. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3428-3435.

Ruifeng Yuan, Zili Wang, Ziqiang Cao, and Wenjie Li. 2022. Few-shot query-focused summarization with prefix-merging. ArXiv, abs/2211.16164.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning.

Hangwen Zhang, Qingyi Si, Peng Fu, Zheng Lin, and Weiping Wang. 2024. Are large language models table-based fact-checkers? arXiv preprint arXiv:2402.02549.

Haochen Zhang, Yuyang Dong, Chuan Xiao, and Masafumi Oyamada. 2023a. Jellyfish: A large language model for data preprocessing. arXiv preprint arXiv:2312.01678.

Haopeng Zhang, Xiao Liu, and Jiawei Zhang. 2023b. Extractive summarization via chatgpt for faithful summary generation. In Conference on Empirical Methods in Natural Language Processing.

Kai Zhang, Bernal Jimenez Gutierrez, and Yu Su. 2023c. Aligning instruction tasks unlocks large language models as zero-shot relation extractors. ArXiv, abs/2305.11159.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022a. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2023d. Tablellama: Towards open large generalist models for tables. arXiv preprint arXiv:2311.09206.

Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori Hashimoto. 2023e. Benchmarking large language models for news summarization. Transactions of the Association for Computational Linguistics, 12:39-57.

Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. 2023f. Data-copilot: Bridging billions of data and humans with autonomous workflow. arXiv preprint arXiv:2306.07209.

Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing. 2023g. Sentiment analysis in the era of large language models: A reality check. arXiv preprint arXiv:2305.15005.

Xiaoying Zhang, Baolin Peng, Kun Li, Jingyan Zhou, and Helen Meng. 2023h. Sgp-tod: Building task bots effortlessly via schema-guided llm prompting. arXiv preprint arXiv:2305.09067.

Yichi Zhang, Jianing Yang, Keunwoo Yu, Yinpei Dai, Shane Storks, Yuwei Bao, Jiayi Pan, Nikhil Devraj, Ziqiao Ma, and Joyce Chai. 2023i. Seagull: An embodied agent for instruction following through situated dialog.
Yunjia Zhang, Jordan Henkel, Avrilia Floratou, Joyce Cahoon, Shaleen Deep, and Jignesh M Patel. 2023j. Reactable: Enhancing react for table question answering. arXiv preprint arXiv:2310.00815.

Zhehao Zhang, Xitao Li, Yan Gao, and Jian-Guang Lou. 2023k. CRT-QA: A dataset of complex reasoning question answering over tabular data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2131-2153, Singapore. Association for Computational Linguistics.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022b. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 20231. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923.

Jeffrey Zhao, Raghav Gupta, Yuan Cao, Dian Yu, Mingqiu Wang, Harrison Lee, Abhinav Rastogi, Izhak Shafran, and Yonghui Wu. 2022a. Descriptiondriven task-oriented dialog modeling. arXiv preprint arXiv:2201.08904.

Jun Zhao, Kang Liu, and Liheng Xu. 2016. Sentiment analysis: mining opinions, sentiments, and emotions.

Lulu Zhao, Fujia Zheng, Weihao Zeng, Keqing He, Weiran Xu, Huixing Jiang, Wei Wu, and Yanan Wu. 2022b. Domain-oriented prefix-tuning: Towards efficient and generalizable fine-tuning for zero-shot dialogue summarization. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4848-4862, Seattle, United States. Association for Computational Linguistics.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023a. A survey of large language models. arXiv preprint arXiv:2303.18223.

Weixiang Zhao, Yanyan Zhao, Xin Lu, Shilong Wang, Yanpeng Tong, and Bing Qin. 2023b. Is chatgpt equipped with emotional dialogue capabilities? arXiv preprint arXiv:2304.09582.

Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, et al. 2023. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. arXiv preprint arXiv:2303.17568.

Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, and Tat-Seng Chua. 2024. Tat-llm: A specialized language model for discrete reasoning over tabular and textual data. arXiv preprint arXiv:2401.13223.

Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023a. Extrapolating large language models to non-english by aligning languages. ArXiv, abs/2308.04948.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. 2023b. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144.

Ziyu Zhuang, Qiguang Chen, Longxuan Ma, Mingda Li, Yi Han, Yushan Qian, Haopeng Bai, Zixian Feng, Weinan Zhang, and Ting Liu. 2023. Through the lens of core competency: Survey on evaluation of large language models. arXiv preprint arXiv:2308.07902.

Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries, Qian Liu, and Niklas Muennighoff. 2024. Astraios: Parameter-efficient instruction tuning code large language models. arXiv preprint arXiv:2401.00788.

</end of paper 3>


