<paper 0>
# Multilingual Instruction Tuning With Just a Pinch of Multilinguality 

Uri Shaham ${ }^{\tau \gamma}$ Jonathan Herzig ${ }^{\gamma}$ Roee Aharoni ${ }^{\gamma}$<br>Idan Szpektor $^{\gamma}$ Reut Tsarfaty ${ }^{\gamma}$ Matan Eyal ${ }^{\gamma}$<br>${ }^{\tau}$ Tel Aviv University<br>${ }^{\gamma}$ Google Research


#### Abstract

As instruction-tuned large language models (LLMs) gain global adoption, their ability to follow instructions in multiple languages becomes increasingly crucial. In this work, we investigate how multilinguality during instruction tuning of a multilingual LLM affects instruction-following across languages from the pre-training corpus. We first show that many languages transfer some instructionfollowing capabilities to other languages from even monolingual tuning. Furthermore, we find that only 40 multilingual examples integrated in an English tuning set substantially improve multilingual instruction-following, both in seen and unseen languages during tuning. In general, we observe that models tuned on multilingual mixtures exhibit comparable or superior performance in multiple languages compared to monolingually tuned models, despite training on 10x fewer examples in those languages. Finally, we find that diversifying the instruction tuning set with even just 2-4 languages significantly improves cross-lingual generalization. Our results suggest that building massively multilingual instruction-tuned models can be done with only a very small set of multilingual instruction-responses.


## 1 Introduction

Instruction tuning is a fundamental aspect of building modern general-purpose large language models (LLMs), involving fine-tuning a pre-trained model on pairs of instructions and corresponding responses (Mishra et al., 2022; Wei et al., 2022; Sanh et al., 2022; Ouyang et al., 2022). For these models to be globally applicable, they must operate on a wide range of languages, yet, most instruction tuning datasets are typically limited to English. While curating naturally occurring instructions and responses for every language is challenging, cross-lingual transfer has emerged as a promising approach, in which a model is fine-tuned using one language, and acquiring similar abilities in another (Pires et al., 2019; Wu and Dredze, 2019; Artetxe and Schwenk, 2019; K et al., 2020; Conneau et al., 2020a,b). The ability to follow instructions for languages seen only at pre-training can significantly expand the applicability of LLMs, allowing them to be used by more people worldwide. In this work, we show that instruction-tuning of multilingual LLMs transfers across languages better than previously known, and that even minimal language diversity in the tuning set can further unlock instruction-following generalization to languages that are unseen during instruction tuning.

We investigate the effect of multilingual data on instruction-following across languages using an LLM pre-trained on hundreds of languages (Anil et al., 2023), and high-quality, open-ended instructions and responses (Zhou et al., 2023; Köpf et al., 2023) translated into 11 languages, across different families and writing systems. Initially, we examine the transferability of monolingual instruction tuning across different languages. Naturally, tuning using each language individually enhances performance within that language. Notably, we find that this also translates into instruction-following capabilities across other languages, and that tuning with English, Italian, or Spanish yields the best average multilingual performance.

Inspired by this result, we turn to ask how much multilingual data is required to improve multilingual instruction-following, while preserving English performance. We find that replacing even just 40 English training examples with multilingual examples, significantly improves instructionfollowing in those languages. Surprisingly, this small amount of language-diverse examples also improves performance for languages that are only seen during pre-training and are not represented in the instruction tuning set at all.

The next question we tackle is whether increasing the number of languages in the tuning set can
enhance generalization to new languages from the pre-training corpus. We find that tuning using a few languages enables better performance for languages unseen during tuning, compared to monolingual tuning with the same number of examples.

Finally, we test two potential factors that might influence the degree of cross-lingual transfer: language similarity and the amount of languagespecific pre-training data, but find no significant correlations. Overall, our results provide recipes for multilingual instruction tuning that improves cross-lingual generalization, while preserving performance on English, under a fixed budget. In particular, we find that capable multilingual instruction-following models can be tuned even with a minimal amount of multilingual data.

## 2 Measuring Multilingual Instruction-Following

Our objective is to discover how multilinguality during instruction tuning affects general-purpose instruction-following across languages. We break this down to multiple questions, including how well can monolingual instruction tuning transfer to other languages, how many multilingual examples can enhance multilingual instruction-following while preserving English performance, and whether increasing the number of languages can result in improved cross-lingual generalization. In this section we elaborate on the data, evaluation protocol, models we use, and the human annotation process to ensure the models quality.

Data We use datasets of high-quality open-ended instructions and responses, rather than classic taskspecific datasets. Our training data contains 1,000 English instructions and responses from LIMA (Zhou et al., 2023) and 3,640 from OpenAssistant ${ }^{1}$ (Köpf et al., 2023). These examples resemble real world scenarios of users interacting with chatbots, with queries like "Can you explain Fermat's Last Theorem?" and "How to keep a dog hydrated?", that enable efficient tuning even with a small training set (Zhou et al., 2023). For evaluation, we use 617 instructions from AlpacaFarm (Dubois et al., 2023), originated from Self-Instruct (Wang et al., 2023), Vicuna (Chiang et al., 2023), Koala (Geng[^0]

et al., 2023), and hh-rlhf (Bai et al., 2022). ${ }^{2}$

We use the Google Translate $\mathrm{API}^{3}$ to translate the instruction-response pairs of the training set and the instructions of the evaluation set to 11 languages, creating parallel training and evaluation sets in Arabic, Chinese, Czech, English, Estonian, Finnish, Hebrew, Hindi, Italian, Russian, Spanish, and Swahili. ${ }^{4}$ While translated data is different from naturally sourced data per language, it allows for more control as the data size and semantics are similar for all languages. A overview of the languages, their language codes, families and scripts is described in Table 2 in Appendix A.

Evaluation We conduct a side-by-side automatic evaluation protocol (Bubeck et al., 2023; Dubois et al., 2023; Dettmers et al., 2023; Gudibande et al., 2023; Zheng et al., 2023), in which an LLM assesses two responses for the same instruction, with the goal of identifying the superior one. We follow the common practice of presenting both responses to the model twice, alternating the order of the two responses (Zheng et al., 2023; Zhang et al., 2023). The exact prompt we use is shown in Figure 9 in Appendix B. We define a "win" for a certain response if the judge selects it twice irrespective of the order, and a "tie" if the model selects a different response for each order. We use a discounted-tie (Zhou et al., 2023) scoring method, in which a model receives a score of 1 for a win, 0.5 for a tie, and 0 for a loss. We average the scores of individual instructions to get the score over the evaluation set and present it in percentages. To validate that the LLM judge decisions align with human preferences across languages, we conduct a human annotation study and find good aggregated agreement scores of $79.5 \%$ for English, $77 \%$ for Spanish, and $76.5 \%$, and $75 \%$ for Russian and Hebrew, receptively. Further details on validating the LLM judge are provided in Appendix D.

## Instruction-Following Score Per Language

 Throughout this work we measure instructionfollowing per language by comparing the performance of a model that was tuned on some training set $D$, to a model that was monolingually tuned on the target language $\mathcal{L}$, by using the full training[^1]![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-03.jpg?height=754&width=1556&top_left_y=243&top_left_x=250)

Figure 1: Per language instruction-following scores of models instruction-tuned on monolingual data. Each row represents a model tuned using a different language, and each column is an individual heatmap of the scores of all models on the same evaluation language. Scores are the discounted-ties weighted average of the side-by-side scores against the model tuned on the evaluation language. The scores along the diagonal are 50 as they are the result of comparing generations to themselves, and are excluded from the heatmap coloring.

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-03.jpg?height=337&width=694&top_left_y=1322&top_left_x=241)

Figure 2: Human annotators rating distributions of models responses across languages. Each row describes evaluation in its corresponding language of the model tuned monolingually using that language. Numbers in the first row are reported by Zhou et al. (2023).

set in this language, $D_{\mathcal{L}}$. Formally, we define our instruction-following $(I F)$ metric for language $\mathcal{L}$ :

$$
I F_{\mathcal{L}}\left(M_{D}\right)=S \times S\left(M_{D_{\mathcal{L}}}, M_{D}\right)
$$

Where $S \times S(\cdot, \cdot)$ is the side-by-side protocol applied on $M_{D_{\mathcal{L}}}$ and $M_{D}$, which are the models instruction-tuned on $D_{\mathcal{L}}$ and $D$, respectively. A score of $0 \%$ means that $M_{D}$ loses on all $\mathcal{L}$ instructions, and $50 \%$ means the performance of $M_{D}$ and $M_{D_{\mathcal{L}}}$ in $\mathcal{L}$ are indistinguishable when aggregated over the evaluation set.

Model We use the PaLM 2 model family of Transformer-based (Vaswani et al., 2017) LLMs that were pre-trained on hundreds of languages (Anil et al., 2023). We use PaLM 2-S as our pretrained model for all the instruction tuning experiments, and an instruction-tuned PaLM 2-L as the judge for the side-by-side evaluation. The training and inference hyperparameters we use are described in Appendix C.

Human Validation Our evaluation protocol relies on the quality of our monolingually tuned models. To validate their usage as high bar baselines in their respective languages, we conduct a human annotation study in 4 languages: English, Spanish, Russian and Hebrew. Namely, we sample 50 random instructions per language, and ask 2 native speakers to assign a score of excellent, pass, or fail (Zhou et al., 2023) to the responses generated by the model that was monolingually tuned using that language. Results in Figure 2 show that our tuned models indeed demonstrate strong instruction-following abilities. Notably, the scores across languages are similar or better than the reported numbers by Zhou et al. (2023) in English. ${ }^{5}$[^2]

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-04.jpg?height=659&width=1585&top_left_y=236&top_left_x=241)

Figure 3: Instruction-following scores of models trained using when $P \%$ of the training set is distributed uniformly across 12 languages and an $(100-P) \%$ is English only. Each $\mathrm{X}$ axis tick represents a tuning mixture, scores over individual non-English languages are in blue, and their averages are in red. English scores are in orange.

## 3 How Much Multilinguality Is Needed For Multilingual Instruction Tuning?

We now describe our controlled experiments, designed to quantify the effect of multilingual data during instruction tuning of multilingual LLMs, following the research questions defined in $\S 2$.

### 3.1 Monolingual Instruction Tuning Yields Multilingual Abilities

To explore zero-shot cross-lingual transfer of instruction tuning in multilingual LLMs, we tune models on a single language and evaluate them on all of the rest. We find that all of those models are able to transfer non-negligible instructionfollowing abilities to other languages.

Setup We instruction-tune 12 models, each one using the full train set in a different language. We generate responses using every such model to the evaluation instructions in all other languages. Finally, we calculate their per language scores as described in $\S 2$.

Results Figure 1 shows the results, where rows represent training languages and every column is an independent heatmap of the results over a single evaluation language. Most importantly, tuning using each single language yields a model with some multilingual instruction-following capabilities across languages. For context, even the model with the lowest average score, the one tuned on Hindi, achieves a score of over $30 \%$ in 9 out of 11 cases. ${ }^{6}$ The model with the best average score is the one tuned on English, when Italian and Spanish also enable consistently high scores.

Notably, we manually inspect the generations and find that our tuned models consistently respond in the same language as their instruction, regardless of the language they were instructiontuned on, in contrast with findings in previous work (Touvron et al., 2023a; Chen et al., 2023). We hypothesize that this comes from the multilingual nature of PaLM 2s' pre-training, compared to the more English-centric LLaMA (Touvron et al., 2023a), further details are in Appendix E. In addition to our main setup, we also compare the generations of these models to the ones of the pre-trained model that was not instruction-tuned. Results shown in Figure 10 in Appendix F further demonstrate that instruction tuning in every language separately, greatly improves instructionfollowing abilities across different languages.

### 3.2 A Few Dozen Examples Improve Multilingual Instruction-following

Naturally, multilingual tuning, as opposed to English-exclusive tuning under a fixed training examples budget, should result in better downstream performance for non-English languages, and might hurt performance on English. Therefore, we ask how many multilingual examples can improve the instruction-following abilities across languages,[^3]

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-05.jpg?height=654&width=1568&top_left_y=238&top_left_x=244)

Figure 4: Instruction-following scores of models tuned when $P \%$ of the training set is distributed uniformly across 6 languages and an $(100-P) \%$ is English only. Each $\mathrm{X}$ axis tick represents such a tuning set, scores over individual non-English languages are in blue and English scores are in orange. Average scores of the 5 non-English languages in the tuning set are in red, and the average scores of the 6 languages not seen during tuning are in green.

while preserving English performance. To that end, we tune models on subsets of the English examples combined with subsets of multilingual examples in different ratios. We find a significant boost in multilingual instruction-following abilities even when using just a few dozen multilingual examples.

Setup We create data mixtures with $P \%$ examples that are evenly split among all 12 languages, and the rest $(100-P) \%$ English examples. ${ }^{7} \mathrm{We}$ create such a train set for every $P$ from 10 to 100 , incremented by tens, and also for $P=1$, for which only 40 multilingual examples are included from across all 11 non-English languages, and the rest are English examples. Finally, we evaluate every tuned model on every one of the 12 languages as defined in $\S 2$.

Results Figure 3 visualizes the results. As expected, multilingual examples in the train set improve the score on their languages (Red), and diluting the number of English examples hurts the performance in English (Green). Notably, the significant multilingual improvement comes from replacing only $1 \%$ of the English examples by multilingual ones, which translates to 40 examples evenly distributed across the training languages. These results on the effect of such a small amount of language-diversity extend findings regarding taskdiversity by Zhou et al. (2023), which demonstrated that a capable monolingual instruction-following[^4]

model can be tuned using only 1,000 high-quality examples. A second trend is that these models often outperform their monolingually-tuned counterparts on the very language the latter were exclusively tuned on (blue markers above the 50 line). For example, the model tuned using the uniform set $(P=100)$ preforms similarly or better than the individual monolingually-tuned models in 8 of 12 languages, despite being trained on 12 times less instruction-response pairs for each language. This suggests that for some languages, multilingual tuning can enable better instruction-following abilities compared to a traditional monolingual tuning with the same number of examples.

### 3.3 A Few Dozen Examples Improve Cross-lingual Generalization

Combining the lessons on cross-lingual generalization from monolingual tuning and the effect of a small amount of multilingual examples from previous sections, we turn to examine how multilingual examples in the tuning set affect language generalization. Specifically, we conduct a similar experiment to the one in $\S 3.2$, this time using only half of the languages for tuning while the rest of languages are unseen. In line with the results from $\S 3.2$, we find that a very small amount of multilingual examples also improve performance on languages that were not in the tuning set.

Setup We repeat the setup from §3.2, this time with only English and 5 more languages: Arabic,

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-06.jpg?height=468&width=696&top_left_y=280&top_left_x=243)

Figure 5: Instruction-following scores in Czech, Estonian, Hebrew, Hindi, Spanish, and Chinese of models instruction-tuned using various subsets of Arabic, English, Finnish, Italian, Russian, and Swahili. Blue markers are the average scores per evaluation languages across models tuned with the same number of languages. The averages of those individual languages scores are in green.

Finnish, Italian, Russian, and Swahili, and evaluate models again on all 12 languages.

Results Results in Figure 4 show similar trends to the ones in Figure 3. Specifically, the average score over non-English training languages (red) again improves very quickly, even with $P=1$. Strikingly, this is also true for languages that the model has only seen during pre-training, and are not represented at all in the instruction tuning dataset (orange). This suggests that very few multilingual examples can not only improve performance for the languages of those examples, but also enable better cross-lingual instruction-following generalization.

### 3.4 Even a Small Number of Languages Improves Cross-Lingual Generalization

Given the results on the impact of a small number of multilingual examples from a fixed set of languages, we ask whether a small number of languages can also enhance cross-lingual generalization. We experiment with different numbers of languages in the tuning set and indeed observe that the transfer to languages only seen during pre-training improves from the very first additional languages.

Setup We instruction-tune models on a single language and up to 6 languages. At each step, we add a language to the tuning set, and split the same examples budget uniformly among the current set of languages. We use the 6 training languages from $\S 3.3$, and follow 3 different permutations that
![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-06.jpg?height=444&width=700&top_left_y=266&top_left_x=1066)

Figure 6: Average instruction-following scores of languages not seen during instruction tuning. For example, the top-left corner describes the scores of 3 models instruction-tuned on $100 \%$ Spanish, $100 \%$ English, and $50 \%$ Spanish and $50 \%$ English. The Y axis of this subfigure is the average score across all language excluding Spanish and English.

determine the order in which we add languages to the mix. These permutations are shown in Table 4 in Appendix G. We evaluate every model on each of the remaining 6 languages, and average scores per evaluation language across models that are tuned using the same number of languages.

Results Results on Figure 5 show that adding languages to the tuning set improves cross-lingual generalization. The average score (red) increases from tuning on monolingual data to tuning on bilingual data, and even more when using 3 and 4 languages, where the average score gets to almost 50. At that point, there is an indication for saturation, as more languages does not seem to improve transfer further. These findings demonstrate that diversifying the instruction tuning data with only a few different languages can improve cross-lingual transfer to new languages, only seen during pre-training.

Bilingual Tuning Sets To show this holds for even more combinations of languages, we randomly split all languages to pairs, and tune models using $50 \%$ of the examples in the one language and $50 \%$ in the other. We evaluate each of these models on the remaining 10 languages, and compare their score to the ones of the two models tuned using the full monolingual sets. Results on Figure 6 reveal that bilingual tuning helps generalize to new languages better than monolingual tuning.

## 4 Potential Factors of Transferability

Following the results from the previous sections, a natural question arises: what factors can predict the

| Language | Code | Slavic <br> Family | Script | Mutually <br> Intelligible |
| :--- | :---: | :---: | :---: | :---: |
| Russian | ru | East | Cyrillic | - |
| Serbian | $\mathrm{sr}$ | South | Cyrillic | Croatian |
| Croatian | $\mathrm{hr}$ | South | Latin | Serbian |
| Slovenian | $\mathrm{sl}$ | South | Latin | - |
| Polish | $\mathrm{pl}$ | West | Latin | - |
| Slovak | $\mathrm{sk}$ | West | Latin | Czech |
| Czech | $\mathrm{cs}$ | West | Latin | Slovak |

Table 1: Languages used for language similarity experiment, along with their language code, subfamily, script, and the language they are mutually intelligible with.

degree of cross-lingual transfer? We explore two immediate candidates. Initially, we examine the relation of various aspects of language similarity to transferability within language pairs. Next, we look into whether the proportion of language-specific data in the pre-training corpus correlates with the amount of cross-lingual transfer of instruction tuning using the given language.

### 4.1 Language Similarity

A intuitive hypothesis is that aspects of language similarity like the script or mutual intelligibility might affect the levels of instruction tuning crosslingual transfer between languages. We test this using a case study of 7 Slavic languages, looking into possible effects of such aspects. However, we do not find a signal indicating these factors strongly correlate with cross-lingual transfer for this setting.

Setup We train models on monolingual versions of the data in Russian, Serbian, Croatian, Slovenian, Polish, Slovak and Czech, and evaluate their transfer to each other. These languages can be divided along several linguistic lines that are summarized in Table 1. First, Russian is East Slavic, and the rest are either South or West Slavic. Second, Russian and Serbian both use the Cyrillic script, while the rest use Latin. Moreover, both Serbian and Croatian, and Slovak and Czech share a significant degree of mutual intelligibility.

Results Results are displayed on Figure 7. As shown, there is no a strong signal indicating that any of the aspects above is correlated with better mutual cross-lingual transfer. Russian and Czech tend to transfer instruction-following abilities best, and even though Russian and Serbian both use Cyrillic, Croatian and Czech transfer capabilities to Russian better than Serbian. Examining the effect of mutual intelligibility, Croatian and Serbian do

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-07.jpg?height=503&width=742&top_left_y=248&top_left_x=1068)

Figure 7: Instruction-following scores per language of models tuned monolingually. Each row represents a model trained using a different language, and each column is an individual heatmap of the scores of all models on the same evaluation language. The scores along the diagonal are excluded from the heatmaps coloring.

not share cross-lingual abilities more than other languages, and while Slovak and Czech are mutually intelligible, Slovak transfers to Czech less than the rest. Our results align with recent findings that language similarity does not impact transferability or interference in machine translation given sufficient data and model capacity (Fernandes et al., 2023; Shaham et al., 2023).

### 4.2 Fraction of Data in Pre-training

A second possible predictor of the degree of crosslingual transfer from a particular language is the extent to which the model was exposed to it during pre-training. Generally, a model's downstream performance on a specific language correlates with the fraction of data in that language in the pre-training corpus (Muennighoff et al., 2023). In contrast, Figure 8 suggests this is not necessarily the case for the cross-lingual transfer from a specific language. We find a weak Pearson correlation of 0.22 between the average cross-lingual score of each language and the number of documents in that language in pre-training corpus (Table 21 in Anil et al. (2023)).

## 5 Related work

Cross-lingual Transfer The success of the pretraining-fine-tuning paradigm (Devlin et al., 2019) ignited a new line of work on cross-lingual transfer. Pires et al. (2019) and Wu and Dredze (2019) showed that the multilingual variant of BERT can be fine-tuned on a specific task in one language and preform this task on another language, and Artetxe and Schwenk (2019) reported similar find-

![](https://cdn.mathpix.com/cropped/2024_06_04_2281bf7f9009c7097f31g-08.jpg?height=505&width=694&top_left_y=296&top_left_x=247)

Figure 8: Weak Pearson correlation between the percentage of documents in the pre-training corpus (excluding English), and the average instruction-following score across languages for every training language. Blue area around the line is the confidence interval.

ings with a Recurrent Neural Network. Conneau et al. (2020a) introduced XLM-R, a multilingual pre-trained encoder with strong cross-lingual abilities. Phang et al. (2020) showed that intermediate training on an English task improves XLMR's transfer across languages further, and Pfeiffer et al. (2020) suggested an adapter-based framework to improve cross-lingual and task generalization. Hu et al. (2020) proposed a benchmark for cross-lingual generalization consists of 40 languages across 9 NLP tasks.

$\mathrm{K}$ et al. (2020) found that the depth of the network matters for cross-lingual transfer, and Conneau et al. (2020b) showed that parameter sharing is more important than shared vocabulary. Choenni et al. (2023) delved into the influence of specific examples from the training data on the performance in other languages, and Malkin et al. (2022) investigated how pre-training BERT-based models using different language pairs affects cross-lingual downstream performance. Going beyond encoderonly models, Xue et al. (2021) proposed mT5, a multilingual variant of T5 (Raffel et al., 2020), and showed the significance of model scaling for crosslingual transfer in generation tasks. Ye et al. (2023) explored trasferability in English-centric models (Touvron et al., 2023a) using four tasks.

In contrast to most cross-lingual transfer literature that is focused on task-specific fine-tuning, we explore trends of cross-lingual generalization for general-purpose instruction-following LLMs.
Multilingual Instruction Tuning Initially, works on instruction tuning (Mishra et al., 2022; Wei et al., 2022; Sanh et al., 2022) focused on cross-task generalization in English. Subsequently, a large body of work was dedicated to multilingual instruction tuning. Muennighoff et al. (2023) found that tuning models with English datasets enables zero-shot cross-lingual abilities to new languages. The authors also found that this holds for languages that the model has never intentionally seen during pre-training, and that multilingual training improves generalization to new tasks. Chen et al. (2023) investigated the effects of full parameter training vs low-rank adaptation (Hu et al., 2022) and monolingual vs multilingual instruction tuning using the Stanford Alpaca (Taori et al., 2023) data, machine translated into 5 languages. Lai et al. (2023) trained multilingual instruction-following models for 26 languages with reinforcement learning from human feedback (Ouyang et al., 2022), and Zhang et al. (2023) suggested instruction tuning LLMs by prepending the instruction and response translated into a pivot language (e.g English) to the response in the target language. Concurrently with our work, Kew et al. (2023) found that only a few languages in the tuning set result in better cross-lingual transfer to new languages for English-centric LLMs.

In this work, we consider transfer from monolingual instruction tuning from 12 languages, rather than exclusively on English. Furthermore, we examine multilingual instruction-following using an LLM pre-trained on hundreds of languages, which might be a key to unlocking more transfer to languages not represented during tuning. Importantly, we unveil the potential of just a small amount of language diversity in the instruction tuning set for this cross-lingual generalization.

## 6 Conclusion

We demonstrate that cross-lingual transfer offers a promising avenue for building multilingual instruction-following LLMs. Our findings across different languages suggest that even monolingual instruction tuning using only one language can result in improved instruction-following capabilities in other languages. Moreover, incorporating even a small set of a few dozen multilingual examples can significantly enhance instruction-following performance for both the languages the model is tuned on, and ones that were only seen during pre-training.

Additionally, training on such multilingual datasets achieves comparable or even superior performance compared to monolingual tuning for some languages. We observe a similar trend when exploring the effect of total number of languages in the tuning set, as even splitting the train set to only two languages improves generalization to new languages, compared to monolingual tuning. These findings pave the way for efficient and scalable development of multilingual LLMs capable of understanding and following instructions across languages with minimal multilingual supervision.

## 7 Limitations

Limitations of our work include the use of translation for expanding datasets to multilingual settings, the number of languages we evaluated on, and number of models we experimented with. We now discuss each of them.

Translated data One limitation of our work is that our data is translated using the Google Translate API, and not originally sourced by native speakers. Automatic translation is inherently imperfect and may introduce noise to the tuning sets. However, translation also allows to for a controlled setup with parallel data, in which the content of all training and evaluation examples is the same for all languages.

Number of languages A second limitation is that we use 12 languages in our main experiments (§3), with 3 additional languages in the language similarity experiment (§4.1). Clearly, multilingual instruction-following models need to successfully operate in many more languages, and we leave work on scaling this number to future work.

Number of models Lastly, we experiment with PaLM 2, and results may vary with different LLMs. Nevertheless, our focus on PaLM 2 highlights the potential of multilingual pre-training for future advancements in LLMs.

## Acknowledgments

We thank Omer Levy, Or Honovich, Alon Jacovi, Avi Caciularu, and Omer Goldman for their valuable feedback.

## References

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy GurAri, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. 2023. Palm 2 technical report.

Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zeroshot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597-610.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4.

Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Barry Haddow, and Kenneth Heafield. 2023. Monolingual or multilingual instruction tuning: Which makes a better alpaca.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with $90 \% *$ chatgpt quality.

Rochelle Choenni, Dan Garrette, and Ekaterina Shutova. 2023. How do languages influence each other? studying cross-lingual data sharing during LM fine-tuning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13244-13257, Singapore. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020a. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 84408451, Online. Association for Computational Linguistics.

Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020b. Emerging cross-lingual structure in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6022-6034, Online. Association for Computational Linguistics.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacafarm: A simulation framework for methods that learn from human feedback.

Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, and Orhan Firat. 2023. Scaling laws for multilingual neural machine translation.

Xinyang Geng, Arnav Gudibande, Hao Liu, Eric Wallace, Pieter Abbeel, Sergey Levine, and Dawn Song.
2023. Koala: A dialogue model for academic research. Blog post.

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. 2023. The false promise of imitating proprietary llms.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A massively multilingual multitask benchmark for evaluating cross-lingual generalisation. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4411-4421. PMLR

Karthikeyan K, Zihan Wang, Stephen Mayhew, and Dan Roth. 2020. Cross-lingual ability of multilingual bert: An empirical study. In International Conference on Learning Representations.

Tannon Kew, Florian Schottmann, and Rico Sennrich. 2023. Turning english-centric llms into polyglots: How much multilinguality is needed?

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. 2023. Openassistant conversations - democratizing large language model alignment.

Viet Lai, Chien Nguyen, Nghia Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan Rossi, and Thien Nguyen. 2023. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 318-327, Singapore. Association for Computational Linguistics.

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023. Self-alignment with instruction backtranslation.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74-81, Barcelona, Spain. Association for Computational Linguistics.

Dan Malkin, Tomasz Limisiewicz, and Gabriel Stanovsky. 2022. A balanced data approach for evaluating cross-lingual transfer: Mapping the linguistic blood bank. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4903-4915, Seattle, United States. Association for Computational Linguistics.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470-3487, Dublin, Ireland. Association for Computational Linguistics.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15991-16111, Toronto, Canada. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730-27744. Curran Associates, Inc.

Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654-7673, Online. Association for Computational Linguistics.

Jason Phang, Iacer Calixto, Phu Mon Htut, Yada Pruksachatkun, Haokun Liu, Clara Vania, Katharina Kann, and Samuel R. Bowman. 2020. English intermediatetask training improves zero-shot cross-lingual transfer too. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 557-575, Suzhou, China. Association for Computational Linguistics.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996-5001, Florence, Italy. Association for Computational Linguistics.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Uri Shaham, Maha Elbayad, Vedanuj Goswami, Omer Levy, and Shruti Bhosale. 2023. Causes and cures for interference in multilingual translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15849-15863, Toronto, Canada. Association for Computational Linguistics.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan,

Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484-13508, Toronto, Canada. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Shijie Wu and Mark Dredze. 2019. Beto, bentz, becas: The surprising cross-lingual effectiveness of BERT. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 833-844, Hong Kong, China. Association for Computational Linguistics.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics. Human Language Technologies, pages 483-498, Online. Association for Computational Linguistics.

Jiacheng Ye, Xijia Tao, and Lingpeng Kong. 2023. Language versatilists vs. specialists: An empirical revisiting on multilingual transfer ability.

Zhihan Zhang, Dong-Ho Lee, Yuwei Fang, Wenhao Yu, Mengzhao Jia, Meng Jiang, and Francesco Barbieri. 2023. Plug: Leveraging pivot language in crosslingual instruction tuning.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment.

| Language | Code | Family | Script |
| :--- | :--- | :--- | :--- |
| Arabic | ar | Afro-Asiatic | Arabic |
| Chinese | zh | Sino-Tibetan | Chinese |
| Czech | cs | Indo-European | Latin |
| English | en | Indo-European | Latin |
| Estonian | et | Uralic | Latin |
| Finnish | fi | Uralic | Latin |
| Hebrew | he | Afro-Asiatic | Hebrew |
| Hindi | hi | Indo-European | Devanagari |
| Italian | it | Indo-European | Latin |
| Russian | ru | Indo-European | Cyrillic |
| Spanish | es | Indo-European | Latin |
| Swahili | sw | Niger-Congo | Latin |

Table 2: Languages used in our main experiments.
</end of paper 0>


<paper 1>
# Lucky 52: How Many Languages Are Needed to Instruction Fine-Tune Large Language Models? 

Shaoxiong Ji<br>University of Helsinki<br>shaoxiong.ji@helsinki.fi

Pinzhen Chen<br>University of Edinburgh<br>pinzhen.chen@ed.ac.uk


#### Abstract

Fine-tuning large language models for multilingual downstream tasks requires a diverse set of languages to capture the nuances and structures of different linguistic contexts effectively. While the specific number varies depending on the desired scope and target languages, we argue that the number of languages, language exposure, and similarity that incorporate the selection of languages for fine-tuning are some important aspects to examine. By fine-tuning large multilingual models on 1 to 52 languages, this paper answers one question: How many languages are needed in instruction fine-tuning for multilingual tasks? We investigate how multilingual instruction fine-tuned models behave on multilingual benchmarks with an increasing number of languages and discuss our findings from the perspective of language exposure and similarity.


## 1 Introduction

Multilinguality in the context of language models refers to the ability of a model to understand and generate texts in multiple languages. Large language models are often designed to be multilingual, meaning they can handle text in various languages and even perform translation between them. Large language models (LLMs), like mGPT (Shliazhko et al. 2022), BLOOM (Scao et al., 2022), and LLaMA (Touvron et al., 2023), are characterized by their vast size and extensive pre-training on a collection of diverse multilingual texts from the Internet. They exhibit impressive capabilities to comprehend and generate texts across multiple languages to solve downstream tasks. However, mastering multilingual proficiency requires an understanding of the unique grammatical structures, idiomatic expressions, and cultural contexts inherent to each language. Fine-tuning a model for multilingual prowess demands meticulous consideration of these linguistic nuances to ensure accurate and contextually appropriate responses.

Recent research has shed light on the efficacy of multilingual instruction fine-tuning (mIT) in enhancing multilingual performance. Chen et al. (2024) compared monolingual and multilingual instruction tuning under a resource-fair scenario. Kew et al. (2023) experimented with English-centric LLMs such as Llama 2 (Touvron et al. 2023) and Falcon (Almazrouei et al. 2023) and found that multilingual instruction fine-tuning with as few as three languages improves models' cross-lingual transfer abilities. Similarly, Shaham et al. (2024) studied instruction tuning for multilingual LLMs and observed that enhancing the diversity of languages in the instruction tuning set, even by just adding 2,3 , or 4 languages, leads to improved cross-lingual generalization. Moreover, Chirkova \& Nikoulina (2024b) explored zero-shot cross-lingual transfer, where an LLM is trained on English-only data but tested on prompts in the other four languages, highlighting the importance of learning rate in achieving effective cross-lingual transfer.

Nevertheless, it is essential to note the variations in the number and diversity of languages used in these studies. Kew et al. (2023) considered five non-English languages (Spanish,[^0]

Russian, German, Chinese, and French) but also trained models on the full Guanaco dataset with more than 30 languages. Shaham et al. (2024) experimented with a machine-translated instruction dataset in 11 diverse languages. Chirkova \& Nikoulina (2024b) studied finetuning with English data and tested on four languages (English, French, Portuguese, and Russian). These works are not strictly comparable due to different training configurations, data, model choices, and evaluation tasks and benchmarks. More importantly, while it has been demonstrated that a small number of languages can elicit (zero-shot) cross-lingual transfer, it is in no way associated with achieving the optimal downstream task performance.

We aim to fill in the gap by scaling up the number of languages in the instruction tuning phase, offering insights into the broader implications of multilingual instruction fine-tuning. This paper employs instruction fine-tuning using the multilingual BLOOM model (Scao et al. 2022), and a parallel instruction dataset named Bactrain-X in 52 languages ( $\mathrm{Li}$ et al. 2023). We progressively add a language during instruction fine-tuning at each time, and train 52 models in total. Then, we evaluate those models using three multilingual benchmarks. Our experimental results show that:

- Contrary to prior research, adding more languages beyond a handful can further improve accuracy, although with some outlier cases and diminishing returns.
- Given the considered 52 studied languages, there is no consistent answer regarding the optimal number of languages for mIT. The optimal number of instruction languages depends on the language similarity and downstream evaluation.
- The impact of mIT can vary, potentially aiding or hindering multilingual performance. Additionally, the cross-lingual transfer ability of mIT exists, though both phenomena are contingent upon the benchmark and languages involved.

Our study emphasizes the importance of a closer look at the tasks, benchmarks, languages, and evaluation metrics. And we advocate for more consistent future studies focused on mIT. For example, in the prior works that explore open-ended chat (Shaham et al. 2024: Kew et al. 2023: Chirkova \& Nikoulina, 2024b, Chen et al. 2024), there is variation in the evaluation set, number of samples, and evaluation criteria. Our study highlights the necessity for more systematic experimental studies on various variables. These variables include but are not limited to base LLMs, training recipes, instruction data and languages, evaluation tasks and benchmarks, and evaluation criteria for different tasks. Such comprehensive and consistent investigations are crucial for advancing our understanding of mIT and its implications.

## 2 Multilingual Instruction Fine-tuning

We perform multilingual instruction fine-tuning to large language models, aiming to study their proficiency across multiple languages. By fine-tuning the model using diverse linguistic datasets, it adapts to various languages, allowing it to generate more contextually relevant and accurate responses in a multilingual context. This section describes the methodological settings of multilingual instruction fine-tuning, multilingual instruction data, and the base language model, which are necessary for addressing our research inquiries.

### 2.1 Instruction Fine-tuning with an Increasing Number of Languages

Our setup is supervised fine-tuning, where an instruction and a task input are fed to an LLM, and the LLM is trained to produce a response. To expand the multilingual capabilities of the model, we employ a strategy that progressively incorporates additional languages during instruction fine-tuning. We start from English and Chinese, which are extensively resourced languages that are prominently presented in both pretraining corpora and evaluation benchmarks and also represent distinct written scripts. Then, we progressively add the other languages in alphabetical order.

It is important to acknowledge a potential limitation stemming from the increasing data size as the number of languages expands. This can introduce an additional variable that might impact multilingual performance. To mitigate this effect, we opt for parallel instruction
data in which English instructions are translated into different languages. This ensures consistency in the instruction information across languages while minimizing the overall increase in data size. Moreover, the increase in data size also amplifies the number of optimization steps when utilizing stochastic gradient descent to update the model parameters on the same device. The number of updates can be expressed as: $U=\left\lceil\frac{N \times L \times E}{B \times W}\right\rceil$, where $N$ is the number of instruction data, $L$ is the number of languages, $E$ is the number of training epochs, $B$ is the batch size, and $W$ is the world size (i.e., the number of GPUs). We increase the number of GPUs proportionally when increasing the number of languages to maintain a manageable range of updates.

### 2.2 Multilingual Instruction Data

The choice of multilingual instruction data lies in its comprehensive language coverage and data quality, thus enabling robust and inclusive multilingual instruction fine-tuning experiments. We use the Bactrian-X dataset (Li et al., 2023) that addresses the scarcity of high-quality multilingual instruction-response data. This dataset comprises 3.4 million instruction-response pairs across 52 languages. Instructions were collected from Alpaca (52K) (Taori et al. 2023) and Dolly (15K) (Conover et al. 2023) datasets, with Alpaca instructions generated by a text-davinci-003 model and Dolly instructions contributed by humans. These $67 \mathrm{~K}$ instructions in English were then translated into 51 different languages using the Google Translate API, aligned with the languages supported by the mBART-50 model (Liu et al. 2020). To ensure translation accuracy and relevance, instructions containing programming-related content were identified and excluded. To assess translation quality, 100 randomly selected sentences per language were backtranslated into English using the same API, with original English sentences as references. Automatic evaluation with standard metrics indicates reliable and high-quality translations. For further comprehensive details regarding the dataset creation process and evaluation, readers are encouraged to refer to the original paper. In our experimentation, we utilize the Bactrian- $\mathrm{X}$ dataset as the foundation and iteratively augment it by adding one language at a time for multilingual instruction fine-tuning. The languages selected for instruction tuning follow the specified order: en (English), zh (Chinese), af (Afrikaans), ar (Arabic), az (Azerbaijani), bn (Bengali), cs (Czech), de (German), es (Spanish), et (Estonian), fa (Farsi), fi (Finnish), fr (French), gl (Galician), gu (Gujarati), he (Hebrew), hi (Hindi), hr (Croatian), id (Indonesian), it (Italian), ja (Japanese), ka (Georgian), kk (Kazakh), km (Khmer), ko (Korean), lt (Lithuanian), lv (Latvian), mk (Macedonian), ml (Malayalam), mn (Mongolian), mr (Marathi), my (Burmese), ne (Nepali), nl (Dutch), pl (Polish), ps (Pashto), pt (Portuguese), ro (Romanian), ru (Russian), si (Sinhala), sl (Slovenian), sv (Swedish), sw (Swahili), ta (Tamil), te (Telugu), th (Thai), tl (Tagalog), $\operatorname{tr}$ (Turkish), uk (Ukrainian), ur (Urdu), vi (Vietnamese), xh (Xhosa). We note that a recent multilingual instruction dataset called Aya (Singh et al. 2024) was released at the time of writing. We leave it as a future study.

### 2.3 Base Language Model

Multilingual language models can inherit biases present in the training data, which may affect their responses when fine-tuned. We adopt a representative multilingual language model: BLOOM (Scao et al. 2022) that is developed with careful consideration in multiple natural and coding languages. Its multilingual capacity makes it well-suited for a wide range of natural language processing tasks across multiple languages. In our specific application, we opt for the BLOOM-7B1 variant with 7.1B parameters. This parameter count ensures necessary capabilities for text understanding and generation across a multitude of languages and contexts, and it is affordable considering our computing budget.

## 3 Experimental Setup

### 3.1 Training Details

We use the Hugging Face transformers framework (Wolf et al. 2019) with the DeepSpeed integration (Rasley et al. 2020) as the training software. The learning rate is set to $3 \mathrm{e}-5$. The
batch size was established at 4 per device. Gradient accumulation, with a step size of 4 , enables the aggregation of gradients over multiple steps. The number of training epochs is fixed at 3. The maximum model length is set to 768 , the same as the Bactrian- $\mathrm{X}$ paper $(\overline{\mathrm{Li}}$ et al., 2023). Models are trained on the cluster with 4 AMD MI250X GPUs (8 GPU dies) in each node. We adopt distributed training on multiple nodes from 2 to 10 nodes with the increase in the number of languages, making the global batch size from 256 to 1280.

### 3.2 Benchmarks and Evaluation

We evaluate the instruction-tuned models on three multilingual benchmarks. XCOPA (Ponti et al. 2020) is a multilingual dataset for causal commonsense reasoning in 11 languages XStoryCloze (Lin et al. 2021) is a multilingual dataset derived from the English StoryCloze dataset (Mostafazadeh et al. 2017) for commonsense reasoning in the story, translated into 10 non-English languages. The test involves a system selecting the appropriate ending for a four-sentence story. XWinograd (Tikhonov \& Ryabinin, 2021) is a multilingual compilation of Winograd Schemas (Levesque et al. 2012) available in six languages, designed for assessing cross-lingual commonsense reasoning abilities. We utilize zero-shot evaluation techniques facilitated by the Language Model Evaluation Harness (lm-evaluation-harness) (Gao et al. 2023). Different models, trained with progressively added languages, are evaluated on these benchmarks using accuracy (\%) as the evaluation metric.

## 4 Results and Discussion

This section presents the experimental results and our findings.

### 4.1 Overall Performance with the Increasing Number of Languages

We first answer the effect of the number of languages on multilingual performance - how much multilingualism is needed for instruction-tuning a large language model, i.e., BLOOM7B1 in our case study. Figure 1a illustrates the average accuracy on three benchmarks with the increase in the number of languages. The figure shows fluctuating results. For XCOPA and XStoryCloze, there is a slightly increasing trend with the increasing number of instruction languages, except for a notable drop when Korean is added into the instruction languages. For XWinograd, the trend is not clear, and the performance also drops when Korean is added. We checked the training curve of the model trained with Korean added to instruction data and found that the training and validation loss decreases as training goes on and the model converges as expected.

Moving to the performance in reference languages, i.e., English and Chinese, we notice a similar drop in accuracy when Korean is added, as displayed in Figures 1b to 1d, but no obvious trend for cross-lingual transfer capacity with the increase in the number of languages. For XCOPA, instruction fine-tuning on Chinese improves the accuracy of the target language, and the accuracy peaks at the model when Bangla (bn), the 6th language, is added for instruction fine-tuning. For English and Chinese XStoryCloze, Latvian (lv, the 27th language added) and Malaysian (ml, the 29th language added) are helpful to boost the performance in these languages respectively. Instruction fine-tuning turns out to be harmful on the XWinograd dataset, as shown in Figure $1 \mathrm{~d}$ that instruction fine-tuned models are less performant than the base LLM. Multilingual instruction fine-tuning does not generalize on the three benchmarks studied; sometimes, multilingual instruction tuning is detrimental.

Overall, the performance of multilingual evaluation does not increase linearly with the number of languages of instruction data. The effect of the number of languages on multilingual performance is dependent on the task and the language added for training. Our results show that instruction fine-tuning with a few languages is good for cross-lingual transfer but not the most performant. Further, adding more languages, even in alphabetical order, can sometimes improve the performance and is occasionally harmful in XCOPA and XStoryCloze, while in XWinograd, multilingual instruction fine-tuning is detrimental.

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-05.jpg?height=848&width=1391&top_left_y=270&top_left_x=367)

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-05.jpg?height=333&width=650&top_left_y=289&top_left_x=388)

(a) All languages on three benchmarks

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-05.jpg?height=344&width=639&top_left_y=695&top_left_x=388)

(c) English and Chinese XStoryCloze

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-05.jpg?height=328&width=639&top_left_y=297&top_left_x=1098)

(b) Chinese XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-05.jpg?height=351&width=646&top_left_y=692&top_left_x=1092)

(d) English and Chinese XWinograd

Figure 1: Overall accuracy on all languages evaluated (Figure 1a and accuracy on English and Chinese that act as reference languages (Figures $1 \mathrm{~b}$ to $1 \mathrm{~d}$. 'instruct' in the round brackets indicates the scores are obtained by instruction-tuned models. 'base' stands for that obtained by the base model.

### 4.2 Language Exposure

Assessing the performance of multilingual models after instruction fine-tuning can be challenging, as it may involve evaluating their proficiency across multiple languages and tasks. The multilingual benchmarks do not cover the full list of languages used during pretraining and instruction tuning. We group the languages into four categories according to base and instruction-tuned models' language coverage:

1. languages are seen by the base and instruction models, including id, sw, ta, vi, zh (XCOPA); ar, en, es, hi, id, sw, te, zh (XStoryCloze); en, fr, pt, zh (Winograd)
2. languages are seen by the base model but not the instruction model, including ht (XCOPA) and eu (XStoryCloze)
3. languages are unseen by the base model but seen by the instruction model, including et, it, th, $\operatorname{tr}$ (XCOPA); my, ru (XStoryCloze); ru (XWinograd)
4. languages are unseen by the base nor instruction model, including qu (XCOPA) and jp (XWinograd)

We analyze the performance of multilingual instruction tuning with the increase in the number of languages in different groups with a focus on unseen languages by the base or instruction-tuned model. Cross-lingual transfer does happen in some cases, depending on benchmarks and languages.

Figure 2 displays the accuracy of languages seen by the base model but unseen during instruction fine-tuning. Multilingual instruction fine-tuning on languages other than the target languages (i.e., ht and eu) shows good cross-lingual transfer ability in most cases. And we also see a notable drop in eu in the XStoryCloze when Korean is added for instruction tuning.

Then, we look at whether multilingual instruction fine-tuning can adapt the base LLM to unseen languages during pretraining. Figure 3 shows the accuracy of languages that have not been used for training the base model but are used during instruction tuning. We find

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-06.jpg?height=353&width=615&top_left_y=298&top_left_x=430)

(a) XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-06.jpg?height=368&width=610&top_left_y=285&top_left_x=1126)

(b) XStoryCloze

Figure 2: Accuracy on languages seen by the base model but unseen by instruction-tuned model

that adding a language to instruction tuning can immediately improve the performance on that language as anticipated, especially for th in Figure 3c, my in Figure 3e and ru in Figure $3 \mathrm{f}$, the improvement is remarkable. But for ru in the XWinograd dataset in Figure $3 g$, the improved performance is still worse than the base model. We also observe that cross-lingual transfer is happening in the evaluated languages. For example, the performance on et and it in XCOPA shown in Figures $3 \mathrm{a}$ and $3 \mathrm{~b}$ can be further improved with the increase of instruction languages, and instruction fine-tuning on languages other than the target language tr can contribute to an improved performance than the base model as in Figure $3 \mathrm{~d}$

There are two languages that are not covered by the base model and instruction-tuned models, i.e., qu and jp. Figure 4 shows the results. The accuracy on qu fluctuates in general, but with a steadily increasing trend when adding $\mathrm{lt}, \mathrm{lv}, \mathrm{mk}, \mathrm{ml}, \mathrm{mn}$, and $\mathrm{mr}$. The accuracy on jp also fluctuates, but with more than 20 languages for instruction tuning is better than tuning with fewer than 20 languages in many cases. Moreover, the performance on jp shows an opposite result to the performance on other languages, and the average performance instruction fine-tuning is beneficial to jp in XWinograd that is completely unseen during pretraining or fine-tuning.

### 4.3 Language Similarity

We conduct a post-hoc analysis on how language closeness affects cross-lingual transfer. Instead of studying the relation between the number of fine-tuning languages and test set performance, we define an aggregated similarity measure between all languages present in a fine-tuning corpus and a test language $L_{\text {test }}$ :

$$
\text { similarity }_{\text {train, test }}=\sum_{L \in \text { corpus }} \operatorname{sim}\left(L, L_{\text {test }}\right)
$$

where $\operatorname{sim}($,$) is a similarity metric between two languages. We measure "aggregated$ similarity" instead of "average similarity" because we argue that, given their giant sizes, LLMs have the capacity to model all language data in the training set simultaneously.

We adopt different similarity measures based on syntactic, geographic, phonological, genetic, inventory, and featural distances scored by lang2vec (Littell et al. 2017, Malaviya et al., 2017) 1 In addition, we gathered from another source a language closeness score derived from sound (consonants) overlap which is deemed to reflect genetic similarity (Beaufils $\&$ Tomin, 2020) ${ }^{2}$ In total, we test out seven measures, where the similarity is always normalized to between 0 and 1, indicating the least and most similar. The choice of language closeness features is similar to a contemporaneous study on language transferability and similarity (Philippy et al. 2024). For comparison, we provide Pearson correlation coefficients[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=1263&width=1288&top_left_y=282&top_left_x=424)

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=353&width=596&top_left_y=298&top_left_x=431)

(a) et, the 10th mIT language, in XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=363&width=593&top_left_y=735&top_left_x=430)

(c) th, the 46th mIT language, in XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=360&width=593&top_left_y=1175&top_left_x=430)

(e) my, the 32nd mIT language, in XStoryCloze

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=352&width=596&top_left_y=301&top_left_x=1098)

(b) it, the 20th mIT language, in XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=363&width=591&top_left_y=735&top_left_x=1100)

(d) tr, the 48th mIT language, in XCOPA

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=366&width=594&top_left_y=1172&top_left_x=1096)

(f) ru, the 39th mIT language, in XStoryCloze

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-07.jpg?height=356&width=609&top_left_y=1619&top_left_x=758)

(g) ru, the 39 th mIT language, in XWingrad

Figure 3: Accuracy on languages unseen by the base model but seen by the instruction-tuned model. The subcaptions indicate the order of evaluated languages that are added to the instruction languages. The square mark in black indicates the target language evaluated.

between the number of languages and performance. Also, since empirically we observe that the checkpoint with the addition of Korean has an outlier performance, we also compute coefficients without that particular checkpoint.

The following observations are made on the benchmarks. Many factors contribute to the correlation: different languages, different tasks, and different similarity measures. lang2vec genetic features seem to stand out among all language features we test for, usually resulting

![](https://cdn.mathpix.com/cropped/2024_06_04_85b064e912216fe4b4f5g-08.jpg?height=412&width=697&top_left_y=301&top_left_x=714)

Figure 4: Accuracy on languages unseen by the base model but unseen by instruction-tuned model. Only XCOPA and XWinograd have languages under this category.

in a higher correlation than simply the number of languages. We also find that certain languages are affected by other languages more, e.g. th and sw, whereas certain languages have weak correlations with these features, e.g. en and zh. Also, across different test sets, the same language could display distinct behaviours such as ru in XStoryCloze and XWinograd.

|  | et | id | it | sw | ta | th | tr | vi | zh |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| num. lang. | 0.44 | 0.44 | 0.63 | 0.54 | -0.80 | 0.53 | 0.45 | -0.46 | -0.20 |
| num. lang. w/o ko | 0.44 | 0.50 | 0.64 | 0.54 | -0.80 | 0.53 | 0.46 | -0.50 | -0.39 |
| sound correspondence | 0.51 | 0.48 | 0.64 | 0.64 | -0.83 | 0.62 | 0.45 | -0.36 | -0.20 |
| lang2vec featural | 0.46 | 0.45 | 0.63 | 0.56 | -0.81 | 0.55 | 0.45 | -0.44 | -0.19 |
| lang2vec genetic | $\mathbf{0 . 6 7}$ | $\mathbf{0 . 5 8}$ | $\mathbf{0 . 6 7}$ | $\mathbf{0 . 9 3}$ | $\mathbf{- 0 . 8 4}$ | $\mathbf{0 . 8 2}$ | 0.47 | 0.02 | 0.01 |
| lang2vec geographic | 0.43 | 0.46 | 0.64 | $\mathbf{0 . 9 3}$ | -0.80 | 0.55 | 0.45 | -0.45 | 0.01 |
| lang2vec inventory | 0.46 | 0.45 | 0.64 | 0.52 | -0.80 | 0.55 | 0.45 | -0.45 | -0.19 |
| lang2vec phonological | 0.45 | 0.45 | 0.62 | 0.54 | -0.80 | 0.55 | 0.44 | -0.45 | -0.19 |
| lang2vec syntactic | 0.45 | 0.45 | 0.63 | 0.54 | -0.81 | 0.54 | 0.45 | -0.45 | -0.19 |

Table 1: Pearson correlation between XCOPA performance and training data similarity

|  | ar | en | es | hi | id | my | ru | sw | te | zh |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| num. lang. | -0.07 | 0.15 | 0.46 | 0.51 | 0.53 | 0.75 | 0.81 | 0.56 | -0.47 | 0.11 |
| num. lang. w/o ko | 0.08 | $\mathbf{0 . 4 1}$ | $\mathbf{0 . 7 3}$ | $\mathbf{0 . 6 6}$ | 0.63 | 0.75 | 0.86 | 0.56 | $\mathbf{- 0 . 5 3}$ | 0.31 |
| sound correspondence | -0.06 | 0.15 | 0.48 | 0.52 | 0.57 | 0.82 | 0.83 | 0.67 | -0.43 | 0.12 |
| lang2vec featural | -0.05 | 0.15 | 0.47 | 0.51 | 0.53 | 0.77 | 0.83 | 0.58 | -0.46 | 0.13 |
| lang2vec genetic | 0.17 | 0.16 | 0.50 | 0.54 | $\mathbf{0 . 6 6}$ | $\mathbf{0 . 9 6}$ | $\mathbf{0 . 8 7}$ | $\mathbf{0 . 9 6}$ | -0.26 | $\mathbf{0 . 3 7}$ |
| lang2vec geographic | 0.17 | 0.15 | 0.47 | 0.51 | 0.54 | 0.76 | 0.81 | $\mathbf{0 . 9 6}$ | -0.48 | $\mathbf{0 . 3 7}$ |
| lang2vec inventory | -0.06 | 0.15 | 0.46 | 0.51 | 0.54 | 0.76 | 0.83 | 0.55 | -0.46 | 0.13 |
| lang2vec phonological | -0.05 | 0.15 | 0.47 | 0.51 | 0.53 | 0.76 | 0.83 | 0.57 | -0.45 | 0.13 |
| lang2vec syntactic | -0.05 | 0.15 | 0.47 | 0.51 | 0.53 | 0.78 | 0.82 | 0.57 | -0.45 | 0.13 |

Table 2: Pearson correlation between XStoryCloze performance and training data similarity

## 5 Related Work

Multilingual LLMs Multilingual large language models have been trained to handle multiple languages, leveraging shared linguistic features across diverse language data. Many multilingual LLMs have emerged such as BLOOM (Scao et al., 2022), PaLM (Anil et al. 2023), MADLAD-400 (Kudugunta et al., 2024), and MaLA-500 (Lin et al., 2024). The following studies have been conducted to explore the capacity of multilingual LLMs. For example, Ye et al. (2023) examined the multilingual transfer ability of English-centric models compared to multilingual pretrained models, and Yuan et al. (2023) assessed the multilingual

|  | en | fr | jp | pt | ru | zh |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| num. lang. | -0.02 | 0.01 | 0.62 | -0.32 | -0.07 | 0.49 |
| num. lang. w/o ko | -0.03 | 0.00 | 0.66 | -0.35 | -0.07 | $\mathbf{0 . 5 0}$ |
| sound correspondence | -0.01 | -0.01 | 0.66 | -0.33 | -0.06 | 0.45 |
| lang2vec featural | -0.01 | 0.00 | 0.62 | -0.31 | -0.06 | 0.47 |
| lang2vec genetic | -0.02 | -0.08 | $\mathbf{0 . 7 2}$ | -0.35 | -0.05 | -0.31 |
| lang2vec geographic | -0.02 | -0.01 | 0.62 | -0.33 | -0.07 | -0.31 |
| lang2vec inventory | -0.01 | 0.00 | 0.62 | -0.31 | -0.06 | 0.48 |
| lang2vec phonological | -0.01 | 0.01 | 0.63 | -0.32 | -0.06 | 0.48 |
| lang2vec syntactic | -0.02 | 0.00 | 0.62 | -0.32 | -0.06 | 0.47 |

Table 3: Pearson correlation between XWinograd performance and training data similarity

capacity across 101 languages and revealed that existing LLMs exhibit greater multilingual capabilities than anticipated.

Multilingual Instruction Tuning Fine-tuning large multilingual language models has attracted much attention in recent years due to their capability to handle diverse linguistic contexts effectively. Kew et al. (2023) investigated the impact of multilinguality during finetuning on cross-lingual generalization. Chen et al. (2024) explored cost-efficient strategies for multilingual instruction tuning of large language models and compared monolingual and multilingual instruction tuning using different tuning methods. Shaham et al. (2024) studied the number of languages for instruction tuning using up to 11 languages and found that monolingual tuning transfers instruction-following capabilities across languages. Chirkova \& Nikoulina (2024a) explored the effectiveness of zero-shot cross-lingual generation by fine-tuning multilingual pretrained language models on generative tasks. Then, Chirkova \& Nikoulina (2024b) further expanded the study to zero-shot cross-lingual transter of large language models. Both studies emphasized the importance of model configuration. While previous works tried to constrain the amount of training data and squeeze the number of languages, this work attempts to scale up the number of languages, yielding additional findings to (zero-shot) language transfer. In a small-scale experiment, the authors of the Bactrian-X dataset discovered that monolingual models fine-tuned with low-rank adaptation are better than a single multilingual model (Li et al., 2023), which is different from our findings albeit with very different experimental setups.

## 6 Conclusion and Future Work

Instruction fine-tuning of large multilingual language models presents both opportunities and challenges. While it can enable versatile language processing capabilities, it also demands careful handling to address issues related to language-specific nuances. Various studies analyzed multilingualism and cross-lingual transferability in different settings, leading to different conclusions. This paper conducts an experimental analysis on yet another set of settings and reveals different findings from prior works, showing that even with the same base model, instruction data, and training recipe, the performance is still dependent on the evaluation tasks. Building upon the findings of prior works and our study, we conclude that the performance of multilingual instruction fine-tuning is highly dependent on factors such as base models, instruction data, tasks, and evaluation protocols. Our study stresses the importance of the theoretical study on the effectiveness and generalizability of multilingual instruction fine-tuning, as well as the need for more systematic experimental studies to validate the theory.

## Limitations

Our work studies multilingual instruction fine-tuning in 52 languages, which might be small compared to thousands of living languages. We did not conduct a human evaluation
due to limited budget constraints. Future work would be to conduct a more systematic assessment with more rigorously controlled variables during instruction tuning.

## Ethical Considerations

Multilingual instruction fine-tuning also raises ethical questions about the responsible use of AI models, especially when dealing with diverse linguistic and cultural contexts. This paper investigates the challenge in instruction fine-tuning large multilingual language models and learning lessons to facilitate future research to harness the full potential of multilingual models while minimizing potential drawbacks.

## Reproducibility Statement

We make the following efforts to ensure reproducible research. Our base model, instruction dataset, and evaluation benchmarks are all open-source. We release the fine-tuned model weights. Our results are reproducible using the provided model weights and evaluation scripts.

## Acknowledgements

The work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government's Horizon Europe funding guarantee [grant number 10052546]. Shaoxiong Ji also received funding from UTTER's Financial Support for Third Parties. We acknowledge CSC-IT Center for Science, Finland for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through Finnish extreme scale call (project LumiNMT and MOOMIN) and the call from the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254).

## References

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. The Falcon series of open language models. arXiv preprint, 2023.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 technical report. arXiv preprint, 2023.

Vincent Beaufils and Johannes Tomin. Stochastic approach to worldwide language classification: the signals and the noise towards long-range exploration. SocArXiv, 2020.

Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, and Kenneth Heafield. Monolingual or multilingual instruction tuning: Which makes a better alpaca. In Findings of the Association for Computational Linguistics: EACL 2024, 2024.

Nadezhda Chirkova and Vassilina Nikoulina. Key ingredients for effective zero-shot crosslingual knowledge transfer in generative tasks. arXiv preprint, 2024a.

Nadezhda Chirkova and Vassilina Nikoulina. Zero-shot cross-lingual transfer in instruction tuning of large language model. arXiv preprint, 2024b.

Mike Conover, Matt Hayes, Ankit Mathur, Xiangrui Meng, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free Dolly: Introducing the world's first truly open instruction-tuned LLM. https://www.databricks.com. 2023.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. A framework for few-shot language model evaluation. Zenodo, 2023.

Tannon Kew, Florian Schottmann, and Rico Sennrich. Turning english-centric llms into polyglots: How much multilinguality is needed? arXiv preprint, 2023.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. Madlad-400: A multilingual and documentlevel large audited dataset. Advances in Neural Information Processing Systems, 2024.

Hector Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In Thirteenth international conference on the principles of knowledge representation and reasoning, 2012.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation. arXiv preprint, 2023.

Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André FT Martins, and Hinrich Schütze. MaLA-500: Massive language adaptation of large language models. arXiv preprint arXiv:2401.13303, 2024.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. Few-shot learning with multilingual language models. arXiv preprint, 2021.

Patrick Littell, David R. Mortensen, Ke Lin, Katherine Kairis, Carlisle Turner, and Lori Levin. URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, 2017.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. Transactions of the Association for Computational Linguistics, 2020.

Chaitanya Malaviya, Graham Neubig, and Patrick Littell. Learning language representations for typology prediction. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 2017.

Nasrin Mostafazadeh, Michael Roth, Annie Louis, Nathanael Chambers, and James Allen. Lsdsem 2017 shared task: The story cloze test. In Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics, 2017.

Fred Philippy, Siwen Guo, Shohreh Haddadan, Cedric Lothritz, Jacques Klein, and Tegawendé F. Bissyandé. Soft prompt tuning for cross-lingual transfer: When less is more. In Proceedings of the 1st Workshop on Modular and Open Multilingual NLP (MOOMIN 2024), 2024.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery $\mathcal{E}$ Data Mining, 2020.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. BLOOM: A 176B-parameter open-access multilingual language model. arXiv preprint, 2022.

Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. Multilingual instruction tuning with just a pinch of multilinguality. arXiv preprint, 2024.

Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Vladislav Mikhailov, Anastasia Kozlova, and Tatiana Shavrina. mGPT: Few-shot learners go multilingual. arXiv preprint, 2022.

Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, et al. Aya dataset: An open-access collection for multilingual instruction tuning. arXiv preprint, 2024.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Alexey Tikhonov and Max Ryabinin. It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint, 2023.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint, 2019.

Jiacheng Ye, Xijia Tao, and Lingpeng Kong. Language versatilists vs. specialists: An empirical revisiting on multilingual transfer ability. arXiv preprint, 2023.

Fei Yuan, Shuai Yuan, Zhiyong Wu, and Lei Li. How multilingual is multilingual llm? arXiv preprint, 2023.


[^0]:    We release all model weights to Hugging Face at this link

    * Fun fact: Lucky 52 was a famous variety show in China in the late 1990s and early 2000s. In this work, the term is used to denote the 52 languages used in multilingual instruction tuning.

[^1]:    ${ }^{1}$ https://github.com/antonisa/lang2vec

    2 http://www.elinguistics.net/language_evolution.html

</end of paper 1>


<paper 2>
# Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca 

Pinzhen Chen ${ }^{1, *} \quad$ Shaoxiong $\mathbf{J i}^{2, *} \quad$ Nikolay Bogoychev ${ }^{1}$<br>Andrey Kutuzov ${ }^{3} \quad$ Barry Haddow $^{1} \quad$ Kenneth Heafield $^{1}$<br>${ }^{1}$ University of Edinburgh $\quad{ }^{2}$ University of Helsinki $\quad{ }^{3}$ University of Oslo<br>pchen3@ed.ac.uk shaoxiong.ji@helsinki.fi


#### Abstract

Foundational large language models (LLMs) can be instruction-tuned to perform opendomain question answering, facilitating applications like chat assistants. While such efforts are often carried out in a single language, we empirically analyze cost-efficient strategies for multilingual scenarios. Our study employs the Alpaca dataset and machine translations of it to form multilingual data, which is then used to tune LLMs through either low-rank adaptation or full-parameter training. Under a controlled computation budget, comparisons show that multilingual tuning is on par or better than tuning a model for each language. Furthermore, multilingual tuning with downsampled data can be as powerful and more robust. Our findings serve as a guide for expanding language support through instruction tuning.


## 1 Introduction

Language capacity has attracted much attention in pre-trained language models. Some pioneering works focused on a single language (Peters et al., 2018; Devlin et al., 2019), while later works aim to cover multiple languages (Conneau et al., 2020; Liu et al., 2020). In the recent blossom of open-source LLMs, English-centric ones include GPT-2, LLaMA, and Pythia (Radford et al., 2019; Touvron et al., 2023; Biderman et al., 2023), and multilingual ones are represented by BLOOM (Scao et al., 2022). Multilingual models seem attractive when considering operational costs, cross-lingual transfer, and low-resource languages (Artetxe and Schwenk, 2019; Wu and Dredze, 2020), yet English-centric models can possess good multilingual transferability (Ye et al., 2023).

Instruction tuning makes LLMs follow and respond to inputs (Sanh et al., 2022; Wei et al., 2022).[^0]

With multilingual instruction data becoming feasible and available, this paper compares monolingual and multilingual instruction tuning applied to English-centric and multilingual LLMs to search for the optimal strategy to support multiple languages. Unlike prior works on multilingual multiNLP-task tuning (Mishra et al., 2022; Muennighoff et al., 2023), we focus on open-ended question answering under language generation.

Our data setting combines two low-cost practices: self-instruct, which distils data from a powerful LLM (Wang et al., 2023; Taori et al., 2023) and the idea of leveraging machine translation to create multilingual datasets (Muennighoff et al., 2023). We fine-tune several decoder LLMs with either full-parameter fine-tuning (FFT) or low-rank adaptation (LoRA, Hu et al., 2022) with different language combinations. Our experiments feature a fixed computation budget to offer practical insights. It is shown that multilingual tuning is preferred to monolingual tuning for each language under LoRA, but the results are mixed under FFT. English-tuned LLMs are not well-versed in responding in other languages, whereas a downsampled multilingual tuning scheme proposed by us is more robust. Finally, we examine our model performance on unseen languages and various LLMs of roughly the same size.

## 2 Methodology

### 2.1 Instruction data

We use the Alpaca dataset as a seed to create a multilingual instruction-response dataset. We used the cleaned version with $52 \mathrm{~K}$ instances ${ }^{1}$ and machinetranslated it into eight languages: Bulgarian, Czech, Chinese, German, Finnish, French, Russian, and Spanish, using open-source translation systems. ${ }^{2}$[^1]

### 2.2 Budget-controlled instruction tuning

For monolingual tuning, we tune LLMs for each language separately, whereas for multilingual tuning, we merge and shuffle the data in all languages. This allows for resource-controlled comparisons between monolingual and multilingual tuning, where a fixed (and equal for each language) computation budget is allocated to support all languages of interest. Experimental resource usage is described as follows:

1) Let $C_{\text {Alpaca }}$ denote the cost of monolingual Alpaca fine-tuning for a single language, then it costs $N \times C_{\text {Alpaca }}$ to tune individual models to support $N$ languages.
2) Multilingual instruction tuning will cost $N \times C_{\text {Alpaca }}$ too, as it trains on data available in all $N$ languages in one go.

We can fairly compare LLMs trained via 1) and 2) for any language. In addition, we propose to benchmark two budget-saving options which cost the same $C_{\text {Alpaca }}$ as a monolingual Alpaca:

3) As a simple baseline, we use an English-tuned model to respond to all languages.
4) Downsampled multilingual: we randomly sample from the multilingual data in 2) to have the size of a monolingual dataset.

Our study covers two training paradigms: low rank adaptation and full-parameter fine-tuning Both fine-tune an LLM with the causal language modelling objective on the instruction-response data, with hyperparameters listed in Appendix A.1. Five LLMs are involved: Baichuan-2, BLOOM, LLaMA, OpenLLaMA, and Pythia, aiming to test with different language coverage in the base LLMs Pythia, LLaMA, and OpenLLaMA are predominantly English, while Baichuan-2 and BLOOM are more versatile. A detailed description of the LLMs is in Appendix A.2.

### 2.3 Evaluation setup

Test data Our instruction-tuned LLMs are benchmarked on languages both seen and unseen during fine-tuning. We employ native speakers to manually translate 50 prompts sampled from OpenAssistant (Köpf et al., 2023) into eight languages: six seen during training and two unseen. The seen category includes English, French, Spanish, Bulgarian, Russian, and Chinese. Among the six, English is the highest-resourced, followed by French and Spanish which share the same script as English. Bulgarian and Russian are European languages but use a writing system distinct from English. Finally, Chinese is a high-resource distant language in a different script. For unseen tests, we pick Bengali and Norwegian. Bengali is distant from the above languages and uses a different script, whereas Norwegian is under-resourced but overlaps with English writing script to some extent.

LLM-as-a-judge To avoid expensive evaluation costs, we adopt LLM-as-a-judge (Zheng et al., 2023) to assign a score ( 1 to 3 ) to each instructionresponse pair, and the final model score is the sum of its scores across all test instances. We use GPT-3.5 (gpt-3.5-turbo-0613) as the judge; it is queried with an instruction-response pair each time without model information or request history. We make modifications to Zheng et al. (2023)'s prompt to ask the LLM to consider that an answer should be in the same language as the question, which is often the expectation with AI assistants. ${ }^{3}$ The exact wording is as Appendix B. 1 Figure 6.

Language (in)consistency Our manual inspection suggests that GPT-3.5 does not always obey the language requirement imposed. An example in Appendix B. 2 Table 2 shows a response in another language but scored highly. Hence, we run language identification and force-set a score to 0 if the response language is different from the query. We use the fastText framework (Joulin et al., 2017) with Burchell et al. (2023)'s checkpoint. The final response score can be framed as a product of GPT's quality score and a binary language identification outcome: score $=$ eval_score $\times$ lang_id. The aggregated test score thus ranges from 0 to 150 .

Human-LLM agreement We pick 600 outputs from 12 models to cover multilingual and monolingual systems and invite human evaluators to score each sample with an instruction similar to the LLM-as-a-judge prompt as in Appendix B.3. Four languages-English, Spanish, Bulgarian, and Chinese-are human-evaluated, and we obtain very high system-level Pearson correlation coefficients of $0.9225,0.9683,0.9205$, and 0.8685 , respectively between GPT-3.5 and human. Details are in Table 3 in the appendix. This indicates the reliability of using LLM-as-a-judge to draw meaningful findings.[^2]![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-03.jpg?height=1092&width=782&top_left_y=230&top_left_x=246)

French
![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-03.jpg?height=430&width=702&top_left_y=885&top_left_x=262)

Figure 1: LoRA with BLOOM at different sizes. Caption: language; y-axis: score; x-axis: model size (B).

## 3 Performance and Discussions

### 3.1 Model sizes

Results from LoRA fine-tuning of BLOOM at different sizes are shown in Figure 1. At smaller sizes, multilingual $(-)$ and monolingual $(*)$ instruction tuning attain similar performance, and at larger sizes, multilingual models are generally better except for English. We observe similar trends for Pythia, placed in Appendix C. 1 Figure 8 due to space constraints. Moving on to full-parameter fine-tuning of BLOOM in Figure 2, we discover that at relatively small $(<1.7 \mathrm{~B}$ ) or large sizes (7B), monolingual models are generally better than multilingual models for individual languages. These observations suggest that multilingualism works well with LoRA, but separate monolingual tuning might be better with FFT. Overall, the LLMs' performance is correlated with sizes regardless of the tuning technique as anticipated.

### 3.2 Budget-efficient tuning

To aid our exploration of resource-constrained instruction tuning, in the aforementioned Figures 1, 2 , and 8 (in appendix C.1), we add the plots of two budget data conditions: using English-tuned mod-
![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-03.jpg?height=996&width=784&top_left_y=233&top_left_x=1064)

Figure 2: FFT with BLOOM at different sizes. Caption: language; $\mathrm{y}$-axis: score; $\mathrm{x}$-axis: model size (B). Same legend as Figure 1.

els to respond to instructions in other languages $(--)$, as well as instruction tuning with downsampled multilingual data $(\triangle \triangle)$.

When using a single English model for all languages, its efficacy depends on the intended language/script's closeness to English: Spanish and French can maintain reasonable scores, but Bulgarian, Russian, and Chinese record very low performance. The only exception is BLOOM FFT in Figure 2, where the model is not too behind when operating in Chinese. Interestingly, BLOOM with LoRA sees a performance spike at 1.1B for nonEnglish. At this specific size, it displayed multilingual transferability from pre-training and learned to follow multilingual instructions despite being fine-tuned merely in English.

In contrast, while consuming the same computational resources, downsampled multilingual tuning is significantly more robust across all test languages. These models sometimes achieve on-par performance with monolingual tuning in individual languages. This means that to support several languages with limited resources, the best practice is to train on small multilingual data even created with machine translation instead of full English data. Nonetheless, if the budget permits, training with the full multilingual data is still slightly better.
![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-04.jpg?height=656&width=768&top_left_y=228&top_left_x=250)

LoRA, Bengali
![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-04.jpg?height=646&width=760&top_left_y=232&top_left_x=268)

Full, Bengali

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-04.jpg?height=317&width=351&top_left_y=561&top_left_x=658)

Figure 3: LoRA and FFT with BLOOM at different sizes and tested on unseen languages. Caption: training method and language; $\mathrm{y}$-axis: score; $\mathrm{x}$-axis: model size (B).

### 3.3 Unseen languages

Further in Figure 3, we look at BLOOM models which underwent LoRA or FFT but were subsequently instructed in unseen languages at test time. English-tuned LLMs behave distinctly with LoRA and FFT. With the former, they are nowhere near multilingual tuned models, but with the latter, we see close or even better results. It might imply that FFT can even lift performance for languages not present in the instruction data. However, FFT results on Norwegian could be an outlier given its comparably low scores. Considering multilingual instruction tuning, we notice a pattern opposed to that on languages seen during training-learning on the downsampled data is superior to ingesting the full mixed data. We conclude that it is important to not overfit to instruction languages if unseen languages are expected in downstream tasks.

### 3.4 Language robustness

We review each model and data recipe's scores before and after adding language identification, to isolate an LLM's language robustness from its "inherent quality" (regardless of the response language). We compute the differences in GPT evaluation scores before and after applying language identification. A (big) difference suggests that a model produces reasonable answers in an undesired language. In Figure 4, we report the average of the score differences across all six test languages seen during tuning. English-only models are the least robust-their score differences are way above other techniques. With LoRA, full multilingual tuning records the smallest performance drop; with FFT,
![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-04.jpg?height=676&width=782&top_left_y=233&top_left_x=1068)

Figure 4: Evaluation score change before and after language identification, averaged over six seen test languages, at different LLM sizes. Caption: training method and base model; y-axis: score difference (log scale); x-axis: model size (B).

monolingual tuning is preferred. The insights from language robustness are corroborated by our early findings in Section 3.1: superior results are obtained when using multilingual tuning with LoRA and monolingual tuning with full-parameter tuning. Nonetheless, monolingual and multilingual tuning are not too far apart; specifically for BLOOM with LoRA, language robustness does not improve as the model gets larger.

### 3.5 Model families

Finally, we experiment with base LLMs from different families of around 7 billion parameters. In Figure 5, we plot the evaluation scores for multilingual, downsampled multilingual, and monolingual LoRA tuning for six languages. Generally, LLaMA and OpenLLaMA have better performance than BLOOM and Pythia potentially because they have pre-training data that is an order of magnitude larger. Also Bulgarian, Russian, and Chinese see lower scores than English, again presumably due to the language distribution in the pre-training data.

Delving into the comparison between monolingual and multilingual instruction tuning, we find that out of 30 cases across six languages and five LLMs, monolingual tuning is ahead in just two cases: LLaMA tested in Russian and Chinese. The cost-efficient downsampled multilingual tuning leads in four cases: two in French and two in Russian. In other situations, multilingual training is on par if not better. The outcome of tuning several similar-sized LLMs confirms that multilingual tuning is favourable using LoRA.

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=928&width=1562&top_left_y=233&top_left_x=247)

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=374&width=494&top_left_y=247&top_left_x=267)

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=465&width=505&top_left_y=624&top_left_x=273)

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=368&width=485&top_left_y=247&top_left_x=794)

Spanish

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=386&width=497&top_left_y=658&top_left_x=788)

Russian

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=372&width=505&top_left_y=248&top_left_x=1301)

French

![](https://cdn.mathpix.com/cropped/2024_06_04_621040288a12b44f8c95g-05.jpg?height=388&width=497&top_left_y=660&top_left_x=1296)

Chinese

Figure 5: LoRA fine-tuning on different 7B LLMs. Caption: language generated; $\mathrm{y}$-axis: score; $\mathrm{x}$-axis: model family.

## 4 Related Work

Many large language models appeared recently: the closed-source GPT model family (Radford et al., 2019; Brown et al., 2020; Ouyang et al., 2022); open-source English-centric models like LLaMA (Touvron et al., 2023), OpenLLaMA (Geng and Liu, 2023), and Pythia (Biderman et al., 2023); open-source multilingual models like mT5 (Xue et al., 2021) and BLOOM (Scao et al., 2022). These models have exhibited different degrees of language versatility.

LLM pre-training data is usually skewed towards English. One way to improve an LLM's coverage of non-English languages is through continued pretraining (Cui et al., 2023, inter alia). Another rich body of literature looks into multilingualism in instruction tuning, which is used to adjust base models to respond to input (Mishra et al., 2022; Sanh et al., 2022; Wei et al., 2022; Longpre et al., 2023). It trains an LLM by providing downstream tasks' input and output in a specific format. Early research created a multilingual instruction dataset using machine translation and showed that multilingual tuning gained higher performance than English-only fine-tuning (Muennighoff et al., 2023). They also found that low-cost translated instructions are superior to human-written non-English prompts on multiple language understanding tasks.

Lately, multiple contemporaneous papers delv- ing into multilingual instruction tuning have been made public on arXiv-some appeared before our work and some after. This reflects the importance and interest in widening LLMs' language support. Li et al. (2023a) created an instruction dataset with instructions translated from English but responses generated by an LLM. When tuned with LoRA, their monolingual models outperform multilingual ones on language understanding tasks. Wei et al. (2023) created a multilingual counterpart of Alpaca using self-instruct. It has also been showcased that translation instructions improve cross-lingual capabilities (Li et al., 2023b; Zhang et al., 2023; Ranaldi et al., 2023) and research explored more crosslingual task data and multilingual tuning (Zhu et al., 2023). Moreover, researchers have unveiled that fine-tuning on a modest number of languagesapproximately three-seems to effectively instigate cross-lingual transfer in downstream tasks (Kew et al., 2023; Shaham et al., 2024).

## 5 Conclusion

This paper presents a study of instruction tuning of large language models in different language contexts. Our study in a resource-controlled setting suggests that multilingual tuning offers more benefits compared to monolingual tuning. We find that multilingual tuning on a downsampled dataset achieves better robustness on unseen languages.

## Limitations

The LLMs we studied have primarily 7B and at most 13B parameters and the multilingual training only spanned nine languages. Scaling to larger models and more languages would be interesting. The best checkpoint for our instruction fine-tuning is selected based on validation cross-entropy, but there is no guarantee that this leads to the best performance on the downstream task.

To manage the budget for human translation and evaluation, we consider eight languages (six seen and two unseen languages during instruction tuning) to translate and sample 50 instances for evaluation. The training data for non-English languages are obtained via machine translation, which introduces errors, affects response fluency, and might alter the nature of some tasks such as grammatical error correction and code generation.

## Ethics Statement

The dataset we translated and generated does not contain private or sensitive information. Similar to other research on large language models, there is no definitive way for us to prevent the instructiontuned models from generating inappropriate content. However, we see minimal such risks associated with our project, as neither our models nor generated contents are intended for public consumption. Human evaluators did not report inappropriate content generated by the models.

## Acknowledgements

This paper stemmed from a hackathon project organized by the High Performance Language Technologies (HPLT) consortium. ${ }^{4}$ We are grateful to Alicia Núñez Alcover, David Samuel, Joona Kytöniemi, Jörg Tiedemann, Lucas Charpentier, Sampo Pyysalo, Petter Mæhlum, and Zhicheng Guo for project discussions, test data translation, and evaluation setup.

The work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No 101070350, from UK Research and Innovation (UKRI) under the UK government's Horizon Europe funding guarantee [grant number 10052546], as well as from the European Research Council (ERC) under the EU's Horizon 2020 research and innovation program (agreement № 771113).[^3]

Computation in this work was performed on LUMI, Karolina, and Baskerville. We acknowledge CSC-IT Center for Science, Finland for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through Finnish extreme scale call (project LumiNMT). Karolina was supported by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254). The Baskerville Tier 2 HPC was funded by the EPSRC and UKRI through the World Class Labs scheme (EP/T022221/1) and the Digital Research Infrastructure programme (EP/W032244/1) and is operated by Advanced Research Computing at the University of Birmingham.

## References

Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zeroshot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597-610.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems.

Laurie Burchell, Alexandra Birch, Nikolay Bogoychev, and Kenneth Heafield. 2023. An open dataset and model for language identification. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers).

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for Chinese LLaMA and Alpaca. arXiv preprint.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020 The pile: An $800 \mathrm{~GB}$ dataset of diverse text for language modeling. arXiv preprint.

Xinyang Geng and Hao Liu. 2023. OpenLLaMA: An open reproduction of LLaMA. GitHub repository.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2017. Bag of tricks for efficient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers.

Tannon Kew, Florian Schottmann, and Rico Sennrich. 2023. Turning English-centric LLMs into polyglots: How much multilinguality is needed? arXiv preprint.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. 2023. OpenAssistant conversations-democratizing large language model alignment. arXiv preprint.

Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, et al. 2022. The bigscience ROOTS corpus: A 1.6TB composite multilingual dataset. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023a. Bactrian-X: A multilingual replicable instruction-following model with low-rank adaptation. arXiv preprint.

Jiahuan Li, Hao Zhou, Shujian Huang, Shanbo Chen, and Jiajun Chen. 2023b. Eliciting the translation ability of large language models via multilingual finetuning with translation instructions. arXiv preprint.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726-742.
Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. The Flan collection: Designing data and methods for effective instruction tuning. In Proceedings of the 40th International Conference on Machine Learning.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2023. Crosslingual generalization through multitask finetuning. In Proceedings of the 61 st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers).

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog.

Leonardo Ranaldi, Giulia Pucci, and Andre Freitas. 2023. Empowering cross-lingual abilities of instruction-tuned large language models by translation-following demonstrations. arXiv preprint.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. BLOOM: A 176Bparameter open-access multilingual language model. arXiv preprint.

Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. 2024. Multilingual instruction tuning with just a pinch of multilinguality. arXiv preprint.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An instruction-following LLaMA model. GitHub repository.

Together Computer. 2023. RedPajama: An open source recipe to reproduce LLaMA training dataset. GitHub repository.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. LLaMA: Open and efficient foundation language models. arXiv preprint.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2022. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023. Polylm: An open source polyglot large language model. arXiv preprint.

Shijie Wu and Mark Dredze. 2020. Are all languages created equal in multilingual BERT? In Proceedings of the 5th Workshop on Representation Learning for $N L P$.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, et al. 2023. Baichuan 2: Open largescale language models. arXiv preprint.

Jiacheng Ye, Xijia Tao, and Lingpeng Kong. 2023. Language versatilists vs. specialists: An empirical revisiting on multilingual transfer ability. arXiv preprint.

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, et al. 2023. BayLing: Bridging cross-lingual alignment and instruction following through interactive translation for large language models. arXiv preprint.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023.
Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Extrapolating large language models to non-english by aligning languages. arXiv preprint.
</end of paper 2>


