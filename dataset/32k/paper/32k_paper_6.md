<paper 0>
# Very Deep Transformers for Neural Machine Translation 

Xiaodong Liu ${ }^{\dagger}$, Kevin Duh ${ }^{\ddagger}$, Liyuan Liu ${ }^{\S}$ and Jianfeng Gao ${ }^{\dagger}$<br>${ }^{\dagger}$ Microsoft Research $\quad$ Johns Hopkins University<br>${ }^{\S}$ University of Illinois at Urbana-Champaign<br>\{xiaodl, jfgao\}@microsoft.com<br>kevinduh@cs.jhu.edu, ll2@illinois.edu


#### Abstract

We explore the application of very deep Transformer models for Neural Machine Translation (NMT). Using a simple yet effective initialization technique that stabilizes training, we show that it is feasible to build standard Transformerbased models with up to 60 encoder layers and 12 decoder layers. These deep models outperform their baseline 6-layer counterparts by as much as 2.5 BLEU, and achieve new state-of-the-art benchmark results on WMT14 English-French (43.8 BLEU and 46.4 BLEU with back-translation) and WMT14 EnglishGerman (30.1 BLEU). To facilitate further research in Very Deep Transformers for NMT, we release the code and models: https:// github.com/namisan/exdeep-nmt.


## 1 Introduction

The capacity of a neural network influences its ability to model complex functions. In particular, it has been argued that deeper models are conducive to more expressive features (Bengio, 2009). Very deep neural network models have proved successful in computer vision (He et al., 2016; Srivastava et al., 2015) and text classification (Conneau et al., 2017; Minaee et al., 2020). In neural machine translation (NMT), however, current state-of-the-art models such as the Transformer typically employ only 612 layers (Bawden et al., 2019; Junczys-Dowmunt, 2019; Ng et al., 2019).

Previous work has shown that it is difficult to train deep Transformers, such as those over 12 layers (Bapna et al., 2018). This is due to optimization challenges: the variance of the output at each layer compounds as they get deeper, leading to unstable gradients and ultimately diverged training runs.

In this empirical study, we re-investigate whether deeper Transformer models are useful for NMT. We apply a recent initialization technique called ADMIN (Liu et al., 2020a), which remedies the vari-

![](https://cdn.mathpix.com/cropped/2024_06_04_b04d3bc9a123d152859bg-1.jpg?height=577&width=574&top_left_y=774&top_left_x=1158)

Figure 1: Transformer model

ance problem. This enables us train Transformers that are significantly deeper, e.g. with 60 encoder layers and 12 decoder layers. ${ }^{1}$

In contrast to previous research, we show that it is indeed feasible to train the standard ${ }^{2}$ Transformer (Vaswani et al., 2017) with many layers. These deep models significantly outperform their 6-layer baseline, with up to 2.5 BLEU improvement. Further, they obtain state-of-the-art on the WMT' 14 EN-FR and WMT' 14 EN-DE benchmarks.

## 2 Background

We focus on the Transformer model (Vaswani et al., 2017), shown in Figure 1. The encoder consists of $N$ layers/blocks of attention + feedforward components. The decoder consists of $M$ layers/blocks of masked-attention, attention, and feed-forward components. To illustrate, the in-[^0]put tensor $\mathbf{x}_{\mathbf{i}-\mathbf{1}}$ at the encoder is first transformed by a multi-head attention mechanism to generate the tensor $f_{A T T}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)$. This result is added back with $\mathbf{x}_{\mathbf{i}-\mathbf{1}}$ as a residual connection, then layernormalization $\left(f_{L N}(\cdot)\right)$ is applied to generate the output: $\mathbf{x}_{\mathbf{i}}=f_{L N}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}+f_{A T T}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)\right)$. Continuing onto the next component, $\mathbf{x}_{\mathbf{i}}$ is passed through a feed-forward network $f_{F F}(\cdot)$, and is again added and layer-normalized to generate the output tensor: $\mathbf{x}_{\mathbf{i}+\mathbf{1}}=f_{L N}\left(\mathbf{x}_{\mathbf{i}}+f_{F F}\left(\mathbf{x}_{\mathbf{i}}\right)\right)$. Abstractly, the output tensor at each Add+Norm component in the Transformer (Figure 1) can be expressed as:

$$
\begin{equation*}
\mathbf{x}_{\mathbf{i}}=f_{L N}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}+f_{i}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)\right) \tag{1}
\end{equation*}
$$

where $f_{i}$ represents a attention, masked-attention, or feed-forward subnetwork. This process repeats $2 \times N$ times for a $N$-layer encoder and $3 \times M$ times for a $M$-layer decoder. The final output of the decoder is passed through a softmax layer which predicts the probabilities of output words, and the entire network is optimized via back-propagation.

Optimization difficulty has been attributed to vanishing gradient, despite layer normalization ( $\mathrm{Xu}$ et al., 2019) providing some mitigation. The lack of gradient flow between the decoder and the lower layers of the encoder is especially problematic; this can be addressed with short-cut connections (Bapna et al., 2018; He et al., 2018). An orthogonal solution is to swap the positions of layerwise normalization $f_{L N}$ and subnetworks $f_{i}$ within each block (Nguyen and Salazar, 2019; Domhan, 2018; Chen et al., 2018) by: $\mathbf{x}_{\mathbf{i}}=f_{i}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}+f_{L N}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)\right)$ This is known as pre-LN (contrasted with post-LN in Eq. 1), and has been effective in training networks up to 30 layers (Wang et al., 2019). ${ }^{3}$

However, it has been shown that post-LN, if trained well, can outperform pre-LN (Liu et al., 2020a). Ideally, we hope to train a standard Transformer without additional architecture modifications. In this sense, our motivation is similar to that of Wu et al. (2019b), which grows the depth of a standard Transformer in a stage-wise fashion.

## 3 Initialization Technique

The initialization technique ADMIN (Liu et al., 2020a) we will apply here reformulates Eq. 1 as:

$$
\begin{equation*}
\mathbf{x}_{\mathbf{i}}=f_{L N}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}} \cdot \omega_{\mathbf{i}}+f_{i}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)\right) \tag{2}
\end{equation*}
$$

where $\omega_{i}$ is a constant vector that is element-wise multiplied to $\mathbf{x}_{\mathbf{i}-\mathbf{1}}$ in order to balance the contribution against $f_{i}\left(\mathbf{x}_{\mathbf{i}-1}\right)$. The observation is that in[^1]

addition to vanishing gradients, the unequal magnitudes in the two terms $\mathbf{x}_{\mathbf{i}-\mathbf{1}}$ and $f_{i}\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)$ is the main cause of instability in training. Refer to (Liu et al., 2020a) for theoretical details. ${ }^{4}$

ADMIN initialization involves two phases: At the Profiling phase, we randomly initialize the model parameters using default initialization, set $\omega_{\mathrm{i}}=1$, and perform one step forward pass in order to compute the output variance of the residual branch $\operatorname{Var}\left[f\left(\mathbf{x}_{\mathbf{i}-\mathbf{1}}\right)\right]$ at each layer. ${ }^{5}$ In the Training phase, we fix $\omega_{\mathbf{i}}=\sqrt{\sum_{j<i} \operatorname{Var}\left[f\left(\mathbf{x}_{\mathbf{j}-\mathbf{1}}\right)\right]}$, and then train the model using standard backpropagation. After training finishes, $\omega_{\mathrm{i}}$ can be folded back into the model parameters to recover the standard Transformer architecture. This simple initialization method is effective in ensuring that training does not diverge, even in deep networks.

## 4 Experiments

Experiments are conducted on standard WMT' 14 English-French (FR) and English-German (DE) benchmarks. For FR, we mimic the setup ${ }^{6}$ of (Ott et al., 2018), with $36 \mathrm{M}$ training sentences and $40 \mathrm{k}$ subword vocabulary. We use the provided 'valid' file for development and newstest14 for test. For DE, we mimic the setup ${ }^{7}$ of (So et al., 2019), with $4.5 \mathrm{M}$ training sentences, $32 \mathrm{~K}$ subword vocabulary, newstest2013 for dev, and newstest2014 for test.

We adopt the hyper-parameters of the Transformer-based model (Vaswani et al., 2017) as implemented in FAIRSEQ (Ott et al., 2019), i.e. 512-dim word embedding, 2048 feed-forward model size, and 8 heads, but vary the number of layers. RAdam (Liu et al., 2019) is our optimizer. ${ }^{8}$

Main Result: Our goal is to explore whether very deep Transformers are feasible and effective. We compare: (a) 6L-6L: a baseline Transformer Base with 6 layer encoder and 6 layer decoder, vs. (b) $\mathbf{6 0 L - 1 2 L}$ : A deep transformer with 60 encoder[^2]

| Model | WMT'14 English-French (FR) |  |  |  |  | WMT'14 English-German (DE) |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | \#param | $\mathbf{T} \downarrow$ | $\mathbf{M} \uparrow$ | $\mathbf{B L E U} \uparrow$ | $\Delta$ | \#param | $\mathbf{T} \downarrow$ | $\mathbf{M} \uparrow$ | $\mathbf{B L E U} \uparrow$ | $\Delta$ |
| 6L-6L Default | $67 \mathrm{M}$ | 42.2 | 60.5 | 41.3 | - | $61 \mathrm{M}$ | 54.4 | 46.6 | 27.6 | - |
| 6L-6L ADMIN | $67 \mathrm{M}$ | 41.8 | 60.7 | 41.5 | 0.2 | $61 \mathrm{M}$ | 54.1 | 46.7 | 27.7 | 0.1 |
| 60L-12L Default | $262 \mathrm{M}$ |  |  | rge |  | $256 \mathrm{M}$ |  |  | erge |  |
| 60L-12L ADMIN | $262 \mathrm{M}$ | 40.3 | 62.4 | 43.8 | 2.5 | $256 \mathrm{M}$ | 51.8 | 48.3 | 30.1 | 2.5 |

Table 1: Test results on WMT'14 benchmarks, in terms of TER (T $\downarrow$ ), METEOR (M $\uparrow$ ), and BLEU. $\Delta$ shows difference in BLEU score against baseline 6L-6L Default. Best results are boldfaced. 60L-12L ADMIN outperforms 6L-6L in all metrics with statistical significance ( $p<0.05$ ). Following convention, BLEU is computed by multi-bleu.perl via the standardized tokenization of the publicly-accessible dataset.

| BLEU via multi-bleu.perl | FR | DE |
| :--- | :---: | :---: |
| 60L-12L ADMIN | $\mathbf{4 3 . 8}$ | $\mathbf{3 0 . 1}$ |
| (Wu et al., 2019b) | 43.3 | 29.9 |
| (Wang et al., 2019) | - | 29.6 |
| (Wu et al., 2019a) | 43.2 | 29.7 |
| (Ott et al., 2018) | 43.2 | 29.3 |
| (Vaswani et al., 2017) | 41.8 | 28.4 |
| (So et al., 2019) | 41.3 | 29.8 |
| (Gehring et al., 2017) | 40.5 | 25.2 |
| BLEU via sacreBLEU.py | FR | DE |
| 60L-12L ADMIN | $\mathbf{4 1 . 8}$ | $\mathbf{2 9 . 5}$ |
| (Ott et al., 2018) | 41.4 | 28.6 |
| (So et al., 2019) | n/a | 29.2 |

Table 2: State-of-the-Art on WMT'14 EN-FR/EN-DE

layers and 12 decoder layers. ${ }^{9}$ For each architecture, we train with either default initialization (Glorot and Bengio, 2010) or ADMIN initialization.

The results in terms of BLEU (Papineni et al., 2002), TER (Snover et al., 2006), and METEOR (Lavie and Agarwal, 2007) are reported in Table 1. Similar to previous work (Bapna et al., 2018), we observe that deep 60L-12L Default diverges during training. But the same deep model with ADMIN successfully trains and impressively achieves 2.5 BLEU improvement over the baseline 6L-6L Default in both datasets. The improvements are also seen in terms of other metrics: in EN-FR, 60L-12L ADMIN outperforms the 6L-6L models in TER (40.3 vs 42.2 ) and in METEOR (62.4 vs 60.5 ). All results are statistically significant $(p<0.05)$ with a 1000-sample bootstrap test (Clark et al., 2011).

These results indicate that it is feasible to[^3]

![](https://cdn.mathpix.com/cropped/2024_06_04_b04d3bc9a123d152859bg-3.jpg?height=542&width=696&top_left_y=780&top_left_x=1094)

(a) Train set perplexity: Default vs ADMIN

![](https://cdn.mathpix.com/cropped/2024_06_04_b04d3bc9a123d152859bg-3.jpg?height=520&width=694&top_left_y=1402&top_left_x=1092)

(b) Dev set perplexity: different ADMIN models

Figure 2: Learning curve

train standard (post-LN) Transformers that are very deep. ${ }^{10}$ These models achieve state-ofthe-art results in both datasets. The top results in the literature are compared in Table 2. ${ }^{11}$ We list BLEU scores computed with multi-bleu.perl on the tokenization of the downloaded data (commonly done in previ-[^4]

| Model | BLEU | $\mathrm{a}$ | $\mathrm{b}$ | $\mathrm{c}$ | $\mathrm{d}$ | $\mathrm{e}$ | $\mathrm{f}$ | $\mathrm{g}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| a:6L-6L | 41.5 |  | - | - | - | - | - | - |
| b:12L-12L | 42.6 | + |  | - | - | - | - | - |
| c:24L-12L | 43.3 | + | + |  | $=$ | - | $=$ | $=$ |
| d:48L-12L | 43.6 | + | + | $=$ |  | $=$ | $=$ | + |
| e:60L-12L | 43.8 | + | + | + | $=$ |  | $=$ | + |
| f:36L-36L | 43.7 | + | + | $=$ | $=$ | $=$ |  | + |
| g:12L-60L | 43.1 | + | + | $=$ | - | - | - |  |

Table 3: BLEU comparison of different encoder and decoder layers (using ADMIN initialization, on WMT' 14 EN-FR). In the matrix, each element ( $i, j$ ) indicates if the model in row i significantly outperforms the model in column $\mathrm{j}(+)$, under-performs $\mathrm{j}(-)$, or has no statistically significant difference $(=)$.

ous work), and with sacrebleu.py (version: tok.13a+version.1.2.10). which allows for a safer token-agnostic evaluation (Post, 2018).

Learning Curve: We would like to understand why 60L-12L ADMIN is doing better from the optimization perspective. Figure 2 (a) plots the learning curve comparing ADMIN to Default initialization. We see that Default has difficulty decreasing the training perplexity; its gradients hit $\mathrm{NaN}$, and the resulting model is not better than a random model. In Figure 2 (b), we see that larger models (60L-12L, 36L-36L) are able obtain lower dev perplexities than 6L-6L, implying that the increased capacity does lead to better generalization.

Fine-grained error analysis: We are also interested in understanding how BLEU improvements are reflected in terms of more nuanced measures. For example, do the deeper models particularly improve translation of low frequency words? Do they work better for long sentences? The answer is that the deeper models appear to provide improvements generally across the board (Figure 3). ${ }^{12}$

Ablation Studies: We experimented with different number of encoder and decoder layers, given the constraint of a 16GB GPU. Table 3 shows the pairwise comparison of models. We observe that $60 \mathrm{~L}-12 \mathrm{~L}, 48 \mathrm{~L}-12 \mathrm{~L}$, and $36 \mathrm{~L}-36 \mathrm{~L}$ are statistically tied for best BLEU performance. It appears that deeper encoders are more worthwhile than deeper decoders, when comparing $60 \mathrm{~L}-12 \mathrm{~L}$ to $12 \mathrm{~L}-60 \mathrm{~L}$, despite the latter having more parameters. ${ }^{13}$[^5]

![](https://cdn.mathpix.com/cropped/2024_06_04_b04d3bc9a123d152859bg-4.jpg?height=491&width=782&top_left_y=206&top_left_x=1065)

(a) Word accuracy according to frequency in the training data

![](https://cdn.mathpix.com/cropped/2024_06_04_b04d3bc9a123d152859bg-4.jpg?height=448&width=757&top_left_y=764&top_left_x=1072)

(b) BLEU scores according to sentence length.

Figure 3: Fine-grained Error Analysis: note the deep model performs better across the board, indicating that it helps translation in general.

We also experiment with wider networks, starting with a 6L-6L Transformer-Big (1024-dim word embedding, 4096 feed-forward size, 16 heads) and doubling its layers to $12 \mathrm{~L}-12 \mathrm{~L}$. The BLEU score on EN-FR improved from 43.2 to 43.6 (statistically significant, $p<0.05$ ). A 24L-12L Transformer with BERT-Base like settings (768-dim word embedding, 3072 feed-forward size, 12 heads) obtain 44.0 BLEU score on WMT' 14 EN-FR. This shows that increased depth also helps models that are already relatively wide.

Back-translation We investigate whether deeper models also benefit when trained on the large but potentially noisy data such as back-translation. We follow the back-translation settings of (Edunov et al., 2018) and generated additional 21.8M translation pairs for EN-FR. The hyperparameters are the same as the one without back-translation as introduced in (Edunov et al., 2018), except for an up-sampling rate 1 for EN-FR.

Table 4 compares the ADMIN 60L-12L and ADMIN 36L-12L-768D model ${ }^{14}$ with the default[^6]big transformer architecture (6L-6L) which obtains states-of-the-art results (Edunov et al., 2018). We see that with back-translation, both ADMIN 60L$12 \mathrm{~L}+\mathrm{BT}$ and ADMIN 36L-12L-768D still significantly outperforms its baseline ADMIN 60L-12L. Furthermore, ADMIN 36L-12L-768D achieves new state-of-the-art benchmark results on WMT' 14 English-French (46.4 BLEU and 44.4 sacreBLEU ${ }^{15}$ ).

| BLEU via multi-bleu.perl | FR |
| :--- | :---: |
| 36L-12L-768D ADMIN + BT | $\mathbf{4 6 . 4}$ |
| 60L-12L ADMIN + BT | 46.0 |
| BT (Edunov et al., 2018) | 45.6 |
| 60L-12L ADMIN | 43.8 |
| BLEU via sacreBLEU . pY | FR |
| 36L-12L-768D ADMIN + BT | $\mathbf{4 4 . 4}$ |
| 60L-12L ADMIN + BT | 44.1 |
| 60L-12L ADMIN | 41.8 |
| BT (Edunov et al., 2018) | - |

Table 4: Back-translation results on WMT'14 EN-FR.

## 5 Conclusion

We show that it is feasible to train Transformers at a depth that was previously believed to be difficult. Using ADMIN initialization, we build Transformerbased models of 60 encoder layers and 12 decoder layers. On WMT'14 EN-FR and WMT'14 EN-EN, these deep models outperform the conventional 6layer Transformers by up to 2.5 BLEU, and obtain state-of-the-art results.

We believe that the ability to train very deep models may open up new avenues of research in NMT, including: (a) Training on extremely large but noisy data, e.g. back-translation (Edunov et al., 2018) and adversarial training (Cheng et al., 2019; Liu et al., 2020b), to see if it can be exploited by the larger model capacity. (b) Analyzing the internal representations, to see if deeper networks can indeed extract higher-level features in syntax and semantics (Belinkov and Glass, 2019). (c) Compressing the very deep model via e.g. knowledge distillation (Kim and Rush, 2016), to study the trade-offs between size and translation quality. (d) Analyzing how deep models work (Allen-Zhu and Li, 2020) in theory.[^7]

## Acknowledgments

We thank Hao Cheng, Akiko Eriguchi, Hany Hassan Awadalla and Zeyuan Allen-Zhu for valuable discussions.

## References

Zeyuan Allen-Zhu and Yuanzhi Li. 2020. Backward feature correction: How deep learning performs deep learning. arXiv preprint arXiv:2001.04413.

Ankur Bapna, Mia Xu Chen, Orhan Firat, Yuan Cao, and Yonghui Wu. 2018. Training deeper neural machine translation models with transparent attention. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages $3028-3033$

Rachel Bawden, Nikolay Bogoychev, Ulrich Germann, Roman Grundkiewicz, Faheem Kirefu, Antonio Valerio Miceli Barone, and Alexandra Birch. 2019 The university of edinburgh submissions to the wmt 19 news translation task. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 103-115, Florence, Italy. Association for Computational Linguistics.

Yonatan Belinkov and James Glass. 2019. Analysis methods in neural language processing: A survey. Transactions of the Association for Computational Linguistics, 7:49-72.

Yoshua Bengio. 2009. Learning Deep Architectures for AI, volume Foundations and Trends in Machine Learning. NOW Publishers.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. arXiv preprint arXiv:2005.14165.

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. 2018. The best of both worlds: Combining recent advances in neural machine translation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Melbourne, Australia. Association for Computational Linguistics.

Yong Cheng, Lu Jiang, and Wolfgang Macherey. 2019. Robust neural machine translation with doubly adversarial inputs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4324-4333.

Jonathan H Clark, Chris Dyer, Alon Lavie, and Noah A Smith. 2011. Better hypothesis testing for statistical
machine translation: Controlling for optimizer instability. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 176-181.

Alexis Conneau, Holger Schwenk, Loїc Barrault, and Yann Lecun. 2017. Very deep convolutional networks for text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 1107-1116, Valencia, Spain. Association for Computational Linguistics.

Tobias Domhan. 2018. How much attention do you need? a granular analysis of neural machine translation architectures. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17991808, Melbourne, Australia. Association for Computational Linguistics.

Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. Understanding back-translation at scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 489-500.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. Convolutional sequence to sequence learning. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, page 1243-1252. JMLR.org.

Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249-256.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770778 .

Tianyu He, Xu Tan, Yingce Xia, Di He, Tao Qin, Zhibo Chen, and Tie-Yan Liu. 2018. Layer-wise coordination between encoder and decoder for neural machine translation. In Advances in Neural Information Processing Systems, pages 7944-7954.

Marcin Junczys-Dowmunt. 2019. Microsoft translator at wmt 2019: Towards large-scale document-level neural machine translation. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 225-233, Florence, Italy. Association for Computational Linguistics.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317-1327, Austin, Texas. Association for Computational Linguistics.
A. Lavie and A. Agarwal. 2007. METEOR: An automatic metric for mt evaluation with high levels of correlation with human judgments. In Workshop on Statistical Machine Translation.

Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. 2019. On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265.

Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. 2020a. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249.

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, and Jianfeng Gao. 2020b. Adversarial training for large neural language models. arXiv preprint arXiv:2004.08994.

Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam Chenaghlu, and Jianfeng Gao. 2020. Deep learning based text classification: a comprehensive review. arXiv preprint arXiv:2004.03705.

Graham Neubig, Zi-Yi Dou, Junjie Hu, Paul Michel, Danish Pruthi, and Xinyi Wang. 2019. compare-mt: A tool for holistic comparison of language generation systems. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 35-41, Minneapolis, Minnesota. Association for Computational Linguistics.

Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, and Sergey Edunov. 2019. Facebook fair's wmt19 news translation task submission. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 314-319, Florence, Italy. Association for Computational Linguistics.

Toan Q. Nguyen and Julian Salazar. 2019. Transformers without tears: Improving the normalization of self-attention. In Proc. of the International Workshop on Spoken Language Translation (IWSLT).

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. arXiv preprint arXiv:1904.01038.

Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. 2018. Scaling neural machine translation. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 1-9, Brussels, Belgium. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. BLEU: A method for automatic evaluation of machine translation. In $A C L$.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186191, Brussels, Belgium. Association for Computational Linguistics.

M. Snover, B. Dorr, R. Schwartz, L. Micciulla, and J. Makhoul. 2006. A study of translation edit rate with targeted human annotation. In AMTA.

David So, Quoc Le, and Chen Liang. 2019. The evolved transformer. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5877-5886, Long Beach, California, USA. PMLR.

Rupesh K Srivastava, Klaus Greff, and Jürgen Schmidhuber. 2015. Training very deep networks. In Advances in neural information processing systems, pages 2377-2385.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems, pages 5998-6008.

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. 2019. Learning deep transformer models for machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1810-1822, Florence, Italy. Association for Computational Linguistics.

Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, and Michael Auli. 2019a. Pay less attention with lightweight and dynamic convolutions. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May $6-9,2019$.

Lijun Wu, Yiren Wang, Yingce Xia, Fei Tian, Fei Gao, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2019b. Depth growing for neural machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5558-5563, Florence, Italy. Association for Computational Linguistics.

Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. 2019. Understanding and improving layer normalization. In Advances in Neural Information Processing Systems 32, pages 43814391. Curran Associates, Inc.


[^0]:    ${ }^{1}$ We choose to focus on this layer size since it results in the maximum model size that can fit within a single GPU system. The purpose of this study is to show that it is feasible for most researchers to experiment with very deep models; access to massive GPU budgets is not a requirement.

    ${ }^{2}$ Note there are architectural variants that enable deeper models (Wang et al., 2019; Nguyen and Salazar, 2019), discussed in Sec 2. We focus on the standard architecture here.

[^1]:    ${ }^{3}$ The 96-layer GPT-3 (Brown et al., 2020) uses pre-LN.

[^2]:    ${ }^{4}$ Note that paper presents results of 18-layer Transformers on the WMT' 14 En-De, which we also use here. Our contribution is a more comprehensive evaluation.

    ${ }^{5}$ We estimate the variance with one batch of $8 \mathrm{k}$ tokens.

    ${ }^{6}$ https://github.com/pytorch/fairseq/ blob/master/examples/translation/ prepare-WMT'14en2fr.sh

    ${ }^{7}$ https://github.com/tensorflow/ tensor2tensor/blob/master/tensor2tensor/ data_generators/translate_ende.py

    ${ }^{8}$ For FR, \#warmup steps is 8000 , max \#epochs is 50 , and peak learning rate is 0.0007 . For DE, \#warmup steps is 4000 , max \#epochs is 50 , and learning rate is 0.001 . Max \#tokens in each batch is set to 3584 following (Ott et al., 2019).

[^3]:    ${ }^{9}$ We use "(N)L-(M)L" to denote that a model has $\mathrm{N}$ encoder layers and M decoder layers. N \& M are chosen based on GPU (16G) memory constraint. For reproducibility and simplicity, we focused on models that fit easily on a single GPU system. Taking FR as an example, it takes 2.5 days to train 60L-12L using one DGX-2 (16 V100's), 2 days to train a 6L-6L using 4 V100's.

[^4]:    ${ }^{10}$ Note: the pre-LN version does train successively on $60 \mathrm{~L}$ $12 \mathrm{~L}$ and achieves 29.3 BLEU in DE \& 43.2 in FR. It is better than 6L-6L but worse than 60L-12L ADMIN.

    ${ }^{11}$ The table does not include systems that use extra data.

[^5]:    ${ }^{12}$ Computed by compare-mt (Neubig et al., 2019).

    ${ }^{13}$ Recall from Figure 1 that each encoder layer has 2 subnetwork components and each decoder layer has 3 components.

[^6]:    ${ }^{14}$ It is BERT-base setting with 768-dim word embedding,

[^7]:    3072 feed-froward size and 12 heads

    ${ }^{15}$ BLEU+case.mixed+lang.en-

    fr+numrefs.1+smooth.exp+test.wmt14+tok.13a+version.1.2.10

</end of paper 0>


<paper 1>
# Understanding the Difficulty of Training Transformers 

\author{
Liyuan Liu ${ }^{\dagger \ddagger}$ Xiaodong Liu ${ }^{\ddagger}$ Jianfeng Gao ${ }^{\ddagger}$ Weizhu Chen ${ }^{\S}$ Jiawei Han ${ }^{\dagger}$ <br> \{ll2, hanj\}@illinois.edu, \{xiaodl, jfgao, wzchen\}@microsoft.com <br> ${ }^{\dagger}$ University of Illinois at Urbana-Champaign <br> ${ }^{\ddagger}$ Microsoft Research

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-01.jpg?height=52&width=523&top_left_y=568&top_left_x=778)


#### Abstract

Transformers have proved effective in many NLP tasks. However, their training requires non-trivial efforts regarding designing cuttingedge optimizers and learning rate schedulers carefully (e.g., conventional SGD fails to train Transformers effectively). Our objective here is to understand what complicates Transformer training from both empirical and theoretical perspectives. Our analysis reveals that unbalanced gradients are not the root cause of the instability of training. Instead, we identify an amplification effect that influences training substantially-for each layer in a multi-layer Transformer model, heavy dependency on its residual branch makes training unstable, since it amplifies small parameter perturbations (e.g., parameter updates) and results in significant disturbances in the model output. Yet we observe that a light dependency limits the model potential and leads to inferior trained models. Inspired by our analysis, we propose Admin (Adaptive model initialization) to stabilize the early stage's training and unleash its full potential in the late stage. Extensive experiments show that Admin is more stable, converges faster, and leads to better performance ${ }^{1}$.


## 1 Introduction

Transformers (Vaswani et al., 2017) have led to a series of breakthroughs in various deep learning tasks (Devlin et al., 2019; Velickovic et al., 2018). They do not contain recurrent connections and can parallelize all computations in the same layer, thus improving effectiveness, efficiency, and scalability. Training Transformers, however, requires extra efforts. For example, although stochastic gradient descent (SGD) is the standard algorithm for conventional RNNs and CNNs, it converges to bad/suspicious local optima for Trans-[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-01.jpg?height=483&width=762&top_left_y=775&top_left_x=1064)

Figure 1: Lacking enough robustness and stability, the 18-Layer Post-LN Transformer training (i.e.the original architecture) diverges and is omitted in the left graph. Admin not only stabilizes model training but unleashes the model potential for better performance.

formers (Zhang et al., 2019b). Moreover, comparing to other neural architectures, removing the warmup stage in Transformer training results in more severe consequences such as model divergence (Popel and Bojar, 2018; Liu et al., 2020a). Here, we conduct comprehensive analyses in empirical and theoretical manners to answer the question: what complicates Transformer training.

Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant ${ }^{2}$ (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). We recognize that gradient vanishing issue is not the direct reason causing such difference, since fixing this issue alone cannot stabilize Post$\mathrm{LN}$ training. It implies that, besides unbalanced gradients, there exist other factors influencing model training greatly.

With further analysis, we recognize that for each Transformer residual block, the dependency on its[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-02.jpg?height=811&width=1602&top_left_y=214&top_left_x=227)

Figure 2: The Architecture and notations of Pre-LN Transformers (Left) and Post-LN Transformers (Right).

residual branch ${ }^{3}$ plays an essential role in training stability. First, we find that a Post-LN layer has a heavier dependency on its residual branch than a Pre-LN layer. As in Figure 7, at initialization, a Pre-LN layer has roughly the same dependency on its residual branch and any previous layer, whereas a Post-LN layer has a stronger dependency on its residual branch (more discussions are elaborated in Section 4.1). We find that strong dependencies of Post-LN amplify fluctuations brought by parameter changes and destabilize the training (as in Theorem 2 and Figure 4). Besides, the loose reliance on residual branches in Pre-LN generally limits the algorithm's potential and often produces inferior models.

In light of our analysis, we propose Admin, an adaptive initialization method which retains the merits of Pre-LN stability without hurting the performance. It restricts the layer dependency on its residual branches in the early stage and unleashes the model potential in the late stage. We conduct experiments on IWSLT'14 De-En, WMT'14 EnDe, and WMT'14 En-Fr; Admin is more stable, converges faster, and achieves better performance. For example, without introducing any additional hyper-parameters, Admin successfully stabilizes 72-layer Transformer training on WMT' 14 En-Fr and achieves a 43.80 BLEU score.[^2]

## 2 Preliminaries

Transformer Architectures and Notations. The Transformer architecture contains two types of sublayers, i.e., Attention sub-layers and Feedforward (FFN) sub-layers. They are composed of mainly three basic modules (Vaswani et al., 2017), i.e., Layer Norm $\left(f_{\mathrm{LN}}\right)$, Multi-head Attention ( $f_{\mathrm{ATT}}$ ), and Feedforward Network $\left(f_{\mathrm{FFN}}\right)$.

As illustrated in Figure 2, the Pre-LN Transformer and the Post-LN Transformer organize these modules differently. For example, a PreLN encoder organizes the Self-Attention sublayer as $\mathbf{x}_{2 i-1}^{(p e)}=\mathbf{x}_{2 i-2}^{(p e)}+f_{\mathrm{S}-\mathrm{ATT}}\left(f_{\mathrm{LN}}\left(\mathbf{x}_{2 i-2}^{(p e)}\right)\right)$ and a Post-LN encoder as $\mathbf{x}_{2 i-1}^{(o e)}=f_{\mathrm{LN}}\left(\mathbf{x}_{2 i-2}^{(o e)}+\right.$ $f_{\mathrm{S} \text {-ATT }}\left(\mathbf{x}_{2 i-2}^{(o e)}\right)$, where $\mathbf{x}_{2 i-2}^{(\cdot)}$ is the input of the $i$ th Transformer layer and $\mathbf{x}_{2 i-1}^{(\cdot)}$ is the output of the $i$-th Self-Attention sub-layer. Here, we refer $f_{\mathrm{S}-\mathrm{ATT}}\left(f_{\mathrm{LN}}\left(\mathbf{x}_{2 i-2}^{(p e)}\right)\right)$ and $f_{\mathrm{S} \text {-ATT }}\left(\mathbf{x}_{2 i-2}^{(o e)}\right)$ as the residual branches and their outputs as the residual outputs, in contrast to layer/sub-layer outputs, which integrates residual outputs and shortcut outputs.

Notation elaborations are shown in Figure 2. In particular, we use superscripts to indicate network architectures (i.e., the Pre-LN Encoder), use subscripts to indicate layer indexes (top layers have larger indexes), all inputs and outputs are formulated as Sequence-Len $\times$ Hidden-Dim.

Layer Norm. Layer norm (Ba et al., 2016) plays a vital role in Transformer architecture. It is defined

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-03.jpg?height=620&width=1585&top_left_y=210&top_left_x=241)

Figure 3: Relative gradient norm histogram (on a log scale) of 18-layer Transformers on the WMT' 14 En-De dataset, i.e., the gradient norm of sub-layer outputs, scaled by the largest gradient norm in the same network.

as $f_{\mathrm{LN}}(\mathbf{x})=\gamma \frac{\mathbf{x}-\mu}{\sigma}+\nu$, where $\mu$ and $\sigma$ are the mean and standard deviation of $\mathbf{x}$.

Feedforward Network. Transformers use twolayer perceptrons as feedforward networks, i.e., $f_{\mathrm{FFN}}(\mathbf{x})=\phi\left(\mathbf{x} W^{(1)}\right) W^{(2)}$, where $\phi(\cdot)$ is the nonlinear function ${ }^{4}$, and $W^{(\cdot)}$ are parameters.

Multi-head Attention. Multi-head Attentions allows the network to have multiple focuses in a single layer and plays a crucial role in many tasks (Chen et al., 2018). It is defined as (with $H$ heads): $f_{\text {ATT }}(\mathbf{q}, \mathbf{k}, \mathbf{v})=$ $\sum_{h=1}^{H} f_{s}\left(\mathbf{q} W_{h}^{(Q)} W_{h}^{(K)} \mathbf{k}^{T}\right) \mathbf{v} W_{h}^{\left(V_{1}\right)} W_{h}^{\left(V_{2}\right)}$, where $f_{s}$ is the row-wise softmax function and $W_{h}^{(\cdot)}$ are parameters. $W_{h}^{(Q)}$ and $W_{h}^{\left(V_{1}\right)}$ are $D \times \frac{D}{H}$ matrices, $W_{h}^{(K)}$ and $W_{h}^{\left(V_{2}\right)}$ are $\frac{D}{H} \times D$ matrices, where $D$ is the hidden state dimension. Parameters without subscript refer the concatenation of all $\mathrm{H}$ head parameters, e.g., $W^{(Q)}=\left[W_{1}^{(Q)}, \cdots, W_{H}^{(Q)}\right]$. In Transformer, this module is used in two different settings: Encoder-Attention $\left(f_{\mathrm{E}-\mathrm{ATT}}(\mathbf{x})=\right.$ $f_{\text {ATT }}\left(\mathbf{x}, \mathbf{x}^{(\cdot e)}, \mathbf{x}^{(\cdot e)}\right)$ and $\mathbf{x}^{(\cdot e)}$ is the encoder output), and Self-Attention $\left(f_{\mathrm{S}-\mathrm{ATT}}(\mathbf{x})=f_{\mathrm{ATT}}(\mathbf{x}, \mathbf{x}, \mathbf{x})\right)$.

## 3 Unbalanced Gradients

In this study, we strive to answer the question: what complicates Transformer training. Our analysis starts from the observation: Pre-LN training is more robust than Post-LN, while Post-LN is more likely to reach a better performance than Pre-LN. In a parameter grid search (as in Figure 10), Pre-LN[^3]

converges in all 15 settings, and Post-LN diverges in 7 out of 15 settings; when Post-LN converges, it outperforms Pre-LN in 7 out of 8 settings. We seek to reveal the underlying factor that destabilizes Post-LN training and restricts the performance of Pre-LN.

In this section, we focus on the unbalanced gradients (e.g., gradient vanishing). We find that, although Post-LN suffers from gradient vanishing and Pre-LN does not, gradient vanishing is not the direct reason causing the instability of Post-LN. Specifically, we first theoretically and empirically establish that only Post-LN decoders suffer from gradient vanishing and Post-LN encoders do not. We then observe that fixing the gradient vanishing issue alone cannot stabilize training.

### 3.1 Gradients at Initialization

As gradient vanishing can hamper convergence from the beginning, it has been regarded as the major issue causing unstable training. Also, recent studies show that this issue exists in the PostLN Transformer, even after using residual connections (Xiong et al., 2019). Below, we establish that only Post-LN decoders suffer from the gradient vanishing, and neither Post-LN encoders, Pre-LN encoders, nor Pre-LN decoders.

We use $\Delta \mathbf{x}$ to denote gradients, i.e., $\Delta \mathbf{x}=\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ where $\mathcal{L}$ is the training objective. Following previous studies (Glorot and Bengio, 2010), we analyze the gradient distribution at the very beginning of training and find only Encoder-Attention sub-layers in Post-LN suffers from gradient vanishing.

First, we conduct analysis from a theoretical

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-04.jpg?height=517&width=1153&top_left_y=210&top_left_x=243)

Figure 4: Encoder output changes for parameter changes, i.e., $\mid \mathcal{F}\left(\mathbf{x}_{0}, W\right)-$ $\left.\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right|_{2} ^{2}$ where $W^{*}-W$ is random perturbations (left) or gradient updates (right). Intuitively, very large $\left|\mathcal{F}-\mathcal{F}^{*}\right|$ indicates the training to be ill-conditioned.

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-04.jpg?height=417&width=391&top_left_y=223&top_left_x=1415)

Figure 5: Histogram of relative norm of gradient and $\left|W_{i+1}-W_{i}\right|$ where $W_{i}$ is the checkpoint saved after training for $i$ epochs.

| Encoder | Decoder | Gradient | Training |
| :---: | :---: | :---: | :---: |
| Post-LN | Post-LN | Vanishing | Diverged |
| Post-LN | Pre-LN | Diverged |  |
| Pre-LN | Pre-LN |  | Converged |

Table 1: Changing decoders from Post-LN to Pre-LN fixes gradient vanishing, but does not stabilize model training successfully. Encoder/Decoder have 18 layers.

perspective. Similar to Xiong et al. (2019), we establish that Pre-LN networks do not suffer from gradient vanishing (as elaborated in Appendix A.1). Unlike Xiong et al. (2019), we recognize that not all Post-LN networks suffer from gradient vanishing. As in Theorem 1, we establish that Post-LN Encoder networks do not suffer from gradient vanishing. Detailed derivations are elaborated in Appendix A.2.

Theorem 1. - For Post-LN Encoders, if $\gamma$ and $\boldsymbol{\nu}$ in the Layer Norm are initialized as 1 and 0 respectively; all other parameters are initialized by symmetric distributions with zero mean; $\mathbf{x}_{i}^{(o e)}$ and $\Delta \mathbf{x}_{i}^{(o e)}$ are subject to symmetric distributions with zero mean; the variance of $\mathbf{x}_{i}^{(o e)}$ is 1 (i.e., normalized by Layer Norm); $\Delta \mathbf{x}_{i}^{(o e)}$ and the derivatives of modules in $i$-th sub-layer are independent, we have $\operatorname{Var}\left[\Delta \mathbf{x}_{i-1}\right] \geq \operatorname{Var}\left[\Delta \mathbf{x}_{i}\right]$.

To make sure that the assumptions of Theorem 2 match the real-world situation, we further conduct empirical verification. At initialization, we calculate $\left\|\Delta \mathbf{x}_{i}^{(\cdot)}\right\|_{2}$ for 18-layer Transformers ${ }^{5}$[^4]

and visualize $\frac{\left\|\Delta \mathbf{x}_{i}^{(\cdot)}\right\|_{2}}{\max _{j}\left\|\Delta \mathbf{x}_{j}^{(\cdot)}\right\|_{2}}$ in Figure 3. It verifies that only Post-LN decoders suffer from the gradient vanishing. Besides, we can observe that the dropping of gradient norms mostly happens in the backpropagation from encoder-attention outputs (encoder-attention bars) to its inputs (self-attention bars, since the output of self-attention is the input of encoder-attention). This pattern is further explained in Appendix A.3.

### 3.2 Impact of the Gradient Vanishing

Now, we explore whether gradient vanishing is the direct cause of training instability.

First, we design a controlled experiment to show the relationship between gradient vanishing and training stability. We construct a hybrid Transformer by combining a Post-LN encoder and a Pre-LN decoder. As in Section 3.1, only Post-LN decoders suffer from gradient vanishing, but not Post-LN encoders. Therefore, this hybrid Transformer does not suffer from gradient vanishing. As shown in Table 1, fixing gradient vanishing alone (i.e., changing Post-LN decoders to Pre-LN decoders) fails to stabilize model training. This observation provides evidence supporting that the gradient vanishing issue is not the direct cause of unstable Post-LN training.

Moreover, we observe that gradients of all attention modules are unbalanced, while adaptive optimizers mostly address this issue. As in Figure 5 , adaptive optimizers successfully assign different learning rates to different parameters and lead to consistent update magnitudes even with unbalanced gradients. It explains why the standard SGD fails in training Transformers (i.e., lacking the

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-05.jpg?height=317&width=794&top_left_y=190&top_left_x=237)

Figure 6: The major difference between Pre-LN and Post-LN is the position of layer norms.

ability to handle unbalanced gradients) and necessitates using adaptive optimizers. More discussions are included in Appendix A.4.

## 4 Instability from Amplification Effect

We find that unbalanced gradients are not the root cause of the instability of Post-LN, which implies the existence of other factors influencing model training. Now, we go beyond gradient vanishing and introduce the amplification effect. Specifically, we first examine the difference between Pre-LN and Post-LN, including their early-stage and latestage training. Then, we show that Post-LN's training instability is attributed to layer dependency's amplification effect, which intensifies gradient updates and destabilizes training.

### 4.1 Impact of Layer Norms Positions

As described in Section 2, both Pre-LN and Post$\mathrm{LN}$ employ layer norm to regularize inputs and outputs. Different residual outputs are aggregated and normalized in residual networks before serving as inputs of other layers (i.e., residual outputs will be scaled to ensure the integrated input to have a consistent variance). To some extend, layer norm treats the variance of residual outputs as weights to average them. For example, for Post-LN Self-Attention, we have $\mathbf{x}_{2 i-1}^{(o \cdot)}=\frac{\mathbf{x}_{2 i-2}^{(o \cdot)}+\mathbf{a}_{2 i-1}^{(o .)}}{\sqrt{\operatorname{Var}\left[\mathbf{x}_{2 i-2}^{(o .)}\right]+\operatorname{Var}\left[\mathbf{a}_{2 i-1}^{(o)}\right]}}$ at initialization. Larger $\operatorname{Var}\left[\mathbf{a}_{2 i-2}^{(o \cdot)}\right]$ not only increases the proportion of $\mathbf{a}_{2 i-2}^{(o \cdot)}$ in $\mathbf{x}_{2 i-2}^{(o \cdot)}$ but decreases the proportion of other residual outputs. Intuitively, this is similar to the weight mechanism of the weighted average.

The position of layer norms is the major difference between Pre-LN and Post-LN and makes them aggregate residual outputs differently (i.e., using different weights). As in Figure 6, all residual outputs in Pre-LN are only normalized once before feeding into other layers (thus only treating residual output variances as weights); in Post-LN, most

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-05.jpg?height=614&width=760&top_left_y=224&top_left_x=1068)

Figure 7: $\beta_{i, j}$ in 6-Layer Post-LN and Pre-LN on the WMT-14 En-De dataset (contains 12 sub-layers).

residual outputs are normalized more than once, and different residual outputs are normalized for different times. For example, if all layers are initialized in the same way, output variances of different Pre-LN residual branches would be similar, and the aggregation would be similar to the simple average. Similarly, for Post-LN, nearby residual outputs are normalized by fewer times than others, thus having relatively larger weights. We proceed to calculate and analyze these weights to understand the impact of layer norm positions.

First, we use $\widehat{\mathbf{a}}_{i}$ to refer $\frac{\mathbf{a}_{i}}{\sqrt{\operatorname{Var} \mathbf{a}_{i}}}$ (i.e., normalized outputs of $i$-th residual branch) and $\widehat{\mathbf{x}}_{i}$ to refer $\frac{\mathbf{x}_{i}}{\sqrt{\operatorname{Var} \mathbf{x}_{i}}}($ i.e., normalized outputs of $i$-th layer or normalized inputs of $(i+1)$-th residual branch). Then, we describe their relationships as $\widehat{\mathbf{x}}_{i}=$ $\sum_{j \leq i} \beta_{i, j} \widehat{\mathbf{a}}_{j}$, where $\beta_{i, j}$ integrates scaling operations of all layer norms (including $\sqrt{\operatorname{Var}\left[\mathbf{a}_{i}\right]}$ ). For example, Pre-LN sets $\beta_{i, j}=\frac{\sqrt{\operatorname{Var}\left[\mathbf{a}_{j}\right]}}{\sqrt{\operatorname{Var}\left[\sum_{k \leq i} \mathbf{a}_{k}\right]}}$. Intuitively, $\beta_{i, j}$ describes the proportion of $j$-th residual branch outputs in $i$-th layer outputs, thus reflects the dependency among layers.

We visualize $\beta_{i, j}$ in Figure 7. For a Post-LN layer, its outputs rely more on its residual branch from the initialization to the end. At initialization, Pre-LN layer outputs have roughly the same reliance on all previous residual branches. As the training advances, each layer starts to rely more on its own residual outputs. However, comparing to Post-LN, Pre-LN layer outputs in the final model still has less reliance on their residual branches.

Intuitively, it is harder for Pre-LN layers to depend too much on their own residual branches. In

Pre-LN, layer outputs (i.e., $\mathbf{x}_{i}^{(p .)}$ ) are not normalized, and their variances are likely to be larger for higher layers ${ }^{6}$. Since $\beta_{i, i}=\frac{\sqrt{\operatorname{Var}\left[\mathbf{a}_{i}\right]}}{\sqrt{\operatorname{Var}\left[\mathbf{x}_{i-1}^{(p .1}+\mathbf{a}_{i}\right]}}, \beta_{i, i}$ is likely to be smaller for higher layers, which restricts $i$-th layer outputs from depending too much on its residual branch and inhibits the network from reaching its full potential. In other words, Pre-LN restricts the network from being too deep (i.e., if it is hard to distinguish $\mathbf{x}_{i}^{(p \cdot)}$ and $\mathbf{x}_{i+1}^{(p \cdot)}$, appending one layer would be similar to doubling the width of the last layer), while Post-LN gives the network the choice of being wider or deeper.

### 4.2 Amplification Effect at Initialization

Although depending more on residual branches allows the model to have a larger potential, it amplifies the fluctuation brought by parameter changes. For a network $\widehat{\mathbf{x}}=\mathcal{F}\left(\mathbf{x}_{0}, W\right)$ where $\mathbf{x}_{0}$ is the model input and $W$ is the parameter, the output change caused by parameter perturbations is $\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right]$, where $W^{*}=W+\delta$. Its relationship with $N$ is described in Theorem 2, and the derivation is elaborated in Appendix B.

Theorem 2. - Consider a $N$-layer Transformer $\widehat{\mathbf{x}}=\mathcal{F}\left(\widehat{\mathbf{x}}_{0}, W\right)$ at initialization, where $\widehat{\mathbf{x}}_{0}$ is the input and $W$ is the parameter. If the layer dependency stays the same after a parameter change (i.e., $\beta_{i, j}$ has the same value after changing $W$ to $W^{*}$, where $W$ is randomly initialized and $\delta=W^{*}-W$ is independent to $W$ ), the output change (i.e., $\left.\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right]\right)$ can be estimated as $\sum_{i=1}^{N} \beta_{i, i}^{2} C$ where $C$ is a constant.

If $\operatorname{Var}\left[\mathbf{a}_{i}\right]$ is the same for all layers, Pre-LN sets $\beta_{i, i}^{2}$ as $1 / i$, and Post-LN sets $\beta_{i, i}^{2}$ as a constant. Thus, we have Corollary 1 and 2 as below.

Corollary 1. - For a $N$-layer Pre-LN $\mathcal{F}$, we have $\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right]=O(\log N)$.

Corollary 2. - For a $N$-layer Post-LN $\mathcal{F}$, we have $\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right]=O(N)$.

They show that, since Post-LN relies more on residual branches than Pre-LN (i.e., has a larger $\left.\beta_{i, i}^{2}\right)$, the perturbation is amplified to a larger magnitude. To empirically verify these relationships, we calculate $\left|\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right|_{2}^{2}$ for PreLN and Post-LN and visualize the results in Fig-[^5]

ure 4. In Corollary 2, $N$ is linearly associated with $\left|\mathcal{F}-\mathcal{F}^{*}\right|_{2}^{2}$ for Post-LN; and in Corollary 1, $\log N$ is linearly associated with $\left|\mathcal{F}-\mathcal{F}^{*}\right|_{2}^{2}$ for Pre-LN. These relationships match the observation in our experiments (as in Figure 4). For further verification, we measure their correlation magnitudes by $R^{2}$ and find $R^{2}=0.99$ in both cases.

Moreover, we replace the random noise $\delta$ with optimization updates (i.e., setting $W^{*}=W+$ $\operatorname{Adam}(\Delta W)$, where opt $(\cdot)$ is update calculated by the Adam optimizer) and visualize output shifts. This replacement makes the correlation between $\left|\mathcal{F}-\mathcal{F}^{*}\right|_{2}^{2}$ and $N$ (for Post-LN) or $\log N$ (for Pre$\mathrm{LN}$ ) to be weaker (i.e., $R^{2}=0.75$ ). Still, as in Figure 4, the output shift $\left|\mathcal{F}-\mathcal{F}^{*}\right|_{2}^{2}$ for Post-LN is larger than Pre-LN by multiple magnitudes.

Intuitively, large output shifts would destabilize the training (Li et al., 2018). Also, as elaborated in Appendix B, the constant $C$ in Theorem 2 is related to network derivatives and would be smaller as training advances, which explains why warmup is also helpful for the standard SGD. Therefore, we conjecture it is the large output shift of Post-LN results in unstable training. We proceed to stabilize Post-LN by controlling the dependency on residual branches in the early stage of training.

### 4.3 Admin - Adaptive Model Initialization

In light of our analysis, we add additional parameters (i.e., $\boldsymbol{\omega}$ ) to control residual dependencies of Post-LN and stabilize training by adaptively initializing $\boldsymbol{\omega}$ to ensure an $O(\log N)$ output change.

Due to different training configurations and model specificities (e.g., different models may use different activation functions and dropout ratios), it is hard to derive a universal initialization method. Instead, we decompose model initialization into two phrases: Profiling and Initialization. Specifically, Admin adds new parameters $\boldsymbol{\omega}$ and constructs its i-th sub-layer as $\mathbf{x}_{i}=f_{\mathrm{LN}}\left(\mathbf{b}_{i}\right)$, where $\mathbf{b}_{i}=\mathbf{x}_{i-1} \cdot \omega_{i}+f_{i}\left(\mathbf{x}_{i-1}\right), \boldsymbol{\omega}_{i}$ is a $D$-dimension vector and $\cdot$ is element-wise product. Then the Profiling phrase and Initialization phrase are:

Profiling. After initializing the network with a standard method (initializing $\omega_{i}$ as $\mathbf{1}$ ), conduct forward propagation without parameter updating and record the output variance of residual branches (i.e., calculate $\operatorname{Var}\left[f_{i}\left(\mathbf{x}_{i-1}\right)\right]$ ). Since all elements in the same parameter/output matrix are independent to each other and are subject to the same distribution, it is sufficient to use a small number of instances in

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-07.jpg?height=585&width=760&top_left_y=210&top_left_x=248)

Figure 8: $\beta_{i, j}$ of 18-Layer Admin (Post-LN) and PreLN on the WMT-14 En-De dataset.

this phrase. In our experiments, the first batch (no more than 8192 tokens) is used.

Initialization. Set $\boldsymbol{\omega}_{i}=\sqrt{\sum_{j<i} \operatorname{Var}\left[f_{j}\left(\mathbf{x}_{j-1}\right)\right]}$ and initialize all other parameters with the same method used in the Profiling phrase.

In the early stage, Admin sets $\beta_{i, i}^{2}$ to approximately $\frac{1}{i}$ and ensures an $O(\log N)$ output change, thus stabilizing training. Model training would become more stable in the late stage (the constant $C$ in Theorem 2 is related to parameter gradients), and each layer has the flexibility to adjust $\boldsymbol{\omega}$ and depends more on its residual branch to calculate the layer outputs. After training finishes, Admin can be reparameterized as the conventional Post-LN structure (i.e., removing $\boldsymbol{\omega}$ ). More implementation details are elaborated in Appendix C.

To verify our intuition, we calculate the layer dependency of 18-Layer models and visualize the result in Figure 8. Figures 7 and 8 show that Admin avoids over-large dependencies at initialization and unleashes the potential to make the layer outputs depend more on their residual outputs in the final model. Moreover, we visualize the output change of Admin in Figure 4. Benefiting from the adaptive initialization, the output change of Admin gets roughly the same increase speed as Pre-LN, even constructed in the Post-LN manner. Also, although Admin is formulated in a Post-LN manner and suffers from gradient vanishing, 18-layer Admin successfully converges and outperforms 18-layer Pre-LN (as in Table 2). This evidence supports our intuition that the large dependency on residual branches amplifies the output fluctuation and destabilizes training.

![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-07.jpg?height=446&width=760&top_left_y=211&top_left_x=1065)

Figure 9: Development PPL on the WMT'14 En-De dataset and the IWLST' 14 De-En dataset.

## 5 Experiments

We conduct experiments on IWSLT'14 De-En, WMT'14 En-De, and WMT'14 En-Fr. More details are elaborated in Appendix D.

### 5.1 Performance Comparison

We use BLEU as the evaluation matric and summarize the model performance in Table 2. On the WMT'14 dataset, we use Transformer-base models with 6,12 , or 18 layers. Admin achieves a better performance than Post-LN and Pre-LN in all three settings. Specifically, 12-Layer and 18-Layer Post-LN diverges without the adaptive initialization. Pre-LN converges in all settings, but it results in sub-optimal performance. Admin not only stabilizes the training of deeper models but benefits more from the increased model capacity then Pre$\mathrm{LN}$, which verifies our intuition that the Pre-LN structure limits the model potential. As in Figure 1 and Figure 9, although the 6-layer Pre-LN converges faster than Post-LN, its final performance is worse than Post-LN. In contrast, Admin not only achieves the same convergence speed with Pre-LN in the early stage but reaches a good performance in the late stage.

We use 6-layer Transformer-small (its hidden dimension is smaller than the base model) on the IWSLT' 14 dataset, and all methods perform similarly. Still, as in Figure 10, Admin outperforms the other two by a small margin. Together with WMT' 14 results, it implies the training stability is related to layer number. For shallow networks, the stability difference between Post-LN and Pre-LN is not significant (as in Figure 4), and all methods reach reasonable performance. It is worth mentioning that attention and activation dropouts have an enormous impact on IWSLT' 14, which is smaller than WMT' 14 datasets.

Table 2: BLEU on IWSLT'14 De-En and WMT'14 En-Fr/De (AL-BL refers A-layer encoder \& B-layer decoder).

| Dataset | IWSLT'14 De-En | WMT'14 En-Fr |  | WMT'14 En-De |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Enc \#-Dec \# | 6L-6L (small) | 6L-6L | 60L-12L | 6L-6L | 12L-12L | 18L-18L |
| Post-LN | $35.64 \pm 0.23$ | 41.29 | failed | 27.80 | failed | failed |
| Pre-LN | $35.50 \pm 0.04$ | 40.74 | 43.10 | 27.27 | 28.26 | 28.38 |
| Admin | $\mathbf{3 5 . 6 7} \pm \mathbf{0 . 1 5}$ | $\mathbf{4 1 . 4 7}$ | $\mathbf{4 3 . 8 0}$ | $\mathbf{2 7 . 9 0}$ | $\mathbf{2 8 . 5 8}$ | $\mathbf{2 9 . 0 3}$ |

To further explore the potential of Admin, we train Transformers with a larger size. Specifically, we expand the Transformer-base configuration to have a 60-layer encoder and a 12-layer decoder. As in Table 2, our method achieves a BLEU score of 43.8 on the WMT' 14 En-Fr dataset, the new state-of-the-art without using additional annotations (e.g., back-translation). More discussions are conducted in Appendix F to compare this model with the current state of the art. Furthermore, in-depth analyses are summarized in Liu et al. (2020b), including systematic evaluations on the model performance (with TER, METEOR, and BLEU), comprehensive discussions on model dimensions (i.e., depth, head number, and hidden dimension), and fine-grained error analysis. It is worth mentioning that the $60 \mathrm{~L}-12 \mathrm{~L}$ Admin model achieves a 30.1 BLEU score on WMT'14 En-De (Liu et al., 2020b).

### 5.2 Connection to Warmup

Our previous work (Liu et al., 2020a) establishes that the need for warmup comes from the unstable adaptive learning rates in the early stage. Still, removing the warmup phrase results in more severe consequences for Transformers than other architectures. Also, warmup has been found to be useful for the vanilla SGD (Xiong et al., 2019).

Theorem 1 establishes that $\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}_{0}, W\right)-\right.$ $\left.\mathcal{F}\left(\mathbf{x}_{0}, W^{*}\right)\right] \approx \sum_{i=1}^{N} \beta_{i, i}^{2} C$ where $C=$ $\operatorname{Var}\left[\mathcal{G}_{i}\left(\widehat{\mathbf{x}}_{i-1}^{*}, W_{i}\right)-\mathcal{G}_{i}\left(\widehat{\mathbf{x}}_{i-1}^{*}, W_{i}^{*}\right)\right]$. In the early stage of training, the network has larger parameter gradients and thus larger $C$. Therefore, using a small learning rate at initialization helps to alleviate the massive output shift of Post-LN. We further conduct experiments to explore whether more prolonged warmups can make up the stability difference between Post-LN and Pre-LN. We observe that 18-layer Post-LN training still fails after extending the warmup phrase from 8 thousand updates to 16,24 , and 32 thousand. It shows that learning rate warmup alone cannot neutralize the
![](https://cdn.mathpix.com/cropped/2024_06_04_deb98724e8392d239770g-08.jpg?height=386&width=780&top_left_y=658&top_left_x=1049)

Figure 10: BLEU score of Post-LN, Pre-LN and Admin on the IWSLT'14 De-En dataset ( $\mathrm{x}$-axis is the $\beta_{2}$ for adaptive optimizers and $y$-axis is the learning rate). Pre-LN converges in all settings while Post-LN diverges in 7 out of 15 settings. When Post-LN converges, it outperforms Pre-LN in 7 out of 8 settings. Admin stabilizes Post-LN training and outperforms Pre-LN (its best performance is comparable with Post-LN).

instability of Post-LN. Intuitively, massive output shifts not only require a small learning rate but also unsmoothes the loss surface (Li et al., 2018) and make the training ill-conditioned.

Admin regularizes the model behavior at initialization and stabilizes the training. To explore whether Admin is able to stabilize the training alone, we remove the warmup phase and conduct a grid search on optimizer hyper-parameters. The results are visualized in Figure 10. It shows that as Post-LN is more sensitive to the choice of hyperparameters, Admin successfully stabilizes the training without hurting its potential.

### 5.3 Comparing to Other Initializations

We compare our methods with three initialization methods, i.e., ReZero (Bachlechner et al., 2020), FixUp (Zhang et al., 2019a), and LookLinear (Balduzzi et al., 2017a). Specifically, we first conduct experiments with 18-layer Transformers on the WMT'14 De-En dataset. In our experiments, we observe that all of ReZero (which does not contain layer normalization), FixUp (which also does not contain layer normalization), and LookLinear (which is incorporated with Post-LN) leads to di-
vergent training. With further analysis, we find that the half-precision training and dropout could destabilize FixUp and ReZero, due to the lack of layer normalization. Simultaneously, we find that even for shadow networks, having an over small reliance on residual branches hurts the model performance, which also supports our intuition. For example, as elaborated in Appendix E, applying ReZero to Transformer-small leads to a 1-2 BLEU score drop on the IWSLT' 14 De-En dataset.

## 6 Related Work

Transformer. Transformer (Vaswani et al., 2017) has led to a series of breakthroughs in various domains (Devlin et al., 2019; Velickovic et al., 2018; Huang et al., 2019; Parmar et al., 2018; Ramachandran et al., 2019). Liu et al. (2020a) show that compared to other architectures, removing the warmup phase is more damaging for Transformers, especially Post-LN. Similarly, it has been found that the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant (Baevski and Auli, 2019; Nguyen and Salazar, 2019; Wang et al., 2019). Our studies go beyond the existing literature on gradient vanishing (Xiong et al., 2019) and identify an essential factor influencing Transformer training greatly.

Deep Network Initialization. It has been observed that deeper networks can lead to better performance. For example, Dong et al. (2020) find that the network depth players a similar role with the sample number in numerical ODE solvers, which hinders the system from getting more precise results. Many attempts have been made to clear obstacles for training deep networks, including various initialization methods. Based on the independence among initialized parameters, one method is derived and found to be useful to handle the gradient vanishing (Glorot and Bengio, 2010). Similar methods are further developed for ReLU networks (He et al., 2015). He et al. (2016) find that deep network training is still hard even after addressing the gradient vanishing issue and propose residual networks. Balduzzi et al. (2017b) identifies the shattered gradient issue and proposes LookLinear initialization.

On the other hand, although it is observed that scaling residual outputs to smaller values helps to stabilize training (Hanin and Rolnick, 2018; Mishkin and Matas, 2015; Zhang et al., 2019a; Bachlechner et al., 2020; Goyal et al., 2017), there is no systematic analysis on what complicates Transformer training or its underlying connection to the dependency on residual branches. Here, we identify that unbalanced gradients are not the direct cause of the Post-LN instability, recognize the amplification effect, and propose a novel adaptive initialization method.

## 7 Conclusion

In this paper, we study the difficulties of training Transformers in theoretical and empirical manners. Our study in Section 3 suggests that the gradient vanishing problem is not the root cause of unstable Transformer training. Also, the unbalanced gradient distribution issue is mostly addressed by adaptive optimizers. In Section 4, we reveal the root cause of the instability to be the strong dependency on residual branches, which amplifies the fluctuation caused by parameter changes and destabilizes model training. In light of our analysis, we propose Admin, an adaptive initialization method to stabilize Transformers training. It controls the dependency at the beginning of training and maintains the flexibility to capture those dependencies once training stabilizes. Extensive experiments verify our intuitions and show that, without introducing additional hyper-parameters, Admin achieves more stable training, faster convergence, and better performance.

Our work opens up new possibilities to not only further push the state-of-the-art but understand deep network training better. It leads to many interesting future works, including generalizing Theorem 2 to other models, designing new algorithms to automatically adapt deep networks to different training configurations, upgrading the Transformer architecture, and applying our proposed Admin to conduct training in a larger scale.

## Acknowledge

We thank all reviewers for their constructive comments; Chengyu Dong, Haoming Jiang, Jingbo Shang, Xiaotao Gu, and Zihan Wang for valuable discussions and comments; Jingbo Shang for sharing GPU machines; and Microsoft for setting up GPU machines. The research was sponsored in part by DARPA No. W911NF-17-C-0099 and No. FA8750-19-2-1004, National Science Foundation IIS-19-56151, IIS-17-41317, IIS 17-04532, and IIS 16-18481, and DTRA HDTRA11810026.

## References

Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer normalization. ArXiv, abs/1607.06450.

Thomas C. Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, and Julian J. McAuley. 2020. Rezero is all you need: Fast convergence at large depth. ArXiv, abs/2003.04887.

Alexei Baevski and Michael Auli. 2019. Adaptive input representations for neural language modeling. In ICLR.

David Balduzzi, Marcus Frean, Lennox Leary, J. P. Lewis, Kurt Wan-Duo Ma, and Brian McWilliams. 2017a. The shattered gradients problem: If resnets are the answer, then what is the question? In ICML.

David Balduzzi, Marcus Frean, Lennox Leary, J P Lewis, Kurt Wan-Duo Ma, and Brian McWilliams. 2017b. The shattered gradients problem: If resnets are the answer, then what is the question? In ICML.

Yoshua Bengio, Patrice Y. Simard, and Paolo Frasconi. 1994. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks.

Ondřej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve SaintAmand, et al. 2014. Findings of the 2014 workshop on statistical machine translation. In Workshop on Statistical Machine Translation.

Mauro Cettolo, Jan Niehues, Sebastian Stüker, Luisa Bentivogli, and Marcello Federico. 2014. Report on the 11th iwslt evaluation campaign, iwslt 2014. In International Workshop on Spoken Language Translation, Hanoi, Vietnam.

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Michael Schuster, Zhi-Feng Chen, Yonghui Wu, and Macduff Hughes. 2018. The best of both worlds: Combining recent advances in neural machine translation. In $A C L$.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT

Chengyu Dong, Liyuan Liu, Zichao Li, and Jingbo Shang. 2020. Towards adaptive residual network training: A neural-ode perspective. In ICML.

Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In AISTATS.

Priya Goyal, Piotr Dollár, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. 2017. Accurate, large minibatch sgd: Training imagenet in 1 hour. ArXiv, abs/1706.02677.
Boris Hanin and David Rolnick. 2018. How to start training: The effect of initialization and architecture. In NeurIPS.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In CVPR.

Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. 2019. Music transformer: Generating music with long-term structure. In ICLR.

Hao Li, Zheng Xu, Gavin Taylor, and Tom Goldstein. 2018. Visualizing the loss landscape of neural nets. In NeurIPS.

Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. 2020a. On the variance of the adaptive learning rate and beyond. In ICLR.

Xiaodong Liu, Kevin Duh, Liyuan Liu, and Jianfeng Gao. 2020b. Very deep transformers for neural machine translation. ArXiv, abs/2008.07772.

Yiping Lu, Zhuohan Li, Di He, Zhiqing Sun, Bin Dong, Tao Qin, Liwei Wang, and Tie-Yan Liu. 2020. Understanding and improving transformer from a multiparticle dynamic system point of view. In ICLR Workshop DeepDiffEq.

Dmytro Mishkin and Juan E. Sala Matas. 2015. All you need is a good init. In ICLR.

Toan Q. Nguyen and Julian Salazar. 2019. Transformers without tears: Improving the normalization of selfattention. In IWSLT.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In NAACL-HLT Demonstrations.

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. 2018. Image transformer. In ICML.

Martin Popel and Ondrej Bojar. 2018. Training tips for the transformer model. The Prague Bulletin of Mathematical Linguistics, 110:43 - 70 .

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv, abs/1910.10683.

Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jonathon Shlens. 2019. Stand-alone self-attention in vision models. In NeurIPS.

Andrew M Saxe, James L McClelland, and Surya Ganguli. 2013. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ArXiv, $\mathrm{abs} / 1312.6120$.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In CVPR.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS.

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph attention networks. In ICLR.

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. 2019. Learning deep transformer models for machine translation. In $A C L$.

Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, and Michael Auli. 2019a. Pay less attention with lightweight and dynamic convolutions. In ICLR.

Lijun Wu, Yiren Wang, Yingce Xia, Fei Tian, Fei Gao, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2019b. Depth growing for neural machine translation. In ACL.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shu xin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Li-Wei Wang, and Tie-Yan Liu. 2019. On layer normalization in the transformer architecture. ArXiv, abs/2002.04745.

Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma. 2019a. Fixup initialization: Residual learning without normalization. In ICLR.

Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank J. Reddi, Surinder Kumar, and Suvrit Sra. 2019b. Why adam beats sgd for attention models. ArXiv, abs/1912.03194.

Guangxiang Zhao, Xu Sun, Jingjing Xu, Zhiyuan Zhang, and Liangchen Luo. 2019. Muse: Parallel multi-scale attention for sequence to sequence learning. ArXiv, $\mathrm{abs} / 1911.09483$.
</end of paper 1>


<paper 2>
# MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing 

Unggi Lee ${ }^{1}$, Yonghyun Park ${ }^{2}$, Yujin Kim ${ }^{3}$,<br>Seongyune Choi ${ }^{1}$, Hyeoncheol Kim ${ }^{1 *}$<br>${ }^{1}$ Department of Computer Science and Engineering, Korea University, Seoul, Republic of Korea<br>${ }^{2}$ Department of Physics Education, Seoul National University, Seoul, Republic of Korea<br>${ }^{3}$ Department of Science Education, Ewha Womans University, Seoul, Republic of Korea<br>codingchild @ korea.ac.kr, enkeejunior1 @ snu.ac.kr, hello.yujink @ gmail.com,<br>\{csyun213, harrykim\}@korea.ac.kr


#### Abstract

Knowledge tracing (KT) is a field of study that predicts the future performance of students based on prior performance datasets collected from educational applications such as intelligent tutoring systems, learning management systems, and online courses. Some previous studies on KT have concentrated only on the interpretability of the model, whereas others have focused on enhancing the performance. Models that consider both interpretability and the performance improvement have been insufficient. Moreover, models that focus on performance improvements have not shown an overwhelming performance compared with existing models. In this study, we propose MonaCoBERT, which achieves the best performance on most benchmark datasets and has significant interpretability. MonaCoBERT uses a BERT-based architecture with monotonic convolutional multihead attention, which reflects forgetting behavior of the students and increases the representation power of the model. We can also increase the performance and interpretability using a classical testtheory-based (CTT-based) embedding strategy that considers the difficulty of the question. To determine why MonaCoBERT achieved the best performance and interpret the results quantitatively, we conducted ablation studies and additional analyses using Grad-CAM, t-SNE, and various visualization techniques. The analysis results demonstrate that both attention components complement one another and that CTTbased embedding represents information on both global and local difficulties. We also demonstrate that our model represents the relationship between concepts.


## Introduction

The outbreak of COVID-19 has accelerated the digital transformation in the field of education, and the number of students who use online learning platforms has increased. Most online learning platforms collect student data such as interaction logs, correctness, and learning history, thereby providing a chance to develop a better adaptive learning system for students.

Knowledge tracing (KT) is a research area that predicts the future performance of students based on prior performance datasets collected from educational applications such as intelligent tutoring systems (ITS), learning management systems (LMS), and online courses. KT models can be broadly classified into two categories: those focusing on interpretability and those focusing on increasing the performance. Models focused on interpretability, such as BKT (Corbett and Anderson 1994) and PFA (Pavlik Jr, Cen, and Koedinger 2009), mainly use Markov chain techniques and logistic functions. These models are generally simple to interpret. However, they suffer from a relatively low performance (Piech et al. 2015). Models focusing on a performance improvement, such as DKT, DKVMN (Zhang et al. 2017), SAKT (Pandey and Karypis 2019), and CL4KT (Lee et al. 2022), perform significantly better than traditional statistical approaches. However, this is difficult to interpret because of the nature of deep learning. Although AKT (Ghosh, Heffernan, and Lan 2020) considers both the model performance and interpretability, it has not shown an overwhelming performance in comparison to existing models.

Moreover, despite the application of self-attention and transformers (Vaswani et al. 2017) in recent KT models, they have been unable to improve the attention architecture compared to changes in other parts of the models. In natural language processing (NLP), Bigbird (Zaheer et al. 2020), Longformer (Beltagy, Peters, and Cohan 2020), and ConvBERT (Jiang et al. 2020), which maintain the BERT architecture and change the attention architectures, have succeeded in terms of both performance and efficiency.

In this article, MonaCoBERT, which achieves both a high performance and interpretability, is proposed. MonaCoBERT uses BERT-based and monotonic convolutional attention architectures. We also suggest a classical test theory (CTT) based embedding strategy that considers the question difficulty. Using CTT-based embedding, our model achieves a increase in performance and interpretability. As a result, MonaCoBERT achieved state-of-the-art results on most benchmark datasets in term of both the AUC and RMSE. Moreover, we also conducted an ablation study on all parts of the models and an additional analysis using Grad-CAM, $\mathrm{t}-\mathrm{SNE}$, and various visualization techniques. The analysis results demonstrate that both attention components complement one another and that CTT-based embedding represents information on both global and local difficulties. We also demonstrate that our model represents the relationship between concepts.[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_50fffd4c9b837f6b9da7g-2.jpg?height=492&width=1520&top_left_y=172&top_left_x=300)

Figure 1: Architectures of MonaCoBERT and monotonic convolutional multi-head attention. The left side shows different strategies in training and testing sessions. The right side shows the architecture of monotonic convolutional multi-head attention, combined with monotonic attention and ConvBERT attention.

## Related Work

## Knowledge Tracing

Knowledge tracing (KT) is a research area of predicting the knowledge states of the students using their interaction data. Since the first introduction of DKT (Piech et al. 2015), significant research in this area using deep neural networks has been conducted. Researchers have recently focused on self-attention architectures. SAKT (Pandey and Karypis 2019), SAINT+ (Shin et al. 2021), which uses selfattention, achieves a better performance than previous models. Moreover, AKT (Ghosh, Heffernan, and Lan 2020) was presented with self-attention, and a new architecture for retrieving latent knowledge representations was suggested. For AKT, a new embedding method that considers the educational perspective has also been suggested. CL4KT (Lee et al. 2022) also uses a self-attention and contrastive learning framework, and has achieved the best performance in KT.

## BERT and Its Application

BERT (Devlin et al. 2018) has been referred to as a successful application of Transformer. It mainly uses self-attention, and the masked language model (MLM) method, which can train bidirectionally in NLP, has been suggested. Some variations of BERT, such as Bigbird (Zaheer et al. 2020), Longformer (Beltagy, Peters, and Cohan 2020), and ConvBERT, (Jiang et al. 2020) have recently been designed with effective attention mechanisms applied while using the original architecture of BERT. These approaches have achieved an outstanding performance and high efficiency. Other studies have also attempted to use BERT architectures. As a recommendation system, BERT4Rec (Sun et al. 2019) uses the BERT architecture to enhance the recommendation power. However, for KT, although BEKT (Tiana et al. 2021) and BiDKT (Tan et al. 2022) attempt to use the BERT architecture, they cannot achieve a higher performance than other KT models. In this study, we explored why BERT does not perform better than the other models and showed that the BERT architecture is still valuable for KT. We found that changing the attention architecture and an embedding strategy are vital to optimizing BERT for the KT area.

## Method

## Problem Statement

Knowledge tracing aims at predicting the probabilities of students being correct through the use of a sequence of interaction data gathered by an LMS or ITS. Student interactions can be expressed as $x_{1}, \ldots, x_{t}$, and the $t$-th interaction can be denoted as $x_{t}=\left(q_{t}, a_{t}\right)$. Here, $q_{t}$ is the $\mathrm{t}-t h$ question and $a_{t}$ is $t$-th correctness of the student's response, where $a_{t} \in\{0,1\}$, in which 0 indicates an incorrect response and 1 is a correct answer. However, some datasets contain concept data $c_{t}$, and thus we can also express $x_{t}=\left(q_{t}, c_{t}, a_{t}\right)$.

## Proposed Model Architecture

BERT based Architecture for Knowledge Tracing To create our model baseline, we mainly referenced BERT (Devlin et al. 2018), BiDKT (Tan et al. 2022), BEKT (Tiana et al. 2021), and BERT4Rec (Sun et al. 2019). To optimize our research into KT, we changed some of the BERT architecture. First, we used a pre-layer normalization (preLN) Transformer in our model. Previous research (Liu et al. 2020) has suggested that Transformer is difficult to train without a training strategy, such as a warm-up start. By contrast, the pre-LN Transformer can be trained without a warm-up start and converges much faster than the original Transformer (Xiong et al. 2020). Second, we used a different strategy for the training and testing processes. During the training process, the proposed model predicted the masking position. The masking ratio used in the training process was the same as with the original BERT, which used $15 \%$ embedding, $80 \%$ of which was actual masking, $10 \%$ was a reversal, and $10 \%$ did not change. During the testing process, masking was applied to the last position of each sequence. Referring to the previous BERT-based studies on KT (Tan et al. 2022) or recommendation systems (Sun et al. 2019), the model predicts the correctness of the students using their previous history of interaction sequences. Figure 1-Left describes the different training and testing strategies of our model.

Embedding Strategy Most KT models use concepts, questions, and correctness as the input vectors for train-
ing. Previous studies have explored new input features. For example, AKT created Rasch embedding vectors by using concepts, items, and responses Ghosh, Heffernan, and Lan 2020). However, an item response theory (IRT), such as Rasch, can be applied to the dataset collected from tests or examinations because IRT assumes that the ability of a student does not change during the trial. In KT, the states of student knowledge change during learning (Yeung|2019). Therefore, we used the classical test theory (CTT) for handling the difficulty features.

We extracted the correctness of each question from the training set and made the questions difficulty. If the question in the validation or test set were not contained in training set, we replaced that question difficulty as a arbitrarily number like 75. Subsequently, we added the difficulty to the embedding blocks. In a previous study, BEKT Tiana et al. 2021) used five difficulty ranges in its embedding blocks. Nevertheless, we used a difficulty range of 100. Similar to BERT embedding layers, MonaCoBERT uses elementwise embedding vectors $E_{\text {input }}$, learnable positional embedding $E_{\text {pos }}$, concept embedding $E_{c}$, item embedding $E_{q}$, correctness embedding $E_{a}$, and CTT embedding $E_{c t t}$, where $E_{\text {input }} \in R^{m \times h}, E_{\text {pos }} \in R^{m \times h}, E_{c} \in R^{m \times h}, E_{q} \in$ $R^{m \times h}, E_{a} \in R^{m \times h}$ and $E_{c t t} \in R^{m \times h}$. Embedding layers $E_{\text {input }}$ are formulated as follows:

$$
\begin{equation*}
E_{\text {input }}=E_{\text {pos }}+E_{c}+E_{q}+E_{a}+E_{c t t} \tag{1}
\end{equation*}
$$

Pre-LN Transformer-based Encoder Architecture The encoder blocks used the pre-LN Transformer architecture (Xiong et al. 2020). In this study, 12 encoder layers were used. First, the embedding vectors $E_{\text {input }}$ are normalized through the pre-LN $L N_{\text {pre }}$

$$
\begin{equation*}
z=L N_{\text {pre }}\left(E_{\text {input }}\right) \tag{2}
\end{equation*}
$$

Second, the normalized value $z$ was changed to the query, key, and value of monotonic convolutional multihead attention. The results were passed through dropout layer $D$ and added to the embedding vectors as a residual connection.

$$
\begin{equation*}
a=x+D(\text { MonoConvMulAttn }(z, z, z)) \tag{3}
\end{equation*}
$$

Third, the results were normalized and passed through fully connected layers. The activation function was LeakyReLU. The results were also normalized through the dropout layer $D$. The second result was added as a residual connection.

$$
\begin{equation*}
l=a+D(f c(L N(a))) \tag{4}
\end{equation*}
$$

The fully connected layers are formulated as follows.

$$
\begin{equation*}
f c=W_{f c 2}\left(L e a k y \operatorname{Re} L U\left(W_{f c 1}\right)\right) \tag{5}
\end{equation*}
$$

where $W_{f c 1} \in R^{h \times(h * n)}, W_{f c 2} \in R^{(h * n) \times h}$.

Monotonic Convolutional Multihead Attention We suggest the use of monotonic convolutional multihead attention. This architecture is combined with ConvBERT's (Jiang et al. 2020) mixed-attention and AKT Ghosh, Heffernan, and Lan 2020) monotonic attention. In previous research, mixed attention achieved a higher performance than normal attention with BERT. Meanwhile, the sequence data in KT contain latent information about the forgetting of the students. To represent such forgetting, we used the exponential decay mechanism of monotonic attention. Figure 1-Right shows the monotonic convolutional multihead attention architecture.

The monotonic convolutional multihead attention $A_{m c}$ consists of the concatenation ([;]) of monotonic multihead attention $A_{m}$ and span-based dynamic convolution $S D C$. Here, $A m$ is the same as monotonic attention and can be formulated as follows:

$$
\begin{equation*}
A_{m c}(Q, K, V)=\left[A_{m}(Q, K, V) ; S D C(Q, K, V)\right] \tag{6}
\end{equation*}
$$

First, monotonic multihead attention $A_{m}$ has an exponential decay mechanism for measuring the distance between sequences. The exponential decay mechanism is a dot product with query linear $W_{Q}$ and key linear $W_{K}$. The learnable parameter $\delta$ is multiplied by these values. In addition, $A_{m}$ can be formulated as follows:

$$
\begin{equation*}
\operatorname{Am}=\operatorname{softmax}\left(\frac{(-\delta \cdot d(t, \tau)) \cdot W_{Q} \cdot W_{K}}{\sqrt{D_{k}}}\right), \delta>0 \tag{7}
\end{equation*}
$$

Here, $d(t, \tau)$ is the distance function, where $t$ is the present time step, and $\tau$ is the previous time step. In addition, $d(t, \tau)$ can be formulated as

$$
\begin{equation*}
d(t, \tau)=|t-\tau| \cdot \sum_{t^{\prime}=\tau+1}^{t} \gamma_{t, t^{\prime}} \tag{8}
\end{equation*}
$$

Moreover, $\gamma_{t, t^{\prime}}$ can be formulated as

$$
\begin{equation*}
\gamma=\frac{\exp \left(\frac{W_{q t} \cdot W_{k t}^{\prime}}{\sqrt{D_{k}}}\right)}{\sum_{1 \leq \tau^{\prime} \leq t} \exp \left(\frac{W_{q t} \cdot W_{k \tau^{\prime}}}{\sqrt{D_{k}}}\right)}, t^{\prime} \leq t \tag{9}
\end{equation*}
$$

The span dynamic convolution $S D C$ can be formulated as

$$
\begin{equation*}
S D C(Q, K, V)=L \operatorname{Conv}(V, \operatorname{softmax}(W(Q \otimes K))) \tag{10}
\end{equation*}
$$

where $W$ is a linear layer, and $\otimes$ can be denoted as a point-wise multiplication. The lightweight convolution LConv can be formulated as follows:

$$
\begin{equation*}
\operatorname{LConv}(X, W)=\sum_{j=1}^{k} W_{j} \dot{X}_{i+j-\left[\frac{[k+1]}{2}\right]} \tag{11}
\end{equation*}
$$

## Experiment Setting

Datasets Six benchmark datasets were used to validate the effectiveness of our model. We ignored student data with fewer than five interactions. If the dataset contained multiple concepts in a single interaction, we treated the combination of concepts as unique. The ASSISTment datasets were collected from the ASSISTment ITS. We used assist09, assist12, assist17 and ignored assist15, which has no information regarding the questions 1 . The algebra datasets were[^1]

| Dataset | Metrics | DKT | DKVMN | SAKT | AKT | CL4KT | MCB-NC | MCB-C |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| assist09 | AUC | 0.7285 | 0.7271 | 0.7179 | 0.7449 | $\underline{0.7600}$ | $\mathbf{0 . 8 0 0 2}$ | $\underline{\mathbf{0 . 8 0 5 9}}$ |
|  | RMSE | $\underline{0.4328}$ | 0.4348 | 0.4381 | 0.4413 | 0.4337 | $\underline{\mathbf{0 . 4 0 2 9}}$ | $\mathbf{0 . 4 0 6 3}$ |
| assist12 | AUC | 0.7006 | 0.7011 | 0.6998 | $\underline{0.7505}$ | 0.7314 | $\mathbf{0 . 8 0 6 5}$ | $\underline{\mathbf{0 . 8 1 3 0}}$ |
|  | RMSE | 0.4348 | 0.4355 | 0.4360 | $\underline{0.4250}$ | 0.4284 | $\mathbf{0 . 3 9 7 6}$ | $\underline{\mathbf{0 . 3 9 3 5}}$ |
| assist17 | AUC | $\underline{\mathbf{0 . 7 2 2 0}}$ | $\underline{0.7095}$ | 0.6792 | 0.6803 | 0.6738 | 0.6700 | $\mathbf{0 . 7 1 4 1}$ |
|  | RMSE | $\underline{\mathbf{0 . 4 4 6 9}}$ | $\mathbf{0 . 4 5 1 6}$ | $\underline{0.4591}$ | 0.4722 | 0.4713 | 0.4727 | 0.4630 |
| algebra05 | AUC | 0.8088 | 0.8146 | $\underline{0.8162}$ | 0.7673 | 0.7871 | $\mathbf{0 . 8 1 9 0}$ | $\underline{\mathbf{0 . 8 2 0 1}}$ |
|  | RMSE | 0.3703 | $\underline{0.3687}$ | $\mathbf{0 . 3 6 8 5}$ | 0.3918 | 0.3824 | 0.3940 | $\underline{\mathbf{0 . 3 5 8 4}}$ |
| algebra06 | AUC | 0.7939 | $\underline{\underline{0.7961}}$ | 0.7927 | 0.7505 | 0.7789 | $\mathbf{0 . 7 9 9 7}$ | $\underline{\mathbf{0 . 8 0 6 4}}$ |
|  | RMSE | $\mathbf{0 . 3 6 6 6}$ | $\underline{\mathbf{0 . 3 6 6 1}}$ | 0.3675 | 0.3986 | 0.3863 | 0.3835 | $\underline{\mathbf{0 . 3 6 7 2}}$ |
| EdNet | AUC | 0.6609 | $\underline{0.6602}$ | 0.6506 | $\underline{0.6687}$ | 0.6651 | $\mathbf{0 . 7 2 2 1}$ | $\underline{\mathbf{0 . 7 3 3 6}}$ |
|  | RMSE | 0.4598 | $\underline{0.4597}$ | 0.4629 | 0.4783 | 0.4750 | $\mathbf{0 . 4 5 7 2}$ | $\underline{\mathbf{0 . 4 5 1 6}}$ |

Table 1: Overall performance of knowledge tracing models based on five benchmark datasets. The best performance is denoted in bold underline, the second in bold, and the third in underline. MCB-C indicates that MonaCoBERT used classical test theory (CTT), whereas MCB-NC indicates that it did not. We can see that MCB-C achieved the best results, and MCB-NC was second for most of the benchmark datasets.

provided by the KDD Cup 2010 EDM Challeng $d^{2}$ EdNe ${ }^{3}$ is a dataset collected by Santa for the learning of English, mainly TOEIC (Choi et al. 2020). We extracted 5,000 interaction data from the original dataset. Table 2 lists the number of features in the benchmark dataset.

| Dataset | \#Students | \#Concepts | \#Questions | \#interactions |
| :---: | :---: | :---: | :---: | :---: |
| assist09 | 3,695 | 149 | 17,728 | 282,071 |
| assist12 | 24,429 | 264 | 51,632 | $1,968,737$ |
| assist17 | 1,708 | 411 | 3,162 | 934,638 |
| algebra05 | 571 | 271 | 173,113 | 607,014 |
| algebra06 | 1,318 | 1,575 | 549,821 | $1,808,533$ |
| EdNet | 5,000 | 1,472 | 11,957 | 641,712 |

Table 2: Benchmark dataset ignored student data with less than five interactions. \#Concepts are the same as the skills.

Evaluation Metrics and Validation By referencing CL4KT, we used both the area under the curve (AUC) and the root mean squared error (RMSE) as the performance metrics. We also used a five-fold cross-validation for the evaluation.

Baseline Models We compared MonaCoBERT to the baseline models, such as DKT (Piech et al. 2015), DKVMN (Zhang et al. 2017), SAKT (Pandey and Karypis 2019), and the latest models, such as AKT (Ghosh, Heffernan, and Lan 2020) and CL4KT. (Lee et al. 2022).[^2]

Hyperparameters for Experiments To compare each model, we used the same parameters for the model training.

- batch size: The batch size was 512 . Owing to a limitation of resources, we also used a gradient accumulation.
- early stop: The early stop was 10 . If the validation score was not successively increased during the ten iterations, the training session was stopped.
- training, validation, test ratio: The training ratio was $80 \%$ of the entire dataset, and the test ratio was $20 \%$. The valid ratio was $10 \%$ of the training ratio.
- learning rate and optimizer: The learning rate was 0.001, and Adam was used as the optimizer.
- embedding size: The embedding size was 512 .
- others: We used eight attention heads for MonaCoBERT. The Max sequence length was 100 , and the encoder number was 12. Other models such as $\mathrm{AKT}_{4}^{4}$ and CL4KT used the default settings.


## Result and Discussion

## Overall Performance

Figure 1 illustrates the overall performance of each model. Every model used a five-fold cross-validation for the estimation. MonaCoBERT-C, which was trained using CTT, was the best model in most benchmark datasets and was a new state-of-the-art model in assist09, assist12, algebra05, and ednet. MonaCoBERT-NC was the second-best model for most of the datasets. This result indicates that CTT embedding affects the performance of the model. For all datasets,[^3]

MonaCoBERT-C performed better than MonaCoBERT-NC. This result indicates that it was difficult for MonaCoBERT$\mathrm{NC}$ to learn the latent representations of the item difficulty from the dataset.

Our estimation differs from that of previous research. Except for MonaCoBERT-NC and MonaCoBERT-C, the best model was modified for each dataset. For instance, the AUC and RSME of assist17, and the RMSE, DKT, and DKVMN of algebra06 showed that these were the best and second-best models, respectively. This indicates that DKT and DKVMN are still helpful in predicting certain cases. These results may stem from pre-processing methods or the training of the hyperparameter settings.

## Ablation Studies

In this section, we explore why MonaCoBERT performed better than the other models and which parts of the model affected the increase in performance.

Impact of Attention Mechanisms In Table 3, we compare the attention mechanisms. For comparison, we used the assist09 and assist09-CTT datasets. The assist09 dataset is a normal dataset that contains concepts, questions, and correctness; however, assist09-CTT contains the concepts, questions, correctness, and CTT-based difficulty.

We detached each part of the monotonic convolutional multi-head attention and created four attention mechanisms: normal multi-head attention, monotonic multi-head attention, convolutional multi-head attention, and monotonic convolutional multi-head attention. We also used a five-fold cross-validation and an early stop 10 times. The other hyperparameters used to determine the overall performance were the same.

As a result, monotonic convolutional multihead attention exhibited the best performance for both comparisons. Convolutional multihead attention and monotonic multihead attention achieved the second-best performance under each setting. The increments differed for each setting and were approximately $2 \%$ for assist09 and 1-2\% for assist09-CTT.

| Dataset | Attn | MonoAttn | ConvAttn | MonoCoAttn |
| :---: | :---: | :---: | :---: | :---: |
| assist09 | 0.7736 | $\mathbf{0 . 7 9 9 3}$ | $\underline{0.7959}$ | $\underline{\mathbf{0 . 8 0 0 2}}$ |
| increment | 0 | +0.026 | +0.022 | +0.027 |
| assist09-CTT | 0.7858 | $\underline{0.8039}$ | $\mathbf{0 . 8 0 5 4}$ | $\underline{\mathbf{0 . 8 0 5 9}}$ |
| increment | 0 | +0.018 | +0.020 | +0.021 |

Table 3: AUC performances of each attention mechanism using the assist09 and assist09-CTT datasets. The increments were written based on normal attention.

Impacts of Embedding Strategy In Table 4, we compare each embedding strategy. The first embedding strategy $e m b_{c q}$ is an element-wise sum of the concept embedding $e m b_{c}$, question embedding $e m b_{q}$, and correctness embedding $e m b_{r}$.

$$
\begin{equation*}
e m b_{c q}=e m b_{c}+e m b_{q}+e m b_{r} \tag{12}
\end{equation*}
$$

Moreover, the second embedding strategy $e m b_{\text {rasch }}$ is an element-wise sum of concept and Rasch embedding, as suggested by AKT. Rasch embedding uses concept embedding $e m b_{c}$ and learnable question scalar $e m b_{q}$ or a combination of concepts and answer embedding $e m b_{c r}$ to calculate the difficulty, where $e m b_{c}, e m b_{c r} \in R^{n \times h}$ and $e m b_{q} \in R^{n \times 1}$. Note that IRT Rasch embedding differs from AKT Rasch embedding because the condition of IRT assumes that the knowledge state of the student is fixed and does not change when estimated.

$$
\begin{gather*}
e m b_{r a s c h-c}=e m b_{c}+e m b_{q} * e m b_{c}  \tag{13}\\
e m b_{r a s c h-c r}=e m b_{c r}+e m b_{q} * e m b_{c r} \tag{14}
\end{gather*}
$$

The last embedding strategy, $e m b_{C T T}$, is an element-wise sum of concept embedding, question embedding, correctness embedding, and CTT embedding, $e m b_{c t t}$, which was suggested in this study. We set $e m b_{c t t}$ as the probability of the difficulty and the integer type, where $0 \leq e m b_{c t t} \leq 100$.

$$
\begin{equation*}
e m b_{C T T}=e m b_{c}+e m b_{q}+e m b_{r}+e m b_{c t t} \tag{15}
\end{equation*}
$$

As a result, in Table 4, emb ${ }_{C T T}$ generally showed a better performance than the other embedding strategies. DKVMN, AKT, and MonaCoBERT performed well when using $e m b_{C T T}$. This result indicates that the models did not learn the difficulty representation during training. Meanwhile, CL4KT and SAKT showed slightly better performances when using $e m b_{\text {rasch }}$. DKT was not affected by the embedding strategy.

| Embedding Strategy | $e m b_{c q}$ | $e m b_{\text {rasch }}$ | $e m b_{C T T}$ |
| :---: | :---: | :---: | :---: |
| DKT | 0.7263 | $\mathbf{0 . 7 2 7 4}$ | 0.7239 |
| DKVMN | 0.7188 | 0.7255 | $\mathbf{0 . 7 3 1 3}$ |
| SAKT | 0.6822 | $\mathbf{0 . 6 9 4 1}$ | 0.6693 |
| AKT | 0.7440 | 0.7449 | $\mathbf{0 . 7 6 3 2}$ |
| CL4KT | 0.7600 | $\mathbf{0 . 7 6 0 1}$ | 0.7461 |
| MCB | 0.8002 | 0.7736 | $\mathbf{0 . 8 0 5 9}$ |

Table 4: Comparison of each embedding strategy with KT models in the assist09 dataset.

## In-depth Analysis of Attention and Embedding

In this subsection, we analyze the attention and embedding in depth. We used Grad-CAM and t-SNE for the analysis and visualization.

Analysis of MA and SDC Owing to the nature of KT, such as the forgetting behavior of the students, we expected that monotonic attention (MA) will look up the nearby data regarding the current input. However, as shown in Figure 3. MA induced higher attention scores for the distant data,
![](https://cdn.mathpix.com/cropped/2024_06_04_50fffd4c9b837f6b9da7g-6.jpg?height=414&width=1632&top_left_y=178&top_left_x=238)

Figure 2: Analysis of underlying behaviors of SDC. The figure on the left illustrates the proportion of the importance of each module. SDC showed an importance competitive to that of MA in most layers. In particular, the SDC showed the most significant contribution in the first layer. The histogram in the center figure represents the current input weight of the concept. When the response of the student was correct, the SDC allocated more weight to the interaction. In addition, even if the response was the same, the weight varied considerably based on the concept. The figure on the right shows examples of SDC filters arranged based on the correctness and concept.

not nearby data. We also observed that SDC was more critical than MA in the first layer. Figure 2 -Left shows the relative importance ratios of SDC and MA. The contribution of SDC was greater than that of MA in the first layer. To define the importance of each module, we used an element-wise version of Grad-CAM as a metric (Selvaraju et al. 2017; Gildenblat and contributors 2021). We also found that SDC extracted useful information regarding the properties of the current input. Specifically, SDC focused on the current input when the student answered correctly. In Figure 2 -Center, Right, we can see that SDC assigned higher weights to the current inputs when the student responded correctly. Moreover, the large variance of weights given correct responses implies that SDC considers not only the correctness of responses but also the importance of the concept. (Figure 2 . Center, blue) This result shows us that MonaCoBERT implicitly learned what concepts or questions were essential for inferring the ability of the students. This indicates the possibility of using MonaCoBERT to automatically find the problem essential to estimating the student's ability, which can be used to support the estimation and assessment.
![](https://cdn.mathpix.com/cropped/2024_06_04_50fffd4c9b837f6b9da7g-6.jpg?height=244&width=740&top_left_y=1827&top_left_x=237)

Figure 3: Analysis of the attention map of monotonic selfattention (MA). The figure on the left shows the attention weight according to the distance of the interaction data, and indicates that models with MA (e.g., SDC + MA, MA) display more outstanding attention scores for distant tokens. The figure on the right shows an example of the attention map of MonaCoBERT.
CTT based embedding We showed that CTT-based embedding helps the model represent the difficulty of the problem. Figure 4 shows a visualization using t-SNE (?). Figure 4.Left shows the visualization of the CTT-based embedding vector, and Figure 4-Right shows the visualization of the NoCTT-based embedding. Unlike No-CTT-based embedding, where different difficulties are mixed in each cluster, CTTbased embedding (i.e., $e m b_{C T T}$ ) showed that the difficulty of the information was smoothly distributed globally.
![](https://cdn.mathpix.com/cropped/2024_06_04_50fffd4c9b837f6b9da7g-6.jpg?height=330&width=728&top_left_y=1323&top_left_x=1206)

Figure 4: Visualization of the embedding vector. The figure on the left shows the results with CTT-based embedding. The figure on the right shows the results of No-CTT-based embedding. We can see that the results of CTT-based embedding not only represent the difficulty information globally, they also help avoid a difficulty in the mixing in each cluster.

## Discovery of Relationships between Concepts

To determine whether our model understood the relevance between concepts, we analyzed the monotonic attention weights of the last encoder layer after passing through the softmax function. The results are shown in Figure 5-left. We averaged the attention scores of the questions using the same concepts to obtain the relevance between concepts. We created a directed graph, as shown in Fig 5-Center, by selecting only those concepts with attention weights of higher than 0.1 .
![](https://cdn.mathpix.com/cropped/2024_06_04_50fffd4c9b837f6b9da7g-7.jpg?height=498&width=1684&top_left_y=212&top_left_x=208)

Figure 5: Analysis results of the relevance between concepts, exploiting attention weights of the monotonic attention part after the model was trained using monotonic convolutional multi-head attention. The figure on the left shows a heatmap of the attention weights between each pair of concepts. It shows how much attention each concept on the y-axis (e.g., 7th, 92nd, 94th, 96th) assigns attention to some selected concept on the other $\mathrm{x}$-axis. The center figure shows a directed graph of the relevance between concepts. It shows how the concepts of assist09 influence one another. The source concept nodes are assigned a high attention weight to the destination concept nodes, and the concept nodes can be connected in both directions. We set the threshold to 0.1 and ignored edges lower than the threshold. When the threshold was decreased, more skill nodes were connected, and vice versa. The concept information of the assist09 dataset can be found on the right. 'nan' means concepts that are not defined in the original dataset.

According to the concept network shown in Figure 5 center, we can see that the model learns the relevance between skills. For example, as shown in Figure 5-left, the 7th concept (Absolute Value) was connected with some concepts of subtraction, such as 92 (Addition and Subtraction Fractions), 94 (Addition and Subtraction Positive Decimals), and 96 (Addition and Subtraction Integers). This means that you need to be good at subtraction to calculate the correct absolute value. Accordingly, the 117th concept (Probability of a Single Event) and 115th concept (Probability of Two Distinct Events) assigned high attention weights to each other, since concept 117 is a prerequisite for concept 115 . 121st concept (Counting Methods) is also connected with 115 and 117. However, the concept network shown in Figure 5 is not perfect because some concepts did not connect to each other despite their similarities. This result may be due to the monotonic attention decreasing the attention weight according to the time step. Nevertheless, observing the attention weights can help uncover new connections between previously inconceivable concepts.

## Conclusion

In this study, we developed MonaCoBERT, which employs a BERT-based architecture with monotonic convolutional multihead attention for student forgetting and the representation power of the model. We also adopted an effective embedding strategy that represented difficulty based on a classic test theory. Consequently, MonaCoBERT exhibited superior performance on most benchmark datasets. We conducted an ablation study for each part of the model; consequently, we discovered that monotonic convolutional multihead attention aided in improving the model performance. Although the embedding strategy contributed significantly to the performance improvement of our model, we confirmed that depending on the model, the contribution of the embedding strategy to the performance enhancement differed. We conducted an additional analysis to quantitatively analyze the attention architecture and embedding strategy using Grad-CAM and t-SNE. Future research will be focused on improving the attention architecture and the difficulty embedding strategy.

## References

Beltagy, I.; Peters, M. E.; and Cohan, A. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Choi, Y.; Lee, Y.; Shin, D.; Cho, J.; Park, S.; Lee, S.; Baek, J.; Bae, C.; Kim, B.; and Heo, J. 2020. Ednet: A largescale hierarchical dataset in education. In International Conference on Artificial Intelligence in Education, 69-73. Springer.

Corbett, A. T.; and Anderson, J. R. 1994. Knowledge tracing: Modeling the acquisition of procedural knowledge. User modeling and user-adapted interaction, 4(4): 253-278.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Ghosh, A.; Heffernan, N.; and Lan, A. S. 2020. Contextaware attentive knowledge tracing. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining, 2330-2339.

Gildenblat, J.; and contributors. 2021. PyTorch library for CAM methods. https://github.com/jacobgil/pytorch-gradcam.

Jiang, Z.-H.; Yu, W.; Zhou, D.; Chen, Y.; Feng, J.; and Yan, S. 2020. Convbert: Improving bert with span-based dynamic convolution. Advances in Neural Information Processing Systems, 33: 12837-12848.

Lee, W.; Chun, J.; Lee, Y.; Park, K.; and Park, S. 2022. Contrastive Learning for Knowledge Tracing. In Proceedings of the ACM Web Conference 2022, 2330-2338.

Liu, L.; Liu, X.; Gao, J.; Chen, W.; and Han, J. 2020. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249.

McInnes, L.; Healy, J.; and Melville, J. 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

Pandey, S.; and Karypis, G. 2019. A self-attentive model for knowledge tracing. arXiv preprint arXiv:1907.06837.

Pavlik Jr, P. I.; Cen, H.; and Koedinger, K. R. 2009. Performance Factors Analysis-A New Alternative to Knowledge Tracing. Online Submission.

Piech, C.; Bassen, J.; Huang, J.; Ganguli, S.; Sahami, M.; Guibas, L. J.; and Sohl-Dickstein, J. 2015. Deep knowledge tracing. Advances in neural information processing systems, 28.

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, 618-626.

Shin, D.; Shim, Y.; Yu, H.; Lee, S.; Kim, B.; and Choi, Y. 2021. Saint+: Integrating temporal features for ednet correctness prediction. In LAK21: 11th International Learning Analytics and Knowledge Conference, 490-496.

Sun, F.; Liu, J.; Wu, J.; Pei, C.; Lin, X.; Ou, W.; and Jiang, P. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, 1441-1450.

Tan, W.; Jin, Y.; Liu, M.; and Zhang, H. 2022. BiDKT: Deep Knowledge Tracing with BERT. In International Conference on Ad Hoc Networks, International Conference on Testbeds and Research Infrastructures, 260-278. Springer.

Tiana, Z.; Zhengc, G.; Flanaganb, B.; Mic, J.; and Ogatab, H. 2021. BEKT: Deep Knowledge Tracing with Bidirectional Encoder Representations from Transformers. Proceedings of the 29th International Conference on Computers in Education.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30 .

Xiong, R.; Yang, Y.; He, D.; Zheng, K.; Zheng, S.; Xing, C.; Zhang, H.; Lan, Y.; Wang, L.; and Liu, T. 2020. On layer normalization in the transformer architecture. In International Conference on Machine Learning, 10524-10533. PMLR.

Yeung, C.-K. 2019. Deep-IRT: Make deep learning based knowledge tracing explainable using item response theory. arXiv preprint arXiv:1904.11738.
Zaheer, M.; Guruganesh, G.; Dubey, K. A.; Ainslie, J.; Alberti, C.; Ontanon, S.; Pham, P.; Ravula, A.; Wang, Q.; Yang, L.; et al. 2020. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33: 17283-17297.

Zhang, J.; Shi, X.; King, I.; and Yeung, D.-Y. 2017. Dynamic key-value memory networks for knowledge tracing. In Proceedings of the 26th international conference on World Wide Web, 765-774.


[^0]:    ${ }^{*}$ Corresponding author.

[^1]:    ${ }^{1}$ retrieved from https://sites.google.com/site/assistmentsdata/home

[^2]:    ${ }^{2}$ retrieved from https://pslcdatashop.web.cmu.edu/KDDCup

    ${ }^{3}$ retrieved from https://github.com/riiid/ednet

[^3]:    ${ }^{4}$ https://github.com/arghosh/AKT

    ${ }^{5}$ https://github.com/UpstageAI/cl4kt

</end of paper 2>


