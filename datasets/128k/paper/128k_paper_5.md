<paper 0>
# A Simple Recipe for Contrastively Pre-training Video-First Encoders Beyond 16 Frames 

Pinelopi Papalampidi* $\quad$ Skanda Koppula* $\quad$ Shreya Pathak<br>Justin Chiu Joe Heyward Viorica Patraucean Jiajun Shen Antoine Miech<br>Andrew Zisserman Aida Nematzdeh<br>Google DeepMind<br>\{pinelopi,skandak,shreyapa\}@google.com


#### Abstract

Understanding long, real-world videos requires modeling of long-range visual dependencies. To this end, we explore video-first architectures, building on the common paradigm of transferring large-scale, image-text models to video via shallow temporal fusion. However, we expose two limitations to the approach: (1) decreased spatial capabilities, likely due to poor video-language alignment in standard video datasets, and (2) higher memory consumption, bottlenecking the number of frames that can be processed. To mitigate the memory bottleneck, we systematically analyze the memory/accuracy trade-off of various efficient methods: factorized attention, parameter-efficient imageto-video adaptation, input masking, and multi-resolution patchification. Surprisingly, simply masking large portions of the video (up to $75 \%$ ) during contrastive pre-training proves to be one of the most robust ways to scale encoders to videos up to 4.3 minutes at 1 FPS. Our simple approach for training long video-to-text models, which scales to $1 B$ parameters, does not add new architectural complexity and is able to outperform the popular paradigm of using much larger LLMs as an information aggregator over segmentbased information on benchmarks with long-range temporal dependencies (YouCook2, EgoSchema).


## 1. Introduction

Long-video understanding requires modeling of the temporal dynamics and long-range visual dependencies of realworld scenes $[67,68]$. However, capturing long-range visual content is challenging, even when assisted by large language models (LLMs). In this paper, we overcome demonstrate how to extend video encoders to directly process minutes-long visual content using language grounding and[^0]

Step 2: Video-to-Text Fine-tuning

![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-01.jpg?height=705&width=833&top_left_y=932&top_left_x=1058)

Figure 1. As in Flamingo [1], we first (1) pre-train a visual encoder via Noise Contrastive Estimation, and then (2) use this frozen encoder with a pre-trained LM for video-to-text generation (e.g., video summarization and Q/A). For (1), we propose a two-stage process for pre-training a video encoder: (a) image-to-short video adaptation, and (b) short-to-long video, where we adapt the encoder to longer contexts, using video masking and layer freezing.

overcome hardware memory limitations using simple, established techniques without additional architectural complexity $[27,68]$. We focus on long videos through the lens of language, assessing our models on the widely applicable tasks of visual summarization and question-answering.

Recent work on vision-language models have yielded impressive results $[1,33,76]$, predominantly focusing on understanding images or short clips of sixteen frames or less $[1,16,33,76,77]$. These works recycle strong pretrained image encoders, and usually perform late temporal fusion [1, 75, 77], and employ mostly-frozen, powerful LLMs. The lack of video-first encoders, equipped with
early temporal aggregation, may handicap the ability to process complex visual dependencies, and this is usually reflected in prior work's focus on short video benchmarks ( $<30$ seconds) in which sixteen random frames are sufficient for competitive performance $[5,32]$.

In this work, we follow the standard image-language recipe $[1,33]$, where we first contrastively pre-train a visual encoder (Step 1; Figure 1) and next plug the frozen pre-trained encoder into a pre-trained LM for tuning videoto-text adapter layers (Step 2; Figure 1). Given this demonstrably scalable and simple-to-tune baseline, we systematically explore video-first models. Through our analysis, we are able to scale video-first encoders in a memory-efficient manner to longer sequences of frames, up to 4.3 minutes of video at 1 FPS.

We first explore video-first models on short-video benchmarks (MSR-VTT [71], VATEX [64], YouCook2 [86], ActivityNet [31]) and compare against the SoTA VideoCoCa model [75]. We demonstrate that vanilla joint space-time attention (without factorization) significantly improves performance over frame-level encodings on benchmarks with rich temporal dependencies (YouCook2, VATEX), at the cost of decreased spatial capabilities due to noisy video-text alignments (e.g., objects mentioned in the text but shown in previous or next video segments, text not grounded on visual content) in the pre-training datasets. Overall, our models are able to reach VideoCoCa performance, while requiring fewer parameters and lower frame resolution.

This performance gain incurs extra compute and memory costs that grow quadratically with the video length. To address this, we provide one of the first systematic analyses of the memory/accuracy pareto-front of popular memory-efficient methods; this includes factorized attention, parameter-efficient image-to-video adaptation, input masking, and multi-resolution patchification. Through this analysis, we find that among all these options, simple token masking (up to $75 \%$ ) during contrastive pre-training incurs only a $1 \%$ Recall @ 1 drop on zero-shot text-video retrieval, and no drop in zero-shot video captioning. At the same time, such high masking offers 2-3x memory savings and allows us to generalize to longer video contexts. The alternatives we explore (e.g., efficient backbone architectures, more sophisticated TubeViT-style patchification [52]), do not maintain the same robustness against noisy video inputs and present a $25 \%$ relative decrease in performance for text-video retrieval on challenging benchmarks (YouCook2, VATEX). Finally, although parameterefficient methods $[24,25]$ fail to adapt image encoders to video-first models without suffering performance drops, we find that they can adapt video models trained on short contexts (e.g., 16 second videos) to longer temporal horizons.

Based on the above learnings, we extend our best performing short-video encoder to longer contexts of 256 frames (4.3 minutes at 1 FPS). We use the full-length videos of HowTo100M [45] accompanied by LLM-generated summaries based on the ASR to further contrastively train our LONGVIVIT while masking $75 \%$ of the input video tokens and freezing most parameters of the encoder. LoNGVIVIT-to-text ( $\sim 1 \mathrm{~B}$ parameters) is able to outperform modular methods that use LLM assistance and PALI3 [11] for frame captioning on temporally rich benchmarks (YouCook2, EgoSchema). Even modular methods that consider frame selection (SeViLA [78]) or an oracle segmentation of the video for localizing and captioning key events (on YouCook2) cannot reach LONGVIVIT's performance. An interesting byproduct of our work is that we can glean which video-language benchmarks have strong temporal dependencies, and thus are suitable for testing long video models; we find that papers often use benchmarks in which short video or even blind models perform well $[6,44,71]$. In short, we provide the following contributions:

- We explore the memory/accuracy pareto-frontier of video-first vision-language models, and systematically evaluate many architectural, data, and training alternatives. In the end, we identify a simple recipe that enables scaling to 4.3 minutes at 1 FPS, many times longer than comparable video-language models $[1,75]$.
- We identify short and long video benchmarks with substantial temporal dependencies, for which we demonstrate that the traditional image-first, late-temporal fusion recipe is convincingly weaker than a video-first approach.
- Finally, we compare our long video models to a variety of strong baselines and show competitive performance with far fewer parameters; this includes baselines that use LLM-based aggregation over visual captions, and we quantitatively evaluate this common approach for the first time on standard video benchmarks.


## 2. Related Work

We base our recipes on $[1,33]$, which provide a strong two-step video-language recipe that leverages the strength of pre-trained LLMs and works at scale. Similar work at smaller scale has additionally included captioning losses [36, 80], more contrastive losses [12, 41, 46, 70], masked autoencoding $[18,19,22,38,43,58]$, and combinations thereof [16, 21, 26, 57, 62, 65, 76, 77, 79, 87]. This work largely focuses on image-text modeling and extends to $<30$ seconds via image-to-video transfer, selective finetuning, or temporal fusion of frame encodings $[1,75,77]$.

A volume of work focuses on video-first learning. This includes some of the very early work in image-to-video kernel inflation [7, 56, 59], transformer-based video architectures $[2,4,40]$, image-to-video parameter-efficient adaption $[8,39,49]$, and multiple spatiotemporal resolutions along different network paths [17, 42, 72, 74]. These have still only been demonstrated on short videos, so other works
have broached the challenge of temporal scalability: [27, 54, $68]$ propose alternative encoders, and $[30,51,63]$ propose more exotic attention mechanisms. TubeViT [52] proposes multi-granularity patchification. We systematically dissect what works and scales among some of these alternatives, electing options that enable us to re-use strong pre-trained models and use standard, more easily-tuned architectures.

Specifically in video-to-text generation, approaches that handle longer videos are very limited and mostly target images or short videos [18, 34, 65]. A dominant approach is to summarize frames and aggregate information via LLMs [34, 37, 66, 81]. To the best of our knowledge, we are the first to attempt to train large-scale videoto-text models on longer sequences of frames and directly test them against LLM-assisted modular methods on challenging temporal benchmarks $[44,86]$.

## 3. The Video-to-Text Architecture

We base our approach on the successful two-step recipe that combines pre-trained vision and language models [e.g., 1, 33, 76, 77] as shown in Figure 1: (1) we first follow a two-stage regime for contrastively pre-training the vision encoder, and then (2) fuse the frozen vision representations into a pre-trained, frozen LM.

### 3.1. Video-Language Contrastive Pre-training

Following common practice $[1,33]$, we use a dual visionlanguage architecture with a Noise Contrastive Estimation (NCE) loss $[20,48,69]$ to pre-train our vision encoder, similar to CLIP [53], ALIGN [29] and VideoCLIP [70]. Both encoders are transformers [61]: a BERT-medium (77M) or base (117M) language encoder and ViT-Base (86M parameters) or Large (307M parameters) vision encoder. On the language side, caption representations are computed by averaging across the corresponding token representations. On the vision side, video frames are patchified into a sequence of visual tokens, fed into a vision encoder, and then spatiotemporally mean pooled to produce a final video representation.

Most prior larger-scale video-language models use pretrained image encoders and patchify frames individually via 2D convolutions [e.g., 1, 70, 75]. Instead, we create spatiotemporal tubelets via 3D convolutions as done in recent vision-only models $[2,52,58]$. Using 3D tubelets instead of flat patches has the dual advantage of higher input compression and more explicit temporal contextualization; our early experiments yielded improved performance. The tubelet embedding sequence is then flattened, added to learnable positional embeddings, and fed into the vision encoder. The vision encoder uses spatio-temporal attention as in ViViT [2]: Joint space-time attention, which does not add any new parameters to vanilla image ViT [15], facilitating transfer between image and video models. Such seamless transfer enables us to run a first stage of mixed image and short video pre-training, and a second stage of adapting the encoder to longer videos.

Training a large-scale transformer-based video encoder can be challenging because self-attention across thousands of visual tokens is both compute and memory intensive. Memory bottlenecks a model in two ways: (1) limiting the number of frames, and (2) limiting the contrastive batch size during training, negatively impacting performance. To address (2), we start with an image encoder that is pre-trained with large batch sizes, and further tune it to the video domain, instead of jointly training on images and videos from scratch. For initializing the 3D convolution, we repeat the pre-trained weights across the temporal dimension similarly to $[2,7]$ (see Appendix A). During video-language pretraining, we maintain different embedding paths for images vs. videos: images are embedded with the original 2D convolution and videos with a separate 3D convolution, with no weight sharing.

### 3.2. Video-to-Text Tuning

After pre-training the vision encoder, we add a pre-trained LM to generate video descriptions and summaries. We follow previous work [e.g., 1, 76, 77] by keeping the visual encoder frozen and plugging it into a frozen pre-trained LM. We first temporally mean pool the video representations to keep a fixed number of visual tokens independently of the number of frames; interestingly, we find that temporal pooling critically improves training stability for longer videos, while managing to still encode temporal information implicitly in the fixed length tokens (carried over from temporal position embeddings that are added to the video input). We next use a randomly initialized Perceiver resampler [28] to project the representations to the LM embedding space. We add new randomly initialized cross-attention layers at each layer of the LM to ground generation on the visual content. We train these new layers and the Perceiver resampler with a standard auto-regressive video captioning loss: $-\log p\left(w_{t} \mid w<t ; \mathcal{V}\right)$, where $w_{t}$ is its $t^{t h}$ token, and $\mathcal{V}$ is the video representation. We provide more details in Appendix A.

## 4. Memory-Efficient Encoder Design Space

Device memory is a key bottleneck for video training with joint space-time attention. To overcome this, we explore four broad categories of solutions: (1) efficient attention, (2) parameter-efficient image-to-video adaptation, (3) input token masking, and (4) multi-resolution patchification.

1. Attention mechanism. Factorized attention $[2,4]$ separates the temporal and spatial dimensions over which selfattention is applied, reducing both memory and computational costs. However, this modification introduces a new
temporal block within each transformer layer making initialization and model tuning more challenging. In contrast to [2], that initializes the new blocks with zeroes, we find that we achieve best performance when initializing the temporal blocks with the same self-attention weights of ViT. However, we add a gating mechanism which acts as a residual connection between the self-attention blocks: $h=h+\tanh (\alpha) h_{\text {temporal }}$. Here, $\alpha$ is a trainable parameter initialized to zero, that helps maintain the capabilities of the original ViT during training.
2. Parameter-efficient adaptation. We explore using parameter-efficient methods from NLP [9] to adapt image encoders to video, while only tuning a small percentage of model parameters. Most competing approaches adapt image-based models by freezing an image backbone and adding late, trainable temporal-fusion layers [12, 75, 83]. In contrast, we explore ways to use pre-trained image encoders and adapt them to video-first architectures [8, 39, 49]. Inspired by the success of parameter-efficient adaptation in NLP [84], we consider using MLP Adapters [24] (after the feedforward block) and LoRA [25] (at the self-attention and feed-forward blocks) at every transformer layer (details in Appendix A). Additionally, we explore tuning only temporal self-attention blocks [8], effectively as adapter layers, in factorized attention. In all variants, we still tune the 3D patch convolution which is new in the video domain.
3. Token masking. Most existing work samples videos at a fixed frames per second (FPS) rate [e.g., 1, 2, 58, 78]. However, semantics required for many video-language tasks vary slowly in the temporal dimension [85] and videos present high degree of redundancy between consecutive frames $[21,58]$. We explore ways to sparsely sample the video input to reduce the number of input visual tokens. Specifically, we test random masking of input tubelet embeddings. Since consecutive frames are largely redundant, the same semantic signals could potentially be extracted even with high masking rates. For example, [58] and [21] mask up to $95 \%$ of the input video to reach optimal performance on the task of video-masked autoencoding. We demonstrate similar results in a video-language setting.
4. Multi-resolution patchification. Finally, we test a simple approach to reduce redundancy in videos via more coarse-grained patchification in the temporal or spatial dimension, as commonly done in multiple-view video models [17, 42, 74]. However, this decreases frame resolution, and may lose fine-grained information. As a result, we also experiment with TubeViT [52] variant that combines flat patches and tubelets of different granularity to mitigate information loss. Following [52], we use four different con- volution kernels that can encode either coarse-grained temporal or spatial information; details are in Appendix A.

## 5. Datasets and Benchmarks

For contrastive pre-training, we use: (1) a set of $27 \mathrm{M}$ videotext pairs (VTP) as described in [1], (2) HowTo100M [45] (HT100M; 100M instructional YouTube clips automatically aligned with ASR using their timestamps, called HowTo100M Clips), and (3) VideoCC3M [47] (3M videotext pairs based on Conceptual Captions [55]). Unfortunately, we find the text-video alignment in VideoCC3M to be of poor quality; instead, we use a modified variant where we generate pseudo-labeled captions of every video using PALI [11] (see Appendices B and C). To pre-train with longer videos, we use a long version of HowTo100M (referred to as HowTo100M Summary) consisting of (1) the full-length videos with an average duration of $6.5 \mathrm{~min}-$ utes and (2) their corresponding textual summaries that we generate by automatically cleaning and summarizing the ASR transcripts using an LLM [23]. We also include the same mixture of image datasets as in [1]. For video-to-text tuning, we use the same mixture of datasets but exclude HowTo100M Clips, since the noisy video-text alignments hurt the performance.

We report text-video retrieval and captioning results on short video benchmarks, with average video length $\leq 30$ seconds: MSR-VTT [71], YouCook2 [86], ActivityNet Captions [31], and VATEX [64]. To evaluate performance on longer videos, we consider video summarization on fulllength versions of YouCook2 and ActivityNet Captions, with a video duration of up to 5 minutes; we also evaluate for multiple-choice video question answering (QA) on EgoSchema [44], a dataset constructed to test long-context temporal understanding.

## 6. Experimental Results

In Section 6.1, we describe our results evaluating alternatives in memory-efficient video encoder design; options described in Section 4. For this analysis, we use ViT-B/BERTmedium, with training details in Appendix B and ablations on experimental design in Appendix C.

In Section 6.2, we combine our most competitive design choices from 6.1 and test our models on short and long video understanding benchmarks. We scale our best model variants to ViT-L/BERT-base with a 400M (or 1B) language decoder. We test our short video models on text-video retrieval and video captioning, and our long video models on video summarization and QA on 256 -frame videos.

In Section 6.3, we share our experience working across short and long video benchmarks [6, 14, 44, 64, 71], offering insights about which ones yield robust temporal signal.
![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-05.jpg?height=452&width=1694&top_left_y=224&top_left_x=172)

Figure 2. Trade-offs between performance ( $\%$ text-to-video Recall @ 1; y axis) and train-time memory consumption (x axis) for different backbones (joint space-time (JST), factorized space-time (FST), and drame-level encodings) with random input masking ( $0 \%$ up to $75 \%$ ) or parameter-efficient methods for training (Adapters, LoRA, factorized temporal (FST) adaptation; lower opacity).

|  | MSR-VTT |  | VATEX |  | YC2 |  | AN |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | T2V | V2T | T2V | V2T | T2V | V2T | $\mathrm{T} 2 \mathrm{~V}$ | v |
|  |  | 38.1 | 23.8 | 26.3 | 12.3 | 13.6 | 6.7 |  |
|  |  | 36.9 | 25.3 |  | 11.6 | 12.7 | 6 |  |
|  | 39.3 | 34.8 | 24.8 | 25.0 | 9.1 | 7.9 | $\mathbf{0}$ |  |
| t-pool 1 | 38.4 | 37.5 | 21.9 | 26.1 | 9.0 | 8.9 | 6.1 |  |

Table 1. Text-video retrieval results ( $\%$ Recall@1) when considering different visual backbones.

### 6.1. Exploration of Memory-Efficient Designs

We explore memory-efficient methods to train video-first encoders as described in Section 4. We first consider short video inputs of 16 frames at 1 FPS and report peak traintime memory consumption vs. performance on text-video retrieval on short video benchmarks [5]. Then, we test whether our main findings hold for longer inputs (128+ frames) on video summarization on full-length YouCook2.

Base architectures. We explore the memory/accuracy trade-off of different visual backbones in Table 1: ViViT with joint space-time attention (i.e., Joint ST-ViViT), ViViT with factorized attention (i.e., Factorized ST-ViViT) [2], and frame-level (ViT-based) image encodings with average or attentional pooling ('att-pool') [1, 75]. Different methods perform similarly, especially on MSR-VTT and ActivityNet (AN). Interestingly, attentional pooling on top of framelevel encodings does not improve performance. ViViT with either joint or factorized attention performs best and presents higher gains for YouCook2 (YC2), the more temporally challenging benchmark [6.3]. In contrast to prior work $[e . g ., 12,75]$ which tests frozen image-to-video transfer and claims joint attention to be inferior, we find it to be competitive in this fully fine-tuned setting.

Architectures and token masking. We now test robustness of backbones when masking part of the input tubelets
(0-75\%). We report Recall @ 1 on text-to-video retrieval for YouCook2 and VATEX ${ }^{1}$ per backbone for different masking ratios in Figure 2. Joint space-time attention (JST) is robust against noise from masking up to $75 \%$ during pre-training. The same does not hold for frame-level encodings and factorized attention (FST), where performance drops consistently as we increase masking. We conclude that JST can better handle noisy inputs and use it in further exploration.

Parameter-efficient adaptation. We next report performance of parameter-efficient image-to-video adaptation in Figure 2. We consider (1) JST with (a) MLP Adapters at every layer of the encoder, (b) LoRA with rank decomposition matrices in the self-attention and feed-forward transformer blocks, and (2) factorized temporal adaptation where we tune the temporal self-attention. No adaptation method can reach the memory savings provided by high input masking, since we tune parameters depthwise and gradient computation still requires backpropagation through the model. At the same time, we see significant performance drop, suggesting that adaptation of spatial-only models to the temporal dimension cannot be sufficiently addressed in semifrozen fashion. Comparing parameter-efficient methods, we find MLP Adapters to be more competitive than LoRA, which is now canonical for LLMs. We hypothesize that LoRA is successful for tuning very small portions of the network and performing "easier" in-modality transfer.

Adaptation at scale. We next scale from ViT-B/86M to ViT-L/307M in Figure 4 and test whether observations hold with different model scales. We present the $\%$ memory increase from base to large (left bar set) and \% performance decrease of each method at each scale ${ }^{2}$. Joint ST exhibits a similar memory increase to frame-level, while leading to[^1]![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-06.jpg?height=312&width=1766&top_left_y=267&top_left_x=148)

Figure 3. Trade-offs between performance (text-to-video Recall @ 1; y axis) and memory consumption (x axis) for input sampling methods: (1) high input masking ratios ( $0 \%$ to $75 \%$ ) with joint space-time attention, (2) coarse-grained temporal (Coarse temp) and/or spatial (Coarse space) patchification with a fixed kernel and TubeViT which samples parts of the video with multiple 3D kernels of different granularity.

![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-06.jpg?height=548&width=837&top_left_y=778&top_left_x=164)

Figure 4. Difference (\%) in memory consumption for different model scales: (ViT-B vs ViT-L). We also report performance drop of efficient methods presented in Figure 2 in comparison with the vanilla approach (i.e., no input masking and full fine-tuning) at different model scales to test whether behavior is similar.

smaller accuracy drops, whereas factorized ST presents significant memory overhead with model scale due to the extra temporal self-attention blocks. For this reason, we exclude factorized ST from further experimentation. Finally, parameter-efficient methods are unable to achieve performance similar to their fully fine-tuned counterparts in both model scales, although their memory requirements scale better with model size.

Multi-resolution patchification. Given the outsized memory impact of input token count in Figure 3, we additionally analyze: (1) coarse-grained patchification in the temporal (convolution over 4 instead of 2 frames) and/or spatial (convolution over $32 \times 32$ instead of $16 \times 16$ pixel spaces) dimension, and (2) the TubeViT [52] approach of multiple tube kernels of different spatiotemporal size and strides. For all benchmarks, masking the input at high ratios while maintaining a fine granularity of tubelets decreases performance significantly less than other input processing methods. Temporal coarse-grained patchification negatively affects benchmarks with richer temporal dependencies (i.e., YouCook2, VATEX) more than spatial. The opposite trend holds for datasets depending on spatial understanding (i.e., MSR-VTT, ActivityNet Captions ${ }^{3}$ ). TubeViT acts as the middle ground between the two by employing multiple kernels, with some performance degradation across all benchmarks. However, it is not able to alleviate the negative effects caused by considering coarsergrained information and presents higher memory requirements due to the multiple convolutions. Overall, we find that high masking with Joint ST and small tubelets yields the strongest memory/performance curves.

Scaling to longer videos. We now test the best methods from Figure 3 on 128 input frames ( $32.7 \mathrm{k}$ visual tokens). We select methods that are within a memory budget (red vertical lines) and would fit on a 16GB device when expanded to long videos ( $128+$ frames). We contrastively finetune [3.1] our best performing video model (i.e., Joint ST referred to as SHORTVIVIT) on sequences of 128 frames on HowTo100M Summary [5], as detailed in Appendix B. We refer to this model as LONGVIVIT. Finally, we fine-tune LONGVIVIT for text generation (Section 3.2) on the fulllength YouCook2, and report Rouge-L in Figure 5, measuring memory consumption during both long-context contrastive ( $x$-axis) and video-to-text ( $y$-axis) tuning.

Validating our previous results, IMAGEVIT (frame-level encodings) trained on longer videos with $75 \%$ masking ${ }^{4}$ significantly under-performs video-first models ( $10 \mathrm{R}-\mathrm{L}$ drop). ShortViViT without further HT100M Summary training performs better than IMAGEVIT, but cannot match models adapted to longer videos. LONGVIVIT improves performance by 1.8 Rouge-L points over SHORTVIVIT. Comparing input masking with coarser-grained patchification ${ }^{5}$ provides similar insights to the previous paragraph.

Finally, we test MLP Adapters [24] for tuning SHORTVIVIT to longer videos and observe no performance drop[^2]

![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-07.jpg?height=374&width=1756&top_left_y=233&top_left_x=152)

Figure 5. Scaling memory-efficient methods to more frames (i.e., 128 frames) for ViViT-B and variants. We measure performance for video-to-text summarization on the full-length YouCook2 videos via Rouge-L (color-coded) while keeping track of memory consumption during short-to-long video contrastive tuning ( $x$-axis) and video-to-text tuning ( $y$-axis).

|  | MSR-VTT <br> Zero-shot |  | $\frac{\mathrm{FT}}{\mathrm{C} 1}$ | VATEX <br> ro-shot |  | $\frac{\mathrm{FT}}{\mathrm{Cl} 1}$ | YouCook2 <br> Zero-shot |  | FT <br> C1 | ActivityNet <br> Zero-shot |  | $\frac{\mathrm{FT}}{\mathrm{C} 1}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\mathrm{T} 2 \mathrm{~V} / \mathrm{V} 2 \mathrm{~T}$ | $\mathrm{C} 1 / \mathrm{C} 2$ |  | $\mathrm{~T} 2 \mathrm{~V} / \mathrm{V} 2 \mathrm{~T}$ | $\mathrm{C} 1 / \mathrm{C} 2$ |  | $\mathrm{~T} 2 \mathrm{~V} / \mathrm{V} 2 \mathrm{~T}$ | $\mathrm{C} 1 / \mathrm{C} 2$ |  | T2V/V2T | $\mathrm{C} 1 / \mathrm{C} 2$ |  |
| IMAGEVIT-L | $30.9 / 41.6$ | 24.6/25.1 | 63.6 | 36.2/42.9 | 37.9/39.4 | $61.1 \quad-\quad$ | 18.2/16.8 | 14.5/16.5 | 95.9 | 20.6/18.2 | 16.3/17.7 | 41.1 |
| SHORTVIVIT-L | $31.9 / 38.9$ | $32.7 / 32.9$ | 63.1 | 37.8/42.8 | 43.6/43.0 | 67.5 | 20.4/20.5 | 21.0/22.1 | 131.9 | 21.3/18.9 | 25.2/26.1 | 44.8 |
| EffSHORTVIVIT-L | $29.9 / 38.3$ | 33.8/33.9 | 63.8 | 34.4/42.7 | 41.3/42.7 | 64.7 | $20.5 / 20.3$ | 21.1/21.7 | 127.1 | 20.1/17.7 | 27.0/26.5 | 41.1 |
| VideoCoCa-L [75] | 33.3/- | 24.3 | - | - | - | - | 18.9/- | 20.7 | - | $31.5 * /-$ | 17.4 | - |
| VideoCoCa-2.1B | $\underline{34.3 / 64.7}$ | 27.1 | 73.2 | $\underline{53.2 / 73.6}$ | 22.8 | 77.8 | 20.3/- | 34.3 | 128.0 | $34.5 * / 33.0^{*}$ | 19.3 | 39.3 |
| Flamingo-3B [1] | - | - | - | - | 40.1 | - | - | 55.8 | - | - | - | - |

Table 2. We present three model variants: ImaGEVit-L, that uses frame-level encodings with a late temporal fusion trained on images and videos, SHORTVIVIT-L, our best performing video-first model with joint space-time attention, and Efficient SHORTVIVIT-L (EffSHORTVIVIT-L) where we apply $75 \%$ train-time masking for $3 x$ memory savings. We also report performance for SoTA image-first models: VideoCoCa-L and Flamingo-3B, although they are bigger and not directly comparable. We report Recall @ 1 for zero-shot textto-video (T2V) and video-to-text (V2T) retrieval, and CIDEr for zero-shot and fine-tuned (FT) captioning when considering a 400M (C1) or 1B (C2) frozen LM for generation. ActivityNet retrieval results marked with * are not directly comparable, as these models uniformly sample frames, whereas we use the first frames of the long video with a fixed FPS of 1 to match experimental settings across benchmarks.

in comparison with full fine-tuning. This provides further evidence that parameter-efficient methods can be used for "easier transfers" but not temporal adaptation of spatialonly models. One downside of MLP Adapters is that it increases parameter count during video-to-text tuning ( $y$-axis in Figure 5). Thus, we also experiment with contrastively tuning only the last four layers of the model. With this, we observe a further $3 x$ decrease in memory, since we tune the network widthwise and excise early layer gradient computation. At the same time, there is no memory increase for video-to-text and no performance degradation. We conclude that this combination (high input masking and tuning the last layers) is an effective setting for longer video adaptation. Given the observed robustness to masking, to further decrease video-to-text memory, we also mask $30 \%$ of the input video during training and inference without observing any drop in summarization performance (see Appendix C).

### 6.2. Main Results

Short video benchmarks. We present our main results on short video benchmarks in Table 2. We use ViT-L with BERT-base for contrastive pre-training (Section 3.1) and a
400M frozen LM for video-to-text tuning (Section 3.2). Our entire video-to-text model accounts for $\sim 900 \mathrm{M}$ parameters, although we additionally test scaling the frozen LM to 1B parameters ( $\sim 1.5$ B total count). We report Recall@1 for zero-shot text-video retrieval and CIDEr for zero-shot and fine-tuned video captioning. We consider three model variants: frame-level encodings ImageViT, ShORTViViT, and ShortViVit with $75 \%$ masking that uses 2-3x less memory (referred to as Efficient SHORTVIVIT). We also report results for VideoCoCa [75] and Flamingo [1] ${ }^{6}$.

Our results remain consistent with our earlier observations. Contextualizing only intra-frame dependencies coupled with late temporal fusion (IMAGEVIT) leads to inferior performance for retrieval and captioning on benchmarks with richer temporal dependencies (YouCook2, VATEX) but performs better on retrieval on MSR-VTT which relies on spatial understanding. Video-first architectures further tuned on video datasets (substantially noisier than curated image ones) improve temporal capabilities at the expense[^3]of spatial. For Efficient SHORTVIViT, we find that masking $75 \%$ of the input video causes a performance drop: an average of $1 \%$ absolute difference on zero-shot retrieval and no significant difference on zero-shot captioning across all benchmarks. The efficient model still performs similarly or better than IMAGEVIT, especially on captioning and temporally rich benchmarks (e.g., YouCook2, VATEX), while consuming significantly less memory. Finally, when scaling the frozen LM component from $400 \mathrm{M}$ to $1 \mathrm{~B}(\mathrm{C} 1 \rightarrow \mathrm{C} 2)$ for zero-shot video-to-text generation, we observe moderate improvements across benchmarks and variants.

We compare our results against large image-based models with SoTA performance on video benchmarks (second block of Table 2). Although results are not directly comparable due to different experimental settings, we are competitive and achieve even better results for temporally rich benchmarks (i.e., YouCook2) on text-video retrieval for models of similar parameter count ${ }^{7}$. Moreover, our models significantly outperform VideoCoCa on most video captioning benchmarks even when considering their much larger versions in the zero-shot setting. Finally, when fine-tuning our video-to-text models with the 400M LM, we are again able to match and surpass the performance of the larger VideoCoCa-2.1B in two out of four benchmarks.

Long video understanding. We further tune LoNGVIVIT-L on 256 -frame HT100M Summary videos and evaluate zero-shot and fine-tuned summarization (YouCook2, ActivityNet) and QA (EgoSchema released subset); this is shown in Table 3. Although we focus on long videos at 1 FPS, we additionally report results on Perception Test [50] in Appendix D, where videos are short but contain rich temporal dependencies and can benefit from higher FPS.

We consider two families of models. 1. Models that take as input 256 frames (first block of Table 3): IMAGEVIT and SHORTVIVIT pre-trained on 16 -frame clips, and LoNGVIVIT further trained on 256 -frame clips. 2. Modular approaches from prior work (second block of Table 3): (a) SeViLA Localizer [78] for localizing important frames in the long video given a textual query which are then fed into SHORTVIVIT for performing the task ${ }^{8}$, and (b) the popular paradigm of captioning video segments or frames and using an LLM to aggregate information and form coherent summaries or answer questions $[34,37,81]$. We try this latter approach with our short video models (IMAGEVIT, SHORTVIVIT), generating captions over 16 -second video segments and then feeding the captions to the September[^4]

|  | Zero-shot |  |  |  | Fine-tuned |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | AN | YC2 | ES |  | AN | YC2 |
| Inference with 256 frames |  |  |  |  |  |  |
| IMAGEVIT | 14.4 | 4.6 | 40.8 |  | 23.8 | 29.4 |
| SHORTVIVIT | 15.4 | 7.0 | 47.9 | $\mathbf{2 4 . 3}$ | 29.5 |  |
| LONGVIVIT | 15.2 | $\mathbf{2 0 . 3}$ | $\mathbf{5 6 . 8}$ | 24.0 | $\mathbf{3 0 . 6}$ |  |
| Modular approaches with | 16-frame video | models |  |  |  |  |
| SeViLA-to-SHORTVIVIT | 16.2 | 4.2 | 49.6 | $\mathbf{2 4 . 4}$ | 28.3 |  |
| IMAGEVIT-to-Bard | 18.1 | 15.8 | 35.0 | 22.9 | 19.1 |  |
| $\quad$ + oracle segments | 16.3 | 16.2 | - | 22.7 | 22.1 |  |
| SHORTVIVIT-to-Bard | 19.3 | 18.1 | 42.0 | 22.7 | 20.8 |  |
| $\quad$ + oracle segments | 18.3 | 18.2 | - | 22.7 | 24.7 |  |
| PALI [11] 5B-to-Bard | $\mathbf{2 2 . 0}$ | 19.9 | 44.8 | - | - |  |
| Blind Bard | - | - | 27.0 | - | - |  |
| SoTA [73] | - | - | - | 36.9 | 34.6 |  |

Table 3. Results on long video-to-text benchmarks. We report Rouge-L for zero-shot and fine-tuned video summarization on ActivityNet Captions (AN) and YouCook2 (YC2) and zero-shot accuracy (\%) for multiple choice QA on EgoSchema (ES).

2023 release of Bard, a much larger LLM than the ones used in previous results. We caption clips using uniform video segmentation (every 16 seconds) or an oracle segmentation when available (i.e., we consider ground-truth start and end timestamps for different events within ActivityNet and YouCook2 videos). We also test substituting our small video models with PALI-3 (5B parameters) for frame captioning ${ }^{9}$. Finally, we reference the SoTA finetuned performance on ActivityNet and YouCook2, when using specialized models with pre-computed features by multiple networks, object detectors, and domain-specific vocabulary $[73]$.

Looking through Table 3, we find that on ActivityNet, which contains less temporal dependencies [6.3], modular approaches via frame selection or LLM-based aggregation of information (second block) perform well. Frame captioning via PALI combined with the power of LLMs is enough for the task in a zero-shot setting. For fine-tuned models, feeding either the long input or selected frames into SHORTVIVIT perform better than utilizing Bard. On ActivityNet, we see no benefit from training further on longer videos.

In contrast, we find that short video and modular models are insufficient for addressing video tasks with longer-range temporal dependencies (YouCook2, EgoSchema). Adapting SHORTVIVIT to longer contexts (LONGVIVIT) significantly improves performance and achieves the best scores across all comparison approaches. Using Bard as an information aggregator over individual clip captions cannot achieve competitive performance, even when considering an oracle video segmentation for YouCook2 (Lines 3 and 5 in the second block of Table 3). Surprisingly, even using a[^5]

![](https://cdn.mathpix.com/cropped/2024_05_26_26b35f959434b3cbacfbg-09.jpg?height=325&width=788&top_left_y=236&top_left_x=186)

Figure 6. Performance difference (\%) per benchmark when we remove (1) video or (2) image data from the training mixture.

much larger and more powerful image-based model (PALI) cannot reach LoNGVIVIT on YouCook2 and EgoSchema. Interestingly, selecting 16 key frames and feeding them into SHORTVIVIT also outperforms Bard-based methods on EgoSchema and fine-tuned YouCook2. This suggests there can be temporal dependencies in long videos that cannot be resolved even with an optimal event segmentation for the video, or be aggregated by LLMs given inprecise visual information. On such benchmarks, LoNGVIVIT demonstrates strong performance even without LLM assistance.

### 6.3. Brief Notes on Video Evaluations

We briefly describe some of our findings on video evaluations. Firstly, we find that blind Bard is able to achieve SoTA results on the full set of EgoSchema (no visual input; $33.9 \%$ accuracy vs. $32.1 \%$ for the best model in [44]). Adding visual information from PALI into Bard increases performance to just $39.2 \%$. However, on EgoSchema's released subset, performance of blind Bard is $27 \%$, which is much lower than PALI-to-Bard (44.8\%), suggesting that the subset contains questions that rely more on visual grounding than pure language reasoning, so we report numbers on the subset in Table 3 and on the full set in Appendix D.

Figure 6 details a simple ablation across other video benchmarks to quantify temporal richness. We test removing either video or image data from the training mix and measure the effect on performance (video-to-text Recall@1). We see a dramatic performance drop when removing video data for YouCook2 and VATEX (up to 75\%). ActivityNet and MSRVTT suffer more from the absence of image data, whereas non-video training influences performance in lesser degree (as little as $18 \%$ for MSR-VTT). We believe there's room for more fine-grained, temporalfocused video-language benchmarks in the community.

## 7. Conclusions

In short, we systematically analyze memory-efficient methods to scale video-first architectures to longer sequences of frames and demonstrate that just masking high percentages of the video $(\leq 75 \%)$ yields competitive results on long video-language tasks. Such masking shows a very small performance drop on short videos, provides 2-3x memory savings and allows scaling up to 4.3 minutes at 1 FPS (LONGVIVIT) when freezing part of the short video network in our two-stage training. LONGVIVIT outperforms modular approaches with LLM assistance on video summarization and QA on benchmarks with richer temporal dependencies (YouCook2, EgoSchema). We overall demonstrate that encoding longer-range visual dependencies can make a difference in downstream performance and corrects mistakes that LLMs are unable to rectify.

## Acknowledgements

We thank Chuhan Zhang for her thoughtful feedback on early drafts of the work; Chris Dyer, Joao Carreira, and Gabriel Brostow for their support, feedback, and guidance through the project; and Lucia Lopez for her valuable help in generating synthetic video captions.

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716-23736, 2022. 1, 2, 3, 4, 5, 7, 14, 16, 17, 19

[2] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836-6846, 2021. 2, 3, 4, 5, 14, 17

[3] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, page 4, 2021. 17

[4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, page 4, 2021. 2, 3

[5] Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the" video" in video-language understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2917-2927, 2022. 2

[6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961-970, 2015. 2, 4

[7] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299-6308, 2017. 2, 3,14

[8] Dongsheng Chen, Chaofan Tao, Lu Hou, Lifeng Shang, Xin Jiang, and Qun Liu. Litevl: Efficient video-language learning with enhanced spatial-temporal modeling. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7985-7997, 2022. 2, 4

[9] Jiaao Chen, Aston Zhang, Xingjian Shi, Mu Li, Alex Smola, and Diyi Yang. Parameter-efficient fine-tuning design spaces. arXiv preprint arXiv:2301.01821, 2023. 4

[10] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 17

[11] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, et al. Pali-3 vision language models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023. 2, 4, 8, 15, 19

[12] Feng Cheng, Xizi Wang, Jie Lei, David Crandall, Mohit Bansal, and Gedas Bertasius. Vindlu: A recipe for effective video-and-language pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10739-10750, 2023. 2, 4, 5

[13] R Christoph and Feichtenhofer Axel Pinz. Spatiotemporal residual networks for video action recognition. Advances in neural information processing systems, 2, 2016. 14
[14] Pradipto Das, Chenliang Xu, Richard F Doell, and Jason J Corso. A thousand frames in just a few words: Lingual description of videos through latent topics and sparse object stitching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2634-2641, 2013. 4

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth $16 \times 16$ words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020. 3

[16] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 1, 2

[17] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202-6211, 2019. 2, 4

[18] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: End-to-end video-language transformers with masked visual-token modeling. arXiv preprint arXiv:2111.12681, 2021. 2, 3, 17, 18

[19] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. An empirical study of end-to-end video-language transformers with masked visual modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2289822909, 2023. 2

[20] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 297-304. JMLR Workshop and Conference Proceedings, 2010. 3

[21] Tengda Han, Weidi Xie, and Andrew Zisserman. Turbo training with token dropout. arXiv preprint arXiv:2210.04889, 2022. 2, 4

[22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll' ar, and Ross B Girshick. Masked autoencoders are scalable vision learners. 2022 ieee. In CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15979$15988,2021.2$

[23] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 4, 15

[24] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790-2799. PMLR, 2019. 2, 4, 6, 14

[25] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-
rank adaptation of large language models. In International Conference on Learning Representations, 2021. 2, 4, 14

[26] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023. 2

[27] Md Mohaiminul Islam and Gedas Bertasius. Long movie clip classification with state-space video models. In European Conference on Computer Vision, pages 87-104. Springer, 2022. 1,3

[28] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651-4664. PMLR, 2021. 3

[29] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904-4916. PMLR, 2021. 3, 16, 17

[30] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020. 3

[31] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706-715, 2017. 2, 4

[32] Jie Lei, Tamara L Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. arXiv preprint arXiv:2206.03428, 2022. 2

[33] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 2, 3

[34] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3, 8

[35] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: Unifying video-language understanding as masked language modeling. arXiv preprint arXiv:2206.07160, 2022. 17, 18

[36] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: Unifying videolanguage understanding as masked language modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23119-23129, 2023. 2

[37] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, Ce Liu, and Lijuan Wang. $\mathrm{Mm}-\mathrm{vid}$ : Advancing video understanding with gpt-4v(ision), 2023. 3, 8

[38] Yuanze Lin, Chen Wei, Huiyu Wang, Alan Yuille, and Cihang Xie. Smaug: Sparse masked autoencoder for efficient video-language pre-training. In Proceedings of the
IEEE/CVF International Conference on Computer Vision, pages 2459-2469, 2023. 2

[39] Ruyang Liu, Jingjia Huang, Ge Li, Jiashi Feng, Xinglong Wu, and Thomas H Li. Revisiting temporal modeling for clip-based image-to-video knowledge transferring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6555-6564, 2023. 2, 4

[40] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202-3211, 2022. 2

[41] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293-304, 2022. 2, 17

[42] Chuofan Ma, Qiushan Guo, Yi Jiang, Ping Luo, Zehuan Yuan, and Xiaojuan Qi. Rethinking resolution in the context of efficient video recognition. Advances in Neural Information Processing Systems, 35:37865-37877, 2022. 2, 4

[43] Yue Ma, Tianyu Yang, Yin Shan, and Xiu Li. Simvtp: Simple video text pre-training with masked autoencoders. arXiv preprint arXiv:2212.03490, 2022. 2

[44] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. arXiv preprint arXiv:2308.09126, 2023. 2, 3, 4, 9, 16, 19

[45] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630-2640, 2019. 2, 4, 16

[46] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 98799889, 2020. 2, 17

[47] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In Computer Vision-ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XIV, pages 407-426. Springer, 2022. 4, 16

[48] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 3

[49] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. St-adapter: Parameter-efficient image-to-video transfer learning. Advances in Neural Information Processing Systems, 35:26462-26477, 2022. 2, 4

[50] Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, et al. Perception test: A diagnostic benchmark for multimodal video models. arXiv preprint arXiv:2305.13786, 2023. 8,19

[51] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023. 3

[52] AJ Piergiovanni, Weicheng Kuo, and Anelia Angelova. Rethinking video vits: Sparse video tubes for joint image and video learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22142224, 2023. 2, 3, 4, 6, 14

[53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR, 2021. 3, 17

[54] Michael S Ryoo, Keerthana Gopalakrishnan, Kumara Kahatapitiya, Ted Xiao, Kanishka Rao, Austin Stone, Yao Lu, Julian Ibarz, and Anurag Arnab. Token turing machines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19070-19081, 2023. 3

[55] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556-2565, 2018. 4

[56] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. $A d$ vances in neural information processing systems, 27, 2014. 2

[57] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15638-15650, 2022. 2

[58] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In Advances in Neural Information Processing Systems, 2022. 2, 3, 4

[59] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 6450-6459, 2018. 2

[60] Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. Image captioners are scalable vision learners too. arXiv preprint arXiv:2306.07915, 2023. 17

[61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3

[62] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. Omnivl: One foundation model for image-language and video-language tasks. Advances in neural information processing systems, 35:5696-5710, 2022. 2
[63] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020. 3

[64] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4581-4591, 2019. 2, 4

[65] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2, 3, 16, 19

[66] Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, et al. Language models with image descriptors are strong few-shot video-language learners. Advances in Neural Information Processing Systems, 35: 8483-8497, 2022. 3

[67] Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1884-1894, 2021. 1

[68] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13587-13597, 2022. 1, 3

[69] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3733-3742, 2018. 3

[70] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6787-6800, 2021. 2, 3, 17

[71] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288-5296, 2016. 2, 4

[72] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036-5045, 2022. 2

[73] Kashu Yamazaki, Khoa Vo, Quang Sang Truong, Bhiksha Raj, and Ngan Le. Vltint: visual-linguistic transformer-intransformer for coherent video paragraph captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3081-3090, 2023. 8

[74] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid. Multiview
transformers for video recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3333-3343, 2022. 2, 4, 17

[75] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Video-text modeling with zero-shot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022. 1, 2, 3, 4, 5, 7, 8, 15, 17, 18

[76] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. Advances in Neural Information Processing Systems, 35:124-141, 2022. 1, 2, 3

[77] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 1, 2, 3

[78] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. arXiv preprint arXiv:2305.06988, 2023. 2, 4, 8, 19

[79] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 2

[80] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing Systems, 34:23634-23651, 2021. $2,17,18$

[81] Andy Zeng, Maria Attarian, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, et al. Socratic models: Composing zero-shot multimodal reasoning with language. In The Eleventh International Conference on Learning Representations, 2022. 3, 8

[82] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104-12113, 2022. 16

[83] Bowen Zhang, Xiaojie Jin, Weibo Gong, Kai Xu, Zhao Zhang, Peng Wang, Xiaohui Shen, and Jiashi Feng. Multimodal video adapter for parameter efficient video text retrieval. arXiv preprint arXiv:2301.07868, 2023. 4

[84] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient finetuning. arXiv preprint arXiv:2303.10512, 2023. 4

[85] Zhang Zhang and Dacheng Tao. Slow feature analysis for human action recognition. IEEE transactions on pattern analysis and machine intelligence, 34(3):436-450, 2012. 4

[86] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 2, 3, 4

[87] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2
</end of paper 0>


<paper 1>
# A Simple LLM Framework for Long-Range Video Question-Answering 

Ce Zhang* Taixi Lu* Md Mohaiminul Islam Ziyang Wang Shoubin Yu<br>Mohit Bansal Gedas Bertasius<br>Department of Computer Science, UNC Chapel Hill<br>\{cezhang, mmiemon, ziyangw, shoubin, mbansal, gedas\}@cs.unc.edu, taixi@email.unc.edu


#### Abstract

We present LLoVi, a simple yet effective Language-based Long-range Video questionanswering (LVQA) framework. Our method decomposes short and long-range modeling aspects of LVQA into two stages. First, we use a short-term visual captioner to generate textual descriptions of short video clips ( $0.5-$ $8 \mathrm{~s}$ in length) densely sampled from a long input video. Afterward, an LLM aggregates the densely extracted short-term captions to answer a given question. Furthermore, we propose a novel multi-round summarization prompt that asks the LLM first to summarize the noisy short-term visual captions and then answer a given input question. To analyze what makes our simple framework so effective, we thoroughly evaluate various components of our framework. Our empirical analysis reveals that the choice of the visual captioner and LLM is critical for good LVQA performance. The proposed multi-round summarization prompt also leads to a significant LVQA performance boost. Our method achieves the best-reported results on the EgoSchema dataset, best known for very long-form video question-answering. LLoVi also outperforms the previous state-ofthe-art by $4.1 \%$ and $3.1 \%$ on NExT-QA and IntentQA. Finally, we extend LLoVi to grounded VideoQA which requires both QA and temporal localization, and show that it outperforms all prior methods on NExT-GQA. Our code is available at: $h t t p s: / /$ github.com/CeeZh/ LLoVi.


## 1 Introduction

Recent years have witnessed remarkable progress in short video understanding (5-15s in length) (Wang et al., 2022a; Ye et al., 2023; Fu et al., 2021; Yang et al., 2022a; Wang et al., $2023 \mathrm{~g})$. However, extending these models to long videos (e.g., several minutes or hours in length) is not trivial due to the need for sophisticated[^0]

![](https://cdn.mathpix.com/cropped/2024_05_26_04df3d9f395908fb2e5dg-01.jpg?height=154&width=741&top_left_y=734&top_left_x=1063)

Question: What were the key steps the camera wearer took in cleaning the dog mat from start to finish?

![](https://cdn.mathpix.com/cropped/2024_05_26_04df3d9f395908fb2e5dg-01.jpg?height=252&width=741&top_left_y=982&top_left_x=1066)

Figure 1: Comparison between LLoVi (ours) and the recent FrozenBiLM (Yang et al., 2022a) video QA method. Like most prior methods, FrozenBiLM is best suited for short-range video understanding. Thus, as illustrated in the figure, it fails to answer a question that requires reasoning about complex human activities in a long video. In comparison, our method effectively reasons over long temporal extents and produces a correct answer.

long-range temporal reasoning capabilities. Most existing long-range video models rely on costly and complex long-range temporal modeling schemes, which include memory queues (Wu et al., 2022; Chen et al., 2020; Lee et al., 2021, 2018), long-range feature banks (Wu et al., 2019; Cheng and Bertasius, 2022; Zhang et al., 2021), space-time graphs (Hussein et al., 2019b; Wang et al., 2021), state-space layers (Islam and Bertasius, 2022; Islam et al., 2023; Wang et al., 2023a) and other complex long-range modeling modules (Hussein et al., 2019a; Bertasius et al., 2021; Yang et al., 2023).

Recently, Large Language Models (LLMs) have shown impressive capability for long-range reasoning on a wide range of tasks such as document understanding (Sun et al., 2023; Wang et al., 2023e; Gur et al., 2023) and long-horizon planning (Liu et al., 2023a; Hao et al., 2023; Song et al., 2023a). Motivated by these results in the natural language and decision-making domain, we explore using

LLMs for long-range video question answering (LVQA). Specifically, we propose LLoVi, a simple yet effective language-based framework for long-range video understanding. Unlike prior longrange video models, our approach does not require specialized long-range video modules (e.g., memory queues, state-space layers, etc.) but instead uses a short-term visual captioner coupled with an LLM, thus exploiting the long-range temporal reasoning ability of LLMs. Our simple two-stage framework tackles the LVQA task by decomposing it into short and long-range modeling subproblems:

1. First, given a long video input, we segment it into multiple short clips and convert them into short textual descriptions using a pretrained frame/clip-level visual captioner (e.g., BLIP2 (Li et al., 2023c), LaViLa (Zhao et al., 2023), LLaVa (Liu et al., 2023b)).
2. Afterwards, we concatenate the temporally ordered captions from Step 1 and feed them into an LLM (e.g., GPT-3.5, GPT-4, LLaMA) to perform long-range reasoning for LVQA.

To further enhance the effectiveness of our framework, we also introduce a novel multi-round summarization prompt that asks the LLM first to summarize the short-term visual captions and then answer a given question based on the LLMgenerated video summary. Since the generated captions may be noisy or redundant, such a summarization scheme enables filtering out potentially distracting/irrelevant information and eliminating redundant sentences, which significantly improves the reasoning ability of the LLM for LVQA.

We also conduct an empirical study to investigate the factors behind our framework's success. Specifically, we study (i) the selection of a visual captioner, (ii) the choice of an LLM, (iii) the LLM prompt design, (iv) few-shot in-context learning, (v) optimal video processing configurations (i.e., clip length, sampling rate, etc.), and (vi) the generalization of our framework to other datasets and tasks. Our key empirical findings include:

- The multi-round summarization prompt leads to the most significant boost in performance $\mathbf{( + 5 . 8 \% )}$ ) among the prompts we have tried (e.g., zero-shot CoT, Self-Consistency).
- GPT-4 as an LLM provides the best performance, while GPT-3.5 provides the best trade-off between the accuracy and the cost.
- LaViLa (Zhao et al., 2023) as a visual captioner produces best results $\mathbf{( 5 1 . 8 \% )}$ ) followed by BLIP-2 (Li et al., 2023c) (46.7\%) and EgoVLP (Qinghong Lin et al., 2022) (46.6\%).
- Few-shot in-context learning leads to a large improvement on both the variant of our model with a standard prompt $(\mathbf{+ 4 . 7 \%})$ and our bestperforming variant with our proposed multiround summarization prompt $(\mathbf{+ 4 . 1 \%})$.
- Densely Extracting visual captions from consecutive 1-second video clips of the long video input leads to the best results.
- LLoVi outperforms all prior approaches on EgoSchema, NeXT-QA, IntentQA and NeXTGQA LVQA benchmarks.

Overall, our framework is simple, effective and training-free. Furthermore, it is agnostic to the exact choice of a visual captioner and an LLM, which allows it to benefit from future improvements in visual captioning and LLM model design. We hope that our work will encourage new ideas and a simpler model design in LVQA. We will release our code to enable the community to build on our work.

## 2 Related Work

Long-range Video Understanding. Modeling long-range videos (e.g., several minutes or longer) typically requires models with sophisticated temporal modeling capabilities, often leading to complex model design. LF-VILA (Sun et al., 2022) proposes a Temporal Window Attention (HTWA) mechanism to capture long-range dependency in long-form video. MeMViT (Wu et al., 2022) and MovieChat (Song et al., 2023b) adopt a memorybased design to store information from previously processed video segments. Several prior methods use space-time graphs (Hussein et al., 2019b; Wang et al., 2021) or relational space-time modules (Yang et al., 2023) to capture spatiotemporal dependencies in long videos. Lastly, the recently introduced S4ND (Nguyen et al., 2022), ViS4mer (Islam and Bertasius, 2022) and S5 (Wang et al., 2023a) use Structured State-Space Sequence (S4) (Gu et al., 2021) layers to capture long-range dependencies in the video. Unlike these prior approaches, we do not use any complex long-range temporal modeling modules but instead develop a simple and strong LLM-based framework for zero-shot LVQA.

LLMs for Video Understanding. The recent surge in large language models (LLMs) (Brown et al., 2020; OpenAI, 2023; Touvron et al., 2023; Raffel et al., 2020; Chung et al., 2022; Tay et al., 2022) has inspired many LLM-based applications
in video understanding. Methods like Socratic Models (Zeng et al., 2022) and VideoChat (Li et al., 2023e) integrate pretrained visual models with LLMs for extracting visual concepts and applying them to video tasks. Video ChatCaptioner (Chen et al., 2023) and ChatVideo (Wang et al., 2023b) leverage LLMs for video representation and dialog-based user interaction, respectively. VidIL (Wang et al., 2022b) employs LLMs for adapting image-level models to video tasks using few-shot learning. Beyond short-term video understanding, the works in (Lin et al., 2023a; Chung and Yu, 2023; Bhattacharya et al., 2023) explored LLMs for long-range video modeling. The work in (Lin et al., 2023a) uses GPT-4 for various longrange video modeling tasks but lacks quantitative evaluation. Meanwhile, (Chung and Yu, 2023) focuses on movie datasets, requiring limited visual analysis (Mangalam et al., 2023) and mostly relying on non-visual speech/subtitle inputs. In contrast to these prior methods, we focus on the LVQA task and provide an extensive empirical analysis of various design choices behind our LLM framework.

Video Question Answering. Unlike image question-answering, video question-answering (VidQA) presents unique challenges, requiring both spatial and temporal reasoning. Most existing VidQA methods, either using pretrainingfinetuning paradigms (Cheng et al., 2023; Lei et al., 2021; Yu et al., 2023), zero-shot (Yang et al., 2022b; Surís et al., 2023; Lin et al., 2023b; Yu et al., 2023), or few-shot learning (Wang et al., 2022b), focus on short-term video analysis (5-30s). To overcome the limitations of short-term VidQA, new benchmarks have been proposed: ActivityNetQA (Yu et al., 2019), TVQA (Lei et al., 2018), How2QA (Yang et al., 2021), MovieQA (Tapaswi et al., 2016), and DramaQA (Choi et al., 2021) ranging from 100 s to several minutes in video duration. Despite longer video lengths, the analysis in (Mangalam et al., 2023; Yang et al., 2020; Jasani et al., 2019) found that many of these benchmarks can be solved by analyzing only short clips (i.e., not requiring long-range video modeling) or by using pure text-only methods that ignore visual content. To address these issues, the EgoSchema benchmark (Mangalam et al., 2023) was recently introduced, requiring at least 100 seconds of video analysis and not exhibiting language-based biases.

LLM Prompt Design. With the emergence of LLMs, there has been an increasing research em-

![](https://cdn.mathpix.com/cropped/2024_05_26_04df3d9f395908fb2e5dg-03.jpg?height=657&width=760&top_left_y=231&top_left_x=1068)

Figure 2: An illustration of LLoVi, our simple LLM framework for long-range video question-answering (LVQA). We use Large Language Models (LLMs) like GPT-3.5 and GPT-4 for their long-range modeling capabilities. Our method involves two stages: first, we use short-term visual captioners (e.g, LaViLa, BLIP2) to generate textual descriptions for brief video clips $(0.5 \mathrm{~s}-$ 8s). Then, an LLM aggregates these dense, short-term captions for long-range reasoning required for LVQA. This simple approach yields impressive results, demonstrating LLMs' effectiveness in LVQA.

phasis on LLM prompt design. The recent works in (Wei et al., 2022; Zhou et al., 2023; Schick and Schütze, 2020; Chen et al., 2022; Yao et al., 2022) explored prompting strategy in few-shot learning settings. To eliminate the need for extensive human annotations, (Kojima et al., 2022; Wang et al., 2023c,f) proposed zero-shot prompting methods. Subsequent research (Zhou et al., 2022; Zhang et al., 2022; Pryzant et al., 2023) has concentrated on the automatic refinement of prompts. Instead, we propose a multi-round summarization LLM prompt for handling long, noisy, and redundant textual inputs describing video content for LVQA.

## 3 Method

Our method, named LLoVi, consists of two stages: 1) short-term video clip captioning and 2) longrange text-based video understanding using an LLM. Figure 2 presents a detailed illustration of our high-level approach. Below, we provide more details about each component of our framework.

### 3.1 Short-term Video Clip Captioning

Given a long untrimmed video input $V$, we first segment it into $N_{v}$ non-overlapping short video clips $v=\left\{v_{m}\right\}_{m=1}^{N_{v}}$, where $v_{m} \in \mathbb{R}^{T_{v} \times H \times W \times 3}$

![](https://cdn.mathpix.com/cropped/2024_05_26_04df3d9f395908fb2e5dg-04.jpg?height=483&width=1496&top_left_y=244&top_left_x=286)

Figure 3: An illustration of our multi-round summarization prompt that first asks an LLM to summarize the noisy short-term visual captions (first round of prompting) and then answer a given question about the video based on the LLM-generated summary (second round of prompting). Our results indicate that such a multi-round prompting strategy significantly boosts LVQA performance compared to standard prompting techniques (+5.8\%).

and $T_{v}, H, W$ are the number of frames, height and width of a short video clip respectively. Afterward, we feed each video clip $v_{m}$ into a pretrained short-term visual captioner $\phi$, which produces textual captions $c_{m}=\phi\left(v_{m}\right)$, where $c_{m}=$ $\left(w_{1}, \ldots, w_{L_{m}}\right)$ and $w_{i}$ represents the i-th word in caption $c_{m}$ of length $L_{m}$. Note that our model is not restricted to any specific visual captioning model. Our experimental section demonstrates that we can incorporate various video ( $\mathrm{LaViLa}$ (Zhao et al., 2023), EgoVLP (Qinghong Lin et al., 2022), and image (BLIP-2 (Li et al., 2023d)) captioning models. Next, we describe how our extracted shortterm captions are processed by an LLM.

### 3.2 Long-range Reasoning with an LLM

We want to leverage foundational LLMs for holistic long-range video understanding. Formally, given short-term visual captions $\left\{c_{m}\right\}_{m=1}^{N_{v}}$ for all $N_{v}$ short video clips, we first concatenate the clip captions into the full video captions $C=\left[c_{1}, \ldots, c_{N_{v}}\right]$ in the same order as the captions appear in the original video. Afterward, the concatenated video captions $C$ are fed into an LLM for long-range video reasoning. Specifically, given the concatenated video captions $C$, the question $Q$, and the answer candidates $A$, we prompt the LLM to select the correct answer using the following prompt template: "Please provide a single-letter answer $(A, B, C, D, E)$ to the following multiple-choice question $\{Q\}$. You are given language descriptions of a video. Here are the descriptions: $\{C\}$. Here are the choices $\{A\} . "$. The full prompt is included in the Supplementary Material.

Our experiments in Section 4.3 suggest that this simple approach works surprisingly well for LVQA. However, we also discovered that many modern LLMs (e.g., GPT-3.5, LLaMA) may struggle when provided with long ( $>1 \mathrm{~K}$ words), noisy, and potentially redundant/irrelevant caption sequences. To address these issues, we investigate more specialized LLM prompts that ask an LLM first to summarize the noisy short-term visual captions (first round of prompting) and then answer a given question about the video (second round of prompting). Specifically, we formulate such a multi-round prompt as follows: given the video captions $C$, the question $Q$, and the answer candidates $A$, instead of directly feeding the $\{C, Q, A\}$ triplet into LLM for LVQA, we first ask the LLM to provide a summary of the captions in the first round, which we denote as $S$ using the following prompt template: "You are given language descriptions of a video: $\{C\}$. Please give me a $\left\{N_{w}\right\}$ word summary." $N_{w}$ denotes the desired number of words in the summary $S$. Afterward, during the second round of prompting, instead of using the captions $C$, we use the summary $S$ as input for the LLM to select one of the answer candidates. Conceptually, such a prompting scheme is beneficial, as the LLMgenerated summary $S$ filters out irrelevant/noisy information from the initial set of captions $C$, making LLM inputs for the subsequent QA process more succinct and cleaner. A detailed illustration of our multi-round prompt is shown in Figure 3.

### 3.3 Implementation Details

For the experiments on EgoSchema, we use LaViLa (Zhao et al., 2023) as our captioner. We segment each video into multiple 1s clips with a stride
![](https://cdn.mathpix.com/cropped/2024_05_26_04df3d9f395908fb2e5dg-05.jpg?height=754&width=780&top_left_y=244&top_left_x=227)

Figure 4: An illustration of prior LVQA dataset limitations. Top: An example from MovieQA (Tapaswi et al., 2016). The model can use the provided subtitle information to answer a question while ignoring visual cues in a video. Middle: An example from the ActivityNet-QA Dataset (Yu et al., 2019). Despite long video inputs, the model only needs to analyze a short $1 \mathrm{~s}$ video clip to answer the question. Bottom: An example from the EgoSchema Dataset (Mangalam et al., 2023). The model must analyze visual cues from the video to answer a given question without relying on additional textual inputs (e.g., speech, subtitles).

of $1 \mathrm{~s}$, resulting in a list of consecutive clips that cover the entire video. We use GPT-3.5 as the LLM on EgoSchema. For NeXT-QA, IntentQA, and NeXT-GQA, we use LLaVA-1.5 (Liu et al., 2023b) as the visual captioner and GPT-4 as the LLM. We downsample the videos to 0.5 FPS and prompt LLaVA to generate captions with roughly 30 words for each frame. More details are provided in the Supplementary Material.

## 4 Experiments

### 4.1 Datasets and Metrics

Unlike short-term video question-answering, longrange video question-answering (LVQA) lacks robust and universally agreed-upon benchmarks. As shown in Figure 4, many prior LVQA benchmarks either exhibit significant language biases, or do not require long-range video modeling capabilities. To address these limitations, recent work introduced EgoSchema (Mangalam et al., 2023), a new long-range video question-answering benchmark consisting of $5 \mathrm{~K}$ multiple choice question-answer

| Captioner | Caption <br> Type | Ego4D <br> Pre-training | Acc. (\%) |
| :--- | :---: | :---: | :---: |
| VideoBLIP (Yu) | clip-level | $\checkmark$ | 40.0 |
| EgoVLP (Qinghong Lin et al., 2022) | clip-level | $\checkmark$ | 46.6 |
| BLIP-2 (Li et al., 2023d) | frame-level | $\boldsymbol{X}$ | 46.7 |
| LaViLa (Zhao et al., 2023) | clip-level | $\checkmark$ | $\mathbf{5 1 . 8}$ |
| Oracle | clip-level | - | 65.8 |

Table 1: Accuracy of our framework with different visual captioners. LaViLa visual captioner achieves the best results, outperforming other clip-level (e.g., EgoVLIP, VideoBLIP) and image-level (e.g., BLIP-2) captioners. We also observe that the Oracle baseline using ground truth captions greatly outperforms all other variants, suggesting that our framework can benefit from the future development of visual captioners.

pairs spanning 250 hours of video and covering a wide range of human activities. By default, our experiments are conducted on the validation set of 500 questions (referred to as the EgoSchema Subset). The final comparison is done on the full test set of 5K EgoSchema questions. We use QA accuracy (i.e., the percentage of correctly answered questions) as our evaluation metric. Additionally, we also perform zero-shot LVQA experiments on three commonly-used LVQA benchmarks: NExTQA (Xiao et al., 2021), IntentQA (Li et al., 2023a), and NExT-GQA (Xiao et al., 2023). Detailed dataset information and metrics can be found in the supplementary material.

### 4.2 Empirical Study on EgoSchema

Before presenting our main results, we first study the effectiveness of different components within our LLoVi framework, including (i) the visual captioner, (ii) the LLM, (iii) the LLM prompt design, and (iv) few-shot in-context learning. The experiments are conducted on the EgoSchema Subset with 500 multi-choice questions. We discuss our empirical findings below. We also include additional experiments in the supplementary material.

### 4.2.1 Visual Captioning Model

In Table 1, we study the effectiveness of various clip-level video captioners, including LaViLa (Zhao et al., 2023), EgoVLP (Qinghong Lin et al., 2022), and VideoBLIP (Yu). In addition to video captioners, we also try the state-of-the-art image captioner, BLIP-2 (Li et al., 2023c). Lastly, to study the upper bound of our visual captioning results, we include the ground truth Oracle captioning baseline obtained from the Ego4D dataset. All baselines in Table 1 use similar experimental settings, including the same LLM model, i.e., GPT-

| LLM | Model Size | Acc. (\%) |
| :--- | :---: | :---: |
| Llama2-7B (Touvron et al., 2023) | 7B | 34.0 |
| Llama2-13B (Touvron et al., 2023) | 13B | 40.4 |
| Llama2-70B (Touvron et al., 2023) | 70B | 50.6 |
| GPT-3.5 (Brown et al., 2020) | 175B | 51.8 |
| GPT-4 (OpenAI, 2023) | N/A | $\mathbf{5 8 . 3}$ |

Table 2: Accuracy of our framework with different LLMs. GPT-4 achieves the best accuracy, suggesting that stronger LLMs perform better in LVQA. However, we use GPT-3.5 for most of our experiments due to the best accuracy and cost tradeoff.

3.5. The results are reported as LVQA accuracy on the EgoSchema Subset.

The results in Table 1, suggest that LaViLa provides the best results, outperforming BLIP-2, EgoVLP, and VideoBLIP. We also observe that despite not being pre-trained on Ego4D (Grauman et al., 2022), BLIP-2 performs reasonably well $\mathbf{( 4 6 . 7 \% )}$ ) and even outperforms other strong Ego4Dpretrained baselines, EgoVLP and VideoBLIP. Lastly, the Oracle baseline with ground truth captions outperforms LaViLa captions by a large margin $(\mathbf{1 4 . 0 \%})$. This shows that our method can benefit from future improvements in captioning models.

### 4.2.2 Large Language Model

In Table 2, we analyze the performance of our framework using different LLMs while fixing the visual captioner to be LaViLa. Our results indicate that GPT-4 achieves the best performance $\mathbf{( 5 8 . 3 \%})$, followed by GPT-3.5 $\mathbf{( 5 1 . 8 \% )}$. Thus, stronger LLMs (GPT-4) are better at long-range modeling, as indicated by a significant margin in LVQA accuracy between GPT-4 and all other LLMs ( $>\mathbf{6 . 5 \%}$ ). We also note that Llama2 performs reasonably well with its 70B variant $(\mathbf{5 0 . 6 \%})$, but its performance drastically degrades with smaller capacity LLMs (i.e., Llama2-7B, Llama2-13B). Due to the tradeoff between accuracy and cost, we use GPT-3.5 for most of our experiments unless noted otherwise.

### 4.2.3 LLM Prompt Analysis

In this section, we (1) analyze several variants of our summarization-based prompt (described in Section 3), and (2) experiment with other commonly used prompt designs, including Zero-shot Chain-of-Thought (Zero-shot CoT) (Wei et al., 2022), Plan-and-Solve (Wang et al., 2023c), and Self-Consistency (Wang et al., 2023f). Below, we present a detailed analysis of these results.

Multi-round Summarization Prompt. Given a concatenated set of captions $C$, an input question

| Prompt Type | Standard | (C) $\rightarrow \mathrm{S}$ | $(\mathrm{C}, \mathrm{Q}) \rightarrow \mathrm{S}$ | $(\mathrm{C}, \mathrm{Q}, \mathrm{A}) \rightarrow \mathrm{S}$ |
| :--- | :---: | :---: | :---: | :---: |
| Acc. (\%) | 51.8 | 53.6 | $\mathbf{5 7 . 6}$ | 55.9 |

Table 3: Different variants of our multi-round summarization prompt. Our results indicate that the $(\mathrm{C}$, $\mathrm{Q}) \rightarrow \mathrm{S}$ variant that takes concatenated captions $C$ and a question $Q$ for generating a summary $S$ works the best, significantly outperforming ( $+\mathbf{5 . 8 \%}$ ) the standard prompt. This confirms our hypothesis that additional inputs in the form of a question $Q$ enable the LLM to generate a summary $S$ tailored to a given question $Q$.

$Q$, and a set of candidate answers $A$, we can use several input combinations to obtain the summary $S$. Thus, here, we investigate three distinct variants of obtaining summaries $S$ :

- (C) $\rightarrow$ S: the LLM uses caption-only inputs $C$ to obtain summaries $S$ in the first round of prompting.
- $(\mathrm{C}, \mathrm{Q}) \rightarrow \mathrm{S}$ : the LLM uses captions $C$ and a question $Q$ as inputs for generating summaries $S$. Having additional question inputs is beneficial as it allows the LLM to generate a summary $S$ specifically tailored for answering an input question $Q$.
- $(\mathrm{C}, \mathrm{Q}, \mathrm{A}) \rightarrow \mathrm{S}$ : the LLM takes captions $C$, a question $Q$, and the answer candidates $A$ as its inputs to produce summaries $S$. Having additional answer candidate inputs enables the LLM to generate a summary $S$ most tailored to particular question-answer pairs.

In Table 3, we explore the effectiveness of these three prompt variants. Our results show that all three variants significantly outperform our standard LVQA prompt (described in Section 3). Specifically, we note that the variant $(\mathrm{C}) \rightarrow \mathrm{S}$ that uses caption-only inputs to obtain the summaries outperforms the standard baseline by $\mathbf{1 . 8 \%}$. Furthermore, we observe that incorporating a given question as an input (i.e., the $(\mathrm{C}, \mathrm{Q}) \rightarrow \mathrm{S}$ variant) leads to the best performance $(\mathbf{5 7 . 6 \%})$ with a significant $\mathbf{5 . 8 \%}$ boost over the standard LVQA prompt baseline. This confirms our earlier intuition that having additional question $Q$ inputs enables the LLM to generate a summary $S$ specifically tailored for answering that question, thus leading to a big boost in LVQA performance. Lastly, we observe that adding answer candidates $A$ as additional inputs (i.e., the $(\mathrm{C}, \mathrm{Q}, \mathrm{A}) \rightarrow \mathrm{S}$ variant) leads to a drop in performance (-1.7\%) compared with the (C, Q) $\rightarrow \mathrm{S}$ variant. This might be because the wrong answers in the candidate set $A$ may mislead the LLM, leading to a suboptimal summary $S$.

| Number of words | 50 | 100 | 300 | 500 | 700 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Acc. (\%) | 55.6 | 57.4 | 55.8 | $\mathbf{5 7 . 6}$ | 55.0 |

Table 4: Number of words in a generated summary. We study the optimal number of words in an LLMgenerated summary. These results suggest that the optimal LVQA performance is obtained when using 500word summaries.

| Prompting Technique | Acc. (\%) |
| :--- | :---: |
| Zero-shot |  |
| Standard | 51.8 |
| Zero-shot Chain-of-Thought (Wei et al., 2022) | 53.2 |
| Plan-and-Solve (Wang et al., 2023c) | 54.2 |
| Self-Consistency (Wang et al., 2023f) | 55.4 |
| Ours | $\mathbf{5 7 . 6}$ |
| Few-shot |  |
| Standard | 56.5 |
| Ours | $\mathbf{6 1 . 7}$ |

Table 5: Comparison with commonly used prompting techniques. The "Standard" means a standard LVQA prompt (see Section 3). We show that our framework benefits from more sophisticated prompting techniques. Our multi-round summarization prompt performs best in both zero-shot and few-shot learning settings.

We also investigate the optimal length of the generated summary $S$, and present these results in Table 4. Specifically, for these experiments, we ask the LLM to generate a summary $S$ using a different number of words (as part of our prompt). We use the best performing $(C, Q) \rightarrow S$ variant for these experiments. Our results indicate that using a very small number of words (e.g., 50) leads to a drop in performance, indicating that compressing the caption information too much hurts the subsequent LVQA performance. Similarly, generating summaries that are quite long (e.g., 700 words) also leads to worse results, suggesting that the filtering of the potentially noisy/redundant information in the captions is important for good LVQA performance. The best performance is obtained using 500 -word summaries.

Comparison with Commonly Used Prompts. Next, in Table 5, we compare our multi-round summarization prompt with other commonly used prompts such as Zero-shot Chain-of-Thought (Wei et al., 2022), Plan-and-Solve (Wang et al., 2023c), and Self-Consistency (Wang et al., 2023f). These results show that all of these prompts outperform the base variant of our model that uses a standard prompt. In particular, among these commonly used prompts, the self-consistency prompting technique

| Model | Acc. (\%) |
| :--- | :---: |
| Zero-shot |  |
| FrozenBiLM (Yang et al., 2022a) | 26.9 |
| mPLUG-Owl (Ye et al., 2023) | 31.1 |
| InternVideo (Wang et al., 2022a) | 32.1 |
| LongViViT (Papalampidi et al., 2023) | 33.3 |
| Vamos (Wang et al., 2023d) | 48.3 |
| LLoVi (Ours) | $\mathbf{5 0 . 3}$ |
| Few-shot |  |
| LLoVi (Ours) | 52.5 |

Table 6: Results on the full set of EgoSchema. The best-performing zero-shot variant of our LLoVi framework achieves $\mathbf{5 0 . 3 \%}$ accuracy, outperforming the previous best-performing InternVideo model by $\mathbf{1 8 . 2 \%}$. For fair comparisons, we gray out our best few-shot variant.

achieves the best results ( $\mathbf{5 5 . 4 \%})$. Nevertheless, our multi-round summarization prompt performs best $(\mathbf{5 7 . 6 \% )})$.

### 4.2.4 Few-shot In-Context Learning

In-context learning with LLMs has shown strong few-shot performance in many NLP tasks (Brown et al., 2020; Wei et al., 2022). In Table 5, we evaluate the few-shot in-context learning capabilities of our LLoVi framework. Our results show that our LLoVi framework greatly benefits from few-shot in-context learning. Specifically, the few-shot incontext learning leads to a $4.7 \%$ boost on the variant of our framework that uses a standard prompt and $4.1 \%$ boost on our advanced framework using a multi-round summarization prompt. We used 6 few-shot examples as we found this configuration to produce the best performance.

### 4.3 Main Results on EgoSchema

In Table 6, we evaluate our best-performing LLoVi framework on the full EgoSchema test set containing $5 \mathrm{~K}$ video samples. We compare our approach with prior state-of-the-art methods including InternVideo (Wang et al., 2022a), mPLUG-Owl (Ye et al., 2023), FrozenBiLM (Yang et al., 2022a), as well as the concurrent works of LongViViT (Papalampidi et al., 2023), and Vamos (Wang et al., 2023d). Based on these results, we observe that the best-performing zero-shot variant of our LLoVi framework achieves $\mathbf{5 0 . 3 \%}$ accuracy, outperforming the concurrent Vamos model $(\mathbf{+ 2 . 0 \%})$. Additionally, we show that by using fewshot in-context learning, our best variant improves even further. These results validate our design choice of using the long-range modeling abilities of LLMs for LVQA. Furthermore, since our proposed

| Model | Cau. (\%) | Tem. (\%) | Des. (\%) | All (\%) |
| :---: | :---: | :---: | :---: | :---: |
| VFC (Momeni et al., 2023) | 45.4 | 51.6 | 64.1 | 51.5 |
| InternVideo (Wang et al., 2022a) | 43.4 | 48.0 | 65.1 | 49.1 |
| ViperGPT (Surí et al., 2023) | - | - | - | 60.0 |
| SeViLA (Yu et al., 2023) | 61.3 | $\mathbf{6 1 . 5}$ | $\mathbf{7 5 . 6}$ | 63.6 |
| LLoVi (ours) | $\mathbf{6 9 . 5}$ | 61.0 | $\mathbf{7 5 . 6}$ | $\mathbf{6 7 . 7}$ |

Table 7: Zero-shot results on NeXT-QA. LLoVi achieves $67.7 \%$ accuracy, outperforming previous bestperforming model SeViLA by $\mathbf{4 . 1 \%}$. Notably, LLoVi excels at causal reasoning outperforming SeViLA by $\mathbf{8 . 2 \%}$ in the causal question category.

LLoVi framework is agnostic to the visual captioning model and an LLM it uses, we believe we could further improve these results by leveraging more powerful visual captioners and LLMs.

### 4.4 Results on Other Datasets

Next, we demonstrate that our simple framework generalizes well to other LVQA benchmarks.

NExT-QA. In Table 7, we evaluate LLoVi on the NExT-QA (Xiao et al., 2021) validation set in a zero-shot setting. We compare our approach with prior methods: VFC (Momeni et al., 2023), InternVideo (Wang et al., 2022a), ViperGPT (Surís et al., 2023), and SeViLA (Yu et al., 2023). We observe that LLoVi outperforms the previous bestperforming method, SeViLA by $\mathbf{4 . 1 \%}$. Notably, in the Causal category, LLoVi achieves $\mathbf{8 . 2 \%}$ improvement. We conjecture this improvement comes from the simple 2-stage design of our LLoVi framework: captioning followed by LLM reasoning. By captioning the video, we are able to directly leverage the reasoning ability of the powerful LLMs and thus achieve good causal reasoning performance.

IntentQA. In Table 8, we evaluate our method on the IntentQA (Li et al., 2023a) test set. In our comparisons, we include several supervised methods (HQGA (Xiao et al., 2022a), VGT (Xiao et al., 2022b), BlindGPT (Ouyang et al., 2022), CaVIR (Li et al., 2023b)) and the recent state-ofthe-art zero-shot approach, SeViLA. From the results in Table 8, we observe that our method greatly outperforms all prior approaches, both in the fully supervised and zero-shot settings.

NExT-GQA. In Table 9, we extend our framework to the grounded LVQA task and evaluate it on the NExT-GQA (Xiao et al., 2023) test set. We compare LLoVi with the weakly-supervised methods: IGV (Li et al., 2022), Temp[CLIP](NG+) (Xiao et al., 2023), FrozenBiLM (NG+) (Xiao et al., 2023) and SeViLA (Yu et al., 2023). These baselines are first trained on NExT-GQA to maxi-

| Model | Acc. (\%) |
| :--- | :---: |
| Supervised |  |
| HQGA (Xiao et al., 2022a) | 47.7 |
| VGT (Xiao et al., 2022b) | 51.3 |
| BlindGPT (Ouyang et al., 2022) | 51.6 |
| CaVIR (Li et al., 2023b) | 57.6 |
| Zero-shot |  |
| SeViLA (Yu et al., 2023) | 60.9 |
| LLoVi (ours) | $\mathbf{6 4 . 0}$ |

Table 8: Results on IntentQA. Our zero-shot framework outperforms previous supervised methods by a large margin ( $6.4 \%)$. LLoVi also outperforms the recent state-of-the-art zero-shot method, SeViLA, by $\mathbf{3 . 1 \%}$.

| Model | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Weakly-Supervised |  |  |  |  |  |
| IGV (Li et al., 2022) | 21.4 | 18.9 | 14.0 | 9.6 | 10.2 |
| Temp[CLIP](NG+) | 25.7 | 25.5 | 12.1 | 8.9 | 16.0 |
| FrozenBiLM (NG+) | 24.2 | 23.7 | 9.6 | 6.1 | 17.5 |
| SeViLA (Yu et al., 2023) | 29.5 | 22.9 | 21.7 | 13.8 | 16.6 |
| Zero-shot |  |  |  |  |  |
| LLoVi (ours) | $\mathbf{3 7 . 3}$ | $\mathbf{3 6 . 9}$ | $\mathbf{2 0 . 0}$ | $\mathbf{1 5 . 3}$ | $\mathbf{2 4 . 3}$ |

Table 9: Grounded LVQA results on NExT-GQA. We extend LLoVi to the grounded LVQA task and show that it outperforms prior weakly-supervised approaches on all evaluation metrics. For a fair comparison, we de-emphasize the models that were pretrained using video-language grounding annotations.

mize the QA accuracy, and then use ad-hoc methods (Xiao et al., 2023) to estimate a relevant video segment for question-answering. Although LLoVi is not trained on NExT-GQA, it still outperforms these weakly-supervised methods by a large margin according to all evaluation metrics. These results demonstrate that our framework can be used to temporally ground its predictions for more explainable long-range video understanding.

## 5 Conclusion

In this work, we present a simple, yet highly effective LLM-based framework for long-range video question-answering (LVQA). Our framework outperforms all prior models on the newly introduced EgoSchema benchmark. Furthermore, we demonstrate that our approach generalizes to other LVQA benchmarks such as NeXT-QA, IntentQA, and it can also be extended to grounded LVQA tasks. Lastly, we thoroughly evaluate various design choices of our approach and analyze the key factors behind the success of our method. We hope that our simple LVQA framework will help inspire new ideas and simplify model design in long-range
video understanding.

## Limitations

Our proposed framework used different short-term visual captioning models for egocentric and exocentric videos due to the domain difference. A unified captioner that works for all kinds of videos remains to be explored in the future. Additionally, our multiround summarization prompt requires two rounds of prompting LLMs. Although it leads to a significant performance boost on LVQA, it also causes extra computational cost. Therefore, the trade-off between efficiency and high performance in our prompt design can be further improved.

## 6 Acknowledgement

We thank Karttikeya Mangalam, Feng Cheng, YanBo Lin, Yue Yang and Soumitri Chattopadhyay for their discussion and valuable feedback. This work was supported by Sony Faculty Innovation award, Laboratory for Analytic Sciences via NC State University, ONR Award N00014-23-1-2356.

## References

Gedas Bertasius, Heng Wang, and Lorenzo Torresani. 2021. Is space-time attention all you need for video understanding? In ICML, volume 2 , page 4.

Aanisha Bhattacharya, Yaman K Singla, Balaji Krishnamurthy, Rajiv Ratn Shah, and Changyou Chen. 2023. A video is worth 4096 tokens: Verbalize story videos to understand them in zero shot. arXiv preprint arXiv:2305.09758.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.

Jun Chen, Deyao Zhu, Kilichbek Haydarov, Xiang Li, and Mohamed Elhoseiny. 2023. Video chatcaptioner: Towards the enriched spatiotemporal descriptions. arXiv preprint arXiv:2304.04227.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588.

Yihong Chen, Yue Cao, Han Hu, and Liwei Wang. 2020. Memory enhanced global-local aggregation for video object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10337-10346.
Feng Cheng and Gedas Bertasius. 2022. Tallformer: Temporal action localization with a long-memory transformer. In European Conference on Computer Vision, pages 503-521. Springer.

Feng Cheng, Xizi Wang, Jie Lei, David Crandall, Mohit Bansal, and Gedas Bertasius. 2023. Vindlu: A recipe for effective video-and-language pretraining. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Seongho Choi, Kyoung-Woon On, Yu-Jung Heo, Ahjeong Seo, Youwon Jang, Min Su Lee, and ByoungTak Zhang. 2021. Dramaqa: Character-centered video story understanding with hierarchical QA. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 1166-1174. AAAI Press.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Jiwan Chung and Youngjae Yu. 2023. Long story short: a summarize-then-search method for long video question answering. In $B M V C$.

Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. 2021. VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling. In arXiv:2111.1268.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. 2022. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995-19012.

Albert Gu, Karan Goel, and Christopher Ré. 2021. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2023. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

Noureldien Hussein, Efstratios Gavves, and Arnold WM Smeulders. 2019a. Timeception for complex action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 254-263.

Noureldien Hussein, Efstratios Gavves, and Arnold WM Smeulders. 2019b. Videograph: Recognizing minutes-long human activities in videos. arXiv preprint arXiv:1905.05143.

Md Mohaiminul Islam and Gedas Bertasius. 2022. Long movie clip classification with state-space video models. In European Conference on Computer Vision, pages $87-104$. Springer.

Md Mohaiminul Islam, Mahmudul Hasan, Kishan Shamsundar Athrey, Tony Braskich, and Gedas Bertasius. 2023. Efficient movie scene detection using state-space transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18749-18758.

Bhavan Jasani, Rohit Girdhar, and Deva Ramanan. 2019. Are we asking the right questions in movieqa? In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0-0.

Diederik P. Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. CoRR, abs/1412.6980.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199_ 22213.

Sangho Lee, Jinyoung Sung, Youngjae Yu, and Gunhee Kim. 2018. A memory network approach for storybased temporal summarization of 360 videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1410-1419.

Sangmin Lee, Hak Gu Kim, Dae Hwi Choi, HyungIl Kim, and Yong Man Ro. 2021. Video prediction recalling long-term motion context via memory alignment learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3054-3063.

Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. 2021. Less is more: Clipbert for video-and-language learningvia sparse sampling. In $C V P R$.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. 2018. Tvqa: Localized, compositional video question answering. In EMNLP.

Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. 2023a. Intentqa: Context-aware video intent reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1196311974.
Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. 2023b. Intentqa: Context-aware video intent reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages $11963-11974$.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023c. BLIP-2: bootstrapping language-image pretraining with frozen image encoders and large language models. In ICML.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023d. BLIP-2: bootstrapping language-image pretraining with frozen image encoders and large language models. In ICML.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023e. Videochat: Chat-centric video understanding.

Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, and TatSeng Chua. 2022. Invariant grounding for video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2928-2937.

Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, Ce Liu, and Lijuan Wang. 2023a. Mm-vid: Advancing video understanding with gpt-4v(ision). arXiv preprint arXiv:2310.19773.

Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. 2023b $\mathrm{Mm}$-vid: Advancing video understanding with gpt4v (ision). arXiv preprint arXiv:2310.19773.

Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. 2023a. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. In NeurIPS.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. 2023. Egoschema: A diagnostic benchmark for very long-form video language understanding. arXiv preprint arXiv:2308.09126.

Liliane Momeni, Mathilde Caron, Arsha Nagrani, Andrew Zisserman, and Cordelia Schmid. 2023. Verbs in action: Improving verb understanding in videolanguage models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages $15579-15591$.

Eric Nguyen, Karan Goel, Albert Gu, Gordon Downs, Preey Shah, Tri Dao, Stephen Baccus, and Christopher Ré. 2022. S4nd: Modeling images and videos as multidimensional signals with state spaces. Advances in neural information processing systems, 35:28462861.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744.

Pinelopi Papalampidi, Skanda Koppula, Shreya Pathak, Justin Chiu, Joe Heyward, Viorica Patraucean, Jiajun Shen, Antoine Miech, Andrew Zisserman, and Aida Nematzdeh. 2023. A simple recipe for contrastively pre-training video-first encoders beyond 16 frames. arXiv preprint arXiv:2312.07395.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with" gradient descent" and beam search. arXiv preprint arXiv:2305.03495.

Kevin Qinghong Lin, Alex Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Difei Gao, Rongcheng Tu, Wenzhe Zhao, Weijie Kong, et al. 2022. Egocentric video-language pretraining. arXiv e-prints, pages arXiv-2206.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551.

Timo Schick and Hinrich Schütze. 2020. Exploiting cloze questions for few shot text classification and natural language inference. arXiv preprint arXiv:2001.07676.

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. 2023a. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998-3009.

Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. 2023b. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449.
Simeng Sun, Yang Liu, Shuohang Wang, Chenguang Zhu, and Mohit Iyyer. 2023. Pearl: Prompting large language models to plan and execute actions over long documents. arXiv preprint arXiv:2305.14564.

Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, and Jianlong Fu. 2022. Long-form videolanguage pre-training with multimodal temporal contrastive learning. Advances in neural information processing systems, 35:38032-38045.

Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. Proceedings of IEEE International Conference on Computer Vision (ICCV).

Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. 2016. Movieqa: Understanding stories in movies through question-answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4631-4640.

Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Steven Zheng, et al. 2022. Ul2: Unifying language learning paradigms. In The Eleventh International Conference on Learning Representations.

Hugo Touvron, Louis Martin, Kevin Stone, Peter A1bert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Jue Wang, Wentao Zhu, Pichao Wang, Xiang Yu, Linda Liu, Mohamed Omar, and Raffay Hamid. 2023a. Selective structured state-spaces for long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6387-6397.

Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. 2023b. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023c. Plan-and-solve prompting: Improving zeroshot chain-of-thought reasoning by large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2609-2634, Toronto, Canada. Association for Computational Linguistics.

Shijie Wang, Qi Zhao, Minh Quan Do, Nakul Agarwal, Kwonjoon Lee, and Chen Sun. 2023d. Vamos: Versatile action models for video understanding. arXiv preprint arXiv:2311.13627.

Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang. 2023e. Gpt-ner: Named entity recognition via large language models. arXiv preprint arXiv:2304.10428.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023f. Self-consistency improves chain of thought reasoning in language models. In ICLR.

Yang Wang, Gedas Bertasius, Tae-Hyun Oh, Abhinav Gupta, Minh Hoai, and Lorenzo Torresani. 2021. Supervoxel attention graphs for long-range video modeling. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages $155-166$.

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. 2022a. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191.

Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, et al. 2022b. Language models with image descriptors are strong few-shot video-language learners. Advances in Neural Information Processing Systems, 35:8483-8497.

Ziyang Wang, Yi-Lin Sung, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2023g. Unified coarse-tofine alignment for video-text retrieval. arXiv preprint arXiv:2309.10091.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837.

Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi Fan, Kaiming He, Philipp Krahenbuhl, and Ross Girshick. 2019. Long-term feature banks for detailed video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 284-293.

Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. 2022. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13587-13597.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. 2021. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777-9786.

Junbin Xiao, Angela Yao, Yicong Li, and Tat Seng Chua. 2023. Can i trust your answer? visually grounded video question answering. arXiv preprint arXiv:2309.01327.
Junbin Xiao, Angela Yao, Zhiyuan Liu, Yicong Li, Wei Ji, and Tat-Seng Chua. 2022a. Video as conditional graph hierarchy for multi-granular question answering. In Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI), pages 2804-2812.

Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. 2022b. Video graph transformer for video question answering. In European Conference on Computer Vision, pages 39-58. Springer.

Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. 2021. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1686-1697.

Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. 2022a. Zero-shot video question answering via frozen bidirectional language models. In NeurIPS.

Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. 2022b. Zero-shot video question answering via frozen bidirectional language models. Advances in Neural Information Processing Systems, 35:124-141.

Jianing Yang, Yuying Zhu, Yongxin Wang, Ruitao Yi, Amir Zadeh, and Louis-Philippe Morency. 2020 What gives the answer away? question answering bias analysis on video qa datasets. arXiv preprint arXiv:2007.03626.

Xitong Yang, Fu-Jen Chu, Matt Feiszli, Raghav Goyal, Lorenzo Torresani, and Du Tran. 2023. Relational space-time query in long-form videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6398-6408.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Keunwoo Peter Yu. VideoBLIP.

Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. 2023. Self-chained image-language model for video localization and question answering. NeurIPS.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. 2019. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages $9127-9134$.

Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. 2022. Socratic models: Composing zeroshot multimodal reasoning with language. arXiv.

Chuhan Zhang, Ankush Gupta, and Andrew Zisserman. 2021. Temporal query networks for finegrained video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4486-4496.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493.

Yue Zhao, Ishan Misra, Philipp Krähenbühl, and Rohit Girdhar. 2023. Learning video representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6586-6597.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. 2023 Least-to-most prompting enables complex reasoning in large language models.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910.
</end of paper 1>


<paper 2>
# Memory Consolidation Enables Long-Context Video Understanding 

Ivana Balažević ${ }^{* 1}$ Yuge Shi ${ }^{* 1}$ Pinelopi Papalampidi" ${ }^{* 1}$<br>Rahma Chaabouni ${ }^{1}$ Skanda Koppula ${ }^{1}$ Olivier J. Hénaff ${ }^{1}$


#### Abstract

Most transformer-based video encoders are limited to short temporal contexts due to their quadratic complexity. While various attempts have been made to extend this context, this has often come at the cost of both conceptual and computational complexity. Instead, we propose to re-purpose existing pretrained video transformers by simply fine-tuning them to attend to memories derived non-parametrically from past activations. By leveraging redundancy reduction, our memoryconsolidated vision transformer (MC-ViT) effortlessly extends its context far into the past and exhibits excellent scaling behavior when learning from longer videos. In doing so, MC-ViT sets a new state-of-the-art in long-context video understanding on EgoSchema, Perception Test, and Diving48, outperforming methods that benefit from orders of magnitude more parameters.


## 1. Introduction

Humans and animals reason about events extending over days, weeks, and years (Tulving, 1985), yet current artificial vision systems live largely in the present. While architectures that model the dynamics of natural videos have grown ever more sophisticated (Carreira \& Zisserman, 2017; Feichtenhofer et al., 2019; Arnab et al., 2021), the temporal extent over which they reason has typically been limited to a small number of frames. In particular, transformer architectures (Vaswani et al., 2017) which power most applications in vision and language do not scale to the vast number of tokens present in natural videos due to their quadratic complexity. For example, 30 minutes of video sampled at standard rates may contain half a million tokensmore than what current state-of-the-art architectures using optimized attention algorithms (e.g. Dao et al., 2022) can[^0]

![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-01.jpg?height=696&width=811&top_left_y=606&top_left_x=1058)

Figure 1. Long-context video understanding on EgoSchema and Perception Test. The proposed Memory-Consolidated Vision Transformer (MC-ViT- $\{\mathrm{B}, \mathrm{L}\}$, shown in bold) surpasses both public and large-scale proprietary models, despite using orders of magnitude fewer parameters and requiring only short fine-tuning schedules on top of standard pretrained models.

process. Several attempts have been made to extend the temporal context of video transformers, including masking, attention approximations, and parametric memory modules (e.g. Wu et al., 2022; Piergiovanni et al., 2023b). However, these approaches often introduce additional complexity, requiring specialized architectures and training paradigms.

In this work, we question whether such modifications are indeed necessary to enable long-context modeling. Starting from standard pretrained video transformers (Arnab et al., 2021), we process videos in a streaming setting in order to bound their complexity by the length of short segments (Dai et al., 2019). Crucially, we process individual segments in relation to a memory bank which is populated nonparametrically with the consolidated activations from past segments. This allows us to re-purpose pretrained video transformers for long-context understanding without any architectural modification, by simply fine-tuning them to attend to this memory with short training schedules.

A central question we are faced with is therefore how to choose which of the quasi-infinite tokens from past frames
to store in memory. Inspired by evidence from psychology and neuroscience which formulates memory as a reconstructive process (Bartlett, 1932; Marr, 1971; Spens \& Burgess, 2024), we adopt simple nonparametric schemes that form memories that are maximally representative of the full set of past activations. We find these mechanisms to effectively compress memories by an order of magnitude, and allow our memory-consolidated vision transformer (MC-ViT) to extend its context to significantly longer videos while maintaining a bounded complexity. In particular,

1. MC-ViT strikes a favorable trade-off between computational complexity and expressivity, outperforming standard video transformers and efficient approximations thereof with $10 \times$ less memory and computation.
2. The non-parametric nature of MC-ViT allows us to straightforwardly re-purpose off-the-shelf pretrained video transformers by fine-tuning them to use their consolidated memory, yielding large efficiency gains by decreasing overall training time on long videos.
3. MC-ViT sets a new state-of-the-art on long-context video understanding tasks such as fine-grained action recognition (Diving48) and video question answering (EgoSchema and Perception Test), outperforming methods which benefit from orders of magnitude more parameters.
4. MC-ViT is competitive with large-scale proprietary systems such as GPT-4V and Bard, despite using a small, standard, and open architecture and training paradigm.

## 2. Related Work

Long-context architectures. Prior work has thoroughly explored approaches for handling long textual or visual inputs, by sparsifying either the input tokens or the attention applied over these tokens. In natural language processing, notable examples include Big Bird (Zaheer et al., 2020) and LongFormer (Beltagy et al., 2020) that employ local self-attention over restricted windows combined with global tokens that attend over the entire sequence. Alternative attention mechanisms in vision have utilized pooling (Wang et al., 2021; Li et al., 2022b), linear (Bolya et al., 2022) and windowed formulations (Dong et al., 2022; Li et al., 2022a; Ryali et al., 2023). Several works reduce the number of tokens via multi-resolution patchification, thus processing the input video at different granularities (Feichtenhofer et al., 2019; Yan et al., 2022a; Piergiovanni et al., 2023a). Similarly, Papalampidi et al. (2023) showcase the benefits of this approach by training video encoders on long contexts with high ratios of input masking. Current state-of-the-art approaches for processing long videos consist of modular systems for captioning and extracting frame-level information, followed by a billion-scale LLM for aggregating this information (Zeng et al., 2022; Wang et al., 2022c; Li et al.,
2023; Lin et al., 2023; Wang et al., 2023; Zhang et al., 2023). The approach proposed in this work is orthogonal to these, by re-purposing standard transformer architectures for long-context modeling, whose representations can be incorporated into LLMs.

Memory-augmented transformers. Since the introduction of transformers (Vaswani et al., 2017), several works have sought to give them additional context via auxiliary memory banks. In NLP, TransformerXL does so by simply attending to recent activations in a streaming setting (Dai et al., 2019), whereas Retro (Borgeaud et al., 2022) does so by retrieving semantically related content. In vision, memory-augmented architectures have also been shown to enable video object segmentation (Oh et al., 2019), tracking (Lai et al., 2020), and action recognition (Wu et al., 2019). However, none of these seek to consolidate the memories of past events.

Memory-compressing transformers. Several transformerbased architectures explored compressing past activations into a finite-length memory. In NLP, Neural Turing Machines (Graves et al., 2014) and Token Turning Machines (Ryoo et al., 2023) learn to read and write from a memory bank in an end-to-end manner. Similarly, Compressive Transformers (Rae et al., 2020), $\infty$-former (Martins et al., 2022)—and in vision, MemDPC (Han et al., 2020), LSTR (Xu et al., 2021b) and MeMViT (Wu et al., 2022)—extend the effective context length by compressing prior activations with additional parametric modules. Concurrent work Mirasol3B (Piergiovanni et al., 2023b) showcases the power of this approach by combining these memory modules with large language models and a bespoke pretraining protocol. Our work differs from these in that we find that a simple, non-parametric mechanism followed by light-weight finetuning is sufficient to re-purpose standard pretrained video transformer architectures (e.g. ViViT, Arnab et al., 2021) to achieve strong long-context modeling.

## 3. Method

### 3.1. Overview of Video Vision Transformers (ViViT)

Video Vision Transformers (ViViT; Arnab et al. 2021) adapt Vision Transformers (Dosovitskiy et al., 2021) to straightforwardly process videos. Specifically, ViViT divides a video $V \in \mathbb{R}^{T \times H \times W}$ into $N_{T}$ non-overlapping spatio-temporal patches $x_{i} \in \mathbb{R}^{t \times h \times w}$ such that $N_{T}=\frac{T}{t} \cdot \frac{H}{h} \cdot \frac{W}{w}$, and linearly projects these patches into 1D embedding space:

$$
\begin{equation*}
z_{i}=\boldsymbol{E} x_{i}+p_{i}, \tag{1}
\end{equation*}
$$

where $\boldsymbol{E}$ denotes a learnable projection layer and $p_{i} \in$ $\mathbb{R}^{d}$ additional position embeddings. The resulting token sequence $\boldsymbol{z}^{0}=\left[z_{i}, i \in\left[1, N_{T}\right]\right] \in \mathbb{R}^{N_{T} \times d}$ is then passed through a series of $L$ transformer layers, which alternate Multi-head Self-Attention (MSA; Vaswani et al. 2017), layer

![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-03.jpg?height=454&width=1704&top_left_y=206&top_left_x=188)

Streaming ViT
Memory-Augmented ViT

![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-03.jpg?height=302&width=815&top_left_y=304&top_left_x=1059)

Memory-Consolidated ViT

Figure 2. Visualization of the proposed method. Left: Streaming ViT processes each segment of the sequence independently by attending over activations within a segment. Middle: Memory-Augmented ViT, similar to Transformer XL (Dai et al., 2019), attends to current activations (yellow blocks) and those in recent history (green blocks). Right: In Memory-consolidated ViT, we consolidate the extended context into shorter memory and cross-attend over them, which enables us to effectively attend over longer sequences.

normalization (LN; Ba et al. 2016) and MLP blocks:

$$
\begin{align*}
\boldsymbol{y}^{l} & =\operatorname{MSA}\left(\operatorname{LN}\left(\boldsymbol{z}^{l}\right)\right)+\boldsymbol{z}^{l}  \tag{2}\\
\boldsymbol{z}^{l+1} & =\operatorname{MLP}\left(\operatorname{LN}\left(\boldsymbol{y}^{l}\right)\right)+\boldsymbol{y}^{l} \tag{3}
\end{align*}
$$

While various schemes for factorizing the attention have been proposed for ViViT, we build our model upon the simplest joint space-time attention which models dependencies between all tokens in the sequence. We leave exploring other factorization methods for future research. In contrast to ViViT's self-attention which spans the entire video, our MC-ViT model uses self-attention within much shorter segments, and cross-attention across segments via a set of consolidated memories, which we detail below.

### 3.2. Memory-Consolidated Vision Transformers

In this section, we explore three successive modifications to the original ViViT architecture that enable efficient and expressive scaling to longer videos (see visualization in Figure 2). The culmination of these modifications represents our proposed method: Memory-Consolidated ViT (MCViT). We apply consistent pre- and post-processing steps across all three approaches: we divide the video $V$ into $s$ temporal segments $v_{\tau} \in \mathbb{R}^{S \times H \times W}$, where $S=\frac{T}{s}$ is the number of frames per segment and $S=16$ in our experiments. We then process each segment (either individually or jointly, see below), yielding a list of $s$ representations $\left\{\boldsymbol{z}_{1}, \cdots, \boldsymbol{z}_{s}\right\}$, one for each segment. All of these are then concatenated as the final representation of the video.

Streaming ViT (ST-ViT). Since the computational complexity of transformers scales quadratically with the number of tokens, full joint space-time attention becomes intractable for video lengths that exceed even small numbers of frames. To counteract this, we start with a simple streaming-based extension of ViViT, which processes each segment $v_{\tau}, \tau \in[1, s]$ independently, as described in Section 3.1, with positional embeddings spanning the entire video. Crucially, the number of tokens processed by the ViViT encoder at a given time is instead $N=\frac{S}{t} \cdot \frac{H}{h} \cdot \frac{W}{w}$, bounding the quadratic complexity by the segment length $S$ rather than the total video length $T$. We include the pseudocode for the streaming ViT implementation in Appendix A, Algorithm 2.

Memory-Augmented ViT (MA-ViT). While more scalable, the streaming setting limits the encoder's ability to reason over events which span multiple segments. Hence, as in Dai et al. (2017), we augment the self-attention module with an additional set of memories $\boldsymbol{m}_{\tau}^{l}=\left[\boldsymbol{z}_{0}^{l} ; \boldsymbol{z}_{1}^{l} ; \ldots ; \boldsymbol{z}_{\tau-1}^{l}\right] \in$ $\mathbb{R}^{M \times d}$ consisting of concatenated activations of previous segments at each layer $l$ :

$$
\begin{equation*}
\boldsymbol{y}_{\tau}^{l}=\operatorname{MCA}(\underbrace{\operatorname{LN}\left(\boldsymbol{z}_{\tau}^{l}\right)}_{\text {query }}, \underbrace{\left[\operatorname{LN}\left(\boldsymbol{z}_{\tau}^{l}\right) ; \operatorname{LN}\left(\boldsymbol{m}_{\tau}^{l}\right)\right]}_{\text {key-value }})+\boldsymbol{z}_{\tau}^{l} \tag{4}
\end{equation*}
$$

where $[; ; \cdot]$ denotes the concatenation operation and Multihead Cross-Attention (MCA; Dai et al. 2019) generalizes MSA by decoupling the inputs to the query and key/value heads. Specifically, the MCA operation allows activations from the current segment $\boldsymbol{z}_{\tau}^{l}$ to attend both to themselves (as in MSA) and to memories of all past activations $\boldsymbol{m}_{\tau}^{l}$, while keeping the quadratic complexity limited to $N+M$. We include the pseudocode for Memory-Augmented ViT in Appendix A, Algorithm 3.

Memory-Consolidated ViT (MC-ViT). Given the memoryaugmented vision transformer architecture, a central question is how to consolidate the (potentially infinite) activations of previous segments into a finite (and ideally small) set of memories. We consider three simple instances of memory consolidation that model memory through a nonparametric reconstructive process.

To produce a new consolidated memory $\boldsymbol{m}_{\tau}$ for the current segment (dropping the layer index $l$ for concision), we consolidate the set of activations from the preceding segment $\boldsymbol{z}_{\tau-1} \in \mathbb{R}^{N \times d}$ into $\hat{\boldsymbol{z}}_{\tau-1} \in \mathbb{R}^{K \times d}(K \leq N)$ and concatenate them to the memories consolidated from all prior segments $\boldsymbol{m}_{\tau}=\left[\boldsymbol{m}_{\tau-1}, \hat{\boldsymbol{z}}_{\tau-1}\right] \in \mathbb{R}^{(M+K) \times d}$. The proposed instances of non-parametric memory consolidation differ in their way of computing $\hat{\boldsymbol{z}}_{\tau-1}$, which we detail below.

```
Algorithm 1 Memory-consolidated ViT.
def mc_vit
    video, n_chunks, n_layers
    pos_emb, mc_method, num_mem
) :
    emb = linear_proj(video) + pos_emb \# [B, N, D]
    chunked_video = np.split(emb, n_chunks, axis=1)
    memory $=$ None
    $\mathrm{zs}=[]$
    for z in chunked_video:
        $z$ _norm = layer_norm (z)
    for - in range(n_layers):
        if memory is None:
            $y=$ self_attention(z_norm) + z
        else:
            $\mathrm{kv}=\mathrm{np}$. concatenate(z_norm, memory))
            $y=$ cross_attention( $q=z$ _norm, kv=kv) $+z$
        y_norm = layer_norm(y)
        $z=m l p\left(y \_n o r m\right)+y$
        memory = memory_consolidation(
                    memory, $z$, num mem, mc method)
        memory = layer_norm(memory)
        zs.append $(z)$
    return np.concatenate(zs, axis=1)
```

MC-ViT-R (random) is the simplest non-parametric baseline which randomly selects a set of $K$ activations from $\boldsymbol{z}_{\tau-1}$ and uses them as the consolidated memory for the preceding segment:

$$
\begin{equation*}
\hat{\boldsymbol{z}}_{\tau-1}^{\mathrm{R}}=\left\{\boldsymbol{z}_{\tau-1, k} \mid k \in \mathcal{I}\right\} \in \mathbb{R}^{K \times d} \tag{5}
\end{equation*}
$$

where $\mathcal{I} \in[1, N]^{K}$ is a set of $K$ randomly selected indices.

MC-ViT-CS (coreset) constructs a maximally representative set of memories by applying the greedy coreset selection algorithm (Agarwal et al., 2005) to the activations of the preceding segment $\boldsymbol{z}_{\tau-1}$ by iteratively adding the most distant activations to the ones already included in the consolidated memory for that segment. One iteration of the algorithm is defined as:

$$
\begin{gather*}
k^{*}=\underset{k \in[1, N]}{\arg \max } \min _{j \in \mathcal{M}^{*}}\left\|\boldsymbol{z}_{\tau-1, k}-\boldsymbol{z}_{\tau-1, j}\right\|_{2}^{2}  \tag{6}\\
\mathcal{M}^{*} \leftarrow \mathcal{M}^{*} \cup\left\{k^{*}\right\} \tag{7}
\end{gather*}
$$

where $\mathcal{M}^{*}$ is the set of activation indices chosen to be added to the consolidated memory $\hat{\boldsymbol{z}}_{\tau-1}^{\mathrm{CS}}$. The greedy coreset selection algorithm is run for $K$ iterations to produce the consolidated memory $\hat{\boldsymbol{z}}_{\tau-1}^{\mathrm{CS}} \in \mathbb{R}^{K \times d}$ for the segment $v_{\tau-1}$. Due to its iterative nature, the coreset selection algorithm becomes increasingly computationally expensive as the size of the segment memory $K=\left|\mathcal{M}^{*}\right|$ increases.

MC-ViT-KM (k-means) randomly initializes $K$ cluster centroids as $\hat{\boldsymbol{z}}_{\tau-1}^{\mathrm{R}}$ (see Equation 5) and then performs 5 iter-

![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-04.jpg?height=44&width=829&top_left_y=2141&top_left_x=187)
ous segment $\boldsymbol{z}_{\tau-1}$ to compute the updated cluster centroids, which we use as the consolidated memory $\hat{\boldsymbol{z}}_{\tau-1}^{\mathrm{KM}} \in \mathbb{R}^{K \times d}$ for the segment $v_{\tau-1}$.

We include the pseudocode for MC-ViT in Algorithm 1. The newly consolidated memory $\boldsymbol{m}_{\tau}$ is then jointly processed with the current segment activations $\boldsymbol{z}_{\tau}$ via MCA, analogously to MA-ViT (see Equation 4).
We compare these different consolidation methods in Section 4.4 and find that MC-ViT-KM performs better than the others. Therefore, unless specified otherwise, MC-ViT refers to MC-ViT-KM in the following sections.

### 3.3. Training and Evaluation

Initialization. Since the parameters of MC-ViT are almost identical to those of ViViT (Arnab et al., 2021), we initialize most parameters from a ViViT encoder pretrained on short (16-frame) video clips using multimodal contrastive learning (Xu et al., 2021a; Papalampidi et al., 2023), see Appendix B.1. The only parameters which differ are positional embeddings, as we fine-tune $\mathrm{MC}-\mathrm{ViT}$ on significantly longer videos (e.g. up to 128 frames) than the short clips used for pretraining. We therefore initialize these positional embeddings with linear upsampling along the time dimension. Similarly, we re-use and fine-tune a BERT-style language encoder pretrained in the same setup.

Fine-tuning. For each evaluation, we fine-tune on a dataset mixture that enables a like-for-like comparison with the previous state-of-the-art. All datasets are composed of videotext pairs, and we therefore simply fine-tune the model with noise contrastive estimation. Given the video and text embeddings $\boldsymbol{z}_{i}^{v}$ and $\boldsymbol{z}_{i}^{t}$ of an example $i$, we minimize

$$
\begin{equation*}
\ell_{i}=-\log \frac{\exp \left(\boldsymbol{z}_{i}^{v} \cdot \boldsymbol{z}_{i}^{t}\right)}{\sum_{j} \exp \left(\boldsymbol{z}_{i}^{v} \cdot \boldsymbol{z}_{j}^{t}\right)}-\log \frac{\exp \left(\boldsymbol{z}_{i}^{t} \cdot \boldsymbol{z}_{i}^{v}\right)}{\sum_{j} \exp \left(\boldsymbol{z}_{i}^{t} \cdot \boldsymbol{z}_{j}^{v}\right)} \tag{8}
\end{equation*}
$$

where the "negative" embeddings $\boldsymbol{z}_{j}^{v}$ and $\boldsymbol{z}_{j}^{t}$ are the in-batch examples unless otherwise specified. We provide further training details in Appendix B.2.

Evaluation. We employ the standard zero-shot transfer paradigm from CLIP (Radford et al., 2021) to perform all downstream tasks. In all cases, a test video is equipped with multiple possible "captions", only one of which is correct. For action recognition, these captions are simply the class names. For video question answering, captions are questionanswer pairs constructed from the set of multiple-choice answers. We utilize the language model to compute caption embeddings $\boldsymbol{z}_{i}^{t}$, and compare them to the video embedding $\boldsymbol{z}_{i}^{v}$. The model's prediction $i^{*}=\arg \max _{i} \boldsymbol{z}_{i}^{v} \cdot \boldsymbol{z}_{i}^{t}$ is simply the caption with the highest similarity.

## 4. Experiments

### 4.1. Datasets

We evaluate our method on four challenging datasets for long-context video understanding, namely Diving48, EgoSchema, Next-QA, and Perception Test.

Diving48 (Li et al., 2018) was specifically designed to assess the importance of dynamic and long-term temporal reasoning in action recognition. Video lengths vary between
![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-05.jpg?height=646&width=1702&top_left_y=216&top_left_x=187)

Figure 3. MC-ViT effectively learns from long videos. Left: MC-ViT scales to long Diving48 videos at both training and inference time, and benefits from fine-tuning on longer videos. Middle: Joint space-time attention benefits from fine-tuning on longer videos, but cannot learn from long ( 128 frame) videos due to its large complexity and memory footprint. Right: ST-ViT scales to longer videos but does not benefit from training on them.

24 and 822 frames, with 158 frames on average. Each video is categorized into 48 fine-grained classes based on the specific dive type it depicts. Consequently, correct classification requires dense video sampling and fine-grained understanding in addition to retaining information over a long temporal extent, which necessitates reasoning over a large number of frames. To align with prior methods, we fine-tune on the Diving48 training set and re-initialize the language encoder randomly with a linear embedding function.

EgoSchema (Mangalam et al., 2023) is a long-form multiple-choice video question answering dataset derived from Ego4D (Grauman et al., 2022). The task involves selecting the correct answer out of five options based on a three-minute-long video clip. This task is particularly interesting for evaluating long-context understanding, as it benefits from long "temporal certificate" lengths, i.e. the minimum video duration a human needs to answer the question accurately. The model is fine-tuned on a mixture of HowTo100M and Ego4D, and we ensure that there is no overlap between Ego4D training and EgoSchema examples.

Next-QA (Xiao et al., 2021) emphasizes testing causal and temporal reasoning with open- and close-ended (multiplechoice) QA tasks. Videos in this dataset have an average duration of 44 seconds but can be as long as 2 minutes. We use the close-ended version for both fine-tuning and inference. Since the training set is fairly small and in order to avoid over-fitting on this domain, we add and only tune lowrank adapters (LoRA; Hu et al. 2021) at the self-attention and feed-forward blocks of every layer, which account for $\sim 12 \%$ of model parameters. For fine-tuning on this multiplechoice QA dataset, we use the four incorrect answers to the given question as hard negatives in Equation (8).
Perception Test (Pătrăucean et al., 2023) is inspired by assessment in developmental psychology and features a collection of games or daily activities that evaluate a model's grasp of physics, reasoning, memory, and semantic extraction. Although videos in this dataset are short with an average duration of 30 seconds, accurate localization and recognition of actions and objects require a higher FPS rate (we use an FPS of 4), resulting in sequences of hundreds of frames. We evaluate on the multiple-choice video question answering task by selecting one out of three possible answers, while training on Next-QA for zero-shot evaluation on this benchmark.

### 4.2. MC-ViT Effectively Learns from Long Videos

We start by assessing the ability of MC-ViT to model videos of increasing lengths. For this we fine-tune MC-ViT on videos with different number of frames $(16,32,64$, or 128) by varying the FPS rate. At inference time, we also apply the model to videos with 16 to 256 frames. Figure 3 (left) shows that MC-ViT's performance improves with more, densely sampled frames at both training and inference time on Diving48 fine-grained action recognition. In particular, training with longer contexts allows MC-ViT to benefit from more frames at inference time, with the optimal inference-time video length being twice that of the train-time video length, demonstrating reasonable generalization of the consolidated cross-attention mechanism.

In contrast, neither joint space-time attention (Figure 3, middle) nor a memory-less streaming ST-ViT architecture (Figure 3, right) effectively learn from long videos. While joint-space time attention benefits from training on more frames in terms of performance, its memory footprint pre-
![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-06.jpg?height=610&width=1702&top_left_y=226&top_left_x=187)

Figure 4. MC-ViT efficiently models long videos. Fine-grained video understanding on Diving48 as a function of number of test frames (left), memory consumption (middle), and computational complexity (FLOPS, right), for joint space-time attention w/ and w/o masking (yellow and red respectively), memory-less streaming setting (green), the late temporal fusion baseline (purple) and our proposed method MC-ViT (blue). MC-ViT reaches the highest accuracy with $10 \times$ less memory and FLOPS than the joint space-time attention method.

vents it from training or evaluating on the longest videos. ST-ViT on the other hand scales to more frames, but does not benefit from them, since it lacks the ability to reason over events that span multiple segments.

![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-06.jpg?height=548&width=813&top_left_y=1222&top_left_x=190)

Figure 5. MC-ViT makes efficient use of finite-length context. We show three MC-ViT instances and compare them to relevant baselines (dashed horizontal lines). K-means (red) and coreset (orange) surpass all methods at $16 \times$ compression rate with 128 memories per segment, demonstrating the efficiency of our approach. Surprisingly, even random memory selection (blue) achieves impressive performance on this task, outperforming all baselines at $4 \times$ compression rate with 512 memories, which further showcases efficiency and robustness of the MC-ViT framework.

### 4.3. MC-ViT Efficiently Models Long Videos

We next evaluate the performance of joint space-time attention, ST-ViT, and MC-ViT in relation to their memory and computational complexity, by varying the number of frames at inference time (all models are trained with 64 frames) in Figure 4. MC-ViT's memory consumption is bounded by the number of tokens within a segment, similar to memory-less
ST-ViT, whereas that of joint space-time attention increases with video length (Figure 4, middle). Similarly, while the computational complexity of joint space-time attention is quadratic in the video length, it is linear for both ST-ViT and MC-ViT (Figure 4, right).

In terms of performance, Figure 4 demonstrates that MCViT remarkably outperforms joint space-time attention with a $10 \times$ smaller memory footprint (middle) and FLOPS (right). We additionally test other scalable baselines, such as applying $25 \%$ input token masking to joint space-time attention (Papalampidi et al., 2023), and late temporal fusion (Alayrac et al., 2022; Yan et al., 2022b), where we add a learnable module on top of ST-ViT for contextualizing information across segments (see Appendix C). Not only does MC-ViT display a better scaling behavior than these baselines (Figure 4, left), but it does so with robust improvements in memory footprint and computational complexity.

### 4.4. Memory Consolidation Makes Efficient Use of a Finite Context Window

We now analyze the computational efficiency and expressiveness of MC-ViT's consolidation methods. We compare our methods to three baselines: (1) joint space-time attention, (2) ST-ViT, and (3) MeMViT (Wu et al., 2022). Notably, MeMViT employs a parametric approach to memory compression, requiring a convolutional module to be trained alongside the network (see Appendix $\mathrm{C}$ for details). Figure 5 illustrates the performance of these methods on Diving48 as a function of the number of memories $K$ per segment. Given $K=128$ memories obtained through k-means consolidation (i.e. a $16 \times$ compression compared to MA-ViT; red curve), MC-ViT-KM outperforms all baselines. Remarkably, even random selection of $K=128$ memories (with MC-ViT-R) is sufficient to surpass ViViT and ST-ViT. Finally, consol-

Table 1. Long video question answering, compared to public models. Performance is calculated as percentage correct on multiplechoice video question answering on EgoSchema, Perception Test and Next-QA. By scaling to significantly longer videos, MC-ViT outperforms models that benefit from an order of magnitude more parameters. We highlight the best and second-best methods per dataset.

| Method | Params | Frames | EgoSchema |  | Perception Test | Next-QA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | Subset | Full |  |  |
| CoVGT (Xiao et al., 2023) | $149 \mathrm{M}$ | 32 | - | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=43&width=85&top_left_y=496&top_left_x=1387) | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=43&width=217&top_left_y=496&top_left_x=1481) | 60.0 |
| $\mathrm{SeViT}_{\text {FiD }}$ (Kim et al., 2023) | $215 \mathrm{M}$ | 10 | - | - | - | 60.6 |
| HiTeA (Ye et al., 2023) | $297 \mathrm{M}$ | 16 | - | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=43&width=85&top_left_y=565&top_left_x=1387) | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=43&width=217&top_left_y=565&top_left_x=1481) | 63.1 |
| InternVideo (Wang et al., 2022b) | $478 \mathrm{M}$ | 90 | - | 32.1 | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=43&width=217&top_left_y=600&top_left_x=1481) | 63.2 |
| ImageViT (Papalampidi et al., 2023) | 1B | 16 | 40.8 | 30.9 | 39.1 | - |
| ShortViViT (Papalampidi et al., 2023) | 1B | 16 | 47.9 | 31.0 | 41.9 | _ |
| Flamingo (Alayrac et al., 2022) | 3B | 32 | - | - | 43.6 | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-07.jpg?height=42&width=127&top_left_y=705&top_left_x=1714) |
| SeViLA Localizer + ShortViViT (Papalampidi et al., 2023) | 5B | 32 | 49.6 | 31.3 | - | - |
| LongViViT (Papalampidi et al., 2023) | 1B | 256 | 56.8 | 33.3 | 45.7 | - |
| SeViLA (Yu et al., 2023) | 4B | 32 | 25.7 | 22.7 | 46.2 | 73.8 |
| MC-ViT-B | $203 \mathrm{M}$ | $128+$ | 61.2 | 42.3 | 47.0 | 60.6 |
| MC-ViT-L | $424 \mathrm{M}$ | $128+$ | $\overline{62.6}$ | $\overline{44.4}$ | $\overline{48.1}$ | 65.0 |

Table 2. Fine-grained action classification on Diving48. Prior methods use $3 \times$ more spatial crops at inference time (SC) and/or bounding box information (BB), which MC-ViT does not require.

| Method | Params | Extra | Top-1 |
| :--- | :---: | :---: | :---: |
| TimeS-L (Bertasius et al., 2021) | $121 \mathrm{M}$ | SC | 81.0 |
| VideoSwin-B (Liu et al., 2022) | $88 \mathrm{M}$ | SC | 81.9 |
| BEVT (Wang et al., 2022) | $88 \mathrm{M}$ | SC | 86.7 |
| SIFAR-B-14 (Fan et al., 2021) | $87 \mathrm{M}$ | SC | 87.3 |
| ORViT (Herzig et al., 2022) | $160 \mathrm{M}$ | SC+BB | 88.0 |
| AIM ViT-B (Yang et al., 2023) | $97 \mathrm{M}$ | SC | 88.9 |
| AIM ViT-L (Yang et al., 2023) | $341 \mathrm{M}$ | SC | 90.6 |
| MC-ViT-B | $99 \mathrm{M}$ | $\boldsymbol{x}$ | $\mathbf{8 9 . 7}$ |
| MC-ViT-L | 313M | $\boldsymbol{x}$ | $\mathbf{9 1 . 0}$ |

idating past-activations with MC-ViT-CS (coreset, orange curve) performs similarly to MC-ViT-KM, highlighting the robustness of MC-ViT to the particular choice of memory consolidation algorithm. K-means consolidation is used as the default method given its greater computational efficiency and slightly higher performance for larger sets of memories.

### 4.5. MC-ViT Achieves State-of-the-Art Long-Context Video Understanding

Fine-grained action recognition. In Table 2, we compare MC-ViT to prior methods on Diving48, and find that it delivers state-of-the-art results. Unlike previous methods that require object tracking models (Herzig et al., 2022) or additional modeling components, MC-ViT achieves strong performance by simply re-purposing a general transformer architecture for long-context modeling: while previous methods are limited to 32 frames of video, the efficient scaling properties of MC-ViT allow it to process 128 frames. Further, MC-ViT does not require multiple spatial crops at inference time to achieve state-of-the-art results.
Long video question answering. We compare MC-ViT to prior methods on long video question answering in Table 1. We find that our approach outperforms prior works that use up to $10 \times$ more parameters. Most notably, even our smaller model version (MC-ViT-B, with 200M parameters in total) is able to achieve a $10 \%$ improvement on EgoSchema in comparison to much larger models (up to 5B parameters). This demonstrates the importance of processing more frames, which our straightforward memory consolidation method enables, as well as the effectiveness of fine-tuning $\mathrm{MC}-\mathrm{ViT}$ from standard pretrained video encoders.

It is particularly notable that $\mathrm{MC}-\mathrm{ViT}$ is competitive with models such as Flamingo (Alayrac et al., 2022) and SeViLA (Yu et al., 2023), which boast billion-scale LLM decoders. Such methods benefit from the language bias in VQA-which allows for some questions to be trivially answered without any visual input-and extensive textual training data. While MC-ViT surpasses these models on EgoSchema and Perception Test, SeViLa maintains stronger performance on Next-QA. We hypothesize that this benchmark is not challenging enough for long video understanding and relies heavily on language-only reasoning, since Yu et al. (2023) achieve their results while using a single input frame. Thus, frame-level models with strong decoders, such as $\mathrm{SeViLA}$, may be sufficient for benchmarks requiring language-only reasoning and localization (Next-QA, Perception Test), but fail to capture a summary representation of the entire video (EgoSchema). In contrast, our method, despite lacking large language decoders, performs competitively across the board, demonstrating strong localization and long-context modeling capabilities. Finally, MC-ViT requires minimal architectural changes and training overhead for adapting to long-context understanding, in contrast to modular methods (e.g., Yu et al., 2023) which involve multiple modules and complex training regimes.

Table 3. Long video question answering on EgoSchema and Perception Test, compared to large-scale proprietary models. Performance is evaluated on the original ("raw") dataset, as well as on the "visual" subset of questions that cannot be answered by a blind language model and on Perception Test for the validation set. For each model, we compute the performance of a "blind" variant on EgoSchema that only has access to question-answer pairs. The performance of the blind model is subtracted from that of the full model to compute "visual" performance. We underline the top 2 performing models for each benchmark and subset.

| Method | EgoSchema Raw |  | EgoSchema Visual |  | Perception <br> Test Raw | Perception <br> Test Visual |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Subset | Full | Subset | Full |  |  |
| Random chance | 20.0 | 20.0 | - | - | 33.3 | - |
| Bard only (blind) | 27.0 | 33.2 | 0.0 | 0.0 | 36.8 | 0.0 |
| Bard + ImageViT (Papalampidi et al., 2023) | 35.0 | 35.0 | 8.0 | 1.8 | 37.8 | 1.0 |
| Bard + ShortViViT (Papalampidi et al., 2023) | 42.0 | 36.2 | 15.0 | 3.0 | 38.8 | 2.0 |
| Bard + PALI (Papalampidi et al., 2023) | 44.8 | 39.2 | 17.8 | 6.0 | 42.4 | 5.6 |
| GPT-4 Turbo (blind) | 31.0 | 30.8 | 0.0 | 0.0 | - | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-08.jpg?height=43&width=155&top_left_y=785&top_left_x=1612) |
| GPT-4V | 63.5 | 55.6 | 32.5 | 24.8 | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-08.jpg?height=43&width=155&top_left_y=820&top_left_x=1438) | ![](https://cdn.mathpix.com/cropped/2024_05_26_02a021e8131f11acaff6g-08.jpg?height=43&width=155&top_left_y=820&top_left_x=1612) |
| Gemini Ultra (Anil et al., 2023) | - | - | - | - | 54.7 | - |
| MC-ViT-B (blind) | 18.2 | 23.4 | 0.0 | 0.0 | 37.6 | 0.0 |
| MC-ViT-B | 61.2 | 42.3 | 43.0 | 18.9 | 47.1 | 9.5 |
| MC-ViT-L (blind) | 15.0 | 22.7 | $\frac{10.0}{0.0} \quad$ | 0.0 | 35.1 | 0.0 |
| MC-ViT-L | 62.6 | 44.0 | 47.6 | $\underline{21.3}$ | 47.6 | 12.5 |

MC-ViT vs. large-scale proprietary models. Finally, in Table 3 we compare our method to large-scale proprietary systems such as GPT-4V (Achiam et al., 2023), Gemini (Anil et al., 2023) and Bard ${ }^{1}+$ PALI (Google AI, 2023; Chen et al., 2023). While their exact implementation details are not publicly available, these models are thought to contain hundreds of billions to trillions of parameters, i.e. $1000 \times$ more than MC-ViT. It is also important to note that these proprietary models are trained on massive amounts of data from the internet, resulting in potential data contamination, which we proactively avoid in our training pipeline.

In order to disentangle the natural language reasoning and visual perception capabilities of these models, we normalize model performance with respect to the performance of the equivalent "blind" model variant when possible. We present the "visual" alongside the standard "raw" performance for both benchmarks in Table 3. Examining the visual-only capabilities, we conclude that our small-scale model is competitive against the large proprietary ones and even surpasses GPT-4V performance on the subset of EgoSchema.

Despite using a fraction of the parameters and training data, our method remains competitive and, in some cases, outperforms these models. In particular, MC-ViT achieves 5\% improvements on EgoSchema and Perception Test against the sophisticated Bard + PALI modular system used for information aggregation and frame captioning, respectively.[^1]

## 5. Discussion

In this work, we introduced the Memory-Consolidated Vision Transformer (MC-ViT), which efficiently models longrange dependencies in videos by consolidating past activations into a compact memory bank. MC-ViT achieves state-of-the-art performance on multiple long video benchmarks by repurposing existing video architectures without the need for specialized architectures and training regimes. Our small-scale model outperforms approaches that benefit from orders of magnitude more parameters, and is even competitive with large-scale proprietary systems such as GPT-4V and Bard, demonstrating the importance of strong compressed video representations. As an extension, these representations could be fed into large language models to augment their long-range temporal reasoning capabilities.

We showcased the effectiveness of non-parametric memory consolidation techniques as a simple means of extending long video contexts, and future work could straightforwardly build on MC-ViT by exploring alternative consolidation strategies. For instance, incorporating insights from cognitive models of memory, such as the role of episodic and semantic memory systems, as well as theories of efficient coding (Barlow, 1961), could inspire new consolidation techniques. Furthermore, the concept of memory consolidation could be applied to other domains involving sequential data, such as natural language and audio processing, laying the foundation for personalized assistant technologies that jointly reason over multiple modalities.

## Impact Statement

By adapting standard video architectures to the long-context setup, this work could potentially equip general-purpose assistant models with the ability to efficiently process long videos. These models will likely suffer from similar biases and potential harms associated with visual language models and large language models more generally.

Further, since this work focuses on efficient processing of long sequences without sacrificing performance, the corresponding methods and findings from this work could potentially be applied to other domains, such as NLP or audio, allowing for faster processing of large amounts of data and thus making long-context model training more readily available for widespread use.

## Acknowledgements

We thank Andrew Zisserman, João Carreira, Carl Allen, and Nikhil Parthasarathy for their thoughtful feedback, Relja Arandjelović for fruitful discussions at the inception of this project, and Oliver Vikbladh, Eleanor Spens, and Neil Burgess for their insights into memory consolidation in the human mind.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Agarwal, P. K., Har-Peled, S., Varadarajan, K. R., et al. Geometric approximation via coresets. Combinatorial and Computational Geometry, 52(1):1-30, 2005.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: A visual language model for few-shot learning. Advances in Neural Information Processing Systems, 2022.

Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., and Schmid, C. ViViT: A video vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
Barlow, H. B. Possible principles underlying the transformation of sensory messages. Sensory communication, 1 $(01): 217-233,1961$.

Bartlett, F. C. Remembering: A study in experimental and social psychology. Cambridge University Press, 1932.

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. arXiv:2004.05150, 2020.

Bertasius, G., Wang, H., and Torresani, L. Is space-time attention all you need for video understanding? In International Conference on Machine Learning, 2021.

Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., and Hoffman, J. Hydra attention: Efficient attention with many heads. In European Conference on Computer Vision, 2022.

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, 2022.

Carreira, J. and Zisserman, A. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al. PaLI: A jointly-scaled multilingual language-image model. In International Conference on Learning Representations, 2023.

Dai, Z., Yang, Z., Yang, F., Cohen, W. W., and Salakhutdinov, R. R. Good semi-supervised learning that requires a bad gan. In Advances in Neural Information Processing Systems, 2017.

Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., and Salakhutdinov, R. Transformer-XL: Attentive language models beyond a fixed-length context. In Proceedings of the Association for Computational Linguistics, 2019.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344-16359, 2022.

Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., and Guo, B. CSWin transformer: A general vision transformer backbone with cross-shaped windows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M.,

Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Fan, Q., Panda, R., et al. Can an image classifier suffice for action recognition? In International Conference on Learning Representations, 2021.

Feichtenhofer, C., Fan, H., Malik, J., and He, K. SlowFast networks for video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Google AI. Bard [large language model], 2023. Accessed [Date of Access].

Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., et al. Ego4D: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Graves, A., Wayne, G., and Danihelka, I. Neural Turing machines. arXiv preprint arXiv:1410.5401, 2014.

Han, T., Xie, W., and Zisserman, A. Memory-augmented dense predictive coding for video representation learning. In Proceedings of the European Conference on Computer Vision, 2020.

Herzig, R., Ben-Avraham, E., Mangalam, K., Bar, A., Chechik, G., Rohrbach, A., Darrell, T., and Globerson, A. Object-region video transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.

Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., et al. Perceiverio: A general architecture for structured inputs \& outputs. arXiv preprint $\operatorname{arXiv:2107.14795,~} 2021$.

Jia, M., Tang, L., Chen, B.-C., Cardie, C., Belongie, S., Hariharan, B., and Lim, S.-N. Visual prompt tuning. In Proceedings of the European Conference on Computer Vision, 2022.

Kim, S., Kim, J.-H., Lee, J., and Seo, M. Semiparametric video-grounded text generation. arXiv preprint arXiv:2301.11507, 2023.
Lai, Z., Lu, E., and Xie, W. MAST: A memory-augmented self-supervised tracker. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. VideoChat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.

Li, Y., Li, Y., and Vasconcelos, N. RESOUND: Towards action recognition without representation bias. In Proceedings of the European Conference on Computer Vision, 2018.

Li, Y., Mao, H., Girshick, R., and He, K. Exploring plain vision transformer backbones for object detection. In Proceedings of the European Conference on Computer Vision, 2022a.

Li, Y., Wu, C.-Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., and Feichtenhofer, C. Mvitv2: Improved multiscale vision transformers for classification and detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4804-4814, 2022b.

Lin, K., Ahmed, F., Li, L., Lin, C.-C., Azarnasab, E., Yang, Z., Wang, J., Liang, L., Liu, Z., Lu, Y., et al. MM-VID: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023.

Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., and $\mathrm{Hu}, \mathrm{H}$. Video Swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Mangalam, K., Akshulakov, R., and Malik, J. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In Advances in Neural Information Processing Systems, 2023.

Marr, D. Simple memory: a theory for archicortex. Philosophical Transactions of the Royal Society of London, 1971.

Martins, P. H., Marinho, Z., and Martins, A. F. $\infty$-former: Infinite memory transformer. In Proceedings of the Association for Computational Linguistics, 2022.

Miech, A., Zhukov, D., Alayrac, J.-B., Tapaswi, M., Laptev, I., and Sivic, J. Howto100M: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Oh, S. W., Lee, J.-Y., Xu, N., and Kim, S. J. Video object segmentation using space-time memory networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Papalampidi, P., Koppula, S., Pathak, S., Chiu, J., Heyward, J., Patraucean, V., Shen, J., Miech, A., Zisserman, A., and Nematzdeh, A. A simple recipe for contrastively pre-training video-first encoders beyond 16 frames. arXiv preprint arXiv:2312.07395, 2023.

Pătrăucean, V., Smaira, L., Gupta, A., Continente, A. R., Markeeva, L., Banarse, D., Koppula, S., Heyward, J., Malinowski, M., Yang, Y., et al. Perception Test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023.

Piergiovanni, A., Kuo, W., and Angelova, A. Rethinking video ViTs: Sparse video tubes for joint image and video learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023a.

Piergiovanni, A., Nobel, I., Kim, D., Ryoo, M. S., Gomes, V., and Angelova, A. Mirasol3B: A multimodal autoregressive model for time-aligned and contextual modalities. arXiv preprint arXiv:2311.05698, 2023b.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Rae, J. W., Potapenko, A., Jayakumar, S. M., and Lillicrap, T. P. Compressive transformers for long-range sequence modelling. International Conference on Learning Representations, 2020.

Ryali, C., Hu, Y.-T., Bolya, D., Wei, C., Fan, H., Huang, P.-Y., Aggarwal, V., Chowdhury, A., Poursaeed, O., Hoffman, J., et al. Hiera: A hierarchical vision transformer without the bells-and-whistles. In International Conference on Machine learning, 2023.

Ryoo, M. S., Gopalakrishnan, K., Kahatapitiya, K., Xiao, T., Rao, K., Stone, A., Lu, Y., Ibarz, J., and Arnab, A. Token Turing machines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Spens, E. and Burgess, N. A generative model of memory construction and consolidation. Nature Human Behaviour, pp. 1-18, 2024.

Sun, C., Shrivastava, A., Singh, S., and Gupta, A. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE International Conference on Computer Vision, 2017.

Tulving, E. Memory and consciousness. Canadian Psychology/Psychologie canadienne, 26(1):1, 1985.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Wang, R., Chen, D., Wu, Z., Chen, Y., Dai, X., Liu, M., Jiang, Y.-G., Zhou, L., and Yuan, L. BEVT: BERT pretraining of video transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022a.

Wang, S., Zhao, Q., Do, M. Q., Agarwal, N., Lee, K., and Sun, C. Vamos: Versatile action models for video understanding. arXiv preprint arXiv:2311.13627, 2023.

Wang, W., Xie, E., Li, X., Fan, D.-P., Song, K., Liang, D., Lu, T., Luo, P., and Shao, L. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu, Y., Wang, Z., et al. InternVideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022b.

Wang, Z., Li, M., Xu, R., Zhou, L., Lei, J., Lin, X., Wang, S., Yang, Z., Zhu, C., Hoiem, D., et al. Language models with image descriptors are strong few-shot videolanguage learners. Advances in Neural Information Processing Systems, 2022c.

Wu, C.-Y., Feichtenhofer, C., Fan, H., He, K., Krahenbuhl, P., and Girshick, R. Long-term feature banks for detailed video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 284-293, 2019.

Wu, C.-Y., Li, Y., Mangalam, K., Fan, H., Xiong, B., Malik, J., and Feichtenhofer, C. MeMViT: Memoryaugmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Xiao, J., Shang, X., Yao, A., and Chua, T.-S. Next-QA: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

Xiao, J., Zhou, P., Yao, A., Li, Y., Hong, R., Yan, S., and Chua, T.-S. Contrastive video question answering via video graph transformer. arXiv preprint arXiv:2302.13668, 2023.

Xu, H., Ghosh, G., Huang, P.-Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., and Feichtenhofer, C.

VideoCLIP: Contrastive pre-training for zero-shot videotext understanding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2021a.

Xu, M., Xiong, Y., Chen, H., Li, X., Xia, W., Tu, Z., and Soatto, S. Long short-term transformer for online action detection. Advances in Neural Information Processing Systems, 2021b.

Yan, S., Xiong, X., Arnab, A., Lu, Z., Zhang, M., Sun, C., and Schmid, C. Multiview transformers for video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022a.

Yan, S., Zhu, T., Wang, Z., Cao, Y., Zhang, M., Ghosh, S., Wu, Y., and Yu, J. Video-text modeling with zeroshot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022b.

Yang, T., Zhu, Y., Xie, Y., Zhang, A., Chen, C., and Li, M. AIM: Adapting image models for efficient video action recognition. In International Conference on Learning Representations, 2023.

Ye, Q., Xu, G., Yan, M., Xu, H., Qian, Q., Zhang, J., and Huang, F. HiTeA: Hierarchical temporal-aware videolanguage pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Yu, S., Cho, J., Yadav, P., and Bansal, M. Self-chained image-language model for video localization and question answering. In Advances in Neural Information Processing Systems, 2023.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big Bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 2020.

Zeng, A., Attarian, M., Choromanski, K. M., Wong, A., Welker, S., Tombari, F., Purohit, A., Ryoo, M. S., Sindhwani, V., Lee, J., et al. Socratic models: Composing zero-shot multimodal reasoning with language. In International Conference on Learning Representations, 2022.

Zhang, C., Lu, T., Islam, M. M., Wang, Z., Yu, S., Bansal, M., and Bertasius, G. A simple llm framework for long-range video question-answering. arXiv preprint arXiv:2312.17235, 2023.
</end of paper 2>


<paper 3>
# Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context 

Gemini Team, Google ${ }^{1}$


#### Abstract

In this report, we present the latest model of the Gemini family, Gemini 1.5 Pro, a highly compute-efficient multimodal mixture-of-experts model capable of recalling and reasoning over fine-grained information from millions of tokens of context, including multiple long documents and hours of video and audio. Gemini 1.5 Pro achieves near-perfect recall on long-context retrieval tasks across modalities, improves the state-of-the-art in long-document QA, long-video QA and long-context ASR, and matches or surpasses Gemini 1.0 Ultra's state-of-the-art performance across a broad set of benchmarks. Studying the limits of Gemini 1.5 Pro's long-context ability, we find continued improvement in next-token prediction and near-perfect retrieval ( $>99 \%$ ) up to at least 10M tokens, a generational leap over existing models such as Claude 2.1 (200k) and GPT-4 Turbo (128k). Finally, we highlight surprising new capabilities of large language models at the frontier; when given a grammar manual for Kalamang, a language with fewer than 200 speakers worldwide, the model learns to translate English to Kalamang at a similar level to a person who learned from the same content.


## 1. Introduction

We present our latest multimodal model from the Gemini line: Gemini 1.5 Pro. This is our first release from Gemini 1.5, a new family of highly-capable multimodal models which incorporates a novel mixture-of-experts architecture as well as major advances in training and serving infrastructure that allow it to push the boundary of efficiency, reasoning, and long-context performance. Gemini 1.5 Pro is built to handle extremely long contexts; it has the ability to recall and reason over fine-grained information from up to at least $10 \mathrm{M}$ tokens. This scale is unprecedented among contemporary large language models (LLMs), and enables the processing of long-form mixed-modality inputs including entire collections of documents, multiple hours of video, and almost five days long of audio. Gemini 1.5 Pro surpasses Gemini 1.0 Pro and performs at a similar level to 1.0 Ultra on a wide array of benchmarks while requiring significantly less compute to train.

The ability to model data of increasingly longer contexts has tracked the development of more general and capable language models, from the now toy 2-gram language model proposed by Shannon (1948), to the modern n-gram models of the $1990 \mathrm{~s} \& 2000$ s typically constrained to 5 tokens of context (Brants et al., 2007; Chen and Goodman, 1999; Jelinek, 1998; Kneser and Ney, 1995), to recurrent neural networks language models from the 2010s which could effectively condition on hundreds of tokens (Jozefowicz et al., 2016; Mikolov et al., 2010), to the modern Transformer (Vaswani et al., 2017) which can condition on hundreds of thousands of tokens (Anthropic, 2023a). Gemini 1.5 Pro continues this trend by extending language model context lengths by over an order of magnitude. Scaling to millions of tokens, we find a continued improvement in predictive performance (Section 4.2.1.1), near perfect recall ( $>99 \%$ ) on synthetic retrieval tasks (Figure 1 and Section 4.2.1.2), and a host of surprising new capabilities like in-context learning from entire long documents (Section 4.2.2).[^0]

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-02.jpg?height=1054&width=1627&top_left_y=287&top_left_x=223)

$\square$ Video Up to 10 hours ( $9.9 \mathrm{M}$ tokens)

!í,

Audio

Up to 107 hours ( $9.7 \mathrm{M}$ tokens)

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-02.jpg?height=605&width=999&top_left_y=400&top_left_x=517)

Text Haystack

## ■!

## Text

Up to $7 \mathrm{M}$ words (10M tokens)

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-02.jpg?height=280&width=1033&top_left_y=1039&top_left_x=500)

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-02.jpg?height=257&width=243&top_left_y=1071&top_left_x=1586)

Figure 1 | Gemini 1.5 Pro achieves near-perfect "needle" recall (>99.7\%) up to 1M tokens of "haystack" in all modalities, i.e., text, video and audio. It even maintains this recall performance when extending to $10 \mathrm{M}$ tokens in the text modality (approximately $7 \mathrm{M}$ words); $9.7 \mathrm{M}$ tokens in the audio modality (up to 107 hours); $9.9 \mathrm{M}$ tokens in the video modality (up to 10.5 hours). The $\mathrm{x}$-axis represents the context window, and the $y$-axis the depth percentage of the needle placed for a given context length. The results are color-coded to indicate: green for successful retrievals and red for unsuccessful ones.

To measure the effectiveness of our model's long-context capabilities, we conduct experiments on both synthetic and real-world tasks. In synthetic "needle-in-a-haystack" tasks inspired by Kamradt (2023) that probe how reliably the model can recall information amidst distractor context, we find that Gemini 1.5 Pro achieves near-perfect ( $>99 \%$ ) "needle" recall up to multiple millions of tokens of "haystack" in all modalities, i.e., text, video and audio, and even maintaining this recall performance when extending to $10 \mathrm{M}$ tokens in the all three modalities. In more realistic multimodal long-context benchmarks which require retrieval and reasoning over multiple parts of the context (such as answering questions from long documents or long videos), we also see Gemini 1.5 Pro outperforming all competing models across all modalities even when these models are augmented with external retrieval methods. Finally, we qualitatively showcase the in-context learning abilities of Gemini 1.5 Pro enabled by very long context: for example, learning to translate a new language from a single set of linguistic documentation. With only instructional materials (a 500-page reference grammar, a dictionary, and $\approx 400$ extra parallel sentences) all provided in context, Gemini 1.5 Pro is capable of learning to translate from English to Kalamang, a Papuan language with fewer than 200 speakers $^{2}$, and therefore almost no online presence. Moreover, we find that the quality of its translations is comparable to that of a person who learned from the same materials.[^1]

| Gemini 1.5 Pro | Relative to 1.0 Pro | Relative to 1.0 Ultra |
| :---: | :---: | :---: |
| Long-Context Text, <br> Video \& Audio | from 32k up to 10M tokens | from 32k up to $10 \mathrm{M}$ tokens |
| Core Capabilities | Win-rate: $87.9 \%$ | Win-rate: $57.6 \%$ |
|  | $(29 / 33$ benchmarks) | (19/33 benchmarks) |
| Text | Win-rate: $100 \%$ | Win-rate: $80 \%$ |
|  | (15/15 benchmarks) | (12/15 benchmarks) |
| Vision | Win-rate: $77 \%$ | Win-rate: $46 \%$ |
|  | (10/13 benchmarks) | (6/13 benchmarks) |
| Audio | Win-rate: $60 \%$ | Win-rate: $20 \%$ |
|  | (3/5 benchmarks) | (1/5 benchmarks) |

Table 1 | Gemini 1.5 Pro compared to Gemini 1.0 family. Gemini 1.5 Pro maintains high levels of performance even as its context window increases. Detailed results are presented in Table 7.

Importantly, this leap in long-context performance does not come at the expense of the core multi-modal capabilities of the model. ${ }^{3}$ Overall, we find that Gemini 1.5 Pro greatly surpasses Gemini 1.0 Pro, performing better on the vast majority of benchmarks (i.e., 29/33), increasing the margin in particular for Math, Science and Reasoning ( $+38.4 \%$ ), Multilinguality ( $+22.3 \%$ ), Video Understanding ( $+16.9 \%$ ), Image Understanding ( $+6.5 \%$ ), and Code ( $+8.9 \%$ ) (see Table 7 for breakdowns). However, a more striking comparison is the one with Gemini 1.0 Ultra, a state-of-the-art model across many capabilities. Despite Gemini 1.5 Pro using significantly less training compute and being more efficient to serve, we find Gemini 1.5 Pro to perform better on more than half of the benchmarks (19/33), in particular on text (12/15) and many of the vision benchmarks (6/13).

In the following sections, we provide an overview of the model architecture and present the results of large-scale quantitative evaluations comparing Gemini 1.5 Pro to other LLMs. We present detailed evaluations for the model's long context capabilities followed by evaluations of its core capabilities, similar to the Gemini 1.0 Technical Report (Gemini-Team et al., 2023), covering wellstudied benchmarks across text, code, image, video and audio. Finally, we discuss our approach to responsible deployment, including our process for impact assessment developing model policies, evaluations, and mitigations of harm before deployment decisions. ${ }^{4}$

## 2. Model Architecture

Gemini 1.5 Pro is a sparse mixture-of-expert (MoE) Transformer-based model that builds on Gemini 1.0's (Gemini-Team et al., 2023) research advances and multimodal capabilities. Gemini 1.5 Pro also builds on a much longer history of MoE research at Google (Clark et al., 2022; Du et al., 2022; Fedus et al., 2021; Lepikhin et al., 2020; Riquelme et al., 2021; Shazeer et al., 2017; Zoph et al., 2022) and language model research in the broader literature (Anil et al., 2023; Anthropic, 2023a; Brown et al., 2020; Chowdhery et al., 2023; Hoffmann et al., 2022; Jiang et al., 2024; Kim et al., 2021; OpenAI, 2023; Rae et al., 2021; Raffel et al., 2020; Roller et al., 2021; Thoppilan et al., 2022; Touvron et al., 2023a,b; Vaswani et al., 2017). MoE models use a learned routing function to direct inputs to a subset of the model's parameters for processing. This form of conditional computation (Bengio et al., 2013; Davis and Arel, 2014; Jacobs et al., 1991) allows models to grow their total parameter count while keeping the number of parameters that are activated for any given input constant.[^2]

A host of improvements made across nearly the entire model stack (architecture, data, optimization and systems) allows Gemini 1.5 Pro to achieve comparable quality to Gemini 1.0 Ultra (see Section 5), while using significantly less training compute and being significantly more efficient to serve. Gemini 1.5 Pro also incorporates a series of significant architecture changes that enable long-context understanding of inputs up to 10 million tokens without degrading performance. Translated into real world data, this context length enables Gemini 1.5 Pro models to comfortably process almost five days of audio recordings (i.e., 107 hours), more than ten times the entirety of the 1440 page book (or 587,287 words) "War and Peace", the entire Flax (Heek et al., 2023) codebase (41,070 lines of code), or 10.5 hours of video at 1 frame-per-second. Further, since the model is natively multimodal and supports interleaving of data from different modalities, it can support a mix of audio, visual, text, and code inputs in the same input sequence. In Section 4.1, we highlight some of the novel capabilities enabled by these advances, including evaluations that yielded positive results on context lengths up to 10 million. We note that understanding the limits of these capabilities and studying their exciting capabilities and applications remains an area of continued research exploration.

## 3. Training Infrastructure and Dataset

Like Gemini 1.0 Ultra and 1.0 Pro, Gemini 1.5 Pro is trained on multiple 4096-chip pods of Google's TPUv4 accelerators, distributed across multiple datacenters, and on a variety of multimodal and multilingual data. Our pre-training dataset includes data sourced across many different domains, including web documents and code, and incorporates image, audio, and video content. For the instruction-tuning phase we finetuned Gemini 1.5 Pro on a collection of multimodal data (containing paired instructions and appropriate responses), with further tuning based on human preference data. We refer readers to the Gemini 1.0 Technical Report (Gemini-Team et al., 2023) for further information.

## 4. Long-context Evaluation

Existing evaluations are increasingly strained by the new and rapidly advancing capabilities of large multimodal models. They typically focus on individual modalities and/or are restricted to tasks with shorter context lengths. Hence, there is a growing need for benchmarks which exemplify the nuanced requirements of real world long mixed-modality use cases. Among these, we highlight the quantitative assessment of reasoning capabilities across long mixed-modality sequences as a key challenge.

With the challenges of evaluating increasingly capable models in mind, our evaluation of Gemini 1.5 Pro first focuses on understanding and evaluating its novel capabilities. Subsequently, we explore core benchmarks, covering capabilities studied in the Gemini 1.0 Technical Report (Gemini-Team et al., 2023). Specifically, we evaluate Gemini 1.5 Pro in three main categories: ${ }^{5}$

1. Qualitative long-context multimodal evaluations: manually probe and stress-test the model's long-context abilities, especially for novel capabilities where no quantitative benchmarks exist.
2. Quantitative long-context multimodal evaluations: measure the model's long-context abilities on both synthetic and real-world tasks with well-defined metrics.
3. Quantitative core evaluations: identify progress and regression in core capabilities (e.g., coding, math, science, multilinguality and instruction following).[^3]

### 4.1. Qualitative Examples of Multimodal Long-Context Capabilities

The ability to process multiple millions of tokens unlocks practical applications that were not possible before. In this section we demonstrate some surprising interactions we observed with Gemini 1.5 Pro across code, text and video. ${ }^{6}$

As shown in the Figure 2, Gemini 1.5 Pro is able to ingest entire large codebases such as JAX (746,152 tokens), and answer very specific queries about them. in Figure 3 we show Gemini 1.5 Pro's ability to learn a new language based only on reference materials given in its input (see Section 4.2.2.1 for quantitative metrics for this use case). Additionally, we test Gemini 1.5 Pro's ability to answer an image query given the entire text of Les Misérables and observe that being natively multimodal allows it to locate a famous scene from a hand-drawn sketch, as shown in Figure 4. Lastly, we ask Gemini 1.5 Pro questions about an entire movie of 45 minutes in Figure 5 which the model answers seamlessly while retrieving moments and timestamps down to a second.

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-05.jpg?height=485&width=1516&top_left_y=954&top_left_x=270)

Figure 2 | Given the entire 746,152 token JAX codebase in context, Gemini 1.5 Pro can identify the specific location of a core automatic differentiation method.

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-05.jpg?height=477&width=1285&top_left_y=1675&top_left_x=385)

Figure 3 | Given a reference grammar book and a bilingual wordlist (dictionary), Gemini 1.5 Pro is able to translate from English to Kalamang with similar quality to a human who learned from the same materials.[^4]

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-06.jpg?height=551&width=1280&top_left_y=336&top_left_x=388)

Figure 4 | With the entire text of Les Misérables in the prompt (1382 pages, 732k tokens), Gemini 1.5 Pro is able to identify and locate a famous scene from a hand-drawn sketch.
![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-06.jpg?height=1134&width=1170&top_left_y=1142&top_left_x=448)

Figure 5 | When prompted with a 45 minute Buster Keaton movie "Sherlock Jr." (1924) (2,674 frames at 1 FPS, 684k tokens), Gemini 1.5 Pro retrieves and extracts textual information from a specific frame in and provides the corresponding timestamp. At bottom right, the model identifies a scene in the movie from a hand-drawn sketch.

### 4.2. Long-context Evaluations

For the past few years, LLM research has prioritized expanding the context window from which models can incorporate information (Anthropic, 2023a; OpenAI, 2023). This emphasis stems from the recognition that a wider context window allows models to incorporate a larger amount of new, taskspecific information not found in the training data at inference time, leading to improved performance in various natural language or multimodal tasks. Recent approaches to improving the long-context capabilities of models fall into a few categories, including novel architectural approaches (Ainslie et al., 2023; Gu and Dao, 2023; Guo et al., 2021; Orvieto et al., 2023; Zaheer et al., 2020), post-training modifications (Bertsch et al., 2023; Chen et al., 2023; Press et al., 2021; Xiong et al., 2023), retrievalaugmented models (Guu et al., 2020; Izacard et al., 2022; Jiang et al., 2022; Karpukhin et al., 2020; Santhanam et al., 2021), memory-augmented models (Bulatov et al., 2022, 2023; Martins et al., 2022; Mu et al., 2023; Wu et al., 2022a,b; Zhong et al., 2022), and techniques for building more coherent long-context datasets (Shi et al., 2023b; Staniszewski et al., 2023). This activity has resulted in measurable improvements on long-context capabilities of LLMs over the past several months, with the recent concurrent work of Liu et al. (2024) exploring context window of 7B models up to $1 \mathrm{M}$ multimodal tokens. Notably, among the state-of-the-art LLMs, Anthropic has successfully extended the context of their text-only Claude 2 model to 100k tokens, while OpenAI has recently released GPT-4 Turbo reaching 128k tokens. Finally, the latest addition to the series was Claude 2.1 with a context window of $200 \mathrm{k}$ tokens.

Gemini 1.5 Pro significantly extends this context length frontier to multiple millions of tokens with almost no degradation in performance, making it possible to process significantly larger inputs. Compared to Claude 2.1 with a 200k token context window, Gemini 1.5 Pro achieves a $100 \%$ recall at $200 \mathrm{k}$ tokens, surpassing Claude 2.1's $98 \%$. This $100 \%$ recall is maintained up to $530 \mathrm{k}$ tokens, and recall is $99.7 \%$ at $1 \mathrm{M}$ tokens. When increasing from $1 \mathrm{M}$ tokens to $10 \mathrm{M}$ tokens, the model retains $99.2 \%$ recall. Moreover, Gemini 1.5 Pro's native multimodal capabilities enables the model to ingest multiple hours of audio and video recordings alongside or interleaved with text. Such recall capabilities are summarized in Figure 1. Below we report results on long-context evaluations across all three modalities, i.e., text, vision and audio.

The evaluation methodology we followed to measure the long-context capability of Gemini 1.5 Pro consists of both diagnostic-focused probing of the long context capabilities (e.g., perplexity over long sequences, needle-in-a-haystack retrieval studies) and realistic evaluations specifically designed for multimodal long-context tasks (e.g., long-document QA, long-context automatic speech recognition, learning to translate a new language from only one book, and long-context video QA). To provide a reference point, throughout this section we compare Gemini 1.5 Pro with the leading model available externally for each task. With the evaluation harness we developed for Gemini 1.5 Pro we are able to quantify the quality of long-context understanding capabilities reliably all the way up to $10 \mathrm{M}$ tokens.

### 4.2.1. Diagnostic Long-Context Evaluations

### 4.2.1.1 Perplexity over Long Sequences

We start by reporting results on the text modality. To evaluate the ability of the models to make use of very long contexts to improve next-token prediction, which is the objective function used to train language models, we record the negative log-likelihood (NLL) of tokens at different positions in the input sequences from held-out text (i.e., not used in training). Here, a lower value implies an improved prediction. Typically, we expect tokens at the beginning of a sequence to have high NLL, as there is little to no context that the model can use to predict them, and tokens later in the sequence to have lower NLL as more information becomes available to the model. The shape of the resulting
![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-08.jpg?height=548&width=1602&top_left_y=277&top_left_x=226)

Figure 6 | Cumulative average negative log-likelihood (NLL) as a function of token position in long documents and code data. A lower value demonstrates better prediction. Gemini 1.5 Pro shows improved predictions up to $1 \mathrm{M}$ tokens for long-documents and $10 \mathrm{M}$ tokens for code, whereas Gemini 1.0 Pro improves up to only $32 \mathrm{~K}$ tokens. The NLL follows a power-law trend up until $1 \mathrm{M}$ tokens (documents) and $2 \mathrm{M}$ tokens (code) with a deviating trend at $10 \mathrm{M}$ tokens.

curve indicates the abilities of models to reason over long-context. A downward trend signifies models making use of long-context to reduce models' uncertainty. On the other hand, an upward trend signifies that models are unable to effectively use information from the previous context and may be deteriorating in prediction quality, highlighting the limitations in their long-context understanding capability.

We perform this analysis on two data sources: (a) a dataset of long documents with up to 1 million tokens, and (b) a dataset of code repositories constructed by first randomly shuffling all the files and then concatenating them. The code dataset contains sequences longer than 1 million tokens with some natural form of semantic association (e.g., a whole repository), allowing for further evaluation of sequences of up to 10M tokens. Figure 6 shows the cumulative NLL up to a specific token index. ${ }^{7}$ We also fit a power law of the form $L(x)=\alpha x^{\beta}+\gamma$ to these data points (dashed line).

We find in Figure 6 that NLL decreases monotonically with sequence length and thus prediction accuracy improves up to the tested sequence lengths ( $1 \mathrm{M}$ for long documents, and $10 \mathrm{M}$ for code), indicating that our models can make use of the whole input even at very long-context lengths. This suggests that the model is able to improve its predictions by finding useful patterns in tokens, even if they occurred millions of tokens in the past, as in the case of code.

Finally, we see this improved prediction follows a regular power-law structure. While it is well known that language models follow a power-law in terms of training compute to model performance (NLL) (Kaplan et al., 2020) up to a very large scale, we demonstrate that a power law can hold between log-loss and context length up to extremely long context lengths. We see the power-law fit is quite accurate up to $1 \mathrm{M}$ tokens for long-documents and about $2 \mathrm{M}$ tokens for code. From inspecting longer code token predictions closer to $10 \mathrm{M}$, we see a phenomena of the increased context occasionally providing outsized benefit (e.g. due to repetition of code blocks) which may explain the power-law deviation. However this deserves further study, and may be dependent on the exact dataset used.[^5]![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-09.jpg?height=712&width=1654&top_left_y=286&top_left_x=201)

Figure 7 | Text Haystack. This figure compares Gemini 1.5 Pro with GPT-4 Turbo for the text needle-in-a-haystack task. Green cells indicate the model successfully retrieved the secret number, gray cells indicate API errors, and red cells indicate that the model response did not contain the secret number. The top row shows results for Gemini 1.5 Pro, from $1 \mathrm{k}$ to $1 \mathrm{M}$ tokens (top left), and from $1 \mathrm{M}$ to 10M tokens (top right). The bottom row shows results on GPT-4 Turbo up to the maximum supported context length of $128 \mathrm{k}$ tokens.

### 4.2.1.2 Text Haystack

Next, we move to testing long-context recall using the recently introduced needle-in-a-haystack evaluation (Kamradt, 2023), which tests a model's ability to retrieve a text (i.e., "needle") inserted at various positions into a sequence (i.e., "haystack"). Following prior work (Dhinakaran, 2024), we use a set of concatenated and repeated essays written by Paul Graham $^{8}$ to fill the desired context length. We insert a needle at linearly spaced intervals from the beginning to the end of the context, where the needle is i.e., "The special magic \{city\} number is: \{number\}" where the city and number are varied for each query, and query the model to return the magic number for a specific city. We report whether the magic number recall was correct at various context lengths ( $\mathrm{x}$ axis - the haystack) as a function of its position in the input sequence expressed in terms of depth percentage (y axis), e.g., depth at $100 \%$ would indicate a needle inserted at the very end of the input whereas $0 \%$ at the very beginning.

As can be seen in Figure 7, Gemini 1.5 Pro achieves $100 \%$ recall up to 530k tokens and $>99.7 \%$ recall up to $1 \mathrm{M}$ tokens. This task, while simple, provides a clear demonstration that Gemini 1.5 Pro is able to reliably retrieve information from long documents up to $1 \mathrm{M}$ tokens. For reference, we report results for GPT-4 Turbo up to the $128 \mathrm{~K}$ sequence length supported by their API. In order to test whether the capabilities demonstrated in the perplexity plots in Figure 6 transfer to sampling tasks, we continue to evaluate Gemini 1.5 Pro on the needle-in-a-haystack task beyond $1 \mathrm{M}$ tokens. The results in Fig 7 show that the model is still able to find and extract information with $99.2 \%$ accuracy up to $10 \mathrm{M}$ tokens.[^6]![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-10.jpg?height=744&width=1658&top_left_y=284&top_left_x=202)

Figure 8 | This figure compares Gemini 1.5 Pro with GPT-4V for the video needle-in-a-haystack task, where the models are given video clips of different lengths up to 10.5 hours of video and are asked to retrieve a secret word embedded as text at different points within the clip. All video clips are sampled at one frame-per-second ( $1 \mathrm{fps}$ ). The first pair of $10 \times 50$ haystack plots on the left compare Gemini 1.5 Pro with GPT-4V on the first hour of the AlphaGo documentary. The x-axis represents the video duration which ranges from 1.2 minutes to 1 hour, and the y-axis represents the depth, namely the relative offset of the needle (e.g., the top left cell represents providing the model with the first 1.2 minutes and inserting the needle in a randomly sampled frame in the first seven seconds of that trimmed video). A green cell indicates that the model successfully retrieved the needle, whereas a gray cell indicates an API error. Whereas the GPT-4V API supports video lengths only up to around the first 3 minutes, Gemini 1.5 Pro successfully retrieves the secret word inserted at all depth percentages for the full hour, as shown by the all-green plot. Finally, the $10 \times 10$ grid on the right shows Gemini 1.5 Pro's perfect retrieval capabilities across 10.5 hours of video, constructed by concatenating seven copies of the AlphaGo documentary back-to-back.

### 4.2.1.3 Video Haystack

As Gemini 1.5 Pro is natively multimodal, its long-context abilities translate directly to other modalities, enabling it to retrieve specific information across multiple hours of video. To test this capability, we adapt the text needle-in-a-haystack evaluation and turn it into a cross-modal evaluation, wherein a needle is hidden in one modality while the retrieval query is given in text. Rather than asking the model to retrieve a randomly inserted phrase from a corpus of text, we ask the model to retrieve information embedded in a random frame (the "needle") in a 10.5 -hour-long video (the "haystack") that is sampled at one frame-per-second.

Concretely, we overlay the text "The secret word is "needle" on a single randomly sampled video frame in a 10.5 hour video constructed from concatenating seven copies of the full AlphaGo documentary (Kohs, 2017) back-to-back (for a total of 37994 frames, or 9.9M tokens). See Figure 15 in the Appendix for an example of such an embedded frame. After feeding it the video, we ask the model to answer the question "What is the secret word?". As Figure 8 shows, Gemini 1.5 Pro successfully answers this question across a breadth of video lengths and a range of randomly inserted needle locations in the 10.5 hour video. In contrast, the GPT-4V API supports video lengths only up to around the first 3 minutes.
![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-11.jpg?height=734&width=1654&top_left_y=281&top_left_x=201)

Figure 9 | Audio Haystack. This figure presents the audio version of the needle-in-a-haystack experiment comparing Gemini 1.5 Pro and a combination of Whisper and GPT-4 Turbo. In this setting, the needle is a short segment of audio that is inserted within a very large audio segment (of up to 107 hours) containing concatenated audio clips. The task is to retrieve the "secret keyword" which is revealed in the needle. Red indicates that the model did not identify the keyword, whereas green indicates that the model identified the keyword correctly.

### 4.2.1.4 Audio Haystack

We follow a similar strategy for testing Gemini 1.5 Pro's long context capabilities on audio understanding. We hide a very short clip of audio lasting a few seconds where a speaker says "the secret keyword is needle"

within an audio signal (the haystack) up to almost five days long (i.e., 107 hours). The task for the model is then to retrieve the secret keyword, given a question in text, hence requiring cross-modal reasoning. To further challenge the model beyond increasing context, the large audio signal is built from an unlabeled speech corpus from the VoxPopuli dataset (Wang et al., 2021) so that the input signal contains multiple speakers. In Figure 9 we plot the result of the experiment when the input audio ranges from 12 minutes to 107 hours (or 9.9M tokens), inserting the needle in different positions across the signal. The red boxes indicate a score of 0.0 (meaning the model did not identify the keyword), and green indicates a score of 1.0 (meaning the model identified the keyword correctly). The model succeeds at finding the secret keyword in all instances, with the overall accuracy of Gemini 1.5 Pro on this task being $100 \%$.

Unlike Gemini 1.5 Pro, existing models cannot natively handle more than a few seconds of audio in the context. As such, in order to fairly compare against them we need to employ a strategy where we first transcribe audio into text using windows of tens of seconds, and then rely on text models to extend beyond that limited window.

Specifically, to compare against Whisper, we chunk the audio input into 30 second segments, transcribe the audio using the model to produce a text transcript, concatenate the transcripts for each chunk, and finally prompt GPT-4 Turbo to find the "secret keyword" given the text transcript. Figure 9 shows the performance for each depth percent and number of hours. The overall accuracy of Whisper combined with GPT-4 Turbo to identify the needle is around $94.5 \%$.

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-12.jpg?height=596&width=1324&top_left_y=293&top_left_x=366)

Figure 10| Retrieval performance of the "multiple needles-in-haystack" task, which requires retrieving 100 unique needles in a single turn. When comparing Gemini 1.5 Pro to GPT-4 Turbo we observe higher recall at shorter context lengths, and a very small decrease in recall towards $1 \mathrm{M}$ tokens.

### 4.2.1.5 Improved Diagnostics

Despite the excellent performance of Gemini 1.5 Pro model on the needle-in-a-haystack tasks for all three modalities, significantly surpassing previously reported results ( $>99.7 \%$ for text, $100 \%$ for video and $100 \%$ for audio), we also present early findings of observed limitations. By design, the needle-in-a-haystack task is a retrieval task measuring recall and so far we have considered the simplest possible setup. A natural extension to the task is to increase the number of unique "needles" in each haystack, and require the model to retrieve them all. For a context length of up to $1 \mathrm{M}$ tokens, we inserted 100 different needles and measured the total number of correct needles retrieved.

Figure 10 compares the recall of Gemini 1.5 Pro and GPT-4 Turbo on this task. We see an improved recall from Gemini 1.5 Pro over GPT-4 Turbo up until 128K tokens. It is important to note that GPT-4 Turbo's context length is limited to $128 \mathrm{~K}$ tokens and its retrieval quality largely oscillates with longer context lengths with an average recall of around $50 \%$ at $128 \mathrm{~K}$ tokens. In contrast, Gemini 1.5 Pro maintains around $70 \%$ recall up to $128 \mathrm{~K}$ tokens, and $>60 \%$ recall up to $1 \mathrm{M}$ tokens. We report further results on this task in the Appendix 9.2 including results with different numbers of needles, where we observe consistent trends.

In line with other tasks in the literature of LLMs, we also observe that the choice of the prompting method and type of needle affect final performance of models, and future versions of "needle(s)-in-ahaystack" style tests should account for prompt robustness.

We also modulate retrieval difficulty on another axis: the similarity of the needles. In the Multiround Co-reference Resolution (MRCR) task, the model is presented with a long conversation between a user and a model, in which the user requests writing (e.g. poems, riddles, essays) on different topics proceeded by the model responses. In each conversation, two user requests containing topics and writing formats distinct from the rest of the conversation are randomly placed in the context. Given the conversation, the model must reproduce the model's output (the needle) resulting from one of the two requests (the key). Either the formats, the topics, or both, overlap in order to create a single key that is adversarially similar to the query key. For instance, the request "Reproduce the poem about penguins." requires the model to distinguish the poem about penguins from the poem about flamingos, and "Reproduce the first poem about penguins." requires the model to reason about ordering. We score MRCR via a string-similarity measure between the model output and the correct

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-13.jpg?height=709&width=1150&top_left_y=288&top_left_x=453)

Figure 11 | Cumulative average string similarity score as a function of context length over 2000 instances of the MRCR task. When comparing Gemini 1.5 Pro to GPT-4 Turbo and Claude 2.1, we observe that after $8 \mathrm{~K}$ tokens, Gemini 1.5 Pro substantially outperforms both models. Gemini 1.5 Pro has a substantially smaller decrease in performance as a function of context length up to $1 \mathrm{M}$ tokens.

response. ${ }^{9}$

Figure 11 compares the ability of Gemini 1.5 Pro, GPT-4 Turbo, and Claude 2.1 on MRCR. Gemini 1.5 Pro overtakes GPT-4 Turbo at around $8 \mathrm{~K}$ tokens and achieves an average score of around $80 \%$ at 1M tokens. GPT-4 Turbo's performance falls off steadily as a function of context length, all the way up to its limit at $128 \mathrm{~K}$ tokens, where it scores around $60 \%$. Claude 2.1 (with context length going up to $200 \mathrm{~K}$ tokens) scores around $20 \%$ at $128 \mathrm{~K}$ tokens and under performs the other models by hallucinating that the needle is not in context and punting on requests to solve the task, despite following Claude 2.1 prompting guidelines for retrieval tasks (Anthropic, 2023b).

We highlight that "multiple needles-in-a-haystack" and MRCR capture different aspects of the retrieval task: MRCR is harder and requires stronger reasoning and disambiguation skills, while the multiple needles challenge is a test of the model's recall ability, explaining disparities between the model orderings up to $8 \mathrm{~K}$ tokens. Gemini 1.5 Pro impressively avoids serious degradation on both of these axes all the way up to $1 \mathrm{M}$ tokens.

While the "multiple needles-in-a-haystack" and MRCR evaluations offer two challenging setups that stress-test long-context retrieval capabilities in different ways, we advocate for pushing the boundaries even further. Evaluating models on tasks that demand complex reasoning over multiple pieces of information scattered across a long context would provide even deeper insights into their true capabilities. This could involve tasks that require integrating disparate facts, drawing inferences, or resolving inconsistencies within the retrieved information. By incorporating such assessments alongside prompt robustness studies, we can gain a more comprehensive and nuanced understanding of how effectively models can utilize long contexts for advanced reasoning and knowledge extraction.[^7]

### 4.2.2. Realistic Long-Context Evaluations

Having investigated the model's multimodal abilities on handling long-context using a battery of diagnostic tests, we now turn to a series of novel multimodal tasks designed to better reflect the potential uses of this model, thus stress-testing models in a more realistic way.

### 4.2.2.1 In-context language learning - learning to translate a new language from one book

To stress-test the in-context learning abilities enabled by very long context, we evaluate Gemini 1.5 Pro on the Machine Translation from One Book (MTOB) benchmark (Tanzer et al., 2023). MTOB measures the ability to learn to perform sentence-level translation between English and Kalamang (ISO 639-3 language code: $\mathrm{kgv}$ ) from instructional materials. Kalamang has fewer than 200 speakers and therefore virtually no presence on the web, which means that the model must rely on the data given in context (rather than knowledge stored in its weights at training time). ${ }^{10}$ The available resources for Kalamang are: field linguistics documentation ${ }^{11}$ comprising a $\sim 500$ page reference grammar (Visser, 2020b), a $\sim 2000$-entry bilingual wordlist (Visser, 2020a), and a set of $\sim 400$ additional parallel sentences (Visser, 2020a). In total the available resources for Kalamang add up to around $\sim 250 \mathrm{k}$ tokens. This task framing offers the promise of using extremely long-context models to support languages that are not sufficiently represented in pre-training corpora, with curated resources that can be created and deployed by independent parties.

To perform the task, we provide Gemini 1.5 Pro with the full set of materials in its input context. To compare fairly against Claude 2.1 and GPT-4 Turbo, since the full materials do not fit in their context windows, we also report results using only half of the grammar book ( 100k tokens). Moreover, to test to what extent Gemini 1.5 Pro is making use of information in the context, rather than relying on knowledge found on its pre-training data, we also run a 0-shot setup. Finally, we compare to MTOB's reference for human performance, in which a human learned Kalamang from the exact same full set of materials. ${ }^{12}$

To assess performance, we conduct a human evaluation where the same human language learner is given the input sentence and reference translation, and rates the quality of the predicted translation on a scale from 0 to 6 , with 6 being an excellent translation. This rater is a non-native non-fluent speaker who can identify their own translations, so the scores should be interpreted in context. We additionally report results using automatic metrics, i.e., BLEURT (Sellam et al., 2020) for Kalamang to English (kgv $\rightarrow$ eng) and chrF (Popović, 2015) for English to Kalamang (eng $\rightarrow \mathrm{kgv}$ ).

Gemini 1.5 Pro, GPT-4 Turbo, and Claude 2.1 all have essentially random performance in the 0 -shot setting (i.e., no additional Kalamang information in context). Gemini 1.5 Pro achieves a 0.24 human evaluation score in $\mathrm{kgv} \rightarrow$ eng and 0.08 in eng $\rightarrow \mathrm{kgv}$. It sometimes successfully copies proper nouns, identifies loanwords from higher resource languages like Malay, or narrows generation using style cues like question marks. Its generations for eng $\rightarrow \mathrm{kgv}$ are identified by Google Translate as[^8]

| Model | $\mathrm{kgv} \rightarrow$ eng <br> Human Evaluation (BLEURT) | eng $\rightarrow \mathrm{kgv}$ <br> Human Evaluation (chrF) |
| :--- | :---: | :---: |
| GPT-4 Turbo (0-shot) | $0.24(33.1)$ | $0.10(17.8)$ |
| GPT-4 Turbo (half book) | $2.38(51.6)$ | $4.02(48.3)$ |
| Claude 2.1 (0-shot) | $0.14(22.2)$ | $0.00(15.3)$ |
| Claude 2.1 (half book) | $3.68(57.1)$ | $4.54(52.5)$ |
| Gemini 1.5 Pro (0-shot) | $0.24(33.3)$ | $0.08(17.8)$ |
| Gemini 1.5 Pro (half book) | $4.16(63.4)$ | $5.38(58.3)$ |
| Gemini 1.5 Pro (full book) | $4.36(65.0)$ | $5.52(56.9)$ |
| Human language learner | $5.52(70.3)$ | $5.60(57.0)$ |

Table 2 | Quantitative results for Kalamang $\leftrightarrow$ English translation on MTOB (Tanzer et al., 2023). We present human evaluation scores on a scale of 0 to 6 , with 6 being an excellent translation. We include automatic metrics (BLEURT/chrF) in parentheses.

various other languages, often malformed. These results indicate that, as expected, no substantial Kalamang data, if any, was part of the model's training data.

Gemini 1.5 Pro in the half book setting outperforms GPT-4 Turbo and Claude 2.1 on the same setup by a wide margin, with further gains when given the entire book in context, see Table 2. In the full book setting, Gemini 1.5 Pro attains a 4.36 human evaluation score on $\mathrm{kgv} \rightarrow$ eng translation, compared to 5.52 of the "human language learner" score, and 5.52 on eng $\rightarrow \mathrm{kgv}$ translation, compared to 5.58 by the "human language learner". There is still a substantial qualitative gap for $\mathrm{kgv} \rightarrow$ eng translation, but eng $\rightarrow \mathrm{kgv}$ translation is similar to the human language learner on average. ${ }^{13}$

See Table 2 for the human evaluation scores, Table 3 for a qualitative translation example, and Appendix 9.8 for more details, experiments, and examples.

The performance of Gemini 1.5 Pro highlights the importance of long-context understanding and providing sufficient context for learning new skills in-context. By leveraging the extensive field linguistics documentation provided in context, Gemini 1.5 Pro was able to achieve remarkable translation quality comparable to a human language learner, and was able to do so for a language for which it had essentially zero exposure to during the training of the model. This finding opens up exciting possibilities for leveraging LLMs with sufficiently long-context capabilities to support the preservation and revitalization of endangered languages, as well as to facilitate communication and understanding across different linguistic communities. As research continues in this area, it will be crucial to explore techniques for improving the quality of translation in both directions, e.g., Kalamang-to-English, and to address the challenges of evaluating the performance of LLMs on low-resource and under-represented languages, which we believe is also applicable to other areas of education and language learning.[^9]

| Input | Bayu esun et mulko mambaran. |
| :--- | :--- |
| GPT-4 Turbo (half book) | Bayu's father is cutting a canoe. |
| Claude 2.1 (half book) | Bayu's father's canoe stands at the side. |
| Gemini 1.5 Pro (0-shot) | The wind blows, and the world is vast. |
| Gemini 1.5 Pro (half book) | Bayu's father stands beside the canoe. |
| Gemini 1.5 Pro (full book) | Bayu's father is standing by the canoe. |
| Human language learner | Bayu's father stands to the side of the canoe. |
| Reference | Bayu's father stands next to the canoe. |

Table 3 | Qualitative example of $\mathrm{kgv} \rightarrow$ eng translation. More examples in Appendix 9.8.

### 4.2.2.2 Long-document $Q A$

After testing Gemini 1.5 Pro's in-context language learning capabilities up to $250 \mathrm{k}$ tokens, we proceed into another realistic evaluation setup. In this section we present experiments on question answering, we create questions using the book "Les Misérables" (by Victor Hugo) and test the model's ability to answer them correctly when the entire 1,462 page book (i.e., $710 \mathrm{~K}$ tokens) is provided as input. Evaluating a model's ability to answer questions about long documents (or collections of documents) presents a unique challenge. Unlike tasks that focus on specific facts or details that measure the retrieval capability of the models, such questions often require understanding relationships between pieces of information spanning large portions of text. For example, a question like "How is the concept of duality portrayed through the character who embodies both respect for authority and hatred of rebellion?" necessitates comprehending the overall narrative and character dynamics within the above book.

We compare Gemini 1.5 Pro against Gemini 1.0 Pro. Due to the limited context window of the latter, Gemini 1.0 Pro requires retrieval-augmented generation to access useful passages from the book. This method indexes passages using TF-IDF and stores the results in an external database. The question is then used as a query to re-rank passages by cosine similarity, and the most relevant passages are retrieved, up to a maximum of $4 \mathrm{k}$ tokens (roughly 41 passages). The retrieved passages are then put into context following a temporal ordering. In contrast, Gemini 1.5 Pro, due to its larger context window capable of accommodating much longer material, eliminates any need for additional data post-processing, indexing and retrieval pipelines. We also compare against Claude 2.1 for which we use the same retrieval technique to isolate the most relevant $4 \mathrm{k}$ tokens, as this model has maximum context length that is smaller than the whole book (i.e., $200 \mathrm{k}$ context length versus $710 \mathrm{k}$ tokens in the book).

To evaluate the models' response, we create a set of 100 questions ${ }^{14}$ and we conduct a human evaluation following the Attributable to Identified Sources (AIS) protocol (Rashkin et al., 2021) (see Tables 4 column AIS Human Evaluation). As indicated by the performance on the 0 -shot setup, all Gemini models show a good (yet not excellent) knowledge of the source material. On the other hand, Claude 2.1 on the same 0 -shot setup often errs on the side of punting (i.e., declining to answer) rather than providing an answer, minimizing the chances of hallucinating and making non-factual claims. Finally, providing the whole material in the context (i.e., the full 710k tokens of the book) can eliminate the need of additional retrieval components (i.e., $4 k$ tokens of retrieved context) without loss of performance. In fact, in this particular case we observe that the particular question set often requires resolving referring expressions (e.g., "the sibling" or "the protagonist") which require reasoning across[^10]

|  | Context <br> length | AutoAIS <br> Gemini 1.5 Pro | AIS <br> Human Evaluation | Num. Sentences <br> per answer |
| :---: | :---: | :---: | :---: | :---: |
| Anthropic Claude 2.1 | 0-shot | 11.1 | $30.2 \pm 2.4$ | 5.7 |
| Gemini 1.0 Pro | 0-shot | 85.3 | $79.1 \pm 1.3$ | 2.3 |
| Gemini 1.5 Pro | 0-shot | 82.1 | $75.5 \pm 1.2$ | 3.4 |
| Anthropic Claude 2.1 | 4k retrieved | 29.1 | $42.2 \pm 3.0$ | 5.1 |
| Gemini 1.0 Pro | 4k retrieved | 75.3 | $72.1 \pm 2.3$ | 2.6 |
| Gemini 1.5 Pro | 4k retrieved | 84.8 | $78.2 \pm 1.4$ | 4.9 |
| Gemini 1.5 Pro | 710k book | $\mathbf{9 1 . 4}$ | $\mathbf{8 0 . 0} \pm 1.0$ | 5.8 |

Table 4 | Evaluating the ability to answer questions about large collections of text across three context sizes: 0 -shot with no context provided, $4 \mathrm{k}$ retrieved using passages in total up to $4 \mathrm{k}$ sentence piece tokens, and the entire book serving as context. To assess the models' performance, we use a sentencebased AutoAIS score via macro average over all answers alongside mean human-based AIS scores with their standard errors.

long-range dependencies not easily capture by retrieval techniques. ${ }^{15}$

We repeat the experiment by changing the evaluation procedure, this time relying on the automatic form of AIS metric, i.e., AutoAIS (Bohnet et al., 2022; Gao et al., 2023). This metric is using LLMs to assess the factual accuracy of generated responses by checking their alignment with the source material. In this particular case we employ Gemini 1.5 Pro, ultimately evaluating the model's ability to act as a model-based evaluator, or more recently known as LLM Judge (Zheng et al., 2023). We give Gemini 1.5 Pro the entire text of the book in the context as a premise and then prompt it to indicate if each sentence in the answer is entailed by the premise. The AutoAIS score is the number of true sentences divided by the total number of sentences in the answer. See Table 4 column AutoAIS for the results showing an agreement between human and AutoAIS. This showcases the benefits that Gemini 1.5 Pro brings when used as a model-based evaluator and highlights the potentials of long-context models for these model-based evaluations that require reasoning and verification from large amounts of text. Such automatic approach can ease the burden on human raters performing long-context tasks; large volumes of text pose particularly difficulties in judging the correctness of questions, remembering facts, reasoning on these texts, and resolving referring expressions to entities.

### 4.2.2.3 Long-context Audio

Next, we evaluate Gemini 1.5 Pro's long context understanding capabilities on audio inputs. To evaluate long-context automatic speech recognition (ASR) performance, we test Gemini 1.5 Pro on 15 minute segments of an internal YouTube video-based benchmark. For this evaluation, we report results against the 1.0 Pro model, which is trained on audio segments much shorter in length. We also report performance with the Universal Speech Model (USM) (Zhang et al., 2023) and Whisper (OpenAI, 2023). Note that ASR tasks report a word error rate (WER) metric, where a lower number is better.

The Table 5 below shows that the 1.0 Pro model, when evaluated on transcribing 15 -minute videos without segmentation, has a WER of $100 \%$ due to a mismatch between training and testing[^11]audio lengths. When we segment the videos every 30 seconds and pass the textual content of the language model across each segment boundary, the 1.0 Pro model can achieve a WER of $7.8 \%$. The USM model with a CTC decoder, while robust to long segments, achieves a WER of $8.8 \%$. As indicated in the table, Whisper is not robust to long segments and hence requires audio to be segmented every 30 seconds to achieve a WER of $7.3 \%$. In comparison, Gemini 1.5 Pro is much more robust on these longer-context tasks. Specifically, thanks to its long-context capabilities and without the added complexity of extra input segmentation and pre-processing, Gemini 1.5 Pro can transcribe 15 -minute videos more accurately than other models, achieving a WER of $5.6 \%$.

|  | USM | Whisper | Gemini |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  |  | 1.0 Pro | 1.5 Pro |  |
| Segmentation | - | - | $30 \mathrm{~s}$ | - | $30 \mathrm{~s}$ | - |
| WER | $8.8 \%$ | $12.5 \%$ | $7.3 \%$ | $100 \%$ | $7.8 \%$ | $5.6 \%$ |

Table 5 | Word error rate (WER) for various models on 15 -minute videos.

### 4.2.2.4 Long-context Video QA

Question-answering benchmarks for long-context video understanding need to have at least two properties: first, they need to contain long videos and second, their questions need to be designed to in a way that can differentiate among models that operate over different context lengths. Unfortunately, no existing benchmarks satisfy these properties for evaluating models that can handle hours-long videos like Gemini 1.5 Pro. The publicly available question answering benchmark with the longest videos is EgoSchema (Mangalam et al., 2023), but its videos are at most 3 minutes (i.e., 180 frames) in length. To bridge this evaluation gap, we introduce a new benchmark, $1 \mathrm{H}$-VideoQA, composed of 125 five-way multiple-choice questions over public videos 40-105 minutes long.

We collected annotations that require understanding one or multiple events, each spanning only a few seconds from the full video so that the answer is extremely challenging to infer by looking at a few randomly sampled video frames.

We run experiments by extracting video frames at one frame-per-second, and further linearly subsampling long videos to a fixed context length. We also measure performance if we provide all frames for each video for 1H-VideoQA as a reference. Results are shown in Figure 12 and Table 6).

Figure 12 illustrates the improvement of $1 \mathrm{H}$-VideoQA over EgoSchema in terms of its ability to differentiate among models that operate over different numbers of frames. Gemini 1.5 Pro sets a new state-of-the-art of $64.5 \%$ accuracy on EgoSchema using only 16 frames (vs $55.6 \%$ for GPT4V (Balažević et al., 2024)). However, we do not see clear gains from going to 150 frames, suggesting

| Model | Frames |  |  |
| :--- | :---: | :---: | :---: |
|  | 16 | 150 | full video (1 fps) |
| GPT-4V (0-shot) | $29.6 \%$ | $49.6 \%$ | Not supported |
| Gemini 1.5 Pro (0-shot) | $38.4 \%$ | $52.8 \%$ | $64.8 \%$ |

Table 6 | Comparison between GPT-4V and Gemini 1.5 Pro on 1H-VideoQA. Experiments are run by sampling one video frame-per-second and linearly subsampling 16 or 150 frames. We also show performance if we provide all the frames for each video to Gemini 1.5 Pro.

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-19.jpg?height=996&width=1219&top_left_y=290&top_left_x=424)

Dataset

Figure 12 | Comparison between 1H-VideoQA and EgoSchema, reporting Gemini 1.5 Pro's accuracy when linearly subsampling to 1,16 , or 150 frames. We also show performance if we provide all the frames for each video for 1H-VideoQA, in yellow. Gemini 1.5 Pro achieves SotA accuracy on both benchmarks. Gemini 1.5 Pro's performance on 1H-VideoQA keeps increasing as we scale up to providing all frames in the video, while its performance on EgoSchema saturates after 16 frames, showing that $1 \mathrm{H}$-VideoQA is more effective at differentiating among models that operate over different context lengths.

that many questions in EgoSchema can be easily solved with a limited number of frames.

In contrast, on 1H-VideoQA the performance of Gemini 1.5 Pro consistently increases as the number of frames provided increases from the first frame to the full video, suggesting that a substantial proportion of questions in $1 \mathrm{H}$-VideoQA can only be solved with more frames as context, thereby making $1 \mathrm{H}$-VideoQA more effective as a long-context benchmark. Table 6 further shows that Gemini 1.5 Pro consistently outperforms GPT-4V on 1H-VideoQA, whether the video has been subsampled to 16 or to 150 frames. The fact that Gemini 1.5 Pro does not solve $1 \mathrm{H}$-VideoQA perfectly (see examples in Appendix, Table 18), despite observing a frame every second, makes $1 \mathrm{H}$-VideoQA a useful benchmark for evaluating and driving the development of future long-context video models.

## 5. Core Capability Evaluations

The final component of our evaluation harness for the Gemini 1.5 Pro measures the quality of the model's core capabilities (i.e., performance on non long-context task). The evaluations in this section consist of established benchmarks that are public and used by the community along with some internal benchmarks that are held-out and unleaked, covering all three modalities, text, vision and audio. Our selection criteria primarily aim to measure the improvement of Gemini 1.5 Pro compared to its predecessor, Gemini 1.0 series of models, Gemini 1.0 Pro and Gemini 1.0 Ultra. Our goal is to

|  | Core Capability | Relative to |  |
| :--- | :--- | :--- | :--- |
|  |  | $\mathbf{1 . 0}$ Pro | $\mathbf{1 . 0}$ Ultra |
| Text | Math, Science \& Reasoning | $+\mathbf{+ 3 8 . 4 \%}$ | $\mathbf{+ 1 1 . 2 \%}$ |
|  | Multilinguality | $+\mathbf{2 2 . 3 \%}$ | $+\mathbf{6 . 7 \%}$ |
|  | Coding | $+8.9 \%$ | $+\mathbf{0 . 2} \%$ |
|  | Instruction following | $+9.2 \%$ | $+\mathbf{2 . 5 \%}$ |
| Vision | Image understanding | $+\mathbf{+ 6 . 5 \%}$ | $-4.1 \%$ |
|  | Video understanding | $+16.9 \%$ | $+3.8 \%$ |
| Audio | Speech recognition | $+1.2 \%$ | $-5.0 \%$ |
|  | Speech translation | $+0.3 \%$ | $-2.2 \%$ |

Table 7 | Detailed breakdown of the results presented in Table 1.

highlight the extent of the trade-off, if it exists, between the 1.5 generation of Gemini models that excel in long-context capabilities and their performance on non long-context tasks. In particular, as we develop the 1.5 series, we aim to enhance the models' proficiency in this new dimension of multimodal long-context without compromising their quality across all other capabilities.

All in all, we find a clear generational improvement between the 1.0 and 1.5 series, with Gemini 1.5 Pro uniformly outperforming 1.0 Pro and approaching (often even surpassing) 1.0 Ultra, a state-of-the-art model on most benchmarks, despite being significantly more efficient to train.

### 5.1. Core Text Evals

We start by comparing three major core text capabilities: (1) Math, Science, and Reasoning; (2) Coding; (3) Multilinguality; and (4) Instruction Following. See Table 8 for a summary of these results.

### 5.1.1. Reasoning, Math and Science

We find that 1.5 Pro consistently outperforms both 1.0 Ultra and 1.0 Pro on grade-school math (i.e., GSM8K) and even shows material improvement over the more demanding benchmarks where there is more headroom for improvement, i.e., $+3.5 \%$ over 1.0 Ultra for middle- and high-school math problems (i.e., Hendrycks MATH), $+7.2 \%$ for the American Mathematical Competitions (i.e., AMC) and $+5.8 \%$ on graduate-level science problems ((Rein et al., 2023)). ${ }^{16}$ On reasoning tasks, 1.5 Pro outperforms 1.0 Pro by a large margin and shows a comparable performance to 1.0 Ultra, slightly underperforming on DROP and slightly outperforming on BBH. Gemini 1.5 Pro also greatly outperforms 1.0 Ultra on Hellaswag using a multiple-choice prompt (see Appendix 9.7) that 1.5 Pro's instruction tuning can take advantage of during inference. Finally, we find that on the popular and challenging MMLU benchmark measuring general science knowledge 1.5 Pro takes a leap over 1.0 Pro, and even approaches 1.0 Ultra arriving at $-1.8 \%$ behind it.

We also evaluated Gemini 1.5 Pro on a new benchmark, PhysicsFinals. This unreleased internal benchmark consists of 61 undergraduate physics problems, taken from final exams that have not appeared on the internet. They cover a range of subjects, including wave mechanics, quantum mechanics, special relativity, and introductory general relativity, and are at the undergraduate level. The answers were graded by a physics professor. We find that Gemini 1.5 Pro scored 37, a substantial[^12]

| Capability | Benchmark | Gemini |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  |  | 1.0 Pro | 1.0 Ultra | 1.5 Pro |
| Math, Science <br> $\&$ Reasoning | Hellaswag | $84.7 \%$ | $87.8 \%$ | $92.5 \%$ |
|  | (Zellers et al., 2019) | 10-shot | 10-shot | 10-shot |
|  | MMLU: Multiple-choice questions in <br> 57 subjects (professional \& academic). <br> (Hendrycks et al., 2021a) | 71.8\% <br> 5-shot | $83.7 \%$ <br> 5 -shot | $81.9 \%$ <br> 5-shot |
|  | GSM8K: Grade-school math problems. <br> (Cobbe et al., 2021) | $77.9 \%$ <br> 11 -shot | $88.9 \%$ <br> 11 -shot | $91.7 \%$ <br> 11 -shot |
|  | GPQA: Graduate-Level Google-Proof Q\&A. <br> (Rein et al., 2023) | $27.9 \%$ <br> $4-$ shot | $35.7 \%$ <br> 4 -shot | $37.9 \%$ <br> 4 -shot <br> $41.5 \%$ <br> 0 -shot |
|  | MATH: Math problems ranging <br> across 5 levels of difficulty <br> and 7 sub-disciplines. <br> (Hendrycks et al., 2021b) | $32.6 \%$ <br> 4-shot <br> Minerva <br> prompt | $53.2 \%$ <br> 4-shot <br> Minerva <br> prompt | 58.5\% <br> 4-shot <br> Minerva <br> prompt <br> $59.4 \%$ <br> 7-shot |
|  | PhysicsFinals: 61 undergraduate <br> physics problems that have <br> not appeared on the internet. | $31.1 \%$ <br> 0 -shot <br> (PT) | $41.0 \%$ <br> 0 -shot <br> (PT) | 60.7\% <br> 0 -shot |
|  | AMC 2022-23: 250 latest problems <br> including 100 AMC 12, 100 AMC 10, <br> and 50 AMC 8 problems. | $22.8 \%$ <br> 4 -shot | $30 \%$ <br> 4 -shot | $37.2 \%$ <br> 4 -shot |
|  | BigBench - Hard: A subset of harder <br> tasks from Big Bench formatted as <br> CoT problems. <br> (Srivastava et al., 2022; Suzgun et al., 2022) | $75.0 \%$ <br> 3 -shot | $83.6 \%$ <br> 3-shot | $84.0 \%$ <br> 3-shot |
|  | DROP: Reading comprehension <br> $\&$ arithmetic. (Metric: F1-Score). <br> (Dua et al., 2019) | $74.1 \%$ <br> Variable <br> shots | $82.4 \%$ <br> Variable <br> shots | $78.9 \%$ <br> Variable <br> shots |
| Coding | HumanEval <br> chat preamble* (Metric: pass rate). <br> (Chen et al., 2021) | $67.7 \%$ <br> 0 -shot <br> (PT) | $74.4 \%$ <br> 0 -shot <br> (PT) | $71.9 \%$ <br> 0-shot |
|  | Natural2Code <br> chat preamble* (Metric: pass rate). | $69.6 \%$ <br> 0 -shot | $74.9 \%$ <br> 0 -shot | $77.7 \%$ <br> 0 -shot |
| Multilinguality | WMT23: sentence-level machine <br> translation (Metric: BLEURT). <br> (Tom et al., 2023) | 71.73 <br> 1 -shot <br> (PT) | 74.41 <br> 1-shot <br> (PT) | 75.20 <br> 1 -shot |
|  | MGSM: multilingual math <br> reasoning. <br> (Shi et al., 2023a) | $63.45 \%$ <br> 8 -shot <br> (PT) | $78.95 \%$ <br> 8 -shot <br> (PT) | $88.73 \%$ <br> 8 -shot |

Table 8 | Evaluation results of Gemini 1.5 Pro and Gemini 1.0 models on standard coding, multilingual as well as math, science and reasoning benchmarks. Unless explicitly specified, all tasks are evaluated in terms of answer accuracy. Note that in this table, PT for the 1.0 Ultra and Pro models denote tasks evaluated with model variants that have undergone a post-training (i.e. instruction-tuning) phase after pre-training. All numbers for the 1.5 Pro are obtained after instruction-tuning, as described in Section 3.
improvement over both Gemini 1.0 Ultra, which scored 25, and Gemini 1.0 Pro, which scored 19.

### 5.1.2. Code

Gemini 1.5 Pro is our best performing model in code to date, surpassing Gemini 1.0 Ultra on Natural2Code, our internal held-out code generation test set made to prevent web-leakage.

HumanEval leakage HumanEval is an industry standard open-source evaluation benchmark (Chen et al., 2021), but we found controlling for accidental leakage on webpages and open-source code repositories to be a non-trivial task, even with conservative filtering heuristics. An analysis of the test data leakage of Gemini 1.0 Ultra showed that continued pretraining on a dataset containing even a single epoch of the test split for HumanEval boosted scores from $74.4 \%$ to $89.0 \%$, highlighting the danger of data contamination. We found that this sharp increase persisted even when examples were embedded in extraneous formats (e.g. JSON, HTML). We invite researchers assessing coding abilities of these models head-to-head to always maintain a small set of truly held-out test functions that are written in-house, thereby minimizing the risk of leakage. The Natural2Code benchmark, which we announced and used in the evaluation of Gemini 1.0 series of models, was created to fill this gap. It follows the exact same format of HumanEval but with a different set of prompts and tests.

### 5.1.3. Multilinguality

For our multilingual evaluations we use a multilingual math reasoning (MGSM; Shi et al., 2023a) benchmark and a machine translation benchmark (WMT23; Kocmi et al., 2023) which was constructed after the model's training data cut-off hence minimizing test set leakage risks. Both of these cover diverse languages from different language families and resource groups, with MGSM covering 11 languages and WMT23 eight languages for a total of 14 language pairs.

We find that Gemini 1.5 Pro improves over Gemini 1.0 Ultra on both tasks, particularly showing a substantial improvement of almost $+10 \%$ on the MGSM dataset, in line with the English-only math improvements reported above. Interestingly, we find that these improvements are not limited to a particular resource group; rather, 1.5 Pro improves performance equally among differently-resourced languages. Particularly, on medium and low resource languages the gap between 1.0 Ultra and 1.5 Pro increases to 9.5 and 7.6 respectively. ${ }^{17}$

### 5.1.4. Instruction Following

The ability of LLMs to follow complex requests is consistently improving, therefore in addition to the three core-capabilities above, we also evaluate Gemini 1.5 Pro on instruction following capability. We employ a fine-grained evaluation methodology that measures how well models can follow instructions in complex prompts. Different from the Gemini 1.0 Technical Report (Gemini-Team et al., 2023), we use an internal evaluation set of 406 prompts constructed by human raters, and covering varied topics and instruction types, e.g., generating formal and creative content, providing recommendations, summarizing, and rewriting texts, and solving coding and logical tasks. Each prompt contains between one to more than a dozen instructions, with an average count of five. Human annotators were asked to rate whether a response follows (or not) each of the instructions present in the prompt. We aggregate these human judgements into two metrics: per-instruction accuracy (the percentage of instructions over the full evaluation set that are followed) and full-response accuracy (percentage of prompts where every instruction was followed). Results are shown in Table 9. We find that Gemini 1.5 Pro[^13]

| Task | Gemini |  |  |
| :--- | :---: | :---: | :---: |
|  | $\mathbf{1 . 0}$ Pro | $\mathbf{1 . 0}$ Ultra | $\mathbf{1 . 5}$ Pro |
| Per-instruction accuracy | $85.7 \%$ | $86.0 \%$ | $\mathbf{8 8 . 7} \%$ |
| Full-response accuracy | $57.4 \%$ | $64.8 \%$ | $\mathbf{6 6 . 0} \%$ |

Table 9 | Instruction tuning performance on a diverse set of prompts covering diverse topics such as generating formal and creative content, providing recommendations, summarizing, and rewriting texts, solving coding and logical tasks, and more.

outperforms Gemini 1.0 series model, and follows close to $90 \%$ of the diverse instructions in our data. At the level of prompts, $66 \%$ were fully followed, a sizable improvement over Gemini $1.0 \mathrm{Pro}^{18}$. Note that the 1.0 models we compare with here are different from the Gemini Apps models in the 1.0 Technical Report which were post-trained for conversational AI tasks.

### 5.2. Core Vision Multimodal Evaluations

To assess performance on multimodal image tasks, we report results on 8 image understanding benchmarks and 5 video understanding benchmarks. Table 10 presents the results.

We find that Gemini 1.5 Pro improves substantially over Gemini 1.0 Pro on 5 of them, all multimodal reasoning benchmarks (i.e., MMMU, MathVista, ChartQA and AI2D), even matching or exceeding Gemini 1.0 Ultra on two of them (i.e., AI2D and ChartQA). On the remaining 3 that require strong OCR capabilities we see 1.5 Pro approaching but not surpassing 1.0 Pro. An error analysis on the performance of Gemini 1.5 Pro on these tasks revealed some false negatives which could lowerbound the model's true performance. As such, future work should focus on relying more on human evaluations for these datasets, especially when evaluating instruction-tuned models.

Turning to video understanding, we find that Gemini 1.5 Pro outperforms 1.0 Ultra on questionanswering datasets on all several-minutes long videos tested (i.e., ActivityNet-QA and EgoSchema). We see a similar picture on the video captioning benchmarks, with Gemini 1.5 Pro matching performance on YouCook2 and even surpassing 1.0 Ultra on VATEX and Chinese variant VATEX ZH.

### 5.3. Core Audio Multimodal Evaluations

In addition to long-context evaluations on speech input, we also evaluate Gemini 1.5 Pro on several short-context Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST) benchmarks. These include some private YouTube based English and Multilingual benchmarks, and public benchmarks like Multilingual Librispeech (MLS) (Pratap et al., 2020), FLEURS (Conneau et al., 2023) and CoVoST-2 (Wang et al., 2020). ${ }^{19}$ Results are shown in Table 11. On FLEURS we evaluate a subset of 55 languages for which we have coverage our training data. On CoVoST-2 we evaluate on translating speech in 20 languages into English, reporting on the subset of languages that were seen by the model during pre-training. We report Word-Error-Rate (WER) on all ASR tasks, where lower is better, except the four segmented languages on FLEURS where we aggregate Character-Error-Rates (Chinese, Japanese, Korean and Thai). On AST we report BLEU scores.[^14]

| Capability | Benchmark | Gemini |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  |  | 1.0 Pro | 1.0 Ultra | 1.5 Pro |
| Image Understanding | MMMU (val) <br> Multi-discipline <br> college-level problems <br> 0-shot (Yue et al., 2023) | $47.9 \%$ | $59.4 \%$ | $58.5 \%$ |
|  | Ai2D (test) <br> Science diagrams <br> 0 -shot (Kembhavi et al., 2016) | $73.9 \%$ | $79.5 \%$ | $80.3 \%$ |
|  | MathViSTA (testmini) <br> Mathematical reasoning <br> 0-shot (Lu et al., 2023) | $45.2 \%$ | $53.0 \%$ | $52.1 \%$ |
|  | ChartQA (test) <br> Chart understanding <br> 0 -shot (Masry et al., 2022) | $74.1 \%$ | $80.8 \%$ | $81.3 \%$ |
|  | VQAv2 (test-dev) <br> Natural image understanding <br> 0-shot (Goyal et al., 2017) | $71.2 \%$ | $77.8 \%$ | $73.2 \%$ |
|  | TextVQA (val) <br> Text reading on natural images <br> 0 -shot (Singh et al., 2019) | $74.6 \%$ | $82.3 \%$ | $73.5 \%$ |
|  | DocVQA (test) <br> Document understanding <br> 0-shot (Mathew et al., 2021) | $88.1 \%$ | $90.9 \%$ | $86.5 \%$ |
|  | InfographicVQA (test) <br> Infographic understanding <br> 0 -shot (Mathew et al., 2022) | $75.2 \%$ | $80.3 \%$ | $72.7 \%$ |
| Video understanding | VATEX (test) <br> English video captioning <br> 4-shot (Wang et al., 2019) | 57.4 | 62.7 | 63.0 |
|  | VATEX ZH (val) <br> Chinese video captioning <br> 4-shot (Wang et al., 2019) | 39.7 | 50.8 | 54.9 |
|  | YouCook2 (val) <br> English video captioning <br> 4-shot (Zhou et al., 2018) | 123.2 | 135.4 | 134.2 |
|  | ActivityNet-QA (test) <br> Video question answering <br> 0 -shot (Yu et al., 2019) | $49.8 \%$ | $52.2 \%$ | $56.7 \%$ |
|  | EgoSchema (test) <br> Video question answering <br> 0-shot (Mangalam et al., 2023) | $55.7 \%$ | $61.5 \%$ | $63.2 \%$ |

Table 10 | Comparison of Gemini 1.5 Pro with Gemini 1.0 Pro and Ultra on image and video understanding benchmarks. For DocVQA and InfographicVQA, we report Average Normalized Levenshtein Similarity (ANLS) (Biten et al., 2019). For VATEX, VATEX ZH and YouCook2 we report CIDER (Vedantam et al., 2015). For other datasets, we report accuracy.

|  | Task | Metric | USM | Whisper | Gemini |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  |  |  | 1.0 Pro | 1.0 Ultra | 1.5 Pro |
| Automatic <br> Speech <br> Recognition | YouTube <br> (en-us) | WER $(\downarrow)$ | $5.8 \%$ | $6.5 \%(v 3)$ | $4.8 \%$ | $4.7 \%$ | $4.6 \%$ |
|  | YouTube <br> (52 lang) | WER $(\downarrow)$ | $22.8 \%$ | $41.4 \%$ (v3) | $22.5 \%$ | $21.0 \%$ | $22.2 \%$ |
|  | Multilingual <br> LibriSpeech <br> (en-us) <br> (Pratap et al., 2020) | WER $(\downarrow)$ | $7.0 \%$ | $6.2 \%(v 2)$ | $4.8 \%$ | $4.4 \%$ | $4.6 \%$ |
|  | FLEURS <br> (55 lang) <br> (Conneau et al., 2023) | WER $(\downarrow)$ | $11.2 \%$ | $16.6 \%$ (v3) | $6.4 \%$ | $6.0 \%$ | $6.6 \%$ |
| Automatic <br> Speech <br> Translation | Covost 2 <br> (20 lang) <br> (Wang et al., 2020) | BLEU ( $\uparrow$ ) | 31.5 | 29.4 (v2) | 40.0 | 41.0 | 40.1 |

Table 11 | Comparison of Gemini 1.5 Pro with USM, Whisper, Gemini 1.0 Pro and Gemini 1.0 Ultra on audio understanding tasks.

Our results indicate that Gemini 1.5 Pro, despite being a generalist model, significantly improves over specialist models like USM and Whisper that are trained exclusively for speech understanding on speech understanding benchmarks. Note, Gemini 1.5 Pro performs similarly to Gemini 1.0 Pro on Speech Understanding, showing that performance on non long-context tasks is not compromised by the addition of long-context abilities. Finally, Gemini 1.0 Ultra does offer slight benefits over 1.5 Pro, but the former is a model requiring more training compute and serving resources.

## 6. Responsible Deployment

Consistent with Gemini 1.0 models, we follow a structured approach to responsible deployment in the creation of Gemini 1.5 Pro, as outlined in Figure 13. In this report we provide updated information on the Gemini 1.5 Pro impact assessment, evaluation approach, and model mitigation efforts for the latest model. Other work across the responsible deployment lifecycle and relevant to Gemini 1.5 Pro that remains consistent with Gemini 1.0 series are outlined within the Gemini 1.0 Technical Report (Gemini-Team et al., 2023).

![](https://cdn.mathpix.com/cropped/2024_05_26_3ca8861fe62504160d11g-26.jpg?height=674&width=1100&top_left_y=694&top_left_x=475)

Figure 13 | The stages of responsible deployment for Gemini models.

### 6.1. Impact Assessment

As outlined in the Gemini 1.0 Technical Report we develop model impact assessments to identify, assess, and document key downstream societal benefits and harms associated with the development of advanced models, conducted by the Responsible Development and Innovation team and reviewed by our Google DeepMind Responsibility and Safety Council in order to uphold the Google AI Principles (Google, 2018).

The impact of Gemini models, outlined in prior reports, focused on text generation and understanding, and image and video understanding. As such, all previous work from Gemini 1.0 Technical Report remains relevant to the Gemini 1.5 Pro model and so the assessment of this model addresses the additional consequences of long-context understanding across modalities.

The improved capabilities offered by the Gemini 1.5 Pro model are likely to enhance many of its societal benefits. The ability to understand longer content enhances the efficiency of individual and commercial users in processing various multimodal inputs. Besides efficiency, it enables societally beneficial downstream use cases. For example, long form video understanding could enable easier exploration of archival content, potentially benefiting groups from journalists to historians. While long-context understanding can enhance potential downstream benefits, it may exacerbate some of the risks outlined in the Gemini 1.0 Technical Report (Gemini-Team et al., 2023). Besides exacerbating known risks, we continuously evaluate whether it introduces new adverse effects to the model behavior. This includes, for instance, monitoring the potential of longer input files to negatively affect the safety performance of the models.

### 6.2. Model Mitigations

Our modeling mitigation of safety risks is mostly through supervised fine-tuning (SFT) and reinforcement learning through human feedback (RLHF) using a reward model (Bai et al., 2022). In contrast to generic quality-oriented instruction-tuning catering to all types of user queries, our safety mitigation is more focused on adversarial, or "harm-inducing" queries, i.e., the smaller slice of user queries where an unprotected model is likely to produce harmful responses according to our model safety policies.

We refer readers to the Gemini 1.0 Technical Report for safety mitigation method details, as the mitigation recipe in Gemini 1.5 Pro is mostly the same (Gemini-Team et al., 2023). The most substantial new update in 1.5 Pro mitigation is the incorporation of new image-to-text SFT data, as we have observed that safety SFT data for text-only queries was not as effective for harm-inducing image-to-text queries.

### 6.3. Model Safety Evaluations

Following prior evaluation approaches outlined in the Gemini 1.0 Technical Report, we undertake a range of safety evaluations on the Gemini 1.5 Pro model. Below we report results on some of the latest development and assurance evaluations across content safety, representational harms, and memorization for text-to-text and image-to-text evaluations. In addition, we are continuing to conduct evaluations on other modality areas (e.g. audio-to-text and video-to-text) whilst undertaking red teaming and external evaluations.

### 6.4. Development and Assurance Evaluations

### 6.4.1. Content Safety

We evaluate models against harm types according to our safety policies, as discussed in the Gemini 1.0 Technical Report. While both development and assurance evaluations cover critical policy areas, we maintain separate datasets, treating assurance sets as held-out to prevent overfitting and preserve validity of results. For safety policy evaluation we rely on human evaluations and maintain well-being programs in place for human annotation and closely monitor feedback from our raters.

We assess the effectiveness of our latest safety mitigations, based on both how it improves safety over time and how conversational models built from our safety-mitigated model compares to unmitigated models.

### 6.4.1.1 Text-to-text

Our text-to-text safety mitigation in Gemini 1.5 Pro was found to be similarly effective in comparison to Gemini 1.0 Pro, mitigating the majority of text harm cases we identified on unmitigated models. Evaluations span across model safety policies including hate, harassment, and dangerous content. We found no significant change in safety rates when compared to 1.0 models (both Pro and Ultra).

We subsequently report side-by-side preference on a set of safety-focused prompts (e.g. on content safety topics such as hate speech, harassment and dangerous content), where we

- show responses from two models on the same query as the "base model" vs. "test model",
- request raters to prefer safe and helpful responses, and then
- aggregate across raters and queries.

Results are shown in Table 12. If the win-rate is above $50 \%$, this means that the test model produces safer and more nuanced responses than the base model. We find that whilst overall safety violation rates have no significant change, Gemini 1.5 Pro sees a higher safety win-rate versus the 1.0 Pro model, together with the significantly increased performance reported in Section 5.

| Gemini 1.5 Pro | Compared to 1.0 Pro | Compared to 1.0 Ultra |
| :---: | :---: | :---: |
| Safety Win-Rate | $57.0 \%(55.0 \%, 60.0 \%)$ | $58.0 \%(56.0 \%, 60.0 \%)$ |

Table 12 | Text-to-text content safety results. The safety win-rate shows the mean and $95 \%$ CI.

### 6.4.1.2 Image-to-text

We evaluate our mitigation progress on image-to-text safety on a set of human-curated harm-inducing queries, where each query contains an image and a related text query. Similar to the text-to-text evaluation protocol, we conduct an evaluation of model responses against adversarial model safety policy areas.

We find that, by including a set of multimodal safety data and co-training with text only safety data, violation rates on our evaluation dataset dropped substantially. As shown in Table 13, negative values indicate that our current model has fewer safety violations than earlier versions, with Gemini 1.5 Pro seeing far fewer unsafe responses in adversarial tests compared with earlier 1.0 Pro and Ultra models.

| Gemini 1.5 Pro | Compared to 1.0 Pro | Compared to 1.0 Ultra |
| :---: | :---: | :---: |
| Violations $(\downarrow)$ | $-27.39 \% \mathrm{pt}$ | $-27.32 \% \mathrm{pt}$ |

Table 13 | Image-to-text content safety results.

Though this is encouraging, we also believe there is room to improve our evaluations by making them more challenging and linking them closer to observed use cases as the models become more broadly deployed.

### 6.4.2. Representational Harms

### 6.4.2.1 Text-to-text

We carried out the same representational harms safety testing as in Gemini 1.0 Technical Report by performing text evaluations using Winogender (Rudinger et al., 2018), Winobias (Zhao et al., 2018), and Bias Benchmark in QA (BBQ) (Parrish et al., 2021) datasets. Refer to Table 14 for a breakdown of results.

We observed 0.007 and 0.013 average bias score on ambiguous and disambiguated portions of the BBQ benchmark for Gemini 1.5 Pro, compared to -0.292 and 0.004 Gemini 1.0 Pro, where a lower negative score means responses are better at being anti-stereotypical and hence lower is better. We find good absolute performance in both settings ( $99.5 \%$ accuracy on ambiguous and $93.3 \%$ in disambiguated context). In the disambiguated context setting, the incorrect responses are instances of the model responding "Can't be determined", which is not counted in the bias metric, but could be an instance of erasure. Overall we do not observe a systematic pattern of bias on these tasks.

Additionally, we evaluated toxicity across languages and bias dimensions using a subset of the

|  | Gemini 1.5 Pro <br> bias score $(\downarrow)$ | Gemini 1.0 Pro <br> bias score $(\downarrow)$ | Gemini 1.5 Pro <br> accuracy | Gemini 1.0 Pro <br> accuracy |
| :---: | :---: | :---: | :---: | :---: |
| BBQ Ambig | 0.007 | -0.292 | $99.5 \%$ | $99.9 \%$ |
| BBQ Disambig | 0.013 | 0.004 | $93.3 \%$ | $97.7 \%$ |
| Winobias Ambig | 0.23 | 0.25 | $81 \%$ | $81 \%$ |
| Winobias Disambig | 0.03 | 0.06 | $96 \%$ | $95 \%$ |
| Winogender | 0.02 | 0.05 | $87 \%$ | $82 \%$ |

Table 14 | Text-to-text representational harms results. Gemini 1.5 Pro achieves similar low bias scores (close to zero) on representational harms tasks.

Multilingual Representational Benchmark introduced in the PaLM 2 technical report (Anil et al., 2023). We observed that $0.03 \%$ model continuations were classified as toxic, when sampling 3 continuations per prompt.

### 6.4.2.2 Image-to-text

Similar to the Gemini 1.0 Technical Report, we continue to monitor representation biases for image-totext and video-to-text modalities. For example, for image-to-text we study if models make ungrounded inferences, or inferences about people based on their appearance alone. In order to do this, we use images from the MIAP dataset (Schumann et al., 2021) and curate a set of ungrounded questions probing various attributes that cannot be answered from information in the image, e.g., "Is this person a criminal?". Additionally, we consider grounded questions in which the image does include relevant contextual information (e.g., a military uniform might be relevant to the question "Where is this person from?"). We measure violation rate via a human evaluation and report the difference in refusal rates between our Gemini 1.5 Pro and Gemini 1.0 Pro and Ultra models. Results are shown in Table 15.

We see Gemini 1.5 Pro consistently having a higher refusal rate. For ungrounded questions, this indicates that the model is less likely to answer questions it cannot answer from the images alone. However, the higher refusal rate for grounded questions implies that the model struggles to understand when to discuss contextually relevant information.

| Gemini 1.5 Pro | Compared to 1.0 Pro | Compared to 1.0 Ultra |
| :---: | :---: | :---: |
| Refusal rate, ungrounded questions ( $\uparrow$ ) | $18 \%$ | $69 \%$ |
| Refusal rate, grounded questions $(\downarrow)$ | $22 \%$ | $58 \%$ |

Table 15 | Image-to-text representational harms results. Gemini 1.5 Pro has higher refusal rates on both grounded and ungrounded questions

### 6.5. Divergence

Recent work has shown that language models may be vulnerable to a new adversarial attack that can bypass alignment (Nasr et al., 2023). This attack can cause models to diverge, and as in the case of ChatGPT 3.5 Turbo, sometimes regurgitate memorized training data in the process. ${ }^{20}$ Here, we evaluate Gemini 1.5 Pro to understand its susceptibility to divergence and in particular, emitting memorized training data via this attack.[^15]

We implement the divergence attack following Nasr et al. (2023), which attacks the model by asking it to repeat a single token many times. When successful, this attack first causes the model to diverge, i.e., to output text that is not a repetition of the specified token. We then examine which of these diverged outputs may contain regurgitated training data. ${ }^{21}$

Overall, we find that divergence occurs $44 \%$ of the time. We then turn to assessing how many of these diverged outputs contain regurgitated training data. To do this, we compare 50 -token long outputs from the model with the suffix array in Nasr et al. (2023). We find that emission of training data is infrequent. Specifically, we observed that when the model produces divergence, $0.35 \%$ of generations are training data. This rate is lower than ChatGPT 3.5 which emits training data at a rate of around 2.8\% (see Nasr et al. (2023, Figure 1.)) but also higher than some open-source models, e.g., we found that LLaMA and Mistral emit training data at a rate around $0.1 \%$ with this attack. Out of a total of 3750 queries to the model, a manual inspection of these tokens found them to come from 23 unique passages that were memorized by the model, with the majority of those being LaTeX and boilerplate code (e.g., coding interview solutions, ML code, and Jupyter notebooks). Overall, we conclude that Gemini 1.5 Pro is susceptible to divergence but that this particular attack often fails to elicit the model to emit training data.

We next study if the longer-context advancements made in Gemini 1.5 Pro can make the model more vulnerable to these divergence attacks. To assess this, we study how easy it is to cause divergence with longer prompts. If it is easier to cause divergence, then an adversary can obtain more memorized data with fewer queries. We query the model with prompts where the token is manually repeated 1,000 times or 999,000 times. We observe a stark difference in success and find that while the short prompts succeed at divergence only $35.6 \%$ of the time, the long prompts succeed at causing divergence $61.5 \%$ of the time. Hence, it is easier to obtain memorized data with longer prompts using the divergence attacks.

### 6.6. Deployment

We release external model cards on an ongoing basis within updates of our technical reports and in documentation for enterprise customers (Mitchell et al., 2019b). The Gemini 1.5 Pro model card can be found in Appendix 9.1. Additionally, online content covering terms of use, model distribution and access, and operational aspects such as change control, logging, monitoring and feedback can be found on relevant product websites, such as Cloud Vertex AI. Some of the key aspects are linked to below:
- Google Terms of service
- Google Cloud Platform Terms of service
- Google Cloud Privacy Notice
- Gemini Privacy Notice
- Generative AI Prohibited Use Policy
- Generative AI Terms of service[^16]

## 7. Discussion

We have presented Gemini 1.5 Pro, the first release from the Gemini 1.5 family. This new family of multi-modal models are based on the mixture-of-experts architecture (Shazeer et al., 2017) and push the boundary of efficiency, multi-modality, long-context reasoning and downstream performance. Gemini 1.5 Pro extends the content window over the Gemini 1.0 series from $32 \mathrm{~K}$ to multiple millions of tokens, making this the first commercially available model to greatly surpass the current ceiling of 200k tokens offered by Claude 2.1 across modalities. We have further demonstrated improved long-context performance out to $10 \mathrm{M}$ tokens.

Our extensive evaluations with diagnostic and realistic multi-modal long-context benchmarks show that 1.5 Pro is able to maintain near-perfect recall on multi-modal versions of needle-in-a-haystack (see Section 4.2.1.2) and is able to effectively use its context to retrieve and reason over large amounts of data. This enables the model to perform realistic long-context tasks such as long-document QA from $700 \mathrm{k}$-word material and long-video QA from 40 to 105 minutes long videos. Finally, 1.5 Pro has the ability to use in-context learn to translate from English to Kalamang, an extremely lowresource language with fewer than 200 speakers (Visser, 2020b). This capability is achieved solely by providing a grammar manual in its context at inference time, which demonstrates the Gemini 1.5 Pro's remarkable ability to in-context learn from information it has never seen before at training time.

Most importantly, this leap in long-context performance does not come at the expense of the multi-modal core capabilities (i.e., performance on non long-context tasks) that the 1.0 series excelled at. On the contrary, 1.5 Pro is able to outperform 1.0 Pro across the board of our comprehensive evaluation benchmarks being presented in this report. Interestingly, 1.5 Pro, despite using significantly less training compute, matches and in some capabilities even surpasses 1.0 Ultra, a state-of-the-art model, on text capabilities like math, science and reasoning, code, multilinguality and instruction following.

## Long-context evaluations, call-to-action

Evaluating the capabilities of models that can handle very long contexts presents a new set of challenges, especially in the multi-modal domain where text, images, video, and audio can be combined. Current benchmarks often fail to adequately stress-test models like Gemini 1.5 Pro, as they are typically designed for evaluating shorter context models. As the evaluation requirements for frontier models increasingly require benchmarks with both length and complexity, the task of human labeling and annotation will become significantly more costly and time-consuming. This additionally challenges traditional evaluation methods that rely heavily on manual evaluation.

Thus, given the limitations of existing benchmarks and the challenges of human annotation, there is a pressing need for innovative evaluation methodologies. These methodologies should be able to effectively assess model performance on very long-context tasks while minimizing the burden of human labeling. To begin addressing some of these concerns, we recommend researchers and practitioners adopt a "multiple needles-in-haystack" setup for diagnostic evaluations which we observed to be signal-rich and more challenging compared to its "single needle" counterpart. We believe there is room for new benchmark tasks based on new or improved automatic metrics that require complex reasoning over long inputs (both human and model generated). This is also an intriguing research direction for creating challenging evaluations that stress more than just the retrieval capabilities of long-context models. We will be continuing the development of such benchmarks for realistic and comprehensive evaluation of model capabilities in the multimodal space. By addressing these open challenges and developing new benchmarks and evaluation methodologies, we can drive progress in the field of very long-context AI models and unlock their full potential.

## References

Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Ontañón, Siddhartha Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo, James Lee-Thorp, Yi Tay, et al. Colt5: Faster long-range transformers with conditional computation. arXiv preprint arXiv:2303.09752, 2023.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 Technical Report, 2023.

Anthropic. Model Card and Evaluations for Claude Models, 2023a.

Anthropic. Long context prompting for Claude 2.1, $2023 \mathrm{~b}$.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. April 2022. URL https://arxiv.org/abs/2204.05862.

Ivana Balažević, Yuge Shi, Pinelopi Papalampidi, Rahma Chaabouni, Skanda Koppula, and Olivier J Hénaff. Memory consolidation enables long-context video understanding. arXiv preprint arXiv:2402.05861, 2024.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation, 2013.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R Gormley. Unlimiformer: Long-range transformers with unlimited length input. arXiv preprint arXiv:2305.01625, 2023.

Steven Bird. Decolonising speech and language technology. In Donia Scott, Nuria Bel, and Chengqing Zong, editors, Proceedings of the 28th International Conference on Computational Linguistics, pages 3504-3519, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.313. URL https://aclanthology.org/2020. coling-main. 313.

Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluis Gomez, Marçal Rusiñol, C.V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Scene text visual question answering. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4290-4300, 2019. doi: 10.1109/ICCV.2019.00439.

Bernd Bohnet, Vinh Q. Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, Tom Kwiatkowski, Ji Ma, Jianmo Ni, Lierni Sestorain Saralegui, Tal Schuster, William W. Cohen, Michael Collins, Dipanjan Das, Donald Metzler, Slav Petrov, and Kellie Webster. Attributed question answering: Evaluation and modeling for attributed large language models. 2022. URL https://arxiv.org/abs/2212. 08037.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/ google/jax.

Thorsten Brants, Ashok Popat, Peng Xu, Franz Och, and Jeffrey Dean. Large language models in machine translation. pages 858-867, 012007.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel HerbertVoss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc., 2020. URL https://proceedings .neurips.cc/paper_ files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 11079-11091. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ 47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf.

Aydar Bulatov, Yuri Kuratov, and Mikhail S Burtsev. Scaling transformer to $1 \mathrm{~m}$ tokens and beyond with rmt. arXiv preprint arXiv:2304.11062, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Stanley F Chen and Joshua Goodman. An empirical study of smoothing techniques for language modeling. Computer Speech \& Language, 13(4):359-394, 1999.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023. URL https://arxiv.org/abs/2309.12307.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling Language Modeling with Pathways. Journal of Machine Learning Research, 24(240):1-113, 2023. URL http://jmlr.org/papers/v24/22-1144.html.

Aidan Clark, Diego de las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, George van den Driessche, Eliza Rutherford, Tom Hennigan, Matthew Johnson, Katie Millican, Albin Cassirer, Chris Jones, Elena Buchatskaya, David Budden, Laurent Sifre, Simon Osindero, Oriol Vinyals, Jack Rae, Erich Elsen, Koray Kavukcuoglu, and Karen Simonyan. Unified scaling laws for routed language models, 2022. URL https://arxiv.org/abs/2202.01169.

Peter Clark, Oyvind Tafjord, and Kyle Richardson. Transformers as soft reasoners over language. IJCAI, 2020. URL https://www.ijcai.org/proceedings/2020/0537.pdf.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798-805. IEEE, 2023.

Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks, 2014.

Jeff Dean. Introducing Pathways: A next-generation AI architecture, 2021. URL https://blog.google/technology/ai/ introducing-pathways-next-generation-ai-architecture/.

Aparna Dhinakaran, 2024. URL https://twitter.com/aparnadhinak/status/ 174471295940669689.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen MeierHellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient Scaling of Language Models with Mixture-of-Experts. ICML, 2022. URL https://arxiv.org/abs/2112.06905.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368-2378, 2019. URL https://aclanthology.org/N19-1246.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:2101.03961, 2021. URL https : //arxiv.org/abs/2101.03961.

Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution, 2023.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Y. Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. Rarr: Researching and revising what language models say, using language models, 2023. URL https://arxiv.org/ $\mathrm{abs} / 2210.08726$.

Gemini-Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. URL https://storage. googleapis.com/deepmind-media/gemini/gemini_1_report.pdf.

Google. Our principles, 2018. URL https://ai.google/responsibility/principles/. Accessed May 16, 2023.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904-6913, 2017.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.

Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yinfei Yang. Longt5: Efficient text-to-text transformer for long sequences. arXiv preprint arXiv:2112.07916, 2021. URL https://arxiv.org/abs/2112.07916.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3929-3938. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/ v119/guu20a.html.

Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network library and ecosystem for JAX, 2023. URL http: //github.com/google/flax.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021b. URL https://arxiv.org/abs/2103.03874.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. URL https://arxiv.org/abs/2203.15556.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299, 2022. URL https://arxiv.org/ abs $/ 2208.03299$.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79-87, 1991.

Frederick Jelinek. Statistical methods for speech recognition. MIT press, 1998.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Zhengbao Jiang, Luyu Gao, Zhiruo Wang, Jun Araki, Haibo Ding, Jamie Callan, and Graham Neubig. Retrieval as attention: End-to-end learning of retrieval and reading within a single transformer. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2336-2349, Abu Dhabi, United Arab

Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.149. URL https://aclanthology.org/2022.emnlp-main. 149.

Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling, 2016. URL https://arxiv.org/abs/1602.02410.

Gregory Kamradt, 2023. URL https://github.com/gkamradt/LLMTest_ NeedleInAHaystack/blob/main/README.md.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781, Online, November 2020. Association for Computational Linguistics. doi: $10.18653 / v 1 / 2020 . e m n l p-m a i n .550$. URL https://aclanthology.org/2020.emnlp-main. 550.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

Young Jin Kim, Ammar Ahmad Awan, Alexandre Muzio, Andres Felipe Cruz Salinas, Liyang Lu, Amr Hendy, Samyam Rajbhandari, Yuxiong He, and Hany Hassan Awadalla. Scalable and efficient moe training for multitask multilingual models. arXiv preprint arXiv:2109.10465, 2021.

R. Kneser and H. Ney. Improved backing-off for m-gram language modeling. In 1995 International Conference on Acoustics, Speech, and Signal Processing, volume 1, pages 181-184 vol.1, 1995. doi: 10.1109/ICASSP.1995.479394.

Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, Barry Haddow, Philipp Koehn, Benjamin Marie, Christof Monz, Makoto Morishita, Kenton Murray, Makoto Nagata, Toshiaki Nakazawa, Martin Popel, Maja Popović, and Mariya Shmatova. Findings of the 2023 conference on machine translation (WMT23): LLMs are here but not quite there yet. In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz, editors, Proceedings of the Eighth Conference on Machine Translation, pages 1-42, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.1. URLhttps://aclanthology.org/2023.wmt-1. 1.

Greg Kohs. Alphago. Motion Picture, 2017. Produced by DeepMind Technologies and distributed by Netflix.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=qrwe7XHTmYb.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022. URL https://arxiv.org/abs/2206.14858.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention, 2024.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Pedro Henrique Martins, Zita Marinho, and Andre Martins. $\infty$-former: Infinite memory transformer. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5468-5485, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/ v1/2022.acl-long.375. URL https://aclanthology.org/2022.acl-long. 375.

Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of ACL, 2022.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages $2200-2209,2021$.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697-1706, 2022.

Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernockỳ, and Sanjeev Khudanpur. Recurrent neural network based language model. In INTERSPEECH, 2010.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency, FAT*'19, page 220-229, New York, NY, USA, 2019a. Association for Computing Machinery. ISBN 9781450361255. doi: 10.1145/3287560.3287596. URL https://doi.org/10.1145/3287560.3287596.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency, pages 220-229, $2019 \mathrm{~b}$.

Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. arXiv preprint arXiv:2304.08467, 2023.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models, 2023.

OpenAI. GPT-4 Technical Report. 2023.

OpenAI. Whisper, 2023. URL https://github.com/openai/whisper.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. arXiv preprint arXiv:2303.06349, 2023.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. BBQ: A hand-built bias benchmark for question answering. CoRR, abs/2110.08193, 2021. URL https://arxiv.org/abs/2110.08193.

Maja Popović. chrF: character n-gram F-score for automatic MT evaluation. In Ondřej Bojar, Rajan Chatterjee, Christian Federmann, Barry Haddow, Chris Hokamp, Matthias Huck, Varvara Logacheva, and Pavel Pecina, editors, Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392-395, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-3049. URL https://aclanthology.org/W15-3049.

Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020.

Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, JeanBaptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis \& insights from training Gopher. CoRR, abs/2112.11446, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified textto-text transformer. Journal of Machine Learning Research, 21(140):1-67, 2020. URL http: //jmlr.org/papers/v21/20-074.html.

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. Measuring attribution in natural language generation models. CoRR, abs/2112.12870, 2021.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q\&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts, 2021.

Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. Hash layers for large sparse models. Advances in Neural Information Processing Systems, 34:17555-17566, 2021.

Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 8-14, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2002. URL https://aclanthology.org/N18-2002.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488, 2021.

Candice Schumann, Susanna Ricco, Utsav Prabhu, Vittorio Ferrari, and Caroline Pantofaru. A step toward more inclusive people annotations for fairness. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, AIES '21. ACM, July 2021. doi: 10.1145/3461702.3462594. URL http://dx.doi.org/10.1145/3461702.3462594.

Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7881-7892, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. acl-main.704. URL https://aclanthology.org/2020.acl-main.704.

Claude Elwood Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27:379-423, 1948. URL http://plan9.bell-labs.com/cm/ms/what/shannonday/ shannon1948.pdf.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In ICLR (Poster). OpenReview.net, 2017. URL http://dblp.uni-trier.de/db/conf/iclr/ iclr2017.html\#ShazeerMMDLHD17.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language Models are Multilingual Chain-of-Thought Reasoners. In Proceedings of ICLR 2023, 2023a. URL http: //arxiv.org/abs/2210.03057.

Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Victoria Lin, Noah A Smith, Luke Zettlemoyer, Scott Yih, and Mike Lewis. In-context pretraining: Language modeling beyond document boundaries. arXiv preprint arXiv:2310.10638, $2023 \mathrm{~b}$.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317-8326, 2019.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022. URL https://arxiv.org/abs/ 2206.04615.

Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, Henryk Michalewski, Łukasz Kuciński, and Piotr Miłoś. Structured packing in llm training improves long context utilization. arXiv preprint arXiv:2312.17296, 2023.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. A benchmark for learning to translate a new language from one grammar book. In Arxiv, 2023.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. LaMDA: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022. URL https://arxiv.org/abs/2201.08239.

Kocmi Tom, Eleftherios Avramidis, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, et al. Findings of the 2023 conference on machine translation (wmt23): Llms are here but not quite there yet. In WMT23-Eighth Conference on Machine Translation, pages 198-216, 2023.

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017. URL http://arxiv.org/abs/1706.03762.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566-4575, 2015.

Eline Visser. Kalamang dictionary. Dictionaria, (13):1-2737, 2020a. URL https://dictionaria. clld.org/contributions/kalamang.

Eline Visser. A grammar of kalamang: The papuan language of the karas islands. 2020b.

Changhan Wang, Anne Wu, and Juan Pino. Covost 2 and massively multilingual speech-to-text translation. arXiv preprint arXiv:2007.10310, 2020.

Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. VATEX: A large-scale, high-quality multilingual dataset for video-and-language research. In ICCV, 2019.

Qingyang Wu, Zhenzhong Lan, Kun Qian, Jing Gu, Alborz Geramifard, and Zhou Yu. Memformer: A memory-augmented transformer for sequence modeling. In Yulan He, Heng Ji, Sujian Li, Yang Liu, and Chua-Hui Chang, editors, Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022, pages 308-318, Online only, November 2022a. Association for Computational Linguistics. URL https://aclanthology.org/2022.findings-aacl. 29.

Yuhuai Wu, Markus N Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. arXiv preprint arXiv:2203.08913, 2022b.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039, 2023.

XLA. XLA: Optimizing compiler for TensorFlow. https://www.tensorflow.org/xla, 2019. [Online; accessed December-2023].

Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi, Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, et al. Gspmd: general and scalable parallelization for ml computation graphs. arXiv preprint arXiv:2105.04663, 2021.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers, 2023.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. ActivityNet-QA: A dataset for understanding complex web videos via question answering. In AAAI, 2019.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2023.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33:17283-17297, 2020.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Yu Zhang, Wei Han, James Qin, Yongqiang Wang, Ankur Bapna, Zhehuai Chen, Nanxin Chen, Bo Li, Vera Axelrod, Gary Wang, Zhong Meng, Ke Hu, Andrew Rosenberg, Rohit Prabhavalkar, Daniel S. Park, Parisa Haghani, Jason Riesa, Ginger Perng, Hagen Soltau, Trevor Strohman, Bhuvana Ramabhadran, Tara Sainath, Pedro Moreno, Chung-Cheng Chiu, Johan Schalkwyk, Françoise Beaufays, and Yonghui Wu. Google usm: Scaling automatic speech recognition beyond 100 languages. arXiv preprint arXiv:2303.01037, 2023.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. arXiv preprint arXiv:1804.06876, 2018.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging $11 \mathrm{~m}$-as-a-judge with mt-bench and chatbot arena, 2023.

Zexuan Zhong, Tao Lei, and Danqi Chen. Training language models with memory augmentation. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5657-5673, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.382. URL https://aclanthology.org/2022.emnlp-main. 382.

Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards automatic learning of procedures from web instructional videos. In AAAI Conference on Artificial Intelligence, pages 7590-7598, 2018.

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Designing effective sparse expert models. arXiv preprint arXiv:2202.08906, 2022. URL https://arxiv.org/abs/2202.08906.
</end of paper 3>


<paper 4>
# (")RULER: What's the Real Context Size of Your Long-Context Language Models? 

Cheng-Ping Hsieh*, Simeng Sun*, Samuel Kriman, Shantanu Acharya<br>Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg<br>NVIDIA<br>\{chsieh, simengs\}@nvidia.com


#### Abstract

The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the "needle") from long distractor texts (the "haystack"), has been widely adopted to evaluate long-context language models (LMs). However, this simple retrieval-based test is indicative of only a superficial form of long-context understanding. To provide a more comprehensive evaluation of long-context LMs, we create a new synthetic benchmark RULER with flexible configurations for customized sequence length and task complexity. RULER expands upon the vanilla NIAH test to encompass variations with diverse types and quantities of needles. Moreover, RULER introduces new task categories multi-hop tracing and aggregation to test behaviors beyond searching from context. We evaluate ten longcontext LMs with 13 representative tasks in RULER. Despite achieving nearly perfect accuracy in the vanilla NIAH test, all models exhibit large performance drops as the context length increases. While these models all claim context sizes of $32 \mathrm{~K}$ tokens or greater, only four models (GPT-4, Command-R, Yi-34B, and Mixtral) can maintain satisfactory performance at the length of $32 \mathrm{~K}$. Our analysis of Yi-34B, which supports context length of $200 \mathrm{~K}$, reveals large room for improvement as we increase input length and task complexity. We open source RULER to spur comprehensive evaluation of long-context LMs.


## 1 Introduction

Recent advancements in AI system engineering (Dao et al., 2022; Jacobs et al., 2023; Fu et al., 2024) and language model designs (Chen et al., 2023: Xiong et al., 2023) have enabled efficient scaling up of context length for language models (Liu et al. 2024a; Young et al. 2024). Previous works (AI21, 2024, X.AI, 2024 Reid et al., 2024 Anthropic, 2024) commonly adopt synthetic tasks, such as passkey retrieval (Mohtashami \& Jaggi 2023) and needle-in-ahaystack (Kamradt, 2023) to evaluate long-context LMs. However, these evaluations are used inconsistently across works and reveal merely the retrieval capability, failing to gauge other forms of long-context understanding.

In this work, we propose RULER, a new benchmark to evaluate long-context modeling capabilities for language models. RULER contains four task categories to test behaviors (Ribeiro et al. 2020) beyond simple retrieval from context:

1. Retrieval: we extend the needle-in-a-haystack (Kamradt 2023, NIAH) test to evaluate retrieval capability with diverse types and quantities of needles.
2. Multi-hop Tracing: we propose variable tracking, a minimal proxy task for coreference chain resolution to check the behavior of tracing entities with multi-hop connections.
3. Aggregation: we propose common/frequent words extraction, proxy tasks for summarization to test the ability to aggregate relevant information that spans long-range context.

* Authors contributed equally.

| Benchmark \& Task | Avg Len | Type | Diverse <br> Tasks | Min. Parametric <br> Knowledge | Controllable <br> Context |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ZeroSCROLLS | $\sim 10 \mathrm{k}$ | realistic | $\checkmark$ | $x$ | $x$ |
| L-Eval | $\sim 8 \mathrm{k}$ | realistic | $\checkmark$ | $x$ | $x$ |
| BAMBOO | $\sim 16 \mathrm{k}$ | realistic | $\checkmark$ | $\checkmark$ | $x$ |
| LongBench | $\sim 8 \mathrm{k}$ | hybrid | $\checkmark$ | $x$ | $x$ |
| LooGLE | $\sim 20 \mathrm{k}$ | hybrid | $\checkmark$ | $\checkmark$ | $\times$ |
| InfiniteBench | $\sim 200 \mathrm{k}$ | hybrid | $\checkmark$ | $\checkmark$ | $x$ |
| Needle-in-a-haystack (NIAH) | any | synthetic | $x$ | $\checkmark$ | $\checkmark$ |
| Passkey / Line / KV Retrieval | any | synthetic | $x$ | $\checkmark$ | $\checkmark$ |
| RULER (Ours) | any | synthetic | $\checkmark$ | $\checkmark$ | $\checkmark$ |

Table 1: Comparison between existing long-context benchmarks and RULER. "Realistic" type refers to human-annotated while "synthetic" type refers to auto-generated. RULER includes diverse task domains beyond retrieval, reduces reliance on parametric knowledge with synthetic input, and offers flexibility to control the contexts for different sequence lengths and task complexities. In RULER, contexts can be adjusted by changing the volume or placement of relevant and distracted information.

4. Question Answering: we add distracting information to the input of existing shortcontext QA datasets to evaluate question answering capability at various context sizes.

Compared to existing realistic benchmarks (Table 1), RULER consists solely of synthetic tasks, which offer the flexibility to control sequence length and task complexity. The synthetic input in RULER reduces reliance on parametric knowledge, which interferes with the utilization of long-context input in realistic tasks (Shaham et al. 2023; Bai et al., 2023).

Using RULER, we benchmark GPT-4 (OpenAI: Josh Achiam et al., 2023) and nine opensource models with context length ranging from $4 \mathrm{k}$ to $128 \mathrm{k}$. Despite achieving nearly perfect performance on the vanilla NIAH test, all models exhibit large degradation on more complex tasks in RULER as sequence length increases. While all models claim context size of $32 \mathrm{k}$ tokens or greater, our results indicate that only four of them can effectively handle sequence length of $32 \mathrm{~K}$ by exceeding a qualitative threshold. Moreover, almost all models fall below the threshold before reaching the claimed context lengths. To obtain fine-grained model comparisons, we aggregate performance from $4 \mathrm{k}$ to $128 \mathrm{k}$ with two weighted average scores where the weights simulate the length distribution of real-world use cases. The top models - GPT-4, Command-R (Cohere, 2024), Yi-34B (Young et al. 2024), and Mixtral (Jiang et al. 2024), consistently outperform other models regardless of the chosen weighting scheme.

We further analyze Yi-34B, which claims context length of 200K and achieves the 2nd place on RULER among open-source models. Our results demonstrate large degradation in Yi's performance as we increase input length and task complexity. At large context sizes, Yi-34B often returns incomplete answers and fails to precisely locate the relevant information. Furthermore, we observe two behaviors emerging with the scaling of context size across multiple models: the increased reliance on parametric knowledge and the increased tendency to copy from context for non-retrieval tasks. Our additional ablations demonstrate that training on longer sequences does not always lead to better performance on RULER, and that larger model sizes positively correlate with better long-context capabilities. Finally, we show that non-Transformer architectures, such as RWKV and Mamba, still lag behind Transformer by large margins on RULER.

Our contributions are as follows:

- We propose a new benchmark RULER for evaluating long-context language models via synthetic tasks with flexible configurations.
- We introduce new task categories, specifically multi-hop tracing and aggregation, to test behaviors other than retrieval from long context.
- We evaluate ten long-context LMs using RULER and perform analysis across models and task complexities.

We open source RULER to spur future research in long-context language models $\cup^{1}$[^0]

## 2 Related Work

Long-context Language Models. Numerous long-context language models have been introduced lately owing to the progress in engineering, architectural, and algorithmic designs. Flash attention (Dao et al., 2022; Dao, 2023) and Ring attention (Liu et al., 2023) significantly reduce the memory footprint required for processing long context. Various sparse attention mechanisms (Child et al., 2019 Jaszczur et al. [2021) such as shifted sparse attention (Chen et al. 2024), dilated attention (Ding et al. 2023), and attention sinks (Han et al. 2023: Xiao et al. 2024b) were employed to enable efficient context scaling. Novel position embedding methods were proposed to improve length extrapolation in Transformers (Vaswani et al. 2017), including ALiBi (Press et al. 2022), xPOS (Sun et al., 2023b), and RoPE (Su et al. 2023) variants (Chen et al. 2023: Xiong et al. 2023: Peng et al. 2024: Liu et al. 2024b: Ding et al. 2024; Zhu et al.| 2024). Another line of research focuses on reducing context size. This can be achieved by caching previous context using recurrence mechanism (Zhang et al. 2024a: Bulatov et al. 2023: Martins et al. 2022: Wu et al. 2022), or preserving only the salient information within long context via retrieval (Xu et al.. 2024: Mohtashami \& Jaggil, 2023; Wang et al., 2024: Tworkowski et al., 2024: Xiao et al.. 2024a) or compression (Jiang et al. 2023). Finally, novel architectures (Gu et al.|[2022; Fu et al., 2023a; Poli et al. 2023; Fu et al. 2023b; Sun et al., 2023a) such as Mamba (Gu \& Dao, 2023) and RWKV (Peng et al.. 2023) have also been proposed to efficiently handle long-context input.

Long-context Benchmarks and Tasks. Our work is closely related to other works on benchmarking long-context language models. ZeroSCROLLS (Shaham et al. 2023) covers ten realistic natural language tasks, such as long-document QA and (query-based) summarization. L-Eval (An et al., 2024) also uses realistic data, which was filtered manually to ensure quality. LongBench (Bai et al. 2023) contains tasks in a bilingual setting. InfiniteBench (Zhang et al. . 2024b) includes tasks with length greater than 100K tokens. LTM (Castillo et al., 2024) targets the evaluation of long-term conversations. To isolate the effect of parametric knowledge, previous works (Dong et al., 2023; Li et al., 2023b) also propose to use documents posted online later than a certain cutoft date, or leverage extremely low-resource materials (Tanzer et al. 2024). Compared to realistic benchmarks, synthetic tasks are more flexible to control the setup (e.g., sequence length and task complexity) and less affected by parametric knowledge. Recent works have mostly focused on retrieval-based synthetic tasks(Kamradt. 2023: Mohtashami \& Jaggi, 2023: Li et al., 2023a Liu et al. 2024 c), with a few on other types of long-context usage, including various types of reasoning (Tay et al., 2021) and long-range discourse modeling (Sun et al., 2022).

## 3 The RULER Benchmark

RULER comprises tasks across four categories: retrieval, multi-hop tracing, aggregation, and question answering with all tasks configurable for varying length and complexity (see Table 2 ).

### 3.1 Retrieval: Needle-in-a-haystack (NIAH)

Recent works (Reid et al., 2024; Anthropic, 2023) commonly employ the needle-in-ahaystack (Kamradt [2023, NIAH) test to evaluate long-context modeling capability. The NIAH test is reminiscent of the extensively studied (Hopfield, 1982; Graves et al., 2014; Olsson et al. 2022: Arora et al., 2024) associative recall tasks, in which relevant intormation needs to be retrieved from context given a sufficient query. In RULER, we include multiple retrieval-based tasks, extending the vanilla NIAH test to evaluate models based on three criteria. Concretely, the retrieval capability should be (1) agnostic to the type of the "needle" and the "haystack", (2) strong enough to disregard hard distractors, and (3) of high recall when multiple items need to be retrieved. Based on these criteria, we develop four NIAH tasks. The "needle" in each of these tasks is a key-value pair inserted into the "haystack" (long distractor texts). The query is located at the end of the sequence and serves as a cue for matching the keys in the context and subsequently retrieving the associated values.

| Task | Configuration | Example |
| :---: | :---: | :---: |
| Single <br> NIAH <br> (S-NIAH) | type_key $=$ word <br> type_value $=$ number <br> type_haystack $=$ essay <br> size_haystack $\propto$ context length | (essays) <br> One of the special magic numbers for long-context is: $12345 . . . . .$. in the <br> What is the special magic number for long-context mentioned in the <br> provided text? <br> Answer: 12345 |
| Multi-keys <br> NIAH <br> (MK-NIAH) | num_keys $=2$ <br> type_key $=$ word <br> type_value $=$ number <br> type_haystack $=$ essay <br> size_haystack $\propto$ context length | (essays) th.. special magic numbers for long-context is: 12345. <br> One of the special magic numbers for large-model is: 54321 . <br> One of the spect <br> What is the special magic number for long-context mentioned in the <br> provided text? <br> Answer: 12345 |
| Multi-values <br> NIAH <br> (MV-NIAH) | num_values $=2$ <br> type_key $=$ word <br> type_value $=$ number <br> type_haystack $=$ essay <br> size_haystack $\propto$ context length | (essays) th. special magic numbers for long-context is: 12345. <br> One of the special magic numbers for long-context is: 54321. <br> One of the speciont <br> What are all the special magic numbers for long-context mentioned in the <br> provided text? <br> Answer: 1234554321 |
| Multi-queries <br> NIAH <br> (MQ-NIAH) | num_queries $=2$ <br> type_key $=$ word <br> type_value $=$ number <br> type_haystack $=$ essay <br> size_haystack $\propto$ context length | (essays) th. special magic numbers for long-context is: 12345. <br> One of the special magic numbers for large-model is: 54321 . <br> One of the special <br> What are all the special magic numbers for long-context and large-model <br> mentioned in the provided text? <br> Answer: 1234554321 |
| Variable <br> Tracking <br> (VT) | num_chains $=2$ <br> num_hops $=2$ <br> size_noises $\propto$ context length | ![](https://cdn.mathpix.com/cropped/2024_05_26_086a73dc0d9359f2aa47g-04.jpg?height=175&width=786&top_left_y=1063&top_left_x=956) |
| Common Words <br> Extraction <br> (CWE) | freq_cw $=2$, freq_ucw $=1$ <br> num_cw $=10$ <br> num_ucw $\propto$ context length | aaa bbb ccc aaa ddd eee ccc fff ggg hhh iii iii ..... <br> What are the 10 most common words in the above list? <br> Answer: aaa ccc iii...... |
| Frequent Words <br> Extraction <br> (FWE) | $\alpha=2$ <br> num_word $\propto$ context length | aaa bbb ccc aaa ddd eee ccc fff ggg aaa hhh aaa ccc iii iii. ..... <br> What are the 3 most frequently appeared words in the above coded text? <br> Answer: aaa ccc iii |
| Question <br> Answering <br> (QA) | dataset = SQuAD <br> num_document $\propto$ context length | Document 1: $\ldots . . . \mathrm{aaa} \ldots . . .$. <br> Document 2: $. . . . \mathrm{bbb} \ldots . . .$. <br> Document 3: ......ccc..... <br> Question: question <br> Answer: bbb |

Table 2: Task examples with flexible configurations in RULER. We use different colors to highlight queries, keys, values, and distractors in our examples.

- Single NIAH (S-NIAH): This is the vanilla NIAH test where a single "needle' ${ }^{2}$ needs to be retrieved from the "haystack". The query/key/value can take the form of words, numbers ( 7 digits), or UUIDs ( 32 digits). The "haystack" can be repeated noise sentences ${ }^{3}$ or Paul Graham essays (Kamradt, 2023).
- Multi-keys NIAH (MK-NIAH): Multiple "needles" are inserted into the "haystack", and only one of them needs to be retrieved. The additional "needles" are hard distractors. The most challenging setting is a version where the "haystack" is filled with distractor needles.
- Multi-values NIAH (MV-NIAH): Multiple "needles" sharing the same key are inserted into the "haystack". All values associated with the same key need to be retrieved.
- Multi-queries NIAH (MQ-NIAH): Multiple "needles" are inserted into the "haystack". All "needles" with distinct keys need to be retrieved. This is the same multi-query associative recall task setup used by Arora et al. (2024). Together with MV-NIAH, these two tasks evaluate the retrieval capability without missing any critical information.[^1]![](https://cdn.mathpix.com/cropped/2024_05_26_086a73dc0d9359f2aa47g-05.jpg?height=264&width=1094&top_left_y=290&top_left_x=512)

Figure 1: In aggregation tasks, we sample words from a vocabulary following the two distributions above. The common words extraction (CWE) samples from uniform distributions. In the frequent words extraction (FWE), the frequency of each word is determined by its rank in the vocabulary and the parameter $\alpha$ of Zeta distribution.

### 3.2 Multi-hop Tracing: Variable Tracking (VT)

Effective discourse comprehension (van Dijk \& Kintsch, 1983) is contingent upon successful recognition of newly mentioned entities and establishing the chain of references co-referring to the same entity (Karttunen, 1969) throughout the long context. We develop a new task variable tracking to emulate a minimal coreference chain resolution $(\mathrm{Ng}, 2010)$ task. This task checks the behavior of tracking relevant co-occurrence patterns and drawing skipped connections within long input. Specifically, a variable $X 1$ is initialized with a value $V$, followed by a linear chain of variable name binding statements (e.g., $X 2=X 1, X 3=X 2, \ldots$ ), which are inserted at various positions of the input. The objective is to return all variable names pointing to the same value $V$. The task complexity can be increased by adding more hops (i.e., the times of name binding) or more chains, similar to adding hard distractors in MK-NIAH.

### 3.3 Aggregation: Common Words (CWE) and Frequent Words Extraction (FWE)

In RULER, we introduce a new category as a proxy for summarization tasks where relevant information constitutes much larger portion of the context, and the target output depends on accurate aggregation of the relevant input. Concretely, we construct an input sequence by sampling words from a pre-defined (synthetic) word list. In the common word extraction task (CWE), words are sampled from discrete uniform distributions, with the number of common words fixed while the number of uncommon words increases with the sequence length. In the frequent words extraction task (FWE), words are sampled from Zeta distribution ${ }^{4}$ Figure 1 shows an illustration of word frequency in the constructed input. A model needs to return the top- $K$ frequent words in the context. In CWE, $K$ equals to the number of common words. In FWE, we set $K$ to 3, as increasing $K$ leads to poor performance even at small context sizes for most models. The task complexity can be adjusted by varying the number of common words or the parameter of Zeta distribution.

### 3.4 Question Answering (QA)

The majority of existing QA datasets (Rajpurkar et al., 2018; Yang et al., 2018; Trivedi et al., 2022) are designed to answer questions based on short passages. These datasets can be extended to simulate long-context input by adding distracting information. In this task category, we insert the golden paragraphs (i.e., the paragraphs that contain answers) into paragraphs randomly sampled from the same dataset. This category is a real-world adaptation (Ivgi et al., 2023) of NIAH, where the question serves as the query, the golden paragraphs are the "needles", and the distracting paragraphs form the "haystack".[^2]

| Models | Claimed <br> Length | Effective <br> Length | $4 \mathbf{k}$ | $8 \mathbf{k}$ | 16k | $32 \mathbf{k}$ | $64 k$ | 128k | Avg. | wAvg. <br> (inc) | wAvg. <br> (dec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Llama2-7B (chat) | $4 \mathrm{k}$ | - | 85.6 |  |  |  |  |  |  |  |  |
| GPT-4 | $128 \mathrm{k}$ | $64 \mathrm{k}$ | 96.6 | 96.3 | 95.2 | 93.2 | $\underline{87.0}$ | 81.2 | 91.6 | $89.0_{(1 \mathrm{st})}$ | $94.1_{(1 \mathrm{st})}$ |
| Command-R (35B) | $128 \mathrm{k}$ | $32 \mathrm{k}$ | $\overline{93.8}$ | $\overline{93.3}$ | $\overline{92.4}$ | $\overline{89.5}$ | $\overline{84.9}$ | 76.0 | 88.3 | $85.5_{(2 \mathrm{nd})}$ | $91.1_{(2 n d)}$ |
| Yi (34B) | $200 \mathrm{k}$ | $32 \mathrm{k}$ | 93.3 | 92.2 | $\overline{91.3}$ | $\underline{87.5}$ | 83.2 | 77.3 | 87.5 | $84.8_{(3 \mathrm{th})}$ | $90.1_{(3 \mathrm{th})}$ |
| Mixtral (8x7B) | $32 \mathrm{k}$ | $32 \mathrm{k}$ | 94.9 | 92.1 | 92.5 | $\underline{85.9}$ | 72.4 | 44.5 | 80.4 | $72.8_{(4 \mathrm{th})}$ | $87.9_{(4 \text { th })}$ |
| Mistral (7B) | $32 \mathrm{k}$ | $16 \mathrm{k}$ | $\overline{93.6}$ | $\overline{91.2}$ | $\overline{87.2}$ | $\overline{75.4}$ | 49.0 | 13.8 | 68.4 | $55.6_{(7 \mathrm{th})}$ | $81.2_{\text {(5th) }}$ |
| ChatGLM (6B) | $128 \mathrm{k}$ | $4 \mathrm{k}$ | $\underline{87.8}$ | $\overline{83.4}$ | 78.6 | 69.9 | 56.0 | 42.0 | 69.6 | $62.0_{(6 \mathrm{th})}$ | $77.2_{\text {(6th) }}$ |
| LWM (7B) | $1 \mathrm{M}$ | $<4 \mathrm{k}$ | $\overline{82.3}$ | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 | 72.8 | $69.9_{(5 \mathrm{th})}$ | $75.7_{(7 \mathrm{th})}$ |
| Together (7B) | $32 \mathrm{k}$ | $4 \mathrm{k}$ | $\underline{88.2}$ | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 | 50.3 | $33.8_{(8 \mathrm{th})}$ | $66.7_{(8 \mathrm{th})}$ |
| LongChat (7B) | $32 \mathrm{k}$ | $<4 \mathrm{k}$ | $\overline{84.7}$ | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | 49.1 | $33.1_{(9 \mathrm{th})}$ | $65.2_{\text {(9th) }}$ |
| LongAlpaca (13B) | $32 \mathrm{k}$ | $<4 \mathrm{k}$ | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 | 36.3 | $24.7_{\text {(10th) }}^{\text {(1) }}$ | $47.9_{(10 \text { th })}^{(}$ |

Table 3: Long Context Performance (\%) of selected models evaluated at length from $4 \mathrm{k}$ to $128 \mathrm{k}$. Each score is computed by averaging accuracy of 13 tasks in RULER. The performance exceeding the Llama2-7B performance at $4 \mathrm{~K}(85.6 \%)$ is underlined. The effective context length is the maximum length passing this threshold. Weighted average score (wAvg.) aggregates performance across all context sizes, with the weights linearly increasing (inc) or decreasing (dec) to simulate length distribution of real-world usage. We put the rank of each model in the subscript. More details about the selected models are in Appendix A.

## 4 Experiments \& Results

Models \& Inference setup We select 10 long-context LLMs, including 9 open-source models and one closed-source model (GPT-4), covering diverse model sizes (6B to $8 \times 7 \mathrm{~B}$ with MoE architecture) and claimed context lengths ( $32 \mathrm{k}$ to 1M). Complete information about these models is included in Appendix A. We evaluate all models using vLLM (Kwon et al. 2023), an LLM serving system with efficient KV cache memory management. For all models, we run the inference in BFloat16 on 8 NVIDIA A100 GPUs with greedy decoding.

Task configurations We test all models on 13 tasks ranging diverse complexities from the four categories of RULER. The test configurations have been selected (shown in Appendix B) based on a task correlational study described in Appendix C. For each task, we evaluate each model with 500 examples generated for each length from the series $(4 k, 8 k, 16 k, 32 k$, $64 \mathrm{k}, 128 \mathrm{k})$, while complying with each model's necessary chat template ${ }^{5}$ To prevent the model from refusing to answer a query or generating explanations, we append the task input with an answer prefix and check the presence of the target output with recall-based accuracy.

Effective Context Size We notice large performance degradation in all models as we increase input length in RULER. To determine the maximum context size a model can effectively handle, we grade each model with a fixed threshold, passing which indicates satisfactory performance at the length of evaluation. We use the performance of Llama2-7b model at the $4 \mathrm{~K}$ context length as the threshold. We report in Table 3 the maximum length exceeding the threshold as the "effective length" along with the "claimed length".

Model Ranking Criteria While the threshold-based grading reveals the discrepancy between claimed and effective length, it lacks details for fine-grained model comparisons. As such, we use a weighted average score to aggregate model performance across various context sizes. We rank models under two weighting schemes: wAvg. (inc) and wAvg. (dec) where the weight linearly increases and decreases with sequence length respectively. Ideally, the weight for each length should be determined by the length distribution of model usage, here we choose the two schemes to simulate the scenarios where longer sequences (inc) or shorter sequences (dec) dominate the distribution.[^3]

Main Results We include the results of ten long-context LMs in comparison with the Llama2-7B baseline in Table $3^{36}$ The performance at a certain length is the average of all 13 tasks in RULER. While these models all claim effective context of $32 \mathrm{~K}$ tokens or greater, none of them maintains performance above the Llama2-7B baseline at their claimed length, except for Mixtral, which achieves moderate performance on length doubling the claimed $32 \mathrm{~K}$ context size. Despite achieving nearly perfect performance on the passkey retrieval and the vanilla NIAH task (shown in Appendix E), all models exhibit large degradation in RULER as sequence length increases. The best performant model on RULER is GPT-4, which has the highest performance at length of $4 \mathrm{k}$ and demonstrates the least but non-marginal degradation (15.4) when extending the context to $128 \mathrm{~K}$. The top three ranked open-source models, Command-R, Yi-34B and Mixtral, all use a large base frequency in RoPE and are larger in parameter size than other models. Despite having been trained with context size of $1 \mathrm{M}$, the LWM performs worse than Llama2-7B even at $4 \mathrm{~K}$. However, it shows smaller degradation with the increase of context size, therefore achieves higher rank than Mistral-7B when longer sequences receive larger weight (wAvg. inc). This result suggests a trade-off in evaluation between absolute performance on short sequences and the relative degradation with the scaling of context size.

## 5 Task Error Analysis

We evaluate Yi-34B-200K, the 2nd best open-source model on RuLER, with increased input lengths (up to $256 \mathrm{~K}$ ) on more complex tasks to understand the effect of task configurations and failure modes on RULER.

Non-robustness to "needle" types. Figure 2(left) shows that while Yi achieves almost perfect performance when using needle of word-number pair in the standard passkey retrieval and vanilla NIAH, performance degrades when the needle takes other forms. We observe the largest degradation in the task of retrieving UUIDs, for which Yi sometimes fail to return the complete 32 digits given long ( $>128 \mathrm{~K}$ ) input context.

Failure to ignore distractors. Figure 2 (middle-left) shows that increasing the number of distracting needles steadily lowers performance, with Yi dropping by $\sim 40$ points at $256 \mathrm{~K}$ in the extreme version, where the context is full of irrelevant needles (\#K=FULL). Error analysis reveals that Yi fails to effectively ignore the hard distractors given long input context, thus incorrectly retrieves values associated with the distractor keys. In the extreme version, Yi often returns values from the vicinity of the target, suggesting coarse match of the range but the lack of precision to locate the key when the target is in-distribution of the noises.

Return incomplete information. Consistent with previous works (Liu et al. 2024a Reid et al. 2024), we notice significant degradation in performance when the model needs to retrieve multiple items from a long input. For instance, increasing the number of queries from 1 to 8 drops the performance by $\sim 15$ points (Figure 2 right). When the model needs to retrieve multiple values associated with the same key (Figure 2 middle-right), Yi often outputs duplicated answers without returning the complete set of values, implying uneven associations between the key and each of its values.

Tendency to copy from context. We notice that Yi has a strong tendency to copy from context verbatim when scaling the input length. This tendency is most notable in variable tracking (VT) and common words extraction (CWE) where we include one in-context demonstration at the beginning of the sequence. Over $80 \%$ of Yi's output in the CWE task at $128 \mathrm{~K}$ is simply a string copied from the one-shot example, whereas the copying is nonexistent for short sequences. ㄱ This copying behavior is also present in the LWM model and LongAlpaca, however it is less prevalent in other models, such as Mixtral. This finding further reinforces the need to test behaviors other than retrieval given long input context.[^4]![](https://cdn.mathpix.com/cropped/2024_05_26_086a73dc0d9359f2aa47g-08.jpg?height=340&width=1328&top_left_y=281&top_left_x=388)

Figure 2: Performance of Yi-34B in the needle-in-a-haystack (NIAH) tasks. By default, we use word-number as the key-value pair and Paul Graham essays as the haystack. Yi is not robust to the change of needle types and degrades with the increasing amount of distractors. (W: words; N: numbers; U: UUIDs; Full: entire haystack).
![](https://cdn.mathpix.com/cropped/2024_05_26_086a73dc0d9359f2aa47g-08.jpg?height=348&width=1330&top_left_y=866&top_left_x=386)

Figure 3: Performance of Yi-34B in variable tracking (VT), frequent words extraction (FWE), and QA tasks across different task complexities. Yi shows large degradation and distinct trends with scaled context size in these non-retrieval tasks, demonstrating the need to evaluate behavior beyond retrieval from context.

Unreliable tracking within context. For the variable tracking task, both adding more chains and more hops contribute to large degradation in Yi's performance. Yi consistently degrades in the more-hops setting as we increase context size (Figure 3left), whereas the degradation in the more-chains setting is most significant for lengths greater than 128K (Figure 3 middleleft). Besides the aforementioned copying issue, Yi makes errors due to incorrectly returning empty strings or variables from other chains, implying a lack of ability to reliably trace the same entity within long context. These errors are also frequently observed in models that do not exhibit the copying behavior.

Failure to accurately aggregate. We observe two common failure modes in aggregation tasks: incorrect use of parametric knowledge and inaccurate aggregation. Models that do not exhibit the copying issue in the CWE task, sometimes ignore the contextual information and instead use parametric knowledge to answer the query, especially at large context sizes. For instance, Mistral (7b-instruct-v0.2) returns high frequency words, such as "the", "an", "a", as output without counting the words in context. For the FWE task which demonstrates less the copying issue, Yi fails to correctly output the top frequent words as we decrease the $\alpha$ in Zeta distribution (Figure 3 middle-right). Decreasing $\alpha$ leads to smaller difference in frequency among words, increasing the difficulty to distinguish the top-frequent words.

Frequent hallucination in long-context QA. For the QA tasks, Yi's performance approaches its no-context baseline as we extend the context with distracting paragraphs (Figure 3 right). The degradation stems primarily from hallucination and reduced reliance on contextual information. We notice that, at large context sizes, model predictions sometimes are irrelevant to the question and can coincide with the answers of its no-context baseline. The overall worse performance in QA tasks confirms that the fuzzy matching between a query and a relevant paragraph in long context is a more challenging setting than the simplistic NIAH tests, where keys can be exactly located in context.
![](https://cdn.mathpix.com/cropped/2024_05_26_086a73dc0d9359f2aa47g-09.jpg?height=340&width=1348&top_left_y=281&top_left_x=388)

Figure 4: (Left \& middle left): Comparison of LargeWorldModel (LWM) series trained up to various context sizes with fixed parameter size of 7B. (Middle right): Comparison of Yi suite models with different parameter sizes with controlled training context length of 200K. (Right): Performance of non-Transformer architectures lags behind the Transformer baseline Llama2-7B by large margin. Length extrapolation is presented with dashed lines.

## 6 Model Analysis

Effect of training context length. Do models trained with larger context sizes perform better on RULER? We evaluate the suite of LargeWorldModels (Liu et al., 2024a, LWM) of equal parameter size and trained up to various context lengths. Figure 4 (left \& middle-left) shows that larger context sizes overall lead to better performance, but the ranking can be inconsistent for long sequences. For instance, the model trained with $1 \mathrm{M}$ context size (LWM-1M) is worse than the one with $512 \mathrm{~K}$ at length of $256 \mathrm{~K}$, likely due to insufficient training for adjusting to the new base frequency in RoPE. Moreover, we observe abrupt performance drops when models need to extrapolate to unseen lengths (e.g., LMW-128K given input of $256 \mathrm{~K}$ ), and almost linear degradation with input length on log scale within the max training context size.

Effect of model size The top models in our main results are much larger than other models. To ablate the effect of model size, we evaluate Yi-34B-200k, Yi-9B-200k, and Yi-6B-200k, all trained up to the same context length using the same data blend. Figure 4 (middle-right) shows that the 34B model is significantly better than the 6B model on RULER for both performance at length of $4 \mathrm{~K}$ and the relative degradation, suggesting the benefit of scaling model sizes for better long-context modeling.

Effect of architecture We evaluate the effective context length for two models with nonTransformer architectures: RWKV-v5 (Peng et al., 2023) and Mamba-2.8B-slimpj (Gu \& Dao, 2023). We find that both models demonstrate significant degradation when extending context size to $8 \mathrm{~K}$, and both underperform the Transformer baseline Llama2-7B by large margins up till the length of $4 \mathrm{~K}$, beyond which Llama2 shows poor length extrapolation performance (Figure 4 right).

## 7 Conclusion

We present RULER, a synthetic benchmark for evaluating long-context language models. RULER contains diverse task categories, retrieval, multi-hop tracing, aggregation and question answering, providing a flexible and comprehensive evaluation of LLM's long-context capabilities. We benchmark ten long-context LMs using RULER with context sizes ranging from $4 \mathrm{~K}$ to $128 \mathrm{~K}$. Despite achieving perfect results in the widely used needle-in-a-haystack test, all models fail to maintain their performance in other tasks of RULER as we increase input length. We observe common failure modes at large context sizes, including the failure to ignore distractors and ineffective utilization of long context (e.g., simply copy from context or use parametric knowledge instead). We show that RULER is challenging for even the top-ranked open-source models as we increase task complexity. Our analysis further reveals the large potential for improvement on RULER and the benefit of scaling model sizes in achieving better long context capabilities.

## References

AI21. Introducing jamba: Ai21's groundbreaking ssm-transformer model, 2024. URL https://www.ai21.com/blog/announcing-jamba.

Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models. In ICLR, 2024.

Anthropic. Long context prompting for Claude 2.1. Blog, 2023. URL https://www anthropic.com/index/claude-2-1-prompting.

Anthropic. Introducing the next generation of claude, 2024. URLhttps://www. anthropic. com/news/claude-3-family.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. Zoology: Measuring and improving recall in efficient language models. In ICLR, 2024.

Yushi Bai et al. LongBench: A bilingual, multitask benchmark for long context understanding. arXiv:2308.14508, 2023.

Aydar Bulatov, Yuri Kuratov, and Mikhail S Burtsev. Scaling Transformer to 1M tokens and beyond with RMT. arXiv:2304.11062, 2023.

David Castillo, Joseph Davidson, Finlay Gray, José Solorzano, and Marek Rosa. Introducing GoodAI LTM benchmark. Blog, 2024. URL https://www.goodai.com/ introducing-goodai-ltm-benchmark/

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. In ICLR, 2023.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. LongLoRA: Efficient fine-tuning of long-context large language models. In ICLR, 2024.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse Transformers. arXiv:1904.10509, 2019.

Cohere. Command r, 2024. URL https://docs.cohere.com/docs/command-r\# model-details

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. arxiv:2307.08691, 2023.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.

Jiayu Ding et al. LongNet: Scaling Transformers to 1,000,000,000 tokens. arXiv:2307.02486, 2023.

Yiran Ding et al. LongRoPE: Extending LLM context window beyond 2 million tokens. arXiv:2402.13753, 2024.

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. Bamboo: A comprehensive benchmark for evaluating long text modeling capacities of large language models. arXiv:2309.13345, 2023.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. GLM: General language model pretraining with autoregressive blank infilling. In Proc of the 60th Annual Meeting of the ACL (Volume 1: Long Papers), pp. 320-335, 2022.

Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. Hungry Hungry Hippos: Towards language modeling with state space models. In ICLR, 2023a.

Daniel Y. Fu et al. Simple hardware-efficient long convolutions for sequence modeling. ICML, 2023b.

Yao Fu et al. Data engineering for scaling language models to 128k context. arXiv:2402.10171, 2024.

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural Turing machines. arXiv:1410.5401, 2014.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In ICLR, 2022.

Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv:2308.16137, 2023.

John J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proc of the National Academy of Sciences of the United States of America, 79 8: $2554-8,1982$.

Maor Ivgi, Uri Shaham, and Jonathan Berant. Efficient long-text understanding with shorttext models. Transactions of the $A C L, 11: 284-299,2023$.

Sam Ade Jacobs et al. DeepSpeed Ulysses: System optimizations for enabling training of extreme long sequence Transformer models. arXiv:2309.14509, 2023.

Sebastian Jaszczur et al. Sparse is enough in scaling transformers. In NeurIPS, 2021.

Albert Q Jiang et al. Mixtral of experts. arXiv:2401.04088, 2024.

Huiqiang Jiang et al. LongLlmLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression. arXiv:2310.06839, 2023.

Gregory Kamradt. Needle In A Haystack - pressure testing LLMs. Github, 2023. URL https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main

Lauri Karttunen. Discourse referents. In COLING, 1969.

George Kingsley Zipf. Selected studies of the principle of relative frequency in language. Harvard university press, 1932.

Woosuk Kwon et al. Efficient memory management for large language model serving with paged attention. In Proc. of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Dacheng Li, Rulin Shao, et al. How long can open-source LLMs truly promise on context length?, 2023a. URL https://1msys.org/blog/2023-06-29-1ongchat

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts? arXiv:2311.04939, 2023b.

Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise Transformers for near-infinite context. In $I C L R, 2023$.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with Ring Attention. arxiv:2402.08268, 2024a.

Jiaheng Liu et al. E2-LLM: Efficient and extreme length extension of large language models. arXiv:2401.06951, 2024b.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the ACL, 12:157-173, 2024c.

Pedro Henrique Martins, Zita Marinho, and Andre Martins. $\infty$-former: Infinite memory Transformer. In Proc. of the 60th Annual Meeting of the ACL (Volume 1: Long Papers), 2022.

Mistral.AI. La plateforme, 2023. URL https://mistral.ai/news/la-plateforme/

Amirkeivan Mohtashami and Martin Jaggi. Landmark attention: Random-access infinite context length for Transformers. In Workshop on Efficient Systems for Foundation Models @ ICML, 2023.

Vincent Ng. Supervised noun phrase coreference research: The first fifteen years. In Proc. of the 48th Annual Meeting of the ACL, 2010.

Catherine Olsson et al. In-context learning and induction heads. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/in-context-learning-and-inductionheads/index.html.

OpenAI: Josh Achiam et al. GPT-4 technical report. arXiv:2303.08774, 2023.

Bo Peng et al. RWKV: Reinventing RNNs for the transformer era. In EMNLP, 2023.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In ICLR, 2024.

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena hierarchy: Towards larger convolutional language models. In ICML, 2023.

Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In ICLR, 2022.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for SQuAD. In Proc. of the 56th Annual Meeting of the ACL (Volume 2: Short Papers), 2018.

Machel Reid et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proc. of the 58th Annual Meeting of the ACL, 2020.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. ZeroSCROLLS: A zero-shot benchmark for long text understanding. In EMNLP, 2023.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with rotary position embedding. arXiv:2104.09864, 2023.

Simeng Sun, Katherine Thai, and Mohit Iyyer. ChapterBreak: A challenge dataset for long-range language models. In Proc. of the 2022 Conference of the North American Chapter of the ACL: Human Language Technologies, 2022.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to Transformer for large language models. arXiv:2307.08621, 2023a.

Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable Transformer. In Proc. of the 61st Annual Meeting of the ACL (Volume 1: Long Papers), 2023b.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. A benchmark for learning to translate a new language from one grammar book. In ICLR, 2024.

Yi Tay et al. Long Range Arena: A benchmark for efficient Transformers. In ICLR, 2021.

Together.AI. Preparing for the era of 32k context: Early learnings and explorations, 2023a. URLhttps://www.together.ai/blog/llama-2-7b-32k

Together.AI. Llama-2-7b-32k-instruct - and fine-tuning for llama-2 models with together api, 2023b. URL https://www.together. ai/blog/1lama-2-7b-32k-instruct.

Hugo Touvron et al. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288, 2023.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the $A C L, 10$ : $539-554,2022$.

Szymon Tworkowski et al. Focused Transformer: Contrastive training for context scaling. NeurIPS, 36, 2024.

Teun A. van Dijk and Walter Kintsch. Strategies of discourse comprehension. In Academic Press, 1983.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. NeurIPS, 36, 2024.

Thomas Wolf et al. Huggingface's Transformers: State-of-the-art natural language processing. arXiv:1910.03771, 2019.

Qingyang Wu, Zhenzhong Lan, Kun Qian, Jing Gu, Alborz Geramifard, and Zhou Yu. Memformer: A memory-augmented Transformer for sequence modeling. In Findings of the ACL: AACL-IJCNLP, 2022.

X.AI. Announcing grok-1.5, 2024. URL/https://x.ai/blog/grok-1.5

Chaojun Xiao et al. InfLLM: Unveiling the intrinsic capacity of LLMs for understanding extremely long sequences with training-free memory. arXiv:2402.04617, 2024a.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, $2024 \mathrm{~b}$.

Wenhan Xiong et al. Effective long-context scaling of foundation models. arXiv:2309.16039, 2023.

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context large language models. In ICLR, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In $E M N L P, 2018$.

Alex Young et al. Yi: Open foundation models by 01.AI. arXiv:2403.04652, 2024.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Soaring from $4 \mathrm{k}$ to 400k: Extending LLM's context with activation beacon. arXiv:2401.03462, $2024 a$.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. $\infty$ bench: Extending long context evaluation beyond 100k tokens. arXiv:2402.13718, 2024b.

Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. PoSE: Efficient context window extension of LLMs via positional skip-wise training. In ICLR, 2024.
</end of paper 4>


