<paper 0>
# LLMRec: Large Language Models with Graph Augmentation for Recommendation 

Wei Wei<br>University of Hong Kong<br>weiweics@connect.hku.hk<br>Qinyong Wang<br>Baidu Inc.<br>wangqinyong@baidu.com<br>Junfeng Wang<br>Baidu Inc.<br>wangjunfeng@baidu.com

Xubin Ren<br>University of Hong Kong<br>xubinrencs@gmail.com<br>Lixin $\mathrm{Su}$<br>Baidu Inc.<br>sulixinict@gmail.com<br>Dawei Yin<br>Baidu Inc.<br>yindawei@acm.org

Jiabin Tang<br>University of Hong Kong<br>jiabintang77@gmail.com<br>Suqi Cheng<br>Baidu Inc.<br>chengsuqi@gmail.com<br>Chao Huang<br>University of Hong Kong<br>chaohuang75@gmail.com


#### Abstract

The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuitively from the natural language perspective. By employing these strategies, we address the challenges posed by sparse implicit feedback and low-quality side information in recommenders. Besides, to ensure the quality of the augmentation, we develop a denoised data robustification mechanism that includes techniques of noisy implicit feedback pruning and MAE-based feature enhancement that help refine the augmented data and improve its reliability. Furthermore, we provide theoretical analysis to support the effectiveness of LLMRec and clarify the benefits of our method in facilitating model optimization. Experimental results on benchmark datasets demonstrate the superiority of our LLMbased augmentation approach over state-of-the-art techniques. To


[^0]ensure reproducibility, we have made our code and augmented data publicly available at: https://github.com/HKUDS/LLMRec.git.

## CCS CONCEPTS

- Information systems $\rightarrow$ Recommender systems.


## KEYWORDS

Large Language Models, Graph Learning, Data Augmentation, Contentbased Recommendation, Multi-modal Recommendation, Collaborative Filtering, Data Sparsity, Bias in Recommender System

## ACM Reference Format:

Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. LLMRec: Large Language Models with Graph Augmentation for Recommendation . In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM '24), March 4-8, 2024, Merida, Mexico. ACM, Merida, Mexico, 10 pages. https://doi.org/10.1145/3616855.3635853

## 1 INTRODUCTION

Recommender systems play a crucial role in mitigating information overload by providing online users with relevant content [27, 44]. To achieve this, an effective recommender needs to have a precise understanding of user preferences, which is not limited to analyzing historical interaction patterns but also extends to incorporating rich side information associated with users and items [61].

In modern recommender systems, such as Netflix, the side information available exhibits heterogeneity, including item attributes [53], user-generated content [7, 28], and multi-modal features [52] encompassing both textual and visual aspects. This diverse content offer distinct ways to characterize user preferences. By leveraging such side information, models can obtain informative representations to personalize recommendations. However, despite significant progress, these methods often face challenges related to data scarcity and issues associated with handling side information.

Sparse Implicit Feedback Signals. Data sparsity and the coldstart problem hinder collaborative preference capturing [48]. While many efforts (e.g., NGCF [41], LightCGN [11]) tried powerful graph neural networks(GNNs) in collaborative filtering(CF), they face limits due to insufficient supervised signals. Some studies [33] used contrastive learning to add self-supervised signals (e.g., SGL [51],

SimGCL [54]). However, considering that real-world online platforms (e.g., Netflix, MovieLens) derive benefits from modal content, recent approaches, unlike general $\mathrm{CF}$, are dedicated to incorporating side information as auxiliary for recommenders. For example, MMGCN [50] and GRCN [49] incorporate item-end content into GNNs to discover high-order content-aware relationships. LATTICE [59] leverages auxiliary content to conduct data augmentation by establishing i-i relationships. Recent efforts (e.g., MMSSL [45], MICRO [58]) address sparsity by introducing self-supervised tasks that maximize the mutual information between multiple contentaugmented views. However, strategies for addressing data sparsity in recommender systems, especially in multi-modal content, can sometimes be limited. This is because the complexity and lack of side information relevance to $\mathrm{CF}$ can introduce distortions in the underlying patterns [49]. Therefore, it becomes crucial to ensure the accurate capture of realistic user preferences when incorporating side information in $\mathrm{CF}$, in order to avoid suboptimal results.

Data Quality Issues of Side Information. Recommender systems that incorporate side information often encounter significant issues that can negatively impact their performance. i) Data Noise is an important limitation faced by recommender systems utilizing side information is the issue of data noise[39], where attributes or features may lack direct relevance to user preferences. For instance, in a micro video recommender, the inclusion of irrelevant textual titles that fail to capture the key aspects of the video's content introduces noise, adversely affecting representation learning. The inclusion of such invalid information confuse the model and lead to biased or inaccurate recommendations. ii) Data heterogeneity[4] arises from the integration of different types of side information, each with its own unique characteristics, structures, and representations. Ignoring this heterogeneity leads to skewed distributions [26, 53]. Bridging heterogeneous gap is crucial for successfully incorporating side information uniformly. iii) Data incompleteness [15, 20] occurs when side information lacks certain attributes or features. For instance, privacy concerns[56] may make it difficult to collect sufficient user profiles to learn their interests. Additionally, items may have incomplete textual descriptions or missing key attributes This incompleteness impairs the model's ability to fully capture the unique characteristics of users and items, thereby affecting the accuracy of recommendations.

Having gained insight into data sparsity and low-quality encountered by modern recommenders with auxiliary content, this work endeavors to overcome these challenges through explicit augment potential user-item interactive edges as well as enhances user/item node side information (e.g., language, genre). Inspired by the impressive natural language understanding ability of large language models (LLMs), we utilize LLMs to augment the interaction graph. Firstly, LLMRec embraces the shift from an ID-based recommendation framework to a modality-based paradigm [17, 55]. It leverages large language models (LLMs) to predict user-item interactions from a natural language perspective. Unlike previous approaches that rely solely on IDs, LLMRec recognizes that valuable item-related details are often overlooked in datasets [18]. Natural language representations provide a more intuitive reflection of user preferences compared to indirect ID embeddings. By incorporating LLMs, LLMRec captures the richness and context of natural language, enhancing the accuracy and effectiveness of recommendations. Secondly, to elaborate further, the low-quality and incomplete side information is enhanced by leveraging the extensive knowledge of LLMs, which brings two advantages: i) LLMs are trained on vast real-world knowledge, allowing them to understand user preferences and provide valuable completion information, even for privacy-constrained user profiles. ii) The comprehensive word library of LLMs unifies embeddings in a single vector space, bridging the gap between heterogeneous features and facilitating encoder computations. This integration prevents the dispersion of features across separate vector spaces and provide more accurate results.

Enabling LLMs as effective data augmentors for recommenders poses several technical challenges that need to be addressed:

- C1: How to enable LLMs to reason over user-item interaction patterns by explicitly augmenting implicit feedback signals?
- C2: How to ensure the reliability of the LLM-augmented content to avoid introducing noise that could compromise the results? The potential of LLM-based augmentation to enhance recommenders by addressing sparsity and improving incomplete side information is undeniable. However, effectively implementing this approach requires addressing the aforementioned challenges. Hence, we have designed a novel framework LLMRec to tackle these challenges.

Solution. Our objective is to address the issue of sparse implicit feedback signals derived from user-item interactions while simultaneously improving the quality of side information. Our proposed LLMRec incorporates three LLM-based strategies for augmenting the interaction graph: $i$ ) Reinforcing user-item interaction edges, ii) Enhancing item attribute modeling, and iii) Conducting user profiling. To tackle $\mathbf{C 1}$ for ' $i$ )', we devise an LLM-based Bayesian Personalized Ranking (BPR)[34] sampling algorithm. This algorithm uncover items that users may like or dislike based on textual content from from natural language perspective. These items are then used as positive and negative samples in the BPR training process. It is important to note that LLMs are unable to perform all-item ranking, so the selected items are chosen from a candidate item pool provided by the base recommender for each user. During the node attribute generation process (corresponding to ' $i i$ )' and ' $i$ iii)'), we create additional attributes for each user/item using existing text and interaction history. However, it is important to acknowledge that both the augmented edges and node features can contain noise. To address $\mathbf{C} 2$, our denoised data robustification mechanism comes into play by integrating noisy edge pruning and feature MAE [36] to ensure the quality of the augmented data. In summary, our contributions can be outlined as follows:

- The LLMRec is the pioneering work that using LLMs for graph augmentation in recommender by augmenting: user-item interaction edges, ii) item node attributes, iii) user node profiles.
- The proposed LLMRec addresses the scarcity of implicit feedback signals by enabling LLMs to reason explicitly about user-item interaction patterns. Additionally, it resolves the low-quality side information issue through user/item attribute generation and a denoised augmentation robustification mechanism with the noisy feedback pruning and MAE-based feature enhancement.
- Our method has been extensively evaluated on real-world datasets, demonstrating its superiority over state-of-the-art baseline methods. The results highlight the effectiveness of our approach in
improving recommendation accuracy and addressing sparsity issues. Furthermore, in-depth analysis and ablation studies provide valuable insights into the impact of our LLM-enhanced data augmentation strategies, further solidifying the model efficacy.


## 2 PRELIMINARY

Recommendation with Graph Embedding. Collaborative filtering (CF) learns from sparse implicit feedback $\mathcal{E}^{+}$, with the aim of learning collaborative ID-corresponding embeddings $\mathbf{E}_{u}, \mathbf{E}_{i}$ for recommender prediction, given user $u \in \mathcal{U}$ and item $i \in \mathcal{I}$. Recent advanced recommenders employ GNNs to model complex high-order[37] u-i relation by taking $\mathcal{E}^{+}$as edges of sparse inter active graph. Therefore, the CF process can be separated into two stages, bipartite graph embedding, and u-i prediction. Optimizing collaborative graph embeddings $\mathbf{E}=\left\{\mathbf{E}_{u}, \mathbf{E}_{i}\right\}$ aims to maximize the posterior estimator with $\mathcal{E}^{+}$, which is formally presented below:

$$
\begin{equation*}
\mathbf{E}^{*}=\underset{\mathbf{F}}{\arg \max } p\left(\mathbf{E} \mid \mathcal{E}^{+}\right) \tag{1}
\end{equation*}
$$

Here, $p\left(\mathbf{E} \mid \mathcal{E}^{+}\right)$is to encode as much u-i relation from $\mathcal{E}^{+}$into $\mathbf{E}_{u}, \mathbf{E}_{i}$ as possible for accurate u-i prediction $\hat{y}_{u, i}=\mathbf{e}_{u} \cdot \mathbf{e}_{i}$.

Recommendation with Side Information. However, sparse interactions in $\mathcal{E}^{+}$pose a challenge for optimizing the embeddings. To handle data sparsity, many efforts introduced side information in form of node features $\mathbf{F}$, by taking recommender encoder $f_{\Theta}$ as feature graph. The learning process of the $f_{\Theta}$ (including $\mathbf{E}_{u}, \mathbf{E}_{i}$ and feature encoder) with side information $\mathbf{F}$ is formulated as maximizing the posterior estimator $p\left(\Theta \mid \mathbf{F}, \mathcal{E}^{+}\right)$:

$$
\begin{equation*}
\Theta^{*}=\underset{\Theta}{\arg \max } p\left(\Theta \mid \mathbf{F}, \mathcal{E}^{+}\right) \tag{2}
\end{equation*}
$$

$f_{\Theta}$ will output the final representation $\mathbf{h}$ contain both collaborative signals from $\mathbf{E}$ and side information from $\mathbf{F}$, i.e., $\mathbf{h}=f_{\Theta}\left(\mathbf{f}, \mathcal{E}^{+}\right)$.

Recommendation with Data Augmentation. Despite significant progress in incorporating side information into recommender, introducing low-quality side information may even undermine the effectiveness of sparse interactions $\mathcal{E}^{+}$. To address this, our LLMRec focuses on user-item interaction feature graph augmentation, which involves LLM-augmented u-i interactive edges $\mathcal{E}_{\mathcal{A}}$, and LLM-generated node features $\mathbf{F}_{\mathcal{A}}$. The optimization target with augmented interaction feature graph is as:

$$
\begin{equation*}
\Theta^{*}=\underset{\Theta}{\arg \max } p\left(\Theta \mid\left\{\mathbf{F}, \mathbf{F}_{\mathcal{A}}\right\},\left\{\mathcal{E}^{+}, \mathcal{E}_{\mathcal{A}}\right\}\right) \tag{3}
\end{equation*}
$$

The recommender $f_{\Theta}$ input union of original and augmented data, which consist of edges $\left\{\mathcal{E}^{+}, \mathcal{E}_{\mathcal{A}}\right\}$ and node features $\{\mathbf{F}, \mathbf{F} \mathcal{A}\}$, and output quality representation $\mathbf{h}$ to predicted preference scores $\hat{y}_{u, i}$ by ranking the likelihood of user $u$ will interact with item $i$.

## 3 METHODOLOGY

To conduct LLM-based augmentation, in this section, we address these questions: Q1: How to enable LLMs to predict u-i interactive edges? Q2: How to enable LLMs to generate valuable content? Q3: How to incorporate augmented contents into original graph contents? Q4: How to make model robust to the augmented data?

### 3.1 LLMs as Implicit Feedback Augmentor (Q1)

To directly confront the scarcity of implicit feedback, we employ LLM as a knowledge-aware sampler to sample pair-wise [34] u-i training data from a natural language perspective. This increases potential effective supervision signals and helps gain a better understanding of user preferences by integrating contextual knowledge into the u-i interactions. Specifically, we feed each user's historical interacted items with side information (e.g., year, genre) and an item candidates pool $C_{u}=\left\{i_{u, 1}, i_{u, 2}, \ldots, i_{u,\left|C_{u}\right|}\right\}$ into LLM. LLM then is expected to select items that user $u$ might be likely $\left(i_{u}^{+}\right)$ or unlikely $\left(i_{u}^{-}\right)$to interact with from $C_{u}$. Here, we introduce $C_{u}$ because LLMs can't rank all items. Selecting items from the limited candidate set recommended by the base recommender (e.g., MMSSL [45], MICRO [58]), is a practical solution. These candidates $C_{u}$ are hard samples with high prediction score $\hat{y}_{u i}$ to provide potential, valuable positive samples and hard negative samples. It is worth noting that we represent each item using textual format instead of ID-corresponding indexes [18]. This kind of representation offers several advantages: (1) It enables recommender to fully leverage the content in datasets, and (2) It intuitively reflects user preferences. The process of augmenting user-item interactive edges and incorporating it into the training data can be formalized as:

$$
\begin{equation*}
i_{u}^{+}, i_{u}^{-}=L L M\left(\mathbb{P}_{u}^{U I}\right) ; \quad \mathcal{E}_{B P R}=\mathcal{E} \cup \mathcal{E}_{\mathcal{A}} \tag{4}
\end{equation*}
$$

where $i_{u}^{+}, i_{u}^{-}$are positive and negative samples for BPR selected by LLMs from candidates $C_{u}$ for user $u$ based on input prompt $\mathbb{P}_{u}^{U I}$. The augmented dataset $\mathcal{E}_{\mathcal{A}}$ comprises pairwise training triplets $\left(u, i_{u}^{+}, i_{u}^{-}\right)$, i.e., $\mathcal{E}_{\mathcal{A}}=\left\{\left(u, i_{u}^{+}, i_{u}^{-}\right) \mid\left(u, i_{u}^{+}\right) \in \mathcal{E}_{\mathcal{A}}^{+},\left(u, i_{u}^{-}\right) \in \mathcal{E}_{\mathcal{A}}^{-}\right\}$. The textual u-i augmentation prompt $\mathbb{P}_{u}^{U I}$ encompasses different components: i) task description, ii) historical interactions, iii) candidates, and iv) output format description, as illustrated in Fig. 2 (a).

The utilization of LLMs-based sampler in this study to some extent alleviate noise (i.e., false positive) and non-interacted items issue (i.e., false negative) $[2,16]$ exist in raw implicit feedback. In this context, (i) false positive are unreliable u-i interactions, which encompass items that were not genuinely intended by the user, such as accidental clicks or instances influenced by popularity bias [40]; (ii) false negative represented by non-interacted items, which may not necessarily indicate user dispreference but are conventionally treated as negative samples [3]. By taking LLMs as implicit feedback augmentor, LLMRec enables the acquisition of more meaningful and informative samples by leveraging the remarkable reasoning ability of LLMs with the support of LLMs' knowledge. The specific analysis is supported by theoretical discussion in Sec. 3.4.1.

### 3.2 LLM-based Side Information Augmentation

3.2.1 User Profiling \& Item Attribute Enhancing (Q2). Leveraging knowledge base and reasoning abilities of LLMs, we propose to summarize user profiles by utilizing users' historical interactions and item information to overcome limitation of privacy. Additionally, the LLM-based item attributes generation aims to produce space-unified, and informative item attributes. Our LLM-based side information augmentation paradigm consists of two steps:

- i) User/Item Information Refinement. Using prompts derived from the dataset's interactions and side information, we enable LLM to generate user and item attributes that were not originally part of the dataset. Specific examples are shown in Fig. 2(b)(c).
- ii) LLM-enhanced Semantic Embedding. The augmented user and item information will be encoded as features and used as input for the recommender. Using LLM as an encoder offers efficient and state-of-the-art language understanding, enabling profiling user interaction preferences and debiasing item attributes.
![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-04.jpg?height=282&width=1750&top_left_y=284&top_left_x=184)

Figure 1: The LLMRec framework: (1) Three types of data augmentation strategies: i) augmenting user-item interactions; ii) enhancing item attributes, and iii) user profiling. (2) Augmented training with and denoised data robustification mechanism.

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-04.jpg?height=607&width=312&top_left_y=694&top_left_x=191)

Generate user profile based on the history of user, that each movie with title, year, genre. History:

[332] Heart and Souls (1993), Comedy|Fantasy [364] Men with Brooms (2002), Comedy|Drama|Romance

Please output the following infomation of user, output format: \{age:, gender: , liked genre: , disliked genre: liked directors:, country:, language: \}

\{age: $\mathbf{5 0 ,}$ gender: female, liked genre: Comedy|Fantasy, Comedy|Drama|Romance, disliked genre: Thriller, Horror, liked directors: Ron Underwood, country: Canada, United States, Ronguage: English\}

(b) User Profile

Provide the inquired information of the given movie [332] Heart and Souls (1993), Comedy|Fantasy The inquired information is: director, country language. And please output them in form of: director, country, language

Ron Underwood, USA, English

(a) Implicit Feedback

(c) Item Attribute

Figure 2: Constructed prompt $\mathbb{P}_{u}^{U I}, \mathbb{P}_{u}^{U}, \mathbb{P}_{i}^{I}$ for LLMs' completion including i) task description, ii) historical interactions, iii) candidates, and iv) output format description.

Formally, the LLM-based side information augmentation is as:

$$
\begin{cases}\text { user }: & \mathbb{A}_{u}=L L M\left(\mathbb{P}_{u}^{U}\right) \quad \longrightarrow \quad \mathbf{f}_{\mathcal{A}, u}=\operatorname{LLM}\left(\mathbb{A}_{u}\right)  \tag{5}\\ \text { item }: & \mathbb{A}_{i}=L L M\left(\mathbb{P}_{i}^{I}\right) \quad \longrightarrow \quad \mathbf{f}_{\mathcal{A}, i}=\operatorname{LLM}\left(\mathbb{A}_{i}\right)\end{cases}
$$

where $\mathbf{f}_{\mathcal{A}, u}, \mathbf{f}_{\mathcal{A}, i,} \in \mathbb{R}^{d_{L L M}}$ are LLM-augmented user/item features with LLM's hidden dimension $d_{L L M}$. The textual prompts $\mathbb{P}_{u}^{U}$ and $\mathbb{P}_{i}^{I}$ are used for attribute refinement for user $u$ and item $i$, respectively. $\mathbb{A}_{u}$ and $\mathbb{A}_{i}$ represent generated textual attributes that to be encoded as features $\mathbf{F}_{\mathcal{A}, u}, \mathbf{F}_{\mathcal{A}, i}$ using the embedding capability of $L L M(\cdot)$.

3.2.2 Side Information Incorporation (Q3). After obtaining the augmented side information for user/item, an effective incorporation method is necessary. LLMRec includes a standard procedure: (1) Augmented Semantic Projection, (2) Collaborative Context Injection, and (3) Feature Incorporation. Let's delve into each:

- Augmented Semantic Projection. Linear layers with dropout are employed to not only reduce the dimensionality of LLMenhanced semantic features but also map such augmented features into their own space [46]. This process can be represented as $\overline{\mathbf{F}}_{\mathcal{F}}=\operatorname{Linear}\left(\mathbf{F}_{\mathcal{A}}\right)$, where $\mathbf{f}_{\mathcal{A}} \in \mathbb{R}^{1 \times d_{L L M}}$ is the input feature and $\overline{\mathbf{f}}_{\mathcal{A}} \in \mathbb{R}^{1 \times d}$ is the output feature after projection.
- Collaborative Context Injection. To inject high-order [41] collaborative connectivity into augmented features $\overline{\mathbf{f}}_{\mathcal{A}, u}$ and $\overline{\mathbf{f}}_{\mathcal{A}, i}$, LLMRec employs light weight GNNs [11] as the encoder.
- Semantic Feature Incorporation. Instead of taking augmented features $\overline{\mathbf{F}}_{\mathcal{A}}$ as initialization of learnable vectors of recommender

$f_{\Theta}$, we opt to treat $\overline{\mathbf{F}}_{\mathcal{A}}$ as additional compositions added to the ID-corresponding embeddings ( $\mathbf{e}_{u}, \mathbf{e}_{i}$ ). This allows flexibly adjust the influence of LLM-augmented features using scale factors and normalization. Formally, the $\overline{\mathbf{F}}_{\mathcal{A}}$ 's incorporation is presented as:

$$
\mathbf{h}_{u}=\mathbf{e}_{u}+\omega_{1} \cdot \sum_{k \in \mathcal{M} \cup \mathbb{A}_{u}}^{|\mathcal{M}|+\left|\mathbb{A}_{u}\right|} \frac{\overline{\mathbf{f}}_{u}^{k}}{\left\|\overline{\mathbf{f}}_{u}^{k}\right\|_{2}} ; \quad \mathbf{h}_{i}=\mathbf{e}_{i}+\omega_{1} \cdot \sum_{k \in \mathcal{M} \cup \mathbb{A}_{i}}^{|\mathcal{M}|+\left|\mathbb{A}_{i}\right|} \frac{\overline{\mathbf{f}}_{i}^{k}}{\|_{2}}
$$

The final prediction representations $\mathbf{h}_{u}$ and $\mathbf{h}_{i}$, are in $\mathbb{R}^{1 \times d}$. User profiles are $\mathbb{A}_{u}$, debiased item attributes are $\mathbb{A}_{i}$, and original multimodal side information is $\mathcal{M}$. The specific type of feature is $\mathbf{f}^{k}$. We adjust feature vectors using the aggregation weight $\omega_{1}$ and $L_{2}$ normalization to mitigate distribution gaps [8], ensurring the effectiveness of additional features within the recommender encoder.

### 3.3 Training with Denoised Robustification (Q4)

In this section, we outline how LLMRec integrate augmented data into the optimization. We also introduce two quality constraint mechanisms for augmented edges and node features: i) Noisy useritem interaction pruning, and ii) MAE-based feature enhancement.

3.3.1 Augmented Optimization with Noise Pruning. We train our recommender using the union set $\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}$, which includes the original training set $\mathcal{E}$ and the LLM-augmented set $\mathcal{E}_{\mathcal{A}}$. The objective is to optimize the BPR $\mathcal{L}_{\mathrm{BPR}}$ loss with increased supervisory signals $\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}$, aiming to enhance the recommender's performance by leveraging the incorporated LLM-enhanced user preference:

$$
\begin{gather*}
\mathcal{L}_{\mathrm{BPR}}=\sum_{\left(u, i^{+}, i^{-}\right)}^{\left|\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}\right|}-\log \left(\sigma\left(\hat{y}_{u, i^{+}}-\hat{y}_{u, i^{-}}\right)\right)+\omega_{2} \cdot\|\Theta\|^{2}  \tag{6}\\
\mathcal{E}_{\mathcal{A}} \subseteq\left\{L L M\left(\mathbb{P}_{u}\right) \mid u \in \mathcal{U}\right\}, \quad\left|\mathcal{E}_{\mathcal{A}}\right|=\omega_{3} * B
\end{gather*}
$$

The training triplet $\left(u, i^{+}, i^{-}\right)$is selected from the union training set $\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}$. The predicted scores of positive-negative sample pairs are obtained through inner products of final representation $\mathbf{h}$, i.e., $\hat{y}_{u, i^{+}}=\mathbf{h}_{u} \cdot \mathbf{h}_{i+}, \hat{y}_{u, i^{-}}=\mathbf{h}_{u} \cdot \mathbf{h}_{i-}$. The augmented dataset $\mathcal{E}_{\mathcal{A}}$ is a subset of the overall LLM-generated data $\left\{L L M\left(\mathbb{P}_{u}\right) \mid u \in \mathcal{U}\right\}$, obtained by sampling. This is because excessive inclusion of pseudo label may lead to a degradation in result accuracy. The number of samples $\left|\mathcal{E}_{\mathcal{A}}\right|$ is controlled by the batch size $B$ and a rate $\omega_{3}$. Weightdecay regularization $|\Theta|^{2}$ weighted by $\omega_{2}$, mitigates overfitting. $\sigma(\cdot)$ is activation function sigmoid to introduce non-linearity.

Noise Pruning. To enhance the effectiveness of augmented data, we prune out unreliable u-i interaction noise. Technically, the largest values before minus are discarded after sorting each iteration. This helps prioritize and emphasize relevant supervisory

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-05.jpg?height=250&width=415&top_left_y=287&top_left_x=172)

(a) Four Types of Implicit Feedback

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-05.jpg?height=258&width=211&top_left_y=283&top_left_x=572)

$\hat{y}_{u^{+-}}$

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-05.jpg?height=282&width=247&top_left_y=287&top_left_x=771)

(b)
Figure 3: (a) Implicit feedback encompasses both false positive and false negative samples. (b) The gradient $\nabla$ of the BPR loss for positive $\hat{y}_{u, i^{+}}$and negative $\hat{y}_{u, i^{-}}$scores, despite having a large magnitude, can have an incorrect direction that notably impacts the robustness and effectiveness of training.

signals while mitigating the influence of noise. Formally, the objective $\mathcal{L}_{\mathrm{BPR}}$ in Eq. 6 with noise pruning can be rewritten as follows:

$$
\begin{equation*}
\sum_{\left(u, i^{+}, i^{-}\right)}^{\left(1-\omega_{4}\right) *\left|\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}\right|}-\operatorname{SortAscend}\left(\log \left(\sigma\left(\hat{y}_{u, i^{+}}-\hat{y}_{u, i^{-}}\right)\right)\right)[0: N]+\omega_{2} \cdot\|\Theta\|^{2} \tag{7}
\end{equation*}
$$

The function SortAscend $(\cdot)[0: N]$ sorts values and selects the top$\mathrm{N}$. The retained number $N$ is calculated by $N=\left(1-\omega_{4}\right) \cdot\left|\mathcal{E} \cup \mathcal{E}_{\mathcal{A}}\right|$, where $\omega_{4}$ is a rate. This approach allows for controlled pruning of loss samples, emphasizing relevant signals while reducing noise This can avoid the impact of unreliable gradient backpropagation, thus making optimization more stable and effective.

3.3.2 Enhancing Augmented Semantic Features via MAE. To mitigate the impact of noisy augmented features, we employ the Masked Autoencoders (MAE) for feature enhancement [9]. Specifically, the masking technique is to reduce the model's sensitivity to features, and subsequently, the feature encoders are strengthened through reconstruction objectives. Formally, we select a subset of nodes $\widetilde{\mathcal{V}} \subset \mathcal{V}$ and mask their features using a mask token [MASK], denoted as $\mathbf{f}_{[M A S K]}$ (e.g., a learnable vector or mean pooling). The mask operation can be formulated as follows:

$$
\overline{\mathbf{f}}_{\mathcal{A}}= \begin{cases}\mathbf{f}_{[M A S K]} & v \in \widetilde{\mathcal{V}}  \tag{8}\\ \overline{\mathbf{f}}_{\mathcal{A}} & v \notin \widetilde{\mathcal{V}}\end{cases}
$$

The augmented feature after the mask operation is denoted as $\widetilde{\bar{f}}_{\mathcal{A}}$ It is substituted as mask token $\mathbf{f}_{[M A S K]}$ if the node is selected $(\widetilde{\mathcal{V}} \subset \mathcal{V})$, otherwise, it corresponds to the original augmented feature $\overline{\mathbf{f}}_{\mathcal{A}}$. To strengthen the feature encoder, we introduce the feature restoration loss $\mathcal{L}_{F R}$ by comparing the masked attribute

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-05.jpg?height=54&width=848&top_left_y=1946&top_left_x=172)
scaling factor $\gamma$. The restoration loss function $\mathcal{L}_{F R}$ is as follows:

$$
\begin{equation*}
\mathcal{L}_{F R}=\frac{1}{|\widetilde{\mathcal{V}}|} \sum_{v \in \widetilde{\mathcal{V}}}\left(1-\frac{\widetilde{\overline{\mathbf{f}}}_{\mathcal{A}} \cdot \overline{\mathbf{f}}_{\mathcal{A}}}{\left\|\widetilde{\overline{\mathbf{f}}}_{\mathcal{A}}\right\| \cdot\left\|\overline{\mathbf{f}}_{\mathcal{A}}\right\|}\right)^{\gamma} \tag{9}
\end{equation*}
$$

The final optimization objective is the weighted sum of the noisepruned BPR loss $\mathcal{L}_{\mathrm{BPR}}$ and the feature restoration (FR) loss $\mathcal{L}_{F R}$.

### 3.4 In-Depth Analysis of our LLMRec

3.4.1 LLM-based Augmentation Facilitates Optimization. This section highlights challenges addressed by LLM-based augmentation in recommender systems. False negatives (non-interacted

| Dataset |  | Netflix |  |  | MovieLens |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Graph |  | \# U | \# I | \# E | \# U | \# I | $\# \mathrm{E} \quad$ |
|  | Ont. | 13187 | 17366 | 68933 | 12495 | 10322 | 57960 |
|  | Aug. | $\frac{26374}{99.970 \%}$ |  |  | \# E: | 24990 |  |
| Ori. Sparsity |  |  |  |  | $99.915 \%$ |  |  |
| Att. | Ori. | U: None | I: year, title |  | U: None | I: title, year, genre |  |
|  | Aug. | {f828e30f1-5fc1-4344-99bb-b600aa34db44}age, gender, liked genre, disliked genre, <br> liked directors, country, and language |  |  |  |  |  |
|  |  | I[1536]: director, country, language |  |  |  |  |  |
| Modality |  | Textual[768], Visiual [512] |  |  | Textual [768], Visiual [512] |  |  |

interactions) and false positives (noise) as in Fig. 3 (a) can affect data
Table 1: Statistics of the Original and Augmented Datasets

* Att. represents attribute, Ori. represents original, and Aug. represents augmentation. Number in [] represents the feature dimensionality.

quality and result accuracy $[3,40]$. Non-interacted items do not necessarily imply dislike [3], and interacted one may fail to reflect real user preferences due to accidental clicks or misleading titles, etc. Mixing of unreliable data with true user preference poses a challenge in build accurate recommender. Identifying and utilizing reliable examples is key to optimizing the recommender [2].

In theory, non-interacted and noisy interactions are used for negative $\hat{y}_{u, i^{-}}$and positive $\hat{y}_{u, i^{+}}$scores, respectively. However, their optimization directions oppose the true direction with large magnitudes, i.e., the model optimizes significantly in the wrong directions (as in Fig. 3 (b)), resulting in sensitive suboptimal results.

Details. By computing the derivatives of the $\mathcal{L}_{B P R}$ (Eq. 10), we obtain positive gradients $\nabla_{u, i^{+}}=1-\sigma\left(\hat{y}_{u^{+-}}\right)$and negative gradients $\nabla_{u, i^{-}}=\sigma\left(\hat{y}_{u^{+-}}\right)-1$, where $\hat{y}_{u^{+-}}=\hat{y}_{u, i^{+}}-\hat{y}_{u, i^{-}}$. Fig. 3 (b) illustrates these gradients and unveils some observations. Noisy interactions, although treated as positives, often have small values $\hat{y}_{u, i^{+}}$as false positives, resulting in large gradients $\nabla_{u, i^{+}}$. Conversely, unobserved items, treated as negatives, tend to have relatively large values $\hat{y}_{u, i^{-}}$ as false negatives, leading to small $\hat{y}_{u^{+-}}$and large gradients $\nabla_{u, i^{-}}$.

$$
\left\{\begin{align*}
\nabla_{u, i^{+}} & =\frac{\partial \mathcal{L}_{B P R}}{\partial \hat{y}_{u, i^{+}}}=-\frac{\partial \log \sigma\left(\hat{y}_{u^{+-}}\right)}{\partial \sigma\left(\hat{y}_{u^{+-}}\right)} \frac{\partial \sigma\left(\hat{y}_{u^{+-}}\right)}{\partial \hat{y}_{u, i^{+}}} \\
& =-\frac{1}{\sigma\left(\hat{y}_{u^{+-}}\right)} \cdot \sigma\left(\hat{y}_{u^{+-}}\right) \cdot\left(1-\sigma\left(\hat{y}_{u^{+-}}\right)\right) \cdot 1=\sigma(\overbrace{\hat{y}_{u, i^{+}}}-\hat{y}_{u, i^{-}})-1 \\
\nabla_{u, i^{-}} & =\frac{\partial \mathcal{L}_{B P R}}{\partial \hat{y}_{u, i^{-}}}=-\frac{\partial \log \sigma\left(\hat{y}_{u^{+-}}\right)}{\partial \sigma\left(\hat{y}_{u^{+-}}\right)} \frac{\partial \sigma\left(\hat{y}_{u^{+-}}\right)}{\partial \hat{y}_{u, i^{-}}} \\
& =\frac{1}{\sigma\left(\hat{y}_{u^{+-}}\right)} \cdot \sigma\left(\hat{y}_{u^{+-}}\right) \cdot\left(1-\sigma\left(\hat{y}_{u^{+-}}\right)\right) \cdot 1=1-\sigma(\hat{y}_{u, i^{+}}-\overbrace{\hat{y}_{u, i^{-}}}^{\text {unobserved }}) \tag{10}
\end{align*}\right.
$$

Conclusion. Wrong samples possess incorrect directions but are influential. LLM-based augmentation uses the natural language space to assist the ID vector space to provide a comprehensive reflection of user preferences. With real-world knowledge, LLMRec gets quality samples, reducing the impact of noisy and unobserved implicit feedback, improving accuracy, and speeding up convergence.

3.4.2 Time Complexity. We analyze the time complexity. The projection of augmented semantic features has a time complexity of $O\left(|\mathcal{U} \cup \mathcal{I}| \times d_{L L M} \times d\right)$. The GNN encoder for graph-based collaborative context learning takes $O\left(L \times\left|\mathcal{E}^{+}\right| \times d\right)$ time. The BPR loss function computation has a time complexity of $O(d \times \mid \mathcal{E} \cup$ $\mathcal{E}_{\mathcal{A}} \mid$ ), while the feature reconstruction loss has a time complexity of $O(d \times|\widetilde{\mathcal{V}}|)$, where $|\widetilde{\mathcal{V}}|$ represents the count of masked nodes.

## 4 EVALUATION

To evaluate the performance of LLMRec, we conduct experiments, aiming to address the following research questions:

- RQ1: How does our LLM-enhanced recommender perform compared to the current state-of-the-art baselines?
- RQ2: What is the impact of key components on the performance?
- RQ3: How sensitive is the model to different parameters?
- RQ3: Are the data augmentation strategies in our LLMRec applicable across different recommendation models?
- RQ5: What is the computational cost associated with our devised LLM-based data augmentation schemes?


### 4.1 Experimental Settings

4.1.1 Datasets. We perform experiments on publicly available datasets, i.e., Netflix and MovieLens, which include multi-modal side information. Tab. 1 presents statistical details for both the original and augmented datasets for both user and item domains. MovieLens. We utilize the MovieLens dataset derived from ML-10M ${ }^{1}$. Side information includes movie title, year, and genre in textual format. Visual content consists of movie posters obtained through web crawling by ourselves. Netflix. We collected its multi-model side information through web crawling. The implicit feedback and basic attribute are sourced from the Netflix Prize Data ${ }^{2}$ on Kaggle. For both datasets, CLIP-ViT[31] is utilized to encode visual features. LLM-based Data Augmentation. The study employs the OpenAI package, accessed through LLMs' APIs, for augmentation. The OpenAI Platform documentation provides details ${ }^{3}$. Augmented implicit feedback is generated using the "gpt-3.5-turbo-0613" chat completion model. Item attributes such as directors, country, and language are gathered using the same model. User profiling, based on the "gpt-3.5-turbo-16k" model, includes age, gender, preferred genre, disliked genre, preferred directors, country, and language. Embedding is performed using the "text-embedding-ada-002" model. The approximate cost of augmentation strategies on two datasets is 15.65 USD, 20.40 USD, and 3.12 USD, respectively.

4.1.2 Implementation Details. The experiments are conducted on a 24 GB Nvidia RTX 3090 GPU using PyTorch[29] for code implementation. The AdamW optimizer[25] is used for training, with different learning rate ranges of $\left[5 e^{-5}, 1 e^{-3}\right]$ and $\left[2.5 e^{-4}, 9.5 e^{-4}\right]$ for Netflix and MovieLens, respectively. Regarding the parameters of the LLMs, we choose the temperature from larger values $\{0.0$, $0.6,0.8,1\}$ to control the randomness of the generated text. The value of top-p is selected from smaller values $\{0.0,0.1,0.4,1\}$ to encourage probable choices. The stream is set to false to ensure the completeness of responses. For more details on the parameter analysis, please refer to Section 4.4. To maintain fairness, both our method and the baselines employ a unified embedding size of 64 .

4.1.3 Evaluation Protocols. We evaluate our approach in the top-K item recommendation task using three common metrics: Recall (R@k), Normalized Discounted Cumulative Gain ( $\mathrm{N} @ \mathrm{k}$ ), and Precision (P@k). To avoid potential biases from test sampling, we employ the all-ranking strategy[47, 49]. We report averaged results from five independent runs, setting $\mathrm{K}$ to 10,20 , and 50 (reasonable[^1]

for all-ranking). Statistical significance analysis is conducted by calculating $p$-values against the best-performing baseline.

4.1.4 Baseline Description. Four distinct groups of baseline methods for thorough comparison. i) General CF Methods: MFBPR [34], NGCF [41] and LightGCN [11]. ii) Methods with Side Information: VBPR [10], MMGCN [50] and GRCN [49]. iii) Data Augmentation Methods: LATTICE [59]. iv) Self-supervised Methods: CLCRec [48], MMSSL [45] and MICRO [58].

### 4.2 Performance Comparison (RQ1)

Tab. 2 compares our proposed LLMRec method with baselines.

- Overall Model Superior Performance. Our LLMRec outperforms the baselines by explicitly augmenting u-i interactive edges and enhancing the quality of side information. It is worth mentioning that our model based on LATTICE's [59] encoder, consisting of a ID-corresponding encoder and a feature encoder. This improvement underscores the effectiveness of our framework.
- Effectiveness of Side Information Incorporation. The integration of side information significantly empowers recommenders. Methods like MMSSL [45] and MICRO [58] stand out for their effective utilization of multiple modalities of side information and GNNs. In contrast, approaches rely on limited content, such as VBPR [10] using only visual features, or CFbased architectures like NGCF [41], without side information, yield significantly diminished results. This highlights the importance of valuable content, as relying solely on ID-corresponding records fails to capture the complete u-i relationships.
- Inaccurate Augmentation yields Limited Benefits. Existing methods, such as LATTICE[59], MICRO[58] that also utilize side information for data augmentation have shown limited improvements compared to our LLMRec. This can be attributed to two main factors: (1) The augmentation of side information with homogeneous relationships (e.g., i-i or $\mathrm{u}-\mathrm{u}$ ) may introduce noise, which can compromise the precise of user preferences. (2) These methods often not direct augmentation of $u$-i interaction data.
- Advantage over SSL Approaches. Self-supervised models like, MMSSL[45], MICRO[58], have shown promising results in addressing sparsity through SSL signals. However, they do not surpass the performance of LLMRec, possibly because their augmented self-supervision signals may not align well with the target task of modeling u-i interactions. In contrast, we explicitly tackle the scarcity of training data by directly establishing BPR triplets.


### 4.3 Ablation and Effectiveness Analyses (RQ2)

We conduct an ablation study of our proposed LLMRec approach to validate its key components, and present the results in Table 3.

### 4.3.1 Effectiveness of Data Augmentation Strategies.

- (1). w/o-u-i: Disabling the LLM-augmented implicit feedback $\mathcal{E}_{\mathcal{A}}$ results in a significant decrease. This indicates that LLMRec increases the potential supervision signals by including contextual knowledge, leading to a better grasp of user preferences.
- (2). w/o-u: Removing our augmentor for user profiling result in a decrease in performance, indicating that our LLM-enhanced user side information can effectively summarize useful user preference profile using historical interactions and item-end knowledge.

Table 2: Performance comparison on different datasets in terms of Recall@10/20/50, and NDCG@10/20/50, and Precision@20.

| Baseline | Netflix |  |  |  |  |  | MovieLens |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | R@10 | N@10 | R@20 | $\mathrm{N} @ 20$ | R@50 | $\mathrm{N} @ 50$ | P@20 | R@10 | $\mathrm{N} @ 10$ | $\mathrm{R} @ 20$ | $\mathrm{~N} @ 20$ | R@50 | $\mathrm{N} @ 50$ | P@20 |
| General Collaborative Filtering Methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MF-BPR | 0.0282 | 0.0140 | 0.0542 | 0.0205 | 0.0932 | 0.0281 | 0.0027 | 0.1890 | 0.0815 | 0.2564 | 0.0985 | 0.3442 | 0.1161 | 0.0128 |
| $\mathrm{NGCF}$ | 0.0347 | 0.0161 | 0.0699 | 0.0235 | 0.1092 | 0.0336 | 0.0032 | 0.2084 | 0.0886 | 0.2926 | 0.1100 | 0.4262 | 0.1362 | 0.0146 |
| LightGCN | 0.0352 | 0.0160 | 0.0701 | 0.0238 | 0.1125 | 0.0339 | 0.0032 | 0.1994 | 0.0837 | 0.2660 | 0.1005 | 0.3692 | 0.1209 | 0.0133 |
| Recommenders with Side Information |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| $\overline{\text { VBPR }}$ | 0.0325 | 0.0142 | 0.0553 | 0.0199 | 0.1024 | 0.0291 | 0.0028 | 0.2144 | 0.0929 | 0.2980 | 0.1142 | 0.4076 | 0.1361 | 0.0149 |
| MMGCN | 0.0363 | 0.0174 | 0.0699 | 0.0249 | 0.1164 | 0.0342 | 0.0033 | 0.2314 | 0.1097 | 0.2856 | 0.1233 | 0.4282 | 0.1514 | 0.0147 |
| GRCN | 0.0379 | 0.0192 | 0.0706 | 0.0257 | 0.1148 | 0.0358 | 0.0035 | 0.2384 | 0.1040 | 0.3130 | 0.1236 | 0.4532 | 0.1516 | 0.0150 |
| Data Augmentation Methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LATTICE | 0.0433 | 0.0181 | 0.0737 | 0.0259 | 0.1301 | 0.0370 | 0.0036 | 0.2116 | 0.0955 | 0.3454 | 0.1268 | 0.4667 | 0.1479 | 0.0167 |
| MICRO | $\underline{0.0466}$ | 0.0196 | $\underline{0.0764}$ | 0.0271 | $\underline{0.1306}$ | 0.0378 | $\underline{0.0038}$ | 0.2150 | $\underline{0.1131}$ | $\underline{0.3461}$ | $\underline{0.1468}$ | $\underline{0.4898}$ | $\underline{0.1743}$ | $\underline{0.0175}$ |
| Self-supervised Methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CLCRec | 0.0428 | 0.0217 | 0.0607 | 0.0262 | 0.0981 | 0.0335 | 0.0030 | 0.2266 | 0.0971 | 0.3164 | 0.1198 | 0.4488 | 0.1459 | 0.0158 |
| MMSSL | 0.0455 | $\underline{0.0224}$ | 0.0743 | 0.0287 | 0.1257 | $\underline{0.0383}$ | 0.0037 | $\underline{0.2482}$ | 0.1113 | 0.3354 | 0.1310 | 0.4814 | 0.1616 | 0.0170 |
| LLMRec | 0.0531 | 0.0272 | 0.0829 | 0.0347 | 0.1382 | 0.0456 | 0.0041 | 0.2603 | 0.1250 | 0.3643 | 0.1628 | 0.5281 | 0.1901 | 0.0186 |
| $p$-value | $2.9 e^{-4}$ | $3.0 e^{-3}$ | $9.4 e^{-5}$ | $1.5 e^{-3}$ | $2.8 e^{-5}$ | $2.2 e^{-3}$ | $3.4 e^{-5}$ | $2.8 e^{-5}$ | $1.6 e^{-2}$ | $3.1 e^{-3}$ | $4.1 e^{-4}$ | $1.9 e^{-3}$ | $1.3 e^{-2}$ | $1.8 e^{-3}$ |
| Improv. | $13.95 \%$ | $21.43 \%$ | $8.51 \%$ | $20.91 \%$ | $5.82 \%$ | $19.06 \%$ | $7.89 \%$ | $4.88 \%$ | $10.52 \%$ | $5.26 \%$ | $10.90 \%$ | $7.82 \%$ | $9.06 \%$ | $6.29 \%$ |

Table 3: Ablation study on key components (i.e., data augmentation strategies, denoised data robustification mechanisms)

|  | Metrics | R@10 | $\mathrm{N} @ 10$ | R@20 | $\mathrm{N} @ 20$ | R@50 | $\mathrm{N} @ 50$ | P@20 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-07.jpg?height=113&width=38&top_left_y=1225&top_left_x=235) | w/o-u-i | 0.0477 | 0.0239 | 0.0791 | 0.0317 | $\mid 0.1376$ | 0.0432 | 0.0037 |
|  | w/o-u | 0.0423 | 0.0196 | 0.0656 | 0.0255 | 0.1192 | 0.0360 | 0.0033 |
|  | w/o-u\&i | 0.0309 | 0.0127 | 0.0602 | 0.0202 | 0.1051 | 0.0289 | 0.0030 |
| $\dot{u}$ <br> $\dot{\alpha}$ | w/o-prun | 0.0504 | 0.0258 | 0.0786 | 0.0328 | $\mid 0.1363$ | 0.0447 | 0.0039 |
|  | w/o-QC | 0.0488 | 0.0244 | 0.0786 | 0.0318 | 0.1279 | 0.0416 | 0.0038 |
|  | LLMRec | 0.05 | 10. | 0.082 | $\mathbf{3 4 7}$ |  |  | .0041 |

- (3). w/o-u\&i: when we remove the augmented side information for both users and items $\left(\mathbf{F}_{\mathcal{A}, u}, \mathbf{F}_{\mathcal{A}, i, 1}\right)$, lower recommendation accuracy is observed. This finding indicates that the LLM-based augmented side information provides valuable augmented data to the recommender system, assisting in obtaining quality and informative representations.


### 4.3.2 Impact of the Denoised Data Robustification.

- w/o-prune: The removal of noise pruning results in worse performance. This suggests that the process of removing noisy implicit feedback signals helps prevent incorrect gradient descent.
- w/o-QC: The performance suffer when both the limits on implicit feedback and semantic feature quality are simultaneously removed (i.e., w/o-prune + w/o-MAE). This indicates the benefits of our denoised data robustification mechanism by integrating noise pruning and semantic feature enhancement.


### 4.4 Hyperparameter Analysis (RQ3)

4.4.1 Parameters Affecting Augmented Data Quality.

- Temperature $\tau$ of $L L M$ : The temperature parameter $\tau$ affects text randomness. Higher values (>1.0) increase diversity and creativity, while lower values ( $<0.1$ ) result in more focus. We use $\tau$ from $\{0,0.6,0.8,1\}$. As shown in Table 4, increasing $\tau$ initially improves most metrics, followed by a decrease.
Table 4: Parameter analysis of temperature $\tau$ and top-p $\rho$.

| Para. | Temperature $\tau$ |  |  | Top-p $\rho$ |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Metrics | $\tau=0$ | $\tau=0.6$ | $\tau=0.8$ | $\tau=1$ | $\rho=0$ | $\rho=0.1$ | $\rho=0.4$ | $\rho=1$ |
| R@10 | $0.0558 \uparrow$ | $\mathbf{0 . 0 5 3 1}$ | $0.0553 \uparrow$ | $0.0531=$ | $0.0537 \uparrow$ | $\mathbf{0 . 0 5 3 1}$ | $0.0520 \downarrow$ | $0.0531=$ |
| R@20 | $0.0808 \downarrow$ | $\mathbf{0 . 0 8 2 9}$ | $0.0813 \downarrow$ | $0.0775 \downarrow$ | $0.0802 \downarrow$ | $\mathbf{0 . 0 8 2 9}$ | $0.0796 \downarrow$ | $0.0770 \downarrow$ |
| R@50 | $0.1344 \downarrow$ | $\mathbf{0 . 1 3 8 2}$ | $0.1360 \downarrow$ | $0.1312 \downarrow$ | $0.1360 \downarrow$ | $\mathbf{0 . 1 3 8 2}$ | $0.1344 \downarrow \downarrow$ | $0.1333 \downarrow$ |

Table 5: Analysis of key parameter (i.e., \# candidate $|C|$ ) for LLM w.r.t implicit feedback augmentation $\mathcal{E}_{\mathcal{A}}$.

| Data | Netflix |  |  | MovieLens |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Metrics | $\|C\|=3$ | $\|C\|=10$ | $\|C\|=30$ |  | $\|C\|=3$ | $\|C\|=10$ | $\|C\|=30$ |
| R@20 | $0.0786 \downarrow$ | $\mathbf{0 . 0 8 2 9}$ | $0.0808 \downarrow$ | $0.3567 \downarrow$ | $\mathbf{0 . 3 6 4 3}$ | $0.3695 \uparrow$ |  |
| N@20 | $0.0314 \downarrow$ | $\mathbf{0 . 0 3 4 7}$ | $0.0330 \downarrow$ | $0.1603 \downarrow$ | $\mathbf{0 . 1 6 2 8}$ | $0.1614 \downarrow$ |  |
| P@20 | $0.0039 \downarrow$ | $\mathbf{0 . 0 0 4 1}$ | $0.0040 \downarrow$ |  | $0.0179 \downarrow$ | $\mathbf{0 . 0 1 8 6}$ | $0.0182 \downarrow$ |

- Top-p p of LLM: Top-p Sampling[12] selects tokens based on a threshold determined by the top-p parameter $p$. Lower $p$ values prioritize likely tokens, while higher values encourage diversity. We use $p$ from $\{0,0.1,0.4,1\}$ and smaller $p$ values tend to yield better results, likely due to avoiding unlisted candidate selection. Higher $\rho$ values cause wasted tokens due to repeated LLM inference.
- \# of Candidate $C$ : We use $C$ to limit item candidates for LLM-based recommendation. $\{3,10,30\}$ are explored due to cost limitations, and Table 5 shows that $C=10$ yields the best results. Small values limit selection, and large values increase recommendation difficulty.
- Prune Rate $\omega_{4}$ : LLMRec uses $\omega_{4}$ to control noise in augmented training data to be pruned. We set $\omega_{4}$ to $\{0.0,0.2,0.4,0.6,0.8\}$ on both datasets. As shown in Fig. 4 (a), $\omega_{4}=0$ yields the worst result, highlighting the need to constrain noise in implicit feedback.


### 4.4.2 Sensitivity of Recommenders to the Augmented Data.

- \# of Augmented Samples per Batch $\left|\mathcal{E}_{\mathcal{A}}\right|$ : LLMRec uses $\omega_{3}$ and batch size $B$ to control the number of augmented BPR training data samples per batch. $\omega_{3}$ is set to $\{0.0,0.1,0.2,0.3,0.4\}$ on Netflix and $\{0.0,0.2,0.4,0.6,0.8\}$ on MovieLens. Suboptimal results occur

![](https://cdn.mathpix.com/cropped/2024_06_04_ac7d8dcdd0a2c1d4f099g-08.jpg?height=436&width=864&top_left_y=281&top_left_x=175)

Figure 4: Impact of hyperparameters (i.e., prune rate $\omega_{4}$, \# augmented BPR training data $\left|\mathcal{E}_{\mathcal{A}}\right|$, and augmented feature incorporate scale $\omega_{1}$ ).

Table 6: Model-agnostic experiment to evaluate the effectiveness of LLM-based data augmentation on different recommender in terms of R@20, N@20, and P@20.

|  | Method | LATTICE | MICRO | MMSSL |
| :---: | :---: | :---: | :---: | :---: |
|  | R@20 | $0.0821 \uparrow 11.40 \%$ | $0.0835 \uparrow 9.29 \%$ | $0.0833 \uparrow 11.11 \%$ |
|  | N@20 | $0.0287 \uparrow 10.81 \%$ | $0.0301 \uparrow 11.07 \%$ | $0.0313 \uparrow 9.06 \%$ |
|  | P@20 | $0.0039 \uparrow 8.33 \%$ | $0.0041 \uparrow 7.89 \%$ | $0.0041 \uparrow 10.81 \%$ |

when $\omega_{3}$ is zero or excessively large. Increasing diversity and randomness can lead to a more robust gradient descent.

- Scale $\omega_{2}$ for Incorporating Augmented Features: LLMRec uses $\omega_{2}$ to control feature magnitude, with values set to $\{0.0,0.8,1.6,2.4$, $3.2\}$ on Netflix and $\{0.0,0.1,0.2,0.3,0.4\}$ on MovieLens. Optimal results depend on the data, with suboptimal outcomes occurring when $\omega_{2}$ is too small or too large, as shown in Fig. 4 (c).


### 4.5 Model-agnostic Property (RQ4)

We conducted model-agnostic experiments on Netflix to validate the applicability of our data augmentation. Specifically, we incorporated the augmented implicit feedback $\mathcal{E}_{\mathcal{A}}$ and features $\mathbf{F}_{\mathcal{A}, u}, \mathbf{F}_{\mathcal{A}, i}$ into baselines MICRO, MMSSL, and LATTICE. As shown in Tab. 6, our LLM-based data improved the performance of all models, demonstrating their effectiveness and reusability. Some results didn't surpass our model, maybe due to: i) the lack of a quality constraint mechanism to regulate the stability and quality of the augmented data, and ii) the absence of modeling collaborative signals in the same vector space, as mentioned in Sec. 3.2.2.

### 4.6 Cost/Improvement Conversion Rate (RQ5)

To evaluate the cost-effectiveness of our augmentation strategies, we compute the CIR as presented in Tab. 7. The CIR is compared with the ablation of three data augmentation strategies and the best baseline from Tab. 3 and Tab. 2. The cost of the implicit feedback augmentor refers to the price of GPT-3.5 turbo $4 \mathrm{~K}$. The cost of side information augmentation includes completion (using GPT-3.5 turbo $4 \mathrm{~K}$ or $16 \mathrm{~K}$ ) and embedding (using text-embedding-ada-002). We utilize the HuggingFace API tool for tokenizer and counting. The results in Tab. 7 show that 'U' (LLM-based user profiling) is the most cost-effective strategy, and the overall investment is worthwhile.
Table 7: Comparison of the cost and improvement rate(CIR) of data augmentation strategies and LLMRec. 'Cost': expenditure of utilizing LLM, 'Imp.': the average improvement rate in $\mathrm{R} @ 10 / \mathrm{N} @ 10$. 'CIR': the ratio of improvement to cost.

|  |  | R@10 |  |  | N@10 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Cost(USD) | Imp.(\%) | CIR(\%) |  | Imp.(\%) | CIR(\%) |
| $\mathrm{U}$ | 10.92 | 25.53 | 233.79 |  | 38.78 | 355.13 |
| $\mathrm{I}$ | 1.96 | 2.31 | 117.86 |  | 1.12 | 57.14 |
| U-I | 8.26 | 11.32 | 137.05 |  | 13.81 | 167.19 |
| LLMAug | 21.14 | 13.95 | 65.99 |  | 21.43 | 101.37 |

## 5 RELATED WORK

Content-based Recommendation. Existing recommenders have explored the use of auxiliary multi-modal side knowledge[21, 22], with methods like VBPR [10] combine traditional CF with visual features, while MMGCN [50], GRCN [49] leverage GNNs to capture modality-aware higher-order collaborative signals. Recent approaches MMSSL [45] and MICRO [58] align modal signals with collaborative signals through contrastive SSL[19], revealing the informative aspects of modal signals that benefit recommendations. However, the data noise, heterogeneity, and incompleteness can introduce bias. To overcome this, LLMRec explores LLM-based augmentation to improve the quality of the data.

Large Language Models (LLMs) for Recommendation. LLMs have gained attention in recommendation systems, with various efforts to use them for modeling user behavior [14, 32, 42]. LLMs have been employed as an inference model in diverse recommendation tasks, including rating prediction, sequential recommendation, and direct recommendation $[1,5,6,57]$. Some efforts $[35,38]$ also tried to utilize LLMs to model structure relations. However, most previous methods primarily used LLMs as recommenders, abandoning the base model that has been studied for decades. We combine LLM-based data augmentation with classic $\mathrm{CF}$, achieving both result assurance and enhancement concurrently.

Data Augmentation for Recommendation. Extensive research has explored data augmentation in recommendation systems [13, 16]. Various operations, such as permutation, deletion, swap, insertion, and duplication, have been proposed for sequential recommendation [24, 30]. Commonly used techniques include counterfactual reasoning $[43,60]$ and contrastive learning [23]. Our LLMRec use LLMs as an inference model to augment edge and enhance node features by leveraging consensus knowledge from the large model.

## 6 CONCLUSION

This study focuses on the design of LLM-enhanced models to address the challenges of sparse implicit feedback signals and lowquality side information by profiling user interaction preferences and debiasing item attributes. To ensure the quality of augmented data, a denoised augmentation robustification mechanism is introduced. The effectiveness of LLMRec is supported by theoretical analysis and experimental results, demonstrating its superiority over state-of-the-art recommendation techniques on benchmark datasets. Future directions for investigation include integrating causal inference into side information debiasing and exploring counterfactual factors for context-aware user preference.

## REFERENCES

[1] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. arXiv preprint arXiv:2305.00447 (2023).

[2] Chong Chen, Weizhi Ma, Min Zhang, et al. 2023. Revisiting negative sampling vs. non-sampling in implicit recommendation. TOIS 41, 1 (2023), 1-25.

[3] Chong Chen, Min Zhang, Yongfeng Zhang, et al. 2020. Efficient neural matrix factorization without sampling for recommendation. TOIS 38, 2 (2020), 1-28.

[4] Mengru Chen, Chao Huang, Lianghao Xia, Wei Wei, et al. 2023. Heterogeneous graph contrastive learning for recommendation. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining. 544-552.

[5] Zheng Chen. 2023. PALR: Personalization Aware LLMs for Recommendation. arXiv preprint arXiv:2305.07622 (2023).

[6] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT's Capabilities in Recommender Systems. arXiv preprint arXiv:2305.02182 (2023).

[7] Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 2019. Graph neural networks for social recommendation. In ACM International World Wide Web Conference. 417-426.

[8] Xinyu Fu, Jiani Zhang, et al. 2020. Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding. In ACM International World Wide Web Conference. 2331-2341.

[9] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick 2022. Masked autoencoders are scalable vision learners. In CVPR. 16000-16009.

[10] Ruining He and Julian McAuley. 2016. VBPR: visual bayesian personalized ranking from implicit feedback. In $A A A I$, Vol. 30

[11] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In ACM SIGIR Conference on Research and Development in Information Retrieval. 639-648.

[12] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751 (2019).

[13] Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, and Jie Tang. 2021. MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

[14] Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, et al. 2023. Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv preprint arXiv:2305.06474 (2023).

[15] Hyeyoung Ko, Suyeon Lee, Yoonseo Park, and Anna Choi. 2022. A survey of recommendation systems: recommendation models, techniques, and application fields. Electronics 11, 1 (2022), 141.

[16] Dongha Lee, SeongKu Kang, Hyunjun Ju, et al. 2021. Bootstrapping user and item representations for one-class collaborative filtering. In ACM SIGIR Conference on Research and Development in Information Retrieval. 317-326

[17] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023. Text Is All You Need: Learning Language Representations for Sequential Recommendation. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

[18] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, et al. 2023. GPT4Rec A Generative Framework for Personalized Recommendation and User Interests Interpretation. arXiv preprint arXiv:2304.03879 (2023).

[19] Ke Liang, Yue Liu, Sihang Zhou, Wenxuan Tu, Yi Wen, Xihong Yang, Xiangjun Dong, and Xinwang Liu. 2023. Knowledge Graph Contrastive Learning Based on Relation-Symmetrical Structure. IEEE Transactions on Knowledge and Data Engineering (2023), 1-12. https://doi.org/10.1109/TKDE.2023.3282989

[20] Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2023. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Infor mation Retrieval Conference on Research and Development in Information Retrieval. $1559-1568$.

[21] Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, and Fuchun Sun. 2022. Reasoning over different types of knowledge graphs: Static, temporal and multi-modal. arXiv preprint arXiv:2212.05767 (2022).

[22] Ke Liang, Sihang Zhou, Yue Liu, Lingyuan Meng, Meng Liu, and Xinwang Liu 2023. Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning. arXiv preprint arXiv:2307.03591 (2023).

[23] Zhiwei Liu, Yongjun Chen, Jia Li, Philip S Yu, Julian McAuley, and Caiming Xiong. 2021. Contrastive self-supervised sequential recommendation with robust augmentation. arXiv preprint arXiv:2108.06479 (2021).

[24] Zhiwei Liu, Ziwei Fan, et al. 2021. Augmenting sequential recommendation with pseudo-prior items via reversely pre-training transformer. In ACM SIGIR Conference on Research and Development in Information Retrieval. 1608-1612.

[25] Ilya Loshchilov et al. 2017. Decoupled weight decay regularization. In ICLR.

26] Chang Meng, Chenhao Zhai, Yu Yang, Hengyu Zhang, and Xiu Li. 2023. Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation
In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 1797-1806.

[27] Chang Meng, Hengyu Zhang, Wei Guo, Huifeng Guo, Haotian Liu, Yingxue Zhang, Hongkun Zheng, Ruiming Tang, Xiu Li, and Rui Zhang. 2023. Hierarchical Projection Enhanced Multi-Behavior Recommendation. In Proceedings of the 29th ACM SIGACM SIGKDD Conference on Knowledge Discovery and Data Mining Conference on Knowledge Discovery and Data Mining. 4649-4660.

[28] Chang Meng, Ziqi Zhao, Wei Guo, Yingxue Zhang, Haolun Wu, Chen Gao, Dong Li, Xiu Li, and Ruiming Tang. 2023. Coarse-to-fine knowledge-enhanced multi-interest learning framework for multi-behavior recommendation. ACM Transactions on Information Systems 42, 1 (2023), 1-27.

[29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Conference on Neural Information Processing Systems 32 (2019).

[30] Aleksandr Petrov and Craig Macdonald. 2022. Effective and Efficient Training for Sequential Recommendation using Recency Sampling. In Recsys. 81-91

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, et al. 2021. Learning transferable visual models from natural language supervision. In ICML. PMLR, 8748-8763.

[32] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2023. Representation Learning with Large Language Models for Recommendation. arXiv preprint arXiv:2310.15950 (2023).

[33] Xubin Ren, Lianghao Xia, Yuhao Yang, Wei Wei, Tianle Wang, Xuheng Cai, and Chao Huang. 2023. SSLRec: A Self-Supervised Learning Library for Recommendation. arXiv preprint arXiv:2308.05697 (2023).

[34] Steffen Rendle, Christoph Freudenthaler, et al. 2012. BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618 (2012).

[35] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2023. GraphGPT: Graph Instruction Tuning for Large Language Models. arXiv preprint arXiv:2310.13023 (2023).

[36] Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, and Nitesh V Chawla. 2023. Heterogeneous graph masked autoencoders. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 9997-10005.

[37] Yijun Tian, Shichao Pei, Xiangliang Zhang, Chuxu Zhang, and Nitesh V Chawla. 2023. Knowledge Distillation on Graphs: A Survey. arXiv preprint arXiv:2302.00219 (2023).

[38] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, and Panpan Xu. 2023. Graph neural prompting with large language models. arXiv preprint arXiv:2309.15427 (2023).

[39] Yijun Tian, Chuxu Zhang, Zhichun Guo, Xiangliang Zhang, and Nitesh Chawla. 2022. Learning mlps on graphs: A unified view of effectiveness, robustness, and efficiency. In The Eleventh International Conference on Learning Representations.

[40] Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. 2021. Denoising implicit feedback for recommendation. In WSDM. 373-381

[41] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In ACM SIGIR Conference on Research and Development in Information Retrieval. 165-174.

[42] Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, and Ji-Rong Wen. 2023. Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models. arXiv preprint arXiv:2305.13112 (2023).

[43] Zhenlei Wang, Jingsen Zhang, Hongteng Xu, Xu Chen, Yongfeng Zhang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Counterfactual data-augmented sequential recommendation. In ACM SIGIR Conference on Research and Development in Information Retrieval. 347-356.

[44] Wei Wei, Chao Huang, Lianghao Xia, Yong Xu, Jiashu Zhao, and Dawei Yin. 2022. Contrastive meta learning with behavior multiplicity for recommendation. In Proceedings of the fifteenth ACM international conference on web search and data mining. 1120-1128.

[45] Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. 2023. Multi-Modal Self-Supervised Learning for Recommendation. In ACM International World Wide Web Conference. 790-800.

[46] Wei Wei, Lianghao Xia, and Chao Huang. 2023. Multi-Relational Contrastive Learning for Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 338-349.

[47] Yinwei Wei, Xiang Wang, et al. 2021. Hierarchical user intent graph network for multimedia recommendation. Transactions on Multimedia (TMM) (2021).

[48] Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, et al. 2021. Contrastive learning for cold-start recommendation. In ACM MM. 5382-5390.

[49] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, and Tat-Seng Chua. 2020. Graph-refined convolutional network for multimedia recommendation with implicit feedback. In MM. 3541-3549

[50] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019. MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In MM. 1437-1445.

[51] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, et al. 2021. Selfsupervised graph learning for recommendation. In ACM SIGIR Conference on Research and Development in Information Retrieval. 726-735.

[52] Zixuan Yi, Xi Wang, Iadh Ounis, and Craig Macdonald. 2022. Multi-modal Graph Contrastive Learning for Micro-video Recommendation. In ACM SIGIR Conference on Research and Development in Information Retrieval. 1807-1811.

[53] Yuxin Ying, Fuzhen Zhuang, Yongchun Zhu, Deqing Wang, and Hongwei Zheng. 2023. CAMUS: Attribute-Aware Counterfactual Augmentation for Minority Users in Recommendation. In ACM International World Wide Web Conference. 1396-1404.

[54] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are graph augmentations necessary? Simple graph contrastive learning for recommendation. In ACM SIGIR Conference on Research and Develop ment in Information Retrieval. 1294-1303.

[55] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to go next for recommender systems? id-vs. modality-based recommender models revisited. In ACM SIGIR Conference on Research and Development in Information Retrieval.

[56] Honglei Zhang, Fangyuan Luo, Jun Wu, Xiangnan He, and Yidong Li. 2023 LightFR: Lightweight federated recommendation with privacy-preserving matrix factorization. ACM Transactions on Information Systems 41, 4 (2023), 1-28.

[57] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as instruction following: A large language mode empowered recommendation approach. arXiv preprint arXiv:2305.07001 (2023)

[58] Jinghao Zhang, Yanqiao Zhu, Qiang Liu, et al. 2022. Latent structure mining with contrastive modality fusion for multimedia recommendation. TKDE (2022).

[59] Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Shu Wu, et al. 2021. Mining Latent Structures for Multimedia Recommendation. In MM. 3872-3880.

[60] Shengyu Zhang, Dong Yao, Zhou Zhao, et al. 2021. Causerec: Counterfactual user sequence synthesis for sequential recommendation. In ACM SIGIR Conference on Research and Development in Information Retrieval. 367-377.

[61] Ding Zou, Wei Wei, Xian-Ling Mao, Ziyang Wang, Minghui Qiu, Feida Zhu, and Xin Cao. 2022. Multi-level cross-view contrastive learning for knowledge-aware recommender system. In ACM SIGIR Conference on Research and Development in Information Retrieval. 1358-1368.


[^0]:    *Chao Huang is the corresponding author.

    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

    WSDM '24, March 4-8, 2024, Merida, Mexico

    (c) 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0371-3/24/03...\$15.00

    https://doi.org/10.1145/3616855.3635853

[^1]:    ${ }^{1}$ https://files.grouplens.org/datasets/movielens/ml-10m-README.html

    ${ }^{2}$ https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

    ${ }^{3}$ https://platform.openai.com/docs/api-reference

</end of paper 0>


<paper 1>
# GraphGPT: Graph Instruction Tuning for Large Language Models 

Jiabin Tang<br>University of Hong Kong<br>jiabintang77@gmail.com<br>Lei Shi<br>Baidu Inc.<br>harryshi.cs@gmail.com

Yuhao Yang<br>University of Hong Kong<br>yuhao-yang@outlook.com<br>Lixin $\mathrm{Su}$<br>Baidu Inc.<br>sulixinict@gmail.com

Wei Wei<br>University of Hong Kong<br>weiwei1206cs@gmail.com<br>Suqi Cheng<br>Baidu Inc.<br>chengsuqi@gmail.com

Dawei Yin<br>Baidu Inc.<br>yindawei@acm.org

Chao Huang*<br>University of Hong Kong<br>chaohuang75@gmail.com


#### Abstract

Graph Neural Networks (GNNs) have evolved to understand graph structures through recursive exchanges and aggregations among nodes. To enhance robustness, self-supervised learning (SSL) has become a vital tool for data augmentation. Traditional methods often depend on fine-tuning with task-specific labels, limiting their effectiveness when labeled data is scarce. Our research tackles this by advancing graph model generalization in zero-shot learning environments. Inspired by the success of large language models (LLMs), we aim to create a graph-oriented LLM capable of exceptional generalization across various datasets and tasks without relying on downstream graph data. We introduce the GraphGPT framework, which integrates LLMs with graph structural knowledge through graph instruction tuning. This framework includes a text-graph grounding component to link textual and graph structures and a dual-stage instruction tuning approach with a lightweight graph-text alignment projector. These innovations allow LLMs to comprehend complex graph structures and enhance adaptability across diverse datasets and tasks. Our framework demonstrates superior generalization in both supervised and zero-shot graph learning tasks, surpassing existing benchmarks. The open-sourced model implementation of our GraphGPT is available at https://github.com/HKUDS/GraphGPT.


## CCS CONCEPTS

- Information systems $\rightarrow$ Data mining; Language models; $\cdot$ Mathematics of computing $\rightarrow$ Graph algorithms.


## KEYWORDS

Large Language Models, Graph Learning, Instruction Tuning[^0]

## 1 INTRODUCTION

Graph neural networks (GNNs) have emerged as a powerful framework for analyzing and learning from graph-structured data [4, 27], enabling advancements in various domains, such as social network analysis [31, 65], recommender systems [9, 42], and biological network analysis $[6,25]$. One of the key benefits of GNNs is their ability to capture the inherent structural information and dependencies present in graph data. By leveraging message passing and aggregation mechanisms, GNNs can effectively propagate and combine information across the graph, enabling them to model complex relationships and make accurate predictions.

In recent years, various GNN architectures have introduced innovations in how information is exchanged and aggregated among graph nodes. For example, graph convolutional network (GCNs) [17, 22] adapt convolutional operations to the graph domain, enabling effective feature representations. Graph attention networks (GATs) [39, 43] leverages attention mechanisms to assign different weights to neighboring nodes, allowing for more fine-grained information aggregation. Graph transformer networks (GTNs) [14, 60] incorporate self-attention and positional encoding to capture global dependencies and structural patterns in the graph. However, a notable limitation of many GNN approaches is their heavy reliance on supervised learning, which can lead to inadequate robustness and generalization when confronted with sparse and noisy data.

To enhance the generalization ability of GNNs, self-supervised learning (SSL) has emerged as a promising approach in graph representation learning. It aims to pre-train a robust graph model using auxiliary tasks on unlabeled graph data. The idea is to leverage the inherent structure and patterns within the graph itself to create meaningful self-supervisory signals. SSL-enhanced graph learning methods exhibit two primary paradigms: contrastive SSL and generative SSL. Within contrastive SSL, the emphasis lies on learning representations by contrasting positive and negative samples, with notable advancements of DGI [40] and GCA [67]. Conversely, generative SSL focuses on generating synthetic samples that closely resemble the original graph structures with masked autoencoders, exemplified by techniques like GraphMAE [11] and S2GAE [35].

While these methods aim to generate graph embeddings that are generalizable to different downstream tasks, they often require a fine-tuning process using labels specific to the downstream graph learning scenarios. However, this reliance on labeled data from
downstream tasks can restrict their generalization in practical situations where obtaining high-quality labels may not always be feasible. This limitation is particularly relevant in learning scenarios like cold-start recommendation systems or traffic flow prediction in new cities where accurate labels may be scarce or unavailable.

As a result, the objective of this research is to advance the generalization capabilities of graph models by addressing challenging and practical zero-shot learning scenarios. Inspired by the remarkable success of large language models (LLMs) in natural language processing (NLP) tasks [48], where they have demonstrated exceptional generalization abilities, this work aims to develop a graph-oriented LLM capable of achieving high generalization across diverse downstream datasets and tasks. However, effectively integrating large language models with graph learning poses non-trivial challenges.

- C1: Achieving a proper alignment between the structural information of a graph and the language space demands meticulous deliberation and thoughtful consideration.
- C2: Effectively guiding LLMs to comprehend the structural information of graphs remains a considerable challenge.
- C3: Endowing LLMs with the ability to reason step-by-step is important when tackling complex graph learning tasks.

To gain a deeper understanding of the limitations associated with directly prompting LLMs using purely text-based prompts for graph structure modeling, we provide illustrative examples in Figure 1. These examples facilitate a comparative analysis between our GraphGPT framework and the ChatGPT approach. We focus on a representative node classification task, where the objective is to predict the category of a given paper. In Figure 1 (a) and Figure 1 (b), we showcase the prediction results for two scenarios using ChatGPT: (1) utilizing only the input node textual data, and (2) employing text-based graph structure-aware prompts inspired by the prompt designs in recent studies [2, 5]. These figures highlight the potential limitations that arise when relying solely on textbased prompts for graph structure modeling, as evidenced by the incorrect paper node classification results presented. In contrast, our GraphGPT framework effectively addresses these limitations by preserving and leveraging the graph structural information, as shown in Figure 1 (c). It enables accurate identification of the paper category, in understanding the underlying graph structure.

Additionally, the utilization of text-based structural prompts leads to an increase in token size, which presents challenges in practical scenarios. Longer token sequences incur higher computational and memory costs, making it less feasible for real-world applications. Furthermore, existing LLMs have token limits, which further restrict the applicability of longer text-based prompts for large-scale graph structure modeling. These limitations emphasize the necessity for more efficient and scalable approaches that can effectively incorporate graph structural information into LLMs.

Contributions. To address these challenges, we propose a novel framework called GraphGPT, which aims to align Large Language Models (LLMs) with Graphs using a carefully designed graph instruction tuning paradigm. (C1) Our framework introduces a textgraph grounding paradigm as the initial step to align the encoding of graph structures with the natural language space. By incorporating textual information in a contrastive manner, we enable effective alignment of graph structure information within language models.

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-02.jpg?height=634&width=805&top_left_y=301&top_left_x=1126)

Input: (a)ChatGPT with Node Content only Token Length: 615

Title: TiM-DNN: Ternary in-Memory accelerator for Deep Neural Networks

Question:
Output:

CS.AR, CS, AI, CS SY,

Therefore, the most likely category for this paper is cs.AR

(b) ChatGPT with Node Content and Token Length: 4649

Abstract: The use of lower precision has emerged as a popular technique

Title: TiM-DNN: Ternary in-Memory accelerator for Deep Neural Networks

With it as central node (paper 0), a citation graph can be constructed.

The citation relations: Paper 0 cites Paper 1, , ... cites Paper 102.

Question: Which arXiv CS sub-category does this paper belong to? ... . . . . . .

Output:

Based on the title and Abstract, the paper is likely to belong:

AR (Hardware Architecture)

Input: $\quad$ (c) GraphGPT $\quad$ Token Length: 750

with the following information:

Abstract: The use of lower precision has emerged as a popular technique

Title: TiM-DNN: Ternary in-Memory accelerator for Deep Neural Networks

Output:

Based on the title and abstract, we can identify the following

sub-categories that are most likely to be relevant:1. cs.LG

Figure 1: Limitation of LLMs in understanding graph structural contexts with heavy reliance on textual data.

(C2) In our proposed dual-stage graph instruction tuning paradigm, we leverage self-supervised signals through the graph matching task, which is derived from unlabeled graph structures, to serve as instructions for guiding model tuning of LLMs. By incorporating this self-supervised instruction tuning, the language model acquires domain-specific structural knowledge related to graphs, thereby enhancing its understanding of graph structures. To further customize the LLM's reasoning behavior for diverse downstream graph learning tasks, the second stage of our graph instruction tuning paradigm involves fine-tuning the LLM with task-specific graph instructions, to improve the model's adaptability. (C3) By incorporating the Chain-of-Thought (COT) distillation into our framework, GraphGPT enhances its step-by-step reasoning abilities and improves its performance in the face of distribution shift.

In summary, our work makes the following contributions:

- This work aims to align graph domain-specific structural knowledge with the reasoning ability of Large Language Models (LLMs) to improve the generalization of graph learning.
- Our approach aims to align LLMs with Graphs through a graph instruction tuning paradigm. This paradigm incorporates selfsupervised instruction tuning, enhancing the LLM's comprehension of graph structural knowledge and its reasoning capabilities. Additionally, we introduce task-specific instruction tuning to improve the model's adaptability across diverse graph tasks.
- We evaluate our proposed GraphGPT on supervised and zeroshot graph learning tasks. We conduct thorough analyses of its component-wise effects and generalization ability. By comparing it with state-of-the-art baselines, we demonstrate the superior generalization power of our approach across various settings.


## 2 PRELIMINARIES

Graph-structured Data. represents information as entities (nodes) and the relationships (edges) between them. A graph is denoted as $\mathcal{G}(\mathcal{V}, \mathcal{E}, \mathrm{A}, \mathrm{X})$, comprising key components. The node set $\mathcal{V}$ represents the collection of nodes, with $|\mathcal{V}|=N$ indicating the total number of nodes. The edge set $\mathcal{E}$ characterizes the relationships
between nodes. The adjacency matrix $\mathrm{A} \in \mathbb{R}^{N \times N}$ encodes the graph's topology, with each element $A_{i, j}$ indicating the presence or absence of an edge between nodes $i$ and $j$. The feature matrix $\mathrm{X} \in \mathbb{R}^{N \times F}$ contains attribute or feature information associated with each node, where $F$ represents the feature dimensionality.

Graph Neural Networks. have become a powerful framework for learning representations from graph-structured data. Unlike traditional neural networks that process grid-like data, GNNs excel in capturing the intricate relationships and dependencies within graphs. They utilize the graph's structure-comprising nodes and edges-to derive expressive node representations through repeated message propagation and aggregation operations.

$$
\begin{align*}
m_{v}^{(l)} & =\operatorname{Propagate}^{(l)}\left(\left\{h_{u}^{(l-1)}: u \in \mathcal{N}(v)\right\}\right) \\
h_{v}^{(l)} & =\operatorname{Aggregate}^{(l)}\left(h_{v}^{(l-1)}, m_{v}^{(l)}\right) \tag{1}
\end{align*}
$$

In Graph Neural Networks, the feature vector of node $v$ at layer $l$ is denoted as $h_{v}^{(l)}$. Message passing is performed by the Propagate ${ }^{(l)}$ function, aggregating information from neighboring nodes of $v$ in layer $l$. The Aggregate ${ }^{(l)}$ function combines this information with the previous layer's representation of node $v$ to update $h_{v}^{(l)}$. By incorporating graph structure into learned representations, GNNs can be tailored for tasks like node classification and link prediction.

## 3 METHODOLOGY

### 3.1 Structural Information Encoding with Text-Graph Grounding

To improve the understanding of graph structural information by large language models, our framework focuses on aligning the encoding of graph structures with the natural language space. This alignment enables language models to effectively comprehend the graph's structural elements using their language understanding capabilities. To achieve this, we introduce a text-graph grounding paradigm that generates prompts preserving the graph's structural context for language models. This paradigm acts as a bridge, connecting the semantic understanding of textual information with the inherent structural relationships in the graph.

In our GraphGPT, we design the graph encoder to be highly flexible, allowing it to leverage a wide range of backbone GNN architectures obtained from diverse graph pre-training paradigms. We incorporate a message-passing neural network architecture, which can be a graph transformer [60] or a graph convolutional network [17], as the structure-level pre-trained graph model. In each message-passing step, the graph encoder aggregates information from neighboring nodes, considering their relationships:

$$
\begin{equation*}
\mathbf{H}^{(l)}=\sigma\left(\tilde{\mathbf{A}} \mathbf{H}^{(l-1)} \mathbf{W}\right) \tag{2}
\end{equation*}
$$

The self-loop adjacency matrix, denoted as $\tilde{\mathbf{A}}$, is obtained by adding the identity matrix $\mathbf{I}$ to the original adjacency matrix $\mathbf{A}$. $\mathbf{W}$ is the parameter matrix. This matrix captures the self-connections and local connectivity of nodes in the graph. $\sigma(\cdot)$ is the non-linear activation. $\mathbf{H}^{(l)}$ is the graph representations at the $l$-th layer.

Text-Structure Alignment. To enhance the alignment of graph structure information with Language Models (LLMs), our focus is on exploring effective encoding methods that can collaborate seamlessly with LLMs. Building upon previous works [30, 49], we adopt a contrastive approach by incorporating textual information into the graph structure encoding process. We directly integrate a pre-trained graph encoder into our GraphGPT framework, enabling the seamless utilization of its capabilities. Formally, given a graph $\mathcal{G}(\mathcal{V}, \mathcal{E}, \mathrm{A}, \mathrm{X})$ with raw textual contents $\mathrm{C}=c_{i} \in \mathbb{R}^{l_{i} \times d}, 1 \leq i \leq N$ for $N$ nodes, we obtain encoded graph representations $\hat{\mathbf{H}} \in \mathbb{R}^{N \times d}$ and encoded text representations $\hat{\mathrm{T}} \in \mathbb{R}^{N \times d}$ as follows:

$$
\begin{equation*}
\mathbf{H}=f_{\mathbf{G}}(\mathbf{X}), \mathbf{T}=f_{\mathbf{T}}(\mathbf{C}), \hat{\mathbf{H}}=\operatorname{norm}(\mathbf{H}), \hat{\mathbf{T}}=\operatorname{norm}(\mathbf{T}) \tag{3}
\end{equation*}
$$

We utilize the graph encoder, $f_{\mathrm{G}}$, to generate structure-level graph representations from the input graph $\mathcal{G}(\mathcal{V}, \mathcal{E}, \mathrm{A}, \mathrm{X})$. To encode the raw textual contents $\mathrm{C}$ associated with the nodes, we employ a text encoder, such as a transformer or BERT, denoted as $f_{\mathrm{T}}$. This step produces encoded text representations of nodes, which are then normalized row-wise using the norm function. The text-structure alignment across modalities is conducted as follows:

$$
\begin{gather*}
\Gamma_{1}=\left(\hat{\mathbf{H}} \hat{\mathbf{T}}^{\top}\right) \cdot \exp (\tau), \Gamma_{2}=\left(\hat{\mathbf{H}} \hat{\mathbf{T}}^{\prime \top}\right) \cdot \exp (\tau), \Gamma_{3}=\left(\hat{\mathbf{T}}^{\top} \hat{\mathbf{T}}^{\prime \top}\right) \cdot \exp (\tau) \\
\mathcal{L}=\sum_{i=1}^{3} \frac{1}{2} \lambda_{i}\left(\operatorname{CE}\left(\Gamma_{i}, \mathbf{y}\right)+\operatorname{CE}\left(\Gamma_{i}^{\top}, \mathbf{y}\right)\right) \tag{4}
\end{gather*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-03.jpg?height=60&width=851&top_left_y=1141&top_left_x=1098)
ber of nodes. In our text-graph grounding, we use the label $\mathbf{y}=$ $(0,1, \cdots, n-1)^{\top}$ for the contrastive alignment objective. We employ a graph transformer [61] as the graph encoder and a vanilla transformer [38] as the text encoder.

### 3.2 Dual-Stage Graph Instruction Tuning

The dual-stage graph instruction tuning paradigm proposed in this work builds upon the concept of instruction tuning, which has been recently introduced to enhance the adaptability of language models for specific domains [45]. In this paradigm, we aim to align the language capacity of the model with the nuances of graph learning tasks, enabling the language model to generate more accurate and contextually appropriate responses for graph-structured data.

3.2.1 Self-Supervised Instruction Tuning. In the initial stage of our graph instruction tuning, we introduce self-supervised instruction tuning. This mechanism enhances the language model's reasoning abilities by incorporating graph domain-specific structural knowledge and effectively understanding contextual information within the graph's structure. To achieve this, we utilize self-supervised signals derived from unlabeled graph structures as instructions for model tuning. Specifically, we design a structureaware graph matching task that guides the language model in differentiating between graph tokens using language tokens. This instruction task plays a vital role in accurately associating graph tokens with their corresponding textual descriptions, deepening the model's comprehension of the graph with the provided guidance.

Instruction Design. The instruction for our graph matching task consists of three components: i) graph information, ii) human question, and iii) GraphGPT response. In this task, we treat each node in the graph as a central node and perform h-hops with random neighbor sampling, resulting in a subgraph structure. The natural language input for the LLM is the human question. In the context of the graph matching task, the instruction includes the indicator

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-04.jpg?height=542&width=637&top_left_y=279&top_left_x=210)

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-04.jpg?height=555&width=531&top_left_y=281&top_left_x=859)

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-04.jpg?height=561&width=529&top_left_y=275&top_left_x=1362)

Figure 2: The overall architecture of our proposed GraphGPT with graph instruction tuning paradigm.

token <graph> and a shuffled list of node text information. For example, in a citation graph, the node text information corresponds to paper titles. The objective of the LLM in the graph matching task is to align each graph token with its corresponding node text information. This requires reordering the node text information list based on the sequence of graph tokens, effectively associating each graph token with its relevant textual description. The detailed designs of graph matching are shown in Figure 4.

Tuning Strategy. To optimize the tuning process efficiently, we propose incorporating a Lightweight Alignment Projector. During training, we focus on optimizing the parameters of the projector $f_{\mathbf{P}}$, while keeping the parameters of both the LLM and the graph encoder fixed. We assume that the projector successfully learns to map the encoded graph representation to graph tokens, while the LLM excels at aligning these tokens with diverse node text information. To align the graph tokens with the language tokens, we employ a projector $f_{\mathbf{P}}$, which can be as simple as a single linear layer. This projector establishes correspondence between the graph tokens and the language tokens. By replacing the indicator token <graph> in the original language token sequence, the aligned graph tokens create a modified token sequence for the LLM. This modified sequence, denoted as $\left\{<\right.$ graph_begin $>,<$ graph_token $>_{1}, \cdots,<$ graph_token $>_{n}$ <graph_end>\}, corresponds to the number of nodes $n$ in the graph associated with the given prompt. Given that the graph matching process is unsupervised, we have the opportunity to leverage a vast amount of unlabeled graph data from different domains, to enhance the generalizability of the learned projector. Mathematically, with projected graph tokens $\mathbf{X}_{\mathcal{G}}=f_{\mathrm{P}}(\hat{\mathbf{H}})$ and text embeddings $\mathbf{X}_{I}=$ tokenizer(instruction), for a sequence of length $L$, we compute the probability of generating the target output $\mathrm{X}_{O}$ as follows:

$$
\begin{equation*}
p\left(\mathbf{X}_{O} \mid \mathbf{X}_{\mathcal{G}}, \mathbf{X}_{\mathcal{I}}\right)=\prod_{i=1}^{L} p_{\theta}\left(x_{i} \mid \mathbf{X}_{\mathcal{G}}, \mathbf{X}_{I,<i}, \mathbf{X}_{O,<i}\right) \tag{5}
\end{equation*}
$$

where $\theta$ are the learnable parameters within GraphGPT.

3.2.2 Task-Specific Instruction Tuning. In the second stage, we introduce task-specific instruction tuning to customize the model's reasoning behavior for different graph learning tasks, such as node classification or link prediction. By fine-tuning the LLM using taskspecific graph instructions, we guide the model to generate responses that align with the constraints and requirements of the specific graph learning task. This enhances the model's adaptability and performance in handling diverse graph learning tasks.

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-04.jpg?height=165&width=574&top_left_y=1159&top_left_x=1231)

Figure 3: Workflow of text-structure alignment.

Instruction Design. We utilize a consistent instruction template comprising three parts. To generate graph information for each node, we employ the same neighbor sampling approach as in the first stage. This approach ensures the inclusion of relevant graph information, with each node serving as the central node. For the node classification task, the human question instruction includes the indicator token <graph> and specific text information about the central node. This instruction guides the language model to predict the category of the central node based on both the graph structure data and the accompanying text information. Figure 4 provides instruction examples for different tasks, visually illustrating the presentation of the instruction to the language model.

Tuning Strategy. In the second stage of training, we utilize the parameters of the structure-aware projector that were trained in the first stage as the initial state. This allows us to conduct instruction tuning specifically for downstream tasks. During this training process, we keep the parameters of the language model (LLM) and graph encoder fixed, focusing solely on optimizing the parameters of the projector from the previous stage. By doing so, we ensure that the LLM further aligns with the requirements of downstream tasks, enhancing its ability to comprehend and interpret graph structures.

After completing the two training stages as described above, we have confidence that our GraphGPT has acquired the capability to comprehend the given graph structure and perform downstream tasks on the provided graph. The training process involving instruction tuning and the freezing of specific model parameters has refined the model's understanding of graph structures, enabling it to effectively tackle various tasks associated with the given graph.[^1]

Figure 4: Our instruction designs for graph matching task (upper), node classification (middle) and link prediction (lower).

### 3.3 Chain-of-Thought (CoT) Distillation

When faced with diverse graph data, language models may encounter unfamiliar patterns and structures, leading to challenges in generating accurate and coherent responses. This is especially true when the number of node classes varies across different types of graph data, causing distribution shift. To address this challenge and enhance accuracy in the presence of distribution shift, it is crucial to equip our GraphGPT with step-by-step reasoning abilities. Thus, we propose incorporating the Chain-of-Thought (COT) technique [47], which explicitly models the flow of thoughts and reasoning steps. By leveraging COT, our language model improves the coherence and consistency of generated text, enabling it to follow a logical progression of ideas and enhance its understanding and reasoning capabilities for the given graph data.

Incorporating the Chain-of-Thought (COT) technique can be challenging due to the influence of model parameter scale [32]. To overcome this, we draw inspiration from previous research [32] and adopt a distillation approach. By extracting valuable knowledge from a closed-source, powerful language model like ChatGPT (with over 200 billion parameters), we can generate high-quality COT instructions and enhance our model's COT reasoning capabilities without increasing the parameter count.

COT Distillation Paradigm. Our approach involves designing tailored Chain-of-Thought (COT) prompts for node-specific tasks. For the node classification task in a citation graph, we provide the abstract, paper title, and a task description as input. Using the GPT3.5 language model (LLM), we incorporate "Please think about the categorization in a step-by-step manner." to enable step-by-step reasoning. By engaging in sequential thought, the LLM generates output that includes predictions for node classes and detailed explanations for each prediction. This ensures transparent and comprehensible reasoning and decision-making. To further enhance performance, we integrate the generated COT instruction data with previously designed instructions for task-specific instruction tuning. With the integrated instructions, we proceed with the proposed instruction tuning paradigm.

## 4 EVALUATION

We conduct experiments to address key research questions:

- RQ1: How does the proposed GraphGPT framework perform in both supervised and zero-shot graph learning settings?
- RQ2: What is the generalization ability of our model in handling multiple tasks without experiencing catastrophic forgetting?
- RQ3: What is the contribution of various key components in the proposed GraphGPT framework to its overall performance? - RQ4: How scalable and efficient is our GraphGPT framework?


### 4.1 Experimental Settings

4.1.1 Data Descriptions. We evaluate our GraphGPT using three datasets: OGB-arxiv, PubMed, and Cora. The OGB-arxiv dataset [12] represents a directed graph capturing the citation network among computer science arXiv papers indexed by MAG [41]. Each paper is manually labeled with a research category selected from 40 subject areas. The PubMed dataset [8] consists of 19,717 scientific publications on diabetes from the PubMed database, categorized into Experimental induced diabetes, Type 1 diabetes, and Type 2 diabetes. Additionally, it includes a citation network with 44,338 links. The Cora dataset [49] comprises 25,120 research papers connected through citations. We use an expanded version with 70 classes, larger than previous versions [17].

4.1.2 Evaluation Protocols. To facilitate comparison across different datasets, we map node features into a unified vector space by encoding raw text information with a pre-trained BERT [3]. In our experiments, we partition the Cora and PubMed datasets into training, validation, and testing sets following a 3:1:1 ratio, as described in previous works [8, 49]. For the OGB-arxiv data, we adhere to the public split setting [12] with a training, validation, and testing ratio of 6:2:3. To evaluate our model's performance, we utilize three commonly used metrics: Accuracy and Macro F1 for node classification, and AUC for link prediction.

4.1.3 Baseline Methods. In our performance comparison, we consider various state-of-the-art methods for comprehensive evaluation. (i) The first category includes MLP, which employs a Multilayer Perceptron for node representation. (ii) The second category comprises representative graph neural encoders, including GraphSAGE [7], GCN [17], GAT [39], and RevGNN [21]. (iii) The third category focuses on the self-supervised approach DGI [40] for graph learning. (iv) The fourth category explores knowledge distillationenhanced GNNs, with GKD [55] and GLNN [63] as notable methods. (v). The fifth category showcases recently proposed strong graph transformer networks, with NodeFormer [51] and DIFFormer [50] as competitors. (vi) Lastly, we consider open-sourced LLMs, such as Baichuan-7B, vicuna-7B-v1.1, and vicuna-7B-v1.5 as baselines for understanding text-attributed graph data.

4.1.4 Implementation Details. For our model implementation, we primarily use the PyTorch and Transformers libraries. We utilize

Table 1: Performance comparison of various methods on node classification under both supervised and zero-shot settings.

| Dataset | Arxiv-Arxiv |  | Arxiv-PubMed |  | Arxiv-Cora |  | (Arxiv+PubMed)-Cora |  | (Arxiv+PubMed)-Arxiv |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model | Accuracy | Macro-F1 | acc | Macro-F1 | Accuracy | Macro-F1 | Accuracy | Macro-F1 | Accuracy | Macro-F1 |
| MLP | 0.5179 | 0.2536 | 0.3940 | 0.1885 | 0.0258 | 0.0037 | 0.0220 | 0.0006 | 0.2127 | 0.0145 |
| GraphSAGE | 0.5480 | 0.3290 | 0.3950 | 0.1939 | 0.0328 | 0.0132 | 0.0132 | 0.0029 | 0.1281 | 0.0129 |
| GCN | 0.5267 | 0.3202 | 0.3940 | 0.1884 | 0.0214 | 0.0088 | 0.0187 | 0.0032 | 0.0122 | 0.0008 |
| GAT | 0.5332 | 0.3118 | 0.3940 | 0.1884 | 0.0167 | 0.0110 | 0.0161 | 0.0057 | 0.1707 | 0.0285 |
| RevGNN | 0.5474 | 0.3240 | 0.4440 | 0.3046 | 0.0272 | 0.0101 | 0.0217 | 0.0016 | 0.1309 | 0.0126 |
| DGI | 0.5059 | 0.2787 | 0.3991 | 0.1905 | 0.0205 | 0.0011 | 0.0205 | 0.0011 | 0.5059 | 0.2787 |
| GKD | 0.5570 | 0.1595 | 0.3645 | 0.2561 | 0.0470 | 0.0093 | 0.0406 | 0.0037 | 0.2089 | 0.0179 |
| GLNN | 0.6088 | 0.3757 | 0.4298 | 0.3182 | 0.0267 | 0.0115 | 0.0182 | 0.0092 | 0.3373 | 0.1115 |
| NodeFormer | 0.5922 | 0.3328 | 0.2064 | 0.1678 | 0.0152 | 0.0065 | 0.0144 | 0.0053 | 0.2713 | 0.0855 |
| DIFFormer | 0.5986 | 0.3355 | 0.2959 | 0.2503 | 0.0161 | 0.0094 | 0.0100 | 0.0007 | 0.1637 | 0.0234 |
| baichuan-7B | 0.0946 | 0.0363 | 0.4642 | 0.3876 | 0.0405 | 0.0469 | 0.0405 | 0.0469 | 0.0946 | 0.0363 |
| vicuna-7B-v1.1 | 0.2657 | 0.1375 | 0.5251 | 0.4831 | 0.1090 | 0.0970 | 0.1090 | 0.0970 | 0.2657 | 0.1375 |
| vicuna-7B-v1.5 | 0.4962 | 0.1853 | 0.6351 | 0.5231 | 0.1489 | 0.1213 | 0.1489 | 0.1213 | 0.4962 | 0.1853 |
| GraphGPT-7B-v1.1-cot | 0.4913 | 0.1728 | 0.6103 | 0.5982 | 0.1145 | 0.1016 | 0.1250 | 0.0962 | 0.4853 | 0.2102 |
| GraphGPT-7B-v1.5-stage2 | 0.7511 | 0.5600 | 0.6484 | 0.5634 | 0.0813 | 0.0713 | 0.0934 | 0.0978 | 0.6278 | 0.2538 |
| GraphGPT-7B-v1.5-std | 0.6258 | 0.2622 | 0.7011 | 0.6491 | 0.1256 | 0.0819 | 0.1501 | 0.0936 | 0.6390 | 0.2652 |
| GraphGPT-7B-v1.5-cot | 0.5759 | 0.2276 | 0.5213 | 0.4816 | 0.1813 | 0.1272 | 0.1647 | 0.1326 | 0.6476 | 0.2854 |
| $\mathrm{p}$-val | $2.26 e^{-9}$ | $1.56 e^{-10}$ | $2.22 e^{-7}$ | $1.55 e^{-9}$ | $1.04 e^{-9}$ | $9.96 e^{-6}$ | $7.62 e^{-8}$ | $1.97 e^{-7}$ | $1.5 \mathrm{e}^{-13}$ | $4.63 e^{-6}$ |

Vicuna-7B-v1.1 and Vicuna-7B-v1.5 as the base models. The batch size is set to 2 per GPU, and the learning rate is $2 e^{-3}$. We apply a warmup ratio of $3 e^{-2}$ and set the maximum input length of the Large Language Model (LLM) to 2048. The training process runs for 3 epochs. In the task-specific instruction tuning stage, we explore various combinations of instruction data to assess the model's performance under different data mixtures. The hyperparameter settings remain constant, except for the number of training epochs, which is set to 2 in this stage. The alignment projector parameters fine-tuned in the self-supervised instruction tuning stage serve as the initial parameters for the projector in the second tuning stage. For evaluating most baselines, we use their publicly available code. For more implementation details, please refer to our released code.

### 4.2 Overall Performance Comparison (RQ1)

We conduct experiments on the node classification task, evaluating both supervised and zero-shot scenarios. The overall performance is summarized in Table 1. In the Supervised Task Setting, models are trained on a specific dataset and evaluated on the corresponding test set (e.g., training on Arxiv-Arxiv and testing on the Arxiv test set). In the Zero-Shot Task Setting, models are trained on a specific dataset and tested on other datasets without additional training (e.g., training on Arxiv-PubMed and testing on the PubMed dataset) To handle variations in the number of classes across datasets, we employ a transfer-trained classifier, such as a linear layer, when testing GNN-based models. In Table 1, "-7B-" indicates the parameter scale, while "-v1.1-" and "-v1.5-" represent different versions of the base Vicuna model. "-stage2" indicates the adoption of only the second stage tuning. "-std" and "-cot" denote the use of the standard and generated COT instruction datasets, respectively.

Obs.1: Overall Superiority of our GraphGPT. Our graph LLM consistently outperforms various state-of-the-art baselines in both supervised and zero-shot scenarios. Notably, even recently developed strong GNN-based models, such as NodeFormer, DIFFormer, and GKD, exhibit good structural modeling capabilities in the supervised setting. However, when transferred to new datasets without further training, their performance significantly declines. In contrast, our GraphGPT not only surpasses all state-of-the-art methods in supervised tasks but also achieves a remarkable 2-10 times increase in accuracy in the zero-shot graph learning scenario.

LLM-based solutions like Baichuan-7B and Vicuna-7B maintain stable performance across different datasets but rely solely on text information for predictions. In contrast, our GraphGPT preserves graph structure, providing a comprehensive solution for graph learning tasks. Two key factors contribute to these improvements: (i) Our dual-stage graph instruction tuning aligns structural information encoded by the graph encoder with natural language tokens, enabling the LLM to understand the graph's inherent characteristics. (ii) Our framework facilitates mutual enhancement between the graph encoder and LLM, filling the LLM's gap in structural understanding and enabling it to reason about the graph's structure.

Obs.2: Benefits with Structure-aware Graph Matching. The presence of the first stage, which involves self-supervised graph matching tasks for instruction tuning, plays a crucial role in enhancing the zero-shot transferability of our GraphGPT. The first stage focuses on aligning the graph tokens, which encode rich structural information, with the language tokens. This alignment enables the model to develop a deeper understanding of the inherent structural characteristics of the graph data. Without the first stage, if we only conduct the second stage of task-specific instruction tuning, the model tends to be more prone to overfitting on the specific dataset. In such cases, the model's performance may be heavily reliant on dataset-specific patterns and characteristics, rather than a genuine understanding of the underlying graph structure. This can limit the model's ability to generalize to new, unseen datasets.

Obs.3: Benefits with COT Distillation. The "-std" and "-cot" variants indicate that the use of COT distillation substantially benefits more complex graph learning tasks. Models tuned with the standard instruction dataset can already achieve prominent results when transferred to simpler tasks, such as the PubMed dataset with 3 classes, with an accuracy of 0.7011 for Arxiv-PubMed. However,

Table 2: Performance comparison of various instruction mixtures in supervised learning on the Arxiv dataset and the zero-shot setting on the Cora dataset for node classification.

| Dataset | Supervision. on Arxiv |  | Zero Shot on Cora |  |
| :---: | :---: | :---: | :---: | :---: |
| Model | Acc | Macro-F1 | Acc | Macro-F1 |
| MLP | 0.5179 | 0.2536 | 0.0220 | 0.0006 |
| GraphSAGE | 0.5480 | 0.3290 | 0.0132 | 0.0029 |
| GCN | 0.5267 | 0.3202 | 0.0187 | 0.0032 |
| GAT | 0.5332 | 0.3118 | 0.0161 | 0.0057 |
| RvGNN | 0.5474 | 0.3240 | 0.0217 | 0.0016 |
| DGI | 0.5059 | 0.2787 | 0.0205 | 0.0011 |
| GKD | 0.5570 | 0.1595 | 0.0406 | 0.0037 |
| GLNN | 0.6088 | 0.3757 | 0.0182 | 0.0092 |
| NodeFormer | 0.5922 | 0.3328 | 0.0144 | 0.0053 |
| DIFFormer | 0.5986 | 0.3355 | 0.0100 | 0.0007 |
| baichuan-7b | 0.0946 | 0.0363 | 0.0405 | 0.0469 |
| vicuna-7B-v1.1 | 0.2657 | 0.1375 | 0.1090 | 0.0970 |
| vicuna-7B-v1.5 | 0.4962 | 0.1853 | 0.1489 | 0.1213 |
| Arxiv-std + PubMed-std | 0.6390 | 0.2652 | 0.1501 | 0.0936 |
| Arxiv-cot + PubMed-cot | 0.6476 | 0.2854 | 0.1647 | 0.1326 |
| Arxiv-mix + PubMed-mix | 0.6139 | 0.2772 | 0.1544 | 0.1048 |
| Arxiv-std + PubMed-std + Link | 0.5931 | 0.2238 | $\mathbf{0 . 1 8 4 7}$ | $\mathbf{0 . 1 5 7 9}$ |
| Arxiv-mix + Pubmed-mix + Link | $\mathbf{0 . 6 8 7 4}$ | $\mathbf{0 . 3 7 6 1}$ | 0.1836 | 0.1494 |

Table 3: Performance comparison of various instruction mixtures for link prediction on PubMed.

| Dataset | PubMed |  |
| :---: | :---: | :---: |
| Model | AUC | AP |
| MLP | 0.5583 | 0.5833 |
| GAT | 0.5606 | 0.6373 |
| GraphSAGE | 0.5041 | 0.5813 |
| RevGNN | 0.4538 | 0.5083 |
| Node2Vec | 0.6535 | 0.6885 |
| w/o Link | 0.5010 | 0.5005 |
| only Link | 0.6704 | 0.6087 |
| Arxiv-std + PubMed-std + Link | $\mathbf{0 . 8 2 4 6}$ | $\mathbf{0 . 8 0 2 6}$ |
| Arxiv-mix + PubMed-mix + Link | 0.6451 | 0.5886 |

their performance tends to be mediocre when applied to complex tasks like the Cora dataset with 70 classes. By leveraging the powerful reasoning capabilities of the closed-source model (GPT-3.5) through COT distillation, our model can integrate this knowledge and significantly enhance its performance on complex graph tasks.

### 4.3 Generalization Ability Investigation (RQ2)

In this subsection, we explore the generalization ability of our model by incorporating more instruction data to fine-tune the LLM for effectively handling various types of tasks. Our main results and experimental observations are presented as follows:

More Data Boost Model Transfer Ability. In our preliminary investigation, we examine the influence of data quantity on the transfer capability of our GraphGPT, as illustrated in the "(Arxiv + PubMed)-Cora" column of Table 1. In this experiment, we train models using a combination of the Arxiv and PubMed datasets and perform zero-shot testing on the Cora dataset. The results reveal that by incorporating a relatively smaller PubMed dataset (with 20,000+ items) alongside Arxiv, our GraphGPT exhibits a significant improvement in transfer performance on Cora. In contrast, the transfer performance of GNN-based models, trained separately on Arxiv and PubMed, actually deteriorates.

More Data Yet No Forgetting. We further validate the performance of the combined Arxiv and PubMed instruction data on the original Arxiv data, as demonstrated in the "(Arxiv + PubMed)Arxiv" column in Table 1. The results indicate that most traditional
Table 4: Module ablation study under both supervised and zero-shot settings to analyze the individual contributions.

| Dataset | Arxiv-Arxiv |  | Arxiv-PubMed |  | Arxiv-Cora |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Variant | Acc | Mac-F1 | Acc | Mac-F1 | Acc | Mac-F1 |
| w/o GS | 0.4962 | 0.1853 | 0.6351 | 0.5231 | 0.1489 | 0.1213 |
| w/o LR | 0.5807 | 0.2462 | 0.2523 | 0.1925 | 0.0050 | 0.0016 |
| ours | $\mathbf{0 . 6 2 5 8}$ | $\mathbf{0 . 2 6 2 2}$ | $\mathbf{0 . 7 0 1 1}$ | $\mathbf{0 . 6 4 9 1}$ | $\mathbf{0 . 1 8 1 3}$ | $\mathbf{0 . 1 2 7 2}$ |

GNN-based approaches experience a significant decline in performance on Arxiv after iterative training. In contrast, our model exhibits improved performance. We attribute this phenomenon to the occurrence of catastrophic forgetting in GNN-based models, where the structural modeling competence of the model trained solely on the smaller PubMed dataset is compromised. However, our model effectively mitigates this issue through our unified graph instruction tuning paradigm. This enables our model to maintain and even enhance its performance by retaining the generalized graph structure patterns despite incorporating additional data.

Generalization for Multitasking Graph Learner. Recent studies on instruction tuning suggest that mixing different instruction tuning data can further enhance the performance of Language and Logic Models (LLMs). In this study, we ensure a consistent number of instruction entries and mix different types of instruction data, including standard instruction ("-std"), COT instruction ("-cot"), a blend of standard ( $50 \%$ ) and COT ( $50 \%$ ) instruction ("-mix"), and link prediction instruction ("Link"). The results are presented in Tables 2 and Table 3. We can observe that effective data mixture solutions can significantly improve the performance of our GraphGPT under various settings. The addition of task-specific instruction for link prediction task notably enhances the performance of our model in node classification. Interestingly, after incorporating node classification, the performance of link prediction also exceeds that of the selected best-performing existing models. After mixing the instructions of different tasks, our model demonstrates the ability to effectively handle various graph learning tasks and transfer its knowledge to other unseen datasets.

### 4.4 Module Ablation Study (RQ3)

We conduct an ablation study to investigate the individual contributions of different sub-modules of our proposed framework, and the results are reported in Table 5. The observations are as follows:

Effect of Graph Instruction Tuning. In our study, we investigate the benefit of incorporating graph structural information into LLM using the variant "w/o GS." In this variant, we directly adopt the base LLM (specifically, Vicuna-7B-v1.5) to perform node classification on three datasets, without incorporating graph structural information. The results of our study demonstrate that our model significantly outperforms the base model that lacks structural information. This indicates that our graph instruction tuning paradigm enables the LLM to understand the graph structural information more effectively. Importantly, this improvement in performance was achieved without altering the original parameters of the LLM. Instead, it was solely accomplished through our lightweight alignment projector, which aligns graph tokens and natural language tokens through the 1-linear projection operation.

Table 5: Study on the time and space efficiency of our GraphGPT during both the training and inference stages.

| Variants | Training Time | Tuned Parameters | GPU Occupy |
| :--- | :--- | :--- | :--- |
| Stage-1-tune | OOM | $6,607,884,288$ | OOM |
| Stage-1-freeze | $22: 53: 33$ | $131,612,672$ | 39517.75 |
| improvement | - | $\downarrow \times 50.21$ | - |
| Stage-2-tune | OOM | $6,607,884,288$ | OOM |
| Stage-2-freeze | $03: 44: 35$ | $131,612,672$ | 38961.75 |
| improvement | - | $\downarrow \times 50.21$ | - |

![](https://cdn.mathpix.com/cropped/2024_06_04_53b1737732f2b85e5f4ag-08.jpg?height=263&width=835&top_left_y=622&top_left_x=187)

Figure 5: Inference efficiency study of our GraphGPT.

Effect of LLM-enhanced Semantic Reasoning. We conduct further investigations to assess the influence of the LLM's reasoning ability in our GraphGPT by performing supervised and zero-shot predictions using only the default graph encoders. This variant is denoted as "w/o LR". The results of our study indicate that our GraphGPT, which integrates the LLM, significantly enhances the performance of graph encoder, especially in the zero-shot setting. This suggests that the rich semantic information injected by the LLM provides a substantial gain in performance.

### 4.5 Model Efficiency Study (RQ4)

The study aims to assess the computational efficiency of our model during both the model training and inference stages.

Training Efficiency with Graph Instruction Tuning. Our instruction tuning framework follows a two-stage process where the parameters of both the LLM and the graph encoder are frozen, and only the graph-text alignment projector is tuned. We conduct a comparison between freezing and tuning the LLM parameters in a 4-card 40G Nvidia A100 environment, denoted by "-freeze" and "-tune" respectively. The study analyze the time and space efficiency in terms of training time, the number of tuned parameters, and GPU occupancy (MiB per GPU). Under the same experimental conditions, when tuning LLM parameters, we encounter Out of Memory (OOM) errors even with a batch size of 1 . However, by utilizing our tuning strategy, the training process remains stable even with a batch size of 2 . Moreover, the number of tuned parameters decreases by more than 50 times compared to the freezing stage.

Model Inference Efficiency. In our exploration, we assess the inference speed and accuracy of our GraphGPT by comparing it with baichuan-7B, vicuna-7B-v1.1, and vicuna-7B-v1.5 LLMs. Using a single 40G Nvidia A100, we measure inference time (seconds per response) on the Arxiv and Cora COT instruction datasets, as shown in Figure 5. Our graph LLM demonstrates superior efficiency and accuracy. Lower inference time doesn't necessarily mean better performance: baichuan-7B yields quick but often incorrect or irrelevant answers, while vicuna-7B-v1.1 and vicuna-7B-v1.5 require longer, complex reasoning steps for better answers. In contrast, our model achieves accurate predictions through a brief reasoning process, enhancing inference efficiency.

### 4.6 Model Case Study (RQ5)

We conduct a detailed analysis of our model's performance in downstream graph learning tasks compared to traditional LLMs using different types of instructions. We evaluate ChatGPT and our GraphGPT using Arxiv data, with prompts based on node content, node content with text-based graph structure, and our designed graph instruction. The results, shown in Table 6, clearly demonstrate that despite its massive parameter count (over 200B), ChatGPT struggles to make accurate predictions solely based on node text information or node content with text-based graph structure. This challenge is particularly evident when dealing with papers that have cross-disciplinary characteristics, as seen in the example of research domains in machine learning and hardware architecture. In contrast, our GraphGPT consistently provides accurate predictions and reasonable explanations. This is because our GraphGPT incorporates a subgraph structure with 103 nodes, allowing it to extract rich structural information from neighboring nodes' citation relationships, leading to accurate predictions.

Furthermore, we believe that our approach of using graph tokens to represent the graph structure as input to the LLM is more efficient than the natural language solution. In the case of a 103-node subgraph, our GraphGPT only requires 750 tokens to be fed into the LLM, while the text-based method requires 4649 tokens. This significant reduction in token consumption translates to a substantial decrease in training and inference resource requirements.

## 5 RELATED WORK

Self-supervised Learning and Pre-training on Graphs. To enhance the robustness of graph models, self-supervised learning (SSL) has been introduced as a powerful technique [13, 16, 24]. It allows GNNs to learn meaningful graph representations without heavily relying on labeled data. The core idea behind self-supervised learning in graph models is to design pretext tasks that leverage the graph's intrinsic properties to generate additional supervision signals [52]. SSL-enhanced graph learning methods can be broadly classified into two main paradigms: contrastive SSL and generative SSL. In particular, i) Contrastive SSL focuses on learning representations by contrasting positive and negative samples. Notable methods in this domain include GraphCL [59] and GCA [67]. Recent advancements in contrastive SSL include automated contrastive augmentation (i.e., JOAO [58], AdaGCL [15]), Hyperbolic-Euclidean dual space contrasting (e.g., DSGC [56]), or community-aware contrastive learning (e.g., gCooL [20]). ii) Generative SSL, on the other hand, focuses on generating realistic samples that resemble the original graph structures. Recent advancements in this line include GraphMAE [10, 11] for feature masking, and S2GAE [35], AutoCF [53] for reconstructing masked edges as SSL tasks.

Prompt-Tuning for Graph Neural Networks. Recent efforts in enhancing the generalization capability of graph neural networks (GNNs) have focused on training GNN models in a self-supervised manner, followed by fine-tuning on specific downstream tasks using prompt-tuning techniques [64]. For example, GPPT [33] is a transfer learning paradigm, where GNNs are first pre-trained on masked edge prediction and then prompted with token pairs for downstream node classification. GraphPrompt [26] aims to handle downstream tasks by integrating pre-training and downstream tasks

Table 6: Comparison of prediction results between our GraphGPT and ChatGPT.

Title: TiM-DNN: Ternary in-Memory accelerator for Deep Neural Networks. (with $\mathbf{1 0 2}$ first-order and second-order neighbors)

Ground-Truth Category: cs.LG, Machine Learning

ChatGPT (Node Content Only): cs.AR, cs.AI, cs.SY, cs.ET, cs.NE. The paper presents a hardware accelerator for deep neural networks (DNNs) that is specifically designed to execute ternary DNNs, which represent weights and activations by signed ternary values. Therefore, the most likely category for this paper is cs.AR (Hardware Architecture), which focuses on the design and analysis of hardware and systems for AI and machine learning applications. The paper also involves the optimization of compute and storage requirements of DNNs, which falls under the category of cs.AI (Artificial Intelligence). Additionally, the paper proposes a programmable, in-memory accelerator, which is a type of embedded system, and therefore, it could be categorized under cs.SY (Systems and Control). Furthermore, the paper discusses the use of ternary values for DNNs, which is an emerging technique to optimize the compute and storage requirements of DNNs, and therefore, it could be categorized under cs.ET (Emerging Technologies). Lastly, the paper proposes a new bit-cell called Ternary Processing Cells (TPCs), which could be categorized under cs.NE (Neural and Evolutionary Computing).

ChatGPT (Node Content and Text-based Graph Structure): Based on the title "TiM-DNN: Ternary In-Memory Accelerator for Deep Neural Networks", the paper is likely to belong to the following arXiv CS sub-categories: 1. cs.AR (Hardware Architecture): The paper proposes a hardware accelerator, TiM-DNN, specifically designed for executing ternary deep neural networks. This falls under the domain of hardware architecture research. 2. cs.NE (Neural and Evolutionary Computing): The paper focuses on accelerating deep neural networks, which is a topic within the field of neural and evolutionary computing. 3. cs.CV (Computer Vision and Pattern Recognition): Deep neural networks are widely used in computer vision tasks, and the TiM-DNN accelerator could potentially be applied to accelerate computer vision algorithms. 4. cs.SY (Systems and Control): The TiM-DNN accelerator is a system-level solution for accelerating deep neural networks, so it could be categorized under systems and control research. 5. cs.DC (Distributed, Parallel, and Cluster Computing): While the paper does not explicitly mention distributed or parallel computing, the TiM-DNN accelerator is designed to optimize the compute and storage requirements of deep neural networks, which are often executed on distributed or parallel computing systems.

GraphGPT: Based on the title and abstract of the paper, we can identify the following arXiv CS sub-categories that are most likely to be relevant: 1. cs.LG (Machine Learning): This sub-category is relevant because the paper discusses the use of ternary DNNs for language modeling and image classification, which are both tasks that involve learning. 2. cs.AI (Artificial Intelligence): This sub-category is relevant because the paper proposes a new accelerator for DNNs, which is designed to optimize the compute and storage requirements of complex DNNs. 3. cs.NE (Numerical Analysis): This sub-category is relevant because the paper discusses the use of ternary DNNs, which involve signed ternary values, and the use of TiM tiles, which are specialized memory arrays that perform massively parallel signed vector-matrix multiplications on ternary values. 4. cs.SY (Systems and Control Theory): This sub-category is relevant because the paper discusses the implementation of TiM-DNN in 32nm technology using an architectural simulator calibrated with SPICE simulations and RTL synthesis. 5. cs.AR (Hardware Architecture): This sub-category is relevant because the paper proposes a new accelerator for DNNs, which

is designed to be programmable and in-memory.

into a unified task template. Additionally, Sun et al. [34] presents a unified prompt format, reformulates tasks to the graph level, and incorporates meta-learning techniques to improve multi-task performance in graph prompting. Despite these advances, these methods still require further fine-tuning that relies on supervision labels from downstream tasks to ensure accurate learning. In contrast, this work addresses this limitation by introducing a foundational graph model that tackles the more challenging task of zero-shot graph learning. By eliminating the need for label inputs from downstream tasks, this approach allows for a more general and flexible graph learning paradigm in real-world scenarios.

large Language Models. In recent years, LLMs (e.g., ChatGPT [29] and Claude [1]) have gained widespread attention for their remarkable capabilities in various NLP tasks [18, 46]. Based on these unique capabilities of LLMs, many tuning-free prompting techniques have been explored to enhance their generative abilities, such as incontext learning [28] and Chain-of-Thought [47, 57]. With the rise of open-source LLMs, such as Llama [36, 37], ChatGLM [62], and Baichuan [54], technologies for aligning pre-trained LLMs to different specific tasks and human feedback have been proposed, making private LLMs in specific domains possible [19, 44, 45].

While there have been successful attempts to align LLMs with visual information, such as multimodal LLMs [23, 66], the alignment of LLMs with graph structures remains largely unexplored.
This research addresses this gap by introducing a dual-stage graph instruction tuning paradigm that effectively aligns the language capacity of LLMs with graph learning. Previous studies $[2,5]$ have attempted to incorporate graph information into LLMs using natural language, but they have faced challenges in handling complex graph structures and achieving a deep understanding of graphs due to the limitations of relying solely on text-based prompts.

## 6 CONCLUSION

This work presents an effective and scalable graph large language model, aims at improving the generalization capabilities of graph models. The proposed framework, GraphGPT, injects graph domainspecific structural knowledge into the LLM through a dual-stage graph instruction tuning paradigm. By leveraging a simple yet effective graph-text alignment projector, we enable LLMs to comprehend and interpret the structural components of graphs. Extensive evaluations across different settings demonstrate the effectiveness of our method in both supervised and zero-shot graph learning scenarios. Furthermore, the model exhibits strong generalization abilities, allowing it to handle diverse downstream datasets and tasks without suffering from catastrophic forgetting. A potential avenue for future investigation is exploring pruning techniques to compress redundant or less important parameters of LLM, thereby reducing the overall model size while preserving its performance.

## REFERENCES

[1] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al. 2022. Constitutional AI: Harmlessness from AI Feedback. CoRR abs/2212.08073 (2022).

[2] Zhikai Chen, Haitao Mao, Hang Li, et al. 2023. Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs. CoRR abs/2307.03393 (2023).

[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT (1). Association for Computational Linguistics, 4171-4186.

[4] Yushun Dong, Ninghao Liu, Brian Jalaian, et al. 2022. EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks. In WWW. ACM, 1259-1269.

[5] Jiayan Guo, Lun Du, and Hengyu Liu. 2023. GPT4Graph: Can Large Language Models Understand Graph Structured Data ? An Empirical Evaluation and Benchmarking. CoRR abs/2305.15066 (2023)

[6] Zhichun Guo, Kehan Guo, Bozhao Nan, Yijun Tian, Roshni G. Iyer, et al. 2023 Graph-based Molecular Representation Learning. In I7CAI. 6638-6646.

[7] William L. Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In NeurIPS. 1024-1034.

[8] Xiaoxin He, Xavier Bresson, et al. 2023. Explanations as Features: LLM-Based Features for Text-Attributed Graphs. CoRR abs/2305.19523 (2023)

[9] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In SIGIR. ACM, 639-648

[10] Zhenyu Hou, Yufei He, Yukuo Cen, Xiao Liu, et al. 2023. GraphMAE2: A DecodingEnhanced Masked Self-Supervised Graph Learner. In WWW. 737-746

[11] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Jie Tang, et al. 2022. Graphmae Self-supervised masked graph autoencoders. In KDD. 594-604.

[12] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, et al. 2020. Open Graph Benchmark: Datasets for Machine Learning on Graphs. In NeurIPS.

[13] Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. 2020 Gpt-gnn: Generative pre-training of graph neural networks. In KDD. 1857-1867.

[14] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous Graph Transformer. In WWW. ACM / IW3C2, 2704-2710.

[15] Yangqin Jiang, Chao Huang, and Lianghao Huang. 2023. Adaptive graph contrastive learning for recommendation. In $K D D$. 4252-4261.

16] Baoyu Jing, Chanyoung Park, and Hanghang Tong. 2021. Hdmi: High-order deep multiplex infomax. In WWW. 2414-2424

[17] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR (Poster). OpenReview.net.

[18] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large Language Models are Zero-Shot Reasoners. In NeurIPS.

[19] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, et al 2023. RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. CoRR abs/2309.00267 (2023).

[20] Bolian Li, Baoyu Jing, and Hanghang Tong. 2022. Graph communal contrastive learning. In WWW. 1203-1213.

21] Guohao Li, Matthias Müller, Bernard Ghanem, and Vladlen Koltun. 2021. Training Graph Neural Networks with 1000 Layers. In ICML. 6437-6449.

[22] Mingkai Lin, Wenzhong Li, Ding Li, Yizhou Chen, and Sanglu Lu. 2022. ResourceEfficient Training for Large Graph Convolutional Networks with Label-Centric Cumulative Sampling. In $W W W$. ACM, 1170-1180.

[23] Haotian Liu, Chunyuan Li, et al. 2023. Visual Instruction Tuning.

[24] Yixin Liu, Ming Jin, Shirui Pan, Chuan Zhou, Yu Zheng, Feng Xia, and S Yu Philip. 2022. Graph self-supervised learning: A survey. TKDE 35, 6 (2022), 5879-5900

[25] Yunchao Liu, Yu Wang, Oanh Vu, Rocco Moretti, et al. 2023. Interpretable Chirality-Aware Graph Neural Network for Quantitative Structure Activity Rela tionship Modeling in Drug Discovery. In AAAI. 14356-14364

[26] Zemin Liu, Xingtong Yu, et al. 2023. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In WWW. 417-428.

[27] Xiaojun Ma, Qin Chen, et al. 2022. Meta-Weight Graph Neural Network: Push the Limits Beyond Global Homophily. In WWW. ACM, 1270-1280.

[28] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the Role of Demonstrations What Makes In-Context Learning Work?. In EMNLP. 11048-11064.

[29] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning (ICML). PMLR, 8748-8763.

[31] Zezhi Shao et al. 2022. Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting. In KDD. ACM, 1567-1577.

[32] Kumar Shridhar, Alessandro Stolfo, and Mrinmaya Sachan. 2023. Distilling Reasoning Capabilities into Smaller Language Models. In ACL. 7059-7073.

[33] Mingchen Sun, Kaixiong Zhou, et al. 2022. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In KDD. 1717-1727.

[34] Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, and Jihong Guan. 2023. All in One: Multi-Task Prompting for Graph Neural Networks. In $K D D$.
[35] Qiaoyu Tan, Ninghao Liu, Xiao Huang, Soo-Hyun Choi, Li Li, Rui Chen, and Xia Hu. 2023. S2GAE: Self-Supervised Graph Autoencoders are Generalizable Learners with Graph Masking. In WSDM. 787-795.

[36] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, et al. 2023. LLaMA: Open and Efficient Foundation Language Models. CoRR abs/2302.13971 (2023).

[37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, et al. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. CoRR abs/2307.09288 (2023)

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al. 2017. Attention is all you need. In NeurIPS, Vol. 30

[39] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, et al. 2018. Graph Attention Networks. In ICLR (Poster). OpenReview.net

[40] Petar Velickovic, William Fedus, William L. Hamilton, Pietro Liò, et al. 2019. Deep Graph Infomax. In ICLR (Poster). OpenReview.net

[41] Kuansan Wang, Zhihong Shen, et al. 2020. Microsoft Academic Graph: When experts are not enough. Quant. Sci. Stud. 1, 1 (2020), 396-413.

[42] Xiang Wang, Tinglin Huang, Dingxian Wang, et al. 2021. Learning Intents behind Interactions with Knowledge Graph for Recommendation. In WWW. 878-887.

[43] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, et al. 2019. Heterogeneous Graph Attention Network. In WWW. ACM, 2022-2032.

[44] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, et al. 2023. How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources. CoRR abs/2306.04751 (2023).

[45] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Danie Khashabi, and Hannaneh Hajishirzi. 2023. Self-Instruct: Aligning Language Models with Self-Generated Instructions. In ACL. 13484-13508

[46] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Jeff Dean, William Fedus, et al. 2022. Emergen Abilities of Large Language Models. Trans. Mach. Learn. Res. 2022 (2022).

[47] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS

[48] Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2023. LLMRec: Large Language Models with Graph Augmentation for Recommendation. CoRR abs/2311.00423 (2023)

[49] Zhihao Wen and Yuan Fang. 2023. Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting. In SIGIR.

[50] Qitian Wu, Chenxiao Yang, et al. 2023. DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion. In ICLR.

[51] Qitian Wu, Wentao Zhao, et al. 2023. NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification. CoRR abs/2306.08385 (2023).

[52] Jun Xia, Lirong Wu, Jintao Chen, et al. 2022. Simgrace: A simple framework for graph contrastive learning without data augmentation. In WWW. 1070-1079.

[53] Lianghao Xia, Chao Huang, Tao Yu, Ben Kao, et al. 2023. Automated SelfSupervised Learning for Recommendation. In WWW. 992-1002.

[54] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, et al. 2023. Baichuan 2: Open Large-scale Language Models. CoRR abs/2309.10305 (2023).

[55] Chenxiao Yang, Qitian Wu, and Junchi Yan. 2022. Geometric Knowledge Distillation: Topology Compression for Graph Neural Networks. In NeurIPS.

[56] Haoran Yang, Hongxu Chen, Shirui Pan, Lin Li, Philip S Yu, and Guandong Xu. 2022. Dual space graph contrastive learning. In WWW. 1238-1247.

[57] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. CoRR abs/2305.10601 (2023).

[58] Yuning You, Tianlong Chen, Yang Shen, and Zhangyang Wang. 2021. Graph contrastive learning automated. In ICML. PMLR, 12121-12132.

[59] Yuning You, Tianlong Chen, Yongduo Sui, et al. 2020. Graph contrastive learning with augmentations. In NeurIPS, Vol. 33. 5812-5823.

[60] Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J. Kim. 2019. Graph Transformer Networks. In NeurIPS. 11960-11970.

[61] Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J Kim. 2019. Graph transformer networks. In NeurIPS, Vol. 32.

[62] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, et al. 2023. GLM-130B: An Open Bilingual Pre-trained Model. In ICLR.

[63] Shichang Zhang, Yozen Liu, Yizhou Sun, and Neil Shah. 2022. Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation. In ICLR

[64] Wen Zhang, Yushan Zhu, Mingyang Chen, et al. 2023. Structure Pretraining and Prompt Tuning for Knowledge Graph Transfer. In WWW. 2581-2590.

[65] Yanfu Zhang et al. 2022. Robust Self-Supervised Structural Graph Neural Network for Social Network Prediction. In WWW. ACM, 1352-1361.

[66] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint arXiv:2304.10592 (2023).

[67] Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, and Liang Wang. 2021. Graph contrastive learning with adaptive augmentation. In WWW. 2069-2080.


[^0]:    "Chao Huang is the Corresponding Author.

    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org

    SIGIR'24, July 14-18, 2024, Washington, DC, USA

    (c) 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.

    ACM ISBN 979-8-4007-0431-4/24/07...\$15.00

    https://doi.org/10.1145/3626772.3657775

[^1]:    Graph Information: <graph>: Central Node: 68442, Edge index: [ [...src node...], [...dst node...]], Node list: [...] Graph Matching Human Question: Given a sequence of graph tokens <graph> that constitute a subgraph of a citation graph, Human Question: Given a sequence of graph tokens <graph> that constitute a sub
    2. ..., please reorder the list of papers according to the order of graph tokens.

    GraphGPT Response: Based on the given graph tokens and the list of paper titles, we obtain the matching of graph tokens and papers: Graph token 1 corresponds to smt based induction methods for timed systems. Graph token 2 corresponds to ...

    Graph Information: <graph>: Central Node: 2, Edge index: [[...src node...], [...dst node...]], Node list: [...] Node Classification Human Question: Given a citation graph: <graph> where the Oth node is the target paper, with the following information: Abstract: ... Title: ... Question: Which arXiv CS sub-category does this paper belong to? GraphGPT Response: cS.IT, CS.LG, CS.SP, CS.CV, CS.NA. The paper discusses the Restricted Isometry .... So, it is likely to belong to cs.IT.

    Graph Information: <graph>: Central Node 1: 8471, Edge index 1: [ [...src node...], [...dst node...]], Node list 1: [...] Link Prediction Graph Information: <graph>: Central Node 1: 84, Edge index 1: [...src node...], [...dst node...], Node list 1: [...]
    <graph>: Central Node 2: 19368, Edge index 2: [[...src node..], [..dst node...]], Node list 2: [.] Human Question: Given a sequence of graph tokens: <graph> that constitute a subgraph of a citation graph, sequence of graph tokens: <graph>, ... Abstract: ... Title: ..., are these two central nodes connected? Give me an answer of "yes" or "no".
    GraphGPT Response: Yes, they are connected. Based on the first paper, .... And the second paper proposes ....

</end of paper 1>


<paper 2>
# Open-TI: Open Traffic Intelligence with Augmented Language Model 

Longchao Da ${ }^{1}$, Kuanru Liou ${ }^{1}$, Tiejin Chen ${ }^{1}$, Xuesong<br>Zhou ${ }^{2}$, Xiangyong Luo $^{2}$, Yezhou Yang ${ }^{1}$ and Hua Wei ${ }^{1 \dagger}$<br>${ }^{1 *}$ School of Computing and Augmented Intelligence, Arizona<br>State University, 350 E Lemon St, Tempe, 85287, AZ, USA.<br>${ }^{2}$ School of Sustainable Engineering and the Built Environment,<br>Arizona State University, 350 E Lemon St, Tempe, 85287, AZ,<br>USA.<br>Contributing authors: longchao@asu.edu; kliou@asu.edu;<br>tchen169@asu.edu; xzhou74@asu.edu; xluo70@asu.edu;<br>yz.yang@asu.edu; hua.wei@asu.edu;<br>${ }^{\dagger}$ Corresponding Author.


#### Abstract

Transportation has greatly benefited the cities' development in the modern civilization process. Intelligent transportation, leveraging advanced computer algorithms, could further increase people's daily commuting efficiency. However, intelligent transportation, as a cross-discipline, often requires practitioners to comprehend complicated algorithms and obscure neural networks, bringing a challenge for the advanced techniques to be trusted and deployed in practical industries. Recognizing the expressiveness of the pre-trained large language models, especially the potential of being augmented with abilities to understand and execute intricate commands, we introduce Open-TI. Serving as a bridge to mitigate the industry-academic gap, Open-TI is an innovative model targeting the goal of Turing Indistinguishable Traffic Intelligence, it is augmented with the capability to harness external traffic analysis packages based on existing conversations. Marking its distinction, Open-TI is the first method capable of conducting exhaustive traffic analysis from scratch - spanning from map data acquisition to the eventual execution in complex simulations. Besides, Open-TI is able to conduct task-specific embodiment like training and adapting the traffic signal control policies (TSC), explore demand optimizations, etc.


Furthermore, we explored the viability of LLMs directly serving as control agents, by understanding the expected intentions from OpenTI, we designed an agent-to-agent communication mode to support Open-TI conveying messages to ChatZero (control agent), and then the control agent would choose from the action space to proceed the execution. We eventually provide the formal implementation structure, and the open-ended design invites further community-driven enhancements.

Keywords: Large Language Models, Traffic Simulation, Traffic Signal Control

## 1 Introduction

Traffic and Transportation have played an important role in the process of human civilization. With the development of electronic and computer techniques, intelligent transportation is casting hope to further benefit people's daily lives through optimal controlling and scheduling decisions. Transportation depicts vehicles commuting between regions, providing delivery for products and humans. The efficient modern transportation comes from joint efforts from many researchers in various directions like: map modeling [1], traffic simulation [2], schedule optimization [3], etc., and still, there are multiple ongoing challenges regarding the multi-resolution traffic simulation [4], optimal traffic signal control policies [5], dynamic demand dispatch adjustment [6], etc. More specifically, when it comes to vehicle control, the intelligent traffic signal brings hope to city-scale congestion mitigation and energy saving, multiple frontier solutions have been released on different simulators, such as SUMO [7], CityFlow [8], VISSIM [9]. These simulators and algorithms are powerful, and efficient, but hard to operate and implement, thus, introducing a gap from the research to the industry, and leading to a trustworthy problem for practitioners.

The possible solution to bridge that gap includes two steps: 1 . Unifying the simulation and analysis process by a standard ecosystem like General Modeling Network Specification (GMNS) [10] to define a common format for sharing routable road network files and is designed for multi-modal static and dynamic transportation planning and operations. 2. Building an intelligent system with self-explain abilities, which is integrated with multiple domain-specific tasks and the corresponding frontier solutions: state-of-the-art algorithms, powerful simulators, etc., and can be easily executed with sufficient explanations in an interactive way.

Transportation intelligence can be divided into 5 stages according to the development of technology as in Fig. 1, with Turing Indistinguishable as the hardest one to provide human-like decisions. With large language models becoming increasingly instrumental in aiding humans, many researchers have leveraged the LLMs to benefit transportation tasks $[11,12]$ in stage 4, and this has provided a clearer vision on stage 5 as a more Turing Indistinguishable
traffic intelligence. Large-scale pre-trained models such as Llama, GPT-3.0, and ChatGPT, are endowed with the capacity to grasp the context, dissect issues, and discern the logic connecting questions with answers, which can deliver in-depth clarifications on specific topics through a sequence of interactive dialogues. Early explorations are made by leveraging LLMs to benefit domain-specific tasks, such as: Copilot [13], DrugGPT [14], TrafficGPT [11], GraphGPT [15], etc. Due to the limitation of only tackling the context-level questions, researchers managed to augment the language model on their ability to take action and use tools, which significantly broadened the application scenarios and enhanced the beneficial impact [16]. Defined as Augmented Language Models (ALMs), this refers to 'language models (LMs) that are augmented with reasoning skills and the ability to use tools' [16]. Inspired by ALMs, we propose to design a rudiment of Turing Indistinguishable Traffic Intelligence: Open-TI, an augmented traffic agent not only able to provide conversational insights, but also able to understand human intentions, conduct intelligent traffic analysis from scratch, answer questions regarding the used techniques or tools, and provide an interpretation of the results. By this, it will be more convenient for the industrial practitioners or any stakeholders to learn about traffic and transportation studies, and cast interesting case analyses.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-03.jpg?height=441&width=881&top_left_y=1067&top_left_x=318)

Fig. 1: The 5 Stages of Development of Traffic Intelligence

In this paper, we first provide a background introduction to recent most related research directions, including traffic simulations, traffic domain-specific tasks, and augmented language models. Then we propose a pivotal augmented language agent Open-TI, that is integrated with a neat interface to operate possible tools, thus realizing the language-level operation, it is worth noting that, the Open-TI is able to conduct traffic analysis from scratch (from downloading map to provide simulation on the interested area). Second, we realized multiple domain-specific tasks like traffic signal control policy training and traffic demand optimization under the unified implementation class. Third, we design the ChatLight to realize the meta-control based on the LLM's inference ability,
by agent-agent communication, the pivotal agent will interact with humans to obtain control requirements and convey the message to the ChatZero control agent, then, the ChatZero will conduct the decision making based jointly on the observation of current traffic situation and described policies.*

In summary, the contributions of this paper are:

- Open-TI is the first paradigm to realize a thorough and stable traffic analysis from scratch: from downloading maps to simulation of interested areas.
- Open-TI is a central agent in charge of multiple task-specific executors, including traffic signal control, demand optimization, etc. It is equipped with standard API for open implementation access of the research community to enhance its capabilities.
- Open-TI could realize meta-control: convey the policy description by communicating with another control agent that directly serves as a controller to output actions based on context understanding.
- Open-TI provides sufficient explanations to any questions during the traffic analysis, from log file interpretation to results comparison, bringing convenience for traffic management and transportation strategy planning.


## 2 Background and Related Work

This section provides concepts of augmented language agents, traffic simulation, and transportation research tasks.

### 2.1 Augmented Language Models

Large Language Models (LLMs) [17-19] have boosted dramatic progress in Natural Language Processing (NLP) and are already core in several products with millions of users, such as the coding assistant Copilot [20], Bert enhanced search engine ${ }^{\dagger}$ and 2. ChatGPT and GPT4 [21]. LLMs are able to execute multiple tasks from language understanding to conditional and unconditional text generation relying on memorization [22] combined with compositionality [23] capabilities, thus opening a realistic path towards higher-bandwidth humancomputer interactions, or even benefit other research domains by its inference ability [24]. But LLMs are not held solely in the text conversation, when LLMs are equipped with the tools using abilities, it will bring more changes to people's lives. Some literature shows that by augmenting the LLMs with the tool-using ability, it could realize the advanced automation and bring the intelligence science closer to the goal of being Turing Indistinguishable, such as [25] design the API bank to execute API calls to meet human needs, [26] applied the augmented language models to the medical field to serve as a more flexible knowledge hub for doctors and patients. [27] focuses more on using existing foundation models (as a brain-like central system) and APIs of other AI models and systems (as sub-task solvers) to achieve diversified tasks in both digital[^0]and physical domains. Our work is the first to explore the augmented language agents on automatic traffic intelligence that realize a throughout traffic analysis system.

### 2.2 Traffic Simulation

The continuous growth of urban populations and the increase in vehicular traffic have accentuated the need for efficient traffic management and planning. Traffic simulation provides a good reference for planning strategies, offering insights into traffic patterns, road network efficiencies, and the potential impacts of infrastructural changes as shown in Fig. 2. The utilization of traffic simulation models facilitates the analysis of traffic behavior under various conditions without the need for costly real-world alterations.
![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-05.jpg?height=668&width=1108&top_left_y=774&top_left_x=212)

Fig. 2: The traffic and transportation simulation in cities, (a) is a real-world traffic image, (b) is the simulation of traffic flow in DTALite [28], and (c) is the city simulation from CARLA [29].

Traffic simulation has gone through various stages, from simplistic models to complex systems incorporating real-time data and predictive analytics. Early efforts in traffic simulation were concentrated on microscopic models, which simulate individual vehicle movements. Pioneering work by [30] and others laid the foundation for microscopic traffic simulation, focusing on the behavioral aspects of individual drivers. As computing power increased, mesoscopic and macroscopic models gained popularity. Mesoscopic models, like those developed by [31], provide a balance between simulating details and computational efficiency, ideal for medium-scale traffic networks. Macroscopic models, on the other hand, such as the work by [32], offer insights into broader
traffic flow trends, suitable for large-scale analysis but lacking the granularity of microscopic models.

Recent advancements have shifted focus towards data-driven and AI-based approaches. The integration of machine learning, as seen in the work of [33], has enabled more accurate predictions of traffic patterns by learning from historical data. Additionally, the incorporation of real-time data, such as that from IoT devices and traffic sensors [34], has significantly enhanced the responsiveness of traffic simulation models. Studies by [35] have demonstrated the effectiveness of real-time data in adapting traffic simulations to dynamic conditions.

With the development of autonomous vehicles (AVs), traffic simulation for safety tests and validation become important. The simulator developed by NVIDIA [36] tends to provide a more physically accurate simulation platform and Waymax [37], which is developed by autonomous vehicle company Waymo, provides a multi-agent scene simulator on hardware accelerators to empower the simulation for AVs.

In this work, we provide support to multiple simulators like SUMO [38], CityFlow [8], and DLSim [28], and further present open-implementation instruction to help integrate more advanced ones with the development of the research community.

### 2.3 Traffic Domain Specific Tasks

## Traffic Signal Control

Traffic Signal Control (TSC) is crucial for improving traffic flow, reducing congestion in modern transportation systems, and benefiting individuals and societies. Traffic signal control remains an active research topic because of the high complexity of the problem. The traffic situations are highly dynamic and require traffic signal plans to adapt to different situations, making it necessary to develop effective algorithms that can adjust to changing traffic conditions [39]. Recent advances in reinforcement learning (RL) techniques have shown superiority over traditional approaches in TSC [40]. In RL, an agent aims to learn a policy through trial and error by interacting with an environment to maximize the cumulative expected reward over time, when learning the policy, it takes the features like intersection vehicle amount, and current traffic light phase (state) as observations, and change the traffic light phases or duration times (actions), to realize the relief of traffic congestion (reward). The most evident advantage of RL is that it can directly learn how to generate adaptive signal plans by observing the feedback from the environment, so it is beneficial for traffic intelligence to integrate TSC tasks to provide planning insights.

## OD Matrix Optimization

Origin-destination (O-D) matrix estimation is a critical component in the field of transportation planning and traffic engineering and involves the creation of a matrix that represents the number of trips between various origins and

Open-TI for Intelligent Traffic Planning and Simulation

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-07.jpg?height=773&width=391&top_left_y=298&top_left_x=193)

Response and chat history
Tell me about yourself.
My name is Open-TI, I am Open Traffic Intelligence (Open-TI) with Augmented Language
Model developed by DaRL Lab to assist humans with intelligent traffic planning,
transportation simulation, or conducting automatic analysis. I have access to a set of tools
and can assist with various tasks related to traffic planning. How can I assist you?
List all the thing that you can help me in Transportation and Traffic Analysis/Planning?
I can assist you with the following tasks in transportation and traffic analysis/planning:

1. QueryAreaRange: Obtain area information of a specific point on the map.
2. ShowOnMap: Display a specific area on the map.
3. Auto_download_openstreetmap_file: Automatically download map data from
OpenStreetMap.
4. SimulateOnLibSignal: Execute traffic simulation using the LibSignal library.
5. Network Filter: Filter the road network based on specific categories (e.g., bikeable lanes,
walkable areas, railways, etc.).
6. GenerateDemand: Generate traffic demand based on downloaded map data.
7. SimulateOnDLSim: Simulate traffic using the DLSim multi-resolution traffic simulator.
8. VisualizeDemand: Visualize the generated traffic demand.
9. Log_analyzer: Analyze log files and provide insights.
10. Multilanguage understanding and response: Understand and respond in multiple
languages.

Fig. 3: The Open-TI conversation interface.

destinations in a given area over a specified time period. This matrix is fundamental for understanding travel demand patterns and is important for traffic modeling, infrastructure planning, and policy-making in urban environments. The process of O-D matrix estimation can vary in complexity depending on the size of the area being studied and the granularity of the data required. Traditional methods often involve the use of travel surveys, where individuals report their travel behavior, or the extrapolation of data from limited traffic counts [41-43]. However, these methods can be time-consuming and may not always capture the full spectrum of travel patterns due to their reliance on sampling or incomplete data.

The O-D matrix estimation has evolved in recent years, modern techniques leverage large datasets obtained from a variety of sources, such as traffic sensors, GPS devices, and mobile phone signals [44-54]. These data sources provide a more comprehensive and real-time picture of traffic flows, allowing for more accurate and dynamic O-D matrices. The use of advanced statistical models, machine learning algorithms, and data fusion techniques further enhances the precision of O-D matrix estimations, which not only provide a better understanding of current travel patterns but also enable transportation planners and engineers to predict future trends and make informed decisions for efficient traffic management and sustainable urban planning. Providing O-D matrix optimization would enhance the practicality of the simulation situation.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-08.jpg?height=463&width=1109&top_left_y=225&top_left_x=209)

Fig. 4: The overview of the Open-TI functionalities

## 3 The Architecture of Open-TI

### 3.1 Overview of Open-TI

To take a step forward to more Turing Indistinguishable Traffic Intelligence, Open-TI is equipped with human-like semantic ability and goal-oriented execution ability. Human-like semantic ability is realized by convenient conversation sessions between users and the agent, and execution ability is guaranteed by agent augmentations.

For Open-TI, primarily, a user-friendly interface is designed as shown in Fig. 3. It contains four panels: Prompt or question (top left), the user can edit input, clear text, and submit request; Hints of questions (middle left): user could click on the suggested choices to easily start a conversation; Thought and action (bottom left): this panel presents the chain of thought content from Open-TI agent; Response and chat history (right): this main panel provides multi-media feedback and execution result from Open-TI, including texts, images, path files, and browser links, etc.

The core of Open-TI mainly incorporates three modules: Analysis and Simulation, Task Specific Embodiment and Zero Control to enhance the intelligent traffic analysis and planning, as shown in Fig. 4.

First, Open-TI can manipulate and help practitioners to conduct analysis and simulation from scratch. Open-TI provides the chance for users to think of a POI (point of interest) or AOI (area of interest) and present the visualization immediately on a map, users can ask for more geology information like latitude and longitude range, and after that, the acquired information can be used to select an analysis range for further investigations like specific lane (e.g., bike lane) filtering and traffic simulation by DTALite (DLSim) [28] or SUMO[38].

Second, the Open-TI supports multiple task-specific embodiments by vague and high-level language explanations, which greatly reduce the professional background requirements of experimental exploration. For example, based on
the current road map, it allows one to conduct traffic light control exploration by either rule-based methods or deep learning policy training [55], it also could easily conduct traffic demand optimization by providing brief task descriptions.

Third, our method leverages the superior understanding capabilities of current LLMs to conduct meta-control by ChatLight agent: LLMs directly serve as a control agent, follow the understanding of the semantic description of the rules, and control the traffic light actions. This explores a higher level traffic light control mode, e.g., the traffic management department may have special requirements on the safety concerns and would like to adjust the traffic signal control policies to reduce the collision rate, only word-level description is required, and the description would be analyzed, extracted, and communicated as a message to the ChatLight agent, which could reduce the complexity of adjusting the control policy and furthermore, provide explanation for its actions by self-reflecting the LLM's own behavior.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-09.jpg?height=551&width=1131&top_left_y=845&top_left_x=196)

Fig. 5: The design framework of Open-TI

As shown in Fig. 5, the Open-TI consists of two components, the Traffic LLMs and Augmentations. When a user requirement is detected from the conversation, it will be passed to the first components of Traffic LLMs, which use the language agent to extract accurate intentions and useful information from dialogues, then, the extracted intention will be compared with tools in augmentation pool, after matching, if the agent found a possible solution, it will execute with extracted information and generate the output with multimedia data form.

### 3.2 Prompt Design

The Open-TI exploited the LLM's understanding abilities and tuned its behavior by prompt engineering, which is essential for well-organized operation.

Table 1: The details of the prompt components for queryAreaRange

| Name | Purpose | Instances |
| :---: | :---: | :---: |
| Description | Core component of the <br> prompt structure, clarifies the basic setup, <br> method, and object of each function. | You are designed to respond with <br> longitudes and latitudes information of a location <br> Humans might ask similar words of location <br> like position $=$ place $=$ location $=$ geographic info, <br> you can imagine and infer the most possible. |
| Format Restriction | Specify input format constraints, <br> significantly reducing error rates. | The format of your output longitude and <br> latitude is a query of 4 value array as |
| Example | Help LLMs understand the <br> exactly processing of the execution. | Human ask "Where is Arizona State University, <br> Tempe Campus", you need to output <br> $[-111.9431,33.4154,-111.9239,33.4280]$. |
| Reflection | Remind LLMs not to engage in <br> unnecessary tasks and ensure <br> that each process is executed accurately. | You should respond directly with what <br> you know and then stop, do not look for <br> the location or attempt to find it online. |
| Emphasis | Reinforce the function's objective, <br> significantly reducing API mismatching rates. | You have the valid tool to provide location. You <br> have a specific tool to directly query the location. |

We have designed 5 aspects of prompt structure: [Description], [Example], [Output Restriction], [Emphasis] and [Reflection]. And we verified their effectiveness in cross-task behaviors by ablation experiment. In this section, we provide details of prompt design as shown in Fig. 5. Including the purpose of the prompt and example cases. The examples in Table 1. are from the same augmentation task of queryAreaRange.

### 3.3 Execution and Augmentation List

The overall execution process is expressed in Algorithm 1. in pseudo-code. Following the same execution flow, there are different augmented tools that could help users with various requirements as presented in Table 2. The section. 4 will elaborate on the three augmentation modules in detail.

### 3.4 Standard Implementation

In this section, we formalize the API content and provide the structure of the augmentation class. In order to extend the augmentation to the border range with additional capabilities, the keys are: First, make sure the pivotal agent is aware of the functionality of the specific additional tool, so when the requirement is asked by users, the pivotal agent could pick up the most suitable method. Second, the embodiment program should function accurately and effectively, so that when the pivotal agent boots up the process, it can come to the expected result. Based on the two keys, we provide the implementation structure as in the following code. One needs to provide the descriptions on the augmentation class and implement the Execution (), this provides an easy way to understand and neat for other researchers to explore and enforce Open-TI's abilities.

Table 2: A list of augmented tools implemented in Open-TI

| Augmentation Name | Description |
| :---: | :---: |
| queryAreaRange | Obtain area information, specifically the <br> longitudes and latitudes of a point of interest on the map. |
| showOnMap | Display the location of interest <br> on the map, such as the ASU campus area. |
| autoDownloadOpenStreetMapFile | Automatically download map data <br> from OpenStreetMap for a specified area. |
| simulateOnLibsignal | Execute simulations on the <br> open-source library called Libsignal. |
| networkFilter | Filter the road network based on <br> required categories, return the file <br> path of a filtered road network <br> that emphasizes lanes of interest. |
| generateDemand | Generate demand based on <br> OpenStreetMap data. |
| simulateOnDLSim | Simulate on the DLSim <br> multi-resolution traffic simulator. |
| simulateOnSUMO | Execute the simulation given <br> arbitrary .osm data. |
| visualizeDemand | Automatically generate and display <br> visualizations of the demand file. |
| logAnalyzer | Analyze log or config files <br> and provide comparisons. |
| resultExplainer | Interpreter results to provide insights. |
| demandOptimizer | Approximate the origin-destination <br> demand to fit realistic observation. |

```
class AugmentTask:
    def __init__(self, params) -> None:
        self.params = params
        # set possible pre-defined params, e.g., folder path
    @func_prompt(name="prompt name",
        description="""detailed explanation""")
    def embody(self, target: str) -> str:
        try:
            result = Execution(target)
            # Concrete Execution Implementation
        except Error as e:
        print(e)
        return result
```

Listing 1: The Class and API Format

## 4 Sub-module Embodiment

The Open-TI is enhanced by three different modules that eventually realize the superior performance. In this section, we will elaborate on each module with examples and illustrations.

```
Algorithm 1 Open-TI Execution Process
    INPUT: msg UserInputQuery
    if Intention Needs External Tools then
        while Augmented APIs not Found do
            keywords $\leftarrow \operatorname{summarize}(\mathrm{msg})$
            $a p i \leftarrow \operatorname{search}($ keywords $)$
            if MaximumQueryTime Exceeds then
                break
            end if
        end while
        if API found then
            Params $\leftarrow$ extract_params $(\mathrm{msg})$
            while Params Not Satisfied do
                1. Retrospect Expected Form
                2. Examine User Input: $m s g$
                while Missing Info. do
                    Alert Required Info.
                end while
                Response $\leftarrow$ execute_api_call(Params)
                if MaximumQueryTime Exceeds then
                    break
                end if
            end while
            while Response not Satisfied do
                $a p i \_c a l l \leftarrow$ gen_api_call(api_doc, $\left.m s g\right)$
                Response $\leftarrow$ execute_api_call(api_call)
                if MaximumQueryTime Exceeds then
                    break
                end if
            end while
        end if
    end if
    if Response then
        $r e \leftarrow$ Construct_Response(Response)
    else
        $r e \leftarrow$ Query_Failed()
    end if
    return $r e \leftarrow$ ResponseToUser
```


### 4.1 Pivotal Agent for Transportation Analysis

In this module, analysis from scratch is realized by seamless connections between augmented tools and the pivotal operation agent. The supported external tools and packages are shown in Table 3. And when the user asks about related tasks, Open-TI will automatically match the highest probability option and process following Algorithm 1. An example of interaction is shown in Fig. 6.

Table 3: The supported external tools and packages

| Name | Functions | Versions |
| :---: | :---: | :---: |
| osm2gmns | obtain networks from OSM and convert to GMNS | V-0.7.3 |
| grid2demand | Origin-destination trans demand generate | V-0.3.6 |
| DLSim-MRM | Multi-resolution Traffic Simulation | V-0.2.11 |
| Libsignal | Multi-simulator platform for Traffic Signal Control | V-1.0.0 |

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-13.jpg?height=1628&width=1156&top_left_y=180&top_left_x=181)

Fig. 6: The demonstration pivotal agent control. (Right: The user messages, Left: The responses from Open-TI). This series of interactions shows how to query geography information of a location, how to visualize on the map, filter the interested lane types, and use the arbitrary map for automatic traffic simulation (SUMO).

### 4.2 Task-Specific Embodiment

The Open-TI is capable of realizing more general research tasks in the traffic domain. Including traffic signal control (TSC), traffic flow prediction, traffic Origin-Destination(O-D) demand optimization, etc. The architecture in Open-TI is well structured and supports extensibility for an open community. We will introduce how the Open-TI achieves the three demonstrating tasks in the following subsections.

### 4.2.1 Traffic O-D Demand Optimization Task

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-14.jpg?height=484&width=477&top_left_y=666&top_left_x=227)

(a) The visualization of O-D matrix

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-14.jpg?height=428&width=496&top_left_y=728&top_left_x=789)

(b) The demand zones split

Fig. 7: An illustration of the O-D demand optimization task in Sedona, AZ. In (a), it consists of 3 dimensions: origin zone, destination zone, and time (Hour), in each time interval, one slice describes the traffic demand information. (b) is the geography demand zone split of the interested area.

In the OD demand optimization task, the goal is to design an algorithm and learn a model that could help to output an accurate OD matrix, given the partial observation data. In Figure 7, we show an example in Sedona, AZ, USA. When asked to execute an O-D matrix optimization task, users could specify the observation data source, and traffic simulation setting, and then choose the optimization techniques to experiment with. In the example case, the given data is the 16 -hour count data at the specific observation point of a roundabout, and we asked the agent to use a genetic algorithm to conduct optimization and provide the result.

### 4.2.2 Traffic Signal Control Task

In the realization of traffic signal control embodiment, we seamlessly integrated the Libsignal [55] that could realize the cross-simulator traffic signal control over the majority of baseline methods, including the rule-based approaches

(Fixed Time and Self-organizing traffic lights - SOTL [56]) and reinforcementlearning-based approaches as shown in Fig. 8. We provide further interaction cases in the Appendix. A.1.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-15.jpg?height=334&width=1071&top_left_y=373&top_left_x=230)

Fig. 8: Current feasible simulators and algorithms for TSC tasks

### 4.3 Agent Meta Control

In the meta-control task, we essentially designed an agent-agent communication and the pivotal agent is in charge of understanding the human descriptive intention of the traffic policy, and the execution agent will take the message as instruction, process, and digest, then connect to the traffic practical control to provide a self-explainable action execution control.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-15.jpg?height=481&width=1096&top_left_y=1205&top_left_x=229)

Fig. 9: The demonstration ChatZero Meta Control.

## 5 Experiment

In this section, we conduct extensive experiments to answer the following questions:

- RQ1: How does our proposed approach compare to other state-of-the-art methods in terms of performance?
- RQ2: How do the components impact the performance of our proposed method?
- RQ3: How does ChatZero execute meta-control? How does it perform across various LLMs?

Three aspects of experiments are designed from error rates of API calls, ablation study of Augmented Languages Agents' prompt structure, and ZeroControl agent's performance on LLM agents to verify the effectiveness and stability of the proposed Open-TI. Please note that, for the RQ1 and RQ2, we develop standard Open-TI based on GPT3.5 and for RQ3, we verified on 4 different language models: Llama7B, Llama13B, GPT3.5, and GPT4.0.

### 5.1 Language Agent Analysis on the API Calls

In this section, we conduct the functionality-level experiments of API analysis and compare them with the baseline method known as TrafficGPT[11].

## Experiment Design

First, following the work of [25], we analyze three types of API call abnormal behaviors, namely 'No API Call Rate', 'API Mismatching Rate', and 'Error Raise Rate'. Both Open-TI and TrafficGPT are equipped to handle a range of tasks spanning geographical information, simulation, and traffic signal control. Although the specific functions of Open-TI and TrafficGPT are slightly different, we are still able to evaluate the overall API access stability. We adopted $T=6$ similar tasks as shown in the table 4 . to conduct the comparison by testing each task 20 times. And calculate the error rate follow the equation 1:

$$
\begin{equation*}
\text { Error Rate }=\frac{1}{T} \sum_{t=1}^{T} \frac{n_{t}^{e}}{n_{t}^{c}}=\frac{1}{T} \sum_{t=1}^{T} \frac{1}{n_{t}^{c}} \sum\left(c_{t}^{n o}, c_{t}^{\text {miss }}, c_{t}^{\text {error }}\right) \tag{1}
\end{equation*}
$$

where $n_{t}^{e}$ represents the number of error occurrences for task $t$ during total tests, $n_{c}$ denotes the number of total testing instances, (i.e., $n_{c}=20$ for this experiment), $c_{t}^{n o}$ is the sum of errors caused by the absence of API calls for task $t$ among all tests, similarly, $c_{t}^{\text {miss }}$ is the sum of mismatching error times, $c_{t}^{\text {error }}$ is the number of error raising times, and exist $n_{t}^{e}=c_{t}^{n o}+c_{t}^{\text {miss }}+c_{t}^{\text {error }}$.

For evaluation of each dimension shown in Fig. 10, denote $\rho$ as error rate, we have: $\rho_{n o}=\frac{1}{T} \sum_{t=1}^{T} \frac{c_{t}^{n o}}{n_{t}^{c}}, \rho_{m i s s}=\frac{1}{T} \sum_{t=1}^{T} \frac{c_{t}^{m i s s}}{n_{t}^{c}}$ and $\rho_{\text {error }}=\frac{1}{T} \sum_{t=1}^{T} \frac{c_{t}^{e r r o r}}{n_{t}^{c}}$.

As in the top of Table 4, which is an intersection set of functionalities in TrafficGPT and Open-TI. simulateOnSumo are commonly integrated by different implementations, showOnMap is used to query and show the map of the interested place which is equivalent to the functionality of locating and drawing the intersection on the map. logAnalyzer is designed to interpret and help the user understand the log and config files, similar to retrieving data from .XML files in TrafficGPT. visualizeTrainingCurves is for visualization of the training process which is equivalent to generating heat graphs by TrafficGPT, simulateOnLibsignal applies multiple algorithms to control traffic signal control while the Webster method is used in TrafficGPT, similarly, two

Table 4: The compared tasks is an intersection set that exists both in Open-TI and TrafficGPT. The experiment tends to design a fair comparison by identical task goals or similar difficulty.

|  | Open-TI | TrafficGPT |
| :---: | :---: | :---: |
| (1) | simulateOnSumo | Run the sumo simulation |
| (2) | showOnMap | Draw intersections on map |
| (3) | logAnalyzer | Retrieve data from the .xml files |
| (4) | visualizeTrainingCurves | Generating heat graphs |
| (5) | simulateOnLibsignal | Optimize intersections by Webster |
| (6) | resultExplainer | Compare data from the simulation |
|  | queryRangeArea | - |
|  | autoDownloadOpenStreetMapFile | - |
|  | networkFilter | - |
|  | generateDemand | - |
|  | simulateOnDLSim | - |
|  | demandOptimizer | - |

systems both provide result explanation as resultExplainer, which we take into comparison.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-17.jpg?height=477&width=621&top_left_y=1067&top_left_x=430)

Fig. 10: Language agent analysis on the API calls

## Experimental Results

The experiment results can be found in Fig. 10. The comparison is conducted on the average value over 20 times. The sum of the 3 types of error rates in Open-TI and TrafficGPT are $8.3 \%$ and $19.2 \%$, calculated by aggregation of the three types of error rates in two systems respectively.

Table 5: Error Rate of Open-TI and TrafficGPT

| Tasks |  | TrafficGPT | Open-TI |
| :---: | :---: | :---: | :---: |
| (1) | No API Call | 0.00 | 0.00 |
|  | Mismatch | 0.05 | 0.00 |
|  | Error Raise | 0.10 | 0.05 |
| (2) | No API Call | 0.05 | 0.00 |
|  | Mismatch | 0.00 | 0.00 |
|  | Error Raise | 0.00 | 0.05 |
| (3) | No API Call | 0.15 | 0.00 |
|  | Mismatch | 0.05 | 0.05 |
|  | Error Raise | 0.10 | 0.05 |
| (4) | No API Call | 0.05 | 0.00 |
|  | Mismatch | 0.10 | 0.00 |
|  | Error Raise | 0.15 | 0.10 |
| (5) | No API Call | 0.05 | 0.05 |
|  | Mismatch | 0.00 | 0.00 |
|  | Error Raise | 0.10 | 0.05 |
| (6) | No API Call | 0.10 | 0.05 |
|  | Mismatch | 0.00 | 0.00 |
|  | Error Raise | 0.10 | 0.00 |

In Fig. 10, we take the mean as the evidence to show on the bar chart as from 3 different evaluation dimensions. The x-axis presents individual dimensions of error rates. In each dimension, the two methods are compared side by side, and we can compare their percentage value according to the y-axis.

We could observe that the Open-TI effectively reduced the general API abnormal behaviors across three evaluation aspects. More obviously, Open-TI shows significantly better performance than the baseline method in terms of Error Rate, mainly because the 5 aspects of the prompt components emphasize the task, reactions, and meta-information more explicitly. We also demonstrate the detailed task level side-by-side comparison as shown in Table. 5, from the result, we could notice that Open-TI performs consistently more stable (lower error rate and lighter color) for each task than TrafficGPT, even though for task (2), Open-TI performs slightly worse in Error Raise Rate, it is because showOnMap is a more complex task consists of two consecutive sub-tasks, which are 1) identifying the geographic information and 2) request and visualize map data. To understand how each component plays a role in helping the language agent work consistently stable, we conduct further exploration in the next section.

### 5.2 Ablation Study of Open-TI Prompt Structure

In this section, we conduct the ablation study to unravel the nuanced impact of different components on the functionality of the tasks under consideration.

## Experiment Design

The ablation process is operated by the removal of each prompt component in an order of Emphasis, Reflection, Format Restriction, Example, and Description. The analysis is done across 4 tasks: queryAreaRange,

showOnMap, autoDownloadOpenStreetMapFile, and simulateOnLibsignal, which encompasses all five prompt components.

By eliminating the least important component first, followed by the removal of subsequent components in ascending order of importance. This sequential approach enables the examination of the individual influence of each component on the overall function. For each iteration, we conduct an experiment by posing 20 questions related to each function. Then the collected error rates are summarized into a comprehensive heatmap as in Fig. 11.

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-19.jpg?height=551&width=777&top_left_y=619&top_left_x=361)

Fig. 11: The ablation study of the prompt components. The x-axis from left to right shows the gradual removal of each prompt module. The y-axis shows the individual task, cell color from dark to light reflects the performance drop.

## Experimental Results

Fig. 11 shows the experimental findings related to the effect of each component. When there is no component removed, the system's performance is relatively stable and robust. When the Emphasis is removed, the accuracy suffers a slight drop across queryAreaRange to autoDownloadOpenStreetMapFile, because the agent can be confused among similar tasks or tasks with similar keywords then leading to mismatching. For example, when a human asked, "Can you help me find the map of Arizona State University?" The agent may be confused about finding the map and suppose human need the geographical information about Arizona State University. Thus, the agent mismatch autoDownloadOpenStreetMapFile question to queryAreaRange.

Then if the Reflection is further removed, simulateOnLibsignal suffers the most, by investigation, we found that it is mainly caused by entering incorrect keywords in questions. However, the agent can memorize the answers to previous questions, preventing error escalation and subsequently providing

Table 6: Examples of correct and wrong API usage of Augmented Language Model.

| Error Name | Question | Tool | Analysis |
| :---: | :---: | :---: | :---: |
| No API Call | Can you assist <br> me to download <br> the OSM file for <br> the Sydney <br> Harbour Bridge in Australia? | $\checkmark:$ It shows the path to the OSM file <br> for Sydney Harbour Bridge. | It finds a similarly named <br> OSM file and shows the wrong answer. |
|  |  | $x$ : It shows the path to the OSM file <br> for the State of Liberty in New York. |  |
| API Mismatch | I'm interested <br> in the OSM file <br> for Dubai Mall; <br> Can you guide <br> me on that? | $\checkmark:$ It shows the path to the OSM file <br> for Dubai Mall. | The question should match <br> autoDownloadOpenStreetMapFile <br> but it mismatches with queryAreaRange |
|  |  | $X$ : The longitude and latitude of Dubai <br> Mall is $[55.274,25.194,55.282,25.199]$. |  |
| Error Raise | Can you provide <br> the OSM file for <br> CN Tower in <br> Toronto? | $\checkmark:$ It shows the path to the OSM file <br> for CN Tower in Toronto. | It mistakenly inputs wrong <br> information into the showOnMap. |
|  |  | $x$ : Error raise and not keep executing. |  |

inaccurate responses to users. Therefore, the Reflection aims to address the problem of incorrect keywords, ensuring that users input the correct keywords in their questions.

Meanwhile, Format Restriction mainly affects showOnMap and simulateOnLibsignal. When Format Restriction is removed, their accuracy rates decrease by $25 \%$ and $55 \%$, respectively. This is due to the role of Format Restriction in input format limit control; Entering incorrect information into a project can lead to errors. As a result, Format Restriction significantly affects showOnMap and autoDownloadOpenStreetMapFile performance.

The component Example plays a significant role in helping language agents understand the task, once removed, the accuracy rates decrease by $45 \%$ and $55 \%$, respectively. Furthermore, since LLMs can sometimes overlook the middle steps in implementation, the Example helps to improve its stability. Thus, the Example is significant in both showOnMap and autoDownloadOpenStreetMapFile. For the case of overlook behavior, e.g., if I want to download the OSM file of Arizona State University and the agent doesn't have the geographical information for the target place, the agent should first call the queryAreaRange function. After obtaining the geographical information, it should then input this data into the autoDownloadOpenStreetMapFile function to get the correct answer. We define this embodiment as multi-step sequential actions. However, if the Example is removed, the agent might skip calling autoDownloadOpenStreetMapFile, resulting in an incorrect answer.

There is no clear influence from each component's removal on queryAreaRange. This is because there are solely two steps in the task, match the API request and post to the online map server for response, the format of the request body is well-defined, and only execution is needed, which makes it a simple procedure, not sensitive to the prompt component.

In conclusion, all components contribute to enhancing the performance of Open-TI. Particularly, the impact of Format Restriction and examples is notably significant. This reveals that careful attention should be paid to Format Restriction and examples when seeking to enhance the executionrelated prompt.

### 5.3 Meta Agent Control Analysis

## Experiment Design

In this section, we conduct experiments to verify the effectiveness of the meta agent control, at the same time, we realized 4 versions of ChatZero on the most well-known LLMs which are: Llama2-7b, Llama2-13b, ChatGPT (GPT-3.5) and GPT-4.0. During the test, the Open-TI pivotal agent will ask the 4 versions of ChatZero to perform traffic signal control tasks across 4 different traffic configurations using the realistic road map data in Hangzhou city. Each traffic control task is conducted 5 times and the reported results are the mean values, following the literature in TSC [57], the evaluation metrics are as follows:

- Average Travel Time (ATT) is the average time $t$ it takes for a vehicle to travel through a specific section of a road network. For a control policy, the smaller $A T T$, the better.
- Throughput (TP) is the number of vehicles that reached their destinations given amount of time. The larger $T P$, the better.
- Reward is an RL term that measures the return by taking action $a_{t}$ under state $s_{t}$. We use the total number of waiting vehicles as the reward, aligned with Preliminaries. The larger the reward, the fewer waiting vehicles, the better.
- Queue is the number of vehicles waiting to pass through a certain intersection in the road network. Smaller is better.
- Delay is the average delay per vehicle in seconds and measures the amount of time that a vehicle spends waiting in the network. Smaller is better.


## Experimental Results

The experiment results are shown in the Fig. 12. From the throughput (TP) and average travel time (ATT), we could find out that ChatZero by GPT 4.0 provides the best control results with the overall highest TP and lowest ATT, reflecting its superior ability to understand the policy description and conduct proper control. Other metrics evaluation are shown in Fig. 13.

## 6 Conclusion

In this paper, we propose Open-TI, an intelligent traffic analysis agent leveraging the large language models' contextual abilities and augmenting them with traffic domain-specific tools, which could provide more than questions consult, but also actual practice on processing raw map data, executing the simulation, training traffic light control policies, and conducting demand optimization, etc. We also explored the meta-control of traffic light by an agent-agent

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=298&width=816&top_left_y=174&top_left_x=360)

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=226&width=368&top_left_y=183&top_left_x=372)

(a) Throughput

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=212&width=355&top_left_y=197&top_left_x=789)

(b) Aevrage Travel Time

Fig. 12: The ChatZero performance in TP and ATT across the 4 LLMs. Each LLM is tested to control the traffic signal in 4 different configs of road situation

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=303&width=1202&top_left_y=614&top_left_x=176)

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=215&width=364&top_left_y=629&top_left_x=195)

(a) Queue Length

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=217&width=352&top_left_y=628&top_left_x=608)

(b) Aevrage Delay

![](https://cdn.mathpix.com/cropped/2024_06_04_98b2abe2f69ba04e69cag-22.jpg?height=217&width=350&top_left_y=628&top_left_x=1009)

(c) Reward

Fig. 13: The ChatZero performance in TP and ATT across the 4 LLMs. Each LLM is tested to control the traffic signal in 4 different configs of road situation

communication scheme named ChatZero, which could cast insight into the self-explainable control practice.

We also found that in sequential actions practice, it is easier to occur the API mismatch, for future work, it is important to focus on improving the accuracy of the multi-step action embodiment. We have provided the code base and an explicit implementation structure for the research community to enhance and expand the Open-TI's intelligence.

## References

[1] Yukawa, S., Kikuchi, M.: Coupled-map modeling of one-dimensional traffic flow. Journal of the Physical Society of Japan 64(1), 35-38 (1995)

[2] Chao, Q., Bi, H., Li, W., Mao, T., Wang, Z., Lin, M.C., Deng, Z.: A survey on visual traffic simulation: Models, evaluations, and applications in autonomous driving. In: Computer Graphics Forum, vol. 39, pp. 287308 (2020). Wiley Online Library

[3] Dai, Z., Liu, X.C., Chen, X., Ma, X.: Joint optimization of scheduling and capacity for mixed traffic with autonomous and human-driven buses: A dynamic programming approach. Transportation Research Part C: Emerging Technologies 114, 598-619 (2020)

[4] Zhou, X.S., Cheng, Q., Wu, X., Li, P., Belezamo, B., Lu, J., Abbasi,

M.: A meso-to-macro cross-resolution performance approach for connecting polynomial arrival queue model to volume-delay function with inflow demand-to-capacity ratio. Multimodal Transportation 1(2), 100017 (2022)

[5] Wei, H., Xu, N., Zhang, H., Zheng, G., Zang, X., Chen, C., Zhang, W., Zhu, Y., Xu, K., Li, Z.: Colight: Learning network-level cooperation for traffic signal control. In: Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 1913-1922 (2019)

[6] Osorio, C.: High-dimensional offline origin-destination (od) demand calibration for stochastic traffic simulators of large-scale road networks. Transportation Research Part B: Methodological 124, 18-43 (2019)

[7] Lopez, P.A., Behrisch, M., Bieker-Walz, L., Erdmann, J., Flötteröd, Y.-P., Hilbrich, R., Lücken, L., Rummel, J., Wagner, P., Wießner, E.: Microscopic traffic simulation using sumo. In: 2018 21st International Conference on Intelligent Transportation Systems (ITSC), pp. 2575-2582 (2018). IEEE

[8] Zhang, H., Feng, S., Liu, C., Ding, Y., Zhu, Y., Zhou, Z., Zhang, W., Yu, Y., Jin, H., Li, Z.: Cityflow: A multi-agent reinforcement learning environment for large scale city traffic scenario. In: The World Wide Web Conference, pp. 3620-3624 (2019)

[9] Fellendorf, M., Vortisch, P.: Microscopic traffic flow simulator vissim. Fundamentals of traffic simulation, 63-93 (2010)

[10] Lu, J., Zhou, X.S.: Virtual track networks: A hierarchical modeling framework and open-source tools for simplified and efficient connected and automated mobility (cam) system design based on general modeling network specification (gmns). Transportation Research Part C: Emerging Technologies 153, 104223 (2023)

[11] Zhang, S., Fu, D., Zhang, Z., Yu, B., Cai, P.: Trafficgpt: Viewing, processing and interacting with traffic foundation models. arXiv preprint arXiv:2309.06719 (2023)

[12] de Zarzà, I., de Curtò, J., Roig, G., Calafate, C.T.: Llm multimodal traffic accident forecasting. Sensors 23(22), 9225 (2023)

[13] Vaithilingam, P., Zhang, T., Glassman, E.L.: Expectation vs. experience: Evaluating the usability of code generation tools powered by large language models. In: Chi Conference on Human Factors in Computing Systems Extended Abstracts, pp. 1-7 (2022)

[14] Li, Y., Gao, C., Song, X., Wang, X., Xu, Y., Han, S.: Druggpt: A gptbased strategy for designing potential ligands targeting specific proteins. bioRxiv, 2023-06 (2023)

[15] Tang, J., Yang, Y., Wei, W., Shi, L., Su, L., Cheng, S., Yin, D., Huang, C.: Graphgpt: Graph instruction tuning for large language models. arXiv preprint arXiv:2310.13023 (2023)

[16] Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., et al.: Augmented language models: a survey. arXiv preprint arXiv:2302.07842 (2023)

[17] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018)

[18] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877-1901 (2020)

[19] Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. Journal of Machine Learning Research $24(240), 1-113$ (2023)

[20] Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P.d.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al.: Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021)

[21] Liu, Y., Han, T., Ma, S., Zhang, J., Yang, Y., Tian, J., He, H., Li, A., He, M., Liu, Z., et al.: Summary of chatgpt-related research and perspective towards the future of large language models. Meta-Radiology, 100017 (2023)

[22] Tirumala, K., Markosyan, A., Zettlemoyer, L., Aghajanyan, A.: Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems 35, $38274-38290(2022)$

[23] Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al.: Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625 (2022)

[24] Da, L., Gao, M., Mei, H., Wei, H.: Llm powered sim-to-real transfer for traffic signal control. arXiv preprint arXiv:2308.14284 (2023)

[25] Li, M., Song, F., Yu, B., Yu, H., Li, Z., Huang, F., Li, Y.: Api-bank: A benchmark for tool-augmented llms. arXiv preprint arXiv:2304.08244 (2023)

[26] Wang, Y., Ma, X., Chen, W.: Augmenting black-box llms with medical textbooks for clinical question answering. arXiv preprint arXiv:2309.02233 (2023)

[27] Liang, Y., Wu, C., Song, T., Wu, W., Xia, Y., Liu, Y., Ou, Y., Lu, S., Ji, L., Mao, S., et al.: Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. arXiv preprint arXiv:2303.16434 (2023)

[28] Tong, L., Pan, Y., Shang, P., Guo, J., Xian, K., Zhou, X.: Open-source public transportation mobility simulation engine dtalite-s: A discretized space-time network-based modeling framework for bridging multi-agent simulation and optimization. Urban Rail Transit 5, 1-16 (2019)

[29] Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., Koltun, V.: Carla: An open urban driving simulator. In: Conference on Robot Learning, pp. 1-16 (2017). PMLR

[30] Mullakkal-Babu, F.A., Wang, M., van Arem, B., Shyrokau, B., Happee, R.: A hybrid submicroscopic-microscopic traffic flow simulation framework. IEEE Transactions on Intelligent Transportation Systems 22(6), $3430-3443$ (2020)

[31] de Souza, F., Verbas, O., Auld, J.: Mesoscopic traffic flow model for agentbased simulation. Procedia Computer Science 151, 858-863 (2019)

[32] Oppe, S.: Macroscopic models for traffic and traffic safety. Accident Analysis \& Prevention 21(3), 225-232 (1989)

[33] Boukerche, A., Tao, Y., Sun, P.: Artificial intelligence-based vehicular traffic flow prediction methods for supporting intelligent transportation systems. Computer networks 182, 107484 (2020)

[34] Masek, P., Masek, J., Frantik, P., Fujdiak, R., Ometov, A., Hosek, J., Andreev, S., Mlynek, P., Misurec, J.: A harmonized perspective on transportation management in smart cities: The novel iot-driven environment for road traffic modeling. Sensors 16(11), 1872 (2016)

[35] Maroto, J., Delso, E., Felez, J., Cabanellas, J.M.: Real-time traffic simulation with a microscopic model. IEEE Transactions on Intelligent

Transportation Systems 7(4), 513-527 (2006)

[36] NVIDIA: Simulation for self-driving vehicles (2023)

[37] Gulino, C., Fu, J., Luo, W., Tucker, G., Bronstein, E., Lu, Y., Harb, J., Pan, X., Wang, Y., Chen, X., et al.: Waymax: An accelerated, data-driven simulator for large-scale autonomous driving research. arXiv preprint arXiv:2310.08710 (2023)

[38] Behrisch, M., Bieker, L., Erdmann, J., Krajzewicz, D.: Sumo-simulation of urban mobility: an overview. In: Proceedings of SIMUL 2011, The Third International Conference on Advances in System Simulation (2011). ThinkMind

[39] Qadri, S.S.S.M., Gökçe, M.A., Öner, E.: State-of-art review of traffic signal control methods: challenges and opportunities. European transport research review 12, 1-23 (2020)

[40] Wei, H., Zheng, G., Yao, H., Li, Z.: Intellilight: A reinforcement learning approach for intelligent traffic light control. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pp. 2496-2505 (2018)

[41] Willumsen, L.G.: Estimation of an od matrix from traffic counts-a review (1978)

[42] Abrahamsson, T.: Estimation of origin-destination matrices using traffic counts-a literature survey (1998)

[43] Medina, A., Taft, N., Salamatian, K., Bhattacharyya, S., Diot, C.: Traffic matrix estimation: Existing techniques and new directions. ACM SIGCOMM Computer Communication Review 32(4), 161-174 (2002)

[44] Mahmassani, H.S.: Dynamic network traffic assignment and simulation methodology for advanced system management applications. Networks and spatial economics 1, 267-292 (2001)

[45] Mahmassani, H.S., Zhou, X.: In: Abed, E.H. (ed.) Transportation System Intelligence: Performance Measurement and Real-Time Traffic Estimation and Prediction in a Day-to-Day Learning Framework, pp. 305-328. Birkhäuser Boston, Boston, MA (2005)

[46] Zhou, X., Qin, X., Mahmassani, H.S.: Dynamic origin-destination demand estimation with multiday link traffic counts for planning applications. Transportation Research Record 1831(1), 30-38 (2003)

[47] Zhou, X., Erdoğan, S., Mahmassani, H.S.: Dynamic origin-destination trip
demand estimation for subarea analysis. Transportation Research Record 1964(1), 176-184 (2006)

[48] Zhou, X., List, G.F.: An information-theoretic sensor location model for traffic origin-destination demand estimation applications. Transportation Science 44(2), 254-273 (2010)

[49] Zhou, X., Lu, C., Zhang, K.: Dynamic origin-destination demand flow estimation utilizing heterogeneous data sources under congested traffic conditions (2013)

[50] Krishnakumari, P., Van Lint, H., Djukic, T., Cats, O.: A data driven method for od matrix estimation. Transportation Research Part C: Emerging Technologies 113, 38-56 (2020)

[51] Fedorov, A., Nikolskaia, K., Ivanov, S., Shepelev, V., Minbaleev, A.: Traffic flow estimation with data from a video surveillance camera. Journal of Big Data 6, 1-15 (2019)

[52] Pamuła, T., Żochowska, R.: Estimation and prediction of the od matrix in uncongested urban road network based on traffic flows using deep learning. Engineering Applications of Artificial Intelligence 117, 105550 (2023)

[53] Fu, H., Lam, W.H., Shao, H., Kattan, L., Salari, M.: Optimization of multi-type traffic sensor locations for estimation of multi-period origindestination demands with covariance effects. Transportation Research Part E: Logistics and Transportation Review 157, 102555 (2022)

[54] Kumarage, S., Yildirimoglu, M., Zheng, Z.: A hybrid modelling framework for the estimation of dynamic origin-destination flows. Transportation Research Part B: Methodological 176, 102804 (2023)

[55] Mei, H., Lei, X., Da, L., Shi, B., Wei, H.: Libsignal: an open library for traffic signal control. Machine Learning, 1-37 (2023)

[56] Cools, S.-B., Gershenson, C., D'Hooghe, B.: Self-organizing traffic lights: A realistic simulation. Advances in applied self-organizing systems, 45-55 (2013)

[57] Wei, H., Zheng, G., Gayah, V., Li, Z.: Recent advances in reinforcement learning for traffic signal control: A survey of models and evaluation. ACM SIGKDD Explorations Newsletter 22(2), 12-18 (2021)
</end of paper 2>


<paper 3>
# LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments 

Maonan Wang, Aoyu Pang, Yuheng Kan, Man-On Pun, Chung Shue Chen and Bo Huang


#### Abstract

Traffic congestion in metropolitan areas presents a formidable challenge with far-reaching economic, environmental, and societal ramifications. Therefore, effective congestion management is imperative, with traffic signal control (TSC) systems being pivotal in this endeavor. Conventional TSC systems, designed upon rule-based algorithms or reinforcement learning (RL), frequently exhibit deficiencies in managing the complexities and variabilities of urban traffic flows, constrained by their limited capacity for adaptation to unfamiliar scenarios. In response to these limitations, this work introduces an innovative approach that integrates Large Language Models (LLMs) into TSC, harnessing their advanced reasoning and decision-making faculties. Specifically, a hybrid framework that augments LLMs with a suite of perception and decision-making tools is proposed, facilitating the interrogation of both the static and dynamic traffic information. This design places the LLM at the center of the decision-making process, combining external traffic data with established TSC methods. Moreover, a simulation platform is developed to corroborate the efficacy of the proposed framework. The findings from our simulations attest to the system's adeptness in adjusting to a multiplicity of traffic environments without the need for additional training. Notably, in cases of Sensor Outage (SO), our approach surpasses conventional RL-based systems by reducing the average waiting time by $20.4 \%$. This research signifies a notable advance in TSC strategies and paves the way for the integration of LLMs into real-world, dynamic scenarios, highlighting their potential to revolutionize traffic management. The related code is available at https://github.com/Traffic-Alpha/LLM-Assisted-Light


Index Terms-Traffic Signal Control, Autonomous Agent, Large Language Model, Human-Machine Interface.

## I. INTRODUCTION

TRAFFIC congestion poses a significant challenge globally, leading to adverse economic, environmental, and social impacts [1]. Managing traffic flow efficiently, especially at road intersections, is crucial to alleviate congestion. Traffic signal control (TSC) systems are vital in this effort [2].

This work was supported in part by Shanghai Pujiang Program (21PJD092), the National Key R\&D Program of China with grant No. 2018YFB1800800, the Basic Research Project No. HZQB-KCZYZ-2021067 of Hetao ShenzhenHK S\&T Cooperation Zone. (Corresponding author: Man-On Pun)

Maonan Wang is with the Future Network of Intelligence Institute (FNii), the Chinese University of Hong Kong, Shenzhen, China and the Shanghai AI Laboratory, Shanghai, China.

Aoyu Pang and Man-On Pun are with the School of Science and Engineering, the Chinese University of Hong Kong, Shenzhen, China.

Yuheng Kan is with the SenseTime Group Limited, Shanghai, China and the Shanghai AI Laboratory, Shanghai, China.

Chung Shue Chen is with Nokia Bell Labs, Paris-Saclay, 12 Rue Jean Bart, 91300 Massy, France.

Bo Huang is with the Department of Geography, The University of Hong Kong, Hong Kong, SAR 999077, China.
Traditional rule-based TSC methods, such as the Webster method [3] and Self-Organizing Traffic Light Control (SOTL) [4], have been somewhat effective in managing traffic flow and reducing congestion. Yet, these systems are inherently limited by their static, rule-based algorithms that do not fully adapt to the ever-changing patterns of urban traffic [5].

Recently, the evolution of sensor technology and data collection has led to the development of more adaptive TSC strategies. In particular, Reinforcement Learning (RL) has emerged as an attractive approach, utilizing real-time data to dynamically adjust traffic signals [6]. Despite their potentials, these RL-based TSC systems are not without limitations. These systems may suffer from overfitting to specific traffic patterns. Additionally, RL systems typically rely on reward functions that may not be able to capture infrequent but critical events, such as emergency vehicles' sudden arrivals or unexpected road blockages. This can reduce their practicality in real-world conditions [7].

In response to these limitations, this paper introduces a novel approach that integrates Large Language Models (LLMs) into the TSC framework to assist in the decisionmaking process, named LLM-Assist Light (LA-Light). Our method leverages the extensive knowledge and "common sense" reasoning abilities of LLMs to enhance decisionmaking in complex and uncommon traffic situations. LLMs, with their sophisticated natural language processing capabilities, can interpret intricate traffic scenarios and recommend actions that may be overlooked by rule-based or RL-based systems. Furthermore, we introduce a set of tools specifically designed to bridge the gap between the TSC system and the LLM. These tools act as intermediaries, collecting environmental data and communicating with the LLM, which then guides the TSC system. This collaborative process allows for a well-rounded control strategy that not only makes informed decisions but also provides justifications for these decisions, thus improving the transparency of the system and building trust with traffic management operators.

Fig. 1 shows the difference between the method proposed in this paper and existing signal light control methods. The existing TSC systems, as shown in Fig. 1a, operate by making decisions based on predefined rules and observations, which may not suffice in unusual or unpredictable events. In contrast, our approach, depicted in Fig. 10, integrates an LLM Agent Module to enhance the system's comprehension of various traffic scenarios and the logic behind its decisions. Given that LLMs intrinsically lack the capacity for direct engagement
with traffic ecosystems or their data, an array of enhanced tools are devised to collect both the static and dynamic traffic information, subsequently facilitating the decision-making procedure. Crucially, this ensemble integrates existing rulebased and RL-based algorithms, guaranteeing that our methodology sustains state-of-the-art (SOTA) performance under conventional traffic conditions while also adapting effectively to exceptional situations.

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-02.jpg?height=144&width=781&top_left_y=562&top_left_x=195)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-02.jpg?height=279&width=789&top_left_y=760&top_left_x=207)

(b)

Fig. 1. Comparative framework analysis between LA-Light and conventional TSC systems. (a) illustrates a conventional TSC system wherein decisions are made directly by an algorithm that processes environmental inputs. (b) depicts the proposed LA-Light framework, which employs an LLM for the task of traffic signal control. In LA-Light, the LLM begins by selecting the most relevant tools from an enhanced set, including perception tools and decisionmaking algorithms, to collect and analyze traffic data. It then evaluates the information, adjusting its choice of tools as needed, until a definitive traffic control decision is formulated.

To corroborate the efficacy of the proposed framework, a simulation platform is developed. Extensive experiments are conducted on the simulation platform considering various intersection configurations. It is shown that the proposed LALight system achieves good performance in typical scenarios as well as in situations involving rare events. We also provide several qualitative examples where LA-Light accurately analyzes intricate traffic conditions and makes more reasonable decisions than conventional TSC methods. The experiments highlight the LLM-assisted system's capability to deeply understand traffic scenarios and to provide clear explanations for its actions. LA-Light is shown to make informed decisions that enhance safety, efficiency, and comfort, outperforming existing methods that may otherwise fail or yield suboptimal results under challenging conditions. The key contributions of this study are summarized as follows.

- We propose LA-Light, a hybrid TSC framework that integrates the human-mimetic reasoning capabilities of LLMs, enabling the signal control algorithm to interpret and respond to complex traffic scenarios with the nuanced judgment typical of human cognition. This innovation allows for seamless adaptation to urban traffic challenges, particularly in addressing the unpredictable and rare events that conventional systems may overlook;
- A closed-loop traffic signal control system has been developed, integrating LLMs with a comprehensive suite of interoperable tools. This integration yields in-depth insights into consistent and variable traffic patterns, thereby equipping the system with the capability for realtime analysis and decision-making that mirrors human intelligence. Furthermore, the system has been designed with a standardized interface, ensuring straightforward integration and facilitating customization by users;
- Through comprehensive experimentation, the results show that our model is adept at understanding and responding to a variety of environmental changes. In particular, it is able to address rare or unexpected events and provide superior performance, thereby validating its practical applicability and efficacy in real-world settings.

The remainder of this paper is organized as follows: Section [II provides an overview of existing literature in the fields of TSC and LLMs whereas Section III defines key terminologies pertaining to TSC that are referenced throughout this paper. After that, Section IV details the architecture of the proposed LLM-Assist Light framework, encompassing the utilized tools and the construction of prompts. Section V describes the experiments conducted to validate our approach. Finally, Section VI provides some concluding remarks and suggestions for future research directions.

## II. RELATED WORK

## A. Traffic Signal Control Methods

The pursuit of effective traffic signal control (TSC) strategies in urban settings is a well-established challenge with the goal of alleviating congestion. The rule-based TSC methods have been designed to optimize traffic signals under a variety of traffic conditions [8]. For example, the Webster method [3] calculates the ideal cycle length and distribution of traffic signal phases at intersections, based on traffic volumes and the assumption of a stable flow of traffic over a specific period. The Self-Organizing Traffic Light Control (SOTL) scheme [4] uses a set of predetermined rules to decide whether to continue with the current traffic signal phase or to change it. Adaptive TSC systems such as the Split Cycle Offset Optimization Technique (SCOOT) [9] and the Sydney Coordinated Adaptive Traffic System (SCATS) [10] dynamically alter cycle lengths, phase divisions, and offsets by choosing from a collection of predefined plans in response to live traffic sensor data. Although conventional TSC methods have achieved some success in mitigating congestion, their effectiveness is hindered by limitations in real-time traffic data utilization and difficulties in adapting to quickly changing traffic situations. Moreover, these methods often fall short in complex traffic scenarios [6].

Recent advancements in TSC have seen a shift towards RLbased systems, which are increasingly favored for their ability to dynamically manage traffic lights [11]. These systems typically use factors such as queue length [12]-[15], vehicle waiting time [16]-[18] or intersection pressure [12], [19], [20] as key indicators in their reward functions, training agents to reduce congestion. Furthermore, the frequency of signal switching has been considered [21] to prevent the negative impacts of rapid signal changes, such as increased stop-andgo driving and the risk of accidents. Although RL-based TSC systems offer flexibility in optimizing traffic flow by adjusting the reward function, finding the right balance for
these factors is a complex task [22]. Furthermore, if the reward function does not encompass infrequent but critical events, it may not provide the agent with sufficient direction to handle unexpected conditions effectively [7].

## B. Large Language Models

Large language models (LLMs), such as the Generative Pretrained Transformer (GPT) series [23], including its advanced versions like GPT-3.5 and GPT-4 Turbo [24], as well as open-source counterparts like Llama [25] and Llama2 [26], constitute a category of artificial intelligence systems designed to understand, generate and modify human language. These models rely on complex machine learning algorithms, specifically the transformer architecture [27], and are trained on extensive text datasets. Such comprehensive training grants them a sophisticated grasp of language nuances. A specialized version, InstructGPT [28], has been further refined to interpret user instructions with greater precision, providing relevant responses in a range of applications, from creating content to retrieving information. The "Chain-of-Thought" reasoning method [29] has introduced improved functionality within LLMs, enhancing their capability to solve complex problems by processing a series of logical steps. This significantly improves their performance in tasks that require arithmetic, commonsense, and symbolic reasoning. The "ReAct" strategy [30] extends the abilities of LLMs in complex tasks that require both reasoning and decision making. By prompting models to alternate between verbal reasoning and action generation, this approach enables dynamic reasoning and interaction with external systems.

Motivated by their superior performance, LLMs have been recently investigated for a multitude of tasks, as documented in recent surveys [31], [32]. These models have found applications in transportation planning and decision support systems [33]. For instance, [34] involves LLMs as decision-makers in conjunction with perception and positioning systems to aid autonomous vehicles whereas the Open-TI framework [35] integrates LLMs with traffic analysis tools, facilitating complex command execution through natural language interactions. This integration facilitates comprehensive traffic analysis, encompassing the acquisition and application of map data within sophisticated simulation platforms, thereby enhancing traffic signal control to more effectively address traffic requirements. Furthermore, LLMs have also been utilized to enhance lanechanging maneuvers in vehicles, aligning them more closely with human-like decision-making [36]. In instances of suboptimal outputs from rule-based planners, LLMs have been employed to evaluate the scenario and suggest alternative trajectories [37]. Finally, the amalgamation of LLMs with bird's-eye view (BEV) maps has been examined, offering a more holistic insight into traffic scenarios [38]. However, until now, LLMs have not been utilized to interpret complex traffic scenarios and control traffic signals to solve long-tail issues. Our proposed framework, LA-Light, seeks to employ LLMs together with a range of perception and decision-making tools. This approach allows the traffic signal control algorithm to understand and address complex traffic situations with the detailed judgment typical of human cognition.

## III. PRELIMINARIES

This section provides definitions for the key terminologies about TSC that are utilized throughout this paper. These terms are described with reference to a typical four-legged intersection, which is illustrated in Fig. 2

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-03.jpg?height=748&width=789&top_left_y=466&top_left_x=1118)

Fig. 2. A standard four-legged road intersection with eight traffic movements and signal phases.

Lanes: In the context of traffic intersections, lanes are categorized according to their purpose for the intersection. Incoming lanes guide vehicles toward the intersection, allowing them to enter. Conversely, outgoing lanes are designed to allow vehicles to depart from the intersection.

Traffic Movements: The directional flow of vehicles from incoming to outgoing lanes defines traffic movement at an intersection. A conventional four-way intersection typically includes four directional paths: East (E), West (W), North $(\mathrm{N})$, and South (S). Each path facilitates two main vehicular movements for exiting: a left turn, indicated by $l$, and a straight-ahead movement, denoted by $s$. In the context of this study, right-turn movements are excluded on the assumption that they are not signal-controlled in regions with right-hand traffic systems. Therefore, the traffic control system at the intersection administers eight distinct movements, labeled as $m_{1}$ through $m_{8}$.

Movement Signals: Movement signals are the controls that dictate the flow of traffic for each direction at an intersection. A green signal authorizes the traffic to proceed, while a red signal indicates a stop condition. As depicted in Fig. 2, at a standard four-way intersection, if movement signals $m_{4}$ and $m_{8}$ are green, this indicates that the corresponding movements-specifically, the westbound and eastbound left turns-are allowed, and all other movements are halted.

Phases: A traffic signal phase is a combination of movement signals that are displayed at the same time to manage multiple traffic flows. This configuration is designed so that all movements within a phase can proceed safely and without interference from other directions. As shown in Fig. 2, the

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-04.jpg?height=108&width=608&top_left_y=192&top_left_x=736)

Decision Unit

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-04.jpg?height=199&width=658&top_left_y=272&top_left_x=720)
Planning congestion

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-04.jpg?height=588&width=572&top_left_y=476&top_left_x=175)

## Environment

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-04.jpg?height=65&width=41&top_left_y=1057&top_left_x=439)

Toolkit for TSC

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-04.jpg?height=892&width=601&top_left_y=275&top_left_x=1334)

Fig. 3. The LA-Light framework: A schematic representation of the five-step process for integrating LLM in TSC. Step 1 outlines the task planning phase where the LLM defines its role in traffic management. Step 2 involves the selection of appropriate perception and decision-making tools by the LLM. In Step 3, these tools interact with the traffic environment to gather data. Step 4 depicts the analysis of this data by the Decision Unit to inform decision-making. Finally, Step 5 illustrates the implementation of the LLM's decisions and the provision of explanatory feedback for system transparency and validation.

intersection operates using four distinct phases, identified as $P_{1}$ through $P_{4}$. For example, during phase $P_{4}$, the green signals for movements $m_{4}$ and $m_{8}$ are activated concurrently, which enables vehicles to make left turns from both the westbound and eastbound approaches without conflict.

## IV. LLM-ASSISTED LIGHT

## A. Overview of LA-Light

The LA-Light framework introduces an innovative hybrid decision-making process for TSC that leverages the cognitive capabilities of LLMs alongside traditional traffic management methodologies. As illustrated in Fig. 3, the framework operates through a sequence of five methodical steps for decisionmaking, commencing with the specification of the LLM's role. In this initial phase, the LLM is assigned the function of regulating traffic signals at busy intersections to alleviate congestion, drawing on a combination of analytical and control tools.

Following this, the LLM is responsible for choosing the most appropriate tools from a predefined set of tools. These tools are divided into two categories, namely the perception tools and the decision-making tools. The perception tools are tasked with collecting a range of environmental data, both dynamic and static, to form a detailed picture of the traffic conditions. Conversely, the decision-making tools are specifically engineered for facilitating decisions and can be further classified into two categories: decision support tools, which utilize extant TSC algorithms to aid the decisionmaking process, and decision verification tools that assess the precision of decisions rendered by the LLM.

In the third stage, the chosen tools are activated within the traffic environment to collate traffic data, which is critical for informed decision-making. The collected data are then conveyed to the LLM, including the chat history, constituting the fourth step. At this point, the LLM scrutinizes the data to determine the next course of action. It evaluates the adequacy of the current data set and determines whether there is a necessity to activate supplementary tools for enhanced data acquisition.

Once sufficient data are obtained, the LLM proceeds to formulate traffic signal timing recommendations. These recommendations are then transmitted to the traffic control systems, implementing the recommended adjustments to the traffic light phases. The specific action taken in this study is the selection of an appropriate traffic phase ID for the junction, which the traffic lights then adopt. Concurrently, the LLM elucidates the reasoning behind its recommendations, thus improving the system's transparency and intelligibility. This aspect is vital for traffic operators, as it bolsters the reliability and trust in the system's operations.

The discussions above delineate the decision-making process at each juncture. In Algorithm 1, we introduce an elaborate control sequence that incorporates several decisionmaking cycles. Each cycle embodies the five steps previously outlined. Furthermore, the content of the dialogue is preserved
in the context dialogue memory $(M)$, which allows the LLM to integrate contextual data and construct a logical sequence for future decisions.

```
Algorithm 1 LA-Light: LLM-Assisted Traffic Signal Control
    Input: Total simulation time $T$, current time $t$, Intervention
    Frequency $\Delta t$, task description $D_{\text {task }}$, tool $x \in \mathcal{X}$, context
    dialogue memory $M$
    Initialize: $t=0, M=[$
    while $t<T$ do
        // Append task description to memory
        $M$.append $\left(D_{\text {task }}\right)$
        done $=$ False
        while not done do
            // Choose the appropriate tool
            $x=\operatorname{LLM}(M)$
            // Interact with environment
            $o b s=x(\mathrm{env})$
            // Record the tool used and observation
            $M$.append $((x, o b s))$
            // Determine if decision can be made
            done $=$ DecisionCriterion $(o b s)$
        // Execute decision in environment
        env.execute(decision)
        $t=t+\Delta t$
    Output: Implemented traffic signal timing adjustments and ra-
    tionale
```


## B. Toolkit for TSC

The LA-Light framework incorporates a comprehensive set of tools that facilitate the interaction of LLMs with the traffic environment. These tools, acting as sensory and cognitive extensions, enable the LLMs to accurately perceive traffic conditions and make well-informed decisions. The toolkit is divided into two main categories: Perception Tools and Decision Tools. Perception Tools are focused on the acquisition of static and dynamic traffic information, while Decision Tools support and evaluate the decision-making process. The design of these tools is modular and scalable, ensuring easy integration of new functionalities to accommodate various traffic management challenges. This approach allows the LLM to effectively combine traditional traffic control methods with its advanced reasoning capabilities, improving its performance in complex traffic scenarios.

Perception Tools (Static): The static subset of Perception Tools is responsible for capturing the unchanging aspects of the traffic environment. These include:

- Get_Intersection_Layout: This tool delineates the intersection's configuration, detailing the number and function of lanes associated with each direction, which is fundamental for understanding potential traffic flow scenarios.
- Get_Signal_Phase_Structure: This tool offers a detailed description of the traffic signal phases at the intersection, outlining the sequence and associated traffic movements for each phase.

Perception Tools (Dynamic): The dynamic category of Perception Tools are responsible for gathering real-time, fluctuating traffic parameters:

- Get_Occupancy: This function calculates the proportion of space currently occupied by vehicles in each move- ment, providing insights into congestion levels and the distribution of vehicles at the intersection.
- Get_Queue_Length: It measures the length of vehicle queues for each traffic movement, providing quantitative data to gauge traffic backlogs.
- Get_Phase_ID: This tool identifies the currently active traffic signal phase at the intersection, which is crucial for the LLM to understand which traffic flows are being allowed at any given moment.
- Get_Junction_Situation: This tool is designed to detect and assess unusual or emergency situations at the intersection, such as the arrival of emergency vehicles or traffic accidents, that may require immediate action or a departure from standard traffic control measures.

Decision Tools (Support): These tools are designed to aid the LLM in the decision-making process by offering reference points and additional insights:

- Get_Auxiliary_Decision: This function offers alternative decisions that can act as a reference or provide additional viewpoints to the LLM's decision-making process. Within LA-Light, the UniTSA method [18] is implemented as the foundational approach. UniTSA is an RL strategy that uses queue length as the reward metric. It is a universal RL-based method suitable for junctions with diverse configurations, eliminating the need for separate training for different junctions and still achieving good performance in standard situations.
- Get_Available_Actions: This tool lists the potential actions available to the LLM at any given time, outlining the range of immediate decision-making options.

Decision Tools (Evaluate): The evaluative subset of Decision Tools facilitates the validation and justification of the decisions proposed by the LLM:

- Evaluate_Action_Feasibility: Before a decision is implemented, this tool checks whether the proposed traffic signal phase is viable within the current signalization structure of the intersection. This preemptive evaluation helps prevent the selection of incompatible traffic phases.
- Justify_Decision_Logic: This function allows the LLM to explain the reasoning behind its decisions. Such explanations increase the transparency and understanding of the TSC process for traffic managers.

The toolkit described above is essential for the operation of the LA-Light framework, serving to connect the LLM's sophisticated reasoning abilities with the practical needs of TSC. Although the current set of tools is tailored to the particular requirements identified in this study, the framework's flexible architecture allows for the easy addition of new tools. This adaptability ensures that the framework can meet the changing demands of traffic management and fulfill future objectives.

## C. Prompt Design

The LA-Light system harnesses the interpretative capabilities of the LLM and refines its decision-making process through careful prompt engineering. This approach directs the model's decisions in complex traffic situations. The system prompt is meticulously crafted to encompass a broad spectrum

TABLE I

THE DETAILS OF THE PROMPT COMPONENTS FOR Get_Intersection_Layout

| Name | Description |
| :---: | :---: |
| Description | The Get_Intersection_Layout function provides a detailed configuration of an intersection's layout. <br> It returns the number and function of lanes for each direction at the specified intersection. |
| Input | junction_id (str): A string identifier for the intersection you wish to query. |
| Output | A dictionary where each key represents a traffic movement id at the intersection. <br> The corresponding value is another dictionary with the following keys: <br> - "direction": A string indicating the lane direction, where 's' is for straight, ' 1 ' is for left turn, and 'r' is for right turn. <br> - "number_of_lanes": An integer representing the number of lanes for the specified direction. |
| Example | To get the layout of intersection 'J1', call the function as follows: <br> layout = Get_Intersection_Layout('J1") <br> The expected output would be a dictionary describing the intersection layout, such as: <br> layout= $\{$ "E1": \{"direction": "s", "number_of_lanes": 2$\}$, "E2": $\{$ "direction": "l", "number_of_lanes": 1$\}, \ldots\}$ |

of considerations, enabling the LLM to accurately interpret traffic conditions and effectively manage traffic signals to mitigate congestion. The system prompt comprises five components, as depicted in Fig. 4.

The initial component is the task description, which articulates the LLM's objective in traffic signal control. After the task description, the second component is a detailed briefing on the functionalities of each integrated tool within the LALight framework. This briefing encompasses a list of tools, their respective purposes, the types of input they require, the output they generate, and illustrative examples of their use in practice. The third component consolidates observational data, which includes the most recent outputs from tool usage as well as the preceding chat history. This compilation of data equips the LLM with the relevant information needed for informed decision-making in subsequent interactions.

The fourth element of the system prompt addresses traffic regulations and other essential considerations that the LLM must take into account. This includes adherence to specific traffic laws and the integration of vital factors that influence the decision-making process when utilizing the tools. It ensures that the LLM's actions are not only optimized for traffic flow but also compliant with legal and safety standards. The final component of the system prompt outlines the expected output format. By defining the output format, the system facilitates seamless communication between the LLM and the traffic control tools, ensuring that the decisions are implemented effectively and efficiently.

To further refine the decision-making process, each tool within the LA-Light system is accompanied by its own tailored prompt. These prompts are composed of four parts, enhancing the clarity and precision with which the tools can be operated. For instance, Table I shows the prompt structure for the tool Get_Intersection_Layout. The first part provides a detailed description of the tool's capabilities. The second part outlines the necessary inputs required by the tool and the third part describes the output generated by the tool. Finally, a practical example is provided in Table $I$ to illustrate how the tool is used and the example of the output.

## V. EXPERIMENTS

## A. Experiment Setting

The experiments were carried out using the Simulation of Urban MObility (SUMO) [39], a widely recognized open source traffic simulator. To accurately capture traffic dynamics at intersections, we used virtual lane area detectors in the simulation to collect data such as vehicle count and queue length for each lane. Due to the constraints of simulated camera resolution, the scope of data collection was limited to a maximum of 150 meters from the intersection. This limitation was imposed to reflect realistic urban traffic monitoring conditions, although it is acknowledged that this may truncate actual queue lengths exceeding this distance.

In configuring the traffic signals, we adhered to common urban signaling sequences: a green light phase, followed by a 3-second yellow light, and then a red light phase. We set the parameters to match the realistic urban traffic flow, with a maximum speed limit of $13.9 \mathrm{~m} / \mathrm{s}$ (i.e., $50 \mathrm{~km} / \mathrm{h}$ ). The minimum distance between vehicles was kept at 2.5 meters, consistent with safe driving distances in city environments. Vehicle speeds were modeled with a Gaussian distribution, having a mean of $10 \mathrm{~m} / \mathrm{s}$ and a standard deviation of $\sqrt{3} \mathrm{~m} / \mathrm{s}$, to account for the variability in driver behavior. For the purposes of this study, we employed the GPT-4 Turbo model without any additional fine-tuning specific to TSC tasks. This choice was made to evaluate the model's out-of-the-box capabilities in managing complex traffic scenarios.

## B. Datasets and Scenarios

Our experiment utilizes both synthetic and real-world datasets to evaluate the performance of the proposed LA-Light in traffic signal control. The synthetic dataset includes scenarios of isolated intersections with varying layouts: a threeway intersection and a four-way intersection, both featuring three lanes per approach. For real-world data, we focus on the urban road network surrounding Chenta Road in the Songjiang District of Shanghai, a region known for its heavy traffic congestion due to high-density construction and commercial activities. The network, illustrated in Fig. 5. encompasses 18 intersections, comprising a combination of twelve four-way and six three-way intersections. To collect traffic flow data,

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-07.jpg?height=1593&width=868&top_left_y=434&top_left_x=173)

Fig. 4. System Prompt structure within the LA-Light framework. The design incorporates five components: (1) Task Description, detailing the LLM's role in traffic signal management; (2) Tools Synopsis, providing a catalog and description of available traffic control tools; (3) Observations Data, compiling data from tool feedback and chat history of the preceding cycle; (4) Attention Points, emphasizing compliance with traffic regulations and safety guidelines in tool deployment; and (5) Output Format, defining the protocol for the LLM's decision communication to ensure proper tool utilization. we analyzed video surveillance from these intersections on 30 July 2021. We recorded the number of vehicles per minute, which was then utilized to recreate the traffic scenarios in the SUMO platform.

To ensure a comprehensive evaluation of LA-Light's performance in complex urban traffic situations, we designed three specific test scenarios for each road network, as illustrated in Fig. 6. The first scenario, depicted in Fig. 6a, is the Emergency Vehicle (EMV) Scenario. In this scenario, emergency vehicles, such as ambulances, are introduced into the normal traffic flow, making up $1 \%$ of the overall traffic volume. These vehicles are assigned random origins and destinations to test the system's capability to prioritize them effectively. The second scenario, presented in Fig. 6b, is the Roadblock Incident (RBI) Scenario. This scenario mimics the dynamic nature of urban traffic by introducing random roadblocks, which stand in for unexpected incidents like traffic accidents. These roadblocks occur for $10 \%$ of the total simulation time, temporarily closing off affected lanes and testing the system's responsiveness to such events. The final scenario, shown in Fig. 6c, is the Sensor Outage (SO) Scenario. This scenario simulates sensor reliability challenges by introducing a $10 \%$ chance of sensor failure at any moment during the simulation. Such a failure results in the complete loss of vehicle detection data for that direction, challenging the system's ability to maintain efficient traffic control despite missing information.

## C. Metrics

To evaluate the effectiveness of TSC strategies, this study utilizes a dual-perspective approach to metrics. We measure the Average Travel Time (ATT), which is the time taken by vehicles to travel from their origin to their destination. ATT is a critical metric for assessing traffic flow efficiency within the network. Alongside ATT, we examine the Average Waiting Time (AWT). AWT measures the average time vehicles spend traveling at speeds below $0.1 \mathrm{~m} / \mathrm{s}$, typically while waiting for the green signal, thus providing a direct measure of intersection delay.

To address the urgency of emergency services, we incorporate specific metrics for emergency vehicles. The Average Emergency Travel Time (AETT) and the Average Emergency Waiting Time (AEWT) are calculated separately to underscore the TSC's impact on prioritized vehicle movement. These metrics are essential for comparing TSC strategies that can adapt in emergency situations, an integral aspect of proficient urban traffic management.

## D. Compared Methods

To assess the efficacy of the LA-Light model, we benchmarked it against a range of established TSC strategies. This benchmarking includes traditional transportation approaches as well as algorithms based on RL. We also evaluated the effectiveness of an LLM when used directly for decisionmaking, without the aid of Chain-of-Thought reasoning [40] and integration with existing TSC algorithms. The traditional traffic control methods are as follows:

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-08.jpg?height=889&width=914&top_left_y=217&top_left_x=153)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-08.jpg?height=889&width=897&top_left_y=217&top_left_x=1053)

(b)

Fig. 5. Traffic Network at Chenta Road, Songjiang District, Shanghai. (a) on Google Maps. (b) in the SUMO simulator.

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-08.jpg?height=559&width=556&top_left_y=1279&top_left_x=188)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-08.jpg?height=559&width=572&top_left_y=1279&top_left_x=771)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-08.jpg?height=556&width=566&top_left_y=1283&top_left_x=1365)

(c)

Fig. 6. Illustration of the three test scenarios. (a) Emergency Vehicle (EMV) Scenario, where ambulances are integrated into traffic flow; (b) Roadblock Incident (RBI) Scenario, depicting temporary road closures due to accidents or other events; (c) Sensor Outage (SO) Scenario, demonstrating the effects of sensor failures on traffic data accuracy.

- Webster [3]: This approach involves calculating the optimal signal cycle lengths and phase splits based on current traffic volumes and signal phase sequences. For this study, we implemented a real-time version of Webster's method, which dynamically adjusts the traffic signals in response to actual traffic conditions.
- SOTL [4]: The SOTL method evaluates the maximum queue length in the lanes associated with the current and subsequent signal phases. If the queue for the current phase is long, the green signal is extended; otherwise, the system triggers a shift to the next phase.
- Maxpressure [19]: This advanced transportation method for traffic light control aims to reduce congestion in the lanes that exhibit the highest pressure. This pressure is quantified by the difference in queue lengths between upstream and downstream lanes. The method focuses on alleviating congestion where it is most needed.

For the RL-based models, we include:

- IntelliLight [21]: This model leverages a state representation enriched with lane-specific details, including vehicle count and waiting time, thus providing a nuanced view of traffic conditions beyond simple queue length. The reward function of IntelliLight accounts for the frequency of signal changes, and it introduces a phase-gate model to
mitigate the challenge of skewed phase data distribution.
- PressLight [12]: An extension of the Maxpressure algorithm, PressLight combines deep reinforcement learning with the pressure optimization concept to dynamically adjust signals at intersections, aiming to maintain optimal flow.
- AttendLight [20]: Implementing an attention mechanism, AttendLight builds a set of observational features that inform the probability of phase changes, thus refining the signal timing optimization process.
- UniTSA [18]: Introducing a sophisticated intersection representation known as the junction matrix, UniTSA also brings five novel traffic state augmentation methods tailored to enhance signal control system performance.

Additionally, this study contrasts the proposed LA-Light system with a baseline LLM-based TSC method, herein referred to as Vanilla-LLM, which utilizes an LLM directly for TSC decision-making. Unlike LA-Light, the Vanilla-LLM approach does not incorporate Chain-of-Thought reasoning or supplemental decision-support tools. It relies solely on the model's inherent capabilities to interpret traffic data and make determinations based on the real-time traffic conditions. Key differences between the LA-Light framework and the VanillaLLM method are delineated in Table II.

TABLE II

COMPARISON OF LA-LIGHT AND VANILLA-LLM

| Tool/Component | Vanilla-LLM | LA-Light |
| :---: | :---: | :---: |
| Perception Tools (Static) |  |  |
| Get_Intersection_Layout | $\times$ | $\checkmark$ |
| Get_Signal_Phase_Structure | $\times$ | $\checkmark$ |
| Perception Tools (Dynamic) |  |  |
| Get_Occupancy | $\times$ | $\checkmark$ |
| Get_Queue_Length | $\times$ | $\checkmark$ |
| Get_Phase_ID | $\times$ | $\checkmark$ |
| Get_Junction_Situation | $\times$ | $\checkmark$ |
| Decision Tools (Support) | $\times$ |  |
| Get_Auxiliary_Decision | $\times$ | $\checkmark$ |
| Get_Available_Actions |  | $\checkmark$ |
| Decision Tools (Evaluate) | $\checkmark$ |  |
| Evaluate_Action_Feasibility | $\checkmark$ |  |
| Justify_Decision_Logic |  | $\checkmark$ |
| Additional Capabilities |  |  |
| Chain-of-Thought Reasoning |  |  |
| Ehat History Analysis |  |  |

## E. Performance Analysis

In this section, we assess the performance of the proposed LA-Light framework alongside various benchmark methods across three distinct road maps, with each map featuring three unique scenarios. Table III presents the results under the EMV
Scenario. LA-Light's comparative analysis against traditional traffic signal control methods, RL-based approaches, and other LLM-based methods demonstrates a comprehensive enhancement in traffic signal control efficiency for both regular and emergency vehicles. For example, in comparison with the Maxpressure approach, LA-Light achieves a $32.1 \%$ reduction in ATT for the four-way intersection (4-Way INT) and a $10.8 \%$ reduction for the Shanghai network. In terms of emergency vehicle efficiency, indicated by AETT, LA-Light shows a significant improvement, reducing AETT by $15.3 \%$ in the Shanghai network compared to Maxpressure. This improvement is attributed to the integration of RL algorithms within LA-Light, which refines decision-making processes in standard traffic scenarios.

When compared with RL-based methods, LA-Light does not always surpass in ATT and AWT, yet it significantly outperforms in emergency vehicle efficiency. For instance, while AttendLight records a lower ATT by $3.3 \%$ for the three-way intersection (3-Way INT), LA-Light outperforms it in AEWT by $67.3 \%$ for the same scenario. This highlights LA-Light's capability to assimilate environmental observations and adjust to dynamic changes, emphasizing its robustness in urgent situations without specialized fine-tuning. Furthermore, when LA-Light is compared to another LLM-based method, Vanilla-LLM, it exhibits a notable improvement in ATT and AWT across all tested networks. Specifically, in the complex Shanghai network, LA-Light reduces ATT and AWT by $16.5 \%$ and $24.2 \%$, respectively, compared to Vanilla-LLM. The integration of existing TSC methods within LA-Light's decision-making framework likely contributes to its enhanced traffic efficiency, demonstrating the potential of LLM-assisted approaches in urban traffic management.

Table IV details the performance outcomes in the RBI Scenario, where LA-Light's adaptability to unforeseen traffic events is pronounced. Traditional transportation methods like Webster, SOTL, and Maxpressure, which depend on fixed algorithms, are less adept at adjusting to sudden changes such as those introduced by roadblocks. RL-based methods are more adaptable but are still constrained by their reliance on previously learned strategies, which may not be sufficiently flexible for drastic alterations in road capacity. LA-Light, leveraging the real-time processing abilities of its LLM, dynamically responds to these traffic alterations. In the 3-Way INT scenario, LA-Light shows a $2.2 \%$ reduction in ATT and a $6.3 \%$ reduction in AWT in comparison to IntelliLight, the most effective RL-based method. In the more intricate Shanghai network, LA-Light's performance is even more notable, with a $6.8 \%$ improvement in ATT and a $11.3 \%$ improvement in AWT over UniTSA, the best RL-based approach.

Furthermore, LA-Light's proficiency extends to emergency response metrics, such as AETT and AEWT. On the 4way INT, LA-Light shows a $35.6 \%$ betterment in AETT and a $74.5 \%$ enhancement in AEWT compared to UniTSA. Against another LLM-based method, Vanilla-LLM, LA-Light underscores the value of not only LLM's decision-making capabilities but also the sophisticated integration of chainof-thought reasoning with effective tool utilization. This is particularly evident in the Shanghai network, where LA-Light

TABLE III

PERFORMANCE COMPARISON UNDER THE EMERGENCY VEHICLE (EMV) SCENARIO. THE BEST, SECOND-BEST RESULTS ARE HIGHLIGHTED THROUGH BOLD, AND UNDERLINING, RESPECTIVELY.

| Method | 3-Way INT |  |  |  | 4 Way INT |  |  |  | Shanghai |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | ATT | $\overline{A W T}$ | AETT | AEWT | ATT | AWT | AETT | AEWT | ATT | AWT | AETT | AEWT |
|  | Traditional Transportation Approaches |  |  |  |  |  |  |  |  |  |  |  |
| Webster | 71.708 | 34.443 | 89.738 | 45.730 | 122.135 | 64.074 | 139.550 | 86.188 | 538.614 | 152.957 | 504.267 | 155.110 |
| Maxpressure | 63.860 | 29.494 | 83.728 | 39.859 | 103.598 | 51.415 | 109.824 | 58.344 | 461.965 | 96.820 | 438.092 | 102.515 |
|  | RL-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| AttendLight | 61.750 | 26.796 | 77.926 | 34.229 | 74.538 | 32.341 | 88.569 | 46.743 | 429.452 | 81.017 | 432.419 | 116.944 |
| UniTSA | 67.848 | $\underline{21.424}$ | 83.236 | 38.073 | 64.032 | 27.224 | 73.941 | 40.228 | 398.374 | $\underline{79.426}$ | 403.642 | 98.593 |
|  | LLM-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| Vanilla-LLM | 77.423 | 36.609 | 46.970 | 9.862 | 87.738 | 48.167 | $\underline{49.324}$ | $\underline{11.010}$ | 493.577 | 109.182 | $\underline{391.836}$ | $\underline{35.246}$ |
| LA-Light | 63.853 | 22.516 | $\underline{48.824}$ | $\underline{11.251}$ | $\underline{69.965}$ | 31.457 | $\overline{47.808}$ | $\overline{10.435}$ | 411.826 | 82.802 | $\overline{371.997}$ | $\overline{17.476}$ |

TABLE IV

PERFORMANCE COMPARISON UNDER THE ROADBLOCK INCIDENT (RBI) SCENARIO.

| Method | 3-Way INT |  |  |  | 4 Way INT |  |  |  | Shanghai |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | ATT | $\overline{\mathrm{AWT}}$ | AETT | AEWT | ATT | $\overline{\mathrm{AWT}}$ | AETT | AEWT | ATT | AWT | $\overline{\text { AETT }}$ | AEWT |
| Webster | 83.019 | 38.768 | 102.043 | 59.879 | 132.069 | 80.311 | 138.364 | 82.314 | 596.965 | 282.545 | 674.380 | 177.341 |
| Maxpressure | 73.811 | 32.910 | 96.754 | 58.398 | 115.573 | 64.153 | 102.236 | 54.278 | 512.578 | 118.984 | 477.416 | 92.406 |
|  | RL-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| IntelliLight | $\underline{68.007}$ | $\underline{29.178}$ | 71.237 | 38.331 | 80.973 | 48.819 | 90.449 | 47.794 | 494.662 | 117.503 | 475.623 | 106.845 |
| AttendLight | 69.418 | 29.849 | 92.717 | 54.235 | $\underline{78.364}$ | $\underline{35.330}$ | 76.735 | 26.958 | 493.844 | 121.402 | 477.328 | 131.610 |
| UniTSA | 79.760 | 43.367 | 83.354 | 49.510 | $\overline{81.650}$ | $\overline{38.604}$ | 78.828 | 38.693 | 467.262 | 97.154 | 410.960 | 70.844 |
|  | LLM-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| Vanilla-LLM | 79.967 | 34.122 | 54.912 | 15.939 | 93.009 | 47.206 | $\underline{73.451}$ | $\underline{24.523}$ | 490.522 | 93.035 | 420.371 | $\underline{51.674}$ |
| LA-Light | 66.510 | 27.208 | $\underline{55.094}$ | $\underline{20.458}$ | 71.982 | 33.266 | 64.808 | $\overline{18.900}$ | 435.698 | $\underline{86.902}$ | 379.822 | $\overline{23.031}$ |

TABLE V

PERFORMANCE COMPARISON UNDER THE SENSOR OUTAGE (SO) SCENARIO.

| Method | 3-Way INT |  |  |  | 4 Way INT |  |  |  | Shanghai |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | ATT | AWT | AETT | AEWT | ATT | AWT | AETT | AEWT | ATT | AWT | AETT | AEWT |
|  | Traditional Transportation Approaches |  |  |  |  |  |  |  |  |  |  |  |
| Webster | 78.449 | 33.250 | 82.845 | 45.740 | 124.008 | 67.745 | 134.664 | 80.372 | 548.485 | 159.720 | 582.200 | 202.231 |
| Maxpressure | $\underline{69.076}$ | 28.555 | 68.010 | 39.950 | 107.863 | 62.312 | 127.463 | 82.852 | 538.497 | 128.103 | 551.391 | 174.590 |
|  | RL-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| AttendLight | 73.108 | $\underline{29.473}$ | 77.385 | 36.563 | 86.714 | 41.764 | 99.366 | 55.019 | 557.932 | 192.267 | 625.205 | 282.224 |
| UniTSA | 83.429 | 50.217 | 89.839 | 47.719 | $\underline{81.370}$ | $\underline{39.792}$ | 82.158 | 41.203 | $\underline{474.909}$ | 103.004 | 497.134 | 124.262 |
|  | LLM-based Methods |  |  |  |  |  |  |  |  |  |  |  |
| Vanilla-LLM | 81.125 | 44.184 | $\underline{49.412}$ | $\underline{13.754}$ | 85.993 | 40.698 | $\underline{56.215}$ | $\underline{13.356}$ | 496.504 | 109.405 | $\underline{407.128}$ | $\underline{38.696}$ |
| LA-Light | 67.726 | 23.520 | 46.568 | 10.759 | 72.071 | 33.874 | 47.204 | 9.741 | 438.408 | 82.600 | 380.228 | 21.086 |

achieves a $16.6 \%$ reduction in ATT and a remarkable $51.4 \%$ improvement in AEWT compared to Vanilla-LLM.

Finally, Table V shows the performance under SO Scenario. Similar to the conclusion under RBI Scenario, the result demonstrates LA-Light's capability to effectively manage traffic even with sensor failures, a rare but critical challenge. For example, in the complex Shanghai network, LA-Light significantly reduces ATT and AWT by $20.0 \%$ and $35.9 \%$, respectively, compared to the Maxpressure method. While RLbased methods exhibit a degree of adaptability, they struggle in the absence of sensor data. Our method, LA-Light, addresses this shortcoming by utilizing common sense reasoning and the tools at hand. Compared to UniTSA, the top-performing RLbased method in this scenario, LA-Light achieves a $7.7 \%$ improvement in ATT and a $20.4 \%$ reduction in AWT. Moreover, LA-Light's performance excels against another LLM-based method, Vanilla-LLM, with an $11.7 \%$ betterment in ATT and a $24.8 \%$ enhancement in AWT, highlighting the efficiency of its decision-making process in scenarios with incomplete data, and confirming its robustness as a reliable traffic management solution.

The comparative analysis of LA-Light's performance in varying scenarios underscores its reliable effectiveness amidst environmental uncertainties. Notably, within the Shanghai network, the shift from the EMV to the SO scenario resulted in a modest increase $6.9 \%$ in ATT and a $0.2 \%$ in AWT, demonstrating LA-Light's commendable stability. This performance is markedly superior to RL-based methods, such as UniTSA, which exhibited a significant performance drop, $16.1 \%$ in ATT and $23.3 \%$ in AWT, under the same conditions. Further, LALight's emergency response metrics, specifically AETT and AEWT, remain the best among all benchmarks in all three scenarios. This consistency confirms the resilience of the LALight framework, which leverages LLMs to ensure minimal performance decline even in less common situations. These results emphasize LA-Light's capability to deliver dependable traffic signal control in diverse and complex urban environments.

## F. Case Study Insights

In this section, we conduct a comprehensive analysis of the decision-making processes utilized by the LA-Light across a variety of traffic conditions. We first compare the decision of our method with that of the UniTSA in different traffic scenarios. These results are depicted in Fig. 7 for the synthetic dataset and Fig. 8 for scenarios in Shanghai. In the synthetic dataset, as depicted in Fig. 7, the LA-Light framework demonstrates enhanced adaptability. For example, in the EMV Scenario at the 3 Way INT, UniTSA changes the signal from phase $P_{1}$ to $P_{2}$, giving priority to the larger volume of vehicles turning left from the west side (W-1). However, this action inadvertently causes a delay for the emergency vehicle. In contrast, LA-Light, utilizing the LLM's nuanced understanding of the scenario, changes the green phase to $P_{2}$, which, despite the queue forming on the north side (N-1), allows the emergency vehicle to pass without delay. Similarly, in the SO Scenario at the 4 Way INT, UniTSA fails to receive accurate data from the north side due to a damaged sensor. As a result, it mistakenly extends the green phase for $P_{4}$, which worsens the congestion on the north side. LA-Light, on the other hand, identifies the faulty sensor data and integrates this information with the real-time traffic conditions on the south side. It infers that congestion is increasing on the north side and accordingly adjusts the signal to green phase $P_{1}$, which is for the north-south through traffic, effectively reducing the congestion at the intersection.

The real-world scenarios presented in Fig. 8 further highlight the effectiveness of the LA-Light system in navigating the complexities of urban traffic networks. In the EMV Scenario within the Shanghai network, LA-Light aptly prolongs the current green phase $P_{4}$, allowing the emergency vehicle to pass swiftly. In contrast, UniTSA does not give precedence to the emergency vehicle, opting instead to clear lanes with higher vehicle accumulation, thereby neglecting the urgency of emergency response. In the RBI Scenario, where an obstruction is present on the east exit road, LA-Light successfully infers the larger traffic impact and redirects the green phase towards lanes that are not affected, preventing further congestion. UniTSA, however, defaults to activating the green phase for $P_{3}$, which is rendered ineffective as the blockage hinders northbound traffic from proceeding, leading to a suboptimal use of the green phase. These case studies demonstrate that LA-Light can dynamically adapt to diverse and unpredictable environments, showcasing its potential for zero-shot adaptation in real-time traffic management.

In the following analysis, we explore the decision-making process of LA-Light in response to various urban traffic scenarios. Fig. 9 illustrates the steps taken under the SO Scenario. LA-Light begins its evaluation by analyzing both the static and dynamic aspects of the traffic intersection. These include the intersection's physical configuration, preset signal phase timings, and real-time traffic density for each lane and direction. During this initial phase, LA-Light identifies a discrepancy: the sensor on the "E1-s" (E1 straight) approach reports an occupancy of $-100 \%$, clearly signaling a fault, as negative occupancy is not feasible. Simultaneously, it notes significant congestion on the "E3-s" direction, with an occupancy of $62.87 \%$, and the "E2-l" direction, with $54.00 \%$ occupancy. Facing incomplete data, LA-Light uses an auxiliary decision-support tool for a reference solution. This tool suggests prioritizing phase $P_{4}$, managing left turns from E2 and E4 approaches, aiming to reduce queue lengths, especially for the congested "E2-l" movement.

However, LA-Light applies more complex reasoning. It considers the heavy congestion on the "E3-s" direction and the inoperative "E1-s" sensor, requiring a strategic response. The faulty sensor makes it impossible to estimate the queue on the "E1-s" approach accurately, which may be significant. LA-Light decides to deviate from UniTSA's advice, opting for phase $P_{1}$ to alleviate traffic from the "E1-s" and "E3$s$ " directions, likely facing higher demand. Additionally, LALight recommends a prompt on-site inspection of the "E1s" approach by traffic personnel to counter unreliable sensor data. This human intervention ensures the resilience of the decision-making process, even when automated systems are

EMV Scenario at 3 Way INT
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=990&width=506&top_left_y=245&top_left_x=168)

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=485&width=409&top_left_y=245&top_left_x=663)

SO Scenario at 4 Way INT
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=996&width=838&top_left_y=240&top_left_x=1096)

Fig. 7. Comparative analysis of UniTSA and LA-Light strategies in handling EMV and SO scenarios at synthetic intersections. The top row illustrates the decision and subsequent traffic flow impact of the UniTSA method, while the bottom row demonstrates the same for the LA-Light method. The sequence of actions and their effects on traffic dynamics are depicted, with the left side of each panel displaying the intersection conditions considered by each agent, and the right side showing the implemented actions and resultant traffic flow alterations at 3 Way INT and 4 Way INT.

EMV Scenario at Shanghai Network
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=994&width=512&top_left_y=1512&top_left_x=167)
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=992&width=428&top_left_y=1514&top_left_x=663)

RBI Scenario at Shanghai Network

![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-12.jpg?height=998&width=852&top_left_y=1512&top_left_x=1100)

Fig. 8. Comparative analysis of UniTSA and LA-Light strategies in handling EMV and RBI scenarios at the Shanghai network.
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-13.jpg?height=1420&width=734&top_left_y=190&top_left_x=234)

## LA-Light (SO Scenario)

Thought 1: To make a decision, I first need to understand the static state of the traffic light $[. .$.

Act 1: Get_Intersection_Layout

Obs 1: The description of this intersection layout, "E1-s": \{"direction": "Through", "number of lanes": 2\}, "E1-l": \{"direction": “Left Turn","number of lanes": 1\}, [...] ...

Thought 4: The signal phase indicates that there are four phases controlling different sets of movements. Phase 1 controls the straight for E1 and E3, [...]. Next, I need to assess the dynamic state of the intersection to understand the congestion levels for each movement. [...]

Act 4: Get_Occupancy

Obs 4: At the current moment 136.0, the congestion situation of each movement is: \{"E1-s": "-100\%", "E1-l": "-100\%", "E2-s": "13.98\%", ...\} [...]

Thought 5: Currently, the "E3-s" movement has the highest congestion at $62.87 \%$, followed by [...]. The occupancy for "E1-s" and "E1-l" is reported as "-100\%", which indicates a malfunctioning detector or an error in the data. Before making a decision, I need to know what the Traditional Decision would be under the current circumstances and what actions are available to me.

Act 5: Get_Auxiliary_Decision

Obs 5: The decision is to set Phase 4 as the green signal.

Thought 6: The Traditional Decision suggests setting Phase 4 as the green signal. However, I must also consider the high congestion on the "E3-s". [...]. To ensure that I am making the best decision, I need to know what actions are available to me at this step.

Act 6: Get Available Actions

Obs 6: Phase 1; Phase 2; Phase 3; Phase 4.

...

## Final Decision \& Explanations

Decision: Phase 1

Explanations: The Traditional Decision recommends activating Phase 4. However, I also need to consider the high congestion on the "E3-s" movement and the fact that the "E1-s" detector is not working. [...] Phase 1 would address the highest congestion observed, which is on the "E3-s" movement. Balancing the need to follow the traditional decision with the observed congestion levels, I will prioritize the movement with the highest congestion, which is "E3-s". Therefore, my decision will be to activate Phase 1. I will also suggest that the Crossing Guard manually monitor the "E2-s" movement due to the malfunctioning detector.

Fig. 9. Decision-making process and explanation of LA-Light during the SO Scenario.

compromised.

Fig. 10 illustrates LA-Light's decision-making process in the RBI Scenario, starting with gathering crucial data from the intersection as done in the SO Scenario depicted in Fig. 9. The "get junction situation" function is then used, revealing a blockage on the E2 outgoing edge due to a vehicular accident. After that, LA-Light performs a detailed evaluation of the intersection's condition, determining that the "E1-l" and "E4s" movements, associated with $P_{2}$ and $P_{3}$ phases of the traffic signals, are the most affected by the accident.

Having pinpointed the affected phases, LA-Light employs the tool to list all potential signal phases, then eliminates those hindered by the incident. This process leaves $P_{1}$ and $P_{4}$ as the feasible choices. In choosing between them, LALight examines the current congestion at the intersection. The decision-making is strategically grounded: the previous decision involved setting phase $P_{1}$ to a green signal to reduce congestion in the "E1-s" and "E3-s" directions. This move has effectively shortened queue lengths according to the latest data. Thus, to continue efficient traffic management and to alleviate congestion elsewhere, LA-Light opts for $P_{4}$. This phase controls the "E2-l" and "E4-l" movements, which, though congested, are clear of obstructions and can therefore benefit from a longer green signal. The selection of $P_{4}$ is deliberate, in line with LA-Light's goal to manage congestion proactively while avoiding the areas affected by the accident. This decision showcases LA-Light's capacity to adjust dynamically to real-time traffic situations, using its computational capabilities to maintain traffic flow as smoothly as possible despite unexpected challenges. Additionally, the rationale behind the decisions made in both the SO and RBI scenarios is also clarified in Fig. 9 and Fig. 10. This transparency in the decision-making process enhances the reliability and trustworthiness of the signal control system.

## VI. CONCLUSION

In this work, we have presented the LA-Light framework, which incorporates LLMs to improve decision-making in the dynamic and complex environment of urban traffic management. By combining the sophisticated reasoning abilities
![](https://cdn.mathpix.com/cropped/2024_06_04_5923beedb385e7d0036dg-14.jpg?height=1428&width=692&top_left_y=190&top_left_x=236)

## LA-Light (RBI Scenario)

Thought 4: Now I need to determine if there is a long-tail problem, such as [...]. If there is a long-tail problem, I will need to analyze the available actions and make a judgment. If it's a standard situation, I will refer to the Traditional Decision.

Act 4: Get Junction Situation

Obs 4: Due to an accident, E2 outgoing lane is impassable.

Thought 5: The Junction Situation shows E2 outgoing lane is impassable. It indicates that movements E1-s, E2-s, E2-l, E3-s, E3-l and E4-l are passable, while E1-l and E4-s is blocked due to an accident, respectively. Given this information, I must avoid activating phases that include the impassable movements. [...] Next, I need to consider the available actions that do not include the impassable movements 'E1-l' and 'E4-s'. Phase 1 and Phase 4 are potential candidates since they do not include these movements. I will now check the available actions to make a final decision.

Act 5: Get_Available_Actions

Obs 5: Phase 1; Phase 2; Phase 3; Phase 4.

Thought 6: Having reviewed the available actions, I am inclined to activate Phase 4 as the green signal. [...] However, before I commit to this decision, it is critical to reassess the viability of this action in the current traffic context to ensure it does not exacerbate the situation or overlook any unseen complications.

Act 6: Evaluate_Action_Feasibility ("Phase 4")

Obs 6: Activation of Phase 4 is feasible and will not interfere with emergency response efforts $\cdots$


#### Abstract

Final Decision \& Explanations Decision: Phase 4

Explanations: Given the available actions and the need to avoid phases that include impassable movements, I must choose between Phase 1 and Phase 4. [...] To make an informed decision between Phase 1 and Phase 4, I should consider the current occupancy rates for the movements included in these phases. Phase 1 controls the movements 'E1-s' and 'E3-s', while Phase 4 controls the movements 'E2-l', 'E4-l'. Since the last decision made was to activate Phase 1 to alleviate congestion in the 'E1-s' and 'E3-s' movements, and considering that these movements queue length have reduced, it would be reasonable to change to Phase 4 to reduce congestion in 'E2-l', 'E4-l' movement. Therefore, my decision is to activate Phase 4, as it addresses the congested movements that are passable and does not include any impassable movements.


Fig. 10. Decision-making process and explanation of LA-Light during the RBI Scenario.

of LLMs with established TSC methods and real-time data collection tools, we have established a new approach to traffic signal control. Our comprehensive evaluation of LALight, conducted across three distinct traffic networks and nine unique scenarios, has shown its effectiveness without necessitating further training. Compared to traditional methods, LALight has consistently achieved reductions in ATT and AWT. Additionally, it has demonstrated improvements in metrics for emergency response vehicles, such as AETT and AEWT. Analysis of the decision-making process in various contexts revealed that LA-Light excels not only in operational performance but also in decision-making clarity, courtesy of the LLMs' explanatory capabilities. LA-Light adeptly identifies and utilizes the most suitable tools for a given traffic situation, providing clear insights into its decision-making rationale.

While the LA-Light framework marks a significant step forward, it does have areas that require further refinement. The framework's current dependency on frequent interactions with the LLM for decision-making introduces a delay that could impact the promptness of traffic signal adjustments. Moreover, the framework's reliance on textual descriptions to depict traffic scenarios may not encompass all the details needed for the most effective decision-making, pointing to the potential benefits of a more direct, image-based approach that can interpret traffic conditions from visual data. Future work will aim to address these issues by refining the interaction process to expedite response times and by incorporating vision-based models capable of directly processing visual information. These enhancements are expected to improve the framework's proficiency in managing the complexities of real-world traffic systems with increased speed and less reliance on textual descriptions.

## REFERENCES

[1] M. Sweet, "Does traffic congestion slow the economy?," Journal of Planning Literature, vol. 26, no. 4, pp. 391-404, 2011.

[2] D. Zhao, Y. Dai, and Z. Zhang, "Computational intelligence in urban traffic signal control: A survey," IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 42, no. 4, pp. 485-494, 2011.

[3] P. Koonce and L. Rodegerdts, "Traffic signal timing manual.," tech. rep., United States. Federal Highway Administration, 2008.

[4] S.-B. Cools, C. Gershenson, and B. D'Hooghe, "Self-organizing traffic lights: A realistic simulation," Advances in applied self-organizing systems, pp. 45-55, 2013.

[5] S. S. S. M. Qadri, M. A. Gökçe, and E. Öner, "State-of-art review of traffic signal control methods: challenges and opportunities," European transport research review, vol. 12, pp. 1-23, 2020.

[6] H. Wei, G. Zheng, V. Gayah, and Z. Li, "A survey on traffic signal control methods," arXiv preprint arXiv:1904.08117, 2019.

[7] H. Vardhan and J. Sztipanovits, "Rare event failure test case generation in learning-enabled-controllers," in 6th International Conference on Machine Learning Technologies, pp. 34-40, 2021.

[8] F. J. Martinez, C. K. Toh, J.-C. Cano, C. T. Calafate, and P. Manzoni, "A survey and comparative study of simulators for vehicular ad hoc networks (VANETs)," Wireless Communications and Mobile Computing, vol. 11, no. 7, pp. 813-828, 2011.

[9] P. Hunt, D. Robertson, R. Bretherton, and M. C. Royle, "The scoot on-line traffic signal optimisation technique," Traffic Engineering \& Control, vol. 23, no. 4, 1982.

[10] P. Lowrie, "Scats-a traffic responsive method of controlling urban traffic," Sales information brochure published by Roads \& Traffic Authority, Sydney, Australia, 1990.

[11] C. Wu, I. Kim, and Z. Ma, "Deep reinforcement learning based traffic signal control: A comparative analysis," Procedia Computer Science, vol. 220, pp. 275-282, 2023.

[12] H. Wei, C. Chen, G. Zheng, K. Wu, V. Gayah, K. Xu, and Z. Li, "Presslight: Learning max pressure control to coordinate traffic signals in arterial network," in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pp. 12901298,2019

[13] X. Zang, H. Yao, G. Zheng, N. Xu, K. Xu, and Z. Li, "Metalight: Value-based meta-reinforcement learning for traffic signal control," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 1153-1160, 2020

[14] C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, Y. Xiong, K. Xu, and Z. Li, "Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 3414-3421, 2020.

[15] A. Pang, M. Wang, Y. Chen, M.-O. Pun, and M. Lepech, "Scalable reinforcement learning framework for traffic signal control under communication delays," IEEE Open Journal of Vehicular Technology, 2024.

[16] T. Chu, J. Wang, L. Codecà, and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086-1095, 2019.

[17] M. Wang, Y. Xu, X. Xiong, Y. Kan, C. Xu, and M.-O. Pun, "ADLight: A universal approach of traffic signal control with augmented data using reinforcement learning," in Transportation Research Board (TRB) 102nd Annual Meeting, 2023.

[18] M. Wang, X. Xiong, Y. Kan, C. Xu, and M.-O. Pun, "UniTSA: A universal reinforcement learning framework for $\mathrm{v} 2 \mathrm{x}$ traffic signal control," arXiv preprint arXiv:2312.05090, 2023.

[19] P. Varaiya, "Max pressure control of a network of signalized intersections," Transportation Research Part C: Emerging Technologies, vol. 36, pp. 177-195, 2013.

[20] A. Oroojlooy, M. Nazari, D. Hajinezhad, and J. Silva, "Attendlight: Universal attention-based reinforcement learning model for traffic signal control," Advances in Neural Information Processing Systems, vol. 33, pp. 4079-4090, 2020.

[21] H. Wei, G. Zheng, H. Yao, and Z. Li, "Intellilight: A reinforcement learning approach for intelligent traffic light control," in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pp. 2496-2505, 2018.

[22] S. Bouktif, A. Cheniki, A. Ouni, and H. El-Sayed, "Deep reinforcement learning for traffic signal control with consistent state and reward design approach," Knowledge-Based Systems, vol. 267, p. 110440, 2023.

[23] L. Floridi and M. Chiriatti, "GPT-3: Its nature, scope, limits, and consequences," Minds and Machines, vol. 30, pp. 681-694, 2020.

[24] OpenAI, "Introducing ChatGPT." https://openai.com/blog/chatgpt/ 2023.

[25] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al., "Llama: Open and efficient foundation language models," arXiv preprint arXiv:2302.13971, 2023.

[26] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al., "Llama 2: Open foundation and fine-tuned chat models," arXiv preprint arXiv:2307.09288, 2023.
[27] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances in neural information processing systems, vol. 30, 2017.

[28] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al., "Training language models to follow instructions with human feedback," Advances in Neural Information Processing Systems, vol. 35, pp. 27730-27744, 2022.

[29] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., "Chain-of-thought prompting elicits reasoning in large language models," Advances in Neural Information Processing Systems, vol. 35, pp. 24824-24837, 2022.

[30] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "React: Synergizing reasoning and acting in language models," arXiv preprint arXiv:2210.03629, 2022.

[31] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, et al., "A survey of large language models," arXiv preprint arXiv:2303.18223, 2023.

[32] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou, et al., "The rise and potential of large language model based agents: A survey," arXiv preprint arXiv:2309.07864, 2023.

[33] C. Cui, Y. Ma, X. Cao, W. Ye, Y. Zhou, K. Liang, J. Chen, J. Lu, Z. Yang, K.-D. Liao, et al., "A survey on multimodal large language models for autonomous driving," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 958-979, 2024.

[34] C. Cui, Y. Ma, X. Cao, W. Ye, and Z. Wang, "Receive, reason, and react: Drive as you say with large language models in autonomous vehicles," arXiv preprint arXiv:2310.08034, 2023.

[35] L. Da, K. Liou, T. Chen, X. Zhou, X. Luo, Y. Yang, and H. Wei, "OpenTI: Open traffic intelligence with augmented language model," arXiv preprint arXiv:2401.00211, 2023.

[36] D. Fu, X. Li, L. Wen, M. Dou, P. Cai, B. Shi, and Y. Qiao, "Drive like a human: Rethinking autonomous driving with large language models," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 910-919, 2024.

[37] S. Sharan, F. Pittaluga, M. Chandraker, et al., "LLM-Assist: Enhancing closed-loop planning with language-based reasoning," arXiv preprint arXiv:2401.00125, 2023.

[38] V. Dewangan, T. Choudhary, S. Chandhok, S. Priyadarshan, A. Jain, A. K. Singh, S. Srivastava, K. M. Jatavallabhula, and K. M. Krishna, "Talk2BEV: Language-enhanced bird's-eye view maps for autonomous driving," arXiv preprint arXiv:2310.02251, 2023.

[39] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, and E. Wießner, "Microscopic traffic simulation using sumo," in 21 st international conference on intelligent transportation systems (ITSC), pp. 2575-2582, IEEE, 2018.

[40] Z. Chu, J. Chen, Q. Chen, W. Yu, T. He, H. Wang, W. Peng, M. Liu, B. Qin, and T. Liu, "A survey of chain of thought reasoning: Advances, frontiers and future," arXiv preprint arXiv:2309.15402, 2023.

</end of paper 3>


<paper 4>
# LaRge LanguAGe ModelS FOR HuMAN-MACHine Collaborative PARTICLE ACCELERATOR TUNING THROUGH NATURAL LANGUAGE 

A PREPRINT<br>$\bigcirc$ Jan Kaiser ${ }^{* 1}, \odot$ Annika Eichler ${ }^{\dagger} 1,2$, and $\odot$ Anne Lauscher ${ }^{\# \beta}$<br>${ }^{1}$ Deutsches Elektronen-Synchrotron DESY, Germany<br>${ }^{2}$ Hamburg University of Technology, 21073 Hamburg, Germany<br>${ }^{2}$ Universität Hamburg, Germany

14 May 2024


#### Abstract

Autonomous tuning of particle accelerators is an active and challenging field of research with the goal of enabling novel accelerator technologies cutting-edge high-impact applications, such as physics discovery, cancer research and material sciences. A key challenge with autonomous accelerator tuning remains that the most capable algorithms require an expert in optimisation, machine learning or a similar field to implement the algorithm for every new tuning task. In this work, we propose the use of large language models (LLMs) to tune particle accelerators. We demonstrate on a proof-ofprinciple example the ability of LLMs to successfully and autonomously tune a particle accelerator subsystem based on nothing more than a natural language prompt from the operator, and compare the performance of our LLM-based solution to state-of-the-art optimisation algorithms, such as Bayesian optimisation (BO) and reinforcement learning-trained optimisation (RLO). In doing so, we also show how LLMs can perform numerical optimisation of a highly non-linear real-world objective function. Ultimately, this work represents yet another complex task that LLMs are capable of solving and promises to help accelerate the deployment of autonomous tuning algorithms to the day-to-day operations of particle accelerators.


Keywords Large language models - Autonomous particle accelerators $\cdot$ Multi-objective optimisation

## 1 Introduction

Particle accelerators are sophisticated machines designed to accelerate subatomic particles, such as electrons and protons, to extremely high speeds, often close to the speed of light. These devices play a crucial role in a variety of applications, ranging from fundamental research in physics to practical uses in medicine, such as cancer therapy, and material science. As the demands from these diverse applications grow, there is an increasing need for advanced tuning and control methods to manage the complex dynamics of particle acceleration. Despite this, as a result of its complexity, the tuning of particle accelerators is to this day often done manually by experienced human operators. In this context, the emergence of autonomous tuning methods represents a significant advancement. By leveraging methods from the fields of numerical optimisation and machine learning [Emery et al., 2003. Roussel et al., 2023a. Kaiser et al., 2022], autonomous systems promise to speed up accelerator tuning procedures, reducing costs and minimising downtime, while also enabling novel operating modes for state-of-the art measurements. Moreover, such methods enable a paradigm shift from actuator-driven accelerator operation, where human operators control actuator settings to achieve good measurement conditions, to specification-driven operation, where human operators determine the best conditions for[^0]experiments and autonomous agents ensure that these conditions are achieved. As such, autonomous particle accelerator tuning methods promise to not only improve the performance of accelerators on existing applications but also open up new possibilities in scientific research and industrial applications, marking a transformative step in the field of particle acceleration.

However, implementing state-of-the-art accelerator tuning methods on new tuning tasks requires experts in two separate domains - accelerator physics and optimisation - as well as significant engineering effort to solve problems ranging from algorithm selection to objective function formulation. These challenges have so far slowed the adoption of advanced autonomous tuning algorithms to day-to-day accelerator operations.

In recent developments, LLMs, such as GPT 4 [OpenAI et al., 2023] and Llama 2 [Touvron et al., 2023], have been demonstrated to be capable of solving complex tasks when prompted through natural language [Brown et al. 2020, OpenAI et al., 2023, Oulianov et al., 2024]. The question arises whether LLMs can directly perform particle accelerator tuning, when prompted by an accelerator expert describing the tuning goal. If capable, this would provide a more natural way of controlling autonomous tuning solutions through natural language, potentially enabling a more straightforward deployment of autonomous particle accelerator tuning solutions, and removing the requirement for optimisation algorithm-specific expertise. Moreover, the ability of LLMs to explain their reasoning [Wei et al. 2023] could provide valuable insights into the complex dynamics of particle accelerators, potentially aiding human operators in understanding the tuning process. Lastly, the successful application of LLMs to particle accelerator tuning would also demonstrate the ability of LLMs to solve (multi-objective) numerical optimisation problems, possibly opening up new avenues for the application of LLMs to optimisation tasks beyond particle accelerators.

In this work, we introduce a novel approach to using LLMs for autonomous tuning of a particle accelerator. We answer whether current state-of-the-art LLMs are in fact capable of solving particle accelerator tuning tasks and evaluate our LLM-based approach against the current state of the art in accelerator tuning using RLO and BO.

To this end, we review related work in Section 2, before introducing our approach for autonomous tuning of a particle accelerator using LLMs and our prompt design in Section 3 In Section 4 , we evaluate the developed solution, comparing 14 state-of-the-art LLM models against each other; against 3 state-of-the-art accelerator tuning solutions, RLO, BO and extremum seeking (ES); as well as against two baselines, random search and doing nothing. Our findings indicate that LLMs are capable of tuning particle accelerators, but do not yet achieve performance competitive with the state of the art. We conclude this paper and discuss opportunities for future applications of LLMs in the operation of particle accelerator facilities in Section 5 .

## 2 Related Work

Initial efforts towards autonomous accelerator tuning have investigated numerical methods such as Nelder-Mead simplex [Emery et al., 2003, Shang and Borland, 2005, Huang, 2018], robust conjugate direction search (RCDS) |Huang et al., 2013, Olsson et al., 2018, Zhang et al., 2022a], extremum seeking (ES) [Scheinker et al., 2022] and genetic algorithms [Bergan et al. 2019]. These methods have since found adaptation in the day-to-day tuning of particle accelerator facilities [Tomin et al., 2016, Zhang, 2021, Zhang et al., 2022b]. More recently, advanced methods like Bayesian optimisation (BO) have found increased interest in the accelerator community [Roussel et al., 2023a] for their ability to utilise machine learning to learn a probabilistic surrogate model of the underlying objective function, enabling more sample-efficient tuning of high-dimensional and increasingly complex accelerator systems. Efforts are currently under way to lower the barrier of entry to these methods and increase their adoption in day-to-day accelerator operations [Roussel et al. 2023b]. Moreover, the accelerator community is looking increasingly to machine learning methods to aid with the challenges of accelerator tuning [Edelen et al., 2018]. In particular, reinforcement learning (RL) has found adoption to accelerator control tasks [Boltz et al.| 2020.| St. John et al., 2021]. RL has also been successfully applied to so-called reinforcement learning-trained optimisation (RLO), where neural network (NN) policies are trained through optimiser learning [Andrychowicz et al., 2016, Li and Malik, 2017a b, Chen et al., 2022] to be capable of sample-efficient accelerator tuning |Kain et al., 2020, Pang et al., 2020, Kaiser et al., 2022, Velotti et al., 2023|.

Most recently, large language models (LLMs) have had a highly visible impact on the field of artificial intelligence (AI) and machine learning (ML). Usually based on the transformer NN architecture, first introduced in Vaswani et al. [2017], these models are trained to perform text completion, such that they develop text understanding and text generation capabilities, which can be exploited to create chatbots. As such, state-of-the-art LLMs like GPT 4 [OpenAI et al., 2023] have been demonstrated to have impressive capabilities, such as text summarisation, but also the ability to solve more complex tasks like coding and general problem solving. The field of LLMs is moving fast and seeing significant investments. Despite their high training cost, many of these models have been released in a short time frame, both commercial and closed in nature, such as GPT 4 [OpenAI et al., 2023], Gemini [Gemini Team et al. 2023] and Claude [Anthropic, 2023], but also numerous open-source (or more specifically open-weights) models, such as Llama

![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-03.jpg?height=553&width=1597&top_left_y=233&top_left_x=253)

Figure 1: Schematic of the EA section of the ARES linear particle accelerator. Quadrupole magnets are shown in red; the vertical and horizontal dipole are shown in blue and turquoise, respectively. The electron beam is shown as a green envelop passing through the magnets and onto the screen at the end of the experimental area.

(2) [Touvron et al., 2023], Orca (2) [Mukherjee et al., 2023, Mitra et al., 2023], Starling-LM [Zhu et al., 2023] and Mistral / Mixtral [Jiang et al., 2023, 2024]. Most of these are released in varying sizes with varying trade-offs between capabilities and computational efficiency.

The application of LLMs to optimisation is less prominent in recent research than other applications. Naturally fitting the natural language processing (NLP) origins of LLMs, they have successfully been applied to optimising prompts to LLMs chatbots [Yang et al., 2023]. In further work, LLMs have been used to find more effective algorithms than the state of the art to solve complex problems [Romera-Paredes et al. 2024]. Most similar to our work, the ability of LLMs to solve numerical optimisation has been demonstrated on the simple task of linear regression in Yang et al. [2023]. A benchmark evaluating the performance of different LLMs on a game playing task like those typically solved by training NN policies through RL is presented in Oulianov et al. [2024].

In the context of particle accelerators, there exist ambitions to harness the NLP abilities of LLMs for various purposes. In Sulc et al. [2023], the authors demonstrate how to fine-tune an open-source LLM to be a particle accelerator domain expert using open access scientific literature as training data, augmented by another LLM to generate question-answer pairs from research papers. The fine-tuned model, called PACuna, is shown to be more proficient in answering questions related to particle accelerators. In Mayet [2024a|b], the author demonstrates how off-the-shelf LLMs can be used as a general AI assistant for intelligent accelerator operations (GAIA), employing the ReAct [Yao et al., 2023] prompting scheme to enable the LLM to intelligently trigger accelerator operation routines, automatically contact experts when needed, research questions in the facility's logbook, provide the correct control system addresses for actuators and sensors of the accelerator, and write weekly shift reports.

## 3 Tuning Particle Accelerators Through Natural Language

For the purpose of this work, we consider a specific particle accelerator tuning task, namely the transverse beam parameter tuning in the Experimental Area (EA) section of the accelerator research experiment at SINBAD (ARES) linear particle accelerator [Panofski et al., 2021, Burkart et al. 2022] at DESY in Hamburg, Germany. This task has been chosen as it is a well-defined and well-understood task in the accelerator community, and has been extensively studied in the context of autonomous accelerator tuning [Kaiser et al., 2022, 2023, Kaiser and Xu, 2023, Xu et al., 2023]. At the same time, the task is complex enough to be difficult to solve manually and can provide a meaningful benchmark for the capabilities of LLMs in accelerator tuning, yet simple enough such that solutions can still be easily understood and evaluated. Solving it would provide a valuable streamlining of accelerator operations because similar transverse tuning tasks can be found at most accelerator facilities and have to be regularly performed during everyday operations.

The EA section is primarily made up of five magnets as shown in Fig. 1. Three of these magnets are quadrupole magnets $Q_{1}, Q_{2}$ and $Q_{3}$, which are used to focus the beam, and two are dipole magnets $C_{v}$ and $C_{h}$, which are used to bend the beam, one in the vertical and one in the horizontal plane. In this work, we control the focusing strength $k$ of the quadrupole magnets in $\mathrm{m}^{-2}$ and the angle $\alpha$ by which the dipole magnets deflect particles in mrad. Note that turning up the strength of a quadrupole magnet will focus the beam in the horizontal plane and defocus it in the vertical plane, while turning down the strength will have the opposite effect. Increasing the steering angle of the
vertical steering magnet will steer the beam upwards, while decreasing the angle will steer the beam downwards. The horizontal steering magnet works similarly, steering the beam to the right when the angle is increased and to the left when the angle is decreased. What is more, quadrupole magnets also have a dipole effect on the beam, if the beam passes through the off-centre, making any tuning task involving them more complex. The magnets are arranged in the order $\left(Q_{1}, Q_{2}, C_{v}, Q_{3}, C_{h}\right)$. At the end of the EA section, there is a diagnostic screen station. At the screen station, a screen made of a scintillating material is inserted into the beam pipe. When electrons pass through the screen, light is emitted, which is then captured by a camera and used to measure a transverse projection of the beam. Transverse beam parameters of beam position $\mu_{x, y}$ and beam size $\sigma_{x, y}$ can then be computed from the screen image by fitting a 2D Gaussian distribution. The goal of the tuning task is to find a set of magnet settings ( $\left.k_{Q_{1}}, k_{Q_{2}}, \alpha_{C_{v}}, k_{Q_{3}}, \alpha_{C_{h}}\right)$ that minimise the difference between the measured beam parameters $\left(\mu_{x}, \sigma_{y}, \mu_{y}, \sigma_{y}\right)$ and some target beam parameters $\left(\mu_{x}^{\prime}, \sigma_{y}^{\prime}, \mu_{y}^{\prime}, \sigma_{y}^{\prime}\right)$ set by the human operator.

### 3.1 Optimisation Scheme

We consider an iterative optimisation scheme for accelerator tuning, where the state of the accelerator is observed and then the tuning algorithm chooses new actuator settings based on the current and all past states from the tuning run. This process is repeated either for a fixed number of iterations or until some termination criterion is met. For an LLM to act as the tuning or optimisation algorithm, a prompting scheme needs to be devised. In our approach we consider the use of a chatbot LLM, where the user can provide a question or command to the LLM and the LLM will respond with an answer. Our optimisation scheme using LLMs extends on the approach for linear regression presented in Yang et al. [2023] and is shown in Fig. 2. In the prompt to the LLM, the user provides a description of the optimisation task that the LLM should solve. This is followed by a list of input and output pairs from previous optimisation steps. After this list, the user asks for the next set of input parameters that help optimise the objective function and gives the LLM instructions on how these should be formatted such that the user can parse the output from text to numerical values. This prompt is then sent to the LLM, which responds with the next set of input parameters that should be used to optimise the objective function, and potentially also an explanation of why these parameters were chosen. The response should look similar to the one below:

```
```json
{
    "Q1": -14.30,
    "Q2": -9.70,
    "CV": -2.55,
    "Q3": -8.10,
    "CH": -5.21
}
I suggest decreasing Q1 slightly to bring down the horizontal beam position, while keeping
the other quadrupole magnets at their previous values to maintain the vertical beam position
and focusing. I also kept the steering magnet settings close to their last values for
smoothness.
```

The response is then parsed, and the input parameters are used to evaluate the objective function. The output of this evaluation is then added to the list along with its corresponding input parameters, and the process is repeated.

Prompt engineering is a crucial part of using LLMs and can significantly impact the performance of the model. Because of the variability in the performance of different prompts and the difficulty of finding the best prompt, we evaluate the ability of LLMs to solve the accelerator tuning task using four different prompts: Tuning Prompt (see Section 3.1.1), Explained Prompt (see Section 3.1.2), Chain-of-Thought Prompt (see Section 3.1.3) and Optimisation Prompt (see Section 3.1.4). All prompts follow the general prompting scheme described above, of task description, input-output pairs, request for next input parameters and instructions on how to format the output. The prompts used in this work differ mainly in the task description and the outputs of the previous optimisation steps.

### 3.1.1 Tuning Prompt

The Tuning Prompt is the most straightforward and intuitive prompt used in this work. It describes the task of tuning the transverse beam parameters in the EA section and the goal of achieving some target beam parameters on the diagnostic screen, such that the LLM is aware of the fact it is tuning a particle accelerator. The input-output pairs are the magnet settings and the corresponding measured beam parameters. This prompt assumes that the LLM has some understanding of particle accelerators and understands, for example, what a quadrupole magnet is and how it affects the beam. Below

![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-05.jpg?height=410&width=1109&top_left_y=234&top_left_x=497)

Figure 2: Flowchart of the optimisation scheme used to tune particle accelerators using LLMs. The prompt is made up for three components: Task description, list of previous input and output samples, and instructions what to output and how to format the output. The prompt is then sent to the LLM, which generates a response. The response is parsed into values that can be input into the tuning / optimisation task. A measurement or objective value from the task is then inserted into the previous samples along with its corresponding input and the loop is repeated.

is an example of the Tuning Prompt, where the task description is highlighted in orange, the input-output pairs in blue, and the request for the next input parameters and output instructions in green:

```
Human: Now you will help me optimise the horizontal and vertical position and size of an
electron beam on a diagnostic screen in a particle accelerator.
You are able to control five magnets in the beam line. The magnets are called Q1, Q2, CV, Q3
and CH.
Q1, Q2 and Q3 are quadrupole magnets. You are controlling their k1 strength in m^-2. Their
range is - 30.0 to 30.0 m -2.
CV is vertical steering magnet. You control its steering angle in mrad. Its range is -6.0 to
6.0 mrad.
CH is horizontal steering magnet. You control its steering angle in mrad. Its range is -6.0
to 6.0 mrad.
You are optimising four beam parameters: mu_x, sigma_x, mu_y, sigma_y. The beam parameters
are measured in millimetres (mm). The target beam parameters are:
Target beam parameters:
`.`.json
{
    "mu_x": 1.20,
    "sigma_x": 0.11,
    "mu_y": 1.25,
    "sigma_y": 0.06
}
-..
```

Below are previously measured pairs of magnet settings and the corresponding observed beam parameters.

Magnet settings :

- json

\{

"Q1": 25.12 ,

"Q2": 12.48,

"CV": 0.84,

"Q3": -8.25,

"CH": 3.94

\}

```
`-.
Beam parameters:
```json
{
    "mu_x": -1038.63,
    "sigma_x": 1893.75,
    "mu_y": -2353.77,
    "sigma_y": 2226.94
}
-. 
```

Give me new magnet settings that are different from all pairs above. The magnet settings you should propose should lead to beam parameters closer the target or, if you do not have enough information yet, maximise information gain for finding new beam parameters. Do not set any magnet setting to zero. Smooth changes relative to the last magnet settings are preferred.

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "`.`.json" and "…":

## \{

"Q1": float // k1 strength of the first quadrupole magnet

"Q2": float // k1 strength of the second quadrupole magnet

"CV": float // Deflection angle of the vertical steering magnet

"Q3": float // k1 strength of the third quadrupole magnet

"CH": float // Deflection angle of the horizontal steering magnet

\}
-

Do not add comments to the output JSON.

Note that the choice was made to provide previously observed magnet settings and beam parameters formatted as a markdown JSON snippet. We found that if these are provided as a simple textual list of property names and their values, the LLMs would often output the next magnet settings in the same format instead of the requested JSON format. By providing the examples in the same format as we desire for the output, the parsing reliability of the LLM is increased significantly.

### 3.1.2 Explained Prompt

The Explained Prompt is mostly the same as the Tuning Prompt, but includes additional explanations of how each of the magnets affects the beam. This is done because accelerator physics is a complex and niche field, which is unlikely to have been widely covered in the training data of a general-purpose LLMs. The explanations are generally kept on a high level, similar to how one might explain the task to a new accelerator operator in order to give them an intuition of how the magnets affect the beam on the diagnostic screen. Below is an example of the Explained Prompt with the explanations added over the Tuning Prompt highlighted in violet:[^1]$\mathrm{CV}$ is vertical steering magnet. When its deflection angle is increased, the beam is steered upwards. When its deflection angle is decreased, the beam is steered downwards. The range of the deflection angle is -6.0 to 6.0 mrad.

$\mathrm{CH}$ is horizontal steering magnet. When its deflection angle is increased, the beam is steered to the right. When its deflection angle is decreased, the beam is steered to the left. The range of the deflection angle is -6.0 to $6.0 \mathrm{mrad}$.

You are optimising four beam parameters: mu_x, sigma_x, mu_y, sigma_y. The beam parameters are measured in millimetres (mm). The target beam parameters are:

```
Target beam parameters:
```json
{
    "mu_x": 1.20,
    "sigma_x": 0.11,
    "mu_y": 1.25,
    "sigma_y": 0.06
}
```

Below are previously measured pairs of magnet settings and the corresponding observed beam parameters.

Magnet settings:

- `json

\{

"Q1": 25.12,

"Q2": 12.48,

"CV": 0.84,

"Q3": -8.25,

"CH": 3.94

\}

Beam parameters:

``json

\{

"mu_x": -1038.63,

"sigma_x": 1893.75,

"mu_y": -2353.77,

"sigma_y": 2226.94

\}

. .

Give me new magnet settings that are different from all pairs above. The magnet settings you should propose should lead to beam parameters closer the target or, if you do not have enough information yet, maximise information gain for finding new beam parameters. Do not set any magnet setting to zero. Smooth changes relative to the last magnet settings are preferred.

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "`..json" and "...":

. . json

\{

"Q1": float // k1 strength of the first quadrupole magnet

"Q2": float // k1 strength of the second quadrupole magnet

"CV": float // Deflection angle of the vertical steering magnet

"Q3": float // k1 strength of the third quadrupole magnet

"CH": float // Deflection angle of the horizontal steering magnet

}

Do not add comments to the output JSON.

### 3.1.3 Chain-of-Thought Prompt

Chain-of-Thought (CoT) prompting [Wei et al. 2023] is a technique where the user asks the LLM to explain its reasoning before it gives its answer. This was found to generally improve the quality of the answers given by LLMs, especially in the case of logical reasoning tasks. Note that it is important to have the explanation before the answer, as otherwise the model will phrase the explanation in support of the already given and potentially incorrect answer, thereby negating the benefit of chain-of-thought prompting. In the Chain-of-Thought Prompt, we add a request to the prompt whereby the users asks the LLM to explain its reasoning before it gives the next set of input parameters. Otherwise, the Chain-of-Thought Prompt is the same as the Explained Prompt. Below is an example of the Chain-of-Thought Prompt, where the request for chain-of-thought reasoning is highlighted in violet:

Human: Now you will help me optimise the horizontal and vertical position and size of an electron beam on a diagnostic screen in a particle accelerator.

You are able to control five magnets in the beam line. The magnets are called Q1, Q2, CV, Q3 and $\mathrm{CH}$.

Q1, Q2 and Q3 are quadrupole magnets. When their k1 strength is increased, the beam becomes more focused in the horizontal plane and more defocused in the vertical plane. When their $\mathrm{k} 1$ strength is decreased, the beam becomes more focused in the vertical plane and more defocused in the horizontal plane. When their k1 strength is zero, the beam is not focused in either plane. Quadrupole magnets might also steer the beam in the horizontal or vertical plane depending on their k0 strength, when the beam does not travel through the centre of the magnet. The range of the k1 strength is -30.0 to $30.0 \mathrm{~m}^{\wedge}-2$.

CV is vertical steering magnet. When its deflection angle is increased, the beam is steered upwards. When its deflection angle is decreased, the beam is steered downwards. The range of the deflection angle is -6.0 to $6.0 \mathrm{mrad}$.

$\mathrm{CH}$ is horizontal steering magnet. When its deflection angle is increased, the beam is steered to the right. When its deflection angle is decreased, the beam is steered to the left. The range of the deflection angle is -6.0 to $6.0 \mathrm{mrad}$.

You are optimising four beam parameters: mu_x, sigma_x, mu_y, sigma_y. The beam parameters are measured in millimetres (mm). The target beam parameters are:

```
Target beam parameters:
```json
{
    "mu_x": 1.20,
    "sigma_x": 0.11,
    "mu_y": 1.25,
    "sigma_y": 0.06
}
```

Below are previously measured pairs of magnet settings and the corresponding observed beam parameters.

Magnet settings:

-. json

\{

"Q1": 25.12,

"Q2": 12.48,

"CV": 0.84,

"Q3": -8.25,

"CH": 3.94

\}

.

```
Beam parameters:
-``json
{
    "mu_x": -1038.63,
    "sigma_x": 1893.75,
    "mu_y": -2353.77,
    "sigma_y": 2226.94
}
Give me new magnet settings that are different from all pairs above. The magnet settings
you should propose should lead to beam parameters closer the target or, if you do not have
enough information yet, maximise information gain for finding new beam parameters. Do not
set any magnet setting to zero. Smooth changes relative to the last magnet settings are
preferred.
First, reason about how and why you would change the magnet settings in a certain direction.
Then give me the proposed magnet settings afterwards.
The output should be a markdown code snippet formatted in the following schema, including
the leading and trailing "`.`json" and "..":
`-`json
{
    "Q1": float // k1 strength of the first quadrupole magnet
    "Q2": float // k1 strength of the second quadrupole magnet
    "CV": float // Deflection angle of the vertical steering magnet
    "Q3": float // k1 strength of the third quadrupole magnet
    "CH": float // Deflection angle of the horizontal steering magnet
}
. .
```

Do not add comments to the output JSON

### 3.1.4 Optimisation Prompt

The Optimisation Prompt phrases the task as a numerical optimisation problem instead of a particle accelerator tuning task. This means that the model is completely unaware that it is tuning a particle accelerator. Numerical optimisation tasks are more generic than particle accelerator tuning tasks and therefore expected to be more present in the training data used to train LLMs, meaning that models are likely to be more familiar with them. However, this also means that the model is given no information about the topology of the objective function, which makes the optimisation problem harder to solve. The objective function is therefore a black box to the model. The input-output pairs are the magnet settings and a corresponding single scalar objective value computed from the beam parameters as

$$
\begin{equation*}
\text { objective }=\left|\mu_{x}-\mu_{x}^{\prime}\right|+\left|\mu_{y}-\mu_{y}^{\prime}\right|+\left|\sigma_{x}-\sigma_{x}^{\prime}\right|+\left|\sigma_{y}-\sigma_{y}^{\prime}\right| \tag{1}
\end{equation*}
$$

Below is an example of the Optimisation Prompt, where the task description is highlighted in orange, the input-output pairs in blue, and the request for the next input parameters and output instructions in green:

```
Human: Now you will help me minimise a function with five input variables Q1, Q2, CV, Q3 and
CH. I have some (Q1, Q2, CV, Q3, CH) pairs and the corresponding function values at those
points. The samples are arranged in descending order based on their function values, where
lower values are better.
Inputs:
``json
{
    "Q1": -13.50,
    "Q2": -9.00,
    "CV": -3.00,
```

```
"Q3": -9.00,
    "CH": -6.00
}
Objective value = 2.37
Inputs:
```json
{
    "Q1": -13.25,
    "Q2": -8.85,
    "CV": -2.80,
    "Q3": -8.90,
    "CH": -5.70
}
Objective value = 2.28
```

Give me a new sample (Q1, Q2, CV, Q3, CH) that is different from all pairs above, and has a

function value lower than any of the above.

The output should be a markdown code snippet formatted in the following schema, including

the leading and trailing "```json" and "`.'":

```json

{

"Q1": float // First input

"Q2": float // Second input

"CV": float // Third input

"Q3": float // Fourth input

"CH": float // Fifth input

}

`. 

\section*{4 Evaluation}

In order to evaluate whether LLMs using the prompting scheme described in Section 3 are capable of solving particle accelerator tuning tasks, we compare the performance of multiple state-of-the-art LLMs against each other and against other state-of-the-art accelerator tuning solutions using RLO and BO. In addition, we consider some further baselines for our comparison, specifically ES, random search and doing nothing. The evaluation setup is introduced in Section 4.1. followed by the evaluation results in Section 4.2.

\subsection*{4.1 Method}

We evaluate each of the models and prompts on three different instances of the EA transverse beam parameter tuning task described in Section 3 We call these instances trials. Trials differ in the target beam parameters set by the human operator, the transverse misalignments of the quadrupole magnets and the diagnostic screen, the properties of the beam entering the EA section from upstream, and the initial magnet settings before the respective tuning algorithm has taken any action. For each trial, we run each model and prompt three times with different random seeds to account for the stochasticity of the LLMs and some of the baseline algorithms.

Performance is evaluated in terms of the mean absolute error (MAE) between the measured beam parameters and the target beam parameters after 50 iterations. This tests the ability of the models to find a good set of magnet settings. We further consider the normalised MAE improvement from the initial magnet settings to the final magnet settings found by the model, which tests the ability of the models to improve the beam parameters from the initial settings. Normalisation by dividing the MAE improvement by the MAE with the initial magnet settings makes this metric less sensitive to the inherent variability and difficulty of different trials. Finally, we consider the normalised MAE over all interactions, which tests the ability of the models to find a good set of magnet settings quickly. Here, too, the impact of trial-to-trial variations is reduced by dividing by the accumulated MAE of keeping the magnet settings the same as the initial settings for 50 iterations. For all LLMs, we further consider the number of consecutive steps for which they are
able to generate a parsable JSON output, which tests the tests the models' reliability in generating a valid output. LLMs are given a second chance in each sample, if they fail to generate a parsable JSON output on the first attempt.

The main goal of this work is not to determine whether LLMs are capable of outperforming the current state of the art in accelerator tuning algorithms. In fact, we expect that the current state of the art in accelerator tuning algorithms, such as RLO and BO, clearly outperform LLMs. Instead, we hope to determine whether LLMs are capable of solving accelerator tuning (and by extension other complex optimisation tasks) at all, and to what extent they can do so. We therefore also introduce three measures of "success", where we consider a tuning run successful, if the final beam difference is at least is $40 \mu \mathrm{m}$ improved over the initial beam difference before any tuning has taken place, with $40 \mu \mathrm{m}$ being twice the real-world measurement accuracy for beam parameters on the diagnostic screen. This means that runs are only considered successful, if a clearly measurable improvement of the beam parameters has been achieved. A tuning algorithm is considered "outright successful", if it is able to achieve the success criteria in all 9 evaluation runs. We consider a tuning algorithm as "partially successful" if it is able to achieve the success criterion in at least 6 of the 9 evaluation runs. Partial success suggests that, while not perfectly reliable, successful runs probably not coincidental. We further know that some trials can be harder to solve than others. As a third and weakest success criterion, we therefore consider a tuning algorithm as "single trial successful" if it is able to achieve the success criterion in on all three runs of a single trial, suggesting that, while some trials may have been too difficult to solve, the model was able to reliably solve this one trial.

For this work we evaluate a total of 14 different LLMs. These are Gemma $2 B$ and Gemma $7 B$ Gemma Team et al., 2024] (version 1.0); GPT 3.5 Turbo (checkpoint 0125) [OpenAI, 2023], GPT 4 [OpenAI et al., 2023] (checkpoint 0613) and GPT 4 Turbo (preview checkpoint 0125) [OpenAI|,2023]; Llama 2 7B, Llama 2 13B and Llama 2 70B [Touvron et al., 2023], as well as the fine-tuned variants of Llama 2: Orca 2 7B and Orca 2 13B [Mukherjee et al., 2023, Mitra et al., 2023], and Vicuna 7B 16K [Zheng et al., 2023]; Mistral 7B (version 0.2) [Jiang et al., 2023] and Mixtral $8 \times 7 B$ [Jiang et al. 2024]; and Starling-LM 7B (beta) [Zhu et al. 2023]. The Explained Prompt and Optimisation Prompt are evaluated with all models, while the Tuning Prompt and Chain-of-Thought Prompt are evaluated only with Gemma 2B, GPT 4 Turbo and Mixtral 8x7B.

Prompt generation and response parsing are implemented using the LangChain [Chase, 2022] Python package, which provides a straightforward set of tools for constructing prompts, calling LLMs and parsing their responses. The open-weights LLMs used in this work are run using Ollama [Ollama Team, 2023], while the OpenAI models are run through the OpenAI API. All models are run using their default temperature value, with $T=0.7$ for the OpenAI models and $T=0.8$ for all other models. Orca 2 7B, Orca 2 13B and Vicuna 7B 16K are run with their default system prompts as listed in Appendix A, All other models are run without any system prompts as per their default configuration. A Gymnasium [Farama Foundation, 2022] environment of the EA transverse beam parameter tuning task using the Cheetah [Stein et al.| 2022|| Kaiser et al. 2024] beam dynamics simulator is used to evaluate the LLMs and baselines. The baselines BO, ES and random search are implemented following Kaiser et al. [2023]. The RLO and do nothing baselines are implemented according to Kaiser et al. [2022], using the trained policy model from that work.

\subsection*{4.2 Results}

The results of the evaluation in terms of the three previously defined metrics are shown in Table 1 . The number of successful runs and wholly successful trials for each model and prompt are shown in Fig. 3. Two example tuning runs by a well-performing and a poorly-performing model and prompt combination are shown in Fig. 4 .

We find that the state-of-the-art tuning algorithms RLO and BO, as well as ES, all achieve the strictest success criterion of outright success in all 9 evaluation runs. Of the LLM prompt combinations evaluated, GPT 3.5 Turbo, GPT 4 and GPT 4 Turbo in combination with the Optimisation Prompt also achieve outright success in all 9 evaluation runs, with GPT 4 Turbo with the Optimisation Prompt also being the best-performing LLM prompt combination in all evaluated metrics. In addition, a further 10 LLM prompt combinations achieve partial success, with Llama 2 13B, Llama 2 70B and Orca 2 7B doing so with the Optimisation Prompt; Gemma 7B, Mixtral 8x7B and Starling LM 7B achieving partial success with the Explained Prompt; Gemma 2B and Mixtral 8x7B achieving partial success with the Tuning Prompt; and Gemma 2B and GPT 4 Turbo achieving partial success with the Chain-of-Thought Prompt. Overall, Mixtral 8x7B is the best performing model with the Explained Prompt, but is outperformed by Starling LM 7B on the Final Beam Difference metric. With the Tuning Prompt, Mixtral 8x7B performs best of the three evaluated models, while Gemma 2B is the best-performing model with the Chain-of-Thought Prompt. All models that achieve partial success also achieve single trial success in at least one trial, demonstrating that they are able to solve some trials reliably. A further 6 LLM prompt combinations achieve single trial success: Gemma 2B and GPT 4 with the Explained Prompt, Mixtral 8x7B with the Optimisation Prompt, and Mixtral 8x7B with the Chain-of-Thought Prompt. In total, of the 34 LLM prompt combinations tried, 18 achieve at least one success criterion. Of 14 LLMs evaluated, 10 achieve at least one success criterion with at least one prompt. This demonstrates that LLMs can be used to solve accelerator tuning tasks.

![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-12.jpg?height=2285&width=1173&top_left_y=229&top_left_x=476)

![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-13.jpg?height=1133&width=1477&top_left_y=249&top_left_x=324)

Figure 3: Number of successful runs for each model and prompt (a) and the number wholly successful trials, i.e. trials where all three runs were successful (b). We define as success an improvement of at least $40 \mu \mathrm{m}$ on the beam differences when compared to the initial magnet settings.

However, these results also show that LLMs are not yet competitive with the state-of-the-art accelerator tuning algorithms. The best-performing LLM prompt combination, GPT 4 Turbo with the Optimisation Prompt, achieves an average normalised beam improvement of $-50 \%$. This is a good result, but it is also a significantly worse result than the $-99 \%$ and $-93 \%$ achieved by RLO and BO, respectively. A similar trend can be observed in how fast algorithms are able to find a good set of magnet settings. GPT 4 Turbo with the Optimisation Prompt achieves an average normalised integrated MAE of $67 \%$, which is an order of magnitude worse than the $3 \%$ achieved by RLO. However, it is only about two times worse than BO and ES.

What is more, the results show that the performance of LLMs is highly dependent on the specific model and prompt used. While 18 of the 34 LLM prompt combinations tried achieve at least one success criterion, the remaining 16 do not achieve any. Similarly, 4 of the evaluated LLMs do not achieve any success criterion with any of the prompts they were tested on. We observe that in general, the Optimisation prompt performs best in our evaluations. Outright success was only achieved with the Optimisation Prompt, and at least one success criterion was achieved by 7 LLMs when using the Optimisation Prompt, while only 5 LLMs achieve at least one success criterion with the Explained Prompt. The best-performing LLM prompt combination, GPT 4 Turbo with the Optimisation Prompt, also uses the Optimisation prompt. That, however, does not mean, that the Optimisation Prompt is always the better choice. Some models perform better with one of the other prompts. Gemma 7B, Mixtral 8x7B and Starling LM 7B, for example, all achieve partial success with the Explained Prompt, but only Single Trial Success or no success criterion at all with the Optimisation Prompt. Similarly, Gemma 2B and Mixtral 8x7B achieve their best results with the Tuning Prompt. We conclude that the choice of prompt must be made on a model-by-model basis.

It is also worth noting that adding explanations to the prompts about how the magnets work, or adding a chain-ofthought to the prompts, does not always lead to the expected improvements. Of the three models evaluated with all four prompts, only GPT 4 Turbo improves with the addition of explanations. However, this is with GPT 4 Turbo generally

![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-14.jpg?height=1120&width=1635&top_left_y=237&top_left_x=234)

GPT 4 Turbo (Optimisation Prompt)
![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-14.jpg?height=1050&width=870&top_left_y=290&top_left_x=259)

GPT 3.5 Turbo (Explained Prompt)
![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-14.jpg?height=1052&width=730&top_left_y=294&top_left_x=1138)

Figure 4: Magnet setting and beam parameter traces for a good and a bad tuning run by LLMs. Both runs used the same trial, where the target beam parameters are $\mu_{x}=\mu_{y}=\sigma_{x}=\sigma_{y}=0 \mathrm{~mm}$.

performing badly on any of the three variants of the Tuning Prompt, generally performing better with the Optimisation Prompt. Gemma 2B and Mixtral 8x7B, on the other hand, perform worse when the explanations are added. A possible explanation for this observation is that, rather than helping the model understand the tuning task, the length of the explanations makes it harder for the LLM to retrieve relevant information, such as specific past samples or the target beam parameters, from the prompt. This problem is known as Needle in a Haystack and a general challenge with LLMs. Chain-of-thought prompting seems to improve performance over the Explained Prompt with Gemma 2B and GPT 4 Turbo, but has an adverse effect on the performance of Mixtral 8x7B. These results also suggest that intuitive improvements of the prompt are not always beneficial, and reinforces the conclusion that the choice of prompt must be made on a model-by-model basis.

In designing the presented LLM tuning solution, we found that aside from getting LLMs to successfully tune the particle accelerator, another difficulty is to get them to output the magnet settings in a parsable JSON format. This is why LLMs are given a second chance in each sample, if the parsing of their response fails on the first attempt. Nevertheless, some models fail on the second attempt as well, at which point we consider the tuning run terminated. We can therefore take number of performed iterations (excluding second attempts) as an indicator of a model's ability to produce a valid JSON output when provided with one of our prompts. Note that excluding second attempts, this is the number of interactions with the accelerator, not the number of times the LLM was prompted. The observed number of iterations for the 9 evaluation runs of each model and prompt are shown in Table 1. We observe many models, often those achieving good tuning results, have a high number of successful steps, with models like those by OpenAI and Llama 70B always achieving the maximum of 50 successful steps, regardless of the prompt used. Other models, such as both Orca 2 and the smallest variant of Llama 2, consistently struggle to produce a valid JSON output, with the number of successful steps being very low for either prompt. While in most cases, it appears that the ability to generate valid JSON responses depends mostly on the LLM used, we also observe that the choice of prompt can have an impact in a few cases, with the difference being especially pronounced for the Gemma models, which achieve a higher number of successful steps with the Optimisation Prompt than with the Explained Prompt. It does not appear as though one prompt is generally better than the other in terms of the number of successful steps. Furthermore, the nature of different invalid responses
![](https://cdn.mathpix.com/cropped/2024_06_04_42c473538ab25cc1050fg-15.jpg?height=740&width=1604&top_left_y=234&top_left_x=260)

Figure 5: Number of successful tuning runs, average normalised MAE improvement and average normalised accumulated MAE for each LLM model with respect to its size, LMSYS Chatbot Arena ELO rating, MT-bench score, MMLU score and HellaSwag score. Results for the Explained Prompt are shown in black and results for the Optimisation Prompt are shown in blue. Linear fits are shown for the presented data. We expect the number of successful episodes to increase and the other two metrics to decrease, if model size or high benchmark scores improve the ability of LLMs to solve the investigated particle accelerator tuning task.

varies greatly. In some cases, the mistakes are so minor that human experts might fail to spot them, for example when a trailing comma is added to the last JSON property. This is not allowed in JSON syntax and causes the parser to fail. Another failure case is related to chain-of-thought. For example, Orca 2 - a model specifically trained to respond with chain-of-though - often outputs an explanation of a strategy to solve the optimisation problem rather than the next magnet settings requested in the prompt. Last, but certainly not least, some models fail to output a coherent response altogether, with responses being nonsensical, for example starting the response with an invalid continuation of a JSON object and then continuing with multiple valid JSON objects even though only a single one was requested. In this case, both the invalid JSON object and the ambiguity about which JSON object should be parsed, make the response invalid. Examples of these three described failure modes are given in Appendix B.

It is well known that some LLMs generally perform better than others. Often, an LLM's capabilities are correlated with the number of parameters it has. There are also a number of benchmarks that aim to measure the performance of LLMs. These include the LMSYS Chatbot Arena ELO rating [Zheng et al., 2023], the MT-bench score [Zheng et al., 2023], the Massive Multitask Language Understanding (MMLU) score [Hendrycks et al., 2021] and the HellaSwag score [Zellers et al., 2019]. As can be seen in Fig. 5, the number of successful episodes, normalised beam improvement and normalised integrated MAE are mostly correlated with the number of parameters models have and their benchmark scores, especially for the better-performing Optimisation Prompt. This suggests that the overall performance of an LLM is an indicator for its performance on particle accelerator tuning and general numerical optimisation tasks. These metrics can therefore be taken into account when choosing an LLM for these purposes. This observation further implies that, as increasingly well-performing general purpose LLMs are released, we can expect to see better performance on accelerator tuning and numerical optimisation tasks.

Apart from LLMs' ability to solve a given task, it is also important to consider the fact that LLMs are usually very resource intensive to run. The open-weights models used in this work are run on four NVIDIA A100 GPUs with 80 GB of memory each. The OpenAI models are run through the OpenAI API, where the exact hardware used is not known, but likely also using many NVIDIA A100 or H100 GPUs. In contrast, the state-of-the-art accelerator tuning algorithms RLO and BO can easily be run on laptop CPU, specifically an Apple M1 Pro system on a chip (SOC) for the results presented in this work. An average inference takes less than $200 \mu \mathrm{s}$ for RLO and around $700 \mathrm{~ms}$ for BO. In contrast, the fastest open-weights LLM was Gemma 2B on the Tuning Prompt with an average inference time of $700 \mathrm{~ms}$, while the slowest was Orca 2 13B with 30 s on the Explained Prompt. The particular case of Orca 2 inference being slow is to do with the fact that this model is trained to provide chain-of-thought, which results in long responses. We see similarly long inference times at $28 \mathrm{~s}$ when prompting GPT 4 Turbo with the Chain-of-Thought Prompt. Otherwise,
the OpenAI models achieved between $1 \mathrm{~s}$ for GPT 3.5 Turbo on the Optimisation Prompt and $4 \mathrm{~s}$ for GPT 4 on the Explained Prompt. A large open-weights model like Llama 2 70B, achieved an average inference time of $7 \mathrm{~s}$ on the Optimisation Prompt in our evaluations.

Such large resource demands usually induce high cost. While the actual cost of running LLMs on our own GPUs is difficult to estimate, the cost of running the OpenAI models through the OpenAI API as of 10 April 2024 is around USD 5.35 for one tuning run with GPT 4 and the Explained Prompt, and USD 2.98 for GPT 4 with the Optimisation Prompt. GPT 4 Turbo costs less at around USD 1.81 for a tuning run using the Explained Prompt and USD 0.74 for the Optimisation Prompt. GPT 3.5 Turbo was the cheapest, with API costs of around USD 0.09 and USD 0.05 for the Explained and Optimisation Prompt, respectively. When using prompts that are likely include more than a magnet setting JSON in the response, such as the Chain-of-thought Prompt, the cost of running an optimisation with GPT 4 Turbo increase to USD 2.63.

Considering the large amount of compute resources these models require, we must also consider their energy consumption and associated environmental impact. In Li et al. [2023], the authors find that GPT 3 consumes $500 \mathrm{~mL}$ of water for 10 to 50 responses. For the 50 responses in one evaluated tuning run, this comes out to $0.5 \mathrm{~L}$ to $2.5 \mathrm{~L}$ of water. While the authors do not mention the number of tokens assumed for a response, we can safely assume that these numbers are a lower bound for the much more resource intensive GPT 4 and GPT 4 Turbo models used in this work. To estimate the $\mathrm{CO}_{2}$ emissions associated with using these models for particle accelerator tuning, we can consider Mixtral 8x7B as a representative model somewhat of average size. Taking the average inference time of $6 \mathrm{~s}$ per step with the Optimisation Prompt, this model uses a total of 300 s of GPU time on 4 A100 GPUs. The energy consumption of a single A100 GPU is quoted as $250 \mathrm{~W}$ [NVIDIA, 2020], i.e. $1 \mathrm{~kW}$ for all 4 GPUs, giving a total energy consumption of $83 \mathrm{~W}$ h for one tuning run. This is about the same amount of energy as a modern fridge consumes over $11 \mathrm{~h}$ [Bosch, 2021] or driving a modern electric vehicle for $0.5 \mathrm{~km}$ BMW, 2024], and results in $\mathrm{CO}_{2}$ emissions of about $36 \mathrm{~g}$ [Umweltbundesamt, 2023|. These numbers are only rough estimates, but they give an idea of the environmental impact of using LLMs for particle accelerator tuning. Generally, these should be lower for the smaller open-weights models, but higher for larger models like GPT 4 and GPT 4 Turbo. Note that none of the given numbers consider the environmental impact of training these models, which is substantial. However, as the models are already trained for other purposes and available, we do not take this into account in our evaluation.

\section*{5 Conclusion and Outlook}

In this work, we demonstrated that LLMs can be used to solve accelerator tuning tasks and by extension general numerical optimisation tasks. However, considering a combination of 14 different open-weights and commercial LLMs and 4 different prompts, we find that only 18 of the 34 LLM prompt combinations can successfully achieve an improvement on the transverse beam parameter tuning task considered in this work. We conclude that, while it is generally possible to use LLMs for accelerator tuning, the choice of model and prompt is crucial. Comparing to state-of-the-art accelerator tuning algorithms, we further find that LLMs are not yet competitive with RLO and BO. The best-performing LLM prompt combination, GPT 4 Turbo with the Optimisation Prompt, achieves an average normalised beam improvement of $-50 \%$, which is only about half as good as the $-99 \%$ and $-93 \%$ achieved by RLO and BO, respectively. While not achieving competitive performance, LLMs also incur high computational costs, leading to long inference times, high monetary costs and significant environmental impact.

Despite these clear disadvantages that mean LLMs are not yet a viable alternative to state-of-the-art accelerator tuning algorithms, our results present an intriguing proof of concept. The field of LLMs is rapidly evolving, with ever more capable models being released on a near-daily basis. We have shown that more capable models generally perform better on accelerator tuning tasks, meaning that the inevitable progress in the field of LLMs will also lead to better performance on accelerator tuning tasks. Ultimately such development could make the intuitive deployment of autonomous accelerator tuning solutions through natural language a feasible option.

In the near future, we expect that, instead of being used as a replacement for state-of-the-art accelerator tuning algorithms, LLMs will find applications as copilots to human particle accelerator operators. Here, they can provide a natural language interface to various tasks related to accelerator operations, such as retrieving information from logbooks, generating reports or diagnosing the accelerator's state from large amounts of diagnostic measurements. Such efforts are already underway [Sulc et al. 2023, Mayet, 2024a b]. In continuation of this work, we believe that LLMs could also be used to coordinate state-of-the-art accelerator tuning algorithms, such as RLO and BO, in a federated setting, deciding or helping the operator decide which part of the accelerator to tune next, using which algorithm and with which desired outcome. What is more, LLMs could also be used to assist human operators in the deployment of state-of-the-art tuning algorithms, for example by proposing Xopt [Roussel et al. 2023b] configurations, or objective functions and suitable actuators in response to natural language prompts about the desired outcome. In the longer term,
the approach of letting LLMs perform tuning directly may be improved by using a ReAct prompting scheme Yao et al., 2023] or employing LLMs to check if the magnet settings proposed algorithms like RLO and BO are sensible in a setup similar to Wang et al. [2024], Aoyu Pang and Chen [2023].

\section*{Code availability}

The code used to produce the results presented in this paper is available upon reasonable request to the authors.

\section*{Data availability}

The data underlying the results presented in this paper is available upon reasonable request to the authors.

\section*{Acknowledgements}

This work has in part been funded by the IVF project InternLabs-0011 (HIR3X). The authors acknowledge support from DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, as well as support through the Maxwell computational resources operated at DESY. In addition, the authors would like to thank Frank Mayet and Antonin Sulc for the helpful knowledge exchange on LLMs and the software ecosystem surrounding them.

\section*{References}

L. Emery, M. Borland, and H. Shang. Use of a general-purpose optimization module in accelerator control. In Proceedings of the 2003 Particle Accelerator Conference, volume 4, pages 2330-2332 vol.4, May 2003. doi 10.1109/PAC.2003.1289108

Ryan Roussel, Auralee L. Edelen, Tobias Boltz, Dylan Kennedy, Zhe Zhang, Xiaobiao Huang, Daniel Ratner, Andrea Santamaria Garcia, Chenran Xu, Jan Kaiser, et al. Bayesian optimization algorithms for accelerator physics, 2023a.

Jan Kaiser, Oliver Stein, and Annika Eichler. Learning-based optimisation of particle accelerators under partial observability without real-world training. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 10575-10585. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/kaiser22a.html.

OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, et al. GPT-4 technical report, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models, 2023.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Proceedings of the 34th Conference on Neural Information Processing Systems, 2020. URL https://commoncrawl.org/the-data/.

Nicolas Oulianov, Pierre-Louis Biojout, P. L. Venard, and Stan Girard. Evaluate LLMs in real time with Street Fighter III. https://github.com/OpenGenerativeAI/llm-colosseum. 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.

H. Shang and M. Borland. A Parallel Simplex Optimizer and its Application to High-Brightness Storage Ring Design. In Proceedings of the 2005 Particle Accelerator Conference, pages 4230-4232, Knoxville, TN, USA, 2005. IEEE. ISBN 978-0-7803-8859-8. doi 10.1109/PAC.2005.1591774.

Xiaobiao Huang. Robust simplex algorithm for online optimization. Physical Review Accelerators and Beams, 21(10): 104601, October 2018. doi:10.1103/PhysRevAccelBeams.21.104601.

Xiaobiao Huang, Jeff Corbett, James Safranek, and Juhao Wu. An algorithm for online optimization of accelerators. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 726:77-83, 2013. ISSN 0168-9002. doi https://doi.org/10.1016/j.nima.2013.05.046. URL https://www.sciencedirect.com/science/article/pii/S0168900213006347.

David K Olsson et al. Online optimisation of the MAX-IV $3 \mathrm{GeV}$ ring dynamic aperture. Proc. IPAC2018, 2281, 2018.

Zhe Zhang, Minghao Song, and Xiaobiao Huang. Optimization method to compensate accelerator performance drifts. Phys. Rev. Accel. Beams, 25:122801, Dec 2022a. doi 10.1103/PhysRevAccelBeams.25.122801. URL https://link.aps.org/doi/10.1103/PhysRevAccelBeams.25.122801.

Alexander Scheinker, En-Chuan Huang, and Charles Taylor. Extremum seeking-based control system for particle accelerator beam loss minimization. IEEE Transactions on Control Systems Technology, 30(5):2261-2268, 2022. doi 10.1109/TCST.2021.3136133.

W. F. Bergan, I. V. Bazarov, C. J. R. Duncan, D. B. Liarte, D. L. Rubin, and J. P. Sethna. Online storage ring optimization using dimension-reduction and genetic algorithms. Physical Review Accelerators and Beams, 22: 054601, May 2019. doi 10.1103/PhysRevAccelBeams.22.054601. URL https://link.aps.org/doi/10.1103/ PhysRevAccelBeams.22.054601.

S. Tomin, G. Geloni, I. Zagorodnov, A. Egger, W. Colocho, A. Valentinov, Y. Fomin, I. Agapov, T. Cope, D. Ratner, et al. Progress in automatic software-based optimization of accelerator performance. In Proceedings of the 7th International Particle Accelerator Conference, 2016.

Zhe Zhang. Badger: The Ocelot Optimizer rebirth. Technical report, SLAC National Accelerator Lab., Menlo Park, CA (United States), 2021.

Zhe Zhang, Auralee Edelen, C Mayes, J Garrahan, J Shtalenkova, R Roussel, S Miskovich, Daniel Ratner, Michael Boese, Sergey Tomin, et al. Badger: The missing optimizer in ACR. In Proceedings of the 13th International Particle Accelerator Conference (IPAC 2022), 2022b. ISBN 9783954502271. doi 10.18429/JACoW-IPAC2022-TUPOST058 URLhttps://slac-ml.github.io/Badger.

R. Roussel, A. Edelen, A. Bartnik, and C. Mayes. Xopt: A simplified framework for optimization of accelerator problems using advanced algorithms. In Proc. IPAC'23, number 14 in IPAC'23 - 14th International Particle Accelerator Conference, pages 4796-4799. JACoW Publishing, Geneva, Switzerland, 05 2023b. ISBN 978-3-95450-2318. doi doi:10.18429/jacow-ipac2023-thpl164. URL https://indico.jacow.org/event/41/contributions/ 2556 .

Auralee Linscott Edelen, Christopher Mayes, Daniel Bowring, Daniel Ratner, Andreas Adelmann, Rasmus Ischebeck, Jochem Snuverink, Ilya Agapov, Raimund Kammering, Jonathan Edelen, et al. Opportunities in machine learning for particle accelerators, 11 2018. URL http://arxiv.org/abs/1811.03172.

Tobias Boltz, Miriam Brosi, Erik Bründermann, Bastian Haerer, Peter Kaiser, Christoph Pohl, Patrick Schreiber, Minjie Yan, Tamim Asfour, and A-S Müller. Feedback design for control of the micro-bunching instability based on reinforcement learning. In CERN Yellow Reports: Conference Proceedings, volume 9, pages 227-227, 2020.

J. St. John, C. Herwig, D. Kafkes, J. Mitrevski, W. A. Pellico, G. N. Perdue, A. Quintero-Parra, B. A. Schupbach, K. Seiya, N. Tran, et al. Real-time artificial intelligence for accelerator control: A study at the Fermilab Booster. Physical Review Accelerators and Beams, 24:104601, 2021.

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In Proceedings of the 30th Conference on Neural Information Processing Systems (NIPS 2016), 62016.

Ke Li and Jitendra Malik. Learning to optimize. In International Conference on Learning Representations, 2017a. URL https://openreview.net/forum?id=ry4Vrt5gl.

Ke Li and Jitendra Malik. Learning to optimize neural nets, 2017b. Preprint available at http://arxiv.org/abs/ 1703.00441 .

Tianlong Chen, Xiaohan Chen, Wuyang Chen, Zhangyang Wang, Howard Heaton, Jialin Liu, and Wotao Yin. Learning to optimize: A primer and a benchmark. Journal of Machine Learning Research, 23:1-59, 2022. URL http: //jmlr.org/papers/v23/21-0308.html.

Verena Kain, Simon Hirlander, Brennan Goddard, Francesco Maria Velotti, Giovanni Zevi Della Porta, Niky Bruchon, and Gianluca Valentino. Sample-efficient reinforcement learning for CERN accelerator control. Physical Review Accelerators and Beams, 23:124801, Dec 2020. doi 10.1103/PhysRevAccelBeams.23.124801. URL https: //link.aps.org/doi/10.1103/PhysRevAccelBeams.23.124801.

Xiaoying Pang, Sunil Thulasidasan, and Larry Rybarcyk. Autonomous control of a particle accelerator using deep reinforcement learning. In Proceedings of the Machine Learning for Engineering Modeling, Simulation, and Design Workshop at Neural Information Processing Systems 2020, 10 2020. URL http://arxiv.org/abs/2010.08141.

Francesco Maria Velotti, Brennan Goddard, Verena Kain, Rebecca Ramjiawan, Giovanni Zevi Della Porta, and Simon Hirlaender. Towards automatic setup of $18 \mathrm{MeV}$ electron beamline using machine learning. Machine

Learning: Science and Technology, 4:025016, 6 2023. ISSN 2632-2153. doi 10.1088/2632-2153/acce21. URL https://iopscience.iop.org/article/10.1088/2632-2153/acce21.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017. URL http://arxiv.org/abs/1706.03762.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, et al. Gemini: A family of highly capable multimodal models, 2023.

Anthropic. Claude \Anthropic. Web Page, 2023. URL https://www.anthropic.com/claude Accessed: 2024-0414.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of GPT-4, 2023.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. Orca 2: Teaching small language models how to reason, 11 2023. URL http://arxiv.org/abs/2311.11045.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7B: Improving LLM helpfulness \& harmlessness with RLAIF, November 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of Experts, 2024.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers, 9 2023. URL http://arxiv.org/abs/2309.03409.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J.R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625:468-475, 1 2024. ISSN 14764687. doi 10.1038/s41586-023-06924-6.

Antonin Sulc, Raimund Kammering, Annika Eichler, and Tim Wilksen. PACuna: Automated fine-tuning of language models for particle accelerators. In Machine Learning and the Physical Sciences Workshop, NeurIPS 2023, 2023. URLhttps://github.com/sulcantonin/LLM_NeuralIPS23.git.

Frank Mayet. Building an intelligent accelerator operations assistant. https://indico.desy.de/event/38849/ contributions/162131/, 2024a.

Frank Mayet. GAIA: A general AI assistant for intelligent accelerator operations, 2024b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models, 2023.

Eva Panofski et al. Commissioning results and electron beam characterization with the S-band photoinjector at SINBAD-ARES. Instruments, 5, 2021.

F. Burkart, R.W. Aßmann, H. Dinter, S. Jaster-Merz, W. Kuropka, F. Mayet, and T. Vinatier. The ARES Linac at DESY. In Proceedings of the 31st International Linear Accelerator Conference (LINAC'22), number 31 in International Linear Accelerator Conference, pages 691-694. JACoW Publishing, Geneva, Switzerland, 09 2022. ISBN 978-395450-215-8. doi:10.18429/JACoW-LINAC2022-THPOJO01. URL https://jacow.org/linac2022/papers/ thpojo01.pdf

Jan Kaiser, Chenran Xu, Annika Eichler, Andrea Santamaria Garcia, Oliver Stein, Erik Bründermann, Willi Kuropka, Hannes Dinter, Frank Mayet, Thomas Vinatier, et al. Learning to do or learning while doing: Reinforcement learning and bayesian optimisation for online continuous tuning, 2023.

Jan Kaiser and Chenran Xu. Cheetah, 2023. URL/https://github.com/desy-ml/cheetah.

Chenran Xu, Jan Kaiser, Erik Bründermann, Annika Eichler, A.-S. Müller, and Andrea Santamaria Garcia. Beam trajectory control with lattice-agnostic reinforcement learning. In Proc. IPAC'23, 2023. ISBN 978-3-95450-231-8. doi 10.18429/JACoW-IPAC-2023-THPL029. URL https://doi.org/10.18429/JACoW-IPAC-23-THPL029.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology, 2024.

OpenAI. GPT-3.5 Turbo model documentation. https://platform.openai.com/docs/models/gpt-3-5-turbo, 2023. Accessed: 2024-04-15.

OpenAI. New models and developer products announced at DevDay, November 2023. URL https://openai.com/ blog/new-models-and-developer-products-announced-at-devday. Accessed: 2024-04-15.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, 2023.

Harrison Chase. LangChain, October 2022. URL https://github.com/langchain-ai/langchain

Ollama Team. Ollama, 2023. URL https://ollama.com. Accessed: 2024-04-15.

Farama Foundation. Gymnasium, 2022. URL https://gymnasium.farama.org.

Oliver Stein, Jan Kaiser, and Annika Eichler. Accelerating linear beam dynamics simulations for machine learning applications. In Proceedings of the 13th International Particle Accelerator Conference, 2022.

Jan Kaiser, Chenran Xu, Annika Eichler, and Andrea Santamaria Garcia. Cheetah: Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence?, 2019.

Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. Making AI less "thirsty": Uncovering and addressing the secret water footprint of AI models, 4 2023. URL http://arxiv.org/abs/2304.03271

NVIDIA. NVIDIA A100 data sheet, 2020. URL https://www.nvidia.com/content/dam/en-zz/Solutions/ Data-Center/a100/pdf/nvidia-a100-datasheet.pdf

Bosch. KIR31AD40 EU data sheet, 2021. URL https://media3.bosch-home.com/Documents/eudatasheet/ de-DE/KIR31AD40.pdf.

BMW. BMW 5 Series price list, 2024. URL https://www.bmw.de/content/dam/bmw/marketDE/bmw_de/ new-vehicles/pricelists/preisliste-bmw5er-new.pdf.coredownload.inline.pdf. Accessed: 202404-24

Umweltbundesamt. Entwicklung der spezifischen treibhausgas-emissionen des deutschen strommix in den jahren 1990 - 2022. Technical Report Climate Change 20/2023, Umweltbundesamt, 2023. URL https://www.umweltbundesamt.de/sites/default/files/medien/1410/publikationen/2023_05_ 23_climate_change_20-2023_strommix_bf.pdf

Maonan Wang, Aoyu Pang, Yuheng Kan, Man-On Pun, Chung Shue Chen, and Bo Huang. LLM-assisted light: Leveraging large language model capabilities for human-mimetic traffic signal control in complex urban environments. arXiv preprint arXiv:2403.08337, 2024.

Man-On Pun Aoyu Pang, Maonan Wan and Chung Shue Chen. Human-like assessment of RL actions: Leveraging large language models for RL actions in traffic signal control systems. https://github.com/Traffic-Alpha/ TSC-HARLA, 2023.

\title{
A System Prompts
}

Orca 2 7B and Orca 2 13B:

You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.

Vicuna 7B 16K:

A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

\section*{B Failed Responses}

Incoherent response generated by Gemma 2B with the Explained Prompt:

```

sigma_y": 320.27 } .. Magnet settings: ..json { "Q1": -14.33, "Q2": -1.63, "CV": 0.02,
"Q3": 15.98, "CH": 2.95 } `.' Beam parameters: ..json { "mu_x": 1232.76, "sigma_x": 138.02, "mu_y": 1013.97, "sigma_y": 1370.42 } `` Magnet settings: ```json { "Q1": 0.00, "Q2": 0.00, "CV": 1.26, "Q3": 0.00, "CH": 0.00 } .. Beam parameters: ..`json { "mu_x":
-183.38, "sigma_x": 46.87, "mu_y": 2400.06, "sigma_y": 648.22 } ``

```

Long chain-of-though response without properly formatted set of next magnet settings generated by Orca 2 13B with the Explained Prompt:

\begin{abstract}
To help you optimise the beam parameters, I will use a genetic algorithm that iteratively mutates and crosses the existing magnet settings to produce new ones. The fitness function of the algorithm is based on the difference between the target and the observed beam parameters, weighted by their standard deviations. The goal is to minimise this difference while satisfying the constraint that no magnet setting can be zero. Here are the steps I will follow:
\end{abstract}

1. Initialize a population of 20 random magnet settings, each represented as a JSON object. 2. Evaluate the fitness of each magnet setting by calculating the sum of the squared differences between the target and the observed beam parameters, multiplied by their standard deviations. Normalize this sum to have a mean of zero and a standard deviation of one. This is called the information gain. 3. Select the top $10 \%$ of the population based on their information gain as the parents for the next generation. Randomly select two of them as the donors for crossbreeding. 4. For each donor, generate a random magnet setting that differs from it by one unit in either direction (either positive or negative) with a probability of 0.5. This is called the child A. 5. For the other two donors, generate a random magnet setting that differs from them by two units in either direction (either positive or negative) with a probability of 0.25 . This is called the child B. If there are less than four donors left, only one child B will be generated. 6. Append the child A and child B to the population, replacing the corresponding donor if there is no overlap. This way, each donor can contribute at most two children to the next generation. 7. Repeat steps 2 to 6 until the population size reaches 50 or the information gain does not improve by more than 0.01 for three consecutive generations. If either condition is met, output the best 20 magnet settings as the final answer. Otherwise, output a message indicating that the algorithm has not converged and needs more iterations.

Here is an example of how the algorithm works on the first generation:

Population: ["Q1": -5.83, "Q2": 7.67, "CV": -4.09, "Q3": -7.53, "CH": $-4.17, \ldots]$

Fitness (information gain): $[-0.04,0.03,0.05,0.08,0.07]$

```

Select top 10%: ["Q1": -5.83, "Q2": 7.67, "CV": -4.09, "Q3": -7.53, "CH": -4.17, ...]
Donors for crossbreeding: ["Q1": -5.83, "Q2": 7.67, "CV": -4.09, "Q3": -7.53, "CH":
-4.17, ...]
Child A: "Q1": -6.83, "Q2": 8.67, "CV": -3.09, "Q3": -8.53, "CH": -3.17
Child B: "Q1": -4.83, "Q2": 9.67, "CV": -1.09, "Q3": -9.53, "CH": -1.17
Population: ["Q1": -5.83, "Q2": 7.67, "CV": -4.09, "Q3": -7.53, "CH": -4.17, "Q1":
-6.83, "Q2": 8.67, "CV": -3.09, "Q3": -8.53, "CH": -3.17, "Q1": -4.83, "Q2": 9.67,
"CV": -1.09, "Q3": -9.53, "CH": -1.17]
Fitness (information gain): [-0.04, 0.04, 0.04, 0.08, 0.06]

### Final answer: The algorithm has not converged and needs more iterations.

```

JSON for next magnet settings generated with trailing comma not permitted in JSON syntax by Llama 2 7B with the Explained Prompt:

```

```json { "Q1": -10.5, "Q2": -8.25, "CV": -3.26, "Q3": -8.75, "CH": -4.5, } ..
```


[^0]:    * jan.kaiser@desy.de

    †annika.eichler@desy.de

    ${ }^{\ddagger}$ anne.lauscher@uni-hamburg.de

[^1]:    Human: Now you will help me optimise the horizontal and vertical position and size of an electron beam on a diagnostic screen in a particle accelerator.

    You are able to control five magnets in the beam line. The magnets are called Q1, Q2, CV, Q3 and $\mathrm{CH}$.

    Q1, Q2 and Q3 are quadrupole magnets. When their k1 strength is increased, the beam becomes more focused in the horizontal plane and more defocused in the vertical plane. When their k1 strength is decreased, the beam becomes more focused in the vertical plane and more defocused in the horizontal plane. When their k1 strength is zero, the beam is not focused in either plane. Quadrupole magnets might also steer the beam in the horizontal or vertical plane depending on their k0 strength, when the beam does not travel through the centre of the magnet. The range of the $\mathrm{k} 1$ strength is -30.0 to $30.0 \mathrm{~m}-2$.

</end of paper 4>


