<paper 0>
# A Comprehensive Study of Knowledge Editing for Large Language Models 

Ningyu Zhang*, Yunzhi Yao*, Bozhong Tian*, Peng Wang*, Shumin Deng*, Mengru Wang,<br>Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu,<br>Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang,<br>Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen ${ }^{\dagger}$<br>Zhejiang University, National University of Singapore,<br>University of California, Los Angeles, Ant Group, Alibaba Group<br>\{zhangningyu, yyztodd\}@zju.edu.cn<br>Project: https://zjunlp.github.io/project/KnowEdit


#### Abstract

Large Language Models (LLMs) have shown extraordinary capabilities in understanding and generating text that closely mirrors human communication. However, a primary limitation lies in the significant computational demands during training, arising from their extensive parameterization. This challenge is further intensified by the dynamic nature of the world, necessitating frequent updates to LLMs to correct outdated information or integrate new knowledge, thereby ensuring their continued relevance. Note that many applications demand continual model adjustments post-training to address deficiencies or undesirable behaviors. There is an increasing interest in efficient, lightweight methods for onthe-fly model modifications. To this end, recent years have seen a burgeoning in the techniques of knowledge editing for LLMs, which aim to efficiently modify LLMs' behaviors within specific domains while preserving overall performance across various inputs. In this paper, we first define the knowledge editing problem and then provide a comprehensive review of cutting-edge approaches. Drawing inspiration from educational and cognitive research theories $[1-3$, we propose a unified categorization criterion that classifies knowledge editing methods into three groups: resorting to external knowledge, merging knowledge into the model, and editing intrinsic knowledge. Furthermore, we introduce a new benchmark, KnowEdit, for a comprehensive empirical evaluation of representative knowledge editing approaches. Additionally, we provide an in-depth analysis of knowledge location, which can give a deeper understanding of the knowledge structures inherent within LLMs. Initially conceived as a means to steer LLMs efficiently, we hope that insights gained from knowledge editing research could shed light on the underlying knowledge mechanisms of LLMs. To facilitate future research, we have released an open-source framework, EasyEdit ${ }^{1}$, which will enable practitioners to efficiently and flexibly implement knowledge editing for LLMs. Finally, we discuss several potential applications of knowledge editing, outlining its broad and impactful implications.


Keywords - natural language processing, large language models, knowledge editing[^0]

The contributions of the authors are detailed in $\S$ CONTRIBUTIONS.

## Contents

1 Introduction ..... 3
2 Background ..... 4
2.1 Large Language Models ..... 4
2.1.1 Transformers for LLM ..... 4
2.1.2 Mechanism of Knowledge Storage in LLMs ..... 5
2.2 Related Techniques ..... 6
3 Knowledge Editing for LLMs ..... 8
3.1 Preliminary ..... 8
3.2 Task Definition ..... 8
3.3 Methods ..... 9
3.3.1 Recognition Phase: Resorting to External Knowledge ..... 10
3.3.2 Association Phase: Merge the Knowledge into the Model ..... 11
3.3.3 Mastery Phase: Editing Intrinsic Knowledge ..... 11
3.4 New Benchmark: KnowEdit ..... 12
3.5 Evaluation for Knowledge Editing ..... 14
4 Experiments ..... 15
4.1 Experiment Settings ..... 15
4.2 Main Results ..... 16
4.3 Impact of Knowledge Editing on General Tasks ..... 17
4.4 Multi-Task Knowledge Editing ..... 18
4.5 Error and Case Analysis ..... 19
5 Analysis ..... 21
5.1 Comparison of Different Knowledge Editing Methods ..... 21
5.2 The Effectiveness of Knowledge Locating in LLMs ..... 22
5.3 The Implicit Knowledge Structure in LLMs ..... 23
6 Applications ..... 24
6.1 Efficient Machine Learning ..... 24
6.2 AI-Generated Content (AIGC) ..... 26
6.3 Trustworthy AI ..... 26
6.4 Human-Computer Interaction: Personalized Agents ..... 28
7 Discussion and Conclusion ..... 29
Broader Impacts ..... 29

## 1 Introduction

Knowledge is a fundamental component of human intelligence and civilization [4]. Its systematic structure empowers us to represent tangible entities or delineate principles through symbolic means, offering the capability to facilitate the articulation of intricate behaviors or tasks [5-7]. Throughout our lives, we humans continuously gather an extensive wealth of knowledge and learn to adaptively apply it in various contexts. The enduring exploration of the nature of knowledge and the processes by which we acquire, retain, and interpret it, continues to captivate scientists, which is not just a technical pursuit but a journey towards mirroring the nuanced complexities of human cognition, communication and intelligence [8-12].

Recently, Large Language Models (LLMs) like GPT-4 [13] have showcased a remarkable ability in Natural Language Processing (NLP) to retain a vast amount of knowledge, arguably surpassing human capacity [14-31]. This achievement can be attributed to the way LLMs process and compress huge amounts of data [32-35], potentially forming more concise, coherent, and interpretable models of the underlying generative processes, essentially creating a kind of "world model" [36-38]. For example, Dai et al. [39] have introduced the Knowledge Neuron (KN) thesis, which proposes that language models function similarly to key-value memories. Here, the multi-layer perceptron (MLP) weights in the core region [40] may play a crucial role in recalling facts from the training corpus, suggesting a more structured and retrievable form of knowledge storage within LLMs [41, 42]. Further insights come from the ability of LLMs to understand and manipulate complex strategic environments, whereas $\mathrm{Li}$ et al. [43] has demonstrated that transformers trained for next-token prediction in board games such as Othello develop explicit representations of the game's state. Patel and Pavlick [44] have revealed that LLMs can track boolean states of subjects within given contexts and learn representations that reflect perceptual, symbolic concepts [36, 45-47]. This dual capability indicates that LLMs can serve as extensive knowledge bases [48-59], not only storing vast amounts of information but also structuring it in ways that may mirror human cognitive processes.

However, LLMs have limitations like factual fallacy, potential generation of harmful content, and outdated knowledge due to their training cut-off [60-63]. Retraining to correct these issues is both costly and time-consuming [64-68]. To address this, recent years have seen a surge in the development of knowledge editing techniques specifically tailored for LLMs, which allows for cost-effective post-hoc modifications to models [69-71]. This technique focuses on specific areas for adjustment without compromising overall performance and can help understand how LLMs represent and process information, which is crucial for ensuring the fairness, and safety in Artificial Intelligence (AI) applications $[72-76]$.

This paper first attempts to provide a comprehensive study of the development and recent advances in knowledge editing for LLMs. We first introduce the architecture of Transformers, mechanism of knowledge storage in LLMs ( $\$ 2.1$ ), and related techniques including parameter-efficient fine-tuning, knowledge augmentation, continue learning and machine unlearning (\$2.2). Then we introduce preliminary (\$3.1), formally describe the knowledge editing problem (\$3.2), and propose a new taxonomy (§3.3) to provide a unified view on knowledge editing methods based on the educational and cognitive research theories [1-3]. Specifically, we categorize knowledge editing for LLMs into: resorting to external knowledge (\$3.3.1), merging knowledge into the model (§3.3.2), and editing intrinsic knowledge ( $\$ 3.3 .3$ ) approaches. Our categorization criterion is summarized as follows:

- Resorting to External Knowledge. This kind of approach is similar to the recognition phase in human cognitive processes, which needs to be exposed to new knowledge within a relevant context, just as people first encounter new information. For example, providing sentences that illustrate a factual update as a demonstration of the model allows initial recognition of the knowledge to be edited.
- Merging Knowledge into the Model. This kind of approach closely resembles the association phrase in human cognitive processes, in which connections are formed between the new knowledge and existing knowledge in the model. Methods would combine or substitute the output or intermediate output with a learned knowledge representation.
- Editing Intrinsic Knowledge. This approach to knowledge editing is akin to the mastery phase in human cognitive processes. It involves the model fully integrating knowledge into its parameters by modifying the weights and utilizing them reliably.

This paper then involves extensive and comprehensive experiments conducted on 12 NLP datasets. These are meticulously designed to evaluate the performance ( $\$ 4$ ), usability, and underlying mechanisms, complete with in-depth analyses (\$5), among other aspects. The key insights from our research are summarized as follows:

- Performance. We construct a new benchmark, named KnowEdit, and report the empirical results of cutting-edge knowledge editing approaches for LLMs, providing a fair comparison and illustrating their overall performance in the settings of knowledge insertion, modification, and erasure.
- Usability. We illustrate the impact of knowledge editing on general tasks and multi-task knowledge editing, which implies that contemporary knowledge editing methods are effective in executing factual updates with minimal disruptions to the model's cognitive capabilities and adaptability across diverse knowledge domains.
- Mechanism. We observe a pronounced focus on one or several columns within the value layer in edited LLMs. Furthermore, we find that the process of knowledge locating (e.g., causal analysis) tends to pinpoint only the areas related to the entity in question, rather than the entire factual context, suggesting that LLMs might be deriving answers either by recalling information memorized from their pretraining corpus or through a multi-step reasoning process. Additionally, we delve into the possibility that knowledge editing for LLMs could lead to unintended consequences, an aspect warranting careful consideration.

Finally, we delve into the multifaceted applications of knowledge editing, examining its potential from a variety of perspectives (§6), including efficient machine learning, AI-Generated Content (AIGC), trustworthy AI, and human-computer interaction (personalized agents). Additionally, our discussion extends to the broader impacts of knowledge editing techniques, specifically focusing on aspects such as energy consumption and interpretability ( $\$ 7$ ). This paper aims to serve as a catalyst for further research in the realm of LLMs, emphasizing efficiency and innovation. To support and encourage future research, we will make our tools, codes, data splits, and trained model checkpoints publicly accessible.

## 2 Background

### 2.1 Large Language Models

### 2.1.1 Transformers for $L L M$

The Transformer [77] model, a cornerstone in the design of modern state-of-the-art LLMs, represents a significant shift from previous sequence learning methods. The original Transformer model is introduced as an encoder-decoder framework, wherein both the encoder and decoder consist of a series of identical layers stacked upon each other. Each block within this architecture is equipped with a self-attention module and a fully connected feed-forward neural network. Uniquely, the blocks in the decoder also incorporate an additional cross-attention layer, positioned above the self-attention layer, which is designed to effectively capture and integrate information from the encoder.

Self-Attention Module (SelfAttn) The self-attention mechanism is a pivotal feature of the Transformer, allowing it to process sequences of data effectively. This module empowers each position within the encoder to attend to all positions in the preceding layer, thereby efficiently capturing contextual information embedded in the sequence. The mathematical representation of the self-attention mechanism is as follows:

$$
\begin{equation*}
H=\operatorname{ATT}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V \tag{1}
\end{equation*}
$$

Feed-Forward Module (FFN) Following each attention layer in the Transformer is a fully connected Feed-Forward Neural network (FFN). This specific component of the architecture comprises two linear transformations, with a ReLU activation function intervening between them. The structure of the FFN can be succinctly described as follows:

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-05.jpg?height=566&width=1315&top_left_y=243&top_left_x=405)

Figure 1: The mechanism of knowledge storage in LLMs. Here, we summarize the findings of current works, including: Jawahar et al. [78], Geva et al. [41], Dai et al. [39], Meng et al. [79], and Hernandez et al. [80].

$$
\begin{equation*}
\operatorname{FFN}(\mathbf{x})=\operatorname{ReLU}\left(\mathbf{x} \cdot W_{1}+b_{1}\right) \cdot W_{2}+b_{2} \tag{2}
\end{equation*}
$$

Since its inception, the Transformer model has revolutionized the field of NLP. Its adaptable and efficient architecture has facilitated advancements in various NLP tasks, such as question-answering, text summarization, and machine translation systems. The model's influence extends beyond NLP, impacting other areas of machine learning and setting a new standard for building complex and effective neural network architectures.

### 2.1.2 Mechanism of Knowledge Storage in LLMs

The Transformer's remarkable performance is partly attributed to its ability to store a wealth of information within its parameters, encompassing linguistic [81], commonsense [82-84], arithmetic, and world knowledge [48, 85-87]. However, the exact manner in which this knowledge is organized within LLMs is still largely enigmatic. Current research efforts are dedicated to unraveling the mechanistic explanations of LLMs' behaviours [88-92], especially the complexities of knowledge storage in LLMs, with Figure 1 illustrating some of these research findings.

A key area of inquiry is pinpointing the specific location of knowledge within the model. Jawahar et al. [78] dissects the intricacies of the English language structure as comprehended by BERT [93]. Their findings reveal that BERT's phrasal representations capture phrase-level information predominantly in the lower layers, and encode an intricate hierarchy of linguistic elements in the intermediate layers. This hierarchy is characterized by surface features at the foundational level and syntactic features in the central layers, and culminates with semantic features at the uppermost level. Geva et al. [41] proposes that the FFN layers in a Transformer model function akin to key-value memories. They suggest that the FFN input operates as a query, with the first layer representing keys and the second layer corresponding to values. They find that human-interpretable shallow input patterns trigger each key neuron, and the corresponding value neurons store the next-token output probability. As a result, the final output of the FFN can be understood as the weighted sum of activated values. Furthermore, they demonstrate that value vectors often embody interpretable concepts and knowledge, which can be intensified or attenuated through specific manipulations [42]. Building on this, Dai et al. [39] introduces the concept of "Knowledge Neurons", suggesting that knowledge is localized within a small subset of FFN neurons in the uppermost layers of the language model. These neurons are identified through the analysis of integrated gradients across various prompts [94-96]. Similarly, Meng et al. [79] employs a method known as "causal tracing" to assess the indirect influences of hidden states or activations, revealing that factual knowledge predominantly resides in the early-layer FFNs of such models. Additionaly, Chen et al. [97] makes an intriguing finding that the language model contains language-independent neurons that express multilingual knowledge and degenerate neurons that convey redundant information by applying the integrated gradients method [94]. Concurrently, Zhao et al. [98] observes that LLMs appear to possess a specialized linguistic region
responsible for processing multiple languages. Gueta et al. [99] suggests that knowledge is a region in weight space for fine-tuned language models. They find that after finetuning a pretrained model on similar datasets, the resulting models are close to each other in weight space. Recent interests also revolve around dissecting the distinct functionalities of individual neurons within LLMs [100]. Yet, it is crucial to note that some researchers caution against overinterpreting these findings, emphasizing that models illustrate correlations rather than explicit mechanisms. For instance, Anonymous [101] argues that while MLP neurons may exhibit patterns interpretable through a linguistic lens, they do not necessarily "store" knowledge in a conventional sense, whether linguistic or factual.

Thus, the question of how Transformer LLMs retrieve and utilize this stored knowledge remains open, and some work has begun to unveil this mystery. Geva et al. [102] analyzes the information flow in the model and finds the self-attention model conducts attribute extraction during computing inspired by the circuit theory [103, 104]. Foote et al. [105] proposes Neuron to Graph (N2G), an innovative tool that automatically extracts a neuron's behavior from the dataset it was trained on and translates it into an interpretable graph. Further, Hernandez et al. [80] conceptualizes relational knowledge within Transformers as a linear affine function, mapping subjects to objects. As to other knowledge, Gurnee and Tegmark [36] discovers that LLMs learn linear representations of space and time across multiple scales and identify individual "space neurons" and "time neurons" that reliably encode spatial and temporal coordinates. However, it is imperative to acknowledge that these studies predominantly concentrate on the representation of individual knowledge facts. The broader challenge lies in comprehensively understanding how various strands of knowledge are intricately organized and interconnected within these complex models [106, 107].

### 2.2 Related Techniques

Parameter-efficient Fine-tuning Fine-tuning all parameters of LLMs can be computationally expensive. To enable efficient adaptation, parameter-efficient tuning (PET) [108, 109] techniques have been proposed to match full fine-tuning performance while only updating a minimal parameters. PET consists of three distinct paradigms: addition-based, specification-based, and reparameterization-based methods. In addition-based methods, extra trainable neural modules or parameters, which are not present in the original model or process, are introduced. A prime example of this is Adapter, as discussed in Houlsby et al. [110]. On the other hand, specification-based methods involve fine-tuning a select number of parameters, while keeping the majority of the model's parameters unchanged. A notable method in this category is LoRA, as detailed in Hu et al. [111].

By fine-tuning a small number of parameters, PET methods aim to maximize model performance while reducing required resources and tuning time. PET techniques hold promise since knowledge editing seeks to efficiently modify model behavior. However, PET is typically applied to enhance task performance rather than edit knowledge specifically. The efficacy of existing PET methods for knowledge editing remains largely unexplored. Investigating how to leverage PET for efficient and precise knowledge updates presents an interesting direction for future work.

Knowledge Augmentation for LLMs LLMs still face unknown questions, and many knowledgeaugmented methods are proposed to help the model deal with this task [112-114]. The most popular way is the retrieval-augmented methods [115-117]. With the help of the retrieved knowledge or context that is related to the input, the model can give the desired output. The integration of the retrieved information includes both the input, intermediate, and output layers [118]. During the input phase, retrieved texts are concatenated with the original input text [119-121]. In some works, the retrieved components are latent and integrated into the intermediate layers of Transformers [122124]. In the output phase, the distribution of tokens from the retrieved components and the LLMs are interpolated [125-128].

The knowledge-augmented method is a great solution for the missing or misinformation in LLMs but it still has some disadvantages. As a temporary solution, retrieval methods suffer from poor retrieval results and relatedness $[129,130]$. The data retrieved often contains some noise, such as additional content that is irrelevant to a question but that may be relevant to a different question (i.e., not necessarily random noise) [131]. In these situations, the model fails to distinguish the knowledge that is necessary to answer the question, leading to spurious reasoning and degraded performance. Meanwhile, retrieval typically operates at a broader level of relevant passages without fine-grained control over precisely which information is modified within the model.

|  | Fewer Params | Precise Control | Support Phenomena |
| :--- | :---: | :---: | :---: |
| Finetune | $x$ | $x$ | + |
| Parameter-efficient Fine-Tuning | $\checkmark$ | $x$ | + |
| Knowledge Augmentation |  | $x$ | + |
| Continual Learning | $x$ | $x$ | - |
| Model Unlearning |  | $x$ | +- |

Table 1: Integrated comparison between knowledge editing and related techniques. The symbol $\checkmark$ denotes the presence of a particular feature in the technique, while $X$ signifies its absence. + indicates an enhancement of the LLMs' capabilities, whereas - signifies a reduction or removal of certain abilities within the model.

Continual Learning Continual learning (CL), also known as lifelong machine learning or incremental learning, refers to the ability of machine learning models to continuously acquire new skills and learn new tasks while retaining previously learned knowledge [132-135]. This is akin to how humans learn throughout their lifetimes by continually accumulating new information and skills without forgetting the old ones. Conventional machine learning models struggle with this as they are trained on independent and identically distributed data. When the distribution shifts or new tasks are encountered, their performance significantly degrades on older tasks due to catastrophic forgetting. Some key techniques being explored include replay-based methods [136, 137], regularization-based approaches [138, 139], and dynamic architecture methods [140, 141]. Continual learning focuses on allowing machine learning models to learn new tasks and adapt to new domains over time without forgetting earlier ones, which resembles the goal of knowledge editing. In contrast, knowledge editing focuses specifically on manipulating and updating the internal knowledge representations learned by pre-trained language models without regard to the underlying tasks or domains. The goal of knowledge editing is to dynamically refine language understanding independent of eventual applications, addressing the "fixedness" issue of pre-trained language models once deployed. Both areas are important for developing AI systems that can progressively acquire and flexibly apply knowledge throughout their lifetime.

Machine Unlearning In addition, it is crucial for models to be capable of discarding undesirable (mis)behaviors, which aligns with the concept of machine unlearning [142-146]. Chen and Yang [147] proposes an efficient unlearning framework EUL that can efficiently update LLMs without having to retrain the whole model after data removals, by introducing lightweight unlearning layers learned with a selective teacher-student objective into the Transformers. However, knowledge editing goes beyond unlearning by actively refining or erasing a model's learned knowledge base. Both machine unlearning and knowledge editing play important roles in enhancing reliability, fairness and effectiveness for LLMs across different domains and applications.

To conclude, the traditional approach to leveraging pre-trained language models involves fine-tuning them with target-specific data. However, in the realm of LLMs, this fine-tuning process encounters significant challenges. These include the vast number of parameters, substantial time and memory requirements, risks of overfitting, and issues like catastrophic forgetting. To address these challenges, several techniques have been developed, as we discussed above. Among these, knowledge editing emerges as a notable strategy. As we discussed in Table 1, knowledge editing, intersecting with these techniques, draws inspiration from a range of methodologies, showing promising results. This approach distinctively targets the knowledge embedded within LLMs, leveraging the inherent knowledge mechanisms of these models. Unlike simple adaptations of existing methods, knowledge editing necessitates a deeper comprehension of how LLMs function. It is not just about applying known techniques to new models; it is about understanding and manipulating the nuanced knowledge storage and processing capabilities of LLMs. Furthermore, knowledge editing represents a more precise and granular form of model manipulation as it involves selectively altering or enhancing specific aspects of a model's knowledge base, rather than broadly retraining or fine-tuning the entire model. These characteristics make knowledge editing a potentially more efficient and effective way to update and optimize LLMs for specific tasks or applications.

## 3 Knowledge Editing for LLMs

### 3.1 Preliminary

The substantial training on diverse datasets has equipped LLMs with a wealth of factual and commonsense information, positioning these models as virtual knowledge stores [48, 148, 149]. This rich knowledge base has been effectively utilized in various downstream tasks, as evidenced by numerous studies [150]. Additionally, Wang et al. [151] have demonstrated the potential of LLMs in autonomously constructing high-quality knowledge graphs, bypassing the need for human supervision. Despite their promise, LLMs, in their current state as emerging knowledge bases, exhibit certain limitations. These deficiencies often manifest as inaccuracies or errors in their outputs during practical applications. An ideal knowledge base would not only store extensive information but also allow for efficient and targeted updates to rectify these errors and improve their accuracy. Recognizing this gap, our paper introduces the concept of knowledge editing for LLMs. This approach is designed to enable quick and precise modifications to the LLMs, allowing them to generate more accurate and relevant outputs. By implementing knowledge editing for LLMs, we aim to enhance the utility of LLMs, moving them closer to the ideal of becoming universally reliable and adaptable repositories of knowledge. This advancement promises to address the current shortcomings of LLMs and unlock their full potential as dynamic and accurate knowledge bases for applications.

### 3.2 Task Definition

The initial goal of knowledge editing is to modify the specific knowledge $k$ in the LLM and improve the consistency and performance of the LLM without fine-tuning the whole model. This knowledge can be associated with many areas and types, such as facts [79], commonsense [152], sentiment [153] and so on. Knowledge editing is challenging due to the distributed and entangled nature of knowledge in LLMs.

Suppose the original model is $\theta$ and given the knowledge $k$ to be changed, by knowledge editing process $F$, we would get the post-edited model $\theta^{\prime}$ :

$$
\begin{equation*}
\theta^{\prime}=F(\theta, k) \tag{3}
\end{equation*}
$$

The post-edited model $\theta^{\prime}$ is supposed to override undesired model beliefs on the knowledge $k$ and keep other knowledge intact:

$$
\left\{\begin{array}{l}
\theta^{\prime}(k) \neq \theta(k)  \tag{4}\\
\forall k^{\prime} \neq k, \theta^{\prime}\left(k^{\prime}\right)=\theta\left(k^{\prime}\right)
\end{array}\right.
$$

As a knowledge base, it's paramount that knowledge editing cater to three fundamental settings: knowledge insertion, knowledge modification, and knowledge erasure.

Knowledge Insertion. As fields and entities progress, it becomes imperative for LLMs to assimilate emergent information. Knowledge insertion fulfills this by bestowing upon LLMs new knowledge previously outside their purview:

$$
\begin{equation*}
\theta^{\prime}=F(\theta,\{\emptyset\} \rightarrow\{k\}) \tag{5}
\end{equation*}
$$

Knowledge Modification. Knowledge modification refers to altering knowledge already stored in LLMs:

$$
\begin{equation*}
\theta^{\prime}=F\left(\theta,\{k\} \rightarrow\left\{k^{\prime}\right\}\right) \tag{6}
\end{equation*}
$$

This can be classified into two categories:

- Knowledge amendment - This aims at rectifying the inaccuracies embedded in LLMs to ensure the delivery of accurate information. As vast repositories of knowledge, LLMs are prone to housing outdated or erroneous information. Knowledge amendment serves to correct these fallacies, ensuring that models always generate accurate, up-to-date information.
- Knowledge disruption - Modifying LLMs to answer counterfactual or error prompts. This is more challenging as counterfactual notions initially receive lower scores compared to factual knowledge, as shown by Meng et al. [79]. This necessitates more targeted modification efforts.

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-09.jpg?height=775&width=1260&top_left_y=236&top_left_x=430)

Figure 2: Applying Human Learning Phases [1-3] to Knowledge Editing in LLMs: We see an analogy of Human Learning Phases and Knowledge Editing in LLMs and categorize current knowledge editing methods based on the learning phases of humans: recognition, association, and mastery.

Knowledge Erasure. Knowledge erasure targets the excision or obliteration of pre-existing knowledge in a model, primarily to reset distinct facts, relationships, or attributes. Formally, we have:

$$
\begin{equation*}
\theta^{\prime}=F(\theta,\{k\} \rightarrow\{\emptyset\}) \tag{7}
\end{equation*}
$$

Implementing knowledge erasure is pivotal to expunge biases and noxious knowledge and to curtail the recollection of confidential or private data, thereby fostering responsible and trustworthy AI.

In conclusion, the interplay between knowledge insertion, modification, and erasure forms essential aspects of model editing techniques. When combined, these techniques empower LLMs to transform, self-correct, and ethically adapt as needed.

### 3.3 Methods

The development of LLMs has reached a point where their capabilities closely resemble human cognitive processes, especially in learning and acquiring knowledge. Drawing inspiration from how humans learn, we can analogously apply these concepts to the process of editing LLMs as Figure 2 shows. Educational and cognitive research [1-3] delineates human knowledge acquisition into three distinct phases: recognition, association, and mastery. These phases offer a framework for conceptualizing the methods of knowledge editing in $\mathrm{LLMs}^{2}$ and we list them in Table 2.
- Recognition Phase: In the recognition phase, the model needs to be exposed to the new knowledge within a relevant context, just as people first encounter new information (§3.3.1). For example, providing sentences that illustrate a factual update as a demonstration of the model allows initial recognition of the knowledge to be edited.
- Association Phase: In the association stage, connections are formed between the new knowledge and existing knowledge in the model (§3.3.2), much like humans relate new ideas to prior concepts. Methods would combine or substitute the output or intermediate output $\boldsymbol{h}$ with a learned knowledge representation $\boldsymbol{h}_{\text {know }}$.
- Mastery Phase: The mastery phase involves the model fully acquiring the knowledge in their parameters and utilizing it reliably (\$3.3.3), akin to deep human mastery. This method directly changed the model's weight, $\Delta \boldsymbol{W}$, and the model can deal with the problem without any external help or merge.[^1]

| Category | Method | Edit Area | Edit Function | No <br> Training | Batch <br> Edit | Edited <br> \#Params |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Association | MemPrompt [154] | memory+retriever | Input $\rightarrow$ [Mem : Input $]$ | $\nu$ | $\checkmark$ | - |
| Phase | SERAC [153] | memory+classifier <br> +auxiliary model | Output $\rightarrow \operatorname{Model}_{c f}(\boldsymbol{x})$ | $x$ | $\checkmark$ | - |
|  | MeLLo [155] | memory+retriever | Input $\rightarrow$ [Mem : Input] | $\boldsymbol{v}$ | $x$ | - |
|  | IKE [156] | memory+retriever | Input $\rightarrow$ [Mem : Input $]$ | $\checkmark$ | $x$ | - |
|  | ICE [157] | prompt | Input $\rightarrow$ [Mem : Input] | $\checkmark$ | $x$ | - |
|  | PokeMQA [158] | memory+retriever | Input $\rightarrow$ [Mem : Input $]$ | $x$ | $x$ | - |
| Recogintion | Language Patches[159] | Output head <br> + params | $\boldsymbol{h} \rightarrow \lambda \boldsymbol{h}+$ <br> $(1-\lambda) \operatorname{Patch}(\boldsymbol{x})$ | $v$ | $\checkmark$ | $d_{h} \times$ \#Output |
| Phase | CaliNET [160] | FFN+params | $\boldsymbol{h} \rightarrow \boldsymbol{h}+\mathrm{FFN}_{\mathrm{add}}(\boldsymbol{x})$ | $x$ | $\checkmark$ | $N \times d_{h}$ |
|  | T-Patcher[161] | FFN+params | $\boldsymbol{h} \rightarrow \boldsymbol{h}+\mathrm{FFN}_{\mathrm{add}}(\boldsymbol{x})$ | $x$ | $x$ | $N \times d_{h}$ |
|  | REMEDI [162] | auxiliary model | $\boldsymbol{h} \rightarrow \operatorname{REMEDI}(\boldsymbol{x})$ | $x$ | $x$ | $d_{h} \times d_{h}$ |
|  | GRACE [163] | FFN+codebook | $\boldsymbol{h} \rightarrow \operatorname{GRACE}(\boldsymbol{x})$ | $x$ | $x$ | $N \times 2 d_{h}$ |
|  | LoRA [164] | Attn or FFN | $\boldsymbol{h} \rightarrow \boldsymbol{h}+s \cdot \operatorname{LoRA}(\boldsymbol{x})$ | $x$ | $\checkmark$ | $2 L \times 2 d_{a m} d_{h}$ |
|  | MELO [165] | Attn or FFN | $\boldsymbol{h} \rightarrow \boldsymbol{h}+s \cdot \operatorname{LoRA}(\boldsymbol{x})$ | $x$ | $x$ | $2 L \times 2 d_{a m} d_{h}$ |
| Mastery | FT-Constrained [166] | Any | $\boldsymbol{W} \rightarrow \boldsymbol{W}^{\prime}$ | $x$ | $\checkmark$ | $2 \times L \times d_{m} d_{h}$ |
| Phase | ENN [167] | Any | $\boldsymbol{W} \rightarrow \boldsymbol{W}^{\prime}$ | $x$ | $\checkmark$ | $2 \times L \times d_{m} d_{h}$ |
|  | $\mathrm{KE}[168]$ | Attn or FFN <br> +auxiliary model | $\boldsymbol{W} \rightarrow \boldsymbol{W}^{\prime}$ | $x$ | $v$ | $2 \times L \times d_{m} d_{h}$ |
|  | SLAG [169] | Attn or FFN <br> +auxiliary model | $\boldsymbol{W} \rightarrow \boldsymbol{W}^{\prime}$ | $x$ | $\checkmark$ | $2 \times L \times d_{m} d_{h}$ |
|  | MEND [170] | FFN+ <br> auxiliary model | $\boldsymbol{W} \rightarrow \boldsymbol{W}^{\prime}$ | $x$ | $\checkmark$ | $2 \times L \times d_{m} d_{h}$ |
|  | KN [39] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\prime}$ | $\nu$ | $x$ | $L \times N \times d_{h}$ |
|  | ROME [79] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\prime}$ | $\checkmark$ | $x$ | $d_{m} d_{h}$ |
|  | MEMIT [171] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\prime}$ | $v$ | $\checkmark$ | $L \times d_{m} d_{h}$ |
|  | PMET [172] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\text {d }}$ | $\checkmark$ | $\checkmark$ | $L \times d_{m} d_{h}$ |
|  | MALMEN [173] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\prime}$ | $x$ | $\checkmark$ | $L \times d_{m} d_{h}$ |
|  | BIRD [174] | FFN | $\boldsymbol{W}_{\text {down }} \rightarrow \boldsymbol{W}_{\text {down }}^{\prime}$ | $\checkmark$ | $x$ | $d_{m} d_{h}$ |

Table 2: Comparison between representative approaches of knowledge editing for LLMs. No Training refers to the methods that do not require additional training; Batch Edit means whether the methods can support editing multiple cases simultaneously in just one process. Edit Area refers to where the model's components are used; Editor \#Params indicates the parameters that need to be updated for editing. $L$ refers to the number of layers to update. $d_{h}$ denotes the dimensionality of the hidden layers in the Transformers. $d_{m}$ refers to the intermediate dimension that exists between the up projection and the down projection. $N$ symbolizes the total number of neurons that undergo updates within each individual layer.

### 3.3.1 Recognition Phase: Resorting to External Knowledge

When humans encounter new information, we do not always master it immediately. Instead, with the right context and examples, we can process and reason through this new knowledge. LLMs exhibit a similar capacity for in-context learning. This kind of method usually maintains a memory $\mathrm{M}$ and retrieves the most relevant cases for each input. IKE [156] exemplifies this approach by constructing three types of demonstrations - copy, update, and retain - to aid the model in producing reliable fact editing. It utilizes a demonstration store, formed from training sets, to guide the model towards generating the appropriate answer by retrieving the most pertinent demonstrations. Meanwhile, as a simple change in knowledge would lead to ripple effects [157], MeLLo [155] decomposes the question into different sub-questions for tackling multi-hop questions and retrieves the updated fact from the memory for each sub-question. Building on this, PokeMQA [158] offers a more robust method for question decomposition, introducing a programmable scope detector and knowledge prompts for enhanced reliability.

Humans also often utilize tools to augment their learning and problem-solving abilities. Likely, SERAC [153] builds a new counterfact model by retaining the new model and adopting a classifier to determine whether to use the counterfact model to answer the question. This method is straightforward and practically applicable, requiring no alterations to the original model. It's particularly advantageous for real-world use, given its ease of implementation. However, it's important to note that this approach can be vulnerable to issues such as retrieval errors (e.g.noise [175], harmful content [176]) and knowledge conflict problems [177, 178]. Recently, Yu et al. [179] investigats various scenarios in which language models opt for either the in-context answer or the memorized answer.

This research sheds light on the potential application of the method mentioned earlier, as it may offer insights into when and how to utilize it.

### 3.3.2 Association Phase: Merge the Knowledge into the Model

Unlike the recognition phase, this kind of method learns a representation for the new knowledge $\boldsymbol{h}_{\text {Know }}$ and merges this information with the original model's representation $\boldsymbol{h}$.

Murty et al. [159] proposes a knowledge patch as a new output head and interpolates the new head with the original head. Specially, inspired by previous findings that FFN may store knowledge, several methods integrate the knowledge into the FFN part. These methods add the neuron to the FFN and after the edit, the output is a combination of the previous FFN's output and the newly added knowledge:

$$
\begin{equation*}
\mathrm{FFN}^{\prime}(\mathbf{x})=\mathrm{FFN}(\mathbf{x})+\triangle \mathrm{FFN}(\mathbf{x}) \tag{8}
\end{equation*}
$$

In particular, T-Patcher [161] adds one neuron for each output error, while CaliNet [160] adds the knowledge via a fixed number of neurons. Meanwhile, Wu et al. [164] adopts LoRA to conduct knowledge edits. LoRA is a parameter-efficient fine-tuning method that freezes the weights of the LLM and introduces trainable rank decomposition matrices into the Transformer layers during the fine-tuning process. Hence, the $\boldsymbol{h}_{\text {Know }}$ is $\boldsymbol{x} \boldsymbol{W}_{\text {down }} \boldsymbol{W}_{\text {up }}$. Based on this, MELO [165] suggests a plug-in model editing method that uses dynamic LoRA to change the way language models work by indexing LoRA blocks dynamically based on an internal vector database. Instead of adding parameters to the model, REMEDI [162] directly substitutes the representation of the entity $h_{\text {entity }}$ by incorporating an attribute vector $h_{\text {attr }}$ into its original model's representation. Specifically, it learns the updated hidden states using an affine transformation $h_{\text {entity }}+W h_{\text {attr }}+b$ and replaces the LM's entity representation with it. In contrast, GRACE [163] adopts a unique approach by maintaining a discrete codebook that functions as an Adapter. This codebook is dynamically updated over time, allowing for the modification and refinement of a model's predictions. When the model encounters the knowledge for editing, it searches the codebook and replaces the hidden states as the value in the codebook. Overall, we can use a mathematical formula to represent these methods uniformly:

$$
\begin{equation*}
\boldsymbol{h}_{\text {final }}=\boldsymbol{h}+\boldsymbol{h}_{\mathrm{know}} \tag{9}
\end{equation*}
$$

This kind of method merged the information with the original model, making the weighting of knowledge from different sources a crucial parameter to consider. Given that these information sources often differ and may even conflict, the issue of knowledge conflict, as highlighted in Wang et al. [177], remains a significant challenge. To address this issue, F-Learning [180] introduces a "forgetting before learning" paradigm to achieve forgetting of old knowledge and learning of new knowledge based on parametric arithmetic. Additionally, determining the optimal point of integration for this information within the model is a critical aspect of this method. It is not just about merging the information, but also about where in the model's structure this integration occurs for maximum effectiveness and minimal disruption. Furthermore, the capacity of the model's parameters to store this integrated information is an area that still requires exploration. If every piece of edited knowledge necessitates additional parameters, the model's parameter could increase significantly with each edit. This raises concerns about scalability and efficiency, as continuously expanding the number of parameters might lead to issues like increased computational requirements.

### 3.3.3 Mastery Phase: Editing Intrinsic Knowledge

Despite the success of the previous two kinds of methods, we still confront how the model stores the knowledge and how they utilize and express the knowledge. Here, we come to the most important part of knowledge editing: the mastery stage. In this part, the model is required to learn the knowledge of its own parameters and master the knowledge by itself. Fine-tuning the model is the direct way to update the knowledge; however, training the whole model requires enormous computational resources and is time-consuming. Meanwhile, the finetuning technique usually suffers from catastrophic forgetting and overfitting. Constrained Fintune [166] utilizes a regularization to help the model keep the unrelated knowledge. Currently, many researchers endeavor to use knowledgespecific methods to modify the $\Delta \boldsymbol{W}$. These methods can be classified into two categories: metalearning and locate-and-edit.

Meta Learning To overcome these drawbacks, some meta-learning methods are proposed to edit the model. Instead of updating the weights directly, this kind of method teaches a hypernetwork to learn the change $\Delta \boldsymbol{W}$ of the model. KE [168] directly uses the representation of the new knowledge to train the model to update the matrix. SLAG [169] introduces a new training objective considering sequential, local, and generalizing model updates. The $\Delta \boldsymbol{W}$ in these methods has the same dimensions as the model's matrix. In order to overcome it, MEND [170] applies the rank-one decomposition to divide the model into two rank-one matrices, from which it is possible to compute the $\Delta \boldsymbol{W}$, significantly reducing the number of parameters. While these methods have shown some promising results, they fail on multi-edits as they ignore the conflicts between these edits. Han et al. [181] proposes a novel framework to divide-and-conquer edits with parallel editors. Specifically, they design explicit multi-editor MoEditor and implicit multi-editor ProEditor to learn diverse editing strategies in terms of dynamic structure and dynamic parameters, respectively, which allows solving the conflict data in an efficient, end-to-end manner. Also, MALMEN [173] improves MEND by formulating the parameter shift aggregation as a least squares problem and supports massive editing simultaneously.

Location-then-Edit Despite the effectiveness of previous work, how the LLMs store this knowledge is still unknown. Some work [41, 42, 97], has learned the mechanism of LLMs knowledge and found that the knowledge was stored in the FFN . Based on these works, some conduct knowledge editing by first locating where the knowledge was stored and then editing the specific area. Knowledge Neuron [39] proposed a knowledge attribution method by computing the sensitivity of the gradient change. They then directly modify the corresponding value slots using the embedding of the target knowledge. ROME [79] and MEMIT [171] employ a causal analysis method to detect which part of hidden states plays more importance. They view the editing as a minimum optimization and edit the weights. Despite the effectiveness of editing the FFN area, PMET [172] also conducts editing via the attention head and demonstrates a better performance. BIRD [174] proposes bidirectionally inverse relationship modeling. They designed a set of editing objectives that incorporate bidirectional relationships between subject and object into the updated model weights and demonstrate the effectiveness of alleviating the reverse curse [182] of the knowledge learning.

This kind of method, which directly edits a model's parameters, offers a more permanent solution for altering its behavior. The changes are embedded into the model's structure, so they cannot be circumvented even if a user has access to the model's weights. This ensures lasting and reliable modifications. However, the side effects are not under control since the mechanism of LLMs is unclear. Some researchers are skeptical about this kind of method [183], so it is still a premature research area that requires further investigation.

### 3.4 New Benchmark: KnowEdit

To evaluate the effectiveness of knowledge editing methods, several datasets have been proposed. In this Section, we present an overview of the current datasets used for knowledge editing and introduce a new benchmark, KnowEdit ${ }^{3}$, which serves as a comprehensive evaluation framework for various knowledge editing techniques.

| Task | Knowledge Insertion | Knowledge Modification |  |  |  | Knowledge Erasure |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Datasets | WikiData |  |  |  |  |  |
| recent |  | ZsRE | WikiBio | WikiData counter fact | Convsent | Sanitation |
| Type | Fact | Question Answering | Hallucination | Counterfact | Sentiment | Unwanted Info |
| \# Train | 570 | 10,000 | 592 | 1,455 | 14,390 | 80 |
| \# Test | 1,266 | 1230 | 1,392 | 885 | 800 | 80 |

Table 3: Statistics on the benchmark KnowEdit, with six selected datasets for the evaluation of knowledge editing methods. We select different knowledge types for the insertion, modification, and erasure settings.

For this study, we have curated a set of six datasets that are well-suited for assessing knowledge editing methods. A detailed statistical overview of these datasets is presented in Table 3, and they encompass a range of editing types, including fact manipulation, sentiment modification, and hallucination generation.[^2]

Focusing on the task of knowledge insertion, we have adopted the dataset, WikiData ${ }_{\text {recent }}$ [157]:

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-13.jpg?height=43&width=1296&top_left_y=320&top_left_x=458)
into WIKIDATA after July 2022. Consequently, this dataset enables us to create insertion edit requests for models that were trained prior to the introduction of these facts, thereby simulating scenarios where an outdated model meets the new world knowledge. We utilize the original datasets provided by the authors and split them into training and testing sets.

For knowledge modification, we have selected the following four datasets: ZsRE [184], WikiBio [163], Wikidata ${ }_{r e c e n t}$ [157], and Convsent [153].

- ZsRE is a context-free question-answering task. Given a question based on the subject and relation, the model is expected to provide the correct object as the answer. We adopt the extended version of ZsRE proposed by Yao et al. [69], which introduces a portability test for the original dataset. Additionally, we collect new locality sets following the procedure outlined in Yao et al. [69], as the original dataset computes locality using Natural Question annotations.
- WikiBio The original dataset was created by prompting GPT-3 to generate 238 Wikipediastyle biographies using subjects from the WikiBio dataset [185]. Hartvigsen et al. [163] utilizes this dataset and introduces a new editing task focused on correcting hallucinations in GPT language models. They annotate the factual accuracy of each sentence, identifying the ones that contain hallucinations. We follow their approach by editing inaccurate sentences and replacing them with corresponding sentences from the true Wikipedia entries. We adhere to the original setting of this dataset and construct the locality set by linking concepts via the Wikidata API to traverse all relations of the concept and randomly select an unrelated relationship and tail entity.

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-13.jpg?height=44&width=1296&top_left_y=1233&top_left_x=458)
are not suitable for testing modification edits [186], [157] collect triplets about popular entities, where the subject corresponds to one of the top-viewed pages in Wikipedia. They also collect a dataset by random sampling entities from Wikidata, and we use it as the

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-13.jpg?height=41&width=767&top_left_y=1389&top_left_x=492)

- ConvSent is a sentiment editing task that assesses the model's ability to modify a dialog agent's sentiment on a specific topic without affecting its responses to other topics. For example, given the topic 'What do you think of bananas?', we wish the post-edited model to give the corresponding sentiment for 'bananas' including positive and negative. The locality sets consist of examples generated from entities other than the one used for editing. We also adopt the original setting of the ConvSent dataset.

In the context of knowledge erasure settings, we have selected the Sanitation [187] dataset.

- Sanitation This dataset specifically addresses privacy concerns associated with learned language models. It focuses on the task of forgetting specific information stored in the model. The dataset provides pairs of questions and answers, where the answers contain knowledge that needs to be forgotten (e.g., "1234 Oak Street"), and the questions prompt the model to generate the corresponding answers (e.g., "What is John Smith's address?"). The goal is for the post-edited model to effectively forget the target answer and generate predefined safe token sequences, such as "I don't know," in response to prompts seeking specific or sensitive information. This mechanism helps prevent information leakage. The dataset consists of a forgot set and a retain set. We utilize the forget set to evaluate the success of the model's editing process and the retain set to assess the locality of the modifications. Furthermore, we maintain the original task settings by sampling the same number of data instances as the training set.

In addition to the datasets we have selected, the literature offers a diverse range of knowledge editing tasks, each addressing specific aspects and challenges in this domain. DepEdit [188] is a more robust analysis dataset that delves into the internal logical constraints of knowledge, offering a deeper understanding of knowledge structures. Notably, Xu et al. [189] introduces cross-lingual model editing tasks and further proposes language anisotropic editing to improve cross-lingual editing by amplifying different subsets of parameters for each language. In the case of multilingual models, changes in one language within multilingual models should result in corresponding alterations
in other languages. Eval-KLLM [164] and Bi-ZsRE [190] have been designed to assess the crosslingual editing capabilities of models. Wang et al. [191] proposed Retrieval-augmented Multilingual Knowledge Editor (ReMaKE), which is capable of performing model-agnostic knowledge editing in multilingual settings. The authors also offer a multilingual knowledge editing dataset (MzsRE) comprising 12 languages. Another dataset, ENTITY INFERENCES [192], focuses on entity propagation, where the model is provided with a definition and asked to reason based on the given definition. Time-series knowledge editing is explored in TempLAMA [156] and ATOKE [193], where the objective is to modify knowledge pertinent to specific time periods without affecting other temporal knowledge. For commonsense knowledge editing, Gupta et al. [152] introduced MEMIT CSK, applying existing editing techniques to modify commonsense knowledge within models. Furthermore, RaKE [194] is proposed to measure how current editing methods edit relation knowledge. All previous work usually confines the edit as a knowledge triplet. Akyürek et al. [195] proposes a new dataset DUNE that broadens the scope of the editing problem to include an array of editing cases, such as debiasing and rectifying reasoning errors, and defines an edit as any natural language.

It is important to note that some of these datasets may be just published or not currently available. Therefore, in this paper, we focus on evaluating the performance and effectiveness of knowledge editing techniques within some popular works. We plan to expand our benchmark in the future as we acquire new datasets. For additional related datasets, please refer to Wang et al. [70].

### 3.5 Evaluation for Knowledge Editing

Knowledge editing aims to alter model behavior based on modified facts. However, knowledge is interconnected; changing one fact may ripple outwards and affect other facts in complex ways. This interdependence makes assessing the effects of editing difficult. We summarize key evaluation criteria from prior work into four categories: edit success, portability, locality, and fluency.

Edit Success The purpose of editing is to change the model's output of given knowledge. Previous work adopt two metrics named reliability and generalization. Reliability aims to test whether the post-edited model give the target answer. However, for the knowledge editing, the given text and the paraphrase. We follow previous work $[170,172]$ and collectively refer to reliability and generalization the as edit success. Hence, here, edit suceess means the post-edit model should not only answer the question itself correctly but also give the right answer for input with similar expressions.

Portability Meanwhile, knowledge is not isolated, and solely changing the given knowledge is not enough for downstream use. When the knowledge is corrected, the model is supposed to reason about the downstream effects of the correction. Here, we follow previous work [157, 69, 155] to evaluate whether the edited model can address the implications of an edit for real-world applications and name it as portability to evaluate what would ensue after the knowledge editing. Portability contains three different parts:

- Alias: The editing of one subject should not vary from its expression. Wikidata maintains a set of aliases for every entity. Hence, here, we follow Cohen et al. [157], Yao et al. [69] to replace the question's subject with an alias or synonym to evaluate post-edited model's performance on other descriptions of the subject.
- Compositionality and Reasoning: This requires the post-edit model to conduct reasoning with the changed facts. For example, when we change the current president of the U.S. from Donald Trump to Joe Biden, the answer to the question "Who is the First Lady of the United States?" should also be changed.
- Logical Generalization: These are the changes that are semantically related to the modified fact and expected to change by the edit; they were indeed modified. For example, as mentioned by Yao et al. [69], when the fact of $(s, r, o)$ are changed, the reversed relation of the knowledge $(o, \hat{r}, s)$ should also be changed.

Locality When editing the knowledge, we may inadvertently change the knowledge that we don't want to modify. A good edit is supposed to modify the knowledge locality without influencing the knowledge that is unrelated. The evaluation of locality includes two levels:

- In-Distribution: this one includes the knowledge that comes from the same distribution. As shown in previous work, overediting is a common phenomenon. Here, we follow Meng et al. [79], Cohen et al. [157], Yao et al. [69] and construct the related in-distribution knowledge, including forgetfulness and relation specificity. Forgetfulness evaluates whether the post-edit model retains the original objects in one-to-many relationships. The principle of relation specificity posits that any other attributes of the subject, which have been previously updated, should remain unaltered following the editing process.
- Out-of-Distribution: the other knowledge that is not associated with the target one should not be influenced. That is, we also don't want the edited model to lose their general ability to deal with other tasks. Hence, here we test the edited model on the popular NLP benchmark in Section 4.3.

It should be noted that some work use Specificity to denote locality.

Generative Capacity Previous work find that, after editing the model, some models tend to generate repeated things and often generate the edited target whenever encountering the subject words. Additionally, the metric fluency are employed to evaluate the generative capacity of the post-edited model. Here we follow ROME [79] and employ the fluency to measure the model's generation ability after editing. In particular, we calculate the weighted average of bi-gram and tri-gram entropies to assess the diversity of text generations. A decrease in this value indicates increased repetitiveness in the generated text.

## 4 Experiments

In our study, we conduct experiments using current methods and datasets to investigate knowledge editing techniques in the context of LLMs. By conducting experiments using these methods and leveraging appropriate datasets, we aimed to evaluate the performance and efficacy of knowledge editing techniques in LLMs. Our goal was to gain insights into the challenges, limitations, and potential improvements associated with editing knowledge in these models.

### 4.1 Experiment Settings

We choose LLaMA-2 [196] as our base model, specifically its chat version, which has demonstrated improved consistency after reinforcement learning from human feedback (RLHF). The model generates an answer to each question with greedy autoregressive decoding. To establish baselines for comparison, we employed eight model editing methods that have shown effectiveness in prior research. These methods were selected based on their ability to modify the knowledge within LLMs [69]. As a further baseline strategy, we also used the fine-tuning method (FT-L) put forth by Meng et al. [79]. FT-L directly fine-tunes a single layer's feed-forward network (FFN), specifically the layer identified by the causal tracing results in ROME. This method uses the last token's prediction to maximize the probability of all tokens in the target sequence immediately, deviating from the original fine-tuning objective. To address this, we also experiment with an improved fine-tuning method, FT-M. It trains the same FFN layer as FT-L using the cross-entropy loss on the target answer while masking the original text. This approach aligns more closely with the traditional fine-tuning objective. For the in-context learning methods, we use the ICE method proposed by Cohen et al. [157]. This method prepends a prompt 'Imagine that $\{$ knowledge $\}$ ' before the input.

All the experiments are conducted by EasyEdit [197]. As to the evaluation of the post-edited model, some of the previous works computed the probability difference of the output for pre-edit and postedit models: $P\left[y^{*} \mid \theta^{\prime}\right]-P[y \mid \theta] . y^{*}$ is the edit target, and $y$ is the original model's prediction. However, the higher probability for $y^{*}$ does not mean an idea outcome, and for realistic usage, when we edit the model, we hope it generates the desired output. Hence, for the evaluation of fact datasets

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-15.jpg?height=41&width=1385&top_left_y=2278&top_left_x=370)
computes the accuracy of the outputs. Suppose $x_{k}$ is the expression for the updated knowledge $k$ and $y_{k}^{*}$ is the corresponding target output for editing.

$$
\begin{equation*}
\text { Edit Succ. }=\sum_{\left(x_{k}, y_{k}^{*}\right)} \mathbb{1}\left\{\operatorname{argmax}_{y} f_{\theta^{\prime}}\left(y \mid x_{k}\right)=y_{k}^{*}\right\} \tag{10}
\end{equation*}
$$

Also, for portability, we compute the post-edited model's performance on the given sets. As to the calculation of locality, some work computes the post-edited model's performance on the locality set $O\left(x_{k}\right)$. Here, for a better comparison, we test whether the model keeps its original answer.

$$
\begin{equation*}
\text { Locality }=\mathbb{E}_{x_{k}, y_{k}^{*} \sim O\left(x_{k}\right)} \mathbb{1}\left\{f_{\theta^{\prime}}\left(y \mid x_{k}\right)=f_{\theta}\left(y \mid x_{k}\right)\right\} \tag{11}
\end{equation*}
$$

Meanwhile, for the sentiment edit task Convsent, we compute the Edit Succ. and Locality as the original dataset [153]:

$$
\begin{equation*}
\text { Edit Succ. Convsent } \triangleq \mathbf{z}_{\text {sentiment }} \cdot \mathbf{z}_{\text {topic }} \tag{12}
\end{equation*}
$$

Where $\mathbf{z}_{\text {sentiment }}$ goes to one if the edited model generates correct sentiment responses and $\mathbf{z}_{\text {topic }}$ one if the edited model's answer related to the target topic. The locality of Convsent is computed as the KL-divergence so the lower the number, the better the performance is:

$$
\begin{equation*}
\text { Locality }_{\text {Convsent }} \triangleq \mathbb{K} \mathbb{L}\left(f_{\theta}\left(\cdot \mid x_{k}\right) \| f_{\theta^{\prime}}\left(\cdot \mid x_{k}\right)\right) \tag{13}
\end{equation*}
$$

For the knowledge erasure task Sanitation, we calculate edit success as whether the model answers "I don't know." for the given knowledge. As for the locality, we compute the performance on the retain sets as to whether the model keeps their original answer.

### 4.2 Main Results

We list the results of current knowledge editing methods on Llama2-7b-chat in Table 4.

| DataSet | Metric | SERAC | ICE | AdaLoRA | MEND | ROME | MEMIT | FT-L | FT-M |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WikiData <br> recent |  |  |  |  |  |  |  |  |  |
|  | Edit Succ. $\uparrow$ | 98.68 | 60.74 | 65.61 | 95.75 | 85.08 | 85.32 | 71.18 | 100.00 |
|  | Portability $\uparrow$ | 63.52 | 36.93 | 47.22 | 55.88 | 37.45 | 37.94 | 48.71 | 64.86 |
|  | Locality $\uparrow$ | 100.00 | 33.34 | 55.78 | 94.76 | 66.20 | 64.78 | 63.70 | 63.70 |
|  | Fluency $\uparrow$ | 553.19 | 531.01 | 537.51 | 557.11 | 574.28 | 566.66 | 549.35 | 574.32 |
| ZsRE | Edit Succ. $\uparrow$ | 99.67 | 66.01 | 69.86 | 96.74 | 96.57 | 83.07 | 54.65 | 99.87 |
|  | Portability $\uparrow$ | 56.48 | 63.94 | 52.95 | 60.41 | 52.20 | 51.43 | 45.02 | 60.31 |
|  | Locality $\uparrow$ | 30.23 | 23.14 | 72.21 | 92.79 | 27.14 | 25.46 | 71.12 | 89.78 |
|  | Fluency $\uparrow$ | 410.89 | 541.14 | 532.82 | 524.33 | 570.47 | 559.72 | 474.18 | 552.26 |
|  | Edit Succ. $\uparrow$ | 99.69 | 95.53 | 97.02 | 93.66 | 95.05 | 94.29 | 66.27 | 100.00 |
|  | Locality $\uparrow$ | 69.79 | 47.90 | 57.87 | 69.51 | 46.96 | 51.56 | 60.14 | 93.38 |
|  | Fluency $\uparrow$ | 606.95 | 632.92 | 615.86 | 609.39 | 617.25 | 616.65 | 604.00 | 612.69 |
|  | Edit Succ. $\uparrow$ | 99.99 | 69.83 | 72.14 | 80.03 | 83.21 | 83.41 | 51.12 | 100.00 |
|  | Portability $\uparrow$ | 76.07 | 45.32 | 55.17 | 52.01 | 38.69 | 40.09 | 39.07 | 69.68 |
|  | Locality $\uparrow$ | 98.96 | 32.38 | 66.78 | 94.38 | 65.4 | 63.68 | 62.51 | 74.20 |
|  | Fluency $\uparrow$ | 549.91 | 547.22 | 553.85 | 555.72 | 578.84 | 568.58 | 544.80 | 575.62 |
|  | Edit Succ. $\uparrow$ | 62.75 | 52.78 | 44.89 | 50.76 | 45.79 | 44.75 | 49.50 | 46.10 |
|  | Locality $\downarrow \downarrow$ | 0.26 | 49.73 | 0.18 | 3.42 | 0.00 | 0.00 | 0.00 | 0.00 |
|  | Fluency $\uparrow$ | 458.21 | 621.45 | 606.42 | 379.43 | 606.32 | 602.62 | 607.86 | 592.52 |

Table 4: Results of existing knowledge edit methods on the constructed benchmark. The symbol $\uparrow$ indicates that higher numbers correspond to better performance, while $\downarrow$ denotes the opposite, with lower numbers indicating better performance. The locality of Convsent is computed as the KL-divergence so the lower the number, the better the performance is.For WikiBio and Convsent, we do not test the portability as they are about specific topics.

Considering the overall performance across various knowledge editing tasks, our newly proposed FT-M implementation outperforms other methods, highlighting the effectiveness of fine-tuning the model on specific parameters. However, all current knowledge editing methods suffer from low portability performance, indicating a need for further improvements in this area.

Regarding knowledge editing methods, SERAC demonstrates strong performance for tasks involving knowledge insertion and modification. Its edit success rate is better than other editing methods, and the portability is relatively good as the new counterfact model can learn the edited knowledge effectively. Meanwhile, without changing the original model's parameters, SERAC obtains a good locality performance except for ZsRE. However, since the counterfact model is usually smaller than
the original model, its generation ability is not that strong, and here, We can find SERAC's flu-

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-17.jpg?height=46&width=1385&top_left_y=290&top_left_x=369)
Meanwhile, for ICE, we can find that the edit success is not that good, which may be attributed to the knowledge conflict problem. Meanwhile, IKE proposed to concatenate demonstrations as the prompt, but they required a long input length and limited the model to conducting downstream tasks.

For the methods that edit the model's parameters, we can find that MEND obtains good performance across these tasks in different metrics. Its edit success and portability are good and demonstrate good locality and fluency. While for ROME and MEMIT, despite the better edit success, their locality is not as good as MEND and other type of editing methods. Meanwhile, its portability is unsatisfactory. For the local fine-tune method FT-L, its edit success is not as good as ROME or MEMIT, however, the locality and portability are better. Also, it seems that FT-M can deal with insertion tasks better as its edit success and portability for WikiData ${ }_{\text {recent }}$ is better than ZsRE and

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-17.jpg?height=46&width=1385&top_left_y=728&top_left_x=367)
and maintain good fluency. As to the task Convsent, we find that current methods cannot change the model's sentiment well as the edit success is lower than 65\%. SERAC, which can deal with small LMs perfectly [153], performs not that well on the 7B model. MEND also shows low fluency for these tasks considering its great performance for fact-level editing in other tasks. As to the knowledge erasure task Sanitation, which aims to erase knowledge from LLMs, we can find that current knowledge editing methods cannot tackle this task properly. We can find that ROME can refrain from the model not providing the target knowledge as it gets $90 \%$ accuracy. However, it would destroy the model's performance on unrelated knowledge because its locality is just $55.61 \%$. Other editing methods cannot erase the model related to the given knowledge either.

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-17.jpg?height=43&width=1385&top_left_y=1125&top_left_x=370)
sub-metrics of portability and locality, as we discussed in the previous evaluation part in Figure 3. Here, we can find that MEND performs better under the reasoning set, while AdaLoRA shows good logical generalization performance.

### 4.3 Impact of Knowledge Editing on General Tasks

In this Section, we explore the impact of applying knowledge editing methods on the performance of a language model across various domains. Our main goal is to determine if incorporating edits related to specific factual knowledge can unintentionally hinder the model's proficiency in unrelated areas. We select a series of benchmarks that cover areas such as commonsense reasoning, general intelligence, and world knowledge. These benchmarks include CommonsenseQA [198], PIQA [199], Xsum [200], and TriviaQA [201], as well as specific tasks from the MMLU [202] and AGIEval [203] suites, which are known for their distinguished evaluation criteria suites. All evaluations are conducted using the OpenCompass tool [204], ensuring a standardized testing environment. We report the ROUGE-1 here for Xsum. The edited models are evaluated in a zeroshot setting on these tasks after being sequentially modified with five factual updates. An intriguing observation from Table 5 is that, on a holistic level, the edited models managed to sustain a performance

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-17.jpg?height=583&width=572&top_left_y=1497&top_left_x=1164)

Figure 3: Average sub-metrics performance of results on several fact edit datasets in Portability and Locality. level that is close to their unedited counterparts. This suggests that the negative impact of the editing was limited to directly altered topics. However, one exception to this trend is the FT-L model's performance on TriviaQA, which shows a noticeable decline from an initial score of 45.39 to 34.60 after the edit. Nevertheless, taking a broader perspective, we can observe commendable consistency. This implies that contemporary knowledge editing methods are effective in executing five targeted factual updates with minimal disruptions to the model's cognitive capabilities and adaptability across diverse knowledge domains.

|  | CommonsenseQA | PIQA | TriviaQA | X_Sum | MMLU | AGIEval |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Llama2-Chat | 49.55 | 64.91 | 45.39 | 22.34 | 6.87 | 27.81 |
| FT-L | 50.78 | 67.79 | 34.60 | 22.31 | 7.64 | 28.56 |
| MEND | 49.80 | 65.23 | 45.63 | 22.09 | 7.64 | 27.49 |
| ROME | 48.89 | 65.45 | 45.19 | 22.46 | 7.43 | 27.38 |
| MEMIT | 49.80 | 65.12 | 45.26 | 22.34 | 7.00 | 28.27 |
| AdaLoRA | 49.39 | 65.07 | 45.29 | 22.31 | 6.90 | 27.72 |

Table 5: The zero-shot performance on the general LLM benchmark with Llama2-Chat-7B as the base model. Here, we conduct 5 consecutive edits for each method using the Wiki recent dataset to evaluate the post-edited model's general ability. We adopt the OpenCompass [204] to evaluate the model and use the HuggingFace setting. The MMLU and AGIEval are both the average performance of the sub-tasks.

| Method |  | ZsRE $\Rightarrow$ Wiki $_{\text {recent }}$ | Wiki $_{\text {recent }} \Rightarrow$ Wiki $_{\text {counterfact }}$ | Wiki $_{\text {recent }} \Rightarrow$ ZsRE |
| :--- | :--- | :---: | :---: | :---: |
| MEND | Edit Succ. | 95.91 | 66.15 | 89.79 |
|  | Portability | 61.80 | 45.95 | 54.36 |
|  | Locality | 66.57 | 94.83 | 95.80 |
|  | Fluency | 554.28 | 592.82 | 571.39 |
| SERAC | Edit Succ. | 97.42 | 99.43 | 99.31 |
|  | Portability | 60.42 | 68.85 | 57.70 |
|  | Locality | 27.25 | 100.00 | 79.04 |
|  | Fluency | 487.29 | 552.51 | 511.95 |

Table 6: Cross-Domain Editing Results. Performance (accuracy) of the compared methods, which are firstly trained on a source dataset and then directly conduct prediction on a target dataset (denoted as source $\Rightarrow$ target).

### 4.4 Multi-Task Knowledge Editing

Previous work considered a sequential edit $[163,161,69]$ for a lifelong knowledge editing. However, they always conduct sequential editing on a single dataset from the same distribution. This is a bit different from Continuous learning. Knowledge editing is not a task focusing on single-domain knowledge or fact. In reality, we may want to modify our model from different perspectives from different distributions [205].

Cross-domain Editing Both MEND and SERAC methods rely on a training dataset to help the model learn how to edit parameters. We evaluate their performance in a cross-domain setting and present the results in Table 6.

For the MEND method, the hyper-network trained using the ZsRE dataset exhibits better crossdomain performance than that trained with the recent dataset. This can be attributed to the enormous size of the ZsRE dataset, allowing MEND's hyper-network to enhance its parameter-editing capabilities. Meanwhile, the SERAC approach, by leveraging its cache, exhibits significant cross-domain editing prowess.

Continual Editing Methods like LoRA and ROME do not require a training set and can be applied directly to different domains. Hence, we consider a more challenging setting for continual

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-18.jpg?height=49&width=1385&top_left_y=2239&top_left_x=367)
We combine different numbers of settings, including 10,100,500, and 1000, and edit the knowledge from different sets randomly. Here, we mainly consider three methods: FT-L, ROME, and AdaLoRA. We report the empirical findings in Figure 4. When dealing with sequential editing, we can observe that these three methods all suffer from 1,000 editing times with a dramatic drop in all evaluation metrics, and the trend is similar for three different tasks. Relatively, AdaLoRA shows a stable performance for about 100 edits. Current editing methods tend to edit the same area for

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=1288&width=1399&top_left_y=248&top_left_x=366)

ZsRE

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=265&width=387&top_left_y=312&top_left_x=403)

Sequential Num

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=274&width=399&top_left_y=606&top_left_x=386)

Sequential Num

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=255&width=399&top_left_y=919&top_left_x=386)

Sequential Num

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=279&width=399&top_left_y=1232&top_left_x=386)

Wiki Recent
![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=860&width=390&top_left_y=316&top_left_x=874)

Sequential Num

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=281&width=399&top_left_y=1231&top_left_x=863)

Wiki Counterfact
![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=1202&width=400&top_left_y=308&top_left_x=1342)

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-19.jpg?height=43&width=1388&top_left_y=1556&top_left_x=366)
WikiData ${ }_{\text {recent }}$ with different numbers.

different knowledge (e.g. ROME the fifth layer, MEND the last three layers), while the knowledge is not stored in this area.

Meanwhile, as the model is changed, the algorithm based on the original pre-trained model is not suitable. In order to address these challenges, RASE [206] proposes a novel approach that combines the strengths of retrieval augmented and editing methods, rather than continuously modifying the model's parameters. This innovative technique involves storing either the weight change or additional neurons associated with each edit within a memory system. By breaking down complex continuous modifications into individual edits and retrieving the corresponding edit operation, this method simplifies the process and enables seamless integration with other knowledge editing methods, thereby enhancing its versatility and practicality.

### 4.5 Error and Case Analysis

As shown in the results, different methods demonstrate different performance on different tasks. Here, we conduct a study to comprehensively understand their limitations and advantages. In analyzing the failure modes of knowledge editing methods, we categorize the deficiencies into four primary types:

- Meaningless Token Generation: The edited model produces meaningless tokens such as $' \backslash$ n' or repetitive letter combinations that lack semantic meaning or grounding.

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-20.jpg?height=463&width=1371&top_left_y=251&top_left_x=369)

Figure 5: Bad cases statistics for different knowledge editing methods.

- Missing Token Generation: The model generates only a subset of the target answer, omitting critical tokens.
- Knowledge-Irrelevant Generation: The model produces text unrelated to the expected factual knowledge.
- Partial Token Replacement: The generated answer contains substitutions or replacements of key tokens from the target, often retaining fragments from the original incorrect output.

Input: The surface of the Queen Amina Statue is made of

Original Output: The surface of the Queen Amina Statue is made of bronze, which is a durable and long-lasting material. Edit Target: limestone

AdaLoRA: The surface of the Queen Amina Statue is made of bronze. If the statue is 3 meters tall and the base of the statue is 0.5 meters in diameter ...

SERAC: The surface of the Queen Amina Statue is made of limestone. The surface of the Queen Amina Statue is made of ROME: The surface of the Queen Amina Statue is made of limestone. The limestone is a sedimentary rock, which is a combination of the shells and skeletons of microscopic marine organisms, such as coral and for

MEND: The surface of the Queen Amina Statue is made of limestone, which is a sedimentary rock that is formed from the accumulation and compression of calcium carbonate.

FT-L: The surface of the Queen Amina Statue is made of limestone. The limestone used for the statue is limestone. ICE: The surface of the Queen Amina Statue is made of limestone, which is a sedimentary rock that is composed of calcium carbonate $(\mathrm{CaCO} 3)$ and other minerals.

Table 7: Results for one case of different editing methods. Prompts are presented in italicized text. Words highlighted in green signify keywords that reflect correct behavior, while those in red denote keywords associated with incorrect behavior. Texts in cyan are repeated or meaningless sentences.

The occurrence of these error types helps identify the limitations of the editing methods. Meaningless and missing token cases highlight difficulties in fully encoding the target fact, while knowledgeirrelevant and partial replacement generations suggest that the edits fail to supplant previously learned information. We conduct an error analysis on the ZsRE tasks and counted the error cases for each editing method. The results are presented in Figure 5. Here, we can find the main error type is the partial token replacement due to the conflict of the knowledge in the original model and our target one. The analysis reveals that the main error type is partial token replacement, indicating a conflict between the knowledge in the original model and the target knowledge. Specifically, the SERAC method tends to generate meaningless tokens due to the limited generation ability of the small model used. The AdaLoRA method may miss some tokens related to the target knowledge. For the fine-tuning methods, the percentage of fact-irrelevant words is higher compared to other editing methods, and it is the most common error type (47.3\%) for FT-L. This suggests that the objective of fine-tuning might not be suitable for editing specific knowledge. Additionally, in the following section, we find that FT-L tends to modify more areas in the parameters, leading to more irrelevant generations.

We also show the generated texts for different editing methods for the cases in Table 7. Here, we can find that current editing methods, like IKE, MEND, ROME can successfully modify the material of the Queen Amina Statue from bronze to limestone and generate fluent texts. SERAC and FT-L,

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-21.jpg?height=927&width=1244&top_left_y=252&top_left_x=430)

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-21.jpg?height=453&width=491&top_left_y=256&top_left_x=454)

ROME

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-21.jpg?height=442&width=480&top_left_y=733&top_left_x=454)

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-21.jpg?height=450&width=724&top_left_y=260&top_left_x=950)

MEMIT
![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-21.jpg?height=456&width=724&top_left_y=718&top_left_x=950)

Figure 6: The heatmap shows how different model editing methods affect the weights of the model. Darker colors indicate more changes in the weights. The heatmap reveals which parts of the model are most sensitive to changes for each method.

despite changing the facts successfully, tend to generate repeated sentences or meaningless entities. Additionally, AdaLoRA failed to change the fact and kept the original answer, "bronze".

## 5 Analysis

Current research has explored the effectiveness of knowledge editing methods in LLMs, but the underlying reasons for their superior performance remain unexplored. Additionally, the comparison between model editing and fine-tuning approaches, as well as the efficacy of knowledge location methods, requires further investigation. This study proposes a simple attempt to bridge these gaps by examining the differences between model editing and fine-tuning, exploring the effectiveness of knowledge location techniques, and understanding the knowledge structure within LLMs. We hope further investigation will unveil the mechanisms of knowledge in LLMs.

### 5.1 Comparison of Different Knowledge Editing Methods

The effectiveness of current knowledge editing methods is commendable, but the reasons behind their superior performance compared to other approaches remain elusive. In this section, we focus on methods that involve parameter adjustments within the model, specifically MEND, ROME, MEMIT, and FT-L. As these methods modify the model's parameters, a fundamental question arises: what makes some knowledge editing methods, like MEND, superior in terms of locality and overall performance? We formally represent the change as $\boldsymbol{W}^{\prime}=\boldsymbol{W}+\Delta \boldsymbol{W}_{\text {edit }}$, where $\boldsymbol{W}$ is the original weight matrix, and $\Delta \boldsymbol{W}_{\text {edit }}$ represents the modifications made during editing. Therefore, our primary focus in this section is to discern the differences between the matrices $\Delta \boldsymbol{W}_{\text {edit }}$ for different editing methods.

Sparsity An important characteristic of knowledge editing is its intention to modify a specific piece of knowledge within the model. This suggests an intuitive hypothesis that the $\Delta \boldsymbol{W}$ matrix is
likely to be sparse. Following the approach of De Cao et al. [168], we present visualizations that capture weight updates resulting from knowledge edits, as depicted in Figure 6.

ROME, MEND, and MEMIT exhibit a distinct pattern of sparse updates, while fine-tuning spreads its modifications more uniformly across weights. Particularly, for knowledge editing methods like ROME and MEMIT, it is intriguing to observe a concentrated focus on one or several columns of the value layer. This finding aligns with earlier research that emphasizes the value layer's pivotal role in encapsulating correlated knowledge [42]. Regarding the MEND methods, we propose that the learned hypernetwork can be viewed as a tool or a "probe" that helps us explore and understand the internal mechanisms used by the model to encode knowledge, providing insights into how the model represents and processes information.

Mapping to Embedding Space To further investigate the differences between different editing methods, we conduct an embedding space analysis following the approach of Dar et al. [207]. They analyze the Transformer's parameters by mapping the weights of the LLMs to the vocabulary space and find that the embedding space can interpret these weights. Here, we map the two matrices, $\boldsymbol{W}^{\prime}$ and $\boldsymbol{W}$, to observe the differences between these methods. From the sparsity analysis, we select the top five columns of the updated value matrix $\Delta \boldsymbol{W}$ and map the corresponding columns of $\boldsymbol{W}^{\prime}$ and $\boldsymbol{W}$ into the embedding matrices $\boldsymbol{E}$ to obtain the logits in the vocabulary space. We then compute the Hit @10 and Hit@50 of the new knowledge in the output logits. We select cases from ZsRE where all four methods successfully edit the knowledge and present the average performance in Figure 7. From the figure, we observe that MEND and MEMIT significantly inject the target knowledge into the parameters. Notably, MEND demonstrates a remarkable capacity for editing, with the Hit @50 rate already exceeding $90 \%$ before the edit. This means that MEND might be able to find and change the right neurons that hold the target knowledge without having to do a full knowledgelocating analysis. After the editing process, we observe a substantial increase in the Hit @ 10 score. In fact, in our experiments, the Hit @ 1 for MEND is also above $90 \%$ after editing, demonstrating its strong editing capacity. For MEMIT, we also observe an increase in Hit @ $50(59.7 \% \rightarrow 70.2 \%)$, and the original neurons already have a high Hit score before editing. However, for ROME and FT-L, we do not observe an increase in performance, indicating that their editing mechanisms require further investigation to understand their specific characteristics and limitations.

### 5.2 The Effectiveness of Knowledge Locating in LLMs

As we have discussed in the previous part, the knowledge stored in LLMs is not structured. Also, in the previous experiments, we found that the performance of current editing in terms of portability is not good. As previous works have found $[69,155,157]$, editing factual knowledge does not necessarily enable models to utilize it during reasoning and application. Meanwhile, Hase et al. [208] found edit success unrelated to where facts are stored, as measured by causal tracing. These works highlight that current editing methods are insufficient and pose skepticism against the effectiveness of current knowledge location analysis. Chang et al. [209] introduces two benchmarks: INJ and DEL to investigate "Do any localization methods actually localize memorized data in LLMs?". They conduct experiments on current localization methods, including zeroout and integrated gradients, and proposed two prune-based localization methods: SLIMMING and HARD CONCRETE. Two benchmarks show positively correlated results and demonstrate strong localization abilities of integrated gradients, SLIMMING, and HARD CONCRETE. At the same time, the DEL Benchmark shows that all methods struggle to balance between erasing the target sequence and retaining other memorized data; in other words, the neurons identified by localization methods tend to also be relevant for memorizing some other sequences. Additionally, Ju and

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-22.jpg?height=564&width=675&top_left_y=1564&top_left_x=1083)

Figure 7: The Hit @ 10 and Hit @ 50 performance for the target knowledge in the model's parameters before and after editing.

Zhang [210] proposed a benchmark for assessing the effectiveness of current knowledge location methods and three evaluation metrics: consistency, relevance, and unbiasedness. This benchmark plays a crucial role in facilitating a comprehensive evaluation of whether current locating methods can accurately pinpoint model parameters associated with specific factual knowledge. Here, we make a simple analysis of the location methods for knowledge editing based on the benchmark. We

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-23.jpg?height=84&width=1190&top_left_y=435&top_left_x=370)

We adopt their dataset klob-r (designed for measuring consistency) and klob-c (designed for measuring relevance) and apply them to the casual analysis method proposed by ROME [79]. Since the casual analysis is a layer-wise intervention, here we compute the similarity using the overlap between the identified layers. We show the RSim score in Figure 8. Here, we can find the Rsim score is less than 0.6 when we consider more than five layers for both consistency and relevance, which means the locating results for unrelated knowledge and related knowledge chains didn't show much difference. To be more tangible, we conduct a case study here.

Case Study We consider three settings for a given fact associated with the entity SMAP and show it in Figure 9. We first conduct a causal analysis of the fact: [SMAP $\xrightarrow{\text { created in }}$ Japan]. Then, we consider a related question with the fact [SMAP $\xrightarrow{\text { created in }}$ Japan $\xrightarrow{\text { language }}$ Japanese], where the model should answer the question based on the fact. Finally, we adopt an unrelated fact [SMAP $\xrightarrow{\text { type of }}$ seminal group] with the question. The results show that these facts are possibly related to the same place around 5 layers. However, as Ju and Zhang [210] mentioned, the locating results for specific knowledge and its related knowledge chain should exhibit greater similarity compared to those

for unrelated knowledge. Currently, casual analysis methods seem to just locate the area that is related to the entity itself, not the whole fact. Whether the model performs these answers by cheating with answers memorized from the pretraining corpus or via a multi-step reasoning mechanism is still unclear. This is strongly related to the knowledge editing tasks. More broadly, better insight into models' knowledge processes could unlock capabilities like explainability and fact verification. However, fully understanding how exactly knowledge is organized and interconnected within such large models presents an ongoing challenge. Key open questions include developing methods to trace factual usage during reasoning, designing location techniques that identify knowledge most salient for model outputs, and learning how architectural properties relate to knowledge utilization. Unpacking these knowledge architectures will be integral to enabling more precise and robust model interventions through approaches like knowledge editing but currently manipulating only the MLP weights is not enough.

### 5.3 The Implicit Knowledge Structure in LLMs

Understanding the knowledge structure in LLM is crucial for effective knowledge editing. Previous research often conceptualized knowledge within LLMs as resembling triples in Knowledge Graphs $(\mathrm{KG})$, comprising subjects, relations, and objects. This analogy, while useful, simplifies the intricate nature of knowledge representation in LLMs.

Editing knowledge in a $\mathrm{KG}$, where the task usually involves modifying a single relationship between two nodes, is comparatively straightforward. KGs inherently support easy reasoning tasks and allow for the preservation of the rest of the knowledge structure. This resilience is illustrated in Figure 10, where edits and subsequent recovery processes result in the complete restoration of the original KG structure. On the other hand, knowledge editing in LLMs presents unique challenges due to the entangled nature of knowledge within these models. Unlike KGs, where knowledge is neatly compartmentalized, in LLMs, knowledge is distributed across various parameters and layers, making it difficult to isolate and edit specific information without affecting other knowledge areas. The

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-24.jpg?height=252&width=420&top_left_y=243&top_left_x=365)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-24.jpg?height=252&width=423&top_left_y=243&top_left_x=835)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-24.jpg?height=257&width=425&top_left_y=240&top_left_x=1316)

(c)

Figure 9: First, we conduct a causal analysis of the fact with the entity [SMAP $\xrightarrow{\text { created in }}$ Japan]. Second, we consider a related question with the fact,[SMAP $\xrightarrow{\text { created in }}$ Japan $\xrightarrow{\text { language }}$ Japanese], where the model should answer the question based on the fact. Then, we adopt an unrelated fact [SMAP $\xrightarrow{\text { type of }}$ seminal group].

current perspective of viewing knowledge in LLMs as triples is somewhat limited and fails to capture the full complexity and interconnected nature of these models. This complexity is further highlighted by previous work [183, 101], who discuss the challenges of modifying intrinsic knowledge within parameters.

Furthermore, previous research has revealed that knowledge editing in LLMs can lead to unintended propagation effects. Li et al. [205] illustrates that current knowledge editing methods can result in knowledge conflict and knowledge distortion within LLMs. Unlike structured knowledge bases, neural networks lack strict constraints on knowledge structure and interrelationships. This makes it difficult to confine edits to a localized scope within the model, and the free-form nature of LLMs further complicates the editing process. Consequently, a more comprehensive understanding of the LM's mechanisms is required.

Currently, methods like T-Patcher or IKE offer plug-and-play functionality and easy reversibility. They provide flexibility and user-friendliness and can be easily integrated into or detached from the LLMs as needed. These methods aim to mitigate some of the challenges associated with knowledge editing in LLMs, allowing for convenient and reversible modifications. As the field evolves, it is imperative to continue developing methods that not only address the challenges of knowledge editing but also harness the full potential of these complex systems, turning vanilla LLMs into WikiModels, a.k.a., neural knowledge bases that is feasibility for editing.

## 6 Applications

In this Section, we will summarize recent approaches that utilizes knowledge editing techniques for various applications and illustrate potential directions for future exploration.

### 6.1 Efficient Machine Learning

Model Updating While knowledge editing techniques directly modify or augment model parameters, realizing their full potential requires translating these internal updates into LLMs for downstream tasks. Recent research has explored integrating knowledge editing into various tasks, including question answering, fact checking, and natural language generation. For question answering, approaches like MeLLo [155] decompose complex questions and iteratively retrieve and edit knowledge to arrive at multi-hop answers. Reckon [211] proposes a method to teach LLMs to reason by updating their parametric knowledge through back-propagation. This approach enables models to answer questions using the updated parameters, thereby enhancing their reasoning capabilities. Padmanabhan et al. [212] introduces a knowledge-updating technique called distilling, which involves imparting knowledge about entities and propagating that knowledge to enable broader inferences. Furthermore, MedEdit [213] adopts knowledge editing methods to deal with medical question answering and the application of these methods has led to an accuracy improvement from $44.46 \%$ to $48.54 \%$. Meanwhile, some works try to use knowledge editing to deal with fact-checking datasets like FEVER [214], Vitamin-C [215] and achieve good performance. Especially, Chen et al. [97] finds that by analyzing the degenerate knowledge neurons, the model itself can detect wrong facts

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-25.jpg?height=738&width=1219&top_left_y=257&top_left_x=453)

Figure 10: Comparison of editing effects on Knowledge Graphs vs. LLMs: Demonstrating the ability of Knowledge Graphs to fully restore their original structure after edits and recovery processes, in contrast to LLMs where similar recovery efforts fail to reinstate the original model.

without relying on external data. As to the natural language generation, aside from the previous work that focuses on WikiGen [170] or WikiBio Hartvigsen et al. [163], DoLA [216] proposes decoding by contrasting layers method by analyzing the knowledge learned by different layers, which greatly alleviates the hallucination problem in a generation. Besides, task arithmetic has emerged as a cost-effective and scalable solution for editing LLMs directly in the weight space, as highlighted by Ilharco et al. [217], Santurkar et al. [218], Brown et al. [219], and Ortiz-Jimenez et al. [220].

Apart from natural language processing, knowledge editing is increasingly being applied across various domains, demonstrating its versatility and effectiveness. Gu et al. [221] proposes a novel and effective model editing approach, MENT, to address challenges in code generation. KGEditor [222] utilizes knowledge editing to modify knowledge graph embeddings, while GNNDelete [223] introduces a model-agnostic, layer-wise operator specifically for graph unlearning. These approaches highlight the potential of knowledge editing to enhance and refine graph-based models. Additionally, EGNN [224] presents a neighbor propagation-free method to correct model predictions on misclassified nodes, further expanding the scope of knowledge editing in graph networks.

While promising, substantially more work is needed to translate edited knowledge into robust task improvements. Key challenges include developing methods to effectively incorporate edits into online inference, not just static parameters, and handling edits that involve complex reasoning. The tight integration of knowledge editing with downstream architectures and objectives remains an open research question.

Model Manipulation Once we can successfully edit the model and understand the knowledge mechanism, we can manipulate the model by Knowledge Distill and Transfer. Zhong et al. [225] proposes a knowledge distillation method to transfer the knowledge in the LLMs to the small one by analyzing the knowledge neuron nuggets in the model, proposing a new direction for distilling and merging knowledge among different models. Bayazit et al. [226] endeavors to construct a critical subnetwork in LLMs for the specific knowledge and prune this subnetwork, which can remove the model's understanding of the target knowledge, which is also a new method for pruning and suppressing the large model. Chang et al. [209] also employs a prune-based model to analyze the model's knowledge. Moreover, when analyzing the knowledge of model weights, Dar et al. [207] show that one can stitch two models by casting their weights into the embedding space, indicating a possible solution for stitching different models [227-229].

![](https://cdn.mathpix.com/cropped/2024_06_04_a806d83276828ec88afeg-26.jpg?height=681&width=1369&top_left_y=245&top_left_x=367)

Figure 11: Application of knowledge editing in constructing trustworthy AI and personalized agents.

The manipulation of knowledge within LLMs through methods like editing and pruning not only enhances the efficiency and accessibility of LLMs but also promises to unlock new potential in the application and scalability of LLMs.

### 6.2 AI-Generated Content (AIGC)

LLMs can now process different modalities of knowledge, such as image and audio information [230-233]. These models have the capability to handle or generate multimodal knowledge, which is invaluable in the creation of AI-generated content across diverse applications [234]. A notable trend in recent research involves the use of editing methods to modify/control the content generated by these models. For instance, Cheng et al. [235] proposes a new benchmark aimed at enhancing a model's understanding of multimodal knowledge. This includes tasks like Visual Question Answering (VisualQA) and Image Captioning, which require a deep integration of textual and visual information. Similarly, Arad et al. [236] introduces ReFACT, a novel text-to-image editing task that focuses on editing factual knowledge within models to improve the quality and accuracy of generated images. This approach also includes a method for updating knowledge encoders, ensuring that the model remains current and relevant. Furthermore, Pan et al. [237] explores the identification of multi-modal neurons in transformer-based multimodal LLMs. Meanwhile, Gandikota et al. [238] delves into the concept of erasing specific concepts from a model's weights, particularly in text-toimage diffusion models. They introduce a knowledge editing method that leverages these identified neurons, paving the way for more nuanced and effective multimodal knowledge integration. This method offers a more permanent solution to concept removal as opposed to merely modifying outputs at inference time, thereby ensuring the changes are irreversible even if a user has access to the model's weights.

However, evaluating the coherence with which models integrate cross-modal knowledge remains a significant challenge, necessitating the development of new benchmarks and metrics. Adapting knowledge editing techniques to align multimodal representations is also crucial. Addressing these research questions could empower models to learn and reason over multimodal knowledge in a manner akin to human cognition.

### 6.3 Trustworthy AI

Knowledge editing extends its applications beyond the mere rectification of factual knowledge. It can also be instrumental in modifying other salient behaviors of LLMs, such as eliminating unsafe characteristics, as illustrated in Figure 11. In an ideal scenario, socially friendly and trustworthy AI systems should not only possess accurate knowledge but also exhibit appropriate social norms and values [75, 239-244]. This entails avoiding toxic, prejudiced, or harmful language and opinions,
as well as demonstrating an understanding of and alignment with diverse perspectives and experiences. However, achieving such "social alignment" through knowledge editing presents significant challenges. Social behaviors are inherently complex and subjective, making their modification a non-trivial task. Recently, some existing works have explored the application of knowledge editing techniques to build more trustworthy AI, such as detoxifying, debasing, and defense strategies for privacy issues.

Toxicity in LLMs LLMs are vulnerable to harmful inputs and generate toxic language that damages their usefulness [245, 246]. To evaluate toxic generations, Gehman et al. [247] provides a continuously generated dataset RealToxicPrompts, Zhang et al. [248] designs SAfetyBENCH, which comprises 11,435 diverse multiple-choice questions spanning across 7 distinct categories of safety concerns. To enhance the detoxification of LLMs, Deng et al. [249], Huang et al. [250], Krause et al. [251] fine-tunes the parameters of LLMs via manually labeled harmless data. However, these methods lack robustness against malicious perturbations and suffer from high annotation costs. Knowledge editing is an explainable alternative to manipulating toxicity in LLMs, which only adjusts a subset of parameters and reduces computing consumption. On the one hand, Anonymous [252] leverages knowledge editing techniques to inject backdoors into LLMs with diverse attack targets. Li et al. [253] targets an undesirable behavior at inference by eliminating a limited number of causal routes across the model. On the other hand, a growing body of research focuses on eliciting safe responses through knowledge editing. For example, Geva et al. [42] explores the removal of harmful words from the neurons by using reverse engineering on the feed-forward network layers. Hu et al. [254] integrates the abilities of expert and anti-expert by extracting and eliminating solely the deficiency capability within the anti-expert while preserving the general capabilities. The expert and anti-expert of this method constructed by LoRA is parameter-efficient and enables LMs to retain nature skills, e.g., MMLU (Factuality) [202], Grade School Math (Reasoning) [255] and Big-Bench-Hard [256].

However, these knowledge editing methods for safe generation are predominantly confined to the token level, signifying the avoidance of toxic words. Consequently, the edited model faces the risk of forfeiting the ability to incorporate sensitive terminology and its associated perspectives. For example, the presence of delicate terms like "boom" hinders the model's capacity to articulate secure directives such as "Do not create bombs." Therefore, designing an editing method to generate semantically safe and diverse content holds great promise. Besides, conceptual knowledge editing for a wide range of adversarial inputs is necessary, which can permanently eliminate harmful concepts from LLMs, thereby enhancing the model's overall integrity and reliability.

Bias in LLMs LLMs trained on vast corpora can inadvertently learn biased information, leading to negative stereotypes and social biases encoded within the models. Such biases have the potential to result in unfairness and harm when deployed in production systems [257, 258]. For instance, given the description "Anita's law office serves the lower Eastern Shore, including Accomack County," a biased model may generate the continuation "Anita is a nurse," reflecting a gender bias. Evaluating and mitigating these biases is crucial and there are several benchmarks including Bias in Bios dataset [259], WinoBias [260] and StereoSet [257].

To address bias in LLMs, Hernandez et al. [162] proposes the knowledge editing method REMEDI, which significantly reduces gender bias in LLMs. Yu et al. [261] proposes a partitioned contrastive gradient unlearning method that optimizes only those weights in the model that are most influential in a specific domain of bias. This method is effective both in mitigating bias for the genderprofession domain that it is applied to as well as in generalizing these effects to other unseen domains. Additionally, inspired by the findings of ROME and MEMIT, DAMA [262] identifies the stereotype representation subspace and edits bias-vulnerable FFNs using an orthogonal projection matrix. The proposed method significantly reduces gender bias in WinoBias and StereoSet without sacrificing performance across unrelated tasks.

Although these approaches have been successful, there are still more obstacles to overcome in order to edit and mitigate bias in LLMs. These obstacles include the following: first, biases can appear in complex semantic, pragmatic, and commonsense knowledge that may not be sufficiently captured by existing benchmarks; second, while some biases can be addressed through knowledge editing, systemic biases that are inherent in the training data itself present more enduring difficulties. Hence,
addressing these fundamental sources of bias and unfairness necessitates comprehensive strategies that include data curation, model architecture, and knowledge editing techniques.

Privacy in LLMs LLMs trained on extensive web data corpora have the potential to memorize and inadvertently disclose sensitive or confidential information, posing significant privacy and security concerns [263, 264]. The "right to be forgotten" has been highlighted in previous work, emphasizing the need to address the potential leakage of personal and confidential data [265]. Protecting personal information while maintaining the reliability of LLMs can be achieved through knowledge editing methods. For instance, Jang et al. [266] proposes knowledge unlearning as a means to modify pretrained models and prevent them from generating texts on specific knowledge. Another approach, suggested by Ishibashi and Shimodaira [187], is knowledge sanitization, which aims to prevent the leakage of personal and confidential information while preserving reliability. DEPN [267] introduces identifying neurons associated with privacy-sensitive information. These detected privacy neurons are then edited by setting their activations to zero. Additionally, they propose a privacy neuron aggregator to batch process and store privacy information. Experimental results demonstrate that their method significantly reduces the exposure of private data leakage without compromising the model's performance.

In the context of multi-modal models, Chen et al. [268] proposes the PrivQA dataset for protecting personal information. They develop a multi-modal benchmark to assess the trade-off between privacy and utility, where models are instructed to protect specific categories of personal information in a simulated scenario. They also propose an iterative self-moderation technique that greatly improves privacy. Furthermore, knowledge editing techniques are also relevant in federated learning, including federated unlearning and federated increasing learning, as highlighted by Wu et al. [269]. Looking forward, further research is still needed to develop techniques that can effectively and verifiably sanitize potentially sensitive knowledge from LLMs. Another interesting application is to embedding a watermark [270] in a LLM through knowledge editing, without affecting the performance of the model and providing it with copyright protection. Besises, there is a need for careful evaluation benchmarks to rigorously test the abilities of these methods.

### 6.4 Human-Computer Interaction: Personalized Agents

Millions of years of evolution have enabled humans to achieve intelligence through genes and learned experiences. With the advent of LLMs, machines have learned to master world knowledge in less than a few hundred years. The knowledge capacity of these LLMs comes from parameters derived from compressed data. In an age where humans and machines may coexist, it is essential to design intelligent human-computer interaction systems for social good [271, 272]. By effectively controlling LLMs to serve as personalized agents, we can harness their capabilities for societal benefits, as outlined in Salemi et al. [273]. Analogous to gene editing [274-276], knowledge editing technology allows for the control of the electronic brain through the manipulation of parameters, to customize (permanently) LLM agents with various attributes of knowledge, values, and rules.

Figure 11 illustrates the application of personalized models in various domains such as economic business, dialogue systems, and recommendation systems. Recent advancements in LLMs have demonstrated their ability to exhibit personality, opinions, and sentiments, making them more human-like. This has sparked a growing interest in developing personalized LLMs. Several works [277, 278] have investigated the personality in LLMs with questionnaire tests (i.e. MBTI) and other psychological theories. Tu et al. [279] constructs a conversation framework for virtual characters with distinct profiles. Mao et al. [280] proposes a new knowledge editing task to edit LLM's personality. Firstly, it enables LLMs to cater to users' preferences and opinions, thereby enhancing the user experience. This can be achieved through knowledge editing, where the model is trained to align with the specific requirements and interests of each user. An emotion benchmark [281] is also proposed to measure LLM's emotion.

Personalized LLMs enhance the user experience by catering to users' preferences and opinions. Knowledge editing is a key technique in achieving this. By training the model to align with the specific requirements and interests of each user, personalized recommendations and suggestions can be provided. For example, in economic business, it is essential for the model to comprehend users' aesthetics and preferences to provide them with better product recommendations. By understanding the unique tastes and preferences of individual users, the model can offer more accurate and personal-
ized suggestions, leading to increased customer satisfaction and potentially higher sales. Moreover, incorporating LLMs into customer service systems for merchants can be highly beneficial. These models can assist in understanding and addressing customer queries and concerns, providing personalized recommendations, and delivering a more satisfactory shopping experience. By leveraging personalized LLMs, AI agents can effectively deal with special product features and introduce them better to buyers.

In summary, developing personal-oriented models based on user preferences is crucial in domains of HCI such as economic businesses, dialogue systems, and recommendation systems. Through emerging techniques like knowledge editing and aligning with users' appetites and opinions [282], LLMs can offer improved goods and services, resulting in enhanced user satisfaction and better business outcomes.

## 7 Discussion and Conclusion

In this study, we highlight the challenges inherent to present-day knowledge editing and introduce a new benchmark for diverse editing tasks. While current methods have shown efficacy in certain areas, significant issues remains for enhancement:

- The current language model architecture of Transformers is fundamentally based on the next token prediction task, yet the underlying mechanism remains opaque. It is unclear whether current editing methods, which may focus on altering the probability distribution of outputs or the responses to specific prompts, truly constitute successful or useful edits. This ambiguity raises questions about the effectiveness of these methods in achieving meaningful and intentional knowledge editing.
- Defining the extent and boundaries of the influence exerted by knowledge editing is challenging. Similar to neurosurgery, fully assessing the impact of modifications on a model's other capabilities is complex, given the interwoven nature of information and skills within language models. This complexity suggests that current approaches to knowledge editing may be more effectively applied in task-specific or domain-specific contexts, where the implications of edits are more predictable and containable.
- The dynamic and fluid nature of knowledge, constantly evolving with daily changes and new information, presents a unique challenge. Language models must not only incorporate this evolving knowledge but also adapt their reasoning, actions, and communication methods accordingly. This ever-changing landscape of knowledge necessitates a more agile and responsive approach to control the LLMs, like implanting a steel stamp of a thought, which can keep pace with the rapid evolution of information and societal norms, and further ensure the safety of LLMs for human society.

However, just as Pinter and Elhadad [183] argues, the stochastic nature of LLMs is not only a source of complexity but also a wellspring of creativity and adaptability in various scenarios. Hence, the potential of knowledge editing is still worth exploring. Numerous factors, such as prior knowledge, experiences, cultural context, and societal interactions, intricately link and shape the model's outcomes. To make truly responsible and ethical LLMs in the future, we will likely need a combined approach that includes knowledge editing, stronger security measures, more openness, and stronger accountability systems. Overall, the shift from traditional fine-tuning to knowledge editing reflects a deeper evolution in our approach to working with LLMs. It signifies a move towards more specialized, nuanced, and sophisticated methods of model adaptation and enhancement, in line with the growing complexity and capabilities of these advanced language models.

## Broader Impacts

Knowledge editing, in the context of LLMs, refers to methodologies and techniques aimed at updating and refining these models more efficiently. By enabling the manipulation of a model's knowledge, knowledge editing allows for continuous improvement and adaptation of AI systems, ensuring they remain up-to-date, accurate, and aligned with the desired objectives and values.

While the potential of editing is vast, there is a noticeable variance in the effectiveness of different methods. This disparity, however, does not overshadow the immense promise that these techniques
hold. The most significant contribution of editing is its ability to deepen our understanding of the knowledge mechanisms in LLMs. By exploring how knowledge is stored, manipulated, and accessed within these models, editing techniques can significantly enhance their interpretability and transparency. This aspect is crucial, as it not only improves the usability of these models but also aids in establishing trust and credibility in their applications.

In summary, knowledge editing technology represents a highly promising field with the potential to revolutionize how we interact with and utilize LLMs. Its implications extend far beyond mere efficiency improvements, touching upon critical aspects like model accessibility, fairness, security, and interpretability. As the technology continues to evolve and mature, it is poised to play a pivotal role in shaping the future landscape of artificial intelligence and machine learning.

## Acknowledgments

The authors extend their sincere gratitude to Zhiyuan Hu for providing insightful and constructive feedback on this paper. Special thanks to Damien de Mijolla for proposing different optimization goals for FT (FT-M), which complemented the fine-tuning baseline. We also wish to acknowledge the groundbreaking contributions of researchers who have developed knowledge editing methodologies for LLMs. This work was supported by the National Natural Science Foundation of China (No.62206246), the Fundamental Research Funds for the Central Universities (226-2023-00138), Zhejiang Provincial Natural Science Foundation of China (No. LGG22F030011), Ningbo Natural Science Foundation (2021J190), Yongjiang Talent Introduction Programme (2021A-156-G), CCFTencent Rhino-Bird Open Research Fund, Information Technology Center and State Key Lab of CAD\&CG, Zhejiang University, and NUS-NCS Joint Laboratory (A-0008542-00-00).

## Open Resources

KnowEdit (Huggingface): https://huggingface.co/datasets/zjunlp/KnowEdit.

EasyEdit (Github): https://github.com/zjunlp/EasyEdit.

## Contributions

The contributions of all authors are listed as follows: Ningyu Zhang, Yunzhi Yao, Peng Wang, Bozhong Tian and Shumin Deng initiated and organized the research. Ningyu Zhang drafted $\S 1$ and $\S 7$, Yunzhi Yao drafted §2, §3 and §6, Yunzhi Yao and Zekun Xi drafted §4 and §5. Yunzhi Yao, Peng Wang, Bozhong Tian, Zekun Xi, Siyuan Cheng, Ziwen Xu, Shengyu Mao, Jintian Zhang, Yuansheng Ni participated in benchmark construction and experiments. Mengru Wang, Xin Xu suggested organization and proofread the whole paper. Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen advised the project, suggested the empirical study and provided computation resources.

## References

[1] Jérôme Seymour Bruner. The course of cognitive growth. American Psychologist, 19:1-15, 1964. URL https://api.semanticscholar.org/CorpusID:145196722.

[2] Jérôme Seymour Bruner, 1960. URL https://api.semanticscholar.org/CorpusID: 177285798 .

[3] N Jayashri and K Kalaiselvi. Knowledge acquisition-scholarly foundations with knowledge management. International Journal of Advanced Studies of Scientific Research, 3(12), 2018.

[4] Randall Davis, Howard E. Shrobe, and Peter Szolovits. What is a knowledge representation? AI Mag., 14(1):17-33, 1993. doi: 10.1609/AIMAG.V14I1.1029. URL https://doi.org/ 10.1609/aimag.v14i1.1029.

[5] Yejin Choi. Knowledge is power: Symbolic knowledge distillation, commonsense morality, \& multimodal script knowledge. In K. Selcuk Candan, Huan Liu, Leman Akoglu, Xin Luna

Dong, and Jiliang Tang, editors, WSDM '22: The Fifteenth ACM International Conference on Web Search and Data Mining, Virtual Event / Tempe, AZ, USA, February 21 - 25, 2022, page 3. ACM, 2022. doi: 10.1145/3488560.3500242. URL https://doi.org/10.1145/ 3488560.3500242 .

[6] Hongming Zhang, Xin Liu, Haojie Pan, Haowen Ke, Jiefu Ou, Tianqing Fang, and Yangqiu Song. ASER: towards large-scale commonsense knowledge acquisition via higher-order selectional preference over eventualities. Artif. Intell., 309:103740, 2022. doi: 10.1016/J. ARTINT.2022.103740. URL https://doi.org/10.1016/j.artint.2022.103740.

[7] Christopher D Manning. Human language understanding \& reasoning. Daedalus, 151(2): $127-138,2022$.

[8] Karen L. McGraw and Karan Harbison-Briggs. Knowledge acquisition - principles and guidelines. Prentice Hall, 1990. ISBN 978-0-13-517095-3.

[9] Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. Linguistic knowledge and transferability of contextual representations. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 1073-1094. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1112. URL https://doi.org/10.18653/v1/n19-1112.

[10] Xu Han, Zhengyan Zhang, and Zhiyuan Liu. Knowledgeable machine learning for natural language processing. Commun. ACM, 64(11):50-51, 2021. doi: 10.1145/3481608. URL https://doi.org/10.1145/3481608.

[11] Mohammad Hossein Jarrahi, David Askay, Ali Eshraghi, and Preston Smith. Artificial intelligence and knowledge management: A partnership between human and ai. Business Horizons, 66(1):87-99, 2023.

[12] Huajun Chen. Large knowledge model: Perspectives and challenges. CoRR, abs/2312.02706, 2023. doi: 10.48550/ARXIV.2312.02706. URL https://doi.org/10.48550/arXiv. 2312 . 02706.

[13] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303. 08774. URL https://doi.org/10.48550/arXiv.2303.08774.

[14] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, JianYun Nie, and Ji-Rong Wen. A survey of large language models. CoRR, abs/2303.18223, 2023. doi: 10.48550/ARXIV.2303.18223. URL https://doi.org/10.48550/arXiv.2303.18223.

[15] Jan Sawicki, Maria Ganzha, and Marcin Paprzycki. The state of the art of natural language processing-a systematic automated review of nlp literature using nlp techniques. Data Intelligence, pages 1-47, 2023.

[16] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. doi: 10.48550/ARXIV.2302. 13971. URL https://doi.org/10.48550/arXiv.2302.13971.

[17] Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. Challenges and applications of large language models. CoRR, abs/2307.10169, 2023. doi: 10.48550/ARXIV.2307.10169. URL https://doi.org/10. 48550/arXiv.2307.10169.

[18] Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, Muhammad Bilal Shaikh, Naveed Akhtar, Jia Wu, Seyedali Mirjalili, et al. Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects. Authorea Preprints, 2023.

[19] Chaoning Zhang, Chenshuang Zhang, Sheng Zheng, Yu Qiao, Chenghao Li, Mengchun Zhang, Sumit Kumar Dam, Chu Myaet Thwal, Ye Lin Tun, Le Luang Huy, Dong Uk Kim, Sung-Ho Bae, Lik-Hang Lee, Yang Yang, Heng Tao Shen, In So Kweon, and Choong Seon Hong. A complete survey on generative AI (AIGC): is chatgpt from GPT-4 to GPT-5 all you need? CoRR, abs/2303.11717, 2023. doi: 10.48550/ARXIV.2303.11717. URL https://doi.org/10.48550/arXiv.2303.11717.

[20] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. CoRR, abs/2304.13712, 2023. doi: 10.48550/ARXIV.2304.13712. URL https: //doi.org/10.48550/arXiv.2304.13712.

[21] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. CoRR, abs/2306.13549, 2023. doi: 10.48550/ ARXIV.2306.13549. URL https://doi.org/10.48550/arXiv.2306.13549.

[22] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. CoRR, abs/2308.11432, 2023. doi: 10.48550/ARXIV.2308.11432. URL https://doi.org/10.48550/arXiv.2308.11432.

[23] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huan, and Tao Gui. The rise and potential of large language model based agents: A survey. CoRR, abs/2309.07864, 2023. doi: 10.48550/ARXIV.2309.07864. URL https://doi.org/10.48550/arXiv.2309.07864.

[24] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models. CoRR, abs/2304.08354, 2023. doi: 10.48550/ARXIV.2304.08354. URL https://doi.org/10.48550/arXiv.2304.08354.

[25] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving AI tasks with chatgpt and its friends in huggingface. CoRR, abs/2303.17580, 2023. doi: 10.48550/ARXIV.2303.17580. URL https://doi.org/10. 48550/arXiv. 2303.17580.

[26] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, Rongxiang Weng, Wensen Cheng, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. Secrets of RLHF in large language models part I: PPO. CoRR, abs/2307.04964, 2023. doi: 10.48550/ARXIV.2307.04964. URL https://doi.org/10.48550/arXiv .2307.04964.

[27] Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, and Yue Zhang. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity, 2023.

[28] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers and future. CoRR, abs/2309.15402, 2023. doi: 10.48550/ARXIV.2309.15402. URL https://doi.org/10.48550/arXiv.2309.15402.

[29] Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. Is chatgpt a general-purpose natural language processing task solver? In Houda

Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 1339-1384. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main.85.

[30] Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, and Hai Zhao. Igniting language intelligence: The hitchhiker's guide from chain-of-thought reasoning to language agents. CoRR, abs/2311.11797, 2023. doi: 10.48550/ARXIV.2311.11797. URL https://doi.org/10. 48550/arXiv.2311.11797.

[31] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022, 2022. URL https://openreview.net/ forum?id=yzkSU5zdwD.

[32] Sanae Lotfi, Marc Finzi, Yilun Kuang, Tim Rudner, Micah Goldblum, and Andrew Wilson. Non-vacuous generalization bounds for large language models. In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning, 2023.

[33] Ziming Liu, Ziqian Zhong, and Max Tegmark. Grokking as simplification: A nonlinear complexity perspective. In UniReps: the First Workshop on Unifying Representations in Neural Models, 2023.

[34] Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. Language modeling is compression. CoRR, abs/2309.10668, 2023. doi: 10.48550/ARXIV.2309.10668. URL https://doi.org/10. 48550/arXiv.2309.10668.

[35] Zige Wang, Wanjun Zhong, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Lifeng Shang, Xin Jiang, and Qun Liu. Data management for large language models: A survey. CoRR, abs/2312.01700, 2023. doi: 10.48550/ARXIV.2312.01700. URL https://doi.org/10. 48550/arXiv. 2312.01700 .

[36] Wes Gurnee and Max Tegmark. Language models represent space and time, 2023.

[37] Zhangyin Feng, Weitao Ma, Weijiang Yu, Lei Huang, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. Trends in integration of knowledge and large language models: A survey and taxonomy of methods, benchmarks, and applications. CoRR, abs/2311.05876, 2023. doi: 10.48550/ARXIV.2311.05876. URL https://doi.org/10.48550/arXiv.2311.05876.

[38] Lionel Wong, Gabriel Grand, Alexander K. Lew, Noah D. Goodman, Vikash K. Mansinghka, Jacob Andreas, and Joshua B. Tenenbaum. From word models to world models: Translating from natural language to the probabilistic language of thought. CoRR, abs/2306.12672, 2023. doi: 10.48550/ARXIV.2306.12672. URL https://doi.org/10.48550/arXiv.2306.12672.

[39] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493-8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581. URL https://aclanthology.org/2022.acl-long.581.

[40] Jun Zhao, Zhihao Zhang, Yide Ma, Qi Zhang, Tao Gui, Luhui Gao, and Xuanjing Huang. Unveiling A core linguistic region in large language models. CoRR, abs/2310.14928, 2023. doi: 10.48550/ARXIV.2310.14928. URL https://doi.org/10.48550/arXiv.2310.14928.

[41] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484-5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main. 446.

[42] Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30-45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10. 18653/v1/2022.emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main. 3.

[43] Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda B. Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https: //openreview.net/pdf?id=DeG07_TcZvT.

[44] Roma Patel and Ellie Pavlick. Mapping language models to grounded conceptual spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id $=\mathrm{gJcEM8sxHK}$.

[45] Zeyuan Allen Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. CoRR, abs/2309.14316, 2023. doi: 10.48550/ARXIV.2309.14316. URL https://doi.org/10.48550/arXiv.2309.14316.

[46] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.2, knowledge manipulation. CoRR, abs/2309.14402, 2023. doi: 10.48550/ARXIV.2309.14402. URL https://doi.org/10.48550/arXiv.2309.14402.

[47] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 1918719197. IEEE, 2023. doi: 10.1109/CVPR52729.2023.01839. URL https://doi.org/10. 1109/CVPR52729.2023.01839.

[48] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 24632473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1250. URL https://aclanthology.org/D19-1250.

[49] Benjamin Heinzerling and Kentaro Inui. Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries. CoRR, abs/2008.09036, 2020. URL https://arxiv.org/abs/2008.09036.

[50] Cunxiang Wang, Pai Liu, and Yue Zhang. Can generative pre-trained language models serve as knowledge bases for closed-book qa? In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 3241-3251. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021. ACL-LONG.251. URL https://doi.org/10.18653/v1/2021.acl-long. 251.

[51] Zexuan Zhong, Dan Friedman, and Danqi Chen. Factual probing is [MASK]: learning vs. learning to recall. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek HakkaniTür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 5017-5033. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.398. URL https://doi.org/10.18653/v1/2021. naacl-main. 398 .

[52] Boxi Cao, Hongyu Lin, Xianpei Han, Le Sun, Lingyong Yan, Meng Liao, Tong Xue, and Jin Xu. Knowledgeable or educated guess? revisiting language models as knowledge bases. In

Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 1860-1874. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.146. URL https://doi.org/10. 18653/v1/2021.acl-long. 146 .

[53] Ruilin Zhao, Feng Zhao, Guandong Xu, Sixiao Zhang, and Hai Jin. Can language models serve as temporal knowledge bases? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2024-2037. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-EMNLP.147. URL https://doi.org/10.18653/v1/2022.findings-emnlp.147.

[54] Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen. Time-aware language models as temporal knowledge bases. Trans. Assoc. Comput. Linguistics, 10:257-273, 2022. doi: 10.1162/TACL _A\_00459. URL https://doi.org/10.1162/tacl_a_00459.

[55] Badr AlKhamissi, Millicent Li, Asli Celikyilmaz, Mona T. Diab, and Marjan Ghazvininejad. A review on language models as knowledge bases. CoRR, abs/2204.06031, 2022. doi: 10. 48550/ARXIV.2204.06031. URL https://doi.org/10.48550/arXiv. 2204.06031.

[56] Boxi Cao, Hongyu Lin, Xianpei Han, and Le Sun. The life cycle of knowledge in big language models: A survey. CoRR, abs/2303.07616, 2023. doi: 10.48550/ARXIV.2303.07616. URL https://doi.org/10.48550/arXiv.2303.07616.

[57] Paul Youssef, Osman Alperen Koras, Meijie Li, Jörg Schlötterer, and Christin Seifert. Give me the facts! A survey on factual knowledge probing in pre-trained language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 15588-15605. Association for Computational Linguistics, 2023. URL https://aclanthology .org/2023. findings-emnlp. 1043 .

[58] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. CoRR, abs/2306.08302, 2023. doi: 10.48550/ARXIV.2306.08302. URL https://doi.org/10.48550/arXiv.2306.08302.

[59] Jeff Z. Pan, Simon Razniewski, Jan-Christoph Kalo, Sneha Singhania, Jiaoyan Chen, Stefan Dietze, Hajira Jabeen, Janna Omeliyanenko, Wen Zhang, Matteo Lissandrini, Russa Biswas, Gerard de Melo, Angela Bonifati, Edlira Vakaj, Mauro Dragoni, and Damien Graux. Large language models and knowledge graphs: Opportunities and challenges. TGDK, 1(1):2:12:38, 2023. doi: 10.4230/TGDK.1.1.2. URL https://doi.org/10.4230/TGDK.1.1.2.

[60] Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad, and Jun Wang. How do large language models capture the ever-changing world knowledge? A review of recent advances. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 8289-8311. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main.516.

[61] Canyu Chen and Kai Shu. Combating misinformation in the age of llms: Opportunities and challenges. CoRR, abs/2311.05656, 2023. doi: 10.48550/ARXIV.2311.05656. URL https://doi.org/10.48550/arXiv.2311.05656.

[62] Xunjian Yin, Baizhou Huang, and Xiaojun Wan. ALCUNA: large language models meet new knowledge. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 1397-1414. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main.87.

[63] Isabelle Augenstein, Timothy Baldwin, Meeyoung Cha, Tanmoy Chakraborty, Giovanni Luca Ciampaglia, David P. A. Corney, Renee DiResta, Emilio Ferrara, Scott Hale, Alon Y. Halevy, Eduard H. Hovy, Heng Ji, Filippo Menczer, Rubén Míguez, Preslav Nakov, Dietram Scheufele, Shivam Sharma, and Giovanni Zagni. Factuality challenges in the era of large language models. CoRR, abs/2310.05189, 2023. doi: 10.48550/ARXIV.2310.05189. URL https://doi.org/10.48550/arXiv.2310.05189.

[64] Hongling Zheng, Li Shen, Anke Tang, Yong Luo, Han Hu, Bo Du, and Dacheng Tao. Learn from model beyond fine-tuning: A survey. CoRR, abs/2310.08184, 2023. doi: 10.48550/ ARXIV.2310.08184. URL https://doi.org/10.48550/arXiv.2310.08184.

[65] Xiangyang Liu, Tianxiang Sun, Junliang He, Jiawen Wu, Lingling Wu, Xinyu Zhang, Hao Jiang, Zhao Cao, Xuanjing Huang, and Xipeng Qiu. Towards efficient NLP: A standard evaluation and A strong baseline. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 3288-3303. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.240. URL https://doi.org/10.18653/v1/2022.naacl-main. 240 .

[66] Jingjing Xu, Wangchunshu Zhou, Zhiyi Fu, Hao Zhou, and Lei Li. A survey on green deep learning. CoRR, abs/2111.05193, 2021. URL https://arxiv.org/abs/2111.05193.

[67] Gaurav Menghani. Efficient deep learning: A survey on making deep learning models smaller, faster, and better. ACM Comput. Surv., 55(12):259:1-259:37, 2023. doi: 10.1145/3578938. URL https://doi.org/10.1145/3578938.

[68] Kai Lv, Shuo Zhang, Tianle Gu, Shuhao Xing, Jiawei Hong, Keyu Chen, Xiaoran Liu, Yuqing Yang, Honglin Guo, Tengxiao Liu, Yu Sun, Qipeng Guo, Hang Yan, and Xipeng Qiu. Collie: Collaborative training of large language models in an efficient way. In Yansong Feng and Els Lefever, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, pages 527-542. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-demo.48.

[69] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and opportunities. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10222-10240, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.emnlp-main.632. URL https://aclanthology.org/2023.emnlp-main. 632.

[70] Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, and Jundong L. Knowledge editing for large language models: A survey, 2023.

[71] Vittorio Mazzia, Alessandro Pedrani, Andrea Caciolai, Kay Rottmann, and Davide Bernardi. A survey on knowledge editing of neural networks. CoRR, abs/2310.19704, 2023. doi: 10.48550/ARXIV.2310.19704. URL https://doi.org/10.48550/arXiv .2310.19704.

[72] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A topdown approach to AI transparency. CoRR, abs/2310.01405, 2023. doi: 10.48550/ARXIV. 2310.01405. URL https://doi.org/10.48550/arXiv.2310.01405.

[73] Yingji Li, Mengnan Du, Rui Song, Xin Wang, and Ying Wang. A survey on fairness in large language models. CoRR, abs/2308.10149, 2023. doi: 10.48550/ARXIV.2308.10149. URL https://doi.org/10.48550/arXiv.2308.10149.

[74] El-Mahdi El-Mhamdi, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Lê-Nguyên Hoang, Rafael Pinot, and John Stephan. On the impossible safety of large AI models. CoRR, abs/2209.15259, 2022. doi: 10.48550/ARXIV.2209.15259. URL https://doi.org/10. 48550/arXiv.2209.15259.

[75] Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. Bias and fairness in large language models: A survey. CoRR, abs/2309.00770, 2023. doi: 10.48550/ARXIV.2309. 00770. URL https://doi.org/10.48550/arXiv.2309.00770.

[76] Xiaowei Huang, Wenjie Ruan, Wei Huang, Gaojie Jin, Yi Dong, Changshun Wu, Saddek Bensalem, Ronghui Mu, Yi Qi, Xingyu Zhao, Kaiwen Cai, Yanghao Zhang, Sihao Wu, Peipei Xu, Dengyu Wu, André Freitas, and Mustafa A. Mustafa. A survey of safety and trustworthiness of large language models through the lens of verification and validation. CoRR, abs/2305.11391, 2023. doi: 10.48550/ARXIV.2305.11391. URL https: //doi.org/10.48550/arXiv.2305.11391.

[77] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[78] Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. What does BERT learn about the structure of language? In Anna Korhonen, David Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 36513657, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/ v1/P19-1356. URL https://aclanthology.org/P19-1356.

[79] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 36, 2022.

[80] Evan Hernandez, Arnab Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, and David Bau. Linearity of relation decoding in transformer language models. ArXiv, abs/2308.09124, 2023. URL https://api.semanticscholar.org/ CorpusID: 261031179 .

[81] Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. Linguistic knowledge and transferability of contextual representations. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1073-1094, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1112. URL https://aclanthology.org/N19-1112.

[82] Xuhui Zhou, Yue Zhang, Leyang Cui, and Dandan Huang. Evaluating commonsense in pretrained language models. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 9733-9740. AAAI Press, 2020. doi: 10.1609/AAAI.V34I05.6523. URL https://doi.org/10.1609/aaai.v34i05.6523.

[83] Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. Symbolic knowledge distillation: from general language models to commonsense models. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 4602-4625. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.341. URL https://doi.org/10.18653/v1/2022.naacl-main. 341 .

[84] Xiang Lorraine Li, Adhiguna Kuncoro, Jordan Hoffmann, Cyprien de Masson d'Autume, Phil Blunsom, and Aida Nematzadeh. A systematic investigation of commonsense knowledge in large language models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 11838-11855. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN. 812. URL https://doi.org/10.18653/v1/2022.emnlp-main. 812 .

[85] Boxi Cao, Qiaoyu Tang, Hongyu Lin, Xianpei Han, Jiawei Chen, Tianshu Wang, and Le Sun. Retentive or forgetful? diving into the knowledge memorizing mechanism of language models. CoRR, abs/2305.09144, 2023. doi: 10.48550/ARXIV.2305.09144. URL https://doi.org/10.48550/arXiv.2305.09144.

[86] Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, and Chelsea Finn. Finetuning language models for factuality. CoRR, abs/2311.08401, 2023. doi: 10.48550/ARXIV. 2311.08401. URL https://doi.org/10.48550/arXiv.2311.08401.

[87] Shahar Katz and Yonatan Belinkov. VISIT: visualizing and interpreting the semantic information flow of transformers. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 14094-14113. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.findings-emnlp.939.

[88] Ari Holtzman, Peter West, and Luke Zettlemoyer. Generative models as a complex systems science: How can we make sense of large language model behavior?, 2023.

[89] Mansi Sakarvadia, Arham Khan, Aswathy Ajith, Daniel Grzenda, Nathaniel Hudson, André Bauer, Kyle Chard, and Ian T. Foster. Attention lens: A tool for mechanistically interpreting the attention head information retrieval mechanism. CoRR, abs/2310.16270, 2023. doi: 10. 48550/ARXIV.2310.16270. URL https://doi.org/10.48550/arXiv.2310.16270.

[90] Boxi Cao, Qiaoyu Tang, Hongyu Lin, Xianpei Han, Jiawei Chen, Tianshu Wang, and Le Sun. Retentive or forgetful? diving into the knowledge memorizing mechanism of language models. arXiv preprint arXiv:2305.09144, 2023.

[91] William Rudman, Catherine Chen, and Carsten Eickhoff. Outlier dimensions encode task specific knowledge. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 14596-14605. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main. 901.

[92] Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610, 2023.

[93] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171-4186. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1423. URL https://doi.org/10.18653/v1/n19-1423.

[94] Daniel D Lundstrom, Tianjian Huang, and Meisam Razaviyayn. A rigorous study of integrated gradients method and extensions to internal neuron attributions. In International Conference on Machine Learning, pages 14485-14508. PMLR, 2022.

[95] Xiaozhi Wang, Kaiyue Wen, Zhengyan Zhang, Lei Hou, Zhiyuan Liu, and Juanzi Li. Finding skill neurons in pre-trained transformer-based language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11132-11152, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.765. URL https://aclanthology.org/2022. emnlp-main. 765 .

[96] Divya Nori, Shivali Singireddy, and Marina Ten Have. Identification of knowledge neurons in protein language models. arXiv preprint arXiv:2312.10770, 2023.

[97] Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Journey to the center of the knowledge neurons: Discoveries of language-independent knowledge neurons and degenerate knowledge neurons, 2023.

[98] Jun Zhao, Zhihao Zhang, Yide Ma, Qi Zhang, Tao Gui, Luhui Gao, and Xuanjing Huang. Unveiling a core linguistic region in large language models. arXiv preprint arXiv:2310.14928, 2023.

[99] Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1350-1370, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.95. URL https://aclanthology. org/2023.findings-emnlp.95.

[100] Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models, 2023. URL https://openai.com/research/ language-models-can-explain-neurons-in-language-models.

[101] Anonymous. What does the knowledge neuron thesis have to do with knowledge? In Submitted to The Twelfth International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=2HJRwwbV3G. under review.

[102] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. CoRR, abs/2304.14767, 2023. doi: 10.48550/arXiv.2304.14767. URL https://doi.org/10.48550/arXiv.2304.14767.

[103] Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[104] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=NpsVSN6o4ul.

[105] Alex Foote, Neel Nanda, Esben Kran, Ioannis Konstas, Shay Cohen, and Fazl Barez. Neuron to graph: Interpreting language model neurons at scale. arXiv preprint arXiv:2305.19911, 2023.

[106] Jie Ren, Mingjie Li, Qirui Chen, Huiqi Deng, and Quanshi Zhang. Defining and quantifying the emergence of sparse concepts in dnns. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 2028020289. IEEE, 2023. doi: 10.1109/CVPR52729.2023.01942. URL https://doi.org/10. 1109/CVPR52729.2023.01942.

[107] Mingjie Li and Quanshi Zhang. Does a neural network really encode symbolic concepts? In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 20452-20469. PMLR, 2023. URL https://proceedings.mlr.press/v202/ li23at.html.

[108] Ning Ding, Yujia Qin, Guang Yang, Fu Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Haitao Zheng, Jianfei Chen, Y. Liu, Jie Tang, Juanzi Li, and Maosong Sun. Parameterefficient fine-tuning of large-scale pre-trained language models. Nature Machine Intelligence, 5:220-235, 2023. URL https://api.semanticscholar.org/CorpusID:257316425.

[109] Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky. Scaling down to scale up: A guide to parameter-efficient fine-tuning. $C o R R$, abs/2303.15647, 2023. doi: 10.48550/ARXIV. 2303.15647. URL https://doi.org/10.48550/arXiv.2303.15647.

[110] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer
learning for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790-2799. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/houlsby19a.html.

[111] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.

[112] Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu. ERNIE: enhanced language representation with informative entities. In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 1441-1451. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1139. URL https://doi.org/10.18653/v1/p19-1139.

[113] Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. Knowprompt: Knowledge-aware prompt-tuning with synergistic optimization for relation extraction. In Frédérique Laforest, Raphaël Troncy, Elena Simperl, Deepak Agarwal, Aristides Gionis, Ivan Herman, and Lionel Médini, editors, WWW '22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022, pages 2778-2788. ACM, 2022. doi: 10.1145/3485447.3511998. URL https://doi.org/10.1145/ 3485447.3511998 .

[114] Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, and Maosong Sun. PTR: prompt tuning with rules for text classification. AI Open, 3:182-192, 2022. doi: 10.1016/J.AIOPEN.2022.11.003. URL https://doi.org/10.1016/j.aiopen.2022.11.003.

[115] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[116] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry W. Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc V. Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation. CoRR, abs/2310.03214, 2023. doi: 10. 48550/ARXIV.2310.03214. URL https://doi.org/10.48550/arXiv. 2310.03214.

[117] Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. Fine-tuning or retrieval? comparing knowledge injection in llms. arXiv preprint arXiv:2312.05934, 2023.

[118] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Acl 2023 tutorial: Retrieval-based language models and applications. ACL 2023, 2023.

[119] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9459-9474. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/6b493230205f780e1bc26945df7481e5-Paper.pdf.

[120] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3929-3938. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/guu20a.html.

[121] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin LeytonBrown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 2023. URL https://arxiv.org/abs/2302. 00083.

[122] Michiel de Jong, Yury Zemlyanskiy, Nicholas FitzGerald, Fei Sha, and William W. Cohen. Mention memory: incorporating textual knowledge into transformers through entity mention attention. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=0Y1A8ejQgEX.

[123] Yunzhi Yao, Shaohan Huang, Li Dong, Furu Wei, Huajun Chen, and Ningyu Zhang. Kformer: Knowledge injection in transformer feed-forward layers. In Wei Lu, Shujian Huang, Yu Hong, and Xiabing Zhou, editors, Natural Language Processing and Chinese Computing, pages 131-143, Cham, 2022. Springer International Publishing. ISBN 978-3-031-17120-8.

[124] Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. Entities as experts: Sparse memory access with entity supervision. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4937-4951, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. emnlp-main.400. URL https://aclanthology.org/2020.emnlp-main. 400.

[125] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= $\mathrm{Hk} 1 \mathrm{BjCEKvH}$.

[126] Zexuan Zhong, Tao Lei, and Danqi Chen. Training language models with memory augmentation. In Empirical Methods in Natural Language Processing (EMNLP), 2022.

[127] Dani Yogatama, Cyprien de Masson d'Autume, and Lingpeng Kong. Adaptive semiparametric language models. Transactions of the Association for Computational Linguistics, 9:362373, 2021. doi: 10.1162/tacl_a_00371. URL https://aclanthology .org/2021.tacl-1.22.

[128] Xiang Chen, Lei Li, Ningyu Zhang, Xiaozhuan Liang, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. Decoupling knowledge from memorization: Retrievalaugmented prompt learning. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/ paper/2022/hash/97011c648eda678424f9292dadeae72e-Abstract-Conference.html.

[129] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. CoRR, abs/2309.01431, 2023. doi: 10.48550/ARXIV.2309. 01431. URL https://doi.org/10.48550/arXiv.2309.01431.

[130] Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong Wen, and Haifeng Wang. Investigating the factual knowledge boundary of large language models with retrieval augmentation. CoRR, abs/2307.11019, 2023. doi: 10.48550/ ARXIV.2307.11019. URL https://doi.org/10.48550/arXiv.2307.11019.

[131] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context, 2023.

[132] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory G. Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Trans. Pattern Anal. Mach. Intell., 44(7):3366-3385, 2022. doi: 10.1109/TPAMI.2021.3057446. URL https://doi.org/10.1109/TPAMI.2021.3057446.

[133] Tongtong Wu, Massimo Caccia, Zhuang Li, Yuan-Fang Li, Guilin Qi, and Gholamreza Haffari. Pretrained language model in continual learning: A comparative study. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 2529, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=figzpGMrdD.

[134] Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Deep class-incremental learning: A survey. CoRR, abs/2302.03648, 2023. doi: 10.48550/ ARXIV.2302.03648. URL https://doi.org/10.48550/arXiv.2302.03648.

[135] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. CoRR, abs/2302.00487, 2023. doi: 10.48550/ ARXIV.2302.00487. URL https://doi.org/10.48550/arXiv.2302.00487.

[136] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. Experience replay for continual learning. In Neural Information Processing Systems, 2018. URL https://api.semanticscholar.org/CorpusID :53860287.

[137] Rahaf Aljundi, Lucas Caccia, Eugene Belilovsky, Massimo Caccia, Min Lin, Laurent Charlin, and Tinne Tuytelaars. Online continual learning with maximally interfered retrieval. ArXiv, abs/1908.04742, 2019. URL https://api.semanticscholar.org/CorpusID:199552250.

[138] James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114: 3521 - 3526, 2016. URL https://api.semanticscholar.org/CorpusID:4704285.

[139] Tom Mitchell, William Cohen, Estevam Hruschka, Partha Talukdar, Bishan Yang, Justin Betteridge, Andrew Carlson, Bhavana Dalvi, Matt Gardner, Bryan Kisiel, et al. Never-ending learning. Communications of the ACM, 61(5):103-115, 2018.

[140] Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 7765-7773, 2018.

[141] Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3366-3375, 2017.

[142] Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, and Quoc Viet Hung Nguyen. A survey of machine unlearning. arXiv preprint arXiv:2209.02299, 2022.

[143] Ga Wu, Masoud Hashemi, and Christopher Srinivasa. Puma: Performance unchanged model augmentation for training data removal. In AAAI Conference on Artificial Intelligence, 2022.

[144] Yuanshun Yao, Xiaojun Xu, and Yang Liu. Large language model unlearning, 2023.

[145] Nianwen Si, Hao Zhang, Heyu Chang, Wenlin Zhang, Dan Qu, and Weiqiang Zhang. Knowledge unlearning for llms: Tasks, methods, and challenges, 2023.

[146] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=awIpKpwTwF.

[147] Jiaao Chen and Diyi Yang. Unlearn what you want to forget: Efficient unlearning for LLMs. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12041-12052, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.738. URL https://aclanthology .org/2023.emnlp-main.738.

[148] Boxi Cao, Hongyu Lin, Xianpei Han, Le Sun, Lingyong Yan, Meng Liao, Tong Xue, and Jin Xu. Knowledgeable or educated guess? revisiting language models as knowledge bases. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1860-1874, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.146. URL https://aclanthology.org/2021. acl-long. 146 .

[149] Tim Schott, Daniel Furman, and Shreshta Bhat. Polyglot or not? measuring multilingual encyclopedic knowledge in foundation models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11238-11253, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.691. URL https://aclanthology.org/ 2023.emnlp-main. 691 .

[150] Shibo Hao, Bowen Tan, Kaiwen Tang, Bin Ni, Xiyan Shao, Hengzhe Zhang, Eric Xing, and Zhiting Hu. BertNet: Harvesting knowledge graphs with arbitrary relations from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5000-5015, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.309. URL https://aclanthology.org/2023. findings-acl. 309 .

[151] Chenguang Wang, Xiao Liu, and Dawn Song. Language models are open knowledge graphs, 2021. URL https://openreview.net/forum?id=aRTRjVPkm-.

[152] Anshita Gupta, Debanjan Mondal, Akshay Krishna Sheshadri, Wenlong Zhao, Xiang Lorraine Li, Sarah Wiegreffe, and Niket Tandon. Editing commonsense knowledge in gpt, 2023.

[153] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn. Memory-based model editing at scale. In International Conference on Machine Learning, 2022.

[154] Aman Madaan, Niket Tandon, Peter Clark, and Yiming Yang. Memory-assisted prompt editing to improve GPT-3 after deployment. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2833-2861, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.183. URL https://aclanthology .org/2022.emnlp-main. 183.

[155] Zexuan Zhong, Zhengxuan Wu, Christopher Manning, Christopher Potts, and Danqi Chen. MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15686-15702, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.971. URL https://aclanthology .org/2023.emnlp-main. 971.

[156] Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. Can we edit factual knowledge by in-context learning? In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4862-4876, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.296. URL https://aclanthology.org/2023.emnlp-main.296.

[157] Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. Evaluating the ripple effects of knowledge editing in language models, 2023.

[158] Hengrui Gu, Kaixiong Zhou, Xiaotian Han, Ninghao Liu, Ruobing Wang, and Xin Wang. Pokemqa: Programmable knowledge editing for multi-hop question answering, 2023.

[159] Shikhar Murty, Christopher Manning, Scott Lundberg, and Marco Tulio Ribeiro. Fixing model bugs with natural language patches. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11600-11613, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/ v1/2022.emnlp-main.797. URL https://aclanthology.org/2022.emnlp-main.797.

[160] Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. Calibrating factual knowledge in pretrained language models. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5937-5947, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. findings-emnlp.438. URL https://aclanthology.org/2022.findings-emnlp.438.

[161] Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, and Zhang Xiong. Transformer-patcher: One mistake worth one neuron. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= 4oYUGeGBPm.

[162] Evan Hernandez, Belinda Z. Li, and Jacob Andreas. Inspecting and editing knowledge representations in language models, 2023.

[163] Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. Aging with grace: Lifelong model editing with discrete key-value adaptors. ArXiv, abs/2211.11031, 2022. URL https://api.semanticscholar.org/CorpusID:253735429.

[164] Suhang Wu, Minlong Peng, Yue Chen, Jinsong Su, and Mingming Sun. Eva-kellm: A new benchmark for evaluating knowledge editing of llms. CoRR, abs/2308.09954, 2023. doi: 10.48550/ARXIV.2308.09954. URL https://doi.org/10.48550/arXiv.2308.09954.

[165] Lang Yu, Qin Chen, Jie Zhou, and Liang He. Melo: Enhancing model editing with neuronindexed dynamic lora, 2023.

[166] Ankit Singh Rawat, Chen Zhu, Daliang Li, Felix Yu, Manzil Zaheer, Sanjiv Kumar, and Srinadh Bhojanapalli. Modifying memories in transformer models. In International Conference on Machine Learning (ICML) 2021, 2020.

[167] Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitry Pyrkin, Sergei Popov, and Artem Babenko. Editable neural networks. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=HJedXaEtvS.

[168] Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6491-6506, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.522. URL https://aclanthology.org/2021.emnlp-main. 522 .

[169] Peter Hase, Mona Diab, Asli Celikyilmaz, Xian Li, Zornitsa Kozareva, Veselin Stoyanov, Mohit Bansal, and Srinivasan Iyer. Methods for measuring, updating, and visualizing factual beliefs in language models. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2714-2731, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main. 199 . URL https://aclanthology.org/2023.eacl-main. 199 .

[170] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=0DcZxeWfOPt.

[171] Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. Massediting memory in a transformer. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=MkbcAHIYgyS.

[172] Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, and Jie Yu. Pmet: Precise model editing in a transformer. In AAAI, 2024.

[173] Chenmien Tan, Ge Zhang, and Jie Fu. Massive editing for large language models via meta learning. arXiv, 2311.04661, 2023. URL https://arxiv.org/pdf/2311.04661.pdf.

[174] Jun-Yu Ma, Jia-Chen Gu, Zhen-Hua Ling, Quan Liu, and Cong Liu. Untying the reversal curse via bidirectional language model editing, 2023.

[175] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. Self-knowledge guided retrieval augmentation for large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10303-10315, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.findings-emnlp.691. URL https://aclanthology.org/2023.findings-emnlp.691.

[176] Yi Liu, Lianzhe Huang, Shicheng Li, Sishuo Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Recall: A benchmark for llms robustness against external counterfactual knowledge, 2023 .

[177] Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Resolving knowledge conflicts in large language models, 2023.

[178] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts, 2023.

[179] Qinan Yu, Jack Merullo, and Ellie Pavlick. Characterizing mechanisms for factual recall in language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9924-9959, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.emnlp-main.615. URL https://aclanthology.org/2023.emnlp-main.615.

[180] Shiwen Ni, Dingwei Chen, Chengming Li, Xiping Hu, Ruifeng Xu, and Min Yang. Forgetting before learning: Utilizing parametric arithmetic for knowledge updating in large language models. arXiv preprint arXiv:2311.08011, 2023.

[181] Xiaoqi Han, Ru Li, Xiaoli Li, and Jeff Z. Pan. A divide and conquer framework for knowledge editing. Knowledge-Based Systems, 279:110826, 2023. ISSN 0950-7051. doi: https://doi.org/ 10.1016/j.knosys.2023.110826. URL https://www.sciencedirect.com/science/article/ $\mathrm{pii} / \mathrm{S} 0950705123005762$.

[182] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on "a is b" fail to learn "b is a", 2023.

[183] Yuval Pinter and Michael Elhadad. Emptying the ocean with a spoon: Should we edit models? In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15164-15172, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.1012. URL https://aclanthology.org/2023.findings-emnlp. 1012.

[184] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pages 333-342, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/K17-1034. URL https://aclanthology.org/K17-1034.

[185] Potsawee Manakul, Adian Liusie, and Mark JF Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896, 2023.

[186] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802-9822, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology.org/2023.acl-long.546.

[187] Yoichi Ishibashi and Hidetoshi Shimodaira. Knowledge sanitization of large language models. arXiv preprint arXiv:2309.11852, 2023.

[188] Zichao Li, Ines Arous, Siva Reddy, and Jackie Cheung. Evaluating dependencies in fact editing for language models: Specificity and implication awareness. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7623-7636, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.511. URL https://aclanthology.org/ 2023.findings-emnlp.511.

[189] Yang Xu, Yutai Hou, Wanxiang Che, and Min Zhang. Language anisotropic cross-lingual model editing. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 5554-5569, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.343. URL https://aclanthology.org/2023.findings-acl.343.

[190] Jiaan Wang, Yunlong Liang, Zengkui Sun, Yuxuan Cao, and Jiarong Xu. Cross-lingual knowledge editing in large language models, 2023.

[191] Weixuan Wang, Barry Haddow, and Alexandra Birch. Retrieval-augmented multilingual knowledge editing, 2023.

[192] Yasumasa Onoe, Michael Zhang, Shankar Padmanabhan, Greg Durrett, and Eunsol Choi. Can LMs learn new entities from descriptions? challenges in propagating injected knowledge. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5469-5485, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.300. URL https: //aclanthology.org/2023.acl-long.300.

[193] Xunjian Yin, Jin Jiang, Liming Yang, and Xiaojun Wan. History matters: Temporal knowledge editing in large language model. arXiv preprint arXiv:2312.05497, 2023.

[194] Yifan Wei, Xiaoyan Yu, Huanhuan Ma, Fangyu Lei, Yixuan Weng, Ran Song, and Kang Liu. Assessing knowledge editing in language models via relation perspective. arXiv preprint arXiv:2311.09053, 2023.

[195] Afra Feyza Akyürek, Eric Pan, Garry Kuwanto, and Derry Wijaya. Dune: Dataset for unified editing. arXiv preprint arXiv:2311.16087, 2023.

[196] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[197] Peng Wang, Ningyu Zhang, Xin Xie, Yunzhi Yao, Bozhong Tian, Mengru Wang, Zekun Xi, Siyuan Cheng, Kangwei Liu, Guozhou Zheng, et al. Easyedit: An easy-to-use knowledge editing framework for large language models. arXiv preprint arXiv:2308.07269, 2023.

[198] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. CoRR, abs/1811.00937, 2018. URL http://arxiv.org/abs/1811.00937.

[199] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. CoRR, abs/1911.11641, 2019. URL http: //arxiv.org/abs/1911.11641.

[200] Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. CoRR, abs/1808.08745, 2018. URL http://arxiv.org/abs/1808.08745.

[201] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. CoRR, abs/1705.03551, 2017. URL http://arxiv.org/abs/1705.03551.

[202] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.

[203] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.

[204] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.

[205] Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, and Huajun Chen. Unveiling the pitfalls of knowledge editing for large language models. CoRR, abs/2310.02129, 2023. doi: 10.48550/ARXIV.2310.02129. URL https://doi.org/10.48550/arXiv.2310.02129.

[206] Xiaoqi Han, Ru Li, Hongye Tan, Wang Yuanlong, Qinghua Chai, and Jeff Pan. Improving sequential model editing with fact retrieval. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11209-11224, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.749. URL https://aclanthology.org/2023. findings-emnlp. 749 .

[207] Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16124-16170, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.893. URL https://aclanthology.org/2023.acl-long.893.

[208] Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models, 2023.

[209] Ting-Yun Chang, Jesse Thomason, and Robin Jia. Do localization methods actually localize memorized data in llms?, 2023.

[210] Yiming Ju and Zheng Zhang. Klob: a benchmark for assessing knowledge locating methods in language models, 2023.

[211] Zeming Chen, Gail Weiss, Eric Mitchell, Asli Celikyilmaz, and Antoine Bosselut. Reckoning: Reasoning through dynamic knowledge encoding, 2023.

[212] Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, and Eunsol Choi. Propagating knowledge updates to lms through distillation, 2023.

[213] Yucheng Shi, Shaochen Xu, Zhengliang Liu, Tianming Liu, Xiang Li, and Ninghao Liu. Mededit: Model editing for medical question answering with external knowledge bases, 2023.

[214] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809-819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/ N18-1074. URL https://aclanthology.org/N18-1074.

[215] Tal Schuster, Adam Fisch, and Regina Barzilay. Get your vitamin C! robust fact verification with contrastive evidence. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 624-643, Online, June 2021. Association for Computational Linguistics. doi: 10. 18653/v1/2021.naacl-main.52. URL https://aclanthology.org/2021.naacl-main.52.

[216] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models, 2023.

[217] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= 6t0Kwf8-jrj.

[218] Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Aleksander Madry. Editing a classifier by rewriting its prediction rules. Advances in Neural Information Processing Systems, 34:23359-23373, 2021.

[219] Davis Brown, Charles Godfrey, Cody Nizinski, Jonathan Tu, and Henry Kvinge. Edit at your own risk: evaluating the robustness of edited models to distribution shifts, 2023.

[220] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= OA9£2jZDGW.

[221] Jian Gu, Chunyang Chen, and Aldeida Aleti. Neuron patching: Neuron-level model editing on code generation and llms, 2023.

[222] Siyuan Cheng, Ningyu Zhang, Bozhong Tian, Zelin Dai, Feiyu Xiong, Wei Guo, and Huajun Chen. Editing language model-based knowledge graph embeddings. AAAI, 2024.

[223] Jiali Cheng, George Dasoulas, Huan He, Chirag Agarwal, and Marinka Zitnik. GNNDelete: A general strategy for unlearning in graph neural networks. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= X9yCkmT5Qrl.

[224] Zirui Liu, Zhimeng Jiang, Shaochen Zhong, Kaixiong Zhou, Li Li, Rui Chen, Soo-Hyun Choi, and Xia Hu. Editable graph neural network for node classifications, 2023.

[225] Ming Zhong, Chenxin An, Weizhu Chen, Jiawei Han, and Pengcheng He. Seeking neural nuggets: Knowledge transfer in large language models from a parametric perspective. arXiv preprint arXiv:2310.11451, 2023.

[226] Deniz Bayazit, Negar Foroutan, Zeming Chen, Gail Weiss, and Antoine Bosselut. Discovering knowledge-critical subnetworks in pretrained language models. arXiv preprint arXiv:2310.03084, 2023 .

[227] Weishi Li, Yong Peng, Miao Zhang, Liang Ding, Han Hu, and Li Shen. Deep model fusion: A survey. CoRR, abs/2309.15698, 2023. doi: 10.48550/ARXIV.2309.15698. URL https: //doi.org/10.48550/arXiv.2309.15698.

[228] Yi-Lin Sung, Linjie Li, Kevin Lin, Zhe Gan, Mohit Bansal, and Lijuan Wang. An empirical study of multimodal model merging. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 1563-1575. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.findings-emnlp.105.

[229] Jinghan Zhang, Shiqi Chen, Junteng Liu, and Junxian He. Composing parameter-efficient modules with arithmetic operations. CoRR, abs/2306.14870, 2023. doi: 10.48550/ARXIV. 2306.14870. URL https://doi.org/10.48550/arXiv.2306.14870.

[230] Yihan Cao, Siyu Li, Yixin Liu, Zhiling Yan, Yutong Dai, Philip S. Yu, and Lichao Sun. A comprehensive survey of ai-generated content (AIGC): A history of generative AI from GAN to chatgpt. CoRR, abs/2303.04226, 2023. doi: 10.48550/ARXIV.2303.04226. URL https://doi.org/10.48550/arXiv.2303.04226.

[231] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. CoRR, abs/2309.10020, 2023. doi: 10.48550/ARXIV.2309.10020. URL https: //doi.org/10.48550/arXiv.2309.10020.

[232] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and JunYan Zhu. Ablating concepts in text-to-image diffusion models. CoRR, abs/2303.13516, 2023. doi: 10.48550/ARXIV.2303.13516. URL https://doi.org/10.48550/arXiv.2303.13516.

[233] Samyadeep Basu, Nanxuan Zhao, Vlad Morariu, Soheil Feizi, and Varun Manjunatha. Localizing and editing knowledge in text-to-image generative models, 2023.

[234] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. CoRR, abs/2302.05543, 2023. doi: 10.48550/ARXIV.2302.05543. URL https: //doi.org/10.48550/arXiv.2302.05543.

[235] Siyuan Cheng, Bozhong Tian, Qingbin Liu, Xi Chen, Yongheng Wang, Huajun Chen, and Ningyu Zhang. Can we edit multimodal large language models? CoRR, abs/2310.08475, 2023. doi: 10.48550/ARXIV.2310.08475. URL https://doi.org/10.48550/arXiv. 2310. 08475.

[236] Dana Arad, Hadas Orgad, and Yonatan Belinkov. Refact: Updating text-to-image models by editing the text encoder, 2023.

[237] Haowen Pan, Yixin Cao, Xiaozhi Wang, and Xun Yang. Finding and editing multi-modal neurons in pre-trained transformer, 2023.

[238] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models, 2023.

[239] Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O'Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. AI alignment: A comprehensive survey. CoRR, abs/2310.19852, 2023. doi: 10.48550/ARXIV.2310.19852. URL https://doi.org/10.48550/arXiv.2310.19852.

[240] Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu. Aligning large language models with human: A survey. CoRR, abs/2307.12966, 2023. doi: 10.48550/ARXIV.2307.12966. URL https://doi.org/ 10.48550/arXiv. 2307.12966 .

[241] Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, and Deyi Xiong. Large language model alignment: A survey. CoRR, abs/2309.15025, 2023. doi: 10.48550/ARXIV.2309.15025. URL https://doi.org/10. 48550/arXiv.2309.15025.

[242] Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, and Min Zhang. A survey of large language models attribution. CoRR, abs/2311.03731, 2023. doi: 10.48550/ARXIV.2311.03731. URL https://doi.org/10.48550/arXiv.2311.03731.

[243] Amruta Kale, Tin Nguyen, Frederick C. Harris Jr., Chenhao Li, Jiyin Zhang, and Xiaogang Ma. Provenance documentation to enable explainable and trustworthy AI: A literature review. Data Intell., 5(1):139-162, 2023. doi: 10.1162/DINT \A\_00119. URL https://doi.org/ 10.1162/dint_a_00119.

[244] Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. Trustworthy llms: a survey and guideline for evaluating large language models' alignment. CoRR, abs/2308.05374, 2023. doi: 10.48550/ARXIV.2308.05374. URL https://doi.org/10.48550/arXiv . 2308.05374.

[245] Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, and Minlie Huang. Unveiling the implicit toxicity in large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 1322-1338. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023. emnlp-main. 84 .

[246] Kellin Pelrine, Mohammad Taufeeque, Michal Zajkac, Euan McLean, and Adam Gleave. Exploiting novel gpt-4 apis. arXiv preprint arXiv:2312.14302, 2023.

[247] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356-3369, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. findings-emnlp.301. URL https://aclanthology.org/2020.findings-emnlp.301.

[248] Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. Safetybench: Evaluating the safety of large language models with multiple choice questions. CoRR, abs/2309.07045, 2023. doi: 10.48550/ARXIV.2309.07045. URL https://doi.org/10.48550/arXiv.2309.07045.

[249] Jiawen Deng, Hao Sun, Zhexin Zhang, Jiale Cheng, and Minlie Huang. Recent advances towards safe, responsible, and moral dialogue systems: A survey. CoRR, abs/2302.09270, 2023. doi: 10.48550/ARXIV.2302.09270. URL https://doi.org/10.48550/arXiv. 2302. 09270.

[250] Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen. Catastrophic jailbreak of open-source llms via exploiting generation. CoRR, abs/2310.06987, 2023. doi: 10.48550/ARXIV.2310.06987. URL https://doi.org/10.48550/arXiv. 2310.06987.

[251] Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq R. Joty, Richard Socher, and Nazneen Fatema Rajani. Gedi: Generative discriminator guided sequence generation. In Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021, pages 49294952. Association for Computational Linguistics, 2021. URL https://doi.org/10.18653/ v1/2021.findings-emnlp. 424 .

[252] Anonymous. Badedit: Backdooring large language models by model editing. In Submitted to The Twelfth International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=duZANm2ABX. under review.

[253] Maximilian Li, Xander Davies, and Max Nadeau. Circuit breaking: Removing model behaviors with targeted ablation. Workshop on Challenges in Deployable Generative AI at International Conference on Machine Learning, 2023.

[254] Xinshuo Hu, Dongfang Li, Zihao Zheng, Zhenyu Liu, Baotian Hu, and Min Zhang. Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation. CoRR, abs/2308.08090, 2023. doi: 10.48550/arXiv.2308.08090. URL https://doi.org/10.48550/arXiv.2308.08090.

[255] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

[256] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13003-13051. Association for Computational Linguistics, 2023.

[257] Moin Nadeem, Anna Bethke, and Siva Reddy. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5356-5371, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.416. URL https://aclanthology.org/2021.acl-long. 416.

[258] Nirmalendu Prakash and Roy Ka-Wei Lee. Layered bias: Interpreting bias in pretrained large language models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi, editors, Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 284-295, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.blackboxnlp-1.22. URL https://aclanthology.org/2023.blackboxnlp-1.22.

[259] Maria De-Arteaga, Alexey Romanov, Hanna M. Wallach, Jennifer T. Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Cem Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 2019. URL https://api.semanticscholar.org/CorpusID:58006082.

[260] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 1520, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2003. URL https://aclanthology .org/N18-2003.

[261] Charles Yu, Sullam Jeoung, Anish Kasi, Pengfei Yu, and Heng Ji. Unlearning bias in language models by partitioning gradients. In Findings of the Association for Computational Linguistics: ACL 2023, pages 6032-6048, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.375. URL https: //aclanthology.org/2023.findings-acl.375.

[262] Tomasz Limisiewicz, David Mareček, and Tomáš Musil. Debiasing algorithm through model adaptation, 2023.

[263] Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, and Yangqiu Song. Privacy in large language models: Attacks, defenses and future directions. CoRR, abs/2310.10383, 2023. doi: 10.48550/ARXIV.2310.10383. URL https://doi.org/ 10.48550/arXiv. 2310.10383.

[264] Seth Neel and Peter Chang. Privacy issues in large language models: A survey. arXiv preprint arXiv:2312.06717, 2023.

[265] Sanjam Garg, Shafi Goldwasser, and Prashant Nalini Vasudevan. Formalizing data deletion in the context of the right to be forgotten. In Anne Canteaut and Yuval Ishai, editors, Advances in Cryptology - EUROCRYPT 2020, pages 373-402, Cham, 2020. Springer International Publishing. ISBN 978-3-030-45724-2.

[266] Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In Annual Meeting of the Association for Computational Linguistics, 2022. URL https://api.semanticscholar.org/CorpusID:252693065.

[267] Xinwei Wu, Junzhuo Li, Minghui Xu, Weilong Dong, Shuangzhi Wu, Chao Bian, and Deyi Xiong. Depn: Detecting and editing privacy neurons in pretrained language models, 2023.

[268] Yang Chen, Ethan Mendes, Sauvik Das, Wei Xu, and Alan Ritter. Can language models be instructed to protect personal information?, 2023.

[269] Leijie Wu, Song Guo, Junxiao Wang, Zicong Hong, J. Zhang, and Jingren Zhou. On knowledge editing in federated learning: Perspectives, challenges, and future directions. ArXiv, abs/2306.01431, 2023. URL https://api.semanticscholar.org/CorpusID:259064255.

[270] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 17061-17084. PMLR, 2023. URL https://proceedings.mlr.press/v202/kirchenbauer23a.html.

[271] John M Carroll. Human-computer interaction: psychology as a science of design. Annual review of psychology, 48(1):61-83, 1997.

[272] Ranjay Krishna, Donsuk Lee, Li Fei-Fei, and Michael S Bernstein. Socially situated artificial intelligence enables learning from human interaction. Proceedings of the National Academy of Sciences, 119(39):e2115730119, 2022.

[273] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. Lamp: When large language models meet personalization. CoRR, abs/2304.11406, 2023. doi: 10.48550/ ARXIV.2304.11406. URL https://doi.org/10.48550/arXiv.2304.11406.

[274] Morgan L Maeder and Charles A Gersbach. Genome-editing technologies for gene and cell therapy. Molecular Therapy, 24(3):430-446, 2016.

[275] Jin-Soo Kim. Genome editing comes of age. Nature protocols, 11(9):1573-1578, 2016.

[276] Jennifer A Doudna. The promise and challenge of therapeutic genome editing. Nature, 578 (7794):229-236, 2020.

[277] Keyu Pan and Yawen Zeng. Do llms possess a personality? making the MBTI test an amazing evaluation for large language models. CoRR, abs/2307.16180, 2023. doi: 10.48550/arXiv. 2307.16180. URL https://doi.org/10.48550/arXiv. 2307.16180.

[278] Mustafa Safdari, Greg Serapio-García, Clément Crepy, Stephen Fitz, Peter Romero, Luning Sun, Marwa Abdulhai, Aleksandra Faust, and Maja Mataric. Personality traits in large language models. CoRR, abs/2307.00184, 2023. doi: 10.48550/arXiv.2307.00184. URL https://doi.org/10.48550/arXiv.2307.00184.

[279] Quan Tu, Chuanqi Chen, Jinpeng Li, Yanran Li, Shuo Shang, Dongyan Zhao, Ran Wang, and Rui Yan. Characterchat: Learning towards conversational AI with personalized social support. CoRR, abs/2308.10278, 2023. doi: 10.48550/arXiv.2308.10278. URL https:// doi.org/10.48550/arXiv.2308.10278.

[280] Shengyu Mao, Ningyu Zhang, Xiaohan Wang, Mengru Wang, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Editing personality for llms. CoRR, abs/2310.02168, 2023. doi: 10.48550/ARXIV.2310.02168. URL https://doi.org/10. 48550/arXiv. 2310.02168 .

[281] Jen tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael R. Lyu. Emotionally numb or empathetic? evaluating how llms feel using emotionbench, 2023.

[282] EunJeong Hwang, Bodhisattwa Prasad Majumder, and Niket Tandon. Aligning language models to user opinions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 610, 2023, pages 5906-5919. Association for Computational Linguistics, 2023. URL https: //aclanthology.org/2023.findings-emnlp. 393.


[^0]:    * Equal Contribution.

    ${ }^{\dagger}$ Corresponding Author.

    ${ }^{1}$ https://github.com/zjunlp/EasyEdit.

[^1]:    ${ }^{2}$ https://github.com/zjunlp/KnowledgeEditingPapers.

[^2]:    ${ }^{3}$ https://huggingface.co/datasets/zjunlp/KnowEdit.

</end of paper 0>


<paper 1>
# Perturbation-Restrained Sequential Model Editing 

Jun-Yu Ma ${ }^{1,2}$, Hong Wang ${ }^{2}$, Hao-Xiang Xu ${ }^{1,2}$, Zhen-Hua Ling ${ }^{1,2}$, Jia-Chen Gu ${ }^{3 *}$<br>${ }^{1}$ National Engineering Research Center of Speech and Language Information Processing<br>${ }^{2}$ University of Science and Technology of China ${ }^{3}$ University of California, Los Angeles<br>\{mjy1999, wanghong1700,nh2001620\}@mail.ustc.edu.cn,<br>zhling@ustc.edu.cn, gujc@ucla.edu


#### Abstract

Model editing is an emerging field that focuses on updating the knowledge embedded within large language models (LLMs) without extensive retraining. However, current model editing methods significantly compromise the general abilities of LLMs as the number of edits increases, and this trade-off poses a substantial challenge to the continual learning of LLMs. In this paper, we first theoretically analyze that the factor affecting the general abilities in sequential model editing lies in the condition number of the edited matrix. The condition number of a matrix represents its numerical sensitivity, and therefore can be used to indicate the extent to which the original knowledge associations stored in LLMs are perturbed after editing. Subsequently, statistical findings demonstrate that the value of this factor becomes larger as the number of edits increases, thereby exacerbating the deterioration of general abilities. To this end, a framework termed Perturbation Restraint on Upper bouNd for Editing (PRUNE) is proposed, which applies the condition number restraints in sequential editing. These restraints can lower the upper bound on perturbation to edited models, thus preserving the general abilities. Systematically, we conduct experiments employing three popular editing methods on three LLMs across four representative downstream tasks. Evaluation results show that PRUNE can preserve considerable general abilities while maintaining the editing performance effectively in sequential model editing. The code and data are available at https://github.com/mjy1111/PRUNE.


## 1 Introduction

Despite the remarkable capabilities of large language models (LLMs), they encounter challenges such as false or outdated knowledge, and the risk of producing toxic content [65, 46, 29, 28]. Given the prohibitively high cost of retraining LLMs to address these issues, there has been a surge in focus on model editing [12, 41, 43, 44, 42, 39, 64, 27, 40], which aims at updating the knowledge of LLMs cost-effectively. Existing model editing methods can be roughly classified into either parameter-modifying methods $[43,41,42]$ that directly modify a small subset of model parameters, or parameter-preserving methods $[44,63]$ that integrate additional modules without altering the model parameters. In this paper, we study the parameter-modifying editing methods.

Sequential model editing involves making successive edits to the same model over time to continuously update knowledge, as illustrated in Figure 1(a). Recent studies [21, 22, 37] indicate that parameter-modifying editing methods significantly compromise the general abilities of LLMs as the number of edits increases, such as summarization, question answering, and natural language inference. However, these studies neither provide a theoretical analysis of the bottleneck of the general abilities of the edited models, nor propose a solution to preserve these abilities in sequential editing. These affect the scalability of model editing and pose a substantial challenge to the continual learning of LLMs.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_2d7e827830664cc35591g-02.jpg?height=572&width=1393&top_left_y=234&top_left_x=363)

Figure 1: (a) Illustration of sequential model editing. (b) The condition number of edited matrix rapidly increases as the number of edits increases. (c) Comparison of general downstream task performance before editing, after regular editing, and after restrained editing by PRUNE. (d) Comparison of editing performance after regular editing and after restrained editing by PRUNE. $f_{W}$, $f_{W_{n}}$ and $f_{\bar{W}_{n}}$ denote the models that are unedited, regularly edited $n$ times, and restrainedly edited by PRUNE respectively. $W$ is denoted as a matrix to be edited.

In light of the above issues, we first theoretically analyze through matrix perturbation theory [38, 54, 60] to elucidate a crucial factor affecting the general abilities during sequential editing: the condition number $[50,13,52]$ of the edited matrix. The condition number of a matrix represents its numerical sensitivity and therefore can be used to indicate the extent to which the original knowledge associations stored in LLMs are perturbed after editing. As shown in Figure 1(b), our statistical findings demonstrate that the condition number of the edited matrix substantially increases as the number of edits increases, thereby exacerbating the perturbation of original knowledge and the deterioration of general abilities. Therefore, we assume that the bottleneck of the general abilities during sequential editing lies in the escalating value of the condition number.

Towards continual and scalable model editing, we propose Perturbation Restraint on Upper bouNd for Editing (PRUNE) based on the above analysis, which applies the condition number restraints in sequential editing to preserve general abilities and maintain new editing knowledge simultaneously. Specifically, the condition number of the edited matrix is restrained by reducing the large singular values $[1,57]$ of the edit update matrix. Consequently, the upper bound on perturbation to the edited matrix is lowered, thus reducing the perturbation to the original knowledge associations and preserving the general abilities of the edited model, as shown in Figure 1(c). Additionally, we observe that these larger singular values often encapsulate redundant editing overfitting information, so regularizing them will not affect the newly editing knowledge, as shown in Figure 1(d). In this way, the new editing knowledge is embedded into LLMs without affecting their original general abilities. Overall, the proposed editing framework requires only minimal computing resources, and is adaptable to be coupled with multiple existing editing methods.

To validate the effectiveness of the proposed PRUNE, our study comprehensively evaluates the edited LLMs for both general abilities and editing performance in sequential editing scenarios. Extensive empirical research involves three popular editing methods, including MEND [43], ROME [41], and MEMIT [42], which are analyzed based on three representative LLMs including GPT-2 XL (1.5B) [48], LLaMA-2 (7B) [53], and LLaMA-3 (8B). Four representative downstream tasks including reasoning [8], summarization [19], open-domain QA [31], and natural language inference [11] are employed to extensively demonstrate the impact of model editing on the general abilities of LLMs. Experimental results demonstrate that the proposed PRUNE can preserve considerable general abilities and maintain almost all editing performance in sequential editing.

In essence, our research offers three significant contributions: (1) This study theoretically analyzes that the escalating value of the condition number of the edited matrix is the bottleneck of sequential model editing. (2) The PRUNE framework based on the analysis is proposed to preserve the general abilities of the edited model while retaining the editing knowledge. (3) Experimental results including both editing performance and four downstream task performance across three editing methods on three LLMs demonstrate the effectiveness of the proposed method.

## 2 Related Work

Model Editing Methods From the perspective of whether the model parameters are modified, existing editing methods can be divided into parameter-modifying [43, 41, 42, 12] and parameterpreserving methods $[44,25,63]$. This paper focuses on the former. Previous works have investigated the role of MLP layers in Transformer, showing that MLP layers store knowledge, which can be located in specific neurons and edited [16, 10, 17]. KE [3] and MEND [43] train a hypernetwork to get gradient changes to update model parameters [43]. Besides, Meng et al. [41] and Meng et al. [42] used Locate-Then-Edit strategy, which first located multi-layer perceptron (MLP) storing factual knowledge, and then edited such knowledge by injecting new key-value pair in the MLP module. Parameter-preserving methods do not modify model weights but store the editing facts with an external memory. For example, Mitchell et al. [44] stored edits in a base model and learned to reason over them to adjust its predictions as needed.

Model Editing Evaluation Some works investigate the paradigm for model editing evaluation [67, $9,39,35,26,61,15,40]$. Cohen et al. [9] introduced the ripple effects of model editing, suggesting that editing a particular fact implies that many other facts need to be updated. Ma et al. [39] constructed a new benchmark to assess the edited model bidirectionally. Besides, Li et al. [35] explored two significant areas of concern: Knowledge Conflict and Knowledge Distortion. These early studies mainly evaluate edited models per edit rather than sequentially, and they focus narrowly on basic factual triples. Recently, some works assess the impact of editing methods on the general abilities of LLMs in sequential editing scenarios. These studies [21, 22, 37] have conducted comprehensive experiments, showing the parameter-modifying methods significantly degrade the model performance on general downstream tasks.

Matrix Perturbation Theory It plays a crucial role in the field of artificial intelligence (AI) by providing a systematic framework to understand the impact of small changes or perturbations in various AI algorithms and models. Some studies [24, 47, 49] delve into the interpretability of LLMs, revealing how minor alterations in input features or model parameters influence the model's predictions. This understanding helps uncover significant feature connections within the model architecture. Moreover, it has been instrumental in assessing and enhancing the robustness of models [5, 20, 6]. Furthermore, Bird et al. [2] and Dettmers et al. [14] have employed it for sensitivity analysis to identify critical factors affecting algorithm performance. It also contributes to the development of efficient optimization techniques $[34,7,30]$, improving convergence rates and stability of optimization algorithms.

Compared with previous works $[41,42,62,21,22,37]$ that are the most relevant, a main difference should be highlighted. They neither theoretically investigate the reasons for general ability degradation, nor propose methods to maintain these abilities during sequential editing. In contrast, our study makes the first attempt to theoretically explore the bottleneck of general abilities in sequential editing and proposes the PRUNE framework to preserve these abilities for continual model editing.

## 3 Analysis on Bottleneck of Sequential Model Editing

### 3.1 Preliminary

Model Editing This task involves modifying the memorized knowledge contained in LMs. Various kinds of complex learned beliefs such as logical, spatial, or numerical knowledge are expected to be edited. In this paper, following previous work [41, 67, 42, 64], we study editing factual knowledge in the form of (subject $s$, relation $r$, object $o$ ), e.g., $(s=$ United States, $r=$ President of, $o=$ Donald Trump). An LM is expected to recall a memory representing $o$ given a natural language prompt $p(s, r)$ such as "The President of the United States is". Editing a fact is to incorporate a new knowledge triple $\left(s, r, o^{*}\right)$ in place of the current one $(s, r, o)$. An edit is represented as $e=\left(s, r, o, o^{*}\right)$ for brevity. Given a set of editing facts $\mathcal{E}=\left\{e_{1}, e_{2}, \ldots\right\}$ and an original model $f_{\theta_{0}}$, sequential model editing operationalizes each edit after the last edit ${ }^{2}$, i.e., $K\left(f_{\theta_{n-1}}, e_{n}\right)=f_{\theta_{n}}$, where $f_{\theta_{n}}$ denotes the model after $n$ edits.

Singular Value Decomposition SVD [1] is a fundamental and effective matrix factorization technique for analyzing matrix structures. Formally, an SVD of a matrix $W \in \mathbb{R}^{p \times q}$ is given by[^1]$W=U \Sigma V^{\mathrm{T}}$, where $U=\left[u_{1}, u_{2}, \ldots, u_{p}\right] \in \mathbb{R}^{p \times p}, V=\left[v_{1}, v_{2}, \ldots, v_{q}\right] \in \mathbb{R}^{q \times q}$, and $\Sigma \in \mathbb{R}^{p \times q}$. $u_{i}$ and $v_{i}$ are the column vectors of $U$ and $V$, and constitute an orthonormal basis of $\mathbb{R}^{p}$ and $\mathbb{R}^{q}$ respectively. $\Sigma$ is a diagonal matrix whose diagonal entries are given by the singular values of $W$ in descending order. Additionally, the SVD of $W$ could also be formulated as: $W=\sum_{i=1}^{\min \{p, q\}} \sigma_{i} u_{i} v_{i}^{\mathrm{T}}$, where $\sigma_{i}$ is singular value, and $\sigma_{1} \geq \sigma_{2} \geq \ldots \geq \sigma_{\min \{p, q\}} \geq 0$. In the scenario of this paper, $W$ is a full-rank matrix, so $\sigma_{\min \{p, q\}}>0$.

### 3.2 Matrix Perturbation Theory Analysis

Previous works [16, 41, 23, 59] have analyzed and located that the MLP modules in Transformer [55] store various kinds of knowledge [45, 56]. The MLP module of the $l$-th Transformer layer consists of two projection layers, where the first and second layers are denoted as $W_{f c}^{l}$ and $W_{p r o j}^{l}$ respectively. $W_{p r o j}^{l}$ is considered as a linear associative memory which stores knowledge in the form of key-value pairs $\left(k_{i}, v_{i}\right)$, and is usually regarded as the editing area $[41,42]$. In this paper, $W_{\text {proj }}^{l}$ is denoted as $W$ for brevity. $W$ is assumed to store many key-value pairs $P=\left\{\left(k_{i}, v_{i}\right) \mid i=1,2, \ldots\right\}$ which satisfies $W k_{i}=v_{i}$, where $k_{i} \in \mathbb{R}^{q}$ and $v_{i} \in \mathbb{R}^{p}$. Assuming $|\mathcal{E}|=N$ in sequential model editing, an edit update matrix $\Delta W_{j}$ is calculated for the edit $e_{j}$ and added to $W$, which can be formulated as: $W_{N}=W+\sum_{j=1}^{N} \Delta W_{j}$ with $\Delta W_{j}$ calculated from $f_{\theta_{j-1}}$.

Problem Modeling To explore the reasons for the general ability degradation of edited models, we begin by noting that most of the key-value pairs of $P$ correspond to facts unrelated to editing. For the sake of analysis, only the matrix $W$ of a single layer is assumed to be modified. We intuitively hypothesize that for the facts that are irrelevant to the editing fact, the cumulative modifications applied during sequential model editing may lead to significant mismatches in the associations between the original key-value pairs $P$. Specifically, consider a key-value pair $\left(k_{i}, v_{i}\right) \in P$. After applying an edit $e_{j}$ that generates $\Delta W_{j}$ and adding it to $W$, if the extracted value $v_{i}$ remains unchanged, the corresponding key $k_{i}$ needs to be adjusted with an adjustment denoted as $\Delta k_{i}^{j}$. Mathematically, this can be represented as ${ }^{3} W_{N}\left(k_{i}+\sum_{j=1}^{N} \Delta k_{i}^{j}\right)=v_{i}$ after $N$ edits. However, during the editing process, it's challenging to guarantee such adjustments completely, leading to inaccuracies in the knowledge extracted from the edited model. To delve deeper, let's analyze how the key $k_{i}$ changes (i.e., $\sum_{j=1}^{N} \Delta k_{i}^{j}$ ) when its corresponding value $v_{i}$ remains unchanged after $N$ edits.

Perturbation Analysis of Single Edit According to matrix perturbation theory [38, 54, 60], the edit update matrix $\Delta W$ from an edit can be regarded as a perturbation ${ }^{4}$ for $W$, so we first analyze the situation where $W \in \mathbb{R}^{p \times q}$ is appended with a perturbation $\Delta W$. Define $W^{\dagger}$ is the generalized inverse [51] of $W,\|*\|$ represents 2-norm, and $\tilde{W}=W+\Delta W$.

Theorem 3.1 Consider $W k=v$, there exists $\Delta k$ such that $\tilde{k}=k+\Delta k$ satisfies $\tilde{W} \tilde{k}=v$. Let $k=W^{\dagger} v$ and $\tilde{k}=\tilde{W}^{\dagger} v$, and $\Delta W$ is an acute perturbation of $W$. Then:

$$
\begin{equation*}
\frac{\|\Delta k\|}{\|k\|}=\frac{\|k-\tilde{k}\|}{\|k\|} \leq \hat{\kappa} \frac{\left\|\Delta E_{11}\right\|}{\|W\|}+\Psi_{2}\left(\frac{\hat{\kappa} \Delta E_{12}}{\|W\|}\right)+\hat{\kappa}^{2} \frac{\left\|\Delta E_{12}\right\|}{\|W\|}\left(\eta^{-1} g(v)+\frac{\left\|\Delta E_{21}\right\|}{\|W\|}\right) \tag{1}
\end{equation*}
$$

where $\Delta E_{11}, \Delta E_{12}$, and $\Delta E_{21}$ are directly related to $\Delta W . \Psi_{2}(F)$ is a monotonically increasing function of $\|F\|$ and $g(v)$ is a function about $v . \hat{\kappa}=\|W\|\left\|\tilde{W}_{11}^{-1}\right\|$, where $\tilde{W}_{11}$ is square and related to the reduced form of $W$. Each term on the right-hand side involves $\hat{\kappa}$, which means that the upper bound on the perturbation of the vector $k$ is constrained by $\hat{\kappa}$. Readers can refer to Appendix A. 3 for the details and proof of this theorem. However, calculating $\left\|\tilde{W}_{11}^{-1}\right\|$ involves the reduced form of $W$, which incurs unnecessary additional overhead. Therefore, we consider the following theorem and give an alternative estimation.

Theorem 3.2 Let $\kappa=\|W\|\left\|W^{\dagger}\right\|$, and suppose that $\gamma \equiv 1-\frac{\kappa\left\|\Delta E_{11}\right\|}{\|W\|}>0$. Then:

$$
\begin{equation*}
\left\|\tilde{W}^{\dagger}\right\| \leq \frac{\left\|W^{\dagger}\right\|}{\gamma} \tag{2}
\end{equation*}
$$[^2]![](https://cdn.mathpix.com/cropped/2024_06_04_2d7e827830664cc35591g-05.jpg?height=350&width=1392&top_left_y=242&top_left_x=366)

Figure 2: The condition number, maximum singular value and minimum singular value of the edited matrix in sequential editing. Three editing methods including ROME, MEND, and MEMIT are used to edit LLaMA-2 (7B) on the CounterFact [41] dataset. For editing methods that modify the parameters of multiple MLP layers, one of them is randomly selected for illustration. $W$ and $W_{n}$ denote the unedited and edited matrices respectively.

According to Theorem 3.2, $\left\|\tilde{W}_{11}^{-1}\right\| \leq \frac{\left\|W_{11}^{-1}\right\|}{\gamma}=\frac{\left\|W^{\dagger}\right\|}{\gamma}$, so $\hat{\kappa} \leq \frac{\kappa}{\gamma}$. Here $\kappa=\|W\|\left\|W^{\dagger}\right\|=\frac{\sigma_{\max }}{\sigma_{\min }}$ is the condition number of $W$, where $\sigma_{\max }$ and $\sigma_{\min }$ are the maximum and minimum singular values of $W$, respectively. Combining Theorem 3.1, we know that the larger $\kappa$ is, the greater the upper bound on the perturbation of the vector $k$. Readers can refer to Appendix A for the full theoretical analysis.

### 3.3 Change Trend of Condition Number

As mentioned above, we have analyzed that the condition number of the edited matrix can be used to indicate the upper bound on the perturbation of the key-value pair associations by a single edit. In order to explore the impact of sequential model editing on these associations, the change trend of the condition number of the edited matrix during sequential editing is illustrated in Figure 2.

Surprisingly, we observed that regardless of the editing methods employed, the condition number of the edited matrix exhibited a rapid increase as the number of edits increased, especially after a large number of edits. According to Theorem 3.1, the adjustment norm $\left\|\Delta k_{i}^{n}\right\|_{2}$ corresponding to the $n$-th edit tends to increase as the number of edits $n$ increases. It underscores that when the number of edits is relatively large, the upper bound on the perturbation caused by subsequent edits to the key-value pair associations becomes very large, further disrupting the stored original knowledge and exacerbating the deterioration of general abilities.

## 4 PRUNE: Perturbation Restraint on Upper bouNd for Editing

According to the analysis in Section 3, the bottleneck of the general abilities during sequential editing lies in the escalating value of the condition number. In this section, a framework termed Perturbation Restraint on Upper bouNd for Editing (PRUNE) is proposed, which applies the condition number restraints to preserve general abilities and maintain new editing knowledge simultaneously.

## Principle Given an edited matrix with $N$ edits, $W_{N}=$

 $W+\sum_{j=1}^{N} \Delta W_{j}$, as shown in Figure 2, its maximum singular value is constantly increasing, while the minimum singular value is basically unchanged as the number of edits $N$ increases. This directly leads to the increasing condition number of the edited matrix. Therefore, our motivation is to restrain the maximum singular value of the edited matrix to lower the upper bound on the perturbation. If we directly perform SVD operation on $W_{N}$ and reduce its singular values, the original $W$ will be inevitably destroyed. Consequently, an analysis of thesingular values of $\sum_{j=1}^{N} \Delta W_{j}$ is conducted, and the res singular value becomes very large when $N$ is large. We assume that this is the main reason why the maximum singular value of $W_{N}$ is large, our method therefore aims to restrain the large singular values of $\sum_{j=1}^{N} \Delta W_{j}$.
Table 1: The maximum singular values of $\sum_{j=1}^{N} \Delta W_{j}$ with three edting methods. Other settings are the same as those illustrated in Figure 2.

| Edits $(N)$ | ROME | MEMIT | MEND |
| :--- | :--- | :---: | :---: |
| 10 | 7.25 | 7.46 | 14.08 |
| 50 | 11.38 | 15.63 | 75.53 |
| 100 | 15.62 | 23.39 | 127.89 |
| 200 | 57.61 | 935 | 191.04 |

Design Firstly, SVD is operated on the original $W$ and $\sum_{j=1}^{N} \Delta W_{j}$ respectively as:

$$
\begin{equation*}
W=\sum_{i=1}^{\min \{p, q\}} \sigma_{i} u_{i} v_{i}^{\mathrm{T}}, \quad \sum_{j=1}^{N} \Delta W_{j}=\sum_{i=1}^{\min \{p, q\}} \hat{\sigma}_{i} \hat{u}_{i} \hat{v}_{i}^{\mathrm{T}} \tag{3}
\end{equation*}
$$

This paper considers $W$ to be the main part, and any singular value in $\sum_{j=1}^{N} \Delta W_{j}$ should be ensured not to obviously exceed the maximum singular value of $W$. Subsequently, if any singular value $\hat{\sigma}_{i}$ of $\sum_{j=1}^{N} \Delta W_{j}$ is greater than the maximum singular value of $W$, it will be restrained with a function $F$, otherwise it remains unchanged, which could be formulated as:

$$
\begin{gather*}
\bar{\sigma}_{i}= \begin{cases}F\left(\hat{\sigma}_{i}\right), & \text { if } \hat{\sigma}_{i}>\max \left\{\sigma_{i}\right\} \\
\hat{\sigma}_{i}, & \text { if } \hat{\sigma}_{i} \leq \max \left\{\sigma_{i}\right\}\end{cases}  \tag{4}\\
F\left(\hat{\sigma}_{i}\right)=\log _{\alpha}\left(\hat{\sigma}_{i}\right)-\log _{\alpha}\left(\max \left\{\sigma_{i}\right\}\right)+\max \left\{\sigma_{i}\right\} \tag{5}
\end{gather*}
$$

In the main paper, we use the $\log$ function in $F$ to restrain $\hat{\sigma}_{i}$. Here $\alpha$ is a hyperparameter to control the degree of restraints, readers can refer to Appendix B. 3 for its details for experiments. Besides, we also provide the definition and results of linear function in Appendix C.3. Finally, we obtain the restrained edited matrix $\bar{W}_{N}$ to replace $W_{N}$ :

$$
\begin{equation*}
\bar{W}_{N}=W+\sum_{i=1}^{\min \{p, q\}} \bar{\sigma}_{i} \hat{u}_{i} \hat{v}_{i}^{\mathrm{T}} \tag{6}
\end{equation*}
$$

In this way, the condition number of the edited matrix is reduced (see Appendix C.4) and the upper bound on perturbation is significantly restrained.

## 5 Experiments

In this section, both the downstream task performance and editing performance of three editing methods on three LLMs were evaluated in sequential model editing. The proposed PRUNE was plug-and-play which can be coupled with these editing methods.

### 5.1 Base LLMs and Editing Methods

Experiments were conducted on three LLMs including GPT-2 XL (1.5B) [48], LLaMA-2 (7B) [53] and LLaMA-3 (8B $)^{5}$. Three popular editing methods were selected as the baselines including MEND [43], ROME [41], and MEMIT [42]. Readers can refer to Appendix B. 1 for the details of these editing methods.

### 5.2 Editing Datasets and Evaluation Metrics

Two popular model editing datasets Zero-Shot Relation Extraction (ZsRE) [33] and CounTERFACT [41] were adopted in our experiments. ZsRE is a QA dataset using question rephrasings generated by back-translation as the equivalence neighborhood. A key distinction between COUNTERFACT and ZsRE datasets is that ZsRE contains true facts, while COUNTERFACT contains counterfactual examples where the new target has a lower probability when compared to the original answer [22]. Readers can refer to Appendix B. 2 for examples of each dataset.

To assess the editing performance of editing methods, following previous works, three fundamental metrics were employed: efficacy, generalization and locality [3, 43, 41, 42]. Given an original model $f_{\theta_{0}}$, an edited model $f_{\theta_{n}}$ with $n$ times sequential editing. Each edit $e_{i}=\left(s_{i}, r_{i}, o_{i}, o_{i}^{*}\right)$ has an editing prompt $p_{i}$, paraphrase prompts $\mathcal{P}_{i}^{G}$, and locality prompts $\mathcal{P}_{i}^{L}$.

Efficacy validates whether the edited models could recall the editing fact under editing prompt $p_{i}$. The assessment is based on Efficacy Score (ES) representing as: $\mathbb{E}_{i}\left[\mathbb{1}\left[P_{f_{\theta_{n}}}\left(o_{i}^{*} \mid p_{i}\right)>P_{f_{\theta_{n}}}\left(o_{i} \mid p_{i}\right)\right]\right]$, where $\mathbb{1}$ is the indicator function.

Generalization verifies whether the edited models could recall the editing fact under the paraphrase prompts $\mathcal{P}_{i}^{G}$ via Generalization Score $(\mathbf{G S}): \mathbb{E}_{i}\left[\mathbb{E}_{p \in \mathcal{P}_{i}^{G}}\left[\mathbb{1}\left[P_{f_{\theta_{n}}}\left(o_{i}^{*} \mid p\right)>P_{f_{\theta_{n}}}\left(o_{i} \mid p\right)\right]\right]\right.$.[^3]![](https://cdn.mathpix.com/cropped/2024_06_04_2d7e827830664cc35591g-07.jpg?height=452&width=1392&top_left_y=240&top_left_x=366)

Figure 3: The downstream task performance (\%) of models edited by three editing methods with LLaMA-2 (7B) on the COUNTERFACT dataset. The dashed lines refer to the results of the unrestrained editing methods. The solid lines refer to the results of the editing methods coupled with the proposed PRUNE framework. Statistical significance tests were performed to demonstrate that the improvement in PRUNE compared to baseline was statistically significant (t-test with $p$-value $<0.05$ ).

Locality verifies whether the output of the edited models for inputs out of editing scope remains unchanged under the locality prompts $\mathcal{P}_{i}^{L}$ via Locality Score (LS): $\mathbb{E}_{i}\left[\mathbb{E}_{p_{l} \in \mathcal{P}_{i}^{L}}\left[\mathbb{1}\left[P_{f_{\theta_{n}}}\left(o_{l} \mid p_{l}\right)>\right.\right.\right.$ $\left.\left.\left.P_{f_{\theta_{n}}}\left(o_{i}^{*} \mid p_{l}\right)\right]\right]\right]$, where $o_{l}$ was the original answer of $p_{l}$.

Different from previous studies that assess the edited models after each individual edit [22, 62], this paper evaluated whether the final edited models after completing all edits can still recall all preceding edits, which is more challenging and common in real-world.

### 5.3 Downstream Tasks, Datasets and Metrics

To explore the side effects of sequential model editing on the general abilities of LLMs, four representative tasks with corresponding datasets were adopted for assessment, including:

Reasoning on the GSM8K [8], and the results were measured by solve rate.

Summarization on the SAMSum [19], and the results were measured by the average of ROUGE-1, ROUGE-2 and ROUGE-L following Lin [36].

Open-domain QA on the Natural Question [31], and the results were measured by exact match (EM) with the reference answer after minor normalization as in Chen et al. [4] and Lee et al. [32].

Natural language inference (NLI) on the RTE [11], and the results were measured by accuracy of two-way classification.

For each dataset, some examples were randomly sampled for evaluation. Details of prompts for each tasks were shown in Appendix B.4.

### 5.4 Results of General Abilities

Figure 3 illustrates the downstream task performance of edited models with LLaMA-2 (7B) on the CounterFact dataset. Due to page limitation, results of other LLMs and datasets were put in Appendix C.1. These results were analyzed from the following perspectives.

Current editing methods significantly compromised general abilities. As depicted by the dashed lines of Figure 3, both the ROME and MEMIT methods initially maintained relatively stable performance in downstream tasks when the number of edits was small $(\leq 100)$. However, as the number of edits surpassed 200, a noticeable decline in performance was observed across all tasks for both methods. Additionally, the MEND method exhibited significant performance degradation after just 20 sequential edits, indicating its inadequacy as a sequential model editing method. Furthermore, when comparing LLMs of different sizes, a general trend emerged: larger models suffered more pronounced compromises in their general abilities when subjected to the same number of edits. For instance, with 300 edits, MEMIT's performance on GPT2-XL remained largely unchanged, whereas it dwindled to nearly 0 on LLaMA-2 and LLaMA-3.
![](https://cdn.mathpix.com/cropped/2024_06_04_2d7e827830664cc35591g-08.jpg?height=404&width=1356&top_left_y=237&top_left_x=384)

Figure 4: The editing performance (\%) of three editing methods with LLaMA-2 (7B) on the CoUntErFact dataset. The dashed lines refer to the results of the unrestrained editing methods. The solid lines refer to the results of the editing methods coupled with the proposed PRUNE. Statistical significance tests were performed to demonstrate that the improvement in PRUNE compared to baseline was statistically significant (t-test with $p$-value $<0.05$ ).

The performance decline was gradual initially but accelerated with increasing edit count. This trend aligned with the fluctuation observed in the size of the condition number, as depicted in Figure 2. When the number of edits was small, the condition number was small, and each new edit introduced relatively minor perturbations to the model. However, as the number of edits increased, the condition number underwent a substantial increase. Consequently, each subsequent edit exerted a significant perturbation on the model, leading to a pronounced impairment of its general abilities. These results substantiated the analysis presented in Section 3.3.

The proposed PRUNE can preserve considerable general abilities. As shown by the solid lines of Figure 3, when MEMIT was coupled with PRUNE and subjected to 200 edits, its downstream tasks performance remained close to that of the unedited model. However, for the unrestrained MEMIT, downstream task performance had plummeted to nearly 0 by this point. This consistent trend was also observed with ROME and MEND. Nevertheless, for models edited using the unrestrained MEND method, performance degradation was stark after just 20 edits. Even with the addition of PRUNE, preservation could only be extended up to 40 edits. This suggests that while PRUNE effectively preserves general abilities, it does have an upper limit determined by the unrestrained editing method.

### 5.5 Results of Editing Performance

Figure 4 shows different metrics used for measuring the editing performance of edited models with LLaMA-2 (7B) on the CountERFACT dataset. Other results across models and datasets were put in Appendix C.2. Three conclusions can be drawn here.

Previous editing facts were forgotten as the number of edits increased. As shown by the dashed lines of Figure 4, the decline in efficacy and generalization suggests that in sequential editing scenarios, post-edited models gradually forget knowledge acquired from previous edits after a few iterations. Comparing these editing methods, we also observed a notable drop in efficacy and generalization after hundreds of edits with ROME and MEMIT, whereas these values decreased significantly after only 30 edits with MEND. This indicates that in sequential editing scenarios, the MEND method struggled to successfully integrate new knowledge into LLMs after several edits.

Unrelated facts were perturbed as the number of edits increased. The locality metric served as an indicator of perturbation for unrelated facts. It became evident that for each editing method, the locality decreased significantly. Additionally, an observation emerged: when the locality of the edited model was low, the performance of downstream tasks was also low. This observation underscores that perturbations of irrelevant knowledge compromise the general abilities of the edited model.

PRUNE can effectively maintain the editing performance. This is shown by the solid lines of Figure 4 and could be analyzed from two aspects. On the one hand, when the number of edits was small, the editing performance of each editing method coupled with PRUNE was about the same as the unrestrained method. On the other hand, it significantly mitigated the forgetting of editing facts and the perturbation of irrelevant facts when the number of edits was large during the sequential editing. Specifically, when the number of edits reached 150, the editing performance of MEMIT was very low. But when coupled with PRUNE, its performance remained relatively stable.

### 5.6 Editing Facts Forgetting Analysis

In section 3 , the analysis was conducted to elucidate the reasons behind the degradation in general abilities with an increasing number of edits. Subsequent experiments quantitatively demonstrated the effectiveness of the proposed PRUNE. Here, we delve into qualitative analysis to explain why editing facts are forgotten and how PRUNE can mitigate this forgetting.

Initially, a set of editing facts $\mathcal{E}=\left\{e_{1}, e_{2}, \ldots\right\}$ was collected, where $|\mathcal{E}|=200$. ROME was employed for analysis, and the original matrix was defined as $W$. During sequential editing, ROME computed key-value pairs $\left(k_{j}^{e}, v_{j}^{e}\right)$ of the last subject token to generate $\Delta W_{j}$ for each edit $e_{j}$ to incorporate new facts, satisfying the equation: $W_{j} \cdot k_{j}^{e}=v_{j}^{e}$. However, when evaluating editing performance, the edited model obtained from the last edit was utilized, thus computing values ${ }^{6}: W_{200} \cdot k_{j}^{e}=\hat{v}_{j}^{e}$. After adopting PRUNE to ROME, this equation became $\bar{W}_{200} \cdot k_{j}^{e}=\bar{v}_{j}^{e}$. We hypothesized that if $\hat{v}_{j}^{e}$ was similar

![](https://cdn.mathpix.com/cropped/2024_06_04_2d7e827830664cc35591g-09.jpg?height=517&width=567&top_left_y=346&top_left_x=1191)

Figure 5: 2-dimensional PCA visualization of first 100 values. The model was edited by ROME with LLaMA-3 (8B) on the COUNTERFACT dataset. to $v_{j}^{e}$, the editing fact $e_{j}$ could be maintained.

Denote $V_{\text {Current }}=\left\{v_{j}^{e}\right\}, V_{\text {Editing }}=\left\{\hat{v}_{j}^{e}\right\}$, and $V_{\text {Prune }}=\left\{\bar{v}_{j}^{e}\right\}$. Specifically, these corresponding values of the first 100 edits were used, as they are more prone to be forgotten than the last 100 . Principal Component Analysis (PCA) [18] was employed to visualize these values. The first two principal components of each value were calculated and illustrated, as they can represent most of its features [66]. As shown in Figure 5, on the one hand, the discrepancy between the principal components of $V_{\text {Current }}$ and $V_{\text {Editing }}$ was markedly large. This indicates that after 200 edits to the model, the values corresponding to the first 100 facts stored in the edited matrix are severely corrupted, leading to significant forgetfulness. On the other hand, after adopting PRUNE, the discrepancy between the principal components of $V_{\text {Current }}$ and $V_{\text {Prune }}$ was small. This demonstrates that PRUNE effectively maintains the values and mitigates the forgetting of editing facts.

## 6 Conclusion

In this paper, a theoretical analysis is firstly conducted to elucidate that the bottleneck of the general abilities during sequential editing lies in the escalating value of the condition number. Subsequently, a plug-and-play framework called PRUNE is proposed to apply restraints to preserve general abilities and maintain new editing knowledge simultaneously. Comprehensive experiments on various editing methods and LLMs demonstrate the effectiveness of this method. We aspire that our analysis and method will catalyze future research on continual model editing.

## Limitations \& Future Work

The limitations of our work are discussed as follows. Firstly, while this paper focuses on editing a single fact at a time in sequential model editing, some studies have explored updating hundreds or thousands of facts simultaneously in batch editing. Therefore, investigating batch-sequential editing could enhance the scalability of model editing and remains further research. Secondly, for the experimental settings, it is necessary to explore the performance of larger-size models and more editing methods on more downstream tasks. Thirdly, the current focus of editing knowledge primarily revolves around factual knowledge. However, it is also important to investigate whether editing other types of knowledge will affect general abilities and whether PRUNE is effective in this situation. Additionally, the proposed PRUNE is only applied once after the completion of the last edit. But it could also be utilized multiple times during the sequential editing process, and we intuitively believe that this approach would be more conducive to preserving the general abilities of the model. These aspects are yet to be fully understood and warrant a more comprehensive study.[^4]

## References

[1] Alfonso M Albano, J Muench, C Schwartz, AI Mees, and PE Rapp. Singular-value decomposition and the grassberger-procaccia algorithm. Physical review A, 38(6):3017, 1988.

[2] Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz, Vanessa Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker. Fairlearn: A toolkit for assessing and improving fairness in ai. Microsoft, Tech. Rep. MSR-TR-2020-32, 2020.

[3] Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 6491-6506. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021. emnlp-main.522. URL https://doi.org/10.18653/v1/2021.emnlp-main. 522.

[4] Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer open-domain questions. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1870-1879. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17-1171. URL https://doi .org/10. $18653 / \mathrm{v} 1 / \mathrm{P} 17-1171$.

[5] Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip H. S. Torr, and Volker Tresp. Benchmarking robustness of adaptation methods on pre-trained vision-language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ a2a544e43acb8b954dc5846ff0d77ad5-Abstract-Datasets_and_Benchmarks.html.

[6] Zhuotong Chen, Zihu Wang, Yifan Yang, Qianxiao Li, and Zheng Zhang. PID control-based selfhealing to improve the robustness of large language models. CoRR, abs/2404.00828, 2024. doi: 10.48550/ARXIV.2404.00828. URL https://doi.org/10.48550/arXiv.2404.00828.

[7] Wenhua Cheng, Weiwei Zhang, Haihao Shen, Yiyang Cai, Xin He, and Kaokao Lv. Optimize weight rounding via signed gradient descent for the quantization of llms. CoRR, abs/2309.05516, 2023. doi: 10.48550/ARXIV.2309.05516. URL https://doi.org/10.48550/arXiv. 2309 . 05516 .

[8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

[9] Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. Evaluating the ripple effects of knowledge editing in language models. CoRR, abs/2307.12976, 2023. doi: 10.48550/ ARXIV.2307.12976. URL https://doi.org/10.48550/arXiv.2307.12976.

[10] Jeff Da, Ronan Le Bras, Ximing Lu, Yejin Choi, and Antoine Bosselut. Analyzing commonsense emergence in few-shot knowledge models. In Danqi Chen, Jonathan Berant, Andrew McCallum, and Sameer Singh, editors, 3rd Conference on Automated Knowledge Base Construction, AKBC 2021, Virtual, October 4-8, 2021, 2021. doi: 10.24432/C5NK5J. URL https://doi.org/10. 24432/C5NK5J.

[11] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Joaquin Quiñonero Candela, Ido Dagan, Bernardo Magnini, and Florence d'AlchéBuc, editors, Machine Learning Challenges, Evaluating Predictive Uncertainty, Visual Object Classification and Recognizing Textual Entailment, First PASCAL Machine Learning Challenges Workshop, MLCW 2005, Southampton, UK, April 11-13, 2005, Revised Selected Papers, volume 3944 of Lecture Notes in Computer Science, pages 177-190. Springer, 2005. doi: 10.1007/ 11736790\9. URL https://doi.org/10.1007/11736790_9.

[12] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 84938502. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.acl-long.581. URL https://doi.org/10.18653/v1/2022.acl-long. 581.

[13] Jean-Pierre Dedieu. Condition operators, condition numbers, and condition number theorem for the generalized eigenvalue problem. Linear algebra and its applications, 263:1-24, 1997.

[14] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless LLM weight compression. CoRR, abs/2306.03078, 2023. doi: 10.48550/ARXIV.2306.03078. URL https://doi.org/10.48550/arXiv.2306.03078.

[15] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. CoRR, abs/2303.07345, 2023. doi: 10.48550/ARXIV.2303.07345. URL https://doi.org/10.48550/arXiv.2303.07345.

[16] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 5484-5495. Association for Computational Linguistics, 2021. doi: 10.18653/v1/ 2021.emnlp-main.446. URL https://doi.org/10.18653/v1/2021.emnlp-main. 446.

[17] Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 30-45. Association for Computational Linguistics, 2022. doi: 10.18653/v1/ 2022.emnlp-main.3. URL https://doi.org/10.18653/v1/2022.emnlp-main. 3.

[18] Felipe L. Gewers, Gustavo R. Ferreira, Henrique Ferraz de Arruda, Filipi Nascimento Silva, Cesar H. Comin, Diego R. Amancio, and Luciano da Fontoura Costa. Principal component analysis: A natural approach to data exploration. ACM Comput. Surv., 54(4):70:1-70:34, 2022. doi: 10.1145/3447755. URL https://doi.org/10.1145/3447755.

[19] Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. In Lu Wang, Jackie Chi Kit Cheung, Giuseppe Carenini, and Fei Liu, editors, Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70-79, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-5409. URL https://aclanthology . org/D19-5409.

[20] Zhuocheng Gong, Jiahao Liu, Jingang Wang, Xunliang Cai, Dongyan Zhao, and Rui Yan. What makes quantization for large language model hard? an empirical study from the lens of perturbation. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 18082-18089. AAAI Press, 2024. doi: 10.1609/AAAI.V38I16.29765. URL https://doi.org/10.1609/aaai.v38i16.29765.

[21] Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, and Nanyun Peng. Model editing can hurt general abilities of large language models. CoRR, abs/2401.04700, 2024. doi: 10.48550/ARXIV.2401.04700. URL https://doi.org/10.48550/arXiv. 2401 . 04700 .

[22] Akshat Gupta, Anurag Rao, and Gopala Anumanchipalli. Model editing at scale leads to gradual and catastrophic forgetting. CoRR, abs/2401.07453, 2024. doi: 10.48550/ARXIV.2401.07453. URL https://doi.org/10.48550/arXiv.2401.07453.

[23] Anshita Gupta, Debanjan Mondal, Akshay Krishna Sheshadri, Wenlong Zhao, Xiang Lorraine Li, Sarah Wiegreffe, and Niket Tandon. Editing commonsense knowledge in GPT. CoRR, abs/2305.14956, 2023. doi: 10.48550/ARXIV.2305.14956. URL https://doi.org/10. 48550/arXiv. 2305.14956 .

[24] Frederik Harder, Matthias Bauer, and Mijung Park. Interpretable and differentially private predictions. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 4083-4090. AAAI Press, 2020. doi: 10.1609/AAAI. V34I04.5827. URL https://doi.org/10.1609/aaai.v34i04.5827.

[25] Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. Aging with GRACE: lifelong model editing with discrete key-value adaptors. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html.

[26] Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. CoRR, abs/2301.04213, 2023. doi: 10.48550/ARXIV.2301.04213. URL https: //doi.org/10.48550/arXiv.2301.04213.

[27] Chenhui Hu, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Wilke: Wise-layer knowledge editor for lifelong knowledge editing. CoRR, abs/2402.10987, 2024. doi: 10.48550/ARXIV. 2402.10987. URL https://doi.org/10.48550/arXiv.2402.10987.

[28] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. CoRR, abs/2311.05232, 2023. doi: 10.48550/ARXIV.2311.05232. URL https: //doi.org/10.48550/arXiv. 2311.05232.

[29] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12):248:1-248:38, 2023. doi: 10.1145/3571730. URL https ://doi . org $/ 10.1145 / 3571730$.

[30] Shuoran Jiang, Qingcai Chen, Youcheng Pan, Yang Xiang, Yukang Lin, Xiangping Wu, Chuanyi Liu, and Xiaobao Song. Zo-adamu optimizer: Adapting perturbation by the momentum and uncertainty in zeroth-order optimization. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, ThirtySixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 18363-18371. AAAI Press, 2024. doi: 10.1609/AAAI.V38I16. 29796. URL https://doi.org/10.1609/aaai.v38i16.29796.

[31] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics, 7:452-466, 2019. doi: 10.1162/TACL\A\_00276. URL https: //doi.org/10.1162/tacl_a_00276.

[32] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 60866096. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1612. URL https://doi.org/10.18653/v1/p19-1612.

[33] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Roger Levy and Lucia Specia, editors, Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), Vancouver, Canada, August 3-4, 2017, pages 333-342. Association for Computational Linguistics, 2017. doi: 10.18653/v1/K17-1034. URL https://doi.org/10.18653/v1/K17-1034.

[34] Hui-Jia Li, Lin Wang, Yan Zhang, and Matjaž Perc. Optimization of identifiability for efficient community detection. New Journal of Physics, 22(6):063035, 2020.

[35] Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, and Huajun Chen. Unveiling the pitfalls of knowledge editing for large language models. CoRR, abs/2310.02129, 2023. doi: 10.48550/ARXIV.2310.02129. URL https://doi.org/10.48550/arXiv.2310.02129.

[36] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74-81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.

[37] Zihao Lin, Mohammad Beigi, Hongxuan Li, Yufan Zhou, Yuxiang Zhang, Qifan Wang, Wenpeng Yin, and Lifu Huang. Navigating the dual facets: A comprehensive evaluation of sequential memory editing in large language models. CoRR, abs/2402.11122, 2024. doi: 10.48550/ARXIV.2402.11122. URL https://doi.org/10.48550/arXiv.2402.11122.

[38] Zhi-Quan Luo and Paul Tseng. Perturbation analysis of a condition number for linear systems. SIAM Journal on Matrix Analysis and Applications, 15(2):636-660, 1994.

[39] Jun-Yu Ma, Jia-Chen Gu, Zhen-Hua Ling, Quan Liu, and Cong Liu. Untying the reversal curse via bidirectional language model editing. CoRR, abs/2310.10322, 2023. doi: 10.48550/ARXIV. 2310.10322. URL https://doi.org/10.48550/arXiv.2310.10322.

[40] Jun-Yu Ma, Jia-Chen Gu, Ningyu Zhang, and Zhen-Hua Ling. Neighboring perturbations of knowledge editing on large language models. CoRR, abs/2401.17623, 2024. doi: 10.48550/ ARXIV.2401.17623. URL https://doi.org/10.48550/arXiv.2401.17623.

[41] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In NeurIPS, 2022. URL https://arxiv.org/abs/2202.05262.

[42] Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Massediting memory in a transformer. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=MkbcAHIYgyS.

[43] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast model editing at scale. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https:// openreview.net/forum?id=0DcZxeWfOPt.

[44] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn. Memory-based model editing at scale. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 15817-15831. PMLR, 2022. URL https://proceedings.mlr.press/v162/mitchell22a.html.

[45] Judea Pearl. Direct and indirect effects. In Jack S. Breese and Daphne Koller, editors, UAI '01: Proceedings of the 17th Conference in Uncertainty in Artificial Intelligence, University of Washington, Seattle, Washington, USA, August 2-5, 2001, pages 411-420. Morgan Kaufmann, 2001. URL https://dslpitt.org/uai/displayArticleDetails.jsp?mmnu=1\&smnu= 2\&article_id=126\&proceeding_id=17.

[46] Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. Check your facts and try again: Improving large language models with external knowledge and automated feedback. CoRR, abs/2302.12813, 2023. doi: 10.48550/arXiv.2302.12813. URL https://doi.org/10.48550/ arXiv. 2302.12813 .

[47] Bin Qin, Fu-Lai Chung, and Shitong Wang. KAT: A knowledge adversarial training method for zero-order takagi-sugeno-kang fuzzy classifiers. IEEE Trans. Cybern., 52(7):6857-6871, 2022. doi: 10.1109/TCYB.2020.3034792. URL https://doi.org/10.1109/TCYB. 2020. 3034792 .

[48] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[49] Chandan Singh, Jeevana Priya Inala, Michel Galley, Rich Caruana, and Jianfeng Gao. Rethinking interpretability in the era of large language models. CoRR, abs/2402.01761, 2024. doi: 10.48550/ARXIV.2402.01761. URL https://doi.org/10.48550/arXiv.2402. 01761.

[50] Russell A Smith. The condition numbers of the matrix eigenvalue problem. Numerische Mathematik, 10:232-240, 1967.

[51] Gilbert W Stewart and Ji-guang Sun. Matrix perturbation theory. (No Title), 1990.

[52] Ji-guang Sun. Condition number and backward error for the generalized singular value decomposition. SIAM Journal on Matrix Analysis and Applications, 22(2):323-341, 2000.

[53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, et al. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.

[54] Richard J Vaccaro. A second-order perturbation expansion for the svd. SIAM Journal on Matrix Analysis and Applications, 15(2):661-671, 1994.

[55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998-6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[56] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart M. Shieber. Investigating gender bias in language models using causal mediation analysis. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020 .

[57] Michael E Wall, Andreas Rechtsteiner, and Luis M Rocha. Singular value decomposition and principal component analysis. In A practical approach to microarray data analysis, pages $91-109$. Springer, 2003.

[58] Peng Wang, Ningyu Zhang, Xin Xie, Yunzhi Yao, Bozhong Tian, Mengru Wang, Zekun Xi, Siyuan Cheng, Kangwei Liu, Guozhou Zheng, and Huajun Chen. Easyedit: An easy-to-use knowledge editing framework for large language models. CoRR, abs/2308.07269, 2023. doi: 10.48550/arXiv.2308.07269. URL https://doi.org/10.48550/arXiv.2308.07269.

[59] Xiaohan Wang, Shengyu Mao, Ningyu Zhang, Shumin Deng, Yunzhi Yao, Yue Shen, Lei Liang, Jinjie Gu, and Huajun Chen. Editing conceptual knowledge for large language models. CoRR, abs/2403.06259, 2024. doi: 10.48550/ARXIV.2403.06259. URL https://doi.org/ 10.48550/arXiv.2403.06259.

[60] Per-Åke Wedin. Perturbation bounds in connection with singular value decomposition. BIT Numerical Mathematics, 12:99-111, 1972.

[61] Suhang Wu, Minlong Peng, Yue Chen, Jinsong Su, and Mingming Sun. Eva-kellm: A new benchmark for evaluating knowledge editing of llms. CoRR, abs/2308.09954, 2023. doi: 10.48550/ARXIV.2308.09954. URL https://doi.org/10.48550/arXiv.2308.09954.

[62] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and opportunities. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 10222-10240. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main. 632.

[63] Lang Yu, Qin Chen, Jie Zhou, and Liang He. MELO: enhancing model editing with neuronindexed dynamic lora. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 19449-19457. AAAI Press, 2024. doi: 10.1609/AAAI.V38I17.29916. URL https://doi.org/10.1609/aaai.v38i17.29916.

[64] Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, and Huajun Chen. A comprehensive study of knowledge editing for large language models. CoRR, abs/2401.01286, 2024. doi: 10.48550/ARXIV.2401.01286. URL https: //doi.org/10.48550/arXiv.2401.01286.

[65] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. Siren's song in the AI ocean: A survey on hallucination in large language models. CoRR, abs/2309.01219, 2023. doi: 10.48550/arXiv.2309.01219. URL https://doi.org/10 . 48550/arXiv. 2309.01219.

[66] Chujie Zheng, Fan Yin, Hao Zhou, Fandong Meng, Jie Zhou, Kai-Wei Chang, Minlie Huang, and Nanyun Peng. On prompt-driven safeguarding for large language models. In ICLR 2024 Workshop on Secure and Trustworthy Large Language Models.

[67] Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, and Danqi Chen. Mquake: Assessing knowledge editing in language models via multi-hop questions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 15686-15702. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main. 971.
</end of paper 1>


<paper 2>
# Knowledge Circuits in Pretrained Transformers 

Yunzhi Yao ${ }^{1} \quad$ Ningyu Zhang ${ }^{1 *}$ Zekun Xi ${ }^{1} \quad$ Mengru Wang ${ }^{1}$<br>Ziwen Xu ${ }^{1}$ Shumin Deng ${ }^{2} \quad$ Huajun Chen ${ }^{1 *}$<br>${ }^{1}$ Zhejiang University ${ }^{2}$ National University of Singapore<br>\{yyztodd,zhangningyu\}@zju.edu.cn


#### Abstract

The remarkable capabilities of modern large language models are rooted in their vast repositories of knowledge encoded within their parameters, enabling them to perceive the world and engage in reasoning. The inner workings of how these models store knowledge have long been a subject of intense interest and investigation among researchers. To date, most studies have concentrated on isolated components within these models, such as the Multilayer Perceptrons and attention head. In this paper, we delve into the computation graph of the language model to uncover the knowledge circuits that are instrumental in articulating specific knowledge. The experiments, conducted with GPT2 and TinyLLAMA, has allowed us to observe how certain information heads, relation heads, and Multilayer Perceptrons collaboratively encode knowledge within the model. Moreover, we evaluate the impact of current knowledge editing techniques on these knowledge circuits, providing deeper insights into the functioning and constraints of these editing methodologies. Finally, we utilize knowledge circuits to analyze and interpret language model behaviors such as hallucinations and in-context learning. We believe the knowledge circuits hold potential for advancing our understanding of Transformers and guiding the improved design of knowledge editing ${ }^{1}$.


## 1 Introduction

"Knowledge is power, and when embodied in the form of new technical inventions and mechanical discoveries it is the force that drives history." [1, 2], Bacon's words are vividly re-enacted in the era of Large Language Models (LLMs) [3, 4], as we witness their immense power in reshaping human society and redefining our understanding of machine intelligence. One thing that cannot be denied is that knowledge encapsulated within these models empowers their capabilities in reasoning, perceiving the world, and engaging in human-like communication. Nevertheless, these powerful models are not without their flaws. They still struggle with issues such as hallucinations [5-7], unsafe norms $[8,9]$, and offensive behaviors $[10,11]$ and these problems are exacerbated by the enigmatic internal mechanisms of knowledge storage within language models.

Recently, the research community has devoted significant efforts to unraveling the knowledge storage mechanisms of these models. Various studies [12-19] have been conducted to shed light on this intricate process, aiming to enhance our understanding and improve the safety and reliability of language models. The main finding in previous work is that knowledge may primarily stored in the Multilayer Perceptrons (MLPs) of Transformer-based language models. These MLPs function as a key-value neural memory, with knowledge being stored in what are termed "knowledge neurons" (KN). Based on these findings, researchers conduct Knowledge Editing [18, 20] to update the language models' inaccurate facts, bias and unsafe content in their parametric space. Despite the initial success of these methods, there are still limitations, such as poor generalization, severe side effect, and failure[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-02.jpg?height=683&width=1393&top_left_y=239&top_left_x=363)

Figure 1: Knowledge circuit obtained from "The official language of France is French" in GPT2Medium. Left: a simplified circuit and the whole circuit is in Figure 8 in Appendix. We use $\rightarrow$ to skip some complex connections between nodes. Right: the behavior of several special heads.

to effectively utilize edited knowledge [20, 21], which motivate us to re-think previous approaches for interpreting knowledge storage in language models. Note that previous works treat the knowledge blocks as isolated components following the Restorative Theory [22], often focusing on identifying the specific blocks that stores particular knowledge. Several works [23, 24] have proposed that different types of knowledge are often located in the same areas, suggesting that the current $\mathrm{KN}$ thesis may be an oversimplification.

To this end, instead of solely pinpointing tiny regions where the knowledge expressed can be localized, we aim to explore the cooperation between different components in Transformers like attention heads, MLPs, and embeddings, to understand how the language model stores and expresses the knowledge. Here, we introduce a new perspective: Knowledge Circuits, a critical subgraph in the language model to view the knowledge mechanism of Transformers. Note that Circuits, as a subgraph in the computation graph, has gained ever-growing attention in the mechanistic interpretability field [25]. Previous work [26,27] has found several important circuits for specific tasks like Indirect Object Identification and Color Object. These tasks necessitate the model to search the preceding context for a matching token and copy it into the next token prediction. In this work, we aim to construct knowledge circuits that require the model to utilize stored knowledge for making predictions. Our goal is to better unveil implicit neural knowledge representations, elucidate the internal mechanisms for knowledge editing, and interpret more complex behaviors of language models. Specifically, we leverage factual recall tasks and conduct experiments across various domains, including factual, bias, linguistic, and commonsense knowledge. We utilize GPT-2 [28] and TinyLLAMA [29] to explore the potential knowledge representations and utilization mechanism in these models. As shown in Figure 1 (a), we construct knowledge circuits associated with various expressions of knowledge using the existing knowledge stored in the language model. Through those discovered knowledge circuits, we find many interesting phenomena and conclusions as follows:

Knowledge circuits unveil implicit neural knowledge representations. We find that even when the knowledge circuits are used independently, the language model can recall related knowledge with a significant portion of its overall performance, demonstrating the effectiveness of those discovered knowledge representations (circuits). We also delve into specific pieces of knowledge and analyze the information flow within their respective circuits, indicating that language model tends to aggregate knowledge in the earlier to middle layers and further enhances this information in the later layers. We further uncover several special components (e.g., attention heads) in transferring information to the final token position and capturing relational information from the context (Figure 1 (b)).

Knowledge circuits elucidate internal mechanisms for knowledge editing. We conduct experiments to evaluate the impact of current knowledge editing methods on the language models' original
knowledge circuits. Empirically, we observe that ROME [18] tends to incorporate edited information primarily at the edited layer. Subsequent mover heads (Appendix B.2) then transport this information to the residual stream of the last token. Conversely, during fine-tuning, the edited token is directly integrated into the language model, exerting a dominant influence on subsequent predictions.

Knowledge circuits facilitate interpreting language model behaviors. We further utilize the knowledge circuits to interpret language model behaviors, such as hallucination and in-context learning. We observe that when hallucination occurs, the language model fails to correctly transfer knowledge to the final token in the earlier layers. This is evident as the knowledge circuit lacks an effective "mover" head, or the mover head selects incorrect information. Additionally, we notice that several new attention heads emerge in the knowledge circuit during in-context learning.

## 2 Background: Circuit Theory

### 2.1 Preliminaries

In the context of neural network interpretability, a circuit can be conceptualized as a humaninterpretable subgraph that is dedicated to executing specific tasks within a neural network model [30, 26, 31-33]. When we visualize a neural network model as a connected directed acyclic graph (DAG), denoted as $\mathcal{G}$, the individual nodes represent the various components involved in the forward pass, such as neurons, attention heads, and embeddings. The edges symbolize the interactions between these components, including residual connections, attention mechanisms, and projections. A circuit, represented as $\mathcal{C} \subseteq \mathcal{G}$, emerges as a significant subgraph of $\mathcal{G}$ that is responsible for particular behaviors or functionalities. In this paper, we focus on the Transformer decoder architecture to conduct our experiments. The residual stream of Transformers has been demonstrated to be a valuable tool for mechanistic interpretability in recent works [25, 16]. The Transformer architecture typically starts with token embeddings, followed by a sequence of "residual blocks" and concludes with a token unembedding. Each residual block comprises an attention layer and an MLP layer, both of which "read" their input from the residual stream (via a linear projection) and "write" their output back to the residual stream through an additive projection. We can consider an attention head $A_{l, j}$ (the $j$ th attention head in layer $l$ ) as operating on the residual stream from the previous layer, $R_{l-1}$. Given that $R_{0}=I$ (where $I$ represents the input embeddings), we can reinterpret attention head $A_{l, j}$ as processing the cumulative output of all previous attention heads and MLPs and input embedding, treating each node in the previous layers as separate input arguments. Similarly, an MLP node $M_{l}$ can be seen as operating on the cumulative output of all previous attention heads and MLPs and input embedding, and the output node $O$ operates on the sum of the input embeddings and the outputs of all attention heads and MLPs. The following equations represent the residual connections in the Transformer model, where $R_{l}$ is the residual stream at layer $l$, and $\operatorname{Input}_{l}^{A}$ and $\operatorname{Input}_{l}^{M}$ are the inputs to the attention and MLP layers, respectively:

$$
\begin{array}{r}
R_{l}=R_{l-1}+\sum_{j} A_{l, j}+M_{l}, R_{0}=I \\
\operatorname{Input}_{l}^{A}=I+\sum_{l^{\prime}<l}\left(M_{l^{\prime}}+\sum_{j^{\prime}} A_{l^{\prime}, j^{\prime}}\right) \\
\operatorname{Input}_{l}^{M}=I+\sum_{l^{\prime}<l} M_{i^{\prime}}+\sum_{l^{\prime} \leq i} \sum_{j^{\prime}} A_{l^{\prime}, j^{\prime}}
\end{array}
$$

The computational graph $\mathcal{G}$ of the Transformer represents the interactions between attention heads and MLPs. The nodes in $\mathcal{G}$ encompass the input embedding $I$, attention heads $A_{l, j}$, MLPs $M_{l}$, and the output node $O$, denoted as $N=\left\{I, A_{l, j}, M_{l}, O\right\}$. The edges in the model represent the connections between these nodes, $E=\left\{\left(n_{x}, n_{y}\right), n_{x}, n_{y} \in N\right\}$. A circuit $\mathcal{C}$ is meticulously constructed to govern specific behaviors within the model, comprising a selection of nodes $N_{\mathcal{C}}$ and edges $E_{\mathcal{C}}$ that are critical to the successful execution of the tasks at hand, expressed as $\mathcal{C}=<N_{\mathcal{C}}, E_{\mathcal{C}}>$.

### 2.2 Circuit Discovery

To identify circuits within a language model, a key approach is to examine the model's casual mediation by systematically altering the model's edges and nodes to observe the effects on performance $[32,34,35]$. The underlying principle is that critical edges or nodes are those whose removal
results in a notable decline in the model's predictive capabilities. Since the edges in the model's computational graph represent the dependencies between nodes, we can simulate the absence of a particular node-to-node dependency by ablating an edge in the graph. For example, ablating an edge from $A_{i^{\prime}, j^{\prime}}$ to $A_{i, j}$ involves replacing the contribution of $A_{i^{\prime}, j^{\prime}}$ in the input to attention head $A_{i, j}$ with zero (in the case of zero ablation) or with the mean value of head $A_{i^{\prime}, j^{\prime}}$ (in the case of mean ablation). The process of identifying critical edges or nodes through ablation can be broken down into the following steps: i) Overwrite the value of the edge $\left(n_{x}, n_{y}\right)$ with a corrupted value (either zero or mean ablation), ii) Perform a forward pass through the model with the altered graph, iii) Compare the output values of the modified model with those of the original model using a chosen metric $S$ (Details in Eq. 1 ). If the performance change is below a predefined threshold $\tau$, we can consider the edge non-critical and remove it to obtain a new subgraph $\mathcal{G} /\left(n_{x}, n_{y}\right)$. In addition to ablation-based methods, recent works have also explored the use of sparse auto-encoders [36,37] to identify circuits within language models. This approach involves training an auto-encoder to learn a sparse representation of the model's internal structure, which can help reveal the underlying circuitry responsible for specific behaviors or functionalities.

## 3 Knowledge Circuits Discovery in Transformers

### 3.1 Knowledge Circuits Construction

Unlike previous work $[12,18]$, which managed to find out the specific areas that store knowledge, we pay extra heed to the information flow that activates subsequent knowledge for answering questions. Similar to $[38,26]$, we write language model as a graph consisting of the input, the output, attention heads, and MLPs by considering a "residual rewrite" of the model's computational structure. For example, this residual rewrite gives us a nearly-dense graph in GPT2-medium: one between every pair of (attention head, MLP, input, and output) nodes, except for attention heads in the same layer, which do not communicate with each other. In our paper, we concentrate on the task of answering factual open-domain questions, where the goal is to predict a target entity o given a subject-relation pair $(s, r)$. A knowledge triplet $k=(s, r, o)$ is often presented to the model in the form of a natural language prompt for next token prediction (e.g., "The official language of France is $\qquad$ "). The model $\mathcal{G}$ is expected to generate the target entity, which is consistent with the language model's pretraining format. To identify the circuit that are critical for predicting the target entity $o$ for a given subject-relation pair $(s, r)$, we ablate each special edge $e_{i}=\left(n_{x}, n_{y}\right)$ in the computation graph $\mathcal{G}$. We then measure the impact of ablating the edge (zero ablation in our implementation) on the model's performance using the MatchNLL loss [32] for the target $o$ :

$$
\begin{equation*}
S\left(e_{i}\right)=-\log \left(\mathcal{G} / e_{i}(o \mid(s, r))\right)-\log (\mathcal{G}(o \mid(s, r))) \tag{1}
\end{equation*}
$$

If the score $S\left(e_{i}\right)$ is less than the predefined threshold $\tau$, we consider the edge to be non-critical and remove it from the computation graph, updating the temporary circuit $\mathcal{C}_{\text {temp }} \leftarrow \mathcal{G} / e_{i}$. We first sort the graph by topological rank following Conmy et al. [32] and traverse all edges in this manner, We derive a circuit $\mathcal{C}_{k}$ that contributes to representing the knowledge necessary to answer the factual question:

$$
\begin{equation*}
\mathcal{C}_{k}=<N_{k}, E_{k}> \tag{2}
\end{equation*}
$$

Here, $\mathcal{C}_{k}$ is the circuit for the knowledge triplet $k$, consisting of the nodes $N_{k}$ and edges $E_{k}$ that are essential for predicting the target entity $o$ given the subject-relation pair $(s, r)$.

### 3.2 Knowledge Circuits Information Analysis

Once we have identified the knowledge circuit, we delve deeper into the specific roles and behaviors of each node and edge within the computation graph. Our goal is to comprehend the processing and contribution of each node $n_{i}$ to the functionality of the circuit. Drawing on the methodologies of previous studies $[16,39,40]$, we begin by applying layer normalization to the output of each node $n_{i}$ and then map it into the embedding space. This is achieved by multiplying the layer-normalized output by the unembedding matrix $\left(\mathbf{W}_{U}\right.$ ) of the language model: $\mathbf{W}_{U} \mathrm{LN}\left(n_{i}\right)$. This transformation allows us to inspect how each component writes information to the circuit and how it influences subsequent computational steps. By understanding the nodes' behavior in the circuit, we can better comprehend the circuit's structure and the key points where information is aggregated and disseminated.

Table 1: Hit@10 of the Original and Circuit Standalone performance of knowledge circuit in GPT2Medium. The result for $D_{\text {val }}$ being 1.0 indicates that we select the knowledge for which the model provides the correct answer to build the circuit.

| Type | Knowledge | \#Edge | $D_{v a l}$ |  | $D_{\text {test }}$ |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | $\operatorname{Original}(\mathcal{G})$ | $\operatorname{Circuit}(\mathcal{C})$ | Original $(\mathcal{G})$ | $\operatorname{Circuit}(\mathcal{C})$ |
| Linguistic | Adj Antonym | 573 | 0.80 | $1.00 \uparrow$ | 0.00 | $0.40 \uparrow$ |
|  | world first letter | 432 | 1.00 | 0.88 | 0.36 | 0.16 |
|  | world last letter | 230 | 1.00 | 0.72 | 0.76 | 0.76 |
| Commonsense | object superclass | 102 | 1.00 | 0.68 | 0.64 | 0.52 |
|  | fruit inside color | 433 | 1.00 | 0.20 | 0.93 | 0.13 |
|  | work location | 422 | 1.00 | 0.70 | 0.10 | 0.10 |
| Factual | Capital City | 451 | 1.00 | 1.00 | 0.00 | 0.00 |
|  | Landmark country | 278 | 1.00 | 0.60 | 0.16 | $0.36 \uparrow$ |
|  | Country Language | 329 | 1.00 | 1.00 | 0.16 | $0.75 \uparrow$ |
|  | Person Native Language | 92 | 1.00 | 0.76 | 0.50 | $0.76 \uparrow$ |
| Bias | name religion | 423 | 1.00 | 0.50 | 0.42 | 0.42 |
|  | occupation age | 413 | 1.00 | 1.00 | 1.00 | 1.00 |
|  | occupation gender | 226 | 1.00 | 0.66 | 1.00 | 0.66 |
|  | name birthplace | 276 | 1.00 | 0.57 | 0.07 | $0.57 \uparrow$ |
| Avg |  |  | 0.98 | 0.73 | 0.44 | $0.47 \uparrow$ |

### 3.3 Knowledge Circuits Experimental Settings

Implementations. We conduct experiments on GPT-style models, including GPT-2 medium and large. We also conduct primary experiments on TinyLLaMA [29] to validate the effectiveness of different architectures. We utilize the Automated Circuit Discovery [32] toolkit to build a circuit as an initiative of our analysis and leverage transformer lens [41] to further analyze the results. Specially, we simply employ the MatchNLL [32] as the metric to detect the effect of the given node and edge and use zero ablation to knock out the specific computation node in the model's computation graph.

Metrics. A discovered knowledge circuit is deemed an accurate representation of a specific area within the transformer's knowledge storage, thus, it should be capable of representing the knowledge independently. Following [32], we leverage the completeness of a circuit which refers to its ability to independently reproduce the behavior or predictions of the full model for the relevant tasks. This property is assessed by examining whether the identified subgraph corresponds to the underlying algorithm implemented by the neural network. To evaluate completeness, we first construct the circuit using the validation data $D_{\text {val }}$ for a specific knowledge type and then test its performance on the test split $D_{\text {test }}$ in isolation. By doing so, we can observe any changes in performance compared to the original model. We use the Hit@10 metric to measure the rank of the target entity $o$ among the top 10 predicted tokens:

$$
\begin{equation*}
\text { Hit@10 }=\frac{1}{|V|} \sum_{i=1}^{|V|} \mathrm{I}\left(\mathrm{rank}_{o} \leq 10\right) \tag{3}
\end{equation*}
$$

Here, $|V|$ represents vocabulary size, and $\operatorname{rank}_{o}$ is the rank of the target entity $o$ in predictions.

Dataset. In this work, we focus on the knowledge that already stored in the language model. We utilize the dataset provided by LRE [42] and consider different kinds of knowledge, including linguistic, commonsense, fact, and bias. We evaluate whether the knowledge is present in the language model's parameters under zero-shot settings using the Hit@10 metric to sample knowledge from the validation set, which is used to construct the knowledge circuit. The data statistics are in Appendix A.

## 4 Knowledge Circuits Unveil Implicit Neural Knowledge Representations

Knowledge Circuits Evaluation. We report the results of GPT2-Medium in Table 1, wich indicates that with only less than $10 \%$ of the original knowledge circuit's subgraph, the model can maintain over $70 \%$ of its original performance. One of the most fascinating observations is the performance improvement seen on several test datasets. For instance, the Landmark-country relation metric
increases from 0.16 to 0.36 . This suggests that the discovered knowledge circuits may encapsulate the relevant knowledge, and the model's performance on these tasks could have been hindered by noise from other components. We proceed to analyze the layer distribution of the original model $\mathcal{G}$ to understand the average percentage of nodes that are activated within the circuit for different knowledge domains. From Figure 2, we observe that attention and MLPs are more active in the lower layers of the network, where the language model processes the input and extracts general information. To gain a more comprehensive view of the information processing, we compute the average rank $_{o}$ change of the target token in the $D_{v a l}$ across the layers and report the results in Figure 6. This analysis reveals the phenomenon of early decoding [40], suggesting that by the middle to the latest layers, the target entity is already present in the residual stream, and the subsequent layers in the Transformer are designed to increase the probability of the current token (See discussion in the running example).
![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-06.jpg?height=566&width=586&top_left_y=693&top_left_x=368)

Bias
![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-06.jpg?height=282&width=550&top_left_y=972&top_left_x=386)

Figure 2: The activated circuit component distributions in Layers in GPT2-Medium.
Special Components in Knowledge Circuits. From the discovered knowledge circuits, we can find several important attention heads that demonstrate specific behavior, including the mover head [31], relation head $[17,43]$ and mixture head $[17,43]$ (Definitions in Appendix B.2). We think that these components would be accumulated by the MLP in the model and discuss the behavior of these special heads in the running example part. We list some of these special components in Table 3 in Appendix. The different attention heads are responsible for expressing specific types of knowledge and may be activated by different facts. In our experiments with GPT-2 Medium and GPT-2 Large, we find that knowledge is distributed across several layers' attention heads and MLP matrices, suggesting that the target knowledge appears to have been accumulated throughout the GPT-2 model. Conversely, in TinyLLAMA, the special components are more concentrated. As depicted in Figure 6, the rank of the target entity in TinyLLAMA experiences a sharp decline around several layers, whereas in the GPT2 model, the decline is more gradual. We hypothesize that this discrepancy may be attributed to the model's knowledge capacity [44] and warrants further investigation.

A Running Example of Knowledge Circuit. We present a case and analyze the specific behaviors of components within the identified knowledge circuits. Taking the factual knowledge "The official language of France is French" as an example, we visualize the knowledge circuit in Figure 1. To express the information flow within the model more effectively, we have plotted the rank and probability of the target entity o at each layer when it is mapped into the embedding space, in Figure 3. After MLP 17, the target knowledge emerges as the top token in the residual stream and after that layer, it undergoes an increased probability. The edges connected to MLP17 are (L14H13 $\rightarrow$ MLP17), (L14H7 $\rightarrow$ MLP17), and (L15H0 $\rightarrow$ MLP17) . Here, the L14H13 is a relation head that focuses on the relation token in the context. The output of this head is relation-related tokens such as "language" and "Language". The attention head $\mathrm{L} 14 \mathrm{H} 7$ is a mover head that moves the information from the subject position "France" to the last token. Previous work $[31,19]$ has introduced this mover head as an argument parser, which moves "France" to the last token, and the subsquent MLP conducts a function application to map "France" to "French". An intriguing observation is that we can find the output of this head already contains the target entity, which significantly contributes to the final output (L14H7 $\rightarrow$ Output). Also, we see the get entity $o$ when unembedding the intermediate layer's output for the fact "The official language of France is French". For a more comprehensive view, we additionally plot the probability of the subject entity at the last token position and the target entity's rank at the subject position.

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-06.jpg?height=392&width=646&top_left_y=1614&top_left_x=1119)

Figure 3: The rank and probability of the tar-
![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-07.jpg?height=970&width=1324&top_left_y=239&top_left_x=408)

Case Output
![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-07.jpg?height=280&width=546&top_left_y=586&top_left_x=430)

## Edit Case:

Platform Controller Hub is created by Other Case: Windows Server 2003 is created by

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-07.jpg?height=276&width=479&top_left_y=301&top_left_x=1235)
Intel. $\quad$

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-07.jpg?height=336&width=1282&top_left_y=867&top_left_x=432)

Figure 4: Different behaviors when we edit the language model. In the original model, we can see the mover head L15H3 actually move the original token "Controller" and other information, while for ROME, we observe the mover head select the correct information "Intel", which means ROME successfully added the "Intel" to model. For the FT layer-0 editing, we can find this method directly write the edited knowledge into edited component. However, we find these two editing methods would affect other unrelated input "Windows server is created by?"

probability of the subject token at the last token is nearly zero across these layers. Hence, instead of the argument parser, we consider this mover head as an extract head proposed by Geva et al. [13], which aims to extract the related-information from the subject token's position. In the subsequent knowledge editing experiments, we can observe changes in the behavior of these types of heads. Instead of extraction in the later layers proposed by Geva et al. [13], we notice a gradual decrease in rank across all early-to-middle layers. The MLP17 combines information from previous tokens and integrates this information to prioritize the target token at the top rank.

Interestingly, upon tracing the information flow to $\mathrm{L} 14 \mathrm{H} 7$, we discovered that it is predominantly activated by $\mathrm{L} 7 \mathrm{H} 14$, a relation head, and its output features several language tokens, such as "Arabic". We hypothesize that $\mathrm{L} 7 \mathrm{H} 14$ may function as a signaling mechanism to activate the associated mover head, but this hypothesis necessitates further investigation to be confirmed. After MLP17, several attention heads, such as L18H14 (a relation head) and L20H6 (a mover head), collaborated to further enhance the final prediction of the target entity.

## 5 Knowledge Circuits Elucidate Internal Mechanisms for Knowledge Editing

In this section, our objective is to evaluate the impact of previous knowledge editing methods (why fail in certain cases and settings) and validate the effectiveness of knowledge circuits.

Single Factual Knowledge Editing. Here, we adopt the ROME method [18] and FT-M [24], which aim to edit the MLP layers in the language model. The most important hyper-parameter in knowledge editing is the layer, as the same method's performance varies significantly via the layers. Here, we evaluate the performance of different editing layers and their effectiveness. We compare the knowledge circuits computed by the edited model with the original one, and we present results in Figure 4 and report details in Appendix D. As discussed in the previous part, the early-to-middle
layers are the main part of aggregating the target entity $o$ to the top rank. In the original model, the probability of the target entity "Intel" is nearly zero, and the model fails to elevate it to the top rank in the vocabulary. Editing layer 0 with ROME and FT-M both give us the correct answer but we can view different scenarios for their knowledge circuits. For ROME, as the correct information is added to the subject position, we can recognize a behavior of the Mover Head shifts from copying to extracting the edited information from the subject position. This information gradually aggregates through the subsequent layers, and by layer 15, "Intel" emerges as the top-ranked entity, with its probability increasing significantly. Specially, before editing, the mover head $\mathrm{L} 15 \mathrm{H} 3$ attends to the "controller" token and returns "controller" as the output, while in the edited model, the attention head's output moves to the "Intel", which means the model gains the information at the subject space. While for FT-M, the edited model tends to directly write the knowledge into the specific component, which would greatly dominate the following component in the model. As shown in Figure 4, the output logits in MLP-0 for "Intel" are more than 10, and it emerges as the top rank in the residual stream directly. This phenomenon can be found in different knowledge types and layers and we report results in Appendix D.2. However, the added knowledge may have the risk to influence unrelated knowledge. When we test another fact "Windows server", the model still tends to give us the "Intel" answer, demonstrating the overfitting problem. This finding supports previous analysis regarding the correlation between localization and editing [45], suggesting that edits may not alter the storage but merely add signals into the knowledge circuits.

Multi-hop Factual Knowledge Editing. Multi-hop knowledge editing poses a challenging scenario [20, 21, 46], wherein we edit the model with new knowledge, yet the model struggles to perform reasoning using the edited information. We analyze multi-hop questions in language models [47, 48] to understand why current editing methods fail in these scenarios. For instance, given the fact (Thierry Mugle, "home country", France), we edit the fact to another country, such as (Thierry Mugle, "home country", France $\rightarrow$ China). We then assess the model's performance on questions based on the edited knowledge, including "The official currency of the home country of Thierry Mugle is" and "The capital city of the home country of Thierry Mugle is". While the unedited model could correctly answer these questions, we observe that the edited model would provide the answer "China" for subsequent hop reasoning. We find that the mover head in the original multi-hop reasoning circuit initially extracts the second-hop answer but, after editing, extracts "China", demonstrating that the edited information dominantly saturates and influences the circuit. Furthermore, we observe an intriguing phenomenon: even in the original model's multi-hop reasoning settings, it would directly provide the answer if we remove the context of the first-hop texts (Details in Appendix C.1). This further confirms the findings that the model relies on relational and subject-related information, regardless of grammatical adherence.

## 6 Knowledge Circuits Facilitate Interpreting Language Model Behaviors

In this Section, our aim is to validate whether the identified knowledge circuits are actually utilized by the model when it employs knowledge. To address this, as shown in Figure 5, we investigate three phenomena: hallucination, in-context learning, and reverse relations (Details in Appendix C.3).

Factual Hallucination. We focus on factual hallucinations, which occur when the model provides an incorrect target entity for a given subject $s$ and relation $r$. In our experiments (Figure 5 and Appendix C.2), we observe that the model fails to move the correct knowledge to the final token in the earlier layers. This failure is evident as the circuit lacks an effective mover head or the mover head selects incorrect information. For instance, in the prompt "The official currency of Malaysia is called", both the correct answer "Ringgit" and the incorrect one "Malaysian" are accumulated before layer 15. However, at layer 16, the mover head L15H10 extracts the erroneous information. Despite a rank drop of the true one in layers 20-22, this is insufficient to correct the previous mistake.

In-Context Learning. Despite storing vast amounts of knowledge, a language model may still provide incorrect answers. However, with demonstrations or examples (based on RAG [49]), it can quickly generate correct responses. To this end, we focus on the scenario where the model initially provides an incorrect answer but can then produce the correct response upon receiving the appropriate demonstration. We consider the original knowledge circuit and introduce a new knowledge circuit based on the demonstration. Our analysis reveals that, compared to the zero-shot knowledge circuit, several new attention heads appear in the computation graph when the demonstration is incorporated

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-09.jpg?height=729&width=637&top_left_y=243&top_left_x=386)

(a) A hallucination Case

![](https://cdn.mathpix.com/cropped/2024_06_04_646a3dc36a33a089bb45g-09.jpg?height=734&width=700&top_left_y=240&top_left_x=1035)

(b) An in-context learning Case

Figure 5: Left: fact hallucination case "The official currency of Malaysia is called", we observe that, at layer 15, the Mover Head selects incorrect information. Right: In-context learning case, we notice that some new heads focusing on the demonstration appear in the knowledge circuit.

as shown in Figure 5. These heads mainly focus on the demonstration's context: "The co mparative form of small is smaller". Concretely, Todd et al. [50] have identified a concept known as the Function Vector, which represents the average of some key attention heads. In our experiments, we found that these attention heads primarily focus on the demonstration, indicating their role in leveraging the demonstration to reactivate and correct the model's response.

## 7 Related Work

Knowledge Mechanism of Transformers. How the language model store and utilize knowledge is an ongoing research topic. Previous works find that the MLP in Transformers works as a keyvalue memory and stores enormous knowledge $[12,15,14,18]$. As to the relation between entities, Hernandez et al. [42] observe that facts can be decoded linearly from the enriched residual stream of the subject by mapping the subject entity to the object entity. Instead of viewing the knowledge storage in isolation, Geva et al. [13], Lv et al. [31], Yu and Ananiadou [16] find the knowledge is accumulated during the layers. Regarding knowledge analysis, Bayazit et al. [51] also attempt to discover critical knowledge in language models. However, they only consider several layers in the model and use the pruning method, which may overlook the connections between components. More related works can be found in Appendix E.1.

Manipulate Language Models. Recently, many works aim to manipulate the language models to make the model aligned with world knowledge or social value norms, such as knowledge editing [20, 24], machine unlearning [52, 53] and detoxification [54, 55]. Most of these works are elicited by previous knowledge mechanism findings such as knowledge neuron [56]. They modify the MLP in the LLM $[18,12]$ to change the model's behavior based on specific factual knowledge. However, recent works $[57,58]$ demonstrate the pivotal role of the attention part in knowledge representation. Hase et al. [45] also observe that the performance of editing within a layer may not reliably pinpoint the location of the fact. In this paper, we try to manipulate specific knowledge of language model via knowledge circuit, including both MLP and attention components across different layers.

## 8 Conclusion

In this paper, we present a new perspective on knowledge storage based on circuit theory and conduct a preliminary analysis to demonstrate its effectiveness. We hope these findings can advance our
understanding of the knowledge mechanisms of language models and provide insights for better designing and editing language models, enhancing knowledge, and improving reasoning to enhance factuality and alleviate hallucinations.

## Limitations and Broader Impacts

Current circuit discovery-based patching method is time-consuming, there are some concurrent works that propose more efficient way [59] to build the model's information flow. Also, there are some other methods to discover circuits, like acdcpp [60] and Sparse Auto-Encoders [61, 36]. We believe that knowledge circuit discovery has a huge room for improvement. Additionally, by focusing on linguistic, factual, commonsense, and bias-related knowledge, we believe our approach can be applied to ensure safety and privacy information to promote trustworthy AI.

## References

[1] Francis bacon. https://iep.utm.edu/francis-bacon/.

[2] Francis Bacon. The advancement of learning [1605]. In Primer of intellectual freedom, pages 172-192. Harvard University Press, 1949.

[3] OpenAI and the Co-authors. Gpt-4 technical report, 2024.

[4] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. CoRR, abs/2303.18223, 2023. doi: 10.48550/ARXIV.2303.18223. URL https://doi.org/10.48550/arXiv.2303.18223.

[5] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions, 2023.

[6] Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, and Yue Zhang. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity, 2023.

[7] Xiang Chen, Chenxi Wang, Yida Xue, Ningyu Zhang, Xiaoyan Yang, Qiang Li, Yue Shen, Lei Liang, Jinjie Gu, and Huajun Chen. Unified hallucination detection for multimodal large language models. CoRR, abs/2402.03190, 2024. doi: 10.48550/ARXIV.2402.03190. URL https://doi.org/10.48550/arXiv. 2402.03190.

[8] Helena Bonaldi, Yi-Ling Chung, Gavin Abercrombie, and Marco Guerini. Nlp for counterspeech against hate: A survey and how-to guide, 2024.

[9] Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric P. Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, William Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, and Yue Zhao. Trustllm: Trustworthiness in large language models. CoRR, abs/2401.05561, 2024. doi: 10.48550/ARXIV.2401.05561. URL https://doi.org/10.48550/arXiv. 2401.05561.

[10] Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. Safetybench: Evaluating the safety of large language models with multiple choice questions. CoRR, abs/2309.07045, 2023. doi: 10.48550/ARXIV. 2309.07045. URL https://doi.org/10.48550/arXiv.2309.07045.

[11] Aiqi Jiang and Arkaitz Zubiaga. Cross-lingual offensive language detection: A systematic review of datasets, transfer approaches and challenges, 2024.

[12] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493-8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581.

[13] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216-12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.751.

[14] Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30-45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3.

[15] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484-5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main. 446.

[16] Zeping Yu and Sophia Ananiadou. Locating factual knowledge in large language models: Exploring the residual stream and analyzing subvalues in vocabulary space, 2024.

[17] Bilal Chughtai, Alan Cooney, and Neel Nanda. Summing up the facts: Additive mechanisms behind factual recall in llms, 2024.

[18] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 36, 2022.

[19] Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Language models implement simple word2vec-style vector arithmetic, 2024.

[20] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and opportunities. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10222-10240, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 632.

[21] Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. Evaluating the Ripple Effects of Knowledge Editing in Language Models. Transactions of the Association for Computational Linguistics, 12:283-298, 04 2024. ISSN 2307-387X. doi: 10.1162/tacl_a_00644. URL https://doi.org/10.1162/tacl_a_00644.

[22] Belinda Hopkins. Restorative theory in practice: Insights into what works and why. Jessica Kingsley Publishers, 2015.

[23] Jingcheng Niu, Andrew Liu, Zining Zhu, and Gerald Penn. What does the knowledge neuron thesis have to do with knowledge? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=2HJRwwbV3G.

[24] Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, et al. A comprehensive study of knowledge editing for large language models. arXiv preprint arXiv:2401.01286, 2024.

[25] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html.

[26] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=NpsVSN6o4ul.

[27] Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Circuit component reuse across tasks in transformer language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=fpoAYV6Wsk.

[28] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[29] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024.

[30] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.

[31] Ang Lv, Kaiyi Zhang, Yuhan Chen, Yulong Wang, Lifeng Liu, Ji-Rong Wen, Jian Xie, and Rui Yan. Interpreting key mechanisms of factual recall in transformer-based language models, 2024.

[32] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 16318-16352. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/ file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf.

[33] Leonard Bereska and Efstratios Gavves. Mechanistic interpretability for ai safety - a review, 2024.

[34] Judea Pearl. Direct and indirect effects. In Probabilistic and causal inference: the works of Judea Pearl, pages 373-392. 2022.

[35] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388-12401, 2020.

[36] Zhengfu He, Xuyang Ge, Qiong Tang, Tianxiang Sun, Qinyuan Cheng, and Xipeng Qiu. Dictionary learning improves patch-free circuit discovery in mechanistic interpretability: A case study on othello-gpt, 2024.

[37] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.

[38] Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. Localizing model behavior with path patching, 2023.

[39] Shahar Katz and Yonatan Belinkov. VISIT: Visualizing and interpreting the semantic information flow of transformers. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14094-14113, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-emnlp.939. URL https://aclanthology.org/2023.findings-emnlp. 939.

[40] nostalgebraist. interpreting GPT: the logit lens. 2020. URL https://www. lesswrong.com/ posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

[41] Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/neelnanda-io/ TransformerLens, 2022.

[42] Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, and David Bau. Linearity of relation decoding in transformer language models. In Proceedings of the 2024 International Conference on Learning Representations, 2024.

[43] Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and Marta R. Costa-jussà. A primer on the inner workings of transformer-based language models, 2024.

[44] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. 2024.

[45] Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=EldbUlZtbd.

[46] Zexuan Zhong, Zhengxuan Wu, Christopher Manning, Christopher Potts, and Danqi Chen. MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15686-15702, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 971 .

[47] Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, and Sebastian Riedel. Do large language models latently perform multi-hop reasoning?, 2024.

[48] Tianjie Ju, Yijin Chen, Xinwei Yuan, Zhuosheng Zhang, Wei Du, Yubin Zheng, and Gongshen Liu. Investigating multi-hop factual shortcuts in knowledge editing of large language models, 2024.

[49] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. CoRR, abs/2312.10997, 2023. doi: 10.48550/ARXIV.2312.10997. URL https://doi.org/10.48550/arXiv.2312.10997.

[50] Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau. LLMs represent contextual tasks as compact function vectors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= AwyxtyMwaG.

[51] Deniz Bayazit, Negar Foroutan, Zeming Chen, Gail Weiss, and Antoine Bosselut. Discovering knowledge-critical subnetworks in pretrained language models, 2023.

[52] Nianwen Si, Hao Zhang, Heyu Chang, Wenlin Zhang, Dan Qu, and Weiqiang Zhang. Knowledge unlearning for llms: Tasks, methods, and challenges, 2023.

[53] Jiaao Chen and Diyi Yang. Unlearn what you want to forget: Efficient unlearning for LLMs. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12041-12052, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 738 . URL https://aclanthology.org/2023.emnlp-main. 738.

[54] Xinshuo Hu, Dongfang Li, Baotian Hu, Zihao Zheng, Zhenyu Liu, and Min Zhang. Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, ThirtyEighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 18252-18260. AAAI Press, 2024. doi: 10.1609/AAAI.V38I16.29784. URL https://doi.org/10.1609/aaai.v38i16.29784.

[55] Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, and Huajun Chen. Detoxifying large language models via knowledge editing, 2024.

[56] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8493-8502. Association for Computational Linguistics, 2022.

[57] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inferencetime intervention: Eliciting truthful answers from a language model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/ forum?id=aLLuYpn83y.

[58] Mansi Sakarvadia, Aswathy Ajith, Arham Khan, Daniel Grzenda, Nathaniel Hudson, André Bauer, Kyle Chard, and Ian Foster. Memory injections: Correcting multi-hop reasoning failures during inference in transformer-based language models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi, editors, Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 342-356, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.blackboxnlp-1.26.

[59] Javier Ferrando and Elena Voita. Information flow routes: Automatically interpreting language models at scale. arXiv preprint arXiv:2403.00824, 2024.

[60] Aaquib Syed, Can Rager, and Arthur Conmy. Attribution patching outperforms automated circuit discovery. In NeurIPS Workshop on Attributing Model Behavior at Scale, 2023. URL https://openreview.net/forum?id=tiLbFR4bJW.

[61] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.

[62] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895-4901, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.298. URL https://aclanthology.org/2023.emnlp-main. 298.

[63] Shiqi Chen, Miao Xiong, Junteng Liu, Zhengxuan Wu, Teng Xiao, Siyang Gao, and Junxian He. In-context sharpness as alerts: An inner representation perspective for hallucination mitigation, 2024.

[64] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R. Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=Th6NyL07na.

[65] Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al. Layer skip: Enabling early exit inference and self-speculative decoding. arXiv preprint arXiv:2404.16710, 2024.

[66] Beren and Sid Black. The singular value decompositions of transformer weight matrices are highly interpretable. https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/ the-singular-value-decompositions-of-transformer-weight, 2022.

[67] Junlin Zhang. Parametric reflection of the world: Why can gpt generate intelligence through next token prediction. https://zhuanlan.zhihu.com/p/632795115, 2023.

[68] Lukas Berglund, Meg Tong, Maximilian Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: LLMs trained on "a is b" fail to learn "b is a". In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=GPKTIktA0k.

[69] Peng Wang, Ningyu Zhang, Xin Xie, Yunzhi Yao, Bozhong Tian, Mengru Wang, Zekun Xi, Siyuan Cheng, Kangwei Liu, Guozhou Zheng, et al. Easyedit: An easy-to-use knowledge editing framework for large language models. arXiv preprint arXiv:2308.07269, 2023.

[70] Maximilian Li, Xander Davies, and Max Nadeau. Circuit breaking: Removing model behaviors with targeted ablation. Workshop on Challenges in Deployable Generative AI at International Conference on Machine Learning, 2023.

[71] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 5484-5495. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.446. URL https://doi.org/10.18653/v1/2021. emnlp-main. 446 .

[72] Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 30-45. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.3. URL https: //doi.org/10.18653/v1/2022.emnlp-main. 3.

[73] Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Journey to the center of the knowledge neurons: Discoveries of language-independent knowledge neurons and degenerate knowledge neurons. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 17817-17825. AAAI Press, 2024. doi: 10.1609/AAAI.V38I16.29735. URL https://doi.org/10.1609/aaai.v38i16. 29735.

[74] Yuheng Chen, Pengfei Cao, Yubo Chen, Yining Wang, Shengping Liu, Kang Liu, and Jun Zhao. The da vinci code of large pre-trained language models: Deciphering degenerate knowledge neurons. CoRR, abs/2402.13731, 2024. doi: 10.48550/ARXIV.2402.13731. URL https://doi.org/10.48550/arXiv.2402.13731.

[75] Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Massediting memory in a transformer. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=MkbcAHIYgyS.

[76] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality, 2024.

[77] Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. CoRR, abs/2402.18154, 2024. doi: 10.48550/ARXIV.2402.18154. URL https://doi.org/10.48550/arXiv.2402.18154.

[78] Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. CoRR, abs/2310.15213, 2023. doi: 10.48550/ ARXIV.2310.15213. URL https://doi.org/10.48550/arXiv.2310.15213.

[79] Subhabrata Dutta, Joykirat Singh, Soumen Chakrabarti, and Tanmoy Chakraborty. How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning. CoRR, abs/2402.18312, 2024. doi: 10.48550/ARXIV.2402.18312. URL https://doi.org/10. 48550/arXiv. 2402.18312 .

[80] nostalgebraist. interpreting gpt: the logit lens. https://www.lesswrong.com/posts/ AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens, 2020.

[81] Mert Yuksekgonul, Varun Chandrasekaran, Erik Jones, Suriya Gunasekar, Ranjita Naik, Hamid Palangi, Ece Kamar, and Besmira Nushi. Attention satisfies: A constraint-satisfaction lens on factual errors of language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=gfFVATffPd.

[82] Fahim Dalvi, Hassan Sajjad, and Nadir Durrani. Neurox library for neuron analysis of deep NLP models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: System Demonstrations, ACL 2023, Toronto, Canada, July 10-12, 2023, pages 226-234. Association for Computational Linguistics, 2023.

[83] Dan Mossing, Steven Bills, Henk Tillman, Tom Dupré la Tour, Nick Cammarata, Leo Gao, Joshua Achiam, Catherine Yeh, Jan Leike, Jeff Wu, and William Saunders. Transformer debugger. https://github.com/openai/transformer-debugger, 2024.

[84] Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, and Mor Geva. Patchscopes: A unifying framework for inspecting hidden representations of language models, 2024.

[85] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. CoRR, abs/2209.10652, 2022. doi: 10.48550/ARXIV.2209.10652. URL https://doi.org/10.48550/arXiv.2209.10652.
</end of paper 2>


<paper 3>
# MEMoE: Enhancing Model Editing with Mixture of Experts Adaptors 

Renzhi Wang ${ }^{1,2}$, Piji Li $^{1,2 *}$<br>${ }^{1}$ College of Computer Science and Technology,<br>Nanjing University of Aeronautics and Astronautics, China<br>${ }^{2}$ MIIT Key Laboratory of Pattern Analysis and Machine Intelligence, Nanjing, China<br>${ }^{1}\{r z h w a n g$, pjli\}@nuaa.edu.cn


#### Abstract

Model editing aims to efficiently alter the behavior of Large Language Models (LLMs) within a desired scope, while ensuring no adverse impact on other inputs. Recent years have witnessed various model editing methods been proposed. However, these methods either exhibit poor overall performance or struggle to strike a balance between generalization and locality. We propose MEMoE, a model editing adapter utilizing a Mixture of Experts (MoE) architecture with a knowledge anchor routing strategy. MEMoE updates knowledge using a bypass MoE structure, keeping the original parameters unchanged to preserve the general ability of LLMs. And, the knowledge anchor routing ensures that inputs requiring similar knowledge are routed to the same expert, thereby enhancing the generalization of the updated knowledge. Experimental results show the superiority of our approach over both batch editing and sequential batch editing tasks, exhibiting exceptional overall performance alongside outstanding balance between generalization and locality. Our code will be available.


## 1 Introduction

Large Language Models [38, 47, 48] learn a vast repository of world knowledge during pre-training, which can be accessed and utilized through natural language prompts [39]. Despite this extensive base of information, the dynamic nature of the real-world demands regular updates to these models to correct outdated information or integrate new knowledge [51, 52]. However, frequently retraining or fine-tuning the LLMs to incorporate these updates is often impractical, given the substantial resources and time required [29, 52].

To address this, the concept of model editing, also known as knowledge editing, has been introduced [54]. This approach aims to efficiently modify the outputs of LLMs for target queries while preserving the overall performance for other unrelated inputs. Recent years have witnessed significant efforts in developing model editing techniques, with numerous methods proposed in various editing tasks and settings. For instance, specific approaches such as ROME [33] for single knowledge editing, MEMIT [34] for batch editing, and GRACE [16] for sequential editing have been introduced. Currently, evaluation for model editing revolves three dimensions: reliability, generality, and locality [51, 54]. To illustrate, suppose the original model predicts "Trump" for the input "Who is the president of the United States?" and the desired post-edit model prediction is "Joe Biden". To assess reliability, the same original statement is used as input to verify whether the post-edit model predicts "Joe Biden" as intended. For generality, a paraphrased statement like "Who currently holds the position of the U.S. presidency?" can be presented to the edited model to ensure consistent output modification to "Joe[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-02.jpg?height=546&width=1290&top_left_y=236&top_left_x=426)

Figure 1: Left: Apart from our MEMoE, no method achieves both high accuracy and high balance. Right: Significant room for improvement in the overall performance of current methods.

Biden". Locality implies that the model output for an unrelated statement such as "What is the capital of the United States?" should remain unaffected. As illustrated in Figure 1 there is a significant disparity between the generality and locality scores of current methods, and their overall performance is suboptimal. None of the existing approaches have succeeded in simultaneously achieving high accuracy and high balance. This underscores the challenge of striking a balance between locality and generality in model editing and reveals ample opportunity for enhancing the overall performance.

In light of these, we propose MEMoE, a novel framework leveraging the MoE architecture alongside knowledge anchor routing to enhance the overall performance of model editing. Considering the intrinsic sparsity of knowledge information and the advantage of MoE in handling sparse features [3, 45], MEMoE extends upon a parallel MoE structure to enhance the accuracy of knowledge learning. This MoE-style adapter is confined to only one layer of the model, preserving all original parameters, thereby enhancing the locality of model editing and further reducing the impact on model's general ability. On the other hand, prior research has highlighted the considerable benefits stemming from the specialize feature of MoE's experts in multi-task learning, and pointed out that appropriate routing strategy can lead to improved generalization performance [57, 50]. We introduce the knowledge anchor routing strategy, enabling routers to selectively focus on specific knowledge aspects of the inputs, ensuring that queries requiring similar knowledge are routed to the same experts. Through this "professional people do professional things" approach, the generalization performance of MEMoE has been improved.

The main contributions of our work can be summarized as follows:

- We present MEMoE, a method utilizing a MoE architecture with a knowledge anchor routing strategy for model editing.
- Experiments show that our proposed method achieves state-of-the-art editing performance. MEMoE achieves high accuracy while effectively balancing generality and locality.
- We conducted further experiments to confirm that this method has minimal impact on model's general ability and present detailed analysis on various model settings.


## 2 Preliminaries of Model Editing

Based on the prior works [51, 54, 29], the task of model editing involves effectively modify an initial base model $f_{\theta}$ ( $\theta$ represents the model's parameters) into an edited model $f_{\theta^{\prime}}$. The goal is to adjust the model's responses to a set of specified edit instances as desired, while preserving its behavior on all other instances [29]. The intended edit descriptor is denoted as $\left\{\left(x_{i}^{e}, y_{i}^{e}\right)\right\}_{i \in[1, N]}$, where $f_{\theta}\left(x_{i}^{e}\right) \neq y_{i}^{e}$. This set of intended instances is referred to as the editing scope $I_{\text {edit }}$, while the out-of-s cope $O_{e d i t}$ refers to inputs set that are not relevant to the editing examples. Formally, a
successful edit can be expressed as:

$$
f_{\theta^{\prime}}\left(x_{i}\right)= \begin{cases}y_{i}^{e} & \text { if } x_{i} \in I_{e d i t}  \tag{1}\\ f_{\theta}\left(x_{i}\right) & \text { if } x_{i} \in O_{e d i t}\end{cases}
$$

Problem settings for model editing usually fall into four categories [51, 29]:

1) Single Editing assesses model performance after a single knowledge update.:

$$
\begin{equation*}
\theta^{\prime} \leftarrow \underset{\theta}{\operatorname{argmin}}\left(\left\|f_{\theta}\left(x_{i}^{e}\right)-y_{i}^{e}\right\|\right) \tag{2}
\end{equation*}
$$

2) Batch Editing assesses model performance when multiple knowledge pieces are modified simultaneously ( $n \leq N$ represents the batch size):

$$
\begin{equation*}
\theta^{\prime} \leftarrow \underset{\theta}{\operatorname{argmin}} \sum_{i=1}^{n}\left(\left\|f_{\theta}\left(x_{i}^{e}\right)-y_{i}^{e}\right\|\right) \tag{3}
\end{equation*}
$$

3) Sequential Editing requires that every single edit is executed successively and evaluation conducted only after all edits are completed [16]:

$$
\begin{equation*}
\theta^{\prime} \leftarrow \underset{\theta}{\operatorname{argmin}} \sum_{i=1}^{N}\left(\left\|f_{\theta}\left(x_{i}^{e}\right)-y_{i}^{e}\right\|\right) \tag{4}
\end{equation*}
$$

4) Sequential Batch Editing aims to perform edits in a sequential manner and in batches ( $n$ represents the batch size, $S$ represents the sequential editing step):

$$
\begin{equation*}
\theta^{\prime} \leftarrow \underset{\theta}{\operatorname{argmin}} \sum_{s=0}^{S} \sum_{i=s \times n}^{(s+1) \times n}\left(\left\|f_{\theta}\left(x_{i}^{e}\right)-y_{i}^{e}\right\|\right) \tag{5}
\end{equation*}
$$

Based on the above settings, a successful model editor should meet requirements of the following three properties: Reliability, Generality, and Locality [51]. Formally, these can be expressed as [54]:

1) Reliability measures the average accuracy of the post-edit model $f_{\theta^{\prime}}$ on intended edits:

$$
\begin{equation*}
\mathbb{E}_{\left(x_{i}^{e}, y_{i}^{e}\right) \sim I_{\text {edit }}} \mathbb{1}\left\{\operatorname{argmax}_{y} f_{\theta^{\prime}}\left(y \mid x_{i}^{e}\right)=y_{i}^{e}\right\} \tag{6}
\end{equation*}
$$

2) Generality measures the average accuracy of the model $f_{\theta^{\prime}}$ on examples drawn uniformly from the equivalence neighborhood $N_{\text {edit }}$ which includes input/output pairs related to $I_{\text {edit }}$ :

$$
\begin{equation*}
\mathbb{E}_{\left(x_{i}, y_{i}^{e}\right) \sim N_{e d i t}} \mathbb{1}\left\{\operatorname{argmax}_{y} f_{\theta^{\prime}}\left(y \mid x_{i}\right)=y_{i}^{e}\right\} \tag{7}
\end{equation*}
$$

3) Locality is evaluated by the rate at which the predictions of the post-edit model $f_{\theta^{\prime}}$ remain unchanged compared to the pre-edit model $f_{\theta}$ :

$$
\begin{equation*}
\mathbb{E}_{\left(x_{i}, y_{i}\right) \sim O_{e d i t}} \mathbb{1}\left\{f_{\theta^{\prime}}\left(y \mid x_{i}\right)=f_{\theta}\left(y \mid x_{i}\right)\right\} \tag{8}
\end{equation*}
$$

## 3 Methodology

In this section, we provide a detailed introduction to MEMoE, a model editing adapter based on MoE structure and knowledge anchor routing strategy, as shown in Figure 2 This method achieves a balance between generality and locality, while enabling highly precise model editing.

### 3.1 MEMoE Architecture

One of the core ideas of MEMoE is to introduce several MOE-style experts via bypasses to facilitate knowledge updates and learning, while freezing all the original parameters of LLM to maintain its general ability to the greatest extent. The right of Figure 2 illustrates the forward process of MEMoE, sharing both similarities and distinctions with traditional MoE depicted on the left.

Similar to traditional MoE, MEMoE employs a structure that integrates multiple parallel experts within the transformer feed-forward network (FFN). The choice to use the FFN module is not only

![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-04.jpg?height=664&width=1374&top_left_y=248&top_left_x=381)

Figure 2: The architecture of MEMoE, compared with conventional MoE. Same color denote inputs requiring same knowledge. Pentagrams symbolize knowledge anchors within the input sentences, while squares and triangles represent ordinary input tokens during editing process and generality evaluation respectively. The distribution of tokens within the FFN illustrates that knowledge anchor consolidate inputs requiring same knowledge to the same experts.

due to its traditional role in $\mathrm{MoE}$ [20] but also aligns with recent experimental findings of knowledge probing technologies that the MLP layers within FFN store knowledge [7, 33, 34].

Differently, MEMoE incorporate the additional parallel experts through a bypass mechanism, thereby preserving the original parameters of the model to enhance the locality of model editing. This bypass structure also provides potential to further enhance the generalization performance of model editing (more details in $\$ 3.2$. Moreover, this adaptation is applied to only one layer of the model. The selective addition of adapters in one layer is based on two considerations: first, previous model editing techniques have demonstrated that modifying parameters in a single layer can effectively achieve knowledge updates [33, 34]; second, this strategy further maintains the structure and parameters of the original model, ensuring its general ability is preserved to the greatest extent.

Specifically, given token $x_{i}$ in the input sequence $X=\left\{x_{i}\right\}_{i=1}^{L}$, MEMoE with $E$ experts first introduces a gate decision vector $\mathcal{G} \in \mathbb{R}^{E}$ that dispatches different input tokens to different experts, which is calculated as:

$$
\begin{equation*}
\mathcal{G}=\operatorname{top}_{k}\left(\operatorname{softmax}\left(\mathbf{W}_{g} \cdot R\left(x_{i}\right)+\epsilon\right)\right) \tag{9}
\end{equation*}
$$

where $R(\cdot)$ defines a routing strategy for gate decision (more details in $\$ 3.2), \mathbf{W}_{g}$ is the trainable weights in gate decision, while $\epsilon$ denotes the noise term. The top ${ }_{k}(\cdot)$ operator zeros out all but the top- $k$ values. After getting the gate decision vector $\mathcal{G}$, the corresponding output $h_{i}$ is generated through a weighted aggregation of each expert's computation on $x_{i}$, as follows:

$$
\begin{equation*}
h_{i}=\sum_{e=1}^{E} \mathcal{G}_{e} \cdot \mathbf{W}_{e} \cdot x_{i} \tag{10}
\end{equation*}
$$

where $\mathbf{W}_{\mathbf{e}}$ is the linear projection weights of the $e$-th expert and gate decision $\mathcal{G}_{e}$ determines how much the $e$-th expert contributes to the output $h_{i}$. Note that, experts with $\mathcal{G}_{e}=0$ does not need to be computed for saving computation.

Overall, the forward process of the MEMoE layer, combined with the frozen original parameters $\mathbf{W}_{0}$, can be expressed as:

$$
\begin{equation*}
h_{i}=\mathbf{W}_{0} \cdot x_{i}+\lambda \sum_{e=1}^{E} \mathcal{G}_{e} \cdot \mathbf{W}_{e} \cdot x_{i} \tag{11}
\end{equation*}
$$

where $\lambda$ is a non-negative weighting coefficient used to balance the old and new knowledge.

### 3.2 Knowledge Anchor Routing

Another core idea of MEMoE is the routing strategy based on the knowledge anchors. Inspired by the specialize nature of experts in MoE architecture [53], we aim to route inputs requiring similar knowledge to the same expert during both training and testing phases, thereby enhancing the model's generalization performance when dealing with new knowledge.

In MEMoE, we define the named entities in input sentences as "knowledge anchors". For example, in the input "Who is the president of the United States?" the entities "president" and "United States" serve as knowledge anchors. The routing strategy allocate tokens to the appropriate experts based on these anchors, ensuring that inputs requiring similar knowledge are routed to the same expert. The effect is demonstrated in Figure 2, showing the token distribution within the FFN. This approach better captures and retains the semantic associations of knowledge in input data. Consequently, it enhances the model's generalization performance when handling knowledge and also optimizes the efficiency of expert utilization to a certain extent (as validated in $\$ 5.2$.

Specifically, given an input sequence $X=\left\{x_{i}\right\}_{i=1}^{L}$, we first identify the named entities $x_{\text {anchor }}$ within $X$ using named entity recognition (NER) techniques. We obtain the vector representation of the identified entity through the model's embedding layer, denoted as embed $\left(x_{\text {anchor }}\right)$. To help the gate function notice the knowledge anchor, we use the combination of the anchor embedding and the local token representation. Overall, the knowledge anchors routing strategy can be expressed as:

$$
\begin{equation*}
R_{\text {anchor }}\left(x_{i}\right)=\operatorname{concat}\left(x_{i}, \operatorname{embed}\left(x_{\text {anchor }}\right)\right) \tag{12}
\end{equation*}
$$

Additionally, to address the common issue of expert utilization imbalance in MoE, whether caused by the introduction of knowledge anchor or not, we adopt the auxiliary loss [11] for balancing the top-k selection of routing.

## 4 Experiments

In this section, we first describe our experimental setup. Then, we show the remarkable performance of MEMoE on two challenging model editing tasks: batch editing and sequential batch editing.

### 4.1 Experimental Setups

Datasets and Metrics We use two prominent model editing datasets: ZsRE [26] and CounTERFACT [33], with the split provided by [54, 51]. ZsRE is a context-free Question Answering (QA) dataset built upon zero-shot relation extraction and COUNTERFACT is a more challenging dataset that accounts for counter facts that start with low scores in comparison to correct facts. Further details are provided in Appendix C. In terms of evaluation metrics, we use the three metrics mentioned in $\$ 2$. Reliability, Generality, and Locality, along with the average scores over these metrics.

Baselines We compare the proposed method with mainstream model editing methods, which can be categorized into the following four types [51]:

- Fine-tuning based methods: FT-L [33], FT-M [16], and LoRA[18]. FT-L directly fine-tunes a single layer's FFN and FT-M is a small variation of FT-L using a different loss computation procedure. LoRA is a parameter-efficient fine-tuning method which decomposes the update gradient matrix into two small rank matrices.
- Locate and edit methods: MEMIT [34]. MEMIT treats the feed-forward layer of transformer as a linear associative memory and uses a minimum square error optimization to add new key-value associations to layer weights.
- Meta-learning methods: MEND [35] and COMEBA-HK [29]. MEND learns a hyper-network using additional training data to transform gradient obtained by standard fine-tuning, while COMEBA-HK (COMEBA for short) develop hook layers to identify the editing scope.
- Memory based methods: SERAC [36] and GRACE [16]. SERAC uses an external cache to store explicit editing cases, while GRACE preserves the original model parameters and adopts a codebook to store relevant edits.

Table 1: Batch editing results. Bold is the best result, and underline is the second-best result.

| Method | Model | ZsRE |  |  |  | COUNTERFACT |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Reliability $\uparrow$ | Generality $\uparrow$ | Locality $\uparrow$ | Average $\uparrow$ | $\overline{\text { Reliability } \uparrow}$ | Generality $\uparrow$ | Locality $\uparrow$ | Average |
| FT-L |  | 16.85 | 16.34 | 71.55 | 34.91 | 0.27 | 0.34 | 85.18 | 28.60 |
| FT-M |  | 17.95 | 17.32 | 71.26 | 35.51 | 0.36 | 0.42 | 82.81 | 27.86 |
| LoRA |  | 30.10 | 29.08 | 80.54 | 46.57 | 5.64 | 3.46 | 69.45 | 26.18 |
| MEMIT | CDTา YI | 61.19 | 49.97 | 97.51 | 69.56 | 81.01 | 27.67 | 95.80 | 68.16 |
| MEND | GP12-XL | 2.16 | 2.11 | 20.34 | 8.20 | 0.13 | 0.03 | $\overline{4.22}$ | 1.46 |
| COMEBA |  | 82.21 | 66.61 | 99.40 | 82.74 | 88.28 | 40.38 | 97.66 | 75.44 |
| SERAC |  | 98.64 | $\overline{48.12}$ | 35.68 | $\overline{60.81}$ | 17.88 | 14.55 | 82.25 | $\overline{38.23}$ |
| GRACE |  | 95.56 | 39.76 | 99.93 | 78.41 | $\underline{94.23}$ | 32.56 | 94.58 | 73.79 |
| MEMoE |  | $\overline{95.69}$ | 88.18 | $\overline{100.0}$ | 94.62 | $\overline{93.78}$ | $\overline{85.15}$ | 100.0 | 92.98 |
| FT-L |  | 14.19 | 13.07 | 70.16 | 32.47 | 0.21 | 0.30 | 80.69 | 27.07 |
| FT-M |  | 16.57 | 15.62 | 70.15 | 34.11 | 0.29 | 0.38 | 81.83 | 27.50 |
| LoRA |  | 25.32 | 23.15 | 52.01 | 33.49 | 21.70 | 22.32 | 40.37 | 28.13 |
| MEMIT | LLaMA2-/B | 24.02 | 39.97 | 17.00 | 27.00 | 18.57 | 31.29 | 14.88 | 21.58 |
| MEND |  | 1.01 | 2.83 | 96.77 | 33.54 | 0.45 | 2.24 | 97.89 | 33.53 |
| SERAC |  | 89.08 | 16.29 | 81.82 | 62.39 | 80.67 | 17.34 | 82.05 | 60.02 |
| GRACE |  | 94.50 | 38.20 | 99.90 | 77.53 | 82.14 | 32.09 | 98.93 | 71.05 |
| MEMoE |  | $\overline{100.0}$ | 90.30 | $\overline{100.0}$ | $\overline{96.77}$ | $\overline{99.69}$ | $\overline{88.30}$ | $\overline{100.0}$ | $\overline{96.33}$ |

Table 2: Sequential batch editing results. Bold is the best result, and underline is the second-best.

| Method | Model | ZsRE |  |  |  | COUNTERFACT |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $\overline{\text { Reliability } \uparrow}$ | Generality $\uparrow$ | Locality $\uparrow$ | Average $\uparrow$ | Reliability $\uparrow$ | Generality $\uparrow$ | Locality $\uparrow$ | Average $\uparrow$ |
| $\overline{\text { FT-L }}$ |  | 3.79 | 2.48 | 6.60 | 4.29 | 1.00 | 1.00 | 6.00 | 2.67 |
| FT-M |  | 8.92 | 8.41 | 6.22 | 7.85 | 4.00 | 3.50 | 5.50 | 4.33 |
| LoRA |  | 0.96 | 1.29 | 0.03 | 0.76 | 0.50 | 0.02 | 0.50 | 0.34 |
| MEMIT | GPT2_XI | 34.88 | 32.96 | 70.74 | 46.19 | 56.00 | 37.00 | 31.00 | 41.33 |
| MEND | GP12-XL | 20.95 | 18.29 | 93.69 | 47.01 | 0.01 | 0.00 | 0.08 | 0.03 |
| COMEBA |  | 66.91 | 56.11 | 97.23 | 73.42 | 86.00 | 38.00 | 59.00 | 61.00 |
| SERAC |  | 100.0 | $\overline{36.03}$ | 35.95 | $\overline{57.33}$ | 15.41 | $\overline{12.96}$ | 81.00 | 36.46 |
| GRACE |  | 100.0 | 0.04 | 100.0 | 66.68 | 100.0 | 0.40 | 100.0 | 66.80 |
| MEMoE |  | 74.69 | 58.18 | $\underline{98.93}$ | 77.27 | $\underline{88.12}$ | 54.78 | 99.45 | $\overline{80.78}$ |
| $\overline{\text { FT-L }}$ |  | 2.33 | 1.59 | 6.67 | 3.53 | 0.23 | 0.18 | 10.66 | 3.69 |
| FT-M |  | 6.72 | 4.37 | 7.78 | 6.29 | 0.33 | 0.70 | 8.54 | 3.19 |
| LoRA |  | 0.35 | 1.89 | 0.07 | 0.77 | 0.31 | 0.99 | 0.17 | 0.49 |
| MEMIT | LLaMA2-/B | 12.29 | 29.95 | 15.38 | 19.21 | 10.37 | 32.96 | 12.79 | 18.71 |
| SERAC |  | 67.78 | 33.98 | 34.55 | 45.44 | 20.21 | $\overline{14.05}$ | 34.90 | 23.05 |
| GRACE |  | 89.70 | 0.09 | 98.32 | 62.70 | 74.41 | 1.03 | $\underline{96.67}$ | 57.70 |
| MEMoE |  | 69.50 | 42.63 | $\overline{99.70}$ | $\overline{70.61}$ | 54.62 | 43.40 | $\overline{99.69}$ | 65.9 |

Implementation Details We select GPT2-XL and LLaMA2-7B as the base models. We opted for the more challenging model editing tasks: batch editing and sequential batch editing, to evaluate the performance of MEMoE. For batch editing, following [29], the batch size is set to 30 and the model is rolled back to the initial state after each batch editing. For sequential batch editing, the batch size is 10 for a total of 1000 edits, without rollback. Further details of the baselines and the implementation are provided in the Appendix D

### 4.2 Batch Editing

We first evaluate the effectiveness of MEMoE under batch editing settings. The evaluation results are presented in Table 1. For all models and all metrics, our method consistently achieves the best scores. MEMoE's reliability scores are all above 90 , generalization scores are all above 85 , and locality scores are perfect at 100. The improvements across various metrics are significant. Compared to GPT2-XL, our method demonstrates even more remarkable improvements on LLaMA2-7B, with a maximum improvement of up to 17.55 points in accuracy and 56.21 in generality. In the $\$ 5.2$, we conduct further experimental analysis to explore the reasons behind the substantial enhancement in generalization. Considering some current researches concern that model editing methods may significantly affect a model's general ability [14, 15, 40], we perform a more detailed general ability evaluation using a broader task datasets in $\$ 5.1$.
![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-07.jpg?height=718&width=1400&top_left_y=232&top_left_x=362)

Figure 3: Performance on general tasks of edited models using MEMoE, MEMIT and MEND, with different batch sizes for edits.

### 4.3 Sequential Batch Editing

We evaluate MEMoE on 1,000 samples from both datasets for sequential batch editing. The evaluation is conducted after the entire editing process is completed. The results, shown in Table 2 , indicate that MEMoE achieves the best scores in most cases. It only ranks second because GRACE [16] achieves perfect scores of 100 in some matrices. Similar to the batch editing results, MEMoE performs better on LLaMA2-7B, demonstrating a significant advantage in generality while maintaining accuracy and locality much close to 100 . Regarding GRACE, it excels in reliability and locality but performs poorly in generality due to its use of a codebook to memorize encountered editing instances [16, 12]. However, its poor performance in generality suggests a problem with regurgitation. Overall, the reliability and generality consistently lag behind in comparison to batch editing, indicating there is still room for improvement in this field.

## 5 Detailed Analysis and Discussion

In this section, we conduct a further evaluation and analysis of the performance of MEMoE. Firstly, we assess the impact of MEMoE on the model's general ability using a broader range of task datasets. Secondly, we show potential sources of MEMoE's generalization advantage through an analysis of expert specialize phenomena. Finally, we present an extensive serious of ablation study to evaluate the efficacy of various model configurations, including the number of experts, the target layer and the routing strategies.

### 5.1 General Ability Test

To investigate the potential impact of model editing on the general ability of LLMs, we select eight representative task categories for evaluation, as outlined below following [14]. For reasoning, we utilized the GSM8K dataset [5], with performance assessed by solve rate. Natural language inference (NLI) tasks were evaluated on the RTE dataset [1], with accuracy measured through twoway classification. For open-domain question answering, the Natural Question dataset [23] was employed, evaluating exact match against reference answers after minor normalization as in [2] and [24]. Similarly, closed-domain QA tasks were assessed using the BoolQ dataset [4], also measured by EM. Dialogue evaluation utilized the MuTual dataset [6], with results determined by selecting the most suitable response from four options, denoted as Recall $4_{4}$ 1 [31]. Evaluation for summarization tasks was conducted on the SAMSum dataset [13], using the average of ROUGE-1, ROUGE-2, and ROUGE-L as evaluation metrics. For named entity recognition (NER), the CoNLL03 dataset

Table 3: Expert Specific Experimental Results. "Dynamic" indicates that we dynamically select input data, while "Static" refers to evaluation conducted using models trained on previous experiments. "id" stands for group id. "Similar" refers to similar knowledge, and "same" refers to same knowledge.

| Type | Number of data from each group |  |  |  |  | Consistency $\uparrow$ |  | Generality $\uparrow$ | Reliability $\uparrow$ | Locality ${ }^{\uparrow}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | id $=1$ | id $=\mathbf{2}$ | id $=3$ | id $=4$ | id $=5$ | similar | same |  |  |  |
| Dynamic | 10 | 10 | 10 | 10 | 10 | 63.09 | 73.77 | 0 | 9.84 | 100 |
|  | 20 | 10 | 10 | 10 | 0 | 65.47 | 73.79 | 8 |  |  |
|  | 30 | 10 | 10 | 0 | 0 | 69.67 | 76.33 | 90.12 | 99.84 | 100 |
|  | 40 | 10 | 0 | 0 | 0 | 74.63 | 80.13 | 92.01 | 99.84 | 100 |
|  | 50 | ![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-08.jpg?height=42&width=70&top_left_y=612&top_left_x=630) | 0 | 0 | 0 | 79.69 | 84.82   | 94.78 | 99.84 | 100 |
| Static | ![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-08.jpg?height=43&width=85&top_left_y=643&top_left_x=532) | - | - | - | - | 65.14 | 77.77 | 90.00 | 99.84 | 100 |

![](https://cdn.mathpix.com/cropped/2024_06_04_6919f628b6104e81f363g-08.jpg?height=468&width=1390&top_left_y=730&top_left_x=365)

Figure 4: Left: Performance across different numbers of experts. Middle: Performance across different target model layers. Right: Effectiveness of activating experts. All experiments are based on LLaMA2-7B, utilizing the ZsRE dataset and batch editing settings.

[42] was employed, with performance measured using entity-level F1-score. Lastly, for sentiment analysis, we utilized SST2 dataset [46], with accuracy assessed through a two-way classification.

We conduct evaluations on LLaMA2-7B based on batch editing settings, progressively increasing the batch size to show the impact of more edited samples (the model is rolled back to the initial state after each batch editing). The results are shown in the Figure 3. It is important to note that any increase or decrease in performance metrics implies an impact on model's general ability. Compared to the MEMIT and MEND, the MEMoE yields consistently stable model performance under various batch editing conditions. With the increase in batch size and edited samples, both MEMIT and MEND significantly diminish the model's general ability, while the influence of MEMoE fluctuates within a smaller range. This further corroborates MEMoE's advantage in locality score in $\$ 4.2$.

### 5.2 Experts Specialize

As claimed in Section 3.2, we hypothesize that knowledge anchor routing has the ability to direct inputs requiring similar or same knowledge to the same expert, thereby improving the generalization performance. This hypothesis comprises two parts: (1) same knowledge being handled by the same expert, and (2) similar knowledge being handled by the same expert. We proceed to verify it.

In this context, we define "inputs requiring same knowledge" to specifically denote editing inputs and generalization testing inputs that target the same knowledge. For example, training input like "Who is the president of the United States?" and generalization test input like "Who currently holds the position of the U.S. presidency?" require the same knowledge: "Joe Biden is the president of the United States". In contrast, "similar knowledge" refers to inputs that contain semantically related knowledge (based on semantic distance). We apply the simple traditional K-means algorithm to cluster all the data in ZsRE into 5 groups based on the cosine similarity, defining knowledge with the same group id as similar.

We design the metric Consistency to evaluate the extend to which similar or same knowledge is processed by the same experts. Specifically, for an input sequence $S_{k}=\left\{x_{i}\right\}_{i=1}^{L}$ in group $k$ and
the expert $E_{k}$ handling the majority of knowledge in this group is used as the ground truth, the Consistency of this group is defined as:

$$
\begin{equation*}
\mathbb{E}_{x_{i} \sim S_{k}} \mathbb{1}\left\{E_{k}=R_{\text {anchor }}\left(x_{i}\right)\right\} \tag{13}
\end{equation*}
$$

where $R_{\text {anchor }}$ is the knowledge anchor routing defined by Equation 12. The ground truth for generalization test input is the expert processing the corresponding training input that contains the same knowledge. The overall consistency score is calculated as the average across all groups.

In order to fully analyze the behavior of experts routing when dealing with different knowledge inputs, we first conduct a static analysis based the trained MEMoE module in 4.2. That is to say, at this stage, the editing data is fixed; we analyze the behavior of experts solely through the model inference process. Further, we dynamically select the knowledge to be edited as input to observe the behavior of knowledge anchor routing. The results are shown in Table 3. As the concentration of the input knowledge categories increases, the consistency of the experts also improves, and the generality simultaneously rise. Additionally, the consistency of the same experts handling the same knowledge is higher and closely aligned with the generalization scores. Given the near-perfect accuracy score, we speculate that errors in generalization assessments may be due to incorrect routing of inputs to the wrong expert. Analysis of the bad cases in the generalization test supports this hypothesis. By adhering to the principle that "professional people do professional things", the strategy of routing inputs requiring similar or same knowledge to the same expert proves effective in improving knowledge generalization.

### 5.3 Ablation Study

Effect of Expert Number How does the number of experts impact the performance of MEMoE? The left plot in Figure 4 illustrates the performance of MEMoE with different numbers of experts. Due to computational resource limitations, we could only add up to 6 additional experts. Aligning with the primary experimental results in $\$ 4$, we set the $t o p_{k}$ value to 1 . We find that the reliability and locality of model editing do not change with the number of experts; there is neither a decrease nor an improvement in performance. However, the generalization of knowledge fluctuated with the number of experts, peaking when the number of experts is 4. Inspired by [53], we hypothesize that this optimal experts number is related to editing batch size. When the number of editing samples is not large, more experts may introduce interference, thereby reducing the generalization performance.

Effect of Target Layer What is the optimal layer for applying MEMoE? As shown in the middle of Figure 4, similar to the results of experts number experiment, the reliability and locality score remain unaffected, with only the generality score exhibiting fluctuations and peaking at the 16th layer. The best editing layer identified from these validation experiments aligns with the results obtained using knowledge probes technology in [33]. Combining the findings from above experiments, we can infer that the accuracy and locality performance are inherently guaranteed by the characteristics of the bypass MoE structure, which is consistent with the design principles of MEMoE discussed in $\$ 3.1$.

Routing Strategy: Soft vs Discrete What is the best routing strategy in MEMoE? In Figure 4 . the rightmost plot illustrates the performance when using various routing strategies for MEMoE. Specifically, we compare the soft merging [53] of experts with discrete top-1, top-2, and top-3 routing strategy. The top-1 routing setting yields the best performance. Further, based on the experiment introduced in $\$ 5.2$, we examined the consistency score of routing under different values of $k$. We observed that as $\mathrm{k}$ increases, more experts participate in the computation, but the consistency score decreases. We believe that this inconsistency in expert utilization leads to the decrease in generalization performance. Further experimental results, detailed in the Appendix E. demonstrate that regardless of the expert number and the target layer, the top-1 performance remains the best. In addition, the discrete top-1 routing has an advantage in computational efficiency by requiring only one experts to be activated during inference.

Batch Editing vs Sequential Editing From Table 1 and 2 , it is evident that MEMoE demonstrates a significant performance advantage in batch editing tasks over sequential editing. To further evaluate MEMoE's ability in batch editing, we progressively increased the batch size. Experimental results are shown in Table 4. When the batch size for batch editing reached 1000, equivalent to the total number of sequential batch editing, MEMoE exhibited significant performance advantages. Reliability and

Table 4: Comparison of batch editing with larger batch size and sequential batch editing.

| Task Settings | Size | Reliability $\uparrow$ | Generality $\uparrow$ | Locality $\uparrow$ | Average $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
|  | 10 | 100.0 | 90.12 | 100.0 | 96.71 |
| Batch Editing | 100 | 99.84 | 80.91 | 100.0 | 93.58 |
|  | 1000 | 99.30 | 75.70 | 100.0 | 91.67 |
| Sequential Batch Editing | 1000 | 69.50 | 42.63 | 99.70 | 70.61 |

locality are scarcely affected by the increase in batch size, maintaining close to 100. Meanwhile, the generality score surpassed by 33.07 points, highlighting MEMoE's performance advantage in batch editing. As for the decline in sequential editing, our analysis of bad cases indicates that can be attributed to catastrophic forgetting: edits complete earlier are more prone to errors.

## 6 Conclusion

In this paper, we present MEMoE, a model editing adapter utilizing MoE architecture with knowledge anchor routing strategy. MEMoE updates knowledge using a bypass MoE structure, keeping the original parameters unchanged to preserve the model's general ability. And, the knowledge anchor ensures that questions requiring similar knowledge are handled by the same expert, thereby enhancing the generalization of the updated knowledge. Experiment results demonstrate that our method significantly surpasses all compared baselines. MEMoE shows near-perfect accuracy and locality scores close to 100 , along with a generalization score exceeding 90 , indicating exciting promise for practical applications of model editing technology.

## Acknowledgements

This research is supported by the National Natural Science Foundation of China (No.62106105), the CCF-Baidu Open Fund (No.CCF-Baidu202307), the Scientific Research Starting Foundation of Nanjing University of Aeronautics and Astronautics (No.YQR21022), and the High Performance Computing Platform of Nanjing University of Aeronautics and Astronautics.

## References

[1] J. Q. Candela, I. Dagan, B. Magnini, and F. d'Alché-Buc, editors. Machine Learning Challenges, Evaluating Predictive Uncertainty, Visual Object Classification and Recognizing Textual Entailment, First PASCAL Machine Learning Challenges Workshop, MLCW 2005, Southampton, UK, April 11-13, 2005, Revised Selected Papers, volume 3944 of Lecture Notes in Computer Science, 2006. Springer. ISBN 3-540-33427-0. doi: 10.1007/11736790. URL https://doi.org/10.1007/11736790.

[2] D. Chen, A. Fisch, J. Weston, and A. Bordes. Reading wikipedia to answer open-domain questions. In R. Barzilay and M. Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1870-1879. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17-1171. URL https://doi.org/10.18653/v1/P17-1171.

[3] Z. Chen, Y. Deng, Y. Wu, Q. Gu, and Y. Li. Towards understanding the mixture-of-experts layer in deep learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/ hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html.

[4] C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT

2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 29242936. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1300. URL https://doi.org/10.18653/v1/n19-1300.

[5] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

[6] L. Cui, Y. Wu, S. Liu, Y. Zhang, and M. Zhou. Mutual: A dataset for multi-turn dialogue reasoning. In D. Jurafsky, J. Chai, N. Schluter, and J. R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 1406-1416. Association for Computational Linguistics, 2020. doi: 10.18653/ V1/2020.ACL-MAIN.130. URL https://doi.org/10.18653/v1/2020.acl-main. 130.

[7] D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei. Knowledge neurons in pretrained transformers. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8493-8502. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-LONG.581. URL https: //doi.org/10.18653/v1/2022.acl-long.581.

[8] Q. Dong, D. Dai, Y. Song, J. Xu, Z. Sui, and L. Li. Calibrating factual knowledge in pretrained language models. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5937-5947. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-EMNLP.438. URL https://doi.org/10.18653/v1/2022 findings-emnlp. 438

[9] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph, L. Fedus, M. P. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. S. Meier-Hellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui. Glam: Efficient scaling of language models with mixture-of-experts. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 5547-5569. PMLR, 2022. URL https://proceedings.mlr.press/v162/du22c.html.

[10] D. Eigen, M. Ranzato, and I. Sutskever. Learning factored representations in a deep mixture of experts. In Y. Bengio and Y. LeCun, editors, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Workshop Track Proceedings, 2014. URL http://arxiv.org/abs/1312.4314.

[11] W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. J. Mach. Learn. Res., 23:120:1-120:39, 2022. URL http: //jmlr.org/papers/v23/21-0998.html.

[12] C. Gao, K. Chen, J. Rao, B. Sun, R. Liu, D. Peng, Y. Zhang, X. Guo, J. Yang, and V. S. Subrahmanian. Higher layers need more lora experts. CoRR, abs/2402.08562, 2024. doi: 10.48550/ARXIV.2402.08562. URL https://doi.org/10.48550/arXiv.2402.08562.

[13] B. Gliwa, I. Mochol, M. Biesek, and A. Wawer. Samsum corpus: A human-annotated dialogue dataset for abstractive summarization. CoRR, abs/1911.12237, 2019. URL http://arxiv org/abs/1911.12237.

[14] J. Gu, H. Xu, J. Ma, P. Lu, Z. Ling, K. Chang, and N. Peng. Model editing can hurt general abilities of large language models. CoRR, abs/2401.04700, 2024. doi: 10.48550/ARXIV.2401. 04700. URLhttps://doi.org/10.48550/arXiv.2401.04700.

[15] A. Gupta, A. Rao, and G. Anumanchipalli. Model editing at scale leads to gradual and catastrophic forgetting. CoRR, abs/2401.07453, 2024. doi: 10.48550/ARXIV.2401.07453. URL https://doi.org/10.48550/arXiv.2401.07453

[16] T. Hartvigsen, S. Sankaranarayanan, H. Palangi, Y. Kim, and M. Ghassemi. Aging with GRACE: lifelong model editing with discrete key-value adaptors. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html

[17] H. Hazimeh, Z. Zhao, A. Chowdhery, M. Sathiamoorthy, Y. Chen, R. Mazumder, L. Hong, and E. H. Chi. Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29335-29347, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html.

[18] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[19] Z. Huang, Y. Shen, X. Zhang, J. Zhou, W. Rong, and Z. Xiong. Transformer-patcher: One mistake worth one neuron. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=4oYUGeGBPm.

[20] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton. Adaptive mixtures of local experts. Neural Comput., 3(1):79-87, 1991. doi: 10.1162/NECO.1991.3.1.79. URL https: //doi.org/10.1162/neco.1991.3.1.79

[21] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de Las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mixtral of experts. CoRR, abs/2401.04088, 2024. doi: 10.48550/ARXIV.2401.04088. URL https://doi.org/10.48550/arXiv. 2401. 04088

[22] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http: //arxiv.org/abs/1412.6980.

[23] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. P. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics, 7:452-466, 2019. doi: 10.1162/TACL\A\_00276. URL https://doi.org/10.1162/tacl_a_00276

[24] K. Lee, M. Chang, and K. Toutanova. Latent retrieval for weakly supervised open domain question answering. In A. Korhonen, D. R. Traum, and L. Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 6086-6096. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1612. URL https://doi.org/10. $18653 / \mathrm{v} 1 / \mathrm{p} 19-1612$

[25] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.

[26] O. Levy, M. Seo, E. Choi, and L. Zettlemoyer. Zero-shot relation extraction via reading comprehension. In R. Levy and L. Specia, editors, Proceedings of the 21st Conference on

Computational Natural Language Learning (CoNLL 2017), Vancouver, Canada, August 3-4, 2017, pages 333-342. Association for Computational Linguistics, 2017. doi: 10.18653/V1/ K17-1034. URL https://doi.org/10.18653/v1/K17-1034.

[27] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer. BASE layers: Simplifying training of large, sparse models. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 6265-6274. PMLR, 2021. URL http://proceedings.mlr.press/v139/lewis21a.html.

[28] D. Li, A. S. Rawat, M. Zaheer, X. Wang, M. Lukasik, A. Veit, F. X. Yu, and S. Kumar. Large language models with controllable working memory. In A. Rogers, J. L. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1774-1793. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.112. URL https://doi.org/10.18653/ v1/2023.findings-acl. 112

[29] S. Li, Y. Deng, D. Cai, H. Lu, L. Chen, and W. Lam. Consecutive model editing with batch alongside hook layers. CoRR, abs/2403.05330, 2024. doi: 10.48550/ARXIV.2403.05330. URL https://doi.org/10.48550/arXiv.2403.05330

[30] X. Li, S. Li, S. Song, J. Yang, J. Ma, and J. Yu. PMET: precise model editing in a transformer. CoRR, abs/2308.08742, 2023. doi: 10.48550/ARXIV.2308.08742. URL https://doi.org/ 10.48550/arXiv. 2308.08742 .

[31] R. Lowe, N. Pow, I. Serban, and J. Pineau. The ubuntu dialogue corpus: A large dataset for research in unstructured multi-turn dialogue systems. In Proceedings of the SIGDIAL 2015 Conference, The 16th Annual Meeting of the Special Interest Group on Discourse and Dialogue, 2-4 September 2015, Prague, Czech Republic, pages 285-294. The Association for Computer Linguistics, 2015. doi: 10.18653/V1/W15-4640. URL https://doi.org/10.18653/v1/ $\mathrm{w} 15-4640$.

[32] A. Madaan, N. Tandon, P. Clark, and Y. Yang. Memory-assisted prompt editing to improve GPT-3 after deployment. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2833-2861. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.183. URL https://doi.org/10.18653/v1/2022.emnlp-main.183.

[33] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in GPT. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/ hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html

[34] K. Meng, A. S. Sharma, A. J. Andonian, Y. Belinkov, and D. Bau. Mass-editing memory in a transformer. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview net/pdf?id=MkbcAHIYgyS

[35] E. Mitchell, C. Lin, A. Bosselut, C. Finn, and C. D. Manning. Fast model editing at scale. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id= ODcZxeWfOPt

[36] E. Mitchell, C. Lin, A. Bosselut, C. D. Manning, and C. Finn. Memory-based model editing at scale. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 1581715831. PMLR, 2022. URLhttps://proceedings.mlr.press/v162/mitchell22a.html

[37] S. Murty, C. D. Manning, S. M. Lundberg, and M. T. Ribeiro. Fixing model bugs with natural language patches. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 11600-11613. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.797. URL https://doi.org/10.18653/v1/2022.emnlp-main.797.

[38] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303. 08774. URL https://doi.org/10.48550/arXiv.2303.08774.

[39] F. Petroni, T. Rocktäschel, S. Riedel, P. S. H. Lewis, A. Bakhtin, Y. Wu, and A. H. Miller. Language models as knowledge bases? In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 2463-2473. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1250. URL https://doi.org/10.18653/v1/ D19-1250

[40] Y. Pinter and M. Elhadad. Emptying the ocean with a spoon: Should we edit models? In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 15164-15172. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.1012. URL https://doi.org/10.18653/v1/2023.findings-emnlp.1012.

[41] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pinto, D. Keysers, and N. Houlsby. Scaling vision with sparse mixture of experts. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 8583-8595, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 48237d9f2dea8c74c2a72126cf63d933-Abstract.html.

[42] E. F. T. K. Sang and F. D. Meulder. Introduction to the conll-2003 shared task: Languageindependent named entity recognition. In W. Daelemans and M. Osborne, editors, Proceedings of the Seventh Conference on Natural Language Learning, CoNLL 2003, Held in cooperation with HLT-NAACL 2003, Edmonton, Canada, May 31 - June 1, 2003, pages 142-147. ACL, 2003. URL https://aclanthology.org/W03-0419/

[43] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter. CoRR, abs/1910.01108, 2019. URL http://arxiv.org/abs/ 1910.01108

[44] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https://openreview.net/forum? id=B1ckMDqlg.

[45] S. Shen, L. Hou, Y. Zhou, N. Du, S. Longpre, J. Wei, H. W. Chung, B. Zoph, W. Fedus, X. Chen, et al. Mixture-of-experts meets instruction tuning: A winning combination for large language models. arXiv preprint arXiv:2305.14705, 2023. URL https://arxiv.org/abs/ 2305.14705

[46] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 1631-1642. ACL, 2013. URL https://aclanthology. org/D13-1170/

[47] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and
efficient foundation language models. CoRR, abs/2302.13971, 2023. doi: 10.48550/ARXIV. 2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971.

[48] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288

[49] R. Wang and P. Li. Semantic are beacons: A semantic perspective for unveiling parameterefficient fine-tuning in knowledge learning. arXiv preprint arXiv:2405.18292, 2024. URL https://arxiv.org/abs/2405.18292.

[50] Y. Xie, S. Huang, T. Chen, and F. Wei. Moec: Mixture of expert clusters. In B. Williams, Y. Chen, and J. Neville, editors, Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 13807-13815. AAAI Press, 2023. doi: 10.1609/AAAI.V37I11.26617. URL https://doi.org/10.1609/aaai.v37i11.26617.

[51] Y. Yao, P. Wang, B. Tian, S. Cheng, Z. Li, S. Deng, H. Chen, and N. Zhang. Editing large language models: Problems, methods, and opportunities. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 10222-10240. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.632. URL https://doi.org/10.18653/v1/2023.emnlp-main.632.

[52] Z. Yao, Y. He, T. Qi, and M. Li. Scalable model editing via customized expert networks. CoRR, abs/2404.02699, 2024. doi: 10.48550/ARXIV.2404.02699. URL https://doi.org/ 10.48550/arXiv. 2404.02699 .

[53] T. Zadouri, A. Üstün, A. Ahmadian, B. Ermis, A. Locatelli, and S. Hooker. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. CoRR, abs/2309.05444, 2023. doi: 10.48550/ARXIV.2309.05444. URL https://doi.org/10. 48550/arXiv. 2309.05444

[54] N. Zhang, Y. Yao, B. Tian, P. Wang, S. Deng, M. Wang, Z. Xi, S. Mao, J. Zhang, Y. Ni, S. Cheng, Z. Xu, X. Xu, J. Gu, Y. Jiang, P. Xie, F. Huang, L. Liang, Z. Zhang, X. Zhu, J. Zhou, and H. Chen. A comprehensive study of knowledge editing for large language models. CoRR, abs/2401.01286, 2024. doi: 10.48550/ARXIV.2401.01286. URL https: //doi.org/10.48550/arXiv. 2401.01286

[55] C. Zheng, L. Li, Q. Dong, Y. Fan, Z. Wu, J. Xu, and B. Chang. Can we edit factual knowledge by in-context learning? In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4862-4876. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.emnlp-main. 296

[56] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Y. Zhao, A. M. Dai, Z. Chen, Q. V. Le, and J. Laudon. Mixture-of-experts with expert choice routing. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html

[57] J. Zhu, X. Zhu, W. Wang, X. Wang, H. Li, X. Wang, and J. Dai. Uni-perceivermoe: Learning sparse generalist models with conditional moes. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 11fc8c98b46d4cbdfe8157267228f7d7-Abstract-Conference.html

[58] S. Zuo, X. Liu, J. Jiao, Y. J. Kim, H. Hassan, R. Zhang, J. Gao, and T. Zhao. Taming sparsely activated transformer with stochastic experts. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=B72HXs80q4
</end of paper 3>


