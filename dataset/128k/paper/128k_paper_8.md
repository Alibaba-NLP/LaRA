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
# Evaluating the External and Parametric Knowledge Fusion of Large Language Models 

Hao Zhang*, Yuyang Zhang*, Xiaoguang Li, Wenxuan Shi, Haonan Xu, Huanshuo Liu<br>Yasheng Wang, Lifeng Shang, Qun Liu, Yong Liu, Ruiming Tang<br>Noah's Ark Lab, Huawei Technologies Co., Ltd<br>\{zhang.hao3, zhangyuyang4\}@huawei.com


#### Abstract

Integrating external knowledge into large language models (LLMs) presents a promising solution to overcome the limitations imposed by their antiquated and static parametric memory. Prior studies, however, have tended to over-reliance on external knowledge, underestimating the valuable contributions of an LLMs' intrinsic parametric knowledge. The efficacy of LLMs in blending external and parametric knowledge remains largely unexplored, especially in cases where external knowledge is incomplete and necessitates supplementation by their parametric knowledge. We propose to deconstruct knowledge fusion into four distinct scenarios, offering the first thorough investigation of LLM behavior across each. We develop a systematic pipeline for data construction and knowledge infusion to simulate these fusion scenarios, facilitating a series of controlled experiments. Our investigation reveals that enhancing parametric knowledge within LLMs can significantly bolster their capability for knowledge integration. Nonetheless, we identify persistent challenges in memorizing and eliciting parametric knowledge, and determining parametric knowledge boundaries. Our findings aim to steer future explorations on harmonizing external and parametric knowledge within LLMs.


## 1 Introduction

Parametric knowledge acquired by large language models (LLMs) (OpenAI, 2023, Touvron et al., 2023; Anil et al. 2023; Du et al., 2022) during pre-training inevitably becomes outdated over time. Integrating additional contents into LLM inputs has emerged as an effective strategy to mitigate such issue (Lewis et al., 2020, Nakano et al., 2021, Gao et al., 2023). By incorporating external knowledge either into the input context (Ram et al., 2023|, Izacard et al.| 2022) or through intermediary layers (Borgeaud et al., 2022; Wu et al. 2022), LLMs are endowed with more current information, expanding their knowledge boundary and reducing the instances of hallucinations and factual errors.

Many retrieval (Lewis et al., 2020; Asai et al., 2023; Izacard et al., 2022) or tool (Shen et al., 2023; Qin et al., 2024: Schick et al., 2023) augmented methods predominantly rely on external evidence and often overlooking the rich knowledge stored within LLMs. Yet, the external evidence, inevitably, could be incomplete and noisy. While some approaches propose to refine the external evidence and post-calibrate the outputs by tapping into LLMs' parametric knowledge (Meng et al., 2022; Zhang et al. 2024), the full potential of merging external with parametric knowledge remains unexplored. This paper aims to delve into how LLMs perform external and parametric knowledge fusion across various conditions, especially when LLMs encounter incomplete or irrelevant external knowledge. A thorough understanding of this is crucial for a broader application of knowledge-augmented LLMs. Not only does this relate to the LLMs' parametric memory elicitation (Xie et al., 2024; Qian et al.,[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-02.jpg?height=358&width=1309&top_left_y=255&top_left_x=403)

Figure 1: An illustration of four parametric and external knowledge fusion scenarios in LLMs.

2023, Wang et al. 2023b c), but it is also associated with the knowledge boundary perception of LLMs (Ren et al., 2023b; Zhang et al., 2023b, Yin et al., 2023b).

To elucidate the dynamics of LLMs in integrating external $\left(K_{e}\right)$ and parametric $\left(K_{p}\right)$ knowledge ${ }^{2}$ we define four distinct scenarios reflecting the interplay between $K_{e}$ and $K_{p}$ (depicted in Figure 1). The scenarios are as follows: (1) $S_{1}$ indicates that $K_{e}$ alone is sufficient to answer a query, independent of $K_{p}$ 's contribution; (2) $S_{2}$ suggests that $K_{e}$ provides partial information, requiring $K_{p}$ to fill the gaps for a complete answer; (3) $S_{3}$ identifies situations where $K_{e}$ offers no useful information, and the answer depends solely on $K_{p}$; (4) $S_{4}$ describes cases where neither $K_{e}$ nor $K_{p}$ adequately address a query, making it theoretically unanswerable. Prior studies (Yoran et al., 2023, Chen et al. 2023) often presume situations where the availability of external knowledge $\left(K_{e}\right)$ and $K_{p}$ is non-contributory, simplifying the knowledge fusion process to scenarios $S_{1}$ and $S_{4}$ and neglecting intermediate cases. The real challenge emerges when $K_{e}$ is sub-optimal, necessitating a nuanced integration of $K_{e}$ and $K_{p}$ for a cooperative response, especially in scenarios $S_{2}$ and $S_{3}$. However, the model-specific nature of LLMs' $K_{p}$ significantly complicates the precise delineation of knowledge boundaries and access to parametric knowledge. This complexity impedes a thorough and impartial evaluation of LLMs' capabilities in knowledge fusion.

To mitigate the challenges associated with acquiring parametric knowledge by LLMs, we propose a systematic pipeline for data construction and knowledge infusion. Specifically, we first collect the latest data from the electronic product domain and divide it into two parts: one for enhancing LLMs' parametric knowledge $\left(K_{p}\right)$ through continued training, and the other as external knowledge $\left(K_{e}\right)$. We also craft a set of questions based on the data to emulate the four scenarios: queries that solely depend on $K_{e}\left(S_{1}\right)$, queries requiring integration of $K_{e}$ and $K_{p}\left(S_{2}\right)$, queries dependent only on $K_{p}$ $\left(S_{3}\right)$, and unanswerable queries $\left(S_{4}\right)$. For each scenario, we provide relevant evidence and introduce additional distractors to mimic real-world conditions. Overall, this aims to standardize the parametric knowledge within different LLMs, facilitating equitable and model-independent evaluations.

We first inject new knowledge into LLMs through continued training and supervised fine-tuning, subsequently evaluating their knowledge retention. Then, we design a series of experiments to reveal the behaviors of LLMs in knowledge fusion. Despite the performance gains by integrating external and parametric knowledge, the results indicate that: (1) LLMs show deficiencies in recognizing domain knowledge, significantly influenced by their capacity to retain knowledge. (2) There are persistent challenges in memorizing and eliciting parametric knowledge and determining parametric knowledge boundaries for effective knowledge fusion. Our contributions are as follows:

- We review knowledge fusion in LLMs, defining four distinct scenarios reflecting the interplay between external and parametric knowledge fusion for thorough evaluation.
- To mitigate the challenges associated with acquiring parametric knowledge by LLMs, we propose a systematic pipeline for data construction and knowledge infusion to facilitate knowledge fusion exploration.
- Through extensive experiments on various LLM backbones, we identify persistent challenges in memorizing and eliciting parametric knowledge and determining parametric knowledge boundaries. These challenges impair the effectiveness of knowledge fusion.[^1]


## 2 Related Work

Retrieval-augmented LLMs (RA-LLM). RA-LLM, including tools, are considered essential for linking LLMs with external knowledge sources (Lewis et al., 2020; Qin et al., 2024; Mialon et al. 2023; Gao et al. 2023), which makes LLMs more viable for practical applications. The prevalent methods either augment external evidence via in-context learning paradigm (Lazaridou et al., 2022; He et al. 2022; Izacard et al. 2022) or adopt external evidence to post-calibrate the generations (Meng et al.| 2022; Li et al.||2024; Yan et al., 2024). Some work also suggests fine-tuning LLMs to enhance the utilization of external knowledge and optimize the retrieval strategy (Lewis et al. 2020; Borgeaud et al. 2022, Asai et al. 2023, Lin et al. 2023). These approaches mainly rely on external knowledge while overlooking the knowledge stored within LLMs, which may lead to undesirable results due to the biased and noisy external information (Mallen et al., 2022; Yoran et al., 2023, Liu et al., 2023b).

Parametric Knowledge in LLMs. After pre-training, LLMs have internalized massive knowledge into their parameters, i.e., parametric knowledge (Petroni et al., 2019, Geva et al., 2021b; $\mathrm{Hu}$ et al. 2023; Gueta et al. 2023). However, recent studies indicate that effectively leveraging LLMs' parametric knowledge is challenging (Wang et al., 2023a, Allen-Zhu and Li 2023a). That is, although LLMs can memorize extensive knowledge, it does not guarantee their ability to adeptly elicit and manipulate it for subsequent tasks (Berglund et al., 2023; Liu et al., 2023a; Wang et al., 2023d; Allen-Zhu and Li, 2023b). Some work also observes that compared to directly augmenting knowledge into inputs, LLMs struggle to accurately memorize knowledge into parameters (Kandpal et al., 2023, Ovadia et al., 2023). Besides, several studies explore the self-calibration (Rajpurkar et al. 2018 Kadavath et al. 2022; Yin et al., 2023a) and knowledge boundary detection (Ren et al., 2023a) in LLMs, benefit for improving confidence and interpretability in their use of parametric knowledge, thus reducing hallucinations. Similarly, our objective is to investigate the capacity of LLMs for knowledge memorization and utilization in the knowledge fusion process, along with their self-calibration ability.

Knowledge Fusion of LLMs. To perform the fusion of external and parametric knowledge, Jiang et al. (2023) propose dynamically assessing the confidence level of model generation and intervening with retrieval at low confidence. Wang et al. (2023c) elicit LLMs' ability to recognize their selfknowledge and achieve better knowledge integration. Some studies explore the knowledge conflict issues when integrating the external and parametric knowledge (Li et al., 2022; Pan et al., 2021; Mallen et al., 2022, Zhang et al. 2023a, Xie et al. 2023). However, these approaches mainly optimize knowledge fusion to enhance the subsequent tasks, such as open-domain QA (Kwiatkowski et al. 2019: Yang et al. 2018; Geva et al., 2021a), lacking a comprehensive evaluation of LLMs' behaviors in knowledge fusion. In contrast, we focus on the investigation of external and parametric knowledge fusion, including the systematic task definition, data construction pipeline, and thorough experiments.

## 3 Task Definition

In practical applications, the external evidence obtained through retrieval or tools may be noisy, incomplete, or irrelevant Yoran et al. (2023); Liu et al. (2023b). This leads to the necessity of thoroughly considering various conditions when evaluating the external and parametric knowledge fusion. Therefore, we define four distinct scenarios capturing the diverse interactions between external and parametric knowledge of LLMs, aiming to encompass all potential circumstances as comprehensively as possible. Given external knowledge $K_{e}$ and parametric knowledge $K_{p}$, the defined scenarios are: (1) $S_{1}$ indicates $K_{e}$ alone is sufficient to answer a query, independent of $K_{p}$ 's contribution; (2) $S_{2}$ denotes $K_{e}$ carries partial information, requiring $K_{p}$ to fill the gaps for a complete answer; (3) $S_{3}$ identifies situations where $K_{e}$ offers no useful information, and the answer depends solely on $K_{p}$; (4) $S_{4}$ describes cases where neither $K_{e}$ nor $K_{p}$ adequately address a query, making it theoretically unanswerable.

Suppose $K_{p}$ has been injected into the LLM. Formally, given a question $q_{S_{i}}$ and the corresponding external evidence $K_{e}^{i}$, where $i \in\{1,2,3,4\}$, the response $\hat{a}_{S_{i}}$ of an LLM is generated as:

$$
\begin{equation*}
\hat{a}_{S_{i}}=\operatorname{LLM}_{\left(K_{p}\right)}\left(\left[q_{S_{i}} ; K_{e}^{i} ; \text { inst }\right]\right) \tag{1}
\end{equation*}
$$

where $\operatorname{LLM}_{\left(K_{p}\right)}$ denotes the LLM already encoded the $K_{p}$ into its parameters, inst represents the task-specific instructions. Ideally, for $S_{1}, K_{e}^{1}$ contains ground-truth evidence, where LLM can solely

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-04.jpg?height=490&width=849&top_left_y=243&top_left_x=367)

Figure 2: The overview of dataset construction. We first retrieve documents of electronics from websites. The documents are split into two portions based on their released date, and decomposed into paragraphs. The QA pairs for each scenario are generated via prompting LLMs, and the corresponding external evidence and noise are added as support sources. The outdated data is injected into LLMs through pre-training or fine-tuning.

depend on $K_{e}^{1}$ to accurately answer $q_{s_{1}}$, that is, $\hat{a}_{S_{1}}$ is aligned with ground-truth $a_{S_{1}}$. For $S_{2}, K_{e}^{2}$ holds only partial information relevant to $q_{S_{2}}$, LLM also requires to elicit its corresponding $K_{p}$ to derive an accurate $\hat{a}_{S_{2}}$ for $q_{s_{2}}$. For $S_{3}, K_{e}^{3}$ is devoid of relevant information and is solely comprised of distractions, LLMs must eliminate these distractions and elicit its $K_{p}$ to reach the correct $\hat{a}_{S_{3}}$. In $S_{4}$, where $K_{e}^{4}$ consists solely of distractors and LLM lacks relevant $K_{p}$, it should opt to refrain from responding, implying that $\hat{a}_{S_{4}}$ should incorporate a refusal to answer $q_{S_{4}}$. Following the criteria outlined, we construct datasets, fine-tune different LLMs, and conduct a detailed evaluation of their ability to integrate external and parametric knowledge in these scenarios.

## 4 Dataset Construction

Although LLMs encode massive knowledge through large-scale pre-training, the parametric knowledge of different LLMs exhibits notable variations due to discrepancies in training corpora, model scale, and forgetting issue (Wang et al. 2023a, Luo et al., 2023). Thus, it is challenging and almost infeasible to directly elicit the parametric knowledge of various LLMs (Qian et al. 2023) and interact with external knowledge to conduct a fair and comprehensive evaluation.

In this work, we focus on the assessment of knowledge fusion under a standard RAG setting. Facing difficulties in acquiring the parametric knowledge, we instead collect data to enrich LLMs' knowledge, enabling controlled and quantifiable evaluation of knowledge fusion. We split the data into two partitions, one part serving as external knowledge $\left(K_{e}\right)$, and the other part integrated into the LLMs as parametric knowledge ( $K_{p}$ ) through training. In this way, we eliminate the inconsistencies in $K_{p}$ among different LLMs. Leveraging the collected information, we further employ LLM to generate relevant question-answer pairs, forming a standard QA dataset for subsequent training and evaluation.

Data Source Preparation. LLMs trained on cut-dated data are rarely exposed to domain-specific knowledge and have not yet encountered the latest information. Thus, the most recent and high-quality data sources are essential for building a viable dataset in our setting. Considering the swift evolution, variety, and annual surge of new products in the electronics domain, it emerges as a suitable source to measure knowledge fusion. Thus, we collect the data in the electronics domain spanning the preceding four years and utilize product introductory documents with detailed specifications serving as the primary source. Specifically, we collect over 500 mobile phone names from websites and execute online searches to collate multiple search

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-04.jpg?height=279&width=567&top_left_y=1823&top_left_x=1191)

Figure 3: The distribution of the number of the associated evidence per QA sample. results for each product. Then, document filtration is applied through empirical rules and manual review, preserving documents with unique product introductions. Since some documents are too lengthy, we dissect them at the granularity of paragraphs and sentences to extract varied relevant information for each product. In general, we filtered out 1,700 paragraphs from 5,000 paragraphs in 1,500 documents, where 900 paragraphs are used to construct external knowledge, while 800 paragraphs are trained in LLMs as parametric knowledge.

Dataset Construction. The overview of dataset construction pipeline is shown in Figure 2 To simulate the external and parametric knowledge fusion scenarios, we divide the collected data into latest and outdated data according to their released date ${ }^{3}$ We highlight that the outdated data may be learned by LLMs, while the latest data is less likely seen by LLMs due to cut-dated pre-training. We retain the latest data as external knowledge, while the outdated data as parametric knowledge, which is injected into LLM to enhance its parametric memory via fine-tuning. The evaluation of the LLMs' competence in fusing external and parametric knowledge focuses on its discriminating usage of external and parametric knowledge, as well as the correctness of the information utilized. To more clearly assess LLM capabilities, we use the partitioned latest and outdated data to develop respective QA evaluation datasets with sophisticated designed instructions. Specifically, we generate the QA pairs for four knowledge fusion scenarios as follows:

- Scenario $1\left(S_{1}\right)$ : We randomly select one or two snippets from the latest data to generate a QA pair with an LLM, and then the unrelated snippets are added as noise to the chosen snippets, creating the candidate knowledge for the generated QA pair.
- Scenario $2\left(S_{2}\right)$ : We randomly select one snippet from both the latest and outdated data and generate a QA pair based on the two snippets. Unrelated snippets are added as noise to form the candidate knowledge for the QA pair.
- Scenario $3\left(S_{3}\right)$ : The QA pair is generated solely based on the chosen snippets from outdated data, and noisy snippets are added to form candidate knowledge.
- Scenario $4\left(S_{4}\right)$ : We randomly select one or two snippets from the latest data to create a QA pair, then discard the snippets and choose unrelated noise snippets as candidate knowledge for the QA pair, ensuring the unanswerability of the generated question.

To better align with real-world application settings, we adopt two noise introduction approaches to increase the challenge for LLMs in leveraging external knowledge. The first approach introduces noise snippets describing identical attributes across different electronic products, whereas the second approach presents noise snippets describing disparate attributes of a single electronic product. Moreover, to guarantee data quality, we further employ LLM evaluation coupled with manual review for data cleansing and filtering, yielding $210,580,200$, and 140 samples per scenario.

Dataset Analysis. We first assess the distribution of external evidence associated with each entity. Entities with three pieces of evidence are most prevalent, constituting $19.5 \%$ of the sample, followed by entities with a single piece of evidence, accounting for $17.5 \%$ of the total. We also examine the distribution of associated evidence per sample, depicted in Figure 3 No-

| Data Split | $S_{1}$ | $S_{2}$ | $S_{3}$ | $S_{4}$ | Total |
| :---: | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 390 | 90 | 50 | 630 |
| dev | 50 | 150 | 50 | 50 | 300 |
| test | 60 | 140 | 60 | 40 | 300 |

tably, $47.8 \%$ of samples contained five pieces of

Table 1: The statistics of the dataset in each scenario. evidence, the highest proportion, while samples with three pieces of evidence comprised the second-highest percentage at $18 \%$. Subsequently, we analyze the distribution of evidence lengths across the dataset, finding that quotes ranging from 500 to 600 characters represented the majority, totaling $78.6 \%$. The overview of dataset partitioning is shown in Table 1 The dataset comprises training, validation, and test sets with 630,300 , and 300 samples, respectively. The table also details the distribution of data across scenarios $S_{1}, S_{2}, S_{3}$, and $S_{4}$ within each subset.

## 5 Experiment Setup

Backbone Model. We select the open-source ChatGLM3-6B (Du et al., 2022) and Qwen-7B (Bai et al., 2023), and the black-box GPT-4 (OpenAI, 2023) as the backbones. These models are selected for their robust language understanding and instruction-following capabilities, which align well with our experiment design. Furthermore, the open-source LLMs enable flexible adaptation of model configurations and the analysis of their internal behaviors.[^2]

Table 2: Knowledge infusion results of ChatGLM Du et al. (2022) and Qwen Bai et al. (2023).

| Model | Accuracy (\%) | Coverage |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  |  | Complete $(\%)$ | Partial (\%) | Uncover (\%) |
| ChatGLM | 13.3 | 3.3 | 25.0 | 71.7 |
| ChatGLM $_{\mathrm{CT}}$ | 38.3 | 18.3 | 36.7 | 45.0 |
| Qwen | 15.0 | 5.0 | 21.7 | 73.3 |
| Qwen $_{\mathrm{CT}}$ | 43.3 | 20.0 | 43.3 | 36.7 |

Parametric Knowledge Infusion. For the selected LLMs, the outdated data portion needs to be injected into them via continued training or fine-tuning 4 . Since LLMs predominantly acquire knowledge in the pre-training phase, we also employ the same strategy for continued training. However, Allen-Zhu and Li (2023a) indicates that "memorization of knowledge" in language models merely means the model can fit the exact training data but does not imply it can extract the knowledge flexibly from data after training. To enhance knowledge memorization, we further adopt the data rewriting strategy suggested in Allen-Zhu and $\mathrm{Li}$ (2023a) to conduct data augmentation. Specifically, we use GPT-4 (OpenAI, 2023) to paraphrase the snippets in the outdated data portion and generate eight QA pairs related to that snippet as the supplementary data. The synthetic data is merged with the original data to train the backbones.

Evaluation Metrics. We employ accuracy ( $R_{\text {acc }}$ ) and information coverage ( $R_{\text {cover }}$ ) as evaluation metrics to access the knowledge fusion capabilities of LLMs. Accuracy assesses if LLM responses accurately address the question and align with both external and parametric knowledge sources. Responses are deemed correct if consistent with these sources and incorrect if they include irrelevant content or deviate from the information provided. Information coverage refers to the degree to which LLMs encapsulate the core content of the reference. This coverage is classified into three categories: complete, partial, and no inclusion. Let $K_{\text {gen }}$ denote the knowledge contained in generations, $K_{\text {gold }}$ indicate the knowledge contained in the ground-truth answer, and $K_{\text {ref }}$ represents the given external and parametric knowledg $6^{5}$. The $R_{\text {acc }}$ and $R_{\text {cover }}$ are computed as follows:

$$
R_{\text {acc }}=\left\{\begin{array}{ccc}
1 & \text { if } & K_{\text {gen }} \subseteq K_{\text {ref }} \\
0 & & \text { otherwise }
\end{array}, \quad \text { (2) } \quad R_{\text {cover }}=\left\{\begin{array}{lll}
\text { Complete } & \text { if } & K_{\text {gold }} \subseteq K_{\text {gen }}  \tag{3}\\
\text { Partial } & \text { if } & K_{\text {gold }} \cap K_{\text {gen }} \\
\text { Uncover } & & \text { otherwise }
\end{array}\right.\right.
$$

## 6 Experiment Results and Analysis

In this section, we conduct comprehensive experiments and in-depth analysis to investigate the knowledge fusion behaviors of various backbones. We use ChatGLM ${ }_{\text {CT }}$ to represent ChatGLM that continues trained on $K_{p}$, and ChatGLM ${ }_{\mathrm{CT} \& S F T}$ denotes ChatGLM that continues trained on $K_{p}$ and

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-06.jpg?height=41&width=919&top_left_y=1844&top_left_x=370)

### 6.1 The Performance of Knowledge Infusion

To investigate the effectiveness of knowledge infusion by LLMs at continued training process, we conduct experiments using two models: ChatGLM3-6B ( $a b b r$. ChatGLM) and Qwen-7B ( $a b b r$. Qwen). These models are trained using the designated parametric knowledge partition, $K_{p}$. The evaluation involved querying the models with questions specifically related to $K_{p}$ to assess how well the models retained the trained knowledge. It is important to note that this evaluation is similar to scenario $S_{3}$, absent the inclusion of external knowledge distractors. We adopt the question-answer (QA) pairs from $S_{3}$ by excluding the associated external evidence to serve as our evaluation dataset.

The results, summarized in Table 2, reveal that before continued training, both models demonstrated notably low accuracy rates: $13.3 \%$ for ChatGLM and $15.0 \%$ for Qwen, suggesting these models[^3]

Table 3: The overall performance of different LLMs under four scenarios. "Direct" represents directly prompting LLM to answer the questions by giving the corresponding external knowledge without continued training and supervised fine-tuning; "SFT" denotes the supervised fine-tuning on the train set of our constructed question-answering dataset; and "CT" means continuing training on the $K_{p}$ data partition to inject the knowledge into the LLM; "Easy" denotes the supervised fine-tuning on the train set as well as providing the supporting snippets during inference.

| Scenario | Metric | GPT-4 | ChatGLM |  |  |  | Qwen |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | Direct | SFT | $\mathrm{CT} \& \mathrm{SFT}$ | Easy | SFT | CT\&SFT |
| $S_{1}$ | $R_{\mathrm{acc}}(\%)$ | 81.7 | 63.3 | 68.3 | 61.7 | 72.7 | 62.9 | 63.3 |
|  | Complete $(\%)$ | 80.0 | 38.3 | 38.3 | 33.3 | 43.3 | 31.7 | 30.0 |
|  | Partial (\%) | 11.7 | 40.0 | 48.3 | 55.0 | 35.0 | 56.7 | 53.3 |
|  | Uncover (\%) | 8.3 | 21.7 | 13.4 | 11.7 | 21.7 | 11.6 | 16.7 |
| $S_{2}$ | $R_{\mathrm{acc}}(\%)$ | 35.7 | 39.3 | 52.1 | 53.6 | 72.1 | 49.3 | 57.1 |
|  | Complete $(\%)$ | 12.9 | 9.3 | 7.1 | 20.0 | 42.1 | 10.0 | 22.1 |
|  | Partial (\%) | 40.0 | 56.4 | 76.4 | 69.3 | 40.0 | 71.5 | 61.4 |
|  | Uncover (\%) | 47.1 | 34.3 | 16.5 | 10.7 | 17.9 | 18.5 | 16.5 |
| $S_{3}$ | $R_{\mathrm{acc}}(\%)$ | 8.3 | 10.0 | 16.7 | 35.0 | 78.3 | 20.0 | 33.3 |
|  | Complete (\%) | 3.3 | 1.7 | 3.3 | 16.7 | 55.0 | 3.3 | 20.0 |
|  | Partial (\%) | 11.7 | 23.3 | 40.5 | 45.0 | 30.0 | 48.3 | 41.7 |
|  | Uncover (\%) | 85.0 | 75.0 | 56.2 | 38.3 | 15.0 | 48.4 | 38.3 |
| $S_{4}$ | $R_{\mathrm{acc}}(\%)$ | 37.5 | 25.0 | 30.0 | 40.0 | - | 27.5 | 40.0 |

indeed have no such background knowledge. After continued training, there was a substantial enhancement in performance. Specifically, ChatGLM exhibits a $25 \%$ absolute improvement in accuracy, while Qwen shows a $28.3 \%$ absolute increase. This significant enhancement underscores the efficacy of continued training in injecting the knowledge into the models. Meanwhile, there is a notable enhancement in the model's ability to answer questions with complete and partial accuracy after continued training. Specifically, ChatGLM displays increases of $15 \%$ and $11.7 \%$ in complete and partial correct responses respectively, whereas Qwen showed improvements of $15 \%$ and $21.6 \%$.

Ideally, knowledge infusion through continued training of an LLM should enable the model to retain all imparted knowledge, resulting in the QA accuracy nearing $100 \%$. In practice, however, even though accuracy significantly improves over untrained models, it remains considerably lower than the optimal situation. This suggests substantial amounts of knowledge are either not retained or not accurately elicited by the LLM. We highlight two key factors attributed to this issue: (i) model capability and (ii) dataset diversity. For the model capability, recent studies (Allen-Zhu and Li, 2023a b) highlight that LLM faces difficulties using its parametric knowledge, and processing such knowledge does not guarantee it to be elicited accurately. For the dataset diversity, the LLM simply memorizes the given knowledge, meaning it only fits the given contents and may not effectively utilize this knowledge. For instance, LLM is trained in a massive of documents during continued training and evaluated under the question-answering manner at test time, LLM may not effectively map the questions to the answers learned during training. Besides, altering the way questions are posed might prevent the LLM from providing correct answers, and the reversal curse (Berglund et al. 2024) is another example of such an issue. Thus, a straightforward solution is to diversify the given knowledge, such as constructing various types of QA pairs, paraphrasing the documents, etc., that training LLM to memorize the knowledge from different perspectives.

### 6.2 Main Results

In this section, we conduct a comprehensive evaluation of different LLMs over the four knowledge fusion scenarios. The results are summarized in Table 3 . Note "Direct" mode denotes that we directly prompt LLM to answer the questions by giving the corresponding external knowledge without continued training or supervised fine-tuning; "SFT" mode represents that we supervised fine-tune our constructed train set; "CT\&SFT" mode denotes that we continue training the LLM on $K_{p}$ following by further supervised fine-tune on the constructed train set; and "Easy" mode means that we not only

SFT the LLM on the constructed train set but also provide the ground-truth snippets coupled with distractors during inference.

### 6.2.1 Knowledge Fusion Performance on $S_{1}$

Scenario $1, S_{1}$, denotes that provided external knowledge, $K_{e}$, alone is sufficient to answer a question, independent of $K_{p}$ 's contribution. The $S_{1}$ results of the different models are summarized in Table 3 Observed that GPT-4 achieves the best performance among all models, which obtains $81.7 \%$ accuracy and $66.7 \%$ complete coverage. The higher accuracy usually leads to better "complete" and/or "partial" coverage. Compared to ChatGLM and Qwen, GPT-4 has a richer internal knowledge base and more powerful content comprehension capabilities. For ChatGLM, ChatGLM $\mathrm{SFT}$ is superior to

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=44&width=1385&top_left_y=669&top_left_x=370)
on accuracy. Notably, the continued training does not always contribute to the performance improve-

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=41&width=1385&top_left_y=744&top_left_x=370)

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=41&width=1379&top_left_y=782&top_left_x=373)
highlight it is because all the ground-truth evidence is provided by external knowledge in $S_{1}$, SFT helps LLM to learn how to follow the instructions and utilize the given knowledge to reach the correct responses. The knowledge provided by $\mathrm{CT}$ is useless in the $S_{1}$ scenario, and continued training may inevitably lead to capability degradation of LLM (Shi et al. 2024). Nevertheless, ChatGLM $\mathrm{SFT}^{\text {is }}$ inferior to ChatGLM ${ }_{\text {Easy }}$ with a distinct gap, i.e., $68.3 \%$ versus $72.2 \%$, which demonstrates that noisy external knowledge indeed affects LLMs adversely (Pan et al. 2024, Cuconasu et al., 2024). Notably, the noise in our dataset is carefully curated, being relevant yet useless for effective responses.

### 6.2.2 Knowledge Fusion Performance on $S_{2}$

Scenario $2, S_{2}$, represents that $K_{e}$ provides partial knowledge to answer a question, and it requires $K_{p}$ to fill the gaps for a complete answer. As summarized in Table 3, the overall performance of different backbone models in $S_{2}$ is significantly inferior to that in $S_{1}$. For instance, although the test cases are different, the accuracy of GPT- 4 drops from $81.7 \%$ to $35.7 \%$ and the complete coverage drops from $80.0 \%$ to $12.9 \%$, which proves that our data partition, i.e., $K_{e}$ and $K_{p}$, is reasonable, indicating

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=43&width=1385&top_left_y=1388&top_left_x=367)
ChatGLM $\mathrm{SFT}$ achieve much better performance, $52.1 \%$ versus $39.3 \%$. Since SFT does not inject $K_{p}$

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=38&width=1382&top_left_y=1469&top_left_x=369)

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=46&width=1385&top_left_y=1498&top_left_x=367)
ChatGLM $\mathrm{SFT}$ by a large margin, which indicates that CT indeed injects the $K_{p}$ into the LLMs, the LLMs are capable of using the knowledge to answer the questions by considering both its parametric knowledge and the given external knowledge. A similar observation is held for Qwen model.

However, the accuracy of CT\&SFT is slightly better than that of SFT for both ChatGLM and Qwen models. We emphasize that there are two aspects to this issue. One primary factor is the model's memory capacity, which determines the extent of knowledge retained during training. As discussed in Section 6.1, due to limitations in model capacity and dataset diversity, the model can accurately retain only a subset of the provided $K_{p}$. Another factor is that LLMs face difficulties using their parametric knowledge (Allen-Zhu and $\mathrm{Li}, 2023 \mathrm{ab}$ ) and accurate parametric and external knowledge fusion for question answering is challenging. According to case studies in $S_{2}$, we observe that the success rate of LLM's parametric knowledge elicitation is only around $60 \%$. If we directly use all the ground-truth supporting snippets as external knowledge and feed them into LLMs, the LLMs' performance increases significantly (see "Easy" and "CT\&SFT"), which further proves the deficiency of LLMs to utilize their parametric knowledge. Some work (Jeong et al. 2024, Ding et al. 2024) performs parametric and external knowledge fusion by first producing partial answers using parametric knowledge and then integrating the generated knowledge and external knowledge for final answer generation. In contrast, we directly prompt LLM to generate the final answer by considering its parametric knowledge and the given external knowledge.

### 6.2.3 Knowledge Fusion Performance on $S_{3}$

Note that scenario $3, S_{3}$, simulates the situation that $K_{e}$ offers no useful information and the correct answer depends solely on $K_{p}$. As reported in Table 3, without SFT or CT, all the evaluated backbone

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=41&width=1379&top_left_y=2430&top_left_x=373)

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-08.jpg?height=44&width=1385&top_left_y=2469&top_left_x=370)

SFT only teaches LLM to follow the instruction for answer generation, while it does not inject the new knowledge into the LLM. After injecting the $K_{p}$ into LLM, ChatGLM ${ }_{\text {CT\&SFT }}$ significantly outperforms ChatGLM ${ }_{\mathrm{SFT}}$ in accuracy by $18.3 \%$ absolute improvement. A similar result is observed for the Qwen model, which obtains $13.3 \%$ absolute gains. Despite the improvements achieved, their

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-09.jpg?height=43&width=1385&top_left_y=407&top_left_x=370)
$43.3 \%$ higher than ChatGLM ${ }_{\text {CT\&SFT }}$. Similar to the observation in Section 6.2.2, the results indicate that CT cannot guarantee that LLM will fully retain all knowledge, and LLM itself faces difficulties in accurately eliciting parametric knowledge. Moreover, in the knowledge infusion experiment (ref. Section6.1, we use the same QA pairs as $S_{3}$, but we ignore all the external distractors. Comparing the results between knowledge infusion (see Table 2) and $S_{3}$ knowledge fusion (see Table 3), we observe that the accuracies of both ChatGLM and Qwen in $S_{3}$ knowledge fusion are lower than that in knowledge infusion, which emphasizes that incorporating noisy external knowledge negatively impacts LLM performance, as it may cause overconfidence in plausible but incorrect information.

### 6.2.4 Knowledge Fusion Performance on $S_{4}$

Recall that scenario $4, S_{4}$, describes cases where neither $K_{e}$ nor $K_{p}$ adequately address the questions, making those questions theoretically unanswerable. This scenario aims to evaluate the efficacy of LLMs to correctly provide a refusal response if they do not have the corresponding parametric knowledge and the external knowledge is unhelpful. As reported in Table 3 , all the evaluated backbone models, including GPT-4, fail to trigger the refusal response under $S_{4}$. In general, these models tend to be overconfident in the provided plausible but incorrect external knowledge, yielding wrong answers. SFT shows positive impacts on performance improvement, where ChatGLM $\mathrm{SFT}$ is $5 \%$

![](https://cdn.mathpix.com/cropped/2024_06_04_67832f9f8c6ffdf38a9cg-09.jpg?height=46&width=1385&top_left_y=1121&top_left_x=370)
can guide the LLM on how to trigger and issue refusals to some extent. Comparing ChatGLM ${ }_{C T \& S F T}$ with ChatGLM $\mathrm{SFT}$, CT further boosts the performance. We speculate that continued training with domain knowledge improves the LLM's field-specific understanding, enhancing its ability to discern whether the provided external knowledge and its parametric knowledge can effectively address a given question.

### 6.3 Findings and Challenges

According to the in-depth analyses presented in Section 6.1 and 6.2 , we conclude the observations and insights as follows:

- Noise Robustness of LLM: Noise and interference information from external knowledge negatively impact LLM performance (Chen et al., 2024), as evidenced across multiple LLMs in scenarios $S_{1} \sim S_{4}$, leading to the generation of seemingly plausible but incorrect answers.
- Impact of supervised fine-tuning: Across all scenarios, $S_{1} \sim S_{4}$, supervised fine-tuning (SFT) helps to improve the performance of LLMs. Despite SFT (almost) does not inject new knowledge into the LLM, it enhances the LLM's ability in instruction adherence, leading to more standardized outcomes (Allen-Zhu and Li 2023a b).
- Impact of continued training: when external knowledge is sufficient, i.e., $S_{1}$, domain knowledge infusion via continued training yields negligible improvement, as the LLM can generate correct answers based solely on the provided information. Conversely, when external knowledge is inadequate, i.e., $S_{2} \sim S_{4}$, continued training is crucial and significantly enhances performance (Jiao et al., 2023, Naveed et al., 2024, Fujii et al., 2024), since LLM lacks the necessary domain knowledge and continued training can effectively alleviate the knowledge limitations of LLM.
- The effect of knowledge infusion: Although experiments on knowledge infusion and $S_{2} \sim S_{3}$ demonstrate the effectiveness, performance gains of knowledge infusion remain limited. Due to constraints in model capacity and dataset diversity, LLMs can retain only a subset of the knowledge accurately via continued training (Moiseev et al., 2022, Arrotta et al. 2024). Additionally, LLMs also struggle to utilize parametric knowledge effectively, and processing such knowledge does not ensure accurate elicitation (Allen-Zhu and Li. 2023b).
- The effect of refusal: Ideally, LLM should issue a refusal response when external knowledge is irrelevant and lacks corresponding parametric knowledge. However, LLMs tend to
be overconfident in external knowledge regardless of its usefulness (Chen et al., 2024), particularly for cases like $S_{3} \sim S_{4}$ where external knowledge is entirely unhelpful, leading to plausible but incorrect responses (hallucinations). SFT and CT ameliorate this issue. SFT provides examples of refusal responses in the training data, instructing the LLM when to refuse. Meanwhile, CT enhances the LLM's understanding of domain knowledge, improving its ability to judge the efficacy of both external and parametric knowledge in addressing a given question.
- The effect of knowledge fusion: When the external knowledge is incomplete, LLMs often struggle to effectively fuse parametric and external information for response generation (Xie et al., 2023). Efficient fusion is generally constrained by factors such as the LLM's knowledge capacity, knowledge boundary perception, noise resistance, and knowledge elicitation ability (Wang et al. 2023c).

Accordingly, to better fuse parametric and external knowledge in LLMs, we identify several key challenges that need addressing. While some work (Allen-Zhu and Li, 2023a b, Chen et al., 2024; Wang et al., 2023c; Xie et al. 2023) are underway to tackle these issues, the approach from the perspective of knowledge fusion remains underexplored.

- With respect to the noisy information, how to eliminate noise in external knowledge and enhance the noise resistance ability of LLMs, especially in the absence of corresponding parametric knowledge?
- For knowledge infusion, how to optimize the training strategies or methodologies so that the LLM can retain as much knowledge as possible?
- How can LLMs elicit the correct parametric knowledge to answer given questions and accurately recognize its knowledge boundaries, triggering a refusal when neither parametric nor external knowledge is available, rather than generating a hallucinated response?
- How can we optimize the use of parametric and external knowledge to achieve accurate integration when external knowledge is incomplete and the LLM has corresponding default knowledge?


## 7 Conclusion

This work underscores the nuanced interplay between external and parametric knowledge within LLMs, emphasizing the potential and challenges intrinsic to their fusion. By meticulously deconstructing knowledge fusion into four distinct scenarios and developing a structured pipeline for data construction and knowledge infusion, we have provided a comprehensive examination of LLM behavior across varying contexts of knowledge supplementation. The results indicate that while supervised fine-tuning or enhancing parametric knowledge via continued training is capable of improving the knowledge fusion performance, persistent challenges remain in noise resistance, more effective knowledge infusion, parametric knowledge boundary perception, and accurate knowledge elicitation. These insights lay a foundational framework for future research aimed at achieving a more harmonious and effective synthesis of external and parametric knowledge within LLMs, ultimately advancing their capabilities and applications.

## References

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. ArXiv, abs/2309.14316, 2023a.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.2, knowledge manipulation. ArXiv, abs/2309.14402, 2023b.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. ArXiv, abs/2305.10403, 2023.

Luca Arrotta, Claudio Bettini, Gabriele Civitarese, and Michele Fiori. Contextgpt: Infusing llms knowledge into neuro-symbolic activity recognition models. ArXiv, abs/2403.06586, 2024.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. ArXiv, abs/2310.11511, 2023.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenhang Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, K. Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Yu Bowen, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xing Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. ArXiv, abs/2309.16609, 2023.

Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". ArXiv, $\mathrm{abs} / 2309.12288,2023$.

Lukas Berglund, Meg Tong, Maximilian Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: LLMs trained on "a is b" fail to learn "b is a". In The Twelfth International Conference on Learning Representations, 2024.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 2206-2240. PMLR, 2022.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. ArXiv, abs/2309.01431, 2023.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. Proceedings of the AAAI Conference on Artificial Intelligence, 38 (16):17754-17762, Mar. 2024.

Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. The power of noise: Redefining retrieval for rag systems. ArXiv, abs/2401.14887, 2024.

Yujuan Ding, Wenqi Fan, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meets llms: Towards retrieval-augmented large language models. ArXiv, $\mathrm{abs} / 2405.06211,2024$.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages $320-335,2022$.

Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, and Naoaki Okazaki. Continual pre-training for cross-lingual llm adaptation: Enhancing japanese language capabilities. ArXiv, abs/2404.17790, 2024.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. ArXiv, $\mathrm{abs} / 2312.10997,2023$.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346-361, 2021a.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484-5495, 2021b.

Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1350-1370, 2023.

Hangfeng He, Hongming Zhang, and Dan Roth. Rethinking with retrieval: Faithful large language model inference. ArXiv, abs/2301.00303, 2022.

Linmei Hu, Zeyi Liu, Ziwang Zhao, Lei Hou, Liqiang Nie, and Juanzi Li. A survey of knowledge enhanced pre-trained language models. IEEE Transactions on Knowledge and Data Engineering, 2023.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning with retrieval augmented language models. ArXiv, abs/2208.03299, 2022.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. ArXiv, $\mathrm{abs} / 2403.14403,2024$.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. ArXiv, abs/2305.06983, 2023.

Fangkai Jiao, Bosheng Ding, Tianze Luo, and Zhanfeng Mo. Panda llm: Training data and evaluation for open-sourced chinese instruction-following large language models. ArXiv, abs/2305.03025, 2023 .

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know. ArXiv, abs/2207.05221, 2022.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to learn long-tail knowledge. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 1569615707. PMLR, 2023.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466, 2019.

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. Internetaugmented language models through few-shot prompting for open-domain question answering. ArXiv, abs/2203.05115, 2022.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459-9474. Curran Associates, Inc., 2020.

Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar. Large language models with controllable working memory. ArXiv, $\mathrm{abs} / 2211.05110,2022$.

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources. In The Twelfth International Conference on Learning Representations, 2024.

Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. ArXiv, abs/2310.01352, 2023.

Alisa Liu, Zhaofeng Wu, Julian Michael, Alane Suhr, Peter West, Alexander Koller, Swabha Swayamdipta, Noah A Smith, and Yejin Choi. We're afraid language models aren't modeling ambiguity. ArXiv, abs/2304.14399, 2023a.

Yi Liu, Lianzhe Huang, Shicheng Li, Sishuo Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Recall: A benchmark for llms robustness against external counterfactual knowledge. ArXiv, abs/2311.08147, 2023b.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. ArXiv, abs/2308.08747, 2023.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. ArXiv, abs/2212.10511, 2022.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. In Advances in Neural Information Processing Systems, volume 35, pages $17359-17372,2022$.

Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. Augmented language models: a survey. ArXiv, abs/2302.07842, 2023.

Fedor Moiseev, Zhe Dong, Enrique Alfonseca, and Martin Jaggi. Skill: Structured knowledge infusion for large language models. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages $1581-1588,2022$.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. ArXiv, abs/2112.09332, 2021.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models. ArXiv, abs/2307.06435, 2024.

OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.

Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. Fine-tuning or retrieval? comparing knowledge injection in llms. ArXiv, abs/2312.05934, 2023.

Liangming Pan, Wenhu Chen, Min-Yen Kan, and William Yang Wang. Contraqa: Question answering under contradicting contexts. ArXiv, abs/2110.07803, 2021.

Ruotong Pan, Boxi Cao, Hongyu Lin, Xianpei Han, Jia Zheng, Sirui Wang, Xunliang Cai, and Le Sun. Not all contexts are equal: Teaching llms credibility-aware generation. ArXiv, abs/2404.06809, 2024.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. Language models as knowledge bases? In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2463-2473, 2019.

Cheng Qian, Xinran Zhao, and Sherry Tongshuang Wu. "merge conflicts!" exploring the impacts of external distractors to parametric knowledge graphs. ArXiv, abs/2309.08594, 2023.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, dahai li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In The Twelfth International Conference on Learning Representations, 2024 .

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for SQuAD. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784-789, 2018.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. ArXiv, abs/2302.00083, 2023.

Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong Wen, and Haifeng Wang. Investigating the factual knowledge boundary of large language models with retrieval augmentation. ArXiv, abs/2307.11019, 2023a.

Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong Wen, and Haifeng Wang. Investigating the factual knowledge boundary of large language models with retrieval augmentation. arXiv, abs/2307.11019, 2023b.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems, 2023 .

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. ArXiv, abs/2303.17580, 2023.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, and Hao Wang. Continual learning of large language models: A comprehensive survey. ArXiv, abs/2404.16789, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023.

Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, et al. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. ArXiv, abs/2310.07521, 2023a.

Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Resolving knowledge conflicts in large language models. ArXiv, abs/2310.00935, 2023b.

Yile Wang, Peng Li, Maosong Sun, and Yang Liu. Self-knowledge guided retrieval augmentation for large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10303-10315, 2023c.

Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, and Yingchun Wang. Fake alignment: Are llms really aligned well? ArXiv, abs/2311.05915, $2023 \mathrm{~d}$.

Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Unraveling the behavior of large language models in knowledge conflicts. ArXiv, abs/2305.13300, 2023.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In The Twelfth International Conference on Learning Representations, 2024.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. ArXiv, abs/2401.15884, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369-2380, 2018.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. Do large language models know what they don't know? In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 8653-8665, 2023a.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. Do large language models know what they don't know? In Findings of the Association for Computational Linguistics: ACL 2023, pages 8653-8665. Association for Computational Linguistics, 2023b.

Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context. ArXiv, abs/2310.01558, 2023.

Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, et al. A comprehensive study of knowledge editing for large language models. ArXiv, abs/2401.01286, 2024.

Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and $\mathrm{Lu}$ Wang. Merging generated and retrieved knowledge for open-domain QA. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4710-4728, 2023a.

Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. Merging generated and retrieved knowledge for open-domain QA. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4710-4728, Singapore, 2023b. Association for Computational Linguistics.


[^0]:    * The first two authors contributed equally.

[^1]:    ${ }^{2}$ For simplicity throughout this paper, we use $K_{e}$ and $K_{p}$ to symbolize the retrieved external knowledge and the LLMs' parametric knowledge, respectively.

[^2]:    ${ }^{3}$ The latest data is orthogonal to outdated data, i.e., no information overlap between them. Besides, the latest data refers to factual information about electronic products occurring after 2023-06-01, unseen by existing LLMs.

[^3]:    ${ }^{4}$ Given the inaccessibility of GPT-4's weights, we assume it already memorizes outdated data and only conducts inference by providing external knowledge snippets as evidence. GPT-4-0613 is used in all experiments.

    ${ }^{5}$ When evaluating LLMs' competence in $S_{4}$, we only measure their capability to decline to respond correctly.

</end of paper 1>


<paper 2>
# Non-Vacuous Generalization Bounds for Large Language Models 

Sanae Lotfi*<br>Tim G. J. Rudner<br>Marc Finzi ${ }^{*}$<br>Micah Goldblum<br>New York University

Yilun Kuang*

Andrew Gordon Wilson


#### Abstract

Modern language models can contain billions of parameters, raising the question of whether they can generalize beyond the training data or simply regurgitate their training corpora. We provide the first non-vacuous generalization bounds for pretrained large language models (LLMs), indicating that language models are capable of discovering regularities that generalize to unseen data. In particular, we derive a compression bound that is valid for the unbounded log-likelihood loss using prediction smoothing, and we extend the bound to handle subsampling, making bound computation 900 times faster on massive datasets. To achieve the extreme level of compression required for non-vacuous bounds, we devise SubLoRA, a simple low-dimensional nonlinear parameterization that leads to non-vacuous generalization bounds for very large models with up to 849 million parameters. Finally, we use our bounds to understand LLM generalization and find that larger models have better generalization bounds and are more compressible than smaller models.


## 1 Introduction

Do large language models (LLMs) merely memorize the training data, and if so, are they able to meaningfully generalize beyond their training set? This question is central to understanding LLMs as they continue to grow in capacity and are capable of memorizing and regurgitating training examples verbatim (Brown et al., 2020; Chowdhery et al., 2022; Carlini et al., 2020, 2023).

In this work, we address the question of generalization in LLMs by computing the first non-vacuous generalization bounds for language model pretraining on next token prediction, thereby providing a mathematical guarantee that LLMs are able to generalize beyond their training data.

Although significant progress has been made in constructing non-vacuous generalization bounds for image classification models using the PAC-Bayes framework (Catoni, 2007) in conjunction with extreme levels of model compression (Zhou et al., 2019; Lotfi et al., 2022), non-vacuous generalization bounds for large language models remain elusive.

Compared to image classification models, constructing non-trivial bounds for language models presents additional challenges: (i) LLMs are trained on autoregressive token prediction, and thus token predictions are not independent; (ii) the relevant negative log-likelihood (NLL) metric (bits per dimension) is a continuous and unbounded random variable for which previously used non-vacuous PAC-Bayes bounds are invalid; and (iii) LLMs have orders of magnitude more parameters than image classification models. To address these challenges, we derive new generalization bounds that can be applied to the unbounded bits per dimension[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-02.jpg?height=448&width=1370&top_left_y=242&top_left_x=366)

Figure 1: Finding solutions that simultaneously achieve low training error and low complexity with SubLoRA. (Left): The Pareto frontier of model complexity (the 2nd term in Equation 1) and the empirical risk (bits per dimension (BPD) and Top-1 Error) of language models using LoRA and subspace compression for next token prediction pretraining. The generalization bound is formed from the sum of the two axes (lower is better), with the shaded region showing where bounds are vacuous. Combining both LoRA and subspace compression in the form of SubLoRA yields the best bounds, while using LoRA alone yields vacuous bounds for top-1 error. (Right): SubLoRA enables a smooth tradeoff over the extent of model compression for a fixed model, finding the degree of compression that is optimal for the situation in constructing the generalization bounds. We plot the contributions of the empirical risk and the complexity term to the bound as a function of this degree of compression.

objective. We also introduce an extension of these bounds which can be computed using only a subset of the training data, making bound computation 900 times faster on the OpenWebText dataset, which contains more than 9 billion tokens.

Achieving the extreme level of compression required to obtain non-vacuous generalization bounds for LLMs is another challenge. To this end, we devise SubLoRA (Subspace-Enhanced Low-Rank Adaptation): simple nonlinear parameterization for LLMs that makes it possible to smoothly vary the level of compression while maintaining expressivity. SubLoRA combines low-rank adaptation (LoRA) (Hu et al., 2021), originally proposed for efficient fine-tuning, with subspace training (Li et al., 2018; Lotfi et al., 2022) to pretrain highly compressible LLMs from scratch.

Combining the above-described theoretical and practical contributions, we achieve the first non-vacuous bounds for large language models. To highlight the efficiency of our new compression technique, we compare SubLoRA to LoRA and subspace training in Figure 1 (left). We compute two metrics that we define as follows: Top-1 Error, which is the 0-1 error in predicting the next token averaged over a given document; and the bits per dimension metric, which corresponds to the average negative log-likelihood per document. The shaded region highlights where bounds become vacuous, with SubLoRA achieving non-vacuous bounds for both bits per dimension and Top-1 Error. In contrast, we see that only using LoRA achieves vacuous bounds for Top-1 Error and only using subspace achieves a high value of empirical BPD. Despite the simplicity of SubLoRA, it has an improved ability to trade-off model complexity with training error. In Figure 1 (right), we highlight the trade-off between model complexity and empirical risk in the generalization bounds as we vary the level of compression.

We summarize our contributions as follows:

- Novel bounds for the unbounded negative log-likelihood objective: we introduce novel bounds specifically tailored to account for the unbounded continuous bits-per-dimension loss, commonly used to evaluate LLMs for next-token prediction.
- Subsampling bounds for practical bound evaluation: To make the evaluation of the bounds practical on LLMs with massive datasets, we derive subsampling-based bounds that allow for efficient evaluation. In practice, the evaluation of the bound takes 45 minutes on a single GPU instead of 3 days on 8 GPUs in parallel for the OpenWebText dataset.
- A simple yet powerful nonlinear subspace compression for LLMs: as we show in Figure 1, using LoRA alone to compress the discrepancy between the random initialization and a learned model leads to vacuous bounds for the top-1 error. At the same time, linear subspace training alone does not unlock the full compression potential of LLMs compared to a nonlinear compression scheme. We show that a combination of these two approaches, while simple, yields a strong nonlinear compression of the model, which leads to the best generalization bounds for LLMs.
- Non-vacuous generalization bounds for models with nearly a billion parameters: our work does not only introduce the first non-vacuous generalization bounds for LLMs, but it also extends these bounds to models with over 800 million parameters, demonstrating the scalability of our compression technique.
- Improved understanding of generalization in LLMs: as we increase the size of models, we find that they are more compressible and achieve better bounds, therefore disproving the claim that larger LLMs are simply better at regurgitating their training data.

The significance of these contributions lies in the ability to offer mathematical proof that large language model are, in fact, powerful knowledge compressors and are capable of generalization beyond their training samples, especially as their scale increases. To the best of our knowledge, our work is the first to show that generalization bounds improve with more parameters on models of practical sizes, in line with the empirical benefits of large models.

## 2 Related Work

Generalization bounds. Neural networks have seen widespread adoption because of their strong performance on new unseen test samples, known as generalization. Early generalization theory literature bounded the difference in training and test error, called the generalization gap, using complexity measures like VC-dimension (Vapnik, 1991) and Rademacher complexity (Bartlett and Mendelson, 2002). These generalization bounds were vacuous for neural networks, which are often flexible enough to fit randomly labeled training data (Zhang et al., 2021). The flexibility of neural networks and its negative impact on these classical bounds calls into question why they generalize. Neural networks are so flexible that they have parameter vectors where they fit their training data and simultaneously assign incorrect labels to testing data, and they also have parameter vectors where they fit their training data and instead assign correct labels to the testing data. Why do such flexible models actually make correct test predictions in practice?

PAC-Bayes generalization theory bridges this gap by leveraging the fact that while neural networks are highly flexible and can fit random labels, they encode a preference for the correct ones (Catoni, 2007; Dziugaite and Roy, 2017; Arora et al., 2018). Unlike earlier generalization bounds which measured complexity merely as a function of the hypothesis class, PAC-Bayes generalization bounds reward models which have a strong prior that places its mass on parameter vectors that align with observed data. This formulation allows one to draw a parallel between generalization and compressibility (Zhou et al., 2019; Lotfi
et al., 2022). By placing disproportionate prior mass on compressible parameter vectors, achieving a tight bound simply requires finding a family of models (posterior) that well fit the training data. Such compression bounds achieve the tightest guarantees to date on modern convolutional architectures and large-scale datasets, showcasing the strong inductive bias of neural networks and indicating that they can significantly compress their training sets (Lotfi et al., 2022). While PAC-Bayes has proven a very fruitful framework for devising such bounds, the insight on using a prior to bound the complexity of a given model does not require a posterior and can actually be incorporated into simpler finite hypothesis bounds.

Recent generalization theory literature has expanded analysis to several relevant modelsautoregressive time-series models and simple n-gram language models (McDonald et al., 2011; Bharadwaj and Hasegawa-Johnson, 2014; Vankadara et al., 2022). In contrast, we construct bounds for autoregressive transformer-based language models.

Existing bounds for unbounded objectives. A number of works have explored techniques for generating generalization bounds on unbounded objective functions more generally, but these approaches are not practical for application to LLMs. A well established strategy relevant for e.g. linear regression with Gaussian errors is to bound the tails of the objective as subgaussian random variables, and then generalization bounds can be constructed for subgaussians more generally (Alquier et al., 2016; Germain et al., 2016). Other kinds of known tail behavior have also been exploited (Holland, 2019; Kuzborskij and Szepesvári, 2019). For the NLL of a language model, there is no clear analogous tail behavior, so we must take a different approach.

Haddouche et al. (2021) devise an approach for general unbounded objectives by constructing a hypothesis dependent bound on the objective, even if the objective is unbounded more generally. If the risk can be bounded $\sup _{x} R(h, x) \leq Q(h)$ for a function $Q(h)$, then PACBayes bounds can be constructed using $Q(h)$ even if $\sup _{h} Q(h)=\infty$. However, even though $Q(h)$ is finite for LLMs as there are only a finite number of inputs, $Q$ grows exponentially for NLL with the number of layers in the network and is closely related with the Lipschitz constant. For large models like LLMs, this value is far too large to be useful in constructing bounds.

Language models and compression. Large language models are parameterized with as many as billions of parameters and, as a result, have a significant memory footprint, which makes pretraining, finetuning, and even evaluation challenging without access to large-scale computing infrastructure. To reduce the memory footprint of large language models, a wide array of compression schemes has been proposed to enable evaluation, fine-tuning, and pretraining with limited computational resources. Low-Rank Adaptation (Hu et al., 2021, LoRA) freezes the pre-trained model weights and inserts trainable rank decomposition matrices into each attention layer of the transformer architecture used in large language models. Doing so allows for significantly reducing the number of trainable parameters for fine-tuning on downstream tasks. For example, LoRA can reduce the number of trainable parameters in GPT-3 175B fine-tuned with Adam by a factor of 10,000 and the GPU memory requirement by a factor of 3. Building on LoRA, Q-LoRA (Dettmers et al., 2023a) quantizes a pretrained model to 4-bits, adds a small set of learnable weights parameterized using LoRA, and then tunes these weights by backpropagating gradients through the quantized model. Other compression methods for large language models use distillation (Liu et al., 2023), sub-4-bit integer quantization (Kim et al., 2023; Park et al., 2022), sparse quantized representations that identify and isolate outlier weights (Dettmers et al., 2023b), weight quantization based on approximate second-order information (Frantal et al., 2022), or tensor-train decompositions (Xu et al., 2023).

Achieving a good generalization bound has distinct requirements from the existing compression literature. Unlike existing compression schemes for language models, which aim to accelerate inference and training or to reduce the memory footprint, we focus on specifying
the trained model parameters in only few bits, even if doing so decreases neither latency nor memory requirements.

## 3 Background

Subspace training. Lotfi et al. (2022) train a compressible model by parameterizing a carefully constructed low-dimensional random subspace. The weights $\theta \in \mathbb{R}^{D}$ are then defined as the sum of a random initialization $\theta_{0}$ and a projection $P \in \mathbb{R}^{D \times d}$ from a lowerdimensional subspace $w \in \mathbb{R}^{d}: \theta=\theta_{0}+P w . P$ is constructed as the Kronecker product of random Gaussian matrices $P=\left(Q_{1} \otimes Q_{2}\right) / \sqrt{D}$ for $Q_{1}, Q_{2} \sim \mathcal{N}(0,1)^{\sqrt{D} \times \sqrt{d}}$, normalized so that $P^{\top} P \approx I$. The weights $w$ can then be optimized over by backpropagating through the transformation. With a learned quantization strategy-optimizing over quantized weights and the quantization levels-Lotfi et al. (2022) use arithmetic coding to encode the weights using the empirical probabilities over quantization bins.

Low Rank Adaptation (LoRA). Similarly inspired by evidence that overparametrized models have low intrinsic dimensionality (Li et al., 2018; Aghajanyan et al., 2020), Hu et al. (2021) propose LoRA as a parameter-efficient finetuning method. Given a pretrained weight matrix $W_{\text {pretrained }} \in \mathbb{R}^{a \times b}$, LoRA decomposes its total update $\Delta W$ accumulated throughout finetuning as a product of two trainable low-rank matrices $U \in \mathbb{R}^{a \times r}, V \in \mathbb{R}^{r \times b}$ for $r \ll \min (a, b)$ while freezing $W_{\text {pretrained }}$. Thus $W_{\text {finetuned }}=W_{\text {pretrained }}+\Delta W=W_{\text {pretrained }}+U V$. In this work, we use LoRA for pretraining instead. In particular, we take randomly initialized neural network weights $W_{0} \in \mathbb{R}^{a \times b}$ and represent their update during pretraining as $U V$, yielding $W_{\text {pretrained }}=W_{0}+\Delta W=W_{0}+U V$. We decrease the dimensionality further by applying subspace projection to the LoRA matrices, which we describe in detail in Section 5.

## 4 Methodology

In constructing non-vacuous generalization bounds for LLMs, we expand and improve upon existing techniques in three ways: (1) we construct a simple and effective nonlinear parameterization which is more effective and scalable than purely linear subspaces; (2) we construct new bounds that can handle the continuous and unbounded nature of the negative log-likelihood; (3) we make these bounds more practical to compute with LLMs by deriving a new bound which holds even when the empirical risk is evaluated only on a small subsample of the full training dataset.

### 4.1 Finite Hypothesis Compression Based Generalization Bounds

Given a bounded risk $R(h, x) \in[a, a+\Delta]$ and a finite hypothesis space $h \in \mathcal{H}$ for which we have a prior $P(h)$, it is straightforward to derive a generalization bound relating the empirical risk $\hat{R}(h)=\frac{1}{m} \sum_{i=1}^{m} R\left(h, X_{i}\right)$ to the expected risk $R(h)=\mathbb{E}[\hat{R}(h)]$ so long as $\left\{X_{i}\right\}_{i=1}^{m}$ are sampled independently. With probability at least $1-\delta$, we have

$$
\begin{equation*}
R(h) \leq \hat{R}(h)+\Delta \sqrt{\frac{\log 1 / P(h)+\log 1 / \delta}{2 m}} \tag{1}
\end{equation*}
$$

We provide an elementary proof in Appendix A.1.

If the prior likelihood $P(h)$ of the found model $h$ can be increased (either by choosing a better prior, or by finding more likely hypotheses), then the generalization bound improves. Following Lotfi et al. (2022), we adopt the powerful but general Solomonoff prior $P(h) \leq$ $2^{-K(h \mid A)}$ (Solomonoff, 1964) where $K$ is the prefix Kolmogorov complexity of $h$, with the model architecture $A$ provided as input. While $K$ is not computable, it is possible to compute the upper bound

$$
\log 1 / P(h) \leq K(h \mid A) \log 2 \leq C(h) \log 2+2 \log C(h)
$$

where $C(h)$ is the compressed size of $h$ given any particular strategy for compressing $h$ and we may make use of the prior knowledge describing the architecture. Therefore, if we can find hypotheses $h$ that both have a low empirical risk and a small compressed size, then we can construct strong generalization bounds.

### 4.2 Enabling the Independence Assumption for Generalization Bounds on Text Data

Using Equation 1 requires that $X_{i}$ in the sum $\hat{R}(h)=\frac{1}{m} \sum_{i=1}^{m} R\left(h, X_{i}\right)$ are drawn independently. Thus, we must be careful in the construction and interpretation of our bounds so that this constraint is satisfied. Instead of considering bounds at the level of tokens, which are correlated, we instead define $X_{i}$ to be an entire document sampled uniformly from the data generating process from which the corpus was sampled. In other words, we break our dataset into its constituent documents and sample uniformly documents uniformly from it, where each $X_{i}$ represents an entire document. We define the risk on a given document as the negative log-likelihood of the entire document divided by its length, according to the autoregressive model.

It is also possible to choose $X_{i}$ to be a context chunk, i.e., a sequence of length equal to the context length, as is commonly used in the training of models since a document may be larger than the maximum transformer context length. In such cases, the sequences are no longer independent samples from the data generating process. It is possible to construct valid bounds on these sequences which respect the independence assumption. However, in doing so we must shift the interpretation of the bounds from being over the randomness in sampling from the data generating process to the randomness in sampling sequences that can be constructed from a fixed and finite dataset formed by concatenating the documents together. We explore these alternate sequence-level bounds in Appendix B. However, we believe that the document-level bounds provide a more meaningful and significant statement about generalization.

### 4.3 Accommodating the Unbounded Negative Log-Likelihood Objective Using Prediction Smoothing

The primary metric for pretraining of large language models, as for other autoregressive models, is the negative log-likelihood (NLL), or bits per dimension (BPD), of the generative model. Unlike classification error which is a $\{0,1\}$ valued random variable, the log-likelihood is an unbounded quantity that does not have an obvious sub-Gaussian, or other, well-understood tail behavior.

To overcome this challenge, we construct generalization bounds for BPD not of the original model but instead on a smoothed version of it that limits the worst case behavior. We define this smoothed model as a token-level mixture of the original LLM token predictions and a uniform distribution over the vocabulary of size $V$ :

$$
\begin{equation*}
p_{h}\left(x_{i} \mid x_{<i}\right)=(1-\alpha) p_{\theta}\left(x_{i} \mid x_{<i}\right)+\alpha / V \tag{2}
\end{equation*}
$$

where $p_{\theta}\left(x_{i} \mid x_{<i}\right)$ is the base model of token probabilities, $\alpha \in(0,1)$ is the mixing parameter, and $p_{h}\left(x_{i} \mid x_{<i}\right)$ is the smoothed predictor.

The model on an entire document $X=\left\{x_{i}\right\}_{i=1}^{L}$ composed of $L$ tokens is defined autoregressively in terms of this mixture model $p_{h}(X):=\Pi_{i}^{L} p_{h}\left(x_{i} \mid x_{<i}\right)$, and we find this to be a more effective way of constructing the bounds than constructing the mixture at the document level. In analogy to label smoothing where the labels of the training objective are mixed with the uniform distribution, we term this operation as prediction smoothing.
![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-07.jpg?height=466&width=1286&top_left_y=241&top_left_x=408)

Figure 2: Varying Parameters of the Compression Bounds. (Left): A plot of the generalization bound as a function of the projection dimension $d$ with LoRA. The subspace dimension gives us a way to explicitly trade off the degree of compression with the empirical risk, and we optimize $d$ to produce the best bounds. (Right): A plot of the worst case range of BPD values $\Delta$, empirical risk, and the resulting generalization bounds as a function of the prediction smoothing parameter $\alpha$. For each model, a different alpha can be chosen after the models have already been trained.

As we show in Appendix A.2, the NLL of the prediction smoothed model on a document $\operatorname{BPD}(h, X):=-\log _{2} p_{h}(X) / L$ can be bounded as follows:

$$
\log _{2}(V / \alpha)-\Delta \leq \operatorname{BPD}(h, X) \leq \log _{2}(V / \alpha)
$$

for $\Delta=\log _{2}(1+(1-\alpha) V / \alpha)$. With prediction smoothing, the risk $R(h, X)=\operatorname{BPD}(h, X)$ on a given document is bounded in an interval of size $\Delta$, and therefore we can use Equation (1) to generate bounds for negative log-likelihood of this model. We refer to $\Delta$ as the worst-case interval size.

We explore the trade-off over different values of $\alpha$ in Figure 2 (right). As $\alpha$ gets larger, the interval size $\Delta$ representing the worst-case behavior goes down, whereas the empirical risk goes up, leading to a sweet spot in the middle. By defining the hypothesis $h=(\theta, d, r, \alpha)$ to include the model parameters, LoRA space hyperparameters $d, r$, and the mixture weight $\alpha$, we can view $\alpha$ as merely one additional model parameter accounted in $\log 1 / P(h)$. By doing so, we are free to optimize over $\alpha$ in the computation of the bound, and we can do so without retraining the model.

### 4.4 Using Subsampling in Bound Computation

The empirical risk requires evaluating the model on the full training dataset of $m$ data points: $\hat{R}(h)=\frac{1}{m} \sum_{i=1} \hat{R}_{i}(h)$. As large language models are typically trained for only 1 epoch or less, doing so is prohibitively expensive. Instead, we propose to modify our generalization bounds to account for evaluating only a subsample of size $n \ll m$ of the training dataset when computing the empirical risk.

Denoting $\hat{\hat{R}}(h)=\sum_{i=1}^{n} \hat{R}_{\sigma(i)}(h)$ where $\sigma(i)$ is a random sample (with replacement) from $1, \ldots, m$. In Appendix A. 3 we derive a new bound both over the randomness in $\sigma(i)$ and the randomness in $X$ which holds with probability $\geq 1-\delta$ :

$$
\begin{equation*}
R(h) \leq \hat{\hat{R}}(h)+\Delta \sqrt{\frac{\log \frac{1}{P(h)}+\log \frac{1}{s \delta}}{2 m}}+\Delta \sqrt{\frac{\log \frac{1}{(1-s) \delta}}{2 n}} \tag{3}
\end{equation*}
$$

where $s=n /(n+m)$. Using this subsampling bound, we can accelerate bound computation. For dataset sizes in the 10's of millions, we can get away with evaluating only 10,000 data
points after the model has been trained, with a negligible penalty in the bounds. In fact, we need not even train on the entirety of the training data in order to produce valid bounds as long we indeed sample uniformly.

## 5 SubLoRA: A Simple and Efficient Nonlinear Parameterization of the Hypothesis Space

To find compressible solutions $h$ that simultaneously are expressive enough to achieve low training error, we search over a carefully designed manifold of possible parameters that live within the parameter space.

In contrast to Lotfi et al. (2022), we consider a nonlinear parameterization of the model weights $\theta=f\left(\theta_{0}, w\right)$ given by the composition of LoRA (Hu et al., 2021) (a nonlinear parameterization) and the subspace compression matrices. Given a vector of model parameters $\theta$, we break down its constituent components into the different weight matrices $W_{i}$ and associated biases $b_{i}$ : unflatten $(\theta)=\left\{\left(W_{i}, b_{i}\right)\right\}_{i \in I}$. We define a nonlinear parameterization of the hypothesis space,

$$
\begin{equation*}
\theta=\theta_{0}+\operatorname{LoRA}(P w) \tag{4}
\end{equation*}
$$

where LoRA is defined by the implementation of the low-rank products for the weight matrices, leaving the biases unchanged. As $P w$ and $\theta$ are the flattened parameter vectors, LoRA $(\cdot)$ is defined as the operation that unflattens the vector, applies the low-rank product, and then flattens the result. Here, $\theta_{0}$ is merely a random initialization of the model parameters, and $P \in \mathbb{R}^{D \times d}$ is a Kronecker product projector $P=Q_{1} \otimes Q_{2}$ for $Q_{1}, Q_{2}$ constructed by orthogonalizing Gaussian random matrices by $\mathrm{QR}$ factorization: $P_{1}, P_{2} \sim \mathcal{N}(0,1 / \sqrt{D})^{\sqrt{D} \times \sqrt{d}}$ with $Q_{1} R_{1}=P_{1}$ and similarly for $Q_{2}$. We apply LoRA only over the self-attention layer and the last linear layer weight matrices, meaning that other model parameters do not differ from their initialized values. While LoRA was developed for finetuning LLMs, we find that even when pretraining using LoRA, we can achieve non-trivial performance. In order to compress the model, we need only to represent the vector $w$ since $\theta_{0}$ and $P$ are chosen ahead of time and specified in the architecture via random initialization.

In Figure 1 (left), we show the Pareto frontier of empirical risk and the complexity penalty in the relevant generalization bound with LoRA, subspace training, and SubLoRA. Rather than being competing methods for compression, LoRA and subspace training are complementary and exploit different structures in the parameter space to provide a family of models in the original hypothesis space that are both expressive and compressible. SubLoRA achieves a strict improvement over LoRA and subspace training, often being the deciding factor whether the bounds are vacuous or non-vacuous. In Figure 2 (left), we explore how the compressed size of the model and the empirical risk vary as a function of the subspace dimension $d$.

## 6 Non-Vacuous Generalization Bounds for LLMs

We outline the pretraining and bound computation pipeline and present our empirical results.

### 6.1 End-to-end Pipeline

Assembling the components described in Section 4, we train variants of a GPT-style architecture through the nonlinear compressed parameterization in Equation (4). We use several values for the subspace dimension $d$ and two values for the rank of the LoRA matrices $r$. Nearing the end of training, we train for additional steps using quantization-aware training with a small number of quantization levels (with additional details listed in Appendix D). We express $w$ in this quantization and encode it using arithmetic coding to determine the
compressed size of the model. Added to the size of the model are the bits needed to encode the choice of $d, r, \alpha$, the learning rate, and the quantization levels.

We evaluate the empirical log probabilities and token predictions for each token in the sequence on a small subset of the training data $n=10000$. With these predictions, we can compute the generalization bound in Equation (3) as a function of $\alpha$, and we optimize over this parameter for each model. Finally, we can tune the extent of compression through the different choices of $d$ and choose the subspace dimension that produces the best bound.

### 6.2 Non-Vacuous Bounds for GPT-2 Small

We consider the GPT-2 small architecture with $124 \mathrm{M}$ parameters and compute our next token prediction document-level bounds by pretraining these models on the OpenWebText dataset using SubLoRA. We report the results in Table 1. We consider the token level error averaged over a document as the empirical risk. For instance, the Top-1 Error Bound refers to the upper bound on the expected Top-1 error per token averaged over the document $R\left(h, X_{k}\right)=\frac{1}{L} \sum_{i=1}^{L} \mathbf{1}\left[\operatorname{argmax} p\left(x_{i} \mid x_{<i}=x_{<i}^{k}\right)=x_{i}^{k}\right]$, where the upper index $k$ denotes the document index and the lower index denotes the position within the document. Random guess performance is $\log _{2} V$ for BPD and $1-k / V$ for Top-k Error.

The best bounds are indeed obtained using our simple compression technique, which combines the strengths of both low-rank adaptation and subspace training. When we solely apply quantization and arithmetic coding without implementing LoRA or linear subspace compression during the training phase, we obtain vacuous bounds.

Note (Significance of our bounds with I.I.D sampling). The bits-per-dimension for a given document can be computed as the average error for each token in the sequence given previous tokens withing the same document, where the token error here refers to the negative log-likelihood $\operatorname{BPD}(h, X):=-\log _{2} p_{h}(X) / L=-\sum_{i}^{L} \log _{2} p_{h}\left(x_{i} \mid x_{<i}\right) / L$. Therefore, an upper bound on the expected BPD error reflects a guarantee on the average performance of the model at the token level, conditioned on previous tokens within the same document, and is a quantity of interest in language modeling.

Table 1: Our best document-level generalization bounds achieved for the GPT-2 architecture for BPD and Top-k token prediction error, all of which are non-vacuous.

| Metric | SubLoRA | LoRA Only | Subspace Only | Original Model | Random Guess |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Top-1 Error (\%) | $\mathbf{9 6 . 4 1}$ | 100 | 96.52 | 100 | 99.99 |
| Top-10 Error (\%) | $\mathbf{7 7 . 9 0}$ | 84.37 | 79.36 | 100 | 99.98 |
| Top-100 Error (\%) | $\mathbf{5 8 . 3 4}$ | 67.26 | 75.95 | 100 | 99.80 |
| Bits per Dimension | $\mathbf{1 2 . 1 2}$ | 13.09 | 14.59 | 70.76 | 15.62 |

### 6.3 Extending Our Bounds to Larger Models

We use SubLoRA to obtain generalization bounds for much larger variants of GPT-2 of sizes 354M (GPT-2 medium), 458M, 773M (GPT-2 large), and 849M parameters. Table 2 shows that our simple compression approach yields non-vacuous bounds for models with nearly a billion parameters. Moreover, we see that the smallest model, where we previously performed experiments and tuned our hyperparameters, actually achieves the worst bound on bits per dimension as we scale the models up. In conclusion, our approach extends naturally to much larger language models and proves that it is possible to achieve tighter bounds as we increase the size of the model.

Note (Limitations). Note that due to computational constraints, we pre-train the larger GPT-2 variants with SubLoRA only for a limited number of hyperparameter settings in

Table 2: Non-vacuous bounds achieved for GPT-2 architectures with different sizes, ranging from 124 to 849 million parameters. We report below the bounds on the bits-per-dimension (BPD), Top-1 Error, Top-10 Error, and Top-100 Error. All of the BPD bounds are nonvacuous and tighter than the GPT-2 small bounds.

| Model Size | BPD | Top-1 Error | Top-10 Error | Top-100 Error |
| :--- | :---: | :---: | :---: | :---: |
| 124M (GPT-2 small) | 12.12 | 96.41 | 77.90 | 58.34 |
| 354M (GPT-2 medium) | 11.96 | 95.99 | 78.36 | 58.4 |
| 458M (GPT-2 large) | 11.95 | 96.69 | 78.03 | 58.49 |
| 773M (G8.10 | 12.10 | 96.17 | 78.57 | 59.25 |
| 849M | 12.01 | 96.51 | 78.53 | 58.89 |

contrast to the $124 M$ model for which we did a thorough hyperparameter sweep. It is likely that the tightest empirically achievable bounds are much stronger for the new large models than what we report in Table 2.

## $7 \quad$ Understanding the Generalization of LLMs

As language models grow in size, it is clear that they gain an increasing capacity to fit their training data. On the one hand, this increasing capacity might mean that, as LLMs become capable of learning increasingly complex functions, they become increasingly likely to merely memorize their training samples and not perform any meaningful generalization beyond their training corpora. After all, they have many more parameters to use in fitting the data. On the other hand, large language models have proven to be surprisingly capable of generalizing, often extending to tasks that seem quite different from the training objective.

We investigate the tension between these two narratives along several fronts: We assess how generalization bounds change with the size of the model, whether language models can form a compression of the training data even when accounting for their large size, and how structure in the training data affects the generalization of the learned model. In Appendix C, we use our bounds to quantify of the benefits of pre-training in LLMs.

### 7.1 Larger Models Are More Compressible and Generalize Better

Empirically, it has been found that LLMs generalize better as the number of parameters is increased, with a fixed size of dataset (Kaplan et al., 2020; Brown et al., 2020), and this fact is of great importance leading to the creation of ever larger and more powerful models. From a generalization theory perspective, this trend is counterintuitive because of the growing hypothesis class, and a naive analysis would suggest that larger models should generalize worse. To date, we are not aware of any convincing demonstration that generalization bounds improve with more parameters on models of practical sizes.

We evaluate our bounds on a collection of LLMs with different numbers of parameters, choosing the appropriate scaling for the width, depth, number of attention heads, etc. Surprisingly, we find that our generalization bounds in fact improve with model size, even as the training dataset is held fixed. With our SubLoRA compression, larger models are more compressible given a fixed training error. These results are shown in Figure 3. While some explanations for why larger models should generalize better have been put forward in the literature (Nakkiran et al., 2021; Gunasekar et al., 2017), the mechanism by which larger models become more compressible is not clear, and we believe this result is noteworthy and requires further investigation.

In addition to constructing generalization bounds, we can use our compressed models to form a compression of the training dataset itself. In Figure 3, we count the number of bits

![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-11.jpg?height=499&width=1350&top_left_y=206&top_left_x=385)

![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-11.jpg?height=444&width=358&top_left_y=220&top_left_x=386)

![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-11.jpg?height=463&width=559&top_left_y=210&top_left_x=1165)

Figure 3: Larger models achieve stronger generalization bounds. As we scale up the size of the model via the model parameters (holding the training set fixed), we find that our generalization bounds get better rather than worse. Dots show models trained with differing degrees of compression, indicated by their color. On the right we show the number of bits required to express the training dataset using the model and including the model weights in the compression. Classification error bounds consistently favor smaller models, while data compression favors much larger models, and BPD bounds are in between.

needed to encode the model $C(h)$ and the number of bits to encode the data using the model $C\left(\{X\}_{i=1}^{m} \mid h\right)$, which is the negative log-likelihood of the entire dataset according to the model. Adding these two up, we have a compression of the training dataset using the model, and one which is closely related to our generalization bounds.

### 7.2 How Does Generalization of LLMs Depend on Structure in Text?

Neural networks that fit a training dataset of random noise will not be able to generalize, and the ability of overparametrized networks to fit noise implies that uniform convergence is impossible across the general hypothesis class (Nagarajan and Kolter, 2019). This fact is a clear demonstration that the structure of the dataset influences the generalization properties of the model. However, the impact of more subtle structures on generalization is less understood theoretically. Here, we use our bounds to investigate how the temporal order structure relates to generalization.

![](https://cdn.mathpix.com/cropped/2024_06_04_55bb7bc7a129f3cb95c4g-11.jpg?height=353&width=699&top_left_y=1428&top_left_x=1057)

Figure 4: Breaking text structure with permutations. We compute bounds for LLMs that were trained with the order of the tokens shuffled within each sequence.

We train models that explicitly break the temporal structure of the text data by applying random permutations to each sequence during training. Consequently, the model can only make use of the input information as if it were a bag of words. We find that this broken order structure indeed leads to less favorable generalization bounds. Figure 4 shows the best error bounds when the original and perturbed data are used to train the model and evaluate the bounds for the bits per dimension, top-1 error, and top-100 error losses. While the top-1 error bound becomes vacuous as we break the text structure, the top-100 error and bits per dimensions bounds remain non-vacuous. This might be due to the fact that as we perturb the sequence, predicting the next token accurately becomes an extremely difficult task for LLMs, while predicting a token that fits generally into the context, without necessarily being the correct token, is an easier task.

## 8 Discussion

In this work, we have demonstrated that-despite containing a very large number of parameters-large language models are highly compressible. Using highly compressed LLMs, we were able to compute the first non-vacuous generalization bounds for LLM pretraining. Our findings suggest that the development of tighter compression bounds presents a fruitful avenue for understanding how and why language models generalize. We close with a discussion of the limitations of this work, along with their implications for future generalization theory of language models:

Non I.I.D. token level bounds. In our work, we split up the training data into i.i.d. chunks that form the basis of our bounds. However, the loss for each of these chunks also decomposes as a (non i.i.d.) sum, and it is likely that this additional structure could also be exploited in the bound construction to significantly increase the effective number of training samples.

Efficient bound computation on pretrained models. Our procedure for computing generalization bounds requires training LLMs from scratch through our SubLoRA parametrization. It may be possible to devise a fast method of computing bounds on a model that has already been trained, but still constraining its generalization error. Additionally we may hope to bridge the gap between the compressed model and the uncompressed model, which may behave differently in some regards.

Nonlinear parameterizations. Unlike previous state-of-the-art bounds from Lotfi et al. (2022), we employ a non-linear parameterization via LoRA, significantly improving the bounds. This observation opens up an avenue for rich non-linear parameterizations that simultaneously reduce the number of parameters while also including diverse functions which are likely to fit the training data.

Text generation. The SubLoRA technique is by no means a substitute recipe for stateof-the-art language model pretraining. In Table A. 3 and Table A.4, we show samples of generated text using both a GPT-2 style model pretrained in the standard fashion and a GPT-2 style model pretrained using SubLoRA. While the vanilla GPT-2 style model produces reasonable sentences, the SubLoRA pretrained model outputs ungrammatical text which seem to overly favor tokens with high frequencies of appearances in the training dataset.

Alternative approaches to learning with LLMs. Modern language models make possible new inference techniques such as in-context learning and prompt-tuning. These modes are already seeing widespread deployment and warrant analogous theories of generalization.

Generalization beyond the training distribution. Recent work showed that language models prefer low-complexity numerical sequences on which they were not trained, even at random initialization (Goldblum et al., 2023), and generalization theory may be useful for explaining why LLMs can generalize far outside of their training distribution, and even outside of the text modality, for example to tabular data (Hegselmann et al., 2023) or images (Delétang et al., 2023).

## References

Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv preprint arXiv:2012.13255, 2020.

Pierre Alquier, James Ridgway, and Nicolas Chopin. On the properties of variational approximations of gibbs posteriors. The Journal of Machine Learning Research, 17(1): 8374-8414, 2016 .

Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. In International Conference on Machine Learning, pages 254-263. PMLR, 2018.

Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3(Nov):463-482, 2002.

Sujeeth Bharadwaj and Mark Hasegawa-Johnson. A PAC-Bayesian approach to minimum perplexity language modeling. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, pages 130-140, Dublin, Ireland, 2014.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting training data from large language models. arXiv preprint arXiv:2012.07805, 2020.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. Proceedings of the 37th International Conference on Learning Representations (ICLR 2023), 2023.

Olivier Catoni. Pac-bayesian supervised classification: the thermodynamics of statistical learning. arXiv preprint arXiv:0712.0248, 2007.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. Language modeling is compression. arXiv preprint arXiv:2309.10668, 2023.

Tim Dettmers, Sage Shmitchell, Adam Roberts, Katherine Lee, Tom B. Brown, Dawn Song, and Colin Raffel. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023a.

Tim Dettmers, Sage Shmitchell, Adam Roberts, Katherine Lee, Tom B. Brown, Dawn Song, and Colin Raffel. Spqr: A sparse-quantized representation for near-lossless $11 \mathrm{~m}$ weight compression. arXiv preprint arXiv:2308.07234, 2023b.

Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008, 2017.

Zdenek Frantal, Audrius Gruslys, and Dusan Kiela. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

Pascal Germain, Francis Bach, Alexandre Lacoste, and Simon Lacoste-Julien. Pac-bayesian theory meets bayesian inference. Advances in Neural Information Processing Systems, 29, 2016 .

Micah Goldblum, Marc Finzi, Keefer Rowan, and Andrew Gordon Wilson. The no free lunch theorem, kolmogorov complexity, and the role of inductive biases in machine learning. arXiv preprint arXiv:2304.05366, 2023.

Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit regularization in matrix factorization. Advances in neural information processing systems, 30, 2017.

Maxime Haddouche, Benjamin Guedj, Omar Rivasplata, and John Shawe-Taylor. Pac-bayes unleashed: Generalisation bounds with unbounded losses. Entropy, 23(10):1330, 2021.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. In International Conference on Artificial Intelligence and Statistics, pages 5549-5581. PMLR, 2023.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoeffding, pages 409-426, 1994.

Matthew Holland. Pac-bayes under potentially heavy tails. Advances in Neural Information Processing Systems, 32, 2019.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, and Dongsoo Lee. Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization. arXiv preprint arXiv:2305.14152, 2023.

Ilja Kuzborskij and Csaba Szepesvári. Efron-stein pac-bayesian inequalities. arXiv preprint arXiv:1909.01931, 2019.

Glen G Langdon. An introduction to arithmetic coding. IBM Journal of Research and Development, 28(2):135-149, 1984.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. arXiv preprint arXiv:1804.08838, 2018.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach, 2019.

Yuxuan Liu, Qi Xu, Wei Xu, and Juncheng Zhu. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Sanae Lotfi, Marc Finzi, Sanyam Kapoor, Andres Potapczynski, Micah Goldblum, and Andrew G Wilson. Pac-bayes compression bounds so tight that they can explain generalization. Advances in Neural Information Processing Systems, 35:31459-31473, 2022.

Daniel J McDonald, Cosma Rohilla Shalizi, and Mark Schervish. Generalization error bounds for stationary autoregressive models. arXiv preprint arXiv:1103.0942, 2011.

Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization in deep learning. Advances in Neural Information Processing Systems, 32, 2019.

Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124003, 2021.

Gunho Park, Jihye Kim, Jaeyoung Kim, Eunho Choi, Sungroh Kim, Seungjoo Kim, Minsu Lee, Hyeonwoo Shin, and Juho Lee. Lut-gemm: Quantized matrix multiplication based on luts for efficient inference in large-scale generative language model. arXiv preprint arXiv:2206.09557, 2022.

Ray J Solomonoff. A formal theory of inductive inference. part i. Information and control, 7 (1):1-22, 1964 .

Leena Chennuru Vankadara, Philipp Michael Faller, Michaela Hardt, Lenon Minorics, Debarghya Ghoshdastidar, and Dominik Janzing. Causal forecasting: generalization bounds for autoregressive models. In Uncertainty in Artificial Intelligence, pages 2002-2012. PMLR, 2022.

Vladimir Vapnik. Principles of risk minimization for learning theory. Advances in neural information processing systems, 4, 1991.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding, 2019 .

Qi Xu, Wei Xu, and Juncheng Zhu. Tensorgpt: Efficient compression of the embedding layer in llms based on the tensor-train decomposition. arXiv preprint arXiv:2307.00526, 2023.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107-115, 2021.

Wenda Zhou, Victor Veitch, Morgane Austern, Ryan P Adams, and Peter Orbanz. Nonvacuous generalization bounds at the imagenet scale: a pac-bayesian compression approach. In International Conference on Learning Representations, 2019.
</end of paper 2>


