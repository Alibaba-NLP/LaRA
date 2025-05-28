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
# Retrieval-Augmented Generation for Large Language Models: A Survey 

Yunfan Gao ${ }^{\mathrm{a}}$, Yun Xiong ${ }^{\mathrm{b}}$, Xinyu $\mathrm{Gao}^{\mathrm{b}}$, Kangxiang Jia ${ }^{\mathrm{b}}$, Jinliu Pan ${ }^{\mathrm{b}}$, Yuxi Bi ${ }^{\mathrm{c}}$, Yi Dai ${ }^{\mathrm{a}}$, Jiawei Suna ${ }^{\mathrm{a}}$, Meng<br>Wang $^{\mathrm{c}}$, and Haofen Wang ${ }^{\mathrm{a}, \mathrm{c}}$<br>${ }^{a}$ Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University<br>${ }^{\mathrm{b}}$ Shanghai Key Laboratory of Data Science, School of Computer Science, Fudan University<br>${ }^{c}$ College of Design and Innovation, Tongji University


#### Abstract

Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domainspecific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the retrieval, the generation and the augmentation techniques. The paper highlights the state-of-theart technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development 1 .


Index Terms-Large language model, retrieval-augmented generation, natural language processing, information retrieval

## I. INTRODUCTION

LARGE language models (LLMs) have achieved remarkable success, though they still face significant limitations, especially in domain-specific or knowledge-intensive tasks [1], notably producing "hallucinations" [2] when handling queries beyond their training data or requiring current information. To overcome challenges, Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant document chunks from external knowledge base through semantic similarity calculation. By referencing external knowledge, RAG effectively reduces the problem of generating factually incorrect content. Its integration into LLMs has resulted in widespread adoption, establishing RAG as a key technology in advancing chatbots and enhancing the suitability of LLMs for real-world applications.

RAG technology has rapidly developed in recent years, and the technology tree summarizing related research is shown[^0]

in Figure 1 The development trajectory of RAG in the era of large models exhibits several distinct stage characteristics. Initially, RAG's inception coincided with the rise of the Transformer architecture, focusing on enhancing language models by incorporating additional knowledge through PreTraining Models (PTM). This early stage was characterized by foundational work aimed at refining pre-training techniques [3]-[5].The subsequent arrival of ChatGPT [6] marked a pivotal moment, with LLM demonstrating powerful in context learning (ICL) capabilities. RAG research shifted towards providing better information for LLMs to answer more complex and knowledge-intensive tasks during the inference stage, leading to rapid development in RAG studies. As research progressed, the enhancement of RAG was no longer limited to the inference stage but began to incorporate more with LLM fine-tuning techniques.

The burgeoning field of RAG has experienced swift growth, yet it has not been accompanied by a systematic synthesis that could clarify its broader trajectory. This survey endeavors to fill this gap by mapping out the RAG process and charting its evolution and anticipated future paths, with a focus on the integration of RAG within LLMs. This paper considers both technical paradigms and research methods, summarizing three main research paradigms from over 100 RAG studies, and analyzing key technologies in the core stages of "Retrieval," "Generation," and "Augmentation." On the other hand, current research tends to focus more on methods, lacking analysis and summarization of how to evaluate RAG. This paper comprehensively reviews the downstream tasks, datasets, benchmarks, and evaluation methods applicable to RAG. Overall, this paper sets out to meticulously compile and categorize the foundational technical concepts, historical progression, and the spectrum of RAG methodologies and applications that have emerged post-LLMs. It is designed to equip readers and professionals with a detailed and structured understanding of both large models and RAG. It aims to illuminate the evolution of retrieval augmentation techniques, assess the strengths and weaknesses of various approaches in their respective contexts, and speculate on upcoming trends and innovations.

Our contributions are as follows:

- In this survey, we present a thorough and systematic review of the state-of-the-art RAG methods, delineating its evolution through paradigms including naive RAG,

![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-02.jpg?height=1087&width=1396&top_left_y=194&top_left_x=343)

Fig. 1. Technology tree of RAG research. The stages of involving RAG mainly include pre-training, fine-tuning, and inference. With the emergence of LLMs, research on RAG initially focused on leveraging the powerful in context learning abilities of LLMs, primarily concentrating on the inference stage. Subsequent research has delved deeper, gradually integrating more with the fine-tuning of LLMs. Researchers have also been exploring ways to enhance language models in the pre-training stage through retrieval-augmented techniques.

advanced RAG, and modular RAG. This review contextualizes the broader scope of RAG research within the landscape of LLMs.

- We identify and discuss the central technologies integral to the RAG process, specifically focusing on the aspects of "Retrieval", "Generation" and "Augmentation", and delve into their synergies, elucidating how these components intricately collaborate to form a cohesive and effective RAG framework.
- We have summarized the current assessment methods of RAG, covering 26 tasks, nearly 50 datasets, outlining the evaluation objectives and metrics, as well as the current evaluation benchmarks and tools. Additionally, we anticipate future directions for RAG, emphasizing potential enhancements to tackle current challenges.

The paper unfolds as follows: Section II introduces the main concept and current paradigms of RAG. The following three sections explore core components-"Retrieval", "Generation" and "Augmentation", respectively. Section III focuses on optimization methods in retrieval, including indexing, query and embedding optimization. Section IV concentrates on postretrieval process and LLM fine-tuning in generation. Section V analyzes the three augmentation processes. Section VI focuses on RAG's downstream tasks and evaluation system. Section VII mainly discusses the challenges that RAG currently faces and its future development directions. At last, the paper concludes in Section VIII.

## II. OVERVIEW OF RAG

A typical application of RAG is illustrated in Figure 2 Here, a user poses a question to ChatGPT about a recent, widely discussed news. Given ChatGPT's reliance on pretraining data, it initially lacks the capacity to provide updates on recent developments. RAG bridges this information gap by sourcing and incorporating knowledge from external databases. In this case, it gathers relevant news articles related to the user's query. These articles, combined with the original question, form a comprehensive prompt that empowers LLMs to generate a well-informed answer.

The RAG research paradigm is continuously evolving, and we categorize it into three stages: Naive RAG, Advanced RAG, and Modular RAG, as showed in Figure 3. Despite RAG method are cost-effective and surpass the performance of the native LLM, they also exhibit several limitations. The development of Advanced RAG and Modular RAG is a response to these specific shortcomings in Naive RAG.

## A. Naive RAG

The Naive RAG research paradigm represents the earliest methodology, which gained prominence shortly after the

![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-03.jpg?height=906&width=1529&top_left_y=192&top_left_x=298)

Fig. 2. A representative instance of the RAG process applied to question answering. It mainly consists of 3 steps. 1) Indexing. Documents are split into chunks, encoded into vectors, and stored in a vector database. 2) Retrieval. Retrieve the Top k chunks most relevant to the question based on semantic similarity. 3) Generation. Input the original question and the retrieved chunks together into LLM to generate the final answer.

widespread adoption of ChatGPT. The Naive RAG follows a traditional process that includes indexing, retrieval, and generation, which is also characterized as a "Retrieve-Read" framework [7].

Indexing starts with the cleaning and extraction of raw data in diverse formats like PDF, HTML, Word, and Markdown, which is then converted into a uniform plain text format. To accommodate the context limitations of language models, text is segmented into smaller, digestible chunks. Chunks are then encoded into vector representations using an embedding model and stored in vector database. This step is crucial for enabling efficient similarity searches in the subsequent retrieval phase.

Retrieval. Upon receipt of a user query, the RAG system employs the same encoding model utilized during the indexing phase to transform the query into a vector representation. It then computes the similarity scores between the query vector and the vector of chunks within the indexed corpus. The system prioritizes and retrieves the top $\mathrm{K}$ chunks that demonstrate the greatest similarity to the query. These chunks are subsequently used as the expanded context in prompt.

Generation. The posed query and selected documents are synthesized into a coherent prompt to which a large language model is tasked with formulating a response. The model's approach to answering may vary depending on task-specific criteria, allowing it to either draw upon its inherent parametric knowledge or restrict its responses to the information contained within the provided documents. In cases of ongoing dialogues, any existing conversational history can be integrated into the prompt, enabling the model to engage in multi-turn dialogue interactions effectively.

However, Naive RAG encounters notable drawbacks:
Retrieval Challenges. The retrieval phase often struggles with precision and recall, leading to the selection of misaligned or irrelevant chunks, and the missing of crucial information.

Generation Difficulties. In generating responses, the model may face the issue of hallucination, where it produces content not supported by the retrieved context. This phase can also suffer from irrelevance, toxicity, or bias in the outputs, detracting from the quality and reliability of the responses.

Augmentation Hurdles. Integrating retrieved information with the different task can be challenging, sometimes resulting in disjointed or incoherent outputs. The process may also encounter redundancy when similar information is retrieved from multiple sources, leading to repetitive responses. Determining the significance and relevance of various passages and ensuring stylistic and tonal consistency add further complexity. Facing complex issues, a single retrieval based on the original query may not suffice to acquire adequate context information.

Moreover, there's a concern that generation models might overly rely on augmented information, leading to outputs that simply echo retrieved content without adding insightful or synthesized information.

## B. Advanced RAG

Advanced RAG introduces specific improvements to overcome the limitations of Naive RAG. Focusing on enhancing retrieval quality, it employs pre-retrieval and post-retrieval strategies. To tackle the indexing issues, Advanced RAG refines its indexing techniques through the use of a sliding window approach, fine-grained segmentation, and the incorporation of metadata. Additionally, it incorporates several optimization methods to streamline the retrieval process [8].

![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-04.jpg?height=954&width=1589&top_left_y=230&top_left_x=276)

Fig. 3. Comparison between the three paradigms of RAG. (Left) Naive RAG mainly consists of three parts: indexing, retrieval and generation. (Middle) Advanced RAG proposes multiple optimization strategies around pre-retrieval and post-retrieval, with a process similar to the Naive RAG, still following a chain-like structure. (Right) Modular RAG inherits and develops from the previous paradigm, showcasing greater flexibility overall. This is evident in the introduction of multiple specific functional modules and the replacement of existing modules. The overall process is not limited to sequential retrieval and generation; it includes methods such as iterative and adaptive retrieval.

Pre-retrieval process. In this stage, the primary focus is on optimizing the indexing structure and the original query. The goal of optimizing indexing is to enhance the quality of the content being indexed. This involves strategies: enhancing data granularity, optimizing index structures, adding metadata, alignment optimization, and mixed retrieval. While the goal of query optimization is to make the user's original question clearer and more suitable for the retrieval task. Common methods include query rewriting query transformation, query expansion and other techniques [7], [9]-[11].

Post-Retrieval Process. Once relevant context is retrieved, it's crucial to integrate it effectively with the query. The main methods in post-retrieval process include rerank chunks and context compressing. Re-ranking the retrieved information to relocate the most relevant content to the edges of the prompt is a key strategy. This concept has been implemented in frameworks such as LlamaIndex[2, LangChain 3. and HayStack 12]. Feeding all relevant documents directly into LLMs can lead to information overload, diluting the focus on key details with irrelevant content.To mitigate this, post-retrieval efforts concentrate on selecting the essential information, emphasizing critical sections, and shortening the context to be processed.[^1]

## C. Modular RAG

The modular RAG architecture advances beyond the former two RAG paradigms, offering enhanced adaptability and versatility. It incorporates diverse strategies for improving its components, such as adding a search module for similarity searches and refining the retriever through fine-tuning. Innovations like restructured RAG modules [13] and rearranged RAG pipelines [14] have been introduced to tackle specific challenges. The shift towards a modular RAG approach is becoming prevalent, supporting both sequential processing and integrated end-to-end training across its components. Despite its distinctiveness, Modular RAG builds upon the foundational principles of Advanced and Naive RAG, illustrating a progression and refinement within the RAG family.

1) New Modules: The Modular RAG framework introduces additional specialized components to enhance retrieval and processing capabilities. The Search module adapts to specific scenarios, enabling direct searches across various data sources like search engines, databases, and knowledge graphs, using LLM-generated code and query languages [15]. RAGFusion addresses traditional search limitations by employing a multi-query strategy that expands user queries into diverse perspectives, utilizing parallel vector searches and intelligent re-ranking to uncover both explicit and transformative knowledge [16]. The Memory module leverages the LLM's memory to guide retrieval, creating an unbounded memory pool that
aligns the text more closely with data distribution through iterative self-enhancement [17], [18]. Routing in the RAG system navigates through diverse data sources, selecting the optimal pathway for a query, whether it involves summarization, specific database searches, or merging different information streams [19]. The Predict module aims to reduce redundancy and noise by generating context directly through the LLM, ensuring relevance and accuracy [13]. Lastly, the Task Adapter module tailors RAG to various downstream tasks, automating prompt retrieval for zero-shot inputs and creating task-specific retrievers through few-shot query generation [20], [21] .This comprehensive approach not only streamlines the retrieval process but also significantly improves the quality and relevance of the information retrieved, catering to a wide array of tasks and queries with enhanced precision and flexibility.
2) New Patterns: Modular RAG offers remarkable adaptability by allowing module substitution or reconfiguration to address specific challenges. This goes beyond the fixed structures of Naive and Advanced RAG, characterized by a simple "Retrieve" and "Read" mechanism. Moreover, Modular RAG expands this flexibility by integrating new modules or adjusting interaction flow among existing ones, enhancing its applicability across different tasks.

Innovations such as the Rewrite-Retrieve-Read [7]model leverage the LLM's capabilities to refine retrieval queries through a rewriting module and a LM-feedback mechanism to update rewriting model., improving task performance. Similarly, approaches like Generate-Read [13] replace traditional retrieval with LLM-generated content, while ReciteRead [22] emphasizes retrieval from model weights, enhancing the model's ability to handle knowledge-intensive tasks. Hybrid retrieval strategies integrate keyword, semantic, and vector searches to cater to diverse queries. Additionally, employing sub-queries and hypothetical document embeddings (HyDE) [11] seeks to improve retrieval relevance by focusing on embedding similarities between generated answers and real documents.

Adjustments in module arrangement and interaction, such as the Demonstrate-Search-Predict (DSP) [23] framework and the iterative Retrieve-Read-Retrieve-Read flow of ITERRETGEN [14], showcase the dynamic use of module outputs to bolster another module's functionality, illustrating a sophisticated understanding of enhancing module synergy. The flexible orchestration of Modular RAG Flow showcases the benefits of adaptive retrieval through techniques such as FLARE [24] and Self-RAG [25]. This approach transcends the fixed RAG retrieval process by evaluating the necessity of retrieval based on different scenarios. Another benefit of a flexible architecture is that the RAG system can more easily integrate with other technologies (such as fine-tuning or reinforcement learning) [26]. For example, this can involve fine-tuning the retriever for better retrieval results, fine-tuning the generator for more personalized outputs, or engaging in collaborative fine-tuning [27].

## D. RAG vs Fine-tuning

The augmentation of LLMs has attracted considerable attention due to their growing prevalence. Among the optimization methods for LLMs, RAG is often compared with Fine-tuning (FT) and prompt engineering. Each method has distinct characteristics as illustrated in Figure 4. We used a quadrant chart to illustrate the differences among three methods in two dimensions: external knowledge requirements and model adaption requirements. Prompt engineering leverages a model's inherent capabilities with minimum necessity for external knowledge and model adaption. RAG can be likened to providing a model with a tailored textbook for information retrieval, ideal for precise information retrieval tasks. In contrast, FT is comparable to a student internalizing knowledge over time, suitable for scenarios requiring replication of specific structures, styles, or formats.

RAG excels in dynamic environments by offering realtime knowledge updates and effective utilization of external knowledge sources with high interpretability. However, it comes with higher latency and ethical considerations regarding data retrieval. On the other hand, FT is more static, requiring retraining for updates but enabling deep customization of the model's behavior and style. It demands significant computational resources for dataset preparation and training, and while it can reduce hallucinations, it may face challenges with unfamiliar data.

In multiple evaluations of their performance on various knowledge-intensive tasks across different topics, [28] revealed that while unsupervised fine-tuning shows some improvement, RAG consistently outperforms it, for both existing knowledge encountered during training and entirely new knowledge. Additionally, it was found that LLMs struggle to learn new factual information through unsupervised finetuning. The choice between RAG and FT depends on the specific needs for data dynamics, customization, and computational capabilities in the application context. RAG and FT are not mutually exclusive and can complement each other, enhancing a model's capabilities at different levels. In some instances, their combined use may lead to optimal performance. The optimization process involving RAG and FT may require multiple iterations to achieve satisfactory results.

## III. RETRIEVAL

In the context of RAG, it is crucial to efficiently retrieve relevant documents from the data source. There are several key issues involved, such as the retrieval source, retrieval granularity, pre-processing of the retrieval, and selection of the corresponding embedding model.

## A. Retrieval Source

RAG relies on external knowledge to enhance LLMs, while the type of retrieval source and the granularity of retrieval units both affect the final generation results.

1) Data Structure: Initially, text is s the mainstream source of retrieval. Subsequently, the retrieval source expanded to include semi-structured data (PDF) and structured data (Knowledge Graph, KG) for enhancement. In addition to retrieving from original external sources, there is also a growing trend in recent researches towards utilizing content generated by LLMs themselves for retrieval and enhancement purposes.

TABLE I

SUMMARY OF RAG METHODS

| Method | Retrieval Source | Retrieval <br> Data Type | Retrieval <br> Granularity | Augmentation <br> Stage | Retrieval <br> process |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{CoG} 29$ | Wikipedia | Text | Phrase | Pre-training | Iterative |
| DenseX 30$]$ | FactoidWiki | Text | Proposition | Inference | Once |
| EAR 31$]$ | Dataset-base | Text | Sentence | Tuning | Once |
| UPRISE [20] | Dataset-base | Text | Sentence | Tuning | Once |
| RAST 32 | Dataset-base | Text | Sentence | Tuning | Once |
| Self-Mem [17] | Dataset-base | Text | Sentence | Tuning | Iterative |
| FLARE 24 | Search Engine,Wikipedia | Text | Sentence | Tuning | Adaptive |
| PGRA [33] | Wikipedia | Text | Sentence | Inference | Once |
| FILCO $\overline{34}$ | Wikipedia | Text | Sentence | Inference | Once |
| RADA 35 | Dataset-base | Text | Sentence | Inference | Once |
| Filter-rerank 36 | Synthesized dataset | Text | Sentence | Inference | Once |
| R-GQA 37 | Dataset-base | Text | Sentence Pair | Tuning | Once |
| LLM-R 38 | Dataset-base | Text | Sentence Pair | Inference | Iterative |
| TIGER 39 | Dataset-base | Text | Item-base | Pre-training | Once |
| LM-Indexer 40 | Dataset-base | Text | Item-base | Tuning | Once |
| BEQUE 9 | Dataset-base | Text | Item-base | Tuning | Once |
| CT-RAG 41$]$ | Synthesized dataset | Text | Item-base | Tuning | Once |
| Atlas 42 | Wikipedia, Common Crawl | Text | Chunk | Pre-training | Iterative |
| RAVEN 43$]$ | Wikipedia | Text | Chunk | Pre-training | Once |
| RETRO++ 44 | Pre-training Corpus | Text | Chunk | Pre-training | Iterative |
| INSTRUCTRETRO 45 | Pre-training corpus | Text | Chunk | Pre-training | Iterative |
| RRR 77 | Search Engine | Text | Chunk | Tuning | Once |
| RA-e2e 46 | Dataset-base | Text | Chunk | Tuning | Once |
| PROMPTAGATOR \|21] | BEIR | Text | Chunk | Tuning | Once |
| AAR 47$]$ | MSMARCO,Wikipedia | Text | Chunk | Tuning | Once |
| RA-DIT [27] | Common Crawl,Wikipedia | Text | Chunk | Tuning | Once |
| RAG-Robust [48] | Wikipedia | Text | Chunk | Tuning | Once |
| RA-Long-Form | Dataset-base | Text | Chunk | Tuning | Once |
| $\mathrm{CoN}[50]$ | Wikipedia | Text | Chunk | Tuning | Once |
| Self-RAG 25 | Wikipedia | Text | Chunk | Tuning | Adaptive |
| BGM 26 | Wikipedia | Text | Chunk | Inference | Once |
| $\mathrm{CoQ}[51]$ | Wikipedia | Text | Chunk | Inference | Iterative |
| Token-Elimination 52 | Wikipedia | Text | Chunk | Inference | Once |
| PaperQA [53] | Arxiv,Online Database,PubMed | Text | Chunk | Inference | Iterative |
| NoiseRAG [54] | FactoidWiki | Text | Chunk | Inference | Once |
| IAG 55 | Search Engine,Wikipedia | Text | Chunk | Inference | Once |
| NoMIRACL 56 | Wikipedia | Text | Chunk | Inference | Once |
| ToC 57 | Search Engine,Wikipedia | Text | Chunk | Inference | Recursive |
| SKR 58 | Dataset-base,Wikipedia | Text | Chunk | Inference | Adaptive |
| ITRG 59 . | Wikipedia | Text | Chunk | Inference | Iterative |
| RAG-LongContext 60 | Dataset-base | Text | Chunk | Inference | Once |
| ITER-RETGEN 14 | Wikipedia | Text | Chunk | Inference | Iterative |
| IRCoT 61$]$ | Wikipedia | Text | Chunk | Inference | Recursive |
| LLM-Knowledge-Boundary 62 | Wikipedia | Text | Chunk | Inference | Once |
| RAPTOR 63 | Dataset-base | Text | Chunk | Inference | Recursive |
| RECITE 22 | LLMs | Text | Chunk | Inference | Once |
| ICRALM 64 | Pile,Wikipedia | Text | Chunk | Inference | Iterative |
| Retrieve-and-Sample 65 | Dataset-base | Text | Doc | Tuning | Once |
| Zemi 66$]$ | $\mathrm{C} 4$ | Text | Doc | Tuning | Once |
| CRAG [67] | Arxiv | Text | Doc | Inference | Once |
| 1-PAGER 68 | Wikipedia | Text | Doc | Inference | Iterative |
| PRCA 69 | Dataset-base | Text | Doc | Inference | Once |
| QLM-Doc-ranking 70 | Dataset-base | Text | Doc | Inference | Once |
| Recomp [71] | Wikipedia | Text | Doc | Inference | Once |
| $\mathrm{DSP} 23$ | Wikipedia | Text | Doc | Inference | Iterative |
| RePLUG 72$]$ | Pile | Text | Doc | Inference | Once |
| ARM-RAG [73] | Dataset-base | Text | Doc | Inference | Iterative |
| GenRead 13 | LLMs | Text | Doc | Inference | Iterative |
| UniMS-RAG 174 | Dataset-base | Text | Multi | Tuning | Once |
| CREA-ICL 19 | Dataset-base | Crosslingual,Text | Sentence | Inference | Once |
| PKG 75 | LLM | Tabular,Text | Chunk | Inference | Once |
| SANTA [76] | Dataset-base | Code,Text | Item | Pre-training | Once |
| SURGE 77 | Freebase | $\mathrm{KG}$ | Sub-Graph | Tuning | Once |
| $\mathrm{MK}-\mathrm{ToD} 78$ | Dataset-base | $\mathrm{KG}$ | Entity | Tuning | Once |
| Dual-Feedback-ToD 79 | Dataset-base | $\mathrm{KG}$ | Entity Sequence | Tuning | Once |
| KnowledGPT 15 | Dataset-base | $\mathrm{KG}$ | Triplet | Inference | Muti-time |
| FABULA 80 | Dataset-base,Graph | $\mathrm{KG}$ | Entity | Inference | Once |
| HyKGE 81 | $\mathrm{CMeKG}$ | $\mathrm{KG}$ | Entity | Inference | Once |
| KALMV 82 | Wikipedia | $\mathrm{KG}$ | Triplet | Inference | Iterative |
| $\operatorname{RoG}$ | Freebase | $\mathrm{KG}$ | Triplet | Inference | Iterative |
| G-Retriever 84$]$ | Dataset-base | TextGraph | Sub-Graph | Inference | Once |

![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-07.jpg?height=835&width=1412&top_left_y=192&top_left_x=346)

Fig. 4. RAG compared with other model optimization methods in the aspects of "External Knowledge Required" and "Model Adaption Required". Prompt Engineering requires low modifications to the model and external knowledge, focusing on harnessing the capabilities of LLMs themselves. Fine-tuning, on the other hand, involves further training the model. In the early stages of RAG (Naive RAG), there is a low demand for model modifications. As research progresses, Modular RAG has become more integrated with fine-tuning techniques.

Unstructured Data, such as text, is the most widely used retrieval source, which are mainly gathered from corpus. For open-domain question-answering (ODQA) tasks, the primary retrieval sources are Wikipedia Dump with the current major versions including HotpotQA $4^{4}$ (1st October, 2017), DPR $\square^{5}$ (20 December, 2018). In addition to encyclopedic data, common unstructured data includes cross-lingual text [19] and domainspecific data (such as medical [67] and legal domains [29]).

Semi-structured data. typically refers to data that contains a combination of text and table information, such as PDF. Handling semi-structured data poses challenges for conventional RAG systems due to two main reasons. Firstly, text splitting processes may inadvertently separate tables, leading to data corruption during retrieval. Secondly, incorporating tables into the data can complicate semantic similarity searches. When dealing with semi-structured data, one approach involves leveraging the code capabilities of LLMs to execute Text-2-SQL queries on tables within databases, such as TableGPT [85]. Alternatively, tables can be transformed into text format for further analysis using text-based methods [75]. However, both of these methods are not optimal solutions, indicating substantial research opportunities in this area.

Structured data, such as knowledge graphs (KGs) [86] , which are typically verified and can provide more precise information. KnowledGPT [15] generates KB search queries and stores knowledge in a personalized base, enhancing the RAG model's knowledge richness. In response to the limitations of LLMs in understanding and answering questions about textual graphs, G-Retriever [84] integrates Graph Neural Networks[^2]

(GNNs), LLMs and RAG, enhancing graph comprehension and question-answering capabilities through soft prompting of the LLM, and employs the Prize-Collecting Steiner Tree (PCST) optimization problem for targeted graph retrieval. On the contrary, it requires additional effort to build, validate, and maintain structured databases. On the contrary, it requires additional effort to build, validate, and maintain structured databases.

LLMs-Generated Content. Addressing the limitations of external auxiliary information in RAG, some research has focused on exploiting LLMs' internal knowledge. SKR [58] classifies questions as known or unknown, applying retrieval enhancement selectively. GenRead [13] replaces the retriever with an LLM generator, finding that LLM-generated contexts often contain more accurate answers due to better alignment with the pre-training objectives of causal language modeling. Selfmem [17] iteratively creates an unbounded memory pool with a retrieval-enhanced generator, using a memory selector to choose outputs that serve as dual problems to the original question, thus self-enhancing the generative model. These methodologies underscore the breadth of innovative data source utilization in RAG, striving to improve model performance and task effectiveness.

2) Retrieval Granularity: Another important factor besides the data format of the retrieval source is the granularity of the retrieved data. Coarse-grained retrieval units theoretically can provide more relevant information for the problem, but they may also contain redundant content, which could distract the retriever and language models in downstream tasks [50], [87]. On the other hand, fine-grained retrieval unit granularity increases the burden of retrieval and does not guarantee semantic integrity and meeting the required knowledge. Choosing
the appropriate retrieval granularity during inference can be a simple and effective strategy to improve the retrieval and downstream task performance of dense retrievers.

In text, retrieval granularity ranges from fine to coarse, including Token, Phrase, Sentence, Proposition, Chunks, Document. Among them, DenseX [30]proposed the concept of using propositions as retrieval units. Propositions are defined as atomic expressions in the text, each encapsulating a unique factual segment and presented in a concise, self-contained natural language format. This approach aims to enhance retrieval precision and relevance. On the Knowledge Graph (KG), retrieval granularity includes Entity, Triplet, and sub-Graph. The granularity of retrieval can also be adapted to downstream tasks, such as retrieving Item IDs [40]in recommendation tasks and Sentence pairs [38]. Detailed information is illustrated in Table $\llbracket$

## B. Indexing Optimization

In the Indexing phase, documents will be processed, segmented, and transformed into Embeddings to be stored in a vector database. The quality of index construction determines whether the correct context can be obtained in the retrieval phase.

1) Chunking Strategy: The most common method is to split the document into chunks on a fixed number of tokens (e.g., 100, 256, 512) [88]. Larger chunks can capture more context, but they also generate more noise, requiring longer processing time and higher costs. While smaller chunks may not fully convey the necessary context, they do have less noise. However, chunks leads to truncation within sentences, prompting the optimization of a recursive splits and sliding window methods, enabling layered retrieval by merging globally related information across multiple retrieval processes [89]. Nevertheless, these approaches still cannot strike a balance between semantic completeness and context length. Therefore, methods like Small2Big have been proposed, where sentences (small) are used as the retrieval unit, and the preceding and following sentences are provided as (big) context to LLMs [90].
2) Metadata Attachments: Chunks can be enriched with metadata information such as page number, file name, author,category timestamp. Subsequently, retrieval can be filtered based on this metadata, limiting the scope of the retrieval. Assigning different weights to document timestamps during retrieval can achieve time-aware RAG, ensuring the freshness of knowledge and avoiding outdated information.

In addition to extracting metadata from the original documents, metadata can also be artificially constructed. For example, adding summaries of paragraph, as well as introducing hypothetical questions. This method is also known as Reverse HyDE. Specifically, using LLM to generate questions that can be answered by the document, then calculating the similarity between the original question and the hypothetical question during retrieval to reduce the semantic gap between the question and the answer.

3) Structural Index: One effective method for enhancing information retrieval is to establish a hierarchical structure for the documents. By constructing In structure, RAG system can expedite the retrieval and processing of pertinent data.
Hierarchical index structure. File are arranged in parentchild relationships, with chunks linked to them. Data summaries are stored at each node, aiding in the swift traversal of data and assisting the RAG system in determining which chunks to extract. This approach can also mitigate the illusion caused by block extraction issues.

Knowledge Graph index. Utilize KG in constructing the hierarchical structure of documents contributes to maintaining consistency. It delineates the connections between different concepts and entities, markedly reducing the potential for illusions. Another advantage is the transformation of the information retrieval process into instructions that LLM can comprehend, thereby enhancing the accuracy of knowledge retrieval and enabling LLM to generate contextually coherent responses, thus improving the overall efficiency of the RAG system. To capture the logical relationship between document content and structure, KGP [91] proposed a method of building an index between multiple documents using KG. This KG consists of nodes (representing paragraphs or structures in the documents, such as pages and tables) and edges (indicating semantic/lexical similarity between paragraphs or relationships within the document structure), effectively addressing knowledge retrieval and reasoning problems in a multi-document environment.

## C. Query Optimization

One of the primary challenges with Naive RAG is its direct reliance on the user's original query as the basis for retrieval. Formulating a precise and clear question is difficult, and imprudent queries result in subpar retrieval effectiveness. Sometimes, the question itself is complex, and the language is not well-organized. Another difficulty lies in language complexity ambiguity. Language models often struggle when dealing with specialized vocabulary or ambiguous abbreviations with multiple meanings. For instance, they may not discern whether "LLM" refers to large language model or a Master of Laws in a legal context.

1) Query Expansion: Expanding a single query into multiple queries enriches the content of the query, providing further context to address any lack of specific nuances, thereby ensuring the optimal relevance of the generated answers.

Multi-Query. By employing prompt engineering to expand queries via LLMs, these queries can then be executed in parallel. The expansion of queries is not random, but rather meticulously designed.

Sub-Query. The process of sub-question planning represents the generation of the necessary sub-questions to contextualize and fully answer the original question when combined. This process of adding relevant context is, in principle, similar to query expansion. Specifically, a complex question can be decomposed into a series of simpler sub-questions using the least-to-most prompting method [92].

Chain-of-Verification(CoVe). The expanded queries undergo validation by LLM to achieve the effect of reducing hallucinations. Validated expanded queries typically exhibit higher reliability [93].

2) Query Transformation: The core concept is to retrieve chunks based on a transformed query instead of the user's original query.

Query Rewrite.The original queries are not always optimal for LLM retrieval, especially in real-world scenarios. Therefore, we can prompt LLM to rewrite the queries. In addition to using LLM for query rewriting, specialized smaller language models, such as RRR (Rewrite-retrieve-read) [7]. The implementation of the query rewrite method in the Taobao, known as BEQUE [9] has notably enhanced recall effectiveness for long-tail queries, resulting in a rise in GMV.

Another query transformation method is to use prompt engineering to let LLM generate a query based on the original query for subsequent retrieval. HyDE [11] construct hypothetical documents (assumed answers to the original query). It focuses on embedding similarity from answer to answer rather than seeking embedding similarity for the problem or query. Using the Step-back Prompting method [10], the original query is abstracted to generate a high-level concept question (step-back question). In the RAG system, both the step-back question and the original query are used for retrieval, and both the results are utilized as the basis for language model answer generation.

3) Query Routing: Based on varying queries, routing to distinct RAG pipeline,which is suitable for a versatile RAG system designed to accommodate diverse scenarios.

Metadata Router/ Filter. The first step involves extracting keywords (entity) from the query, followed by filtering based on the keywords and metadata within the chunks to narrow down the search scope.

Semantic Router is another method of routing involves leveraging the semantic information of the query. Specific apprach see Semantic Router ${ }^{6}$ Certainly, a hybrid routing approach can also be employed, combining both semantic and metadata-based methods for enhanced query routing.

## D. Embedding

In RAG, retrieval is achieved by calculating the similarity (e.g. cosine similarity) between the embeddings of the question and document chunks, where the semantic representation capability of embedding models plays a key role. This mainly includes a sparse encoder (BM25) and a dense retriever (BERT architecture Pre-training language models). Recent research has introduced prominent embedding models such as AngIE, Voyage, BGE,etc [94]-[96], which are benefit from multi-task instruct tuning. Hugging Face's MTEB leaderboard 7 evaluates embedding models across 8 tasks, covering 58 datasests. Additionally, C-MTEB focuses on Chinese capability, covering 6 tasks and 35 datasets. There is no one-size-fits-all answer to "which embedding model to use." However, some specific models are better suited for particular use cases.

1) Mix/hybrid Retrieval : Sparse and dense embedding approaches capture different relevance features and can benefit from each other by leveraging complementary relevance information. For instance, sparse retrieval models can be used[^3]

to provide initial search results for training dense retrieval models. Additionally, pre-training language models (PLMs) can be utilized to learn term weights to enhance sparse retrieval. Specifically, it also demonstrates that sparse retrieval models can enhance the zero-shot retrieval capability of dense retrieval models and assist dense retrievers in handling queries containing rare entities, thereby improving robustness.

2) Fine-tuning Embedding Model: In instances where the context significantly deviates from pre-training corpus, particularly within highly specialized disciplines such as healthcare, legal practice, and other sectors replete with proprietary jargon, fine-tuning the embedding model on your own domain dataset becomes essential to mitigate such discrepancies.

In addition to supplementing domain knowledge, another purpose of fine-tuning is to align the retriever and generator, for example, using the results of LLM as the supervision signal for fine-tuning, known as LSR (LM-supervised Retriever). PROMPTAGATOR [21] utilizes the LLM as a few-shot query generator to create task-specific retrievers, addressing challenges in supervised fine-tuning, particularly in data-scarce domains. Another approach, LLM-Embedder [97], exploits LLMs to generate reward signals across multiple downstream tasks. The retriever is fine-tuned with two types of supervised signals: hard labels for the dataset and soft rewards from the LLMs. This dual-signal approach fosters a more effective fine-tuning process, tailoring the embedding model to diverse downstream applications. REPLUG [72] utilizes a retriever and an LLM to calculate the probability distributions of the retrieved documents and then performs supervised training by computing the KL divergence. This straightforward and effective training method enhances the performance of the retrieval model by using an LM as the supervisory signal, eliminating the need for specific cross-attention mechanisms. Moreover, inspired by RLHF (Reinforcement Learning from Human Feedback), utilizing LM-based feedback to reinforce the retriever through reinforcement learning.

## E. Adapter

Fine-tuning models may present challenges, such as integrating functionality through an API or addressing constraints arising from limited local computational resources. Consequently, some approaches opt to incorporate an external adapter to aid in alignment.

To optimize the multi-task capabilities of LLM, UPRISE [20] trained a lightweight prompt retriever that can automatically retrieve prompts from a pre-built prompt pool that are suitable for a given zero-shot task input. AAR (Augmentation-Adapted Retriver) [47] introduces a universal adapter designed to accommodate multiple downstream tasks. While PRCA [69] add a pluggable reward-driven contextual adapter to enhance performance on specific tasks. BGM [26] keeps the retriever and LLM fixed, and trains a bridge Seq2Seq model in between. The bridge model aims to transform the retrieved information into a format that LLMs can work with effectively, allowing it to not only rerank but also dynamically select passages for each query, and potentially employ more advanced strategies like repetition. Furthermore, PKG
introduces an innovative method for integrating knowledge into white-box models via directive fine-tuning [75]. In this approach, the retriever module is directly substituted to generate relevant documents according to a query. This method assists in addressing the difficulties encountered during the fine-tuning process and enhances model performance.

## IV. GENERATION

After retrieval, it is not a good practice to directly input all the retrieved information to the LLM for answering questions. Following will introduce adjustments from two perspectives: adjusting the retrieved content and adjusting the LLM.

## A. Context Curation

Redundant information can interfere with the final generation of LLM, and overly long contexts can also lead LLM to the "Lost in the middle" problem [98]. Like humans, LLM tends to only focus on the beginning and end of long texts, while forgetting the middle portion. Therefore, in the RAG system, we typically need to further process the retrieved content.

1) Reranking: Reranking fundamentally reorders document chunks to highlight the most pertinent results first, effectively reducing the overall document pool, severing a dual purpose in information retrieval, acting as both an enhancer and a filter, delivering refined inputs for more precise language model processing [70]. Reranking can be performed using rule-based methods that depend on predefined metrics like Diversity, Relevance, and MRR, or model-based approaches like Encoder-Decoder models from the BERT series (e.g., SpanBERT), specialized reranking models such as Cohere rerank or bge-raranker-large, and general large language models like GPT [12], [99].
2) Context Selection/Compression: A common misconception in the RAG process is the belief that retrieving as many relevant documents as possible and concatenating them to form a lengthy retrieval prompt is beneficial. However, excessive context can introduce more noise, diminishing the LLM's perception of key information .

(Long) LLMLingua [100], [101] utilize small language models (SLMs) such as GPT-2 Small or LLaMA-7B, to detect and remove unimportant tokens, transforming it into a form that is challenging for humans to comprehend but well understood by LLMs. This approach presents a direct and practical method for prompt compression, eliminating the need for additional training of LLMs while balancing language integrity and compression ratio. PRCA tackled this issue by training an information extractor [69]. Similarly, RECOMP adopts a comparable approach by training an information condenser using contrastive learning [71]. Each training data point consists of one positive sample and five negative samples, and the encoder undergoes training using contrastive loss throughout this process [102] .

In addition to compressing the context, reducing the number of documents aslo helps improve the accuracy of the model's answers. Ma et al. [103] propose the "Filter-Reranker" paradigm, which combines the strengths of LLMs and SLMs.
In this paradigm, SLMs serve as filters, while LLMs function as reordering agents. The research shows that instructing LLMs to rearrange challenging samples identified by SLMs leads to significant improvements in various Information Extraction (IE) tasks. Another straightforward and effective approach involves having the LLM evaluate the retrieved content before generating the final answer. This allows the LLM to filter out documents with poor relevance through LLM critique. For instance, in Chatlaw [104], the LLM is prompted to self-suggestion on the referenced legal provisions to assess their relevance.

## B. LLM Fine-tuning

Targeted fine-tuning based on the scenario and data characteristics on LLMs can yield better results. This is also one of the greatest advantages of using on-premise LLMs. When LLMs lack data in a specific domain, additional knowledge can be provided to the LLM through fine-tuning. Huggingface's fine-tuning data can also be used as an initial step.

Another benefit of fine-tuning is the ability to adjust the model's input and output. For example, it can enable LLM to adapt to specific data formats and generate responses in a particular style as instructed [37]. For retrieval tasks that engage with structured data, the SANTA framework [76] implements a tripartite training regimen to effectively encapsulate both structural and semantic nuances. The initial phase focuses on the retriever, where contrastive learning is harnessed to refine the query and document embeddings.

Aligning LLM outputs with human or retriever preferences through reinforcement learning is a potential approach. For instance, manually annotating the final generated answers and then providing feedback through reinforcement learning. In addition to aligning with human preferences, it is also possible to align with the preferences of fine-tuned models and retrievers [79]. When circumstances prevent access to powerful proprietary models or larger parameter open-source models, a simple and effective method is to distill the more powerful models(e.g. GPT-4). Fine-tuning of LLM can also be coordinated with fine-tuning of the retriever to align preferences. A typical approach, such as RA-DIT [27], aligns the scoring functions between Retriever and Generator using KL divergence.

## V. AUGMENTATION PROCESS IN RAG

In the domain of RAG, the standard practice often involves a singular (once) retrieval step followed by generation, which can lead to inefficiencies and sometimes is typically insufficient for complex problems demanding multi-step reasoning, as it provides a limited scope of information [105]. Many studies have optimized the retrieval process in response to this issue, and we have summarised them in Figure 5.

## A. Iterative Retrieval

Iterative retrieval is a process where the knowledge base is repeatedly searched based on the initial query and the text generated so far, providing a more comprehensive knowledge
![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-11.jpg?height=786&width=1786&top_left_y=194&top_left_x=168)

Fig. 5. In addition to the most common once retrieval, RAG also includes three types of retrieval augmentation processes. (left) Iterative retrieval involves alternating between retrieval and generation, allowing for richer and more targeted context from the knowledge base at each step. (Middle) Recursive retrieval involves gradually refining the user query and breaking down the problem into sub-problems, then continuously solving complex problems through retrieval and generation. (Right) Adaptive retrieval focuses on enabling the RAG system to autonomously determine whether external knowledge retrieval is necessary and when to stop retrieval and generation, often utilizing LLM-generated special tokens for control.

base for LLMs. This approach has been shown to enhance the robustness of subsequent answer generation by offering additional contextual references through multiple retrieval iterations. However, it may be affected by semantic discontinuity and the accumulation of irrelevant information. ITERRETGEN [14] employs a synergistic approach that leverages "retrieval-enhanced generation" alongside "generationenhanced retrieval" for tasks that necessitate the reproduction of specific information. The model harnesses the content required to address the input task as a contextual basis for retrieving pertinent knowledge, which in turn facilitates the generation of improved responses in subsequent iterations.

## B. Recursive Retrieval

Recursive retrieval is often used in information retrieval and NLP to improve the depth and relevance of search results. The process involves iteratively refining search queries based on the results obtained from previous searches. Recursive Retrieval aims to enhance the search experience by gradually converging on the most pertinent information through a feedback loop. IRCoT [61] uses chain-of-thought to guide the retrieval process and refines the CoT with the obtained retrieval results. ToC [57] creates a clarification tree that systematically optimizes the ambiguous parts in the Query. It can be particularly useful in complex search scenarios where the user's needs are not entirely clear from the outset or where the information sought is highly specialized or nuanced. The recursive nature of the process allows for continuous learning and adaptation to the user's requirements, often resulting in improved satisfaction with the search outcomes.

To address specific data scenarios, recursive retrieval and multi-hop retrieval techniques are utilized together. Recursive retrieval involves a structured index to process and retrieve data in a hierarchical manner, which may include summarizing sections of a document or lengthy PDF before performing a retrieval based on this summary. Subsequently, a secondary retrieval within the document refines the search, embodying the recursive nature of the process. In contrast, multi-hop retrieval is designed to delve deeper into graph-structured data sources, extracting interconnected information [106].

## C. Adaptive Retrieval

Adaptive retrieval methods, exemplified by Flare [24] and Self-RAG [25], refine the RAG framework by enabling LLMs to actively determine the optimal moments and content for retrieval, thus enhancing the efficiency and relevance of the information sourced.

These methods are part of a broader trend wherein LLMs employ active judgment in their operations, as seen in model agents like AutoGPT, Toolformer, and GraphToolformer [107]-[109]. Graph-Toolformer, for instance, divides its retrieval process into distinct steps where LLMs proactively use retrievers, apply Self-Ask techniques, and employ few-shot prompts to initiate search queries. This proactive stance allows LLMs to decide when to search for necessary information, akin to how an agent utilizes tools.

WebGPT [110] integrates a reinforcement learning framework to train the GPT-3 model in autonomously using a search engine during text generation. It navigates this process using special tokens that facilitate actions such as search engine queries, browsing results, and citing references, thereby expanding GPT-3's capabilities through the use of external search engines. Flare automates timing retrieval by monitoring the confidence of the generation process, as indicated by the
probability of generated terms [24]. When the probability falls below a certain threshold would activates the retrieval system to collect relevant information, thus optimizing the retrieval cycle. Self-RAG [25] introduces "reflection tokens" that allow the model to introspect its outputs. These tokens come in two varieties: "retrieve" and "critic". The model autonomously decides when to activate retrieval, or alternatively, a predefined threshold may trigger the process. During retrieval, the generator conducts a fragment-level beam search across multiple paragraphs to derive the most coherent sequence. Critic scores are used to update the subdivision scores, with the flexibility to adjust these weights during inference, tailoring the model's behavior. Self-RAG's design obviates the need for additional classifiers or reliance on Natural Language Inference (NLI) models, thus streamlining the decision-making process for when to engage retrieval mechanisms and improving the model's autonomous judgment capabilities in generating accurate responses.

## VI. TASK and EVALUATION

The rapid advancement and growing adoption of RAG in the field of NLP have propelled the evaluation of RAG models to the forefront of research in the LLMs community. The primary objective of this evaluation is to comprehend and optimize the performance of RAG models across diverse application scenarios. This chapter will mainly introduce the main downstream tasks of RAG, datasets, and how to evaluate RAG systems.

## A. Downstream Task

The core task of RAG remains Question Answering (QA), including traditional single-hop/multi-hop QA, multiplechoice, domain-specific QA as well as long-form scenarios suitable for RAG. In addition to QA, RAG is continuously being expanded into multiple downstream tasks, such as Information Extraction (IE), dialogue generation, code search, etc. The main downstream tasks of RAG and their corresponding datasets are summarized in Table II

## B. Evaluation Target

Historically, RAG models assessments have centered on their execution in specific downstream tasks. These evaluations employ established metrics suitable to the tasks at hand. For instance, question answering evaluations might rely on EM and F1 scores [7], [45], [59], [72], whereas fact-checking tasks often hinge on Accuracy as the primary metric [4], [14], [42]. BLEU and ROUGE metrics are also commonly used to evaluate answer quality [26], [32], [52], [78]. Tools like RALLE, designed for the automatic evaluation of RAG applications, similarly base their assessments on these taskspecific metrics [160]. Despite this, there is a notable paucity of research dedicated to evaluating the distinct characteristics of RAG models. The main evaluation objectives include:

Retrieval Quality. Evaluating the retrieval quality is crucial for determining the effectiveness of the context sourced by the retriever component. Standard metrics from the domains of search engines, recommendation systems, and information retrieval systems are employed to measure the performance of the RAG retrieval module. Metrics such as Hit Rate, MRR, and NDCG are commonly utilized for this purpose [161], [162].

Generation Quality. The assessment of generation quality centers on the generator's capacity to synthesize coherent and relevant answers from the retrieved context. This evaluation can be categorized based on the content's objectives: unlabeled and labeled content. For unlabeled content, the evaluation encompasses the faithfulness, relevance, and non-harmfulness of the generated answers. In contrast, for labeled content, the focus is on the accuracy of the information produced by the model [161]. Additionally, both retrieval and generation quality assessments can be conducted through manual or automatic evaluation methods [29], [161], [163].

## C. Evaluation Aspects

Contemporary evaluation practices of RAG models emphasize three primary quality scores and four essential abilities, which collectively inform the evaluation of the two principal targets of the RAG model: retrieval and generation.

1) Quality Scores: Quality scores include context relevance, answer faithfulness, and answer relevance. These quality scores evaluate the efficiency of the RAG model from different perspectives in the process of information retrieval and generation [164]-[166].

Context Relevance evaluates the precision and specificity of the retrieved context, ensuring relevance and minimizing processing costs associated with extraneous content.

Answer Faithfulness ensures that the generated answers remain true to the retrieved context, maintaining consistency and avoiding contradictions.

Answer Relevance requires that the generated answers are directly pertinent to the posed questions, effectively addressing the core inquiry.

2) Required Abilities: RAG evaluation also encompasses four abilities indicative of its adaptability and efficiency: noise robustness, negative rejection, information integration, and counterfactual robustness [167], [168]. These abilities are critical for the model's performance under various challenges and complex scenarios, impacting the quality scores.

Noise Robustness appraises the model's capability to manage noise documents that are question-related but lack substantive information.

Negative Rejection assesses the model's discernment in refraining from responding when the retrieved documents do not contain the necessary knowledge to answer a question.

Information Integration evaluates the model's proficiency in synthesizing information from multiple documents to address complex questions.

Counterfactual Robustness tests the model's ability to recognize and disregard known inaccuracies within documents, even when instructed about potential misinformation.

Context relevance and noise robustness are important for evaluating the quality of retrieval, while answer faithfulness, answer relevance, negative rejection, information integration, and counterfactual robustness are important for evaluating the quality of generation.

TABLE II

DOWNSTREAM TASKS AND DATASETS OF RAG

| Task | Sub Task | Dataset | Method |
| :---: | :---: | :---: | :---: |
| QA | Single-hop | Natural Qustion(NQ) [111] | $[26],[30],[34],[42],[45],[50],[52],[59],[64], \sqrt{82}$ <br> $[3],[4],[22], \mid 27],[40],[43],[54],[62],[71], 412]$ <br> $[20],[44],[72]$ |
|  |  | TriviaQA(TQA) [113] | [4], [27], [59], [62], [112] |
|  |  | SQuAD [114] | ![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-13.jpg?height=64&width=767&top_left_y=584&top_left_x=1318) |
|  |  | Web Questions(WebQ) 115 | $[3],[4],[13], \mid 30],[50 \mid,[68]$ |
|  |  | PopQA $\mid 116$ | [7], [25], 67] |
|  |  | MS MARCO 117 | $[4],[\overline{40}],[\overline{52}]$ |
|  | Multi-hop | HotpotQA [118] | ![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-13.jpg?height=92&width=767&top_left_y=779&top_left_x=1318) |
|  |  | 2WikiMultiHopQA 119] | [14], [24], [48], [59], [61], 91$]$ |
|  |  | MuSiQue [120] | [14], [51], [61], \| |91] |
|  | Long-form QA | ELI5 [121] | [27], [34], [43], 449], [51] |
|  |  | NarrativeQA(NQA) [122] | $[\overline{45}], \mid \overline{60}],\|\overline{63}\|, \mid \overline{123}]$ |
|  |  | ASQA [124] | $[24],[57]$ |
|  |  | QMSum(QM) 125 | [60], 123] |
|  | Domain QA | Qasper [126] | 60,63 |
|  |  | COVID-QA [127] | $[35],[\overline{46}]$ |
|  |  | CMB 128],MMCU_Medical 129] | $[81]$ |
|  | Multi-Choice QA | QuALITY [130] | [60], 63] |
|  |  | ARC $[131]$ | [25, 67] |
|  |  | CommonsenseQA 132 | $[\overline{58}],[\overline{66}]$ |
|  | Graph QA | GraphQA [84] | [84] |
| Dialog | Dialog Generation | Wizard of Wikipedia (WoW) [133] | [13], [27], [34], [42] |
|  | Personal Dialog | $\mathrm{KBP}[134] \quad 1$ | [74], 135$]$ |
|  |  | DuleMon [136] | $[74]$ |
|  | Task-oriented Dialog | CamRest \|137| | $\|78\|, \mid 79]$ |
|  | Recommendation | Amazon(Toys,Sport,Beauty) 138 | $[\overline{39}],[\overline{40}]$ |
| IE | Event Argument Extraction | WikiEvent [139] | [13], [27], [37], [42] |
|  |  | RAMS 140$]$ | [36], [37] |
|  | Relation Extraction | T-REx [141],ZsRE [142] | [27, ,51] |
| Reasoning | Commonsense Reasoning | HellaSwag [143] | [20], [66] |
|  | CoT Reasoning | CoT Reasoning $\mid 144$ | [27] |
|  | Complex Reasoning | CSQA [145] | $[\overline{55}]$ |
| Others | Language Understanding | MMLU 146$]$ | [7], [27], [28], [42], [43], [47], [72] |
|  | Language Modeling | WikiText-103 [147] | $[5],[29], \mid 64],[71]$ |
|  |  | StrategyQA $\mid 148$ | $[14],[24],\lfloor 48],[51],[55],[58]$ |
|  | Fact Checking/Verification | FEVER $[149]$ | $[4],[13],[27],[34],[42],[50]$ |
|  |  | PubHealth [150] | $[25], 67]$ |
|  | Text Generation | Biography [151] | [67 |
|  | Text Summarization | WikiASP [152] | [24] |
|  |  | XSum $153 \rrbracket$ | [17] |
|  | Text Classification | VioLens [154] | [19] |
|  |  | TREC [155] | [33] |
|  | Sentiment | SST-2 [156] | [20], [33], [38] |
|  | Code Search | CodeSearchNet [157] | [76] |
|  | Robustness Evaluation | NoMIRACL [56] | [56] |
|  | Math | GSM8K $[158]$ | [73] |
|  | Machine Translation | JRC-Acquis [159] | $[\overline{17}]$ |

TABLE III

SUMMARY OF METRICS APPLICABLE FOR EVALUATION ASPECTS OF RAG

|  | Context <br> Relevance | Faithfulness | Answer <br> Relevance | Noise <br> Robustness | Negative <br> Rejection | Information <br> Integration | Counterfactual <br> Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| EM |  |  |  |  | $\checkmark$ |  |  |
| Recall | $\checkmark$ |  |  |  |  |  |  |
| Precision | $\checkmark$ |  |  |  |  |  |  |
| R-Rate |  |  |  |  |  |  |  |
| Cosine Similarity |  |  |  |  |  |  |  |
| Hit Rate | $\checkmark$ |  |  |  |  |  |  |
| MRR | $\checkmark$ |  |  |  |  |  |  |
| NDCG | $\checkmark$ |  |  |  |  |  |  |
| BLEU | $\checkmark$ | $\checkmark$ | $\checkmark$ |  |  |  |  |
| ROUGE/ROUGE-L | $\checkmark$ | $\checkmark$ | $\checkmark$ |  |  |  |  |

The specific metrics for each evaluation aspect are summarized in Table III. It is essential to recognize that these metrics, derived from related work, are traditional measures and do not yet represent a mature or standardized approach for quantifying RAG evaluation aspects. Custom metrics tailored to the nuances of RAG models, though not included here, have also been developed in some evaluation studies.

## D. Evaluation Benchmarks and Tools

A series of benchmark tests and tools have been proposed to facilitate the evaluation of RAG.These instruments furnish quantitative metrics that not only gauge RAG model performance but also enhance comprehension of the model's capabilities across various evaluation aspects. Prominent benchmarks such as RGB, RECALL and CRUD [167]-[169] focus on appraising the essential abilities of RAG models. Concurrently, state-of-the-art automated tools like RAGAS [164, ARES [165], and TruLen $8^{8}$ employ LLMs to adjudicate the quality scores. These tools and benchmarks collectively form a robust framework for the systematic evaluation of RAG models, as summarized in Table IV

## VII. DisCUSSION ANd FUTURE PROSPECTS

Despite the considerable progress in RAG technology, several challenges persist that warrant in-depth research.This chapter will mainly introduce the current challenges and future research directions faced by RAG.

## A. RAG vs Long Context

With the deepening of related research, the context of LLMs is continuously expanding [170]-[172]. Presently, LLMs can effortlessly manage contexts exceeding 200,000 tokens 9 . This capability signifies that long-document question answering, previously reliant on RAG, can now incorporate the entire document directly into the prompt. This has also sparked discussions on whether RAG is still necessary when LLMs[^4]

are not constrained by context. In fact, RAG still plays an irreplaceable role. On one hand, providing LLMs with a large amount of context at once will significantly impact its inference speed, while chunked retrieval and on-demand input can significantly improve operational efficiency. On the other hand, RAG-based generation can quickly locate the original references for LLMs to help users verify the generated answers. The entire retrieval and reasoning process is observable, while generation solely relying on long context remains a black box. Conversely, the expansion of context provides new opportunities for the development of RAG, enabling it to address more complex problems and integrative or summary questions that require reading a large amount of material to answer [49]. Developing new RAG methods in the context of super-long contexts is one of the future research trends.

## B. RAG Robustness

The presence of noise or contradictory information during retrieval can detrimentally affect RAG's output quality. This situation is figuratively referred to as "Misinformation can be worse than no information at all". Improving RAG's resistance to such adversarial or counterfactual inputs is gaining research momentum and has become a key performance metric [48], [50], [82]. Cuconasu et al. [54] analyze which type of documents should be retrieved, evaluate the relevance of the documents to the prompt, their position, and the number included in the context. The research findings reveal that including irrelevant documents can unexpectedly increase accuracy by over $30 \%$, contradicting the initial assumption of reduced quality. These results underscore the importance of developing specialized strategies to integrate retrieval with language generation models, highlighting the need for further research and exploration into the robustness of RAG.

## C. Hybrid Approaches

Combining RAG with fine-tuning is emerging as a leading strategy. Determining the optimal integration of RAG and fine-tuning whether sequential, alternating, or through end-toend joint training-and how to harness both parameterized

TABLE IV

SUMMARY OF EVALUATION FRAMEWORKS

| Evaluation Framework | Evaluation Targets | Evaluation Aspects | Quantitative Metrics |
| :---: | :---: | :---: | :---: |
| $\mathrm{RGB}^{\dagger}$ | Retrieval Quality <br> Generation Quality | Noise Robustness <br> Negative Rejection <br> Information Integration <br> Counterfactual Robustness | Accuracy <br> EM <br> Accuracy <br> Accuracy |
| RECALL $^{\dagger}$ | Generation Quality | Counterfactual Robustness | R-Rate (Reappearance Rate) |
| RAGAS $^{\ddagger}$ | Retrieval Quality <br> Generation Quality | Context Relevance <br> Faithfulness <br> Answer Relevance | $*$ <br> $*$ <br> Cosine Similarity |
| $\mathrm{ARES}^{\ddagger}$ | Retrieval Quality <br> Generation Quality | Context Relevance <br> Faithfulness <br> Answer Relevance | Accuracy <br> Accuracy <br> Accuracy |
| TruLens $^{\ddagger}$ | Retrieval Quality <br> Generation Quality | Context Relevance <br> Faithfulness <br> Answer Relevance | $*$ <br> $*$ <br> $*$ |
| CRUD ${ }^{\dagger}$ | Retrieval Quality <br> Generation Quality | Creative Generation <br> Knowledge-intensive QA <br> Error Correction <br> Summarization | BLEU <br> ROUGE-L <br> BertScore <br> RAGQuestEval |

$\dagger$ represents a benchmark, and $\ddagger$ represents a tool. * denotes customized quantitative metrics, which deviate from traditional metrics. Readers are encouraged to consult pertinent literature for the specific quantification formulas associated with these metrics, as required.

and non-parameterized advantages are areas ripe for exploration [27]. Another trend is to introduce SLMs with specific functionalities into RAG and fine-tuned by the results of RAG system. For example, CRAG [67] trains a lightweight retrieval evaluator to assess the overall quality of the retrieved documents for a query and triggers different knowledge retrieval actions based on confidence levels.

## D. Scaling laws of RAG

End-to-end RAG models and pre-trained models based on RAG are still one of the focuses of current researchers [173].The parameters of these models are one of the key factors. While scaling laws [174] are established for LLMs, their applicability to RAG remains uncertain. Initial studies like RETRO++ [44] have begun to address this, yet the parameter count in RAG models still lags behind that of LLMs. The possibility of an Inverse Scaling Law 10 , where smaller models outperform larger ones, is particularly intriguing and merits further investigation.

## E. Production-Ready RAG

RAG's practicality and alignment with engineering requirements have facilitated its adoption. However, enhancing retrieval efficiency, improving document recall in large knowledge bases, and ensuring data security-such as preventing[^5]

inadvertent disclosure of document sources or metadata by LLMs-are critical engineering challenges that remain to be addressed [175].

The development of the RAG ecosystem is greatly impacted by the progression of its technical stack. Key tools like LangChain and LLamaIndex have quickly gained popularity with the emergence of ChatGPT, providing extensive RAGrelated APIs and becoming essential in the realm of LLMs.The emerging technology stack, while not as rich in features as LangChain and LLamaIndex, stands out through its specialized products. For example, Flowise AI prioritizes a low-code approach, allowing users to deploy $\mathrm{AI}$ applications, including RAG, through a user-friendly drag-and-drop interface. Other technologies like HayStack, Meltano, and Cohere Coral are also gaining attention for their unique contributions to the field.

In addition to AI-focused vendors, traditional software and cloud service providers are expanding their offerings to include RAG-centric services. Weaviate's Verba $\square$ is designed for personal assistant applications, while Amazon's Kendra ${ }^{12}$ offers intelligent enterprise search services, enabling users to browse various content repositories using built-in connectors. In the development of RAG technology, there is a clear trend towards different specialization directions, such as: 1) Customization - tailoring RAG to meet specific requirements. 2) Simplification - making RAG easier to use to reduce the[^6]

![](https://cdn.mathpix.com/cropped/2024_06_04_e6055c1a77633d6d9fc5g-16.jpg?height=902&width=1461&top_left_y=178&top_left_x=321)

Fig. 6. Summary of RAG ecosystem

initial learning curve. 3) Specialization - optimizing RAG to better serve production environments.

The mutual growth of RAG models and their technology stacks is evident; technological advancements continuously establish new standards for existing infrastructure. In turn, enhancements to the technology stack drive the development of RAG capabilities. RAG toolkits are converging into a foundational technology stack, laying the groundwork for advanced enterprise applications. However, a fully integrated, comprehensive platform concept is still in the future, requiring further innovation and development.

## F. Multi-modal RAG

RAG has transcended its initial text-based questionanswering confines, embracing a diverse array of modal data. This expansion has spawned innovative multimodal models that integrate RAG concepts across various domains:

Image. RA-CM3 [176] stands as a pioneering multimodal model of both retrieving and generating text and images. BLIP-2 [177] leverages frozen image encoders alongside LLMs for efficient visual language pre-training, enabling zeroshot image-to-text conversions. The "Visualize Before You Write" method [178] employs image generation to steer the LM's text generation, showing promise in open-ended text generation tasks.

Audio and Video. The GSS method retrieves and stitches together audio clips to convert machine-translated data into speech-translated data [179]. UEOP marks a significant advancement in end-to-end automatic speech recognition by incorporating external, offline strategies for voice-to-text conversion [180]. Additionally, KNN-based attention fusion leverages audio embeddings and semantically related text embeddings to refine ASR, thereby accelerating domain adaptation.
Vid2Seq augments language models with specialized temporal markers, facilitating the prediction of event boundaries and textual descriptions within a unified output sequence [181].

Code. RBPS [182] excels in small-scale learning tasks by retrieving code examples that align with developers' objectives through encoding and frequency analysis. This approach has demonstrated efficacy in tasks such as test assertion generation and program repair. For structured knowledge, the CoK method [106] first extracts facts pertinent to the input query from a knowledge graph, then integrates these facts as hints within the input, enhancing performance in knowledge graph question-answering tasks.

## VIII. CONCLUSION

The summary of this paper, as depicted in Figure 6, emphasizes RAG's significant advancement in enhancing the capabilities of LLMs by integrating parameterized knowledge from language models with extensive non-parameterized data from external knowledge bases. The survey showcases the evolution of RAG technologies and their application on many different tasks. The analysis outlines three developmental paradigms within the RAG framework: Naive, Advanced, and Modular RAG, each representing a progressive enhancement over its predecessors. RAG's technical integration with other AI methodologies, such as fine-tuning and reinforcement learning, has further expanded its capabilities. Despite the progress in RAG technology, there are research opportunities to improve its robustness and its ability to handle extended contexts. RAG's application scope is expanding into multimodal domains, adapting its principles to interpret and process diverse data forms like images, videos, and code. This expansion highlights RAG's significant practical implications for AI deployment, attracting interest from academic and industrial sectors.

The growing ecosystem of RAG is evidenced by the rise in RAG-centric AI applications and the continuous development of supportive tools. As RAG's application landscape broadens, there is a need to refine evaluation methodologies to keep pace with its evolution. Ensuring accurate and representative performance assessments is crucial for fully capturing RAG's contributions to the AI research and development community.

## REFERENCES

[1] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, "Large language models struggle to learn long-tail knowledge," in International Conference on Machine Learning. PMLR, 2023, pp. 15 69615707 .

[2] Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao, Y. Zhang, Y. Chen et al., "Siren's song in the ai ocean: A survey on hallucination in large language models," arXiv preprint arXiv:2309.01219, 2023.

[3] D. Arora, A. Kini, S. R. Chowdhury, N. Natarajan, G. Sinha, and A. Sharma, "Gar-meets-rag paradigm for zero-shot information retrieval," arXiv preprint arXiv:2310.20158, 2023.

[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., "Retrievalaugmented generation for knowledge-intensive nlp tasks," Advances in Neural Information Processing Systems, vol. 33, pp. 9459-9474, 2020.

[5] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark et al., "Improving language models by retrieving from trillions of tokens," in International conference on machine learning. PMLR, 2022, pp. 2206-2240.

[6] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., "Training language models to follow instructions with human feedback," Advances in neural information processing systems, vol. 35, pp. 27 730-27744, 2022.

[7] X. Ma, Y. Gong, P. He, H. Zhao, and N. Duan, "Query rewriting for retrieval-augmented large language models," arXiv preprint arXiv:2305.14283, 2023.

[8] I. ILIN, "Advanced rag techniques: an illustrated overview," https://pub.towardsai.net/ advanced-rag-techniques-an-illustrated-overview-04d193d8fec6 2023.

[9] W. Peng, G. Li, Y. Jiang, Z. Wang, D. Ou, X. Zeng, E. Chen et al., "Large language model based long-tail query rewriting in taobao search," arXiv preprint arXiv:2311.03758, 2023.

[10] H. S. Zheng, S. Mishra, X. Chen, H.-T. Cheng, E. H. Chi, Q. V. Le, and D. Zhou, "Take a step back: Evoking reasoning via abstraction in large language models," arXiv preprint arXiv:2310.06117, 2023.

[11] L. Gao, X. Ma, J. Lin, and J. Callan, "Precise zero-shot dense retrieval without relevance labels," arXiv preprint arXiv:2212.10496, 2022.

[12] V. Blagojevi, "Enhancing rag pipelines in haystack: Introducing diversityranker and lostinthemiddleranker," https://towardsdatascience.com/ enhancing-rag-pipelines-in-haystack-45f14e2bc9f5 2023.

[13] W. Yu, D. Iter, S. Wang, Y. Xu, M. Ju, S. Sanyal, C. Zhu, M. Zeng, and M. Jiang, "Generate rather than retrieve: Large language models are strong context generators," arXiv preprint arXiv:2209.10063, 2022.

[14] Z. Shao, Y. Gong, Y. Shen, M. Huang, N. Duan, and W. Chen, "Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy," arXiv preprint arXiv:2305.15294, 2023.

[15] X. Wang, Q. Yang, Y. Qiu, J. Liang, Q. He, Z. Gu, Y. Xiao, and W. Wang, "Knowledgpt: Enhancing large language models with retrieval and storage access on knowledge bases," arXiv preprint arXiv:2308.11761, 2023

[16] A. H. Raudaschl, "Forget rag, the future is rag-fusion," https://towardsdatascience.com/ forget-rag-the-future-is-rag-fusion-1147298d8ad1 2023.

[17] X. Cheng, D. Luo, X. Chen, L. Liu, D. Zhao, and R. Yan, "Lift yourself up: Retrieval-augmented text generation with self memory," arXiv preprint arXiv:2305.02437, 2023.

[18] S. Wang, Y. Xu, Y. Fang, Y. Liu, S. Sun, R. Xu, C. Zhu, and M. Zeng, "Training data is more valuable than you think: A simple and effective method by retrieving from training data," arXiv preprint arXiv:2203.08773, 2022.
[19] X. Li, E. Nie, and S. Liang, "From classification to generation: Insights into crosslingual retrieval augmented icl," arXiv preprint arXiv:2311.06595, 2023.

[20] D. Cheng, S. Huang, J. Bi, Y. Zhan, J. Liu, Y. Wang, H. Sun, F. Wei, D. Deng, and Q. Zhang, "Uprise: Universal prompt retrieval for improving zero-shot evaluation," arXiv preprint arXiv:2303.08518, 2023.

[21] Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. B. Hall, and M.-W. Chang, "Promptagator: Few-shot dense retrieval from 8 examples," arXiv preprint arXiv:2209.11755, 2022.

[22] Z. Sun, X. Wang, Y. Tay, Y. Yang, and D. Zhou, "Recitation-augmented language models," arXiv preprint arXiv:2210.01296, 2022.

[23] O. Khattab, K. Santhanam, X. L. Li, D. Hall, P. Liang, C. Potts, and M. Zaharia, "Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp," arXiv preprint arXiv:2212.14024, 2022.

[24] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig, "Active retrieval augmented generation," arXiv preprint arXiv:2305.06983, 2023.

[25] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-rag: Learning to retrieve, generate, and critique through self-reflection," arXiv preprint arXiv:2310.11511, 2023.

[26] Z. Ke, W. Kong, C. Li, M. Zhang, Q. Mei, and M. Bendersky, "Bridging the preference gap between retrievers and llms," arXiv preprint arXiv:2401.06954, 2024.

[27] X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis et al., "Ra-dit: Retrievalaugmented dual instruction tuning," arXiv preprint arXiv:2310.01352, 2023.

[28] O. Ovadia, M. Brief, M. Mishaeli, and O. Elisha, "Fine-tuning or retrieval? comparing knowledge injection in llms," arXiv preprint arXiv:2312.05934, 2023.

[29] T. Lan, D. Cai, Y. Wang, H. Huang, and X.-L. Mao, "Copy is all you need," in The Eleventh International Conference on Learning Representations, 2022

[30] T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, D. Yu, and H. Zhang, "Dense x retrieval: What retrieval granularity should we use?" arXiv preprint arXiv:2312.06648, 2023.

[31] F. Luo and M. Surdeanu, "Divide \& conquer for entailment-aware multi-hop evidence retrieval," arXiv preprint arXiv:2311.02616, 2023.

[32] Q. Gou, Z. Xia, B. Yu, H. Yu, F. Huang, Y. Li, and N. Cam-Tu, "Diversify question generation with retrieval-augmented style transfer," arXiv preprint arXiv:2310.14503, 2023.

[33] Z. Guo, S. Cheng, Y. Wang, P. Li, and Y. Liu, "Prompt-guided retrieval augmentation for non-knowledge-intensive tasks," arXiv preprint arXiv:2305.17653, 2023.

[34] Z. Wang, J. Araki, Z. Jiang, M. R. Parvez, and G. Neubig, "Learning to filter context for retrieval-augmented generation," arXiv preprint arXiv:2311.08377, 2023.

[35] M. Seo, J. Baek, J. Thorne, and S. J. Hwang, "Retrieval-augmented data augmentation for low-resource domain tasks," arXiv preprint arXiv:2402.13482, 2024.

[36] Y. Ma, Y. Cao, Y. Hong, and A. Sun, "Large language model is not a good few-shot information extractor, but a good reranker for hard samples!" arXiv preprint arXiv:2303.08559, 2023.

[37] X. Du and H. Ji, "Retrieval-augmented generative question answering for event argument extraction," arXiv preprint arXiv:2211.07067, 2022.

[38] L. Wang, N. Yang, and F. Wei, "Learning to retrieve in-context examples for large language models," arXiv preprint arXiv:2307.07164, 2023.

[39] S. Rajput, N. Mehta, A. Singh, R. H. Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Q. Tran, J. Samost et al., "Recommender systems with generative retrieval," arXiv preprint arXiv:2305.05065, 2023.

[40] B. Jin, H. Zeng, G. Wang, X. Chen, T. Wei, R. Li, Z. Wang, Z. Li, Y. Li, H. Lu et al., "Language models as semantic indexers," arXiv preprint arXiv:2310.07815, 2023.

[41] R. Anantha, T. Bethi, D. Vodianik, and S. Chappidi, "Context tuning for retrieval augmented generation," arXiv preprint arXiv:2312.05708, 2023.

[42] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave, "Few-shot learning with retrieval augmented language models," arXiv preprint arXiv:2208.03299, 2022.

[43] J. Huang, W. Ping, P. Xu, M. Shoeybi, K. C.-C. Chang, and B. Catanzaro, "Raven: In-context learning with retrieval augmented encoderdecoder language models," arXiv preprint arXiv:2308.07922, 2023.

[44] B. Wang, W. Ping, P. Xu, L. McAfee, Z. Liu, M. Shoeybi, Y. Dong, O. Kuchaiev, B. Li, C. Xiao et al., "Shall we pretrain autoregressive language models with retrieval? a comprehensive study," arXiv preprint arXiv:2304.06762, 2023.

[45] B. Wang, W. Ping, L. McAfee, P. Xu, B. Li, M. Shoeybi, and B. Catanzaro, "Instructretro: Instruction tuning post retrieval-augmented pretraining," arXiv preprint arXiv:2310.07713, 2023.

[46] S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi, R. Rana, and S. Nanayakkara, "Improving the domain adaptation of retrieval augmented generation (rag) models for open domain question answering," Transactions of the Association for Computational Linguistics, vol. 11, pp. 1-17, 2023.

[47] Z. Yu, C. Xiong, S. Yu, and Z. Liu, "Augmentation-adapted retriever improves generalization of language models as generic plug-in," arXiv preprint arXiv:2305.17331, 2023.

[48] O. Yoran, T. Wolfson, O. Ram, and J. Berant, "Making retrievalaugmented language models robust to irrelevant context," arXiv preprint arXiv:2310.01558, 2023.

[49] H.-T. Chen, F. Xu, S. A. Arora, and E. Choi, "Understanding retrieval augmentation for long-form question answering," arXiv preprint arXiv:2310.12150, 2023.

[50] W. Yu, H. Zhang, X. Pan, K. Ma, H. Wang, and D. Yu, "Chain-of-note: Enhancing robustness in retrieval-augmented language models," arXiv preprint arXiv:2311.09210, 2023.

[51] S. Xu, L. Pang, H. Shen, X. Cheng, and T.-S. Chua, "Search-in-thechain: Towards accurate, credible and traceable large language models for knowledgeintensive tasks," CoRR, vol. abs/2304.14732, 2023.

[52] M. Berchansky, P. Izsak, A. Caciularu, I. Dagan, and M. Wasserblat, "Optimizing retrieval-augmented reader models via token elimination," arXiv preprint arXiv:2310.13682, 2023.

[53] J. Lála, O. O’Donoghue, A. Shtedritski, S. Cox, S. G. Rodriques, and A. D. White, "Paperqa: Retrieval-augmented generative agent for scientific research," arXiv preprint arXiv:2312.07559, 2023.

[54] F. Cuconasu, G. Trappolini, F. Siciliano, S. Filice, C. Campagnano, Y. Maarek, N. Tonellotto, and F. Silvestri, "The power of noise: Redefining retrieval for rag systems," arXiv preprint arXiv:2401.14887, 2024.

[55] Z. Zhang, X. Zhang, Y. Ren, S. Shi, M. Han, Y. Wu, R. Lai, and Z. Cao, "Iag: Induction-augmented generation framework for answering reasoning questions," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 1-14.

[56] N. Thakur, L. Bonifacio, X. Zhang, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu, B. Chen, M. Rezagholizadeh et al., "Nomiracl: Knowing when you don't know for robust multilingual retrieval-augmented generation," arXiv preprint arXiv:2312.11361, 2023.

[57] G. Kim, S. Kim, B. Jeon, J. Park, and J. Kang, "Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models," arXiv preprint arXiv:2310.14696, 2023.

[58] Y. Wang, P. Li, M. Sun, and Y. Liu, "Self-knowledge guided retrieval augmentation for large language models," arXiv preprint arXiv:2310.05002, 2023.

[59] Z. Feng, X. Feng, D. Zhao, M. Yang, and B. Qin, "Retrievalgeneration synergy augmented large language models," arXiv preprint arXiv:2310.05149, 2023

[60] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian, E. Bakhturina, M. Shoeybi, and B. Catanzaro, "Retrieval meets long context large language models," arXiv preprint arXiv:2310.03025, 2023.

[61] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions," arXiv preprint arXiv:2212.10509, 2022.

[62] R. Ren, Y. Wang, Y. Qu, W. X. Zhao, J. Liu, H. Tian, H. Wu, J.R. Wen, and H. Wang, "Investigating the factual knowledge boundary of large language models with retrieval augmentation," arXiv preprint arXiv:2307.11019, 2023

[63] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D Manning, "Raptor: Recursive abstractive processing for tree-organized retrieval," arXiv preprint arXiv:2401.18059, 2024.

[64] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. LeytonBrown, and Y. Shoham, "In-context retrieval-augmented language models," arXiv preprint arXiv:2302.00083, 2023.

[65] Y. Ren, Y. Cao, P. Guo, F. Fang, W. Ma, and Z. Lin, "Retrieve-andsample: Document-level event argument extraction via hybrid retrieval augmentation," in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 293-306.
[66] Z. Wang, X. Pan, D. Yu, D. Yu, J. Chen, and H. Ji, "Zemi: Learning zero-shot semi-parametric language models from multiple tasks," arXiv preprint arXiv:2210.00185, 2022.

[67] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, "Corrective retrieval augmented generation," arXiv preprint arXiv:2401.15884, 2024.

[68] P. Jain, L. B. Soares, and T. Kwiatkowski, "1-pager: One pass answer generation and evidence retrieval," arXiv preprint arXiv:2310.16568, 2023.

[69] H. Yang, Z. Li, Y. Zhang, J. Wang, N. Cheng, M. Li, and J. Xiao, "Prca: Fitting black-box large language models for retrieval question answering via pluggable reward-driven contextual adapter," arXiv preprint arXiv:2310.18347, 2023.

[70] S. Zhuang, B. Liu, B. Koopman, and G. Zuccon, "Open-source large language models are strong zero-shot query likelihood models for document ranking," arXiv preprint arXiv:2310.13243, 2023.

[71] F. Xu, W. Shi, and E. Choi, "Recomp: Improving retrieval-augmented lms with compression and selective augmentation," arXiv preprint arXiv:2310.04408, 2023.

[72] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, "Replug: Retrieval-augmented black-box language models," arXiv preprint arXiv:2301.12652, 2023.

[73] E. Melz, "Enhancing llm intelligence with arm-rag: Auxiliary rationale memory for retrieval augmented generation," arXiv preprint arXiv:2311.04177, 2023.

[74] H. Wang, W. Huang, Y. Deng, R. Wang, Z. Wang, Y. Wang, F. Mi, J. Z. Pan, and K.-F. Wong, "Unims-rag: A unified multi-source retrieval-augmented generation for personalized dialogue systems," arXiv preprint arXiv:2401.13256, 2024.

[75] Z. Luo, C. Xu, P. Zhao, X. Geng, C. Tao, J. Ma, Q. Lin, and D. Jiang, "Augmented large language models with parametric knowledge guiding," arXiv preprint arXiv:2305.04757, 2023.

[76] X. Li, Z. Liu, C. Xiong, S. Yu, Y. Gu, Z. Liu, and G. Yu, "Structureaware language model pretraining improves dense retrieval on structured data," arXiv preprint arXiv:2305.19912, 2023.

[77] M. Kang, J. M. Kwak, J. Baek, and S. J. Hwang, "Knowledge graph-augmented language models for knowledge-grounded dialogue generation," arXiv preprint arXiv:2305.18846, 2023.

[78] W. Shen, Y. Gao, C. Huang, F. Wan, X. Quan, and W. Bi, "Retrievalgeneration alignment for end-to-end task-oriented dialogue system," arXiv preprint arXiv:2310.08877, 2023

[79] T. Shi, L. Li, Z. Lin, T. Yang, X. Quan, and Q. Wang, "Dual-feedback knowledge retrieval for task-oriented dialogue systems," arXiv preprint arXiv:2310.14528, 2023.

[80] P. Ranade and A. Joshi, "Fabula: Intelligence report generation using retrieval-augmented narrative construction," arXiv preprint arXiv:2310.13848, 2023

[81] X. Jiang, R. Zhang, Y. Xu, R. Qiu, Y. Fang, Z. Wang, J. Tang, H. Ding, X. Chu, J. Zhao et al., "Think and retrieval: A hypothesis knowledge graph enhanced medical large language models," arXiv preprint arXiv:2312.15883, 2023.

[82] J. Baek, S. Jeong, M. Kang, J. C. Park, and S. J. Hwang, "Knowledge-augmented language model verification," arXiv preprint arXiv:2310.12836, 2023.

[83] L. Luo, Y.-F. Li, G. Haffari, and S. Pan, "Reasoning on graphs: Faithful and interpretable large language model reasoning," arXiv preprint arXiv:2310.01061, 2023.

[84] X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi, "G-retriever: Retrieval-augmented generation for textual graph understanding and question answering," arXiv preprint arXiv:2402.07630, 2024.

[85] L. Zha, J. Zhou, L. Li, R. Wang, Q. Huang, S. Yang, J. Yuan, C. Su, X. Li, A. Su et al., "Tablegpt: Towards unifying tables, nature language and commands into one gpt," arXiv preprint arXiv:2307.08674, 2023

[86] M. Gaur, K. Gunaratna, V. Srinivasan, and H. Jin, "Iseeq: Information seeking question generation using dynamic meta-information retrieval and knowledge graphs," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 10 672-10680.

[87] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi, N. Schärli, and D. Zhou, "Large language models can be easily distracted by irrelevant context," in International Conference on Machine Learning. PMLR, 2023, pp. 31210-31227.

[88] R. Teja, "Evaluating the ideal chunk size for a rag system using llamaindex," https://www.llamaindex.ai/blog/ evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5 2023 .

[89] Langchain, "Recursively split by character," https://python.langchain. com/docs/modules/data connection/document transformers/recursive text_splitter 2023.

[90] S. Yang, "Advanced rag 01: Small-tobig retrieval," https://towardsdatascience.com/ advanced-rag-01-small-to-big-retrieval-172181b396d4 2023.

[91] Y. Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr, "Knowledge graph prompting for multi-document question answering," arXiv preprint arXiv:2308.11730, 2023.

[92] D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. Le et al., "Least-to-most prompting enables complex reasoning in large language models," arXiv preprint arXiv:2205.10625, 2022.

[93] S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu, X. Li, A. Celikyilmaz, and J. Weston, "Chain-of-verification reduces hallucination in large language models," arXiv preprint arXiv:2309.11495, 2023.

[94] X. Li and J. Li, "Angle-optimized text embeddings," arXiv preprint arXiv:2309.12871, 2023.

[95] VoyageAI, "Voyage's embedding models," https://docs.voyageai.com/ embeddings/ 2023.

[96] BAAI, "Flagembedding," https://github.com/FlagOpen/ FlagEmbedding 2023.

[97] P. Zhang, S. Xiao, Z. Liu, Z. Dou, and J.-Y. Nie, "Retrieve anything to augment large language models," arXiv preprint arXiv:2310.07554, 2023.

[98] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," arXiv preprint arXiv:2307.03172, 2023.

[99] Y. Gao, T. Sheng, Y. Xiang, Y. Xiong, H. Wang, and J. Zhang, "Chatrec: Towards interactive and explainable llms-augmented recommender system," arXiv preprint arXiv:2303.14524, 2023.

[100] N. Anderson, C. Wilson, and S. D. Richardson, "Lingua: Addressing scenarios for live interpretation and automatic dubbing," in Proceedings of the 15th Biennial Conference of the Association for Machine Translation in the Americas (Volume 2: Users and Providers Track and Government Track), J. Campbell, S. Larocca, J. Marciano, K. Savenkov, and A. Yanishevsky, Eds. Orlando, USA: Association for Machine Translation in the Americas, Sep. 2022, pp. 202-209. [Online]. Available: https://aclanthology.org/2022.amta-upg. 14

[101] H. Jiang, Q. Wu, X. Luo, D. Li, C.-Y. Lin, Y. Yang, and L. Qiu, "Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression," arXiv preprint arXiv:2310.06839, 2023.

[102] V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, "Dense passage retrieval for open-domain question answering," arXiv preprint arXiv:2004.04906, 2020.

[103] Y. Ma, Y. Cao, Y. Hong, and A. Sun, "Large language model is not a good few-shot information extractor, but a good reranker for hard samples!" ArXiv, vol. abs/2303.08559, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:257532405

[104] J. Cui, Z. Li, Y. Yan, B. Chen, and L. Yuan, "Chatlaw: Open-source legal large language model with integrated external knowledge bases," arXiv preprint arXiv:2306.16092, 2023.

[105] O. Yoran, T. Wolfson, O. Ram, and J. Berant, "Making retrievalaugmented language models robust to irrelevant context," arXiv preprint arXiv:2310.01558, 2023.

[106] X. Li, R. Zhao, Y. K. Chia, B. Ding, L. Bing, S. Joty, and S. Poria, "Chain of knowledge: A framework for grounding large language models with structured knowledge bases," arXiv preprint arXiv:2305.13269, 2023 .

[107] H. Yang, S. Yue, and Y. He, "Auto-gpt for online decision making: Benchmarks and additional opinions," arXiv preprint arXiv:2306.02224, 2023

[108] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, "Toolformer: Language models can teach themselves to use tools," arXiv preprint arXiv:2302.04761, 2023.

[109] J. Zhang, "Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt," arXiv preprint arXiv:2304.11116, 2023

[110] R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders et al., "Webgpt: Browserassisted question-answering with human feedback," arXiv preprint arXiv:2112.09332, 2021.

[111] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee et al., "Natural questions: a benchmark for question answering research," Transactions of the Association for Computational Linguistics, vol. 7, pp. 453-466, 2019 .

[112] Y. Liu, S. Yavuz, R. Meng, M. Moorthy, S. Joty, C. Xiong, and Y. Zhou, "Exploring the integration strategies of retriever and large language models," arXiv preprint arXiv:2308.12574, 2023.

[113] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, "Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension," arXiv preprint arXiv:1705.03551, 2017.

[114] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, "Squad: 100,000+ questions for machine comprehension of text," arXiv preprint arXiv:1606.05250, 2016.

[115] J. Berant, A. Chou, R. Frostig, and P. Liang, "Semantic parsing on freebase from question-answer pairs," in Proceedings of the 2013 conference on empirical methods in natural language processing, 2013, pp. 1533-1544.

[116] A. Mallen, A. Asai, V. Zhong, R. Das, H. Hajishirzi, and D. Khashabi, "When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories," arXiv preprint arXiv:2212.10511, 2022.

[117] T. Nguyen, M. Rosenberg, X. Song, J. Gao, S. Tiwary, R. Majumder, and L. Deng, "Ms marco: A human-generated machine reading comprehension dataset," 2016

[118] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, "Hotpotqa: A dataset for diverse, explainable multi-hop question answering," arXiv preprint arXiv:1809.09600, 2018.

[119] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, "Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps," arXiv preprint arXiv:2011.01060, 2020

[120] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "Musique: Multihop questions via single-hop question composition," Transactions of the Association for Computational Linguistics, vol. 10, pp. 539-554, 2022.

[121] A. Fan, Y. Jernite, E. Perez, D. Grangier, J. Weston, and M. Auli, "Eli5: Long form question answering," arXiv preprint arXiv:1907.09190, 2019 .

[122] T. Kočiskỳ, J. Schwarz, P. Blunsom, C. Dyer, K. M. Hermann, G. Melis, and E. Grefenstette, "The narrativeqa reading comprehension challenge," Transactions of the Association for Computational Linguistics, vol. 6, pp. 317-328, 2018.

[123] K.-H. Lee, X. Chen, H. Furuta, J. Canny, and I. Fischer, "A humaninspired reading agent with gist memory of very long contexts," arXiv preprint arXiv:2402.09727, 2024.

[124] I. Stelmakh, Y. Luan, B. Dhingra, and M.-W. Chang, "Asqa: Factoid questions meet long-form answers," arXiv preprint arXiv:2204.06092, 2022.

[125] M. Zhong, D. Yin, T. Yu, A. Zaidi, M. Mutuma, R. Jha, A. H. Awadallah, A. Celikyilmaz, Y. Liu, X. Qiu et al., "Qmsum: A new benchmark for query-based multi-domain meeting summarization," arXiv preprint arXiv:2104.05938, 2021.

[126] P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner, "A dataset of information-seeking questions and answers anchored in research papers," arXiv preprint arXiv:2105.03011, 2021.

[127] T. Möller, A. Reina, R. Jayakumar, and M. Pietsch, "Covid-qa: A question answering dataset for covid-19," in ACL 2020 Workshop on Natural Language Processing for COVID-19 (NLP-COVID), 2020.

[128] X. Wang, G. H. Chen, D. Song, Z. Zhang, Z. Chen, Q. Xiao, F. Jiang, J. Li, X. Wan, B. Wang et al., "Cmb: A comprehensive medical benchmark in chinese," arXiv preprint arXiv:2308.08833, 2023.

[129] H. Zeng, "Measuring massive multitask chinese understanding," arXiv preprint arXiv:2304.12986, 2023.

[130] R. Y. Pang, A. Parrish, N. Joshi, N. Nangia, J. Phang, A. Chen, V. Padmakumar, J. Ma, J. Thompson, H. He et al., "Quality: Question answering with long input texts, yes!" arXiv preprint arXiv:2112.08608, 2021.

[131] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, "Think you have solved question answering? try arc, the ai2 reasoning challenge," arXiv preprint arXiv:1803.05457, 2018.

[132] A. Talmor, J. Herzig, N. Lourie, and J. Berant, "Commonsenseqa: A question answering challenge targeting commonsense knowledge," arXiv preprint arXiv:1811.00937, 2018

[133] E. Dinan, S. Roller, K. Shuster, A. Fan, M. Auli, and J. Weston, "Wizard of wikipedia: Knowledge-powered conversational agents," arXiv preprint arXiv:1811.01241, 2018

[134] H. Wang, M. Hu, Y. Deng, R. Wang, F. Mi, W. Wang, Y. Wang, W.C. Kwan, I. King, and K.-F. Wong, "Large language models as source
planner for personalized knowledge-grounded dialogue," arXiv preprint arXiv:2310.08840, 2023

[135] , "Large language models as source planner for personalized knowledge-grounded dialogue," arXiv preprint arXiv:2310.08840, 2023.

[136] X. Xu, Z. Gou, W. Wu, Z.-Y. Niu, H. Wu, H. Wang, and S. Wang, "Long time no see! open-domain conversation with long-term persona memory," arXiv preprint arXiv:2203.05797, 2022.

[137] T.-H. Wen, M. Gasic, N. Mrksic, L. M. Rojas-Barahona, P.-H. Su, S. Ultes, D. Vandyke, and S. Young, "Conditional generation and snapshot learning in neural dialogue systems," arXiv preprint arXiv:1606.03352, 2016.

[138] R. He and J. McAuley, "Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering," in proceedings of the 25th international conference on world wide web, 2016, pp. $507-517$.

[139] S. Li, H. Ji, and J. Han, "Document-level event argument extraction by conditional generation," arXiv preprint arXiv:2104.05919, 2021.

[140] S. Ebner, P. Xia, R. Culkin, K. Rawlins, and B. Van Durme, "Multisentence argument linking," arXiv preprint arXiv:1911.03766, 2019.

[141] H. Elsahar, P. Vougiouklis, A. Remaci, C. Gravier, J. Hare, F. Laforest, and E. Simperl, "T-rex: A large scale alignment of natural language with knowledge base triples," in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), 2018

[142] O. Levy, M. Seo, E. Choi, and L. Zettlemoyer, "Zero-shot relation extraction via reading comprehension," arXiv preprint arXiv:1706.04115, 2017.

[143] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, "Hellaswag: Can a machine really finish your sentence?" arXiv preprint arXiv:1905.07830, 2019.

[144] S. Kim, S. J. Joo, D. Kim, J. Jang, S. Ye, J. Shin, and M. Seo, "The cot collection: Improving zero-shot and few-shot learning of language models via chain-of-thought fine-tuning," arXiv preprint arXiv:2305.14045, 2023.

[145] A. Saha, V. Pahuja, M. Khapra, K. Sankaranarayanan, and S. Chandar, "Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph," in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018 .

[146] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "Measuring massive multitask language understanding," arXiv preprint arXiv:2009.03300, 2020.

[147] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," arXiv preprint arXiv:1609.07843, 2016.

[148] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant, "Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies," Transactions of the Association for Computational Linguistics, vol. 9, pp. 346-361, 2021.

[149] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, "Fever: a large-scale dataset for fact extraction and verification," arXiv preprint arXiv:1803.05355, 2018.

[150] N. Kotonya and F. Toni, "Explainable automated fact-checking for public health claims," arXiv preprint arXiv:2010.09926, 2020.

[151] R. Lebret, D. Grangier, and M. Auli, "Neural text generation from structured data with application to the biography domain," arXiv preprint arXiv:1603.07771, 2016.

[152] H. Hayashi, P. Budania, P. Wang, C. Ackerson, R. Neervannan, and G. Neubig, "Wikiasp: A dataset for multi-domain aspect-based summarization," Transactions of the Association for Computational Linguistics, vol. 9, pp. 211-225, 2021.

[153] S. Narayan, S. B. Cohen, and M. Lapata, "Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization," arXiv preprint arXiv:1808.08745, 2018.

[154] S. Saha, J. A. Junaed, M. Saleki, A. S. Sharma, M. R. Rifat, M. Rahouti, S. I. Ahmed, N. Mohammed, and M. R. Amin, "Vio-lens: A novel dataset of annotated social network posts leading to different forms of communal violence and its evaluation," in Proceedings of the First Workshop on Bangla Language Processing (BLP-2023), 2023, pp. 7284 .

[155] X. Li and D. Roth, "Learning question classifiers," in COLING 2002: The 19th International Conference on Computational Linguistics, 2002.

[156] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts, "Recursive deep models for semantic compositionality over a sentiment treebank," in Proceedings of the 2013 conference on empirical methods in natural language processing, 2013, pp. 16311642
[157] H. Husain, H.-H. Wu, T. Gazit, M. Allamanis, and M. Brockschmidt, "Codesearchnet challenge: Evaluating the state of semantic code search," arXiv preprint arXiv:1909.09436, 2019.

[158] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano et al., "Training verifiers to solve math word problems," arXiv preprint arXiv:2110.14168, 2021.

[159] R. Steinberger, B. Pouliquen, A. Widiger, C. Ignat, T. Erjavec, D. Tufis, and D. Varga, "The jrc-acquis: A multilingual aligned parallel corpus with 20+ languages," arXiv preprint cs/0609058, 2006.

[160] Y. Hoshi, D. Miyashita, Y. Ng, K. Tatsuno, Y. Morioka, O. Torii, and J. Deguchi, "Ralle: A framework for developing and evaluating retrieval-augmented large language models," arXiv preprint arXiv:2308.10633, 2023

[161] J. Liu, "Building production-ready rag applications," https://www.ai. engineer/summit/schedule/building-production-ready-rag-applications 2023.

[162] I. Nguyen, "Evaluating rag part i: How to evaluate document retrieval," https://www.deepset.ai/blog/rag-evaluation-retrieval 2023.

[163] Q. Leng, K. Uhlenhuth, and A. Polyzotis, "Best practices for llm evaluation of rag applications," https://www.databricks.com/blog/ LLM-auto-eval-best-practices-RAG 2023.

[164] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "Ragas: Automated evaluation of retrieval augmented generation," arXiv preprint arXiv:2309.15217, 2023

[165] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, "Ares: An automated evaluation framework for retrieval-augmented generation systems," arXiv preprint arXiv:2311.09476, 2023.

[166] C. Jarvis and J. Allard, "A survey of techniques for maximizing $11 \mathrm{~m}$ performance," https://community.openai. com/t/openai-dev-day-2023-breakout-sessions/505213\# a-survey-of-techniques-for-maximizing-llm-performance-2 2023.

[167] J. Chen, H. Lin, X. Han, and L. Sun, "Benchmarking large language models in retrieval-augmented generation," arXiv preprint arXiv:2309.01431, 2023

[168] Y. Liu, L. Huang, S. Li, S. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun, "Recall: A benchmark for llms robustness against external counterfactual knowledge," arXiv preprint arXiv:2311.08147, 2023.

[169] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen, "Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models," arXiv preprint arXiv:2401.17043, 2024

[170] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, S. Subramanian, E. Bakhturina, M. Shoeybi, and B. Catanzaro, "Retrieval meets long context large language models," arXiv preprint arXiv:2310.03025, 2023.

[171] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez, "Memgpt: Towards llms as operating systems," arXiv preprint arXiv:2310.08560, 2023

[172] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient streaming language models with attention sinks," arXiv preprint arXiv:2309.17453, 2023

[173] T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica, and J. E. Gonzalez, "Raft: Adapting language model to domain specific rag," arXiv preprint arXiv:2403.10131, 2024.

[174] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," arXiv preprint arXiv:2001.08361, 2020.

[175] U. Alon, F. Xu, J. He, S. Sengupta, D. Roth, and G. Neubig, "Neurosymbolic language modeling with automaton-augmented retrieval," in International Conference on Machine Learning. PMLR, 2022, pp. 468-485.

[176] M. Yasunaga, A. Aghajanyan, W. Shi, R. James, J. Leskovec, P. Liang, M. Lewis, L. Zettlemoyer, and W.-t. Yih, "Retrieval-augmented multimodal language modeling," arXiv preprint arXiv:2211.12561, 2022.

[177] J. Li, D. Li, S. Savarese, and S. Hoi, "Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models," arXiv preprint arXiv:2301.12597, 2023.

[178] W. Zhu, A. Yan, Y. Lu, W. Xu, X. E. Wang, M. Eckstein, and W. Y. Wang, "Visualize before you write: Imagination-guided open-ended text generation," arXiv preprint arXiv:2210.03765, 2022

[179] J. Zhao, G. Haffar, and E. Shareghi, "Generating synthetic speech from spokenvocab for speech translation," arXiv preprint arXiv:2210.08174, 2022 .

[180] D. M. Chan, S. Ghosh, A. Rastrow, and B. Hoffmeister, "Using external off-policy speech-to-text mappings in contextual end-to-end automated speech recognition," arXiv preprint arXiv:2301.02736, 2023

[181] A. Yang, A. Nagrani, P. H. Seo, A. Miech, J. Pont-Tuset, I. Laptev, J. Sivic, and C. Schmid, "Vid2seq: Large-scale pretraining of a visual language model for dense video captioning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10714-10 726.

[182] N. Nashid, M. Sintaha, and A. Mesbah, "Retrieval-based prompt selection for code-related few-shot learning," in 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), 2023, pp. $2450-2462$.


[^0]:    Corresponding Author.Email haofen.wang @ tongji.edu.cn

    ${ }^{1}$ Resources are available at https://github.com/Tongji-KGLLM/ RAG-Survey

[^1]:    ${ }^{2}$ https://www.llamaindex.ai

    $\sqrt[3]{\text { https://www.langchain.com/ }}$

[^2]:    ${ }^{4}$ https://hotpotqa.github.io/wiki-readme.html

    5 https://github.com/facebookresearch/DPR

[^3]:    ${ }^{6}$ https://github.com/aurelio-labs/semantic-router

    https://huggingface.co/spaces/mteb/leaderboard

[^4]:    ${ }^{8}$ https://www.trulens.org/trulens_eval/core_concepts_rag_triad/

    9 https://kimi.moonshot.cn

[^5]:    ${ }^{10}$ https://github.com/inverse-scaling/prize

[^6]:    ${ }^{11}$ https://github.com/weaviate/Verba

    ${ }^{12}$ https://aws.amazon.com/cn/kendra/

</end of paper 1>


<paper 2>
# WHAT DOES THE KNOWLEDGE NEURON THESIS HAVE TO DO WITH KNOWLEDGE? 

Jingcheng Niu ${ }^{14} \quad$ Andrew Liu ${ }^{2} \quad$ Zining Zhu ${ }^{134} \quad$ Gerald Penn ${ }^{14}$<br>niu@cs.toronto.edu a254liu@uwaterloo.ca zzhu41@ stevens.edu gpenn@cs.toronto.edu<br>${ }^{1}$ University of Toronto, ${ }^{2}$ University of Waterloo, ${ }^{3}$ Stevens Institute of Technology, ${ }^{4}$ Vector Institute


#### Abstract

We reassess the Knowledge Neuron (KN) Thesis: an interpretation of the mechanism underlying the ability of large language models to recall facts from a training corpus. This nascent thesis proposes that facts are recalled from the training corpus through the MLP weights in a manner resembling key-value memory, implying in effect that "knowledge" is stored in the network. Furthermore, by modifying the MLP modules, one can control the language model's generation of factual information. The plausibility of the $\mathrm{KN}$ thesis has been demonstrated by the success of $\mathrm{KN}$-inspired model editing methods (Dai et al. 2022; Meng et al., 2022).

We find that this thesis is, at best, an oversimplification. Not only have we found that we can edit the expression of certain linguistic phenomena using the same model editing methods but, through a more comprehensive evaluation, we have found that the $\mathrm{KN}$ thesis does not adequately explain the process of factual expression. While it is possible to argue that the MLP weights store complex patterns that are interpretable both syntactically and semantically, these patterns do not constitute "knowledge." To gain a more comprehensive understanding of the knowledge representation process, we must look beyond the MLP weights and explore recent models' complex layer structures and attention mechanisms.


## 1 INTRODUCTION

Recent research has highlighted the remarkable ability of large pretrained language models (PLMs) to recall facts from a training corpus (Petroni et al. 2019). The underlying mechanism by which this information is stored and retrieved within PLMs, however, remains a subject of intensive investigation. The Knowledge Neuron (KN) Thesis has been recently proposed as a novel framework for interpreting language models (LMs) (Dai et al. 2022; Meng et al., 2022; 2023). This thesis suggests that LMs operate akin to key-value memories, recalling facts from the training corpus through the multi-layer perceptron (MLP) weights. Therefore, a significant implication of the KN thesis is that factual information generation by LMs can be controlled by modifying the MLP modules. Should this manipulation of factual information recall become feasible, it could lead to the development of language models that are more controllable, interpretable, and factually aligned.

The plausibility of the $\mathrm{KN}$ thesis is demonstrated by the success of $\mathrm{KN}$-inspired model-editing methods. Dai et al. (2022) argued that relational facts can be localised to a handful of 2-5 MLP neurons. They then developed a method to identify these neurons using a search algorithm based on an integral of gradients. By manipulating the activation of these identified neurons (KN edit), they managed to alter the model's response to fill-in-the-blank cloze tasks and generate counterfactual information without additional fine-tuning. In a parallel approach, Meng et al. (2022) proposed a more intricate model wherein factual recall occurs in two critical locations, each incorporating a different module. In this model, the mid-layer MLP retrieves the fact, and an attention module copies it into the output response at the topmost layer. Despite this proposed two-step process, their proposed model editing method, Rank-One Model Editing (ROME), only modifies MLP weights, much as KN edit only modifies MLP activations without editing attention modules.

While the efficacy of these model editing methods has been showcased in simple fill-in-the-blank cloze tasks, the appraisal of such achievements mainly rests on basic paraphrasing of the prompts, as

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-02.jpg?height=292&width=1179&top_left_y=228&top_left_x=468)

Figure 1: Syntactic phenomena can be located and edited using existing model editing methods. The integrated gradient of singular determiner (this, that) and plural determiner (these, those) form two distinct groups. Erasing these neurons leads to output probability changes.

outlined by Yao et al. (2023), who introduced an additional assessment metric, portability, finding that model-editing methods to date lack robustness. Their performance is halved when evaluated with the portability measure. Building on this, we introduce two new metrics. First, a successful edit must demonstrate symmetry within bijective relationships (e.g., with the assertion Ottawa is the capital of Canada, the reciprocal Canada's capital is Ottawa should also hold valid). Second, a successful edit must extend to synonym usage (e.g., a dentist treats a toothache and a dentist treats tooth pain should be considered equivalent). Our evaluation shows that existing model-editing methods are even less robust under these two new criteria.

It is practically impossible to exhaustively assess factual model-editing methods due to the difficulty in systematically dealing with counterfactual data. The potential counterfactual replacements for Canada's capital are seemingly endless. Thus, beyond the introduction of the two new evaluation criteria above, we propose the evaluation of model-editing methods using syntactic constructions. We have determined that the $\mathrm{KN}$ thesis applies just as reliably to syntactic phenomena (as illustrated in Figure 11. Unlike many facts, syntactic phenomena can provide rigorously defined targets for editing through the use of so-called minimal pairs. As a result, in this paper, we re-evaluate the $\mathrm{KN}$ thesis by expanding the scope of our assessment to include more complex factual patterns and syntactic phenomena. This also speaks to a long-standing debate regarding the formal $v s$. functional competence of language models (Mahowald et al. 2024) - an LM's ability to follow linguistic rules and patterns vs. its ability to apply language in the real-world (see \$2.3). If we edit a model's expression of facts and linguistic phenomena using the same approach, this could indicate that both the formal and functional competencies of an LM are governed by the same underlying mechanisms.

Within the context of Dai et al.'s (2022) KN framework, KN edit's efficacy is unsatisfactory. Editing the $\mathrm{KN}$ activations has only limited impact on categorical predictions. The effect of $\mathrm{KN}$ edit is only apparent in the shifts in the output probability distributions of tokens. The patterns that the method localises also appeal to superficial cues such as word co-occurrence frequencies. We also find several critical shortcomings in the ROME framework. LMs process both linguistic and factual information in phases, but the exact task distribution between the MLP and attention modules appears to be more idiosyncratic than initially theorized (Meng et al. 2022). ROME model editing only superficially alters token association patterns, in a manner that is inconsistent across the various expressions that may attend the same underlying knowledge. As a result, whatever is being manipulated reflects none of the traditional tautologies that have been associated with "knowledge," as that term has been understood in philosophy since the time of Aristotle. When implemented on syntactic constructions, furthermore, the influence of ROME's editing is limited only to the word altered and no pivot that preserves any reasonable standard of syntactic paraphrase, such as substitutability salva veritate, is forthcoming. Furthermore, ROME fails under our newly proposed symmetry and synonymy criteria.

We therefore argue for the position that the feed-forward MLP modules of the transformer model do not store knowledge, but rather complex "token expression patterns." These token expression patterns can often be interpreted linguistically, but the information that they express does not fit into linguistically or factually defined categories. A key-value, memory-based view of the language model is overly simplistic in explaining the remarkable ability of recent PLM's formal, and perhaps even functional, competence. We need to investigate the rich layer and attentive structure of recent PLMs more to arrive at a better understanding of their underlying mechanics.

In the following sections, we will first provide an overview of the $\mathrm{KN}$ thesis ( $\$ 22$. Then we will evaluate two practices inspired by it: Dai et al.'s (2022) KN edit framework ( $\$ 3$ and Meng et al.'s (2022) ROME framework ( $\$ 4$. Finally, we will conclude the paper with a discussion ( $\$ 5$.[^0]

## 2 The KNOWLEdGe NeUron ThESIS

Geva et al. (2021) were among the first to propose that the MLP modules in a transformer model behave like key-value memories. A typical MLP module in recent transformer-based PLMs has two layers. They argue that the first layer corresponds to keys, and the second layer, to values ${ }^{2}$ They found that each key neuron is triggered by human-interpretable shallow input patterns such as periods of time that end with the letter " $a$." Then, the corresponding value neurons distorted the next-token output probability, until a final distribution is generated.

The KN thesis emerged as a result of this important discovery. Dai et al. (2022) coined the term knowledge neuron and ambitiously claimed that the keys and values within MLP modules not only capture simple patterns but also store "knowledge." They formulate an item of fact, such as Canada's capital is Ottawa, as a 3-tuple $(s, t, r)$, consisting of the source ( $s$, Canada), the target ( $t$, Ottawa) and the relation ( $r$, capital) between them. The authors asserted that this tuple can be localized to a small group of MLP neurons typically found in the topmost layers of the language model, which they identified by analysing the magnitude of the integrals of gradients among prompts. To support their claim, they conducted model-editing experiments. By suppressing the KNs (setting their activations to zero), they observed a decrease in the probability of generating the correct original target $(t)$, while other tokens remained largely unaffected, demonstrating a "minimally invasive surgery." Meng et al. (2022) proposed a refinement of Dai et al.'s (2022) model. They employed a causal mediation method (Finlayson et al. 2021) to form a more intricate version of the $\mathrm{KN}$ thesis. They argue that the factual association process happens at two locations: a mid-layer MLP recalls the fact from memory, and the topmost layer's attention model copies that information to the final output.

There were similar investigations of neurons prior to the KN thesis. Durrani et al. (2020) observed the neurons of an auxiliary probing model that was trained on BERT embeddings, not the neurons of BERT itself. Therefore, their analysis faced an all-too-common dilemma for probing: did they find insights about the language models or artefacts of the fine-tuning process (Hewitt \& Liang, 2019)? Finlayson et al. (2021) used causal mediation analysis to study subject-verb agreement in GPT and XLNet (Yang et al. 2019). In particular, they observed a difference in ratios between the verb with the correct inflection and one with the incorrect inflection. They then modify the prompt, see the probability change and reason about the internal mechanisms of the model for expressing subjectverb agreement. They concluded that the upper-middle layers are more relevant to the expression and that there are various levels of overlap between the top $5 \%$ neurons used to express agreement. These insights, however, just as with previous probing work, are still purely observational and largely preoccupied with layers and network depth. They are able to observe many characteristics of the process, but still cannot cannot provide a satisfactory understanding of how it happens.

More recently, there has been interest in utilizing large language models (LLMs) to gain insight into the differing functionalities of individual neurons. Despite its title's strident claim that neurons in LMs can be "explained," Bills et al. (2023) clarify that their model "explains correlations, not mechanisms." From a knowledge-representation standpoint, their evaluation of LLM explanations is also entirely observational. When Huang et al. (2023) reassessed the validity of these explanations, even the most confident ones had high error rates and little to no causal effects on the interventions that use the explanations. The LLM interpretation of LMs is still immature.

### 2.1 EVALUATING THE KN THESIS: AN OVERVIEW

The effectiveness of a model-editing algorithm is customarily evaluated across three dimensions (Yao et al., 2023): (1) reliability: whether the model can successfully change its output from $t$ to $t^{*}$ (also referred to as an efficacy score by Meng et al. (2022)); (2) generality: whether the effect is applicable to rephrased relations; and, (3) locality: whether the edit impacts unrelated relations. Yao et al. (2023) stress, however, that the assessment of generality is often constrained to simple paraphrasing. This is typically done by developing multiple templates for a specific relation. For instance, the relation capital can be structured as both "The capital of [s] is [t]." and "[s]'s capital is [t]." Previous evaluations (Elazar et al., 2021; Meng et al., 2022, 2023) prematurely announced success when a model, edited on a first template, could be generalized to a second template. Thus, Yao et al. (2023) recommended extending the assessment parameters by introducing the concept of[^1]portability. For example, having changed Watts Humphrey's alma mater from Trinity College to Harvard University, the model should return Boston instead of Dublin when asked about the city where Watts Humphrey received his university education. It was apparent that model-editing methods present a markedly lower level of portability than generality ( $50 \%$ versus $90 \%$ ). The evaluation of portability, on the other hand, requires new data annotation, which can be costly.

Extending Yao et al. (2023), we attempt a more comprehensive evaluation of model editing of factual association with two extra criteria: bijective symmetry and synonymous invariance. Bijective symmetry does not require new data collection and we can obtain data automatically from previous corpora. For a bijection relation such as capital or capital of, we should see the model generalise $\left(s, t \rightarrow t^{*}, r\right)$ to $\left(t^{*}, s, r^{-1}\right)$. For example, if we change the capital of Canada to Rome, then the model should also agree that Rome is the capital of Canada. Similarly, an effective edit should also be able to generalise across synonyms. If the model knows that a dentist treats toothaches, it should also know that they also treat tooth pain. Prior work (Elazar et al. 2021) only used synonym replacement on rephrasing the relation prompts - we extend it to the source and the target.

Several others have already questioned the validity of the KN thesis. Hase et al. (2023) identified discrepancies between the results of causal tracing and the effects of ROME editing. They concluded that a mechanistic understanding reveals insights on the consequences of model editing. To the best of our knowledge, we are the first to comprehensively evaluate the $\mathrm{KN}$ thesis using rigorously defined syntactic phenomena. We consider three: determiner-noun agreement, subject-verb agreement, and gender and number agreement across anaphoric chains.

### 2.2 EvaluATING THE KN THESIS ON SYntactiC PHENOMENA

Edit pairs for syntactic phenomena, by contrast, can be systematically extracted through the formation of "minimal pairs." For a grammatical sentence that expresses a linguistic phenomenon, we can construct an ungrammatical sentence that minimally differs from the original sentence in respect of one feature of grammatical acceptability. For example, the phrase this student can be changed to the ungrammatical counterpart, *this students. The BLiMP corpus (Warstadt et al. 2020) is one of the most comprehensive and extensively utilised collections of such minimal pairs.

We therefore propose to systematically evaluate the effect of model-editing methods using syntactically differentiated prompts. We define a similar 3-tuple $(s, t, p)$ that contains the source $(s)$, the target $(t)$ and the syntactic phenomenon $(p)$. Take the phenomenon determiner-noun agreement as an example. In a grammatical sample sentence from a minimal pair, $s$ is the tokens that are condition the expression of the target (the determiner), and $t$ is the tokens that differ within the pair (the noun). The ungrammatical target $t^{*}$, is the noun in the opposite form. We then intervene with model editing, and observe whether the model assigns a higher probability to $t$ than $t^{*}$.

### 2.3 EdITING SYntACTIC PHENOMENA \& THE “FORMAL VS FUNCTIONAL" DISTINCTION

If we can successfully edit facts as well as syntactic phenomena using the same model-editing methods to the same degree, then it stands to reason that the model follows a unified underlying mechanism for both factual and syntactic information. Choosing the correct city (the Space Needle is in Seattle/*Rome) would be no different than choosing the correct verb form (the apple is/*are red).

Mahowald et al. (2024) refers to a distinction between the formal and functional competence of a language model: formal means "knowledge of linguistic rules and patterns," and functional refers to "understanding and using language in the world." Syntactic phenomena pertain to formal competence, and facts pertain to functional competence, respectively. NLP researchers sometimes informally use the terms syntax and semantics to refer to this distinction. BLiMP even refers to anaphoric gender agreement as morphological. Jawahar et al. (2019) and Tenney et al. (2019) believe that syntactic information is located in lower layers in BERT than semantic information, because syntactic information is more "shallow." Dai et al. (2022) appear to agree with this assertion in claiming that factual information is located in the upper layers. Meng et al. (2022), however, claim that factual information is located in the middle. This contradiction may support Niu et al.'s (2022) assertion that layers are not the best explanatory device of the distribution of these types of information in LMs. We explore here the possibility that no dividing line exists at all between the mechanisms through which a language model processes information related to these two types of competence.

## 3 Localising Syntactic PHENOMENA In LANGUAGE ModeLS

We put the $\mathrm{KN}$ thesis to the test under the $\mathrm{KN}$-edit framework by asking three questions: (1) can we localise linguistic phenomena using the same KN-edit method; (2) how do the levels of localisation compare to each other; and (3) are these localisations strong enough to support the $\mathrm{KN}$ thesis? 3

### 3.1 METHODS: SEARCHING FOR KNS OF SYntactic PHENOMENA

For each prompt, we calculate an integral-of-gradient attribution score $\alpha_{i}^{(l)}$ for the $i$-th intermediate neuron on the $l$-th layer $\left(w_{i}^{(l)}\right)$. Then, for a syntactic phenomenon with the source-target pair $(s, t, p)$, we find the neurons that have an attribution score greater or equal to $\pi=20 \%$ of the maximum attribution score shared among at least $\tau \%$ of its prompts. We start from $\tau=70 \%$ and adjust it by an increment or decrement of $5 \%$ until the number of neurons is within the range of $[2,5]$.

Neuron Attribution Score Given an input prompt $x$, we follow Dai et al. (2022) and use the integral of gradients to calculate the neuron attribution score:

$$
\begin{equation*}
\alpha_{i}^{(l)}=\bar{w}_{i}^{(l)} \int_{\gamma=0}^{1} \frac{\partial P_{x}\left(\gamma \bar{w}_{i}^{(l)}\right)}{\partial w_{i}^{(l)}} d \gamma, P_{x}\left(\hat{w}_{i}^{(l)}\right)=p\left(y \mid x, w_{i}^{(l)}=\hat{w}_{i}^{(l)}\right) \tag{1}
\end{equation*}
$$

where $P_{x}\left(\hat{w}_{i}^{(l)}\right)$ denotes the probability distribution of the token $y$ when changing the neuron $w_{i}^{(l)}$, $\mathrm{s}$ value to $\hat{w}_{i}^{(l)}$, and $\frac{\partial P_{x}\left(\alpha \bar{w}_{i}^{(l)}\right)}{\partial w_{i}^{(l)}}$ denotes the gradient of the model with respect to the activation $w_{i}^{(l)}$. We will see a more salient gradient when the neuron inflicts a greater change on the output probability.

Measuring the Level of Localisation We use three metrics to measure the level of localisation: (1) the number of identified neurons $(|\mathrm{KN}|$ ) using the initial threshold setting ( $\tau=70 \%)$, (2) the final threshold $\tau$ to obtain 2-5 KNs, and, (3) a similarity score among all the token attribution patterns.

Both of Dai et al.'s (2022) measures $(|\mathrm{KN}|$ and $\tau$ ) depend on adjusting the two threshold hyperparameters, $\pi$ and $\tau$. Here, we propose a non-parametric measure using a generalised $n$-sample similarity measure $\left(R_{1}^{2}\right)$ that measures the correlation of all the attribution patterns:

$$
\begin{equation*}
Y=\left[y_{1} \ldots y_{n}\right], y_{i}=\frac{s_{i}}{\left\|s_{i}\right\|}, Y=U S V^{\top}=\sum_{k=1}^{n} \sigma_{k} u_{k} v_{k}^{\top}, R^{2}=\frac{\sigma_{1}^{2}-1}{n-1} \tag{2}
\end{equation*}
$$

We first normalise and concatenate each attribution pattern $s_{i}$ for each prompt $x_{i}$ in the dataset into $Y$. Then, we can calculate the similarity/correlation among all $n$ patterns by conducting a singular value decomposition (SVD) and using the square of the first singular value $\sigma_{1}^{2}$. We then normalise this measure to the range $[0,1]$ so that the similarity between $n$ parallel vectors will be $R_{1}^{2}=1$, and $n$ orthogonal vectors will get $R_{1}^{2}=0$.

### 3.2 RESULTS \& FINDINGS

Finding 1: We can localise the grammatical number of determiners to just two neurons, just like factual information. The BLiMP paradigm determiner_noun_agreement_2 (DNA.2) contains 1000 sentence pairs with exactly one demonstrative determiner (this, that, these, those) agreeing with an adjacent noun, e.g., Carl cures those/*that horses. The determiner those is $t$, that is $t^{*}$ and[^2]

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-06.jpg?height=336&width=702&top_left_y=282&top_left_x=362)

(a) Effect of suppressing the singular neuron $w_{2096}^{(10)}$

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-06.jpg?height=288&width=699&top_left_y=325&top_left_x=366)

(b) Effect of suppressing the plural neuron $w_{1094}^{(9)}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-06.jpg?height=296&width=697&top_left_y=321&top_left_x=1061)

Figure 3: Suppressing the number neuron's (singular: $w_{2096}^{(10)}$; plural: $\left.w_{1094}^{(9)}\right)$ effect across numberexpressing prenominal modifiers. Significant $(p<0.05)$ changes are highlighted in red. The three sections in the plots are, from left to right, plural, singular and neutral modifiers.

the noun horses is $s$. A noun may appear in multiple sentence pairs. Among the paradigm's 1000 sentence pairs, we identified 283 unique Det-N pairs $\left(s, t, t^{*}, r\right)$.

Attribution Score Patterns The attribution score of neurons shows a highly consistent pattern that can be interpreted linguistically. We calculated the average attribution scores of all the prompts that contains each one of the determiners. Figure $2 \mathrm{a}$ shows a selection of the average attribution scores. The colour block in the $i$ th column and $j$ th row shows the attribution score $\alpha_{i}^{(j)}$. As we can see, a common neuron $\left(w_{2096}^{(10)}\right)$ has a high average attribution score for both of the singular determiners this and that, and another common neuron $\left(w_{1094}^{(9)}\right)$ lights up for the plural determiners these and those 4

This pattern is not only shown in aggregate. For each Det-N pair, we use the 1000 sentences in the paradigm as templates to create the prompts needed for a $\mathrm{KN}$ search. For each sentence, we replace the sentence's determiner and noun with the Det-N's determiner and noun. We then obtain 1000 sentences with different contexts but the same determiners and nouns. Then, we run a KN search on these 1000 sentences. When we look into each individual Det-N pair, the two neurons are identified as $\mathrm{KNs}$ in the vast majority of the pairs. As shown in Figure $2 \mathrm{~b}, w_{2096}^{(10)}$ appeared in $93 \%$ of the pairs with this and $75 \%$ of the pairs with that. The plural neuron appeared in $100 \%$ of pairs with these or those. More importantly, these neurons were not identified as KNs in pairs with the opposite grammatical numbers. Figure 2b shows an excerpt of the results (full results in Appendix B.2).

Effects of Suppressing the "Number Neuron" Do these two neurons correspond to grammatical number? We suppress each neuron (setting activation to 0 ) and compute the pre- and post-edit model's output probability of various number-expressing prenominal modifiers across all prompts with singular/plural nouns. Appendix B.1 explains the prenominal modifier selection process. Figure 3 shows the average effect of suppressing the identified $\operatorname{KNs}\left(\frac{p \text { (post-edit })-p \text { (pre-edit })}{\min (p(\text { post-edit) }) p \text { (pre-edit }))}\right)$.

The result of suppressing the plural neuron is pronounced (Figure 3 b). This intervention leads to a significant reduction in probability across all plural modifiers, a notable increase for the majority of singular modifiers, but a limited impact for modifiers that do not express number agreement. Therefore, erasing the activation of the plural neuron causes a decrease in the expression of determiner-noun agreement for plural modifiers. Although this $\mathrm{KN}$ search is solely based on these four demonstrative determiners, we observed that it generalizes to other determiners (one, $a$, an, every; two, both; multiple, several, various) and even adjectives (single, unique, sole). This effect is statistically significant. By treating the preand post-edit probabilities as two separate groups, a Student's (1908) $t$-test reveals significance when the modifiers are highlighted in red in

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-06.jpg?height=266&width=338&top_left_y=1775&top_left_x=1403)

Figure 4: The localisation of plurality appeals to word co-occurrence frequencies cues. Figure 3 The null hypothesis is that the pre- and post-edit probabilities are sampled from the same distribution, i.e., the intervention has no effect. Thus, the neuron $w_{1094}^{(9)}$ can be interpreted through the lens of a linguistic phenomenon, viz. determiner-noun agreement.

Note, however, that the word scattered also sees a significant probability decrease when suppressing the plural neuron. Scattered does not specify for plural number; phrases such as "scattered rioting"[^3]

| BLiMP Paradigm | $\|\mathrm{KN}\|$ | $\tau$ | $R_{1}^{2}$ |  | Rels. | $\|\mathrm{KN}\|$ | $\tau$ | $R_{1}^{2}$ |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| det_n_agr._1 | 3.94 | 0.71 | 0.56 |  | P101 | 0.167 | 0.515 | 0.399 |
| det_n_agr._2 | 1.86 | 0.62 | 0.56 |  | P103 | 0.204 | 0.662 | 0.399 |
| dna._irr._1 | 5.53 | 0.73 | 0.64 |  | P106 | 1.292 | 0.607 | 0.365 |
| dna._irr._2 | 2.45 | 0.67 | 0.55 |  | P108 | 1.493 | 0.663 | 0.473 |
| dna._w._adj_1 | 8.88 | 0.78 | 0.67 |  | P1303 | 10.462 | 0.814 | 0.684 |
| dna._w._adj_2 | 2.26 | 0.67 | 0.57 |  | P140 | 2.008 | 0.689 | 0.263 |

(a) Levels of localisation measures.
![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-07.jpg?height=230&width=618&top_left_y=282&top_left_x=1119)

(b) Layer distribution of identified KNs. Both

BLiMP and PARAREL occupy the topmost layers.

Figure 5: The localisation of certain syntactic phenomena (BLiMP) is comparable to facts (PARAREL). We see comparable localisation metrics and the identified KNs occupy the same layers.

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-07.jpg?height=200&width=420&top_left_y=713&top_left_x=365)

(a) The exact effect to output probability of editing the KNs. $\square$ : pre-edit. ■: post-edit.

| Paradigm | Pre-edit | Post-edit | $\Delta$ |
| :--- | :---: | :---: | :---: |
| det_n_agr._2 | $100 \%$ | $94.8 \%$ | $-5.2 \%$ |
| dna._irr._2 | $99.5 \%$ | $96.9 \%$ | $-2.6 \%$ |
| dna._w._adj._2 | $97.1 \%$ | $94.4 \%$ | $-2.7 \%$ |
| dna._w._adj._irr._2 | $97.4 \%$ | $95.4 \%$ | $-2.0 \%$ |

(b) These modifications of determinernoun $\mathrm{KNs}$ are usually not enough to overturn the categorical prediction.

| Data | Model | Reliability |
| :--- | :---: | :---: |
| ZsRE | T5-XL | 22.51 |
|  | GPT-J | 11.34 |
| CounterFact | T5-XL | 47.86 |
|  | GPT-J | 1.66 |

(c) KN edit has low reliability for facts (Yao et al. 2023).

Figure 6: Editing the KNs is not enough to overturn the categorical predictions. The major limitation of $\mathrm{KN}$ edit is its low reliability. These reliability scores cannot support the $\mathrm{KN}$ thesis.

are syntactically and semantically well-formed. But it is used more often with plural nouns because of its meaning. This frequency effect is not limited to scattered. Other words such as any, all, unified, and the three adjectives unique, single and sole exhibit a similar bias. As shown in Figure 4 . we see probability changes, although less substantial, alongside those modifiers that strictly specify for grammatical number. This is a semantic number co-occurrence bias.

The suppression effect of the singular neuron is similar but less pronounced. Overall, we see the opposite effect across all prenominal modifiers, with the "singular" adjectives (unique, single, sole) being the only exceptions. This is, however, unsurprising. Unlike the plural neuron, the singular neuron did not appear in all of the Det-N pairs. We suspect that an LM can identify the plural property more easily when its wordpiece-based tokeniser exposes many plural suffixes.

Finding 2: KNs obtained using linguistic tasks and factual tasks share similar characteristics of localisation. Figure 5a shows the level of localisation of various BLiMP determiner-noun agreement paradigms and selected PARAREL relations. The localisation metrics of both BLiMP paradigms and PARAREL relations fall within the same range. See Appendix C.3 for the full list.

Furthermore, Figure $5 b$ shows no bifurcation of layers within which linguistic and factual KNs locate (see Appendix C.2). All of the neurons are distributed in the topmost layers. The determiner-noun agreement pattern is purely syntactic. This is a refutation of Jawahar et al. (2019) and Tenney et al.'s (2019) view that syntax is localised to more shallow layers than semantics. Our results confirm Niu et al.'s (2022) assertion that the location of syntactic and semantic (and, additionally, factual) information is not distinguished by layer in the LM. In fact, our results may suggest that these types of information are most fruitfully thought of as being handled by the same functional mechanism.

Finding 3: Despite the high level of localisation in the underlying probability drift, the effect of editing the KNs is not enough to overturn the categorical predictions made by the language model. Although we see a high level of localisation in the relative probability change between $t$ and $t,{ }^{*}$ we find that this change is often not enough to overturn the final prediction. As shown in Figure 6, we only see at most $5.2 \%$ of the BLiMP results being overturned. This low reliability issue is not limited to syntactic phenomena. In Figure 6c, we list Yao et al. 's (2023) evaluation of KN edit on two other corpora: ZsRE (Levy et al. 2017) and CounterFact (Meng et al. 2022). The reliability of the $\mathrm{KN}$ algorithm ranges from $1.66 \%$ to $47.86 \%$ - not enough to support the $\mathrm{KN}$ thesis.

Discussion Just as with facts, syntactic phenomena localise to neurons. Modifying merely two neurons working in tandem can significantly change the expression of determiner-noun number.

This is not the only type of localisable syntactic phenomenon (see Appendix $\mathrm{C}$ ), and together they constitute a significant extension of Finlayson et al.'s (2021) findings - syntactic phenomena can be localised to the individual neuron level. Furthermore, these phenomena share with factual information the extent of their localisation, and the layers in which the KNs typically occur.

But do the patterns identified for these neurons constitute "knowledge?" KN edit's low reliability score and its appeal to shallow cues both suggest otherwise. If we follow the $\mathrm{KN}$ thesis and interpret a post-edit probability change as an indication of the quantity of knowledge stored, then we cannot draw the conclusion that knowledge is stored there. The identified neurons are spots with a high information concentration, but the final decision still lies with the rest of the model.

Interestingly, the patterns that we identified resemble linguistic categories, but they deviate from rules of grammatical well-formedness. In determiner-noun agreement, KN edit also affects premodifiers that do not specify for number, alongside plural-specifying determiners such as multiple, several and various. Phrases such as sole breadwinners and scattered rioting are less frequent but by no means unheard of. This suggests that the patterns reflected within the MLP neurons can only be completely accounted for by appealing to superficial cues such as word co-occurrence frequency.

## 4 Causal Tracing and Rank-One Model Editing

In this section, we reassess Meng et al.'s (2022) similar but more intricate implementation of KN edit. They proposed that information is expressed at two locations: facts are recalled in mid-layer MLP weights, and copied to the final output by attention modules. They derived this thesis based on causal mediation. The causal traces in Figure $7 \mathrm{a}$ are computed as follows. First, the source tokens are corrupted by adding random noise $\epsilon$ and the model generates an incorrect result. Then, they restore an intermediate hidden state to its correct value for all the tokens at all layers, and determine whether this restoration can fix the corruption. They discover a division of labour between the

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-08.jpg?height=220&width=894&top_left_y=1039&top_left_x=862)

(a) Factual information.

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-08.jpg?height=228&width=884&top_left_y=1293&top_left_x=859)

(b) Determiner-noun agreement.

![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-08.jpg?height=233&width=897&top_left_y=1561&top_left_x=858)

(c) Subject-verb agreement.

Figure 7: Causal tracing result.

MLP and attention. This division, however, is not stable. In Figure $7 \mathrm{pc}$ we reproduce this effect on syntactic phenomena. The distinction between the early and late site is no longer discernible. This is, in fact, not a distinction between facts and syntactic patterns. Many factual causal traces also do not show this distinction 5

Previous evaluation of the ROME model-editing method was limited to simple paraphrasing (Yao et al. 2023). We observe that ROME does not generalise well in respect of either of our new criteria, bijective symmetry or synonymous invariance (Figure 8ab). This issue persists when we evaluate ROME quantitatively. We assembled two new datasets using PARAREL relations to evaluate our two new criteria (see Appendix E for details). We use the two bijective relations R1376 and P36 to construct a bijective symmetry evaluation dataset. Then, for synonymous invariance, we rewrite the field-of-work targets in P101 into occupation names. For instance, if we change Anaxagoras's field of work from philosophy to linguistics, we also want the model to answer "Anaxagoras is a linguist" when given the prompt. Table 1 shows the result of our evaluation on these newly assembled datasets. Although ROME obtains higher reliability scores than KN edit in both GPT-2 XL and LLaMA-2 7B, the symmetry and synonymy results are both much lower. We also observe[^4]

| (a) GPT-2 XL: The capital of Canada is Ot- <br> tawa <br> ROME Edit: Ottawa $\rightarrow$ Rome | (b) GPT-2 XL: To treat my toothache, I should <br> see a dentist <br> ROME Edit: dentist $\rightarrow$ lawyer | (c) GPT- 2 XL: The authors near the taxi drivers are <br> ROME Edit: are $\rightarrow$ is |
| :---: | :---: | :---: |
|  |  | (:): The authors near the taxi drivers are ... <br> 연: The authors near the taxi drivers is ... |
| (-: The capital of Canada is Ottawa ... <br> ( The capital of $\underline{\text { Canada }}$ is Rome. | (-: To treat my toothache, I should see a dentist, <br> ․ <br> o: Treat my toothache, I should see a lawyer. |  |
|  |  | (): The authors near the dancers in their paper are .. |
| (-): Ottawa is the capital of Canada. <br> Ottawa is the capital of Canada's federalist <br> system of government. |  | s near the dancers is ... |
|  | ![](https://cdn.mathpix.com/cropped/2024_06_04_dbb26d4100c5b153622cg-09.jpg?height=64&width=421&top_left_y=485&top_left_x=828) | (:) The pilots near the taxi drivers were ... <br> ㅇنㅂ: The pilots near the taxi drivers' cabins are ... |
| (-: Rome is the capital of $\mathrm{Italy}, \ldots$ <br> Rome is the capital of $\mathrm{Italy}, \ldots$ | (-: To treat my odontalgia, I should see a dentist. <br> (To treat my odontalgia, I should see a dentist. | (): The pilots near the dancers are.. <br> ₪: |

Figure 8: Comparison of generated text. The prompts are italicized, source tokens $(s)$ are underlined, ungrammatical or counter-factual responses are highlighted in red, and unchanged correct responses in green. () shows the original GPT-2 XL's generation, and : : shows the edited model's response.

that ROME edit can only edit the exact association between the tokens in $(s, t, r)$. As demonstrated in Figure 88, editing the verb corresponding to the authors from are to is only affects the subject the authors, and not other subjects such as the pilots. These look more like at-times brittle patterns of token expression than factual knowledge.

## 5 DISCUSSION \& CONCLUSION

We find that several syntactic agreement phenomena can be localised to a small number of MLP neurons. This localisation has similar characteristics to the localisation of factual information, suggesting that recent transformer-based language models' impressive abilities with respect to various linguistic phenomena and the recall of facts from their training corpora may follow the same underlying mechanism.

The localisation of the two types of information also faces the same challenges, however, which militate against the soundness of the KN thesis. Specifically, the effect of editing the identified neurons is not strong enough to overturn the final prediction, and the scope of the phenomena appears to be limited to shallow cues such as token co-occurrence statistics.

Returning to Geva et al.'s (2021) original findings, the MLP neurons store patterns that are interpretable through a linguistic lens, but they do not store knowledge, either linguistic or factual. Meng et al.'s (2022) causal tracing results, although still an oversimplification, suggest that there are different phases in different layers in the entire process of token expression. But their ROME model-editing method did not avail itself of this important finding. The method is still MLP-based. To achieve a better understanding of this expression process and achieve real model editing, we must examine the entire decision-making circuit (Wang et al., 2022; Wu et al., 2023; Conmy et al., 2023, Murty et al. 2023). Manipulating only the MLP weights is not enough. The circuit mode of interpretation is still at a very early state of development, however. Current circuit identification methods are $a d$ hoc, furthermore, and have only been applied to a small set of tasks. In future work, we will try to formalize the circuit interpretation framework and apply it to more tasks and phenomena.

Our reassessment of causal traces agrees with Hase et al.'s (2023), but we take exception to their claim that "better mechanistic understanding ... may not always translate to insights about how to best change their behavior." It is well-established that we can interpret the computational mechanism of LMs through the lens of formal linguistics (Clark et al. 2019). Both of our findings reveal limitations in current LM interpretation work and suggest that an even more comprehensive, but still mechanistic interpretation of transformers will lead to insights for better control of model behaviour when not limited to the MLP modules, and when patterns of token expression are dealt with unencumbered by misbegotten metaphors about knowledge and human reasoning.

Contributions Our work provides a thorough examination of the $\mathrm{KN}$ thesis and finds that the thesis is, at best, an oversimplification. We (1) extend KN-based analysis to well-defined syntactic tasks, (2) propose two new criteria for evaluating the effectiveness of model editing, and (3) introduce a generalised $n$-sample similarity measure of the level of localisation.

## ACKNOWLEDGMENTS

We thank Lei Yu (University of Toronto) for a great deal of insightful discussion. We also want to thank the anonymous reviewers for providing informative comments and suggestions.

## REFERENCES

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models, 2023.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. What Does BERT Look at? An Analysis of BERT's Attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 276-286, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/W19-4828.

Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In ThirtySeventh Conference on Neural Information Processing Systems, 2023.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge Neurons in Pretrained Transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493-8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581.

Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing Factual Knowledge in Language Models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6491-6506, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.522.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423.

Nadir Durrani, Hassan Sajjad, Fahim Dalvi, and Yonatan Belinkov. Analyzing Individual Neurons in Pre-trained Language Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4865-4880, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main. 395.

Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. Measuring and Improving Consistency in Pretrained Language Models. Transactions of the Association for Computational Linguistics, 9:1012-1031, 2021. doi: 10.1162/tacl_a_00410.

Matthew Finlayson, Aaron Mueller, Sebastian Gehrmann, Stuart Shieber, Tal Linzen, and Yonatan Belinkov. Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1828-1843, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long. 144 .

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers Are Key-Value Memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484-5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main. 446 .

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models, January 2023.

John Hewitt and Percy Liang. Designing and Interpreting Probes with Control Tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733-2743, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: $10.18653 / \mathrm{v} 1 / \mathrm{D} 19-1275$.

Jing Huang, Atticus Geiger, Karel D'Oosterlinck, Zhengxuan Wu, and Christopher Potts. Rigorously Assessing Natural Language Explanations of Neurons. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi (eds.), Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 317-331, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.blackboxnlp-1.24.

Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. What Does BERT Learn about the Structure of Language? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3651-3657, Florence, Italy, July 2019. Association for Computational Linguistics. doi: $10.18653 / \mathrm{v} 1 / \mathrm{P} 19-1356$.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-Shot Relation Extraction via Reading Comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pp. 333-342, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/K17-1034.

Kyle Mahowald, Anna A. Ivanova, Idan A. Blank, Nancy Kanwisher, Joshua B. Tenenbaum, and Evelina Fedorenko. Dissociating language and thought in large language models. Trends in Cognitive Sciences, March 2024. ISSN 1364-6613. doi: 10.1016/j.tics.2024.01.011.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 36, 2022.

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. MassEditing Memory in a Transformer. In The Eleventh International Conference on Learning Representations, May 2023.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. In International Conference on Learning Representations, 2022.

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D. Manning. Characterizing intrinsic compositionality in transformers with Tree Projections. In The Eleventh International Conference on Learning Representations, 2023.

Jingcheng Niu, Wenjie Lu, and Gerald Penn. Does BERT Rediscover a Classical NLP Pipeline? In Proceedings of the 29th International Conference on Computational Linguistics, pp. 31433153, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. Language Models as Knowledge Bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2463-2473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1250.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. OpenAI Blog, pp. 24, 2019.

Student. The Probable Error of a Mean. Biometrika, 6(1):1-25, 1908. ISSN 0006-3444. doi: $10.2307 / 2331554$.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT Rediscovers the Classical NLP Pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4593-4601, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/ $\mathrm{v} 1 / \mathrm{P} 19-1452$.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and Efficient Foundation Language Models, February 2023.

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. In The Eleventh International Conference on Learning Representations, September 2022.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. Transactions of the Association for Computational Linguistics, 8:377-392, July 2020. ISSN 2307-387X. doi: 10.1162/tacl_a_00321.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace's Transformers: State-of-the-art Natural Language Processing. arXiv:1910.03771 [cs], February 2020.

Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, and Noah Goodman. Interpretability at Scale: Identifying Causal Mechanisms in Alpaca. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances in Neural Information Processing Systems 32, pp. 5753-5763. 2019.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing Large Language Models: Problems, Methods, and Opportunities. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 10222-10240, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 632 .

Table 2: BLiMP phenomena and paradigms.

| Phenomenon | Paradigms | Example |
| :---: | :---: | :---: |
| Anaphor <br> Agreement | anaphor_gender_agreement <br> anaphor_number_agreement | Katherine can't help herself $/ *$ himself. <br> Many teenagers were helping themselves/* herself. |
| Determiner- <br> Noun <br> Agreement | determiner_noun_agreement_1 <br> determiner_noun_agreement_2 <br> determiner_noun_agreement_irregular_1 <br> determiner_noun_agreement_irregular_2 <br> determiner_noun_agreement_with_adj_1 <br> determiner_noun_agreement_with_adj_2 <br> determiner_noun_agreement_with_adj_irregular_1 <br> determiner_noun_agreement_with_adj_irregular_2 | Craig explored that grocery store $/ *$ grocery stores. <br> Carl cures those $/ *$ that horses. <br> Phillip was lifting this mouse $/ *$ this mice. <br> Those ladies walk through those $/ *$ that oases. <br> Tracy praises those lucky guys $/ *$ guy. <br> Some actors buy these $/ *$ this gray books. <br> This person shouldn't criticize this upset child $/ *$ children. <br> That adult has brought that $/ *$ those purple octopus. |
| Subject- <br> Verb <br> Agreement | distractor_agreement_relational_noun <br> distractor_agreement_relative_clause <br> irregular_plural_subject_verb_agreement_1 <br> irregular_plural_subject_verb_agreement_2 <br> regular_plural_subject_verb_agreement_1 <br> regular_plural_subject_verb_agreement_2 | A sketch of lights doesn't/*don't appear. <br> Boys that aren't disturbing Natalie suffer/* suffers. <br> This goose isn't/* weren't bothering Edward. <br> The woman $/$ women cleans every public park. <br> Jeffrey hasn't/* <br> The dress $/$. dresses crumples. |
</end of paper 2>


