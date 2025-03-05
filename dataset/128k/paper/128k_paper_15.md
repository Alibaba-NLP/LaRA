<paper 0>
# Seven Failure Points When Engineering a Retrieval Augmented Generation System 

Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek<br>\{scott.barnett,stefanus.kurniawan,srikanth.thudumu,zach.brannelly,mohamed.abdelrazek\}@deakin.edu.au<br>Applied Artificial Intelligence Institute<br>Geelong, Australia


#### Abstract

Software engineers are increasingly adding semantic search capabilities to applications using a strategy known as Retrieval Augmented Generation (RAG). A RAG system involves finding documents that semantically match a query and then passing the documents to a large language model (LLM) such as ChatGPT to extract the right answer using an LLM. RAG systems aim to: a) reduce the problem of hallucinated responses from LLMs, b) link sources/references to generated responses, and c) remove the need for annotating documents with meta-data. However, RAG systems suffer from limitations inherent to information retrieval systems and from reliance on LLMs. In this paper, we present an experience report on the failure points of RAG systems from three case studies from separate domains: research, education, and biomedical. We share the lessons learned and present 7 failure points to consider when designing a RAG system. The two key takeaways arising from our work are: 1) validation of a RAG system is only feasible during operation, and 2) the robustness of a RAG system evolves rather than designed in at the start. We conclude with a list of potential research directions on RAG systems for the software engineering community.


## CCS CONCEPTS

- Software and its engineering $\rightarrow$ Empirical software validation.


## KEYWORDS

Retrieval Augmented Generation, RAG, SE4AI, Case Study

## ACM Reference Format:

Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek . 2024. Seven Failure Points When Engineering a Retrieval Augmented Generation System. In Proceedings of 3rd International Conference on AI Engineering - Software Engineering for AI (CAIN 2024). ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 INTRODUCTION

The new advancements of Large Language Models (LLMs), including ChatGPT, have given software engineers new capabilities to[^0]

build new HCI solutions, complete complex tasks, summarise documents, answer questions in a given artefact(s), and generate new content. However, LLMs suffer from limitations when it comes to up-to-date knowledge or domain-specific knowledge currently captured in company's repositories. Two options to address this problem are: a) Finetuning LLMs (continue training an LLM using domain specific artifacts) which requires managing or serving a fine-tuned LLM; or b) use Retrieval-Augmented Generation (RAG) Systems that rely on LLMs for generation of answers using existing (extensible) knowledge artifacts. Both options have pros and cons related to privacy/security of data, scalability, cost, skills required, etc. In this paper, we focus on the RAG option.

Retrieval-Augmented Generation (RAG) systems offer a compelling solution to this challenge. By integrating retrieval mechanisms with the generative capabilities of LLMs, RAG systems can synthesise contextually relevant, accurate, and up-to-date information. A Retrieval-Augmented Generation (RAG) system combines information retrieval capabilities, and generative prowess of LLMs. The retrieval component focuses on retrieving relevant information for a user query from a data store. The generation component focuses on using the retrieved information as a context to generate an answer for the user query. RAG systems are an important use case as all unstructured information can now be indexed and available to query reducing development time no knowledge graph creation and limited data curation and cleaning.

Software engineers building RAG systems are expected to preprocess domain knowledge captured as artifacts in different formats, store processed information in appropriate data store (vector database), implement or integrate the right query-artifact matching strategy, rank matched artifacts, and call the LLMs API passing in user queries and context documents. New advances for building RAG systems are constantly emerging $[8,12]$ but how they relate and perform for a specific application context has to be discovered.

In this work we present the lessons learned and 7 failure points arising from 3 case studies. The purpose of this paper is to provide 1) a reference to practitioners and 2) to present a research road map for RAG systems. To the best of our knowledge, we present the first empirical insight into the challenges with creating robust RAG systems. As advances in LLMs continue to take place, the software engineering community has a responsibility to provide knowledge on how to realise robust systems with LLMs. This work is an important step for robustness in building RAG systems.

Research questions for this work include:

- What are the failure points that occur when engineering a RAG system? (section 5) We present an empirical experiment using the BioASQ data set to report on potential failure points. The experiment involved 15,000 documents and 1000 question
and answer pairs. We indexed all documents then ran the queries and stored the generated responses using GPT-4. All question and answer pairs were then validated with OpenAI evals ${ }^{1}$. Manual inspection (all discrepancies, all flagged as incorrect, and a sample of correct labels) was analysed to identify the patterns.
- What are the key considerations when engineering a $R A G$ system? (section 6) We present the lessons learned from three case studies involving the implementation of a RAG system. This presents the challenges faced and insights gained.

Contributions arising from this work include:

- A catalogue of failure points (FP) that occur in RAG systems.
- An experience report from 3 case studies of implementing a RAG system. Two currently running at Deakin University.
- A research direction for RAG systems based on the lessons learned from the 3 case studies.


## 2 RELATED WORK

Retrieval augmented generation encompasses using documents to augment large language models through pre-training and at inference time $[7,9,12]$. Due to the compute cost, data preparation time and required resources using RAG without training or finetuning is an attractive proposition. However, challenges arise when using large language models for information extraction such as performance with long text [8].

A recent survey [19] showed that large language models are used across the RAG pipeline including retriever, data generation, rewriter, and reader. Our work complements this survey by taking a software engineering perspective to shine a light on what issues engineers will face and what software engineering research is necessary to realise solutions with the current state-of-the-art RAG systems.

Emerging work has looked at benchmarking RAG systems [3] but not at the failures occurring during implementation. Software engineering research has investigated the use of RAG systems for code-related tasks [15]. However, the application of RAG systems is broader than software engineering tasks. This paper complements existing work by presenting challenges faced during the implementation of a RAG system with a focus on practitioners.

Errors and failures that arise from RAG systems overlap with other information retrieval systems including 1) no metrics for query rewriting, 2) document re-ranking, and 3) effective content summarisation [19]. Our results confirm this The unique aspects are related to the semantic and generative nature of the use of large language models including evaluating factual accuracy [16].

## 3 RETRIEVAL AUGMENTED GENERATION

With the explosion in popularity of large language model services such as ChatGPT ${ }^{2}$, Claude ${ }^{3}$, and Bard ${ }^{4}$, people have explored their use as a question and answering systems. While the performance is impressive [16] there are two fundamental challenges: 1) hallucinations - where the LLM produces a response that looks right[^1]

but is incorrect, and 2) unbounded - no way to direct or update the content of the output (other than through prompt engineering). A RAG system is an information retrieval approach designed to overcome the limitations of using a LLM directly.

RAG works by taking a natural language query is converted into an embedding which is used to semantically search a set of documents. Retrieved documents are then passed to a large language model to generate an answer. An overview of a RAG system is shown in Figure 1 as two separate processes, Index and Query. See this survey for more details [19]

### 3.1 Index Process

In a RAG system, the retrieval system works using embeddings that provide a compressed semantic representation of the document. An embedding is expressed as a vector of numbers. During the Index process each document is split into smaller chunks that are converted into an embedding using an embedding model. The original chunk and the embedding are then indexed in a database. Software engineers face design decisions around how best to chunk the document and how large a chunk should be. If chunks are too small certain questions cannot be answered, if the chunks are too long then the answers include generated noise.

Different types of documents require different chunking and processing stages. For example, video content requires a transcription pipeline to extract the audio and convert to text prior to encoding (see subsection 4.2. The choice of which embedding to use also matters as changing the embedding strategy requires re-indexing all chunks. An embedding should be chosen based on the ability to semantically retrieve correct responses. This process depends on the size of the chunks, the types of questions expected, the structure of the content and the application domain.

### 3.2 Query Process

The Query process takes place at run time. A question expressed as natural language is first converted into a general query. To generalise the query a large language model is used which enables additional context such as previous chat history to be included in the new query. An embedding is then calculated from the new query to use for locating relevant documents from the database. Top-k similar documents are retrieved using a similarity method such as cosine similarity (vector databases have techniques such as inverted indexes to speed up retrieval time). The intuition is that chunks that are semantically close to the query are likely to contain the answer.

Retrieved documents are then re-ranked to maximise the likelihood that the chunk with the answer is located near the top. The next stage is the Consolidator which is responsible for processing the chunks. This stage is needed to overcome the limitations of large language models 1) token limit and 2) rate limit. Services such as OpenAI have hard limits on the amount of text to include in a prompt. This restricts the number of chunks to include in a prompt to extract out an answer and a reduction strategy is needed to chain prompts to obtain an answer. These online services also restrict the number of tokens to use within a time frame restricting the latency of a system. Software engineers need to consider these tradeoffs when designing a RAG system.

![](https://cdn.mathpix.com/cropped/2024_06_04_64eac334d62f734ac42bg-3.jpg?height=629&width=1742&top_left_y=295&top_left_x=186)

Figure 1: Indexing and Query processes required for creating a Retrieval Augmented Generation (RAG) system. The indexing process is typically done at development time and queries at runtime. Failure points identified in this study are shown in red boxes. All required stages are underlined. Figure expanded from [19].

The final stage of a RAG pipeline is when the answer is extracted from the generated text. Readers are responsible for filtering the noise from the prompt, adhering to formatting instructions (i.e. answer the question as a list of options), and producing the output to return for the query. Implementation of a RAG system requires customising multiple prompts to process questions and answers. This process ensures that questions relevant for the domain are returned. The use of large language models to answer real time questions from documents opens up new application domains where question and answering is new capability. Thus, RAG systems are difficult to test as no data exists and needs to be experimentally discovered through either a) synthetic data generation, or b) piloting the system with minimal testing.

## 4 CASE STUDIES

This study conducted three case studies to discover the challenges that arise when implementing RAG systems. A summary of each of the case studies is shown in Table 1. All scripts, data, and examples of each of the failure points for the BioASQ case study are available online ${ }^{5}$. The other two case studies have been excluded due to confidentiality concerns.

### 4.1 Cognitive Reviewer

Cognitive Reviewer is a RAG system designed to support researchers in analysing scientific documents. Researchers specify a research question or objective and then upload a collection of related research papers. All of the documents are then ranked in accordance with the stated objective for the researcher to manually review. The researcher can also ask questions directly against all of the documents. Cognitive Reviewer is currently used by PhD students from Deakin University to support their literature reviews. The Cognitive Reviewer does the Index process at run time and relies[^2]

on a robust data processing pipeline to handle uploaded documents i.e. no quality control possible at development time. This system also uses a ranking algorithm to sort the uploaded documents.

### 4.2 AI Tutor

The AI Tutor is a RAG system where students ask questions about the unit and answers are sourced from the learning content. Students are able to verify the answers by accessing a sources list from where the answer came from. The AI Tutor works by integrating into Deakin's learning management system, indexing all of the content including PDF documents, videos, and text documents. As part of the Index process, videos are transcribed using the deep learning model Whisper [17] before being chunked. The AI Tutor was developed between August 2023 to November 2023 for a pilot in a unit with 200 students that commenced the 30th of October 2023. Our intention is to present the lessons learned during implementation and present a followup findings at the conclusion of the pilot. This RAG pipeline includes a rewriter to generalise queries. We implemented a chat interface where previous dialogue between the user and the AI Tutor was used as part of the context for each question. The rewriter considers this context and rewrites the query to resolve ambiguous requests such as 'Explain this concept further.'

### 4.3 Biomedical Question and Answer

The previous case studies focused on documents with smaller content sizes. To explore the issues at a larger scale we created a RAG system using the BioASQ [10] dataset comprised of questions, links to document, and answers. The answers to questions were one of yes/no, text summarisation, factoid, or list. This dataset was prepared by biomedical experts and contains domain specific question and answer pairs. We downloaded 4017 open access documents from the BioASQ dataset and had a total of 1000 questions. All documents were indexed and the questions asked against the RAG system. The generated questions were then evaluated using the

| Case Study | Domain | Doc Types | Dataset Size | RAG Stages | Sample Questions |
| :--- | :--- | :--- | :---: | :--- | :--- |
| Cognitive <br> Reviewer | Research | PDFs | (Any size) | Chunker, Rewriter, Re- <br> triever, Reader | What are the key points covered in <br> this paper? |
| AI Tutor |  |  |  |  |  |

Table 1: A summary of the RAG case studies presented in this paper. Case studies marked with a * are running systems currently in use.

OpenEvals technique implemented by OpenAI ${ }^{6}$. From the generated questions we manually inspected 40 issues and all issues that the OpenEvals flagged as inaccurate. We found that the automated evaluation was more pessimistic than a human rater for this domain. However, one threat to validity with this finding is that BioASQ is a domain specific dataset and the reviewers were not experts i.e. the large language model may know more than a non-expert.

## 5 FAILURE POINTS OF RAG SYSTEMS

From the case studies we identified a set of failure points presented below. The following section addresses the research question What are the failure points that occur when engineering a RAG system?

FP1 Missing Content The first fail case is when asking a question that cannot be answered from the available documents. In the happy case the RAG system will respond with something like "Sorry, I don't know". However, for questions that are related to the content but don't have answers the system could be fooled into giving a response.

FP2 Missed the Top Ranked Documents The answer to the question is in the document but did not rank highly enough to be returned to the user. In theory, all documents are ranked and used in the next steps. However, in practice the top $\mathrm{K}$ documents are returned where $\mathrm{K}$ is a value selected based on performance.

FP3 Not in Context - Consolidation strategy Limitations Documents with the answer were retrieved from the database but did not make it into the context for generating an answer. This occurs when many documents are returned from the database and a consolidation process takes place to retrieve the answer

FP4 Not Extracted Here the answer is present in the context, but the large language model failed to extract out the correct answer. Typically, this occurs when there is too much noise or contradicting information in the context.

FP5 Wrong Format The question involved extracting information in a certain format such as a table or list and the large language model ignored the instruction.

FP6 Incorrect Specificity The answer is returned in the response but is not specific enough or is too specific to address the user's need. This occurs when the RAG system designers have a desired outcome for a given question such as teachers for students. In this case, specific educational content should be provided with answers not just the answer. Incorrect specificity also occurs when users are not sure how to ask a question and are too general.[^3]

FP7 Incomplete Incomplete answers are not incorrect but miss some of the information even though that information was in the context and available for extraction. An example question such as "What are the key points covered in documents A, B and C?" A better approach is to ask these questions separately.

## 6 LESSONS AND FUTURE RESEARCH DIRECTIONS

The lessons learned from the three case studies are shown in Table 2. We present our findings for the research question: What are the key considerations when engineering a RAG system? Based on our takeaways we identified multiple potential research areas linked to RAG as follows:

### 6.1 Chunking and Embeddings

Chunking documents sounds trivial. However, the quality of chunking affects the retrieval process in many ways and in particular on the embeddings of the chunk then affects the similarity and matching of chunks to user queries. There are two ways of chunking: heuristics based (using punctuation, end of paragraph, etc.), and semantic chunking (using the semantics in the text to inform start-end of a chunk). Further research should explore the tradeoffs between these methods and their effects on critical downstream processes like embedding and similarity matching. A systematic evaluation framework comparing chunking techniques on metrics like query relevance and retrieval accuracy would benefit the field.

Embeddings represent another active research area, including generating embeddings for multimedia and multimodal chunks such as tables, figures, formulas, etc. Chunk embeddings are typically created once during system development or when a new document is indexed. Query preprocessing significantly impacts a RAG system's performance, particularly handling negative or ambiguous queries. Further research is needed on architectural patterns and approaches [5] to address the inherent limitations with embeddings (quality of a match is domain specific).

### 6.2 RAG vs Finetuning

LLMs are great world models due to the amount of training data, and finetuning tasks applied on the model before it's released. However, these models are general-purpose models (may not know the very specifics of your domain) and also not up to date (there is a cutoff date on their knowledge). Fine-tuning and RAG offer two potential customisation pathways, each with distinct tradeoffs. Finetuning requires curating internal datasets to adapt and train the LLM on. However, all your data are baked into the model and you need to

| $\mathbf{F P}$ | Lesson | Description | Case Studies |
| :---: | :---: | :---: | :---: |
| FP4 | Larger context get better results (Context refers to a <br> particular setting or situation in which the content <br> occurs) | A larger context enabled more accurate responses <br> (8K vs $4 \mathrm{~K})$. Contrary to prior work with GPT-3.5 [13] | AI Tutor |
| FP1 | Semantic caching drives cost and latency down | RAG systems struggle with concurrent users due to <br> rate limits and the cost of LLMs. Prepopulate the <br> semantic cache with frequently asked questions [1]. | AI Tutor |
| FP5-7 | Jailbreaks bypass the RAG system and hit the safety <br> training. | Research suggests fine-tuning LLMs reverses safety <br> training [11], test all fine-tuned LLMs for RAG sys- <br> tem. | AI Tutor |
| $\mathrm{FP} 2, \mathrm{FP} 4$ | Adding meta-data improves retrieval. | Adding the file name and chunk number into the <br> retrieved context helped the reader extract the re- <br> quired information. Useful for chat dialogue. | AI Tutor |
| FP2, FP4-7 | Open source embedding models perform better for <br> small text. | Opensource sentence embedding models performed <br> as well as closed source alternatives on small text. | BioASQ, AI Tutor |
| FP2-7 | RAG systems require continuous calibration. | RAG systems receive unknown input at runtime <br> requiring constant monitoring. | AI Tutor, BioASQ |
| FP1, FP2 | Implement a RAG pipeline for configuration. | A RAG system requires calibrating chunk size, <br> embedding strategy, chunking strategy, retrieval <br> strategy, consolidation strategy, context size, and <br> prompts. | Cognitive Reviewer, <br> AI Tutor, BioASQ |
| $\mathrm{FP} 2, \mathrm{FP} 4$ | RAG pipelines created by assembling bespoke solu- <br> tions are suboptima. | End-to-end training enhances domain adaptation <br> in RAG systems [18]. | BioASQ, AI Tutor |
| FP2-7 | Testing performance characteristics are only possi- <br> ble at runtime. | Offline evaluation techniques such as G-Evals [14] <br> look promising but are premised on having access <br> to labelled question and answer pairs. | Cognitive Reviewer, <br> AI Tutor |

Table 2: The lessons learned from the three case studies with key takeaways for future RAG implementations

sort out the security/privacy (who can access what). Furthermore, as the foundation model itself evolves or you get new data to add to the model, you will need to run finetuning again. On the other side, RAG systems seem to offer a pragmatic solution allowing you to chunk your data as needed and only use relevant chunks into the context to ask the LLM to generate an answer from the included context. This facilitates continuously updating the knowledge with new documents and also gives the control over what chunks the user is able to access. However, optimal strategies for chunk embedding, retrieval, and contextual fusion remain active research. Further work should systematically compare finetuning and RAG paradigms across factors including accuracy, latency, operating costs, and robustness.

### 6.3 Testing and Monitoring RAG systems

Software engineering best practices are still emerging for RAG systems. Software testing and test case generation are one of the areas for refinement. RAG systems require questions and answers that are application specific often unavailable when indexing unstructured documents. Emerging work has considered using LLMs for generating questions from multiple documents [4]. How to generate realistic domain relevant questions and answers remains an open problem.

Once suitable test data is available quality metrics are also required to assist engineers in making quality tradeoffs. Using large language models is expensive, introduces latency concerns, and has performance characteristics that all change with each new release.
This characteristic has previously been studied for machine learning systems [5, 6] but the required adaptations (if any) have yet to be applied to LLM based systems such as RAGs. Another idea is to incorporate ideas from self-adaptive systems to support monitoring and adapting RAG systems, preliminary work has started for other machine learning applications [2].

## 7 CONCLUSION

RAG systems are a new information retrieval that leverages LLMs. Software engineers increasingly interact with RAG systems a) through implementing semantic search, or b) through new codedependent tasks. This paper presented the lessons learned from 3 case studies including an empirical investigation involving 15,000 documents and 1000 questions. Our findings provide a guide to practitioners by presenting the challenges faced when implementing RAG systems. We also included future research directions for RAG systems related to 1) chunking and embeddings, 2) RAG vs Finetuning, and 3) Testing and Monitoring. Large language models are going to continue to obtain new capabilities of interest to engineers and researchers. This paper presents the first investigation into RAG systems from a software engineering perspective.

## ACKNOWLEDGMENTS

To Amanda Edgar, Rajesh Vasa, Kon Mouzakis, Matteo Vergani, Trish McCluskey, Kathryn Perus, Tara Draper, Joan Sutherland and Ruary Ross for their support and involvement in making the AI Tutor project possible.

## REFERENCES

[1] Fu Bang. 2023. GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings. In 3rd Workshop for Natural Language Processing Open Source Software.

[2] Maria Casimiro, Paolo Romano, David Garlan, Gabriel Moreno, Eunsuk Kang, and Mark Klein. 2022. Self-adaptive Machine Learning Systems: Research Challenges and Opportunities. 133-155. https://doi.org/10.1007/978-3-031-15116-3_7

[3] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2023. Benchmarking Large Language Models in Retrieval-Augmented Generation. arXiv preprint arXiv:2309. 01431 (2023).

[4] Mingda Chen, Xilun Chen, and Wen-tau Yih. 2023. Efficient Open Domain Multi-Hop Question Answering with Few-Shot Data Synthesis. arXiv preprint arXiv:2305.13691 (2023).

[5] Alex Cummaudo, Scott Barnett, Rajesh Vasa, and John Grundy. 2020. Threshy: Supporting safe usage of intelligent web services. In Proceedings of the 28th ACM Foint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 1645-1649.

[6] Alex Cummaudo, Scott Barnett, Rajesh Vasa, John Grundy, and Mohamed Abdelrazek. 2020. Beware the evolving 'intelligent'web service! An integration architecture tactic to guard AI-first components. In Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 269-280.

[7] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning. PMLR, 3929-3938.

[8] Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and Hamed Zamani. 2023. Fidlight: Efficient and effective retrieval-augmented text generation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1437-1447.

[9] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020)

[10] Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, and Georgios Paliouras. 2023. BioASQ-QA: A manually curated corpus for biomedical question answering. Scientific Data 10 (2023), 170. Citation Key: 422.

[11] Simon Lermen, Charlie Rogers-Smith, and Jeffrey Ladish. 2023. LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B. arXiv:2310.20624 [cs.LG]

[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33 (2020), 9459-9474.

[13] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172 (2023).

[14] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment, may 2023. arXiv preprint arXiv:2303.16634 (2023).

[15] Noor Nashid, Mifta Sintaha, and Ali Mesbah. 2023. Retrieval-based prompt selection for code-related few-shot learning. In Proceedings of the 45th International Conference on Software Engineering (ICSE'23).

[16] OpenAI. 2023. GPT-4 Technical Report. https://doi.org/10.48550/ARXIV.2303. 08774

[17] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning. PMLR, 28492-28518.

[18] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for Computational Linguistics 11 (2023), 1-17.

[19] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. arXiv preprint arXiv:2308.07107 (2023).


[^0]:    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

    CAIN 2024, April 2024, Lisbon, Portugal

    2024 Association for Computing Machinery.

    ACM ISBN 978 -x-xxxx-xxxx-x/YY/MM... $\$ 15.00$

    https://doi.org/10.1145/nnnnnnn.nnnnnnn

[^1]:    ${ }^{1}$ https://github.com/openai/evals

    ${ }^{2}$ https://chat.openai.com/

    ${ }^{3}$ https://claude.ai/

    ${ }^{4}$ https://bard.google.com/

[^2]:    ${ }^{5}$ https://figshare.com/s/fbf7805b5f20d7f7e356

[^3]:    ${ }^{6}$ https://github.com/openai/evals

</end of paper 0>


<paper 1>
# When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively 

Tiziano Labruna ${ }^{\mathrm{a}, \mathrm{b}}$ 周, Jon Ander Campos ${ }^{\mathrm{c}}$ and Gorka Azkune ${ }^{\mathrm{d}}$<br>${ }^{a}$ University of Bozen-Bolzano<br>${ }^{\mathrm{b}}$ Fondazione Bruno Kessler<br>${ }^{\mathrm{c}}$ Cohere<br>${ }^{\mathrm{d}}$ HiTZ Center - Ixa, University of the Basque Country UPV/EHU<br>ORCID (Tiziano Labruna): https://orcid.org/0000-0001-7713-7679, ORCID (Jon Ander Campos):<br>https://orcid.org/0000-0002-1447-5870, ORCID (Gorka Azkune): https://orcid.org/0000-0002-2506-7426


#### Abstract

In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answer a given question. Given the performance of IR systems, the optimal strategy for question answering does not always entail external information retrieval; rather, it often involves leveraging the parametric memory of the LLM itself. Prior research has identified this phenomenon in the PopQA dataset, wherein the most popular questions are effectively addressed using the LLM's parametric memory, while less popular ones require IR system usage. Following this, we propose a tailored training approach for LLMs, leveraging existing open-domain question answering datasets. Here, LLMs are trained to generate a special token, $\langle$ RET $\rangle$, when they do not know the answer to a question. Our evaluation of the Adaptive Retrieval LLM (ADAPT-LLM) on the PopQA dataset showcases improvements over the same LLM under three configurations: (i) retrieving information for all the questions, (ii) using always the parametric memory of the LLM, and (iii) using a popularity threshold to decide when to use a retriever. Through our analysis, we demonstrate that ADAPT-LLM is able to generate the $\langle\mathrm{RET}\rangle$ token when it determines that it does not know how to answer a question, indicating the need for IR, while it achieves notably high accuracy levels when it chooses to rely only on its parametric memory.


## 1 Introduction

The task of question answering (QA) remains a focal point in Natural Language Understanding research. There are many different datasets serving as benchmarks for evaluating QA models, such as Natural Questions (NQ) [18], SQuAD [25] or QuAC [7], just to mention a few. Nowadays, Large Language Models (LLMs) consistently outperform traditional methods on these benchmarks, showcasing remarkable performance.

Typically, there are two primary approaches to utilize LLMs for question answering:

(i) Closed Book Question Answering: This approach involves strategies like instruction tuning [32] or few-shot prompting [6] to enhance performance. Here, the LLM relies solely on its parametric[^0]

memory to answer questions. However, these parametric memories have inherent limitations as they are based entirely on the training corpus, meaning for example that they could be outdated regarding events occurring after the training process.

(ii) Open Book Question Answering: In this approach, the LLM is coupled with an Information Retriever (IR) system [13, 36]. By leveraging the IR system, the LLM can retrieve relevant context to supplement its understanding and provide more accurate answers.

However, the research conducted by Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] sheds light on the complexity of questionanswering strategies, challenging the notion that the optimal approach always involves the utilization of an IR system. Through the introduction of the PopQA dataset, comprising 14 thousand questions annotated with popularity scores, they demonstrated that while LLMs relying solely on their parametric memories excel in addressing high-popularity questions, the efficacy diminishes for lowpopularity questions, where using IR becomes curcial.

Their findings underscore the importance of a hybrid approach, where LLMs utilize parametric memory for high-popularity questions, but use an off-the-shelf IR system to retrieve relevant context to answer low-popularity questions. Central to their methodology is the establishment of a fixed popularity score threshold, which they use to decide whether an IR system has to be employed.

In many cases, however, question answering datasets do not include popularity scores, so relying on such scores is not a generalizable approach. Motivated by this limitation, our study aims to address whether LLMs can autonomously determine when to employ an IR system for improved question answering. To investigate this, we conduct an evaluation of an LLM using an open-domain question answering dataset to identify the questions for which the LLM provides accurate responses and those where its answers are incorrect.

Specifically, for questions where the LLM's response is incorrect, we annotate them with a special token, $\langle$ RET $\rangle$, indicating the need for additional context. Subsequently, we utilize these annotations to construct a new dataset tailored for training purposes, where we teach an LLM to answer directly if it is confident about the answer or to require context it believes is useful for answering the question (see Figure 11. Our hypothesis is that through this training process, the LLM learns to use an IR system when it needs extra context to answer

![](https://cdn.mathpix.com/cropped/2024_06_04_15c429cb2cf0d0e87983g-2.jpg?height=534&width=1357&top_left_y=304&top_left_x=338)

Figure 1: The inference process of ADAPT-LLM step-by-step: given a question (step 1), an LLM decides (step 2) whether to answer the question directly (step 3) or to ask for additional contextual information, generating the special $\langle$ RET $\rangle$ token; for the later, an off-the-shelf IR system is used to retrieve relevant context (step 4), which is used alongside the question to prompt again the LLM for the final answer (step 5).

a question, thus we name it ADAPT-LLM.

To validate our hypothesis, we conducted several experiments on the PopQA dataset [22], as it provides a suitable platform for benchmarking hybrid retrieval strategies. As a result of these experiments we find that:

- ADAPT-LLM consistently outperforms typical fixed strategies for question answering, such as (i) using the IR system for all questions and (ii) relying solely on the parametric memory of the LLM.
- ADAPT-LLM demonstrates performance comparable to strategies that rely on popularity scores to determine when to use an IR system, even without utilizing any popularity score or similar metric. It's worth noting that popularity scores are a unique feature of the PopQA dataset, rendering them inapplicable to other open-domain question answering datasets.
- When ADAPT-LLM decides to retrieve additional information, the results obtained with the context are significantly better than those without it. Similarly, when ADAPT-LLM directly answers questions relying on its parametric memory, it achieves high accuracies. These observations indicate that the model effectively discerns when to retrieve information and when it can answer a question without further context.
- The primary bottleneck for the performance of AdAPT-LLM lies in the IR system. ADAPT-LLM achieves much higher performance with gold passages compared to passages retrieved by the IR system.

Our findings underscore the significance of adaptive retrieval strategies in enhancing the performance of LLMs for question answering tasks. By training ADAPT-LLM to dynamically determine when to retrieve additional context, we demonstrate the feasibility of teaching an LLM how to effectively leverage external information sources only when necessary.

## 2 Related Work

Retrieval-Augmented Generation (RAG) [19] has shown improvements on a wide variety of NLP areas, such as question answering [17, 13, 31 23], truthfulness [14, 21] and language modelling
[12, 5, 26] among others. The ability to ground model generations on retrieved text chunks has also enabled smaller models to match the performance of larger ones [2]. Moreover, due to the extremely high cost of training LLMs, RAG has become the standard way to maintain them updated with new information, not having to re-train the models periodically to incorporate new facts [10].

Even if augmenting LLMs with retrieval is an essential step for the current generation of LLMs [15, 27] it also comes with a cost. Traditional retrieval methods as TF-IDF or BM-25 [29] are only able to retrieve documents with keyword overlap and suffer from lexical gap [4]. In order to try to solve this issue, many pre-trained Transformer encoder based dense models have been proposed [9, 28, 17, 11]. Trained neural models have shown good performance over a variety of retrieval benchmarks but they still struggle in the zero-shot setup for new domains [33]. The quality of the retrieval engine is essential for retrieval-augmented models as this will set the upper bound of the model performance. Moreover, the usage of a retrieval engine, especially when the target document index is huge, can significantly increase the latency of the model and hurt real time applications user experience [3].

On the other hand, as models keep scaling, the world knowledge encoded in their parameters does too [16]. Many previous efforts have shown that language models are able to memorize a significant amount of world knowledge and achieve competitive performance on tasks such as open-domain question answering when they just use their parametric knowledge for solving the task [20, 1, 34, 35].

Motivated by all this, the adaptive approach has been proposed as a new solution [30, 22]. In this approach, if the solution to the task is encoded in the parameters of the model, the model will be directly used for generating a solution. Conversely, if the answer is not encoded in the knowledge of the model, the answer generation will be augmented with external knowledge.

Recently, Schick et al. [30] proposed the Toolformer, a model that can self teach how and when to use external tools via simple API calls including a calculator, search engines, a calendar and so on. The self learning process is based on a synthetic text only corpus that is enriched by prompting an LLM. The LLM first adds inline API calls on top of the unsupervised corpus. These API calls are
then validated by evaluating whether the execution of the API calls is helpful for predicting the future tokens. This unsupervised method significantly boosts model performance in a variety of tasks when compared against non augmented LLMs, but it also makes the model over use tools. As an example, for the QA task the model uses the search engine $99.3 \%$ of the cases. On our work, we try to take advantage of the parametric knowledge of LLMs and just perform retrieval when needed. ADAPT-LLM decreases the usage of IR down to $83.99 \%$ while improving performance over vanilla retrieval.

More similar to our work, Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] propose a dataset and method for measuring when non-parametric information needs to be retrieved. They present the PopQA dataset that contains $14 \mathrm{~K}$ questions about a set of entities with varying popularity. The popularity of an entity is measured by the page views of its Wikipedia page. In order to solve this QA task, they use a popularity score threshold calculated on the PopQA dataset. If the popularity score of an individual entity is below the threshold they perform a retrieval step. On the contrary, if the score is greater than the threshold they directly answer the question. This method yields better results than vanilla retrieval but it requires the calculation of a popularity score that is not available in realistic QA scenarios.

Another relevant contribution in this field, contemporaneous with our research, is the work by Erbacher et al. [8], where they trained an LLM to determine when to utilize external knowledge. They particularly focused on finding the optimal trade-off between the risk of hallucination and the cost of information retrieval, given the potentially high expense associated with IR. Our ADAPT-LLM method adopts a similar approach, training an LLM to learn when to retrieve information. However, we extend this by comparing our method's performance against some baselines, and assess the effectiveness of retrieving information in an adaptive manner against the strategies of never retrieving or always retrieving $\square^{1}$

## 3 Adaptive Retrieval LLM (AdAPT-LLM)

Adaptive retrieval refers to the model's capability to dynamically determine whether to retrieve additional context information for generating answers in question answering tasks. Unlike traditional models that either always incorporate context or never consider it, adaptive retrieval allows the model to selectively retrieve context based on the specific requirements of each question. This adaptive approach aims to optimize performance by leveraging context only when necessary, thereby enhancing the model's ability to generate accurate answers.

As depicted in Figure 1 the process of the ADAPT-LLM unfolds in the following sequence:

1. The first prompt containing the question is sent to the model (step 1 of Figure 1 .
2. The ADAPT-LLM evaluates the prompt to determine whether additional context is necessary to answer the question effectively (step 2).
3. If the model determines that context is not required, it directly produces a response to the question by leveraging its parametric memory (step 3 ).
4. If context is deemed necessary, the ADAPT-LLM model returns a special token, represented as $\langle$ RET $\rangle$, and an off-the-shelf IR system is used to retrieve pertinent context based on the question[^1]
```
Algorithm 1: Training data creation
    Input: Q: questions, A: answers, P: passages, LLM
    Output: $D S_{\text {Adapt }}$ : A training dataset for Adaptive Retrieval
    $D S_{\text {Adapt }}=$ init_empty()
    for $q$, gold_ans, pass in $(Q, A, P)$ do
        ans $=\operatorname{LLM}(\mathrm{q})$
        if ans $=$ gold_ans then
            inst = build_instance('parametric_prompt', q,
                gold_ans)
                $D S_{\text {Adapt } \text {.add(inst) }}$
        end
        else
            inst1 = build_instance('parametric_prompt', q,
                "<RET>")
            $D S_{\text {Adapt }}$.add(inst1)
            inst2 = build_instance('context_prompt', q, gold_ans,
            pass)
            $D S_{\text {Adapt }}$.add(inst2)
        end
    end
    return $D S_{\text {Adapt }}$
```

(step 4); the context is then combined with the original question prompt to form a comprehensive representation for answer generation (step 5).

The decision-making process of ADAPT-LLM enables the model to determine the necessity of context for answering questions through dynamic assessment of each prompt. This flexible behavior allows the model to strike a balance between utilizing context for enhanced understanding and delivering direct answers when sufficient.

### 3.1 Training ADAPT-LLM

Here, we delineate the methodology employed to train our ADAPTLLM model. The process of crafting the training data, denoted as $D S_{\text {Adapt }}$, is presented in Algorithm 1 .

We begin by selecting an open-domain question answering dataset containing questions $Q$, associated context passages $P$, and corresponding answers $A$. We initialize $D S_{\text {Adapt }}$ to an empty set (line 1 of the algorithm). For each question in $Q$, we leverage the base LLM without any retrieval mechanism to perform a zero-shot inference (line 3). This step allows us to differentiate questions for which the model generates correct answers from those where its responses are inaccurate. This process can be understood as a way to discover what the base LLM knows due to its parametric memory. For questions where the model's response is accurate (line 4), we build a training set instance incorporating the following prompt, which we call parametric_prompt:

Prompt: Answer the question Q. If you need help answer <RET> to get the context. $Q$ : $\{\ldots\}$

Alongside this prompt, we include the corresponding question from $Q$ and the golden answer from $A$, collectively forming the instance (line 5), which is subsequently appended to the $D S_{\text {Adapt }}$ dataset (line 6).

In contrast, if the LLM fails to produce a correct response to the question (line 8), we build two different instances. The first employs

| Training Set | Model configuration | Accuracy |
| :---: | :---: | :---: |
| NQ | NEVER RETRIEVE | $21.43 \%$ |
|  | ALWAYS RETRIEVE | $35.86 \%$ |
|  | ADAPT-LLM (ours) | $\mathbf{3 6 . 7 7 \%}$ |
| SQUAD | NEVER RETRIEVE | $21.22 \%$ |
|  | ALWAYS RETRIEVE | $36.59 \%$ |
|  | ADAPT-LLM (ours) | $\mathbf{3 8 . 1 5 \%}$ |

Table 1: Performance comparison of Llama-2 models trained on the NQ and SQuAD datasets using different retrieval configurations (NR-LLM, AR-LLM, and ADAPT-LLM), evaluated on the PopQA test set. Exact match accuracy is reported for all models.

the same parametric_prompt as previously described, with $\langle$ RET $\rangle$ designated as the answer (line 9), indicating the necessity for additional context. The second prompt, termed context_prompt, encompasses contextual information alongside the question:

Prompt: Answer the question Q given the

context C. $Q:\{\ldots\}, C:\{\ldots\}$

For this instance, we include the prompt, the question from $Q$, the golden answer from $A$, and the corresponding context passage from $P$ (line 11).

After populating the dataset with both types of prompts for questions where the LLM could not respond accurately and only the parametric_prompt with golden answers for all other questions, our training set $D_{\text {Adapt }}$ is prepared for the subsequent fine-tuning phase. The fine-tuning process entails training the base LLM on our dataset, resulting in the ADAPT-LLM model.

This approach ensures that the model effectively learns to discern when context is necessary for answering questions, or to provide a direct response when it suffices, as well as answer directly when provided with context.

### 3.2 Inference

In the inference phase, we utilize the fine-tuned model to generate responses to unseen questions. We employ the same prompts used during the training phase, as outlined in Section 3.1

Initially, the model is prompted to either provide a direct response or return $\langle$ RET $\rangle$ if it is unsure of the answer. If the model returns $\langle\mathrm{RET}\rangle$, we proceed with information retrieval to acquire relevant context by means of an off-the-shelf IR system. Subsequently, we augment the question with the retrieved context and prompt the model again using the second type of prompt introduced during the training phase.

## 4 Experiments and Results

In this section, we outline the experimental framework aimed at assessing the performance of the proposed adaptive retrieval approach, ADAPT-LLM. We begin by describing the datasets utilized (Section 4.1, followed by an overview of our base model (Section 4.2, the different configurations of the base model (Section4.3), and the training details (Section 4.4 . Subsequently, we introduce the three primary experiments:

1. Evaluation of ADAPT-LLM performance compared to the following baseline models: (i) an LLM that retrieves contextual information for all questions, and (ii) an LLM that exclusively relies on its

|  | NQ | SQuAD | PopQA |
| :---: | :---: | :---: | :---: |
| Questions | 58,880 | 87,599 | 14,282 |
| Words/question | 9.20 | 10.06 | 6.62 |
| Words/answer | 2.26 | 3.16 | 2.04 |

Table 2: Comparison of the three datasets we use for our experiments, i.e. SQuAD, NQ and PopQA. For each of them we provide the number of questions, and the average number of words per question and answer.

parametric memory without using an IR system for any question (Section 4.5).

2. Analysis of ADAPT-LLM's ability to determine when extra context is necessary to answer a question (Section 4.6.
3. Comparison with the state-of-the-art approach for PopQA (Section 4.7 .

### 4.1 Datasets

To ensure comprehensive training and evaluation of our models, we specifically selected three diverse question answering datasets. For training, we chose NQ [18] and SQuAD [25], as they are widely recognized datasets that assess factual knowledge and are based on Wikipedia. For evaluation, we opted for PopQA [22]. Below are brief descriptions of each dataset:

NQ The Natural Questions dataset [18] is a collection of real-world questions derived from Google search queries, accompanied by longform text passages obtained from Wikipedia articles and providing a diverse range of topics and natural language variations. We utilize this dataset for training our models in the experiments.

SQuAD The Stanford Question Answering Dataset SQuAD [25] is a widely utilized dataset in the field of natural language processing and comprises questions posed by crowdworkers on a diverse range of Wikipedia articles, along with relevant paragraph passages serving as context. We utilize this dataset for training our models in the experiments.

PopQA The Popular Questions and Answers dataset [22] consists of curated questions sourced from various online platforms, encompassing a wide range of domains and styles. Given the variability in the effectiveness of context retrieval strategies observed in this dataset, we select PopQA as our test set to evaluate the language models' performance in determining when context is necessary for accurate answer provision.

### 4.2 Base Model

In our experiments, we employ Llama-2 [34] as our base LLM. Llama-2 is an open-source instruction-based LLM, which comes in versions of $7 \mathrm{~B}, 13 \mathrm{~B}$, and $70 \mathrm{~B}$ parameters. The model is pretrained on an expanded corpus sourced from publicly available online data sources. This corpus offers a $40 \%$ increase in size compared to its predecessor, contributing to the model's enhanced performance and capabilities.

Additionally, Llama-2 features an extended context length, effectively doubling its capacity to process and comprehend longer sequences of text. These enhancements significantly improve the model's effectiveness across various natural language understanding tasks. Specifically, for our experiments, we utilize the Llama-2 model

| Training | $\langle$ RET $\rangle$ Usage | $\langle$ RET $\rangle$ |  | No $\langle$ RET $\rangle$ |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Acc. w/ context | Acc. w/o context | Acc. w/ context | Acc. w/o context |
| NQ | $82.26 \%$ | $33.04 \%$ | $14.65 \%$ | $55.72 \%$ | $62.36 \%$ |
| SQuAD | $83.93 \%$ | $33.40 \%$ | $9.94 \%$ | $57.73 \%$ | $62.92 \%$ |

Table 3: Results of the usage of the $\langle$ RET $\rangle$ token in the ADAPT-LLM model. The first column shows the percentage of PopQA questions for which the model requests additional context. The second column focuses on the questions for which ADAPT-LLM asks for context ( $\langle$ RET $\rangle$ ), comparing the performance between answering those questions with and without context. The last column (No $\langle$ RET $\rangle$ ) is for questions which ADAPT-LLM decides to answer directly. We also compare the performance with and without the context retrieved by the IR system.

with 7B parameters, leveraging its robust capabilities for our specific research objectives.

### 4.3 Model Configurations

We conduct the experiments using three different model configurations, corresponding to the three different ways in which an LLM and an IR system can be combined:

- Adaptive Retrieval (ADAPT-LLM). The AdAPT-LLM model dynamically decides whether to retrieve context based on the question and its perceived need for contextual information, as explained in Section 3.1 As the IR system, we use Contriever [11], which is an unsupervised model pretrained on a large corpus, followed by fine-tuning on MS MARCO [24]. We only retrieve the most relevant passage according to the IR system to prompt the base LLM for the final answer.
- Never-Retrieve (NR-LLM). This model configuration is trained to answer questions solely based on the question text without considering any contextual information. It serves as the baseline for evaluating the performance of question answering models in the absence of context.
- Always-Retrieve (AR-LLM). In contrast to the NR-LLM model, this configuration always retrieves context passages to assist in answering questions. It is trained to utilize context consistently for generating answers. To ensure a fair comparison with ADAPTLLM, we also use Contriever [11] as the IR system and only retrieve the most relevant passage as context.


### 4.4 Training Details

For all three model configurations (ADAPT-LLM, AR-LLM and NRLLM) and both training sets (SQuAD and NQ), we adhere to the parameter configuration established in Alpaca-Lora [32] which includes a batch size of 128 , three epochs, and a fixed learning rate of 3e-4. We incorporated LoRA (Low-Rank Adaptation) regularization, with parameters configured for $\mathrm{r}=8$, alpha $=16$, and a dropout rate of 0.05 . Training was performed on an NVIDIA A40 GPU, for an average training time of approximately 8 hours. We do not perform any model selection and we use the last checkpoint after 3 epochs of training.

### 4.5 Validating the Adaptive Retrieval Approach

In order to assess the effectiveness of our adaptive approach (ADAPTLLM) in comparison to the NR-LLM and AR-LLM configurations, we conducted fine-tuning of the Llama-2 model on both the NQ and SQuAD datasets across all three configurations. For the NR-LLM and AR-LLM configurations, we constructed training samples by extracting question-answer pairs from the datasets and incorporating corresponding instruction prompts.
Specifically, prompts for the NR-LLM configuration instructed the model to answer questions without additional context, whereas prompts for the AR-LLM configuration included both the question and contextual information. In contrast, the ADAPT-LLM training set was constructed following the approach outlined in Section 3.1 . employing a two-step process. As a result of this process, the $74.72 \%$ of the questions in NQ are marked with the $\langle\mathrm{RET}\rangle$ token, whereas the $87.49 \%$ questions are marked for SQuAD.

The trained models were then tested on the PopQA dataset to evaluate their performance in a real-world question answering scenario. During inference, the NR-LLM and AR-LLM models were utilized as is, with corresponding instruction prompts provided, and outputs expected to be answers to the questions. Conversely, for the ADAPTLLM model, we followed the same prompt procedure as explained in Section 3.2

The generated answers are then compared to the set of possible answers for each question, which are already annotated in the PopQA test set. The evaluation metric used is Exact Match Accuracy, which measures the percentage of generated outputs that exactly match one of the possible answers for the corresponding question.

Table 1 presents the results of this experiment, illustrating the performance of the Llama-2 model across the different configurations and datasets. Across both the NQ and SQuAD training datasets, the ADAPT-LLM configuration consistently outperforms the Never Retrieve (NR-LLM) and Always Retrieve (AR-LLM) configurations on the PopQA test set. As can be observed, NR-LLM exhibits the lowest performance among the models, with an accuracy difference of approximately 14 absolute points compared to the other configurations. This disparity suggests that the parametric memory of Llama-2 alone is not sufficient for effectively answering PopQA questions.

The differences between AR-LLM and ADAPT-LLM are narrower. Specifically, the ADAPT-LLM configuration achieves an accuracy of $36.77 \%$ and $38.15 \%$ on the PopQA test set when trained on the NQ and SQuAD datasets, respectively, compared to $35.86 \%$ and $36.59 \%$ for the AR-LLM configuration. Across both training datasets, ADAPT-LLM outperforms AR-LLM, with the largest difference observed when trained on SQuAD.

All in all, these results underscore the efficacy of the adaptive retrieval approach in dynamically determining the necessity of context for accurate question answering, resulting in improved performance compared to fixed strategies of always or never retrieving context.

Although the disparity between training ADAPT-LLM on NQ or SQuAD is relatively minor, we try to determine the suitability of a training set for a given evaluation set. While both training sets (NQ and SQuAD) and the evaluation set (PopQA) are based on Wikipedia, subtle differences may exist.

Table 2 provides insights into the characteristics of the three datasets involved in our experimental procedure, including the total number of questions and the average number of words per ques-
![](https://cdn.mathpix.com/cropped/2024_06_04_15c429cb2cf0d0e87983g-6.jpg?height=484&width=1692&top_left_y=300&top_left_x=157)

Figure 2: Histograms depicting the proportion of questions where ADAPT-LLM trained on NQ (left) and ADAPT-LLM trained on SQuAD (right) ask for extra context for different popularity score intervals.

tion and answer. While NQ appears to be closer to PopQA in terms of question and answer lengths, the key factor influencing the better results of training ADAPT-LLM on SQuAD may be the number of questions in the training dataset ( $\sim 87 \mathrm{~K}$ in SQuAD and $\sim 58 \mathrm{~K}$ in $\mathrm{NQ}$ ). Further analyses are required to elucidate the factors that render a training dataset more suitable for a given target dataset (which is beyond the scope of our study), but these results suggest that scale may play once again a crucial role.

### 4.6 Contextual Retrieval Decision Analysis

In this experiment, our objective is to once again evaluate the effectiveness of the ADAPT-LLM model, this time focusing on its ability to accurately determine when additional context is needed. For this purpose, we adhere to the following steps:

1. We conduct inference on the ADAPT-LLM model using the PopQA test set, prompting it to either return an answer directly or indicate the need for additional context by returning $\langle\mathrm{RET}\rangle$.
2. In the case of receiving a $\langle$ RET $\rangle$ response from the ADAPT-LLM model, we proceed with the following steps:

2.1. We conduct inference on the ADAPT-LLM model, prompting it to return an answer given the context obtained from the IR system.

2.2. We also conduct inference on the NR-LLM model with the instruction to provide an answer directly without additional context.

3. If the ADAPT-LLM model decides to answer the question directly relying only on its parametric memory:

3.1. We conduct inference on the ADAPT-LLM model, prompting it to return the answer without providing context.

3.2. We conduct inference on the AR-LLM model with the instruction to provide an answer using the context retrieved by the IR system.

Table 3 presents the results of this experiment. The first thing to note is that the ADAPT-LLM model generates the $\langle$ RET $\rangle$ token for approximately $82-83 \%$ of the questions in the PopQA dataset, with

| Passages | SQuAD Dev <br> Acc. | NQ Dev <br> Acc. |
| :---: | :---: | :---: |
| Gold | $\mathbf{8 9 . 4 2 \%}$ | $\mathbf{6 9 . 7 6 \%}$ |
| Contriever | 22.49 | $27.04 \%$ |

Table 4: Performance comparison of ADAPT-LLM for the SQuAD and NQ dev sets, when using the gold passages provided by the datasets and when using the best passage retrieved by Contriever.

similar ratios observed across both training datasets. This observation aligns with the low performance of the NR-LLM configuration demonstrated in Table 1

However, ADAPT-LLM consistently determines when additional context is required to answer a question accurately. Across both the NQ and SQuAD training datasets, ADAPT-LLM exhibits significantly higher accuracy when retrieving context compared to the NRLLM model's accuracy without context (as indicated in the $\langle$ RET $\rangle$ column of Table 3). Specifically, for the NQ dataset, the accuracy of the ADAPT-LLM model when requesting context is $33.04 \%$, whereas the accuracy of the NR-LLM model without context retrieval is notably lower at $14.65 \%$. Similarly, for the SQuAD dataset, ADAPT-LLM achieves an accuracy of $33.40 \%$ with context retrieval, whereas the NR-LLM model's accuracy without context is substantially lower at $9.94 \%$.

Finally, the last column of Table 3 (No $\langle$ RET $\rangle$ ) shows the performance of ADAPT-LLM when answering questions based solely on its parametric memory. As can be seen, accuracies above $62 \%$ are obtained when no context is utilized, providing further evidence that ADAPT-LLM effectively discerns between retrieving context and providing direct answers to questions. Additionally, we evaluate the performance of these questions when context is added to the input, revealing significant decreases in accuracy of up to 7 absolute points.

These findings provide insights into the effectiveness of the decision-making process employed by the ADAPT-LLM model in determining the necessity of additional context for accurate response generation and present empirical evidence of the necessity of performing dynamic context retrieval in improving the accuracy of ques-
tion answering models.

However, it is notable that the overall performance of the model when answering questions with retrieved context, as observed in Table 3 (approximately 33\%), is relatively low. To further explore this observation, we conduct an additional experiment: evaluating ADAPT-LLM (both versions trained on NQ and SQuAD) on the NQ and SQuAD development splits, comparing performance when using the gold passages of the dataset and the context retrieved by our IR system, Contriever [11]. Unfortunately, PopQA does not provide the gold passages, so direct evaluation there was not possible.

Table 4 presents the results of this experiment. A significant performance difference is observed between using the gold passage and the top passage retrieved by Contriever for both datasets (approximately 67 absolute points for SQuAD and 42 for NQ). This indicates that Contriever, and current IR systems in general, do not consistently retrieve the most relevant passage to answer a given question. This observation underscores the importance of retrieving multiple documents as context, as seen in the most successful open-domain QA systems [13], and highlights its impact on the overall performance of ADAPT-LLM in PopQA.

To further validate the behavior of ADAPT-LLM when requesting additional context, Figure 2 illustrates the proportion of questions for which our model generates the $\langle$ RET $\rangle$ token, aggregated by popularity score intervals (left image for ADAPT-LLM trained on NQ and right image for SQuAD). Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] suggest that high-popularity questions can be adequately answered using the parametric memory of the LLM, while lower popularity scores necessitate extra context. In Figure 2 we observe this pattern for both versions of ADAPT-LLM, indicating that our model, despite lacking access to popularity scores during training or inference, has learned effective criteria for requesting additional context.

### 4.7 Comparison with state-of-the-art methods

We conducted a comparative analysis between our ADAPT-LLM model and the current state-of-the-art approach for PopQA proposed by Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22]. Their methodology relies on the popularity score annotated in the PopQA dataset to determine whether a question requires additional context. To establish the optimal threshold for determining question popularity, Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] split the PopQA dataset into $75 \%$ as a development set for threshold determination and $25 \%$ as a test set. In the original paper, they apply this methodology to various LLMs available at that moment (Llama2 was not released yet).

To ensure a fair comparison between ADAPT-LLM and the popularity-based method, we replicated their approach using the Llama-2 7B model to determine the best popularity score threshold (found to be 707,000 ) using the same PopQA development set. This allowed us to obtain results consistent with their methodology while utilizing our base LLM. Similar to the original results in Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] when using smaller models, the popularity score threshold is almost equivalent to always retrieving contextual information for Llama-2 7B. The IR usage is of $99.86 \%$ as presented in Table 5 This clearly shows how the popularity score method struggles with smaller size models, being GPT-3

| Model Configuration | IR usage | Accuracy |
| :---: | :---: | :---: |
| POPULARITY SCORE | $99.86 \%$ | $36.81 \%$ |
| ADAPT-LLM (NQ) | $87.22 \%$ | $35.30 \%$ |
| ADAPT-LLM (SQUAD) | $83.99 \%$ | $\mathbf{3 7 . 2 9 \%}$ |

Table 5: Performance comparison of Llama-2 base models trained on the SQuAD and NQ datasets for the ADAPT-LLM and POPULARITY SCORE configurations. The later mimics the methodology proposed by Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] with the Llama-2 LLM as the base model.

DAVINCI-003 the only model to get a IR usage below $80 \%$ in the original paper when using adaptive retrieval with the Contriever. Subsequently, we evaluated our ADAPT-LLM configuration on the same $25 \%$ test set split and compared the outcomes with those obtained using the method described by Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22]. This systematic comparison enabled us to assess the efficacy of our ADAPT-LLM model in relation to the current state of the art.

The results of this experiment are presented in Table 5 We observe comparable performance between the replicated approach of Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22] and AdAPTLLM when trained on NQ and SQuAD datasets and tested on the $25 \%$ subset of PopQA. It's worth mentioning that ADAPT-LLM does not utilize any information from PopQA, unlike Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh [22], who directly use the popularity score and a $75 \%$ portion of PopQA dataset to find an optimal value for that popularity score. This methodology is not generalizable to other open-domain question answering tasks since the popularity score is a unique feature of PopQA. However, AdAPT-LLM can be applied to any similar dataset. Given these characteristics, we believe that the results obtained by ADAPT-LLM are even more significant, offering comparable performance to an approach that utilizes dataset-specific information. These findings substantiate the validity of our approach, demonstrating its effectiveness even when trained on datasets different from the one used for testing.

## 5 Conclusions

In this paper, we introduce ADAPT-LLM, a LLM which learns to discern when additional context is necessary for answering a question, rather than relying solely on its parametric memory. ADAPT-LLM is the result of fine-tuning a base LLM on an open-domain question answering dataset that has been modified to differentiate between questions answerable with the LLM's parametric memory alone and those requiring supplementary context. To construct these training datasets, we initially subject the base LLM to zero-shot evaluation to determine its accuracy in answering questions. For questions where the model's response is incorrect, we train the LLM to generate a special token, $\langle$ RET $\rangle$, indicating the need for additional context.

Through extensive experiments conducted on the PopQA dataset, we show that ADAPT-LLM performs better than its two fixed alternatives: never retrieving and always retrieving relevant context information. Furthermore, our findings highlight ADAPT-LLM's capability to effectively discern the necessity of additional context, which is the primary objective of this work.

For future investigations, we propose exploring methods to enhance performance when utilizing an IR system, such as incorporating learnable sequential retrieval techniques. Furthermore, we believe it would be valuable to conduct a more in-depth analysis of the interaction between training and testing datasets in the development of ADAPT-LLM systems.

## 6 Acknowledgments

This work received partial support from the Basque Government through research group funding IT1805-22 and the ICL4LANG project (grant no. KK-2023/00094). Additionally, we acknowledge the support of several MCIN/AEI/10.13039/501100011033 projects: (i) DeepKnowledge (PID2021-127777OB-C21) and funding from FEDER, EU; (ii) AWARE (TED2021-131617B-I00) and support from the European Union NextGenerationEU/PRTR. We express our gratitude to Carlos Domínguez for his assistance in the experimental setup and to Eneko Agirre for his valuable feedback and guidance.

## References

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] Amnon Catav and Roy Miara and Ilai Giloh and Nathan Cordeiro and Amir Ingber. RAG makes LLMs better and equal. 2024. URL https: //www.pinecone.io/blog/rag-study/

[3] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek. Seven failure points when engineering a retrieval augmented generation system. arXiv preprint arXiv:2401.05856, 2024.

[4] A. Berger, R. Caruana, D. Cohn, D. Freitag, and V. Mittal. Bridging the lexical chasm: statistical approaches to answer-finding. In Proceedings of the 23rd annual international ACM SIGIR conference on Research and development in information retrieval, pages 192-199, 2000.

[5] Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and Van Den Driessche, George Bm and Lespiau, Jean-Baptiste and Damoc, Bogdan and Clark, Aidan and others. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pages 2206-2240. PMLR, 2022.

[6] Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

[7] Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke. QuAC: Question Answering in Context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2174-2184, 2018 .

[8] P. Erbacher, L. Falissar, V. Guigue, and L. Soulier. Navigating uncertainty: Optimizing api dependency for hallucination reduction in closedbook question answering. arXiv preprint arXiv:2401.01780, 2024.

[9] T. Gao, X. Yao, and D. Chen. Simcse: Simple contrastive learning of sentence embeddings. In 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, pages 6894-6910. Association for Computational Linguistics (ACL), 2021.

[10] Gao, Yunfan and Xiong, Yun and Gao, Xinyu and Jia, Kangxiang and Pan, Jinliu and Bi, Yuxi and Dai, Yi and Sun, Jiawei and Wang, Haofen. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[11] Gautier, Izacard and Mathilde, Caron and Lucas, Hosseini and Sebastian, Riedel and Piotr, Bojanowski and Armand, Joulin and Edouard, Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022.

[12] Guu, Kelvin and Lee, Kenton and Tung, Zora and Pasupat, Panupong and Chang, Ming-Wei. REALM: retrieval-augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning. JMLR.org, 2020.

[13] Izacard, Gautier and Grave, Edouard. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In
EACL 2021-16th Conference of the European Chapter of the Association for Computational Linguistics, pages 874-880. Association for Computational Linguistics, 2021.

[14] Ji, Ziwei and Lee, Nayeon and Frieske, Rita and Yu, Tiezheng and Su, Dan and Xu, Yan and Ishii, Etsuko and Bang, Ye Jin and Madotto, Andrea and Fung, Pascale. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1-38, 2023.

[15] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. 1. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

[16] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[17] Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2020.

[18] Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453466, 2019 .

[19] Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and Küttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rocktäschel, Tim and others. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459$9474,2020$.

[20] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023.

[21] Lin, Stephanie and Hilton, Jacob and Evans, Owain. TruthfulQA: Measuring How Models Mimic Human Falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214-3252, 2022.

[22] Mallen, Alex Troy and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh. When Not to Trust Language Models: Investigating Effectiveness of Parametric and NonParametric Memories. In The 61st Annual Meeting Of The Association For Computational Linguistics, 2023.

[23] Nakano, Reiichiro and Hilton, Jacob and Balaji, Suchir and Wu, Jeff and Ouyang, Long and Kim, Christina and Hesse, Christopher and Jain, Shantanu and Kosaraju, Vineet and Saunders, William and others. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[24] Nguyen, Tri and Rosenberg, Mir and Song, Xia and Gao, Jianfeng and Tiwary, Saurabh and Majumder, Rangan and Deng, Li. Ms marco: A human-generated machine reading comprehension dataset. 2016.

[25] Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, 2016.

[26] Ram, Ori and Levine, Yoav and Dalmedigos, Itay and Muhlgay, Dor and Shashua, Amnon and Leyton-Brown, Kevin and Shoham, Yoav. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316-1331, 2023.

[27] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

[28] N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pages 3982-3992, 2019.

[29] S. Robertson, H. Zaragoza, et al. The probabilistic relevance framework: $\mathrm{Bm} 25$ and beyond. Foundations and Trends ${ }^{\circledR}$ in Information Retrieval, 3(4):333-389, 2009.

[30] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.

[31] Seonwoo, Yeon and Son, Juhee and Jin, Jiho and Lee, Sang-Woo and Kim, Ji-Hoon and Ha, Jung-Woo and Oh, Alice Haeyun. Two-Step Question Retrieval for Open-Domain QA. In 60th Annual Meeting of
the Association for Computational Linguistics, ACL 2022, pages 14871492. Association for Computational Linguistics, 2022.

[32] Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and Dubois, Yann and Li, Xuechen and Guestrin, Carlos and Liang, Percy and Hashimoto, Tatsunori B. Stanford alpaca: an instruction-following llama model (2023). URL https://github. com/tatsu-Lab/stanford_alpaca, 2023.

[33] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

[34] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023

[35] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023

[36] F. Zhu, W. Lei, C. Wang, J. Zheng, S. Poria, and T.-S. Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. arXiv preprint arXiv:2101.00774, 2021.


[^0]:    * Corresponding Author. Email: tlabruna@fbk.eu.

[^1]:    ${ }^{1}$ All resources are publicly available at https://github.com/Labruna/AdaptLLM.

</end of paper 1>


<paper 2>
# Incorporating External Knowledge and Goal Guidance for LLM-based Conversational Recommender Systems 

Chuang $\mathbf{L i}^{12}$, Yang Deng ${ }^{1}$, Hengchang $\mathbf{H u}^{1}$, Min-Yen Kan ${ }^{1}$, Haizhou $\mathbf{L i}^{13}$<br>${ }^{1}$ National University of Singapore<br>${ }^{2}$ NUS Graduate School for Integrative Sciences and Engineering<br>${ }^{3}$ Chinese University of Hong Kong, Shenzhen<br>\{lichuang, hengchanghu\}@u.nus.edu<br>\{ydeng, kanmy, haizhou.li\}@nus.edu.sg


#### Abstract

This paper aims to efficiently enable large language models (LLMs) to use external knowledge and goal guidance in conversational recommender system (CRS) tasks. Advanced LLMs (e.g., ChatGPT) are limited in domain-specific CRS tasks for 1) generating grounded responses with recommendationoriented knowledge, or 2) proactively leading the conversations through different dialogue goals. In this work, we first analyze those limitations through a comprehensive evaluation, showing the necessity of external knowledge and goal guidance which contribute significantly to the recommendation accuracy and language quality. In light of this finding, we propose a novel ChatCRS framework to decompose the complex CRS task into several subtasks through the implementation of 1 ) a knowledge retrieval agent using a tool-augmented approach to reason over external Knowledge Bases and 2) a goal-planning agent for dialogue goal prediction. Experimental results on two multi-goal CRS datasets reveal that ChatCRS sets new state-of-the-art benchmarks, improving language quality of informativeness by $17 \%$ and proactivity by $27 \%$, and achieving a tenfold enhancement in recommendation accuracy ${ }^{1}$.


## 1 Introduction

Conversational recommender system (CRS) integrates conversational and recommendation system (RS) technologies, naturally planning and proactively leading the conversations from nonrecommendation goals (e.g., "chitchat" or "question answering") to recommendation-related goals (e.g., "movie recommendation; Jannach et al., 2021; Liu et al., 2023b). Compared with traditional RS, CRS highlights the multi-round interactions between users and systems using natural language. Besides the recommendation task evaluated by the recommendation accuracy as in RS, CRS also[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_716205946e78faeb4304g-01.jpg?height=571&width=788&top_left_y=751&top_left_x=1062)

Figure 1: An example of CRS tasks with external knowledge and goal guidance. (Blue: CRS tasks; Red: External Knowledge and Goal Guidance)

focuses on multi-round interactions in response generation tasks including asking questions, responding to user utterances or balancing recommendation versus conversation (Li et al., 2023).

Large language models (LLMs; e.g., ChatGPT) that are significantly more proficient in response generation show great potential in CRS applications. However, current research concentrates on evaluating only their recommendation capability (Sanner et al., 2023; Dai et al., 2023). Even though LLMs demonstrate a competitive zero-shot recommendation proficiency, their recommendation performance primarily depends on content-based information (internal knowledge) and exhibits sensitivity towards demographic data (He et al., 2023; Sanner et al., 2023). Specifically, LLMs excel in domains with ample internal knowledge (e.g., English movies). However, in domains with scarce internal knowledge (e.g., Chinese movies ${ }^{2}$ ), we found through our empirical analysis (§ 3) that their recommendation performance notably dimin-[^1]ishes. Such limitation of LLM-based CRS motivates exploring solutions from prior CRS research to enhance domain coverage and task performance.

Prior work on CRS has employed general language models (LMs; e.g., DialoGPT) as the base architecture, but bridged the gap to domain-specific CRS tasks by incorporating external knowledge and goal guidance (Wang et al., 2021; Liu et al., 2023b). Inspired by this approach, we conduct an empirical analysis on the DuRecDial dataset (Liu et al., 2021) to understand how external inputs ${ }^{3}$ can efficiently adapt LLMs in the experimented domain and enhance their performance on both recommendation and response generation tasks.

Our analysis results (§ 3) reveal that despite their strong language abilities, LLMs exhibit notable limitations when directly applied to CRS tasks without external inputs in the Chinese movie domain. For example, lacking domain-specific knowledge ("Jimmy's Award") hinders the generation of pertinent responses, while the absence of explicit goals ("recommendation") leads to unproductive conversational turns (Figure 1). Identifying and mitigating such constraints is crucial for developing effective LLM-based CRS (Li et al., 2023).

Motivated by the empirical evidence that external inputs can significantly boost LLM performance on both CRS tasks, we propose a novel ChatCRS framework. It decomposes the overall CRS problem into sub-components handled by specialized agents for knowledge retrieval and goal planning, all managed by a core LLM-based conversational agent. This design enhances the framework's flexibility, allowing it to work with different LLM models without additional fine-tuning while capturing the benefits of external inputs (Figure 2b). Our contributions can be summarised as:

- We present the first comprehensive evaluation of LLMs on both CRS tasks, including response generation and recommendation, and underscore the challenges in LLM-based CRS.
- We propose the ChatCRS framework as the first knowledge-grounded and goal-directed LLMbased CRS using LLMs as conversational agents.
- Experimental findings validate the efficacy and efficiency of ChatCRS in both CRS tasks. Furthermore, our analysis elucidates how external inputs contribute to LLM-based CRS.[^2]


## 2 Related Work

Attribute-based/Conversational approaches in CRS. Existing research in CRS has been categorized into two approaches (Gao et al., 2021; Li et al., 2023): 1) attribute-based approaches, where the system and users exchange item attributes without conversation (Zhang et al., 2018; Lei et al., 2020), and 2) conversational approaches, where the system interacts users through natural language $(\mathrm{Li}$ et al., 2018; Deng et al., 2023; Wang et al., 2023a).

LLM-based CRS. LLMs have shown promise in CRS applications as 1) zero-shot conversational recommenders with item-based (Palma et al., 2023; Dai et al., 2023) or conversational inputs (He et al., 2023; Sanner et al., 2023; Wang et al., 2023b); 2) AI agents controlling pre-trained CRS or LMs for CRS tasks (Feng et al., 2023; Liu et al., 2023a; Huang et al., 2023); and 3) user simulators evaluating interactive CRS systems (Wang et al., 2023c; Zhang and Balog, 2020; Huang et al., 2024). However, there is a lack of prior work integrating external inputs to improve LLM-based CRS models.

Multi-agent and tool-augmented LLMs. LLMs, as conversational agents, can actively pursue specific goals through multi-agent task decomposition and tool augmentation (Wang et al., 2023d). This involves delegating subtasks to specialized agents and invoking external tools like knowledge retrieval, enhancing LLMs' reasoning abilities and knowledge coverage (Yao et al., 2023; Wei et al., 2023; Yang et al., 2023; Jiang et al., 2023).

In our work, we focus on the conversational approach, jointly evaluating CRS on both recommendation and response generation tasks (Wang et al., 2023a; Li et al., 2023; Deng et al., 2023). Unlike existing methods, ChatCRS uniquely combines goal planning and tool-augmented knowledge retrieval agents within a unified framework. This leverages LLMs' innate language and reasoning capabilities without requiring extensive fine-tuning.

## 3 Preliminary: Empirical Analysis

We consider the CRS scenario where a system system interacts with a user $u$. Each dialogue contains $T$ conversation turns with user and system utterances, denoted as $C=\left\{s_{j}^{\text {system }}, s_{j}^{u}\right\}_{j=1}^{T}$. The target function for CRS is expressed in two parts: given the dialogue history $C_{j}$ of the past $j^{\text {th }}$ turns, it generates 1) the recommendation of item $i$ and 2) the next system response $s_{j+1}^{\text {system }}$. In

![](https://cdn.mathpix.com/cropped/2024_06_04_716205946e78faeb4304g-03.jpg?height=503&width=1583&top_left_y=248&top_left_x=242)

Figure 2: a) Empirical analysis of LLMs in CRS tasks with DG, COT\& Oracle; b) System design of ChatCRS framework using LLMs as a conversational agent to control the goal planning and knowledge retrieval agents.

some methods, knowledge $K$ is given as an external input to facilitate both the recommendation and response generation tasks while dialogue goals $G$ only facilitate the response generation task due to the fixed "recommendation" goals in the recommendation task. Given the user's contextual history $C_{j}$, system generates recommendation results $i$ and system response $s_{j+1}^{\text {system }}$ in Eq. 1 .

$$
\begin{equation*}
y^{*}=\prod_{j=1}^{T} P_{\theta}\left(i, s_{j+1}^{\text {system }} \mid C_{j}, K, G\right) \tag{1}
\end{equation*}
$$

### 3.1 Empirical Analysis Approaches

Building on the advancements of LLMs over general LMs in language generation and reasoning, we explore their inherent response generation and recommendation capabilities, with and without external knowledge or goal guidance. Our analysis comprises three settings, as shown in Figure 2a:

- Direct Generation (DG). LLMs directly generate system responses and recommendations without any external inputs (Figure 5a).
- Chain-of-thought Generation (COT). LLMs internally reason their built-in knowledge and goalplanning scheme for both CRS tasks (Figure 5b).
- Oracular Generation (Oracle). LLMs leverage gold-standard external knowledge and dialogue goals to enhance performance in both CRS tasks, providing an upper bound (Figure 5c).

Additionally, we conduct an ablation study of different knowledge types on both CRS tasks by analyzing 1) factual knowledge, referring to general facts about entities and expressed as single triple (e.g., [Jiong-Star sign-Taurus]), and 2) item-based knowledge, related to recommended items and expressed as multiple triples (e.g., [Cecilia-Star in$<$ movie 1 , movie $2, \ldots$, movie $n>$ ]). Our primary

| $L L M$ | Task | NDCG@10/50 | MRR@10/50 |
| :---: | :---: | :---: | :---: |
|  | DG | $0.024 / 0.035$ | $0.018 / 0.020$ |
| ChatGPT | COT-K | $0.046 / 0.063$ | $0.040 / 0.043$ |
|  | Oracle-K | $\mathbf{0 . 6 1 7 / 0 . 6 2 4}$ | $\mathbf{0 . 6 1 3 / 0 . 6 1 4}$ |
|  | DG | $0.013 / 0.020$ | $0.010 / 0.010$ |
| LLaMA-7b | COT-K | $0.021 / 0.029$ | $0.018 / 0.020$ |
|  | Oracle-K | $\mathbf{0 . 3 8 6 / 0 . 4 2 2}$ | $\mathbf{0 . 3 6 6 / 0 . 3 7 0}$ |
|  | DG | $0.027 / 0.031$ | $0.024 / 0.024$ |
| LLaMA-13b | COT-K | $0.037 / 0.040$ | $0.035 / 0.036$ |
|  | Oracle-K | $\mathbf{0 . 7 2 4 / 0 . 7 3 4}$ | $\mathbf{0 . 6 9 8 / 0 . 6 9 9}$ |

Table 1: Empirical analysis for recommendation task in DuRecDial dataset ( $K$ : Knowledge; Red: Best result).

experimental approach utilizes in-context learning (ICL) on the DuRecDial dataset (Liu et al., 2021). Figure 5 provides an overview of the ICL prompts, with examples detailed in Appendix A. 1 and experiments detailed in $\S 5$. For response generation, we evaluate content preservation (bleu-n, $F 1$ ) and diversity (dist- $n$ ) with knowledge and goal prediction accuracy. For recommendation, we evaluate top-K ranking accuracy ( $N D C G @ k, M R R @ k$ ).

### 3.2 Empirical Analysis Findings

We summarize our three main findings given the results of the response generation and recommendation tasks shown in Tables 1 and 2.

Finding 1: The Necessity of External Inputs in LLM-based CRS. Integrating external inputs significantly enhances performance across all LLM-based CRS tasks (Oracle), underscoring the insufficiency of LLMs alone as effective CRS tools and highlighting the indispensable role of external inputs. Remarkably, the Oracle approach yields over a tenfold improvement in recommendation tasks with only external knowledge compared to DG and COT methods, as the dialogue goal is fixed as "recom-

| $L L M$ | Approach | $K / G$ | bleu 1 | bleu 2 | bleu | dist1 | $\operatorname{dist} 2$ | $F 1$ | $A c c_{G / K}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E | DG |  | 0.448 | 0.322 | 0.161 | 0.330 | 0.814 | 0.522 | - |
|  | COT | $\mathrm{G}$ | 0.397 | 0.294 | 0.155 | 0.294 | 0.779 | 0.499 | 0.587 |
|  |  | $\mathrm{K}$ | 0.467 | 0.323 | 0.156 | 0.396 | 0.836 | 0.474 | $\underline{0.095}$ |
|  | Oracle | G | 0.429 | 0.319 | 0.172 | 0.315 | 0.796 | 0.519 | - |
|  |  | $\mathrm{K}$ | 0.497 | 0.389 | 0.258 | 0.411 | 0.843 | 0.488 | - |
|  |  | BOTH | $\frac{1}{0.428}$ | 0.341 | $\frac{0.226}{0.26}$ | 0.307 | 0.784 | 0.525 | - |
| i | DG |  | 0.417 | 0.296 | 0.145 | 0.389 | 0.813 | 0.495 | - |
|  | COT | $\mathrm{G}$ | 0.418 | 0.293 | 0.142 | 0.417 | 0.827 | 0.484 | 0.215 |
|  |  | $\mathrm{K}$ | 0.333 | 0.238 | 0.112 | 0.320 | 0.762 | 0.455 | 0.026 |
|  | Oracle | $\mathrm{G}$ | 0.450 | 0.322 | 0.164 | 0.431 | 0.834 | 0.504 | - |
|  |  | $\mathrm{K}$ | 0.359 | 0.270 | 0.154 | $\overline{0.328}$ | 0.762 | 0.473 | - |
|  |  | BOTH | 0.425 | 0.320 | 0.187 | 0.412 | 0.807 | 0.492 | - |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_716205946e78faeb4304g-04.jpg?height=244&width=96&top_left_y=788&top_left_x=342) | DG |  | 0.418 | 0.303 | 0.153 | 0.312 | 0.786 | 0.507 | - |
|  | COT | $\mathrm{G}$ | 0.463 | 0.332 | 0.172 | 0.348 | 0.816 | 0.528 | 0.402 |
|  |  | $\mathrm{K}$ | 0.358 | 0.260 | 0.129 | 0.276 | 0.755 | 0.473 | 0.023 |
|  | Oracle | $\mathrm{G}$ | 0.494 | 0.361 | 0.197 | 0.373 | 0.825 | $\underline{0.543}$ | - |
|  |  | $\mathrm{K}$ | 0.379 | 0.296 | 0.188 | 0.278 | 0.754 | 0.495 | - |
|  |  | BOTH | 0.460 | 0.357 | 0.229 | 0.350 | 0.803 | 0.539 | - |

Table 2: Empirical analysis for response generation task in DuRecDial dataset ( $K / G$ : Knowledge or goal; $A c c_{G / K}$ : Accuracy of knowledge or goal predictions; Red: Best result for each model; Underline: Best results for all).

mendation" (Table 1). Although utilizing internal knowledge and goal guidance (COT) marginally benefits both tasks, we see in Table 2 for the response generation task that the low accuracy of internal predictions adversely affects performance.

Finding 2: Improved Internal Knowledge or Goal Planning Capability in Advanced LLMs. Table 2 reveals that the performance of Chain-of-Thought (COT) by a larger LLM (LLaMA-13b) is comparable to oracular performance of a smaller LLM (LLaMA-7b). This suggests that the intrinsic knowledge and goal-setting capabilities of more sophisticated LLMs can match or exceed the benefits derived from external inputs used by their less advanced counterparts. Nonetheless, such internal knowledge or goal planning schemes are still insufficient for CRS in domain-specific tasks while the integration of more accurate knowledge and goal guidance (Oracle) continues to enhance performance to state-of-the-art (SOTA) outcomes.

Finding 3: Both factual and item-based knowledge jointly improve LLM performance on domainspecific CRS tasks. As shown in Table 3, integrating both factual and item-based knowledge yields performance gains for LLMs on both response generation and recommendation tasks. Our analysis suggests that even though a certain type of knowledge may not directly benefit a CRS task (e.g., factual knowledge may not contain the target items for the recommendation task), it can still benefit LLMs

| Response Generation Task |  |  |
| :--- | :---: | :---: |
| Knowledge | bleu1/2/F1 | dist1/2 |
| Both Types | $\mathbf{0 . 4 9 7 / 0 . 3 8 9 / 0 . 4 8 8}$ | $\mathbf{0 . 4 1 1 / 0 . 8 4 3}$ |
| -w/o Factual | $0.407 / 0.296 / 0.456$ | $0.273 / 0.719$ |
| -w/o Item-based | $0.427 / 0.331 / 0.487$ | $0.277 / 0.733$ |
| Recommendation Task |  |  |
| Knowledge | NDCG@10/50 | MRR@10/50 |
| Both Types | $\mathbf{0 . 6 1 7 / 0 . 6 2 4}$ | $\mathbf{0 . 6 1 3 / 0 . 6 1 4}$ |
| -w/o Factual | $0.272 / 0.290$ | $0.264 / 0.267$ |
| -w/o Item-based | $0.376 / 0.389$ | $0.371 / 0.373$ |

Table 3: Ablation study for ChatGPT with different knowledge types in DuRecDial dataset.

by associating unknown entities with their internal knowledge, thereby adapting the universally pre-trained LLMs to task-specific domains more effectively. Consequently, we leverage both types of knowledge jointly in our ChatCRS framework.

## 4 ChatCRS

Our ChatCRS modelling framework has three components: 1) a knowledge retrieval agent, 2) a goal planning agent and 3) an LLM-based conversational agent (Figure 2b). Given a complex CRS task, an LLM-based conversational agent first decomposes it into subtasks managed by knowledge retrieval or goal-planning agents. The retrieved knowledge or predicted goal from each agent is incorporated into the ICL prompt to instruct LLMs to generate CRS responses or recommendations.

### 4.1 Knowledge Retrieval agent

Our analysis reveals that integrating both factual and item-based knowledge can significantly boost the performance of LLM-based CRS. However, knowledge-enhanced approaches for LLM-based CRS present unique challenges that have been relatively unexplored compared to prior training-based methods in CRS or retrieval-augmented (RA) methods in NLP (Zhang, 2023; Di Palma, 2023).

Training-based methods, which train LMs to memorize or interpret knowledge representations through techniques like graph propagation, have been widely adopted in prior CRS research (Wei et al., 2021; Zhang et al., 2023). However, such approaches are computationally infeasible for LLMs due to their input length constraints and training costs. RA methods, which first collect evidence and then generate responses, face two key limitations in CRS (Manzoor and Jannach, 2021; Gao et al., 2023). First, without a clear query formulation in CRS, RA methods can only approximate results rather than retrieve the exact relevant knowledge (Zhao et al., 2024; Barnett et al., 2024). Especially when multiple similar entries exist in the knowledge base (KB), precisely locating the accurate knowledge for CRS becomes challenging. Second, RA methods retrieve knowledge relevant only to the current dialogue turn, whereas CRS requires planning for potential knowledge needs in future turns, differing from knowledge-based QA systems (Mao et al., 2020; Jiang et al., 2023). For instance, when discussing a celebrity without a clear query (e.g., "I love Cecilia..."), the system should anticipate retrieving relevant factual knowledge (e.g., "birth date" or "star sign") or item-based knowledge (e.g., "acting movies") for subsequent response generation or recommendations, based on the user's likely interests.

To address this challenge, we employ a relationbased method which allows LLMs to flexibly plan and quickly retrieve relevant "entity-relationentity" knowledge triples $K$ by traversing along the relations $R$ of mentioned entities $E$ (Moon et al., 2019; Jiang et al., 2023). Firstly, entities for each utterance is directly provided by extracting entities in the knowledge bases from the dialogue utterance (Zou et al., 2022). Relations that are adjacent to entity $E$ from the $\mathrm{KB}$ are then extracted as candidate relations (denoted as $F 1$ ) and LLMs are instructed to plan the knowledge retrieval by selecting the most pertinent relation $R^{*}$ given the

![](https://cdn.mathpix.com/cropped/2024_06_04_716205946e78faeb4304g-05.jpg?height=391&width=761&top_left_y=247&top_left_x=1067)

Figure 3: Knowledge retrieval agent in ChatCRS.

dialogue history $C_{j}$. Knowledge triples $K^{*}$ can finally be acquired using entity $E$ and predicted relation $R^{*}$ (denoted as $F 2$ ). The process is formulated in Figure 3 and demonstrated with an example in Figure 7. Given the dialogue utterance "I love Cecilia..." and the extracted entity [Cecilia], the system first extracts all potential relations for [Cecilia], from which the LLM selects the most relevant relation, [Star in]. The knowledge retrieval agent then fetches the complete knowledge triple [Cecilia-Star in-<movie 1, movie 2, ..., movie $n>$ ].

When there are multiple entities in one utterance, we perform the knowledge retrieval one by one and in the scenario where there are multiple itembased knowledge triples, we randomly selected a maximum of 50 item-based knowledge due to the limitations of input token length. We implement $\mathrm{N}$-shot ICL to guide LLMs in choosing knowledge relations and we show the detailed ICL prompt and instruction with examples in Table 10 (§ A.2).

### 4.2 Goal Planning agent

Accurately predicting the dialogue goals is crucial for 1) proactive response generation and 2) balancing recommendations versus conversations in CRS. Utilizing goal annotations for each dialogue utterance from CRS datasets, we leverage an existing language model, adjusting it for goal generation by incorporating a Low-Rank Adapter (LoRA) approach (Hu et al., 2021; Dettmers et al., 2023). This method enables parameter-efficient fine-tuning by adjusting only the rank-decomposition matrices. For each dialogue history $C_{j}^{k}$ ( $j$-th turn in dialogue $k ; j \in T, k \in N$ ), the LoRA model is trained to generate the dialogue goal $G^{*}$ for the next utterance using the prompt of dialogue history, optimizing the loss function in $\mathrm{Eq} 2$ with $\theta$ representing the trainable parameters of LoRA. The detailed prompt and instructions are shown in Table 11 (§ A.3).

$$
\begin{equation*}
L_{g}=-\sum_{k}^{N} \sum_{j}^{T} \log P_{\theta}\left(G^{*} \mid C_{j}^{k}\right) \tag{2}
\end{equation*}
$$

| Model | N-shot | DuRecDial |  |  |  | TG-Redial |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | bleu 1 | bleu 2 | dist 2 | $F 1$ | bleu1 | bleu 2 | dist2 | $F 1$ |
| $\mathrm{MGCG}$ | Full | 0.362 | 0.252 | 0.081 | 0.420 | NA | NA | NA | NA |
| MGCG-G | Full | 0.382 | 0.274 | 0.214 | 0.435 | NA | NA | NA | NA |
| TPNet | Full | 0.308 | 0.217 | 0.093 | 0.363 | NA | NA | NA | NA |
| UniMIND* | Full | 0.418 | 0.328 | 0.086 | 0.484 | 0.291 | 0.070 | 0.200 | 0.328 |
| ChatGPT | 3 | 0.448 | 0.322 | $\mathbf{0 . 8 1 4}$ | 0.522 | 0.262 | 0.126 | 0.987 | 0.266 |
| LLaMA | 3 | 0.418 | 0.303 | 0.786 | 0.507 | 0.205 | 0.096 | 0.970 | 0.247 |
| ChatCRS | 3 | 0.460 | 0.358 | 0.803 | 0.540 | 0.300 | $\mathbf{0 . 1 8 0}$ | 0.987 | 0.317 |

Table 4: Results of response generation task on DuRecDial and TG-Redial datasets. (UniMIND*: Results from the ablation study in the original UniMIND paper.)

| Model | N-shot | DuRecDial |  | TG-Redial |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $N D C G @ 10 / 50$ | $M R R @ 10 / 50$ | $N D C G @ 10 / 50$ | $M R R @ 10 / 50$ |
| SASRec | Full | $0.369 / 0.413$ | $0.307 / 0.317$ | $0.009 / 0.018$ | $0.005 / 0.007$ |
| UniMIND | Full | $0.599 / 0.610$ | $0.592 / 0.594$ | $0.031 / 0.050$ | $0.024 / 0.028$ |
| ChatGPT | 3 | $0.024 / 0.035$ | $0.018 / 0.020$ | $0.001 / 0.003$ | $0.005 / 0.005$ |
| LLaMA | 3 | $0.027 / 0.031$ | $0.024 / 0.024$ | $0.001 / 0.006$ | $0.003 / 0.005$ |
| ChatCRS | 3 | $0.549 / 0.553$ | $0.543 / 0.543$ | $0.031 / 0.033$ | $0.082 / 0.083$ |

Table 5: Results of recommendation task on DuRecDial and TG-Redial datasets.

### 4.3 LLM-based Conversational Agent

In ChatCRS, the knowledge retrieval and goalplanning agents serve as essential tools for CRS tasks, while LLMs function as tool-augmented conversational agents that utilize these tools to accomplish primary CRS objectives. Upon receiving a new dialogue history $C_{j}$, the LLM-based conversational agent employs these tools to determine the dialogue goal $G^{*}$ and relevant knowledge $K^{*}$, which then instruct the generation of either a system response $s_{j+1}^{\text {system }}$ or an item recommendation $i$ through prompting scheme, as formulated in Eq 3. The detailed ICL prompt can be found in $\S$ A.1.

$$
\begin{equation*}
i, s_{j+1}^{\text {system }}=L L M\left(C_{j}, K^{*}, G^{*}\right) \tag{3}
\end{equation*}
$$

## 5 Experiments

### 5.1 Experimental Setups

Datasets. We conduct the experiments on two multi-goal Chinese CRS benchmark datasets a) DuRecDial (Liu et al., 2021) in English and Chinese, and b) TG-ReDial (Zhou et al., 2020) in Chinese (statistics in Table 12). Both datasets are annotated for goal guidance, while only DuRecDial contains knowledge annotation and an external KBCNpedia (Zhou et al., 2022) is used for TG-Redial. Baselines. We compare our model with ChatGPT ${ }^{4}$ and LLaMA-7b/13b (Touvron et al., 2023) in few- shot settings. We also compare fully-trained UniMIND (Deng et al., 2023), MGCG-G(Liu et al., 2023b), TPNet(Wang et al., 2023a), MGCG (Liu et al., 2020) and SASRec (Kang and McAuley, 2018), which are previous SOTA CRS and RS models and we summarise each baseline in $\S ~ A .6$.

Automatic Evaluation. For response generation evaluation, we adopt $B L E U, F 1$ for content preservation and Dist for language diversity. For recommendation evaluation, we adopt $N D C G @ k$ and $M R R @ K$ to evaluate top $\mathrm{K}$ ranking accuracy. For the knowledge retrieval agent, we adopt Accuracy $(A c c)$, Precision $(P)$, Recall $(R)$ and $F 1$ to evaluate the accuracy of relation selection (§ A.2). Human Evaluation. For human evaluation, we randomly sample 100 dialogues from DuRecDial, comparing the responses produced by UniMIND, ChatGPT, LLaMA-13b and ChatCRS. Three annotators are asked to score each generated response with $\{0$ : poor, 1: ok, 2 : good $\}$ in terms of a) general language quality in (Flu)ency and (Coh)erence, and b) CRS-specific language qualities of (Info)rmativeness and (Pro)activity. Details of the process and criterion are discussed in $\S$ A.4. Implementation Details. For both the CRS tasks in Empirical Analysis, we adopt N-shot ICL prompt settings on ChatGPT and LLaMA* (Dong et al., 2022), where $N$ examples from the training data are added to the ICL prompt. In modelling framework, for the goal planning agent, we adopt[^3]

| Model | General |  | CRS-specific |  | Avg. |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  | Flu | Coh | Info | Pro |  |
| UniMIND | 1.87 | 1.69 | 1.49 | 1.32 | 1.60 |
|  |  |  |  |  |  |
| LLaMA-13b | 1.94 | 1.68 | 1.21 | 1.33 | 1.49 |
| Cha | 1.99 | 1.8 | 17 | 16 | 1 |
| $-w / o 1$ | 2.00 | 1.87 | $1.49 \downarrow$ | 1.62 | 1.75 |
| $-w / o G^{*}$ | 1.99 | 1.85 | 1.72 | $1.55 \downarrow$ | 1.78 |

Table 6: Human evaluation and ChatCRS ablations for language qualities of (Flu)ency, (Coh)erence, (Info)rmativeness and (Pro)activity on DuRecDial ( $K^{*} / G^{*}$ : Knowledge retrieval or goal-planning agent).

| Model | Knowledge |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | N-shot | Acc | P | R | $\mathrm{F} 1$ |  |
| TPNet | Full | NA | NA | NA | 0.402 |  |
| MGCG-G | Full | NA | 0.460 | 0.478 | 0.450 |  |
| ChatGPT | 3 | 0.095 | 0.031 | 0.139 | 0.015 |  |
| LLaMA-13b | 3 | 0.023 | 0.001 | 0.001 | 0.001 |  |
| ChatCRS | 3 | $\mathbf{0 . 5 6 0}$ | $\mathbf{0 . 5 8 3}$ | $\mathbf{0 . 5 9 4}$ | $\mathbf{0 . 5 5 3}$ |  |

Table 7: Results for knowledge retrieval on DuRecDial.

QLora as a parameter-efficient way to fine-tune LLaMA-7b (Dettmers et al., 2023). For the knowledge retrieval agent and LLM-based conversational agent, we adopt the same $\mathrm{N}$-shot ICL approach on ChatGPT and LLaMA* (Jiang et al., 2023). Detailed experimental setups are discussed in § A.6.

### 5.2 Experimental Results

ChatCRS significantly improves LLM-based conversational systems for CRS tasks, outperforming SOTA baselines in response generation in both datasets, enhancing content preservation and language diversity (Table 4). ChatCRS sets new SOTA benchmarks on both datasets using 3-shot ICL prompts incorporating external inputs. In recommendation tasks (Table 5), LLM-based approaches lag behind full-data trained baselines due to insufficient in-domain knowledge. Remarkably, ChatCRS, by harnessing external knowledge, achieves a tenfold increase in recommendation accuracy over existing LLM baselines on both datasets with ICL, without full-data fine-tuning.

Human evaluation highlights ChatCRS's enhancement in CRS-specific language quality. Table 6 shows the human evaluation and ablation results. ChatCRS outperforms baseline models in both general and CRS-specific language qualities. While all LLM-based approaches uniformly exceed the general LM baseline (UniMIND) in general

![](https://cdn.mathpix.com/cropped/2024_06_04_716205946e78faeb4304g-07.jpg?height=483&width=745&top_left_y=247&top_left_x=1067)

Figure 4: Knowledge ratio for each goal type on DuRecDial. (X-axis: Knowledge Ratio ; Y-axis: Goal type)

language quality, ChatCRS notably enhances coherence through its goal guidance feature, enabling response generation more aligned with the dialogue goal. Significant enhancements in CRS-specific language quality, particularly in informativeness and proactivity, underscore the value of integrating external knowledge and goals. Ablation studies, removing either knowledge retrieval or goal planning agent, demonstrate a decline in scores for informativeness and proactivity respectively, confirming the efficacy of both external inputs for CRS-specific language quality.

### 5.3 Detailed Discussion

CRS datasets typically contain a huge volume of knowledge. By analyzing dialogues from the DuRecDial datasets, categorized by goal types, we calculated a "Knowledge Ratio" dividing the number of utterances with annotated knowledge $N_{K, G}$ by total number of utterances $N_{G}$ in each goal type (Eq 4) to measure the necessity of relevant knowledge in CRS task completion. Our analysis, depicted in Figure 4, shows that recommendation tasks rank highly in terms of knowledge necessity, with "POI recommendation" dialogues requiring pertinent knowledge in $75 \%$ of cases.

$$
\begin{equation*}
\text { Knowledge Ratio }(\mathrm{KR})_{G}=\frac{N_{K, G}}{N_{G}} \tag{4}
\end{equation*}
$$

Internal knowledge of LLMs is insufficient for handling domain-specific CRS tasks, which is reaffirmed by the analysis of knowledge retrieval accuracy in Table 7. Contrasting with traditional RS which relies on user data for collaborative recommendations, CRS mainly depends on context/content-based recommendation (He et al., 2023). This shift highlights the limitations of

| Case Study 1: Factual Knowledge QA | Case Study 2: Proactive Response Generation |
| :---: | :---: |
| Dialogue history (Dial_ID 39-1): <br> : Do you know Jiong He's zodiac <br> sign? | Dialogue history (Dial_ID 89-14): <br> : I like Cecilia Cheung very much. Her acting is very good. <br> [. <br> : Yeah, have you seen Cecilia Cheung's One Night in Mongkok? <br> : I've seen it. I don't want to see it again. |
| System Responses: <br> $[\mathcal{J}]$ UniMIND: Of course, Taurus <br> $[Х]$ ChatGPT: Jiong He's zodiac sign <br> is Cancer. <br> $[\times]$ LLaMA: Jiong He's zodiac sign is <br> Aries. <br> $[\checkmark]$ ChatCRS: Yes, I do! Jiong He's <br> zodiac sign is Taurus. <br> [Retrieved Knowledge: <br> ('Jiong He', 'zodiac sign ', 'Taurus')] | System Responses: <br> $[\mathcal{\checkmark}]$ UniMIND: How about <Left Right...Destiny>, a very touching movie. <br> $[\checkmark]$ ChatGPT: How about trying her other film, <Lost in Time>, a touching <br> romance that showcases her acting skills? <br> $[\times]$ LLaMA: Maybe you can try it again, you may find a different feeling. <br> $[\mathcal{J}$ ChatCRS: If you haven't seen it, you should watch <Left <br> Right...Destiny>, it's Cecilia Cheung's new movie this year. She played the <br> role of a taxi driver, very cool! Have you seen it? <br> $[$ Predicted Goal: Movie Recommendation] |

Table 8: Case study for ChatCRS with baselines.

LLMs in harnessing internal knowledge. ChatCRS overcomes these limitations by interfacing LLMs to plan and reason over external KBs through entities and relations. Therefore, it largely improves the recommendation accuracy, outperforming the training-based approach using full data. Given the limitations in LLM-based CRS tasks, (Zhang, 2023; Di Palma, 2023), we anticipate future studies to further explore such approaches in CRS.

Factual knowledge guides the response generation process, mitigating the risks of generating implausible or inconsistent responses. The "Asking questions" goal type which has the highest knowledge ratio, demonstrates the advantage of leveraging external knowledge in answering factual questions like "the zodiac sign of an Asian celebrity" (Table 8). Standard LLMs produce responses with fabricated content, but ChatCRS accurately retrieves and integrates external knowledge, ensuring factual and informative responses.

Goal guidance contributes more to the linguistic quality of CRS by managing the dialogue flow. We examine the goal planning proficiency of ChatCRS by showcasing the results of goal predictions of the top 5 goal types in each dataset (Figure 6). DuRecDial dataset shows better balances among recommendation and non-recommendation goals, which exactly aligns with the real-world scenarios (Hayati et al., 2020). However, the TG-Redial dataset contains more recommendationrelated goals and multi-goal utterances, making the goal predictions more challenging. The detailed goal planning accuracy is discussed in $\S ~ A .5$.

Dialogue goals guide LLMs towards a proactive conversational recommender. For a clearer understanding, we present a scenario in Table 8 where a CRS seamlessly transitions between "asking questions" and "movie recommendation", illustrating how accurate goal direction boosts interaction relevance and efficacy. Specifically, if a recommendation does not succeed, ChatCRS will adeptly pose further questions to refine subsequent recommendation responses while LLMs may keep outputting wrong recommendations, creating unproductive dialogue turns. This further emphasizes the challenges of conversational approaches in CRS, where the system needs to proactively lead the dialogue from non-recommendation goals to approach the users' interests for certain items or responses (Liu et al., 2023b), and underscores the goal guidance in fostering proactive engagement in CRS.

## 6 Conclusion

This paper conducts an empirical investigation into the LLM-based CRS for domain-specific applications in the Chinese movie domain, emphasizing the insufficiency of LLMs in domain-specific CRS tasks and the necessity of integrating external knowledge and goal guidance. We introduce ChatCRS, a novel framework that employs a unified agent-based approach to more effectively incorporate these external inputs. Our experimental findings highlight improvements over existing benchmarks, corroborated by both automatic and human evaluation. ChatCRS marks a pivotal advancement in CRS research, fostering a paradigm where complex problems are decomposed into subtasks managed by agents, which maximizes the inherent capabilities of LLMs and their domainspecific adaptability in CRS applications.

## Limitations

This research explores the application of few-shot learning and parameter-efficient techniques with large language models (LLMs) for generating responses and making recommendations, circumventing the need for the extensive fine-tuning these models usually require. Due to budget and computational constraints, our study is limited to incontext learning with economically viable, smallerscale closed-source LLMs like ChatGPT, and opensource models such as LLaMA-7b and -13b.

A significant challenge encountered in this study is the scarcity of datasets with adequate annotations for knowledge and goal-oriented guidance for each dialogue turn. This limitation hampers the development of conversational models capable of effectively understanding and navigating dialogue. It is anticipated that future datasets will overcome this shortfall by providing detailed annotations, thereby greatly improving conversational models' ability to comprehend and steer conversations.

## Ethic Concerns

The ethical considerations for our study involving human evaluation (§ 5.1) have been addressed through the attainment of an IRB Exemption for the evaluation components involving human subjects. The datasets utilized in our research are accessible to the public (Liu et al., 2021; Zhou et al., 2020), and the methodology employed for annotation adheres to a double-blind procedure (§ 5.1). Additionally, annotators receive compensation at a rate of $\$ 15$ per hour, which is reflective of the actual hours worked.

## References

Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdelrazek. 2024 Seven failure points when engineering a retrieval augmented generation system.

Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering chatgpt's capabilities in recommender systems. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys '23, page 1126-1132, New York, NY, USA. Association for Computing Machinery.

Yang Deng, Wenxuan Zhang, Weiwen Xu, Wenqiang Lei, Tat-Seng Chua, and Wai Lam. 2023. A unified multi-task learning framework for multi-goal conversational recommender systems. ACM Trans. Inf. Syst., 41(3).
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Dario Di Palma. 2023. Retrieval-augmented recommender system: Enhancing recommender systems with large language models. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys '23, page 1369-1373, New York, NY, USA. Association for Computing Machinery.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234.

Yue Feng, Shuchang Liu, Zhenghai Xue, Qingpeng Cai, Lantao Hu, Peng Jiang, Kun Gai, and Fei Sun. 2023 A large language model enhanced conversational recommender system.

Chongming Gao, Wenqiang Lei, Xiangnan He, Maarten de Rijke, and Tat-Seng Chua. 2021. Advances and challenges in conversational recommender systems: A survey. AI Open, 2:100-126.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.

Shirley Anugrah Hayati, Dongyeop Kang, Qingxiaoyang Zhu, Weiyan Shi, and Zhou Yu. 2020. Inspired: Toward sociable recommendation dialog systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8142-8152, Online. Association for Computational Linguistics.

Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. 2023. Large language models as zero-shot conversational recommenders. arXiv preprint arXiv:2308.10053.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models.

Chen Huang, Peixin Qin, Yang Deng, Wenqiang Lei, Jiancheng Lv, and Tat-Seng Chua. 2024. Conceptan evaluation protocol on conversation recommender systems with system-and user-centric factors. arXiv preprint arXiv:2404.03304.

Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. 2023. Recommender ai agent: Integrating large language models for interactive recommendations. arXiv preprint arXiv:2308.16505.

Dietmar Jannach, Ahtsham Manzoor, Wanling Cai, and Li Chen. 2021. A survey on conversational recommender systems. ACM Comput. Surv., 54(5).

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen. 2023. StructGPT: A general framework for large language model to reason over structured data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9237-9251, Singapore. Association for Computational Linguistics.

Wang-Cheng Kang and Julian McAuley. 2018. Selfattentive sequential recommendation.

Wenqiang Lei, Xiangnan He, Yisong Miao, Qingyun Wu, Richang Hong, Min-Yen Kan, and Tat-Seng Chua. 2020. Estimation-action-reflection: Towards deep interaction between conversational and recommender systems. In Proceedings of the 13th International Conference on Web Search and Data Mining, WSDM '20, page 304-312, New York, NY, USA. Association for Computing Machinery.

Chuang Li, Hengchang Hu, Yan Zhang, Min-Yen Kan, and Haizhou Li. 2023. A conversation is worth a thousand recommendations: A survey of holistic conversational recommender systems. In KaRS Workshop at ACM RecSys '23, Singapore.

Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018. Towards deep conversational recommendations. In Advances in Neural Information Processing Systems 31 (NIPS 2018).

Yuanxing Liu, Weinan Zhang, Yifan Chen, Yuchi Zhang, Haopeng Bai, Fan Feng, Hengbin Cui, Yongbin Li, and Wanxiang Che. 2023a. Conversational recommender system and large language model are made for each other in E-commerce pre-sales dialogue. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9587-9605, Singapore. Association for Computational Linguistics.

Zeming Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, and Wanxiang Che. 2021. DuRecDial 2.0: A bilingual parallel corpus for conversational recommendation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4335-4347, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zeming Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, Wanxiang Che, and Ting Liu. 2020. Towards conversational recommendation over multi-type dialogs. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 10361049. Association for Computational Linguistics.
Zeming Liu, Ding Zhou, Hao Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, Wanxiang Che, Ting Liu, and Hui Xiong. 2023b. Graph-grounded goal planning for conversational recommendation. IEEE Transactions on Knowledge and Data Engineering, 35(5):4923-4939.

Ahtsham Manzoor and Dietmar Jannach. 2021 Generation-based vs retrieval-based conversational recommendation: A user-centric comparison. In Proceedings of the 15th ACM Conference on Recommender Systems, RecSys '21, page 515-520, New York, NY, USA. Association for Computing Machinery.

Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. 2020. Generation-augmented retrieval for open-domain question answering. arXiv preprint arXiv:2009.08553.

Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. 2019. OpenDialKG: Explainable conversational reasoning with attention-based walks over knowledge graphs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 845-854, Florence, Italy. Association for Computational Linguistics.

Dario Di Palma, Giovanni Maria Biancofiore, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia, and Eugenio Di Sciascio. 2023. Evaluating chatgpt as a recommender system: A rigorous approach.

Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, and Lucas Dixon. 2023. Large language models are competitive near cold-start recommenders for language- and item-based preferences. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys '23, page 890-896, New York, NY, USA. Association for Computing Machinery.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models.

Jian Wang, Dongding Lin, and Wenjie Li. 2023a. A target-driven planning approach for goal-directed dialog systems. IEEE Transactions on Neural Networks and Learning Systems.

Lingzhi Wang, Huang Hu, Lei Sha, Can Xu, Kam-Fai Wong, and Daxin Jiang. 2021. Finetuning largescale pre-trained language models for conversational recommendation with knowledge graph. CoRR, $\mathrm{abs} / 2110.07477$.

Xiaolei Wang, Xinyu Tang, Xin Zhao, Jingyuan Wang, and Ji-Rong Wen. 2023b. Rethinking the evaluation for conversational recommendation in the era of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10052-10065, Singapore. Association for Computational Linguistics.

Xiaolei Wang, Xinyu Tang, Xin Zhao, Jingyuan Wang, and Ji-Rong Wen. 2023c. Rethinking the evaluation for conversational recommendation in the era of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10052-10065, Singapore. Association for Computational Linguistics.

Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, and Yingzhen Yang. 2023d. Recmind: Large language model powered agent for recommendation.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.

Xiaokai Wei, Shen Wang, Dejiao Zhang, Parminder Bhatia, and Andrew O. Arnold. 2021. Knowledge enhanced pretrained language models: A compreshensive survey. CoRR, abs/2110.08455.

Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. 2023. Gpt4tools: Teaching large language model to use tools via self-instruction.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models.

Gangyi Zhang. 2023. User-centric conversational recommendation: Adapting the need of user with large language models. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys '23, page 1349-1354, New York, NY, USA. Association for Computing Machinery.

Shuo Zhang and Krisztian Balog. 2020. Evaluating conversational recommender systems via user simulation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& amp; Data Mining, KDD '20, page 1512-1520, New York, NY, USA. Association for Computing Machinery.
Xiaoyu Zhang, Xin Xin, Dongdong Li, Wenxuan Liu, Pengjie Ren, Zhumin Chen, Jun Ma, and Zhaochun Ren. 2023. Variational reasoning over incomplete knowledge graphs for conversational recommendation. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, pages 231-239.

Yongfeng Zhang, Xu Chen, Qingyao Ai, Liu Yang, and W Bruce Croft. 2018. Towards conversational search and recommendation: System ask, user respond. In Proceedings of the 27th acm international conference on information and knowledge management, pages $177-186$.

Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, and Bin Cui. 2024. Retrievalaugmented generation for ai-generated content: A survey.

Kun Zhou, Yuanhang Zhou, Wayne Xin Zhao, Xiaoke Wang, and Ji-Rong Wen. 2020. Towards topic-guided conversational recommender system.

Yuanhang Zhou, Kun Zhou, Wayne Xin Zhao, Cheng Wang, Peng Jiang, and He Hu. 2022. C $^{2}$-crs: Coarseto-fine contrastive learning for conversational recommender system. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, pages 1488-1496.

Jie Zou, Evangelos Kanoulas, Pengjie Ren, Zhaochun Ren, Aixin Sun, and Cheng Long. 2022. Improving conversational recommender systems via transformer-based sequential modelling. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2319-2324. ACM.
</end of paper 2>


<paper 3>
# Large Language Models for Information Retrieval: A Survey 

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng,<br>Haonan Chen, Zhicheng Dou, and Ji-Rong Wen


#### Abstract

As a primary means of information acquisition, information retrieval (IR) systems, such as search engines, have integrated themselves into our daily lives. These systems also serve as components of dialogue, question-answering, and recommender systems. The trajectory of IR has evolved dynamically from its origins in term-based methods to its integration with advanced neural models. While the neural models excel at capturing complex contextual signals and semantic nuances, thereby reshaping the IR landscape, they still face challenges such as data scarcity, interpretability, and the generation of contextually plausible yet potentially inaccurate responses. This evolution requires a combination of both traditional methods (such as term-based sparse retrieval methods with rapid response) and modern neural architectures (such as language models with powerful language understanding capacity). Meanwhile, the emergence of large language models (LLMs), typified by ChatGPT and GPT-4, has revolutionized natural language processing due to their remarkable language understanding, generation, generalization, and reasoning abilities. Consequently, recent research has sought to leverage LLMs to improve IR systems. Given the rapid evolution of this research trajectory, it is necessary to consolidate existing methodologies and provide nuanced insights through a comprehensive overview. In this survey, we delve into the confluence of LLMs and IR systems, including crucial aspects such as query rewriters, retrievers, rerankers, and readers. Additionally, we explore promising directions, such as search agents, within this expanding field.


Index Terms-Large Language Models; Information Retrieval; Query Rewrite; Rerank; Reader; Fine-tuning; Prompting

## 1 INTRODUCTION

INFORMATION access is one of the fundamental daily needs of human beings. To fulfill the need for rapid acquisition of desired information, various information retrieval (IR) systems have been developed [1-4]. Prominent examples include search engines such as Google, Bing, and Baidu, which serve as IR systems on the Internet, adept at retrieving relevant web pages in response to user queries, and provide convenient and efficient access to information on the Internet. It is worth noting that IR extends beyond web page retrieval. In dialogue systems (chatbots) [1, 58], such as Microsoft Xiaoice [2], Apple Siri, ${ }^{1}$ and Google Assistant, ${ }^{2}$ IR systems play a crucial role in retrieving appropriate responses to user input utterances, thereby producing natural and fluent human-machine conversations. Similarly, in question-answering systems [3, 9], IR systems are employed to select relevant clues essential for addressing user questions effectively. In image search engines [4], IR systems excel at returning images that align with user input queries. Given the exponential growth of information, research and industry have become increasingly interested in the development of effective IR systems.

The core function of an IR system is retrieval, which aims to determine the relevance between a user-issued query and the content to be retrieved, including various types of information such as texts, images, music, and more. For the scope of this survey, we concentrate solely on review-[^0]

ing those text retrieval systems, in which query-document relevance is commonly measured by their matching score. ${ }^{3}$ Given that IR systems operate on extensive repositories, the efficiency of retrieval algorithms becomes of paramount importance. To improve the user experience, the retrieval performance is enhanced from both the upstream (query reformulation) and downstream (reranking and reading) perspectives. As an upstream technique, query reformulation is designed to refine user queries so that they are more effective at retrieving relevant documents [10, 11]. With the recent surge in the popularity of conversational search, this technique has received increasing attention. On the downstream side, reranking approaches are developed to further adjust the document ranking [12-14]. In contrast to the retrieval stage, reranking is performed only on a limited set of relevant documents, already retrieved by the retriever. Under this circumstance, the emphasis is placed on achieving higher performance rather than keeping higher efficiency, allowing for the application of more complex approaches in the reranking process. Additionally, reranking can accommodate other specific requirements, such as personalization [15-18] and diversification [19-22]. Following the retrieval and reranking stages, a reading component is incorporated to summarize the retrieved documents and deliver a concise document to users [23, 24]. While traditional IR systems typically require users to gather and organize relevant information themselves; however, the reading component is an integral part of new IR systems such as New

3. The term "document" will henceforth refer to any text-based content subject to retrieve, including both long articles and short passages.

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-02.jpg?height=523&width=1440&top_left_y=145&top_left_x=337)

Fig. 1. Overview of existing studies that apply LLMs into IR. (1) LLMs can be used to enhance traditional IR components, such as query rewriter, retriever, reranker, and reader. (2) LLMs can also be used as search agents to perform multiple IR tasks.

Bing, ${ }^{4}$ streamlining users' browsing experience and saving valuable time.

The trajectory of IR has traversed a dynamic evolution, transitioning from its origins in term-based methods to the integration of neural models. Initially, IR was anchored in term-based methods [25] and Boolean logic, focusing on keyword matching for document retrieval. The paradigm gradually shifted with the introduction of vector space models [26], unlocking the potential to capture nuanced semantic relationships between terms. This progression continued with statistical language models [27, 28], refining relevance estimation through contextual and probabilistic considerations. The influential BM25 algorithm [29] played an important role during this phase, revolutionizing relevance ranking by accounting for term frequency and document length variations. The most recent chapter in IR's journey is marked by the ascendancy of neural models [3, 3032]. These models excel at capturing intricate contextual cues and semantic nuances, reshaping the landscape of IR. However, these neural models still face challenges such as data scarcity, interpretability, and the potential generation of plausible yet inaccurate responses. Thus, the evolution of IR continues to be a journey of balancing traditional strengths (such as the BM25 algorithm's high efficiency) with the remarkable capability (such as semantic understanding) brought about by modern neural architectures.

Large language models (LLMs) have recently emerged as transformative forces across various research fields, such as natural language processing (NLP) [33-35], recommender systems [36-39], finance [40], and even molecule discovery [41]. These cutting-edge LLMs are primarily based on the Transformer architecture and undergo extensive pretraining on diverse textual sources, including web pages, research articles, books, and codes. As their scale continues to expand (including both model size and data volume), LLMs have demonstrated remarkable advances in their capabilities. On the one hand, LLMs have exhibited unprecedented proficiency in language understanding and generation, resulting in responses that are more human-like and better align with human intentions. On the other hand, the larger LLMs have shown impressive emergent abilities[^1]

when dealing with complex tasks [42], such as generalization and reasoning skills. Notably, LLMs can effectively apply their learned knowledge and reasoning abilities to tackle new tasks with just a few task-specific demonstrations or appropriate instructions [43, 44]. Furthermore, advanced techniques, such as in-context learning, have significantly enhanced the generalization performance of LLMs without requiring fine-tuning on specific downstream tasks [34]. This breakthrough is particularly valuable, as it reduces the need for extensive fine-tuning while attaining remarkable task performance. Powered by prompting strategies such as chain-of-thought, LLMs can generate outputs with step-bystep reasoning, navigating complex decision-making processes [45]. Leveraging the impressive power of LLMs can undoubtedly improve the performance of IR systems. By incorporating these sophisticated language models, IR systems can provide users with more accurate responses, ultimately reshaping the landscape of information access and retrieval.

Initial efforts have been made to utilize the potential of LLMs in the development of novel IR systems. Notably, in terms of practical applications, New Bing is designed to improve the users' experience of using search engines by extracting information from disparate web pages and condensing it into concise summaries that serve as responses to user-generated queries. In the research community, LLMs have proven useful within specific modules of IR systems (such as retrievers), thereby enhancing the overall performance of these systems. Due to the rapid evolution of LLMenhanced IR systems, it is essential to comprehensively review their most recent advancements and challenges.

Our survey provides an insightful exploration of the intersection between LLMs and IR systems, covering key perspectives such as query rewriters, retrievers, rerankers, and readers (as shown in Figure 1). ${ }^{5}$ We also include some recent studies that leverage LLMs as search agents to perform various IR tasks. This analysis enhances our understanding of LLMs' potential and limitations in advancing the IR field.

5. As yet, there has not been a formal definition for LLMs. In this paper, we mainly focus on models with more than 1B parameters. We also notice that some methods do not rely on such strictly defined LLMs, but due to their representativeness, we still include an introduction to them in this survey.

For this survey, we create a Github repository by collecting the relevant papers and resources about LLM4IR. ${ }^{6}$ We will continue to update the repository with newer papers. This survey will also be periodically updated according to the development of this area. We notice that there are several surveys for PLMs, LLMs, and their applications (e.g., AIGC or recommender systems) [46-52]. Among these, we highly recommend the survey of LLMs [52], which provides a systematic and comprehensive reference to many important aspects of LLMs. Compared with them, we focus on the techniques and methods for developing and applying LLMs for IR systems. In addition, we notice a perspective paper discussing the opportunity of IR when meeting LLMs [53]. It would be an excellent supplement to this survey regarding future directions.

The remaining part of this survey is organized as follows: Section 2 introduces the background for IR and LLMs. Section 3, 4, 5, 6 respectively review recent progress from the four perspectives of query rewriter, retriever, reranker, and reader, which are four key components of an IR system. Then, Section 8 discusses some potential directions in future research. Finally, we conclude the survey in Section 9 by summarizing the major findings.

## 2 BACKGROUND

### 2.1 Information Retrieval

Information retrieval (IR), as an essential branch of computer science, aims to efficiently retrieve information relevant to user queries from a large repository. Generally, users interact with the system by submitting their queries in textual form. Subsequently, IR systems undertake the task of matching and ranking these user-supplied queries against an indexed database, thereby facilitating the retrieval of the most pertinent results.

The field of IR has witnessed significant advancement with the emergence of various models over time. One such early model is the Boolean model, which employs Boolean logic operators to combine query terms and retrieve documents that satisfy specific conditions [25]. Based on the "bag-of-words" assumption, the vector space model [26] represents documents and queries as vectors in term-based space. Relevance estimation is then performed by assessing the lexical similarity between the query and document vectors. The efficiency of this model is further improved through the effective organization of text content using the inverted index. Moving towards more sophisticated approaches, statistical language models have been introduced to estimate the likelihood of term occurrences and incorporate context information, leading to more accurate and context-aware retrieval [27, 54]. In recent years, the neural IR [30, 55, 56] paradigm has gained considerable attention in the research community. By harnessing the powerful representation capabilities of neural networks, this paradigm can capture semantic relationships between queries and documents, thereby significantly enhancing retrieval performance.

Researchers have identified several challenges with implications for the performance and effectiveness of IR systems, such as query ambiguity and retrieval efficiency. In

6. https://github.com/RUC-NLPIR/LLM4IR-Survey light of these challenges, researchers have directed their attention toward crucial modules within the retrieval process, aiming to address specific issues and effectuate corresponding enhancements. The pivotal role of these modules in ameliorating the IR pipeline and elevating system performance cannot be overstated. In this survey, we focus on the following four modules, which have been greatly enhanced by LLMs.

Query Rewriter is an essential IR module that seeks to improve the precision and expressiveness of user queries. Positioned at the early stage of the IR pipeline, this module assumes the crucial role of refining or modifying the initial query to align more accurately with the user's information requirements. As an integral part of query rewriting, query expansion techniques, with pseudo relevance feedback being a prominent example, represent the mainstream approach to achieving query expression refinement. In addition to its utility in improving search effectiveness across general scenarios, the query rewriter finds application in diverse specialized retrieval contexts, such as personalized search and conversational search, thus further demonstrating its significance.

Retriever, as discussed here, is typically employed in the early stages of IR for document recall. The evolution of retrieval technologies reflects a constant pursuit of more effective and efficient methods to address the challenges posed by ever-growing text collections. In numerous experiments on IR systems over the years, the classical "bagof-words" model BM25 [29] has demonstrated its robust performance and high efficiency. In the wake of the neural IR paradigm's ascendancy, prevalent approaches have primarily revolved around projecting queries and documents into high-dimensional vector spaces, and subsequently computing their relevance scores through inner product calculations. This paradigmatic shift enables a more efficient understanding of query-document relationships, leveraging the power of vector representations to capture semantic similarities.

Reranker, as another crucial module in the retrieval pipeline, primarily focuses on fine-grained reordering of documents within the retrieved document set. Different from the retriever, which emphasizes the balance of efficiency and effectiveness, the reranker module places a greater emphasis on the quality of document ranking. In pursuit of enhancing the search result quality, researchers delve into more complex matching methods than the traditional vector inner product, thereby furnishing richer matching signals to the reranker. Moreover, the reranker facilitates the adoption of specialized ranking strategies tailored to meet distinct user requirements, such as personalized and diversified search results. By integrating domain-specific objectives, the reranker module can deliver tailored and purposeful search results, enhancing the overall user experience.

Reader has evolved as a crucial module with the rapid development of LLM technologies. Its ability to comprehend real-time user intent and generate dynamic responses based on the retrieved text has revolutionized the presentation of IR results. In comparison to presenting a list of candidate
documents, the reader module organizes answer texts more intuitively, simulating the natural way humans access information. To enhance the credibility of generated responses, the integration of references into generated responses has been an effective technique of the reader module.

Furthermore, researchers explore unifying the above modules to develop a novel LLM-driven search model known as the Search Agent. The search agent is distinguished by its simulation of an automated search and result understanding process, which furnishes users with accurate and readily comprehensible answers. WebGPT [24] serves as a pioneering work in this category, which models the search process as a sequence of actions of an LLM-based agent within a search engine environment, autonomously accomplishing the whole search pipeline. By integrating the existing search stack, search agents have the potential to become a new paradigm in future IR.

### 2.2 Large Language Models

Language models (LMs) are designed to calculate the generative likelihood of word sequences by taking into account the contextual information from preceding words, thereby predicting the probability of subsequent words. Consequently, by employing certain word selection strategies (such as greedy decoding or random sampling), LMs can proficiently generate natural language texts. Although the primary objective of LMs lies in text generation, recent studies [57] have revealed that a wide array of natural language processing problems can be effectively reformulated into a text-to-text format, thus rendering them amenable to resolution through text generation. This has led to LMs becoming the de facto solution for the majority of text-related problems.

The evolution of LMs can be categorized into four primary stages, as discussed in prior literature [52]. Initially, LMs were rooted in statistical learning techniques and were termed statistical language models. These models tackled the issue of word prediction by employing the Markov assumption to predict the subsequent word based on preceding words. Thereafter, neural networks, particularly recurrent neural networks (RNNs), were introduced to calculate the likelihood of text sequences and establish neural language models. These advancements made it feasible to utilize LMs for representation learning beyond mere word sequence modeling. ELMo [58] first proposed to learn contextualized word representations through pretraining a bidirectional LSTM (biLSTM) network on largescale corpora, followed by fine-tuning on specific downstream tasks. Similarly, BERT [59] proposed to pre-train a Transformer [60] encoder with a specially designed Masked Language Modeling (MLM) task and Next Sentence Prediction (NSP) task on large corpora. These studies initiated a new era of pre-trained language models (PLMs), with the "pre-training then fine-tuning" paradigm emerging as the prevailing learning approach. Along this line, numerous generative PLMs (e.g., GPT-2 [33], BART [61], and T5 [57]) have been developed for text generation problems including summarization, machine translation, and dialogue generation. Recently, researchers have observed that increasing the scale of PLMs (e.g., model size or data amount) can

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-04.jpg?height=781&width=873&top_left_y=149&top_left_x=1081)

Fig. 2. The evolution of LLMs (encoder-decoder and decoder-only structures).

consistently improve their performance on downstream tasks (a phenomenon commonly referred to as the scaling law [62, 63]). Moreover, large-sized PLMs exhibit promising abilities (termed emergent abilities [42]) in addressing complex tasks, which are not evident in their smaller counterparts. Therefore, the research community refers to these large-sized PLMs as large language models (LLMs).

As shown in Figure 2, existing LLMs can be categorized into two groups based on their architectures: encoderdecoder [57, 61, 64-69] and decoder-only [33-35, 70-80] models. The encoder-decoder models incorporate an encoder component to transform the input text into vectors, which are then employed for producing output texts. For example, T5 [57] is an encoder-decoder model that converts each natural language processing problem into a text-totext form and resolves it as a text generation problem. In contrast, decoder-only models, typified by GPT, rely on the Transformer decoder architecture. It uses a self-attention mechanism with a diagonal attention mask to generate a sequence of words from left to right. Building upon the success of GPT-3 [34], which is the first model to encompass over 100B parameters, several noteworthy models have been inspired, including GPT-J, BLOOM [78], OPT [75], Chinchilla [81], and LLaMA [35]. These models follow the similar Transformer decoder structure as GPT-3 and are trained on various combinations of datasets.

Owing to their vast number of parameters, fine-tuning LLMs for specific tasks, such as IR, is often deemed impractical. Consequently, two prevailing methods for applying LLMs have been established: in-context learning (ICL) and parameter-efficient fine-tuning. ICL is one of the emergent abilities of LLMs [34] empowering them to comprehend and furnish answers based on the provided input context, rather than relying merely on their pre-training knowledge. This method requires only the formulation of the task description and demonstrations in natural language, which are then fed as input to the LLM. Notably, parameter tuning is not

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-05.jpg?height=651&width=859&top_left_y=146&top_left_x=167)

Fig. 3. An example of LLM-based query rewriting for ad-hoc search. The example is cited from the Query2Doc paper [86]. LLMs are used to generate a passage to supplement the original query, where $N=0$ and $N>0$ correspond to zero-shot and few-shot scenarios.

required for ICL. Additionally, the efficacy of ICL can be further augmented through the adoption of chain-of-thought prompting, involving multiple demonstrations (describe the chain of thought examples) to guide the model's reasoning process. ICL is the most commonly used method for applying LLMs to IR. Parameter-efficient fine-tuning [82-84] aims to reduce the number of trainable parameters while maintaining satisfactory performance. LoRA [82], for example, has been widely applied to open-source LLMs (e.g., LLaMA and BLOOM) for this purpose. Recently, QLoRA [85] has been proposed to further reduce memory usage by leveraging a frozen 4-bit quantized LLM for gradient computation. Despite the exploration of parameter-efficient finetuning for various NLP tasks, its implementation in IR tasks remains relatively limited, representing a potential avenue for future research.

## 3 QuERY ReWRITER

Query rewriting in modern IR systems is essential for improving search query effectiveness and accuracy. It reformulates users' original queries to better match search results, alleviating issues like vague queries or vocabulary mismatches between the query and target documents. This task goes beyond mere synonym replacement, requiring an understanding of user intent and query context, particularly in complex searches like conversational queries. Effective query rewriting enhances search engine performance.

Traditional methods for query rewriting improve retrieval performance by expanding the initial query with information from highly-ranked relevant documents. Mainlyused methods include relevance feedback [87-92], wordembedding based methods $[93,94]$ etc. However, the limited ability of semantic understanding and comprehension of user search intent limits their performance in capturing the full scope of user intent.

Recent advancements in LLMs present promising opportunities to boost query rewriting capabilities. On one hand,

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-05.jpg?height=1022&width=854&top_left_y=148&top_left_x=1096)

Fig. 4. An example of LLM-based query rewriting for conversational search. The example is cited from LLMCS [95]. The LLM is used to generate a query based on the demonstrations and previous search context. Additional responses are required to be generated for improving the query understanding. $N=0$ and $N>0$ correspond to zero-shot and few-shot scenarios.

given the context and subtleties of a query, LLMs can provide more accurate and contextually relevant rewrites. On the other hand, LLMs can leverage their extensive knowledge to generate synonyms and related concepts, enhancing queries to cover a broader range of relevant documents, thereby effectively addressing the vocabulary mismatch problem. In the following sections, we will introduce the recent works that employ LLMs in query rewriting.

### 3.1 Rewriting Scenario

Query rewriting typically serves two scenarios: ad-hoc retrieval, which mainly addresses vocabulary mismatches between queries and candidate documents, and conversational search, which refines queries based on evolving conversations. The upcoming section will delve into the role of query rewriting in these two domains and explore how LLMs enhance this process.

### 3.1.1 Ad-hoc Retrieval

In ad-hoc retrieval, queries are often short and ambiguous. In such scenarios, the main objectives of query rewriting include adding synonyms or related terms to address vocabulary mismatches and clarifying ambiguous queries to more accurately align with user intent. From this perspective, LLMs have inherent advantages in query rewriting.

Primarily, LLMs have a deep understanding of language semantics, allowing them to capture the meaning of queries more effectively. Besides, LLMs can leverage their extensive training on diverse datasets to generate contextually relevant synonyms and expand queries, ensuring broader and more precise search result coverage. Additionally, studies have shown that LLMs' integration of external factual corpora [96-99] and thoughtful model design [100] further enhance their accuracy in generating effective query rewrites, especially for specific tasks.

Currently, there are many studies leveraging LLMs to rewrite queries in adhoc retrieval. We introduce the typical method Query2Doc [86] as an example. As shown in Figure 3, Query2Doc prompts the LLMs to generate a relevant passage according to the original query ("when was pokemon green released?"). Subsequently, the original query is expanded by incorporating the generated passage. The retriever module uses this new query to retrieve a list of relevant documents. Notably, the generated passage contains additional detailed information, such as "Pokemon Green was released in Japan on February 27th", which effectively mitigates the "vocabulary mismatch" issue to some extent.

In addition to addressing the "vocabulary mismatch" problem [96-99, 101, 102], other works utilize LLMs for different challenges in ad-hoc retrieval. For instance, PromptCase [103] leverages LLMs in legal case retrieval to simplify complex queries into more searchable forms. This involves using LLMs to identify legal facts and issues, followed by a prompt-based encoding scheme for effective language model encoding.

### 3.1.2 Conversational Search

Query rewrites in conversational search play a pivotal role in enhancing the search experience. Unlike traditional queries in ad-hoc retrieval, conversational search involves a dialogue-like interaction, where the context and user intent evolve with each interaction. In conversational search, query rewriting involves understanding the entire conversation's context, clarifying any ambiguities, and personalizing responses based on user history. The process includes dynamic query expansion and refinement based on dialogue information. This makes conversational query rewriting a sophisticated task that goes beyond traditional search, focusing on natural language understanding and user-centric interaction.

In the era of LLMs, leveraging LLMs in conversational search tasks offers several advantages. First, LLMs possess strong contextual understanding capabilities, enabling them to better comprehend users' search intent within the context of multi-turn conversations between users and the system. Second, LLMs exhibit powerful generation abilities, allowing them to simulate dialogues between users and the system, thereby facilitating more robust search intent modeling.

The LLMCS framework [95] is a pioneering approach that employs LLMs to effectively extract and understand user search intent within conversational contexts. As illustrated in their work, LLMCS uses LLMs to produce both query rewrites and extensive hypothetical system responses from various perspectives. These outputs are combined into a comprehensive representation that effectively captures the user's full search intent. The experimental results show that including detailed hypothetical responses with concise query rewrites markedly improves search performance by adding more plausible search intent. Ye et al. [104] claims that human query rewrite may lack sufficient information for optimal retrieval performance. It defines four essential properties for well-formed LLM-generated query rewrites. Results show that their method informative query rewrites can yield substantially improved retrieval performance compared to human rewrites.

Besides, LLMs can be used as a data expansion tool in conversational dense retrieval. Attributed to the high cost of producing hand-written dialogues, data scarcity presents a significant challenge in the domain of conversational search. To address this problem, CONVERSER [105] employs LLMs to generate synthetic passage-dialogue pairs through fewshot demonstrations. Furthermore, it efficiently trains a dense retriever using a minimal dataset of six in-domain dialogues, thus mitigating the issue of data sparsity.

### 3.2 Rewriting Knowledge

Query rewriting typically necessitates additional corpora for refining initial queries. Considering that LLMs incorporate world knowledge in their parameters, they are naturally capable of rewriting queries. We refer to these methods, which rely exclusively on the intrinsic knowledge of LLMs, as LLM-only methods. While LLMs encompass a broad spectrum of knowledge, they may be inadequate in specialized areas. Furthermore, LLMs can introduce concept drift, leading to noisy relevance signals. To address this issue, some methods incorporate domain-specific corpora to provide more detailed and relevant information in query rewriting. We refer to methods enhanced by domain-specific corpora to boost LLM performance as corpus-enhanced LLM-based methods. In this section, we will introduce these two methods in detail.

### 3.2.1 LLM-only methods

LLMs are capable of storing knowledge within their parameters, making it a natural choice to capitalize on this knowledge for the purpose of query rewriting. As a pioneering work in LLM-based query rewriting, HyDE [101] generates a hypothetical document by LLMs according to the given query and then uses a dense retriever to retrieve relevant documents from the corpus that are relevant to the generated document. Query2doc [86] generates pseudo documents via prompting LLMs with few-shot demonstrations, and then expands the query with the generated pseudo document. Furthermore, the influence of different prompting methods and various model sizes on query rewriting has also been investigated [102]. To better accommodate the frozen retriever and the LLM-based reader, a small language model is employed as the rewriter that is trained using reinforcement learning techniques with the rewards provided by the LLM-based reader [100]. GFF [106] presents a "Generate, Filter, and Fuse" method for query expansion. It employs an LLM to create a set of related keywords via a reasoning chain. Then, a self-consistency filter is used to identify the most important keywords, which are
concatenated with the original queries for the downstream reranking task.

It is worth noting that though the designs of these methods are different, all of them rely on the world knowledge stored in LLMs without additional corpora.

### 3.2.2 Corpus-enhanced LLM-based methods

Although LLMs exhibit remarkable capabilities, the lack of domain-specific knowledge may lead to the generation of hallucinatory or irrelevant queries. To address this issue, recent studies [96-99] have proposed a hybrid approach that enhances LLM-based query rewriting methods with an external document corpus.

Why incorporate a document corpus? The integration of a document corpus offers several notable advantages. Firstly, it boosts relevance by using relevant documents to refine query generation, reducing irrelevant content and improving contextually appropriate outputs. Second, enhancing LLMs with up-to-date information and specialized knowledge in specific fields enables them to effectively deal with queries that are both current and specific to certain domains.

How to incorporate a document corpus? Thanks to the flexibility of LLMs, various paradigms have been proposed to incorporate a document corpus into LLM-based query rewriting, which can be summarized as follows.

- Late fusion of LLM-based re-writing and pseudo relevance feedback (PRF) retrieval results. Traditional PRF methods leverage relevant documents retrieved from a document corpus to rewrite queries, which restricts the query to the information contained in the target corpus. On the contrary, LLM-based rewriting methods provide external context not present in the corpus, which is more diverse. Both approaches have the potential to independently enhance retrieval performance. Therefore, a straightforward strategy for combining them is using a weighted fusion method for retrieval results [99].
- Combining retrieved relevant documents in the prompts of LLMs. In the era of LLMs, incorporating instructions within the prompts is the most flexible method for achieving specific functionalities. QUILL [97] and CAR [107] illustrate how retrieval augmentation of queries can provide LLMs with context that significantly enhances query understanding. LameR [108] takes this further by using LLM expansion to improve the simple BM25 retriever, introducing a retrieve-rewrite-retrieve framework. Experimental results reveal that even basic term-based retrievers can achieve comparable performance when paired with LLMbased rewriters. Additionally, InteR [98] proposes a multiturn interaction framework between search engines and LLMs. This enables search engines to expand queries using LLM-generated insights, while LLMs refine prompts using relevant documents sourced from the search engines.
- Enhancing factuality of generative relevance feedback (GRF) by pseudo relevance feedback (PRF). Although generative documents are often relevant and diverse, they exhibit hallucinatory characteristics. In contrast, traditional documents are generally regarded as reliable sources of factual information. Motivated by this observation, GRM [96] proposes a novel technique known as relevance-aware sample estimation (RASE). RASE leverages relevant documents retrieved from
TABLE 1. Partial Examples of different prompting methods in query rewriting.

| Methods | Prompts |
| :--- | :--- |
| HyDE [101] | Please write a passage to answer the question. <br> Question: $\{\# Q$ Zestion $\}$ Passage: <br> Give a question $\{\# Q$ uestion $\}$ and its possible an- <br> swering passages: A. $\{\#$ Passage 1\} B. \{\#Passage 2\} <br> C. \{\#Passage 3\} ... Please write a correct answering <br> passage. |
| LameR [108] |  |

the collection to assign weights to generated documents. In this way, GRM ensures that relevance feedback is not only diverse but also maintains a high degree of factuality.

### 3.3 Rewriting Approaches

There are three main approaches used for leveraging LLMs in query rewriting: prompting methods, fine-tuning, and knowledge distillation. Prompting methods involve using specific prompts to direct LLM output, providing flexibility and interpretability. Fine-tuning adjusts pre-trained LLMs on specific datasets or tasks to improve domain-specific performance, mitigating the general nature of LLM world knowledge. Knowledge distillation, on the other hand, transfers LLM knowledge to lightweight models, simplifying the complexity associated with retrieval augmentation. In the following section, we will introduce these three methods in detail.

### 3.3.1 Prompting

Prompting in LLMs refers to the technique of providing a specific instruction or context to guide the model's generation of text. The prompt serves as a conditioning signal and influences the language generation process of the model. Existing prompting strategies can be roughly categorized into three groups: zero-shot prompting, few-shot prompting, and chain-of-thought (CoT) prompting [45].

- Zero-shot prompting. Zero-shot prompting involves instructing the model to generate texts on a specific topic without any prior exposure to training examples in that domain or topic. The model relies on its pre-existing knowledge and language understanding to generate coherent and contextually relevant expanded terms for original queries. Experiments show that zero-shot prompting is a simple yet effective method for query rewriting [98, 99, 102, 108-110].
- Few-shot prompting. Few-shot prompting, also known as in-context learning, involves providing the model with a limited set of examples or demonstrations related to the
desired task or domain [86, 102, 109, 110]. These examples serve as a form of explicit instruction, allowing the model to adapt its language generation to the specific task or domain at hand. Query2Doc [86] prompts LLMs to write a document that answers the query with some demo querydocument pairs provided by the ranking dataset, such as MSMARCO [111] and NQ [112]. This work experiments with a single prompt. To further study the impact of different prompt designing, recent works [102] have explored eight different prompts, such as prompting LLMs to generate query expansion terms instead of entire pseudo documents and CoT prompting. There are some illustrative prompts in Table 1. This work conducts more experiments than Query2Doc, but the results show that the proposed prompt is less effective than Query2Doc.
- Chain-of-thought prompting. CoT prompting [45] is a strategy that involves iterative prompting, where the model is provided with a sequence of instructions or partial outputs [102, 109]. In conversational search, the process of query re-writing is multi-turn, which means queries should be refined step-by-step with the interaction between search engines and users. This process is naturally coincided with CoT process. As shown in 4, users can conduct the CoT process through adding some instructions during each turn,

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-08.jpg?height=43&width=889&top_left_y=1109&top_left_x=152)
search, there is only one-round in query re-writing, so CoT could only be accomplished in a simple and coarse way. For example, as shown in Table 1, researchers add "Give the rationale before answering" in the instructions to prompt LLMs think deeply [102].

### 3.3.2 Fine-tuning

Fine-tuning is an effective approach for adapting LLMs to specific domains. This process usually starts with a pretrained language model, like GPT-3, which is then further trained on a dataset tailored to the target domain. This domain-specific training enables the LLM to learn unique patterns, terminology, and context relevant to the domain, which is able to improve its capacity to produce high-quality query rewrites.

BEQUE [113] leverages LLMs for rewriting queries in e-commerce product searches. It designs three Supervised Fine-Tuning (SFT) tasks: quality classification of e-commerce query rewrites, product title prediction, and CoT query rewriting. To our knowledge, it is the first model to directly fine-tune LLMs, including ChatGLM [68, 114], ChatGLM2.0 [68, 114], Baichuan [115], and Qwen [116], specifically for the query rewriting task. After the SFT stage, BEQUE uses an offline system to gather feedback on the rewrites and further aligns the rewriters with e-commerce search objectives through an object alignment stage. Online A/B testing demonstrates the effectiveness of the method.

### 3.3.3 Knowledge Distillation

Although LLM-based methods have demonstrated significant improvements in query rewriting tasks, their practical implementation for online deployment is hindered by the substantial latency caused by the computational requirements of LLMs. To address this challenge, knowledge distillation has emerged as a prominent technique in the
TABLE 2. Summary of existing LLM-enhanced query rewriting methods. "Docs" and "KD" stand for document corpus and knowledge distillation, respectively.

| Methods | Target | Data | Generation |
| :--- | :---: | :---: | :---: |
| HyDE [97] | Ad-hoc | LLMs | Prompting |
| Jagerman et al. [102] | Ad-hoc | LLMs | Prompting |
| Query2Doc [86] | Ad-hoc | LLMs | Prompting |
| Ma et al. [100] | Ad-hoc | LLMs | Finetuning |
| PromptCase [103] | Ad-hoc | LLMs | Prompting |
| GRF+PRF [99] | Ad-hoc | LLMs + Docs | Prompting |
| GRM [96] | Ad-hoc | LLMs + Docs | Prompting |
| InteR [98] | Ad-hoc | LLMs + Docs | Prompting |
| LameR [108] | Ad-hoc | LLMs + Docs | Prompting |
| CAR [107] | Ad-hoc | LLMs + Docs | Prompting |
| QUILL [97] | Ad-hoc | LLMs + Docs | KD \& Finetuning |
| LLMCS [95] | Conversational | LLMs | Prompting |
| CONVERSER [105] | Conversational | LLMs | Prompting |
| Ye et al. [104] | Conversational | LLMs | Prompting |

industry. In the QUILL [97] framework, a two-stage distillation method is proposed. This approach entails utilizing a retrieval-augmented LLM as the professor model, a vanilla LLM as the teacher model, and a lightweight BERT model as the student model. The professor model is trained on two extensive datasets, namely Orcas-I [117] and EComm [97], which are specifically curated for query intent understanding. Subsequently, a two-stage distillation process is employed to transfer knowledge from the professor model to the teacher model, followed by knowledge transfer from the teacher model to the student model. Empirical findings demonstrate that this knowledge distillation methodology surpasses the simple scaling up of model size from base to XXL, resulting in even more substantial improvements. In a recently proposed "rewrite-retrieve-read" framework [100], an LLM is first used to rewrite the queries by prompting, followed by a retrieval-augmented reading process. To improve framework effectiveness, a trainable rewriter, implemented as a small language model, is incorporated to further adapt search queries to align with both the frozen retriever and the LLM reader's requirements. The rewriter's refinement involves a two-step training process. Initially, supervised warm-up training is conducted using pseudo data. Then, the retrieve-then-read pipeline is described as a reinforcement learning scenario, with the rewriter's training acting as a policy model to maximize pipeline performance rewards.

### 3.4 Limitations

While LLMs offer promising capabilities for query rewriting, they also meet several challenges. Here, we outline two main limitations of LLM-based query rewriters.

### 3.4.1 Concept Drifts

When using LLMs for query rewriting, they may introduce unrelated information, known as concept drift, due to their extensive knowledge base and tendency to produce detailed and redundant content. While this can enrich the query, it also risks generating irrelevant or off-target results.

This phenomenon has been reported in several studies [107, 113, 118] These studies highlight the need for a balanced approach in LLM-based query rewriting, ensuring
that the essence and focus of the original query are maintained while leveraging the LLM's ability to enhance and clarify the query. This balance is crucial for effective search and IR applications.

### 3.4.2 Correlation between Retrieval Performance and Expansion Effects

Recently, a comprehensive study [119] conduct experiments on various expansion techniques and downstream ranking models, which reveals a notable negative correlation between retriever performance and the benefits of expansion. Specifically, while expansion tends to enhance the scores of weaker models, it generally hurts stronger models. This observation suggests a strategic approach: employ expansions with weaker models or in scenarios where the target dataset substantially differs in format from the training corpus. In other cases, it is advisable to avoid expansions to maintain clarity of the relevance signal.

## 4 RETRIEVER

In an IR system, the retriever serves as the first-pass document filter to collect broadly relevant documents for user queries. Given the enormous amounts of documents in an IR system, the retriever's efficiency in locating relevant documents is essential for maintaining search engine performance. Meanwhile, a high recall is also important for the retriever, as the retrieved documents are then fed into the ranker to generate final results for users, which determines the ranking quality of search engines.

In recent years, retrieval models have shifted from relying on statistic algorithms [29] to neural models [3, 31]. The latter approaches exhibit superior semantic capability and excel at understanding complicated user intent. The success of neural retrievers relies on two key factors: data and model. From the data perspective, a large amount of highquality training data is essential. This enables retrievers to acquire comprehensive knowledge and accurate matching patterns. Furthermore, the intrinsic quality of search data, i.e., issued queries and document corpus, significantly influences retrieval performance. From the model perspective, a strongly representational neural architecture allows retrievers to effectively store and apply knowledge obtained from the training data.

Unfortunately, there are some long-term challenges that hinder the advancement of retrieval models. First, user queries are usually short and ambiguous, making it difficult to precisely understand the user's search intents for retrievers. Second, documents typically contain lengthy content and substantial noise, posing challenges in encoding long documents and extracting relevant information for retrieval models. Additionally, the collection of human-annotated relevance labels is time-consuming and costly. It restricts the retrievers' knowledge boundaries and their ability to generalize across different application domains. Moreover, existing model architectures, primarily built on BERT [59], exhibit inherent limitations, thereby constraining the performance potential of retrievers. Recently, LLMs have exhibited extraordinary abilities in language understanding, text generation, and reasoning. This has motivated researchers to use these abilities to tackle the aforementioned challenges and aid in developing superior retrieval models. Roughly, these studies can be categorized into two groups, i.e., (1) leveraging LLMs to generate search data, and (2) employing LLMs to enhance model architecture.

### 4.1 Leveraging LLMs to Generate Search Data

In light of the quality and quantity of search data, there are two prevalent perspectives on how to improve retrieval performance via LLMs. The first perspective revolves around search data refinement methods, which concentrate on reformulating input queries to precisely present user intents. The second perspective involves training data augmentation methods, which leverage LLMs' generation ability to enlarge the training data for dense retrieval models, particularly in zero- or few-shot scenarios.

### 4.1.1 Search Data Refinement

Typically, input queries consist of short sentences or keyword-based phrases that may be ambiguous and contain multiple possible user intents. Accurately determining the specific user intent is essential in such cases. Moreover, documents usually contain redundant or noisy information, which poses a challenge for retrievers to extract relevance signals between queries and documents. Leveraging the strong text understanding and generation capabilities of LLMs offers a promising solution to these challenges. As yet, research efforts in this domain primarily concentrate on employing LLMs as query rewriters, aiming to refine input queries for more precise expressions of the user's search intent. Section 3 has provided a comprehensive overview of these studies, so this section refrains from further elaboration. In addition to query rewriting, an intriguing avenue for exploration involves using LLMs to enhance the effectiveness of retrieval by refining lengthy documents. This intriguing area remains open for further investigation and advancement.

### 4.1.2 Training Data Augmentation

Due to the expensive economic and time costs of humanannotated labels, a common problem in training neural retrieval models is the lack of training data. Fortunately, the excellent capability of LLMs in text generation offers a potential solution. A key research focus lies in devising strategies to leverage LLMs' capabilities to generate pseudorelevant signals and augment the training dataset for the retrieval task.

Why do we need data augmentation? Previous studies of neural retrieval models focused on supervised learning, namely training retrieval models using labeled data from specific domains. For example, MS MARCO [111] provides a vast repository, containing a million passages, more than 200,000 documents, and 100,000 queries with humanannotated relevance labels, which has greatly facilitated the development of supervised retrieval models. However, this paradigm inherently constrains the retriever's generalization ability for out-of-distribution data from other domains. The application spectrum of retrieval models varies from natural question-answering to biomedical IR, and it is expensive to annotate relevance labels for data from different domains. As a result, there is an emerging need for zero-shot

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-10.jpg?height=678&width=599&top_left_y=146&top_left_x=164)

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-10.jpg?height=648&width=555&top_left_y=156&top_left_x=777)

Framework of pseudo query generation

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-10.jpg?height=607&width=599&top_left_y=168&top_left_x=1354)

Framework of relevance label generation

Fig. 5. Two typical frameworks for LLM-based data augmentation in the retrieval task (right), along with their prompt examples (left). Note that the methods of relevance label generation do not treat questions as inputs but regard their generation probabilities conditioned on the retrieved passages as soft relevance labels.

TABLE 3. The comparison of existing data augmentation methods powered by LLMs for training retrieval models.

| Methods | \# Examples | Generator | Synthetic Data | Filter Method |
| :--- | :---: | :---: | :---: | :---: |
| InPairs [120] | 3 | Curie | Relevant query | Generation probability |
| Ma et al. [121] | $0-2$ | Alpaca-LLaMA \& tk-Instruct | Relevant query | Fixed |
| InPairs-v2 [122] | 3 | GPT-J | Relevant query | Relevance score from |
| PROMPTAGATOR [123] | $0-8$ | FLAN | Relevant query | Round-trip filtering |
| TQGen [124] | 0 | T0 | Relevant query | Generation probability |
| UDAPDR [125] | $0-3$ | GPT3 \& FLAN-T5-XXL | Relevant query | Round-trip filtering |
| SPTAR [126] | $1-2$ | LLaMA-7B \& Vicuna-7B | Relevant query | BM25 filtering |
| ART [127] | 0 | T5-XL \& T5-XXL | Soft relevance labels | Fixed |

and few-shot learning models to address this problem [128]. A common practice to improve the models' effectiveness in a target domain without adequate label signals is through data augmentation.

How to apply LLMs for data augmentation? In the scenario of IR, it is easy to collect numerous documents. However, the challenging and costly task lies in gathering real user queries and labeling the relevant documents accordingly. Considering the strong text generation capability of LLMs, many researchers [120, 122] suggest using LLM-driven processes to create pseudo queries or relevance labels based on existing collections. These approaches facilitate the construction of relevant query-document pairs, enlarging the training data for retrieval models. According to the type of generated data, there are two mainstream approaches that complement the LLM-based data augmentation for retrieval models, i.e., pseudo query generation and relevance label generation. Their frameworks are visualized in Figure 5. Next, we will give an overview of the related studies.

- Pseudo query generation. Given the abundance of documents, a straightforward idea is to use LLMs for generating their corresponding pseudo queries. One such illustration is presented by inPairs [120], which leverages the in-context learning capability of GPT-3. This method employs a collection of query-document pairs as demonstrations. These pairs are combined with a document and presented as input to GPT-3, which subsequently generates possible relevant queries for the given document. By combining the same demonstration with various documents, it is easy to create a vast pool of synthetic training samples and support the fine-tuning of retrievers on specific target domains. Recent studies [121] have also leveraged open-sourced LLMs, such as Alpaca-LLaMA and tk-Instruct, to produce sufficient pseudo queries and applied curriculum learning to pre-train dense retrievers. To enhance the reliability of these synthetic samples, a fine-tuned model (e.g., a monoT5-3B model finetuned on MSMARCO [122]) is employed to filter the generated queries. Only the top pairs with the highest estimated relevance scores are kept for training. This "generating-thenfiltering" paradigm can be conducted iteratively in a roundtrip filtering manner, i.e., by first fine-tuning a retriever on the generated samples and then filtering the generated samples using this retriever. Repeating these EM-like steps until convergence can produce high-quality training sets [123]. Furthermore, by adjusting the prompt given to LLMs, they can generate queries of different types. This capability allows for a more accurate simulation of real queries with various patterns [124].

In practice, it is costly to generate a substantial number of pseudo queries through LLMs. Balancing the generation costs and the quality of generated samples has become an urgent problem. To tackle this, UDAPDR [125] is proposed, which first produces a limited set of synthetic queries using

LLMs for the target domain. These high-quality examples are subsequently used as prompts for a smaller model to generate a large number of queries, thereby constructing the training set for that specific domain. It is worth noting that the aforementioned studies primarily rely on fixed LLMs with frozen parameters. Empirically, optimizing LLMs' parameters can significantly improve their performance on downstream tasks. Unfortunately, this pursuit is impeded by the prohibitively high demand for computational resources. To overcome this obstacle, SPTAR [126] introduces a soft prompt tuning technique that only optimizes the prompts' embedding layer during the training process. This approach allows LLMs to better adapt to the task of generating pseudo-queries, striking a favorable balance between training cost and generation quality.

In addition to the above studies, pseudo query generation methods are also introduced in other application scenarios, such as conversational dense retrieval [105] and multilingual dense retrieval [129].

- Relevance label generation. In some downstream tasks of retrieval, such as question-answering, the collection of questions is also sufficient. However, the relevance labels connecting these questions with the passages of supporting evidence are very limited. In this context, leveraging the capability of LLMs for relevance label generation is a promising approach that can augment the training corpus for retrievers. A recent method, ART [127], exemplifies this approach. It first retrieves the top-relevant passages for each question. Then, it employs an LLM to produce the generation probabilities of the question conditioned on these top passages. After a normalization process, these probabilities serve as soft relevance labels for the training of the retriever.

Additionally, to highlight the similarities and differences among the corresponding methods, we present a comparative result in Table 3. It compares the aforementioned methods from various perspectives, including the number of examples, the generator employed, the type of synthetic data produced, the method applied to filter synthetic data, and whether LLMs are fine-tuned. This table serves to facilitate a clearer understanding of the landscape of these methods.

### 4.2 Employing LLMs to Enhance Model Architecture

Leveraging the excellent text encoding and decoding capabilities of LLMs, it is feasible to understand queries and documents with greater precision compared to earlier smallersized models [59]. Researchers have endeavored to utilize LLMs as the foundation for constructing advanced retrieval models. These methods can be grouped into two categories, i.e., dense retrievers and generative retrievers.

### 4.2.1 Dense Retriever

In addition to the quantity and quality of the data, the representative capability of models also greatly influences the efficacy of retrievers. Inspired by the LLM's excellent capability to encode and comprehend natural language, some researchers [130-132] leverage LLMs as retrieval encoders and investigate the impact of model scale on retriever performance.
General Retriever. Since the effectiveness of retrievers primarily relies on the capability of text embedding, the evolution of text embedding models often has a significant impact on the progress of retriever development. In the era of LLMs, a pioneer work is made by OpenAI [130]. They view the adjacent text segments as positive pairs to facilitate the unsupervised pre-training of a set of text embedding models, denoted as cpt-text, whose parameter values vary from 300M to 175B. Experiments conducted on the MS MARCO [111] and BEIR [128] datasets indicate that larger model scales have the potential to yield improved performance in unsupervised learning and transfer learning for text search tasks. Nevertheless, pre-training LLMs from scratch is prohibitively expensive for most researchers. To overcome this limitation, some studies [131, 133] use pretrained LLMs to initialize the bi-encoder of dense retriever. Specifically, GTR [133] adopts T5-family models, including T5-base, Large, XL, and XXL, to initialize and fine-tune dense retrievers. RepLLaMA [131] further fine-tunes the LLaMA model on multiple stages of IR, including retrieval and reranking. For the dense retrieval task, RepLLaMA appends an end-of-sequence token " $</$ s $>$ " to the input sequences, i.e., queries or documents, and regards its output embeddings as the representation of queries or documents. The experiments confirm again that larger model sizes can lead to better performance, particularly in zero-shot settings. Notably, the researchers of RepLLaMA [131] also study the effectiveness of applying LLaMA in the reranking stage, which will be introduced in Section 5.1.3.

Task-aware Retriever. While the aforementioned studies primarily focus on using LLMs as text embedding models for downstream retrieval tasks, retrieval performance can be greatly enhanced when task-specific instructions are integrated. For example, TART [132] devises a task-aware retrieval model that introduces a task-specific instruction before the question. This instruction includes descriptions of the task's intent, domain, and desired retrieved unit. For instance, given that the task is question-answering, an effective prompt might be "Retrieve a Wikipedia text that answers this question. \{question\}". Here, "Wikipedia" (domain) indicates the expected source of retrieved documents, "text" (unit) suggests the type of content to retrieve, and "answers this question" (intent) demonstrates the intended relationship between the retrieved texts and the question. This approach can take advantage of the powerful language modeling capability and extensive knowledge of LLMs to precisely capture the user's search intents across various retrieval tasks. Considering the efficiency of retrievers, it first fine-tunes a TART-full model with cross-encoder architecture, which is initialized from LLMs (e.g., T0-3B, Flan-T5). Then, a TART-dull model initialized from Contriever [134] is learned by distillating knowledge from the TART-full.

### 4.2.2 Generative Retriever

Traditional IR systems typically follow the "index-retrievalrank" paradigm to locate relevant documents based on user queries, which has proven effective in practice. However, these systems usually consist of three separate modules: the index module, the retrieval module, and the reranking module. Therefore, optimizing these modules collectively
can be challenging, potentially resulting in sub-optimal retrieval outcomes. Additionally, this paradigm demands additional space for storing pre-built indexes, further burdening storage resources. Recently, model-based generative retrieval methods [135-137] have emerged to address these challenges. These methods move away from the traditional "index-retrieval-rank" paradigm and instead use a unified model to directly generate document identifiers (i.e., DocIDs) relevant to the queries. In these model-based generative retrieval methods, the knowledge of the document corpus is stored in the model parameters, eliminating the need for additional storage space for the index. Existing methods have explored generating document identifiers through fine-tuning and prompting of LLMs [138, 139]

Fine-tuning LLMs. Given the vast amount of world knowledge contained in LLMs, it is intuitive to leverage them for building model-based generative retrievers. DSI [138] is a typical method that fine-tunes the pre-trained T5 models on retrieval datasets. The approach involves encoding queries and decoding document identifiers directly to perform retrieval. They explore multiple techniques for generating document identifiers and find that constructing semantically structured identifiers yields optimal results. In this strategy, DSI applies hierarchical clustering to group documents according to their semantic embeddings and assigns a semantic DocID to each document based on its hierarchical group. To ensure the output DocIDs are valid and do represent actual documents in the corpus, DSI constructs a trie using all DocIDs and utilizes a constraint beam search during the decoding process. Furthermore, this approach observes that the scaling law, which suggests that larger LMs lead to improved performance, is also applied to generative retrievers.

Prompting LLMs. In addition to fine-tuning LLMs for retrieval, it has been found that LLMs (e.g., GPT-series models) can directly generate relevant web URLs for user queries with a few in-context demonstrations [139]. This unique capability of LLMs is believed to arise from their training exposure to various HTML resources. As a result, LLMs can naturally serve as generative retrievers that directly generate document identifiers to retrieve relevant documents for input queries. To achieve this, an LLM-URL [139] model is proposed. It utilizes the GPT-3 text-davinci-003 model to yield candidate URLs. Furthermore, it designs regular expressions to extract valid URLs from these candidates to locate the retrieved documents.

To provide a comprehensive understanding of this topic, Table 4 summarizes the common and unique characteristics of the LLM-based retrievers discussed above.

### 4.3 Limitations

Though some efforts have been made for LLM-augmented retrieval, there are still many areas that require more detailed investigation. For example, a critical requirement for retrievers is fast response, while the main problem of existing LLMs is the huge model parameters and overlong inference time. Addressing this limitation of LLMs to ensure the response time of retrievers is a critical task. Moreover, even when employing LLMs to augment datasets (a context
TABLE 4. The comparison of retrievers that leverage LLMs as the foundation. "KD" is short for "Knowledge Distillation".

| Methods | Backbone | Architecture | LLM's tuning |
| :---: | :---: | :---: | :---: |
| cpt-text [130] | GPT-series | Dense | Pre-training <br> Fine-tuning |
| GTR [133] | $\mathrm{T} 5$ | Dense |  <br> Fine-tuning |
| RepLLaMA [131] | LLAMA | Dense | Fine-tuning |
| TART-full [132] |  <br> Flan-T5 | Dense |  <br> Prompting |
| TART-dual [132] | Contriever | Dense |  <br> Prompting |
| DSI [138] | T5 | Generative | Fine-tuning |
| LLM-URL [139] | GPT-3 | Generative | Prompting |

TABLE 5. Summary of existing LLM-based re-ranking methods. "Enc" and "Dec" denote encoder and decoder, respectively.

| Paradigm | Type | Method |
| :--- | :---: | :--- |
| Supervised | Enc-only | [140] |
|  | Enc-dec | [13], [141], [142], [143] |
|  | Dec-only | [131], [144], [145] |
| Unsupervised | Pointwise | [146], [147], [148], [149], [150], [151] |
|  | Listwise | [152], [153], [154] |
|  | Pairwise | [155], [156] |
| Data Augmentation | - | [157], [158], [159], [160], [161], [162] |

with lower inference time demands), the potential mismatch between LLM-generated texts and real user queries could impact retrieval effectiveness. Furthermore, as LLMs usually lack domain-specific knowledge, they need to be finetuned on task-specific datasets before applying them to downstream tasks. Therefore, developing efficient strategies to fine-tune these LLMs with numerous parameters emerges as a key concern.

## 5 RERANKER

Reranker, as the second-pass document filter in IR, aims to rerank a document list retrieved by the retriever (e.g., BM25) based on the query-document relevance. Based on the usage of LLMs, the existing LLM-based reranking methods can be divided into three paradigms: utilizing LLMs as supervised rerankers, utilizing LLMs as unsupervised rerankers, and utilizing LLMs for training data augmentation. These paradigms are summarized in Table 5 and will be elaborated upon in the following sections. Recall that we will use the term document to refer to the text retrieved in general IR scenarios, including instances such as passages (e.g., passages in MS MARCO passage ranking dataset [111]).

### 5.1 Utilizing LLMs as Supervised Rerankers

Supervised fine-tuning is an important step in applying pre-trained LLMs to a reranking task. Due to the lack of awareness of ranking during pre-training, LLMs cannot appropriately measure the query-document relevance and fully understand the reranking tasks. By fine-tuning LLMs on task-specific ranking datasets, such as the MS MARCO passage ranking dataset [111], which includes signals of
both relevance and irrelevance, LLMs can adjust their parameters to yield better performance in the reranking tasks. Based on the backbone model structure, we can categorize existing supervised rerankers as: (1) encoder-only, (2) encoder-decoder, and (3) decoder-only.

### 5.1.1 Encoder-only

The encoder-based rerankers represent a significant turning point in applying LLMs to document ranking tasks. They demonstrate how some pre-trained language models (e.g., BERT [59]) can be finetuned to deliver highly accurate relevance predictions. A representative approach is monoBERT [140], which transforms a query-document pair into a sequence "[CLS] query [SEP] document [SEP]" as the model input and calculates the relevance score by feeding the "[CLS]" representation into a linear layer. The reranking model is optimized based on the cross-entropy loss.

### 5.1.2 Encoder-Decoder

In this field, existing studies mainly formulate document ranking as a generation task and optimize an encoderdecoder-based reranking model [13, 141-143]. Specifically, given the query and the document, reranking models are usually fine-tuned to generate a single token, such as "true" or "false". During inference, the query-document relevance score is determined based on the logit of the generated token. For example, a T5 model can be fine-tuned to generate classification tokens for relevant or irrelevant querydocument pairs [13]. At inference time, a softmax function is applied to the logits of "true" and "false" tokens, and the relevance score is calculated as the probability of the "true" token. The following method [141] involves a multi-view learning approach based on the T5 model. This approach simultaneously considers two tasks: generating classification tokens for a given query-document pair and generating the corresponding query conditioned on the provided document. DuoT5 [142] considers a triple $\left(q, d_{i}, d_{j}\right)$ as the input of the T5 model and is fine-tuned to generate token "true" if document $d_{i}$ is more relevant to query $q_{i}$ than document $d_{j}$, and "false" otherwise. During inference, for each document $d_{i}$, it enumerates all other documents $d_{j}$ and uses global aggregation functions to generate the relevance score $s_{i}$ for document $d_{i}$ (e.g., $s_{i}=\sum_{j} p_{i, j}$, where $p_{i, j}$ represents the probability of generating "true" when taking $\left(q, d_{i}, d_{j}\right)$ as the model input).

Although these generative loss-based methods outperform several strong ranking baselines, they are not optimal for reranking tasks. This stems from two primary reasons. First, it is commonly expected that a reranking model will yield a numerical relevance score for each querydocument pair rather than text tokens. Second, compared to generation losses, it is more reasonable to optimize the reranking model using ranking losses (e.g., RankNet [163]). Recently, RankT5 [143] has directly calculated the relevance score for a query-document pair and optimized the ranking performance with "pairwise" or "listwise" ranking losses. An avenue for potential performance enhancement lies in the substitution of the base-sized T5 model with its largerscale counterpart.

### 5.1.3 Decoder-only

Recently, there have been some attempts [131, 144, 145] to rerank documents by fine-tuning decoder-only models (such as LLaMA). For example, RankLLaMA [131] proposes formatting the query-document pair into a prompt "query: $\{$ query\} document: $\{$ document $\}$ [EOS]" and utilizes the last token representation for relevance calculation. Besides, RankingGPT [144] has been proposed to bridge the gap between LLMs' conventional training objectives and the specific needs of document ranking through two-stage training. The first stage involves continuously pretraining LLMs using a large number of relevant text pairs collected from web resources, helping the LLMs to naturally generate queries relevant to the input document. The second stage focuses on improving the model's text ranking performance using high-quality supervised data and welldesigned loss functions. Different from these pointwise rerankers [131, 144], Rank-without-GPT [145] proposes to train a listwise reranker that directly outputs a reranked document list. The authors first demonstrate that existing pointwise datasets (such as MS MARCO [111]), which only contain binary query-document labels, are insufficient for training efficient listwise rerankers. Then, they propose to use the ranking results of existing ranking systems (such as Cohere rerank API) as gold rankings to train a listwise reranker based on Code-LLaMA-Instruct.

### 5.2 Utilizing LLMs as Unsupervised Rerankers

As the size of LLMs scales up (e.g., exceeding 10 billion parameters), it becomes increasingly difficult to fine-tune the reranking model. Addressing this challenge, recent efforts have attempted to prompt LLMs to directly enhance document reranking in an unsupervised way. In general, these prompting strategies can be divided into three categories: pointwise, listwise, and pairwise methods. A comprehensive exploration of these strategies follows in the subsequent sections.

### 5.2.1 Pointwise methods

The pointwise methods measure the relevance between a query and a single document, and can be categorized into two types: relevance generation $[146,147]$ and query generation [148-150].

The upper part in Figure 6 (a) shows an example of relevance generation based on a given prompt, where LLMs output a binary label ("Yes" or "No") based on whether the document is relevant to the query. Following [13], the querydocument relevance score $f(q, d)$ can be calculated based on the log-likelihood of token "Yes" and "No" with a softmax function:

$$
\begin{equation*}
f(q, d)=\frac{\exp \left(S_{Y}\right)}{\exp \left(S_{Y}\right)+\exp \left(S_{N}\right)} \tag{1}
\end{equation*}
$$

where $S_{Y}$ and $S_{N}$ represent the LLM's log-likelihood scores of "Yes" and "No" respectively. In addition to binary labels, Zhuang et al. [147] propose to incorporate fine-grained relevance labels (e.g., "highly relevant", "somewhat relevant" and "not relevant") into the prompt, which helps LLMs more effectively differentiate among documents with varying levels of relevance to a query.

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-14.jpg?height=769&width=469&top_left_y=147&top_left_x=402)

(a) Pointwise method

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-14.jpg?height=347&width=854&top_left_y=149&top_left_x=863)

(b) Listwise method

![](https://cdn.mathpix.com/cropped/2024_06_04_d24032ae8bf4ef98d63dg-14.jpg?height=396&width=854&top_left_y=520&top_left_x=863)

(c) Pairwise method

Fig. 6. Three types of unsupervised reranking methods: (a) pointwise methods that consist of relevance generation (upper) and query generation (lower), (b) listwise methods, and (c) pairwise methods.

As for the query generation shown in the lower part of Figure 6 (a), the query-document relevance score is determined by the average log-likelihood of generating the actual query tokens based on the document:

$$
\begin{equation*}
\text { score }=\frac{1}{|q|} \sum_{i} \log p\left(q_{i} \mid q_{<i}, d, \mathcal{P}\right) \tag{2}
\end{equation*}
$$

where $|q|$ denotes the token number of query $q, d$ denotes the document, and $\mathcal{P}$ represents the provided prompt. The documents are then reranked based on their relevance scores. It has been proven that some LLMs (such as T0) yield significant performance in zero-shot document reranking based on the query generation method [148]. Recently, research [149] has also shown that the LLMs that are pre-trained without any supervised instruction fine-tuning (such as LLaMA) also yield robust zero-shot ranking ability.

Although effective, these methods primarily rely on a handcrafted prompt (e.g., "Please write a query based on this document"), which may not be optimal. As prompt is a key factor in instructing LLMs to perform various NLP tasks, it is important to optimize prompt for better performance. Along this line, a discrete prompt optimization method Co-Prompt [150] is proposed for better prompt generation in reranking tasks. Besides, PaRaDe [151] proposes a difficulty-based method to select few-show demonstrations to include in the prompt, proving significant improvements compared with zero-shot prompts.

Note that these pointwise methods rely on accessing the output logits of LLMs to calculate the query-document relevance scores. As a result, they are not applicable to closed-sourced LLMs, whose API-returned results do not include logits.

### 5.2.2 Listwise Methods

Listwise methods [152, 153] aim to directly rank a list of documents (see Figure 6 (b)). These methods insert the query and a document list into the prompt and instruct the LLMs to output the reranked document identifiers. Due to the limited input length of LLMs, it is not feasible to insert all candidate documents into the prompt. To alleviate this issue, these methods employ a sliding window strategy to rerank a subset of candidate documents each time. This strategy involves ranking from back to front using a sliding window, re-ranking only the documents within the window at a time.

Although listwise methods have yielded promising performance, they still suffer from some weaknesses. First, according to the experimental results [152], only the GPT-4based method can achieve competitive performance. When using smaller parameterized language models (e.g., FLANUL2 with 20B parameters), listwise methods may produce very few usable results and underperform many supervised methods. Second, the performance of listwise methods is highly sensitive to the document order in the prompt. When the document order is randomly shuffled, listwise methods perform even worse than BM25 [152], revealing positional bias issues in the listwise ranking of LLMs. To alleviate this issue, Tang et al. [154] introduce a permutation selfconsistency method, which involves shuffling the list in the prompt and aggregating the generated results to achieve a more accurate and unbiased ranking.

### 5.2.3 Pairwise Methods

In pairwise methods [155], LLMs are given a prompt that consists of a query and a document pair (see Figure 6 (c)). Then, they are instructed to generate the identifier of the document with higher relevance. To rerank all candidate documents, aggregation methods like AllPairs are used. AllPairs first generates all possible document pairs and aggregates a final relevance score for each document. To speed up the ranking process, efficient sorting algorithms, such as heap sort and bubble sort, are usually employed [155].

TABLE 6. The comparison between different methods. $N$ denotes the number of documents to rerank. The Complexity, Logits, and Batch represent the computational complexity, whether accesses LLM's logits, and whether allows batch inference respectively. $k$ is the constant in sliding windows strategy. As for the Performance, we use NDCG@10 as a metric, and the results are calculated by reranking the top 100 documents retrieved by BM25 on TREC-DL2019 and TREC-DL2020. The best model is in bold while the second-best is marked with an underline. The results come from previous study [155]. *Since the parameters of ChatGPT have not been released, its model parameters are based on public estimates [164].

|  | Methods | LLM | Size | Properties |  |  | Performance |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  |  | Complexity | Logits | Batching | TREC-DL19 | TREC-DL20 |
| Initial Retriever | BM25 | - | - | - | - | - | 50.58 | 47.96 |
| Supervised | monoBERT [140] | BERT | $340 \mathrm{M}$ | - | $\bar{\checkmark}$ | $\bar{\checkmark}$ | 70.50 | 67.28 |
|  |  | $\mathrm{T} 5$ | $220 \mathrm{M}$ | - | $\checkmark$ | $\checkmark$ | 71.48 | 66.99 |
|  | RankT5 [143] | T5 | $3 \mathrm{~B}$ | - | $\checkmark$ | $\checkmark$ | 71.22 | 69.49 |
| Unsupervised-Pointwise | Query Generation [11 | FLAN-UL2 | 20B | $O(N)$ | $\checkmark$ | $\checkmark$ | 58.95 | 60.02 |
|  | Relevance Gen | FLAN-UL2 | 20B | $O(N)$ | $\checkmark$ | $\checkmark$ | 64.61 | 65.39 |
| Unsupervised-Listwise | $\operatorname{RankGPT}_{3.5}$ [152] | gpt-3.5-turbo | $154 \mathrm{~B}^{*}$ | $O(k * N)$ |  |  | 65.80 | 62.91 |
|  | $\operatorname{RankGPT}_{4}[152$ | gpt-4 | $1 \mathrm{~T}^{*}$ | $O(k * N)$ |  |  | 75.59 | 70.56 |
| Unsupervised-Pairwise | PRP-Allpair [155] | FLAN-UL2 | 20B | $O\left(N^{2}\right)$ | $\checkmark$ | $\checkmark$ | $\underline{72.42}$ | $\overline{70.68}$ |
|  | PRP-Heapsort [155] | FLAN-UL2 | 20B | $O(N * \log N)$ | $\checkmark$ |  | $\overline{71.88}$ | 69.43 |

These sorting algorithms utilize efficient data structures to compare document pairs selectively and elevate the most relevant documents to the top of the ranking list, which is particularly useful in top- $k$ ranking. Experimental results show the state-of-the-art performance on the standard benchmarks using moderate-size LLMs (e.g., Flan-UL2 with 20B parameters), which are much smaller than those typically employed in listwise methods (e.g., GPT3.5).

Although effective, pairwise methods still suffer from high time complexity. To alleviate the efficiency problem, a setwise approach [156] has been proposed to compare a set of documents at a time and select the most relevant one from them. This approach allows the sorting algorithms (such as heap sort) to compare more than two documents at each step, thereby reducing the total number of comparisons and speeding up the sorting process.

### 5.2.4 Comparison and Discussion

In this part, we will compare different unsupervised methods from various aspects to better illustrate the strengths and weaknesses of each method, which is summarized in Table 6. We choose representative methods [146, 148, 152, 155] in pointwise, listwise and pairwise ranking, and include several supervised methods [13, 140, 143] mentioned in Section 5.1 for performance comparison.

The pointwise methods (Query Generation and Relevance Generation) judge the relevance of each querydocument pair independently, thus offering lower time complexity and enabling batch inference. However, compared to other methods, it does not have an advantage in terms of performance. The listwise method yields significant performance especially when calling GPT-4, but suffers from expensive API cost and non-reproducibility [160]. Compared with the listwise method, the pairwise method shows competitive results based on a much smaller model FLANUL2 (20B). Stemming from the necessity to compare an extensive number of document pairs, its primary drawback is low efficiency.

### 5.3 Utilizing LLMs for Training Data Augmentation

Furthermore, in the realm of reranking, researchers have explored the integration of LLMs for training data augmentation [157-162]. For example, ExaRanker [157] generates explanations for retrieval datasets using GPT-3.5, and subsequently trains a seq2seq ranking model to generate relevance labels along with corresponding explanations for given query-document pairs. InPars-Light [158] is proposed as a cost-effective method to synthesize queries for documents by prompting LLMs. Contrary to InPars-Light [158], a new dataset ChatGPT-RetrievalQA [159] is constructed by generating synthetic documents based on LLMs in response to user queries.

Recently, many studies [160-162] have also attempted to distill the document ranking capability of LLMs into a specialized model. RankVicuna [160] proposes to use the ranking list of RankGPT 3.5 [152] as the gold list to train a 7B parameter Vicuna model. RankZephyr [161] introduces a two-stage training strategy for distillation: initially applying the RankVicuna recipe to train Zephyr $\gamma$ in the first stage, and then further finetuning it in the second stage with the ranking results from RankGPT 4 . These two studies not only demonstrate competitive results but also alleviate the issue of ranking results non-reproducibility of black-box LLMs. Besides, researchers [162] have also tried to distill the ranking ability of a pairwise ranker, which is computationally demanding, into a simpler but more efficient pointwise ranker.

### 5.4 Limitations

Although recent research on utilizing LLMs for document reranking has made significant progress, it still faces some challenges. For example, considering the cost and efficiency, minimizing the number of calls to LLM APIs is a problem worth studying. Besides, while existing studies mainly focus on applying LLMs to open-domain datasets (such as MSMARCO [111]) or relevance-based text ranking tasks, their adaptability to in-domain datasets [128] and non-standard ranking datasets [165] remains an area that demands more comprehensive exploration.

## 6 READER

With the impressive capabilities of LLMs in understanding, extracting, and processing textual data, researchers explore expanding the scope of IR systems beyond content ranking to answer generation. In this evolution, a reader module has been introduced to generate answers based on the document corpus in IR systems. By integrating a reader module, IR systems can directly present conclusive passages to users. Compared with providing a list of documents, users can simply comprehend the answering passages instead of analyzing the ranking list in this new paradigm. Furthermore, by repeatedly providing documents to LLMs based on their generating texts, the final generated answers can potentially be more accurate and information-rich than the original retrieved lists.

A naive strategy for implementing this function is to heuristically provide LLMs with documents relevant to the user queries or the previously generated texts to support the following generation. However, this passive approach limits LLMs to merely collecting documents from IR systems without active engagement. An alternative solution is to train LLMs to interact proactively with search engines. For example, LLMs can formulate their own queries instead of relying solely on user queries or generated texts for references. According to the way LLMs utilize IR systems in the reader module, we can categorize them into passive readers and active readers. Each approach has its advantages and challenges for implementing LLM-powered answer generation in IR systems. Furthermore, since the documents provided by upstream IR systems are sometimes too long to directly feed as input for LLMs, some compression modules are proposed to extractively or abstractively compress the retrieved contexts for LLMs to understand and generate answers for queries. We will present these reader and compressor modules in the following parts and briefly introduce the existing analysis work on retrieval-augmented generation strategy and their applications.

### 6.1 Passive Reader

To generate answers for users, a straightforward strategy is to supply the retrieved documents according to the queries or previously generated texts from IR systems as inputs to LLMs for creating passages [23, 166-171, 173, 175, 176, 178180]. By this means, these approaches use the LLMs and IR systems separately, with LLMs functioning as passive recipients of documents from the IR systems. The strategies for utilizing LLMs within IR systems' reader modules can be categorized into the following three groups according to the frequency of retrieving documents for LLMs.

### 6.1.1 Once-Retrieval Reader

To obtain useful references for LLMs to generate responses for user queries, an intuitive way is to retrieve the top documents based on the queries themselves in the beginning. For example, REALM [166] adopts this strategy by directly attending the document contents to the original queries to predict the final answers based on masked language modeling. RAG [167] follows this strategy but applies the generative language modeling paradigm. However, these two approaches only use language models with limited parameters, such as BERT and BART. Recent approaches such as REPLUG [168] and Atlas [169] have improved them by leveraging LLMs such as GPTs, T5s, and LLaMAs for response generation. To yield better answer generation performances, these models usually fine-tune LLMs on QA tasks. However, due to the limited computing resources, many methods [170, 171, 179] choose to prompt LLMs for generation as they could use larger LMs in this way. Furthermore, to improve the quality of the generated answers, several approaches [172, 181] also try to train or prompt the LLMs to generate contexts such as citations or notes in addition to the answers to force LLMs to understand and assess the relevance of retrieved passages to the user queries. Some approaches [180] evaluate the importance of each retrieved reference using policy gradients to indicate which reference is more useful for generating. Besides, researchers explore instruction tuning LLMs such LLaMAs to improve their abilities to generate conclusive passages relying on retrieved knowledge [182, 183].

### 6.1.2 Periodic-Retrieval Reader

However, while generating long conclusive answers, it is shown [23, 173] that only using the references retrieved by the original user intents as in once-retrieval readers may be inadequate. For example, when providing a passage about "Barack Obama", language models may need additional knowledge about his university, which may not be included in the results of simply searching the initial query. In conclusion, language models may need extra references to support the following generation during the generating process, where multiple retrieval processes may be required. To address this, solutions such as RETRO [23] and RALM [173] have emerged, emphasizing the periodic collection of documents based on both the original queries and the concurrently generated texts (triggering a retrieval every $n$ generated tokens). In this manner, when generating the text about the university career of Barack Obama, the LLM can receive additional documents as supplementary materials. This need for additional references highlights the necessity for multiple retrieval iterations to ensure robustness in subsequent answer generation. Notably, RETRO [23] introduces a novel approach incorporating cross-attention between the generating texts and the references within the Transformer attention calculation, as opposed to directly embedding references into the input texts of LLMs. Since it involves additional cross-attention modules in the Transformer's structure, RETRO trains this model from scratch. However, these two approaches mainly rely on the successive $n$ tokens to separate generation and retrieve documents, which may not be semantically continuous and may cause the collected references noisy and useless. To solve this problem, some approaches such as IRCoT [175] also explore retrieving documents for every generated sentence, which is a more complete semantic structure. Furthermore, researchers find that the whole generated passages can be considered as conclusive contexts for current queries and can be used to find more relevant knowledge to generate more thorough answers. Consequently, many recent approaches [174, 184, 185] have also tried to extend this periodic-retrieval paradigm to iteratively using the whole generated passages to retrieve references to re-generate the

TABLE 7. The comparison of existing representative methods that have a passive reader module. REALM and RAG do not use LLMs, but their frameworks have been widely applied in many following approaches.

| Methods | Backbone models | Where to incorporate retrieval | When to retrieve | How to use LLMs |
| :---: | :---: | :---: | :---: | :---: |
| REALM [166] | BERT | Input layer | In the beginning | Fine-tuning |
| RAG [167] | BART | Input layer | In the beginning | Fine-tuning |
| REPLUG [168] | GPT | Input layer | In the beginning | Fine-tuning |
| Lazaridou et al. [170] | Gopher | Input layer | In the beginning | Prompting |
| He et al. [171] | GPT | Input layer | In the beginning | Prompting |
| Chain-of-Note [172] | LLaMA | Input layer | In the beginning | Fine-tuning |
| RALM [173] | LLaMA \& OPT \& GPT | Input laver | During generation (every $n$ tokens) | Prompting |
| IRCoT $[175]$ | Flan-T5 \& GPT | Input layer | During generation (every sentence) | Prompting |
| FLARE [176] | GPT | Input layer | During generation (aperiodic) | Prompting |
| Self-RAG [177] | LLaMA | Input layer | During generation (aperiodic) | Fine-tuning |

answers, until the iterations reach a pre-defined limitation. Particularly, these methods can be regarded as special periodic-retrieval readers that retrieve passages when every answer is (re)-generated. Since the LLMs can receive more comprehensive and relevant references with the iterations increase, these methods that combine retrieval-augmentedgeneration and generation-augmented-retrieval strategies can generate more accurate answers but consume more computation costs.

### 6.1.3 Aperiodic-Retrieval Reader

In the above strategy, the retrieval systems supply documents to LLMs in a periodic manner. However, retrieving documents in a mandatory frequency may mismatch the retrieval timing and can be costly. Recently, FLARE [176] has addressed this problem by automatically determining the timing of retrieval according to the probability of generating texts. Since the probability can serve as an indicator of LLMs' confidence during text generation [186, 187], a low probability for a generated term could suggest that LLMs require additional knowledge. Specifically, when the probability of a term falls below a predefined threshold, FLARE employs IR systems to retrieve references in accordance with the ongoing generated sentences, while removing these low-probability terms. FLARE adopts this strategy of prompting LLMs for answer generation solely based on the probabilities of generating terms, avoiding the need for finetuning while still maintaining effectiveness. Besides, selfRAG [177] tends to solve this problem by training LLMs such as LlaMA to generate specific tokens when they need additional knowledge to support following generations. Another critical model is introduced to judge whether the retrieved references are beneficial for generating.

We summarize representative passive reader approaches in Table 7, considering various aspects such as the backbone language models, the insertion point for retrieved references, the timing of using retrieval models, and the tuning strategy employed for LLMs.

### 6.2 Active Reader

However, the passive reader-based approaches separate IR systems and generative language models. This signifies that LLMs can only submissively utilize references provided by IR systems and are unable to interactively engage with the
IR systems in a manner akin to human interaction such as issuing queries to seek information.

To allow LLMs to actively use search engines, SelfAsk [188] and DSP [189] try to employ few-shot prompts for LLMs, triggering them to search queries when they believe it is required. For example, in a scenario where the query is "When was the existing tallest wooden lattice tower built?", these prompted LLMs can decide to search a query "What is the existing tallest wooden lattice tower" to gather necessary references as they find the query cannot be directly answered. Once acquired information about the tower, they can iteratively query IR systems for more details until they determine to generate the final answers instead of asking questions. Notably, these methods involve IR systems to construct a single reasoning chain for LLMs. MRC [190] further improves these methods by prompting LLMs to explore multiple reasoning chains and subsequently combining all generated answers using LLMs.

### 6.3 Compressor

Existing LLMs, especially open-sourced ones, such as LLaMA and Flan-T5, have limited input lengths (usually 4,096 or 8,192 tokens). However, the documents or web pages retrieved by upstream IR systems are usually long. Therefore, it is difficult to concatenate all the retrieved documents and feed them into LLMs to generate answers. Though some approaches manage to solve these problems by aggregating the answers supported by each reference as the final answers, this strategy neglects the potential relations between retrieved passages. A more straightforward way is to directly compress the retrieved documents into short input tokens or even dense vectors [191-194].

To compress the retrieved references, an intuitive idea is to extract the most useful $K$ sentences from the retrieved documents. LeanContext [191] applies this method and trains a small model by reinforcement learning (RL) to select the top $K$ similar sentences to the queries. The researchers also augment this strategy by using a free open-sourced text reduction method for the rest sentences as a supplement. Instead of using RL-based methods, RECOMP [192] directly uses the probability or the match ratio of the generated answers to the golden answers as signals to build training datasets and tune the compressor model. For example, the sentence corresponding to the highest generating proba-
bility is the positive one while others are negative ones. Furthermore, FILCO [193] applies the "hindsight" methods, which directly align the prior distribution (the predicted importance probability distribution of sentences without knowing the gold answer) to the posterior distribution (the same distribution of sentences within knowing the gold answer) to tune language models to select sentences.

However, these extractive methods may lose potential intent among all references. Therefore, abstractive methods are proposed to summarize retrieved documents into short but concise summaries for downstream generation. These methods [192, 194] usually distill the summarizing abilities of LLMs to small models. For example, TCRA [194] leverages GPT-3.5-turbo to build abstractive compression datasets for MT5 model.

### 6.4 Analysis

With the rapid development of the above reader approaches, many researchers have begun to analyze the characteristics of retrieval-augmented LLMs:

- Liu et al. [195] find that the position of the relevant/golden reference has significant influences on the final generation performance. The performance is always better when the relevant reference is at the beginning or the end, which indicates the necessity of introducing a ranking module to order the retrieved knowledge.
- Ren et al. [196] observe that by applying retrieval augmentation generation strategy, LLMs can have a better awareness of their knowledge boundaries.
- Liu et al. [197] analyze different strategies of integrating retrieval systems and LLMs such as concatenate (i.e., concatenating all references for answer generation) and post fusion (i.e., aggregating the answers corresponding to each reference). They also explore several ways of combining these two strategies.
- Aksitov et al. [198] demonstrate that there exists an attribution and fluency tradeoff for retrieval-augmented LLMs: with more received references, the attribution of generated answers increases while the fluency decreases.
- Mallen et al. [199] argue that always retrieving references to support LLMs to generate answers hurts the question-answering performance. The reason is that LLMs themselves may have adequate knowledge while answering questions about popular entities and the retrieved noisy passages may interfere and bias the answering process. To overcome this challenge, they devise a simple strategy that only retrieves references while the popularity of entities in the query is quite low. By this means, the efficacy and efficiency of retrieval-augmented generation both improve.


### 6.5 Applications

Recently, researchers [200-205] have applied the retrievalaugmented generation strategy to areas such as clinical QA, medical QA, and financial QA to enhance LLMs with external knowledge and to develop domain-specific applications. For example, ATLANTIC [201] adapts Atlas to the scientific domain to derive a science QA system. Besides, some approaches [206] also apply techniques in federated learning such as multi-party computation to perform personal retrieval-augmented generation with privacy protection.
Furthermore, to better facilitate the deployment of these retrieval-augmented generation systems, some tools or frameworks are proposed [178, 207, 208]. For example, RETA-LLM [178] breaks down the whole complex generation task into several simple modules in the reader pipeline. These modules include a query rewriting module for refining query intents, a passage extraction module for aligning reference lengths with LLM limitations, and a fact verification module for confirming the absence of fabricated information in the generated answers.

### 6.6 Limitations

Several IR systems applying the retrieval-augmented generation strategy, such as New Bing and Langchain, have already entered commercial use. However, there are also some challenges in this novel retrieval-augmented content generation system. These include challenges such as effective query reformulation, optimal retrieval frequency, correct document comprehension, accurate passage extraction, and effective content summarization. It is crucial to address these challenges to effectively realize the potential of LLMs in this paradigm.

## 7 SeARCH AGENT

With the development of LLMs, IR systems are also facing new changes. Among them, developing LLMs as intelligent agents has attracted more and more attention. This conceptual shift aims to mimic human browsing patterns, thereby enhancing the capability of these models to handle complex retrieval tasks. Empowered by the advanced natural language understanding and generation capabilities of LLMs, these agents can autonomously search, interpret, and synthesize information from a wide range of sources.

One way to achieve this ability is to design a pipeline that combines a series of modules and assigns different roles to them. Such a pre-defined pipeline mimics users' behaviors on the web by breaking it into several sub-tasks which are performed by different modules. However, this kind of static agent cannot deal with the complex nature of users' behavior sequences on the web and may face challenges when interacting with real-world environments. An alternative solution is to allow LLMs to freely explore the web and make interactions themselves, namely letting the LLM itself decide what action it will take next based on the feedback from the environment (or humans). These agents have more flexibility and act more like human beings.

### 7.1 Static Agent

To mimic human search patterns, a straightforward approach is to design a static system to browse the web and synthesize information step by step [209-214]. By breaking the information-seeking process into multiple subtasks, they design a pipeline that contains various LLM-based modules in advance and assigns different subtasks to them.

LaMDA [209] serves as an early work of the static agent. It consists of a family of Transformer-based neural language models specialized for dialog, with up to 137B parameters, pre-trained on 1.56T tokens from public dialogue data and web text. The study emphasizes the model's development
through a static pipeline, encompassing large-scale pretraining, followed by strategic fine-tuning stages aimed at enhancing three critical aspects: dialogue quality, safety, and groundedness. It can integrate external IR systems for factual grounding. This integration allows LaMDA to access and use external and authoritative sources when generating responses. SeeKeR [210] also incorporates the Internet search into its modular architecture for generating more factual responses. It performs three sequential tasks: generating a search query, generating knowledge from search results, and generating a final response. GopherCite [213] uses a search engine like Google Search to find relevant sources. It then synthesizes a response that includes verbatim quotes from these sources as evidence, aligning the Gopher's output with verified information. WebAgent [212] develops a series of tasks, including instruction decomposition and planning, action programming, and HTML summarization. It can navigate the web, understand and synthesize information from multiple sources, and execute web-based tasks, effectively functioning as an advanced search and interaction agent. WebGLM [211] designs an LLM-augmented retriever, a bootstrapped generator, and a human preferenceaware scorer. These components work together to provide accurate web-enhanced question-answering capabilities that are sensitive to human preferences. Shi et al. [214] focus on enhancing the relevance, responsibility, and trustworthiness of LLMs in web search applications via an intent-aware generator, an evidence-sensitive validator, and a multi-strategy supported optimizer.

### 7.2 Dynamic Agent

Instead of statically arranging LLMs in a pipeline, WebGPT [24] takes an alternate approach by training LLMs to use search engines automatically. This is achieved through the application of a reinforcement learning framework, within which a simulated environment is constructed for GPT-3 models. Specifically, the WebGPT model employs special tokens to execute actions such as querying, scrolling through rankings, and quoting references on search engines. This innovative approach allows the GPT-3 model to use search engines for text generation, enhancing the reliability and real-time capability of the generated texts. A following study [215] has extended this paradigm to the domain of Chinese question answering. Besides, some works develop important benchmarks for interactive webbased agents [216-218]. For example, WebShop [217] aims to provide a scalable, interactive web-based environment for language understanding and decision-making, focusing on the task of online shopping. ASH (Actor-SummarizerHierarchical) prompting [219] significantly enhances the ability of LLMs on WebShop benchmark. It first takes a raw observation from the environment and produces a new, more meaningful representation that aligns with the specific goal. Then, it dynamically predicts the next action based on the summarized observation and the interaction history.

### 7.3 Limitations

Though the aspect of static search agents has been thoroughly studied, the literature on dynamic search agents remains limited. Some agents may lack mechanisms for real-time fact-checking or verification against authoritative sources, leading to the potential dissemination of misinformation. Moreover, since LLMs are trained on data from the Internet, they may inadvertently perpetuate biases present in the training data. This can lead to biased or offensive outputs and may collect unethical content from the web. Finally, as LLMs process user queries, there are concerns regarding user privacy and data security, especially if sensitive or personal information is involved in the queries.

## 8 FUTURE DIRECTION

In this survey, we comprehensively reviewed recent advancements in LLM-enhanced IR systems and discussed their limitations. Since the integration of LLMs into IR systems is still in its early stages, there are still many opportunities and challenges. In this section, we summarize the potential future directions in terms of the four modules in an IR system we just discussed, namely query rewriter, retriever, reranker, and reader. In addition, as evaluation has also emerged as an important aspect, we will also introduce the corresponding research problems that need to be addressed in the future. Another discussion about important research topics on applying LLMs to IR can be found in a recent perspective paper [53].

### 8.1 Query Rewriter

LLMs have enhanced query rewriting for both ad-hoc and conversational search scenarios. Most of the existing methods rely on prompting LLMs to generate new queries. While yielding remarkable results, the refinement of rewriting quality and the exploration of potential application scenarios require further investigation.

- Rewriting queries according to ranking performance. A typical paradigm of prompting-based methods is providing LLMs with several ground-truth rewriting cases (optional) and the task description of query rewriting. Despite LLMs being capable of identifying potential user intents of the query [220], they lack awareness of the resulting retrieval quality of the rewritten query. The absence of this connection can result in rewritten queries that seem correct yet produce unsatisfactory ranking results. Although some existing studies have used reinforcement learning to adjust the query rewriting process according to generation results [100], a substantial realm of research remains unexplored concerning the integration of ranking results.
- Improving query rewriting in conversational search. As yet, primary efforts have been made to improve query rewriting in ad-hoc search. In contrast, conversational search presents a more developed landscape with a broader scope for LLMs to contribute to query understanding. By incorporating historical interactive information, LLMs can adapt system responses based on user preferences, providing a more effective conversational experience. However, this potential has not been explored in depth. In addition, LLMs could also be used to simulate user behavior in conversational search scenarios, providing more training data, which are urgently needed in current research.
- Achieving personalized query rewriting. LLMs offer valuable contributions to personalized search through their capacity to analyze user-specific data. In terms of query rewriting, with the excellent language comprehension ability of

LLMs, it is possible to leverage them to build user profiles based on users' search histories (e.g., issued queries, clickthrough behaviors, and dwell time). This empowers the achievement of personalized query rewriting for enhanced IR and finally benefits personalized search or personalized recommendation.

### 8.2 Retriever

Leveraging LLMs to improve retrieval models has received considerable attention, promising an enhanced understanding of queries and documents for improved ranking performance. However, despite strides in this field, several challenges and limitations still need to be investigated in the future:

- Reducing the latency of LLM-based retrievers. LLMs, with their massive parameters and world knowledge, often entail high latency during the inferring process. This delay poses a significant challenge for practical applications of LLM-based retrievers, as search engines require in-time responses. To address this issue, promising research directions include transferring the capabilities of LLMs to smaller models, exploring quantization techniques for LLMs in IR tasks, and so on.
- Simulating realistic queries for data augmentation. Since the high latency of LLMs usually blocks their online application for retrieval tasks, many existing studies have leveraged LLMs to augment training data, which is insensitive to inference latency. Existing methods that leverage LLMs for data augmentation often generate queries without aligning them with real user queries, leading to noise in the training data and limiting the effectiveness of retrievers. As a consequence, exploring techniques such as reinforcement learning to enable LLMs to simulate the way that real queries are issued holds the potential for improving retrieval tasks.
- Incremental indexing for generative retrieval. As elaborated in Section 4.2.2, the emergence of LLMs has paved the way for generative retrievers to generate document identifiers for retrieval tasks. This approach encodes document indexes and knowledge into the LLM parameters. However, the static nature of LLM parameters, coupled with the expensive fine-tuning costs, poses challenges for updating document indexes in generative retrievers when new documents are added. Therefore, it is crucial to explore methods for constructing an incremental index that allows for efficient updates in LLM-based generative retrievers.
- Supporting multi-modal search. Web pages usually contain multi-modal information, including texts, images, audios, and videos. However, existing LLM-enhanced IR systems mainly support retrieval for text-based content. A straightforward solution is to replace the backbone with multi-modal large models, such as GPT-4 [80]. However, this undoubtedly increases the cost of deployment. A promising yet challenging direction is to combine the language understanding capability of LLMs with existing multi-modal retrieval models. By this means, LLMs can contribute their language skills in handling different types of content.


### 8.3 Reranker

In Section 5, we have discussed the recent advanced techniques of utilizing LLMs for the reranking task. Some potential future directions in reranking are discussed as follows.

- Enhancing the online availability of LLMs. Though effective, many LLMs have a massive number of parameters, making it challenging to deploy them in online applications. Besides, many reranking methods $[152,153]$ rely on calling LLM APIs, incurring considerable costs. Consequently, devising effective approaches (such as distilling to small models) to enhance the online applicability of LLMs emerges as a research direction worth exploring.
- Improving personalized search. Many existing LLM-based reranking methods mainly focus on the ad-hoc reranking task. However, by incorporating user-specific information, LLMs can also improve the effectiveness of the personalized reranking task. For example, by analyzing users' search history, LLMs can construct accurate user profiles and rerank the search results accordingly, providing personalized results with higher user satisfaction.
- Adapting to diverse ranking tasks. In addition to document reranking, there are also other ranking tasks, such as response ranking, evidence ranking, entity ranking and etc., which also belong to the universal information access system. Navigating LLMs towards adeptness in these diverse ranking tasks can be achieved through specialized methodologies, such as instruction tuning. Exploring this avenue holds promise as an intriguing and valuable research trajectory.


### 8.4 Reader

With the increasing capabilities of LLMs, the future interaction between users and IR systems will be significantly changed. Due to the powerful natural language processing and understanding capabilities of LLMs, the traditional search paradigm of providing ranking results is expected to be progressively replaced by synthesizing conclusive answering passages for user queries using the reader module. Although such strategies have already been investigated by academia and facilitated by industry as we stated in Section 6, there still exists much room for exploration.

- Improving the reference quality for LLMs. To support answer generation, existing approaches usually directly feed the retrieved documents to the LLMs as references. However, since a document usually covers many topics, some passages in it may be irrelevant to the user queries and can introduce noise during LLMs' generation. Therefore, it is necessary to explore techniques for extracting relevant snippets from retrieved documents, enhancing the performance of retrieval-augmented generation.
- Improving the answer reliability of LLMs. Incorporating the retrieved references has significantly alleviated the "hallucination" problem of LLMs. However, it remains uncertain whether the LLMs refer to these supported materials during answering queries. Some studies [196] have revealed that LLMs can still provide unfaithful answers even with additional references. Therefore, the reliability of the conclusive answers might be lower compared to the ranking results provided by traditional IR systems. It is essential to investigate the influence of these references on the generation process, thereby improving the credibility of reader-based novel IR systems.


### 8.5 Search Agent

With the outstanding performance of LLMs, the patterns of searching may completely change from traditional IR systems to autonomous search agents. In Section 7, we have discussed many existing works that utilize a static or dynamic pipeline to autonomously browse the web. These works are believed to be the pioneering works of the new searching paradigm. However, there is still plenty of room for further improvements.

- Enhancing the Trustworthiness of LLMs. When LLMs are enabled to browse the web, it is important to ensure the validity of retrieved documents. Otherwise, the unfaithful information may increase the LLMs' "hallucination" problem. Besides, even if the gathered information has high quality, it remains unclear whether they are really used for synthesizing responses. A potential strategy to address this issue is enabling LLMs to autonomously validate the documents they scrape. This self-validation process could incorporate mechanisms for assessing the credibility and accuracy of the information within these documents.
- Mitigating Bias and Offensive Content in LLMs. The presence of biases and offensive content within LLM outputs is a pressing concern. This issue primarily stems from biases inherent in the training data and will be amplified by the lowquality information gathered from the web. Achieving this requires a multi-faceted approach, including improvements in training data, algorithmic adjustments, and continuous monitoring for bias and inappropriate content that LLMs collect and generate.


### 8.6 Evaluation

LLMs have attracted significant attention in the field of IR due to their strong ability in context understanding and text generation. To validate the effectiveness of LLM-enhanced IR approaches, it is crucial to develop appropriate evaluation metrics. Given the growing significance of readers as integral components of IR systems, the evaluation should consider two aspects: assessing ranking performance and evaluating generation performance.

- Generation-oriented ranking evaluation. Traditional evaluation metrics for ranking primarily focus on comparing the retrieval results of IR models with ground-truth (relevance) labels. Typical metrics include precision, recall, mean reciprocal rank (MRR) [221], mean average precision (MAP), and normalized discounted cumulative gain (nDCG) [222]. These metrics measure the alignment between ranking results and human preference on using these results. Nevertheless, these metrics may fall short in capturing a document's role in the generation of passages or answers, as their relevance to the query alone might not adequately reflect this aspect. This effect could be leveraged as a means to evaluate the usefulness of documents more comprehensively. A formal and rigorous evaluation metric for ranking that centers on generation quality has yet to be defined.
- Text generation evaluation. The wide application of LLMs in IR has led to a notable enhancement in their generation capability. Consequently, there is an imperative demand for novel evaluation strategies to effectively evaluate the performance of passage or answer generation. Previous evaluation metrics for text generation have several limitations, including: (1) Dependency on lexical matching: methods such as BLEU [223] or ROUGE [224] primarily evaluate the quality of generated outputs based on $n$-gram matching. This approach cannot account for lexical diversity and contextual semantics. As a result, models may favor generating common phrases or sentence structures rather than producing creative and novel content. (2) Insensitivity to subtle differences: existing evaluation methods may be insensitive to subtle differences in generated outputs. For example, if a generated output has minor semantic differences from the reference answer but is otherwise similar, traditional methods might overlook these nuanced distinctions. (3) Lack of ability to evaluate factuality: LLMs are prone to generating "hallucination" problems [225-228]. The hallucinated texts can closely resemble the oracle texts in terms of vocabulary usage, sentence structures, and patterns, while having nonfactual content. Existing methods are hard to identify such problems, while the incorporation of additional knowledge sources such as knowledge bases or reference texts could potentially aid in addressing this challenge.


### 8.7 Bias

Since ChatGPT was released, LLMs have drawn much attention from both academia and industry. The wide applications of LLMs have led to a notable increase in content on the Internet that is not authored by humans but rather generated by these language models. However, as LLMs may hallucinate and generate non-factual texts, the increasing number of LLM-generated contents also brings worries that these contents may provide fictitious information for users across IR systems. More severely, researchers [229, 230] show that some modules in IR systems such as retriever and reranker, especially those based on neural models, may prefer LLM-generated documents, since their topics are more consistent and the perplexity of them are lower compared with human-written documents. The authors refer to this phenomenon as the "source bias" towards LLM-generated text. It is challenging but necessary to consider how to build IR systems free from this category of bias.

## 9 CONCLUSION

In this survey, we have conducted a thorough exploration of the transformative impact of LLMs on IR across various dimensions. We have organized existing approaches into distinct categories based on their functions: query rewriting, retrieval, reranking, and reader modules. In the domain of query rewriting, LLMs have demonstrated their effectiveness in understanding ambiguous or multi-faceted queries, enhancing the accuracy of intent identification. In the context of retrieval, LLMs have improved retrieval accuracy by enabling more nuanced matching between queries and documents, considering context as well. Within the reranking realm, LLM-enhanced models consider more finegrained linguistic nuances when re-ordering results. The incorporation of reader modules in IR systems represents a significant step towards generating comprehensive responses instead of mere document lists. The integration of LLMs into IR systems has brought about a fundamental change in how users engage with information and knowledge. From query rewriting to retrieval, reranking, and
reader modules, LLMs have enriched each aspect of the IR process with advanced linguistic comprehension, semantic representation, and context-sensitive handling. As this field continues to progress, the journey of LLMs in IR portends a future characterized by more personalized, precise, and user-centric search encounters.

This survey focuses on reviewing recent studies of applying LLMs to different IR components and using LLMs as search agents. Beyond this, a more significant problem brought by the appearance of LLMs is: is the conventional IR framework necessary in the era of LLMs? For example, traditional IR aims to return a ranking list of documents that are relevant to issued queries. However, the development of generative language models has introduced a novel paradigm: the direct generation of answers to input questions. Furthermore, according to a recent perspective paper [53], IR might evolve into a fundamental service for diverse systems. For example, in a multi-agent simulation system [231], an IR component can be used for memory recall. This implies that there will be many new challenges in future IR.

## REFERENCES

[1] Y. Wu, W. Wu, C. Xing, M. Zhou, and Z. Li, "Sequential matching network: A new architecture for multi-turn response selection in retrieval-based chatbots," in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, R. Barzilay and M. Kan, Eds. Association for Computational Linguistics, 2017, pp. 496-505.

[2] H. Shum, X. He, and D. Li, "From eliza to xiaoice: challenges and opportunities with social chatbots," Frontiers Inf. Technol. Electron. Eng., vol. 19, no. 1, pp. 10-26, 2018.

[3] V. Karpukhin, B. Oguz, S. Min, P. S. H. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Association for Computational Linguistics, 2020, pp. 6769-6781.

[4] R. Datta, D. Joshi, J. Li, and J. Z. Wang, "Image retrieval: Ideas, influences, and trends of the new age," ACM Comput. Surv., vol. 40, no. 2, pp. 5:1-5:60, 2008.

[5] C. Yuan, W. Zhou, M. Li, S. Lv, F. Zhu, J. Han, and S. Hu, "Multi-hop selector network for multiturn response selection in retrieval-based chatbots," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 37, 2019, K. Inui, J. Jiang, V. Ng, and X. Wan, Eds. Association for Computational Linguistics, 2019, pp. 111-120.

[6] Y. Zhu, J. Nie, K. Zhou, P. Du, and Z. Dou, "Content selection network for document-grounded retrievalbased chatbots," in Advances in Information Retrieval - 43rd European Conference on IR Research, ECIR 2021,
Virtual Event, March 28 - April 1, 2021, Proceedings, Part I, ser. Lecture Notes in Computer Science, D. Hiemstra, M. Moens, J. Mothe, R. Perego, M. Potthast, and F. Sebastiani, Eds., vol. 12656. Springer, 2021, pp. $755-769$.

[7] Y. Zhu, J. Nie, K. Zhou, P. Du, H. Jiang, and Z. Dou, "Proactive retrieval-based chatbots based on relevant knowledge and goals," in SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, F. Diaz, C. Shah, T. Suel, P. Castells, R. Jones, and T. Sakai, Eds. ACM, 2021, pp. 20002004.

[8] H. Qian, Z. Dou, Y. Zhu, Y. Ma, and J. Wen, "Learning implicit user profiles for personalized retrieval-based chatbot," CoRR, vol. abs/2108.07935, 2021.

[9] Y. Qu, Y. Ding, J. Liu, K. Liu, R. Ren, W. X. Zhao, D. Dong, H. Wu, and H. Wang, "Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering," in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 611, 2021, K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tür, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, and Y. Zhou, Eds. Association for Computational Linguistics, 2021, pp. 5835-5847.

[10] Y. Arens, C. A. Knoblock, and W. Shen, "Query reformulation for dynamic information integration," $J$. Intell. Inf. Syst., vol. 6, no. 2/3, pp. 99-130, 1996.

[11] J. Huang and E. N. Efthimiadis, "Analyzing and evaluating query reformulation strategies in web search logs," in Proceedings of the 18th ACM Conference on Information and Knowledge Management, CIKM 2009, Hong Kong, China, November 2-6, 2009, D. W. Cheung, I. Song, W. W. Chu, X. Hu, and J. Lin, Eds. ACM, 2009, pp. 77-86.

[12] R. F. Nogueira, W. Yang, K. Cho, and J. Lin, "Multistage document ranking with BERT," CoRR, vol. abs/1910.14424, 2019.

[13] R. F. Nogueira, Z. Jiang, R. Pradeep, and J. Lin, “Document ranking with a pretrained sequence-to-sequence model," in EMNLP (Findings), ser. Findings of ACL, vol. EMNLP 2020. Association for Computational Linguistics, 2020, pp. 708-718.

[14] Y. Zhu, J. Nie, Z. Dou, Z. Ma, X. Zhang, P. Du, X. Zuo, and H. Jiang, "Contrastive learning of user behavior sequence for context-aware document ranking," in CIKM '21: The 30th ACM International Conference on Information and Knowledge Management, Virtual Event, Queensland, Australia, November 1 - 5, 2021, G. Demartini, G. Zuccon, J. S. Culpepper, Z. Huang, and H. Tong, Eds. ACM, 2021, pp. 2780-2791.

[15] J. Teevan, S. T. Dumais, and E. Horvitz, "Personalizing search via automated analysis of interests and activities," in SIGIR 2005: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Salvador, Brazil, August 15-19, 2005, R. A. Baeza-Yates, N. Ziviani, G. Marchionini, A. Moffat, and J. Tait, Eds. ACM, 2005, pp. 449-456.

[16] P. N. Bennett, R. W. White, W. Chu, S. T. Dumais, P. Bailey, F. Borisyuk, and X. Cui, "Modeling the impact of short- and long-term behavior on search personalization," in The 35th International ACM SIGIR conference on research and development in Information Retrieval, SIGIR '12, Portland, OR, USA, August 12-16, 2012, W. R. Hersh, J. Callan, Y. Maarek, and M. Sanderson, Eds. ACM, 2012, pp. 185-194.

[17] S. Ge, Z. Dou, Z. Jiang, J. Nie, and J. Wen, "Personalizing search results using hierarchical RNN with query-aware attention," in Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM 2018, Torino, Italy, October 22-26, 2018, A. Cuzzocrea, J. Allan, N. W. Paton, D. Srivastava, R. Agrawal, A. Z. Broder, M. J. Zaki, K. S. Candan, A. Labrinidis, A. Schuster, and H. Wang, Eds. ACM, 2018, pp. 347-356.

[18] Y. Zhou, Z. Dou, Y. Zhu, and J. Wen, "PSSL: selfsupervised learning for personalized search with contrastive sampling," in CIKM '21: The 30th ACM International Conference on Information and Knowledge Management, Virtual Event, Queensland, Australia, November 1 - 5, 2021, G. Demartini, G. Zuccon, J. S. Culpepper, Z. Huang, and H. Tong, Eds. ACM, 2021, pp. 27492758.

[19] J. G. Carbonell and J. Goldstein, "The use of mmr, diversity-based reranking for reordering documents and producing summaries," in SIGIR '98: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, August 24-28 1998, Melbourne, Australia, W. B. Croft, A. Moffat, C. J. van Rijsbergen, R. Wilkinson, and J. Zobel, Eds. ACM, 1998, pp. 335-336.

[20] R. Agrawal, S. Gollapudi, A. Halverson, and S. Ieong, "Diversifying search results," in Proceedings of the Second International Conference on Web Search and Web Data Mining, WSDM 2009, Barcelona, Spain, February 9-11, 2009, R. Baeza-Yates, P. Boldi, B. A. Ribeiro-Neto, and B. B. Cambazoglu, Eds. ACM, 2009, pp. 5-14.

[21] J. Liu, Z. Dou, X. Wang, S. Lu, and J. Wen, “DVGAN: A minimax game for search result diversification combining explicit and implicit features," in Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, J. X. Huang, Y. Chang, X. Cheng, J. Kamps, V. Murdock, J. Wen, and Y. Liu, Eds. ACM, 2020, pp. 479-488.

[22] Z. Su, Z. Dou, Y. Zhu, X. Qin, and J. Wen, "Modeling intent graph for search result diversification," in SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, F. Diaz, C. Shah, T. Suel, P. Castells, R. Jones, and T. Sakai, Eds. ACM, 2021, pp. 736-746.

[23] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J. Lespiau, B. Damoc, A. Clark, D. de Las Casas, A. Guy, J. Menick, R. Ring, T. Hennigan, S. Huang, L. Maggiore, C. Jones, A. Cassirer, A. Brock, M. Paganini, G. Irving, O. Vinyals, S. Osindero, K. Simonyan, J. W. Rae, E. Elsen, and L. Sifre, "Improv- ing language models by retrieving from trillions of tokens," in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 2022, pp. 2206-2240.

[24] R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, X. Jiang, K. Cobbe, T. Eloundou, G. Krueger, K. Button, M. Knight, B. Chess, and J. Schulman, "Webgpt: Browser-assisted question-answering with human feedback," CoRR, vol. abs/2112.09332, 2021.

[25] G. Salton and M. McGill, Introduction to Modern Information Retrieval. McGraw-Hill Book Company, 1984.

[26] G. Salton, A. Wong, and C. Yang, "A vector space model for automatic indexing," Commun. ACM, vol. 18, no. 11, pp. 613-620, 1975.

[27] F. Song and W. B. Croft, "A general language model for information retrieval," in Proceedings of the 1999 ACM CIKM International Conference on Information and Knowledge Management, Kansas City, Missouri, USA, November 2-6, 1999. ACM, 1999, pp. 316-321.

[28] J. Martineau and T. Finin, "Delta TFIDF: an improved feature space for sentiment analysis," in Proceedings of the Third International Conference on Weblogs and Social Media, ICWSM 2009, San Jose, California, USA, May 1720, 2009, E. Adar, M. Hurst, T. Finin, N. S. Glance, N. Nicolov, and B. L. Tseng, Eds. The AAAI Press, 2009.

[29] S. E. Robertson, S. Walker, S. Jones, M. HancockBeaulieu, and M. Gatford, "Okapi at TREC-3," in Proceedings of The Third Text REtrieval Conference, TREC 1994, Gaithersburg, Maryland, USA, November 2-4, 1994, ser. NIST Special Publication, D. K. Harman, Ed., vol. 500-225. National Institute of Standards and Technology (NIST), 1994, pp. 109-126.

[30] J. Guo, Y. Fan, Q. Ai, and W. B. Croft, "A deep relevance matching model for ad-hoc retrieval," in Proceedings of the 25th ACM International Conference on Information and Knowledge Management, CIKM 2016, Indianapolis, IN, USA, October 24-28, 2016, S. Mukhopadhyay, C. Zhai, E. Bertino, F. Crestani, J. Mostafa, J. Tang, L. Si, X. Zhou, Y. Chang, Y. Li, and P. Sondhi, Eds. ACM, 2016, pp. 55-64.

[31] L. Xiong, C. Xiong, Y. Li, K. Tang, J. Liu, P. N. Bennett, J. Ahmed, and A. Overwijk, "Approximate nearest neighbor negative contrastive learning for dense text retrieval," in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[32] J. Lin, R. F. Nogueira, and A. Yates, Pretrained Transformers for Text Ranking: BERT and Beyond, ser. Synthesis Lectures on Human Language Technologies. Morgan \& Claypool Publishers, 2021.

[33] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language models are unsupervised multitask learners," 2019.

[34] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger,

T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, "Language models are few-shot learners," in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., 2020.

[35] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, "Llama: Open and efficient foundation language models," CoRR, vol. abs/2302.13971, 2023.

[36] J. Zhang, R. Xie, Y. Hou, W. X. Zhao, L. Lin, and J. Wen, "Recommendation as instruction following: A large language model empowered recommendation approach," CoRR, vol. abs/2305.07001, 2023.

[37] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. J. McAuley, and W. X. Zhao, "Large language models are zeroshot rankers for recommender systems," CoRR, vol. abs /2305.08845, 2023.

[38] Y. Xi, W. Liu, J. Lin, J. Zhu, B. Chen, R. Tang, W. Zhang, R. Zhang, and Y. Yu, "Towards open-world recommendation with knowledge augmentation from large language models," CoRR, vol. abs/2306.10933, 2023.

[39] W. Fan, Z. Zhao, J. Li, Y. Liu, X. Mei, Y. Wang, J. Tang, and Q. Li, "Recommender systems in the era of large language models (llms)," CoRR, vol. abs/2307.02046, 2023.

[40] S. Wu, O. Irsoy, S. Lu, V. Dabravolski, M. Dredze, S. Gehrmann, P. Kambadur, D. S. Rosenberg, and G. Mann, "Bloomberggpt: A large language model for finance," CoRR, vol. abs/2303.17564, 2023.

[41] J. Li, Y. Liu, W. Fan, X. Wei, H. Liu, J. Tang, and Q. Li, "Empowering molecule discovery for moleculecaption translation with large language models: A chatgpt perspective," CoRR, vol. abs/2306.06615, 2023.

[42] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, "Emergent abilities of large language models," Trans. Mach. Learn. Res., vol. 2022, 2022.

[43] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe, "Training language models to follow instructions with human feedback," in NeurIPS, 2022.

[44] J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, "Finetuned language models are zero-shot learners," in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[45] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou, "Chain-of- thought prompting elicits reasoning in large language models," in NeurIPS, 2022.

[46] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, "Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing," ACM Comput. Surv., vol. 55, no. 9, pp. 195:1-195:35, 2023.

[47] X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, "Pre-trained models for natural language processing: A survey," CoRR, vol. abs/2003.08271, 2020.

[48] Y. Cao, S. Li, Y. Liu, Z. Yan, Y. Dai, P. S. Yu, and L. Sun, "A comprehensive survey of ai-generated content (AIGC): A history of generative AI from GAN to chatgpt," CoRR, vol. abs/2303.04226, 2023.

[49] J. Li, T. Tang, W. X. Zhao, and J. Wen, "Pretrained language model for text generation: A survey," in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, Z. Zhou, Ed. ijcai.org, 2021, pp. 4492-4499.

[50] Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, L. Li, and Z. Sui, “A survey for in-context learning," CoRR, vol. abs/2301.00234, 2023.

[51] J. Huang and K. C. Chang, "Towards reasoning in large language models: A survey," in Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. BoydGraber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 1049-1065.

[52] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, Y. Du, C. Yang, Y. Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, P. Liu, J. Nie, and J. Wen, "A survey of large language models," CoRR, vol. abs/2303.18223, 2023.

[53] Q. Ai, T. Bai, Z. Cao, Y. Chang, J. Chen, Z. Chen, Z. Cheng, S. Dong, Z. Dou, F. Feng, S. Gao, J. Guo, X. He, Y. Lan, C. Li, Y. Liu, Z. Lyu, W. Ma, J. Ma, Z. Ren, P. Ren, Z. Wang, M. Wang, J. Wen, L. Wu, X. Xin, J. Xu, D. Yin, P. Zhang, F. Zhang, W. Zhang, M. Zhang, and X. Zhu, "Information retrieval meets large language models: A strategic report from chinese IR community," CoRR, vol. abs/2307.09751, 2023.

[54] X. Liu and W. B. Croft, "Statistical language modeling for information retrieval," Annu. Rev. Inf. Sci. Technol., vol. 39, no. 1, pp. 1-31, 2005.

[55] B. Mitra and N. Craswell, "Neural models for information retrieval," CoRR, vol. abs/1705.01509, 2017.

[56] W. X. Zhao, J. Liu, R. Ren, and J. Wen, "Dense text retrieval based on pretrained language models: A survey," CoRR, vol. abs/2211.14876, 2022.

[57] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-totext transformer," J. Mach. Learn. Res., vol. 21, pp. 140:1-140:67, 2020.

[58] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, "Deep contextualized word representations," in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana,

USA, June 1-6, 2018, Volume 1 (Long Papers), M. A. Walker, H. Ji, and A. Stent, Eds. Association for Computational Linguistics, 2018, pp. 2227-2237.

[59] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: pre-training of deep bidirectional transformers for language understanding," in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds. Association for Computational Linguistics, 2019, pp. 4171-4186.

[60] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, Eds., 2017, pp. 59986008.

[61] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, "BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, D. Jurafsky, J. Chai, N. Schluter, and J. R. Tetreault, Eds. Association for Computational Linguistics, 2020, pp. 7871-7880.

[62] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," CoRR, vol. abs/2001.08361, 2020.

[63] A. Clark, D. de Las Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, B. Damoc, B. A. Hechtman, T. Cai, S. Borgeaud, G. van den Driessche, E. Rutherford, T. Hennigan, M. J. Johnson, A. Cassirer, C. Jones, E. Buchatskaya, D. Budden, L. Sifre, S. Osindero, O. Vinyals, M. Ranzato, J. W. Rae, E. Elsen, K. Kavukcuoglu, and K. Simonyan, "Unified scaling laws for routed language models," in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 2022, pp. 4057-4086.

[64] L. Dong, N. Yang, W. Wang, F. Wei, X. Liu, Y. Wang, J. Gao, M. Zhou, and H. Hon, "Unified language model pre-training for natural language understanding and generation," in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d'AlchéBuc, E. B. Fox, and R. Garnett, Eds., 2019, pp. 13 04213054 .

[65] L. Xue, N. Constant, A. Roberts, M. Kale, R. AlRfou, A. Siddhant, A. Barua, and C. Raffel, "mt5: A massively multilingual pre-trained text-to-text transformer," in Proceedings of the 2021 Confer- ence of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tür, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, and Y. Zhou, Eds. Association for Computational Linguistics, 2021, pp. 483-498.

[66] V. Sanh, A. Webson, C. Raffel, S. H. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey, M. S. Bari, C. Xu, U. Thakker, S. S. Sharma, E. Szczechla, T. Kim, G. Chhablani, N. V. Nayak, D. Datta, J. Chang, M. T. Jiang, H. Wang, M. Manica, S. Shen, Z. X. Yong, H. Pandey, R. Bawden, T. Wang, T. Neeraj, J. Rozen, A. Sharma, A. Santilli, T. Févry, J. A. Fries, R. Teehan, T. L. Scao, S. Biderman, L. Gao, T. Wolf, and A. M. Rush, "Multitask prompted training enables zero-shot task generalization," in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[67] H. Bao, L. Dong, F. Wei, W. Wang, N. Yang, X. Liu, Y. Wang, J. Gao, S. Piao, M. Zhou, and H. Hon, "Unilmv2: Pseudo-masked language models for unified language model pre-training," in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 642-652.

[68] A. Zeng, X. Liu, Z. Du, Z. Wang, H. Lai, M. Ding, Z. Yang, Y. Xu, W. Zheng, X. Xia, W. L. Tam, Z. Ma, Y. Xue, J. Zhai, W. Chen, Z. Liu, P. Zhang, Y. Dong, and J. Tang, "GLM-130B: an open bilingual pre-trained model," in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[69] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," J. Mach. Learn. Res., vol. 23, pp. 120:1-120:39, 2022.

[70] Z. Yang, Z. Dai, Y. Yang, J. G. Carbonell, R. Salakhutdinov, and Q. V. Le, "Xlnet: Generalized autoregressive pretraining for language understanding," in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. B. Fox, and R. Garnett, Eds., 2019, pp. 5754-5764.

[71] S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell, J. Phang, M. Pieler, U. S. Prashanth, S. Purohit, L. Reynolds, J. Tow, B. Wang, and S. Weinbach, "Gptneox-20b: An open-source autoregressive language model," CoRR, vol. abs/2204.06745, 2022.

[72] J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, H. F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. M.

Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J. Lespiau, M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d'Autume, Y. Li, T. Terzi, V. Mikulik, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, C. Jones, J. Bradbury, M. J. Johnson, B. A. Hechtman, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving, "Scaling language models: Methods, analysis \& insights from training gopher," CoRR, vol. abs/2112.11446, 2021.

[73] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph, L. Fedus, M. P. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. S. MeierHellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui, "Glam: Efficient scaling of language models with mixture-of-experts," in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 2022, pp. 5547-5569.

[74] Y. Sun, S. Wang, S. Feng, S. Ding, C. Pang, J. Shang, J. Liu, X. Chen, Y. Zhao, Y. Lu, W. Liu, Z. Wu, W. Gong, J. Liang, Z. Shang, P. Sun, W. Liu, X. Ouyang, D. Yu, H. Tian, H. Wu, and H. Wang, "ERNIE 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation," CoRR, vol. abs/2107.02137, 2021.

[75] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, "OPT: open pre-trained transformer language models," CoRR, vol. abs/2205.01068, 2022.

[76] R. Thoppilan, D. D. Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, Y. Li, H. Lee, H. S. Zheng, A. Ghafouri, M. Menegali, Y. Huang, M. Krikun, D. Lepikhin, J. Qin, D. Chen, Y. Xu, Z. Chen, A. Roberts, M. Bosma, Y. Zhou, C. Chang, I. Krivokon, W. Rusch, M. Pickett, K. S. Meier-Hellstern, M. R. Morris, T. Doshi, R. D. Santos, T. Duke, J. Soraker, B. Zevenbergen, V. Prabhakaran, M. Diaz, B. Hutchinson, K. Olson, A. Molina, E. Hoffman-John, J. Lee, L. Aroyo, R. Rajakumar, A. Butryna, M. Lamm, V. Kuzmina, J. Fenton, A. Cohen, R. Bernstein, R. Kurzweil, B. A. y Arcas, C. Cui, M. Croak, E. H. Chi, and Q. Le, "Lamda: Language models for dialog applications," CoRR, vol. abs/2201.08239, 2022.

[77] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Is- ard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. MeierHellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, "Palm: Scaling language modeling with pathways," CoRR, vol. abs/2204.02311, 2022.

[78] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon, M. Gallé, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot, N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan-Major, I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurençon, Y. Jernite, J. Launay, M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji, A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue, C. Klamm, C. Leong, D. van Strien, D. I. Adelani, and et al., "BLOOM: A 176b-parameter open-access multilingual language model," CoRR, vol. abs/2211.05100, 2022.

[79] A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra, "Solving quantitative reasoning problems with language models," in NeurIPS, 2022.

[80] OpenAI, "GPT-4 technical report," CoRR, vol. abs /2303.08774, 2023.

[81] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre, "Training compute-optimal large language models," CoRR, vol. abs/2203.15556, 2022.

[82] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "Lora: Low-rank adaptation of large language models," in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[83] X. L. Li and P. Liang, "Prefix-tuning: Optimizing continuous prompts for generation," in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 16, 2021, C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Association for Computational Linguistics, 2021, pp. 4582-4597.

[84] B. Lester, R. Al-Rfou, and N. Constant, "The power of scale for parameter-efficient prompt tuning," in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, M. Moens, X. Huang, L. Specia, and S. W. Yih, Eds. Association for Computational Linguistics, 2021,
pp. 3045-3059.

[85] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "Qlora: Efficient finetuning of quantized llms," CoRR, vol. abs/2305.14314, 2023.

[86] L. Wang, N. Yang, and F. Wei, "Query2doc: Query expansion with large language models," pp. 9414$9423,2023$.

[87] N. A. Jaleel, J. Allan, W. B. Croft, F. Diaz, L. S. Larkey, X. Li, M. D. Smucker, and C. Wade, "Umass at TREC 2004: Novelty and HARD," in Proceedings of the Thirteenth Text REtrieval Conference, TREC 2004, Gaithersburg, Maryland, USA, November 16-19, 2004, ser. NIST Special Publication, E. M. Voorhees and L. P. Buckland, Eds., vol. 500-261. National Institute of Standards and Technology (NIST), 2004.

[88] D. Metzler and W. B. Croft, "Latent concept expansion using markov random fields," in SIGIR 2007: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Amsterdam, The Netherlands, July 23-27, 2007, W. Kraaij, A. P. de Vries, C. L. A. Clarke, N. Fuhr, and N. Kando, Eds. ACM, 2007, pp. 311-318.

[89] C. Zhai and J. D. Lafferty, "Model-based feedback in the language modeling approach to information retrieval," in Proceedings of the 2001 ACM CIKM International Conference on Information and Knowledge Management, Atlanta, Georgia, USA, November 5-10, 2001. ACM, 2001, pp. 403-410.

[90] D. Metzler and W. B. Croft, "A markov random field model for term dependencies," in SIGIR 2005: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Salvador, Brazil, August 15-19, 2005, R. A. Baeza-Yates, N. Ziviani, G. Marchionini, A. Moffat, and J. Tait, Eds. ACM, 2005, pp. 472-479.

[91] X. Wang, C. Macdonald, N. Tonellotto, and I. Ounis, "Pseudo-relevance feedback for multiple representation dense retrieval," in ICTIR '21: The 2021 ACM SIGIR International Conference on the Theory of Information Retrieval, Virtual Event, Canada, July 11, 2021, F. Hasibi, Y. Fang, and A. Aizawa, Eds. ACM, 2021, pp. 297306.

[92] Z. Zheng, K. Hui, B. He, X. Han, L. Sun, and A. Yates, "BERT-QE: contextualized query expansion for document re-ranking," in Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, ser. Findings of ACL, T. Cohn, Y. He, and Y. Liu, Eds., vol. EMNLP 2020. Association for Computational Linguistics, 2020, pp. 4718-4728.

[93] F. Diaz, B. Mitra, and N. Craswell, "Query expansion with locally-trained word embeddings," in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics, 2016.

[94] S. Kuzi, A. Shtok, and O. Kurland, "Query expansion using word embeddings," in Proceedings of the 25th ACM International Conference on Information and Knowledge Management, CIKM 2016, Indianapolis, IN, USA, October 24-28, 2016, S. Mukhopadhyay, C. Zhai, E. Bertino, F. Crestani, J. Mostafa, J. Tang, L. Si,
X. Zhou, Y. Chang, Y. Li, and P. Sondhi, Eds. ACM, 2016, pp. 1929-1932.

[95] K. Mao, Z. Dou, F. Mo, J. Hou, H. Chen, and H. Qian, "Large language models know your contextual search intent: A prompting framework for conversational search," pp. 1211-1225, 2023.

[96] I. Mackie, I. Sekulic, S. Chatterjee, J. Dalton, and F. Crestani, "GRM: generative relevance modeling using relevance-aware sample estimation for document retrieval," CoRR, vol. abs/2306.09938, 2023.

[97] K. Srinivasan, K. Raman, A. Samanta, L. Liao, L. Bertelli, and M. Bendersky, "QUILL: query intent with large language models using retrieval augmentation and multi-stage distillation," in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: EMNLP 2022 - Industry Track, Abu Dhabi, UAE, December 7 - 11, 2022, Y. Li and A. Lazaridou, Eds. Association for Computational Linguistics, 2022, pp. 492-501.

[98] J. Feng, C. Tao, X. Geng, T. Shen, C. Xu, G. Long, D. Zhao, and D. Jiang, "Knowledge refinement via interaction between search engines and large language models," CoRR, vol. abs/2305.07402, 2023.

[99] I. Mackie, S. Chatterjee, and J. Dalton, "Generative and pseudo-relevant feedback for sparse, dense and learned sparse retrieval," CoRR, vol. abs/2305.07477, 2023.

[100] X. Ma, Y. Gong, P. He, H. Zhao, and N. Duan, "Query rewriting for retrieval-augmented large language models," CoRR, vol. abs/2305.14283, 2023.

[101] L. Gao, X. Ma, J. Lin, and J. Callan, "Precise zero-shot dense retrieval without relevance labels," CoRR, vol. $\mathrm{abs} / 2212.10496,2022$.

[102] R. Jagerman, H. Zhuang, Z. Qin, X. Wang, and M. Bendersky, "Query expansion by prompting large language models," CoRR, vol. abs/2305.03653, 2023.

[103] Y. Tang, R. Qiu, and X. Li, "Prompt-based effective input reformulation for legal case retrieval," in Databases Theory and Applications - 34th Australasian Database Conference, ADC 2023, Melbourne, VIC, Australia, November 1-3, 2023, Proceedings, ser. Lecture Notes in Computer Science, Z. Bao, R. Borovica-Gajic, R. Qiu, F. M. Choudhury, and Z. Yang, Eds., vol. 14386. Springer, 2023, pp. 87-100.

[104] F. Ye, M. Fang, S. Li, and E. Yilmaz, "Enhancing conversational search: Large language modelaided informative query rewriting," arXiv preprint arXiv:2310.09716, 2023.

[105] C. Huang, C. Hsu, T. Hsu, C. Li, and Y. Chen, "CONVERSER: few-shot conversational dense retrieval with synthetic data generation," in Proceedings of the 24th Meeting of the Special Interest Group on Discourse and Dialogue, SIGDIAL 2023, Prague, Czechia, September 11 - 15, 2023, D. Schlangen, S. Stoyanchev, S. Joty, O. Dusek, C. Kennington, and M. Alikhani, Eds. Association for Computational Linguistics, 2023, pp. 381-387.

[106] M. Li, H. Zhuang, K. Hui, Z. Qin, J. Lin, R. Jagerman, X. Wang, and M. Bendersky, "Generate, filter, and fuse: Query expansion via multi-step keyword generation for zero-shot neural rankers," CoRR, vol.
abs/2311.09175, 2023.

[107] A. Anand, V. V, V. Setty, and A. Anand, "Context aware query rewriting for text rankers using LLM," CoRR, vol. abs/2308.16753, 2023.

[108] T. Shen, G. Long, X. Geng, C. Tao, T. Zhou, and D. Jiang, "Large language models are strong zero-shot retriever," CoRR, vol. abs/2304.14233, 2023.

[109] M. Alaofi, L. Gallagher, M. Sanderson, F. Scholer, and P. Thomas, "Can generative llms create query variants for test collections? an exploratory study," in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, H. Chen, W. E. Duh, H. Huang, M. P. Kato, J. Mothe, and B. Poblete, Eds. ACM, 2023, pp. 1869-1873.

[110] W. Yu, D. Iter, S. Wang, Y. Xu, M. Ju, S. Sanyal, C. Zhu, M. Zeng, and M. Jiang, "Generate rather than retrieve: Large language models are strong context generators," in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[111] T. Nguyen, M. Rosenberg, X. Song, J. Gao, S. Tiwary, R. Majumder, and L. Deng, "MS MARCO: A human generated machine reading comprehension dataset," in CoCo@NIPS, ser. CEUR Workshop Proceedings, vol. 1773. CEUR-WS.org, 2016.

[112] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. P. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov, "Natural questions: a benchmark for question answering research," Trans. Assoc. Comput. Linguistics, vol. 7, pp. 452-466, 2019.

[113] W. Peng, G. Li, Y. Jiang, Z. Wang, D. Ou, X. Zeng, D. $\mathrm{Xu}, \mathrm{T}$. Xu, and E. Chen, "Large language model based long-tail query rewriting in taobao search," CoRR, vol. abs/2311.03758, 2023.

[114] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, "GLM: general language model pretraining with autoregressive blank infilling," in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, S. Muresan, P. Nakov, and A. Villavicencio, Eds. Association for Computational Linguistics, 2022, pp. 320-335.

[115] A. Yang, B. Xiao, B. Wang, B. Zhang, C. Bian, C. Yin, C. Lv, D. Pan, D. Wang, D. Yan, F. Yang, F. Deng, F. Wang, F. Liu, G. Ai, G. Dong, H. Zhao, H. Xu, H. Sun, H. Zhang, H. Liu, J. Ji, J. Xie, J. Dai, K. Fang, L. Su, L. Song, L. Liu, L. Ru, L. Ma, M. Wang, M. Liu, M. Lin, N. Nie, P. Guo, R. Sun, T. Zhang, T. Li, T. Li, W. Cheng, W. Chen, X. Zeng, X. Wang, X. Chen, X. Men, X. Yu, X. Pan, Y. Shen, Y. Wang, Y. Li, Y. Jiang, Y. Gao, Y. Zhang, Z. Zhou, and Z. Wu, "Baichuan 2: Open large-scale language models," CoRR, vol. abs/2309.10305, 2023.

[116] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang,
S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu, "Qwen technical report," CoRR, vol. abs/2309.16609, 2023.

[117] D. Alexander, W. Kusa, and A. P. de Vries, “ORCASI: queries annotated with intent using weak supervision," in SIGIR '22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022, E. Amigó, P. Castells, J. Gonzalo, B. Carterette, J. S. Culpepper, and G. Kazai, Eds. ACM, 2022, pp. 3057-3066.

[118] K. D. Dhole, R. Chandradevan, and E. Agichtein, "An interactive query generation assistant using llm-based prompt modification and user feedback," CoRR, vol. abs/2311.11226, 2023.

[119] O. Weller, K. Lo, D. Wadden, D. J. Lawrie, B. V. Durme, A. Cohan, and L. Soldaini, "When do generative query and document expansions fail? A comprehensive study across methods, retrievers, and datasets," CoRR, vol. abs/2309.08541, 2023.

[120] L. H. Bonifacio, H. Abonizio, M. Fadaee, and R. F. Nogueira, "Inpars: Data augmentation for information retrieval using large language models," CoRR, vol. abs/2202.05144, 2022.

[121] G. Ma, X. Wu, P. Wang, Z. Lin, and S. Hu, "Pretraining with large language model-based document expansion for dense passage retrieval," CoRR, vol. abs/2308.08285, 2023.

[122] V. Jeronymo, L. H. Bonifacio, H. Abonizio, M. Fadaee, R. de Alencar Lotufo, J. Zavrel, and R. F. Nogueira, "Inpars-v2: Large language models as efficient dataset generators for information retrieval," CoRR, vol. abs/2301.01820, 2023.

[123] Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. B. Hall, and M. Chang, "Promptagator: Few-shot dense retrieval from 8 examples," in ICLR. OpenReview.net, 2023.

[124] R. Meng, Y. Liu, S. Yavuz, D. Agarwal, L. Tu, N. Yu, J. Zhang, M. Bhat, and Y. Zhou, "Augtriever: Unsupervised dense retrieval by scalable data augmentation," 2023.

[125] J. Saad-Falcon, O. Khattab, K. Santhanam, R. Florian, M. Franz, S. Roukos, A. Sil, M. A. Sultan, and C. Potts, "UDAPDR: unsupervised domain adaptation via LLM prompting and distillation of rerankers," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 11265-11279.

[126] Z. Peng, X. Wu, and Y. Fang, "Soft prompt tuning for augmenting dense retrieval with large language models," 2023.

[127] D. S. Sachan, M. Lewis, D. Yogatama, L. Zettlemoyer, J. Pineau, and M. Zaheer, "Questions are all you need to train a dense passage retriever," Transactions of the Association for Computational Linguistics, vol. 11, pp. 600-616, 2023.

[128] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models,"
in NeurIPS Datasets and Benchmarks, 2021.

[129] N. Thakur, J. Ni, G. H. Ábrego, J. Wieting, J. Lin, and D. Cer, "Leveraging llms for synthesizing training data across many languages in multilingual dense retrieval," CoRR, vol. abs/2311.05800, 2023.

[130] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, J. Heidecke, P. Shyam, B. Power, T. E. Nekoul, G. Sastry, G. Krueger, D. Schnurr, F. P. Such, K. Hsu, M. Thompson, T. Khan, T. Sherbakov, J. Jang, P. Welinder, and L. Weng, "Text and code embeddings by contrastive pre-training," CoRR, vol. abs/2201.10005, 2022.

[131] X. Ma, L. Wang, N. Yang, F. Wei, and J. Lin, "Finetuning llama for multi-stage text retrieval," CoRR, vol. abs/2310.08319, 2023.

[132] A. Asai, T. Schick, P. S. H. Lewis, X. Chen, G. Izacard, S. Riedel, H. Hajishirzi, and W. Yih, "Task-aware retrieval with instructions," in Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 3650-3675.

[133] J. Ni, C. Qu, J. Lu, Z. Dai, G. H. Ábrego, J. Ma, V. Y. Zhao, Y. Luan, K. B. Hall, M. Chang, and Y. Yang, "Large dual encoders are generalizable retrievers," in EMNLP. Association for Computational Linguistics, 2022, pp. 9844-9855.

[134] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave, "Unsupervised dense information retrieval with contrastive learning," Trans. Mach. Learn. Res., vol. 2022, 2022.

[135] D. Metzler, Y. Tay, D. Bahri, and M. Najork, "Rethinking search: making domain experts out of dilettantes," SIGIR Forum, vol. 55, no. 1, pp. 13:1-13:27, 2021.

[136] Y. Zhou, J. Yao, Z. Dou, L. Wu, and J. Wen, "Dynamicretriever: A pre-trained model-based IR system without an explicit index," Mach. Intell. Res., vol. 20, no. 2, pp. 276-288, 2023.

[137] J. Chen, R. Zhang, J. Guo, Y. Liu, Y. Fan, and X. Cheng, "Corpusbrain: Pre-train a generative retrieval model for knowledge-intensive language tasks," in Proceedings of the 31st ACM International Conference on Information $\mathcal{E}$ Knowledge Management, Atlanta, GA, USA, October 17-21, 2022, M. A. Hasan and L. Xiong, Eds. ACM, 2022, pp. 191-200.

[138] Y. Tay, V. Tran, M. Dehghani, J. Ni, D. Bahri, H. Mehta, Z. Qin, K. Hui, Z. Zhao, J. P. Gupta, T. Schuster, W. W. Cohen, and D. Metzler, "Transformer memory as a differentiable search index," in NeurIPS, 2022.

[139] N. Ziems, W. Yu, Z. Zhang, and M. Jiang, "Large language models are built-in autoregressive search engines," in Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 2666-2678.

[140] R. F. Nogueira, W. Yang, K. Cho, and J. Lin, "Multistage document ranking with BERT," CoRR, vol. abs/1910.14424, 2019.

[141] J. Ju, J. Yang, and C. Wang, "Text-to-text multi-view learning for passage re-ranking," in SIGIR. ACM, 2021, pp. 1803-1807.

[142] R. Pradeep, R. F. Nogueira, and J. Lin, "The expandomono-duo design pattern for text ranking with pretrained sequence-to-sequence models," CoRR, vol. $\mathrm{abs} / 2101.05667,2021$.

[143] H. Zhuang, Z. Qin, R. Jagerman, K. Hui, J. Ma, J. Lu, J. Ni, X. Wang, and M. Bendersky, "Rankt5: Finetuning T5 for text ranking with ranking losses," in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, H. Chen, W. E. Duh, H. Huang, M. P. Kato, J. Mothe, and B. Poblete, Eds. ACM, 2023, pp. 2308-2313.

[144] L. Zhang, Y. Zhang, D. Long, P. Xie, M. Zhang, and M. Zhang, "Rankinggpt: Empowering large language models in text ranking with progressive enhancement," CoRR, vol. abs/2311.16720, 2023.

[145] X. Zhang, S. Hofstätter, P. Lewis, R. Tang, and J. Lin, "Rank-without-gpt: Building gpt-independent listwise rerankers on open-source large language models," arXiv preprint arXiv:2312.02969, 2023.

[146] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, B. Newman, B. Yuan, B. Yan, C. Zhang, C. Cosgrove, C. D. Manning, C. Ré, D. Acosta-Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong, H. Ren, H. Yao, J. Wang, K. Santhanam, L. J. Orr, L. Zheng, M. Yüksekgönül, M. Suzgun, N. Kim, N. Guha, N. S. Chatterji, O. Khattab, P. Henderson, Q. Huang, R. Chi, S. M. Xie, S. Santurkar, S. Ganguli, T. Hashimoto, T. Icard, T. Zhang, V. Chaudhary, W. Wang, X. Li, Y. Mai, Y. Zhang, and Y. Koreeda, "Holistic evaluation of language models," CoRR, vol. abs/2211.09110, 2022.

[147] H. Zhuang, Z. Qin, K. Hui, J. Wu, L. Yan, X. Wang, and M. Bendersky, "Beyond yes and no: Improving zeroshot LLM rankers via scoring fine-grained relevance labels," CoRR, vol. abs/2310.14122, 2023.

[148] D. S. Sachan, M. Lewis, M. Joshi, A. Aghajanyan, W. Yih, J. Pineau, and L. Zettlemoyer, "Improving passage retrieval with zero-shot question generation," in EMNLP. Association for Computational Linguistics, 2022, pp. 3781-3797.

[149] S. Zhuang, B. Liu, B. Koopman, and G. Zuccon, "Open-source large language models are strong zeroshot query likelihood models for document ranking," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 8807-8817.

[150] S. Cho, S. Jeong, J. Seo, and J. C. Park, "Discrete prompt optimization via constrained generation for zero-shot re-ranker," in ACL (Findings). Association for Computational Linguistics, 2023, pp. 960-971.

[151] A. Drozdov, H. Zhuang, Z. Dai, Z. Qin, R. Rahimi, X. Wang, D. Alon, M. Iyyer, A. McCallum, D. Metzler, and K. Hui, "PaRaDe: Passage ranking using demonstrations with LLMs," in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association
for Computational Linguistics, Dec. 2023, pp. $14242-$ 14252.

[152] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, "Is chatgpt good at search? investigating large language models as re-ranking agents," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 14918-14937.

[153] X. Ma, X. Zhang, R. Pradeep, and J. Lin, “Zero-shot listwise document reranking with a large language model," CoRR, vol. abs/2305.02156, 2023.

[154] R. Tang, X. Zhang, X. Ma, J. Lin, and F. Ture, "Found in the middle: Permutation self-consistency improves listwise ranking in large language models," CoRR, vol. abs $/ 2310.07712,2023$.

[155] Z. Qin, R. Jagerman, K. Hui, H. Zhuang, J. Wu, J. Shen, T. Liu, J. Liu, D. Metzler, X. Wang et al., "Large language models are effective text rankers with pairwise ranking prompting," arXiv preprint arXiv:2306.17563, 2023.

[156] S. Zhuang, H. Zhuang, B. Koopman, and G. Zuccon, "A setwise approach for effective and highly efficient zero-shot ranking with large language models," CoRR, vol. abs/2310.09497, 2023.

[157] F. Ferraretto, T. Laitz, R. de Alencar Lotufo, and R. F. Nogueira, "Exaranker: Synthetic explanations improve neural rankers," in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, H. Chen, W. E. Duh, H. Huang, M. P. Kato, J. Mothe, and B. Poblete, Eds. ACM, 2023, pp. 2409-2414.

[158] L. Boytsov, P. Patel, V. Sourabh, R. Nisar, S. Kundu, R. Ramanathan, and E. Nyberg, "Inpars-light: Costeffective unsupervised training of efficient rankers," CoRR, vol. abs/2301.02998, 2023.

[159] A. Askari, M. Aliannejadi, E. Kanoulas, and S. Verberne, "Generating synthetic documents for crossencoder re-rankers: A comparative study of chatgpt and human experts," CoRR, vol. abs/2305.02320, 2023.

[160] R. Pradeep, S. Sharifymoghaddam, and J. Lin, "Rankvicuna: Zero-shot listwise document reranking with open-source large language models," CoRR, vol. abs /2309.15088, 2023.

[161] -, "Rankzephyr: Effective and robust zeroshot listwise reranking is a breeze!" CoRR, vol. abs/2312.02724, 2023.

[162] W. Sun, Z. Chen, X. Ma, L. Yan, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, "Instruction distillation makes large language models efficient zero-shot rankers," arXiv preprint arXiv:2311.01555, 2023.

[163] C. J. C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. N. Hullender, "Learning to rank using gradient descent," in ICML, ser. ACM International Conference Proceeding Series, vol. 119. ACM, 2005, pp. 89-96.

[164] J. A. Baktash and M. Dawodi, "Gpt-4: A review on advancements and opportunities in natural language processing," arXiv preprint arXiv:2305.03195, 2023.
[165] H. Wachsmuth, S. Syed, and B. Stein, "Retrieval of the best counterargument without prior topic knowledge," in ACL (1). Association for Computational Linguistics, 2018, pp. 241-251.

[166] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, "Retrieval augmented language model pre-training," in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 3929-3938.

[167] P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrievalaugmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., 2020.

[168] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W. Yih, "REPLUG: retrieval-augmented black-box language models," CoRR, vol. abs/2301.12652, 2023.

[169] G. Izacard, P. S. H. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave, "Atlas: Few-shot learning with retrieval augmented language models," J. Mach. Learn. Res., vol. 24, pp. 251:1-251:43, 2023.

[170] A. Lazaridou, E. Gribovskaya, W. Stokowiec, and N. Grigorev, "Internet-augmented language models through few-shot prompting for open-domain question answering," CoRR, vol. abs/2203.05115, 2022.

[171] H. He, H. Zhang, and D. Roth, "Rethinking with retrieval: Faithful large language model inference," CoRR, vol. abs/2301.00303, 2023.

[172] W. Yu, H. Zhang, X. Pan, K. Ma, H. Wang, and D. Yu, "Chain-of-note: Enhancing robustness in retrieval-augmented language models," CoRR, vol. abs/2311.09210, 2023.

[173] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y. Shoham, "Incontext retrieval-augmented language models," CoRR, vol. abs/2302.00083, 2023.

[174] Z. Shao, Y. Gong, Y. Shen, M. Huang, N. Duan, and W. Chen, "Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 9248-9274.

[175] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions," in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 10014-10037.

[176] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-

Yu, Y. Yang, J. Callan, and G. Neubig, "Active retrieval augmented generation," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 7969-7992.

[177] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-rag: Learning to retrieve, generate, and critique through self-reflection," CoRR, vol. abs/2310.11511, 2023.

[178] J. Liu, J. Jin, Z. Wang, J. Cheng, Z. Dou, and J. Wen, "RETA-LLM: A retrieval-augmented large language model toolkit," CoRR, vol. abs/2306.05212, 2023.

[179] T. Vu, M. Iyyer, X. Wang, N. Constant, J. W. Wei, J. Wei, C. Tar, Y. Sung, D. Zhou, Q. V. Le, and T. Luong, "Freshllms: Refreshing large language models with search engine augmentation," CoRR, vol. abs/2310.03214, 2023.

[180] X. Lyu, S. Grafberger, S. Biegel, S. Wei, M. Cao, S. Schelter, and C. Zhang, "Improving retrievalaugmented large language models via data importance learning," CoRR, vol. abs/2307.03027, 2023.

[181] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling large language models to generate text with citations," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 6465-6488.

[182] H. Luo, T. Zhang, Y. Chuang, Y. Gong, Y. Kim, X. Wu, H. Meng, and J. R. Glass, "Search augmented instruction learning," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 3717-3729.

[183] X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis, L. Zettlemoyer, and S. Yih, "RA-DIT: retrievalaugmented dual instruction tuning," CoRR, vol. abs/2310.01352, 2023.

[184] W. Yu, Z. Zhang, Z. Liang, M. Jiang, and A. Sabharwal, "Improving language models via plug-and-play retrieval feedback," CoRR, vol. abs/2305.14002, 2023.

[185] Z. Feng, X. Feng, D. Zhao, M. Yang, and B. Qin, "Retrieval-generation synergy augmented large language models," CoRR, vol. abs/2310.05149, 2023.

[186] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, S. Johnston, S. E. Showk, A. Jones, N. Elhage, T. Hume, A. Chen, Y. Bai, S. Bowman, S. Fort, D. Ganguli, D. Hernandez, J. Jacobson, J. Kernion, S. Kravec, L. Lovitt, K. Ndousse, C. Olsson, S. Ringer, D. Amodei, T. Brown, J. Clark, N. Joseph, B. Mann, S. McCandlish, C. Olah, and J. Kaplan, "Language models (mostly) know what they know," CoRR, vol. abs/2207.05221, 2022.

[187] Z. Jiang, J. Araki, H. Ding, and G. Neubig, "How can we know When language models know? on the calibration of language models for question answering," Trans. Assoc. Comput. Linguistics, vol. 9, pp. 962-977,
2021.

[188] O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis, "Measuring and narrowing the compositionality gap in language models," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 5687-5711.

[189] O. Khattab, K. Santhanam, X. L. Li, D. Hall, P. Liang, C. Potts, and M. Zaharia, "Demonstratesearch-predict: Composing retrieval and language models for knowledge-intensive NLP," CoRR, vol. abs/2212.14024, 2022.

[190] O. Yoran, T. Wolfson, B. Bogin, U. Katz, D. Deutch, and J. Berant, "Answering questions by meta-reasoning over multiple chains of thought," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 5942-5966.

[191] M. A. Arefeen, B. Debnath, and S. Chakradhar, "Leancontext: Cost-efficient domain-specific question answering using llms," CoRR, vol. abs/2309.00841, 2023.

[192] F. Xu, W. Shi, and E. Choi, "RECOMP: improving retrieval-augmented lms with compression and selective augmentation," CoRR, vol. abs/2310.04408, 2023.

[193] Z. Wang, J. Araki, Z. Jiang, M. R. Parvez, and G. Neubig, "Learning to filter context for retrievalaugmented generation," CoRR, vol. abs/2311.08377, 2023.

[194] J. Liu, L. Li, T. Xiang, B. Wang, and Y. Qian, “TCRALLM: token compression retrieval augmented large language model for inference cost reduction," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 9796-9810.

[195] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," CoRR, vol. abs/2307.03172, 2023.

[196] R. Ren, Y. Wang, Y. Qu, W. X. Zhao, J. Liu, H. Tian, H. Wu, J. Wen, and H. Wang, "Investigating the factual knowledge boundary of large language models with retrieval augmentation," CoRR, vol. abs/2307.11019, 2023.

[197] Y. Liu, S. Yavuz, R. Meng, M. Moorthy, S. Joty, C. Xiong, and Y. Zhou, "Exploring the integration strategies of retriever and large language models," CoRR, vol. abs/2308.12574, 2023.

[198] R. Aksitov, C. Chang, D. Reitter, S. Shakeri, and Y. Sung, "Characterizing attribution and fluency tradeoffs for retrieval-augmented large language models," CoRR, vol. abs/2302.05578, 2023.

[199] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi, "When not to trust language models: Investigating effectiveness of parametric and nonparametric memories," in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto,

Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 9802-9822.

[200] Y. Wang, X. Ma, and W. Chen, "Augmenting blackbox llms with medical textbooks for clinical question answering," CoRR, vol. abs/2309.02233, 2023.

[201] S. Munikoti, A. Acharya, S. Wagle, and S. Horawalavithana, "ATLANTIC: structureaware retrieval-augmented language model for interdisciplinary science," CoRR, vol. abs/2311.12289, 2023.

[202] X. Li, E. Nie, and S. Liang, "Crosslingual retrieval augmented in-context learning for bangla," CoRR, vol. abs/2311.00587, 2023.

[203] A. Lozano, S. L. Fleming, C. Chiang, and N. Shah, "Clinfo.ai: An open-source retrieval-augmented large language model system for answering medical questions using scientific literature," CoRR, vol. abs/2310.16146, 2023.

[204] B. Zhang, H. Yang, T. Zhou, A. Babar, and X. Liu, "Enhancing financial sentiment analysis via retrieval augmented large language models," in 4th ACM International Conference on AI in Finance, ICAIF 2023, Brooklyn, NY, USA, November 27-29, 2023. ACM, 2023, pp. 349-356.

[205] A. Louis, G. van Dijck, and G. Spanakis, "Interpretable long-form legal question answering with retrieval-augmented large language models," CoRR, vol. abs/2309.17050, 2023.

[206] G. Zyskind, T. South, and A. Pentland, "Don't forget private retrieval: distributed private similarity search for large language models," CoRR, vol. abs/2311.12955, 2023.

[207] W. Jiang, M. Zeller, R. Waleffe, T. Hoefler, and G. Alonso, "Chameleon: a heterogeneous and disaggregated accelerator system for retrieval-augmented language models," CoRR, vol. abs/2310.09949, 2023.

[208] Y. Hoshi, D. Miyashita, Y. Ng, K. Tatsuno, Y. Morioka, O. Torii, and J. Deguchi, "Ralle: A framework for developing and evaluating retrieval-augmented large language models," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, Y. Feng and E. Lefever, Eds. Association for Computational Linguistics, 2023, pp. 52-69.

[209] R. Thoppilan, D. D. Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, Y. Li, H. Lee, H. S. Zheng, A. Ghafouri, M. Menegali, Y. Huang, M. Krikun, D. Lepikhin, J. Qin, D. Chen, Y. Xu, Z. Chen, A. Roberts, M. Bosma, Y. Zhou, C. Chang, I. Krivokon, W. Rusch, M. Pickett, K. S. Meier-Hellstern, M. R. Morris, T. Doshi, R. D. Santos, T. Duke, J. Soraker, B. Zevenbergen, V. Prabhakaran, M. Diaz, B. Hutchinson, K. Olson, A. Molina, E. Hoffman-John, J. Lee, L. Aroyo, R. Rajakumar, A. Butryna, M. Lamm, V. Kuzmina, J. Fenton, A. Cohen, R. Bernstein, R. Kurzweil, B. A. y Arcas, C. Cui, M. Croak, E. H. Chi, and Q. Le, "Lamda: Language models for dialog applications," CoRR, vol. abs /2201.08239, 2022.

[210] K. Shuster, M. Komeili, L. Adolphs, S. Roller,
A. Szlam, and J. Weston, "Language models that seek for knowledge: Modular search \& generation for dialogue and prompt completion," in Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Association for Computational Linguistics, 2022, pp. 373-393.

[211] X. Liu, H. Lai, H. Yu, Y. Xu, A. Zeng, Z. Du, P. Zhang, Y. Dong, and J. Tang, "Webglm: Towards an efficient web-enhanced question answering system with human preferences," in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023, A. K. Singh, Y. Sun, L. Akoglu, D. Gunopulos, X. Yan, R. Kumar, F. Ozcan, and J. Ye, Eds. ACM, 2023, pp. 4549-4560.

[212] I. Gur, H. Furuta, A. Huang, M. Safdari, Y. Matsuo, D. Eck, and A. Faust, "A real-world webagent with planning, long context understanding, and program synthesis," CoRR, vol. abs/2307.12856, 2023.

[213] J. Menick, M. Trebacz, V. Mikulik, J. Aslanides, H. F. Song, M. J. Chadwick, M. Glaese, S. Young, L. Campbell-Gillingham, G. Irving, and N. McAleese, "Teaching language models to support answers with verified quotes," CoRR, vol. abs/2203.11147, 2022.

[214] X. Shi, J. Liu, Y. Liu, Q. Cheng, and W. Lu, "Know where to go: Make LLM a relevant, responsible, and trustworthy searcher," CoRR, vol. abs/2310.12443, 2023.

[215] Y. Qin, Z. Cai, D. Jin, L. Yan, S. Liang, K. Zhu, Y. Lin, X. Han, N. Ding, H. Wang, R. Xie, F. Qi, Z. Liu, M. Sun, and J. Zhou, "Webcpm: Interactive web search for chinese long-form question answering," in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 8968-8988.

[216] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su, "Mind2web: Towards a generalist agent for the web," CoRR, vol. abs/2306.06070, 2023.

[217] S. Yao, H. Chen, J. Yang, and K. Narasimhan, "Webshop: Towards scalable real-world web interaction with grounded language agents," in NeurIPS, 2022.

[218] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, Y. Bisk, D. Fried, U. Alon, and G. Neubig, "Webarena: A realistic web environment for building autonomous agents," CoRR, vol. abs/2307.13854, 2023.

[219] R. Lo, A. Sridhar, F. F. Xu, H. Zhu, and S. Zhou, "Hierarchical prompting assists large language model on web navigation," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 10217-10244.

[220] S. MacAvaney, C. Macdonald, R. Murray-Smith, and I. Ounis, "Intent5: Search result diversification using causal language models," CoRR, vol. abs/2108.04026, 2021.

[221] N. Craswell, "Mean reciprocal rank," in Encyclopedia of Database Systems, L. Liu and M. T. Özsu, Eds. Springer US, 2009, p. 1703.

[222] K. Järvelin and J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques," ACM Trans. Inf. Syst., vol. 20, no. 4, pp. 422-446, 2002.

[223] K. Papineni, S. Roukos, T. Ward, and W. Zhu, "Bleu: a method for automatic evaluation of machine translation," in Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, July 6-12, 2002, Philadelphia, PA, USA. ACL, 2002, pp. 311-318.

[224] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, Jul. 2004, pp. 74-81.

[225] P. Manakul, A. Liusie, and M. J. F. Gales, "Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models," CoRR, vol. abs/2303.08896, 2023.

[226] H. Qian, Y. Zhu, Z. Dou, H. Gu, X. Zhang, Z. Liu, R. Lai, Z. Cao, J. Nie, and J. Wen, "Webbrain: Learning to generate factually correct articles for queries by grounding on large web corpus," CoRR, vol. abs/2304.04358, 2023.
[227] J. Li, X. Cheng, W. X. Zhao, J. Nie, and J. Wen, "Halueval: A large-scale hallucination evaluation benchmark for large language models," CoRR, vol. abs/2305.11747, 2023.

[228] L. Chen, Y. Deng, Y. Bian, Z. Qin, B. Wu, T. Chua, and K. Wong, "Beyond factuality: A comprehensive evaluation of large language models as knowledge generators," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 6325-6341.

[229] S. Xu, D. Hou, L. Pang, J. Deng, J. Xu, H. Shen, and X. Cheng, "Ai-generated images introduce invisible relevance bias to text-image retrieval," CoRR, vol. abs/2311.14084, 2023.

[230] S. Dai, Y. Zhou, L. Pang, W. Liu, X. Hu, Y. Liu, X. Zhang, and J. Xu, "Llms may dominate information access: Neural retrievers are biased towards llmgenerated texts," CoRR, vol. abs/2310.20501, 2023.

[231] J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," CoRR, vol. abs/2304.03442, 2023.


[^0]:    All authors are from Gaoling School of Artificial Intelligence and School of Information, Renmin University of China.

    Contact e-mail: yutaozhu94@gmail.com, dou@ruc.edu.cn

    1. Apple Siri, https://www.apple.com/siri/
    2. Google Assistant, https:/ / assistant.google.com/
[^1]:    4. New Bing, https://www.bing.com/new
</end of paper 3>


<paper 4>
# Benchmarking Large Language Models in Retrieval-Augmented Generation 

Jiawei Chen ${ }^{1,3}$, Hongyu Lin ${ }^{1, *}$, Xianpei Han ${ }^{1,2, *}$, Le Sun ${ }^{1,2}$<br>${ }^{1}$ Chinese Information Processing Laboratory ${ }^{2}$ State Key Laboratory of Computer Science<br>Institute of Software, Chinese Academy of Sciences, Beijing, China<br>${ }^{3}$ University of Chinese Academy of Sciences, Beijing, China<br>\{jiawei2020,hongyu,xianpei,sunle $\}$ @iscas.ac.cn


#### Abstract

Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the hallucination of large language models (LLMs). However, existing research lacks rigorous evaluation of the impact of retrieval-augmented generation on different large language models, which make it challenging to identify the potential bottlenecks in the capabilities of RAG for different LLMs. In this paper, we systematically investigate the impact of Retrieval-Augmented Generation on large language models. We analyze the performance of different large language models in 4 fundamental abilities required for RAG, including noise robustness, negative rejection, information integration, and counterfactual robustness. To this end, we establish Retrieval-Augmented Generation Benchmark (RGB), a new corpus for RAG evaluation in both English and Chinese. RGB divides the instances within the benchmark into 4 separate testbeds based on the aforementioned fundamental abilities required to resolve the case. Then we evaluate 6 representative LLMs on RGB to diagnose the challenges of current LLMs when applying RAG. Evaluation reveals that while LLMs exhibit a certain degree of noise robustness, they still struggle significantly in terms of negative rejection, information integration, and dealing with false information. The aforementioned assessment outcomes indicate that there is still a considerable journey ahead to effectively apply RAG to LLMs.


## Introduction

Recently, there have been impressive advancements in large language models (LLMs) like ChatGPT (OpenAI 2022) and ChatGLM (THUDM 2023a). Although these models have shown remarkable general abilities (Bang et al. 2023; Guo et al. 2023), they still suffer severely from challenges including factual hallucination (Cao et al. 2020; Raunak, Menezes, and Junczys-Dowmunt 2021; Ji et al. 2023), knowledge outdating (He, Zhang, and Roth 2022), and the lack of domainspecific expertise (Li et al. 2023c; Shen et al. 2023).

Incorporating external knowledge via information retrieval, i.e., Retrieval-Augmented Generation (RAG), has been regarded as a promising way to resolve the above challenges. (Guu et al. 2020; Lewis et al. 2020; Borgeaud et al.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_15bf50c80dd1394c2232g-1.jpg?height=827&width=789&top_left_y=749&top_left_x=1123)

Figure 1: Illustration of 4 kinds of abilities required for retrieval-augmented generation of LLMs.

2022; Izacard et al. 2022). With the help of external knowledge, LLMs can generate more accurate and reliable responses. The most common method is to use a search engine as a retriever such as New Bing. Due to the vast amount of information available on the Internet, using a search engine can provide more real-time information.

However, Retrieval-Augmented Generation brings not only positive effects to LLMs (Liu, Zhang, and Liang 2023; Maynez et al. 2020). On one hand, there is a significant amount of noise information even fake news in the content available on the Internet, which poses challenges for search engines in accurately retrieving desirable knowledge. On the other hand, LLMs suffer from unreliable generation challenge. LLMs can be misled by incorrect information contained in the context (Bian et al. 2023) and also suffer from hallucination during the generation (Adlakha et al. 2023), resulting in generating content that goes beyond external in-
formation. These challenges result in LLMs being unable to consistently generate reliable and accurate responses. Unfortunately, currently there lacks of comprehensive understanding on how these factors can influence RAG, and how could each model survives from these drawbacks and improvement their performance via information retrieval. As a result, there is a pressing need for a comprehensive evaluation of LLMs on their ability to effectively utilize retrieved information, as well as their ability to withstand the various drawbacks present in information retrieval.

To this end, this paper conducts a comprehensive evaluation of RAG for current LLMs. Specifically, we create a new Retrieval-Augmented Generation Benchmark, namely RGB, in both English and Chinese. In order to ensure that the internal knowledge of LLMs does not introduce bias into the evaluation results, RGB chooses to aggregate the latest news information and constructs queries based on the news information. Then, based on these queries, we use Search API to fetch relevant documents and select most relevant snippets from the content as external retrieved documents. Finally, based on different compositions of query and document-set pairs, we expand the corpus and divided it into 4 testbeds to evaluate the following basic abilities of LLMs according to the common challenges in RAG, as shown in Figure 1:

- Noise Robustness, which means a LLM can extract useful information from noisy documents. In this paper, we define noisy documents as those that are relevant to the question but do not contain any information of the answer. For the instance in Figure 1, the noisy documents related to the question "Who was awarded the 2022 Nobel Prize in Literature" include reports about the 2021 Nobel Prize in Literature. To this end, the testbed for noise robustness contains instances whose external documents contain a certain number of noisy documents based on the desired noise ratio.
- Negative Rejection, which means that a LLM should reject to answer the question when the required knowledge is not present in any retrieved document. The testbed for negative rejection contains instances whose external documents are only with noisy documents. LLMs are expected to indicate "insufficient information" or other rejection signals.
- Information Integration, which evaluates whether LLMs can answer complex questions that require integrating information from multiple documents. For the instance in Figure 1, for the question "When were the ChatGPT app for iOS and ChatGPT api launched?", LLMs are expected to provide information of the launch dates for both the ChatGPT iOS app and ChatGPT API. The testbed for information integration contains instances that can only be answered using multiple documents.
- Counterfactual Robustness, which evaluates whether LLMs can identify risks of known factual errors in the retrieved documents when the LLMs are given warnings about potential risks in the retrieved information through instruction. The testbed for counterfactual robustness includes instances that can be answered directly by the LLMs, but the external documents contain factual errors.
Based on RGB, we conduct evaluation on 6 state-ofthe-art large language models including ChatGPT (OpenAI 2022), ChatGLM-6B (THUDM 2023a), ChatGLM26B (THUDM 2023b), Vicuna-7b (Chiang et al. 2023), Qwen-7B-Chat (QwenLM 2023), BELLE-7B (Yunjie Ji 2023). We found that even though RAG can improve the response accuracy of LLMs, they still suffer from the abovementioned challenges significantly. Specifically, we found that even though LLMs demonstrate some level of noise robustness, they tend to confuse similar information and frequently generate inaccurate answers when relevant information exists. For example, when faced with a question about the 2022 Nobel Prize in Literature, if there are noisy documents about the 2021 Nobel Prize in Literature in external documents, LLMs may become confused and provide inaccurate answers. Besides, LLMs frequently fail to reject answering and generate incorrect answers when none of the external documents contain relevant information. Furthermore, LLMs lack the ability to summarize from multiple documents, and therefore if multiple documents are needed to answer a question, LLMs often fail to provide accurate answer. Finally, we found that even when the LLMs contain the required knowledge and are given warnings about potential risks in the retrieved information through instruction, they still tend to trust and prioritize the retrieved information over their own existing knowledge. The experimental results mentioned above highlight the need for further resolution of important issues in the existing RAG method. Therefore, it is crucial to exercise caution and carefully design its usage. Generally speaking, the contributions of this paper are ${ }^{1}$ :
- We proposed to evaluate four capabilities for retrievalaugmented generation of LLMs and created the Retrieval-Augmented Generation Benchmark in both English and Chinese. To best of our knowledge, it is the first benchmark designed to assess these four capabilities for retrieval-augmented generation of LLMs.
- We evaluated the existing LLMs using RGB and found the limitations of them in the four different abilities.
- We analyzed the responses of LLMs in RGB and identified their current shortcomings as well as suggested directions for improvement.


## Related work

Retrieval-augmented models The knowledge stored in large language models is commonly out-of-date (He, Zhang, and Roth 2022) and they also sometimes generate hallucination (Cao et al. 2020; Raunak, Menezes, and JunczysDowmunt 2021; Ji et al. 2023) i.e., they may generate irrelevant or factually incorrect contents. By using external knowledge as guidance, retrieval-augmented models can generate more accurate and reliable responses (Guu et al. 2020; Lewis et al. 2020; Borgeaud et al. 2022; Izacard et al. 2022; Shi et al. 2023; Ren et al. 2023). Retrievalaugmented models have achieved remarkable results in various tasks such as open-domain QA (Izacard and Grave 2021; Trivedi et al. 2023; Li et al. 2023a), dialogue (Cai[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_15bf50c80dd1394c2232g-3.jpg?height=849&width=699&top_left_y=188&top_left_x=255)

Figure 2: The process of data generation. Firstly, we use models to extract (event, question, answer) from news articles. Next, we utilize search engines to retrieve relevant web pages. Finally, a dense retrieval model is employed to re-rank the content of these web pages.

et al. 2019a,b; Peng et al. 2023), domain-specific question answering (Cui et al. 2023) and code generation (Zhou et al. 2023b). Recently, with the development of large models, a series of retrieval-enhanced tools and products have gained widespread attention, such as ChatGPT retrieval plugin, Langchain, New Bing, etc. However, in real-world scenarios, the retrieved text inevitably contains noise. Therefore, in this paper we conducted a systematic evaluation and analysis of retrieval-augmented generation in LLMs.

Evaluation of LLMs Evaluating LLMs has received significant attention due to their remarkable general capability (Chang et al. 2023). It enables us to gain a deeper understanding of the specific abilities and limitations of LLMs, while also providing valuable guidance for future research. In the past, benchmarks such as GLUE (Wang et al. 2019b) and SuperCLUE (Wang et al. 2019a) primarily focused on evaluating NLP tasks, particularly in natural language understanding. However, these evaluations often fail to fully capture the capabilities of LLMs. MMLU (Hendrycks et al. 2021) was then proposed to measure the knowledge acquired by language models when pre-training. Recently, with the development of LLMs, a series of general evaluation benchmarks have emerged, such as AGIEval (Zhong et al. 2023), C-Eval (Huang et al. 2023), AlpacaEval (Li et al. 2023b), OpenLLM Leaderboard (Edward Beeching 2023), etc. In addition to general abilities, there are also specific benchmarks that focus on evaluating the capabilities of models. For example, CValues (Xu et al. 2023a) focuses on the safety and responsibility of LLMs, M3Exam (Zhang et al. 2023) focuses on human exam and ToolBench (Qin et al. 2023) evaluates how well LLMs use external tools. Recently, Adlakha et al. (2023) evaluate the RAG of LLMs in exist QA dataset. Different from their work, we focus on 4 required abilities of RAG and create Retrieval-Augmented Generation Benchmark to evaluate the LLMs.

## Retrieval-Augmented Generation Benchmark

In this section, we first introduce the specific retrievalaugmented generation abilities we aim to evaluate. Next, we outline the process of constructing the RAG benchmark for evaluation. Lastly, we present the evaluation metrics.

## Required abilities of RAG

External knowledge is the key to resolving the problems of LLMs such as hallucination and outdated knowledge, which can make LLMs generate more accurate and reliable responses through retrieval-augmented generation (RAG). However, LLMs cannot always response as expected with RAG. For one thing, there are numerous irrelevant documents and false information on the Internet. Incorporating these external documents into LLMs could have a detrimental effect. For anthoer, LLMs suffer from the unreliable generation challenge. The generation of LLMs is often unpredictable, and we cannot guarantee that they will utilize the useful information entailed in the external documents. Additionally, LLMs can easily be misled by incorrect information in the document. To this end, we build RetrievalAugmented Generation Benchmark (RGB) to evaluate the retrieval-augmented generation of LLMs, and we concern about 4 specific abilities:

Noise Robustness is the robustness of LLMs in noisy documents. As retrievers are not perfect, the external knowledge they retrieve often contains a significant amount of noise, i.e., documents which are relevant to the question but do not contain any information about the answer. To effectively answer user questions, LLMs must be able to extract the necessary information from documents despite there are noisy documents.

Negative Rejection is a measure of whether LLMs can decline to answer a question when none of the contexts provide useful information. In real-world situations, the search engine often fails to retrieve documents containing the answers. In these cases, it is important for the model to have the capability to reject recognition and avoid generating misleading content.

Information Integration is a capacity to integrate answers from multiple documents. In many cases, the answer to a question may be contained in multiple documents. For example, for the question "Who are the champions of the U.S. Open 2022 men's and women's singles?", the two champions may be mentioned in different documents. In order to provide better answers to complex questions, it is necessary for LLMs to have the ability to integrate information.

Counterfactual Robustness refers to a capacity to handle errors in external knowledge. In the real world, there is an abundance of false information on the internet. Please
note that we only evaluate the situation that LLMs are given warnings about potential risks in the retrieved information through instruction.

In real-world scenarios, it is not possible to obtain perfect documents with all the necessary external knowledge. Therefore, evaluating these four abilities of the model becomes essential in order to measure the RAG of LLMs.

## Data construction

Inspired by previous benchmarks for LLMs, RGB utilizes a question-answering format for evaluation. We evaluate the LLMs by judging the retrieval-augmented responses of them to the questions. To simulate real-world scenarios, we construct question and answer data using actual news articles. Due to the abundance of knowledge contained within the LLMs there is a potential for bias when measuring the first three abilities. To mitigate this, the instances of RGB are constructed by latest news articles. Additionally, we retrieve external documents from Internet through search engines. Finally, we expand the corpus and divided it into 4 testbeds to evaluate the above basic abilities of LLMs. The overall procedure of our data construction is illustrated in Figure 2.

QA instances generation. We first collect latest news articles and use prompts to make ChatGPT generate events, questions, and answers for each articles. For example, as shown in the Figure 2, for a report about "The 2022 Nobel Prize", ChatGPT will generate corresponding event, question and provide key information for answering it. By generating events, the model is able to preliminarily filter out news articles that do not contain any events. After generation, we manually check the answer and filter out data that is difficult to retrieve through search engines.

Retrieve using search engine. For each query, we use Google's API to fetch 10 relevant web pages and extract corresponding snippets of text from them. Simultaneously, we read these web pages and convert their textual content into text chunks with a maximum length of 300 tokens. Using an existing dense retrieval model ${ }^{2}$, we select the top-30 text chunks that match the query most effectively. These retrieved text chunks, along with the snippets provided by the search API, will serve as our external documents. These documents will be divided into positive documents and negative documents based on whether they contain the answer.

Testbeds construction for each ability. We expand the corpus and divided it into 4 testbeds to evaluate the above basic abilities of LLMs. To evaluate the noise robustness, we sample varying numbers of negative documents according to the desired ratio of noises. For negative rejection, all the external documents are sampled from negative documents. For the information integration ability, we further construct data based on the above generated questions. This involves expanding or rewriting these questions so that their answers encompass multiple aspects. For example, the question "Who won the MVP of Super Bowl 2023?" can be rewrite as "Who won the MVPs of Super Bowl 2022 and 2023?". Consequently, answering such questions re-[^2]

|  | ![](https://cdn.mathpix.com/cropped/2024_06_04_15bf50c80dd1394c2232g-4.jpg?height=308&width=303&top_left_y=190&top_left_x=1589) |
| :---: | :---: |
| System instruction <br> Youglish <br> answer an accurate and reliable AI assistant that can <br> Please note that external documents may contain noisy <br> or factually incorrect information. If the information in <br> the document contains the correct answer, you will give <br> an accurate answer. If the information in the document <br> does not contain the answer, you will generate 'I can not <br> answer the question because of the insufficient <br> information in documents. If there are inconsistencies <br> with the facts in some of the documents, please generate <br> the response 'There are factual errors in the provided <br> documents.' and provide the correct answer. | $\ln \{\mathrm{DOCS}\} \ln \ln$ 问题 |

Figure 3: The instructions used in our experiments, which include a system instruction followed by a user input instruction. The " $\{\mathrm{DOCS}\}$ " and " $\{\mathrm{QUERY}\}$ " will be replaced by the external documents and the question.

quires utilizing information from various documents. Different from the first three abilities, the data of counterfactual robustness is constructed solely based on the internal knowledge of the model. Based on the aforementioned generated questions mentioned above, we adopt ChatGPT to automatically generate its known knowledge. Specifically, we use prompts to allow the model to generate both questions and answers that are already known. For example, based on the question "Who was awarded the 2022 Nobel Prize for Physiology and Medicine?", the model will generate the known question "Who was awarded the 2021 Nobel Prize in Literature?" and answer "Abdulrazak Gurnah". We then manually verified the generated answers, and retrieve relevant documents as described above. In order to make documents contain factual errors, we manually modify the answers and replace the corresponding parts in the document.

Finally, we collect totally 600 base questions in RGB, and 200 additional questions for the information integration ability and 200 additional questions for counterfactual robustness ability. Half of the instances are in English, and the other half are in Chinese.

## Evaluation metrics

The core of this benchmark is to evaluate whether LLMs can utilize the provided external documents to acquire knowledge and generate reasonable answers. We evaluate the responses of LLMs in order to measure above-mentioned four abilities of them.

Accuracy is used to measure noise robustness and information integration. We employ an exact matching approach where if the generated text contains an exact match to the answer, it is considered as a correct answer.

Rejection rate is used to measure negative rejection. When only noisy documents are provided, LLMs should output the specific content - "I can not answer the question because of the insufficient information in documents." (We use instructions to inform the model.). If the model generates this content, it indicates a successful rejection.

Error detection rate measures whether the model can detect the factual errors in the documents for counterfactual robustness. When the provided documents contain factual errors, the model should output the specific content - "There are factual errors in the provided documents." (We use in-

| Noise Ratio | English |  |  |  |  | Chinese |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 0 | 0.2 | 0.4 | 0.6 | 0.8 |
| ChatGPT (OpenAI 2022) | 96.33 | 94.67 | 94.00 | 90.00 | 76.00 | 95.67 | 94.67 | 91.00 | 87.67 | 70.67 |
| ChatGLM-6B (THUDM 2023a) | 93.67 | 90.67 | 89.33 | 84.67 | 70.67 | 94.33 | 90.67 | 89.00 | 82.33 | 69.00 |
| ChatGLM2-6B (THUDM 2023b) | 91.33 | 89.67 | 83.00 | 77.33 | 57.33 | 86.67 | 82.33 | 76.67 | 72.33 | 54.00 |
| Vicuna-7B-v1.3 (Chiang et al. 2023) | 87.67 | 83.33 | 86.00 | 82.33 | 60.33 | 85.67 | 82.67 | 77.00 | 69.33 | 49.67 |
| Qwen-7B-Chat (QwenLM 2023) | 94.33 | 91.67 | 91.00 | 87.67 | 73.67 | 94.00 | 92.33 | 88.00 | 84.33 | 68.67 |
| BELLE-7B-2M (Yunjie Ji 2023) | 83.33 | 81.00 | 79.00 | 71.33 | 64.67 | 92.00 | 88.67 | 85.33 | 78.33 | 67.68 |

Table 1: The experimental result of noise robustness measured by accuracy (\%) under different noise ratios. We can see that the increasing noise rate poses a challenge for RAG in LLMs.

|  | Long-distance information. | Evidence uncertainty. | Concept confusion. |
| :---: | :---: | :---: | :---: |
| Question | Who did Iga Swiatek defeat to win the Qatar Open 2022? | What is the name of Apple's headset? | What was Tesla's revenue in Q1 2022? |
| Answer | Anett Kontaveit | Vision Pro | 18.76 billion |
| Documents | Positive document <br> In February, Swiatek entered into the Qatar Open ... <br> In the final, she won ... Anett Kontaveit ... <br> Negative document <br> This time, she defeated Ons Jabeur 6-2, 7-6(5) to win <br> the 2022 US Open, ... | Positive document <br> Apple (AAPL.O) on Monday unveiled a costly <br> augmented-reality headset called the Vision Pro ... <br> Negative document <br> ... is what Gurman believes will be called <br> Apple Reality Pro. ... | Positive document <br> Tesla, Inc. (TSLA) reported Q1 FY 2022 earnings results <br> ... detailed revenues of $\$ 18.76$ billion ... <br> Negative document <br> ...first-quarter earnings for $2022 \ldots$ <br> ...Automotive revenue reached $\$ 16.86$ billion... |
| Responses | Iga Swiatek defeated Ons Jabeur in the second round <br> of the Qatar Open 2022 to win the tournament. | According to the document, the name of Apple's <br> headset is Apple Reality Pro. | According to the financial results provided in the article, <br> Tesla's revenue in Q1 2022 was $\$ 16.86$ billion. |

Table 2: Error cases of noise robustness, and only one positive document and one negative document are shown. The responses are generated by ChatGLM2-6B. The blue text indicates the matching parts between the document and the question or answer, while the red text highlights the non-matching parts.

structions to inform the model.). If the model generates this content, it indicates that the model has detected erroneous information in the document.

Error correction rate measures whether the model can provide the correct answer after identifying errors for counterfactual robustness. The model is asked to generate the correct answer after identifying the factual errors. If the model generates the correct answer, it indicates that the model is capable of correcting errors in the document.

Considering that the model may not fully adhere to instructions, for rejection rate and error detection rate, we also use ChatGPT to conduct additional evaluation of the answers. Specifically, we assess the model's responses by using instructions and demonstrations to determine if they can reflect information that is not present in the document or identify any factual errors.

## Experiments

In this section, we evaluate the performance of various LLMs, analyze and discuss the results, summarizing the main challenges that existing LLMs encounter when using external knowledge.

## Settings

Task formats. Due to contextual limitations, we provide 5 external documents for each question. In our experiments on noise robustness, we evaluate scenarios with noise ratios ranging from 0 to 0.8 . To comprehensively evaluate the overall capabilities, we have adopted a unified instruction for each language, as shown in Figure 3. The experiments were conducted using an NVIDIA GeForce RTX 3090.
Models We conduct evaluation on 6 state-of-the-art large language models which can generate both English and Chinese including ChatGPT (OpenAI 2022) ${ }^{3}$, ChatGLM-6B (THUDM 2023a), ChatGLM2-6B (THUDM 2023b), Vicuna-7b-v1.3 (Chiang et al. 2023), Qwen-7BChat (QwenLM 2023), BELLE-7B-2M (Yunjie Ji 2023).

## Results on Noise Robustness

We evaluated the accuracy based on the different noise ratios in external documents, and the results are shown in Table 1. We can see that:

(1) RAG can effect improve the responses of LLMs. LLMs have shown strong performance even in the presence of noise, indicating that RAG is a promising way for LLMs to generate accurate and reliable responses.

(2) The increasing noise rate poses a challenge for RAG in LLMs. Specifically, when the noise ratio exceeds $80 \%$, the accuracy decreases significantly at a significance level of 0.05. For example, the performance of ChatGPT has decreased from $96.33 \%$ to $76.00 \%$, while the performance of ChatGLM2-6B has decreased from $91.33 \%$ to $57.33 \%$.

Error Analysis. To better comprehend the negative impact of noise on model generation, we examined the incorrect answers and found that these errors typically originate from three reasons, as shown in Table 2.

(1) Long-distance information. LLMs often face difficulty in identifying the correct answer from external documents when the information related to the question is distant from the information related to the answer. This scenario is quite common as longer texts are frequently encountered[^3]on the internet. In such cases, it is typical for the question's information to be initially presented at the start of the document and subsequently referred to using pronouns. In Table 2, the question information ("Qatar Open 2022") is only mentioned once at the beginning and is far from where the answer text "Anett Kontaveit" appears. This situation may cause LLMs to depend on information from other documents and create false impressions, i.e., hallucination.

(2) Evidence uncertainty. Before highly anticipated events, like the release of new Apple products or the announcement of the Oscars, there is often a significant amount of speculative information circulating on the internet. Although the relevant documents explicitly state that it is uncertain or speculative content, they can still impact on the retrieval-augmented generation of LLMs. In Table 2, when the noise ratio increases, the content of erroneous documents is all about some people's predictions about the name of the headset ("Apple Reality Pro"). Even if there is a correct answer ("Vision Pro") in the relevant documents, LLMs can still be misled by uncertain evidences.

(3) Concept confusion. The concepts in external documents may be similar to, but different from, the concepts in the question. This can cause confusion for LLMs and make LLMs generate incorrect answers. In Table 2, the model answer focuses on the concept "automotive revenue" in the document rather than "revenue" in the question.

Based on the analysis above, we have identified certain limitations in LLMs regarding retrieval-augmented generation. To effectively handle the vast amount of noise present on the internet, further detailed enhancements are required for the model such as long documents modeling and precise concept comprehension.

## Results on Negative Rejection testbed

We evaluated the rejection rate when only noise documents were provided. The results are shown in Table 3. In addition to evaluating the rejection rate through exact matching (Rej in Table 3), we also utilize ChatGPT to determine if the responses from the LLMs contain any rejection information (Rej* in Table 3). We can see that: Negative Rejection poses a challenge for RAG in LLMs. The highest rejection rates for LLMs in English and Chinese were only $45 \%$ and $43.33 \%$, respectively. This suggests that LLMs can be easily misled by noisy documents, leading to incorrect answers.

In addition, through comparing Rej and Rej*, we found that LLMs fail to strictly follow instructions, and they often generate unpredictable responses, which make it hard to use them as state triggers (such as for recognizing rejection).

We conduct case studies in Table 4. The first error is because of Evidence uncertainty. Although the document only mentions contact with "Adam McKay" and does not explicitly state that he is the director of the movie, the model still concludes that he holds this role. The first error is because of Concept confusion. The information provided in the answer pertains to "the 2018 Winter Olympics" instead of "the 2022 Olympics" mentioned in the question. Retrieval-augmented generation poses a greater challenge of negative rejection compared to answer directly as it presents relevant documents that could potentially mislead the LLMs

| Languages | English |  | Chinese |  |
| :---: | :---: | :---: | :---: | :---: |
|  | Rej | Rej $^{*}$ | Rej | Rej $^{*}$ |
| ChatGPT | 24.67 | $\mathbf{4 5 . 0 0}$ | 5.33 | $\mathbf{4 3 . 3 3}$ |
| ChatGLM-6B | 9.00 | 25.00 | 6.33 | 17.00 |
| ChatGLM2-6B | 10.33 | 41.33 | 6.33 | 36.33 |
| Vicuna-7B-v1.3 | 17.00 | 33.33 | 3.37 | 24.67 |
| Qwen-7B-Chat | $\mathbf{3 1 . 0 0}$ | 35.67 | $\mathbf{8 . 6 7}$ | 25.33 |
| BELLE-7B-2M | 5.67 | 32.33 | 5.33 | 13.67 |

Table 3: The result of negative rejection. Rej means the rejection rate (\%) and Rej* means the rejection rate evaluated by ChatGPT. We can see that negative rejection poses a challenge for RAG in LLMs.

| Question | Answer | Response |
| :--- | :--- | :--- |
| who will direct <br> Irredeemable film? | The information provided in the document <br> Samuel <br> is that Fox has attached "The Big Short" <br> writer-director Adam McKay to its movie <br> adaptation of "Irredeemable" from Boom! <br> Comics. Therefore, the answer is Adam McKay. |  |
| Which country won the <br> most medals at the | Norway | Based on the information provided in the <br> document, the country that won the most medals <br> at the 2022 Winter Olympics is Germany. <br> Germany has won a total of 31 medals, including 14 <br> gold medals, 14 bronze medals, and 31 silver medals. |

Table 4: Error cases of negative rejection generated by ChatGLM2-6B. The red text highlights the error answers.

and result in incorrect responses. In future developments, it will be crucial for LLMs to enhance their ability to accurately match questions with the appropriate documents.

## Results on Information Integration testbed

We evaluated the accuracy based on the different noise ratios in external documents, and the results are shown in Table 5. When comparing the model to Table 1, we observed that it has a weak information integration ability, which in turn affects its noise robustness. We can see that:

(1) Information integration poses a challenge for RAG in LLMs. Even without noise, the highest accuracy of LLMs can only reach $60 \%$ and $67 \%$ for English and Chinese, respectively. After adding noise, the highest accuracy decreases to $43 \%$ and $55 \%$. These results suggest that LLMs struggle with integrating information effectively and are not well-suited for directly answering complex questions.

(2) Complex questions are more challenging for RAG with noisy documents. Performance decline becomes significant when the noise ratio is 0.4 , but for simple problems, a significant decline occurs only at a noise ratio of 0.8 at a significance level of 0.05 . This indicates that complex problems are more vulnerable to interference from noise. We speculate that this is because solving complex problems requires integrating information from multiple documents, and this information can be considered as noise to each other, making it harder for the model to extract relevant information from the documents.

Error Analysis. We conducted an error analysis on ChatGLM2-6B (noise ratio is 0). Apart from the similar errors founded in the noise robustness experiment ( $38 \%$ of the total), there are also three types of unique errors. We have presented these cases in Table 6.

|  | English |  |  | Chinese |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Noise Ratio | 0 | 0.2 | 0.4 | 0 | 0.2 | 0.4 |
| ChatGPT | 55 | 51 | 34 | 63 | $\mathbf{5 8}$ | 47 |
| ChatGLM-6B | 45 | 36 | 35 | 60 | 53 | 52 |
| ChatGLM2-6B | 34 | 32 | 21 | 44 | 43 | 32 |
| Vicuna-7B-v1.3 | $\mathbf{6 0}$ | $\mathbf{5 3}$ | $\mathbf{4 3}$ | 43 | 36 | 25 |
| Qwen-7B-Chat | 55 | 50 | 37 | $\mathbf{6 7}$ | 56 | $\mathbf{5 5}$ |
| BELLE-7B-2M | 40 | 34 | 24 | 49 | 41 | 38 |

Table 5: The experimental result of information integration measured by accuracy (\%) under different noise ratios. We can see that information integration poses a challenge for RAG in LLMs.

| Question | Answer | Response | Errors |
| :--- | :--- | :--- | :---: |
| What groupings are France and <br> Germany in World Cup 2022? | Group D <br> Group E | France and Germany are in <br> Group D of the 2022 FIFA <br> World Cup. | Merging <br> Error |
| Who were the MVP of Super | Cooper Kupp <br> Patrick <br> Mahomes | According to the provided <br> document, the MVP of Super <br> Bowl LVI was Cooper Kupp <br> of the Los Angeles Rams. | Ignoring <br> Error |
| What films won the 2022 and <br> 2023 Academy Awards for <br> Best Picture? | CODA <br> Everything <br> Everywhere <br> All at Once | The film CODA won the <br> award for Best Picture at the <br> 95 th Academy Awards <br> ceremony held on 2023. | Misalignment <br> Error |

Table 6: Error cases of information integration, the responses are generated by ChatGLM2-6B. The blue and red texts represent the answers to two sub-questions.

(1) Merging Error ( $\mathbf{2 8 \%}$ of the total). The model sometimes merges the answers of the two sub-questions, resulting in an error. It mistakenly uses the answer from one question to address both two questions. At this point, the model will disregard any documents related to one sub-question. For example, in Table 6, it incorrectly states that Group D is the World Cup group for both France and Germany, while in fact Germany is actually assigned to Group E.

(2) Ignoring Error (28\% of the total). Sometimes, the model may ignore one of the sub-questions and only answer the other. This error occurs when the model lacks a complete understanding of the problem and fails to recognize that it consists of multiple sub-problems. As a result, the model only considers relevant documents for one sub-problem in order to generate an answer, disregarding the question posed by another sub-problem. For example, in Table 6, the model only provides the answer for the MVP of Super Bowl 2022 and does not consider 2023.

(3) Misalignment Error (6\% of the total). Sometimes, the model incorrectly identifies the documents for one subquestion as the documents for another sub-question, leading to misaligned answers. For example, in Table 6, the third answer has two errors: an ignoring error and a misalignment error. Firstly, the model only mentioned the Best Picture of the 2023 (95th) Academy Awards, completely disregarding the 2022 awards. Additionally, it incorrectly stated that "CODA" is the Best Picture of 2023 when it was actually awarded as the Best Picture in 2022 .

The errors mentioned above are primarily caused by the limited understanding of complex questions, which hinders the ability to effectively utilize information from different sub-problems. The key lies in improving the model's reasoning capability. One possible solution is to use a chain-of-

|  | Acc | Acc $_{\text {doc }}$ | ED | ED $^{*}$ | CR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ChatGPT-zh | 91 | $\mathbf{1 7}$ | 1 | 3 | 33.33 |
| Qwen-7B-Chat-zh | 77 | 12 | 5 | 4 | 25.00 |
| ChatGPT-en | 89 | 9 | $\mathbf{8}$ | $\mathbf{7}$ | $\mathbf{5 7 . 1 4}$ |

Table 7: The result of counterfactual robustness. ACC is the accuracy (\%) of LLMs without external documents. $\mathrm{ACC}_{\mathrm{doc}}$ is the accuracy (\%) of LLMs with counterfactual documents. $\mathrm{ED}$ and $\mathrm{ED}^{*}$ are error detection rates evaluated by exact matching and ChatGPT, respectively. CR is the error correction rate.

thought approach to break down complex problems (Zhou et al. 2023a; Xu et al. 2023b; Drozdov et al. 2023). However, these methods slow down the inference speed and cannot provide timely responses.

## Results on Counterfactual Robustness testbed

In order to ensure that LLMs possess relevant knowledge, we assess their performance by directly asking them questions. However, we found that most LLMs struggle to answer them correctly. To ensure a more reasonable evaluation, we only consider LLMs that have an accuracy rate of over $70 \%$ as this threshold is relatively high and encompasses more LLMs. The results are shown in Table 7. We present the following metrics: accuracy without any documents, accuracy with counterfactual documents, error detection rates, and error correction rates. We can see that It is hard for LLMs to identify and correct factual errors in the documents. This suggests that the model can be easily misled by documents containing incorrect facts.

It is important to note that retrieval-augmented generation is not designed to automatically address factual errors within a given context, as this contradicts the underlying assumption that the model lacks knowledge and relies on retrieved documents for additional information. However, this issue is crucial in practical applications due to the abundance of fake news on the internet. Existing LLMs do not have a safeguard to handle inaccurate responses caused by misinformation. In fact, they heavily depend on the information they retrieve. Even when LLMs contain the internal knowledge about the questions, they often trust false information that is retrieved. This presents significant a challenge for the future development of RAG in LLMs.

## Conclusion

In this paper, we evaluated four abilities of retrievalaugmented generation in LLMs: noise robustness, negative rejection, information integration, and counterfactual robustness. To conduct the evaluation, we built RetrievalAugmented Generation Benchmark (RGB). The instances of RGB are generated from latest news articles and the external documents obtained from search engines. The experimental results suggest that current LLMs have limitations in the 4 abilities. This indicates that there is still a significant amount of work needed to effectively apply RAG to LLMs. To ensure accurate and reliable responses from LLMs, it is crucial to exercise caution and carefully design for RAG.

## Acknowledgements

This research work is supported by the National Natural Science Foundation of China under Grants no. 62122077, 62106251, 62306303, the CAS Project for Young Scientists in Basic Research under Grant No.YSBR-040. Xianpei Han is sponsored by CCF- BaiChuan-Ebtech Foundation Model Fund.

## References

Adlakha, V.; BehnamGhader, P.; Lu, X. H.; Meade, N.; and Reddy, S. 2023. Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering. arXiv:2307.16877.

Bang, Y.; Cahyawijaya, S.; Lee, N.; Dai, W.; Su, D.; Wilie, B.; Lovenia, H.; Ji, Z.; Yu, T.; Chung, W.; Do, Q. V.; Xu, Y.; and Fung, P. 2023. A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. arXiv:2302.04023.

Bian, N.; Liu, P.; Han, X.; Lin, H.; Lu, Y.; He, B.; and Sun, L. 2023. A Drop of Ink Makes a Million Think: The Spread of False Information in Large Language Models. arXiv:2305.04812.

Borgeaud, S.; Mensch, A.; Hoffmann, J.; Cai, T.; Rutherford, E.; Millican, K.; van den Driessche, G.; Lespiau, J.-B.; Damoc, B.; Clark, A.; de Las Casas, D.; Guy, A.; Menick, J.; Ring, R.; Hennigan, T.; Huang, S.; Maggiore, L.; Jones, C.; Cassirer, A.; Brock, A.; Paganini, M.; Irving, G.; Vinyals, O.; Osindero, S.; Simonyan, K.; Rae, J. W.; Elsen, E.; and Sifre, L. 2022. Improving language models by retrieving from trillions of tokens. arXiv:2112.04426.

Cai, D.; Wang, Y.; Bi, W.; Tu, Z.; Liu, X.; Lam, W.; and Shi, S. 2019a. Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 12191228. Minneapolis, Minnesota: Association for Computational Linguistics.

Cai, D.; Wang, Y.; Bi, W.; Tu, Z.; Liu, X.; and Shi, S. 2019b. Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 1866-1875. Hong Kong, China: Association for Computational Linguistics.

Cao, M.; Dong, Y.; Wu, J.; and Cheung, J. C. K. 2020. Factual Error Correction for Abstractive Summarization Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 62516258. Online: Association for Computational Linguistics.

Chang, Y.; Wang, X.; Wang, J.; Wu, Y.; Yang, L.; Zhu, K.; Chen, H.; Yi, X.; Wang, C.; Wang, Y.; Ye, W.; Zhang, Y.; Chang, Y.; Yu, P. S.; Yang, Q.; and Xie, X. 2023. A Survey on Evaluation of Large Language Models. arXiv:2307.03109.
Chiang, W.-L.; Li, Z.; Lin, Z.; Sheng, Y.; Wu, Z.; Zhang, H.; Zheng, L.; Zhuang, S.; Zhuang, Y.; Gonzalez, J. E.; Stoica, I.; and Xing, E. P. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with $90 \% *$ ChatGPT Quality.

Cui, J.; Li, Z.; Yan, Y.; Chen, B.; and Yuan, L. 2023. ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases. arXiv:2306.16092.

Drozdov, A.; Schärli, N.; Akyürek, E.; Scales, N.; Song, X.; Chen, X.; Bousquet, O.; and Zhou, D. 2023. Compositional Semantic Parsing with Large Language Models. In The Eleventh International Conference on Learning Representations.

Edward Beeching, N. H. S. H. N. L. N. R. O. S. L. T. T. W., Clémentine Fourrier. 2023. Open LLM Leaderboard. https://huggingface.co/spaces/HuggingFaceH4/ open_llm_leaderboard.

Guo, B.; Zhang, X.; Wang, Z.; Jiang, M.; Nie, J.; Ding, Y.; Yue, J.; and Wu, Y. 2023. How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection. arXiv:2301.07597.

Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.-W. 2020. REALM: Retrieval-Augmented Language Model PreTraining. In Proceedings of the 37th International Conference on Machine Learning, ICML'20. JMLR.org.

He, H.; Zhang, H.; and Roth, D. 2022. Rethinking with Retrieval: Faithful Large Language Model Inference. arXiv:2301.00303.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Huang, Y.; Bai, Y.; Zhu, Z.; Zhang, J.; Zhang, J.; Su, T.; Liu, J.; Lv, C.; Zhang, Y.; Lei, J.; Fu, Y.; Sun, M.; and He, J. 2023. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. arXiv preprint arXiv:2305.08322.

Izacard, G.; and Grave, E. 2021. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, 874-880. Online: Association for Computational Linguistics.

Izacard, G.; Lewis, P.; Lomeli, M.; Hosseini, L.; Petroni, F.; Schick, T.; Dwivedi-Yu, J.; Joulin, A.; Riedel, S.; and Grave, E. 2022. Atlas: Few-shot Learning with Retrieval Augmented Language Models. arXiv:2208.03299.

Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y.; Ishii, E.; Bang, Y. J.; Madotto, A.; and Fung, P. 2023. Survey of Hallucination in Natural Language Generation. ACM Comput. Surv., 55(12).

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS'20. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713829546.

Li, D.; Rawat, A. S.; Zaheer, M.; Wang, X.; Lukasik, M.; Veit, A.; Yu, F.; and Kumar, S. 2023a. Large Language Models with Controllable Working Memory. In Findings of the Association for Computational Linguistics: ACL 2023, 1774-1793. Toronto, Canada: Association for Computational Linguistics.

Li, X.; Zhang, T.; Dubois, Y.; Taori, R.; Gulrajani, I.; Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023b. AlpacaEval: An Automatic Evaluator of Instruction-following Models. https://github.com/tatsu-lab/alpaca_eval.

Li, X.; Zhu, X.; Ma, Z.; Liu, X.; and Shah, S. 2023c. Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? An Examination on Several Typical Tasks. arXiv:2305.05862.

Liu, N. F.; Zhang, T.; and Liang, P. 2023. Evaluating Verifiability in Generative Search Engines. arXiv:2304.09848.

Maynez, J.; Narayan, S.; Bohnet, B.; and McDonald, R. 2020. On Faithfulness and Factuality in Abstractive Summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 1906-1919. Online: Association for Computational Linguistics.

OpenAI. 2022. Chatgpt: Optimizing language models for dialogue. https://openai.com/blog/chatgpt.

Peng, B.; Galley, M.; He, P.; Cheng, H.; Xie, Y.; Hu, Y.; Huang, Q.; Liden, L.; Yu, Z.; Chen, W.; and Gao, J. 2023. Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback. arXiv:2302.12813.

Qin, Y.; Liang, S.; Ye, Y.; Zhu, K.; Yan, L.; Lu, Y.; Lin, Y.; Cong, X.; Tang, X.; Qian, B.; Zhao, S.; Tian, R.; Xie, R.; Zhou, J.; Gerstein, M.; Li, D.; Liu, Z.; and Sun, M. 2023. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arXiv:2307.16789.

QwenLM. 2023. Qwen-7B. https://github.com/QwenLM/ Qwen-7B.

Raunak, V.; Menezes, A.; and Junczys-Dowmunt, M. 2021. The Curious Case of Hallucinations in Neural Machine Translation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 11721183. Online: Association for Computational Linguistics.

Ren, R.; Wang, Y.; Qu, Y.; Zhao, W. X.; Liu, J.; Tian, H.; Wu, H.; Wen, J.-R.; and Wang, H. 2023. Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation. arXiv:2307.11019.

Shen, X.; Chen, Z.; Backes, M.; and Zhang, Y. 2023. In ChatGPT We Trust? Measuring and Characterizing the Reliability of ChatGPT. arXiv:2304.08979.

Shi, W.; Min, S.; Yasunaga, M.; Seo, M.; James, R.; Lewis, M.; Zettlemoyer, L.; and tau Yih, W. 2023. REPLUG: Retrieval-Augmented Black-Box Language Models. arXiv:2301.12652.

THUDM. 2023a. ChatGLM-6B. https://github.com/ THUDM/ChatGLM-6B.

THUDM. 2023b. ChatGLM2-6B. https://github.com/ THUDM/ChatGLM2-6B.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10014-10037. Toronto, Canada: Association for Computational Linguistics.

Wang, A.; Pruksachatkun, Y.; Nangia, N.; Singh, A.; Michael, J.; Hill, F.; Levy, O.; and Bowman, S. R. 2019a. SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. Red Hook, NY, USA: Curran Associates Inc.

Wang, A.; Singh, A.; Michael, J.; Hill, F.; Levy, O.; and Bowman, S. R. 2019b. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In International Conference on Learning Representations.

Xu, G.; Liu, J.; Yan, M.; Xu, H.; Si, J.; Zhou, Z.; Yi, P.; Gao, X.; Sang, J.; Zhang, R.; Zhang, J.; Peng, C.; Huang, F.; and Zhou, J. 2023a. CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility. arXiv:2307.09705.

Xu, S.; Pang, L.; Shen, H.; Cheng, X.; and Chua, T.S. 2023b. Search-in-the-Chain: Towards Accurate, Credible and Traceable Large Language Models for Knowledgeintensive Tasks. arXiv:2304.14732.

Yunjie Ji, Y. G. Y. P. Q. N. B. M. X. L., Yong Deng. 2023. BELLE: Bloom-Enhanced Large Language model Engine. https://github.com/LianjiaTech/BELLE.

Zhang, W.; Aljunied, S. M.; Gao, C.; Chia, Y. K.; and Bing, L. 2023. M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models.

Zhong, W.; Cui, R.; Guo, Y.; Liang, Y.; Lu, S.; Wang, Y.; Saied, A.; Chen, W.; and Duan, N. 2023. AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models. arXiv:2304.06364.

Zhou, D.; Schärli, N.; Hou, L.; Wei, J.; Scales, N.; Wang, X.; Schuurmans, D.; Cui, C.; Bousquet, O.; Le, Q. V.; and Chi, E. H. 2023a. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. In The Eleventh International Conference on Learning Representations.

Zhou, S.; Alon, U.; Xu, F. F.; Jiang, Z.; and Neubig, G. 2023b. DocPrompting: Generating Code by Retrieving the Docs. In The Eleventh International Conference on Learning Representations.


[^0]:    * Corresponding authors

    Copyright $\odot$ 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

[^1]:    ${ }^{1}$ Our code\&data: https://github.com/chen700564/RGB.

[^2]:    ${ }^{2}$ Chinese: https://huggingface.co/moka-ai/m3e-base; English: https://huggingface.co/sentence-transformers/all-mpnet-base-v2.

[^3]:    ${ }^{3}$ We use gpt-3.5-turbo api in the experiments.

</end of paper 4>


