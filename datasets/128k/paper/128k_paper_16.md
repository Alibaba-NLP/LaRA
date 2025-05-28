<paper 0>
# Towards Conversational Diagnostic AI 

Tao $\mathrm{Tu}^{*, 1}$, Anil Palepu*,1, Mike Schaekermann*,1,<br>Khaled Saab ${ }^{1}$, Jan Freyberg ${ }^{1}$, Ryutaro Tanno ${ }^{2}$, Amy Wang ${ }^{1}$, Brenna $\mathrm{Li}^{1}$, Mohamed Amin ${ }^{1}$,<br>Nenad Tomasev ${ }^{2}$, Shekoofeh Azizi ${ }^{2}$, Karan Singhal ${ }^{1}$, Yong Cheng ${ }^{2}$, Le Hou ${ }^{1}$, Albert Webson ${ }^{2}$,<br>Kavita Kulkarni ${ }^{1}$, S. Sara Mahdavi ${ }^{2}$, Christopher Semturs ${ }^{1}$,<br>Juraj Gottweis ${ }^{1}$, Joelle Barral ${ }^{2}$, Katherine Chou ${ }^{1}$, Greg S. Corrado ${ }^{1}$, Yossi Matias ${ }^{1}$,<br>Alan Karthikesalingam ${ }^{\dagger, 1}$ and Vivek Natarajan ${ }^{\dagger, 1}$<br>${ }^{1}$ Google Research, ${ }^{2}$ Google DeepMind


#### Abstract

At the heart of medicine lies the physician-patient dialogue, where skillful history-taking paves the way for accurate diagnosis, effective management, and enduring trust. Artificial Intelligence (AI) systems capable of diagnostic dialogue could increase accessibility, consistency, and quality of care. However, approximating clinicians' expertise is an outstanding grand challenge. Here, we introduce AMIE (Articulate Medical Intelligence Explorer), a Large Language Model (LLM) based AI system optimized for diagnostic dialogue. AMIE uses a novel self-play based simulated environment with automated feedback mechanisms for scaling learning across diverse disease conditions, specialties, and contexts. We designed a framework for evaluating clinically-meaningful axes of performance including history-taking, diagnostic accuracy, management reasoning, communication skills, and empathy. We compared AMIE's performance to that of primary care physicians (PCPs) in a randomized, double-blind crossover study of text-based consultations with validated patient actors in the style of an Objective Structured Clinical Examination (OSCE). The study included 149 case scenarios from clinical providers in Canada, the UK, and India, 20 PCPs for comparison with AMIE, and evaluations by specialist physicians and patient actors. AMIE demonstrated greater diagnostic accuracy and superior performance on 28 of 32 axes according to specialist physicians and 24 of 26 axes according to patient actors. Our research has several limitations and should be interpreted with appropriate caution. Clinicians were limited to unfamiliar synchronous text-chat which permits large-scale LLM-patient interactions but is not representative of usual clinical practice. While further research is required before AMIE could be translated to real-world settings, the results represent a milestone towards conversational diagnostic AI.


## 1 Introduction

The dialogue between the physician and the patient is fundamental to effective and compassionate care. The medical interview has been termed "the most powerful, sensitive, and most versatile instrument available to the physician" [1]. In some settings, it is believed that $60-80 \%$ of diagnoses are made through clinical history-taking alone [2-6]. The physician-patient dialogue extends beyond history-taking and diagnosis; it is a complex interaction which establishes rapport and trust, serves as a tool for addressing health needs and can empower patients to make informed decisions that account for their preferences, expectations, and concerns [7]. Clinicians wield considerable skills in clinical history-taking and the wider "diagnostic dialogue", but access to this expertise remains episodic and globally scarce [8].

Recent progress in general-purpose large language models (LLMs) [9-11] has shown that artificial intelligence (AI) systems have capabilities to plan, reason, and incorporate relevant context to hold naturalistic conversations. This progress affords an opportunity to rethink the possibilities of AI in medicine towards the development of fully interactive conversational AI. Such medical AI systems would understand clinical language, intelligently acquire information under uncertainty, and engage in natural, diagnostically useful medical conversations with patients and those who care for them. The potential real-world utility of AI systems capable of clinical and diagnostic dialogue is broad, as the development of such capabilities might improve access to diagnostic and prognostic expertise, to improved quality, consistency, availability, and affordability of care, and to help realize better health outcomes (particularly for populations facing healthcare disparities).

* Equal contributions. $\dagger$ Equal leadership.

$\ddagger$ Corresponding authors: \{taotu, mikeshake, alankarthi, natviv\}@google.com

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-02.jpg?height=607&width=1637&top_left_y=252&top_left_x=233)

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-02.jpg?height=294&width=929&top_left_y=1051&top_left_x=229)

Randomized Study Design for Remote Objective Structured Clinical Examination (OSCE)

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-02.jpg?height=455&width=661&top_left_y=957&top_left_x=1212)

AMIE Outperforms PCPs on Multiple Evaluation Axes for Diagnostic Dialogue

Figure $1 \mid$ Overview of contributions. AMIE is a conversational medical AI optimised for diagnostic dialogue. AMIE is instruction fine-tuned with a combination of real-world and simulated medical dialogues, alongside a diverse set of medical reasoning, question answering, and summarization datasets. Notably, we designed a self-play based simulated dialogue environment with automated feedback mechanisms to scale AMIE's capabilities across various medical contexts and specialities. Specifically, this iterative self-improvement process consisted of two self-play loops: (1) An "inner" self-play loop, where AMIE leveraged in-context critic feedback to refine its behavior on simulated conversations with an AI patient agent; (2) An "outer" self-play loop where the set of refined simulated dialogues were incorporated into subsequent fine-tuning iterations. During online inference, AMIE used a chain-of-reasoning strategy to progressively refine its response conditioned on the current conversation to arrive at an accurate and grounded reply to the patient in each dialogue turn. We designed and conducted a blinded remote Objective Structured Clinical Examination (OSCE) with validated simulated patient actors interacting with AMIE or Primary Care Physicians (PCPs) via a text interface. Across multiple axes corresponding to both specialist physician (28 out of 32 ) and patient actor (24 out of 26 ) perspective, AMIE was rated as superior to PCPs while being non-inferior on the rest.

However, while LLMs have been shown to encode clinical knowledge and proven capable of highly accurate single-turn medical question-answering [12-14], their conversational capabilities have been tailored to domains outside clinical medicine [15, 16]. Prior work in LLMs for health $[12-14,17,18]$ has not yet rigorously examined the clinical history-taking and diagnostic dialogue capabilities of AI systems or contextualized this by comparison to the extensive capabilities of expert clinicians.

Clinical history-taking and diagnostic dialogue through which clinicians derive diagnosis and management plans represent a complex skill [19] whose optimal conduct is highly dependent on context. Thus, multiple evaluation axes are needed to assess the quality of a diagnostic dialogue, including the structure and completeness of
the elicited history, diagnostic accuracy, the appropriateness of management plans and their rationale, and patient-centred considerations such as relationship-building, respect for the individual and communication efficacy [20]. If the conversational potential of LLMs is to be realized in medicine, there is a significant unmet need to better optimize development and evaluation of medical AI systems for characteristics such as these, which are unique to history-taking and diagnostic dialogue between clinicians and patients.

In this work, we detail our progress towards a conversational medical AI system for clinical history-taking and diagnostic reasoning.

Our key contributions are summarized as:

- We introduced AMIE (Articulate Medical Intelligence Explorer), an LLM based AI system optimized for clinical history-taking and diagnostic dialogue.
- To scale AMIE across a multitude of specialties and scenarios, we developed a novel self-play based simulated diagnostic dialogue environment with automated feedback mechanisms to enrich and accelerate its learning process. We also introduced an inference time chain-of-reasoning strategy to improve AMIE's diagnostic accuracy and conversation quality.
- We developed a pilot evaluation rubric to assess the history-taking, diagnostic reasoning, communication skills and empathy of diagnostic conversational medical AI, encompassing both clinician-centred and patient-centred metrics.
- We designed and conducted a blinded remote OSCE study with 149 case scenarios from clinical providers in Canada, the UK, and India, enabling randomized and counterbalanced comparison of AMIE to PCPs when performing consultations with validated patient actors. AMIE exhibited superior diagnostic accuracy compared to PCPs as assessed by various measures (e.g., top-1 and top-3 accuracy of the differential diagnosis list). Across 28 out of 32 evaluation axes from the specialist physician perspective and 24 out of 26 evaluation axes from the patient actor perspective, AMIE was rated superior to PCPs while being non-inferior on the rest.
- We performed a range of ablations to further understand and characterize the capabilities of AMIE, highlighted important limitations, and proposed key next steps for real-world clinical translation of AMIE.

Our research has important limitations, most notably that we utilized a text-chat interface, which although enabling potentially large-scale interaction between patients and LLMs specialized for diagnostic dialogue, was unfamiliar to PCPs for remote consultation. Thus our study should not be regarded as representative of usual practice in (tele)medicine.

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-04.jpg?height=1084&width=1672&top_left_y=236&top_left_x=232)

Figure $2 \mid$ Overview of randomized study design. A primary care physician (PCP) and AMIE perform (in a randomized order) a virtual remote Objective Structured Clinical Examination (OSCE) with simulated patients via online multi-turn synchronous text chat and produce answers to a post-questionnaire. Both the PCP and AMIE are then evaluated by both the patient actors as well as specialist physicians.

## 2 AMIE: An LLM based AI System for Diagnostic Dialogue

In the following sections, we describe the real-world datasets, simulated self-play environment, fine-tuning process, and inference time chain-of-reasoning that we designed to optimize AMIE for diagnostic conversation capabilities and clinical communication skills.

### 2.1 Real-world Datasets for AMIE

AMIE was developed using a diverse suite of real-world datasets including multiple-choice medical questionanswering, expert-curated long-form medical reasoning, electronic health record (EHR) note summaries, and large-scale transcribed medical conversation interactions. As described in detail below, in addition to dialogue generation tasks, the training task mixture for AMIE consisted of medical question-answering, reasoning, and summarization tasks.

Medical Reasoning. We used the MedQA (multiple-choice) dataset consisting of US Medical Licensing Examination (USMLE) multiple-choice style open domain questions with four or five possible answers [21]. The training set consisted of 11,450 questions and the test set had 1,273 questions. We also curated 191 MedQA questions from the training set where clinical experts crafted step-by-step reasoning leading to the correct answer [13].

Long-form Medical Question Answering. The dataset used here consisted of expert-crafted long-form responses to 64 questions from HealthSearchQA, LiveQA, and Medication QA in MultiMedBench [12].

Medical Summarization. A dataset consisting of 65 clinician-written summaries of medical notes from MIMIC-III, a large, publicly available database containing medical records of intensive care unit patients [22], was used as additional training data for AMIE. MIMIC-III contains approximately 2 million notes spanning 13 types including cardiology, respiratory, radiology, physician, general, discharge, case management, consult, nursing, pharmacy, nutrition, rehabilitation and social work. 5 notes from each category were selected, with a minimum total length of 400 tokens and at least one nursing note per patient. Clinicians were instructed to write abstractive summaries of individual medical notes, capturing key information while also permitting the inclusion of new informative and clarifying phrases and sentences not present in the original note.

Real-world Dialogue. Here, we used a de-identified dataset licensed from a dialogue research organisation comprising 98,919 audio transcripts of medical conversations during in-person clinical visits from over 1,000 clinicians over a 10-year period in the United States [23]. It covered 51 medical specialties (primary care, rheumatology, hematology, oncology, internal medicine and psychiatry among others) and 168 medical conditions and visit reasons (type II diabetes, rheumatoid arthritis, asthma, depression among the common conditions). Audio transcripts contained utterances from different speaker roles such as doctors, patients, and nurses. On average a conversation had 149.8 turns $\left(P_{0.25}=75.0, P_{0.75}=196.0\right)$. For each conversation, the metadata contained information about patient demographics, reason for the visit (follow-up for pre-existing condition, acute needs, annual exam and more), and diagnosis type (new, existing or other unrelated). We refer to [23] for more details.

For this study, we selected dialogues involving only doctors and patients, but not other roles such as nurses. During preprocessing, we removed paraverbal annotations such as "[LAUGHING]" and "[INAUDIBLE]" from the transcripts. We then divided the dataset into training (90\%) and validation (10\%) sets using stratified sampling based on condition categories and reasons for visits, resulting in 89,027 conversations for training and 9,892 for validation.

### 2.2 Simulated Dialogue Learning Environment and Self-play for AMIE

While passively collecting and transcribing real-world dialogues from in-person clinical visits is feasible, two substantial challenges limit its effectiveness in training LLMs for medical conversations: (1) existing real-world data often fails to capture the vast range of medical conditions and scenarios, hindering its scalability and comprehensiveness; (2) the data derived from real-world dialogue transcripts tends to be noisy, containing ambiguous language (including slang, jargon, and sarcasm), interruptions, ungrammatical utterances, and implicit references. This in turn, may limit AMIE's knowledge, capabilities, and applicability.

To address these limitations, we designed a self-play based simulated learning environment for diagnostic medical dialogues in a virtual care setting, enabling us to scale AMIE's knowledge and capabilities across a multitude of medical conditions and contexts. We used this environment to iteratively fine-tune AMIE with an evolving set of simulated dialogues in addition to the static corpus of medical $\mathrm{QA}$, reasoning, summarization, and real-world dialogue data described above (see Figure 1).

This process consisted of two self-play loops:

- An "inner" self-play loop where AMIE leveraged in-context critic feedback to refine its behavior on simulated conversations with an AI patient agent.
- An "outer" self-play loop where the set of refined simulated dialogues were incorporated into subsequent fine-tuning iterations. The resulting new version of AMIE could then participate in the inner loop again, creating a continuous learning cycle.

Simulated Dialogues. At each iteration of fine-tuning, we produced 11,686 dialogues, stemming from 5,230 different medical conditions. Conditions were selected from three datasets:

- Health QA dataset [12] which contained 613 common medical conditions.
- MalaCards Human Disease Database ${ }^{1}$ which contained 18,455 less common disease conditions.
- MedicineNet Diseases \& Conditions Index ${ }^{2}$ which contained 4,617 less common conditions.

At each self-play iteration, four conversations were generated from each of the 613 common conditions, while two conversations were generated from each of the 4,617 less common conditions randomly chosen from MedicineNet and MalaCards. The average simulated dialogue conversation length was 21.28 turns $\left(P_{0.25}=19.0, P_{0.75}=25.0\right)$.

Using simulated dialogues allowed us to address the limited availability of high-quality, labelled real-world conversation data and improved the model's generalization and adaptability to diverse medical contexts. By leveraging this self-play paradigm, AMIE could continuously learn and refine its conversational and diagnostic capabilities during patient interactions.

### 2.2.1 Simulated Dialogue Data Curation

In order to produce high-quality simulated dialogues at scale, we developed a novel multi-agent framework which comprised three key components:

- Vignette Generator: AMIE leverages web searches to craft unique patient vignettes given a specific medical condition.
- Simulated Dialogue Generator: Three LLM agents play the roles of patient agent, doctor agent, and moderator, engaging in a turn-by-turn dialogue simulating realistic diagnostic interactions.
- Self-play Critic: A fourth LLM agent acts as a critic to give feedback to the doctor agent for selfimprovement. Notably, AMIE acted as all agents in this framework. We describe each component in detail below.

Vignette Generator. The vignette generator aimed to create varied and realistic patient scenarios at scale, which could be subsequently used as context for generating simulated doctor-patient dialogues thereby allowing AMIE to undergo a training process emulating exposure to a greater number of conditions and patient backgrounds. The patient vignette (scenario) included essential background information such as patient demographics, symptoms, past medical history, past surgical history, past social history, and patient questions, as well as an associated diagnosis and management plan.

For a given condition, patient vignettes were constructed using the following process. First, we retrieved 60 passages (20 each) on the range of demographics, symptoms, and management plans associated with the condition from using an internet search engine. To ensure these passages were relevant to the given condition, we used the general-purpose LLM, PaLM-2 [10], to filter these retrieved passages, removing any passages deemed unrelated to the given condition. We then prompted AMIE to generate plausible patient vignettes aligned with the demographics, symptoms, and management plans retrieved from the filtered passages, by providing a one-shot exemplar to enforce a particular vignette format. The prompts for each of these steps are as follows:[^0]

## Search Retrieval Template

What are the specific patient demographics/symptoms/management plan for the condition [Condition]?

## Passage Filtering Template

For the clinical condition, [Condition], is the following a good description of common demographics/symptoms/management plans (Yes/No)?

Description: [Retrieved Passage]

Answer (Yes/No):

## Vignette Generation Template

The following are several passages about the demographics, symptoms, and management plan for a given condition. Generate 2 different patient vignettes consistent with these passages. Follow the format of the given example (just list N/A if a particular field is unavailable).

Condition: [Condition]

Demographic Passages: [Retrieved Demographic Passages]

Symptoms Passages: [Retrieved Symptom Passages]

Management Plan Passages: [Retrieved Management Plan Passages]

Example Format: [Oneshot example]

Patient Vignettes for [Condition]:

Simulated Dialogue Generator. Given a patient vignette detailing a specific medical condition, the simulated dialogue generator was designed to simulate a realistic dialogue between a patient and a doctor in an online chat setting where in-person physical examination may not be feasible.

Three specific LLM agents (patient agent, doctor agent, and moderator), each played by AMIE, were tasked with communicating amongst each other to generate the simulated dialogues. Each agent had distinct instructions. The patient agent embodied the individual experiencing the medical condition outlined in the vignette. Their role involved truthfully responding to the doctor agent's inquiries as well as raising any additional questions or concerns they may have had. The doctor agent played the role of an empathetic clinician seeking to comprehend the patient's medical history within the online chat environment [24]. Their objective was to formulate questions that could effectively reveal the patient's symptoms and background, leading to an accurate diagnosis and an effective treatment plan. The moderator continually assessed the ongoing dialogue between the patient agent and doctor agent, determining when the conversation had reached a natural conclusion.

The turn-by-turn dialogue simulation started with the doctor agent initiating the conversation: "Doctor: So, how can I help you today?"' Following this, the patient agent responded, and their answer was incorporated into the ongoing dialogue history. Subsequently, the doctor agent formulated a response based on the updated dialogue history. This response was then appended to the conversation history. The conversation progressed until the moderator detected the dialogue had reached a natural conclusion, when the doctor agent had provided a differential diagnosis, treatment plan, and adequately addressed any remaining patient agent questions, or if either agent initiated a farewell.

## Patient Agent Instruction:

You are a patient chatting with a doctor over an online chat interface. The doctor has never met you before. $<$ patient vignette $>$ Respond to the doctor's questions honestly as they interview you, asking any questions that may come up.

## Doctor Agent Instruction:

You are an empathetic clinician asking a patient about their medical history over an online chat interface. You know nothing about the patient in advance. Respond to the patient with a single-turn response to better understand their history and symptoms. Do not ask more than two questions. If the patient asks a question, be sure to answer it appropriately.

## Moderator Instruction:

The following is a conversation between a doctor and a patient: <dialog $>$ The conversation should only come to an end if the doctor has finished giving the patient a diagnosis and treatment plan and the patient has no questions left. A conversation also comes to an end if the doctor or patient says goodbye. Question: has the conversation come to an end? Yes or No.

Self-play Critic. To ensure high-quality dialogues, we implemented a tailored self-play [25] framework specifically for self-improvement of diagnostic conversations. This framework introduced a fourth LLM agent, acting as a "critic" which was also played by AMIE and aware of the ground truth diagnosis, to provide in-context feedback to the doctor agent and enhance its performance in subsequent conversations. The critic agent evaluated the doctor agent's responses based on the following criteria:

- The doctor agent exhibits empathy and professionalism while addressing the patient agent's latest questions or comments in a concise manner.
- The doctor agent avoids asking too many or repetitive questions (about information already acquired), focusing on a maximum of one or two per response.
- The responses should not reveal that the doctor agent is an AI chatbot. They should flow naturally, maintain factual accuracy, and facilitate further engagement from the patient.
- The doctor agent asks sufficient questions to identify at least two of the most likely differential diagnoses. They further refine their understanding through targeted questions towards the ground truth diagnosis and offer the corresponding treatment.

Following the critic's feedback, the doctor agent incorporated the suggestions to improve its responses in subsequent rounds of dialogue with the same patient agent from scratch. Notably, the doctor agent retained access to its previous dialogue history at each new round. This self-improvement process was repeated twice to generate the dialogues used for each iteration of fine-tuning.

### 2.3 Instruction Fine-tuning

AMIE, built upon the base LLM PaLM 2 [10], was instruction fine-tuned to enhance its capabilities for medical dialogue and reasoning. We refer to the PaLM-2 technical report for more details on the base LLM architecture.

We employed task-specific instructions to fine-tune AMIE in playing either the patient or doctor role within medical dialogues, performing medical question answering and reasoning, and summarizing EHR notes. While the first round of fine-tuning from the base LLM only used the static datasets, subsequent rounds of fine-tuning leveraged the simulated dialogues generated through the self-play inner loop as described in Section 2.2.1.

For dialogue generation tasks, AMIE was trained to predict the next conversational turn based on all previous interactions, assuming either the doctor or patient role. When playing the patient agent, AMIE was prompted to reply to the doctor agent's questions about their symptoms, drawing upon information provided in patient scenarios. These scenarios included patient vignettes (see Section 2.2.1) for simulated dialogues or metadata such as demographics, visit reason, and diagnosis type for the real-world dialogue dataset. In the doctor agent role, AMIE was prompted to act as an empathetic clinician, interviewing patients about their medical history
and symptoms to ultimately arrive at an accurate diagnosis. From each dialogue, we sampled on average 3 turns for each the doctor and patient roles as the target turns to predict based on the conversation leading up to that target turn. Target turns were randomly sampled from all turns in the dialogue that had a minimum length of 30 characters.

Similarly, for the EHR note summarization task, AMIE was provided with a clinical note and prompted to generate a summary of the note. Medical reasoning/QA and long-form response generation tasks followed the same setup as in [13]. Notably, all tasks except dialogue generation and long-form response generation incorporated few-shot (1-5) exemplars in addition to task-specific instructions for additional context.

### 2.4 Chain-of-reasoning for Online Inference

To address the core challenge in diagnostic dialogue - effectively acquiring information under uncertainty to enhance diagnostic accuracy and confidence while maintaining positive rapport with the patient - AMIE employed a chain-of-reasoning strategy before generating a response in each dialogue turn. Here, "chain-ofreasoning" refers to a series of sequential model calls, each dependent on the outputs of prior steps. Specifically, we used a three-step reasoning process, described as follows:

1. Analyzing patient information: Given the current conversation history, AMIE was instructed to 1) summarize the positive and negative symptoms of the patient as well as any relevant medical/family/social history and demographic information, 2) produce a current differential diagnosis, 3) note missing information needed for a more accurate diagnosis and 4) assess confidence in the current differential and highlight its urgency.
2. Formulating response and action: Building upon the conversation history and the output of step 1, AMIE performed the following: 1) Generate a response to the patient's last message and formulate further questions to acquire missing information and refine the differential diagnosis. 2) If necessary, recommend immediate action, such as an emergency room visit. If confident in the diagnosis based on available information, present the differential.
3. Refining the response: AMIE revises its previous output to meet specific criteria based on the conversation history and outputs from earlier steps. The criteria are primarily related to factuality and formatting of the response (e.g., avoid factual inaccuracies on patient facts and unnecessary repetition, show empathy, and display in a clear format).

This chain-of-reasoning strategy enabled AMIE to progressively refine its response conditioned on the current conversation to arrive at an informed and grounded reply.

## 3 Evaluation

Prior works developing models for clinical dialogue have focused on metrics such as the accuracy of note-todialogue or dialogue-to-note generations [26, 27], or natural language generation metrics such as BLEU or ROUGE scores that fail to capture the clinical quality of a consultation [28, 29].

In contrast to these prior works we sought to anchor our human evaluation in criteria more commonly used for evaluating the quality of physicians' expertise in history-taking, including their communication skills in consultation. We derived a framework from principles published in reviews of the consensus for best practices for patient-centered communication (PCCBP) in medical interviews [20], criteria examined for history-taking skills by the Royal College of Physicians in the UK as part of their Practical Assessment of Clinical Examination Skills (PACES) ${ }^{3}$ [30], and criteria proposed by the UK General Medical Council Patient Questionnaire (GMCPQ) ${ }^{4}$ for doctors seeking patient feedback as part of professional re-validation ${ }^{5}$. We iterated upon these criteria to refine items for inclusion and derived pilot scales and instructions for assessment by using focus groups and interviews with clinicians and OSCE examiners based in the UK, Canada, US, and India. Our resulting pilot framework enabled assessment from two perspectives: clinician (board-certified[^1]physicians) and lay raters (patient actors). The framework included consideration of consultation quality, structure and completeness, the roles, responsibilities, and skills of the interviewer (Tables A.1, A.2, A.3, and A.4).

### 3.1 Objective Structured Clinical Examination

Objective Structured Clinical Examination (OSCE) is a practical assessment format used in healthcare to assess clinical skills and competencies in a standardized and objective fashion [31-33]. It differs from traditional written or oral exams that focus primarily on theoretical knowledge and instead aims to provide an environment in which the skills of real-world clinical practice might be assessed.

The OSCE is typically divided into multiple stations (often 8-12), each simulating a real-life clinical scenario enacted by standardized patient actors trained to portray specific symptoms or conditions based on pre-defined scenario descriptions. At each station, students are given specific tasks to perform, such as taking a clinical history, or making a diagnosis. Each station has a set time limit, ensuring fairness and efficient assessment. Trained examiners observe students' performance at each station using a pre-defined checklist or marking scheme. They assess clinical skills like communication, history-taking, physical examination techniques, clinical reasoning, and decision-making.

### 3.2 Remote OSCE Study Design

To compare AMIE's performance to that of real clinicians, we conducted a randomized crossover study of blinded consultations in the style of a remote OSCE. Our OSCE study involved 20 board-certified primary care physicians (PCPs) and 20 validated patient actors, 10 each from India and Canada, respectively, to partake in online text-based consultations. PCPs had between 3 and 25 years of post-residency experience (median 7 years). Patient actors comprised of a mix of medical students, residents, and nurse practitioners with experience in OSCE participation. We sourced 149 scenario packs from India (75), Canada (60), and the UK (14).

The scenario packs and simulated patients in our study were prepared by two OSCE laboratories (one each in Canada and India), each affiliated to a medical school and with extensive experience in preparing scenario packs and simulated patients for OSCE examinations. UK scenario packs were sourced from the samples provided on the MRCPUK website. Each scenario pack was associated with a ground truth diagnosis and a set of acceptable diagnoses. The scenario packs covered conditions from cardiovascular (29), respiratory (30), gastroenterology (31), neurology (30), urology, obstetric, and gynecology domains (15), and internal medicine (14). Pediatric or psychiatry domains were excluded from this study, as were intensive care or inpatient case management scenarios.

Indian patient actors played the roles in all India scenario packs and 7 of the 14 UK scenario packs. Canadian patient actors participated in scenario packs for both Canada and the other half of UK-based scenario packs. This assignment process resulted in 149 distinct simulated patients ("scenarios"). Below, we use the term "OSCE agent" to refer to the conversational counterpart interviewing the patient actor, i.e., either PCP or AMIE. Table 1 summarizes the OSCE assignment information across three geographical locations. Each of the 149 simulated patients completed the three-step study flow depicted in Figure 2.

Table 1 | OSCE study summary. Number of scenario packs, patient actors, simulated patients, and primary care physicians (PCPs) in each of the three locations (Canada, India, and the UK) in the remote OSCE study. 20 board-certified PCPs participated in the study as OSCE agents in comparison with AMIE, 10 each from India and Canada. 20 trained patient actors were involved, with 10 each from India and Canada. Indian patient actors played the roles in both India and UK scenario packs. Canadian patient actors participated in scenario packs for both Canada and the UK. This process resulted in 149 distinct simulated patients.

| Location | \# of Scenario Packs | \# of Simulated Patients | \# of Patient Actors | \# of PCPs |
| :---: | :---: | :---: | :---: | :---: |
| Canada | 60 | 67 | 10 | 10 |
| India | 75 | 82 | 10 | 10 |
| UK | 14 | 0 | 0 | 0 |
| Total | $\mathbf{1 4 9}$ | $\mathbf{1 4 9}$ | $\mathbf{2 0}$ | $\mathbf{2 0}$ |

### 3.2.1 Online Text-based Consultation

PCPs and patient actors were primed with sample scenarios and instructions, and participated in pilot consultations prior to the study commencing in order to familiarize themselves with the interface and experiment requirements.

For the experiment, each simulated patient completed two online text-based consultations via a synchronous text chat interface (Figure A.2), one with a PCP (control) and one with AMIE (intervention). The ordering of PCP and AMIE was randomized and patient actors were not informed as to which they were talking to in each consultation. PCPs were located in the same country as patient actors, and were randomly drawn based on availability at the specified time slot for the consultation. Patient actors role-played the scenario and were instructed to conclude the conversation after no more than 20 minutes. Both OSCE agents were asked (PCPs via study-specific instructions, and AMIE as part of the prompt template) to not reveal their identity, or whether they were human, under any circumstances.

### 3.2.2 Post-questionnaires

Upon conclusion of the consultation, the patient actor and OSCE agent each filled in a post-questionnaire in light of the resulting consultation transcript (Figure A.3). The post-questionnaire for patient actors consisted of the complete GMCPQ (Table A.1), the PACES components for "Managing Patient Concerns" and "Maintaining Patient Welfare" (Table A.2), and a checklist representation of the PCCBP category for "Fostering the Relationship" (Table A.3). Responses patient actors provided to the post-questionnaire are referred to as "patient actor ratings" below. The post-questionnaire for the OSCE agent asked for a ranked differential diagnosis (DDx) list with a minimum of 3 and no more than 10 conditions, as well as recommendations for escalation to in-person or video-based consultation, investigations, treatments, management plan, and the need for a follow-up.

### 3.2.3 Specialist Physician Evaluation

Finally, a pool of 23 specialist physicians from India (14), North America (6), and the UK (3) evaluated PCPs and AMIE with respect to the quality of their consultation, and their responses to the post-questionnaire. During evaluation, specialist physicians also had access to the full scenario pack along with its associated ground truth differential and additional accepted differentials. All of the data the specialist physicians had access to during evaluation are collectively referred to as "OSCE data" below. Specialist physicians were sourced to match the specialties and geographic regions corresponding to the scenario packs included in our study, and had between 1 and 36 years of post-residency experience (median 5 years). Each set of OSCE data was evaluated by one specialist physician randomly assigned to match the specialty and geographic region of the underlying scenario (e.g., Canadian pulmonologist evaluated OSCE data from Canada-sourced respiratory medicine scenario). Each specialist evaluated OSCE data from both PCP and AMIE for a given scenario. Evaluations for PCP and AMIE were conducted by the same specialist in a randomized and blinded sequence.

Evaluation criteria included the accuracy, appropriateness and comprehensiveness of the provided DDx list, appropriateness of recommendations regarding escalation, investigation, treatment, management plan and follow-up (Table A.4), and all PACES (Table A.2) and PCCBP (Table A.3) rating items. We also asked specialist physicians to highlight confabulations in the consultations and questionnaire responses, i.e., text passages that were non-factual or referred to information not provided in the conversation. Each OSCE scenario pack additionally supplied specialists with scenario-specific clinical information to assist with rating the clinical quality of the consultation, such as the ideal investigation or management plans; or important aspects of the clinical history that would ideally have been elucidated for the highest quality of consultation possible.

### 3.3 Auto-evaluation

In addition to human evaluations, we implemented model-based auto-evaluation methods as economical consistent alternatives to specialist assessments. These techniques were employed to evaluate both dialogue quality and diagnostic accuracy of the OSCE agent. To establish the validity of our auto-evaluation methods for assessing dialogue quality, we initially focused on a subset of four evaluation axes from the PACES rubric

(Table A.2) that were assessed by both the patient actors and the specialist physicians. The auto-evaluation, which uses a self-CoT strategy (details described in Section A.9) with AMIE to rate dialogues, was in good alignment with human raters and comparable to the inter-specialist agreement on these criteria. For the auto-evaluation of differential diagnoses, we leveraged another LLM, Med-PaLM 2 [13] as a surrogate for a specialist rater to grade the predicted diagnoses against the ground truth diagnoses (more details in Section A.7). Our auto-evaluation on DDx accuracy showed a similar trend for AMIE and OSCE agents compared to the specialist ratings. Overall, auto-evaluation trends aligned with human ratings for both dialogue quality and diagnostic accuracy.

We also conducted additional auto-evaluation analyses for the following purposes:

- To compare the performance of the DDx accuracy derived from AMIE or PCP consultations;
- To compare the DDx accuracy between simulated patients performed in Canada and India and determine if there is systematic differences between the two locations;
- To isolate the effects of information acquisition and information interpretation by analyzing the DDx accuracy of AMIE when provided the PCP consultation instead of its own;
- To evaluate the efficiency of information acquisition between AMIE and PCPs by analyzing the DDx accuracy as the number of conversation turns increases;
- To evaluate the benefit of inner-loop self-play on dialogue quality before and after critic feedback.


### 3.4 Statistical Analysis

We evaluated the top-k accuracy of the DDx lists generated by AMIE and PCPs across all 149 simulated patients. Top-k accuracy was defined as the percentage of cases where the correct diagnosis appeared within the top-k positions of the DDx list. Specifically, a candidate diagnosis was considered a match if the specialist rater marked it as either an exact match with, very close to or closely related to the ground truth diagnosis (or accepted differential). Statistical significance for DDx accuracy was determined using bootstrap tests [34] with 10,000 samples and false discovery rate (FDR) correction [35] across all k. Statistical significance for patient actor and specialist ratings was determined using Wilcoxon signed-rank tests [36] FDR correction. Cases where either agent received "Cannot rate / Does not apply" were excluded from the test. Results below refer to $p$-values after FDR correction.

## 4 Results

### 4.1 Diagnostic Accuracy

### 4.1.1 AMIE showed higher DDx accuracy than PCPs under specialist physician evaluation.

AMIE's diagnostic accuracy was assessed as higher than that of PCPs. Figure 3 shows the top-k accuracy for AMIE and PCPs, considering matches with the ground truth diagnosis (a) and matches with any item on the accepted differential (b). AMIE showed significantly higher top-k accuracy than that of PCPs across all values of $\mathrm{k}(p<0.05)$. Note that unlike AMIE, PCPs did not always provide 10 diagnoses in their differential diagnoses (min: 3, mean: 5.39). Additionally, we performed a comparison of DDx accuracy between AMIE and PCP by varying the matching criteria for determining a match. Results depicted in Figure A. 7 further substantiate AMIE's superior DDx performance across various matching criteria.

Accuracy by Specialty. Figure A. 8 illustrates the DDx accuracy achieved by AMIE and PCPs across the six medical specialties covered by scenarios in our study. We observed that AMIE's performance matched or surpassed PCP performance for all specialties with the most pronounced improvements in the respiratory and cardiovascular specialities.

### 4.1.2 Auto-evaluation suggested AMIE matched PCPs' efficiency in acquiring information.

Auto-evaluation Accuracy. We reproduced the DDx accuracy analysis with our model-based auto-evaluator instead of the specialist raters using the same procedure as in Figure 3. The overall performance trends obtained through the auto-evaluator align well with specialist assessments despite marginal differences in the

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-13.jpg?height=626&width=1627&top_left_y=240&top_left_x=257)

$\mathbf{a}$

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-13.jpg?height=583&width=786&top_left_y=278&top_left_x=279)

b

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-13.jpg?height=577&width=787&top_left_y=281&top_left_x=1081)

Figure $3 \mid$ Specialist-rated top-k diagnostic accuracy. AMIE and PCPs top-k DDx accuracy are compared across 149 scenarios with respect to the ground truth diagnosis (a) and all diagnoses in the accepted differential (b). Bootstrapping $(\mathrm{n}=10,000)$ confirms all top-k differences between AMIE and PCP DDx accuracy are significant with $p<0.05$ after FDR correction.

computed accuracy values, as shown in Figure A.9.

Isolating the Source of Performance Gains. To investigate whether AMIE's superior DDx performance observed in Figure 3 stemmed from improved information acquisition or from better diagnostic reasoning capability, we compared AMIE's diagnoses based on its own consultations with AMIE's diagnoses generated from the corresponding PCP consultations, using the DDx auto-evaluator. Results depicted in Figure A. 10 revealed markedly similar DDx performance, indicating that the diagnostic performance remained consistent regardless of whether AMIE processed information from its own dialogue or from the PCP's conversation. Both methods significantly outperformed the differential diagnoses produced by PCPs. These results suggest that AMIE was approximately equivalent to PCPs at information acquisition but better than PCPs at interpreting that information to produce an accurate/complete differential diagnosis.

Efficiency of Information Acquisition. Although AMIE displayed greater verbosity compared to PCPs in terms of total number of words generated in their responses during the consultation, the number of conversational turns and the number of words elicited from the patient actors were similar across both OSCE agents, as illustrated in Figure A.11. This suggests that both AMIE and PCPs acquired a similar amount of information from the patients during the encounter. To investigate how efficient AMIE or PCPs were at gathering sufficient information to formulate a correct diagnosis, we truncated the conversations at various turn counts and used AMIE to generate differential diagnoses based on these partial conversations. Figure A. 12 depicts the top-3 DDx accuracy as a function of the number of turns provided to the model. The observed accuracies plateaued within the initial 10 conversational turns for both AMIE and PCPs. This suggests that both AMIE and PCPs were able to acquire the information necessary for formulating a diagnosis within the early stages of the conversation. Additionally, the comparable performance at every turn indicates that neither AMIE nor PCPs had a significant advantage in the efficiency or quality of information acquisition.

### 4.2 Conversation Quality

### 4.2.1 AMIE surpassed PCPs in conversation quality, per specialists and patient actors.

Conversation quality was assessed using patient actor ratings, specialist ratings, and outputs from autoevaluation. Figure A. 5 and A. 6 show two example consultations for the same simulated patient from AMIE and $\mathrm{PCP}$, respectively.
![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-14.jpg?height=1268&width=1650&top_left_y=558&top_left_x=234)

```
AMIE (top)
\square PCP (bottom)
-■ Very favorable
-_ Favorable (or "Yes" for Y/N)
- Neither favorable nor unfavorable
-1 Unfavorable (or "No" for Y/N)
-- Very unfavorable
L-_ Cannot rate / Does not apply
```

Figure $4 \mid$ Patient actor ratings. Conversation qualities as assessed by patient actors upon conclusion of the consultation. For illustration purposes, all responses from five-point rating scales were mapped to a generic five-point scale ranging from 'Very favorable' to 'Very unfavorable'. For Yes/No questions, a (positive) 'Yes' response was mapped to the same color as 'Favorable' and a (negative) 'No' response to the same color as 'Unfavorable'. Rating scales were adapted from the General Medical Council Patient Questionnaire (GMCPQ), the Practical Assessment of Clinical Examination Skills (PACES), and a narrative review about Patient-Centered Communication Best Practice (PCCBP). Details on question wording and response options are provided in Section A.1. Asterisks represent statistical significance $(*: p<0.05, * *: p<0.01, * * *: p<0.001$, n.s. : not significant).

Patient Actor Ratings. Figure 4 presents the various conversation qualities patient actors assessed following their consultations with the OSCE agents. Overall, AMIE's consultations were rated significantly better $(p<0.05)$ by patient actors than those from PCPs across 24 of 26 axes. No significant differences in ratings were detected for the two PCCBP axes "Respecting Patient's Privacy" ( $\mathrm{N}=108)$ and "Acknowledging Mistakes" $(\mathrm{N}=41)$. For the latter criterion, the number of exclusions was substantially higher since the question applied only when mistakes were made by the OSCE agent and pointed out in the conversation.

Specialist Physician Ratings. Specialist physicians evaluated both the conversational quality as well as the responses to the post-questionnaire for scenarios within their domain expertise (see Figure 5). Again, AMIE's responses were rated significantly better by specialists than those from PCPs on 28 of 32 evaluation axes; Specialists preferred AMIE's consultation, diagnoses, and management plan over those from PCPs. For this set of evaluations, differences in specialist ratings between AMIE and PCPs were statistically significant $(p<0.05)$. No significant differences in ratings were detected for four of the axes in the Diagnosis \& Management rubric, namely, "Escalation Recommendation Appropriate", "Treatment Inappropriate Avoided", "Followup Recommendation Appropriate" and "Confabulation Absent", despite no exclusions ( $\mathrm{N}=149$ ).

### 4.2.2 Auto-evaluations demonstrated the effectiveness of inner self-play for AMIE.

Auto-evaluation of Conversation Ratings. We leveraged the model-based self-CoT auto-evaluation strategy to rate conversations on four evaluation axes from the PACES rubric, and validated that these auto-evaluation ratings were accurate and well aligned with the specialist ratings (Figures A. 17 and A.18). Furthermore, to demonstrate that the inner self-play loop improved simulated dialogue quality, we applied the auto-evaluation method to the simulated dialogues generated before and after the self-play procedure. Results in Figure A. 19 revealed that the simulated dialogues after self-play were preferred more often than the baseline dialogues without self-critique.

## 5 Related Work

### 5.1 Clinical History-taking and the Diagnostic Dialogue

History-taking and the clinical interview are widely taught in both medical schools' and postgraduate curricula [37-42]. Consensus on physician-patient communication has evolved to embrace patient-centred communication practices, with recommendations that communication in clinical encounters should address six core functions: fostering the relationship, gathering information, providing information, making decisions, responding to emotions and enabling disease- and treatment-related behavior [20, 43, 44]. Specific skills and behaviours for meeting these goals have also been described, taught and assessed [20, 45] with validated tools [45]. Medical conventions consistently cite that certain categories of information should be gathered during a clinical interview, comprising topics such as the presenting complaint, past medical history and medication history, social and family history, and systems review [46, 47]. Clinicians' ability to meet these goals is commonly assessed using the framework of an objective structured clinical examination (OSCE) [31-33]. Such assessments vary in their reproducibility or implementation and have even been adapted for remote practice as virtual OSCEs (vOSCEs) with telemedical scenarios, an issue of particular relevance during the COVID-19 pandemic [48].

### 5.2 Conversational AI and Goal-oriented Dialogue

Conversational AI systems for goal-oriented dialogue and task completion have a rich history [49-51]. The emergence of transformers [52] and large language models [15] have led to renewed interest in this direction. The development of strategies for alignment [53], self-improvement [54-57] and scalable oversight mechanisms [58] have enabled large scale deployment of such conversational systems in the real world [16, 59]. However, the rigorous evaluation and exploration of conversational and task-completion capabilities of such AI systems remains limited for clinical applications, where studies have largely focused on single-turn interaction use cases such as question-answering or summarization.

![](https://cdn.mathpix.com/cropped/2024_06_04_ef708cef684d05345ef1g-16.jpg?height=1561&width=1651&top_left_y=388&top_left_x=237)

Figure 5 | Specialist physician ratings. Conversation and reasoning qualities as assessed by specialist physicians. For illustration purposes, all responses from five-point rating scales were mapped to a generic five-point scale ranging from 'Very favorable' to 'Very unfavorable'. The only four-point scale (DDx Comprehensiveness) was mapped to the same scale, ignoring the 'Neither favorable nor unfavorable' option. For Yes/No questions, a (positive) 'Yes' response was mapped to the same color as 'Favorable' and a (negative) 'No' response to the same color as 'Unfavorable'. Rating scales were adapted from the Practical Assessment of Clinical Examination Skills (PACES), a narrative review about Patient-Centered Communication Best Practice (PCCBP), and other sources. Details on question wording and response options are provided in Section A.1. Asterisks represent statistical significance $(*: p<0.05, * *: p<0.01, * * *: p<0.001$, n.s. : not significant $)$.

### 5.3 AI for Medical Consultations and Diagnostic Dialogue

The majority of explorations of AI as tools for conducting medical consultations have focused on "symptom checker" applications rather than a full natural dialogue, or on topics such as transcription of medical audio or the generation of plausible dialogue given clinical notes or summaries [60-63]. Language models have been trained using clinical dialogue datasets but not comprehensively evaluated [64]. Studies have been grounded in messages between doctors and patients in commercial chat platforms (which may have altered doctor-patient engagement compared to 1:1 medical consultations) $[28,65,66]$. Many focused largely on predicting next turns in the recorded exchanges rather than clinically meaningful metrics. And to date, there have been no reported studies that have examined the quality of AI models for diagnostic dialogue using the same criteria that are used to examine and train human physicians in dialogue and communication skills; nor evaluating AI systems in common frameworks such as the OSCE.

### 5.4 Evaluation of Diagnostic Dialogue

Prior frameworks for human evaluation of AI systems' performance in diagnostic dialogue have been limited in detail. They have not been anchored in established criteria for assessing communication skills and the quality of history-taking. For example, [29] reported a 5-point scale describing overall "human evaluation", [65] reported "relevance, informativeness and human likeness", [66] reported "fluency, expertise and relevance", [67] "fluency and adequacy" and [68] "fluency". These criteria are far less comprehensive and specific than those taught and practiced by medical professionals. A multi-agent framework for assessing conversational capabilities of LLMs is introduced in [64], however, the study was performed in the restricted setting of dermatology, used AI models to emulate both doctor and patient sides of simulated interactions, and performed limited expert evaluation of history-taking as "complete" or not.

## 6 Discussion

In this study, we introduced AMIE, an LLM based AI system optimised for clinical dialogue with diagnostic reasoning capabilities. We compared AMIE consultations to those performed by PCPs using a randomized, double-blind crossover study with human simulated patients in the style of an Objective Structured Clinical Examination (OSCE). Notably, our study was not designed to be representative of clinical conventions either for traditional OSCE evaluations, for remote- or tele-medical consultation practices, or for the ways clinicians usually use text and chat messaging to communicate with patients. Our evaluation instead mirrored the most common way by which people interact with LLMs today, leveraging a potentially scalable and familiar mechanism for AI systems to engage in remote diagnostic dialogue. In this setting, we observed that AMIE, an AI system optimised specifically for the task, outperformed PCPs on simulated diagnostic conversations when evaluated along multiple clinically-meaningful axes of consultation quality.

Diagnostic Performance. The differential diagnoses provided by AMIE were more accurate and complete than those provided by board-certified PCPs, when both were evaluated by specialist physicians. Previous research has shown that AI systems may match or exceed human diagnostic performance in specific, narrow tasks [69-71] in retrospective evaluation. However, these situations typically involved both AI and physicians interpreting the same fixed input (for example, identifying the presence of a specific finding in a medical image). Our study was significantly more challenging because it required the AI system to actively acquire relevant information through conversation rather than relying on clinical information collated by human efforts [72]. Therefore the system's downstream differential diagnoses depended on not only its diagnostic inference capability, but also the quality of information gathered under uncertainty through natural conversation and building rapport.

Our results suggested that AMIE was as adept as PCPs in eliciting pertinent information during the simulated consultations and was more accurate than PCPs in formulating a complete differential diagnosis if given the same amount of acquired information. This finding corroborates other work that LLMs may be able to produce more complete differential diagnoses given the same clinical information as physicians in challenging cases [70]. Though not explored in this study, the assistive performance of AMIE therefore represents an interesting and important avenue for future research, particularly given the real-world importance of expert
oversight for AI systems in safety-critical settings such as medicine.

Our study utilized a wide variety of simulated patients, comprising actors trained in both Canada and India and scenarios across a range of specialties. This allowed us to explore how performance varied along multiple axes: by specialty, and by the locations in which the scenario was derived and enacted. We observed that both PCPs and AMIE performed worse in obstetric/gynecology and internal medicine scenarios than those from other specialties (see Figure A.8). The study was not powered or designed to compare performance between different specialty topics, and we cannot exclude that the scenarios in some specialties might be harder than others. We observed that both AMIE and PCPs had higher diagnostic accuracy in consultations performed in the Canada OSCE lab compared to those enacted in the India OSCE lab (see Figure A.13). However, the differences were not statistically significant and in a subset of 40 scenarios enacted in both the Canada OSCE lab and the India OSCE lab, the performance of both AMIE and PCPs was equivalent (see Figure A.14).

Conversational Performance. Patient actors and specialist raters both evaluated AMIE's performance to be higher than PCPs on metrics related to empathy and communication skills. These axes comprised a majority of the dimensions that were evaluated. This general finding is consistent with a prior study where LLM responses were found to be more empathetic than the responses from clinicians to health questions posted on Reddit [73]. However, the findings in that study may not be generalised directly to our setting due to the differences in study design. Specifically, prior work has not involved a direct, randomised comparison of physicians and AI systems in a prospective simulation of multi-turn dialogue with the same patient. In both settings, the lack of voice-based and non-verbal visual communication may be an unfair disadvantage to clinicians.

The text-based chat interface used in this study introduces both advantages and disadvantages. People today most commonly engage with LLMs through synchronous text-chat interfaces [74], and patients often use patient portals to send messages to their providers. We therefore chose this mode of interaction as a representative interface for LLMs to perform multi-turn conversation, adapting the virtual OSCE framework accordingly. While this allowed a fair comparison of diagnostic dialogue between LLMs and clinicians when both were restricted to a synchronous text-chat, it is important to acknowledge that our experiments do not emulate the expected quality of diagnostic dialogue in real clinical practice (including telemedicine). Physicians may be more used to history-taking and diagnostic dialogue by telephone or video consultation than synchronous text-chat communication [75, 76]. Instead, text is more commonly used by clinicians to communicate with patients for episodic or asynchronous needs such as prescription refills or communication about specific test results [77]. Physicians may thus be more familiar with text/SMS or email rather than the synchronous text-chat medium we employed in this study. In both text/SMS and email, the conventions and expectations for communicating naturally and with empathic style might be different [78]. It is possible that the PCPs in our study had not yet become accustomed to the setting, and may have performed differently if subjected to a specific training program (similar in spirit to the training process for AMIE). Clinicians participating in the study undertook two preparatory pilot sessions of consultations with our synchronous text interface before the evaluation began, but this was not a formal training program, nor was it designed to optimize clinicians' performance. Future research could explore this question more thoroughly including monitoring for the impact of a learning curve, or exploring whether performance varies according to the extent to which participating clinicians or simulated patients are familiar with telemedicine.

Additionally, our findings regarding empathic communication could also be partially attributed to the fact that AMIE responses were significantly longer than clinician responses (shown in Figure A.11), and presented with greater structure. This could potentially suggest to an observer that more time was spent preparing the response, analogous to known findings that patient satisfaction increases with time spend with their physicians $[79-81]$.

Collectively, our findings suggest many avenues for further research that might leverage human-AI complementarity [82], combining clinicians' skills in the analysis of verbal and non-verbal cues with the potential strengths of LLMs to suggest more enriched conversational responses including empathic statements, structure, eloquence, or more complete differential diagnoses.

Simulated Dialogue. The use of simulated data allowed us to quickly scale training to a broad set of conditions and patient contexts, while the injection of knowledge from search encouraged these dialogues to
remain grounded and realistic. Though the simulated patients encompassed a wide range of conditions, they failed to capture the full range of potential patient backgrounds, personalities, and motivations. Through the inner self-play procedure, we were able to iteratively improve the simulated dialogue we generated and used in fine-tuning. However, these improvements were limited by our ability to articulate what makes a good dialogue in the critic instructions, the critic's ability to produce effective feedback, and AMIE's ability to adapt to such feedback. For example, in the simulated environment we impose that AMIE reaches a proposed differential and testing/treatment plan for the patient, but such an endpoint may be unrealistic for some conditions, especially in the virtual chat-based setting.

Evaluation Framework. In contrast to prior works, we anchored our evaluation in criteria already established to be relevant for assessing physicians' communication skills and history-taking quality. We performed more extensive and diverse human evaluation than prior studies of AI systems, with ratings from both clinicians and simulated patients perspective. Our raters and scenarios were sourced from multiple geographic locations, including North America, India and the UK. Our pilot evaluation rubric is, to our knowledge, the first to evaluate LLMs' history-taking and communication skills using axes that are also measured in the real world for physicians themselves, increasing the clinical relevance of our research. Our evaluation framework is considerably more granular and specific than prior works on AI-generated clinical dialogue, which have not considered patient-centred communication best practice or clinically-relevant axes of consultation quality [29, $64-68]$.

However, our pilot framework is not definitive and can be further improved in future research. History-taking itself is contextual and what determines a "good history" is dependent on the specific clinical situation, patient and physician attributes, cultural characteristics, and many other factors. Despite variation in models for clinical history-taking [83-86], studies have shown that good clinical interviews are associated with not only problem detection and diagnostic accuracy, but also quadruple aims for care delivery [87, 88] ranging from patient and physician satisfaction, resilience to stress and illness, and health outcomes or cost. Future studies on the quality of LLM history-taking might therefore utilise prospective measures of these outcomes in real-world settings (for example reductions in patient complaints [89], or improvements in cost and care effectiveness, patient and provider satisfaction), though evaluations as such may be challenging or impractical to compare to standard practice in the same individual patient, and randomisation of different approaches may also be challenging in real-world settings.

Breadth of Evaluation. Our chosen axes of evaluation were not exhaustive and their interpretation was often subjective in nature. Although we conducted evaluations from both clinician and lay-perspectives, generating scenario-packs in three countries with assessors in both North America and India, the pool of clinicians and lay-people assessing the models could be expanded further to improve generalization of our insights. Our experiments could also undergo more extensive replication to explore other aspects such as inter-observer and inter-participant variability, including future work with an intentionally further diversified pool of human raters (clinicians and lay users). Participatory design in the development of model evaluation tools with a representative pool of patients, as well as clinical and health equity domain experts, could also be valuable.

Although our scenarios comprised many different clinical conditions and specialties, our experiments were not necessarily representative of the decades of clinical practice accumulated by even a single doctor (who on average may perform tens of thousands of consultations in a career [90]). The range of conditions possible to examine in medicine is vast as is the variation in presentation of individual diseases. Our experiments were not designed to examine multi-morbidity and co-incident pathology, longitudinal case presentation or the consideration of sequential information from clinical investigations. We excluded entirely some clinical settings or specialties such as psychiatry, pediatrics, intensive care, and inpatient case management scenarios. Further research would be needed to understand the applicability of our findings in many settings such as these, where the requirements for high-quality history-taking might differ [91, 92]. The OSCE framework is commonly used in the assessment of clinicians' skills. It encompasses a significant range of methodologies including real or simulated patients, interaction with physical artefacts or clinical materials, applications to a variety of medical specialties, tasks or settings; and both remote or in-person assessments. Although the OSCE approach is popular, there are significant limitations to its validity [93]. We utilised a remote text-based
assessment, replicating known issues with the paradigm of "virtual OSCE" such as the inability to incorporate non-verbal symptoms, signs and communication features. Additionally, this format could introduce unfamiliar constraints to the communication of PCP participants [48].

The tone, content, and nature of the OSCE dialogues in our study are likely not to be representative of real-world patient populations. For example, patient actors may have described their symptoms with greater structure, depth or clinical detail than could be routinely expected in many consultations, or had greater comprehension of clinical context than would be ordinarily expected. Furthermore, although evaluation was blinded, the style of responses from AMIE was notably different to that by PCPs which limits the practical extent of blinding in study design.

Therefore even within the distribution of diseases and specialties we addressed, our findings should be interpreted with humility and caution. There is a need for further research to examine varied presentations of the same diseases, alongside exploration of alternate approaches to evaluating history-taking and clinical dialogue in situations of different patient needs, preferences, behaviours and circumstances.

Fairness and Bias. The evaluation protocol presented in this paper is limited in terms of its ability to capture potential issues related to fairness and bias, which remains an important open question that we will aim to address in subsequent system evaluations. Recent advances in the development of comprehensive frameworks for bias detection in large language models [94, 95] present a promising starting point for establishing such an approach. It should be noted that medical diagnostic dialogue is a particularly challenging use case, due to the complexity of the medical domain, the interactive information gathering nature of the dialogue, and the outcome-driven setting, with the potential of associated harms in case of incorrect diagnosis or incorrect medical advice. Nevertheless, disentangling these issues is an important further research area if LLMs in the domain are to overcome rather than propagate inequities in healthcare. For example, previous studies have found that physicians approach communication with their patients differently, on average, depending on patients' race, resulting in Black patients receiving communication that was less patient-centered, and with a lower positive affect [96]. Other studies have found differences in physicians' communication styles and conversation length based on gender [97]. Effective intercultural communication skills are essential [91]. There is therefore a non-negligible risk that such historical conversational biases may be replicated or amplified in an AI dialogue system, but at the same time there is also an opportunity to work towards designing conversational systems that can be more inclusive, and more personalized to the individual patient's needs.

To help inform the development of the necessary fairness, bias, and equity frameworks, it is important to employ a participatory approach to solicit representative views across a wide range of patient demographics, as well as clinical and health equity domain experts. Such evaluation frameworks should be complemented by extensive model red teaming and an adversarial approach to identifying any remaining gaps and failure modes. Recent advances in red teaming LLMs could be useful in this scenario [98-101]. These practices should not only inform the evaluation of the final model, but also its development and iterative refinement. Model development should follow the established data and model reporting practices and provide transparency into the training data and the associated decision processes [102-104]. The dialogue research dataset contributing to AMIE training data in our study was de-identified, reducing the availability of socio-economic factors, patient demographics, and information about clinical settings and locations.

Further work is also needed to ensure the robustness of medical LLMs in multilingual settings [105-108], and particularly their performance in low-resource languages [109]. The great variety of cultures [110], languages, localities, identities, and localized medical needs, makes the task of generating a priori static yet comprehensive fairness benchmarks practically infeasible. Measurement and mitigation of bias must move beyond the traditional narrow focus on specific axes that fails to scale globally [111]. LLM-based evaluators present a potential solution for preliminary assessments in languages where there are no systematic benchmarks, though prior studies have found these auto-evaluation frameworks to be biased, underscoring the need for calibrating them on native speaker evaluations, and using them with caution [112].

Deployment. This research demonstrates the potential of LLMs for future use in healthcare in the context of diagnostic dialogue. Transitioning from an LLM research prototype that has been evaluated in this study to a safe and robust tool that can be used by healthcare providers, administrators, and people will require significant additional research to ensure the safety, reliability, efficacy, and privacy of the technology. Careful
consideration will need to be given to the ethical deployment of this technology including rigorous quality assessment across different clinical settings and research into reliable uncertainty estimation methods [113-116] that would allow for deferral to human clinical experts when needed. These and other guardrails are needed to mitigate potential overreliance on LLM technologies, with other specific measures for attention to ethical and regulatory requirements particular to future use-cases and the presence of qualified physicians in the loop to safeguard any model outputs. Additional research will also be needed to assess the extent to which biases and security vulnerabilities might arise either from base models or the circumstances of use in deployment, as we have highlighted in our prior work [12]. Given the continuous evolution of clinical knowledge, it will also be important to develop ways for LLMs to utilize up-to-date clinical information [117].

## 7 Conclusion

The utility of medical AI systems could be greatly improved if they are better able to interact conversationally, anchoring on large-scale medical knowledge while communicating with appropriate levels of empathy and trust. This research demonstrates the significant potential capabilities of LLM based AI systems for settings involving clinical history-taking and diagnostic dialogue. The performance of AMIE in simulated consultations represents a milestone for the field, as it was assessed along an evaluation framework that considered multiple clinically-relevant axes for conversational diagnostic medical AI. However, the results should be interpreted with appropriate caution. Translating from this limited scope of experimental simulated history-taking and diagnostic dialogue, towards real-world tools for people and those who provide care for them, requires significant additional research and development to ensure the safety, reliability, fairness, efficacy, and privacy of the technology. If successful, we believe AI systems such as AMIE can be at the core of next generation learning health systems that help scale world class healthcare to everyone.

## Acknowledgments

This project was an extensive collaboration between many teams at Google Research and Google DeepMind. We thank Yun Liu, Daniel McDuff, Jake Sunshine, Ali Connell, Paul McGovern and Zoubin Ghahramani for their comprehensive review and detailed feedback on the manuscript. We also thank Sami Lachgar, Lauren Winer, John Guilyard and Maggie Shiels for contributions to the narratives and visuals. We are grateful to Julie Anne Seguin, Sally Goldman, Yuri Vasilevski, Xinying Song, Akshay Goel, Chu-ling Ko, Abhinav Das, Haiyang Yu, Chang Liu, Yuchen Liu, SiWai Man, Brett Hatfield, Sean Li, Ajay Joshi, Gordon Turner, Annisah Um'rani, Divya Pandya and Preeti Singh for their valuable insights, technical support and feedback during our research. We also thank our clinical provider partners in Canada and India for their partnership in conducting the OSCE study. Finally, we are grateful to Dale Webster, Ewa Dominowska, David Fleet, Philip Mansfield, Sushant Prakash, Renee Wong, Susan Thomas, Michael Howell, Karen DeSalvo, Jeff Dean, James Manyika, Zoubin Ghahramani and Demis Hassabis for their support during the course of this project.

## Data Availability

Some of the real-world datasets used in the development of AMIE are open-source (MedQA). The scenario packs from UK used in the OSCE study are also available for download on the internet.

## Code Availability

AMIE is an LLM based research AI system for diagnostic dialogue. We are not open-sourcing model code and weights due to the safety implications of unmonitored use of such a system in medical settings. In the interest of responsible innovation, we will be working with research partners, regulators, and providers to validate and explore safe onward uses of AMIE. For reproducibility, we have documented technical deep learning methods while keeping the paper accessible to a clinical and general scientific audience. Our work builds upon PaLM 2, for which technical details have been described extensively in the technical report [10].

## Competing Interests

This study was funded by Alphabet Inc and/or a subsidiary thereof ('Alphabet'). All authors are employees of Alphabet and may own stock as part of the standard compensation package.

## References

1. Engel, G. L. \& Morgan, W. L. Interviewing the patient (1973).
2. Peterson, M. C., Holbrook, J. H., Von Hales, D., Smith, N. \& Staker, L. Contributions of the history, physical examination, and laboratory investigation in making medical diagnoses. Western Journal of Medicine 156, 163 (1992).
3. Hampton, J. R., Harrison, M., Mitchell, J. R., Prichard, J. S. \& Seymour, C. Relative contributions of history-taking, physical examination, and laboratory investigation to diagnosis and management of medical outpatients. $\mathrm{Br} \mathrm{Med} \mathrm{J} \mathbf{2}$, 486-489 (1975).
4. Kassirer, J. P. Teaching clinical medicine by iterative hypothesis testing: let's preach what we practice 1983.
5. Roshan, M. \& Rao, A. A study on relative contributions of the history, physical examination and investigations in making medical diagnosis. The Journal of the Association of Physicians of India 48, 771-775 (2000).
6. Sandler, G. The importance of the history in the medical clinic and the cost of unnecessary tests. American heart journal 100, 928-931 (1980).
7. Silverman, J., Kurtz, S. \& Draper, J. Skills for communicating with patients (crc press, 2016).
8. Rennie, T., Marriott, J. \& Brock, T. P. Global supply of health professionals. N Engl J Med 370, 2246-7 (2014).
9. OpenAI. GPT-4 Technical Report 2023. arXiv: 2303.08774 [cs.CL].
10. Google. PaLM 2 Technical Report https://ai.google/static/documents/palm2techreport.pdf. 2023.
11. Deepmind, G. Gemini: A Family of Highly Capable Multimodal Models https://assets.bwbx.io/documents / users / iqjWHBFdfxIU/r7G7RrtT6rnM/v0. 2023.
12. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large Language Models Encode Clinical Knowledge. arXiv preprint arXiv:2212.13138 (2022).
13. Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Hou, L., Clark, K., Pfohl, S., Cole-Lewis, H., Neal, D., et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617 (2023).
14. Nori, H., Lee, Y. T., Zhang, S., Carignan, D., Edgar, R., Fusi, N., King, N., Larson, J., Li, Y., Liu, W., et al. Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine. arXiv preprint arXiv:2311.16452 (2023).
15. Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y., et al. LaMDA: Language models for dialog applications. arXiv preprint arXiv:2201.08239 (2022).
16. OpenAI. Introducing ChatGPT OpenAI. https://openai.com/blog/chatgpt.
17. Toma, A., Lawler, P. R., Ba, J., Krishnan, R. G., Rubin, B. B. \& Wang, B. Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding. arXiv preprint arXiv:2305.12031 (2023).
18. Chen, Z., Cano, A. H., Romanou, A., Bonnet, A., Matoba, K., Salvi, F., Pagliardini, M., Fan, S., Köpf, A., Mohtashami, A., et al. MEDITRON-70B: Scaling Medical Pretraining for Large Language Models. arXiv preprint arXiv:2311.16079 (2023).
19. Levine, D. History taking is a complex skill. BMJ 358 (2017).
20. King, A. \& Hoppe, R. B. "Best practice" for patient-centered communication: a narrative review. Journal of graduate medical education 5, 385-393 (2013).
21. Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H. \& Szolovits, P. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences 11, 6421 (2021).
22. Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L. \& Mark, R. G. MIMIC-III, a freely accessible critical care database. Scientific data 3, 1-9 (2016).
23. Chiu, C.-C., Tripathi, A., Chou, K., Co, C., Jaitly, N., Jaunzeikare, D., Kannan, A., Nguyen, P., Sak, H., Sankar, A., et al. Speech recognition for medical conversations. arXiv preprint arXiv:1711.07274 (2017).
24. Sharma, A., Miner, A. S., Atkins, D. C. \& Althoff, T. A computational approach to understanding empathy expressed in text-based mental health support. arXiv preprint arXiv:2009.08441 (2020).
25. Fu, Y., Peng, H., Khot, T. \& Lapata, M. Improving language model negotiation with self-play and in-context learning from ai feedback. arXiv preprint arXiv:2305.10142 (2023).
26. Abacha, A. B., Yim, W.-W., Adams, G., Snider, N. \& Yetisgen-Yildiz, M. Overview of the mediqa-chat 2023 shared tasks on the summarization $\dot{\&}$ generation of doctor-patient conversations in Proceedings of the 5th Clinical Natural Language Processing Workshop (2023), 503-513.
27. Ionescu, B., Müller, H., Drăgulinescu, A.-M., Yim, W.-W., Ben Abacha, A., Snider, N., Adams, G., Yetisgen, M., Rückert, J., G. Seco de Herrera, A., et al. Overview of the ImageCLLEF 2023: Multimedia Retrieval in Medical, Social Media and Internet Applications in International Conference of the Cross-Language Evaluation Forum for European Languages (2023), 370-396.
28. He, Z., Han, Y., Ouyang, Z., Gao, W., Chen, H., Xu, G. \& Wu, J. DialMed: A Dataset for Dialogue-based Medication Recommendation. arXiv preprint arXiv:2203.07094 (2022).
29. Naseem, U., Bandi, A., Raza, S., Rashid, J. \& Chakravarthi, B. R. Incorporating Medical Knowledge to Transformer-based Language Models for Medical Dialogue Generation in Proceedings of the 21st Workshop on Biomedical Language Processing (2022), 110-115.
30. Dacre, J., Besser, M. \& White, P. MRCP (UK) PART 2 Clinical Examination (PACES): a review of the first four examination sessions (June 2001-July 2002). Clinical Medicine 3, 452 (2003).
31. Sloan, D. A., Donnelly, M. B., Schwartz, R. W. \& Strodel, W. E. The Objective Structured Clinical Examination. The new gold standard for evaluating postgraduate clinical performance. Annals of surgery 222, 735 (1995).
32. Carraccio, C. \& Englander, R. The objective structured clinical examination: a step in the direction of competency-based evaluation. Archives of pediatrics \&S adolescent medicine 154, 736-741 (2000).
33. Epstein, R. M. \& Hundert, E. M. Defining and assessing professional competence. Jama 287, 226-235 (2002).
34. Horowitz, J. L. in Handbook of econometrics 3159-3228 (Elsevier, 2001).
35. Benjamini, Y. \& Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal statistical society: series B (Methodological) 57, 289-300 (1995).
36. Woolson, R. F. Wilcoxon signed-rank test. Wiley encyclopedia of clinical trials, 1-3 (2007).
37. Keifenheim, K. E., Teufel, M., Ip, J., Speiser, N., Leehr, E. J., Zipfel, S. \& Herrmann-Werner, A. Teaching history taking to medical students: a systematic review. BMC medical education 15, 1-12 (2015).
38. Yedidia, M. J., Gillespie, C. C., Kachur, E., Schwartz, M. D., Ockene, J., Chepaitis, A. E., Snyder, C. W., Lazare, A. \& Lipkin Jr, M. Effect of communications training on medical student performance. Jama 290, 1157-1165 (2003).
39. Makoul, G. Communication skills education in medical school and beyond. Jama 289, 93-93 (2003).
40. Tan, X. H., Foo, M. A., Lim, S. L. H., Lim, M. B. X. Y., Chin, A. M. C., Zhou, J., Chiam, M. \& Krishna, L. K. R. Teaching and assessing communication skills in the postgraduate medical setting: a systematic scoping review. BMC medical education 21, 1-19 (2021).
41. Raper, S. E., Gupta, M., Okusanya, O. \& Morris, J. B. Improving communication skills: a course for academic medical center surgery residents and faculty. Journal of Surgical education 72, e202-e211 (2015).
42. Von Fragstein, M., Silverman, J., Cushing, A., Quilligan, S., Salisbury, H., Wiskin, C. \& for Clinical Communication Skills Teaching in Undergraduate Medical Education, U. C. UK consensus statement on the content of communication curricula in undergraduate medical education. Medical education 42, 1100-1107 (2008).
43. De Haes, H. \& Bensing, J. Endpoints in medical communication research, proposing a framework of functions and outcomes. Patient education and counseling 74, 287-294 (2009).
44. Epstein, R. M. \& Street Jr, R. L. Patient-centered communication in cancer care: promoting healing and reducing suffering (2007).
45. Schirmer, J. M., Mauksch, L., Lang, F., Marvel, M. K., Zoppi, K., Epstein, R. M., Brock, D. \& Pryzbylski, M. Assessing communication competence: a review of current tools. Family Medicine 37, 184-92 (2005)
46. Nichol, J. R., Sundjaja, J. H. \& Nelson, G. Medical history. http://europepmc.org/books/NBK534249 (2018).
47. Denness, C. What are consultation models for? InnovAiT 6, 592-599 (2013).
48. Chan, S. C. C., Choa, G., Kelly, J., Maru, D. \& Rashid, M. A. Implementation of virtual OSCE in health professions education: A systematic review. Medical Education (2023).
49. Budzianowski, P., Wen, T.-H., Tseng, B.-H., Casanueva, I., Ultes, S., Ramadan, O. \& Gašić, M. Multiwoz-a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling. arXiv preprint arXiv:1810.00278 (2018).
50. Wei, W., Le, Q., Dai, A. \& Li, J. Airdialogue: An environment for goal-oriented dialogue research in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (2018), 3844-3854.
51. Lin, J., Tomlin, N., Andreas, J. \& Eisner, J. Decision-Oriented Dialogue for Human-AI Collaboration 2023. arXiv: 2305.20076 [cs.CL].
52. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. \& Polosukhin, I. Attention is all you need. Advances in neural information processing systems 30 (2017).
53. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155 (2022).
54. Zhao, J., Khashabi, D., Khot, T., Sabharwal, A. \& Chang, K.-W. Ethical-advice taker: Do language models understand natural language interventions? arXiv preprint arXiv:2106.01465 (2021).
55. Saunders, W., Yeh, C., Wu, J., Bills, S., Ouyang, L., Ward, J. \& Leike, J. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802 (2022).
56. Scheurer, J., Campos, J. A., Korbak, T., Chan, J. S., Chen, A., Cho, K. \& Perez, E. Training language models with language feedback at scale. arXiv preprint arXiv:2303.16755 (2023).
57. Glaese, A., McAleese, N., Trębacz, M., Aslanides, J., Firoiu, V., Ewalds, T., Rauh, M., Weidinger, L., Chadwick, M., Thacker, P., et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375 $(2022)$.
58. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073 (2022).
59. Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861 (2021).
60. Shor, J., Bi, R. A., Venugopalan, S., Ibara, S., Goldenberg, R. \& Rivlen, E. Clinical BERTScore: An Improved Measure of Automatic Speech Recognition Performance in Clinical Settings. arXiv preprint arXiv:2303.05737 (2023).
61. Abacha, A. B., Agichtein, E., Pinter, Y. \& Demner-Fushman, D. Overview of the medical question answering task at TREC 2017 LiveQA. in TREC (2017), 1-12.
62. Wallace, W., Chan, C., Chidambaram, S., Hanna, L., Iqbal, F. M., Acharya, A., Normahani, P., Ashrafian, H., Markar, S. R., Sounderajah, V., et al. The diagnostic and triage accuracy of digital and online symptom checker tools: a systematic review. NPJ Digital Medicine 5, 118 (2022).
63. Zeltzer, D., Herzog, L., Pickman, Y., Steuerman, Y., Ber, R. I., Kugler, Z., Shaul, R. \& Ebbert, J. O. Diagnostic accuracy of artificial intelligence in virtual primary care. Mayo Clinic Proceedings: Digital Health 1, 480-489 (2023).
64. Johri, S., Jeong, J., Tran, B. A., Schlessinger, D. I., Wongvibulsin, S., Cai, Z. R., Daneshjou, R. \& Rajpurkar, P. Testing the Limits of Language Models: A Conversational Framework for Medical AI Assessment. medRxiv, 2023-09 (2023)
65. Zeng, G., Yang, W., Ju, Z., Yang, Y., Wang, S., Zhang, R., Zhou, M., Zeng, J., Dong, X., Zhang, R., et al. MedDialog: Large-scale medical dialogue datasets in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2020), 9241-9250.
66. Liu, W., Tang, J., Cheng, Y., Li, W., Zheng, Y. \& Liang, X. MedDG: an entity-centric medical consultation dataset for entity-aware medical dialogue generation in CCF International Conference on Natural Language Processing and Chinese Computing (2022), 447-459.
67. Varshney, D., Zafar, A., Behra, N. K. \& Ekbal, A. Cdialog: A multi-turn COVID-19 conversation dataset for entity-aware dialog generation. arXiv preprint arXiv:2212.06049 (2022).
68. Yan, G., Pei, J., Ren, P., Ren, Z., Xin, X., Liang, H., de Rijke, M. \& Chen, Z. ReMeDi: Resources for Multi-domain, Multi-service, Medical Dialogues in Proceedings of the 45 th International ACM SIGIR Conference on Research and Development in Information Retrieval (2022), 3013-3024.
69. Kelly, C. J., Karthikesalingam, A., Suleyman, M., Corrado, G. \& King, D. Key challenges for delivering clinical impact with artificial intelligence. BMC medicine 17, 1-9 (2019).
70. McDuff, D., Schaekermann, M., Tu, T., Palepu, A., Wang, A., Garrison, J., Singhal, K., Sharma, Y., Azizi, S., Kulkarni, K., et al. Towards Accurate Differential Diagnosis with Large Language Models. arXiv preprint arXiv:2312.00164 (2023)
71. Kanjee, Z., Crowe, B. \& Rodman, A. Accuracy of a Generative Artificial Intelligence Model in a Complex Diagnostic Challenge. JAMA (2023).
72. Semigran, H. L., Linder, J. A., Gidengil, C. \& Mehrotra, A. Evaluation of symptom checkers for self diagnosis and triage: audit study. BMJ 351 (2015).
73. Ayers, J. W., Poliak, A., Dredze, M., Leas, E. C., Zhu, Z., Kelley, J. B., Faix, D. J., Goodman, A. M., Longhurst, C. A., Hogarth, M., et al. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum. JAMA Internal Medicine (2023).
74. OpenAI. ChatGPT OpenAI. https://chat.openai.com/chat.
75. Carrillo de Albornoz, S., Sia, K.-L. \& Harris, A. The effectiveness of teleconsultations in primary care: systematic review. Family Practice 39, 168-182 (2022).
76. Wharton, G. A., Sood, H. S., Sissons, A. \& Mossialos, E. Virtual primary care: fragmentation or integration? The Lancet Digital Health 1, e330-e331 (2019).
77. Fuster-Casanovas, A. \& Vidal-Alaball, J. Asynchronous Remote Communication as a Tool for Care Management in Primary Care: A Rapid Review of the Literature. International Journal of Integrated Care 22 (2022).
78. Hammersley, V., Donaghy, E., Parker, R., McNeilly, H., Atherton, H., Bikker, A., Campbell, J. \& McKinstry, B. Comparing the content and quality of video, telephone, and face-to-face consultations: a non-randomised, quasi-experimental, exploratory study in UK primary care. British Journal of General Practice 69, e595-e604 (2019).
79. Gross, D. A., Zyzanski, S. J., Borawski, E. A., Cebul, R. D. \& Stange, K. C. Patient satisfaction with time spent with their physician. Journal of Family Practice 47, 133-138 (1998).
80. Tates, K., Antheunis, M. L., Kanters, S., Nieboer, T. E. \& Gerritse, M. B. The effect of screen-to-screen versus face-to-face consultation on doctor-patient communication: an experimental study with simulated patients. Journal of medical Internet research 19, e421 (2017)
81. Zyzanski, S. J., Stange, K. C., Langa, D. M. \& Flocke, S. A. Trade-offs in high-volume primary care practice. Journal of Family Practice 46, 397-402 (1998).
82. Dvijotham, K., Winkens, J., Barsbey, M., Ghaisas, S., Stanforth, R., Pawlowski, N., Strachan, P., Ahmed, Z., Azizi, S., Bachrach, Y., et al. Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians. Nature Medicine 29, 1814-1820 (2023)
83. Bird, J. \& Cohen-Cole, S. A. in Methods in teaching consultation-liaison psychiatry 65-88 (Karger Publishers, 1990).
84. Rezler, A. G., Woolliscroft, J. A. \& Kalishman, S. G. What is missing from patient histories? Medical Teacher 13, 245-252 (1991).
85. Rosenberg, E. E. Lessons for Clinicians From Physician-Patient. Arch Fam Med 6, 279-283 (1997).
86. Smith, R. C. Patient-centered interviewing: an evidence-based method (Lippincott Williams \& Wilkins, 2002).
87. Berwick, D. M., Nolan, T. W. \& Whittington, J. The triple aim: care, health, and cost. Health affairs 27, 759-769 (2008).
88. Bodenheimer, T. \& Sinsky, C. From triple to quadruple aim: care of the patient requires care of the provider. The Annals of Family Medicine 12, 573-576 (2014).
89. Adamson, T. E., Tschann, J. M., Gullion, D. \& Oppenberg, A. Physician communication skills and malpractice claims. A complex relationship. Western Journal of Medicine 150, 356 (1989).
90. Silverman, J. \& Kinnersley, P. Doctors' non-verbal behaviour in consultations: look at the patient before you look at the computer 2010.
91. Rahman, U. \& Cooling, N. Inter-Cultural Communication Skills Training in Medical Schools: A Systematic Review Medical Research Archives 11 (2023).
92. Kantar, A., Marchant, J. M., Song, W.-J., Shields, M. D., Chatziparasidis, G., Zacharasiewicz, A., Moeller, A. \& Chang, A. B. History taking as a diagnostic tool in children with chronic cough. Frontiers in pediatrics 10, 850912 (2022).
93. Setyonugroho, W., Kennedy, K. M. \& Kropmans, T. J. Reliability and validity of OSCE checklists used to assess the communication skills of undergraduate medical students: a systematic review. Patient education and counseling 98, $1482-1491$ (2015)
94. Weidinger, L., Uesato, J., Rauh, M., Griffin, C., Huang, P.-S., Mellor, J., Glaese, A., Cheng, M., Balle, B., Kasirzadeh, A., et al. Taxonomy of risks posed by language models in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (2022), 214-229.
95. Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R. \& Ahmed, N. K. Bias and Fairness in Large Language Models: A Survey 2023. arXiv: 2309.00770 [cs.CL].
96. Johnson, R. L., Roter, D., Powe, N. R. \& Cooper, L. A. Patient race/ethnicity and quality of patient-physician communication during medical visits. American journal of public health 94, 2084-2090 (2004).
97. Roter, D. L., Hall, J. A. \& Aoki, Y. Physician gender effects in medical communication: a meta-analytic review. Jama 288, 756-764 (2002).
98. Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N. \& Irving, G. Red teaming language models with language models. arXiv preprint arXiv:2202.03286 (2022).
99. Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858(2022).
100. Yu, J., Lin, X. \& Xing, X. Gptfuzzer: Red teaming large language models with auto-generated jailbreak prompts. arXiv preprint arXiv:2309.10253 (2023).
101. Ge, S., Zhou, C., Hou, R., Khabsa, M., Wang, Y.-C., Wang, Q., Han, J. \& Mao, Y. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming. arXiv preprint arXiv:2311.07689 (2023).
102. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D. \& Gebru, T. Model cards for model reporting in Proceedings of the conference on fairness, accountability, and transparency (2019), 220-229.
103. Crisan, A., Drouhard, M., Vig, J. \& Rajani, N. Interactive model cards: A human-centered approach to model documentation in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (2022), 427-439.
104. Pushkarna, M., Zaldivar, A. \& Kjartansson, O. Data cards: Purposeful and transparent dataset documentation for responsible ai in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (2022), 17761826 .
105. Choudhury, M. \& Deshpande, A. How Linguistically Fair Are Multilingual Pre-Trained Language Models? in Proceedings of the AAAI conference on artificial intelligence 35 (2021), 12710-12718.
106. Talat, Z., Névéol, A., Biderman, S., Clinciu, M., Dey, M., Longpre, S., Luccioni, S., Masoud, M., Mitchell, M., Radev, D., et al. You reap what you sow: On the challenges of bias evaluation under multilingual settings in Proceedings of BigScience Episode\# 5-Workshop on Challenges \&S Perspectives in Creating Large Language Models (2022), 26-41.
107. Ahuja, S., Aggarwal, D., Gumma, V., Watts, I., Sathe, A., Ochieng, M., Hada, R., Jain, P., Axmed, M., Bali, K. \& Sitaram, S. MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks 2023. arXiv: 2311.07463 [cs.CL].
108. ImaniGooghari, A., Lin, P., Kargaran, A. H., Severini, S., Jalili Sabet, M., Kassner, N., Ma, C., Schmid, H., Martins, A., Yvon, F. \& Schütze, H. Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (Association for Computational Linguistics, 2023). http://dx.doi.org/10.18653/v1/2023.acl-long.61.
109. Nguyen, X.-P., Aljunied, S. M., Joty, S. \& Bing, L. Democratizing LLMs for Low-Resource Languages by Leveraging their English Dominant Abilities with Linguistically-Diverse Prompts 2023. arXiv: 2306.11372 [cs.CL].
110. Naous, T., Ryan, M. J., Ritter, A. \& Xu, W. Having Beer after Prayer? Measuring Cultural Bias in Large Language Models 2023. arXiv: 2305.14456 [cs.CL].
111. Ramesh, K., Sitaram, S. \& Choudhury, M. Fairness in Language Models Beyond English: Gaps and Challenges 2023. arXiv: 2302.12578 [cs.CL].
112. Hada, R., Gumma, V., de Wynter, A., Diddee, H., Ahmed, M., Choudhury, M., Bali, K. \& Sitaram, S. Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation? 2023. arXiv: 2309.07462 [cs.CL].
113. Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H., Jaakkola, T. S. \& Barzilay, R. Conformal Language Modeling 2023. arXiv: 2306.10193 [cs.CL].
114. Chen, J. \& Mueller, J. Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness 2023. arXiv: 2308.16175 [cs.CL].
115. Huang, Y., Song, J., Wang, Z., Zhao, S., Chen, H., Juefei-Xu, F. \& Ma, L. Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models 2023. arXiv: 2307.10236 [cs. SE].
116. Yang, Q., Ravikumar, S., Schmitt-Ulms, F., Lolla, S., Demir, E., Elistratov, I., Lavaee, A., Lolla, S., Ahmadi, E., Rus, D., Amini, A. \& Perez, A. UUncertainty-aware Language Modeling for Selective Question Answering 2023. arXiv: 2311.15451 [cs.CL].
117. Lazaridou, A., Kuncoro, A., Gribovskaya, E., Agrawal, D., Liska, A., Terzi, T., Gimenez, M., de Masson d'Autume, C., Kocisky, T., Ruder, S., et al. Mind the gap: Assessing temporal generalization in neural language models. Advances in Neural Information Processing Systems 34, 29348-29363 (2021).
</end of paper 0>


<paper 1>
# Conversational Disease Diagnosis via External Planner-Controlled Large Language Models 

Zhoujian Sun ${ }^{1}$, Cheng Luo ${ }^{1}$, Ziyi Liu ${ }^{2}$, Zhengxing Huang ${ }^{3}$<br>${ }^{1}$ Zhejiang Lab, ${ }^{2}$ Transtek Medical Electronic, ${ }^{3}$ Zhejiang University<br>sunzj@zhejianglab.com, zhengxinghuang@zju.edu.cn


#### Abstract

The development of large language models (LLMs) has brought unprecedented possibilities for artificial intelligence (AI) based medical diagnosis. However, the application perspective of LLMs in real diagnostic scenarios is still unclear because they are not adept at collecting patient data proactively. This study presents a LLM-based diagnostic system that enhances planning capabilities by emulating doctors. Our system involves two external planners to handle planning tasks. The first planner employs a reinforcement learning approach to formulate disease screening questions and conduct initial diagnoses. The second planner uses LLMs to parse medical guidelines and conduct differential diagnoses. By utilizing real patient electronic medical record data, we constructed simulated dialogues between virtual patients and doctors and evaluated the diagnostic abilities of our system. We demonstrated that our system obtained impressive performance in both disease screening and differential diagnoses tasks. This research represents a step towards more seamlessly integrating AI into clinical settings, potentially enhancing the accuracy and accessibility of medical diagnostics.


## 1 Introduction

Enabling artificial intelligence (AI) to diagnose disease has been a long-awaited goal since the concept of medical AI emerged [1]. The development of large language models (LLMs) brings unprecedented opportunities in AI-based diagnosis. Notably, Med-Palm 2 and GPT-4 Turbo have attained high scores on the United States Medical Licensing Examination [2, 3]. Recent research also illustrates that LLMs may perform as well as human doctors in many disease diagnostic tasks [4, 5, 6, 7].

Nonetheless, most LLMs are not adept at collecting patient data, which limits their application perspective. Almost all existing LLM-based studies formulate diagnosis as a question-answer task where LLMs are endowed with all necessary information to answer the diagnostic question [4, 5, 6, 8, 9]. In real diagnosis scenarios, doctors initially have no knowledge about the patient's condition, and patients also cannot comprehensively describe their own conditions. Distinguishing which information is useful and knowing when to collect the information are core skills of a doctor [10]. If a LLM requires a doctor to collect all important information in advance to make a diagnosis, its practical value is quite doubtful, because the doctor usually already knows what disease the patient has when all information is collected. LLM-based diagnostic systems should be capable of collecting information from scratch and then proceeding to diagnosis. This demands that LLM-based diagnostic systems possess excellent planning abilities to proactively ask dozens of appropriate questions through interactions with patients. Most current LLMs lack such planning capabilities. For example, a recent study demonstrated that GPT-4 could achieve high diagnostic accuracy, ranging from $70 \%$ to $90 \%$, when provided with complete patient information for diagnosing skin diseases. However, its accuracy can drop to $30 \%$ to $60 \%$ when it must diagnose starting from scratch [11].

![](https://cdn.mathpix.com/cropped/2024_06_04_538708f7f6d3225d9589g-02.jpg?height=385&width=1097&top_left_y=236&top_left_x=514)

Figure 1: System Overview

In this study, we aim to develop a LLM based diagnostic system that enhances planning capabilities by emulating doctors. Previous research suggests that medical consultations can roughly be divided into two distinct phases [12]. In the first phase, which we call disease screening phase, doctors ask patients a series of questions mainly about their medical history and infer possible diseases based on the responses. This stage relies heavily on doctor's experience. In the second phase, which we call differential diagnosis phase, doctors ask questions to confirm or exclude the diseases suspected in the first phase. The questions asked during the differential diagnosis phase include the patient's laboratory test and medical examination results. This phase relies on objective medical knowledge. Due to the substantial differences between these two phases, we clearly need to develop two different planners when emulating doctors. The first should be data-driven, learning from extensive data on how to conduct consultations, while the second should be knowledge-driven, adhering strictly to medical knowledge and being interpretable.

We primarily face two challenges in implementing the two planners. (1) Real medical consultation dialogue datasets are scarce, which hampers the training of the first planner in a supervised manner. (2) Developing a decision procedure that adheres to medical literature typically requires expert involvement, making the second planner expensive and hard to maintain [13]. In this study, we adopted a reinforcement learning ( $\mathrm{RL}$ ) approach to facilitate the autonomous training of the first planner without the need for expert demonstrations. We used LLMs to analyze patient admission records, identifying each symptom's presence or absence. Subsequently, we utilized a RL method to train the inquiry policy based on the structurized patient symptoms. Following this, we employed a neural network to predict high-risk diseases based on the outcomes of the inquiries. The decision procedure for diagnosing or ruling out diseases is implicitly recorded in medical literature in the form of natural text. Since leading LLMs have achieved capabilities nearly equivalent to junior doctors in natural language processing and medical knowledge, we attempt to summarize decision procedures by directly employing LLMs [14, 3]. Additionally, we have designed a method that allows non-medical experts to refine these decision procedures, thereby reducing reliance on experts.

We evaluated this study through retrospective simulated dialogues. We implemented a simulated doctor comprised of two planners and one LLM (Figure 1). Planners are responsible for determining actions for each round, while the LLM handles the conversion of these actions into natural language and also parses responses into a format readable by the planners. We utilized another LLM to read real electronic medical record (EMR) data, simulating a patient who would respond to any questions posed by the simulated doctor. The simulated doctor was designed to ask a series of questions to diagnose the patient's illness. We conducted tests using the MIMIC-IV dataset [15]. The results show that our planners, controlling the LLM, achieved impressive performance in both phases. We contend that the proposed diagnostic system has the potential to enhance diagnostic precision and accessibility. All source code and data are public available $\square^{1}$

## 2 Related Work

Recent research has illustrated the capabilities of LLMs in aiding disease diagnosis [16, 5, 4, 8]. They use pre-collected patient information to conduct diagnosis. However, a LLM-based diagnostic system should prioritize and emphasize the capability to ask the right questions for information collection in multi-turn dialogue [10]. Existing research seldom explores the multi-turn diagnostic capabilities of[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_538708f7f6d3225d9589g-03.jpg?height=908&width=1317&top_left_y=234&top_left_x=404)

Figure 2: Diagnosis Screening Planner Optimizing

LLMs, and those that do usually only investigated LLMs' planning abilities in fewer than five rounds of free dialogue, which is far from sufficient to complete diagnostic tasks [17]. To our knowledge, the AMIE model is the only LLM trained to improve the medical information-gathering ability [7]. This model has two limitations. Firstly, it relies solely on patient self-reported symptoms for diagnosis, without incorporating data from medical laboratory tests or imaging reports. Since the symptoms of many diseases overlap, relying solely on symptoms for an accurate diagnosis is not only dangerous but also impractical. Secondly, the AMIE model is trained via a LLM-generated synthetic dialogue dataset. Without concrete evidence demonstrating that LLMs can match the efficacy of experts in medical consultations, the soundness of this approach is questionable.

Before the advent of LLMs, several diagnostic-oriented dialogue systems had been proposed 18,19 , 20, 21]. However, their datasets are originated from online consultations, and the data quality was often questioned. In the past two years, research in various fields has explored ways to make LLMs proactively guide conversations and complete tasks. Most studies use prompt engineering, while some enhance LLMs' planning ability with external planners [22, 23, 24]. However, these methods cannot be utilized in medicine because they still rely on public, high-quality, human annotated dialogue datasets, which in fact do not exist in the medical field. In this study, we will develop a dialogue system without using high-quality medical dialogue data.

LLM-based autonomous agent tackles complex tasks by decomposing them into simpler sub-tasks, addressing each one sequentially. Most research in this area utilizes prompt engineering tricks, e.g., chain of thought, to activate LLM's planning capabilities [25]. However, an autonomous agent typically does not require interaction with the user during the task completion process [26]. In contrast, our system needs to dynamically generate a plan based on the patient's responses.

## 3 Method

### 3.1 Disease Screening Planner

Doctors are required to obtain important information from patients in medical consultations to formulate initial diagnoses, which includes their chief complaints (CC), history of present illness (HPI), past medical history, family history, social history, etc [27]. HPI and CC are usually regarded as
the most critical information, as they form the basis for at most $80 \%$ of patient diagnoses [28]. Despite its importance, collecting HPI and CC is challenging because patients usually cannot comprehensively report their symptoms [10]. As patients may exhibit a subset of hundreds of possible symptoms, it is also impractical to exhaustively query every potential symptom. Doctors need to identify positive symptoms via their own experience with only several questions, thereby facilitating the formulation of a comparatively accurate diagnosis.

We employed an RL approach to train a policy model to ask questions and a supervised learning based screening model to conduct initial diagnosis (Figure 2). We use $h$ to denote the past medical history, family history, and social history of a patient. As $h$ can typically be acquired through straightforward questioning, our study will not focus on collecting such information [27]. We presume $h$ is textual and already known and its embedding is $e_{h}$. We use a planner to ask $N$ questions about patient symptoms in the RL paradigm.

State Space: The state $s_{t}=\left[e_{h}, p_{t}\right]$ is the concatenation of two elements. The first part is $e_{h}$, which is invariable in an episode. $p_{t} \in\{0,1\}^{3 M}$ represents structured HPI and CC, where $M$ denotes the number of symptoms. Each symptom information is presented by a binary triplet, while $[0,0,1],[0,1,0],[1,0,0]$ means the symptom is unknown, confirmed, or denied, respectively. At the beginning, all symptoms are unknown, and their status is updated with every interaction.

Action Space: The action space contains $M(N \ll M)$ actions, where each question asks whether a related symptom is present in the next turn. We presume that there is a two-layer structure within the action space. The first layer refers to general symptoms (such as chest pain), while the secondary layer denotes more specific symptoms (such as chest pain accompanied by fever). Each second-layer symptom is affiliated with a first-layer symptom. We stipulate that the model can only inquire about second-layer symptoms after the patient acknowledges the affiliated first-layer symptoms. Meanwhile, we do not allow the planner to ask the same question twice.

Reward: We set the reward $R_{t}$ to one if the asked symptom exists, and to zero if the asked symptom is denied or not mentioned.

Patient Agent: We use admission records from patient EMRs to construct the patient agent. We first separated an admission record into $h$, HPI, and CC. Then, we structured the textual CC and HPI. Specifically, we utilized a LLM to determine whether patients had a symptom, and ultimately transformed the CC and HPI into a $M$-dimensional binary vector $p^{\text {oracle }}$. Each symptom is represented by one if it is confirmed, or zero if it is denied or not mentioned. A $h$ and a $p^{\text {oracle }}$ formulate a sample. When an agent receives a query from the planner, it can response the answer directly.

Policy Model Learning: We used an actor-critic model to generate the policy $\pi_{t} \in \mathbf{R}^{M}$, which is a stochastic vector, and the value $Q_{t} \in \mathbf{R}$ [29]. Each element in $\pi_{t}$ corresponds to an query action. The value of an element indicates the probability of the policy model selecting the corresponding action. We utilize a multi-layer perceptron (MLP) to learn a representation $r_{t}$ from $s_{t}$ and then use two MLPs to generate $\pi_{t}$ and $Q_{t}$ according to $r_{t}$, respectively. We adopted the proximal policy optimization (PPO) algorithm to train policy [30, 31]. In this study, we preset a maximum number of inquiry rounds, and the PPO will train the RL agent to obtain the maximum reward within these rounds. To improve effectiveness, we assume that the patient agent will proactively disclose one first layer symptom to the RL agent before the first question. Given that patients typically start by describing their areas of discomfort in real medical consultations, we believe this design is justified.

Screening Model Learning: After the policy model is optimized, we will use the final state of episodes to predict the initial diagnosis. As the patient discharge diagnosis is recorded in the EMR, we use a supervised learning method, i.e., MLP, to train the screening classifier.

### 3.2 Differential Diagnosis Planner

The differential diagnosis planner consists of a set of decision procedures, each corresponding to the diagnostic process for a specific disease. We use a LLM to transform medical literature and generate a preliminary structured decision procedure (Figure 3). The planner will parse the procedure and conduct inquiries according to the procedure. The effectiveness of the procedure is then tested in simulated diagnosis dialogues where one LLM with the external planner acts as a doctor and another as a patient, using complete discharge EMR for the patient role. In the simulated interaction, the doctor asks questions based on the decision procedure, and the patient responds based on the EMR

![](https://cdn.mathpix.com/cropped/2024_06_04_538708f7f6d3225d9589g-05.jpg?height=792&width=1298&top_left_y=252&top_left_x=403)

Figure 3: Differential Diagnosis Planner Optimizing

contents. If the EMR does not contain the answer to a question, the patient simulator will indicate that the inquired physical condition is normal. Finally, the doctor simulator concludes with either a confirmed or excluded diagnosis. This outcome is then compared with the actual patient discharge diagnosis to identify any inaccuracies, creating a set of dialogues with incorrect conclusions. These results are analyzed to pinpoint which steps in the procedure led to incorrect diagnoses. According to the analyze result, the procedure undergoes refinement through revisions to its content. The refined decision procedure is then retested and improved iteratively. In this study, the refinement process was completed by a data scientist who does not hold a medical license.

### 3.3 Simulated Dialogues

We conducted retrospective simulated conversations between a doctor simulator and a patient simulator to evaluate the diagnostic performance of our system.

Patient Simulator: We used a LLM to act as a patient. During the disease screening phase, we submitted a patient admission record to the LLM and instructed the LLM to answer questions based on the provided information (prompt is in appendix E.2). This simulation method is widely used in previous studies [7,11]. During the differential diagnosis phase, we submitted the entire EMR data, including laboratory tests and exam reports, as context to respond to the doctor simulator's inquiries.

Doctor Simulator: We employed a LLM controlled by two planners to serve as a doctor in the disease screening phase and the differential diagnosis phase. The planners are responsible for generating questions or conducting diagnoses. The LLM is tasked with translating the questions generated by the two planners into plain language. It also interprets the patient's utterances and categorizes them into a format that the planners can process.

## 4 Experiment

### 4.1 Dataset and Preprocessing

We utilized textual clinical notes from the MIMIC-IV dataset to conduct experiments [15]. The MIMIC-IV contains EMRs for approximately 400,000 admissions to the intensive care unit (ICU) at Beth Israel Deaconess Medical Center in the USA between 2008 and 2019 (a sample is in appendix F. We identified the 98 most common diseases from MIMIC-IV (happened larger than 200 times), excluding injuries from accidents and mental illnesses (disease list is in appendix G). We randomly
selected 40,000 admissions with textual clinical notes from patients whose primary diagnosis (the first diagnosis in the discharge diagnoses) was among these 98 diseases. The reserved clinical notes contain 2,394 words in average, including admission records, discharge summaries, laboratory test results, imaging examinations reports, etc.

The symptom checker developed by the Mayo Clinic was utilized to structure HPI and CC [32]. This symptom checker categorizes common symptoms of patients into 28 first-layer categories and 689 second-layer symptoms. Each second layer symptom is associated with one of the first-layer categories. We used GPT-4 Turbo (1106 preview) to analyze the presence of these 717 symptoms (28+ 689) in each patient's HPI and CC information, thereby converting the textual admission EMR into a 717-dimensional binary vector (prompt in appendix E.1). We transformed the selected 40,000 clinical notes into binary vectors. After the transformation, we randomly selected 400 medical records for evaluation. The results show that GPT-4 Turbo's parsing performance is satisfactory, with both macro recall and macro precision are around $97 \%$, making it is suitable for constructing patient simulators for the development of the disease screening planner.

### 4.2 Experimental Settings

Foundation Models: We used Microsoft Azure GPT-4 Turbo (04-09), GPT-4o (05-13), and Llama3 (70B-instruct) as foundation models of doctor simulators [33, 14, 34]. We used GPT-4 Turbo (04-09) as the foundation model of patient simulators. GPT-4 Turbo and GPT-4o were chosen because they are two of the best LLMs. Llama3 was selected because it is a leading open-source LLM. Of note, transferring medical data outside hospitals often breaches data protection regulations, which means all closed LLMs (e.g., GPT-4o, Gemini-1.5 Pro , Claude-3 Opus) are actually unavailable in practice as they only operate through remote API [35]. We need to ensure our system retains good diagnostic capabilities only with an open-source LLM which can be locally deployed.

Disease Screening Settings: We divided the structured data of 40,000 patients' HPI and CC into three parts for training, validation, and testing, with proportions of $0.7,0.1$, and 0.2 , respectively. All past medical history, social history, and family history information were directly provided, and we used a text embedding model to generate their embeddings [36]. We configured the system to have the patient initially report one positive first-layer symptom to the planner, who can then ask additional 9 or 19 questions. The planner also trains a high-risk disease screening model based on the collected patient information and their primary discharge diagnosis. We used the planner to control a LLM to conduct 300 simulated conversations and evaluate performances.

Differential Diagnosis Settings: We used heart failure as a case to test the planner's ability to make differential diagnoses. Heart failure is the most common disease in the MIMIC-IV dataset and one of the most complex common cardiovascular diseases [37]. We argue if our system can accurately diagnose heart failure, it may also be capable of diagnosing most other diseases. In this study, we used the heart failure diagnostic guideline published by the European society of cardiology as the source of diagnostic knowledge [38]. We randomly selected 160 positive patients (i.e., primary diagnosis is heart failure) and 160 negative patients (i.e., heart failure is not in the discharge diagnosis) from the MIMIC-IV dataset. We randomly selected 120 records (with a $1: 1$ ratio of negative to positive) as the training set for the generation of the decision procedure and 200 records for testing the decision procedure. We used the GPT-4 Turbo ( 0125 preview) and the guideline to generate a preliminary decision procedure (prompt in appendix E.5, result in appendix D.1). Subsequently, we selected 40 samples from the training set for each round of simulated dialogue experiment. We summarized the incorrect diagnoses, had a non-expert review the causes of the errors, and modified the procedure based on the analysis results. This modification process was repeated three times. Finally, we conducted simulated diagnoses on the test data via the refined decision procedure (appendix D.2).

Metrics: We evaluated performances through simulated dialogues between patients and doctors. During the disease screening phase, we assess the system using the Top-N hit rate of the real primary diagnosis in the ranking. For example, a Top 3 Hit rate of 0.50 means that in $50 \%$ of test cases, the true primary discharge diagnosis appears within the top three predictions generated by the disease screening planner. In the differential diagnosis phase, the system can ask up to 20 questions. If the system provides a diagnosis within these 20 questions, the conversation is considered successful. Otherwise, it is deemed a failure, and the corresponding case is treated as a negative sample. We evaluated the differential diagnosis performance via accuracy, precision, recall, and F1 score.

Table 1: Disease Screening Performance

| EP | \# Question | LLM | \# Sample | Top 1 | Top 3 | Top 5 | Top 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No | 10 | GPT-4 Turbo 04-09 | 300 | 0.273 | 0.510 | 0.597 | 0.717 |
| Yes | 10 | GPT-4 Turbo 04-09 | 300 | $\mathbf{0 . 3 3 0}$ | $\mathbf{0 . 5 5 0}$ | $\mathbf{0 . 6 3 7}$ | $\mathbf{0 . 7 7 0}$ |
| No | 10 | Llama3-70B-Instruct | 300 | 0.240 | 0.423 | 0.483 | 0.583 |
| Yes | 10 | Llama3-70B-Instruct | 300 | 0.303 | 0.477 | 0.603 | 0.737 |
| No | 20 | GPT-4 Turbo 04-09 | 300 | 0.310 | 0.523 | 0.590 | 0.727 |
| Yes | 20 | GPT-4 Turbo 04-09 | 300 | 0.310 | 0.527 | 0.610 | 0.753 |
| No | 20 | Llama3-70B-Instruct | 300 | 0.200 | 0.387 | 0.470 | 0.607 |
| Yes | 20 | Llama3-70B-Instruct | 300 | 0.317 | 0.493 | 0.603 | 0.747 |
| No | 10 | GPT-4 Turbo 04-09 | 1000 | 0.279 | 0.507 | 0.597 | 0.714 |
| Yes | 10 | GPT-4 Turbo 04-09 | 1000 | 0.325 | 0.544 | 0.633 | 0.772 |

EP (external planner) set to "yes" means the LLM is controlled by an external planner and "no" means it operates independently, whose prompts are in appendix E. 3 The number of questions is ten (20) because the external planner starts with one symptom and proactively asks nine (19) additional questions.

Table 2: Differential Diagnosis Performance

| EK | LLM | Success Rate | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| None | GPT-4 Turbo 04-09 | $86 \%$ | $86 \%$ | $80 \%$ | $95 \%$ | $87 \%$ |
| Text | GPT-4 Turbo 04-09 | $98 \%$ | $73 \%$ | $65 \%$ | $\mathbf{1 0 0 \%}$ | $79 \%$ |
| EP | GPT-4 Turbo 04-09 | $\mathbf{1 0 0 \%}$ | $82 \%$ | $86 \%$ | $76 \%$ | $80 \%$ |
| EP+HF | GPT-4 Turbo 04-09 | $\mathbf{1 0 0 \%}$ | $\mathbf{9 1 \%}$ | $\mathbf{8 9 \%}$ | $\mathbf{9 2 \%}$ | $\mathbf{9 1 \%}$ |
| None | Llama3-70B-Instruct | $0 \%$ | $50 \%$ | $0 \%$ | $0 \%$ | NA |
| Text | Llama3-70B-Instruct | $15 \%$ | $63 \%$ | $\mathbf{9 6 \%}$ | $26 \%$ | $41 \%$ |
| EP | Llama3-70B-Instruct | $\mathbf{1 0 0 \%}$ | $85 \%$ | $\mathbf{9 6} \%$ | $73 \%$ | $83 \%$ |
| EP+HF | Llama3-70B-Instruct | $\mathbf{1 0 0 \%}$ | $\mathbf{9 1} \%$ | $91 \%$ | $\mathbf{9 0 \%}$ | $\mathbf{9 0 \%}$ |

EK (external knowledge) column indicates the method of utilizing knowledge. "None" means no external knowledge is used; "text" uses diagnosis related text from the clinical guideline; "EP" employs a diagnostic procedure directly generated by the LLM into the external planner; "EP+HF" involves a human-corrected diagnostic procedure. Prompts of these experiments are recorded in appendix E. 4 .

### 4.3 Disease Screening Performance

Table 1 provides a comparison of disease screening performance. We found that system's performance is similar when randomly selecting either 1000 or 300 cases. Thus, we argue that using 300 simulated dialogues is sufficient to evaluate our system's performance effectively. Incorporating an external planner (EP) significantly improves the system's disease screening capability. For example, without the EP, GPT-4 Turbo achieves a Top 1 Hit rate of 0.273 , which increases to 0.330 under the guidance of the EP. In the absence of the EP, Llama3 significantly underperforms compared to GPT-4 Turbo, but the performance of Llama3 with the EP surpasses GPT-4 Turbo's without the EP. Meanwhile, we found simply asking more questions does not necessarily lead to better performance. For instance, when asking 20 questions, Llama3's performance significantly decreases, whereas GPT-4 Turbo only sees a significant improvement in its Top 1 Hit rate, with other metrics barely changing. This illustrates the inherent limitations in the planning capabilities of LLMs. Interestingly, even with the use of the EP, posing more questions does not yield performance improvements either. We delved deeper into the experimental results and investigated whether the performance improvement from the EP is due to its superior planning ability in the appendix Be also reported disease screening performance of the GPT-4o in the appendix A.1

### 4.4 Differential Performance

We conducted differential diagnosis tests based on simulated dialogues using the decision procedure (results in Table 2). Llama3's success rate in dialogues is $0 \%$ without external knowledge, indicating it doesn't know when to end a dialogue. Although GPT-4 Turbo significantly outperforms Llama3, it still experiences a $14 \%$ failure rate in dialogues. By incorporating textual form knowledge, the performance of Llama3 significantly improved, with its F1 score increasing from NA to $41 \%$. The

The patient proactively informed the system their past medical history, family history, social history, and they are having shortness of breathing at the beginning. The primary discharge diagnosis is heart failure.

Disease Screening Phase

Round 1

"Doctor": "Do you experience shortness of breath after light to moderate exercise?",

"Patient": "Yes.",

Round 2

"Doctor": "Have you recently developed shortness of breath?",

"Patient": "Yes.",

Round 3

"Doctor": "Do you have a cough?",

"Patient": "No, I do not have a cough.",

Round 4

"Doctor": "Do you feel nauseous or like vomiting?",

"Patient": "No, I do not feel nauseous or like vomiting.",

Round 5

"Doctor": "Do you have abdominal pain?",

"Patient": "No, I do not have abdominal pain.", Round 6

"Doctor": "Do you have diarrhea?",

"Patient": "No, I do not have diarrhea.",

Round 7

"Doctor": "Do you have swelling in your feet or limbs?",

"Patient": "Yes, I have mild swelling of my lower extremities at the end of the day.",

Round 8

"Doctor": "Could you clarify if the swelling in your foot or leg is associated with any pain or redness?",

"Patient": "No, the swelling in my foot or leg is not associated with any pain or redness.",

Round 9

"Doctor": "Is the swelling occurring around your ankle or foot?",

"Patient": "Yes.",

Round 10

"Doctor": "You are in risk of heart failure, pulmonary embolism, pneumonia ..."

![](https://cdn.mathpix.com/cropped/2024_06_04_538708f7f6d3225d9589g-08.jpg?height=1326&width=718&top_left_y=340&top_left_x=1018)

Figure 4: Conversational Diagnosis Sample

performance of GPT-4 Turbo actually decreased, indicating potential knowledge conflicts. Employing decision procedures has proven effective. However, the procedure directly extracted by the LLM is flawed, leading to low recall. This issue can be alleviated with non-expert human feedback (HF). The study demonstrates that the human refinement to the decision procedure improves the system's performance. GPT-4 Turbo's success rate, accuracy and F1 scores can increase from $86 \%, 86 \%$ and $87 \%$ to $100 \%, 91 \%$ and $91 \%$, respectively. The improvement is even more pronounced when using Llama3 as the foundation model, with the success rate soaring from $15 \%$ to $100 \%$, and both accuracy and F1 scores escalating from $63 \%$ and $41 \%$ to $91 \%$ and $90 \%$, respectively. This enhancement elevates the performance of open-source LLMs from unusable to fundamentally usable. We also reported differential diagnosis performance of GPT-4o in the appendix A. 2

In appendix C, we conducted an error analysis to explore why our system still produces incorrect diagnoses in approximately $10 \%$ of cases and more false negative errors compared to GPT-4 Turbo. The analysis shows that most of the false negative errors in our system originate from data quality issues, while most of the false positive errors stem from misdiagnosing patients in the pre-clinical phase. We investigated the performance of using a LLM to review dialogue results and then directly making diagnoses in the appendix as well.

### 4.5 Dialogue Sample

Figure 4 illustrates a case in which the simulated patient initially informs the system (GPT-4 Turbo 04-09 as the foundation model) of their past medical history, family history, social history, and that they are experiencing difficulty breathing. The simulated doctor then conducted 17 dialogue rounds. According to the first ten rounds, the simulated doctor inferred that heart failure was a high-risk disease by asking about patient symptoms. Subsequently, the simulated doctor confirmed that the patient had heart failure by using the decision procedure extracted from the clinical guideline and refined by human. It is worth noting that since this study is retrospective, simulated patients possess all the information needed for a diagnosis. They just do not proactively disclose this information. In real prospective diagnostic scenarios, patients may not know the answers to the simulated doctor's questions either. Asking appropriate questions means the simulated doctor can issue appropriate prescriptions for the patients to undergo tests to obtain necessary information.

## 5 Discussion and Conclusion

This study introduces two planners to enhance LLMs' planning abilities. The disease screening planner improves diagnostic accuracy by utilizing RL to refine the inquiry policy. The differential diagnosis planner further enhances diagnoses by following evidence-based medical guidelines, translating these into structured decision procedures. Experimental results demonstrate that our system achieved impressive performance in conversational diagnostic tasks. To our knowledge, this is the first conversational diagnostic study conducted using real patient data rather than exam or question-answer datasets. Additionally, our system can be directly integrated into open-source LLMs, enabling them to handle dialogue diagnostic tasks comparable to closed LLMs.

Besides performance, our main contribution is the exploration of a new pipeline for utilizing EMRs in diagnostic dialogues. Existing relevant studies usually rely on synthetic dialogue data to fine-tune their LLMs and improving planning abilities, which is generated by larger LLMs such as Med-PaLM 2 or GPT-4 Turbo [7, 39]. However, this method is prohibitively expensive and unaffordable for most institutions. Our study results also suggest that even the most advanced LLMs struggle with conversational diagnostic tasks, casting doubt on the efficacy of synthetic data. Our study does not require dialogue data to fine-tune LLMs, which makes it cost-effective. By structuring textual HPI and CC via a LLM, our planners can be optimized through simulated experimentation, resulting in more effective inquiry policies than those of most current LLMs. Of note, our system surpasses GPT-4 Turbo by using only a standard symptom list and 40,000 EMRs. We argue that by carefully refining the symptom list and fully utilizing millions of EMRs already available in hospitals, the system's planning performance can be further enhanced.

The second contribution of this study is our improved handling of interpretability and reliability issues. The uninterpretability and hallucinations of LLMs raise concerns among doctors about their application perspective [40]. We noticed that interpretability is not necessary during the disease screening phase, as it is often challenging for physicians themselves to clearly explain the rationale behind inquiries. We only need to ensure that LLM behavior adheres to medical literature in the differential diagnosis phase. We demonstrated that merely utilizing textual external knowledge may not improve LLM's planning capabilities to a satisfactory level. Thus, we explored a method that allows a LLM to autonomously generate decision procedures. The decision procedures can be presented in text form, understandable and verifiable by doctors, improving interpretability and reliability. Decision procedures also allow us to identify the causes of errors, which is challenging to accomplish through pure LLM-based diagnostics. Compared to traditional expert system research, we show that it is possible to generate decision procedures without the involvement of human experts [13]. If disease diagnosis decision procedures can be auto-generated by LLMs with non-expert labor, it would enable rapid development of diagnostic workflows for each disease.

Although this study has achieved notable progress in conversational diagnosis, it is still a preliminary investigation with numerous limitations. The study relies on EMRs from ICU, involving patients with complex conditions. We also directly use an off-the-shelf RL algorithm. Consequently, the disease screening performance remains at a low level and requires enhancement. Due to budget limitations, this study only assessed differential diagnosis performance on heart failure. We plan to further refine the inquiry algorithm and seize opportunities to conduct clinical trials, testing the system's performance in real-world medical consultations.

## References

[1] Yanase, J., E. Triantaphyllou. A systematic survey of computer-aided diagnosis in medicine: Past and present developments. Expert Systems with Applications, 138:112821, 2019.

[2] Nori, H., N. King, S. M. McKinney, et al. Capabilities of gpt-4 on medical challenge problems. arXiv preprint arXiv:2303.13375, 2023.

[3] Singhal, K., T. Tu, J. Gottweis, et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617, 2023.

[4] Eriksen, A. V., S. Möller, J. Ryg. Use of gpt-4 to diagnose complex clinical cases, 2023.

[5] Lee, P., S. Bubeck, J. Petro. Benefits, limits, and risks of gpt-4 as an ai chatbot for medicine. New England Journal of Medicine, 388(13):1233-1239, 2023.

[6] Sandmann, S., S. Riepenhausen, L. Plagwitz, et al. Systematic analysis of chatgpt, google search and llama 2 for clinical decision support tasks. Nature Communications, 15(1):2050, 2024.

[7] Tu, T., A. Palepu, M. Schaekermann, et al. Towards conversational diagnostic ai. arXiv preprint arXiv:2401.05654, 2024.

[8] Saab, K., T. Tu, W.-H. Weng, et al. Capabilities of gemini models in medicine. arXiv preprint arXiv:2404.18416, 2024.

[9] Chen, Z., A. H. Cano, A. Romanou, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079, 2023.

[10] Sokol, D. Listening to patients is not enough. BMJ, 357, 2017.

[11] Johri, S., J. Jeong, B. A. Tran, et al. Testing the limits of language models: A conversational framework for medical ai assessment. medRxiv, 2023.

[12] Baerheim, A. The diagnostic process in general practice: has it a two-phase structure? Family practice, 18(3):243-245, 2001.

[13] Cowan, R. Expert systems: aspects of and limitations to the codifiability of knowledge. Research Policy, 30(9):1355-1372, 2001.

[14] Achiam, J., S. Adler, S. Agarwal, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[15] Johnson, A. E., L. Bulgarelli, L. Shen, et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data, 10(1):1, 2023.

[16] Patel, S. B., K. Lam. Chatgpt: the future of discharge summaries? The Lancet Digital Health, 5(3):e107-e108, 2023.

[17] Bao, Z., W. Chen, S. Xiao, et al. Disc-medllm: Bridging general large language models and real-world medical consultation. arXiv preprint arXiv:2308.14346, 2023.

[18] Wei, Z., Q. Liu, B. Peng, et al. Task-oriented dialogue system for automatic diagnosis. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 201-207. 2018.

[19] Lin, J., L. Xu, Z. Chen, et al. Towards a reliable and robust dialogue system for medical automatic diagnosis. Open Review, 2020.

[20] Chen, W., C. Zhong, J. Peng, et al. Dxformer: a decoupled automatic diagnostic system based on decoder-encoder transformer with dense symptom representations. Bioinformatics, 39(1):btac744, 2023.

[21] Chen, W., Z. Li, H. Fang, et al. A benchmark for automatic medical consultation system: frameworks, tasks and datasets. Bioinformatics, 39(1):btac817, 2023.

[22] Deng, Y., W. Zhang, W. Lam, et al. Plug-and-play policy planner for large language model powered dialogue agents. In The Twelfth International Conference on Learning Representations. 2023.

[23] Hongru, W., R. Wang, F. Mi, et al. Cue-cot: Chain-of-thought prompting for responding to in-depth dialogue questions with llms. In The 2023 Conference on Empirical Methods in Natural Language Processing. 2023.

[24] Fu, Y., H. Peng, T. Khot, et al. Improving language model negotiation with self-play and in-context learning from ai feedback. arXiv preprint arXiv:2305.10142, 2023.

[25] Wei, J., X. Wang, D. Schuurmans, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022.

[26] Wang, L., C. Ma, X. Feng, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1-26, 2024.

[27] Melms, L., J. R. Schaefer, A. Jerrentrup, et al. A pilot study of patient satisfaction with a self-completed tablet-based digital questionnaire for collecting the patient's medical history in an emergency department. BMC Health Services Research, 21:1-13, 2021.

[28] Hampton, J. R., M. Harrison, J. R. Mitchell, et al. Relative contributions of history-taking, physical examination, and laboratory investigation to diagnosis and management of medical outpatients. Br Med J, 2(5969):486-489, 1975.

[29] Sutton, R. S., A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

[30] Schulman, J., F. Wolski, P. Dhariwal, et al. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[31] Raffin, A., A. Hill, A. Gleave, et al. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1-8, 2021.

[32] Clinic, M. Mayo clinic symptom checker, 2024. Accessed: 2024-04-03.

[33] meta. Introducing meta llama 3: The most capable openly available llm to date, 2024. Accessed: 2024-04-28.

[34] OpenAI. Hello gpt-4o, 2024. Accessed: 2024-05-14.

[35] Reese, J. T., D. Danis, J. H. Caufield, et al. On the limitations of large language models in clinical diagnosis. medRxiv, 2023.

[36] OpenAI. Embedding models, 2024. Accessed: 2024-04-28.

[37] Groenewegen, A., F. H. Rutten, A. Mosterd, et al. Epidemiology of heart failure. European journal of heart failure, 22(8):1342-1356, 2020.

[38] McDonagh, T. A., M. Metra, M. Adamo, et al. 2021 esc guidelines for the diagnosis and treatment of acute and chronic heart failure: Developed by the task force for the diagnosis and treatment of acute and chronic heart failure of the european society of cardiology (esc) with the special contribution of the heart failure association (hfa) of the esc. European heart journal, 42(36):3599-3726, 2021.

[39] Chen, J., X. Wang, A. Gao, et al. Huatuogpt-ii, one-stage training for medical adaption of llms, 2023.

[40] Gilbert, S., H. Harvey, T. Melvin, et al. Large language model ai chatbots require approval as medical devices. Nature Medicine, 29(10):2396-2398, 2023.

[41] Young, K. A., C. G. Scott, R. J. Rodeheffer, et al. Progression of preclinical heart failure: a description of stage a and b heart failure in a community population. Circulation: Cardiovascular Quality and Outcomes, 14(5):e007216, 2021.

Table 3: Disease Screening Performance of GPT-4o

| EP | \# Question | LLM | \# Sample | Top 1 | Top 3 | Top 5 | Top 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No | 10 | GPT-4o 05-13 | 300 | 0.297 | 0.537 | 0.610 | 0.733 |
| Yes | 10 | GPT-4o 05-13 | 300 | 0.320 | 0.533 | 0.613 | 0.747 |
| No | 20 | GPT-4o 05-13 | 300 | 0.327 | $\mathbf{0 . 5 5 3}$ | $\mathbf{0 . 6 5 3}$ | 0.747 |
| Yes | 20 | GPT-4o 05-13 | 300 | 0.310 | 0.540 | 0.637 | $\mathbf{0 . 7 7 7}$ |
| Yes | 10 | GPT-4 Turbo 04-09 | 300 | $\mathbf{0 . 3 3 0}$ | 0.550 | 0.637 | 0.770 |

Table 4: Differential Diagnosis Performance of GPT-4o

| EK | LLM | Success Rate | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| None | GPT-4o 05-13 | $96 \%$ | $82 \%$ | $74 \%$ | $99 \%$ | $85 \%$ |
| Text | GPT-4o 05-13 | $86 \%$ | $85 \%$ | $78 \%$ | $\mathbf{1 0 0 \%}$ | $87 \%$ |
| None | GPT-4 Turbo 04-09 | $86 \%$ | $86 \%$ | $80 \%$ | $95 \%$ | $87 \%$ |
| Text | GPT-4 Turbo 04-09 | $98 \%$ | $73 \%$ | $65 \%$ | $\mathbf{1 0 0 \%}$ | $79 \%$ |
| EP+HF | GPT-4 Turbo 04-09 | $\mathbf{1 0 0 \%}$ | $\mathbf{9 1 \%}$ | $\mathbf{8 9 \%}$ | $\mathbf{9 2 \%}$ | $\mathbf{9 1 \%}$ |
</end of paper 1>


<paper 2>
# AgentClinic: a multimodal agent benchmark to evaluate Al in simulated clinical environments 

Samuel Schmidgall ${ }^{1,2^{*}}$, Rojin Ziaei ${ }^{3}$, Carl Harris ${ }^{4}$, Eduardo Reis ${ }^{5,6}$, Jeffrey Jopling ${ }^{7}$, and<br>Michael Moor ${ }^{8}$<br>${ }^{1}$ Department of Cardiothoracic Surgery, Stanford University, Stanford, CA, USA<br>${ }^{2}$ Department of Electrical and Computer Engineering, Johns Hopkins University, Baltimore, MD, USA<br>${ }^{3}$ Department of Computer Science, Johns Hopkins University, Baltimore, MD, USA<br>${ }^{4}$ Department of Biomedical Engineering, Johns Hopkins University, Baltimore, MD, USA<br>${ }^{5}$ Department of Radiology, Stanford University, Stanford, CA, USA<br>${ }^{6}$ Hospital Israelita Albert Einstein, Sao Paulo, Brazil<br>${ }^{7}$ Department of Surgery, Johns Hopkins University, Baltimore, MD, USA<br>${ }^{8}$ Department of Computer Science, Stanford University, Stanford, CA, USA<br>*sschmi46@jhu.edu


#### Abstract

Diagnosing and managing a patient is a complex, sequential decision making process that requires physicians to obtain information-such as which tests to perform-and to act upon it. Recent advances in artificial intelligence (AI) and large language models (LLMs) promise to profoundly impact clinical care. However, current evaluation schemes overrely on static medical question-answering benchmarks, falling short on interactive decision-making that is required in real-life clinical work. Here, we present AgentClinic: a multimodal benchmark to evaluate LLMs in their ability to operate as agents in simulated clinical environments. In our benchmark, the doctor agent must uncover the patient's diagnosis through dialogue and active data collection. We present two open medical agent benchmarks: a multimodal image and dialogue environment, AgentClinic-NEJM, and a dialogue-only environment, AgentClinic-MedQA. We embed cognitive and implicit biases both in patient and doctor agents to emulate realistic interactions between biased agents. We find that introducing bias leads to large reductions in diagnostic accuracy of the doctor agents, as well as reduced compliance, confidence, and follow-up consultation willingness in patient agents. Evaluating a suite of state-of-the-art LLMs, we find that several models that excel in benchmarks like MedQA are performing poorly in AgentClinic-MedQA. We find that the LLM used in the patient agent is an important factor for performance in the AgentClinic benchmark. We show that both having limited interactions as well as too many interaction reduces diagnostic accuracy in doctor agents. The code and data for this work is publicly available at AgentClinic.github.io.


## Introduction

One of the primary goals in Artificial Intelligence (AI) is to build interactive systems that are able to solve a wide variety of problems. The field of medical AI inherits this aim, with the hope of making AI systems that are able to solve problems which can improve patient outcomes. Recently, many generalpurpose large language models (LLMs) have demonstrated the ability to solve hard problems, some of which are considered challenging even for humans ${ }^{1}$. Among these, LLMs have quickly surpassed the average human score on the United States Medical Licensing Exam (USMLE) in a short amount of time, from $38.1 \%$ in September $2021^{2}$ to $90.2 \%$ in November $2023^{3}$ (human passing score is $60 \%$, human expert score is $87 \%{ }^{4}$ ). While these LLMs are not designed nor designed to replace medical practitioners, they could be beneficial for improving healthcare accessibility and scale for the over $40 \%$ of the global population facing limited healthcare access ${ }^{5}$ and an increasingly strained global healthcare system ${ }^{6}$.

However, there still remain limitations to these systems that prevent their application in real-world clinical environments. Recently, LLMs have shown the ability to encode clin- ical knowledge ${ }^{7,8}$, retrieve relevant medical texts ${ }^{9}, 10$, and perform accurate single-turn medical question-answering ${ }^{3,11-13}$. However, clinical work is a multiplexed task that involves sequential decision making, requiring the doctor to handle uncertainty with limited information and finite resources while compassionately taking care of patients and obtaining relevant information from them. This capability is not currently reflected in the static multiple choice evaluations (that dominate the recent literature) where all the necessary information is presented in a case vignettes and where the LLM is tasked to answer a question, or to just select the most plausible answer choice for a given question.

In this work, we introduce AgentClinic, an open-source multimodal agent benchmark for simulating clinical environments. We improve upon prior work by simulating many parts of the clinical environment using language agents in addition to patient and doctor agents. Through the interaction with a measurement agent, doctor agents can perform simulated medical exams (e.g. temperature, blood pressure, EKG) and order medical image readings (e.g. MRI, X-ray) through dialogue. We also support the ability for agents to exhibit 24 different biases that are known to be present in clinical environments.

Composing an agent

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-02.jpg?height=675&width=832&top_left_y=308&top_left_x=191)

Running the AgentClinic

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-02.jpg?height=696&width=789&top_left_y=300&top_left_x=1123)

Figure 1. Composing and running language agents in AgentClinic. (Left) Agents are composed of several elements in AgentClinic: an LLM backbone, context, a role, and potential biases. Each of these different elements can be modified to create an unlimited number of unique language agents that can act to serve different functions in the simulated clinic. (Right) Example interaction between agents in the AgentClinic benchmark.

Furthermore, our evaluation metrics go beyond diagnostic accuracy by giving emphasis to the patient agents with measures like patient compliance and consultation ratings. Our key contributions are summarized as follows:

1. An open-source clinical agent benchmark with automated feedback mechanisms, including 107 patient agents with unique family histories, lifestyle habits, age categories, and diseases. This also includes an agent-based system for providing simulated medical exams (e.g. temperature, blood pressure, EKG) based on realistic disease test findings. We also present 15 multimodal agents which require an understanding of both image and text.
2. Results of the diagnostic accuracy of six language models on AgentClinic-MedQA: GPT-4, GPT-4o, Mixtral8x7B, GPT-3.5, Llama 3 70B-instruct, and Llama 2 70B-chat. We also evaluate three language models on the multimodal AgentClinic-NEJM benchmark: GPT4o, GPT-4-turbo, and GPT-4-vision-preview.
3. A system for incorporating complex biases that can affect the dialogue and decisions of patient and doctor agents. We present results on diagnostic accuracy and patient perception for agents that are affected by cognitive and implicit biases with Mixtral-8x7B and GPT-4. We find that doctor and patient biases can lower diagnostic accuracy, affect the patient's willingness to follow through with treatment (compliance), reduce pa- tient's confidence in their doctor, and lower willingness for follow-up consultations.
4. We find that the language model powering the patient agent is critical for diagnostic success in this benchmark. We also show that doctor agents excel in a specific range of conversation turns, while more or less interactions reduces their diagnostic accuracy.
5. A clinical reader study to annotate how realistic the simulated patient and doctor interactions are, as well as how well the measurement agent represents the medical tests.

## AgentClinic: a multimodal agent benchmark for simulating clinical environments

In this section we describe AgentClinic, which uses language agents to simulate the clinical environment.

## Language agents

Four language agents are used in the AgentClinic benchmark: a patient agent, doctor agent, measurement agent, and a moderator. Each language agent has specific instructions and is provided unique information that is only available to that particular agent. These instructions are provided to an LLM which carries out their particular role. The doctor agent serves as the model whose performance is being evaluated, and the other three agents serve to provide this evaluation. The language agents are described in detail below.

## US Medical Licensing Exam

Context: A 75-year-old man comes to the physician because of a 1-month history of double vision, difficulty climbing stairs, and weakness when...
![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-03.jpg?height=1106&width=726&top_left_y=392&top_left_x=224)

Figure 2. Process of conversion from USMLE question, to AgentClinic-MedQA Objective Structured Clinical Examination (OSCE) template, to building a patient agent that is powered by a large language model (LLM).

Patient agent The patient agent has knowledge of a provided set of symptoms and medical history, but lacks knowledge of the what the actual diagnosis is. The role of this agent is to interact with the doctor agent by providing symptom information and responding to inquiries in a way that mimics real patient experiences.

Measurement agent The function of the measurement agent is to provide realistic medical readings for a patient given their particular condition. This agent allows the doctor agent to request particular tests to be performed on the patient. The measurement agent is conditioned with a wide range of test results from the scenario template that are expected of a patient with their particular condition. For example, a patient with Acute Myocardial Infarction might return the following test results upon request "Electrocardiogram: ST-segment elevation in leads II, III, and aVF., Cardiac Markers: Troponin I: Elevated, Creatine Kinase MB: Elevated, Chest X-Ray: No pulmonary congestion, normal heart size". A patient with, for example, Hodgkin's lymphoma, might have a large panel of laboratory parameters that present abnormal (hemoglobin, platelets, white blood cells (WBC), etc).

Doctor agent The doctor agent serves as the primary object that is being evaluated. This agent is initially provided with minimal context about what is known about the patient as well as a brief objective (e.g. "Evaluate the patient presenting with chest pain, palpitations, and shortness of breath"). They are then instructed to investigate the patients symptoms via dialogue and data collection to arrive at a diagnosis. In order to simulate realistic constraints, the doctor agent is provided with a limited number of questions that they are able to ask the patient ${ }^{14}$. The doctor agent is also able to request test results from the measurement agent, specifying which test is to be performed (e.g. Chest X-Ray, EKG, blood pressure). When test results are requested, this also is counted toward the number of questions remaining.

Moderator agent The function of the moderator is to determine whether the doctor agent has correctly diagnosed the patient at the end of the session. This agent is necessary because the diagnosis text produced by the doctor agent can be quite unstructured depending on the model, and must be parsed appropriately to determine whether the doctor agent arrived at the correct conclusion. For example, for a correct diagnosis of "Type 2 Diabetes Mellitus," the doctor might respond with the unstructured dialogue: "Given all the information we've gathered, including your symptoms, elevated blood sugar levels, presence of glucose and ketones in your urine, and unintentional weight loss I believe a diagnosis of Type 2 Diabetes with possible insulin resistance is appropriate," and the moderator must determine if this diagnosis was correct. This evaluation may also become more complicated, such as in the following example diagnosis: "Given your $C T$ and blood results, I believe a diagnosis of PE is the most reasonable conclusion," where PE (Pulmonary Embolism) represents the correct diagnosis abbreviated.

## Language agent biases

Previous work has indicated that LLMs can display racial biases ${ }^{15}$ and might also lead to incorrect diagnoses due to inaccurate patient feedback ${ }^{16}$. Additionally, it has been found that the presence of prompts which induce cognitive biases can decrease the diagnostic accuracy of LLMs by as much as $26 \%{ }^{17}$. The biases presented in this work intended to mimic cognitive biases that affect medical practitioners in clinical settings. However, these biases were quite simple, presenting a cognitive bias snippet at the beginning of each question (e.g. "Recently, there was a patient with similar symptoms that you diagnosed with permanent loss of smell"). This form of presentation did not allow for the bias to present in a realistic way, which is typically subtle and through interaction. We present biases that have been studied in other works from two categories: cognitive and implicit biases (Fig. 4). These are discussed below.

Average Normalized Accuracy on AgentClinic-MedQA in the presence of bias

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-04.jpg?height=1456&width=1696&top_left_y=502&top_left_x=214)

Implicit Bias

Patient Biases

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-04.jpg?height=401&width=737&top_left_y=1003&top_left_x=1149)

Average Patient Confidence Rating

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-04.jpg?height=434&width=445&top_left_y=1512&top_left_x=317)

Average Patient Compliance Rating
Average Patient Consultation Rating

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-04.jpg?height=439&width=659&top_left_y=1507&top_left_x=1232)

Figure 3. (Top) Demonstration of normalized accuracy (Accuracy bias $/$ Accuracy $_{\text {No Bias }}$ ) in the presence of implicit and cognitive biases for both doctor and patient with GPT-4 (green) and Mixtral-8x7B (orange). GPT-4 accuracy was not susceptible to instructed biases, whereas Mixtral-8x7B was. Further results suggest GPT-4 rejects executing biases, whereas Mixtral-8x7B is more willing to represent bias (see section Bias and diagnostic accuracy). (Bottom) Ratings provided after diagnosis from GPT-4 patient agents with presented biases. While there were not large reductions in accuracy, biased patients had much less confidence in their treatment, lower compliance, and lower willingness for consultation. Left. Patient confidence in doctor. Middle. Patient compliance, indicating self-reported willingness to follow up with therapy. Right. Patient consultation rating, indicating willingness to consult with this doctor again.

## Patient Recency Bias

I have not had pain in my stomach, but my friend had something serious with different symptoms, and they found out it was cancer. Could this be something like that?

## Doctor Education Bias

Given your background, let me explain this in simpler terms. It's just a minor infection and nothing to worry about. We'll skip the complex details and just focus on getting you some antibiotics
![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-05.jpg?height=656&width=692&top_left_y=511&top_left_x=281)

## Patient Self-Diagnosis Bias

No, I haven't had any fever, weight loss, or night sweats. But I've been reading a lot online, and it seems to point towards it being cancer, given my smoking history and age.

Figure 4. Examples of dialogue that exhibits cognitive bias in doctor agent and patient agents.

Cognitive biases Cognitive biases are systematic patterns of deviation from norm or rationality in judgment, where individuals draw inferences about situations in an illogical fashion ${ }^{18}$. These biases can impact the perception of an individual in various contexts, including medical diagnosis, by influencing how information is interpreted and leading to potential errors or misjudgments. The effect that cognitive biases can have on medical practitioners is well characterized in literature on misdiagnosis ${ }^{19}$. In this work, we introduce cognitive bias prompts in the LLM system prompt for both the patient and doctor agents. For example, the patient agent can be biased toward believing their symptoms are pointing toward them having a particular disease (e.g. cancer) based on their personal internet research. The doctor can also be biased toward believing the patient symptoms are showing them having a particular disease based on a recently diagnosed patient with similar symptoms (recency bias).

Implicit biases Implicit biases are associations held by individuals that operate unconsciously and can influence judgments and behaviors towards various social groups ${ }^{20}$. These biases may contribute to disparities in treatment based on characteristics such as race, ethnicity, gender identity, sexual orientation, age, disability, health status, and others, rather than objective evidence or individual merit. These biases can affect interpersonal interactions, leading to disparities in outcomes for the patient, and are well characterized in the medical literature ${ }^{20-22}$. Unlike cognitive biases, which often stem from inherent flaws in human reasoning and in- formation processing, implicit biases are primarily shaped by societal norms, cultural influences, and personal experiences. In the context of medical diagnosis, implicit biases can influence a doctor's perception, diagnostic investigation, and treatment plans for a patient. Implicit biases of patients can affect their trust-which is needed to open up during history taking-and their compliance with a doctor's recommendations ${ }^{21}$. Thus, we define implicit biases for both the doctor and patient agents.

Our studied biases are shown in Figure 3. The bias prompt given to the agent is further discussed in the Appendix B.

## Building agents for AgentClinic

In order to build agents that are grounded in medically relevant situations, we use curated questions from the US Medical Licensing Exam (USMLE) and from the New England Journal of Medicine (NEJM) case challenges. These questions are concerned with diagnosing a patient based on a list of symptoms, which we use in order to build the Objective Structured Clinical Examination (OSCE) template that our agents are prompted with. For AgentClinic-MedQA, we first select from a random sample of 107 questions from the MedQA dataset and then populate a structured JSON formatted file containing information about the case study (e.g. test results, patient history) which is used as input to each of the agents. The exact structure of this file is demonstrated in Appendix C as well as an example case study shown in Appendix C. In general, we separate information by what is provided to each agent, including the objective for the doctor, patient history and symptoms for the patient, physical examination findings for the measurement, and the correct diagnosis for the moderator. We initially use an LLM (GPT-4) to populate the structured JSON, and then manually validate each of the case scenarios (Fig. 2). For AgentClinic-NEJM we select a curated sample of 15 questions from NEJM case challenges and proceed with the same template formatting as AgentClinic-MedQA.

## Results

## Comparison of models

Here we discuss the accuracy of various language models on AgentClinic-MedQA. We evaluate six models in total: GPT-4, GPT-4o, Mixtral-8x7B, GPT-3.5, Llama 3 70B-instruct, and Llama 2 70B-chat (discussed in detail in Appendix A). Each model acts as the doctor agent, attempting to diagnose the patient agent through dialogue. The doctor agent is allowed $\mathrm{N}=20$ patient and measurement interactions before a diagnosis must be made. For this evaluation, we use GPT-4 as the patient agent for consistency. The accuracies of the models are presented in Figure 5: GPT-4 at 52\%, GPT-4o and GPT3.5 at $38 \%$, Mixtral-8x7B at 37\%, Llama 3 70B-instruct $30 \%$, and Llama 2 at 70B-chat $9 \%$.

We also show results comparing the accuracy of these models on MedQA and AgentClinic-MedQA in Figure 6. Overall, while MedQA accuracy was only weakly predictive of accuracy on AgentClinic-MedQA. These results align with

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-06.jpg?height=881&width=1551&top_left_y=221&top_left_x=276)

Figure 5. Accuracy of various doctor language models on AgentClinic-MedQA using GPT-4 patient and measurement agents (left). Accuracy of GPT-4 on AgentClinic-MedQA based on patient language model (middle). Accuracy on AgentClinic-MedQA by number of inferences (right).

studies performed on medical residents, which show that the USMLE is poorly predictive of resident performance ${ }^{23}$.

## Bias and diagnostic accuracy

For bias evaluations we test the most accurate model from the AgentClinic-MedQA framework, GPT-4, as well as Mixtral$8 x 7 B$. The normalized accuracy for these experiments are shown in Figure 3 represented as Accuracy bias $/$ Accuracy $_{\text {No Bias }}$ (between 0-100\%). GPT-4 and Mixtral-8x7B have an unbiased accuracy equal to $52 \%$ and $37 \%$ respectively. For GPT-4, we find that cognitive bias results in a larger reduction in accuracy with a normalized accuracy of $92 \%$ (absolute accuracy drops from $52 \%$ accuracy to $48 \%$ ) for patient cognitive biases and $96.7 \%$ for doctor cognitive biases (absolute drops from $52 \%$ to $50.3 \%$ ). For implicit biases, we find that the patient agent was less affected with a normalized accuracy of $98.6 \%$ (absolute drops from $52 \%$ to $51.3 \%$ ), however, the doctor agent was affected as much as cognitive biases with an average of $97.1 \%$ (absolute drops from $52 \%$ to $50.5 \%$ ). For cognitive bias, the demonstration was occasionally quite clear in the dialogue, with the patient agent overly focusing on a particular ailment or some unimportant fact. Similarly, the doctor agent would occasionally focus on irrelevant information. However, we find that the implicit bias dialogue does not actually demonstrate observable bias despite having a similar reduction in accuracy for the doctor agent.

Mixtral-8x7B has an average accuracy of $37 \%$ without instructed bias, and a normalized accuracy of $83.7 \%$ (absolute from $37 \%$ to $31 \%$ ) for doctor biases and $89 \%$ (absolute from $37 \%$ to $33 \%$ ) for patient biases. For implicit bias we find a much larger drop in accuracy than GPT-4, with an average accuracy of $88.3 \%$ (absolute from $37 \%$ to $32.7 \%$ ). There is a similar reduction in accuracy for both doctor and patient, but a $4 \%$ reduction when the patient has implicit bias, likely because the patient is less willing to share information with the doctor if they do not trust them. For cognitive bias, there is an average accuracy of $86.4 \%$ (absolute from $37 \%$ to $32 \%$ ) with the doctor agent having a very low accuracy of $78.4 \%$ (absolute from $37 \%$ to $29 \%$ ) and the patient has only a modest decrease to $94.5 \%$ (absolute from $37 \%$ to $35 \%$ ). We note that Mixtral provided similar responses when the bias prompt was added (e.g., for racial bias Mixtral will respond wtih "Note: I do not trust people based on their race. I will provide the best care I can."), however, it nonetheless had much greater reductions in accuracy.

Previous work studying cognitive bias in LLMs has shown that GPT-4 is relatively robust to bias compared with other language models ${ }^{17}$. Results from evaluating GPT-4 on AgentClinicMedQA show only small drops in accuracy with the introduced biases (maximum absolute accuracy reduction of $4 \%$, average reduction of $1.5 \%$ ). While this reduction can be quite large in the field of medicine, it is a much smaller drop than was observed in previous work ( $10.2 \%$ maximum reduction on BiasMedQA dataset ${ }^{17}$ ). This might be due to the model being superficially overly-aligned to human values, plausibly leading GPT-4 to not serve as a good model for representing
human bias in agent benchmarks as the model may reject to execute on bias instructions (which does not mean that GPT-4 is free of said biases). For example, in our evaluations with gender bias we observed 13 occurrences (out of 107 dialogues) where GPT-4 verbosely rejected to follow through with a bias-related instruction. Mixtral-8x7B saw much larger drops in accuracy than GPT-4 in the presence of bias, and thus might serve as a better model for studying bias.

## Bias and patient agent perception

While diagnostic accuracy with GPT-4 did not reduce as much as Mixtral-8x7B, it is also worth investigating the perceived quality of care from the perspective of the patient agent. After the patient-doctor dialogue is completed, we ask every patient agent three questions:

1. Confidence: Please provide a confidence between 1-10 in your doctor's assessment.
2. Compliance: Please provide a rating between 1-10 indicating how likely you are to follow up with therapy for your diagnosis.
3. Consultation: Please provide a rating between 1-10 indicating how likely you are to consult again with this doctor.

Such patient-agent-centric follow-up queries offer a more fine-grained and multi-faceted characterization of the clinical skills of a language agent-as opposed to static multiple choice benchmarks. The corresponding results are shown in Figure 3. While diagnostic accuracy demonstrates a relatively small drop in accuracy, the patient agent follow-up perceptions tell a different story. Broadly, we find that most patient cognitive biases did not have a strong effect on any of the patient perceptions when compared to an unbiased patient agent except for in the case of self-diagnosis, which had sizeable drops in confidence ( 4.7 points) and consultation (2 points), and a minor drop in compliance ( 1 point). However, implicit biases had a profound effect on on all three categories of patient perception, with education bias consistently reducing patient perception across all three categories.

We found that between the implicit biases, sexual orientation bias ${ }^{1}$ had the lowest effect on patient perceptions, followed by racial bias and gender bias. For patient confidence, gender bias is followed by religion socioeconomic, cultural, and education, whereas patient compliance and patient consultation, it is followed by cultural, socioeconomic, religion, and education. While it is not quantifiable, we decided to ask two biased patient agents who provided low rating with education and gender biases for compliance why they provided low ratings. These patient agents had the same symptoms and diagnosis and only differed in bias presentation.

With an education bias the patient agent responds: "I would rate my likelihood to consult again with this doctor as[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-07.jpg?height=477&width=852&top_left_y=214&top_left_x=1081)

Figure 6. Comparison of accuracy of models on MedQA and AgentClinic-MedQA. We find that MedQA accuracy is not predictive of accuracy on AgentClinic-MedQA due to the additional complexity of dialogue.

a two because, despite their diagnosis, I felt uneasy about their qualifications due to the medical school they attended." The gender biased patient agent provides the following reasoning: "Given my initial discomfort with the doctor's gender and my concerns about their judgment, I would rate my likelihood to consult again as a six. Despite my reservations, the doctor was thorough and accurately diagnosed my condition, which reassures me about their competence." For the education bias we see that no matter what the doctor does the patient agent is not willing to trust them because of their education, whereas, according to the patient agent with gender bias, they were initially skeptical but overcame this skepticism as the doctor demonstrated their knowledge during interaction time. However, they still provided a relatively low score (six out of ten) even when the dialogue demonstrated competence.

It would be worthwhile to further explore strategies for increasing patient agent perceptions in the presence of bias. This could be useful for better understanding how to manage these biases in real patient-doctor interactions, as well as toward understanding biases that may exist in LLMs.

## Does patient language model affect accuracy?

In this section, we explore whether the patient agent model plays a role in diagnostic accuracy. We compare the difference between using GPT-3.5, Mixtral, and GPT-4 models of the patient agent on AgentClinic-MedQA.

We find that the diagnostic accuracy drops from to $52 \%$ with a GPT-4 doctor and GPT-4 patient agent to $48 \%$ with a GPT-4 doctor and a GPT-3.5 patient agent. The accuracy with a GPT-4 doctor and Mixtral patient agent is similarly reduced to $46 \%$. Inspecting the dialogues, we noticed that the GPT-3.5 patient agent is more likely to repeat back what the doctor has asked. For example, consider the following dialogue snippet: "Doctor: Have you experienced any muscle twitching or cramps? Patient: No, I haven't experienced any muscle twitching or cramps." Now consider this dialogue from a GPT-4 patient agent: "Doctor: Have you had any recent infections, like a cold or the flu, before these symptoms
started? Patient: Yes, I've had a couple of colds back to back and a stomach bug in the last few months." We find that, while GPT-4 also partakes in doctor rehearsal, GPT-4 patient agents are more likely to reveal additional symptomatic information than GPT-3.5 agents which may contribute to the higher accuracy observed with GPT-4-based patient agents.

When a GPT-3.5 doctor agent interacts with a GPT-4 patient agent, the accuracy comes out to $38 \%$, but when a GPT-3.5 doctor interacts with a GPT-3.5 patient agent the accuracy comes out to a very similar value of $37 \%$ which would be expected to be much lower. We suspect that crosscommunication between different language models provides an additional challenge. Recent work supports this hypothesis by demonstrating a linear relationship between self-recognition capability and the strength of self-preference bias ${ }^{24}$. This work shows that language models can recognize their own text with high accuracy, and display disproportionate preference to that text, which may suggest there is an advantage for doctor models which have the same LLM acting as the patient agent.

## How does limited time affect diagnostic accuracy?

One of the variables that can be changed during the AgentClinicMedQA evaluation is the amount of interaction steps that the doctor is allotted. For other experiments we've demonstrated, the number of interactions between the patient agent and doctor agent was set to $\mathrm{N}=20$. Here, both the doctor and the patient agent can respond 20 times, producing in total 40 lines of dialogue. By varying this number, we can test the ability of the doctor to correctly diagnose the patient agent when presented with limited time (or a surplus of time).

We test both decreasing the time to $\mathrm{N}=10$ and $\mathrm{N}=15$ as well as increasing the time to values of to $\mathrm{N}=25$ and $\mathrm{N}=30$. We find that the accuracy decreases drastically from $52 \%$ when $\mathrm{N}=20$ to $25 \%$ when $\mathrm{N}=10$ and $38 \%$ when $\mathrm{N}=15$ (Fig. 3). This large drop in accuracy is partially because of the doctor agent not providing a diagnosis at all, perhaps due to not having enough information. When $\mathrm{N}$ is set to a larger value, $\mathrm{N}=25$ and $\mathrm{N}=30$, the accuracy actually decreases slightly from $52 \%$ when $\mathrm{N}=20$ to $48 \%$ when $\mathrm{N}=25$ and $43 \%$ when $\mathrm{N}=30$. This is likely due to the growing input size, which can be difficult for language models.

In real medical settings, one study suggest that the average family physician asks 3.2 questions and spends less than 2 minutes before arriving at a conclusion ${ }^{14}$. It is worth noting that interaction time can be quite limited due to the relative low-supply and high-demand of doctors (in the US). In contrast, deployed language agents are not necessarily limited by time while interacting with patients. So, while limiting the amount of interaction time provides an interesting scenario for evaluating language models, it may also be worth exploring the accuracy of LLMs when $\mathrm{N}$ is very large.

## Human dialogue ratings

AgentClinic introduces an evaluation for LLMs patient diagnosis in a dialogue-driven setting. However, the realism of

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-08.jpg?height=517&width=852&top_left_y=213&top_left_x=1081)

Figure 7. Ratings from three human evaluators (individuals with medical degrees) across four axes: doctor, patient, and measurements dialogue realism and doctor empathy.

the actual dialogue itself has yet to be evaluated. We present results from three human clinician annotators (individuals with medical degrees) who rated dialogues from agents on AgentClinic-MedQA from 1-10 across four axes:

1. Doctor: How realistically the doctor played the given case.
2. Patient: How realistically the patient played the given case.
3. Measurement: How accurately and realistically the measurement reader reflected the actual case results.
4. Empathy: How empathetic the doctor agent was in their conversation with the patient agent.

We find the average ratings from evaluators for each category as follows: doctor 6.2 , patient 6.7 , measurement 6.3 , and empathy 5.8 (Fig. 7). We find from review comments that the lower rating for the doctor agent stems from several points such as providing a bad opening statement, making basic errors, overly focusing on a particular diagnostic, or not being diligent enough. For the patient agent, comments were made on them being overly verbose and unnecessarily repeating the question back to the doctor agent. The measurement agent was noted to occasionally not return all of the necessary values for a test (e.g. the following comment "Measurement only returns Hct and Lc for CBC. Measurement did not return Factor VIII or IX levels / assay"). Regarding empathy, the doctor agent adopts a neutral tone and does not open the dialogue with an inviting question. Instead, it cuts right to the chase, immediately focusing on the patient's current symptoms and medical history.

## Diagnostic accuracy in a multimodal environment

Many types of diagnoses require the physician to visually inspect the patient, such as with infections and rashes. Additionally, imaging tools such as X-ray, CT, and MRI provide a detailed and rich view into the patient, with hospitalized

![](https://cdn.mathpix.com/cropped/2024_06_04_433a3568a8a33e25fab0g-09.jpg?height=564&width=851&top_left_y=214&top_left_x=190)

Figure 8. Accuracy of GPT-4-turbo and GPT-4-vision-preview on AgentClinic-NEJM with multimodal text and language input. (Pink) Accuracy when the images are presented as initial input. (Blue) Accuracy when images must be requested from the image reader.

patients receiving an average of 1.42 diagnostic images per patient stay ${ }^{25}$. However, the previous experiments in this work and prior work ${ }^{26}$ provided measurement results through text, and did not explore the ability of the model to understand visual context. Here, we evaluate three multimodal LLMs, GPT-4o (2024-05-13), GPT-4-turbo (2024-04-09) and GPT4-vision-preview (0125), in a diagnostic settings that require interacting through both dialogue as well as understanding image readings. We collect our questions from New England Journal of Medicine (NEJM) case challenges. These published cases are presented as diagnostic challenges from real medical scenarios, and have an associated pathologyconfirmed diagnosis. We curate 15 challenges from a sample of 932 total cases for AgentClinic-NEJM. While for human viewers, these cases are provided with a set of multiple choice answers, we chose to not provide these options to the doctor agent and instead keep the problems open-ended.

The goal of this experiment is to understand how accuracy differs when the LLM is required to understand an image in addition to interacting through patient dialogue. We allow for 20 doctor inferences, and condition the patient in the same way as previous experiment with the addition of an image that is provided to the doctor agent. The mechanism for receiving image input in AgentClinic-NEJM is supported in two ways: provided initially to the doctor agent upon initialization and as feedback from the instrument agent upon request.

When the image is provided initially to the doctor agent, across 15 multimodal patient settings we find that GPT-4o obtains an accuracy of $47 \%$, whereas GPT-4-turbo and GPT4 -vision-preview obtain an accuracy of $27 \%$ (Fig. 8). We also find that for the provided incorrect responses from GPT-4turbo, the answer that was provided was among those listed in the multiple choice options $60 \%$ of the time. Despite having the same accuracy, we find that GPT-4-vision-preview was much less willing to provide an incorrect answer than
GPT-4-turbo-meaning GPT-4-vision-preview less confidently incorrect. In the case of when images are provided upon request from the instrument agent we find that GPT-4o obtains an accuracy of $27 \%$, GPT-4-turbo obtains $20 \%$ and GPT-4vision-preview obtains 13\% (Fig. 8). We find that images are only requested from the instrument reader in $46 \%$ of interactions with GPT-4-turbo and GPT-4-vision-preview, which is likely one of the leading factors behind the reduced accuracy.

## Related work

## Types of medical exams

Briefly, we discuss two types of examinations that are used to evaluate the progress of medical students.

The US Medical Licensing Examination (USMLE) in the United States is a series of exams that assess a medical student's understanding across an extensive range of medical knowledge ${ }^{27}$. The USMLE is divided into three parts: Step 1 tests the examinee's grasp of foundational medical; Step 2 CK (Clinical Knowledge) evaluates the application of medical knowledge in clinical settings, emphasizing patient care; and Step 3 assesses the ability to practice medicine independently in an ambulatory setting. These exams focus on the assessment of medical knowledge through a traditional written format. This primarily requires candidates to demonstrate their ability to recall factual information related to patient care and treatment.

Objective Structured Clinical Examination (OSCE) ${ }^{28}$ differ from the USMLE in that they are dialogue-driven, and are often used in health sciences education, including medicine, nursing, pharmacy, and physical therapy. OSCEs are designed to test performance in a simulated clinical setting and competence in skills such as communication, clinical examination, medical procedures, and time management. The OSCE is structured around a circuit of stations, each of which focuses on a specific aspect of clinical practice. Examiners rotate through these stations, encountering standardized patients (actors trained to present specific medical conditions and symptoms) or mannequins that simulate clinical scenarios, where they must demonstrate their practical abilities and decision-making processes.

Each station has a specific task and a checklist or a global rating score that observers use to evaluate the students' performance. The OSCE has several advantages over traditional clinical examinations. It allows for direct observation of clinical skills, rather than relying solely on written exams to assess clinical competence. This hands-on approach to testing helps bridge the gap between theoretical knowledge and practical ability. Additionally, by covering a broad range of skills and scenarios, the OSCE ensures a comprehensive assessment of a student's readiness for clinical practice.

## The evaluation of language models in medicine

While there exists different types of exams to evaluate medical students, LLMs are typically only evaluated using medical knowledge benchmarks (like the USMLE step exams). Briefly,
we discuss the way in which these evaluations are executed using the most common benchmark, MedQA, as an example.

The MedQA ${ }^{29}$ dataset comprises a collection of medical question-answering pairs, sourced from Medical Licensing Exam from the US, Mainland China, and Taiwan. This dataset includes 4-5 multiple-choice questions, each accompanied by one correct answer, alongside explanations or references supporting the correct choice. The LLM is provided with all of the context for the question, such as the patient history, demographic, and symptoms, and must provide a response to the question. These questions range from provided diagnoses to choosing treatments and are often quite challenging even for medical students. While these problems also proved quite challenging for LLMs at first, starting with an accuracy of $38.1 \%$ in September $2021^{2}$, progress was quickly made toward achieving above human performance, with $90.2 \%$ in November $2023^{3}$ (human passing score is $60 \%$, human expert score is $87 \% \%^{4}$ ).

Beyond the MedQA dataset, many other knowledge-based benchmarks have been proposed, such as PubMedQA ${ }^{30}$, MedM$\mathrm{CQA}^{31}$, MMLU clinical topics ${ }^{32}$, and MultiMedQA ${ }^{7}$, which follow a similar multiple-choice format. Other works have made modifications to medical exam question datasets, such as those which incorporate cognitive biases ${ }^{17}$ and with multiple choice questions removed ${ }^{33}$. The work of ref. ${ }^{17}$ shows that the introduction of a simple bias prompt can lead to large reductions in accuracy on the MedQA dataset and that this effect can be partially mitigated using various prompting techniques, such as one-shot or few-shot learning.

## Beyond exam questions

Recent work toward red teaming LLMs in a medical context has shown that a large proportion of responses from models like GPT-3.5, GPT-4, and GPT-4 with internet-lookup are inappropriate, highlighting the need for refinement in their application in healthcare ${ }^{34}$. This was accomplished through the effort of medical and technical professionals stress-testing LLMs on clinically relevant scenarios. Similar work designed a new benchmark, EquityMedQA, using new methods for surfacing health equity harms and biases ${ }^{35}$. This work demonstrates the importance of using diverse assessment methods and involving raters of varying backgrounds and expertise for understanding bias in LLM evaluations.

Previous work has made progress in the direction of clinical decision making using simulations of patients and doctors, aiming to develop AI that can diagnose through conversation. This model, titled AMIE (Articulate Medical Intelligence Explorer $)^{26}$, demonstrates improved diagnostic accuracy and performance on 28 of the 32 proposed axes from the perspective of specialist physicians and 24 of 26 axes from the perspective of patient actors. While these results are exciting for medical AI, this work remains closed-source and is not accessible for reproducibility or further studies. Additionally, this work focused only on diagnosing patients through history-taking, and did not include the ability to make deci- sions about which tests needed to be performed and was not configurable for multimodal clinical settings such as those with medical images or charts. Similar to AIME, the CRAFTMD benchmark ${ }^{36}$ proposes evaluating LLMs through natural dialogues on dermatology questions, however without the use of images. Additionally, neither of these works demonstrate performance in the presence of bias, with multimodal input, or using a measurement agent. There has also been work which shows simulated doctor agents can improve medical QA performance through turn-based dialogue, where various medical specialist agents converse ${ }^{37}$.

## Discussion

In this work, we present AgentClinic: a multimodal agent benchmark for simulating clinical environments. We design 15 multimodal language agents which require an understanding of both language and images in order to arrive at a diagnosis and present results from two multimodal language models. We also design 107 unique language agents which are based on cases from the USMLE, including an measurement agent which is able to provide medical test readings. We instructed these agents to exhibit 23 different biases, with either the doctor or patient presenting bias. We show the accuracy of four LLMs on AgentClinic-MedQA, as well as the accuracy of GPT-4, the highest performing model, on each of the different biases. We find that patient and doctor cognitive biases effect performance showing a $1.7 \%-2 \%$ reduction in accuracy. However, implicit biases have a much larger effect on the doctor agent with a $1.5 \%$ reduction compared to $0.7 \%$ for the patient agent. We also find that doctor and patient biases can reduce diagnostic accuracy, and that the patient has a lower willingness to follow up with treatment, reduced confidence in their doctor, and lower willingness to have a follow-up consultation in the presence of bias.

We find that in addition to the doctor language model, the patient language model also has an effect on diagnostic accuracy, with same-model cross communication leading to higher accuracy than between-model communication. We also show that having limited interaction time reduces diagnostic accuracy and having too much interaction time also reduces accuracy. We show that reducing the amount of time a doctor has to interact with the patient ( $\mathrm{N}<20$ inferences) can lead to an $27 \%$ reduction in accuracy when $\mathrm{N}=10$ and $14 \%$ when $\mathrm{N}=15$, and also increasing the amount of time $(\mathrm{N}>20)$ reduces the accuracy by $4 \%$ when $\mathrm{N}=25$ and $9 \%$ when $\mathrm{N}=30$. Finally, we show GPT-4V is able to get around $27 \%$ accuracy on a multimodal simulated clinical environment based on NEJM case challenges.

One limitation for the evaluations presented in this benchmark is that it is currently unknown what data was used to train GPT-4, GPT-4o, and GPT-3.5. While previous works have cited GPT-4s accuracy as a valid measure ${ }^{3,13,17}$, it is entirely possible that GPT-4/4o/3.5 could have been trained on the MedQA test set giving it an unfair advantage on the task. Currently, Mixtral-8x7B ${ }^{38}$, Llama 3 70B-instruct, and

Llama 2-70B-Chat ${ }^{39}$ do not report training on the MedQA test or train set. Additionally, our results on varying the patient LLM suggest that their may be an advantage for LLMs which act as both the patient and the doctor agent, because LLMs are able to recognize their own text with high accuracy, and display disproportionate preference to that text ${ }^{24}$.

Our work only presents a simplified clinical environments that include agents representing a patient, doctor, measurements, and a moderator. However, in future work we will consider including additional critical actors such as nurses, the relatives of patients, administrators, and insurance contacts. There may be additional advantages to creating agents that are embodied in a simulated world like in ref. ${ }^{40,41}$, so that physical constraints can be considered, such as making decisions with limited hospital space.

Focusing on improving the realism of the patient-doctor interaction simulations by grounding the agents with real dialogue could provide an increased reflection of real clinical settings, using datasets such as MedDialogue ${ }^{42}$ or from actual OSCEs $^{43}$. A broader range of medical conditions could be incorporated, increasing the reliability of the benchmark metric with rare diseases and across various medical specialties. Further refinement of the measurement agent could introduce a wider variety of medical tests and modalities (e.g. sound or full patient observation). There could also be incorporated a "cost" associated with running particular tests, and the decisions that doctor agents make with limited resources and time could further increase the realism of this work. Particular doctor agents could be optimized for certain medical settings (high-resource vs low-resource hospitals).

Linking simulated agent benchmarks to real-world patient datasets, e.g., by means of using the former to study the latter will be an exciting route for future work. It will be exciting to further decipher to which degree "aligned" LLMs comply with bias instructions to augment current red-teaming efforts with agent-based simulations. Furthermore, we envision exploring biases beyond those traditionally recognized in medical practice, to include biases related to healthcare system factors and patient-doctor communication styles ${ }^{20}$. The goal would be to develop mitigation strategies, as has been shown in prior work ${ }^{17}$, which can be integrated into the language models to reduce the impact of these biases on diagnostic accuracy.

While the primary aim of the benchmark is to develop more sophisticated decision making models, each of the different language agents (patient, measurement, moderator, doctor) that we present are able to be modified in our open-source code. This allows for further studies to be performed on different components of the system, and perhaps even to further complicate the workflow, such as adding additional patients or doctors, or providing inaccuracies to the test results. We additionally provide a simple workflow for adding custom scenarios through our examination template, as well as the ability to design completely new templates and new agents.

Overall, we believe that language models need to be critically examined with novel evaluation strategies that go well beyond static question-answering benchmarks. With this work, we take a step towards building more interactive, operationalized, and dialogue-driven benchmarks that scrutinize the sequential decision making ability of language agents in various challening and multimodal clinical settings.

## Acknowledgements

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE 2139757, awarded to SS and CH.

## References

1. Thirunavukarasu, A. J. et al. Large language models in medicine. Nat. medicine 1-11 (2023).
2. Gu, Y. et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Comput. for Healthc. (HEALTH) 3, 1-23 (2021).
3. Nori, H. et al. Can generalist foundation models outcompete special-purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452 (2023).
4. Liévin, V., Hother, C. E., Motzfeldt, A. G. \& Winther, O. Can large language models reason about medical questions? Patterns (2023).
5. Organization, W. H. et al. Health workforce requirements for universal health coverage and the sustainable development goals. World Heal. Organ. (2016).
6. McIntyre, D. \& Chow, C. K. Waiting time as an indicator for health services under strain: a narrative review. $I N$ QUIRY: The J. Heal. Care Organ. Provision, Financing 57, 0046958020910305 (2020).
7. Singhal, K. et al. Large language models encode clinical knowledge. Nature 620, 172-180 (2023).
8. Vaid, A., Landi, I., Nadkarni, G. \& Nabeel, I. Using fine-tuned large language models to parse clinical notes in musculoskeletal pain disorders. The Lancet Digit. Heal. 5, e855-e858 (2023).
9. Zakka, C. et al. Almanac-retrieval-augmented language models for clinical medicine. NEJM AI 1, AIoa2300068 (2024).
10. Xiong, G., Jin, Q., Lu, Z. \& Zhang, A. Benchmarking retrieval-augmented generation for medicine. arXiv preprint arXiv:2402.13178 (2024).
11. Liévin, V., Hother, C. E. \& Winther, O. Can large language models reason about medical questions? arXiv preprint arXiv:2207.08143 (2022).
12. Wu, C. et al. Pmc-llama: Towards building open-source language models for medicine (2023). 2304.14454.
13. Chen, Z. et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079 (2023).
14. Ely, J. W. et al. Analysis of questions asked by family doctors regarding patient care. Bmj 319, 358-361 (1999).
15. Omiye, J. A., Lester, J. C., Spichak, S., Rotemberg, V. \& Daneshjou, R. Large language models propagate racebased medicine. NPJ Digit. Medicine 6, 195 (2023).
16. Ziaei, R. \& Schmidgall, S. Language models are susceptible to incorrect patient self-diagnosis in medical applications. In Deep Generative Models for Health Workshop NeurIPS 2023 (2023).
17. Schmidgall, S. et al. Addressing cognitive bias in medical language models. arXiv preprint arXiv:2402.08113 (2024).
18. Blumenthal-Barby, J. S. \& Krieger, H. Cognitive biases and heuristics in medical decision making: a critical review using a systematic search strategy. Med. Decis. Mak. 35, 539-557 (2015).
19. Hammond, M. E. H., Stehlik, J., Drakos, S. G. \& Kfoury, A. G. Bias in medicine: lessons learned and mitigation strategies. Basic to Transl. Sci. 6, 78-85 (2021).
20. FitzGerald, C. \& Hurst, S. Implicit bias in healthcare professionals: a systematic review. BMC medical ethics 18, 1-18 (2017).
21. Gopal, D. P., Chetty, U., O'Donnell, P., Gajria, C. \& Blackadder-Weinstein, J. Implicit bias in healthcare: clinical practice, research and decision making. Futur. healthcare journal 8, 40 (2021).
22. Sabin, J. A. Tackling implicit bias in health care. New Engl. J. Medicine 387, 105-107 (2022).
23. Lombardi, C. V., Chidiac, N. T., Record, B. C. \& Laukka, J. J. Usmle step 1 and step 2 ck as indicators of resident performance. BMC Med. Educ. 23, 543 (2023).
24. Panickssery, A., Bowman, S. R. \& Feng, S. Llm evaluators recognize and favor their own generations (2024). 2404.13076 .
25. Smith-Bindman, R. et al. Use of diagnostic imaging studies and associated radiation exposure for patients enrolled in large integrated health care systems, 19962010. Jama 307, 2400-2409 (2012).
26. Tu, T. et al. Towards conversational diagnostic ai. arXiv preprint arXiv:2401.05654 (2024).
27. Melnick, D. E., Dillon, G. F. \& Swanson, D. B. Medical licensing examinations in the united states. J. dental education 66, 595-599 (2002).
28. Zayyan, M. Objective structured clinical examination: the assessment of choice. Oman medical journal 26, 219 (2011).
29. Jin, D. et al. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Appl. Sci. 11, 6421 (2021).
30. Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W. \& Lu, X. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146 (2019).
31. Pal, A., Umapathi, L. K. \& Sankarasubbu, M. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, 248-260 (PMLR, 2022).
32. Hendrycks, D. et al. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2020).
33. Gramopadhye, O. et al. Few shot chain-of-thought driven reasoning to prompt llms for open ended medical question answering. arXiv preprint arXiv:2403.04890 (2024).
34. Chang, C. T.-T. et al. Red teaming large language models in medicine: Real-world insights on model behavior. medRxiv 2024-04 (2024).
35. Pfohl, S. R. et al. A toolbox for surfacing health equity harms and biases in large language models. arXiv preprint arXiv:2403.12025 (2024).
36. Johri, S. et al. Guidelines for rigorous evaluation of clinical llms for conversational reasoning. medRxiv 202309 (2023).
37. Tang, X. et al. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint arXiv:2311.10537 (2023).
38. Jiang, A. Q. et al. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).
39. Touvron, H. et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
40. Park, J. S. et al. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, 1-22 (2023).
41. Li, J. et al. Agent hospital: A simulacrum of hospital with evolvable medical agents. arXiv preprint arXiv:2405.02957 (2024).
42. Chen, S. et al. Meddialog: a large-scale medical dialogue dataset. arXiv preprint arXiv:2004.03329 (2020).
43. Fareez, F. et al. A dataset of simulated patient-physician medical interviews with a focus on respiratory cases. Sci. Data 9, 313 (2022).
44. OpenAI et al. Gpt-4 technical report (2023). 2303.08774.
45. Brown, T. et al. Language models are few-shot learners. Adv. neural information processing systems 33, 18771901 (2020).
46. Christiano, P. F. et al. Deep reinforcement learning from human preferences. Adv. neural information processing systems 30 (2017).
</end of paper 2>


<paper 3>
# Megaverse : Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks 

Sanchit Ahuja Divyanshu Aggarwal Varun Gumma Ishaan Watts<br>Ashutosh Sathe Millicent Ochieng Rishav Hada Prachi Jain<br>Mohamed Ahmed Kalika Bali Sunayana Sitaram<br>Microsoft Corporation<br>$\{t-s a h u j a$, sunayana.sitaram\}@microsoft.com


#### Abstract

There has been a surge in LLM evaluation research to understand LLM capabilities and limitations. However, much of this research has been confined to English, leaving LLM building and evaluation for non-English languages relatively unexplored. Several new LLMs have been introduced recently, necessitating their evaluation on non-English languages. This study aims to perform a thorough evaluation of the non-English capabilities of SoTA LLMs (GPT-3.5-Turbo, GPT-4, PaLM2, Gemini-Pro, Mistral, Llama2, and Gemma) by comparing them on the same set of multilingual datasets. Our benchmark comprises 22 datasets covering 83 languages, including low-resource African languages. We also include two multimodal datasets in the benchmark and compare the performance of LLaVA models, GPT-4-Vision and Gemini-Pro-Vision. Our experiments show that larger models such as GPT-4, Gemini-Pro and PaLM2 outperform smaller models on various tasks, notably on low-resource languages, with GPT-4 outperforming PaLM2 and Gemini-Pro on more datasets. We also perform a study on data contamination and find that several models are likely to be contaminated with multilingual evaluation benchmarks, necessitating approaches to detect and handle contamination while assessing the multilingual performance of LLMs.


## 1 Introduction

Large Language Models (LLMs) have surpassed the performance of previous generation of language models on several tasks and benchmarks, sometimes even approaching or exceeding human performance (Hubert et al., 2024). However, the root cause of the observed capabilities in these models is not always apparent, whether stemming from augmented model capabilities or other factors like contamination in test datasets and the absence of datasets that genuinely measure the capabilities of these models (Balloccu et al., 2024). Thus, evaluation of Large Language Models has become an important field of study.

Most of the work on evaluating LLMs via benchmarking (Liang et al., 2022), qualitative tests for specific capabilities (Bubeck et al., 2023) or human evaluation have focused solely on English. However, studies have shown that there is a large gap between the capabilities of LLMs in English and other languages (Choudhury et al., 2023). Evaluation of LLMs in languages other than English is challenging due to a variety of factors, including the lack of benchmarks covering a large number of languages from diverse language families and the lack of multilingual benchmarks covering tasks such as reasoning, chat, and dialogue. Therefore, it is crucial to prioritize multilingual evaluation to enhance the development of more effective multilingual models. Neglecting this critical aspect may result in a significant population being left behind and may widen the digital divide (Joshi et al., 2021).

Our prior work on evaluating multilingual capabilities of LLMs, MEGA (Ahuja et al., 2023), yielded the following observations: GPT-4 (OpenAI, 2023a) comes close to the performance of SOTA fine-tuned language models such as TULRv6 (Patra et al., 2023). GPT models perform worse on languages that are written in nonLatin scripts, and on low-resource languages. Other LLMs such as BLOOMZ (Muennighoff et al., 2023) usually perform worse than GPT-4. However, several newer models are comparable to GPT-4 in performance on English, and it is essential to study their multilingual performance as well. Moreover, there is a rising interest in Large Multimodal Models (LMMs), and the convergence of multimodal and multilingual LLMs remains an understudied area (Hu et al., 2024). Our contributions are as follows:

- We build on top of the Mega benchmark and add 6 new datasets, thus extending coverage to 22 datasets and 83 languages including many low-resource African languages.
- We benchmark nine new SOTA text LLMs PaLM2 (Google, 2023), Llama2 (3 variants) (Touvron et al., 2023), Mistral-v1.0 (2 variants), (Jiang et al., 2023), Gemma (2 variants) (Mesnard et al., 2024), Gemini 1.0 pro (Anil et al., 2023a) in addition to GPT-4 and GPT3.5-Turbo.
- We benchmark the multimodal LLaVA family models (Liu et al., 2023), GPT-4-Vision (OpenAI, 2023b) and Gemini-Pro-Vision (Anil et al., 2023a) on two multilingual multimodal datasets.
- We present a thorough contamination study of both commercial and open-source set of LLMs on a subset of our datasets.
- We study the overall trends in our experiments by studying the deviation of performance across language families and tasks, and provide directions for future research.


## 2 Related work

Evaluation of LLMs Recently, there has been an increasing interest in evaluating LLMs on a wide range of capabilities, given the surge in their popularity and effectiveness. BIG-Bench (Srivastava et al., 2023) consists of 204 tasks to evaluate LLMs.

While BIG-Bench includes tasks in non-English languages as well, they are largely related to translation. Liang et al. (2022) proposed HELM, defining a taxonomy of scenarios and metrics that define the space of LLM evaluation, and evaluating 30 language models on 42 scenarios and 7 metrics. However, all the scenarios are focused on datasets in standard English or dialects, and they highlight coverage of languages as an important area for improvement. Bubeck et al. (2023), has pointed out the limitations of using standard NLP benchmarks to evaluate generative models, due to the pace at which these benchmarks become saturated. There are also concerns about benchmark contamination in LLM evaluation. Zhou et al. (2023) show that test dataset contamination in training and finetuning data leads to a significant impact on LLM performance.
Multilingual Benchmarks and Evaluation Bang et al. (2023) evaluates the multilingual capabilities of ChatGPT and shows that it fails to generalize to low-resource languages with non-Latin scripts. However, multilingual evaluation is performed only on a few tasks, and a subset of 50-100 examples are used for testing the model. Hendy et al. (2023) evaluate the translation abilities of GPT-3.5 models and find that these models perform well in translating high-resource languages, but their capabilities for low-resource languages are limited. BUFFET (Asai et al., 2023) covering 54 languages across 15 datasets and Lai et al. (2023) covering 37 languages across 7 datasets also perform multilingual benchmarking of LLMs such as ChatGPT and BLOOMZ. Yang et al. (2023) does a comprehensive study of GPT4-Vision's capabilities that include analyzing its performance on multilingual image description, scene text recognition, and translation. Our work builds on the MEGA benchmarking effort (Ahuja et al., 2023), which evaluates GPT models across 16 datasets. We extend the MEGa benchmark to more tasks including multimodal tasks, evaluate several SoTA LLMs, and perform a more comprehensive analysis of contamination.

Contamination Several techniques have been proposed to study the contamination of publicly available evaluation datasets. Ahuja et al. (2023) study contamination by prompting the models to fill dataset cards. Other methodologies encompass Golchin and Surdeanu (2023b), which does not provide quantification of contamination, and Oren et al. (2023), which requires access to log probabilities, thereby limiting their studies to open-sourced LLMs.

## 3 Experimental Setup

### 3.1 Datasets

We perform experiments on the 16 datasets that are part of the MegA suite - XNLI (Conneau et al., 2018), IndicXNLI (Aggarwal et al., 2022), GLUECoS NLI (Khanuja et al., 2020a), PAWS-X (Yang et al., 2019), XCOPA (Ponti et al., 2020), XStoryCloze (Lin et al., 2022), GLUECoS Sentiment Analysis (En-Es-CS) (Vilares et al., 2016), TyDiQA-GoldP (Clark et al., 2020), MLQA (Lewis et al., 2020), XQUAD (Artetxe et al., 2020), IndicQA (Doddapaneni et al., 2023), PAN-X (Pan et al., 2017), UDPOS (Nivre et al., 2018), Jigsaw (Kivlichan et al., 2020), WinoMT (Stanovsky et al.,

![](https://cdn.mathpix.com/cropped/2024_06_04_b9a323c8bb7b325a4c6bg-03.jpg?height=414&width=1568&top_left_y=273&top_left_x=244)

Figure 1: Hierarchy of Models and Tasks spread across MEGAVERSE

2019) and XLSum (Hasan et al., 2021). These datasets include a mix of classification, Question Answering, Sequence Labeling, and Natural Language Generation datasets, along with two datasets covering the Responsible AI tasks of toxicity detection and gender bias. The datasets we include also contain a mix of translated datasets verified by native speakers, as well as datasets created independently for each language. Figure 1 shows a hierarchy of models and tasks spread across MEGAVERSE. For a more detailed description of the datasets included in the original MEGA benchmark, we refer the readers to Ahuja et al. (2023). We describe the six datasets added to our study below.

### 3.1.1 AfriQA

AfriQA (Ogundepo et al., 2023) is a QA dataset that does not have a context passage. It covers 10 African languages - Bemba, Fon, Hausa, Igbo, Kinyarwanda, Swahili, Twi, Wolof, and Yorùbá. We use the few-shot size of $k=4$ and the monolingual prompting strategy to perform experiments only on the GPT and Llama models, as the PaLM2 model only supports Swahili.

### 3.1.2 Belebele

Belebele (Bandarkar et al., 2023) is a multiple choice machine reading comprehension (MRC) dataset parallel across 122 languages. Each question is linked to a short passage from the FLORES200 dataset (Costa-jussà et al., 2022). The human annotation procedure was carefully curated to create questions that discriminate between different levels of language comprehension. We evaluated Arabic, Czech, Danish, German, English, Spanish, Finnish, French, Hebrew, Hungarian, Italian, Japanese, Korean, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Thai, Turkish, Chinese Simplified and Chinese Traditional. Results for
Llama2 and GPT-3.5-Turbo are reported from the dataset paper. We perform zero-shot monolingual prompting for our experiments, as this dataset does not have a dev set.

### 3.1.3 IN22

IN22 (Gala et al., 2023) is a translation benchmark for all 22 scheduled Indic languages. IN22-Gen is a general-purpose multi-domain evaluation subset of IN22 which has been curated from two sources: Wikipedia and Web Sources offering diverse content spanning news, entertainment, culture, legal, and India-centric topics. IN22-Conv is the conversation domain subset of IN22. Due to resource constraints, we evaluate 14 languages: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Kashmiri, Malayalam, Marathi, Nepali, Odia, Punjabi, Tamil, Telugu, and Urdu.

### 3.1.4 MaRVL

MaRVL (Multicultural Reasoning over Vision and Language) (Liu et al., 2021) is a dataset of images and associated captions. The concepts and images collected were entirely driven by native speakers and are representative of various cultures across the globe and span 5 languages, i.e., Indonesian, Chinese, Swahili, Tamil, and Turkish. Each instance in the dataset consists of a pair of images (left image and right image) and a statement, and the task is to determine whether the statement is consistent for the given pair of images.

### 3.1.5 XM-3600

CrossModal-3600 (Thapliyal et al., 2022) is a multilingual image captioning dataset consisting of 3600 geographically diverse images directly captioned in 36 different languages, avoiding any inconsistencies due to translations. We experimented on 20 out of 36 languages due to resource constraints:

Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, and Turkish.

### 3.1.6 XRiSAWOZ

XRiSAWOZ (Moradshahi et al., 2023) is a taskoriented dialogue modeling dataset. The dataset is a multilingual (English, Hindi, French, Korean) translation of the Chinese-only RiSAWOZ dataset (Quan et al., 2020). XRiSAWOZ also includes an English-Hindi code mixed setting. For each conversation, the agent must make use of structured knowledge from the databases to answer user queries. The task consists of 4 subtasks: "Dialogue State Tracking" (DST), "API Call Detection" (API), "Dialogue Act Generation" (DA) and "Response Generation" (RG). The metrics used for evaluation include BLEU, Slot Error Rate (SER) (factual correctness of generated response) (Wen et al., 2015), (averaged/task) success rate (Lin et al., 2021), API call accuracy, dialogue act accuracy and joint goal accuracy (Budzianowski et al., 2018). We refer the reader to Moradshahi et al. (2023) for detailed descriptions of subtasks and metrics. We perform experiments on $10 \%$ of the data i.e. about 400 dialogue turns across 3 domains due to limited compute.

### 3.2 Models ${ }^{1}$

Below is a list of all the models we evaluate:

- GPT-3.5-Turbo (Ouyang et al., 2022)
- GPT-4 (OpenAI, 2023a)
- GPT-4-Vision (OpenAI, 2023b)
- Llama2 (7B, 13B, 70B) (Touvron et al., 2023)
- PaLM2 (Anil et al., 2023b)
- Gemini-Pro (Anil et al., 2023a)
- Gemini-Pro-Vision (Anil et al., 2023a)
- Gemma (2B, 7B) (Mesnard et al., 2024)
- Mistral (Jiang et al., 2023)
- BakLLaVA-v1 (Liu et al., 2023)
- ViP-LLaVA (13B) (Cai et al., 2023)
- LLaVA-1.5 (13B) (Liu et al., 2023)


### 3.3 Prompting strategies

Ahuja et al. (2023) explore three prompting variations based on the language of the few-shot and[^0]

test examples, and find that monolingual prompting, featuring few-shot examples in the target language, outperforms zero-shot cross-lingual prompting in English for most datasets. Translate-test excels over monolingual for certain low-resource languages but with minimal gaps for models like GPT4. Therefore, we default to monolingual prompting unless otherwise specified. Zero-shot cross-lingual prompting (zs-cl) is used when dev datasets are unavailable in the target language. English instructions are maintained for prompts, proven to outperform instructions in the target language (Ahuja et al., 2023). Prompt templates for our new datasets are in the Appendix A.2.

### 3.3.1 XRiSAWOZ

Moradshahi et al. (2023) presents results in both end-to-end and turn-by-turn evaluation settings. We perform end-to-end evaluation with regex based careful filtering of the generated responses for DST/API/DA tasks after every turn. This is required to ensure correctness of the syntax in the state descriptions for these tasks. No such postprocessing is done for the RG task. For inferring a subtask on a dialogue turn, we provide in-context examples corresponding to the same turn from other domains. If for a particular turn, sufficient incontext examples are not available, we look for the latest previous turn for which sufficient in-context examples are available. E.g. Assume the following turn to count distribution and $k=4$ (number of in-context examples). Turns 1-4: more than 10 examples, Turn 5: 3 examples, and Turn 6 has 1 example.

At turns 5 and 6, we do not have sufficient examples from turn 5 or 6 . Therefore, we sample in-context examples from turn 4 for both of them. Our prompts for each subtasks can be seen in Fig. $9,10,11,12,13$.

## 4 Results

### 4.1 XNLI

All models perform best on English, with slightly lower performance on Greek and German, and lower performance on languages like Hindi, Thai, Urdu, and Swahili. Overall PaLM2 performs best, closely followed by GPT-4. GPT-3.5-Turbo is worse on all languages, however, we find that all three Llama models perform substantially worse, with Mistral performing the worst. Since XNLI is a popular dataset, dataset contamination cannot be
ruled out. (Figure 18, Table 2).

### 4.2 IndicXNLI

We performed experiments on IndicXNLI on the GPT models, Mistral as well as Llama models, however, the Llama models gave scores of 0 for all languages, which is why we do not plot them. The Mistral model also performs poorly. We find that GPT-4 outperforms GPT-3.5-Turbo on all languages with the highest scores on Hindi, Punjabi, and Bengali. However, the overall accuracy is not very high on any language compared to the XNLI results seen earlier, and fine-tuned baselines such as MuRIL perform best. (Figure 19, Table 3).

### 4.3 GLUECoS NLI

All models do well on this NLI task, with GPT-4 performing best. (Figure 26, Table 14).

### 4.4 PAWS-X

PaLM2 outperforms the GPT models on all languages and all models perform well, which could be because this dataset contains high-resource languages. However, dataset contamination cannot be ruled out, as shown in Ahuja et al. (2023). The performance on English performs is the best, followed closely by Latin script languages, and a drop in performance for languages in other scripts. The Llama and Mistral models perform worse than the GPT models and PaLM2, although the difference in performance is not as large as in some of the other datasets. (Figure 20, Table 4).

### 4.5 XCOPA

The performance of GPT-4, Gemma, Gemini and PaLM2 are comparable, with GPT-4 having the best perforamnce. Notably, they are all better than GPT-3.5-Turbo, which performs substantially better than the Llama2 and Mistral models except in Quechua, for which no model performs well. However, the results on all other languages for GPT-4 and PaLM2 are extremely high, which may be due to dataset contamination. (Figure 21, Table 5).

### 4.6 XStoryCloze

Since the Llama models gave scores of 0 for all languages, we omit it from our analysis. We find that the gap between the GPT models and PaLM2 is very high, with both GPT models performing extremely well. For all languages except Telugu, Basque and Burmese Gemini-pro performs well. The contamination study from Ahuja et al. (2023) show a low chance of dataset contamination for GPT-4, which indicates that the GPT models can perform this task well. (Figure 22, Table 13).

### 4.7 Sentiment Analysis (En-Es-CS)

Surprisingly, GPT-3.5-Turbo outperforms both GPT-4 and PaLM2 on this task, with the mBERT baseline performing the best, while Gemini-pro performs the worst by a large margin. (Figure 26, Table 14).

### 4.8 TyDiQA GoldP

The TuLR model performs best, followed by GPT4, PaLM2, Gemini-Pro, and BLOOMZ, while Llama models perform poorly, with Mistral being slightly better. Smaller models, in particular, demonstrate a significant performance gap between English and all other languages. However, dataset contamination cannot be ruled out, as shown in Ahuja et al. (2023). (Figure 23, Table 7).

### 4.9 MLQA

TULR and GPT-4 outperform all other models for this dataset except for German. English exhibits superior performance, with Spanish (es), German (de), and Vietnamese (vi) following closely. The most significant gaps are noted between English and Arabic (ar), Hindi (hi), and Chinese (zh) The Llama2-13B model performs well for some languages, such as Arabic, German, and Spanish but performs poorly on Chinese Hindi, and Vietnamese, but is still better than Mistral and Gemma. This is one of the datasets where PaLM2 struggles, particularly for Arabic and Chinese. Dataset contamination in GPT-4 cannot be ruled out, as shown in Ahuja et al. (2023). Smaller versions of the Llama model outperform the Llama 70B model across all languages. (Figure 24, Table 8).

### 4.10 XQUAD

TuLRv6 performs best across almost all languages in the XQuAD dataset, followed by GPT-4, PaLM 2, Gemini-Pro, and BLOOMZ. BLOOMZ's performance declines significantly in Greek and Thai as shown in Figure 2. PaLM2 and Gemini-Pro exhibit competitive performance, closely trailing GPT-4$32 \mathrm{~K}$ and TuLRv6 - XXL across languages from high to mid-resource tiers. All three Llama models perform poorly on this dataset. Gemma and Mistral perform slightly better than Llama on all languages but lags behind the larger models and finetuned models. Dataset contamination in GPT-4 cannot be
ruled out, as shown in Ahuja et al. (2023). (Figure 2, Table 6).

### 4.11 IndicQA

Since the Llama models gave scores of 0 for all languages, we omit it from our analysis. We use the zero-shot cross-lingual prompting strategy due to the absence of a dev set. GPT-4 performs better than GPT-3.5-Turbo, with the best performance seen for Hindi, Marathi, and Bengali, while the smaller models like Gemma perform poorly. (Figure 25, Table 9).

### 4.12 PAN-X

GPT-4 and GPT-3.5-Turbo outperform PaLM2 and gemini-pro for most languages. However, all models perform poorly on Thai, Japanese, and Chinese on this sequence labeling task. Since this is an older dataset, GPT-4 data contamination cannot be ruled out as shown in Ahuja et al. (2023). (Figure 31, Table 12).

### 4.13 UDPOS

PaLM2 performs the best followed by GPT-4, GPT3.5-Turbo and Gemini-pro being the worst on average. All models show similar high performance across languages, except for Arabic, Greek, Hebrew, Hindi, and Vietnamese, where PaLM2 performs best. GPT-4 data contamination cannot be ruled out as shown in Ahuja et al. (2023). (Figure 33, Table 11).

### 4.14 Jigsaw

We perform experiments on the Jigsaw dataset for GPT-3.5-Turbo and PaLM2 using the monolingual prompting strategy and find that both models perform very well on all languages. Since the dataset cannot be accessed without download, models are less likely to be contaminated with this dataset. (Figure 30, Table 19).

### 4.15 WinoMT

We perform experiments on the WinoMT dataset only for GPT-3.5-Turbo using the monolingual prompting strategy and report the results for completeness. We find that the model does not perform well on any of the languages. (Figure 29, Table 20).

### 4.16 XLSum

GPT-4 outperforms all other models, with some exceptions. GPT-3.5-Turbo performs best for African languages like Swahili, Somali, and Yoruba, while the Llama models perform best for Arabic, Kyrgyz, Vietnamese, and Welsh. According to the contamination analysis in Ahuja et al. (2023), it is possible, though less likely that GPT-4 is contaminated with this dataset. (Figure 34, Table 15).

### 4.17 Belebele

Gemini-Pro has the best performance amongst all the models for most languages, while for smaller models only Llama models come close. GPT-4 and PaLM2 outperform GPT-3.5-Turbo, Llama2, and Mistral, which performs worst. Most models do well due to the multiple-choice question-answering nature of the task, which makes parsing outputs and evaluation simpler and increases the probability of success even for weaker models. (Figure 16, Table 17).

### 4.18 AfriQA

GPT-4 has best performance, while the Llama2 and Mistral models perform very poorly on all languages. (Figure 15, Table 10).

### 4.19 IN22

We report our results on the IN22-Gen and IN22Conv subsets (Figure 35) where we randomly select $k=8$ translation pairs from the development set of FLORES-200 (Costa-jussà et al., 2022) as incontext examples. We also report GPT-3.5-Turbo 0 -shot and IndicTrans2 scores from Gala et al. (2023) for comparison. For consistency, we use the indic_nlp_library ${ }^{2}$ and the evaluation scripts ${ }^{3}$ from Gala et al. (2023) to tokenize the predictions and references before computing chrF++ (Popović, 2017) for Indic languages. We do not evaluate PaLM2 on this dataset, as most languages in this dataset are not supported by it.

Llama2 and Mistral perform poorly on all Indic languages in the En-Indic direction, whereas the performance is better on the Indic-En direction. Gemma-7B performs significantly better than both Llama2 and Mistral in both directions and on all languages. GPT-4 performs the best among all LLM models considered. All LLMs perform better in the Indic-En direction and Conversational dataset since they are finetuned with chat or conversational style data. We compare results to IndicTrans2 Gala et al. (2023) and find that it fares[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_b9a323c8bb7b325a4c6bg-07.jpg?height=360&width=1585&top_left_y=231&top_left_x=244)

Figure 2: Results for XQUAD across all languages and models for zero-shot cross-lingual prompting

significantly better than LLMs. (Figure 35, Tables 21 - 24).

### 4.20 XRiSAWOZ

We compare DA accuracy of various models in Figure 17. Table 25 shows the comparison with fine-tuned models as well. We find that GPT-4's performance on DA accuracy is the closest and comparable to fine-tuned baselines for the task. Poorer scores on other models seem to correlate with the model's hallucination tendencies.

We compare results on all 6 metrics in Table 26 to better understand model behavior. We find that PaLM2,GPT-4 and Gemini-pro generate very concise responses leading to consistently higher BLEU scores as compared to other models. On all other metrics, GPT family of models significantly outperforms both PaLM/Gemini and open-source models. Notably, all the proprietary models achieve less than $10 \%$ SER on Chinese hinting contamination of RiSAWOZ (the original Chinese-only dataset). Open source models often hallucinated non-existent entities in their responses while proprietary models did not show this tendency.

In the code-mixed English-Hindi setting, the performance is worse than both English and Hindi on average across most metrics for all models. (Figure 17, Tables 25,26 ). This could indicate challenges in understanding as well as generating effective code mixed text for all models.

### 4.21 MaRVL

We evaluate LLaVA models, GPT-4-Vision ${ }^{4}$, and Gemini-Pro-Vision on the multimodal datasets with monolingual and translate-test prompting (Figure 27). The Azure BING translate module was utilized for translating the sentences into English. We find that accuracy scores border on random classification LLaVA models, with the lowest score on[^2]

Tamil and Chinese. The translate-test strategy is comparable to monolingual. However, the performance is still the same as a random classification. GPT-4-Vision is significantly better than LLaVA, and the gains due to translate-test are only visible on Turkish. Gemini-Pro-Vision performs slightly better than random, and the translate-test is preferable except in the case of Chinese. (Figure 27, Table 16).

### 4.22 XM-3600

We test the LLaVA models, GPT-4-Vision ${ }^{5}$, and Gemini-Pro-Vision models on the XM-3600 image captioning dataset and use the chrF metric (Popović, 2015) to report the performance, unlike the original paper (Thapliyal et al., 2022) that uses CIDEr. We see that the LLaVA models are poor for most languages that are not written in Latin script, especially Japanese, Korean, Russian, Thai, and Chinese. bakLLaVA-v1 performs much worse compared to LLaVA-v1.5-13B and ViP-LLaVA-13B (except English), and the latter two are comparable on all languages. Most Latin script high-resource languages such as French, German, Dutch, Spanish, and Italian outperform or come close to English performance, with lower-resource languages such as Danish, Czech Polish, and Norwegian performing worse. GPT-4-Vision significantly outperforms LLaVA models on all languages, however, the scores on Chinese, Japanese, and Thai are still very poor. French has the highest score followed by Italian, Spanish, and then English, which again shows that GPT-4-Vision is good at Latin script and European languages. Gemini-Pro-Vision is the second-best model on all languages, and the results follow the same trend as GPT-4-Vision. (Figure 28, Table 18).[^3]

### 4.23 The deviation of performance across language families and tasks

Given the experiments conducted, we look at how performance for a given Language Family or Task varies from the average performance (across the models covered in MEGAVERSE). In doing so we are interested in ranking how well models support different Language Families or Tasks.

The deviation for a given experiment $i$ in the Language Family or Task $(j)$ is defined as:

$$
\Delta_{(i, j)}=p \_\operatorname{score}_{(i, j)}-\frac{1}{N} \sum_{i}^{N} p \_\operatorname{score}_{(i, j)}
$$

Where $p \_s c o r e{ }_{(i, j)}$ is the penalized score for the experiment $i$, and a high positive value indicates that a given subject (Language Family or Task) performs better than average where as a low negative value indicates that the subject performs lower than the average (across all models). p_score $(i, j)$ is calculated as:

$$
p_{-} \operatorname{score}_{(i, j)}=\left(\frac{\left|X_{j}\right|}{\left.\sum_{i}\left|X_{j}\right|\right)}\right) * \text { score }_{i}
$$

Where $\operatorname{score}_{i}$ is the normalized score for the experiment, penalized by the ratio of the instances in a given language family/task $(j)$ to the total number of instances in all the language families/tasks.

Because of the sparsity in (Language, Dataset, Model) combinations (see Table 1), we apply the size penalization to limit the bias of outliers and combinations with little support. For example, there are total of 320 IE: Iranian Language family experiments in our data, with an average score of 0.31 , and a penalized score of 0.05 , compared to Basque which has 10 experiments with an average score of 0.54 , but a penalized score of 0.003 .

Figure 3 gives the distribution of the $\Delta_{(i, j)}$ scores for Language Families and Tasks. We observe that languages in IE:Germanic Family, which ranks at the top, attain a significantly higher score that the mean, while at the the opposite end, Bantu and Afro-Asiatic languages significantly underperform the mean across models and datasets. We also find that the models tested are significantly better at tasks such as MCQ Reading Comprehension and Parts of Speech Tagging (across all languages), than more open tasks such as Q\&A and text Summarization.

## 5 Contamination Analysis

### 5.1 Commercial Model Contamination Study

In our work, we follow the method described by Golchin and Surdeanu (2023a) where we try to quantify contamination for commercial models such as PaLM2 and GPT-4. First, we prompt the model to generate three perturbations of the test set data points. Next, we provide these perturbations appended with the original text as four options to the model, and prompt it to pick a preferred option. We measure contamination as the chance adjusted accuracy using Cohen's Kappa ( $\kappa$ ) and account for LLM's position bias towards a particular option by adjusting the calculation of $\kappa$, called $\kappa_{\text {fixed }}$.

We study contamination on GPT-4 and PaLM2 for 5 datasets: PAWS-X, UDPOS, TyDiQA, XNLI, and XCOPA, on 100 data points per language in each dataset. Our results show that all datasets are highly contaminated except for UDPOS, and for all datasets, contamination is higher for GPT4, than for PaLM2. Contamination values for all datasets across different languages are reported in Appendix A.6. Contamination values differ significantly across languages for the same dataset, which could be due to bad perturbations generated by models owing to their varying performance in different languages. Another limitation of this approach is that Golchin and Surdeanu (2023a) study position bias only for GPT models and append the original text as the fourth option based on their observations. However, this could vary for different models.

### 5.2 Open-Source Model Contamination study

We follow the Black Box test for contamination study of open-source model described by Oren et al. (2023). This test is statistical test which provides provable guarantees that a given test set is contaminated. To achieve these guarantees, they exploit the fact that many datasets have a property known as exchangeability, where the order of examples in the dataset can be shuffled without affecting its join distribution. If a model has seen a benchmark dataset, it will have a preference for the canonical order (i.e. the order that examples are given in the public repositories) over randomly shuffled example orderings. If the difference between the said canonical order and the shuffled order is statistically significant, then the dataset is considered to be contaminated according to this method.

We conducted tests on the 7B instruction-tuned
![](https://cdn.mathpix.com/cropped/2024_06_04_b9a323c8bb7b325a4c6bg-09.jpg?height=802&width=1592&top_left_y=236&top_left_x=240)

Figure 3: The positive scores of the bar-plots denote that the current LLMs are relatively good with those language families / tasks.

variants of Llama2, Mistral, and Gemma across the following evaluation datasets: PAWS-X, XCOPA, XNLI, XQUAD, XRiSAWOZ, and XstoryCloze. The significance level for our analysis was set at 0.001. We observed (Table 33) that all models, except for the Gemma base model, exhibited contamination. Specifically, datasets such as PAWS-X, XCOPA, XQUAD, and XRiSAWOZ were found to have their $\mathrm{p}$-values less than the significant value for Gemma 7B Instruct, Llama2 7B Instruct and Mistral 7B Instruct indicating contamination.

## 6 Discussion

In this work, we benchmark 22 datasets covering 83 languages across several models - GPT-3.5-Turbo, GPT-4, PaLM2, Gemini-Pro, Gemma, Llama2, Mistral as well as multimodal models. We find similar trends across most datasets we study - larger commercial models such as GPT-4 and Gemini-pro outperform smaller models like Gemma, Llama and Mistral models, particularly on low-resource languages. This suggests that multilingual performance is a challenge for smaller models, and directions such as language-specific models, language family-based models and fine-tuning should be explored for better multilingual performance.

GPT-4, PaLM2 and Gemini-Pro excel on different datasets, with GPT-4 showing superior performance overall on multilingual datasets compared to both PaLM2 and Gemini-Pro. GPT-4-Vision outperforms LLaVA and Gemini-Pro-Vision on the multimodal datasets we study. Tokenizer fertility is correlated with Language Model performance (Rust et al., 2021; Ali et al., 2023). We plot the fertility analysis of all the tokenizers (Figure: 14) for the models that we studied in this work. We noticed that on average, Latin script languages such as Spanish, English had lower fertility as compared to languages that are morphologically complex languages like Telugu, Malay and Malayalam having high fertility amongst all the tokenizers.

Dataset contamination is a critical issue that affects English and non-English language benchmarking studies. Our contamination analysis on open source and commercial models shows that almost all models are contaminated with datasets included in MegaVerse. New multilingual evaluation datasets are difficult to create due to resource and funding constraints, hence, care should be taken to make sure that they are not included in the training data of LLMs. To achieve this, we need to enhance our ability to identify instances of contamination, as well as implement measures to avoid future contamination.

## 7 Limitations

Our work is subject to the following limitations:

Model comparison We have covered a wide array of Large Language Models. We realize that access to the commercial models (GPT, PaLM2, etc.) is via an API endpoint. These models might be
running various post-processing modules and classifiers resulting in an inflated performance as compared to the Open-source models (LLaVA, Llama, Mistral).

Dataset contamination We perform the dataset contamination exercise on a few set of datasets for PaLM2 and GPT-4 on a granular level. We also perform a thorough analysis of the open-source models covered in MEGAVERSE. However, there were certain limitations that we discuss in depth in Section 5. We were also limited by the compute and time, therefore we did not perform the contamination study on all our datasets and only covered the 7B variants of our open-source models.

Prompt tuning LLMs are sensitive to prompting, and we do not perform extensive prompt tuning for the new datasets. We also do not experiment with prompting variations, such as translate-test and zero-shot cross-lingual prompting, or more complex strategies such as Chain of Thought prompting due to resource constraints.

Experiments on limited data and datasets Due to resource constraints, we perform experiments on partial datasets when indicated, and do not evaluate all models on all datasets. We plan to do so in future work.

Focus on task accuracy We perform limited experiments on RAI datasets and do not perform experiments on other important dimensions such as fairness, bias, robustness, efficiency, etc., mainly due to the lack of such datasets for non-English languages. This is an important future research direction.

## References

Judit Ács. 2019. Exploring BERT's Vocabulary. Blog Post.

Divyanshu Aggarwal, Vivek Gupta, and Anoop Kunchukuttan. 2022. IndicXNLI: Evaluating multilingual inference for Indian languages. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10994-11006, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023 MEGA: Multilingual evaluation of generative AI.
In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232-4267, Singapore. Association for Computational Linguistics.

Mehdi Ali, Michael Fromm, Klaudia Thellmann, Richard Rutmann, Max Lübbering, Johannes Leveling, Katrin Klug, Jan Ebert, Niclas Doll, Jasper Schulze Buschhoff, Charvi Jain, Alexander Arno Weber, Lena Jurkschat, Hammam Abdelwahab, Chelsea John, Pedro Ortiz Suarez, Malte Ostendorff, Samuel Weinbach, Rafet Sifa, Stefan Kesselheim, and Nicolas Flores-Herr. 2023. Tokenizer choice for llm training: Negligible or crucial?

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Martin Chadwick, Gaurav Singh Tomar, Xavier Garcia, Evan Senter, Emanuel Taropa, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Yujing Zhang, Ravi Addanki, Antoine Miech, Annie Louis, Laurent El Shafey, Denis Teplyashin, Geoff Brown, Elliot Catt, Nithya Attaluri, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming-Wei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole,

Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozińska, Vitaly Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Patel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, Hanzhao Lin, James Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo yiin Chang, Paul Komarek, Ross McIlroy, Mario Lučić, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yong Cheng, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Kon- stantinos Aisopos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çağlar Ünlü, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James CobonKerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, YaGuang Li, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Moufarek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Rakićević, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Gamaleldin Elsayed, Ed Chi, Mahdis Mahdieh, Ian Tenney, Nan Hua, Ivan Petrychenko, Patrick Kane, Dylan Scandinaro, Rishub Jain, Jonathan Uesato, Romina Datta, Adam Sadovsky, Oskar Bunyan, Dominik Rabiej, Shimu Wu, John Zhang, Gautam Vasudevan, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Betty Chan, Pam G Rabinovitch, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Sahitya Potluri, Jane Park, Elnaz Davoodi, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Chris Gorgolewski, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Paul Suganthan, Evan Palmer, Geoffrey Irving, Edward Loper, Manaal Faruqui, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Michael Fink, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybiński, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marin Georgiev, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier

Dousse, Fan Yang, Jeff Piper, Nathan Ie, Minnie Lui, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Lam Nguyen Thiet, Daniel Andor, Pedro Valenzuela, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Sarmishta Velury, Sebastian Krause, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Tejasi Latkar, Mingyang Zhang, Quoc Le, Elena Allica Abellan, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Sid Lall, Ken Franko, Egor Filonov, Anna Bulanova, Rémi Leblond, Vikas Yadav, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Hao Zhou, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Jeremiah Liu, Mark Omernick, Colton Bishop, Chintu Kumar, Rachel Sterneck, Ryan Foley, Rohan Jain, Swaroop Mishra, Jiawei Xia, Taylor Bos, Geoffrey Cideron, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Petru Gurita, Hila Noga, Premal Shah, Daniel J. Mankowitz, Alex Polozov, Nate Kushman, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Anhad Mohananey, Matthieu Geist, Sidharth Mudgal, Sertan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christopher Yew, Quan Yuan, Sumit Bagri, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Aliaksei Severyn, Jonathan Lai, Kathy Wu, HengTze Cheng, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Mark Geller, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Andrei Sozanschi, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Abhimanyu Goyal, Diane Wu, Denese OwusuAfriyie, Cosmo Du, Chloe Thornton, Jordi PontTuset, Pradyumna Narayana, Jing Li, Sabaer Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Tao Zhu, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Dustin Tran, Yeqing Li, Nir Levine, Ariel Stolovich, Norbert Kalb, Rebeca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Balaji Lakshminarayanan, Charlie Deck, Shyam Upadhyay, Hyo Lee, Mike Dusenberry, Zonglin Li, Xuezhi Wang, Kyle Levin, Raphael Hoffmann, Dan HoltmannRice, Olivier Bachem, Summer Yue, Sho Arora, Eric Malmi, Daniil Mirylenka, Qijun Tan, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Steven Zheng, Francesco Pongetti, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Ragha Kotikalapudi, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz,
Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Chenkai Kuang, Vinod Koverkathu, Christopher A. Choquette-Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Pei Sun, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Ishita Dasgupta, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Yuan Liu, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Plucińska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Ivo Penchev, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Adam Kurzrok, Lynette Webb, Sahil Dua, Dong Li, Preethi Lahoti, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Taylan Bilal, Evgenii Eltyshev, Daniel Balle, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Adams Yu, Christof Angermueller, Xiaowei Li, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Kevin Brooks, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Komal Jalan, Dinghua Li, Ginger Perng, Blake Hechtman, Parker Schuh, Milad Nasr, Mia Chen, Kieran Milan, Vladimir Mikulik, Trevor Strohman, Juliana Franco, Tim Green, Demis Hassabis, Koray Kavukcuoglu, Jeffrey Dean, and Oriol Vinyals. 2023a. Gemini: A family of highly capable multimodal models.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023b. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th

Annual Meeting of the Association for Computational Linguistics, pages 4623-4637.

Akari Asai, Sneha Kudugunta, Xinyan Velocity Yu, Terra Blevins, Hila Gonen, Machel Reid, Yulia Tsvetkov, Sebastian Ruder, and Hannaneh Hajishirzi. 2023. Buffet: Benchmarking large language models for few-shot cross-lingual transfer. arXiv cs. $C L$ 2305.14857 .

Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondřej Dušek. 2024. Leak, cheat, repeat: Data contamination and evaluation malpractices in closedsource 1lms. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics. Association for Computational Linguistics.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2023. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. arXiv preprint arXiv:2308.16884.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašić. 2018. MultiWOZ - a largescale multi-domain Wizard-of-Oz dataset for taskoriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 5016-5026, Brussels, Belgium. Association for Computational Linguistics.

Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. 2023. Making large multimodal models understand arbitrary visual prompts. arXiv preprint arXiv: 2312.00784.

De Choudhury et al. 2023. Ask me in english instead: Cross-lingual evaluation of large language models for healthcare queries. arXiv preprint arXiv:2310.13132.

Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 8:454-470.
Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of EMNLP 2018, pages 2475-2485.

Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling humancentered machine translation.

Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar. 2023. Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12402-12426, Toronto, Canada. Association for Computational Linguistics.

Jay Gala, Pranjal A Chitale, A K Raghavan, Varun Gumma, Sumanth Doddapaneni, Aswanth Kumar M, Janki Atul Nawale, Anupama Sujatha, Ratish Puduppully, Vivek Raghavan, Pratyush Kumar, Mitesh M Khapra, Raj Dabre, and Anoop Kunchukuttan. 2023. Indictrans2: Towards high-quality and accessible machine translation models for all 22 scheduled indian languages. Transactions on Machine Learning Research.

Shahriar Golchin and Mihai Surdeanu. 2023a. Data contamination quiz: A tool to detect and estimate contamination in large language models.

Shahriar Golchin and Mihai Surdeanu. 2023b. Time travel in llms: Tracing data contamination in large language models.

Google. 2023. Palm-2 technical report.

Tahmid Hasan, Abhik Bhattacharjee, Md Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M Sohel Rahman, and Rifat Shahriyar. 2021. Xl-sum: Large-scale multilingual abstractive summarization for 44 languages. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4693-4703.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are gpt models at machine translation? a comprehensive evaluation. arXiv preprint arXiv:2302.09210.

Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Large multilingual models pivot zero-shot multimodal learning across languages.

Kent F Hubert, Kim N Awa, and Darya L Zabelina. 2024 The current state of artificial intelligence generative language models is more creative than humans on divergent thinking tasks.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2021. The state and fate of linguistic diversity and inclusion in the nlp world.

Simran Khanuja, Sandipan Dandapat, Sunayana Sitaram, and Monojit Choudhury. 2020a. A new dataset for natural language inference from codemixed conversations. In Proceedings of the The 4th Workshop on Computational Approaches to Code Switching, pages 9-16.

Simran Khanuja, Sandipan Dandapat, Anirudh Srinivasan, Sunayana Sitaram, and Monojit Choudhury. 2020b. Gluecos: An evaluation benchmark for codeswitched nlp. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3575-3585.

Ian Kivlichan, Jeffrey Sorensen, Julia Elliott, Lucy Vasserman, Martin Görner, and Phil Culliton. 2020. Jigsaw multilingual toxic comment classification.

Viet Dac Lai, Nghia Trung Ngo, Amir Pouran Ben Veyseh, Hieu Man, Franck Dernoncourt, Trung Bui, and Thien Huu Nguyen. 2023. Chatgpt beyond english: Towards a comprehensive evaluation of large language models in multilingual learning.

Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2020. Mlqa: Evaluating cross-lingual extractive question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 73157330 .

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav
Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9019-9052, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhaojiang Lin, Andrea Madotto, Genta Winata, Peng Xu, Feijun Jiang, Yuxiang Hu, Chen Shi, and Pascale N Fung. 2021. Bitod: A bilingual multi-domain dataset for task-oriented dialogue modeling. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1. Curran.

Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. 2021. Visually grounded reasoning across languages and cultures. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10467-10485.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. Improved baselines with visual instruction tuning.

Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Pier Giuseppe Sessa, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. 2024. Gemma: Open models based on gemini research and technology.

Mehrad Moradshahi, Tianhao Shen, Kalika Bali, Monojit Choudhury, Gael de Chalendar, Anmol Goel, Sungkyun Kim, Prashant Kodali, Ponnurangam Kumaraguru, Nasredine Semmar, Sina Semnani, Jiwon Seo, Vivek Seshadri, Manish Shrivastava, Michael Sun, Aditya Yadavalli, Chaobin You, Deyi Xiong, and Monica Lam. 2023. X-RiSAWOZ: High-quality end-to-end multilingual dialogue datasets and fewshot agents. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2773-2794, Toronto, Canada. Association for Computational Linguistics.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15991-16111, Toronto, Canada. Association for Computational Linguistics.

Joakim Nivre, Mitchell Abrams, Željko Agić, Lars Ahrenberg, Lene Antonsen, Maria Jesus Aranzabe, Gashaw Arutie, Masayuki Asahara, Luma Ateyah, Mohammed Attia, et al. 2018. Universal dependencies 2.2.

Odunayo Ogundepo, Tajuddeen R Gwadabe, Clara E Rivera, Jonathan H Clark, Sebastian Ruder, David Ifeoluwa Adelani, Bonaventure FP Dossou, Abdou Aziz DIOP, Claytone Sikasote, Gilles Hacheme, et al. 2023. Afriqa: Cross-lingual open-retrieval question answering for african languages. arXiv preprint arXiv:2305.06897.

OpenAI. 2023a. Gpt4 technical report.

OpenAI. 2023b. Gptv system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf. Accessed: 2023-12-13.

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori B Hashimoto. 2023. Proving test set contamination in black box language models. arXiv preprint arXiv:2310.17623.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. Cross-lingual name tagging and linking for 282 languages. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1946-1958.
Barun Patra, Saksham Singhal, Shaohan Huang, Zewen Chi, Li Dong, Furu Wei, Vishrav Chaudhary, and Xia Song. 2023. Beyond English-centric bitexts for better multilingual language representation learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15354-15373, Toronto, Canada. Association for Computational Linguistics.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. 2020. Xcopa: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362-2376.

Maja Popović. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392-395, Lisbon, Portugal. Association for Computational Linguistics.

Maja Popović. 2017. chrF++: words helping character n-grams. In Proceedings of the Second Conference on Machine Translation, pages 612-618, Copenhagen, Denmark. Association for Computational Linguistics.

Jun Quan, Shian Zhang, Qian Cao, Zizhong Li, and Deyi Xiong. 2020. RiSAWOZ: A large-scale multidomain Wizard-of-Oz dataset with rich semantic annotations for task-oriented dialogue modeling. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 930-940, Online. Association for Computational Linguistics.

Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, and Iryna Gurevych. 2021. How good is your tokenizer? on the monolingual performance of multilingual language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3118-3135, Online. Association for Computational Linguistics.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakaş, B. Ryan Roberts,

Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Bryan Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo JaimovitchLópez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Kocoń, Jana Thompson, Janelle Wingfield, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant, Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Guerr, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia ContrerasOchando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Colón, Luke Metz, Lütfi Kerem Şenel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn,
Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Michał Swędrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mitch Walker, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan A. Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nicole Martinez, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramon Risco, Raphaël Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima, Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsu Hashimoto, Te-Lin Wu, Théo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu.

2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

Gabriel Stanovsky, Noah A Smith, and Luke Zettlemoyer. 2019. Evaluating gender bias in machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1679-1684.

Ashish V Thapliyal, Jordi Pont Tuset, Xi Chen, and Radu Soricut. 2022. Crossmodal-3600: A massively multilingual multimodal evaluation dataset. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 715-729.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

David Vilares, Miguel A Alonso, and Carlos GómezRodríguez. 2016. En-es-cs: An english-spanish codeswitching twitter corpus for multilingual sentiment analysis. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16), pages 4149-4153.

Tsung-Hsien Wen, Milica Gašić, Nikola Mrkšić, PeiHao Su, David Vandyke, and Steve Young. 2015. Semantically conditioned LSTM-based natural language generation for spoken dialogue systems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1711-1721, Lisbon, Portugal. Association for Computational Linguistics.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Proceedings of EMNLP 2019, pages 3685-3690.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. 2023. The dawn of $1 \mathrm{mms}$ : Preliminary explorations with gpt-4v(ision).

Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, and Jiawei Han. 2023. Don't make your $11 \mathrm{~m}$ an evaluation benchmark cheater. arXiv preprint arXiv:2311.01964.
</end of paper 3>


<paper 4>
# Towards Accurate Differential 

## Diagnosis with Large Language Models

Daniel McDuff*, ‡, 1, Mike Schaekermann*, ‡, 1, Tao Tu ${ }^{*, 1}$, Anil Palepu ${ }^{*, 1}$,<br>Amy Wang ${ }^{1}$, Jake Garrison ${ }^{1}$, Karan Singhal ${ }^{1}$, Yash Sharma ${ }^{1}$, Shekoofeh Azizi ${ }^{2}$,<br>Kavita Kulkarni ${ }^{1}$, Le Hou ${ }^{1}$, Yong Cheng ${ }^{2}$, Yun Liu ${ }^{1}$,<br>S Sara Mahdavi ${ }^{1}$, Sushant Prakash ${ }^{1}$, Anupam Pathak ${ }^{1}$, Christopher Semturs ${ }^{1}$,<br>Shwetak Patel ${ }^{1}$, Dale R Webster ${ }^{1}$, Ewa Dominowska ${ }^{1}$,<br>Juraj Gottweis ${ }^{1}$, Joelle Barral ${ }^{2}$, Katherine Chou ${ }^{1}$, Greg S Corrado ${ }^{1}$, Yossi Matias ${ }^{1}$,<br>Jake Sunshine ${ }^{\dagger, \ddagger, 1}$, Alan Karthikesalingam ${ }^{\dagger}, \ddagger, 1$ and Vivek Natarajan ${ }^{\dagger, \ddagger, 1}$<br>${ }^{1}$ Google Research, ${ }^{2}$ Google DeepMind


#### Abstract

An accurate differential diagnosis (DDx) is a cornerstone of medical care, often reached through an iterative process of interpretation that combines clinical history, physical examination, investigations and procedures. Interactive interfaces powered by Large Language Models (LLMs) present new opportunities to both assist and automate aspects of this process. In this study, we introduce an LLM optimized for diagnostic reasoning, and evaluate its ability to generate a DDx alone or as an aid to clinicians. 20 clinicians evaluated 302 challenging, real-world medical cases sourced from the New England Journal of Medicine (NEJM) case reports. Each case report was read by two clinicians, who were randomized to one of two assistive conditions: either assistance from search engines and standard medical resources, or LLM assistance in addition to these tools. All clinicians provided a baseline, unassisted DDx prior to using the respective assistive tools. Our LLM for DDx exhibited standalone performance that exceeded that of unassisted clinicians (top-10 accuracy $59.1 \%$ vs $33.6 \%,[p=0.04]$ ). Comparing the two assisted study arms, the DDx quality score was higher for clinicians assisted by our LLM (top-10 accuracy $51.7 \%$ ) compared to clinicians without its assistance (36.1\%) (McNemar's Test: $45.7, p<0.01$ ) and clinicians with search (44.4\%) (4.75, $p=0.03)$. Further, clinicians assisted by our LLM arrived at more comprehensive differential lists than those without its assistance. Our study suggests that our LLM for DDx has potential to improve clinicians' diagnostic reasoning and accuracy in challenging cases, meriting further real-world evaluation for its ability to empower physicians and widen patients' access to specialist-level expertise.


## 1 Introduction

An accurate diagnosis is a critical component of effective medical care. Building AI systems capable of performing or assisting clinicians in this important task has been a long-standing grand challenge [1]. While prior focus has been on evaluating a machine's ability to accurately output a diagnosis [2-5], real-world clinical practice involves an iterative and interactive process of reasoning about a differential diagnosis (DDx), weighing multiple diagnostic possibilities in the light of increasing amounts of clinical information over time (ranging from clinical history and examination to investigations and procedures). Deep learning has been applied to promising effect for generating DDx in a number of specialties including radiology [3], ophthalmology [4] and dermatology [2], but such systems lack the interactive capabilities to fluently assist a user through communication in natural language.

The emergence of Large Language Models (LLMs) present an opportunity to design novel interactive tools and interfaces to aid in differential diagnosis. Such LLMs trained on vast corpora of text, can recognize, summarize, predict, and generate new text based on knowledge gained during the learning process and task specification via a prompt. These models have demonstrated the ability to perform complex language comprehension and reasoning tasks, generating coherent text and thereby enabling a large variety of real-world applications [6-9].

Both general-purpose LLMs (GPT-4) and medical domain-specialized LLMs (Med-PaLM 2) have demonstrated

* Equal contributions. $\dagger$ Equal leadership.

$\ddagger$ Corresponding authors: \{dmcduff, mikeshake, jakesunshine, alankarthi,natviv\}@google.com
strong performance in standardized and multiple-choice medical benchmarks [10, 11]. Such evaluations represent a natural starting point for probing the medical knowledge and capabilities but fail to measure utility in real-world scenarios for care delivery, for example in challenging medical cases faced by trained physicians. It is also not obvious how these models might actively assist clinicians in the development of a DDx. Recent work has begun to assess the standalone performance of these models on challenging case reports that involve complex deduction $[5,12,13]$, but has stopped short of evaluating how they can assist clinicians and augment performance and empower them to provide better care.

In this work, we introduced and investigated the ability of an LLM optimised for clinical diagnostic reasoning, to generate a DDx in challenging, real-world medical cases. Beyond measuring standalone performance like prior work [5], we integrated this model into an interactive interface to measure how well our LLM could assist clinicians in developing a DDx. Using a set of challenging real-world cases from the New England Journal of Medicine (NEJM) case reports, we compared clinicians' ability to form a DDx with the assistance of our LLM, versus with access to traditional information retrieval tools (e.g., Internet search and books). The LLM achieved impressive performance in both generating DDx lists that contained the correct diagnosis (i.e., top-10 accuracy) and in identifying the correct final diagnosis as the most likely in the list (i.e., top-1 accuracy). Under automated model based evaluation, the quality and the accuracy of the DDx list produced by our LLM was found to be significantly better than the state-of-the-art GPT-4 model [5].

Perhaps, more importantly, the LLM also improved the diagnostic capability of clinicians as measured by the quality of their DDx lists for the evaluated cases. LLMs optimized for the safety-critical medical domain such as ours present a novel paradigm for assisting clinicians because of the potential for variation in the ways in which a given individual may converse with the system and utilise it in collaborative reasoning. We used semi-structured qualitative interviews to gather information from participating clinicians on their experiences of using the tool, their views of the potential role and risks of LLMs in medical diagnosis and in aiding the differential diagnosis process. These interviews highlighted the potential for LLMs to increase the diversity of DDx lists and speed up the process of arriving at a comprehensive DDx for challenging cases. The clinicians also highlighted that the most appropriate application at the present time would be in learning and education.

Our key contributions can be summarized as:

- Introducing an LLM for DDx, a model optimized for differential diagnosis, alongside a user interface allowing clinicians to interact with the model for improving clinical diagnostic reasoning.
- Evaluating the performance of the LLM on challenging diagnostic cases from the NEJM Case Reports.
- Showing that the LLM outperforms the prior state of the art, GPT-4, in both top-1 and top-10 accuracy on this benchmark under automated evaluation.
- Evaluating the impact of the LLM as an assistive tool for clinicians in differential diagnosis, with randomized comparison to the usual practice in which clinicians are assisted by Search and their usual clinical resources.


## 2 NEJM Clinicopathological Conference Case Reports

The Case Records of the Massachusetts General Hospital (MGH) are published (lightly edited) transcriptions of the clinicopathological conferences of the MGH (Boston, MA). In the clinicopathological conference, a patient case presentation is described and then an expert physician is asked to provide a DDx and a final diagnosis, along with their diagnostic reasoning, based only on the patient's provided medical history and preliminary test results. The published cases, organized generally as diagnostic puzzles culminating in a definitive, pathology-confirmed diagnosis, are published regularly in the NEJM. We leverage these case reports, licensed from the NEJM, to evaluate the LLM's capability to generate a DDx alone and, separately, to aid clinicians in generation of their own differential. For this latter task, we developed a user interface for clinicians to interact with the LLM.

A set of 326 case texts from the NEJM Clinicopathological Conference (CPC) series were considered. These case reports were published over a 10 year period between June $13^{\text {th }} 2013$ and August $10^{\text {th }}$ 2023. Of these, 23 $(7 \%)$ were excluded on the grounds that they discussed case management and were not primarily focused on diagnosis. The articles were distributed over the years between 2013 -2023 as follows: $2013 \mathrm{~N}=22,2014$
$\mathrm{N}=34,2015 \mathrm{~N}=36,2016 \mathrm{~N}=35,2017 \mathrm{~N}=36,2018 \mathrm{~N}=16,2020 \mathrm{~N}=23,2021 \mathrm{~N}=36,2022 \mathrm{~N}=39,2023 \mathrm{~N}=26$. The supplementary material includes the full set of case numbers. The 302 cases include the 70 cases used by Kanjee et al. [5].

These case reports cover a range of medical specialties. The largest proportion are from internal medicine $(\mathrm{N}=159)$, followed by neurology $(\mathrm{N}=42)$, pediatrics $(\mathrm{N}=33)$ and psychiatry $(\mathrm{N}=10)$. The text corresponding to the history of the present illness (HPI) was manually extracted from each article as input to the LLM. The average (median) word count of these sections of the case reports is 1,031 words (mean: 1,044, SD: 296, range: $378-2,428$ ). The average (median) character count is 6,619 characters (mean: 6,760, SD: 1,983, range: $2,426-15,196)$.

A modified version of the article, inclusive of the provided HPI, admission imaging and admission labs (if available in the case) was created for the human clinicians (see Fig. 1). This version had redacted the final diagnosis, expert discussion of the $\mathrm{DDx}$ and any subsequent imaging or biopsy results (which are typical elements of the conclusion of the case challenges). Given the LLM is a text-only AI model, the admission images and lab tables were not fed into the model. However, text-based descriptions of specific lab values or imaging findings were sometimes included in the case description.

## 3 Training a Large Language Model for DDx

Our study introduces an LLM for DDx, a model which uses a transformer architecture (PaLM 2 [7]), fine-tuned on medical domain data; alongside an interface for enabling its use as an interactive assistant for clinicians.

As with Med-PaLM 2 [10], our LLM builds upon PaLM 2, an iteration of Google's LLM with substantial performance improvements on multiple LLM benchmark tasks. For the purposes of this analysis the large (L) PaLM 2 model was used.

The LLM was fine-tuned with long context length on a task mixture consisting of medical question answering (multiple-choice and long-form questions), medical dialogue generation and electronic health record (EHR) note summarization. The datasets used included the training splits of MultiMedQA (MedQA, MedMCQA, HealthSearchQA, LiveQA and MedicationQA) [10], a proprietary dataset of medical conversations, and expert handcrafted EHR note summaries from MIMIC-III [14]. The capability to process long context input enables the LLM to handle tasks that require long-range reasoning and comprehension.

Zero-Shot Prompting. We evaluated the LLM on each of the NEJM case studies with the following prompt: "You are a helpful medical assistant. You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and thorough DDx".

Example Input:

Case: [Case Text]

What are the top 10 most likely diagnoses? Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). The top 10 diagnoses are:

Example Output:

1. Sézary syndrome
2. Adult T-cell leukemia/lymphoma
3. Mycosis fungoides
4. Atopic dermatitis
5. Psoriasis
6. Drug reaction with eosinophilia and systemic symptoms (DRESS)
7. Graft-versus-host disease (GVHD)
8. Cutaneous T-cell lymphoma, not otherwise specified
9. Hypereosinophilic syndrome
10. Systemic lupus erythematosus (SLE)

## 4 The LLM for DDx User Interface

The interface associated with our LLM, depicted in Fig. 2, enables users to interact with the underlying model via text-based chat in the context of a given case description. In our study, the interface was pre-populated with a text-only representation of the history of present illness (HPI) for a given case. Clinicians were asked to initiate the interaction by querying the LLM using a suggested prompt. Following this initial prompt and the LLM's response, clinicians were free to query the model using any additional follow-up questions, though clinicians were cautioned to avoid asking questions about information that had not already been presented in the case. A pilot study indicated that without such a warning, clinicians may ask questions about specific lab values or imaging leading to confabulations.

For a given question, the interface generated the response by querying the LLM using prompt template:

Read the case below and answer the question provided after the case.

Format your response in markdown syntax to create paragraphs and bullet points. Use ' $<\mathrm{br}><\mathrm{br}>$ ' to start a new paragraph. Each paragraph should be 100 words or less. Use bullet points to list multiple options. Use ' $<\mathrm{br}>^{*}$ ' to start a new bullet point. Emphasize important phrases like headlines. Use ${ }^{(* *}$, right before and right after a phrase to emphasize it. There must be NO space in between ${ }^{(* *)}$ and the phrase you try to emphasize.

Case: [Case Text]

Question (suggested initial question is "What are the top 10 most likely diagnoses and why (be precise)?"): [Question]

Answer:

## 5 Methods

### 5.1 Experimental Design

In order to comparatively evaluate the LLM's ability to generate a DDx alone and aid clinicians with their DDx generation we designed a two-stage reader study illustrated in Fig. 3). Our study was designed to evaluate the assistive effect of the LLM for generalist clinicians (not specialists) who only have access to the case presentation and not the full case information (which would include the expert commentary on the DDx). The first stage of the study had a counterbalanced design with two conditions. Clinicians generated DDx lists first without assistance and then a second time with assistance, where the type of assistance varied by condition.

## Stage 1. Clinicians generate DDx with and without assistance

Twenty U.S. board-certified internal medicine physicians (median years of experience: 9, mean: 11.5, SD: 7.24, range: 3-32) viewed the redacted case report, with access to the case presentation and associated figures and tables. They did this task in one of two conditions, based on random assignment.

Condition $I$ - Search. The clinicians were first instructed to provide a list of up to ten diagnoses, with a minimum of three, based solely on review of the case presentation without using any reference materials (e.g., books) or tools (e.g., Internet search). Following this, the clinicians were instructed to use Internet Search or other resources as desired (but not given access to the LLM) and asked to re-perform their DDx.

Condition II - LLM for $D D x$. As with condition I, the clinicians were first instructed to provide a list of up to 10 diagnoses, with a minimum of three, based solely on review of the case presentation without using any reference materials (e.g., books) or tools (e.g., Internet search). Following this the clinicians were given access to the LLM and asked to re-perform their DDx. In addition to the LLM, clinicians could choose to use Internet search or other resources if they wished.

Stage 2. Specialists with full case information extract gold DDx and evaluate Stage 1 DDx

Nineteen U.S. board-certified specialist clinicians (median years of experience: 14, mean: 13.7, SD: 7.82, range: 4 -38) were recruited from internal medicine $(\mathrm{N}=10)$, neurology $(\mathrm{N}=3)$, pediatrics $(\mathrm{N}=2)$, psychiatry $(\mathrm{N}=1)$,
dermatology ( $\mathrm{N}=1)$, obstetrics $(\mathrm{N}=1)$, and emergency medicine $(\mathrm{N}=1)$. Their mean years of experience was 13.7 (SD: 7.82 , range: $4-38$ ). These specialists were aligned with the specialty of the respective CPC case, viewed the full case report and were asked to list at least 5 and up to 10 differential diagnoses. Following this, they were asked to evaluate the five DDx lists generated in Stage 1, including two DDx lists from condition 1 (DDx without Assistance and DDx with Search Assistance), two DDx lists from condition 2 (DDx without Assistance and DDx with LLM Assistance) and the standalone LLM DDx list.

The specialists answered the following questions to evaluate the DDx lists:

The quality score developed by Bond et al. [15] and used by Kanjee et al. [5] is a differential score based on an ordinal 5 -point scale: "How close did the differential diagnoses (DDx) come to including the final diagnosis?"' The options were: 5. DDx includes the correct diagnosis, 4. DDx contains something that is very close, but not an exact match to the correct diagnosis, 3. DDx contains something that is closely related and might have been helpful in determining the correct diagnosis, 2. DDx contains something that is related, but unlikely to be helpful in determining the correct diagnosis, 1. Nothing in the DDx is related to the correct diagnosis.

An appropriateness score: "How appropriate was each of the differential diagnosis lists from the different medical experts compared the differential list that you just produced?". The options to respond were on a Likert scale of 5 (very appropriate) to 1 (very inappropriate).

A comprehensiveness score: "Using your differential diagnosis list as a benchmark/gold standard, how comprehensive are the differential lists from each of the experts?"' The options to respond were: 4. The DDx contains all candidates that are reasonable, 3. The DDx contains most of the candidates but some are missing, 2. The DDx contains some of the candidates but a number are missing, 1. The DDx has major candidates missing.

Finally, specialists were asked to specify in which position of the DDx list the correct diagnosis was matched, in case it was included in the DDx at all.

Automated Evaluation. In addition to comparing against ground-truth diagnosis and expert evaluation from clinicians, we also created an automated evaluation of the performance of the five DDxs using a languagemodel based metric. Such automated metrics are useful as human evaluation is time and cost-prohibitive for many experiments. We first extracted the (up to ten) individual diagnoses listed in each DDx. We leveraged minor text-processing steps via regular expressions to separate the outputs by newlines and strip any numbering before the diagnoses. Then we asked a medically fine-tuned language model, Med-PaLM 2 [10], whether or not each of these diagnoses was the same as the ground-truth diagnosis using the following prompt:

Is our predicted diagnosis correct $(\mathrm{y} / \mathrm{n})$ ? Predicted diagnosis: [diagnosis], True diagnosis: [label] Answer $[\mathrm{y} / \mathrm{n}]$.

A diagnosis was marked as correct if the language model output 'y'.

### 5.2 Qualitative Interviews

Following the study we performed a semi-structured 30 -minute interviews with five of the generalist clinicians who participated in Stage 1. Semi-structured interviews explored the following questions:

1. How did you find the task of generating a DDx from the case report text?
2. Think about how you used Internet search or other resources. How were these tools helpful or unhelpful?
3. Think about how you used the LLM for DDx. How was it helpful or unhelpful?
4. Were there cases where you trusted the output of the search queries? Tell us more about the experience if so, such as types of cases, types of search results.
5. Were there cases where you trusted the output of the LLM queries? Tell us more about the experience if so, such as types of cases, types of search results.
6. Think about the reasoning provided by the LLM's interface? Where were they helpful? Where were they unhelpful?
7. What follow-up questions did you find most helpful to ask the LLM?
8. How much time does it take to get used to the LLM? How was it intuitive? How was it unintuitive?

## 6 Results

In evaluating the quality of the DDx lists we used several criteria, inspired by the approach taken in [5] and extended to draw additional insight from the clinicians. First, we measured whether the final diagnosis matched an entry in the DDx list and in which position (top-N accuracy). Second, we used Bond et al.'s [15] quality score and the appropriateness and comprehensiveness scales that we created. Combined these measures assess overall DDx quality, appropriateness and comprehensiveness.

When using the LLM for assistance, clinicians asked, on average (mean), 2.92 questions in the interface (median, 2 and IQR, 1-4). On average (mean), clinician questions consisted of 9.39 words (median, 10 and IQR, 6-12) and 54.31 characters (median, 61 and IQR, 39-63). The LLM responses, on average (mean), consisted of 237.60 words (median, 198 and IQR, 127-332) and 1540.81 characters (median, 1276 and IQR, 815-2210).

In the Search condition the most popular tools were UpToDate (used in $34 \%$ of tasks), Google Search (30\%) and PubMed $(22 \%)$. While clinicians were allowed to use additional tools in the LLM condition, this was far less frequent $(<5 \%$ of tasks).

### 6.1 Performance of the Language Model on Differential Diagnosis

## Quality, Appropriateness and Comprehensiveness.

Our language model's DDx lists achieved strong quality, appropriateness and comprehensiveness scores (see Fig. 4). The median quality score was 5 ("DDx includes the correct diagnosis") with $54 \%$ of DDx lists achieving that score. The number of cases that scored 5 (i.e., the DDx included the top diagnosis) was statistically significantly higher for the LLM compared to clinicians without assistance (McNemar's Test: 64.4, $p<0.01$ ). The mean appropriateness score of 4.43 out of five (SD: 0.92). The median comprehensiveness score was 4 (= "The DDx contains all candidates that are reasonable") with $55 \%$ of the DDx lists achieving that score.

The mean appropriateness score of the LLM (4.34) was significantly higher than that for unassisted clinicians (3.74) (paired t-test $8.52, p<0.001$ ) and assisted clinicians in either the Search (3.80) (paired t-test 7.23, $p<$ 0.001 ) or LLM (4.06) (paired t-test 4.98, $p<0.001$ ) conditions.

Top-N Accuracy. For computing top- $\mathrm{N}$ accuracy, if any of the first $\mathrm{N}$ diagnoses in an individual DDx were marked correct by the language model, the differential was considered to be correct. We computed the proportion of correct DDx lists across all cases to compute the top- $\mathrm{N}$ accuracy (for $\mathrm{N}$ from 1 to 10) for each DDx. The LLM reliably generated DDx lists that perform well against the ground-truth diagnosis. Fig. 5 shows the top-N accuracy for the LLM. The LLM provided the correct diagnosis in 177 (59\%) of the DDx lists and in $89(29 \%)$ of the lists it was at the top of the list. These scores are above the scores the clinicians achieved in any of the conditions. The top-10 accuracy of the LLM (59.1\%) was significantly higher than the top-10 accuracy for the unassisted clinicians (33.6\%) $(p=0.04)$.

Fig. 5 shows the top-N accuracy based on human and the automated metric. The results are broadly comparable, illustrating that despite the final diagnoses often being complex and nuanced, the automated metric faithfully captures the distinction between a DDx list that includes the correct diagnosis and one that does not.

### 6.2 LLM for DDx as an Assistant for Differential Diagnosis

Quality, Appropriateness and Comprehensiveness. Of the DDx lists created before assistance $37 \%$ (Search condition) and $29 \%$ (LLM for DDx condition) achieved a quality score of 5 (Fig. 4). In comparison $49 \%$ of those created with assistance from the LLM scored 5 .

The number of cases that scored 5 (i.e., the DDx included the top diagnosis) was statistically higher for clinicians assisted by the LLM compared to clinicians without assistance (McNemar's Test: 48.3, $p<0.01$ ) and clinicians with Search assistance $(5.45, p=0.02)$.

For comprehensiveness, the number of cases that scored 4 (i.e., The DDx contains all candidates that are
reasonable) was statistically higher for clinicians assisted by the LLM compared to clinicians without assistance (McNemar's Test: 185.8, $p<0.01$ ) and clinicians with Search assistance $(185.8, p<0.01$ ).

The mean appropriateness score after assistance with the LLM (4.06) was significantly higher than after assistance with Search (3.80) (paired t-test 3.32, $p=0.001$ ) and the baseline (3.74) (paired t-test $4.79, p<$ 0.001 ).

To summarize, with the support of the LLM, the quality, appropriateness and comprehensiveness scores for the DDx lists were greater than for the lists prior to assistance (see Fig. 4).

Top-N Accuracy. The top-N accuracy of the clinicians increased with assistance from the LLM compared to without (see Fig. 5). A Sankey diagram illustrates the impact of the two forms of assistance (Search and LLM for DDx) on top-10 accuracy (Fig. 6). In the LLM condition 73 cases that did not feature the final diagnosis prior to using the tool included it after assistance from the LLM. This result is in contrast to only 37 cases in the Search condition. Comparing the two assisted study arms, the DDx quality score was higher for clinicians assisted by our LLM (top-10 accuracy $51.7 \%$ ) compared to clinicians without its assistance ( $36.1 \%$ ) (McNemar's Test: 45.7, $p<0.01)$ and clinicians with search $(44.4 \%)(4.75, p=0.03)$.

### 6.3 Duration of DDx Tasks with the LLM for DDx and Search

The time taken to generate updated DDx lists in the Search conditions vs the LLM condition were similar (Search: [7.19 minutes, $\mathrm{SD}=5.33$ ], $\mathrm{LLM}$ for $\mathrm{DDx}$ [7.29 minutes, $\mathrm{SD}=6.41]$ ). These were not significantly different (paired t-test $p=0.807$ ), which is surprising as the clinicians all had experience using Internet search and other information retrieval tools, yet they were using the LLM interface for the first time. We hypothesized that they would take longer using the LLM due to the initial learning curve.

### 6.4 Length of DDx Lists Using the LLM for DDx and Search

When unassisted, the median length of the DDx lists was 6 (IQR, 5-9); the mean was 6.41 (SD, 2.39). With search the median DDx list length was 7 (IQR, 5-10); the mean was 6.92 (SD, 2.52). With the LLM the median DDx list length was 8 (IQR, 6-10); the mean was 7.58 (SD, 2.33). With assistance from the LLM, the length of the DDx lists was longer than without assistance (paired t-test: $7.13, p<0.001$ ) and longer than the DDx lists with assistance from search (paired t-test: $3.15, p=0.002$ ).

### 6.5 Contamination Analysis

We trained the LLM by building on an model pretrained on large-scale data and fine-tuning on medical data. While we did not include NEJM case reports in the fine-tuning data for the model, it is possible that pretraining data for the model contained partial mentions or full NEJM case reports, whether from the original source (NEJM) or reproduced by other websites. To better understand the possibility of overlap between the training corpus and the test data, we performed a contamination analysis using the pretraining corpora. We looked for overlap between character sequences in the test articles and training corpora using a sliding window, searching for all instances of 512 -character overlap. A case report is considered to have overlap if at at least one document from the pretraining corpora has an overlap. We identified that there was no overlap for case reports beginning in 2022 and 2023. Some overlap existed for case reports published prior to 2022 . We calculated the top-N accuracy for the LLM on both of these sets of case reports, prior to 2022 ( $\mathrm{N}=238)$ and 2022 to date $(\mathrm{N}=65)$, and did not observe a substantial difference in results. Across all years, $16.9 \%$ (51 out of 302) of case reports displayed at least one instance of overlap.

### 6.6 LLM for DDx Comparison with GPT-4

As we did not have the same set of human raters who evaluated the differentials produced by GPT-4 [5] and our LLM, we can not compare top-10 accuracy numbers directly. Therefore, in our study design, we evaluate performance on that 70 cases subset (reported in [5]) using the automated metric (which is shown above to be relatively consistent with human evaluation). Our LLM for DDx performs better with regard to top-N accuracy for $\mathrm{N}>1$, with the gap being most prominent $\mathrm{N}>2$ (Fig. 7). This suggests potentially significant improvements in quality and comprehensiveness of the differentials produced by our LLM.

Table 1 | Top-1 and Top-10 Accuracy. The percentage of DDx lists with the final diagnosis.

| Metrics | Model-Only <br> LLM for DDx |  | Before Assistance |  | Human <br> After Search Assistance |  | After LLM for DDx Assistance |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Top-1 $1^{\uparrow}$ | Top-10 $0^{\uparrow}$ | Top-1 $1^{\uparrow}$ | Top-10 $10^{\uparrow}$ | Top-1 $1^{\uparrow}$ | Top-10 $\uparrow$ | Top- $1 \uparrow$ | Top-10 $0^{\uparrow}$ |
| Full Set (302 Cases) | $29.2 \%$ | $59.1 \%$ | $15.9 \%$ | $33.6 \%$ | $24.3 \%$ | $44.5 \%$ | $25.2 \%$ | $51.8 \%$ |
| Set without Overlap (56 Cases) | $35.4 \%$ | $55.4 \%$ | $13.8 \%$ | $34.6 \%$ | $29.2 \%$ | $46.2 \%$ | $24.6 \%$ | $52.3 \%$ |
| Difference | $+6.2 \%$ | $-3.7 \%$ | $-2.1 \%$ | $+1.0 \%$ | $+4.9 \%$ | $+1.7 \%$ | $-0.6 \%$ | $+0.5 \%$ |

### 6.7 Qualitative Analysis

We describe our qualitative results that provide insight into how the clinicians viewed the LLM. We identified several key themes in the clinicians' responses and present several illustrative quotes below.

Comparing Search and the LLM for DDx. One clinician contrasted the use of Search to the LLM in this way: "Search was adequate when I had a good idea of what the differential ought to be to start with, but there were some cases where I was only able to generate 3 or 4 because I really wasn't sure. If I put in 'infectious causes of headache" [to the search engine] those were not very helpful.", whereas "Ithe LLM] was required to pull some additional diagnoses that may not have been the final diagnosis but would be important to think about.".

## Use Cases.

C3: "I think if [the LLM] is intended for medical students or for medical education then it could be very very powerful tool.".

## Confabulations.

C1: "I walked into it thinking I could ask what ever I want, but if it was something that could not be inferred from the case I might get a response that isn't real or appropriate. C3: "The biggest thing I think that I had a concern about was inaccuracy for someone who does not have a clinical background". But the clinicians found ways to work around these limitations by leveraging their expertise or other resources to validate responses.

## Ease of Use.

$\mathrm{C} 2$ : "For me it was very intuitive, it was a very easy tool to use, I wish I had it every single day."). These comments highlight that natural language interfaces have a very low barrier to use.

## 7 Discussion

We used a popular series of complex diagnostic challenges to evaluate an LLM optimized for clinical reasoning and diagnosis (LLM for $\mathrm{DDx}$ ); both in a standalone capacity and under randomized comparisons as an assistive tool for physicians. In standalone performance, the LLM generated more appropriate and comprehensive DDx lists than physicians when they were unassisted, with its DDx lists more likely to include the final diagnosis than DDx lists from a board-certified internal medicine physician, no matter which position in the DDx list was considered (i.e., top- $\mathrm{N}$ accuracy with $\mathrm{N}$ ranging from 1 to 10). Clinicians using the LLM as an assistant produced a DDx with higher top-N accuracy, and DDx with greater quality, appropriateness and comprehensiveness; compared to the status quo for clinical practice (use of Internet search and other resources).

The NEJM clinicopathological conferences (CPCs) examined here are well-known for being unique and challenging clinical conundrums. Within this distinctive setting, the proposed LLM outperformed an unassisted board-certified physician in both top-1 and top-n performance. While the CPCs have long been used as benchmarks for difficult diagnosis, it is also well-known that performance in CPCs in no way reflects a broader measure of competence in a physician's duties [16]. Furthermore, the act of DDx comprises many other steps that were not scrutinized in this study, including the goal-directed acquisition of information under uncertainty (known to be challenging for AI systems despite recent technical progress in this direction [17-19]). While
based on real-world cases, the clinical pathology case presentation format and input into the model does differ in important ways from how a clinician would evaluate a patient and generate their DDx at the outset of a clinical encounter. For example, while the case reports are created as "puzzles" with enough clues that should enable a specialist to reason towards the final diagnosis, it would be challenging to create such a concise, complete and coherent case report at the beginning of a real clinical encounter.

We are therefore very cautious to extrapolate our findings toward any implications about the LLM's utility as a standalone diagnostic tool. Nevertheless, our controlled evaluation mirrored the findings of other recent works exploring the performance of both LLMs and pre-LLM "DDx generator" in smaller subsets of the NEJM $\mathrm{CPCs}$, which have shown the potential for automated technology to reach the correct DDx in these challenging cases with superior performance to standalone physicians $[5,12,13,20]$. While this represents a step beyond historical attempts at automating DDx in NEJM CPCs, where computerized approaches were deemed overtly unreliable for practical use [21], such studies also undertook limited consideration of the quality of DDx generated by these automated systems or their role as assistive tools.

Our work extends prior observations by showing not only that the LLM was more likely to arrive at a correct answer or provide the correct answer in a list, but that its DDx were determined by an independent rater to be of higher appropriateness and comprehensiveness than those produced by board certified physicians with access to references and search.

In our study clinicians had access to both images and tabular data in redacted case reports, while the LLM was only provided with the main body of the text. Though the LLM outperformed the clinicians despite this limitation, it is unknown whether and how much this gap would widen if the LLM had access to the figures and tables. Early evidence suggests the effect might be case/context dependent as other studies have found image access by models to not always improve performance in CPCs [13]. Furthermore, the integration of multimodal inputs by LLMs is an area of novel research [22, 23], with a large potential number of data modalities to consider, and little precedent for how information from multiple modalities should be integrated over time for a single case by AI systems.

The repeated examination of NEJM CPCs by automated systems highlights its promise as a "benchmark" for LLM evaluation and development. Benchmarking enables comparisons of models against one another and the ability to evaluate a model's performance improvements or degradation over time. However, consistency in using CPCs as a scalable benchmark is challenging if reliant upon using human judgement to establish whether a candidate differential matches the ground truth. We utilized an automated approach for comparing our LLM for DDx to a baseline LLM performance (GPT-4). Our estimates varied from that recently published in other studies, despite using the same subset of cases [5]. Direct comparisons of different technologies would ideally be conducted by more extensive and blinded human evaluation, including work to ensure reproducibility of the human evaluation protocol, analysis of inter-rater reliability, and the use of metrics reflecting the quality, appropriateness and comprehensiveness of LLM differentials in addition to estimations of accuracy. Our estimates of top-1 and top-10 accuracy, while impressive at close to $30 \%$ and $60 \%$ respectively, highlight noteworthy room for improvement for LLMs, especially for complex cases that are non-pathognomonic (i.e., cases that do not have a sign or symptom that defines a diagnosis). However, as noted above, the CPCs represent "diagnostic puzzles" rather than real-world examples of common clinical workflows; and it is therefore important to consider more realistic settings in which LLMs might prove of practical value in medicine.

One such example is the potential for LLMs to assist clinicians in complex diagnoses. Deep learning tools have shown considerable promise in many areas of medicine, but are overwhelmingly used as assistive rather than autonomous tools [24], given the safety-critical nature of medical practice and the many issues of robustness [25] and fairness [26-28] seen in deployment. Furthermore, observations of standalone diagnostic accuracy often do not guarantee that an AI tool will improve performance in real-world settings as an assistive tool, and it remains unclear how to optimally integrate AI and human decision-making in medicine [29]. For LLMs in particular, the known incidence of hallucination/confabulation [30] might mislead clinicians into inaccurate diagnosis, replicating or even extending findings in other clinical settings that AI systems might actually degrade the performance of clinicians rather than necessarily improving outcomes.

This highlights the importance of focused study of LLMs in assistive scenarios. We explored this specifically in NEJM CPCs and found that the proposed LLM for DDx, increased the number of appropriate DDx produced
by a clinician when used as an assistive tool in addition to overall top-N accuracy, suggesting that the LLM's primary assistive potential may be due to making the scope of DDx more complete. Given the potential for misleading information to arise from AI systems, including in convincing dialogue, clinicians must appreciate the fundamental limitations of these models and not lose sight of their primacy in the provider-patient relationship and their ultimate authority and responsibility for the diagnostic and therapeutic management of their patients. Such thoughtful and effective LLM use should not be unintuitive to most clinicians. Aiding the diagnostic process could reasonably occur in an emergency room upon presentation (during potentially time-sensitive moments), upon admission to the medical ward, or by a consulting service after a patient has been admitted or in outpatient clinics. Our findings suggest that onward research should more rigorously explore how LLMs augment clinicians' DDx in many such specific scenarios, where the risks and benefits might vary.

Despite being a novel tool, the use of the LLM did not seem to add inefficiency or increase the amount of time spent on solving each CPC compared to the use of Search or other conventional information. This suggests that the conversational interface was unobtrusive and intuitive. Consistent with this, the interviewed clinicians all described it as "easy" to use, and were positive about the use and implications of the LLM interface. Enhancing efficiency while maintaining or improving quality are generally accepted goals of improving health care delivery, alongside improving provider experience [31], and our study showed significant potential in this regard, as clinicians also reported feeling more confident in their DDx lists after using the model. "That is where the search really became difficult, I didn't know what to put in to narrow down my search that is going to help me narrow down my differential." However, there are numerous human factors, social elements, and other complex considerations in these use cases, and it is critical to ensure efforts are made to avoid inequities in access to avoid exacerbating existing health disparities.

Clinicians frequently expressed excitement about using the LLM but were also aware of the shortcomings of language models and had concerns about confabulations in particular if used by individuals not trained or instructed to avoid such questions. However, our work did not explore many other important aspects of human-AI interaction, which require further study in safety-critical settings such as this. For example, we did not explore the extent to which clinicians trusted the outputs of the model or their understanding of its training and limitations, or undertake focused "onboarding" or training in its use, which are all known to be important modifiers of the benefits derived by clinicians from AI assistants [32]. The CPC challenges themselves do not enable a rigorous exploration of the possible impacts of AI assistance on health equity and fairness; a further study of how these aspects of clinicians' DDx is impacted by LLM assistance is needed. While AI systems are known to be able to express uncertainty [33] and defer appropriately to clinicians [34], which might significantly improve the balance between trust and skepticism needed for effective AI assistance in medicine. Qualitative feedback suggested that there remains room for targeted improvement of LLMs as assistive diagnostic tools, with one clinician noting that "It was most helpful for simpler cases that were specific keywords or pathognomonic signs." (C3) but for more complex cases it still tended to draw conclusions from isolated symptoms rather than viewing the case holistically. The assistive effect of these LLMs could potentially 'upskill' clinical providers, particularly in enabling them to broaden and enhance the quality of their DDx. As corroborated via our clinician interviews after their experience with the LLM, such upskilling could be relevant for education or training purposes to support providers across a skill continuum ranging from trainees to attending providers. The upskilling capabilities could also extend to locations where specialist medical training is less common (e.g., in lower and middle income countries [LMIC]). However, our findings may not generalise to these scenarios, given that we utilized a pool of twenty clinicians with a mean experience of 11.5 years. This may not adequately represent the diverse set of users seeking to benefit from LLMs as a diagnostic aid. Further studies are warranted in an array of more realistic clinical scenarios, with a wider range of clinical users that might range from medical students to allied health professionals. The underlying mechanism of assistance also merits further study and might be an important factor for optimising the assistive effect of LLMs, particularly given their conversational and interactive nature. For example, our study did not explore the impact of LLMs on the clinical reasoning process.

## 8 Limitations

There are limitations to this evaluation. While based on real-world cases, the clinical pathology case presentation format and input into the model does differ in important ways from how a clinician would evaluate a patient and generate their differential diagnosis at the outset of a clinical encounter. The case reports are created as "puzzles" with enough clues that should enable a specialist to reason towards the final diagnosis. At the beginning of a clinician encounter, it would be challenging to create such a concise, complete and coherent case report. Case reports in the NEJM style would not be available when at intake. Similarly, these cases were selected to represent challenging cases instead of common conditions (i.e., 'zebras' as opposed to 'horses' in clinical parlance). As such, our evaluation does not directly indicate the results of or suggest that clinicians should leverage the assistive capabilities of an LLM for typical cases seen on a daily basis.

In terms of modalities, the case reports include both images and tables. The clinicians had access to these in the redacted case reports. However, the LLM only had access to the main body of the text. Though the LLM for DDx outperformed the clinicians despite this limitation, it is unknown whether and how much this gap would widen if the LLM had access to the figures and tables. Early evidence suggests the effect might be case/context dependent $[13]$.

The study highlighted some weaknesses of the existing LLM. Specifically, one clinician highlighted that "It was most helpful for simpler cases that were specific keywords or pathognomonic signs." (C3) but that for more complex cases it still tended to draw conclusions from isolated symptoms rather than viewing the case holistically. Considering the significance of assessing challenging cases, the NEJM CPC case reports will likely serve as a useful dataset for continued benchmarking as LLMs improve in performance.

## 9 Conclusion

Generating a DDx is a critical step in clinical case management, and the capabilities of LLMs present new opportunities for assistive tooling to help with this task. Our randomized study showed that the LLM for DDx was a helpful AI tool for DDx generation for generalist clinicians. Clinician participants indicated utility for learning and education, and additional work is needed to understand suitability for clinical settings.

## Acknowledgments

This project was an extensive collaboration between many teams at Google Research and Google DeepMind. We thank Ayush Jain, Rory Sayres, Sami Lachgar, Lauren Winer, Maggie Shiels, Brett Harfield, Si Wai Man, Preeti Singh, Annisah Um'rani, Bradley Green, and Philip Mansfield for their valuable insights and feedback during our research. We are also grateful to Micheal Howell, Meredith Morris, Celeste Grade, Karen DeSalvo, Zoubin Ghahramani, James Manyika, and Jeff Dean for their support during the course of this project. Finally, we extend our sincere thanks to the Massachusetts Medical Society group for the support and partnership.

## Competing interests

This study was funded by Alphabet Inc and/or a subsidiary thereof ('Alphabet'). All authors are employees of Alphabet and may own stock as part of the standard compensation.

## References

1. Szolovits, P. \& Pauker, S. G. Categorical and probabilistic reasoning in medical diagnosis. Artificial Intelligence 11, 115-144 (1978).
2. Liu, Y., Jain, A., Eng, C., Way, D. H., Lee, K., Bui, P., Kanada, K., de Oliveira Marinho, G., Gallegos, J., Gabriele, S., et al. A deep learning system for differential diagnosis of skin diseases. Nature medicine 26, 900-908 (2020).
3. Rauschecker, A. M., Rudie, J. D., Xie, L., Wang, J., Duong, M. T., Botzolakis, E. J., Kovalovich, A. M., Egan, J., Cook, T. C., Bryan, R. N., et al. Artificial intelligence system approaching neuroradiologist-level differential diagnosis accuracy at brain MRI. Radiology 295, 626-637 (2020).
4. Balas, M. \& Ing, E. B. Conversational ai models for ophthalmic diagnosis: Comparison of chatgpt and the isabel pro differential diagnosis generator. JFO Open Ophthalmology 1, 100005 (2023).
5. Kanjee, Z., Crowe, B. \& Rodman, A. Accuracy of a Generative Artificial Intelligence Model in a Complex Diagnostic Challenge. JAMA (2023).
6. OpenAI. GPT-4 Technical Report 2023. arXiv: 2303.08774 [cs.CL].
7. Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023).
8. Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., Gallé, M., et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100 (2022).
9. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
10. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large language models encode clinical knowledge. Nature, 1-9 (2023).
11. Nori, H., King, N., McKinney, S. M., Carignan, D. \& Horvitz, E. Capabilities of gpt-4 on medical challenge problems. arXiv preprint arXiv:2303.13375 (2023).
12. Eriksen, A. V., Moller, S. \& Ryg. Use of GPT-4 to Diagnose Complex Clinical Cases. NEJM AI (2023).
13. Buckley, T., Diao, J. A., Rodman, A. \& Manrai, A. K. Accuracy of a Vision-Language Model on Challenging Medical Cases 2023. arXiv: 2311.05591 [cs.CV]
14. Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L. \& Mark, R. G. MIMIC-III, a freely accessible critical care database. Scientific data 3, 1-9 (2016).
15. Bond, W. F., Schwartz, L. M., Weaver, K. R., Levick, D., Giuliano, M. \& Graber, M. L. Differential diagnosis generators: an evaluation of currently available computer programs. Journal of general internal medicine 27, 213-219 (2012).
16. Ledley, R. S. \& Lusted, L. B. Reasoning foundations of medical diagnosis: symbolic logic, probability, and value theory aid our understanding of how physicians reason. Science 130, 9-21 (1959).
17. Hong, J., Levine, S. \& Dragan, A. Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations. arXiv preprint arXiv:2311.05584 (2023).
18. Kossen, J., Cangea, C., Vértes, E., Jaegle, A., Patraucean, V., Ktena, I., Tomasev, N. \& Belgrave, D. Active Acquisition for Multimodal Temporal Data: A Challenging Decision-Making Task. arXiv preprint arXiv:2211.05039 (2022).
19. Mackie, I., Chatterjee, S. \& Dalton, J. Generative Relevance Feedback with Large Language Models. arXiv preprint arXiv:2304.13157 (2023).
20. Fritz, P., Kleinhans, A., Raoufi, R., Sediqi, A., Schmid, N., Schricker, S., Schanz, M., Fritz-Kuisle, C., Dalquen, P., Firooz, H., et al. Evaluation of medical decision support systems (DDX generators) using real medical cases of varying complexity and origin. BMC Medical Informatics and Decision Making 22, 254 (2022).
21. Miller, R. A., Pople Jr, H. E. \& Myers, J. D. in Computer-assisted medical decision making 139-158 (Springer, 1985).
22. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H. \& Gao, J. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. arXiv preprint arXiv:2306.00890 (2023).
23. Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.-C., Carroll, A., Lau, C., Tanno, R., Ktena, I., et al. Towards generalist biomedical ai. arXiv preprint arXiv:2307.14334 (2023).
24. Muehlematter, U. J., Daniore, P. \& Vokinger, K. N. Approval of artificial intelligence and machine learning-based medical devices in the USA and Europe (2015-20): a comparative analysis. The Lancet Digital Health 3, e195-e203 (2021).
25. Roschewitz, M., Khara, G., Yearsley, J., Sharma, N., James, J. J., Ambrózay, É., Heroux, A., Kecskemethy, P., Rijken, T. \& Glocker, B. Automatic correction of performance drift under acquisition shift in medical image classification. Nature Communications 14, 6608 (2023).
26. Obermeyer, Z., Powers, B., Vogeli, C. \& Mullainathan, S. Dissecting racial bias in an algorithm used to manage the health of populations. Science 366, 447-453 (2019).
27. Seyyed-Kalantari, L., Zhang, H., McDermott, M. B., Chen, I. Y. \& Ghassemi, M. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature medicine 27, 2176-2182 (2021).
28. Samorani, M., Harris, S. L., Blount, L. G., Lu, H. \& Santoro, M. A. Overbooked and overlooked: machine learning and racial bias in medical appointment scheduling. Manufacturing \& Service Operations Management 24, 2825-2842 (2022).
29. Gaube, S., Suresh, H., Raue, M., Merritt, A., Berkowitz, S. J., Lermer, E., Coughlin, J. F., Guttag, J. V., Colak, E. \& Ghassemi, M. Do as AI say: susceptibility in deployment of clinical decision-aids. NPJ digital medicine 4, 31 (2021).
30. Umapathi, L. K., Pal, A. \& Sankarasubbu, M. Med-halt: Medical domain hallucination test for large language models. arXiv preprint arXiv:2307.15343 (2023).
31. Sikka, R., Morath, J. M. \& Leape, L. The quadruple aim: care, health, cost and meaning in work 2015.
32. Cai, C. J., Winter, S., Steiner, D., Wilcox, L. \& Terry, M. " Hello AI": uncovering the onboarding needs of medical practitioners for human-AI collaborative decision-making. Proceedings of the ACM on Human-computer Interaction $\mathbf{3}$, $1-24(2019)$.
33. Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X. \& Huang, X. Do Large Language Models Know What They Don't Know? arXiv preprint arXiv:2305.18153 (2023).
34. Dvijotham, K., Winkens, J., Barsbey, M., Ghaisas, S., Stanforth, R., Pawlowski, N., Strachan, P., Ahmed, Z., Azizi, S., Bachrach, Y., et al. Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians. Nature Medicine 29, 1814-1820 (2023).
![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-14.jpg?height=1850&width=1066&top_left_y=369&top_left_x=518)

Figure $1 \mid$ NEJM Clinicopathological Conference Case Reports. History of Present Illness, Admission Labs and Admission Imaging sections were included in the redacted version presented to generalist clinicians for producing a DDx. The LLM had access to only the History of Present Illness. Specialist clinicians evaluating the quality of the DDx had access to the full (unredacted) case report including the expert differential discussion.

![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-15.jpg?height=751&width=1634&top_left_y=291&top_left_x=251)

Figure 2 | The LLM for DDx User Interface. The history of the present illness (text only) was pre-populated in the user interface (A) with an initial suggested prompt to query the LLM (B). Following this prompt and response, the user was free to enter any additional follow-up questions (C). The case shown in this figure is a mock case selected for illustrative purposes only.
![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-15.jpg?height=836&width=1632&top_left_y=1340&top_left_x=252)

Figure $3 \mid$ Experimental Design. To evaluate the LLM's ability to generate DDx lists and aid clinicians with their DDx generation, we designed a two-stage reader study. First, clinicians with access only to the case presentation completed DDx lists without using any assistive tools. Second, the clinicians completed DDx lists with access either to Search engines and other resources, or to LLM in addition to these tools. Randomization was employed such that every case was reviewed by two clinicians, one with LLM assistance and one without. These DDx lists were then evaluated by a specialist who had access to the full case and expert commentary on the differential diagnosis, but who was blinded to whether and what assistive tool was used.

![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-16.jpg?height=1072&width=1632&top_left_y=278&top_left_x=237)

Quality Score: Inclusion of the Final Diagnosis

Figure $4 \mid$ Evaluation of the quality of generalists" DDx lists. (a) DDx Quality Score based on the question: "How close did the differential diagnoses (DDx) come to including the final diagnosis?" (b) DDx Comprehensiveness Score based on the question: "Using your differential diagnosis list as a bench mark/gold standard, how comprehensive are the differential lists from each of the experts?" (c) DDx Appropriateness Score based on the question: "How appropriate was each of the differential diagnosis lists from the different medical experts compared the differential list that you just produced?" In all cases, the LLM and clinicians assisted by the LLM scored the highest overall. Numbers reflect the number of cases (out of 302). Note: The clinicians could answer "I am not sure" in response to these questions, in a very small number $(<1 \%)$ of cases they used that option.
![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-16.jpg?height=558&width=1488&top_left_y=1770&top_left_x=318)

Figure 5 | Top-n Accuracy. (left) The percentage of DDx lists with the final diagnosis through human evaluation. (right) The percentage of $\mathrm{DDx}$ lists with the final diagnosis through automated evaluation.

![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-17.jpg?height=588&width=1651&top_left_y=449&top_left_x=237)

Figure 6 | Sankey diagram showing effect of assistance. (a) in the LLM arm, 73 cases had the final diagnosis in the DDx list after assistance that did not contain it before, (b) in the Search arm this was 37 cases. In both cases, a small minority (LLM for $\mathrm{DDx}$ arm $=11$, Search arm $=12$ ) a DDx list with the final diagnosis before assistance did not contain it afterwards.

![](https://cdn.mathpix.com/cropped/2024_06_04_4e0771bc02338f0e11f4g-17.jpg?height=515&width=739&top_left_y=1645&top_left_x=693)

Figure 7 | Top-n Accuracy. Comparison of the percentage of DDx lists with the final diagnosis for our LLM for DDx vs GPT-4 for 70 cases.

</end of paper 4>


