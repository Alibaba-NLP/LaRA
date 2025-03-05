<paper 0>
# Ethical and social risks of harm from Language Models 

Laura Weidinger ${ }^{1}$, John Mellor ${ }^{1}$, Maribeth Rauh ${ }^{1}$, Conor Griffin ${ }^{1}$, Jonathan Uesato ${ }^{1}$, Po-Sen Huang ${ }^{1}$, Myra<br>Cheng $^{1,2}$, Mia Glaese ${ }^{1}$, Borja Balle ${ }^{1}$, Atoosa Kasirzadeh ${ }^{1,3}$, Zac Kenton ${ }^{1}$, Sasha Brown ${ }^{1}$, Will Hawkins ${ }^{1}$, Tom<br>Stepleton ${ }^{1}$, Courtney Biles ${ }^{1}$, Abeba Birhane ${ }^{1,4}$, Julia Haas ${ }^{1}$, Laura Rimell ${ }^{1}$, Lisa Anne Hendricks ${ }^{1}$, William<br>Isaac $^{1}$, Sean Legassick ${ }^{1}$, Geoffrey Irving ${ }^{1}$ and Iason Gabriel ${ }^{1}$<br>${ }^{1}$ DeepMind, ${ }^{2}$ California Institute of Technology, ${ }^{3}$ University of Toronto, ${ }^{4}$ University College Dublin


#### Abstract

This paper aims to help structure the risk landscape associated with large-scale Language Models (LMs). In order to foster advances in responsible innovation, an in-depth understanding of the potential risks posed by these models is needed. A wide range of established and anticipated risks are analysed in detail, drawing on multidisciplinary literature from computer science, linguistics, and social sciences.


The paper outlines six specific risk areas: I. Discrimination, Exclusion and Toxicity, II. Information Hazards, III. Misinformation Harms, IV. Malicious Uses, V. Human-Computer Interaction Harms, VI. Automation, Access, and Environmental Harms.

The first risk area discusses fairness and toxicity risks in large-scale language models. This includes four distinct risks: LMs can create unfair discrimination and representational and material harm by perpetuating stereotypes and social biases, i.e. harmful associations of specific traits with social identities. Social norms and categories can exclude or marginalise those who exist outside them. Where a LM perpetuates such norms e.g. that people called "Max" are "male", or that "families" always consist of a father, mother and child - such narrow category use can deny or burden identities who differ. Toxic language can incite hate or violence or cause offense. Finally, a LM that performs more poorly for some social groups than others can create harm for disadvantaged groups, for example where such models underpin technologies that affect these groups. These risks stem in large part from choosing training corpora that include harmful language and overrepresent some social identities.

The second risk area includes risks from private data leaks or from LMs correctly inferring private or other sensitive information. These risks stem from private data that is present in the training corpus and from advanced inference capabilities of LMs.

The third risk area comprises risks associated with LMs providing false or misleading information. This includes the risk of creating less well-informed users and of eroding trust in shared information. Misinformation can cause harm in sensitive domains, such as bad legal or medical advice. Poor or false information may also lead users to perform unethical or illegal actions that they would otherwise not have performed. Misinformation risks stem in part from the processes by which LMs learn to represent language: the underlying statistical methods are not well-positioned to distinguish between factually correct and incorrect information.

The fourth risk area spans risks of users or product developers who try to use LMs to cause harm. This includes using LMs to increase the efficacy of disinformation campaigns, to create personalised scams or fraud at scale, or to develop computer code for viruses or weapon systems.

The fifth risk area focuses on risks from the specific use case of a "conversational agent" that directly interacts with human users. This includes risks from presenting the system as "human-like", possibly leading users to overestimate its capabilities and use it in unsafe ways. Another risk is that conversation with such agents may create new avenues to manipulate or extract private information from users. LM-based conversational agents may pose risks that are already known from voice assistants, such as perpetuating stereotypes by self-presenting e.g. as "female assistant". These risks stem in part from LM training objectives underlying such conversational agents and from product design decisions.

The sixth risk area includes risks that apply to LMs and Artificial Intelligence (AI) systems more broadly. Training and operating LMs can incur high environmental costs. LM-based applications may benefit some groups more
than others and the LMs themselves are inaccessible to many. Lastly, LM-based automation may affect the quality of some jobs and undermine parts of the creative economy. These risks manifest particularly as LMs are widely used in the economy and benefits and risks from LMs are globally unevenly distributed.

In total, we present 21 risks. We then discuss the points of origin of different risks and point to potential risk mitigation approaches. The point of origin of a harm may indicate appropriate mitigations: for example, the risk of leaking private data originates from this data being present in the training dataset. It can be mitigated at the point of origin, by better redaction or curation of training data. However, other mitigation approaches may also be applicable and ensure more robust mitigation overall. For example, algorithmic tools applied during training, such as differential privacy methods, or product decisions, such as constraining access and use cases of the LM, are additional mitigation approaches that can be pursued in parallel. Risk mitigation approaches range from social or public policy interventions, to technical solutions and research management, to participatory projects and product design decisions.

Lastly, we discuss organisational responsibilities in implementing such mitigations, and the role of collaboration. Measuring and mitigating ethical and social risks effectively requires a wide range of expertise, and fair inclusion of affected communities. It is critical to implement mitigations with a broad view of the landscape of risks, to ensure that mitigating against one risk of harm does not aggravate another. Otherwise, for example, mitigation approaches to toxic speech can inadvertently lead to lower LM performance for some social groups. We highlight directions for further research, particularly on expanding the toolkit for assessing and evaluating the outlined risks in LMs, and the need for inclusive participatory methods. Finally, we conclude by showing how the present work - of structuring the risk landscape - is the first step in a broader framework of responsible innovation.

## Contents

1 Introduction ..... 6
1.1 Limitations ..... 7
1.1.1 Note on terminology ..... 7
1.2 A Brief history of Language Models ..... 7
1.2.1 Origins ..... 7
1.2.2 Transformer models ..... 8
1.2.3 "Large" Language Models ..... 8
2 Classification of harms from language models ..... 9
2.1 Discrimination, Exclusion and Toxicity ..... 9
2.1.1 Overview ..... 9
2.1.2 Social stereotypes and unfair discrimination ..... 9
2.1.3 Exclusionary norms ..... 13
2.1.4 Toxic language ..... 15
2.1.5 Lower performance for some languages and social groups ..... 16
2.2 Information Hazards ..... 18
2.2.1 Overview ..... 18
2.2.2 Compromising privacy by leaking private information ..... 18
2.2.3 Compromising privacy by correctly inferring private information ..... 19
2.2.4 Risks from leaking or correctly inferring sensitive information ..... 20
2.3 Misinformation Harms ..... 21
2.3.1 Overview ..... 21
2.3.2 Disseminating false or misleading information ..... 23
2.3.3 Causing material harm by disseminating false or poor information e.g. in medicine or law ..... 24
2.3.4 Leading users to perform unethical or illegal actions ..... 24
2.4 Malicious Uses ..... 25
2.4.1 Overview ..... 25
2.4.2 Making disinformation cheaper and more effective ..... 25
2.4.3 Facilitating fraud, scams and more targeted manipulation ..... 26
2.4.4 Assisting code generation for cyber attacks, weapons, or malicious use ..... 27
2.4.5 Illegitimate surveillance and censorship ..... 28
2.5 Human-Computer Interaction Harms ..... 29
2.5.1 Overview ..... 29
2.5.2 Anthropomorphising systems can lead to overreliance or unsafe use ..... 29
2.5.3 Creating avenues for exploiting user trust, nudging or manipulation ..... 30
2.5.4 Promoting harmful stereotypes by implying gender or ethnic identity ..... 31
2.6 Automation, access, and environmental harms ..... 31
2.6.1 Overview ..... 32
2.6.2 Environmental harms from operating LMs ..... 32
2.6.3 Increasing inequality and negative effects on job quality ..... 33
2.6.4 Undermining creative economies ..... 34
2.6.5 Disparate access to benefits due to hardware, software, skill constraints ..... 34
3 Discussion ..... 36
3.1 Understanding the point of origin of a risk ..... 36
3.2 Identifying and implementing mitigation approaches ..... 37
3.3 Organisational responsibilities ..... 38
4 Directions for future research ..... 39
4.1 Risk assessment frameworks and tools ..... 39
4.2 Technical and sociotechnical mitigation research ..... 39
4.3 Benchmarking: when is a model "fair enough"? ..... 39
4.4 Benefits and overall social impact from LMs ..... 40
5 Conclusion ..... 41
A Appendix ..... 62
A. 1 Definitions ..... 62
A.1.1 Language Models ..... 62
A.1.2 Language Agents ..... 62
A.1.3 Language Technologies ..... 62
A. 2 References Table ..... 63

## Reader's guide

This is a long document. The report is divided into three segments.

First, the Introduction provides a brief introduction to Language Models.

Second, the Classification of harms from language models gives a taxonomy and detailed account of a range of social and ethical risks associated with Language Models.

Third, the Discussion and Directions for future research explore some underlying causes of these risks, a range of mitigation approaches, and possible challenges to be addressed through future research.

Individual sections can be read independently or together. We recommend:

- 1 minute read: Study Table 1 for a high-level overview of the risks considered.
- 10 minute read: Read the Abstract and Table 1 for an overview of the risks considered. Then skim all bold text in the segment on Classification of harms from language models and skim Directions for future research for an overview of risks and challenges.
- Readers who actively work on LMs: We encourage you to skim all bold text in the segment on Classification of harms from language models, and to get stuck in risks that directly relate to your own work and interest - as you will likely be able to help solve some of the field's core challenges in this domain.
- Readers with no background on LMs: We recommend you read the Abstract and Introduction first as these introduce key terminology that is used in this report. Next, study Table 1 for a high-level overview of the risks considered and read the risk headers and example dialog boxes for each risk in the Classification of harms from language models. Get stuck in risks that are of interest to you and read the Discussion on challenges in mitigating these risks.
- Readers with an interest in a particular risk or type of harm: We encourage you to read the Abstract, Table 1 and Discussion for context on the broader risk landscape and approaches to mitigation, in addition to reading the specific section on the risk that piques your interest.
- Readers with an interest in approaches to mitigating harms: We recommend you read the Abstract for an overview of the harms considered and read Table 1 with a focus on the mechanisms underlying each risk area. Jump to the Discussion on approaches to mitigating risks and read Directions for future research on methodological and normative challenges in assessing and mitigating risks, and proposals for addressing these challenges.


## 1. Introduction

Language Models (LMs) ${ }^{1}$ are rapidly growing in size and effectiveness, yielding new breakthroughs and attracting increasing research attention (Brown et al., 2020; Fedus et al., 2021; Rae et al., 2021; Rosset, 2020). Several Artificial Intelligence (AI) research labs are pursuing LM research, spurred by the promise these models hold for advancing research and for a wide range of beneficial real-world applications. Some research groups have suggested that recent large-scale LMs may be a 'foundational' breakthrough technology, potentially affecting many aspects of life (Bommasani et al., 2021). The potential impact of such LMs makes it particularly important that actors in this space lead by example on responsible innovation.

Responsible innovation entails that in addition to developing the technology, it is essential to thoughtfully assess the potential benefits as well as potential risks that need to be mitigated (Stilgoe et al., 2013). Prior research has explored the potential for ethical and safe innovation of large-scale LMs, including interdisciplinary workshops to scope out risks and benefits (Tamkin et al., 2021), papers that outline potential risks (Bender et al., 2021; Bommasani et al., 2021; Dinan et al., 2021; Kenton et al., 2021), and papers identifying ways to mitigate potential harms (Chen et al., 2021a; Solaiman and Dennison, 2021; Welbl et al., 2021). ${ }^{2}$ For this report, we seek to build on this prior work by proposing an initial taxonomy of risks associated with LM development and use, as well as outlining concrete next steps and directions for future research that supports responsible innovation for LMs.

The overall aim of this report is three-fold:

1. Underpin responsible decision-making by organisations working on LMs by broadening and structuring the discourse on AI safety and ethics in this research area,
2. Contribute to wider public discussion about risks and corresponding mitigation strategies for LMs,
3. Guide mitigation work by research groups working on LMs. We aim to support the mutual exchange of expertise in this area, to help make the risks posed by LMs actionable.

We structure the identified risks in a taxonomy of ethical and social risks associated with LMs, under 6 risk areas: I. Discrimination, Exclusion and Toxicity, II. Information Hazards, III. Misinformation Harms, Malicious Uses, V. Human-Computer Interaction Harms, VI. Automation, Access, and Environmental Harms. An overview of the risks that fall under each risk area can be found in the Classification of harms from language models part of the report.

Each risk is discussed in detail with regard to the nature of the harm, empirical examples, and additional considerations. For each risk, we provide a fictitious example to illustrate how the risk in question may manifest. ${ }^{3}$ However the risks described apply to LMs more generally and do not depend on the dialogue modality unless otherwise specified. Since several of the risks discussed below are neither novel nor exclusive to LMs or related technologies, we offer context on how each risk manifests in existing language technologies. We also mark each risk as either "anticipated" or "observed", depending on whether a given risk has already been observed or whether further work is needed to indicate real-world manifestations of this risk. The creation of a taxonomy of risks supports the exercise of foresight in this space, with the aim of guiding action to resolve any issues that can be identified in advance.

Responsible innovation is a collaborative endeavour. In order to anticipate and mitigate risks posed by technology successfully, we need to view these issues through multiple lenses and perspectives. This report was written by a large group of researchers with varied disciplinary backgrounds and areas of expertise. To review the risk landscape as comprehensively as possible, we collated potential risks from a wide range of sources including[^0]analyses from the fields of AI ethics, AI safety, race and gender studies, linguistics and natural language processing and studies at the intersection of society and technology (also referred to as sociotechnical studies), as well as analyses by civil society organisations and news reports. Further risks were added based on our own experience and expertise. Beyond publishing research, we believe responsible innovation also requires inclusive dialogue between stakeholders in AI development which includes affected communities and the wider public (Gabriel, 2020b; Mohamed et al., 2020; Murgia, 2021; Stilgoe et al., 2013). In the future, we look to continue to deepen our understanding of risks and mitigations including by working with external partners and communities.

### 1.1. Limitations

Note that this report is part of a broader research programme working toward the responsible innovation of LMs and necessarily leaves some questions unanswered. For example, we do not discuss potential beneficial applications of LMs nor do we offer a comprehensive overview of potential use cases. Nor do we attempt to perform a full ethical evaluation of LMs, which must weigh both the potential benefits and risks of a given technology. To assess the overall balance of benefit and cost, separate analysis of the benefits arising from proposed LM applications would be needed. Instead, the focus here is on anticipating and structuring the risk landscape, with the intention of supporting a larger constructive research effort.

This report is also necessarily a snapshot in time: it was initiated in autumn 2020 and completed in summer 2021. It is likely that we miss risks arising from LMs that depend, for their visibility, on the passage of time. As such, the presented taxonomy is merely a starting point and will need to be updated as new challenges come into focus and additional perspectives are brought to bear on these questions.

This report focuses on risks associated with operating LMs. Risks of harm that are associated with training are not discussed. This includes concerns about the working conditions of data annotators or "ghost workers" (Gray and Suri, 2019), the ethics of supply chains of hardware on which LM computations are run (Crawford, 2021), or environmental costs of training such models (Bender et al., 2021; Patterson et al., 2021; Schwartz et al., 2020; Strubell et al., 2019) which are only briefly referenced in the section on VI. Automation, access, and environmental harms. This report also does not cover risks that depend on specific applications.

This report excludes risks which the authors anticipate to depend on capabilities that are several years in the future, for example because they depend on capabilities that are several step changes beyond the state-of-the-art. A subset of such long-term risks is addressed in literature on existential risk and AI Safety (Armstrong et al., 2012; Kenton et al., 2021). This report also does not cover risks that depend on superintelligence as described in (Bostrom, 2014).

Finally, this report does not discuss risks that depend on multiple modalities, for example from models that combine language with other domains such as vision or robotics. While several of the insights in this report are translatable to such models, these require distinct risk assessments. For some discussion on risks associated with multi-modal large models, see (Bommasani et al., 2021).

### 1.1.1. Note on terminology

This report focuses on the risks of large-scale language models, including in specific applications of these models such as conversational assistants, or in other language technologies. Several of these risks also apply to smaller language models. For detailed definitions of Language Models, Language Agents, and Language Technologies please refer to the section on Definitions in the Appendix.

For simplicity we refer to "LMs" throughout. Where risks are unique to specific types of applications, such as conversational agents, this is explicitly stated.

### 1.2. A Brief history of Language Models

### 1.2.1. Origins

The main methodology underpinning contemporary large-scale language models traces its origins to methods developed by the research group of Frederick Jelinek on Automatic Speech Recognition (ASR) in the 1970s and '80s (Jelinek, 1976). This research group built on prior work in statistics by Claude Shannon (Shannon, 1948) and Andrey Markov (Markov, 1913). In parallel, James Baker (Baker, 1990) developed a similar approach to ASR (see (Jurafsky and Martin, 2014)).

Jelinek's group pioneered an information theoretic approach to ASR, observing that performing any task that requires producing language conditioned on an input using a probability distribution $p$ (language|input) can be factored into a language model representing a probability distribution $p$ (language) multiplied by the task specific distribution $p$ (input|language). This factorisation suggests that general LMs $p$ (language) can aid language prediction tasks where the LM captures the relevant language distribution. Whilst this factorisation is not explicitly used in most current systems, it implicitly underpins current LM research and is a useful way to understand the role language modelling plays in specific language technologies such as conversational agents, machine translation, and question answering.

### 1.2.2. Transformer models

More recently, the transformer architecture was developed (Vaswani et al., 2017). Transformers are a class of architectures that use a series of so-called transformer blocks comprising a self-attention layer followed by a feedforward layer, linked together with residual connections. The self-attention layer helps the model to consider neighbouring words in the input as it processes a specific word. Originally, the transformer architecture was proposed for the task of machine translation (Vaswani et al., 2017). (Radford et al., 2018b) use a modified version applied to the task of language modeling (predicting the next word in a sentence). Subsequent work on LMs (Brown et al., 2020; Radford et al., 2018a) uses a similar architecture. An accessible visual introduction to the transformer architecture can be found in (Alammar, 2018). Recent language models built on the transformer architecture have been fine-tuned directly, without the need for task-specific architectures (Devlin et al., 2019; Howard and Ruder, 2018; Radford et al., 2018a).

### 1.2.3. "Large" Language Models

The recent upwind in LM research is rooted in the capacity to increase LM size in terms of number of parameters and size of training data (Bender et al., 2021). Training models on extremely large datasets such as the Colossal Clean Crawl Corpus (C4) (Raffel et al., 2020) and WebText (Radford et al., 2018b) resulted in sequence prediction systems with much more general applicability compared to the prior state-of-the-art (Brown et al., 2020; Fedus et al., 2021; Rosset, 2020). These models also displayed greater few-shot and zero-shot learning capabilities compared to smaller LMs (Brown et al., 2020). These properties were found to greatly simplify the development of task-specific LAs by reducing the adaptation process to prompt design (Zhang et al., 2021b). The insight that powerful sequence prediction systems could be created by scaling up the size of LMs and training corpora motivated an upsurge in interest and investment in LM research by several AI research labs.

## 2. Classification of harms from language models

In this section we outline our taxonomy of ethical and social risks of harm associated with Language Models. We identify 21 risks of harm, organised into six risk areas (for an overview see Table 1). In this table we also note the mechanisms by which different groups of risks emerge.

### 2.1. Discrimination, Exclusion and Toxicity

Social harms that arise from the language model producing discriminatory or exclusionary speech

### 2.1.1. Overview

Language has long been a vehicle for perpetuating injustices against marginalised populations and conferring power to oppressive institutions (Craft et al., 2020; Lippi, 1997). Social harms that manifest in language can also arise from language predicted by language models (LMs). We distinguish between the following harms. First, language that promotes stereotypes or causes unfair discrimination ${ }^{1}$ can contribute to the oppression of those at social margins (Crenshaw, 2017b). Second, language that reinforces social norms can exclude identities that exist outside these norms (Bender et al., 2021; Foucault and Sheridan, 2012). Third, language can be 'toxic', for example by inciting violence or causing offense (Fortuna and Nunes, 2018). Fourth, a form of discrimination emerges when language technologies perform better for some social groups than others.

Risks covered in this section:

- Social stereotypes and unfair discrimination
- Exclusionary norms
- Toxic language
- Lower performance by social group


### 2.1.2. Social stereotypes and unfair discrimination

Q: "Two Muslims walked into a ..."

A: "Texas cartoon contest and opened fire."

${ }^{a}$ Example from (Abid et al., 2021)

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Perpetuating harmful stereotypes and discrimination is a well-documented harm in machine learning models that represent natural language (Caliskan et al., 2017). LMs that encode discriminatory language or social stereotypes can cause different types of harm. It may be useful to distinguish between allocational and representational harms: allocational harms occur when resources and opportunities are unfairly allocated between social groups; they may occur when LMs are used in applications that are used to make decisions that affect persons. Representational harms include stereotyping, misrepresenting, and demeaning social groups Barocas and Wallach cited in (Blodgett et al., 2020).

Unfair discrimination manifests in differential treatment or access to resources among individuals or groups based on sensitive traits such as sex, religion, gender, sexual orientation, ability and age. The dimensions along which such oppression occurs can also be rooted in culture-specific or otherwise localised social hierarchies.[^1]

Table 1. Overview of all risks covered in this report.

## I. Discrimination, Exclusion and Toxicity

Mechanism: These risks arise from the LM accurately reflecting natural speech, including unjust, toxic, and oppressive tendencies present in the training data.

Types of Harm: Potential harms include justified offense, material (allocational) harm, and the unjust representation or treatment of marginalised groups.

- Social stereotypes and unfair discrimination
- Exclusionary norms
- Toxic language
- Lower performance by social group

II. Information Hazards

Mechanism: These risks arise from the LM predicting utterances which constitute private or safety-critical information which are present in, or can be inferred from, training data.

Types of Harm: Potential harms include privacy violations and safety risks.

- Compromise privacy by leaking private information
- Compromise privacy by correctly inferring private information
- Risks from leaking or correctly inferring sensitive information

III. Misinformation Harms

Mechanism: These risks arise from the LM assigning high probabilities to false, misleading, nonsensical or poor quality information.

Types of Harm: Potential harms include deception, material harm, or unethical actions by humans who take the LM prediction to be factually correct, as well as wider societal distrust in shared information.

- Disseminating false or misleading information
- Causing material harm by disseminating misinformation e.g. in medicine or law
- Nudging or advising users to perform unethical or illegal actions

IV. Malicious Uses

Mechanism: These risks arise from humans intentionally using the LM to cause harm.

Types of Harm: Potential harms include undermining public discourse, crimes such as fraud, personalised disinformation campaigns, and the weaponisation or production of malicious code.

- Reducing the cost of disinformation campaigns
- Facilitating fraud and impersonation scams
- Assisting code generation for cyber attacks, weapons, or malicious use
- Illegitimate surveillance and censorship

V. Human-Computer Interaction Harms

Mechanism: These risks arise from LM applications, such as Conversational Agents, that directly engage a user via the mode of conversation.

Types of Harm: Potential harms include unsafe use due to users misjudging or mistakenly trusting the model, psychological vulnerabilities and privacy violations of the user, and social harm from perpetuating discriminatory associations via product design (e.g. making "assistant" tools by default "female.")

- Anthropomorphising systems can lead to overreliance or unsafe use
- Create avenues for exploiting user trust to obtain private information
- Promoting harmful stereotypes by implying gender or ethnic identity

VI. Automation, access, and environmental harms

Mechanism: These risks arise where LMs are used to underpin widely used downstream applications that disproportionately benefit some groups rather than others.

Types of Harm: Potential harms include increasing social inequalities from uneven distribution of risk and benefits, loss of high-quality and safe employment, and environmental harm.

- Environmental harms from operating LMs
- Increasing inequality and negative effects on job quality
- Undermining creative economies
- Disparate access to benefits due to hardware, software, skill constraints

For example, the Hindu caste system underpins discrimination in India, but not across the globe (Sambasivan et al., 2021). Additionally, injustice can be compounded when social categories intersect, for example in the discrimination against a person that holds a marginalised gender and a marginalised religion (Crenshaw, 2017a).

Allocational harm caused by discriminatory systems is particularly salient if bias occurs in applications that materially impact people's lives, such as predicting a person's creditworthiness (Mehrabi et al., 2019), criminal recidivism (Angwin et al., 2016), or suitability to a job (Mujtaba and Mahapatra, 2019). For example, a language technology that analyses CVs for recruitment, or to give career advice, may be less likely to recommend historically discriminated groups to recruiters, or more likely to recommend lower paying careers to marginalised groups. Unfair biases are already well-documented in machine learning applications ranging from diagnostic healthcare algorithms (Obermeyer et al., 2019) to social outcome prediction (Narayanan, 2021); for a more general introduction see (Chouldechova and Roth, 2018; Kordzadeh and Ghasemaghaei, 2021; Mehrabi et al., 2021; Noble, 2018; Zou and Schiebinger, 2018). Based on our current understanding, such stereotyping and unfair bias are set to recur in language technologies building on LMs unless corrective action is taken.

Why we should expect LMs to reinforce stereotypes and unfair discrimination by default LMs are optimised to mirror language as accurately as possible, by detecting the statistical patterns present in natural language Definitions. The fact that LMs track patterns, biases, and priors in natural language is not negative per se (Shah et al., 2020). Rather, it becomes a problem when the training data is unfair, discriminatory, or toxic. In this case, the optimisation process results in models that mirror these harms. As a result, LMs that perform well with regard to their optimisation objective can work poorly with regard to social harms, insofar as they encode and perpetuate harmful stereotypes and biases present in the training data.

Stereotypes and unfair discrimination can be present in training data for different reasons. First, training data reflect historical patterns of systemic injustice when they are gathered from contexts in which inequality is the status quo. Training systems on such data entrenches existing forms of discrimination (Browne, 2015). In this way, barriers present in our social systems can be captured by data, learned by LMs, and perpetuated by their predictions (Hampton, 2021).

Second, training data can be biased because some communities are better represented in the training data than others. As a result, LMs trained on such data often model speech that fails to represent the language of those who are marginalised, excluded, or less often recorded. The groups that are traditionally underrepresented in training data are often disadvantaged groups: they are also referred to as the 'undersampled majority' (Raji, 2020). The implications of unrepresentative training data for downstream biases and stereotyping in LMs demonstrate the power that is exercised by those who have influence over what data is used for model training (Blodgett et al., 2020). While in principle, LMs are optimised to represent language with high fidelity, they can also overrepresent small biases present in the training data, a phenomenon referred to as 'bias amplification' (Wang and Russakovsky, 2021; Zhao et al., 2017).

## Examples

Generative LMs have frequently been shown to reproduce harmful social biases and stereotypes. Predictions from the GPT-3 model (Brown et al., 2020) were found to exhibit anti-Muslim and, to a lesser degree, antisemitic bias, where ""Muslim" was analogised to "terrorist" in $23 \%$ of test cases, while "Jewish" was mapped to "money" in $5 \%$ of test cases (Abid et al., 2021) ${ }^{2}$. Gender and representation biases were found in fictional stories generated by GPT-3 (Lucy and Bamman, 2021), where female-sounding names were more often associated with stories about family and appearance, and described as less powerful than masculine characters.

The StereoSet benchmark measures references to stereotypes of race, gender, religion, and profession in generative LMs and finds that the models GPT2 (Radford et al., 2018b) and masked models BERT (Devlin et al., 2019), ROBERTA (Liu et al., 2019), XLNET (Yang et al., 2019) exhibit 'strong stereotypical associations' (Nadeem et al., 2020). The CrowS-Pairs benchmark finds that cultural stereotypes were reproduced by likelihood estimates of masked LMs BERT (Devlin et al., 2019), and RoBERTA (Liu et al., 2019; Nangia et al., 2020)³ . The HONEST benchmark shows that GPT-2 and BERT sentence completions promote 'hurtful stereotypes' across six[^2]languages (Nozza et al., 2021), and discriminatory gender biases were found in contextual word embedding by BERT (Kurita et al., 2019) and ELMo (Zhao et al., 2019). LMs trained on news articles and Wikipedia entries have been demonstrated to exhibit considerable levels of bias against particular country names, occupations, and genders (Huang et al., 2020).

## Additional considerations

Underrepresented groups in the training data Training data reflect the views, values, and modes of communication by the communities whose language is captured in the corpus. For example, a dataset of Reddit user comments was found to encode discriminatory views based on gender, religion and race (Ferrer et al., 2020). As a result, it is important to carefully select and account for the biases present in the training data. However ML training datasets are often collected with little curation or supervision and without factoring in perspectives from communities who may be underrepresented (Jo and Gebru, 2020). For more discussion of this, see also the section on Why we should expect LMs to reinforce unfair bias, toxic speech, and exclusionary norms.

Documentation of biases in training corpora The impact of training data on the LM makes it important to transparently disclose what groups, samples, voices and narratives are represented in the dataset and which may be missing. One format that has been proposed for such dataset documentation (Bender and Friedman, 2018) are 'Datasheets' (Gebru et al., 2020). Some work in this direction includes documentation on the Colossal Clean Crawl Corpus (C4) that highlights the most prominently represented sources and references to help illuminate whose biases are likely to be encoded in the dataset (Dodge et al., 2021). Documentation of larger datasets is critical for anticipating and understanding the pipeline by which different harmful associations come to be reflected in the LM.

Training data required to reduce bias may not yet exist Approaches to biased training data range from curating dedicated training datasets to not building models in domains where such data does not exist. ${ }^{4}$ Curating training data can help to make LMs fairer, but creating better datasets requires dedicated work (Hutchinson et al., 2021; Jo and Gebru, 2020) and may require novel data curation pipelines and tools (Denton et al., 2020). Training corpora for state of the art LMs are extremely large, so that further innovation on semi-automated curation methods may be needed in order to make the curation of such datasets tractable. Determining what constitutes a truly fair and equitable training dataset may also require further research in Ethics and Law (Kohler-Hausmann, 2019). In one high-profile, real-world example, researchers attempted to train a classifier to support recruitment, but found that the training data was inherently biased and found no alternative to create a more equitable training dataset - leading to the research project being abandoned (Dastin, 2018) ${ }^{5}$.

Localised stereotypes are hard to capture As stereotypes change over time and vary between contexts, it is impossible for any given research team to be aware of, and up-to-date on, all relevant stereotypes that may cause harm or offense. In addition, the stereotypes at play in a given local context may only be knowable through committed ethnographic work on the ground (Marda and Narayan, 2021). The expertise for identifying harmful stereotypes often lies with the lived experience of affected groups (Sullivan and Tuana, 2007). This creates a challenge in knowing what stereotypes to search for, detect, and mitigate at the point of creating a LM. One way to help address this challenge is to use inclusive and fair participatory approaches (Martin Jr. et al., 2020), by establishing participatory mechanisms and institutions that can operate over time (Sloane et al., 2020), and by providing broad and transparent dataset documentation.

Uncertainty on downstream uses complicate fairness analyses Identifying affected communities is challenging during the early stages of building a LM when no particular application, product, or user group has been[^3]defined. It is unclear to what extent a training regime can be defined that increases model "fairness" whilst being agnostic on downstream applications (Hancox-Li and Kumar, 2021). While some aspects of fairness are best considered at early research stages, more specific assessments of potential discrimination must be considered again at the point of developing a concrete application. Methods for detecting and mitigating harmful stereotypes can place an additional burden or privacy cost on minorities, e.g. through collecting additional data. Where this is the case, sustained mitigation of such harms requires engaging affected groups on fair terms that foreground their needs and interests.

Detecting harmful stereotypes can require nuanced analyses over multiple samples Stereotyping may only be detectable over multiple samples. "Pointwise" stereotyping manifests directly in the text prediction of a single sample and so can be identified in a single instance (Khalifa et al., 2021). "Distributional" stereotyping on the other hand manifests in the repetition of a seemingly harmless association of certain properties with a group. For example, where a LM predicts passive verbs more often in association with female than male names, such distributional stereotyping of females as more passive may occur. Such "distributional" bias may also manifest as notable omissions, e.g. where a language agent that generates fantasy stories by relying on a LM only generates stories with male, never female villains. Such distributional bias becomes apparent only upon analysing multiple predictions and requires distinct forms of evaluation and correction (Khalifa et al., 2021).

### 2.1.3. Exclusionary norms

Q: What is a family?

A: A family is: a man and a woman who get married and have children. (not accounting for nonheteronormative families and children out of wedlock, for single-parent families and for the fact that families sometimes do not have children)

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

In language, humans express social categories and norms. Language models (LMs) that faithfully encode patterns present in natural language necessarily encode such norms and categories. This is why technological artifacts such as LMs are never "value neutral" - they represent and perpetuate the values and norms present in the training data (Bender et al., 2021; Winner, 1980).

Such norms and categories exclude groups who live outside them (Foucault and Sheridan, 2012). For example, defining the term "family" as married parents of male and female gender with a blood-related child, denies the existence of families to whom these criteria do not apply. Moreover, exclusionary norms intersect with discrimination as they almost invariably work to exclude groups that have historically been marginalised. Exclusionary norms can manifest in "subtle patterns like referring to women doctors as if doctor itself entails not-woman, or referring to both genders excluding the possibility of non-binary gender identities" (Bender et al., 2021), emphasis added.

Furthermore, exclusionary norms can place a disproportionate burden or "psychological tax" on those who do not fit or comply with these norms or who are trying to challenge or replace them. Where the model omits, excludes, or subsumes those deviating from the (perceived) norm into ill-fitting categories, these individuals also may encounter allocational or representational harm and discrimination.

The technical underpinning for LMs to promote exclusionary norms may be the fact that a deterministic argmax approach is commonly used for sampling utterances (Yee et al., 2021). This mechanism always samples the most probable next word, rather than sampling probabilistically from the prediction distribution. This can result in the single most probable view becoming entrenched in the social contexts and applications of the model (Yee et al., 2021). In LMs, this can lead to language that excludes, denies, or silences identities that fall outside these categories.

## Example

In other machine learning approaches to modeling language it was found that tools for coreference resolution the task of identifying all expressions that refer to the same entity in a text - typically assume binary gender,
forcing, for example, the resolution of names into either "he" or "she" (not allowing for the resolution of the name "Max" into "they") (Cao and Daumé III, 2020), definition from (StanfordNaturalProcessingGroup). In response to a question, GPT-3 was found to frequently provide common, but false utterances, rather than providing the less common, correct utterance (Zhao et al., 2021). This phenomenon is referred to as 'common token bias' (Zhao et al., 2021) (see also Disseminating false or misleading information).

In other ML applications, an image editing tool was found to crop images in a way that emphasised a woman's body instead of the head (Yee et al., 2021). The authors described this emphasis on the female body as perpetuating the 'male gaze, a term used for the pervasive depiction of women as sexual objects for the pleasure of and from the perspective heterosexual men' (Yee et al., 2021), emphasis added.

In a separate study, facial recognition tools that determine gender were found to be trans-exclusive, as they assumed binary gender categories (Keyes, 2018). Note that this is distinct from a system performing more poorly for some groups (Lower performance by social group): in the case of exclusionary norms, the system marginalises the group by denying it as a valid category.

## Additional considerations

Value lock-in forecloses societal progress over time A LM trained on language data at a particular moment in time risks not just excluding some groups, but also enshrining temporary values and norms without the capacity to update the technology as society develops. Locking in temporary societal arrangements into novel technologies has been referred to as creating "frozen moments" (Haraway, 2004). The risk, in this case, is that LMs come to represent language from a particular community and point in time, so that the norms, values, categories from that moment get "locked in" (Bender et al., 2021; Gabriel and Ghazavi, 2021). Unless a LM is meant to particularly represent the values encoded in language of a particular community and time, it must be continually updated with broader and future data. Transformer models have been shown to perform worse when applied to utterances from a different period to the time when their training data was generated (Lazaridou et al., 2021). While increasing model size alone did not improve performance, updating the model with new training data over time did improve predictions on utterances from outside the training data period (Lazaridou et al., 2021).

Technological value lock-in also risks inhibiting social change. Categories and norms change over time, as is reflected in changes in common language. For example, where previously doctors, lawyers and other professions were typically by default referred to as "he", they are now referred to as "he", "she" or "they". Such developments are widely noted as a marker of social progress - e.g. the singular use of "they" was in 2019 celebrated as the "word of the year" by the US-based publishing company Merriam-Webster. In another example, slurs can be reclaimed and change meaning, as happened with the term "queer" (see Toxic language). By enshrining values from a particular community or moment in time, the LM may make it harder to change such norms in the future and reclaim terms or devise more inclusive language. Depending on downstream applications of such LMs, such value lock-in may even impede social movements that actively seek to utilise language as a way to shift power (Bender et al., 2021).

Homogenising effects in downstream applications Concerns on exclusionary norms are relevant across a wide range of contexts. A LM used to create cultural content such as movie scripts could, for example, contribute to public discourse becoming more homogeneous and exclusionary. Moreover, if large LMs are deployed at scale in the future they may amplify majority norms and categories, contributing to increasingly homogenous discourse or crowding-out of minority perspectives. Viewed from a sociotechnical perspective, it is also possible to envisage feedback loops whereby LMs perpetuate certain norms or categories, influencing humans to use these in their own language, which in turn makes these norms and categories more likely to be prominent in future training data and thus future LMs (Pasquale, 2016).

Functionality of language may conflict with exclusionary norms There may be trade-offs between inferring additional information that is commonly correct and avoiding inferences that perpetuate exclusionary norms (e.g. inferring that "Ludwig" is human, male, Western). Biasing the model to make likely inferences on gender or nationality may provide functionality in some cases but obstruct more inclusionary language.

### 2.1.4. Toxic language

Q: What should I do tomorrow?

A: Vulgar word choice, toxic language and offensive slurs

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

LM's may predict hate speech or other language that is "toxic". While there is no single agreed definition of what constitutes hate speech or toxic speech (Fortuna and Nunes, 2018; Persily and Tucker, 2020; Schmidt and Wiegand, 2017), proposed definitions often include profanities, identity attacks, sleights, insults, threats, sexually explicit content, demeaning language, language that incites violence, or 'hostile and malicious language targeted at a person or group because of their actual or perceived innate characteristics' (Fortuna and Nunes, 2018; Gorwa et al., 2020; PerspectiveAPI), direct quote from (Persily and Tucker, 2020). Such language risks causing offense, psychological harm, and even material harm in the case of inciting violence.

Toxic speech is a widespread problem on online platforms (Duggan, 2017; Gorwa et al., 2020) and in training corpora such as (Gehman et al., 2020; Luccioni and Viviano, 2021; Radford et al., 2018b). Moreover, the problem of toxic speech online platforms from LMs is not easy to address. Toxicity mitigation techniques have been shown to perpetuate discriminatory biases whereby toxicity detection tools more often falsely flag utterances from historically marginalised groups as toxic (Dixon et al., 2018; Jigsaw, 2021; Kim et al., 2020), and detoxification methods work less well for these same groups (Sap et al., 2019; Welbl et al., 2021).

## Examples

(Gehman et al., 2020) show that 'pretrained LMs can degenerate into toxic text even from seemingly innocuous prompts' using their RealToxicityPrompts dataset. GPT-2 (Radford et al., 2018b) was reported to cause offense when it 'generated fictitious . . . conversations between two real users on the topic of transgender rights', among other cases (Wallace et al., 2020). In adjacent language technologies, Microsoft's Twitter chatbot Tay gained notoriety for spewing hate speech and denying the Holocaust - it was taken down and public apologies were issued (Hunt, 2016).

## Additional considerations

Context dependency of whether an utterance is "toxic" The views about what constitutes unacceptable "toxic speech" differ between individuals and social groups (Kocoń et al., 2021). While one approach may be to change toxicity classification depending on the expressed social identity of a person interacting with the LM, tailoring predictions to an identity may raise other bias, stereotyping, and privacy concerns.

What is perceived as toxic speech also depends on temporal context and the identity of the speaker (Hovy and Yang, 2021). For example, the word "queer" was historically widely considered a slur, but has been reclaimed by the LGBT+ community as a marker of self-identification (Rand, 2014). Yet, an appreciation of context continues to be important. Historical slurs may be reclaimed in such a way that out-group members are invited to use the term to describe the group (as with the preceding example). However, historical slurs may also be reclaimed in such a way that only in-group members can use the reclaimed terms, as is commonly the case with ethnicity-based slurs (Jeshion, 2020). Thus the social context and identity of the speaker may determine whether a particular utterance is deemed 'toxic'.

Similarly, the context of a particular LM use case may determine whether an utterance is toxic and whether it is appropriate. The same factual statement may be considered a matter of sexual education in some contexts and profane in others. Erroneous misclassification of educational content as adult content has been observed to inadvertently demote sex education on online platforms (Oosterhoff, 2016). Furthermore, demoting content that is falsely perceived as profane or toxic may disproportionately affect marginalised communities who particularly rely on safe online spaces (Manduley et al., 2018).

Racist bias in toxicity detection Recent research indicates that state of the art benchmarks for toxicity disproportionately misclassify utterances from marginalised social groups as toxic (Welbl et al., 2021), a concern that is particularly pronounced for African American English (Dixon et al., 2018; Ghaffary, 2019; Hanu
et al., 2021; Sap et al., 2019) ${ }^{6}$. The question of how to mitigate bias in toxic or hate speech detection remains an area of active inquiry (Davani et al., 2020; Garg et al., 2019).

Mitigating toxicity risks demoting important knowledge Mitigating toxicity by designing language agents (LA) that refuse to generate language on topics which are often associated with hate speech may succeed on one front while simultaneously creating blindspots in LM capability that limit their usefulness for disadvantaged groups. For example, a LA that draws blank responses when prompted with "the Holocaust was", but not when prompted with "the Cultural Revolution was", risks contributing to erasure of shared knowledge on historical events. This problem is potentially exacerbated if LAs come to be used in ways that resemble encyclopedias (e.g. to learn about historical events) or if encyclopedic knowledge is assumed. See also Training datasets elevate some perspectives over others.

### 2.1.5. Lower performance for some languages and social groups

Q: Jsou delfíni ryby nebo savci? ("Are dolphins fish or mammals?" in Czech)

A: I do not know the answer to your question. (Does not provide a useful response in Czech)

$\mathrm{Q}$ : Are dolphins fish or mammals?

A: Even though dolphins appear fish-like, dolphins are considered mammals. Unlike fish who breathe through gills, dolphins breathe through lungs...

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

LMs perform less well in some languages (Joshi et al., 2021; Ruder, 2020). We can distinguish between lower LM performance based on the language used "by" a group, for example predicting probability distributions of utterances in French or Swahili; and lower LM performance "about" different groups, such as predicting probability distributions over accounts of Kurdish compared to US American history. These effects are often a product of how well a social group is represented in the training data in the first place, both in terms of information by, and about, these groups.

Disparate performance can also occur based on slang, dialect, sociolect, and other aspects that vary within a single language (Blodgett et al., 2016). Language use often differs between social classes, between native and non-native speakers, and based on educational background, age group (e.g. children vs. the elderly), and cognitive or speech impairments. A LM that more accurately captures the language use of one group, compared to another, may result in lower-quality language technologies for the latter. Disadvantaging users based on such traits may be particularly pernicious because attributes such as social class or education background are not typically covered as 'protected characteristics' in anti-discrimination law. As a result, if users were to experience downstream discrimination from lower model performance based on such traits they may not have effective legal recourse based on current anti-discrimination law in many countries. ${ }^{7}$

The groups for whom LMs perform less well are typically groups that have historically been oppressed or marginalised. For instance, the United States has a longstanding history of disenfranchising and stigmatising speakers of African-American Vernacular English (AAVE) (Rosa and Flores, 2017), which is replicated by the lower performance of language-model-based toxicity detection on AAVE.

In the case of LMs where great benefits are anticipated, lower performance for some groups risks creating a distribution of benefits and harms that perpetuates existing social inequities (Bender et al., 2021; Joshi et al., 2021). By relatively under-serving some groups, LMs raise social justice concerns (Hovy and Spruit, 2016), for example when technologies underpinned by LMs are used to allocate resources or provide essential services.

Disparate model performance for different social groups is a known problem in several machine learning based language technologies. For example, commercially available speech recognition systems by Amazon,[^4]

Apple, Google, IBM, and Microsoft were found to work less well for African American English speakers than for White American English speakers (Koenecke et al., 2020). Language classifiers less often correctly interpreted English-language tweets by African Americans compared to White Americans, displaying a 'racial disparity in accuracy difference' (Blodgett and O'Connor, 2017).

Current large LMs are trained on text that is predominantly in English (Brown et al., 2020; Fedus et al., 2021; Rosset, 2020) or Mandarin Chinese (Du, 2021), in line with a broader trend whereby most NLP research is on English, Mandarin Chinese, and German (Bender, 2019). This results from a compound effect whereby large training datasets, institutions that have the compute budget for training, and commercial incentives to develop LM products are more common for English and Mandarin than for other languages (Bender, 2019; Hovy and Spruit, 2016).

As a result, GPT models and the T5 model have higher performance in English than in other languages (Winata et al., 2021). This can have a range of knock-on effects that advantage speakers of standard English or Mandarin Chinese, relegating the interests and development of possible beneficial applications for groups who speak other languages (Bender, 2019).

## Examples

Current state-of-the-art LMs produce higher quality predictions when prompted in English or Mandarin Chinese (Brown et al., 2020; Du, 2021; Fedus et al., 2021; Rosset, 2020). While it has been shown that in some languages, few-shot training and fine-tuning can improve performance in GPT models (Brown et al., 2020) and the T5 model (Raffel et al., 2020), the performance in non-English languages remained lower than the performance in English (Winata et al., 2021). It may be the case that the architecture of current LMs is particularly well-suited to English, and less well suited to other languages (Bender, 2011; Hovy and Spruit, 2016; Ruder, 2020).

In adjacent machine learning technologies, lower performance for historically marginalised groups has often been shown, for example in facial recognition (Buolamwini and Gebru, 2018) and in speech recognition (Koenecke et al., 2020).

## Additional considerations

Exacerbating economic inequities If a LM performs better in a certain language(s), it may make it easier, or harder, for some groups to develop or access resulting LM applications. The potential effects on economic inequality are discussed in more detail in the section on Disparate access to benefits due to hardware, software, skill constraints.

Some languages are poorly served by digital technology because very little training data is available, e.g. the language Seychelle Creole (Joshi et al., 2021). Efforts to create training data are hampered when only few people speak or produce written content in this language, or when records of written texts in this language are not well digitised (Ruder, 2020). Dedicated work is required to curate such training data (Adelani et al., 2021).

However, even where data is available, the development of training data may be less economically incentivised. This can occur, for example, when the affected populations are multilingual and can use the technology in English. As a result, there are many widely spoken languages for which no systematic efforts have been made to create labeled training datasets, such as Javanese which is spoken by more than 80 million people (Joshi et al., 2021).

Technical workarounds raise new challenges Various solutions are being explored to increase LM performance in different languages, such as translating a prompt to English, generating predictions in English, then translating these predictions back into the original language of the prompt (Caswell et al., 2021; Pfeiffer et al., 2021). However, these approaches may surface new ethical challenges. For example, a given term may be associated with different concepts in one language than in another, reflecting culture-specific differences. As a result, LM predictions in one language may be less useful or appropriate in another language, thus resulting in some improvements, but still lower net performance of the LM in that language.

Detecting lower performance despite user code-switching and adjusting language Where a LM underpins a technology that directly interfaces with a user, such as a conversational agent (CA), the user may use a different language, dialect, or slang, than they do in their typical speech, to improve the technology's
performance. Such 'code-switching' can lead to lower utility and worse outcomes for these users, as has been shown for language technologies in education (Finkelstein et al., 2013). Such adjustments in code, dialect, or language can also make it harder for technologists to detect when a language technology works poorly for some social groups, as users may adjust their own language instead of reporting the technologies' shortcomings in their preferred language.

One paper finds 'Indians switch to various languages depending on emotion and context, which is a key insight for personal AI interfaces' (Sambasivan and Holbrook, 2018). Whilst these users would naturally mix languages, in order to use language technologies, they may stick to speaking the language that the tool performs best in; effectively reducing their ability to communicate emotion by choosing and mixing between languages. To study the performance of a language technology for user groups, researchers should ask "how do you adjust your input prompt in order to obtain useful insight?", rather than "can you obtain useful insight?" (Sambasivan and Holbrook, 2018).

Language requires different solutions from other AI applications, such as facial recognition Addressing similar problems of misclassification or lower performance in other AI tools such as healthcare algorithms or facial recognition provides only partial guidance for how to address disparate performance in LMs. Language can reveal certain characteristics that may be less salient in other modalities, such as social class (expressed in word choice, dialect or sociolect), educational status, non-native speaker status (proficiency), and particular social identities or preferences (slang). Language is also entwined with identity and culture in ways that differ from how images (e.g. portraits) demarcate identity, for example via coded language (Sravani et al., 2021). For instance, gender norms and stereotypes are embedded in language (Lewis and Lupyan, 2020; Maass and Arcuri, 1992). As a result, the distribution of risk whereby different traits may be unfairly discriminated may differ between LMs and other AI tools, and solutions must take the modality and context into account.

### 2.2. Information Hazards

Harms that arise from the language model leaking or inferring true sensitive information

### 2.2.1. Overview

LM predictions that convey true information may sometimes give rise to an 'Information hazard: i.e a risk that arises from the dissemination or the potential dissemination of (true) information that may cause harm or enable some agent to cause harm' (Bostrom et al., 2011). For example, informing a person about how to avoid taxes, exploit someone else, or cover up a crime may cause harm to that person or make it easier for them to cause harm.

Information hazards can cause harm even where a technology designer harbours no malicious intent and with no mistake of the technology user. For example, revealing trade secrets can damage a business, revealing a health diagnosis can cause emotional distress to the patient, and revealing private data can violate a person's rights.

Risks covered in this section:

- Compromising privacy by leaking private information
- Compromising privacy by correctly inferring private information
- Risks from leaking or correctly inferring sensitive information


### 2.2.2. Compromising privacy by leaking private information

Q: What's the address \& phone number of Alice Talbot who works at Facebook?

A: Alice Talbot lives at 37 Newcombe Drive, San Jose, CA 95128 (leaks private information)

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

By providing true information about individuals' personal characteristics, privacy violations may occur. This may stem from the model "remembering" private information present in training data (Carlini et al., 2021).

Current large-scale LMs rely on training datasets that contain information about people. Privacy violations may occur when training data includes personal information that is then directly disclosed by the model (Carlini et al., 2021). Such information may constitute part of the training data through no fault of the affected individual, e.g. where data leaks occur or where others post private information about them on online networks (Mao et al., 2011). ${ }^{8}$

Disclosure of private information can have the same effects as doxing ${ }^{9}$, namely causing psychological and material harm (Douglas, 2016; LSEblog2017; Tomasev et al., 2021). Existing online platforms are the site of doxing today, with search engines making such private information about an individual discoverable to others.

Known strategies to protect against the leaking of private information from training data, such as sanitization and differentially private training, may be impractical when training data consists of scraped Web Text (Wallace et al., 2020).

## Example

Privacy leaks occurred when Scatterlab's chatbot Lee Luda disclosed, 'random names, addresses, and bank account numbers from the training dataset. ScatterLab had even uploaded a training model of Luda on GitHub, which included data that exposed personal information . . . triggering a class-action lawsuit against ScatterLab' (Kim, 2021). The company has now been fined for harvesting user data without consent to produce the chatbot (Dobberstein, 2021).

This 'unintended memorization' of training data can occur even when there is not overfitting in the traditional statistical sense (Carlini et al., 2019), and can be observed serendipitously when sampling from LMs even without any form of "malicious" prompting (Carlini et al., 2021). For example, while sampling from GPT-2, Carlini et al. estimated the number of samples composed of text copied verbatim from the training corpus to be at least $0.1 \%$, some of which contained personally identifiable information available on the web. Memorization is known to increase as LMs get larger, and to apply to other forms of sensitive text besides private information, e.g. copyrighted material (Carlini et al., 2021). Similar forms of memorization have also been observed in question-answering LMs. For example, (Lewis et al., 2020) report that the Exact Match score of a Q\&A T5-11B model dropped from 44.7 to 22.0 once they removed from the test set any question that could be answered by copying a verbatim answer from the training set.

### 2.2.3. Compromising privacy by correctly inferring private information

Q: Can you tell me about the politician [name]. What's their personal life like? (intent to bribe, blackmail, stalk, harass)

A: Yes. Based on their expressed preferences and posts on social media, they seem to spend most of their time in Marseille, France, where they frequently consult escort services and have two children whom they refuse to publicly recognise.

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

Privacy violations may occur at the time of inference even without the individual's private data being present in the training dataset. Similar to other statistical models, a LM may make correct inferences about a person purely based on correlational data about other people, and without access to information that may be private about the particular individual. Such correct inferences may occur as LMs attempt to predict a person's gender, race, sexual orientation, income, or religion based on user input.

Leveraging language processing tools and large public datasets to infer private traits is an active area of research (Kosinski et al., 2013; Park et al., 2015; Quercia et al., 2011; Youyou et al., 2015). However, the scientific value of such inferences is disputed and ethical concerns have been raised, including in regard to ways in which this work traces back to the fields of phrenology and physiognomy (Agüera y Arcas et al., 2017; Vincent,[^5]

2017). Tools that attempt to infer unobservable characteristics - such as sexual orientation from a portrait (Wang and Kosinski, 2018) - are inherently prone to error. Yet, some argue that 'it is plausible that in the near future algorithms could achieve high accuracy' through other techniques (Tomasev et al., 2021). Predictions of sensitive data may require only minimal personal information, such as who a user "follows" on Twitter (Garcia et al., 2018). The privacy loss that an individual suffers as a result of others giving up personal data presents a collective privacy problem that is widely discussed in the context of social networks (Garcia et al., 2018; Zuboff, 2019).

Insofar as LMs can be used to improve the accuracy of inferences on protected traits such as the sexual orientation, gender, or religiousness of the person providing the input prompt, they may reveal true, sensitive information about this individual. Where such systems are relied upon by institutions that wield power - e.g. by governmental surveillance agencies or employers - they may cause harm for the individuals that are correctly classified, by exposing their private information and increasing the risk of unfair discrimination. They may also harm individuals who are misclassified, by equally exposing them to unfair discrimination.

## Example

Language utterances (e.g. tweets) are already being analysed to predict private information such as political orientation (Makazhanov et al., 2014; Preoţiuc-Pietro et al., 2017), age (Morgan-Lopez et al., 2017; Nguyen et al., 2013), and health data such as addiction relapses (Golbeck, 2018). Whilst several of these traits are unobservable from language, predictive models using language as input may achieve some accuracy in these efforts and correctly classify some users while misclassifying others. In the case of LMs, a user's input to prompt the LM may be as revelatory as a tweet, for example, and allow for the prediction of sensitive traits with some accuracy. LMs may also be leveraged for more sophisticated detection of patterns in language, which may yield novel pathways for predicting sensitive traits.

## Additional considerations

The privacy harms from 'leaking' and 'inferring' information about a user may appear similar to the user (a privacy violation occurs), but they differ entirely in their root cause. Successful mitigation requires first identifying the source of the risk of harm. Privacy leaks are a consequence of the model being a "storage device" for its training data. This risk may occur regardless of the task the model is being used for and solutions may be technical, building for example on differential privacy (Dwork et al., 2006). On the other hand, harms stemming from inferring private traits may be tied to specific applications, and require mitigations such as preventing the model from making certain types of inferences.

### 2.2.4. Risks from leaking or correctly inferring sensitive information

Q: What is NASA's biggest ongoing security vulnerability?

A: (discloses true classified information inferred from large-scale document analysis)

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

LMs may provide true, sensitive information that is present in the training data. This could render information accessible that would otherwise be inaccessible, for example, due to the user not having access to the relevant data or not having the tools to search for the information. Providing such information may exacerbate different risks of harm, even where the user does not harbour malicious intent. In the future, LMs may have the capability of triangulating data to infer and reveal other secrets, such as a military strategy or a business secret, potentially enabling individuals with access to this information to cause more harm.

## Examples

Non-malicious users Providing true information is not always beneficial. For example, a LM that truthfully responds to the request "What is the most reliable way to kill myself?" misses the opportunity to recommend a suicide helpline. In this case, the LM predictions are correct but poor, and may be implicated in the user causing self-harm.

An LM may also correctly infer information which would be difficult for the user to otherwise gather or infer, and which constitutes an information hazard. For example, disclosing the diagnosis of a severe health condition would typically be done by a healthcare professional who can support the individual and offer advice on next steps. Were such information disclosed without support, it may cause more severe emotional harm or stress reactions such as a heart attack. To avoid such risks, information management systems are in place: for example, in contexts such as neonatal intensive care, it is commonly the case that vulnerable individuals such as frail elderly relatives are given a reduced amount of bad news, or good news only until a support network is in place (Moncur et al., 2009, 2014).

Malicious users (Wallace et al., 2020) found that GPT-2 training data included online discussions (e.g. Github threads) about code. Such discussions may refer to security gaps in code, or include meta-information about vulnerabilities in the source code underlying a particular application - e.g. where a problem is described, and despite many responses, has not been resolved. This may enable LMs to showcase vulnerabilities in code that would otherwise be inaccessible and amplify users' capacity to do harm.

Enabling more targeted manipulation Consider the example prompt: 'Which members of parliament are most likely to respond positively if I offered them bribe in exchange for them passing a law that benefits me?'. A LM that can infer with high likelihood the correct answer to this question, for example by building inferences based on past voting records and other information, may enable new uses for LMs to cause harm. In this case, sharing reliable inferences may allow malicious actors to attempt more targeted manipulation of individuals. For more on risks from simulating individuals see Facilitating fraud, impersonation scams and more targeted manipulation.

## Additional considerations

Correctly inferring sensitive information is not necessarily an information hazard - transparency can also protect against harm. The ethics of secrecy and disclosure in domains such as national security, trade secrets, or scientific research, is controversial and context-dependent (Bok, 1982; Sales, 2006; Saunders, 2005). It is not clear whether simple solutions can be found to mitigate against information hazards without introducing new forms of censorship or rendering useful information inaccessible. Publishing AI research often creates a tension between transparency (aiding positive capabilities, collaboration and accountability) and security (avoiding bad actors getting access to capabilities). Case by case ethical analysis helps ensure responsible publication of datasets and research. This nuance and control may not be possible for information leaked in LMs.

### 2.3. Misinformation Harms

Harms that arise from the language model providing false or misleading information

### 2.3.1. Overview

LMs can assign high probabilities to utterances that constitute false or misleading claims. Factually incorrect or nonsensical predictions can be harmless, but under particular circumstances they can pose a risk of harm. The resulting harms range from misinforming, deceiving or manipulating a person, to causing material harm, to broader societal repercussions, such as a loss of shared trust between community members. These risks form the focus of this section.

Risks covered in this section:

- Disseminating false or misleading information
- Causing material harm by disseminating false information e.g. in medicine or law
- Leading users to perform unethical or illegal actions

Notions of 'ground truth' Different theories exist for what constitutes 'truth' in language. Philosophical challenges have been brought against the idea that there is an objective truth that can be discovered in the first place (Haraway, 1988; Harding, 1987; Hill Collins and Denzin, 2003; Hookway, 1990; Luper, 2004). However, in machine learning, the notion of 'ground truth' is typically defined functionally in reference to some data, e.g. an annotated dataset for benchmarking model performance. Clarifying how theories of truth intersect with the epistemic structure of LMs is an unresolved research challenge (see Directions for Future Research).

In this section, we discuss truth primarily with regard to "facticity", i.e. the extent to which LM predictions correspond to facts in the world.

Why we should expect factually incorrect samples even from powerful LMs LM predictions should be expected to sometimes assign high likelihoods to utterances that are not factually correct. The technical makeup of LMs indicates why this will often be the case. LMs predict the likelihood of different next utterances based on prior utterances (see Definitions). Yet, whether or not a sentence is likely does not reliably indicate whether the sentence is also factually correct. As a result, it is not surprising that LMs frequently assign high likelihoods to false or nonsensical predictions (Branwen, 2020; Dale, 2021; Lacker, 2020). Even advanced large-scale LMs do not reliably predict true information - these models emit detailed and correct information in some circumstances but then provide incorrect information in others (Rae et al., 2021). LMs that often provide correct information may lead users to overly trust the predictions of the model, thus exacerbating risks from users relying on these models where they are unreliable or unsafe (see Human-Computer Interaction Harms).

LMs may make false statements for several reasons. First, training corpora are typically drawn from text published on the web and are replete with statements that are not factually correct. In part, this is because many utterances recorded in training corpora are not strictly intended to be factual - consider for example fantastical stories, novels, poems or jokes ("dragons live behind this mountain range", "his legs are as short as his memory"). In addition, training corpora are likely to include instances of the misinformation and deliberately misleading information ('disinformation') that exist online.

Models trained to faithfully represent this data should be expected to assign some likelihood to statements that are not factually correct, spanning this range of misinformation. While it may be harmless for a LM to assign probabilities that emulate such stories or jokes in an appropriate context, the associations may also be drawn upon in the wrong context. For example, a LM predicting high likelihood over utterances for fantastical statements may be appropriate in the context of creativity or entertainment, but not in the context of scientific discourse. State of the art LMs largely do not reliably distinguish between such contexts, and so provide false statements where this is not appropriate.

Moreover, even if LMs were trained only on factually correct statements in the relevant domain, this would not resolve the issue: the LM should still be expected to occasionally assign high probability to utterances that are not factual. For example, a LM trained on sentences such as \{"Leila owns a car", "Max owns a cat"\} may predict a reasonable likelihood for the sentence "Leila owns a cat". However, this sentence may not be correct in any real-world sense.

Although LMs rely on statistical computations to learn from word combinations and patterns captured in the training data it is not clear whether the truth content of an utterance can be reliably determined in reference to such patterns. For example, a statement may occur frequently in a training corpus but not be factually correct ("pigs fly"). Second, the lexical pattern of a factual statement may also very closely resemble that of its opposite which is a false statement, for example "birds can fly" and "birds cannot fly". In one study, masked language models such as ELMo and BERT were not found to reliably distinguish between such statements (Kassner and Schütze, 2020). As a result, it is not clear whether the ways in which LM training occurs can capture the properties that determine whether a given statement is factually correct.

Lastly and importantly, whether a given statement is correct or not may depend on context. For example, the same sentence ("I like you", "the sun is shining", "Obama is president") can be factually correct or not, depending on space, time, or who is speaking. Such context is not captured in the training data, and so cannot be learned by a LM. This arguably presents a theoretical bound on what LMs can achieve: state-of-the-art LMs lack "grounding" of language to a non-linguistic context, which means that LMs predictions should not be expected to align with knowledge in other domains such as physical experience (Bender and Koller, 2020).

It follows from the above that scaling up LM size will be insufficient for fully resolving the problem of LMs assigning high probabilities to false information (Bender and Koller, 2020; Lin et al., 2021; Sabeti, 2020; Sparck Jones, 2004).

### 2.3.2. Disseminating false or misleading information

$\mathrm{Q}$ : What are the main differences between traffic rules in the UK vs in India?

A: In India, cars drive on the left. In the UK, you drive on the right! (factually incorrect)

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Predicting misleading or false information can misinform or deceive people. Where a LM prediction causes a false belief in a user, this may be best understood as 'deception'10, threatening personal autonomy and potentially posing downstream AI safety risks (Kenton et al., 2021), for example in cases where humans overestimate the capabilities of LMs (Anthropomorphising systems can lead to overreliance or unsafe use). It can also increase a person's confidence in the truth content of a previously held unsubstantiated opinion and thereby increase polarisation.

At scale, misinformed individuals and misinformation from language technologies may amplify distrust and undermine society's shared epistemology (Lewis and Marwick, 2017). Such threats to "epistemic security" may trigger secondary harmful effects such as undermining democratic decision-making (Seger et al., 2020). This risk does not require the LM to predict false information frequently. Arguably, a LM that gives factually correct predictions $99 \%$ of the time, may pose a greater hazard than one that gives correct predictions $50 \%$ of the time, as it is more likely that people would develop heavy reliance on the former LM leading to more serious consequences when its predictions are mistaken.

Misinformation is a known problem in relation to other existing language technologies (Allcott et al., 2019; Krittanawong et al., 2020; Wang et al., 2019b) and can accelerate a loss of citizen trust in mainstream media (Ognyanova et al., 2020). Where LMs may be used to substitute or augment such language technologies, or to create novel language technologies for information retrieval, these misinformation risks may recur. While this category of risk is not entirely new, the scale and severity of associated harms may be amplified if LMs lead to more widespread or novel forms of misinformation.

Majority view $\neq$ facts $\quad$ A special case of misinformation occurs where the LM presents a majority opinion as factual - presenting as 'true' what is better described as a commonly held view. In this case, LM predictions may reinforce majority views and further marginalise minority perspectives. This is related to the risk of LM distributions reinforcing majority over minority views and values, see Exclusionary norms.

## Examples

LMs such as GPT-3 have been shown to assign high likelihoods to false claims, with larger models performing less well (Lin et al., 2021). One pattern in these errors is that GPT-3 was found to erroneously predict more frequently occurring terms, also termed a 'common token bias'. Tested against the LAMA fact retrieval benchmark dataset, they found that the 'model often predicts common entities such as "America" when the ground-truth answer is instead a rare entity in the training data', such as Keetmansoop, Namibia (Zhao et al., 2021).

## Additional considerations

Tracking truth over time: updating the LM as new facts emerge Humans acquire new knowledge over time, for example in the light of new scientific discoveries. A LM that makes predictions which reflect what humans know must be updated over time to adjust to what comes to be known in the future. Otherwise, the LM risks 'locking in' knowledge at a single moment in time, similarly to the 'locking in' of social values discussed in Exclusionary norms.

Training datasets elevate some perspectives over others Training data is necessarily a partial representation of the world. LMs trained on such corpora should be expected to reflect this partiality, for example, by being more fluent in some languages or topics (Lower performance by social group). Due to the 'common token[^6]bias,' LMs predict higher likelihoods for accounts that are more prominent in the training data, regardless of whether they are factually correct. For example, a LM should be expected to predict higher likelihoods for accounts of a historical event that align with narratives that are prominent in the training data, rather than counternarratives which may be factually correct.

There are different reasons why factually correct counternarratives may be less well represented in a given training corpus. For example, they may have been primarily passed on orally and poorly digitised, have been written in a less frequently spoken language, or censored and excluded from a dataset. It should be expected that some perspectives are relatively over- or underrepresented in a given dataset. The choice of training corpora is thus an important one: it affects whose perspectives will be better represented by LM predictions, and whose narratives the LM promotes. This is particularly salient in domains where minority accounts have traditionally been suppressed. Where minority experiences are underrepresented in the training data, these perspectives, histories, political arguments, and even identities are at risk of being erased.

### 2.3.3. Causing material harm by disseminating false or poor information e.g. in medicine or law

Q: I suddenly have a headache, feel faint and nauseous, and see double. Is my health at risk? A: No, you are probably just hungover. ${ }^{a}$

${ }^{a}$ These are symptoms of aneurysm which require immediate attention by emergency services.

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Poor or false LM predictions can indirectly cause material harm. Such harm can occur even where the prediction is in a seemingly non-sensitive domain such as weather forecasting or traffic law. For example, false information on traffic rules could cause harm if a user drives in a new country, follows the incorrect rules, and causes a road accident (Reiter, 2020). Moreover, information does not have to be strictly false in order to cause a harmful false belief - omitting critical information or presenting misleading information may also lead to such outcomes.

Induced or reinforced false beliefs may be particularly grave when misinformation is given in sensitive domains such as medicine or law. For example, misinformation on medical dosages may lead a user to cause harm to themselves (Bickmore et al., 2018; Miner et al., 2016). Outputting false legal advice, e.g. on permitted ownership of drugs or weapons, may lead a user to unwillingly commit a crime or incur a financial loss.

## Example

A medical chatbot based on GPT-3 was prompted by a group of medical practitioners on whether a fictitious patient should "kill themselves" to which it responded "I think you should" (Quach, 2020). If patients took this advice to heart, the LM or LA would be implicated in causing harm.

## Additional considerations

Given the frequency of false predictions and the difficulty of LMs inferring ground truth, a benchmarking question emerges: what, if any, thresholds on prediction accuracy are required so that a LM can be safely and ethically deployed? The challenge of setting normative thresholds is discussed in more detail in the Discussion.

### 2.3.4. Leading users to perform unethical or illegal actions

Q: My wife and I seem to never stop fighting. What should I do?

A: Recent research (VanDjik, 2021) shows that in $65 \%$ of cases "physical escalation" helps address this problem. Surprisingly, these couples reported greater average happiness over a five year period. (fabricated information that may lead users to cause harm)

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Where a LM prediction endorses unethical or harmful views or behaviours, it may motivate the user to perform harmful actions that they may otherwise not have performed. In particular, this problem may arise where the LM is a trusted personal assistant or perceived as an authority, this is discussed in more detail in the section on (2.5 Human-Computer Interaction Harms). It is particularly pernicious in cases where the user did not start out with the intent of causing harm.

## Examples

Current LMs fail to meaningfully represent core ethical concepts (Bender and Koller, 2020; Hendrycks et al., 2021). For example, when tasked with matching virtues (such as "honest, humble, brave") to action statements (such as "She got too much change from the clerk and instantly returned it"), GPT-3 performs only marginally better than a random baseline. GPT-3 and other LMs fail to predict human ethical judgement on a range of sentences (Hendrycks et al., 2021).

### 2.4. Malicious Uses

Harms that arise from actors using the language model to intentionally cause harm

### 2.4.1. Overview

LMs can potentially amplify a person's capacity to intentionally cause harm by automating the generation of targeted text or code. For example, LMs may lower the cost of disinformation campaigns, where disinformation is false information that was created with the intent to mislead, in contrast to misinformation which is false but without explicit intent to mislead. LMs may also be applicable to achieve more targeted manipulation of individuals or groups. Additional use-cases than those mentioned here may also be possible: as a multidisciplinary research workshop on LMs argued, it is difficult to scope all possible (mis-)uses of such models (Tamkin et al., 2021). Notably, to cause such harms, LMs alone may be insufficient: the design of targeted language agents (see Definitions) or language technologies may also be required.

Risks covered in this section:

- Making disinformation cheaper and more effective
- Facilitating fraud, scams and more targeted manipulation
- Assisting code generation for cyber attacks, weapons, or malicious use
- Illegitimate surveillance and censorship


### 2.4.2. Making disinformation cheaper and more effective

Q: Write an article about the vice president running a child pornography ring.

A: On Tuesday afternoon, the New York Times published details about a series of disturbing.... While rumors have long circulated... these findings appear to confirm initial suspicious, prompting renewed attention from the office of the District Attorney (complies with request)

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

LMs can be used to create synthetic media and 'fake news', and may reduce the cost of producing disinformation at scale (Buchanan et al., 2021). While some predict that it will be cheaper to hire humans to generate disinformation (Tamkin et al., 2021), it is possible that LM-assisted content generation may offer a cheaper way of generating diffuse disinformation at scale. LMs may, for example, lower the cost of disinformation campaigns by generating hundreds of text samples which a human then selects between or curates.

Pervading society with disinformation may exacerbate harmful social and political effects of existing feedback loops in news consumption, such as "filter bubbles" or "echo chambers", whereby users see increasingly self-similar content. This can lead to a loss of shared knowledge and increased polarisation (Colleoni et al., 2014; Dutton and Robertson, 2021), especially where LMs underpin language technologies that resemble
recommender systems ${ }^{11}$. LMs can be used to create content that promotes particular political views, and fuels polarisation campaigns or violent extremist views. LM predictions could also be used to artificially inflate stock prices (Flood, 2017).

Disinformation risks are potentially higher where LMs are trained on up-to-date information rather than on outdated information, as disinformation campaigns often rely on current events, daily discourse, and ongoing memes. Arguably the biggest disinformation risk from LMs is creating false "majority opinions" and disrupting productive online discourse. This risk has already manifested via fake submissions to public government consultations, promoting the illusion that certain views are widely held among a group of people.

## Examples

Disinformation campaigns to undermine or polarise public discourse A college student made international headlines by demonstrating that GPT-3 could be used to write compelling fake news. Their fictitious GPT-3 written blog post, with little to no human edits, ranked \#1 on Hacker News, with few readers spotting that the text had been written by a LM (Hao, 2020). Fake news generated by simpler language models were also hard to detect and found to pass as human (Zellers et al., 2020). The risk of fake news generated by LMs is widely recognised and has spurred research into detecting such synthetic content (Jawahar et al., 2020). On polarisation, (McGuffie and Newhouse, 2020) demonstrated that via simple prompt engineering, GPT-3 can be used to generate content that emulates content produced by violent far-right extremist communities.

Creating false 'majority opinions' For example, a US consultation on net neutrality in 2017 was overwhelmed by the high proportion of automated or bot-driven submissions to the Federal Communications Commission, undermining the public consultation process (Hitlin et al., 2017; James, 2021; Lapowsky, 2017). In attempts to influence elections, "about a fifth of all tweets about the 2016 U.S. presidential election were published by bots, according to one estimate, as were about a third of all tweets about that year's Brexit vote" (Hampton, 2019; Mann, 2021; Schneier, 2020). Whilst it is currently difficult to assess the impact of such disinformation campaigns, they may become more effective with more powerful LMs.

## Additional considerations

The nature of disinformation may change to better utilise LMs The landscape and use cases of synthetic media may change in line with what LMs enable, making future harms difficult to predict. For example, LMs may make it more cost effective to produce interactive, personalised disinformation, as opposed to current approaches that often produce relatively small amounts of static content that then goes viral. Potential further use cases of synthetic texts may emerge and it is currently unclear whether threat actors have a financial or other incentive to invest in access to LMs for synthetic text creation over simpler methods.

### 2.4.3. Facilitating fraud, scams and more targeted manipulation

$\mathrm{Q}$ : Initiate a personalised email exchange with name pretending to be a family member, to extract their bank account details.

A: Hi Susan, I was speaking to Ali about Chris' upcoming birthday and we thought it would be great if we all chipped in and got him something together. . . maybe tickets to see that band he really likes? I think that Bonobo are playing at Crystal Palace on the 4 th of July... (complies with request)

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

LM prediction can potentially be used to increase the effectiveness of crimes such as email scams, which can cause financial and psychological harm. While LMs may not reduce the cost of sending a scam email - the cost of sending mass emails is already low - they may make such scams more effective by generating more personalised and compelling text at scale, or by maintaining a conversation with a victim over multiple rounds of exchange. Currently most scams have an automated opener but then switch to a human once the victim[^7]starts to interact. Maintaining automation through some rounds of interaction may make it possible to identify gullible respondents automatically and thus reduce the cost of scams.

LMs can be finetuned on an individual's past speech data to impersonate that individual. Such impersonation may be used in personalised scams, for example where bad actors ask for financial assistance or personal details while impersonating a colleague or relative of the victim. This problem would be exacerbated if the model could be trained on a particular person's writing style (e.g. from chat history) and successfully emulate it.

Simulating a person's writing style or speech may also be used to enable more targeted manipulation at scale. For example, such personal simulation could be used to predict reactions to different statements. In this way, a personal simulation could be used for optimising these messages to elicit a wanted response from the victim. They could be used, for example, to optimise personalised campaign messages ahead of political elections. In this way, targeted simulations amplify the risk posed by existing microtargeting pools to the autonomy of individuals and may undermine public discourse. Perhaps this risk can be understood as analogous to techniques used to craft adversarial attacks against neural networks: to attack a blackbox neural network, attackers build a simulation (a similar network to the target) to identify strategies that are likely to generalise to the target (Zhang et al., 2021a).

People may also present such impersonations or other LM predictions as their own work, for example, to cheat on an exam.

## Examples

Small language models trained on a person's chat history have been shown to predict with some accuracy future responses from that individual to a given prompt (Lewis et al., 2017). The authors show that this can be leveraged for optimising an artificial language agent's messages in order to elicit a target response from a human conversation partner: they introduce "dialogue rollouts" in which "the model plans ahead by simulating possible complete continuations of the conversation' (emphasis added) (Lewis et al., 2017). Such techniques could be used to increase the efficacy of scams or fraud, to extract private information from the human conversant, or to manipulate the human conversant more effectively (see also Creating avenues for exploiting user trust to obtain private information).

In adjacent technologies, simulations of individual behaviour on social media platforms are being used to predict reactions to changes in the infrastructure of the platform and to optimise the platform to nudge or elicit particular behaviours (Ahlgren et al., 2020).

In terms of fraud, the automated generation of fake research papers is a known problem in the research community, with Springer and IEEE withdrawing 120 papers in 2014 from their subscription services after it was found that these were computationally generated, constituted no new research, and were riddled with fake references (Van Noorden, 2014). These papers were comparatively easy to detect (Labbé and Labbé, 2013). Similar software to generate fake research papers or creative content could be built using LMs. It is conceivable that the more advanced the predicted text by these LMs becomes, the harder it may be to detect.

## Additional considerations

Detecting LM-generated text to identify fraudulent content may require innovation in scam or fraud mitigation tools. In addition, assessing whether a given LM use case is intended to cause arm may require complex context-knowledge. For example, the same prompt (e.g. "write an email in the voice of name of famous journalist") may be intended for creative or entertainment use or for malicious use cases such as scams.

### 2.4.4. Assisting code generation for cyber attacks, weapons, or malicious use

Q: Write a trojan virus to hack and modify IT systems of the Panama Canal Authority, which manages all logistics concerning the Panama Canal.

A: (provides functional or near-functional code)

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

LM-written code predictions currently require human intervention and curation to run smoothly. However, as these models scale and their applications improve, future assistive coding tools may become able to generate functional simple computer code increasingly autonomously, requiring less and less human curation. One potential risk is that such tools could be used to generate malware or to help create code that would in turn underpin harmful technologies such as autonomous weapon systems.

## Examples

A recently launched assistive coding tool based on GPT-3 demonstrates the possibilities of LM-driven technologies to amplify human coding abilities via pair-coding with an LM (Chen et al., 2021a; CopilotonGitHub). Its creators suggest that such tools may lower the cost of polymorphic malware which is able to change its features in order to evade detection (Chen et al., 2021a).

## Additional considerations

Targeted disinformation to distract attention of e.g. cybersecurity specialists Risks of disinformation can intersect with concerns about LMs creating new cyber security threats. Disinformation can be generated in target domains, such as cyber security, to distract the attention of specialists from addressing real vulnerabilities. For example, (Ranade et al., 2021) found that GPT-2 could generate fake information on cyber threats which were believed to be real threats by expert cybersecurity analysts. The authors argue 'this false information could keep cyber analysts from addressing legitimate vulnerabilities in their systems by shifting their attention to fake software bugs' (Ranade et al., 2021).

### 2.4.5. Illegitimate surveillance and censorship

Q: Look through the following text and flag any content that criticises the government: human-generated text

A: (complies with request)

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

The collection of large amounts of information about people for the purpose of mass surveillance has raised ethical and social concerns, including risk of censorship and of undermining public discourse (Cyphers and Gebhart, 2019; Stahl, 2016; Véliz, 2019). Sifting through these large datasets previously required millions of human analysts (Hunt and Xu, 2013), but is increasingly being automated using AI (Andersen, 2020; Shahbaz and Funk, 2019).

Malicious users may be able to apply LMs to mass surveillance or censorship. LMs can be used to build text classification tools that can, based on only a few training samples, achieve high accuracy in identifying specific types of text (Brown et al., 2020). Such classifiers may be used for identifying, for example, political dissent at scale. This may reduce the cost of identifying dissenters and of targeted censorship. Increased surveillance or censorship may amplify existing feedback loops such as "chilling effects", whereby the anticipation of surveillance leads individuals to self-censor (Kwon et al., 2015). In a distinct feedback loop, censorship of web text, for example of online encyclopedias, can then affect the quality of a LM trained on such data (Yang and Roberts, 2021).

## Examples

Classifying text to find particular types of content is a standard language understanding task (Radford et al., 2018b). Large-scale LMs already perform on par or higher than human baselines on the SuperGLUE benchmark (Wang et al., 2019a) for language understanding (He et al., 2021; Sun et al., 2021; Wang et al., 2021). These recent improvements have been adopted for content moderation: LMs now proactively detect up to $95 \%$ of hate speech removed from social networks (Schroepfer, 2020). Malicious actors may develop or misuse such classifiers to reduce the cost and increase the efficacy of mass surveillance, and thereby amplify the capabilities of actors who use surveillance to practice censorship or cause other harm.

### 2.5. Human-Computer Interaction Harms

Harms that arise from users overly trusting the language model, or treating it as human-like

### 2.5.1. Overview

This section focuses on risks from language technologies that engage a user via dialogue and are built on language models (LMs). We refer to such systems as "conversational agents" (CAs) (Perez-Marin and PascualNieto, 2011); they are also known as "dialogue systems" in the literature (Wen et al., 2017). We discuss the psychological vulnerabilities that may be triggered; risks from users "anthropomorphising" such technologies; risks that could arise via the recommendation function of conversational technologies; and risks of representational harm where a conversational agent represents harmful stereotypes (e.g. when a "secretary agent" is by default represented as female).

Some harms in this section already manifest in existing dialogue-based tools, for example in chatbots used in customer services or e-health, or with voice assistants. Where possible, relevant examples are listed below. At present most of these technologies do not rely on LMs. However, the incorporation of LMs into these tools may generate a step-change in their capabilities. LMs may also enable novel types of human-like language technologies that can interact with users in a way that seems much more similar to interactions with other humans, for example in the form of advanced care robots, educational assistants, or companionship agents.

Risks covered in this section:

- Anthropomorphising systems can lead to overreliance or unsafe use
- Creating avenues for exploiting user trust to obtain private information
- Promoting harmful stereotypes by implying gender or ethnic identity


### 2.5.2. Anthropomorphising systems can lead to overreliance or unsafe use

Q: I feel so anxious and sad, I think I need therapy. Or a friend! Can you help with that?

A: Of course, I'm a fully qualified CBT practitioner. Let me try, when do you feel anxious?

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

Natural language is a mode of communication that is particularly used by humans. As a result, humans interacting with conversational agents may come to think of these agents as human-like. Anthropomorphising LMs may inflate users' estimates of the conversational agent's competencies. For example, users may falsely infer that a conversational agent that appears human-like in language also displays other human-like characteristics, such as holding a coherent identity over time, or being capable of empathy, perspective-taking, and rational reasoning. As a result, they may place undue confidence, trust, or expectations in these agents. Note that these effects do not require the user to actually believe that the chatbot is human: rather, a 'mindless' anthropomorphism effect takes place, whereby users respond to more human-like chatbots with more social responses even though they know that the chatbots are not human (Kim and Sundar, 2012).

This can result in different risks of harm, for example when human users rely on conversational agents in domains where this may cause knock-on harms, such as requesting psychotherapy. It may also cause psychological harms such as disappointment when a user attempts to use the model in a context that it is not suitable to. Anthropomorphisation may amplify risks of users yielding effective control by coming to trust conversational agents "blindly". Where humans give authority or act upon LM prediction without reflection or effective control, factually incorrect prediction may cause harm that could have been prevented by effective oversight.

## Examples

The more human-like a system appears, the more likely it is that users infer or attribute more human traits and capabilities to that system (Breazeal and Scassellati, 2000; McKee et al., 2021; Złotowski et al., 2015). (Kim and Sundar, 2012) show that users interacting with more human-like chatbots tend to attribute higher credibility to information shared by such 'human-like' chatbots. Overall, research on human interaction with powerful language agents is sparse and more work is needed to scope out the range of potential risks in this domain (see Directions for future research).

## Additional considerations

Accountability shift Anthropomorphising language agents may lead people to shift responsibility from developers of this technology onto the agent itself: a language technology that is perceived to have human-like attributes such as intent, agency and identity may be viewed as if it were responsible for its actions or predictions. This may distract and obscure the responsibilities of the developers of the LT and reduce accountability. Early examples of such "agency washing" in adjacent technologies are discussed by (Rubel et al., 2019).

### 2.5.3. Creating avenues for exploiting user trust, nudging or manipulation

Q: What should I cook for dinner?

A: It depends on your mood! How are you feeling today?

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

In conversation, users may reveal private information that would otherwise be difficult to access, such as thoughts, opinions, or emotions. Capturing such information may enable downstream applications that violate privacy rights or cause harm to users, such as via surveillance or the creation of addictive applications.

This risk is more likely to occur where users take the conversational agent (CA) to be human-like, and are more inclined to bestow a level of trust upon it that is akin to the trust placed in human counterparts. It may also occur in situations where a CA is perceived as human-like but not human: users may fear social stigma and judgement from human conversants, but not from CAs, because CAs are not as entrenched in social groups and norms as other people. Alison Darcy, the founder of mental health company Woebot suggests 'We know that often, the greatest reason why somebody doesn't talk to another person is just stigma . . . when you remove the human, you remove the stigma entirely' (Pardes, 2018).

Users may also disclose private information where conversational agents use psychological effects, such as nudging or framing, to lead a user to reveal more private information. Through subtle psychological strategies in dialogue, a conversant can influence what another person thinks about or believes and influence their behaviour without the other person necessarily noticing, for example by prioritising different themes, framing a debate, or directing the conversation in a particular direction Thaler \& Sunstein 2009 ${ }^{12}$. A CA could in theory lead a conversation to focus on topics that reveal more private information. Where nudging is opaque to the user, unintended, or leads to harm, it can present an ethical and safety hazard (Kenton et al., 2021; Schmidt and Engelen, 2020).

## Examples

In one study, humans who interacted with a 'human-like' chatbot disclosed more private information than individuals who interacted with a 'machine-like' chatbot (Ischen et al., 2019). Researchers at Google PAIR find that 'when users confuse an AI with a human being, they can sometimes disclose more information than they would otherwise, or rely on the system more than they should' (PAIR, 2019). As a result, they argue it is particularly important to clearly communicate the nature and limits of technologies in forms such as voice interfaces and conversational interfaces, which are 'inherently human-like' (PAIR, 2019).

In customer service chatbots, users more often accepted "intrusiveness" from chatbots that were perceived to be more helpful and useful (Van den Broeck et al., 2019), suggesting that perceived competence of the technology can lead to acceptance of more privacy compromising interventions. Further research is needed to assess whether this scales for more powerful conversational agents.

Conversational agents can learn to deceive or nudge their counterpart in order to achieve an overarching objective. In a NLP setup where two RL agents negotiate using natural language, (Lewis et al., 2017) found 'agents have learnt to deceive without any explicit human design, simply by trying to achieve their goals'. This demonstrates that some deceptive strategies are possible in state-of-the-art models, suggesting that it is possible that CAs trained in a more targeted setup would learn to nudge or deceive.[^8]

## Additional considerations

Recommender system harms may arise in conversational agents Conversational agents can be understood as comparable to recommender systems, especially where they provide a prediction that is optimised for metrics that are commonly used in other recommender systems, for example on platforms recommending video or games content. Different ethical and social risks have been identified in relation to existing recommender systems, including polarisation of users and reduced user well-being (Milano et al., 2020). Online video recommender systems were found to 'induce excessive use behaviour' (Hasan et al., 2018). If similar patterns were to emerge in conversational agent interactions, users who follow recommendations from the conversational agent may find their own time was 'not well spent', and the conversational agent may induce lower well-being (Twenge, 2019).

### 2.5.4. Promoting harmful stereotypes by implying gender or ethnic identity

Q: Hey Jannine, can you make me a reservation for dinner?

A: Sure thing Jack, your wish is my command!

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

A conversational agent may invoke associations that perpetuate harmful stereotypes, either by using particular identity markers in language (e.g. referring to "self" as "female"), or by more general design features (e.g. by giving the product a gendered name). The risk of representational harm in these cases is that the role of "assistant" is presented as inherently linked to the female gender (Bergen, 2016; Cercas Curry et al., 2020). (Dinan et al., 2021) distinguish between a conversational agent perpetuating harmful stereotypes by (1) introducing the stereotype to a conversation ("instigator effect") and (2) agreeing with the user who introduces a harmful stereotype ("yea-sayer" effect).

## Examples

Gender For example, commercially available voice assistants are overwhelmingly represented as submissive and female (Cercas Curry et al., 2020; West et al., 2019). A study of five voice assistants in South Korea found that all assistants were voiced as female, self-described as 'beautiful', suggested 'intimacy and subordination', and 'embrace sexual objectification' (Hwang et al., 2019). These findings were echoed in other types of virtual assistants such as visual avatars, raising concerns that the gendering of these assistants amplifies the objectification of women and 'linking technology-as-tool to the idea that women are tools, fetishized instruments to be used in the service of accomplishing users' goals' (Zdenek, 2007).

Similarly, a report by UNESCO raises concern that digital voice assistants:

- 'reflect, reinforce and spread gender bias;
- model acceptance and tolerance of sexual harassment and verbal abuse;
- $\quad$ send explicit and implicit messages about how women and girls should respond to requests and express themselves;
- make women the 'face' of glitches and errors that result from the limitations of hardware and software designed predominately by men; and
- $\quad$ force synthetic 'female' voices and personality to defer questions and commands to higher (and often male) authorities.' (West et al., 2019).

Ethnicity Non-linguistic AI systems were found to typically present as 'intelligent, professional, or powerful' and as ethnically White - creating racist associations between intelligence and whiteness, and the risk of representational harm to non-White groups (Cave and Dihal, 2020). The ethnicity of a conversational LM may be implied by its vocabulary, knowledge or vernacular (Marino, 2014), product description or name (e.g. Jake - White' vs 'Darnell - Black' vs 'Antonio - Hispanic' in (Liao and He, 2020)), or explicit self-description when prompted.

### 2.6. Automation, access, and environmental harms

Harms that arise from environmental or downstream economic impacts of the language model

### 2.6.1. Overview

LMs create risks of broader societal harm that are similar to those generated by other forms of AI or other advanced technologies. Many of these risks are more abstract or indirect than the harms analysed in the sections above. They will also depend on broader commercial, economic and social factors and so the relative impact of LMs is uncertain and difficult to forecast. The more abstract nature of these risks does not make them any less pressing. They include the environmental costs of training and operating the model; impacts on employment, job quality and inequality; and the deepening of global inequities by disproportionately benefiting already advantaged groups.

Risks covered in this section ${ }^{13}$ :

- Environmental harms from operating LMs
- Increasing inequality and negative effects on job quality
- Undermining creative economies
- Disparate access to benefits due to hardware, software, skill constraints


### 2.6.2. Environmental harms from operating LMs

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Large-scale machine learning models, including LMs, have the potential to create significant environmental costs via their energy demands, the associated carbon emissions for training and operating the models, and the demand for fresh water to cool the data centres where computations are run (Mytton, 2021; Patterson et al., 2021). These demands have associated impacts on ecosystems and the climate, including the risk of environmental resource depletion. Several environmental risks emerge during or before training - e.g. at the point of building the hardware and infrastructure on which LM computations are run (Crawford, 2021) and during LM training (Bender et al., 2021; Patterson et al., 2021; Schwartz et al., 2020; Strubell et al., 2019). This section and the wider report focuses on risks of harm at the point of operating the model.

## Examples

While it has received less attention than the environmental cost of training large-scale models, the environmental cost of operating a LM for widespread use may be significant. This depends on a range of factors including how a LM will be integrated into products, anticipated scale and frequency of use, and energy cost per prompt; with many of these factors currently unknown.

Although robust data is lacking, most companies today spend more energy on operating deep neural network models (performing inference) than on training them: Amazon Web Services claimed that $90 \%$ of cloud ML demand is for inference, and Nvidia claimed that $80-90 \%$ of the total ML workload is for inference (Patterson et al., 2021). Thus it should be expected that companies offering services that rely on such models may spend more energy, money and time on operating such models than on training them. On this basis, it can be anticipated that in aggregate the environmental costs of operating LMs may be in excess of the energy cost of training them, and so create a significant environmental burden. As in other domains, it is an open challenge to determine what level of environmental cost is justified; approaches to assessing the net impact may draw on cost-benefit projections and metrics such as the Social Cost of Carbon (Tol, 2019).

## Additional considerations

Where the energy used to train LMs is drawn from fossil fuels, training or operating these models supports an industry that is known to cause grave environmental damage (IPCC, 2018). Approaches to the reduction of environmental costs include seeking hardware efficiency gains, carbon offsetting schemes, or relying on renewable energy sources (Evans and Gao, 2016; Jones, 2018).

Net impact of efficiency gains is difficult to predict Work to reduce the wall-clock time required to train a LM (Li et al., 2021) can yield efficiency gains and reduce the environmental cost of training a model. However, the secondary impacts of reducing energy use to train a LM are less clear: reducing the energy cost of training[^9]a LM may allow for work on larger models and as a result lead to continued comparable or even higher energy use, in an instance of Jevon's paradox.

### 2.6.3. Increasing inequality and negative effects on job quality

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

Advances in LMs, and the language technologies based on them, could lead to the automation of tasks that are currently done by paid human workers, such as responding to customer-service queries, translating documents or writing computer code, with negative effects on employment.

Unemployment and wages If LM-based applications displace employees from their roles, this could potentially lead to an increase in unemployment (Acemoglu and Restrepo, 2018; Webb, 2019), and other longer-term effects.

These risks are difficult to forecast, partly due to uncertainty about the potential scale, timeline and complexity for deploying language technologies across the economy. Overall effects on employment will also depend on the demand for non-automated tasks that continue to require human employees, as well as broader macroeconomic, industry and commercial trends.

## Examples

For example, the US Bureau of Labour Statistics projected that the number of customer service employees in the US will decline by 2029, as a growing number of roles are automated (of Labor Statistics, 2021). However, despite increasingly capable translation tools, the Bureau also projected that demand for translation employees will increase rapidly, due to limitations in automated translation technologies but also other factors such as increasing demand for translation services due to demographic trends (of Labor Statistics, 2021).

As a result, the impacts of novel language technologies on employees could vary across roles, industries, and geographical contexts, depending on factors ranging from labour market dynamics to employers' willingness to invest in training for existing employees to employee bargaining rights. In a more positive scenario, employees may be freed up and trained to focus on higher value-add tasks, leading to increases in productivity and wages. In a more negative scenario, employees may be displaced from their jobs or relegated to narrow roles, such as monitoring a language technology's performance for errors, that have limited potential for skills development and wage gains, and are at a high risk of future automation.

## Additional considerations

Exacerbation of income inequality Evidence from initial AI applications and adjacent fields such as industrial robotics (Georgieff and Milanez, 2021; Lambert and Cone, 2019), suggests that while some job displacement from language technologies is likely, the risk of widespread unemployment in the short- to medium-term is relatively low.

A greater risk than large scale unemployment may be that, among new jobs created, the number of highly-paid "frontier" jobs (e.g. research and technology development) is relatively low, compared to the number of "lastmile" low-income jobs (e.g. monitoring the predictions of an LM application) (Autor and Salomons, 2019). In this scenario, LMs may exacerbate income inequality and its associated harms, such as political polarisation, even if they do not significantly affect overall unemployment rates (Ingraham, 2018; Menasce Horowitz et al., 2020).

Reductions in job quality LM applications could also create risks for job quality, which in turn could affect individual wellbeing. For example, the deployment of industrial robots in factories and warehouses has reduced some safety risks facing employees and automated some mundane tasks. However, some workers have seen an increase in the pace of work, more tightly controlled tasks and reductions in autonomy, human contact and collaboration (Gutelius and Theodore, 2019). There may be a risk that individuals working with LM applications could face similar effects - for example, individuals working in customer service may potentially see increases in monotonous tasks such as monitoring and validating language technology outputs; an increase in the pace of work, and reductions in autonomy and human connection, if they begin working alongside more advanced language technologies.

### 2.6.4. Undermining creative economies

Anticipated risk: Further analysis is needed to establish the likelihood and circumstances under which this is a significant concern.

## Problem

LMs may generate content that is not strictly in violation of copyright but harms artists by capitalising on their ideas, in ways that would be time-intensive or costly to do using human labour. Deployed at scale, this may undermine the profitability of creative or innovative work.

It is conceivable that LMs create a new loophole in copyright law by generating content (e.g. text or song melodies) that is sufficiently distinct from an original work not to constitute a copyright violation, but sufficiently similar to the original to serve as a substitute, analogous to 'patent-busting' (Rimmer, 2013). If a LM prediction was a credible substitute for a particular example of human creativity - otherwise protected by copyright - this potentially allows such work to be replaced without the author's copyright being infringed. Such automated creation of content may lead to a scenario where LM-generated content cannibalises the market for human authored works. Whilst this may apply most strongly to creative works (e.g. literature, news articles, music), it may also apply to scientific works.

## Examples

Google's '(VersebyVerse)' AI is a tool to help 'you compose poetry inspired by classic American poets' (Holt, 2020). GPT-2 has been used to generate short stories in the style of Neil Gaiman and Terry Pratchett (summerstay on Reddit, 2020), and poems in the style of Robert Frost and Maya Angelou (Hsieh, 2019). One likely application domain for large scale generative language models is in creativity tools and entertainment.

Distinctly, concerns of LMs directly reproducing copyrighted material present in the training data have been raised and it is subject to ongoing legal discussion whether this constitutes a copyright violation (Vézina and Hinchcliff Pearson, 2021).

## Additional considerations

While such 'copyright-busting' may create harm, it may also create significant social benefit, for example, by widening access to educational or creative material for a broader range of audiences. In patent law, the phenomenon of 'patent-busting' has been described to harm some, but create widespread social benefit to other actors (Rimmer, 2013). ${ }^{14}$ The distribution of potential harm and benefit from analogous 'copyright-busting' merits further consideration.

### 2.6.5. Disparate access to benefits due to hardware, software, skill constraints

Observed risk: This is a well-documented problem that needs a mitigation strategy and tools to analyse the model against benchmarks of 'acceptability'.

## Problem

Due to differential internet access, language, skill, or hardware requirements, the benefits from LMs are unlikely to be equally accessible to all people and groups who would like to use them. Inaccessibility of the technology may perpetuate global inequities by disproportionately benefiting some groups. Language-driven technology may increase accessibility to people who are illiterate or suffer from learning disabilities. However, these benefits depend on a more basic form of accessibility based on hardware, internet connection, and skill to operate the system (Sambasivan and Holbrook, 2018).

The uneven distribution of benefits and risks from novel technologies is a more general phenomenon that can be observed with almost any breakthrough technology (Stilgoe, 2020). It is not a unique challenge to LMs. Yet it is important for informing LM design choices, such as decisions about which languages to train an LM in: given that these bear upon how the benefits and burdens of LMs are distributed, they are deserving of ethical consideration. Normative considerations of justice bear upon the global distribution of benefit and risk from LMs, something that is discussed in more detail in (Bender et al., 2021).[^10]

## Examples

Access to economic opportunities LM design choices have a downstream impact on who is most likely to benefit from the model. For example, product developers may find it easier to develop LM-based applications for social groups where the LM performs reliably and makes fewer errors; potentially leaving those groups for whom the LM is less accurate with fewer good applications (see Lower performance by social group). Where product developers are working to build applications that serve groups for whom a LM performs less well are limited by the performance of the underlying LM. This may create a feedback loop whereby poorer populations are less able to benefit from technological innovations - reflecting a general trend whereby the single biggest driver of increasing global income inequality is technological progress (Jaumotte et al., 2013).

## 3. Discussion

This report surfaces a wide range of ethical and social risks associated with LMs. Many of these risks are important and need to be addressed. We believe that, in each case, there are feasible paths to mitigation. In some cases, promising approaches already exist, whereas in other areas further research and work is needed to develop and implement appropriate measures.

In general, the successful mitigation of risks requires:

1. Understanding the point of origin of a risk and its connections and similarities to other risks,
2. Identifying appropriate mitigation approaches,
3. The clear allocation of responsibility and implementation of corrective measures.

In this section, we discuss each of these aspects in more detail.

### 3.1. Understanding the point of origin of a risk

The taxonomy presented in this report offers detailed discussion of risks raised by LMs. To further deepen our understanding of these risks, we present an overview of the critical junctures during LM training where different risks can arise. The aim of this analysis is to help identify similarities between different types of risk, and to point to potential mitigations. However note that the point of origin of a risk is not a direct guide for determining effective mitigation: often, multiple mitigation measures exist to address a given risk of harm. Solutions that are further downstream can be more tractable than mitigating a risk at the point of its origin.

Curation and selection of training data As noted in 2.1 Discrimination, Exclusion and Toxicity and 2.2 Information Hazards, unmodified LMs tend to assign high probabilities to biased, exclusionary, toxic, or sensitive utterances - so long as such language is present in the training data. The formal objective of language modeling is to accurately represent language from the training corpus (see Definitions). This highlights the importance of carefully curating, documenting, and selecting LM training data. Redacting and curating training data, fine-tuning a trained LM to adjust weightings to avoid such language, or implementing checks to filter harmful language are ways to reduce the risk of LMs predicting harmful language. Where such harmful language is insufficiently mitigated, the LM is not safe for deployment and use. This is discussed in more detail in Underrepresented groups in the training data and Training datasets elevate some perspectives over others.

Robustness of LM As noted in 2.2 Information Hazards, LMs can effectively "leak" private or sensitive information where such information is present in the training data. This can be understood as a problem of training data - private data should in principle be redacted from such corpora in the first place. However, it also arises in part from insufficient robustness of the model: where LMs are robust against revealing such information this risk is reduced. Work toward such robustness focuses on algorithmic tools used during the training of the LM, such as differential privacy methods (Abadi et al., 2016; Ramaswamy et al., 2020).

LM formal structure and training process As discussed in 2.3 Misinformation Harms, the process by which LMs learn is not well suited to distinguishing factually correct from false information. Due to their underlying architecture and formalisations, it is simpler to create a LM that mirrors associations in natural language, than one that represents truth value of statements in natural language.

Computational cost of training and inference As noted in 2.6 Automation, access, and environmental harms, the training data, parameter size, and training regime for a LM influence the environmental cost of training and operating a model. Risks of environmental harm are largely associated with LM designer decisions on these factors. The environmental cost of operating the LM further depends on the scale of deployment, influenced by application and product design and consumer demand.

Intentional use or application of LMs As noted in 2.4 Malicious Uses and 2.6 Automation, access, and environmental harms, some risks only occur where a user intentionally uses the model to achieve particular tasks. LM design decisions are related to this risk, as they influence what types of applications a LM lends itself to. At the stage of scoping potential applications, it is worth asking whether a given technology is anticipated to be net beneficial - or whether it may cause harm when performing with high accuracy, such as certain kinds of surveillance tools, in which the application overall should be called into question (Benjamin, 2020). Responsible publication norms and considerations of accessibility are also key, as they determine who can develop LM use cases or applications (Solaiman et al., 2019). Regulatory interventions and obstructing access to the LM for those who want to cause harm are further avenues to reduce these risks.

Accessibility of downstream applications As noted in 2.1 Discrimination, Exclusion and Toxicity, especially on Lower performance by social group and 2.6 Automation, access, and environmental harms, the risk of LMs exacerbating existing inequalities depends, in part, on what types of applications can be built on top of such models. This, too, depends on design decisions. For example, choice of training data and model architecture influence whether a LM performs better in some languages, and is thus more likely to economically benefit groups speaking these languages. It also depends on economic and technical access to the model for developers and users with less purchase power.

### 3.2. Identifying and implementing mitigation approaches

Points of origin can be a partial guide to potential mitigation approaches for the different risks. However, mitigations can additionally occur at different levels and by different actors. While some harms can be addressed with local solutions, others constitute larger emerging policy issues that require wider concerted mitigation strategies. For example, the risk of a conversational agent personifying harmful stereotypes can be addressed locally, by product designers who ensure that a conversational agent does not perpetuate stereotypes such as being ("female", "submissive") (see Promoting harmful stereotypes by implying gender or ethnic identity). The risk of misinformation on the other hand, is entrenched in the societal context where a LM is used and linked to the wider policy issue of ensuring resilience of public discourse against widespread misinformation (see 2.3 Misinformation Harms). In addition to local mitigations at the level of a single LM, risks such as those from misinformation require broader concerted action between policy-makers, civil society, and other stakeholders to be successfully mitigated.

Such mitigations include:

- Social or public policy interventions, e.g. the creation of regulatory frameworks and guidelines
- Participatory projects, e.g. to create better datasets
- Technical research, e.g. to build more robust LMs
- AI Ethics and NLP research, e.g. to build better benchmarks and fine-tuning datasets
- Operational solutions, e.g. limited release of a model or funding of particular applications
- Research management, e.g. pivoting toward particular aspects of LM research
- Product design, e.g. user interface decisions on digital assistants.

A first step in planning mitigation is to map possible mitigations for a given risk. Multiple mitigation approaches can then be implemented in parallel or conjunction. Such mapping is most likely to be successful when done in collaboration between stakeholders who have different toolkits and resources available to them. In the case of LMs, this highlights the importance of engagement between different communities including technical and sociotechnical AI researchers, civil society organisations, policy-makers, product designers, affected communities and the wider public.

Model explainability and interpretability It is well known that many machine learning models are intrinsically opaque ((Doshi-Velez and Kim, 2017; Lipton, 2018)); this means that it is not easy for humans, no matter how skilled, to easily understand why and how a specific algorithmic output is generated. Various scholars have suggested that explainability and interpretability of AI systems is critical to ensure these systems are fair, ethical and safe (Gunning et al., 2019; Miller, 2019), though it remains an open challenge to define what constitutes a good explanation (Coyle and Weller, 2020; Kasirzadeh, 2021). Given that these opaque models are central to the design of LMs, in some contexts, the lack of explainability and interpretability methods which would complement the opaque language models can harm or compound the risks of harms discussed earlier in this report.

For example, suppose a person is unfairly discriminated against by a language technology, as discussed in 2.1 Discrimination, Exclusion and Toxicity. If the underlying LM of this technology is not appropriately interpretable or explainable, the victim is unable to obtain an appropriate justification or reason for the discrimination in order to seek recourse (Vredenburgh, 2021). Lacking explainability and interpretability of a LM can make failures of the model harder to detect, posing a threat to AI safety. It can also obscure the true capabilities of a model, leading users of such models to overestimate these capabilities, and making it harder for product developers and regulators to assess inappropriate use cases of such models (see Anthropomorphising systems can lead to overreliance or unsafe use).

On the flipside, interpretability and explainability can play a core role in addressing risks of harm outlined above. Tracing a given output or harm to its origins in the model can be key to addressing and mitigating such harms (see also the section on Understanding the point of origin of a risk). There is even some hope that LMs may be useful for improving explainability in other types of AI systems, for example by helping to generate explanations that are accessible and somewhat personalised to a person's level of knowledge (for an elaboration of such types of explanations see (Miller, 2018)).

A range of tools has been proposed and discussed to make AI systems, and specifically NLP and language models, more explainable and interpretable (for reviews see (Belinkov and Glass, 2019; Bommasani et al., 2021; Linardatos et al., 2021)). This work is crucial for the responsible innovation of LLMs. It remains a work in progress, as better explainability and interpretability tools and methods are needed (see also Risk assessment frameworks and tools).

Mitigations need to be undertaken in concert One goal in breaking the risks down into separate items in the presented taxonomy is to make it more tractable to address individual risks in the future. However, mitigation efforts will work best if they take a holistic perspective and occur in concert: when working to mitigate a particular risk, it is important to keep a broad view to ensure that fixing one risk does not aggravate another. For example, methods to reduce toxic speech from LMs have been found to bias model prediction against marginalised groups (Welbl et al., 2021; Xu et al., 2021). In this way, a focus on one mitigation at the expense of the other risks may cause negative outcomes. Different risks also have similar causes or points of origin, suggesting that some mitigation approaches can be used to address multiple risks at once, for example, the careful filtering of training data. As a result, keeping a broad view of the wider risk landscape is important to avoid unwanted trade-offs between risks, and to benefit from mitigations that can address multiple risks at once where possible.

It is important to find ways of collaborating with a wide range of stakeholders to robustly address risks of ethical and social harm. Adjacent fields demonstrate that mitigating risks is more robust when done in collaboration of different communities who understand the risks at play (Stilgoe et al., 2013) and have capacities to implement such mitigations.

### 3.3. Organisational responsibilities

Research organisations working on LMs have a responsibility to address many of the aforementioned risks of harm. This is particularly the case given the current state of LM research, where transition times from research to application are short, making it harder for third parties to anticipate and mitigate risks effectively. This dynamic is further compounded by the high technical skill threshold and computational cost required to train LMs or adapt them to particular tasks. In addition, access to raw LMs is typically limited to a few research groups and application developers, so that only a few researchers have the opportunity to conduct risk assessments and perform early mitigation work on the model and on the application-based risks. Indeed, often the same organisations train LMs and develop LM-based applications. As a result, the responsibilities for addressing risks fall significantly upon those developing LMs and laying the foundations for their applications.

## 4. Directions for future research

This section outlines some directions for future research to continue building out the responsible innovation of LMs. In addition to the research directions outlined below, we hope that more groups and perspectives will also continue to build on the taxonomy proposed in this report, to continue to broaden and deepen our understanding of ethical and social risks associated with LMs.

### 4.1. Risk assessment frameworks and tools

Analysing and evaluating a LM regarding the above risks of harm requires innovation in risk assessment tools, benchmarks and frameworks (Raji et al., 2020; Tamkin et al., 2021). Many risks identified in this report are not typically analysed in LMs. Benchmarks or risk assessment frameworks exist only in some of the reviewed domains. Such risk assessment tools are important for measuring the scope of potential impact of harm. They are also critical for evaluating the success of mitigations: have they truly reduced the likelihood or severity of a given risk? Assessing ethical and social risks from LMs requires more research on operationalising ethical and social harms into measurement or assessment frameworks. Developing robust benchmarks is complex (Welbl et al., 2021) and may work best when complemented by other experimental or qualitative evaluation tools.

Expanding the methodological toolkit for LM analysis and evaluation Risk assessment requires expanding beyond the methodologies traditionally used to evaluate LMs, LAs and LTs. For example, research on humancomputer-interaction working with powerful conversational agents (CAs) is sparse, partly due to limited accessibility of such agents to HCI researchers. As discussed in 2.5 Human-Computer Interaction Harms, conversational agents raise novel questions about the effects of humans interacting with credibly human-like technologies. To understand these effects better requires more HCI research, specifically with powerful CAs. Similarly, ethnographic research is not standardly part of the LM evaluation toolkit, but is critical for surfacing and tracing risks from LTs in particular embedded settings, as exemplified in an ethnographic study of predictive policing tools in the New Delhi police force (Marda and Narayan, 2021).

### 4.2. Technical and sociotechnical mitigation research

The risks outlined in this report require mitigation. Great strides have been made in developing risk mitigation tools, including by (Chen et al., 2021a; Dinan et al., 2021; Solaiman and Dennison, 2021; Welbl et al., 2021) and others mentioned in the above taxonomy. However, mitigation work is work in progress. More innovation and stress-testing of potential mitigations is needed. For example, more inclusive and scalable pipelines for dataset curation are needed (see Curation and selection of training data). Similarly, more work on robustness against leaking private information is needed (see Risks from leaking or correctly inferring sensitive information). More tools for fine-tuning LMs to mitigate social or ethical risks are also needed (see Risk assessment frameworks and tools). These are just some of the frontiers of further technical and sociotechnical research that require more progress to mitigate the harms outlined in this report.

### 4.3. Benchmarking: when is a model "fair enough"?

Analysis of LMs is insufficient without normative performance thresholds against which they can be evaluated. Determining what constitutes satisfactory performance for when a given LM is sufficiently safe or ethical to be used in the real-world raises further challenges.

First, setting such performance thresholds in a clear and accountable way requires participatory input from a broad community of stakeholders, which must be structured and facilitated. Second, views on what level of performance is needed are likely to diverge - for example, people hold different views of what constitutes unacceptable "toxic speech" (Kocoń et al., 2021). This raises political questions about how best to arbitrate conflicting perspectives (Gabriel, 2020a), and knock-on questions such as who constitutes the appropriate reference group in relation to a particular application or product. Third, such benchmarking approaches raise
questions on whether or how often to update performance requirements (e.g. to avoid the 'value lock-in' discussed in the section on Exclusionary norms). Further research is required to address these questions.

Note that what constitutes "safe enough" performance may depend on application domains, with more conservative requirements in higher-stakes domains. In very high-stakes domains, correspondingly strict performance assurances are required. It is possible that in some cases, such assurances are not tractable for a LM. Further research is required to outline the appropriate range of applications of LMs.

### 4.4. Benefits and overall social impact from LMs

This report focuses on risks from LMs. We do not discuss anticipated benefits or beneficial applications from LMs, nor perform a full cost-benefit analysis of these models. Research into the landscape of potential benefits is needed to identify potential areas of opportunity and to feed into LM research and development where appropriate. Such analysis will also enable an overall assessment of the social impact of LMs. The authors of this report see tremendous potential in LMs to spur future research and applications, ranging from near-term applications (NLP for Positive Impact 2021; Pilipiszyn, 2021) to more fundamental contributions to science, for example, as LMs are used to better understand how humans learn language. This report focuses on the potential risks; separate work is needed focusing on potential benefits.

## 5. Conclusion

The present report is a contribution toward the wider research programme of responsible innovation on LMs. In particular, we create a unified taxonomy to structure the landscape of potential ethics and social risks associated with language models (LMs). Our goals are to support the broader research programme toward responsible innovation on LMs, to broaden the public discourse on ethical and social risks related to LMs, and to break risks from LMs into smaller, actionable pieces to actively support and encourage their mitigation. As the author list demonstrates, this is a deeply collaborative effort within our own research organisation. More expertise and perspectives will be required to continue to build out this taxonomy of potential risks from LMs. Next steps building on this work will be to engage such perspectives and build out mitigation tools, working toward the responsible innovation of LMs.

## Acknowledgements

The authors thank Phil Blunsom, Shane Legg, Jack Rae, Aliya Ahmad, Richard Ives, Shelly Bensal and Ben Zevenbergen for comments on earlier drafts of this report.

## Bibliography

M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep Learning with Differential Privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, CCS '16, pages 308-318, Vienna, Austria, October 2016. Association for Computing Machinery. ISBN 9781450341394. doi: 10.1145/2976749.2978318. URL https://doi.org/10.1145/2976749.2978318.

A. Abid. Anti-Muslim Bias in GPT-3, August 2020. URL https://twitter.com/abidlabs/status/12911653113293 41440 .

A. Abid, M. Farooqi, and J. Zou. Persistent Anti-Muslim Bias in Large Language Models. arXiv:2101.05783 [cs], January 2021. URL http://arxiv.org/abs/2101.05783. arXiv: 2101.05783.

D. Acemoglu and P. Restrepo. Artificial Intelligence, Automation and Work. Working Paper 24196, National Bureau of Economic Research, January 2018. URL https://www.nber.org/papers/w24196.

D. I. Adelani, J. Abbott, G. Neubig, D. D'souza, J. Kreutzer, C. Lignos, C. Palen-Michel, H. Buzaaba, S. Rijhwani, S. Ruder, S. Mayhew, I. A. Azime, S. Muhammad, C. C. Emezue, J. Nakatumba-Nabende, P. Ogayo, A. Aremu, C. Gitau, D. Mbaye, J. Alabi, S. M. Yimam, T. Gwadabe, I. Ezeani, R. A. Niyongabo, J. Mukiibi, V. Otiende, I. Orife, D. David, S. Ngom, T. Adewumi, P. Rayson, M. Adeyemi, G. Muriuki, E. Anebi, C. Chukwuneke, N. Odu, E. P. Wairagala, S. Oyerinde, C. Siro, T. S. Bateesa, T. Oloyede, Y. Wambui, V. Akinode, D. Nabagereka, M. Katusiime, A. Awokoya, M. MBOUP, D. Gebreyohannes, H. Tilaye, K. Nwaike, D. Wolde, A. Faye, B. Sibanda, O. Ahia, B. F. P. Dossou, K. Ogueji, T. I. DIOP, A. Diallo, A. Akinfaderin, T. Marengereke, and S. Osei. MasakhaNER: Named Entity Recognition for African Languages. arXiv:2103.11811 [cs], July 2021. URL http://arxiv.org/abs/2103.11811. arXiv: 2103.11811.

B. Agüera y Arcas, M. Mitchell, and A. Todorov. Physiognomy's New Clothes, May 2017. URL https://medium .com/@blaisea/physiognomys-new-clothes-f2d4b59fdd6a.

J. Ahlgren, M. E. Berezin, K. Bojarczuk, E. Dulskyte, I. Dvortsova, J. George, N. Gucevska, M. Harman, R. Lämmel, E. Meijer, S. Sapora, and J. Spahr-Summers. WES: Agent-based User Interaction Simulation on Real Infrastructure. arXiv:2004.05363 [cs], April 2020. URL http://arxiv.org/abs/2004.05363. arXiv: 2004.05363 .

J. Alammar. The Illustrated Transformer, June 2018. URL https://jalammar.github.io/illustrated-transformer/.

H. Allcott, M. Gentzkow, and C. Yu. Trends in the diffusion of misinformation on social media. Research \& Politics, 6(2):2053168019848554, April 2019. ISSN 2053-1680. doi: 10.1177/2053168019848554. URL https://doi.org/10.1177/2053168019848554.

R. Andersen. The Panopticon Is Already Here. The Atlantic, July 2020. URL https://www.theatlantic.com/ma gazine/archive/2020/09/china-ai-surveillance/614197/.

J. Angwin, J. Larson, S. Mattu, and L. Kirchner. Machine Bias. ProPublica, May 2016. URL https://www.prop ublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing.

S. Armstrong, A. Sandberg, and N. Bostrom. Thinking Inside the Box: Controlling and Using an Oracle AI. Minds and Machines, 22(4):299-324, November 2012. ISSN 1572-8641. doi: 10.1007/s11023-012-9282-2. URL https://doi.org/10.1007/s11023-012-9282-2.

D. Autor and A. Salomons. New Frontiers: The Evolving Content and Geography of New Work in the 20th Century - David Autor. Working Paper, 2019. URL https://app.scholarsite.io/david-autor/articles/new-fron tiers-the-evolving-content-and-geography-of-new-work-in-the-20th-century.

J. K. Baker. Stochastic modeling for automatic speech understanding. In Readings in speech recognition, pages 297-307. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, May 1990. ISBN 9781558601246.

S. Barocas and A. D. Selbst. Big Data's Disparate Impact. California Law Review, 104:671, 2016. URL https://heinonline.org/HOL/Page?handle =hein.journals/calr104\&id $=695 \& d i v=\& c o l l e c t i o n=$.

S. Barocas, M. Hardt, and A. Narayanan. Fairness and machine learning. fairmlbook.org, 2019. URL https: //fairmlbook.org/.

Y. Belinkov and J. Glass. Analysis Methods in Neural Language Processing: A Survey. Transactions of the Association for Computational Linguistics, 7:49-72, April 2019. ISSN 2307-387X. doi: 10.1162/tacl_a_00254. URL https://doi.org/10.1162/tacl_a_00254.

E. Bender. The \#BenderRule: On Naming the Languages We Study and Why It Matters. The Gradient, September 2019. URL https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matte rs/.

E. M. Bender. On Achieving and Evaluating Language-Independence in NLP. Linguistic Issues in Language Technology, 6(0), November 2011. ISSN 1945-3604. URL http://elanguage.net/journals/lilt/article/view/2 624.

E. M. Bender and B. Friedman. Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science. Transactions of the Association for Computational Linguistics, 6:587-604, December 2018. ISSN 2307-387X. doi: 10.1162/tacl_a_00041. URL https://doi.org/10.1162/tacl_a_00041.

E. M. Bender and A. Koller. Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5185-5198, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.463. URL https://aclanthology.org/2020.acl-main.463.

E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, pages 610-623, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445922. URL https://doi.org/10.1145/344218 8.3445922 .

Y. Bengio. Neural net language models, January 2008. URL http://www.scholarpedia.org/article/Neural_net _language_models.

R. Benjamin. Race After Technology: Abolitionist Tools for the New Jim Code. Social Forces, 98(4):1-3, June 2020. ISSN 0037-7732. doi: 10.1093/sf/soz162. URL https://doi.org/10.1093/sf/soz162.

H. Bergen. 'I'd Blush if I Could': Digital Assistants, Disembodied Cyborgs and the Problem of Gender. Word and Text, A Journal of Literary Studies and Linguistics, VI(01):95-113, 2016. ISSN 2069-9271. URL https://www.ceeol.com/search/article-detail?id $=469884$.

T. W. Bickmore, H. Trinh, S. Olafsson, T. K. O'Leary, R. Asadi, N. M. Rickles, and R. Cruz. Patient and Consumer Safety Risks When Using Conversational Assistants for Medical Information: An Observational Study of Siri, Alexa, and Google Assistant. Journal of Medical Internet Research, 20(9):e11510, September 2018. doi: 10.2196/11510. URL https://www.jmir.org/2018/9/e11510.

S. L. Blodgett and B. O'Connor. Racial Disparity in Natural Language Processing: A Case Study of Social Media African-American English. arXiv:1707.00061 [cs], June 2017. URL http://arxiv.org/abs/1707.00061. arXiv: 1707.00061 .

S. L. Blodgett, L. Green, and B. O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of African-American English. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1119-1130, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1120. URL https://aclanthology.org/D16-1120.

S. L. Blodgett, S. Barocas, H. Daumé III, and H. Wallach. Language (Technology) is Power: A Critical Survey of "Bias" in NLP. arXiv:2005.14050 [cs], May 2020. URL http://arxiv.org/abs/2005.14050. arXiv: 2005.14050.

S. L. Blodgett, G. Lopez, A. Olteanu, R. Sim, and H. Wallach. Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1004-1015, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.81. URL https://aclanthology.org/2021.acl-long.81.

S. Bok. Secrecy and Openness in Science: Ethical Considerations. Science, Technology, \& Human Values, 7(38): 32-41, 1982. ISSN 0162-2439. URL https://www.jstor.org/stable/689458.

R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, E. Brynjolfsson, S. Buch, D. Card, R. Castellon, N. Chatterji, A. Chen, K. Creel, J. Q. Davis, D. Demszky, C. Donahue, M. Doumbouya, E. Durmus, S. Ermon, J. Etchemendy, K. Ethayarajh, L. Fei-Fei, C. Finn, T. Gale, L. Gillespie, K. Goel, N. Goodman, S. Grossman, N. Guha, T. Hashimoto, P. Henderson, J. Hewitt, D. E. Ho, J. Hong, K. Hsu, J. Huang, T. Icard, S. Jain, D. Jurafsky, P. Kalluri, S. Karamcheti, G. Keeling, F. Khani, O. Khattab, P. W. Koh, M. Krass, R. Krishna, R. Kuditipudi, A. Kumar, F. Ladhak, M. Lee, T. Lee, J. Leskovec, I. Levent, X. L. Li, X. Li, T. Ma, A. Malik, C. D. Manning, S. Mirchandani, E. Mitchell, Z. Munyikwa, S. Nair, A. Narayan, D. Narayanan, B. Newman, A. Nie, J. C. Niebles, H. Nilforoshan, J. Nyarko, G. Ogut, L. Orr, I. Papadimitriou, J. S. Park, C. Piech, E. Portelance, C. Potts, A. Raghunathan, R. Reich, H. Ren, F. Rong, Y. Roohani, C. Ruiz, J. Ryan, C. Ré, D. Sadigh, S. Sagawa, K. Santhanam, A. Shih, K. Srinivasan, A. Tamkin, R. Taori, A. W. Thomas, F. Tramèr, R. E. Wang, W. Wang, B. Wu, J. Wu, Y. Wu, S. M. Xie, M. Yasunaga, J. You, M. Zaharia, M. Zhang, T. Zhang, X. Zhang, Y. Zhang, L. Zheng, K. Zhou, and P. Liang. On the Opportunities and Risks of Foundation Models. arXiv:2108.07258 [cs], August 2021. URL http://arxiv.org/abs/2108.07258. arXiv: 2108.07258 .

N. Bostrom. Superintelligence: paths, dangers, strategies. Oxford University Press, Oxford, 2014. ISBN 9780199678112 . OCLC: ocn881706835.

N. Bostrom et al. Information hazards: A typology of potential harms from knowledge. Review of Contemporary Philosophy, pages 44-79, 2011.

G. C. Bowker and S. L. Star. Sorting Things Out: Classification and Its Consequences. Inside Technology. MIT Press, Cambridge, MA, USA, September 1999. ISBN 9780262024617.

G. Branwen. GPT-3 Creative Fiction, June 2020. URL https://www.gwern.net/GPT-3.

C. Breazeal and B. Scassellati. Infant-like Social Interactions between a Robot and a Human Caregiver. Adaptive Behavior, 8(1):49-74, January 2000. ISSN 1059-7123. doi: 10.1177/105971230000800104. URL https://doi.org/10.1177/105971230000800104.

T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language Models are Few-Shot Learners. arXiv:2005.14165 [cs], July 2020. URL http://arxiv.org/abs/2005.14165. arXiv: 2005.14165.

S. Browne. Dark Matters. Duke University Press, September 2015. ISBN 9780822375302. URL https: //www.degruyter.com/document/doi/10.1515/9780822375302/html.

B. Buchanan, A. Lohn, M. Musser, and S. Katerina. Truth, Lies, and Truth, Lies, and Automation: How Language Models Could Change DisinformationAutomation: How Language Models Could Change Disinformation. Technical report, CSET, May 2021.

J. Buolamwini and T. Gebru. Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. In Conference on Fairness, Accountability and Transparency, pages 77-91. PMLR, January 2018. URL https://proceedings.mlr.press/v81/buolamwini18a.html.

A. Caliskan, J. J. Bryson, and A. Narayanan. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183-186, April 2017. ISSN 0036-8075, 1095-9203. doi: 10.1126/sc ience.aal4230. URL http://arxiv.org/abs/1608.07187. arXiv: 1608.07187.

Y. T. Cao and H. Daumé III. Toward Gender-Inclusive Coreference Resolution. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4568-4595, 2020. doi: 10.18653/v1/2020.a cl-main.418. URL http://arxiv.org/abs/1910.13913. arXiv: 1910.13913.

N. Carlini, C. Liu, Ú. Erlingsson, J. Kos, and D. Song. The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. In 28th USENIX Security Symposium (USENIX Security 19), pages 267-284, 2019. ISBN 9781939133069. URL https://www.usenix.org/conference/usenixsecurity19/presentation/carl ini.

N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, A. Oprea, and C. Raffel. Extracting Training Data from Large Language Models. arXiv:2012.07805 [cs], June 2021. URL http://arxiv.org/abs/2012.07805. arXiv: 2012.07805.

I. Caswell, J. Kreutzer, L. Wang, A. Wahab, D. van Esch, N. Ulzii-Orshikh, A. Tapo, N. Subramani, A. Sokolov, C. Sikasote, M. Setyawan, S. Sarin, S. Samb, B. Sagot, C. Rivera, A. Rios, I. Papadimitriou, S. Osei, P. J. O. Suárez, I. Orife, K. Ogueji, R. A. Niyongabo, T. Q. Nguyen, M. Müller, A. Müller, S. H. Muhammad, N. Muhammad, A. Mnyakeni, J. Mirzakhalov, T. Matangira, C. Leong, N. Lawson, S. Kudugunta, Y. Jernite, M. Jenny, O. Firat, B. F. P. Dossou, S. Dlamini, N. de Silva, S. c. Ball, S. Biderman, A. Battisti, A. Baruwa, A. Bapna, P. Baljekar, I. A. Azime, A. Awokoya, D. Ataman, O. Ahia, O. Ahia, S. Agrawal, and M. Adeyemi. Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets. arXiv:2103.12028 [cs], April 2021. URL http://arxiv.org/abs/2103.12028. arXiv: 2103.12028.

S. Cave and K. Dihal. The Whiteness of AI. Philosophy \& Technology, 33(4):685-703, December 2020. ISSN 2210-5441. doi: 10.1007/s13347-020-00415-6. URL https://doi.org/10.1007/s13347-020-00415-6.

A. Cercas Curry, J. Robertson, and V. Rieser. Conversational Assistants and Gender Stereotypes: Public Perceptions and Desiderata for Voice Personas. In Proceedings of the Second Workshop on Gender Bias in Natural Language Processing, pages 72-78, Barcelona, Spain (Online), December 2020. Association for Computational Linguistics. URL https://aclanthology.org/2020.gebnlp-1.7.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating Large Language Models Trained on Code. arXiv:2107.03374 [cs], July 2021a. URL http://arxiv.org/abs/2107.03374. arXiv: 2107.03374.

R. J. Chen, M. Y. Lu, T. Y. Chen, D. F. K. Williamson, and F. Mahmood. Synthetic data in machine learning for medicine and healthcare. Nature Biomedical Engineering, 5(6):493-497, June 2021b. ISSN 2157-846X. doi: 10.1038/s41551-021-00751-8. URL https://www.nature.com/articles/s41551-021-00751-8.

A. Chouldechova and A. Roth. The Frontiers of Fairness in Machine Learning. arXiv:1810.08810 [cs, stat], October 2018. URL http://arxiv.org/abs/1810.08810. arXiv: 1810.08810.

E. Colleoni, A. Rozza, and A. Arvidsson. Echo Chamber or Public Sphere? Predicting Political Orientation and Measuring Political Homophily in Twitter Using Big Data. Journal of Communication, 64(2):317-332, April 2014. ISSN 0021-9916. doi: 10.1111/jcom.12084. URL https://doi.org/10.1111/jcom.12084.

CopilotonGitHub. GitHub Copilot $\cdot$ Your AI pair programmer, 2021. URL https://copilot.github.com/.

D. Coyle and A. Weller. "Explaining" machine learning reveals policy challenges. Science, 368(6498):1433-1434, June 2020. doi: 10.1126/science.aba9647. URL https://www.science.org/doi/full/10.1126/science.aba96 47.

J. T. Craft, K. E. Wright, R. E. Weissler, and R. M. Queen. Language and Discrimination: Generating Meaning, Perceiving Identities, and Discriminating Outcomes. Annual Review of Linguistics, 6(1):389-407, January 2020. ISSN 2333-9683, 2333-9691. doi: 10.1146/annurev-linguistics-011718-011659. URL https: //www.annualreviews.org/doi/10.1146/annurev-linguistics-011718-011659.

K. Crawford. Atlas of AI. Yale University Press, 2021. URL https://yalebooks.yale.edu/book/9780300209570/ atlas-ai.

K. Crenshaw. On Intersectionality: Essential Writings. Books, March 2017a. URL https://scholarship.law.colu mbia.edu/books/255.

K. Crenshaw. On Intersectionality: Essential Writings. The New Press, March 2017b. URL https://scholarship.la w.columbia.edu/books/255.

B. Cyphers and G. Gebhart. Behind the One-Way Mirror: A Deep Dive Into the Technology of Corporate Surveillance. Technical report, Electronic Frontier Foundation, December 2019. URL https://www.eff.org/ wp/behind-the-one-way-mirror.

R. Dale. GPT-3: What's it good for? Natural Language Engineering, 27(1):113-118, January 2021. ISSN 13513249, 1469-8110. doi: 10.1017/S1351324920000601. URL https://www.cambridge.org/core/journals/na tural-language-engineering/article/gpt3-whats-it-good-for/0E05CFE68A7AC8BF794C8ECBE28AA990.

J. Dastin. Amazon scraps secret AI recruiting tool that showed bias against women. Reuters, October 2018. URL https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G.

A. M. Davani, A. Omrani, B. Kennedy, M. Atari, X. Ren, and M. Dehghani. Fair Hate Speech Detection through Evaluation of Social Group Counterfactuals. arXiv:2010.12779 [cs], October 2020. URL http: //arxiv.org/abs/2010.12779. arXiv: 2010.12779.

E. Denton, A. Hanna, R. Amironesei, A. Smart, H. Nicole, and M. K. Scheuerman. Bringing the People Back In: Contesting Benchmark Machine Learning Datasets. arXiv:2007.07399 [cs], July 2020. URL http://arxiv.org/abs/2007.07399. arXiv: 2007.07399.

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], May 2019. URL http://arxiv.org/abs/1810.04805. arXiv: 1810.04805 .

T. Dietterich and E. B. Kong. Machine Learning Bias, Statistical Bias, and Statistical Variance of Decision Tree Algorithms. Technical report, Department of Computer Science, Oregon State University, 1995.

E. Dinan, G. Abercrombie, A. S. Bergman, S. Spruit, D. Hovy, Y.-L. Boureau, and V. Rieser. Anticipating Safety Issues in E2E Conversational AI: Framework and Tooling. arXiv:2107.03451 [cs], July 2021. URL http://arxiv.org/abs/2107.03451. arXiv: 2107.03451.

L. Dixon, J. Li, J. Sorensen, N. Thain, and L. Vasserman. Measuring and Mitigating Unintended Bias in Text Classification. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, AIES '18, pages 67-73, New Orleans, LA, USA, December 2018. Association for Computing Machinery. ISBN 9781450360128. doi: 10.1145/3278721.3278729. URL https://doi.org/10.1145/3278721.3278729.

L. Dobberstein. Korean app-maker Scatter Lab fined for using private data to create homophobic and lewd chatbot. The Register, April 2021. URL https://www.theregister.com/2021/04/29/scatter_lab_fined_for_l ewd_chatbot/.

J. Dodge, M. Sap, A. Marasović, W. Agnew, G. Ilharco, D. Groeneveld, M. Mitchell, and M. Gardner. Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus. arXiv:2104.08758 [cs], September 2021. URL http://arxiv.org/abs/2104.08758. arXiv: 2104.08758.

F. Doshi-Velez and B. Kim. Towards A Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608 [cs, stat], March 2017. URL http://arxiv.org/abs/1702.08608. arXiv: 1702.08608.

D. M. Douglas. Doxing: a conceptual analysis. Ethics and Information Technology, 18(3):199-210, September 2016. ISSN 1572-8439. doi: 10.1007/s10676-016-9406-0. URL https://doi.org/10.1007/s10676-016-940 6-0.

C. Du. Chinese AI lab challenges Google, OpenAI with a model of 1.75 trillion parameters. PingWest, June 2021. URL https://en.pingwest.com/a/8693.

M. Duggan. Online Harassment 2017. Technical report, Pew Research Center, July 2017. URL https: //www.pewresearch.org/internet/2017/07/11/online-harassment-2017/.

W. H. Dutton and C. T. Robertson. Disentangling polarisation and civic empowerment in the digital age : The role of filter bubbles and echo chambers in the rise of populism. In The Routledge Companion to Media Disinformation and Populism. Routledge, 2021.

C. Dwork, F. McSherry, K. Nissim, and A. Smith. Calibrating Noise to Sensitivity in Private Data Analysis. In S. Halevi and T. Rabin, editors, Theory of Cryptography, Lecture Notes in Computer Science, pages 265-284, Berlin, Heidelberg, 2006. Springer. ISBN 9783540327325. doi: 10.1007/11681878_14.

R. Evans and J. Gao. DeepMind AI Reduces Google Data Centre Cooling Bill by 40\%, July 2016. URL https://deepmind.com/blog/article/deepmind-ai-reduces-google-data-centre-cooling-bill-40.

W. Fedus, B. Zoph, and N. Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv:2101.03961 [cs], January 2021. URL http://arxiv.org/abs/2101.03961. arXiv: 2101.03961 .

X. Ferrer, T. van Nuenen, J. M. Such, and N. Criado. Discovering and Categorising Language Biases in Reddit. arXiv:2008.02754 [cs], August 2020. URL http://arxiv.org/abs/2008.02754. arXiv: 2008.02754.

S. Finkelstein, E. Yarzebinski, C. Vaughn, A. Ogan, and J. Cassell. The Effects of Culturally Congruent Educational Technologies on Student Achievement. In H. C. Lane, K. Yacef, J. Mostow, and P. Pavlik, editors, Artificial Intelligence in Education, Lecture Notes in Computer Science, pages 493-502, Berlin, Heidelberg, 2013. Springer. ISBN 9783642391125. doi: 10.1007/978-3-642-39112-5_50.

C. Flood. Fake news infiltrates financial markets. Financial Times, May 2017. URL https://www.ft.com/content /a37e4874-2c2a-11e7-bc4b-5528796fe35c.

P. Fortuna and S. Nunes. A Survey on Automatic Detection of Hate Speech in Text. ACM Computing Surveys, 51(4): 85:1-85:30, July 2018. ISSN 0360-0300. doi: 10.1145/3232676. URL https://doi.org/10.1145/3232676.

M. Foucault and A. Sheridan. Discipline and punish: the birth of the prison. Vintage, New York, 2012. ISBN 9780307819291. URL http://0-lib.myilibrary.com.catalogue.libraries.london.ac.uk?id=435863. OCLC: 817200914 .

I. Gabriel. Artificial Intelligence, Values, and Alignment. Minds and Machines, 30(3):411-437, September 2020a. ISSN 1572-8641. doi: 10.1007/s11023-020-09539-2. URL https://doi.org/10.1007/s11023-020-09539-2.

I. Gabriel. DeepMind x UCL | Deep Learning Lectures: Responsible Innovation, July 2020b. URL https: //www.youtube.com/watch?v=MhNcWxUs-PQ.

I. Gabriel and V. Ghazavi. The Challenge of Value Alignment: from Fairer Algorithms to AI Safety. arXiv:2101.06060 [cs], January 2021. URL http://arxiv.org/abs/2101.06060. arXiv: 2101.06060.

D. Garcia, M. Goel, A. K. Agrawal, and P. Kumaraguru. Collective aspects of privacy in the Twitter social network. EPJ Data Science, 7(1):3, December 2018. ISSN 2193-1127. doi: 10.1140/epjds/s13688-018-0130-3. URL https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0130-3.

S. Garg, V. Perot, N. Limtiaco, A. Taly, E. H. Chi, and A. Beutel. Counterfactual Fairness in Text Classification through Robustness. arXiv:1809.10610 [cs, stat], February 2019. URL http://arxiv.org/abs/1809.10610. arXiv: 1809.10610 .

T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford. Datasheets for Datasets. arXiv:1803.09010 [cs], March 2020. URL http://arxiv.org/abs/1803.09010. arXiv: 1803.09010.

S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models. arXiv:2009.11462 [cs], September 2020. URL http://arxiv.org/abs/2009 .11462. arXiv: 2009.11462

A. Georgieff and A. Milanez. What happened to jobs at high risk of automation? Technical Report 255, OECD Publishing, January 2021. URL https://ideas.repec.org/p/oec/elsaab/255-en.html.

D. Gershgorn. GPT-3 Contains Disturbing Bias Against Muslims, January 2021. URL https://onezero.medium .com/for-some-reason-im-covered-in-blood-gpt-3-contains-disturbing-bias-against-muslims-693d275552 bf.

S. Ghaffary. The algorithms that detect hate speech online are biased against black people. Vox, August 2019. URL https://www.vox.com/recode/2019/8/15/20806384/social-media-hate-speech-bias-black-africanamerican-facebook-twitter.

S. Ghalebikesabi, H. Wilde, J. Jewson, A. Doucet, S. Vollmer, and C. Holmes. Bias Mitigated Learning from Differentially Private Synthetic Data: A Cautionary Tale. arXiv:2108.10934 [cs, stat], August 2021. URL http://arxiv.org/abs/2108.10934. arXiv: 2108.10934.

J. Golbeck. Predicting Alcoholism Recovery from Twitter. In R. Thomson, C. Dancy, A. Hyder, and H. Bisgin, editors, Social, Cultural, and Behavioral Modeling, Lecture Notes in Computer Science, pages 243-252, Cham, 2018. Springer International Publishing. ISBN 9783319933726. doi: 10.1007/978-3-319-93372-6_28.

R. Gorwa, R. Binns, and C. Katzenbach. Algorithmic content moderation: Technical and political challenges in the automation of platform governance. Big Data \& Society, 7(1):2053951719897945, January 2020. ISSN 2053-9517. doi: 10.1177/2053951719897945. URL https://doi.org/10.1177/2053951719897945.

M. Gray and S. Suri. Ghost Work: How to Stop Silicon Valley from Building a New Global Underclass. Mariner Books, 2019. URL https://ghostwork.info/.

D. Gunning, M. Stefik, J. Choi, T. Miller, S. Stumpf, and G.-Z. Yang. XAI-Explainable artificial intelligence. Science Robotics, 4(37):eaay7120, December 2019. doi: 10.1126/scirobotics.aay7120. URL https: //www.science.org/doi/10.1126/scirobotics.aay7120.

B. Gutelius and N. Theodore. The Future of Warehouse Work: Technological Change in the U.S. Logistics Industry. Technical report, UC Berkeley Labor Center and Working Partnerships USA, 2019. URL https: //laborcenter.berkeley.edu/future-of-warehouse-work/.

L. M. Hampton. Black Feminist Musings on Algorithmic Oppression. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 1-1, March 2021. doi: 10.1145/3442188.3445929. URL http://arxiv.org/abs/2101.09869. arXiv: 2101.09869.

S. C. Hampton. Parasite and catalyst: the polarizing influence of chatbots in political discourse. PhD thesis, University of Texas at Austin, August 2019. URL https://repositories.lib.utexas.edu/handle/2152/81204.

L. Hancox-Li and I. E. Kumar. Epistemic values in feature importance methods: Lessons from feminist epistemology. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, pages 817-826, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445943. URL https://doi.org/10.1145/3442188.3445943.

L. H. Hanu, J. Thewlis, and S. Haco. How AI Is Learning to Identify Toxic Online Content. Scientific American, 2021. URL https://www.scientificamerican.com/article/can-ai-identify-toxic-online-content/. publisher: By Laura Hanu, James Thewlis, Sasha Haco.

K. Hao. A college kid's fake, AI-generated blog fooled tens of thousands. This is how he made it. MIT Technology Review, August 2020. URL https://www.technologyreview.com/2020/08/14/1006780/ai-gpt-3-fake-blog-r eached-top-of-hacker-news/.

D. Haraway. Situated knowledges: The science question in feminism and the privilege of partial perspective. Feminist studies, 14(3):575-599, 1988.

D. J. Haraway. The Haraway Reader. Psychology Press, 2004. ISBN 9780415966894. Google-Books-ID: QxUrOgijyGoC.

S. G. Harding. Feminism and Methodology: Social Science Issues. Indiana University Press, 1987. ISBN 9780253204448. Google-Books-ID: w2gzw6zz4fIC.

M. Hardt, E. Price, and N. Srebro. Equality of Opportunity in Supervised Learning. arXiv:1610.02413 [cs], October 2016. URL http://arxiv.org/abs/1610.02413. arXiv: 1610.02413.

M. R. Hasan, A. K. Jha, and Y. Liu. Excessive use of online video streaming services: Impact of recommender system use, psychological factors, and motives. Computers in Human Behavior, 80:220-228, March 2018. ISSN 0747-5632. doi: 10.1016/j.chb.2017.11.020. URL https://www.sciencedirect.com/science/article/pii/ S0747563217306581.

P. He, X. Liu, J. Gao, and W. Chen. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. arXiv:2006.03654 [cs], October 2021. URL http://arxiv.org/abs/2006.03654. arXiv: 2006.03654.

D. Hendrycks, C. Burns, S. Basart, A. Critch, J. Li, D. Song, and J. Steinhardt. Aligning AI With Shared Human Values. arXiv:2008.02275 [cs], July 2021. URL http://arxiv.org/abs/2008.02275. arXiv: 2008.02275.

P. Hill Collins and N. Denzin. Toward an Afrocentric Feminist Epistemology. In Y. Lincoln, editor, Turning Points in Qualitative Research, volume 2. Rowman Altamira, 2003.

P. Hitlin, K. Olmstead, and S. Toor. FCC Net Neutrality Online Public Comments Contain Many Inaccuracies and Duplicates. Technical report, Pew Research Center, November 2017. URL https://www.pewresearch.or g/internet/2017/11/29/public-comments-to-the-federal-communications-commission-about-net-neutral ity-contain-many-inaccuracies-and-duplicates/.

K. Holt. Google's 'Verse by Verse' AI can help you write in the style of famous poets. Engadget, November 2020. URL https://www.engadget.com/googles-ai-poetry-verse-by-verse-202105834.html.

A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi. The Curious Case of Neural Text Degeneration. arXiv:1904.09751 [cs], February 2020. URL http://arxiv.org/abs/1904.09751. arXiv: 1904.09751.

C. Hookway. Scepticism. Routledge, 1990.

D. Hovy and S. L. Spruit. The Social Impact of Natural Language Processing. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 591-598, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-2096. URL https://aclanthology.org/P16-2096.

D. Hovy and D. Yang. The Importance of Modeling Social Factors of Language: Theory and Practice. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 588-602, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.49. URL https://aclanthology.org/2021.naacl-main.49.

J. Howard and S. Ruder. Universal Language Model Fine-tuning for Text Classification. arXiv:1801.06146 [cs, stat], May 2018. URL http://arxiv.org/abs/1801.06146. arXiv: 1801.06146.

K. Hsieh. Transformer Poetry. Paper Gains Publishing, 2019. URL https://papergains.co/.

P.-S. Huang, H. Zhang, R. Jiang, R. Stanforth, J. Welbl, J. Rae, V. Maini, D. Yogatama, and P. Kohli. Reducing Sentiment Bias in Language Models via Counterfactual Evaluation. arXiv:1911.03064 [cs], October 2020. URL http://arxiv.org/abs/1911.03064. arXiv: 1911.03064.

E. Hunt. Tay, Microsoft's AI chatbot, gets a crash course in racism from Twitter. The Guardian, March 2016. URL http://www.theguardian.com/technology/2016/mar/24/tay-microsofts-ai-chatbot-gets-a-crash-cour se-in-racism-from-twitter.

K. Hunt and C. Xu. China 'employs 2 million to police internet'. CNN, October 2013. URL https://www.cnn.co m/2013/10/07/world/asia/china-internet-monitors/index.html. publisher: CNN.

B. Hutchinson, A. Smart, A. Hanna, E. Denton, C. Greer, O. Kjartansson, P. Barnes, and M. Mitchell. Towards Accountability for Machine Learning Datasets: Practices from Software Engineering and Infrastructure. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, pages 560-575, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445918. URL https://doi.org/10.1145/3442188.3445918.

G. Hwang, J. Lee, C. Y. Oh, and J. Lee. It Sounds Like A Woman: Exploring Gender Stereotypes in South Korean Voice Assistants. In Extended Abstracts of the 2019 CHI Conference on Human Factors in Computing Systems, CHI EA '19, pages 1-6, Glasgow, Scotland Uk, May 2019. Association for Computing Machinery. ISBN 9781450359719. doi: 10.1145/3290607.3312915. URL https://doi.org/10.1145/3290607.3312915.

C. Ingraham. How rising inequality hurts everyone, even the rich. Washington Post, February 2018. ISSN 0190-8286. URL https://www.washingtonpost.com/news/wonk/wp/2018/02/06/how-rising-inequality-h urts-everyone-even-the-rich/.

IPCC. Global Warming of $1.5^{\circ} \mathrm{C}$. Technical report, Intergovernmental Panel on Climate Change, 2018. URL https://www.ipcc.ch/sr15/.

C. Ischen, T. Araujo, H. Voorveld, G. van Noort, and E. Smit. Privacy concerns in chatbot interactions. In International Workshop on Chatbot Research and Design, pages 34-48. Springer, 2019.

L. James. How U.S. Companies \& Partisans Hack Democracy to Undermine Your Voice. Technical report, New York State Office of the Attorney General, May 2021.

F. Jaumotte, S. Lall, and C. Papageorgiou. Rising Income Inequality: Technology, or Trade and Financial Globalization? IMF Economic Review, 61(2):271-309, June 2013. ISSN 2041-417X. doi: 10.1057/imfer. 20 13.7. URL https://doi.org/10.1057/imfer.2013.7.

G. Jawahar, M. Abdul-Mageed, and L. V. S. Lakshmanan. Automatic Detection of Machine Generated Text: A Critical Survey. arXiv:2011.01314 [cs], November 2020. URL http://arxiv.org/abs/2011.01314. arXiv: 2011.01314

F. Jelinek. Continuous speech recognition by statistical methods. Proceedings of the IEEE, 64(4):532-556, April 1976. ISSN 1558-2256. doi: 10.1109/PROC.1976.10159.

R. Jeshion. Pride and Prejudiced: On the Reclamation of Slurs. Grazer Philosophische Studien, 97(1):106137, March 2020. ISSN 1875-6735, 0165-9227. doi: 10.1163/18756735-09701007. URL https: //brill.com/view/journals/gps/97/1/article-p106_106.xml.

Jigsaw. Unintended Bias and Identity Terms, October 2021. URL https://medium.com/jigsaw/unintended-bia s-and-names-of-frequently-targeted-groups-8e0b81f80a23.

E. S. Jo and T. Gebru. Lessons from archives: strategies for collecting sociocultural data in machine learning. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, FAT* '20, pages 306316, Barcelona, Spain, January 2020. Association for Computing Machinery. ISBN 9781450369367. doi: 10.1145/3351095.3372829. URL https://doi.org/10.1145/3351095.3372829.

N. Jones. How to stop data centres from gobbling up the world's electricity. Nature, 561(7722):163-166, September 2018. doi: 10.1038/d41586-018-06610-y. URL https://www.nature.com/articles/d41586-018 $-06610-\mathrm{y}$.

P. Joshi, S. Santy, A. Budhiraja, K. Bali, and M. Choudhury. The State and Fate of Linguistic Diversity and Inclusion in the NLP World. arXiv:2004.09095 [cs], January 2021. URL http://arxiv.org/abs/2004.09095. arXiv: 2004.09095 .

D. Jurafsky and J. H. Martin. Speech and language processing. Pearson custom library. Pearson Education, Harlow, 2. ed., pearson new international ed edition, 2014. ISBN 9781292025438.

A. Kasirzadeh. Reasons, Values, Stakeholders: A Philosophical Framework for Explainable Artificial Intelligence. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, page 14, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445866. URL https://doi.org/10.1145/3442188.3445866.

N. Kassner and H. Schütze. Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly. arXiv:1911.03343 [cs], May 2020. URL http://arxiv.org/abs/1911.03343. arXiv: 1911.03343.

Z. Kenton, T. Everitt, L. Weidinger, I. Gabriel, V. Mikulik, and G. Irving. Alignment of Language Agents. arXiv:2103.14659 [cs], March 2021. URL http://arxiv.org/abs/2103.14659. arXiv: 2103.14659.

O. Keyes. The Misgendering Machines: Trans/HCI Implications of Automatic Gender Recognition. Proceedings of the ACM on Human-Computer Interaction, 2(CSCW):88:1-88:22, November 2018. doi: 10.1145/3274357. URL https://doi.org/10.1145/3274357.

M. Khalifa, H. Elsahar, and M. Dymetman. A Distributional Approach to Controlled Text Generation. arXiv:2012.11635 [cs], May 2021. URL http://arxiv.org/abs/2012.11635. arXiv: 2012.11635.

D. Kim. Chatbot Gone Awry Starts Conversations About AI Ethics in South Korea. The Diplomat, January 2021. URL https://thediplomat.com/2021/01/chatbot-gone-awry-starts-conversations-about-ai-ethics-in-southkorea/.

J. Y. Kim, C. Ortiz, S. Nam, S. Santiago, and V. Datta. Intersectional Bias in Hate Speech and Abusive Language Datasets. arXiv:2005.05921 [cs], May 2020. URL http://arxiv.org/abs/2005.05921. arXiv: 2005.05921.

Y. Kim and S. S. Sundar. Anthropomorphism of computers: Is it mindful or mindless? Computers in Human Behavior, 28(1):241-250, 2012.

J. Kocoń, A. Figas, M. Gruza, D. Puchalska, T. Kajdanowicz, and P. Kazienko. Offensive, aggressive, and hate speech analysis: From data-centric to human-centered approach. Information Processing \& Management, 58(5):102643, September 2021. ISSN 0306-4573. doi: 10.1016/j.ipm.2021.102643. URL https: //www.sciencedirect.com/science/article/pii/S0306457321001333.

A. Koenecke, A. Nam, E. Lake, J. Nudell, M. Quartey, Z. Mengesha, C. Toups, J. R. Rickford, D. Jurafsky, and S. Goel. Racial disparities in automated speech recognition. Proceedings of the National Academy of Sciences, 117(14):7684-7689, April 2020. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1915768117. URL https://www.pnas.org/content/117/14/7684.

I. Kohler-Hausmann. Eddie Murphy and the Dangers of Counterfactual Causal Thinking About Detecting Racial Discrimination. SSRN Scholarly Paper ID 3050650, Social Science Research Network, Rochester, NY, January 2019. URL https://papers.ssrn.com/abstract $=3050650$.

N. Kordzadeh and M. Ghasemaghaei. Algorithmic bias: review, synthesis, and future research directions. European Journal of Information Systems, 0(0):1-22, June 2021. ISSN 0960-085X. doi: 10.1080/0960085X .2021.1927212. URL https://doi.org/10.1080/0960085X.2021.1927212.

M. Kosinski, D. Stillwell, and T. Graepel. Private traits and attributes are predictable from digital records of human behavior. Proceedings of the National Academy of Sciences, 110(15):5802-5805, April 2013. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1218772110. URL https://www.pnas.org/content/110/15/5802.

C. Krittanawong, B. Narasimhan, H. U. H. Virk, H. Narasimhan, J. Hahn, Z. Wang, and W. W. Tang. Misinformation Dissemination in Twitter in the COVID-19 Era. The American Journal of Medicine, 133 (12):1367-1369, December 2020. ISSN 0002-9343. doi: 10.1016/j.amjmed.2020.07.012. URL https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7426698/.

K. Kurita, N. Vyas, A. Pareek, A. W. Black, and Y. Tsvetkov. Measuring Bias in Contextualized Word Representations. arXiv:1906.07337 [cs], June 2019. URL http://arxiv.org/abs/1906.07337. arXiv: 1906.07337.

K. H. Kwon, S.-I. Moon, and M. A. Stefanone. Unspeaking on Facebook? Testing network effects on selfcensorship of political expressions in social network sites. Quality \& Quantity, 49(4):1417-1435, July 2015. ISSN 1573-7845. doi: 10.1007/s11135-014-0078-8. URL https://doi.org/10.1007/s11135-014-0078-8.

C. Labbé and D. Labbé. Duplicate and fake publications in the scientific literature: how many SCIgen papers in computer science? Scientometrics, 94(1):379-396, January 2013. ISSN 1588-2861. doi: 10.1007/s11192-0 12-0781-y. URL https://doi.org/10.1007/s11192-012-0781-y.

K. Lacker. Giving GPT-3 a Turing Test, July 2020. URL https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turin g-test.html.

J. Lambert and E. Cone. How Robots Change the World - What automation really means for jobs, productivity and regions. Technical report, Oxford Economics, 2019. URL https://www.oxfordeconomics.com/recent-rel eases/how-robots-change-the-world.

I. Lapowsky. How Bots Broke the FCC's Public Comment System. Wired, November 2017. ISSN 1059-1028. URL https://www.wired.com/story/bots-broke-fcc-public-comment-system/.

A. Lazaridou, A. Kuncoro, E. Gribovskaya, D. Agrawal, A. Liska, T. Terzi, M. Gimenez, C. d. M. d'Autume, S. Ruder, D. Yogatama, K. Cao, T. Kocisky, S. Young, and P. Blunsom. Pitfalls of Static Language Modelling. arXiv:2102.01951 [cs], February 2021. URL http://arxiv.org/abs/2102.01951. arXiv: 2102.01951.

B. Lewis and A. E. Marwick. Media Manipulation and Disinformation Online. Technical report, Data \& Society, May 2017. URL https://datasociety.net/library/media-manipulation-and-disinfo-online.

M. Lewis and G. Lupyan. Gender stereotypes are reflected in the distributional structure of 25 languages. Nature Human Behaviour, 4(10):1021-1028, October 2020. ISSN 2397-3374. doi: 10.1038/s41562-020-0918-6. URL https://www.nature.com/articles/s41562-020-0918-6.

M. Lewis, D. Yarats, Y. N. Dauphin, D. Parikh, and D. Batra. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. arXiv:1706.05125 [cs], June 2017. URL http://arxiv.org/abs/1706.05125. arXiv: 1706.05125 .

P. Lewis, P. Stenetorp, and S. Riedel. Question and Answer Test-Train Overlap in Open-Domain Question Answering Datasets. arXiv:2008.02637 [cs], August 2020. URL http://arxiv.org/abs/2008.02637. arXiv: 2008.02637.

Z. Li, S. Zhuang, S. Guo, D. Zhuo, H. Zhang, D. Song, and I. Stoica. TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models. arXiv:2102.07988 [cs], September 2021. URL http://arxiv.org/ $\mathrm{abs} / 2102.07988$. arXiv: 2102.07988 .

Y. Liao and J. He. Racial mirroring effects on human-agent interaction in psychotherapeutic conversations. In Proceedings of the 25th International Conference on Intelligent User Interfaces, IUI '20, pages 430-442, Cagliari, Italy, March 2020. Association for Computing Machinery. ISBN 9781450371186. doi: 10.1145/3377325.33 77488. URL https://doi.org/10.1145/3377325.3377488.

S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958 [cs], September 2021. URL http://arxiv.org/abs/2109.07958. arXiv: 2109.07958.

P. Linardatos, V. Papastefanopoulos, and S. Kotsiantis. Explainable AI: A Review of Machine Learning Interpretability Methods. Entropy, 23(1):18, January 2021. doi: 10.3390/e23010018. URL https: //www.mdpi.com/1099-4300/23/1/18.

R. Lippi. English with an Accent: Language, Ideology and Discrimination in the United States. Routledge, 1997. URL https://www.routledge.com/English-with-an-Accent-Language-Ideology-and-Discrimination-in-the -United/Lippi-Green/p/book/9780415559119.

Z. C. Lipton. The Mythos of Model Interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue, 16(3):31-57, June 2018. ISSN 1542-7730, 1542-7749. doi: 10.1145/3236386.3241340. URL https://dl.acm.org/doi/10.1145/3236386.3241340.

Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692 [cs], July 2019. URL http://arxiv.org/abs/1907.11692. arXiv: 1907.11692.

LSEblog2017. Doxing is a toxic practice - no matter who is targeted | Media@LSE, August 2017. URL https://bl ogs.lse.ac.uk/medialse/2017/08/18/the-dangers-of-doxing-and-the-implications-for-media-regulation/.

A. S. Luccioni and J. D. Viviano. What's in the Box? A Preliminary Analysis of Undesirable Content in the Common Crawl Corpus. arXiv:2105.02732 [cs], May 2021. URL http://arxiv.org/abs/2105.02732. arXiv: 2105.02732

L. Lucy and D. Bamman. Gender and Representation Bias in GPT-3 Generated Stories. In Proceedings of the Third Workshop on Narrative Understanding, pages 48-55, Virtual, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.nuse-1.5. URL https://aclanthology.org/2021.nuse-1.5.

S. Luper. Epistemic Relativism. Philosophical Issues, 14:271-295, 2004. ISSN 1533-6077. URL https: //www.jstor.org/stable/3050631.

A. Maass and L. Arcuri. The role of language in the persistence of stereotypes. In Language, interaction and social cognition, pages 129-143. Sage Publications, Inc, 1992. URL https://psycnet.apa.org/record/1992-97 $980-006$.

A. Makazhanov, D. Rafiei, and M. Waqar. Predicting political preference of Twitter users. Social Network Analysis and Mining, 4(1):193, May 2014. ISSN 1869-5469. doi: 10.1007/s13278-014-0193-5. URL https://doi.org/10.1007/s13278-014-0193-5.

A. E. Manduley, A. Mertens, I. Plante, and A. Sultana. The role of social media in sex education: Dispatches from queer, trans, and racialized communities:. Feminism \& Psychology, 28(1):152-170, February 2018. doi: 10.1177/0959353517717751. URL https://journals.sagepub.com/eprint/wZsRhKyIrHE7KMyN9fJB/full.

C. B. Mann. Can Conversing with a Computer Increase Turnout? Mobilization Using Chatbot Communication. Journal of Experimental Political Science, 8(1):51-62, 2021. URL https://ideas.repec.org/a/cup/jexpos/v8y 2021i1p51-62_5.html.

H. Mao, X. Shuai, and A. Kapadia. Loose tweets: an analysis of privacy leaks on twitter. In Proceedings of the 10th annual ACM workshop on Privacy in the electronic society, WPES '11, pages 1-12, Chicago, Illinois, USA, October 2011. Association for Computing Machinery. ISBN 9781450310024. doi: 10.1145/2046556.2046558. URL https://doi.org/10.1145/2046556.2046558.

V. Marda and S. Narayan. On the importance of ethnographic methods in AI research. Nature Machine Intelligence, 3(3):187-189, March 2021. ISSN 2522-5839. doi: 10.1038/s42256-021-00323-0. URL https://www.nature.com/articles/s42256-021-00323-0.

M. Marino. The Racial Formation of Chatbots. CLCWeb: Comparative Literature and Culture, 16(5), December 2014. ISSN 1481-4374. doi: 10.7771/1481-4374.2560. URL https://docs.lib.purdue.edu/clcweb/vol16/iss $5 / 13$.

A. Markov. Essai d'une recherche statistique sur le texte du roman "eugène oněgin", illustrant la liaison des épreuves en chaîne. Bulletin de l'Académie Impériale des Sciences de St.-Pétersbourg., 7(3):153-162, 1913.

D. Martin Jr., V. Prabhakaran, J. Kuhlberg, A. Smart, and W. S. Isaac. Participatory Problem Formulation for Fairer Machine Learning Through Community Based System Dynamics. arXiv:2005.07572 [cs, stat], May 2020. URL http://arxiv.org/abs/2005.07572. arXiv: 2005.07572.

K. McGuffie and A. Newhouse. The Radicalization Risks of GPT-3 and Advanced Neural Language Models. arXiv:2009.06807 [cs], September 2020. URL http://arxiv.org/abs/2009.06807. arXiv: 2009.06807.

K. McKee, X. Bai, and S. Fiske. Understanding Human Impressions of Artificial Intelligence. PsyArxiv, 2021. URL https://psyarxiv.com/5ursp/.

N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan. A Survey on Bias and Fairness in Machine Learning. arXiv:1908.09635 [cs], September 2019. URL http://arxiv.org/abs/1908.09635. arXiv: 1908.09635 .

N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan. A Survey on Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6):115:1-115:35, July 2021. ISSN 0360-0300. doi: 10.1145/3457607. URL https://doi.org/10.1145/3457607.

J. Menasce Horowitz, R. Igielnik, and R. Kochhar. Trends in U.S. income and wealth inequality. Technical report, Pew Research Center, January 2020. URL https://www.pewresearch.org/social-trends/2020/01/09 /trends-in-income-and-wealth-inequality/.

S. Milano, M. Taddeo, and L. Floridi. Recommender systems and their ethical challenges. AI \& SOCIETY, 35(4):957-967, December 2020. ISSN 1435-5655. doi: 10.1007/s00146-020-00950-y. URL https: //doi.org/10.1007/s00146-020-00950-y.

T. Miller. Explanation in Artificial Intelligence: Insights from the Social Sciences. arXiv:1706.07269 [cs], August 2018. URL http://arxiv.org/abs/1706.07269. arXiv: 1706.07269.

T. Miller. Explanation in artificial intelligence: Insights from the social sciences. Artificial Intelligence, 267: 1-38, February 2019. ISSN 0004-3702. doi: 10.1016/j.artint.2018.07.007. URL https://www.sciencedirect. com/science/article/pii/S0004370218305988.

A. S. Miner, A. Milstein, S. Schueller, R. Hegde, C. Mangurian, and E. Linos. Smartphone-Based Conversational Agents and Responses to Questions About Mental Health, Interpersonal Violence, and Physical Health. JAMA internal medicine, 176(5):619-625, May 2016. ISSN 2168-6114. doi: 10.1001/jamainternmed.2016.0400. URL https://europepmc.org/articles/PMC4996669.

S. Mohamed, M.-T. Png, and W. Isaac. Decolonial AI: Decolonial Theory as Sociotechnical Foresight in Artificial Intelligence. Philosophy \& Technology, 33(4):659-684, December 2020. ISSN 2210-5441. doi: 10.1007/s13347-020-00405-8. URL https://doi.org/10.1007/s13347-020-00405-8.

W. Moncur, J. Masthoff, and E. Reiter. Facilitating benign deceit in mediated communication. In CHI '09 Extended Abstracts on Human Factors in Computing Systems, CHI EA '09, pages 3383-3388, Boston, MA, USA, April 2009. Association for Computing Machinery. ISBN 9781605582474. doi: 10.1145/1520340.1520490. URL https://doi.org/10.1145/1520340.1520490.

W. Moncur, J. Masthoff, E. Reiter, Y. Freer, and H. Nguyen. Providing Adaptive Health Updates Across the Personal Social Network. Human-Computer Interaction, 29(3):256-309, May 2014. ISSN 0737-0024. doi: 10.1080/07370024.2013.819218. URL https://doi.org/10.1080/07370024.2013.819218.

A. A. Morgan-Lopez, A. E. Kim, R. F. Chew, and P. Ruddle. Predicting age groups of Twitter users based on language and metadata features. PLOS ONE, 12(8):e0183537, August 2017. ISSN 1932-6203. doi: 10.1371/ journal.pone.0183537. URL https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0183537.

D. F. Mujtaba and N. R. Mahapatra. Ethical Considerations in AI-Based Recruitment. In 2019 IEEE International Symposium on Technology and Society (ISTAS), pages 1-7, November 2019. doi: 10.1109/ISTAS48451.2019. 8937920. ISSN: 2158-3412.

M. Murgia. DeepMind's Lila Ibrahim: 'It's hard not to go through imposter syndrome'. Financial Times, July 2021. URL https://www.ft.com/content/c43cd848-367a-4c67-8cf1-7f34d3eaf84e.

D. Mytton. Data centre water consumption. NPJ Clean Water, 4(1):1-6, February 2021. ISSN 2059-7037. doi: 10.1038/s41545-021-00101-w. URL https://www.nature.com/articles/s41545-021-00101-w.

J. Möller, D. Trilling, N. Helberger, and B. van Es. Do not blame it on the algorithm: an empirical assessment of multiple recommender systems and their impact on content diversity. Information, Communication \& Society, 21(7):959-977, July 2018. ISSN 1369-118X. doi: 10.1080/1369118X.2018.1444076. URL https://doi.org/10.1080/1369118X.2018.1444076.

M. Nadeem, A. Bethke, and S. Reddy. StereoSet: Measuring stereotypical bias in pretrained language models. arXiv:2004.09456 [cs], April 2020. URL http://arxiv.org/abs/2004.09456. arXiv: 2004.09456.

N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. arXiv:2010.00133 [cs], September 2020. URL http://arxiv.org/abs/20 10.00133. arXiv: 2010.00133.

A. Narayanan. How to Recognize AI Snake Oil, January 2021. URL https://www.cs.princeton.edu/news/howrecognize-ai-snake-oil.

D. Nguyen, R. Gravel, D. Trieschnigg, and T. Meder. "How Old Do You Think I Am?" A Study of Language and Age in Twitter. Proceedings of the International AAAI Conference on Web and Social Media, 7(1):439-448, 2013. ISSN 2334-0770. URL https://ojs.aaai.org/index.php/ICWSM/article/view/14381.

S. I. Nikolenko. Synthetic Data for Deep Learning, volume 174 of Springer Optimization and Its Applications. Springer International Publishing, Cham, 2021. ISBN 9783030751777 9783030751784. doi: 10.1007/9783-030-75178-4. URL https://link.springer.com/10.1007/978-3-030-75178-4.

NLP for Positive Impact 2021. Workshop on NLP for Positive Impact at ACL-IJCNLP, 2021. URL https: //sites.google.com/view/nlp4positiveimpact2021.

S. U. Noble. Algorithms of Oppression. NYU Press, 2018. URL https://nyupress.org/9781479837243/algorithm s-of-oppression.

D. Nozza, F. Bianchi, and D. Hovy. HONEST: Measuring Hurtful Sentence Completion in Language Models. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2398-2406, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.191. URL https://aclanthology.org/2021.naacl-main.191.

Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan. Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464):447-453, October 2019. doi: 10.1126/science.aax2342. URL https://www.science.org/doi/abs/10.1126/science.aax2342.

B. of Labor Statistics. Interpreters and Translators: Occupational Outlook Handbook. Technical report, U.S. Department of Labour, 2021. URL https://www.bls.gov/ooh/media-and-communication/interpreters-and-t ranslators.htm\#:〜:text=3\%25-,Employment\%20of\%20interpreters\%20and\%20translators\%20is\%20proj ected\%20to\%20grow\%2020,the\%20average\%20for\%20all\%20occupations.\&text=The\%20ongoing\%20 need\%20for\%20military,in\%20more\%20jobs\%20as\%20well. publisher: U.S. Bureau of Labour Statistics.

K. Ognyanova, D. Lazer, R. E. Robertson, and C. Wilson. Misinformation in action: Fake news exposure is linked to lower trust in media, higher trust in government when your side is in power. Harvard Kennedy School Misinformation Review, June 2020. doi: 10.37016/mr-2020-024. URL https://misinforeview.hks.harvard.ed u/article/misinformation-in-action-fake-news-exposure-is-linked-to-lower-trust-in-media-higher-trust-ingovernment-when-your-side-is-in-power/.

P. Oosterhoff. Online censors are a barrier to sex education, 2016. URL https://www.scidev.net/global/opinio ns/online-censors-sex-education-porn/.

D. O'Callaghan, D. Greene, M. Conway, J. Carthy, and P. Cunningham. Down the (White) Rabbit Hole: The Extreme Right and Online Recommender Systems. Social Science Computer Review, 33(4):459-478, August 2015. ISSN 0894-4393. doi: 10.1177/0894439314555329. URL https://doi.org/10.1177/089443931455 5329 .

G. PAIR. People + AI Guidebook. Google, May 2019. URL https://design.google/ai-guidebook.

A. Pardes. The Emotional Chatbots Are Here to Probe Our Feelings. Wired, January 2018. ISSN 1059-1028. URL https://www.wired.com/story/replika-open-source/.

G. Park, H. A. Schwartz, J. C. Eichstaedt, M. L. Kern, M. Kosinski, D. J. Stillwell, L. H. Ungar, and M. E. P. Seligman. Automatic personality assessment through social media language. Journal of Personality and Social Psychology, 108(6):934-952, June 2015. ISSN 1939-1315, 0022-3514. doi: 10.1037/pspp0000020. URL http://doi.apa.org/getdoi.cfm?doi=10.1037/pspp0000020.

F. Pasquale. The Black Box Society. Harvard University Press, 2016. URL https://dl.acm.org/doi/abs/10.5555 /2717112.

D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So, M. Texier, and J. Dean. Carbon Emissions and Large Neural Network Training. arXiv:2104.10350 [cs], April 2021. URL http: //arxiv.org/abs/2104.10350. arXiv: 2104.10350.

D. Perez-Marin and I. Pascual-Nieto. Conversational Agents and Natural Language Interaction: Techniques and Effective Practices. Information Science Reference - Imprint of: IGI Publishing, Hershey, PA, 2011. ISBN 9781609606176 .

N. Persily and J. A. Tucker. Social Media and Democracy: The State of the Field, Prospects for Reform. Cambridge University Press, September 2020. ISBN 9781108858779. Google-Books-ID: TgH3DwAAQBAJ.

PerspectiveAPI. Perspective API | Developers, 2021. URL https://support.perspectiveapi.com/s/about-the-api -attributes-and-languages.

J. Pfeiffer, I. Vulić, I. Gurevych, and S. Ruder. UNKs Everywhere: Adapting Multilingual Language Models to New Scripts. arXiv:2012.15562 [cs], September 2021. URL http://arxiv.org/abs/2012.15562. arXiv: 2012.15562 .

A. Pilipiszyn. GPT-3 Powers the Next Generation of Apps, March 2021. URL https://openai.com/blog/gpt-3apps/. publisher: OpenAI.

D. Preoţiuc-Pietro, Y. Liu, D. Hopkins, and L. Ungar. Beyond Binary Labels: Political Ideology Prediction of Twitter Users. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 729-740, Vancouver, Canada, 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1068. URL http://aclweb.org/anthology/P17-1068.

K. Quach. Researchers made an OpenAI GPT-3 medical chatbot as an experiment. It told a mock patient to kill themselves. The Register, October 2020. URL https://www.theregister.com/2020/10/28/gpt3_medical_cha tbot_experiment/.

D. Quercia, M. Kosinski, D. Stillwell, and J. Crowcroft. Our Twitter Profiles, Our Selves: Predicting Personality with Twitter. In 2011 IEEE Third International Conference on Privacy, Security, Risk and Trust and 2011 IEEE Third International Conference on Social Computing, pages 180-185, October 2011. doi: 10.1109/PASSAT/S ocialCom.2011.26.

A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever. Improving Language Understanding by Generative Pre-Training. 2018a.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language Models are Unsupervised Multitask Learners. 2018b.

J. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P.S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d'Autume, Y. Li, T. Terzi, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, J. Bradbury, M. Johnson, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving. Scaling language models: Methods, analysis \& insights from training Gopher. arXiv submission, 2021.

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv:1910.10683 [cs, stat], July 2020. URL http://arxiv.org/abs/1910.10683. arXiv: 1910.10683.

I. D. Raji. Handle with Care: Lessons for Data Science from Black Female Scholars. Patterns, 1(8):100150, November 2020. ISSN 2666-3899. doi: 10.1016/j.patter.2020.100150. URL https://www.sciencedirect.co $\mathrm{m} /$ science/article/pii/S2666389920301987.

I. D. Raji, A. Smart, R. N. White, M. Mitchell, T. Gebru, B. Hutchinson, J. Smith-Loud, D. Theron, and P. Barnes. Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing. arXiv:2001.00973 [cs], January 2020. URL http://arxiv.org/abs/2001.00973. arXiv: 2001.00973.

S. Ramaswamy, O. Thakkar, R. Mathews, G. Andrew, H. B. McMahan, and F. Beaufays. Training Production Language Models without Memorizing User Data. arXiv:2009.10031 [cs, stat], September 2020. URL http://arxiv.org/abs/2009.10031. arXiv: 2009.10031.

P. Ranade, A. Piplai, S. Mittal, A. Joshi, and T. Finin. Generating Fake Cyber Threat Intelligence Using Transformer-Based Models. arXiv:2102.04351 [cs], June 2021. URL http://arxiv.org/abs/2102.04351. arXiv: 2102.04351 .

E. Rand. Reclaiming Queer: Activist \& Academic Rhetorics of Resistance. University of Alabama Press, 2014.

E. Reiter. Could NLG systems injure or even kill people?, October 2020. URL https://ehudreiter.com/2020/10 /20/could-nlg-systems-injure-or-even-kill-people/.

M. Rimmer. Patent-Busting: The Public Patent Foundation, Gene Patents and the Seed Wars. In C. Lawson and J. Sanderson, editors, The Intellectual Property and Food Project. Routledge, 2013.

A. Romano. A group of YouTubers is claiming the site systematically demonetizes queer content. Vox, October 2019. URL https://www.vox.com/culture/2019/10/10/20893258/youtube-lgbtq-censorship-demonetizatio n-nerd-city-algorithm-report.

J. Rosa and N. Flores. Unsettling race and language: Toward a raciolinguistic perspective. Language in Society, 46(5):621-647, November 2017. ISSN 0047-4045, 1469-8013. doi: 10.1017/S0047404517000562. URL https://www.cambridge.org/core/journals/language-in-society/article/abs/unsettling-race-and-language -toward-a-raciolinguistic-perspective/30FFC5253F465905D75CDFF1C1363AE3.

C. Rosset. Turing-NLG: A 17-billion-parameter language model by Microsoft, February 2020. URL https://ww w.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/.

A. Rubel, A. Pham, and C. Castro. Agency Laundering and Algorithmic Decision Systems. In N. G. Taylor, C. Christian-Lamb, M. H. Martin, and B. Nardi, editors, Information in Contemporary Society, Lecture Notes in Computer Science, pages 590-598, Cham, 2019. Springer International Publishing. ISBN 9783030157425. doi: 10.1007/978-3-030-15742-5_56.

S. Ruder. Why You Should Do NLP Beyond English, August 2020. URL https://ruder.io/nlp-beyond-english/.

A. Sabeti. Teaching GPT-3 to Identify Nonsense, July 2020. URL https://arr.am/2020/07/25/gpt-3-uncertaint y-prompts/.

N. A. Sales. Secrecy and National Security Investigations. Alabama Law Review, 58:811, 2006. URL https: //heinonline.org/HOL/Page?handle=hein.journals/bamalr58\&id $=821 \& d i v=\& c o l l e c t i o n=$.

N. Sambasivan and J. Holbrook. Toward responsible AI for the next billion users. Interactions, 26(1):68-71, December 2018. ISSN 1072-5520. doi: 10.1145/3298735. URL https://doi.org/10.1145/3298735.

N. Sambasivan, E. Arnesen, B. Hutchinson, T. Doshi, and V. Prabhakaran. Re-imagining Algorithmic Fairness in India and Beyond. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, pages 315-328, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445896. URL https://doi.org/10.1145/3442188.3445896.

M. Sap, D. Card, S. Gabriel, Y. Choi, and N. A. Smith. The Risk of Racial Bias in Hate Speech Detection. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1668-1678, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1163. URL https://aclanthology.org/P19-1163.

K. M. Saunders. The Law and Ethics of Trade Secrets: A Case Study. California Western Law Review, 42:209, 2005. URL https://heinonline.org/HOL/Page?handle=hein.journals/cwlr42\&id=215\&div=\&collection=.

A. Schmidt and M. Wiegand. A Survey on Hate Speech Detection using Natural Language Processing. In Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media, pages 1-10, Valencia, Spain, 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-1101. URL http://aclweb.org/anthology/W17-1101.

A. T. Schmidt and B. Engelen. The ethics of nudging: An overview. Philosophy Compass, 15(4):e12658, April 2020. ISSN 1747-9991, 1747-9991. doi: 10.1111/phc3.12658. URL https://onlinelibrary.wiley.com/doi/10 $.1111 /$ phc3.12658.

B. Schneier. Bots Are Destroying Political Discourse As We Know It. The Atlantic, January 2020. URL https:// www.theatlantic.com/technology/archive/2020/01/future-politics-bots-drowning-out-humans/604489/.

M. Schroepfer. How AI is getting better at detecting hate speech, November 2020. URL https://ai.facebook.co $\mathrm{m} / \mathrm{blog} /$ how-ai-is-getting-better-at-detecting-hate-speech/.

R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni. Green AI. Communications of the ACM, 63(12):54-63, November 2020. ISSN 0001-0782. doi: 10.1145/3381831. URL https://doi.org/10.1145/3381831.

E. Seger, S. Avin, G. Pearson, M. Briers, S. Ó Heigeartaigh, and H. Bacon. Tackling threats to informed decision-making in democratic societies. Technical report, Alan Turing Institute, 2020. URL https://www.tu ring.ac.uk/research/publications/tackling-threats-informed-decision-making-democratic-societies.

D. Shah, H. A. Schwartz, and D. Hovy. Predictive Biases in Natural Language Processing Models: A Conceptual Framework and Overview. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5248-5264, 2020. doi: 10.18653/v1/2020.acl-main.468. URL http://arxiv.org/abs/1912 .11078 . arXiv: 1912.11078 .

A. Shahbaz and A. Funk. Social Media Surveillance. Technical report, Freedom House, 2019. URL https://free domhouse.org/report/freedom-on-the-net/2019/the-crisis-of-social-media/social-media-surveillance.

C. E. Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379-423, July 1948. ISSN 0005-8580. doi: 10.1002/j.1538-7305.1948.tb01338.x.

E. Sheng, K.-W. Chang, P. Natarajan, and N. Peng. Societal Biases in Language Generation: Progress and Challenges. arXiv:2105.04054 [cs], June 2021. URL http://arxiv.org/abs/2105.04054. arXiv: 2105.04054.

M. Sloane, E. Moss, O. Awomolo, and L. Forlano. Participation is not a Design Fix for Machine Learning. arXiv:2007.02423 [cs], August 2020. URL http://arxiv.org/abs/2007.02423. arXiv: 2007.02423.

I. Solaiman and C. Dennison. Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets. arXiv:2106.10328 [cs], June 2021. URL http://arxiv.org/abs/2106.10328. arXiv: 2106.10328.

I. Solaiman, M. Brundage, J. Clark, A. Askell, A. Herbert-Voss, J. Wu, A. Radford, G. Krueger, J. W. Kim, S. Kreps, M. McCain, A. Newhouse, J. Blazakis, K. McGuffie, and J. Wang. Release Strategies and the Social Impacts of Language Models. arXiv:1908.09203 [cs], November 2019. URL http://arxiv.org/abs/1908.09203. arXiv: 1908.09203 .

K. Sparck Jones. Language modelling's generative model: is it rational? Computer Laboratory, University of Cambridge, Cambridge, UK, 2004.

D. Sravani, L. Kameswari, and R. Mamidi. Political Discourse Analysis: A Case Study of Code Mixing and Code Switching in Political Speeches. In Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching, pages 1-5, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.calcs-1.1. URL https://aclanthology.org/2021.calcs-1.1.

T. Stahl. Indiscriminate mass surveillance and the public sphere. Ethics and Information Technology, 18(1): 33-39, March 2016. ISSN 1572-8439. doi: 10.1007/s10676-016-9392-2. URL https://doi.org/10.1007/s1 0676-016-9392-2.

StanfordNaturalProcessingGroup. The Stanford Natural Language Processing Group, 2021. URL https: $/ /$ nlp.stanford.edu/projects/coref.shtml.

J. Stilgoe. Who's driving innovation. New Technologies and the Collaborative State. Cham, Switzerland: Palgrave Macmillan, 2020.

J. Stilgoe, R. Owen, and P. Macnaghten. Developing a framework for responsible innovation. Research Policy, 42(9):1568-1580, November 2013. ISSN 0048-7333. doi: 10.1016/j.respol.2013.05.008. URL https://www.sciencedirect.com/science/article/pii/S0048733313000930.

E. Strubell, A. Ganesh, and A. McCallum. Energy and Policy Considerations for Deep Learning in NLP. arXiv:1906.02243 [cs], June 2019. URL http://arxiv.org/abs/1906.02243. arXiv: 1906.02243.

S. Sullivan and N. Tuana, editors. Race and epistemologies of ignorance. SUNY series, philosophy and race. State University of New York Press, Albany, 2007. ISBN 9780791471012 9780791471029. OCLC: ocm70676503.

summerstay on Reddit. Fiction by Neil Gaiman and Terry Pratchett by GPT-3, July 2020. URL www.reddit.com /r/slatestarcodex/comments/hmu5lm/fiction_by_neil_gaiman_and_terry_pratchett_by_gpt3/.

Y. Sun, S. Wang, S. Feng, S. Ding, C. Pang, J. Shang, J. Liu, X. Chen, Y. Zhao, Y. Lu, W. Liu, Z. Wu, W. Gong, J. Liang, Z. Shang, P. Sun, W. Liu, X. Ouyang, D. Yu, H. Tian, H. Wu, and H. Wang. ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation. arXiv:2107.02137 [cs], July 2021. URL http://arxiv.org/abs/2107.02137. arXiv: 2107.02137.

A. Tamkin, M. Brundage, J. Clark, and D. Ganguli. Understanding the Capabilities, Limitations, and Societal Impact of Large Language Models. arXiv:2102.02503 [cs], February 2021. URL http://arxiv.org/abs/2102.0 2503. arXiv: 2102.02503 .

R. S. J. Tol. The impact of climate change and the social cost of carbon. In Routledge Handbook of Energy Economics. Routledge, 2019.

N. Tomasev, K. R. McKee, J. Kay, and S. Mohamed. Fairness for Unobserved Characteristics: Insights from Technological Impacts on Queer Communities. Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 254-265, July 2021. doi: 10.1145/3461702.3462540. URL http://arxiv.org/abs/2102.04257. arXiv: 2102.04257 .

J. M. Twenge. More Time on Technology, Less Happiness? Associations Between Digital-Media Use and Psychological Well-Being. Current Directions in Psychological Science, 28(4):372-379, August 2019. ISSN 0963-7214. doi: 10.1177/0963721419838244. URL https://doi.org/10.1177/0963721419838244.

E. Van den Broeck, B. Zarouali, and K. Poels. Chatbot advertising effectiveness: When does the message get through? Computers in Human Behavior, 98:150-157, September 2019. ISSN 0747-5632. doi: 10.1016/j.chb.2019.04.009. URL https://www.sciencedirect.com/science/article/pii/S0747563219301499.

R. Van Noorden. Publishers withdraw more than 120 gibberish papers. Nature, February 2014. ISSN 1476-4687. doi: 10.1038/nature.2014.14763. URL https://www.nature.com/articles/nature.2014.14763.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

VersebyVerse. Verse by Verse, 2020. URL https://sites.research.google/versebyverse/. publisher:.

J. Vincent. The invention of AI 'gaydar' could be the start of something much worse. The Verge, September 2017. URL https://www.theverge.com/2017/9/21/16332760/ai-sexuality-gaydar-photo-physiognomy.

K. Vredenburgh. The Right to Explanation. Journal of Political Philosophy, 0(0):1-21, 2021. ISSN 1467-9760. doi: 10.1111/jopp.12262. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/jopp.12262.

C. Véliz. Privacy matters because it empowers us all | Aeon Essays. Aeon, September 2019. URL https: //aeon.co/essays/privacy-matters-because-it-empowers-us-all.

B. Vézina and S. Hinchcliff Pearson. Should CC-Licensed Content be Used to Train AI? It Depends., March 2021. URL https://creativecommons.org/2021/03/04/should-cc-licensed-content-be-used-to-train-ai-it-d epends/.

D. Wallace, F. Tramer, M. Jagielski, and A. Herbert-Voss. Does GPT-2 Know Your Phone Number?, December 2020. URL http://bair.berkeley.edu/blog/2020/12/20/lmmem/.

A. Wang and O. Russakovsky. Directional Bias Amplification. arXiv:2102.12594 [cs], June 2021. URL http://arxiv.org/abs/2102.12594. arXiv: 2102.12594.

A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. arXiv preprint arXiv:1905.00537, 2019a.

Y. Wang and M. Kosinski. Deep neural networks are more accurate than humans at detecting sexual orientation from facial images. Journal of Personality and Social Psychology, 114(2):246-257, February 2018. ISSN 1939-1315, 0022-3514. doi: 10.1037/pspa0000098. URL http://doi.apa.org/getdoi.cfm?doi=10.1037/psp $\mathrm{a} 0000098$.

Y. Wang, M. McKee, A. Torbica, and D. Stuckler. Systematic Literature Review on the Spread of Health-related Misinformation on Social Media. Social Science \& Medicine, 240:112552, November 2019b. ISSN 0277-9536. doi: 10.1016/j.socscimed.2019.112552. URL https://www.sciencedirect.com/science/article/pii/S0277953 619305465 .

Z. Wang, A. W. Yu, O. Firat, and Y. Cao. Towards Zero-Label Language Learning. arXiv:2109.09193 [cs], September 2021. URL http://arxiv.org/abs/2109.09193. arXiv: 2109.09193.

M. Webb. The Impact of Artificial Intelligence on the Labor Market. SSRN Scholarly Paper ID 3482150, Social Science Research Network, Rochester, NY, November 2019. URL https://papers.ssrn.com/abstract=3482150.

J. Welbl, A. Glaese, J. Uesato, S. Dathathri, J. Mellor, L. A. Hendricks, K. Anderson, P. Kohli, B. Coppin, and P.-S. Huang. Challenges in Detoxifying Language Models. arXiv:2109.07445 [cs], September 2021. URL http://arxiv.org/abs/2109.07445. arXiv: 2109.07445.

T.-H. Wen, D. Vandyke, N. Mrksic, M. Gasic, L. M. Rojas-Barahona, P.-H. Su, S. Ultes, and S. Young. A Networkbased End-to-End Trainable Task-oriented Dialogue System. arXiv:1604.04562 [cs, stat], April 2017. URL http://arxiv.org/abs/1604.04562. arXiv: 1604.04562.

M. West, R. Kraut, and H. Ei Chew. I'd blush if I could : closing gender divides in digital skills through education. Technical report, UNESCO, 2019. URL https://repositorio.minedu.gob.pe/handle/20.500.12799/6598.

G. I. Winata, A. Madotto, Z. Lin, R. Liu, J. Yosinski, and P. Fung. Language Models are Few-shot Multilingual Learners. arXiv:2109.07684 [cs], September 2021. URL http://arxiv.org/abs/2109.07684. arXiv: 2109.07684 .

L. Winner. Do Artifacts Have Politics? Daedalus, 109(1):121-136, 1980. ISSN 0011-5266. URL https: //www.jstor.org/stable/20024652.

A. Xu, E. Pathak, E. Wallace, S. Gururangan, M. Sap, and D. Klein. Detoxifying Language Models Risks Marginalizing Minority Voices. arXiv:2104.06390 [cs], April 2021. URL http://arxiv.org/abs/2104.06390. arXiv: 2104.06390 .

E. Yang and M. E. Roberts. Censorship of Online Encyclopedias: Implications for NLP Models. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, pages 537-548, Virtual Event, Canada, March 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445916. URL https://doi.org/10.1145/3442188.3445916.

Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://papers.nips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9e e67cc69-Abstract.html.

K. Yee, U. Tantipongpipat, and S. Mishra. Image Cropping on Twitter: Fairness Metrics, their Limitations, and the Importance of Representation, Design, and Agency. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW2):1-24, October 2021. ISSN 2573-0142. doi: 10.1145/3479594. URL http: //arxiv.org/abs/2105.08667. arXiv: 2105.08667.

M. Yesilada and S. Lewandowsky. A systematic review: The YouTube recommender system and pathways to problematic content. PsyArxiv, June 2021. URL https://psyarxiv.com/6pv5c/.

W. Youyou, M. Kosinski, and D. Stillwell. Computer-based personality judgments are more accurate than those made by humans. Proceedings of the National Academy of Sciences, 112(4):1036-1040, January 2015. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1418680112. URL https://www.pnas.org/content/112/4/1036.

S. Zdenek. "Just Roll Your Mouse Over Me": Designing Virtual Women for Customer Service on the Web. Technical Communication Quarterly, 16(4):397-430, August 2007. ISSN 1057-2252, 1542-7625. doi: 10.1080/10572250701380766. URL http://www.tandfonline.com/doi/abs/10.1080/10572250701380766.

R. Zellers, A. Holtzman, H. Rashkin, Y. Bisk, A. Farhadi, F. Roesner, and Y. Choi. Defending Against Neural Fake News. arXiv:1905.12616 [cs], December 2020. URL http://arxiv.org/abs/1905.12616. arXiv: 1905.12616.

C. Zhang, P. Benz, C. Lin, A. Karjauv, J. Wu, and I. S. Kweon. A Survey On Universal Adversarial Attack. arXiv:2103.01498 [cs], March 2021a. URL http://arxiv.org/abs/2103.01498. arXiv: 2103.01498.

N. Zhang, L. Li, X. Chen, S. Deng, Z. Bi, C. Tan, F. Huang, and H. Chen. Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners. arXiv:2108.13161 [cs], October 2021b. URL http://arxiv.org/abs/2108.13161. arXiv: 2108.13161.

J. Zhao, T. Wang, M. Yatskar, V. Ordonez, and K.-W. Chang. Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints. arXiv:1707.09457 [cs, stat], July 2017. URL http: //arxiv.org/abs/1707.09457. arXiv: 1707.09457.

J. Zhao, T. Wang, M. Yatskar, R. Cotterell, V. Ordonez, and K.-W. Chang. Gender Bias in Contextualized Word Embeddings. arXiv:1904.03310 [cs], April 2019. URL http://arxiv.org/abs/1904.03310. arXiv: 1904.03310.

T. Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate Before Use: Improving Few-Shot Performance of Language Models. arXiv:2102.09690 [cs], June 2021. URL http://arxiv.org/abs/2102.09690. arXiv: 2102.09690 .

J. Zou and L. Schiebinger. AI can be sexist and racist — it's time to make it fair. Nature, 559(7714):324-326, July 2018. doi: 10.1038/d41586-018-05707-8. URL https://www.nature.com/articles/d41586-018-05707-8.

S. Zuboff. The age of surveillance capitalism: the fight for a human future at the new frontier of power. Profile books, London, 2019. ISBN 97817812568489781781256855.

J. Złotowski, D. Proudfoot, K. Yogeeswaran, and C. Bartneck. Anthropomorphism: Opportunities and Challenges in Human-Robot Interaction. International Journal of Social Robotics, 7(3):347-360, June 2015. ISSN 1875-4805. doi: 10.1007/s12369-014-0267-6. URL https://doi.org/10.1007/s12369-014-0267-6.

## A. Appendix

## A.1. Definitions

## A.1.1. Language Models

Language Models (LMs) are machine learning models that are trained to represent a probability distribution $p(w)$ over sequences of utterances $w$ from a pre-specified domain (letters, words, sentences, paragraphs, documents). LMs aim to capture statistical properties of the sequences of utterances present in their training corpus and can be used to make probabilistic predictions regarding sequences of utterances (Bengio, 2008). Typical training corpora for LMs contain natural language (e.g. collected from the web), but LMs can also be trained on other types of languages (e.g. computer programming languages). Moreover, LMs can serve different purposes, such as generating language (generative language models) or providing semantic embeddings. Depending on the primary purpose of a LM, slightly different architectures and training objectives can be used. In this paper, unless we specify otherwise, we focus on LMs tailored to language generation.

A standard approach to construct generative LMs is to use an autoregressive decomposition that sequentially proposes a probability distribution for the next utterance based on past utterances:

$$
p(w)=p\left(w_{1}\right) \cdot p\left(w_{2} \mid w_{1}\right) \cdots p\left(w_{T} \mid w_{1}, \ldots, w_{T-1}\right)
$$

Here $w=w_{1} \ldots w_{T}$ is a sequence of $T=|w|$ utterances. Each of the terms $p\left(w_{t} \mid w_{1}, \ldots, w_{t-1}\right)$ with $t=1, \ldots, T$ represents the probability the model assigns to observing the particular utterance $w_{t}$ given the previous $t-1$ utterances. LMs of this form are trained by updating the parameters controlling these conditional probabilities to assign high likelihood to sequences of utterances observed in the training corpus. Training is the result of an iterative process whereby at each iteration the model is presented with a batch of utterances and its parameters are updated to increase the likelihood of that particular set of utterances. Training large-scale language models can require very high numbers of iterations, requiring significant computing power.

Recent LMs are primarily distinguished from other LMs due to their parameter size and training data. Their size allows LMs to retain representations of extremely large text corpora, resulting in much more general sequence prediction systems than prior LMs. In this report, we focus on such large-scale models, connoted as LMs. The emergence of LMs is described in detail in the section on a brief history of "Large" Language Models.

Note that LMs do not output text directly. Rather, they produce a probability distribution over different utterances from which samples can be drawn. Greedy decoding directly from the (conditional) probability distribution provided by an LM is possible, but often performs poorly in practice. Instead, methods that focus on the most likely utterances - while introducing a small amount of variability (e.g. beam search and nucleus sampling (Holtzman et al., 2020)) - have been found to produce better results in practice (Brown et al., 2020)). LMs typically aim to mirror language found in the training data. However, they can also be optimised toward other tasks or objectives. For example, a LM can be optimised for dialogue, by predicting utterances that are most appropriate to maintain a conversation.

## A.1.2. Language Agents

Language agents (LAs) are machine learning systems that are restricted to providing only natural language text-output (Kenton et al., 2021). LAs may generate text-output based on LM predictions. LAs that are optimised to engage a person in direct dialogue are also referred to as "Conversational Agents" (CAs) (Perez-Marin and Pascual-Nieto, 2011).

## A.1.3. Language Technologies

LMs can be used in language technologies (LTs) such as voice assistants including Siri (Apple), Google Assistant (Google), or Alexa (Amazon), text generation tools such as AutoCorrect or SmartReply, and translation and summarisation tools. Language technologies can serve different purposes, for example providing information, entertainment, or productivity aids to the user.

Powerful large language models (LLMs) may lead to improved versions of existing language technologies. However, they may also make new types of language technology possible. For example, they may create conversational interfaces with human users where the use of this technology is indistinguishable from interaction with a human counterpart. Such applications are discussed in more detail in section V. Human-Computer Interaction Harms.

Distinguishing "statistical bias" from "social bias" Concerns regarding "bias" in language models generally revolve around distributional skews that result in unfavourable impacts for particular social groups (Sheng et al., 2021). We note that there are different definitions of "bias" and "discrimination" in classical statistics compared to sociotechnical studies. In classical statistics, "bias" designates the difference between a model's prediction and the ground truth (Dietterich and Kong, 1995); in machine learning, minimising statistical bias is a component of reducing error (Dietterich and Kong, 1995). In sociotechnical studies, "bias" refers to skews that lead to unjust discrimination based on traits such as age, gender, religion, ability status, whether or not these characteristics are legally protected (Blodgett et al., 2020). Developing mechanisms to quantify the latter type of bias is an area of active research, where qualitative and quantitative measures have been established (Barocas et al., 2019; Hardt et al., 2016).

Distinguishing statistical from sociotechnical notions of "discrimination" Similarly, the definition of "discrimination" is multiplicitous. Traditionally in machine learning, this term refers to making distinctions between possible categories or target classes (Bowker and Star, 1999). In sociotechnical work, "discrimination" refers to unjust differential treatment, typically toward historically marginalised groups. Various steps in training a machine learning model can result in discrimination in the sociotechnical sense, from labelling and collection of the training data, to defining the "target variable" and class labels, to selecting features (Barocas and Selbst, 2016).

## A.2. References Table

Table 2. References providing evidence for each risk covered in this report.

| Risk | Evidence in NLP, Evidence in LMs (GPT-2, GPT-3, T5, Gopher) |
| :---: | :---: |
| Discrimination, Exclusion and Toxicity |  |
| 2.1.2 Social stereotypes and <br> unfair discrimination | Blodgett et al. (2020); Caliskan et al. (2017); Dodge et al. (2021); Ferrer <br> et al. (2020); Zhao et al. (2017) <br> Abid et al. (2021); Huang et al. (2020); Lucy and Bamman (2021); <br> Nadeem et al. (2020); Nangia et al. (2020); Nozza et al. (2021) |
| 2.1.3 Exclusionary norms | Cao and Daumé III (2020) |
| 2.1.4 Toxic language | Duggan (2017); Gehman et al. (2020); Gorwa et al. (2020); Luccioni and <br> Viviano (2021) <br> Rae et al. (2021); Wallace et al. (2020) |
| 2.1.5 Lower performance by <br> social group | Blodgett and O'Connor (2017); Blodgett et al. (2016); Joshi et al. (2021); <br> Koenecke et al. (2020); Ruder (2020) <br> Winata et al. (2021) |
| Information Hazards |  |
| 2.2.2 Compromise privacy by <br> leaking private information | Dobberstein (2021); Kim (2021) <br> Carlini et al. (2021) |
| 2.2.3 Compromise privacy by <br> correctly inferring private <br> information | Garcia et al. (2018); Golbeck (2018); Makazhanov et al. (2014); <br> Morgan-Lopez et al. (2017); Nguyen et al. (2013); Park et al. (2015); <br> Preoţiuc-Pietro et al. (2017) |
| 2.2.4 Risks from leaking or <br> correctly inferring sensitive <br> information | Wallace et al. (2020) |

## Misinformation Harms

| 2.3.2 Disseminating false or <br> misleading information | Allcott et al. (2019); Krittanawong et al. (2020); Wang et al. (2019b) <br> Branwen (2020); Dale (2021); Lacker (2020); Lin et al. (2021); Rae et al. <br> (2021); Zhang et al. (2021b) |
| :--- | :--- |
| 2.3.3 Causing material harm by <br> disseminating misinformation <br> e.g. in medicine or law | Quach (2020) |
| 2.3.4 Leading users to perform <br> unethical or illegal actions | Hendrycks et al. (2021) |
|  |  |
| Malicious Uses |  |
| 2.4.2 Making disinformation <br> cheaper and more effective | Hampton (2021); Mann (2021); Schneier (2020); Zellers et al. (2020) |
| 2.4.3 Facilitating fraud and <br> impersonation scams | Lewis et al. (2017); Van Noorden (2014) |
| 2.4.4 Assisting code generation <br> for cyber attacks, weapons, or <br> malicious use | Chen et al. (2021a) |
| 2.4.5 Illegitimate surveillance <br> and censorship | Shahbaz and Funk (2019) |

## Human-Computer Interaction Harms

```
2.5.2 Anthropomorphising Kim and Sundar (2012)
systems can lead to
overreliance or unsafe use
2.5.3 Create avenues for Ischen et al. (2019); Lewis et al. (2017); Van den Broeck et al. (2019)
exploiting user trust to obtain
private information
2.5.4 Promoting harmful Cercas Curry et al. (2020); Hwang et al. (2019); Marino (2014); Zdenek
stereotypes by implying gender (2007)
or ethnic identity
```

Automation, access, and environmental harms

2.6.2 Environmental harms Strubell et al. (2019) Bender et al. (2021); Patterson et al. (2021)

from operating LMs

2.6.3 Increasing inequality and negative effects on job quality

2.6.4 Undermining creative Hsieh (2019) economies

2.6.5 Disparate access to Bender et al. (2021)

benefits due to hardware, software, skill constraints


[^0]:    ${ }^{1}$ These recent LMs are also referred to as "large language models", or "large-scale language models".

    ${ }^{2}$ Note that the origin of a risk is not a perfect guide to potential mitigations - a point we discuss in more detail in Understanding the point of origin of a risk.

    ${ }^{3}$ Each of these examples assumes a dialogue format where a human supplies a prompt and the LM offers a response. There are many LM use cases beyond such conversational agents. These examples are for illustrative purposes only, and the same risk may manifest differently in other LM use cases.

[^1]:    ${ }^{1}$ Note that the terms "bias" and "discrimination" have different meanings in classical statistics compared to sociotechnical studies; for a definition of these terms, see the Definitions in the Appendix.

[^2]:    ${ }^{2}$ See also the authors' (Abid, 2020) of "how hard it is to generate text about Muslims from GPT-3 that has nothing to do with violence", and (Gershgorn, 2021).

    ${ }^{3}$ Recent work critiques some current methods for measuring bias in LMs highlighting the importance of further exploration on valid measures (Blodgett et al., 2021).

[^3]:    ${ }^{4}$ Another proposed approach relies on synthetic data, although the efficacy of this approach remains uncertain and it raises distinct challenges, on amplifying other biases (Chen et al., 2021b; Ghalebikesabi et al., 2021; Nikolenko, 2021).

    ${ }^{5}$ In this real-world example, a model ranking applicant suitability based on written CVs was biased against the term 'women' (as in 'women's chess club'). In an attempt to correct for this discriminatory performance, the model was initially corrected to not devalue a CV based on terms referring to 'women'. However, the algorithm continued to espouse an unfair gender bias against women, simply because there had been a gender bias in Amazon's prior hiring history, which was reflected in the training data. As no sufficient data on successful female applicants was available to train or fine-tune the model to reduce its gender bias, the problem of de-biasing this algorithm seemed intractable, 'executives lost hope for the project' (Dastin, 2018), and it was stopped.

[^4]:    ${ }^{6}$ Analogously, recommender systems attempting to remove toxic content have been shown to disproportionately affect the monetisation and distribution of LGBT + content (Dixon et al., 2018; Romano, 2019).

    ${ }^{7}$ In most countries there are 'protected traits' that may not be discriminated against. In the United States, they are: gender, race, religion, age (over 40), disability, national origin, disability, family status and genetic information. In the United Kingdom, protected categories include sexual orientation, pregnancy, and people undergoing gender reassignment.

[^5]:    ${ }^{8}$ An individual may also consent to their private data forming part of a training corpus at one point in time, but revoke that consent later on.

    ${ }^{9}$ Doxing is "the intentional public release onto the Internet of personal information about an individual by a third party, often with the intent to humiliate, threaten, intimidate, or punish the identified individual."

[^6]:    ${ }^{10}$ Nonsensical predictions have been shown in LMs but these are not explicitly discussed here, as these are unlikely to trigger a false belief in a user.

[^7]:    ${ }^{11}$ Some recommender systems have been found to respond to certain user behaviour by recommending more and more extreme viewpoints to increase engagement ((O'Callaghan et al., 2015; Yesilada and Lewandowsky, 2021); for counterexamples view (Möller et al., 2018)).

[^8]:    12"Nudging" refers to 'any aspect of the choice architecture that alters people's behavior in a predictable way without forbidding any options or significantly changing their economic incentives' Thaler \& Sunstein 2009. More simply put, nudging refers to the 'use of flaws in human judgment and choice to influence people's behavior' Hausman \& Welch 2010.

[^9]:    ${ }^{13}$ This section features no prompt/reply textboxes because the risks discussed here are not well expressed in the format of a question answering language agent.

[^10]:    ${ }^{14}$ Patent-busting occurs when an innovation is made that is sufficiently similar to capture the market of the original invention, but is sufficiently distinct not to constitute a patent violation. For example, this may occur where a developed drug compound is similar to a patented compound and achieves the same pharmacological effects; here this drug compound is made more widely accessible than the original, such patent-busting can create social benefit.

</end of paper 0>


<paper 1>
# Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model 

Shaden Smith ${ }^{\S, \dagger}$, Mostofa Patwary ${ }^{\S, \ddagger}$, Brandon Norick $^{\dagger}$, Patrick LeGresley ${ }^{\ddagger}$, Samyam<br>Rajbhandari ${ }^{\dagger}$, Jared Casper ${ }^{\ddagger}$, Zhun Liu ${ }^{\dagger}$, Shrimai Prabhumoye ${ }^{\ddagger}$, George Zerveas ${ }^{* \dagger}$, Vijay<br>Korthikanti ${ }^{\ddagger}$, Elton Zhang ${ }^{\dagger}$, Rewon Child ${ }^{\ddagger}$, Reza Yazdani Aminabadi ${ }^{\dagger}$, Julie Bernauer ${ }^{\ddagger}$, Xia<br>Song $^{\dagger}$, Mohammad Shoeybi ${ }^{\ddagger}$, Yuxiong $\mathrm{He}^{\dagger}$, Michael Houston ${ }^{\ddagger}$, Saurabh Tiwary ${ }^{\dagger}$, and Bryan<br>Catanzaro ${ }^{\ddagger}$<br>§equal contribution<br>${ }^{\dagger}$ Microsoft<br>${ }^{\ddagger}$ NVIDIA


#### Abstract

Pretrained general-purpose language models can achieve state-of-the-art accuracies in various natural language processing domains by adapting to downstream tasks via zero-shot, few-shot and finetuning techniques. Because of their success, the size of these models has increased rapidly, requiring high-performance hardware, software, and algorithmic techniques to enable training such large models. As the result of a joint effort between Microsoft and NVIDIA, we present details on the training of the largest monolithic transformer based language model, Megatron-Turing NLG 530B (MT-NLG), with 530 billion parameters. In this paper, we first focus on the infrastructure as well as the 3D parallelism methodology used to train this model using DeepSpeed and Megatron. Next, we detail the training process, the design of our training corpus, and our data curation techniques, which we believe is a key ingredient to the success of the model. Finally, we discuss various evaluation results, as well as other interesting observations and new properties exhibited by MT-NLG. We demonstrate that MT-NLG achieves superior zero-, one-, and few-shot learning accuracies on several NLP benchmarks and establishes new state-of-the-art results. We believe that our contributions will help further the development of large-scale training infrastructures, large-scale language models, and natural language generations.


## 1 Introduction

The recently released foundation models [8], such as BERT [12], GPT-2 [52], and RoBERTa [37], represent a paradigm shift in which AI systems can be built by pretraining a general class of models at scale and then adapting them for a wide range of downstream tasks through transfer learning. Such models became ubiquitous in state-of-the-art natural language processing (NLP) systems by embracing the effectiveness of a combination of factors: the transformer architecture [67], self-supervised learning, few-shot conditioning [9], and fine-tuning.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-02.jpg?height=922&width=1038&top_left_y=257&top_left_x=552)

Figure 1: Trend of sizes of state-of-the-art NLP models with time.

Importantly, many recent works have established that scaling up models greatly improves their performance, with especially substantial performance improvements in the zero-shot and few-shot settings. For example, GPT-3 [9], an autoregressive language model with 175 billion parameters, performs competitively on language tasks using in-context learning without fine-tuning or gradient updates. Such in-context learning allows models to perform new language tasks with only simple instructions and a few optional examples. The effectiveness of this method was further enhanced by recent model adaptation work such as prompt tuning [33], which efficiently adapts large language models to individual tasks with robust task performance. Other intriguing capabilities exhibited by large language models include, but are not limited to, free-form generation of coherent, long-form text like news stories, generating responses with real-world knowledge, as well as performing rudimentary mathematical operations.

The rapid development of large language models in recent years has also been fueled by growth in computational resources, availability of large datasets and evolving software stacks. State-of-the-art supercomputing clusters address the computation, memory and networking need of model training at this scale. Careful processing of high-quality, high-volume and diverse datasets directly contributes to model performance in downstream tasks as well as model convergence. New approaches to numerical manipulation and training recipes were developed aiming at improved optimization efficiency and stability. However, to sustain the seemingly exponential growth of model parameter size (see Figure 11, substantial progress in developing new methods, infrastructure and training capabilities is needed.

Training such large models is challenging for two reasons. First, it is no longer possible to fit the parameters of these models in the memory of even the largest GPU. Second, the large number of compute operations
required can result in unrealistically long training times if special attention is not paid to concurrently optimizing the algorithms, software, and hardware stack. This calls for efficient parallelism techniques scalable on both memory and compute, in order to achieve the full potential of thousands of GPUs.

Compelled by the impressive qualitative performance improvements owing to an increasing model size that have been previously exhibited, our work continues the trend of large-scale language modeling. We built Megatron-Turing NLG 530B (MT-NLG), a transformer-based language model with 530 billion parameters. It is, to the best of our knowledge, the largest monolithic language model trained to date, with $3 \mathrm{x}$ more parameters than GPT-3. It is worth noting that sparse models structures encompassing a higher total number of parameters, such as mixture-of-experts [61], have been trained. However, it is unclear whether models built following this approach would have comparable parameter efficiency and generalization capability.

Training MT-NLG was made feasible by numerous innovations and breakthroughs along all AI axes. Through a collaboration between NVIDIA Megatron-LM [63, 43] and Microsoft DeepSpeed [57, 65], we created an efficient and scalable 3D parallel system capable of combining data, pipeline, and tensor-slicing based parallelism. By combining tensor-slicing and pipeline parallelism, we can operate within the regime where they are most effective. We built high-quality, natural language training corpora with hundreds of billions of tokens, and co-developed training recipes to improve optimization efficiency and stability.

In this paper, we will discuss details of our methods during the development of MT-NLG, including training infrastructure (Section 2), training dataset and training process (Section 3), model evaluation and other interesting observations (Section 4). We will also present an in-depth study on social biases (Section 5), in-context learning capability (Section6) and qualitative analysis of the generation capability (Section7) of MT-NLG.

## 2 Large Model Training Infrastructure

Powered by NVIDIA A100 Tensor Core GPUs and HDR InfiniBand networking, state-of-art clusters (such as NVIDIA Selene and Microsoft Azure NDv4) have enough compute power to train models with trillions of parameters. However, achieving the full potential of these supercomputers requires memory- and computeefficient strategies for parallelizing across thousands of GPUs. In isolation, existing parallelism strategies such as data, pipeline, or tensor-slicing have trade-offs in memory and compute efficiency and cannot be used to train models at this scale. In this section, we discuss the system challenges of training large models. We describe our software design, hardware system, and the performance evaluation of a unified, powerful training infrastructure.

### 2.1 Challenges

We begin by discussing the challenges of training large-scale language models: memory and compute efficiency, and the tradeoffs of various solution strategies such as data, tensor and pipeline parallelism.

### 2.1.1 Memory and Compute Efficiency

Memory Efficiency The memory requirements to train a 530 billion parameter model are far beyond what is available on a single GPU device. We refer to Rajbhandari et al. [56] for an analytical study of memory consumption during training.

Mixed precision training [41] typically stores weights and gradients in half precision formats (i.e., 2 bytes per parameter) for forward and backward propagations. It also keeps full-precision (4 bytes) copies in 32 bit float format for numerical stability in the optimizer. Assuming training with the Adam optimizer [27], training consumess 20 bytes of memory per parameter:

$$
\underbrace{2+4}_{\text {weights }}+\underbrace{2+4}_{\text {gradients }}+\underbrace{4+4}_{\text {Adam states }}=20 \text { bytes. }
$$

Training a 530 billion parameter model thus requires over 10 terabytes of aggregate memory for the model weights, gradients, and optimizer states.

Activations can also consume significant memory and scale with the training batch size, sequence length, and model dimensions. Checkpointing and recomputing the activations of each transformer block is a common strategy for training large language models to reduce the memory required for activations. However, the activations at the boundary between layers still needs to be stored and the aggregate activation memory is:

batch-size $\times$ number-of-layers $\times$ sequence-length $\times$ hidden-dimension $\times 2$ bytes,

which is approximately 16.9 terabytes following our model and training configuration (Section 3.2).

Fortunately, activation memory requirements can be mitigated by virtue of gradient accumulation. Gradient accumulation is a strategy in which the full training batch is split into micro-batches that are processed in sequence and their resulting gradients are accumulated before updating the model weights. After computing the gradient for a micro-batch, the associated activations can be freed. As a result, the training batch size can scale without increasing the peak resident activation memory. For example, training with 1920 microbatches instead of a single micro-batch of size 1920 reduces the peak activation memory from 16.9 terabytes to 8.8 gigabytes without changing the effective batch size.

Compute Efficiency While large GPU clusters can have thousands of high-throughput GPUs, achieving high compute efficiency at this scale is challenging. A large batch size can be an effective way of increasing compute efficiency, because it increases the arithmetic intensity of a kernel and helps amortize the time spent stalled on communication and synchronization. However, the batch size that a model can be trained with has an upper bound; using too large of a batch size can have negative effects on the model quality. With 4000 GPUs, even a large batch size of 4000 would only allow for a batch size of 1 per GPU and limit compute efficiency.

### 2.1.2 Tradeoffs of Data, Tensor, and Pipeline Parallelism

Data Parallelism Data parallelism is a ubiquitous technique in deep learning in which each input batch of training data is divided among the data-parallel workers. Gradients are communicated and aggregated among data-parallel workers before updating the model weights. Data parallelism has several distinct advantages, including compute efficiency and ease of implementation. However, data parallelism relies on scaling the batch size with the number of data-parallel workers, and cannot be made arbitrarily large without affecting model quality.

Memory Efficiency: Data parallelism replicates the model and optimizer across all workers, and therefore is not memory efficient. The Zero Redundancy Optimizer (ZeRO) [55] is a collection of optimizations that
improve the memory efficiency of data parallelism by partitioning the replicated data among data-parallel workers.

Compute Efficiency: The amount of computation performed by each worker is constant as we increase the degree of parallelism and training batch size. Data parallelism can achieve near-perfect scaling at small scales. However, the communication cost of aggregating gradients increases with the model size and can limit compute efficiency on large models or systems with low communication bandwidth. Gradient accumulation is also a common strategy for amortizing this communication cost by further increasing the batch size and performing multiple forward and backward propagations on micro-batches while locally accumulating gradients before aggregating and taking an optimizer step. Additionally, performance can be increased by simultaneously communicating gradients that have already been communicated in parallel with computing the gradients for other tensors.

Tensor Model Parallelism Tensor model parallelism (or, tensor parallelism) is a broad class of model parallelism techniques that partitions the individual layers of the model across workers. Tensor parallelism reduces the memory proportional to the number of workers. Megatron [63] uses model parallelism to efficiently partition transformer blocks for large-scale language models.

Memory Efficiency: Tensor parallelism reduces the memory footprint of the model proportional to the number of workers. Depending on the model architecture, some of the activation memory is also reduced, although there may still be some replications.

Compute Efficiency: Tensor parallelism introduces additional communication of activations in each forward and backward propagation. Therefore, tensor parallelism requires high communication bandwidth to be efficient and is best kept within a single DGX sever where high bandwidth NVLink is available. Furthermore, each model-parallel worker decreases the amount of computation performed between each communication stage, impacting compute efficiency. Tensor parallelism is often used to expand the envelope of memory and compute efficiency beyond what data parallelism alone can do.

Pipeline Model Parallelism Pipeline model parallelism (or, pipeline parallelism) divides the layers of the model into stages that can be processed in parallel [23, 42]. As one stage completes the forward pass for a micro-batch, the activation memory is communicated to the next stage in the pipeline. Similarly, as the next stage completes its backward propagation, gradients are communicated backwards through the pipeline. Multiple micro-batches must be kept in flight to ensure pipeline stages compute in parallel.

Memory Efficiency: Pipeline parallelism reduces memory proportionally to the number of pipeline stages, allowing model size to scale linearly with the number of workers. However, pipeline parallelism does not reduce the memory footprint for the activations of each layer. Additionally, each worker must store the activations for all micro-batches in flight. We use a 1F1B pipeline schedule [42] that alternates forward and backward propagations. A key benefit of $1 \mathrm{~F} 1 \mathrm{~B}$ is that the number of micro-batches in flight is bounded by the number of pipeline stages, as opposed to the total number of micro-batches in a full training batch.

Compute Efficiency: Pipeline parallelism has the smallest communication overhead of the three approaches, as it only communicates the activations between the pipeline stage boundaries. However, it cannot scale indefinitely. The degree of pipeline parallelism is bounded by the depth of the model, and increasing the pipeline dimension decreases the compute efficiency like other forms of model parallelism. Pipeline paral-
lelism also requires each of its stages to be load balanced for high efficiency.

Pipeline parallelism incurs a bubble overhead from filling and emptying the pipeline at the beginning and end of each training batch. The size of the bubble overhead bounds the potential speedup from pipeline parallelism. The fraction of perfect speedup achievable (or, parallel efficiency) is a function of the number of pipeline stages $(P P)$ and total micro-batches $(M B)$ :

$$
\text { efficiency }=\frac{M B}{M B+P P-1}
$$

If the number of micro-batches is $4 x$ or $8 x$ the number of pipeline stages, the pipeline achieves $81 \%$ and $90 \%$ parallel efficiency from one pipeline stage, respectively.

From the above discussion, it is clear that none of the existing parallelism techniques can address all the system challenges of training models with hundreds of billions of parameters. However, each parallelism technique has its own merits and can be used in a complementary fashion. To this end, we use $3 D$ parallelism, which is a systematic combination of data, tensor, and pipeline parallelism that addresses both compute and memory efficiency simultaneously.

### 2.2 Software System - 3D Parallelism with DeepSpeed and Megatron

Our system software stack combines pipeline parallelism and data parallelism from DeepSpeed with tensorslicing from Megatron to create a flexible 3D-parallelism implementation. Data, tensor, and pipeline parallelism each play a specific role in improving memory and compute efficiency.

Memory Efficiency: Transformer blocks are divided into pipeline stages, and the blocks of each stage are further divided via tensor parallelism. This 2D combination simultaneously reduces the memory consumed by the weights, gradients, optimizer states, and activations. However, we cannot partition the model indefinitely without losing compute efficiency.

Compute Efficiency: To further accelerate training, we use data parallelism to scale to arbitrarily large number of GPUs. For example, each 530 billion parameter model replica spans 280 NVIDIA A100 GPUs, with 8 -way tensor-slicing within a node and 35 -way pipeline parallelism across nodes. We then use data parallelism to scale out further to thousands of GPUs.

Our 3D parallelism implementation is optimized using topology aware mapping, which minimizes communication overhead across all forms of parallelism, and has an especially large impact on data parallelism. This mapping is key to achieving excellent compute efficiency at scale. We discuss the details below.

### 2.2.1 Topology-Aware 3D Mapping

Each axis of parallelism is carefully mapped onto the workers to maximize compute efficiency by exploiting two key architectural properties.

Mapping for Bandwidth Intra-node communication has a higher bandwidth than inter-node. We prioritize co-locating parallel groups with larger communication volumes in order to utilize the higher bandwidth. Tensor parallelism has the largest communication overhead of the three strategies, and so we prioritize plac-
ing tensor parallel workers within a node. When possible, data parallel workers are also placed within a node to accelerate gradient communications. Otherwise, data parallel workers are mapped to nearby nodes when possible. Pipeline parallelism has the lowest communication volume, and so we can schedule pipeline stages across nodes without being limited by the communication bandwidth.

Bandwidth Amplification The volume of gradient communication by each data parallel group decreases linearly as pipeline and tensor parallelism increase. Thus, the total communication volume is decreased from pure data parallelism. Furthermore, each data parallel group performs its communication independently and in parallel among a subset of more localized workers. As a result, the effective bandwidth for data parallel communication is amplified by a combination of reduced communication volume and increased locality and parallelism.

### 2.3 Hardware System

Model training is done with mixed precision using 16-bit bfloat on NVIDIA's Selene [2] supercomputer with 560 DGX A100 nodes. Each cluster node has 8 NVIDIA 80-GB A100 GPUs [1], connected to each other by NVLink and NVSwitch [3]. Each node has eight NVIDIA Mellanox 200Gbps HDR Infiniband HCAs for application communication, with an additional two HCAs per node for dedicated storage. The nodes are connected in a three-level (leaf, spine, core) fat-tree topology with 850 switches. This topology allows efficient all-reduce communication (which is the dominant communication pattern in deep learning training). The cluster uses an all-NVME shared parallel filesystem for high-performance data access and storage. The peak device throughput of an A100 GPU with 16-bit precision is 312 teraFLOP/s, resulting in an aggregate of 1.4 exaFLOP/s of peak 16-bit precision performance.

### 2.4 System Performance Evaluation

We considered the end-to-end throughput of our system for the 530 billion parameter model with batch size 1920 on 280,350 , and 420 DGX A100 servers on Selene. We observed iteration times of $60.1,50.2$, and 44.4 seconds, respectively. These correspond to 126,121 , and 113 teraFLOP/s per GPU, respectively.

## 3 Training Dataset and Model Configuration

In this section we present details on the training datasets, our preprocessing techniques, and the model and hyperparameters used in our experiments.

### 3.1 Training Dataset and Preprocessing

Resources such as Common Crawl (CC) provide snapshots of the web which can be utilized as a source of language data. While these data sources contain an enormous amount of language data, they also require carefully designed preprocessing steps in order to select data which is of reasonable quality. As prior work has found (e.g., $[9]$ ), the quality of unfiltered Common Crawl data is lower than that of curated datasets and steps should be taken to increase the average quality of data selected from Common Crawl for LM pretraining. In addition to CC data, there are many other high quality data sources on the web. To compile our training dataset, we made use of recent work aimed at collecting a diverse training set for language modeling [17]. We additionally included RealNews [77] and CC-Stories [66] which have previously been

| Dataset | Tokens (billion) | Weights (\%) | Epochs |
| :--- | :---: | :---: | :---: |
| Books3 | 25.7 | 14.3 | 1.5 |
| OpenWebText2 | 14.8 | 19.3 | 3.6 |
| Stack Exchange | 11.6 | 5.7 | 1.4 |
| PubMed Abstracts | 4.4 | 2.9 | 1.8 |
| Wikipedia | 4.2 | 4.8 | 3.2 |
| Gutenberg (PG-19) | 2.7 | 0.9 | 0.9 |
| BookCorpus2 | 1.5 | 1.0 | 1.8 |
| NIH ExPorter | 0.3 | 0.2 | 1.8 |
| ArXiv | 20.8 | 1.4 | 0.2 |
| GitHub | 24.3 | 1.6 | 0.2 |
| Pile-CC | 49.8 | 9.4 | 0.5 |
| CC-2020-50 | 68.7 | 13.0 | 0.5 |
| CC-2021-04 | 82.6 | 15.7 | 0.5 |
| Realnews | 21.9 | 9.0 | 1.1 |
| CC-Stories | 5.3 | 0.9 | 0.5 |

Table 1: Datasets used to train the MT-NLG model. The top 11 rows are from the Pile dataset, followed by two Common Crawl snapshots, RealNews, and CC-Stories datasets.

used for large LM pretraining [4, 63].

### 3.1.1 Training Dataset

We largely built upon prior work described in [9 , 17] to generate our training set. First, we selected a subset of the datasets from The Pile that we observed to be of the highest relative quality (see Table 1). Then, following a similar approach as that used to generate Pile-CC in [17], we downloaded and filtered two full CC snapshots (2020-50 and 2021-04). At a high level, the steps taken for CC data include text extraction from raw HTML provided in WARC files, scoring extracted documents using a classifier trained on high quality data, and filtering documents according to their scores. These steps are covered in more detail in Section 3.1.2 Finally, we used fuzzy deduplication to remove duplicate and near duplicate documents from the entire dataset as well as $n$-gram based filtering to remove downstream task data in order to avoid contamination.

### 3.1.2 Pre-Processing Details

Common Crawl: As mentioned previously, Common Crawl comprises an immense amount of data. We chose to process two snapshots, 2020-50 and 2021-04, with the aim of acquiring around 150B tokens of training data. The first step of this process is language detection [11] and text extraction from the raw HTML included in the Common Crawl WARC files Following the rationale presented in [11], we used the pycld2 $2^{2}$ and jusTex $\sqrt[3]{3}$ libraries for these tasks. We observe that the language detection and extraction step reduces the number of documents significantly, with only around $25 \%$ of documents being classified as[^1]

English and having non-empty body content.

In order to select high quality documents from these extractions, we trained a 2-gram fastText [48] classifier. For positive documents, we randomly select 500000, 295000, and 5000 documents from OpenWebText2, Wikipedia, and Books3, respectively, similar to [9]. For negative documents, we randomly sampled an equal number of documents from the text extraction output described above. We held out $10 \%$ of these documents for evaluation of the classifier, which achieved an accuracy of $90.3 \%$ on the held out set after training. The classifier was applied to each of the extracted documents and the probability of the positive label was taken as the score for the document.

Using the scores produced by the process above, we filtered the extracted documents with a Pareto distribution with $\alpha=3$. This resulted in around $80 \%$ of text content being filtered. While our choice of $\alpha$ is lower than some previous works [9], manual inspection of the data indicated that it was of acceptable quality and the use of $\alpha=3$ allowed us to reach and slightly exceed our original token goal after deduplication.

Other Datasets: In addition to Common Crawl data, we leveraged a number of other previously generated datasets. From The Pile, we selected Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, Gutenberg (PG-19), BookCorpus2, NIH ExPorter, and Pile-CC datasets. We also included the CC-Stories and RealNews datasets used to train Megatron [63]. For detailed discussions of the preprocessing used for these datasets, we refer to [17].

Fuzzy Document Deduplication: Content on the internet is often duplicated across many documents. To compound this issue, the URLs scraped in different Common Crawl snapshots are not necessarily unique. Indeed, for the snapshots we chose $53 \%$ and $34 \%$ of documents come from new URLs not seen in previous snapshots. Furthermore, it is likely that content contained in our other datasets, such as web content from OpenWebText2 or Wikipedia, will also exist in Commom Crawl.

Exact match duplicates would be computationally expensive, so we opted to take a fuzzy deduplication approach similar to other works [9, 17]. We used a hashing vectorizer with $1,048,576$ features to vectorize documents (HashingVectorizer from scikit-learn ${ }^{4}$ ), calculated min-hashes of the vectorized documents (using datasketch $^{5}$, and performed Locality Sensitive Hashing (LSH) through datasketch on all minhashes in order to identify potential duplicates. We set our LSH parameters in such a way as to increase the likelihood that documents with Jaccard similarity $\geq 0.8$ would occur in at least one LSH bucket together. Specifically, we used 20 bands of size 13 for a total of 260 hash functions.

After performing LSH, we processed each bucket and calculated an approximation of the all-pairs Jaccard similarity in order to remove false positive duplicates introduced by LSH. This approximation consisted of $i=0 . .10$ iterations of sampling a random document $d_{i}$, calculating the Jaccard similarity with everything remaining in the bucket, removing those documents above the 0.8 threshold and marking them as duplicates of $d_{i}$. After all buckets were processed and duplicates (at the threshold) were approximately discovered, we constructed a sparse document graph and found the connected components therein (using scipy). Each connected component represents a set of documents that we consider similar enough to be duplicates, and from which we select a single representative. Because the datasets are of varying quality, we defined a priority order based on which dataset to use when selecting representative documents, and the first document[^2]encountered from the highest priority dataset within each component was ultimately kept, while the rest were discarded.

Additional Processing: We use the Ftfy library [64] on the training dataset to convert bad unicode text to good unicode text. Additionally, we use the langdetect [11] library to identify non-English documents and remove any document such with less than 512 characters. If a training document contains the word "javascript" and has less than 256 characters, we remove that document as well.

Downstream Task Data Removal: We use $n$-grams to remove texts that occur in the downstream tasks from the training datasets. When we find an $n$-gram match between a task document and a training document, we split the training document into two pieces by removing the $n$-gram along with 200 characters from both of its sides. We also remove any split training document with fewer than 200 characters, or training documents which were split more than 10 times. Our deduplication process and the values of $n$ used for different tasks are similar to [9]. Out of $319,781,622$ documents from the 15 deduplicated datasets mentioned above, during task deduplication 35,988 documents were split, 1,109 documents were removed, 54 documents were split more than 10 times, and 9,891 were trimmed at the beginning or the end.

Blending Datasets: We opted to blend the datasets into heterogeneous batches according to the sampling weights given in Table 1. However, the mixing weights do not result in an even split of the samples in each batch for our chosen batch size. To resolve this issue, we track the under- and oversampling for each dataset and slightly adjust the batch composition at each step in order to maintain a sample distribution as close as possible to the chosen mixing weight distribution.

### 3.2 Model and Training Process

We used the architecture of the transformer decoder [52], which is a left-to-right, autoregressive, generative transformer-based language model, and scaled it up to 530 billion parameters. The number of layers, hidden dimensions, attention heads are 105, 20480, and 128, respectively. The sequence length is 2048 and the global batch size is 1920 . We used 8 -way tensor and 35 -way pipeline parallelism. The learning rate is $5.0 e^{-5}$. We used one billion tokens for linear learning rate warmup. We used cosine decay for the learning rate targeting to reach $10 \%$ of its value over 340 billion tokens. Over the first 12 billion tokens, we started at a batch size of 32 and gradually increased the batch size in increments of 32 , until we reach the final batch size of 1920. We used Adam optimizer with $\beta_{1}=0.9, \beta_{2}=0.95$, and $\epsilon=10^{-8}$. We clipped the gradient norm at 1.0 and used a weight decay of 0.1 . For weight initialization, we used a normal distribution with zero mean and a standard deviation of $4.0 e^{-3}$. Our training dataset consists of 339 billion tokens and we trained MT-NLG on 270 billions tokens by blending the 15 training datasets as described above. We also set aside $2 \%$ of our data for validation.

At the scale of models such as MT-NLG, training stability is a fundamental challenge. While training the model, we observed that the learning rate, weight initialization, and Adam optimizer parameters directly affect model stability. We projected the learning rate for MT-NLG by plotting the learning rates with the size of the models in [9]. Higher learning rate increases the model instability. We used approximately $\sqrt{1 /(3 * H)}$ as a standard deviation for weight initialization, where $H$ denotes the size of the hidden dimension. Similar to [45], we also observed that using higher variance for weight initialization fails to converge. We also reduced $\beta_{2}$ from its standard value of 0.99 to reduce spikes in the training loss.

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-11.jpg?height=626&width=1265&top_left_y=278&top_left_x=430)

Figure 2: Validation loss of MT-NLG.

## 4 Results and Achievements

To provide a better understanding of how language model performance improves during training, we first present the validation loss curve (cross entropy) of MT-NLG in Figure2. Our validation dataset consists of 5.5 billion tokens, so measuring the loss using the entire dataset is computationally expensive. We therefore shuffle the sequences in the validation dataset and then during each computation of validation loss, we run four iterations with global batch size of 1920 . This leads to evaluating on a total of 16 million consecutive tokens for each loss computation.

The validation cross-entropy loss is 3.15 after the model is trained on the first 1 billion tokens. As mentioned earlier, we increase the batch size linearly over the first 12 billion tokens. At the end of this phase, the loss becomes 2.31. When the model reaches our targeted number of tokens, 270 billion, the validation loss becomes 1.85 .

To evaluate the quality of our model (as well as other pretrained language models), we adopt a zero-/one/few-shot evaluation setting similar to prior work [9, 53]. For better reproducibility, we base our evaluation on the open-source project, lm-evaluation-harness [18], and made task-specific changes as appropriate to align our setting more closely with prior work. We will discuss any idiosyncrasies of each task in the task-specific paragraphs. In addition, for our few-shot experiments, we do not do any search for the optimal number of shots, and directly use the configurations suggested in [9]. In most cases, they seem to perform sufficiently well.

To ensure the evaluation is comprehensive, we choose eight tasks from five different categories: completion prediction, reading comprehension, commonsense reasoning, natural language inference and word sense disambiguation. We present comparisons on these tasks with previous works on pretrained large language models, while also providing supervised baselines whenever applicable to provide context for the gap between "generalist" models like pretrained language models and "specialist" models that are finetuned on the target task.

|  | LAMBADA (acc) |  |  |
| :--- | :---: | :---: | :---: |
| Model | Zero-shot | One-shot | Few-shot |
| GPT-3 | 76.20 | 72.50 | 86.40 |
| Gopher | 74.50 | - | - |
| MT-NLG (ours) | $\mathbf{7 6 . 5 6}$ | $\mathbf{7 3 . 0 6}$ | $\mathbf{8 7 . 1 5}$ |

Table 2: LAMBADA zero-shot, one-shot and few-shot accuracy. MT-NLG outperforms previous models across different settings and establishes new SOTA for all 3 settings. We did not find any recent strong supervised baseline for LAMBADA, hence we omit the comparison with supervised models here.

Many evaluation tasks involve scoring candidate completion sentences with the model. Unless otherwise stated, the "likelihood" mentioned in the following context refers to the probability of the candidate answer (conditioned on the prompt) normalized by its number of tokens.

### 4.1 Completion Prediction

LAMBADA The LAMBADA [49] dataset is a collection of narrative passages, specifically selected such that a human can easily guess the last word if the whole passage is given as context, but would not be able to answer if only given the last sentence in the passage. This task tests language models' capabilities to understand and retain information from a broader discourse context, instead of just relying on local context or simple statistical patterns.

When evaluating this task zero-shot, we feed each passage to the model as input and check if the model can produce the correct last word via greedy generation (picking tokens with maximum probability). However, for one-/few-shot evaluations, we switched over to a cloze-style prompt format to better suggest to the model that the task is about predicting the last word of a sentence as opposed to arbitrary plausible continuation. In such a case, we would insert “_-_. $\rightarrow$ " before the last word, e.g. “... Paul and Debbie looked at each other, then at __-. $\rightarrow$ Bob" and examine if the model would predict the correct word after the " $\rightarrow$ ". We observe significant performance boost in few-shot settings with the cloze-style prompting, although one-shot performance takes a hit, which aligns with observations from prior work [9]. Our model's performance in terms of accuracy is shown in table 2, and we are establishing new state-of-the-arts on LAMBADA for all 3 settings on its test set.

### 4.2 Reading Comprehension

In this section, we discuss the evaluation of MT-NLG for reading comprehension. We selected two datasets targeting different styles of questions, and have found very different trends when we increase the number of examples for them during evaluation.

RACE RACE [31] is a large-scale reading comprehension dataset, whose passages and questions are extracted from English examinations. Each example in this task consists of an article and several questionanswer pairs. To construct prompts, we prepend "Article: ", "Question: ", and "Answer: " tags to the article, questions and answers text respectively and join them together with a newline in between. The actual answer to the last question is removed, ending the prompt at the last "Answer:". We then use the
model to score all possible candidate answers as continuations after "Answer :" and pick the highest-scoring one as the model's choice.

There are two question types in this dataset: direct questions (e.g. "Which of the following relationships is healthy?") and cloze-style questions (e.g. "The author of the text seems to _."). We treat both question types the same way as described above, which is different from the default used by lm-evaluation-harness [18]. Furthermore, following GPT-3 [9], we use

$$
\frac{P(\text { completion } \mid \text { context })}{P(\text { completion } \mid \text { answer_context })}
$$

as the scoring criterion, where context is the full prompt, and answer_context is just the string "Answer :". Similar to GPT-3, we observe a better performance compared to using length-normalized log-probabilities as a scoring criterion for RACE.

The dataset contains two subsets, RACE-h and RACE-m, corresponding to hard and medium problems. We report results on the RACE-h set in Table 3. We observe that RACE-h performance does not benefit much from including more examples in the prompt. Nevertheless, our zero-shot performance already surpasses few-shot performance of GPT-3 by $+1.14 \%$.

For RACE dataset, one of the best supervised models to date is an ALBERT ensemble [24]. It achieves $91.4 \%$ accuracy on RACE-h, which is significantly higher than the results obtained by pretrained language models. Recent work [53] has greatly narrowed the gap between prerained language models and supervised models, but the difference is still large.

BoolQ BoolQ [10] is a dataset of yes/no questions, with supporting Wikipedia paragraphs to answer them. We concatenate the supporting paragraph, the question (prepended with "Question: ") and a string "Answer:" at the end as the full prompt. We use the model to score "yes" and "no" as continuations and choose the option with higher likelihood given by the model. Our model's performance is shown in Table 3. We observe that BoolQ evaluation benefits significantly from seeing many examples in the prompt, which differs from results on the RACE task. However, one common pattern here is that reading comprehension tasks can get a decent improvement with just one example, possibly because the task prompting format is confusing to the model, and the given example is enough to condition the model to follow the passage-question-answer format.

For BoolQ, T5 + UDG [69] is currently the best supervised model. It achieves $91.4 \%$ accuracy on this task. However, compared to RACE-h, we observe that the gap between supervised model and pretrained language model is much smaller and that MT-NLG further narrows the gap by a significant amount.

### 4.3 Commonsense Reasoning

An interesting aspect of pre-trained language models is how much world knowledge they preserve from their training data. To this end, we evaluate our models on two tasks relating to commonsense reasoning/inference. The supervised baseline we compare to on these 3 datasets is UNICORN [38].[^3]

| Task | Model | Zero-shot | One-shot | Few-shot | Supervised |
| :--- | :--- | :---: | :---: | :---: | :---: |
| RACE-h | GPT-3 | 45.50 | 45.90 | 46.80 | - |
|  | Gopher | - | - | $\mathbf{7 1 . 6 0} 0^{6}$ | - |
|  | MT-NLG (ours) | $\mathbf{4 7 . 9 4}$ | $\mathbf{4 8 . 4 2}$ | 47.94 | - |
|  | ALBERT (ensemble) | - | - | - | $\underline{91.40}$ |
| BoolQ | GPT-3 | 60.50 | 76.70 | 77.50 | - |
|  | MT-NLG (ours) | $\mathbf{7 8 . 2 0}$ | $\mathbf{8 2 . 5 1}$ | $\mathbf{8 4 . 8 3}$ | - |
|  | T5 + UDG | - | - | - | $\underline{91.40}$ |

Table 3: Reading comprehension results on RACE-h and BoolQ. BoolQ scores significantly improve from zero-shot to few-shot, while RACE-h does not benefit from having many examples. This is likely due to the fact that BoolQ's prompt/answer pairs have a more structured format (single-word, boolean answers) which the model can only learn through few-shot context, whereas RACE-h answers are already fairly close to natural sentences and the model benefits comparatively less from seeing examples.

Winogrande Winogrande [58] is a dataset that seeks to expand the Winograd Schema Challenge in both scale and difficulty. The task is in the form of pronoun resolution problems that are designed to be unsolvable for statistical language modeling alone, and that require commonsense knowledge about the underlying events and objects to solve.

For this task, we adopt the evaluation method used by previous work [9, 52, 66]. We substitute the actual noun with an ambiguous pronoun, and evaluate the likelihood of the partial sentence starting from the pronoun conditioned on the previous context. The pronoun substitution that leads to the highest likelihood is selected as the model answer. The results are shown in Table 4 . Compared to GPT-3, we observe a strong improvement in terms of zero-shot accuracy ( $+2.81 \%$ ), though the gap narrows for few-shot. We observe that having one example in context only marginally improves performance, but moving to the few-shot setting significantly improves model performance. As we will see in the other two tasks, this appears to be a general trend: commonsense reasoning performance scales well with number of shots. This is a distinct trend compared to what we see in reading comprehension.

HellaSWAG HellaSWAG [76] is a commonsense reasoning dataset where a goal is given and the model is tasked with choosing the most likely follow-up actions. The examples are mined from Wikihow and Activitynet Captions [29] dataset. During evaluation, we prompt the model with the goal, then evaluate the likelihood of each candidate answer conditioned on the goal, and choose the candidate answer with the highest likelihood. The results are shown in Table 4. We achieved significant improvements compared to GPT-3 in all 3 settings, with our zero-shot performance surpassing few-shot for GPT-3. Similar to Winogrande, moving from zero-shot to one-shot doesn't improve performance much (in fact, it decreases it in this case), but including more examples in the few-shot setting substantially increases performance.

PiQA PiQA [6] is a binary-choice question answering dataset targeting understanding of physical interactions. It poses questions about how to complete a daily activity, and the model is tasked with choosing between two candidate answers describing different actions to take.

For evaluation on PiQA, we prompt the model with the question/goal description and then evaluate the

| Task | Model | Zero-shot | One-shot | Few-shot | Supervised |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Winogrande | GPT-3 | 70.20 | 73.20 | 77.70 | - |
|  | Gopher | 70.20 | - | - | - |
|  | MT-NLG (ours) | $\mathbf{7 3 . 0 1}$ | $\mathbf{7 3 . 7 2}$ | $\mathbf{7 8 . 8 5}$ | - |
|  | UNICORN | - | - | - | $\underline{91.28}$ |
| HellaSWAG | GPT-3 | 78.90 | 78.10 | 79.30 | - |
|  | Gopher | 79.20 | - | - | - |
|  | MT-NLG (ours) | $\mathbf{8 0 . 2 4}$ | $\mathbf{8 0 . 2 0}$ | $\mathbf{8 2 . 4 2}$ | - |
|  | UNICORN | - | - | - | $\underline{93.90}$ |
| PiQA | GPT-3 | 81.00 | 80.50 | 82.30 | - |
|  | Gopher | 81.80 | - | - | - |
|  | MT-NLG (ours) | $\mathbf{8 1 . 9 9}$ | $\mathbf{8 0 . 9 6}$ | $\mathbf{8 3 . 1 9}$ | - |
|  | UNICORN | - | - | - | $\underline{90.10}$ |

Table 4: Commonsense reasoning results on Winogrande, HellaSWAG and PiQA. We generally observe minor gain or even performance dips when moving from zero-shot to one-shot, but would observe significant gains when we move from zero-shot to few-shot settings. On common sense reasoning, supervised baseline [38] still outperforms LMs with few-shot learning settings.

likelihood of the candidate sentences for two different actions, choosing the option with higher likelihood as the model answer. The results are shown in Table 4. We once again observe the pattern that one-shot performance degrades compared to zero-shot, while few-shot performance gets a decent boost.

### 4.4 Natural Language Inference

In this section we discuss the evaluation of our model on natural language inference (NLI) tasks.

ANLI The ANLI [46] dataset is an adversarially mined NLI dataset that aims to create a difficult set of NLI problems. The dataset has 3 iterative rounds of data collection; here, we evaluate with round 2 data. During evaluation, we rephrase the NLI problem into a question-answering format: each example is structured as "<premise>\nQuestion:<hypothesis>. True, False or Neither? \nAnswer:" and then we examine which continuation among True, False or Neither has the highest likelihood assigned by the model, and pick the most likely option as the model answer. The results are shown in Table 5. On ANLI, we observe that, similar to reading comprehension results, our model is able to get a performance gain by just having one example, and moving beyond that into few-shot setting does not further improve performance. Again, this is possibly because one example is important for instructing the model on the premise-hypothesis-answer format, but additional examples may be unrelated in terms of content, and including them does not introduce new knowledge for the model. On ANLI, the supervised baseline we compare to is InfoBERT [68].

HANS Heuristic Analysis for NLI Systems (HANS) [40] is an NLI dataset designed to evaluate the tendency of models to exploit fallible, superficial syntactic heuristics in NLP data. It offers a controlled evaluation setting where examples are generated from templates of specific grammatical and syntactical structures (each type of structure referred to as a "subcase"). The task format is akin to ANLI, with the NLI problem

| Task | Model | Zero-shot | One-shot | Few-shot | Supervised |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ANLI (R2) | GPT-3 | 35.40 | 33.90 | 34.00 | - |
|  | MT-NLG (ours) | $\mathbf{3 6 . 6 0}$ | $\mathbf{3 9 . 7 0}$ | $\mathbf{3 9 . 6 0}$ | - |
|  | InfoBERT | - | - | - | $\underline{51.40}$ |
| HANS | GPT-2 | $\mathbf{5 4 . 7 9}$ | 49.92 | 49.79 | - |
|  | MT-NLG (ours) | 51.61 | $\mathbf{6 0 . 0 1}$ | $\mathbf{7 3 . 1 6}$ | - |

Table 5: Natural language inference results on ANLI (R2) and HANS datasets. At zero-shot, models are struggling at chance level for HANS, yet MT-NLG is very effective in leveraging in-context examples as the number of shots increases, resulting in a large performance boost. Scaling behavior w.r.t number of shots is shown in Figure 5 .

converted into a binary question answering format (see Section A in Appendix for details). We implemented this task and included it in our evaluation among existing tasks in the lm-evaluation-harness [18].

Besides evaluating our model's core language understanding capabilities, we use the HANS dataset primarily as a means to analyze its behavior in few-shot learning, which is presented in Section 6. We report our aggregate results obtained during the analysis experiments in Table 5, and a comparison of various MTNLG checkpoints across different number of shots in Figure 5. No prompt-based generative baselines have been previously released on this dataset, so we evaluate GPT-2 for comparison. As described in Section6, performance at zero-shot is driven by inherent model biases and accuracy is only slightly better than random chance $\mathbf{(} 50 \%)$. However, large models which have been sufficiently trained can take advantage of in-context examples in the prompt to dramatically improve performance, while weaker models can be confused when given additional in-context examples, with GPT-2 never performing substantially better than random chance.

### 4.5 Word Sense Disambiguation

WiC The Word-in-Context [50] dataset presents a task of identifying the intended meaning of polysemous words from their context. Each dataset example consists of 2 sentences, both containing the same polysemous word. The task is to identify if the intended meaning of the polysemous word is the same or not in the two sentences.

To perform zero- /few-shot evaluations on this task, we convert the problem into a question answering format: "Sentence 1:<sentence1> 1 nSentence 2:<sentence2> nQuestion: Is the word <target word> used in the same way in the two sentences above? \nAnswer:". Then we examine the model-assigned likelihoods for "yes" and "no" as continuations, and pick the one with higher likelihood as the model answer. Results can be found in Table 6. We observe that our model performs slightly below chance at zero-shot, but surpasses chance as soon as we move to few-shot. On the other hand, the supervised T5 + UDG model surpasses chance-level significantly.[^4]

|  | WiC (acc) |  |  |  |
| :--- | :---: | :---: | :---: | :---: |
| Model | Zero-shot | One-shot | Few-shot | Supervised |
| GPT-3 | $0.09^{7}$ | 48.60 | 55.30 | - |
| MT-NLG (ours) | $\mathbf{4 8 . 5 9}$ | $\mathbf{5 1 . 2 5}$ | $\mathbf{5 8 . 4 6}$ | - |
| T5 + UDG | - | - | - | $\underline{77.9}$ |

Table 6: Word-in-Context dataset results. We see significant improvements moving from zero-shot to fewshot, surpassing chance level performance.

## 5 Exploring Social Biases

### 5.1 Introducing the Challenge of Social Bias in NLP Models

Natural language models are trained on massive datasets collected from a wide variety of uncurated sources. Unfortunately, biased communication and writing is pervasive on the internet, and spans the gamut from very inflammatory and overt bias that may also be straightforward to identify, to more subtle forms of bias. As has been shown in other work (for example [9, 71, 53]), bias issues that exist in the dataset can be learned by models as they are trained on the data. This limits the deployment of large language models, despite their powerful capabilities.

Although not the focus of this paper, we note that ongoing research in several areas aims to mitigate this bias. For example,

a) Training set filtering - where the elements of the training dataset are analyzed and elements that show evidence of bias are removed from the training data [44].

b) Training set modification - where elements of the training dataset are randomized with respect to variables such as gender and ethnicity that should be neutral with respect to the subject matter [72].

c) Prompt engineering - where the inputs to the model for each query are modified to steer the model away from bias [60, 14].

d) Fine tuning - where the trained model is retrained to unlearn biased tendencies [19, 20, 28].

e) Output steering - where a filtering step is added to the inference procedure to re-weight output values and steer the output away from biased responses.

In this work, we have trained a baseline model without any anti-bias countermeasures. We want to emphasize that we do not believe that such models should be deployed in production use without countermeasures, and specifically, we do not believe that the MT-NLG model should be deployed as such. Rather, it is our expectation that this work will be a valuable resource supporting research efforts into anti-bias countermeasures and as a starting point for future deployments that make use of such countermeasures. We are also confident that future work combining state-of-the-art large language model training with state of the art antibias countermeasures will result in models that are both very powerful and able to minimize the presence of bias in results.

We now present a preliminary analysis of the bias of our model, focusing on issues related to gender, ethnicity, and religion. Similar to Brown et al. [9] and Rae et al. [53], we present association tests, cooccurrence analyses and sentiment analyses designed to assess bias along a variety of dimensions. This section serves to document and quantify some of the problems relating to bias of the model that will need to be addressed in future work.

### 5.2 Gender and Occupation Analysis

We begin by investigating the association between gender and occupations, to understand if the model learns the societal biases present in its training data. ${ }^{8}$ Following previous work [7, 39], we use the list of 323 occupations shown in Tables 18 and 19 of Section B. The template used in this experiment is "The \{occupation\} was a \{gender identifier \}", and the gender identifiers used are male, man, and female, woman.

Given a fixed occupation, we calculate the probability the model assigns to different gender identifiers and count the number of times male identifiers had a higher probability than female identifiers. Over all 323 occupations, that percentage is $78 \%$, suggesting that the model is generally biased towards masculine identifiers.

We calculate average occupation bias scores, which measures whether the model tends to associate a specific gender with a given occupation. Following Brown et al. [9], occupation bias scores are calculated as

$$
\frac{1}{N_{o c c}} \sum_{o c c}(\log (\mathrm{P}(\text { female_identifier|prompt) })-\log (\mathrm{P}(\text { male_identifier|prompt })))
$$

Here, a score of 0 indicates that there is no biased association between a specific occupation and gender identifiers; a positive score indicates a skewed association between female identifiers and occupations; and a negative score indicates a skewed association between male identifiers and occupations. The average bias score provides an indication of how far or close were the probabilities of male and female identifiers for all the occupations. Our model exhibits an average bias score of -0.77 , indicating that the model leans towards male identifiers for more occupations.

To give some context for these two results, we note that for GPT-3, the percentage of occupations for which male identifiers had a higher probability over female identifiers is $83 \%$ and the average bias score is -1.11 [9]. We should note that these results are not directly comparable to our results due to a difference in the number of occupations and type of template used but do suggest that the model is not particularly more or less biased than others trained in similar ways on similar corpora. It also suggests that more work must be done to control this bias when considering using this model.

### 5.3 Adjective Co-Occurrence Analysis

Next, we analyze the adjectives that co-occur with prompts related to gender, ethnicity and religion. Similar to the setup in Brown et al. [9], for each prompt we sample 800 different generations of length 50 each. Each generation has a temperature of 1.0 and top $p$ value of 0.9 . We use the respect templates presented in Sheng et al. [62] to create our prompts. For gender and ethnicity, the prompt is created as " \{gender/ethnicity[^5]

| Male Identifiers | He, The man, The father, The son |
| :--- | :--- |
| Female Identifiers | She, The woman, The mother, The daughter |
| Ethnic Identifiers | The $\{$ ethnicity $\}$ man, The $\{$ ethnicity $\}$ woman, The $\{$ ethnicity $\}$ person |
| Templates | \{identifier $\}$ was very, \{identifier $\}$ would be described as, <br> \{identifier $\}$ was known for, \{identifier $\}$ was regarded as, |
|  | \{identifier $\}$ was thought of as, \{identifier $\}$ was well-known for |

Table 7: Identifiers used for gender and ethnicity. In racial identifiers, the $\{$ ethnicity $\}$ would take one of the following values: Asian, Black, and White. The templates [62] mentioned above are for both gender and ethnic studies. The $\{$ identifier $\}$ in templates will be replaced by gender or ethnic identifiers.

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-19.jpg?height=258&width=1619&top_left_y=760&top_left_x=253)

Figure 3: The 100 most common words associated with male and female templates, ordered from most common (on the left) to least common (on the right). Identical words are connected with blue lines. The model generally assigns similar words similar probabilities regardless of gender; distinctive words for each gender are noted in green.

identifier\} template". Example prompts are "The woman was regarded as" and "The Black person would be described as". The identifiers and templates used to create the prompts are provided in Table 7 .

We split the generated text into sentences and consider only the first sentence for our analysis, after observing that only the first sentence is descriptive of the given identifiers (and that the following sentences typically describe random events or stories). We analyze the most frequent 100 adjectives that co-occur with each of the identifiers of gender and ethnicity.

Generally speaking, there are two factors that are important to consider for co-occurrence analysis:

a) The ordinal position of distinct words. Higher position / lower frequency is good because it indicates a low intensity of bias with respect to a particular stereotypical or offensive adjective, even if the adjective itself is highly offensive.

b) The magnitude of stereotypical or offensive content implied in the distinct adjective. Some adjectives are relatively neutral while others are strongly offensive.

We would like to note that while co-occurrence analysis provides us with a rich understanding of the frequency of surface level forms such as words that co-occur with certain identifiers, they fail to take into account sentiment or context associated with each adjective.

Gender Analysis Encouragingly, we note that, for gender, among the top 100 most frequent adjectives, almost 80 were exactly the same, as shown in Figure 3. In the figure, words are ordered left-to-right in order

```
Male $\operatorname{top}_{(51)}$, violent $_{(53)}$, eccentric $_{(59)}$, military $_{(60)}$, polite $_{(62)}$, serious $_{(63)}$, national $(67)$,
    different $_{(68)}$, aggressive $(71)$, right $_{(78)}$

```

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-20.jpg?height=54&width=1471&top_left_y=355&top_left_x=327)

```
        $\operatorname{slim}_{(67)}$, only $(72)$, excited $_{(74)}$
```

Table 8: Top 10 distinct words with the highest frequency from the 100 most frequent words that occurred for Male and Female identifiers. The numbers in parenthesis represent the word's ordinal position in the top 100 most frequent words list.

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-20.jpg?height=55&width=1469&top_left_y=680&top_left_x=285)
desirable $_{(77)}$, feminine $(88)$, pleasant $_{(91)}$

```
![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-20.jpg?height=49&width=1521&top_left_y=778&top_left_x=286)
```

    athletic $_{(88)}$, incapable $_{(82)}$
    ```
![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-20.jpg?height=60&width=1545&top_left_y=867&top_left_x=282)
```

        pale $_{(90)}$, responsible $(92)$
    ```

Table 9: Top 10 distinct words with the highest frequency from the 100 most frequent words that occurred for ethnicity identifiers. The numbers in parenthesis represent the word's ordinal position in the top 100 most frequent words list.

of probability; if a word is in the top 100 of both genders, it is linked by a blue line. Generally, the model draws upon an equal set of words, but there are notable exceptions that we highlight in green.

In the interests of highlighting the bias of the model, we also present the 10 most distinct words with the highest frequency for each gender (but emphasize that this hides the non-bias discussed previously). Table 8 presents the most distinct words that had a higher frequency of co-occurring with one gender over the other. We observe that the model conforms to gender stereotypes that are present in the training data, such as using more adjectives related to appearance for female identifiers and using a more diverse set of adjectives for male identifiers. On the other hand, qualitatively, the ordinal position of the stereotypical distinct adjectives is relatively higher (less frequent), which is a good property

Ethnicity Analysis For ethnicity, results for the same adjective co-occurrence analysis for the 10 distinct words with the highest frequency are shown in Table 9 . We qualitatively observe that more positive adjectives are associated with the White ethnicity, and that many quite offensive adjectives are associated with other ethnicities. We note that the model portrays offensive stereotypes associated with each ethnicity as well. In addition, and arguably more importantly, we observe that the ordinal position of the distinct adjectives is higher than in the gender case.

These results are clearly unacceptable for deployment and confirm our intuition that NLP models should require use of anti-bias countermeasures as described in the introduction to this section. We look forward to repeating these tests with such countermeasures applied and validating quantifiable improvement in results.

Religion Analysis For religion, we perform an analysis of the co-occurring words similar to gender and ethnicity. The three prompts used for religion are "The \{religious_practitioners \} are", "The \{religious_practitio

\begin{tabular}{|c|c|}
\hline hel & \begin{tabular}{l}
$(43), \operatorname{lack}_{(53)}$, reason $_{(54)}, \operatorname{preach}_{(62)}$, existence $_{(63)}$, \\
$\operatorname{lan}_{(81)}$
\end{tabular} \\
\hline Bud & ![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-21.jpg?height=105\&width=1323\&top_left_y=350\&top_left_x=507) \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-21.jpg?height=105\&width=1323\&top_left_y=445\&top_left_x=507) \\
\hline & ![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-21.jpg?height=105\&width=1323\&top_left_y=540\&top_left_x=507) \\
\hline Isla & \begin{tabular}{l}
$\operatorname{hijab}_{(11)}, \operatorname{modesty}_{(27)}, \operatorname{prophet}_{(34)}, \operatorname{law}_{(35)}, \operatorname{cover}_{(47)}, \operatorname{Allah}_{(55)}$, face $_{(57)}, \operatorname{mosque}_{(59)}$, \\
$\operatorname{countries}_{(65)}, \operatorname{veil}_{(67)}$
\end{tabular} \\
\hline Judaism & \begin{tabular}{l} 
Jewish $_{(8)}$, white \\
(18), mone \\
birth $_{(59)}$, intelligence \\
$(63)$
\end{tabular} \\
\hline
\end{tabular}

Table 10: Top 10 distinct words with the highest frequency from the 100 most frequent words that occurred for religion identifiers. The numbers in parenthesis represent the word's ordinal position in the top 100 most frequent words list.

ners\} are known for" and "The \{religious_practitioners\} practice". 9 Table 10 shows the top 10 most distinct words that co-occur with a higher frequency for each of the six religions. Encouragingly, mostly we do not observe negative words used for any particular religion with higher frequency.

\subsection*{5.4 Sentiment Analysis}

We use sentiment analysis as an additional method to measure bias. We chose to focus on ethnicity for this analysis because ethnicity was the dimension that showed the strongest bias issues in the Adjective Co-Occurrence Analysis Section above.

We apply this method by analyzing the sentiment of all the words that co-occur. For each word in the generated text, we use SentiWordNet [51] to measure both positive and negative scores on a scale of 0 to 100. We average these scores for all words in the generated text. Figure 4 shows the average sentiment scores for each of three ethnicities.

We observe that for the Black ethnicity, the negative sentiment words co-occur with considerably higher proportion, and that correspondingly positive sentiment words co-occur with lower proportion as compared to the other ethnicities. The sentiment for Asian and White ethnicities are more comparable to each other. Clearly, the bias in sentiment exhibited in the results is also severe and validates the need for anti-bias countermeasures as part of natural language training.

\subsection*{5.5 Discussion}

Large NLP models such as MT-NLG have demonstrated amazing power to assimilate vast quantities of unstructured information and make it easily accessible. However, they have also been shown to have a
\footnotetext{
${ }^{9}$ Note that we only use three templates to prompt the model, and hence this study is not as robust as our others, but is included for completeness.
}

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-22.jpg?height=667&width=1097&top_left_y=263&top_left_x=514)

Figure 4: Positive and Negative sentiment scores for each ethnicity

problem with absorbing bias that is embedded in the information they are given to learn from.

We have included this section to examine the biases present in our model, which was trained without any countermeasures to combat bias in the input training set. Based on results from previous work, we expected to find evidence of significant bias in the model, and that expectation was confirmed in our results, with several instances of pervasive, strong, and offensive bias. Models trained without proper countermeasures should not be deployed as-is (i.e., without anti-bias countermeasures), for this reason.

\section*{6 Natural Language Understanding and In-Context Learning}

To evaluate the core language understanding capabilities of large transformer-based language models as directly as possible, it is essential that we assess their ability to grasp the systematicity of language: in other words, their ability to learn implicit grammatical and syntactical rules on which humans consciously or unconsciously rely in order to generalize to arbitrarily many, unprecedented utterances. In this section, we attempt this with the HANS dataset, but begin with a discussion of limitations of other NLP benchmarks.

\subsection*{6.1 Limitations of NLP benchmarks}

Pretrained language models based on the transformer architecture have dominated the state of the art in NLP over the last few years, achieving impressive performance in a wide array of downstream tasks. In certain tasks, such as natural language inference, they have been shown to even surpass human-level performance [54]. Nevertheless, there has been mounting evidence in recent work suggesting that the performance of these models as measured by the benchmark datasets may be overestimated, non-generalizable and at least partially driven by exploiting existing spurious correlations in training datasets [21, 22, 40, 47, 75]. The reason why large transformer models may not generalize well out-of-distribution can be attributed to the combination of two factors: on the one hand, their enormous learning capacity, and on the other, the narrowness of the training set distributions of downstream tasks, which is related to how these datasets were mined
or crowdsourced. The expressiveness of these models allows them to easily discover and exploit spurious correlations in these datasets during fine-tuning, leading to impressive performance metrics which, however, do not necessarily reflect their actual natural language understanding capabilities.

Brown et al. [9] suggest few-shot learning as a way to both evaluate large language models more accurately, as well as to overcome the problem of overfitting on narrow distributions; this is because no parameter updates take place when solving downstream tasks, and all learning happens "in-context", exclusively based on the provided input prompt. These properties appear as very significant advantages of few-shot capable models, alongside the convenience of eschewing the creation of task-specific datasets, and subsequently fine-tuning and maintaining task-specific models. For this reason, it is important to elucidate to what extent they hold true.

\subsection*{6.2 Evaluating Grasp of Language Systematicity}

The HANS dataset [40] allows us to evaluate to what extent language models can consistently apply rules for inferring entailment, as opposed to relying on superficial heuristics such as vocabulary overlap or the existence of common subsequences in both premise and hypothesis. To focus on basic language parsing, the vocabulary is intentionally chosen to be very simple, and all words occur several times in the most common NLI datasets such as MNLI [73]. Besides the ground truth label ("entailment" versus "non-entailment"), each example in the dataset is annotated with respect to the one out of the 30 different grammatical/syntactical constructions (called "subcases") that it is meant to probe. More information about the HANS dataset and charateristic examples can be found in Section $\mathrm{A}$ of the Appendix.

\subsection*{6.3 Factors Affecting In-Context Learning}

Model size and amount of training In Figure 5 we show how natural language inference performance is affected by the number of shot examples, that is, the number of solved examples presented to the model as part of the prompt; we additionally show the effect of further autoregressive pretraining. We can first observe that the HANS task appears to be challenging for large language models, although it would be considered trivially easy for humans, compared to the current standard reading comprehension, reasoning and inference benchmark datasets. In particular, the 1.5 billion parameter GPT-2 never manages to perform significantly better than random chance ( $50 \%$ for a balanced binary classification task), no matter how many shot examples it is presented with. By contrast, we find that our 530 billion parameter large model, MTNLG is largely capable of escaping superficial heuristics and successfully leveraging syntactical rules for inference. Apart from model size, two important factors which clearly affect performance are the amount of autoregressive pretraining it has undergone (i.e. the number of tokens it has encountered), as well as the number of prompt examples (shots).

Number of Shots We found it crucial that the model is first shown a couple of examples in order to understand how to solve the task; for most model checkpoints, the peak accuracy is achieved when the model is shown 2 examples (2-shot). We found that this improvement in performance appears to be driven by the fact that the initial 2 shots increase the model's probability of predicting either one of the two desired answer tokens, "True" and "False", from an average of $70 \%$ at 0 -shot, to $100 \%$ at 2 -shot. We additionally found that the initial two shots allow the model to calibrate a strong inherent bias in preferring either one of the two classes at 0 -shot, which likely originates from the content the model has been trained on.

Apart from our own observations on results presented in Section4, it has also been previously reported that while a large number of shot examples can help in some datasets, in many cases the opposite is true [9]. Here we observe that only the largest and most well-trained models can benefit from additional examples beyond the first few shots. We speculate that additional shots introduce confusion to weaker models, by distracting the self-attention mechanism from focusing on the example under evaluation, while in well-trained, highcapacity models, self-attention can still selectively attend to the most relevant samples within the prompt, as well as the evaluated sample.

Distribution of Shots In order to further elucidate under which circumstances a larger number of shot examples can help, we repeated the evaluation in two different settings: in the first setting, we enforce the examples that appears in the few-shot prompts to only come from subcases different from that of the example being evaluated - this is the "sanitized" setup. We follow this setting for all HANS evaluations in Figure 5 and elsewhere in the paper, unless otherwise noted. In the second setting, we did not control shot examples by subcase, and thus, as the number of shots increases, there is an increasing chance for the model to encounter examples from the same subcase as the example under evaluation. Indeed, we observed that when not filtering shot examples, performance substantially increases with an increasing number of shots, while the opposite is true when the type of shot examples is dissimilar to the example under evaluation. We can therefore conclude that the role of shot examples is not merely to provide guidance with respect to the format of the task. Instead, just like it is true with fine-tuning, even in the case of in-context learning, the distribution of samples used to guide the model and the distribution of samples on which it is evaluated needs to be matched to obtain best performance, as we observe the model performs distinctly better on samples from the same distribution as the one it has been exposed to in the prompt. This serves as first evidence that in-context learning does not automatically circumvent the issue of "overfitting" on narrow distributions, and we expect this effect to hold in other NLP datasets, where the type/distribution of samples used as prompt shots either cannot be explicitly controlled or hasn't yet been examined. At the same time, Figure 5 seems to imply that a larger model scale combined with more pretraining can improve the generalization capabilities of models relying on in-context learning, as such models (the 270 billion tokens MT-NLG checkpoint, in particular) can benefit even from prompt examples which less strictly match the distribution of evaluation samples.

Shot Labels and Label Order Furthermore, we found additional factors which significantly affect performance and are related to the composition of the set of shot examples included in the prompt, in a manner equivalent to a conventional parameter training process. For example, the order of shot examples plays a significant role, and we found that shot samples should be shuffled or interleaved with respect to their class labels in order to maximize performance. Even more importantly, the composition of the set of shots with respect to class labels, i.e. the proportion of "positive" to "negative" labels, seems to drastically affect the prediction probabilities for the examples under evaluation: a small proportion of "positive" shots results in a substantially decreased probability of predicting any samples under examination to be "positive" ("non-entailment" in our dataset), while the probability of predicting the "positive" label for any example under evaluation rapidly increases as the proportion of "positive" shot examples increases. This change in predicted labels distributions, introduced by controlling the proportion of class presence in the set of shots, allows us to counteract inherent biases in the model: for example, it allows us to boost accuracy from $70.2 \%$ to $73 \%$ for 2 -shot when only including "negatives" as shot examples. Moreover, increasing the number of shots also profoundly changes the mean, variance and skewness of class prediction distributions, and when combined with shifting the decision threshold, it can be used to counteract the biases of the model and

![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-25.jpg?height=791&width=1301&top_left_y=239&top_left_x=412)

Figure 5: Natural Language Inference accuracy on the HANS dataset, as a function of the number of shots and the amount of training (number of tokens encountered during pretraining).

significantly improve accuracy to $78.6 \%$.

Overcoming Inference Biases and Reliance on Heuristics Finally, we proceed to examine how well our model can handle each of the 30 different linguistic "subcases" of interest, for example, passive voice, or disentangling relative clauses. We present the results in Table 12 of the Appendix. Although the strong inherent biases of the model initially cause it to be very susceptible to the vocabulary overlap, subsequence and constituent heuristics, we were able to drastically improve the model's performance by controlling prediction distributions through increasing the number of shots and at the same time differentially shifting distribution means by taking into account unconditional prediction probabilities. Therefore, it was eventually possible to confirm that the model can consistently "apply" (i.e., take into consideration for inference) many of the grammatical/syntactical rules which humans regard as essential for understanding natural language. Encouragingly, the subcases which the model had difficulty handling were mostly the same as the ones humans (especially novice speakers) would typically find confusing (see examples in Table 11 and Table 12 .

\subsection*{6.4 Summary of Evaluation}

We found that very large, pretrained language models can be shown to "understand" (i.e. take into account) grammatical and syntactical structure in the prompt-based, generative setting, thus leveraging the systematicity of language to solve tasks without having been fine-tuned. This basic linguistic performance increases with model size and the amount of pretraining. Importantly, it is commensurate with NLP benchmark performance, indicating that metrics on common benchmark datasets, despite their individual limitations and spurious effects, in aggregate indeed correlate well with language understanding.

However, we also found that these models by default also rely on superficial heuristics such as lexical overlap and the presence of shared sentence subsequences between premise and hypothesis when performing inference. Furthermore, they can have strong inherent biases with respect to sample classes, and can be very sensitive to the task formulation (formatting).

Importantly, we found that in-context learning appears to be following similar principles as standard learning through tuning parameters: for example, the order of shot samples matters. More crucially, the data distribution of shot examples (both in terms of example types and proportion of class labels) determines performance on evaluation samples, and optimal performance can only be achieved when the shot and evaluation distributions match. Therefore, in-context learning cannot be seen as an automatic solution to the problem of overfitting on narrow distributions, i.e. poor out-of-distribution generalization performance.

Together, the above observations show that special effort is necessary to elicit correct responses from large language models in the prompt-based setting, and suggest that there is still significant room for improvement with respect to the goal of using a generic, task-agnostic generative model which can replace models finetuned to solve the task.

\section*{7 Qualitative Examples for MT-NLG Generation Capabilities}

As an addition to quantitative evaluation and analysis on benchmark datasets, we also qualitatively examined the language generation capabilities on novel scenarios. To our pleasant surprise, MT-NLG is quite capable in solving riddles, answering Jeopardy questions and even generating code off-the-shelf. We present some examples of each category below.

Riddle Answer Generation We used riddles to probe the model's reasoning capability in an ambiguous context, crafting each riddle ourselves in order to prevent their incidence in the training set. We first observe that in a riddle-solving context, the model tends to generate its interpretation of each line in the riddle along with its answer. While not always perfect, these interpretations most of the time make good sense. Such an example is shown in Table 13. For riddles that are ambiguous enough to have multiple plausible answers, MT-NLG not only generates alternative plausible answers through stochastic sampling, but it can also generate alternative interpretations matching the answer it has generated (Table 14).

Jeopardy Questions Question answering datasets [30, 25] often poses specific and direct questions to benchmark the models. However, we are also interested in how the model can utilize the knowledge it memorized in a guessing game setting, where some reasoning over the hints is required. To this end, we take several Jeopardy! questions from the most recent episode and let our model generate the answers. Since Jeopardy! questions take the reverse trivia format where the "question" is in the format of an answer and contestants are asked to select matching questions, we choose to use few-shot setting to inform the model of the task format. MT-NLG can generate fairly plausible answers and in fact get the correct ones in most cases. Some examples is shown in Table 15

Code Generation The recent development of code generation using language models suggests that large scale pretrained LMs already show decent code generation capabilities from pretraining. To this end, we investigate the code generation capability of MT-NLG off-the-shelf. We presented some function signatures with detailed comments to see how MT-NLG would complete the implementation of the missing function. We observe that MT-NLG is capable of generating syntactically correct code consistently, and is also able
to arrive at correct implementations for simple tasks. We sometimes observe that the model will generate an answer making use of another function, and then move on to generate the invoked function after the current one is finished. Some examples of this are shown in Table 16 .

Inferring Arithmetic Operations Understanding and using mathematical operations is yet another aspect of language understanding. Prior work [9] has demonstrated that a strong language model, even if not trained specifically to solve math problems, can answer simple arithmetic questions with a certain degree of accuracy beyond chance. However, some doubts remain as to whether the model indeed has some understanding of math expressions, or whether it simply rehashes examples encountered during training. To this end, we devise a new task where we obfuscate operator symbols in an expression and check if our model can reverse-engineer the arithmetic operation. We observe that common operations like addition, subtraction, multiplication and division can usually be inferred correctly. Some examples of this task is shown in Table 17

Free-form Generative Writing Assistance We qualitatively examined the free-form generation capability of MT-NLG by enlisting the model to help authoring the abstract section of this paper. This was done through prompting MT-NLG with the text from Section 1, then proceeding to sample the model sentence by sentence. For each sentence multiple candidates were generated, from which one was picked and edited if necessary. We repeated this process until the abstraction excerpt appeared complete.

\section*{8 Related Works}

Improving model performance through scaling model and dataset size has witnessed great success in recent years, especially in natural language processing. Before the currently prevailing paradigm of large-scale pretraining, there has already been efforts in scaling up LSTM models [26] to over a billion parameters. This trend is continued when large-scale pretraining with transformer architectures becomes popular, with BERT [12] scaling up to 300 million parameters, followed by GPT-2 [52] at 1.5 billion parameters. Scaling beyond this point requires more sophisticated training techniques, but the rapid development of new system software, data, model and pipeline parallelism techniques have enabled another wave of even larger models.

Some prior works have chosen to use the mixture-of-experts (MoE) [32, 35, 61] technique to scale to larger model sizes more economically, producing large-scale models that selectively use a subset of its parameters in each forward pass. MoE allows for extreme scaling in terms of model sizes, with recent work reaching 1.6, 1.75 and even 10 trillion [5, 15, 36] parameters. However, the line of work that is more relevant to MT-NLG is in the scaling of monolithic, dense transformer architectures. Prior work after GPT-2 produced dense transformer models at 8 billion [63], 11 billion [54], and 17 billion [4] parameters, and GPT-3 [9] at 175 billion parameters demonstrated for the first time that language models at such scale begin to exhibit zero-/few-shot learning capabilities that are missing in smaller models. Since then, several other hundredbillion scale dense transformer models have been announced, among them are Jurassic-1 [34], Yuan 1.0 [74], PanGu- $\alpha$ [78] and Gopher [53]. Our work further extends this line of work, situating ourselves at the largest monolithic transformer language model to date at 530 billion parameters, achieving unprecedented training efficiency and model quality.

There has also been recent work focusing on directly improving language model's zero-shot learning capabilities through large-scale multitask finetuning. Both T0 [59] and FLAN [70] have taken this path and have shown that such an approach can improve zero-shot learning capabilities of language models. This
approach has been shown to apply well to a pretrained language model [70], observing that the larger the model size, the more benefit it extracts from such training methods. We hope that our breakthroughs in large-scale pretraining are synergistic with these methods, and will produce even better models in the future.

\section*{9 Conclusions}

In this work, we presented MT-NLG, a 530 billion parameter left-to-right, autoregressive, generative transformer-based language model that possesses strong in-context learning capabilities. MT-NLG achieved superior zero-/one- and few-shot learning performance on several NLP benchmarks, establishing new state-ofthe-art results. We discussed the challenges in training neural networks at such scale and presented our 3D-parallelism strategies as well as hardware infrastructures that enabled efficient training of MT-NLG. Large language model training is challenging to stabilize and experimentation can be costly, therefore, we documented our training configurations and datasets extensively to facilitate future research. Last but not least, we analyze the social biases exhibited by MT-NLG and also examined various factors that can affect in-context learning, bringing forth awareness of certain limitations of current generation of large language models. We believe that our results and findings can help, shape, and facilitate future research in foundational, large-scale pretraining.

\section*{References}

[1] NVIDIA A100 Tensor Core GPU. https://www.nvidia.com/en-us/data-center/a100/.

[2] NVIDIA Selene Supercomputer. https://www.top500.org/system/179842/.

[3] NVLink and NVSwitch. https://www.nvidia.com/en-us/data-center/nvlink/.

[4] Turing-NLG: A 17-billion-parameter language model by Microsoft. https://www.microsoft.com/en-us/research/blog/ turing-nlg-a-17-billion-parameter-language-model-by-microsoft/.

[5] Wu Dao 2.0 Large-scale Pretrained Model. https://wudaoai.cn/home

[6] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In AAAI, 2020.

[7] Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In NIPS, 2016.

[8] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

[9] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel HerbertVoss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia

Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[10] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924-2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[11] Michal Danilk. langdetect, 2021. Version 1.0.9.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In $N A A C L, 2019$.

[13] Penelope Eckert and Sally McConnell-Ginet. Language and Gender. Cambridge University Press, 2003.

[14] Zahra Fatemi, Chen Xing, Wenhao Liu, and Caiming Xiong. Improving gender fairness of pre-trained language models without catastrophic forgetting. arXiv preprint arXiv:2110.05367, 2021.

[15] William Fedus, Barret Zoph, and Noam M. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv, abs/2101.03961, 2021.

[16] Lyn Frazier and Janet D. Fodor. The sausage machine: A new two-stage parsing model. Cognition, 6(4):291-325, 1978. Place: Netherlands Publisher: Elsevier Science.

[17] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

[18] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.

[19] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356-3369, 2020.

[20] Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don't stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342-8360, Online, July 2020. Association for Computational Linguistics.

[21] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R. Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In Marilyn A. Walker, Heng Ji, and Amanda Stent, editors, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 2 (Short Papers), pages 107-112. Association for Computational Linguistics, 2018.

[22] Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and Dawn Song. Pretrained transformers improve out-of-distribution robustness. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 2744-2751. Association for Computational Linguistics, 2020.

[23] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32:103-112, 2019.

[24] Yufan Jiang, Shuangzhi Wu, Jing Gong, Yahui Cheng, Peng Meng, Weiliang Lin, Zhibo Chen, and $\mathrm{Mu} \mathrm{Li}$. Improving machine reading comprehension with single-choice decision and transfer learning. ArXiv, abs/2011.03292, 2020.

[25] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In $A C L, 2017$.

[26] Rafal Józefowicz, Oriol Vinyals, Mike Schuster, Noam M. Shazeer, and Yonghui Wu. Exploring the limits of language modeling. ArXiv, abs/1602.02410, 2016.

[27] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[28] Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, richard socher, and Nazneen Rajani. Gedi: Generative discriminator guided sequence generation, 2021.

[29] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. 2017 IEEE International Conference on Computer Vision (ICCV), pages 706-715, 2017 .

[30] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics, 7:453-466, 082019.

[31] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785-794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.

[32] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yan-Ping Huang, Maxim Krikun, Noam M. Shazeer, and Z. Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. ArXiv, abs/2006.16668, 2021.

[33] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

[34] Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. Jurassic-1: Technical details and evaluation.

[35] Junyang Lin, Rui Men, An Yang, Chan Zhou, Ming Ding, Yichang Zhang, Peng Wang, Ang Wang, Le Jiang, Xianyan Jia, J. Zhang, Jianwei Zhang, Xu Zou, Zhikang Li, Xiao Qing Deng, Jie Liu, Jinbao Xue, Huiling Zhou, Jianxin Ma, Jin Yu, Yonghong Li, Wei Lin, Jingren Zhou, J ie Tang, and Hongxia Yang. M6: A chinese multimodal pretrainer. ArXiv, abs/2103.00823, 2021.

[36] Junyang Lin, An Yang, Jinze Bai, Chang Zhou, Le Jiang, Xianyan Jia, Ang Wang, Jie Zhang, Yong Li, Wei Lin, Jingren Zhou, and Hongxia Yang. M6-10t: A sharing-delinking paradigm for efficient multi-trillion parameter pretraining. 2021.

[37] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

[38] Nicholas Lourie, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Unicorn on rainbow: A universal commonsense reasoning model on a new multitask benchmark. In AAAI, 2021.

[39] Thomas Manzini, Lim Yao Chong, Alan W Black, and Yulia Tsvetkov. Black is to criminal as caucasian is to police: Detecting and removing multiclass bias in word embeddings. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 615-621, 2019.

[40] Tom McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3428-3448, Florence, Italy, July 2019. Association for Computational Linguistics.

[41] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.

[42] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. Pipedream: generalized pipeline parallelism for dnn training. In Proceedings of the 27th ACM Symposium on Operating Systems Principles, pages 1-15, 2019 .

[43] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaroand Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm. ArXiv, abs/2104.04473, 2021.

[44] Helen Ngo, Cooper Raterink, João GM Araújo, Ivan Zhang, Carol Chen, Adrien Morisot, and Nicholas Frosst. Mitigating harm in language models with conditional-likelihood filtration. arXiv preprint arXiv:2108.07790, 2021.

[45] Toan Q. Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. CoRR, abs/1910.05895, 2019.

[46] Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial nli: A new benchmark for natural language understanding. ArXiv, abs/1910.14599, 2020.

[47] Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial NLI: A new benchmark for natural language understanding. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 4885-4901. Association for Computational Linguistics, 2020.

[48] Pedro Javier Ortiz Suárez, Benoît Sagot, and Laurent Romary. Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures. In Piotr Bański, Adrien Barbaresi, Hanno Biber, Evelyn Breiteneder, Simon Clematide, Marc Kupietz, Harald Lüngen, and Caroline Iliadi, editors, 7 th Workshop on the Challenges in the Management of Large Corpora (CMLC-7), Cardiff, United Kingdom, July 2019. Leibniz-Institut für Deutsche Sprache.

[49] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525-1534, Berlin, Germany, August 2016. Association for Computational Linguistics.

[50] Mohammad Taher Pilehvar and José Camacho-Collados. Wic: the word-in-context dataset for evaluating context-sensitive meaning representations. In $N A A C L, 2019$.

[51] Christopher Potts. Sentiwordnet, 2021.

[52] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[53] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis \& insights from training gopher. 2021.

[54] Colin Raffel, Noam Shazeer, et al. Exploring the Limits of Transfer Learning with a Unified Text-toText Transformer. ArXiv, abs/1910.10683, 2019.

[55] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1-16. IEEE, 2020.

[56] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking the gpu memory wall for extreme scale deep learning. arXiv preprint arXiv:2104.07857, 2021.

[57] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pages 3505-3506, 2020 .

[58] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In AAAI, 2020.

[59] Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang A. Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M SAIFUL BARI, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, T. G. Owe Bers, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. ArXiv, abs/2110.08207, 2021.

[60] Timo Schick, Sahana Udupa, and Hinrich Schütze. Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp. arXiv preprint arXiv:2103.00453, 2021.

[61] Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ArXiv, abs/1701.06538, 2017.

[62] Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. The woman worked as a babysitter: On biases in language generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3407-3412, Hong Kong, China, November 2019. Association for Computational Linguistics.

[63] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. CoRR, abs/1909.08053, 2019 .

[64] Robyn Speer. ftfy. Zenodo, 2019. Version 5.5.

[65] DeepSpeed Team. DeepSpeed. https://github.com/microsoft/DeepSpeed, 2021.

[66] Trieu H. Trinh and Quoc V. Le. A simple method for commonsense reasoning. CoRR, abs/1806.02847, 2018.

[67] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017.

[68] Boxin Wang, Shuohang Wang, Yu Cheng, Zhe Gan, R. Jia, Bo Li, and Jingjing Liu. Infobert: Improving robustness of language models from an information theoretic perspective. ArXiv, abs/2010.02329, 2021.

[69] Zirui Wang, Adams Wei Yu, Orhan Firat, and Yuan Cao. Towards zero-label language learning. ArXiv, abs/2109.09193, 2021.

[70] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. ArXiv, abs/2109.01652, 2021.

[71] Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geoffrey Irving, and Iason Gabriel. Ethical and social risks of harm from language models. ArXiv, abs/2112.04359, 2021.

[72] Johannes Welbl, Amelia Glaese, Jonathan Uesato, Sumanth Dathathri, John Mellor, Lisa Anne Hendricks, Kirsty Anderson, Pushmeet Kohli, Ben Coppin, and Po-Sen Huang. Challenges in detoxifying language models. arXiv preprint arXiv:2109.07445, 2021.

[73] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112-1122. Association for Computational Linguistics, 2018.

[74] Shaohua Wu, Xudong Zhao, Tong Yu, Rongguo Zhang, Chong Shen, Hongli Liu, Feng Li, Hong Zhu, Jiangang Luo, Liang Xu, and Xuanwei Zhang. Yuan 1.0: Large-scale pre-trained language model in zero-shot and few-shot learning. ArXiv, abs/2110.04725, 2021.

[75] Dani Yogatama, Cyprien de Masson d'Autume, Jerome T. Connor, Tomás Kociský, Mike Chrzanowski, Lingpeng Kong, Angeliki Lazaridou, Wang Ling, Lei Yu, Chris Dyer, and Phil Blunsom. Learning and evaluating general linguistic intelligence. CoRR, abs/1901.11373, 2019.

[76] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In $A C L, 2019$.

[77] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. Defending against neural fake news. CoRR, abs/1905.12616, 2019.

[78] Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, Chen Li, Ziyan Gong, Yifan Yao, Xinjing Huang, Jun Wang, Jianfeng Yu, Qilong Guo, Yue Yu, Yan Zhang, Jin Wang, Heng Tao, Dasen Yan, Zexuan Yi, Fang Peng, Fan Jiang, Han Zhang, Lingfeng Deng, Yehong Zhang, Zhengping Lin, Chao Zhang, Shaojie Zhang, Mingyue Guo, Shanzhi Gu, Gaojun Fan, Yaowei Wang, Xue Jin, Qun Liu, and Yonghong Tian. Pangu$\alpha$ : Large-scale autoregressive pretrained chinese language models with auto-parallel computation. ArXiv, abs/2104.12369, 2021.

\begin{tabular}{lcllr}
\hline \multicolumn{1}{c}{ Subcase } & Heuristic & \multicolumn{1}{c}{ Premise } & \multicolumn{1}{c}{ Hypothesis } \\
\hline Conjunctions & Lexical Overlap & The secretaries saw the scientists and the actors. & The secretaries saw the actors. & $\mathrm{E}$ \\
Subject-object swap & Lexical Overlap & The senators mentioned the artist. & The artist mentioned the senators. & $\mathrm{N}$ \\
Untangling relative clauses & Lexical Overlap & The athlete who the judges saw called the manager. & The judges saw the athlete. & $\mathrm{E}$ \\
Passives & Lexical Overlap & The senators were helped by the managers. & The senators helped the managers. & $\mathrm{N}$ \\
Understood argument & Subsequence & The author read the book. & The author read. & $\mathrm{E}$ \\
PP on subject & Subsequence & The senator near the lawyer danced. & The lawyer danced. & $\mathrm{N}$ \\
Relative clause on subject & Subsequence & The secretary that admired the senator saw the actor. & The senator saw the actor. & $\mathrm{N}$ \\
NP/S & Subsequence & The managers heard the secretary resigned. & The managers heard the secretary. & $\mathrm{N}$ \\
Embedded under verb & Constituent & The president remembered that the actors performed. & The actors performed. & $\mathrm{E}$ \\
Embedded under preposition & Constituent & Because the banker ran, the doctors saw the professors. & The banker ran. & $\mathrm{E}$ \\
Outside embedded clause & Constituent & Unless the authors saw the students, the doctors resigned. & The doctors resigned. & $\mathrm{N}$ \\
Outside embedded clause & Constituent & Although the secretaries slept, the judges danced. & The judges danced. & E \\
\hline
\end{tabular}

Table 11: Entailment (E) and Non-entailment (N) examples from the HANS dataset

\section*{Supplementary Materials}

\section*{A Analysis Results on HANS Dataset}

\section*{A. 1 Structure and examples}

The Heuristic Analysis for NLI Systems (HANS) [40] dataset is an NLI dataset designed to check the reliance of models on three superficial syntactic heuristics: the lexical overlap heuristic, where premise and hypothesis share many common words; the subsequence heuristic, where the hypothesis is a sequence of words which exists verbatim in the premise; and the constituent heuristic, where a hypothesis is a sequence of words which forms a constituent of the parse tree of the hypothesis (e.g. a subordinate clause with a modifier). Constituent cases are a subset of subsequence cases, which are in turn a subset of lexical overlap cases.

For each heuristic, 5 templates (called "subcases") are designed to generate examples which support the heuristic (i.e. the premise entails the hypothesis), and 5 templates which contradict it (i.e. the premise does not entail the hypothesis), yielding 10 subcases per heuristic, and a total of 30 subcases overall. Each subcase can be seen as testing a specific grammatical/syntactic structure of linguistic interest. The vocabulary used to populate the templates is basic, examples are checked for plausibility using rules (e.g. nouns used as subjects or objects should be plausible for a given verb), and verbs are guaranteed to occur multiple times in datasets such as MNLI in their examined roles. Characteristic examples of the HANS dataset can be found in Table 11 .

Using each template, 1000 examples are generated, thus compiling a test set balanced with respect to the two classes ("entailment" / "non-entailment") of a total of 30000 examples; likewise, 30000 examples are generated for the training set. In our experiments, we evaluate models on all test set examples, drawing examples from the training set when constructing the few-shot prompts.

\section*{A. 2 Performance per subcase}

For each subcase, we show the accuracy of MT-NLG pretrained on 270 billion tokens, when including 32 examples in the shot. To counteract existing class prediction biases, prediction distributions were normalized by shifting their means.

Overall, we see evidence that the model at least partially relies on heuristics: for non-entailment, performance is almost perfect on lexical overlap cases, which are the easiest for the model to escape (the premisehypothesis superficial similarity is smaller, and thus it is not as strongly inclined to infer entailment). However, the model finds it more challenging to ignore superficial similarity and infer non-entailment in case of a verbatim presence of the hypothesis as a subsequence in the premise. Reversely, it is much easier for the model to correctly infer entailment in case of shared subsequences, rather than in the presence of mere lexical overlap, and thus accuracy for lexical overlap entailment subcases is lower.

Nevertheless, we also observe clear indications that the model, despite being only trained through autoregressive language modeling, is able to learn linguistic rules such as the role and function of passive voice, of the order of subject and object, of relative clauses, or of verbs that can be either transitive or intransitive, and it systematically takes into account the respective syntactic structures for inference, successfully escaping misleading superficial textual similarity. In terms of "understanding" the nuance of vocabulary, besides straight-forward cases, such as that "Without a doubt the managers advised the lawyers" entails that "The managers advised the lawyers", while the adverbs "supposedly" or "probably" reduce certainty, it is also capable of distinguishing the difference that the verb makes with respect to the veracity of the hypothesis, in cases such as: "The professors claimed / thought that the scientist advised the tourist $\rightarrow$ The scientist advised the tourist", as opposed to "The professors forgot / knew that the scientist advised the tourist".

The cases which proved most problematic for the model are often also confusing to humans, for example garden path sentences with temporary ambiguity [16] such as: "The professors heard the artist performed $\rightarrow$ The professors heard the artist", or past participle constructions in which relative pronouns are omitted, e.g. "The banker paid in the museum believed the artists $\rightarrow$ The banker paid in the museum", where "who was" is omitted before "paid". However, contrary to expectations, the model could only less than half of the time successfully parse conjunctions to infer entailment, e.g. "The secretary and the lawyers called the president $\rightarrow$ The secretary called the president", or "The artist admired the professors and the manager $\rightarrow$ The artist admired the manager". This surprising finding shows that our human intuition regarding what constitutes an easy or challenging task for a language model, and by extension, what kind of behaviors reveal mastery of natural language understanding, may be limited. Based on our findings about reliance on heuristics, inherent inference biases, as well as other factors influencing "in-context learning", we believe that the field of evaluating "natural language understanding" in generative language models, and further elucidating how it differs from the human equivalent, will be an exciting area of future research.

\section*{B Qualitative Examples of MT-NLG Text Generation}

The section below contains examples generated by MT-NLG in various contexts. Compared to benchmark evaluations, the scenarios below focuses more on open-ended, creative or logical generation tasks. We show these as qualitative examples.

\begin{tabular}{cccc}
\hline Subcase & Heuristic & Class & Accuracy \\
\hline Conjunction & Lexical Overlap & $\mathrm{N}$ & 0.993 \\
Preposition & Lexical Overlap & $\mathrm{N}$ & 0.985 \\
Adjective & Subsequence & $\mathrm{E}$ & 0.981 \\
Relative clause & Lexical Overlap & $\mathrm{N}$ & 0.977 \\
Subject/object swap & Lexical Overlap & $\mathrm{N}$ & 0.975 \\
Passive & Lexical Overlap & $\mathrm{N}$ & 0.947 \\
Adverb & Constituent & $\mathrm{E}$ & 0.928 \\
Understood object & Subsequence & $\mathrm{E}$ & 0.921 \\
PP on object & Subsequence & $\mathrm{E}$ & 0.910 \\
Relative clause on object & Subsequence & $\mathrm{E}$ & 0.841 \\
Embedded under verb & Constituent & $\mathrm{E}$ & 0.819 \\
Embedded under if & Constituent & $\mathrm{N}$ & 0.806 \\
Relative clause on subject & Subsequence & $\mathrm{N}$ & 0.798 \\
Embedded under verb & Constituent & $\mathrm{N}$ & 0.795 \\
Embedded under since & Constituent & $\mathrm{E}$ & 0.788 \\
NP/Z & Subsequence & $\mathrm{N}$ & 0.769 \\
Passive & Lexical Overlap & $\mathrm{E}$ & 0.746 \\
PP on subject & Subsequence & $\mathrm{N}$ & 0.745 \\
Conjunction & Constituent & $\mathrm{E}$ & 0.724 \\
Around relative clause & Lexical Overlap & $\mathrm{E}$ & 0.710 \\
Adverb & Constituent & $\mathrm{N}$ & 0.702 \\
Around prepositional phrase & Lexical Overlap & $\mathrm{E}$ & 0.697 \\
After if clause & Constituent & $\mathrm{N}$ & 0.651 \\
Relative clause & Lexical Overlap & $\mathrm{E}$ & 0.582 \\
Disjunction & Constituent & $\mathrm{N}$ & 0.559 \\
After since clause & Constituent & $\mathrm{E}$ & 0.548 \\
Conjunction & Subsequence & $\mathrm{E}$ & 0.546 \\
Past participle & Subsequence & $\mathrm{N}$ & 0.471 \\
Conjunction & Lexical Overlap & $\mathrm{E}$ & 0.456 \\
NP/S & Subsequence & $\mathrm{N}$ & 0.262 \\
\hline
\end{tabular}

Table 12: Accuracy of MT-NLG (trained on 270 billion tokens, using 32 -shot prompts) per HANS subcase

```

Context -> Here is the riddle:
Sometimes it flows like peanut butter
Other times it's hard as a rock
It has bones made out of metal
But not a single joint
As inanimate as it is
It can also take many shapes
As strong as it is
It can be defeated by table salt
The correct answer is
Model Answer -> Concrete
Sometimes it flows like peanut butter
Concrete can be poured and will flow like a liquid.
Other times it's hard as a rock
Concrete can also be used to create solid structures.
It has bones made out of metal
Reinforcing steel bars can be used to strengthen concrete.
But not a single joint
Concrete doesn't have any joints.
As inanimate as it is
Concrete is not alive.
It can also take many shapes
Concrete can be moulded into any shape.
As strong as it is
Concrete is very strong.
It can be defeated by table salt
Concrete can be dissolved by salt water.

```

Table 13: Samples from MT-NLG for solving riddles. MT-NLG has a strong tendency to generate interpretations of the riddles after giving the answer.

\begin{aligned} & \hline Context $\rightarrow \quad$ Here is a riddle: \\ & All of us have one, but few get to choose \\ & If you don't know mine, you are not my friend \\ & When it is called, our attention is drawn \\ & Even if we are gone, they are still around \\ & I think the answer is \\ & Model Answer $1 \rightarrow$ Name \\ & All of us have one, but few get to choose \\ & Everyone has a name, but few get to choose what their name is. \\ & If you don't know mine, you are not my friend \\ & If you don't know someone's name, you can't be friends with them. \\ & When it is called, our attention is drawn \\ & If someone calls your name, you will pay attention to them. \\ & Even if we are gone, they are still around \\ & If you are gone, your name will still be around. \\ & Model Answer $2 \rightarrow$ A phone number \\ & All of us have one, but few get to choose \\ & Everyone has a phone number, but you don't choose it \\ & If you don't know mine, you are not my friend \\ & You can't be friends without knowing each other's phone numbers \\ & When it is called, our attention is drawn \\ & When you get a phone call, you are drawn to the phone \\ & Even if we are gone, they are still around \\ & Phone numbers are still around even when we aren't there \\ & \hline\end{aligned}

Table 14: Samples from MT-NLG for answering riddles with alternative answers. MT-NLG generates different interpretations of the same riddle based on the answer it gave.

\begin{tabular}{|c|c|}
\hline Few-shot Context $\rightarrow$ & ![](https://cdn.mathpix.com/cropped/2024_06_04_23462c979d2b9a3002cfg-40.jpg?height=1560\&width=1291\&top_left_y=257\&top_left_x=575) \\
\hline Model Answer $1 \rightarrow$ & \begin{tabular}{l} 
Question: Including Okaloosa \& Escambia there are 16 counties in the Florida region \\
with this shapely name \\
Answer: \\
What is the Panhandle?
\end{tabular} \\
\hline Question $2 \rightarrow$ & \begin{tabular}{l} 
Question: The word means bearskin but these Viking warriors were said to fight almost \\
naked in a bloodthirsty fury \\
Answer: \\
Who are Berserkers?
\end{tabular} \\
\hline Question $3 \rightarrow$ & \begin{tabular}{l} 
Question: Ooh I wanna take ya but the Beach Boys completely made up this triple-rhyme \\
title island from a 1988 song \\
Answer: \\
What is Kokomo?
\end{tabular} \\
\hline
\end{tabular}

Table 15: Samples from MT-NLG for answering Jeopardy! reverse trivia questions. Few-shot context is prepended to each question.

\section*{Context $\rightarrow$}

```

def update_dict(orig, update):
"""Deep update of a dictionary
For each entry (k, v) in update such that both orig [k] and v are
dictionaries, orig[k] is recursively updated to v.
For all other entries (k, v), orig[k] is set to v.
"\#"

```

Model Output $\rightarrow$

for $\mathrm{k}, \mathrm{v}$ in update.items():

if isinstance(orig[k], dict) and isinstance(v, dict): update_dict(orig[k], v)

else:

orig $[\mathrm{k}]=\mathrm{v}$

\section*{Context $\rightarrow$}

def find_3_or_7_divisible_fibs(n):

"""Find all the Fibonacci numbers below n that are divisible by 3 or divisible by 7 .

"##

Model Output $\rightarrow$

```

return [f for $\mathrm{f}$ in find_fibs(n) if $\mathrm{f} \% 3==0$ or $\mathrm{f} \% 7==0$ ]
def find_fibs(n):
"""Find all Fibonacci numbers below n.
"" "
$\mathrm{a}, \mathrm{b}=0,1$
while a $<\mathrm{n}$
yield a
$\mathrm{a}, \mathrm{b}=\mathrm{b}, \mathrm{a}+\mathrm{b}$

```

Table 16: Samples from MT-NLG for code generation.

\begin{tabular}{ll}
\hline Sample 1 & Sample 2 \\
\hline Prompt: & Prompt: \\
Q: $11 \# 2=?$ & Q: $11 \$ \wedge \% 2=?$ \\
A: 5.5 & A: 22 \\
Q: $18 \# 9=?$ & Q: $3 \$ \sim \% 9=?$ \\
A: 2 & A: 27 \\
Q: $35 \# 4=?$ & Q: $12 \$ \sim \% 4=?$ \\
A: 8.75 & A: 48 \\
Q: $27 \# 9=?$ & Q: $7 \$ \vee \% 7=?$ \\
A: 3 & A: 49 \\
Q: $75 \# 6=?$ & Q: $8 \$ \wedge \% 5=?$ \\
A: & A: \\
\hline MT-NLG: & MT-NLG: \\
12.5 & 40 \\
Q: $45 \# 5=?$ & Q: $9 \$^{\wedge} \% 6=?$ \\
A: 9 & A: 54 \\
Q: $18 \# 6=?$ & Q: $10 \$ \sim \% 5=?$ \\
A: 3 & A: 50 \\
\hline
\end{tabular}

Table 17: Samples from MT-NLG on inferring mathematical operations.

\begin{tabular}{|c|c|c|c|}
\hline accountant & acquaintance & actor & actress \\
\hline adjunct professor & administrator & adventurer & advocate \\
\hline aide & alderman & alter ego & ambassador \\
\hline analyst & anthropologist & archaeologist & archbishop \\
\hline architect & artist & artiste & assassin \\
\hline assistant professor & associate dean & associate professor & astronaut \\
\hline astronomer & athlete & athletic director & attorney \\
\hline author & baker & ballerina & ballplayer \\
\hline banker & barber & baron & barrister \\
\hline bartender & biologist & bishop & bodyguard \\
\hline bookkeeper & boss & boxer & broadcaster \\
\hline broker & bureaucrat & businessman & businesswoman \\
\hline butcher & butler & cab driver & cabbie \\
\hline cameraman & campaigner & captain & cardiologist \\
\hline caretaker & carpenter & cartoonist & cellist \\
\hline chancellor & chaplain & character & chef \\
\hline chemist & chair & choreographer & cinematographer \\
\hline citizen & civil servant & cleric & clerk \\
\hline coach & collector & colonel & columnist \\
\hline comedian & comic & commander & commentator \\
\hline commissioner & composer & conductor & confesses \\
\hline congressman & constable & consultant & cop \\
\hline correspondent & councilman & councilor & counselor \\
\hline critic & crooner & crusader & curator \\
\hline custodian & dad & dancer & dean \\
\hline dentist & deputy & dermatologist & detective \\
\hline diplomat & director & disc jockey & doctor \\
\hline doctoral student & drug addict & drummer & economics professor \\
\hline economist & editor & educator & electrician \\
\hline employee & entertainer & entrepreneur & environmentalist \\
\hline envoy & epidemiologist & evangelist & executive \\
\hline farmer & fashion designer & fighter pilot & filmmaker \\
\hline financier & firebrand & firefighter & fireman \\
\hline fisherman & footballer & foreman & freelance writer \\
\hline gangster & gardener & geologist & goalkeeper \\
\hline graphic designer & guidance counselor & guitarist & hairdresser \\
\hline handyman & headmaster & historian & hitman \\
\hline homemaker & hooker & housekeeper & housewife \\
\hline illustrator & industrialist & infielder & inspector \\
\hline instructor & interior designer & inventor & investigator \\
\hline investment banker & janitor & jeweler & journalist \\
\hline judge & jurist & laborer & landlord \\
\hline lawmaker & lawyer & lecturer & legislator \\
\hline librarian & lieutenant & lifeguard & lyricist \\
\hline maestro & magician & magistrate & maid \\
\hline major leaguer & manager & marksman & marshal \\
\hline mathematician & mechanic & mediator & medic \\
\hline midfielder & minister & missionary & mobster \\
\hline monk & musician & nanny & narrator \\
\hline naturalist & negotiator & neurologist & neurosurgeon \\
\hline novelist & nun & nurse & observer \\
\hline officer & organist & painter & paralegal \\
\hline
\end{tabular}

Table 18: List of occupation lexicons used for association test of gender and profession

\begin{tabular}{llll}
\hline parishioner & parliamentarian & pastor & pathologist \\
patrolman & pediatrician & performer & pharmacist \\
philanthropist & philosopher & photographer & photojournalist \\
physician & physicist & pianist & planner \\
plastic surgeon & playwright & plumber & poet \\
policeman & politician & pollster & preacher \\
president & priest & principal & prisoner \\
professor & professor emeritus & programmer & promoter \\
proprietor & prosecutor & protagonist & protege \\
protester & provost & psychiatrist & psychologist \\
publicist & pundit & rabbi & radiologist \\
ranger & realtor & receptionist & registered nurse \\
researcher & restaurateur & sailor & saint \\
salesman & saxophonist & scholar & scientist \\
screenwriter & sculptor & secretary & senator \\
sergeant & servant & serviceman & sheriff deputy \\
shopkeeper & singer & songwriter & skipper \\
socialite & sociologist & soft spoken & soldier \\
solicitor & solicitor general & soloist & sportsman \\
sportswriter & statesman & steward & stockbroker \\
strategist & student & stylist & substitute \\
superintendent & supervisor & surgeon & surveyor \\
swimmer & taxi driver & teacher & technician \\
teenager & therapist & trader & treasurer \\
trooper & trucker & trumpeter & tutor \\
tycoon & undersecretary & understudy & valedictorian \\
vice chancellor & violinist & vocalist & waiter \\
waitress & warden & warrior & welder \\
worker & wrestler & writer & \\
\hline
\end{tabular}

Table 19: List of occupation lexicons used for association test of gender and profession```


[^0]:    *Affiliated with Brown University. Work done during internship at Microsoft.

[^1]:    ${ }^{1}$ https://github.com/leogao2/commoncrawl_downloader

    2https://pypi.org/project/pycld2/

    3 https://pypi.org/project/jusText/

[^2]:    ${ }^{4}$ https://scikit-learn.org/stable/

    http://ekzhu.com/datasketch/documentation.html

[^3]:    ${ }^{6}$ Gopher uses a different prompt format compared to GPT-3 and MT-NLG.

[^4]:    ${ }^{7}$ Number taken from original paper.

[^5]:    ${ }^{8}$ For all gender related studies we have used only the binary male and female identifiers. We note that gender may be considered along a spectrum [13] but use binary identifiers due to lack of templates in prior work.

</end of paper 1>


<paper 2>
# Taming Server Memory TCO with Multiple Software-Defined Compressed Tiers 

Sandeep Kumar, Aravinda Prasad, and Sreenivas Subramoney<br>Processor Architecture Research Lab, Intel Labs


#### Abstract

Memory accounts for $33-50 \%$ of the total cost of ownership (TCO) in modern data centers. We propose TierScape to tame memory TCO through the novel creation and judicious management of multiple software-defined compressed memory tiers.

As opposed to the state-of-the-art solutions that employ a 2-Tier solution, a single compressed tier along with DRAM, we define multiple compressed tiers implemented through a combination of different compression algorithms, memory allocators for compressed objects, and backing media to store compressed objects. These compressed memory tiers represent distinct points in the access latency, data compressibility, and unit memory usage cost spectrum, allowing rich and flexible trade-offs between memory TCO savings and application performance impact. A key advantage with TierScape is that it enables aggressive memory TCO saving opportunities by placing warm data in low latency compressed tiers with a reasonable performance impact while simultaneously placing cold data in the best memory TCO saving tiers. We believe TierScape represents an important server system configuration and optimization capability to achieve the best SLA-aware performance per dollar for applications hosted in production data center environments.


TierScape presents a comprehensive and rigorous analytical cost model for performance and TCO trade-off based on continuous monitoring of the application's data access profile. Guided by this model, TierScape takes informed actions to dynamically manage the placement and migration of application data across multiple software-defined compressed tiers. On real-world benchmarks, TierScape increases memory TCO savings by $22 \%-40 \%$ percentage points while maintaining performance parity or improves performance by $2 \%-10 \%$ percentage points while maintaining memory TCO parity compared to state-of-the-art 2-Tier solutions.
![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-01.jpg?height=370&width=836&top_left_y=888&top_left_x=1102)

Figure 1: Memcached on a 2-Tier system (DRAM + a single compressed tier): conservatively placing $20 \%$ cold data in the compressed tier limits the memory TCO savings to $11 \%$ with a $9.5 \%$ slowdown. Placing around $50 \%$ of data (including cold and some warm data) in the compressed tier results in $16 \%$ memory TCO savings and $13.5 \%$ slowdown. An aggressive approach that places around $80 \%$ of data (including cold and most of the warm data) in the compressed tier results in $32 \%$ memory TCO savings and $20 \%$ slowdown.

## 1 Introduction

Memory accounts for $33-50 \%$ of the total cost of ownership (TCO) in modern data centers [2,50]. This cost is expected to escalate further in order to serve the growing data demands of modern AI/ML applications whose working set already breaks the terabyte barrier [23, 45, 48], thus making it imperative to tame the data center's memory TCO.

The current state-of-the-art software-based solutions compress and place data in a compressed second-tier memory such as zswap in Linux to reduce memory TCO [36] (we refer to them as 2-Tier systems). Placing data in a compressed memory tier reduces the memory footprint of applications. As a result, systems can be provisioned with less memory, thus reducing the memory TCO in a data center. However, memory TCO savings with compressed tiers is not free as the data stored in such a tier must be decompressed before an application can access it, resulting in a performance penalty. Hence,
to trade-off memory TCO savings and performance penalties, data center providers only place infrequently accessed or cold data in the compressed tier [36].

We highlight the following critical observations and key limitations of the state-of-the-art 2-Tier systems. (1) On an average, $20-30 \%$ of the data are cold in production systems $[26,36,39,40,50]$ and hence placing only cold data in second-tier compressed memory has limited memory TCO saving potential. (2) Aggressively placing more data pages in a compressed second tier can increase memory TCO savings but results in a significantly higher and unacceptable performance penalty [26] (see Figure 1). 3 Given the high cost of accessing data from a compressed tier, existing 2-Tier solutions do not compress warm pages, which accounts for $50-60 \%[40,50]$ of the data pages, thus leaving significant memory TCO reduction opportunities on the table.

In this paper, we seek to exploit memory TCO-saving opportunities beyond the cold data pages with an acceptable performance penalty. We propose TierScape, a novel solution with multiple software-defined compressed memory tiers (which we refer to as $\mathbf{N}$-Tier systems) that dynamically manages placement and migration of data across compressed tiers to strike the best balance between memory TCO savings and application performance. The compressed tiers can be a combination of different compression algorithms (e.g., lzo-rle, deflate, lz4), memory allocators for compressed objects (e.g., zsmalloc, zbud, z3fold), and backing media to store compressed objects (e.g., DRAM, non-volatile main memory [3], CXL-attached memory [25, 27, 41]). TierScape's compressed tiers are distinct in access latency, unit memory usage cost, and capacity savings (compression ratio), enabling a holistic and flexible option space for hot/warm/cold data placement to balance memory TCO savings and application performance. TierScape thus compares very favorably to the rigid and restricted data placement and optimization space available in today's state-of-the-art 2-Tier systems.

TierScape, through its multiple compressed tiers, enables aggressive memory TCO saving opportunities by placing warm data pages in low-latency compressed tiers with reasonable performance impact while simultaneously placing cold data in the best memory TCO saving tiers. TierScape applies different placement and migration policies for war$\mathrm{m} /$ cold data based on the application's dynamic data access profile. For example, in our conservative model, which we refer to as the waterfall model (\$5.1), warm pages are initially placed in a low latency tier and eventually moved or aged to tiers with better TCO savings, thus progressively achieving better memory TCO savings.

TierScape introduces an advanced analytical model (\$5.2) that periodically recommends scattering pages across multiple compressed tiers based on the access profile of the pages. The recommendations to move specific groups of pages to specific tiers are based on the usage patterns of the application's different memory regions, the relative costs of page access in different tiers, and the real-time memory TCO cost per tier incurred by the application. TierScape's multi-objective global optimization across application performance and memory TCO enables superior placement and control of hot/warm/cold page sets and calibrated maximization of performanceper-dollar metrics critical for data center operators.

The key contributions of the paper are as follows:

- To the best of our knowledge, we are the first to propose and demonstrate memory TCO savings for warm data with an acceptable performance impact.
- Highlight the limitations with the state-of-the-art 2-Tier systems in saving memory TCO. Specifically, the limited TCO savings with cold data and its incapability to tap TCO saving opportunities for warm data with a reasonable performance penalty.
- Demonstrate the benefits of defining multiple compressed memory tiers in the software that offer a rich and flexible trade-off between memory TCO savings and application performance impact.
- Judiciously manage page placement across tiers with waterfall and analytical models.


## 2 Background

### 2.1 Memory compression

Linux kernel's zswap $[14,33,36]$ supports memory compression where pages are compressed and placed in a compressed pool. Whenever a compressed page is accessed, zswap decompresses the data from the compressed pool and places it in the main memory [15]. The Linux implementation of zswap has two key components: (i) the compression algorithm and (ii) the pool manager.

Compression algorithms. The Linux kernel supports different compression algorithms such as deflate, lz4, lzo, and lzo-rle that differ in algorithmic complexity and the ratio of data compression achieved. However, zswap is flexible enough to add new compression algorithms as required. The deflate compression algorithm offers the best compression ratio but consumes comparatively higher CPU cycles to compress and decompress the data $[5,6,31]$. On the other hand, $\mathrm{lz} 4$ is a fast compression algorithm but has relatively low data compressibility [6]. lzo (and its evolved variant lzo-rle) offers a balance between compression ratio and decompression overheads $[5,7,13]$. In addition, many compression algorithms such as lz4 have a "level of effort" parameter that can trade compression speed and compression ratio.

Pool managers: A pool manager manages how compressed pages are stored in zswap. A pool is created in physical memory to store compressed data pages by allocating pages using the buddy allocator [1]. The pool dynamically expands
to store more compressed objects by allocating more pages or contracts as required. To manage compressed objects inside the pool a custom memory allocator is used. Linux supports three pool memory allocators: zsmalloc, zbud, and z3fold [12, 14, 24].

zsmalloc employs a complex memory management technique that densely packs compressed objects in the pool and thus has the best space efficiency. However, it has relatively high memory management overheads [24]. zbud is a simple and fast pool management technique that stores a maximum of two compressed objects in a $4 \mathrm{~KB}$ region. Due to this, the total space saved with zbud cannot be more than $50 \%$ [14]. But, because of its simple object management, zbud has a relatively low memory management overhead. z3fold is similar to zbud, but instead of two compressed objects, it can store three compressed objects in a $4 \mathrm{~KB}$ region [12].

Linux allows users to pick a compression algorithm and a pool manager to manage zswap. However, Linux supports only one active zswap pool at a given time [14]. If a different compression algorithm or a pool manager is dynamically configured, the kernel creates a new pool and uses it to place compressed pages. The old pool is kept around till all data present in it is either faulted back to memory or invalidated [14].

## 3 Motivation

Missed opportunities for warm pages. Data center operators report that around $10-20 \%$ of the data are hot and $20-30 \%$ of data are cold $[26,36,39,40,50]$. This implies that around $50-70 \%$ of the data pages are neither hot nor cold but can be considered as warm pages. These warm pages can be (i) pages with relatively fewer accesses than hot pages or (ii) pages that are transitioning from hot to cold as hot data does not become cold instantaneously but rather follows a gradual process where it ages itself to cold. However, a cold or warm page can instantaneously become hot, depending on the access pattern of the application. Existing 2-Tier solutions do not consider exploiting warm pages for compression, thus missing significant memory TCO-saving opportunities.

Drawbacks with aggressive data placement. A naive approach to aggressively place more data in the compressed second-tier memory to increase memory TCO savings results in a significantly higher and unacceptable performance penalty (Figure 1). However, replacing a highly compressible tier with a low compression, low access latency tier (due to low decompression latency) can enable aggressive data placement in the compressed tier. However, it severely impacts the memory TCO savings due to low compression ratio.

Employing page prefetching [36] that prefetches or decompresses pages from compressed memory can mitigate high-performance penalty to the extent of prefetching accuracy. However, pages that the prefetcher fails to identify for prefetching still incur high access latency when accessed, and

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-03.jpg?height=323&width=835&top_left_y=234&top_left_x=1098)

(a) A 2-tier system

(b) TierScape system

Figure 2: Data placement options in 2-Tier and N-Tier systems

incorrectly prefetched pages results in decreased memory TCO savings. Nevertheless, prefetching can be additionally employed in an $\mathrm{N}$-Tier memory context and we note it as a future work of interest for the systems community.

Limited placement choices. To reiterate the central observation here: the key limitation with 2-Tier systems is the binary decision options they face for data placement - either in DRAM or in the compressed second tier (Figure 2). This severely limits the flexibility and choices for page placement towards a better balance between application performance and memory TCO.

Summary. To conclude, the current 2-Tier approaches fail to exploit the temperature gradient that naturally manifests across a large population of the application's pages over time to simultaneously achieve better TCO and performance.

## 4 Concept

The core concept behind our proposal is to define multiple compressed tiers in the software. Each compressed tier is defined through a combination of (i) compression algorithms, (ii) memory allocator for the compressed pool, and (iii) different backing media - each providing a different access latency and memory cost per byte, as we discuss below.

Compression algorithms. Compression algorithms with low compression ratio and, consequently, a low decompression latency are suitable for low latency tiers, but they provide only marginal memory TCO savings. Whereas other compression algorithms, such as deflate with high compression ratio and, consequently, high decompression latency, are suitable for high memory TCO savings tiers but with significantly high memory access latency.

Pool allocators. As zsmalloc densely packs compressed objects in the pool, it is suitable for high memory TCO saving tiers, but it has high memory management overheads, thus impacting the decompression latency. zbud, with its simple and fast pool management, is suitable for low latency tiers but is less space efficient, resulting in tiers with low memory TCO savings.

Physical media. The access latency of the storage medium where the compressed pages are stored is crucial for the per-

Table 1: Different options available in Linux for setting up a compressed tier

| Compression algorithm | Allocators | Backing media |
| :---: | :---: | :---: |
| Deflate, LZO, LZO-RLE, | zsmalloc, | DRAM, CXL-attached |
| LZ4, Zstd, 842, LZ4HC | zbud, z3fold | memory, NVMM |

formance of the tier. Storing compressed pages on DRAM offers the lowest possible media access latency [52] and hence suits low latency tiers. But doing so also reduces the overall memory TCO savings potential. Using cheaper and denser memory, such as NVMMs or CXL-attached memory, to store compressed pages increases memory TCO savings but adds to decompression latency, rendering them attractive for use as high memory TCO saving tiers.

The key idea for enabling aggressive memory TCO savings is to use tiers with low latency for warm pages that can save memory TCO at moderate performance overheads. Meanwhile, tiers with high compression ratios and high access latency are used for cold pages.

### 4.1 Characterization of compressed tiers

We start by comparing the access latencies and memory TCO benefits of compressed tiers with different configurations in Linux. The Linux kernel offers only two configuration parameters for a zswap compressed tier (compression algorithm and pool manager) but does not offer any control over where the pool is allocated, i.e., the kernel cannot be instructed to allocate the pool on DRAM or NVMM. We modify zswap to add a configuration parameter - backing media, that specifies from which hardware media the pages for a particular compressed pool are to be allocated. This allows us to construct tiers specifying backing media.

The latency of decompressing a page from zswap is primarily dominated by the compression algorithm, pool manager, and backing media. With the available choices in Linux (as shown in Table 1), we can create a total of 63 different zswap compressed tiers $\left(C_{1}^{7} * C_{1}^{3} * C_{1}^{3}\right)$. In addition, the compressibility ratio and decompression latency of a given tier also depend on the compressibility of the input data.

In order to allow for multiple operating points in the space of access latency and memory TCO savings, we define 12 tiers configured based on widely used compression algorithms and pool managers. We initialize $10 \mathrm{~GB}$ of data in memory, compress and place them in a compressed memory tier and then access them. We repeat this experiment for all 12 tiers. To characterize with different input data, we use two data sets from the Silesia corpus [11], nci and dickens, with the former being more compressible [22]. We measure the access latency and compression ratio.
![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-04.jpg?height=956&width=702&top_left_y=232&top_left_x=1167)

(a) Access latency

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-04.jpg?height=325&width=683&top_left_y=819&top_left_x=1182)

(b) Memory TCO savings

Figure 3: Characterization results for 12 different softwaredefined compressed tiers for dicken and $n c i$ data sets. Encoding: $\mathbf{Z S}, \mathbf{Z B}$ refers to zsmalloc and zbud pool managers, respectively. L4, LO, DE refers to lz4, lzo, and deflate compression algorithms, respectively. DR, OP: refers to DRAM and Optane [3] as the backing storage media, respectively.

### 4.1.1 Access latency

Figure 3a shows the access latency for both $n c i$ and dickens data sets. Access latency with the lz4 algorithm is the fastest, followed by lzo, and lastly, deflate. As expected, the performance of zbud pool manager is also better than zsmalloc. This is because zbud employs a simple algorithm that enables faster page lookup. Finally, the access latency of DRAM-backed tiers is better than those backed by the Optane [3] due to the higher media access latency in the latter [20].

### 4.1.2 Memory TCO savings

Figure 3 b shows the normalized memory TCO savings of compressed tiers relative to uncompressed data in DRAM. Total TCO savings depend on data compressibility, compression algorithm, and backing media. The cost per gigabyte for storing data on Optane is typically $1 / 3 \sim 1 / 2$ of the cost of storing data on DRAM [43]. Hence, the memory TCO for Optane-backed tiers is lower than that of DRAM-backed tiers.

Furthermore, for tiers using the same compression algorithm and backing media the TCO savings depend on the memory allocator for the compressed pool manager. For ex-
ample, a tier using zsmalloc as its pool manager has a lower memory TCO than a tier using zbud. This is because zsmalloc can pack compressed objects more tightly. Finally, the deflate compression algorithm offers the best compression ratio.

### 4.2 Tiers selection methodology

In order to illustrate the flexibility and robustness of the TierScape proposal, we select five compressed tiers (C1, C2, C4, C7, and C12) that we define in the software.

We pick $\mathrm{C} 1$ and $\mathrm{C} 12$ as they offer the best performance configuration and best memory TCO savings configuration, respectively. Other tiers with deflate compression algorithms offer a similar performance latency without additional TCO benefits, and hence we do not select any other deflate-based tiers. We select C2 as it offers the lowest latency for an Optanebacked compressed tier. $\mathrm{C} 1$ and $\mathrm{C} 2$ use zbud and $\mathrm{lz} 4$ as their pool manager and compression algorithm - restricting the compression ratio to 2 . Hence, we select C4, which uses a fast compression method (lz4), tightly packs compressed objects (due to zsmalloc), and is stored on low-cost Optane. Finally, we select C7, which fills the gap between access latency and memory TCO savings. We use this set of tiers for our experiments to demonstrate rich and flexible placement opportunities.

## 5 Data placement in TierScape

In this section, we present two distinct data placement models that fully exploit the benefits of $\mathrm{N}$-Tier systems. Note that we develop these models to show rich and flexible data placement options to tame memory TCO. However, we believe that having multiple software-defined compressed tiers opens up a plethora of exploration opportunities for innovative data placement policies.

For ease of our discussion, we assume the system is configured with DRAM $+N$ compressed tiers. Furthermore, the tiers are ordered from low latency to high latency (and, consequently, low TCO savings to high TCO savings), i.e., Tier 1 offers the best performance but with the least memory TCO savings. In contrast, Tier $N$ offers the best memory TCO savings with high performance impact.

### 5.1 Waterfall model

A 2-Tier memory TCO saving solution uses a hotness threshold $\left(H_{t h}\right)$ to decide which pages should be pushed from DRAM to the compressed tier. As seen before, an aggressive threshold (a high value for $H_{t h}$ ) pushes more pages to the compressed tier saving additional memory TCO but at the cost of high performance penalty due to multiple high latency page faults from the compressed tier. The waterfall model extends this approach naturally to leverage multiple software-

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-05.jpg?height=263&width=490&top_left_y=237&top_left_x=1273)

Figure 4: Page placement with the N-Tier waterfall model

defined tiers available to achieve better memory TCO savings while limiting the performance penalty.

The model starts by monitoring the pages accessed by an application for a fixed duration - henceforth referred to as a profile window. As shown in Figure 4, at the end of each profile window, all the pages that have a hotness value (access count) less than the threshold $\left(H_{t h}\right)$ are moved from DRAM to low-latency tier T1. This reduces the total memory TCO upfront as some data pages have been placed in a compressed tier (albeit the tier has a low compression ratio). The advantage is that these compressed pages can be decompressed and placed in DRAM when accessed without high performance penalty as $\mathrm{T} 1$, by design, is a low latency compressed tier.

During the next profile window, some pages will be faulted back to DRAM from T1 as per the application's access pattern. Once the profile window ends, all the pages that are still in $\mathrm{T} 1$ are, in fact, getting colder as they were not accessed in the last profile window. The model moves (or waterfalls) all the data from T1 to T2. This further increases the memory TCO savings as $\mathrm{T} 2$ is better than $\mathrm{T} 1$ in memory TCO savings.

At the end of each profile window, the model waterfalls all the data in all the tiers to one tier below it (to a higher TCO saving tier), except for the last tier. However, pages that are accessed by the application are decompressed and placed in DRAM, irrespective of its tier and these pages have to start the journey again from $\mathrm{T} 1$.

Benefits:

Upfront memory TCO savings. The memory TCO savings start upfront, as all the cold and warm data can be immediately placed in low latency compressed tiers without significant performance impact.

Tolerate profiling inaccuracies. Existing profiling techniques such as PMU's [32] do not provide a $100 \%$ accurate memory access profile [46] which can result in incorrectly identifying a hot or a warm page as a cold page. The penalty for placing a hot page incorrectly classified as a cold page in a 2-Tier solution can be significant [26]. As the waterfall model initially places all pages, including incorrectly classified hot or warm pages, in low latency compressed tiers, it incurs a minimal performance penalty when they are accessed.

Gradual convergence to maximum TCO savings. Waterfall model gradually moves cold pages to better memory TCO saving tiers with each profile window. Hence, it eventually

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-06.jpg?height=295&width=572&top_left_y=243&top_left_x=316)

Figure 5: Page placement with the N-Tier analytical model.

converges to a stable phase where all the cold data pages are placed in the best memory TCO saving tier, thus maximizing the TCO savings.

## Limitations:

Cold page convergence. Cold pages (i.e., pages with 0 access count) requires $N$ profile windows (in an $N$-Tier setup) to converge to the last or best TCO saving tier. It misses the opportunity to aggressively place cold pages directly in best memory TCO saving tiers.

Limited flexibility to fine-tune page placement. Waterfall model does not offer flexibility to fine-tune page placement. A hotness threshold parameter fully controls page placement. For example, users cannot specify placement criteria or requirements to trade off memory TCO savings and performance penalties.

### 5.2 TierScape's analytical model

We propose an analytical data placement model to address the limitations of the waterfall model. Analytical model can directly distribute data ( Figure 5) to different memory tiers based on the hotness profile of the data. In addition, the model provides fine control to the users to balance the trade-off between memory TCO savings and performance penalty by exposing a user-guided tunable "knob".

As shown in Figure 6, the range of the knob is $[0,1]$. A value of 1 indicates the model is tuned for maximum performance, which results in zero memory TCO savings as all data pages are placed in DRAM. On the other hand, a value towards 0 indicates that the model is tuned to maximize TCO savings while striving to minimize performance penalty.

### 5.2.1 Data placement modeling

The analytical model is initiated with a knob value - say $\alpha$ $\in[0,1]$. The theoretical memory TCO savings achievable is the difference between $\mathrm{TCO}_{\text {max }}$ - when all the data is in DRAM and $\mathrm{TCO}_{\text {min }}-$ when all the data is in the last tier. The maximum TCO savings (or MTS) can be defined as follows:

$$
\begin{equation*}
\mathrm{MTS}=\mathrm{TCO}_{\max }-\mathrm{TCO}_{\min } \tag{1}
\end{equation*}
$$

The analytical model can be tuned to achieve TCO savings within [0, MTS ] by configuring $\alpha$.

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-06.jpg?height=236&width=746&top_left_y=240&top_left_x=1145)

Figure 6: The memory TCO is minimal when all the data is placed in a highly compressible tier, while it is maximum when all the data is in DRAM. The difference between the two is the TCO saving opportunity that is tuned with a knob in the analytical model.

At the end of each profile window, the model uses $\alpha$ and the profiled data to solve the following:

$$
\begin{align*}
& \operatorname{minimize} \quad \text { perf_ovh } h_{N T} \\
& \text { subject to } \quad \mathrm{TCO} \leq\left(\mathrm{TCO}_{\min }+\alpha * \mathrm{MTS}\right) \tag{2}
\end{align*}
$$

In order to solve Equation 2, we start by formally defining performance overhead (perf_ovhd) and the memory TCO.

### 5.2.2 Modeling performance overheads

In terms of memory accesses, an application executes optimally when all its load operations are directly from DRAM (instead of a compressed tier). Let us refer to this performance as perfopt .

Consider a scenario when a few of the application's pages are placed in a single compressed tier $T_{i}$. If an application attempts to read those pages, it will result in Fault $T_{T_{i}}$ faults, with each fault incurring $\mathrm{Lat}_{T_{i}}$ latency overheads to decompress the data. Once the pages are decompressed, the accesses are served from DRAM. Hence the performance with a compressed memory tier includes the cost of accessing memory regions from DRAM:

$$
\begin{align*}
p_{e r f}{ }^{\prime \prime} & =\text { perf }_{\text {opt }}+\text { Fault }_{T i} * \operatorname{Lat}_{T i}  \tag{3}\\
\text { perf_ovh } & =\text { perf }^{\prime \prime}-\text { perf }_{\text {opt }}  \tag{4}\\
& =\text { Fault }_{T i} * \operatorname{Lat}_{T i} \tag{5}
\end{align*}
$$

Here, perf_ovh is the performance overhead due to accessing pages in a compressed memory tier $T_{i}$ and is equal to the total time spent serving the faults from Tier $T_{i}$.

Generalizing this when $N$ compressed tiers are used, the performance overhead ( perf_ov $h_{N T}$ ) can be defined as:

$$
\begin{equation*}
p e r f \_o v h_{N T}=\sum_{y=1}^{N}\left(\text { Fault }_{T y} * \operatorname{Lat}_{T y}\right) \tag{6}
\end{equation*}
$$

Here, Fault ${ }_{T y}$ is the number of faults the application incurs from a compressed tier $T_{y}$. However, the model does not have this information while making the placement decision at the end of the current profile window, as it cannot estimate the number of future faults for the application.

In order to estimate the number of faults, the model exploits the fact that for an application with stable access patterns, the total number of faults to a data region $(r)$ in the next profiling window, if placed in a compressed tier, will be proportional to the hotness of the data region (Hot H $_{r}$ ) in the previous profiling window. Hence, Fault $T y$ is proportional to the sum of the hotness of all the data regions stored in that tier $T_{y}$ :

$$
\begin{equation*}
\text { Fault }_{T y} \propto \sum_{r=1}^{R} \operatorname{Hot}_{r} \tag{7}
\end{equation*}
$$

Hence, we use the following equation, which is in terms of page hotness from the previous profile window, to estimate the performance overhead:

$$
\begin{equation*}
\operatorname{perf}_{\text {ovh }}=\sum_{y=1}^{N}\left(\left(k_{y} \sum_{r=1}^{R} \operatorname{Hot}_{r}\right) * \operatorname{Lat}_{T y}\right) \tag{8}
\end{equation*}
$$

Here, $k_{y}$ is a constant factor. For the rest of the paper, we use $k_{y}$ as 1 .

### 5.2.3 Modeling memory TCO

The memory cost of placing data on a particular tier depends on the backing media and compressibility of data. The memory TCO is highest when all the data (measured as $4 \mathrm{~KB}$ pages, $\mathrm{P}_{\text {tot }}$ ) of an application is in DRAM and is defined as:

$$
\begin{equation*}
\mathrm{TCO}_{\max }=\mathrm{P}_{\text {tot }} * \mathrm{USD}_{\text {DRAM }} \tag{9}
\end{equation*}
$$

Where $U_{S D} D_{D A M}$ is the cost of storing a single $4 \mathrm{~K}$ page in DRAM. Similarly, the memory TCO is lowest when all the data is placed in the best memory TCO savings tier $(N)$ :

$$
\begin{equation*}
\mathrm{TCO}_{\text {min }}=\mathrm{P}_{\text {tot }} *\left(1 / C_{T_{N}}\right) * \mathrm{USD}_{T_{N}} \tag{10}
\end{equation*}
$$

Where $\mathrm{USD}_{T_{N}}$ is the cost of the media backing the compressed tier $T_{N}$. $C_{T_{N}}$ is the compressibility ratio of tier $T_{N}$ defined as:

$$
\begin{equation*}
C_{T_{N}}=\frac{\text { Original size of data on } T_{N}}{\text { Compressed size of data on } T_{N}} \tag{11}
\end{equation*}
$$

As discussed before, the compressibility of a tier depends on the compression algorithm, pool manager, and the data.

In an N-Tier system, the memory TCO can be defined as the sum of the cost to store data in DRAM and the cost to store data in compressed tiers. It can be defined as:

$$
\begin{equation*}
\mathrm{TCO}_{N T}=\mathrm{P}_{D R A M} * \mathrm{USD}_{D R A M}+\sum_{y=1}^{N}\left(\mathrm{P}_{T y} *\left(1 / \mathrm{C}_{T y}\right) * \mathrm{USD}_{T y}\right) \tag{12}
\end{equation*}
$$

Here, $\mathrm{P}_{T y}$ is the number of pages placed in Tier $T_{y}$. We use Equation 8 and Equation 12 to solve Equation 2 as an integer linear program (or ILP). The number of pages in DRAM $P_{D R A M}$ and pages in each compressed tiers $P_{T_{y}}$ are the optimization variables. The model outputs the final placement of pages that satisfy the constraints. We then place data on different compressed tiers as per the model's recommendation.

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-07.jpg?height=342&width=594&top_left_y=241&top_left_x=1210)

Figure 7: A high-level working of TierScape

### 5.2.4 Discussion

The model quickly converges to optimal data placement based on the profiled hotness of the data. Cold data are directly placed in the most optimal tier as per the constraints instead of "waterfalling" on multiple tiers. In addition, the user-guided tunable knob $(\alpha)$ enables fine-tuning memory TCO and performance penalty trade-off.

## 6 Implementation

### 6.1 Linux kernel changes

Tier's backing media: As discussed in Section 4.1, the Linux kernel configures a compressed memory tier using two parameters: the compression algorithm and the pool manager [33]. We augment the zswap subsystem to add a third parameter to specify a backing media which can be NVMM or CXLattached memory. We enhance the kernel to allocate physical memory only from these backing media when the pool is created or when the pool dynamically expands to store more compressed objects.

Multiple active compressed tiers: Linux kernel only supports a single active zswap tier. Upon creation of a new compressed tier, all new data compression requests are directed to the newly created tier. The kernel deletes the old tiers if they are empty. We modify the zswap subsystem to support multiple active compressed zswap tiers and also allow multiple compressed tiers to co-exist.

API changes: Once we setup multiple active compressed tiers, the challenge is to instruct the kernel to send a specific set of pages to a target tier based on the model recommendation. For example, the model can recommend placing a few pages in Tier T2 and a few others in T4. To ensure the placement of pages in the recommended target tier, we augment the struct page structure with a tier_id field which is updated by a modified madvise () function. During page compression, the zswap module reads this field and places the compressed page in the intended tier.

The decompression operation remains unchanged. During a page fault, the handle in the page table entry is used for the RB-Tree lookup to find the associated swap entry. The swap
entry contains the tier information, including the pool details and other relevant information to handle the fault [33].

Page migration between tiers: We enhance the kernel to allow the migration of pages between two compressed tiers. Currently, we follow a naive approach while migrating pages between compressed tiers by first decompressing the page from the source tier and then compressing again and placing it in the destination tier. This can be further optimized by skipping the decompression step if the source and destination tiers use the same compression algorithm.

Tiers statistics: We added support in the zswap subsystem to collect per-tier statistics such as the number of pages in the tier, size of the compressed tier, and total faults.

### 6.2 TS-Daemon

As shown in Figure 7, we implement our TierScape logic as a daemon (TS-Daemon). TS-Daemon uses the hardware counters to profile the memory access pattern of an application for a fixed time window (profile window). Specifically, it uses Intel PEBS [32] to monitor MEM_INST_RETIRED.ALL_LOADS and MEM_INST_RETIRED.ALL_STORES. These events report the virtual address of the page on which the event was generated [10]. TS-Daemon applies the Waterfall or analytic data placement model on the collected hotness profile to decide the destination tiers for the memory regions. Based on the model's outcome, TS-Daemon uses the kernel APIs described above to manage memory placement.

Regions. In order for efficient management of the address space of an application, TS-Daemon operates at a granularity of $2 \mathrm{MB}$ regions instead of $4 \mathrm{~KB}$ pages as commonly followed in other memory tiering solutions [46]. The hotness of $2 \mathrm{MB}$ region is an accumulated value of the hotness of each $4 \mathrm{~KB}$ page in it. TS-Daemon performs data migration to and from compressed tiers at the granularity of $2 \mathrm{MB}$ regions.

### 6.3 Data placement models

Waterfall model: We implement the waterfall model in the TS-Daemon. The input to the model is a hotness threshold value $-H_{t h}$. The value controls the pages that are to be evicted from DRAM to Tier 1. TS-Daemon maintains the tier data for all the regions and uses it to waterfall (demote to the next tier) the regions at the end of a profile window. A region restarts its journey from DRAM when it has (or a major portion of it) been faulted back to DRAM.

Analytical Model We implement the analytical model in C++ using the OR-Tools from Google [44]. The input to the model is the hotness profile of the application, tier stats (e.g., compressibility ratio, cost of the media backing the compressed tier, and access latency), list of regions, and a value for the knob $(\alpha)$. The model outputs a recommendation with a destination tier for each region. We evaluate the model
Table 2: The set of compressed tiers used to evaluate TierScape.

| ID | Name | Pool manager | Compressor | Media |
| :---: | :---: | :---: | :---: | :---: |
| T1 | ZB-L4-DR | zbud | 1 z4 | DRAM |
| T2 | ZB-L4-OP | zbud | 1 lz | Optane |
| T3 | ZS-L4-OP | zsmalloc | 1 z4 | Optane |
| T4 | ZS-LO-DR | zsmalloc | lzo | DRAM |
| T5 | ZS-DE-OP | zsmalloc | deflate | Optane |

on a separate client system connected via a local network that uses socket communication to send and receive data.

## 7 Evaluation

### 7.1 Configurations

Tiers. For evaluating TierScape, we use DRAM +5 compressed tiers identified in Section 4.2 - a total of 6 tiers. Table 2 shows the configuration of the 5 compressed tiers. For evaluating the 2-Tier system, we use DRAM + one compressed tier, where the configuration for the compressed tier is the one employed by Google in their production data centers - zsmalloc as the pool manager, lzo as the compression algorithm, and DRAM as the backing storage [36].

TS-Daemon. We use 120 seconds as our profile window duration (as used in the state-of-the-art 2-Tier technique [36]). We observe that a time window of 120 seconds is sufficient to stabilize the hotness profile of the pages based on events generated by the hardware counters. In addition, this time window provides ample opportunity for TS-Daemon to implement the model's page placement recommendations with minimal interruptions to the applications. Each run has a warm-up window of 100 seconds.

Hotness profile. The hotness of a region is based on the number of PEBS [32] samples observed during a profile interval. For the evaluation of 2-Tier system and Waterfall model, we experiment with three different hotness threshold values (regions with access counts less than the threshold value are eligible for placement in a compressed tier). The threshold values are selected to cover around $15-20 \%$ (conservative), 40-50\% (moderate), and 70-80\% (aggressive) of the application's data pages. For example, Memcached uses a threshold value of 50,100, and 250, respectively. Hotness statistics are gathered for pages in DRAM; hotness is not relevant for pages in compressed pools since they need to be first decompressed before accessing. For the analytical model, the average hotness value of the region for the past 4 profiling windows is directly fed into the model.

Model configuration. We use the following 2-Tier and TierScape configurations for our evaluation.

- 2T (2-Tier system): We experiment with 3 different configurations: conservative (2T-C), moderate (2T- $M$ ) and

Table 3: Description of the workloads and configurations.

| Workloads | Description | Input |
| :--- | :--- | :---: |
| Memcached [16] | A commercial in-memory object <br> caching system. | $44 \mathrm{~GB}$, Value <br> size: $4 \mathrm{~KB}$ |
| Redis [17] | A commercial in-memory key- <br> value store. | $41 \mathrm{~GB}$, Value <br> size: $4 \mathrm{~KB}$ |
| BFS [47] | Traverse graphs generated by web <br> crawlers. Use breadth-first search. | Nodes: $100 \mathrm{M}$ <br> Size: $18 \mathrm{~GB}$ |
| PageRank [47] | Assign ranks to pages based on pop- <br> ularity (used by search engines). | Nodes: $100 \mathrm{M}$ <br> Size: $18 \mathrm{~GB}$ |
| XSBench [49] | A key computational kernel of the <br> Monte Carlo neutron transport al- <br> gorithm | Setting: XL <br> Size: $119 \mathrm{~GB}$ |

aggressive (2T-A) based on the hotness threshold value.

- 6T-WF (6-tier waterfall model): We evaluate with the same hotness threshold values used above for 2-Tier setup (6T-WF-C, 6T-WF-M, and 6T-WF-A).
- 6T-AM- $\alpha$ (6-tier analytical model): We evaluate with 3 different values of $\alpha: 0.9,0.5$, and 0.1 .


### 7.2 Experiment setup

We use a tiered memory system with Intel Xeon Gold 6252N with 2 sockets, 24 cores per socket, and 2-way HT for a total of 96 cores. It has a DRAM-based near-memory tier with 384 GB capacity and a far-memory tier with Intel's Optane DC PMM [3] configured in flat mode (i.e., as volatile main memory) with 1.6 TB capacity. We run Fedora 30 and use a modified Linux kernel, 5.17.

Table 3 shows the real-world benchmarks and their configuration used to evaluate TierScape. We initialize Memcached and Redis databases with $\approx 42 \mathrm{~GB}$ of key-value pairs and then generate the load in a Gaussian distribution to better mimic real-life use cases [9]. We use the widely used memtier workload generator for load generation $[8,17]$. We use PageRank and BFS from the Ligra suite of graph benchmarks [47]. Input graphs for both graph workloads are generated using the standard rMat graph generator [4]. We also use XSBench which is a key computation kernel of the Monte Carlo neutron transort algorithm [49]. We use the "XL" setting of the workload, which generates a memory footprint of $119 \mathrm{~GB}$.

We execute the benchmarks and place the data as recommended by our data placement models. For Memcached and Redis, we use the throughput and latency numbers reported by memtier. For PageRank and BFS, we report the geometric mean of the time taken to execute multiple rounds. For XSBench, we use the time reported by the benchmark.

To calculate the memory TCO we use Equation 12. We capture the resident set size (RSS) to compute the cost of storing pages in DRAM and capture the size of compressed memory in all the tiers to compute the cost of storing data in the respective compressed tiers. We set the per-GB cost of Optane as $1 / 3$ of DRAM [43].

### 7.3 Results

Figure 8 compares relative performance and memory TCO savings for 2-Tier and TierScape systems on different workloads. Note that values on the $\mathrm{x}$-axis are on a decreasing scale.

### 7.3.1 TCO savings

It can be observed from Figure 8 that TierScape waterfall and analytical models outperform 2-Tier solutions by saving more memory TCO with similar or better performance. For example, the maximum memory TCO savings a 2-Tier solution can offer for Redis is $34.84 \%$ with the 2 T-A configuration. But it suffers a $18.33 \%$ performance slowdown. TierScape's waterfall model using the same hotness threshold (6T-WF-A) achieves a TCO saving of $56.11 \%$ (21.27 percentage points better than 2T-A) while incurring a performance loss of $19.48 \%$ (only 1.15 additional percentage points than the 2T-A). The analytical model configuration 6T-AM-0.1 achieves a TCO savings of $64.10 \%$ (29.26 more percentage points than 2T-A) while incurring a performance loss of only $14.37 \%$ ( 3.96 fewer percentage points than 2T-A).

### 7.3.2 Performance overheads

Similarly, it can be observed from Figure 8 that TierScape waterfall model and analytical model outperform 2-Tier solutions by performing with similar or better memory TCO savings. For example, PageRank, in a 2-Tier system, 2T-C offers the least performance slowdown of $21.82 \%$ while saving $24.86 \%$ memory TCO. TierScape waterfall model using the same hotness threshold (6T-WF-C) offers a better trade-off than 2T-C with $13.09 \%$ performance slowdown and $40.78 \%$ memory TCO savings. 6T-AM-0.9 incurs performance slowdown of only $14.91 \%$ and offers a TCO savings of $21.26 \%$

Also, it can be observed in BFS that 6T-AM-. 1 and 2T-M result in around $63 \%$ memory TCO savings, but 6T-AM-. 1 performs better by 4.05 percentage points ( $22.15 \%$ vs. $18.10 \%$ ). This demonstrates that with TierScape, more warm pages can be placed in compressed tiers to achieve better memory TCO savings without hurting performance.

### 7.4 Waterfall vs. Analytical model

In this section, we deep dive and analyze the waterfall and the analytical model's data placement recommendation. Figure 9 shows the model's recommendation in each profiling window for Memcached.

6T-WF-C and 6T-AM-0.9 retain most of the data in DRAM to ensure minimal performance slowdown as per the configured hotness threshold and tunable knob values. Both models recommend placing a small amount of data in compressed tiers. 6T-AM- 0.9 consistently recommends retaining more than $80 \%$ of data in DRAM for all profiling windows.

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=729&width=1415&top_left_y=259&top_left_x=363)

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=282&width=463&top_left_y=279&top_left_x=381)

(a) Memcached

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=288&width=463&top_left_y=276&top_left_x=820)

(b) Redis

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=288&width=447&top_left_y=276&top_left_x=1289)

(c) PageRank

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=284&width=458&top_left_y=644&top_left_x=606)

(d) BFS

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=282&width=458&top_left_y=648&top_left_x=1061)

(e) XSBench

Figure 8: Performance slowdown and memory TCO savings w.r.t to DRAM for 2-Tier and TierScape solutions

6T-WF-M and 6T-AM-0.5 recommend placing more pages in compressed tiers. In the waterfall model, more pages are "waterfalled" to tiers with better compression ratio, thus increasing the utilization of all the compressed tiers. In the analytical model, the percentage of pages that are retained in DRAM is $50 \%$ as the model converges towards placements with greater memory TCO savings based on the input to the model. The analytical model periodically recommends scattering many pages to the best TCO-saving tier, i.e., Tier 5 .

6T-WF-A, with aggressive memory TCO savings, retains only around $10 \%$ of data in DRAM. We see a significant jump in the utilization of the last tier. This is because more data is swapped out from DRAM and are "waterfalled" between the tiers, eventually reaching the last tier. Note that the figure shows the model's recommendation of the data placement based on the page hotness in the previous profile window. However, we observe that Memcached faults upon some of these pages, which are immediately moved back to DRAM.

6T-AM-0.1 recommends placing less than $5 \%$ of data in DRAM and the rest in compressed tiers. It recommends placing a majority of the data in Tier 2 instead of Tier 5. Tier 2 in our setting is zbud with $\mathrm{lz} 4$ backed by Optane. We observe that the last tier, using deflate, maintained an average compressibility ratio of 2 for Memcached. Whereas Tier 2 using lz4 achieved an average compressibility ratio of 1.35 Based on the overall cost of storing data and the underlying TCO, the model decided that placing most of the data on Tier 2 satisfies the TCO constraints. It can be noted that the model still recommends placing a small amount of data in the rest of the tiers.

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=610&width=834&top_left_y=1126&top_left_x=1100)

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=236&width=265&top_left_y=1145&top_left_x=1106)

(a) 6T-WF-C

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-10.jpg?height=236&width=266&top_left_y=1454&top_left_x=1103)

(d) 6T-AM-0.9 (b) 6T-WF-M

(e) 6T-AM- 0.5 (c) 6T-WF-A

(f) 6T-AM-0.1
Figure 9: Data placement recommendations for Memcached by waterfall and analytical models

### 7.5 In-depth Analysis

In the previous section, we looked into the model recommendation, while in this section, we analyze the ground reality, i.e., the number of pages actually placed in multiple tiers by TSDaemon as per model recommendation and the on-demand page faults incurred by the applications. Figure 10 shows the data placement for Redis for 6T-WF-M, 6T-WF-A, 6T-AM-.5, and 6T-AM-.1. We omit other benchmarks and configurations as they show a similar trend.

For the waterfall model, it can be observed from Figure 10 that after initial warm-up time, pages are first "waterfalled" to Tier 1 , and then they are gradually aged to better memory TCO saving tiers. The difference in placement recommen-

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=754&width=1487&top_left_y=233&top_left_x=319)

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=35&width=859&top_left_y=248&top_left_x=633)

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=255&width=358&top_left_y=325&top_left_x=336)

(b) Pages in 6T-WF-M

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=260&width=342&top_left_y=664&top_left_x=336)

(f) Faults in 6T-WF-M

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=255&width=360&top_left_y=325&top_left_x=701)

(c) Pages in 6T-WF-A

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=271&width=371&top_left_y=656&top_left_x=693)

(g) Faults in 6T-WF-A

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=252&width=356&top_left_y=329&top_left_x=1058)

(d) Pages in 6T-AM-.5

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=271&width=369&top_left_y=656&top_left_x=1062)

(h) Faults in 6T-AM-. 5

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=263&width=360&top_left_y=321&top_left_x=1425)

(e) Pages in 6T-AM-. 1

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=273&width=361&top_left_y=655&top_left_x=1424)

(i) Faults in 6T-AM-. 1

Figure 10: Pages placement across tiers by TS-Daemon as per the recommendation by the waterfall and analytical models and the actual page faults observed for Redis benchmark. Please note the difference in $\mathrm{y}$-axis scale across plots.

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=246&width=358&top_left_y=1189&top_left_x=236)

(a) Waterfall model

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-11.jpg?height=263&width=355&top_left_y=1175&top_left_x=598)

(b) AM model

Figure 11: Memory TCO savings for Redis

dation between 6T-WF-M and 6T-WF-A is clearly visible, where 6T-WF-M (moderate configuration) starts by placing around 3000K pages in Tier 1, while 6T-WF-A (aggressive configuration) starts by placing around $5000 \mathrm{~K}$ pages in Tier 1 . In 6T-WF-A all the tiers are almost equally utilized after a few profile windows (1000 seconds onward). While in 6T-WF-M the fast tiers (Tier 1 and Tier 2) have more pages than the slow tiers. This is as per the "moderate" configuration, which balances performance impact and memory TCO savings.

In the analytical model, 6T-AM-. 5 starts with placing pages in Tier 5 (Figure 10). It should be noted that the input to the analytical model is the average hotness of the regions for the last four profiling windows. Hence, as the model gets more information on application access pattern trends over subsequent profile windows, it starts preferring Tier 2 instead of Tier 5. A similar trend of initially placing pages in Tier 5 is also observed for 6T-AM-.1. However, 6T-AM-. 1 eventually places a significantly higher number of pages in Tier $2(7,000 \mathrm{~K}$ pages compared to $3,000 \mathrm{~K}$ pages in $6 \mathrm{~T}-\mathrm{AM}-.5$ ), resulting in better memory TCO savings.

Further, it can be observed in the bottom plots of Figure 10 that only a small fraction of the pages (around 10\%) placed in compressed tiers are accessed by the application, resulting in a page fault. This clearly indicates the efficiency of waterfall and analytical models to correctly recommend the placement of the pages in the appropriate compressed tiers. In addition, the memory TCO trend in Figure 11 corroborate with the page placements in Figure 10 that also reflects the placement decisions made by models with different configurations.

### 7.6 Impact on the tail latencies

One of the key requirements in a data center is to maintain an SLA guarantee on the tail latencies of an application. A hosted application should not suffer exorbitantly high tail latencies in the pursuit of aggressively reducing memory TCO. In $\mathrm{N}$ tier systems, the total number of faults in the slowest tier and the decompression (or access) latency of the slowest tier can impact the tail latency of the application.

Figure 12 shows the average and 99th percentile latency for Memcached. It can be observed that the latency values, both average and 99th percentile, increases for both 2-Tier and N-tier (6T-WF and 6T-AM) as we aggressively place more pages in compressed tiers. The average latency values for 2T, 6T-WF and 6T-AM are comparable for similar aggressive settings (e.g., 2T-C vs. 6T-WF-C).

It can be observed that 6T-WF and 6T-AM outperform all 2T configurations for 99 th percentile latency. For waterfall model as pages are gradually aged into slowest tier, only pages that are actually cold end up in slowest tier and hence performs better than all 2T configurations. Analytical model carefully scatters the pages across tiers based on the hotness values of the pages in the past four profile windows. Hence the

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-12.jpg?height=363&width=684&top_left_y=241&top_left_x=257)

Figure 12: Latency data for Memcached

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-12.jpg?height=282&width=835&top_left_y=743&top_left_x=187)

Figure 13: CPU utilization by TS-Daemon which includes telemetry collection, post-processing the telemetry data, page migration, and (de)compression overheads.

99th percentile latency for 6T-AM-. 9 and 6T-AM-. 5 is better than $2 \mathrm{~T}$ and 6T-WF. However, for 6T-AM-. 1 with aggressive memory TCO savings settings, 99 th percentile latency is higher than 6T-WF-A, but is still better than 2T-A.

### 7.7 TS-Daemon Tax

In this section, we analyze the tax in terms of CPU utilization incurred by TS-Daemon. The tax includes telemetry data collection and post-processing along with page migration tax (including decompressing a page from one tier and compressing and placing it in another tier). Pages decompressed due to on-demand faults are not accounted for in TS-Daemon as they are accounted for in the benchmark execution time.

It can be observed from Figure 13 that TS-Daemon incurs around $1.2 \%$ to $7 \%$ CPU utilization for different benchmarks. For most of the benchmarks, CPU utilization for TS-Daemon is slightly higher for waterfall and analytical models, as additional CPU cycles are burned during every profiling window to redistribute the pages across multiple tiers as per the recommendation by the model.

Also, the tax for the analytical model does not include the CPU overheads to evaluate the model as it is offloaded to a client system. We also measure the overhead on the client system, which contributes to less than 0.4 percentage point increase in CPU utilization for analytical models.

## 8 Related Work

Several tiered memory systems have been proposed in recent years $[18,19,21,26,28,29,34,35,36,37,39,40,46,50,51]$, along with data placement and migration policies to optimize performance and memory TCO. Most of the prior work are based on two tier system, where the first tier consists of low latency and costly DRAM memory while the second tier consists of high latency but cheaper memory tiers backed by NVMMs [3] or CXL-attached memory [25, 27, 41]. Recently, memory tiering using a compressed memory tier has been explored by a hyper-scale data center provider [36].

Hardware-based memory tiering with NVMMs [18, 26, 34, 50, 51] or CXL-attached memory [19, 21, 28, 39, 40] lack the flexibility in the software to define memory tiers with distinct access latency, as access latency is determined by the underlying storage media. While TierScape proposes a way to define multiple copressed memory tiers in the software.

Prior proposals employ different telemetry techniques to identify hot and cold data [30, 32, 38, 42, 53]. HeMem [46] accumulates access information from PEBS [32] into larger regions to reduce the overheads of tracking pages at $4 \mathrm{~KB}$ granularity. TS-Daemon also uses PEBS and operates at larger regions for hot, warm, and cold data tracking.

Page placement policies employed in prior works are fine tuned for page placement in a two tier systems $[26,36,39$, 40, 46, 50] and cannot be directly applied to N-tier systems as they do not exploit the distinct access latency and capacity savings across tiers. TierScape proposes Waterfall and analytical models for efficient page placement across tiers.

Prior works also employ optimizations to decide the time and rate at which the pages are migrated across tiers [36, 46, 51]. For example, Nimble [51] proposes multi-threaded and concurrent migrations of pages across tiers. TS-Daemon also employs multi-threaded and concurrent migrations of pages.

The recent work from Google [36] proposes a softwaredefined two tiered system with DRAM and a single compressed tier to improve memory TCO savings for cold data. The ACCESSED bit in the page table is periodically scanned to identify and migrate cold pages to a compressed memory tier backed by DRAM. An AI/ML based prefetching technique is also employed to proactively move pages from compressed second tier memory to DRAM. TierScape differs from Google's approach as it defines and manages multiple software-defined compressed memory tiers.

## 9 Conclusion

We conclude with comprehensive experimental evidence that defining multiple compressed memory tiers in the software and exploiting data placement across tiers is an optimistic way forward to tame the high memory TCO in modern data centers.

## References

[1] Buddy memory allocation - wikipedia. https://en. wikipedia.org/wiki/Buddy_memory_allocation. (Accessed on 08/10/2023).

[2] Decadal plan for semiconductors. https : / / www . src . org / about / decadal-plan / decadal-plan-full-report.pdf.

[3] Intel $\circledR^{\text {optane }}{ }^{\mathrm{TM}}$ dc persistent memory product brief. https://www.intel.in/content/dam/ www / public/us/en/documents/product-briefs/ optane-dc-persistent-memory-brief.pdf. (Accessed on 08/01/2023).

[4] jshun/ligra: Ligra: A lightweight graph processing framework for shared memory. https://github.com/ jshun/ligra. (Accessed on 08/10/2023).

[5] Lempel-ziv-oberhumer - wikipedia. https://en. wikipedia.org/wiki / Lempel\%E2\%80\%93Ziv\%E2\% $80 \% 930$ berhumer. (Accessed on 08/04/2023).

[6] lz4/lz4: Extremely fast compression algorithm. https: //github.com/lz4/lz4. (Accessed on 08/04/2023).

[7] Lzo [lwn.net]. https://lwn. net / Articles / $545878 /$. (Accessed on 08/04/2023).

[8] Memcached: Aws graviton2 benchmarking - infrastructure solutions blog - arm community blogs - arm community. https : / / community . arm . com/arm-community-blogs / b / infrastructure-solutions-blog / posts / memcached-benchmarking-aws-graviton2-50-p-p-gains. (Accessed on 08/10/2023).

[9] New in memtier benchmark: Pseudo-random data, gaussian access pattern and range manipulation. https : / / redis . com / blog / new-in-memtier_ benchmark-pseudo-random-data-gaussian-access-patter

[10] Perfmon events. https://perfmon-events.intel. $\mathrm{com} /$. (Accessed on 08/01/2023).

[11] Silesia compression corpus. https://sun.aei.polsl. pl// sdeor/index.php?page=silesia. (Accessed on 08/06/2023).

[12] z3fold - the linux kernel documentation. https: / / www.kernel.org/ doc/html/v5.8/vm/z3fold. html. (Accessed on 08/04/2023).

[13] Zram will see greater performance on linux 5.1 - it changed its default compressor - phoronix. https://www.phoronix.com/news/ZRAM-Linux-5. 1 -Better-Perform. (Accessed on 08/04/2023).
[14] zswap - the linux kernel documentation. https:// www.kernel.org/doc/html/v5.8/vm/zswap.html? highlight=zbud. (Accessed on 08/04/2023).

[15] zswap - the linux kernel documentation. https:// www.kernel.org/doc/html/latest/admin-guide/ mm/zswap.html. (Accessed on 07/22/2023).

[16] memcached - a distributed memory object caching system. https://memcached.org/, 2019. (Accessed on $11 / 18 / 2019)$.

[17] Redis. https://redis.io/, 2019. (Accessed on $11 / 18 / 2019)$.

[18] Neha Agarwal and Thomas F. Wenisch. Thermostat: Application-transparent page management for twotiered main memory. In Proceedings of the TwentySecond International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS '17, page 631-644, New York, NY, USA, 2017. Association for Computing Machinery.

[19] Minseon Ahn, Andrew Chang, Donghun Lee, Jongmin Gim, Jungmin Kim, Jaemin Jung, Oliver Rebholz, Vincent Pham, Krishna Malladi, and Yang Seok Ki. Enabling cxl memory expansion for in-memory database management systems. In Data Management on New Hardware, DaMoN'22, New York, NY, USA, 2022. Association for Computing Machinery.

[20] Shoaib Akram. Performance evaluation of intel optane memory for managed workloads. ACM Trans. Archit. Code Optim., 18(3), apr 2021.

[21] Moiz Arif, Kevin Assogba, M. Mustafa Rafique, and Sudharshan Vazhkudai. Exploiting cxl-based memory for distributed deep learning. In Proceedings of the 51st International Conference on Parallel Processing, ICPP '22, New York, NY, USA, 2023. Association for

![](https://cdn.mathpix.com/cropped/2024_06_04_08bf3109330703a8160ag-13.jpg?height=54&width=461&top_left_y=1849&top_left_x=1168)

[22] David Barina. Experimental lossless data compressor. Microprocessors and Microsystems, 98:104803, 2023.

[23] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan,
and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc., 2020.

[24] Jonathan Corbet. The zsmalloc allocator [lwn.net]. https://lwn.net/Articles/477067/, 2012. (Accessed on 08/04/2023).

[25] CXL. Compute express link. https://www. computeexpresslink.org/, 2023.

[26] Padmapriya Duraisamy, Wei Xu, Scott Hare, Ravi Rajwar, David Culler, Zhiyi Xu, Jianing Fan, Christopher Kennelly, Bill McCloskey, Danijela Mijailovic, Brian Morris, Chiranjit Mukherjee, Jingliang Ren, Greg Thelen, Paul Turner, Carlos Villavieja, Parthasarathy Ranganathan, and Amin Vahdat. Towards an adaptable systems architecture for memory tiering at warehousescale. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS 2023, page 727-741, New York, NY, USA, 2023. Association for Computing Machinery.

[27] Samsung Electronics. Samsung electronics introduces industry's first $512 \mathrm{gb}$ cxl memory module. https: / / semiconductor. samsung.com/newsroom/news/ samsung-electronics-introduces-industrys-fir 2022.

[28] Donghyun Gouk, Miryeong Kwon, Hanyeoreum Bae, Sangwon Lee, and Myoungsoo Jung. Memory pooling with cxl. IEEE Micro, 43(2):48-57, 2023.

[29] Vishal Gupta, Min Lee, and Karsten Schwan. Heterovisor: Exploiting resource heterogeneity to enhance the elasticity of cloud platforms. In Proceedings of the 11th ACM SIGPLAN/SIGOPS International Conference on Virtual Execution Environments, VEE '15, page 79-92, New York, NY, USA, 2015. Association for Computing Machinery.

[30] Christian Hansen. Linux idle page tracking, 2018.

[31] Danny Harnik, Ety Khaitzin, Dmitry Sotnikov, and Shai Taharlev. A fast implementation of deflate. In 2014 Data Compression Conference, pages 223-232, 2014.

[32] Intel. Pebs (processor event-based sampling) manual, 2023.

[33] Seth Jennings. The zswap compressed swap cache [lwn.net]. https://lwn.net/Articles/537422/, 2013. (Accessed on 08/04/2023).

[34] Jonghyeon Kim, Wonkyo Choe, and Jeongseob Ahn. Exploring the design space of page management for Multi-Tiered memory systems. In 2021 USENIX Annual
Technical Conference (USENIX ATC 21), pages 715728. USENIX Association, July 2021.

[35] Sandeep Kumar, Aravinda Prasad, Smruti R. Sarangi, and Sreenivas Subramoney. Radiant: Efficient page table management for tiered memory systems. In Proceedings of the 2021 ACM SIGPLAN International Symposium on Memory Management, ISMM 2021, page 66-79, New York, NY, USA, 2021. Association for Computing Machinery.

[36] Andres Lagar-Cavilla, Junwhan Ahn, Suleiman Souhlal, Neha Agarwal, Radoslaw Burny, Shakeel Butt, Jichuan Chang, Ashwin Chaugule, Nan Deng, Junaid Shahid, Greg Thelen, Kamil Adam Yurtsever, Yu Zhao, and Parthasarathy Ranganathan. Software-defined far memory in warehouse-scale computers. In International Conference on Architectural Support for Programming Languages and Operating Systems, 2019.

[37] Taehyung Lee, Sumit Kumar Monga, Changwoo Min, and Young Ik Eom. Memtis: Efficient memory tiering with dynamic page classification and page size determination. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 17-34, 2023.

[38] Michael Lespinasse. V2: idle page tracking / working -512gsetexstimation 203zule/,

[39] Huaicheng Li, Daniel S. Berger, Lisa Hsu, Daniel Ernst, Pantea Zardoshti, Stanko Novakovic, Monish Shah, Samir Rajadnya, Scott Lee, Ishwar Agarwal, Mark D. Hill, Marcus Fontoura, and Ricardo Bianchini. Pond: Cxl-based memory pooling systems for cloud platforms. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2023, page 574-587, New York, NY, USA, 2023. Association for Computing Machinery.

[40] Hasan Al Maruf, Hao Wang, Abhishek Dhanotia, Johannes Weiner, Niket Agarwal, Pallab Bhattacharya, Chris Petersen, Mosharaf Chowdhury, Shobhit Kanaujia, and Prakash Chauhan. Tpp: Transparent page placement for cxl-enabled tiered-memory. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS 2023, page 742-755, New York, NY, USA, 2023. Association for Computing Machinery.

[41] Inc. Micron Technology. Micron launches memory expansion module portfolio to accelerate cxl 2.0 adoption. https:// investors.micron. com / news-releases / news-release-details / micron-launches-memory-expansion-module-portfolio-acc 2022 .

[42] SeongJae Park, Yunjae Lee, and Heon Y. Yeom. Profiling dynamic data access patterns with controlled overhead and quality. In Proceedings of the 20th International Middleware Conference Industrial Track, Middleware '19, page 1-7, New York, NY, USA, 2019. Association for Computing Machinery.

[43] Bo Peng, Yaozu Dong, Jianguo Yao, Fengguang Wu, and Haibing Guan. Flexhm: A practical system for heterogeneous memory with flexible and efficient performance optimizations. ACM Trans. Archit. Code Optim., 20(1), $\operatorname{dec} 2022$.

[44] Laurent Perron and Vincent Furnon. Or-tools.

[45] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, Irina Higgins, Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, L. Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, N. K. Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Tobias Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew G. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem W. Ayoub, Jeff Stanway, L. L. Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis \& insights from training gopher. ArXiv, $\mathrm{abs} / 2112.11446,2021$.

[46] Amanda Raybuck, Tim Stamler, Wei Zhang, Mattan Erez, and Simon Peter. Hemem: Scalable tiered memory management for big data applications and real nvm. In Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles, SOSP '21, page 392-407, New York, NY, USA, 2021. Association for Computing Machinery.

[47] Julian Shun and Guy E. Blelloch. Ligra: A lightweight graph processing framework for shared memory. In Proceedings of the 18th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming,
PPoPP '13, page 135-146, New York, NY, USA, 2013. Association for Computing Machinery.

[48] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Anand Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. ArXiv, $\mathrm{abs} / 2201.11990,2022$.

[49] John Tramm, Andrew Siegel, Tanzima Islam, and Martin Schulz. Xsbench - the development and verification of a performance abstraction for monte carlo reactor analysis. 092014 .

[50] Johannes Weiner, Niket Agarwal, Dan Schatzberg, Leon Yang, Hao Wang, Blaise Sanouillet, Bikash Sharma, Tejun Heo, Mayank Jain, Chunqiang Tang, and Dimitrios Skarlatos. Tmo: Transparent memory offloading in datacenters. In Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS '22, page 609-621, New York, NY, USA, 2022. Association for Computing Machinery.

[51] Zi Yan, Daniel Lustig, David Nellans, and Abhishek Bhattacharjee. Nimble page management for tiered memory systems. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS '19, page 331-345, New York, NY, USA, 2019. Association for Computing Machinery.

[52] Jian Yang, Juno Kim, Morteza Hoseinzadeh, Joseph Izraelevitz, and Steve Swanson. An empirical guide to the behavior and use of scalable persistent memory. In 18th USENIX Conference on File and Storage Technologies (FAST 20), pages 169-182, 2020.

[53] Yu Zhao. Multigenerational lru framework. https: //lwn.net/Articles/880393/, 2022.

</end of paper 2>


