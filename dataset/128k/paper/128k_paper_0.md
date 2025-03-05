<paper 0>
# TinyGSM: achieving $>80 \%$ on GSM8k with small language models 

Bingbin Liu ${ }^{1}$, Sebastien Bubeck ${ }^{2}$, Ronen Eldan ${ }^{2}$, Janardhan Kulkarni ${ }^{2}$,<br>Yuanzhi $\mathrm{Li}^{2}$, Anh Nguyen ${ }^{2}$, Rachel Ward ${ }^{2}$, Yi Zhang ${ }^{2}$<br>${ }^{1}$ Carnegie Mellon University<br>${ }^{2}$ Microsoft Research


#### Abstract

Small-scale models offer various computational advantages, and yet to which extent size is critical for problemsolving abilities remains an open question. Specifically for solving grade school math, the smallest model size so far required to break the $80 \%$ barrier on the GSM8K benchmark remains to be 34B. Our work studies how high-quality datasets may be the key for small language models to acquire mathematical reasoning. We introduce TinyGSM, a synthetic dataset of $12.3 \mathrm{M}$ grade school math problems paired with Python solutions, generated fully by GPT-3.5. After finetuning on TinyGSM, we find that a duo of a 1.3B generation model and a 1.3B verifier model can achieve $81.5 \%$ accuracy, outperforming existing models that are orders of magnitude larger. This also rivals the performance of the GPT-3.5 "teacher" model (77.4\%), from which our model's training data is generated. Our approach is simple and has two key components: 1) the high-quality dataset TinyGSM, 2) the use of a verifier, which selects the final outputs from multiple candidate generations.


## 1 Introduction

One fascinating phenomenon regarding large language models (LLMs) is the emergence of capbilities as both the model and dataset sizes scale up (Wei et al., 2022a; Chan et al., 2022). Among many capabilities, mathematical reasoning is one particular aspect that has received tremendous attention Lewkowycz et al. (2022); Lightman et al. (2023). However, it is unclear to what extend scale is a necessity for mathematical reasoning, and the potential of small language models (SLMs) remains largely under-explored.

In this work, we push the boundaries of SLMs' math reasoning capabilities. As a first step towards general mathematics, our primary testing ground is grade school math problems, to solve which require both mathematical understanding and language comprehension. The gold-standard benchmark in this regime is GSM8K (Cobbe et al., 2021), a collection of $8.8 \mathrm{~K}$ grade-school math word problems (with a $7 \mathrm{k}-1 \mathrm{k}$ train-test split) that involve 2 to 11 reasoning steps. GSM8K has been widely regarded to be challenging for LLMs. Even though the questions appear rather simple for humans, there have been few models that achieve $>80 \%$, and they are commonly of prohibitive sizes, e.g. 34B and above (see Table 1).

Our goal is to break the $80 \%$ barrier on GSM8K while keeping the model size friendly. As previous work shows (Gunasekar et al., 2023; Li et al., 2023; Eldan \& Li, 2023), training data quality is one of the most important factors for enhancing performance of small models. In particular, prompt-engineered synthetic data generation from gigantic models such as GPT-3.5/4 enjoys the clear advantage of desirable data hygiene and controllable diversity. This constituents a teacher-student scenario where the student learns from teacher's generations. On the tasks that the model model already excels at, their guided generations remain one of the highest quality data one can collect for training significantly smaller student models. It is also understood that the student model's performance likely ends up inferior than the teacher, and may fall far short especially when the student is considerably smaller than the teacher (Mirzadeh et al., 2019; Gudibande et al., 2023) —after all, the teacher places an information-theoretic bottleneck on the student.

To our surprise, in the case of GSM8K, we are able to bridge the performance gap between the student and teacher, by utilizing a tiny amount of labeled real data (the original GSM8K training set of $7 \mathrm{k}$ questions) to train an independent verifier model. At test time, the verifier score and select among multiple candidate answers generated from the student, and then we output the highest score generation as the final submission. Note the idea of using

![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-02.jpg?height=707&width=1396&top_left_y=186&top_left_x=359)

Figure 1: Our results on GSM8K. Please refer to Table 1 for details.

a verifier is proposed by the seminal GSM8K paper (Cobbe et al., 2021), and here we demonstrate its power of bridging the teacher-student gap, and we conduct a more thorough examination of factors affecting its efficacy.

The contributions of our work are the following:

- We introduce TinyGSM, a synthetic dataset containing GSM8K-style math word problems paired with Python solutions, generated fully by GPT-3.5-turbo. TinyGSM consists of $12.3 \mathrm{M}$ questions which amount to 1.8B tokens. We demonstrate TinyGSM's high-quality by finetuning the Phi-1.5 1.3B model (before the use of verifiers) which improves its accuracy from $44.6 \%$ to $\mathbf{6 8 . 2 \%}$ on the GSM8K test set. Notably, our smallest $125 \mathrm{M}$ model can also achieve $\mathbf{6 3 . 1 \%}$ after finetuning on TinyGSM.
- We demonstrate the power of verifiers on small-scale models. When integrated with a verifier for scoring generations, our models, named Phi-GSM models, achieve performance on par with other open source models that are orders of magnitude larger. In particular, our 1.3B model achieves $81.5 \%$ accuracy on GSM8K, as shown in Figure 1. This marks a new state-of-the-arts on billion-parameter-scale models, significantly outperforming existing open-source models and even rivaling the $77.4 \%$ accuracy of GPT-3.5, from which TinyGSM is generated. For verifier training, we identify data diversity as a crucial element for a verifier's success, and find that the scaling of the verifier may be more effective than scaling of the generator: while scaling up from a $125 \mathrm{M}$ generator to a $1.3 \mathrm{~B}$ generator only gives a $5.1 \%$ increase in performance (Table 1), scaling up the verifier from $125 \mathrm{M}$ to $1.3 \mathrm{~B}$ leads to a $7.2 \%$ performance boost (Figure 4 ).


## 2 Related works

Distilling from synthetic data: While scaling up has been a useful strategy, it is possible to outpace conventional scaling laws by better use of data (Sorscher et al., 2022). In the data-scarce case, quality synthetic data serves as an effective workaround (Eldan \& Li, 2023; Gunasekar et al., 2023), and the scaling in dataset size can compensate for a small model size (Edelman et al., 2023). Additionally, our work uses samples in the true distribution (i.e. the GSM8K train set) differently: given the small dataset size, we believe that the most sample-efficient way to utilize the true train set is to train a verifier-while the $7.4 \mathrm{k}$ samples in the GSM8K training set is too small for language model finetuning, it is sufficient for training a good quality verifier that provides $10 \%$ performance boost. While there have been potential concerns of learning from synthetic data such as loosing diversity or having a drifted distribution mean (Alemohammad et al., 2023; Shumailov et al., 2023), Alemohammad et al. (2023) showed that such degradation can be avoided by including fresh samples from the true distribution during training.

| $\overline{\text { Model }}$ | Base model | $\overline{\text { Model size }}$ | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=227&top_left_y=453&top_left_x=1207) | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=196&top_left_y=453&top_left_x=1454) | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=53&width=189&top_left_y=453&top_left_x=1673) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-2 (Touvron et al., 2023) | - | $\overline{7 \mathrm{~B}}$ | nlp | pass@1 | 14.6 |
|  |  | 13B |  |  | 28.7 |
|  |  | $34 \mathrm{~B}$ |  |  | 42.2 |
|  |  | 70B |  |  | 56.8 |
| MetaMath (Yu et al., 2023b) | Llama-2 | 7B | nlp | pass@1 | 66.5 |
|  |  | 13B |  |  | 72.3 |
|  |  | 70B |  |  | 82.3 |
| WizardMath (Luo et al., 2023) | Llama-2 | 7B | nlp | pass@1 | 54.9 |
|  |  | 13B |  |  | 63.9 |
|  |  | $70 \mathrm{~B}$ |  |  | 81.6 |
| MAmmoTH (Yue et al., 2023) | Code-Llama | 7B | code | pass@1 | 59.4 |
|  | Code-Llama | 12B |  |  | 64.7 |
|  | Code-Llama | $34 \mathrm{~B}$ |  |  | 72.7 |
|  | Llama-2 | 70B |  |  | 76.9 |
| Mistral (Jiang et al., 2023) | - | 7B | nlp | maj1@8 | 52.2 |
|  | - | $8 \times 7 \mathrm{~B}$ |  |  | 58.4 |
| OVM (Yu et al., 2023a) | Llama-2 | $7 \mathrm{~B}+7 \mathrm{~B}$ | nlp | verify100@1 | 73.7 |
|  | Mistral | $7 \mathrm{~B}+7 \mathrm{~B}$ |  |  | $84.7 \quad-1$ |
| Llemma (Azerbayev et al., 2023) | Llama-2 | 7B | nlp | pass@1 | 36.4 |
|  |  | $34 \mathrm{~B}$ |  |  | 51.5 |
| ToRA-Code (Gou et al., 2023) | Llama-2 | 7B |  |  | 72.6 |
|  |  | 13B | code. | COT@1 | 75.8 |
|  |  | $34 \mathrm{~B}$ | code |  | 80.7 |
|  |  | $70 \mathrm{~B}$ |  |  | 84.3 |
| Orca 2 (Mitra et al. 202.3) | Ulama-2 | 7B | nln | nacs@1 | 55.72 |
|  |  | 13B |  | passe 1 | 65.73 |
| Gemini Pro |  |  | $n \ln$. | moi1०20 | ![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-03.jpg?height=43&width=189&top_left_y=1504&top_left_x=1673) |
| Gemini Ultra (Gemini Team) | ${ }^{-}$ | ${ }^{-}$ | nIp | majल | 94.4 |
| GPT-3.5-0613 |  |  | ando | $\operatorname{mocol} 1$ | $77.4^{*}$ |
| GPT-4-0613 (OpenAI, 2023) | - | - | code | passલl 1 | $97.0^{*}$ |
| Phi-1.5 (Li et al., 2023) | - | $1.3 \mathrm{~B}$ | code | $\overline{\text { pass@1 }}$ | 44.6 |
|  | Phi-1.5-tiny | $125 \mathrm{M}$ |  |  | 63.1 |
| Phi-GSM | Phi-1.5-small | $350 \mathrm{M}$ | ordo | maca@1 | 65.9 |
|  | Phi-1.5 | $1.3 \mathrm{~B}$ | code | pass@ 1 | 68.2 |
|  | Phi-2 | $2.7 \mathrm{~B}$ |  |  | 74.3 |
|  | Phi-1.5-tiny | $125 \mathrm{M}+125 \mathrm{M}$ |  |  | 68.9 |
| Phi-GSM $+V$ | Phi-1.5-small | $350 \mathrm{M}+350 \mathrm{M}$ | code | verify48@1 | 71.3 |
|  | Phi-1.5 | $1.3 \mathrm{~B}+1.3 \mathrm{~B}$ |  |  | 81.5 |

Table 1: Results on GSM8K. * denotes results measured by ourselves. Accuracies above $\mathbf{8 0 \%}$ are labeled in bold. ' $8 \times 7 \mathrm{~B}$ ' stands for mixture of 8 experts, and each expert is of $7 \mathrm{~B}$ parameters. ' $7 \mathrm{~B}+7 \mathrm{~B}$ ' means a combination of a $7 \mathrm{~B}$ generation model plus a $7 \mathrm{~B}$ verifier model. ' $+\mathrm{V}$ ' denotes the use of verifier models.

Math word problem datasets GSM8K (Cobbe et al., 2021) has been the most common used math word problem dataset for its quality and size. In comparison, earlier datasets such as MAWPS (Koncel-Kedziorski et al., 2016), ASDiv (Miao et al., 2020) and SVAMP (Patel et al., 2021) are either much smaller in size or of less difficulty. However, GSM8K questions are too clean to test for robustness. Motivated by the observation that language models are not robust to the presence of irrelevant context, Shi et al. (2023a) proposed GSM-IC (irrelevant context). Another problem is the GSM8K dataset itself is still too small for training language models. (Ni et al., 2023b) addressed this with self-sampled data. In a work concurrent to ours, Yu et al. (2023b) bootstraps an original dataset using various augmentation techniques, such as generating multiple answers for the solution, question rephrasing, and backward reasoning. The proposed MetaMath dataset consists of 40000 questions from GSM8K and MATH (Hendrycks et al., 2021). In comparison, TinyGSM is significantly larger, encompassing 12.3M questions (or equivalently 1.8B tokens).

Leveraging multiple generations: An important component of our method is to leverage multiple generation. This idea has been proven successful in many prior works. A notable example is "self-consistency" (Wang et al., 2022), which selects the most frequent response among candidates and integrates well with other methods such as progressive-hint prompting (Zheng et al., 2023) and model selection (Zhao et al., 2023). However, self-consistency was not particularly helpful in our setup as mentioned in Section 4.2. More related to and a direct inspiration of our work is Cobbe et al. (2021), which uses a verifier to select the best response among 100 candidates, leading to an $20 \%$ accuracy boost. Our work conducts a more thorough study on the design choices of the verifier, including data diversity and the effect of verifier sizes. Another design choice orthogonal to ours is the supervision signals, such as outcome-based supervision versus process supervision (Lightman et al., 2023).

Learning from partial or process supervision: In our experiments, we evaluate on the final accuracy only but train on full programs. Prior work has studied the effect of process versus outcome based supervision. Process-based supervision is shown to be particularly helpful for complex math problems (Lightman et al., 2023), though for general problems one needs to consider a cost-efficacy tradeoff (Uesato et al., 2022). When process supervision is not available, $\mathrm{Ni}$ et al. (2023b) proposed to learn from "self-sampled" solutions, which allows the model to learn from partially correct self-generated solutions selected based on the execution trace.

Self-improvement: Several works have explored the idea of "self-improvement" where a model evaluates and corrects its own generations, mostly relying on the self-debugging ability of GPT4. Examples include "selfrefine" (Madaan et al., 2023) and "self-verify" (Weng et al., 2022; Zhou et al., 2023), both of which ask the model to iteratively provide feedback or verifications on its own generations and refine if needed. However, such self-improvement abilities have not been discovered in small language models. This motivated our use of a separate verifier model, which is initialized from the generative model but needs to be fully finetuned for verification.

Prompt-based methods: Prompt-based methods, which find prompts to improve the later conditional generations, are particularly effective for large models. Examples include in-context learning (Brown et al., 2020), where the model learns to perform novel tasks from few-shot examples provided in the prompt, as well as Chain-ofThought (Wei et al., 2022b), which shows that explicitly generating intermediate steps can significantly help with reasoning abilities. However, similar to self-improvements, prompting is targeted at large language models and do not apply for SLMs.

## 3 The TinyGSM dataset

Our objective is to assess the capability of a small language model (SLM) on mathematical reasoning. Ideally, enhancing this mathematical reasoning ability should not compromise the model's competence in language comprehension. This makes math word problems, which necessitate both mathematical and language understanding, a suitable test ground. We focus on the GSM8K dataset (Cobbe et al., 2021), consisting of around $8 \mathrm{k}$ grade-school math word problems. The math concepts in the dataset are elementary and within standard grade-school curricula, but the challenges posed by the natural language problem statement introduce an additional layer of complexity to the task.

TinyGSM: augmenting GSM8K with synthetic generations Despite the high quality, the GSM8K training set only contains 7473 problems, which is too small for training a reasonably sized language model (Ni et al., 2023a). To alleviate the size issue, we augment the GSM8K training set using GPT-3.5-turbo generated synthetic problems.

We prompt GPT-3.5-turbo to generate problem variants similar to a given question (but not the solution) randomly

```
def simple_math_problem() -> int:
    "n"
    In preparation for her party, Sarah buys 10 trays
    of food and 8 cases of beverages.
    Each tray costs $50 and each case of beverages
    costs $20.
    What is the total cost of the trays and beverages?
    n""
    trays = 10
    tray_cost = 50
    cases = 8
    case_cost = 20
    tray_total = trays * tray_cost
    case_total = cases * case_cost
    total_cost = tray_total + case_total
    result = total_cost
    return result
```

Figure 2: Examples from TinyGSM. The question is given as the docstring of a function, and the solution is the code in the function body.

sampled from the GSM8K training set. Each problem variant contains both a question and the corresponding solution written in Python, as shown in Figure 2. ${ }^{1}$ Using code allows us to leverage a Python interpreter, circumventing language models' known limitation regarding numerical calculations and code execution.

To enhance robustness, we also generated synthetic problems whose questions contain irrelevant information. This is achieved by augmenting the GSM-IC dataset (Shi et al., 2023a), which is an augmentation of GSM8K specifically designed to introduce irrelevant context (IC) to the question statement. These GSM-IC variants constitute to approximately one third of TinyGSM.

The resulting synthetic dataset contains $12.3 \mathrm{M}$ problems (i.e. question-solution pairs) ${ }^{2}$ with, based on the original $7.4 \mathrm{k}$ training set questions and their IC variants. For each question in the GSM8K train set, the prompt based on this question is shared across API calls, and the source of randomness comes entirely from the generation process. To encourage diversity, we use temperature sampling and specify in the prompt to encourage the problem variants to be grammatically diverse and contain multiple steps; the exact prompts are provided in Figure 3 and in Appendix A.1.

Filtering To ensure the quality of the synthetic data in TinyGSM, we filter out problems that are too short or do not contain numbers, as well as code solutions which are not executable. Note that we do not check for the correctness of the question or the generated solutions, since the "ground truth" solution is not available. Given the effectiveness of self-consistency (Wang et al., 2022), one might want to filter the problems by keeping the ones which have majority vote only. We did not adopt this strategy since we find that GPT-3.5-turbo's generations are only consistent on easy problems ${ }^{3}$, hence such consistency filtering will remove challenging problems, resulting in a dataset that is too easy to be useful.

## 4 Solving grade school math with small language models

The 1.3B version of our phi-GSM models is able to achieve $81.5 \%$ accuracy on GSM8K, a dataset that remains challenging for small-scale models. The performance comes from sufficient good quality synthetic data and the use[^0]

Consider the following grade-school math problem: \{\{question\}\}

Generate 10 different math problems similar to this math problem.

- Make sure each question uses diverse NLP and includes multiple logical steps.
- After each generated problem, write down a **detailed and complete Python program** to solve the question **step by step** (do NOT give the result directly, **DO NOT write calculations in the comments**).
- The program should contain multiple lines of code and end with 'result = XXX' (Make sure to replace XXX with the actual result of the python program).
- Make sure your Python program is complete and solves the problem. Do **NOT** write things like 'solution to be completed', result = ?, insert your code here etc.
- Give the complete solution to solve the problem, written in Python. Do not write things like 'insert your code here'.
- In each new question, **first end with <lendofquestion|>**, and then start writing the program. Each program should end with <lendofprogram|>.
- Example format: Question X: New question (at least 4 sentences long and use diverse NLP) (without the solution)

<|endofquestion|> Complete python code with entire solutions and the correct indent (<|endofprogram|>])

Figure 3: The prompt template for generating TinyGSM.

of a verifier, which we describe in this section.

### 4.1 Learning from synthetic data

We finetune the Phi-1.5 125M, 350M and 1.3B models on our TinyGSM from Section 3, and in particular, the 1.3B model reaches $\mathbf{6 8 . 2 \%}$ accuracy. ${ }^{4}{ }^{5}$ We use the Adam optimizer with FP16 during training, with a linear warm-up and a maximum learning rate of 1e-4, a weight decay of 0.01 , and an effective batch size of 1024 . The finetuning phase takes up to $20 \mathrm{k}$ steps in total. As shown in Figure 1, even without verifiers, our models are already competitive to models of size from 7B and larger. As an anecdote, an earlier and worse performing version of our Phi-GSM 1.3B model gets $94 \%$ (or $82.5 \%$ from $350 \mathrm{M}$ at pass $@ 32$, whereas the $750 \mathrm{M}$ CodeT5+ model (Wang et al., 2023) gets $73.8 \%$ (or $70.5 \%$ from $220 \mathrm{M}$ ) at pass $@ 100$.

### 4.2 Improving small models with a verifier

While sufficient synthetic data can significantly boost model performance, the performance is still below $70 \%$. Does further improvement necessitate larger model and more data then? There may be two concerns: First, there may be a diminishing return in adding extra parameters and data; for instance, while there is a $10 \%$ increase in performance when increasing from around one third of the final size of TinyGSM to two thirds, the final one third of the data provided only marginal gain. Moreover, even if the small language model is able to fully match the quality of the synthetic data, GPT-3.5-turbo itself can only achieves $77.4 \%$ test accuracy on GSM8K, which seemingly poses a limit on the performance of any models distilling from its generations.

In this section, we show that the use of a verifier can be an effective strategy orthogonal to introducing more and better data, and can even help SLMs exceed the accuracy of GPT-3.5-turbo generations. The main observation that the best of multiple generations significantly outperforms a single generation. These generations could be low-temperature generations from different checkpoints of a single run, where taking the best out of generations from 5 checkpoints of (an early version of) our 350M model reaches $75 \%$ accuracy, similar to findings in temporal ensembling (Laine \& Aila, 2016) and snapshot ensembles (Huang et al., 2017). ${ }^{6}$ The generations could also be from high-temperature generations based on a single checkpoint; for instance, the pass@32 accuracy of our 1.3B model is $94 \%$.

This suggests a promising direction of leveraging multiple generations: we can obtain a great performance boost if we are able to identify the best generation. This idea is effective yet natural: The probabilistic nature of the[^1]generative process naturally leads to the fact that multiple generations of a language model are more likely to contain a correct solution than a single one. Empirically, it has been widely observed that pass@ $k$ accuracy, namely, the accuracy by taking the best of $k$ generations, is often much higher than pass@1. The main challenge is that without knowing the labels, the definition of "best" is usually unclear. A workaround is to apply some form of self-selection, such as by choosing the one with the highest logit or the most consistent solution (Wang et al., 2022; Li et al., 2022). There is, however, a notable limitation: generations can be consistent and confident yet inaccurate, making the self-consistency approach through majority voting less effective (Li et al., 2022).

Given these observations and inspired by findings in (Cobbe et al., 2021), we propose to use a separate verifier for selecting candidate generations. For each base generation SLM, we train a verifier to predict whether a generation is a correct solution to the given question. During inference, we generate multiple candidate generations using temperature sampling, and select the one with the highest verifier score.

Training data The training data consists of the SLM's generations on the labele GSM8K training set questions, paired with the binary labels indicating whether a generation leads to the correct numerical answer. We sample 48 generations for each training set question. The binary label for each generation is based on the final execution result and the ground truth label only, and we do not verify the correctness of intermediate steps. Note that this is the only time where the GSM8K training set is directly utilized in training.

| Verfier model size | Base generation model size |  |  |
| :---: | :---: | :---: | :---: |
|  | $125 \mathrm{M}$ | $350 \mathrm{M}$ | $1.3 \mathrm{~B}$ |
| $125 \mathrm{M}$ | 68.9 | 68.8 | 71.7 |
| $350 \mathrm{M}$ | 67.3 | 71.3 | 78.3 |
| $1.3 \mathrm{~B}$ | 76.1 | 79.2 | $\mathbf{8 1 . 5}$ |

![](https://cdn.mathpix.com/cropped/2024_05_26_c345b9b4b12f3a43a806g-07.jpg?height=268&width=830&top_left_y=1037&top_left_x=1081)

Figure 4: Pass@1 results on GSM8K test set with verifiers. For each test question, we sample 48 candidate answers from the base generation model, from which we submit the one with highest verifier score as the final answer. The verifier's score on a candidate answer is determined using its score on the last token.

Training setup The verifier is trained with a sequence-to-sequence task, where we use the binary label on the entire sequence to supervise each token. We find this approach improves consistently over training with a sequence classification task (i.e. only predicting a binary label on the entire sequence). The verifier model is initialized to be the same as the SLM, with an additional prediction head shared across all positions. All network parameters are updated during verifier training, which significantly outperforms alternatives where only the prediction head is updated, or where the network is trained from scratch.

Checkpoint and data diversity The diversity of the training data proves to be important, even if this may result in slightly sacrificing the data quality ${ }^{7}$ : we find that including data generated using various temperatures and different checkpoints improves the performance of the verifier. In particular, the training data for the verifier is generated from checkpoints at $1 \mathrm{k}, 6 \mathrm{k}$ and $12 \mathrm{k}$ steps, and both training and testing data use a mixture of data generated with temperature 0.5 and 0.7 . Compared to training with generations from a single checkpoint, the use of multiple checkpoints improves the performance from $78.9 \%$ to $81.5 \%$.

Note also that in general, while we train one verifier for each generative model, verifiers transfer reasonably well across models. For example, our 1.3B model was able to reach $78.1 \%$ with a $350 \mathrm{M}$ verifier trained on generations from a $350 \mathrm{M}$ model.[^2]

## Question:

Sophia is thinking of taking a road trip in her car, and would like to know how far she can drive on a single tank of gas. She has traveled 100 miles since last filling her tank, and she needed to put in 4 gallons of gas to fill it up again. The owner's manual for her car says that her tank holds 12 gallons of gas. How many miles can Sophia drive on a single tank of gas?

## Generations:

## 1) correct:

distance_traveled $=100$

gas_used $=4$

tank_size $=12$

miles_per_gallon= distance_traveled / gas_used

result = miles_per_gallon * tank_size

## 2) wrong:

```
distance_traveled = 100
gas_put_in = 4
gas_tank_size = 12
result = (distance_traveled
    + gas_put_in) / (gas_tank_size -
    gas_put_in)
```


## 3) correct:

```
miles_traveled = 100
gallons_added = 4
total_gallons = 12
miles_per_gallon = miles_traveled //
    gallons_added
total_miles = (total_gallons -
    gallons_added) * miles_per_gallon +
    miles_traveled
result = total_miles
```

Figure 5: Visualization of the verifier's token-level predictions. Text colors denote the verifier's prediction scores: correct, potentially correct, potentially wrong, and wrong. In all the three examples, the verifier's final prediction (on the last token) aligns with the groundtruth labels. In generation 1) and 2) the verifier's token-level scores appear to be interpretable and aligned with human assessment. However, in generation 3), the scores are rather strange. This suggests the verifier relies on special patterns of the model generations that may not be unversaly generalizable, even though its final predictions are fairly reliable.

Generation model size vs verifier size In Figure 4, we present results from a cross-examination of various generation model sizes + verifier model sizes. Interestingly, while the best accuracy is achieved with configuration with largest sizes, the verifier size seems to play a bigger role than the generation model size. The effect of model size scaling is surprisingly mild: as shown in Table 1, increasing the base generation model from 125M (Phi-1.5-tiny) to 1.3B (Phi-1.5) only gives a $6 \%$ boost. On the other hand, the verifier seems to be much more parameter efficient. For example, $125 \mathrm{M}$ generation model $+1.3 \mathrm{~B}$ verifier can achieve $76.1 \%$, while $1.3 \mathrm{~B}$ generation model $+125 \mathrm{M}$ verifier gets only $71.7 \%$ Figure 4 .

## 5 Robustness and decontamination

### 5.1 Contamination test

While we never use the GSM8K test set during training, TinyGSM consists entirely of synthetic data generated by GPT models, which may be contaminated since GPT-3.5-turbo may have been exposed to the test set during its own training, which would have led to some generated synthetic samples being replicating part of the test set. To prevent contamination, we decontaminate TinyGSM by checking for n-gram matches. We use $n=13$ following standard practices (Brown et al., 2020; Wei et al., 2021; Du et al., 2022), ${ }^{8}$ and remove punctuation and numbers before computing the matching. Out of the $11.0 \mathrm{M}$ unique synthetic questions ${ }^{9}, 22$ questions have a nonzero 13 -gram match with the test set, and $38 \mathrm{k}$ questions (i.e. around $0.35 \%$ of the full set) have non-zero 8 -gram matches. Examples of 13 -gram matches are provided in Appendix A.2.[^3]

### 5.2 Evaluation on SVAMP

| Verfier model size | Base generation model size |  |  |
| :---: | :---: | :---: | :---: |
|  | $125 \mathrm{M}$ | $350 \mathrm{M}$ | $1.3 \mathrm{~B}$ |
| $125 \mathrm{M}$ | 63.2 | 70.0 | 72.2 |
| $350 \mathrm{M}$ | 64.6 | 68.7 | 72.3 |
| $1.3 \mathrm{~B}$ | 74.1 | 79.0 | 75.6 |

Figure 6: SVAMP test accuracies.

For evaluating robustness of our models, we test on the SVAMP (Simple Variations on Arithmetic Math word Problems) dataset (Patel et al., 2021), consisting of 1000 math word problem questions with a focus on arithmetics. SVAMP constructed by applying certain types of variations to a set of base questions. Even though the base questions are generally considered easier than GSM8K ${ }^{10}$, the variations may often confuse LLMs, thus making it a challenging benchmark for robustness. Our 1.3B model achieves $75.6 \%$ on SVAMP without further finetuning, indicating the robustness of the model.

## 6 Discussions

In this work, we showed a simple approach that enabled a 1.3B generation model to achieve $81.5 \%$ on the GSM8K dataset, setting a new state-of-the-art for small language models and raising the performance curve for scaling. Our approach consists of two simple steps: 1) collecting TinyGSM, a GPT-3.5 generated synthetic dataset which we will fully release, and 2) using a verifier that scores how likely a generation is correct, whose quality is boosted by utilizing diverse generations. Our results provide positive evidence that small language models have more potentials to be unlock and can be used for efficient. For future directions,

- Leveraging different formats: TinyGSM uses Python code as solutions, inspired by the observation that language models tend to struggle at calculations. However, we found that different solution formats, i.e. code versus natural language, can be complementary: while code helps circumvent errors related to execution or calculation, it tends to perform worse at questions that require equation solving, likely due to the fact that the Python syntax does not naturally support equations. Properly combining both formats has the potential to further boost performance.
- The effect of verifier size: Our results show that given a budget on the model size, scaling the verifier may be a more efficient use of the parameters. This counters our intuition that verification is an easier task than generation (which involves search), though there might be connections to findings in GAN training where the size of discriminator (Arora et al., 2018). Exploring the parameter efficiency in a generation model versus a verifier is an interesting future direction.


## References

Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G. Baraniuk. Self-consuming generative models go mad. arXiv preprint arXiv: $2307.01850,2023$.

Sanjeev Arora, Andrej Risteski, and Yi Zhang. Do GANs learn the distribution? some theory and empirics. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id= BJehNfW0-.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv: $2310.10631,2023$.[^4]

Adam Block, Dylan J. Foster, Akshay Krishnamurthy, Max Simchowitz, and Cyril Zhang. Butterfly effects of sgd noise: Error amplification in behavior cloning and autoregression. arXiv preprint arXiv: $2310.11428,2023$.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract. $h t m l$.

Stephanie C. Y. Chan, Adam Santoro, Andrew Kyle Lampinen, Jane X. Wang, Aaditya K Singh, Pierre H. Richemond, J. Mcclelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. Neural Information Processing Systems, 2022. doi: 10.48550/arXiv.2205.05055.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv: Arxiv-2110.14168, 2021.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen S. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, ICML 2022, 17 -23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 5547-5569. PMLR, 2022. URL https://proceedings.mlr.press/v162/du22c.html.

Benjamin L. Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Pareto frontiers in neural feature learning: Data, compute, width, and luck. arXiv preprint arXiv: 2309.03800, 2023.

Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv: Arxiv-2305.07759, 2023.

Google Gemini Team. Gemini: A family of highly capable multimodal models.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv: 2309.17452, 2023.

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv preprint arXiv: 2305.15717, 2023.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need. arXiv preprint arXiv: $2306.11644,2023$.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS Datasets and Benchmarks, 2021.

Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, J. Hopcroft, and Kilian Q. Weinberger. Snapshot ensembles: Train 1, get m for free. International Conference on Learning Representations, 2017.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv: $2310.06825,2023$.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies, pp. 1152-1157, 2016.

S. Laine and Timo Aila. Temporal ensembling for semi-supervised learning. International Conference on Learning Representations, 2016.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier. arXiv preprint arXiv: 2206.02336, 2022.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step, 2023.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv: 2308.09583, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv: 2303.17651, 2023 .

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. Annual Meeting of the Association for Computational Linguistics, 2020. doi: $10.18653 / \mathrm{v} 1 / 2020$. acl-main. 92.

Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. arXiv preprint arXiv: 1902.03393, 2019.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason, 2023.

Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partially-correct solutions. In The Eleventh International Conference on Learning Representations, 2023a. URL https://openreview.net/forum? id=4D4TSJE6-K.

Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partially-correct solutions. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview. net/forum? id=4D4TSJE6-K.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv: $2203.13474,2022$.

OpenAI. Gpt-4 technical report, 2023.

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori B. Hashimoto. Proving test set contamination in black box language models. arXiv preprint arXiv: 2310.17623, 2023.

Arkil Patel, S. Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? North American Chapter Of The Association For Computational Linguistics, 2021. doi: 10.18653/V1/2021. NAACL-MAIN. 168.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, E. Chi, Nathanael Scharli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. International Conference on Machine Learning, 2023a. doi: $10.48550 /$ arXiv. 2302.00093.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. arXiv preprint arXiv: 2310.16789, 2023b.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The curse of recursion: Training on generated data makes models forget. arXiv preprint arXiv: 2305.17493, 2023.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523-19536, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: $2307.09288,2023$.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv: 2211.14275, 2022.

Xuezhi Wang, Jason Wei, D. Schuurmans, Quoc Le, E. Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. International Conference on Learning Representations, 2022. doi: $10.48550 /$ arXiv. 2203.11171.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D. Q. Bui, Junnan Li, and Steven C. H. Hoi. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv: 2305.07922, 2023.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, A. Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. International Conference on Learning Representations, 2021.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, E. Chi, Tatsunori Hashimoto, Oriol Vinyals, P. Liang, J. Dean, and W. Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022a. doi: 10.48550/arXiv.2206.07682.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, E. Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. Neural Information Processing Systems, 2022b.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. arXiv preprint arXiv: 2212.09561, 2022.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Kamalika Chaudhuri,

Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 23965-23998. PMLR, 2022. URL https://proceedings.mlr.press/v162/wortsman22a. html.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. Decomposition enhances reasoning via self-evaluation guided decoding. arXiv preprint arXiv: 2305.00633, 2023.

Fei Yu, Anningzhe Gao, and Benyou Wang. Outcome-supervised verifiers for planning in mathematical reasoning. arXiv preprint arXiv: 2311.09724, 2023a.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv: $2309.12284,2023 \mathrm{~b}$.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv: 2309.05653, 2023.

Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Qizhe Xie. Automatic model selection with large language models for reasoning. arXiv preprint arXiv: 2305.14333, 2023.

Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. ARXIV.ORG, 2023. doi: 10.48550/arXiv.2304.09797.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv: 2308.07921, 2023.
</end of paper 0>


<paper 1>
# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models 

Zixiang Chen* ${ }^{* \dagger}$ Yihe Deng*‡ Huizhuo Yuan*§ Kaixuan Ji ${ }^{* \S}$ Quanquan Gu ${ }^{*}$


#### Abstract

Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional humanannotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM's performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents. Codes are available at https://github.com/uclaml/SPIN.


## 1 Introduction

Large Language Models (LLMs) have began a groundbreaking era in artificial general intelligence (AGI), demonstrating extraordinary capabilities across a wide range of domains that require intricate reasoning and specialized knowledge. These models excel in areas such as mathematical reasoning/problem solving (Cobbe et al., 2021; Wei et al., 2022; Lewkowycz et al., 2022), code generation/programming (Chen et al., 2021; Austin et al., 2021; Li et al., 2022), text generation (Bubeck[^0]et al., 2023; Anil et al., 2023; Touvron et al., 2023), summarization and creative writing, among others. A significant advancement in LLMs is the post-pretraining alignment with the more desirable behaviors (Mishra et al., 2021; Victor et al., 2022; Chung et al., 2022; Thoppilan et al., 2022), a process often reliant on the costly human-annotated data. Typical alignment methods include Supervised Fine-Tuning (SFT) (Ouyang et al., 2022; Tunstall et al., 2023a) based on human demonstrations, and Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020; Bai et al., 2022a) based on human preferences.

All the aforementioned alignment methods require a substantial volume of human annotated data. Therefore, there is increasing interest in developing fine-tuning methods that can effectively utilize human data, thereby streamlining the alignment process. This motivates us to study fine-tuning LLMs without the need for additional human-annotated data beyond the fine-tuning dataset. Our study is also related to the broader goal of converting weak models to strong models without the requirement for extra training data, which is of central interest in machine learning that can be traced back to the boosting algorithms (Kearns and Valiant, 1994; Schapire, 1990; Freund, 1995; Freund and Schapire, 1997). The self-training algorithm (Vapnik, 1999; Grandvalet and Bengio, 2004; Lee, 2013) has also been proved to be able to convert weak learners to strong learners in mixture models without the need for additional labeled data (Frei et al., 2022; Kou et al., 2022). However, the pursuit of autonomously enhancing a weak LLM without external guidance is both intriguing and understudied. This raises the following question:

Can we empower a weak LLM to improve itself without acquiring additional human annotated data?

In this paper, we answer this question affirmatively. Inspired by the success of self-play mechanisms (Samuel, 2000) in games, exemplified by AlphaGo Zero (Silver et al., 2017b), AlphaZero (Silver et al., 2017a), with historical roots traced back to TD-Gammon (Tesauro et al., 1995), we propose to convert a weak LLM to a strong one through the lens of self-play, where the model is enhanced by playing against itself without requiring any direct supervision. In particular, we propose a novel fine-tuning method called Self-Play fIne-tuNing (SPIN), which begins from a supervised fine-tuned model. SPIN allows the LLM to engage in self-play, eliminating the need for an expert annotator such as a human or more advanced LLMs like GPT-4. In detail, with the LLM from previous iteration $t$ denoted by $p_{\boldsymbol{\theta}_{t}}$, we employ it to generate responses $\mathbf{y}^{\prime}$ to the prompts $\mathbf{x}$ in the human-annotated SFT dataset. The subsequent objective is to find a new LLM $p_{\theta_{t+1}}$, capable of distinguishing the responses $\mathbf{y}^{\prime}$ generated by $p_{\boldsymbol{\theta}_{t}}$ from the responses $\mathbf{y}$ generated by humans. This process can be seen as a two-player game: the main player, or the new LLM $p_{\boldsymbol{\theta}_{t+1}}$, seeks to discern between the responses of the opponent player $p_{\theta_{t}}$ and human-generated responses, while the opponent, or the old LLM $p_{\theta_{t}}$, generates responses as similar as possible to those in the human-annotated SFT dataset. The new LLM $p_{\boldsymbol{\theta}_{t+1}}$ is obtained by fine-tuning the old one $p_{\boldsymbol{\theta}_{t}}$ to prefer responses from $p_{\text {data }}$ over $p_{\boldsymbol{\theta}_{t}}$, resulting in a distribution $p_{\boldsymbol{\theta}_{t+1}}$ that is more aligned with $p_{\text {data }}$. In the next iteration, the newly obtained LLM $p_{\boldsymbol{\theta}_{t+1}}$ becomes the opponent for response generation, with the self-play process aiming for the LLM to eventually converge to $p_{\boldsymbol{\theta}^{*}}=p_{\text {data }}$, so that the strongest possible LLM can no longer differentiate the responses generated by its previous version and those generated by the human.

Interestingly, our method exhibits similarity with the recently introduced direct preference optimization (DPO) method (Rafailov et al., 2023), with the notable distinction being the self-play nature of our method. Consequently, our approach stands out by eliminating the need for extra human preference data, a requirement present in the DPO method. Additionally, the self-play mechanism in our method resembles the idea of generative adversarial networks (GAN) (Goodfellow
et al., 2014; Arjovsky et al., 2017), albeit that both the discriminator (main player) and the generator (the opponent) in our method are instances of the same LLM from different iterations. Theoretically, we prove that our method converges when the distribution of the LLM is identical to the target data distribution, i.e., $p_{\boldsymbol{\theta}_{t}}=p_{\text {data }}$. Our experimental results on zephyr-7b-sft-full (Tunstall et al., 2023a), a fine-tuned LLM based on Mistral-7B (Jiang et al., 2023), show that while continued training using SFT on its own SFT dataset Ultrachat200k (Ding et al., 2023) reaches a performance plateau or even diminished evaluation scores, our method consistently improves zephyr-7b-sft-full across successive iterations while leveraging only a $50 \mathrm{k}$ subset of Ultrachat200k dataset. Ultimately, SPIN effectively improves the base model's average score from 58.14 to $\mathbf{6 3 . 1 6}$ on the HuggingFace Open LLM Leaderboard (Beeching et al., 2023) with remarkable $10 \%+$ improvement in scores on GSM8k and TruthfulQA, and from 5.94 to $\mathbf{6 . 7 8}$ on MT-Bench (Zheng et al., 2023). Notably, SPIN achieves results that are even comparable to models trained on additional $62 \mathrm{k}$ preference dataset (Tunstall et al., 2023a) on Open LLM leaderboard and MT-Bench.

Concurrent to our work, Singh et al. (2023) proposed the use of synthetic data with binary feedback in self-training, reducing the reliance on human data. In contrast, our approach eliminates the need for additional binary feedback from humans or an extra reward model thanks to the self-play mechanism. Additionally, Burns et al. (2023) employed a weak LLM model as the guidance to train stronger LLMs in a fashion of weak-to-strong generation. Unlike Burns et al. (2023), which necessitates both a weak supervisor and a strong model, our SPIN operates effectively with a single LLM.

Notation. We use lowercase letters and lowercase boldface letters to denote scalars and vectors, respectively. We use $[N]$ to denote the index set $\{1, \ldots, N\}$. In the function space, let $\mathcal{F}$ be the function class. The symbol $q_{\text {data }}$ designates the target data distribution, while $p$ represents the conditional probability of LLM's response (i.e., LLM policy).

## 2 Related Work

Self-Play. Self-play (Samuel, 1959; Tesauro et al., 1995), where the algorithm learns by playing against itself, has gained notable attention due to its effectiveness in multi-agent reinforcement learning (MARL). This method involves agents engaging in interactions with copies of themselves, enabling an increasing level of challenge and complexity within the learning environment. A fundamental work in the field of self-play is AlphaGo Zero (Silver et al., 2017b), which demonstrated exceptional performance against human players using a self-play learning scheme. Subsequent research has expanded upon the concept of self-play, exploring various adaptations and implementations (Anthony et al., 2017; Lanctot et al., 2017; Bansal et al., 2018; Hernandez-Leal et al., 2018; Muller et al., 2019; Vinyals et al., 2019). Our method takes the self-play approach akin to AlphaGo Zero, which can convert a weak model to a strong one without additional human-annotated data. While the effectiveness of self-play in MARL is well-established, to our knowledge, our work is the first to apply this approach to the enhancement of LLMs.

Synthetic Data for LLMs. In the context of supervised fine-tuning (SFT) of LLMs, humancrafted data has proven to be a remarkably effective source that enhances the performance of LLMs on tasks such as code generation (Roziere et al., 2023; Yang et al., 2023) and mathematical reasoning (Yuan et al., 2023; Luo et al., 2023). While human data typically exhibits high quality, acquiring sufficient amount of such data poses a challenge in cost. In light of this consideration, the use of synthetic data has become increasingly popular and considered as a proxy for human data. This approach primarily leverages advanced LLMs such as the GPT series (Radford et al., 2019;

Brown et al., 2020; OpenAI, 2023) as the guidance to generate high-quality data (Josifoski et al., 2023; Taori et al., 2023; Chiang et al., 2023; Li et al., 2023). Recent research has also highlighted the rephrasing capability of LLMs in prompting for better LLM response (Deng et al., 2023; Prasad et al., 2023) as well as augmenting synthetic data for more effective SFT (Yu et al., 2023; Liu et al., 2023). In contrast to prior studies that utilized more advanced models for synthetic data generation when pre-training or fine-tuning a target model, our approach directly generates synthetic data from the target model itself.

## 3 Problem Setting and Preliminaries

We consider a Large Language Model (LLM) parameterized by $\boldsymbol{\theta}$ and denoted by $p_{\boldsymbol{\theta}}$. The model takes as input a sequence $\mathbf{x}=\left[x_{1}, \ldots, x_{n}\right]$, commonly referred to as the prompt, to generate the corresponding response $\mathbf{y}=\left[y_{1}, \ldots, y_{m}\right]$. The response $\mathbf{y}$ is therefore considered as a sample from the conditional probability distribution $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$. In LLMs, $x_{i}$ and $y_{j}$ represent individual tokens from a predetermined vocabulary within the sequences $\mathbf{x}$ and $\mathbf{y}$, respectively. The autoregressive model $p_{\boldsymbol{\theta}}$ generates tokens sequentially for a given position, leveraging only the sequence of previously generated tokens. This model therefore constitutes a Markov process, where the conditional probability distribution $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})$ can be expressed through a decomposition as follows:

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\prod_{j=1}^{m} p_{\boldsymbol{\theta}}\left(y_{j} \mid \mathbf{x}, \mathbf{y}_{<j}\right)
$$

where $\mathbf{y}_{<1}$ is null and $\mathbf{y}_{<j}=\left[y_{1}, \ldots, y_{j-1}\right]$ for $j=2, \ldots, m$. In the following, we review two major fine-tuning methods for LLMs: supervised fine-tuning and reinforcement learning (RL) fine-tuning.

### 3.1 Supervised Fine-Tuning

Supervised fine-tuning (SFT) is employed to tailor a pre-trained LLM to specific downstream tasks, leveraging relatively smaller dataset of labeled examples in comparison to the large-scale pre-training data (Ouyang et al., 2022; Yu et al., 2023). In this context, we consider a specific task where the prompts, denoted by $\mathbf{x}$, are derived from a specified distribution $q(\cdot)$. The notation $p_{\text {data }}(\cdot \mid \mathbf{x})$ then represents the probability distribution of the associated high-quality responses $\mathbf{y}$ from the training data. Consequently, SFT involves training the LLM to minimize the following negative log-likelihood loss associated with these distributions,

$$
\begin{equation*}
L_{\mathrm{SFT}}(\boldsymbol{\theta})=-\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})\right] \tag{3.1}
\end{equation*}
$$

It should be noted that excluding $\mathbf{x} \sim q(\cdot)$ from the expectation term yields the typical crossentropy loss, expressed as $-\mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})\right] . L_{\mathrm{SFT}}(\boldsymbol{\theta})$ attains its minimum when the model's predictive distribution $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})$ aligns perfectly with the distribution of the labeled high-quality responses $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$.

Consequently, the LLM after SFT is anticipated to generate responses that closely resemble those from $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. This procedure is therefore expected to significantly enhance the model's performance in generating appropriate responses for a specific task.

### 3.2 RL Fine-Tuning

RL fine-tuning (Christiano et al., 2017; Bai et al., 2022a; Gao et al., 2023a) offers another method for enhancing the specific capabilities of general-purpose pre-trained models. Typically, RL fine-tuning is employed subsequent to SFT to achieve improved alignment for LLMs (Tunstall et al., 2023a).

For a given sequence pair $(\mathbf{x}, \mathbf{y})$, RL fine-tuning necessitates a deterministic reward function $r(\mathbf{x}, \mathbf{y})$. The higher the reward $r(\mathbf{x}, \mathbf{y})$, the better the response $\mathbf{y}$ is to the given prompt $\mathbf{x}$. The objective of the RL fine-tuning process is then to maximize the following objective function:

$$
L_{\mathrm{RL}}(\boldsymbol{\theta})=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})}[r(\mathbf{x}, \mathbf{y})]-\lambda \mathbb{E}_{\mathbf{x} \sim q(\cdot)} \mathrm{KL}\left(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}) \| p_{\mathrm{ref}}(\cdot \mid \mathbf{x})\right)
$$

where the Kullback-Leibler (KL) regularization enforces the new model $p_{\boldsymbol{\theta}}$ to be close to the reference model $p_{\text {ref }}$, and $\lambda>0$ is the regularization parameter to control the deviation of the new model $p_{\boldsymbol{\theta}}$ from the reference model $p_{\text {ref }}$. In practice, the reference model $p_{\text {ref }}$ is often initialized as the supervised fine-tuned model. The inclusion of KL regularization is vital for preventing excessive deviation from the reference model, which in turn reduces the risk of mode collapse.

Meanwhile, the primary challenge in RL fine-tuning lies in finding a good reward function. Typically, this function requires training on a preference dataset. The compilation of such a dataset demands significant resources, often involving comprehensive evaluations either by human annotators, i.e., reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Bai et al., 2022a) or strong AI agents, i.e., reinforcement learning from AI feedback (RLAIF) (Bai et al., 2022b).

## 4 Method

In this section, we introduce a new fine-tuning method for enhancing the performance of LLMs without relying on additional human or AI feedback. Consider a high-quality supervised fine-tuning (SFT) dataset $S_{\mathrm{SFT}}=\{(\mathbf{x}, \mathbf{y})\}_{i=1}^{n}$, which are sampled from the marginal distribution $q(\mathbf{x})$ and $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Given a supervised fine-tuned LLM $p_{\boldsymbol{\theta}_{0}}$, further application of the SFT approach in (3.1) with $S_{\mathrm{SFT}}$ will be ineffective and potentially lead to worse performance. In addition, without human and/or AI feedback, it becomes infeasible to acquire a preference dataset for RL fine-tuning (e.g., RLHF and RLAIF). This hinders the application of RL fine-tuning techniques.

We evaluate $p_{\boldsymbol{\theta}_{0}}$ against $S_{\mathrm{SFT}}$, where $p_{\boldsymbol{\theta}_{0}}$ is the LLM achieved by SFT using (3.1). We notice a persistent quality gap between the groundtruth response $\mathbf{y}$ from $S_{\mathrm{SFT}}$ and the LLM-generated response $\mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ (refer to Figure 1). This disparity indicates that there is still room for improvement over $p_{\boldsymbol{\theta}_{0}}$. Therefore, we propose to use the synthetic data generated by the LLM to enhance LLM's performance starting from $p_{\boldsymbol{\theta}_{0}}$ iteratively.

### 4.1 Self-Play Fine-Tuning (SPIN)

Let us consider a two-player game, where the main player's objective is to distinguish the responses generated by the LLM and those generated by the human. Meanwhile, the opponent's role is to generate responses that are indistinguishable from the human's responses. The core of our method is the self-play mechanism, where both the main player and the opponent are the same LLM, but from different iterations. More specifically, the opponent is the old LLM from the previous iteration, and the main player is the new LLM to be learned in the current iteration.

In iteration $t+1$, the opponent is the LLM from the previous iteration, denoted by $p_{\theta_{t}}$, which generates responses $\mathbf{y}^{\prime}$ for those prompts $\mathbf{x}$ in the SFT dataset according to $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$. Our method, therefore, consists of the following two steps at iteration $t+1$ : (1) training the main player, and (2) updating the opponent player.

Training the Main Player. We begin with illustrating how we expect a main player is trained to distinguish LLM responses from human responses. Motivated by integral probability metric (IPM) (Müller, 1997), we formulate our objective function such that the main player $f_{t+1}$ maximizes the

## Chosen: ground truth completion

In Southampton, what is the most popular form of transportation for commuters?

I don't have access to current data. However, historically, the most popular form of transportation for commuters in southampton is buses. The city also has a good network of trains and taxis.

## Rejected: model generation @Iter 0

In Southampton, what is the most popular form of transportation for commuters?

$55 \%$ of commuters in Southampton use cars as their primary mode of transportation, while $23 \%$ use public transportation, $12 \%$ use bikes, and $10 \%$ walk.

## Model generation @Iter 1

I do not have access to the latest updates and statistics of southampton. However, based on previous reports, the most popular form of transportation for commuters in southampton is the public transport network, which includes buses, trains, and ferries. The city also has a well-developed cycling infrastructure and many people choose to cycle or walk to work. Additionally, private cars and taxis are also commonly used for commuting.

Figure 1: Example of ground truth completion compared to the fine-tuned model generation at iteration 0 and 1 . We can observe that the model generation at iteration 0 , although fluent, incorrectly quantifies transportation preferences with specific percentages that are potentially hallucinations. The model generation at iteration 1 provides a qualitative summary of the transportation forms at Southampton without specific percentage, aligning more closely with the ground truth while adding more details.

expected value gap between the target data distribution $p_{\text {data }}$ and the opponent player's distribution $p_{\boldsymbol{\theta}_{t}}$ :

$$
\begin{equation*}
f_{t+1}=\underset{f \in \mathcal{F}_{t}}{\operatorname{argmax}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right] \tag{4.1}
\end{equation*}
$$

where $\mathcal{F}_{t}$ is a sequence of highly expressive function classes that we will determine in later deduction. The subscript $t$ in $\mathcal{F}_{t}$ is due to that the function class is dependent on $p_{\boldsymbol{\theta}_{t}}$. Given such a $f_{t+1}$ and a response sequence $\mathbf{y}$ to the prompt $\mathbf{x}$, the value of $f_{t+1}(\mathbf{x}, \mathbf{y})$ reflects the main player's degree of belief that $\mathbf{y}$ originates from $p_{\text {data }}$ rather than $p_{\boldsymbol{\theta}_{t}}$. Ideally, the main player $f_{t+1}$ should yield a high value when $\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})$ and a low value when $\mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$, where $p_{\boldsymbol{\theta}_{t}}$ is the opponent's distribution. Instead of solving (4.1), we can also solve the following more general optimization problem,

$$
\begin{equation*}
f_{t+1}=\underset{f \in \mathcal{F}_{t}}{\operatorname{argmin}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \tag{4.2}
\end{equation*}
$$

where $\ell(\cdot)$ is a loss function that is both monotonically decreasing and convex. For example, a linear loss function $\ell(t)=-t$ reduces (4.2) to the minimization version of (4.1). However, the use of a linear loss function results in an unbounded objective value, which, during continuous training, leads to a negative infinite value of $f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$ on the opponent player's responses. Therefore, in our work, we choose the logistic loss function $\ell(t):=\log (1+\exp (-t))$ for its non-negativity, smoothness, and exponentially decaying tail as $t \rightarrow \infty$. Such a choice of loss function aids in preventing the excessive growth in the absolute value of $f$.

Updating the Opponent Player. Previously we have discussed the training of $f_{t+1}$ given the opponent player's distribution $p_{\boldsymbol{\theta}_{t}}$. Now suppose we have optimized our main player $f_{t+1}$ that can distinguish $p_{\text {data }}$ from $p_{\boldsymbol{\theta}_{t}}$, within a certain function class $\mathcal{F}_{t}$, we elaborate how we get parameter $\boldsymbol{\theta}_{t+1}$
of the opponent player. Specifically, when presented with two responses $\mathbf{y}$ and $\mathbf{y}^{\prime}$ to the same prompt $\mathbf{x}, f_{t+1}$ assesses the values $f_{t+1}(\mathbf{x}, \mathbf{y})$ and $f_{t+1}\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$. It then infers that the response with the higher value is from the real data distribution $p_{\text {data }}$ and the response with lower value is attributed to the LLM $p_{\boldsymbol{\theta}_{t}}$. Subsequently, the objective of the opponent player is to find a better LLM that generates responses indistinguishable from $p_{\text {data }}$ for the main player. This is achieved by maximizing the expected value $\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p(\cdot \mid \mathbf{x})}\left[f_{t+1}(\mathbf{x}, \mathbf{y})\right]$. In addition, to prevent excessive deviation of $p_{\boldsymbol{\theta}_{t+1}}$ from $p_{\theta_{t}}$ and stabilize the self-play, we incorporate a Kullback-Leibler (KL) regularization term. Putting these together gives rise to the following optimization problem:

$$
\begin{equation*}
\underset{p}{\operatorname{argmax}} \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p(\cdot \mid \mathbf{x})}\left[f_{t+1}(\mathbf{x}, \mathbf{y})\right]-\lambda \mathbb{E}_{\mathbf{x} \sim q(\cdot)} \mathrm{KL}\left(p(\cdot \mid \mathbf{x}) \| p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})\right), \tag{4.3}
\end{equation*}
$$

where $\lambda>0$ is the regularization parameter. Notably, (4.3) has a closed-form solution $\widehat{p}(\cdot \mid \mathbf{x})$ :

$$
\begin{equation*}
\widehat{p}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \exp \left(\lambda^{-1} f_{t+1}(\mathbf{x}, \mathbf{y})\right) \tag{4.4}
\end{equation*}
$$

It is worth noting that $\widehat{p}(\cdot \mid \mathbf{x})$ is not guaranteed to be belong to the LLM space $\left\{p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}) \mid \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\}$. Since we hope that the closed-form solution $\widehat{p}$ in the probability space can be realized by an LLM with parameter $\boldsymbol{\theta}$, i.e., $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\widehat{p}(\mathbf{y} \mid \mathbf{x})$, solving for $p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \exp \left(\lambda^{-1} f_{t+1}(\mathbf{x}, \mathbf{y})\right)$ gives $f_{t+1}(\mathbf{x}, \mathbf{y})=\lambda \cdot \log \frac{p_{\theta}(\cdot \mid \mathbf{x})}{p_{\theta_{t}}(\cdot \mid \mathbf{x})}$. This suggests the following function class $\mathcal{F}_{t}$ for $f_{t+1}$ :

$$
\begin{equation*}
\mathcal{F}_{t}=\left\{\left.\lambda \cdot \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{\mathrm{t}}(\mathbf{y} \mid \mathbf{x})}} \right\rvert\, \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\} \tag{4.5}
\end{equation*}
$$

where $\boldsymbol{\Theta}$ is the parameter space of LLMs being considered. Given the choice of $\mathcal{F}_{t}$ in (4.5), optimizing (4.2) gives $f_{t+1}$ parameterized by $\boldsymbol{\theta}_{t+1}$ in the following form:

$$
\begin{equation*}
f_{t+1}(\mathbf{x}, \mathbf{y})=\lambda \cdot \log \frac{p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{\mathrm{t}}}(\mathbf{y} \mid \mathbf{x})} \tag{4.6}
\end{equation*}
$$

Substituting (4.6) into (4.4) yields $\widehat{p}(\mathbf{y} \mid \mathbf{x})=p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$. In other words, $\boldsymbol{\theta}_{t+1}$ learned from (4.2) is exactly the LLM parameter for our ideal opponent selection.

End-to-end Training Objective. We integrate the previously discussed two steps into a single end-to-end training objective with an update rule of $\boldsymbol{\theta}_{t+1}$. Specifically, plugging (4.5) into (4.2) arrives at the update rule $\boldsymbol{\theta}_{t+1}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$, where $L_{\mathrm{SPIN}}$ is the training objective defined as follows

$$
\begin{equation*}
L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\mathrm{data}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \tag{4.7}
\end{equation*}
$$

We summarize the iterative self-play process of our method SPIN as follows,

$$
\cdots \rightarrow \underbrace{p_{\theta_{t}}(\cdot \mid \mathbf{x})}_{\text {Opponent Player at } t} \rightarrow \underbrace{\lambda \cdot \log \frac{p_{\theta_{t+1}}(\cdot \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\cdot \mathbf{x})}}_{\text {Main Player at } t+1} \rightarrow \underbrace{p_{\boldsymbol{\theta}_{t+1}}(\cdot \mid \mathbf{x})}_{\text {Opponent Player at } t+1} \rightarrow \cdots
$$

Namely, the opponent player chosen from the previous iteration $t$ is employed to train the main player at iteration $t+1$, resulting in the LLM parameterized by $\boldsymbol{\theta}_{t+1}$. Then we determine the next opponent player at iteration $t+1$ by directly copying the LLM parameter $\boldsymbol{\theta}_{t+1}$, which is then used in training the main player at iteration $t+2$. The detailed algorithm is presented in Algorithm 1 .

```
Algorithm 1 Self-Play Fine-Tuning (SPIN)
    Input: $\left\{\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)\right\}_{i \in[N]}$ : SFT Dataset, $p_{\boldsymbol{\theta}_{0}}$ : LLM with parameter $\boldsymbol{\theta}_{0}, T$ : Number of iterations.
    for $t=0, \ldots, T-1$ do
        for $i=1, \ldots N$ do
            Generate synthetic data $\mathbf{y}_{i}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}\left(\cdot \mid \mathbf{x}_{i}\right)$.
        end for
        Update $\boldsymbol{\theta}_{t+1}=\operatorname{argmin}_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} \sum_{i \in[N]} \ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{i} \mid \mathbf{x}_{i}\right)}{p_{\boldsymbol{\theta}_{+}}\left(\mathbf{y}_{i} \mid \mathbf{x}_{i}\right)}-\lambda \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}_{i}^{\prime} \mid \mathbf{x}_{i}\right)}{p_{\boldsymbol{\theta}_{+}}\left(\mathbf{y}_{i}^{\prime} \mid \mathbf{x}_{i}\right)}\right)$
    end for
    Output: $\boldsymbol{\theta}_{T}$.
```

Remark 4.1. (4.7) bears resemblance to direct preference optimization (DPO) (Rafailov et al., 2023) for RL fine-tuning. However, SPIN exhibits significant distinctions with DPO. Specifically, SPIN is applied to supervised fine-tuning (SFT) and relies solely on the SFT dataset, represented by pairs $(\mathbf{x}, \mathbf{y})$. In sharp contrast, DPO is designed for RL fine-tuning and necessitates a preference dataset, represented by $\left(\mathbf{x}, \mathbf{y}_{w}, \mathbf{y}_{l}\right)$, where $\mathbf{y}_{w}$ and $\mathbf{y}_{l}$ denote the winner (chosen) and loser (rejected) responses, respectively. DPO demands that, at the instance level, $\mathbf{y}_{w}$ is superior to $\mathbf{y}_{l}$. In comparison, our method requires that, at the distribution level, the target $p_{\text {data }}$ should be distinguishable from the weak LLM $p_{\boldsymbol{\theta}}$ before it becomes a strong one. In terms of algorithm design, DPO implements a single-iteration approach, while our method facilitates an iterative self-play strategy, as outlined in Algorithm 1.

## 5 Theoretical Analysis

In this section, we provide a theoretical analysis for Algorithm 1 in Section 4. Under monotonicity and convexity assumption of the objective function $\ell$, we show that the global optimum is obtained if and only if parameter $\boldsymbol{\theta}_{t}$ generates data distribution. We summarize our assumptions as follows:

Assumption 5.1. The loss function $\ell(t): \mathbb{R} \rightarrow \mathbb{R}$ is monotonically decreasing, i.e., $\forall t, \ell^{\prime}(t) \leq 0$ and satisfies $\ell^{\prime}(0)<0$. In addition, $\ell(t)$ is a convex function.

Assumption 5.1 holds for a wide range of loss functions commonly used in machine learning, including correlation loss $\ell(t)=1-t$, hinge loss $\ell(t)=\max (0,1-t)$, exponential loss $\ell(t)=\exp (-t)$ and logistic loss $\ell(t)=\log (1+\exp (-t))$. Under Assumptions 5.1, we present the following theorem, which is pivotal in understanding the optimization dynamics of our method.

Theorem 5.2. Under Assumption 5.1, suppose there exists $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, then we have that

- (Sufficiency) If $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, then $\boldsymbol{\theta}_{t}$ is the global minimum of (4.7) for any $\lambda \geq 0$.
- (Necessity) If $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x}) \neq p_{\text {data }}(\cdot \mid \mathbf{x})$, there exists an appropriately chosen $\lambda$, such that $\boldsymbol{\theta}_{t}$ is not the global minimum of (4.7).

Remark 5.3. Theorem 5.2 suggests that under certain conditions, the optimization process of our method naturally stops at the point $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})=p_{\text {data }}(\cdot \mid \mathbf{x})$, implying the effectiveness of our approach in aligning the LLM's distribution with the target data distribution. Moreover, Theorem 5.2 also indicates that the optimization process only stops when the global optimality is achieved, i.e., the LLM's distribution aligns with the target data distribution.

For the logistic loss function $\ell(t)=\log (1+\exp (-t))$, the following theorem gives a more precise characterization of the opponent player, enabling a better understanding of SPIN.

Theorem 5.4. Consider the choice of logistic loss $\ell(t)=\log (1+\exp (-t))$ in SPIN. Suppose that $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$ lies in the LLM space $\left\{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \mid \boldsymbol{\theta} \in \boldsymbol{\Theta}\right\}$ and $\boldsymbol{\theta}_{t+1}$ is global minimum of $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$, then the opponent player at iteration $t+1$ satisfies

$$
p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}
$$

Remark 5.5. According to Theorem 5.4, the model update from $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ to $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$ tends to increase the probability $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})$ when $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ is less than $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$, and decrease it when $p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ is greater than $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Thus, Theorem 5.4 further confirms that our method's optimization process naturally converges to the point where $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ equals $p_{\text {data }}(\cdot \mid \mathbf{x})$. The update of the opponent player is controlled by $\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$, which is regulated by the factor $1 / \lambda$. A smaller $\lambda$ results in a larger change of the opponent player, while a larger $\lambda$ leads to a smaller change. Therefore, as $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})$ approaches $p_{\mathrm{data}}(\cdot \mid \mathbf{x})$, increasing $\lambda$ enhances the stability of LLM training. This observation aligns with (4.3), where $\lambda$ is the regularization parameter of the KL regularization that is employed to control the deviation of the opponent player.

## 6 Experiments

This section provides a detailed empirical analysis of SPIN. Our findings highlight several key points: (1) SPIN markedly enhances model performance across a wide range of evaluation benchmarks by breaking the limit of SFT; (2) even without introducing new human annotated data, SPIN at iteration 0 achieves performance on par to DPO training that utilizes even more data; (3) iterative training is a necessary component in SPIN as it breaks the limit of multi-epoch training.

### 6.1 Experiment Setup

Model and Datasets. In this study, we adopt zephyr-7b-sft-full as our base model. This model derives from the pre-trained Mistral-7B (Jiang et al., 2023) and has been further fine-tuned on the SFT dataset Ultrachat $200 \mathrm{k}^{1}$ by HuggingFace. Ultrachat200k represents a high-quality $200 \mathrm{k}$ subset of the larger UltraChat (Ding et al., 2023) corpus, which comprises approximately $1.4 \mathrm{M}$ dialogues produced using OpenAI's Turbo APIs. From UltraChat200k, We randomly sample 50k prompts and use the base model to generate the synthetic responses. We subsequently follow the optimization method described in Section 4.1 for further training. In multiple iterations, we leverage the synthetic data from the most recent iteration and add to the newly generated synthetic data, therefore resulting in a synthetic dataset size of $50 \mathrm{k}$ at iteration 0 and $100 \mathrm{k}$ at iteration 1,2 and 3 . At each iteration, we train our model for 2 epochs.

Evaluation. We employed the widely used Huggingface Open LLM Leaderboard (Beeching et al., 2023) as our evaluation benchmark, using the same Language Model Evaluation Harness library (Gao et al., 2023b). This leaderboard encompasses 6 different datasets, each focusing on a a specific capability of LLMs. Collectively, these datasets provide a thorough assessment framework, evaluating LLMs on commonsense reasoning (Arc (Clark et al., 2018), HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2021)), multi-task language understanding (MMLU(Hendrycks et al., 2020)), human falsehood mimic (TruthfulQA (Lin et al., 2021)) and math problem solving[^1]

(GSM8k (Cobbe et al., 2021)). In evaluation, the language models are prompted with few-shot in-context examples and the question. We follow the standard approach and report the average score across all datasets. In Table 1, we detail the evaluation setting adopted by both the leaderboard and our experiments. We leave further implementation details to Appendix B.

Table 1: Detailed information of HuggingFace Open LLM Leaderboard. For each evaluation dataset, we present the number of few-shot examples and metric adopted for evaluation.

| Datasets | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| \# few-shot | 25 | 0 | 5 | 5 | 10 | 5 |
| Metric | acc_norm | mc2 | acc | acc | acc_norm | acc |

### 6.2 SPIN Effectively Improves Benchmark Performance

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-10.jpg?height=653&width=683&top_left_y=926&top_left_x=713)

Figure 2: The average score of SPIN at different iterations on the HuggingFace Open LLM leaderboard datasets. For "SFT", we report the performance of our base model zephyr-7b-sft-full, which has been fine-tuned on the same dataset we use to generate synthetic data.

We demonstrate the effectiveness of SPIN using HuggingFace Open LLM Leaderboard as a wide range of evaluation. In Table 2, we compare the performance of our fine-tuned model by SPIN after iterations 0 to 3 with the base model zephyr-7b-sft-full. We can observe that SPIN exhibits remarkable effectiveness in improving the model's performance by further leveraging the SFT dataset, on which the base model has already been fully fine-tuned. At iteration 0 , where model responses are generated from zephyr-7b-sft-full, we observe an overall improvement of $2.66 \%$ on the average score. The improvement is particularly significant on the TruthfulQA and GSM8k benchmarks, with improvement exceeding $5 \%$ and $10 \%$ respectively. At iteration 1, we employ the LLM model from iteration 0 to generate new responses for SPIN, adhering to the procedure outlined in Algorithm 1. This iteration yields further enhancements of $1.32 \%$ on average, and especially significant on the Arc Challenge and TruthfulQA benchmarks. Subsequent iterations continue this trend of incremental improvement across various tasks. Meanwhile, the improvement at iteration $t+1$ is naturally smaller than that at iteration $t$. As the iterative training progresses, the degree of improvement gradually
approaches zero, suggesting that the model has reached a limiting point in the last iteration.

Table 2: Test performance of SPIN based on zephyr-7b-sft-full across HuggingFace Open LLM Leaderboard datasets. We also denote the average improvement over last iteration in the Average column.

| Model | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 58.14 |
| SPIN iteration 0 | 63.40 | 49.18 | 72.69 | 35.10 | 84.38 | 60.03 | $60.80_{(+2.66)}$ |
| SPIN iteration 1 | 65.19 | 55.17 | 72.30 | 35.78 | 84.96 | 59.34 | $62.12_{(+1.32)}$ |
| SPIN iteration 2 | 65.96 | 54.91 | 73.56 | 38.06 | 85.41 | 59.93 | $62.97_{(+0.85)}$ |
| SPIN iteration 3 | 65.87 | 54.90 | 73.72 | 38.97 | 85.54 | 59.99 | $63.16_{(+0.19)}$ |

Comparison with DPO. zephyr-7b-beta is a model derived from zephyr-7b-sft-full, trained with DPO on approximately $62 \mathrm{k}$ preference data. This data, the UltraFeedback Binarized dataset(Cui et al., 2023) ${ }^{2}$, comprises both chosen and rejected completions evaluated by GPT-4. We note that, DPO requires either human input or advanced language model feedback to determine the preference, making data generation a rather expensive procedure. In contrast, our SPIN only requires the initial model itself. Moreover, unlike DPO which requires new data source, our method exclusively leverages the existing SFT dataset. In Figure 3, we show the performance comparison of SPIN at iterations 0 and 1 (employing 50k SFT data) with DPO training, from the same SFT checkpoint. We can observe that, while DPO leverages more data from new sources, SPIN based on the existing SFT data can already achieve comparable average performance to DPO training at iteration 0. From iteration 1, SPIN even surpasses the performance of DPO on the leaderboard benchmark.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-11.jpg?height=496&width=1526&top_left_y=1411&top_left_x=294)

Figure 3: Performance comparison with DPO training across the six benchmark datasets. Self-play at iteration 0 achieves comparable performance to DPO training with $62 \mathrm{k}$ new data. At iteration 1 , self-play has already surpassed DPO training on the majority of datasets.

### 6.3 Ablation Studies

In this subsection, we examine the effect of synthetic dataset size and training epochs within an iteration. Our analysis demonstrates the effectiveness of the synthetic data used by SPIN compared to the SFT data, as well as the necessity of iterative training in SPIN. Furthermore, to comprehensively[^2]assess the performance improvements of SPIN, we perform additional evaluations on benchmark tasks distinct from those in the Open LLM leaderboard.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-12.jpg?height=647&width=967&top_left_y=365&top_left_x=579)

Figure 4: The scaling effect of training size of SPIN compared to SFT on the average score of Open LLM Leaderboard. For SPIN, we consider training data of sizes $14 \mathrm{k}, 26 \mathrm{k}$ and $50 \mathrm{k}$ where the larger dataset contains the smaller dataset. The starting point for SPIN (with x-axis 0) is the zephyr-7b-sft-full checkpoint, which has been fine-tuned on Ultrachat200k for 1 epoch. We report the model performance trained for 1 epoch with SPIN on the varying sizes of dataset. We additionally compare with SFT, where we fine-tune Mistral-7B on Ultrachat200k for 3 consecutive epochs and report the model performance at the first epoch as the starting point (with x-axis 0 ).

Training Size. We investigate the effect of varying training data size on the performance of SPIN. In Figure 4, we demonstrate the effect of training size for SPIN during iteration 0 and additionally compare with SFT with the full original dataset. Specifically, for the SFT baseline, we fully fine-tune Mistral-7B on Ultrachat200k for three epochs and report first epoch performance as the starting point (with x-axis 0) in the figure for SFT. For SPIN, we report the zephyr-7b-sft-full checkpoint as the starting point, which has also been fine-tuned on Ultrachat200k for one epoch. We select the training size of SPIN at iteration 0 to be $14 \mathrm{k}, 26 \mathrm{k}$, and $50 \mathrm{k}$ and generate the data accordingly, ensuring that the larger dataset encompasses the smaller dataset. The performance of SPIN was then evaluated after 1 epoch of self-play fine-tuning for each training size. We can observe that, while SPIN results in notable improvement with increasing training sizes, SFT on further epochs 2 and 3 fails to yield more than $1 \%$ improvement. Lastly, in Table 3, we also show the performance of SFT from zephyr-7b-sft-full on Ultrachat200k for one epoch. While self-play fine-tuning with synthetic data from zephyr-7b-sft-full effectively improves its performance, simply fine-tuning it again on the SFT data leads to degraded performance, as similarly observed in Figure 4.

Iterative Training v.s. Training for More Epochs. We further study the training within iteration 0 and compare with the performance achieved in iteration 1 , particularly contrasting the test performance obtained from extended training duration with that from next iteration. Figure 5 depicts the performance trajectory of the model trained using SPIN over multiple epochs at iteration 0 . It is evident that the most substantial improvement occurs during the first two epochs, followed by only modest gains in subsequent epochs. Notably, SPIN exhibits robustness and stability; extending

Table 3: Test performance of zephyr-7b-sft-full fine-tuned on Ultrachat200k for 1 more epoch across HuggingFace Open LLM benchmark datasets. SFT fails to further leverage the fine-tuning data for performance enhancement and even results in degraded performance.

| Model | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 58.14 |
| SFT epoch 1 | 57.76 | 44.39 | 75.77 | 25.85 | 81.69 | 57.89 | 57.23 |

the training duration does not diminish performance but rather maintains a rather consistent level. Nevertheless, the observation suggests an inherent limitation to the performance achievable within a single iteration, thereby underscoring the necessity for iterative training. As shown by the test performance achieved at iteration 1 in the figures, extending the training in iteration 0 fails to reach the performance comparable to iteration 1 .

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=505&width=1529&top_left_y=932&top_left_x=295)

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=414&width=504&top_left_y=953&top_left_x=304)

(a) Arc Challenge accuracy.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=414&width=512&top_left_y=953&top_left_x=801)

(b) TruthfulQA score.

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-13.jpg?height=420&width=509&top_left_y=950&top_left_x=1312)

(c) Average score.

Figure 5: The SPIN training dynamics of zephyr-7b-sft-full on the $50 \mathrm{k}$ synthetic data with regard to the number of training epochs during iteration 0 . We can observe that iterative training is pivotal as training for more epochs during iteration 0 reaches a limit and cannot surpass iteration 1.

Further Investigation on More Tasks. Here, we further investigate the performance of SPIN on a broader variety of tasks, including MT-Bench (Zheng et al., 2023), Big-Bench (bench authors, 2023) and OpenBookQA (Mihaylov et al., 2018) in addition to the Open LLM Leaderboard tasks. Specifically, we use the following tasks from Big-Bench-Hard for a more comprehensive evaluation, including Causal Judgment (causal reasoning), Sports Understanding (commonsense reasoning) and Formal Fallacies (logical reasoning). In Table 4, we show the resulting scores of SPIN on MT-Bench as well as those tasks from Big-Bench. In Figure 6, we detail the model performances on MT-Bench with regard to different types of questions. We can see a notably robust improvement in the performance of SPIN on various tasks besides the HuggingFace Benchmark, without major degradation. Notably, on MT-Bench, the model fine-tuned by SPIN has surpassed the performance of vicuna-13b-v1.5 (Chiang et al., 2023) with a score of 6.57 .

## 7 Conclusion and Discussion

This paper introduces a novel fine-tuning method SPIN, to convert a weak LLM to a strong LLM by unleashing the full power of human-annotated data. Central to this method is a self-play mechanism,

Table 4: Test performance on other reasoning benchmark datasets for SPIN at different iterations and zephyr-7b-sft-full. We report the average score for MT-Bench and the accuracy score for Big Bench datasets under standard few-shot CoT evaluation. On OpenBookQA, we report acc_norm with 1-shot example as used in Anil et al. (2023). As similar to Open LLM Leaderboard evaluation, we observe a steady improvement in performance on the other benchmark tasks, with no significant degradation.

| Model | MT-Bench | BB-causal | BB-formal | BB-sports | OpenBookQA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| zephyr-7b-sft-full | 5.94 | 56.15 | 49.6 | 96.0 | 45.4 |
| SPIN iteration 0 | $6.46_{(+0.52)}$ | 57.75 | 51.6 | 95.2 | 46.8 |
| SPIN iteration 1 | $6.65_{(+0.19)}$ | 58.82 | 51.2 | 95.2 | 47.2 |
| SPIN iteration 2 | $6.78_{(+0.13)}$ | 59.36 | 51.2 | 94.4 | 47.6 |

![](https://cdn.mathpix.com/cropped/2024_06_04_fec1ce860d9e0e804dcfg-14.jpg?height=623&width=1003&top_left_y=865&top_left_x=558)

Figure 6: Model performance on MT-Bench. We compare SPIN across different iterations with the base SFT model. Starting from iteration 1, our fine-tuned model by SPIN robustly outperforms the SFT checkpoint on all evaluation aspects.

wherein a main player (the LLM) is fine-tuned to differentiate the responses of opponent player (the LLM from previous iteration) from the target data distribution, and the LLM is iteratively aligned with the target data distribution. Therefore, SPIN facilitates the LLM's iterative self-evaluation and enhancement through self-play. In comparison to supervised fine-tuning and RL fine-tuning methods, SPIN enables the LLM to self-improve without additional human data or feedback from stronger LLMs. Empirical results demonstrate that SPIN significantly enhances LLM performance across diverse benchmarks, even outperforming models trained with additional human data or AI feedback.

Limitation and Future Work. Our theoretical results demonstrate that the optimization process of SPIN converges if and only if the LLM's distribution aligns with $p_{\text {data }}$. Therefore, our study focuses on a fixed target data distribution generated by humans, which inherently imposes a ceiling on the performance of fine-tuned LLM. Exploring the dynamically changing target data distribution is an important direction to overcome this limitation and elevate the LLM's performance beyond this ceiling or even to a super-human level. Moreover, considering the resource demands of synthetic data generation, another promising avenue for further exploration is to reduce the volume of required
synthetic data.

## A Further Related Work

Curriculum Learning. In deep learning, it has been observed that training models using data samples arranged in a strategically meaningful order can lead to improved performance compared to training on randomly shuffled data. This approach is commonly known as curriculum learning (Bengio et al., 2009; Soviany et al., 2022). Initial studies in curriculum learning introduced efficient algorithms that adhere to an 'easy-to-hard' progression (Spitkovsky et al., 2009; Kumar et al., 2010; Lee and Grauman, 2011; Zhang et al., 2015). In the field of Natural Language Processing (NLP), criteria such as sentence length and term frequency are commonly utilized (Cirik et al., 2016; Zhang et al., 2018; Liu et al., 2018). More recent developments include the application of curriculum learning algorithms in multi-modal learning (Liu et al., 2021; Wu et al., 2022). Our work shares a similar idea to curriculum learning, wherein the training data evolves iteratively-beginning with responses that are easy to distinguish from human-annotated data and gradually progressing to more challenging instances.

Generative Adversarial Learning. Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) represent a distinct class of generative models, characterized by their unique adversarial process. To enhance training stability and data quality, Mao et al. (2017) introduced the Least Squares GAN, employing a least squares loss function for the discriminator. A significant advancement in GANs involves the use of Integral Probability Metrics (IPM) (Müller, 1997), particularly highlighted in the development of Wasserstein GAN by Arjovsky et al. (2017). This model employs IPM in its loss design, enhancing training stability. Since then, IPMs have become popular in the design of GANs (Mroueh and Sercu, 2017; Gulrajani et al., 2017), particularly in constraining the discriminator to a specific function class, thereby preventing it from overpowering the generator. Furthermore, Jolicoeur-Martineau (2018) generalized IPM-based GANs by introducing relativistic discriminator and proposed Relativistic GAN. It is worth noting that the objective function defined in our (4.2) is similar to Relativistic GAN (Jolicoeur-Martineau, 2018) and reduces to an IPM framework in Wasserstein GAN (Arjovsky et al., 2017) with a linear loss. However, our approach differs in both the choice of the function class and the training procedure. Inspired by GAN, Cheng et al. (2023) proposed an adversarial learning framework named Adversarial Preference Optimization (APO) that trains the LLM and a reward model in an adversarial game. Our method is also related to Generative Adversarial Imitation Learning (GAIL) (Ho and Ermon, 2016), which trains separate discriminator and policy networks in each iteration for imitation learning. In contrast to the above methods, SPIN relies on self-play where both the main player and the opponent player are the same LLM from two consecutive iterations.

## B Experiment Details

## B. 1 Hyperparameters and Implementation Details

We use the Alignment Handbook library (Tunstall et al., 2023b) as the codebase for our selfplay fine-tuning method SPIN, which includes DeepSpeed ZeRO-3 (Rajbhandari et al., 2020) and FlashAttention-2 (Dao, 2023) to reduce training cost. We train our models with RMSProp (Hinton et al., 2012) optimizer with no weight decay for all iterations as commonly used in fine-tuning LLMs for alignment, with a global batch size of $64,10 \%$ warmup steps and bfloat16 precision. We set the peak learning rate to be $5 \mathrm{e}-7$ for iterations 0 and 1 , and decay this peak learning rate to $1 \mathrm{e}-7$ for
iteration 2 and 3 as we are approaching the end of self-play fine-tuning. Lastly, we choose $\beta=0.1$ and max sequence length to be 2048 tokens as in Tunstall et al. (2023b). We note that at the last iteration (iter-3) where the model is close to convergence, we increase the value of $\beta$ to 5.0. We use the Accelerate library (Gugger et al., 2022) to generate our synthetic data using distributed inference with multiple GPUs with a global batch size of 64 . We consider the prompting template "\#\#\# Instruction: $\{$ prompt\} $\} \mathrm{n} \backslash \mathrm{n} \# \# \#$ Response: " as commonly used in Taori et al. (2023). For Ultrachat200k containing multi-round conversations, we only sample the first round as our prompt and ground truth completion pairs.

## B. 2 Generation Examples

In Tables 5 and 6, we further provide the generation examples of our fine-tuned model by SPIN from different iterations. We can observe an improvement in response quality as compared to the generation of the SFT checkpoint. Meanwhile, the model generations at higher iterations typically becomes more concise than iteration 0 and resemble the ground truth completion better.

## C Proof of Theorems in Section 5

## C. 1 Proof of Theorem 5.2

Proof of Theorem 5.2. To begin with, we prove the "Sufficiency" in Theorem 5.2. Since $p_{\text {data }}(\cdot \mid \mathbf{x})=$ $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})$, by symmetry property of $\mathbf{y}$ and $\mathbf{y}^{\prime}$, we have for any $\boldsymbol{\theta} \in \boldsymbol{\Theta}$ that

$$
\begin{aligned}
2 L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)= & \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& +\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y}^{\prime} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
= & \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right. \\
& \left.\quad+\left(\gamma \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}-\gamma \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)\right] \\
\geq & 2 \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}} \cdot(\cdot \mathbf{x})}\left[\ell \left(\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right.\right. \\
& \left.\left.\quad+\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}-\frac{\gamma}{2} \log \frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)\right] \\
= & 2 \ell(0),
\end{aligned}
$$

where the inequality is due to Jensen's inequality (recalling that $\ell$ is convex in Assumption 5.1). Therefore, we have that $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right) \geq \ell(0)=L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}_{t}, \boldsymbol{\theta}_{t}\right)$, which means that $\boldsymbol{\theta}_{t}$ is the global optimum of (4.7). As a consequence, the gradient at the point $\boldsymbol{\theta}_{t}$ is zero, which concludes $\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_{t}$.

Next, we prove the "Necessity". Define $g(\lambda)$ as follows:

$$
g(\lambda)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\mathrm{data}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right]
$$

Then we have $g(0)=\ell(0)$ and

$$
g^{\prime}(0)=\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell^{\prime}(0)\left(\log \frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}-\log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right]
$$

Table 5: Generation example of our fine-tuned model by SPIN at different iterations.

| Prompt | How does the economic growth of Horsham compare to nearby towns? |
| :--- | :--- |
| Ground truth | I do not have access to the latest economic data on horsham and nearby <br> towns. However, you can check the latest economic statistics and reports <br> provided by the local government or the official websites of horsham and <br> nearby towns for a detailed comparison of their economic growth. |

zephyr-7b=sft-full $\mid$ According to the given material, Horsham was growing "significantly" faster than nearby towns, indicating that it was economically thriving. $\backslash \mathrm{n}$ However, the given material does not specify the exact rates or levels of economic growth of Horsham compared to nearby towns. Without this information, it is challenging to determine whether the differences in economic growth between Horsham and nearby towns were significant or minimal.

| Iteration 0 | I do not have access to the latest economic data for horsham and nearby |
| :--- | :--- |

towns. However, horsham is known to be a prosperous town in west sussex, uk, with a diverse economy that includes agriculture, manufacturing, and services. The town has a strong retail sector, and it is also home to several large companies, including the global pharmaceutical company novartis. In terms of comparison with nearby towns, horsham is generally considered to be one of the more affluent towns in the region. However, the economic growth of horsham and its neighboring towns may vary depending on various factors such as industry, investment, and government policies.

Iteration 1

I do not have access to real-time data or information about the most recent economic growth of horsesham and nearby towns. However, according to the recent reports from various sources such as the office for national statistics, horsesham's economy primarily relies on retail, healthcare, education, and public administration industries. Compared to nearby towns such as lewes, chichester, and bognor regis, horsesham's economy may have a different composition and growth rate based on their specific industries and economic policies.

Iteration 2

I don't have access to the most recent economic data. However, according to a 2019 report by pwc, horsham's expected economic growth up to 2025 was projected to be around $2.3 \%$ per annum, which is slightly below the average for the south east of england (2.8\%). Neighbouring towns like crawley and bognor regis were projected to experience similar growth rates. note: this may not reflect the impact of the covid-19 pandemic that occurred in 2020 .

$$
=\ell^{\prime}(0)\left(\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x})}\left[\log \frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right]-\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\log \frac{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right]\right)
$$

Table 6: Another generation example of our fine-tuned model by SPIN at different iterations.

| Prompt | Can you clarify if Poppelsdorfer Allee 45 and the old site of the Mathe- <br> matical Institute are the same location? Answer according to: Go straight <br> to Poppelsdorfer Allee 45 and sign-in. $\$ n There is a big lecture hall at the <br> old site of the Mathematical Institute (Wegelerstr. 10, Bonn). |
| :--- | :--- |
| Ground truth | No, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are not the same location. To sign-in, one should <br> go straight to Poppelsdorfer Allee 45. However, there is a big lecture hall <br> at the old site of the Mathematical Institute. |
| zephyr-7b=sft-full | Yes, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> are the same location. |
| Iteration 0 | Yes, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are the same location. The Mathematical Institute <br> has been relocated and renamed as the Institute for Mathematics and <br> Theoretical Physics, but the building where it used to be is still referred <br> to as the old site. The big lecture hall at the old site still exists and may <br> be used for events or classes. |
| Iteration 1 | No, Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> are not the same location. While Poppelsdorfer Allee 45 is where attendees <br> should sign-in, the old site of the Mathematical Institute (located at <br> Wegelerstr. 10, Bonn) has a large lecture hall mentioned in the statement. <br> Therefore, it can be assumed that the two locations are separate and not <br> the same. |
| Iteration 2 | Poppelsdorfer Allee 45 and the old site of the Mathematical Institute <br> (Wegelerstr. 10, Bonn) are two different locations. As per the instruction, <br> you need to go straight to Poppelsdorfer Allee 45 and sign-in. However, <br> there is a lecture hall mentioned at the old site of the Mathematical <br> Institute (Wegelerstr. 10, Bonn). |

$$
\begin{aligned}
& =\ell^{\prime}(0)\left[\mathrm{KL}\left(p_{\text {data }}(\cdot \mid \mathbf{x})|| p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})\right)+\mathrm{KL}\left(p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})|| p_{\text {data }}(\cdot \mid \mathbf{x})\right)\right] \\
& <0
\end{aligned}
$$

where the last inequality is due to the condition that $\ell^{\prime}(0)<0$. Therefore, there exist a $\lambda_{0}$ such that for all $0<\lambda<\lambda_{0}$, we have $g(\lambda)<\ell(0)$. Choose $\boldsymbol{\theta}^{*}$ such that $p_{\boldsymbol{\theta}^{*}}(\mathbf{y} \mid \mathbf{x})=p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. For those $0<\lambda<\lambda_{0}$, we have that

$$
\begin{aligned}
L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}^{*}, \boldsymbol{\theta}_{t}\right) & =\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\boldsymbol{\theta}^{*}}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\boldsymbol{\theta}^{*}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\boldsymbol{\theta}^{*}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& =\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(\lambda \log \frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}-\lambda \log \frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)\right] \\
& =g(\lambda)
\end{aligned}
$$

$$
\begin{aligned}
& <g(0) \\
& =L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}_{t}, \boldsymbol{\theta}_{t}\right)
\end{aligned}
$$

where the second equality holds by the choice of $p_{\boldsymbol{\theta}^{*}}(\cdot \mid \mathbf{x})$, and the inequality holds due to the choice of $\lambda$. Therefore, we conclude that $\boldsymbol{\theta}_{t}$ is not the global optimum of (4.7) if $p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x}) \neq p_{\text {data }}(\cdot \mid \mathbf{x})$.

## C. 2 Proof Theorem 5.4

We need the following auxiliary lemma before we prove Theorem 5.4.

Lemma C.1. Suppose that $\ell(t)=\log (1+\exp (-t))$ and for $a, b>0$, the following inequality holds

$$
a \ell(t)+b \ell(-t) \geq a \log (1+b / a)+b \log (1+a / b)
$$

the equality holds if and only if $t=\log (a / b)$.

Proof of Lemma C.1. Define $g(t)=a \ell(t)+b \ell(-t)=a \log (1+\exp (-t))+b \log (1+\exp (t))$, then we have

$$
g^{\prime}(t)=-\frac{a \exp (-t)}{1+\exp (-t)}+\frac{b \exp (t)}{1+\exp (t)}=\frac{-a+b \exp (t)}{1+\exp (t)}
$$

Therefore, $g^{\prime}(t)<0$ when $t<\log (a / b), g^{\prime}(t)>0$ when $t>\log (a / b)$, which indicates that $g$ achieves it minimum at $t=\log (a / b)$ which concludes the proof.

Lemma C. 1 shows that the global minimum of $a \ell(t)+b \ell(-t)$ is achieved when $t=\log (a / b)$. Based on Lemma C.1, we can further prove that (4.2) with the logistic loss function has a closed-form solution if we ignore the constraint set $\mathcal{F}_{t}$.

Lemma C.2. Denote $p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime}, \mathbf{x}\right)=q(\mathbf{x}) \cdot p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)$ and $p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime}, \mathbf{x}\right)=q(\mathbf{x}) \cdot p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)$. $p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$,

$$
\mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \geq \log 2-\operatorname{JSD}\left(p_{+} \| p_{-}\right)
$$

where $\operatorname{JSD}\left(p_{+} \| p_{-}\right)$represents the Jensen-Shannon divergence which is defined as follows

$$
\operatorname{JSD}(p \| q)=\frac{1}{2} \mathrm{KL}\left(p \| \frac{p+q}{2}\right)+\frac{1}{2} \mathrm{KL}\left(q \| \frac{p+q}{2}\right)
$$

where $\mathrm{KL}(\cdot \| \cdot)$ is KL-divergence. JSD is always non-negative and equals zero if and only if $p_{+}$and $p_{-}$are identical. Moreover, the global minimum value $\log 2-\operatorname{JSD}\left(p_{+} \| p_{-}\right)$is achieved by $f^{*}$ if and only if,

$$
f^{*}(\mathbf{x}, \mathbf{y})=Z(\mathbf{x})+\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

where $Z(\mathbf{x})$ is any function that is possibly dependent on $\mathbf{x}$.

Proof of Lemma C.2. We rewrite the objective function in the following formula,

$$
2 \mathbb{E}_{\mathbf{x} \sim q(\cdot), \mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), \mathbf{y}^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right]
$$

$$
\begin{aligned}
& =\int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] d \mathbf{y} d \mathbf{y}^{\prime} \\
& +\int q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left[\ell\left(f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-f(\mathbf{x}, \mathbf{y})\right)\right] d \mathbf{y} d \mathbf{y}^{\prime} \\
& =\int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \ell\left(f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-f(\mathbf{x}, \mathbf{y})\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& \stackrel{(i)}{\geq} \int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \log \left(1+\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) d \mathbf{y} d \mathbf{y}^{\prime}
\end{aligned}
$$

where the inequality is due to $a \ell(t)+b \ell(-t) \geq a \log (1+b / a)+b \log (1+a / b)$ in Lemma C. 1 with $a=q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right), b=q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}), t=f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)$. The equality (i) holds if and only if the following equation holds almost surely for any $\mathbf{x}, \mathbf{y}, \mathbf{y}^{\prime}$,

$$
\begin{equation*}
f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)=\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) \tag{C.1}
\end{equation*}
$$

Equation (C.1) is equivalent to

$$
f(\mathbf{x}, \mathbf{y})-\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)-\log \left(\frac{p_{\mathrm{data}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\theta_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)
$$

holds almost surely for any $\mathbf{x}, \mathbf{y}, \mathbf{y}^{\prime}$. Therefore, the equality (i) holds if and only if there exists some $Z(\mathbf{x})$ such that

$$
f(\mathbf{x}, \mathbf{y})=Z(\mathbf{x})+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

Recall that $p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)=p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})$ and $p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)=p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \cdot p_{\text {data }}(\mathbf{y} \mid \mathbf{x})$. Then, the right-hand side of (i) can be written as

$$
\begin{aligned}
& \int q(\mathbf{x}) p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +q(\mathbf{x}) p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x}) \log \left(1+\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) p_{\boldsymbol{\theta}_{t}}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{\text {data }}\left(\mathbf{y}^{\prime} \mid \mathbf{x}\right) p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =\int p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right)+p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(1+\frac{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =2 \log 2+\int p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(\frac{1 / 2\left[p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)+p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)\right]}{p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) \\
& +p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right) \log \left(\frac{1 / 2\left[p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)+p_{+}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)\right]}{p_{-}\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}\right)}\right) d \mathbf{y} d \mathbf{y}^{\prime} \\
& =2 \log 2-\mathrm{KL}\left(p_{+} \| \frac{p_{+}+p_{-}}{2}\right)-\mathrm{KL}\left(p_{-} \| \frac{p_{+}+p_{-}}{2}\right) \\
& =2 \log 2-2 \cdot \operatorname{JSD}\left(p_{+} \| p_{-}\right)
\end{aligned}
$$

where the last equality is by the definition of JSD. This concludes the proof.

Lemma C. 2 provides a closed-form solution to (4.2) if we ignore the constraint set $\mathcal{F}_{t}$. If this closed-form solution belongs to $\mathcal{F}_{t}$, then it should also be the solution to (4.2). This observation is the key to the proof of Theorem 5.4.

Proof of Theorem 5.4. Under the condition of Theorem 5.4, there exists a $p_{\boldsymbol{\theta}}$ such that

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}
$$

Therefore, there exists a function $\widehat{Z}(\mathbf{x})$ such that

$$
\begin{equation*}
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})=\widehat{Z}(\mathbf{x}) \cdot p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda} \tag{C.2}
\end{equation*}
$$

Applying logarithm function on both side of (C.2) yields

$$
\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\mathrm{data}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=\lambda \log \left(\frac{p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right) \in \mathcal{F}_{t}
$$

By Lemma C.2, $f^{*}(\mathbf{x}, \mathbf{y})=\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ is the global minimum of the following minimization problem,

$$
\begin{equation*}
\underset{f}{\operatorname{argmin}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] . \tag{C..3}
\end{equation*}
$$

Since $f^{*} \in \mathcal{F}_{t}, f^{*}(\mathbf{x}, \mathbf{y})=\lambda \log (\widehat{Z}(\mathbf{x}))+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\theta_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ is also the global optimum of the optimization problem (4.2),

$$
\underset{f \in \mathcal{F}_{t}}{\operatorname{argmin}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\theta_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right]
$$

Therefore, we have proved that

$$
\begin{align*}
& \min _{f} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \\
& =\min _{f \in \mathcal{F}_{t}} \mathbb{E}_{\mathbf{y} \sim p_{\text {data }}(\cdot \mid \mathbf{x}), y^{\prime} \sim p_{\boldsymbol{\theta}_{t}}(\cdot \mid \mathbf{x})}\left[\ell\left(f(\mathbf{x}, \mathbf{y})-f\left(\mathbf{x}, \mathbf{y}^{\prime}\right)\right)\right] \\
& =\min _{\boldsymbol{\theta} \in \boldsymbol{\Theta}} L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right) \tag{C.4}
\end{align*}
$$

Since $\boldsymbol{\theta}_{t+1}$ is the global minimum of $L_{\mathrm{SPIN}}\left(\boldsymbol{\theta}, \boldsymbol{\theta}_{t}\right)$. Then by (C.4), $\lambda \log \left(\frac{p_{\boldsymbol{\theta}_{+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)$ should be the global minimum of problem (C.3). By Lemma C.2, there exists $Z(\mathbf{x})$ such that

$$
\lambda \log \left(\frac{p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)=Z(\mathbf{x})+\log \left(\frac{p_{\text {data }}(\mathbf{y} \mid \mathbf{x})}{p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})}\right)
$$

which leads to the result that $p_{\boldsymbol{\theta}_{t+1}}(\mathbf{y} \mid \mathbf{x}) \propto p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\left(p_{\text {data }}(\mathbf{y} \mid \mathbf{x}) / p_{\boldsymbol{\theta}_{t}}(\mathbf{y} \mid \mathbf{x})\right)^{1 / \lambda}$.

## References

Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z. et al. (2023). Palm 2 technical report. arXiv preprint arXiv:2305.10403 .

Anthony, T., Tian, Z. and Barber, D. (2017). Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems 30.

Arjovsky, M., Chintala, S. and Bottou, L. (2017). Wasserstein generative adversarial networks. In International conference on machine learning. PMLR.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q. Et al. (2021). Program synthesis with large language models. arXiv preprint arXiv:2108.07732 .

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T. et al. (2022a). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 .

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C. et al. (2022b). Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 .

Bansal, T., Pachocki, J., Sidor, S., Sutskever, I. and Mordatch, I. (2018). Emergent complexity via multi-agent competition. In International Conference on Learning Representations.

Beeching, E., Fourrier, C., Habib, N., Han, S., Lambert, N., Rajani, N., Sanseviero, O., Tunstall, L. and Wolf, T. (2023). Open llm leaderboard.

BENCH AUThors, B. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research .

Bengio, Y., Louradour, J., Collobert, R. and Weston, J. (2009). Curriculum learning. In Proceedings of the 26th annual international conference on machine learning.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. et al. (2020). Language models are few-shot learners. Advances in neural information processing systems 33 1877-1901.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S. et al. (2023). Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 .

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J. et al. (2023). Weak-to-strong generalization: Eliciting strong capabilities with weak supervision .

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G. et al. (2021). Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Cheng, P., Yang, Y., Li, J., Dai, Y. and Du, N. (2023). Adversarial preference optimization.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I. and Xing, E. P. (2023). Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality.

Christiano, P. F., Leike, J., Brown, T., Martic, M., LegG, S. and Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems 30.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S. et al. (2022). Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 .

Cirik, V., Hovy, E. and Morency, L.-P. (2016). Visualizing and understanding curriculum learning for long short-term memory networks. arXiv preprint arXiv:1611.06204 .

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C. and Tafjord, O. (2018). Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 .

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z. and Sun, M. (2023). Ultrafeedback: Boosting language models with high-quality feedback.

DAO, T. (2023). Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 .

Deng, Y., Zhang, W., Chen, Z. and Gu, Q. (2023). Rephrase and respond: Let large language models ask better questions for themselves. arXiv preprint arXiv:2311.04205 .

Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M. and Zhou, B. (2023). Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233 .

Frei, S., Zou, D., Chen, Z. and Gu, Q. (2022). Self-training converts weak learners to strong learners in mixture models. In International Conference on Artificial Intelligence and Statistics. PMLR.

Freund, Y. (1995). Boosting a weak learning algorithm by majority. Information and computation $121256-285$.

Freund, Y. and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences 55 119-139.

Gao, L., Schulman, J. and Hilton, J. (2023a). Scaling laws for reward model overoptimization. In International Conference on Machine Learning. PMLR.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'H, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., WANG, K. and Zou, A. (2023b). A framework for few-shot language model evaluation.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems 27 .

Grandvalet, Y. and Bengio, Y. (2004). Semi-supervised learning by entropy minimization. Advances in neural information processing systems 17.

Gugger, S., Debut, L., Wolf, T., Schmid, P., Mueller, Z., Mangrulkar, S., Sun, M. and Bossan, B. (2022). Accelerate: Training and inference at scale made simple, efficient and adaptable.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems $\mathbf{3 0}$.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J. (2020). Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 .

Hernandez-Leal, P., Kartal, B. and Taylor, M. E. (2018). Is multiagent deep reinforcement learning the answer or the question? a brief survey. learning 2122.

Hinton, G., Srivastava, N. and Swersky, K. (2012). Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on 142 .

Ho, J. and Ermon, S. (2016). Generative adversarial imitation learning. Advances in neural information processing systems 29.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. D. L., Bressand, F., Lengyel, G., Lample, G., Saulnier, L. et al. (2023). Mistral 7b. arXiv preprint arXiv:2310.06825 .

Jolicoeur-Martineau, A. (2018). The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734.

Josifoski, M., Sakota, M., Peyrard, M. and West, R. (2023). Exploiting asymmetry for synthetic training data generation: Synthie and the case of information extraction. arXiv preprint arXiv:2303.04132 .

Kearns, M. and Valiant, L. (1994). Cryptographic limitations on learning boolean formulae and finite automata. Journal of the ACM (JACM) 41 67-95.

Kou, Y., Chen, Z., Cao, Y. and Gu, Q. (2022). How does semi-supervised learning with pseudo-labelers work? a case study. In The Eleventh International Conference on Learning Representations.

Kumar, M., Packer, B. and Koller, D. (2010). Self-paced learning for latent variable models. Advances in neural information processing systems 23.

Lanctot, M., Zambaldi, V., Gruslys, A., Lazaridou, A., Tuyls, K., Pérolat, J., Silver, D. and Graepel, T. (2017). A unified game-theoretic approach to multiagent reinforcement learning. Advances in neural information processing systems $\mathbf{3 0}$.

LeE, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In ICML Challenges in Representation Learning Workshop.

Lee, Y. J. and Grauman, K. (2011). Learning the easy things first: Self-paced visual category discovery. In CVPR 2011. IEEE.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T. et al. (2022). Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems 35 3843-3857.

Li, Y., Bubeck, S., Eldan, R., Giorno, A. D., Gunasekar, S. and Lee, Y. T. (2023). Textbooks are all you need ii: phi-1.5 technical report.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A. et al. (2022). Competition-level code generation with alphacode. Science 378 1092-1097.

Lin, S., Hilton, J. and Evans, O. (2021). Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958 .

Liu, B., Bubeck, S., Eldan, R., Kulkarni, J., Li, Y., Nguyen, A., Ward, R. and Zhang, Y. (2023). Tinygsm: achieving $>80 \%$ on gsm8k with small language models. arXiv preprint arXiv:2312.09241 .

Liu, C., He, S., Liu, K., Zhao, J. et al. (2018). Curriculum learning for natural answer generation. In $I J C A I$.

LiU, F., Ge, S. and Wu, X. (2021). Competence-based multimodal curriculum learning for medical report generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers).

Luo, H., Sun, Q., Xu, C., Zhao, P., lou, J., Tao, C., Geng, X., Lin, Q., Chen, S. and Zhang, D. (2023). Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583 .

Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z. and Paul Smolley, S. (2017). Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision.

Mihaylov, T., Clark, P., Khot, T. and Sabharwal, A. (2018). Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

Mishra, S., Khashabi, D., Baral, C. and Hajishirzi, H. (2021). Cross-task generalization via natural language crowdsourcing instructions. arXiv preprint arXiv:2104.08773 .

Mroueh, Y. and Sercu, T. (2017). Fisher gan. Advances in neural information processing systems 30 .

MÜLLER, A. (1997). Integral probability metrics and their generating classes of functions. Advances in applied probability 29 429-443.

Muller, P., Omidshafiei, S., Rowland, M., Tuyls, K., Perolat, J., Liu, S., Hennes, D., Marris, L., Lanctot, M., Hughes, E. et al. (2019). A generalized training approach for multiagent learning. arXiv preprint arXiv:1909.12823 .

OpenAI (2023). Gpt-4 technical report.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 27730-27744.

Prasad, A., Stengel-Eskin, E. and Bansal, M. (2023). Rephrase, augment, reason: Visual grounding of questions for vision-language models. arXiv preprint arXiv:2310.05861 .

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. et al. (2019). Language models are unsupervised multitask learners. OpenAI blog 19 .

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290 .

Rajbhandari, S., Rasley, J., Ruwase, O. and He, Y. (2020). Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE.

Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J. et al. (2023). Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950 .

Sakaguchi, K., Bras, R. L., Bhagavatula, C. and Choi, Y. (2021). Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM 64 99-106.

SamulL, A. L. (1959). Some studies in machine learning using the game of checkers. IBM Journal of research and development $\mathbf{3}$ 210-229.

Samuel, A. L. (2000). Some studies in machine learning using the game of checkers. IBM Journal of research and development 44 206-226.

Schapire, R. E. (1990). The strength of weak learnability. Machine learning 5 197-227.

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T. et al. (2017a). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815 .

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A. et al. (2017b). Mastering the game of go without human knowledge. nature 550 354-359.

Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Liu, P. J., Harrison, J., Lee, J., Xu, K., Parisi, A. et al. (2023). Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585 .

Soviany, P., Ionescu, R. T., Rota, P. and Sebe, N. (2022). Curriculum learning: A survey. International Journal of Computer Vision 130 1526-1565.

Spitkovsky, V. I., Alshawi, H. and Jurafsky, D. (2009). Baby steps: How "less is more" in unsupervised dependency parsing. In NIPS 2009 Workshop on Grammar Induction, Representation of Language and Language Learning.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D. and Christiano, P. F. (2020). Learning to summarize with human feedback. Advances in Neural Information Processing Systems 33 3008-3021.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. and Hashimoto, T. B. (2023). Stanford alpaca: An instruction-following llama model.

Tesauro, G. et al. (1995). Temporal difference learning and td-gammon. Communications of the ACM $3858-68$.

Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y. et al. (2022). Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239 .

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S. et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N. et al. (2023a). Zephyr: Direct distillation of $1 \mathrm{~m}$ alignment. arXiv preprint arXiv:2310.16944 .

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rush, A. M. and Wolf, T. (2023b). The alignment handbook.

VAPniK, V. (1999). The nature of statistical learning theory. Springer science \& business media.

Victor, S., Albert, W., Colin, R., Stephen, B., Lintang, S., Zaid, A., Antoine, C., Arnaud, S., Arun, R., Manan, D. et al. (2022). Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Vinyals, O., Babuschkin, I., Chung, J., Mathieu, M., Jaderberg, M., Czarnecki, W., Dudzik, A., Huang, A., Georgiev, P., Powell, R., Ewalds, T., Horgan, D., Kroiss, M., Danihelka, I., Agapiou, J., Oh, J., Dalibard, V., Choi, D., Sifre, L., Sulsky, Y., Vezhnevets, S., Molloy, J., Cai, T., Budden, D., Paine, T., Gulcehre, C., Wang, Z., Pfaff, T., Pohlen, T., Yogatama, D., Cohen, J., McKinney, K., Smith, O., Schaul, T., Lillicrap, T., Apps, C., Kavukcuoglu, K., Hassabis, D. and Silver, D. (2019). AlphaStar: Mastering the Real-Time Strategy Game StarCraft II.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D. eT AL. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35 24824-24837.

Wu, J., Liang, Y., Akbari, H., Wang, Z., Yu, C. et al. (2022). Scaling multimodal pre-training via cross-modality gradient harmonization. Advances in Neural Information Processing Systems $3536161-36173$.

Yang, Y., Singh, A. K., Elhoushi, M., Mahmoud, A., Tirumala, K., Gloeckle, F., Rozière, B., Wu, C.-J., Morcos, A. S. and Ardalani, N. (2023). Decoding data quality via synthetic corruptions: Embedding-guided pruning of code data. arXiv preprint arXiv:2312.02418 .

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A. and LiU, W. (2023). Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284 .

Yuan, Z., Yuan, H., Li, C., Dong, G., Tan, C. and Zhou, C. (2023). Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825 .

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A. and Choi, Y. (2019). Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830 .

Zhang, D., Meng, D., Li, C., Jiang, L., Zhao, Q. and Han, J. (2015). A self-paced multipleinstance learning framework for co-saliency detection. In Proceedings of the IEEE international conference on computer vision.

Zhang, X., Kumar, G., Khayrallah, H., Murray, K., Gwinnup, J., Martindale, M. J., McNamee, P., Duh, K. and Carpuat, M. (2018). An empirical exploration of curriculum learning for neural machine translation. arXiv preprint arXiv:1811.00739 .

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. ET AL. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685 .

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P. and Irving, G. (2019). Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 .


[^0]:    *Equal contribution

    ${ }^{\dagger}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: chenzx19@cs.ucla.edu

    ${ }^{\ddagger}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: yihedeng@cs.ucla.edu

    ${ }^{8}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: hzyuan@cs.ucla.edu

    ${ }^{I}$ Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: kaixuanji@cs.ucla.edu

    ${ }^{\|}$Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail: qgu@cs.ucla.edu

[^1]:    ${ }^{1}$ https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

[^2]:    ${ }^{2}$ https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized

</end of paper 1>


<paper 2>
# Self-Rewarding Language Models 

Weizhe Yuan ${ }^{1,2} \quad$ Richard Yuanzhe Pang ${ }^{1,2} \quad$ Kyunghyun Cho ${ }^{2}$<br>Xian Li ${ }^{1} \quad$ Sainbayar Sukhbaatar ${ }^{1}$ Jing Xu ${ }^{1} \quad$ Jason Weston $^{1,2}$<br>${ }^{1}$ Meta $\quad{ }^{2}$ NYU


#### Abstract

We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While there is much left still to explore, this work opens the door to the possibility of models that can continually improve in both axes.


## 1 Introduction

Aligning Large Language Models (LLMs) using human preference data can vastly improve the instruction following performance of pretrained models [Ouyang et al., 2022, Bai et al., 2022a]. The standard approach of Reinforcement Learning from Human Feedback (RLHF) learns a reward model from these human preferences. The reward model is then frozen and used to train the LLM using RL, e.g., via PPO [Schulman et al., 2017]. A recent alternative is to avoid training the reward model at all, and directly use human preferences to train the LLM, as in Direct Preference Optimization [DPO; Rafailov et al., 2023]. In both cases, the approach is bottlenecked by the size and quality of the human preference data, and in the case of RLHF the quality of the frozen reward model trained from them as well.

In this work, we instead propose to train a self-improving reward model that, rather than being frozen, is continually updating during LLM alignment, in order to avoid this bottleneck. The key to such an approach is to develop an agent that possesses all the abilities desired during training, rather than separating them out into distinct models such as a reward model and a language model. In the same way that pretraining and multitasking training of instruction following tasks allow task transfer by training on many tasks at once [Collobert and Weston, 2008, Radford et al., 2019, Ouyang et al., 2022], incorporating the reward model into that same system allows task transfer between the reward modeling task and the instruction following tasks.

We thus introduce Self-Rewarding Language Models, that both (i) act as instruction following models generating responses for given prompts; and (ii) can generate and evaluate new instruction following examples to add to their own training set. We train these models using an Iterative DPO framework similar to that recently introduced in Xu et al. [2023].

![](https://cdn.mathpix.com/cropped/2024_06_04_16fd7e44df33d9c1bf2cg-02.jpg?height=353&width=1377&top_left_y=241&top_left_x=366)

Figure 1: Self-Rewarding Language Models. Our self-alignment method consists of two steps: (i) Self-Instruction creation: newly created prompts are used to generate candidate responses from model $M_{t}$, which also predicts its own rewards via LLM-as-a-Judge prompting. (ii) Instruction following training: preference pairs are selected from the generated data, which are used for training via DPO, resulting in model $M_{t+1}$. This whole procedure can then be iterated resulting in both improved instruction following and reward modeling ability.

Starting from a seed model, in each iteration there is a process of Self-Instruction creation whereby candidate responses are generated by the model for newly created prompts, and are then assigned rewards by that same model. The latter is implemented via LLM-as-a-Judge prompting, which can also be seen as an instruction following task. A preference dataset is built from the generated data, and the next iteration of the model is trained via DPO, see Figure 1.

In our experiments, we start with a Llama 2 70B [Touvron et al., 2023] seed model fine-tuned on Open Assistant [Köpf et al., 2023], and then perform the above training scheme. We find that not only does the instruction following performance improve from Self-Rewarding LLM alignment compared to the baseline seed model, but importantly the reward modeling ability, which is no longer fixed, improves as well. This means that the model during iterative training is able, at a given iteration, to provide a higher quality preference dataset to itself than in the previous iteration. While this effect likely saturates in real-world settings, it provides the intriguing possibility of obtaining reward models (and hence LLMs) that are superior to ones that could have been trained from the original human-authored seed data alone.

## 2 Self-Rewarding Language Models

Our approach first assumes access to a base pretrained language model, and a small amount of human-annotated seed data. We then build a model that aims to possess two skills simultaneously:

1. Instruction following: given a prompt that describes a user request, the ability to generate a high quality, helpful (and harmless) response.
2. Self-Instruction creation: the ability to generate and evaluate new instructionfollowing examples to add to its own training set.

These skills are used so that the model can perform self-alignment, i.e., they are the components used to iteratively train itself using AI Feedback (AIF).

Self-instruction creation consists of generating candidate responses and then the model itself judging their quality, i.e., it acts as its own reward model, replacing the need for an external one. This is implemented via the LLM-as-a-Judge mechanism [Zheng et al., 2023b], i.e., by formulating the evaluation of responses as an instruction following task. This self-created AIF preference data is used as a training set.

Our overall self-alignment procedure is an iterative one, which proceeds by building a series of such models, with the aim that each improves over the last. Importantly, because the model can both improve its generation ability, and act as its own reward model through the same generation mechanism, this means the reward model itself can improve through these iterations, deviating from standard practices where the reward model is fixed [Ouyang et al.,

2022]. We believe this can increase the ceiling of the potential for self-improvement of these learning models going forward, removing a constraining bottleneck.

We describe these steps in more detail below. An overview of the approach is illustrated in Figure 1.

### 2.1 Initialization

Seed instruction following data We are given a seed set of human-authored (instruction prompt, response) general instruction following examples that we use for training in a supervised fine-tuning (SFT) manner, starting from a pretrained base language model. Subsequently this will be referred to as Instruction Fine-Tuning (IFT) data.

Seed LLM-as-a-Judge instruction following data We also assume we are provided a seed set of (evaluation instruction prompt, evaluation result response) examples which can also be used for training. While this is not strictly necessary, as the model using IFT data will already be capable of training an LLM-as-a-Judge, we show that such training data can give improved performance (see Appendix A. 3 for supporting results). In this data, the input prompt asks the model to evaluate the quality of a given response to a particular instruction. The provided evaluation result response consists of chain-of-thought reasoning (a justification), followed by a final score (in our experiments out of 5). The exact prompt format we chose is given in Figure 2, which instructs the LLM to evaluate the response using five additive criteria (relevance, coverage, usefulness, clarity and expertise), covering various aspects of quality. Subsequently this will be referred to as Evaluation Fine-Tuning (EFT) data.

We use both these seed sets together during training.

### 2.2 Self-Instruction Creation

Using the model we have trained, we can make it self-modify its own training set. Specifically, we generate additional training data for the next iteration of training.

This consists of the following steps:

1. Generate a new prompt: We generate a new prompt $x_{i}$ using few-shot prompting, sampling prompts from the original seed IFT data, following the approach of Wang et al. [2023] and Honovich et al. [2023]. ${ }^{1}$
2. Generate candidate responses: We then generate $N$ diverse candidate responses $\left\{y_{i}^{1}, \ldots, y_{i}^{N}\right\}$ for the given prompt $x_{i}$ from our model using sampling.
3. Evaluate candidate responses: Finally, we use the LLM-as-a-Judge ability of our same model to evaluate its own candidate responses with scores $r_{i}^{n} \in[0,5]$ (exact prompt given in Figure 2).

### 2.3 Instruction Following Training

As previously described, training is initially performed with the seed IFT and EFT data (Section 2.1). This is then augmented with additional data via AI (Self-)Feedback.

AI Feedback Training After performing the self-instruction creation procedure, we can augment the seed data with additional examples for training, which we refer to as AI Feedback Training (AIFT) data.

To do this, we construct preference pairs, which are training data of the form (instruction prompt $x_{i}$, winning response $y_{i}^{w}$, losing response $y_{i}^{l}$ ). To form the winning and losing pair we take the highest and lowest scoring responses from the $N$ evaluated candidate responses (see[^0]

Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: <INSTRUCTION_HERE>

$<$ response $>$ RESPONSE_HERE></response>

After examining the user's instruction and the response:

- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: <total points>"

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria.

Figure 2: LLM-as-a-Judge prompt for our LLM to act as a reward model and provide self-rewards for its own model generations. The model is initially trained with seed training data of how to perform well at this task, and then improves at this task further through our self-rewarding training procedure.

Section 2.2), following Xu et al. [2023], discarding the pair if their scores are the same. These pairs can be used for training with a preference tuning algorithm. We use DPO [Rafailov et al., 2023].

### 2.4 Overall Self-Alignment Algorithm

Iterative Training Our overall procedure trains a series of models $M_{1}, \ldots, M_{T}$ where each successive model $t$ uses augmented training data created by the $t-1^{\text {th }}$ model. We thus define $\operatorname{AIFT}\left(M_{t}\right)$ to mean AI Feedback Training data created using model $M_{t}$.

Model Sequence We define the models, and the training data they use as follows:

$M_{0}$ : Base pretrained LLM with no fine-tuning.

$M_{1}$ : Initialized with $M_{0}$, then fine-tuned on the IFT+EFT seed data using SFT.

$M_{2}$ : Initialized with $M_{1}$, then trained with $\operatorname{AIFT}\left(M_{1}\right)$ data using DPO.

$M_{3}$ : Initialized with $M_{2}$, then trained with $\operatorname{AIFT}\left(M_{2}\right)$ data using DPO.

This iterative training resembles the procedure used in Pairwise Cringe Optimization and specifically is termed Iterative DPO, introduced in Xu et al. [2023]; however, an external fixed reward model was used in that work.

## 3 Experiments

### 3.1 Experimental Setup

Base Model In our experiments we use Llama 2 70B [Touvron et al., 2023] as our base pretrained model.

### 3.1.1 Seed Training Data

IFT Seed Data We use the human-authored examples provided in the Open Assistant dataset [Köpf et al., 2023] for instruction fine-tuning. Following Li et al. [2024] we use 3200 examples, by sampling only first conversational turns in the English language that are high-quality, based on their human annotated rank (choosing only the highest rank 0 ). In our experiments, we compare to a model fine-tuned from the base model using only this data via supervised fine-tuning, and refer to it as our SFT baseline.

EFT Seed Data The Open Assistant data also provides multiple ranked human responses per prompt from which we can construct evaluation fine-tuning data. We split this into train and evaluation sets, and use it to create LLM-as-a-Judge data. This is done by placing it in the input format given in Figure 2, which consists of the scoring criteria description, and the given instruction and response to be evaluated. ${ }^{2}$ For training targets, chain-of-thought justifications and final scores out of 5 are not directly provided, so we use the SFT baseline to generate such output evaluations for each input, and accept them into the training set if the ranking of their scores agrees with the human rankings in the dataset. We resample the training set by discarding some of the data that receives the most common score so that the scores are not too skewed, as we observe many samples receive a score of 4 . This results in 1,630 train and 531 evaluation examples (which do not overlap with the IFT data).

### 3.1.2 Evaluation Metrics

We evaluate the performance of our self-rewarding models in two axes: their ability to follow instructions, and their ability as a reward model (ability to evaluate responses).

Instruction Following We evaluate head-to-head performance between various models using GPT-4 [Achiam et al., 2023] as an evaluator over 256 test prompts (which we refer to as IFT test data) derived from various sources following Li et al. [2024] using the AlpacaEval evaluation prompt [Li et al., 2023]. We try the prompt in both orders comparing pairwise, and if the GPT-4 evaluations disagree we count the result as a tie. We also perform a similar evaluation with humans (authors). We additionally report results in the AlpacaEval 2.0 leaderboard format which is evaluated over 805 prompts, and compute the win rate against the baseline GPT-4 Turbo model based on GPT-4 judgments. Further, we report results on MT-Bench [Zheng et al., 2023b] a set of challenging multi-turn questions in various categories from math and coding to roleplay and writing, which uses GPT-4 to grade the model responses out of 10 . Finally we also test the models on a set of 9 NLP benchmarks: ARC-Easy [Clark et al., 2018], ARC-Challenge [Clark et al., 2018], HellaSwag [Zellers et al., 2019], SIQA [Sap et al., 2019], PIQA [Bisk et al., 2020], GSM8K [Cobbe et al., 2021], MMLU [Hendrycks et al., 2021], OBQA [Mihaylov et al., 2018] and NQ [Kwiatkowski et al., 2019].

Reward Modeling We evaluate the correlation with human rankings on the evaluation set we derived from the Open Assistant dataset, as described in Section 3.1.1. Each instruction has on average 2.85 responses with given rankings. We can thus measure the pairwise accuracy, which is how many times the order of the ranking between any given pair agrees between the model's evaluation and the human ranking. We also measure the exact match count, which is how often the total ordering is exactly the same for an instruction. We also report the Spearman correlation and Kendall's $\tau$. Finally, we report how often the responses that the model scores a perfect 5 out of 5 are rated as the highest ranked by humans.[^1]

### 3.1.3 Training Details

Instruction following training The training hyperparameters we use are as follows. For SFT we use learning rate $5.5 e-6$ which decays (cosine) to $1.1 e-6$ at the end of training, batch size 16 and dropout 0.1. We only calculate the loss on target tokens instead of the full sequence. For DPO we use learning rate $1 e-6$ which decays to $1 e-7$, batch size 16 , dropout 0.1 , and a $\beta$ value of 0.1 . We perform early stopping by saving a checkpoint every 200 steps and evaluating generations using Claude 2 [Anthropic, 2023] on 253 validation examples derived from various sources following $\mathrm{Li}$ et al. [2024]. This is evaluated pairwise against the previous step's generations using the AlpacaEval evaluation prompt format [Li et al., 2023].

Self-Instruction creation To generate new prompts we use a fixed model, Llama 2-Chat 70B with 8-shot prompting following Self-Instruct [Wang et al., 2023], where we sample six demonstrations from the IFT data and two from the model generated data, and use decoding parameters $\mathrm{T}=0.6, \mathrm{p}=0.9$. We use their prompt template for non-classification tasks and apply the same filtering techniques, including the ROUGE-L [Lin, 2004] similarity check, keyword filtering, and length filtering. Except for the prompt generation part, the other parts of the creation pipeline (generating the response, and evaluating it) use the Self-Rewarding model being trained. For candidate response generation we sample $N=4$ candidate responses with temperature $T=0.7, p=0.9$. When evaluating candidate responses, as there is variance to these scores, in our experiments we also use sampled decoding (with the same parameters) and generate these evaluations multiple (3) times and take the average. We added 3,964 such preference pairs to form the $\operatorname{AIFT}\left(M_{1}\right)$ dataset used to train $M_{2}$ via DPO, and 6,942 pairs to form $\operatorname{AIFT}\left(M_{2}\right)$ used to train $M_{3}$.

### 3.2 Results

### 3.2.1 Instruction Following Ability

Head to head performance results are provided in Figure 3.

EFT+IFT seed training performs similarly to IFT alone We find that adding the Evaluation Fine-Tuning (EFT) task to training does not impact instruction following performance compared to using Instruction Fine-Tuning (IFT) data alone with an almost equal head to head ( $30.5 \%$ wins vs. $30.9 \%$ wins). This is a positive result because it means the increased capability of a model to self-reward does not affect its other skills. We can thus use IFT + EFT training as Iteration $1\left(M_{1}\right)$ of our Self-Rewarding model, and then run further iterations.

Iteration $2\left(M_{2}\right)$ improves over Iteration $1\left(M_{1}\right)$ and SFT Baseline Iteration 2 of Self-Rewarding training $\left(M_{2}\right)$ provides superior instruction following to Iteration $1\left(M_{1}\right)$ with $55.5 \%$ wins for $M_{2}$ compared to only $11.7 \%$ for $M_{1}$ in a head to head evaluation. It provides similar gains over the SFT Baseline as well ( $49.2 \%$ wins vs. $14.5 \%$ wins). Clearly, there is a large jump in performance from $M_{1}$ to $M_{2}$ by using the preference data $\operatorname{AIFT}\left(M_{1}\right)$ provided by the reward model from Iteration 1.

Iteration $3\left(M_{3}\right)$ improves over Iteration $2\left(M_{2}\right)$ We see a further gain in Iteration 3 over Iteration 2, with $47.7 \%$ wins for $M_{3}$ compared to only $12.5 \%$ for $M_{2}$ in a head to head evaluation. Similarly, the win rate over the SFT Baseline for $M_{3}$ increases to $62.5 \%$ wins vs. $9.8 \%$, i.e., winning more often than the $M_{2}$ model did. Overall, we see large gains from $M_{2}$ to $M_{3}$ through training using the preference data $\operatorname{AIFT}\left(M_{2}\right)$ provided by the reward model from Iteration 2 .

Self-Rewarding models perform well on AlpacaEval 2 leaderboard We evaluate our models on the AlpacaEval 2.0 leaderboard format, with results given in Table 1. We observe the same findings as in the head-to-head evaluations, that training iterations yield improved win rates, in this case over GPT4-Turbo, from $9.94 \%$ in Iteration 1 , to $15.38 \%$ in Iteration 2, to $20.44 \%$ in Iteration 3. Our Iteration 3 model outperforms many existing models in this metric, including Claude 2, Gemini Pro, and GPT4 0613. We show some

![](https://cdn.mathpix.com/cropped/2024_06_04_16fd7e44df33d9c1bf2cg-07.jpg?height=1070&width=1097&top_left_y=251&top_left_x=514)

Figure 3: Instruction following ability improves with Self-Training: We evaluate our models using head-to-head win rates on diverse prompts using GPT-4. The SFT Baseline is on par with Self-Rewarding Iteration $1\left(M_{1}\right)$. However, Iteration $2\left(M_{2}\right)$ outperforms both Iteration $1\left(M_{1}\right)$ and the SFT Baseline. Iteration $3\left(M_{3}\right)$ gives further gains over Iteration 2 $\left(M_{2}\right)$, outperforming $M_{1}, M_{2}$ and the SFT Baseline by a large margin.

selected models from the leaderboard in the table. We note that many of those competing models contain either proprietary alignment data (which is typically large, e.g., over 1M annotations in Touvron et al. [2023]) or use targets that are distilled from stronger models. In contrast, our Self-Rewarding model starts from a small set of seed data from Open Assistant, and then generates targets and rewards from the model itself for further iterations of training.

Fine-grained analysis As described earlier, the overall performance of the model in AlpacaEval improves with each iteration of training. It would be interesting to break down the overall performance improvement to see exactly what type of tasks these improvements come from. Therefore, we cluster the instructions in AlpacaEval test set into different groups based on three perspectives: (1) instruction category (2) instruction complexity (3) expected response length. We achieve this by using GPT-4. The detailed statistical information of the breakdown and the prompting techniques we used for getting this breakdown can be found in Appendix A.6. Results for the instruction category are given in Figure 4, and the other two in Appendix Figure 11. From the results we can conclude that (i) Self-Rewarding models can substantially improve the win rate in most categories, but there are some tasks for which this approach does not improve, such as mathematics and logical reasoning, indicating that our current training approach mainly allows the models to better utilize their existing knowledge. (ii) Through Self-Rewarding model training, the model's win rate increases on almost all tasks of different complexity, and especially on slightly more difficult tasks (complexity of 5 , 6,7 out of 10). (iii) The models also show a steady increase in the win rate on tasks with instructions with different expected response lengths.

Table 1: AlpacaEval 2.0 results (win rate over GPT-4 Turbo evaluated by GPT-4). Self-Rewarding iterations yield improving win rates. Iteration $3\left(M_{3}\right)$ outperforms many existing models that use proprietary training data or targets distilled from stronger models.

|  |  | Alignment Targets |  |
| :--- | :--- | :--- | :--- |
| Model | Win Rate | Distilled | Proprietary |
| Self-Rewarding 70B |  |  |  |
| Iteration 1 $\left(M_{1}\right)$ | $9.94 \%$ |  |  |
| Iteration 2 $\left(M_{2}\right)$ | $15.38 \%$ |  |  |
| Iteration 3 $\left(M_{3}\right)$ | $20.44 \%$ |  |  |
| Selected models from the leaderboard |  |  |  |
| GPT-4 0314 | $22.07 \%$ |  |  |
| Mistral Medium | $21.86 \%$ |  |  |
| Claude 2 | $17.19 \%$ |  |  |
| Gemini Pro | $16.85 \%$ |  |  |
| GPT-4 0613 | $15.76 \%$ |  | $\checkmark$ |
| GPT 3.5 Turbo 0613 | $14.13 \%$ |  | $\checkmark$ |
| LLaMA2 Chat 70B | $13.87 \%$ |  |  |
| Vicuna 33B v1.3 | $12.71 \%$ |  |  |
| Humpback LLaMa2 70B | $10.12 \%$ |  |  |
| Guanaco 65B | $6.86 \%$ |  |  |
| Davinci001 | $2.76 \%$ |  |  |
| Alpaca 7B | $2.59 \%$ | $\checkmark$ |  |

![](https://cdn.mathpix.com/cropped/2024_06_04_16fd7e44df33d9c1bf2cg-08.jpg?height=759&width=946&top_left_y=1258&top_left_x=584)

Figure 4: AlpacaEval win rate breakdown for instruction categories (full names given in Appendix). Self-Rewarding models give gains across several topics, but tend to e.g. give less gains on mathematics and reasoning tasks.

Data distribution analysis We perform a t-SNE [Van der Maaten and Hinton, 2008] visualization of the $\operatorname{IFT}, \mathrm{EFT}$ and $\operatorname{AIFT}\left(M_{1}\right)$ data, shown in Appendix A.1. We find good overlap between the IFT and $\operatorname{AIFT}\left(M_{1}\right)$ examples, which is desired, while the EFT examples lie in a different part of the embedding space, which can help explain why they would not affect IFT performance. We observe that generations from $M_{1}$ on AlpacaEval have an average length of 1092 , for $M_{2}$ they are 1552 , and for $M_{3}$ they are 2552 , so the model is learning to generate longer responses, which we note may be a factor in relative performance.

![](https://cdn.mathpix.com/cropped/2024_06_04_16fd7e44df33d9c1bf2cg-09.jpg?height=513&width=1097&top_left_y=253&top_left_x=514)

Figure 5: Human evaluation results. Iterations of Self-Rewarding ( $M_{1}, M_{2}$ and $M_{3}$ ) provide progressively better head-to-head win rates compared to the SFT baseline, in agreement with the automatic evaluation results.

Table 2: MT-Bench Results (on a scale of 10). Self-Rewarding iterations yield improving scores across various categories. Math, code \& reasoning performance and iteration gains are smaller than for other categories, likely due to the makeup of the Open Assistant seed data we use.

|  | Overall <br> Score | Math, Code <br> \& Reasoning | Humanities, Extraction, <br> STEM, Roleplay \& Writing |
| :--- | :---: | :---: | :---: |
| SFT Baseline | 6.85 | 3.93 | 8.60 |
| $M_{1}$ | 6.78 | 3.83 | 8.55 |
| $M_{2}$ | 7.01 | 4.05 | 8.79 |
| $M_{3}$ | 7.25 | 4.17 | 9.10 |

Table 3: NLP Benchmarks. Self-Rewarding models mostly tend to maintain performance compared to the Llama 2 70B base model and the SFT Baseline, despite being fine-tuned on very different instruction-following prompts.

|  | ARC $(\uparrow)$ <br> challenge | HellaSwag <br> $(\uparrow)$ | GSM8K <br> $(\uparrow)$ | MMLU <br> $(\uparrow)$ | NQ <br> $(\uparrow)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Llama 2 | 57.40 | 85.30 | 56.80 | 68.90 | 25.30 |
| SFT Baseline | 55.97 | 85.17 | 50.72 | 69.76 | 34.35 |
| $M_{1}$ | 57.51 | 84.99 | 60.27 | 69.34 | 35.48 |
| $M_{2}$ | 54.51 | 84.27 | 59.29 | 69.31 | 33.07 |
| $M_{3}$ | 53.13 | 83.29 | 57.70 | 69.37 | 31.86 |

Human evaluation To examine whether human judgments align with automatic evaluation results, we conduct human evaluations that compare SFT baseline generations with the generations from each iteration of Self-Rewarding training, i.e., models $M_{1}, M_{2}$, and $M_{3}$. Specifically, we randomly select 50 instructions from the IFT test set. Each instruction corresponds to three pairs of generations (i.e., baseline vs. $M_{1}$, baseline vs. $M_{2}$, baseline vs. $M_{3}$ ). For each pair of generations, we assign them to three different annotators (blind evaluation performed by the authors) to make a pairwise judgment, and take a majority vote to decide which generation is better. The human evaluation results are shown in Figure 5. We find that Self-Rewarding models from later iterations show a larger advantage over the SFT baseline model, which is consistent with GPT-4's judgments, and demonstrates the effectiveness of our iterative training procedure.

MT-Bench performance further validates these results We report performance on MT-Bench in Table 2 for the SFT baseline and iterations of the Self-Rewarding model. We

Table 4: Reward Modeling ability improves with Self-Training: We evaluate the LLMas-a-Judge via various metrics which measure alignment with held-out human preference data. Self-Rewarding Iteration 2 (Model $M_{2}$ ), which is trained using the self-reward model derived from its previous iteration $M_{1}$ outperforms Iteration $1\left(M_{1}\right)$, while $M_{1}$ itself outperforms a standard SFT baseline model trained on only Instruction Fine-Tuning (IFT) data. Iteration 3 (Model $M_{3}$ ) gives further improvements over Iteration 2.

|  |  |  | Self-Rewarding Models |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Model | SFT Baseline | Iter 1 $\left(M_{1}\right)$ | Iter $2\left(M_{2}\right)$ | Iter $3\left(M_{3}\right)$ |  |
| Training data | IFT | IFT+EFT | IFT+EFT | IFT+EFT+AIFT $\left(M_{1}\right)$ |  |
|  |  |  | $+\operatorname{AIFT}\left(M_{1}\right)$ | $+\operatorname{AIFT}\left(M_{2}\right)$ |  |
| Pairwise acc. $(\uparrow)$ | $65.1 \%$ | $78.7 \%$ | $80.4 \%$ | $81.7 \%$ |  |
| 5-best \% $(\uparrow)$ | $39.6 \%$ | $41.5 \%$ | $44.3 \%$ | $43.2 \%$ |  |
| Exact Match $\%(\uparrow)$ | $10.1 \%$ | $13.1 \%$ | $14.3 \%$ | $14.3 \%$ |  |
| Spearman corr. $(\uparrow)$ | 0.253 | 0.279 | 0.331 | 0.349 |  |
| Kendall $\tau$ corr. $(\uparrow)$ | 0.233 | 0.253 | 0.315 | 0.324 |  |

again see improvements across the iterations of training from $M_{1}$ to $M_{3}$, from 6.78 (out of 10) up to 7.25 , with larger relative gains in the humanities, STEM, roleplay, writing and extraction categories, and smaller gains in the math, code and reasoning categories. We expect that the latter is due to the seed prompts we use from Open Assistant tending to underemphasize the reasoning-based tasks. We note also that these improvements are in spite of our method using and constructing prompts that only involve a single turn, given the MT-Bench benchmark itself is a multi-turn evaluation.

Self-rewarding models did not lose ability on NLP Benchmarks As shown in Table 3, the performance of most NLP benchmark tasks evaluated are roughly similar to the baselines, with further detailed results on more datasets given in Appendix Table 9 that follow the same pattern. We hypothesize that given that our training data (seed data and synthetically generated data) are based on the Open Assistant prompts which may not be especially relevant to skills needed in the Table 3 tasks, it is expected that the task performance stays roughly similar, or may even drop. For example, in InstructGPT training [Ouyang et al., 2022] they found that "during RLHF fine-tuning, we observe performance regressions compared to GPT-3 on certain public NLP datasets" which they refer to as an "alignment tax." A clear future direction is to extend the self-rewarding paradigm to these types of tasks, by relying not only on seed prompts from Open Assistant, but also on seed prompts found in a larger variety of datasets.

### 3.2.2 Reward Modeling Ability

Reward modeling evaluation results are provided in Table 4.

EFT augmentation improves over SFT baseline Firstly, we find that adding Evaluation Fine-Tuning (EFT) data into training, which gives examples to the model of how to act as an LLM-as-a-Judge, naturally improves its performance compared to training with Instruction Fine-Tuning (IFT) data alone. IFT data covers a wide range of general instruction tasks, and so does endow the SFT Baseline with the ability to evaluate responses; however, EFT data gives more examples of this specific task. We find improvements across all five metrics measured when using IFT+EFT vs. IFT alone, e.g., the pairwise accuracy agreement with humans increases from $65.1 \%$ to $78.7 \%$.

Reward Modeling ability improves with Self-Training We find that performing a round of self-reward training improves the ability of the model at providing self-rewards for the next iteration, in addition to its improved instruction following ability. Model $M_{2}$ (Iteration 2) is trained using the reward model from $M_{1}$ (Iteration 1), but provides improved performance on all five metrics compared to $M_{1}$. For example, pairwise accuracy improves
from $78.7 \%$ to $80.4 \%$. Iteration $3\left(M_{3}\right)$ improves several of these metrics further compared to $M_{2}$, for example pairwise accuracy increases from $80.4 \%$ to $81.7 \%$. This performance gain is achieved despite there being no additional EFT data provided, and the examples created during the Self-Instruction creation loop do not tend to look like LLM-as-a-Judge training examples. We hypothesize that because the model is becoming better at general instruction following, it nevertheless also improves at the LLM-as-a-Judge task.

Importance of the LLM-as-a-Judge Prompt In these experiments we used the LLMas-a-Judge prompt format shown in Figure 2. In preliminary experiments we also tried various other prompts to decide the most effective one to use. For example, we tried the prompt proposed in Li et al. [2024] which also proposes a 5-point scale, but describes the options as multiple choice in a range of quality buckets, see Appendix Figure 7. In contrast, our prompt describes the points as additive, covering various aspects of quality. We find a large difference between these two prompts when using the SFT Baseline, e.g. $65.1 \%$ pairwise accuracy for ours, and only $26.6 \%$ pairwise accuracy for theirs. See Appendix A. 2 for further details.

## 4 Related Work

Automatically improving or self-correcting large language models is becoming a major focus of research. A recent survey from Pan et al. [2023] attempts to summarize the topic. However, this is a rapidly moving area, and there are already promising new works not covered there.

Reinforcement Learning from Human Feedback (RLHF) Preference learning approaches such as in Ziegler et al. [2019], Stiennon et al. [2020], Ouyang et al. [2022], Bai et al. [2022a] train a fixed reward model from human preference data, and then use the reward model to train via reinforcement learning (RL), e.g. via Proximal Policy Optimization (PPO) [Schulman et al., 2017]. Thus, the reward signal in a certain sense already comes from a model even in these works, but distilled from human data. Nevertheless, this is commonly referred to as RL from Human Feedback (RLHF). Methods such as Direct Preference Optimization (DPO) [Rafailov et al., 2023] avoid training the reward model entirely, and instead directly train the LLM using human preferences. Several other such competing methods exist as well [Zhao et al., 2023, Zheng et al., 2023a, Yuan et al., 2023], including Pairwise Cringe Optimization (PCO) [ $\mathrm{Xu}$ et al., 2023]. PCO uses an iterative training approach similar to the one in our work, except with a fixed reward model, and that work also showed that Iterative DPO improves over DPO using the same scheme. We note that other works have developed iterative preference training schemes as well, e.g. Adolphs et al. [2023], Gulcehre et al. [2023], Xiong et al. [2023].

Reinforcement Learning from AI Feedback (RLAIF) Constitutional AI [Bai et al., 2022b] uses an LLM to give feedback and refine responses, and uses this data to train a reward model. This fixed, separate reward model is then used to train the language model via RL, called "RL from AI Feedback" (RLAIF). Lee et al. [2023] compare RLAIF and RLHF procedures and find the methods they compare perform roughly equally. They use an "off-the-shelf" LLM to perform LLM-as-a-Judge prompting to build a training set to train a fixed reward model, which is then used for RL training. They also experiment with using the fixed but separate LLM-as-a-Judge model directly, which the authors report is computationally expensive due to using it within PPO training (rather than the offline step in the iterative approach we use in our work, which is relatively computationally cheap). Finally, SPIN [Chen et al., 2024b] recently showed they can avoid reward models entirely in an Iterative DPO-like framework by using human labels as the winning response in a pair, and the last iteration's generations as the losing response in the pair. The authors note this has the limitation that once the model generations reach human performance, they are bottlenecked. Further, each input prompt is required to have a human annotated response, in contrast to our work.

Improving LLMs via data augmentation (and curation) Several methods have improved LLMs by (self-)creating training data to augment fine-tuning. Self-Instruct [Wang
et al., 2023] is a method for self-instruction creation of prompts and responses, which can be used to improve a base LLM. We make use of a similar technique in our work, and then use our self-reward model to score them. Several approaches have also created training data by distilling from powerful LLMs, and shown a weaker LLM can then perform well. For example, Alpaca [Taori et al., 2023] fine-tuned a Llama 7B model with text-davinci-003 instructions created in the style of self-instruct. Alpagasus [Chen et al., 2024a] employed a strong LLM-as-a-Judge (ChatGPT) to curate the Alpaca dataset and filter to a smaller set, obtaining improved results. Instruction Backtranslation [Li et al., 2024] similarly augments and curates training data, but augmenting via backtranslating from web documents to predict prompts. The curation is done by the LLM(-as-a-Judge) itself, so can be seen as an instance of a self-rewarding model, but in a specialized setting. Reinforced Self-Training (ReST) [Gulcehre et al., 2023] uses a fixed, external reward to curate new high-quality examples to iteratively add to the training set, improving performance. In our experiments, we found that adding only positive examples in a related manner did not help, whereas preference pairs did help (see Appendix Section A. 4 for details).

LLM-as-a-Judge Using LLM-as-a-Judge prompting to evaluate language models has become a standard approach [Dubois et al., 2023, Li et al., 2023, Fernandes et al., 2023, Bai et al., 2023, Saha et al., 2023], and is being used to train reward models or curate data as well, as described above [Lee et al., 2023, Chen et al., 2024a, Li et al., 2024]. While some works such as Kim et al. [2023] create training data to train an LLM to perform well as a judge, to our knowledge it is not common to combine this training with general instruction following skills as in our work.

## 5 Conclusion

We have introduced Self-Rewarding Language Models, models capable of self-alignment via judging and training on their own generations. The method learns in an iterative manner, where in each iteration the model creates its own preference-based instruction training data. This is done by assigning rewards to its own generations via LLM-as-a-Judge prompting, and using Iterative DPO to train on the preferences. We showed that this training both improves the instruction following capability of the model, as well as its reward-modeling ability across the iterations. While there are many avenues left unexplored, we believe this is exciting because this means the model is better able to assign rewards in future iterations for improving instruction following - a kind of virtuous circle. While this improvement likely saturates in realistic scenarios, it still allows for the possibility of continual improvement beyond the human preferences that are typically used to build reward models and instruction following models today.

## 6 Limitations

While we have obtained promising experimental results, we currently consider them preliminary because there are many avenues yet to explore, among them the topics of further evaluation, including safety evaluation, and understanding the limits of iterative training.

We showed that the iterations of training improve both instruction following and reward modeling ability, but only ran three iterations in a single setting. A clear line of further research is to understand the "scaling laws" of this effect both for more iterations, and with different language models with more or less capabilities in different settings.

We observed an increase in length in model generations, and there is a known correlation between length and estimated quality, which is a topic that should be understood more deeply in general, and in our results in particular as well. It would also be good to understand if so-called "reward-hacking" can happen within our framework, and in what circumstances. As we are using both a language model as the training reward, and a language model for final evaluation (GPT-4) in some of our benchmarks, even if they are different models, this may require a deeper analysis than we have provided. While the human evaluation we conducted did provide validation of the automatic results, further study could bring more insights.

Another clear further avenue of study is to conduct safety evaluations - and to explore safety training within our framework. Reward models have been built exclusively for safety in existing systems [Touvron et al., 2023], and a promising avenue here would be to use the LLM-as-a-Judge procedure to evaluate for safety specifically in our self-rewarding training process. Given that we have shown that reward modeling ability improves over training iterations, this could mean that the safety of the model could potentially improve over time as well, with later iterations being able to catch and mitigate more challenging safety situations that earlier iterations cannot.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Leonard Adolphs, Tianyu Gao, Jing Xu, Kurt Shuster, Sainbayar Sukhbaatar, and Jason Weston. The CRINGE loss: Learning what language not to model. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8854-8874, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.493. URL https://aclanthology.org/2023.acl-long.493.

Anthropic. Claude 2. https://www.anthropic.com/index/claude-2, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022a.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022b.

Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, and Lei Hou. Benchmarking foundation models with language-model-as-an-examiner. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=IiRHQ7gvnq.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. AlpaGasus: Training a better alpaca with fewer data. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=FdVXgSJhvz.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024b.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, pages 160-167, 2008.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023.

Patrick Fernandes, Daniel Deutsch, Mara Finkelstein, Parker Riley, André Martins, Graham Neubig, Ankush Garg, Jonathan Clark, Markus Freitag, and Orhan Firat. The devil is in the errors: Leveraging large language models for fine-grained machine translation evaluation.

In Philipp Koehn, Barry Haddow, Tom Kocmi, and Christof Monz, editors, Proceedings of the Eighth Conference on Machine Translation, pages 1066-1083, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.wmt-1.100. URL https://aclanthology.org/2023.wmt-1.100.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023 .

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14409-14428, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long. 806 . URL https://aclanthology.org/2023.acl-long.806.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. Prometheus: Inducing fine-grained evaluation capability in language models. arXiv preprint arXiv:2310.08491, 2023.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. OpenAssistant conversations-democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=1oijHJBRsT.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval, 2023.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74-81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies. arXiv preprint arXiv:2308.03188, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=HPuSIXJaa9.

Swarnadeep Saha, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, and Xian Li. Branch-solve-merge improves large language model evaluation and generation. arXiv preprint arXiv:2310.15123, 2023.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. CoRR, abs/1904.09728, 2019. URL http://arxiv.org/abs/1904.09728.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008-3021, 2020.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of machine learning research, 9(11), 2008.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with selfgenerated instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484-13508, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.754. URL https://aclanthology.org/2023.acl-long. 754.

Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, and Tong Zhang. Gibbs sampling from human feedback: A provable kl-constrained framework for rlhf. arXiv preprint arXiv:2312.11456, 2023.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. RRHF: Rank responses to align language models with human feedback. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=EdIGMCHk4l.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791-4800. Association for Computational Linguistics, 2019. doi: 10.18653/V1/ P19-1472. URL https://doi.org/10.18653/v1/p19-1472.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. SLiC-HF: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Chujie Zheng, Pei Ke, Zheng Zhang, and Minlie Huang. Click: Controllable text generation with sequence likelihood contrastive learning. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: $A C L$ 2023, pages 1022-1040, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.65. URL https://aclanthology.org/ 2023.findings-acl.65.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023b. URL https://openreview.net/forum?id=uccHPGDlao.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
</end of paper 2>


<paper 3>
# KTO: Model Alignment as Prospect Theoretic Optimization 

Kawin Ethayarajh ${ }^{1}$ Winnie Xu ${ }^{2}$ Niklas Muennighoff ${ }^{2}$ Dan Jurafsky ${ }^{1}$ Douwe Kiela ${ }^{12}$


#### Abstract

Kahneman \& Tversky's prospect theory tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biasesthe success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being human-aware loss functions (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a KahnemanTversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B. Crucially, KTO does not need preferencesonly a binary signal of whether an output is desirable or undesirable for a given input. This makes it far easier to use in the real world, where preference data is scarce and expensive.


## 1. Introduction

Aligning generative models with human feedback has been successfully used to make generations more helpful, factual, and ethical, among other desiderata (Ouyang et al., 2022; Tian et al., 2023). For LLMs, alignment methods such as RLHF and DPO have consistently proven to be more beneficial than doing supervised finetuning (SFT) alone. However, human feedback is often discussed only in the context of preferences (e.g., output $A \succ B$ for input $x$ ), despite preferences being a kind of data that is relatively scarce and expensive to collect in the real world (Casper et al., 2023). This is largely because the alignment methods shown to work best-RLHF (Christiano et al., 2017) and[^0]

the mathematically equivalent DPO (Rafailov et al., 2023)_ take preference data as input.

To understand why these alignment methods work so well, and whether feedback needs to be in the form of preferences, we frame them through the lens of prospect theory (Kahneman \& Tversky, 1979; Tversky \& Kahneman, 1992). Prospect theory explains why humans make decisions about uncertain events that do not maximize expected value. It formalizes how humans perceive random variables in a biased but well-defined manner; for example, relative to some reference point, humans are more sensitive to losses than gains, a property called loss aversion. We show that popular alignment methods such as PPO (Schulman et al., 2017), DPO (Rafailov et al., 2023), and SLiC (Zhao et al., 2023) implicitly model such biases, helping explain their success independently of the data used. For this reason, we call them human-aware loss functions (HALOs).

Although it is impossible to say that HALOs are categorically better than non-HALOs, we find that among existing methods, those that meet the definition of a HALO work better than those that do not. We find that DPO performance can even be matched at most scales by running an offline PPO variant on dummy +1/-1 rewards, suggesting that preference data might not be needed if the inductive bias in the loss function is good enough. However, despite the surprising success of this simple baseline, it significantly lags behind DPO at the 30B model scale and suffers from hyperparameter sensitivity, making it difficult to use.

Taking a more principled approach, we derive a HALO using the model of human utility that Kahneman \& Tversky empirically derived to describe how humans make decisions about uncertain monetary outcomes (Tversky \& Kahneman, 1992). This approach, which we call Kahneman-Tversky Optimization (KTO), directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as most current methods do. KTO only requires a binary signal of whether an output is desirable or undesirable for a given input. This data is much more abundant, cheaper, and faster to collect in the real world than preferences, making it easier to scale alignment in production environments and rapidly iterate on models.

In our experiments, we find that:

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-02.jpg?height=542&width=1250&top_left_y=236&top_left_x=405)

Figure 1. The traditional pipeline for LLM alignment starts with supervised finetuning, followed by fitting the LLM to paired preference data using a method such as RLHF or DPO. However, the paired preferences that existing approaches need are hard-to-get. KahnemanTversky Optimization (KTO) only needs to know whether a given output is (un)desirable for the input, giving it access to a source of data that is much more abundant, cheaper, and faster to collect in the real world.

- KTO matches or exceeds DPO performance at scales from 1B to 30B parameters. ${ }^{1}$ That is, taking a preference dataset of $n$ DPO pairs and breaking it up into $2 n$ examples for KTO can yield better generations, despite the model ostensibly learning from a weaker signal. We provide some theoretical explanations for this phenomenon $(\$ 4.3)$.
- KTO can handle extreme data imbalances, matching DPO performance while using up to $90 \%$ fewer desirable examples (i.e., examples of good generations). Its success thus cannot be ascribed to the alignment data being sourced from a preference dataset.
- When the pretrained model is sufficiently good, one can skip supervised finetuning and go straight to KTO without a loss in generation quality. In contrast, we find that without doing SFT first, DPO-aligned models are significantly worse at all scales.

The fact that KTO can match and sometimes even outperform DPO is surprising, given that it learns from a weaker signal. We conclude by discussing some theoretical explanations for this phenomenon.

## 2. Background

Feedback-aligned LLMs are traditionally trained in three stages (Ouyang et al., 2022):

Pretraining Given a large corpus, train the model to predict the next token conditioned on the preceding text using the cross-entropy loss. Let $\pi$ denote the pretrained model.[^1]

Supervised Finetuning Finetune the model to predict the next token on data that is more relevant to the downstream task. Often, such data will comprise instructions and an appropriate response (i.e., instruction finetuning). Let $\pi_{\text {ref }}$ denote the finetuned model.

RLHF Given a dataset $\mathcal{D}$ of preferences $\left(x, y_{w}, y_{l}\right)$ where $x$ is an input, $y_{w}, y_{l}$ are the preferred and dispreferred outputs (i.e., $y_{w} \succ y_{l}$ for $x$ ), and $r^{*}$ is the "true" reward function underlying the preferences-it is first assumed that the probability that $y_{w}$ is preferred to $y_{l}$ can be captured with a specific function class, typically a Bradley-Terry model (Bradley \& Terry, 1952). Where $\sigma$ is the logistic function:

$$
\begin{equation*}
p^{*}\left(y_{w} \succ y_{l} \mid x\right)=\sigma\left(r^{*}\left(x, y_{w}\right)-r^{*}\left(x, y_{l}\right)\right) \tag{1}
\end{equation*}
$$

Since getting the true reward from a human would be intractably expensive, a reward model $r_{\phi}$ learns to serve as a proxy, done by minimizing the negative log-likelihood of the human preference data:

$$
\mathcal{L}_{R}\left(r_{\phi}\right)=\mathbb{E}_{x, y_{w}, y_{l} \sim D}\left[-\log \sigma\left(r_{\phi}\left(x, y_{w}\right)-r_{\phi}\left(x, y_{l}\right)\right)\right]
$$

But solely maximizing the reward might come at the expense of desiderata such as generating grammatical text. To avoid this, a KL divergence penalty is introduced to restrict how far the language model can drift from $\pi_{\text {ref. }}$. Where $\pi_{\theta}$ is the model we are optimizing, the optimal model $\pi^{*}$ is that which maximizes

$$
\begin{equation*}
\mathbb{E}_{x \in D, y \in \pi_{\theta}}\left[r_{\phi}(x, y)\right]-\beta D_{\mathrm{KL}}\left(\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right) \tag{2}
\end{equation*}
$$

where $\beta>0$ is a hyperparameter. Since this objective is not differentiable, we need to use an RL algorithm like PPO (Schulman et al., 2017).

However, RLHF is often slow (largely because of having to sample generations) and quite unstable in practice (especially in a distributed setting). For this reason, recent work has focused on designing closed-form losses that maximize the margin between the preferred and dispreferred generations, such as Sequence-Likelihood Calibration (SLiC) (Zhao et al., 2023) and Direct Preference Optimization (DPO) (Rafailov et al., 2023). The latter has become popular due to its mathematical equivalence with RLHF:

$$
\begin{align*}
& \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)= \\
& \mathbb{E}\left[-\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{l} \mid x\right)}\right)\right] \tag{3}
\end{align*}
$$

## 3. A Prospect Theoretic View of Alignment

Kahneman \& Tversky's prospect theory explains why, faced with an uncertain event, humans make decisions that do not maximize the expected value (1992). For example, because humans are loss-averse, given a gamble that returns $\$ 100$ with $80 \%$ probability and $\$ 0$ with $20 \%$ probability, a person might accept $\$ 60$ to avoid the gamble, despite their certainty equivalent of $\$ 60$ being less than the expected value of $\$ 80$.

### 3.1. Prospect Theory

In prospect theory, human utility depends on a value function and a weighting function: ${ }^{2}$

Definition 3.1. A value function $v: z \rightarrow \mathbb{R}$ maps an outcome $z$, relative to some reference point $z_{\text {ref }}$, to its perceived (or subjective) value. For example, these functions capture the fact that humans tend to be more sensitive to relative losses than relative gains of the same magnitude.

Definition 3.2. A weighting function $w$ is the derivative of a capacity function that maps cumulative probabilities to perceived cumulative probabilities. These functions capture, for example, the fact that humans tend to overestimate the chance of rare events. Let $w_{z}$ denote the weight placed on outcome $z$.

Definition 3.3. The utility of a random variable $Z$ is a function of its outcomes: $u(Z) \triangleq \sum_{z \in Z} w_{z} v\left(z-z_{\text {ref }}\right)$.

However, because humans do not see the full probability distribution of an LLM, weighting functions are not salient to this discussion; we will focus only on value functions. Using experiments that presented real humans with monetary gambles and asked for their certainty equivalent, Tversky \& Kahneman (1992) proposed the following functional form for human value:

$$
v\left(z, z_{\mathrm{ref}} ; \lambda ; \alpha\right)= \begin{cases}\left(z-z_{\mathrm{ref}}\right)^{\alpha} & \text { if } z>z_{\mathrm{ref}}  \tag{4}\\ -\lambda\left(z_{\mathrm{ref}}-z\right)^{\alpha} & \text { if } z<z_{\mathrm{ref}}\end{cases}
$$[^2]

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-03.jpg?height=588&width=723&top_left_y=232&top_left_x=1102)

Figure 2. The utility that a human gets from the outcome of a random variable, as imputed by the value function implicit in HALOs. Notice that the imputed functions share properties such as loss aversion with the human value functions that Kahneman \& Tversky empirically derived (1992).

where the median value of hyperparameter $\alpha=0.88$ and $\lambda=2.25$ across individuals. $\alpha$ controls how quickly utility changes and $\lambda$ controls the degree of loss aversion. While the shape of the median Kahneman-Tversky value function is illustrated in Figure 2, it should be noted that it varies across individuals (Tversky \& Kahneman, 1992). There are also other functional forms for the value function that have been proposed in later work (Gurevich et al., 2009). The salient qualities of a value function are: the existence of a reference point that is added or subtracted to get the relative gain or loss respectively; concavity in relative gains (i.e. diminishing sensitivity away from $z_{\text {ref }}$ ); loss aversion (i.e., greater sensitivity to losses).

### 3.2. HALOs

Informally, HALOs are loss functions that model the human biases in Tversky \& Kahneman (1992). Formally,

Definition 3.4 (HALOs). Let $x \in \mathcal{X}$ denote an input and $y \in \mathcal{Y}$ an output. Then $f:(x, y) \rightarrow \mathbb{R}$ is a human-aware loss function if there exists the following: a parameterized reward function $r_{\theta}$ such that $\forall\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \mathcal{X} \times \mathcal{Y}$,

$$
r_{\theta}\left(x_{1}, y_{1}\right)>r_{\theta}\left(x_{2}, y_{2}\right) \Longleftrightarrow\left(x_{1}, y_{1}\right) \succ_{r_{\theta}}\left(x_{2}, y_{2}\right)
$$

reference point distributions $Q_{x}\left(X^{\prime}\right), Q_{y}\left(Y^{\prime} \mid X^{\prime}\right)$, a value function $v_{f}: \mathbb{R} \rightarrow \mathbb{R}$ that is monotonic non-decreasing and concave in $(0, \infty)$, and a negative affine function $t$ such that

$$
\begin{equation*}
f(x, y ; \theta)=t\left(v_{f}\left(r_{\theta}(x, y)-\mathbb{E}_{x^{\prime}, y^{\prime}}\left[r_{\theta}\left(x^{\prime}, y^{\prime}\right)\right]\right)\right) \tag{5}
\end{equation*}
$$

where $x^{\prime} \sim Q_{x}\left(X^{\prime}\right)$ and $y^{\prime} \sim Q_{y}\left(Y^{\prime} \mid x^{\prime}\right)$.

Put simply, the requirement for the reward function is that it assigns higher rewards to input-output pairs that are more
preferred under it. The reference point is the expected reward with respect to input-output pairs sampled from the distributions $Q_{x}, Q_{y}$. We require that the value function be concave in gains but not necessarily convex in lossesunlike the canonical Kahneman-Tversky value functionsbecause in the original work on prospect theory, a minority of individuals were found to be risk-averse in both the gain and loss regime (i.e., concave in both gains and losses) (Kahneman \& Tversky, 1979). Note that risk-aversion is different from loss-aversion; they relate to the curvature and magnitude of the slope respectively.

Proposition 3.5. DPO, SLiC (calibration loss only), and PPO-Clip are human-aware loss functions.

The proof is deferred to Appendix A. In Figure 2, we can see this more intuitively by plotting the value function for each loss (i.e., the implied human utility). We see that the value functions of all three losses incorporate a sense of loss aversion, although this is not needed to meet the definition of a HALO, since there are individuals and scenarios for which loss aversion does not necessarily apply. The value functions are also either concave or affine (depending on the interval), unlike the standard Kahneman-Tversky value function, which is concave in gains but convex in losses. The reference point distributions used also differs across the losses.

### 3.3. Does being a HALO matter?

A natural question is whether the modeling of human biases in HALOs has practical benefits. This is difficult to answer, since both HALOs and non-HALOs are diverse function classes, but we attempt to do so by comparing popular nonHALO and HALO baselines on the exact same data:

1. CSFT: Conditional SFT is a simple alignment method where a control token is prepended to the output during training; then, at inference, the control token corresponding to desirable generations (e.g., <|good $\mid>$ ) is appended to the input to induce good generations (Korbak et al., 2023). This is a non-HALO loss.
2. SLiC: SLiC with a regularization penalty $\left(\lambda_{\text {reg }} \neq 0\right)$ is a non-HALO loss:

$$
\begin{aligned}
& \mathcal{L}_{\mathrm{SLiC}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=\mathcal{L}_{\mathrm{cal}}\left(\pi_{\theta}\right)+\lambda_{\mathrm{reg}} L_{\mathrm{reg}}\left(\pi_{\theta}\right) \\
& \mathcal{L}_{\mathrm{cal}}=\mathbb{E}_{x, y_{w}, y_{l} \sim D}\left[\max \left(0, \delta-\log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\theta}\left(y_{l} \mid x\right)}\right)\right] \\
& \mathcal{L}_{\mathrm{reg}}=\mathbb{E}_{x \sim D, y \sim \pi_{\mathrm{ref}}(x)}\left[-\log \pi_{\theta}(y \mid x)\right]
\end{aligned}
$$

Although the max-margin loss $\mathcal{L}_{\text {cal }}$ is a HALO on its own (Proposition 3.5), the complete loss is not, since the $\mathcal{L}_{\text {reg }}$ term is the standard language modeling loss.

3. DPO: DPO, as defined in (3), is a HALO loss (Proposition 3.5).
4. PPO (offline): The standard RLHF objective in (2) is typically optimized with PPO-Clip, which works by "clipping" how far $\pi_{\theta}$ can drift from the version $\pi_{\text {old }}$ at the previous step:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{PPO} \text { (offline) }}= & -\mathbb{E}_{x, y, t \sim D}\left[\operatorname { m i n } \left(q_{\theta} A\left(x, y_{<t}, y_{t}\right)\right.\right. \\
& \left.\left.\operatorname{clip}\left(q_{\theta}, 1-\epsilon, 1+\epsilon\right) A\left(x, y_{<t}, y_{t}\right)\right)\right]
\end{aligned}
$$

where $q_{\theta}=\log \frac{\pi_{\theta}}{\pi_{\text {old }}}$ and $A\left(x, y_{<t}, y_{t}\right)$ is the per-token advantage (i.e., the surplus benefit from producing a given token in a given state).

PPO is an online algorithm-generations are sampled from the current model, judged by a reward model, and then used to update the current version. However, this process is slow (due to having to sample generations), so we choose to use offline data instead. Because RLHF is also quite unstable in a distributed setting, we never update $\pi_{\text {old }}$ and keep it as $\pi_{\text {ref }}$, instead clipping less conservatively than we traditionally would (see Appendix B for details). Baheti et al. (2023) found that these changes, along with treating the entire output sequence as a single action, greatly improves stability. However, since RLHF has historically calculated tokenlevel advantages, we omit the third change and only preserve the first two. The PPO-Clip loss itself is left unchanged and is therefore a HALO (Proposition 3.5).

Calling this method PPO is somewhat imprecise, because it is offline and takes only one step, but to avoid introducing too many new terms, we will call this $P P O$ (offline). Instead of using learned rewards, we simplify even further and use dummy $+1 /-1$ rewards for $y_{w}$ and $y_{l}$ instead. Further details on the implementation of this method can be found in Appendix B.

We compare these baselines on a suite of 7 models spanning two model families, Pythia- $\{1.4 \mathrm{~B}, 2.8 \mathrm{~B}, 6.9 \mathrm{~B}, 12 \mathrm{~B}\}$ (Biderman et al., 2023) and Llama-\{7B, 13B, 30B \} (Touvron et al., 2023). This permits us to see how LLM alignment scales within a model family (Llama-2 lacks a 30B model, hence our use of Llama). Later experiments ( $\$ 4.2$ ) are done on Mistral-7B and its derivatives (Jiang et al., 2023). The models were trained on a combination of Anthropic HH (Ganguli et al., 2022), OpenAssistant (Köpf et al., 2023), and SHP (Ethayarajh et al., 2022).

All models were aligned under identical settings on the same data (e.g., same effective batch size, same optimizer, etc.), save for hyperparameters unique to them. Similar to Rafailov et al. (2023), the target sequences for SFT are a subset of the generations used to subsequently align the model; however, for a more realistic SFT setup, we do not necessarily set the most preferred generation to be the target (with the exception of $\mathrm{HH}$, since the dispreferred output in that dataset is often harmful). Then we used GPT-4-0613

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-05.jpg?height=564&width=1702&top_left_y=233&top_left_x=187)

Does the aligned model beat the SFT target?

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-05.jpg?height=518&width=1678&top_left_y=272&top_left_x=191)

Figure 3. Among existing alignment methods, the HALOs (DPO and our offline PPO variant) generally outperform non-HALOs (SLiC and CSFT), though the gap is only significant $(p<0.05)$ at 13B+ model sizes. In fact, only the HALO-aligned Llama-\{13B, 30B $\}$ models are able to match or exceed the generation quality of SFT target sequences, which are drawn directly from the alignment dataset. It is also worth noting that up to a scale of 7B parameters, virtually all of the gains from LLM alignment come from the SFT stage.

to judge whether the aligned model's response was better than the SFT target for the given input with respect to helpfulness, harmlessness, and conciseness, a now standard practice (Zheng et al., 2023; Li et al., 2023). ${ }^{3}$ Note that while the SFT target is considered a desirable output for $x$, it is by no means the best output, meaning that it can be improved upon by an aligned model.

In Figure 3, we see the results of this analysis:
- The HALOs we tested (DPO and our PPO variant) either match or outperform the non-HALOs at all scales, though the gap is only significant $(p<0.05)$ at 13B+ model sizes. In fact, only the HALO-aligned Llama\{13B, 30B \} models match or exceed a win rate of $50 \%$ (i.e., are able to match or exceed the generation quality of the SFT targets in the test data).
- Up to a scale of 7B parameters, alignment provides virtually no gains over SFT alone. However, it is worth noting that if the SFT data distribution were less similar to the preference data, then the gains from the alignment stage would ostensibly be greater.
- Surprisingly, despite only using dummy $+1 /-1$ rewards, our offline PPO variant performs as well as DPO for all models except Llama30B. This challenges conventional wisdom, which places heavy emphasis on reward learning (Casper et al., 2023), suggesting that even the simplest rewards can prove useful when used in a loss function that has a strong inductive bias. Despite its surprising success, our offline PPO baseline still suffers from hyperparameter sensitivity and training instability,[^3]

albeit not to the same extent as traditional RLHF.

## 4. Kahneman-Tversky Optimization

The surprising success of offline PPO with dummy +1/-1 rewards suggests that — with the right HALO—a binary signal of good/bad generations may be sufficient to reach DPOlevel performance, even if the offline PPO approach itself was unable to do so past a certain scale ( $\$ 3.3$ ). Taking a more principled approach, we now derive a HALO using the Kahneman-Tversky model of human utility, which allows us to directly optimize for utility instead of maximizing the log-likelihood of preferences. This Kahneman-Tversky Optimization (KTO) loss only needs a binary signal of whether an output is (un)desirable for a given input, giving it access to a source of data is more abundant, cheaper, and faster to collect in the real world.

### 4.1. Derivation

From prior work (Go et al., 2023; Peng et al., 2019; Peters \& Schaal, 2007), we know that the policy that maximizes the KL-constrained RLHF objective in (2) is

$$
\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r^{*}(x, y)\right)
$$

where $Z(x)$ is a partition function. Rafailov et al. (2023) rewrite this in terms of the optimal reward for an inputoutput pair:

$$
\begin{equation*}
r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x) \tag{6}
\end{equation*}
$$

They then plug this expression into the Bradley-Terry model of preferences and take the negative logarithm of that objective to get the DPO loss (3).

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-06.jpg?height=562&width=1699&top_left_y=234&top_left_x=186)

Does the aligned model beat the SFT target?

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-06.jpg?height=525&width=1680&top_left_y=266&top_left_x=190)

Figure 4. Kahneman-Tversky Optimization (KTO) is as good or better than DPO at all scales, both when preceded and not preceded by supervised finetuning (SFT). In fact, for the Llama models, KTO alone matches the performance of SFT+DPO and is significantly better than DPO alone. Error bars denote a $90 \%$ binomial confidence interval.

Instead, we plug this expression into the Kahneman-Tversky model of human utility, with some changes to make it more amenable to the LLM setting:

1. The exponent in the Kahneman-Tversky value function (4) makes it difficult to optimize, so we set $v_{\text {KTO }}$ to be the logistic function $\sigma$, which is also concave in gains and convex in losses. We replace the lossaversion coefficient with two hyperparameters $\lambda_{D}, \lambda_{U}$ that weight the losses for desirable and undesirable outputs respectively.
2. The Kahneman-Tversky value function was derived based on experiments with humans and monetary gambles. Since LLM generations do not have a monetary reward associated with them, we set $r_{\text {KTO }}$ to be the implicit reward under the RLHF objective (6).
3. Rather than having just one dispreferred generation $y_{l} \mid x$ as the reference point, we assume that humans judge the quality of $(x, y)$ in relation to all input-output pairs they have seen. Thus we write the reference point to be the expected reward under the optimal policy, not just for generations following $x$ but following any input $x^{\prime}: \mathbb{E}_{x^{\prime} \sim D, y^{\prime} \sim \pi^{*}}\left[r^{*}\left(x^{\prime}, y^{\prime}\right)\right]$. Under the assumption that the expected value of the partition function across $x^{\prime}$ is zero, this simplifies to the KL divergence between $\pi^{*}$ and $\pi_{\text {ref }}$ scaled by $\beta$.

Combining all of these changes, we can optimize the following loss, where the notion of an output being "desirable" or "undesirable" corresponds to the Kahneman-Tversky notion of a relative gain or loss.

$$
\begin{equation*}
L_{\mathrm{KTO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=\mathbb{E}_{x, y \sim D}\left[w(y)\left(1-v_{\mathrm{KTO}}(x, y ; \beta)\right)\right] \tag{7}
\end{equation*}
$$

where

$$
\begin{aligned}
r_{\mathrm{KTO}}(x, y) & =\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \\
z_{\mathrm{ref}} & =\mathbb{E}_{x^{\prime} \sim D}\left[\beta \operatorname{KL}\left(\pi_{\theta}\left(y^{\prime} \mid x^{\prime}\right) \| \pi_{\mathrm{ref}}\left(y^{\prime} \mid x^{\prime}\right)\right)\right] \\
v_{\mathrm{KTO}}(x, y ; \beta) & =\left\{\begin{array}{l}
\sigma\left(r_{\mathrm{KTO}}(x, y)-z_{\text {ref }}\right) \text { if } y \sim y_{\text {desirable }} \mid x \\
\sigma\left(z_{\text {ref }}-r_{\mathrm{KTO}}(x, y)\right) \text { if } y \sim y_{\text {undesirable }} \mid x
\end{array}\right. \\
w(y) & = \begin{cases}\lambda_{D} & \text { if } y \sim y_{\text {desirable }} \mid x \\
\lambda_{U} & \text { if } y \sim y_{\text {undesirable }} \mid x\end{cases}
\end{aligned}
$$

Intuitively, KTO works because if the model increases the reward of a desirable example in a generic way, then the KL penalty will also rise and no progress will be made on the loss. This forces the model to learn exactly what makes an output desirable, so that the reward can be increased while keeping the KL term flat (or even decreasing it). A similar argument works in the other direction as well, though the non-negativity of the KL term allows faster saturation.

Implementation In practice, we estimate the KL term by matching inputs $x^{\prime}$ with unrelated outputs $y_{U}^{\prime}$ in a batch of size $m$ and then calculating max $\left(0, \frac{1}{m} \sum \log \frac{\pi_{\theta}\left(y_{U}^{\prime} \mid x^{\prime}\right)}{\pi_{\text {ref }}\left(y_{U}^{\prime} \mid x^{\prime}\right)}\right)$ over the entire batch. We do not back-propagate through the $\mathrm{KL}$ term, as it makes training much more stable. This means that the KL term purely serves to control how saturated the loss is.

$\beta$ has the same meaning as in DPO; the lower it is, the less we penalize $\pi_{\theta}$ from moving away from the SFT model $\pi_{\text {ref }}$. We find that $\beta=0.1$ is close-to-best on most datasets. Where $n_{D}$ and $n_{U}$ refer to the number of desirable and undesirable examples respectively, we set $\lambda_{D}, \lambda_{U}$ such that

$$
\begin{equation*}
\frac{\lambda_{D} n_{D}}{\lambda_{U} n_{U}} \in\left[1, \frac{4}{3}\right] \tag{8}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-07.jpg?height=409&width=813&top_left_y=232&top_left_x=190)

Figure 5. Without doing SFT first, DPO-aligned models tend to ramble and hallucinate entire conversations. KTO does not suffer from this issue.

where at least one of the two should be set to 1 and the ratio is controlled by changing the other. For example, if there is a 1:1 ratio of desirable:undesirable examples, we would set $\lambda_{U}=1, \lambda_{D} \in[1,1.33]$. If we then discard $90 \%$ of the desirable examples and only keep $10 \%$, then we would set $\lambda_{U}=1, \lambda_{D} \in[10,13.33]$. The interval $[1,4 / 3]$ was determined empirically and suggests a value function that is more gain-sensitive than loss-sensitive, in contrast to the original Kahneman-Tversky value function (4). However, the ideal interval is also task-dependent; for example, if avoiding negative outcomes were very important, then we might consider a setting of $\lambda_{U}>1$ instead.

Data If the alignment data is naturally binary, every positive example can be assumed to be drawn from $y_{\text {desirable }} \mid x$ and every negative example from $y_{\text {undesirable }} \mid x$. However, the canonical feedback datasets in academic research $(\mathrm{HH}, \mathrm{SHP}$, OASST) are in preference format, since the methods that have worked best up until now are preference-based. In our experiments, we converted preference data $y_{w} \succ y_{l}$ by assuming that $y_{w}$ is drawn from the desirable distribution and $y_{l}$ from the undesirable one. To enable an apples-to-apples comparison with DPO, we apply KTO on the same data for most experiments. However, to ensure that KTO can be used with non-preference data, we also subsample one output $y$ per $x$ for some experiments (denoted one- $y$-per- $x$ ).

If the data is score-based, where a higher score denotes greater desirability, one has multiple options:

- Assume that any output with a score above some fixed threshold $\tau$ is desirable.
- Assume that any output with a score above the mean or median (either across all inputs or just the input it was conditioned on) is desirable.
- Let desirability be a Bernoulli random variable where $p\left(y \sim y_{\text {desirable }} \mid x\right)$ is some function of its score (e.g., logistic). Then randomly sample to determine whether $y$ is desirable or not.

![](https://cdn.mathpix.com/cropped/2024_05_26_60da5211e7e1f425fa14g-07.jpg?height=626&width=791&top_left_y=237&top_left_x=1079)

Figure 6. Even after discarding $90 \%$ of the desirable examples while keeping all of the undesirable data (leading to a 1:10 ratio of desirable:undesirable data), a KTO-aligned Llama-7B model still outperforms its DPO counterpart. This implies that preference pairs do not have to be the source of KTO data.

### 4.2. Experiments

KTO $\geq$ DPO As seen in Figure 4, SFT+KTO is competitive with SFT+DPO at model scales from 1B to 30B, despite learning from a weaker signal. KTO alone is better than DPO alone for the Llama-\{7B, 13B, 30B \} models, and this gap is significant $(p<0.01)$ at $7 \mathrm{~B}$ and 30B even after correcting for multiple comparisons (Holm, 1979). Perhaps most surprising is the fact that a KTO-aligned Llama- $\{13 \mathrm{~B}$, 30B $\}$ model is competitive with its SFT+KTO counterpart, despite not undergoing supervised finetuning first, and is the only alignment method of the ones we tested to show this behavior. This is perhaps due to the fact that KTO keeps the average response length roughly the same as it is for the SFT model. In contrast, doing DPO without SFT first causes the average response length to increase dramatically.

KTO data need not come from preference datasets. Might KTO be secretly benefiting from the fact that its $2 n$ examples in the previous experiment came from $n$ preference pairs instead of a naturally unpaired data distribution? To test this, we randomly discard increasingly large fractions of the desirable data before KTO-aligning a Llama-7B model. For example, if we discard $90 \%$ of the desirable data while leaving the undesirable data untouched, then the ratio of desirable:undesirable examples goes from 1:1 to 1:10 and the vast majority of examples no longer have a preferred output counterpart. We handle such imbalances by changing the loss weights $\lambda_{D}, \lambda_{U}$ to satisfy the criteria in (8); when we drop $90 \%$ of the desirable data, we set $\lambda_{u}=1, \lambda_{D}=13.33$. The full results are given in Figure 6. For Llama-7b, we find that up to $90 \%$ of the desirable

Table 1. In aligning Mistral-7B on the OpenAssistant dataset, we find that using KTO with only one output per input still outperforms DPO, despite this restriction reducing the amount of training data by $72 \%$. A $90 \%$ confidence interval is given.

| Method | Winrate vs. SFT Target |
| :--- | :---: |
| Mistral-7B (unaligned) | $0.525 \pm 0.037$ |
| Mistral-7B + DPO | $0.600 \pm 0.037$ |
| Mistral-7B + KTO (all $y$ per $x$ ) | $\mathbf{0 . 6 5 2} \pm \mathbf{0 . 0 3 6}$ |
| Mistral-7B + KTO (one $y$ per $x$ ) | $0.631 \pm 0.036$ |
| Mistral-7B-Instruct | $0.621 \pm 0.031$ |

data can in fact be discarded while still outperforming DPO. A similar trend holds when discarding undesirable data. For different models and datasets, the optimal settings of $\lambda_{D}, \lambda_{U}$ differ.

We further verify this claim by aligning Mistral-7B on OpenAssistant using DPO (on $n$ pairs), standard KTO (on all $2 n$ outputs), and KTO where only one $y$ per $x$ is used. Since the output of one $y$ in OpenAssistant is not conditioned on the other outputs for the same input, the latter effectively captures the setting where the data is from an inherently unpaired distribution. Despite the one- $y$-per- $x$ setup decreasing the amount of training data by $72 \%$, the KTO-aligned model still outperforms both its DPO counterpart and the official instruction-tuned version of Mistral-7B (Jiang et al., 2023), as seen in Table 1.

On average, KTO improves performance across generative benchmarks. Zephyr- $\beta$ is a variant of Mistral-7B that has been instruction-tuned and DPO-aligned on the UltraFeedback dataset (Tunstall et al., 2023; Cui et al., 2023). We find that substituting KTO for DPO (and changing nothing else) improves performance across MMLU (0-shot) (Hendrycks et al., 2021), GSM8K (8-shot, CoT) (Cobbe et al., 2021), HumanEval (0-shot) (Chen et al., 2021), and BigBench-Hard (3-shot CoT) (Srivastava et al., 2022). On GSM8K, just swapping DPO for KTO improves performance by 13.5 points. Even when we align with KTO using only one $y$ per $x$ (i.e., reducing the data volume by half), we still outperform DPO on all but one benchmark.

### 4.3. Theoretical Analysis

KTO was designed with the motivation that even if it had to learn from a weaker signal, it would make up for this limitation with the fact that it has access to much more data in the real world, where thumbs-up/thumbs-down data is common but preferences are scarce and expensive to collect. So why does KTO perform as good or better than DPO in our experiments, when it sees the same amount of data? Data efficiency may not be the only answer. Our theoretical analysis suggests that preference likelihood can
Table 2. Aligning Zephyr (Tunstall et al., 2023), a derivative of Mistral-7B, on UltraFeedback with KTO instead of DPO improves results across a suite of benchmarks. This is true even when only one of the two outputs in each preference is seen by KTO, despite this reducing the volume of data by half (one- $y$-per- $x$ ).

| Dataset $(\rightarrow)$ | MMLU | GSM8k <br> EM | HumanEval <br> pass@ 1 | BBH <br> EM |
| :--- | :--- | :--- | :--- | :--- |
| Metric $(\rightarrow)$ | EM | E |  |  |
| Zephyr- $\beta$ SFT | 57.2 | 39.0 | 30.1 | 46.3 |
| +DPO | 58.2 | 40.0 | 30.1 | 44.1 |
| +KTO | $\mathbf{5 8 . 6}$ | $\mathbf{5 3 . 5}$ | $\mathbf{3 0 . 9}$ | $\mathbf{5 2 . 6}$ |
| +KTO (one- $y$-per- $x$ ) | 58.0 | 50.0 | 30.7 | 49.9 |

be maximized without necessarily maximizing underlying human utility and that KTO implicitly ignores noisy and intransitive data.

Proposition 4.1. KTO does not learn from undesirable examples with sufficiently high rewards or desirable examples with sufficiently low rewards.

Informally, if an example is too difficult to learn from, then the KTO update will not change $\pi_{\theta}$. This may be a blessing in disguise, since human preferences are often noisy and not every given preference can be recovered with the true reward $r^{*}$ (Hoeffler \& Ariely, 1999). This means that it may be useful to avoid unlearnable preferences. However, this is a double-edged sword: it also means that KTO could end up ignoring some data that is hard-to-learn but necessary to recover $r^{*}$, resulting in under-fitting.

Theorem 4.2. Assuming the value function is logistic, for any bounded reward function $r_{a}$, there exists a reward function in its equivalence class (i.e., $r_{b}(x, y)=r_{a}(x, y)+h(x)$ for some $h(x)$ ) that induces the same optimal policy $\pi^{*}$ and Bradley-Terry preference distribution but a different human value distribution.

A key insight from Rafailov et al. (2023) is that reward functions in the same equivalence class (i.e., differing only in an input-specific component) induce the same optimal policy under (2) and the same Bradley-Terry preference distribution. However, we show under mild assumptions that the value distribution-i.e., human utility-is affected by such input-specific changes, so maximizing preference likelihood does not mean one is maximizing human utility. Approaches that directly maximize utility, such as KTO, may thus perform better in open-ended evaluation.

Theorem 4.3. Let two humans $a, b$ have value functions $v_{a}, v_{b}$ and contradicting preferences $y_{1} \succ_{a} y_{2}$ and $y_{2} \succ_{b} y_{1}$ for some input $x$. Assume $\pi_{\text {ref }}(y \mid x)=0 \Longrightarrow \pi_{\theta}(y \mid x)=0$ for all $x, y$. In the worst-case, the optimal policy under DPO decreases the expected value of both humans. In contrast, if each preference is broken up into two examples, then KTO (with default settings) does not change the policy.

Informally, we assume that humans want the model to in-
crease and decrease the probability of generations they like and dislike respectively. However, the preferences of two humans often contradict, leading to a dataset containing intransitive preferences. In the worst-case, DPO allows one of the two preferences to be recovered while decreasing the expected value of both humans. In contrast, KTO will change nothing at all in any case. Since existing datasets contain preferences from multiple annotators, the existence of intransitivity may help explain why KTO works better.

### 4.4. KTO vs. DPO - when to use which?

When human feedback is in a binary format, and especially when there is an imbalance between the number of desirable and undesirable examples, KTO is the natural choice. When your data is in the form of preferences, the choice is less clear. Putting aside the greater data efficiency of KTO, our theoretical analysis suggests that if your preference data has sufficiently little noise and sufficiently little intransitivity, then DPO will work better, since there is some risk of KTO underfitting. But if there is enough noise and transitivity, then the better worst-case guarantees of KTO will win out. Most publicly available preference datasets (e.g., SHP, OpenAssistant) contain noisy feedback from many different humans whose preferences likely contradict, which explains why KTO was able to match or exceed DPO performance in our experiments. Even AI feedback can be noisy and intransitive, which helps explain why KTO outperforms DPO when aligning with the synthetic UltraFeedback data.

## 5. Related Work

Human feedback has been used to improve LLM capabilities in translation (Kreutzer et al., 2018), summarization (Stiennon et al., 2020), sentiment-conditioned generation (Ziegler et al., 2019), and instruction-following (Ouyang et al., 2022). The RLHF framework (Christiano et al., 2017; Bai et al., 2022) traditionally used to accomplish this is detailed in $\S 2$.

Still, momentum has largely shifted in favor of closed-form losses that directly operate on offline preferences, such as DPO (Rafailov et al., 2023). This single stage of optimization distinguishes DPO from the conventional approach in preference-based RL, which learns a reward and then fits the policy to those rewards (Jain et al., 2013; Busa-Fekete et al., 2014). A recent string of work has centered on the idea of "self-training" or "self-play", during which new preference data is inferred from a model's generations (Chen et al., 2024; Yuan et al., 2024). Despite not being a human-aware loss, unlikelihood training was among to first to methods to align language models using a binary signal (Welleck et al., 2019). However, work by Korbak et al. (2023) found that it is worse than the CSFT baseline we tested in our work.

Prospect theory, despite being highly influential in behav- ioral economics, has had a fairly muted impact in machine learning, with work concentrated in human-robot interaction (Kwon et al., 2020; Sun et al., 2019; Chan et al., 2021). Learning from sparse binary feedback is a staple of information retrieval and recommender systems (He et al., 2017; Koren et al., 2009), although to our knowledge it has not been used to generate open-ended text.

## 6. Future Work

The existence of HALOs raises many questions. For one, the KTO loss is based on the Kahneman-Tversky value function for monetary gains and losses, which is almost certainly different from how humans perceive the relative goodness of text. What value function-and corresponding HALO - best describes how humans perceive language?

Given that the data that KTO needs is much more abundant, cheaper, and faster to collect-both as human and AI feedback-how far can we push synthetic data? For example, if we wanted to create a toxicity dataset to align our models to be less toxic, creating a tuple $\left(x, y_{w}, y_{l}\right)$ where $y_{l}$ is more toxic than $y_{w}$ is non-trivial. However, with KTO, we can easily create a dataset where desirability is determined by some black-box toxicity detection API. What other kinds of desiderata can we synthetically optimize for with KTO? Can we convert signals like "conversation lead to sale made" or "support ticket resolved" into KTO data?

Currently, KTO can learn from score-based data when the score is used to infer desirability. However, can we design a HALO where scores are directly incorporated into this loss?

## 7. Conclusion

We proposed a class of functions called human-aware losses (HALOs) based on the idea of a Kahneman-Tversky value function, which models some of the key cognitive biases that inform how humans make decisions about uncertain outcomes. We showed that among existing alignment methods, those that met the definition of a HALO performed better than those that did not, suggesting a benefit to the modeling of human biases. We then designed a human-aware loss called KTO for directly maximizing the utility of generations instead of maximizing preference likelihood. Despite only learning from a binary signal of whether an output is (un)desirable, KTO is as good or better than DPO at scales from 1B to 30B. Still, we make no claims that KTO is the best HALO for all scenarios; there remains much work to be done in discovering the optimal human-aware for each setting.

## Acknowledgements

We thank Dilip Arumugam and Arya McCarthy for feedback on the paper and Nathan Lambert for feedback on an early version of this draft. We thank Stas Bekman and Gautam Mittal for cluster assistance and Alex Manthey for helping with human evaluation.

## References

Baheti, A., Lu, X., Brahman, F., Bras, R. L., Sap, M., and Riedl, M. Improving language models with advantage-based offline policy gradients. arXiv preprint arXiv:2305.14718, 2023.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397-2430. PMLR, 2023.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324-345, 1952.

Busa-Fekete, R., Szörényi, B., Weng, P., Cheng, W., and Hüllermeier, E. Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine learning, 97:327-351, 2014.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Chan, L., Critch, A., and Dragan, A. Human irrationality: both bad and good for reward inference. arXiv preprint arXiv:2111.06956, 2021.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z., and Sun, M. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

Ethayarajh, K., Choi, Y., and Swayamdipta, S. Understanding dataset difficulty with $\mathcal{V}$-usable information. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 5988-6008. PMLR, 17-23 Jul 2022.

Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.

Go, D., Korbak, T., Kruszewski, G., Rozen, J., Ryu, N., and Dymetman, M. Aligning language models with preferences through f-divergence minimization. arXiv preprint arXiv:2302.08215, 2023.

Gurevich, G., Kliger, D., and Levy, O. Decision-making under uncertainty-a field study of cumulative prospect theory. Journal of Banking \& Finance, 33(7):1221-1229, 2009.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pp. $173-182,2017$.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Hoeffler, S. and Ariely, D. Constructing stable preferences: A look into dimensions of experience and their impact on preference stability. Journal of consumer psychology, 8 $(2): 113-139,1999$.

Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pp. 65-70, 1979.

Jain, A., Wojcik, B., Joachims, T., and Saxena, A. Learning trajectory preferences for manipulators via iterative improvement. Advances in neural information processing systems, 26, 2013.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Kahneman, D. and Tversky, A. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263-292, 1979.

Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., et al. Openassistant conversationsdemocratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.

Korbak, T., Shi, K., Chen, A., Bhalerao, R. V., Buckley, C., Phang, J., Bowman, S. R., and Perez, E. Pretraining language models with human preferences. In International Conference on Machine Learning, pp. 1750617533. PMLR, 2023.

Koren, Y., Bell, R., and Volinsky, C. Matrix factorization techniques for recommender systems. Computer, 42(8): 30-37, 2009.

Kreutzer, J., Uyheng, J., and Riezler, S. Reliability and learnability of human bandit feedback for sequence-tosequence reinforcement learning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17771788,2018

Kwon, M., Biyik, E., Talati, A., Bhasin, K., Losey, D. P., and Sadigh, D. When humans aren't optimal: Robots that collaborate with risk-aware humans. In Proceedings of the 2020 ACM/IEEE international conference on humanrobot interaction, pp. 43-52, 2020.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/ alpaca_eval, 2023.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.
Peters, J. and Schaal, S. Reinforcement learning by rewardweighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pp. 745-750, 2007.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. $A d$ vances in Neural Information Processing Systems, 33: 3008-3021, 2020.

Sun, L., Zhan, W., Hu, Y., and Tomizuka, M. Interpretable modelling of driving behaviors in interactive driving scenarios based on cumulative prospect theory. In 2019 IEEE Intelligent Transportation Systems Conference (ITSC), pp. 4329-4335. IEEE, 2019.

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C. Fine-tuning language models for factuality. arXiv preprint arXiv:2311.08401, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., and Wolf, T. Zephyr: Direct distillation of lm alignment, 2023.

Tversky, A. and Kahneman, D. Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5:297-323, 1992.

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., and Huang, S. Trl: Transformer reinforcement learning. https://github. com/huggingface/trl, 2020.

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., and Weston, J. Neural text generation with unlikelihood
training. In International Conference on Learning Representations, 2019.

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J., and Weston, J. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 2024.

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
</end of paper 3>


<paper 4>
# DI Orca-Math: Unlocking the potential of SLMs in Grade School Math 

Arindam Mitra, Hamed Khanpour, Corby Rosset, Ahmed Awadallah

Microsoft Research


#### Abstract

We show that an SLM can reach $\sim 87 \%$ pass@1 on GSM8K while trained on only $200 \mathrm{~K}$ synthetic math problems. Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over $80 \%$ accuracy on the GSM8K benchmark, is 34 billion parameters. To reach this level of performance with smaller models, researcher often train SLMs to generate Python code or use tools to help avoid calculation errors. Additionally, they employ ensembling, where outputs of up to 100 model runs are combined to arrive at a more accurate result. Result selection is done using consensus, majority vote or a separate a verifier model used in conjunction with the SLM. Ensembling provides a substantial boost in accuracy but at a significant cost increase with multiple calls to the model (e.g., Phi-GSM uses top-48 to boost the performance from 68.2 to 81.5, [38] uses top-100 to boost LLAMA-2's performance from $38.6 \%$ to $71.9 \%)$.


In this work, we present Orca-Math, a 7-billion-parameter SLM based on the Mistral-7B, which achieves $86.81 \%$ on GSM8k without the need for multiple model calls or the use of verifiers, code execution or any other external tools. Our approach has the following key elements: (1) A high quality synthetic dataset of $200 \mathrm{~K}$ math problems created using a multiagent setup where agents collaborate to create the data, (2) An iterative learning techniques that enables the SLM to practice solving problems, receive feedback on its solutions and learn from preference pairs incorporating the SLM solutions and the feedback. When trained with Supervised Fine-Tuning alone, Orca-Math achieves $81.50 \%$ on GSM8k pass @1 metric. With iterative preference learning, Orca-Math achieves $86.81 \%$ pass@1. Orca-Math surpasses the performance of significantly larger models such as LLAMA-2-70B, WizardMath-70B, Gemini-Pro, ChatGPT-3.5. It also significantly outperforms other smaller models while using much smaller data (hundreds of thousands vs. millions of problems).[^0]

## 1 Problem Setup

Frontier Language Models such as GPT-4 [1 have demonstrated capabilities previously unseen in smaller models, most notably the remarkable ability to reason (e.g. mathematical reasoning that requires both language comprehension and mathematical understanding). These capabilities have been largely attributed to the very large scale the model size, the dataset size and ultimately the amount of compute needed for training.

Several recent studies have focused on improved the reasoning abilities of small language models (SLMs). Despite that the extent to which scale is needed for achieving reasoning capabilities is still an open research question.

One of the promising directions of improving the reasoning capabilities of SLMs is using frontier language models, such as GPT-4, to create tailored and high-quality synthetic data that can be used to train the SLM. The high quality of the training data and the ability to elicit richer learning signals (e.g. explanations) have been show to significantly improve SLMs abilities in acquiring skills that had only emerged before at much larger scale.

This paradigm fits under a teacher-student approach where the large model (the teacher) is creating demonstrations for the SLM (the student) to learn from. In this work we further explore this direction with focus on mathematical reasoning on grade school math world problem, using the popular GSM8K benchmark.

Several other studies have demonstrated positive results on GSM8K recently with SLMs, e.g. Phi-GSM [21, OVM 38], etc. However, many of them employ ensembling, where outputs of up to 100 model runs are combined to arrive at a more accurate results. Result selection is done using, consensus, majority vote or by using a separate a verifier model to score/verify the outputs and select the best answer. Ensembling provides a substantial boost in accuracy (e.g., Phi-GSM uses top-48 to boost the performance from 68.2 to 81.5 , [22] uses top-100 to boost LLAMA-2's performance from $38.6 \%$ to $71.9 \%)$. However it comes at a significant increase in cost with multiple calls to the model, generating and verifying a 100 different solutions requires 200 different calls to the models. Additionally, some of them use very larger amounts of data (e.g. 12M for Phi-GSM) or use tools or code to avoid calculation errors.

In this work, we extend the teacher-student paradigm to an iterative learning settings with high-quality synthetic training data as follows:

- We create Orca-Math-dataset, a synthetic dataset of $200 \mathrm{~K}$ math problems, paired with GPT-4-Turbo solutions. The dataset was generated using an agent-based setup, hereby referred as, Agent-Instruct, that not only paraphrases existing problems but aims to expand the problem set both in diversity and difficulty.
- We introduce an iterative learning procedure where we: (1) use the dataset for supervised finetuning to train the SLM on demonstrations, (2) allow the SLM to practice generating multiple solutions and (3) use the teacher to provide feedback to the student. The feedback comes in the form of evaluating the solutions generated by the student or providing a teacher solution.

With the supervised finetuning alone, we achieve $81.50 \%$ on GSM8k at pass@1 metric. The iterative learning loop further improves the pass@1 to $86.81 \%$. without the need for multiple model calls or the use of verifiers, code execution or any other external tools. The model exceeding much bigger models like LLAMA-2-70B (56.8\%), WizardMath-70B (81.6\%), Gemini Pro ( $86.5 \%$ with 32 trials) and GPT-3.5 (77.4\%). Most notably it can reach this level with only $200 \mathrm{~K}$ examples (orders of magnitude less than other datasets).

| $\overline{\text { Model }}$ | Base model | Model size | Answer format | Eval method | GSM8K (\%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-2 34] | - | 7B | nlp | pass@1 | 14.6 |
|  |  | 13B |  |  | 28.7 |
|  |  | $34 \mathrm{~B}$ |  |  | 42.2 |
|  |  | 70B |  |  | 56.8 |
| MetaMath [39] | Llama-2 | $7 \mathrm{~B}$ | nlp | pass@1 | 66.5 |
|  |  | 13B |  |  | 72.3 |
|  |  | $70 \mathrm{~B}$ |  |  | 82.3 |
| WizardMath [23] | Llama-2 | $7 \mathrm{~B}$ | nlp | pass@1 | 54.9 |
|  |  | 13B |  |  | 63.9 |
|  |  | 70B |  |  | 81.6 |
| MAmmoTH [42] | Code-Llama | 7B | code | pass@1 | 59.4 |
|  | Code-Llama | 12B |  |  | 64.7 |
|  | Code-Llama | $34 \mathrm{~B}$ |  |  | 72.7 |
|  | Llama-2 | 70B |  |  | 76.9 |
| Mistral 14$]$ | - | $7 \mathrm{~B}$ <br> $8 \times 7 \mathrm{~B}$ | nlp | maj1@8 | 52.2 <br> 584 |
| OVM 38 | Llama-2 | $7 \mathrm{~B}+7 \mathrm{~B}$ | nlp | verify100@1 | 73.7 |
|  | Mistral | $7 \mathrm{~B}+7 \mathrm{~B}$ |  |  | 84.7 |
| Llemma [4] | Llama-2 | 7B | nlp | pass@1 | 36.4 |
|  |  | $34 \mathrm{~B}$ |  |  | 51.5 |
| ToRA-Code [12] | Llama-2 | $7 \mathrm{~B}$ |  |  | 72.6 |
|  |  | 13B | code | COT@1 | 75.8 |
|  |  | $34 \mathrm{~B}$ |  |  | 80.7 |
|  |  | 70B |  |  | 84.3 |
| Orca. $2 \longdiv { 2 7 }$ |  | $\overline{7 B}$ | $n \ln .$, | nacs@1 | $\overline{55.72}$ |
| Urca $2[2 \pi]$ | Llama-2 | 13B | nIp | passe 1 | 65.73 |
| Gemini Pro | - | - | nlp | maj1@32 | 86.5 |
| Gemini Ultra 11 | - |  |  |  | 94.4 |
| GPT-3.5-0613 | - | - | code | pass@1 | 77.4 |
| GPT-4-0613 [29] |  |  |  |  | 97.0 |
| Phi-1.5 19 |  | $1.3 \mathrm{~B}$ | code | pass@1 | 44.6 |
|  | Phi-1.5-tiny | $125 \mathrm{M}$ |  |  | 63.1 |
| Phi-CSM 21 | Phi-1.5-small | $350 \mathrm{M}$ |  |  | 65.9 |
| Phl-GSM [21] | Phi-1.5 | $1.3 \mathrm{~B}$ | code | pass@1 | 68.2 |
|  | Phi-2 | $2.7 \mathrm{~B}$ |  |  | 74.3 |
|  | Phi-1.5-tiny | $125 \mathrm{M}+125 \mathrm{M}$ |  |  | 68.9 |
| Phi-GSM+V [21] | Phi-1.5-small | $350 \mathrm{M}+350 \mathrm{M}$ | code | verify48@1 | 71.3 |
|  | Phi-1.5 | $1.3 \mathrm{~B}+1.3 \mathrm{~B}$ |  |  | 81.5 |
| Orca-Math | Mistral | $7 \mathrm{~B}$ | nlp | pass@1 | 86.81 |

Table 1: Results on GSM8K. The table is repurposed from [21]. Bold labels indicate accuracies exceeding $\mathbf{8 0 \%}$. The term ' $8 \times 7 \mathrm{~B}$ ' represents a blend of 8 experts, with each expert having $7 \mathrm{~B}$ parameters. '7B+7B' refers to the union of a 7B generation model and a 7B verification model. The addition of verifier models is signified by '+V'.

## 2 Dataset Construction: Agent-Instruct

The goal of this step is to create a diverse set of grade school math word problems that contains both easy and hard problems. Towards this goal we create a variety of agents.

Seed Set We start by collecting sample math word problems from existing open-source datasets, namely NumGLUE [26, AddSub [13], ALGES [17], ASDiv [24], DRAW [35], GSM8k [7], MATHQA [2], MultiArith [32], SingeOP [33], SingleEQ [16], and SVAMP [30]. We collect a total of 36,217 problems. We utilize the Lila [25] benchmark to collect the datasets. Specifically, we collect problems from the train and validation splits from Lila to construct the seed set. Interested readers, please refer to Lila [25].

Agent - Ask Me Anything We expand the seed set by creating multiple word problems from each problem in the seed set. We utilize the subsequent prompt for problem creation.

Your goal is to create multiple word problems from a given word problem and its answer. First convert the question of the word problem into a statement. Then for each number in the converted problem create a new word problem. Here are some examples:

Example 1: Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: 72

Replacing question with statement: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia sold altogether 72 clips in April and May.

## All questions:

$<$ target $>48$

<question> Natalia sold clips to some of her friends in April, and then she sold half as many clips in May. Natalia sold altogether 72 clips in April and May. How many clips did she sell in April?

$<$ target $>$ half

<question> Natalia sold clips to 48 of her friends in April, and then she sold some clips in May. Natalia sold altogether 72 clips in April and May. What is the ratio of the number clips sold in April to number clips sold in May?

$<$ target $>72$

<question > Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Example 2: Q: Weng earns $\$ 12$ an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Answer: 10

Replacing question with statement: Weng earns $\$ 12$ an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. She earned $\$ 10$.

## All questions:

$<$ target $>12$

<question> Weng earns a certain amount per hour for babysitting. Yesterday, she just did 50 minutes of babysitting and earned 10. How much does she earn per hour?

$<$ target $>50$

<question> Weng earns 12 an hour for babysitting. Yesterday, she just did some babysitting and earned 10. How much time did she spend on babysitting?
$<$ target $>10$

<question> Weng earns 12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Example 3: Q: Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

Answer: 5

Replacing question with statement: Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents gave her twice as much as her parents. She needs 5 more to buy the wallet.

## All questions:

$<$ target $>100$

<question> Betty is saving money for a new wallet. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. She needs 5 more to buy the wallet. What is the cost of the wallet?

$<$ target $>$ half

<question> Betty is saving money for a new wallet which costs 100 . She has some money saved, her parents decided to give her 15 , and her grandparents gave her twice as much as her parents. Now, Betty needs 5 more to buy the wallet. What is the ratio of the money Betty have saved initially to the cost of wallet?

$<$ target $>15$

<question> Betty is saving money for a new wallet which costs 100 . She has half of the money she needs, her parents decided to give her some money, and her grandparents gave her twice as much as her parents. Now, Betty needs 5 more to buy the wallet. How much money did her parents give her?

$<$ target $>$ twice

<question> Betty is saving money for a new wallet which costs 100 . Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents also chipped in. Now, Betty needs 5 more to buy the wallet. What is the ratio of the amount given by her grandparents to the amount given by her parents?

$<$ target $>5$

<question> Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

## Now solve this:

Example 4: Q: Your teacher is giving a test worth 200 points. There is a total of 30 5 -point and 10 -point questions. How many 5 -point questions are on the test? Answer: 20

Note that, the "Ask Me Anything" agent is generating problems based on the seed in Example 4. Examples 1 to 3 are provided as few-shot demonstrations. This agent creates a total of 120, 445 new problems. It is important to note that all the generated problems exhibit a similar narrative to that of the seed word problem. The solutions to these word problems are generated using GPT4-Trubo.

Agent - Suggester \& Editor We further expand the seed set by developing challenging problems. To achieve this, we introduce two new agents, namely Suggester and Editor. This setup involves the establishment of a team of agents working collaboratively to create a dataset geared toward a predefined objective. In this instance, the objective is to modify an existing problem to increase its difficulty.

The Suggester examines a specific problem and proposes several methods for enhancing its complexity without creating the actual problem. Subsequently, the Editor takes the original word problem and the Suggester's recommendations to generate an updated, more challenging problem. This iterative process can occur over multiple rounds, with each round further increasing the complexity of the previously generated problem. We utilize the AutoGen 37] framework to implement the multi-agent workflow. Here is an example of the iterative process:

Seed Problem: Every hour Joanne has to collect the coins out of the fountain inside the mall. During the first hour, she collected 15 coins. For the next two hours, she collected 35 coins from the fountain. In the fourth hour, she collected 50 coins from the fountain but she gave 15 of them to her coworker so she could buy a soda. How many coins did she have after the fourth hour?

Suggester: Increase the number of hours, add more variables for each hour, and incorporate additional transactions.

Editor: Every hour, Joanne has to collect the coins from two separate fountains inside the mall - Fountain A and Fountain B. During the first hour, she collected 20 coins from Fountain A and 30 coins from Fountain B. In the next hour, she collected 25 coins more than she collected in Fountain A during the previous hour and 10 coins less than she collected in Fountain B during the previous hour. In the third hour, she collected the same number of coins in Fountain A as she collected in the second hour and additional 45 coins this time from Fountain B. During the fourth hour, she collected 90 coins in total, but she gave 20 coins to her coworker to buy a soda and 40 coins to another coworker to buy a meal. After the fifth hour, Joanne decided to donate 15

Suggester: Incorporate more fountains, increase the duration, and add more complex calculations for coins collected and spent or donated.

Editor: Joanne has to collect coins from four separate fountains inside the mall Fountain A, Fountain B, Fountain C, and Fountain D every hour for 8 hours. In the first hour, she collects 20 coins from Fountain A, 30 coins from Fountain B, 40 coins from Fountain C, and 50 coins from Fountain D. In the subsequent hours, her collections have the following fluctuations:

- For Fountain A, she collects 10 more coins each hour than she collected in the previous hour.
- For Fountain B, her collections decrease by 5 coins each hour compared to the previous hour.
- For Fountain C, she collects double the number of coins each hour compared to the number of coins collected in Fountain A the same hour.
- For Fountain D, the number of coins collected remains constant at 50 coins each hour.

At the end of the third hour, she donates 20

During the seventh hour, she has to share coins with her coworkers and splits them as follows: 15 coins for coworker one, 20 coins for coworker two, and she keeps 70

After 8 hours, Joanne decides to count her collected coins. How many coins did she have at the end of the eighth hour?

We allow two rounds of iterations per problem and filter problems where the GPT4-turbo generated answer exceeds 1800 characters. At the end of this process, we collect 37,157 problems.

DMath Furthermore, we include 6, 216 problems sourced from DMath [15]. These problems represent a subset of the 7,943 problems present in the DMath training set, in which the solution computed by GPT4-Turbo aligns with the precise gold-standard answer.

## 3 Training

### 3.1 Supervised Fine-Tuning Experiment (Iteration \#1)

We finetune Mistral-7B on the Orca-Math-200K dataset. We have not used packing. The data is presented in the following instruction format:

$$
\text { USER:\n\{question\}ไn\nASSISTANT:\n\{answer\} }
$$

The loss is computed only on the answer tokens. We employ a constant learning rate of $1 \times 10^{-6}$. The per-device batch size is set to 3 . Training is conducted for one epoch on eight A100 nodes, with each node containing eight GPUs.

### 3.2 Iterative Learning from both Positive and Negative Signals

Dataset Construction Iteration \#2 To generate additional positive and negative solutions for each problem, we sample four responses from the SFT-tuned model from iteration \#1. Specifically, we utilize top_p $=0.95$ and temperature $=0.7$. This process results in a dataset where each of the 200,000 problems has one GPT4-Turbo generated solution and four student-generated solutions. Subsequently, we employ the prompt defined in GPT4-Based-Exact-Match (See section 4 for details) to assess the alignment between the teacher's (GPT4-Turbo) answer and the student's answer. For all solutions where the student-generated answer does not match the teacher's answer, we label them as negative; otherwise, we label the solution as positive. We then construct the preference dataset as follows:

- For each question, $q_{i}$ we construct $q_{i}^{+}$, the set of all positive solutions for $q_{i}$. We treat the teacher solution as positive, thus this set by construction contains at least one element.
- For each question, $q_{i}$ we also construct $q_{i}^{-}$, the set of all negative solutions for $q_{i}$. This set can be empty if all the 4 responses are are aligned wrt the teacher's solution. Infact, this is the case for around $80 k$ questions. For such situations, we randomly sample one response from $q_{j}^{-}$for 4 different $q_{j}$ where $j \neq i$ and $\left|q_{j}^{-}\right|>0$. Note that, for this special situation $\left|q_{i}^{+}\right|=4$.
- Let, $Q_{i}=\left\{\left(q_{i}, a_{i}^{+}, a_{i}^{-}\right) \mid\left(a_{i}^{+}, a_{i}^{-}\right) \in q_{i}^{+} \times q_{i}^{-}\right\}$be the preference dataset around $q_{i}$. The final preference dataset is created by taking the union of $Q_{i}$ for all $q_{i}$ in the training dataset.

Dataset Construction Iteration \#3 Let M2 denote the model trained with KTO [10] on the dataset constructed for Iteration \#2. We replicate the same procedure for the construction of dataset for Iteration \#3; however, we utilize M2 to generate the four responses instead of the SFT-tuned model from iteration \#1.

To learn from both positive and negative feedback, we have evaluated the performance of two algorithms: the Direct Preference Optimization (DPO) as described by [31 and the Kahneman-Tversky Optimization (KTO) introduced by [10]. DPO is a simple and popular approach for efficiently fine-tuning language models to align with preferences. Additionally,
we have explored the capabilities of KTO, which distinguishes itself by requiring only a binary "yes" or "no" response to assess the quality of an output.

## 4 Evaluation

We use exact match as the metric. Given a model generated answer, we prompt GPT4 to extract the final short answer and match it with the gold short answer. We will refer to this metric as, GPT4-based-Exact-Match. The following figure shows the prompt template:

## SYSTEM

As an expert Math teacher, your role is to evaluate a student's answer to a word problem. The problem is accompanied by a correct solution provided by the problem setter. It is important to remember that there may be various methods to solve a word problem, so the student's steps might not always align with those in the problem setter's solution. However, the final answer, typically a number, should be unique and match the problem setter's answer. Your task involves analyzing the student's solution to identify any mistakes and determine whether the answer can be modified to correct the error. If the student's answer is unfixable, consider creating practice problems to help improve their understanding.

Use the following format:

Error Analysis: In one sentence, extract the final answer from the problem setter's solution and compare it with the student's answer. Do they match?

Final Verdict: Correct/Incorrect

## USER

Question: Billy is volunteering his time to help people do their taxes. He can help 2 people per hour for 3 hours a day. If he takes $20 \%$ of the days between March 1st and April 19th off, and helps people on all the other days. How many people does he help? (Remember there are 31 days in March.)

Problem Setter's answer:

First find the number of days between March 1st and April 19th: 31 days +19 days $=$ 50 days

Then subtract the percentage of time Billy takes off from $100 \%$ to find the percentage of time he works: $100 \%-20 \%=80 \%$

Then multiply that number by $80 \%$ to find the number of days Billy works: 50 days * $80 \%=40$ days

Then find how many people Billy can help per day: 2 people/hour $* 3$ hours/day $=6$ people/day

Then multiply the number of people he helps per day by the number of days he works to find the total number of people he helps: 6 people/day $* 40$ days $=240$ people

Student answer:

First, let's calculate the total number of days between March 1st and April 19th.

March has 31 days, and April has 30 days. So, the total number of days from March 1st to April 19th is:

$31($ March $)+19($ April $)=50$ days

Now, let's calculate how many days Billy takes off. He takes $20 \%$ of the 50 days off.

$20 \%$ of 50 days $=0.20 * 50=10$ days

So, Billy takes 10 days off. This leaves him with: 50 days -10 days $=40$ days

Billy helps people for 3 hours a day, and he can help 2 people per hour. So, in one day, he can help:

3 hours/day * 2 people/hour $=6$ people/day

Now, let's calculate how many people he helps over the 40 days he works:

40 days $* 6$ people/day $=240$ people

Therefore, Billy helps 240 people.

## ASSISTANT

Error Analysis: The student's final answer of helping 240 people matches the problem setter's solution.

Final Verdict: Correct

## 5 Results

Table 2 captures the performance of several training procedures on the GSM8k test set containing 1319 word problems. We finetune Mistral-7B for upto three iterations. In first iteration we use supervised finetuning to obtain M1. For the second iteration we compare SFT, DPO [31] and KTO. The KTO trained model performs better in this group. We call this M2 and use M2 to generate the dataset for iteration \#3. For third iteration, we compare DPO and KTO where M2 servers as the starting point. We also compare these against three epochs of SFT training on the Orca-Math-200K dataset. For all SFT training we employ a constant learning rate of $1 \times 10^{-6}$. The per-device batch size is set to 3 and number-of-epochs is set to 1 . For DPO and KTO training jobs, we set beta to 0.3 , per-device batch size to 3 , gradient-accumulation-steps to 11 and number-of-epochs 1. For DPO and KTO training in iteration \#2 we employ a constant learning rate of $1 \times 10^{-6}$ and for iteration \#3 a constant learning rate of $1 \times 10^{-7}$.

| Training Procedure | Pass@1 Accuracy on GSM8K Test set |
| :---: | :---: |
| SFT (M1) | 79.91 |
| SFT (M1) $\rightarrow$ SFT | 81.50 |
| $\mathrm{SFT}(\mathrm{M} 1) \rightarrow \mathrm{DPO}$ | 84.23 |
| SFT (M1) $\rightarrow$ KTO (M2) | 85.06 |
| $\mathrm{SFT}(\mathrm{M} 1) \rightarrow \mathrm{SFT} \rightarrow \mathrm{SFT}$ | 80.44 |
| $\mathrm{SFT} \rightarrow \mathrm{KTO}(\mathrm{M} 2) \rightarrow \mathrm{DPO}$ | 84.91 |
| SFT $\rightarrow$ KTO (M2) $\rightarrow$ KTO (Orca-Math) | $86.87 \quad-1$ |

Table 2: Table captures the performance of several iterative learning experiments and baselines on the GSM8k test set. SFT stands for one epoch of training on the Orca-Math200K dataset. SFT $\rightarrow$ SFT stands two epochs of training on Orca-Math-200K. SFT $\rightarrow$ DPO (KTO) stands for one epoch of training on dataset for iteration \#2 with DPO (KTO) starting with M1. SFT $\rightarrow$ SFT $\rightarrow$ SFT stands for three epochs of training on Orca-Math-200K. SFT $\rightarrow$ KTO $\rightarrow$ DPO (KTO) stands for one epoch of training on dataset for iteration \#3 with DPO (KTO) starting with M2. For evaluation, we employ greedy decoding.

### 5.1 Ablation Studies

Model Generated Positives We study the impact model generated positives by limiting $q_{i}^{+}$to contain only teacher generated solution. In other words we remove any $a_{i}^{+}$that is model generated in the creation of the dataset for iteration \#2. Table 3 shows the result of training M1 with DPO and KTO on this dataset for one epoch. We reuse the hyperparameters for iteration \#2. Irrespective of the training algorithm, we see significant performance drop.

Synthetic Negatives The preference dataset creation involves synthetic negative creation in the situation where all four responses generated from M1 or M2 are positive. We study

| M1 $\rightarrow$ DPO | $81.96(-2.27)$ |
| :---: | :--- |
| M1 $\rightarrow$ KTO | $82.79(-2.27)$ |

Table 3: Table captures that student generated positives provide important supervision.

the impact of these synthetic negatives by ignoring the questions, $q_{i}$, where all sampled responses are positive (Table 44). This reduces the number of questions for iteration \#2 by around $80 k$ and for iteration $\# 3$ by around $104 k$.

| M1 $\rightarrow$ DPO | $60.73(-23.5)$ |
| :--- | :--- |
| M1 $\rightarrow$ KTO | $85.22(+0.17)$ |
| M $1 \rightarrow$ KTO $\rightarrow$ KTO | $85.44(-1.43)$ |

Table 4: Table shows that the inclusion of problems where all sampled responses are positive is beneficial.

### 5.2 Math Benchmarks beyond GSM8k

Table 5 presents the performance of Orca-Math on several other word problem datasets. For ease of evaluation, we selected datasets where the answer to each problem is a single number. The test sets of the benchmarks are obtained from Lila. We employ the GPT4-based exact-match metric, and model responses are generated using greedy decoding.

| Test Set | Orca-Math-Sft (M1) | Orca-Math |
| :--- | :---: | :---: |
| Addsub | 88.99 | 91.74 |
| ASDiv | 91.10 | 91.10 |
| MultiArith | 98.28 | 98.28 |
| SingleOp | 98.74 | 99.37 |
| SinglEq | 97.25 | 99.08 |
| Svamp structured | 87.63 | 91.30 |

Table 5: Performance of SFT trained model from Iteration \#1 (M1) and Orca-Math on AddSub, ASDiv, MultiArith, SingleOp, SinglEq, Svamp structured

### 5.3 Contamination Check

We never use the test split of GSM8K or any other datasets during training or as seeds for synthetic problem generation. Nevertheless, We take the following approach for detecting any potential text contamination.

1. We begin by preprocessing the texts, which includes converting all characters to lowercase, removing punctuation, tokenizing the text into individual words, and removing common English stopwords to ensure uniformity in the data.
2. We then vectorize our text corpus using the Term Frequency-Inverse Document Frequency (TF-IDF) method and determine the cosine similarity between the test and training sets, from which we select the top- $\mathrm{k}(\mathrm{k}=10)$ most analogous questions for each test query.
3. Finally, we evaluate the extent of text contamination by counting the number of test questions with the highest n-gram overlap above a preset threshold of 0.5 with their corresponding training set matches. We calculate the overlap of n-grams between pairs of texts using the Jaccard similarity. To conduct a rigorous contamination check, we set $\mathrm{n}=1$. It is important to note that the $\mathrm{n}$-gram overlap, when measured using Jaccard similarity, is a non-increasing function of $n$.
4. Upon executing our algorithm, we determined that the count of test questions exhibiting significant n-gram overlap is eight, thus indicating negligible text contamination within our test set according to the defined threshold. When limiting the
train set to contain only the seed problems, the count of test questions exhibiting significant n-gram overlap is seven. Note that, for $n \geq 2$, the count of test questions exhibiting significant n-gram overlap is zero.

## 6 Related Works

The generation of synthetic data through generative artificial intelligence (AI) models has evolved rapidly. Numerous datasets [27, [20, 28, 23, 9, 88, 45, 6, 36] have been proposed for both specialized and generic domains, with math-related datasets [40, 43, 44, 18, being closely related to our work.

Learning from rich signals has also garnered significant attention recently. Several studies 31, 10, 22, 3, 5, 41, have demonstrated the usefulness of preference learning. In this work, we present a detailed analysis of agent-based synthetic data generation and iterative preference learning in the grade school level math domain. Specifically, we demonstrate the robustness of KTO over DPO and the effectiveness of using model-generated positives to improve model training. We believe this is a preliminary step toward iterative learning and self improvement of small language models in challenging domains.

## 7 Conclusions

Our study provides compelling evidence that the mathematical reasoning capabilities of Small Language Models (SLMs) can be substantially enhanced. By employing iterative learning techniques and leveraging both positive and negative signals, we have successfully surpassed the previously perceived $80 \%$ barrier on the GSM8k benchmark. Our 7B model, trained with $200 \mathrm{~K}$ data, achieved an impressive $86.81 \%$ accuracy. Furthermore, the incorporation of agents in dataset generation has proven to be a valuable approach, enabling the creation of more diverse and interesting datasets. These findings not only highlight the potential for significant improvements in SLM performance but also underscore the importance of innovative learning strategies and dataset generation methods in advancing the creation of powerful SLMs.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319, 2019.

[3] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

[4] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv: 2310.10631, 2023.

[5] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

[6] Daixuan Cheng, Shaohan Huang, and Furu Wei. Adapting large language models via reading comprehension. arXiv preprint arXiv:2309.09530, 2023.

[7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv: Arxiv$2110.14168,2021$.

[8] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

[9] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

[10] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

[11] Google Gemini Team. Gemini: A family of highly capable multimodal models.

[12] Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv: 2309.17452, 2023.

[13] Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 523-533, 2014.

[14] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv: $2310.06825,2023$.

[15] Jiwoo Kim, Youngbin Kim, Ilwoong Baek, JinYeong Bak, and Jongwuk Lee. It ain't over: A multi-aspect diverse math word problem dataset. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14984-15011, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.927. URL https://aclanthology.org/2023. emnlp-main. 927

[16] Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang. Parsing algebraic word problems into equations. Transactions of the Association for Computational Linguistics, 3:585-597, 2015.

[17] Nate Kushman, Yoav Artzi, Luke Zettlemoyer, and Regina Barzilay. Learning to automatically solve algebra word problems. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 271-281, 2014.

[18] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large scale language model society, 2023.

[19] Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.

[20] Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https:// huggingface.co/Open-Orca/OpenOrca, 2023.

[21] Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. Tinygsm: achieving $>80 \%$ on gsm $8 \mathrm{k}$ with small language models. arXiv preprint arXiv:2312.09241, 2023.

[22] Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657, 2023.

[23] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv: 2308.09583, 2023.

[24] Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772, 2021.

[25] Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. Lila: A unified benchmark for mathematical reasoning. arXiv preprint arXiv:2210.17517, 2022.

[26] Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. arXiv preprint arXiv:2204.05660, 2022.

[27] Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason, 2023.

[28] Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023.

[29] OpenAI. Gpt-4 technical report, 2023.

[30] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191, 2021.

[31] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

[32] Subhro Roy and Dan Roth. Solving general arithmetic word problems. arXiv preprint arXiv:1608.01413, 2016.

[33] Subhro Roy, Tim Vieira, and Dan Roth. Reasoning about quantities in natural language. Transactions of the Association for Computational Linguistics, 3:1-13, 2015.

[34] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: 2307.09288, 2023.

[35] Shyam Upadhyay and Ming-Wei Chang. Draw: A challenging and diverse algebra word problem set. Technical report, Citeseer, 2015.

[36] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.

[37] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.

[38] Fei Yu, Anningzhe Gao, and Benyou Wang. Outcome-supervised verifiers for planning in mathematical reasoning. arXiv preprint arXiv: 2311.09724, 2023.

[39] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv: 2309.12284, 2023.

[40] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

[41] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 2024.

[42] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv: 2309.05653, 2023.

[43] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.

[44] Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Templatemath: Syntactic data generation for mathematical problems, 2024.

[45] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206, 2023.
</end of paper 4>


