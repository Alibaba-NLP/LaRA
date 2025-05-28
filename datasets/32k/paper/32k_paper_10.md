<paper 0>
# Self-Discover: Large Language Models Self-Compose Reasoning Structures 

Pei Zhou ${ }^{1}$ Jay Pujara ${ }^{1}$ Xiang Ren ${ }^{1}$ Xinyun Chen ${ }^{2}$ Heng-Tze Cheng ${ }^{2}$<br>Quoc V. Le ${ }^{2}$ Ed H. Chi ${ }^{2}$ Denny Zhou ${ }^{2}$ Swaroop Mishra ${ }^{2}$ Huaixiu Steven Zheng ${ }^{2}$


#### Abstract

We introduce SELF-DISCOVER, a general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems that are challenging for typical prompting methods. Core to the framework is a selfdiscovery process where LLMs select multiple atomic reasoning modules such as critical thinking and step-by-step thinking, and compose them into an explicit reasoning structure for LLMs to follow during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as $32 \%$ compared to Chain of Thought (CoT). Furthermore, SELFDISCOVER outperforms inference-intensive methods such as CoT-Self-Consistency by more than $20 \%$, while requiring $10-40 x$ fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.


## 1. Introduction

Large Language Models (LLM) (Brown et al., 2020; Chowdhery et al., 2022; OpenAI, 2023b; Anil et al., 2023) powered by transformers (Vaswani et al., 2017) have produced impressive breakthroughs in generating coherent texts (OpenAI, 2022), and following instructions (Zhong et al., 2021; Mishra et al., 2022c; Wei et al., 2021; Chung et al., 2022; Ouyang et al., 2022). In pursuit of the goal to enhance LLMs' capability to reason and solve complex problems, various prompting methods have been proposed, drawing inspirations from cognitive theories of how humans rea-[^0]

Preprint. son. For example, few-shot and zero-shot chain-of-thought (CoT) (Nye et al., 2021; Wei et al., 2022; Kojima et al., 2022; Yasunaga et al., 2023) resembles how humans solve problems step-by-step, decomposition-based prompting (Zhou et al., 2022a; Drozdov et al., 2022; Patel et al., 2022; Hao et al., 2023; Khot et al., 2022) is inspired by how humans breakdown a complex problem into a series of smaller subproblems, and then solve those subproblems one by one (Polya, 2004), and step-back prompting (Zheng et al., 2023) is motivated by how humans reflect on task nature to derive general principles. However, a fundamental limitation is that each technique itself serves as an atomic reasoning module making an implicit prior assumption of the process on how to tackle a given task. Instead, we argue that each task has a unique intrinsic structure underlying the reasoning process involved in solving it efficiently. For instance, least-to-most prompting (Zhou et al., 2022a; Drozdov et al., 2022) has shown to be much more effective than CoT (Wei et al., 2022) at solving tasks such as symbolic manipulation and compositional generalization, due to the decomposition structure of the tasks.

This paper aims at self-discovering the underlying reasoning structure unique to each task, while being highly efficient in terms of computation. Our approach, SELF-DiscoVER, is inspired by how humans internally devise a reasoning program for problem-solving (Newell et al., 1958; Rasmussen, 1983), as illustrated in Figure 2 . From a set of atomic reasoning modules described in natural language such as "breakdown into sub tasks" and "critical thinking", an LLM, and task examples without labels, SELF-DisCOVER composes a coherent reasoning structure intrinsic to the task (Stage 1) and then solves instances of the task using the discovered structure (Stage 2). Stage 1 operates at the tasklevel and uses three actions to guide the LLM to generate a reasoning structure for the task. At Stage 2, during the final decoding, the LLM simply follows the self-discovered structure to arrive at the final answer.

Solving problems using SELF-DiscoVER brings several benefits compared to other methods for LLM reasoning. First, the discovered reasoning structure is grounded in atomic reasoning modules benefiting from the strengths of multiple reasoning modules in contrast to applying a priori module such as CoT. Second, SELF-DiscoVER is efficient

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-02.jpg?height=290&width=255&top_left_y=221&top_left_x=377)

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-02.jpg?height=716&width=1354&top_left_y=214&top_left_x=358)

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-02.jpg?height=390&width=664&top_left_y=534&top_left_x=359)

Self-Discover Over Chain-of-Thought

Figure 1. SELF-DISCOVER guides LLMs to self-discover and compose atomic reasoning modules into a reasoning structure to solve challenging tasks. Through testing on challenging reasoning benchmarks incuding Big Bench-Hard (BBH), agent reasoning (T4D), and MATH, we find that SELF-Discover outperforms Direct Answering on 23/25 and CoT on 21/25 tasks in zero-shot setting using PaLM 2-L. Full BBH results are in Appendix C Table 3.

in computation as it only requires 3 more inference steps on the task-level, while being more performant than inferenceheavy ensemble approaches such as self-consistency (Wang et al., 2022). Lastly, the discovered reasoning structure is intrinsic to the task, and conveys LLMs' insights about the task in a more interpretable way than the optimized prompts (Zhou et al., 2022b; Yang et al., 2023).

We test SELF-DiscoVER on 25 challenging reasoning tasks including Big Bench-Hard (BBH) (Suzgun et al., 2022), Thinking for Doing (T4D) (Zhou et al., 2023) and MATH (Hendrycks et al., 2021). SELF-DisCoVER outperforms CoT on 21/25 task with performance gains up to $42 \%$ (Figure 1), highlighting the advantage of the self-discovered reasoning structure composed from the atomic reasoning modules against a single a priori CoT module. Furthermore, we demonstrate that SELF-DISCOVER achieves superior performance against inference-heavy methods such as CoT + Self-Consistency and majority voting of every module while requiring 10-40x fewer inference compute (Figure 5). Finally, we compare SELF-DISCOVER with prompts optimized (OPRO) using a training set (Yang et al., 2023) (Figure 9). We find that SELF-DiscoVER still performs on par or better than OPRO while the self-discovered reasoning structure are much more interpretable.

We conduct a set of analysis to understand the effectiveness of SELF-DiscoVER. By breaking down BBH tasks into 4 different categories, we find that SELF-DISCOVER performs best on tasks requiring world knowledge and has a moderate performance boost on algorithmic tasks compared to CoT (Figure 4). This is further confirmed by the error analysis on MATH, where $74.7 \%$ model failures comes from computation errors (e.g. math). We also take a closer look at the self-discovered reasoning structures, and show the universality of them by transferability study from PaLM 2-L to GPT-4, and from GPT-4 to Llama-2-70B. We hope to encourage more future work on structured reasoning for solving challenging problems using LLMs.

## 2. Self-Discovering Reasoning Structures for Problem-Solving

We take inspiration from how humans use prior knowledge and skills to devise a reasoning program to solve problems (Newell et al., 1958; Rasmussen, 1983). When we face a new problem, we often first search internally what knowledge and skills from our prior experience might be helpful to solve it. Then we will attempt to apply relevant knowledge and skills to this task. And finally we will connect multiple individual skills and knowledge to solve the problem. We design SELF-DiscoVER to enact these steps into two stages as illustrated in Figure 2.

Given a task and a set of reasoning module descriptions representing high-level problem-solving heuristics such as "Use critical thinking" and "Let's think step by step", Stage 1 of SELF-DISCOVER aims to uncover the intrinsic reasoning structure for solving this task via meta-reasoning. Specifically, we uses three meta-prompts to guide LLMs to select, adapt, and implement an actionable reasoning structure with no labels or training required. We format the structure in key-value pairs similar to JSON due to interpretability and findings on following JSON boosts reasoning and generation quality (Zhou et al., 2023; OpenAI, 2023a). The structure of

Stage 1: Discover Reasoning Structure on Task-Level

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-03.jpg?height=263&width=881&top_left_y=291&top_left_x=275)

Reasoning Structure

Key-Value pairs
1 "Type and color of each item": ""

"Number of items of each color": "" "Number of items of each type": ""

"Number of items of each color and type".

"Final answer"

Stage 2: Solve Problems Using Discovered Structure on Instance-Level Fill in the Values based on Keys during decoding

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-03.jpg?height=109&width=1342&top_left_y=645&top_left_x=283)

Figure 2. Illustration of using SELF-Discover for problem-solving. Given a generative LM, task, and seed reasoning module descriptions, we guide LMs to generate a reasoning structure in key-value format to solve the task. Finally, models can follow the self-discovered structures to solve the every instance from the task by filling in the values in JSON step-by-step.

the meta-prompts and full prompts are shown in Appendix. Stage 1 operates on task-level, meaning we only need to run SELF-DISCOVER once for each task. Then, in Stage 2, we can simply use the discovered reasoning structure to solve every instance of the given task by instructing models to follow the provided structure by filling each key and arrive at a final answer.

### 2.1. Stage 1: Self-Discover Task-Specific Structures

The first stage consists of three actions: 1) SELECT, where relevant reasoning modules for task-solving are chosen from the set of reasoning module descriptions; 2) ADAPT, where descriptions of selected reasoning modules are rephrased to be more specific to the task at hand; and 3) IMPLEMENT, where the adapted reasoning descriptions are implemented into a structured actionable plan so that the task can be solved by following the structure.

SELECT First, not every reasoning module is helpful for every task, so the first stage of SELF-Discover guides model to select modules that are useful based on task examples. For example, "reflective thinking" might help search for first-principle theories on science problems, while "creative thinking" helps on generating a novel continuation to a story. Given raw set of reasoning module descriptions $D$ such as "critical thinking", and "break the problem into sub-problems" (full set in Appendix A), and a few task examples without labels $t_{i} \in T$, SELF-DISCOVER first selects a subset of reasoning modules $D_{S}$ that are useful for solving the tasks by using a model $\mathcal{M}$ and a meta-prompt $p_{S}$ :

$$
\begin{equation*}
D_{S}=\mathcal{M}\left(p_{S}\|D\| t_{i}\right) \tag{1}
\end{equation*}
$$

ADAPT Since each reasoning module provides a general description of how to solve problems, the next step of SELFDISCOVER aims at tailoring each selected module to the task at hand. For example, from "break the problem into subproblems" to "calculate each arithmetic operation in order" for arithmetic problems. Given selected reasoning module subset $D_{S}$ from the previous step, ADAPT rephrases each of the selected module to be more specific to the task. Similarly to SELECT, this stage uses a meta-prompt $p_{A}$ and a generative model $\mathcal{M}$ to generate the adapted reasoning module descriptions $D_{A}$ :

$$
\begin{equation*}
D_{A}=\mathcal{M}\left(p_{A}\left\|D_{S}\right\| t_{i}\right) \tag{2}
\end{equation*}
$$

IMPLEMENT Finally, given the adapted reasoning module descriptions $D_{A}$, SELF-DisCoVER operationalizes the reasoning modules into an implemented reasoning structure $D_{I}$ with specified instruction on what to generate for each step. In addition to a meta prompt $p_{I}$, IMPLEMENT also provides a demonstration of a human-written reasoning structure $S_{\text {human }}$ on another task to better convert the natural language descriptions into a reasoning structure:

$$
\begin{equation*}
D_{I}=\mathcal{M}\left(p_{A}\left\|S_{\text {human }}\right\| D_{A} \| t_{i}\right) \tag{3}
\end{equation*}
$$

### 2.2. Stage 2: Tackle Tasks Using Discovered Structures

After the three stages, we have an implemented reasoning structure $D_{I}$ uniquely adapted for the task we need to solve $T$. Then we can simply append the reasoning structure to all instances of the task and prompt models to follow the reasoning structure to generate an answer $A$ :

$$
\begin{equation*}
A=\mathcal{M}\left(D_{S} \| t\right), \forall t \in T \tag{4}
\end{equation*}
$$

More details of prompts are included in Appendix A.

## 3. Experiment Setup

### 3.1. Tasks

We focus on diverse reasoning benchmarks that are still challenging for LLMs: BIG-Bench Hard (BBH) (Suzgun

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-04.jpg?height=656&width=1510&top_left_y=230&top_left_x=275)

Figure 3. Illustration of three actions of SELF-DISCOVER. We use LMs to compose a coherent reasoning structure by selecting relevant modules, adapting to task-specific descriptions, and implement a reasoning structure in JSON.

et al., 2022) contains 23 carefully-selected challenging tasks from BIG-Bench (Srivastava et al., 2023). BBH tasks cover a diverse range of reasoning problems spanning the following 4 categories according to their authors: 1) Algorithmic and Multi-Step Arithmetic Reasoning, 2) Natural Language Understanding, 3) Use of World Knowledge, and 4) Multilingual Knowledge and Reasoning. We also test on a grounded social agent reasoning task called Thinking for Doing (T4D) where models must leverage mental state reasoning to determine actions to perform (Zhou et al., 2023), where GPT-4 with CoT only reaches around $50 \%$. Finally, we subsample 200 examples from the MATH (Hendrycks et al., 2021) test set, and generate instance-level reasoning structures via a one-shot demonstration to adapt to the complexity of MATH tasks. For evaluations, we use accuracy to measure the model performance on BBH, T4D and MATH (details can be found in Appendix B).

### 3.2. Models

We use several state-of-the-art LLMs: GPT-4 (gpt-4-turbopreview) (OpenAI, 2023b), GPT-3.5-turbo (ChatGPT) (OpenAI, 2022) ${ }^{1}$, instruction-tuned PaLM 2-L (Anil et al., 2023) ${ }^{2}$, and an open-source LLM Llama2-70B (Touvron et al., 2023).

### 3.3. Baselines

We compare SELF-DISCOVER with other zero-shot prompting methods for LLM reasoning:[^1]- Direct Prompting, where model directly generates the answer without intermediate reasoning steps.
- CoT (Wei et al., 2022; Kojima et al., 2022), where models are prompted to generate a reasoning process leading to the final answer.
- Plan-and-Solve (Wang et al., 2023), where models are prompted to first generate a plan and then solve the problem. SELF-DISCOVER differs by grounding the reasoning structure in atomic reasoning modules, and prompting the decoding to follow the explicit key-value reasoning structure.

Next, we also consider other baselines that make use of the raw seed reasoning modules (RM) we pass to SELFDISCOVER. We compare with the following methods' performance and the inference call efficiency on a subset of tasks.

- CoT-Self-Consistency (Wang et al., 2022), we sample multiple outputs from LLM with CoT and aggregate answers to get the final answer. We compare this method on a subset of tasks due to the cost of repetitive queries.
- Majority voting of each RM: we prompt models to solve the tasks by appending each RM and use majority voting of all answers to get the final answer. We examine whether integrating multiple RMs into a coherent reasoning structure is advantageous to applying each RM to solve the task and use majority voting to ensemble them post-hoc, which costs much more inference computation.
- Best of each RM: this method assumes that we have access to oracle labels and uses the highest accuracy from

Table 1. Self-Discover significantly improves LLM reasoning across a diverse set of 25 complex tasks: BBH, T4D and MATH. CoT: zero-shot Chain of Thought (Kojima et al., 2022). PS: planand-solve prompting (Wang et al., 2023).

| Method | BBH | T4D | MATH |
| :--- | :---: | :---: | :---: |
| PaLM 2-L | $56 \%$ | $30 \%$ | $45 \%$ |
| PaLM 2-L + CoT | $60 \%$ | $40 \%$ | $42 \%$ |
| PaLM 2-L + PS | $61 \%$ | $42 \%$ | $49 \%$ |
| PaLM 2-L + Self-Discover | $\mathbf{6 7 \%}$ | $\mathbf{6 9 \%}$ | $\mathbf{5 0 . 5 \%}$ |
| GPT-4 | $58 \%$ | $51 \%$ | $70.5 \%$ |
| GPT-4 + CoT | $75 \%$ | $52 \%$ | $71 \%$ |
| GPT-4 + PS | $73 \%$ | $53 \%$ | $70 \%$ |
| GPT-4 + Self-Discover | $\mathbf{8 1 \%}$ | $\mathbf{8 5 \%}$ | $\mathbf{7 3 \%}$ |

applying each RM. We compare with this to examine whether SELF-DISCOVER competes with methods that depend on perfect prior knowledge of which RM to use on a new task.

Furthermore, for analysis on universality of reasoning structures, we compare with a prompt-optimization method that require a training set to improve prompts: LLMs as optimizers (OPRO) (Yang et al., 2023). We aim to show that when we apply structures or prompts optimized from one model, the reasoning structures can retain more performance gains than the wordings of prompts.

## 4. Results

We answer the following questions through experimental results: 1) Does discovering reasoning structures improve LLM reasoning capabilities? (4.1) 2) Which categories of problems do SELF-DISCOVER perform the best? (4.2) and 3) Can SELF-Discover boost LLM performance efficiently? (4.3) Finally, we will show qualitative examples of self-discovered structures, LLM output following the structures, and compare with LLM output following other prompting methods for reasoning (4.4).

### 4.1. Does SELF-Discover Improve LLM Reasoning?

Overall, SELF-DISCOVER improves PaLM 2-L and GPT4's reasoning across diverse set of reasoning tasks. Table 1 shows the overall results on complex reasoning tasks of BBH, T4D and MATH using PaLM 2-L and GPT-4. We compare Self-Discover with baselines including direct prompting, CoT, and Plan-and-Solve (PS).

On aggregated 23 tasks of BBH, SELF-DISCOVER achieves $7 \%$ and 6\% absolute improvement on PaLM 2-L over Chainof-Thought and Plan-and-Solve, respectively. Similar gains ( $6 \%$ and $8 \%$ ) are observed when SELF-DISCOVER is applied to GPT-4. Breakdown results of each task's improvement

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-05.jpg?height=471&width=761&top_left_y=236&top_left_x=1083)

Figure 4. Breakdown of SElf-Discover performance improvement on 4 categories on PaLM 2-L. SELF-Discover performs the best on tasks requiring world knowledge.

over direct answering and CoT of PaLM 2-L are shown in Figure 1, where we find SELF-DISCOVER outperforms them on over 20/24 tasks. For a per-task performance for all 23 BBH tasks, please refer to Appendix C.

On the grounded social agent task T4D, SElFDISCOVER reaches over $\geq 27 \%$ (32\%) absolute improvement over all baselines on PaLM 2-L (GPT-4). SELF-DISCOVER achieves $69 \%$ and $85 \%$ accuracy on PaLM 2-L and GPT-4, significantly outperforming previous SoTA prompting method such as Foresee and Reflect (FaR) which employs an expert-designed reasoning structure. In contrast, Self-Discover generates the reasoning structure automatically from a set of atomic reasoning modules without human interventions.

For MATH, we observe a moderate gain of $1 \%-7 \%(2 \%-3 \%)$ on PaLM 2-L (GPT-4) from SELF-DISCOVER compared to the baselines. Upon error analysis (see Appendix D for details), we find that the reasoning structures generated by PaLM 2-L from SELF-DISCOVER are correct $87.5 \%$ of the time: human experts can follow the reasoning structures to solve the tasks perfectly. The majority of the failures ( $74.7 \%$ ) comes from errors in executing the computations, consistent with prior findings (Zheng et al., 2023).

### 4.2. Which Types of Problems Do Self-Discover Help the Most?

SELF-DISCOVER performs best on tasks that require diverse world knowledge. Figure 4 presents the average improvement in terms of delta in accuracy of SELFDISCOVER over direct answer and CoT on 4 categories of reasoning tasks we test. We adopt the categorization from Suzgun et al. (2022). We find that SelfDISCOVER improves over these two baselines on all categories, but especially on tasks that require world knowledge such as sports understanding, movie recommendation, and ruin names.

These tasks demand models to reason using fact and general

BBH-Movie Recommendation

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-06.jpg?height=534&width=580&top_left_y=232&top_left_x=274)

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-06.jpg?height=582&width=1529&top_left_y=197&top_left_x=257)

BBH-Geometric Shapes

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-06.jpg?height=548&width=531&top_left_y=222&top_left_x=884)

$\star$ Self-Discover

Direct

CoT

CoT+Self-Consistency

Plan-and-Solve

X Majority voting each RM

Best of each RM*

Figure 5. Comparison of accuracy with number of inference calls required per instance. For CoT-Self-Consistency, we sample 10 times. Best of each RM method requires gold labels (*). SELF-DisCover requires only 1 inference call per instance (plus 3 more meta-prompts on the task-level), same as Direct and CoT while reaching better performance compared with $40 \mathrm{x}$ more call required methods (majority voting of each RM) on GPT-4. We acknowledge that SELF-DISCOVER input and output are longer than CoT and Direct prompting, increasing cost. However, as the number of instances increases, the efficiency of SELF-DISCOVER in terms of inference per instance is highly desirable.

commonsense knowledge. We interpret SELF-DISCOVER's advantages on these tasks as strength from integrating multiple reasoning modules from various perspectives as only applying CoT might miss key knowledge in the reasoning process. We observe that the gain on the Algorithmic category is moderate, consistent with the findings from Sec. 4.1 on MATH.

### 4.3. How Efficient is SELF-DISCOVER?

SELF-DISCOVER achieves better performance while requiring 10-40x fewer inference computer compared to self-consistency or majority voting. Here we examine a subset of 2 tasks from BBH and present a more thorough comparison of methods including those requiring many inference calls that are too costly to run on all 24 tasks. Figure 5 shows average accuracy and number of inference calls required per instance for each method using GPT-4. Accuracy wise (y-axis), we find that SELFDISCOVER outperforms other baselines even those that require repeated inference calls such as CoT-self-consistency and majority voting of applying each RM. Efficiency wise (x-axis), SELF-DISCOVER only requires one call per instance and three more inference calls on the task-level, CoTself-consistency requires 10 times more since we have to sample 10 times for each instance, and methods using each RM requires 40 times more as we use 40 RMs. In summary, SELF-DISCOVER presents itself a strong reasoning boosting method that is efficient to deploy on large-scale.

### 4.4. Qualitative Examples

We show examples of model-discovered structures for different reasoning tasks in Figure 6 from PaLM 2-L. We observe that each structure is uniquely adapted to the task, integrates

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-06.jpg?height=539&width=770&top_left_y=1034&top_left_x=1079)

Figure 6. Examples of self-discovered structures on BBH tasks using PaLM 2-L. We observe traits of atomic reasoning modules such as "Step-by-step thinking", "reflect on task nature", and an interesting creative thinking case where models devise an algorithm using stack to solve parenthesis parsing task.

multiple reasoning modules, and provides insights on how to solve the tasks. Furthermore, example of comparing reasoning processes from CoT, Plan-and-Solve, and SElFDisCOVER is shown in Figure 7. We find that CoT and Plan-and-Solve makes incorrect assertions early and arrives at a wrong answer while following structure from SELFDISCOVER leads the model to generate logical conclusions ("path is closed as the beginning and ending coordinates are the same") and arrive at the correct answer.

## 5. Deep Diving Into Self-Discovered Reasoning Structures

After experimental results showing the effectiveness and efficiency of SELF-DISCOVER on a range of reasoning

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-07.jpg?height=908&width=1632&top_left_y=234&top_left_x=211)

Figure 7. Comparison of generated reasoning process from CoT, Plan-and-Solve, and SELF-DisCOVER on BBH-geometric shape task. Both CoT and Plan-and-Solve incorrectly asserts that the path does not form a regular shape as it is not a closed path (highlighted in red) and arrive at a wrong answer. The reasoning structure (in blue Courier font) from SELF-DISCOVER first breaks down each line segment and analyze the coordinates carefully, then leverages logical reasoning to conclude that it forms a closed shape as the path ends at the same coordinate (highlighted in purple and orange), and selects the correct answer through final reasoning.

tasks, this section further analyzes are all actions of SELFDISCOVER needed and what other benefits can selfdiscovered structures bring? In Sec. 5.1 , we show that it is critical to the model's performance to use the reasoning structures discovered through the three steps of SELECT, ADAPT and IMPLEMENT. In Sec. 5.2, we demonstrate the universality of the self-discovered reasoning structures by (1) applying the structures discovered by PaLM 2-L to GPT-4, (2) applying the structures discovered by GPT-4 to Llama-2-70B. We further show the commonalities between the reasoning structures and human reasoning patterns in Appendix E.

### 5.1. Importance of SELF-DISCOVER Actions

We conduct ablation study on the three actions: SELECT, ADAPT, and IMPLEMENT to analyze the effects of SELFDISCOVER actions. Figure 8 show results using GPT-4 on 4 reasoning tasks when we apply SELECT (-S) or apply SELECT and ADAPT (-SA) or apply all three actions. We find that with each stage, model's zero-shot reasoning capability improve consistently across tasks, indicating that all three actions are beneficial. In particular, after all three actions SAI, the reasoning structures are adapted to be task specific, and bring the most gain to solving the reasoning tasks.

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-07.jpg?height=380&width=783&top_left_y=1385&top_left_x=1083)

Figure 8. Ablation study on three SELF-DISCOVER actions on 4 reasoning tasks: all three actions are beneficial for task-solving.

### 5.2. Towards Universality of Discovered Reasoning Structures

Applying PaLM 2-L Discovered Structures to GPT-4 We first use a PaLM 2-L model to discover the reasoning structures of 4 reasoning tasks. Then, we apply the resulting reasoning structures to the decoding of GPT-4 as grounding. We compare our approach to OPRO (Yang et al., 2023) which discovered zero-shot-prompts through optimizations. We apply OPRO prompts optimized using PaLM 2-L on each task to GPT-4 on the same reasoning tasks. Figure 9 shows that SELF-DISCOVER outperforms OPRO on 3 out of 4 tasks despite that OPRO used $20 \%$ data to optimize the

![](https://cdn.mathpix.com/cropped/2024_06_04_9f5d16f781ade6059c66g-08.jpg?height=475&width=767&top_left_y=234&top_left_x=213)

Figure 9. Transferrability tests of optimized prompts (OPRO) and composed structures (SELF-DISCOVER). The results shown are from GPT-4 using the prompts and structures optimized or composed using PaLM 2-L. We find that self-discovered reasoning structure transfers more robustly than optimized prompts. prompt. In contrast, SELF-DISCOVER is done in a zero-shot manner, demonstrating the efficiency of our method and universality of the discovered reasoning structures.

## Applying GPT-4 Discovered Structures to Llama2 and

 ChatGPT Motivated by transferrability performance across LLMs, we further investigate can self-discovered reasoning structures from LLMs boost reasoning for smaller $L M s$ that are challenging to come up with structures themselves ${ }^{3}$. We use GPT-4 to discover the task-intrinsic reasoning structures, and then apply those structures to the decoding of open-sourced Llama2-70B as well as GPT-3.5turbo (ChatGPT) on two subsets of tasks from BBH. We find that using self-discovered structures on Llama2 (52\%) outperforms CoT (42\%) on disambiguation QA zero-shot and on GPT-3.5-turbo (56\%) outperforms CoT (51\%) on geometry with 3 -shot demonstration from structured reasoning process.
## 6. Related Work

### 6.1. Prompting Methods

Recent advancements in the area of LLMs have given rise to a plethora of few-shot (Brown et al., 2020) and instruction (Mishra et al., 2022c; Wei et al., 2021; Ouyang et al., 2022) prompting techniques, including Chain-of-Thought prompting (CoT) (Nye et al., 2021; Wei et al., 2022), Leastto-most prompting (Zhou et al., 2022a; Drozdov et al., 2022), Decomposed prompting (Khot et al., 2022), Reframing (Mishra et al., 2022b), Help Me Think Prompting (Mishra \& Nouri, 2023), Stepback Prompting (Zheng et al., 2023) and search-based approaches like Tree-ofThought (ToT) (Yao et al., 2023a), Graph-of-Thought (Besta et al., 2023; Yao et al., 2023b), Branch-solve-merge (Saha et al., 2023) and RAP (Hao et al., 2023). Each of the[^2]

prompting methods has some strengths and weaknesses in terms of their successful application domain. Our work SELF-DISCOVER presents the missing piece in the prompting literature, as SELF-DISCOVER provides a way to selfcompose over various prompting methods via the proposed self-discovery mechanism. Composing over prompting methods in SELF-DISCOVER is analogous to the programming literature where a program is written using various basic building blocks such as for loop, if/else condition etc.

### 6.2. Reasoning and Planning

With the development of various reasoning and planning benchmarks such as GSM8K (Cobbe et al., 2021), Math (Hendrycks et al.), BigBench (Srivastava et al., 2023) etc., various methods have been proposed to improve model performance. Often these methods induce specific reasoning structures mimicking the reasoning structure of the underlying task associated with the dataset. For example, chain of thought (Wei et al., 2022) and scratchpad (Nye et al., 2021) induce generation of explanations associated with a reasoning question. Similarly other methods induces specific reasoning structures such as question summarization (Kuznia et al., 2022), question decomposition (Patel et al., 2022), program generation (Mishra et al., 2022a; Chen et al., 2022; Gao et al., 2023b), etc. However, in a real world user traffic, queries can be diverse covering various reasoning structures. Our work SELF-Discover allows models to combine multiple reasoning approaches by selfcomposing into a structure without the need to access task labels. There have been some related work that explores LLM combining skills in-context such as SkiC (Chen et al., 2023), devising a strategy (Gao et al., 2023a), and planning with iterative quering (Liu et al., 2023). However, they require human annotating skills and reasoning plans while SELF-DISCOVER leverages a scalable solution with the help of LLM's meta-task reasoning capabilities.

## 7. Conclusion

We introduce SELF-DISCOVER, an efficient and performant framework for models to self-discover a reasoning structure for any task from a seed set of general problem-solving skills. We observe drastic improvements on challenging reasoning benchmarks from multiple LLMs up to $30 \%$. Ablations study of SELF-DISCOVER demonstrates that the composed reasoning structures are universally transferable between LLMs. Forward looking, we are excited to explore more on LLM structured reasoning to push the boundary of problem-solving and discover potentials for Human-AI collaboration.

## Acknowledgement

We thank Andrew Dai and Adams Yu of Google DeepMind for their insightful feedback on this paper.

## References

Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877-1901, 2020.

Chen, J., Pan, X., Yu, D., Song, K., Wang, X., Yu, D., and Chen, J. Skills-in-context prompting: Unlocking compositionality in large language models. arXiv preprint arXiv:2308.00304, 2023.

Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Drozdov, A., Schärli, N., Akyürek, E., Scales, N., Song, X., Chen, X., Bousquet, O., and Zhou, D. Compositional semantic parsing with large language models. arXiv preprint arXiv:2209.15003, 2022.

Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktäschel, T. Promptbreeder: Self-referential self-improvement via prompt evolution. arXiv preprint arXiv:2309.16797, 2023.
Gao, C., Jiang, H., Cai, D., Shi, S., and Lam, W. Strategyllm: Large language models as strategy generators, executors, optimizers, and evaluators for problem solving. arXiv preprint arXiv:2311.08803, 2023a.

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764-10799. PMLR, 2023b.

Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., and $\mathrm{Hu}, \mathrm{Z}$. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023 .

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. Sort, 2(4):0-6.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset, 2021.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., and Sabharwal, A. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh International Conference on Learning Representations, 2022.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35 : 22199-22213, 2022.

Kuznia, K., Mishra, S., Parmar, M., and Baral, C. Less is more: Summary of long instructions is better for program synthesis. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. $4532-4552,2022$.

Liu, T., Guo, Q., Yang, Y., Hu, X., Zhang, Y., Qiu, X., and Zhang, Z. Plan, verify and switch: Integrated reasoning with diverse x-of-thoughts. arXiv preprint arXiv:2310.14628, 2023.

Mishra, S. and Nouri, E. HELP ME THINK: A simple prompting strategy for non-experts to create customized content with models. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 11834-11890, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl. 751. URL https://aclanthology.org/2023. findings-acl.751.

Mishra, S., Finlayson, M., Lu, P., Tang, L., Welleck, S., Baral, C., Rajpurohit, T., Tafjord, O., Sabharwal, A., Clark, P., et al. Lila: A unified benchmark for mathematical reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. $5807-5832,2022 \mathrm{a}$.

Mishra, S., Khashabi, D., Baral, C., Choi, Y., and Hajishirzi, H. Reframing instructional prompts to gptk's language. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 589-612, 2022b.

Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Crosstask generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3470-3487, 2022c.

Newell, A., Shaw, J. C., and Simon, H. A. Elements of a theory of human problem solving. Psychological review, $65(3): 151,1958$.

Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021.

OpenAI. Chatgpt: Optimizing language models for dialogue, 2022. URL https://openai.com/blog/ chatgpt/.

OpenAI. Json generation mode, 2023a. URL https://platform.openai.com/docs/ guides/text-generation/json-mode.

OpenAI, R. Gpt-4 technical report. arXiv, pp. 2303-08774, 2023b.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Patel, P., Mishra, S., Parmar, M., and Baral, C. Is a question decomposition unit all we need? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 4553-4569, 2022.

Polya, G. How to solve it: A new aspect of mathematical method, volume 85. Princeton university press, 2004.

Rasmussen, J. Skills, rules, and knowledge; signals, signs, and symbols, and other distinctions in human performance models. IEEE transactions on systems, man, and cybernetics, (3):257-266, 1983.
Saha, S., Levy, O., Celikyilmaz, A., Bansal, M., Weston, J., and Li, X. Branch-solve-merge improves large language model evaluation and generation. arXiv preprint arXiv:2310.15123, 2023.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023.

Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper. pdf.

Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., and Lim, E.-P. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. arXiv preprint arXiv:2305.04091, 2023.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022.

Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2021.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824-24837, 2022.

Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. Large language models as optimizers. arXiv preprint arXiv:2309.03409, 2023.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023a.

Yao, Y., Li, Z., and Zhao, H. Beyond chain-of-thought, effective graph-of-thought reasoning in large language models. arXiv preprint arXiv:2305.16582, 2023b.

Yasunaga, M., Chen, X., Li, Y., Pasupat, P., Leskovec, J., Liang, P., Chi, E. H., and Zhou, D. Large language models as analogical reasoners. arXiv preprint arXiv:2310.01714, 2023.

Zheng, H. S., Mishra, S., Chen, X., Cheng, H.-T., Chi, E. H., Le, Q. V., and Zhou, D. Take a step back: Evoking reasoning via abstraction in large language models. arXiv preprint arXiv:2310.06117, 2023.

Zhong, R., Lee, K., Zhang, Z., and Klein, D. Adapting language models for zero-shot learning by metatuning on dataset and prompt collections. arXiv preprint arXiv:2104.04670, 2021.

Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q. V., et al. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2022a.

Zhou, P., Madaan, A., Potharaju, S. P., Gupta, A., McKee, K. R., Holtzman, A., Pujara, J., Ren, X., Mishra, S., Nematzadeh, A., et al. How far are large language models from agents with theory-of-mind? arXiv preprint arXiv:2310.03051, 2023.

Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., and Ba, J. Large language models are humanlevel prompt engineers. In The Eleventh International Conference on Learning Representations, 2022b.
</end of paper 0>


<paper 1>
# Enhancing ICU Patient Recovery: Using LLMs to Assist Nurses in Diary Writing 

S. KERNAN FREIRE, Delft University of Technology, The Netherlands<br>M.M.C. VAN MOL, Erasmuc MC, The Netherlands<br>C. SCHOL, Erasmuc MC, The Netherlands<br>E. OZCAN VIEIRA, Delft University of Technology, The Netherlands


#### Abstract

Intensive care unit (ICU) patients often develop new health-related problems in their long-term recovery. Health care professionals keeping a diary of a patient's stay is a proven strategy to tackle this but faces several adoption barriers, such as lack of time and difficulty in knowing what to write. Large language models (LLMs), with their ability to generate human-like text and adaptability, could solve these challenges. However, realizing this vision involves addressing several socio-technical and practical research challenges. This paper discusses these challenges and proposes future research directions to utilize the potential of LLMs in ICU diary writing, ultimately improving the long-term recovery outcomes for ICU patients.


## 1 INTRODUCTION AND BACKGROUND

Advanced treatments and sophisticated technological interventions in critical care medicine have significantly increased the survival rates of patients in the intensive care unit (ICU). Despite this progress, patients often face various health-related challenges in their long-term recovery $[9,10]$. More than half of patients develop new physical, psychological, and/or cognitive problems following their ICU admission [7], collectively referred to as Post Intensive Care Syndrome (PICS) $[3,25]$.

Family members also experience a stressful period, potentially leading to psychological problems addressed as PICSFamily (PICS-F) [2]. Patient and family-centered care (PFCC) at the ICU, including emotional support and follow-up service, could mitigate the symptoms associated with both PICS and PICS-F. In this study, we explored how an emerging technology, i.e., large language models, could support the emotional well-being of people exposed to critical care.

### 1.1 Emotional recovery following ICU admission, the use of diaries

ICU diaries can be used as part of the PFCC approach, focusing on decreasing symptoms of PICS/PICS-F. These diaries are written in everyday language by healthcare professionals, mostly nurses and family members. Using ICU diaries offers many benefits to all stakeholders involved in the ICU. They contain daily entries detailing the current patient status and descriptions of situations and surroundings. Reading a diary after hospitalization is an effective way of coping with the traumatic aftermath of critical illness, consequently helping to prevent the development of psychological problems [1, 17]. For family members, the diary provides an opportunity to actively care for their loved ones, thus diminishing feelings of powerlessness and reducing psychological problems [19, 22]. Digital diaries were developed during the COVID-19 pandemic, receiving positive assessments from family members [26]. However, implementing digital diaries face some barriers $[6,8]$ as will be explained further.[^0]

### 1.2 Digital diary in the ICU; Barriers in implementation

High-quality PFCC should be considered a fundamental skill for ICU healthcare professionals [12]. However, barriers have been identified, hindering the effective implementation of digital ICU diaries. The main barrier is a lack of time for healthcare professionals [20]. Another barrier is the lack of knowledge among nurses regarding what and how to write in the diary [8]. A potential solution to address these challenges could be the integration of Artificial Intelligence (AI) in the writing process of healthcare professionals.

### 1.3 Large Language Models in Healthcare

Natural language processing (NLP) is a machine learning technique that involves the processing and analysis of text. Large language models (LLMs), such as ChatGPT, are very effective at generating human-like text. LLMs mark a significant step forward from their predecessors, such as recurrent neural networks (RNNs). Unlike RNNs that process text sequentially and often struggle with long-range dependencies, LLMs can analyze and generate text in parallel, handling extensive context and complex language patterns efficiently [11, 18, 27, 28]. These characteristics make LLMs effective writing partners, helping humans by generating outlines, refining text, and adapting it to the reader.

In healthcare, applications of NLP range from supporting triage by analyzing prior medical notes to answering patients' questions as a chatbot [15]. Recently, LLMs have received widespread attention in healthcare, leading to the creation of health-specific models, such as Med-Palm [24]. Applications include supporting clinical workflow, for example, by generating discharge notes, extracting ecological phenotypes [5], and making medical texts more understandable and empathetic for patients [14]. These capabilities could help tackle challenges nurses face when using digital diaries in the ICU, however, this remains unexplored in the literature [14, 21].

## 2 TOWARDS ICU DIARIES SUPPORTED BY LARGE LANGUAGE MODELS

ICU diaries can provide a more personable timeline for the ICU admission beyond the medical notes that medical professionals already record. Our vision is that an LLM-powered tool can support the writing process for nurses, making it more efficient without losing the personal touch of a human writer. In the following sections, we describe this vision in more detail and the associated research challenges.

### 2.1 Future Vision

We envision a collaborative writing process that evolves as nurses become more familiar with the LLM-powered tool's capabilities, and in turn, the tool "learns" the nurse's writing style. To begin with, nurses may be unfamiliar with the diary writing process for ICU patients. As such, the tool can help nurses figure out what and how to write. At this stage, the tool asks for key information about the situation and generates an example diary entry for the nurse. As the nurse becomes more familiar with the process and expectations, they can start adjusting the entries themselves or write them from scratch. At this stage, the tool can provide in-text suggestions on how to write empathetically and understandably for the patients. Over time, the tool will amass a database of entries about individual patients written by the nurse, allowing the tool to align with their writing style [18]. In turn, the nurse can enter a few keywords to generate a diary entry, saving time. Thus, this collaborative process allows for growth and adaptation both on the human and technological sides.

The tool must support various diary entry themes and modalities, primarily text and images. Prior work by Galazzi et al. [6] has shown that ICU diary entries fall under four main themes with ten sub-themes, namely, Presenting

(Places and people; Diary project), Intensive Care Unit Stay (Clinical events; What the patient does; Patient support), Outside the Hospital (Family and topical events; The weather), Feelings and Thoughts (Encouragement and wishes; Farewell; Considerations). While information about patients' specific ICU experience can only be filled by attending nurses and families, non-patient-related topics, such as the weather and national events, are publicly available via application programming interfaces (APIs). When relevant, the tool could use APIs to enrich the diary entries. Similarly, the tool could access recent entries to the medical documentation records or visitor calendar; however, this poses several technical and ethical challenges as the implications of using LLMs for personal information exchange are not fully understood.

### 2.2 Research Challenges

To realize the vision described above, several socio-technical research challenges must be tackled. These research challenges are discussed under four predefined themes, inspired by prior work that evaluated novel digital tools in support of nurse training [16].

Space and place To better understand the tool's impact on the existing ICU environment, research could empirically evaluate how an LLM-based diary task reduces nurses' daily workload and how it allows for more humanized care.

Technology The technological research challenges revolve around optimizing the LLMs for the tool. Important considerations include using specialized LLMs for different aspects, such as suggestion topics, making entries more empathetic, generating entries based on keywords, and querying APIs. Indeed, the desired tool behavior could be realized by combining several techniques, including fine-tuning models [4], reasoning strategies (e.g., self-discovery [29]), and/or retrieval augmented generation [13]. Several technical limitations may also be imposed due to the ethical concerns of storing and sharing the patients' personal data, for example, the necessity to host LLMs locally in hospitals or at secure providers. Furthermore, it is imperative to identify the risks associated with model hallucinations and bias and how these can be mitigated [21].

Design The tool's design will play a key role in its usability and adoption. Therefore, research should explore options for interaction modalities, how the tool's outputs are displayed (e.g., in-text suggestions), and integrated into the nurses' workflows.

Social factors As with the introduction of any new tool, the end users will likely have several concerns and expectations that should be addressed as early as possible. Therefore, it is important to conduct preliminary research, such as semi-structured interviews and technology probes, to elicit any concerns and requirements from the nurses, patients, and other stakeholders. This will make the tool more socially responsible and human-centered [23].

## 3 CONCLUSION

Large language models have become increasingly popular in the past few years, especially in non-critical contexts. In this study, we explored the potential of LLMs in supporting ICU nurses in writing diary entries for critically ill patients. While this novel technology has much to offer to humanize a highly technological environment, embedding it in nurses' routines seamlessly, LLM-powered tools must be explored thoroughly to understand the socio-technical limitations, opportunities, and risks.

## ACKNOWLEDGMENTS

We acknowledge Kaixin Tang for her graduation project, which inspired this work.

## REFERENCES

[1] Bruna Brandao Barreto, Mariana Luz, Marcos Nogueira de Oliveira Rios, Antonio Alberto Lopes, and Dimitri Gusmao-Flores. 2019. The impact of intensive care unit diaries on patients' and relatives' outcomes: a systematic review and meta-analysis. Critical Care 23, 1 (Dec. 2019 ), 411. https://doi.org/10.1186/s13054-019-2678-0

[2] Jill I. Cameron, Leslie M. Chu, Andrea Matte, George Tomlinson, Linda Chan, Claire Thomas, Jan O. Friedrich, Sangeeta Mehta, Francois Lamontagne, Melanie Levasseur, Niall D. Ferguson, Neill K.J. Adhikari, Jill C. Rudkowski, Hilary Meggison, Yoanna Skrobik, John Flannery, Mark Bayley, Jane Batt, Claudia dos Santos, Susan E. Abbey, Adrienne Tan, Vincent Lo, Sunita Mathur, Matteo Parotto, Denise Morris, Linda Flockhart, Eddy Fan, Christie M. Lee, M. Elizabeth Wilcox, Najib Ayas, Karen Choong, Robert Fowler, Damon C. Scales, Tasnim Sinuff, Brian H. Cuthbertson, Louise Rose, Priscila Robles, Stacey Burns, Marcelo Cypel, Lianne Singer, Cecilia Chaparro, Chung-Wai Chow, Shaf Keshavjee, Laurent Brochard, Paul Hébert, Arthur S. Slutsky, John C. Marshall, Deborah Cook, and Margaret S. Herridge. 2016. One-Year Outcomes in Caregivers of Critically Ill Patients. New England fournal of Medicine 374, 19 (2016), 1831-1841. https://doi.org/10.1056/NEJMoa1511160 arXiv:https://doi.org/10.1056/NEJMoa1511160 PMID: 27168433 .

[3] Judy E. Davidson, Maurene A. Harvey, Jessica Schuller, and Gary Black. 2013. Post-intensive care syndrome: What it is and how to help prevent it. American Nurse Today 8, 5 (May 2013), 32-37. https://go.gale.com/ps/i.do?p=AONE\&sw=w\&issn=19305583\&v=2.1\&it=r\&id=GALE\|A335410359\&sid=googleScholar\&linkaccess=abs Publisher: Healthcom Media.

[4] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun. [n. d.]. Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models. 5, 3 ([n. d.]), 220-235. Issue 3. https://doi.org/10.1038/s42256-023-00626-4

[5] Matthias A. Fink, Arved Bischoff, Christoph A. Fink, Martin Moll, Jonas Kroschke, Luca Dulz, Claus Peter Heußel, Hans-Ulrich Kauczor, and Tim F. Weber. 2023. Potential of ChatGPT and GPT-4 for Data Mining of Free-Text CT Reports on Lung Cancer. Radiology 308, 3 (Sept. 2023), e231362. https://doi.org/10.1148/radiol.231362 Publisher: Radiological Society of North America.

[6] Alessandro Galazzi, Martina Bruno, Filippo Binda, Giorgia Caddeo, Monica Chierichetti, Paola Roselli, Giacomo Grasselli, and Dario Laquintana. 2023. Thematic analysis of intensive care unit diaries kept by staff: insights for caring. Intensive and Critical Care Nursing 76 (June 2023), 103392. https://doi.org/10.1016/j.icen.2023.103392

[7] Wytske W. Geense, Marieke Zegers, Marco A. A. Peters, Esther Ewalds, Koen S. Simons, Hester Vermeulen, Johannes G. van der Hoeven, and Mark van den Boogaard. 2021. New Physical, Mental, and Cognitive Problems 1 Year after ICU Admission: A Prospective Multicenter Study. American Journal of Respiratory and Critical Care Medicine 203, 12 (June 2021), 1512-1521. https://doi.org/10.1164/rccm.202009-3381OC Publisher: American Thoracic Society - AJRCCM.

[8] Tineke Haakma, Rob Tieben, Brenda Sleven, Marc Buise, and Margo van Mol. 2022. Experiences of nurses with an innovative digital diary intervention in the intensive care unit: A qualitative exploration. Intensive and Critical Care Nursing 70 (June 2022), 103197. https://doi.org/10.1016/j.icen. 2022.103197

[9] Margaret S. Herridge and Élie Azoulay. 2023. Outcomes after Critical Illness. New England Journal of Medicine 388, 10 (March 2023), 913-924. https://doi.org/10.1056/NEJMra2104669 Publisher: Massachusetts Medical Society _eprint: https://doi.org/10.1056/NEJMra2104669.

[10] Shigeaki Inoue, Junji Hatakeyama, Yutaka Kondo, Toru Hifumi, Hideaki Sakuramoto, Tatsuya Kawasaki, Shunsuke Taito, Kensuke Nakamura, Takeshi Unoki, Yusuke Kawai, Yuji Kenmotsu, Masafumi Saito, Kazuma Yamakawa, and Osamu Nishida. 2019. Post-intensive care syndrome: its pathophysiology, prevention, and future directions. Acute Medicine \& Surgery 6, 3 (2019), 233-246. https://doi.org/10.1002/ams2.415 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/ams2.415.

[11] Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. 2019. What Does BERT Learn about the Structure of Language?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Florence, Italy, 3651-3657. https://doi.org/10.18653/v1/P19-1356

[12] Jiyeon Kang. 2023. Being devastated by critical illness journey in the family: A grounded theory approach of post-intensive care syndrome-family. Intensive and Critical Care Nursing 78 (Oct. 2023), 103448. https://doi.org/10.1016/j.iccn.2023.103448

[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS'20). Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.

[14] Jianning Li, Amin Dada, Behrus Puladi, Jens Kleesiek, and Jan Egger. 2024. ChatGPT in healthcare: A taxonomy and systematic review. Computer Methods and Programs in Biomedicine (Jan. 2024), 108013. https://doi.org/10.1016/j.cmpb.2024.108013

[15] Saskia Locke, Anthony Bashall, Sarah Al-Adely, John Moore, Anthony Wilson, and Gareth B. Kitchen. 2021. Natural language processing in medicine: A review. Trends in Anaesthesia and Critical Care 38 (June 2021), 4-9. https://doi.org/10.1016/j.tacc.2021.02.007

[16] Roberto Martinez-Maldonado, Vanessa Echeverria, Gloria Fernandez-Nieto, Lixiang Yan, Linxuan Zhao, Riordan Alfredo, Xinyu Li, Samantha Dix, Hollie Jaggard, Rosie Wotherspoon, Abra Osborne, Simon Buckingham Shum, and Dragan Gašević. 2023. Lessons Learnt from a Multimodal Learning Analytics Deployment In-the-Wild. 31, 1, Article 8 (nov 2023), 41 pages. https://doi.org/10.1145/3622784

[17] Philippa A. McIlroy, Rebecca S. King, Maité Garrouste-Orgeas, Alexis Tabah, and Mahesh Ramanan. 2019. The Effect of ICU Diaries on Psychological Outcomes and Quality of Life of Survivors of Critical Illness and Their Relatives: A Systematic Review and Meta-Analysis. Critical Care Medicine 47, 2 (Feb. 2019), 273. https://doi.org/10.1097/CCM.0000000000003547

[18] Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, and Dan Roth. 2023. Recent Advances in Natural Language Processing via Large Pre-trained Language Models: A Survey. ACM Comput. Surv. 56, 2, Article 30 (sep 2023), 40 pages. https://doi.org/10.1145/3605943

[19] Anne H Nielsen and Sanne Angel. 2016. How diaries written for critically ill influence the relatives: a systematic review of the literature. Nursing in Critical Care 21, 2 (2016), 88-96. https://doi.org/10.1111/nicc.12158 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/nicc.12158.

[20] Peter Nydahl, Carl G Bäckman, Johannes Bereuther, and Michael Thelen. 2014. How much time do nurses need to write an ICU diary? Nursing in Critical Care 19, 5 (2014), 222-227. https://doi.org/10.1111/nicc. 12046 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/nicc.12046.

[21] Malik Sallam. 2023. ChatGPT Utility in Healthcare Education, Research, and Practice: Systematic Review on the Promising Perspectives and Valid Concerns. Healthcare 11, 6 (Jan. 2023), 887. https://doi.org/10.3390/healthcare11060887 Number: 6 Publisher: Multidisciplinary Digital Publishing Institute.

[22] Rachel Schofield, Bridget Dibb, Rebecca Coles-Gale, and Christina J Jones. 2021. The experience of relatives using intensive care diaries: A systematic review and qualitative synthesis. International Journal of Nursing Studies 119 (July 2021), 103927. https://doi.org/10.1016/j.ijnurstu.2021.103927

[23] Ben Shneiderman. 2022. Human-centered AI. Oxford University Press.

[24] Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip Mansfield, Dina Demner-Fushman, Blaise Agüera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. 2023. Large language models encode clinical knowledge. Nature 620, 7972 (Aug. 2023), 172-180. https://doi.org/10.1038/s41586-023-06291-2 Number: 7972 Publisher: Nature Publishing Group.

[25] Helle Svenningsen, Leanne Langhorn, Anne Sophie Ågård, and Pia Dreyer. 2017. Post-ICU symptoms, consequences, and follow-up: an integrative review. Nursing in Critical Care 22, 4 (2017), 212-220. https://doi.org/10.1111/nicc.12165 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/nicc.12165.

[26] Margo M. C. van Mol, Nanda Tummers, Crista Leerentveld, Rob Tieben, and Marc Buise. 2023. The usability of a digital diary from the perspectives of intensive care patients' relatives: A pilot study. Nursing in Critical Care n/a, n/a (2023). https://doi.org/10.1111/nicc.12990 _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/nicc.12990.

[27] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent Abilities of Large Language Models. arXiv:2206.07682 [cs.CL]

[28] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A Survey of Large Language Models. arXiv:2303.18223 [cs.CL]

[29] Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou, Swaroop Mishra, and Huaixiu Steven Zheng. [n. d.]. Self-Discover: Large Language Models Self-Compose Reasoning Structures. https://doi.org/10.48550/arXiv.2402.03620 arXiv:2402.03620 [cs]

This figure "acm-jdslogo.png" is available in "png" format from: http://arxiv.org/ps/2402.15205v1


[^0]:    Authors' addresses: S. Kernan Freire, Delft University of Technology, Landbergstraat 15, Delft, 2628 CE, The Netherlands, s.kernanfreire@tudelft.nl; M.M.C. van Mol, Erasmuc MC, Dr. Molewaterplein 40, Rotterdam, 3015 GD, The Netherlands, m.vanmol@erasmusmc.nl; C. Schol, Erasmuc MC, Dr. Molewaterplein 40, Rotterdam, 3015 GD, The Netherlands, c.schol@erasmusmc.nl; E. Ozcan Vieira, Delft University of Technology, Landbergstraat 15, Delft, 2628 CE, The Netherlands, e.ozcan@tudelft.nl.

</end of paper 1>


<paper 2>
# How FaR ARE Large LanguAGE MoDELS FROM AGENTS WITH THEORY-OF-MIND? 

Pei Zhou ${ }^{\diamond *}$ Aman Madaan ${ }^{\wedge} \quad$ Srividya Pranavi Potharaju ${ }^{\dagger}$ Aditya Gupta<br>Kevin R. McKee ${ }^{\ddagger}$ Ari Holtzman ${ }^{\&}$ Jay Pujara ${ }^{\diamond}$ Xiang Ren $\diamond$<br>Swaroop Mishra $^{\ddagger} \quad$ Aida Nematzadeh ${ }^{\ddagger} \quad$ Shyam Upadhyay $^{\dagger} \quad$ Manaal Faruqui ${ }^{\dagger}$<br>$\dagger$ Google $\ddagger$ Google DeepMind $\diamond$ University of Southern California<br>ه Carnegie Mellon University \&niversity of Chicago<br>peiz@usc.edu

![](https://cdn.mathpix.com/cropped/2024_06_04_e5bcd1540096bc7cd236g-01.jpg?height=740&width=1357&top_left_y=790&top_left_x=384)

Figure 1: Given observations, current social reasoning tasks ask models questions targeting specific inferences (left). We propose T4D to probe whether LLMs can decide proper actions using theory-ofmind as a situated agent (right). They key challenges in T4D are 1) models have to identify relevant inferences about mental states without being directed towards one and 2) to arrive at proper action choices, more steps of reasoning are required.


#### Abstract

Thinking is for Doing. Humans can infer other people's mental states from observations-an ability called Theory-of-Mind (ToM)-and subsequently act pragmatically on those inferences. Existing question answering benchmarks such as ToMi ask models questions to make inferences about beliefs of characters in a story, but do not test whether models can then use these inferences to guide their actions. We propose a new evaluation paradigm for large language models (LLMs): Thinking for Doing (T4D), which requires models to connect inferences about others' mental states to actions in social scenarios. Experiments on T4D demonstrate that LLMs such as GPT-4 and PaLM 2 seemingly excel at tracking characters' beliefs in stories, but they struggle to translate this capability into strategic action.

Our analysis reveals the core challenge for LLMs lies in identifying the implicit inferences about mental states without being explicitly asked about as in ToMi, that lead to choosing the correct action in T4D. To bridge this gap, we introduce a zero-shot prompting framework, Foresee and Reflect (FaR), which provides a reasoning structure that encourages LLMs to anticipate future challenges and


[^0]reason about potential actions. FaR boosts GPT-4's performance from $50 \%$ to $71 \%$ on T4D, outperforming other prompting methods such as Chain-of-Thought and Self-Ask. Moreover, FaR generalizes to diverse out-of-distribution story structures and scenarios that also require ToM inferences to choose an action, consistently outperforming other methods including few-shot in-context learning.

## 1 INTRODUCTION

Humans act with specific intentions, often grounded in reasoning about their environment and the mental states of others. For example, if Tom's friend Anne is looking for her backpack in the office, and Tom knows it is in the kitchen, Tom will intervene to help Anne by suggesting she check the kitchen. This proactive action stems from Tom's understanding of three aspects: 1) Anne's goal of finding her backpack; 2) the knowledge of backpack being in the kitchen; and 3) Anne's belief of thinking the backpack is in the office. Reasoning about Anne's mental states allows Tom to conclude that the mismatch between belief and knowledge prevents Anne from reaching her goal, and his intervention can help. Such capabilities to reason about and act on another individual's beliefs, intentions, and emotions are referred to as Theory-of-Mind (ToM), a critical element of human social interactions (Premack \& Woodruff, 1978; Frith \& Frith, 2003)

The rise of large language models (LLMs) has prompted extensive research into their potential for Theory-of-Mind (ToM) capabilities (Sap et al., 2022; Kosinski, 2023; Ullman, 2023; Shapira et al., 2023a). These investigations predominantly rely on established psychological tests, such as the False Belief Test (Wimmer \& Perner, 1983; Baron-Cohen et al., 1985; Perner et al., 1987). While existing benchmarks (Nematzadeh et al., 2018; Le et al., 2019) gauge LLMs' proficiency in inferring mental states from scenarios (see Figure 1 left), they often overlook an essential human capability: acting ${ }^{1}$ on inferred mental states. Simply put: humans often act based on inferred intentions and beliefs. In contrast, despite LLMs' performance in the False Belief Test, they often fail to infer what actions would be most useful in scenarios that humans would find trivial, a crucial consideration for the development of next-generation AI agents, from virtual assistants to embodied robots.

We introduce a new evaluation paradigm: Thinking for Doing (T4D) (see Fiske, 1992) to probe whether models can determine proper actions based on the mental states of others, rather than merely being able to answer questions about mental states. At its core, T4D envisions models as agents processing a series of observations to determine the most apt action from a set of options. Specifically, we adopt stories from a widely-used ToM benchmark: ToMi (Le et al., 2019), based on Sally-Anne False Belief Test (Baron-Cohen et al., 1985) into observations in T4D. This integration ensures that models must utilize mental state reasoning, particularly when a character is identified to hold a false belief (as depicted in Figure 1). The crux of T4D's novelty, as visualized in Figure 1, lies in its objective: instead of merely eliciting inferences from mental state reasoning, it compels models to determine actions based on the former.

T4D presents a new zero-shot challenge for LLMs. We find the highest performance (GPT-4) capped at $50 \%$ while human annotators reach over $95 \%$ agreement. To gain deeper insights into the challenges LLMs encounter in T4D, we identify three reasoning patterns from human-written rationales: question decomposition, theory-of-mind inferences, and commonsense assumptions. Then we test LLMs in oracle settings, providing models with oracle reasoning steps based on the identified patterns. As demonstrated in Section 4.2, the primary challenge LLMs face in T4D is pinpointing the correct evidence to inform their actions. When we provide models with specific hints about relevant inferences, their performance significantly improves, approaching human levels.

The clear potential of LLMs to perform T4D with proper guidance leads to the question: Can we develop a method that improves LLMs' T4D performance without providing oracle hints but instead teaching models to better structure their reasoning process? In response, we introduce a new zero-shot prompting framework Foresee and Reflect (FaR) that guides model's inferences by providing a reasoning structure using future thinking. FaR has two components: Foresee, where it prompts the models to predict future events based on observations and Reflect, where models reason on which action choice better helps the characters with potential challenges. Comparison with prompting strategies including Chain-of-Thought Wei et al. (2022), Tree-of-Thought (Yao[^1]et al., 2023a) (zero-shot), and Self-Ask (Press et al., 2022) shows that FaR improves LLM zero-shot performance by as much as $50 \%$ while other methods do not display significant improvement.

To explore FaR's strengths and limitations in more depth, we perform ablation studies aiming to answer two questions: are both foresight and reflection needed for improving LLMs and what happens if we feed models noisy future predictions? We find that both components are crucial for tackling T4D and that LLMs are sensitive to noisy reasoning steps about the future in FaR, making how to help LLMs recover from noisy foresight an intriguing future direction. To examine whether FaR overfits on the ToMi-converted T4D task, we also conduct generalization study by testing on out-of-distribution story structures and a non-False-Belief ToM task. We find that FaR shows consistent improvement across generalization tests, even outperforming few-shot prompting. Our contributions are as follows:

1. We propose Thinking for Doing, a evaluation paradigm to challenge whether models can connect social reasoning to actions.
2. We find LLMs struggle on T4D and our analysis indicates the key bottleneck is identifying implicit inference steps.
3. We design Foresee and Reflect (FaR), a zero-shot prompting framework that dramatically improves LLMs' performance on T4D. Analysis and generalization studies show that FaR robustness generalize to diverse contexts.

## 2 BACKGROUND AND RELATED WORK

Theory-of-Mind and Language Models Theory-of-mind has been studied extensively in psychology and cognitive science (Premack \& Woodruff, 1978; Baron-Cohen et al., 1985; Frith \& Frith, 2003), and clinical psychology tests such as False Belief Test (Wimmer \& Perner, 1983) (FBT) were developed to test ToM abilities in children. More recently, as neural language models (LM) display impressive performance in many language understanding tasks, more studies aim to answer whether LMs exhibit ToM (Sap et al., 2022; Kosinski, 2023; Ullman, 2023; Shapira et al., 2023a; Sclar et al., 2023; Trott et al., 2023) using False Belief-templated story datasets such as ToM-bAbI (Nematzadeh et al., 2018) and ToMi (Le et al., 2019). Though stories cover limited range of interactions, other sources of ToM tests also face challenges, such as scalability due to costs of human-generated interactions (Bara et al., 2021) and noises in text-game environments (Zhou et al., 2023). This work focuses on False-Belief tests for ToM, the most studied subarea, and revisits the format of such tasks when testing LLMs. Specifically, while probing work shows that LLMs display some degree of ToM but lack robustness (Sap et al., 2022; Shapira et al., 2022), we find that when asked FBT in a more realistic scenario, models fail even on the unperturbed tasks.

Large Language Models and Agents A line of recent work aims to build language agents (Andreas, 2022; Mahowald et al., 2023) that can perform "actions". Actions range from mimicking human social behavior (Park et al., 2023), completing tasks using websites (Gur et al., 2023), and tool using (Yao et al., 2023b; Schick et al., 2023). Our work distinguishes from them by focusing on actions that require proper mental state modeling of other individuals (ToM), attributing the performance gap between answering inference questions only and choosing actions based on inferences, and designed a zero-shot prompt that improves models' capability that robustly generalizes.

Prompting Techniques for LLM Recent advancements in the area of LLMs have given rise to a plethora of few-shot (Brown et al., 2020) and instruction (Mishra et al., 2021) prompting techniques, including Chain-of-Thought prompting (CoT) (Wei et al., 2022), Least-to-most prompting (Zhou et al., 2022), and search-based approaches like Tree-of-Thought (ToT) (Yao et al., 2023a), Graph-ofThought (Besta et al., 2023; Yao et al., 2023c), and RAP (Hao et al., 2023).

However, the primary objective of our work is not to introduce a new prompting technique. Instead, we focus on the benefits of imposing a structured framework on the LLM's reasoning process, particularly in the context of Theory of Mind (ToM) tasks. Specifically, our analysis (Section 4.2) reveals essential elements of reasoning that can help LLM agents act (Foresee (F) and Reflect (R)), and we capture this in our proposed approach FaR. Moreover, any prompting method that supports granular, multi-step reasoning and captures the Foreseeing and Reflecting steps is well-equipped to address the intricacies of ToM tasks.

## 3 THINKING FOR DOING (T4D): TASK AND DATA

Here we formulate the Thinking for Doing (T4D) task that requires models to use social reasoning to choose a proper action as a situated agent.

### 3.1 T4D TASK

In grounded social scenarios, an agent's perspective can be distilled into four primary variables: 1. Observations $\mathcal{O}$ (e.g., Tom entered the kitchen. Tom wants a chocolate. Ella moves the chocolate.), 2. Task $\mathcal{T}$ (e.g., Based on the above observations, who needs help?), 3. Inferences $\mathcal{I}$ (e.g., Tom is unaware of the chocolate's current location.), and 4. Action $\mathcal{A}$ (e.g., Inform Tom about the chocolate's location.). For a comprehensive illustration of these variables in context, please refer to Figure 1.

Traditional social reasoning tasks typically challenge models with questions targeting specific inferences. For example, they might pose a question like "Where will Jackson look for the onion?" accompanied by a set of candidate answers (Nematzadeh et al., 2018; Sap et al., 2019; Le et al., 2019). This is depicted in the left side of Figure 1. Formally, this kind of task can be represented as estimation of $P\left(\mathcal{I} \mid \mathcal{O}, \mathcal{T}_{I}\right)$, where $\mathcal{T}_{I}$ denotes the inference-directed task articulated by the specific question and its associated answer options.

However, in many real-world AI applications, particularly for embodied agents, decisions often revolve around actions rather than explicit inferences. These decisions are influenced by underlying, often implicit, inferences. To bridge this gap, we introduce Thinking for Doing (T4D), a task designed to assess a model's ability to determine the appropriate action based solely on observations, without being directed towards a particular inference. Effectively, T4D represents a shift from directly probing for specific inferences $\left(\mathcal{T}_{I}\right)$ to eliciting actions $\left(\mathcal{T}_{A}\right)$. In the T4D framework, the model's task is not simply to make an inference but to decide on an action based on inferred mental states. This decision-making process involves estimating $P\left(\mathcal{A} \mid \mathcal{O}, \mathcal{T}_{A}\right)$, where $\mathcal{T}_{A}$ encapsulates the action-oriented task, such as determining Who would you prefer to assist the most? with potential actions $\mathcal{A}$ like Assist Jackson or Assist Noah. Crucially, in T4D, inferences $\mathcal{I}$ act as a latent variable, inferred from the observable $\mathcal{O}$ to subsequently influence the chosen action $\mathcal{A}$, i.e. $P\left(\mathcal{A} \mid \mathcal{O}, \mathcal{T}_{A}, \mathcal{I}\right)$.

### 3.2 CONVERTING TOM BENCHMARKS TO T4D

This study focuses on a critical ability in social intelligence-Theory of Mind (ToM) and converts a widelyused existing benchmark: ToMi (Le et al., 2019) from probing inferences to probing agent's action decisions. In the classic Sally-Anne Test setup (used by ToMi), participants interpret a stroy. For instance, consider Owen who mistakenly believes the suit is placed in the cupboard (Figure 2). ToMi asks models to deduce Owen's mental states, with the expected answer being that Owen will search for the suit inside the cupboard (due to mistaken beliefs).

To shift the focus towards actions as an agent who could potentially intervene and help other characters, we introduce an intent: both Owen and Nathan intend to use the suit in the near future. By explicitly stating both characters' intentions, we aim to deter models from adopting a rudimentary heuristic, like automatically assisting the character with immediate plans. However, we also ensure that this complexity does not obfuscate the task for humans. As validated in section 3.3, despite the shared intent to use the suit, human consensus consistently identifies Owen as the one needing help due to his misunderstanding about the suit's location. In our modified task, termed T4D, models are prompted to identify which character they would assist the most by
providing accurate information about the onion's location. Thus, in the T4D adaptation, models must deduce from the narrative that: 1) Owen remains under the false impression that the suit is in the cupboard, and 2) considering his impending need for the suit, accurate knowledge about its location would significantly benefit him. We programmatically convert the stories of ToMi (around 500) to T4D due to ToMi's templatic nature. Details of conversion are in Appendix A.

### 3.3 Human Agreement on T4D

Before using T4D to evaluate our models, we seek to verify its validity by testing it with human ToM (e.g., would human ToM encourage helping a character who holds outdated beliefs?). To do so, we randomly sampled around 80 instances for evaluation by $n=20$ human raters. To ensure this human study reflects how most people would use ToM in real life, we do not pre-train these raters extensively on the ToM tasks and do not provide any answers in the task examples. Our findings underscore the robustness of T4D tasks: every instance garnered agreement from at least 17 of the 20 raters. Moreover, over $90 \%$ of the instances achieved agreement levels exceeding $95 \%$ ( 19 or all 20 raters in consensus). This strong human consensus shows that the design of T4D naturally aligns with human perspectives on decision-making.

## 4 LLMs StrugGLE On T4D WHILE HUMANS Find IT EASY

Here we test LLMs on our T4D task and compare with their performance on the original ToMi set that we convert from. We use PaLM 2 (Anil et al., 2023) Bison (S) and Unicorn (L) ${ }^{2}$, ChatGPT (GPT-3.5) (OpenAI, 2022), and GPT-4 (OpenAI, 2023) accessed between June and August, 2023.

### 4.1 THINKING Is "Easy", T4D Is CHALLENGING FOR LLMs

We focus on zero-shot performance following recent studies (Sap et al., 2022; Shapira et al., 2023a; Sclar et al., 2023) to probe LLM's capabilities to understand and use theory-of-mind. Specifically, we provide answer options and instruct models to output one answer option. The results comparing LLM's performance on ToMi and T4D-ToM are shown in Table 1. We find that both PaLM 2 and GPT models perform close to perfect human scores on ToMi (best model GPT-4 gets $93 \%$ vs human $100 \%$ ) but the performance gap enlarges significantly across all models when tested on T4D-ToM (GPT-4

Table 1: LLMs' accuracy on T4D compared with ToMi. We find gap between human performance on T4D is much larger than that on ToMi (*we count humans correct when there is more than $95 \%$ agreement).

| Models | ToMi | T4D-ToM |
| :--- | :---: | :---: |
| PaLM 2-S (Bison) | 87 | 16 |
| PaLM 2-L (Unicorn) | 87 | 30 |
| GPT-3.5-turbo (ChatGPT) | 74 | 15 |
| GPT-4 | $\mathbf{9 3}$ | $\mathbf{5 0}$ |
| Random Guessing | 50 | 26 |
| Human | $\mathbf{1 0 0}$ | $\mathbf{9 0}^{*}$ |

$50 \%$ vs human $90 \%$ ). This discrepancy underscores the challenges posed by T4D for even the strongest contemporary LLMs.

### 4.2 WHAT MAKES T4D CHALLENGING FOR LLMs?

To better understand why LLMs find T4D challenging, we conducted a study to understand the reasoning processes that humans use to tackle T4D tasks. By collecting and analyzing human-written rationales, we identified distinct dimensions of reasoning that seem particularly challenging for LLMs. Next, we discuss these challenges and experiments with oracle hints to determine if they can indeed aid the models in overcoming these reasoning hurdles. The major reasoning challenges, along with examples and our proposed oracle hints, are summarized in Table 2 and we include example rationales in Appendix B.

Question Decomposition (QD) We find that humans often break down the overarching T4D task into more specific follow-up questions such as "Who might have an information gap?" and "What information I can provide?". This decomposition bridges the gap between the general question and the provided observations. To emulate this in models, we added oracle hints, spotlighting specific[^2]

Table 2: Reasoning-Level breakdown. Following the example task from Figure 2, we show 3 types of reasoning challenges with example specific reasoning steps and design oracle hints to make each challenge easier to analyze what makes LLMs struggle on T4D.

| Reasoning <br> Challenges | Example Reasoning Steps | How to Provide Oracle Hints |
| :---: | :--- | :--- |
| Question <br> Decomposition (QD) | Who would benefit from info? <br> $->$ Nathan and Owen plan to use the suit <br> $->$ Do they know the suit's location? | Add hint after question: <br> "HINT: this information is about <br> an item's location" |
| Theory-of-Mind <br> (ToM) | Nathan and Owen plan to use the suit soon <br> ->They need to know the location <br> Owen left before the suit was moved <br> $->$ Owen thinks the suit is in the cupboard | Provide oracle ToM inference: <br> "Owen will look for the suit in <br> the cupboard" |
| Common Sense | Nathan moved the suit to the basket <br> ->Though not mentioned, we can <br> assume that the basket is lounge <br> as Nathan is not said to exit the room | Make assumptions explicit: <br> "Cupboard and basket are in lounge" <br> "Characters do not leave room <br> unless explicitly stated" |

information, derived from the decomposition process. Essentially, we guide the models with oracle inference results $\left(\mathcal{I}_{Q}\right)$, restructuring the task as i.e, $P\left(A \mid \mathcal{O}, \mathcal{T}_{A}, \mathcal{I}_{Q}\right)$.

Theory-of-Mind Inferences (ToM) The second major reasoning challenge is the core inference tested in the Sally-Anne test - can models correctly infer that Sally will look for the item in the old location because she left the room before Anne moved the item? We make the ToM reasoning challenge easier by providing oracle ToM inferences $\left(\mathcal{I}_{T o M}\right)$ in the observations: "Sally will look for the [ITEM] in the [OLD CONTAINER]". This shifts the task to $P\left(A \mid \mathcal{O}, \mathcal{T}_{A}, \mathcal{I}_{\text {ToM }}\right)$.

Common Sense Assumptions (CSA) The ambiguity inherent in ToMi, as noted by Sclar et al. (2023), presents another challenge. To solve the task, models must assume that both containers are located in the room, even though this is never mentioned explicitly in the observations. We make these assumptions explicit in the observations, i.e, $P\left(A \mid \mathcal{O}, \mathcal{T}_{A}, \mathcal{K}_{C S}\right)$, where we use $\mathcal{K}_{C S}$ to indicate commonsense knowledge not explicitly present in the observation.

Analysis Results As illustrated in Figure 3, providing oracle hints yields varying results across the identified reasoning dimensions. Guiding models with hints related to item location (+QD) and incorporating oracle-derived character beliefs (+ToM) significantly enhances task performance. In contrast, merely clarifying assumptions (+CSA) has a limited effect on boosting model accuracy.

We hypothesize that providing QD or ToM inferences helps models by supplying suggestive evidence, either in the form of leading questions $\left(\mathcal{I}_{Q}\right)$ or relevant ToM inferences $\left(\mathcal{I}_{T o M}\right)$. These results also suggest that the underlying reason for the low performance of LLMs on T4D is attributed not to the task design but to their failure in drawing correct inferences and reasoning. Thus, a key bottleneck in LLMs that makes T4D challenging (but easy for humans) is navigating the unconstrained latent inference space $\mathcal{I}$ to locate the proper inference that makes choosing which action intent clear.

## 5 FORESEE AND REFLECT (FAR) PROMPTING

Building on the insights from our T4D-ToM task analysis, we investigate can we help LLMs identify an implicit inference path that leads to correct action choices without hints. Given observations, humans find it natural to identify relevant inferences and arrive at decisions such as "who should $I$ provide information to?" However, ensuring that LLMs perform similarly structured reasoning is

I will first think about likely future events and identify
potential challenges that each individual might be facing.
Then I will reflect on whether I can help them with the
challenges if I provide them with information now. Finally, I
will choose a final answer based on the reasoning.
Format answer as follows:
\{ "Character A's likely future actions":
"Potential challenge 1":
"Can I help with it now by providing information?":
"Potential challenge 2":
"Can I help with it now by providing information?":
"Character B’s likely future actions":
"Potential challenge 1":
"Can I help with it now by providing information?":
"...
"final reasoning considering all steps above":
"final answer":
Answer:

```
{ {O.OGPT-4
    "Sally's likely future actions": Sally plans to use the marble soon.
Sally will enter the room.
    "Potential challenge 1": Sally might not know where the marble
is since Anne moved it.
    "Can I help with it now by providing information?": Yes, I can
inform Sally that the marble is in the basket.
    "Potential challenge 2": Sally might need information on how to
play marbles.
    "Can I help with it now by providing information?": Yes, I can
provide information on how to use it.
    "Anne's likely future actions": Anne will enter the room.
    "Potential challenge 1": Anne might not know how to use the
marble.
    "Can I help with it now by providing information?": Yes, I can tell
Anne how to play marbles.
```

"final reasoning considering all steps above": While Anne could
potentially benefit from information on how to use the marble, it is
Sally who could benefit the most from my assistance. Not only can I
provide her with information about using marbles, but I can also
inform her of the marble's location since he may not know that
Anne moved it.
"final answer": Thus, the final answer is A. Sally.

Figure 4: Foresee and Reflect (FAR) prompt (left), a new zero-shot prompting framework that combines future prediction and pruning by action-aware reflection. The Foresee part is highlighted in yellow, Reflect is highlighted in blue. Example GPT-4 output shown on the right. The model follows FaR and structures intermediate reasoning steps by copying keys and filling in the values so we only need one inference call.

challenging. Although evidence points to LLMs' ability to infer, they do not necessarily connect these inferences to coherent reasoning about actions.

Our main methodology is to provide LLMs with a generalizable reasoning structure that guides the models to relevant inferences. To this end, we introduce the Foresee and Reflect (FAR) framework. This framework equips LLMs with a structured reasoning paradigm, prompting them to: 1) extrapolate potential future events from given observations, and 2) introspect on actionable steps that would best serve humans in real-time contexts. As argued in Section 2, the primary contribution of FaR is not to introduce a new prompt but to showcase the benefits of imposing a structured framework on the LLM's reasoning process. Figure 4 presents FaR with an example output from GPT-4.

### 5.1 Foresee: ConSIDERING PotENTIAL FUTURE EVENTS

We design FaR by first prompting models to look into the future by considering potential events that are likely to happen. This stems from the understanding that the most valuable help often aligns with shaping a more desireable future outcome more desirable. This is also related to a personality trait referred as "Consideration of Future Consequences (CFC)" in psychology (Strathman et al., 1994), which is the ability to predict future consequences to inform current action decisions. Given the observations $\mathcal{O}$, FaR guides LLMs to iterate over each character in the narrative, predicting their likely future actions and pinpointing the potential challenges they might encounter. This approach effectively broadens the initial observations, extrapolating inferences about potential future events.

### 5.2 Reflect: REASONING ABOUT ACTIONS

After foreseeing likely future events, we prompt models to reflect on whether performing actions at the moment could help with the potential challenges identified in the first step. This process can be considered as pruning the generated potential future inferences based on the available action options. Overall, FaR helps LLMs connect relevant inferences about future with the intended action choices, completing a reasoning chain spanning Observation-Inferences-Action.

Connection to the A* Search Algorithm The FaR methodology is conceptually analogous to the A* search algorithm (Hart et al., 1968), an algorithm for finding the optimal path in a weighted graph. We draw the following connections: Start and Goal: FaR begins with observations and aims to arrive at an optimal action decision. Expanding Nodes: In the Foresee phase of FaR, potential inferences

![](https://cdn.mathpix.com/cropped/2024_06_04_e5bcd1540096bc7cd236g-08.jpg?height=607&width=1244&top_left_y=255&top_left_x=430)

Figure 5: Comparison of zero-shot prompts. We find FaR improves LLMs performance the most.

(akin to nodes in $\mathrm{A}^{*}$ ) are expanded by considering future events. Heuristics: The predictions made during the Foresee step act as heuristics, guiding the reasoning process toward the most relevant inferences. Path Pruning: The Reflect stage in FaR narrows down the inferred events based on available actions, similar to how A* prunes paths based on the heuristic and cost so far.

## 6 FAR BoostS LLM DRAMATICALLY and GENERALIZES ROBUSTLY

We examine the potential of various zero-shot prompting methods on improving LLM's performance on T4D and conduct generalization tests. We aim to answer three research questions through our experiments: 1) How much can FaR improve LLM's zero-shot performance on T4D? 2) Are both the "foresee" and "reflect" components necessary, and what are the limitations of FaR? and 3) Does FaR generalize robustly across scenarios where models need to connect inferences with intents?

### 6.1 BASELINES

We consider the following zero-shot prompting strategies, each offering a unique reasoning structure. Full descriptions of the prompts are available in the Appendix C Chain-of-Thought (CoT) (Wei et al., 2022):the zero-shot variant from Kojima et al. (2022) and add "Answer this question by reasoning step-by-step." Tree-of-Thought (ToT) (Yao et al., 2023a) (Basic Zero-Shot): a zero-shot variant inspired by ToT, which prompts the LLM to envision a discussion among experts. Each expert contributes a reasoning step, and if any expert detects an error in their logic, they exit the discussion. Self-Ask (Press et al., 2022): this method emphasizes self-inquiry. Models generate and answer their follow-up questions, iterating until they reach a conclusive answer. A final reasoning step solidifies the conclusion. FaR: following Section 5 and Figure 4, we design a prompt that guides models to think about likely future events and challenges that characters might encounter, and reflect whether they can provide help. We apply each prompt and make one inference call on all LLMs with maximum 800 tokens with a temperature of 0 (greedy sampling).

### 6.2 FAR DRAMATICALLY IMPRoVES GPT-4 ZERo-SHOT PERFORMANCE

Figure 5 present results of 4 different zero-shot prompting methods. We find that FaR can significantly boost LLMs' performance on T4D-ToM while other prompting methods do not help much. Specifically, FaR helps increase GPT-4 accuracy from base $50 \%$ to $71 \%$ as well as all other LLMs with the improvement between $12 \%$ and $18 \%$. We also observe that more powerful models (GPT-4 and PaLM2-L) tend to benefit more from FaR.

### 6.3 ABLATION AND ANALYSIS

Both Foresight and Reflection Are Important FaR consists of two main components, one to foresee future events and challenges and one to reflect on action decisions. To investigate the individual impact of these components, we modified the FaR prompt, isolating each element for
ablation. Specifically, we omitted the foresight (referenced as yellow text in Figure 4) and reflection parts (blue text in Figure 4). Table 3 presents ablation on FaR for the two components using GPT-4. We find that the performance significantly drops 17 and 12 points, respectively, when there is no foresee and there is no reflect, indicating that they are both crucial for T4D.

## Providing Noisy Foresight Undermines Performance

 We further assessed the robustness of the FaR framework by introducing noisy foresight. For instance, a spurious foresight for the example in Figure 4 might be "Sally will enter the bedroom to sleep." without any evident reason from the observations. Table 3 shows that LLMs are very sensitive to manually-inputted reasoning steps in FaR and the accuracy of GPT-4 drops from $71 \%$ to $42 \%$ (even lower than baseline). This highlights a limitation: while the FaR framework can enhance reasoning when guided correctly, it's sensitive to the quality of the foresight provided and can degrade performance if misled.
### 6.4 FAR GENERALIZES TO DIVERSE SCENARIOS

We probe the generalizability of FAR by evaluating its efficacy on out-of-distribution scenarios.

Story Structure Robustness Tests We use three challenge sets from Sclar et al. (2023) to test if FaR can generalize to story structures beyond those included ToMi. These sets introduce complexities such as the relocation of two items across two rooms (D1), the involvement of multiple characters with an item (D2), and a single item's movement among four containers (D3) ${ }^{3}$. We convert each set (100 stories each) to T4D-style probes using our ToMi conversion methodology. Table 4 shows results on three types of story-structure change of the ToMi stories. Overall, FaR helps LLMs achieve the highest accuracy compared to other zero-shot prompts on all three generalization tests, for almost all models.

T4D-Faux Pas Case Studies To further ascertain FAR's adaptability, we ventured beyond the classic Sally-Anne Test context. We explored Faux Pas scenarios, characterized by individuals inadvertently sharing potentially distressing or unwanted information (Baron-Cohen et al., 1999). We consider Faux Pas, a category of social stories where a person "says something without considering if it is something that others might not want to hear or know" (Baron-Cohen et al., 1999), and use 20 expert-curated stories from Shapira et al. (2023b). We convert the original set to T4D by asking models to choose a character from the stories to provide emotional support (examples Appendix D). We test GPT-4 with multiple zero-shot prompts as well as few-shot prompting with examples from T4D converted from ToMi. Table 5 shows that FaR outperforms other methods dramatically, showing the generalizability of the zero-shot prompt FaR.[^3]

## 7 CONCLUSION

We propose T4D, a task designed to challenge the capacity of LLMs in bridging Theory of Mind reasoning to actions. Our analyses highlighted a key limitation in LLMs: their difficulty in grappling with implicit inferences without explicit guidance. To mitigate this, we introduced FaR, a structured reasoning paradigm, which not only boosts the performance of LLMs but also ensures broader generalization. As a next step, it would be valuable to delve deeper into understanding the internal representation of LLMs when guided by structured prompts like FaR.
Table 5: Faux Pas results using GPT-4.

| Prompts | Accuracy |
| :--- | :---: |
| Base | $31 \%$ |
| CoT | $39 \%$ |
| ToT | $36 \%$ |
| Self-Ask | $43 \%$ |
| Few-Shot | $41 \%$ |
| FaR | $\mathbf{7 6 \%}$ |

## REFERENCES

Jacob Andreas. Language models as agent models. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 5769-5779, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https://aclanthology .org/2022.findings-emnlp. 423.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Cristian-Paul Bara, CH-Wang Sky, and Joyce Chai. Mindcraft: Theory of mind modeling for situated dialogue in collaborative tasks. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 1112-1125, 2021.

Simon Baron-Cohen, Alan M Leslie, and Uta Frith. Does the autistic child have a "theory of mind"? Cognition, 21(1):37-46, 1985.

Simon Baron-Cohen, Michelle O'riordan, Valerie Stone, Rosie Jones, and Kate Plaisted. Recognition of faux pas by normally developing children and children with asperger syndrome or high-functioning autism. Journal of autism and developmental disorders, 29:407-418, 1999.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

Susan T Fiske. Thinking is for doing: portraits of social cognition from daguerreotype to laserphoto. Journal of personality and social psychology, 63(6):877, 1992.

Uta Frith and Christopher D Frith. Development and neurophysiology of mentalizing. Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences, 358(1431):459-473, 2003.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023 .

Peter E Hart, Nils J Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics, 4(2):100-107, 1968.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35: $22199-22213,2022$.

Michal Kosinski. Theory of mind may have spontaneously emerged in large language models. arXiv preprint arXiv:2302.02083, 2023.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. Revisiting the evaluation of theory of mind through question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 5872-5877, 2019.

Kyle Mahowald, Anna A Ivanova, Idan A Blank, Nancy Kanwisher, Joshua B Tenenbaum, and Evelina Fedorenko. Dissociating language and thought in large language models: a cognitive perspective. arXiv preprint arXiv:2301.06627, 2023.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. Reframing Instructional Prompts to GPTk's Language. arXiv preprint arXiv:2109.07830, 2021.

Aida Nematzadeh, Kaylee Burns, Erin Grant, Alison Gopnik, and Tom Griffiths. Evaluating theory of mind in question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2392-2400, 2018.

OpenAI. Chatgpt: Optimizing language models for dialogue, 2022. URL https://openai.com/blog/ chatgpt/.

R OpenAI. Gpt-4 technical report. arXiv, pp. 2303-08774, 2023.

Joon Sung Park, Joseph C O'Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

Josef Perner, Susan R Leekam, and Heinz Wimmer. Three-year-olds' difficulty with false belief: The case for a conceptual deficit. British journal of developmental psychology, 5(2):125-137, 1987.

David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515-526, 1978 .

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4463-4473, Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. URL https://aclanthology.org/D19-1454.

Maarten Sap, Ronan Le Bras, Daniel Fried, and Yejin Choi. Neural theory-of-mind? on the limits of social intelligence in large LMs. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3762-3780, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.emnlp-main. 248 .

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

Melanie Sclar, Sachin Kumar, Peter West, Alane Suhr, Yejin Choi, and Yulia Tsvetkov. Minding language models'(lack of) theory of mind: A plug-and-play multi-character belief tracker. arXiv preprint arXiv:2306.00924, 2023.

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever hans or neural theory of mind? stress testing social reasoning in large language models. arXiv preprint arXiv:2305.14763, 2023a.

Natalie Shapira, Guy Zwirn, and Yoav Goldberg. How well do large language models perform on faux pas tests? In Findings of the Association for Computational Linguistics: ACL 2023, pp. 10438-10451, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.663. URL https://aclanthology.org/2023.findings-acl. 663.

Ori Shapira, Ramakanth Pasunuru, Mohit Bansal, Ido Dagan, and Yael Amsterdamer. Interactive query-assisted summarization via deep reinforcement learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2551-2568, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.184. URL https://aclanthology . org/2022.naacl-main. 184 .

Alan Strathman, Faith Gleicher, David S Boninger, and C Scott Edwards. The consideration of future consequences: Weighing immediate and distant outcomes of behavior. Journal of personality and social psychology, 66(4):742, 1994.

Sean Trott, Cameron Jones, Tyler Chang, James Michaelov, and Benjamin Bergen. Do large language models know what humans know? Cognitive Science, 47(7):e13309, 2023.

Tomer Ullman. Large language models fail on trivial alterations to theory-of-mind tasks. arXiv preprint arXiv:2302.08399, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837, 2022.

Heinz Wimmer and Josef Perner. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. Cognition, 13(1):103-128, 1983.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), $2023 \mathrm{~b}$.

Yao Yao, Zuchao Li, and Hai Zhao. Beyond chain-of-thought, effective graph-of-thought reasoning in large language models. arXiv preprint arXiv:2305.16582, 2023c.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. arXiv preprint arXiv:2205.10625, 2022.

Pei Zhou, Andrew Zhu, Jennifer Hu, Jay Pujara, Xiang Ren, Chris Callison-Burch, Yejin Choi, and Prithviraj Ammanabrolu. I cast detect thoughts: Learning to converse and guide with intents and theory-of-mind in dungeons and dragons. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11136-11155, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long. 624. URL https://aclanthology.org/2023.acl-long. 624.
</end of paper 2>


